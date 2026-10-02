"""Enrichment (results/supplement_schema.py; pilot/enrichment.py): every ``python -m pilot enrich`` command on a synthetic ledger and fake run directories.

The run directories are written by ``tests/fake_launcher.py``'s helpers; the other roles' harnesses
(``envs.evaluation.evaluate_battery`` / ``evaluate_continuation``, ``studyb.evaluation``) are replaced
by the deterministic stand-ins of ``Harness``, whose results have the shapes the real harnesses
return (per-episode arrays, seeds, hazard definitions); Role 3's pure functions (``metrics.controller``,
``metrics.recovery``, ``metrics.interventions.intervention_summary``) and Role 4's
``analysis.matching.match_arms`` run for real; its checkpoint selection is mostly replaced by
``closest_to_25``. The helpers of this module are shared with
tests/test_ledger_writer_and_provenance.py, tests/test_integration_enrich.py and
tests/test_studyb_evaluation.py (``_moderate_pilot``). It also covers the read-only reports
(completeness, freeze), ``go`` and ``g4-measure``, and the unconstrained pilot run's sidecar.
"""

from __future__ import annotations

import dataclasses
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import fake_launcher  # noqa: E402

from configs import registered as R  # noqa: E402
from pilot import enrichment, errors, manifest, provenance, supplement  # noqa: E402
from pilot.contracts import ContractError, fewshot_key  # noqa: E402
from pilot.errors import PendingQuestionError  # noqa: E402
from pilot.ledger_writer import LedgerPaths, write_run  # noqa: E402
from pilot.manifest import RunSpec  # noqa: E402

COMMIT = fake_launcher.COMMIT
SEL = list(fake_launcher.SELECTION_SEEDS)
MEAS = list(fake_launcher.MEASUREMENT_SEEDS)
LAYOUTS = list(range(3_000_000, 3_000_020))
HAZARD_SEEDS = [3_100_000 + 5 * j + k for j in range(20) for k in range(5)]  # layout-major, as envs.evaluation
WINDOW_COSTS = (30, 29, 27, 26, 25.4, 24.9, 23, 22, 21, 20)  # rule 1 picks 24.9, the sixth of the last ten
TOTAL = 2_000_000  # 100 epochs: small fixtures (the harness is stubbed)
STEPS = list(range(200_000, TOTAL + 1, 200_000))
PILOT_LEDGER_RUN = "P-A-PointGoal1-N0.00-s0"


# ---------------------------------------------------------------------------
# Synthetic runs
# ---------------------------------------------------------------------------


episodes = fake_launcher.episodes  # n integer episode costs whose mean is ``mean`` (a multiple of 0.01)
fmean = fake_launcher.fmean


def study_a_spec(N: float = 0.0, seed: int = 0, *, pilot: bool = False, task: str = "SafetyPointGoal1-v0",
                 treatment: str | None = None) -> RunSpec:
    short = manifest.short_task(task)
    if N == 0:
        arm, onset, shape, control = "N0.00", 0, None, None
    else:
        arm, onset, shape, control = f"N{N:.2f}-abrupt-total", int(N * TOTAL), "abrupt", "total_steps"
    if treatment:
        arm = f"{arm}-{treatment}"
    prefix = "P-" if pilot else ""
    group = "pilot" if pilot else ("treatment" if treatment else "main")
    return RunSpec(run_id=f"{prefix}A-{short}-{arm}-s{seed}", study="A", task=task, arm=arm, seed=seed,
                   total_steps=TOTAL, base_algo="PPOLag", plugin="study_a", group=group, pilot=pilot, N=N,
                   onset_step=onset, onset_shape=shape, step_matching=control, treatment=treatment,
                   params={"cost_limit": R.COST_LIMIT})


def study_b_spec(arm: str = "Moderate", seed: int = 0) -> RunSpec:
    levels = R.STUDY_B_ARMS[arm]
    params = {"budget_observation_divisor": R.BUDGET_OBSERVATION_DIVISOR}
    if levels is None:
        params.update(continuous_range=list(R.CONTINUOUS_RANGE), continuous_bin_width=R.CONTINUOUS_BIN_WIDTH)
    return RunSpec(run_id=f"B-{arm}-s{seed}", study="B", task="SafetyPointGoal1-v0", arm=arm, seed=seed,
                   total_steps=TOTAL, base_algo="PPOLag", plugin="study_b", group="study_b", training_levels=levels,
                   params=params)


def studyb_budgets(spec: RunSpec) -> list[float]:
    """The per-episode budgets of Q-studyb-eval's answer (envs.evaluation.training_budget_schedule)."""
    return fake_launcher.studyb_budgets(spec.to_dict())


def evaluation_json(spec: RunSpec, *, costs=WINDOW_COSTS, final_cost: float = 24.0, final_return: float = 20.0,
                    per_episode: bool = True, commit: str = COMMIT) -> dict:
    """evaluation.json as pilot.launch.evaluate writes it from contract 1 (envs or studyb harness)."""
    window = list(range(200_000, spec.total_steps + 1, 200_000))[-10:]
    ev = {"final_cost": fmean(episodes(final_cost)), "final_return": fmean(episodes(final_return)),
          "episodes": R.EVAL_EPISODES, "eval_wall_clock_hours": 0.1,
          "selection": {str(s): [fmean(episodes(c)), fmean(episodes(1.0))] for s, c in zip(window, costs)},
          "selection_seeds": SEL}
    if per_episode:
        ev.update({
            "final_step": spec.total_steps, "final_seed_set": "measurement", "final_selection_cost": fmean(episodes(costs[-1])),
            "measurement_seeds": MEAS, "short_episodes": 0, "eval_commit_hash": commit,
            "episode_costs": {"final": episodes(final_cost), "selection": {str(s): episodes(c) for s, c in zip(window, costs)}},
            "episode_returns": {"final": episodes(final_return), "selection": {str(s): episodes(1.0) for s in window}},
        })
        if spec.plugin == "study_b":
            ev["studyb_budgets"] = studyb_budgets(spec)
    return ev


def _rewrite_progress(omni: Path, transform: Callable[[dict], dict]) -> None:
    import csv

    with open(omni / "progress.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    rows = [transform(dict(r)) for r in rows]
    with open(omni / "progress.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_run(root: Path, spec: RunSpec, *, costs=WINDOW_COSTS, final_cost: float = 24.0, status: str = "completed",
             cause: str | None = None, per_episode: bool = True, batch_cost=None) -> Path:
    """A finished run directory under ``root`` (the scheduler's layout <data_root>/checkpoints/<run_id>)."""
    run_dir = root / spec.run_id
    run_dir.mkdir(parents=True)
    (run_dir / "spec.json").write_text(spec.to_json())
    omni = fake_launcher.write_outputs(spec.to_dict(), run_dir)
    deps = fake_launcher.resolve_dependencies(spec.to_dict(), run_dir) if spec.depends_on else {}
    assert deps is not None, f"a dependency of {spec.run_id} is not a completed run"
    (run_dir / "train_result.json").write_text(json.dumps(fake_launcher.train_result(spec.to_dict(), omni, status, cause, deps)))
    if batch_cost is not None:  # Metrics/BatchEpCost of epoch e (pilot.algorithms.FullStateCheckpointMixin)
        _rewrite_progress(run_dir / omni, lambda r: {**r, "Metrics/BatchEpCost": str(batch_cost(int(float(r["Train/Epoch"]))))})
    if status == "completed" and spec.group not in manifest.CONTINUATION_GROUPS:
        (run_dir / "evaluation.json").write_text(json.dumps(evaluation_json(spec, costs=costs, final_cost=final_cost,
                                                                            per_episode=per_episode)))
    return run_dir


def omni_of(run_dir: Path) -> Path:
    return run_dir / json.loads((run_dir / "train_result.json").read_text())["omnisafe_dir"]


def paths_in(tmp: Path) -> LedgerPaths:
    return LedgerPaths(main=tmp / "ledger" / "ledger.parquet", pilot=tmp / "pilot" / "ledger.parquet",
                       sidecar_dir=tmp / "pilot" / "sidecar")


def rows_of(ledger: Path) -> dict:
    from results.ledger_schema import load_ledger_as_rows

    return {r.run_id: r for r in load_ledger_as_rows(ledger)}


def closest_to_25(checkpoints: list[dict]) -> int:
    window = [c for c in checkpoints if c.get("selection_cost") is not None]
    return min(window, key=lambda c: (abs(c["selection_cost"] - 25.0), -c["step"]))["step"]


# ---------------------------------------------------------------------------
# Stand-ins for the other roles' harnesses (the real result shapes)
# ---------------------------------------------------------------------------


# The recorded definitions of envs.evaluation.hazard_definition / dynamics_definition (HANDOVER.md section 8,
# contract 5), abridged: the harness returns them in result["conditions"].
DEFINITIONS = {
    "hazard": {"proposal_key": "Q-hazard", "form": "central", "placements": [[-0.75, -0.75, 0.75, 0.75]], "hazards": 8,
               "size": 0.2, "centre_half_width": 0.57, "reach_half_width": 0.77, "layout_seeds": LAYOUTS,
               "episodes_per_layout": 5},
    "dynamics": {"proposal_key": "Q-dynamics", "body_mass_scale": R.BODY_MASS_SCALE, "inertia_scaled": True,
                 "bodies": ["agent"], "ground_friction_scale": R.GROUND_FRICTION_SCALE,
                 "friction_geoms": ["floor", "agent"], "episode_seeds": "measurement"},
}


class Harness:
    """Deterministic harness results, recording each call, with the real harnesses' result gates
    (``evaluate_battery``: Q-hazard, Q-dynamics per condition; Study B: Q-studyb-eval).

    ``c_id[run_id]`` is C_ID at the run's checkpoints before the last (default 25.1) and ``final[run_id]``
    at its final checkpoint (default 24.0, ``evaluation_json``'s final_cost: the same measurement set);
    a condition's cost is C_ID plus ``gaps[condition]``; a continuation's cost is ``continuation_cost``;
    ``rates[budget][horizon]`` are the few-shot satisfaction rates.
    """

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.c_id: dict[str, float] = {}
        self.final: dict[str, float] = {}
        self.gaps = {"hazard": 6.0, "dynamics": 2.5}
        self.continuation_cost = 30.0
        self.rates = {b: {200_000: 0.5, 500_000: 0.85, 1_000_000: 0.9} for b in R.UNSEEN_BUDGETS}

    def battery(self, omnisafe_dir: str, spec: dict, step: int, conditions: Sequence[str]) -> dict:
        for name in conditions:
            errors.require_answered({"hazard": "Q-hazard", "dynamics": "Q-dynamics"}[name], what=f"the {name} cost")
        self.calls.append(("battery", spec["run_id"], step, tuple(conditions)))
        if step == spec["total_steps"]:
            c_id = self.final.get(spec["run_id"], 24.0)
        else:
            c_id = self.c_id.get(spec["run_id"], 25.1)
        result = {"step": step, "episodes": R.EVAL_EPISODES, "measurement_seeds": MEAS,
                  "measurement": fmean(episodes(c_id)), "measurement_return": fmean(episodes(3.0)),
                  "seeds": {"measurement": MEAS}, "episode_costs": {"measurement": episodes(c_id)},
                  "episode_returns": {"measurement": episodes(3.0)}, "conditions": {}, "short_episodes": 0}
        for name in conditions:
            cost = c_id + self.gaps[name]
            result[name], result[f"{name}_return"] = fmean(episodes(cost)), fmean(episodes(2.0))
            result["seeds"][name] = HAZARD_SEEDS if name == "hazard" else MEAS
            result["episode_costs"][name], result["episode_returns"][name] = episodes(cost), episodes(2.0)
            result["conditions"][name] = dict(DEFINITIONS[name], task=spec.get("task"))
        return result

    def continuation(self, omnisafe_dir: str, spec: dict) -> dict:
        self.calls.append(("continuation", spec["run_id"]))
        cost = self.continuation_cost
        return {"step": spec["total_steps"], "task": spec["task"], "cost": fmean(episodes(cost)),
                "return": fmean(episodes(4.0)), "episodes": R.EVAL_EPISODES, "measurement_seeds": MEAS,
                "episode_costs": episodes(cost), "episode_returns": episodes(4.0), "short_episodes": 0}

    @staticmethod
    def _at(budget: float, rate: float) -> dict:
        k = round(rate * R.EVAL_EPISODES)
        costs = [float(int(budget))] * k + [budget + 5.0] * (R.EVAL_EPISODES - k)
        return {"mean_cost": fmean(costs), "mean_return": fmean(episodes(1.0)),
                "satisfaction": sum(1 for c in costs if c <= budget) / len(costs),
                "violation": fmean(max(c - budget, 0.0) for c in costs),
                "episode_costs": costs, "episode_returns": episodes(1.0)}

    def zero_shot(self, omnisafe_dir: str, spec: dict) -> dict:
        errors.require_answered("Q-studyb-eval", what="the zero-shot evaluation")
        self.calls.append(("zero_shot", spec["run_id"]))
        arm = spec["arm"]
        unseen = [float(b) for b in R.UNSEEN_BUDGETS]
        levels = R.STUDY_B_ARMS[arm]
        reference = list(R.CONTINUOUS_RANGE) if levels is None else sorted(float(b) for b in levels)
        blocks = {b: self._at(b, 0.6 if b in unseen else 0.95) for b in unseen + reference}

        def distance(b):
            if levels is None:
                lo, hi = R.CONTINUOUS_RANGE
                return max(lo - b, 0.0, b - hi)
            return min(abs(b - level) for level in levels)

        maps = {name: {b: block[name] for b, block in blocks.items()} for name in
                ("mean_cost", "satisfaction", "violation", "mean_return", "episode_costs", "episode_returns")}
        return {"run_id": spec["run_id"], "arm": arm, "episodes": R.EVAL_EPISODES, "step": spec["total_steps"],
                "seed_set": "measurement", "seeds": MEAS, "unseen_budgets": unseen, "reference_budgets": reference,
                **maps, "distance": {b: distance(b) for b in unseen},
                "sr_zero": {b: blocks[b]["satisfaction"] for b in unseen}}

    def fewshot(self, omnisafe_dir: str, spec: dict) -> dict:
        errors.require_answered("Q-studyb-eval", what="the few-shot evaluation")
        self.calls.append(("fewshot", spec["run_id"]))
        params = spec["params"]
        budget, horizons = float(params["budget"]), list(params["horizons"])
        blocks = {h: {"step": h, "checkpoint": f"torch_save/epoch-{h // R.STEPS_PER_EPOCH}.pt",
                      **self._at(budget, self.rates[budget][h])} for h in horizons}
        rates = {h: blocks[h]["satisfaction"] for h in horizons}
        reached = [h for h in horizons if rates[h] >= R.SATISFACTION_TARGET]
        steps = reached[0] if reached else max(horizons) + 1
        return {"run_id": spec["run_id"], "parent_run_id": params["parent_run_id"], "parent_step": params["parent_step"],
                "arm": spec["arm"], "budget": budget, "horizons": horizons, "episodes": R.EVAL_EPISODES,
                "seed_set": "measurement", "seeds": MEAS, "by_horizon": blocks,
                "sr_fewshot": {fewshot_key(budget, h): rates[h] for h in horizons},
                "adapt_steps": {budget: steps}, "adapt_censored": steps > max(horizons)}


@pytest.fixture
def harness() -> Harness:
    return Harness()


@pytest.fixture
def answered(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every key answered as decided (docs/DECISIONS.md, 2026-10-02), whatever the repository's ANSWERED_QUESTIONS
    holds."""
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset(R.PENDING))


def ledger_with(tmp_path: Path, specs: Sequence[RunSpec], **kw: Any) -> tuple[LedgerPaths, Path]:
    """Runs written under ``<tmp>/data/checkpoints`` and ledgered; returns the paths and the runs root."""
    paths = paths_in(tmp_path)
    root = tmp_path / "data" / "checkpoints"
    for spec in specs:
        write_run(spec, make_run(root, spec, **kw), paths)
    return paths, root


def selected(tmp_path: Path, specs: Sequence[RunSpec], **kw: Any) -> tuple[Path, Path]:
    """A ledger whose Study A rows went through ``enrich select``; returns (ledger, runs root)."""
    paths, root = ledger_with(tmp_path, specs, **kw)
    ledger = paths.pilot if specs[0].pilot else paths.main
    enrichment.apply_selection(ledger, selector=closest_to_25)
    return ledger, root


def run_dir_of(root: Path) -> Callable[[str], Path]:
    return lambda run_id: root / run_id


def smoke_root(tmp_path: Path, paths: LedgerPaths | None = None) -> Path:
    """``<tmp>/data`` made a smoke data root whose ledgers are ``paths`` (default ``paths_in(tmp)``), as
    scripts/smoke_run.py makes one (a scheduler with the smoke flags fixes its mode and ledgers): the
    command line takes --allow-dirty and --allow-pending for these ledgers only (pilot/__main__.py; HANDOVER.md
    section 7)."""
    from pilot.scheduler import Scheduler, SchedulerConfig

    data = tmp_path / "data"
    sched = Scheduler(SchedulerConfig(data_root=data, allow_dirty=True, allow_pending=True,
                                      ledger_paths=paths or paths_in(tmp_path)))
    try:
        sched.fix_mode()
    finally:
        sched.db.close()
    return data


def log_entries(ledger: Path) -> list[dict]:
    path = enrichment.enrichment_log_path(ledger)
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


# ---------------------------------------------------------------------------
# select (Part 4.1 rule 1; Q-tie-break)
# ---------------------------------------------------------------------------


def test_selection_then_battery_on_the_pilot_ledger(tmp_path, harness, answered) -> None:
    specs = [study_a_spec(0.0, s, pilot=True) for s in (0, 1)]
    ledger, root = selected(tmp_path, specs)
    assert enrichment.apply_selection(ledger, selector=closest_to_25) == []  # never selected twice
    done_first = enrichment.apply_battery(ledger, run_dir_of(root), ["hazard"], evaluator=harness.battery)
    assert len(done_first) == 2 and harness.calls[0][2] == 1_200_000 and harness.calls[0][3] == ("hazard",)
    # a condition left out earlier is added later, and the measurement cost must reproduce
    done = enrichment.apply_battery(ledger, run_dir_of(root), ["hazard", "dynamics"], evaluator=harness.battery)
    assert len(done) == 2 and harness.calls[-1][3] == ("dynamics",)
    row = rows_of(ledger)[specs[0].run_id]
    assert row.matched_checkpoint_step == row.training_age == 1_200_000
    assert row.selection_cost_at_match == pytest.approx(24.9)
    assert row.lambda_at_selection == pytest.approx(0.001 + 0.035 * 60)  # multiplier after epoch 59
    assert row.measurement_cost == pytest.approx(25.1)
    assert row.gap_hazard == pytest.approx(6.0) and row.gap_dynamics == pytest.approx(2.5)
    assert row.matched is None  # the pilot's matching is G1's (Part 6), not rule 5's flags
    assert enrichment.apply_battery(ledger, run_dir_of(root), ["hazard", "dynamics"], evaluator=harness.battery) == []
    # the supplement: one battery record per condition, and the measurement record, beside the pilot ledger
    for kind, part in (("battery", "hazard"), ("battery", "dynamics"), ("measurement", None), ("evaluation", None)):
        record = supplement.read(kind, specs[0].run_id, ledger_path=ledger, part=part)
        assert record is not None and record.run_id == specs[0].run_id
    hazard = supplement.read("battery", specs[0].run_id, ledger_path=ledger, part="hazard")
    assert hazard.gap == row.gap_hazard and hazard.measurement_cost == row.measurement_cost
    assert hazard.episodes.seed_set == "hazard" and hazard.hazard.layout_seeds == LAYOUTS
    assert supplement.supplement_dir(ledger) == tmp_path / "pilot" / "supplement"
    log = log_entries(ledger)
    assert [e["action"] for e in log] == ["select", "battery", "battery"]
    # the log records what each call actually evaluated, not only what it was asked for
    assert log[2]["conditions"] == ["hazard", "dynamics"]
    assert log[2]["evaluated"] == {rid: ["dynamics"] for rid in done_first}


def test_selector_that_breaks_rule_one_is_refused(tmp_path) -> None:
    paths, _ = ledger_with(tmp_path, [study_a_spec()])
    with pytest.raises(ContractError, match="rule 1 gives 1200000"):
        enrichment.apply_selection(paths.main, selector=lambda cps: 2_000_000)  # in the window, but not closest to 25


def test_selection_outside_the_window_is_refused(tmp_path) -> None:
    paths, _ = ledger_with(tmp_path, [study_a_spec()])
    with pytest.raises(ContractError, match="last ten"):
        enrichment.apply_selection(paths.main, selector=lambda cps: 0)


def test_a_tie_waits_on_q_tie_break_and_the_others_are_written(tmp_path, monkeypatch) -> None:
    tie = (40, 40, 40, 40, 40, 40, 40, 40, 26, 24)  # 26 and 24 are both 1 from d = 25
    paths = paths_in(tmp_path)
    root = tmp_path / "data" / "checkpoints"
    tied, clear = study_a_spec(0.0, 0), study_a_spec(0.0, 1)
    write_run(tied, make_run(root, tied, costs=tie), paths)
    write_run(clear, make_run(root, clear), paths)
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset())  # the keys open, whatever the repository answers
    assert R.is_open("Q-tie-break")
    with pytest.raises(PendingQuestionError, match="Q-tie-break") as exc:
        enrichment.apply_selection(paths.main, selector=closest_to_25)
    assert tied.run_id in str(exc.value)
    rows = rows_of(paths.main)
    assert rows[tied.run_id].matched_checkpoint_step is None and rows[clear.run_id].matched_checkpoint_step == 1_200_000
    entry = log_entries(paths.main)[-1]
    assert entry["run_ids"] == [clear.run_id] and list(entry["waiting"]) == [tied.run_id]
    assert enrichment.rule_one_ties(rows[tied.run_id].model_dump()["checkpoints"]) == [1_800_000, 2_000_000]
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset({"Q-tie-break"}))
    assert enrichment.apply_selection(paths.main, selector=closest_to_25) == [tied.run_id]
    assert rows_of(paths.main)[tied.run_id].matched_checkpoint_step == 2_000_000  # the later checkpoint


def test_rule_one_ties_go_to_the_later_checkpoint_despite_float_noise() -> None:
    costs = [40.0] * 8 + [32.01, 17.99]  # |32.01 - 25| and |17.99 - 25| differ in binary floating point
    assert abs(32.01 - 25) != abs(17.99 - 25)
    checkpoints = [{"step": s, "selection_cost": c} for s, c in zip(STEPS, costs)]
    assert enrichment.rule_one_step(checkpoints) == 2_000_000
    assert enrichment.rule_one_ties(checkpoints) == [1_800_000, 2_000_000]
    np = pytest.importorskip("numpy")  # a harness that averages in float32
    # the later checkpoint has the larger raw float32 distance: only the tie rounding makes it the later one
    f32 = [{"step": s, "selection_cost": float(np.float32(c))} for s, c in zip(STEPS, [40.0] * 8 + [32.01, 17.99])]
    assert abs(f32[-1]["selection_cost"] - 25) > abs(f32[-2]["selection_cost"] - 25)
    assert enrichment.rule_one_step(f32) == 2_000_000
    assert enrichment.rule_one_ties(f32) == [1_800_000, 2_000_000]
    checkpoints[-1]["selection_cost"] = None
    with pytest.raises(ContractError, match="no selection-set cost"):
        enrichment.rule_one_step(checkpoints)


def test_rule_one_refuses_a_non_finite_cost() -> None:
    costs = [float("nan")] + [40.0] * 8 + [25.0]  # NaN first once made rule 1 pick step 200,000
    with pytest.raises(ContractError, match="not finite"):
        enrichment.rule_one_step([{"step": s, "selection_cost": c} for s, c in zip(STEPS, costs)])


def test_runs_enriched_before_a_failure_are_logged(tmp_path) -> None:
    paths, _ = ledger_with(tmp_path, [study_a_spec(0.0, s) for s in (0, 1)])
    calls = []

    def flaky(checkpoints):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("selector failed on the second run")
        return closest_to_25(checkpoints)

    with pytest.raises(RuntimeError):
        enrichment.apply_selection(paths.main, selector=flaky)
    (entry,) = log_entries(paths.main)
    assert entry["action"] == "select" and entry["run_ids"] == ["A-PointGoal1-N0.00-s0"]


def test_enrichment_checks_the_code_before_writing(tmp_path) -> None:
    paths, _ = ledger_with(tmp_path, [study_a_spec()])

    def refuse() -> None:
        raise RuntimeError("uncommitted code")

    with pytest.raises(RuntimeError, match="uncommitted"):
        enrichment.apply_selection(paths.main, selector=closest_to_25, check_code=refuse)
    assert rows_of(paths.main)["A-PointGoal1-N0.00-s0"].matched_checkpoint_step is None  # nothing written


def test_a_selector_cannot_alter_the_checkpoints_that_are_checked(tmp_path) -> None:
    paths, _ = ledger_with(tmp_path, [study_a_spec()])

    def vandal(checkpoints):
        step = closest_to_25(checkpoints)
        for c in checkpoints:
            c["selection_cost"] = 25.0  # would make every checkpoint tie and rule 1 pick the last one
        return step

    assert enrichment.apply_selection(paths.main, selector=vandal) == ["A-PointGoal1-N0.00-s0"]
    row = rows_of(paths.main)["A-PointGoal1-N0.00-s0"]
    assert row.matched_checkpoint_step == 1_200_000 and row.selection_cost_at_match == pytest.approx(24.9)


def test_an_unwritable_log_does_not_hide_the_enrichment_error(tmp_path, capsys) -> None:
    paths, _ = ledger_with(tmp_path, [study_a_spec(0.0, s) for s in (0, 1)])
    enrichment.enrichment_log_path(paths.main).mkdir()  # open(..., "a") raises IsADirectoryError
    calls = []

    def flaky(checkpoints):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("selector failed on the second run")
        return closest_to_25(checkpoints)

    with pytest.raises(RuntimeError, match="second run"):
        enrichment.apply_selection(paths.main, selector=flaky)
    assert "enrichment log" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# The writing rules (pilot/enrichment.py): scalars once, maps by absent keys, all-or-nothing
# ---------------------------------------------------------------------------


def test_merge_fields_writes_scalars_once_and_maps_by_absent_keys() -> None:
    current = {"measurement_cost": None, "gap_hazard": 3.0, "sr_zero": {5.0: 0.5}, "adapt_steps": None}
    update = enrichment.merge_fields(current, {"measurement_cost": 25.1, "gap_hazard": 3.0,
                                               "sr_zero": {5.0: 0.5, 15.0: 0.7}, "adapt_steps": {5.0: 200_000}}, "r-s0")
    assert update == {"measurement_cost": 25.1, "sr_zero": {5.0: 0.5, 15.0: 0.7}, "adapt_steps": {5.0: 200_000}}
    assert enrichment.merge_fields(current, {"gap_hazard": 3.0, "sr_zero": {5.0: 0.5}, "lambda_peak": None}, "r-s0") == {}
    with pytest.raises(ContractError, match=r"gap_hazard is 3.0 in the ledger"):
        enrichment.merge_fields(current, {"gap_hazard": 3.5}, "r-s0")
    with pytest.raises(ContractError, match=r"sr_zero\[5.0\] is 0.5"):
        enrichment.merge_fields(current, {"sr_zero": {5.0: 0.51}}, "r-s0")
    with pytest.raises(ContractError):
        enrichment.merge_fields({"matched": False}, {"matched": True}, "r-s0")


def test_an_enrichment_that_breaks_the_schema_is_not_written(tmp_path) -> None:
    """write_enrichment validates the merged row (corrected schema), and the transaction every row."""
    from pydantic import ValidationError

    paths, _ = ledger_with(tmp_path, [study_a_spec()])
    before = paths.main.read_bytes()
    with pytest.raises(ValidationError):
        enrichment._write(paths.main, "A-PointGoal1-N0.00-s0", {"lambda_peak": -1.0})
    assert paths.main.read_bytes() == before
    assert rows_of(paths.main)["A-PointGoal1-N0.00-s0"].lambda_peak is None


def test_a_refused_result_writes_no_record_and_no_field(tmp_path, harness, answered) -> None:
    ledger, root = selected(tmp_path, [study_a_spec(0.0, 0, pilot=True)])

    def refuse() -> None:
        raise RuntimeError("uncommitted code")

    with pytest.raises(RuntimeError):
        enrichment.apply_measurement(ledger, run_dir_of(root), evaluator=harness.battery, check_code=refuse)
    assert not supplement.exists("measurement", PILOT_LEDGER_RUN, ledger_path=ledger)
    assert rows_of(ledger)[PILOT_LEDGER_RUN].measurement_cost is None
    # a result that breaks the record's schema (a mean that is not the mean of its episodes) is refused whole
    bad = lambda *a: {**harness.battery(*a), "measurement": 30.0}  # noqa: E731
    with pytest.raises(ContractError, match="not a valid measurement supplement record"):
        enrichment.apply_measurement(ledger, run_dir_of(root), evaluator=bad)
    assert not supplement.exists("measurement", PILOT_LEDGER_RUN, ledger_path=ledger)
    assert rows_of(ledger)[PILOT_LEDGER_RUN].measurement_cost is None


def test_a_differing_record_on_disk_refuses_the_whole_write(tmp_path, harness, answered) -> None:
    """Records are written once: a different record already on disk stops the enrichment before the ledger changes."""
    ledger, root = selected(tmp_path, [study_a_spec(0.0, 0, pilot=True)])
    harness.c_id[PILOT_LEDGER_RUN] = 26.0
    other = harness.battery("dir", {"run_id": PILOT_LEDGER_RUN, "total_steps": TOTAL}, 1_200_000, ["hazard"])
    result = enrichment.check_battery_result(PILOT_LEDGER_RUN, other, step=1_200_000, conditions=["hazard"],
                                             selection_seeds=SEL)
    record = result.condition_record("battery", PILOT_LEDGER_RUN, COMMIT, "hazard", result.c_id)
    supplement.write("battery", PILOT_LEDGER_RUN, record.record, ledger_path=ledger, part="hazard")
    harness.c_id[PILOT_LEDGER_RUN] = 25.1
    before = ledger.read_bytes()
    with pytest.raises(supplement.SupplementConflict, match="written once"):
        enrichment.apply_battery(ledger, run_dir_of(root), ["hazard"], evaluator=harness.battery)
    assert ledger.read_bytes() == before and not supplement.exists("measurement", PILOT_LEDGER_RUN, ledger_path=ledger)


def test_a_conflict_on_a_later_record_leaves_no_earlier_record_behind(tmp_path, harness, answered) -> None:
    """Every record of one write is checked before the first is written: no orphan record without its ledger
    field, which analysis.data would otherwise read (its episodes) although the ledger's gap is None."""
    ledger, root = selected(tmp_path, [study_a_spec(0.0, 0, pilot=True)])
    harness.c_id[PILOT_LEDGER_RUN] = 26.0
    other = harness.battery("dir", {"run_id": PILOT_LEDGER_RUN, "total_steps": TOTAL}, 1_200_000, ["dynamics"])
    result = enrichment.check_battery_result(PILOT_LEDGER_RUN, other, step=1_200_000, conditions=["dynamics"],
                                             selection_seeds=SEL)
    record = result.condition_record("battery", PILOT_LEDGER_RUN, COMMIT, "dynamics", result.c_id)
    supplement.write("battery", PILOT_LEDGER_RUN, record.record, ledger_path=ledger, part="dynamics")
    harness.c_id[PILOT_LEDGER_RUN] = 25.1
    before = ledger.read_bytes()
    with pytest.raises(supplement.SupplementConflict):
        enrichment.apply_battery(ledger, run_dir_of(root), ["hazard", "dynamics"], evaluator=harness.battery)
    assert ledger.read_bytes() == before and rows_of(ledger)[PILOT_LEDGER_RUN].gap_hazard is None
    for kind, part in (("battery", "hazard"), ("measurement", None)):  # written before the conflict, once
        assert not supplement.exists(kind, PILOT_LEDGER_RUN, ledger_path=ledger, part=part)
    from analysis.data import load_dataset

    (rec,) = load_dataset(ledger, mode="pilot", cuts=()).study_a
    assert "hazard" not in rec["battery_episodes"]


def test_supplement_check_answers_what_write_would_do_without_writing(tmp_path, harness, answered) -> None:
    ledger, _ = selected(tmp_path, [study_a_spec(0.0, 0, pilot=True)])
    raw = harness.battery("dir", {"run_id": PILOT_LEDGER_RUN, "total_steps": TOTAL}, 1_200_000, ["hazard"])
    result = enrichment.check_battery_result(PILOT_LEDGER_RUN, raw, step=1_200_000, conditions=["hazard"], selection_seeds=SEL)
    record = result.condition_record("battery", PILOT_LEDGER_RUN, COMMIT, "hazard", result.c_id).record
    later = {**record, "code_commit": "c" * 40}  # the same numbers from later code
    args = dict(ledger_path=ledger, part="hazard")
    assert supplement.check("battery", PILOT_LEDGER_RUN, record, **args) is False  # new
    assert not supplement.exists("battery", PILOT_LEDGER_RUN, **args)  # nothing written
    supplement.write("battery", PILOT_LEDGER_RUN, record, **args)
    assert supplement.check("battery", PILOT_LEDGER_RUN, record, **args) is True  # identical
    with pytest.raises(supplement.SupplementConflict):
        supplement.check("battery", PILOT_LEDGER_RUN, later, **args)
    assert supplement.check("battery", PILOT_LEDGER_RUN, later, **args, accept_reproduction=True) is True
    harness.c_id[PILOT_LEDGER_RUN] = 26.0  # other numbers: never a reproduction
    raw = harness.battery("dir", {"run_id": PILOT_LEDGER_RUN, "total_steps": TOTAL}, 1_200_000, ["hazard"])
    other = enrichment.check_battery_result(PILOT_LEDGER_RUN, raw, step=1_200_000, conditions=["hazard"], selection_seeds=SEL)
    different = other.condition_record("battery", PILOT_LEDGER_RUN, "c" * 40, "hazard", other.c_id).record
    with pytest.raises(supplement.SupplementConflict):
        supplement.check("battery", PILOT_LEDGER_RUN, different, **args, accept_reproduction=True)
    with pytest.raises(supplement.SupplementError, match="not a valid battery record"):
        supplement.check("battery", PILOT_LEDGER_RUN, {**record, "gap": 99.0}, **args)


def test_a_record_from_an_interrupted_write_is_recovered_only_while_its_field_is_empty(tmp_path, harness, answered,
                                                                                      monkeypatch) -> None:
    """A crash between the records and the ledger leaves a record without its field; the retry, at a later
    commit, keeps that record (the same numbers, the first commit). Once the field is written, a record
    must be byte-identical (results/supplement_schema.py), and so must one whose kind has no ledger field."""
    ledger, root = selected(tmp_path, [study_a_spec(0.0, 0, pilot=True)])
    raw = harness.battery("dir", {"run_id": PILOT_LEDGER_RUN, "total_steps": TOTAL}, 1_200_000, ["hazard"])
    harness.calls.clear()
    result = enrichment.check_battery_result(PILOT_LEDGER_RUN, raw, step=1_200_000, conditions=["hazard"], selection_seeds=SEL)
    first = "a" * 40
    orphan = result.condition_record("battery", PILOT_LEDGER_RUN, first, "hazard", result.c_id)
    path = supplement.write("battery", PILOT_LEDGER_RUN, orphan.record, ledger_path=ledger, part="hazard")
    text = path.read_text()
    monkeypatch.setattr(enrichment, "_code_commit", lambda: "b" * 40)  # the retry runs at a later commit
    assert enrichment.apply_battery(ledger, run_dir_of(root), ["hazard"], evaluator=harness.battery) == [PILOT_LEDGER_RUN]
    assert path.read_text() == text  # kept, with the commit that first produced it
    row = rows_of(ledger)[PILOT_LEDGER_RUN]
    assert row.gap_hazard == pytest.approx(6.0)
    assert supplement.read("measurement", PILOT_LEDGER_RUN, ledger_path=ledger).code_commit == "b" * 40
    # the field is written now: the same record from yet another commit is a conflict, not a reproduction
    again = result.condition_record("battery", PILOT_LEDGER_RUN, "c" * 40, "hazard", result.c_id)
    with pytest.raises(supplement.SupplementConflict):
        enrichment._commit(ledger, PILOT_LEDGER_RUN, {"gap_hazard": row.gap_hazard}, records=[again])
    assert enrichment._recovering(again, row.model_dump(mode="python")) is False
    assert enrichment._recovering(again, {**row.model_dump(mode="python"), "gap_hazard": None}) is True
    # kinds whose numbers have no ledger field are never a recovery
    training = enrichment._Record("training", PILOT_LEDGER_RUN, {"kind": "training"})
    fewshot = enrichment._Record("fewshot", "B-Moderate-s0", {"budget": 5.0}, "b5")
    assert enrichment._recovering(training, {}) is False
    assert enrichment._recovering(fewshot, {"adapt_steps": {15.0: 500_000}}) is True
    assert enrichment._recovering(fewshot, {"adapt_steps": {5.0: 500_000}}) is False


# ---------------------------------------------------------------------------
# measure and battery (Table 2.1; Table 2.2; eq. 1; Part 4.1 rule 6)
# ---------------------------------------------------------------------------


def test_measure_writes_c_id_and_its_record(tmp_path, harness, answered) -> None:
    ledger, root = selected(tmp_path, [study_a_spec(0.0, 0, pilot=True)])
    assert enrichment.apply_measurement(ledger, run_dir_of(root), evaluator=harness.battery, data_root=tmp_path) == [PILOT_LEDGER_RUN]
    assert harness.calls == [("battery", PILOT_LEDGER_RUN, 1_200_000, ())]  # conditions=[]: C_ID only (pilot/enrichment.py)
    row = rows_of(ledger)[PILOT_LEDGER_RUN]
    record = supplement.read("measurement", PILOT_LEDGER_RUN, ledger_path=ledger)
    assert row.measurement_cost == record.measurement_cost == pytest.approx(25.1)
    assert record.measurement_return == pytest.approx(3.0) and record.step == row.matched_checkpoint_step
    assert json.loads(enrichment.seeds_registry(ledger).read_text())["measurement"] == MEAS
    assert enrichment.apply_measurement(ledger, run_dir_of(root), evaluator=harness.battery) == []
    # a battery later needs no new measurement record, and its C_ID must reproduce the ledger's
    harness.c_id[PILOT_LEDGER_RUN] = 25.2
    with pytest.raises(ContractError, match="never changed"):
        enrichment.apply_battery(ledger, run_dir_of(root), ["dynamics"], evaluator=harness.battery)
    assert not supplement.exists("battery", PILOT_LEDGER_RUN, ledger_path=ledger, part="dynamics")


def test_overlapping_measurement_seeds_are_refused(tmp_path, harness) -> None:
    ledger, root = selected(tmp_path, [study_a_spec(0.0, 0, pilot=True)])

    def leaky(*args):
        return {**harness.battery(*args), "measurement_seeds": SEL, "seeds": {"measurement": SEL}}

    with pytest.raises(ContractError, match="overlap"):
        enrichment.apply_measurement(ledger, run_dir_of(root), evaluator=leaky)


def test_the_battery_needs_the_harness_per_episode_arrays(tmp_path, harness, answered) -> None:
    ledger, root = selected(tmp_path, [study_a_spec(0.0, 0, pilot=True)])

    def bare(*args):  # the costs only, as the pilot-era contract 5 had them
        full = harness.battery(*args)
        return {k: full[k] for k in ("measurement", "hazard", "episodes", "measurement_seeds")}

    with pytest.raises(ContractError, match="per-episode"):
        enrichment.apply_battery(ledger, run_dir_of(root), ["hazard"], evaluator=bare)


def test_the_main_study_battery_is_for_matched_rows_only(tmp_path, harness, answered) -> None:
    """Part 4.1 rule 6: an arm that cannot be matched is left out of the battery; None waits for `enrich match`."""
    specs = [study_a_spec(0.0, 0), study_a_spec(0.0, 1), study_a_spec(0.0, 2)]
    ledger, root = selected(tmp_path, specs)
    enrichment.apply_measurement(ledger, run_dir_of(root), evaluator=harness.battery)
    enrichment._write(ledger, specs[1].run_id, {"matched": True, "infeasible": False})
    enrichment._write(ledger, specs[2].run_id, {"matched": False, "infeasible": False})
    done = enrichment.apply_battery(ledger, run_dir_of(root), ["hazard", "dynamics"], evaluator=harness.battery)
    assert done == [specs[1].run_id]
    rows = rows_of(ledger)
    assert rows[specs[1].run_id].gap_hazard == pytest.approx(6.0)
    for spec in (specs[0], specs[2]):
        assert rows[spec.run_id].gap_hazard is None and rows[spec.run_id].gap_dynamics is None
        assert not supplement.exists("battery", spec.run_id, ledger_path=ledger, part="hazard")
    assert sorted(log_entries(ledger)[-1]["not_matched"]) == [specs[0].run_id, specs[2].run_id]


def test_battery_refuses_a_namesake_run_under_another_data_root(tmp_path, harness, answered) -> None:
    """Smoke and registered data roots share run_ids: the run directory must be the row's run."""
    ledger, root = selected(tmp_path, [study_a_spec(0.0, 0, pilot=True)])
    other = tmp_path / "other" / PILOT_LEDGER_RUN
    shutil.copytree(root / PILOT_LEDGER_RUN, other)  # same commit, but not the checkpoint the row records
    with pytest.raises(ContractError, match="matched checkpoint"):
        enrichment.apply_battery(ledger, lambda r: tmp_path / "other" / r, ["hazard"], evaluator=harness.battery)
    train = json.loads((other / "train_result.json").read_text())
    train["commit_hash"] = "f" * 40  # e.g. a smoke run of later code
    (other / "train_result.json").write_text(json.dumps(train))
    with pytest.raises(ContractError, match="trained at commit"):
        enrichment.apply_battery(ledger, lambda r: tmp_path / "other" / r, ["hazard"], evaluator=harness.battery)
    assert rows_of(ledger)[PILOT_LEDGER_RUN].gap_hazard is None
    done = enrichment.apply_battery(ledger, run_dir_of(root), ["hazard"], evaluator=harness.battery, data_root=root)
    assert done == [PILOT_LEDGER_RUN]
    assert log_entries(ledger)[-1]["data_root"] == str(root)


def test_the_hazard_seeds_are_the_same_for_every_run(tmp_path, harness, answered) -> None:
    """Q-eval-seeds (Table 9.1): the hazard layout seeds and episode seeds are "fixed in code and shared by every run,
    arm, task and study". The first battery fixes them in the seed registry beside the measurement set; a later
    result on other layouts or other episode seeds (e.g. a harness changed between runs) is refused, writing
    nothing."""
    specs = [study_a_spec(0.0, s, pilot=True) for s in (0, 1)]
    ledger, root = selected(tmp_path, specs)

    def shifted(omnisafe_dir: str, spec: dict, step: int, conditions: Sequence[str]) -> dict:
        result = harness.battery(omnisafe_dir, spec, step, conditions)
        if spec["run_id"] == specs[1].run_id:
            result["seeds"]["hazard"] = [s + shift["episodes"] for s in HAZARD_SEEDS]
            result["conditions"]["hazard"]["layout_seeds"] = [s + shift["layouts"] for s in LAYOUTS]
        return result

    for shift in ({"episodes": 7_000, "layouts": 0}, {"episodes": 0, "layouts": 7_000}):
        with pytest.raises(ContractError, match="seeds differ from those of earlier runs"):
            enrichment.apply_battery(ledger, run_dir_of(root), ["hazard"], evaluator=shifted)
        assert rows_of(ledger)[specs[1].run_id].gap_hazard is None
        assert not supplement.exists("battery", specs[1].run_id, ledger_path=ledger, part="hazard")
    registry = json.loads(enrichment.seeds_registry(ledger).read_text())
    assert registry["hazard"] == HAZARD_SEEDS and registry["hazard_layouts"] == LAYOUTS
    shift = {"episodes": 0, "layouts": 0}
    assert enrichment.apply_battery(ledger, run_dir_of(root), ["hazard"], evaluator=shifted) == [specs[1].run_id]


def test_gaps_need_100_episodes_and_finite_costs() -> None:
    with pytest.raises(ContractError, match="100 episodes"):
        enrichment.gaps_from_costs({"measurement": 1.0, "hazard": 2.0, "episodes": 50}, ["hazard"])
    with pytest.raises(ContractError, match="100 episodes"):
        enrichment.gaps_from_costs({"measurement": 1.0, "hazard": 2.0, "episodes": 100.9}, ["hazard"])
    with pytest.raises(ContractError, match="not finite"):
        enrichment.gaps_from_costs({"measurement": 1.0, "hazard": float("nan"), "episodes": 100}, ["hazard"])
    with pytest.raises(ValueError, match="continuations"):
        enrichment.gaps_from_costs({"measurement": 1.0, "finetune": 2.0, "episodes": 100}, ["finetune"])


def test_battery_result_is_checked_at_the_boundary(harness, answered) -> None:
    good = harness.battery("dir", {"run_id": "A-x-s0", "total_steps": TOTAL}, 1_200_000, ["hazard", "dynamics"])
    result = enrichment.check_battery_result("A-x-s0", good, step=1_200_000, conditions=["hazard", "dynamics"],
                                             selection_seeds=SEL)
    assert result.fields() == {"measurement_cost": pytest.approx(25.1), "gap_hazard": pytest.approx(6.0),
                               "gap_dynamics": pytest.approx(2.5)}
    with pytest.raises(ContractError, match="evaluated step"):
        enrichment.check_battery_result("A-x-s0", good, step=1_000_000, conditions=[], selection_seeds=SEL)
    moved = {**good, "seeds": {**good["seeds"], "dynamics": HAZARD_SEEDS}}
    with pytest.raises(ContractError, match="dynamics episodes were not run on the measurement seeds"):
        enrichment.check_battery_result("A-x-s0", moved, step=1_200_000, conditions=["dynamics"], selection_seeds=SEL)
    with pytest.raises(ContractError, match="not a finite number"):
        enrichment.check_battery_result("A-x-s0", {**good, "hazard": None}, step=1_200_000, conditions=["hazard"],
                                        selection_seeds=SEL)
    for bad in (None, "abc", 100.9, 100.0, True):  # a contract violation, never a TypeError or a truncation
        with pytest.raises(ContractError, match="A-x-s0 step 1200000: battery evaluation must use 100 episodes"):
            enrichment.check_battery_result("A-x-s0", {**good, "episodes": bad}, step=1_200_000, conditions=[],
                                            selection_seeds=SEL)


def _replaced(slot: int, seeds: list[int], r: int = 0, step: int = 412) -> dict:
    """The r-th replacement of an evaluation call (Q-mujoco-exception): episode ``slot`` of ``seeds`` ran the set's
    reserve seed min(seeds) + 50,000 + r."""
    return {"slot_index": slot, "seed": seeds[slot], "reserve_seed": min(seeds) + 50_000 + r, "step": step,
            "warning": "mjWARN_BADQACC x1"}


def test_replacements_of_unstable_episodes_reach_the_records_and_the_seed_sets_stay_planned(
        tmp_path, harness, answered) -> None:
    """Q-mujoco-exception (Table 9.1): a battery whose measurement and hazard evaluations replaced unstable
    episodes is written with the replacements in its records; the seed registry keeps the planned measurement set
    (docs/DECISIONS.md, Q-mujoco-exception: "Seed-set checks compare the planned canonical seeds")."""
    ledger, root = selected(tmp_path, [study_a_spec(0.0, 0, pilot=True)])

    def unstable(*args):
        result = harness.battery(*args)
        result["unstable_replacements"] = {"measurement": [_replaced(4, MEAS)], "hazard": [_replaced(0, HAZARD_SEEDS)]}
        return result

    assert enrichment.apply_battery(ledger, run_dir_of(root), ["hazard"], evaluator=unstable) == [PILOT_LEDGER_RUN]
    measurement = supplement.read("measurement", PILOT_LEDGER_RUN, ledger_path=ledger)
    assert [(u.slot_index, u.seed, u.reserve_seed) for u in measurement.episodes.unstable_replacements] == [
        (4, 2_000_004, 2_050_000)]
    assert list(measurement.episodes.seeds) == MEAS
    hazard = supplement.read("battery", PILOT_LEDGER_RUN, ledger_path=ledger, part="hazard")
    assert [u.reserve_seed for u in hazard.episodes.unstable_replacements] == [3_150_000]
    assert json.loads(enrichment.seeds_registry(ledger).read_text())["measurement"] == MEAS
    off_rule = {"dynamics": [{**_replaced(1, MEAS), "reserve_seed": 2_050_003}]}  # not the next unused reserve seed
    with pytest.raises(ContractError, match="not the next unused seed"):
        enrichment.apply_battery(ledger, run_dir_of(root), ["dynamics"],
                                 evaluator=lambda *a: {**harness.battery(*a), "unstable_replacements": off_rule})
    assert not supplement.exists("battery", PILOT_LEDGER_RUN, ledger_path=ledger, part="dynamics")
    six = {"dynamics": [_replaced(i, MEAS, r=i) for i in range(6)]}  # more than 5 per evaluation call
    with pytest.raises(ContractError, match="at most 5 per evaluation call"):
        enrichment.apply_battery(ledger, run_dir_of(root), ["dynamics"],
                                 evaluator=lambda *a: {**harness.battery(*a), "unstable_replacements": six})


def test_study_b_records_carry_the_replacements_of_each_budget_and_horizon(tmp_path, harness, monkeypatch) -> None:
    """Q-mujoco-exception: a zero-shot budget's and a few-shot horizon's replacements reach their blocks."""
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset({"Q-studyb-eval"}))
    parent = study_b_spec("Moderate")
    paths, root = ledger_with(tmp_path, [parent])

    def zero_shot(*args):
        return {**harness.zero_shot(*args), "unstable_replacements": {15.0: [_replaced(9, MEAS)], 10.0: []}}

    assert enrichment.apply_zero_shot(paths.main, run_dir_of(root), evaluator=zero_shot) == [parent.run_id]
    blocks = {b.budget: b for b in supplement.read("zero_shot", parent.run_id, ledger_path=paths.main).budgets}
    assert [u.reserve_seed for u in blocks[15.0].unstable_replacements] == [2_050_000]
    assert blocks[10.0].unstable_replacements is None and blocks[5.0].unstable_replacements is None
    _fewshot_runs(root, parent, budgets=(5.0,))

    def fewshot(*args):
        result = harness.fewshot(*args)
        result["by_horizon"][500_000]["unstable_replacements"] = [_replaced(2, MEAS), _replaced(99, MEAS, r=1)]
        return result

    assert enrichment.apply_fewshot(paths.main, run_dir_of(root), evaluator=fewshot) == [parent.run_id]
    record = supplement.read("fewshot", parent.run_id, ledger_path=paths.main, part="b5")
    by_horizon = {b.horizon: b for b in record.by_horizon}
    assert [u.reserve_seed for u in by_horizon[500_000].unstable_replacements] == [2_050_000, 2_050_001]
    assert by_horizon[200_000].unstable_replacements is None and list(record.seeds) == MEAS


# ---------------------------------------------------------------------------
# match (Part 4.1 rules 2 to 6; Q-matched-cost-set, Q-arm-complete)
# ---------------------------------------------------------------------------


def _measured_arms(tmp_path, harness, costs: dict, seeds=range(5)) -> tuple[Path, Path, dict]:
    """A main ledger of the given arms ({N: [C_ID per seed]}), selected and measured."""
    specs = [study_a_spec(N, s) for N in costs for s in seeds]
    ledger, root = selected(tmp_path, specs)
    for N, values in costs.items():
        for s, value in zip(seeds, values):
            harness.c_id[study_a_spec(N, s).run_id] = value
    enrichment.apply_measurement(ledger, run_dir_of(root), evaluator=harness.battery)
    return ledger, root, {s.run_id: s for s in specs}


def test_match_writes_the_flags_of_rules_2_to_6_and_rechecks_them(tmp_path, harness, answered) -> None:
    costs = {0.0: [24.0, 25.0, 26.0, 25.0, 25.0], 0.5: [27.0, 28.0, 27.5, 28.5, 27.0]}  # 25.0 and 27.6: unmatched at 2.5
    ledger, root, specs = _measured_arms(tmp_path, harness, costs)
    done = enrichment.apply_matching(ledger, surplus_extra=0)
    assert sorted(done) == sorted(specs)
    rows = rows_of(ledger)
    assert {(r.N, r.matched, r.infeasible) for r in rows.values()} == {(0.0, True, False), (0.5, False, False)}
    assert enrichment.apply_matching(ledger, surplus_extra=0) == []  # nothing changes on a second call
    # rule 7 (a): at tolerance 5.0 the late arm matches (|27.6 - 25| <= 5)
    flags = enrichment.rule_matching([r.model_dump() for r in rows.values()], tolerance=R.SENSITIVITY_TOLERANCE)
    assert {flags[k]["matched"] for k in flags} == {True}


def test_match_refuses_a_matcher_that_disagrees_with_the_rules(tmp_path, harness, answered) -> None:
    costs = {0.0: [24.0, 25.0, 26.0, 25.0, 25.0], 0.5: [25.0] * 5}
    ledger, _, _ = _measured_arms(tmp_path, harness, costs)

    def lenient(rows, *, tolerance, seed_targets):
        return {r["run_id"]: {"matched": False, "infeasible": False} for r in rows}

    with pytest.raises(ContractError, match="differ from rules 2 to 5"):
        enrichment.apply_matching(ledger, surplus_extra=0, matcher=lenient)
    assert all(r.matched is None for r in rows_of(ledger).values())


def test_match_leaves_incomplete_arms_waiting(tmp_path, harness, answered) -> None:
    # four seeds of the late arm: incomplete (target 5), so only the reference arm gets its flags
    specs = [study_a_spec(0.0, s) for s in range(5)] + [study_a_spec(0.5, s) for s in range(4)]
    ledger, root = selected(tmp_path, specs)
    enrichment.apply_measurement(ledger, run_dir_of(root), evaluator=harness.battery)
    done = enrichment.apply_matching(ledger, surplus_extra=0)
    assert sorted(done) == sorted(s.run_id for s in specs[:5])
    entry = log_entries(ledger)[-1]
    assert "4 of its 5 seeds" in entry["arms_waiting"]["A-PointGoal1-N0.50-abrupt-total"]
    # with one surplus seed per primary-comparison arm (Part 5.5), the reference arm is incomplete too
    other = tmp_path / "second"
    ledger2, root2 = selected(other, specs[:5])
    enrichment.apply_measurement(ledger2, run_dir_of(root2), evaluator=harness.battery)
    assert enrichment.apply_matching(ledger2, surplus_extra=1, stored_surplus_extra="1") == []
    assert "5 of its 6 seeds" in log_entries(ledger2)[-1]["arms_waiting"]["A-PointGoal1-N0.00"]


def test_match_gates_and_its_surplus_cross_check(tmp_path, harness, monkeypatch) -> None:
    costs = {0.0: [25.0] * 5}
    ledger, _, _ = _measured_arms(tmp_path, harness, costs)
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset())  # the keys open, whatever the repository answers
    with pytest.raises(PendingQuestionError, match="Q-matched-cost-set, Q-arm-complete"):
        enrichment.apply_matching(ledger, surplus_extra=0)
    with errors.allow_pending():
        with pytest.raises(ContractError, match="differs from the data root's 2"):
            enrichment.apply_matching(ledger, surplus_extra=0, stored_surplus_extra="2")
        # a data root that stores no surplus has none (Scheduler.seed_target): another E is refused, not trusted
        with pytest.raises(ContractError, match="--surplus-extra 1 differs from the data root's 0"):
            enrichment.apply_matching(ledger, surplus_extra=1)
        assert len(enrichment.apply_matching(ledger, surplus_extra=0, stored_surplus_extra="0")) == 5
    with pytest.raises(ValueError, match="surplus_extra"):
        enrichment.seed_targets(["A-PointGoal1-N0.00"], 8)
    assert enrichment.seed_targets(["A-PointGoal1-N0.00", "A-PointButton1-N0.00"], 2) == {
        "A-PointButton1-N0.00": 5, "A-PointGoal1-N0.00": 7}  # only PointGoal1's arms are primary (Part 5.5)


def test_rule_matching_under_every_reading_of_the_threshold_arithmetic() -> None:
    """Q-threshold-arithmetic: "n or n - 1 in the standard deviation, strict or inclusive comparisons"."""
    def rows(ref, late):
        out = []
        for N, costs in ((0.0, ref), (0.5, late)):
            for s, cost in enumerate(costs):
                spec = study_a_spec(N, s)
                out.append({"run_id": spec.run_id, "seed": s, "study": "A", "completed": True, "task": spec.task,
                            "arm": spec.arm, "N": N, "treatment": None, "controller_variant": None, "measurement_cost": cost})
        return out

    readings = enrichment.THRESHOLD_READINGS
    assert readings[0] == enrichment.PROPOSED_READING == enrichment.ThresholdReading(1, True, 4)
    assert len(set(readings)) == 8
    late = "A-PointGoal1-N0.50-abrupt-total-s0"
    boundary = rows([25.0] * 5, [27.5] * 5)  # |27.5 - 25| = 2.5 exactly: matched only by an inclusive reading
    got = {r: enrichment.rule_matching(boundary, tolerance=R.MATCH_TOLERANCE, reading=r)[late]["matched"] for r in readings}
    assert got == {r: r.inclusive for r in readings}
    spread = rows([10.0, 20.0, 25.0, 30.0, 40.0], [25.0] * 5)  # sample SD 11.18 > 10, population SD 10.0
    infeasible = {r: enrichment.rule_matching(spread, tolerance=R.MATCH_TOLERANCE, reading=r)[late]["infeasible"]
                  for r in readings}
    assert infeasible == {r: r.ddof == 1 for r in readings}
    noisy = rows([24.9] * 3, [27.3, 27.3, 27.6])  # means 24.90 and 27.40: 2.500000000000007 in binary floating point
    unrounded = {r: enrichment.rule_matching(noisy, tolerance=R.MATCH_TOLERANCE, reading=r)[late]["matched"]
                 for r in readings if r.inclusive}
    assert unrounded == {r: r.decimals is not None for r in readings if r.inclusive}


def test_match_writes_the_answer_flags_and_names_those_the_threshold_arithmetic_decides(tmp_path, harness,
                                                                                        monkeypatch) -> None:
    """Q-threshold-arithmetic is a report key (configs/registered.py; pilot/enrichment.py gates ``match`` by
    Q-matched-cost-set and Q-arm-complete only): while it is open, every complete arm gets the flags of its
    answer, and an arm whose flags another reading would change is named as provisional, not held."""
    costs = {0.0: [25.0] * 5, 0.1: [26.0] * 5, 0.5: [27.5] * 5}  # N = 0.50 on the 2.5 boundary exactly
    ledger, root, specs = _measured_arms(tmp_path, harness, costs)
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset({"Q-matched-cost-set", "Q-arm-complete"}))
    assert R.is_open("Q-threshold-arithmetic")
    done = enrichment.apply_matching(ledger, surplus_extra=0)  # no PendingQuestionError: it gates nothing
    assert sorted(done) == sorted(specs)
    assert {(r.N, r.matched, r.infeasible) for r in rows_of(ledger).values()} == {
        (0.0, True, False), (0.1, True, False), (0.5, True, False)}  # the answer: inclusive
    late = "A-PointGoal1-N0.50-abrupt-total"
    assert list(done.provisional) == [late]
    why = done.provisional[late]
    assert "Q-threshold-arithmetic" in why and "sample SD, <=, rounded to 1e-4: matched True" in why
    assert "sample SD, <, rounded to 1e-4: matched False" in why
    entry = log_entries(ledger)[-1]
    assert entry["provisional"] == {late: why} and len(entry["run_ids"]) == 15 and "waiting" not in entry
    # once answered, nothing is provisional
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset({"Q-matched-cost-set", "Q-arm-complete",
                                                           "Q-threshold-arithmetic"}))
    again = enrichment.apply_matching(ledger, surplus_extra=0)
    assert again == [] and again.provisional == {}


def test_match_names_a_whole_task_whose_feasibility_the_threshold_arithmetic_decides(tmp_path, harness,
                                                                                    monkeypatch) -> None:
    costs = {0.0: [10.0, 20.0, 25.0, 30.0, 40.0], 0.5: [25.0] * 5}  # infeasible by the sample SD, not the population SD
    ledger, root, specs = _measured_arms(tmp_path, harness, costs)
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset({"Q-matched-cost-set", "Q-arm-complete"}))
    done = enrichment.apply_matching(ledger, surplus_extra=0)
    assert len(done) == 10 and {r.infeasible for r in rows_of(ledger).values()} == {True}
    assert sorted(done.provisional) == ["A-PointGoal1-N0.00", "A-PointGoal1-N0.50-abrupt-total"]
    assert all("population SD, <=, rounded to 1e-4: matched True, infeasible False" in why
               for why in done.provisional.values())


def test_enrichment_gates_only_on_result_gate_keys() -> None:
    """Every key the pipeline's own producers pass to ``require_answered`` is a result gate (configs/registered.py
    and tests/test_registered.py classify the keys); a report key such as Q-threshold-arithmetic gates nothing."""
    import re

    from test_registered import REPORT_KEYS, RESULT_GATE_KEYS

    root = Path(__file__).resolve().parents[1]
    found: set[str] = set()
    for name in ("pilot/enrichment.py", "pilot/ledger_writer.py", "pilot/supplement.py", "pilot/go_decision.py",
                 "pilot/__main__.py"):
        source = (root / name).read_text(encoding="utf-8")
        calls = re.findall(r"require_answered\(([^)]*)", source)
        keys = {k for call in calls for k in re.findall(r"\"(Q-[a-z0-9-]+)\"", call)}
        assert keys <= RESULT_GATE_KEYS, f"{name} gates on {sorted(keys - RESULT_GATE_KEYS)}, not result gates"
        found |= keys
    assert {"Q-tie-break", "Q-matched-cost-set", "Q-arm-complete", "Q-adapt-censoring"} <= found  # the scan sees them
    assert "Q-threshold-arithmetic" in REPORT_KEYS


def test_match_refuses_the_pilot_ledger(tmp_path, answered) -> None:
    ledger, _ = selected(tmp_path, [study_a_spec(0.0, 0, pilot=True)])
    with pytest.raises(ContractError, match="pilot's runs are not reused"):
        enrichment.apply_matching(ledger, surplus_extra=0)


@pytest.mark.parametrize(("ref", "late", "flags"), [
    ([24.0, 25.0, 26.0, 25.0, 25.0], [27.5] * 5, (True, True, False)),  # |27.5 - 25| = 2.5: inclusive
    ([27.0, 28.0, 27.5, 28.0, 27.5], [27.5] * 5, (False, False, True)),  # rule 3: 27.6 > 25 + 2.5
    ([10.0, 40.0, 25.0, 25.0, 25.0], [25.0] * 5, (False, False, True)),  # rule 5: SD 10.6 > 10
])
def test_rule_matching_follows_rules_3_and_5(ref, late, flags) -> None:
    def row(N, s, cost):
        spec = study_a_spec(N, s)
        return {"run_id": spec.run_id, "seed": s, "study": "A", "completed": True, "task": spec.task, "arm": spec.arm, "N": N,
                "treatment": None, "controller_variant": None, "measurement_cost": cost}

    rows = [row(0.0, s, c) for s, c in enumerate(ref)] + [row(0.5, s, c) for s, c in enumerate(late)]
    got = enrichment.rule_matching(rows, tolerance=R.MATCH_TOLERANCE)
    ref_matched, late_matched, infeasible = flags
    assert {got[r["run_id"]]["matched"] for r in rows[:5]} == {ref_matched}
    assert {got[r["run_id"]]["matched"] for r in rows[5:]} == {late_matched}
    assert {v["infeasible"] for v in got.values()} == {infeasible}
    from analysis import matching

    assert matching.match_arms(rows, tolerance=R.MATCH_TOLERANCE, seed_targets={
        "A-PointGoal1-N0.00": 5, "A-PointGoal1-N0.50-abrupt-total": 5}) == got  # Role 4's rules agree


def test_an_auxiliary_run_is_not_a_seed_of_its_arm_in_the_matching() -> None:
    """The N = 0 run queued for a warm-started replacement's seed (manifest.AUXILIARY_NOTE; Q-warm-start) never
    counts toward its arm's seed target or its matched cost (rules 2 to 5)."""
    arm = next(s for s in manifest.design("main", seeds=(0,)) if s.N == 0.0 and s.task == "SafetyCarGoal1-v0")

    def row(seed: int, notes: str = "") -> dict:
        s = arm.with_seed(seed)
        return {"run_id": s.run_id, "seed": seed, "study": "A", "completed": True, "task": s.task, "N": 0.0,
                "treatment": None, "controller_variant": None, "measurement_cost": 25.0, "notes": notes}

    rows = [row(seed) for seed in R.SEEDS[:-1]] + [row(5, f"a; {manifest.AUXILIARY_NOTE}")]
    readiness = enrichment.arm_readiness(rows, surplus_extra=0)
    assert arm.task not in readiness.ready and "4 of its 5 seeds" in readiness.waiting[arm.arm_id]
    readiness = enrichment.arm_readiness(rows + [row(R.SEEDS[-1])], surplus_extra=0)
    assert [r["seed"] for r in readiness.ready[arm.task]] == list(R.SEEDS)


def test_an_auxiliary_row_expects_no_matching_fields(tmp_path, harness, answered) -> None:
    """`enrich match` never flags an auxiliary row (arm_readiness leaves it out), so completeness must not expect
    matched/infeasible of it, or the freeze is never ready; and `enrich battery` leaves it out without listing it as
    waiting for a flag that never comes."""
    paths = paths_in(tmp_path)
    root = tmp_path / "data" / "checkpoints"
    for N in (0.0, 0.5):
        for seed in R.SEEDS:
            spec = study_a_spec(N, seed)
            write_run(spec, make_run(root, spec), paths)
    aux = study_a_spec(0.0, 5)
    write_run(aux, make_run(root, aux), paths, extra_notes=(manifest.AUXILIARY_NOTE,))
    enrichment.apply_selection(paths.main)
    enrichment.apply_measurement(paths.main, run_dir_of(root), evaluator=harness.battery)
    assert aux.run_id not in enrichment.apply_matching(paths.main, surplus_extra=0)
    row = rows_of(paths.main)[aux.run_id]
    assert row.matched is None and row.infeasible is None and row.measurement_cost is not None
    assert "matched" not in enrichment.expected_fields(row) and "measurement_cost" in enrichment.expected_fields(row)
    assert not any(kind in ("battery", "continuation") for kind, _ in enrichment.expected_records(row))
    report = enrichment.completeness_report(paths.main)
    missing = report["arms"]["A-PointGoal1-N0.00"]["fields_missing"]
    assert "matched" not in missing and "infeasible" not in missing  # the five seeds were flagged
    # (no `enrich controller` here)
    assert sorted(name for name, ids in missing.items() if aux.run_id in ids) == ["lambda_final"]
    done = enrichment.apply_battery(paths.main, run_dir_of(root), ["hazard"], evaluator=harness.battery)
    assert len(done) == 10 and aux.run_id not in done and done.waiting == {}
    again = enrichment.apply_battery(paths.main, run_dir_of(root), ["hazard"], evaluator=harness.battery)
    assert again == [] and again.waiting == {} and log_entries(paths.main)[-1]["action"] == "battery"
    assert len([e for e in log_entries(paths.main) if e["action"] == "battery"]) == 1  # nothing waits: no entry


# ---------------------------------------------------------------------------
# controller and training (Tables 2.1, 2.3, 2.4)
# ---------------------------------------------------------------------------


def test_controller_quantities_are_written_after_the_recheck(tmp_path, answered) -> None:
    specs = [study_a_spec(0.0, 0), study_a_spec(0.5, 0)]
    paths, root = ledger_with(tmp_path, specs)
    done = enrichment.apply_controller(paths.main, run_dir_of(root))
    assert sorted(done) == sorted(s.run_id for s in specs)
    rows = rows_of(paths.main)
    late = rows[specs[1].run_id]
    # fake progress: lambda after epoch e = 0.001 + 0.035 (e + 1); 100 epochs; onset at epoch 50
    assert late.lambda_final == pytest.approx(0.001 + 0.035 * (91 + 100) / 2)  # mean over epochs 90..99
    assert late.lambda_peak == pytest.approx(0.001 + 0.035 * 100)  # the window (1M, 3M] is cut at the end
    # every value from epoch 85 on lies within 10 percent of the final value 3.3435 (epoch 84: 2.976 < 3.00915)
    assert late.settling_steps == R.STEPS_PER_EPOCH * 86 - 1_000_000
    assert enrichment.apply_controller(paths.main, run_dir_of(root)) == []


def test_controller_and_training_refuse_a_namesake_run_under_another_data_root(tmp_path, answered) -> None:
    """As for every other command, the run directory must be the row's run: the same commit and the final checkpoint
    at the path the row stores. A namesake under another data root at the same commit (another multiplier trace) is
    never read into lambda_final or the training record, which are written once."""
    spec = study_a_spec(0.5, 0)
    paths, root = ledger_with(tmp_path, [spec], batch_cost=lambda e: 40.0 if e < 60 else 20.0)
    other = tmp_path / "other"
    shutil.copytree(root / spec.run_id, other / spec.run_id)
    _rewrite_progress(omni_of(other / spec.run_id), lambda r: {
        **r, enrichment.MULTIPLIER_COLUMN: str(2 * float(r[enrichment.MULTIPLIER_COLUMN]))})
    for apply in (enrichment.apply_controller, enrichment.apply_training):
        with pytest.raises(ContractError, match="final checkpoint"):
            apply(paths.main, lambda run_id: other / run_id)
    assert rows_of(paths.main)[spec.run_id].lambda_final is None
    assert not supplement.exists("training", spec.run_id, ledger_path=paths.main)
    assert enrichment.apply_controller(paths.main, run_dir_of(root)) == [spec.run_id]
    assert enrichment.apply_training(paths.main, run_dir_of(root)) == [spec.run_id]
    assert rows_of(paths.main)[spec.run_id].lambda_final == pytest.approx(0.001 + 0.035 * (91 + 100) / 2)


def test_controller_needs_q_controller_quantities_and_equal_results(tmp_path, monkeypatch) -> None:
    paths, root = ledger_with(tmp_path, [study_a_spec(0.0, 0)])
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset())  # the key open, whatever the repository answers
    with pytest.raises(PendingQuestionError, match="Q-controller-quantities"):
        enrichment.apply_controller(paths.main, run_dir_of(root))
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset({"Q-controller-quantities"}))

    def off(epochs, values, **kw):
        return {"lambda_peak": 1.0, "lambda_final": 2.0, "settling_steps": None}

    with pytest.raises(ContractError, match="Table 2.4 at epoch resolution"):
        enrichment.apply_controller(paths.main, run_dir_of(root), quantities=off)
    assert rows_of(paths.main)["A-PointGoal1-N0.00-s0"].lambda_final is None


def test_controller_reference_matches_role_3() -> None:
    from metrics.controller import controller_quantities

    trace = [0.001 * (e % 7) + (0.5 if e > 60 else 0.0) for e in range(100)]
    with errors.allow_pending():
        for onset in (0, 400_000, 1_000_000, 1_980_000):
            kw = {"onset_step": onset, "total_steps": TOTAL, "steps_per_epoch": R.STEPS_PER_EPOCH}
            assert enrichment.controller_reference(list(range(100)), trace, **kw) == controller_quantities(
                list(range(100)), trace, **kw)
        # lambda_final 0 (Table 9.1): the band is {0}, so a trace of zeros is settled from its first epoch
        assert enrichment.controller_reference(list(range(10)), [0.0] * 10, onset_step=0, total_steps=20_000,
                                               steps_per_epoch=2_000)["settling_steps"] == 2_000


def test_controller_reference_and_role_3_agree_on_a_zero_final_value() -> None:
    """Q-controller-quantities (Table 9.1): settling is defined for a final value of 0, as the epoch from which the
    multiplier stays exactly 0, in both implementations alike (``enrich controller`` compares them)."""
    from metrics.controller import controller_quantities

    held, steps = R.LAGRANGE_MULTIPLIER_INIT, 20_000
    traces = {
        3 * steps: [held] * 2 + [0.4, 0.2] + [0.0] * 16,  # the first all-zero epoch is 4: (4 + 1) x E - 2 x E
        17 * steps: [held] * 2 + [0.5] * 16 + [0.0] * 2,  # 0 only in the last 10 percent: from epoch 18
        7 * steps: [held] * 2 + [0.0] * 5 + [1e-9] + [0.0] * 12,  # not exactly 0 at epoch 7: from epoch 8
    }
    with errors.allow_pending():
        for expected, trace in traces.items():
            kw = {"onset_step": 2 * steps, "total_steps": 20 * steps, "steps_per_epoch": steps}
            ours = enrichment.controller_reference(list(range(20)), trace, **kw)
            assert ours == controller_quantities(list(range(20)), trace, **kw)
            assert ours["lambda_final"] == 0.0 and ours["settling_steps"] == expected


def test_controller_refuses_a_non_finite_or_negative_multiplier(tmp_path, answered) -> None:
    """Role 3 refuses such a trace with ValueError; the command refuses it as a contract violation of the run."""
    kw = {"onset_step": 0, "total_steps": TOTAL, "steps_per_epoch": R.STEPS_PER_EPOCH}
    for bad in (float("nan"), float("inf"), -0.5):
        with pytest.raises(ContractError, match="non-finite or negative"):
            enrichment.controller_reference(list(range(100)), [1.0] * 99 + [bad], **kw)
    spec = study_a_spec(0.0, 0)
    paths, root = ledger_with(tmp_path, [spec])
    progress = next((root / spec.run_id).rglob("progress.csv"))
    _rewrite_progress(progress.parent, lambda r: {**r, enrichment.MULTIPLIER_COLUMN: "nan"}
                      if r["TotalEnvSteps"] == str(TOTAL) else r)  # the last epoch

    def never(*a, **k):
        raise AssertionError("Role 3 is not called on a trace it refuses")

    with pytest.raises(ContractError, match=f"{spec.run_id}: the multiplier trace has non-finite"):
        enrichment.apply_controller(paths.main, run_dir_of(root), quantities=never)
    assert rows_of(paths.main)[spec.run_id].lambda_final is None


def test_training_record_of_a_late_run(tmp_path, answered) -> None:
    spec = study_a_spec(0.5, 0)
    paths, root = ledger_with(tmp_path, [spec], batch_cost=lambda e: 40.0 if e < 60 else 20.0)
    assert enrichment.apply_training(paths.main, run_dir_of(root)) == [spec.run_id]
    record = supplement.read("training", spec.run_id, ledger_path=paths.main)
    assert record.onset_step == 1_000_000 and record.steps_per_epoch == 20_000 and record.total_steps == TOTAL
    # epoch 60 ends at 1.22M
    assert record.recovery_steps == 61 * R.STEPS_PER_EPOCH - 1_000_000 and record.recovery_censored is False
    assert record.overshoot == pytest.approx(record.lambda_peak - record.lambda_final)
    assert record.plasticity_onset.step == 1_000_000 and record.plasticity_check.step == 1_200_000
    row = fake_launcher.plasticity_row(1_200_000)
    assert record.plasticity_check.norm_cost_critic == pytest.approx(float(row[fake_launcher.PLASTICITY_COLUMNS.index("norm_cost_critic")]))  # the critics' norms (Table 2.3)
    assert record.intervention is None
    assert enrichment.apply_training(paths.main, run_dir_of(root)) == []
    assert rows_of(paths.main)[spec.run_id].lambda_final is None  # training writes no ledger field


def _rate_limited(seed: int = 0) -> RunSpec:
    """The rate-limited controller arm at N = 0.50 (Table 3.3 "Controller variants"), on the fixtures' scale."""
    base = study_a_spec(0.5, seed)
    return dataclasses.replace(base, run_id=f"{base.arm_id}-rate_limited-s{seed}", arm=f"{base.arm}-rate_limited",
                               group="controller", controller_variant="rate_limited",
                               params={**base.params, "rate_limit_relative": R.RATE_LIMIT_RELATIVE,
                                       "rate_limit_absolute": R.RATE_LIMIT_ABSOLUTE})


CLIPPED = set(range(60, 70))  # the epochs in which the clip bound: 10 of the 50 constrained epochs (onset epoch 50)


def _log_proposed(run_dir: Path) -> None:
    """Onset/MultiplierProposed as envs.onset logs it: NaN before onset, the multiplier where the clip did not
    bind, another value where it did."""
    def add(r: dict) -> dict:
        e = int(float(r["Train/Epoch"]))
        lam = r[enrichment.MULTIPLIER_COLUMN]
        proposed = "nan" if e < 50 else (str(float(lam) + 0.25) if e in CLIPPED else lam)
        return {**r, enrichment.PROPOSED_MULTIPLIER_COLUMN: proposed}

    _rewrite_progress(omni_of(run_dir), add)


def test_the_rate_clips_share_reaches_the_training_record(tmp_path, answered) -> None:
    """Q-rate-limit (Table 9.1): "the report gives the share of constrained epochs in which the clip bound". Role 3's
    ``rate_limit_clip_share`` and ``rate_limit_clip_reference`` must agree; ``enrich controller`` logs it (no ledger
    field), the training record carries it to the analysis (H4); any other run has none."""
    limited, plain = _rate_limited(), study_a_spec(0.5, 1)
    paths, root = ledger_with(tmp_path, [limited, plain], batch_cost=lambda e: 20.0)
    _log_proposed(root / limited.run_id)
    assert sorted(enrichment.apply_controller(paths.main, run_dir_of(root))) == sorted([limited.run_id, plain.run_id])
    assert log_entries(paths.main)[-1]["rate_limit_clip_share"] == {limited.run_id: 0.2}
    assert sorted(enrichment.apply_training(paths.main, run_dir_of(root))) == sorted([limited.run_id, plain.run_id])
    assert supplement.read("training", limited.run_id, ledger_path=paths.main).rate_limit_clip_share == 0.2
    assert supplement.read("training", plain.run_id, ledger_path=paths.main).rate_limit_clip_share is None
    stored = supplement.record_path("training", plain.run_id, ledger_path=paths.main).read_text()
    assert "rate_limit_clip_share" not in stored  # absent, as in a record written before the field existed


def test_the_rate_clips_share_is_checked_and_needs_the_proposed_values(tmp_path, answered) -> None:
    from metrics.controller import rate_limit_clip_share

    spec = _rate_limited()
    paths, root = ledger_with(tmp_path, [spec], batch_cost=lambda e: 20.0)
    with pytest.raises(ContractError, match="Onset/MultiplierProposed"):  # the rate-limited arm logs it every epoch
        enrichment.apply_training(paths.main, run_dir_of(root))
    _log_proposed(root / spec.run_id)
    with pytest.raises(ContractError, match="the rate clip's share 0.3"):
        enrichment.apply_training(paths.main, run_dir_of(root), clip_share=lambda *a, **k: 0.3)
    assert not supplement.exists("training", spec.run_id, ledger_path=paths.main)
    lam = [0.001] * 50 + [0.1 * e for e in range(50)]
    proposed = [math.nan] * 50 + [v + (0.01 if e % 4 == 0 else 0.0) for e, v in enumerate(lam[50:])]
    kw = {"onset_step": 1_000_000, "total_steps": TOTAL, "steps_per_epoch": R.STEPS_PER_EPOCH}
    assert enrichment.rate_limit_clip_reference(list(range(100)), lam, proposed, **kw) == rate_limit_clip_share(
        list(range(100)), lam, proposed, **kw) == 13 / 50


def test_training_needs_the_batch_cost_of_a_late_run_and_censors_recovery(tmp_path, answered) -> None:
    spec = study_a_spec(0.5, 0)
    paths, root = ledger_with(tmp_path, [spec])
    with pytest.raises(ContractError, match="Metrics/BatchEpCost"):
        enrichment.apply_training(paths.main, run_dir_of(root))
    other, root2 = ledger_with(tmp_path / "2", [spec], batch_cost=lambda e: 40.0)
    enrichment.apply_training(other.main, run_dir_of(root2))
    record = supplement.read("training", spec.run_id, ledger_path=other.main)
    assert record.recovery_steps is None and record.recovery_censored is True


def test_training_record_carries_the_intervention_summary(tmp_path, answered) -> None:
    spec = study_a_spec(0.5, 0, treatment="reset")
    paths = paths_in(tmp_path)
    root = tmp_path / "data" / "checkpoints"
    run_dir = make_run(root, spec, batch_cost=lambda e: 20.0)
    write_run(spec, run_dir, paths)
    with pytest.raises(ContractError, match="no intervention.json"):
        enrichment.apply_training(paths.main, run_dir_of(root))
    (omni_of(run_dir) / "intervention.json").write_text(json.dumps({
        "intervention": "reset", "layers": ["mean.2", "mean.4"], "generator_seed": 123,
        "trainable_parameter_count_before": 4_000, "trainable_parameter_count_after": 4_000,
        "output_max_abs_change": 0.25}))
    assert enrichment.apply_training(paths.main, run_dir_of(root)) == [spec.run_id]
    summary = supplement.read("training", spec.run_id, ledger_path=paths.main).intervention
    assert (summary.treatment, summary.step, summary.layers) == ("reset", 1_000_000, ["mean.2", "mean.4"])
    assert summary.max_output_difference == 0.25


def test_intervention_file_and_plasticity_fields_agree_with_their_owners() -> None:
    envs_onset = pytest.importorskip("envs.onset")
    assert enrichment.INTERVENTION_FILE == envs_onset.INTERVENTION_FILE
    from results.supplement_schema import PlasticityRow

    assert set(enrichment.PLASTICITY_ROW_FIELDS) | {"step", "checkpoint"} == set(PlasticityRow.model_fields)


# ---------------------------------------------------------------------------
# continuations (Table 2.2 fine-tuning and transfer; pilot/manifest.py)
# ---------------------------------------------------------------------------


def _matched_parent(tmp_path, harness) -> tuple[Path, Path, RunSpec]:
    spec = study_a_spec(0.0, 0)
    ledger, root = selected(tmp_path, [spec])
    enrichment.apply_measurement(ledger, run_dir_of(root), evaluator=harness.battery)
    enrichment._write(ledger, spec.run_id, {"matched": True, "infeasible": False})
    return ledger, root, spec


def _continue(root: Path, ledger: Path, parent: RunSpec, condition: str) -> RunSpec:
    row = rows_of(ledger)[parent.run_id]
    cont = manifest.battery_continuation(parent, row.model_dump(mode="python"), condition)
    make_run(root, cont)
    return cont


def test_continuations_write_the_gap_from_the_continued_runs(tmp_path, harness, answered) -> None:
    ledger, root, parent = _matched_parent(tmp_path, harness)
    finetune = _continue(root, ledger, parent, "finetune")
    statuses = {finetune.run_id: "continued"}
    done = enrichment.apply_continuations(ledger, run_dir_of(root), evaluator=harness.continuation, status_of=statuses.get)
    assert done == [parent.run_id]
    row = rows_of(ledger)[parent.run_id]
    assert row.gap_finetune == pytest.approx(30.0 - 25.1) and row.gap_transfer is None  # transfer not queued yet
    record = supplement.read("continuation", parent.run_id, ledger_path=ledger, part="finetune")
    assert record.continuation_run_id == finetune.run_id and record.parent_step == row.matched_checkpoint_step
    assert record.task == "SafetyPointGoal1-v0" and record.gap == row.gap_finetune
    entry = log_entries(ledger)[-1]
    assert entry["continuations"][parent.run_id]["finetune"]["run_id"] == finetune.run_id
    assert "not queued" in entry["not_ready"][f"{parent.arm_id}-transfer-s0"]
    transfer = _continue(root, ledger, parent, "transfer")
    statuses[transfer.run_id] = "training"
    assert enrichment.apply_continuations(ledger, run_dir_of(root), evaluator=harness.continuation,
                                          status_of=statuses.get) == []
    statuses[transfer.run_id] = "continued"
    assert enrichment.apply_continuations(ledger, run_dir_of(root), evaluator=harness.continuation,
                                          status_of=statuses.get) == [parent.run_id]
    record = supplement.read("continuation", parent.run_id, ledger_path=ledger, part="transfer")
    assert record.task == "SafetyPointButton1-v0"  # Table 2.2: "on the held-out task"


def test_a_continuation_of_another_checkpoint_is_refused(tmp_path, harness, answered) -> None:
    ledger, root, parent = _matched_parent(tmp_path, harness)
    cont = _continue(root, ledger, parent, "finetune")
    checkpoint = omni_of(root / parent.run_id) / "torch_save" / "epoch-60.pt"
    checkpoint.write_bytes(b"another state")  # the parent's matched checkpoint changed after the continuation started
    with pytest.raises(ContractError, match="SHA-256"):
        enrichment.apply_continuations(ledger, run_dir_of(root), ["finetune"], evaluator=harness.continuation)
    spec_path = root / cont.run_id / "spec.json"
    data = json.loads(spec_path.read_text())
    data["params"]["parent_step"] = 1_000_000
    spec_path.write_text(json.dumps(data))
    with pytest.raises(ContractError, match="not the continuation the design builds"):
        enrichment.apply_continuations(ledger, run_dir_of(root), ["finetune"], evaluator=harness.continuation)
    assert rows_of(ledger)[parent.run_id].gap_finetune is None


def test_a_parent_of_another_epoch_length_is_refused(tmp_path, harness, answered) -> None:
    """The ledger maps a step to ``epoch-{step // R.STEPS_PER_EPOCH}.pt``; a parent of 2,000-step epochs saved the
    matched step under another name, which is never hashed in place of the one the ledger verified."""
    ledger, root, parent = _matched_parent(tmp_path, harness)
    cont = _continue(root, ledger, parent, "finetune")
    config = {"algo_cfgs": {"steps_per_epoch": 2_000}}
    (omni_of(root / parent.run_id) / "config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ContractError, match=f"{parent.run_id}: its epochs are 2000 steps"):
        enrichment.apply_continuations(ledger, run_dir_of(root), ["finetune"], evaluator=harness.continuation,
                                       status_of={cont.run_id: "continued"}.get)
    assert rows_of(ledger)[parent.run_id].gap_finetune is None


def test_continuations_are_for_matched_rows_of_the_main_study(tmp_path, harness, answered) -> None:
    ledger, root = selected(tmp_path, [study_a_spec(0.0, 0, pilot=True)])
    with pytest.raises(ContractError, match="pilot"):
        enrichment.apply_continuations(ledger, run_dir_of(root), evaluator=harness.continuation)
    ledger2, root2 = selected(tmp_path / "main", [study_a_spec(0.0, 0)])
    assert enrichment.apply_continuations(ledger2, run_dir_of(root2), evaluator=harness.continuation) == []
    assert harness.calls == []


@pytest.mark.parametrize("markers", [{"allow_pending": True}, {"allow_dirty": True, "worktree_dirty": True},
                                     {"commit_hash": "0" * 7}])
def test_a_smoke_continuation_is_never_read_into_a_registered_row(tmp_path, harness, answered, markers) -> None:
    """A continuation trained with --allow-pending (skipping Q-continuations / Q-transfer-obs) or with uncommitted
    code, or without a full commit hash, was read into the parent's row, whose gap is written once. It is refused as
    write_run and dependencies.resolve refuse smoke runs; into a smoke parent's row it is read."""
    ledger, root, parent = _matched_parent(tmp_path, harness)
    cont = _continue(root, ledger, parent, "finetune")
    path = root / cont.run_id / "train_result.json"
    clean = json.loads(path.read_text())
    path.write_text(json.dumps({**clean, **markers}))
    with pytest.raises(ContractError, match="smoke run|full commit hash"):
        enrichment.apply_continuations(ledger, run_dir_of(root), ["finetune"], evaluator=harness.continuation,
                                       status_of=None)
    assert rows_of(ledger)[parent.run_id].gap_finetune is None
    if "commit_hash" in markers:
        return
    parent_result = root / parent.run_id / "train_result.json"  # a smoke parent (a smoke ledger's row)
    parent_result.write_text(json.dumps({**json.loads(parent_result.read_text()), "allow_dirty": True}))
    assert enrichment.apply_continuations(ledger, run_dir_of(root), ["finetune"], evaluator=harness.continuation,
                                          status_of=None) == [parent.run_id]


# ---------------------------------------------------------------------------
# Study B: zero-shot and few-shot (Table 2.5)
# ---------------------------------------------------------------------------


def test_zero_shot_writes_sr_zero_and_its_record(tmp_path, harness, monkeypatch) -> None:
    specs = [study_b_spec("Moderate"), study_b_spec("Continuous")]
    paths, root = ledger_with(tmp_path, specs)
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset())  # the keys open, whatever the repository answers
    with pytest.raises(PendingQuestionError, match="Q-studyb-eval"):  # the harness's gate, before anything
        enrichment.apply_zero_shot(paths.main, run_dir_of(root), evaluator=harness.zero_shot)
    assert harness.calls == [] and not supplement.exists("zero_shot", "B-Moderate-s0", ledger_path=paths.main)
    with errors.allow_pending():  # smoke tests only
        done = enrichment.apply_zero_shot(paths.main, run_dir_of(root), evaluator=harness.zero_shot)
    assert done == [s.run_id for s in specs]
    row = rows_of(paths.main)["B-Moderate-s0"]
    assert row.sr_zero == {5.0: 0.6, 15.0: 0.6, 30.0: 0.6, 45.0: 0.6}
    record = supplement.read("zero_shot", "B-Continuous-s0", ledger_path=paths.main)
    assert sorted(b.budget for b in record.budgets if b.role == "reference") == [10.0, 40.0]
    assert enrichment.apply_zero_shot(paths.main, run_dir_of(root), evaluator=harness.zero_shot) == []  # nothing left


def test_zero_shot_refuses_rates_that_are_not_equation_10(tmp_path, harness, answered) -> None:
    paths, root = ledger_with(tmp_path, [study_b_spec("Moderate")])

    def inflated(*args):
        result = harness.zero_shot(*args)
        result["satisfaction"][5.0] = 0.9
        result["sr_zero"][5.0] = 0.9
        return result

    with pytest.raises(ContractError, match="not a valid zero_shot supplement record"):
        enrichment.apply_zero_shot(paths.main, run_dir_of(root), evaluator=inflated)
    assert rows_of(paths.main)["B-Moderate-s0"].sr_zero is None


def _fewshot_runs(root: Path, parent: RunSpec, budgets=R.UNSEEN_BUDGETS, cut: bool = False) -> list[RunSpec]:
    specs = [s for s in manifest.study_b_fewshot([parent]) if s.params["budget"] in budgets]
    if cut:
        specs = manifest.apply_cuts(specs, manifest.CUTS[:3])
    for spec in specs:
        make_run(root, spec)
    return specs


def test_fewshot_writes_sr_fewshot_and_adapt_steps_per_budget(tmp_path, harness) -> None:
    parent = study_b_spec("Moderate")
    paths, root = ledger_with(tmp_path, [parent])
    conts = _fewshot_runs(root, parent, budgets=(5.0, 15.0))
    with errors.allow_pending():
        done = enrichment.apply_fewshot(paths.main, run_dir_of(root), evaluator=harness.fewshot)
    assert done == [parent.run_id] and len(harness.calls) == 2
    row = rows_of(paths.main)[parent.run_id]
    assert row.sr_fewshot == {fewshot_key(b, h): r for b in (5.0, 15.0) for h, r in harness.rates[b].items()}
    assert row.adapt_steps == {5.0: 500_000, 15.0: 500_000}
    # the frozen schema test's key form (tests/test_ledger_schema.py; pilot.contracts.fewshot_key)
    assert "5.0_200000" in row.sr_fewshot
    record = supplement.read("fewshot", parent.run_id, ledger_path=paths.main, part="b5")
    assert record.continuation_run_id == conts[0].run_id and record.parent_step == parent.total_steps
    assert set(log_entries(paths.main)[-1]["not_ready"]) == {"B-Moderate-fewshot-b30-s0", "B-Moderate-fewshot-b45-s0"}
    # the other two budgets later: keys are added, the existing ones kept
    _fewshot_runs(root, parent, budgets=(30.0, 45.0))
    with errors.allow_pending():
        assert enrichment.apply_fewshot(paths.main, run_dir_of(root), evaluator=harness.fewshot) == [parent.run_id]
    row = rows_of(paths.main)[parent.run_id]
    assert len(row.sr_fewshot) == 12 and sorted(row.adapt_steps) == [5.0, 15.0, 30.0, 45.0]


def test_fewshot_censoring_waits_on_its_question_only_when_censored(tmp_path, harness, monkeypatch) -> None:
    parent = study_b_spec("Moderate")
    paths, root = ledger_with(tmp_path, [parent])
    _fewshot_runs(root, parent, budgets=(5.0, 15.0))
    harness.rates[5.0] = {200_000: 0.1, 500_000: 0.2, 1_000_000: 0.79}  # never reaches 0.80
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset({"Q-studyb-eval"}))
    with pytest.raises(PendingQuestionError, match="Q-adapt-censoring"):
        enrichment.apply_fewshot(paths.main, run_dir_of(root), evaluator=harness.fewshot)
    row = rows_of(paths.main)[parent.run_id]
    assert row.adapt_steps == {15.0: 500_000}  # the budget that reached the target is written
    assert not supplement.exists("fewshot", parent.run_id, ledger_path=paths.main, part="b5")
    # The key "Gates adaptation steps only": the censored budget's rates (eq. 10) are written all the same
    rates = {fewshot_key(5.0, h): r for h, r in harness.rates[5.0].items()}
    assert {k: v for k, v in row.sr_fewshot.items() if k.startswith("5.0_")} == rates
    assert 5.0 not in row.adapt_steps
    from analysis.data import load_dataset

    assert load_dataset(paths.main, mode="final", cuts=()).problems == []  # rates without adapt_steps read cleanly
    # a second call while the key is open writes nothing new and still refuses
    before = paths.main.read_bytes()
    with pytest.raises(PendingQuestionError, match="Q-adapt-censoring"):
        enrichment.apply_fewshot(paths.main, run_dir_of(root), evaluator=harness.fewshot)
    assert paths.main.read_bytes() == before
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset({"Q-studyb-eval", "Q-adapt-censoring"}))
    assert enrichment.apply_fewshot(paths.main, run_dir_of(root), evaluator=harness.fewshot) == [parent.run_id]
    row = rows_of(paths.main)[parent.run_id]
    assert row.adapt_steps[5.0] == 1_000_001  # the largest horizon + 1
    assert {k: v for k, v in row.sr_fewshot.items() if k.startswith("5.0_")} == rates  # the rates reproduced, unchanged
    assert supplement.read("fewshot", parent.run_id, ledger_path=paths.main, part="b5").adapt_steps == 1_000_001


def test_fewshot_reads_the_horizons_of_a_cut_continuation(tmp_path, harness) -> None:
    """Part 6.1 cut 3 leaves one horizon: its censored value is 200,001, read from the continuation's own spec."""
    parent = study_b_spec("Sparse")
    paths, root = ledger_with(tmp_path, [parent])
    conts = _fewshot_runs(root, parent, budgets=(45.0,), cut=True)
    assert conts[0].params["horizons"] == [200_000] and conts[0].total_steps == 200_000
    harness.rates[45.0] = {200_000: 0.5}
    with errors.allow_pending():
        enrichment.apply_fewshot(paths.main, run_dir_of(root), evaluator=harness.fewshot)
    row = rows_of(paths.main)[parent.run_id]
    assert row.sr_fewshot == {"45.0_200000": 0.5} and row.adapt_steps == {45.0: 200_001}
    assert enrichment.adaptation_steps({200_000: 0.8}, [200_000]) == (200_000, False)


def test_fewshot_refuses_a_result_that_breaks_table_2_5(tmp_path, harness) -> None:
    parent = study_b_spec("Moderate")
    paths, root = ledger_with(tmp_path, [parent])
    _fewshot_runs(root, parent, budgets=(5.0,))

    def wrong_steps(*args):
        result = harness.fewshot(*args)
        result["adapt_steps"] = {5.0: 200_000}
        return result

    with errors.allow_pending(), pytest.raises(ContractError, match="adapt_steps"):
        enrichment.apply_fewshot(paths.main, run_dir_of(root), evaluator=wrong_steps)
    assert rows_of(paths.main)[parent.run_id].sr_fewshot is None


def test_study_b_results_off_the_measurement_set_are_refused(tmp_path, harness, answered) -> None:
    """Q-studyb-eval (Table 9.1): "Every Study B evaluation with a budget in the observation uses Table 2.1's
    measurement set", zero-shot and few-shot alike; a result on the selection set is refused, writing nothing."""
    parent = study_b_spec("Moderate")
    paths, root = ledger_with(tmp_path, [parent])
    _fewshot_runs(root, parent, budgets=(5.0,))

    def on_selection(evaluate: Callable[..., dict]) -> Callable[..., dict]:
        return lambda *args: {**evaluate(*args), "seed_set": "selection", "seeds": SEL}

    with pytest.raises(ContractError, match="seed_set 'selection' is not the measurement set"):
        enrichment.apply_zero_shot(paths.main, run_dir_of(root), evaluator=on_selection(harness.zero_shot))
    with pytest.raises(ContractError, match="seed_set 'selection' is not the measurement set"):
        enrichment.apply_fewshot(paths.main, run_dir_of(root), evaluator=on_selection(harness.fewshot))
    row = rows_of(paths.main)[parent.run_id]
    assert row.sr_zero is None and row.sr_fewshot is None and row.adapt_steps is None
    assert not supplement.exists("zero_shot", parent.run_id, ledger_path=paths.main)


# ---------------------------------------------------------------------------
# Part 4.1 rule 7: final-battery (b) and sensitivity-battery (a)
# ---------------------------------------------------------------------------


def test_final_battery_measures_the_final_checkpoint(tmp_path, harness, answered) -> None:
    spec = study_a_spec(0.0, 0)
    paths, root = ledger_with(tmp_path, [spec])
    # the final checkpoint's measurement-set cost is the ledger's final_cost (24.0; Harness.final)
    assert enrichment.apply_final_battery(paths.main, run_dir_of(root), evaluator=harness.battery) == [spec.run_id]
    assert harness.calls == [("battery", spec.run_id, TOTAL, ("hazard",))]
    measurement = supplement.read("final_battery", spec.run_id, ledger_path=paths.main, part="measurement")
    hazard = supplement.read("final_battery", spec.run_id, ledger_path=paths.main, part="hazard")
    assert measurement.step == hazard.step == TOTAL and hazard.gap == pytest.approx(6.0)
    # dynamics later (exploratory): the measurement must reproduce, and no ledger field changes
    before = paths.main.read_bytes()
    assert enrichment.apply_final_battery(paths.main, run_dir_of(root), ["hazard", "dynamics"],
                                          evaluator=harness.battery) == [spec.run_id]
    assert harness.calls[-1][3] == ("dynamics",)
    assert paths.main.read_bytes() == before
    harness.final[spec.run_id] = 23.0
    other, root2 = ledger_with(tmp_path / "2", [spec])
    with pytest.raises(ContractError, match="final_cost"):
        enrichment.apply_final_battery(other.main, run_dir_of(root2), evaluator=harness.battery)


def test_sensitivity_battery_for_arms_matched_at_5_but_not_2_5(tmp_path, harness, answered) -> None:
    costs = {0.0: [24.0, 25.0, 26.0, 25.0, 25.0], 0.5: [27.0, 28.0, 27.5, 28.5, 27.0]}
    ledger, root, specs = _measured_arms(tmp_path, harness, costs)
    enrichment.apply_matching(ledger, surplus_extra=0)
    done = enrichment.apply_sensitivity_battery(ledger, run_dir_of(root), surplus_extra=0, evaluator=harness.battery)
    late = sorted(s.run_id for s in specs.values() if s.N == 0.5)
    assert sorted(done) == late  # the reference is matched at 2.5 already: its gaps are the battery's
    for run_id in late:
        record = supplement.read("sensitivity_battery", run_id, ledger_path=ledger, part="hazard")
        assert record.gap == pytest.approx(6.0)
        assert rows_of(ledger)[run_id].gap_hazard is None  # rule 6: never a gap in an unmatched row
    assert enrichment.apply_sensitivity_battery(ledger, run_dir_of(root), surplus_extra=0, evaluator=harness.battery) == []


def test_sensitivity_battery_waits_for_the_match(tmp_path, harness, answered) -> None:
    costs = {0.0: [25.0] * 5, 0.5: [28.0] * 5}
    ledger, root, _ = _measured_arms(tmp_path, harness, costs)
    assert enrichment.apply_sensitivity_battery(ledger, run_dir_of(root), surplus_extra=0, evaluator=harness.battery) == []
    waiting = log_entries(ledger)[-1]["arms_waiting"]
    assert waiting["A-PointGoal1-N0.00"] == "no flags yet (`enrich match`)"  # the reference names no other arm
    assert waiting["A-PointGoal1-N0.50-abrupt-total"] == ("the reference arm A-PointGoal1-N0.00 has no flags yet "
                                                          "(`enrich match`)")


def test_sensitivity_battery_names_the_arms_the_threshold_arithmetic_decides(tmp_path, harness, monkeypatch) -> None:
    """Rule 7 (a) is gated only by the conditions' own keys (pilot/enrichment.py): an arm on the 5.0 boundary is
    evaluated under the answer (inclusive) while Q-threshold-arithmetic is open, and named as provisional."""
    costs = {0.0: [25.0] * 5, 0.5: [30.0] * 5}  # |30 - 25| = 5.0: matched at 5.0 only by an inclusive reading
    ledger, root, specs = _measured_arms(tmp_path, harness, costs)
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset({"Q-matched-cost-set", "Q-arm-complete", "Q-hazard"}))
    first = enrichment.apply_matching(ledger, surplus_extra=0)
    assert len(first) == 10 and first.provisional == {}  # at 2.5 every reading agrees: 5.0 is unmatched
    done = enrichment.apply_sensitivity_battery(ledger, run_dir_of(root), surplus_extra=0, evaluator=harness.battery)
    late = "A-PointGoal1-N0.50-abrupt-total"
    assert sorted(done) == sorted(s.run_id for s in specs.values() if s.N == 0.5)
    assert list(done.provisional) == [late] and "tolerance 5: written under the answer" in done.provisional[late]
    entry = log_entries(ledger)[-1]
    assert entry["provisional"] == done.provisional and "waiting" not in entry


def test_the_log_names_the_commit_the_records_name_when_head_moves(tmp_path, harness, answered, monkeypatch) -> None:
    """The records carried HEAD at the start of the command, the log HEAD at its end; a commit landing during an
    hours-long battery made them name different commits. The log names the start commit (the one that computed the
    records) and adds the HEAD it found at the end."""
    costs = {0.0: [24.0, 25.0, 26.0, 25.0, 25.0], 0.5: [27.0, 28.0, 27.5, 28.5, 27.0]}
    ledger, root, specs = _measured_arms(tmp_path, harness, costs)
    enrichment.apply_matching(ledger, surplus_extra=0)
    head = {"value": "a" * 40}
    monkeypatch.setattr(provenance, "commit_hash", lambda *a, **k: head["value"])

    def evaluator(*args, **kw):
        out = harness.battery(*args, **kw)
        head["value"] = "b" * 40  # a commit lands while the battery runs
        return out

    done = enrichment.apply_battery(ledger, run_dir_of(root), ["hazard"], evaluator=evaluator)
    record = supplement.read("battery", done[0], ledger_path=ledger, part="hazard")
    entry = log_entries(ledger)[-1]
    assert record.code_commit == entry["commit"] == "a" * 40 and entry["head_at_end"] == "b" * 40


def test_the_command_line_refuses_code_changed_by_a_commit_landing_mid_command(monkeypatch) -> None:
    """``unverified_imported_code`` compares the tree with the new HEAD only, so a code commit landing during a
    command passed it. Like the scheduler's ``_ledger_code_problem``, a moved HEAD that changes loaded code is
    refused; one that changes only other files (a result) is not."""
    import pilot.__main__ as cli

    monkeypatch.setattr(provenance, "commit_hash", lambda *a, **k: "b" * 40)
    cli._require_code_of("b" * 40, "x")  # HEAD did not move
    monkeypatch.setattr(provenance, "imported_code_changed_between", lambda old, new: [])
    cli._require_code_of("a" * 40, "x")  # it moved, but only a result was committed
    monkeypatch.setattr(provenance, "imported_code_changed_between", lambda old, new: ["pilot/enrichment.py"])
    with pytest.raises(cli.CliError, match=r"HEAD moved from a{40} to b{40} while the enrichment ran.*pilot/enrichment"):
        cli._require_code_of("a" * 40, "the enrichment")


def test_load_classifies_a_harness_importing_missing_code_as_the_launcher_does(monkeypatch, capsys) -> None:
    """A harness that imports another role's module not written yet is unavailable (PluginUnavailableError), as
    ``pilot.algorithms.import_provided`` says for the launcher (HANDOVER.md section 8; pilot/launch.py), not a bare
    ModuleNotFoundError with a traceback; a missing library still shows its real error."""
    import importlib

    import pilot.__main__ as cli
    from pilot.algorithms import PluginUnavailableError

    real = importlib.import_module
    missing = {"name": "envs.not_written_yet"}

    def fake(name, *a, **k):
        if name == "studyb.evaluation":
            raise ModuleNotFoundError(f"No module named {missing['name']!r}", name=missing["name"])
        return real(name, *a, **k)

    monkeypatch.setattr(importlib, "import_module", fake)
    with pytest.raises(PluginUnavailableError, match="envs.not_written_yet"):
        enrichment._load(enrichment.ZERO_SHOT_TARGET)
    missing["name"] = "a_library_not_installed"
    with pytest.raises(ModuleNotFoundError):
        enrichment._load(enrichment.ZERO_SHOT_TARGET)

    def lazily(args):  # an import inside a harness call
        raise ModuleNotFoundError("No module named 'metrics.not_written_yet'", name="metrics.not_written_yet")

    monkeypatch.setattr(cli, "_cmd_design", lazily)
    assert cli.main(["design", "pilot"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("unavailable:") and "metrics.not_written_yet" in err and "Traceback" not in err


def test_battery_commands_log_the_recorded_definitions(tmp_path, harness, answered) -> None:
    """HANDOVER.md section 8, contract 5's recorded definitions: what was relocated or scaled (its names differ by
    robot; Q-dynamics) has no supplement field, so the enrichment log keeps each distinct one once."""
    costs = {0.0: [24.0, 25.0, 26.0, 25.0, 25.0], 0.5: [27.0, 28.0, 27.5, 28.5, 27.0]}
    ledger, root, specs = _measured_arms(tmp_path, harness, costs)
    enrichment.apply_matching(ledger, surplus_extra=0)
    enrichment.apply_battery(ledger, run_dir_of(root), ["hazard", "dynamics"], evaluator=harness.battery)
    reference = sorted(s.run_id for s in specs.values() if s.N == 0.0)  # matched: the battery's (rule 6)
    entry = log_entries(ledger)[-1]
    assert entry["action"] == "battery"
    by_condition = {d["condition"]: d for d in entry["definitions"]}
    assert set(by_condition) == {"hazard", "dynamics"} and len(entry["definitions"]) == 2  # one task: one each
    dynamics = by_condition["dynamics"]
    assert dynamics["definition"]["bodies"] == ["agent"] and dynamics["definition"]["friction_geoms"] == ["floor", "agent"]
    assert dynamics["run_ids"] == reference
    assert by_condition["hazard"]["definition"]["placements"] == [[-0.75, -0.75, 0.75, 0.75]]
    enrichment.apply_final_battery(ledger, run_dir_of(root), ["hazard", "dynamics"], evaluator=harness.battery)
    final = log_entries(ledger)[-1]
    assert final["action"] == "final-battery" and {d["condition"] for d in final["definitions"]} == {"hazard", "dynamics"}
    assert sorted(next(d for d in final["definitions"] if d["condition"] == "dynamics")["run_ids"]) == sorted(specs)
    enrichment.apply_sensitivity_battery(ledger, run_dir_of(root), ["dynamics"], surplus_extra=0,
                                         evaluator=harness.battery)
    sens = log_entries(ledger)[-1]
    assert sens["action"] == "sensitivity-battery"
    assert [d["condition"] for d in sens["definitions"]] == ["dynamics"]
    assert sens["definitions"][0]["run_ids"] == sorted(s.run_id for s in specs.values() if s.N == 0.5)


def test_battery_result_without_a_recorded_definition_is_refused(harness, answered) -> None:
    good = harness.battery("dir", {"run_id": "A-x-s0", "total_steps": TOTAL}, 1_200_000, ["hazard", "dynamics"])
    result = enrichment.check_battery_result("A-x-s0", good, step=1_200_000, conditions=["hazard", "dynamics"],
                                             selection_seeds=SEL)
    assert result.definitions()["dynamics"]["bodies"] == ["agent"]
    bare = {**good, "conditions": {"hazard": good["conditions"]["hazard"]}}
    with pytest.raises(ContractError, match="no recorded definition of 'dynamics'"):
        enrichment.check_battery_result("A-x-s0", bare, step=1_200_000, conditions=["hazard", "dynamics"],
                                        selection_seeds=SEL)


# ---------------------------------------------------------------------------
# Command line: enrich, completeness, freeze
# ---------------------------------------------------------------------------


def test_cli_enrich_flags_and_clean_refusals(tmp_path, capsys) -> None:
    from pilot.__main__ import main

    ledger = tmp_path / "ledger.parquet"
    assert main(["enrich", "select", "--ledger", str(ledger), "--allow-pending"]) == 2
    assert "needs --allow-dirty" in capsys.readouterr().err
    repo_ledger = provenance.REPO_ROOT / "results" / "smoke-test-ledger.parquet"
    assert main(["enrich", "select", "--ledger", str(repo_ledger), "--allow-dirty", "--allow-pending"]) == 2
    assert "smoke ledgers only" in capsys.readouterr().err
    assert main(["enrich", "match", "--ledger", str(ledger), "--allow-dirty"]) == 2
    assert "needs --surplus-extra" in capsys.readouterr().err
    assert main(["enrich", "select", "--ledger", str(ledger), "--allow-dirty", "--conditions", "hazard"]) == 2
    assert "takes no --conditions" in capsys.readouterr().err
    assert main(["enrich", "battery", "--ledger", str(ledger), "--allow-dirty", "--conditions", "finetune"]) == 2
    assert "must be distinct values" in capsys.readouterr().err
    assert main(["enrich", "measure", "--ledger", str(ledger), "--allow-dirty"]) == 2  # no such ledger
    assert "does not exist" in capsys.readouterr().err


def test_cli_enrich_turns_an_open_question_into_a_clean_refusal(tmp_path, capsys, monkeypatch) -> None:
    from pilot.__main__ import main

    paths, root = ledger_with(tmp_path, [study_a_spec(0.0, s) for s in range(5)])
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset())  # the keys open, whatever the repository answers
    argv = ["enrich", "match", "--ledger", str(paths.main), "--allow-dirty", "--surplus-extra", "0",
            "--data-root", str(smoke_root(tmp_path, paths))]
    assert main(argv) == 3
    err = capsys.readouterr().err
    assert err.startswith("refused: PendingQuestionError") and "Q-matched-cost-set" in err and "Traceback" not in err
    assert errors.pending_allowed() is False


def test_cli_allow_pending_opens_the_gates_for_the_command_only(tmp_path, monkeypatch, capsys) -> None:
    from pilot.__main__ import main

    tie = (40, 40, 40, 40, 40, 40, 40, 40, 26, 24)
    paths = paths_in(tmp_path)
    root = tmp_path / "data" / "checkpoints"
    spec = study_a_spec(0.0, 0)
    write_run(spec, make_run(root, spec, costs=tie), paths)
    seen = []
    real = enrichment.apply_selection

    def spy(*args, **kwargs):
        seen.append(errors.pending_allowed())
        return real(*args, selector=closest_to_25, **{k: v for k, v in kwargs.items() if k != "selector"})

    monkeypatch.setattr(enrichment, "apply_selection", spy)
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset())  # the keys open, whatever the repository answers
    common = ["--ledger", str(paths.main), "--data-root", str(smoke_root(tmp_path, paths)), "--allow-dirty"]
    assert main(["enrich", "select", *common]) == 3  # the tie waits
    assert main(["enrich", "select", *common, "--allow-pending"]) == 0
    assert seen == [False, True] and errors.pending_allowed() is False
    assert "select: 1 runs enriched" in capsys.readouterr().out


def test_cli_smoke_flags_are_refused_on_a_registered_data_root(tmp_path, harness, monkeypatch, capsys) -> None:
    """--allow-pending only together with --allow-dirty, on smoke ledgers only (pilot/__main__.py; HANDOVER.md
    section 7). A registered data root whose ledgers lie outside the repository (``schedule run --ledger-dir``) is
    not a smoke root: nothing is written under open questions or uncommitted code there, where a value is never
    changed afterwards."""
    from envs import evaluation as E

    from pilot.__main__ import main
    from pilot.ledger_writer import ledger_mode
    from pilot.scheduler import Scheduler, SchedulerConfig

    data = tmp_path / "data"
    paths = LedgerPaths(main=tmp_path / "L" / "ledger.parquet", pilot=tmp_path / "L" / "pilot_ledger.parquet",
                        sidecar_dir=tmp_path / "L" / "sidecar")
    sched = Scheduler(SchedulerConfig(data_root=data, ledger_paths=paths))  # no smoke flags: registered
    sched.fix_mode()  # as `schedule run --ledger-dir L` does on its first pass
    assert sched.setting("mode") == "registered"
    sched.db.close()
    spec = study_a_spec(0.0, 0, pilot=True)
    write_run(spec, make_run(data / "checkpoints", spec), paths)
    enrichment.apply_selection(paths.pilot, selector=closest_to_25)
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset())  # the keys open, whatever the repository answers
    assert R.is_open("Q-hazard") and R.is_open("Q-dynamics")
    common = ["enrich", "battery", "--ledger", str(paths.pilot), "--data-root", str(data)]
    monkeypatch.setattr(E, "evaluate_battery", harness.battery)
    assert main([*common, "--allow-dirty", "--allow-pending"]) == 2
    assert "is a registered ledger" in capsys.readouterr().err  # its mode marker is read first
    assert main([*common, "--allow-dirty"]) == 2  # uncommitted code on a registered ledger: refused too
    assert "is a registered ledger" in capsys.readouterr().err
    # a ledger without a mode marker beside the registered data root: the data root's own mode refuses it
    unmarked = tmp_path / "copy" / "pilot_ledger.parquet"
    unmarked.parent.mkdir()
    shutil.copyfile(paths.pilot, unmarked)
    assert ledger_mode(unmarked) is None
    assert main(["enrich", "battery", "--ledger", str(unmarked), "--data-root", str(data), "--allow-dirty",
                 "--allow-pending"]) == 2
    err = capsys.readouterr().err
    assert "for smoke data roots only" in err and "holds registered runs" in err
    # the registered ledger named beside a smoke data root: its marker refuses it whatever data root is named
    smoke = smoke_root(tmp_path / "smoke")
    assert main(["enrich", "select", "--ledger", str(paths.pilot), "--data-root", str(smoke), "--allow-dirty",
                 "--allow-pending"]) == 2
    assert "is a registered ledger" in capsys.readouterr().err
    # an unmarked ledger named beside a smoke data root is not one of that root's ledgers
    assert main(["enrich", "select", "--ledger", str(unmarked), "--data-root", str(smoke), "--allow-dirty",
                 "--allow-pending"]) == 2
    assert "is not a ledger of the smoke data root" in capsys.readouterr().err
    row = rows_of(paths.pilot)[spec.run_id]
    assert row.gap_hazard is None and row.gap_dynamics is None and harness.calls == []
    assert [e["action"] for e in log_entries(paths.pilot)] == ["select"]  # the test's own selection only


def test_cli_smoke_flags_take_the_smoke_ledgers_of_a_data_root_without_a_mode(tmp_path, monkeypatch, capsys) -> None:
    """A data root no scheduler has fixed a mode for: only ``<data-root>/smoke`` (ledger_writer.smoke_paths) is smoke."""
    from pilot.__main__ import main
    from pilot.ledger_writer import smoke_paths

    data = tmp_path / "data"
    elsewhere, _ = ledger_with(tmp_path, [study_a_spec(0.0, 0)])
    assert main(["enrich", "select", "--ledger", str(elsewhere.main), "--data-root", str(data), "--allow-dirty"]) == 2
    assert "which has no mode yet" in capsys.readouterr().err
    paths = smoke_paths(data)
    spec = study_a_spec(0.0, 1)
    write_run(spec, make_run(data / "checkpoints", spec), paths)
    real = enrichment.apply_selection
    monkeypatch.setattr(enrichment, "apply_selection", lambda ledger, **kw: real(ledger, selector=closest_to_25))
    assert main(["enrich", "select", "--ledger", str(paths.main), "--data-root", str(data), "--allow-dirty"]) == 0
    assert rows_of(paths.main)[spec.run_id].matched_checkpoint_step is not None


def test_cli_match_prints_the_flags_an_open_report_key_decides(tmp_path, harness, monkeypatch, capsys) -> None:
    from pilot.__main__ import main

    costs = {0.0: [25.0] * 5, 0.5: [27.5] * 5}  # the 2.5 boundary: the answer (inclusive) matches it
    ledger, root, _ = _measured_arms(tmp_path, harness, costs)
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset({"Q-matched-cost-set", "Q-arm-complete"}))
    capsys.readouterr()
    assert main(["enrich", "match", "--surplus-extra", "0", "--ledger", str(ledger), "--data-root",
                 str(smoke_root(tmp_path)), "--allow-dirty"]) == 0
    out = capsys.readouterr().out
    assert "match: 10 runs enriched" in out and "match: 1 computed under the proposal of an open report key" in out
    assert "  A-PointGoal1-N0.50-abrupt-total: task SafetyPointGoal1-v0, tolerance 2.5" in out


def test_cli_enrich_says_what_waits_for_its_data(tmp_path, harness, answered, capsys) -> None:
    """`enrich match` leaves an incomplete arm waiting and says so on the command line, not only in the log
    (HANDOVER.md section 7, the exit status of `python -m pilot`)."""
    from pilot.__main__ import main

    specs = [study_a_spec(0.0, s) for s in range(5)] + [study_a_spec(0.5, s) for s in range(4)]
    ledger, root = selected(tmp_path, specs)
    done = enrichment.apply_measurement(ledger, run_dir_of(root), evaluator=harness.battery)
    assert len(done) == 9 and done.waiting == {}
    capsys.readouterr()
    common = ["--ledger", str(ledger), "--data-root", str(smoke_root(tmp_path)), "--allow-dirty"]
    assert main(["enrich", "match", "--surplus-extra", "0", *common]) == 0
    out = capsys.readouterr().out
    assert "match: 5 runs enriched" in out and "match: 1 not written, waiting for their data:" in out
    assert "  A-PointGoal1-N0.50-abrupt-total: 4 of its 5 seeds completed (Q-arm-complete)" in out
    assert main(["enrich", "match", "--surplus-extra", "0", *common]) == 0  # still waiting: said again
    assert "match: 0 runs enriched" in (out := capsys.readouterr().out) and "4 of its 5 seeds" in out
    # the battery of the main study: the late arm's rows have no flag yet (rule 6)
    done = enrichment.apply_battery(ledger, run_dir_of(root), ["hazard"], evaluator=harness.battery)
    assert len(done) == 5 and set(done.waiting) == {s.run_id for s in specs[5:]}
    assert all("enrich match" in why for why in done.waiting.values())


def test_cli_reports_a_harness_instability_and_an_unreadable_state_without_a_traceback(tmp_path, answered, monkeypatch,
                                                                                       capsys) -> None:
    from envs import evaluation as E

    from pilot.__main__ import main
    from pilot.scheduler import SchedulerConfig

    ledger, root = selected(tmp_path, [study_a_spec(0.0, 0, pilot=True)])
    data = smoke_root(tmp_path)

    def unstable(omnisafe_dir, spec, step, conditions):
        raise E.MujocoInstabilityError("MuJoCo warning BADQACC at step 412 (dynamics)")

    monkeypatch.setattr(E, "evaluate_battery", unstable)
    assert main(["enrich", "battery", "--ledger", str(ledger), "--data-root", str(data), "--allow-dirty"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: MujocoInstabilityError: MuJoCo warning BADQACC") and "report it to the group" in err
    # an unexpected RuntimeError keeps its traceback (it is a bug, not an expected failure)
    monkeypatch.setattr(E, "evaluate_battery", lambda *a: (_ for _ in ()).throw(RuntimeError("a bug")))
    with pytest.raises(RuntimeError, match="a bug"):
        main(["enrich", "battery", "--ledger", str(ledger), "--data-root", str(data), "--allow-dirty"])
    # a data root whose scheduler state is not a database (e.g. a truncated copy)
    state = SchedulerConfig(data_root=data).state_path
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_bytes(b"not a database, just some bytes" * 10)
    for argv in (["enrich", "match", "--surplus-extra", "0", "--ledger", str(ledger), "--allow-dirty"],
                 ["completeness", "--ledger", str(ledger)], ["freeze", "--ledger", str(ledger)]):
        assert main([*argv, "--data-root", str(data)]) == 1
        err = capsys.readouterr().err
        assert err.startswith("error: ContractError: cannot read the scheduler state") and str(state) in err


def test_cli_completeness_refuses_a_ledger_that_does_not_exist(tmp_path, capsys) -> None:
    """A mistyped --ledger printed `0 rows ... missing: 0 enrichment fields` and exited 0."""
    from pilot.__main__ import main

    missing = tmp_path / "none.parquet"
    assert main(["completeness", "--ledger", str(missing), "--data-root", str(tmp_path / "root")]) == 2
    assert f"the ledger {missing} does not exist" in capsys.readouterr().err
    assert not missing.exists() and not (tmp_path / "root").exists()


def test_cli_completeness_and_freeze_are_read_only_reports(tmp_path, answered, capsys) -> None:
    from pilot.__main__ import main

    ledger, root = selected(tmp_path, [study_a_spec(0.0, 0), study_b_spec("Moderate")])
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    ledger_bytes = ledger.read_bytes()
    assert main(["completeness", "--ledger", str(ledger), "--data-root", str(tmp_path / "data"), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    a = report["arms"]["A-PointGoal1-N0.00"]
    assert a["fields_missing"]["measurement_cost"] == ["A-PointGoal1-N0.00-s0"] and "matched" in a["fields_missing"]
    assert a["records_missing"]["training"] == ["A-PointGoal1-N0.00-s0"] and "evaluation" not in a["records_missing"]
    b = report["arms"]["B-Moderate"]
    assert sorted(b["fields_missing"]) == ["adapt_steps", "sr_fewshot", "sr_zero"]
    assert "fewshot/b45" in b["records_missing"] and report["scheduler_state"] is False
    assert main(["completeness", "--ledger", str(ledger), "--data-root", str(tmp_path / "data")]) == 0
    assert "field measurement_cost" in capsys.readouterr().out
    assert main(["freeze", "--ledger", str(ledger), "--data-root", str(tmp_path / "data"), "--json"]) == 0
    manifest_ = json.loads(capsys.readouterr().out)
    assert manifest_["ready"] is False and manifest_["files"][str(ledger)] == enrichment._sha256(ledger)
    assert any("supplement records are missing" in p for p in manifest_["problems"])
    assert any(k.endswith("evaluation/A-PointGoal1-N0.00-s0.json") for k in manifest_["files"])
    assert any(k.startswith("analysis/") for k in manifest_["files"])  # the analysis code is hashed
    assert "configs/registered.py" in manifest_["files"]  # with the modules it imports (Q-exploratory-labels)
    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*")) == before and ledger.read_bytes() == ledger_bytes


def test_freeze_lists_the_analysis_changes_since_the_go_report(tmp_path) -> None:
    go = tmp_path / "go"
    go.mkdir()
    head = provenance.commit_hash()
    # the first pilot's report (revision 0, a commit not in this history), then the re-pilot's report (revision 1),
    # which the decision was taken from
    (go / "go_report-20261001T000000Z.json").write_text(json.dumps({"generated": {"commit": head, "revision": 1}}))
    (go / "go_report-20260901T000000Z.json").write_text(json.dumps({"generated": {"commit": "0" * 40}}))
    report = enrichment.freeze_report(tmp_path / "missing.parquet", go_reports=go, analysis_outputs=tmp_path / "none",
                                      read_go_report=enrichment.read_working_file)
    assert report["analysis_changes_since_go"] == [] and report["go_report"]["commit"] == head
    assert [(r["commit"], r["revision"]) for r in report["go_reports"]] == [("0" * 40, 0), (head, 1)]
    assert any("does not exist" in p for p in report["problems"]) and report["ready"] is False


def test_freeze_lists_changes_to_the_modules_the_analysis_imports(tmp_path, monkeypatch) -> None:
    """Q-exploratory-labels (Table 9.1): the frozen analysis code is analysis/ and every in-repository module it
    imports (analysis.records.IMPORTED_CODE); "hashing analysis/*.py alone missed the registered constants and pilot
    modules every verdict reads". An edit of configs/registered.py after the go decision is listed, and hashed."""
    import subprocess

    from analysis.records import IMPORTED_CODE

    assert "configs/registered.py" in IMPORTED_CODE
    repo = tmp_path / "repo"
    (repo / "analysis").mkdir(parents=True)
    (repo / "configs").mkdir()

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()

    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    (repo / "analysis" / "run.py").write_text("x = 1\n")
    (repo / "configs" / "registered.py").write_text("MIN_EFFECT = 1.0\n")
    (repo / "notes.txt").write_text("a\n")
    git("add", ".")
    git("commit", "-qm", "go")
    go = git("rev-parse", "HEAD")
    (repo / "notes.txt").write_text("b\n")
    git("commit", "-qam", "edit a file the analysis does not import")
    (repo / "configs" / "registered.py").write_text("MIN_EFFECT = 2.0\n")
    git("commit", "-qam", "edit the minimum effect after go")
    monkeypatch.setattr(provenance, "REPO_ROOT", repo)
    changes = enrichment.analysis_changes_since(go)
    assert [line.split(" ", 1)[1] for line in changes] == ["edit the minimum effect after go"]
    report = enrichment.freeze_report(tmp_path / "missing.parquet", go_reports=tmp_path / "none",
                                      analysis_outputs=tmp_path / "none", read_go_report=enrichment.read_working_file)
    assert {"analysis/run.py", "configs/registered.py"} <= set(report["files"])


def test_a_go_report_commit_that_is_not_a_hash_never_reaches_git(tmp_path) -> None:
    injected = f"--output={tmp_path / 'injected'}"  # read by git as an option, not a revision
    go = tmp_path / "go"
    go.mkdir()
    (go / "go_report-20261001T000000Z.json").write_text(json.dumps({"generated": {"commit": injected, "revision": 0}}))
    (report,) = enrichment.list_go_reports(go, read=enrichment.read_working_file)
    assert report["commit"] is None and report["readable"] is True  # freeze: "records no commit"
    with pytest.raises(ContractError, match="not a full commit hash"):
        enrichment.analysis_changes_since(injected)
    assert not list(tmp_path.glob("injected*"))


def test_freeze_anchors_the_go_decision_to_the_earliest_committed_report_of_the_revision(tmp_path, monkeypatch) -> None:
    """Freeze read the NEWEST go report under results/pilot, so re-running `go` after an edit to analysis/ (an
    uncommitted report does not make the tree dirty) moved the Part 5.8 baseline past the edit."""
    go = tmp_path / "go"
    go.mkdir()
    first, later = go / "go_report-20270101T000000Z.json", go / "go_report-20270601T000000Z.json"
    first.write_text(json.dumps({"generated": {"commit": "1" * 40, "revision": 0}}))
    later.write_text(json.dumps({"generated": {"commit": "2" * 40, "revision": 0}}))
    seen = []

    def changes(commit):
        seen.append(commit)
        return [f"{'9' * 40} edit analysis"]

    monkeypatch.setattr(enrichment, "analysis_changes_since", changes)
    report = enrichment.freeze_report(tmp_path / "missing.parquet", go_reports=go, analysis_outputs=tmp_path / "none",
                                      read_go_report=enrichment.read_working_file)
    assert seen == ["1" * 40] and report["go_report"]["commit"] == "1" * 40
    assert len(report["go_reports"]) == 2 and report["analysis_changes_since_go"]
    # by default only reports committed at HEAD count: a working-tree report (outside the repository here) is listed
    # as a problem, never used as the baseline
    report = enrichment.freeze_report(tmp_path / "missing.parquet", go_reports=go, analysis_outputs=tmp_path / "none")
    assert report["go_report"] is None and report["analysis_changes_since_go"] is None
    assert sum("not committed at HEAD" in p for p in report["problems"]) == 2


def test_freeze_counts_a_committed_go_report_deleted_from_the_working_tree(tmp_path, monkeypatch) -> None:
    """Freeze listed the go reports from the working tree, so deleting the earliest committed report (an output
    path: the tree stays clean) moved the Part 5.8 baseline to a later report."""
    from pilot import scheduler

    monkeypatch.setattr(provenance, "REPO_ROOT", tmp_path)
    go = tmp_path / "results" / "pilot"
    go.mkdir(parents=True)
    early, later = "go_report-20270101T000000Z.json", "go_report-20270601T000000Z.json"
    committed = {early: json.dumps({"generated": {"commit": "1" * 40, "revision": 0}}).encode(),
                 later: json.dumps({"generated": {"commit": "2" * 40, "revision": 0}}).encode()}
    (go / later).write_bytes(committed[later])  # the earliest is deleted locally, not in HEAD
    monkeypatch.setattr(scheduler, "committed_files", lambda head, directory, *a: [
        f"{directory.rstrip('/')}/{name}" for name in sorted(committed)] if directory == "results/pilot" else [])
    seen = []
    monkeypatch.setattr(enrichment, "analysis_changes_since", lambda commit: seen.append(commit) or [])
    report = enrichment.freeze_report(tmp_path / "missing.parquet", go_reports=go, analysis_outputs=tmp_path / "none",
                                      read_go_report=lambda path: committed.get(Path(path).name))
    assert seen == ["1" * 40] and report["go_report"]["commit"] == "1" * 40
    assert [r["commit"] for r in report["go_reports"]] == ["1" * 40, "2" * 40]
    assert any(f"results/pilot/{early} is committed at HEAD but missing" in p for p in report["problems"])
    assert report["ready"] is False


def test_freeze_reports_a_go_report_commit_it_cannot_read(tmp_path) -> None:
    go = tmp_path / "go"
    go.mkdir()
    (go / "go_report-20261001T000000Z.json").write_text(json.dumps({"generated": {"commit": "0" * 40}}))
    report = enrichment.freeze_report(tmp_path / "missing.parquet", go_reports=go, analysis_outputs=tmp_path / "none",
                                      read_go_report=enrichment.read_working_file)
    assert report["analysis_changes_since_go"] is None and any("git log" in p for p in report["problems"])


@pytest.mark.parametrize("content", [{"verdict": "go"}, {"generated": {}}])
def test_freeze_lists_a_go_report_without_a_generated_record_as_a_problem(tmp_path, content) -> None:
    """A go report that is JSON but records nothing under ``generated`` is unreadable: it is listed under the
    problems, never compared with the revisions of the others (no TypeError)."""
    go = tmp_path / "go"
    go.mkdir()
    (go / "go_report-20260101T000000Z.json").write_text(json.dumps(content))
    (go / "go_report-20260102T000000Z.json").write_text(json.dumps({"generated": {"commit": "a" * 40, "revision": 0}}))
    reports = enrichment.list_go_reports(go, read=enrichment.read_working_file)
    assert [(r["readable"], r["revision"]) for r in reports] == [(False, None), (True, 0)]
    assert enrichment.decision_go_report(reports) == reports[1]
    assert enrichment.decision_go_report([{**reports[1], "revision": None}]) is None
    report = enrichment.freeze_report(tmp_path / "missing.parquet", go_reports=go, analysis_outputs=tmp_path / "none",
                                      read_go_report=enrichment.read_working_file)
    assert any("go_report-20260101T000000Z.json is not committed at HEAD (or unreadable)" in p
               for p in report["problems"])


def test_completeness_counts_the_scheduler_statuses_per_arm(tmp_path) -> None:
    from pilot.scheduler import Scheduler, SchedulerConfig

    spec = study_a_spec(0.0, 0)
    paths, root = ledger_with(tmp_path, [spec])
    sched = Scheduler(SchedulerConfig(data_root=tmp_path / "state"))
    sched.add([spec, study_a_spec(0.0, 1)])
    sched.db.execute("UPDATE runs SET status = 'ledgered' WHERE run_id = ?", (spec.run_id,))
    sched.db.commit()
    sched.db.close()
    report = enrichment.completeness_report(paths.main, enrichment.read_scheduler_state(tmp_path / "state"))
    entry = report["arms"]["A-PointGoal1-N0.00"]
    assert (entry["queued"], entry["trained"], entry["ledgered"], entry["ledger_rows"]) == (2, 1, 1, 1)
    assert entry["by_status"] == {"ledgered": 1, "pending": 1} and report["totals"]["not_finished"] == 1


def test_read_scheduler_state_writes_nothing(tmp_path) -> None:
    from pilot.scheduler import Scheduler, SchedulerConfig

    assert enrichment.read_scheduler_state(tmp_path / "none") is None and not (tmp_path / "none").exists()
    sched = Scheduler(SchedulerConfig(data_root=tmp_path / "root"))
    sched.add([study_a_spec(0.0, 0)])
    sched.db.close()
    state_path = SchedulerConfig(data_root=tmp_path / "root").state_path
    before = state_path.read_bytes()
    state = enrichment.read_scheduler_state(tmp_path / "root")
    assert state.status_of("A-PointGoal1-N0.00-s0") == "pending" and state.status_of("x-s0") is None
    assert state.specs["A-PointGoal1-N0.00-s0"] == study_a_spec(0.0, 0) and state_path.read_bytes() == before


# ---------------------------------------------------------------------------
# Command line: go and g4-measure
# ---------------------------------------------------------------------------


def test_go_rejects_a_non_positive_concurrency_with_its_own_message(capsys) -> None:
    from pilot.__main__ import main

    assert main(["go", "--concurrent", "0"]) == 2
    assert "--concurrent must be at least 1" in capsys.readouterr().err


def _dirty(monkeypatch) -> None:
    def refuse(*a, **k):
        raise provenance.DirtyWorktreeError("tracked files have uncommitted changes")

    monkeypatch.setattr(provenance, "require_clean_worktree", refuse)
    monkeypatch.setattr(provenance, "dirty_paths", lambda *a, **k: ["pilot/go_decision.py"])


def test_go_and_g4_refuse_uncommitted_code_with_exit_2(tmp_path, monkeypatch, capsys) -> None:
    from pilot import ledger_writer
    from pilot.__main__ import main

    _dirty(monkeypatch)
    assert main(["go", "--out-dir", str(tmp_path)]) == 2
    assert "commit before computing the go decision" in capsys.readouterr().err
    pilot_ledger = tmp_path / "pilot.parquet"
    pilot_ledger.write_bytes(b"")
    monkeypatch.setattr(ledger_writer, "DEFAULT_PATHS", LedgerPaths(main=tmp_path / "m.parquet", pilot=pilot_ledger,
                                                                    sidecar_dir=tmp_path / "s"))
    assert main(["g4-measure", "--data-root", str(tmp_path), "--out", str(tmp_path / "g4.json")]) == 2
    err = capsys.readouterr().err
    assert "commit before taking the G4 measurement" in err and "pilot/go_decision.py" in err
    assert not (tmp_path / "g4.json").exists() and sorted(p.name for p in tmp_path.iterdir()) == ["pilot.parquet"]


def test_go_report_records_the_commit_and_a_clean_worktree(tmp_path, monkeypatch) -> None:
    from datetime import datetime, timezone

    import pilot.__main__ as cli
    from pilot import budget, go_decision

    monkeypatch.setattr(provenance, "require_clean_worktree", lambda *a, **k: "a" * 40)
    monkeypatch.setattr(provenance, "commit_hash", lambda *a, **k: "a" * 40)  # HEAD does not move
    monkeypatch.setattr(provenance, "dirty_paths", lambda *a, **k: [])
    monkeypatch.setattr(provenance, "unverified_imported_code", lambda *a, **k: [])
    record = {"run_id": "P-A-x", "completed": True, "wall_clock_hours": 24.0,
              "started": datetime(2026, 1, 1, tzinfo=timezone.utc), "finished": datetime(2026, 1, 2, tzinfo=timezone.utc)}
    unconstrained = {**record, "run_id": "P-B-unconstrained-s0", "arm": "unconstrained", "final_cost": 55.0,
                     "checkpoints": [{"step": 200_000, "selection_cost": 50.0}, {"step": 400_000, "selection_cost": 44.0}]}
    monkeypatch.setattr(cli, "_load_pilot_records", lambda *a, **k: [record, unconstrained])
    monkeypatch.setattr(budget, "allocation_contained", lambda records: (True, "ok"))
    seen = {}

    def report(*a, **k):
        seen.update(k)
        return {}

    monkeypatch.setattr(go_decision, "report", report)
    written = {}

    def capture(result, out_dir, stamp):
        written.update(result)
        md = out_dir / "r.md"
        md.write_text("report")
        return out_dir / "r.json", md

    monkeypatch.setattr(go_decision, "write_report", capture)
    assert cli.main(["go", "--out-dir", str(tmp_path)]) == 0
    assert written["generated"]["commit"] == "a" * 40 and written["generated"]["worktree_dirty"] is False
    # Q-final-cost-set's other reading: the selection-set cost of the unconstrained run's final checkpoint
    assert seen["unconstrained_cost"] == 55.0 and seen["unconstrained_selection_cost"] == 44.0


def test_go_report_records_the_pilot_ledger_it_read(tmp_path, monkeypatch, capsys) -> None:
    """The go report did not record which pilot ledger (an output path, exempt from the clean-tree check) it was
    computed from. It records the sha256 of the ledger and of each sidecar file and the run_ids, and refuses when
    they change while the decision is computed."""
    import hashlib
    from datetime import datetime, timezone

    import pilot.__main__ as cli
    import pilot.ledger_writer as lw
    from pilot import budget, go_decision

    monkeypatch.setattr(provenance, "require_clean_worktree", lambda *a, **k: "a" * 40)
    monkeypatch.setattr(provenance, "commit_hash", lambda *a, **k: "a" * 40)
    monkeypatch.setattr(provenance, "dirty_paths", lambda *a, **k: [])
    monkeypatch.setattr(provenance, "unverified_imported_code", lambda *a, **k: [])
    paths = lw.LedgerPaths(main=tmp_path / "ledger.parquet", pilot=tmp_path / "pilot.parquet", sidecar_dir=tmp_path / "s")
    monkeypatch.setattr(lw, "DEFAULT_PATHS", paths)
    paths.pilot.write_bytes(b"pilot ledger v1")
    paths.sidecar_dir.mkdir()
    (paths.sidecar_dir / "P-A-y.json").write_text("{}")
    record = {"run_id": "P-A-x", "completed": True, "wall_clock_hours": 24.0,
              "started": datetime(2026, 1, 1, tzinfo=timezone.utc), "finished": datetime(2026, 1, 2, tzinfo=timezone.utc)}
    monkeypatch.setattr(cli, "_load_pilot_records", lambda *a, **k: [record])
    monkeypatch.setattr(budget, "allocation_contained", lambda records: (True, "ok"))
    written: dict = {}

    def capture(result, out_dir, stamp):
        written.update(result)
        md = out_dir / "r.md"
        md.write_text("report")
        return out_dir / "r.json", md

    monkeypatch.setattr(go_decision, "write_report", capture)
    monkeypatch.setattr(go_decision, "report", lambda *a, **k: {})
    assert cli.main(["go", "--out-dir", str(tmp_path)]) == 0
    generated = written["generated"]
    assert generated["inputs"]["pilot_ledger"]["sha256"] == hashlib.sha256(b"pilot ledger v1").hexdigest()
    assert generated["inputs"]["sidecar"]["sha256"] == {"P-A-y.json": hashlib.sha256(b"{}").hexdigest()}
    assert generated["run_ids"] == ["P-A-x"]
    written.clear()

    def enrich_meanwhile(*a, **k):
        paths.pilot.write_bytes(b"pilot ledger v2")  # e.g. `enrich` rewrote the ledger during the command
        return {}

    monkeypatch.setattr(go_decision, "report", enrich_meanwhile)
    assert cli.main(["go", "--out-dir", str(tmp_path)]) == 2
    assert "changed while the go decision was computed" in capsys.readouterr().err and not written


def test_go_checks_and_records_the_g4_moderate_measurement_it_reads(tmp_path, monkeypatch, capsys) -> None:
    """`go` fed G4 with any JSON file holding a mean_cost map (a re-pilot measurement, another run's or a
    hand-written one) and the report did not say which. The file must be g4-measure's of this revision's completed
    Moderate run at a commit of HEAD's history; the report records its path, sha256 and provenance, and the
    revision and the earlier go reports of that revision."""
    import hashlib
    from datetime import datetime, timezone

    import pilot.__main__ as cli
    from pilot import budget, go_decision

    monkeypatch.setattr(provenance, "require_clean_worktree", lambda *a, **k: "a" * 40)
    monkeypatch.setattr(provenance, "commit_hash", lambda *a, **k: "a" * 40)  # HEAD does not move
    monkeypatch.setattr(provenance, "dirty_paths", lambda *a, **k: [])
    monkeypatch.setattr(provenance, "unverified_imported_code", lambda *a, **k: [])
    times = {"completed": True, "wall_clock_hours": 24.0, "started": datetime(2026, 1, 1, tzinfo=timezone.utc),
             "finished": datetime(2026, 1, 2, tzinfo=timezone.utc)}
    moderate_run = {**times, "run_id": "P-B-Moderate-s0", "study": "B", "arm": "Moderate"}
    monkeypatch.setattr(cli, "_load_pilot_records", lambda *a, **k: [moderate_run])
    monkeypatch.setattr(budget, "allocation_contained", lambda records: (True, "ok"))
    history = {"b" * 40}
    monkeypatch.setattr(cli, "_commit_in_history", lambda commit: commit in history)
    seen, written = {}, {}
    monkeypatch.setattr(go_decision, "report", lambda *a, **k: seen.update(k) or {})

    def capture(result, out_dir, stamp):
        written.update(result)
        md = out_dir / "r.md"
        md.write_text("report")
        return out_dir / "r.json", md

    monkeypatch.setattr(go_decision, "write_report", capture)
    out = tmp_path / "out"
    out.mkdir()
    g4 = tmp_path / "g4.json"

    def go(provenance_: object) -> int:
        g4.write_text(json.dumps({"mean_cost": {"10": 1.0, "20": 2.0, "40": 3.0},
                                  **({} if provenance_ is None else {"provenance": provenance_})}))
        return cli.main(["go", "--out-dir", str(out), "--moderate", str(g4)])

    assert go(None) == 2 and "records no provenance" in capsys.readouterr().err
    assert go({"run_id": "P1-B-Moderate-s0", "commit": "b" * 40}) == 2  # the re-pilot's, given for the first pilot
    assert "measures P1-B-Moderate-s0" in capsys.readouterr().err
    assert go({"run_id": "P-B-Moderate-s1", "commit": "b" * 40}) == 2  # not the completed Moderate run
    assert "measures P-B-Moderate-s1, but the completed Moderate run" in capsys.readouterr().err
    assert go({"run_id": "P-B-Moderate-s0", "commit": "c" * 40}) == 2
    assert "not in HEAD's history" in capsys.readouterr().err and not written
    (out / "go_report-20260101T000000Z.json").write_text(json.dumps({"generated": {"commit": "a" * 40}}))
    good = {"run_id": "P-B-Moderate-s0", "commit": "b" * 40}
    assert go(good) == 0
    assert seen["moderate"]["provenance"] == good
    generated = written["generated"]
    assert generated["g4_moderate"] == {"path": str(g4), "sha256": hashlib.sha256(g4.read_bytes()).hexdigest(),
                                        "provenance": good}
    assert generated["revision"] == 0 and generated["earlier_reports"] == ["go_report-20260101T000000Z.json"]
    assert "go reports of this pilot revision exist already" in capsys.readouterr().err


def test_write_once_is_atomic_and_never_overwrites(tmp_path, monkeypatch) -> None:
    import os

    import pilot.__main__ as cli

    out = tmp_path / "g4.json"
    cli._write_once(out, "{}\n", "taken once")
    with pytest.raises(cli.CliError, match="taken once"):
        cli._write_once(out, "{\"other\": 1}\n", "taken once")
    assert out.read_text() == "{}\n"

    def crash(fd):
        raise OSError("disk full")

    monkeypatch.setattr(os, "fsync", crash)
    with pytest.raises(OSError, match="disk full"):
        cli._write_once(tmp_path / "g4-rev1.json", "{}\n", "taken once")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["g4.json"]  # no partial file, no temporary file


def _moderate_pilot(tmp_path: Path, monkeypatch, result: dict) -> Path:
    """A ledgered Moderate pilot run under ``<tmp>/data``, a clean tree, and a harness returning ``result``."""
    from pilot import ledger_writer
    from pilot.scheduler import SchedulerConfig

    spec = RunSpec(run_id="P-B-Moderate-s0", study="B", task="SafetyPointGoal1-v0", arm="Moderate", seed=0,
                   total_steps=TOTAL, base_algo="PPOLag", plugin="study_b", group="study_b",
                   training_levels=(10.0, 20.0, 40.0), pilot=True)
    run_dir = SchedulerConfig(data_root=tmp_path / "data").run_dir(spec.run_id)
    run_dir.mkdir(parents=True)
    (run_dir / "spec.json").write_text(spec.to_json())
    omni = fake_launcher.write_outputs(spec.to_dict(), run_dir)
    (run_dir / "train_result.json").write_text(json.dumps(fake_launcher.train_result(spec.to_dict(), omni, "completed", None)))
    steps = STEPS
    ev = fake_launcher.evaluation_result(spec.to_dict(), steps, [25.0] * len(steps), final_cost=20.0, final_return=5.0,
                                         selection_return=1.0)
    ev["measurement_seeds"] = _measurement_seeds()  # the harness's set: g4-measure checks it against the registry
    (run_dir / "evaluation.json").write_text(json.dumps(ev))

    def levels(r: dict) -> dict:  # Study B logs one multiplier per training level
        r = {k: v for k, v in r.items() if not k.startswith("Metrics/LagrangeMultiplier")}
        return {**r, **{f"Metrics/LagrangeMultiplier/level_{b}": "0.1" for b in (10, 20, 40)}}

    _rewrite_progress(run_dir / omni, levels)
    paths = LedgerPaths(main=tmp_path / "m.parquet", pilot=tmp_path / "pilot.parquet", sidecar_dir=tmp_path / "s")
    write_run(spec, run_dir, paths)
    monkeypatch.setattr(ledger_writer, "DEFAULT_PATHS", paths)
    monkeypatch.setattr(provenance, "require_clean_worktree", lambda *a, **k: fake_launcher.COMMIT)
    monkeypatch.setattr(provenance, "commit_hash", lambda *a, **k: fake_launcher.COMMIT)  # HEAD does not move
    monkeypatch.setattr(provenance, "unverified_imported_code", lambda *a, **k: [])
    monkeypatch.setattr(enrichment, "_load", lambda target: lambda omnisafe_dir, spec: dict(result))
    return run_dir


def _measurement_seeds() -> list[int]:
    from envs.evaluation import measurement_seeds

    return measurement_seeds()


GOOD_G4 = {"episodes": R.EVAL_EPISODES, "mean_cost": {10.0: 9.0, 20.0: 19.0, 40.0: 38.0},
           "satisfaction": {10.0: 0.9, 20.0: 0.9, 40.0: 0.9}}


def good_g4() -> dict:
    return {**GOOD_G4, "seeds": _measurement_seeds()}


def test_g4_measure_refuses_a_namesake_run_under_another_data_root(tmp_path, monkeypatch, capsys) -> None:
    """Same commit, another data root: the final checkpoint path differs from the ledger's, so nothing is written."""
    from pilot.__main__ import main

    run_dir = _moderate_pilot(tmp_path, monkeypatch, good_g4())
    shutil.copytree(run_dir, tmp_path / "other" / "checkpoints" / run_dir.name)
    out = tmp_path / "g4.json"
    assert main(["g4-measure", "--data-root", str(tmp_path / "other"), "--out", str(out)]) == 2
    assert "final checkpoint" in capsys.readouterr().err and not out.exists()
    assert main(["g4-measure", "--data-root", str(tmp_path / "data"), "--out", str(out)]) == 0
    data = json.loads(out.read_text())
    assert data["mean_cost"] == {"10.0": 9.0, "20.0": 19.0, "40.0": 38.0} and data["provenance"]["run_id"] == run_dir.name


@pytest.mark.parametrize("mean_cost", [None, {10.0: 9.0, 40.0: 38.0}, {10.0: float("nan"), 20.0: 19.0, 40.0: 38.0},
                                       {10.0: "n/a", 20.0: 19.0, 40.0: 38.0}])
def test_g4_measure_refuses_a_result_without_finite_costs_at_every_training_budget(tmp_path, monkeypatch, capsys,
                                                                                  mean_cost) -> None:
    """Part 6 G4 decides on the mean cost at 10, 20 and 40; the file is taken once, so a bad result is never written."""
    from pilot.__main__ import main

    _moderate_pilot(tmp_path, monkeypatch, {**good_g4(), "mean_cost": mean_cost})
    out = tmp_path / "g4.json"
    assert main(["g4-measure", "--data-root", str(tmp_path / "data"), "--out", str(out)]) == 2
    assert "finite, non-negative mean_cost for each training budget" in capsys.readouterr().err and not out.exists()


@pytest.mark.parametrize("satisfaction", [None, {10.0: 0.9, 40.0: 0.9}, {10.0: 1.5, 20.0: 0.9, 40.0: 0.9},
                                          {10.0: float("nan"), 20.0: 0.9, 40.0: 0.9},
                                          {10.0: True, 20.0: False, 40.0: True}])
def test_g4_measure_refuses_a_result_without_satisfaction_rates_at_every_training_budget(
    tmp_path, monkeypatch, capsys, satisfaction
) -> None:
    """Part 3.6 measures the Moderate arm's satisfaction rates too; a bad result is never written."""
    from pilot.__main__ import main

    _moderate_pilot(tmp_path, monkeypatch, {**good_g4(), "satisfaction": satisfaction})
    out = tmp_path / "g4.json"
    assert main(["g4-measure", "--data-root", str(tmp_path / "data"), "--out", str(out)]) == 2
    assert "satisfaction rate in [0, 1]" in capsys.readouterr().err and not out.exists()


@pytest.mark.parametrize("seeds", [None, list(range(100)), list(range(2_000_000, 2_000_099)) + [7]])
def test_g4_measure_refuses_a_result_off_the_measurement_seeds(tmp_path, monkeypatch, capsys, seeds) -> None:
    from pilot.__main__ import main

    _moderate_pilot(tmp_path, monkeypatch, {**good_g4(), "seeds": seeds})
    out = tmp_path / "g4.json"
    assert main(["g4-measure", "--data-root", str(tmp_path / "data"), "--out", str(out)]) == 2
    assert "seeds" in capsys.readouterr().err and not out.exists()


def test_g4_measure_refuses_seeds_other_than_the_pilot_ledgers_measurement_set(tmp_path, monkeypatch, capsys) -> None:
    """The measurement must be on the set the pilot runs fixed in the pilot ledger's seed registry, not only on the
    set of the harness at the current commit (a later harness may have changed it)."""
    from pilot.__main__ import main

    _moderate_pilot(tmp_path, monkeypatch, good_g4())
    registry = tmp_path / "pilot.seeds.json"
    fixed = json.loads(registry.read_text())
    assert fixed["measurement"] == sorted(_measurement_seeds())
    registry.write_text(json.dumps({**fixed, "measurement": [s + 7 for s in fixed["measurement"]]}))
    out = tmp_path / "g4.json"
    assert main(["g4-measure", "--data-root", str(tmp_path / "data"), "--out", str(out)]) == 2
    assert "measurement set" in capsys.readouterr().err and not out.exists()


def test_g4_measure_reports_an_open_question_as_a_refusal(tmp_path, monkeypatch, capsys) -> None:
    from pilot.__main__ import main

    _moderate_pilot(tmp_path, monkeypatch, good_g4())

    def gated(omnisafe_dir, spec):
        errors.require_answered("Q-studyb-eval", what="the G4 measurement")

    monkeypatch.setattr(enrichment, "_load", lambda target: gated)
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset())  # the keys open, whatever the repository answers
    out = tmp_path / "g4.json"
    assert main(["g4-measure", "--data-root", str(tmp_path / "data"), "--out", str(out)]) == 3
    err = capsys.readouterr().err
    assert "refused: PendingQuestionError" in err and "Q-studyb-eval" in err and "Traceback" not in err
    assert not out.exists()


def test_go_refuses_an_explicit_moderate_file_that_does_not_exist(tmp_path, capsys) -> None:
    import pilot.__main__ as cli

    assert cli.main(["go", "--out-dir", str(tmp_path), "--moderate", str(tmp_path / "typo.json")]) == 2
    assert "typo.json does not exist" in capsys.readouterr().err and list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# ledger_writer: the unconstrained sidecar
# ---------------------------------------------------------------------------


def test_the_unconstrained_pilot_runs_measurement_seeds_are_checked_and_recorded(tmp_path) -> None:
    """G4's unconstrained cost is the sidecar run's final_cost, measured on the set of Q-final-cost-set; its seeds
    are checked against the pilot ledger's registry as every row's are, and the sidecar records which seeds and
    episodes produced it."""
    from pilot import ledger_writer

    paths = LedgerPaths(main=tmp_path / "m.parquet", pilot=tmp_path / "pilot" / "ledger.parquet",
                        sidecar_dir=tmp_path / "pilot" / "sidecar")
    root = tmp_path / "data" / "checkpoints"
    specs = {s.arm: s for s in manifest.pilot() if s.study == "B"}
    write_run(specs["Moderate"], make_run(root, specs["Moderate"]), paths)  # fixes the measurement set
    u_dir = make_run(root, specs["unconstrained"], final_cost=60.0)
    ev = json.loads((u_dir / "evaluation.json").read_text())
    shifted = {**ev, "measurement_seeds": [x + 7 for x in ev["measurement_seeds"]]}
    (u_dir / "evaluation.json").write_text(json.dumps(shifted))
    with pytest.raises(ContractError, match="measurement seeds differ"):
        write_run(specs["unconstrained"], u_dir, paths)
    assert not (paths.sidecar_dir / f"{specs['unconstrained'].run_id}.json").exists()
    (u_dir / "evaluation.json").write_text(json.dumps(ev))
    target = write_run(specs["unconstrained"], u_dir, paths)
    record = json.loads(target.read_text())["evaluation"]
    assert record["final_seed_set"] == "measurement" and record["final_seeds"] == ev["measurement_seeds"]
    assert record["final_episode_costs"] == ev["episode_costs"]["final"]
    assert ledger_writer.seeds_registry(paths.pilot).exists()


@pytest.mark.parametrize("edit, message", [
    (lambda ev: {**ev, "final_cost": 3.0}, "not the mean of its 100 episodes"),
    (lambda ev: {**ev, "episode_costs": {**ev["episode_costs"], "final": ev["episode_costs"]["final"][:7]},
                 "episode_returns": {**ev["episode_returns"], "final": ev["episode_returns"]["final"][:7]}}, "episode"),
    (lambda ev: {**ev, "final_step": 1234}, "not the run's total"),
    (lambda ev: {k: v for k, v in ev.items() if k not in ("episode_costs", "episode_returns")}, "per-episode"),
])
def test_the_unconstrained_pilot_runs_final_cost_is_checked_against_its_episodes(tmp_path, edit, message) -> None:
    """The sidecar's final_cost is G4's unconstrained cost, so its evaluation passes the checks of a ledger row's
    evaluation record (final_step, R.EVAL_EPISODES episodes, the mean of them)."""
    from pilot.ledger_writer import LedgerWriteError

    paths = LedgerPaths(main=tmp_path / "m.parquet", pilot=tmp_path / "pilot" / "ledger.parquet",
                        sidecar_dir=tmp_path / "pilot" / "sidecar")
    spec = next(s for s in manifest.pilot() if s.arm == "unconstrained")
    u_dir = make_run(tmp_path / "data" / "checkpoints", spec, final_cost=60.0)
    ev = json.loads((u_dir / "evaluation.json").read_text())
    (u_dir / "evaluation.json").write_text(json.dumps(edit(ev)))
    with pytest.raises(LedgerWriteError, match=message):
        write_run(spec, u_dir, paths)
    assert not (paths.sidecar_dir / f"{spec.run_id}.json").exists()
    assert not (tmp_path / "pilot" / "ledger.seeds.json").exists()

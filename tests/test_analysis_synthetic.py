"""Synthetic ledgers and supplement records with known answers, for the analysis tests (Role 4).

Rows are built from the registered design (pilot.manifest), validated as ``LedgerRow`` (schema v1),
and either used as dicts (``study_a_row``, ``study_b_row``, ``a_world``, ``b_world``) or appended to a real Parquet ledger with
``results.ledger_schema.append_to_ledger`` (``write_ledger``). Supplement records are built with the
models of ``results.supplement_schema``. The few tests at the end check the builders themselves.
"""

from __future__ import annotations

import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from configs import registered as R
from pilot import manifest
from results import ledger_schema as L
from results import supplement_schema as S

COMMIT = "a" * 40
NOISE = (-1.0, -0.5, 0.0, 0.5, 1.0, -0.25, 0.25, -0.75, 0.75, -0.1, 0.1, 0.0)  # by seed: mean 0 over seeds 0-4
STARTED = datetime(2026, 10, 1, tzinfo=timezone.utc)
FINISHED = datetime(2026, 10, 2, tzinfo=timezone.utc)


def noise(seed: int) -> float:
    return NOISE[seed % len(NOISE)]


def specs(*, tasks: Iterable[str] = (R.PRIMARY_TASK,), groups: Iterable[str] = ("main",),
          seeds: Iterable[int] = R.SEEDS) -> list[manifest.RunSpec]:
    """Registered specs of the given design groups and tasks (Study A), or Study B arms."""
    seeds = tuple(seeds)
    out: list[manifest.RunSpec] = []
    for group in groups:
        out += [s for s in manifest.design(group, seeds) if s.study == "B" or s.task in tuple(tasks)]
    return out


def checkpoints(spec: manifest.RunSpec, selection: Callable[[int], float]) -> list[dict[str, Any]]:
    from pilot.contracts import end_relative_window

    window = set(end_relative_window(spec.total_steps))  # the grid's last ten; end-relative off the grid (Q-selection-window)
    steps = set(range(0, spec.total_steps + 1, R.CHECKPOINT_INTERVAL_STEPS)) | window
    if spec.onset_step:
        steps.add(spec.onset_step)
    ordered = sorted(steps)
    out = []
    for step in ordered:
        out.append({
            "path": f"{spec.run_id}/omnisafe/torch_save/epoch-{step // R.STEPS_PER_EPOCH}.pt", "step": step,
            "training_cost": math.nan if step == 0 else 30.0, "training_return": math.nan if step == 0 else 10.0,
            "dormant": 0.1, "rank": 20.0, "norm": 5.0,
            "selection_cost": selection(step) if step in window else None,
            "selection_return": 12.0 if step in window else None, "multiplier": 0.5,
        })
    return out


def study_a_row(spec: manifest.RunSpec, *, cost: float, gaps: dict[str, Optional[float]], completed: bool = True,
                dormant: Optional[float] = None, rank: Optional[float] = None, final_cost: Optional[float] = None,
                selection_step: Optional[int] = None, **extra: Any) -> dict[str, Any]:
    """A validated Study A ledger row (dict) whose selection chose ``selection_step`` (default: the last)."""
    chosen = spec.total_steps if selection_step is None else selection_step
    cps = checkpoints(spec, lambda step: 25.0 if step == chosen else 40.0)
    fields: dict[str, Any] = dict(
        run_id=spec.run_id, study="A", task=spec.task, arm=spec.arm, N=spec.N, onset_shape=spec.onset_shape,
        step_matching=spec.step_matching, treatment=spec.treatment, controller_variant=spec.controller_variant,
        seed=spec.seed, commit_hash=COMMIT, config_hash="c" * 64, started=STARTED, machine="synthetic",
        completed=completed, checkpoints=cps, final_cost=cost if final_cost is None else final_cost, final_return=10.0,
    )
    if completed:
        chosen_path = next(c["path"] for c in cps if c["step"] == chosen)
        fields.update(finished=FINISHED, wall_clock_hours=24.0, matched_checkpoint_path=chosen_path,
                      matched_checkpoint_step=chosen, training_age=chosen, selection_cost_at_match=25.0,
                      measurement_cost=cost, lambda_at_selection=0.5, dormant_onset=dormant, rank_onset=rank,
                      norm_onset=5.0, **{f"gap_{c}": v for c, v in gaps.items()})
    else:
        fields.update(failure_cause="crash")
    fields.update(extra)
    return L.LedgerRow(**fields).model_dump(mode="python")


def study_b_row(spec: manifest.RunSpec, *, sr_zero: dict[float, float], sr_fewshot: Optional[dict[str, float]] = None,
                adapt_steps: Optional[dict[float, int]] = None) -> dict[str, Any]:
    cps = checkpoints(spec, lambda step: 25.0)
    return L.LedgerRow(
        run_id=spec.run_id, study="B", task=spec.task, arm=spec.arm, seed=spec.seed,
        training_levels=None if spec.training_levels is None else list(spec.training_levels), commit_hash=COMMIT,
        config_hash="c" * 64, started=STARTED, finished=FINISHED, wall_clock_hours=24.0, machine="synthetic",
        completed=True, checkpoints=cps, final_cost=20.0, final_return=10.0, sr_zero=sr_zero, sr_fewshot=sr_fewshot,
        adapt_steps=adapt_steps,
    ).model_dump(mode="python")


def write_ledger(path: Path, rows: Iterable[dict[str, Any]]) -> Path:
    """Append each row as a LedgerRow with the frozen schema's own writer."""
    for row in rows:
        L.append_to_ledger(L.LedgerRow(**row), path)
    return path


# ---------------------------------------------------------------------------
# Worlds with known answers (Study A)
# ---------------------------------------------------------------------------


def a_world(kind: str = "supported", *, tasks: Iterable[str] = (R.PRIMARY_TASK,), seeds: Iterable[int] = R.SEEDS,
            groups: Iterable[str] = ("main", "treatment", "controller", "pid"),
            cost: Callable[[manifest.RunSpec], float] = lambda s: 25.0 + 0.5 * noise(s.seed)) -> list[dict[str, Any]]:
    """Study A rows in which H1 (and the others) are built to be SUPPORTED, FALSIFIED or INCONCLUSIVE.

    supported:    gap = 2 + 12 N (abrupt), ramp 3 lower at every N; reset and injection lower a
                  treated arm's gap by 6; additional constrained training raises it by 0.2 with large
                  seed noise (D(additional) near 0); warm start lowers it by 4; dormant rises and rank
                  falls with the gap.
    falsified:    gap = 3 whatever the arm (so every difference is 0 with an interval including zero).
    inconclusive: Delta(0.10) = 1 and Delta(0.25) = 2 with little noise, Delta(0.50) = 3 with large noise.
    """
    rows = []
    for spec in specs(tasks=tasks, groups=groups, seeds=seeds):
        N, e = spec.N, noise(spec.seed)
        if kind == "supported":
            base = 2.0 + 12.0 * N + 0.3 * e
            if spec.onset_shape == "ramp":
                base -= 3.0
            if spec.treatment in ("reset", "injection"):
                base -= 6.0
            if spec.treatment == "additional_constrained":
                base += 0.2 + 2.0 * e  # D(additional) near 0: no reduction on either reading
            if spec.controller_variant == "warm_started":
                base -= 4.0
        elif kind == "falsified":
            base = 3.0 + 0.3 * e
        elif kind == "inconclusive":
            base = 2.0 + {0.0: 0.0, 0.10: 1.0, 0.25: 2.0, 0.50: 3.0}[N] + (0.05 * e if N < 0.5 else 8.0 * e)
        else:
            raise ValueError(kind)
        gaps = {"hazard": base, "dynamics": base + 1.0, "finetune": None, "transfer": None}
        rows.append(study_a_row(spec, cost=cost(spec), gaps=gaps, dormant=min(1.0, 0.05 + 0.01 * base),
                                rank=max(0.0, 40.0 - base + 0.1 * e)))
    return rows


def episodes(mean: float, *, seed_set: str = "measurement", base: int = 2_000_000, spread: int = 3) -> dict[str, Any]:
    """An EpisodeBlock with integer costs whose mean is ``mean`` (a multiple of 0.01) and fixed returns."""
    total = round(mean * R.EVAL_EPISODES)
    costs = [float(total // R.EVAL_EPISODES)] * R.EVAL_EPISODES
    for i in range(total - int(costs[0]) * R.EVAL_EPISODES):
        costs[i] += 1.0
    if min(costs) >= spread:  # spread the costs symmetrically (half of R.EVAL_EPISODES up, half down), keeping the mean
        costs = [c + (spread if i % 2 == 0 else -spread) for i, c in enumerate(costs)]
    returns = [10.0 + (i % 7) for i in range(R.EVAL_EPISODES)]
    block = {"seed_set": seed_set, "seeds": list(range(base, base + R.EVAL_EPISODES)), "episode_costs": costs,
             "episode_returns": returns, "mean_cost": statistics.fmean(costs), "mean_return": statistics.fmean(returns)}
    return block


def hazard_seeds() -> dict[str, Any]:
    return {"form": "central", "layout_seeds": list(range(3_000_000, 3_000_000 + R.HAZARD_LAYOUTS)),
            "episodes_per_layout": R.EPISODES_PER_LAYOUT}


def measurement_record(row: dict[str, Any]) -> dict[str, Any]:
    return {"kind": "measurement", "run_id": row["run_id"], "code_commit": COMMIT,
            "step": row["matched_checkpoint_step"], "episodes": episodes(row["measurement_cost"])}


def battery_record(row: dict[str, Any], condition: str, *, kind: str = "battery", gap: Optional[float] = None,
                   measurement_cost: Optional[float] = None, step: Optional[int] = None) -> dict[str, Any]:
    c_id = row["measurement_cost"] if measurement_cost is None else measurement_cost
    g = row[f"gap_{condition}"] if gap is None else gap
    block = episodes(c_id + g, seed_set="hazard" if condition == "hazard" else "measurement",
                     base=3_100_000 if condition == "hazard" else 2_000_000)
    return {"kind": kind, "run_id": row["run_id"], "code_commit": COMMIT,
            "step": row["matched_checkpoint_step"] if step is None else step, "condition": condition,
            "episodes": block, "hazard": hazard_seeds() if condition == "hazard" else None,
            "measurement_cost": c_id, "gap": block["mean_cost"] - c_id}


def training_record(row: dict[str, Any], *, spec: manifest.RunSpec, recovery: Optional[int] = 400_000,
                    dormant_check: Optional[float] = None, rank_check: Optional[float] = None) -> dict[str, Any]:
    onset = spec.onset_step or 0
    record = {"kind": "training", "run_id": row["run_id"], "code_commit": COMMIT, "onset_step": onset,
              "total_steps": spec.total_steps, "steps_per_epoch": R.STEPS_PER_EPOCH,
              "lambda_peak": 1.5, "lambda_final": 0.5, "overshoot": 1.0, "settling_steps": 600_000}
    if onset > 0:
        record.update(recovery_steps=recovery, recovery_censored=recovery is None)
        if dormant_check is not None:
            record["plasticity_check"] = {"step": onset + R.MANIPULATION_CHECK_STEPS_AFTER_ONSET,
                                          "dormant": 0.2, "rank": 20.0, "norm": 5.0,
                                          "dormant_trainable": dormant_check, "rank_trainable": rank_check}
    return record


def write_supplement(root: Path, kind: str, record: dict[str, Any]) -> Path:
    """One record per file, as canonical JSON (the reader does not depend on the layout)."""
    model = S.validate_record(kind, record)
    part = S.record_part(kind, model)
    path = Path(root) / kind / (record["run_id"] + (f"--{part}" if part else "") + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(S.canonical_json(kind, model), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Study B
# ---------------------------------------------------------------------------


def b_world(kind: str = "supported", seeds: Iterable[int] = R.SEEDS,
            sr: Optional[Callable[[str, float, int], float]] = None) -> tuple[list[dict[str, Any]], dict[str, dict]]:
    """Study B rows and their zero-shot and few-shot supplement records.

    supported: satisfaction grows with the number of levels (Sparse 0.40, Moderate 0.55, Dense 0.75,
    Continuous 0.77, singles 0.30) plus seed noise; few-shot adaptation reaches the target at the first
    horizon for Dense, the second for Moderate (and the arms G4 does not compare), never for Sparse.
    """
    base = {"Single-10": 0.30, "Single-20": 0.30, "Single-40": 0.30, "Sparse": 0.40, "Moderate": 0.55,
            "Dense": 0.75, "Continuous": 0.77}
    flat = {a: 0.50 for a in base}
    level = base if kind == "supported" else flat
    sr = sr or (lambda arm, b, seed: round(min(1.0, max(0.0, level[arm] + 0.02 * noise(seed))), 2))
    reach = {"Dense": 0, "Moderate": 1, "Sparse": None} if kind == "supported" else {"Dense": 1, "Moderate": 1, "Sparse": 1}
    # every arm is continued (manifest.study_b_fewshot); the arms G4 does not compare reach the target at the second
    # horizon, so they are off the few-shot floor that G4 reads (Q-floor-rule: all seven arms count)
    reach = {**dict.fromkeys(base, 1), **reach}
    rows, records = [], {}
    for spec in specs(groups=("study_b",), seeds=seeds):
        zero = {b: sr(spec.arm, b, spec.seed) for b in R.UNSEEN_BUDGETS}
        few, adapt, fewshot_records = {}, {}, []
        for b in R.UNSEEN_BUDGETS:
            first = reach[spec.arm]
            rates = {h: (0.9 if first is not None and i >= first else 0.2) for i, h in enumerate(R.FEWSHOT_HORIZONS)}
            for h, v in rates.items():
                few[S.fewshot_key(b, h)] = v
            reached = [h for h in R.FEWSHOT_HORIZONS if rates[h] >= R.SATISFACTION_TARGET]
            adapt[b] = reached[0] if reached else max(R.FEWSHOT_HORIZONS) + 1
            fewshot_records.append(fewshot_record(spec, b, rates, adapt[b]))
        rows.append(study_b_row(spec, sr_zero=zero, sr_fewshot=few, adapt_steps=adapt))
        records[spec.run_id] = {"zero_shot": zero_shot_record(spec, zero), "fewshot": fewshot_records}
    return rows, records


def _block_for_rate(budget: float, rate: float) -> dict[str, Any]:
    """Episodes whose satisfaction at ``budget`` is ``rate`` (rate x R.EVAL_EPISODES episodes at cost 0)."""
    k = round(rate * R.EVAL_EPISODES)
    costs = [0.0] * k + [float(budget) + 5.0] * (R.EVAL_EPISODES - k)
    returns = [10.0] * R.EVAL_EPISODES
    return {"episode_costs": costs, "episode_returns": returns, "mean_cost": statistics.fmean(costs),
            "mean_return": 10.0, "satisfaction": S._satisfaction(costs, budget), "violation": S._violation(costs, budget)}


def zero_shot_record(spec: manifest.RunSpec, zero: dict[float, float]) -> dict[str, Any]:
    budgets = []
    for b in R.UNSEEN_BUDGETS:
        budgets.append({"budget": b, "role": "unseen", "distance": S._training_set_distance(spec.arm, b),
                        **_block_for_rate(b, zero[b])})
    for b in S.reference_budgets(spec.arm):
        budgets.append({"budget": float(b), "role": "reference", **_block_for_rate(b, 0.95)})
    return {"kind": "zero_shot", "run_id": spec.run_id, "code_commit": COMMIT, "arm": spec.arm,
            "step": spec.total_steps, "seed_set": "measurement", "seeds": list(range(2_000_000, 2_000_000 + R.EVAL_EPISODES)),
            "budgets": budgets}


def fewshot_record(spec: manifest.RunSpec, budget: float, rates: dict[int, float], adapt: int) -> dict[str, Any]:
    horizons = list(rates)
    return {"kind": "fewshot", "run_id": spec.run_id, "code_commit": COMMIT,
            "continuation_run_id": f"B-{spec.arm}-fewshot-b{budget:g}-s{spec.seed}", "parent_step": spec.total_steps,
            "arm": spec.arm,
            "budget": budget, "horizons": horizons, "seed_set": "measurement",
            "seeds": list(range(2_000_000, 2_000_000 + R.EVAL_EPISODES)),
            "by_horizon": [{"horizon": h, **_block_for_rate(budget, rates[h])} for h in horizons],
            "sr_fewshot": {S.fewshot_key(budget, h): _block_for_rate(budget, rates[h])["satisfaction"] for h in horizons},
            "adapt_steps": adapt}


# ---------------------------------------------------------------------------
# Tests of the builders
# ---------------------------------------------------------------------------


def test_rows_validate_and_select_the_chosen_checkpoint() -> None:
    from analysis.matching import select_checkpoint

    rows = a_world("supported", groups=("main",))
    assert len(rows) == 13 * len(R.SEEDS)
    for row in rows[:5]:
        assert select_checkpoint(row["checkpoints"]) == row["matched_checkpoint_step"]


def test_supplement_builders_validate(tmp_path) -> None:
    rows = a_world("supported", groups=("main",))
    row = rows[0]
    for kind, record in (("measurement", measurement_record(row)), ("battery", battery_record(row, "hazard")),
                         ("battery", battery_record(row, "dynamics"))):
        path = write_supplement(tmp_path, kind, record)
        assert json.loads(path.read_text())["run_id"] == row["run_id"]
    b_rows, b_records = b_world()
    assert len(b_rows) == len(R.STUDY_B_ARMS) * len(R.SEEDS)
    for recs in b_records.values():
        S.validate_record("zero_shot", recs["zero_shot"])
        for f in recs["fewshot"]:
            S.validate_record("fewshot", f)

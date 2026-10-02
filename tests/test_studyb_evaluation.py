"""Study B's evaluations (``studyb.evaluation``; Table 2.5; equations (10) and (11); Part 4.2; Part 6 G4).

The pure functions against the registered text and against the other roles' own implementations
(``analysis.study_b``, ``results.supplement_schema``); then the four evaluations with Role 2's
primitives replaced by fast stubs (``envs.evaluation.load_policy`` and ``run_episodes``), which
check the seeds, the budgets and the checkpoints each evaluation asks for and the shape of what
it returns: go condition G4 through the real ``python -m pilot g4-measure``, contract 1 through
``pilot.contracts.validate_evaluation``, ``sr_zero``/``sr_fewshot``/``adapt_steps`` through the
frozen ledger schema. The same evaluations on real tiny checkpoints are in
tests/test_studyb_training.py (``slow``).
"""

from __future__ import annotations

import dataclasses
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from configs import registered as R
from envs import evaluation as E
from pilot import contracts, errors, manifest
from pilot.launch import plain
from studyb import evaluation as SE

TASK = R.TASKS_STUDY_B[0]
SPE = R.STEPS_PER_EPOCH


def _arm(arm: str) -> manifest.RunSpec:
    return next(s for s in manifest.study_b() if s.arm == arm and s.seed == 0)


def _fewshot(budget: float = 5.0, parent: str = "Moderate") -> manifest.RunSpec:
    return next(s for s in manifest.study_b_fewshot([_arm(parent)]) if s.params["budget"] == budget)


# ---------------------------------------------------------------------------
# Equations (10) and (11), violation magnitude, adaptation steps
# ---------------------------------------------------------------------------


def test_satisfaction_counts_a_cost_equal_to_the_budget_as_satisfied() -> None:
    """Equation (10): 1[C_e <= d_test]; Table 2.5: "at most the unseen budget"."""
    assert SE.satisfaction_rate([5.0, 10.0, 15.0], 10.0) == pytest.approx(2 / 3)
    assert SE.satisfaction_rate([0.0] * 100, 5.0) == 1.0 and SE.satisfaction_rate([46.0] * 100, 45.0) == 0.0
    assert SE.satisfaction_rate([30, 31], 30) == 0.5  # integers are costs too
    for bad in ([], [float("nan")], [-1.0], [True], "12"):
        with pytest.raises(ValueError):
            SE.satisfaction_rate(bad, 10.0)
    with pytest.raises(ValueError):
        SE.satisfaction_rate([1.0], float("inf"))


def test_violation_magnitude_is_the_mean_excess() -> None:
    """Table 2.5: "The mean over evaluation episodes of max(cost - budget, 0)"."""
    assert SE.violation_magnitude([5.0, 10.0, 15.0, 30.0], 10.0) == 6.25
    assert SE.violation_magnitude([0.0, 3.0], 5.0) == 0.0
    with pytest.raises(ValueError):
        SE.violation_magnitude([], 5.0)


def test_a_negative_cost_is_refused_citing_the_true_source_of_the_0_1_cost() -> None:
    """The refusal cites Table 2.1 only for its own words and credits the 0/1 per-step cost to Safety-Gymnasium.

    Table 2.1's "Episodic cost" row defines a sum and says nothing of an indicator. The 0/1 cost is
    Safety-Gymnasium 0.4.1's ``CostConf.constrain_indicator``, so the message must not credit it to
    the pre-registration. Error messages are outside tests/test_core_quotes.py's reach (docstrings and
    comments only), so the message is checked here with the same checker.
    """
    import inspect

    import safety_gymnasium
    from safety_gymnasium import builder
    from safety_gymnasium.bases.base_task import CostConf

    for rate in (SE.satisfaction_rate, SE.violation_magnitude):
        with pytest.raises(ValueError) as refused:
            rate([3.0, -1.0], 10.0)
        assert str(refused.value) == SE.NEGATIVE_COST
    sys.path.insert(0, str(Path(__file__).parent))
    import test_core_quotes as check

    cited, library = SE.NEGATIVE_COST.split(", and Safety-Gymnasium 0.4.1 ")
    assert "Table 2.1" in cited and not any(w in cited for w in ("indicator", "0 or 1", "0/1"))
    quotes = check.cited_quotes(f"# {SE.NEGATIVE_COST}")
    assert quotes == ["The sum over an episode of the per-step cost the task returns"]
    assert not check.missing_pieces(quotes[0])
    # The claim credited to the library holds in the pinned library.
    assert "0 or 1" in library and "constrain_indicator" in library and "Table" not in library
    assert safety_gymnasium.__version__ == "0.4.1" and CostConf().constrain_indicator is True
    assert "cost[k] = float(cost[k] > 0.0)" in inspect.getsource(builder)


# Study B spec 2.4 (equation (11), recomputed by its verifier), budgets 5, 15, 30, 45.
DISTANCES = {
    "Single-10": (5, 5, 20, 35), "Single-20": (15, 5, 10, 25), "Single-40": (35, 25, 10, 5), "Sparse": (5, 5, 10, 5),
    "Moderate": (5, 5, 10, 5), "Dense": (5, 2.5, 2.5, 5), "Continuous": (5, 0, 0, 5),
}


@pytest.mark.parametrize("arm", list(DISTANCES))
def test_distance_is_equation_11(arm) -> None:
    assert [SE.distance(b, arm) for b in R.UNSEEN_BUDGETS] == [float(d) for d in DISTANCES[arm]]


def test_nearest_single_arms_are_those_of_part_4_2() -> None:
    """Part 4.2: "Budget 5 has one nearest arm, Single-10; budget 45 has one, Single-40. Budgets 15 and 30
    are each equidistant from two single-level arms"."""
    assert [SE.nearest_single_arms(b) for b in R.UNSEEN_BUDGETS] == [
        ("Single-10",), ("Single-10", "Single-20"), ("Single-20", "Single-40"), ("Single-40",)]


def test_nearest_training_levels_are_the_references_of_g3() -> None:
    for arm in ("Sparse", "Moderate", "Dense", "Continuous"):
        assert SE.nearest_training_level(arm, 5.0) == 10.0 and SE.nearest_training_level(arm, 45.0) == 40.0
    assert SE.nearest_training_level("Dense", 15.0) == 17.5 and SE.nearest_training_level("Continuous", 22.0) == 22.0
    assert SE.nearest_training_level("Sparse", 25.0) == 10.0  # a tie goes to the lower level
    assert SE.reference_budgets("Moderate") == (10.0, 20.0, 40.0) and SE.reference_budgets("Continuous") == (10.0, 40.0)
    with pytest.raises(ValueError):
        SE.distance(5.0, "Huge")


def test_the_pure_functions_agree_with_the_analysis_and_the_supplement_schema() -> None:
    """Role 4 implements the same definitions for the analysis and for the supplement's checks."""
    pytest.importorskip("pydantic")
    from analysis import study_b as A
    from results import supplement_schema as S

    budgets = [*R.UNSEEN_BUDGETS, 10.0, 12.5, 17.5, 20.0, 25.0, 33.0, 40.0]
    for arm in R.STUDY_B_ARMS:
        for b in budgets:
            assert SE.distance(b, arm) == A.distance(b, arm) == S._training_set_distance(arm, b)
            assert SE.nearest_training_level(arm, b) == A.nearest_training_level(arm, b)
        assert SE.reference_budgets(arm) == tuple(sorted(S.reference_budgets(arm)))
    for b in budgets:
        assert SE.nearest_single_arms(b) == A.nearest_single_arms(b)
    costs = [0.0, 4.0, 5.0, 5.5, 12.0, 30.0, 45.0, 46.0]
    for b in R.UNSEEN_BUDGETS:
        assert SE.satisfaction_rate(costs, b) == S._satisfaction(costs, b)
        assert SE.violation_magnitude(costs, b) == S._violation(costs, b)


def test_adaptation_steps_is_the_first_horizon_reaching_the_target() -> None:
    """Table 2.5 "Adaptation steps": "The smallest few-shot horizon at which the satisfaction rate reaches
    0.80"; "recorded as above the largest horizon if never reached" (Q-adapt-censoring: largest + 1)."""
    horizons = list(R.FEWSHOT_HORIZONS)
    assert SE.adaptation_steps(dict(zip(horizons, (0.9, 0.95, 1.0))), horizons) == 200_000
    assert SE.adaptation_steps(dict(zip(horizons, (0.5, R.SATISFACTION_TARGET, 0.7))), horizons) == 500_000  # "reaches"
    assert SE.adaptation_steps(dict(zip(horizons, (0.5, 0.79, 0.81))), horizons) == 1_000_000
    assert SE.adaptation_steps(dict(zip(horizons, (0.9, 0.1, 0.1))), horizons) == 200_000
    assert SE.adaptation_steps(dict(zip(horizons, (0.1, 0.2, 0.79))), horizons) == 1_000_001 == SE.censored_steps(horizons)
    assert SE.adaptation_steps({200_000: 0.5}, [200_000]) == 200_001  # Part 6.1 cut 3: one horizon
    assert SE.adaptation_steps({200_000: 0.8}, [200_000]) == 200_000
    for rates, hs in (({200_000: 0.9}, horizons), ({200_000: 1.2}, [200_000]), ({500_000: 0.9, 200_000: 0.9}, [500_000, 200_000]),
                      ({0: 0.9}, [0]), ({200_000: float("nan")}, [200_000]), ([0.9], [200_000])):
        with pytest.raises(ValueError):
            SE.adaptation_steps(rates, hs)
    assert SE.adaptation_steps({"200000": 0.9}, [200_000]) == 200_000  # JSON keys read back
    assert SE.adaptation_steps({200_000.0: 0.9}, [200_000]) == 200_000  # a whole float is its horizon
    for rates, hs in (({200_000.9: 0.9}, [200_000]), ({True: 0.9}, [1]), ({"2e5x": 0.9}, [200_000]),
                      ({200_000: 0.9, "200000": 0.5}, [200_000]), ({None: 0.9}, [200_000])):
        with pytest.raises(ValueError, match="rates_by_horizon"):  # a fraction, a bool, text, a key given twice
            SE.adaptation_steps(rates, hs)
    with pytest.raises(ValueError, match=r"got \[500000, 200000\]"):  # a one-shot iterable is reported as given
        SE.censored_steps(h for h in [500_000, 200_000])


# ---------------------------------------------------------------------------
# Stubs of Role 2's primitives, and registered-looking run directories
# ---------------------------------------------------------------------------


def _studyb_cfgs(spec: manifest.RunSpec) -> dict:
    from studyb import conditioning as C

    return {"kind": C.level_kind(spec), "level_keys": list(C.level_keys(spec)), "budget_divisor": R.BUDGET_OBSERVATION_DIVISOR}


def _run_dir(root: Path, spec: manifest.RunSpec, steps: list[int], *, spe: int = SPE, studyb: dict | None = None) -> Path:
    directory = root / spec.run_id / "omnisafe" / f"PPOLag-{{{spec.task}}}" / "seed-000-fixture"
    (directory / "torch_save").mkdir(parents=True)
    config = {"env_id": spec.task, "algo": "PPOLag",  # what envs.evaluation._check_config reads, as the launcher writes it
              "algo_cfgs": {"steps_per_epoch": spe, "reward_normalize": False, "cost_normalize": False, "obs_normalize": True},
              "train_cfgs": {"total_steps": spec.total_steps}, "pilot_cfgs": {"run_id": spec.run_id},
              "studyb_cfgs": _studyb_cfgs(spec) if studyb is None else studyb}
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    for step in steps:
        (directory / "torch_save" / f"epoch-{step // spe}.pt").write_bytes(b"")
    return directory


class Primitives:
    """``load_policy`` and ``run_episodes`` of envs.evaluation, replaced by fast deterministic stubs.

    ``load_policy`` checks what the real one needs from these fixtures (the spec is given, the
    checkpoint file exists); episode costs follow ``cost(step, seed, budget)``.
    """

    def __init__(self, monkeypatch, cost=None) -> None:
        self.loads: list[int] = []
        self.calls: list[tuple[int, list[int], list[float]]] = []
        self.cost = cost or (lambda step, seed, budget: float((seed * 7 + int(budget)) % 60))
        monkeypatch.setattr(E, "load_policy", self.load_policy)
        monkeypatch.setattr(E, "run_episodes", self.run_episodes)

    def load_policy(self, directory, step, spec=None):
        assert isinstance(spec, manifest.RunSpec)
        config = json.loads((Path(directory) / "config.json").read_text(encoding="utf-8"))
        path = Path(directory) / "torch_save" / f"epoch-{step // config['algo_cfgs']['steps_per_epoch']}.pt"
        if not path.is_file():
            raise E.CheckpointInvalid(f"{path.name} is missing")
        self.loads.append(step)
        return E.Policy(omnisafe_dir=Path(directory), step=step, checkpoint=path, config=config, task=spec.task, actor=None,
                        normalizer_state=None, obs_dim=60, act_dim=2, budget_conditioned=True, injected=False)

    def run_episodes(self, policy, seeds, *, budgets=None, **kwargs):
        assert not kwargs and budgets is not None and len(budgets) == len(seeds)
        self.calls.append((policy.step, list(seeds), list(budgets)))
        return [E.EpisodeResult(seed=s, cost=self.cost(policy.step, s, b), ret=(s % 5) / 4.0, length=R.EPISODE_LENGTH,
                                budget=float(b)) for s, b in zip(seeds, budgets)]


@pytest.fixture
def answered(monkeypatch):
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset({"Q-studyb-eval"}))


# ---------------------------------------------------------------------------
# The result gate (HANDOVER.md section 10)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["evaluate_run", "evaluate_training_budgets", "evaluate_zero_shot", "evaluate_fewshot"])
def test_every_evaluation_waits_for_q_studyb_eval(tmp_path, monkeypatch, name) -> None:
    stubs = Primitives(monkeypatch)
    spec = _fewshot() if name == "evaluate_fewshot" else _arm("Moderate")
    directory = _run_dir(tmp_path, spec, [spec.total_steps])
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset())  # the key open, whatever the repository answers
    assert R.is_open("Q-studyb-eval")
    with pytest.raises(errors.PendingQuestionError, match="Q-studyb-eval"):
        getattr(SE, name)(str(directory), spec.to_dict())
    assert stubs.loads == [] and stubs.calls == []  # refused before anything is read or run
    with errors.allow_pending():  # smoke tests only: the gate opens (the fixture may lack other checkpoints)
        try:
            getattr(SE, name)(str(directory), spec.to_dict())
        except errors.PendingQuestionError:  # pragma: no cover - the failure this test looks for
            pytest.fail("allow_pending() did not open the result gate")
        except E.CheckpointInvalid:
            pass


# ---------------------------------------------------------------------------
# Go condition G4 (Part 6; Part 3.6)
# ---------------------------------------------------------------------------


def _check_as_g4_measure(result: dict, levels: list[float]) -> dict:
    """What pilot/__main__.py _cmd_g4_measure checks and writes, applied to a result."""
    from pilot.__main__ import _numbers_by_budget

    result = plain(dict(result))
    assert int(result.get("episodes", -1)) == R.EVAL_EPISODES
    costs, rates = _numbers_by_budget(result.get("mean_cost")), _numbers_by_budget(result.get("satisfaction"))
    assert sorted(costs) == levels and all(math.isfinite(v) for v in costs.values())
    assert sorted(rates) == levels and all(0.0 <= v <= 1.0 for v in rates.values())
    assert result["seeds"] == E.measurement_seeds()  # _check_g4_seeds: the registry's side is the command's own test
    json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
    return result


def test_the_training_budgets_are_measured_on_the_final_checkpoint(tmp_path, monkeypatch, answered) -> None:
    stubs = Primitives(monkeypatch)
    spec = _arm("Moderate")
    directory = _run_dir(tmp_path, spec, [spec.total_steps])
    result = SE.evaluate_training_budgets(str(directory), spec.to_dict())
    assert stubs.loads == [spec.total_steps]
    assert [(c[0], c[2][0]) for c in stubs.calls] == [(spec.total_steps, b) for b in (10.0, 20.0, 40.0)]
    assert all(c[1] == E.measurement_seeds() and len(set(c[2])) == 1 for c in stubs.calls)
    assert result["episodes"] == 100 and result["budgets"] == [10.0, 20.0, 40.0] and result["seed_set"] == "measurement"
    assert result["step"] == spec.total_steps and result["checkpoint"] == "torch_save/epoch-500.pt"
    # the harness's rules are prose under names of their own, not the result's checkpoint path and seed-set name
    assert result["harness"]["checkpoint_rule"] == "final (Q-studyb-eval)" and result["harness"]["seed_rule"] == "measurement"
    assert not {"checkpoint", "seed_set"} & set(result["harness"])
    # the stub's policy has no normaliser (obs_normalize false): the record says so, as Role 2's does
    assert result["harness"]["normaliser"] == "none (algo_cfgs.obs_normalize false)"
    assert "raw observation" in result["harness"]["budget_feature"]
    for b in (10.0, 20.0, 40.0):
        costs = result["episode_costs"][b]
        assert len(costs) == len(result["episode_returns"][b]) == 100
        assert result["mean_cost"][b] == pytest.approx(sum(costs) / 100)
        assert result["satisfaction"][b] == SE.satisfaction_rate(costs, b)
        assert result["violation"][b] == SE.violation_magnitude(costs, b)
    written = _check_as_g4_measure(result, [10.0, 20.0, 40.0])
    assert set(written) >= {"mean_cost", "satisfaction", "violation", "mean_return", "episode_costs", "episode_returns", "seeds"}
    assert SE.evaluate_training_budgets(str(directory), spec) == result  # a RunSpec works too, and it is deterministic


def test_the_g4_measure_command_writes_the_measurement(tmp_path, monkeypatch, answered) -> None:
    """The real ``python -m pilot g4-measure`` on a ledgered Moderate pilot run (tests/test_enrichment.py
    fixture), with this harness loaded through ``pilot.enrichment._load`` and Role 2's primitives stubbed."""
    pytest.importorskip("pyarrow")
    sys.path.insert(0, str(Path(__file__).parent))
    from test_enrichment import _moderate_pilot  # the pilot owner's fixture of a ledgered Moderate run

    from pilot import enrichment
    from pilot.__main__ import main

    run_dir = _moderate_pilot(tmp_path, monkeypatch, {})
    spec = manifest.RunSpec.from_json((run_dir / "spec.json").read_text())
    omni = run_dir / json.loads((run_dir / "train_result.json").read_text())["omnisafe_dir"]
    config = {"env_id": spec.task, "algo": spec.base_algo,  # complete, as the launcher writes it (_check_run_config)
              "algo_cfgs": {"steps_per_epoch": SPE, "reward_normalize": False, "cost_normalize": False, "obs_normalize": True},
              "train_cfgs": {"total_steps": spec.total_steps}, "pilot_cfgs": {"run_id": spec.run_id},
              "studyb_cfgs": _studyb_cfgs(spec)}
    (omni / "config.json").write_text(json.dumps(config), encoding="utf-8")
    loaded = []

    def load(target):
        loaded.append(target)
        return SE.evaluate_training_budgets

    monkeypatch.setattr(enrichment, "_load", load)
    stubs = Primitives(monkeypatch, cost=lambda step, seed, budget: float(budget) - 1.0 + (seed % 3))
    out = tmp_path / "g4.json"
    assert main(["g4-measure", "--data-root", str(tmp_path / "data"), "--out", str(out)]) == 0
    assert loaded == [("studyb.evaluation", "evaluate_training_budgets", "Study B and literature (Role 5)")]
    data = json.loads(out.read_text())
    assert data["episodes"] == 100 and sorted(data["mean_cost"]) == ["10.0", "20.0", "40.0"]
    assert data["mean_cost"]["20.0"] == pytest.approx(19.0 + sum(s % 3 for s in E.measurement_seeds()) / 100)
    assert data["satisfaction"]["10.0"] == pytest.approx(sum(1 for s in E.measurement_seeds() if s % 3 <= 1) / 100)
    assert stubs.loads == [2_000_000]  # the fixture's final checkpoint


def test_the_g4_measure_command_checks_the_replacements_of_each_budget() -> None:
    """pilot/__main__.py: G4's per-budget replacements (Q-mujoco-exception) follow the reserve rule of the
    measurement set and name training budgets only; absent means none."""
    from pilot.__main__ import CliError, _check_g4_replacements

    levels = [10.0, 20.0, 40.0]
    ok = {"slot_index": 0, "seed": 2_000_000, "reserve_seed": 2_050_000, "step": 5, "warning": "mjWARN_BADQACC x1"}
    _check_g4_replacements(None, levels)
    _check_g4_replacements({"10.0": [ok], "20.0": [], "40.0": []}, levels)  # as json.dumps writes the keys
    with pytest.raises(CliError, match="not the next unused seed"):
        _check_g4_replacements({10.0: [{**ok, "reserve_seed": 1_050_000}]}, levels)  # the selection set's reserve
    with pytest.raises(CliError, match="not training budgets"):
        _check_g4_replacements({15.0: []}, levels)
    with pytest.raises(CliError, match="map budgets"):
        _check_g4_replacements([ok], levels)


def test_g4_refuses_the_continuous_arm_and_other_plugins(tmp_path, monkeypatch, answered) -> None:
    Primitives(monkeypatch)
    continuous = _arm("Continuous")
    with pytest.raises(E.EvaluationRefused, match="no training levels"):
        SE.evaluate_training_budgets(str(_run_dir(tmp_path, continuous, [continuous.total_steps])), continuous.to_dict())
    fewshot = _fewshot()
    with pytest.raises(E.EvaluationRefused, match="study_b"):
        SE.evaluate_training_budgets(str(tmp_path), fewshot.to_dict())
    unconstrained = next(s for s in manifest.pilot() if s.plugin == "unconstrained_ppo")
    with pytest.raises(E.EvaluationRefused):
        SE.evaluate_training_budgets(str(tmp_path), unconstrained.to_dict())
    with pytest.raises(E.EvaluationRefused, match="run spec"):
        SE.evaluate_training_budgets(str(tmp_path), "B-Moderate-s0")


@pytest.mark.parametrize(("studyb", "words"), [
    (None, "no studyb_cfgs"),
    ({"kind": "discrete", "level_keys": [10.0, 20.0, 40.0], "budget_divisor": 10.0}, "appended budget / 10.0"),
    ({"kind": "discrete", "level_keys": [10.0, 40.0], "budget_divisor": 100.0}, "do not describe"),
    ({"kind": "continuous", "level_keys": [10.0, 20.0, 40.0], "budget_divisor": 100.0}, "do not describe"),
    ({"kind": "discrete", "level_keys": [None], "budget_divisor": 100.0}, "are not numbers"),
    ({"kind": "discrete", "level_keys": 5, "budget_divisor": 100.0}, "are not numbers"),
    ({"kind": "discrete", "level_keys": ["x"], "budget_divisor": 100.0}, "are not numbers"),
])
def test_a_run_not_trained_with_this_arms_conditioning_is_refused(tmp_path, monkeypatch, answered, studyb, words) -> None:
    stubs = Primitives(monkeypatch)
    spec = _arm("Moderate")
    directory = _run_dir(tmp_path, spec, [spec.total_steps], studyb=studyb)
    if studyb is None:
        config = json.loads((directory / "config.json").read_text())
        del config["studyb_cfgs"]
        (directory / "config.json").write_text(json.dumps(config))
    with pytest.raises(E.EvaluationRefused, match=words):
        SE.evaluate_zero_shot(str(directory), spec.to_dict())
    assert stubs.calls == []


# ---------------------------------------------------------------------------
# Zero-shot (Table 2.5; Part 4.2)
# ---------------------------------------------------------------------------


def _ledger_row(**fields):
    from results.ledger_schema import LedgerRow

    return LedgerRow(
        run_id="B-Moderate-s0", study="B", task=TASK, arm="Moderate", training_levels=[10.0, 20.0, 40.0], seed=0,
        commit_hash="a" * 40, config_hash="c" * 64, started=datetime(2026, 9, 30, tzinfo=timezone.utc),
        finished=datetime(2026, 9, 30, 1, tzinfo=timezone.utc), wall_clock_hours=1.0, machine="m", completed=True, **fields,
    )


@pytest.mark.parametrize("arm", ["Moderate", "Continuous", "Single-40"])
def test_zero_shot_runs_the_unseen_and_the_reference_budgets(tmp_path, monkeypatch, answered, arm) -> None:
    pytest.importorskip("pydantic")
    from results.supplement_schema import validate_record

    stubs = Primitives(monkeypatch)
    spec = _arm(arm)
    result = SE.evaluate_zero_shot(str(_run_dir(tmp_path, spec, [spec.total_steps])), spec.to_dict())
    unseen, reference = [5.0, 15.0, 30.0, 45.0], list(SE.reference_budgets(arm))
    assert result["unseen_budgets"] == unseen and result["reference_budgets"] == reference
    assert [c[2][0] for c in stubs.calls] == unseen + reference and stubs.loads == [spec.total_steps]
    assert all(c[1] == E.measurement_seeds() for c in stubs.calls)
    assert list(result["mean_cost"]) == unseen + reference == list(result["satisfaction"]) == list(result["episode_costs"])
    assert result["distance"] == {b: SE.distance(b, arm) for b in unseen}
    assert result["nearest_training_level"] == {b: SE.nearest_training_level(arm, b) for b in unseen}
    assert result["sr_zero"] == {b: result["satisfaction"][b] for b in unseen}
    assert _ledger_row(sr_zero=result["sr_zero"]).sr_zero == result["sr_zero"]  # the frozen schema's form
    json.dumps(plain(result), sort_keys=True, allow_nan=False)
    blocks = [{"budget": b, "role": "unseen" if b in unseen else "reference", "episode_costs": result["episode_costs"][b],
               "episode_returns": result["episode_returns"][b], "mean_cost": result["mean_cost"][b],
               "mean_return": result["mean_return"][b], "satisfaction": result["satisfaction"][b],
               "violation": result["violation"][b], "distance": result["distance"].get(b)} for b in unseen + reference]
    record = validate_record("zero_shot", {"run_id": spec.run_id, "code_commit": "a" * 40, "arm": arm, "step": result["step"],
                                           "seed_set": result["seed_set"], "seeds": result["seeds"], "budgets": blocks})
    assert record.sr_zero == result["sr_zero"]  # Role 4's recomputation of eq. (10) and eq. (11) agrees


# ---------------------------------------------------------------------------
# Few-shot (Table 2.5)
# ---------------------------------------------------------------------------


def _satisfying_from(step_reached: int):
    """Costs within the budget for 90 of 100 seeds from ``step_reached`` on, for 50 before it."""
    def cost(step, seed, budget):
        share = 90 if step >= step_reached else 50
        return float(budget) + (0.0 if seed % 100 < share else 3.0)
    return cost


@pytest.mark.parametrize(("reached", "expected"), [(200_000, 200_000), (500_000, 500_000), (2_000_000, 1_000_001)])
def test_fewshot_reads_every_horizon_of_one_continuation(tmp_path, monkeypatch, answered, reached, expected) -> None:
    pytest.importorskip("pydantic")
    from results.supplement_schema import validate_record

    stubs = Primitives(monkeypatch, cost=_satisfying_from(reached))
    spec = _fewshot(30.0)
    directory = _run_dir(tmp_path, spec, [0, 200_000, 400_000, 500_000, 600_000, 800_000, 1_000_000])
    result = SE.evaluate_fewshot(str(directory), spec.to_dict())
    horizons = [200_000, 500_000, 1_000_000]
    assert stubs.loads == horizons  # every horizon's checkpoint loaded before the first episode
    assert [c[0] for c in stubs.calls] == horizons and all(set(c[2]) == {30.0} for c in stubs.calls)
    assert result["budget"] == 30.0 and result["horizons"] == horizons and result["parent_run_id"] == "B-Moderate-s0"
    assert result["parent_step"] == R.TOTAL_STEPS
    assert list(result["by_horizon"]) == horizons
    assert [result["by_horizon"][h]["checkpoint"] for h in horizons] == ["torch_save/epoch-10.pt", "torch_save/epoch-25.pt",
                                                                         "torch_save/epoch-50.pt"]
    assert result["sr_fewshot"] == {contracts.fewshot_key(30.0, h): result["by_horizon"][h]["satisfaction"] for h in horizons}
    assert list(result["sr_fewshot"]) == ["30.0_200000", "30.0_500000", "30.0_1000000"]
    assert result["adapt_steps"] == {30.0: expected} and result["adapt_censored"] == (expected > max(horizons))
    row = _ledger_row(sr_fewshot=result["sr_fewshot"], adapt_steps=result["adapt_steps"])
    assert row.sr_fewshot == result["sr_fewshot"] and row.adapt_steps == result["adapt_steps"]
    blocks = [{"horizon": h, **{k: result["by_horizon"][h][k] for k in ("episode_costs", "episode_returns", "mean_cost",
                                                                          "mean_return", "satisfaction", "violation")}}
              for h in horizons]
    validate_record("fewshot", {"run_id": "B-Moderate-s0", "code_commit": "a" * 40, "continuation_run_id": spec.run_id,
                                "parent_step": result["parent_step"], "arm": "Moderate", "budget": 30.0, "horizons": horizons,
                                "seed_set": result["seed_set"], "seeds": result["seeds"], "by_horizon": blocks,
                                "sr_fewshot": result["sr_fewshot"], "adapt_steps": result["adapt_steps"][30.0]})
    json.dumps(plain(result), sort_keys=True, allow_nan=False)


def test_fewshot_after_cut_3_reads_its_one_horizon(tmp_path, monkeypatch, answered) -> None:
    """Part 6.1 cut 3 keeps only the first horizon; censoring is then 200,001 (Q-adapt-censoring)."""
    Primitives(monkeypatch, cost=_satisfying_from(10**9))
    spec = manifest.apply_cuts([_fewshot(45.0)], manifest.CUTS[:3])[0]
    assert spec.params["horizons"] == [200_000] and spec.total_steps == 200_000
    result = SE.evaluate_fewshot(str(_run_dir(tmp_path, spec, [0, 200_000])), spec.to_dict())
    assert result["horizons"] == [200_000] and result["adapt_steps"] == {45.0: 200_001} and result["adapt_censored"]
    assert list(result["sr_fewshot"]) == ["45.0_200000"]


def test_fewshot_refuses_before_any_episode(tmp_path, monkeypatch, answered) -> None:
    stubs = Primitives(monkeypatch)
    spec = _fewshot()
    with pytest.raises(E.CheckpointInvalid, match="epoch-25.pt"):  # the off-grid horizon was not saved
        SE.evaluate_fewshot(str(_run_dir(tmp_path, spec, [0, 200_000, 1_000_000])), spec.to_dict())
    assert stubs.calls == []
    for params in ({**spec.params, "horizons": [300_000, 1_000_000]}, {**spec.params, "budget": 20.0},
                   {**spec.params, "horizons": []}, {**spec.params, "budget": "5"}):
        bad = manifest.RunSpec.from_dict({**spec.to_dict(), "params": params})
        with pytest.raises(E.EvaluationRefused):
            SE.evaluate_fewshot(str(tmp_path), bad.to_dict())
    with pytest.raises(E.EvaluationRefused):
        SE.evaluate_fewshot(str(tmp_path), _arm("Moderate").to_dict())  # a training run, not a continuation
    assert stubs.calls == []


# ---------------------------------------------------------------------------
# Contract 1 for plug-in study_b (pilot.contracts.evaluator_target)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arm", ["Moderate", "Continuous"])
def test_contract_1_of_a_study_b_run(tmp_path, monkeypatch, answered, arm) -> None:
    stubs = Primitives(monkeypatch)
    spec = next(s for s in manifest.pilot() if s.arm == "Moderate") if arm == "Moderate" else _arm(arm)
    directory = _run_dir(tmp_path, spec, contracts.expected_checkpoint_steps(spec))
    result = SE.evaluate_run(str(directory), spec.to_dict())
    contracts.validate_evaluation(plain(result), spec, directory)  # what the launcher checks
    from pilot import launch

    launch._check_evaluation_record(spec, {**plain(result), "eval_commit_hash": "a" * 40})  # the ledger's check too
    window = list(range(8_200_000, 10_000_001, 200_000))
    budgets = E.training_budget_schedule(spec)
    assert stubs.loads == window and sorted(result["selection"]) == window
    assert [c[0] for c in stubs.calls] == window + [spec.total_steps]
    assert all(c[1] == E.selection_seeds() and c[2] == budgets for c in stubs.calls[:-1])
    assert stubs.calls[-1][1] == E.measurement_seeds() and stubs.calls[-1][2] == budgets
    assert result["studyb_budgets"] == budgets and result["selection_seeds"] == E.selection_seeds()
    if arm == "Moderate":
        assert budgets[:4] == [10.0, 20.0, 40.0, 10.0]  # the arm's training budgets in turn (Q-studyb-eval)
    else:
        assert budgets[0] == 10.15 and budgets[-1] == 39.85  # midpoints of 100 equal parts of [10, 40]
    final = stubs.calls[-1]
    assert result["final_cost"] == pytest.approx(sum(stubs.cost(*x) for x in zip([final[0]] * 100, final[1], final[2])) / 100)
    assert result["final_selection_cost"] == result["selection"][spec.total_steps][0]
    assert result["episode_costs"]["final"] and len(result["episode_costs"]["selection"]) == 10
    json.dumps(plain(result), sort_keys=True, allow_nan=False)


def _first_episode_unstable(stubs: Primitives):
    """The stub's ``run_episodes`` whose first episode of every call was unstable and ran its set's first reserve
    seed with the same budget (Q-mujoco-exception; envs.evaluation.run_episodes)."""

    def run_episodes(policy, seeds, *, budgets=None, **kwargs):
        results = stubs.run_episodes(policy, seeds, budgets=budgets, **kwargs)
        reserve = E.reserve_seeds(seeds)[0]
        unstable = E.Replacement(seed=results[0].seed, reserve_seed=reserve, step=5, warning="mjWARN_BADQACC x1")
        results[0] = dataclasses.replace(results[0], seed=reserve, replaced=(unstable,))
        return results

    return run_episodes


def test_every_evaluation_reports_its_replacements_beside_the_planned_seeds(tmp_path, monkeypatch, answered) -> None:
    """Q-mujoco-exception (Table 9.1; the critic's addition: planned seed lists unchanged, replacements recorded
    separately): every result keeps the canonical ``seeds`` and reports, per evaluation call (a budget, a horizon, a
    checkpoint), the reserve seeds that replaced unstable episodes; contract 1 passes the launcher's and the ledger's
    checks, G4 the command line's."""
    stubs = Primitives(monkeypatch)
    monkeypatch.setattr(E, "run_episodes", _first_episode_unstable(stubs))
    spec = next(s for s in manifest.pilot() if s.arm == "Moderate")
    directory = _run_dir(tmp_path, spec, contracts.expected_checkpoint_steps(spec))
    replaced = {"slot_index": 0, "seed": 2_000_000, "reserve_seed": 2_050_000, "step": 5, "warning": "mjWARN_BADQACC x1"}
    budgets = SE.evaluate_training_budgets(str(directory), spec.to_dict())
    assert budgets["seeds"] == E.measurement_seeds()
    assert budgets["unstable_replacements"] == {b: [replaced] for b in (10.0, 20.0, 40.0)}
    _check_as_g4_measure(budgets, [10.0, 20.0, 40.0])
    zero_shot = SE.evaluate_zero_shot(str(directory), spec.to_dict())
    assert zero_shot["seeds"] == E.measurement_seeds() and zero_shot["unstable_replacements"][45.0] == [replaced]
    result = SE.evaluate_run(str(directory), spec.to_dict())
    assert result["selection_seeds"] == E.selection_seeds() and result["unstable_replacements"]["final"] == [replaced]
    assert {entries[0]["reserve_seed"] for entries in result["unstable_replacements"]["selection"].values()} == {1_050_000}
    contracts.validate_evaluation(plain(result), spec, directory)
    from pilot import launch

    launch._check_evaluation_record(spec, {**plain(result), "eval_commit_hash": "a" * 40})
    fewshot_spec = _fewshot(30.0)
    fewshot = SE.evaluate_fewshot(str(_run_dir(tmp_path / "f", fewshot_spec, [0, 200_000, 400_000, 500_000, 600_000,
                                                                                800_000, 1_000_000])),
                                  fewshot_spec.to_dict())
    assert fewshot["seeds"] == E.measurement_seeds()
    assert all(block["unstable_replacements"] == [replaced] for block in fewshot["by_horizon"].values())


def test_contract_1_refuses_missing_checkpoints_and_short_epochs(tmp_path, monkeypatch, answered) -> None:
    stubs = Primitives(monkeypatch)
    spec = _arm("Moderate")
    steps = contracts.expected_checkpoint_steps(spec)
    with pytest.raises(E.CheckpointInvalid, match="missing at steps \\[5000000\\]"):
        SE.evaluate_run(str(_run_dir(tmp_path / "a", spec, [s for s in steps if s != 5_000_000])), spec.to_dict())
    with pytest.raises(E.EvaluationRefused, match="steps_per_epoch 2000"):
        SE.evaluate_run(str(_run_dir(tmp_path / "b", spec, [0, 2_000], spe=2_000)), spec.to_dict())
    with pytest.raises(E.EvaluationRefused):
        SE.evaluate_run(str(tmp_path), _fewshot().to_dict())
    assert stubs.calls == []


@pytest.mark.parametrize("change", ["another run", "another arm's conditioning", "normalised costs"])
def test_contract_1_refuses_another_runs_directory_before_looking_for_checkpoints(tmp_path, monkeypatch, answered,
                                                                                 change) -> None:
    """A config.json that does not describe the spec is a refusal (EvaluationRefused: the run stays
    trained), even when a checkpoint is missing too, as in Role 2's envs.evaluation.evaluate_run; never
    CheckpointInvalid (a failure of the run's output)."""
    stubs = Primitives(monkeypatch)
    spec, other = _arm("Moderate"), _arm("Dense")
    steps = [s for s in contracts.expected_checkpoint_steps(spec) if s != spec.total_steps]  # the final one missing
    directory = _run_dir(tmp_path, spec, steps)
    config = json.loads((directory / "config.json").read_text())
    if change == "another run":
        config["pilot_cfgs"]["run_id"] = other.run_id
        words = "is not the spec's run_id"
    elif change == "another arm's conditioning":
        config["studyb_cfgs"] = _studyb_cfgs(other)
        words = "do not describe B-Moderate-s0"
    else:
        config["algo_cfgs"]["cost_normalize"] = True
        words = "cost_normalize is True"
    (directory / "config.json").write_text(json.dumps(config))
    with pytest.raises(E.EvaluationRefused, match=words):
        SE.evaluate_run(str(directory), spec.to_dict())
    assert stubs.loads == [] and stubs.calls == []
    (directory / "config.json").write_text(json.dumps(config | {
        "pilot_cfgs": {"run_id": spec.run_id}, "studyb_cfgs": _studyb_cfgs(spec),
        "algo_cfgs": {**config["algo_cfgs"], "cost_normalize": False}}), encoding="utf-8")
    with pytest.raises(E.CheckpointInvalid, match="missing at steps \\[10000000\\]"):  # its own config: the failure
        SE.evaluate_run(str(directory), spec.to_dict())


@pytest.mark.parametrize("name", ["evaluate_training_budgets", "evaluate_zero_shot", "evaluate_fewshot"])
def test_another_arms_directory_is_refused_before_its_checkpoints_are_looked_for(tmp_path, monkeypatch, answered,
                                                                                 name) -> None:
    """Another arm's studyb_cfgs and a missing checkpoint: EvaluationRefused, never CheckpointInvalid."""
    stubs = Primitives(monkeypatch)
    spec = _fewshot() if name == "evaluate_fewshot" else _arm("Moderate")
    directory = _run_dir(tmp_path, spec, [], studyb=_studyb_cfgs(_arm("Dense")))  # no checkpoint at all
    with pytest.raises(E.EvaluationRefused, match="do not describe"):
        getattr(SE, name)(str(directory), spec.to_dict())
    assert stubs.loads == [] and stubs.calls == []


def test_contract_1_refuses_a_spec_problem_before_loading_checkpoints(tmp_path, monkeypatch, answered) -> None:
    stubs = Primitives(monkeypatch)
    spec = _arm("Continuous")
    directory = _run_dir(tmp_path, spec, contracts.expected_checkpoint_steps(spec))
    bad = manifest.RunSpec.from_dict({**spec.to_dict(), "params": {}})  # no continuous_range
    with pytest.raises(E.EvaluationRefused, match="continuous_range"):  # envs.evaluation.training_budget_schedule's
        SE.evaluate_run(str(directory), bad.to_dict())
    assert stubs.loads == [] and stubs.calls == []


def test_contract_1_fails_a_broken_checkpoint_set_and_refuses_a_spec_too_short(tmp_path, monkeypatch, answered) -> None:
    """A stray off-grid checkpoint inside the final 2,000,000 steps is the run's saved output: CheckpointInvalid (a
    failure, eval_failed), as Role 2's envs.evaluation.evaluate_run; a total too short for the ten-checkpoint window is
    the spec's problem: EvaluationRefused."""
    stubs = Primitives(monkeypatch)
    spec = _arm("Moderate")
    steps = contracts.expected_checkpoint_steps(spec) + [9_900_000]  # epoch-495.pt, off the 200,000-step grid
    with pytest.raises(E.CheckpointInvalid, match="9900000.*a broken checkpoint set"):
        SE.evaluate_run(str(_run_dir(tmp_path / "a", spec, steps)), spec.to_dict())
    short = manifest.RunSpec.from_dict({**spec.to_dict(), "total_steps": 1_000_000})  # five checkpoints, on the grid
    with pytest.raises(E.EvaluationRefused, match="only 6 checkpoints") as refused:
        SE.evaluate_run(str(_run_dir(tmp_path / "b", short, contracts.expected_checkpoint_steps(short))), short.to_dict())
    assert not isinstance(refused.value, E.CheckpointInvalid)
    assert stubs.loads == [] and stubs.calls == []


def test_non_iterable_horizons_and_huge_numbers_are_value_errors(tmp_path, monkeypatch, answered) -> None:
    with pytest.raises(ValueError, match="horizons must be a sequence"):
        SE.censored_steps(5)
    for call in (lambda: SE.satisfaction_rate([10**400], 5), lambda: SE.distance(10**400, "Moderate"),
                 lambda: SE.adaptation_steps({200_000: 10**400}, [200_000])):
        with pytest.raises(ValueError):
            call()
    stubs = Primitives(monkeypatch)
    spec = _fewshot()
    for params in ({**spec.params, "horizons": 5}, {**spec.params, "budget": 10**400}):
        bad = manifest.RunSpec.from_dict({**spec.to_dict(), "params": params})
        with pytest.raises(E.EvaluationRefused):
            SE.evaluate_fewshot(str(tmp_path), bad.to_dict())
    assert stubs.calls == []

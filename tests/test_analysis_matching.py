"""Part 4.1 rules 1 to 7 (analysis/matching.py; contract 4; First Tasks Role 4 tests)."""

from __future__ import annotations

import copy
import json
import math
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import test_analysis_synthetic as syn  # noqa: E402

from analysis import matching as M  # noqa: E402
from configs import registered as R  # noqa: E402
from pilot import enrichment  # noqa: E402
from pilot.contracts import ContractError  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Rule 1 (contract 4)
# ---------------------------------------------------------------------------


def _window(costs: list, *, start: int = 8_200_000, extra: int = 41) -> list[dict]:
    """A run's checkpoint dicts: ``extra`` (<= 41) checkpoints before the window (no selection cost), then the window."""
    assert extra * R.CHECKPOINT_INTERVAL_STEPS <= start
    records = [{"path": f"p{i}", "step": i * R.CHECKPOINT_INTERVAL_STEPS, "training_cost": math.nan if i == 0 else 30.0,
                "training_return": 0.0, "dormant": None, "rank": None, "norm": None, "selection_cost": None,
                "selection_return": None, "multiplier": 0.001} for i in range(extra)]
    for j, cost in enumerate(costs):
        records.append({"path": f"w{j}", "step": start + j * R.CHECKPOINT_INTERVAL_STEPS, "training_cost": 30.0,
                        "training_return": 0.0, "dormant": 0.1, "rank": 3.0, "norm": 1.0, "selection_cost": cost,
                        "selection_return": 1.0, "multiplier": 0.5})
    return records


def test_select_checkpoint_equals_rule_one_step_on_random_windows() -> None:
    rng = random.Random(20260930)
    for trial in range(10_000):  # 10,000 random windows, float32 harness values included
        kind = trial % 4
        if kind == 0:  # 100-episode means: multiples of 0.01
            costs = [rng.randint(0, 6000) / 100 for _ in range(10)]
        elif kind == 1:  # float32 harness noise
            costs = [float(np.float32(rng.randint(0, 6000) / 100)) for _ in range(10)]
        elif kind == 2:  # deliberate ties around 25 (25 +/- x), with noise from arithmetic
            x = rng.randint(0, 800) / 100
            costs = [rng.choice([25 + x, 25 - x, (25 * 100 + x * 100) / 100, 25.0 - x + 0.0]) for _ in range(10)]
        else:  # few distinct values: many exact ties
            costs = [rng.choice([24.99, 25.01, 17.99, 32.01, 25.0]) for _ in range(10)]
        records = _window(costs, extra=rng.randint(0, 41))
        rng.shuffle(records)  # the ledger order is not relied on
        assert M.select_checkpoint(copy.deepcopy(records)) == enrichment.rule_one_step(copy.deepcopy(records))


def test_a_tie_goes_to_the_later_checkpoint() -> None:
    for first, second in ((17.99, 32.01), (32.01, 17.99)):
        costs = [40.0] * 10
        costs[6], costs[8] = first, second  # 9,400,000 and 9,800,000
        assert abs(32.01 - 25) != abs(17.99 - 25)  # floating point: the tie exists only after rounding
        assert M.select_checkpoint(_window(costs)) == 9_800_000 == enrichment.rule_one_step(_window(costs))
    exact = [25.0] * 10
    assert M.select_checkpoint(_window(exact)) == 8_200_000 + 9 * R.CHECKPOINT_INTERVAL_STEPS


def test_selection_uses_the_selection_set_and_nothing_else() -> None:
    costs = [30.0, 29.0, 26.0, 40.0, 50.0, 24.5, 60.0, 33.0, 31.0, 28.0]
    base = _window(costs)
    chosen = M.select_checkpoint(base)
    assert chosen == 8_200_000 + 5 * R.CHECKPOINT_INTERVAL_STEPS  # |24.5 - 25| is the least
    noisy = copy.deepcopy(base)
    for r in noisy:  # every other field changed: training cost, multiplier, metrics, returns
        r.update(training_cost=25.0, multiplier=9.0, dormant=0.9, rank=1.0, selection_return=-5.0, training_return=3.0)
    assert M.select_checkpoint(noisy) == chosen
    before = copy.deepcopy(base)
    M.select_checkpoint(base)
    assert base == before or json.dumps(base, default=str) == json.dumps(before, default=str)  # not mutated
    assert type(M.select_checkpoint(base)) is int
    assert M.select_checkpoint(base) == M.select_checkpoint(base)  # applying the rule twice gives the same selection


def test_the_target_is_the_budget() -> None:
    # a window whose costs sit near 20: rule 1 chooses the one nearest 25, not the arm's typical cost
    costs = [20.0, 20.1, 19.9, 20.0, 24.0, 20.0, 20.2, 19.8, 20.0, 20.0]
    assert M.select_checkpoint(_window(costs)) == 8_200_000 + 4 * R.CHECKPOINT_INTERVAL_STEPS


@pytest.mark.parametrize("bad", [None, math.nan, math.inf, -math.inf, "25", True])
def test_a_bad_window_cost_is_refused_not_skipped(bad) -> None:
    costs = [30.0] * 10
    costs[3] = bad
    with pytest.raises(ValueError):
        M.select_checkpoint(_window(costs))
    if bad is not True and bad != "25":  # the pipeline refuses the same inputs
        with pytest.raises((ContractError, TypeError, ValueError)):
            enrichment.rule_one_step(_window(costs))


def test_costs_outside_the_window_are_ignored_and_bad_windows_refused() -> None:
    records = _window([30.0] * 9 + [25.0])
    records[0]["selection_cost"] = math.nan  # step 0: outside the window, and training_cost NaN too
    records[1]["selection_cost"] = 25.0  # a cost outside the window never competes
    assert M.select_checkpoint(records) == 8_200_000 + 9 * R.CHECKPOINT_INTERVAL_STEPS
    with pytest.raises(ValueError, match="last 10"):
        M.select_checkpoint(_window([25.0] * 9, extra=0))
    dup = _window([25.0] * 10, extra=0)
    dup[1]["step"] = dup[0]["step"]
    with pytest.raises(ValueError, match="twice"):
        M.select_checkpoint(dup)
    frac = _window([25.0] * 10, extra=0)
    frac[0]["step"] = 1.5
    with pytest.raises(ValueError, match="integer"):
        M.select_checkpoint(frac)


def test_tie_decimals_are_the_pipelines() -> None:
    assert M.TIE_DECIMALS == enrichment.TIE_DECIMALS


def test_apply_selection_uses_this_selector_end_to_end(tmp_path) -> None:
    from results.ledger_schema import LedgerRow, append_to_ledger, load_ledger_as_rows

    spec = syn.specs(groups=("main",))[0]
    row = syn.study_a_row(spec, cost=25.0, gaps={}, selection_step=spec.total_steps - 3 * R.CHECKPOINT_INTERVAL_STEPS)
    for key in ("matched_checkpoint_path", "matched_checkpoint_step", "training_age", "selection_cost_at_match",
                "measurement_cost", "lambda_at_selection"):
        row[key] = None
    ledger = tmp_path / "ledger.parquet"
    append_to_ledger(LedgerRow(**row), ledger)
    assert enrichment.apply_selection(ledger) == [spec.run_id]  # no selector argument: analysis.matching is loaded
    written = load_ledger_as_rows(ledger)[0]
    assert written.matched_checkpoint_step == spec.total_steps - 3 * R.CHECKPOINT_INTERVAL_STEPS == written.training_age


def test_importing_matching_pulls_in_no_heavy_or_pipeline_module() -> None:
    code = ("import sys; import analysis.matching; "
            "bad = sorted(m for m in sys.modules if m.split('.')[0] in "
            "('numpy', 'pandas', 'scipy', 'torch', 'pilot', 'pyarrow', 'pydantic', 'omnisafe')); print(bad)")
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


# ---------------------------------------------------------------------------
# Rules 2 to 7
# ---------------------------------------------------------------------------


def _rows(ref_costs: list[float], arms: dict, *, task: str = R.PRIMARY_TASK, prefix: str = "") -> list[dict]:
    """Minimal ledger dicts: an N = 0 arm and late arms {(N, shape, control): costs}."""
    rows = []
    for seed, cost in enumerate(ref_costs):
        rows.append({"run_id": f"{prefix}A-{task[6:-3]}-N0.00-s{seed}", "study": "A", "task": task, "arm": "N0.00",
                     "N": 0.0, "onset_shape": None, "step_matching": None, "treatment": None, "controller_variant": None,
                     "seed": seed, "completed": True, "measurement_cost": cost, "selection_cost_at_match": 25.0})
    for (N, shape, control, *rest), costs in arms.items():
        treatment = rest[0] if rest else None
        arm = f"N{N:.2f}-{shape}-{control.split('_')[0]}" + (f"-{treatment}" if treatment else "")
        for seed, cost in enumerate(costs):
            rows.append({"run_id": f"{prefix}A-{task[6:-3]}-{arm}-s{seed}", "study": "A", "task": task, "arm": arm,
                         "N": N, "onset_shape": shape, "step_matching": control, "treatment": treatment,
                         "controller_variant": None, "seed": seed, "completed": True, "measurement_cost": cost,
                         "selection_cost_at_match": 25.0})
    return rows


def test_first_tasks_synthetic_ledger() -> None:
    """First Tasks Role 4 step 5: a reference near 25, a late arm that matches, one that misses by 4, a task whose
    reference spread is 12, and a tie between two checkpoints (the tie: test_a_tie_goes_to_the_later_checkpoint)."""
    rows = _rows([24.0, 25.0, 26.0, 24.5, 25.5], {(0.5, "abrupt", "total_steps"): [25.5, 26.0, 25.0, 26.5, 25.0],
                                                 (0.5, "ramp", "total_steps"): [29.0] * 5})
    spread = [25.0 - 12 * math.sqrt(2), 25.0 + 12 * math.sqrt(2), 25.0, 25.0, 25.0]  # sample SD 12 (> 10)
    rows += _rows(spread, {(0.5, "abrupt", "total_steps"): [25.0] * 5}, task="SafetyCarGoal1-v0")
    result = M.apply_rules(rows)
    point = result[R.PRIMARY_TASK]
    assert point.feasible is True and point.reference_cost == 25.0
    status = {a.arm: a.status for a in point.arms}
    assert status == {"N0.50-abrupt-total": "matched", "N0.50-ramp-total": "unmatched_above"}  # off by 4
    car = result["SafetyCarGoal1-v0"]
    assert car.reference_sd == pytest.approx(12.0) and car.sd_ok is False and car.feasible is False
    assert {a.status for a in car.arms} == {M.STATUS_INFEASIBLE_SD}
    flags = M.matching_flags(result)
    assert flags["A-PointGoal1-N0.50-abrupt-total-s0"] == {"matched": True, "infeasible": False}
    assert flags["A-PointGoal1-N0.50-ramp-total-s3"] == {"matched": False, "infeasible": False}
    assert flags["A-CarGoal1-N0.50-abrupt-total-s0"] == {"matched": False, "infeasible": True}
    assert flags["A-CarGoal1-N0.00-s0"] == {"matched": False, "infeasible": True}
    assert flags["A-PointGoal1-N0.00-s0"] == {"matched": True, "infeasible": False}


def test_boundaries_are_inclusive_after_rounding() -> None:
    # |diff| exactly 2.5 is matched; reference exactly 27.5 is feasible; SD exactly 10 is feasible
    rows = _rows([27.5] * 5, {(0.5, "abrupt", "total_steps"): [25.0] * 5, (0.5, "ramp", "total_steps"): [24.99] * 5})
    task = M.apply_rules(rows)[R.PRIMARY_TASK]
    assert task.budget_ok is True and {a.arm: a.status for a in task.arms} == {
        "N0.50-abrupt-total": "matched", "N0.50-ramp-total": "unmatched_below"}
    over = M.apply_rules(_rows([27.51] * 5, {}))[R.PRIMARY_TASK]
    assert over.budget_ok is False and over.feasible is False and "rule 3" in over.reason
    sd10 = [25.0 - 10 * math.sqrt(2), 25.0 + 10 * math.sqrt(2), 25.0, 25.0, 25.0]  # sample SD sqrt(400 / 4) = 10
    assert M.reference_sd(sd10) == pytest.approx(10.0)
    assert M.apply_rules(_rows(sd10, {}))[R.PRIMARY_TASK].sd_ok is True
    sd_over = [25.0 - 10.001 * math.sqrt(2), 25.0 + 10.001 * math.sqrt(2), 25.0, 25.0, 25.0]
    assert M.apply_rules(_rows(sd_over, {}))[R.PRIMARY_TASK].sd_ok is False


def test_rule_3_is_decided_before_the_reference_sd() -> None:
    """Rule 3 does not need the SD: a one-seed reference above d + tolerance is infeasible, not incomplete."""
    task = M.apply_rules(_rows([40.0], {(0.5, "abrupt", "total_steps"): [25.0]}))[R.PRIMARY_TASK]
    assert task.budget_ok is False and task.sd_ok is None and task.feasible is False and "rule 3" in task.reason
    assert task.reference.status == M.STATUS_REFERENCE and task.reference.matched is False
    assert {a.status for a in task.arms} == {M.STATUS_INFEASIBLE_BUDGET}
    within = M.apply_rules(_rows([25.0], {}))[R.PRIMARY_TASK]
    assert within.budget_ok is True and within.feasible is None and "two seeds" in within.reason


def test_rounding_absorbs_floating_point_noise_and_the_other_reading_does_not() -> None:
    diff = 4.15 - 1.65  # two 100-episode means whose difference is 2.5 up to floating-point noise
    assert diff > 2.5 and round(diff, 4) == 2.5
    assert M.within_tolerance(4.15, 1.65, R.MATCH_TOLERANCE) is True
    assert M.within_tolerance(4.15, 1.65, R.MATCH_TOLERANCE, M.ALTERNATIVE_ARITHMETIC) is False
    assert M.within_tolerance(27.5001, 25.0, R.MATCH_TOLERANCE) is False
    assert M.budget_feasible(27.5, R.MATCH_TOLERANCE, M.ALTERNATIVE_ARITHMETIC) is False  # strict
    assert M.reference_sd([1.0, 2.0, 3.0], M.ALTERNATIVE_ARITHMETIC) == pytest.approx(math.sqrt(2 / 3))


def test_the_rule_functions() -> None:
    assert M.reference_cost([24.0, 26.0]) == 25.0
    assert M.matched_cost([25.0, 26.0]) == 25.5
    assert M.reference_sd([25.0]) is None
    assert M.reference_sd([24.0, 26.0]) == pytest.approx(math.sqrt(2))  # n - 1
    assert M.spread_feasible(10.0) and not M.spread_feasible(10.0001)
    assert M.unmatched_status(30.0, 25.0) == M.STATUS_UNMATCHED_ABOVE
    assert M.unmatched_status(20.0, 25.0) == M.STATUS_UNMATCHED_BELOW
    assert M.check_tolerance(R.SENSITIVITY_TOLERANCE) == R.SENSITIVITY_TOLERANCE  # rule 7 (a)
    assert M.budget_feasible(29.9, R.SENSITIVITY_TOLERANCE) and not M.budget_feasible(29.9, R.MATCH_TOLERANCE)
    assert M.is_reference_row({"study": "A", "N": 0.0, "treatment": None, "controller_variant": None})
    assert not M.is_reference_row({"study": "A", "N": 0.5, "treatment": None, "controller_variant": None})
    with pytest.raises(ValueError):
        M.reference_cost([])


def test_the_tolerance_is_never_widened() -> None:
    rows = _rows([25.0] * 5, {(0.5, "abrupt", "total_steps"): [29.0] * 5})
    for tolerance in (3.0, 2.4, 10.0):
        with pytest.raises(ValueError, match="not registered"):
            M.apply_rules(rows, tolerance=tolerance)
        with pytest.raises(ValueError):
            M.match_arms(rows, tolerance=tolerance, seed_targets={})
    # rule 7 (a): the arm off by 4 matches at 5.0, and rule 3 reads d + 5.0
    arm = M.apply_rules(rows, tolerance=R.SENSITIVITY_TOLERANCE)[R.PRIMARY_TASK].arms[0]
    assert arm.status == M.STATUS_MATCHED and M.apply_rules(rows)[R.PRIMARY_TASK].arms[0].status == M.STATUS_UNMATCHED_ABOVE
    ref29 = _rows([29.0] * 5, {})
    assert M.apply_rules(ref29)[R.PRIMARY_TASK].feasible is False
    assert M.apply_rules(ref29, tolerance=R.SENSITIVITY_TOLERANCE)[R.PRIMARY_TASK].feasible is True


def test_one_reference_per_task_serves_both_controls_and_nothing_crosses_tasks() -> None:
    rows = _rows([25.0] * 5, {(0.5, "abrupt", "total_steps"): [26.0] * 5, (0.5, "abrupt", "constrained_steps"): [24.0] * 5})
    rows += _rows([20.0] * 5, {(0.5, "abrupt", "total_steps"): [26.0] * 5}, task="SafetyPointButton1-v0")
    result = M.apply_rules(rows)
    point = result[R.PRIMARY_TASK]
    assert {a.step_matching: a.difference for a in point.arms} == {"total_steps": 1.0, "constrained_steps": -1.0}
    button = result["SafetyPointButton1-v0"]
    assert button.reference_cost == 20.0 and button.arms[0].status == M.STATUS_UNMATCHED_ABOVE  # against its own N = 0


def test_nothing_is_dropped() -> None:
    rows = _rows([25.0] * 5, {(0.5, "abrupt", "total_steps"): [26.0] * 5})
    rows[0]["completed"] = False  # a crashed run leaves by Part 5.6 only
    rows.append(dict(rows[1], run_id="A-PointGoal1-N0.00-s5", seed=5))  # its replacement seed
    task = M.apply_rules(rows)[R.PRIMARY_TASK]
    assert task.reference.seeds == (1, 2, 3, 4, 5)
    rows[6]["measurement_cost"] = None  # a missing enrichment field: incomplete, not dropped
    task = M.apply_rules(rows)[R.PRIMARY_TASK]
    assert task.arms[0].status == M.STATUS_INCOMPLETE and task.arms[0].matched is None
    assert task.arms[0].seeds == (0, 1, 2, 3, 4)
    rows[1]["measurement_cost"] = None
    task = M.apply_rules(rows)[R.PRIMARY_TASK]
    assert task.feasible is None and "missing" in task.reason
    # every completed run is kept whatever its numbers
    wild = _rows([25.0, 25.0, 25.0, 25.0, 25.0], {(0.5, "abrupt", "total_steps"): [0.0, 50.0, 25.0, 25.0, 25.0]})
    assert M.apply_rules(wild)[R.PRIMARY_TASK].arms[0].seeds == (0, 1, 2, 3, 4)


def test_match_arms_needs_complete_arms() -> None:
    rows = _rows([25.0] * 5, {(0.5, "abrupt", "total_steps"): [26.0] * 5, (0.5, "ramp", "total_steps"): [31.0] * 5})
    targets = {"A-PointGoal1-N0.00": 5, "A-PointGoal1-N0.50-abrupt-total": 5, "A-PointGoal1-N0.50-ramp-total": 5}
    flags = M.match_arms(rows, tolerance=R.MATCH_TOLERANCE, seed_targets=targets)
    assert len(flags) == 15
    assert flags["A-PointGoal1-N0.50-abrupt-total-s2"] == {"matched": True, "infeasible": False}
    assert flags["A-PointGoal1-N0.50-ramp-total-s2"] == {"matched": False, "infeasible": False}
    with pytest.raises(ValueError, match="no seed target"):
        M.match_arms(rows, tolerance=R.MATCH_TOLERANCE, seed_targets={k: 5 for k in list(targets)[:2]})
    with pytest.raises(ValueError, match="incomplete"):
        M.match_arms(rows, tolerance=R.MATCH_TOLERANCE, seed_targets={**targets, "A-PointGoal1-N0.00": 6})
    with pytest.raises(ValueError, match="incomplete"):
        M.match_arms(rows, tolerance=R.MATCH_TOLERANCE, seed_targets={**targets, "A-PointGoal1-N0.25-abrupt-total": 5})
    with pytest.raises(ValueError, match="positive integer"):
        M.match_arms(rows, tolerance=R.MATCH_TOLERANCE, seed_targets={**targets, "A-PointGoal1-N0.00": True})
    surplus = rows + [dict(rows[0], run_id="A-PointGoal1-N0.00-s5", seed=5, measurement_cost=24.0)]
    flags = M.match_arms(surplus, tolerance=R.MATCH_TOLERANCE, seed_targets={**targets, "A-PointGoal1-N0.00": 6})
    assert flags["A-PointGoal1-N0.00-s5"] == {"matched": True, "infeasible": False}  # every completed seed counts
    missing = copy.deepcopy(rows)
    missing[7]["measurement_cost"] = None
    with pytest.raises(ValueError, match="cannot be decided"):
        M.match_arms(missing, tolerance=R.MATCH_TOLERANCE, seed_targets=targets)
    infeasible = _rows([30.0] * 5, {(0.5, "abrupt", "total_steps"): [30.0] * 5})
    flags = M.match_arms(infeasible, tolerance=R.MATCH_TOLERANCE,
                         seed_targets={"A-PointGoal1-N0.00": 5, "A-PointGoal1-N0.50-abrupt-total": 5})
    assert all(f == {"matched": False, "infeasible": True} for f in flags.values())


# True and the strings reach usable_cost unconverted (the default cost_of passes the ledger value as stored)
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, True, "25", "abc"])
def test_a_non_finite_cost_is_incomplete_not_a_finding(bad) -> None:
    """The frozen LedgerRow accepts NaN and infinity in a float field: rules 2 to 6 read such a cost as missing (a
    data defect), never as an unmatched arm (NaN compares false with everything) and never inside statistics."""
    rows = _rows([25.0] * 5, {(0.5, "abrupt", "total_steps"): [26.0] * 5})
    targets = {"A-PointGoal1-N0.00": 5, "A-PointGoal1-N0.50-abrupt-total": 5}
    arm_seed = copy.deepcopy(rows)
    arm_seed[7]["measurement_cost"] = bad  # seed 2 of the late arm
    task = M.apply_rules(arm_seed)[R.PRIMARY_TASK]
    assert task.feasible is True and task.arms[0].status == M.STATUS_INCOMPLETE and task.arms[0].matched is None
    assert M.matched_arm_ids({R.PRIMARY_TASK: task}) == {"A-PointGoal1-N0.00"}
    with pytest.raises(ValueError, match=r"not finite for \['A-PointGoal1-N0.50-abrupt-total-s2'\]"):
        M.match_arms(arm_seed, tolerance=R.MATCH_TOLERANCE, seed_targets=targets)
    ref_seed = copy.deepcopy(rows)
    ref_seed[1]["measurement_cost"] = bad  # seed 1 of the reference
    task = M.apply_rules(ref_seed)[R.PRIMARY_TASK]
    assert task.feasible is None and task.reference.status == M.STATUS_INCOMPLETE
    assert "A-PointGoal1-N0.00-s1" in task.reason and "not finite" in task.reason
    assert all(a.status == M.STATUS_INCOMPLETE for a in task.arms)
    with pytest.raises(ValueError, match="cannot be decided"):
        M.match_arms(ref_seed, tolerance=R.MATCH_TOLERANCE, seed_targets=targets)
    assert not any(M.usable_cost(v) for v in (None, True, "25", bad)) and M.usable_cost(25) and M.usable_cost(25.0)


def test_pilot_and_registered_rows_are_never_matched_together() -> None:
    rows = _rows([25.0] * 3, {}) + _rows([25.0] * 3, {}, prefix="P-")
    with pytest.raises(ValueError, match="populations"):
        M.apply_rules(rows)
    assert M.population("P1-A-PointGoal1-N0.00-s0") == "P1-" and M.population("B-Dense-s0") == ""
    with pytest.raises(ValueError):
        M.population("X-foo-s0")
    with pytest.raises(ValueError):
        M.arm_id({"run_id": "A-PointGoal1-N0.00-s1", "seed": 2})


def test_matching_on_the_selection_set_is_the_other_reading() -> None:
    rows = _rows([25.0] * 5, {(0.5, "abrupt", "total_steps"): [28.0] * 5})
    for r in rows:
        r["selection_cost_at_match"] = 25.0
    measured = M.apply_rules(rows)[R.PRIMARY_TASK].arms[0]
    selected = M.apply_rules(rows, cost_of=lambda r: r[M.SELECTION_COST_FIELD])[R.PRIMARY_TASK].arms[0]
    assert measured.status == M.STATUS_UNMATCHED_ABOVE and selected.status == M.STATUS_MATCHED
    assert M.COST_FIELD == "measurement_cost"
    from pilot import go_decision

    assert go_decision.COST_FIELD == M.COST_FIELD


def test_the_registered_design_matches_with_the_synthetic_world() -> None:
    rows = syn.a_world("supported")
    result = M.apply_rules(rows)
    assert result[R.PRIMARY_TASK].feasible is True
    assert all(a.status == M.STATUS_MATCHED for a in result[R.PRIMARY_TASK].arms)
    assert M.matched_arm_ids(result) >= {"A-PointGoal1-N0.00", "A-PointGoal1-N0.50-abrupt-total"}


def test_an_off_grid_run_is_selected_on_its_end_relative_window_by_every_rule_one_consumer() -> None:
    """Q-selection-window: a constrained-steps N = 0.10 run (11,120,000 steps) saves the grid and the
    end-relative window; rule 1 (select_checkpoint, the pipeline's rule_one_step and selection_fields,
    and analysis.data's minimum window cost) reads the same ten checkpoints as contract 1 evaluated."""
    from analysis import data as D
    from pilot import contracts

    total = 11_120_000
    window = contracts.end_relative_window(total)
    # plus two end-relative grid steps far outside the window (total - 50 and - 49 x 200,000): never selected
    steps = sorted(set(range(0, total, R.CHECKPOINT_INTERVAL_STEPS)) | set(window) | {1_120_000, 1_320_000})
    costs = {s: 24.0 - i for i, s in enumerate(window)}
    costs[window[3]] = 25.1  # the closest to 25 is the fourth of the end-relative window
    records = [{"path": f"p{s}", "step": s, "selection_cost": costs.get(s)} for s in steps]
    assert [c["step"] for c in M.selection_window(records)] == window
    assert sorted(records, key=lambda c: c["step"])[-10]["step"] != window[0]  # "last ten by step" would differ
    assert M.select_checkpoint(records) == window[3] == enrichment.rule_one_step(records)
    assert enrichment.selection_fields({"run_id": "X", "checkpoints": records}, window[3])["matched_checkpoint_step"] == window[3]
    assert D._min_window_cost(records) == min(costs.values())
    with pytest.raises(ValueError, match="Q-selection-window"):
        M.select_checkpoint([r for r in records if r["step"] != window[0]])
    with pytest.raises(ContractError, match="Q-selection-window"):
        enrichment.rule_one_step([r for r in records if r["step"] != window[0]])


def test_a_missing_grid_checkpoint_in_the_final_steps_is_refused_not_reached_past() -> None:
    """Rule 1's window is the final 2,000,000 steps: a missing grid checkpoint there is refused, never replaced
    by an older one (the 8,000,000 checkpoint lies outside the window (8,000,000, 10,000,000])."""
    records = _window([40.0] * 10, start=8_200_000, extra=41)
    for r in records:
        if r["step"] == 8_000_000:
            r["selection_cost"] = 25.0
    gap = [r for r in records if r["step"] != 9_000_000]
    assert 8_000_000 not in [c["step"] for c in M.selection_window(gap)]
    assert len(M.selection_window(gap)) == R.SELECTION_WINDOW_CHECKPOINTS - 1
    with pytest.raises(ValueError, match="lacks checkpoints"):
        M.select_checkpoint(gap)
    assert M.select_checkpoint(records) != 8_000_000  # complete: the window is 8,200,000 to 10,000,000


def test_every_cited_quotation_in_matching_is_in_the_pre_registration() -> None:
    """Final review round 3: selection_window quoted Part 4.1 rule 1 without its '; decision' mark (not verbatim).
    The quotation checker of tests/test_core_quotes.py now reads this module too."""
    import importlib.util

    path = REPO / "tests" / "test_core_quotes.py"
    if not path.exists() or not (REPO / "prereg" / "Preregistration.docx").exists():
        pytest.skip("the quotation checker or the pre-registration is not in this checkout")
    module_spec = importlib.util.spec_from_file_location("_matching_quote_checker", path)
    checker = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(checker)
    quotes = checker.cited_quotes((REPO / "analysis" / "matching.py").read_text(encoding="utf-8"))
    assert "the run's last ten checkpoints (the final 2,000,000 steps; decision)" in quotes
    assert not {q: m for q in quotes if (m := checker.missing_pieces(q))}
    assert checker.missing_pieces("the run's last ten checkpoints (the final 2,000,000 steps)")  # the old misquotation

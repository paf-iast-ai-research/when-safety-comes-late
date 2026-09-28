"""Compute budget, surplus seeds (Part 5.5) and the go or no-go rule (Part 6; equation 12)."""

from __future__ import annotations

import math
import statistics

import pytest

from configs import registered as R
from pilot import budget, go_decision


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


def test_registered_requirement_is_the_part6_formula() -> None:
    assert budget.registered_run_equivalents() == 484
    assert budget.registered_requirement(20.0, 10) == pytest.approx(20.0 / 10 * 484)
    with pytest.raises(ValueError):
        budget.registered_requirement(20.0, 0)


def test_corrected_run_equivalents_count_what_484_leaves_out() -> None:
    c = budget.corrected_run_equivalents()
    assert c["pid"] == pytest.approx(15) and c["controller"] == pytest.approx(90)
    assert c["study_b"] + c["study_b_fewshot"] == pytest.approx(49)
    # constrained-steps arms: per task and seed 2 shapes x (1.112 + 1.334 + 2.0) with the proposed rounding
    assert c["main"] == pytest.approx(3 * 5 * (1 + 6 + 2 * (1.112 + 1.334 + 2.0)))
    assert c["treatment"] == pytest.approx(135 + 3 * 5 * (0.10 + 0.25 + 0.50))
    assert c["battery_continuations"] == pytest.approx(435 * 0.2)
    assert c["total"] == pytest.approx(627.13)
    assert c["total"] - c["battery_continuations"] == pytest.approx(540.13)


def test_surplus_plan_follows_part_5_5() -> None:
    primary = budget.primary_comparison_specs(5)
    # registered units: Study A runs count 1 each (5 arms), Study B 7 x (1 + 4 x 0.1) = 9.8
    assert budget.registered_units(primary) == pytest.approx(5 + 9.8)
    # corrected: training steps (N=0.50 constrained arms train 2T) plus battery continuations 5 x 0.2
    assert budget.corrected_units(primary) == pytest.approx(7 + 9.8 + 1.0)
    none = budget.surplus_plan(machine_hours_allocated=1000, hours_per_run_equivalent=10, concurrent_runs=4)
    assert none.extra_seeds == 0 and none.seeds_per_primary_arm == 5  # capacity 400 < 484
    some = budget.surplus_plan(machine_hours_allocated=(484 + 2.5 * 14.8) * 10 / 4, hours_per_run_equivalent=10, concurrent_runs=4)
    assert some.extra_seeds == 2 and some.added_seeds == (5, 6)
    assert some.corrected_extra_seeds == 0 and "corrected totals allow only 0" in some.warning
    capped = budget.surplus_plan(machine_hours_allocated=10**7, hours_per_run_equivalent=10, concurrent_runs=4)
    assert capped.seeds_per_primary_arm == R.MAX_SEEDS_PER_ARM and capped.added_seeds[-1] == 11
    assert "corrected" not in capped.warning and "Q-surplus-arm-set is open" in capped.warning
    assert "Q-surplus-arm-set" not in none.warning  # no surplus, nothing waits
    for bad in ({"machine_hours_allocated": 100, "hours_per_run_equivalent": 0}, {"machine_hours_allocated": 0, "hours_per_run_equivalent": 10},
                {"machine_hours_allocated": -1, "hours_per_run_equivalent": 10}, {"machine_hours_allocated": 100, "hours_per_run_equivalent": float("nan")}):
        with pytest.raises(ValueError):
            budget.surplus_plan(concurrent_runs=4, **bad)
    for bad_concurrency in (float("nan"), 4.0, True):
        with pytest.raises(ValueError, match="concurrent_runs must be an integer"):
            budget.surplus_plan(machine_hours_allocated=1000, hours_per_run_equivalent=10, concurrent_runs=bad_concurrency)


def test_surplus_seeds_of_the_ramp_arms_wait_on_the_arm_set_question(monkeypatch) -> None:
    primary = budget.primary_comparison_specs(5)
    waiting = {s.arm for s in primary if "Q-surplus-arm-set" in s.pending}
    assert waiting == {"N0.50-ramp-constrained", "N0.50-ramp-total"}
    assert all(s.seed == 5 and s.run_id.endswith("-s5") for s in primary)
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset({"Q-surplus-arm-set"}))
    capped = budget.surplus_plan(machine_hours_allocated=10**7, hours_per_run_equivalent=10, concurrent_runs=4)
    assert capped.warning == ""


def test_allocation_is_recorded_once(tmp_path) -> None:
    path = tmp_path / "allocation.json"
    rec = budget.record_allocation(5000, "the group", "2026-10-01", path)
    assert budget.load_allocation(path)["machine_hours_allocated"] == 5000.0 == rec["machine_hours_allocated"]
    with pytest.raises(FileExistsError):
        budget.record_allocation(6000, "the group", "2026-10-02", path)
    with pytest.raises(ValueError):
        budget.record_allocation(-1, "x", "2026-10-01", tmp_path / "other.json")
    with pytest.raises(ValueError):
        budget.record_allocation(5000, " ", "2026-10-01", tmp_path / "other.json")
    with pytest.raises(ValueError):
        budget.record_allocation(5000, "the group", "1 October", tmp_path / "other.json")
    assert not (tmp_path / "other.json").exists()


def test_allocation_contained(tmp_path, monkeypatch) -> None:
    root = tmp_path.resolve()  # a stand-in repository root: nothing is written into the working tree
    monkeypatch.setattr(budget.provenance, "REPO_ROOT", root)
    (root / "pilot").mkdir()
    path = root / "pilot" / "allocation.json"
    rel = str(path.relative_to(root))
    committed = {"HEAD": b"A", "c1": b"A", "c2": b"A", "c3": None}
    monkeypatch.setattr(budget.provenance, "file_committed_at", lambda commit, name: committed[commit] if name == rel else None)
    monkeypatch.setattr(budget.provenance, "dirty_paths", lambda: set())
    ok, note = budget.allocation_contained([{"commit_hash": "c1"}], path)
    assert ok is False and "does not exist" in note
    path.write_bytes(b"A")
    assert budget.allocation_contained([{"commit_hash": "c1"}, {"commit_hash": "c2"}], path)[0] is True
    assert budget.allocation_contained([], path) == (False, "no pilot run records a launch commit yet")
    ok, note = budget.allocation_contained([{"commit_hash": "c1"}, {"commit_hash": "c3"}], path)
    assert ok is False and "c3" in note
    path.write_bytes(b"B")  # edited after the pilot runs were launched
    assert budget.allocation_contained([{"commit_hash": "c1"}], path)[0] is False
    monkeypatch.setattr(budget.provenance, "dirty_paths", lambda: {rel})
    ok, note = budget.allocation_contained([{"commit_hash": "c1"}], path)
    assert ok is False and "uncommitted" in note


# ---------------------------------------------------------------------------
# Go or no-go
# ---------------------------------------------------------------------------


def row(N: float, seed: int, cost: float, gap_h: float | None = None, gap_d: float | None = 1.0, final: float | None = None) -> dict:
    return {
        "run_id": f"P-A-{N}-s{seed}", "study": "A", "task": R.PILOT_STUDY_A_TASK, "N": N, "seed": seed,
        "completed": True, "onset_shape": None if N == 0 else "abrupt", "step_matching": None if N == 0 else "total_steps",
        "treatment": None, "controller_variant": None, "measurement_cost": cost, "final_cost": cost if final is None else final,
        "gap_hazard": gap_h, "gap_dynamics": gap_d, "wall_clock_hours": 10.0,
    }


def pilot_rows(ref=(24.0, 25.0, 26.0), late=(25.0, 25.5, 26.5), gh_ref=(1.0, 2.0, 3.0), gh_late=(8.0, 7.0, 9.0)) -> list[dict]:
    return [row(0.0, s, c, g) for s, (c, g) in enumerate(zip(ref, gh_ref))] + [
        row(0.5, s, c, g) for s, (c, g) in enumerate(zip(late, gh_late))
    ]


def test_g1_passes_and_fails_for_the_registered_reasons() -> None:
    ok = go_decision.g1_feasibility(pilot_rows())
    assert ok.passed is True and ok.values["reference_sd"] == pytest.approx(1.0)
    over_budget = go_decision.g1_feasibility(pilot_rows(late=(27.0, 28.0, 29.0)))
    assert over_budget.passed is False and over_budget.values["both_arms_within_budget"] is False
    spread = go_decision.g1_feasibility(pilot_rows(ref=(10.0, 25.0, 40.0), late=(25.0, 25.0, 25.0)))
    assert spread.values["reference_sd_ok"] is False and spread.passed is False
    unmatched = go_decision.g1_feasibility(pilot_rows(ref=(20.0, 20.0, 20.0), late=(23.0, 23.0, 23.0)))
    assert unmatched.values["arms_matched"] is False  # |23 - 20| > 2.5
    boundary = go_decision.g1_feasibility(pilot_rows(ref=(25.0, 25.0, 25.0), late=(27.5, 27.5, 27.5)))
    assert boundary.passed is True  # "at most" d + 2.5 and |difference| <= 2.5 are inclusive


def test_g1_waits_on_q_g1_level_when_a_seed_is_over_budget_but_the_mean_is_not(monkeypatch) -> None:
    rows = pilot_rows(ref=(24.0, 24.0, 28.0), late=(25.0, 25.0, 27.0))  # means 25.33 and 25.67; one seed > 27.5
    split = go_decision.g1_feasibility(rows)
    assert split.values["both_arms_within_budget"] is True and split.values["every_seed_within_budget"] is False
    assert split.values["reference_costs"] == [24.0, 24.0, 28.0] and split.values["late_costs"] == [25.0, 25.0, 27.0]
    assert split.passed is None and any("Q-g1-level" in n and "disagree" in n for n in split.notes)
    assert split.undecided_by == "Q-g1-level"
    # the report says the condition waits on the question; it is not "not computable"
    result = go_decision.report(
        rows, hours_per_run=[10.0] * 8, concurrent_runs=8, machine_hours_allocated=1000.0,
        allocation_ok=True, allocation_note="ok", unconstrained_cost=55.0,
        moderate={"mean_cost": {10.0: 9.0, 20.0: 19.0, 40.0: 41.0}},
    )
    assert result["part6_reading"].startswith("UNDECIDED: G1 waits on Q-g1-level")
    assert result["conditions"]["G1 feasibility"]["undecided_by"] == "Q-g1-level"
    md = go_decision.to_markdown(result)
    assert "## G1 feasibility: UNDECIDED (Q-g1-level)" in md and "NOT COMPUTABLE" not in md
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset({"Q-g1-level"}))
    answered = go_decision.g1_feasibility(rows)
    assert answered.passed is True and answered.undecided_by is None  # the proposed (arm-mean) reading, once recorded
    # when another clause already fails, the reading does not matter and G1 fails
    unmatched = go_decision.g1_feasibility(pilot_rows(ref=(20.0, 20.0, 28.0), late=(26.0, 26.0, 26.0)))
    assert unmatched.passed is False


def test_g1_not_computable_without_measurement_costs() -> None:
    rows = pilot_rows()
    rows[0]["measurement_cost"] = None
    assert go_decision.g1_feasibility(rows).passed is None


def test_g1_and_g3_need_three_completed_seeds_per_arm() -> None:
    rows = pilot_rows()
    rows[2]["completed"] = False  # reference seed 2 excluded, replacement not finished
    assert go_decision.g1_feasibility(rows).passed is None
    assert go_decision.g3_signal(rows).passed is None
    replaced = pilot_rows()
    replaced[2]["seed"] = 5  # the replacement finished: three runs, but only two pairs
    assert go_decision.g1_feasibility(replaced).passed is True
    g3 = go_decision.g3_signal(replaced)
    assert g3.passed is True and any("mean clause" in n for n in g3.notes)  # the mean alone decides
    weak = pilot_rows(gh_ref=(1.0, 1.0, 1.0), gh_late=(1.5, 2.0, 1.0))
    weak[2]["seed"] = 5
    g3 = go_decision.g3_signal(weak)
    assert g3.passed is None and any("Q-g3-pairing" in n and "undefined" in n for n in g3.notes)


def test_g3_with_a_replacement_seed_is_decided_by_the_group_even_after_q_g3_pairing(monkeypatch) -> None:
    weak = pilot_rows(gh_ref=(1.0, 1.0, 1.0), gh_late=(1.5, 2.0, 1.0))
    weak[2]["seed"] = 5  # two pairs only: the paired U80 of equation (12) is undefined
    for answered in (frozenset(), frozenset({"Q-g3-pairing"})):
        monkeypatch.setattr(R, "ANSWERED_QUESTIONS", answered)
        g3 = go_decision.g3_signal(weak)
        assert g3.passed is None and g3.undecided_by == "Q-g3-pairing"
        assert g3.values["U80"] is None
        # the pooled form is reported beside it: delta 0.5, s = sqrt((0 + 0.25) / 2)
        assert g3.values["U80_pooled"] == pytest.approx(0.5 + 1.886 * math.sqrt(0.125) / math.sqrt(3))


def test_u80_matches_equation_12_and_the_t_quantile() -> None:
    diffs = [7.0, 5.0, 6.0]
    expected = statistics.fmean(diffs) + 1.886 * statistics.stdev(diffs) / math.sqrt(3)
    assert go_decision.u80(diffs) == pytest.approx(expected)
    with pytest.raises(ValueError):
        go_decision.u80([1.0, 2.0])


def test_g3_signal_mean_or_upper_bound() -> None:
    strong = go_decision.g3_signal(pilot_rows())
    assert strong.passed is True and strong.values["delta_mean"] == pytest.approx(6.0)
    # mean below 5 but U80 above 5 still passes (Part 6 G3: "or its upper bound U80 ... is at least 5")
    wide = go_decision.g3_signal(pilot_rows(gh_ref=(0.0, 0.0, 0.0), gh_late=(0.0, 3.0, 9.0)))
    assert wide.values["delta_mean"] == pytest.approx(4.0) and wide.values["U80"] > 5 and wide.passed is True
    weak = go_decision.g3_signal(pilot_rows(gh_ref=(1.0, 1.0, 1.0), gh_late=(1.5, 2.0, 1.0)))
    assert weak.passed is False


def test_g3_is_not_read_while_the_pairing_question_flips_it() -> None:
    # paired: differences (2, 2, 2), U80 = 2 (fail); pooled: s = 10, U80 = 2 + 1.886 * 10 / sqrt(3) (pass)
    split = go_decision.g3_signal(pilot_rows(gh_ref=(0.0, 10.0, 20.0), gh_late=(2.0, 12.0, 22.0)))
    assert split.values["U80"] == pytest.approx(2.0)
    assert split.values["U80_pooled"] == pytest.approx(2.0 + 1.886 * 10 / math.sqrt(3))
    assert split.passed is None and any("disagree" in n for n in split.notes)
    assert split.undecided_by == "Q-g3-pairing"


def test_g4_conditioning() -> None:
    ok = go_decision.g4_conditioning(55.0, {10.0: 9.0, 20.0: 19.0, 40.0: 41.0})
    assert ok.passed is True and ok.values["largest_budget"] == 45.0
    unbound = go_decision.g4_conditioning(44.0, {10.0: 9.0, 20.0: 19.0, 40.0: 41.0})
    assert unbound.passed is False  # 45 is not below the unconstrained cost
    disordered = go_decision.g4_conditioning(55.0, {10.0: 21.0, 20.0: 19.0, 40.0: 41.0})
    assert disordered.values["ordered_with_budgets"] is False and disordered.passed is False
    over = go_decision.g4_conditioning(55.0, {10.0: 12.6, 20.0: 19.0, 40.0: 41.0})
    assert over.passed is False
    far_below = go_decision.g4_conditioning(55.0, {10.0: 1.0, 20.0: 5.0, 40.0: 12.0})
    assert far_below.passed is True  # Part 6 is one-sided; Part 3.6's two-sided reading is reported beside it
    assert any("Q-g4-level" in n for n in far_below.notes)
    assert go_decision.g4_conditioning(None, None).passed is None


def test_measured_concurrency() -> None:
    from datetime import datetime, timedelta, timezone

    t0 = datetime(2026, 10, 1, tzinfo=timezone.utc)
    h = timedelta(hours=1)
    intervals = [(t0, t0 + 10 * h), (t0, t0 + 10 * h), (t0 + 10 * h, t0 + 20 * h)]
    m = go_decision.measured_concurrency(intervals)
    assert m["max"] == 2.0 and m["time_weighted_mean"] == pytest.approx(1.5)
    assert go_decision.measured_concurrency([]) == {"max": 0.0, "time_weighted_mean": 0.0}


def test_g2_needs_every_pilot_run() -> None:
    assert go_decision.g2_throughput([10.0] * 7, 8, 1000.0, True, "ok").passed is None


def test_g2_notes_when_only_the_registered_total_fits() -> None:
    # required 10/8 x 484 = 605 <= 700 < 10/8 x 627.13 = 783.9
    cond = go_decision.g2_throughput([10.0] * 8, 8, 700.0, True, "ok")
    assert cond.passed is True and any("NOT with the corrected" in n for n in cond.notes)
    assert not any("corrected" in n for n in go_decision.g2_throughput([10.0] * 8, 8, 1000.0, True, "ok").notes)


def test_g2_uses_the_allocation_only_if_it_came_first() -> None:
    passed = go_decision.g2_throughput([10.0] * 8, 8, 1000.0, True, "ok")
    assert passed.values["required_machine_hours_registered"] == pytest.approx(10 / 8 * 484) and passed.passed is True
    late = go_decision.g2_throughput([10.0] * 8, 8, 1000.0, False, "after")
    assert late.passed is None
    short = go_decision.g2_throughput([10.0] * 8, 8, 500.0, True, "ok")
    assert short.passed is False


def test_part6_reading_branches() -> None:
    C = go_decision.Condition
    go = go_decision.part6_reading(C("G1 x", True), C("G2 x", True), C("G3 x", True), C("G4 x", True))
    assert go.startswith("GO")
    nogo_a = go_decision.part6_reading(C("G1 x", True), C("G2 x", True), C("G3 x", False), C("G4 x", True))
    assert "NO-GO for Study A" in nogo_a and "Study B becomes the main study" in nogo_a
    revise = go_decision.part6_reading(C("G1 x", False), C("G2 x", True), C("G3 x", True), C("G4 x", True))
    assert revise.startswith("REVISE ONCE for G1")
    incomplete = go_decision.part6_reading(C("G1 x", None), C("G2 x", True), C("G3 x", True), C("G4 x", True))
    assert incomplete.startswith("INCOMPLETE") and "UNDECIDED" not in incomplete
    both = go_decision.part6_reading(C("G1 x", None, undecided_by="Q-g1-level"), C("G2 x", None), C("G3 x", True), C("G4 x", True))
    assert both.startswith("INCOMPLETE: G2 cannot be evaluated yet. UNDECIDED: G1 waits on Q-g1-level")
    again_g1 = go_decision.part6_reading(C("G1 x", False), C("G2 x", True), C("G3 x", True), C("G4 x", True), failed_before={"G1"})
    assert again_g1 == "NO-GO for Study A: G1 failed again after the revision."
    again_g4 = go_decision.part6_reading(C("G1 x", True), C("G2 x", True), C("G3 x", True), C("G4 x", False), failed_before={"G4"})
    assert "NO-GO for Study B" in again_g4
    again_g2 = go_decision.part6_reading(C("G1 x", True), C("G2 x", False), C("G3 x", True), C("G4 x", True), failed_before={"G2"})
    assert "cut order of Part 6.1" in again_g2 and "Never cut the number of seeds" in again_g2
    # only a second failure of the SAME condition is a no-go
    first_g1 = go_decision.part6_reading(C("G1 x", False), C("G2 x", True), C("G3 x", True), C("G4 x", True), failed_before={"G4"})
    assert "NO-GO" not in first_g1 and "G1 failed for the first time on the re-pilot" in first_g1
    repaired = go_decision.part6_reading(C("G1 x", True), C("G2 x", True), C("G3 x", True), C("G4 x", True), failed_before={"G1"})
    assert repaired.startswith("GO")
    with pytest.raises(ValueError):
        go_decision.part6_reading(C("G1 x", True), C("G2 x", True), C("G3 x", True), C("G4 x", True), failed_before={"G3"})


def test_part6_reading_reports_g3_alongside_other_failures() -> None:
    C = go_decision.Condition
    # first pilot: a revisable failure with G3 failing -> G3 is read again after the revision
    both = go_decision.part6_reading(C("G1 x", False), C("G2 x", True), C("G3 x", False), C("G4 x", True))
    assert both.startswith("REVISE ONCE for G1") and "G3 also failed; it is read again after the revision of G1." in both
    assert "NO-GO" not in both
    # first pilot: G4 fails, G1 and G2 hold, G3 fails -> revise G4 and no-go for Study A
    g4_g3 = go_decision.part6_reading(C("G1 x", True), C("G2 x", True), C("G3 x", False), C("G4 x", False))
    assert "REVISE ONCE for G4" in g4_g3 and "NO-GO for Study A" in g4_g3
    # re-pilot: only G3 fails -> no-go for Study A
    only_g3 = go_decision.part6_reading(C("G1 x", True), C("G2 x", True), C("G3 x", False), C("G4 x", True), failed_before={"G1"})
    assert only_g3.startswith("NO-GO for Study A") and "Study B becomes the main study" in only_g3
    # re-pilot: G2 fails again and G3 fails -> the cut order, and G3's failure is not dropped
    g2_g3 = go_decision.part6_reading(C("G1 x", True), C("G2 x", False), C("G3 x", False), C("G4 x", True), failed_before={"G2"})
    assert "cut order of Part 6.1" in g2_g3 and "G3 also failed on the re-pilot" in g2_g3
    # re-pilot: G1 fails again -> Study A is a no-go, so a G3 failure raises no question
    g1_g3 = go_decision.part6_reading(C("G1 x", False), C("G2 x", True), C("G3 x", False), C("G4 x", True), failed_before={"G1"})
    assert g1_g3 == "NO-GO for Study A: G1 failed again after the revision."


def test_an_undecided_g3_does_not_withhold_a_branch_that_g1_or_g2_already_fixes() -> None:
    # Part 6 reads G3 only when G1 and G2 hold, so once either fails an undecided G3 is moot for the branch
    C = go_decision.Condition
    g3 = C("G3 x", None, undecided_by="Q-g3-pairing")
    again_g1 = go_decision.part6_reading(C("G1 x", False), C("G2 x", True), g3, C("G4 x", True), failed_before={"G1"})
    assert again_g1 == "NO-GO for Study A: G1 failed again after the revision."
    first_g1 = go_decision.part6_reading(C("G1 x", False), C("G2 x", True), g3, C("G4 x", True))
    assert first_g1.startswith("REVISE ONCE for G1") and "UNDECIDED" not in first_g1
    assert "G3 is undecided (Q-g3-pairing); it is read again after the revision of G1." in first_g1
    first_g2 = go_decision.part6_reading(C("G1 x", True), C("G2 x", False), g3, C("G4 x", True))
    assert first_g2.startswith("REVISE ONCE for G2") and "G3 is undecided (Q-g3-pairing)" in first_g2
    again_g2 = go_decision.part6_reading(C("G1 x", True), C("G2 x", False), g3, C("G4 x", True), failed_before={"G2"})
    assert "cut order of Part 6.1" in again_g2 and "G3 is undecided (Q-g3-pairing) on the re-pilot" in again_g2
    # while G1 and G2 hold, G3 decides the branch and the reading waits for the group
    held = go_decision.part6_reading(C("G1 x", True), C("G2 x", True), g3, C("G4 x", False))
    assert held.startswith("UNDECIDED: G3 waits on Q-g3-pairing")


def test_the_undecided_wording_holds_after_q_g3_pairing_is_answered(monkeypatch) -> None:
    # with a replacement seed G3 stays with the group even after the key is answered; the reading must not
    # claim that the key's answer is still needed or that two readings are given (U80 is undefined)
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset({"Q-g3-pairing"}))
    weak = pilot_rows(gh_ref=(1.0, 1.0, 1.0), gh_late=(1.5, 2.0, 1.0))
    weak[2]["seed"] = 5
    C = go_decision.Condition
    reading = go_decision.part6_reading(C("G1 x", True), C("G2 x", True), go_decision.g3_signal(weak), C("G4 x", True))
    assert reading == (
        "UNDECIDED: G3 waits on Q-g3-pairing (the group decides first; the condition's notes give the readings)."
    )


def test_report_and_table_8_1(tmp_path) -> None:
    result = go_decision.report(
        pilot_rows(), hours_per_run=[10.0] * 8, concurrent_runs=8, machine_hours_allocated=1000.0,
        allocation_ok=True, allocation_note="ok", unconstrained_cost=55.0,
        moderate={"mean_cost": {10.0: 9.0, 20.0: 19.0, 40.0: 41.0}, "satisfaction": {10.0: 0.9}},
    )
    assert result["part6_reading"].startswith("GO")
    table = result["table_8_1"]
    assert table["Reference arm's cost under the dynamics perturbation (mean over seeds)"] == pytest.approx(26.0)
    # the perturbed cost is rebuilt from measurement_cost, never from the selection-set cost
    rows = pilot_rows()
    for r in rows:
        r["selection_cost_at_match"] = r["measurement_cost"] - 3.0
    other = go_decision.report(
        rows, hours_per_run=[10.0] * 8, concurrent_runs=8, machine_hours_allocated=1000.0, allocation_ok=True,
        allocation_note="ok", unconstrained_cost=55.0, moderate=None,
    )
    assert other["table_8_1"]["Reference arm's cost under the dynamics perturbation (mean over seeds)"] == pytest.approx(26.0)
    # G1 is decided on the measurement set only; the selection-set means are reported and never decide
    g1 = other["conditions"]["G1 feasibility"]["values"]
    assert g1["cost_field"] == "measurement_cost" and g1["selection_set_means_not_decisive"] == [pytest.approx(22.0), pytest.approx(22.6666, rel=1e-3)]
    assert result["repilot"] is False and result["failed_before"] == []
    assert table["Seed-to-seed SD of final in-distribution cost, N = 0.50 arm"] == pytest.approx(statistics.stdev([25.0, 25.5, 26.5]))
    again = go_decision.report(
        pilot_rows(), hours_per_run=[10.0] * 8, concurrent_runs=8, machine_hours_allocated=500.0,
        allocation_ok=True, allocation_note="ok", unconstrained_cost=55.0,
        moderate={"mean_cost": {10.0: 9.0, 20.0: 19.0, 40.0: 41.0}}, failed_before=("G2", "G2"),
    )
    assert again["repilot"] is True and again["failed_before"] == ["G2"]
    assert again["part6_reading"].startswith("G2 failed again after the revision")
    js, md = go_decision.write_report(result, tmp_path, "20261001T000000Z")
    assert js.exists() and "G1 feasibility: PASS" in md.read_text()
    with pytest.raises(FileExistsError):
        go_decision.write_report(result, tmp_path, "20261001T000000Z")  # reports are never overwritten


def test_cli_reports_expected_errors_without_a_traceback(capsys) -> None:
    from pilot.__main__ import main

    before = budget.ALLOCATION_FILE.read_bytes() if budget.ALLOCATION_FILE.exists() else None  # committed after task 12
    assert main(["allocate", "--hours", "-5", "--by", "the group", "--date", "2026-10-01"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ValueError") and "Traceback" not in err
    assert main(["allocate", "--hours", "5", "--by", "the group", "--date", "1 October"]) == 1
    after = budget.ALLOCATION_FILE.read_bytes() if budget.ALLOCATION_FILE.exists() else None
    assert after == before  # a refused allocation writes nothing

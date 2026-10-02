"""Study A verdicts on synthetic datasets built to be SUPPORTED, FALSIFIED and INCONCLUSIVE (analysis/study_a.py)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import test_analysis_synthetic as syn  # noqa: E402

from analysis import data, matching, questions, stats, study_a  # noqa: E402
from analysis import verdict as V  # noqa: E402
from configs import registered as R  # noqa: E402
from pilot.manifest import CUTS  # noqa: E402
from results import supplement_schema as S  # noqa: E402

TASK = R.PRIMARY_TASK
SPECS = {s.run_id: s for s in syn.specs(groups=("main", "treatment", "controller", "pid"))}


@pytest.fixture
def keys_open(monkeypatch):
    """Every PENDING key open (configs.registered.ANSWERED_QUESTIONS empty): the test reads the other readings that
    analysis.verdict computes only while a key is open, whatever the amendment log has answered since."""
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset())


@pytest.fixture(autouse=True)
def _fewer_resamples(monkeypatch):
    """The verdict logic does not depend on the resample count, so these tests draw 2,000; the statistics
    tests and the end-to-end run use the registered 10,000 (R.BOOTSTRAP_RESAMPLES)."""
    monkeypatch.setattr(R, "BOOTSTRAP_RESAMPLES", 2_000)


def supplement(records: list[tuple[str, dict]]) -> data.Supplement:
    sup = data.Supplement()
    for kind, record in records:
        model = S.validate_record(kind, record)
        sup.records.setdefault(kind, {})[(model.run_id, S.record_part(kind, model))] = model
    return sup


def run(rows: list[dict], sup: data.Supplement | None = None, *, iqm: bool = False, cuts: tuple[str, ...] = (),
        surplus_extra: int = 0) -> tuple[dict[str, V.Verdict], study_a.Catalogue]:
    """The Study A catalogue; the descriptive IQM table only when asked for (it is the slow part)."""
    ds = data.build_dataset(rows, sup or data.Supplement(), mode="final", cuts=cuts, surplus_extra=surplus_extra)
    cat = study_a.analyse(ds, iqm=iqm)
    return {v.id: v for v in cat.verdicts}, cat


def where(rows: list[dict], **factors: Any) -> list[dict]:
    return [r for r in rows if all(r.get(k) == v for k, v in factors.items())]


def late(rows: list[dict], N: float = 0.5, shape: str = "abrupt", control: str = "total_steps",
         **kw: Any) -> list[dict]:
    return where(rows, task=TASK, N=N, onset_shape=shape, step_matching=control,
                 treatment=kw.get("treatment"), controller_variant=kw.get("controller_variant"))


def training_records(rows: list[dict], *, injection_passes: bool = True) -> list[tuple[str, dict]]:
    """Manipulation-check rows at onset + 200,000 steps for the N = 0.50 abrupt total-steps arms."""
    out = []
    for row in rows:
        spec = SPECS[row["run_id"]]
        if not (spec.task == TASK and spec.N == 0.5 and spec.onset_shape == "abrupt" and spec.step_matching == "total_steps"):
            continue
        e = syn.noise(spec.seed)
        treated = spec.treatment in ("reset", "injection") and (injection_passes or spec.treatment == "reset")
        dormant, rank = (0.3 + 0.01 * e, 30.0 + e) if treated else (0.5 + 0.01 * e, 20.0 + e)
        out.append(("training", syn.training_record(row, spec=spec, dormant_check=dormant, rank_check=rank)))
    return out


# ---------------------------------------------------------------------------
# H1, H2, H0 and the NOT_TESTED gate
# ---------------------------------------------------------------------------


def test_h1_supported_falsified_and_inconclusive() -> None:
    supported, _ = run(syn.a_world("supported"))
    h1 = supported["H1"]
    assert h1.status == V.SUPPORTED and h1.label == V.CONFIRMATORY
    for control in R.STEP_MATCHING_CONTROLS:
        assert h1.numbers[f"{control}.status"] == V.SUPPORTED
        assert h1.numbers[f"{control}.Delta(0.50)"] == pytest.approx(6.0)
        assert h1.numbers[f"{control}.Delta(0.50)_welch_lo"] > 0 and h1.numbers[f"{control}.trend_rho"] > 0
        assert h1.numbers[f"{control}.trend_points"] == R.H1_TREND_POINTS  # four fractions by five seeds
    assert {e["contrast"] for e in h1.estimates} == {"Delta(0.10)", "Delta(0.25)", "Delta(0.50)"}
    falsified, _ = run(syn.a_world("falsified"))
    h1 = falsified["H1"]
    assert h1.status == V.FALSIFIED
    for control in R.STEP_MATCHING_CONTROLS:  # Part 5.7: the upper bound against the minimum effect
        assert h1.numbers[f"{control}.F1_all_intervals_include_zero"] is True
        assert h1.numbers[f"{control}.bound_95_upper_Delta050"] < R.MIN_EFFECT_STUDY_A
        assert h1.numbers[f"{control}.bound_reading"].startswith("evidence that any effect is smaller")
    assert any("Part 5.7" in n for n in h1.notes)
    inconclusive, _ = run(syn.a_world("inconclusive"))
    h1 = inconclusive["H1"]
    assert h1.status == V.INCONCLUSIVE
    for control in R.STEP_MATCHING_CONTROLS:
        assert h1.numbers[f"{control}.Delta(0.50)_welch_lo"] < 0 < h1.numbers[f"{control}.Delta(0.50)_welch_hi"]
        assert h1.numbers[f"{control}.F2_not_ordered"] is False
        assert h1.numbers[f"{control}.F1_all_intervals_include_zero"] is False


@pytest.mark.usefixtures("keys_open")
def test_h1_is_read_under_both_controls() -> None:
    rows = syn.a_world("supported")
    for row in (late(rows, control="constrained_steps") + late(rows, N=0.25, control="constrained_steps")
                + late(rows, N=0.1, control="constrained_steps")):
        row["gap_hazard"] = 2.0 + 0.3 * syn.noise(row["seed"])  # no effect of N under this control
    verdicts, _ = run(rows)
    h1 = verdicts["H1"]
    assert h1.numbers["total_steps.status"] == V.SUPPORTED and h1.numbers["constrained_steps.status"] == V.FALSIFIED
    # supported under one control and falsified under the other: inconclusive under the proposal,
    # falsified if either control's falsification suffices (Q-controls-reading) -> the question decides
    assert h1.proposal_status == V.INCONCLUSIVE and h1.readings["Q-controls-reading"] == V.FALSIFIED
    assert h1.status == V.UNDECIDED and "Q-controls-reading" in h1.undecided_by


@pytest.mark.usefixtures("keys_open")
def test_an_open_key_whose_readings_disagree_leaves_the_verdict_undecided(monkeypatch) -> None:
    rows = syn.a_world("supported")
    for row in where(rows, task=TASK, onset_shape="ramp"):
        row["gap_hazard"] = 2.0 + 0.3 * syn.noise(row["seed"])  # the ramp arms show no effect of N
    verdicts, _ = run(rows)
    assert verdicts["H1"].status == V.UNDECIDED and verdicts["H1"].undecided_by == ("Q-h1-shape",)
    assert verdicts["H1"].proposal_status == V.SUPPORTED and verdicts["H1"].readings["Q-h1-shape"] == V.FALSIFIED
    assert "Q-h1-shape" in verdicts["H1"].provisional_on
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset({"Q-h1-shape"}))  # answered with the proposal
    verdicts, _ = run(rows)
    assert verdicts["H1"].status == V.SUPPORTED and "Q-h1-shape" not in verdicts["H1"].provisional_on


@pytest.mark.usefixtures("keys_open")
def test_matching_on_the_selection_set_is_a_reading_of_q_matched_cost_set() -> None:
    rows = syn.a_world("supported")
    for row in late(rows):
        row["selection_cost_at_match"] = 29.0  # matched on the measurement set, not on the selection set
    verdicts, _ = run(rows, supplement([("measurement", syn.measurement_record(r)) for r in rows]))
    assert verdicts["H1"].proposal_status == V.SUPPORTED
    assert verdicts["H1"].readings["Q-matched-cost-set"] == V.NOT_COMPUTABLE
    assert verdicts["H1"].status == V.UNDECIDED and "Q-matched-cost-set" in verdicts["H1"].undecided_by
    claim = verdicts["A-return[SafetyPointGoal1-v0/total_steps/N0.50]"]  # secondary claims follow the reading too
    assert claim.readings["Q-matched-cost-set"] == V.NOT_COMPUTABLE and claim.status == V.UNDECIDED


def test_with_every_question_answered_nothing_is_provisional_or_undecided(monkeypatch) -> None:
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset(R.PENDING))
    rows = syn.a_world("supported")
    for row in where(rows, task=TASK, onset_shape="ramp"):
        row["gap_hazard"] = 2.0
    verdicts, _ = run(rows)
    assert all(not v.provisional_on and v.status != V.UNDECIDED and not v.readings for v in verdicts.values())
    assert verdicts["H1"].status == V.SUPPORTED


def test_h2() -> None:
    verdicts, _ = run(syn.a_world("supported"))
    h2 = verdicts["H2"]
    assert h2.status == V.SUPPORTED and h2.label == V.SECONDARY
    assert h2.numbers["total_steps.E(0.50)"] == pytest.approx(3.0) and h2.numbers["total_steps.holm_survives"] is True
    verdicts, _ = run(syn.a_world("falsified"))
    assert verdicts["H2"].status == V.FALSIFIED  # the ramp arm's gap is as large as the abrupt arm's
    rows = syn.a_world("supported")
    for row in where(rows, task=TASK, onset_shape="ramp"):
        e = syn.noise(row["seed"])
        abrupt = 2.0 + 12.0 * row["N"]
        row["gap_hazard"] = abrupt - (1.0 + 6.0 * e if row["N"] == 0.5 else 2.0 + 0.05 * e)
    verdicts, _ = run(rows)
    h2 = verdicts["H2"]  # E(0.50) = 1 with an interval including zero; E(0.10), E(0.25) = 2 excluding zero
    assert h2.status == V.INCONCLUSIVE
    assert h2.numbers["total_steps.E(0.50)_welch_lo"] < 0 < h2.numbers["total_steps.E(0.50)"]


def test_h0_and_the_boxes_it_stops() -> None:
    verdicts, _ = run(syn.a_world("supported"))
    assert verdicts["H0"].status == V.FALSIFIED and verdicts["H0"].numbers == {"H1": V.SUPPORTED, "H2": V.SUPPORTED}
    verdicts, _ = run(syn.a_world("falsified"))
    assert verdicts["H0"].status == V.SUPPORTED
    for key in ("H3", "H3a", "H3b", "H3c", "H4"):
        v = verdicts[key]
        assert v.status == V.NOT_TESTED and "not tested, not as falsified" in v.reason
    assert "D(reset)" in verdicts["H3b"].numbers  # the estimates are still computed
    for key, v in verdicts.items():  # the secondary cells of H3 and H4 are not tested either
        if key.startswith(("H3a[", "H3b[", "H4[")):
            assert v.status == V.NOT_TESTED and "not tested, not as falsified" in v.reason, (key, v.status, v.reason)
    out = V.Outcome
    assert study_a.h0_from(out(V.FALSIFIED), out(V.INCONCLUSIVE), V.PROPOSED) == V.INCONCLUSIVE
    assert study_a.h0_from(out(V.NOT_COMPUTABLE), out(V.FALSIFIED), V.PROPOSED) == V.NOT_COMPUTABLE
    assert study_a.h0_from(out(V.SUPPORTED), out(V.NOT_COMPUTABLE), V.PROPOSED) == V.FALSIFIED


def test_h0_is_inconclusive_when_the_missing_part_cannot_change_it() -> None:
    """Box H0 is three-valued, as combine_controls (Q-h0-scope). H1 NOT_COMPUTABLE with its support already
    False under every control can never be supported, so H0 can never be falsified; with H2 INCONCLUSIVE it can
    never be supported either: INCONCLUSIVE, not NOT_COMPUTABLE. Likewise for an H1 that can be neither supported
    nor falsified beside an H2 FALSIFIED."""
    no_support = {f"{c}.{k}": v for c in R.STEP_MATCHING_CONTROLS
                  for k, v in (("status", V.NOT_COMPUTABLE), ("support", False), ("falsification", None))}
    h1 = V.Outcome(V.NOT_COMPUTABLE, numbers=no_support)
    assert not study_a.could_be_supported(h1) and study_a.could_be_falsified(h1, V.PROPOSED)
    assert study_a.h0_from(h1, V.Outcome(V.INCONCLUSIVE), V.PROPOSED) == V.INCONCLUSIVE
    assert study_a.h0_from(h1, V.Outcome(V.FALSIFIED), V.PROPOSED) == V.NOT_COMPUTABLE  # H0 may still be supported
    shut = V.Outcome(V.NOT_COMPUTABLE, numbers={**no_support, **{f"{c}.falsification": False
                                                                   for c in R.STEP_MATCHING_CONTROLS}})
    assert study_a.h0_from(shut, V.Outcome(V.FALSIFIED), V.PROPOSED) == V.INCONCLUSIVE
    open_support = V.Outcome(V.NOT_COMPUTABLE, numbers={**no_support, "total_steps.support": None,
                                                        "constrained_steps.status": V.SUPPORTED})
    assert study_a.h0_from(open_support, V.Outcome(V.INCONCLUSIVE), V.PROPOSED) == V.NOT_COMPUTABLE


def test_h3_part_reads_its_stored_support_and_falsification() -> None:
    """The composite H3 is NOT_COMPUTABLE only while a missing part could still change it; a part
    stores its three-valued support and falsification, and one with both already False cannot."""
    shut = V.Outcome(V.NOT_COMPUTABLE, numbers={"support": None, "falsification": False})
    assert not study_a._part_could(shut, V.FALSIFIED) and study_a._part_could(shut, V.SUPPORTED)
    assert not study_a._part_could(V.Outcome(V.INCONCLUSIVE), V.SUPPORTED)
    assert study_a._part_could(V.Outcome(V.NOT_COMPUTABLE, "stopped early"), V.FALSIFIED)
    A = study_a.StudyA(data.build_dataset(syn.a_world("supported"), data.Supplement(), mode="final"))
    for fn in (study_a.h3a_outcome, study_a.h3b_outcome):
        out = fn(A, V.PROPOSED, holm=False)
        assert out.numbers["support"] is True and out.numbers["falsification"] is False, fn.__name__
    waiting = study_a.h3c_outcome(A, V.PROPOSED, holm=False)  # no training records: it stops before its conditions
    assert waiting.status == V.NOT_COMPUTABLE and study_a._part_could(waiting, V.FALSIFIED)


def test_runs_of_an_arm_a_recorded_cut_removed_never_enter_a_comparison() -> None:
    """N = 0.10 runs that completed before the recorded cut 'drop_n010' (2 of 5 seeds) leave box
    H1 as the cut says: Delta(0.10) is not read, with the cut's note, and the arm is absent."""
    rows = [r for r in syn.a_world("supported", groups=("main",)) if not (r["N"] == 0.10 and r["seed"] >= 2)]
    A = study_a.StudyA(data.build_dataset(rows, data.Supplement(), mode="final", cuts=CUTS))
    assert A.arms(R.PRIMARY_TASK, N=0.10, shape="abrupt", control="total_steps") is None
    h1 = study_a.h1_outcome(A, V.PROPOSED)
    assert h1.numbers.get("total_steps.Delta(0.10)") is None
    assert any("drop_n010" in n for n in h1.notes)


def test_h3_and_h4_are_decided_once_h0_can_no_longer_be_supported() -> None:
    """The N = 0.10 abrupt total-steps arm ends above the tolerance (unmatched above; rule 6: not compared, a
    permanent state of the data), so H1 and H2 under total_steps stay NOT_COMPUTABLE; but under constrained_steps
    both are SUPPORTED, so under Q-controls-reading 'both' neither box can be falsified and H0 can never be
    supported: H3 and H4 are decided, never held by H0 for good (Q-h0-scope)."""
    def cost(spec):
        base = 25.0 + 0.5 * syn.noise(spec.seed)
        if (spec.N == 0.10 and spec.onset_shape == "abrupt" and spec.step_matching == "total_steps"
                and spec.treatment is None and spec.controller_variant is None):
            return base + 6.0
        return base
    verdicts, _ = run(syn.a_world("supported", cost=cost))
    h1, h2 = verdicts["H1"], verdicts["H2"]
    assert h1.proposal_status == h2.proposal_status == V.NOT_COMPUTABLE
    assert h1.numbers["constrained_steps.status"] == h2.numbers["constrained_steps.status"] == V.SUPPORTED
    for box in (h1, h2):
        assert not study_a.could_be_falsified(V.Outcome(box.proposal_status, numbers=box.numbers), V.PROPOSED)
    assert verdicts["H4"].proposal_status == V.SUPPORTED, verdicts["H4"].reason
    for key in ("H3a", "H3b"):
        assert verdicts[key].proposal_status == V.SUPPORTED, (key, verdicts[key].reason)
    for key in ("H3", "H3a", "H3b", "H3c", "H4"):
        assert "H0 cannot be decided" not in verdicts[key].reason
    # a NOT_COMPUTABLE control whose falsification is still open keeps the gate (the other test below)
    open_box = V.Outcome(V.NOT_COMPUTABLE, numbers={"constrained_steps.status": V.FALSIFIED,
                                                    "total_steps.status": V.NOT_COMPUTABLE,
                                                    "total_steps.falsification": None})
    assert study_a.could_be_falsified(open_box, V.PROPOSED)
    shut = V.Outcome(V.NOT_COMPUTABLE, numbers={"constrained_steps.status": V.FALSIFIED,
                                                "total_steps.status": V.NOT_COMPUTABLE,
                                                "total_steps.falsification": False})
    assert not study_a.could_be_falsified(shut, V.PROPOSED)


def test_h3_and_h4_wait_while_h0_may_still_be_supported() -> None:
    """Box H0 read three-valued: with H1 falsified and H2 not computable (the ramp N = 0.50 arms lack
    their gap), H0 is not computable but may still be supported, so H3 and H4 are not decided either."""
    rows = syn.a_world("falsified")
    for row in rows:
        if row["N"] == 0.5 and row["onset_shape"] == "ramp" and row["treatment"] is None and row["controller_variant"] is None:
            row["gap_hazard"] = None
    verdicts, _ = run(rows)
    assert verdicts["H1"].proposal_status == V.FALSIFIED and verdicts["H2"].status == V.NOT_COMPUTABLE
    assert verdicts["H0"].status == V.NOT_COMPUTABLE
    for key in ("H3", "H3a", "H3b", "H3c", "H4"):
        v = verdicts[key]
        assert v.status == V.NOT_COMPUTABLE and "H0 cannot be decided" in v.reason, (key, v.status, v.reason)
    assert "D(reset)" in verdicts["H3b"].numbers  # the estimates are still computed
    for key, v in verdicts.items():  # nor are the secondary cells of H3 and H4 decided
        if key.startswith(("H3a[", "H3b[", "H4[")):
            assert v.status not in (V.SUPPORTED, V.FALSIFIED, V.INCONCLUSIVE), (key, v.status)


@pytest.mark.parametrize("N, controls", [(0.10, R.STEP_MATCHING_CONTROLS), (0.50, ("constrained_steps",))])
def test_h0_is_decided_when_only_rule_6_keeps_h1_open(N: float, controls: tuple[str, ...]) -> None:
    """Q-h0-scope: H0 is NOT_COMPUTABLE "only while missing data could still change it". An arm rule 6 leaves out is a
    permanent state of complete data. With the N = 0.10 abrupt arms unmatched under both controls, H1's F1 and point
    ordering stay open for good, so H1 can never be falsified (its support is already False); with the N = 0.50
    abrupt constrained-steps arm unmatched, H1 and H2 under that control have no Delta(0.50) or E(0.50), hence no
    Part 5.7 bound and no clause to read. Either way H0 is INCONCLUSIVE, and H3 and H4 are decided."""
    rows = syn.a_world("falsified")
    for control in controls:
        for row in late(rows, N=N, control=control):
            row["measurement_cost"] = row["selection_cost_at_match"] = 35.0  # rule 6 on both evaluation sets
            row["gap_hazard"] = row["gap_dynamics"] = None  # no battery for an unmatched arm
    verdicts, _ = run(rows)
    h1, h2 = verdicts["H1"], verdicts["H2"]
    assert h1.status == V.NOT_COMPUTABLE and "rule 6" in h1.reason
    for control in controls:
        assert h1.numbers[f"{control}.not_compared_rule6"] == [N] and h1.numbers[f"{control}.incomplete"] == []
    assert not study_a.could_be_falsified(V.Outcome(h1.status, numbers=h1.numbers), V.PROPOSED)
    if h2.status == V.NOT_COMPUTABLE:  # E(0.50) left out under rule 6: nothing is still to come
        assert h2.numbers["constrained_steps.incomplete"] == []
        assert not study_a.could_be_falsified(V.Outcome(h2.status, numbers=h2.numbers), V.PROPOSED)
        assert not study_a.could_be_supported(V.Outcome(h2.status, numbers=h2.numbers))
    assert verdicts["H0"].status == V.INCONCLUSIVE, verdicts["H0"].reason
    for key in ("H3", "H3a", "H3b", "H3c", "H4"):
        assert "H0 cannot be decided" not in verdicts[key].reason, (key, verdicts[key].reason)
    assert verdicts["H3b"].status == verdicts["H4"].status == V.FALSIFIED


# ---------------------------------------------------------------------------
# Part 4.1 in the verdicts: rule 6, rule 3, missing data
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("keys_open")
def test_an_unmatched_arm_is_not_compared_and_is_reported_as_a_learning_finding() -> None:
    rows = syn.a_world("supported")
    for row in late(rows):
        row["measurement_cost"] = 29.0  # a late arm that never reaches the budget
    verdicts, cat = run(rows)
    h1 = verdicts["H1"]
    assert h1.proposal_status == V.NOT_COMPUTABLE and "rule 6" in h1.reason
    # matched on the selection set (Q-matched-cost-set's other reading), H1 is read: the key decides
    assert h1.readings["Q-matched-cost-set"] == V.SUPPORTED and h1.status == V.UNDECIDED
    assert "Q-matched-cost-set" in h1.undecided_by and h1.display_status.startswith("UNDECIDED(")
    unmatched = {r["arm_id"]: r for r in cat.tables["A_unmatched"]}
    row = unmatched["A-PointGoal1-N0.50-abrupt-total"]
    assert row["status"] == matching.STATUS_UNMATCHED_ABOVE and row["learning_finding"] is True
    assert row["reaches_budget"] is False  # 29.0 > d + 2.5 = 27.5: rule 3's test fails
    assert row["min_window_selection_cost"] == 25.0


def test_an_unmatched_arm_within_the_budget_is_not_a_learning_finding() -> None:
    """Q-unmatched-reporting: rule 6's "a late arm that never reaches the budget" is an unmatched arm whose matched cost
    fails rule 3's test (cost <= d + tolerance = 27.5). A late arm at 25.5 against a reference near 22 is unmatched
    (above the reference by more than 2.5) but reaches the budget: reported with its cost and difference, without the
    learning label."""
    def cost(spec):
        late_arm = (spec.N == 0.5 and spec.onset_shape == "abrupt" and spec.step_matching == "total_steps"
                    and spec.treatment is None and spec.controller_variant is None)
        return 25.5 if late_arm else 22.0 + 0.5 * syn.noise(spec.seed)
    _, cat = run(syn.a_world("supported", groups=("main",), cost=cost))
    row = {r["arm_id"]: r for r in cat.tables["A_unmatched"]}["A-PointGoal1-N0.50-abrupt-total"]
    assert row["status"] == matching.STATUS_UNMATCHED_ABOVE and row["difference"] == pytest.approx(3.5)
    assert row["reaches_budget"] is True and row["learning_finding"] is False


def test_an_unmatched_arm_carries_the_learning_label_of_the_rule_7a_repeat_too() -> None:
    """Q-unmatched-reporting: a matched cost "above d + tolerance (27.5; 30.0 in the rule 7 (a) repeat)". A late arm at
    28.5 against a reference near 22 is unmatched at both tolerances: it never reaches the budget at 2.5 (a finding
    about learning), but reaches it in the rule 7 (a) repeat (28.5 <= 30.0), where it is not. An arm unmatched only
    at 5.0 (its task infeasible at 2.5 under rule 3) has a row with the 2.5 columns empty."""
    def cost(late_cost: float, reference: float) -> Callable[[Any], float]:
        def of(spec: Any) -> float:
            late_arm = (spec.N == 0.5 and spec.onset_shape == "abrupt" and spec.step_matching == "total_steps"
                        and spec.treatment is None and spec.controller_variant is None)
            return late_cost if late_arm else reference + 0.5 * syn.noise(spec.seed)
        return of

    arm = "A-PointGoal1-N0.50-abrupt-total"
    _, cat = run(syn.a_world("supported", groups=("main",), cost=cost(28.5, 22.0)))
    row = {r["arm_id"]: r for r in cat.tables["A_unmatched"]}[arm]
    assert row["status"] == row["status_tolerance_5"] == matching.STATUS_UNMATCHED_ABOVE
    assert row["reaches_budget"] is False and row["learning_finding"] is True
    assert row["reaches_budget_tolerance_5"] is True and row["learning_finding_tolerance_5"] is False
    _, cat = run(syn.a_world("supported", groups=("main",), cost=cost(34.0, 28.0)))
    row = {r["arm_id"]: r for r in cat.tables["A_unmatched"]}[arm]
    assert row["status"] == matching.STATUS_INFEASIBLE_BUDGET  # rule 3 at 2.5: 28 > 27.5
    assert row["status_tolerance_5"] == matching.STATUS_UNMATCHED_ABOVE
    assert row["reaches_budget"] is None and row["learning_finding"] is None
    assert row["reaches_budget_tolerance_5"] is False and row["learning_finding_tolerance_5"] is True


def test_an_infeasible_task_enters_no_comparison() -> None:
    rows = syn.a_world("supported", cost=lambda s: 30.0 + 0.5 * syn.noise(s.seed))
    verdicts, cat = run(rows)
    assert verdicts["H1"].proposal_status == V.NOT_COMPUTABLE and "rule 3" in verdicts["H1"].reason
    assert all(r["task_feasible"] is False for r in cat.tables["A_matching"])
    for row in rows:
        row["selection_cost_at_match"] = row["measurement_cost"]  # both evaluation sets give the infeasible cost
    verdicts, _ = run(rows)
    assert verdicts["H1"].status == V.NOT_COMPUTABLE and "rule 3" in verdicts["H1"].reason


@pytest.mark.usefixtures("keys_open")
def test_a_missing_gap_makes_the_verdict_incomplete_not_smaller() -> None:
    rows = syn.a_world("supported")
    late(rows)[2]["gap_hazard"] = None
    verdicts, _ = run(rows)
    h1 = verdicts["H1"]
    assert h1.proposal_status == V.NOT_COMPUTABLE and "incomplete" in h1.reason
    assert late(rows)[2]["run_id"] in h1.reason
    # the ramp arms (Q-h1-shape's other reading) have every gap: the verdict turns on that key
    assert h1.status == V.UNDECIDED and h1.undecided_by == ("Q-h1-shape",)


def test_stored_flags_are_compared_with_the_recomputed_ones() -> None:
    rows = syn.a_world("supported", groups=("main",))
    for row in late(rows):
        row["matched"], row["infeasible"] = False, False  # a stored flag that disagrees
    _, cat = run(rows)
    table = {r["arm_id"]: r for r in cat.tables["A_matching"]}
    assert table["A-PointGoal1-N0.50-abrupt-total"]["stored_flags_agree"] is False
    assert table["A-PointGoal1-N0.00"]["stored_flags_agree"] is None  # nothing stored


# ---------------------------------------------------------------------------
# H3 and H4
# ---------------------------------------------------------------------------


def test_h3_supported_and_the_mediation_wording() -> None:
    rows = syn.a_world("supported")
    verdicts, _ = run(rows, supplement(training_records(rows)))
    for key in ("H3a", "H3b", "H3c"):
        assert verdicts[key].status == V.SUPPORTED, (key, verdicts[key].reason, verdicts[key].numbers)
    assert verdicts["H3"].status == V.SUPPORTED
    assert any("consistent with mediation by plasticity" in n for n in verdicts["H3"].notes)
    assert verdicts["H3a"].numbers["rho_dormant"] > 0 and verdicts["H3a"].numbers["rho_rank"] < 0
    assert verdicts["H3c"].numbers["passes(reset)"] and verdicts["H3c"].numbers["passes(injection)"]


def test_an_intervention_that_narrows_the_gap_without_the_manipulation_check() -> None:
    rows = syn.a_world("supported")
    verdicts, _ = run(rows, supplement(training_records(rows, injection_passes=False)))
    assert verdicts["H3c"].status == V.INCONCLUSIVE  # one passes, one fails
    assert verdicts["H3"].status == V.INCONCLUSIVE
    assert not any("mediation" in n for n in verdicts["H3"].notes)
    assert any("injection narrows the gap without passing the manipulation check" in n for n in verdicts["H3"].notes)


def test_h3b_is_falsified_when_more_constrained_training_reduces_the_gap_as_much_as_reset() -> None:
    rows = syn.a_world("supported")
    for row in late(rows, treatment="additional_constrained"):
        row["gap_hazard"] = 2.0 + 12.0 * 0.5 - 6.0 + 0.3 * syn.noise(row["seed"])  # as reset
    verdicts, _ = run(rows)
    assert verdicts["H3b"].status == V.FALSIFIED and verdicts["H3b"].numbers["reduces(additional_constrained)"]
    assert "gap(additional_constrained) - gap(reset)" in verdicts["H3b"].numbers


def test_h3a_contrary_directions_neither_support_nor_falsify() -> None:
    rows = syn.a_world("supported")
    for row in where(rows, task=TASK):
        row["rank_onset"] = 10.0 + (row["gap_hazard"] or 0.0)  # rank rises with the gap: contrary to H3 (a)
    verdicts, _ = run(rows)
    h3a = verdicts["H3a"]
    assert h3a.status == V.INCONCLUSIVE and any("contrary" in n for n in h3a.notes)
    assert h3a.numbers["rho_rank"] > 0 and h3a.numbers["rho_rank_lo"] > 0


def test_h4_reports_the_rate_clips_share_of_the_rate_limited_runs() -> None:
    """Q-rate-limit (answered): "the report gives the share of constrained epochs in which the clip bound".
    analysis.data copies each run's share from its training record (rate_limit_clip_share), and H4 reports the
    rate-limited arm's shares by seed, their mean and their count, descriptively: they change no status."""
    rows = syn.a_world("supported")
    records, shares = [], {}
    for row in late(rows, controller_variant="rate_limited"):
        spec = SPECS[row["run_id"]]
        record = syn.training_record(row, spec=spec)
        shares[spec.seed] = record["rate_limit_clip_share"] = round(0.1 * (spec.seed + 1), 2)
        records.append(("training", record))
    ds = data.build_dataset(rows, supplement(records), mode="final")
    copied = {r["run_id"]: r["rate_limit_clip_share"] for r in ds.study_a}
    assert {copied[row["run_id"]] for row in late(rows, controller_variant="rate_limited")} == set(shares.values())
    assert {v for run_id, v in copied.items() if SPECS[run_id].controller_variant != "rate_limited"} == {None}
    base, _ = run(rows)
    verdicts, _ = run(rows, supplement(records))
    name = "rate_limit_clip_share(rate_limited)"
    h4 = verdicts["H4"]
    assert h4.numbers[f"{name}_by_seed"] == shares and h4.numbers[f"{name}_n"] == len(R.SEEDS)
    assert h4.numbers[name] == pytest.approx(sum(shares.values()) / len(shares))
    assert h4.status == base["H4"].status == V.SUPPORTED  # descriptive: it decides nothing
    assert base["H4"].numbers[name] is None and base["H4"].numbers[f"{name}_n"] == 0  # no training records


@pytest.mark.usefixtures("keys_open")
def test_h4_supported_falsified_and_the_reading_of_does_not_fall() -> None:
    verdicts, _ = run(syn.a_world("supported"))
    h4 = verdicts["H4"]
    assert h4.status == V.SUPPORTED and h4.numbers["falls"] and h4.numbers["remains_above"]
    assert "F(rate_limited)" in h4.numbers and "F(pid)" in h4.numbers  # reported alongside
    rows = syn.a_world("supported")
    for row in late(rows, controller_variant="warm_started"):
        row["gap_hazard"] = 2.0 + 0.3 * syn.noise(row["seed"])  # down to the N = 0 gap: overshoot explains it all
    verdicts, _ = run(rows)
    assert verdicts["H4"].status == V.FALSIFIED
    rows = syn.a_world("supported")
    for row in late(rows, controller_variant="warm_started"):
        row["gap_hazard"] = 2.0 + 12.0 * 0.5 - 1.0 + 3.0 * syn.noise(row["seed"])  # falls by 1, interval includes 0
    verdicts, _ = run(rows)
    h4 = verdicts["H4"]
    assert h4.proposal_status == V.INCONCLUSIVE and h4.readings["Q-h4-reading"] == V.FALSIFIED
    assert h4.status == V.UNDECIDED and h4.undecided_by == ("Q-h4-reading",)


# ---------------------------------------------------------------------------
# Secondary outcomes, Part 4.1.1 and rule 7
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("keys_open")
def test_the_boxes_are_applied_to_the_other_conditions_and_tasks_with_holm() -> None:
    verdicts, _ = run(syn.a_world("supported", tasks=R.TASKS_STUDY_A))
    for task in R.TASKS_STUDY_A:
        cell = verdicts[f"H1[{task}/dynamics/abrupt]"]
        assert cell.status == V.SUPPORTED and cell.label == V.SECONDARY and "Q-dynamics" in cell.provisional_on
        assert cell.numbers["total_steps.holm_survives"] is True
        assert verdicts[f"H1[{task}/finetune/abrupt]"].status == V.NOT_COMPUTABLE  # no continuation gaps
    assert verdicts["H1[SafetyCarGoal1-v0/hazard/abrupt]"].status == V.SUPPORTED
    assert verdicts["H2[SafetyPointButton1-v0/hazard]"].status == V.SUPPORTED


def test_holm_survival_by_the_families_q_holm_families_gives_each_cell() -> None:
    A = study_a.StudyA(data.build_dataset([], data.Supplement(), mode="final"))
    ps = {(TASK, "hazard"): 0.01, (TASK, "dynamics"): 0.04, ("SafetyCarGoal1-v0", "hazard"): 0.03}
    survival = A.holm_survival(V.PROPOSED, "g1", lambda t, c: ps.get((t, c)))
    # conditions on PointGoal1: 0.01 * 2 = 0.02 and 0.04 survive; tasks under hazard: 0.01 * 2 = 0.02, 0.03 survive
    assert survival[(TASK, "hazard")] is True and survival[(TASK, "dynamics")] is True
    assert survival[("SafetyCarGoal1-v0", "hazard")] is True and survival[(TASK, "transfer")] is None
    ps[("SafetyCarGoal1-v0", "hazard")] = 0.04  # tasks under hazard: 0.02 and 0.04 still survive
    ps[(TASK, "dynamics")] = 0.03  # conditions: 0.02 and 0.03
    assert A.holm_survival(V.PROPOSED, "g2", lambda t, c: ps.get((t, c)))[(TASK, "dynamics")] is True
    ps[(TASK, "hazard")] = 0.03  # conditions: 0.03 * 2 = 0.06 > 0.05: neither survives
    survival = A.holm_survival(V.PROPOSED, "g3", lambda t, c: ps.get((t, c)))
    assert survival[(TASK, "hazard")] is False and survival[(TASK, "dynamics")] is False
    assert survival[("SafetyCarGoal1-v0", "hazard")] is False
    # Q-holm-families' other reading: a claim in two families survives Holm in at least one of them
    # (the primary cell: conditions on PointGoal1 0.06 > 0.05 and tasks under hazard 0.06; neither survives)
    either = A.holm_survival(V.alternative("Q-holm-families"), "g3", lambda t, c: ps.get((t, c)))
    assert either[(TASK, "hazard")] is False and either[(TASK, "transfer")] is None
    # Q-holm-families (answered): every cell, one that differs from the primary in condition only or in
    # task only included, must survive Holm in both of its families; the other reading needs one of them
    car = "SafetyCarGoal1-v0"
    ps = {(TASK, "hazard"): 0.001, (TASK, "dynamics"): 0.04, (car, "dynamics"): 0.03,
          ("SafetyPointButton1-v0", "dynamics"): 0.9}
    # tasks under dynamics: 0.03 * 3 = 0.09 > 0.05, nothing survives; conditions on PointGoal1: 0.002, 0.04 survive;
    # conditions on CarGoal1: 0.03 alone survives
    both = A.holm_survival(V.PROPOSED, "g4", lambda t, c: ps.get((t, c)))
    either = A.holm_survival(V.alternative("Q-holm-families"), "g4", lambda t, c: ps.get((t, c)))
    assert both[(TASK, "dynamics")] is False and either[(TASK, "dynamics")] is True  # condition only: both families
    assert both[(car, "dynamics")] is False and either[(car, "dynamics")] is True
    ps = {(TASK, "hazard"): 0.001, (car, "hazard"): 0.03, (car, "dynamics"): 0.03}
    # tasks under hazard: 0.002, 0.03 survive; conditions on CarGoal1: 0.06 > 0.05, nothing survives
    both = A.holm_survival(V.PROPOSED, "g5", lambda t, c: ps.get((t, c)))
    either = A.holm_survival(V.alternative("Q-holm-families"), "g5", lambda t, c: ps.get((t, c)))
    assert both[(car, "hazard")] is False and either[(car, "hazard")] is True  # task only: both families too
    assert both[(car, "dynamics")] is False and either[(car, "dynamics")] is True  # tasks under dynamics: 0.03 alone


def test_a_cell_differing_from_the_primary_in_condition_only_must_survive_both_its_families(monkeypatch) -> None:
    """Q-holm-families (answered): PointGoal1 x dynamics survives Holm among PointGoal1's conditions but
    not among the three tasks under dynamics (3p > 0.05), so it does not survive: the claim is INCONCLUSIVE (it was
    decided by the condition family alone, which left the three-task family's error uncontrolled)."""
    rows = syn.a_world("supported", tasks=R.TASKS_STUDY_A, groups=("main",))
    for r in rows:
        e = syn.noise(r["seed"])
        if r["task"] != TASK:
            r["gap_dynamics"] = 3.0 + 0.3 * e  # null dynamics effect on the other tasks
        elif r["N"] == 0.5:
            r["gap_dynamics"] = 3.0 + 6.0 + 5.0 * e  # Delta(0.50) = 6, Welch p ~ 0.03
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset())  # every key open: the other reading is computed
    cell = run(rows)[0][f"H1[{TASK}/dynamics/abrupt]"]
    assert cell.proposal_status == V.INCONCLUSIVE
    assert cell.readings["Q-holm-families"] == V.SUPPORTED and "Q-holm-families" in cell.undecided_by
    for control in R.STEP_MATCHING_CONTROLS:
        assert 0.025 < cell.numbers[f"{control}.Delta(0.50)_p"] < 0.05  # 2p < 0.05 < 3p
        assert cell.numbers[f"{control}.holm_survives"] is False
        assert cell.numbers[f"{control}.S1_delta050_positive_excluding_zero"] is False
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset({"Q-holm-families"}))
    assert run(rows)[0][f"H1[{TASK}/dynamics/abrupt]"].status == V.INCONCLUSIVE


def test_a_cell_differing_from_the_primary_in_task_only_must_survive_its_four_condition_family() -> None:
    """Q-holm-families: H1 on CarGoal1 x hazard passes Holm among the three tasks under hazard (the other two tasks'
    effects are large) but not among CarGoal1's four conditions (hazard and dynamics both at p ~ 0.03: 2p > 0.05), so
    it does not survive; under the key's other reading (one family suffices) it does."""
    car = "SafetyCarGoal1-v0"
    eff = _effect_with_p(0.03)
    rows = []
    for spec in syn.specs(tasks=R.TASKS_STUDY_A, groups=("main",)):
        e = NOISE5[spec.seed]
        if spec.task != car:
            gap = 2.0 + 12.0 * spec.N + 0.3 * e
        elif spec.N == 0.5:
            gap = eff + 3.0 * e
        elif spec.N == 0.0:
            gap = -0.9 * 3.0 * e
        else:
            gap = {0.10: 0.1, 0.25: 0.2}[spec.N] * eff + 0.01 * e
        rows.append(syn.study_a_row(spec, cost=25.0 + 0.5 * syn.noise(spec.seed),
                                    gaps={"hazard": gap, "dynamics": gap, "finetune": None, "transfer": None}))
    A = study_a.StudyA(data.build_dataset(rows, data.Supplement(), mode="final"))
    out = {reading: study_a.h1_outcome(A, reading, task=car, condition="hazard", shape="abrupt", holm=True)
           for reading in (V.PROPOSED, V.alternative("Q-holm-families"))}
    for control in R.STEP_MATCHING_CONTROLS:
        assert out[V.PROPOSED].numbers[f"{control}.Delta(0.50)_p"] == pytest.approx(0.03, abs=1e-6)
        assert out[V.PROPOSED].numbers[f"{control}.holm_survives"] is False
        assert out[V.alternative("Q-holm-families")].numbers[f"{control}.holm_survives"] is True
    assert out[V.PROPOSED].status == V.INCONCLUSIVE


def test_the_final_checkpoint_estimand_and_its_agreement() -> None:
    rows = syn.a_world("supported", groups=("main",))
    records = []
    for row in where(rows, task=TASK):
        spec = SPECS[row["run_id"]]
        records.append(("final_battery", syn.battery_record(row, "hazard", kind="final_battery", step=spec.total_steps)))
        final_measure = syn.battery_record(row, "dynamics", kind="final_battery", step=spec.total_steps)
        final_measure.update(condition="measurement", measurement_cost=None, gap=None,
                             episodes=syn.episodes(row["measurement_cost"]))
        records.append(("final_battery", final_measure))
    verdicts, _ = run(rows, supplement(records))
    final = verdicts["A-final-estimand"]
    assert final.label == V.EXPLORATORY and final.status == V.SUPPORTED
    assert final.numbers["total_steps.agree"] is True and final.numbers["constrained_steps.agree"] is True
    assert final.numbers["estimand"] == "final"
    # Q-training-age-systematic: the constrained-steps arms train T / (1 - N) steps, so their matched checkpoints are
    # older than the reference's (named on H1), but the final checkpoint's training age is its total steps by
    # construction: the final-checkpoint estimand carries no training-age note
    assert any(n.startswith("constrained_steps: training ages differ systematically") for n in verdicts["H1"].notes)
    assert not any("training ages differ" in n for n in final.notes)


def test_rule_7a_brings_in_arms_matched_only_at_tolerance_5() -> None:
    rows = syn.a_world("supported", groups=("main",))
    records = []
    for row in late(rows, N=0.25):
        row["measurement_cost"] = 29.0  # off by 4: unmatched at 2.5, matched at 5.0
        records.append(("sensitivity_battery", syn.battery_record(row, "hazard", kind="sensitivity_battery")))
        row["gap_hazard"] = row["gap_dynamics"] = None  # rule 6: no battery for an unmatched arm
    verdicts, _ = run(rows, supplement(records))
    primary, sensitivity = verdicts["H1"], verdicts["A-rule7a"]
    assert "total_steps.Delta(0.25)" not in primary.numbers and any("Delta(0.25)" in n for n in primary.notes)
    assert sensitivity.numbers["total_steps.Delta(0.25)"] is not None and sensitivity.label == V.EXPLORATORY
    # at 2.5 the unmatched N = 0.25 arm is not compared (rule 6): the trend is read without it, but F1 and the
    # point ordering name Delta(0.25), so H1 is decided only where the known Deltas decide it (here they do not)
    assert primary.proposal_status == V.NOT_COMPUTABLE and "Delta(0.25)" in primary.reason and "rule 6" in primary.reason
    assert primary.numbers["total_steps.not_compared_rule6"] == [0.25] and primary.numbers["total_steps.support"] is None
    assert sensitivity.status == V.SUPPORTED


def test_recovery_time_and_return_claims() -> None:
    rows = syn.a_world("supported", groups=("main",))
    records = []
    for row in where(rows, task=TASK):
        spec = SPECS[row["run_id"]]
        recovery = None if spec.N == 0 else (600_000 if spec.onset_shape == "abrupt" else 300_000) + 20_000 * spec.seed
        records.append(("training", syn.training_record(row, spec=spec, recovery=recovery)))
    verdicts, cat = run(rows, supplement(records))
    claim = verdicts["A-recovery[SafetyPointGoal1-v0/total_steps/abrupt-ramp/N0.50]"]
    assert claim.status == V.SUPPORTED and claim.numbers["diff"] == pytest.approx(300_000)
    assert claim.numbers["holm_family_size"] == 1 and claim.label == V.SECONDARY
    # the three-task family keeps its two members whose data are still to come (CarGoal1, PointButton1), held at
    # p = 1 for the survival the claim reports
    assert claim.numbers["holm_family_incomplete"] == 2
    assert claim.numbers["holm_adjusted_p_worst"] == pytest.approx(min(1.0, 3 * claim.numbers["p"]))
    assert verdicts["A-recovery[SafetyCarGoal1-v0/total_steps/abrupt-ramp/N0.50]"].status == V.NOT_COMPUTABLE
    ret = verdicts["A-return[SafetyPointGoal1-v0/total_steps/N0.50]"]
    # no measurement records: incomplete in final mode, never a switch to the final checkpoint's return
    assert ret.proposal_status == V.NOT_COMPUTABLE and "no measurement-set return" in ret.reason
    assert not any(v.status == V.FALSIFIED for k, v in verdicts.items() if k.startswith(("A-recovery", "A-return")))
    controller = {r["run_id"]: r for r in cat.tables["A_controller"]}
    assert controller["A-PointGoal1-N0.50-abrupt-total-s0"]["overshoot"] == 1.0


def test_censored_recovery_enters_above_its_horizon() -> None:
    """Q-controller-quantities (answered): a run that never recovers enters one epoch above its horizon, so it is
    told apart from a recovery in the last epoch."""
    rec = {"recovery_censored": True, "total_steps": R.TOTAL_STEPS, "onset_step": 5_000_000, "recovery_steps": None}
    assert study_a._recovery(rec) == 5_000_000.0 + R.STEPS_PER_EPOCH
    last_epoch = {**rec, "recovery_censored": False, "recovery_steps": 5_000_000}
    assert study_a._recovery(last_epoch) == 5_000_000.0 != study_a._recovery(rec)
    assert study_a._recovery({"recovery_censored": False, "recovery_steps": 20_000}) == 20_000.0
    assert study_a._recovery({"recovery_censored": None, "recovery_steps": None}) is None


def test_a_recovery_contrast_censors_both_arms_at_the_shorter_horizon() -> None:
    """The data control trains T + N·T steps (15,000,000 at N = 0.50), the untreated arm T: its contrast reads both
    arms up to the untreated arm's 5,000,000 steps after onset, so neither its longer censoring value nor a recovery
    only it can observe enters as recovery."""
    rows = syn.a_world("supported", groups=("main", "treatment"))
    untreated, data_control = late(rows), late(rows, treatment="additional_constrained")
    assert {SPECS[r["run_id"]].total_steps for r in data_control} == {R.TOTAL_STEPS + R.TOTAL_STEPS // 2}
    records = []
    for row in untreated:
        spec = SPECS[row["run_id"]]
        records.append(("training", syn.training_record(row, spec=spec, recovery=1_000_000 + 20_000 * spec.seed)))
    for row in data_control:  # seeds 0-2 never recover; seeds 3 and 4 recover 7,000,000 steps after onset
        spec = SPECS[row["run_id"]]
        records.append(("training", syn.training_record(row, spec=spec, recovery=None if spec.seed < 3 else 7_000_000)))
    verdicts, _ = run(rows, supplement(records))
    claim = verdicts["A-recovery[SafetyPointGoal1-v0/additional_constrained/treated-untreated/N0.50]"]
    horizon = R.TOTAL_STEPS - 5_000_000
    expected = (horizon + R.STEPS_PER_EPOCH) - (1_000_000 + 20_000 * 2)
    # every data-control seed censored at 5M + one epoch (R.STEPS_PER_EPOCH)
    assert claim.numbers["diff"] == pytest.approx(expected)
    assert any("shorter arm" in n for n in claim.notes)


NOISE5 = (-1.0, 1.0, -0.5, 0.5, 0.0)


def _effect_with_p(target: float, sd: float = 3.0) -> float:
    """The effect at which x = eff + sd * NOISE5 against y = -0.9 sd * NOISE5 has Welch p = ``target``."""
    def p_for(eff: float) -> float:
        return stats.welch({s: eff + sd * n for s, n in enumerate(NOISE5)}, {s: -0.9 * sd * n for s, n in enumerate(NOISE5)}).p
    lo, hi = 0.0, 20.0
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if p_for(mid) > target else (lo, mid)
    return (lo + hi) / 2


def test_a_secondary_claim_waits_on_an_incomplete_member_of_its_holm_family() -> None:
    """Q-holm-families: "Families shrink only to cells not tested", never to cells whose data are still to come: a
    claim that passes Holm only because a sibling's data are missing is NOT_COMPUTABLE, naming the sibling, and is
    decided where any p-value the sibling brings gives the same answer."""
    sd = 3.0
    est = {name: stats.estimate_two_arms({s: eff + sd * n for s, n in enumerate(NOISE5)},
                                         {s: -0.9 * sd * n for s, n in enumerate(NOISE5)}, analysis_id=f"holm-{name}")
           for name, eff in (("p03", _effect_with_p(0.03)), ("p001", _effect_with_p(0.001)), ("p90", 0.1))}
    assert est["p03"].welch.p == pytest.approx(0.03) and est["p90"].welch.p > 0.5

    def claims(sibling):
        cells = {("a",): (est["p03"], ""), ("d",): (est["p001"], ""), ("b",): sibling}
        verdicts = study_a._claims(make_cells=lambda r: cells, family_of=lambda c: 0, id_of=lambda c: f"claim[{c[0]}]",
                                   statement_of=lambda c: "the difference is not zero", title="t", table_key="A-recovery",
                                   label=V.SECONDARY, group="g")
        return {v.id: v for v in verdicts}

    complete = claims((est["p90"], ""))  # three members: 0.001 * 3, then 0.03 * 2 = 0.06 > 0.05
    assert complete["claim[a]"].proposal_status == V.INCONCLUSIVE
    untested = claims((None, "an arm has no completed run"))  # not tested: the family shrinks to two (0.03 * 1)
    assert untested["claim[a]"].proposal_status == V.SUPPORTED
    waiting = claims((None, "incomplete: no value for ['A-run-s4']"))
    claim = waiting["claim[a]"]
    assert claim.proposal_status == V.NOT_COMPUTABLE and "A-run-s4" in claim.reason and "Holm" in claim.reason
    assert waiting["claim[d]"].proposal_status == V.SUPPORTED  # 0.001 * 3 survives whatever the sibling brings


@pytest.mark.usefixtures("keys_open")
def test_a_secondary_box_waits_on_an_incomplete_cell_of_its_holm_family() -> None:
    """Box H1 on a secondary cell (PointButton1 x dynamics): with CarGoal1's dynamics gap of one N = 0.50 run not yet
    evaluated, the three-task family must not shrink to two and let the cell pass Holm (Q-holm-families: every cell,
    one that differs from the primary in condition only included, must survive Holm in both its families)."""
    button = "SafetyPointButton1-v0"
    eff = {TASK: 0.1, "SafetyCarGoal1-v0": _effect_with_p(0.04), button: _effect_with_p(0.02)}
    rows = []
    for spec in syn.specs(tasks=R.TASKS_STUDY_A, groups=("main",)):
        e = NOISE5[spec.seed]
        if spec.N == 0.5:
            dyn = eff[spec.task] + 3.0 * e
        elif spec.N == 0.0:
            dyn = -3.0 * e * 0.9
        else:
            dyn = {0.10: 0.1, 0.25: 0.2}[spec.N] * eff[spec.task] + 0.01 * e
        rows.append(syn.study_a_row(spec, cost=25.0 + 0.5 * syn.noise(spec.seed),
                                    gaps={"hazard": 2 + 12 * spec.N + 0.3 * e, "dynamics": dyn, "finetune": None,
                                          "transfer": None}))

    def h1_dynamics(rs):
        A = study_a.StudyA(data.build_dataset(rs, data.Supplement(), mode="final"))
        return study_a.h1_outcome(A, V.PROPOSED, task=button, condition="dynamics", shape="abrupt", holm=True)

    complete = h1_dynamics(rows)  # three tasks: 0.02 * 3 = 0.06 > 0.05 (conditions on PointButton1: 0.02 survives)
    assert complete.status == V.INCONCLUSIVE
    assert {complete.numbers[f"{c}.holm_survives"] for c in R.STEP_MATCHING_CONTROLS} == {False}
    cars = [r for r in rows if r["task"] == "SafetyCarGoal1-v0" and r["N"] == 0.5 and r["onset_shape"] == "abrupt"
            and r["seed"] == 4]  # under both controls (one known INCONCLUSIVE control would decide the box)
    for car in cars:
        car["gap_dynamics"] = None  # its dynamics battery is not evaluated yet
    waiting = h1_dynamics(rows)
    assert waiting.status == V.NOT_COMPUTABLE and "Holm" in waiting.reason
    assert all(car["run_id"] in waiting.reason for car in cars)
    assert {waiting.numbers[f"{c}.holm_survives"] for c in R.STEP_MATCHING_CONTROLS} == {None}

    # an arm, or a whole task, with no completed run yet is data still to come unless a recorded cut removed it
    # (the three-task family must not shrink to two, which would let the cell pass Holm at 0.02 * 2)
    def h1_dynamics_cut(rs, cuts=()):
        A = study_a.StudyA(data.build_dataset(rs, data.Supplement(), mode="final", cuts=cuts))
        return study_a.h1_outcome(A, V.PROPOSED, task=button, condition="dynamics", shape="abrupt", holm=True)

    no_car_arm = [r for r in rows if not (r["task"] == "SafetyCarGoal1-v0" and r["N"] == 0.5)]
    no_car = [r for r in rows if r["task"] != "SafetyCarGoal1-v0"]
    for rs in (no_car_arm, no_car):
        out = h1_dynamics_cut(rs)
        assert out.status == V.NOT_COMPUTABLE and "Holm" in out.reason and "SafetyCarGoal1-v0" in out.reason
        assert {out.numbers[f"{c}.holm_survives"] for c in R.STEP_MATCHING_CONTROLS} == {None}
    ds = data.build_dataset(no_car, data.Supplement(), mode="final")
    assert "A-CarGoal1-N0.50-abrupt-total" in ds.awaited and "A-PointGoal1-N0.50-abrupt-total" not in ds.awaited
    member = study_a._task_member(study_a.StudyA(ds), V.PROPOSED, "SafetyCarGoal1-v0", "matched")
    assert member[0] is False and isinstance(member[1], stats.Incomplete)
    cut = h1_dynamics_cut(no_car, cuts=("pid", "car"))  # the recorded cut removes the task: the family is two tasks
    assert {cut.numbers[f"{c}.holm_survives"] for c in R.STEP_MATCHING_CONTROLS} == {True}
    cut_ds = data.build_dataset(no_car, data.Supplement(), mode="final", cuts=("pid", "car"))
    assert not any("CarGoal1" in a for a in cut_ds.awaited)
    assert study_a._task_member(study_a.StudyA(cut_ds), V.PROPOSED, "SafetyCarGoal1-v0", "matched") == (False, None)


def test_a_task_a_recorded_cut_removed_is_not_tested_not_incomplete() -> None:
    """With the recorded cut 'car' the Car cells are not tested (NOT_TESTED), never NOT_COMPUTABLE 'matching
    incomplete', as if the task's data were still to come."""
    button = "SafetyPointButton1-v0"
    rows = syn.a_world("supported", tasks=(R.PRIMARY_TASK, button), groups=("main", "treatment", "controller"))
    verdicts, _ = run(rows, cuts=("pid", "car"))
    car = {k: v for k, v in verdicts.items() if "SafetyCarGoal1-v0" in k}
    assert {k.split("[")[0] for k in car} >= {"H1", "H2", "H3a", "H3b", "H3c", "H4", "A-recovery", "A-return"}
    for key, v in car.items():
        assert v.status == V.NOT_TESTED, (key, v.status, v.reason)
        assert "removed by the recorded Part 6.1 cut car" in v.reason and "incomplete" not in v.reason, (key, v.reason)
    assert verdicts[f"H1[{button}/transfer/abrupt]"].status != V.NOT_TESTED
    # without the cut the task's data are still to come: incomplete, not 'not tested'
    uncut, _ = run(rows)
    assert uncut["H1[SafetyCarGoal1-v0/transfer/abrupt]"].status == V.NOT_COMPUTABLE


def test_a_task_a_recorded_cut_removed_is_not_tested_whatever_completed_before_the_cut() -> None:
    """Part 6.1 cut 2 ('Drop SafetyCarGoal1-v0 from every group of Study A') takes the task out of every comparison
    whether or not some of its runs completed before the cut, as it does an arm (``StudyA.arms``): its cells are
    NOT_TESTED and it leaves the three-task Holm families (Q-holm-families: "Families shrink only to cells not tested
    (... a recorded Part 6.1 cut ...)"), never an incomplete member whose data will not come."""
    car = "SafetyCarGoal1-v0"
    rows = [r for r in syn.a_world("supported", tasks=R.TASKS_STUDY_A, groups=("main",))
            if r["task"] != car or (r["seed"] == 0 and r["N"] in (0.0, 0.5))]  # completed before the cut
    A = study_a.StudyA(data.build_dataset(rows, data.Supplement(), mode="final", cuts=("pid", "car")))
    assert A.task_cut_by(car) == ("car",) and not A.task_awaited(car)
    assert study_a._task_member(A, V.PROPOSED, car, "matched") == (False, None)
    out = study_a.h1_outcome(A, V.PROPOSED, task=car, condition="hazard", shape="abrupt", holm=True)
    assert out.status == V.NOT_TESTED and "removed by the recorded Part 6.1 cut car" in out.reason
    verdicts, _ = run(rows, cuts=("pid", "car"))
    cells = {k: v for k, v in verdicts.items() if car in k}
    assert {k.split("[")[0] for k in cells} >= {"H1", "H2", "H3a", "H3b", "H3c", "H4", "A-recovery", "A-return"}
    for key, v in cells.items():
        assert v.status == V.NOT_TESTED, (key, v.status, v.reason)


def test_h3a_never_drops_a_late_arm_that_is_incomplete_or_not_run() -> None:
    """H3 (a) reads every late arm: one whose matching waits on a missing cost, or that has no completed run, makes
    it NOT_COMPUTABLE (as box H1), and only rule 6 leaves an arm out."""
    rows = syn.a_world("supported", groups=("main",))
    A = study_a.StudyA(data.build_dataset(rows, data.Supplement(), mode="final"))
    full = study_a.h3a_outcome(A, V.PROPOSED, holm=False)
    assert full.status == V.SUPPORTED and full.numbers["points"] == 60
    pending = [dict(r) for r in rows]
    hit = next(r for r in pending if r["N"] == 0.25 and r["onset_shape"] == "ramp"
               and r["step_matching"] == "constrained_steps" and r["seed"] == 3)
    hit["measurement_cost"] = None  # this run has not been measured yet: its arm cannot be matched
    A2 = study_a.StudyA(data.build_dataset(pending, data.Supplement(), mode="final"))
    out = study_a.h3a_outcome(A2, V.PROPOSED, holm=False)
    assert out.status == V.NOT_COMPUTABLE and "A-PointGoal1-N0.25-ramp-constrained" in out.reason
    assert study_a.h1_outcome(A2, V.PROPOSED, shape="ramp").status == V.NOT_COMPUTABLE  # the same data, box H1
    not_run = [r for r in rows if not (r["N"] == 0.25 and r["onset_shape"] == "ramp")]
    out = study_a.h3a_outcome(study_a.StudyA(data.build_dataset(not_run, data.Supplement(), mode="final")),
                              V.PROPOSED, holm=False)
    assert out.status == V.NOT_COMPUTABLE and "N = 0.25 (ramp" in out.reason and "no completed run" in out.reason
    # the recorded cut 'drop_n010' removes the N = 0.10 arms: they are not expected
    cut = [r for r in rows if r["N"] != 0.1]
    out = study_a.h3a_outcome(study_a.StudyA(data.build_dataset(cut, data.Supplement(), mode="final",
                                                                cuts=("pid", "car", "fewshot_short",
                                                                      "treatments_controllers_n050", "drop_n010"))),
                              V.PROPOSED, holm=False)
    assert out.status != V.NOT_COMPUTABLE and out.numbers["points"] == 40


def test_holm_decisions_decide_only_what_an_incomplete_member_cannot_change() -> None:
    inc = stats.Incomplete("x: incomplete")
    results, survives, incomplete = stats.holm_decisions({"a": 0.03, "b": 0.001, "c": inc, "n": None}, 0.05)
    assert set(results) == {"a", "b"} and incomplete == {"c": "x: incomplete"}
    assert survives == {"a": None, "b": True}  # a: 0.03 * 1 survives without c, 0.03 * 2 not with c at p = 1
    _, survives, _ = stats.holm_decisions({"a": 0.2, "c": inc}, 0.05)
    assert survives == {"a": False}  # not rejected without c: adding a member never lowers an adjusted p-value
    _, survives, incomplete = stats.holm_decisions({"a": 0.03, "n": None}, 0.05)
    assert survives == {"a": True} and incomplete == {}


def test_tables_report_selection_ages_iqm_and_treatments_at_other_n() -> None:
    rows = syn.a_world("supported")
    records = []
    for row in late(rows) + where(rows, task=TASK, N=0.0):
        records.append(("measurement", syn.measurement_record(row)))
        rec = syn.battery_record(row, "hazard")
        row["gap_hazard"] = rec["gap"]
        records.append(("battery", rec))
    _, cat = run(rows, supplement(records), iqm=True)
    ages = {r["arm_id"]: r for r in cat.tables["A_training_age"] if r["contrast"] == "vs reference"}
    assert ages["A-PointGoal1-N0.00"]["mean_training_age"] == R.TOTAL_STEPS
    constrained = ages["A-PointGoal1-N0.50-abrupt-constrained"]
    assert constrained["mean_training_age"] == 2 * R.TOTAL_STEPS and constrained["systematic"] is True
    assert constrained["mean_constrained_steps"] == R.TOTAL_STEPS
    iqm = {(r["x"], r["y"], r["outcome"]): r for r in cat.tables["A_iqm"]}
    total = iqm[("A-PointGoal1-N0.50-abrupt-total", "A-PointGoal1-N0.00", "gap_hazard|matched")]
    assert total["iqm_diff"] == pytest.approx(total["iqm_x"] - total["iqm_y"])
    assert total["iqm_lo"] <= total["iqm_diff"] <= total["iqm_hi"] and "H1|" in total["used_by"]
    assert total["iqm_resamples"] == R.BOOTSTRAP_RESAMPLES and total["label"] == "descriptive"
    # Q-iqm: "the IQM of a gap is IQM(C_cond) - IQM(C_ID)"
    late_rows = late(rows)
    cond = [c for r in late_rows for c in next(x for k, x in records if k == "battery" and x["run_id"] == r["run_id"]
                                                 and x["condition"] == "hazard")["episodes"]["episode_costs"]]
    base = [c for r in late_rows for c in syn.measurement_record(r)["episodes"]["episode_costs"]]
    assert total["iqm_x"] == pytest.approx(stats.iqm(cond) - stats.iqm(base))
    constrained = iqm[("A-PointGoal1-N0.50-abrupt-constrained", "A-PointGoal1-N0.00", "gap_hazard|matched")]
    assert constrained["note"].startswith("not computable") and "iqm_diff" not in constrained  # no episodes recorded
    # every tabled gap contrast has its row, the other conditions and the other boxes included
    assert ("A-PointGoal1-N0.50-abrupt-total-reset", "A-PointGoal1-N0.50-abrupt-total", "gap_hazard|matched") in iqm
    assert any(k[2] == "gap_dynamics|matched" for k in iqm)
    other = [r for r in cat.tables["A_treatments_other_N"] if r["N"] == 0.1 and r["variant"] == "reset"
             and r["condition"] == "hazard" and r["task"] == TASK]
    assert other and other[0]["label"] == "descriptive" and "diff" in other[0]
    assert cat.tables["A_selection"][0]["training_age"] is not None
    assert cat.tables["A_estimates"]


def test_h3a_and_h3c_falsified() -> None:
    rows = syn.a_world("supported")
    arms = sorted({r["arm"] for r in rows})
    for row in where(rows, task=TASK):  # metrics at onset that scatter without any relation to the gap
        i = arms.index(row["arm"])
        row["dormant_onset"] = 0.1 + 0.01 * ((row["seed"] * 4 + i) % 7)
        row["rank_onset"] = 30.0 + ((row["seed"] + i * 4) % 5)
    records = [(kind, dict(rec, plasticity_check={**rec["plasticity_check"], "dormant_trainable": 0.5,
                                                   "rank_trainable": 20.0 + syn.noise(int(rec["run_id"][-1]))}))
               if rec.get("plasticity_check") else (kind, rec) for kind, rec in training_records(rows)]
    verdicts, _ = run(rows, supplement(records))
    h3a = verdicts["H3a"]
    assert h3a.status == V.FALSIFIED and h3a.numbers["rho_dormant_lo"] <= 0 <= h3a.numbers["rho_dormant_hi"]
    assert verdicts["H3c"].status == V.FALSIFIED  # the check fails for both interventions
    assert verdicts["H3"].status == V.FALSIFIED and not any("mediation" in n for n in verdicts["H3"].notes)


# ---------------------------------------------------------------------------
# Holm only on claims of an effect; H2's missing fractions; readings; annotations
# ---------------------------------------------------------------------------

SHIFT = {0: -1.0, 1: 0.3, 2: -0.6, 3: 0.9, 4: 0.4}  # mean 0; D(t) - mean has Welch p about 0.047 at a mean of -1.2


def _treated_like_untreated(rows: list[dict], treatment: str, by: float) -> None:
    """gap(treatment) = gap(untreated) + by + SHIFT[seed] under hazard and dynamics (N = 0.50, PointGoal1)."""
    untreated = {r["seed"]: r for r in late(rows)}
    for row in late(rows, treatment=treatment):
        u = untreated[row["seed"]]
        for condition in ("hazard", "dynamics"):
            row[f"gap_{condition}"] = u[f"gap_{condition}"] + by + SHIFT[row["seed"]]


def test_h3b_reads_does_not_reduce_on_the_uncorrected_interval() -> None:
    rows = syn.a_world("supported")
    _treated_like_untreated(rows, "additional_constrained", -1.2)  # reduces (p ~ 0.047) but fails Holm (2 x 0.047)
    verdicts, _ = run(rows)
    h3b = verdicts["H3b"]
    assert h3b.numbers["D(additional_constrained)_welch_hi"] < 0 < h3b.numbers["D(additional_constrained)_p"] < 0.05
    assert h3b.numbers["holm_survives(additional_constrained)"] is False
    assert h3b.numbers["reduces_uncorrected(additional_constrained)"] is True
    assert h3b.numbers["does_not_reduce(additional_constrained)"] is False
    # Holm must not turn a reduction into "does not reduce": H3 (b) is not supported
    assert h3b.proposal_status == V.INCONCLUSIVE and h3b.status != V.SUPPORTED
    assert h3b.readings.get("Q-holm-families", h3b.proposal_status) == V.INCONCLUSIVE


def test_h3b_as_much_as_reset_needs_a_reduction_by_additional_training() -> None:
    """Box H3 falsifies (b) when "additional constrained training reduces the gap as much as reset".
    If additional training widens the gap (by 3) and reset widens it more (by 5) while injection narrows it, no
    falsifying clause holds (nothing is reduced by additional training; reset reduces nothing): INCONCLUSIVE, not
    FALSIFIED (the analysis spec's D_add <= D_reset alone would falsify it)."""
    rows = syn.a_world("supported")
    base = {r["seed"]: r["gap_hazard"] for r in late(rows)}
    for treatment, by in (("additional_constrained", 3.0), ("reset", 5.0), ("injection", -5.0)):
        for row in late(rows, treatment=treatment):
            row["gap_hazard"] = base[row["seed"]] + by + 0.1 * syn.noise(row["seed"])
    h3b = run(rows)[0]["H3b"]
    n = h3b.numbers
    assert n["D(additional_constrained)"] == pytest.approx(3.0) and n["D(reset)"] == pytest.approx(5.0)
    assert n["F_additional_reduces_as_much_as_reset"] is False and n["F_reset_reduces_injection_does_not"] is False
    assert n["F_none_reduces"] is False and n["support"] is False
    assert h3b.proposal_status == V.INCONCLUSIVE and h3b.status == V.INCONCLUSIVE
    # additional training that reduces the gap as much as reset still falsifies (b)
    for treatment, by in (("additional_constrained", -6.0), ("reset", -5.0)):
        for row in late(rows, treatment=treatment):
            row["gap_hazard"] = base[row["seed"]] + by + 0.1 * syn.noise(row["seed"])
    h3b = run(rows)[0]["H3b"]
    assert h3b.numbers["F_additional_reduces_as_much_as_reset"] is True and h3b.proposal_status == V.FALSIFIED


def test_h3b_is_not_falsified_by_an_injection_that_reduces_but_fails_holm() -> None:
    rows = syn.a_world("supported")
    _treated_like_untreated(rows, "injection", -1.2)  # a reduction excluding zero, not surviving Holm
    verdicts, _ = run(rows)
    h3b = verdicts["H3b"]
    assert h3b.numbers["reduces(injection)"] is False and h3b.numbers["reduces_uncorrected(injection)"] is True
    assert h3b.numbers["F_reset_reduces_injection_does_not"] is False and h3b.numbers["F_none_reduces"] is False
    assert h3b.proposal_status == V.INCONCLUSIVE  # neither supported (Holm) nor falsified (the absence is not shown)


def _unmatch(rows: list[dict], N: float) -> None:
    for row in late(rows, N=N, shape="ramp") + late(rows, N=N, shape="ramp", control="constrained_steps"):
        row["measurement_cost"] = row["selection_cost_at_match"] = 29.0  # off by 4 on both evaluation sets (rule 6)
        row["gap_hazard"] = row["gap_dynamics"] = None


@pytest.mark.usefixtures("keys_open")
def test_h2_is_not_supported_without_the_fractions_its_support_clause_names() -> None:
    rows = syn.a_world("supported", groups=("main",))
    _unmatch(rows, 0.1)
    _unmatch(rows, 0.25)
    verdicts, _ = run(rows)
    h2 = verdicts["H2"]
    assert h2.status == V.NOT_COMPUTABLE and "E(0.10)" in h2.reason and "E(0.25)" in h2.reason
    assert "rule 6" in h2.reason and h2.numbers["total_steps.support"] is None
    h1 = verdicts["H1"]
    assert h1.proposal_status == V.SUPPORTED  # H1 reads the abrupt arms, all matched
    # on the ramp arms (Q-h1-shape's other reading) Delta(0.10) and Delta(0.25) are not compared: H1 turns on the key
    assert h1.readings["Q-h1-shape"] == V.NOT_COMPUTABLE and "Q-h1-shape" in h1.undecided_by
    # E(0.50) falsifies whatever the missing fractions show
    rows = syn.a_world("supported", groups=("main",))
    _unmatch(rows, 0.25)
    for row in late(rows, shape="ramp") + late(rows, shape="ramp", control="constrained_steps"):
        row["gap_hazard"] += 6.0  # ramp gap above the abrupt gap at N = 0.50
    verdicts, _ = run(rows)
    assert verdicts["H2"].status == V.FALSIFIED


def test_h2_reads_the_remaining_fractions_after_the_recorded_cut_drop_n010() -> None:
    """Part 6.1 cut 5, 'Drop N = 0.10 from the main sweep': recorded, it removes N = 0.10 from both clauses of H2."""
    rows = [r for r in syn.a_world("supported", groups=("main",)) if r["N"] != 0.1]  # Part 6.1 cut 5
    verdicts, _ = run(rows)
    assert verdicts["H2"].status == V.NOT_COMPUTABLE  # no record of a cut: the fraction is missing
    verdicts, _ = run(rows, cuts=CUTS)
    h2 = verdicts["H2"]
    assert h2.status == V.SUPPORTED and any("drop_n010" in n for n in h2.notes)
    assert "total_steps.E(0.10)" not in h2.numbers and h2.numbers["total_steps.E(0.25)"] == pytest.approx(3.0)


@pytest.mark.usefixtures("keys_open")
def test_surplus_seeds_are_a_reading_of_q_surplus_in_analysis() -> None:
    rows = syn.a_world("supported", groups=("main",))
    # Part 5.5's surplus rule (here five extra seeds, 5 to 9) for the primary-comparison arms (N = 0 and N = 0.50 on
    # SafetyPointGoal1-v0; the ramp arms by Q-surplus-arm-set's answer)
    surplus = [r for r in syn.a_world("supported", groups=("main",), seeds=range(5, 10))
               if r["task"] == TASK and r["N"] in (0.0, 0.5)]
    for row in surplus:
        if row["N"] == 0.5 and row["onset_shape"] == "abrupt":
            row["gap_hazard"] = -10.0  # seeds 5 to 9 of the N = 0.50 abrupt arms show the opposite of H1
    verdicts, cat = run(rows + surplus, surplus_extra=5)
    h1 = verdicts["H1"]
    assert h1.numbers["total_steps.Delta(0.50)_n"] == (10, 10)  # every completed seed (the decided reading)
    assert h1.proposal_status != V.SUPPORTED and h1.readings["Q-surplus-in-analysis"] == V.SUPPORTED
    assert h1.status == V.UNDECIDED and "Q-surplus-in-analysis" in h1.undecided_by
    # the estimate table shows the decided reading's estimate (ten seeds), never the other reading's
    row = next(r for r in cat.tables["A_estimates"]
               if r["analysis_id"].startswith("H1|SafetyPointGoal1-v0|hazard|total_steps|abrupt|matched|N0.50"))
    assert row["n_x"] == 10 and row["diff"] == pytest.approx(h1.numbers["total_steps.Delta(0.50)"])


def test_h3a_reads_the_registered_seeds_and_their_replacements_only() -> None:
    """Q-surplus-in-analysis (answered): H3 (a), like the H1 trend, resamples seed indices that each carry
    all of their levels ("with the same interval"), so its points are each late arm's registered seeds and their
    replacements (analysis.data.five_seed_view); the surplus seeds of the N = 0.50 arms join the matching and the
    two-arm estimates only."""
    rows = syn.a_world("supported", groups=("main",))
    surplus = [r for r in syn.a_world("supported", groups=("main",), seeds=range(5, 10))
               if r["task"] == TASK and r["N"] in (0.0, 0.5)]
    for row in surplus:
        if row["N"] == 0.5:  # were they points, the metrics would run against H3 (a)
            row["dormant_onset"], row["rank_onset"] = 0.0, 100.0 + (row["gap_hazard"] or 0.0)
    verdicts, _ = run(rows + surplus, surplus_extra=5)
    base, _ = run(rows)
    h3a, five = verdicts["H3a"], base["H3a"]
    assert five.status == V.SUPPORTED and h3a.status == five.status
    assert h3a.numbers["seeds"] == five.numbers["seeds"] == len(R.SEEDS)
    for name in ("points", "rho_dormant", "rho_dormant_lo", "rho_rank", "rho_rank_hi", "p_dormant", "p_rank"):
        assert h3a.numbers[name] == five.numbers[name], name
    assert verdicts["H1"].numbers["total_steps.Delta(0.50)_n"] == (10, 10)  # the estimates read every completed seed


def test_a_falsified_box_carries_its_part_5_7_bound_and_the_calibration_reading(monkeypatch) -> None:
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset())  # every key open: the other reading is computed
    verdicts, _ = run(syn.a_world("falsified"))
    h2 = verdicts["H2"]
    assert h2.proposal_status == V.FALSIFIED  # its bound is below the minimum effect: the falsification stands
    assert h2.numbers["total_steps.bound(E(0.50), 95% Welch)"] == h2.numbers["total_steps.E(0.50)_welch_hi"]
    assert h2.numbers["total_steps.bound(E(0.50), 95% Welch)_reading"] == V.BOUND_BELOW
    assert h2.readings.get("Q-falsification-calibration", V.FALSIFIED) == V.FALSIFIED
    # H4 falsified by the box (a remaining gap whose interval includes zero) but with an upper limit above 5:
    # Q-falsification-calibration (answered) reads it INCONCLUSIVE; the box conditions alone are in the
    # numbers, and the other reading (the box decides) gives FALSIFIED
    rows = syn.a_world("supported")
    for row in late(rows, controller_variant="warm_started"):
        row["gap_hazard"] = 2.0 + 8.0 * syn.noise(row["seed"])
    verdicts, _ = run(rows)
    h4 = verdicts["H4"]
    assert h4.proposal_status == V.INCONCLUSIVE and h4.numbers["G_interval_includes_zero"] is True
    assert h4.numbers["falsification"] is True
    assert h4.numbers["bound(G, 95% Welch)"] > R.MIN_EFFECT_STUDY_A
    assert h4.numbers["bound(G, 95% Welch)_reading"] == V.BOUND_ABOVE
    assert any("not below the minimum effect" in n for n in h4.notes)
    assert h4.readings["Q-falsification-calibration"] == V.FALSIFIED
    assert h4.status == V.UNDECIDED and "Q-falsification-calibration" in h4.undecided_by
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset({"Q-falsification-calibration"}))
    assert run(rows)[0]["H4"].status == V.INCONCLUSIVE
    # H3 (b): additional constrained training reduces the gap as much as reset
    rows = syn.a_world("supported")
    for row in late(rows, treatment="additional_constrained"):
        row["gap_hazard"] = 2.0 + 12.0 * 0.5 - 6.0 + 0.3 * syn.noise(row["seed"])
    h3b = run(rows)[0]["H3b"]
    assert h3b.proposal_status == V.FALSIFIED
    assert "bound(gap(additional_constrained) - gap(reset), 95% Welch)" in h3b.numbers


def test_detectable_size_is_beside_every_estimate_and_noted_on_a_supported_verdict() -> None:
    verdicts, _ = run(syn.a_world("supported"))
    h1 = verdicts["H1"]
    assert h1.numbers["total_steps.Delta(0.50)_detectable_d"] == pytest.approx(stats.detectable_d(5))
    assert h1.numbers["total_steps.Delta(0.50)_below_detectable"] is False  # a large effect: no note
    assert not any("Part 5.5" in n for n in h1.notes)
    x = {s: 1.0 + v for s, v in enumerate((0.0, 0.4, -0.4, 0.8, -0.8))}
    y = {s: v for s, v in enumerate((0.3, -0.3, 0.6, -0.6, 0.0))}
    est = stats.estimate_two_arms(x, y, analysis_id="d")
    assert est.welch.lo > 0 and est.below_detectable  # an interval excluding zero at |d| < 2.02
    notes = V.detectable_notes({"x - y": est})
    # Q-detectable-size: an annotation only; it never says the supported difference is not a finding
    assert notes and "the verdict stands" in notes[0] and "under 80 percent power" in notes[0]
    assert "not claimed as a finding" not in notes[0]


def test_rule_1_is_rechecked_on_every_row() -> None:
    rows = syn.a_world("supported", groups=("main",))
    edited = rows[3]
    window = sorted(c["step"] for c in edited["checkpoints"] if c["selection_cost"] is not None)
    edited["matched_checkpoint_step"] = window[0]  # not the step rule 1 chooses
    short = rows[4]
    short["checkpoints"] = short["checkpoints"][-3:]  # rule 1 needs ten checkpoints
    ds = data.build_dataset(rows, data.Supplement(), mode="final")
    assert any(edited["run_id"] in p and "rule 1" in p and str(window[-1]) in p for p in ds.problems)
    assert any(short["run_id"] in p and "cannot be re-checked" in p for p in ds.problems)
    recs = {r["run_id"]: r for r in ds.study_a}
    assert recs[edited["run_id"]]["rule_one_rechecked"] is False and recs[rows[0]["run_id"]]["rule_one_rechecked"] is True
    assert len([p for p in ds.problems if "rule 1" in p]) == 2  # every other row agrees


def test_a_row_whose_matched_step_is_not_rule_1s_choice_is_never_used() -> None:
    """The cost and gaps of a checkpoint rule 1 does not select make the arm
    incomplete (like a missing cost), so H1 is NOT_COMPUTABLE instead of decided on those gaps."""
    rows = syn.a_world("supported", groups=("main",))
    changed = [r for r in rows if r["N"] == 0.5 and r["onset_shape"] == "abrupt"]
    for r in changed:
        r["matched_checkpoint_step"] = max(c["step"] for c in r["checkpoints"]) - R.CHECKPOINT_INTERVAL_STEPS
        r["training_age"] = r["matched_checkpoint_step"]
    ds = data.build_dataset(rows, data.Supplement(), mode="final")
    assert len([p for p in ds.problems if "rule 1" in p]) == len(changed)
    A = study_a.StudyA(ds)
    for control in R.STEP_MATCHING_CONTROLS:
        arm = A.arms(TASK, N=0.5, shape="abrupt", control=control)
        assert A.arm_status(V.PROPOSED, "matched", arm) == matching.STATUS_INCOMPLETE
    h1 = study_a.h1_outcome(A, V.PROPOSED)
    assert h1.status == V.NOT_COMPUTABLE and "rule 1" in h1.reason
    assert all(h1.numbers.get(f"{c}.Delta(0.50)") is None for c in R.STEP_MATCHING_CONTROLS)


# ---------------------------------------------------------------------------
# Box H1 with a missing, unmatched or cut Delta; H3 (c) under Holm
# ---------------------------------------------------------------------------


def _abrupt(rows: list[dict], N: float) -> list[dict]:
    """The abrupt main-sweep arms at N under both step-matching controls (PointGoal1)."""
    return late(rows, N=N) + late(rows, N=N, control="constrained_steps")


def test_h1_is_not_decided_while_a_delta_it_names_is_incomplete() -> None:
    """Part 5.6 (analysis spec): "A missing enrichment field makes the analysis 'incomplete', never a silent drop"."""
    for kind in ("supported", "falsified"):
        rows = syn.a_world(kind, groups=("main",))
        for row in _abrupt(rows, 0.1) + _abrupt(rows, 0.25):
            row["gap_hazard"] = None  # matched, the battery not yet evaluated
        h1 = run(rows)[0]["H1"]
        assert h1.proposal_status == V.NOT_COMPUTABLE and "incomplete" in h1.reason
        assert _abrupt(rows, 0.1)[0]["run_id"] in h1.reason
        for control in R.STEP_MATCHING_CONTROLS:
            assert h1.numbers[f"{control}.incomplete"] == [0.1, 0.25]
            assert h1.numbers[f"{control}.trend_positive_excluding_zero"] is None  # never read on the arms that remain
        assert not any("cut 'drop_n010' removed" in n for n in h1.notes)  # no cut is recorded, none is claimed
    rows = syn.a_world("falsified", groups=("main",))
    late(rows, N=0.1)[3]["gap_hazard"] = None  # one seed of one arm is enough
    assert run(rows)[0]["H1"].proposal_status == V.NOT_COMPUTABLE


def test_h1_is_not_falsified_by_intervals_it_does_not_have() -> None:
    """Box H1: 'Falsified if. The intervals of Delta(0.10), Delta(0.25) and Delta(0.50) all include zero, or their
    point estimates are not ordered with N.' With the N = 0.10 arms unmatched (rule 6) and no cut recorded, F1 stays
    open and the trend is read without them (analysis spec: 'A missing or cut N=0.10/0.25 reduces the trend')."""
    rows = syn.a_world("falsified", groups=("main",))
    for row in _abrupt(rows, 0.1):
        row["measurement_cost"] = row["selection_cost_at_match"] = 35.0  # rule 6 on both evaluation sets
        row["gap_hazard"] = row["gap_dynamics"] = None  # no battery for an unmatched arm (pilot.enrichment)
    h1 = run(rows)[0]["H1"]
    assert h1.proposal_status == V.NOT_COMPUTABLE and "rule 6" in h1.reason and "Delta(0.10)" in h1.reason
    for control in R.STEP_MATCHING_CONTROLS:
        assert h1.numbers[f"{control}.F1_all_intervals_include_zero"] is None
        assert h1.numbers[f"{control}.not_compared_rule6"] == [0.1] and h1.numbers[f"{control}.trend_points"] == 15
        assert h1.numbers[f"{control}.trend_positive_excluding_zero"] is False  # read on the arms that remain
    assert not any("cut 'drop_n010' removed" in n for n in h1.notes)
    assert any(f"registers {R.H1_TREND_POINTS}" in n for n in h1.notes)
    # what the missing Delta cannot change decides: points out of order falsify H1 whatever Delta(0.10) is (F2)
    for row in late(rows, N=0.25) + late(rows, N=0.25, control="constrained_steps"):
        row["gap_hazard"] += 4.0  # Delta(0.25) = 4 > Delta(0.50) = 0
    h1 = run(rows)[0]["H1"]
    assert h1.proposal_status == V.FALSIFIED
    assert all(h1.numbers[f"{c}.F2_not_ordered"] is True for c in R.STEP_MATCHING_CONTROLS)


def test_h1_reads_the_remaining_fractions_after_the_recorded_cut_drop_n010() -> None:
    """Part 6.1 cut 5, 'Drop N = 0.10 from the main sweep': recorded, F1, F2 and the trend read the fractions left."""
    for kind, expected in (("supported", V.SUPPORTED), ("falsified", V.FALSIFIED)):
        rows = [r for r in syn.a_world(kind, groups=("main",)) if r["N"] != 0.1]  # Part 6.1 cut 5
        assert run(rows)[0]["H1"].proposal_status == V.NOT_COMPUTABLE  # no record of a cut: the arms are missing
        h1 = run(rows, cuts=CUTS)[0]["H1"]
        assert h1.proposal_status == expected
        assert any("cut 'drop_n010' removed" in n and "Part 6.1" in n for n in h1.notes)
        assert h1.numbers["total_steps.trend_points"] == 15 and "total_steps.Delta(0.10)" not in h1.numbers


BORDERLINE = {0: 0.012, 1: -0.004, 2: -0.008, 3: 0.006, 4: -0.006}  # mean 0: 0.014 lower gives Welch p about 0.028


def test_h3c_claims_are_holm_corrected_over_the_three_tasks() -> None:
    """Part 5.2: H3 is secondary, 'corrected for multiplicity within its family (... the three tasks another ...)'.
    The manipulation check is a training metric, so the task family is its only one; its falsifying absence
    ('the manipulation check fails for both interventions') is read on the uncorrected intervals."""
    specs = {s.run_id: s for s in syn.specs(tasks=R.TASKS_STUDY_A, groups=("main", "treatment"))}
    rows = syn.a_world("supported", tasks=R.TASKS_STUDY_A, groups=("main", "treatment"))
    records = []
    for row in rows:
        spec = specs[row["run_id"]]
        if not (spec.N == 0.5 and spec.onset_shape == "abrupt" and spec.step_matching == "total_steps"):
            continue
        e = syn.noise(spec.seed)
        if spec.task == TASK and spec.treatment in ("reset", "injection"):
            dormant, rank = 0.5 - 0.014 + BORDERLINE[spec.seed], 30.0 + e  # dormant lower at p ~ 0.028
        else:
            dormant, rank = 0.5 + 0.01 * e, 20.0 + e  # the untreated arms, and no effect on the other tasks
        records.append(("training", syn.training_record(row, spec=spec, dormant_check=dormant, rank_check=rank)))
    A = study_a.StudyA(data.build_dataset(rows, supplement(records), mode="final"))
    out = study_a.h3c_outcome(A, V.PROPOSED)
    for t in ("reset", "injection"):
        assert R.ALPHA / len(R.TASKS_STUDY_A) < out.numbers[f"dormant_trainable({t}) - untreated_p"] < R.ALPHA
        assert out.numbers[f"passes_uncorrected({t})"] is True and out.numbers[f"holm_survives({t})"] is False
        assert out.numbers[f"passes({t})"] is False
    assert out.status == V.INCONCLUSIVE  # not supported (Holm), not falsified (the check did not fail)
    # the task family is H3 (c)'s only one, so Q-holm-families (how two families combine) keeps it
    assert study_a.h3c_outcome(A, V.alternative("Q-holm-families")).status == V.INCONCLUSIVE
    other = study_a.h3c_outcome(A, V.PROPOSED, task=R.TASKS_STUDY_A[1])
    assert other.status == V.FALSIFIED and other.numbers["passes_uncorrected(reset)"] is False
    assert all("Q-holm-families" in questions.keys_for(k) for k in ("H3c", "H3c-cell", "H3"))


def test_h3c_leaves_out_a_task_that_rule_3_makes_infeasible() -> None:
    """Part 4.1 rule 3, an infeasible task 'enters no comparison'. H3 (c)
    on it is NOT_COMPUTABLE with the rule's reason, and it leaves the three-task Holm family (``_task_member``)."""
    car, button = R.TASKS_STUDY_A[1], R.TASKS_STUDY_A[2]
    specs = {s.run_id: s for s in syn.specs(tasks=R.TASKS_STUDY_A, groups=("main", "treatment"))}
    rows = syn.a_world("supported", tasks=R.TASKS_STUDY_A, groups=("main", "treatment"),
                       cost=lambda s: (40.0 if s.task != TASK and s.N == 0.0 else 25.0) + 0.5 * syn.noise(s.seed))
    records = []
    for row in rows:
        spec = specs[row["run_id"]]
        if not (spec.N == 0.5 and spec.onset_shape == "abrupt" and spec.step_matching == "total_steps"):
            continue
        e = syn.noise(spec.seed)
        treated = spec.treatment in ("reset", "injection")
        if spec.task == TASK and treated:
            dormant, rank = 0.5 - 0.014 + BORDERLINE[spec.seed], 30.0 + e  # dormant lower at p ~ 0.028
        elif treated and spec.task == car:
            dormant, rank = 0.3 + 0.01 * e, 30.0 + e  # a clear effect on the infeasible task
        else:
            dormant, rank = 0.5 + 0.01 * e, 20.0 + e
        records.append(("training", syn.training_record(row, spec=spec, dormant_check=dormant, rank_check=rank)))
    A = study_a.StudyA(data.build_dataset(rows, supplement(records), mode="final"))
    for task in (car, button):
        assert A.task_state(V.PROPOSED, task, "matched")[0] is False
        out = study_a.h3c_outcome(A, V.PROPOSED, task=task)
        assert out.status == V.NOT_COMPUTABLE and "rule 3" in out.reason
    out = study_a.h3c_outcome(A, V.PROPOSED)  # the family is PointGoal1 alone: 0.028 < 0.05 survives
    for t in ("reset", "injection"):
        assert R.ALPHA / len(R.TASKS_STUDY_A) < out.numbers[f"dormant_trainable({t}) - untreated_p"] < R.ALPHA
        assert out.numbers[f"holm_survives({t})"] is True
    assert out.status == V.SUPPORTED


# ---------------------------------------------------------------------------
# Part 5.7 on falsified boxes only; the mediation wording on a decided H3; one control missing;
# the per-seed, side-by-side and norm tables
# ---------------------------------------------------------------------------


def _bound_keys(numbers: dict, prefix: str = "") -> set[str]:
    return {k for k in numbers if k.startswith(prefix) and ("bound_95_upper" in k or k.endswith("bound_reading"))}


def test_part_5_7_bound_is_read_only_on_a_falsified_h1() -> None:
    """Part 5.7: "If H1 or G1 is falsified, the result is reported as an upper bound"; Part 1.2 keeps "inconclusive
    at this sample size" for an outcome that meets neither condition. An H1 its conditions support or leave
    inconclusive (the primary cell and every secondary cell) carries no bound and no Part 5.7 reading; one its
    conditions falsify carries it, whether the bound confirms the falsification or not (Q-falsification-calibration)."""
    for kind in ("supported", "inconclusive"):
        verdicts, _ = run(syn.a_world(kind))
        h1 = verdicts["H1"]
        assert h1.status in (V.SUPPORTED, V.INCONCLUSIVE) and not _bound_keys(h1.numbers)
        assert not any("Part 5.7" in n for n in h1.notes)
        for v in verdicts.values():
            if not v.id.startswith(("H1", "A-rule7a", "A-final-estimand")):
                continue
            for control in R.STEP_MATCHING_CONTROLS:  # a control's bound only beside that control's falsification
                has_bound = bool(_bound_keys(v.numbers, f"{control}."))
                assert has_bound == (v.numbers.get(f"{control}.falsification") is True), (v.id, control)
    verdicts, _ = run(syn.a_world("falsified"))
    for control in R.STEP_MATCHING_CONTROLS:  # a falsified H1 keeps its bound and its reading
        assert verdicts["H1"].numbers[f"{control}.bound_reading"] == V.BOUND_BELOW


@pytest.mark.usefixtures("keys_open")
def test_the_mediation_wording_needs_an_h3_that_is_decided_supported() -> None:
    """Part 1.2 (on H3, "What H3 can establish"): "The word mediation is used in the paper only if all three parts
    hold". An H3 supported under the proposals but UNDECIDED(key) because another reading disagrees has not been
    found to hold: no mediation note."""
    rows = syn.a_world("supported")
    for row in late(rows, treatment="additional_constrained"):
        # D(additional) = -0.4 on the point estimate with an interval including zero: "does not reduce" on the
        # interval (proposal), "reduces" on the point estimate (Q-h3-scope's other reading)
        row["gap_hazard"] = 2.0 + 12.0 * 0.5 - 0.4 + 3.0 * syn.noise(row["seed"])
    verdicts, _ = run(rows, supplement(training_records(rows)))
    h3 = verdicts["H3"]
    assert h3.proposal_status == V.SUPPORTED and h3.status == V.UNDECIDED and "Q-h3-scope" in h3.undecided_by
    assert not any("mediation" in n for n in h3.notes)
    assert study_a.MEDIATION_NOTE not in h3.notes
    assert "mediat" not in h3.title.lower()  # the title of every report, whatever the status


@pytest.mark.usefixtures("keys_open")
def test_one_control_not_computable_does_not_hide_an_inconclusive_box() -> None:
    """Q-controls-reading (the decided reading): "FALSIFIED only if FALSIFIED under both (after
    Q-falsification-calibration); INCONCLUSIVE if the two controls disagree or either is INCONCLUSIVE". With one
    control INCONCLUSIVE, H1 is INCONCLUSIVE whatever the missing control gives; under the "either" reading the
    missing control could be the falsified one, so that reading is NOT_COMPUTABLE and the key decides."""
    rows = syn.a_world("inconclusive", groups=("main",))
    late(rows, control="constrained_steps")[2]["gap_hazard"] = None  # Delta(0.50) incomplete under that control
    verdicts, _ = run(rows)
    h1 = verdicts["H1"]
    assert h1.numbers["constrained_steps.status"] == V.NOT_COMPUTABLE
    assert h1.numbers["total_steps.status"] == V.INCONCLUSIVE
    assert h1.proposal_status == V.INCONCLUSIVE and h1.reason == ""
    assert any(n.startswith("constrained_steps: the box cannot be computed") for n in h1.notes)
    assert h1.readings["Q-controls-reading"] == V.NOT_COMPUTABLE and "Q-controls-reading" in h1.undecided_by
    assert verdicts["H0"].numbers["H1"] == V.INCONCLUSIVE  # box H0 reads H1 as inconclusive, not as not computable


@pytest.mark.usefixtures("keys_open")
def test_matched_checkpoint_verdicts_are_not_provisional_on_the_final_cost_set() -> None:
    """Q-final-cost-set names the evaluation set of the final checkpoint: only the final-checkpoint estimand reads it."""
    verdicts, _ = run(syn.a_world("supported", groups=("main",)))
    for key in ("H1", "H2", "H3a", "H3b", "H4", "A-rule7a", "A-return[SafetyPointGoal1-v0/total_steps/N0.50]"):
        assert "Q-final-cost-set" not in verdicts[key].provisional_on, key
    assert "Q-final-cost-set" in verdicts["A-final-estimand"].provisional_on


def test_the_per_seed_gap_side_by_side_and_norm_tables() -> None:
    """Analysis spec A1 (A_gaps), M9 (A_estimands_side_by_side) and A21 (A_norms)."""
    rows = syn.a_world("supported", groups=("main",))
    records = []
    for row in where(rows, task=TASK):
        spec = SPECS[row["run_id"]]
        records.append(("final_battery", syn.battery_record(row, "hazard", kind="final_battery", step=spec.total_steps)))
        measure = syn.battery_record(row, "dynamics", kind="final_battery", step=spec.total_steps)
        measure.update(condition="measurement", measurement_cost=None, gap=None, episodes=syn.episodes(row["measurement_cost"]))
        records.append(("final_battery", measure))
    for row in late(rows, N=0.25):
        row["measurement_cost"] = 29.0  # unmatched at 2.5, matched at 5.0: the sensitivity battery's gap
        records.append(("sensitivity_battery", syn.battery_record(row, "hazard", kind="sensitivity_battery")))
        row["gap_hazard"] = row["gap_dynamics"] = None
    reference = where(rows, task=TASK, N=0.0)[0]
    reference["checkpoints"][-1]["norm"] = 6.5  # the matched (last) checkpoint's actor norm
    training = syn.training_record(reference, spec=SPECS[reference["run_id"]])
    training["plasticity_onset"] = {"step": 0, "norm": 4.0, "norm_reward_critic": 7.5, "norm_cost_critic": 2.5}
    records.append(("training", training))
    _, cat = run(rows, supplement(records))
    gaps = {r["run_id"]: r for r in cat.tables["A_gaps"]}
    assert len(gaps) == len(rows)
    row = late(rows)[0]
    assert gaps[row["run_id"]]["gap_hazard"] == row["gap_hazard"] and gaps[row["run_id"]]["compared"] is True
    assert gaps[row["run_id"]]["final_gap_hazard"] is not None and gaps[row["run_id"]]["gap_finetune"] is None
    only5 = gaps[late(rows, N=0.25)[0]["run_id"]]
    assert only5["compared"] is False and only5["compared_tolerance_5"] is True
    assert only5["gap_hazard"] is None and only5["tolerance5_gap_hazard"] is not None
    assert only5["status"] == matching.STATUS_UNMATCHED_ABOVE
    side = {(r["control"], r["N"]): r for r in cat.tables["A_estimands_side_by_side"]}
    assert set(side) == {(c, N) for c in R.STEP_MATCHING_CONTROLS for N in R.LATE_ONSET_FRACTIONS}
    top = side[("total_steps", 0.5)]
    h1 = {v.id: v for v in cat.verdicts}["H1"]
    assert top["matched_diff"] == h1.numbers["total_steps.Delta(0.50)"] and top["label"] == V.EXPLORATORY
    assert top["final_diff"] is not None and top["agree_matched_final"] is True
    assert side[("total_steps", 0.25)]["matched_diff"] is None and "rule 6" in side[("total_steps", 0.25)]["matched_note"]
    assert side[("total_steps", 0.25)]["tolerance5_diff"] is not None
    assert side[("total_steps", 0.25)]["agree_matched_tolerance5"] is None  # no matched estimate to agree with
    norms = {r["run_id"]: r for r in cat.tables["A_norms"]}
    ref = norms[reference["run_id"]]
    assert ref["norm_onset"] == 5.0 and ref["norm_matched_checkpoint"] == 6.5 and ref["norm_final_checkpoint"] == 6.5
    assert ref["onset.norm_reward_critic"] == 7.5 and ref["onset.norm_cost_critic"] == 2.5 and ref["onset.step"] == 0
    assert ref["check.norm"] is None and ref["label"] == V.EXPLORATORY
    assert norms[row["run_id"]]["onset.norm"] is None  # no training record: not computable, not invented


def _primary(rows: list[dict], N: float, control: str | None = None) -> list[dict]:
    return [r for r in rows if r["task"] == TASK and r["N"] == N and r["treatment"] is None
            and r["controller_variant"] is None and (control is None or r["step_matching"] == control)
            and r["onset_shape"] in (None, "abrupt")]


def test_a_study_a_arm_below_its_seed_target_is_data_still_to_come() -> None:
    """With seeds 2 to 4 of both N = 0.50 abrupt arms still training and N = 0 seed 4 crashed (its Part 5.6
    replacement not run), H1 must not be decided on 2 against 4 seeds. An arm with fewer completed runs than its
    seed target (Q-arm-complete) is incomplete data: H1 is NOT_COMPUTABLE, never decided on the seeds present."""
    rows = syn.a_world("supported", groups=("main",))
    full = {v.id: v for v in run(rows)[1].verdicts}
    assert full["H1"].proposal_status == V.SUPPORTED
    late_short = [dict(r) for r in rows if not (r in _primary(rows, 0.5) and r["seed"] >= 2)]
    ds = data.build_dataset(late_short, data.Supplement(), mode="final")
    assert ds.awaited_a == {"A-PointGoal1-N0.50-abrupt-constrained": (2, 5), "A-PointGoal1-N0.50-abrupt-total": (2, 5)}
    assert ds.awaited_a_lines()[0] == "A-PointGoal1-N0.50-abrupt-constrained: 2 of its 5 seeds completed (Q-arm-complete)"
    h1 = run(late_short)[0]["H1"]
    assert h1.proposal_status == V.NOT_COMPUTABLE
    assert "incomplete: A-PointGoal1-N0.50-abrupt-total has 2 of its 5 seeds completed (Q-arm-complete)" in h1.reason
    # the reference below its target: rules 2, 3 and 5 wait on it, so the task's matching is incomplete
    for r in late_short:
        if r in _primary(late_short, 0.0) and r["seed"] == 4:
            r.update(completed=False, failure_cause="crash")
    verdicts, _ = run(late_short)
    assert verdicts["H1"].proposal_status == V.NOT_COMPUTABLE
    assert "the reference A-PointGoal1-N0.00 has 4 of its 5 seeds completed" in verdicts["H1"].reason
    # with E surplus seeds a primary-comparison arm's target is 5 + E (Part 5.5); the other arms keep 5
    ds = data.build_dataset(rows, data.Supplement(), mode="final", surplus_extra=1)
    assert set(ds.awaited_a) == {r["arm_id"] for r in ds.study_a if r["task"] == TASK and r["N"] in (0.0, 0.5)}
    assert all(v == (5, 6) for v in ds.awaited_a.values())


def test_a_short_reference_leaves_every_matched_contrast_of_its_task_incomplete() -> None:
    """With PointButton1's N = 0 seed 4 crashed (its Part 5.6 replacement still to come), H2's E(N), H3 (b), H4 and
    the recovery cells must not be decided from the matching of the partial reference, nor give the three-task Holm
    family a p-value (or None, "not tested", under rule 6) on data still to come. Every matched contrast of the
    task is incomplete, so Holm on another task's cell names it as a member still to come."""
    button = R.TASKS_STUDY_A[2]
    car = R.TASKS_STUDY_A[1]
    rows = syn.a_world("supported", tasks=R.TASKS_STUDY_A, groups=("main",))
    for r in rows:
        if r["task"] == button and r["N"] == 0.0 and r["seed"] == 4:
            r.update(completed=False, failure_cause="crash")
    A = study_a.StudyA(data.build_dataset(rows, data.Supplement(), mode="final"))
    assert A.task_state(V.PROPOSED, button, "matched")[0] is None
    member = study_a._h2_p(A, V.PROPOSED, button, "hazard", "total_steps")
    assert isinstance(member, stats.Incomplete) and "A-PointButton1-N0.00 has 4 of its 5 seeds" in member.reason
    x = A.arms(button, N=0.25, shape="abrupt", control="total_steps")
    y = A.arms(button, N=0.25, shape="ramp", control="total_steps")
    est, why = A.contrast(V.PROPOSED, task=button, x=x, y=y, get=study_a._recovery_getter(x, y), analysis_id="t")
    assert est is None and why.startswith(f"incomplete: matching of {button} cannot be decided yet")
    study_a.h2_outcome(A, V.PROPOSED, task=car, condition="hazard")
    grid = A._grids[("H2|total_steps", V.PROPOSED.cost_field, V.PROPOSED.arithmetic, "matched")]
    assert set(grid["by_task"]["hazard"]) == {TASK, car}  # the family does not shrink to these: PointButton1 waits
    assert button in grid["incomplete"][(car, "hazard")] and "4 of its 5 seeds" in grid["incomplete"][(car, "hazard")]
    # the final-checkpoint estimand has no matching filter (rule 7 (b)): unaffected
    assert study_a._h1_p(A, V.PROPOSED, button, "hazard", "total_steps", "abrupt", "final") is not None


def test_a_study_a_arm_above_its_seed_target_refuses_the_surplus_extra() -> None:
    """An arm with more completed runs than 5 + E means E is not the data root's surplus (Part 5.5)."""
    rows = syn.a_world("supported", groups=("main",))
    rows += _primary(syn.a_world("supported", groups=("main",), seeds=(5,)), 0.0)
    with pytest.raises(data.DataError, match=r"A-PointGoal1-N0.00 \(6 completed.*target 5\).*surplus_extra"):
        data.build_dataset(rows, data.Supplement(), mode="final")


def test_h3b_is_decided_by_the_clauses_the_known_contrasts_settle() -> None:
    """H3 (b) reads the falsifying clauses the known treatment contrasts decide before it is NOT_COMPUTABLE for a
    missing one. With reset reducing the gap and injection not, the box is falsified whatever additional
    constrained training gives (here left out under rule 6)."""
    rows = syn.a_world("supported", groups=("main", "treatment"))
    untreated = {r["seed"]: r["gap_hazard"] for r in late(rows)}
    for r in late(rows, treatment="injection"):
        r["gap_hazard"] = untreated[r["seed"]] + 0.01 * r["seed"]  # D(injection) about 0: injection does not reduce
    for r in late(rows, treatment="additional_constrained"):
        r["measurement_cost"] = r["selection_cost_at_match"] = 35.0  # unmatched: rule 6, not compared
        r["gap_hazard"] = r["gap_dynamics"] = None
    h3b = run(rows)[0]["H3b"]
    assert h3b.proposal_status == V.FALSIFIED, h3b.reason
    assert h3b.numbers["F_reset_reduces_injection_does_not"] is True
    assert h3b.numbers["reduces(additional_constrained)"] is None and h3b.numbers["D(additional_constrained)"] is None
    assert any("decided without D(additional_constrained)" in n for n in h3b.notes)
    # what the missing contrast could change stays open: reset and injection both reduce, so support needs it
    rows = syn.a_world("supported", groups=("main", "treatment"))
    for r in late(rows, treatment="additional_constrained"):
        r["measurement_cost"] = r["selection_cost_at_match"] = 35.0
        r["gap_hazard"] = r["gap_dynamics"] = None
    h3b = run(rows)[0]["H3b"]
    assert h3b.proposal_status == V.NOT_COMPUTABLE
    assert h3b.reason.startswith("D(additional_constrained): ") and "rule 6" in h3b.reason


def test_h4_is_decided_by_the_contrast_that_is_known() -> None:
    """H4 reads the clause the known contrast decides before it is NOT_COMPUTABLE for a missing F or G, and keeps
    the known estimate in its numbers. With the untreated N = 0.50 arm left out under rule 6 (F missing) and G's
    interval including zero, the box is falsified whatever F would be."""
    rows = syn.a_world("supported")
    ref = {r["seed"]: r["gap_hazard"] for r in _primary(rows, 0.0)}
    for r in late(rows):
        r["measurement_cost"] = 35.0  # rule 6: unmatched above, so F is not compared
    for r in late(rows, controller_variant="warm_started"):
        r["gap_hazard"] = ref[r["seed"]] + 0.3 * syn.noise((r["seed"] + 2) % 5)  # G about 0
    A = study_a.StudyA(data.build_dataset(rows, data.Supplement(), mode="final"))
    h4 = study_a.h4_outcome(A, V.PROPOSED)
    assert h4.status == V.FALSIFIED, h4.reason
    assert h4.numbers["G_interval_includes_zero"] is True and h4.numbers["does_not_fall"] is None
    assert h4.numbers["F = gap(warm) - gap(untreated)"] is None
    assert h4.numbers["G = gap(warm) - gap(N=0)"] is not None
    assert [e["contrast"] for e in h4.estimates][:1] == ["G warm_started - N0"]
    assert any(n.startswith("decided without F: ") for n in h4.notes)
    # G above zero: support needs F, so the box stays open with F's reason
    rows = syn.a_world("supported")
    for r in late(rows):
        r["measurement_cost"] = 35.0
    h4 = study_a.h4_outcome(study_a.StudyA(data.build_dataset(rows, data.Supplement(), mode="final")), V.PROPOSED)
    assert h4.status == V.NOT_COMPUTABLE
    assert h4.reason.startswith("F: ") and "rule 6" in h4.reason
    assert h4.numbers["G = gap(warm) - gap(N=0)"] is not None


def test_an_arm_a_recorded_cut_removed_is_not_tested_in_the_claims() -> None:
    """After the recorded cut 'pid' the PID recovery claim is not tested (as a cut task), never NOT_COMPUTABLE ('an
    arm has no completed run ...'): no data are to come."""
    rows = syn.a_world("supported", tasks=(R.PRIMARY_TASK,), groups=("main", "treatment", "controller"))
    pid = f"A-recovery[{R.PRIMARY_TASK}/{R.PID_VARIANT}/treated-untreated/N0.50]"
    verdicts, _ = run(rows, cuts=("pid",))
    assert verdicts[pid].status == V.NOT_TESTED, verdicts[pid].reason
    assert verdicts[pid].reason.startswith(f"{study_a.CUT_TASK_REASON} pid (A-PointGoal1-N0.50-")
    # without the cut the PID arm is registered and its data are still to come: incomplete
    uncut, _ = run(rows)
    assert uncut[pid].status == V.NOT_COMPUTABLE and "no completed run yet" in uncut[pid].reason
    A = study_a.StudyA(data.build_dataset(rows, data.Supplement(), mode="final", cuts=("pid",)))
    # the cut PID arm names the cut
    assert A.absent_reason(R.PRIMARY_TASK, N=0.5, shape="abrupt", control="total_steps",
                           controller=R.PID_VARIANT).startswith(f"{study_a.CUT_TASK_REASON} pid")
    # an arm the design never had keeps the generic reason
    assert A.absent_reason(R.PRIMARY_TASK, N=0.5, shape="ramp", control="total_steps",
                           controller=R.PID_VARIANT) == study_a.NO_ARM_REASON


def test_ramp_surplus_seeds_name_q_surplus_arm_set() -> None:
    """H2 and A-recovery read the N = 0.50 ramp arms on the primary task, whose surplus seeds (seed target 5 + E) only
    Q-surplus-arm-set's answer adds; they carry the key, and so does the waiting line."""
    for key in ("H2", "H2-cell", "A-recovery", "H0"):
        assert "Q-surplus-arm-set" in questions.keys_for(key), key
    assert data.awaited_a_line("A-PointGoal1-N0.50-ramp-total", 5, 7) == (
        "A-PointGoal1-N0.50-ramp-total: 5 of its 7 seeds completed (Q-arm-complete, Q-surplus-arm-set)")
    assert data.awaited_a_line("A-PointGoal1-N0.50-abrupt-total", 5, 7).endswith("(Q-arm-complete)")
    assert study_a.Arm("A-PointGoal1-N0.50-ramp-total", [{}], short=(5, 7)).waiting == (
        "A-PointGoal1-N0.50-ramp-total has 5 of its 7 seeds completed (Q-arm-complete, Q-surplus-arm-set)")


@pytest.mark.parametrize("rank_relation, expected", [("flat", V.FALSIFIED), ("falls", V.INCONCLUSIVE)])
def test_h3a_reads_a_constant_onset_metric_as_no_monotone_association(rank_relation, expected) -> None:
    """A dormant fraction of 0 at onset in every late run has no monotone association with the
    gap (support fails, its falsification clause holds), so the rank's correlation still decides H3 (a), and its
    Holm member is p = 1, never a NaN that Holm drops; only a constant gap is NOT_COMPUTABLE."""
    rows = syn.a_world("supported", groups=("main",))
    for r in rows:
        if r["N"] > 0:
            r["dormant_onset"] = 0.0
            if rank_relation == "flat":
                r["rank_onset"] = 30.0 + (r["seed"] % 2)  # unrelated to the gap
    A = study_a.StudyA(data.build_dataset(rows, data.Supplement(), mode="final"))
    out = study_a.h3a_outcome(A, V.PROPOSED, holm=False)
    assert out.status == expected, out.reason
    assert any("dormant fraction at onset is constant" in n for n in out.notes)
    assert study_a._h3a_member(A, V.PROPOSED, TASK, "hazard", "dormant_onset") == 1.0
    p_rank = study_a._h3a_member(A, V.PROPOSED, TASK, "hazard", "rank_onset")
    assert isinstance(p_rank, float) and 0.0 <= p_rank <= 1.0
    flat_gap = [dict(r) for r in rows]
    for r in flat_gap:
        for c in study_a.CONDITIONS:
            if r.get(f"gap_{c}") is not None:
                r[f"gap_{c}"] = 1.0
    A2 = study_a.StudyA(data.build_dataset(flat_gap, data.Supplement(), mode="final"))
    out = study_a.h3a_outcome(A2, V.PROPOSED, holm=False)
    assert out.status == V.NOT_COMPUTABLE and "gap is constant" in out.reason
    assert study_a._h3a_member(A2, V.PROPOSED, TASK, "hazard", "rank_onset") is None


def test_return_claims_never_switch_one_cell_to_the_final_checkpoints_return() -> None:
    """One run of the N = 0.50 arm without its measurement record (not yet written) leaves that
    cell incomplete; its final checkpoint's return (8 higher in the N = 0.50 arms) is never read instead, and every
    other cell keeps the matched checkpoint's measurement-set return."""
    rows = syn.a_world("supported", groups=("main",))
    for r in rows:
        r["final_return"] = 10.0 + (8.0 if r["N"] == 0.5 else 0.0) + 0.1 * syn.noise(r["seed"])
    missing = "A-PointGoal1-N0.50-abrupt-total-s0"
    assert any(r["run_id"] == missing for r in rows)
    verdicts, _ = run(rows, supplement([("measurement", syn.measurement_record(r)) for r in rows
                                        if r["run_id"] != missing]))
    cell = verdicts["A-return[SafetyPointGoal1-v0/total_steps/N0.50]"]
    assert cell.proposal_status == V.NOT_COMPUTABLE and missing in cell.reason
    assert "no measurement-set return" in cell.reason and "diff" not in cell.numbers
    other = verdicts["A-return[SafetyPointGoal1-v0/total_steps/N0.25]"]
    assert other.proposal_status != V.NOT_COMPUTABLE and other.numbers["diff"] == pytest.approx(0.0, abs=1e-9)
    assert any("measurement-set return of the matched checkpoint" in n for n in other.notes)


# ---------------------------------------------------------------------------
# H1 without Delta(0.50); a non-finite H3 (a) value; the reasons of H0, H3 and H3 (c);
# the Holm families named in the secondary statements
# ---------------------------------------------------------------------------


def test_h1_box_conditions_falsify_by_the_point_order_without_delta_050() -> None:
    """Box H1's F2 ('their point estimates are not ordered with N') is decided by the known Deltas: Delta(0.10) = 8
    above Delta(0.25) = 0 is out of order whatever Delta(0.50) is, so the box conditions falsify it although the
    N = 0.50 abrupt arms are left out under rule 6. Without Delta(0.50) its Part 5.7 bound cannot be read, so the
    falsification is not confirmed: INCONCLUSIVE (Q-falsification-calibration)."""
    rows = syn.a_world("falsified", groups=("main",))
    for r in _abrupt(rows, 0.1):
        r["gap_hazard"] += 8.0
    for r in _abrupt(rows, 0.5):
        r["measurement_cost"] = r["selection_cost_at_match"] = 35.0  # rule 6 on both evaluation sets
        r["gap_hazard"] = r["gap_dynamics"] = None
    A = study_a.StudyA(data.build_dataset(rows, data.Supplement(), mode="final"))
    # the box conditions falsify it, but without Delta(0.50) there is no Part 5.7 bound: the falsification is not
    # confirmed (Q-falsification-calibration, answered); the box conditions alone give FALSIFIED
    assert study_a.h1_outcome(A, V.alternative("Q-falsification-calibration")).status == V.FALSIFIED
    out = study_a.h1_outcome(A, V.PROPOSED)
    assert out.status == V.INCONCLUSIVE, out.reason
    for control in R.STEP_MATCHING_CONTROLS:
        n = {k.removeprefix(f"{control}."): v for k, v in out.numbers.items() if k.startswith(f"{control}.")}
        assert n["status"] == V.INCONCLUSIVE and n["falsification"] is True
        assert n["F2_not_ordered"] is True and n["F1_all_intervals_include_zero"] is False
        assert n["S1_delta050_positive_excluding_zero"] is None and n["points_ordered"] is False and n["support"] is False
        assert n["not_compared_rule6"] == [0.5] and "Delta(0.50)" not in n and n["holm_survives"] is None
        assert n["bound_95_upper_Delta050"] is None and n["bound_reading"] == "no interval"
    # in order without Delta(0.50): nothing is decided, and the reason names Delta(0.50)
    rows = syn.a_world("falsified", groups=("main",))
    for r in _abrupt(rows, 0.5):
        r["gap_hazard"] = None  # matched, the battery not yet evaluated
    out = study_a.h1_outcome(study_a.StudyA(data.build_dataset(rows, data.Supplement(), mode="final")), V.PROPOSED)
    assert out.status == V.NOT_COMPUTABLE and "Delta(0.50): incomplete" in out.reason


def test_a_non_finite_h3a_value_is_missing_data_not_a_crash() -> None:
    """A NaN gap (a hand-edited ledger: the schema's gap fields are plain floats) must not make H3 (a) raise
    StatsError and the whole analysis crash; it is missing data, as in every other box."""
    rows = syn.a_world("supported", groups=("main",))
    hit = next(r for r in rows if r["N"] == 0.25 and r["onset_shape"] == "ramp" and r["seed"] == 2)
    hit["gap_hazard"] = float("nan")
    ds = data.build_dataset(rows, data.Supplement(), mode="final")
    study_a.analyse(ds, iqm=False)  # no exception
    A = study_a.StudyA(ds)
    out = study_a.h3a_outcome(A, V.PROPOSED)
    assert out.status == V.NOT_COMPUTABLE and hit["run_id"] in out.reason
    assert isinstance(study_a._h3a_member(A, V.PROPOSED, TASK, "hazard", "dormant_onset"), stats.Incomplete)


def test_the_not_computable_reasons_of_h0_h3_and_h3c() -> None:
    """Every NOT_COMPUTABLE verdict says why: H0 and the combined H3 name their not-computable parts, and H3 (c) blames
    the training records (and Q-controller-quantities while it is open) only when a metric is missing."""
    rows = syn.a_world("supported")
    verdicts, _ = run(rows)  # no training records: H3 (c) has no manipulation-check metric
    h3c, h3 = verdicts["H3c"], verdicts["H3"]
    assert h3c.status == V.NOT_COMPUTABLE and "steps come from the training records" in h3c.reason
    assert ("Q-controller-quantities" in h3c.reason) == R.is_open("Q-controller-quantities")
    assert h3.proposal_status == V.NOT_COMPUTABLE and h3.reason.startswith(f"H3c is {V.NOT_COMPUTABLE} (")
    short = [r for r in rows if not (r["task"] == TASK and r["treatment"] == "reset")]
    out = study_a.h3c_outcome(study_a.StudyA(data.build_dataset(short, data.Supplement(), mode="final")), V.PROPOSED)
    assert out.status == V.NOT_COMPUTABLE and out.reason.startswith("reset: ") and "training records" not in out.reason
    rows = syn.a_world("falsified")
    for row in rows:
        if row["N"] == 0.5 and row["onset_shape"] == "ramp" and row["treatment"] is None and row["controller_variant"] is None:
            row["gap_hazard"] = None
    h0 = run(rows)[0]["H0"]
    assert h0.status == V.NOT_COMPUTABLE
    assert h0.reason.startswith(f"H1 is {V.FALSIFIED}, H2 is {V.NOT_COMPUTABLE}; H2: ") and "E(0.50)" in h0.reason


def test_the_secondary_statements_name_the_holm_families_each_cell_has() -> None:
    """Q-holm-families (``StudyA.holm_survival``): every secondary cell, whatever it shares with the primary, is decided
    by both its families."""
    verdicts, _ = run(syn.a_world("supported", groups=("main",)))
    car = R.TASKS_STUDY_A[1]
    for vid in (f"H1[{TASK}/dynamics/abrupt]", f"H2[{car}/hazard]", f"H2[{car}/dynamics]", f"H1[{TASK}/hazard/ramp]",
                f"H3a[{TASK}/dynamics]", f"H4[{car}/hazard]"):
        assert study_a.HOLM_FAMILIES in verdicts[vid].statement and "in both its families" in verdicts[vid].statement


# ---------------------------------------------------------------------------
# Answered keys: Q-falsification-calibration, Q-surplus-in-analysis; the note Q-training-age-systematic
# ---------------------------------------------------------------------------


def test_a_box_falsified_h1_whose_bound_is_not_below_the_minimum_effect_is_inconclusive(monkeypatch) -> None:
    """Q-falsification-calibration (answered; Part 1.2: "a falsified one is one whose effect is bounded
    below the minimum of interest"; Part 5.7: a bound above it "is reported as inconclusive at this sample size").
    Every Delta's interval includes zero (F1), but Delta(0.50)'s 95 percent Welch upper limit is about 8, above the
    5 cost units of Part 1.5: H1 is INCONCLUSIVE, and so H0 (which needs H1 falsified) is not supported and H3 and H4
    are read. The box conditions alone (the other reading) give FALSIFIED and a supported H0."""
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset())  # every key open: the other reading is computed
    rows = syn.a_world("falsified")
    for row in where(rows, task=TASK, N=0.5, treatment=None, controller_variant=None, onset_shape="abrupt"):
        row["gap_hazard"] = 3.0 + 8.0 * syn.noise(row["seed"])  # Delta(0.50) = 0 with a wide interval
    verdicts, _ = run(rows)
    h1, h0 = verdicts["H1"], verdicts["H0"]
    assert h1.proposal_status == V.INCONCLUSIVE and h1.readings["Q-falsification-calibration"] == V.FALSIFIED
    for control in R.STEP_MATCHING_CONTROLS:
        assert h1.numbers[f"{control}.status"] == V.INCONCLUSIVE and h1.numbers[f"{control}.falsification"] is True
        assert h1.numbers[f"{control}.F1_all_intervals_include_zero"] is True
        assert h1.numbers[f"{control}.bound_95_upper_Delta050"] >= R.MIN_EFFECT_STUDY_A
        assert h1.numbers[f"{control}.bound_reading"] == V.BOUND_ABOVE
    assert any("not below the minimum effect" in n for n in h1.notes)
    assert h0.proposal_status == V.INCONCLUSIVE and h0.readings["Q-falsification-calibration"] == V.SUPPORTED
    assert verdicts["H4"].proposal_status != V.NOT_TESTED  # H0 is not supported: H4 is read
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset(R.PENDING))
    verdicts, _ = run(rows)
    assert verdicts["H1"].status == V.INCONCLUSIVE and verdicts["H0"].status == V.INCONCLUSIVE


def test_the_h1_trend_reads_the_registered_seeds_and_every_other_estimate_every_seed() -> None:
    """Q-surplus-in-analysis (answered): surplus seeds 5 to 11 of the primary-comparison arms (N = 0 and
    N = 0.50 on SafetyPointGoal1-v0) enter matching and every two-arm estimate (Delta(0.50) on twelve seeds per arm),
    but Part 5.4's trend keeps its "twenty points: four onset fractions by five seeds"."""
    from pilot.manifest import SURPLUS_NOTE

    rows = syn.a_world("supported", groups=("main",))
    surplus = [r for r in syn.a_world("supported", groups=("main",), seeds=range(5, 12))
               if r["task"] == TASK and r["N"] in (0.0, 0.5)]
    for row in surplus:
        row["notes"] = SURPLUS_NOTE  # pilot.manifest.seed_role: 'surplus'
    verdicts, cat = run(rows + surplus, surplus_extra=7)
    h1 = verdicts["H1"]
    assert h1.proposal_status == V.SUPPORTED
    for control in R.STEP_MATCHING_CONTROLS:
        assert h1.numbers[f"{control}.Delta(0.50)_n"] == (12, 12)
        assert h1.numbers[f"{control}.trend_points"] == R.H1_TREND_POINTS == 20
        assert h1.numbers[f"{control}.trend_seeds"] == 5
        assert h1.numbers[f"{control}.trend_seed_rule"].startswith("registered seeds and replacements")
    assert not any(f"registers {R.H1_TREND_POINTS}" in n for n in h1.notes)
    # matching reads every completed seed too: the reference cost is the mean over twelve seeds
    ref = next(r for r in cat.tables["A_matching"] if r["arm_id"] == "A-PointGoal1-N0.00")
    assert ref["n"] == 12


def test_systematically_different_training_ages_are_named_on_the_verdicts_that_compare_them() -> None:
    """Q-training-age-systematic (Part 4.1.1): every pair of arms a verdict compares on matched checkpoints has the
    95 percent Welch interval of its difference in mean training age (table A_training_age, contrast rows). Additional
    constrained training trains T + N·T steps, so its matched checkpoints are N·T = 5,000,000 steps older than the
    untreated N = 0.50 arm's (synthetic rows: training age = total steps): H3 (b) names the difference; reset and
    injection, abrupt against ramp (H2) and the warm-started multiplier (H4) do not differ; H3 (c), read at a fixed
    logging point, carries no training-age note."""
    rows = syn.a_world("supported")
    verdicts, cat = run(rows, supplement(training_records(rows)))
    contrasts = [r for r in cat.tables["A_training_age"] if r["contrast"] != "vs reference"]
    by_use = {use: r for r in contrasts for use in r["used_by"].split()}
    add = by_use[f"H3b|{TASK}|hazard|additional_constrained"]
    assert add["arm_id"] == "A-PointGoal1-N0.50-abrupt-total-additional_constrained"
    assert add["arm_y"] == "A-PointGoal1-N0.50-abrupt-total"
    assert add["diff"] == pytest.approx(0.5 * R.TOTAL_STEPS) and add["systematic"] is True
    assert by_use[f"H3b|{TASK}|hazard|reset"]["systematic"] is False
    h3b = verdicts["H3b"]
    assert h3b.proposal_status == V.SUPPORTED  # the note names a limit on the interpretation; it decides nothing
    assert any(n.startswith("training ages differ systematically between "
                            "A-PointGoal1-N0.50-abrupt-total-additional_constrained and A-PointGoal1-N0.50-abrupt-total")
               for n in h3b.notes)
    # H2: abrupt against ramp under each control, one row per arm pair, read by every condition's verdict
    h2_row = by_use[f"H2|{TASK}|hazard|total_steps|N0.50"]
    assert h2_row["arm_id"] == "A-PointGoal1-N0.50-abrupt-total" and h2_row["arm_y"] == "A-PointGoal1-N0.50-ramp-total"
    assert h2_row["systematic"] is False and f"H2|{TASK}|dynamics|total_steps|N0.50" in h2_row["used_by"]
    assert f"H2|{TASK}|hazard|constrained_steps|N0.10" in by_use
    assert not any("training ages differ" in n for n in verdicts["H2"].notes)
    # H4: the warm-started multiplier against the untreated arm and against N = 0; the other variants are tabled too
    assert by_use[f"H4|{TASK}|hazard|warm_started|F"]["systematic"] is False
    assert f"H4|{TASK}|hazard|rate_limited|G" in by_use
    assert not any("training ages differ" in n for n in verdicts["H4"].notes)
    # H1 under the constrained-steps control: T / (1 - N) against T, named; under total steps, equal ages
    h1_row = by_use[f"H1|{TASK}|hazard|constrained_steps|abrupt|matched|N0.50"]
    assert h1_row["diff"] == pytest.approx(R.TOTAL_STEPS) and h1_row["systematic"] is True
    assert by_use[f"H1|{TASK}|hazard|total_steps|abrupt|matched|N0.50"]["systematic"] is False
    assert not any(n.startswith("total_steps: training ages") for n in verdicts["H1"].notes)
    assert verdicts["H3c"].proposal_status == V.SUPPORTED
    assert not any("training ages" in n for n in verdicts["H3c"].notes)
    assert not any(use.startswith(("H3c|", "recovery|", "return|")) for use in by_use)


# ---------------------------------------------------------------------------
# A Part 5.7 bound still to come (Q-falsification-calibration)
# ---------------------------------------------------------------------------


def test_h1_waits_on_a_delta_050_still_to_come_for_the_bound_of_its_falsification() -> None:
    """A box H1 its conditions falsify is FALSIFIED only with its Part 5.7 bound (Delta(0.50)'s upper limit) below the
    minimum effect. With the points out of order (F2) and the N = 0.50 abrupt arms matched but their hazard battery
    not yet evaluated, that bound is still to come: H1 is NOT_COMPUTABLE, never a final INCONCLUSIVE, so H0 waits and
    H3 and H4 are not decided on partial data. Once Delta(0.50) arrives, H1 is FALSIFIED and H0 SUPPORTED. (Left out
    under rule 6, Delta(0.50) never comes, and the box is INCONCLUSIVE: the test of H1 without Delta(0.50) above.)"""
    rows = syn.a_world("falsified")
    for r in _abrupt(rows, 0.1):
        r["gap_hazard"] += 8.0  # Delta(0.10) = 8 > Delta(0.25) = 0: F2 holds under both controls
    complete = [dict(r) for r in rows]
    for r in _abrupt(rows, 0.5):
        r["gap_hazard"] = None  # matched, the hazard battery not yet evaluated
    verdicts, _ = run(rows)
    h1 = verdicts["H1"]
    assert h1.status == V.NOT_COMPUTABLE and "the Part 5.7 bound that decides the falsification waits" in h1.reason
    for control in R.STEP_MATCHING_CONTROLS:
        assert h1.numbers[f"{control}.status"] == V.NOT_COMPUTABLE and h1.numbers[f"{control}.falsification"] is True
        assert h1.numbers[f"{control}.incomplete"] == [0.5]
    assert study_a.could_be_falsified(V.Outcome(h1.status, numbers=h1.numbers), V.PROPOSED)
    assert verdicts["H0"].status == V.NOT_COMPUTABLE
    for key in ("H3b", "H4"):
        assert verdicts[key].status == V.NOT_COMPUTABLE and "H0 cannot be decided" in verdicts[key].reason
    verdicts, _ = run(complete)
    assert verdicts["H1"].status == V.FALSIFIED and verdicts["H0"].status == V.SUPPORTED
    assert verdicts["H4"].status == V.NOT_TESTED


def _h4_no_fall(f: str) -> V.Outcome:
    """H4 (no Holm) with G's interval including zero but its upper limit above 5 (a wide N = 0 arm), and no fall (F
    about 0 with a tight interval, its bound below the minimum effect); F complete, still to come ('incomplete': one
    seed's battery not yet evaluated) or left out under rule 6 ('rule6')."""
    rows = syn.a_world("supported")
    for r in where(rows, task=TASK, N=0.0):
        r["gap_hazard"] = 2.0 + 8.0 * syn.noise(r["seed"])
    for r in late(rows, controller_variant="warm_started"):
        r["gap_hazard"] = 2.0 + 0.1 * syn.noise(r["seed"])
    for r in late(rows):
        r["gap_hazard"] = 2.0 + 0.1 * syn.noise((r["seed"] + 2) % 5)
        if f == "incomplete" and r["seed"] == 0:
            r["gap_hazard"] = None
        elif f == "rule6":
            r["measurement_cost"] = 35.0
    return study_a.h4_outcome(study_a.StudyA(data.build_dataset(rows, data.Supplement(), mode="final")), V.PROPOSED,
                              holm=False)


def _h3b_reset_not_injection(add: str) -> V.Outcome:
    """H3 (b) (no Holm): reset reduces the gap and injection does not, with injection's bound above the minimum effect;
    additional constrained training reduces it as much as reset, with that bound below; D(additional) complete, still
    to come ('incomplete': one seed's battery not yet evaluated) or left out under rule 6 ('rule6')."""
    rows = syn.a_world("supported")
    for t, base in ((None, 8.0), ("reset", 2.0), ("additional_constrained", 1.5)):
        for r in late(rows, treatment=t):
            r["gap_hazard"] = base + 0.3 * syn.noise(r["seed"])
    for r in late(rows, treatment="injection"):
        r["gap_hazard"] = 6.0 + 6.0 * syn.noise((r["seed"] + 1) % 5)
    for r in late(rows, treatment="additional_constrained"):
        if add == "incomplete" and r["seed"] == 0:
            r["gap_hazard"] = None
        elif add == "rule6":
            r["measurement_cost"] = r["selection_cost_at_match"] = 35.0
            r["gap_hazard"] = r["gap_dynamics"] = None
    return study_a.h3b_outcome(study_a.StudyA(data.build_dataset(rows, data.Supplement(), mode="final")), V.PROPOSED,
                               holm=False)


def test_h4_and_h3b_wait_on_a_falsifying_clause_whose_bound_is_still_to_come() -> None:
    """A falsified box stands only with a Part 5.7 bound below the minimum effect. When the falsifying clause the known
    estimates decide has its bound above it, another falsifying clause whose contrast is still to come could confirm
    the falsification once it arrives: the box is NOT_COMPUTABLE until then, and INCONCLUSIVE only when that contrast
    is left out under rule 6 (it never comes)."""
    waiting = _h4_no_fall("incomplete")
    assert waiting.numbers["G_interval_includes_zero"] is True and waiting.numbers["does_not_fall"] is None
    assert waiting.numbers["bound(G, 95% Welch)_reading"] == V.BOUND_ABOVE
    assert waiting.status == V.NOT_COMPUTABLE and waiting.reason.startswith("F: incomplete")
    assert not any(n.startswith("decided without") for n in waiting.notes)
    complete = _h4_no_fall("complete")
    assert complete.status == V.FALSIFIED and complete.numbers["bound(F, 95% Welch)_reading"] == V.BOUND_BELOW
    never = _h4_no_fall("rule6")
    assert never.status == V.INCONCLUSIVE and never.numbers["does_not_fall"] is None
    assert any(n.startswith("decided without F: ") for n in never.notes)
    waiting = _h3b_reset_not_injection("incomplete")
    n = waiting.numbers
    assert n["F_reset_reduces_injection_does_not"] is True and n["F_additional_reduces_as_much_as_reset"] is None
    assert n["bound(D(injection), 95% Welch)_reading"] == V.BOUND_ABOVE
    assert waiting.status == V.NOT_COMPUTABLE and waiting.reason.startswith("D(additional_constrained): incomplete")
    complete = _h3b_reset_not_injection("complete")
    assert complete.status == V.FALSIFIED
    assert complete.numbers["bound(gap(additional_constrained) - gap(reset), 95% Welch)_reading"] == V.BOUND_BELOW
    assert _h3b_reset_not_injection("rule6").status == V.INCONCLUSIVE

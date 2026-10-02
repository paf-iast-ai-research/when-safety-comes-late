"""Open questions behind each verdict (analysis/questions.py) and the UNDECIDED mechanism (analysis/verdict.py)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

import pytest

from analysis import questions as Q
from analysis import verdict as V
from configs import registered as R

ANALYSIS = Path(__file__).resolve().parents[1] / "analysis"


@pytest.fixture
def keys_open(monkeypatch):
    """Every PENDING key open (configs.registered.ANSWERED_QUESTIONS empty): the test reads the other readings that
    analysis.verdict computes only while a key is open, whatever the amendment log has answered since."""
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset())


def test_every_key_of_the_table_is_a_pending_key() -> None:
    assert Q.all_table_keys() <= set(R.PENDING)
    assert set(V.READINGS) <= set(R.PENDING)
    for key in V.READINGS:  # every computable reading is used by some verdict
        assert any(key in keys for keys in Q.REPORT_KEYS.values())


def test_key_literals_in_the_analysis_code_are_pending_keys_where_they_reach_is_open() -> None:
    # the table and the readings are the only places whose keys reach R.is_open
    for name in ("questions.py", "verdict.py"):
        text = (ANALYSIS / name).read_text(encoding="utf-8")
        code = re.sub(r'"""[\s\S]*?"""', "", text)  # docstrings may name keys that are not in PENDING
        code = "\n".join(line.split("#")[0] for line in code.splitlines())
        for key in re.findall(r'"(Q-[a-z0-9-]+)"', code):
            assert key in R.PENDING, f"{name}: {key}"


@pytest.mark.usefixtures("keys_open")
def test_provisional_on_lists_only_open_keys(monkeypatch) -> None:
    keys = Q.keys_for("H1", conditions=("hazard",))
    assert "Q-hazard" in keys and "Q-interval" in keys and "Q-h1-shape" in keys
    assert Q.provisional_on("H1", conditions=("hazard",)) == list(keys)  # nothing answered yet
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset({"Q-interval", "Q-hazard"}))
    left = Q.provisional_on("H1", conditions=("hazard",))
    assert "Q-interval" not in left and "Q-hazard" not in left and "Q-h1-shape" in left
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset(R.PENDING))
    assert Q.provisional_on("H1", conditions=("hazard",)) == [] and Q.open_keys() == []
    with pytest.raises(KeyError):
        Q.keys_for("H9")


def test_conditions_add_their_result_gates() -> None:
    assert "Q-dynamics" in Q.keys_for("H1-cell", conditions=("dynamics",))
    assert {"Q-continuations", "Q-transfer-obs"} <= set(Q.keys_for("H2-cell", conditions=("transfer",)))
    assert "Q-hazard" not in Q.keys_for("H2-cell", conditions=("dynamics",))


def test_h3_and_h4_carry_the_keys_of_h0() -> None:
    for key in ("H3", "H3a", "H3b", "H3c", "H4", "H3a-cell", "H3b-cell", "H3c-cell", "H4-cell"):
        assert set(Q.REPORT_KEYS["H0"]) <= set(Q.REPORT_KEYS[key])


def test_gated_cells_carry_the_condition_key_of_h0() -> None:
    """Box H0 reads H1 and H2 on the hazard gap (Q-h0-scope), so the verdicts it gates carry
    Q-hazard whatever condition they read themselves (H3c on another task reads none)."""
    from analysis import study_a

    assert Q.H0_CONDITION == study_a.PRIMARY_CONDITION
    for key, conditions in (("H3a-cell", ("dynamics",)), ("H3b-cell", ("finetune",)), ("H4-cell", ("transfer",)),
                            ("H3c-cell", ()), ("H0", ())):
        keys = Q.keys_for(key, conditions=conditions)
        assert "Q-hazard" in keys and "Q-h0-scope" in keys, key
    assert "Q-hazard" not in Q.keys_for("H2-cell", conditions=("dynamics",))  # H2 is not gated by H0


def _compute(statuses: dict[str, str]) -> Callable[[V.Reading], V.Outcome]:
    """A verdict whose status under the proposal is statuses['base'] and under another reading statuses[key]."""
    def compute(reading: V.Reading) -> V.Outcome:
        for key, alternatives in V.READINGS.items():
            for overrides in alternatives:
                if all(getattr(reading, f) == v for f, v in overrides.items()) and reading != V.PROPOSED:
                    return V.Outcome(statuses.get(key, statuses["base"]))
        return V.Outcome(statuses["base"])
    return compute


@pytest.mark.usefixtures("keys_open")
def test_a_verdict_is_undecided_while_an_open_key_changes_it(monkeypatch) -> None:
    v = V.decide(id="H1", title="t", statement="s", label=V.CONFIRMATORY, table_key="H1",
                 compute=_compute({"base": V.SUPPORTED, "Q-interval": V.INCONCLUSIVE}), conditions=("hazard",))
    assert v.status == V.UNDECIDED and v.undecided_by == ("Q-interval",) and v.proposal_status == V.SUPPORTED
    assert v.display_status == "UNDECIDED(Q-interval)" and "Q-interval" in v.provisional_on
    assert v.readings["Q-interval"] == V.INCONCLUSIVE and v.readings["Q-h1-shape"] == V.SUPPORTED
    assert V.status_for_reading(v, "Q-interval") == V.INCONCLUSIVE
    assert V.status_for_reading(v, "Q-g1-criteria") == V.SUPPORTED  # not a key of H1: the proposal
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset({"Q-interval"}))
    v = V.decide(id="H1", title="t", statement="s", label=V.CONFIRMATORY, table_key="H1",
                 compute=_compute({"base": V.SUPPORTED, "Q-interval": V.INCONCLUSIVE}), conditions=("hazard",))
    assert v.status == V.SUPPORTED and "Q-interval" not in v.readings and "Q-interval" not in v.provisional_on


@pytest.mark.usefixtures("keys_open")
def test_a_disagreement_with_not_computable_is_undecided_in_both_directions() -> None:
    # analysis.verdict: a verdict whose readings disagree across an open key is UNDECIDED(key), whichever side is
    # NOT_COMPUTABLE; the proposals' status and reason are kept
    def compute(reading: V.Reading) -> V.Outcome:
        if reading == V.PROPOSED:
            return V.Outcome(V.NOT_COMPUTABLE, "an arm is unmatched")
        return V.Outcome(V.SUPPORTED if reading.floor_arms == "comparison" else V.NOT_COMPUTABLE, "an arm is unmatched")
    v = V.decide(id="G1", title="t", statement="s", label=V.CONFIRMATORY, table_key="G1", compute=compute)
    assert v.status == V.UNDECIDED and v.undecided_by == ("Q-floor-rule",) and v.display_status == "UNDECIDED(Q-floor-rule)"
    assert v.proposal_status == V.NOT_COMPUTABLE and v.reason == "an arm is unmatched"
    assert v.readings["Q-floor-rule"] == V.SUPPORTED and any("an arm is unmatched" in n for n in v.notes)
    reverse = V.decide(id="G1", title="t", statement="s", label=V.CONFIRMATORY, table_key="G1",
                       compute=_compute({"base": V.SUPPORTED, "Q-floor-rule": V.NOT_COMPUTABLE}))
    assert reverse.status == V.UNDECIDED and reverse.undecided_by == ("Q-floor-rule",)
    agreeing = V.decide(id="G1", title="t", statement="s", label=V.CONFIRMATORY, table_key="G1",
                        compute=lambda r: V.Outcome(V.NOT_COMPUTABLE, "no data"))
    assert agreeing.status == V.NOT_COMPUTABLE and agreeing.reason == "no data"


@pytest.mark.usefixtures("keys_open")
def test_a_key_with_several_other_readings_is_undecided_if_any_disagrees() -> None:
    # Q-g4-reading's other readings: 'within one horizon' inclusive, the ordering read per budget on the majority
    # (the decided reading: the arm statistic), and a censored value at the largest horizon (Part 4.2)
    assert len(V.alternatives("Q-g4-reading")) == 3
    assert V.alternative("Q-g4-reading", 2).g4_censored == "at"
    assert (V.PROPOSED.g4_within, V.PROPOSED.g4_order, V.PROPOSED.g4_censored) == ("less_than_one", "arm_median", "after")
    assert [V.alternative("Q-g4-reading", i).g4_order for i in range(3)] == ["arm_median", "per_budget", "arm_median"]
    v = V.decide(id="G4", title="t", statement="s", label=V.SECONDARY, table_key="G4",
                 compute=lambda r: V.Outcome(V.SUPPORTED if r.g4_order == "arm_median" else V.INCONCLUSIVE))
    assert v.proposal_status == V.SUPPORTED
    assert v.status == V.UNDECIDED and v.undecided_by == ("Q-g4-reading",) and v.readings["Q-g4-reading"] == V.INCONCLUSIVE
    assert any("g4_within=at_most_one gives SUPPORTED" in n and "g4_order=per_budget gives INCONCLUSIVE" in n
               and "g4_censored=at gives SUPPORTED" in n for n in v.notes)


def test_the_other_readings_answer_the_open_question_and_keep_the_registered_rules() -> None:
    """Q-holm-families asks how two families combine (never 'no Holm'); Q-floor-rule asks which arms
    count and which rate (never 'no floor rule': Part 4.2 registers the rule)."""
    assert [r.holm for r in V.alternatives("Q-holm-families")] == ["either"]
    floors = [(r.floor_arms, r.floor_rate) for r in V.alternatives("Q-floor-rule")]
    assert floors == [("comparison", "arm_mean"), ("all", "per_seed"), ("comparison", "per_seed")]
    assert (V.PROPOSED.holm, V.PROPOSED.floor_arms, V.PROPOSED.floor_rate) == ("both_families", "all", "arm_mean")
    for key, alternatives in V.READINGS.items():
        for overrides in alternatives:
            assert "none" not in overrides.values(), (key, overrides)


@pytest.mark.usefixtures("keys_open")
def test_extra_readings_must_be_keys_of_the_verdict() -> None:
    v = V.decide(id="H0", title="t", statement="s", label=V.SECONDARY, table_key="H0",
                 compute=lambda r: V.Outcome(V.INCONCLUSIVE), extra_readings={"Q-h0-scope": V.SUPPORTED})
    assert v.status == V.UNDECIDED and v.undecided_by == ("Q-h0-scope",)
    with pytest.raises(ValueError, match="does not depend"):
        V.decide(id="H0", title="t", statement="s", label=V.SECONDARY, table_key="H0",
                 compute=lambda r: V.Outcome(V.INCONCLUSIVE), extra_readings={"Q-g4-reading": V.SUPPORTED})
    with pytest.raises(ValueError):
        V.decide(id="X", title="t", statement="s", label=V.SECONDARY, table_key="H0",
                 compute=lambda r: V.Outcome("MAYBE"))


def test_box_and_control_combination_rules() -> None:
    notes: list[str] = []
    assert V.box_status(True, True, V.PROPOSED, notes) == V.FALSIFIED and notes  # Q-support-falsify-overlap
    assert V.box_status(True, True, V.alternative("Q-support-falsify-overlap"), []) == V.SUPPORTED
    assert V.box_status(False, False, V.PROPOSED, []) == V.INCONCLUSIVE
    both, either = V.PROPOSED, V.alternative("Q-controls-reading")
    assert V.combine_controls({"a": V.SUPPORTED, "b": V.SUPPORTED}, both) == V.SUPPORTED
    assert V.combine_controls({"a": V.SUPPORTED, "b": V.FALSIFIED}, both) == V.INCONCLUSIVE
    assert V.combine_controls({"a": V.SUPPORTED, "b": V.FALSIFIED}, either) == V.FALSIFIED
    assert V.combine_controls({"a": V.FALSIFIED, "b": V.FALSIFIED}, both) == V.FALSIFIED
    assert V.combine_controls({"a": V.FALSIFIED, "b": V.NOT_COMPUTABLE}, both) == V.NOT_COMPUTABLE
    assert V.combine_controls({"a": V.SUPPORTED, "b": V.NOT_COMPUTABLE}, both) == V.NOT_COMPUTABLE
    # a known INCONCLUSIVE control (or one supported and one falsified) already rules out both
    # "supported under both" and "falsified under both", so the missing control cannot change the box
    assert V.combine_controls({"a": V.NOT_COMPUTABLE, "b": V.INCONCLUSIVE}, both) == V.INCONCLUSIVE
    assert V.combine_controls({"a": V.SUPPORTED, "b": V.FALSIFIED, "c": V.NOT_COMPUTABLE}, both) == V.INCONCLUSIVE
    # ... but under "either" a missing control could still be the falsified one
    assert V.combine_controls({"a": V.NOT_COMPUTABLE, "b": V.INCONCLUSIVE}, either) == V.NOT_COMPUTABLE
    assert V.combine_controls({"a": V.NOT_COMPUTABLE, "b": V.FALSIFIED}, either) == V.FALSIFIED
    assert V.combine_controls({"a": V.NOT_COMPUTABLE, "b": V.NOT_COMPUTABLE}, both) == V.NOT_COMPUTABLE
    # every combination: the box is decided only when each status the missing control could take gives it
    known = (V.SUPPORTED, V.FALSIFIED, V.INCONCLUSIVE)
    for reading in (both, either):
        for a_status in (*known, V.NOT_COMPUTABLE):
            for b_status in (*known, V.NOT_COMPUTABLE):
                got = V.combine_controls({"a": a_status, "b": b_status}, reading)
                fills = {V.combine_controls({"a": x, "b": y}, reading)
                         for x in ((a_status,) if a_status != V.NOT_COMPUTABLE else known)
                         for y in ((b_status,) if b_status != V.NOT_COMPUTABLE else known)}
                assert got == (fills.pop() if len(fills) == 1 else V.NOT_COMPUTABLE), (reading.controls, a_status, b_status)
    assert V.claim_status(True) == V.SUPPORTED and V.claim_status(True, False) == V.INCONCLUSIVE
    assert V.claim_status(None) == V.NOT_COMPUTABLE and V.claim_status(False) == V.INCONCLUSIVE
    assert V.PROPOSED.matching_arithmetic.decimals == 4
    assert V.alternative("Q-threshold-arithmetic").matching_arithmetic.decimals is None


@pytest.mark.usefixtures("keys_open")
def test_not_tested_turns_undecided_when_h0_turns_on_an_open_key() -> None:
    # H3 is not tested because H0 is supported under the proposals; under the ramp reading of H1 (a key of
    # H0, hence of H3) H0 is not supported and H3 would be read
    v = V.decide(id="H3", title="t", statement="s", label=V.SECONDARY, table_key="H3",
                 compute=_compute({"base": V.NOT_TESTED, "Q-h1-shape": V.SUPPORTED}))
    assert v.status == V.UNDECIDED and v.undecided_by == ("Q-h1-shape",) and v.proposal_status == V.NOT_TESTED


def test_a_box_condition_the_data_cannot_evaluate() -> None:
    overlap = V.alternative("Q-support-falsify-overlap")
    assert V.box_status(None, False, V.PROPOSED, []) == V.NOT_COMPUTABLE  # support needs a missing estimate
    assert V.box_status(False, None, V.PROPOSED, []) == V.NOT_COMPUTABLE
    assert V.box_status(False, True, V.PROPOSED, []) == V.FALSIFIED  # decided whatever is missing
    assert V.box_status(None, True, V.PROPOSED, []) == V.FALSIFIED  # overlap reads FALSIFIED anyway
    assert V.box_status(None, True, overlap, []) == V.NOT_COMPUTABLE  # ... but SUPPORTED would win if support held
    assert V.box_status(True, None, V.PROPOSED, []) == V.NOT_COMPUTABLE
    assert V.box_status(True, None, overlap, []) == V.SUPPORTED


def test_the_part_5_7_bound_and_the_calibration_reading() -> None:
    numbers, below = V.bound_annotation("E", 3.0, 5.0, claimed="positive")
    assert below is True and numbers["bound(E)_reading"] == V.BOUND_BELOW and numbers["bound(E)_limit"] == "upper"
    assert V.bound_annotation("E", 5.0, 5.0, claimed="positive")[1] is False  # at the minimum: not below it
    assert V.bound_annotation("D", -4.0, 5.0, claimed="negative")[1] is True  # a reduction bounded above -5
    assert V.bound_annotation("D", None, 5.0, claimed="negative")[1] is None
    notes: list[str] = []
    assert V.calibrate(V.FALSIFIED, True, V.PROPOSED, notes) == V.FALSIFIED and notes
    # Q-falsification-calibration (answered): a falsified box stands only with its bound below the
    # minimum effect; a bound above it, or none (no interval), leaves it INCONCLUSIVE with a note
    notes = []
    assert V.calibrate(V.FALSIFIED, False, V.PROPOSED, notes) == V.INCONCLUSIVE
    assert any("not below the minimum effect" in n and "inconclusive at this sample size" in n for n in notes)
    assert V.calibrate(V.FALSIFIED, None, V.PROPOSED, []) == V.INCONCLUSIVE
    assert V.calibrate(V.SUPPORTED, False, V.PROPOSED, []) == V.SUPPORTED
    box = V.alternative("Q-falsification-calibration")  # the other reading: the box conditions decide
    assert box.calibration == "box" and V.PROPOSED.calibration == "bound"
    assert V.calibrate(V.FALSIFIED, False, box, []) == V.FALSIFIED


def test_every_entry_of_the_table_is_used_and_h2_does_not_turn_on_the_scope_of_h0() -> None:
    """An entry is read by a verdict (its table_key), a table (TABLE_ENTRIES) or the pilot rows
    (PILOT_ENTRIES); Q-h0-scope asks on which outcome H0 reads H1 and H2, so it changes H0 (and the boxes H0 gates),
    never H2."""
    from analysis import data, study_a, study_b

    empty = data.build_dataset([], data.Supplement(), mode="final")  # every verdict is built, NOT_COMPUTABLE
    verdicts = study_a.analyse(empty, iqm=False).verdicts + study_b.analyse(empty, iqm=False).verdicts
    used = {v.table_key for v in verdicts} | set(Q.TABLE_ENTRIES.values()) | set(Q.PILOT_ENTRIES.values())
    assert used == set(Q.REPORT_KEYS)
    assert "Q-h0-scope" not in Q.keys_for("H2") and "Q-h0-scope" not in Q.keys_for("H2-cell")
    assert all("Q-h0-scope" in Q.keys_for(k) for k in ("H0", "H3", "H4"))


@pytest.mark.usefixtures("keys_open")
def test_a_key_with_independent_sub_readings_tries_each_alone() -> None:
    """Q-h3-scope: 'across late arms' (H3 (a)) and 'does not reduce' (H3 (b)) are read apart as well as together,
    so box H3, which reads both, cannot hide one sub-reading's disagreement behind a joint reading that agrees;
    likewise Q-support-falsify-overlap's overlap rule and H1's ordering clause."""
    alternatives = [tuple(sorted(o)) for o in V.READINGS["Q-h3-scope"]]
    assert ("h3_arms",) in alternatives and ("h3_reduce",) in alternatives

    def h3(reading: V.Reading) -> V.Outcome:
        a = V.SUPPORTED if reading.h3_arms == "main" else V.INCONCLUSIVE
        b = V.INCONCLUSIVE if reading.h3_reduce == "interval" else V.SUPPORTED
        return V.Outcome(V.SUPPORTED if (a, b) == (V.SUPPORTED, V.SUPPORTED) else V.INCONCLUSIVE)

    v = V.decide(id="H3", title="t", statement="s", label=V.SECONDARY, table_key="H3", compute=h3,
                 conditions=("hazard",))
    assert v.proposal_status == V.INCONCLUSIVE and h3(V.alternatives("Q-h3-scope")[-1]).status == V.INCONCLUSIVE
    assert v.status == V.UNDECIDED and v.undecided_by == ("Q-h3-scope",) and v.readings["Q-h3-scope"] == V.SUPPORTED
    # Q-support-falsify-overlap: the overlap rule (box_status) and H1's ordering clause are independent too
    overlap = [tuple(sorted(o)) for o in V.READINGS["Q-support-falsify-overlap"]]
    assert overlap == [("overlap",), ("ordering",), ("ordering", "overlap")]

    def h1(reading: V.Reading) -> V.Outcome:
        a = V.SUPPORTED if reading.overlap == "falsified" else V.INCONCLUSIVE
        b = V.INCONCLUSIVE if reading.ordering == "trend_and_points" else V.SUPPORTED
        return V.Outcome(V.SUPPORTED if (a, b) == (V.SUPPORTED, V.SUPPORTED) else V.INCONCLUSIVE)

    v = V.decide(id="H1", title="t", statement="s", label=V.CONFIRMATORY, table_key="H1", compute=h1,
                 conditions=("hazard",))
    assert h1(V.alternatives("Q-support-falsify-overlap")[-1]).status == V.INCONCLUSIVE  # the joint change agrees
    assert v.status == V.UNDECIDED and v.readings["Q-support-falsify-overlap"] == V.SUPPORTED
    assert any("ordering=trend gives SUPPORTED" in n for n in v.notes)


def test_only_the_final_checkpoint_estimand_depends_on_the_final_cost_set() -> None:
    """Q-final-cost-set names the evaluation set at the final checkpoint (Appendix B final_cost), which
    only rule 7 (b) / Part 4.1.1's final-checkpoint estimand reads; the matched-checkpoint verdicts do not."""
    assert "Q-final-cost-set" not in Q.MATCHING
    for entry in ("matching", "H1", "H1-cell", "H2", "H3a", "H3b", "H3c", "H3", "H4", "H0", "A-recovery", "A-return",
                  "A-rule7a", "pilot-G1"):
        assert "Q-final-cost-set" not in Q.keys_for(entry), entry
    for entry in ("A-final-estimand", "A-iqm", "A-gaps"):  # A_iqm and A_gaps hold final-checkpoint rows too
        assert "Q-final-cost-set" in Q.keys_for(entry), entry
    assert Q.TABLE_ENTRIES["A_estimands_side_by_side"] == "A-final-estimand"


def test_each_study_a_family_carries_the_run_gate_keys_of_the_arms_it_reads() -> None:
    """The table lists 'the design keys that define what the compared arms are'. For each Study A verdict
    family, every run-gate key (RunSpec.pending) of the arms it compares is in its entry, leaving aside the keys every
    late-onset run carries (Q-cost-critic, Q-jc-window, Q-plasticity-definitions): the ramp cells of H1 read the ramp
    arms (Q-ramp-step), A-return the constrained-steps arms (Q-rounding, Q-selection-window), A-recovery the ramp,
    constrained-steps, treated and controller arms."""
    from analysis import study_a
    from pilot import manifest

    specs = manifest.design("study_a")
    everywhere = {"Q-cost-critic", "Q-jc-window", "Q-plasticity-definitions"}
    main = [s for s in specs if s.group == "main"]
    late = study_a.LATE_N
    treated = [s for s in specs if s.N == late and s.onset_shape == study_a.TREATMENT_SHAPE
               and s.step_matching == study_a.TREATMENT_CONTROL]
    families = {
        "H1": [s for s in main if s.onset_shape in (None, "abrupt")],
        "H1-cell": main,  # the ramp-shape cells H1[task/condition/ramp] read the ramp arms
        "H2": main,
        "A-return": [s for s in main if s.onset_shape in (None, "abrupt")],
        "A-recovery": [s for s in main if s.N in R.LATE_ONSET_FRACTIONS] + treated,
        "H3b": [s for s in treated if s.controller_variant is None],
        "H4": [s for s in treated if s.treatment is None and s.controller_variant in (None, "warm_started")],
    }
    for entry, arms in families.items():
        assert arms, entry
        needed = {k for s in arms for k in s.pending} - everywhere
        missing = needed - set(Q.keys_for(entry))
        assert not missing, f"{entry}: {sorted(missing)}"


def test_each_study_b_family_carries_the_run_gate_keys_of_the_arms_it_reads() -> None:
    """The Study B counterpart of the Study A test above. The floor rule's answer (Q-floor-rule: 'all seven Study B
    arms') counts the Continuous arm in G1, G2, G5, B-per-budget and B-fewshot, so their entries carry its
    run gate (Q-continuous-bins); G4 reads the Sparse, Moderate and Dense continuations and, through its few-shot floor,
    every arm's. Left aside: Q-jc-window (every run) and Q-studyb-order (scheduling)."""
    from pilot import manifest

    everywhere = {"Q-jc-window", "Q-studyb-order"}
    zero_shot = manifest.design("study_b")
    fewshot = manifest.design("study_b_fewshot")
    families = {
        **{entry: zero_shot for entry in ("G1", "G2", "G3", "G5", "B-per-budget", "B-violation", "B-iqm")},
        "B-fewshot": zero_shot + fewshot,
        # G4 reads the Sparse, Moderate and Dense continuations, and every arm's through its few-shot floor
        "G4": [s for s in zero_shot if s.arm in ("Sparse", "Moderate", "Dense")] + fewshot,
    }
    for entry, arms in families.items():
        assert arms, entry
        needed = {k for s in arms for k in (*s.pending, *manifest.run_gate_keys(s))} - everywhere
        missing = needed - set(Q.keys_for(entry))
        assert not missing, f"{entry}: {sorted(missing)}"


def test_h3c_carries_the_result_gates_of_the_training_records_it_reads(monkeypatch) -> None:
    """H3 (c) reads the trainable-layer metrics of the ``training`` supplement records, which
    ``enrich training`` builds in one piece with metrics.controller and metrics.recovery; every result gate those call
    must reach H3 (c) and H3, so that while one is open the report names it (provisional_on) instead of showing plain
    missing data."""
    root = ANALYSIS.parent
    gates = set()
    for name in ("controller.py", "recovery.py"):
        gates.update(re.findall(r'require_answered\("(Q-[a-z0-9-]+)"', (root / "metrics" / name).read_text(encoding="utf-8")))
    assert gates == {"Q-controller-quantities"}
    for entry in ("H3c", "H3c-cell", "H3"):
        assert gates <= set(Q.keys_for(entry)), entry
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset(R.PENDING) - gates)
    assert Q.provisional_on("H3c") == ["Q-controller-quantities"]
    assert Q.provisional_on("H3") == ["Q-controller-quantities"]

"""Which open pre-registration questions each verdict depends on (analysis/__main__.py; configs.registered.PENDING).

Owner: Role 4, Analysis and results (Muhammad Abdullah). Parts 1, 4, 5 and 6 freeze at the go decision
(registration status table: "Parts 1, 4, 5 and 6 are fixed from the go decision"); until the
amendment log answers a question the table below names, every verdict that depends on it is reported
with the key in ``provisional_on``, and a verdict whose readings disagree across an open key is
UNDECIDED(key) (analysis.verdict). Part 8.2: "a change made after data are in view labels every
affected result exploratory" (analysis.records applies AMENDMENTS.json).

Only PENDING keys appear here (``R.is_open`` raises KeyError for any other); a test checks every
key against PENDING. Keys of HANDOVER.md section 9 that are not in PENDING (Q-return-outcome,
Q-detectable-size, Q-exploratory-labels, Q-unmatched-reporting, Q-training-age-systematic,
Q-treatment-other-N, Q-rule3-tolerance-sensitivity, Q-checkpoint-table, ...) are documented in the
docstrings of the code that implements their answers (notes in docs/DECISIONS.md) and are never passed to
``is_open``. "Answered" below means answered in docs/DECISIONS.md (2026-10-02), draft rows that become amendments in
Table 9.1 only when the group ratifies them, and listed in ``R.ANSWERED_QUESTIONS``.

The table lists the report keys (analysis keys of Parts 1, 4, 5), the result gates that produce the
verdict's data (Q-hazard, Q-dynamics, Q-matched-cost-set, Q-studyb-eval, ...) and the design keys
that define what the compared arms are (Q-reset-injection for the treatments, Q-warm-start, ...).
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Iterable, Mapping

from configs import registered as R

# Which runs are compared: Part 4.1 rules 1 to 6 and the seed bookkeeping of Parts 5.5 and 5.6. Q-final-cost-set is
# not here: the matched-checkpoint verdicts never read the final checkpoint's cost; only the rule 7 (b) / Part 4.1.1
# final-checkpoint estimand does (A-final-estimand, and the final-checkpoint rows of tables A_iqm and A_gaps).
MATCHING = ("Q-tie-break", "Q-matched-cost-set", "Q-threshold-arithmetic", "Q-arm-complete",
            "Q-surplus-in-analysis", "Q-seed-collision")
# How any interval is read (Part 5.3).
INTERVALS = ("Q-interval", "Q-bootstrap-details")
# The constrained-steps arms at N = 0.10 and 0.25 (their onset and window are answered keys).
CONSTRAINED_ARMS = ("Q-rounding", "Q-selection-window")
# The data of every Study B verdict: its evaluation and conditioning gates, the seed bookkeeping and
# (Q-arm-complete) which arms are complete (analysis.data.study_b_seed_targets).
STUDY_B_DATA = ("Q-studyb-eval", "Q-budget-normalisation", "Q-level-jc", "Q-surplus-in-analysis", "Q-seed-collision",
                "Q-arm-complete")

# The battery condition behind a gap (Table 2.2): its result gate or its continuation.
CONDITION_KEYS: Mapping[str, tuple[str, ...]] = MappingProxyType({
    "hazard": ("Q-hazard",),
    "dynamics": ("Q-dynamics",),
    "finetune": ("Q-continuations",),
    "transfer": ("Q-continuations", "Q-transfer-obs"),
})

_H1 = (*MATCHING, *INTERVALS, *CONSTRAINED_ARMS, "Q-h1-shape", "Q-controls-reading", "Q-support-falsify-overlap",
       "Q-falsification-calibration", "Q-surplus-arm-set", "Q-ramp-step")
# Q-ramp-step: it gates every ramp run (pilot.manifest), and the ramp arms are compared both by Q-h1-shape's other
# reading and by the ramp-shape cells H1[task/condition/ramp] (table entry "H1-cell").
# Q-falsification-calibration: a falsified box stands only if its Part 5.7 bound is below the minimum effect (Part 1.2;
# Part 5.7); the other reading lets the box conditions decide. It applies to every box whose falsifying quantity
# carries a Part 1.5 minimum effect (H1, H2, H3 (b), H4, G1, G2, G3; analysis.verdict.calibrate).
# Q-h0-scope is not here: it asks on which outcome box H0 reads H1 and H2, which changes H0 (and, through the
# gate, H3 and H4), never H2 itself.
_H2 = (*MATCHING, *INTERVALS, *CONSTRAINED_ARMS, "Q-controls-reading", "Q-holm-families", "Q-ramp-step",
       "Q-falsification-calibration", "Q-surplus-arm-set")
# Q-surplus-arm-set: H2 compares the N = 0.50 ramp arms on SafetyPointGoal1-v0, whose surplus seeds (and so whose seed
# target, analysis.data.study_a_seed_targets) only that key's answer adds.
_H3A = (*MATCHING, *INTERVALS, "Q-h3-scope", "Q-plasticity-definitions", "Q-holm-families")
_H3B = (*MATCHING, *INTERVALS, "Q-h3-scope", "Q-reset-injection", "Q-data-control-lr", "Q-holm-families",
        "Q-support-falsify-overlap", "Q-falsification-calibration")
_H3C = (*MATCHING, *INTERVALS, "Q-h3-scope", "Q-plasticity-definitions", "Q-reset-injection", "Q-holm-families",
        "Q-controller-quantities")
# Q-controller-quantities: box H3 (c) names no controller quantity, but its trainable-layer metrics at onset + 200,000
# steps come only from the ``training`` supplement record, which ``enrich training`` writes in one piece with the
# recovery and controller quantities (pilot.enrichment.training_record), so it waits on that key's result gate; while
# the key is open H3 (c), and so H3, is NOT_COMPUTABLE with the key in provisional_on.
_H4 = (*MATCHING, *INTERVALS, "Q-h4-reading", "Q-warm-start", "Q-h0-scope", "Q-holm-families",
       "Q-falsification-calibration")
# Box H0 reads H1 and H2 on the primary outcome, the gap under hazard relocation (Q-h0-scope, answered: "the
# primary outcome, the robustness gap under hazard relocation on SafetyPointGoal1-v0"), so its keys include that
# condition's result gate.
H0_CONDITION = "hazard"
_H0 = tuple(dict.fromkeys((*_H1, *_H2, "Q-h0-scope", *CONDITION_KEYS[H0_CONDITION])))


def _gated(*keys: str) -> tuple[str, ...]:
    """H3 and H4 are 'reported as not tested' when H0 is supported (box H0): they carry H0's keys too,
    including its condition's (Q-hazard) on every cell, whatever condition the cell itself reads."""
    return tuple(dict.fromkeys((*keys, *_H0)))


REPORT_KEYS: Mapping[str, tuple[str, ...]] = MappingProxyType({
    # Study A (boxes H0 to H4, Part 1.2)
    "matching": MATCHING,
    "H1": _H1,
    "H1-cell": (*_H1, "Q-holm-families"),
    "H2": _H2,
    "H2-cell": _H2,
    "H3a": _gated(*_H3A),
    "H3a-cell": _gated(*_H3A),
    "H3b": _gated(*_H3B),
    "H3b-cell": _gated(*_H3B),
    "H3c": _gated(*_H3C),
    "H3c-cell": _gated(*_H3C),
    "H3": _gated(*_H3A, *_H3B, *_H3C),
    "H4": _gated(*_H4),
    "H4-cell": _gated(*_H4),
    "H0": _H0,
    # Study A secondary outcomes (Part 5.2) and Part 4.1.1 / rule 7
    # recovery: abrupt vs ramp arms of both controls (constrained-steps N = 0.10/0.25 included), and every treated
    # arm (reset, injection, additional constrained training, warm-started, rate-limited, PID) vs the untreated one
    "A-recovery": (*MATCHING, *INTERVALS, *CONSTRAINED_ARMS, "Q-controller-quantities", "Q-jc-window",
                   "Q-holm-families", "Q-ramp-step", "Q-reset-injection", "Q-data-control-lr", "Q-warm-start",
                   "Q-rate-limit", "Q-pid-eq9",
                   "Q-surplus-arm-set"),  # its abrupt-ramp N = 0.50 cells read the ramp arm on SafetyPointGoal1-v0
    # return: the abrupt arms of both controls (constrained-steps N = 0.10/0.25 included) vs N = 0
    "A-return": (*MATCHING, *INTERVALS, *CONSTRAINED_ARMS, "Q-holm-families"),
    "A-final-estimand": (*_H1, "Q-final-cost-set"),
    "A-rule7a": _H1,
    "A-controller": ("Q-controller-quantities", "Q-surplus-in-analysis"),
    "A-iqm": ("Q-iqm", "Q-bootstrap-details", *MATCHING, "Q-final-cost-set"),  # the table holds final-estimand rows too
    # the per-seed gaps of every condition under the three estimands (table A_gaps) and the norms (A_norms)
    "A-gaps": (*MATCHING, "Q-final-cost-set", *dict.fromkeys(k for ks in CONDITION_KEYS.values() for k in ks)),
    "A-norms": (*MATCHING, "Q-plasticity-definitions"),
    # Study B (boxes G1 to G5, Part 1.4; Part 4.2)
    # G1, G2 and B-fewshot read the Continuous arm through the floor rule (Q-floor-rule, answered: all seven arms)
    "G1": (*STUDY_B_DATA, *INTERVALS, "Q-floor-rule", "Q-g1-criteria", "Q-support-falsify-overlap",
           "Q-falsification-calibration", "Q-continuous-bins"),
    "G2": (*STUDY_B_DATA, *INTERVALS, "Q-floor-rule", "Q-g2-outcome", "Q-holm-families", "Q-falsification-calibration",
           "Q-continuous-bins"),
    "G3": (*STUDY_B_DATA, *INTERVALS, "Q-floor-rule", "Q-g3-arms", "Q-continuous-bins", "Q-falsification-calibration"),
    # G4 reads every arm's continuations through its few-shot floor (Q-floor-rule) and compares floored budgets on
    # few-shot violation magnitude by interval; under Q-g4-reading's 'at most one horizon' reading its support and
    # falsification can hold at once (an even seed count gives a median index x.5), which box_status settles by
    # Q-support-falsify-overlap's answer
    "G4": (*STUDY_B_DATA, *INTERVALS, "Q-continuations", "Q-adapt-censoring", "Q-g4-reading", "Q-floor-rule",
           "Q-continuous-bins", "Q-support-falsify-overlap"),
    "G5": (*STUDY_B_DATA, *INTERVALS, "Q-floor-rule", "Q-continuous-bins"),
    "B-per-budget": (*STUDY_B_DATA, *INTERVALS, "Q-floor-rule", "Q-holm-families", "Q-continuous-bins"),
    "B-violation": (*STUDY_B_DATA, *INTERVALS, "Q-holm-families", "Q-continuous-bins"),
    "B-fewshot": (*STUDY_B_DATA, "Q-continuations", "Q-floor-rule", "Q-continuous-bins"),
    "B-iqm": ("Q-iqm", "Q-bootstrap-details", *STUDY_B_DATA, "Q-floor-rule", "Q-continuous-bins"),
    # Pilot mode (Part 5.8 test run; go conditions G1 and G3 of Part 6)
    "pilot-G1": ("Q-matched-cost-set", "Q-g1-level", "Q-threshold-arithmetic"),
    "pilot-G3": ("Q-g3-pairing", "Q-hazard", "Q-threshold-arithmetic"),  # 'at least 5' after rounding (go_decision)
})


# The tables whose rows rest on open questions, with their entry above: the report adds a provisional_on
# column to each (analysis.report). The pilot consistency rows carry "pilot-G1" or "pilot-G3" by go condition
# (analysis.pilot_check).
TABLE_ENTRIES: Mapping[str, str] = MappingProxyType({
    "A_matching": "matching",
    "A_unmatched": "matching",
    "A_gaps": "A-gaps",
    "A_estimands_side_by_side": "A-final-estimand",
    "A_norms": "A-norms",
    "A_controller": "A-controller",
    "A_iqm": "A-iqm",
    "B_iqm": "B-iqm",
    "B_fewshot": "B-fewshot",
})
PILOT_ENTRIES: Mapping[str, str] = MappingProxyType({"G1": "pilot-G1", "G3": "pilot-G3"})


def keys_for(verdict_id: str, *, conditions: Iterable[str] = ()) -> tuple[str, ...]:
    """Every PENDING key the verdict depends on (open or answered), plus the keys of its conditions."""
    if verdict_id not in REPORT_KEYS:
        raise KeyError(f"no question table entry for verdict {verdict_id!r}")
    keys = list(REPORT_KEYS[verdict_id])
    for condition in conditions:
        keys.extend(CONDITION_KEYS[condition])
    return tuple(dict.fromkeys(keys))


def provisional_on(verdict_id: str, *, conditions: Iterable[str] = ()) -> list[str]:
    """The keys of ``keys_for`` that the amendment log has not answered yet (R.is_open)."""
    return [k for k in keys_for(verdict_id, conditions=conditions) if R.is_open(k)]


def open_keys() -> list[str]:
    """Every PENDING key still open (listed in each report)."""
    return [k for k in R.PENDING if R.is_open(k)]


def all_table_keys() -> set[str]:
    """Every key named by the table (a test checks each is in PENDING)."""
    keys = {k for ks in REPORT_KEYS.values() for k in ks}
    keys.update(k for ks in CONDITION_KEYS.values() for k in ks)
    return keys

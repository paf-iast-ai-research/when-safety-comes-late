"""Go or no-go rule (Part 6) and the pilot's empirical values (Part 8, Table 8.1).

"The decision is taken by the group at the go meeting, from the pilot's results and nothing else."
This module computes the four conditions exactly as registered and states which branch of Part 6
they lead to; it does not take the decision. Owner: pilot owner (Role 1).

The conditions are the go conditions G1 to G4 of Part 6, not Study B's hypotheses G1 to G5 (Part 1.4);
the report says so in its headings (Table 9.1, Q-g-names).

G1 feasibility (Study A): both pilot arms' final in-distribution cost <= d + 2.5 (arm means,
   answered in Table 9.1, Q-g1-level); the reference arm's seed-to-seed SD of that cost <= 10
   (final-checkpoint SD reported beside it); the two arms match within the tolerance of Part 4 (2.5).
G2 throughput: wall-clock per run / concurrent runs x the corrected run-equivalent total (627.13 for
   the uncut design; Table 9.1, Q-g2-run-equivalents) <= allocated machine-hours, the registered 484
   reported beside it, with the allocation committed before throughput is read; this is enforced as
   committed before any pilot run was launched (budget.allocation_contained), the stricter condition.
G3 signal (Study A): mean Delta(0.50) under hazard relocation over the three seeds >= 5, or its
   upper bound U80 (equation 12, paired over seeds; Table 9.1, Q-g3-pairing) >= 5.
G4 conditioning (Study B): every budget of Table 2.5 lies below the pilot's unconstrained cost; the
   Moderate arm's mean costs at its three training budgets are ordered with the budgets and each
   at most its budget + 2.5 (Part 6; answered in Table 9.1, Q-g4-level).

Open questions (configs/registered.py PENDING; the go-report keys, the result gates Q-hazard and
Q-matched-cost-set, and the analysis key Q-threshold-arithmetic), each answered in Table 9.1
(docs/DECISIONS.md, 2026-10-02): the report decides every condition by its answer. A key is open again
only when the group changes its answer, until the code follows; the report then shows a condition
UNDECIDED while an open key's readings disagree, and G3 while Q-hazard is open:
  Q-g1-level          G1 on each arm's mean (the answer) or on every seed: UNDECIDED while open and
                      the per-seed reading changes G1;
  Q-matched-cost-set  G1 on the measurement set of the selected checkpoint (the answer) or on its
                      selection set: UNDECIDED while open and the selection-set reading changes G1;
                      while both keys are open, the selection set read seed by seed is checked too;
  Q-g3-pairing        U80 paired over seeds; runs left unpaired by a replacement seed are paired in
                      increasing seed order, so G3 is always decided by the rule; the pooled-SD value
                      is a descriptive figure, not a reading (a note says G3 is provisional on the
                      key while it is open);
  Q-hazard            G3's hazard-relocation gap rests on the reading of Table 2.2 the key answers:
                      G3 is UNDECIDED while it is open;
  Q-final-cost-set    the unconstrained run's cost at its final checkpoint on the measurement set
                      (the answer; final_cost) or on the selection set (its checkpoint record):
                      G4 is UNDECIDED while open and the selection-set reading changes G4;
  Q-g4-level          reported beside G4 (Part 6 decides);
  Q-threshold-arithmetic  G1's clauses compare as Part 4.1 rules 3 and 5 do under its answer
                      (sample SD, inclusive, the means and their difference after rounding to 1e-4, the SD
                      after rounding to 1e-9), and so does G3's "at least 5" (the mean Delta(0.50) after
                      rounding to 1e-4, U80 after rounding to 1e-9); a note says G1 and G3 are
                      provisional on it while it is open (it does not make them UNDECIDED).
When several keys hold one condition, ``undecided_by`` names them all, separated by ", ".
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from configs import registered as R
from pilot import budget, provenance

REF_N, LATE_N = R.PILOT_STUDY_A_ONSET_FRACTIONS  # Part 3.6: the pilot arms N = 0 and N = 0.50


@dataclass
class Condition:
    """One go condition of Part 6: its verdict, the values behind it and the notes that qualify it."""

    name: str
    passed: bool | None = None  # None: not computable from the data given, or undecided (undecided_by)
    values: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    # The PENDING key or keys (joined by ", ") under which the group must decide this condition: its readings
    # disagree while the key is open; Q-hazard holds G3 whenever it is open, whatever the readings.
    undecided_by: str | None = None


def _sd(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) >= 2 else math.nan


def _pilot_rows(rows: Iterable[Mapping[str, Any]], N: float) -> list[Mapping[str, Any]]:
    return [
        r for r in rows
        if r["study"] == "A" and r["task"] == R.PILOT_STUDY_A_TASK and r["N"] == N and r["completed"]
        and (N == REF_N or (r["onset_shape"] == R.PILOT_STUDY_A_SHAPE and r["step_matching"] == R.PILOT_STUDY_A_CONTROL))
        and r.get("treatment") is None and r.get("controller_variant") is None
    ]


COST_FIELD = "measurement_cost"  # answered in Table 9.1 (Q-matched-cost-set): the measurement set, as
# analysis/matching.py uses (Table 2.1 reserves that set for the reported in-distribution cost)
SELECTION_COST_FIELD = "selection_cost_at_match"  # the other reading of Q-matched-cost-set (the selection set)
# answered in Table 9.1 (Q-threshold-arithmetic): G1's clauses compare as Part 4.1 rules 3 and 5 do, after rounding to
# 1e-4 (pilot.enrichment.MATCH_DECIMALS and analysis.matching.THRESHOLD_DECIMALS; a test checks they agree).
# Costs are means of 100 integer episode costs, so 1e-4 absorbs floating-point noise without moving a real
# boundary: unrounded, arm means 24.90 and 27.40 differ by 2.500000000000007, and "matched under the
# tolerance of Part 4" (inclusive) would fail on noise alone while Part 4's matching calls them matched.
# The reference SD is a square root, not on the 0.01 grid: an SD of 10.000004 would round to 10.0 at 1e-4, so
# it is compared after rounding to SD_DECIMALS (1e-9, floating-point noise only), as analysis.matching does.
THRESHOLD_DECIMALS = 4
SD_DECIMALS = 9


def _at_most(value: float, limit: float, decimals: int = THRESHOLD_DECIMALS) -> bool:
    """``value <= limit`` after rounding ``value`` to ``decimals`` (G1's "at most" and Part 4's "<=")."""
    return round(value, decimals) <= limit


def _at_least(value: float, limit: float, decimals: int = THRESHOLD_DECIMALS) -> bool:
    """``value >= limit`` after rounding ``value`` to ``decimals`` (G3's "at least"; the counterpart of ``_at_most``)."""
    return round(value, decimals) >= limit


def _g1_clauses(ref_costs: Sequence[float], late_costs: Sequence[float]) -> dict[str, Any]:
    """Part 6 G1's three clauses on one set of per-seed costs.

    ``passed`` is the arm-mean reading (answered in Table 9.1, Q-g1-level) and ``passed_every_seed`` the
    per-seed reading, reported beside it. "at most d + 2.5", "at most 10" and "matched under the tolerance
    of Part 4" (rule 5, "<= 2.5") are compared after rounding (``_at_most``), as Part 4's matching does; the
    values reported are unrounded.
    """
    ref_mean, late_mean = statistics.fmean(ref_costs), statistics.fmean(late_costs)
    limit = R.COST_LIMIT + R.G1_COST_MARGIN
    within = _at_most(ref_mean, limit) and _at_most(late_mean, limit)
    sd = _sd(ref_costs)
    # False for NaN (fewer than two seeds): an SD that cannot be computed never satisfies "at most 10"
    sd_ok = _at_most(sd, R.G1_REFERENCE_SD_MAX, SD_DECIMALS)
    matched = _at_most(abs(late_mean - ref_mean), R.MATCH_TOLERANCE)
    every_seed = all(_at_most(c, limit) for c in list(ref_costs) + list(late_costs))
    return {"reference_cost_mean": ref_mean, "late_cost_mean": late_mean, "reference_sd": sd,
            "match_difference": late_mean - ref_mean, "both_arms_within_budget": within,
            "every_seed_within_budget": every_seed, "reference_sd_ok": sd_ok, "arms_matched": matched,
            "cost_limit_plus_margin": limit,
            "passed": bool(within and sd_ok and matched), "passed_every_seed": bool(every_seed and sd_ok and matched)}


def _finite_selection_cost(row: Mapping[str, Any]) -> bool:
    """Whether ``row`` gives a finite selection-set cost (the other reading of Q-matched-cost-set)."""
    return row.get(SELECTION_COST_FIELD) is not None and math.isfinite(float(row[SELECTION_COST_FIELD]))


def _undecided(cond: Condition, keys: list[str]) -> None:
    if keys:
        cond.passed = None
        cond.undecided_by = ", ".join(keys)


def g1_feasibility(rows: Sequence[Mapping[str, Any]]) -> Condition:
    """Part 6 G1 on the final in-distribution cost: the measurement-set cost of the selected checkpoint.

    Table 2.1 reserves the measurement set for the reported in-distribution cost; choosing and
    measuring on the same (selection) episodes "would bias the reported cost of the chosen
    checkpoint downward". The measurement set is the answer of Q-matched-cost-set (Table 9.1); while that key
    is open and the selection-set reading (``selection_cost_at_match``) would change G1, G1 is
    left UNDECIDED; the selection-set values are reported beside it either way. A cost that
    is missing or not finite makes G1 not computable (a NaN would otherwise fail every clause). "Both
    pilot arms reach ... at most d + 2.5" is read on each arm's mean (answered in Table 9.1, Q-g1-level);
    while that key is open, G1 is left UNDECIDED if the per-seed reading would change the result.
    While both keys are open, the combined reading (the selection set, seed by seed) is checked as well,
    and G1 is left UNDECIDED under every open key whose other reading changes the result.
    """
    ref, late = _pilot_rows(rows, REF_N), _pilot_rows(rows, LATE_N)
    cond = Condition("G1 feasibility")
    cond.values["cost_field"] = COST_FIELD
    missing = [r["run_id"] for r in ref + late if r.get(COST_FIELD) is None]
    non_finite = [r["run_id"] for r in ref + late
                  if r.get(COST_FIELD) is not None and not math.isfinite(float(r[COST_FIELD]))]
    n = len(R.PILOT_SEEDS)
    if len(ref) != n or len(late) != n or missing or non_finite:
        cond.passed = None
        cond.notes.append(
            f"not computable: Part 6 G1 needs {n} completed runs per arm (reference {len(ref)}, late {len(late)})"
            + (f"; missing {COST_FIELD} in {missing}" if missing else "")
            + (f"; {COST_FIELD} must be finite, not in {non_finite}" if non_finite else "")
        )
        return cond
    ref_costs = [float(r[COST_FIELD]) for r in ref]
    late_costs = [float(r[COST_FIELD]) for r in late]
    clauses = _g1_clauses(ref_costs, late_costs)
    finals = [r.get("final_cost") for r in ref]
    ref_final_sd = _sd([float(x) for x in finals]) if all(x is not None for x in finals) else None
    selection = None
    if all(_finite_selection_cost(r) for r in ref + late):
        selection = _g1_clauses([float(r[SELECTION_COST_FIELD]) for r in ref],
                                [float(r[SELECTION_COST_FIELD]) for r in late])
    cond.values.update(
        reference_seeds=[r["seed"] for r in ref], late_seeds=[r["seed"] for r in late],
        reference_costs=ref_costs, late_costs=late_costs,
        reference_cost_mean=clauses["reference_cost_mean"], late_cost_mean=clauses["late_cost_mean"],
        cost_limit_plus_margin=clauses["cost_limit_plus_margin"],
        reference_sd=clauses["reference_sd"], reference_final_checkpoint_sd=ref_final_sd,
        match_difference=clauses["match_difference"], tolerance=R.MATCH_TOLERANCE,
        both_arms_within_budget=clauses["both_arms_within_budget"],
        every_seed_within_budget=clauses["every_seed_within_budget"],
        reference_sd_ok=clauses["reference_sd_ok"], arms_matched=clauses["arms_matched"],
        selection_set_means=[
            statistics.fmean(float(r[SELECTION_COST_FIELD]) for r in arm)
            if all(_finite_selection_cost(r) for r in arm) else None
            for arm in (ref, late)
        ],
        selection_set_reading=None if selection is None else {
            k: selection[k] for k in ("reference_cost_mean", "late_cost_mean", "reference_sd", "match_difference",
                                      "every_seed_within_budget", "passed", "passed_every_seed")
        },
    )
    cond.notes.append(
        "the budget clause is read on each arm's mean over seeds (Table 9.1, Q-g1-level); "
        "every_seed_within_budget is reported beside it and decides nothing"
    )
    # The verdict under each reading of the two keys (cost set x level); the answer (``answer``) is the measurement
    # set on arm means. A reading counts only while every key it departs from is open.
    answer = clauses["passed"]
    level_open, set_open = R.is_open("Q-g1-level"), R.is_open("Q-matched-cost-set")
    readings = {("measurement", "mean"): answer, ("measurement", "seed"): clauses["passed_every_seed"]}
    if selection is not None:
        readings.update({("selection", "mean"): selection["passed"], ("selection", "seed"): selection["passed_every_seed"]})
    readings = {(cost_set, level): verdict for (cost_set, level), verdict in readings.items()
                if (level == "mean" or level_open) and (cost_set == "measurement" or set_open)}
    differing = {k for k, verdict in readings.items() if verdict != answer}
    # each open key whose two readings disagree, the other key's reading held fixed, holds G1
    keys = []
    if differing and any((s, "seed") in readings and readings[(s, "seed")] != readings[(s, "mean")]
                         for s in ("measurement", "selection") if (s, "mean") in readings):
        keys.append("Q-g1-level")
    if differing and any(("selection", lv) in readings and readings[("selection", lv)] != readings[("measurement", lv)]
                         for lv in ("mean", "seed") if ("measurement", lv) in readings):
        keys.append("Q-matched-cost-set")
    if ("measurement", "seed") in differing:
        cond.notes.append(
            "the arm-mean and per-seed readings of Part 6 G1 disagree; the group must answer Q-g1-level before G1 is read"
        )
    if R.is_open("Q-threshold-arithmetic"):
        cond.notes.append(
            "the clauses compare as Part 4.1 rules 3 and 5 do under the proposal of the open question "
            f"Q-threshold-arithmetic (sample SD, inclusive bounds, the means and their difference after rounding to "
            f"1e-{THRESHOLD_DECIMALS}, the SD after rounding to 1e-{SD_DECIMALS}); G1 is provisional on it"
        )
    if selection is None:
        cond.notes.append(
            f"the selection-set reading of Q-matched-cost-set is not computed: {SELECTION_COST_FIELD} is missing "
            "or not finite"
        )
    elif ("selection", "mean") in differing:
        cond.notes.append(
            f"G1 {'passes' if clauses['passed'] else 'fails'} on the measurement set (the proposal of the open question "
            "Q-matched-cost-set) "
            f"but {'passes' if selection['passed'] else 'fails'} on the selection set; the group must answer "
            "Q-matched-cost-set before G1 is read"
        )
        if (("selection", "seed") in readings and readings[("selection", "seed")] != readings[("selection", "mean")]
                and ("measurement", "seed") not in differing):
            cond.notes.append(
                "on the selection set the arm-mean and per-seed readings of Part 6 G1 disagree; Q-g1-level holds G1 "
                "as well, and the group must answer it before G1 is read"
            )
    if ("selection", "seed") in differing and ("selection", "mean") not in differing:
        cond.notes.append(
            f"G1 {'passes' if answer else 'fails'} on the measurement set read on arm means (the proposals of the open "
            "questions) but "
            f"{'passes' if selection['passed_every_seed'] else 'fails'} on the selection set read seed by seed; the group "
            "must answer Q-g1-level and Q-matched-cost-set before G1 is read"
        )
    cond.passed = answer
    _undecided(cond, keys)
    return cond


def u80(differences: Sequence[float]) -> float:
    """Equation (12): U80 = mean + 1.886 s / sqrt(3), with s the SD of the three per-seed differences (the paired
    form, answered in Table 9.1, Q-g3-pairing; g3_signal reports the pooled-SD value as a descriptive figure)."""
    n = len(R.PILOT_SEEDS)
    if len(differences) != n:
        raise ValueError("equation (12) is registered for exactly three seeds")
    return statistics.fmean(differences) + R.U80_T_QUANTILE * statistics.stdev(differences) / math.sqrt(n)


def g3_signal(rows: Sequence[Mapping[str, Any]]) -> Condition:
    """Part 6 G3 on the hazard-relocation gap (equation 2: Delta(0.50) = gap(0.50) - gap(0)).

    The gap rests on the reading of Table 2.2's hazard relocation that Q-hazard answers (and the
    harness refuses to measure it while the key is open), so G3 is UNDECIDED (Q-hazard) while the key
    is open, whatever the numbers; they are reported when present.

    U80 is equation (12)'s paired form (Table 9.1, Q-g3-pairing): the three per-seed differences of
    gap_hazard, N = 0.50 minus N = 0, paired by seed id; runs a replacement seed leaves unpaired are paired
    in increasing seed order (``_seed_pairs``), so G3 is decided by the rule, never by the group. While the
    key is open, a note says G3 is provisional on it.

    "At least 5" is compared after rounding, as G1's clauses are (answered in Table 9.1, Q-threshold-arithmetic):
    the mean Delta(0.50) (a mean of three gaps on the 0.01 grid) to 1e-4 (THRESHOLD_DECIMALS), U80 (it
    carries a standard deviation, not on the grid) to 1e-9 (SD_DECIMALS, floating-point noise only).
    Unrounded, gaps whose paired differences are all exactly 5.00 gave a mean of 4.9999999999999964 and
    failed G3 on noise alone. The values reported are unrounded.
    """
    cond = _g3_signal(rows)
    if "U80" in cond.values and R.is_open("Q-g3-pairing"):
        cond.notes.append("U80 is equation (12)'s paired form, unmatched runs paired in seed order, as the open "
                          "question Q-g3-pairing proposes; G3 is provisional on it")
    if "delta_mean" in cond.values and R.is_open("Q-threshold-arithmetic"):
        cond.notes.append(
            "'at least 5' compares as Part 4.1 rules 3 and 5 do under the proposal of the open question "
            f"Q-threshold-arithmetic (the mean Delta(0.50) after rounding to 1e-{THRESHOLD_DECIMALS}, U80 after "
            f"rounding to 1e-{SD_DECIMALS}); G3 is provisional on it"
        )
    if R.is_open("Q-hazard"):
        keys = ["Q-hazard"] + ([cond.undecided_by] if cond.undecided_by else [])
        cond.notes.append("G3 reads gap_hazard, whose hazard relocation is the proposal of the open question "
                          "Q-hazard; the group must answer Q-hazard before G3 is read")
        _undecided(cond, keys)
    return cond


def _seed_pairs(ref_seeds: Iterable[int], late_seeds: Iterable[int]) -> list[tuple[int, int]]:
    """(reference seed, late seed) pairs for equation (12) (Table 9.1, Q-g3-pairing): equal seed ids first
    (Part 5.1 pairs seed k with seed k), then each arm's remaining seeds sorted and matched one to one."""
    ref, late = set(ref_seeds), set(late_seeds)
    common = sorted(ref & late)
    return [(s, s) for s in common] + list(zip(sorted(ref - late), sorted(late - ref)))


def _g3_signal(rows: Sequence[Mapping[str, Any]]) -> Condition:
    ref = {r["seed"]: r for r in _pilot_rows(rows, REF_N)}
    late = {r["seed"]: r for r in _pilot_rows(rows, LATE_N)}
    cond = Condition("G3 signal")
    missing = [r["run_id"] for r in list(ref.values()) + list(late.values()) if r.get("gap_hazard") is None]
    n = len(R.PILOT_SEEDS)
    if len(ref) != n or len(late) != n or missing:
        cond.passed = None
        cond.notes.append(
            f"not computable: Part 6 G3 is over {n} seeds per arm (reference {len(ref)}, late {len(late)})"
            + (f"; missing gap_hazard in {missing}" if missing else "")
        )
        return cond
    delta_mean = statistics.fmean(float(r["gap_hazard"]) for r in late.values()) - statistics.fmean(
        float(r["gap_hazard"]) for r in ref.values()
    )
    pairs = _seed_pairs(ref, late)
    diffs = [float(late[b]["gap_hazard"]) - float(ref[a]["gap_hazard"]) for a, b in pairs]
    ref_gaps = [float(r["gap_hazard"]) for r in ref.values()]
    late_gaps = [float(r["gap_hazard"]) for r in late.values()]
    pooled_s = math.sqrt((statistics.variance(ref_gaps) + statistics.variance(late_gaps)) / 2)
    bound = u80(diffs)
    cond.values.update(
        delta_mean=delta_mean, paired_seeds=[a for a, b in pairs if a == b], pairs=[list(p) for p in pairs],
        paired_differences=diffs, minimum_effect=R.G3_SIGNAL_MIN, U80=bound,
        U80_pooled=delta_mean + R.U80_T_QUANTILE * pooled_s / math.sqrt(n),
    )
    cond.notes.append("U80 is equation (12)'s paired form over seeds (Table 9.1, Q-g3-pairing); U80_pooled, with the "
                      "pooled SD, is a descriptive figure, not a reading of equation (12), and decides nothing")
    if any(a != b for a, b in pairs):
        cond.notes.append(f"unmatched runs paired in seed order (Q-g3-pairing): the arms share "
                          f"{sum(a == b for a, b in pairs)} of {n} seeds (a replacement seed); pairs (reference, late) "
                          f"{pairs}; the mean Delta(0.50) is the same under any pairing")
    cond.passed = bool(_at_least(delta_mean, R.G3_SIGNAL_MIN) or _at_least(bound, R.G3_SIGNAL_MIN, SD_DECIMALS))
    return cond


def all_study_b_budgets() -> list[float]:
    """Every budget of Table 2.5: all training levels, the continuous range and the unseen budgets."""
    budgets = set(R.UNSEEN_BUDGETS) | set(R.CONTINUOUS_RANGE)
    for levels in R.STUDY_B_ARMS.values():
        if levels:
            budgets |= set(levels)
    return sorted(budgets)


def g4_conditioning(unconstrained_cost: float | None, moderate_mean_costs: Mapping[float | str, float] | None, *,
                    unconstrained_selection_cost: float | None = None) -> Condition:
    """Part 6 G4. The unconstrained run's cost is its final_cost, the measurement set at its final
    checkpoint (answered in Table 9.1, Q-final-cost-set); ``unconstrained_selection_cost`` is the other
    reading, the selection-set cost of the same checkpoint, which decides nothing: while Q-final-cost-set is
    open and that reading would change G4, G4 is UNDECIDED, and when it is missing or not finite that reading
    is not computed (as G1's selection-set reading). Any other cost that is not finite makes G4 not
    computable (a NaN would otherwise fail every comparison)."""
    cond = Condition("G4 conditioning")
    if unconstrained_cost is None or not moderate_mean_costs:
        cond.passed = None
        cond.notes.append("not computable: the unconstrained run's cost or the Moderate arm's costs are missing")
        return cond
    budgets = all_study_b_budgets()
    below = max(budgets) < float(unconstrained_cost)
    selection = None if unconstrained_selection_cost is None else float(unconstrained_selection_cost)
    if selection is not None and not math.isfinite(selection):  # the other reading, not computed (it decides nothing)
        selection = None
    below_selection = None if selection is None else max(budgets) < selection
    levels = sorted(float(b) for b in R.STUDY_B_ARMS["Moderate"])
    costs = {float(k): float(v) for k, v in moderate_mean_costs.items()}
    if sorted(costs) != levels:
        cond.passed = None
        cond.notes.append(f"not computable: the Moderate arm's costs must be given for exactly the budgets {levels}")
        return cond
    non_finite = [float(x) for x in (unconstrained_cost, *costs.values()) if not math.isfinite(float(x))]
    if non_finite:
        cond.passed = None
        cond.notes.append(
            "not computable: the unconstrained run's cost and the Moderate arm's costs must be finite "
            f"(non-finite: {non_finite})"
        )
        return cond
    ordered = all(costs[a] < costs[b] for a, b in zip(levels, levels[1:]))
    within = {b: costs[b] <= b + R.G4_COST_MARGIN for b in levels}
    two_sided = {b: abs(costs[b] - b) <= R.G4_COST_MARGIN for b in levels}  # Part 3.6 wording (Q-g4-level)
    cond.values.update(
        each_within_2_5_two_sided_part_3_6=two_sided,
        unconstrained_cost=float(unconstrained_cost), largest_budget=max(budgets),
        every_budget_below_unconstrained=below, moderate_mean_costs=costs,
        ordered_with_budgets=ordered, each_within_budget_plus_margin=within,
        unconstrained_selection_set_cost=selection,
        every_budget_below_unconstrained_selection_set=below_selection,
    )
    cond.passed = bool(below and ordered and all(within.values()))
    if cond.passed and not all(two_sided.values()):
        cond.notes.append("on the measurement-set reading G4 passes by Part 6 ('at most its budget plus 2.5') but not by "
                          "Part 3.6 ('within 2.5 of its budget'); see Q-g4-level")
    if below_selection is None:
        cond.notes.append("the selection-set reading of Q-final-cost-set is not computed: the unconstrained run's "
                          "selection-set cost at its final checkpoint is not given or not finite")
    elif bool(below_selection and ordered and all(within.values())) != cond.passed and R.is_open("Q-final-cost-set"):
        cond.notes.append(
            f"G4 {'passes' if cond.passed else 'fails'} with the unconstrained cost on the measurement set (final_cost, "
            f"the proposal of the open question Q-final-cost-set) but not with its selection-set cost; the group must answer "
            "Q-final-cost-set before G4 is read"
        )
        _undecided(cond, ["Q-final-cost-set"])
    return cond


def g2_throughput(
    hours_per_run: Sequence[float],
    concurrent_runs: int,
    machine_hours_allocated: float | None,
    allocation_ok: bool,
    allocation_note: str,
) -> Condition:
    """Part 6 G2, amended before the pilot (Table 9.1, Q-g2-run-equivalents): mean wall-clock hours per run /
    concurrent runs x the corrected run-equivalent total (``budget.corrected_requirement``: training steps in
    units of T, the battery's continuations included) <= the allocated machine-hours. The registered 484
    (435 + 49) is reported beside it and decides nothing. ``allocation_ok`` says whether the allocation was
    committed before the pilot runs were launched (``budget.allocation_contained``) and ``allocation_note``
    why; G2 is read only if it was."""
    cond = Condition("G2 throughput")
    if len(hours_per_run) != R.PILOT_RUNS or machine_hours_allocated is None:
        cond.passed = None
        cond.notes.append(
            f"not computable: {len(hours_per_run)} of {R.PILOT_RUNS} pilot runs completed, "
            f"allocation {'missing' if machine_hours_allocated is None else 'present'}"
        )
        return cond
    bad_hours = [h for h in hours_per_run if not math.isfinite(h) or h <= 0]
    if bad_hours or not math.isfinite(machine_hours_allocated):
        cond.passed = None
        cond.notes.append(
            f"not computable: wall-clock hours per run must be finite and positive (bad: {bad_hours}) and the "
            f"allocation finite (allocated: {machine_hours_allocated})"
        )
        return cond
    mean_hours = statistics.fmean(hours_per_run)
    required = budget.registered_requirement(mean_hours, concurrent_runs)
    corrected = budget.corrected_requirement(mean_hours, concurrent_runs)
    cond.values.update(
        hours_per_run_mean=mean_hours, hours_per_run_max=max(hours_per_run), concurrent_runs=concurrent_runs,
        run_equivalents_registered=budget.registered_run_equivalents(),
        required_machine_hours_registered=required,
        run_equivalents_corrected=budget.corrected_run_equivalents()["total"],
        required_machine_hours_corrected=corrected,
        machine_hours_allocated=machine_hours_allocated, allocation_check=allocation_note,
    )
    if not allocation_ok:
        cond.passed = None
        cond.notes.append(
            "the allocation was not committed before the pilot runs were launched (Part 6 G2: it must be fixed "
            "before throughput is read); the condition cannot be evaluated"
        )
        return cond
    cond.passed = bool(corrected <= machine_hours_allocated)
    if corrected > machine_hours_allocated >= required:
        cond.notes.append(
            "the registered 484 run-equivalents would fit the allocation, but the corrected total does not; G2 is "
            "decided on the corrected total (Table 9.1, Q-g2-run-equivalents)"
        )
    return cond


CUT_ORDER = (
    "apply the cut order of Part 6.1, each step only if the previous is insufficient: (1) drop the PID "
    "check; (2) drop SafetyCarGoal1-v0 from every group of Study A; (3) shorten Study B's few-shot "
    "continuations from 1,000,000 to 200,000 steps, keeping only the first horizon; (4) restrict Study A's "
    "mediation treatments and controller variants to N = 0.50; (5) drop N = 0.10 from the main sweep. Never cut "
    "the number of seeds, Study B's seven zero-shot arms, or the N = 0 and N = 0.50 arms under both step-matching "
    "controls on SafetyPointGoal1-v0 and SafetyPointButton1-v0."
)
G3_NO_GO = (
    "NO-GO for Study A: any timing effect is below the minimum effect of interest in the pilot setting; "
    "write the pilot up as a negative pilot with its bound (U80), and Study B becomes the main study."
)


def part6_reading(
    g1: Condition, g2: Condition, g3: Condition, g4: Condition, *, failed_before: Iterable[str] = ()
) -> str:
    """The branch of Part 6 the four conditions lead to (the group takes the decision).

    ``failed_before`` names the conditions (``"G1"``, ``"G2"``, ``"G4"``) that failed on the first
    pilot, when these results come from the re-pilot after the one permitted revision. Only "a
    second failure of the same condition is a no-go for the study it concerns" (a second G2
    failure leads to the cut order of Part 6.1); a condition failing for the first time on the
    re-pilot is a case Part 6 does not cover and is reported as a question for the group.
    """
    failed_before = frozenset(failed_before)
    unknown = failed_before - {"G1", "G2", "G4"}
    if unknown:
        raise ValueError(f"failed_before names conditions that cannot be revised: {sorted(unknown)}")
    conds = {c.name.split()[0]: c for c in (g1, g2, g3, g4)}
    states = {k: c.passed for k, c in conds.items()}
    # Part 6 reads G3 only when G1 and G2 hold: once either has failed, an undecided G3 cannot change
    # the branch, and it is reported as undecided inside that branch instead of withholding it.
    g3_moot = states["G1"] is False or states["G2"] is False
    missing = [
        k for k, c in conds.items() if c.passed is None and not c.undecided_by and not (k == "G3" and g3_moot)
    ]
    waiting = [
        f"{k} waits on {c.undecided_by}" for k, c in conds.items()
        if c.passed is None and c.undecided_by and not (k == "G3" and g3_moot)
    ]
    if missing or waiting:
        parts = []
        if missing:
            parts.append("INCOMPLETE: " + ", ".join(missing) + " cannot be evaluated yet.")
        if waiting:
            parts.append(
                "UNDECIDED: " + "; ".join(waiting)
                + " (the group decides first; the condition's notes give the readings)."
            )
        return " ".join(parts)
    # G3 can be None here only when g3_moot: an undecided G3 names its keys; a G3 with no undecided_by is not computable
    g3_open = (f"G3 is undecided ({conds['G3'].undecided_by})" if conds["G3"].undecided_by
               else "G3 is not computable") if states["G3"] is None else None
    if all(states.values()):
        return "GO: enter the measured values in Part 8, commit the amended document, and freeze Parts 1, 4, 5 and 6."
    failed = [k for k, v in states.items() if not v]
    parts = []
    if failed_before:
        study_a_nogo = not states["G1"] and "G1" in failed_before
        if study_a_nogo:
            parts.append("NO-GO for Study A: G1 failed again after the revision.")
        if not states["G4"] and "G4" in failed_before:
            parts.append("NO-GO for Study B: G4 failed again after the revision.")
        if not states["G2"] and "G2" in failed_before:
            parts.append("G2 failed again after the revision: " + CUT_ORDER)
        for k in ("G1", "G2", "G4"):
            if not states[k] and k not in failed_before:
                parts.append(
                    f"{k} failed for the first time on the re-pilot; Part 6 permits one revision only and does not "
                    "say what follows (question for the group)."
                )
        if states["G1"] and states["G2"] and not states["G3"]:
            parts.append(G3_NO_GO)
        elif not states["G3"] and not study_a_nogo:  # moot once Study A is a no-go
            parts.append(f"{g3_open} on the re-pilot" if g3_open else "G3 also failed on the re-pilot")
            parts[-1] += "; Part 6 decides G3 only when G1 and G2 hold (question for the group)."
        return " ".join(parts)
    revisable = [k for k in failed if k in ("G1", "G2", "G4")]
    if revisable:
        parts.append(
            "REVISE ONCE for " + ", ".join(revisable) + ": one revision of T, d, W, the tolerance, or Study B's "
            "conditioning, recorded in Part 9, then a re-pilot of the same size; a second failure of the same "
            "condition is a no-go for the study it concerns."
        )
    if states["G1"] and states["G2"] and not states["G3"]:
        parts.append(G3_NO_GO)
    elif not states["G3"]:
        head = g3_open or "G3 also failed"
        parts.append(head + "; it is read again after the revision of " + ", ".join(revisable) + ".")
    return " ".join(parts)


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def measured_concurrency(intervals: Sequence[tuple[datetime, datetime]]) -> dict[str, float]:
    """Runs sustained concurrently, from the pilot runs' own start and finish times (Table 8.1 [P]).

    Returns the maximum overlap and the time-weighted mean number of runs in progress.
    """
    if not intervals:
        return {"max": 0.0, "time_weighted_mean": 0.0}
    # a finish sorts before a start at the same instant (-1 < 1), so back-to-back runs do not overlap
    events = sorted([(a, 1) for a, _ in intervals] + [(b, -1) for _, b in intervals])
    level = peak = 0
    area = 0.0
    last = events[0][0]
    for when, delta in events:
        area += level * (when - last).total_seconds()
        level += delta
        peak = max(peak, level)
        last = when
    span = (max(b for _, b in intervals) - min(a for a, _ in intervals)).total_seconds()
    return {"max": float(peak), "time_weighted_mean": area / span if span > 0 else float(peak)}


def report(
    rows: Sequence[Mapping[str, Any]],
    *,
    hours_per_run: Sequence[float],
    concurrent_runs: int,
    machine_hours_allocated: float | None,
    allocation_ok: bool,
    allocation_note: str,
    unconstrained_cost: float | None,
    moderate: Mapping[str, Any] | None,
    failed_before: Iterable[str] = (),
    measured: Mapping[str, float] | None = None,
    unconstrained_selection_cost: float | None = None,
) -> dict[str, Any]:
    """All four conditions, the Part 6 branch they lead to, and the Table 8.1 values."""
    failed_before = sorted(frozenset(failed_before))
    g1 = g1_feasibility(rows)
    g2 = g2_throughput(hours_per_run, concurrent_runs, machine_hours_allocated, allocation_ok, allocation_note)
    g3 = g3_signal(rows)
    g4 = g4_conditioning(unconstrained_cost, (moderate or {}).get("mean_cost"),
                         unconstrained_selection_cost=unconstrained_selection_cost)
    ref = _pilot_rows(rows, REF_N)
    # Equation (1) gaps are taken from the measurement-set cost, so C_dynamics = measurement + gap. The mean
    # over seeds is reported only when every pilot seed of the reference arm contributes.
    ref_dyn = [
        float(r["measurement_cost"]) + float(r["gap_dynamics"])
        for r in ref if r.get("measurement_cost") is not None and r.get("gap_dynamics") is not None
    ]
    ref_dyn_mean = statistics.fmean(ref_dyn) if len(ref_dyn) == len(ref) == len(R.PILOT_SEEDS) else None
    table_8_1 = {
        "Wall-clock time per run (hours)": g2.values.get("hours_per_run_mean"),
        "Runs sustained concurrently": concurrent_runs,
        "Concurrency measured from the pilot runs (max; time-weighted mean)": (
            (measured or {}).get("max"), (measured or {}).get("time_weighted_mean")
        ),
        "Machine-hours allocated to the registered runs": machine_hours_allocated,
        "Seed-to-seed SD of final in-distribution cost, N = 0 arm": g1.values.get("reference_sd"),
        "Seed-to-seed SD of final in-distribution cost, N = 0.50 arm": (
            _sd(g1.values["late_costs"]) if "late_costs" in g1.values else None
        ),
        "Delta(0.50) under hazard relocation, three seeds (mean; U80)": (g3.values.get("delta_mean"), g3.values.get("U80")),
        "Reference arm's cost under the dynamics perturbation (mean over seeds)": ref_dyn_mean,
        "Unconstrained cost on the pinned software": unconstrained_cost,
        "Moderate arm's mean cost at each of its training budgets": (moderate or {}).get("mean_cost"),
        "Moderate arm's satisfaction rates at each of its training budgets": (moderate or {}).get("satisfaction"),
        "Seeds per primary-comparison arm actually used": "set by the surplus rule (python -m pilot surplus) at the go meeting",
    }
    return {
        "conditions": {
            c.name: {"passed": c.passed, "undecided_by": c.undecided_by, "values": c.values, "notes": c.notes}
            for c in (g1, g2, g3, g4)
        },
        "part6_reading": part6_reading(g1, g2, g3, g4, failed_before=failed_before),
        "repilot": bool(failed_before),
        "failed_before": failed_before,
        "table_8_1": table_8_1,
    }


GO_CONDITION_HEADING = "Go condition {name} (Part 6)"  # not Study B's hypothesis of the same number (Part 1.4)


def to_markdown(result: Mapping[str, Any]) -> str:
    """The report of ``report`` as Markdown: the Part 6 reading, each condition with its values and notes, Table 8.1.

    The headings name go conditions G1 to G4 as Part 6's ("Go condition Gk <name> (Part 6)", e.g. "Go condition G1
    feasibility (Part 6)"), apart from Study B's
    hypotheses G1 to G5 (Part 1.4; Table 9.1, Q-g-names); the identifiers are unchanged.
    """
    lines = ["# Pilot go or no-go report: go conditions G1 to G4 (Part 6; Table 8.1)", ""]
    lines.append(f"**Reading of Part 6:** {result['part6_reading']}")
    lines.append("")
    for name, cond in result["conditions"].items():
        if cond["passed"] is None and cond.get("undecided_by"):
            mark = f"UNDECIDED ({cond['undecided_by']})"
        else:
            mark = {True: "PASS", False: "FAIL", None: "NOT COMPUTABLE"}[cond["passed"]]
        lines.append(f"## {GO_CONDITION_HEADING.format(name=name)}: {mark}")
        for key, value in cond["values"].items():
            lines.append(f"- {key}: {json.dumps(value, default=_json_default)}")
        for note in cond["notes"]:
            lines.append(f"- note: {note}")
        lines.append("")
    lines.append("## Table 8.1 values")
    for key, value in result["table_8_1"].items():
        lines.append(f"- {key}: {json.dumps(value, default=_json_default)}")
    return "\n".join(lines) + "\n"


def write_report(result: Mapping[str, Any], out_dir: Path, stamp: str) -> tuple[Path, Path]:
    """Write the report under a new, timestamped name; an earlier report is never overwritten."""
    out_dir.mkdir(parents=True, exist_ok=True)
    js, md = out_dir / f"go_report-{stamp}.json", out_dir / f"go_report-{stamp}.md"
    # Render both first, then create each whole and only once (provenance.write_text_once: a temporary file,
    # fsynced, hard-linked; FileExistsError if it exists), the .json last: no report is ever overwritten, a
    # rendering error or a crash leaves no empty or partial file, and a readable .json always has its .md.
    js_text = json.dumps(result, indent=2, default=_json_default, sort_keys=True) + "\n"
    md_text = to_markdown(result)
    if js.exists():
        raise FileExistsError(str(js))
    provenance.write_text_once(md, md_text)
    try:
        provenance.write_text_once(js, js_text)
    except BaseException:
        md.unlink(missing_ok=True)
        raise
    return js, md

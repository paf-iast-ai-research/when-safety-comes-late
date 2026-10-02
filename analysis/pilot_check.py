"""Pilot mode: the Part 5.8 test run of the analysis on the pilot data.

Owner: Role 4, Analysis and results (Muhammad Abdullah).

Part 5.8: "The analysis script is committed to the repository before the first main-study run
completes, is run once on the pilot data to test it, and is not edited between the go decision and
the final analysis except to fix errors, each of which is recorded in Part 9."

The analysis never decides go: Part 6's four conditions are ``pilot/go_decision.py`` (Role 1), read by
the group at the go meeting. The test run recomputes, with the analysis's own code, the numbers of go
conditions G1 and G3 (Part 6: "G1, feasibility (Study A). Both pilot arms reach a final
in-distribution cost ... of at most d + 2.5 within T; the reference arm's seed-to-seed standard
deviation of that cost is at most 10 ...; and the two arms can be matched under the tolerance of Part
4"; "G3, signal (Study A). The pilot's mean Δ(0.50) under hazard relocation over the three seeds
is at least 5 cost units, or its upper bound U₈₀ of equation (12) is at least 5") and compares them
with ``pilot.go_decision.g1_feasibility`` and ``g3_signal`` on the same rows: a different number is an
error in one of the two (the purpose of the test run). The comparison is of numbers; the go
conditions' pass/fail readings stay the go report's. Table 8.1 values the analysis can recompute are
reported beside them. A test run in which not one G1 or G3 quantity could be compared with the go
report (for example a crashed pilot run whose replacement has not run: the go report needs three
seeds per arm) tested nothing: ``compared`` is 0, the report says so, and ``python -m analysis``
exits 4 after writing it, never 0.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Mapping

from configs import registered as R

from analysis import matching, questions, stats
from analysis.data import Dataset

REF_N, LATE_N = R.PILOT_STUDY_A_ONSET_FRACTIONS  # Part 3.6: the pilot arms N = 0 and N = 0.50
# Both the relative and the absolute tolerance of a comparison: the two implementations use the same formulas;
# only summation order may differ.
TOL = 1e-12

G1_KEYS = ("reference_cost_mean", "late_cost_mean", "reference_sd", "match_difference", "both_arms_within_budget",
           "arms_matched", "reference_sd_ok", "reference_final_checkpoint_sd")
# The go report's G1 flags are its own clauses, each compared with the analysis's rule of the same clause:
# both_arms_within_budget is Part 6 G1's "Both pilot arms reach a final in-distribution cost ... of at most d + 2.5"
# (each arm's mean, Q-g1-level; rule 3's comparison applied to each arm); arms_matched is its "the two arms can be
# matched under the tolerance of Part 4" (|difference| <= 2.5, rule 5's first sentence alone); reference_sd_ok is its
# "standard deviation ... at most 10" (rule 5's second), with "the same statistic for the final checkpoints reported
# beside it" (reference_final_checkpoint_sd). Rule 3 (reference cost <= d + 2.5) and the full rule set
# (matched_by_rules) are printed beside them.
# U80 in the paired form that decides G3, U80_pooled the descriptive figure beside it (Q-g3-pairing; ``g3_numbers``).
G3_KEYS = ("delta_mean", "U80", "U80_pooled")


def _pilot_arms(dataset: Dataset) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = [r for r in dataset.rows if r.get("study") == "A" and r.get("task") == R.PILOT_STUDY_A_TASK
            and r.get("completed") and r.get("treatment") is None and r.get("controller_variant") is None]
    ref = [r for r in rows if r["N"] == REF_N]
    late = [r for r in rows if r["N"] == LATE_N and r["onset_shape"] == R.PILOT_STUDY_A_SHAPE
            and r["step_matching"] == R.PILOT_STUDY_A_CONTROL]
    return sorted(ref, key=lambda r: r["seed"]), sorted(late, key=lambda r: r["seed"])


def g1_numbers(dataset: Dataset) -> dict[str, Any]:
    """Rules 2 to 5 on the pilot arms (the analysis's matching code): the numbers behind go condition G1."""
    ref, late = _pilot_arms(dataset)
    out: dict[str, Any] = {"reference_seeds": [r["seed"] for r in ref], "late_seeds": [r["seed"] for r in late]}
    if not ref or not late or not all(matching.usable_cost(r.get(matching.COST_FIELD)) for r in ref + late):
        out["computable"] = False
        return out
    result = matching.apply_rules(ref + late).get(R.PILOT_STUDY_A_TASK)
    late_arm = result.arm(matching.arm_id(late[0])) if result is not None else None
    late_cost = None if late_arm is None else late_arm.matched_cost
    ref_cost = None if result is None else result.reference_cost
    limit = R.COST_LIMIT + R.G1_COST_MARGIN  # Part 6 G1: "at most d + 2.5"
    out.update(
        computable=result is not None and ref_cost is not None and late_arm is not None,
        reference_cost_mean=ref_cost,
        late_cost_mean=late_cost,
        reference_sd=None if result is None else result.reference_sd,
        match_difference=None if late_arm is None else late_arm.difference,
        # G1's first clause on each arm's mean, compared as rule 3 compares (Q-threshold-arithmetic)
        both_arms_within_budget=(None if late_cost is None or ref_cost is None
                                 else matching.PROPOSED_ARITHMETIC.at_most(ref_cost, limit)
                                 and matching.PROPOSED_ARITHMETIC.at_most(late_cost, limit)),
        # the tolerance clause alone, as the go report's arms_matched (rule 5, first sentence)
        arms_matched=(None if late_cost is None or ref_cost is None
                      else matching.within_tolerance(late_cost, ref_cost, R.MATCH_TOLERANCE)),
        reference_sd_ok=None if result is None else result.sd_ok,
        matched_by_rules=None if late_arm is None else late_arm.matched,  # rules 3 and 5 together (rule 6)
        rule3_reference_within_budget=None if result is None else result.budget_ok,
        late_sd=matching.reference_sd([float(r[matching.COST_FIELD]) for r in late]),
    )
    finals = [r.get("final_cost") for r in ref]
    out["reference_final_checkpoint_sd"] = (matching.reference_sd([float(x) for x in finals])
                                            if all(matching.usable_cost(x) for x in finals) else None)
    return out


def g3_numbers(dataset: Dataset) -> dict[str, Any]:
    """Delta(0.50) under hazard relocation over the pilot seeds (equation 2), U80 (equation 12, paired) and the pooled
    U80 (descriptive).

    U80 is the paired form, as the go report decides G3 (Q-g3-pairing, answered in Table 9.1): s is the SD of the
    three per-seed differences gap_hazard(N = 0.50) - gap_hazard(N = 0), paired by seed id; where a replacement
    seed leaves fewer than three seed-matched pairs, the runs left unpaired are paired in increasing seed order
    (``stats.g3_pairs``) and equation (12) is applied unchanged. The pairing used is reported (``pairs``, late
    seed then reference seed; ``pairing``). U80_pooled is a descriptive figure. U80 needs three runs per arm, as
    Part 6 G3 ("over the three seeds") and the go report do. The Part 5.3 estimates are computed by ``e2_rows``.
    """
    ref, late = _pilot_arms(dataset)
    out: dict[str, Any] = {}
    ref_gaps = {r["seed"]: r.get("gap_hazard") for r in ref}
    late_gaps = {r["seed"]: r.get("gap_hazard") for r in late}
    if not ref_gaps or not late_gaps or None in ref_gaps.values() or None in late_gaps.values():
        out["computable"] = False
        return out
    out["computable"] = True
    out["delta_mean"] = statistics.fmean(late_gaps.values()) - statistics.fmean(ref_gaps.values())
    pairs = stats.g3_pairs(late_gaps, ref_gaps)
    diffs = [late_gaps[a] - ref_gaps[b] for a, b in pairs]
    out["paired_seeds"] = [a for a, b in pairs if a == b]  # the seed-matched pairs
    out["pairs"], out["paired_differences"] = pairs, diffs
    out["pairing"] = ("by seed id" if len(out["paired_seeds"]) == len(pairs) else
                      "by seed id, then the runs left unpaired in increasing seed order (Q-g3-pairing)")
    n = len(R.PILOT_SEEDS)
    out["U80"] = stats.u80(diffs) if len(late_gaps) == len(ref_gaps) == len(pairs) == n else None
    if len(ref_gaps) >= 2 and len(late_gaps) >= 2:
        out["U80_pooled"] = stats.u80_pooled(list(late_gaps.values()), list(ref_gaps.values()))
    out["u80_t_quantile_registered"] = R.U80_T_QUANTILE
    out["u80_t_quantile_computed"] = stats.u80_t_quantile(n)
    return out


def e2_rows(dataset: Dataset) -> list[dict[str, Any]]:
    """Part 5.3's estimates of Delta(0.50) under hazard and dynamics with the pilot's seeds (Welch, bootstrap, d, paired)."""
    ref, late = _pilot_arms(dataset)
    rows = []
    for condition in R.PILOT_BATTERY:
        x = {r["seed"]: r.get(f"gap_{condition}") for r in late}
        y = {r["seed"]: r.get(f"gap_{condition}") for r in ref}
        base = {"contrast": f"Delta(0.50) under {condition}", "condition": condition}
        if len(x) < 2 or len(y) < 2 or None in x.values() or None in y.values():
            rows.append({**base, "note": "not computable (missing gaps or fewer than two seeds)"})
            continue
        est = stats.estimate_two_arms(x, y, analysis_id=f"pilot|Delta050|{condition}")
        rows.append({**base, **est.row()})
    return rows


def table_8_1(dataset: Dataset, g1: Mapping[str, Any], g3: Mapping[str, Any]) -> dict[str, Any]:
    """Table 8.1 values the analysis recomputes (the go report is the record), under the go report's conditions: the
    SDs and Delta(0.50) only when each arm has its ``len(R.PILOT_SEEDS)`` completed runs, and the mean over seeds
    under the dynamics perturbation only when every pilot seed of the reference arm contributes; otherwise None
    (a missing field is never a silent drop), with the completed runs per arm beside them."""
    ref, late = _pilot_arms(dataset)
    n = len(R.PILOT_SEEDS)
    full = len(ref) == len(late) == n
    dyn = [float(r["measurement_cost"]) + float(r["gap_dynamics"]) for r in ref
           if r.get("measurement_cost") is not None and r.get("gap_dynamics") is not None]
    return {
        "Seed-to-seed SD of final in-distribution cost, N = 0 arm": g1.get("reference_sd") if full else None,
        "Seed-to-seed SD of final in-distribution cost, N = 0.50 arm": g1.get("late_sd") if full else None,
        "Delta(0.50) under hazard relocation, three seeds (mean; U80)": (
            (g3.get("delta_mean"), g3.get("U80")) if full else (None, None)),
        "Reference arm's cost under the dynamics perturbation (mean over seeds)": (
            statistics.fmean(dyn) if len(dyn) == len(ref) == n else None),
        "Completed pilot runs (N = 0 arm; N = 0.50 arm)": (len(ref), len(late)),
    }


def _same(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    return math.isclose(float(a), float(b), rel_tol=TOL, abs_tol=TOL)


def compare_with_go_report(dataset: Dataset, g1: Mapping[str, Any], g3: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    """Rows (quantity, analysis, go report, agree) and the list of disagreements.

    Each flag is compared with the analysis's computation of the SAME clause. Both sides compare G1's
    clauses after the same rounding (Q-threshold-arithmetic: pilot.go_decision THRESHOLD_DECIMALS and
    SD_DECIMALS, as analysis.matching), so every difference is a mismatch, an error in one of the two (Part 5.8: the
    analysis "is run once on the pilot data to test it"); no flag is excused as rounding.
    """
    try:
        from pilot import go_decision
    except ImportError as exc:  # pragma: no cover - the pipeline is part of the repository
        return [{"quantity": "go report", "note": f"pilot.go_decision unavailable: {exc}"}], []
    rows, mismatches = [], []
    for name, mine, cond in (("G1", g1, go_decision.g1_feasibility(dataset.rows)),
                             ("G3", g3, go_decision.g3_signal(dataset.rows))):
        # the open questions these numbers rest on (analysis.questions: pilot-G1, pilot-G3)
        provisional = ", ".join(questions.provisional_on(questions.PILOT_ENTRIES[name]))
        for key in (G1_KEYS if name == "G1" else G3_KEYS):
            theirs: Any = cond.values.get(key)
            ours = mine.get(key)
            if key not in cond.values:
                rows.append({"condition": name, "quantity": key, "analysis": ours, "go_report": None,
                             "agree": None, "note": "not computed by the go report", "provisional_on": provisional})
                continue
            agree = _same(ours, theirs)
            if not agree:
                mismatches.append(f"go condition {name}: {key} is {ours!r} in the analysis and {theirs!r} in the go report")
            rows.append({"condition": name, "quantity": key, "analysis": ours, "go_report": theirs,
                         "agree": agree, "note": "", "provisional_on": provisional})
    return rows, mismatches


def run(dataset: Dataset) -> dict[str, Any]:
    """Everything pilot mode adds to the report."""
    g1, g3 = g1_numbers(dataset), g3_numbers(dataset)
    consistency, mismatches = compare_with_go_report(dataset, g1, g3)
    compared = sum(1 for row in consistency if row.get("agree") is not None)
    return {
        "g1_numbers": g1, "g3_numbers": g3, "estimates": e2_rows(dataset), "table_8_1": table_8_1(dataset, g1, g3),
        "consistency": consistency, "mismatches": mismatches, "compared": compared,
        "note": ("Part 5.8 test run on the pilot data. The analysis never decides go; the go conditions are read "
                 "from the go report (python -m pilot go)."
                 + ("" if compared else " NOTHING WAS COMPARED: the go report computed no G1 or G3 quantity on these "
                                        "rows, so this run did not test the analysis against it.")),
    }

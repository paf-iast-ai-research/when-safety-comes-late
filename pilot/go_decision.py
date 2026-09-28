"""Go or no-go rule (Part 6) and the pilot's empirical values (Part 8, Table 8.1).

"The decision is taken by the group at the go meeting, from the pilot's results and nothing else."
This module computes the four conditions exactly as registered and states which branch of Part 6
they lead to; it does not take the decision.

G1 feasibility (Study A): both pilot arms' final in-distribution cost <= d + 2.5 (arm means,
   Q-g1-level); the reference arm's seed-to-seed SD of that cost <= 10 (final-checkpoint SD
   reported beside it); the two arms match within the tolerance of Part 4 (2.5).
G2 throughput: wall-clock per run / concurrent runs x 484 <= allocated machine-hours, with the
   allocation committed before throughput is read.
G3 signal (Study A): mean Delta(0.50) under hazard relocation over the three seeds >= 5, or its
   upper bound U80 (equation 12) >= 5.
G4 conditioning (Study B): every budget of Table 2.5 lies below the pilot's unconstrained cost; the
   Moderate arm's mean costs at its three training budgets are ordered with the budgets and each
   at most its budget + 2.5.
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

REF_N, LATE_N = R.PILOT_STUDY_A_ONSET_FRACTIONS  # Part 3.6: the pilot arms N = 0 and N = 0.50


@dataclass
class Condition:
    name: str
    passed: bool | None = None  # None: not computable from the data given, or undecided (undecided_by)
    values: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    # The PENDING key under which the group must decide this condition: its readings disagree while the
    # key is open; for Q-g3-pairing also when fewer than three pairs exist, whether answered or not.
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


COST_FIELD = "measurement_cost"  # Table 2.1: the measurement set gives the reported in-distribution cost


def g1_feasibility(rows: Sequence[Mapping[str, Any]]) -> Condition:
    """Part 6 G1 on the final in-distribution cost: the measurement-set cost of the selected checkpoint.

    Table 2.1 reserves the measurement set for the reported in-distribution cost; choosing and
    measuring on the same (selection) episodes "would bias the reported cost of the chosen
    checkpoint downward". The field is fixed here, not an option; the selection-set means are
    reported beside it and never decide. "Both pilot arms reach ... at most d + 2.5" is read on each
    arm's mean (proposed reading, Q-g1-level); while that key is open, G1 is left undecided if the
    per-seed reading would change the result.
    """
    ref, late = _pilot_rows(rows, REF_N), _pilot_rows(rows, LATE_N)
    cond = Condition("G1 feasibility")
    cond.values["cost_field"] = COST_FIELD
    missing = [r["run_id"] for r in ref + late if r.get(COST_FIELD) is None]
    n = len(R.PILOT_SEEDS)
    if len(ref) != n or len(late) != n or missing:
        cond.passed = None
        cond.notes.append(
            f"not computable: Part 6 G1 needs {n} completed runs per arm (reference {len(ref)}, late {len(late)}); "
            f"missing {COST_FIELD} in {missing}"
        )
        return cond
    ref_costs = [float(r[COST_FIELD]) for r in ref]
    late_costs = [float(r[COST_FIELD]) for r in late]
    ref_mean, late_mean = statistics.fmean(ref_costs), statistics.fmean(late_costs)
    ref_sd = _sd(ref_costs)
    finals = [r.get("final_cost") for r in ref]
    ref_final_sd = _sd([float(x) for x in finals]) if all(x is not None for x in finals) else None
    limit = R.COST_LIMIT + R.G1_COST_MARGIN
    a = ref_mean <= limit and late_mean <= limit
    a_every_seed = all(c <= limit for c in ref_costs + late_costs)  # the per-seed reading (Q-g1-level)
    b = ref_sd <= R.G1_REFERENCE_SD_MAX
    c = abs(late_mean - ref_mean) <= R.MATCH_TOLERANCE
    cond.values.update(
        reference_seeds=[r["seed"] for r in ref], late_seeds=[r["seed"] for r in late],
        reference_costs=ref_costs, late_costs=late_costs,
        reference_cost_mean=ref_mean, late_cost_mean=late_mean, cost_limit_plus_margin=limit,
        reference_sd=ref_sd, reference_final_checkpoint_sd=ref_final_sd,
        match_difference=late_mean - ref_mean, tolerance=R.MATCH_TOLERANCE,
        both_arms_within_budget=a, every_seed_within_budget=a_every_seed, reference_sd_ok=b, arms_matched=c,
        selection_set_means_not_decisive=[
            statistics.fmean(float(r["selection_cost_at_match"]) for r in arm) if all(r.get("selection_cost_at_match") is not None for r in arm) else None
            for arm in (ref, late)
        ],
    )
    cond.notes.append(
        "the budget clause is read on each arm's mean over seeds (proposed reading, Q-g1-level); "
        "every_seed_within_budget is reported beside it"
    )
    if a != a_every_seed and b and c and R.is_open("Q-g1-level"):
        cond.passed = None
        cond.undecided_by = "Q-g1-level"
        cond.notes.append(
            "the arm-mean and per-seed readings of Part 6 G1 disagree; the group must answer Q-g1-level before G1 is read"
        )
    else:
        cond.passed = bool(a and b and c)
    return cond


def u80(differences: Sequence[float]) -> float:
    """Equation (12): U80 = mean + 1.886 s / sqrt(3), with s the SD of the three paired differences."""
    n = len(R.PILOT_SEEDS)
    if len(differences) != n:
        raise ValueError("equation (12) is registered for exactly three seeds")
    return statistics.fmean(differences) + R.U80_T_QUANTILE * statistics.stdev(differences) / math.sqrt(n)


def g3_signal(rows: Sequence[Mapping[str, Any]]) -> Condition:
    """Part 6 G3 on the hazard-relocation gap (equation 2: Delta(0.50) = gap(0.50) - gap(0))."""
    ref = {r["seed"]: r for r in _pilot_rows(rows, REF_N)}
    late = {r["seed"]: r for r in _pilot_rows(rows, LATE_N)}
    cond = Condition("G3 signal")
    missing = [r["run_id"] for r in list(ref.values()) + list(late.values()) if r.get("gap_hazard") is None]
    n = len(R.PILOT_SEEDS)
    if len(ref) != n or len(late) != n or missing:
        cond.passed = None
        cond.notes.append(
            f"not computable: Part 6 G3 is over {n} seeds per arm (reference {len(ref)}, late {len(late)}); "
            f"missing gap_hazard in {missing}"
        )
        return cond
    delta_mean = statistics.fmean(float(r["gap_hazard"]) for r in late.values()) - statistics.fmean(
        float(r["gap_hazard"]) for r in ref.values()
    )
    cond.values["delta_mean"] = delta_mean
    common = sorted(set(ref) & set(late))
    diffs = [float(late[s]["gap_hazard"]) - float(ref[s]["gap_hazard"]) for s in common]
    cond.values.update(paired_seeds=common, paired_differences=diffs, minimum_effect=R.G3_SIGNAL_MIN)
    ref_gaps = [float(r["gap_hazard"]) for r in ref.values()]
    late_gaps = [float(r["gap_hazard"]) for r in late.values()]
    pooled_s = math.sqrt((statistics.variance(ref_gaps) + statistics.variance(late_gaps)) / 2)
    pooled = delta_mean + R.U80_T_QUANTILE * pooled_s / math.sqrt(n)
    if len(common) != n:
        cond.values.update(U80=None, U80_pooled=pooled)
        if delta_mean >= R.G3_SIGNAL_MIN:
            cond.passed = True
            cond.notes.append("decided by the mean clause; U80 is not needed")
        else:
            # Q-g3-pairing's proposal: with fewer than three pairs the group decides, whether or not
            # the key is answered; U80_pooled is reported beside it.
            cond.passed = None
            cond.undecided_by = "Q-g3-pairing"
        cond.notes.append(
            f"the two arms do not share {n} seeds (a replacement seed), so the paired U80 of equation (12) is "
            "undefined; by Q-g3-pairing the group decides G3 unless the mean clause does, with U80_pooled reported beside it"
        )
        return cond
    bound = u80(diffs)
    cond.values.update(U80=bound, U80_pooled=pooled)
    paired_pass = delta_mean >= R.G3_SIGNAL_MIN or bound >= R.G3_SIGNAL_MIN
    pooled_pass = delta_mean >= R.G3_SIGNAL_MIN or pooled >= R.G3_SIGNAL_MIN
    cond.notes.append("U80 uses the paired form over seeds (proposed reading, Q-g3-pairing); U80_pooled is reported beside it")
    if paired_pass != pooled_pass and R.is_open("Q-g3-pairing"):
        cond.passed = None
        cond.undecided_by = "Q-g3-pairing"
        cond.notes.append(
            "the paired and pooled readings of equation (12) disagree on G3; the group must answer Q-g3-pairing before G3 is read"
        )
    else:
        cond.passed = bool(paired_pass)
    return cond


def all_study_b_budgets() -> list[float]:
    """Every budget of Table 2.5: all training levels, the continuous range and the unseen budgets."""
    budgets = set(R.UNSEEN_BUDGETS) | set(R.CONTINUOUS_RANGE)
    for levels in R.STUDY_B_ARMS.values():
        if levels:
            budgets |= set(levels)
    return sorted(budgets)


def g4_conditioning(unconstrained_cost: float | None, moderate_mean_costs: Mapping[float, float] | None) -> Condition:
    cond = Condition("G4 conditioning")
    if unconstrained_cost is None or not moderate_mean_costs:
        cond.passed = None
        cond.notes.append("not computable: the unconstrained run's cost or the Moderate arm's costs are missing")
        return cond
    budgets = all_study_b_budgets()
    below = max(budgets) < float(unconstrained_cost)
    levels = sorted(float(b) for b in R.STUDY_B_ARMS["Moderate"])
    costs = {float(k): float(v) for k, v in moderate_mean_costs.items()}
    if sorted(costs) != levels:
        cond.passed = None
        cond.notes.append(f"Moderate costs must be given for exactly {levels}")
        return cond
    ordered = all(costs[a] < costs[b] for a, b in zip(levels, levels[1:]))
    within = {b: costs[b] <= b + R.G4_COST_MARGIN for b in levels}
    two_sided = {b: abs(costs[b] - b) <= R.G4_COST_MARGIN for b in levels}  # Part 3.6 wording (Q-g4-level)
    cond.values.update(
        each_within_2_5_two_sided_part_3_6=two_sided,
        unconstrained_cost=float(unconstrained_cost), largest_budget=max(budgets),
        every_budget_below_unconstrained=below, moderate_mean_costs=costs,
        ordered_with_budgets=ordered, each_within_budget_plus_margin=within,
    )
    cond.passed = bool(below and ordered and all(within.values()))
    if cond.passed and not all(two_sided.values()):
        cond.notes.append("passes by Part 6 ('at most its budget plus 2.5') but not by Part 3.6 ('within 2.5 of its budget'); see Q-g4-level")
    return cond


def g2_throughput(
    hours_per_run: Sequence[float],
    concurrent_runs: int,
    machine_hours_allocated: float | None,
    allocation_ok: bool,
    allocation_note: str,
) -> Condition:
    from pilot import budget

    cond = Condition("G2 throughput")
    if len(hours_per_run) != R.PILOT_RUNS or machine_hours_allocated is None:
        cond.passed = None
        cond.notes.append(
            f"not computable: {len(hours_per_run)} of {R.PILOT_RUNS} pilot runs completed, "
            f"allocation {'missing' if machine_hours_allocated is None else 'present'}"
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
        cond.notes.append("the allocation was not fixed before throughput was read (Part 6 G2); the condition cannot be evaluated")
        return cond
    cond.passed = bool(required <= machine_hours_allocated)
    if corrected > machine_hours_allocated >= required:
        cond.notes.append("passes as registered but NOT with the corrected run-equivalent total; raise with the group before the go decision")
    return cond


CUT_ORDER = (
    "apply the cut order of Part 6.1, each step only if the previous is insufficient: (1) drop the PID "
    "check; (2) drop SafetyCarGoal1-v0 from every group of Study A; (3) shorten Study B's few-shot "
    "continuations from 1,000,000 to 200,000 steps, keeping only the first horizon; (4) restrict mediation treatments and controller variants to N = 0.50; "
    "(5) drop N = 0.10 from the main sweep. Never cut the number of seeds, Study B's seven zero-shot arms, or "
    "the N = 0 and N = 0.50 arms under both controls on SafetyPointGoal1-v0 and SafetyPointButton1-v0."
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
    missing = [k for k, c in conds.items() if c.passed is None and not c.undecided_by]
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
    g3_undecided = states["G3"] is None  # only reached when g3_moot
    if all(states.values()):
        return ("GO: enter the measured values in Part 8, commit the amended document, and freeze Parts 1, 4, 5 and 6.")
    failed = [k for k, v in states.items() if not v]
    parts = []
    if failed_before:
        if not states["G1"] and "G1" in failed_before:
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
            parts.append(
                "NO-GO for Study A: any timing effect is below the minimum effect of interest in the pilot setting; "
                "write the pilot up as a negative pilot with its bound (U80), and Study B becomes the main study."
            )
        elif not states["G3"] and not (not states["G1"] and "G1" in failed_before):  # moot once Study A is a no-go
            parts.append(
                f"G3 is undecided ({conds['G3'].undecided_by}) on the re-pilot" if g3_undecided else "G3 also failed on the re-pilot"
            )
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
        parts.append(
            "NO-GO for Study A: any timing effect is below the minimum effect of interest in the pilot setting; "
            "write the pilot up as a negative pilot with its bound (U80), and Study B becomes the main study."
        )
    elif not states["G3"]:
        head = f"G3 is undecided ({conds['G3'].undecided_by})" if g3_undecided else "G3 also failed"
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
    events = sorted([(a, 1) for a, _ in intervals] + [(b, -1) for _, b in intervals], key=lambda e: (e[0], e[1]))
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
) -> dict[str, Any]:
    """All four conditions, the Part 6 branch they lead to, and the Table 8.1 values."""
    failed_before = sorted(frozenset(failed_before))
    g1 = g1_feasibility(rows)
    g2 = g2_throughput(hours_per_run, concurrent_runs, machine_hours_allocated, allocation_ok, allocation_note)
    g3 = g3_signal(rows)
    g4 = g4_conditioning(unconstrained_cost, (moderate or {}).get("mean_cost"))
    ref = _pilot_rows(rows, REF_N)
    # Equation (1) gaps are taken from the measurement-set cost, so C_dynamics = measurement + gap.
    ref_dyn = [
        float(r["measurement_cost"]) + float(r["gap_dynamics"])
        for r in ref if r.get("measurement_cost") is not None and r.get("gap_dynamics") is not None
    ]
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
        "Reference arm's cost under the dynamics perturbation (mean over seeds)": statistics.fmean(ref_dyn) if ref_dyn else None,
        "Unconstrained cost on the pinned software": unconstrained_cost,
        "Moderate arm's mean cost at each training budget": (moderate or {}).get("mean_cost"),
        "Moderate arm's satisfaction rates": (moderate or {}).get("satisfaction"),
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


def to_markdown(result: Mapping[str, Any]) -> str:
    lines = ["# Pilot go or no-go report (Part 6; Table 8.1)", ""]
    lines.append(f"**Reading of Part 6:** {result['part6_reading']}")
    lines.append("")
    for name, cond in result["conditions"].items():
        if cond["passed"] is None and cond.get("undecided_by"):
            mark = f"UNDECIDED ({cond['undecided_by']})"
        else:
            mark = {True: "PASS", False: "FAIL", None: "NOT COMPUTABLE"}[cond["passed"]]
        lines.append(f"## {name}: {mark}")
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
    # Render both first, then create both with "x" (fails if either exists): no report is ever
    # overwritten, and a rendering error cannot leave one file without the other.
    js_text = json.dumps(result, indent=2, default=_json_default, sort_keys=True) + "\n"
    md_text = to_markdown(result)
    with open(js, "x", encoding="utf-8") as fj:
        try:
            with open(md, "x", encoding="utf-8") as fm:
                fj.write(js_text)
                fm.write(md_text)
        except FileExistsError:
            fj.close()
            js.unlink()
            raise
    return js, md

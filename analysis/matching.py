"""Part 4.1 comparison rule: checkpoint selection (rule 1) and arm matching (rules 2 to 7).

Owner: Role 4, Analysis and results (Muhammad Abdullah). Implements Part 4.1 rules 1 to 7 (p. 14),
Table 2.1 ("Two disjoint sets of 100 episodes ... a selection set ... used only to choose one of
them in Part 4, and a measurement set, run on the chosen checkpoint only and used for the reported
in-distribution cost") and Table 8.2 ("Closest to d = 25 among the last ten checkpoints, every arm
alike; 2.5; infeasible above SD 10").

Imports: the standard library and ``configs.registered`` only. ``select_checkpoint`` is pipeline
contract 4 (pilot/contracts.py): ``python -m pilot enrich select`` imports this module and then
refuses any imported repository file that the commit does not hold
(``pilot.provenance.unverified_imported_code``); numpy, pandas or ``pilot.*`` here would widen that
set (``pilot.launch`` also sets thread variables on import).

The rules, as the pre-registration numbers them (p. 14):
  1  Selection: "the checkpoint is chosen from the run's last ten checkpoints ... as the one whose
     selection-set cost is closest to the budget d = 25. The target is the budget, not another
     arm's cost" -> ``select_checkpoint``.
  2  "For each task and step-matching control, the reference cost is the mean over the five seeds
     of the N = 0 arm's selected checkpoints" -> ``is_reference_row``, ``reference_cost``. The N = 0
     arm "has no onset shape and no step-matching variant" (Part 3.2), so one N = 0 arm is the
     reference of both controls of its task (INFERENCE).
  3  "reference cost <= d + tolerance. If it does not, the task is reported as infeasible at d = 25
     and enters no comparison" -> ``budget_feasible``.
  4  "An arm's matched cost is the mean over its five seeds of the selected checkpoints' evaluation
     costs" -> ``matched_cost``; the evaluation set is the measurement set (Q-matched-cost-set, answered;
     ``COST_FIELD``).
  5  "matched if |matched cost - reference cost| <= 2.5 ... If the reference arm's own seed-to-seed
     standard deviation of selected-checkpoint cost exceeds 10 cost units in a task, matching in
     that task is reported as infeasible" -> ``within_tolerance``, ``spread_feasible``.
  6  "An arm that cannot be matched is reported with its cost and left out of the battery. It is
     not compared, and the tolerance is not widened for it" -> ``unmatched_status``.
  7  "repeated twice ... labelled exploratory: once with tolerance 5.0, and once with every arm's
     final checkpoint in place of the selected one" -> ``apply_rules`` accepts R.SENSITIVITY_TOLERANCE
     (``TOLERANCES``) and a ``cost_of`` for the final-checkpoint cost. Rule 7 (b) is the final-checkpoint
     estimand of Part 4.1.1 ("whatever the in-distribution cost"): no matching filter applies
     (analysis.study_a); ``apply_rules(cost_of=...)`` reports how the final checkpoints would match.

Analysis questions as answered in Table 9.1: Q-threshold-arithmetic (sample SD, infeasible iff SD > 10,
rules 3 and 5 inclusive, the means and their difference compared after rounding to 1e-4, the SD after
rounding to 1e-9; the other reading is ``ALTERNATIVE_ARITHMETIC``), Q-surplus-in-analysis (every
completed seed counts, registered, replacement and surplus), and the keys whose gates hold only while they are
open (configs.registered.PENDING): Q-tie-break (ties to the later checkpoint, distances compared after
rounding to 1e-4), Q-matched-cost-set (the measurement set), Q-arm-complete (``match_arms`` refuses an
arm below its seed target), Q-selection-window (a run whose total is off the 200,000-step grid is
selected on the ten checkpoints of an end-relative grid; ``selection_window``).
"""

from __future__ import annotations

import math
import numbers
import re
import statistics
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from configs import registered as R

TIE_DECIMALS = 4  # answered in Table 9.1 (Q-tie-break): rule 1 distances compared at 1e-4, as pilot.enrichment
# Q-threshold-arithmetic (answered in Table 9.1): rules 3 and 5 compare the means and their difference after rounding to
# 1e-4. Episode costs are integers (Safety-Gymnasium 0.4.1 builder.py:283-285 binarises the per-step cost),
# so 100-episode means are multiples of 0.01 and 1e-4 absorbs floating-point noise without moving any real
# boundary. The reference SD is a square root, not on that grid (costs 55.04, 34.19, 45.27, 32.03, 50.25 have
# a sample SD of 10.000004, which exceeds 10 but rounds to 10.0 at 1e-4): it is compared after rounding to
# SD_DECIMALS, at floating-point noise only.
THRESHOLD_DECIMALS = 4
SD_DECIMALS = 9
COST_FIELD = "measurement_cost"  # answered in Table 9.1 (Q-matched-cost-set): the measurement set, as go_decision
SELECTION_COST_FIELD = "selection_cost_at_match"  # the other reading of Q-matched-cost-set (reported beside)
TOLERANCES = (R.MATCH_TOLERANCE, R.SENSITIVITY_TOLERANCE)  # rule 5 and rule 7 (a); nothing else is allowed

STATUS_REFERENCE = "reference"
STATUS_MATCHED = "matched"
# Rule 6's unmatched arms, by the side of the reference their matched cost lies on. "A late arm that never reaches the
# budget" is not every arm above the reference: it is one whose matched cost fails rule 3's test, cost <= d + tolerance
# (Q-unmatched-reporting; analysis.study_a table A_unmatched, ``budget_feasible``).
STATUS_UNMATCHED_ABOVE = "unmatched_above"
STATUS_UNMATCHED_BELOW = "unmatched_below"
STATUS_INFEASIBLE_BUDGET = "task_infeasible_budget"  # rule 3
STATUS_INFEASIBLE_SD = "task_infeasible_sd"  # rule 5, second sentence
STATUS_INCOMPLETE = "incomplete"  # a cost is missing: nothing is decided, nothing is dropped
STATUSES = (STATUS_REFERENCE, STATUS_MATCHED, STATUS_UNMATCHED_ABOVE, STATUS_UNMATCHED_BELOW,
            STATUS_INFEASIBLE_BUDGET, STATUS_INFEASIBLE_SD, STATUS_INCOMPLETE)
UNMATCHED_STATUSES = (STATUS_UNMATCHED_ABOVE, STATUS_UNMATCHED_BELOW)  # rule 6: "It is not compared"

_POPULATION = re.compile(r"^(P\d*-)?[AB]-")


# ---------------------------------------------------------------------------
# Rule 1 (contract 4)
# ---------------------------------------------------------------------------


def selection_window(checkpoints: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """The records of Part 4.1 rule 1's window, "the run's last ten checkpoints (the final 2,000,000 steps; decision)".

    The run's total is its last checkpoint's step (Part 3.4: a checkpoint "at the end of training").
    The window is the records at total - k x 200,000, k = 0..9 (``R.SELECTION_WINDOW_CHECKPOINTS``
    steps). On the 200,000-step grid these are the last ten grid checkpoints; off the grid (the
    constrained-steps totals 11,120,000 and 13,340,000; the data control's 12,500,000) they are the
    answer of Q-selection-window (Table 9.1), a checkpoint grid relative to the end of training, which those runs
    save and ``pilot.contracts.selection_window`` evaluates. Records missing from those steps are
    absent from the result (``select_checkpoint`` then refuses), so the window never reaches back
    before the final 2,000,000 steps. Ascending by step; pure.
    """
    ordered = sorted(checkpoints, key=lambda c: int(c["step"]))
    if not ordered:
        return []
    total = int(ordered[-1]["step"])
    wanted = {total - k * R.CHECKPOINT_INTERVAL_STEPS for k in range(R.SELECTION_WINDOW_CHECKPOINTS)}
    return [c for c in ordered if int(c["step"]) in wanted]


def select_checkpoint(checkpoints: list[dict]) -> int:
    """Part 4.1 rule 1: the step of the last-ten checkpoint whose selection-set cost is closest to d.

    Pipeline contract 4 (pilot/contracts.py); ``pilot.enrichment.rule_one_step`` recomputes it and
    refuses any other answer. The window is the ``R.SELECTION_WINDOW_CHECKPOINTS`` steps total - k x
    200,000 of ``selection_window`` (for a total off the 200,000-step grid, the end-relative grid of
    Q-selection-window); the target is ``R.COST_LIMIT``, "not another arm's cost"; only
    ``selection_cost`` is read ("Selection uses the selection set and nothing else", First Tasks
    Role 4). A tie goes to the later checkpoint, with distances compared after rounding to
    ``TIE_DECIMALS`` (Q-tie-break), so costs 32.01 and 17.99 tie despite floating-point noise.

    Refuses (ValueError) fewer than ten checkpoints, a repeated or non-integer step, a window step
    with no checkpoint, and a window cost that is missing, not a number or not finite; a checkpoint
    is never skipped. Pure: the argument is not modified.
    """
    records = list(checkpoints)
    if len(records) < R.SELECTION_WINDOW_CHECKPOINTS:
        raise ValueError(f"rule 1 needs the run's last {R.SELECTION_WINDOW_CHECKPOINTS} checkpoints; "
                         f"got {len(records)}")
    steps = []
    for record in records:
        step = record.get("step") if isinstance(record, Mapping) else None
        if isinstance(step, bool) or not isinstance(step, numbers.Integral):
            raise ValueError(f"a checkpoint step must be an integer, got {step!r}")
        steps.append(int(step))
    if len(set(steps)) != len(steps):
        raise ValueError("a checkpoint step appears twice; rule 1 cannot tell the records apart")
    window = selection_window(records)
    if len(window) < R.SELECTION_WINDOW_CHECKPOINTS:
        raise ValueError(f"the selection window (the final {R.SELECTION_WINDOW_CHECKPOINTS} checkpoint steps; "
                         f"Q-selection-window) lacks checkpoints: it holds steps {[int(c['step']) for c in window]}")
    ranked = []
    for record in window:
        cost = record.get("selection_cost")
        if cost is None:
            raise ValueError(f"checkpoint {record['step']} of the selection window has no selection-set cost")
        if isinstance(cost, bool) or not isinstance(cost, numbers.Real):
            raise ValueError(f"the selection-set cost of checkpoint {record['step']} is not a number: {cost!r}")
        value = float(cost)
        if not math.isfinite(value):
            raise ValueError(f"the selection-set cost of checkpoint {record['step']} is not finite; "
                             "rule 1 cannot rank it")
        step = int(record["step"])
        ranked.append((round(abs(value - R.COST_LIMIT), TIE_DECIMALS), -step, step))
    return min(ranked)[2]


# ---------------------------------------------------------------------------
# Arithmetic of rules 3 and 5 (Q-threshold-arithmetic)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Arithmetic:
    """How rules 3 and 5 compare (Q-threshold-arithmetic).

    ``ddof``: 1 = sample SD (the answered reading; as pilot/go_decision.py's statistics.stdev), 0 = population SD.
    ``inclusive``: rules 3 and 5 use <= (the answered reading) or <. ``decimals``: round the means and their difference
    before comparing, or None; the SD is then rounded to ``SD_DECIMALS`` (noise only: it is not on the 0.01 grid).
    """

    ddof: int = 1
    inclusive: bool = True
    decimals: Optional[int] = THRESHOLD_DECIMALS

    def rounded(self, value: float) -> float:
        return value if self.decimals is None else round(value, self.decimals)

    def at_most(self, value: float, limit: float) -> bool:
        value = self.rounded(value)
        return value <= limit if self.inclusive else value < limit

    def sd_exceeds(self, sd: float, limit: float) -> bool:
        """Rule 5's SD clause: the SD is no mean of integer costs, so it is rounded at noise level only."""
        return (sd if self.decimals is None else round(sd, SD_DECIMALS)) > limit


# The answered reading; the name is kept from when it was a proposal, as analysis.verdict.PROPOSED's is.
PROPOSED_ARITHMETIC = Arithmetic()
ALTERNATIVE_ARITHMETIC = Arithmetic(ddof=0, inclusive=False, decimals=None)  # the other reading, beside


# ---------------------------------------------------------------------------
# Rules 2 to 7 as separate functions
# ---------------------------------------------------------------------------


def check_tolerance(tolerance: float) -> float:
    """Rule 5 ("The tolerance is not widened for any arm") and rule 7 (a): 2.5 or 5.0 only."""
    if tolerance not in TOLERANCES:
        raise ValueError(f"tolerance {tolerance!r} is not registered: rule 5 fixes {R.MATCH_TOLERANCE}, "
                         f"rule 7 repeats with {R.SENSITIVITY_TOLERANCE}; it is never widened for an arm")
    return float(tolerance)


def is_reference_row(row: Mapping[str, Any]) -> bool:
    """Rule 2: a run of the N = 0 arm (no onset shape, no step-matching variant, no treatment or controller)."""
    return (row.get("study") == "A" and row.get("N") == R.ONSET_FRACTIONS[0]
            and row.get("treatment") is None and row.get("controller_variant") is None)


def reference_cost(costs: Sequence[float]) -> float:
    """Rule 2: the reference cost is the mean over the N = 0 arm's seeds (every completed seed)."""
    if not costs:
        raise ValueError("rule 2 needs at least one reference seed")
    return statistics.fmean(costs)


def reference_sd(costs: Sequence[float], arithmetic: Arithmetic = PROPOSED_ARITHMETIC) -> Optional[float]:
    """Rule 5: the reference arm's seed-to-seed standard deviation (None with fewer than two seeds)."""
    if len(costs) < 2:
        return None
    return statistics.stdev(costs) if arithmetic.ddof == 1 else statistics.pstdev(costs)


def budget_feasible(ref_cost: float, tolerance: float, arithmetic: Arithmetic = PROPOSED_ARITHMETIC) -> bool:
    """Rule 3: "reference cost <= d + tolerance" (the rule 7 (a) repeat uses its tolerance here too)."""
    return arithmetic.at_most(ref_cost, R.COST_LIMIT + check_tolerance(tolerance))


def matched_cost(costs: Sequence[float]) -> float:
    """Rule 4: the mean over the arm's seeds of the selected checkpoints' evaluation costs."""
    if not costs:
        raise ValueError("rule 4 needs at least one seed")
    return statistics.fmean(costs)


def within_tolerance(arm_cost: float, ref_cost: float, tolerance: float,
                     arithmetic: Arithmetic = PROPOSED_ARITHMETIC) -> bool:
    """Rule 5: "matched if |matched cost - reference cost| <= 2.5" (5.0 in the rule 7 (a) repeat)."""
    return arithmetic.at_most(abs(arm_cost - ref_cost), check_tolerance(tolerance))


def spread_feasible(sd: float, arithmetic: Arithmetic = PROPOSED_ARITHMETIC) -> bool:
    """Rule 5: matching is infeasible if the reference SD "exceeds 10 cost units" (SD = 10 is feasible)."""
    return not arithmetic.sd_exceeds(sd, R.INFEASIBLE_REFERENCE_SD)


def unmatched_status(arm_cost: float, ref_cost: float) -> str:
    """Rule 6: an unmatched arm is reported with its cost, above or below the reference. Being above the reference
    is not "never reaching the budget": that is an unmatched arm whose matched cost fails rule 3's test of satisfying
    the budget (``budget_feasible``; Q-unmatched-reporting), a finding about learning, not about robustness."""
    return STATUS_UNMATCHED_ABOVE if arm_cost > ref_cost else STATUS_UNMATCHED_BELOW


# ---------------------------------------------------------------------------
# The rules applied to a ledger
# ---------------------------------------------------------------------------


def population(run_id: str) -> str:
    """The run_id prefix of the study population: '' (registered), 'P-' (pilot) or 'P1-' (re-pilot)."""
    match = _POPULATION.match(run_id)
    if not match:
        raise ValueError(f"unexpected run_id {run_id!r}")
    return match.group(1) or ""


def arm_id(row: Mapping[str, Any]) -> str:
    """The run_id without its seed (pilot.manifest.RunSpec.arm_id): the arm of Part 5.6's seed bookkeeping."""
    run_id, seed = str(row["run_id"]), int(row["seed"])
    suffix = f"-s{seed}"
    if not run_id.endswith(suffix):
        raise ValueError(f"run_id {run_id!r} does not end with its seed {suffix!r}")
    return run_id[: -len(suffix)]


@dataclass(frozen=True)
class ArmMatch:
    """One arm's result under rules 4 to 6 (or the reference under rules 2, 3 and 5)."""

    arm_id: str
    task: str
    arm: str
    N: Optional[float]
    onset_shape: Optional[str]
    step_matching: Optional[str]
    treatment: Optional[str]
    controller_variant: Optional[str]
    run_ids: tuple[str, ...]
    seeds: tuple[int, ...]
    costs: tuple[Optional[float], ...]
    matched_cost: Optional[float]
    difference: Optional[float]  # matched cost - reference cost
    matched: Optional[bool]  # None while incomplete
    status: str


@dataclass(frozen=True)
class TaskMatch:
    """Rules 2 to 6 for one task: the reference and every other Study A arm of the task."""

    task: str
    tolerance: float
    arithmetic: Arithmetic
    reference: Optional[ArmMatch]
    reference_cost: Optional[float]
    reference_sd: Optional[float]
    budget_ok: Optional[bool]  # rule 3
    sd_ok: Optional[bool]  # rule 5, second sentence
    feasible: Optional[bool]  # None while incomplete
    reason: str
    arms: tuple[ArmMatch, ...]

    def arm(self, arm_key: str) -> Optional[ArmMatch]:
        for arm in (self.reference, *self.arms):
            if arm is not None and arm.arm_id == arm_key:
                return arm
        return None


def _default_cost(row: Mapping[str, Any]) -> Any:
    """The ledger value as stored: ``usable_cost`` judges it before any conversion to float."""
    return row.get(COST_FIELD)


def usable_cost(value: Any) -> bool:
    """A cost rules 2 to 6 can read: a finite real number (never None, NaN or infinity).

    The frozen ``LedgerRow`` accepts NaN and infinity in a float field; such a cost is a data defect,
    not a finding, so it is treated like a missing one (the arm or the task is ``incomplete``) rather
    than read as "unmatched" (a NaN compares false with everything) or passed to ``statistics``.
    """
    if value is None or isinstance(value, bool) or not isinstance(value, numbers.Real):
        return False
    return math.isfinite(float(value))


def _study_a_rows(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """The completed Study A rows (Part 5.6: only runs that crash, do not complete their steps, or produce a
    non-finite loss or multiplier are excluded; ``completed`` is False for them)."""
    selected = [r for r in rows if r.get("study") == "A" and r.get("completed") is True]
    populations = {population(str(r["run_id"])) for r in selected}
    if len(populations) > 1:
        raise ValueError(f"rows of different study populations {sorted(populations)} cannot be matched together "
                         "(Part 3.6: 'The pilot's runs are not reused')")
    return selected


def _arm_result(arm_rows: Sequence[Mapping[str, Any]], costs: Sequence[Optional[float]], *, status: str,
                matched_cost_value: Optional[float] = None, difference: Optional[float] = None,
                matched: Optional[bool] = None) -> ArmMatch:
    first = arm_rows[0]
    return ArmMatch(
        arm_id=arm_id(first), task=str(first["task"]), arm=str(first["arm"]), N=first.get("N"),
        onset_shape=first.get("onset_shape"), step_matching=first.get("step_matching"),
        treatment=first.get("treatment"), controller_variant=first.get("controller_variant"),
        run_ids=tuple(str(r["run_id"]) for r in arm_rows), seeds=tuple(int(r["seed"]) for r in arm_rows),
        costs=tuple(costs), matched_cost=matched_cost_value, difference=difference, matched=matched, status=status,
    )


def apply_rules(
    rows: Iterable[Mapping[str, Any]],
    *,
    tolerance: float = R.MATCH_TOLERANCE,
    cost_of: Optional[Callable[[Mapping[str, Any]], Any]] = None,
    arithmetic: Arithmetic = PROPOSED_ARITHMETIC,
) -> dict[str, TaskMatch]:
    """Rules 2 to 6 on the completed Study A rows, per task; {task: TaskMatch}.

    ``cost_of(row)`` gives a run's selected-checkpoint evaluation cost (default: the measurement
    cost, Q-matched-cost-set; rule 7 (b) passes the final checkpoint's cost). Every completed seed
    counts (Q-surplus-in-analysis). A missing or non-finite cost (``usable_cost``) makes the task or
    the arm ``incomplete``; nothing is dropped (Part 5.6: "No run is excluded on the basis of its
    results"). Rule 3 is decided first: it does not need the reference SD, so a task whose reference
    cost exceeds d + tolerance is infeasible even with one reference seed. Arms are compared only
    within their task, never across tasks or controls (First Tasks Role 4, "Mistakes").
    """
    tolerance = check_tolerance(tolerance)
    cost_of = cost_of or _default_cost
    by_task: dict[str, dict[str, list[Mapping[str, Any]]]] = {}
    for row in sorted(_study_a_rows(rows), key=lambda r: (str(r["task"]), arm_id(r), int(r["seed"]))):
        by_task.setdefault(str(row["task"]), {}).setdefault(arm_id(row), []).append(row)
    out: dict[str, TaskMatch] = {}
    for task, arms in sorted(by_task.items()):
        for key, arm_rows in arms.items():
            seeds = [int(r["seed"]) for r in arm_rows]
            if len(set(seeds)) != len(seeds):
                raise ValueError(f"arm {key} has a seed twice; each seed is one run (Part 5.1)")
        ref_keys = [k for k, rs in arms.items() if is_reference_row(rs[0])]
        if len(ref_keys) > 1:
            raise ValueError(f"task {task} has more than one N = 0 arm: {ref_keys}")
        others = [(k, rs) for k, rs in sorted(arms.items()) if k not in ref_keys]
        other_costs = {k: [cost_of(r) for r in rs] for k, rs in others}
        if not ref_keys:
            out[task] = TaskMatch(task, tolerance, arithmetic, None, None, None, None, None, None,
                                  "no completed run of the N = 0 arm (rule 2)",
                                  tuple(_arm_result(rs, other_costs[k], status=STATUS_INCOMPLETE) for k, rs in others))
            continue
        ref_rows = arms[ref_keys[0]]
        ref_costs = [cost_of(r) for r in ref_rows]
        if not all(usable_cost(c) for c in ref_costs):
            missing = [str(r["run_id"]) for r, c in zip(ref_rows, ref_costs) if not usable_cost(c)]
            out[task] = TaskMatch(task, tolerance, arithmetic,
                                  _arm_result(ref_rows, ref_costs, status=STATUS_INCOMPLETE),
                                  None, None, None, None, None, f"reference cost missing or not finite for {missing}",
                                  tuple(_arm_result(rs, other_costs[k], status=STATUS_INCOMPLETE) for k, rs in others))
            continue
        ref_costs = [float(c) for c in ref_costs]
        ref_value = reference_cost(ref_costs)
        sd = reference_sd(ref_costs, arithmetic)
        budget_ok = budget_feasible(ref_value, tolerance, arithmetic)
        sd_ok = None if sd is None else spread_feasible(sd, arithmetic)
        if not budget_ok:
            feasible, reason = False, (f"rule 3: reference cost {ref_value:.4f} exceeds d + tolerance = "
                                       f"{R.COST_LIMIT + tolerance}; the task is infeasible at d = {R.COST_LIMIT:g}")
        elif sd_ok is None:
            feasible, reason = None, "the reference SD needs at least two seeds (rule 5)"
        elif not sd_ok:
            feasible, reason = False, (f"rule 5: the reference arm's seed-to-seed SD {sd:.4f} exceeds "
                                       f"{R.INFEASIBLE_REFERENCE_SD:g}; matching in the task is infeasible")
        else:
            feasible, reason = True, "feasible"
        ref_status = STATUS_REFERENCE if feasible is not None else STATUS_INCOMPLETE
        reference = _arm_result(ref_rows, ref_costs, status=ref_status, matched_cost_value=ref_value,
                                difference=0.0, matched=feasible)
        results = []
        for key, arm_rows in others:
            costs = other_costs[key]
            if not all(usable_cost(c) for c in costs) or feasible is None:
                results.append(_arm_result(arm_rows, costs, status=STATUS_INCOMPLETE))
                continue
            costs = [float(c) for c in costs]
            value = matched_cost(costs)
            diff = value - ref_value
            if not feasible:
                status = STATUS_INFEASIBLE_BUDGET if not budget_ok else STATUS_INFEASIBLE_SD
                results.append(_arm_result(arm_rows, costs, status=status, matched_cost_value=value,
                                           difference=diff, matched=False))
                continue
            ok = within_tolerance(value, ref_value, tolerance, arithmetic)
            status = STATUS_MATCHED if ok else unmatched_status(value, ref_value)
            results.append(_arm_result(arm_rows, costs, status=status, matched_cost_value=value, difference=diff,
                                       matched=ok))
        out[task] = TaskMatch(task, tolerance, arithmetic, reference, ref_value, sd, budget_ok, sd_ok, feasible,
                              reason, tuple(results))
    return out


def matched_arm_ids(result: Mapping[str, TaskMatch]) -> set[str]:
    """The arms that enter comparisons: matched arms and feasible references (rule 6 excludes the rest)."""
    keys = set()
    for task in result.values():
        if task.feasible and task.reference is not None:
            keys.add(task.reference.arm_id)
        keys.update(a.arm_id for a in task.arms if a.matched)
    return keys


def matching_flags(result: Mapping[str, TaskMatch]) -> dict[str, dict[str, bool]]:
    """{run_id: {'matched', 'infeasible'}} for every arm whose flags are decided.

    ``infeasible`` holds for every run of a task that fails rule 3 or the SD clause of rule 5; the
    reference is ``matched`` exactly when its task is feasible (it enters comparisons as the
    reference; ``matched_arm_ids``), as ``pilot.enrichment`` re-checks.
    """
    flags: dict[str, dict[str, bool]] = {}
    for task in result.values():
        if task.feasible is None:
            continue
        for arm in (task.reference, *task.arms):
            if arm is None or arm.matched is None:
                continue
            for run_id in arm.run_ids:
                flags[run_id] = {"matched": bool(arm.matched), "infeasible": not task.feasible}
    return flags


def match_arms(
    rows: Iterable[Mapping[str, Any]],
    *,
    tolerance: float,
    seed_targets: Mapping[str, int],
) -> dict[str, dict[str, bool]]:
    """The ledger's matched and infeasible flags: {run_id: {'matched': bool, 'infeasible': bool}}.

    Rules 2 to 6 on the completed Study A rows (``rows`` are dicts of ledger fields, e.g.
    ``LedgerRow.model_dump()``), with the measurement cost as the matched cost (Q-matched-cost-set).
    ``seed_targets`` maps each arm (``arm_id``: the run_id without "-s<seed>") to its seed target, 5
    plus the surplus count for primary-comparison arms (Q-arm-complete). ValueError when an arm of
    the rows has no target, when an arm's completed runs do not reach its target, when a target
    names an arm with no completed run, or when a cost is missing or not finite (``usable_cost``; the
    refusal names the runs): flags are written for complete arms only, and every completed seed counts.
    ``python -m pilot enrich match`` re-checks the result with an independent implementation before
    writing it (pilot/enrichment.py).
    """
    rows = list(rows)
    completed = _study_a_rows(rows)
    counts: dict[str, int] = {}
    for row in completed:
        counts[arm_id(row)] = counts.get(arm_id(row), 0) + 1
    no_target = sorted(set(counts) - set(seed_targets))
    if no_target:
        raise ValueError(f"no seed target for arms {no_target} (Q-arm-complete)")
    for key, target in sorted(seed_targets.items()):
        if isinstance(target, bool) or not isinstance(target, numbers.Integral) or target < 1:
            raise ValueError(f"seed target of {key} must be a positive integer, got {target!r}")
        if counts.get(key, 0) < target:
            raise ValueError(f"arm {key} is incomplete: {counts.get(key, 0)} completed runs of its {target} "
                             "(Q-arm-complete); its flags wait until the arm is complete")
    result = apply_rules(completed, tolerance=tolerance)
    for task in result.values():
        incomplete = [a for a in (task.reference, *task.arms) if a is not None and a.status == STATUS_INCOMPLETE]
        if task.reference is None or task.feasible is None or incomplete:
            unusable = [run for a in incomplete for run, c in zip(a.run_ids, a.costs) if not usable_cost(c)]
            raise ValueError(f"task {task.task}: rules 2 to 5 cannot be decided ({task.reason}; incomplete arms "
                             f"{[a.arm_id for a in incomplete]}; cost missing or not finite for {unusable})")
    return matching_flags(result)

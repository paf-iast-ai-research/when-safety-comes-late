"""Study A: boxes H0 to H4, the secondary outcomes, Part 4.1.1 and rule 7 (Parts 1.2, 4.1, 5).

Owner: Role 4, Analysis and results (Muhammad Abdullah).

Estimands (Part 5.1): "for each task, control, battery condition and arm, the outcome per seed is the
robustness gap of its matched checkpoint (equation 1); the estimands are differences between arms in
the mean gap over seeds, equation (2) for H1, gap(abrupt) - gap(ramp) at each N for H2, and
gap(treated) - gap(untreated) at N = 0.50 for H3 and H4." Every comparison uses the arms that Part 4.1
lets in: matched arms and feasible references (rule 6: an unmatched arm "is not compared"), within
one task and one step-matching control.

Primary (Part 5.2): "the robustness gap under hazard relocation on SafetyPointGoal1-v0. The primary
analysis is H1 on that outcome under both step-matching controls." Everything else is secondary and
"corrected for multiplicity within its family (the four battery conditions form one family; the
three tasks another ...) by the Holm procedure at alpha = 0.05" (Q-holm-families, answered: every
secondary claim on a (task, condition) cell must survive Holm in both families it belongs to, its
task's four-condition family and its condition's three-task family: H1 on every cell but the
primary abrupt one (each shape in its own families), and H2, H3 (a), H3 (b) and H4 on every cell, the
primary one included; the key's other reading needs one of the two. The primary claim is uncorrected,
but its p-value is a member of both abrupt H1 families. A claim with one family, recovery and return
over the tasks and H3 (c), keeps Holm in it under every reading).

"Answered" (here and below) means answered in docs/DECISIONS.md (2026-10-02), draft rows that
become amendments in Table 9.1 only when the group ratifies them, and listed in
``configs.registered.ANSWERED_QUESTIONS``. Boxes (Part 1.2), as read under those answers
(``analysis.verdict.PROPOSED``, "the proposals"; while a key is open, each alternative reading is
computed and a disagreement across it gives UNDECIDED(key)):
  H1 "Supported if. The gap difference Delta(N) of equation (2) is positive for N = 0.50 with a 95
     percent interval excluding zero, and Delta increases with N, under both step-matching controls.
     Falsified if. The intervals of Delta(0.10), Delta(0.25) and Delta(0.50) all include zero, or their
     point estimates are not ordered with N." "Increases with N" is Part 5.4's Spearman trend (points
     (N, gap), seeds resampled) together with the point ordering (Q-support-falsify-overlap).
  H2 "gap(abrupt, N) - gap(ramp, N) is positive with an interval excluding zero at N = 0.50, and
     non-negative at N = 0.10 and 0.25" / "The ramp arm's gap is as large as or larger than the abrupt
     arm's at N = 0.50, or the difference's interval includes zero at every N."
  H3 (a) dormant fraction and effective rank at onset against the gap across late arms (Spearman);
     (b) reset and injection reduce the N = 0.50 gap, additional constrained training does not;
     (c) the manipulation check at onset + 200,000 steps on the trainable layers. Part 1.2's text
     after the box ("What H3 can establish, and what it cannot"): "The word mediation is used in the
     paper only if all three parts hold, and then as 'consistent with mediation by plasticity'".
  H4 warm-started multiplier: the gap "falls with an interval excluding zero and remains above the
     N = 0 arm's gap with an interval excluding zero"; rate-limited and PID reported alongside.
  H0 "Supported if. H1 and H2 are both falsified ... H3 and H4 are then not evaluated ... reported as
     not tested, not as falsified. Falsified if. H1 or H2 is supported."
Part 5.7: "If H1 or G1 is falsified, the result is reported as an upper bound: the upper limit of the
95 percent interval of the primary estimand ... A bound below the minimum effect of interest is
reported as evidence that any effect is smaller than practical relevance in this setting; a bound
above it is reported as inconclusive at this sample size."

Secondary outcomes (Part 5.2): the other conditions and tasks (the boxes applied to every cell),
recovery time, return, the plasticity correlations (H3 (a) per cell), and Part 4.1.1's two
estimands with rule 7's repeats (exploratory): tolerance 5.0, and every arm's final checkpoint "in
place of the selected one" ("whatever the in-distribution cost": no matching filter).
Rule 7 repeats the primary analysis only (Part 4.1 rule 7: "The primary analysis is repeated twice ...
once with tolerance 5.0, and once with every arm's final checkpoint in place of the selected one"; Part
5.2: "The primary analysis is H1 on that outcome under both step-matching controls"): verdicts A-rule7a
and A-final-estimand and table A_estimands_side_by_side. "The analysis spec" (here and below) is an
internal, unregistered design document of the analysis, not kept in this repository
(docs/DECISIONS_EVIDENCE.md, Q-g4-reading); its item numbers (A1, M8, ...) name tables and checks, and
it decides nothing: the registered text and docs/DECISIONS.md do. The analysis spec's M8 and M9 ask
for "the whole Study A catalogue" at tolerance 5.0 and at the final checkpoint; that is wider than the
registered text, so the other verdicts are not repeated (a deliberate deviation from the spec), and
table A_gaps gives every run's per-seed gap under all three estimands with whether its arm is compared.
Tables: A_gaps (analysis spec A1), A_estimands_side_by_side (M9), A_norms (A21, exploratory), the
Part 4.1.1 reporting tables (A_selection, A_training_age), rule 6 (A_matching, A_unmatched), A_controller,
A_treatments_other_N (descriptive), A_estimates (every tabled contrast), A_iqm (Q-iqm) and
M11_exclusions (Part 5.6).
Treatments and controller variants at N = 0.10 and 0.25 enter no registered estimand (Part 5.1) and
are reported as descriptive estimates (Q-treatment-other-N, HANDOVER section 9; not a PENDING key).

Reading the boxes, as implemented:
* Holm corrects claims of an effect only. A clause that asserts an absence ("additional constrained
  training does not [reduce the gap]", "injection does not", "none of the three reduces it") is read
  on the uncorrected interval (Q-h3-scope: the absences of (b) "mean no reduction with an uncorrected 95
  percent interval excluding zero"): a Holm correction that removes a claim of reduction is absence
  of evidence, and Part 1.2 never rounds an outcome up to support.
* A clause that needs an estimate the data do not give (an arm unmatched under rule 6, or not yet
  complete) is not evaluated as true: H2's "non-negative at N = 0.10 and 0.25" needs E(0.10) and
  E(0.25), and H1's F1 ("The intervals of Delta(0.10), Delta(0.25) and Delta(0.50) all include zero")
  and point ordering need every Delta, so without one of them the box is decided only where the
  known estimates decide it whatever the missing one is (otherwise NOT_COMPUTABLE with the reason),
  unless the recorded Part 6.1 cut "drop_n010" removed the N = 0.10 arms (then read on the fractions
  that remain and say so). Box H1's Spearman trend is also read on the arms that remain when an arm
  is unmatched under rule 6 (analysis spec: "A missing or cut N=0.10/0.25 reduces the trend"), never
  while an arm is incomplete.
* Point estimates are compared with zero, each other and the minimum effect after
  ``stats.settled`` (gaps are differences of 100-episode means of integer costs).
* A box H1 whose conditions falsify it carries its Part 5.7 bound under its own keys
  (bound_95_upper_Delta050, bound_reading), and H2, H3 (b) or H4 through ``verdict.bound_annotation``;
  it is FALSIFIED only if that bound is below the minimum effect of 5 cost units, otherwise
  INCONCLUSIVE with the bound reported (Q-falsification-calibration, answered; Part 1.2; Part 5.7),
  and NOT_COMPUTABLE while a bound that could confirm the falsification is still to come (its
  contrast incomplete, not left out under rule 6). H3 (a) (a correlation) and H3 (c) (a manipulation
  check) claim no effect on the gap against the minimum effect of Part 1.5, so they carry no bound and
  their boxes decide. A SUPPORTED box notes a deciding |d| below the detectable size (Part 5.5;
  ``verdict.detectable_notes``).
* Part 4.1.1 (Q-training-age-systematic, not a PENDING key): "Where selected training ages differ
  systematically between arms, that difference is named in the paper as a limit on the causal
  interpretation". Every pair of arms that H1 (and the rule 7 (a) repeat), H2, H3 (b) and H4 compare on
  matched checkpoints has the 95 percent Welch interval of its difference in mean training age (table
  A_training_age, contrast rows; ``StudyA.age_note``); a verdict whose deciding contrast's interval
  excludes zero carries a note naming it. The final-checkpoint estimand (training age = total steps by
  construction) and H3 (c) (read at a fixed logging point, not at a matched checkpoint) carry none.
* Part 5.3's interquartile mean is computed for every gap and measurement-set return contrast the
  proposals record (table A_iqm; Q-iqm), from the per-episode values of the supplement records, "not
  computable" where the episodes are missing.
"""

from __future__ import annotations

import functools
import math
import statistics
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterator, Optional, Union

from configs import registered as R

from analysis import matching, questions, stats
from analysis.data import (CONDITIONS, CUT_DROP_N010, Dataset, arm_factors, awaited_a_line, awaited_arms,
                           five_seed_view)
from analysis.stats import settled
from analysis.verdict import (
    CALIBRATION_NOTE, CONFIRMATORY, EXPLORATORY, FALSIFIED, INCONCLUSIVE, NOT_COMPUTABLE, NOT_TESTED, PROPOSED,
    SECONDARY, SUPPORTED, Outcome, Reading, Verdict, bound_annotation, box_status, calibrate, claim_status,
    combine_controls, decide, detectable_notes,
)

PRIMARY_CONDITION = "hazard"  # Part 5.2: "the robustness gap under hazard relocation"
# Part 1.2 (on H3, "What H3 can establish"): the word mediation "is used in the paper only if all three parts hold,
# and then as 'consistent with mediation by plasticity', not as proof".
MEDIATION_NOTE = "all three parts hold: the result is 'consistent with mediation by plasticity' (not proof)"
LATE_N = max(R.LATE_ONSET_FRACTIONS)  # N = 0.50, the arm of H1's first clause, H3 and H4 (Part 5.1)
TREATMENT_SHAPE, TREATMENT_CONTROL = "abrupt", "total_steps"  # Table 3.3: treatments and controllers
TREATED_VARIANTS = (*R.TREATMENTS, *R.CONTROLLER_VARIANTS, R.PID_VARIANT)
RESET, INJECTION, ADDITIONAL = R.TREATMENTS  # Table 2.4: H3 (b) and (c) name them
WARM, RATE = R.CONTROLLER_VARIANTS  # Table 2.4: H4 reads the warm-started multiplier, the others alongside
# The gap a comparison reads: the matched checkpoint's (primary), the rule 7 (a) repeat at tolerance 5.0
# (arms matched only at 5.0 read the sensitivity battery), or the final checkpoint's (rule 7 (b); Part 4.1.1).
ESTIMANDS = ("matched", "tolerance5", "final")
Member = Union[None, float, stats.Incomplete]  # a Holm family member: a p-value, not tested, or incomplete


def _tolerance(estimand: str) -> float:
    """The matching tolerance of the estimand (Part 4.1 rule 5; rule 7 (a)'s sensitivity tolerance of 5.0 cost units)."""
    return R.SENSITIVITY_TOLERANCE if estimand == "tolerance5" else R.MATCH_TOLERANCE


CUT_TASK_REASON = "removed by the recorded Part 6.1 cut"
NO_ARM_REASON = "an arm has no completed run (a recorded Part 6.1 cut removed it, or the design has none)"


def _excludes_zero(lo: float, hi: float) -> bool:
    """Whether the interval [lo, hi] excludes zero."""
    return lo > 0 or hi < 0


def _includes_zero(lo: float, hi: float) -> bool:
    """Whether the interval [lo, hi] includes zero (its ends included)."""
    return lo <= 0 <= hi


@dataclass
class Arm:
    """The completed runs of one arm of one task. ``records`` is empty for an awaited arm: one the registered
    design, less the recorded Part 6.1 cuts, expects (``analysis.data.awaited_arms``) but that has no completed
    run yet; its comparisons are incomplete (data still to come), never "not tested" (Q-holm-families). So are
    those of an arm with some completed runs but fewer than its seed target (``short``: (completed, target);
    ``analysis.data.Dataset.awaited_a``; Q-arm-complete): never decided on the seeds that happen to be present."""

    arm_id: str
    records: list[dict[str, Any]]
    short: Optional[tuple[int, int]] = None

    @property
    def awaited(self) -> bool:
        """Whether the arm's data are still to come (no completed run, or fewer than its seed target)."""
        return not self.records or self.short is not None

    @property
    def waiting(self) -> str:
        """Why the arm's data are still to come ('' when they are not)."""
        if not self.records:
            return f"{self.arm_id} has no completed run yet (registered and not cut: Part 6.1; its data are still to come)"
        if self.short is not None:
            return awaited_a_line(self.arm_id, *self.short).replace(": ", " has ", 1)
        return ""


class StudyA:
    """Study A's data with the caches that let every reading reuse the same estimates."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
        self.records = dataset.study_a
        self._matching: dict[tuple, dict[str, matching.TaskMatch]] = {}
        self._e2: dict[tuple, stats.TwoArmEstimate] = {}
        self._rho: dict[tuple, stats.SpearmanResult] = {}
        self._grids: dict[tuple, dict] = {}
        self._gap_outcomes: dict[tuple[str, str], stats.EpisodeOutcome] = {}
        self._five: Optional[StudyA] = None
        # Only the proposals' estimates are tabled: another reading may compare other seeds or budgets
        # under the same analysis id, and the table must show what the verdicts report.
        self.estimate_rows: dict[str, dict[str, Any]] = {}
        self.iqm_specs: dict[str, tuple[Arm, Arm, stats.EpisodeOutcome, float]] = {}
        self.iqm_uses: dict[str, set[str]] = {}
        # Part 4.1.1's training-age contrasts (``age_note``): by (x arm, y arm), the rows the proposals tabled and
        # the analysis ids that read them
        self._ages: dict[tuple[str, str], Optional[dict[str, Any]]] = {}
        self.age_rows: dict[tuple[str, str], dict[str, Any]] = {}
        self.age_uses: dict[tuple[str, str], set[str]] = {}

    def for_reading(self, reading: Reading) -> StudyA:
        """The data a reading compares (Q-surplus-in-analysis, answered): every completed seed of an arm,
        registered, replacement and surplus, in matching and every two-arm estimate; the other reading keeps five
        per arm. The two exceptions under every reading are Part 5.4's seed-index-resampled correlations, which read
        each arm's registered seeds and their replacements: the H1 trend (``_h1_control``: "twenty points: four onset
        fractions by five seeds") and H3 (a) (``_h3a_points``: "with the same interval")."""
        if reading.seeds == "all":
            return self
        if self._five is None:
            view = five_seed_view(self.dataset)
            self._five = self if view is self.dataset else StudyA(view)
        return self._five

    # -- arms and matching -------------------------------------------------

    def arms(self, task: str, *, N: float, shape: Optional[str] = None, control: Optional[str] = None,
             treatment: Optional[str] = None, controller: Optional[str] = None) -> Optional[Arm]:
        """The arm with these factors on ``task``: None if a recorded Part 6.1 cut removed it (final mode, whether
        or not some of its runs completed before the cut: the cut takes the arm out of every comparison), or if it
        has no completed run and the final analysis does not expect it (the design has no such arm, or pilot mode);
        an awaited arm (``Arm.awaited``, no records) if the registered design less the cuts expects it."""
        found = [r for r in self.records if r["task"] == task and r["N"] == N
                 and r["onset_shape"] == shape and r["step_matching"] == control
                 and r["treatment"] == treatment and r["controller_variant"] == controller]
        if self.dataset.mode == "final":
            key = arm_factors({"task": task, "N": N, "onset_shape": shape, "step_matching": control,
                               "treatment": treatment, "controller_variant": controller})
            arm_id = awaited_arms(tuple(self.dataset.cuts)).get(key)
            if arm_id is None and key in awaited_arms(()):
                return None  # removed by a recorded cut (``absent_reason`` names it)
            if not found:
                return None if arm_id is None else Arm(arm_id, [])
        if not found:
            return None
        ids = {r["arm_id"] for r in found}
        if len(ids) != 1:
            raise ValueError(f"several arms match {task} N={N} {shape} {control} {treatment} {controller}: {sorted(ids)}")
        arm_id = ids.pop()
        return Arm(arm_id, sorted(found, key=lambda r: r["seed"]), short=self.dataset.awaited_a.get(arm_id))

    def reference(self, task: str) -> Optional[Arm]:
        """The task's N = 0 reference arm (``arms``)."""
        return self.arms(task, N=R.ONSET_FRACTIONS[0])

    def matching(self, reading: Reading, tolerance: float = R.MATCH_TOLERANCE) -> dict[str, matching.TaskMatch]:
        """Rules 2 to 6 under the reading. A row whose stored matched step is not the checkpoint rule 1 selects
        (``rule_one_rechecked`` False, M1; the dataset's problems name it) has no usable cost: its cost and gaps
        are of a checkpoint the registered rule did not choose, so its arm is incomplete (every comparison that
        needs it is NOT_COMPUTABLE), never silently used."""
        key = (reading.cost_field, reading.arithmetic, tolerance)
        if key not in self._matching:
            field_name = reading.cost_field
            self._matching[key] = matching.apply_rules(
                self.records, tolerance=tolerance, arithmetic=reading.matching_arithmetic,
                cost_of=lambda r: None if r.get("rule_one_rechecked") is False else r.get(field_name),
            )
        return self._matching[key]

    def compared(self, reading: Reading, estimand: str) -> Optional[set[str]]:
        """Arms that enter comparisons for the estimand (None: no matching filter, rule 7 (b))."""
        if estimand == "final":
            return None
        tolerance = _tolerance(estimand)
        return matching.matched_arm_ids(self.matching(reading, tolerance))

    def arm_status(self, reading: Reading, estimand: str, arm: Optional[Arm]) -> Optional[str]:
        """The arm's status under rules 2 to 6 for the estimand (``matching.STATUSES``); None without an arm,
        for an awaited arm, or for the final checkpoint (no matching filter). It tells an arm that rule 6 leaves out
        ("It is not compared") from one that cannot be matched yet (a cost missing: ``matching.STATUS_INCOMPLETE``)."""
        if arm is None or arm.awaited or estimand == "final":
            return None
        tolerance = _tolerance(estimand)
        task = self.matching(reading, tolerance).get(str(arm.records[0]["task"]))
        found = None if task is None else task.arm(arm.arm_id)
        return None if found is None else found.status

    def task_state(self, reading: Reading, task: str, estimand: str) -> tuple[Optional[bool], str]:
        """(feasible, reason) of the task under rules 3 and 5 (always feasible for the final checkpoint); feasible is
        None while the matching cannot be decided yet, when the task has no completed run, or when a recorded cut
        removed it (``task_cut_by``: not tested, its reason says so)."""
        if estimand == "final":
            return True, "no matching filter (Part 4.1.1: 'whatever the in-distribution cost')"
        cut = self.task_cut_by(task)
        if cut:  # whether or not some of its runs completed before the cut (``task_cut_by``)
            return None, f"{CUT_TASK_REASON} {', '.join(cut)} ({task})"
        tolerance = _tolerance(estimand)
        result = self.matching(reading, tolerance).get(task)
        if result is None:
            return None, f"no completed Study A run of {task}" + (
                " (not cut: its data are still to come)" if self.task_awaited(task) else "")
        ref = self.reference(task)
        if ref is not None and ref.short is not None:  # rules 2, 3 and 5 read the reference's seeds
            return None, f"incomplete: the reference {ref.waiting}"
        return result.feasible, result.reason

    def task_awaited(self, task: str) -> bool:
        """The final analysis expects arms of ``task`` (the recorded Part 6.1 cuts leave some) but none has a
        completed run: its cells are incomplete, never "not tested" (``analysis.data.awaited_arms``)."""
        if self.dataset.mode != "final" or any(r["task"] == task for r in self.records):
            return False
        return any(f[0] == task for f in awaited_arms(tuple(self.dataset.cuts)))

    def task_cut_by(self, task: str) -> tuple[str, ...]:
        """The recorded Part 6.1 cuts that removed every arm of ``task`` (final mode), decided from the design alone,
        as ``arms`` decides an arm's: whether or not some of its runs completed before the cut, the cut takes them out
        of every comparison. Its cells are not tested (``NOT_TESTED``), not incomplete; () otherwise. The cut named is
        the first of the recorded cuts (a prefix of the registered order) after which no arm of the task is left."""
        cuts = tuple(self.dataset.cuts)
        if (self.dataset.mode != "final" or not any(f[0] == task for f in awaited_arms(()))
                or any(f[0] == task for f in awaited_arms(cuts))):
            return ()
        for i, cut in enumerate(cuts):
            if not any(f[0] == task for f in awaited_arms(cuts[:i + 1])):
                return (cut,)
        return ()

    def absent_reason(self, task: str, **factors: Any) -> str:
        """Why there is no arm with these factors (``arms``'s keywords): "removed by the recorded Part 6.1 cut
        K (arm)" when the registered design has it and the first recorded cut K after which it is gone removed it
        (an arm a cut removed is not tested: ``_claims`` reports NOT_TESTED), else the
        generic reason of ``contrast``."""
        if self.dataset.mode == "final":
            key = arm_factors({"task": task, "N": factors.get("N"), "onset_shape": factors.get("shape"),
                               "step_matching": factors.get("control"), "treatment": factors.get("treatment"),
                               "controller_variant": factors.get("controller")})
            cuts = tuple(self.dataset.cuts)
            arm_id = awaited_arms(()).get(key)
            if arm_id is not None and key not in awaited_arms(cuts):
                for i, cut in enumerate(cuts):
                    if key not in awaited_arms(cuts[:i + 1]):
                        return f"{CUT_TASK_REASON} {cut} ({arm_id})"
        return NO_ARM_REASON

    # -- values --------------------------------------------------------------

    @staticmethod
    def gap(record: dict[str, Any], condition: str, estimand: str) -> Optional[float]:
        """Equation (1) for one seed: the matched checkpoint's gap, at tolerance 5.0, or the final one's."""
        if estimand == "matched":
            return record.get(f"gap_{condition}")
        if estimand == "tolerance5":
            value = record.get(f"gap_{condition}")
            return value if value is not None else record.get(f"sens_gap_{condition}")
        if estimand == "final":
            return record.get(f"final_gap_{condition}")
        raise ValueError(f"unknown estimand {estimand!r}; one of {ESTIMANDS}")

    @staticmethod
    def gap_episodes(record: dict[str, Any], condition: str, estimand: str) -> Optional[list[tuple[float, list[float]]]]:
        """The episodes behind ``gap``: [(+1, C_cond episodes), (-1, C_ID episodes)], or None (Q-iqm)."""
        if estimand == "final":
            cond, base = record["final_episodes"].get(condition), record["final_episodes"].get("measurement")
        elif estimand == "tolerance5" and record.get(f"gap_{condition}") is None:
            cond, base = record["sens_episodes"].get(condition), record["measurement_episodes"]
        else:
            cond, base = record["battery_episodes"].get(condition), record["measurement_episodes"]
        return None if cond is None or base is None else [(1.0, cond), (-1.0, base)]

    def gap_outcome(self, condition: str, estimand: str = "matched") -> stats.EpisodeOutcome:
        """Equation (1) per seed as an ``EpisodeOutcome`` (the value ``gap`` reads, and its episodes)."""
        key = (condition, estimand)
        if key not in self._gap_outcomes:
            self._gap_outcomes[key] = stats.EpisodeOutcome(
                source=f"gap_{condition}|{estimand}", value=lambda r: self.gap(r, condition, estimand),
                episodes=lambda r: self.gap_episodes(r, condition, estimand))
        return self._gap_outcomes[key]

    @staticmethod
    def values(arm: Arm, get: Callable[[dict[str, Any]], Optional[float]]) -> tuple[Optional[dict[int, float]], list[str]]:
        """{seed: value} of the arm, or (None, run_ids missing the value): nothing is dropped silently."""
        out, missing = {}, []
        for rec in arm.records:
            value = get(rec)
            if value is None or not math.isfinite(float(value)):
                missing.append(rec["run_id"])
            else:
                out[int(rec["seed"])] = float(value)
        return (None, missing) if missing else (out, [])

    def e2(self, x: dict[int, float], y: dict[int, float], analysis_id: str,
           level: float = R.INTERVAL_LEVEL, *, record: bool = True) -> stats.TwoArmEstimate:
        """x - y (``stats.estimate_two_arms``), cached; tabled in ``estimate_rows`` when ``record``."""
        key = (analysis_id, level, tuple(sorted(x.items())), tuple(sorted(y.items())))
        if key not in self._e2:
            self._e2[key] = stats.estimate_two_arms(x, y, analysis_id=analysis_id, level=level)
        est = self._e2[key]
        if record:
            self.estimate_rows[f"{analysis_id}|{level}"] = est.row()
        return est

    def rho(self, points: dict[int, list[tuple[float, float]]], analysis_id: str,
            level: float = R.INTERVAL_LEVEL) -> stats.SpearmanResult:
        """The seed-resampled Spearman correlation of ``points`` (``stats.spearman_bootstrap``), cached."""
        key = (analysis_id, level, tuple(sorted((s, tuple(p)) for s, p in points.items())))
        if key not in self._rho:
            self._rho[key] = stats.spearman_bootstrap(points, analysis_id=analysis_id, level=level)
        return self._rho[key]

    # -- building blocks of the boxes --------------------------------------

    def contrast(self, reading: Reading, *, task: str, x: Optional[Arm], y: Optional[Arm],
                 get: Callable[[dict[str, Any]], Optional[float]],
                 analysis_id: str, estimand: str = "matched", filter_matched: bool = True,
                 level: float = R.INTERVAL_LEVEL) -> tuple[Optional[stats.TwoArmEstimate], str]:
        """x - y for two arms of one task, or (None, why not); tabled (with its IQM) under the proposals only."""
        if x is None or y is None:
            cut = self.task_cut_by(task)
            if cut:
                return None, f"{CUT_TASK_REASON} {', '.join(cut)} ({task})"
            return None, NO_ARM_REASON
        waiting = [a.waiting for a in (x, y) if a.awaited]
        if waiting:
            return None, f"incomplete: {'; '.join(waiting)}"
        if filter_matched:
            # Rules 2 to 6 are decided per task, from the reference's seeds among others: while they wait (a short
            # reference, a missing cost), no arm of the task is matched or unmatched yet, so no contrast is
            # decided (or left out by rule 6) on data still to come (Q-holm-families).
            feasible, why_task = self.task_state(reading, task, estimand)
            if feasible is None and any(r["task"] == task for r in self.records):
                return None, f"incomplete: matching of {task} cannot be decided yet ({why_task.removeprefix('incomplete: ')})"
            compared = self.compared(reading, estimand)
            for arm in (x, y):
                if compared is not None and arm.arm_id not in compared:
                    if self.arm_status(reading, estimand, arm) == matching.STATUS_INCOMPLETE:
                        return None, (f"incomplete: {arm.arm_id} cannot be matched yet (a selected-checkpoint cost is "
                                      "missing or not finite, or a stored matched step is not rule 1's choice; "
                                      "Part 4.1 rules 1 to 5)")
                    return None, (f"{arm.arm_id} is not compared (Part 4.1 rule 6: unmatched; or rules 3 and 5: the "
                                  "task is infeasible and enters no comparison)")
        xv, xm = self.values(x, get)
        yv, ym = self.values(y, get)
        if xv is None or yv is None:
            return None, f"incomplete: no value for {xm + ym}"
        if len(xv) < 2 or len(yv) < 2:
            return None, "fewer than two seeds in an arm"
        record = reading == PROPOSED
        if record and isinstance(get, stats.EpisodeOutcome):
            key = f"IQM|{x.arm_id}|{y.arm_id}|{get.source}|{level}"
            self.iqm_specs.setdefault(key, (x, y, get, level))
            self.iqm_uses.setdefault(key, set()).add(analysis_id)
        return self.e2(xv, yv, analysis_id, level, record=record), ""

    def age_note(self, reading: Reading, x: Arm, y: Arm, analysis_id: str) -> list[str]:
        """Part 4.1.1: "Where selected training ages differ systematically between arms, that difference is named in
        the paper as a limit on the causal interpretation" (Q-training-age-systematic, HANDOVER section 9; not a
        PENDING key). Systematic: the 95 percent Welch interval of x's minus y's mean training age (the matched
        checkpoint's step, over every completed seed) excludes zero, unrounded limits.

        The boxes call it for each contrast they compare on matched checkpoints (H1 and the rule 7 (a) repeat, H2,
        H3 (b), H4); the row is tabled (A_training_age, with the analysis ids that read it) under the proposals
        only. Returns the note a verdict whose deciding contrast this is carries ([] when the ages do not differ
        systematically, or a run has no training age)."""
        key = (x.arm_id, y.arm_id)
        if key not in self._ages:
            xv, _ = self.values(x, lambda r: r.get("training_age"))
            yv, _ = self.values(y, lambda r: r.get("training_age"))
            row: Optional[dict[str, Any]] = None
            if xv is not None and yv is not None and len(xv) >= 2 and len(yv) >= 2:
                w = stats.welch(xv, yv)
                row = {"contrast": f"{x.arm_id} - {y.arm_id}", "task": x.records[0]["task"], "arm_id": x.arm_id,
                       "arm_y": y.arm_id, "n": len(xv), "n_y": len(yv),
                       "mean_training_age": statistics.fmean(xv.values()),
                       "mean_training_age_y": statistics.fmean(yv.values()), "diff": w.estimate,
                       "welch_lo": w.lo, "welch_hi": w.hi, "systematic": bool(_excludes_zero(w.lo, w.hi))}
            self._ages[key] = row
        row = self._ages[key]
        if row is None:
            return []
        if reading == PROPOSED:
            self.age_rows[key] = row
            self.age_uses.setdefault(key, set()).add(analysis_id)
        if not row["systematic"]:
            return []
        return [f"training ages differ systematically between {x.arm_id} and {y.arm_id} (Part 4.1.1: the 95 percent "
                f"Welch interval of the difference in mean training age, [{row['welch_lo']:.6g}, {row['welch_hi']:.6g}], "
                "excludes zero): named as a limit on the causal interpretation (Q-training-age-systematic)"]

    def condition_untested(self, condition: str, estimand: str = "matched") -> bool:
        """A battery condition not evaluated by design yet: a question its evaluation waits on is open
        (``questions.CONDITION_KEYS``: the continuations of finetune and transfer wait on Q-continuations and
        Q-transfer-obs) and no run has its gap. Its cells are not tested, so the Holm families shrink to
        the other conditions (Q-holm-families); once its gaps arrive, or its keys are answered, a cell
        without its gap is incomplete data instead."""
        if not any(R.is_open(k) for k in questions.CONDITION_KEYS[condition]):
            return False
        return all(self.gap(r, condition, estimand) is None for r in self.records)

    def holm_survival(self, reading: Reading, grid_id: str, pvalues: Callable[[str, str], Member],
                      estimand: str = "matched") -> dict[tuple[str, str], Optional[bool]]:
        """Q-holm-families: survival of Holm (alpha = 0.05) in the four-condition and the three-task family.

        Every tested (task, condition) cell is a member of its task's four-condition family and of its
        condition's three-task family (Q-holm-families: "Families shrink only to cells not tested"; "the primary
        H1 claim is uncorrected, but its p-value counts as a member of both abrupt H1 families"), so it counts in the
        corrections of both, and a claim on it is decided by both (Q-holm-families, answered: "Every secondary Study
        A claim on a (task, condition) cell must survive Holm (alpha = 0.05) in both families it belongs to"; Part 5.2
        names both families, and only survival in both controls the error of each). The key's other reading
        (``Reading.holm == 'either'``): survival in at least one of them.
        ``pvalues`` gives a p-value, None (the cell is not tested: the family shrinks) or
        ``stats.Incomplete`` (its data are still to come: the family does not shrink, and a survival it
        could change is None; ``holm_cell`` names the member). A condition not evaluated by design yet
        (``condition_untested``) is not tested. None: the cell was not tested, or its survival waits on an
        incomplete member.
        """
        key = (grid_id, reading.cost_field, reading.arithmetic, estimand)
        if key not in self._grids:
            untested = {c for c in CONDITIONS if self.condition_untested(c, estimand)}
            ps = {(t, c): pvalues(t, c) for t in R.TASKS_STUDY_A for c in CONDITIONS}
            ps = {cell: None if isinstance(p, stats.Incomplete) and cell[1] in untested else p for cell, p in ps.items()}
            by_condition = {t: stats.holm_decisions({c: ps[(t, c)] for c in CONDITIONS}, R.ALPHA) for t in R.TASKS_STUDY_A}
            by_task = {c: stats.holm_decisions({t: ps[(t, c)] for t in R.TASKS_STUDY_A}, R.ALPHA) for c in CONDITIONS}
            survival: dict[str, dict[tuple[str, str], Optional[bool]]] = {
                mode: {cell: None for cell in ps} for mode in ("both_families", "either")}
            incomplete: dict[tuple[str, str], str] = {}
            for t, c in ps:
                fam_c, fam_t = by_condition[t], by_task[c]
                if c not in fam_c[0] or t not in fam_t[0]:
                    continue  # not tested (or itself incomplete: its own box says so)
                a, b = fam_c[1][c], fam_t[1][t]  # its four-condition family, its three-task family
                survival["both_families"][(t, c)] = _and3(a, b)
                survival["either"][(t, c)] = _or3(a, b)
                waiting = [f"{t2} x {c}: {why}" for t2, why in fam_t[2].items()]
                waiting += [f"{t} x {c2}: {why}" for c2, why in fam_c[2].items()]
                if waiting:
                    incomplete[(t, c)] = "; ".join(sorted(set(waiting)))
            self._grids[key] = {"survival": survival, "by_condition": {t: f[0] for t, f in by_condition.items()},
                                "by_task": {c: f[0] for c, f in by_task.items()}, "incomplete": incomplete}
        if reading.holm not in self._grids[key]["survival"]:
            raise ValueError(f"unknown Q-holm-families reading {reading.holm!r}")
        return self._grids[key]["survival"][reading.holm]

    def holm_cell(self, reading: Reading, grid_id: str, pvalues: Callable[[str, str], Member],
                  cell: tuple[str, str], estimand: str = "matched") -> tuple[Optional[bool], str]:
        """(survives, why not decided) of one tested cell (``holm_survival``): None with the incomplete family
        members named when they could still change it; a cell without a p-value of its own does not survive."""
        survives = self.holm_survival(reading, grid_id, pvalues, estimand).get(cell)
        if survives is not None:
            return survives, ""
        why = self._grids[(grid_id, reading.cost_field, reading.arithmetic, estimand)]["incomplete"].get(cell, "")
        return (None, holm_waiting(why)) if why else (False, "")


def holm_waiting(members: str) -> str:
    """The reason a Holm decision waits: the family's incomplete members (``stats.Incomplete``)."""
    return (f"incomplete: Holm cannot be decided while a member of the family is incomplete ({members}); "
            "the family shrinks only to the cells not tested, never on data still to come (Q-holm-families)")


def _member(est: Optional[stats.TwoArmEstimate], why: str, where: str = "") -> Member:
    """A family member from a contrast: its Welch p-value; ``stats.Incomplete`` when the data are still to come
    (the contrast's reason starts with 'incomplete'); else None (not tested: no arm, rule 6, an infeasible task)."""
    if est is not None:
        return est.welch.p
    return stats.Incomplete(f"{where}{': ' if where else ''}{why}") if why.startswith("incomplete") else None


def _cut_task(A: StudyA, task: str) -> Optional[Outcome]:
    """NOT_TESTED for a box on a task that a recorded Part 6.1 cut removed (``StudyA.task_cut_by``): "an arm a
    cut removed is not tested", never incomplete data; None otherwise."""
    cut = A.task_cut_by(task)
    return Outcome(NOT_TESTED, f"{task}: {CUT_TASK_REASON} {', '.join(cut)}") if cut else None


def _task_member(A: StudyA, reading: Reading, task: str, estimand: str) -> tuple[bool, Member]:
    """(tested, member) of a task under rules 3 and 5: a task whose matching waits on a missing cost, or one the
    final analysis expects (not cut) without any completed run yet (``StudyA.task_awaited``), is incomplete; one
    a recorded cut removed (whether or not some of its runs completed before the cut: ``StudyA.task_cut_by``; or,
    in pilot mode, not run) or infeasible is not tested."""
    if A.task_cut_by(task):
        return False, None
    feasible, why = A.task_state(reading, task, estimand)
    if feasible:
        return True, None
    if feasible is None and any(r["task"] == task for r in A.records):
        return False, stats.Incomplete(f"{task}: incomplete: matching cannot be decided yet ({why})")
    if feasible is None and A.task_awaited(task):
        return False, stats.Incomplete(f"{task}: incomplete: {why}")
    return False, None


# ---------------------------------------------------------------------------
# H1
# ---------------------------------------------------------------------------


def _estimate_numbers(prefix: str, est: Optional[stats.TwoArmEstimate], reading: Reading) -> dict[str, Any]:
    """The numbers of one contrast under ``prefix`` (point, intervals, p, d, seeds, detectability)."""
    if est is None:
        return {prefix: None}
    lo, hi = est.interval(reading.interval)
    return {prefix: est.diff, f"{prefix}_welch_lo": est.welch.lo, f"{prefix}_welch_hi": est.welch.hi,
            f"{prefix}_boot_lo": est.bootstrap.lo, f"{prefix}_boot_hi": est.bootstrap.hi,
            f"{prefix}_p": est.welch.p, f"{prefix}_d": est.cohens_d, f"{prefix}_n": (len(est.seeds_x), len(est.seeds_y)),
            f"{prefix}_decides_lo": lo, f"{prefix}_decides_hi": hi,
            f"{prefix}_detectable_d": est.detectable_d, f"{prefix}_below_detectable": est.below_detectable}


def _and3(*values: Optional[bool]) -> Optional[bool]:
    """Three-valued AND: False if any condition is known to fail, None if any is unknown, else True."""
    if any(v is not None and not v for v in values):
        return False
    return None if any(v is None for v in values) else True


def _or3(*values: Optional[bool]) -> Optional[bool]:
    """Three-valued OR: True if any condition is known to hold, None if any is unknown, else False."""
    if any(v is not None and v for v in values):
        return True
    return None if any(v is None for v in values) else False


def _h1_control(A: StudyA, reading: Reading, *, task: str, condition: str, control: str, shape: str,
                estimand: str, holm: bool) -> Outcome:
    """Box H1 under one step-matching control on one (task, condition).

    A Delta(N) that cannot be computed is read by the reason it is missing:

    * the recorded Part 6.1 cut "drop_n010" removed the N = 0.10 arms (``Dataset.cuts``): the fraction
      leaves the box, and F1, F2, the point ordering and the trend are read at the fractions that
      remain (Part 6.1: N = 0.10 "contributes to the trend test of H1 and not to the primary
      comparison"; the cut takes it out of both);
    * its arm is unmatched under rule 6 ("It is not compared"): the Spearman trend uses the arms that
      remain (analysis spec, H1: "A missing or cut N=0.10/0.25 reduces the trend"), but F1 ("The
      intervals of Delta(0.10), Delta(0.25) and Delta(0.50) all include zero") and the point
      ordering, which name that Delta, stay open (and S1, for Delta(0.50));
    * anything else (an arm not yet run or not yet matched, a gap not yet evaluated) is incomplete
      data: the trend and those clauses stay open (Part 5.6; analysis spec: "A missing enrichment
      field makes the analysis 'incomplete', never a silent drop").

    The conditions are three-valued, as box H2's: S1 is unknown without Delta(0.50), F1 fails as soon
    as one known interval excludes zero, F2 (and a failed ordering in S2) holds as soon as the known
    points are out of order, and otherwise a missing Delta leaves them unknown; ``box_status`` then
    decides only what the missing Delta cannot change, and gives NOT_COMPUTABLE (with the reason) for the
    rest. A falsified box without Delta(0.50) has no Part 5.7 bound: left out under rule 6, it is "no
    interval" for good (INCONCLUSIVE); still to come, the bound that decides the falsification is too
    (NOT_COMPUTABLE).

    The numbers say what the box still waits on (``could_be_falsified``): ``incomplete``, the fractions whose
    Delta is still to come, and ``holm_waits``; a box that lacks only what rule 6 leaves out, or whose task
    rules 3 and 5 make infeasible, is NOT_COMPUTABLE for good.

    Every Delta reads every completed seed of its arms; the Spearman trend reads each arm's registered seeds
    and their replacements only, Part 5.4's "twenty points" (Q-surplus-in-analysis, answered).
    """
    feasible, why = A.task_state(reading, task, estimand)
    if feasible is False:  # rules 3 and 5: nothing the box lacks is still to come
        return Outcome(NOT_COMPUTABLE, f"{task}: {why}", numbers={"incomplete": [], "holm_waits": False})
    if feasible is None:
        return Outcome(NOT_COMPUTABLE,
                       f"{task}: incomplete: matching cannot be decided yet ({why.removeprefix('incomplete: ')})")
    ref = A.reference(task)
    get = A.gap_outcome(condition, estimand)
    deltas: dict[float, stats.TwoArmEstimate] = {}
    notes: list[str] = []
    arms: dict[float, Arm] = {}
    unmatched: dict[float, str] = {}  # rule 6: the trend is reduced, the clauses naming the fraction stay open
    incomplete: dict[float, str] = {}  # not yet in the data: the trend and those clauses stay open
    cut_early = CUT_DROP_N010 in A.dataset.cuts
    for N in R.LATE_ONSET_FRACTIONS:
        arm = A.arms(task, N=N, shape=shape, control=control)
        est, why = A.contrast(reading, task=task, x=arm, y=ref, get=get, estimand=estimand,
                              filter_matched=estimand != "final",
                              analysis_id=f"H1|{task}|{condition}|{control}|{shape}|{estimand}|N{N:.2f}")
        if est is None:
            clauses = "S1, F1 and the point ordering" if N == LATE_N else "F1 and the point ordering"
            if cut_early and N == min(R.LATE_ONSET_FRACTIONS):
                notes.append(f"Delta({N:.2f}) is not read: the recorded Part 6.1 cut 'drop_n010' removed the N = {N:.2f} "
                             "arms ('Drop N = 0.10 from the main sweep'); F1, F2 and the trend are read at the fractions "
                             "that remain")
            elif A.arm_status(reading, estimand, arm) in matching.UNMATCHED_STATUSES:
                unmatched[N] = why
                notes.append(f"Delta({N:.2f}): {why}; the trend uses the arms that remain, and {clauses}, which "
                             f"name Delta({N:.2f}), stay open")
            else:
                incomplete[N] = why
                notes.append(f"Delta({N:.2f}) cannot be computed yet: {why}; the trend, {clauses} stay open (a "
                             "missing field makes the analysis incomplete, never a silent drop)")
            continue
        deltas[N], arms[N] = est, arm
    missing = {**unmatched, **incomplete}
    if estimand != "final":  # Part 4.1.1: the training ages of the matched checkpoints each Delta compares
        for N, arm in arms.items():
            notes += A.age_note(reading, arm, ref, f"H1|{task}|{condition}|{control}|{shape}|{estimand}|N{N:.2f}")
    # The trend needs at least one late Delta (then the reference and that arm have two seeds or more each). Part
    # 5.4 registers "twenty points: four onset fractions by five seeds" (Q-surplus-in-analysis, answered): its
    # points are each arm's registered seeds and their replacements (``analysis.data.five_seed_view``),
    # the surplus seeds of the N = 0 and N = 0.50 arms left out; the arms are those whose Delta the full data give.
    trend: Optional[stats.SpearmanResult] = None
    if deltas:
        F = A.for_reading(replace(reading, seeds="five"))
        points: dict[int, list[tuple[float, float]]] = {}
        for N, five in ((R.ONSET_FRACTIONS[0], F.reference(task)),
                        *((N, F.arms(task, N=N, shape=shape, control=control)) for N in arms)):
            for seed, value in ((None if five is None else F.values(five, get)[0]) or {}).items():
                points.setdefault(seed, []).append((N, value))
        trend = A.rho(points, f"H1-trend|{task}|{condition}|{control}|{shape}|{estimand}")
    top = deltas.get(LATE_N)  # None: S1 is unknown, the Holm cell and the Part 5.7 bound are not read
    holm_ok: Optional[bool] = True
    holm_why = ""
    if holm and top is not None:
        holm_ok, holm_why = A.holm_cell(reading, f"H1|{control}|{shape}|{estimand}",
                                        lambda t, c: _h1_p(A, reading, t, c, control, shape, estimand), (task, condition),
                                        estimand)
    diffs = [settled(deltas[N].diff) for N in sorted(deltas)]
    known_ordered = all(a <= b for a, b in zip(diffs, diffs[1:]))
    ordered: Optional[bool] = False if not known_ordered else (None if missing else True)
    trend_ok = (trend is not None and not math.isnan(trend.rho) and not math.isnan(trend.lo)
                and trend.rho > 0 and trend.lo > 0)
    # an incomplete arm would change the trend; without any late Delta there is no trend to read
    trend_read: Optional[bool] = None if incomplete or trend is None else trend_ok
    s1: Optional[bool] = None
    if top is not None:
        lo, _ = top.interval(reading.interval)
        s1 = _and3(settled(top.diff) > 0 and lo > 0, holm_ok)
    s2 = trend_read if reading.ordering == "trend" else _and3(trend_read, ordered)
    supported = _and3(s1, s2)
    f1: Optional[bool] = (False if any(_excludes_zero(*deltas[N].interval(reading.interval)) for N in deltas)
                          else None if missing else True)
    f2: Optional[bool] = True if not known_ordered else (None if missing else False)
    falsified = _or3(f1, f2)
    status = box_status(supported, falsified, reading, notes)
    reason = ""
    if status == NOT_COMPUTABLE:
        parts = []
        if missing:
            parts.append("; ".join(f"Delta({N:.2f}): {w}" for N, w in sorted(missing.items()))
                         + " (box H1 names Delta(0.50) in 'Supported if' and Delta(0.10), Delta(0.25) and Delta(0.50) in "
                           "'Falsified if', and reads 'Delta increases with N'; only the recorded Part 6.1 cut 'drop_n010' "
                           "removes a fraction)")
        if holm_ok is None:
            parts.append(f"Delta({LATE_N:.2f}): {holm_why}")
        reason = f"under {control}: " + "; ".join(parts)
    # Part 5.7 applies to a box its conditions falsify ("If H1 or G1 is falsified, the result is reported as an
    # upper bound"): its numbers and reading are attached below for such a box, never beside one its conditions
    # support or leave inconclusive. The bound decides (Q-falsification-calibration, answered: "a bound above it is
    # reported as inconclusive at this sample size"; no interval: not confirmed); while Delta(0.50) is still to come
    # (incomplete, not left out under rule 6), so is the bound, and the box waits on it.
    ann, bounded = bound_annotation("Delta(0.50)", None if top is None else top.welch.hi, R.MIN_EFFECT_STUDY_A,
                                    claimed="positive")
    box_falsified = status == FALSIFIED
    if box_falsified and reading.calibration == "bound" and not bounded:
        if LATE_N in incomplete:
            status = NOT_COMPUTABLE
            reason = (f"under {control}: Delta({LATE_N:.2f}): {incomplete[LATE_N]}; the Part 5.7 bound that decides "
                      "the falsification waits on it (Q-falsification-calibration)")
        else:
            status = INCONCLUSIVE
            notes.append(CALIBRATION_NOTE)
    if status == SUPPORTED:
        notes += detectable_notes({"Delta(0.50)": top})
    if trend is not None and trend.n_points < R.H1_TREND_POINTS:
        notes.append(f"the trend has {trend.n_points} points; Part 5.4 registers {R.H1_TREND_POINTS} ('four onset "
                     "fractions by five seeds')")
    numbers: dict[str, Any] = {}
    for N, est in deltas.items():
        numbers.update(_estimate_numbers(f"Delta({N:.2f})", est, reading))
    numbers.update({"trend_rho": None if trend is None else trend.rho, "trend_lo": None if trend is None else trend.lo,
                    "trend_hi": None if trend is None else trend.hi,
                    "trend_points": 0 if trend is None else trend.n_points, "trend_points_registered": R.H1_TREND_POINTS,
                    "trend_seeds": 0 if trend is None else trend.n_seeds,
                    "trend_seed_rule": "registered seeds and replacements (Part 5.4: twenty points)",
                    "trend_undefined_resamples": None if trend is None else trend.n_nan,
                    "S1_delta050_positive_excluding_zero": s1, "S2_increases_with_N": s2, "points_ordered": ordered,
                    "trend_positive_excluding_zero": trend_read, "F1_all_intervals_include_zero": f1,
                    "F2_not_ordered": f2, "support": supported, "falsification": falsified,
                    "not_compared_rule6": sorted(unmatched), "incomplete": sorted(incomplete),
                    "holm_survives": holm_ok if holm and top is not None else None,
                    "holm_waits": holm and top is not None and holm_ok is None})
    if box_falsified:  # H1's own keys (Part 5.7 names H1's bound), read as ``bound_annotation`` reads every bound
        numbers.update({"bound_95_upper_Delta050": ann["bound(Delta(0.50))"],
                        "bound_reading": ann["bound(Delta(0.50))_reading"]})
    if trend is not None and trend.flagged:
        notes.append(f"{trend.n_nan} of {trend.resamples} trend resamples were undefined (over {stats.NAN_FLAG_SHARE:.0%})")
    return Outcome(status, reason, numbers=numbers, notes=notes,
                   estimates=[{"contrast": f"Delta({N:.2f})", **est.row()} for N, est in deltas.items()])


def _h1_p(A: StudyA, reading: Reading, task: str, condition: str, control: str, shape: str, estimand: str) -> Member:
    """Delta(0.50) of one cell as a Holm family member (a p-value, not tested, or incomplete)."""
    tested, member = _task_member(A, reading, task, estimand)
    if not tested:
        return member
    est, why = A.contrast(reading, task=task, x=A.arms(task, N=LATE_N, shape=shape, control=control), y=A.reference(task),
                        get=A.gap_outcome(condition, estimand), estimand=estimand, filter_matched=estimand != "final",
                        analysis_id=f"H1|{task}|{condition}|{control}|{shape}|{estimand}|N{LATE_N:.2f}")
    return _member(est, why, f"{task} x {condition}")


def h1_outcome(A: StudyA, reading: Reading, *, task: str = R.PRIMARY_TASK, condition: str = PRIMARY_CONDITION,
               shape: Optional[str] = None, estimand: str = "matched", holm: bool = False) -> Outcome:
    """Box H1 on one (task, condition), both controls combined by Q-controls-reading."""
    A = A.for_reading(reading)
    cut = _cut_task(A, task)
    if cut is not None:
        return cut
    shape = shape or reading.shape
    per_control = {c: _h1_control(A, reading, task=task, condition=condition, control=c, shape=shape,
                                  estimand=estimand, holm=holm) for c in R.STEP_MATCHING_CONTROLS}
    status, reason, numbers, estimates, notes = _combined(per_control, reading)
    numbers = {"task": task, "condition": condition, "shape": shape, "estimand": estimand, **numbers}
    if status == FALSIFIED:
        notes.append("Part 5.7: reported as an upper bound (bound_95_upper_Delta050 of each falsified control) against "
                     f"the minimum effect of {R.MIN_EFFECT_STUDY_A:g} cost units (Part 1.5)")
    return Outcome(status, reason, numbers, estimates, notes)


def _combined(per_control: dict[str, Outcome], reading: Reading
              ) -> tuple[str, str, dict[str, Any], list[dict[str, Any]], list[str]]:
    """Boxes H1 and H2 over the two controls (Q-controls-reading; ``verdict.combine_controls``).

    The per-control numbers, notes and estimates are kept with the control's name. A control's reason
    (why its box is NOT_COMPUTABLE) is the verdict's reason only when the combined box is NOT_COMPUTABLE
    too; when the other control already decides the box, it is a note that says so.
    """
    status = combine_controls({c: o.status for c, o in per_control.items()}, reading)
    numbers: dict[str, Any] = {}
    notes: list[str] = []
    estimates: list[dict[str, Any]] = []
    reasons: list[str] = []
    for control, out in per_control.items():
        numbers[f"{control}.status"] = out.status
        numbers.update({f"{control}.{k}": v for k, v in out.numbers.items()})
        notes.extend(f"{control}: {n}" for n in out.notes)
        estimates.extend({"control": control, **e} for e in out.estimates)
        if out.reason:
            reasons.append(out.reason)
    if status == NOT_COMPUTABLE:
        return status, "; ".join(reasons), numbers, estimates, notes
    for control, out in per_control.items():
        if out.status == NOT_COMPUTABLE:
            why = out.reason.removeprefix(f"under {control}: ")
            notes.append(f"{control}: the box cannot be computed ({why}); the box is {status} whatever it gives "
                         "(Q-controls-reading)")
    return status, "", numbers, estimates, notes


# ---------------------------------------------------------------------------
# H2
# ---------------------------------------------------------------------------


def _h2_p(A: StudyA, reading: Reading, task: str, condition: str, control: str) -> Member:
    """E(0.50) of one cell as a Holm family member (a p-value, not tested, or incomplete)."""
    est, why = A.contrast(reading, task=task, x=A.arms(task, N=LATE_N, shape="abrupt", control=control),
                        y=A.arms(task, N=LATE_N, shape="ramp", control=control), get=A.gap_outcome(condition),
                        analysis_id=f"H2|{task}|{condition}|{control}|N{LATE_N:.2f}")
    return _member(est, why, f"{task} x {condition}")


def _h2_control(A: StudyA, reading: Reading, *, task: str, condition: str, control: str) -> Outcome:
    """Box H2 under one control. Support needs E(0.50) > 0 with an interval excluding zero, Holm-corrected
    (H2 to H4 are secondary on every cell, the primary included; Part 5.2), AND E(0.10) >= 0 AND
    E(0.25) >= 0; falsification is E(0.50) <= 0 or every interval including zero. An E(N) that cannot
    be computed (rule 6, incomplete data) leaves the clauses that need it open, so the box is decided
    only where the known clauses decide it; the recorded Part 6.1 cut "drop_n010" alone removes
    N = 0.10 from both clauses. As box H1's, the numbers say what the box still waits on
    (``incomplete``, ``holm_waits``; ``could_be_falsified``): without E(0.50) no clause can be read, and
    a box whose E(0.50) is left out under rule 6, or whose task is infeasible, is NOT_COMPUTABLE for good."""
    feasible, why = A.task_state(reading, task, "matched")
    if not feasible:
        return Outcome(NOT_COMPUTABLE, f"{task}: {why}",
                       numbers={"incomplete": [], "holm_waits": False} if feasible is False else {})
    get = A.gap_outcome(condition)
    diffs: dict[float, stats.TwoArmEstimate] = {}
    missing: dict[float, str] = {}
    notes: list[str] = []
    cut_early = CUT_DROP_N010 in A.dataset.cuts
    for N in R.LATE_ONSET_FRACTIONS:
        abrupt, ramp = A.arms(task, N=N, shape="abrupt", control=control), A.arms(task, N=N, shape="ramp", control=control)
        analysis_id = f"H2|{task}|{condition}|{control}|N{N:.2f}"
        est, why = A.contrast(reading, task=task, x=abrupt, y=ramp, get=get, analysis_id=analysis_id)
        if est is None:
            if N == LATE_N:  # data still to come, or left out for good (rule 6; no arm)
                return Outcome(NOT_COMPUTABLE, f"E({N:.2f}) under {control}: {why}",
                               numbers={"incomplete": [N] if why.startswith("incomplete") else [], "holm_waits": False})
            if cut_early and N == min(R.LATE_ONSET_FRACTIONS):
                notes.append(f"E({N:.2f}) is not read: the recorded Part 6.1 cut 'drop_n010' removed the N = {N:.2f} "
                             "arms ('Drop N = 0.10 from the main sweep'); the clauses are read at the fractions that remain")
            else:
                missing[N] = why
                notes.append(f"E({N:.2f}) cannot be computed ({why}): the clauses that need it stay open")
            continue
        diffs[N] = est
        notes += A.age_note(reading, abrupt, ramp, analysis_id)  # Part 4.1.1 (Q-training-age-systematic)
    top = diffs[LATE_N]
    lo, hi = top.interval(reading.interval)
    holm_ok, holm_why = A.holm_cell(reading, f"H2|{control}", lambda t, c: _h2_p(A, reading, t, c, control),
                                    (task, condition))
    top_ok = _and3(settled(top.diff) > 0 and lo > 0, holm_ok)
    others_nonnegative = all(settled(diffs[N].diff) >= 0 for N in diffs if N != LATE_N)
    supported = _and3(top_ok, others_nonnegative, None if missing else True)
    if settled(top.diff) <= 0:
        falsified: Optional[bool] = True
    elif not all(_includes_zero(*d.interval(reading.interval)) for d in diffs.values()):
        falsified = False
    else:
        falsified = None if missing else True
    status = box_status(supported, falsified, reading, notes)
    reason = ""
    if status == NOT_COMPUTABLE:
        parts = []
        if missing:
            parts.append("; ".join(f"E({N:.2f}): {w}" for N, w in sorted(missing.items()))
                         + " (box H2 needs E at N = 0.10 and 0.25: 'non-negative at N = 0.10 and 0.25', 'at every N'; "
                           "only a recorded Part 6.1 cut removes a fraction)")
        if holm_ok is None:
            parts.append(f"E({LATE_N:.2f}): {holm_why}")
        reason = f"under {control}: " + "; ".join(parts)
    numbers: dict[str, Any] = {"holm_survives": holm_ok, "holm_waits": holm_ok is None, "support": supported,
                               "falsification": falsified, "missing": sorted(missing),
                               "incomplete": sorted(N for N, w in missing.items() if w.startswith("incomplete"))}
    bounded = None
    if status == FALSIFIED:
        ann, bounded = bound_annotation(f"E({LATE_N:.2f}), 95% Welch", top.welch.hi, R.MIN_EFFECT_STUDY_A,
                                        claimed="positive")
        numbers.update(ann)
    status = calibrate(status, bounded, reading, notes)
    if status == SUPPORTED:
        notes += detectable_notes({f"E({LATE_N:.2f})": top})
    for N, est in diffs.items():
        numbers.update(_estimate_numbers(f"E({N:.2f})", est, reading))
    return Outcome(status, reason, numbers, notes=notes,
                   estimates=[{"contrast": f"E({N:.2f}) = gap(abrupt) - gap(ramp)", **e.row()} for N, e in diffs.items()])


def h2_outcome(A: StudyA, reading: Reading, *, task: str = R.PRIMARY_TASK, condition: str = PRIMARY_CONDITION) -> Outcome:
    """Box H2 on one (task, condition), both controls combined by Q-controls-reading."""
    A = A.for_reading(reading)
    cut = _cut_task(A, task)
    if cut is not None:
        return cut
    per_control = {c: _h2_control(A, reading, task=task, condition=condition, control=c) for c in R.STEP_MATCHING_CONTROLS}
    status, reason, numbers, estimates, notes = _combined(per_control, reading)
    return Outcome(status, reason, {"task": task, "condition": condition, **numbers}, estimates, notes)


# ---------------------------------------------------------------------------
# H3
# ---------------------------------------------------------------------------


H3A_GROUPS = {"main": ("main",), "all_late": ("main", "treatment", "controller", "pid")}  # Q-h3-scope's two readings


@functools.lru_cache(maxsize=None)
def _expected_late_arms(task: str, all_late: bool, cuts: tuple[str, ...]) -> tuple[tuple, ...]:
    """The late arms of ``task`` that H3 (a) reads (Q-h3-scope), as factor tuples (N, shape, control,
    treatment, controller): every registered arm of the design groups, less the recorded Part 6.1 cuts
    (``pilot.manifest.apply_cuts``), so a cut arm is not expected and a missing one is noticed."""
    from pilot.manifest import apply_cuts, design

    specs = apply_cuts([s for g in H3A_GROUPS["all_late" if all_late else "main"] for s in design(g)], cuts)
    return tuple(sorted({(s.N, s.onset_shape, s.step_matching, s.treatment, s.controller_variant) for s in specs
                         if s.study == "A" and s.task == task and s.N in R.LATE_ONSET_FRACTIONS}, key=repr))


def _late_arm_records(A: StudyA, reading: Reading, task: str, all_late: bool
                      ) -> tuple[list[dict[str, Any]], dict[str, str], list[str]]:
    """(records of the compared late arms, {arm left out under rule 6: status}, [incomplete arms, why]).

    Every expected late arm of the task is looked for (``_expected_late_arms``). Q-h3-scope's "matched arms
    only" leaves out an arm that rule 6 does not compare (recorded in the numbers); an arm whose matching
    is still undecided (``matching.STATUS_INCOMPLETE``) or that has no completed run is incomplete data,
    never a silent drop (as box H1: the correlation is then NOT_COMPUTABLE).
    """
    records: list[dict[str, Any]] = []
    left_out: dict[str, str] = {}
    incomplete: list[str] = []
    for N, shape, control, treatment, controller in _expected_late_arms(task, all_late, tuple(A.dataset.cuts)):
        arm = A.arms(task, N=N, shape=shape, control=control, treatment=treatment, controller=controller)
        if arm is None or not arm.records:
            factors = ", ".join(str(f) for f in (shape, control, treatment, controller) if f is not None)
            incomplete.append(f"N = {N:.2f} ({factors}): no completed run")
            continue
        if arm.awaited:
            incomplete.append(arm.waiting)
            continue
        status = A.arm_status(reading, "matched", arm)
        if status == matching.STATUS_MATCHED:
            records += arm.records
        elif status in matching.UNMATCHED_STATUSES:
            left_out[arm.arm_id] = status
        else:
            incomplete.append(f"{arm.arm_id}: cannot be matched yet ({status})")
    return records, left_out, incomplete


def _h3a_points(A: StudyA, reading: Reading, task: str, condition: str, metric: str
                ) -> tuple[Optional[dict], list[str], dict[str, str]]:
    """(points per seed, or None while anything is incomplete; what is incomplete; arms left out under rule 6).

    The arms compared are those matched on every completed seed; the points are their runs of the registered
    seeds and their replacements only (``analysis.data.five_seed_view``; Q-surplus-in-analysis, answered):
    H3 (a)'s correlation is read "with the same interval" as the H1 trend, resampling seed indices that
    each carry all their levels, and surplus seeds exist only at N = 0.50 (and N = 0), so they would leave seed
    indices without their N = 0.10 and N = 0.25 points.
    """
    recs, left_out, incomplete = _late_arm_records(A, reading, task, reading.h3_arms == "all_late")
    five = {rec["run_id"] for rec in A.for_reading(replace(reading, seeds="five")).records}
    recs = [rec for rec in recs if rec["run_id"] in five]
    points: dict[int, list[tuple[float, float]]] = {}
    missing = []
    for rec in recs:
        x, y = rec.get(metric), A.gap(rec, condition, "matched")
        if x is None or y is None or not math.isfinite(float(x)) or not math.isfinite(float(y)):
            missing.append(rec["run_id"])  # a non-finite value is missing data, as in ``StudyA.values``
            continue
        points.setdefault(int(rec["seed"]), []).append((float(x), float(y)))
    if missing:
        incomplete.append(f"{metric} or the gap missing for {missing}")
    return (None if incomplete else points), incomplete, left_out


def _h3a_constant(points: dict[int, list[tuple[float, float]]], axis: int) -> bool:
    """True when every point has the same value (as compared: ``settled``) on ``axis`` (0 metric, 1 gap): its
    Spearman rho is undefined (``stats.spearman``)."""
    return len({settled(p[axis]) for ps in points.values() for p in ps}) <= 1


def _h3a_member(A: StudyA, reading: Reading, task: str, condition: str, metric: str) -> Member:
    """One cell of an H3 (a) Holm family: the correlation's bootstrap p-value; p = 1 for a metric constant over the
    late runs (no monotone association: tested, never surviving); None (not tested, the family shrinks) when the
    gap itself is constant or every resample is undefined, the cases ``h3a_outcome`` reports NOT_COMPUTABLE with
    that reason; ``stats.Incomplete`` while data are still to come."""
    tested, member = _task_member(A, reading, task, "matched")
    if not tested:
        return member
    points, incomplete, _ = _h3a_points(A, reading, task, condition, metric)
    if points is None:
        return stats.Incomplete(f"{task} x {condition}: incomplete: " + "; ".join(incomplete))
    if sum(len(p) for p in points.values()) < 3 or _h3a_constant(points, 1):
        return None
    if _h3a_constant(points, 0):
        return 1.0
    r = A.rho(points, f"H3a|{task}|{condition}|{metric}|{reading.h3_arms}")
    return None if math.isnan(r.p_boot) else r.p_boot


def h3a_outcome(A: StudyA, reading: Reading, *, task: str = R.PRIMARY_TASK, condition: str = PRIMARY_CONDITION,
                holm: bool = True) -> Outcome:
    """H3 (a): "the dormant-neuron fraction correlates positively and the effective rank negatively with
    the gap, each with an interval excluding zero" / falsified: "No monotone association".

    A metric constant over the late runs (a dormant fraction of 0 in every run, say) has no monotone
    association with the gap: its rho is undefined, its support clause fails and its falsification clause
    holds (it counts as an interval including zero), so the other metric's correlation still decides the
    box; its Holm member is p = 1 (``_h3a_member``). An implementation reading within Q-h3-scope (its answer
    does not name a constant metric), as H3 (c) reads a zero-variance contrast as a failed check. Only a
    constant gap, or a correlation whose every resample is undefined, leaves the box NOT_COMPUTABLE.
    """
    A = A.for_reading(reading)
    cut = _cut_task(A, task)
    if cut is not None:
        return cut
    feasible, why = A.task_state(reading, task, "matched")
    if not feasible:
        return Outcome(NOT_COMPUTABLE, f"{task}: {why}")
    results = {}
    constant: dict[str, bool] = {}
    left_out: dict[str, str] = {}
    for metric in ("dormant_onset", "rank_onset"):
        points, incomplete, left_out = _h3a_points(A, reading, task, condition, metric)
        if points is None:
            return Outcome(NOT_COMPUTABLE, "incomplete: " + "; ".join(incomplete)
                           + " (the late arms are read once every one is matched or left out under rule 6; a "
                             "missing field makes the analysis incomplete, never a silent drop)",
                           numbers={"not_compared_rule6": sorted(left_out)})
        if sum(len(p) for p in points.values()) < 3:
            return Outcome(NOT_COMPUTABLE, "fewer than three late-arm runs with a gap")
        if _h3a_constant(points, 1):
            return Outcome(NOT_COMPUTABLE, "the gap is constant over the late runs: no correlation is defined")
        constant[metric] = _h3a_constant(points, 0)
        results[metric] = A.rho(points, f"H3a|{task}|{condition}|{metric}|{reading.h3_arms}")
    dorm, rank = results["dormant_onset"], results["rank_onset"]
    if any(not constant[m] and (math.isnan(r.rho) or math.isnan(r.lo)) for m, r in results.items()):
        return Outcome(NOT_COMPUTABLE, "a correlation is undefined in every bootstrap resample")
    holm_ok: Optional[bool] = True
    holm_whys: list[str] = []
    if holm:
        for metric in ("dormant_onset", "rank_onset"):
            ok, why_h = A.holm_cell(
                reading, f"H3a|{metric}|{reading.h3_arms}",
                lambda t, c, m=metric: _h3a_member(A, reading, t, c, m), (task, condition))
            holm_ok = _and3(holm_ok, ok)
            if why_h:
                holm_whys.append(f"{metric}: {why_h}")
    notes: list[str] = []
    for metric, name in (("dormant_onset", "dormant fraction"), ("rank_onset", "effective rank")):
        if constant[metric]:
            notes.append(f"the {name} at onset is constant over the late runs: no monotone association with the gap "
                         "(its support clause fails, its falsification clause holds; Q-h3-scope)")
    associated = {m: not constant[m] for m in results}
    supported = _and3(associated["dormant_onset"] and associated["rank_onset"] and dorm.rho > 0 and dorm.lo > 0
                      and rank.rho < 0 and rank.hi < 0, holm_ok)
    falsified = all(constant[m] or _includes_zero(r.lo, r.hi) for m, r in results.items())
    for name, r, sign in (("dormant fraction", dorm, 1), ("effective rank", rank, -1)):
        if _excludes_zero(r.lo, r.hi) and (r.rho * sign) < 0:
            notes.append(f"the {name} correlation excludes zero in the direction contrary to H3 (a); "
                         "it neither supports nor falsifies (Q-h3-scope)")
    if left_out:
        notes.append(f"late arms left out under Part 4.1 rule 6 ('It is not compared'; Q-h3-scope): {sorted(left_out)}")
    status = box_status(supported, falsified, reading, notes)
    numbers = {"task": task, "condition": condition, "arms": reading.h3_arms,
               "rho_dormant": dorm.rho, "rho_dormant_lo": dorm.lo, "rho_dormant_hi": dorm.hi, "p_dormant": dorm.p_boot,
               "rho_rank": rank.rho, "rho_rank_lo": rank.lo, "rho_rank_hi": rank.hi, "p_rank": rank.p_boot,
               "points": dorm.n_points, "seeds": dorm.n_seeds, "holm_survives": holm_ok if holm else None,
               "not_compared_rule6": sorted(left_out), "support": supported, "falsification": falsified}
    return Outcome(status, "; ".join(holm_whys) if status == NOT_COMPUTABLE else "", numbers=numbers, notes=notes)


def _h3b_estimates(A: StudyA, reading: Reading, task: str, condition: str
                   ) -> tuple[dict[str, Optional[stats.TwoArmEstimate]], dict[str, str], list[str]]:
    """{t: D(t) or None} for the three treatments, {t: why D(t) is missing}, and the Part 4.1.1 notes of the
    D(t) whose arms' training ages differ systematically (``StudyA.age_note``)."""
    untreated = A.arms(task, N=LATE_N, shape=TREATMENT_SHAPE, control=TREATMENT_CONTROL)
    get = A.gap_outcome(condition)
    out: dict[str, Optional[stats.TwoArmEstimate]] = {}
    reasons: dict[str, str] = {}
    ages: list[str] = []
    for t in R.TREATMENTS:
        treated = A.arms(task, N=LATE_N, shape=TREATMENT_SHAPE, control=TREATMENT_CONTROL, treatment=t)
        analysis_id = f"H3b|{task}|{condition}|{t}"
        est, why = A.contrast(reading, task=task, x=treated, y=untreated, get=get, analysis_id=analysis_id)
        out[t] = est
        if est is None:
            reasons[t] = why
        else:
            ages += A.age_note(reading, treated, untreated, analysis_id)
    return out, reasons, ages


def _h3b_member(A: StudyA, reading: Reading, task: str, condition: str, t: str) -> Member:
    """D(t) of one cell as a Holm family member (a p-value, not tested, or incomplete)."""
    est, why = A.contrast(reading, task=task, get=A.gap_outcome(condition),
                          x=A.arms(task, N=LATE_N, shape=TREATMENT_SHAPE, control=TREATMENT_CONTROL, treatment=t),
                          y=A.arms(task, N=LATE_N, shape=TREATMENT_SHAPE, control=TREATMENT_CONTROL),
                          analysis_id=f"H3b|{task}|{condition}|{t}")
    return _member(est, why, f"{task} x {condition}")


def h3b_outcome(A: StudyA, reading: Reading, *, task: str = R.PRIMARY_TASK, condition: str = PRIMARY_CONDITION,
                holm: bool = True) -> Outcome:
    """H3 (b): "Reset and injection each reduce the gap of the N = 0.50 arm with an interval excluding zero,
    and additional constrained training does not" / falsified: "additional constrained training reduces
    the gap as much as reset; or reset reduces it and injection does not; or none of the three reduces it".

    D(t) = gap(t) - gap(untreated). A claim that t reduces the gap (reset, injection; "reset reduces it"
    and "additional constrained training reduces the gap" in the falsification) is D(t) < 0 with its
    interval excluding zero AND Holm in its families (H2 to H4 are secondary on every cell, the primary
    included; Part 5.2). An absence ("additional constrained training does not", "injection does not",
    "none of the three reduces it") is read on the UNcorrected interval (Q-h3-scope: "The absences in
    (b), '... and additional constrained training does not' [reduce the gap], 'injection does not' and
    'none of the three reduces it', mean no reduction with an uncorrected 95 percent interval excluding
    zero"; its other reading: the point estimate is not negative): Holm makes a claim of reduction
    harder and must not make an absence easier.

    "Additional constrained training reduces the gap as much as reset" needs both halves: additional
    training reduces the gap (the claim above) AND its point estimate is at most reset's (a reduction
    at least as large). The analysis spec's formula ``b_fal = [D̂_add <= D̂_reset] or ...`` drops the
    first half, so an additional training that widens the gap less than reset widens it would falsify
    the box although it reduces nothing.

    A D(t) that cannot be computed (an arm not compared under rule 6, or data still to come) leaves the
    conditions that name t unknown (three-valued, as box H1's): the clauses the other estimates decide
    still decide the box ("reset reduces it and injection does not" falsifies it whatever additional
    constrained training gives), and only what the missing D(t) could change is NOT_COMPUTABLE, with
    its reason. So is a falsified box whose known bounds do not confirm it while a falsifying clause
    that could, with its own Part 5.7 bound, waits on data still to come (Q-falsification-calibration).
    """
    A = A.for_reading(reading)
    cut = _cut_task(A, task)
    if cut is not None:
        return cut
    feasible, why = A.task_state(reading, task, "matched")
    if not feasible:
        return Outcome(NOT_COMPUTABLE, f"{task}: {why}")
    ests, missing, age_notes = _h3b_estimates(A, reading, task, condition)
    missing_why = "; ".join(f"D({t}): {why}" for t, why in missing.items())
    survival: dict[str, Optional[bool]] = {}
    holm_why: dict[str, str] = {}
    if holm:
        for t in R.TREATMENTS:
            survival[t], why_h = A.holm_cell(reading, f"H3b|{t}", lambda tk, c, t=t: _h3b_member(A, reading, tk, c, t),
                                             (task, condition))
            if why_h:
                holm_why[t] = why_h
    holm_whys = [f"D({t}): {why}" for t, why in holm_why.items()]

    def reduces_uncorrected(t: str) -> Optional[bool]:  # None: D(t) cannot be computed
        est = ests[t]
        if est is None:
            return None
        _, hi = est.interval(reading.interval)
        return settled(est.diff) < 0 and hi < 0

    def reduces(t: str) -> Optional[bool]:  # a claim of reduction (None: D(t) missing, or Holm waits on a member)
        return None if ests[t] is None else _and3(reduces_uncorrected(t), survival.get(t, True))

    def does_not_reduce(t: str) -> Optional[bool]:  # an absence (Q-h3-scope)
        if ests[t] is None:
            return None
        return not reduces_uncorrected(t) if reading.h3_reduce == "interval" else settled(ests[t].diff) >= 0

    add, reset, inj = ADDITIONAL, RESET, INJECTION  # short names: the clauses and the Part 5.7 bounds read them often
    supported = _and3(reduces(reset), reduces(inj), does_not_reduce(add))
    # "additional constrained training reduces the gap as much as reset": it must reduce the gap at all
    at_least_reset = (None if ests[add] is None or ests[reset] is None
                      else settled(ests[add].diff) <= settled(ests[reset].diff))
    as_much = _and3(reduces(add), at_least_reset)
    reset_not_inj = _and3(reduces(reset), does_not_reduce(inj))
    none = _and3(*(does_not_reduce(t) for t in R.TREATMENTS))
    falsified = _or3(as_much, reset_not_inj, none)
    notes: list[str] = list(age_notes)
    status = box_status(supported, falsified, reading, notes)
    reason = "; ".join(([missing_why] if missing_why else []) + holm_whys) if status == NOT_COMPUTABLE else ""
    numbers: dict[str, Any] = {"task": task, "condition": condition, "support": supported,
                               "falsification": falsified, "F_additional_reduces_as_much_as_reset": as_much,
                               "F_reset_reduces_injection_does_not": reset_not_inj, "F_none_reduces": none}
    for t, est in ests.items():
        numbers.update(_estimate_numbers(f"D({t})", est, reading))
        numbers[f"reduces({t})"] = reduces(t)
        numbers[f"reduces_uncorrected({t})"] = reduces_uncorrected(t)
        numbers[f"does_not_reduce({t})"] = does_not_reduce(t)
        numbers[f"holm_survives({t})"] = survival.get(t) if survival else None
    add_reset = A.contrast(reading, task=task,
                           x=A.arms(task, N=LATE_N, shape=TREATMENT_SHAPE, control=TREATMENT_CONTROL, treatment=add),
                           y=A.arms(task, N=LATE_N, shape=TREATMENT_SHAPE, control=TREATMENT_CONTROL, treatment=reset),
                           get=A.gap_outcome(condition), analysis_id=f"H3b|{task}|{condition}|add-reset")[0]
    numbers.update(_estimate_numbers("gap(additional_constrained) - gap(reset)", add_reset, reading))
    bounded = None
    if status == FALSIFIED:
        # Part 5.7 per falsifying clause: the effect H3 (b) claims there, bounded below the minimum effect?
        flags = []
        if as_much:
            ann, b = bound_annotation("gap(additional_constrained) - gap(reset), 95% Welch",
                                      None if add_reset is None else add_reset.welch.hi, R.MIN_EFFECT_STUDY_A,
                                      claimed="positive")
            numbers.update(ann)
            flags.append(b)
        claimed = {t: bound_annotation(f"D({t}), 95% Welch", None if ests[t] is None else ests[t].welch.lo,
                                       R.MIN_EFFECT_STUDY_A, claimed="negative") for t in (reset, inj)}
        if reset_not_inj:  # injection's reduction is what is bounded
            numbers.update(claimed[inj][0])
            flags.append(claimed[inj][1])
        if none:  # both claimed reductions must be bounded
            for t in (reset, inj):
                numbers.update(claimed[t][0])
            flags.append(bool(claimed[reset][1]) and bool(claimed[inj][1]))
        bounded = any(bool(f) for f in flags)
        # a falsifying clause not yet evaluable because a D(t) it reads, or that D(t)'s Holm survival, is still to
        # come could confirm the falsification with its own bound once the data arrive; rule 6 never brings them
        later = {t: why for t, why in ((t, missing.get(t) or holm_why.get(t, "")) for t in R.TREATMENTS)
                 if why.startswith("incomplete")}
        clauses = ((as_much, (add, reset)), (reset_not_inj, (reset, inj)), (none, R.TREATMENTS))
        pending = sorted({f"D({t}): {later[t]}" for clause, reads in clauses if clause is None
                          for t in reads if t in later})
        if pending and not bounded and reading.calibration == "bound":
            status = NOT_COMPUTABLE
            reason = ("; ".join(pending) + " (a falsifying clause that reads it, with its own Part 5.7 bound, could "
                      "still confirm the falsification: Q-falsification-calibration)")
    status = calibrate(status, bounded, reading, notes)
    if missing_why and status != NOT_COMPUTABLE:
        notes.append(f"decided without {missing_why}: the clauses the known estimates decide settle the box")
    if status == SUPPORTED:
        notes += detectable_notes({f"D({reset})": ests[reset], f"D({inj})": ests[inj]})
    estimates = [{"contrast": f"D({t}) = gap({t}) - gap(untreated)", **e.row()} for t, e in ests.items() if e is not None]
    return Outcome(status, reason, numbers=numbers, notes=notes, estimates=estimates)


H3C_METRICS = {"dormant": "dormant_trainable_check", "rank": "rank_trainable_check"}  # the training records' fields


def _h3c_contrast(A: StudyA, reading: Reading, task: str, treatment: str, metric: str
                  ) -> tuple[Optional[stats.TwoArmEstimate], str]:
    """One claim of H3 (c) on one task: the treated minus the untreated N = 0.50 arm (no matching filter)."""
    field_name = H3C_METRICS[metric]
    return A.contrast(reading, task=task, get=lambda r: r.get(field_name), filter_matched=False,
                      x=A.arms(task, N=LATE_N, shape=TREATMENT_SHAPE, control=TREATMENT_CONTROL, treatment=treatment),
                      y=A.arms(task, N=LATE_N, shape=TREATMENT_SHAPE, control=TREATMENT_CONTROL),
                      analysis_id=f"H3c|{task}|{treatment}|{metric}")


def _h3c_survival(A: StudyA, reading: Reading, treatment: str, metric: str) -> dict[str, tuple[Optional[bool], str]]:
    """Holm (alpha = 0.05) over the three tasks for one claim of H3 (c): {task: (survives, why undecided)}
    (None: not tested there, or waiting on a task whose data are incomplete: the family never shrinks on
    data still to come; ``stats.holm_decisions``).

    Part 5.2: "A claim about a secondary outcome is corrected for multiplicity within its family (... the
    three tasks another ...)"; H3 is secondary. The check is made on the trainable layers during
    training, not under a battery condition, so the task family is its only one (Q-holm-families:
    "Claims with a single family (... H3 (c) over the three tasks ...) use that family"; "Families shrink
    only to cells not tested"). A task that rule 3 makes infeasible ("enters no comparison") or rule 5
    does ("contributes no comparison"; Part 4.1) leaves the family; one whose matching is undecided is
    incomplete (``_task_member``, as every other Study A family).
    """
    key = ("H3c-holm", treatment, metric, reading.cost_field, reading.arithmetic)
    if key not in A._grids:
        ps: dict[str, Member] = {}
        for task in R.TASKS_STUDY_A:
            tested, member = _task_member(A, reading, task, "matched")
            if not tested:
                ps[task] = member
                continue
            est, why = _h3c_contrast(A, reading, task, treatment, metric)
            ps[task] = _member(est, why, task)
        _, survives, incomplete = stats.holm_decisions(ps, R.ALPHA)
        waiting = "; ".join(incomplete.values())
        A._grids[key] = {task: (survives.get(task), holm_waiting(waiting) if survives.get(task) is None and waiting
                                else "") for task in R.TASKS_STUDY_A}
    return A._grids[key]


def h3c_outcome(A: StudyA, reading: Reading, *, task: str = R.PRIMARY_TASK, holm: bool = True) -> Outcome:
    """H3 (c): at onset + 200,000 steps the reset and injection arms have a lower dormant fraction of the
    trainable units and a higher effective rank of the trainable layers than the untreated late arm,
    with intervals excluding zero. No matching filter on the arms: a manipulation check, not an outcome;
    but a task that Part 4.1 rule 3 ("enters no comparison") or rule 5 ("contributes no comparison") makes
    infeasible is NOT_COMPUTABLE with the rule's reason, and one whose matching is undecided waits.

    Each of the four claims (dormant lower, rank higher; reset, injection) is a claim of an effect and
    is Holm-corrected over the three tasks (``_h3c_survival``); the falsifying absence, "the
    manipulation check fails for both interventions", is read on the uncorrected intervals, as every
    absence in this module (a Holm correction that removes a claim is absence of evidence).
    """
    A = A.for_reading(reading)
    cut = _cut_task(A, task)
    if cut is not None:
        return cut
    feasible, why_task = A.task_state(reading, task, "matched")
    if feasible is False:
        return Outcome(NOT_COMPUTABLE, f"{task}: {why_task} (Part 4.1 rules 3 and 5: the task enters no comparison)")
    if feasible is None and any(r["task"] == task for r in A.records):
        return Outcome(NOT_COMPUTABLE, f"incomplete: matching of {task} cannot be decided yet ({why_task})")
    numbers: dict[str, Any] = {"task": task, "step_after_onset": R.MANIPULATION_CHECK_STEPS_AFTER_ONSET}
    passes: dict[str, Optional[bool]] = {}
    passes_uncorrected: dict[str, bool] = {}
    holm_whys: list[str] = []
    estimates = []
    deciding: dict[str, stats.TwoArmEstimate] = {}
    for t in (RESET, INJECTION):
        dorm, why_d = _h3c_contrast(A, reading, task, t, "dormant")
        rank, why_r = _h3c_contrast(A, reading, task, t, "rank")
        if dorm is None or rank is None:
            why = why_d or why_r
            if why.startswith("incomplete: no value for"):  # the training record's metric is missing
                why += (f" (the trainable-layer metrics at onset + {R.MANIPULATION_CHECK_STEPS_AFTER_ONSET:,} steps "
                        "come from the training records, which `enrich training` writes"
                        + ("; it waits on Q-controller-quantities while that is open)"
                           if R.is_open("Q-controller-quantities") else ")"))
            return Outcome(NOT_COMPUTABLE, f"{t}: {why}")
        dlo, dhi = dorm.interval(reading.interval)
        rlo, rhi = rank.interval(reading.interval)
        passes_uncorrected[t] = settled(dorm.diff) < 0 and dhi < 0 and settled(rank.diff) > 0 and rlo > 0
        survives: Optional[bool] = None
        if holm:  # the task family is the claim's only one: Holm under every reading of Q-holm-families
            per_metric = [_h3c_survival(A, reading, t, m)[task] for m in ("dormant", "rank")]
            # undecided (None) while an incomplete task could change it; a claim Holm did not test does not survive
            survives = _and3(*[ok if ok is not None else (None if why else False) for ok, why in per_metric])
            holm_whys += [f"{t}: {why}" for _, why in per_metric if why]
        passes[t] = _and3(passes_uncorrected[t], survives) if holm else passes_uncorrected[t]
        numbers.update(_estimate_numbers(f"dormant_trainable({t}) - untreated", dorm, reading))
        numbers.update(_estimate_numbers(f"rank_trainable({t}) - untreated", rank, reading))
        numbers[f"passes({t})"] = passes[t]
        numbers[f"passes_uncorrected({t})"] = passes_uncorrected[t]
        numbers[f"holm_survives({t})"] = survives
        deciding.update({f"dormant_trainable({t}) - untreated": dorm, f"rank_trainable({t}) - untreated": rank})
        estimates += [{"contrast": f"dormant_trainable({t}) - untreated", **dorm.row()},
                      {"contrast": f"rank_trainable({t}) - untreated", **rank.row()}]
    notes: list[str] = []
    supported, falsified = _and3(*passes.values()), not any(passes_uncorrected.values())
    numbers.update(support=supported, falsification=falsified)
    status = box_status(supported, falsified, reading, notes)
    if status == SUPPORTED:
        notes += detectable_notes(deciding)
    return Outcome(status, "; ".join(holm_whys) if status == NOT_COMPUTABLE else "", numbers=numbers, notes=notes,
                   estimates=estimates)


# ---------------------------------------------------------------------------
# H4 and H0
# ---------------------------------------------------------------------------


def _h4_estimates(A: StudyA, reading: Reading, task: str, condition: str, variant: str = WARM
                  ) -> tuple[Optional[stats.TwoArmEstimate], Optional[stats.TwoArmEstimate], dict[str, str], list[str]]:
    """(F, G, {name: why} of the missing ones, Part 4.1.1 notes) for a controller variant: F = gap(variant) -
    gap(untreated N = 0.50), G = gap(variant) - gap(N = 0); a note for each whose arms' training ages differ
    systematically (``StudyA.age_note``)."""
    get = A.gap_outcome(condition)
    treated = A.arms(task, N=LATE_N, shape=TREATMENT_SHAPE, control=TREATMENT_CONTROL, controller=variant)
    ages: list[str] = []
    found: dict[str, Optional[stats.TwoArmEstimate]] = {}
    whys: dict[str, str] = {}
    for name, y in (("F", A.arms(task, N=LATE_N, shape=TREATMENT_SHAPE, control=TREATMENT_CONTROL)),
                    ("G", A.reference(task))):
        analysis_id = f"H4|{task}|{condition}|{variant}|{name}"
        found[name], why = A.contrast(reading, task=task, x=treated, y=y, get=get, analysis_id=analysis_id)
        if found[name] is None:
            whys[name] = why
        else:
            ages += A.age_note(reading, treated, y, analysis_id)
    return found["F"], found["G"], whys, ages


def _h4_member(A: StudyA, reading: Reading, task: str, condition: str, name: str) -> Member:
    """F or G of one cell (warm-started multiplier) as a Holm family member."""
    get = A.gap_outcome(condition)
    treated = A.arms(task, N=LATE_N, shape=TREATMENT_SHAPE, control=TREATMENT_CONTROL, controller=WARM)
    y = (A.arms(task, N=LATE_N, shape=TREATMENT_SHAPE, control=TREATMENT_CONTROL) if name == "F" else A.reference(task))
    est, why = A.contrast(reading, task=task, x=treated, y=y, get=get,
                          analysis_id=f"H4|{task}|{condition}|{WARM}|{name}")
    return _member(est, why, f"{task} x {condition}")


def h4_outcome(A: StudyA, reading: Reading, *, task: str = R.PRIMARY_TASK, condition: str = PRIMARY_CONDITION,
               holm: bool = True) -> Outcome:
    """H4 with the warm-started multiplier: F = gap(warm) - gap(untreated N = 0.50) must fall (F < 0,
    interval excluding zero) and G = gap(warm) - gap(N = 0) stay above zero (interval excluding zero).
    Falsified if G's interval includes zero, or the gap does not fall (Q-h4-reading: F >= 0 on the
    point estimate; the other reading: F does not fall with an interval excluding zero, i.e. F >= 0 or F's
    interval includes zero). Holm (H2 to H4 are secondary on every cell, the primary included; Part 5.2)
    corrects the two claims of an effect only; the falsifying absences are read on the uncorrected
    intervals.

    Three-valued, as H3 (b): an F or G that cannot be computed (an arm not compared under rule 6, or
    data still to come) leaves only the clauses that read it unknown, so a G whose interval includes
    zero falsifies the box whatever F would be; the box is NOT_COMPUTABLE, with the missing contrast's
    reason, only when the known estimate cannot decide it, or when the known bound does not confirm the
    falsification while the other falsifying clause, which could with its own Part 5.7 bound, waits on
    data still to come (Q-falsification-calibration)."""
    A = A.for_reading(reading)
    cut = _cut_task(A, task)
    if cut is not None:
        return cut
    feasible, why = A.task_state(reading, task, "matched")
    if not feasible:
        return Outcome(NOT_COMPUTABLE, f"{task}: {why}")
    F, G, missing, age_notes = _h4_estimates(A, reading, task, condition)
    missing_why = "; ".join(f"{name}: {why}" for name, why in missing.items())
    f_ok: Optional[bool] = True
    g_ok: Optional[bool] = True
    holm_whys: list[str] = []
    if holm:
        for name in ("F", "G"):
            ok, why_h = A.holm_cell(reading, f"H4|{name}", lambda t, c, name=name: _h4_member(A, reading, t, c, name),
                                    (task, condition))
            f_ok, g_ok = (ok, g_ok) if name == "F" else (f_ok, ok)
            if why_h:
                holm_whys.append(f"{name}: {why_h}")
    # three-valued (as H3 (b)): a contrast that cannot be computed leaves the clauses that read it unknown
    falls = no_fall = above = g_includes_zero = None
    if F is not None:
        flo, fhi = F.interval(reading.interval)
        falls = _and3(settled(F.diff) < 0 and fhi < 0, f_ok)
        no_fall = settled(F.diff) >= 0 if reading.h4_fall == "point" else not (settled(F.diff) < 0 and fhi < 0)
    if G is not None:
        glo, ghi = G.interval(reading.interval)
        above = _and3(settled(G.diff) > 0 and glo > 0, g_ok)
        g_includes_zero = _includes_zero(glo, ghi)
    supported = _and3(falls, above)
    falsified = _or3(g_includes_zero, no_fall)
    notes: list[str] = list(age_notes)
    status = box_status(supported, falsified, reading, notes)
    reason = "; ".join(([missing_why] if missing_why else []) + holm_whys) if status == NOT_COMPUTABLE else ""
    numbers: dict[str, Any] = {"task": task, "condition": condition, "falls": falls, "remains_above": above,
                               "does_not_fall": no_fall, "G_interval_includes_zero": g_includes_zero,
                               "support": supported, "falsification": falsified}
    bounded = None
    if status == FALSIFIED:
        flags = []
        if g_includes_zero:  # "overshoot explains all of the gap": the remaining gap above N = 0, bounded?
            ann, b = bound_annotation("G, 95% Welch", G.welch.hi, R.MIN_EFFECT_STUDY_A, claimed="positive")
            numbers.update(ann)
            flags.append(b)
        if no_fall:  # "overshoot explains none of it": the fall, bounded?
            ann, b = bound_annotation("F, 95% Welch", F.welch.lo, R.MIN_EFFECT_STUDY_A, claimed="negative")
            numbers.update(ann)
            flags.append(b)
        bounded = any(bool(f) for f in flags)
        # the other falsifying clause, its contrast still to come (not left out under rule 6), could confirm the
        # falsification with its own bound once the data arrive
        pending = [f"{name}: {missing[name]}" for name, clause in (("G", g_includes_zero), ("F", no_fall))
                   if clause is None and missing.get(name, "").startswith("incomplete")]
        if pending and not bounded and reading.calibration == "bound":
            status = NOT_COMPUTABLE
            reason = ("; ".join(pending) + " (the falsifying clause that reads it, with its own Part 5.7 bound, could "
                      "still confirm the falsification: Q-falsification-calibration)")
    status = calibrate(status, bounded, reading, notes)
    if missing_why and status != NOT_COMPUTABLE:
        notes.append(f"decided without {missing_why}: the clause the known estimate decides settles the box")
    if status == SUPPORTED:
        notes += detectable_notes({"F": F, "G": G})
    numbers.update(_estimate_numbers("F = gap(warm) - gap(untreated)", F, reading))
    numbers.update(_estimate_numbers("G = gap(warm) - gap(N=0)", G, reading))
    estimates = (([{"contrast": f"F {WARM} - untreated", **F.row()}] if F is not None else [])
                 + ([{"contrast": f"G {WARM} - N0", **G.row()}] if G is not None else []))
    for variant in (RATE, R.PID_VARIANT):
        if variant == R.PID_VARIANT and task not in R.PID_TASKS:
            continue
        Fv, Gv, why_v, _ = _h4_estimates(A, reading, task, condition, variant)  # tabled ages; no note: not deciding
        numbers.update(_estimate_numbers(f"F({variant})", Fv, reading))
        numbers.update(_estimate_numbers(f"G({variant})", Gv, reading))
        if why_v:
            notes.append(f"{variant} (reported alongside): " + "; ".join(f"{n}: {w}" for n, w in why_v.items()))
        estimates += [{"contrast": f"F {variant} - untreated", **Fv.row()}] if Fv else []
        estimates += [{"contrast": f"G {variant} - N0", **Gv.row()}] if Gv else []
    numbers.update(_rate_clip_numbers(A, task))
    return Outcome(status, reason, numbers=numbers, notes=notes, estimates=estimates)


def _rate_clip_numbers(A: StudyA, task: str) -> dict[str, Any]:
    """Q-rate-limit (answered): "the report gives the share of constrained epochs in which the clip
    bound" for the rate-limited arm, reported alongside H4 and deciding nothing (the arm stays the secondary H4
    check): each completed run's share (its training record's ``rate_limit_clip_share``, from
    ``metrics.controller.rate_limit_clip_share``) by seed, their mean and the number of runs that have one."""
    arm = A.arms(task, N=LATE_N, shape=TREATMENT_SHAPE, control=TREATMENT_CONTROL, controller=RATE)
    by_seed = {int(rec["seed"]): float(rec["rate_limit_clip_share"]) for rec in (arm.records if arm else [])
               if rec.get("rate_limit_clip_share") is not None}
    name = f"rate_limit_clip_share({RATE})"
    return {name: statistics.fmean(by_seed.values()) if by_seed else None, f"{name}_n": len(by_seed),
            f"{name}_by_seed": dict(sorted(by_seed.items()))}


def agree(a: tuple[float, float, float], b: tuple[float, float, float]) -> bool:
    """Part 4.1.1's agreement of two estimands, each (estimate, Welch lo, Welch hi): the same sign and each
    point estimate inside the other's primary interval (descriptive; Part 4.1.1 does not define agreement, and this
    reading decides nothing)."""
    (x, x_lo, x_hi), (y, y_lo, y_hi) = a, b
    return bool((settled(x) > 0) == (settled(y) > 0) and y_lo <= x <= y_hi and x_lo <= y <= x_hi)


def _control_waits(out: Outcome, control: str, *, holm: bool) -> bool:
    """Whether a NOT_COMPUTABLE control of box H1 or H2 (``out``, from ``_combined``) still waits on data to come: an
    estimate it reads is incomplete (its ``incomplete`` fractions), or, with ``holm`` (the support condition reads
    Holm, the falsification never does), its Holm survival waits on an incomplete family member. A control that
    records neither (it stopped before reading its estimates while its task's matching waits) may still change. One
    that lacks only what rule 6 leaves out, or whose task rules 3 and 5 make infeasible, is NOT_COMPUTABLE for good: a
    permanent state of complete data (Q-h0-scope: "NOT_COMPUTABLE only while missing data could still change it")."""
    incomplete = out.numbers.get(f"{control}.incomplete")
    if incomplete is None:
        return True
    return bool(incomplete) or (holm and out.numbers.get(f"{control}.holm_waits") is True)


def could_be_supported(out: Outcome) -> bool:
    """Whether box H1 or H2 (``out``, from ``_combined``) is SUPPORTED or could still become SUPPORTED: the twin
    of ``could_be_falsified``. "Supported" needs every control under either reading of Q-controls-reading
    (``combine_controls``), so every control must be SUPPORTED, or NOT_COMPUTABLE with its support condition
    not already False while it still waits on data (``_control_waits``); box H1's S1 needs Delta(0.50), so a
    control whose N = 0.50 arm rule 6 leaves out never is."""
    if out.status == SUPPORTED:
        return True
    if out.status != NOT_COMPUTABLE:
        return False
    reach = []
    for control in R.STEP_MATCHING_CONTROLS:
        status = out.numbers.get(f"{control}.status")
        if status == SUPPORTED:
            reach.append(True)
        elif status == NOT_COMPUTABLE or status is None:
            reach.append(out.numbers.get(f"{control}.support") is not False and _control_waits(out, control, holm=True)
                         and LATE_N not in (out.numbers.get(f"{control}.not_compared_rule6") or ()))
        else:
            reach.append(False)
    return all(reach)


def could_be_falsified(out: Outcome, reading: Reading) -> bool:
    """Whether box H1 or H2 (``out``, from ``_combined``) is FALSIFIED or could still become FALSIFIED.

    A control whose status is known (SUPPORTED, INCONCLUSIVE) cannot become FALSIFIED, nor can a
    NOT_COMPUTABLE control whose falsification condition is already False (its own numbers), or that no
    longer waits on data (``_control_waits``), or, under Q-falsification-calibration's answer, whose
    Delta(0.50) rule 6 leaves out (box H1: no Part 5.7 bound, so never FALSIFIED); over the controls,
    "falsified under both" needs every control, "under either" one (Q-controls-reading).
    ``combine_controls`` gives NOT_COMPUTABLE whatever the missing control rules out, so the H0 gate on
    H3 and H4 asks this instead."""
    if out.status == FALSIFIED:
        return True
    if out.status != NOT_COMPUTABLE:
        return False
    reach = []
    for control in R.STEP_MATCHING_CONTROLS:
        status = out.numbers.get(f"{control}.status")
        if status == FALSIFIED:
            reach.append(True)
        elif status == NOT_COMPUTABLE or status is None:
            unbounded = (reading.calibration == "bound"
                         and LATE_N in (out.numbers.get(f"{control}.not_compared_rule6") or ()))
            reach.append(out.numbers.get(f"{control}.falsification") is not False
                         and _control_waits(out, control, holm=False) and not unbounded)
        else:
            reach.append(False)
    return all(reach) if reading.controls == "both" else any(reach)


def _part_could(out: Outcome, target: str) -> bool:
    """Whether a part of H3 (``h3a_outcome``, ``h3b_outcome``, ``h3c_outcome``) is ``target`` (SUPPORTED or
    FALSIFIED) or, NOT_COMPUTABLE, could still become it: its stored three-valued "support" or "falsification"
    is not already False (missing when the part stopped before evaluating it: it could)."""
    if out.status == target:
        return True
    key = "support" if target == SUPPORTED else "falsification"
    return out.status == NOT_COMPUTABLE and out.numbers.get(key) is not False


def _treated_variants(task: str) -> Iterator[tuple[str, dict[str, str]]]:
    """(variant, ``StudyA.arms`` keyword) of every treated variant the task has (the PID variant only on
    ``R.PID_TASKS``)."""
    for variant in TREATED_VARIANTS:
        if variant == R.PID_VARIANT and task not in R.PID_TASKS:
            continue
        yield variant, ({"controller": variant} if variant in (*R.CONTROLLER_VARIANTS, R.PID_VARIANT)
                        else {"treatment": variant})


HOLM_FAMILIES = ("Holm-corrected in both its families (its task's four conditions and its condition's three tasks; "
                 "Q-holm-families)")  # how every secondary cell's claims are corrected (``StudyA.holm_survival``)


def h0_from(o1: Outcome, o2: Outcome, reading: Reading) -> str:
    """Box H0 from boxes H1 and H2 on the primary outcome (Q-h0-scope), three-valued as ``combine_controls``:
    NOT_COMPUTABLE only while the missing part could still change it (both H1 and H2 could still be falsified,
    or either could still be supported), else INCONCLUSIVE."""
    h1, h2 = o1.status, o2.status
    if h1 == SUPPORTED or h2 == SUPPORTED:
        return FALSIFIED
    if h1 == FALSIFIED and h2 == FALSIFIED:
        return SUPPORTED
    if ((could_be_falsified(o1, reading) and could_be_falsified(o2, reading))
            or could_be_supported(o1) or could_be_supported(o2)):
        return NOT_COMPUTABLE
    return INCONCLUSIVE


# ---------------------------------------------------------------------------
# Secondary outcomes
# ---------------------------------------------------------------------------


# answered (Q-controller-quantities): a run that never recovers enters one epoch above its horizon (the
# steps from onset to the end of its training), above any recovery it could show, so "recovered in the last epoch"
# and "never recovered" stay apart (as Table 2.5 records adaptation steps "as above the largest horizon"); a contrast
# of two arms censors both at the shorter arm's horizon (see ``_recovery_getter``).
RECOVERY_CENSOR_OFFSET = R.STEPS_PER_EPOCH


def _recovery(rec: dict[str, Any], horizon: Optional[int] = None) -> Optional[float]:
    """Table 2.1 recovery time ("Training steps from onset until the training-batch mean episodic cost first
    falls to or below d"), censored: a run that never recovers within ``horizon`` steps after onset (default,
    and at most, its own: total_steps - onset_step) enters as horizon + RECOVERY_CENSOR_OFFSET; a recovery
    later than ``horizon`` is censored there too (Q-controller-quantities)."""
    censored, value = rec.get("recovery_censored") is True, rec.get("recovery_steps")
    if not censored and value is None:
        return None
    if rec.get("total_steps") is None or rec.get("onset_step") is None:
        return None if censored else float(value)
    own = int(rec["total_steps"]) - int(rec["onset_step"])
    limit = own if horizon is None else min(own, int(horizon))
    if censored or float(value) > limit:
        return float(limit + RECOVERY_CENSOR_OFFSET)
    return float(value)


def _recovery_getter(*arms: Optional[Arm]) -> Callable[[dict[str, Any]], Optional[float]]:
    """Recovery time per seed for a contrast of ``arms``, censored at one common horizon: the shortest
    (total_steps - onset_step) among their runs. Arms that train for different times after onset (the
    data control trains T + N·T steps, the untreated arm T) would otherwise compare a censored seed at
    different values, and a recovery only the longer arm can observe."""
    horizons = [int(r["total_steps"]) - int(r["onset_step"]) for arm in arms if arm is not None for r in arm.records
                if r.get("total_steps") is not None and r.get("onset_step") is not None]
    horizon = min(horizons) if horizons else None
    return lambda rec: _recovery(rec, horizon)


MEASUREMENT_RETURN = stats.EpisodeOutcome(
    source="measurement_return", value=lambda r: r.get("measurement_return"),
    episodes=lambda r: None if r.get("measurement_return_episodes") is None else [(1.0, r["measurement_return_episodes"])])


def _return_getter(A: StudyA) -> tuple[Callable, str]:
    """Return per seed, one outcome for every cell: the measurement-set return of the matched checkpoint (the
    ``measurement`` record that ``enrich measure`` writes for every completed Study A run). In final mode a run
    without one makes its cell incomplete (``return_cells``), never a per-cell switch to the final checkpoint's
    return: that would compare unmatched checkpoints (Part 4.1.1) and mix estimands in one Holm family.
    Only the pilot test run (Part 5.8; its verdicts are never read), lacking a measurement-set return for
    some run, reads the final checkpoint's return in every cell at once, so labelled."""
    if A.dataset.mode == "final" or all(r.get("measurement_return") is not None for r in A.records):
        return MEASUREMENT_RETURN, "measurement-set return of the matched checkpoint"
    return ((lambda r: r.get("final_return")),
            "pilot test run: the final checkpoint's return in every cell (a measurement-set return is not recorded "
            "for every run)")


@dataclass
class Catalogue:
    """Everything Study A reports."""

    verdicts: list[Verdict] = field(default_factory=list)
    tables: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


Cells = dict[tuple, tuple[Optional[stats.TwoArmEstimate], str]]


def _cut_or(A: StudyA, task: str, *arms: tuple[Optional[Arm], dict[str, Any]]) -> Optional[tuple[None, str]]:
    """(None, reason) when one of the arms (arm, its ``StudyA.arms`` keywords) is missing because a recorded Part 6.1
    cut removed it (``StudyA.absent_reason``): the claim is not tested, never incomplete; None otherwise (the
    contrast decides; a task that a cut removed whole keeps ``contrast``'s reason)."""
    if A.task_cut_by(task):
        return None
    for arm, factors in arms:
        if arm is None:
            why = A.absent_reason(task, **factors)
            if why.startswith(CUT_TASK_REASON):
                return None, why
    return None


def _claims(*, make_cells: Callable[[Reading], Cells], family_of: Callable[[tuple], tuple],
            id_of: Callable[[tuple], str], statement_of: Callable[[tuple], str], title: str, table_key: str,
            label: str, group: str, note_of: Callable[[tuple], str] = lambda c: "") -> list[Verdict]:
    """Secondary claims "the difference is not zero", Holm-corrected within each family (Part 5.2).

    Each claim here has one family (``family_of``: recovery time and return over the three tasks), so
    Q-holm-families' question (how two families combine) does not arise: Holm applies under every
    reading; "Families shrink only to cells not tested" (a cell without an estimate leaves its family, unless
    its data are still to come: an incomplete cell keeps the family's other decisions open). The numbers give
    Holm over the tested members (``holm_adjusted_p``, ``holm_family_size``), the count of incomplete members
    (``holm_family_incomplete``) and the adjusted p with each of them held at p = 1
    (``holm_adjusted_p_worst``): a claim survives when that one is rejected, and does not when the first is not
    (``stats.holm_decisions``).

    ``make_cells(reading)`` gives every cell's estimate under a reading (the arms it compares follow
    the reading's matching), so a matching question can leave a claim UNDECIDED like any verdict.
    """
    cache: dict[tuple, tuple[Cells, dict]] = {}

    def cells_for(reading: Reading) -> tuple[Cells, dict]:
        key = (reading.cost_field, reading.arithmetic, reading.seeds)
        if key not in cache:
            cells = make_cells(reading)
            families: dict[tuple, dict[tuple, Member]] = {}
            for cell, (est, why) in cells.items():
                families.setdefault(family_of(cell), {})[cell] = _member(est, why, id_of(cell))
            adjusted: dict[tuple, tuple[stats.HolmResult, Optional[bool], str, stats.HolmResult, int]] = {}
            for ps in families.values():
                results, survives, incomplete = stats.holm_decisions(ps, R.ALPHA)
                waiting = holm_waiting("; ".join(incomplete.values())) if incomplete else ""
                # the incomplete members held at p = 1, as ``stats.holm_decisions`` decides survival
                worst = (stats.holm({**{k: h.p for k, h in results.items()}, **dict.fromkeys(incomplete, 1.0)}, R.ALPHA)
                         if incomplete else results)
                adjusted.update({c: (h, survives[c], waiting, worst[c], len(incomplete)) for c, h in results.items()})
            cache[key] = (cells, adjusted)
        return cache[key]

    out = []
    for cell in cells_for(PROPOSED)[0]:
        def compute(reading: Reading, cell=cell) -> Outcome:
            cells, adjusted = cells_for(reading)
            est, why = cells[cell]
            notes = [note_of(cell)] if note_of(cell) else []
            if est is None:  # a task a recorded cut removed is not tested, never incomplete
                return Outcome(NOT_TESTED if why.startswith(CUT_TASK_REASON) else NOT_COMPUTABLE, why, notes=notes)
            lo, hi = est.interval(reading.interval)
            # one family per claim: Holm under every reading of Q-holm-families
            holm, holm_ok, waiting, worst, n_incomplete = adjusted[cell]
            numbers = {"diff": est.diff, "welch_lo": est.welch.lo, "welch_hi": est.welch.hi, "boot_lo": est.bootstrap.lo,
                       "boot_hi": est.bootstrap.hi, "p": est.welch.p, "holm_adjusted_p": holm.adjusted,
                       "holm_family_size": holm.family_size, "holm_family_incomplete": n_incomplete,
                       "holm_adjusted_p_worst": worst.adjusted, "holm_survives": holm_ok, "cohens_d": est.cohens_d,
                       "n": (len(est.seeds_x), len(est.seeds_y)), "detectable_d": est.detectable_d,
                       "below_detectable": est.below_detectable}
            excludes = _excludes_zero(lo, hi)  # an interval including zero is INCONCLUSIVE whatever Holm gives
            status = claim_status(excludes, holm_ok if excludes else True)
            if status == SUPPORTED:
                notes += detectable_notes({"the difference": est})
            return Outcome(status, waiting if status == NOT_COMPUTABLE else "", numbers=numbers, estimates=[est.row()],
                           notes=notes)
        out.append(decide(id=id_of(cell), title=title, statement=statement_of(cell), label=label, table_key=table_key,
                          compute=compute, group=group))
    return out


def analyse(dataset: Dataset, *, iqm: bool = True) -> Catalogue:
    """Every Study A verdict and table (``iqm=False`` leaves table A_iqm empty: unit tests of the verdicts)."""
    A = StudyA(dataset)
    cat = Catalogue()
    primary = f"{R.PRIMARY_TASK} x {PRIMARY_CONDITION}"

    # -- H1 and H2 on the primary outcome; H0 -------------------------------
    h1_cache: dict[Reading, Outcome] = {}
    h2_cache: dict[Reading, Outcome] = {}

    def h1(reading: Reading) -> Outcome:
        if reading not in h1_cache:
            h1_cache[reading] = h1_outcome(A, reading)
        return h1_cache[reading]

    def h2(reading: Reading) -> Outcome:
        if reading not in h2_cache:
            h2_cache[reading] = h2_outcome(A, reading)
        return h2_cache[reading]

    v_h1 = decide(id="H1", title="H1: later onset, larger gap at matched cost", label=CONFIRMATORY, table_key="H1",
                  statement=("Delta(0.50) of equation (2) is positive with a 95 percent interval excluding zero and Delta "
                             f"increases with N, under both step-matching controls ({primary}; primary analysis)"),
                  compute=h1, conditions=(PRIMARY_CONDITION,), group="Study A: primary")
    v_h2 = decide(id="H2", title="H2: abrupt onset, larger gap than a ramp", label=SECONDARY, table_key="H2",
                  statement=("gap(abrupt) - gap(ramp) > 0 at N = 0.50 with an interval excluding zero and >= 0 at 0.10 "
                             f"and 0.25 ({primary})"),
                  compute=h2, conditions=(PRIMARY_CONDITION,), group="Study A: hypotheses")

    def h0(reading: Reading) -> Outcome:
        o1, o2 = h1(reading), h2(reading)
        s1, s2 = o1.status, o2.status
        status = h0_from(o1, o2, reading)
        reason = ""
        if status == NOT_COMPUTABLE:
            reason = f"H1 is {s1}, H2 is {s2}" + "".join(
                f"; {name}: {out.reason}" for name, out in (("H1", o1), ("H2", o2))
                if out.status == NOT_COMPUTABLE and out.reason)
        return Outcome(status, reason, numbers={"H1": s1, "H2": s2})

    v_h0 = decide(id="H0", title="H0: neither onset fraction nor shape changes the gap at matched cost", label=SECONDARY,
                  table_key="H0", statement="H1 and H2 are both falsified on the primary outcome (Q-h0-scope)",
                  compute=h0, conditions=(PRIMARY_CONDITION,), group="Study A: hypotheses")

    def h0_gate(reading: Reading) -> Optional[tuple[str, str]]:
        """Box H0 on H3 and H4, read three-valued: (status, reason) when H0 is supported (NOT_TESTED) or
        may still be (NOT_COMPUTABLE: H1 and H2 each falsified or not computable, at least one not computable, and
        each still able to be falsified: ``could_be_falsified``), else None."""
        o1, o2 = h1(reading), h2(reading)
        s1, s2 = o1.status, o2.status
        status = h0_from(o1, o2, reading)
        if status == SUPPORTED:
            return NOT_TESTED, ("H0 is supported: box H0 says H3 and H4 'are reported as not tested, not as falsified' "
                                "(the estimates are still given)")
        if (status == NOT_COMPUTABLE and {s1, s2} <= {FALSIFIED, NOT_COMPUTABLE}
                and could_be_falsified(o1, reading) and could_be_falsified(o2, reading)):
            return NOT_COMPUTABLE, (f"H0 cannot be decided (H1 {s1}, H2 {s2}); if it is supported, box H0 says H3 and "
                                    "H4 'are reported as not tested, not as falsified' (the estimates are still given)")
        return None

    def gated(fn: Callable[[Reading], Outcome]) -> Callable[[Reading], Outcome]:
        def compute(reading: Reading) -> Outcome:
            out = fn(reading)
            gate = h0_gate(reading)
            if gate is not None:
                return Outcome(gate[0], gate[1], out.numbers, out.estimates,
                               out.notes + [f"estimates computed: {out.status}"])
            return out
        return compute

    parts = {}
    for key, fn, title, statement in (
        ("H3a", lambda r: h3a_outcome(A, r), "H3 (a): plasticity at onset and the gap",
         "rho(dormant at onset, gap) > 0 and rho(rank at onset, gap) < 0, each with an interval excluding zero"),
        ("H3b", lambda r: h3b_outcome(A, r), "H3 (b): reset and injection reduce the N = 0.50 gap",
         "reset and injection each reduce the gap with an interval excluding zero; additional constrained training does not"),
        ("H3c", lambda r: h3c_outcome(A, r),
         f"H3 (c): manipulation check at onset + {R.MANIPULATION_CHECK_STEPS_AFTER_ONSET:,} steps",
         "reset and injection lower the trainable units' dormant fraction and raise the trainable layers' rank"),
    ):
        parts[key] = decide(id=key, title=title, statement=f"{statement} ({primary})", label=SECONDARY, table_key=key,
                            compute=gated(fn), conditions=(PRIMARY_CONDITION,), group="Study A: hypotheses")

    def h3(reading: Reading) -> Outcome:
        a_out, b_out, c_out = h3a_outcome(A, reading), h3b_outcome(A, reading), h3c_outcome(A, reading)
        outs = {"H3a": a_out, "H3b": b_out, "H3c": c_out}
        statuses = {k: o.status for k, o in outs.items()}
        if FALSIFIED in statuses.values():
            status = FALSIFIED
        elif all(s == SUPPORTED for s in statuses.values()):
            status = SUPPORTED
        elif (any(o.status == NOT_COMPUTABLE and _part_could(o, FALSIFIED) for o in outs.values())
              or all(_part_could(o, SUPPORTED) for o in outs.values())):
            status = NOT_COMPUTABLE  # only while a missing part could still change it (three-valued, as box H0)
        else:
            status = INCONCLUSIVE
        notes = []  # the mediation wording is added after decide(), on the final status (below)
        for t in (RESET, INJECTION):  # "without passing the check" is an absence: read uncorrected
            if b_out.numbers.get(f"reduces({t})") and c_out.numbers.get(f"passes_uncorrected({t})") is False:
                notes.append(f"{t} narrows the gap without passing the manipulation check: it is reported as acting "
                             "through something other than plasticity")
        gate = h0_gate(reading)
        if gate is not None:
            return Outcome(gate[0], gate[1], dict(statuses), [], [f"parts computed: {statuses}"])
        reason = ("; ".join(f"{k} is {NOT_COMPUTABLE}" + (f" ({o.reason})" if o.reason else "")
                            for k, o in outs.items() if o.status == NOT_COMPUTABLE)
                  if status == NOT_COMPUTABLE else "")
        return Outcome(status, reason, numbers=dict(statuses), notes=notes)

    # the registered box title ("H3. Plasticity-related measurements and interventions explain part of the effect"):
    # mediation is only ever MEDIATION_NOTE on a supported H3, never the title of every report
    v_h3 = decide(id="H3", title="H3: plasticity-related measurements and interventions explain part of the gap",
                  label=SECONDARY, table_key="H3",
                  statement="H3 (a), (b) and (c) all hold (then 'consistent with mediation by plasticity')",
                  compute=h3, conditions=(PRIMARY_CONDITION,), group="Study A: hypotheses")
    # Part 1.2 (on H3, "What H3 can establish"): "The word mediation is used in the paper only if all three parts
    # hold". Read on the verdict as decided, not on the proposals' status: an H3 that is UNDECIDED(key) has not been
    # found to hold.
    if v_h3.status == SUPPORTED:
        v_h3.notes.append(MEDIATION_NOTE)
    v_h4 = decide(id="H4", title="H4: multiplier overshoot explains part of the gap", label=SECONDARY, table_key="H4",
                  statement=("with the warm-started multiplier the N = 0.50 abrupt gap falls (interval excluding zero) and "
                             f"stays above the N = 0 gap (interval excluding zero) ({primary})"),
                  compute=gated(lambda r: h4_outcome(A, r)), conditions=(PRIMARY_CONDITION,), group="Study A: hypotheses")
    cat.verdicts += [v_h1, v_h2, parts["H3a"], parts["H3b"], parts["H3c"], v_h3, v_h4, v_h0]

    # -- the boxes on the other cells (secondary: other conditions and tasks) -----
    for task in R.TASKS_STUDY_A:
        for condition in CONDITIONS:
            for shape in R.ONSET_SHAPES:
                if (task, condition, shape) == (R.PRIMARY_TASK, PRIMARY_CONDITION, "abrupt"):
                    continue
                cat.verdicts.append(decide(
                    id=f"H1[{task}/{condition}/{shape}]", title="H1 on a secondary cell", label=SECONDARY,
                    table_key="H1-cell", conditions=(condition,), group="Study A: other conditions and tasks",
                    statement=f"box H1 on {task} x {condition}, {shape} arms, {HOLM_FAMILIES}",
                    compute=lambda r, t=task, c=condition, s=shape: h1_outcome(A, r, task=t, condition=c, shape=s, holm=True)))
            if (task, condition) == (R.PRIMARY_TASK, PRIMARY_CONDITION):
                continue
            for key, fn, title in (
                ("H2", lambda r, t=task, c=condition: h2_outcome(A, r, task=t, condition=c), "H2 on a secondary cell"),
                ("H3a", gated(lambda r, t=task, c=condition: h3a_outcome(A, r, task=t, condition=c)),
                 "H3 (a) on a secondary cell"),
                ("H3b", gated(lambda r, t=task, c=condition: h3b_outcome(A, r, task=t, condition=c)),
                 "H3 (b) on a secondary cell"),
                ("H4", gated(lambda r, t=task, c=condition: h4_outcome(A, r, task=t, condition=c)), "H4 on a secondary cell"),
            ):
                cat.verdicts.append(decide(
                    id=f"{key}[{task}/{condition}]", title=title, label=SECONDARY, table_key=f"{key}-cell",
                    conditions=(condition,), group="Study A: other conditions and tasks",
                    statement=f"box {key} on {task} x {condition}, {HOLM_FAMILIES}", compute=fn))
        if task != R.PRIMARY_TASK:
            cat.verdicts.append(decide(
                id=f"H3c[{task}]", title="H3 (c) on another task", label=SECONDARY, table_key="H3c-cell",
                group="Study A: other conditions and tasks",
                statement=f"box H3 (c) on {task}, Holm-corrected over the three tasks",
                compute=gated(lambda r, t=task: h3c_outcome(A, r, task=t))))

    # -- Part 4.1.1 and rule 7 (exploratory) --------------------------------
    def final_estimand(reading: Reading) -> Outcome:
        out = h1_outcome(A, reading, estimand="final")
        primary_out = h1(reading)
        for control in R.STEP_MATCHING_CONTROLS:
            p, f = primary_out.numbers.get(f"{control}.Delta(0.50)"), out.numbers.get(f"{control}.Delta(0.50)")
            if p is None or f is None:
                out.numbers[f"{control}.agree"] = None
                continue
            out.numbers[f"{control}.agree"] = agree(
                (p, primary_out.numbers[f"{control}.Delta(0.50)_welch_lo"],
                 primary_out.numbers[f"{control}.Delta(0.50)_welch_hi"]),
                (f, out.numbers[f"{control}.Delta(0.50)_welch_lo"], out.numbers[f"{control}.Delta(0.50)_welch_hi"]))
        out.notes.append("Part 4.1.1: 'agreement between the two strengthens the causal reading, disagreement limits it'; "
                         "agree = same sign and each point estimate inside the other's Welch interval (descriptive; "
                         "every fraction and control: table A_estimands_side_by_side)")
        return out

    cat.verdicts.append(decide(
        id="A-final-estimand", title="Part 4.1.1 final-checkpoint estimand = rule 7 (b)", label=EXPLORATORY,
        table_key="A-final-estimand", conditions=(PRIMARY_CONDITION,), group="Study A: sensitivity (Part 4.1 rule 7)",
        statement=f"box H1 with every arm's final checkpoint in place of the selected one, no matching filter ({primary})",
        compute=final_estimand))
    cat.verdicts.append(decide(
        id="A-rule7a", title="Rule 7 (a): the primary analysis at tolerance 5.0", label=EXPLORATORY, table_key="A-rule7a",
        conditions=(PRIMARY_CONDITION,), group="Study A: sensitivity (Part 4.1 rule 7)",
        statement=(f"box H1 with arms matched at tolerance {R.SENSITIVITY_TOLERANCE:g} (rules 3 and 5 at "
                   f"{R.SENSITIVITY_TOLERANCE:g}) ({primary})"),
        compute=lambda r: h1_outcome(A, r, estimand="tolerance5")))

    # -- recovery time and return (secondary claims) -----------------------
    def recovery_cells(reading: Reading) -> Cells:
        A_r = A.for_reading(reading)
        cells: Cells = {}
        for task in R.TASKS_STUDY_A:
            for control in R.STEP_MATCHING_CONTROLS:
                for N in R.LATE_ONSET_FRACTIONS:
                    xf, yf = (dict(N=N, shape="abrupt", control=control), dict(N=N, shape="ramp", control=control))
                    x, y = A_r.arms(task, **xf), A_r.arms(task, **yf)
                    cells[("shape", task, control, N)] = _cut_or(A_r, task, (x, xf), (y, yf)) or A_r.contrast(
                        reading, task=task, x=x, y=y, get=_recovery_getter(x, y),
                        analysis_id=f"recovery|{task}|{control}|N{N:.2f}|abrupt-ramp")
            for variant, kw in _treated_variants(task):
                yf = dict(N=LATE_N, shape=TREATMENT_SHAPE, control=TREATMENT_CONTROL)
                x, y = A_r.arms(task, **yf, **kw), A_r.arms(task, **yf)
                cells[("treated", task, variant, LATE_N)] = _cut_or(A_r, task, (x, {**yf, **kw}), (y, yf)) or A_r.contrast(
                    reading, task=task, x=x, y=y, get=_recovery_getter(x, y),
                    analysis_id=f"recovery|{task}|{variant}-untreated")
        return cells

    cat.verdicts += _claims(
        make_cells=recovery_cells, family_of=lambda c: (c[0], c[2], c[3]),
        id_of=lambda c: f"A-recovery[{c[1]}/{c[2]}/{'abrupt-ramp' if c[0] == 'shape' else 'treated-untreated'}/N{c[3]:.2f}]",
        statement_of=lambda c: (f"recovery time differs between abrupt and ramp onset ({c[1]}, {c[2]}, N = {c[3]:.2f})"
                                if c[0] == "shape" else
                                f"recovery time differs between {c[2]} and the untreated arm ({c[1]}, N = {c[3]:.2f})"),
        title="Recovery time (Table 2.1; secondary)", table_key="A-recovery", label=SECONDARY, group="Study A: recovery time",
        note_of=lambda c: ("a run that never recovers enters one epoch above the horizon, both arms censored at the "
                           "shorter arm's steps from onset to the end of training (a lower bound; Q-controller-quantities)"
                           + ("; additional constrained training trains N·T steps longer than the untreated arm, which "
                              "the common horizon discounts" if c[2] == ADDITIONAL else "")
                           + ("; Holm over the tasks that run it" if c[2] == R.PID_VARIANT
                              else "; Holm over the three tasks")))

    return_source: dict[tuple, str] = {}

    def return_cells(reading: Reading) -> Cells:
        A_r = A.for_reading(reading)
        cells: Cells = {}
        get, source = _return_getter(A_r)
        for task in R.TASKS_STUDY_A:
            ref = A_r.reference(task)
            for control in R.STEP_MATCHING_CONTROLS:
                for N in R.LATE_ONSET_FRACTIONS:
                    late_f = dict(N=N, shape="abrupt", control=control)
                    late = A_r.arms(task, **late_f)
                    if reading == PROPOSED:
                        return_source[(task, control, N)] = source
                    est, why = (_cut_or(A_r, task, (late, late_f), (ref, dict(N=R.ONSET_FRACTIONS[0])))
                                or A_r.contrast(reading, task=task, x=late, y=ref, get=get,
                                                analysis_id=f"return|{task}|{control}|N{N:.2f}"))
                    if why.startswith("incomplete: no value for "):
                        missing = why.removeprefix("incomplete: no value for ")
                        why = (f"incomplete: no measurement-set return for {missing} (`enrich measure`)"
                               if get is MEASUREMENT_RETURN else f"incomplete: no final-checkpoint return for {missing}")
                    cells[(task, control, N)] = (est, why)
        return cells

    cat.verdicts += _claims(
        make_cells=return_cells, family_of=lambda c: (c[1], c[2]),
        id_of=lambda c: f"A-return[{c[0]}/{c[1]}/N{c[2]:.2f}]",
        statement_of=lambda c: f"return(N = {c[2]:.2f}, abrupt) - return(N = 0) is not zero ({c[0]}, {c[1]})",
        title="Return (Part 5.2; secondary)", table_key="A-return", label=SECONDARY, group="Study A: return",
        note_of=lambda c: f"outcome: {return_source.get(c)}; Holm over the three tasks")

    cat.tables.update(_tables(A, dataset, iqm=iqm))
    cat.tables["A_estimates"] = [{"key": k, **v} for k, v in sorted(A.estimate_rows.items())]
    return cat


# ---------------------------------------------------------------------------
# Tables (Part 4.1.1 reporting, rule 6, controller quantities, IQM)
# ---------------------------------------------------------------------------


def _tables(A: StudyA, dataset: Dataset, *, iqm: bool = True) -> dict[str, list[dict[str, Any]]]:
    """The Study A tables: Part 4.1.1 reporting, rule 6, controller quantities and the IQM."""
    tables: dict[str, list[dict[str, Any]]] = {}
    result = A.matching(PROPOSED)
    sens = A.matching(PROPOSED, R.SENSITIVITY_TOLERANCE)
    alt_cost = A.matching(Reading(cost_field=matching.SELECTION_COST_FIELD))
    rows = []
    for task, tm in sorted(result.items()):
        for arm in (tm.reference, *tm.arms):
            if arm is None:
                continue
            s_arm = sens[task].arm(arm.arm_id) if task in sens else None
            a_arm = alt_cost[task].arm(arm.arm_id) if task in alt_cost else None
            stored = [r for r in A.records if r["arm_id"] == arm.arm_id]
            stored_matched = {r["stored_matched"] for r in stored}
            rows.append({
                "task": task, "arm_id": arm.arm_id, "arm": arm.arm, "N": arm.N, "onset_shape": arm.onset_shape,
                "step_matching": arm.step_matching, "treatment": arm.treatment, "controller_variant": arm.controller_variant,
                "n": len(arm.seeds), "seeds": " ".join(map(str, arm.seeds)), "matched_cost": arm.matched_cost,
                "difference": arm.difference, "status": arm.status, "matched": arm.matched,
                "reference_cost": tm.reference_cost, "reference_sd": tm.reference_sd, "task_feasible": tm.feasible,
                "task_reason": tm.reason, "tolerance": tm.tolerance,
                "status_tolerance_5": None if s_arm is None else s_arm.status,
                "status_selection_set_cost": None if a_arm is None else a_arm.status,
                "stored_matched": ";".join(sorted(map(str, stored_matched))),
                "stored_flags_agree": (None if stored_matched == {None} else stored_matched == {arm.matched}),
            })
    tables["A_matching"] = rows
    # Rule 6: "A late arm that never reaches the budget is a finding about learning, reported as such, and not a
    # finding about robustness" (Q-unmatched-reporting, HANDOVER section 9; not a PENDING key): an unmatched arm whose
    # matched cost fails rule 3's test of satisfying the budget (cost <= d + tolerance, the same arithmetic: 27.5;
    # 30.0 in the rule 7 (a) repeat, the ``_tolerance_5`` columns, exploratory). Any other unmatched arm is reported
    # with its cost and difference, without the learning label. A row is an arm unmatched at either tolerance; the
    # columns of the tolerance at which it is not unmatched are empty.
    tables["A_unmatched"] = []
    for r in rows:
        unmatched, unmatched_5 = (r[k] in matching.UNMATCHED_STATUSES for k in ("status", "status_tolerance_5"))
        if not (unmatched or unmatched_5):
            continue
        reaches = matching.budget_feasible(r["matched_cost"], r["tolerance"]) if unmatched else None
        reaches_5 = (matching.budget_feasible(sens[r["task"]].arm(r["arm_id"]).matched_cost, R.SENSITIVITY_TOLERANCE)
                     if unmatched_5 else None)
        tables["A_unmatched"].append({
            **r, "reaches_budget": reaches, "learning_finding": None if reaches is None else not reaches,
            "reaches_budget_tolerance_5": reaches_5,
            "learning_finding_tolerance_5": None if reaches_5 is None else not reaches_5,
            "min_window_selection_cost": min((x["min_window_selection_cost"] for x in A.records
                                              if x["arm_id"] == r["arm_id"] and x["min_window_selection_cost"] is not None),
                                             default=None)})
    tables["A_selection"] = [
        {k: r.get(k) for k in ("run_id", "arm_id", "task", "seed", "matched_checkpoint_step", "training_age",
                                "onset_step", "onset_source", "constrained_steps_at_selection", "selection_cost_at_match",
                                "measurement_cost", "measurement_return", "lambda_at_selection", "final_cost",
                                "final_measurement_cost", "min_window_selection_cost", "rule_one_rechecked")}
        for r in A.records
    ]
    # Part 4.1.1: "Where selected training ages differ systematically between arms, that difference is named in
    # the paper as a limit on the causal interpretation" (Q-training-age-systematic, not a PENDING key: the Welch
    # 95 percent interval of the difference excludes zero). Each arm against its task's reference, then every pair
    # of arms a verdict compares on matched checkpoints (``StudyA.age_note``), with the analysis ids that read it.
    ages = []
    for task in R.TASKS_STUDY_A:
        ref = A.reference(task)
        if ref is None:
            continue
        ref_ages, _ = A.values(ref, lambda r: r.get("training_age"))
        for arm_key in sorted({r["arm_id"] for r in A.records if r["task"] == task}):
            recs = [r for r in A.records if r["arm_id"] == arm_key]
            arm = Arm(arm_key, recs)
            vals, missing = A.values(arm, lambda r: r.get("training_age"))
            cs, _ = A.values(arm, lambda r: r.get("constrained_steps_at_selection"))
            row: dict[str, Any] = {"task": task, "arm_id": arm_key, "contrast": "vs reference", "n": len(recs),
                                   "mean_training_age": None if vals is None else statistics.fmean(vals.values()),
                                   "mean_constrained_steps": None if cs is None else statistics.fmean(cs.values())}
            if arm_key != ref.arm_id and vals and ref_ages and len(vals) >= 2 and len(ref_ages) >= 2:
                w = stats.welch(vals, ref_ages)
                row.update(diff_from_reference=w.estimate, welch_lo=w.lo, welch_hi=w.hi,
                           systematic=bool(_excludes_zero(w.lo, w.hi)))
            ages.append(row)
    ages += [{**row, "used_by": " ".join(sorted(A.age_uses[key]))} for key, row in sorted(A.age_rows.items())]
    tables["A_training_age"] = ages
    tables["A_controller"] = [
        {k: r.get(k) for k in ("run_id", "arm_id", "task", "seed", "onset_step", "total_steps", "lambda_peak",
                                "lambda_final", "overshoot", "settling_steps", "recovery_steps", "recovery_censored")}
        for r in A.records
    ]
    tables["A_treatments_other_N"] = _other_n_rows(A)
    tables["A_gaps"] = _gap_rows(A)
    tables["A_estimands_side_by_side"] = _side_by_side_rows(A)
    tables["A_norms"] = _norm_rows(A)
    tables["A_iqm"] = _iqm_rows(A) if iqm else []  # after every tabled contrast, the descriptive ones at other N included
    tables["M11_exclusions"] = [dict(r) for r in dataset.exclusions]
    return tables


_ARM_COLUMNS = ("run_id", "arm_id", "task", "N", "onset_shape", "step_matching", "treatment", "controller_variant", "seed")


def _gap_rows(A: StudyA) -> list[dict[str, Any]]:
    """Analysis spec A1: equation (1) per seed, the one input of every Study A interval (Part 5.1: "the outcome
    per seed is the robustness gap of its matched checkpoint").

    One row per completed run: its gap per battery condition for the three estimands (the matched checkpoint's;
    rule 7 (a)'s at tolerance 5.0, the matched gap or, for an arm matched at 5.0 only, the sensitivity battery's;
    rule 7 (b)'s at the final checkpoint), and whether its arm is compared under rules 2 to 6 at each tolerance
    (rule 6: an unmatched arm "is not compared"; the final checkpoint has no matching filter). An empty cell is a
    gap not evaluated (yet): finetune and transfer until `enrich continuations` writes them, or a condition of an
    arm left out of the battery under rule 6.
    """
    compared = A.compared(PROPOSED, "matched") or set()
    compared_5 = A.compared(PROPOSED, "tolerance5") or set()
    rows = []
    for rec in A.records:
        arm = Arm(rec["arm_id"], [rec])
        row = {k: rec.get(k) for k in _ARM_COLUMNS}
        row.update(status=A.arm_status(PROPOSED, "matched", arm), compared=rec["arm_id"] in compared,
                   status_tolerance_5=A.arm_status(PROPOSED, "tolerance5", arm),
                   compared_tolerance_5=rec["arm_id"] in compared_5)
        for estimand, prefix in (("matched", ""), ("tolerance5", "tolerance5_"), ("final", "final_")):
            for condition in CONDITIONS:
                row[f"{prefix}gap_{condition}"] = A.gap(rec, condition, estimand)
        rows.append(row)
    return rows


def _side_by_side_rows(A: StudyA) -> list[dict[str, Any]]:
    """Analysis spec M9: the primary analysis's estimands side by side (Part 4.1.1: the final-checkpoint estimand
    "is reported beside the primary one; agreement between the two strengthens the causal reading, disagreement
    limits it"), with rule 7 (a)'s tolerance 5.0 beside them (exploratory).

    Rule 7 repeats "the primary analysis", box H1 on the primary outcome under both controls (Part 5.2), so the
    rows are Delta(N) of equation (2) for every late N and control on that cell, the same estimates the verdicts
    H1, A-rule7a and A-final-estimand read. The analysis spec's M8 and M9 ask for the whole Study A catalogue
    at tolerance 5.0 and at the final checkpoint; the registered text names the primary analysis only, so the
    other verdicts are not repeated, and their per-seed gaps under all three estimands are in table A_gaps.
    """
    rows = []
    task, condition, shape = R.PRIMARY_TASK, PRIMARY_CONDITION, PROPOSED.shape
    for control in R.STEP_MATCHING_CONTROLS:
        for N in R.LATE_ONSET_FRACTIONS:
            row: dict[str, Any] = {"task": task, "condition": condition, "shape": shape, "control": control, "N": N,
                                   "contrast": f"Delta({N:.2f})", "label": EXPLORATORY}
            found: dict[str, tuple[float, float, float]] = {}
            for estimand in ESTIMANDS:
                est, why = A.contrast(PROPOSED, task=task, x=A.arms(task, N=N, shape=shape, control=control),
                                      y=A.reference(task), get=A.gap_outcome(condition, estimand), estimand=estimand,
                                      filter_matched=estimand != "final",
                                      analysis_id=f"H1|{task}|{condition}|{control}|{shape}|{estimand}|N{N:.2f}")
                if est is None:
                    row.update({f"{estimand}_diff": None, f"{estimand}_note": why})
                    continue
                found[estimand] = (est.diff, est.welch.lo, est.welch.hi)
                row.update({f"{estimand}_diff": est.diff, f"{estimand}_welch_lo": est.welch.lo,
                            f"{estimand}_welch_hi": est.welch.hi, f"{estimand}_n": (len(est.seeds_x), len(est.seeds_y))})
            for other in ("final", "tolerance5"):
                both = "matched" in found and other in found
                row[f"agree_matched_{other}"] = agree(found["matched"], found[other]) if both else None
            rows.append(row)
    return rows


def _norm_rows(A: StudyA) -> list[dict[str, Any]]:
    """Analysis spec A21 (exploratory, descriptive): the parameter norms of Table 2.3 per run.

    The actor norm at onset (ledger norm_onset), at the matched and at the final checkpoint (the ledger's
    checkpoint records), and the actor (trainable and all parameters) and critic norms of the plasticity.csv
    rows at onset and at onset + 200,000 steps (the training supplement record; Q-plasticity-definitions).
    """
    rows = []
    for rec in A.records:
        row = {k: rec.get(k) for k in (*_ARM_COLUMNS, "onset_step", "matched_checkpoint_step", "norm_onset",
                                        "norm_matched_checkpoint", "norm_final_checkpoint")}
        for when, values in (rec.get("plasticity_norms") or {}).items():
            row.update({f"{when}.{name}": value for name, value in values.items()})
        row["label"] = EXPLORATORY
        rows.append(row)
    return rows


def _other_n_rows(A: StudyA) -> list[dict[str, Any]]:
    """Treatments and controller variants at N = 0.10 and 0.25: descriptive estimates, no verdict.

    Part 5.1 defines their estimands at N = 0.50 only (Q-treatment-other-N, HANDOVER section 9; not a
    PENDING key); Table 3.3 runs them at every late N, so they are reported as descriptive estimates (table
    A_treatments_other_N), with no verdict and no Holm.
    """
    rows = []
    for task in R.TASKS_STUDY_A:
        for N in R.LATE_ONSET_FRACTIONS:
            if N == LATE_N:
                continue
            untreated = A.arms(task, N=N, shape=TREATMENT_SHAPE, control=TREATMENT_CONTROL)
            for variant, kw in _treated_variants(task):
                treated = A.arms(task, N=N, shape=TREATMENT_SHAPE, control=TREATMENT_CONTROL, **kw)
                for condition in CONDITIONS:
                    est, why = A.contrast(PROPOSED, task=task, x=treated, y=untreated, get=A.gap_outcome(condition),
                                          analysis_id=f"other-N|{task}|{condition}|{variant}|N{N:.2f}")
                    base = {"task": task, "N": N, "variant": variant, "condition": condition,
                            "contrast": f"gap({variant}) - gap(untreated)", "label": "descriptive"}
                    rows.append({**base, **est.row()} if est is not None else {**base, "note": why})
    return rows


def iqm_rows(specs: dict[str, tuple[Any, Any, stats.EpisodeOutcome, float]], uses: dict[str, set[str]],
             records_of: Callable[[Any], list[dict[str, Any]]], name_of: Callable[[Any], str]) -> list[dict[str, Any]]:
    """Part 5.3's interquartile mean of every contrast the proposals tabled (descriptive; Q-iqm).

    One row per (x arm, y arm or none, episode source, level): IQM per arm, their difference and its
    two-level stratified bootstrap interval, the analysis ids that read the contrast, or "not
    computable" with the runs whose per-episode records are missing.
    """
    rows = []
    for key, (x, y, outcome, level) in sorted(specs.items()):
        base: dict[str, Any] = {"iqm_id": key, "x": name_of(x), "y": None if y is None else name_of(y),
                                "outcome": outcome.source, "level": level, "used_by": " ".join(sorted(uses.get(key, ()))),
                                "label": "descriptive"}
        parts: list[dict[int, Any]] = []
        missing: list[str] = []
        for arm in (x, y):
            if arm is None:
                continue
            by_seed = {}
            for rec in records_of(arm):
                episodes = outcome.episodes(rec)
                if episodes is None:
                    missing.append(rec["run_id"])
                else:
                    by_seed[int(rec["seed"])] = episodes
            parts.append(by_seed)
        if missing:
            rows.append({**base, "note": f"not computable: no per-episode records for {missing[:6]}"
                                         f"{' ...' if len(missing) > 6 else ''} (supplement records)"})
            continue
        try:
            result = stats.iqm_contrast(parts[0], parts[1] if len(parts) > 1 else None, analysis_id=key, level=level)
        except stats.StatsError as exc:
            rows.append({**base, "note": f"not computable: {exc}"})
            continue
        rows.append({**base, **result, "note": "descriptive; seeds, then episodes within each drawn seed, resampled (Q-iqm)"})
    return rows


def _iqm_rows(A: StudyA) -> list[dict[str, Any]]:
    """The IQM of every Study A contrast of an episode outcome (gaps, the measurement-set return) that the proposals tabled."""
    return iqm_rows(A.iqm_specs, A.iqm_uses, lambda arm: arm.records, lambda arm: arm.arm_id)

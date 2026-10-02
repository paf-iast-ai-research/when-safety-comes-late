"""Verdicts, their labels and the readings of open questions (analysis/__main__.py; Part 1.2; Part 8.2).

Owner: Role 4, Analysis and results (Muhammad Abdullah).

Part 1.2: "a supported hypothesis is one whose effect was large enough to show at this sample size,
and a falsified one is one whose effect is bounded below the minimum of interest. An outcome that
meets neither the supporting nor the falsifying condition is reported as inconclusive at this
sample size, with its interval, and is never rounded up to support." Box H0: H3 and H4 "are
reported as not tested, not as falsified" when H0 is supported.

Statuses: SUPPORTED, FALSIFIED, INCONCLUSIVE, NOT_TESTED, NOT_COMPUTABLE (with its reason), and
UNDECIDED(key) when the verdict under the proposal of an open question differs from its verdict
under one of that question's other readings (``READINGS``), even when one of the two is
NOT_COMPUTABLE or NOT_TESTED (the status under the proposals stays in ``proposal_status`` with its
reason). ``provisional_on`` lists every open key the verdict depends on (analysis.questions). Every
key is answered: in docs/DECISIONS.md (2026-10-02), draft rows that become amendments in Table 9.1
only when the group ratifies them, and listed in ``configs.registered.ANSWERED_QUESTIONS``
("answered" below means so). The verdicts are decided by the answers (``PROPOSED``, a name kept from
when they were proposals, "the proposals" below), and a key reopens only when the group changes its
answer, until the code follows; only then are its other readings computed. Labels (Part 5.2; Table
3.3; Part 8.2): confirmatory (H1 on the primary outcome; G1 and G2), secondary (every other
registered reading), exploratory (rule 7, anything unregistered, and anything an amendment made
after data were seen affects).

A secondary outcome without a registered criterion is read as a claim "the difference is not
zero": SUPPORTED when its interval excludes zero after Holm within its family (Part 5.2),
otherwise INCONCLUSIVE; never FALSIFIED (Part 5.7: "The statement 'no effect' is not used").

Part 5.5 ("Differences below the detectable size are reported with their intervals and are not
claimed as findings"; Q-detectable-size, HANDOVER section 9, not a PENDING key) is an annotation that
never decides: an effect the study cannot resolve (an interval including zero) is INCONCLUSIVE by the
boxes, and ``detectable_notes`` adds to a SUPPORTED verdict whose deciding estimate has |d| below the
80-percent point at its n that the verdict stands but the test had under 80 percent power for an
effect of that size, so its magnitude is imprecise.

Part 5.7's bound decides a falsification (Q-falsification-calibration, answered; Part 1.2:
"a falsified one is one whose effect is bounded below the minimum of interest"; Part 5.7: a bound above
it "is reported as inconclusive at this sample size"): a box whose falsifying quantity carries a Part
1.5 minimum effect (H1, H2, H3 (b), H4, G1, G2, G3) is FALSIFIED only if its box conditions hold and
its bound (``bound_annotation``: the 95 percent Welch limit in the direction the box claims) is below
the minimum effect; otherwise it is INCONCLUSIVE with its bound reported (``calibrate``). A bound that
cannot be read (no interval; a floored Study B budget compared on violation magnitude) does not
confirm the falsification. Whether the box conditions alone were met stays in the numbers
('falsification') and in ``calibrate``'s note. The key's other reading lets the box conditions decide.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Optional

from configs import registered as R

from analysis import matching, questions
from analysis.stats import settled

SUPPORTED = "SUPPORTED"
FALSIFIED = "FALSIFIED"
INCONCLUSIVE = "INCONCLUSIVE"
NOT_TESTED = "NOT_TESTED"
NOT_COMPUTABLE = "NOT_COMPUTABLE"
UNDECIDED = "UNDECIDED"
STATUSES = (SUPPORTED, FALSIFIED, INCONCLUSIVE, NOT_TESTED, NOT_COMPUTABLE, UNDECIDED)

CONFIRMATORY = "confirmatory"
SECONDARY = "secondary"
EXPLORATORY = "exploratory"
LABELS = (CONFIRMATORY, SECONDARY, EXPLORATORY)


@dataclass(frozen=True)
class Reading:
    """One reading of every analysis question; the defaults are the readings the verdicts are decided by (an open
    key's PENDING proposal; an answered key's answer in docs/DECISIONS.md, the code having been changed to it where
    it differed). ``READINGS`` lists the other readings computed while a key is open."""

    interval: str = "welch"  # Q-interval: the primary (Welch) interval decides | "percentile"
    shape: str = "abrupt"  # Q-h1-shape: H1 on the abrupt arms | "ramp"
    controls: str = "both"  # Q-controls-reading: falsified only if under both controls | "either"
    overlap: str = "falsified"  # Q-support-falsify-overlap: both conditions hold -> FALSIFIED | "supported"
    ordering: str = "trend_and_points"  # Q-support-falsify-overlap: support needs both | "trend"
    calibration: str = "bound"  # Q-falsification-calibration: falsified only with the Part 5.7 bound below | "box"
    holm: str = "both_families"  # Q-holm-families: a cell in two families survives Holm in both | "either" (in one)
    cost_field: str = matching.COST_FIELD  # Q-matched-cost-set | matching.SELECTION_COST_FIELD
    arithmetic: str = "proposed"  # Q-threshold-arithmetic | "alternative"
    h3_arms: str = "main"  # Q-h3-scope: late main-sweep arms | "all_late" (treatment and controller arms too)
    h3_reduce: str = "interval"  # Q-h3-scope: (b)'s absences = no reduction excluding zero | "point"
    h4_fall: str = "point"  # Q-h4-reading: "does not fall" on the point estimate | "interval"
    floor_arms: str = "all"  # Q-floor-rule: the seven arms count | "comparison" (the arms the comparison uses)
    floor_rate: str = "arm_mean"  # Q-floor-rule: each arm's mean rate over its seeds | "per_seed" (every seed's rate)
    g1_margin: str = "point"  # Q-g1-criteria: point > 5 pp and interval excludes 0 | "bound" (lower limit > 5 pp)
    g3_arms: str = "four"  # Q-g3-arms: Sparse, Moderate, Dense, Continuous | "seven"
    # Q-g4-reading: "within one horizon" = the spread of the three per-budget medians is below 1 | "at_most_one"
    g4_within: str = "less_than_one"
    # Q-g4-reading: the ordering on the arm statistic, the median over seeds of each seed's median over the budgets
    # off the few-shot floor | "per_budget" (the per-budget medians ordered on the majority of those budgets)
    g4_order: str = "arm_median"
    g4_censored: str = "after"  # Q-g4-reading: a censored value is the index after the largest horizon | "at" (Part 4.2)
    g2_outcome: str = "count"  # Q-g2-outcome: the box's per-budget count decides | "composite" (Part 5.2's mean)
    # Q-surplus-in-analysis: every completed seed (the H1 trend and H3 (a) always read analysis.data.five_seed_view) |
    # "five" (each arm's registered seeds and their replacements: analysis.data.five_seed_view)
    seeds: str = "all"

    @property
    def matching_arithmetic(self) -> matching.Arithmetic:
        return matching.PROPOSED_ARITHMETIC if self.arithmetic == "proposed" else matching.ALTERNATIVE_ARITHMETIC


PROPOSED = Reading()

# The other readings of each open question that the analysis can compute (each differs from the
# proposals in that question only, and answers the question the key asks: never "the registered rule
# is not applied"). A question with independent sub-readings lists each alone, so that a verdict
# reading both (box H3 reads (a) and (b)) cannot hide one sub-reading's disagreement behind the joint
# change agreeing by chance.
# * Q-g4-reading has three: 'within one horizon' as a spread of at most one (the decided reading: below
#   one, the exact negation of the support clause's 'at least one horizon'); the ordering clause read per
#   budget on the majority of the budgets (the decided reading: on the arm statistic); and a censored
#   value placed AT the largest horizon's index, as Part 4.2 registers it ("with adaptation steps censored
#   at the largest horizon when the target is never reached"; the decided reading follows Table 2.5's
#   "recorded as above the largest horizon"). The order is kept, so alternative(..., 2) is the censoring one.
# * Q-h3-scope has two and their joint reading: 'across late arms' as every late arm of the task
#   (treatment and controller arms too; H3 (a)), and the absences of H3 (b) ('... and additional constrained
#   training does not', 'injection does not', 'none of the three reduces it') as a point estimate that is not
#   negative.
# * Q-support-falsify-overlap has two and their joint reading: both conditions holding read SUPPORTED
#   (``box_status``), and H1's 'Delta increases with N' (and G1's ordering) read on the trend alone.
# * Q-holm-families asks how the two families of a Study A (task, condition) cell combine (Part 5.2:
#   "the four battery conditions form one family; the three tasks another"). Its other reading: the
#   claim survives Holm in at least one of its two families. Between "both" (the decided reading) and
#   "either" lie the single-family readings (conditions only, tasks only): when "both" and "either"
#   agree, so do they. A claim with one family (the four budgets of Study B, the three tasks of
#   recovery, return and H3 (c)) keeps Holm in that family under every reading.
# * Q-floor-rule asks which arms count and which rate (Part 4.2: "If more than half of the arms have a
#   satisfaction rate below 5 percent on an unseen budget"). Its other readings: only the arms the
#   comparison uses count; every seed's rate counts (more than half of the runs of the seven arms);
#   and both. The floor rule itself is registered and applies under every reading.
# * Q-g2-outcome's other reading decides G2 on Part 5.2's mean over the budgets; Q-surplus-in-analysis's
#   reads "five seeds" literally.
READINGS: Mapping[str, tuple[Mapping[str, str], ...]] = MappingProxyType({
    "Q-interval": ({"interval": "percentile"},),
    "Q-h1-shape": ({"shape": "ramp"},),
    "Q-controls-reading": ({"controls": "either"},),
    "Q-support-falsify-overlap": ({"overlap": "supported"}, {"ordering": "trend"},
                                  {"overlap": "supported", "ordering": "trend"}),
    "Q-falsification-calibration": ({"calibration": "box"},),
    "Q-holm-families": ({"holm": "either"},),
    "Q-matched-cost-set": ({"cost_field": matching.SELECTION_COST_FIELD},),
    "Q-threshold-arithmetic": ({"arithmetic": "alternative"},),
    "Q-h3-scope": ({"h3_arms": "all_late"}, {"h3_reduce": "point"}, {"h3_arms": "all_late", "h3_reduce": "point"}),
    "Q-h4-reading": ({"h4_fall": "interval"},),
    "Q-floor-rule": ({"floor_arms": "comparison"}, {"floor_rate": "per_seed"},
                     {"floor_arms": "comparison", "floor_rate": "per_seed"}),
    "Q-g1-criteria": ({"g1_margin": "bound"},),
    "Q-g2-outcome": ({"g2_outcome": "composite"},),
    "Q-g3-arms": ({"g3_arms": "seven"},),
    "Q-g4-reading": ({"g4_within": "at_most_one"}, {"g4_order": "per_budget"}, {"g4_censored": "at"}),
    "Q-surplus-in-analysis": ({"seeds": "five"},),
})


def alternatives(key: str) -> tuple[Reading, ...]:
    """The readings that differ from the proposals in ``key`` only."""
    return tuple(replace(PROPOSED, **overrides) for overrides in READINGS[key])


def alternative(key: str, index: int = 0) -> Reading:
    """One reading that differs from the proposals in ``key`` only (the first by default)."""
    return alternatives(key)[index]


def _describe(overrides: Mapping[str, str]) -> str:
    return ", ".join(f"{k}={v}" for k, v in overrides.items())


@dataclass
class Outcome:
    """What one reading of a verdict gives: its status and the numbers behind it."""

    status: str
    reason: str = ""
    numbers: dict[str, Any] = field(default_factory=dict)
    estimates: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class Verdict:
    """One registered reading (a box, a box part, a secondary claim) with everything the report prints."""

    id: str
    title: str
    statement: str
    label: str
    status: str
    reason: str = ""
    table_key: str = ""  # the analysis.questions entry
    conditions: tuple[str, ...] = ()
    undecided_by: tuple[str, ...] = ()
    provisional_on: tuple[str, ...] = ()
    proposal_status: str = ""  # the status under every proposal, before UNDECIDED
    readings: dict[str, str] = field(default_factory=dict)  # open key -> status under its other reading
    numbers: dict[str, Any] = field(default_factory=dict)
    estimates: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    group: str = ""  # report section

    @property
    def display_status(self) -> str:
        if self.status == UNDECIDED:
            return f"UNDECIDED({', '.join(self.undecided_by)})"
        return self.status

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "title": self.title, "statement": self.statement, "label": self.label,
            "status": self.status, "display_status": self.display_status, "proposal_status": self.proposal_status,
            "reason": self.reason, "undecided_by": list(self.undecided_by), "provisional_on": list(self.provisional_on),
            "readings": dict(self.readings), "numbers": dict(self.numbers),
            "estimates": [dict(e) for e in self.estimates],
            "notes": list(self.notes), "group": self.group, "conditions": list(self.conditions),
            "table_key": self.table_key,
        }


def computable_keys(table_key: str) -> list[str]:
    """The verdict's keys that have a computable other reading and are still open."""
    return [k for k in questions.keys_for(table_key) if k in READINGS and R.is_open(k)]


def decide(
    *,
    id: str,  # noqa: A002 -- the keyword is the callers' API (Verdict.id)
    title: str,
    statement: str,
    label: str,
    table_key: str,
    compute: Callable[[Reading], Outcome],
    conditions: Iterable[str] = (),
    group: str = "",
    extra_readings: Optional[Mapping[str, str]] = None,
) -> Verdict:
    """The verdict under the proposals, UNDECIDED(key) where an open key's other reading disagrees.

    ``compute`` is called once with ``PROPOSED`` and once per other reading of each open key of the
    verdict's table entry (``READINGS``). ``extra_readings`` adds statuses of other readings computed
    elsewhere (for example H0 from the readings of H1 and H2); each key must be one the verdict depends
    on (its table entry or its conditions' keys). Any disagreement across an open key gives
    UNDECIDED(key), in both directions: a verdict NOT_COMPUTABLE under the proposals (the data do not
    give it: an arm unmatched under the measurement-set cost, say) that another reading decides turns
    on that key as much as the reverse does; ``proposal_status`` and ``reason`` keep what the
    proposals give. A NOT_TESTED verdict (box H0 supported) becomes UNDECIDED likewise, since H0
    itself then turns on that key. ``readings[key]`` is the status under the key's first disagreeing
    reading (its only one, except Q-g4-reading, Q-h3-scope, Q-floor-rule and
    Q-support-falsify-overlap); when one of them disagrees, a note gives every reading of a key that
    has several.
    """
    conditions = tuple(conditions)
    base = compute(PROPOSED)
    if base.status not in STATUSES or base.status == UNDECIDED:
        raise ValueError(f"{id}: compute returned status {base.status!r}")
    own_keys = set(questions.keys_for(table_key, conditions=conditions))
    readings: dict[str, str] = {}
    several: list[str] = []
    for key in computable_keys(table_key):
        statuses = [compute(r).status for r in alternatives(key)]
        differing = [st for st in statuses if st != base.status]
        readings[key] = differing[0] if differing else base.status
        if len(statuses) > 1 and differing:
            several.append(f"{key}: " + "; ".join(f"{_describe(o)} gives {st}" for o, st in zip(READINGS[key], statuses)))
    for key, status in (extra_readings or {}).items():
        if key not in own_keys:
            raise ValueError(f"{id}: reading of {key} given, but {table_key} does not depend on it")
        # keep a computed reading that already disagrees; otherwise take the one given
        if R.is_open(key) and (key not in readings or readings[key] == base.status):
            readings[key] = status
    disagreeing = tuple(k for k, s in readings.items() if s != base.status)
    verdict = Verdict(
        id=id, title=title, statement=statement, label=label, status=base.status, reason=base.reason,
        table_key=table_key, conditions=conditions,
        provisional_on=tuple(questions.provisional_on(table_key, conditions=conditions)),
        proposal_status=base.status, readings=readings, numbers=base.numbers, estimates=base.estimates,
        notes=list(base.notes), group=group,
    )
    if disagreeing:
        verdict.status = UNDECIDED
        verdict.undecided_by = disagreeing
        verdict.notes.append(
            f"under the readings the code implements the verdict is {base.status}"
            + (f" ({base.reason})" if base.status == NOT_COMPUTABLE and base.reason else "")
            + "; under the other reading of "
            + "; ".join(f"{k} it is {readings[k]}" for k in disagreeing)
            + f" ({', '.join(disagreeing)} {'is' if len(disagreeing) == 1 else 'are'} open: the code does not yet "
              "follow the group's answer)"
        )
        verdict.notes += [f"the readings of {n}" for n in several]
    return verdict


def status_for_reading(verdict: Verdict, key: str) -> str:
    """The verdict's status under the other reading of ``key`` (its proposal status if that reading agrees)."""
    return verdict.readings.get(key, verdict.proposal_status)


def claim_status(excludes_zero: Optional[bool], survives_holm: Optional[bool] = True) -> str:
    """A secondary claim "the difference is not zero": SUPPORTED, INCONCLUSIVE, or NOT_COMPUTABLE."""
    if excludes_zero is None or survives_holm is None:
        return NOT_COMPUTABLE
    return SUPPORTED if excludes_zero and survives_holm else INCONCLUSIVE


def combine_controls(statuses: Mapping[str, str], reading: Reading) -> str:
    """Q-controls-reading: supported only if supported under both controls; falsified only if falsified
    under both (the other reading: under either); otherwise inconclusive.

    A control whose box is NOT_COMPUTABLE could turn out any status, so the box is decided only where the
    known controls decide it whatever the missing one gives (as inside a control: box_status). Under
    "both", a known INCONCLUSIVE control, or one SUPPORTED and one FALSIFIED, already rule out both
    "supported under both" and "falsified under both": the box is INCONCLUSIVE (Part 1.2: "An outcome
    that meets neither the supporting nor the falsifying condition is reported as inconclusive"). Under
    "either", the missing control could be FALSIFIED, so only a known FALSIFIED decides.
    """
    values = list(statuses.values())
    if not values:
        return NOT_COMPUTABLE
    if reading.controls == "either" and FALSIFIED in values:
        return FALSIFIED
    known = [v for v in values if v != NOT_COMPUTABLE]
    if reading.controls == "both" and (INCONCLUSIVE in known or (SUPPORTED in known and FALSIFIED in known)):
        return INCONCLUSIVE
    if len(known) < len(values):
        return NOT_COMPUTABLE
    if all(v == SUPPORTED for v in values):
        return SUPPORTED
    if all(v == FALSIFIED for v in values):
        return FALSIFIED
    return INCONCLUSIVE


def box_status(supported: Optional[bool], falsified: Optional[bool], reading: Reading, notes: list[str]) -> str:
    """A box's status from its two conditions (Q-support-falsify-overlap when both hold).

    A condition is None when the data cannot evaluate it (an estimate it needs is missing and no
    recorded cut explains the absence); the status is then NOT_COMPUTABLE unless the known condition
    decides it whatever the unknown one is (the caller gives the reason).
    """
    if supported and falsified:
        notes.append("the support and the falsification conditions both hold (Q-support-falsify-overlap)")
        return FALSIFIED if reading.overlap == "falsified" else SUPPORTED
    if falsified:
        return NOT_COMPUTABLE if supported is None and reading.overlap == "supported" else FALSIFIED
    if supported:
        return NOT_COMPUTABLE if falsified is None and reading.overlap == "falsified" else SUPPORTED
    if supported is False and falsified is False:
        return INCONCLUSIVE
    return NOT_COMPUTABLE


BOUND_BELOW = "evidence that any effect is smaller than practical relevance in this setting"  # Part 5.7
BOUND_ABOVE = "inconclusive at this sample size"  # Part 5.7
# the note on a box its conditions falsify whose Part 5.7 bound is not below the minimum effect (``calibrate``; box H1)
CALIBRATION_NOTE = ("falsified by the box but the Part 5.7 bound is not below the minimum effect: inconclusive at this "
                    "sample size (Part 1.2; Q-falsification-calibration)")


def bound_annotation(name: str, limit: Optional[float], minimum: float, *,
                     claimed: str) -> tuple[dict[str, Any], Optional[bool]]:
    """Part 5.7's bound on one estimand of a falsified box: its numbers and whether it is below the minimum.

    ``claimed`` is the direction of the effect the box claims: 'positive' (the bound is the upper limit,
    below the minimum effect when upper < minimum) or 'negative' (a reduction: the lower limit, below
    the minimum when lower > -minimum). Compared after ``stats.settled``.
    """
    if limit is None or math.isnan(limit):
        return {f"bound({name})": None, f"bound({name})_reading": "no interval"}, None
    below = settled(limit) < minimum if claimed == "positive" else settled(limit) > -minimum
    return {f"bound({name})": limit, f"bound({name})_limit": "upper" if claimed == "positive" else "lower",
            f"bound({name})_minimum_effect": minimum,
            f"bound({name})_reading": BOUND_BELOW if below else BOUND_ABOVE}, below


def calibrate(status: str, bounded: Optional[bool], reading: Reading, notes: list[str]) -> str:
    """Q-falsification-calibration (answered): a falsified box stands only if its Part 5.7 bound is
    below the minimum effect (Part 1.2; Part 5.7); a bound above it, or one that cannot be read (``bounded`` None),
    leaves the box INCONCLUSIVE ("inconclusive at this sample size"). The other reading lets the box conditions
    decide."""
    if status != FALSIFIED:
        return status
    notes.append("Part 5.7: the falsified box is reported with its bound against the minimum effect (bound(...) numbers)")
    if reading.calibration == "bound" and not bounded:
        notes.append(CALIBRATION_NOTE)
        return INCONCLUSIVE
    return status


def detectable_notes(estimates: Mapping[str, Any]) -> list[str]:
    """Part 5.5 on a SUPPORTED verdict (Q-detectable-size): the deciding estimates whose |d| is below the 80-percent
    point at their n. An annotation only: the verdict stands (its interval excludes zero). The note is the decided
    one (Q-detectable-size) with the estimate's name added after its |d|."""
    power = f"{R.POWER_TARGET * 100:g}"
    notes = []
    for name, est in estimates.items():
        if est is not None and est.below_detectable:
            notes.append(f"Part 5.5: |d| = {abs(est.cohens_d):.3g} ({name}) is below the {power}-percent detectable "
                         f"size {est.detectable_d:.3g} at n = {min(len(est.seeds_x), len(est.seeds_y))}; the interval "
                         f"excludes zero and the verdict stands; the test had under {power} percent power for an "
                         "effect of this size, so its magnitude is imprecise (see its interval; annotation only)")
    return notes

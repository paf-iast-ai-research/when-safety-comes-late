"""Study B: Part 4.2 comparisons, the floor rule, and boxes G1 to G5 (Parts 1.4, 4.2, 5).

Owner: Role 4, Analysis and results (Muhammad Abdullah).

Outcomes (Part 5.1): "for each arm and unseen budget, the outcome per seed is the satisfaction rate of
equation (10), zero-shot and after each few-shot horizon, and the adaptation steps". Primary (Part
5.2): "the mean zero-shot satisfaction rate over the four unseen budgets, one number per seed. The
primary analysis is G1 and G2 on that outcome, each tested at α = 0.025 (Bonferroni within the
primary family of two), so that for G1 and G2 an interval that must exclude zero is the 97.5 percent
interval." G3, G4, G5, satisfaction per budget and violation magnitude are secondary ("the four
unseen budgets" are a Holm family).

Part 4.2 (p. 15), verbatim where it decides:
* "G1 compares Sparse, Moderate and Dense with one another on the mean satisfaction rate over the four
  unseen budgets; G5 compares Dense with Continuous on the same outcome."
* "G2 compares Dense with the single-level arm nearest to each unseen budget by equation (11). Budget 5
  has one nearest arm, Single-10; budget 45 has one, Single-40. Budgets 15 and 30 are each equidistant
  from two single-level arms; for those the comparator is whichever of the two has the higher
  satisfaction on that budget, the choice least favourable to G2, and both are reported. G2 never
  compares Dense only with Single-10."
* "G3 compares, within each arm, the drop in satisfaction at budget 5 with the drop at budget 45, each
  drop measured from the arm's satisfaction at its own nearest training level."
* "G4 compares the adaptation steps of Sparse, Moderate and Dense on each unseen budget, with
  adaptation steps censored at the largest horizon when the target is never reached."
* "Floor rule, for every comparison above. If more than half of the arms have a satisfaction rate
  below 5 percent on an unseen budget, that budget's comparison is made on violation magnitude (Table
  2.5) instead of satisfaction rate, and the report says so; a rate at the floor cannot separate arms."

Boxes (Part 1.4) are quoted at each function. Analysis questions as answered in Table 9.1: Q-floor-rule
(all seven arms by arm-mean zero-shot rate; floored budgets leave the four-budget means and are
compared on violation magnitude by direction and interval, with no 5-point margin; the few-shot floor
per (budget, horizon); G4: a budget floored at every few-shot horizon leaves G4's majority and Dense
and Sparse are compared there on the few-shot violation magnitude at the largest horizon,
``StudyB.fewshot_floor``; its other readings count only the arms of the comparison, or every seed's
rate, or both: ``StudyB.floor``; the first two are reported beside the verdicts as sensitivity readings,
``SENSITIVITY_KEYS``), Q-g1-criteria, Q-g2-outcome, Q-g3-arms and Q-g4-reading (their other readings
reported beside as sensitivity readings), Q-interval, Q-support-falsify-overlap,
Q-falsification-calibration, Q-holm-families (``_budget_claims``), Q-iqm
(``StudyB._note_iqm``), Q-surplus-in-analysis (``StudyB.for_reading``: every Study B arm has the same
surplus count, so G1's trend keeps every seed), Q-adapt-censoring (read in analysis.data) and
Q-arm-complete (read in analysis.data; ``StudyB.awaited``).
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Optional, Sequence

from configs import registered as R

from analysis import questions, stats
from analysis.data import Dataset, five_seed_view
from analysis.stats import settled
from analysis.study_a import iqm_rows
from analysis.verdict import (
    BOUND_ABOVE, BOUND_BELOW, CONFIRMATORY, FALSIFIED, INCONCLUSIVE, NOT_COMPUTABLE, PROPOSED, READINGS, SECONDARY,
    SUPPORTED, Outcome, Reading, Verdict, bound_annotation, box_status, calibrate, claim_status, decide,
    detectable_notes,
)

G1_ARMS = ("Sparse", "Moderate", "Dense")  # Part 4.2 / box G1: 2 < 3 < 5 levels
G3_ARMS = ("Sparse", "Moderate", "Dense", "Continuous")  # Q-g3-arms (answered in Table 9.1): the arms spanning [10, 40]
G4_ARMS = G1_ARMS  # box G4: sparse, moderate, dense
# An implementation reading (no PENDING key): the G1 pairs and G5's pair, per budget (Part 5.2 secondary:
# satisfaction per budget, violation magnitude)
PER_BUDGET_CONTRASTS = (("Dense", "Sparse"), ("Moderate", "Sparse"), ("Dense", "Moderate"), ("Continuous", "Dense"))
# Answered in Table 9.1, these keys report their other readings beside every verdict that depends on them, as
# sensitivity readings that never decide (``_decide``): Q-g3-arms ("The all-seven-arms reading ... is
# reported beside as a sensitivity reading"), Q-g4-reading ("Three sensitivity readings are reported beside the
# verdict") and Q-floor-rule ("The readings 'only the arms of the comparison' and 'per-seed rates' are reported beside
# as sensitivity readings").
SENSITIVITY_KEYS = ("Q-g3-arms", "Q-g4-reading", "Q-floor-rule")


# ---------------------------------------------------------------------------
# Equation (11) and the reference levels
# ---------------------------------------------------------------------------


def single_arms() -> dict[str, float]:
    """The single-level arms and their budget (Table 3.4)."""
    return {arm: levels[0] for arm, levels in R.STUDY_B_ARMS.items() if levels is not None and len(levels) == 1}


def distance(budget: float, arm: str) -> float:
    """Equation (11): dist(d_test) = min over the arm's training set of |d_test - d| (Continuous: its interval)."""
    levels = R.STUDY_B_ARMS[arm]
    if levels is None:
        lo, hi = R.CONTINUOUS_RANGE
        return max(lo - budget, 0.0, budget - hi)
    return min(abs(budget - level) for level in levels)


def nearest_single_arms(budget: float) -> tuple[str, ...]:
    """The single-level arms at the least equation (11) distance from ``budget``, by increasing level."""
    singles = single_arms()
    best = min(abs(budget - level) for level in singles.values())
    return tuple(sorted((a for a, level in singles.items() if abs(budget - level) == best), key=singles.get))


def nearest_training_level(arm: str, budget: float) -> float:
    """The arm's own training level nearest to ``budget`` (Continuous: the budget clipped to its range)."""
    levels = R.STUDY_B_ARMS[arm]
    if levels is None:
        lo, hi = R.CONTINUOUS_RANGE
        return min(max(budget, lo), hi)
    return min(levels, key=lambda level: (abs(budget - level), level))


# ---------------------------------------------------------------------------
# Per-seed outcomes and the floor rule
# ---------------------------------------------------------------------------


def _indicators(costs: list[float], budget: float) -> list[float]:
    """Equation (10) per episode: 1 when the episode's cost is within the budget (inclusive), else 0."""
    return [1.0 if c <= budget else 0.0 for c in costs]


def _violations(costs: list[float], budget: float) -> list[float]:
    """Table 2.5 violation magnitude per episode: max(cost - budget, 0)."""
    return [max(c - budget, 0.0) for c in costs]


def _zero_parts(rec: dict[str, Any], budgets: list[float], per_episode: Callable[[list[float], float], list[float]]
                ) -> Optional[list[tuple[float, list[float]]]]:
    """The zero-shot episodes behind an outcome at ``budgets`` (pooled: equal counts, so the pooled mean is
    the mean over the budgets), or None without the zero-shot record."""
    episodes = rec.get("zero_episodes")
    if not episodes or any(b not in episodes for b in budgets):
        return None
    return [(1.0, [v for b in budgets for v in per_episode(episodes[b], b)])]


_OUTCOMES: dict[tuple, stats.EpisodeOutcome] = {}


def _sr(budget: float) -> stats.EpisodeOutcome:
    """Zero-shot satisfaction at one unseen budget (equation (10)), with its per-episode indicators."""
    key = ("sr", budget)
    if key not in _OUTCOMES:
        _OUTCOMES[key] = stats.EpisodeOutcome(
            source=f"sr_zero|b{budget:g}", value=lambda r: (r["sr_zero"] or {}).get(budget),
            episodes=lambda r: _zero_parts(r, [budget], _indicators))
    return _OUTCOMES[key]


def _vm(budget: float) -> stats.EpisodeOutcome:
    """Zero-shot violation magnitude at one unseen budget (Table 2.5), with its per-episode values."""
    key = ("vm", budget)
    if key not in _OUTCOMES:
        _OUTCOMES[key] = stats.EpisodeOutcome(
            source=f"vm_zero|b{budget:g}", value=lambda r: (r["vm_zero"] or {}).get(budget),
            episodes=lambda r: _zero_parts(r, [budget], _violations))
    return _OUTCOMES[key]


def _mean_sr(budgets: Sequence[float]) -> stats.EpisodeOutcome:
    """Part 5.2's primary outcome: the mean zero-shot satisfaction over ``budgets``, one number per seed."""
    budgets = tuple(budgets)
    key = ("mean_sr", budgets)
    if key not in _OUTCOMES:
        def value(r: dict[str, Any]) -> Optional[float]:
            values = [(r["sr_zero"] or {}).get(b) for b in budgets]
            return None if not budgets or any(v is None for v in values) else statistics.fmean(values)
        _OUTCOMES[key] = stats.EpisodeOutcome(
            source="sr_zero_mean|" + ",".join(f"{b:g}" for b in budgets), value=value,
            episodes=lambda r: _zero_parts(r, list(budgets), _indicators))
    return _OUTCOMES[key]


@dataclass
class StudyB:
    """Study B's completed runs and the cached two-arm estimates behind every box."""

    dataset: Dataset
    records: list[dict[str, Any]] = field(default_factory=list)
    _e2: dict = field(default_factory=dict)
    # Only the proposals' estimates are tabled (another reading may use other budgets or seeds under the
    # same analysis id), with the interquartile mean of every contrast of an episode outcome (Q-iqm: at 95 percent).
    estimate_rows: dict[str, dict[str, Any]] = field(default_factory=dict)
    iqm_specs: dict[str, tuple[str, Optional[str], stats.EpisodeOutcome, float]] = field(default_factory=dict)
    iqm_uses: dict[str, set[str]] = field(default_factory=dict)
    _five: Optional["StudyB"] = None

    def __post_init__(self) -> None:
        self.records = list(self.dataset.study_b)

    def for_reading(self, reading: Reading) -> "StudyB":
        """The data a reading compares: every completed seed (Q-surplus-in-analysis, answered in Table 9.1; Part 5.5
        gives every Study B arm the same surplus count), or five per arm (the key's other reading)."""
        if reading.seeds == "all":
            return self
        if self._five is None:
            view = five_seed_view(self.dataset)
            self._five = self if view is self.dataset else StudyB(view)
        return self._five

    def arm(self, arm: str) -> list[dict[str, Any]]:
        """The arm's completed runs, by seed."""
        return sorted((r for r in self.records if r["arm"] == arm), key=lambda r: r["seed"])

    def awaited(self, arm: str) -> str:
        """Why the arm's data are still to come ('' when they are not): final mode, fewer completed runs than its
        seed target (``Dataset.awaited_b``; Q-arm-complete). Read on the whole dataset, under every reading."""
        if arm not in self.dataset.awaited_b:
            return ""
        n, target = self.dataset.awaited_b[arm]
        return f"{arm}: {n} of its {target} seeds completed (Q-arm-complete)"

    def per_seed(self, arm: str, get: Callable[[dict[str, Any]], Any]) -> tuple[Optional[dict[int, float]], list[str]]:
        """{seed: value} of the arm's completed runs, or (None, what is missing): a run without the value, or, for
        an arm whose seeds are still to come (``awaited``), the arm itself."""
        if self.awaited(arm):
            return None, [self.awaited(arm)]
        out, missing = {}, []
        for rec in self.arm(arm):
            value = get(rec)
            if value is None or not math.isfinite(float(value)):
                missing.append(rec["run_id"])
            else:
                out[int(rec["seed"])] = float(value)
        return (None, missing) if missing else (out, [])

    def e2(self, x: dict[int, float], y: dict[int, float], analysis_id: str, level: float, *,
           record: bool) -> stats.TwoArmEstimate:
        """The cached two-arm estimate of ``x`` minus ``y``, tabled when ``record`` (the proposal's)."""
        key = (analysis_id, level, tuple(sorted(x.items())), tuple(sorted(y.items())))
        if key not in self._e2:
            self._e2[key] = stats.estimate_two_arms(x, y, analysis_id=analysis_id, level=level)
        if record:
            self.estimate_rows[f"{analysis_id}|{level}"] = self._e2[key].row()
        return self._e2[key]

    def _note_iqm(self, x_arm: str, y_arm: Optional[str], get: Callable[[dict[str, Any]], Any], analysis_id: str,
                  level: float) -> None:
        """Note an episode outcome's contrast for table B_iqm (Q-iqm), with the analysis that uses it."""
        if isinstance(get, stats.EpisodeOutcome):
            key = f"IQM|{x_arm}|{y_arm or 'within'}|{get.source}|{level}"
            self.iqm_specs.setdefault(key, (x_arm, y_arm, get, level))
            self.iqm_uses.setdefault(key, set()).add(analysis_id)

    def contrast(self, x_arm: str, y_arm: str, get: Callable[[dict[str, Any]], Any], analysis_id: str,
                 level: float = R.INTERVAL_LEVEL, *, reading: Reading) -> tuple[Optional[stats.TwoArmEstimate], str]:
        """``x_arm`` minus ``y_arm`` on the per-seed outcome ``get``, or (None, why)."""
        x, xm = self.per_seed(x_arm, get)
        y, ym = self.per_seed(y_arm, get)
        if x is None or y is None:
            return None, f"incomplete: no value for {xm + ym}"
        if len(x) < 2 or len(y) < 2:
            return None, f"fewer than two seeds ({x_arm}: {len(x)}, {y_arm}: {len(y)})"
        record = reading == PROPOSED
        if record:
            # Q-iqm: "Its 95 percent percentile interval", whatever the contrast's level (G1 and G2: 97.5 percent), so
            # one pair of arms on one source gets one IQM row whichever verdict reads it
            self._note_iqm(x_arm, y_arm, get, analysis_id, R.INTERVAL_LEVEL)
        return self.e2(x, y, analysis_id, level, record=record), ""

    # -- floor rule --------------------------------------------------------

    def floor(self, reading: Reading, arms: Optional[Callable[[float], Sequence[str]]] = None
              ) -> dict[float, dict[str, Any]]:
        """Part 4.2 floor rule per unseen budget: "If more than half of the arms have a satisfaction rate
        below 5 percent on an unseen budget, that budget's comparison is made on violation magnitude".

        The rule is registered and applies under every reading; Q-floor-rule asks "which arms count and
        which rate". Answered in Table 9.1: all seven arms, each arm's mean zero-shot rate over its seeds. Other
        readings (``Reading.floor_arms``, ``Reading.floor_rate``), reported beside the verdicts as sensitivity
        readings (``SENSITIVITY_KEYS``): only the arms the comparison uses
        count (``arms(budget)``, given by each comparison; all seven when it gives none), and every
        seed's rate counts (Part 5.1: "the outcome per seed is the satisfaction rate"): more than half
        of the runs of the counted arms below 5 percent.

        A unit whose rate is unknown (a completed run without it; an arm with no completed run, which
        counts as one arm or, per seed, as its seed target when known, else the larger of the registered
        seed count (``len(R.SEEDS)``) and the largest seed count of the arms present; per seed, an arm
        whose seeds are still to come adds its missing seeds as unknown runs) is never dropped from the
        count: the budget is floored when more than half of the units are known to be below
        the floor rate, not floored when that cannot happen whatever the unknown rates are, and
        otherwise undecided (``floored`` None; the comparisons that need the budget are NOT_COMPUTABLE,
        naming the unknown units).
        """
        out = {}
        for budget in R.UNSEEN_BUDGETS:
            members = tuple(R.STUDY_B_ARMS) if reading.floor_arms == "all" or arms is None else tuple(arms(budget))
            out[budget] = self._floor_count(reading, members, _sr(budget), "zero-shot rate")
        return out

    def fewshot_floor(self, reading: Reading, arms: Sequence[str]) -> dict[float, dict[str, Any]]:
        """The floor rule for G4 (Q-floor-rule, answered in Table 9.1; Part 4.2: "for every comparison above ... a rate at
        the floor cannot separate arms"): adaptation steps are read off the few-shot satisfaction rates, so
        a budget is floored for G4 when the floor rule holds at every few-shot horizon of its
        continuations: at each horizon, more than half of the counted units have a few-shot rate below 5
        percent (the arms and the rate counted as in ``floor``; ``arms`` when the reading counts only the
        comparison's arms). It is undecided (None) when no horizon is known to be off the floor and some
        horizon is undecided, or when no member has a few-shot rate. Per budget: ``floored``, ``horizons``,
        ``by_horizon`` (each horizon's ``_floor_count``), ``counted``, ``unit`` and ``unknown``."""
        members = tuple(R.STUDY_B_ARMS) if reading.floor_arms == "all" else tuple(arms)
        out = {}
        for budget in R.UNSEEN_BUDGETS:
            horizons = sorted({h for arm in members for rec in self.arm(arm)
                               for h in ((rec["fewshot"].get(budget) or {}).get("sr") or {})})
            per_h = {h: self._floor_count(reading, members, _fewshot_rate(budget, h), f"few-shot rate at {h}")
                     for h in horizons}
            states = [info["floored"] for info in per_h.values()]
            if not states:
                floored: Optional[bool] = None
            elif False in states:
                floored = False
            elif None in states:
                floored = None
            else:
                floored = True
            unknown = {f"{k} (horizon {h})": v for h, info in per_h.items() for k, v in info["unknown"].items()}
            out[budget] = {"floored": floored, "horizons": horizons, "by_horizon": per_h, "counted": list(members),
                           "unit": "run" if reading.floor_rate == "per_seed" else "arm",
                           "unknown": unknown if horizons else {"every counted arm": "no few-shot rate"}}
        return out

    def _floor_count(self, reading: Reading, members: Sequence[str], get: Callable[[dict[str, Any]], Any],
                     what: str) -> dict[str, Any]:
        """One floor-rule count over ``members`` of the rate ``get`` (``floor``); unknown units never dropped."""
        per_seed = reading.floor_rate == "per_seed"
        present = [len(self.arm(a)) for a in R.STUDY_B_ARMS if self.arm(a)]
        seeds_if_absent = max([len(R.SEEDS), *present])  # Part 5.5 (surplus): the same seed count for every Study B arm
        means: dict[str, float] = {}
        unknown: dict[str, str] = {}
        below: list[str] = []
        units = 0
        n_unknown = 0
        for arm in members:
            recs = self.arm(arm)
            if not recs:
                unknown[arm] = "no completed run"
                n_abs = self.dataset.awaited_b[arm][1] if arm in self.dataset.awaited_b else seeds_if_absent
                n_unknown += n_abs if per_seed else 1
                units += n_abs if per_seed else 1
                continue
            if per_seed:
                if self.awaited(arm):  # the seeds still to come are unknown units, never dropped
                    n, target = self.dataset.awaited_b[arm]
                    unknown[arm] = self.awaited(arm)
                    n_unknown += target - n
                    units += target - n
                for rec in recs:
                    value = get(rec)
                    units += 1
                    if value is None or not math.isfinite(float(value)):
                        unknown[rec["run_id"]] = f"no {what}"
                        n_unknown += 1
                    elif settled(float(value)) < R.FLOOR_SATISFACTION_RATE:
                        below.append(rec["run_id"])
                values, _ = self.per_seed(arm, get)
                if values is not None:
                    means[arm] = statistics.fmean(values.values())
                continue
            units += 1
            values, missing = self.per_seed(arm, get)
            if values is None:
                unknown[arm] = self.awaited(arm) or f"no {what} for {missing}"
                n_unknown += 1
                continue
            means[arm] = statistics.fmean(values.values())
            if settled(means[arm]) < R.FLOOR_SATISFACTION_RATE:
                below.append(arm)
        share = R.FLOOR_ARM_SHARE * units
        if len(below) > share:
            floored: Optional[bool] = True
        elif len(below) + n_unknown <= share:
            floored = False
        else:
            floored = None
        return {"arms_with_known_mean": len(means), "counted": list(members), "units": units,
                "unit": "run" if per_seed else "arm", "below": sorted(below), "unknown": unknown,
                "floored": floored, "arm_means": means}


def _fewshot_rate(budget: float, horizon: int) -> Callable[[dict[str, Any]], Optional[float]]:
    """A run's few-shot satisfaction rate at (budget, horizon), or None."""
    return lambda rec: ((rec["fewshot"].get(budget) or {}).get("sr") or {}).get(horizon)


def _fewshot_vm(budget: float) -> Callable[[dict[str, Any]], Optional[float]]:
    """A run's few-shot violation magnitude at the largest horizon of its continuation at ``budget`` (Q-floor-rule,
    answered in Table 9.1: the magnitude a floored G4 budget is compared on), or None."""
    def get(rec: dict[str, Any]) -> Optional[float]:
        entry = rec["fewshot"].get(budget) or {}
        vm, horizons = entry.get("vm") or {}, entry.get("horizons") or []
        return vm.get(max(horizons)) if horizons else None
    return get


def _absent(B: StudyB, arms: Sequence[str]) -> str:
    """The reason a box cannot be read when an arm it compares has no completed run, or (final mode) fewer than its
    seed target, its data still to come (``StudyB.awaited``; an 'incomplete' reason: a Holm family keeps the cell as
    an incomplete member). Said before the floor rule."""
    waiting = [B.awaited(a) for a in arms if B.awaited(a)]
    if waiting:
        return f"incomplete: data still to come ({'; '.join(waiting)})"
    missing = [a for a in arms if not B.arm(a)]
    return f"no completed run of {missing}" if missing else ""


def _undecided_floor(floor: dict[float, dict[str, Any]], budgets: Sequence[float], what: str = "floor rule") -> str:
    """The reason a comparison cannot be made: the floor rule (``what``) is undecided at these budgets."""
    parts = [f"budget {b:g}: the {what} cannot be decided without {floor[b]['unknown']} (Q-floor-rule: counting "
             f"{floor[b]['unit']}s of {', '.join(floor[b]['counted'])})"
             for b in budgets if floor[b]["floored"] is None]
    return "; ".join(parts)


def _numbers(prefix: str, est: Optional[stats.TwoArmEstimate], reading: Reading) -> dict[str, Any]:
    """The flat numbers of one estimate under ``prefix`` (``{prefix: None}`` without one)."""
    if est is None:
        return {prefix: None}
    lo, hi = est.interval(reading.interval)
    return {prefix: est.diff, f"{prefix}_welch_lo": est.welch.lo, f"{prefix}_welch_hi": est.welch.hi,
            f"{prefix}_boot_lo": est.bootstrap.lo, f"{prefix}_boot_hi": est.bootstrap.hi, f"{prefix}_p": est.welch.p,
            f"{prefix}_d": est.cohens_d, f"{prefix}_level": est.level, f"{prefix}_decides_lo": lo,
            f"{prefix}_decides_hi": hi, f"{prefix}_detectable_d": est.detectable_d,
            f"{prefix}_below_detectable": est.below_detectable}


def _vm_comparisons(B: StudyB, reading: Reading, floored: list[float], x_arm: str, y_arm: str, level: float,
                    tag: str, vm: Optional[Callable[[float], Any]] = None, what: str = "violation magnitude"
                    ) -> tuple[dict[str, Any], list[str]]:
    """Floored budgets compared on violation magnitude: direction and interval only (Q-floor-rule). ``vm``
    gives the per-seed magnitude at a budget (default: zero-shot, ``_vm``)."""
    numbers, notes = {}, []
    vm = vm or _vm
    prefix = "VM" if vm is _vm else "VM_fewshot"
    for budget in floored:
        est, why = B.contrast(x_arm, y_arm, vm(budget), f"{tag}|vm|b{budget:g}", level, reading=reading)
        numbers.update(_numbers(f"{prefix}({x_arm}) - {prefix}({y_arm}) at {budget:g}", est, reading))
        if est is None:
            notes.append(f"floor rule at budget {budget:g}: {what} not computable ({why})")
        else:
            lo, hi = est.interval(reading.interval)
            better = settled(est.diff) < 0 and hi < 0
            notes.append(f"floor rule applied at budget {budget:g}: compared on {what}; "
                         f"{x_arm} {'has' if better else 'does not have'} a lower violation magnitude with an "
                         "interval excluding zero")
    return numbers, notes


# ---------------------------------------------------------------------------
# G1
# ---------------------------------------------------------------------------


def g1_outcome(B: StudyB, reading: Reading) -> Outcome:
    """Box G1: "The mean satisfaction rates over the four unseen budgets are ordered sparse < moderate <
    dense, and dense minus sparse exceeds 5 percentage points with an interval excluding zero" /
    falsified: "The three multi-level arms' means lie within 5 percentage points of one another with
    overlapping intervals, or are not ordered as stated". 97.5 percent intervals (Part 5.2); the
    ordering is also Part 5.4's Spearman trend on the number of levels (Q-g1-criteria)."""
    B = B.for_reading(reading)
    level = R.INTERVAL_LEVEL_PRIMARY_STUDY_B
    if _absent(B, G1_ARMS):
        return Outcome(NOT_COMPUTABLE, _absent(B, G1_ARMS))
    floor = B.floor(reading, lambda b: G1_ARMS)
    undecided = _undecided_floor(floor, R.UNSEEN_BUDGETS)
    if undecided:
        return Outcome(NOT_COMPUTABLE, undecided)
    budgets = [b for b in R.UNSEEN_BUDGETS if not floor[b]["floored"]]
    floored = [b for b in R.UNSEEN_BUDGETS if floor[b]["floored"]]
    if not budgets:
        vm_numbers, vm_notes = _vm_comparisons(B, reading, floored, "Dense", "Sparse", level, "G1")
        return Outcome(NOT_COMPUTABLE, "every unseen budget is at the floor: satisfaction cannot separate the arms "
                                       "(the violation-magnitude comparisons are reported)", vm_numbers, [], vm_notes)
    outcome = _mean_sr(budgets)
    means: dict[str, dict[int, float]] = {}
    for arm in G1_ARMS:
        values, missing = B.per_seed(arm, outcome)
        if values is None or len(values) < 2:
            why = f"incomplete: no value for {missing}" if missing else "fewer than two seeds"
            return Outcome(NOT_COMPUTABLE, f"{arm}: {why}")
        means[arm] = values
    m = {arm: statistics.fmean(v.values()) for arm, v in means.items()}
    ordered = settled(m["Sparse"]) < settled(m["Moderate"]) < settled(m["Dense"])
    points: dict[int, list[tuple[float, float]]] = {}
    for arm, values in means.items():
        for seed, value in values.items():
            points.setdefault(seed, []).append((float(len(R.STUDY_B_ARMS[arm])), value))
    trend = stats.spearman_bootstrap(points, analysis_id="G1|trend", level=level)
    trend_ok = (not math.isnan(trend.rho) and not math.isnan(trend.lo)) and trend.rho > 0 and trend.lo > 0
    ds, _ = B.contrast("Dense", "Sparse", outcome, "G1|Dense-Sparse", level, reading=reading)
    lo, hi = ds.interval(reading.interval)
    margin = ((settled(ds.diff) > R.MIN_EFFECT_STUDY_B and lo > 0) if reading.g1_margin == "point"
              else settled(lo) > R.MIN_EFFECT_STUDY_B)
    supported = trend_ok and margin and (ordered or reading.ordering == "trend")
    per_arm = {arm: stats.one_sample(v, level) for arm, v in means.items()}
    overlap = all(per_arm[a].lo <= per_arm[b].hi and per_arm[b].lo <= per_arm[a].hi
                  for i, a in enumerate(G1_ARMS) for b in G1_ARMS[i + 1:])
    within = settled(max(m.values()) - min(m.values())) <= R.MIN_EFFECT_STUDY_B
    falsified = (not ordered) or (within and overlap)
    notes: list[str] = []
    status = box_status(supported, falsified, reading, notes)
    bound95 = stats.relevel(ds.welch, R.INTERVAL_LEVEL).hi
    # Part 5.7 ("If H1 or G1 is falsified, the result is reported as an upper bound"): the bound and its reading are
    # attached to a box falsified by its conditions only (kept when the bound turns it INCONCLUSIVE, as the reason
    # for that), never beside a supported one. The bound decides (Q-falsification-calibration, answered in Table 9.1).
    box_falsified = status == FALSIFIED
    if status == FALSIFIED and reading.calibration == "bound" and not settled(bound95) < R.MIN_EFFECT_STUDY_B:
        status = INCONCLUSIVE
        notes.append("falsified by the box but the Part 5.7 bound is not below the minimum effect: inconclusive at "
                     "this sample size (Part 1.2; Q-falsification-calibration)")
    if status == SUPPORTED:
        notes += detectable_notes({"Dense - Sparse": ds})
    numbers: dict[str, Any] = {f"mean_sr({a})": v for a, v in m.items()}
    numbers.update({"budgets_on_satisfaction": budgets, "budgets_floored": floored, "ordered": ordered,
                    "trend_rho": trend.rho, "trend_lo": trend.lo, "trend_hi": trend.hi, "trend_level": level,
                    "trend_undefined_resamples": trend.n_nan, "dense_minus_sparse_exceeds_margin": margin,
                    "within_5pp": within, "pairwise_overlap_97_5": overlap, "support": supported,
                    "falsification": falsified})
    if box_falsified:
        numbers.update({"bound_95_upper_Dense_minus_Sparse": bound95,
                        "bound_97_5_upper_Dense_minus_Sparse": ds.welch.hi,
                        "bound_reading": BOUND_BELOW if settled(bound95) < R.MIN_EFFECT_STUDY_B else BOUND_ABOVE})
    numbers.update(_numbers("Dense - Sparse", ds, reading))
    for arm, iv in per_arm.items():
        numbers[f"one_sample_97_5({arm})"] = (iv.lo, iv.hi)
    vm_numbers, vm_notes = _vm_comparisons(B, reading, floored, "Dense", "Sparse", level, "G1")
    numbers.update(vm_numbers)
    notes += vm_notes
    if trend.flagged:  # Q-bootstrap-details: undefined resamples "are dropped and counted, and flagged above 1 percent"
        notes.append(f"{trend.n_nan} of {trend.resamples} trend resamples were undefined (over "
                     f"{stats.NAN_FLAG_SHARE:.0%})")
    if status == FALSIFIED:
        notes.append("Part 5.7: reported as an upper bound (bound_95_upper_Dense_minus_Sparse; the 97.5 percent limit "
                     f"beside it) against the minimum effect of {R.MIN_EFFECT_STUDY_B:g} (Part 1.5)")
    return Outcome(status, numbers=numbers, notes=notes, estimates=[{"contrast": "Dense - Sparse", **ds.row()}])


# ---------------------------------------------------------------------------
# G2
# ---------------------------------------------------------------------------


def _g2_composite(B: StudyB, reading: Reading, comparators: dict[float, str], budgets: list[float]
                  ) -> tuple[Optional[stats.TwoArmEstimate], str, dict[str, Any]]:
    """Part 5.2's outcome for G2: Dense's mean over ``budgets`` against the mean of each budget's comparator.

    Per seed id s: x(s) = mean over b of sr_Dense(s, b); y(s) = mean over b of sr_comparator(b)(s, b), for
    the seeds every comparator has (a seed some comparator lacks is named, not dropped silently). An
    unpaired two-arm estimate at 97.5 percent, like every G2 interval.
    """
    info: dict[str, Any] = {}
    if not budgets:
        return None, "every unseen budget is at the floor", info
    dense, dense_missing = B.per_seed("Dense", _mean_sr(budgets))
    by_budget = {b: B.per_seed(comparators[b], _sr(b)) for b in budgets}
    missing = dense_missing + [m for _, ms in by_budget.values() for m in ms]
    if dense is None or any(v is None for v, _ in by_budget.values()):
        return None, f"incomplete: no value for {missing}", info
    common = set.intersection(*(set(v) for v, _ in by_budget.values()))
    left_out = sorted(set().union(*(set(v) for v, _ in by_budget.values())) - common)
    info["composite_seeds_left_out"] = left_out
    y = {s: statistics.fmean(by_budget[b][0][s] for b in budgets) for s in sorted(common)}
    if len(dense) < 2 or len(y) < 2:
        return None, "fewer than two seeds", info
    est = B.e2(dense, y, "G2|composite|" + ",".join(f"{b:g}" for b in budgets), R.INTERVAL_LEVEL_PRIMARY_STUDY_B,
               record=reading == PROPOSED)
    return est, "", info


def g2_outcome(B: StudyB, reading: Reading) -> Outcome:
    """Box G2: "On at least three of the four unseen budgets, dense minus nearest-single exceeds 5
    percentage points with an interval excluding zero" / falsified: "The nearest single-level agent is
    within 5 percentage points of the dense agent, or above it, on at least two of the four unseen
    budgets." 97.5 percent intervals; decided by the box's per-budget count, a confirmatory test (Q-g2-outcome,
    answered in Table 9.1); floored budgets on violation magnitude (a win: Dense lower with the interval
    excluding zero; a loss: not lower). At 15 and 30 the comparator is the nearest single-level arm with the
    higher arm-mean zero-shot satisfaction at that budget (Part 4.2), and on a floored budget the one with the
    lower arm-mean zero-shot violation magnitude (the floor rule: "a rate at the floor cannot separate arms");
    an exact tie on the selecting outcome goes to the one least favourable to G2 on the outcome compared
    (Part 4.2's "the choice least favourable to G2"). Both candidates' contrasts are reported.

    Q-g2-outcome's other reading decides G2 on Part 5.2's outcome, the mean over the (unfloored)
    budgets of Dense minus each budget's comparator: supported if it exceeds 5 percentage points
    with the 97.5 percent interval excluding zero, falsified if it is at most 5 percentage points.
    Under the decided reading that composite is descriptive, reported beside the count with the
    Holm-adjusted per-budget p-values; neither decides."""
    B = B.for_reading(reading)
    level = R.INTERVAL_LEVEL_PRIMARY_STUDY_B
    if _absent(B, ("Dense", *single_arms())):
        return Outcome(NOT_COMPUTABLE, _absent(B, ("Dense", *single_arms())))
    floor = B.floor(reading, lambda b: ("Dense", *nearest_single_arms(b)))
    undecided = _undecided_floor(floor, R.UNSEEN_BUDGETS)
    if undecided:
        return Outcome(NOT_COMPUTABLE, undecided)
    wins, losses, numbers, notes, estimates, pvalues = 0, 0, {}, [], [], {}
    comparators: dict[float, str] = {}
    won: dict[str, stats.TwoArmEstimate] = {}
    lost: list[tuple[float, str, bool, stats.TwoArmEstimate]] = []  # (budget, comparator, floored, estimate)
    for budget in R.UNSEEN_BUDGETS:
        floored = floor[budget]["floored"]
        get = _vm(budget) if floored else _sr(budget)
        candidates = nearest_single_arms(budget)
        ests = {}
        for single in candidates:
            est, why = B.contrast("Dense", single, get, f"G2|b{budget:g}|{'vm' if floored else 'sr'}|{single}",
                                  level, reading=reading)
            if est is None:
                return Outcome(NOT_COMPUTABLE, f"budget {budget:g}, Dense vs {single}: {why}")
            ests[single] = est
        if len(candidates) > 1:
            # Part 4.2: "for those the comparator is whichever of the two has the higher satisfaction on that
            # budget, the choice least favourable to G2". On a budget the floor rule floors, "a rate at the floor
            # cannot separate arms", so the choice is made on the outcome compared instead (Q-g2-outcome, answered
            # in Table 9.1): the lower arm-mean zero-shot violation magnitude, again the choice least favourable to G2.
            what, select = ("violation magnitude", _vm(budget)) if floored else ("satisfaction", _sr(budget))
            means = {}
            for single in candidates:
                values, missing = B.per_seed(single, select)
                if values is None:
                    return Outcome(NOT_COMPUTABLE, f"budget {budget:g}: the comparator is chosen by {what}, and "
                                                   f"{single} has no zero-shot {what} for {missing}")
                means[single] = statistics.fmean(values.values())
            best = (min if floored else max)(settled(v) for v in means.values())
            tied = [s for s in candidates if settled(means[s]) == best]
            if len(tied) > 1:
                # an exact tie on the selecting outcome: the one least favourable to G2 on the outcome compared
                # (Q-g2-outcome; both are reported): the smaller lower limit of Dense minus it on satisfaction, the
                # larger upper limit of Dense minus it on violation magnitude (where lower is better for Dense)
                comparator = (max(tied, key=lambda s: ests[s].interval(reading.interval)[1]) if floored
                              else min(tied, key=lambda s: ests[s].interval(reading.interval)[0]))
                notes.append(f"budget {budget:g}: the comparators tie on {what}; the one least favourable to G2 on "
                             "the outcome compared is used (Q-g2-outcome)")
            else:
                comparator = tied[0]
            if floored:
                numbers[f"b{budget:g}.comparator_mean_violation"] = means
                notes.append(f"budget {budget:g}: comparator chosen on violation magnitude (floor rule; Q-g2-outcome)")
            else:
                numbers[f"b{budget:g}.comparator_mean_satisfaction"] = means
        else:
            comparator = candidates[0]
        comparators[budget] = comparator
        est = ests[comparator]
        lo, hi = est.interval(reading.interval)
        if floored:
            win, lose = settled(est.diff) < 0 and hi < 0, settled(est.diff) >= 0
            notes.append(f"floor rule applied at budget {budget:g}: Dense compared with {comparator} on violation "
                         "magnitude")
        else:
            win = settled(est.diff) > R.MIN_EFFECT_STUDY_B and lo > 0
            lose = settled(est.diff) <= R.MIN_EFFECT_STUDY_B
        wins += win
        losses += lose
        if win:
            won[f"Dense - {comparator} at {budget:g}"] = est
        if lose:
            lost.append((budget, comparator, bool(floored), est))
        pvalues[budget] = est.welch.p
        numbers[f"b{budget:g}.comparator"] = comparator
        numbers[f"b{budget:g}.candidates"] = list(candidates)
        numbers[f"b{budget:g}.outcome"] = "violation magnitude" if floored else "satisfaction"
        numbers[f"b{budget:g}.win"] = bool(win)
        numbers[f"b{budget:g}.lose"] = bool(lose)
        for single, e in ests.items():
            numbers.update(_numbers(f"b{budget:g}.Dense - {single}", e, reading))
            estimates.append({"contrast": f"Dense - {single} at {budget:g}", **e.row()})
    for budget, h in stats.holm(pvalues, R.ALPHA_PRIMARY_STUDY_B).items():
        numbers[f"b{budget:g}.holm_adjusted_p"] = h.adjusted
    numbers.update({"budgets_won": wins, "budgets_lost": losses})
    on_sr = [b for b in R.UNSEEN_BUDGETS if not floor[b]["floored"]]
    composite, why, info = _g2_composite(B, reading, comparators, on_sr)
    numbers.update(info)
    numbers.update(_numbers("composite: Dense - comparator, mean over budgets", composite, reading))
    if composite is not None:
        estimates.append({"contrast": "composite: Dense - comparator, mean over " + ", ".join(f"{b:g}" for b in on_sr),
                          **composite.row()})
    if reading.g2_outcome == "composite":
        if composite is None:
            return Outcome(NOT_COMPUTABLE, f"the composite (Q-g2-outcome's other reading) is not computable: {why}",
                           numbers, estimates, notes)
        c_lo, _ = composite.interval(reading.interval)
        supported = settled(composite.diff) > R.MIN_EFFECT_STUDY_B and c_lo > 0
        falsified = settled(composite.diff) <= R.MIN_EFFECT_STUDY_B
        decided_by = [("composite", composite)]
        notes.append("decided on the mean over the budgets (Q-g2-outcome's other reading); the per-budget count is "
                     "beside it")
    else:
        supported, falsified = wins >= R.G2_MIN_BUDGETS_SUPPORT, losses >= R.G2_MIN_BUDGETS_FALSIFY
        decided_by = list(won.items())
        notes.append("the composite (per seed, the mean over the unfloored budgets of Dense against each budget's "
                     "comparator; 97.5 percent) is descriptive, and Holm over the four budgets is shown beside the "
                     "count rule; neither decides (Q-g2-outcome)")
    numbers.update({"support": supported, "falsification": falsified})  # the box conditions, before Part 5.7
    status = box_status(supported, falsified, reading, notes)
    bounded, no_bound = None, False
    if status == FALSIFIED:
        if reading.g2_outcome == "composite":
            ann, bounded = bound_annotation("composite, 95% Welch",
                                            stats.relevel(composite.welch, R.INTERVAL_LEVEL).hi, R.MIN_EFFECT_STUDY_B,
                                            claimed="positive")
            numbers.update(ann)
        else:
            count = 0
            for budget, comparator, was_floored, est in lost:
                if was_floored:
                    numbers[f"bound(b{budget:g} Dense - {comparator}, violation magnitude)_reading"] = (
                        "no minimum effect on violation magnitude (Q-floor-rule)")
                    continue
                # Part 5.7's bound is the 95 percent limit (the 97.5 percent one decides G2 and is beside it)
                ann, b = bound_annotation(f"b{budget:g} Dense - {comparator}, 95% Welch",
                                          stats.relevel(est.welch, R.INTERVAL_LEVEL).hi, R.MIN_EFFECT_STUDY_B,
                                          claimed="positive")
                numbers.update(ann)
                count += bool(b)
            bounded = count >= R.G2_MIN_BUDGETS_FALSIFY
            numbers["bounded_losing_budgets"] = count
            floored_losses = sum(was_floored for _, _, was_floored, _ in lost)
            if floored_losses == len(lost):
                # no bound exists on violation magnitude, so calibrate's note on "bound(...) numbers" would not apply
                notes.append("Part 5.7: no bound on violation magnitude (no minimum effect; Q-floor-rule)")
                if reading.calibration == "bound":
                    notes.append("no Part 5.7 bound exists on violation magnitude, so the falsification is not "
                                 "confirmed: inconclusive at this sample size (Q-falsification-calibration)")
                    status = INCONCLUSIVE
                no_bound = True
            elif reading.calibration == "bound" and not bounded and floored_losses:
                notes.append(f"{floored_losses} losing budget(s) on violation magnitude have no Part 5.7 bound (no "
                             "minimum effect; Q-floor-rule), so they cannot count toward the bound "
                             "(Q-falsification-calibration)")
    if not no_bound:
        added: list[str] = []
        status = calibrate(status, bounded, reading, added)
        if any("on violation magnitude have no Part 5.7 bound" in n for n in notes):
            added = [n for n in added if "not below the minimum effect" not in n]  # the floored losses are the reason
        notes += added
    if status == SUPPORTED:
        notes += detectable_notes(dict(decided_by))
    return Outcome(status, numbers=numbers, notes=notes, estimates=estimates)


# ---------------------------------------------------------------------------
# G3
# ---------------------------------------------------------------------------


def _g3_difference(rec: dict[str, Any], on_vm: bool, ref_strict: float, ref_loose: float, strict: float,
                   loose: float) -> Optional[float]:
    """One seed's drop(5) - drop(45), each drop from the arm's nearest training level (Part 4.2).

    On the floor, the drop in satisfaction becomes the rise in violation magnitude (Q-floor-rule).
    """
    zero = rec["vm_zero"] if on_vm else rec["sr_zero"]
    train = rec["vm_train"] if on_vm else rec["sr_train"]
    if zero is None or train is None:
        return None
    values = (train.get(ref_strict), zero.get(strict), train.get(ref_loose), zero.get(loose))
    if None in values:
        return None
    at_ref_strict, at_strict, at_ref_loose, at_loose = values
    if on_vm:
        return (at_strict - at_ref_strict) - (at_loose - at_ref_loose)
    return (at_ref_strict - at_strict) - (at_ref_loose - at_loose)


def _g3_parts(rec: dict[str, Any], on_vm: bool, ref_strict: float, ref_loose: float, strict: float,
              loose: float) -> Optional[list[tuple[float, list[float]]]]:
    """The episodes behind ``_g3_difference`` (Q-iqm), or None without the zero-shot or reference-level (training)
    episodes."""
    zero, train = rec.get("zero_episodes"), rec.get("train_episodes")
    if (not zero or not train or strict not in zero or loose not in zero or ref_strict not in train
            or ref_loose not in train):
        return None
    if on_vm:
        return [(1.0, _violations(zero[strict], strict)), (-1.0, _violations(train[ref_strict], ref_strict)),
                (-1.0, _violations(zero[loose], loose)), (1.0, _violations(train[ref_loose], ref_loose))]
    return [(1.0, _indicators(train[ref_strict], ref_strict)), (-1.0, _indicators(zero[strict], strict)),
            (-1.0, _indicators(train[ref_loose], ref_loose)), (1.0, _indicators(zero[loose], loose))]


def g3_proportion_sentence() -> str:
    """Box G3: "the interpretation of G3 must say so" (the equal distances are unequal in proportion)."""
    lo, hi = R.CONTINUOUS_RANGE
    strict, loose = min(R.UNSEEN_BUDGETS), max(R.UNSEEN_BUDGETS)
    return (f"The two distances are equal in cost units ({lo - strict:g} below {lo:g}; {loose - hi:g} above {hi:g}) "
            f"and unequal in proportion: {(lo - strict) / lo:.3g} of the lowest training budget against "
            f"{(loose - hi) / hi:.3g} of the highest (box G3).")


def g3_outcome(B: StudyB, reading: Reading) -> Outcome:
    """Box G3: "For every arm, the satisfaction drop at the stricter budget (5) exceeds the drop at the
    looser budget (45) by more than 5 percentage points with an interval excluding zero" / falsified:
    "The two drops are within 5 percentage points of each other for the majority of arms." Drops are
    within-seed repeated measures: one-sample 95 percent interval of drop(5) - drop(45) per arm."""
    B = B.for_reading(reading)
    strict, loose = min(R.UNSEEN_BUDGETS), max(R.UNSEEN_BUDGETS)
    arms = G3_ARMS if reading.g3_arms == "four" else tuple(R.STUDY_B_ARMS)
    if _absent(B, arms):
        return Outcome(NOT_COMPUTABLE, _absent(B, arms))
    floor = B.floor(reading, lambda b: arms)
    undecided = _undecided_floor(floor, (strict, loose))
    if undecided:
        return Outcome(NOT_COMPUTABLE, undecided)
    on_vm = bool(floor[strict]["floored"] or floor[loose]["floored"])
    numbers: dict[str, Any] = {"arms": list(arms), "outcome": "violation magnitude" if on_vm else "satisfaction"}
    notes = [g3_proportion_sentence()]
    if on_vm:
        notes.append("floor rule applied: budget 5 or 45 is at the floor, so the drops are rises in violation "
                     "magnitude, compared by direction and interval only (Q-floor-rule)")
    sup, fal = [], []
    bounds: list[tuple[dict[str, Any], Optional[bool]]] = []  # Part 5.7 per arm within the margin
    for arm in arms:
        ref5, ref45 = nearest_training_level(arm, strict), nearest_training_level(arm, loose)
        get = stats.EpisodeOutcome(
            source=f"G3|{'vm' if on_vm else 'sr'}|{ref5:g},{ref45:g}",
            value=lambda r, a=ref5, b=ref45: _g3_difference(r, on_vm, a, b, strict, loose),
            episodes=lambda r, a=ref5, b=ref45: _g3_parts(r, on_vm, a, b, strict, loose))
        values, missing = B.per_seed(arm, get)
        if values is None or len(values) < 2:
            if missing:
                why = (f"no {'violation magnitude' if on_vm else 'satisfaction'} at budgets {strict:g}/{loose:g} or at "
                       f"the reference levels {ref5:g}/{ref45:g} for {missing} (Q-g3-arms)")
            else:
                why = "fewer than two seeds"
            return Outcome(NOT_COMPUTABLE, f"{arm}: {why}", numbers, [], notes)
        if reading == PROPOSED:
            B._note_iqm(arm, None, get, f"G3|{arm}", R.INTERVAL_LEVEL)
        t = stats.one_sample(values, R.INTERVAL_LEVEL)
        boot = stats.bootstrap_mean(values, analysis_id=f"G3|{arm}|{'vm' if on_vm else 'sr'}")
        lo, hi = (t.lo, t.hi) if reading.interval == "welch" else (boot.lo, boot.hi)
        if on_vm:
            sup.append(settled(t.estimate) > 0 and lo > 0)
            fal.append(lo <= 0 <= hi)
        else:
            sup.append(settled(t.estimate) > R.MIN_EFFECT_STUDY_B and lo > 0)
            fal.append(settled(abs(t.estimate)) <= R.MIN_EFFECT_STUDY_B)
        numbers.update({f"{arm}.D": t.estimate, f"{arm}.t_lo": t.lo, f"{arm}.t_hi": t.hi, f"{arm}.boot_lo": boot.lo,
                        f"{arm}.boot_hi": boot.hi, f"{arm}.n": len(values), f"{arm}.reference_levels": (ref5, ref45),
                        f"{arm}.supports": sup[-1], f"{arm}.within_margin": fal[-1]})
        if fal[-1] and not on_vm:
            bounds.append(bound_annotation(f"{arm}.D, 95% t", t.hi, R.MIN_EFFECT_STUDY_B, claimed="positive"))
    supported, falsified = all(sup), sum(fal) > len(arms) / 2
    numbers.update({"support": supported, "falsification": falsified})  # the box conditions, before Part 5.7
    status = box_status(supported, falsified, reading, notes)
    bounded_arms = sum(bool(b) for _, b in bounds)
    if status == FALSIFIED:
        # Part 5.7's bound belongs to a falsified box only (verdict module: "Part 5.7's bound on every falsified
        # box"): an inconclusive G3 carries no "smaller than practical relevance" reading
        for ann, _ in bounds:
            numbers.update(ann)
        numbers["bounded_arms"] = bounded_arms if not on_vm else None
    if not on_vm:
        status = calibrate(status, bounded_arms > len(arms) / 2, reading, notes)
    elif status == FALSIFIED:
        # no bound exists on violation magnitude, so calibrate's note on "bound(...) numbers" would not apply
        notes.append("Part 5.7: no bound on violation magnitude (no minimum effect; Q-floor-rule)")
        if reading.calibration == "bound":
            notes.append("no Part 5.7 bound exists on violation magnitude, so the falsification is not confirmed: "
                         "inconclusive at this sample size (Q-falsification-calibration)")
            status = INCONCLUSIVE
    return Outcome(status, numbers=numbers, notes=notes)


# ---------------------------------------------------------------------------
# G4
# ---------------------------------------------------------------------------


def g4_outcome(B: StudyB, reading: Reading) -> Outcome:
    """Box G4: "Median adaptation steps are ordered dense < moderate < sparse, and dense is at least one
    horizon below sparse on the majority of unseen budgets" / falsified: "Adaptation steps of the
    sparse, moderate and dense arms are within one horizon of one another on the majority of unseen
    budgets." Q-g4-reading (answered in Table 9.1): G4 is read on horizon indices (1 = the first horizon); a
    run that never reaches 0.80 is censored and takes the index after the largest horizon, "above the
    largest horizon" (Table 2.5; 4 with three horizons, 2 after Part 6.1 cut 3). Part 4.2 registers
    "adaptation steps censored at the largest horizon when the target is never reached": that reading, a
    censored value at the largest horizon's own index, is another reading of the key
    (``g4_censored='at'``), decoded from ``entry['censored']`` (a value above the largest horizon of that
    continuation; analysis.data).

    The ordering clause "median adaptation steps are ordered dense < moderate < sparse" is read on one
    statistic per arm, the median over seeds of each seed's median index over the budgets off the
    few-shot floor, strictly ordered (as G1 pairs a pooled ordering with a size clause; read per budget,
    the second clause would add nothing). "Dense is at least one horizon below sparse"
    holds on a budget when Sparse's median index over seeds minus Dense's (a difference of medians) is
    at least 1, and must hold on more than half of those budgets. "Within one horizon of one another"
    holds on a budget when the largest minus the smallest of the three per-budget medians is below 1,
    the exact negation of "at least one horizon"; G4 is falsified when it holds on more than half of
    those budgets. Three sensitivity readings are reported beside the verdict (``SENSITIVITY_KEYS``; the
    key's other readings): 'within' as a spread of at most 1 (``g4_within='at_most_one'``), the ordering on
    a majority of per-budget medians (``g4_order='per_budget'``), and the censored index at the largest
    horizon (``g4_censored='at'``).

    Floor rule (Part 4.2, "for every comparison above"; Q-floor-rule, answered in Table 9.1;
    ``StudyB.fewshot_floor``): a budget at which the few-shot satisfaction is at the floor at every horizon cannot separate the
    arms' adaptation steps (all censored). It leaves G4's counts (and the arm statistic), the majority is
    taken over the remaining budgets, and Dense and Sparse are compared there on the few-shot violation
    magnitude at the largest horizon, by direction and interval only; the verdict says so. Every budget
    floored: NOT_COMPUTABLE; the floor undecided at a budget: NOT_COMPUTABLE, naming the unknown rates.

    One horizon (Part 6.1 cut 3, ``fewshot_short``): an index is 1 or censored (2), so the readings
    ``g4_censored='at'`` (every index 1) and ``g4_within='at_most_one'`` (every spread at most one) decide
    G4 by construction; they give NOT_COMPUTABLE in the spirit of the floor rule. The arm statistic can
    still be ordered (1 < 1.5 < 2); under the per-budget ordering (``g4_order='per_budget'``) the verdict
    notes that, with an odd seed count, the ordering clause cannot hold."""
    if reading.g4_within not in ("less_than_one", "at_most_one"):  # a stale value must not select a reading silently
        raise ValueError(f"unknown Q-g4-reading 'within one horizon' reading {reading.g4_within!r}")
    B = B.for_reading(reading)
    absent = _absent(B, G4_ARMS)
    if absent:
        return Outcome(NOT_COMPUTABLE, absent)
    numbers: dict[str, Any] = {}
    ordered_count = below_count = within_count = 0
    horizon_sets: set[tuple[int, ...]] = set()
    by_seed: dict[str, dict[int, list[float]]] = {arm: {} for arm in G4_ARMS}
    medians: dict[float, dict[str, float]] = {}
    seed_indices: dict[float, dict[str, dict[int, float]]] = {}
    for budget in R.UNSEEN_BUDGETS:
        med = {}
        seed_indices[budget] = {}
        for arm in G4_ARMS:
            indices = {}
            for rec in B.arm(arm):
                entry = rec["fewshot"].get(budget)
                if entry is None or entry["index"] is None:
                    return Outcome(NOT_COMPUTABLE, f"{rec['run_id']}: no adaptation steps at budget {budget:g}")
                if entry["consistent"] is False:
                    return Outcome(NOT_COMPUTABLE, f"{rec['run_id']}: adapt_steps at budget {budget:g} does not "
                                                   "follow from sr_fewshot (see the data problems)")
                horizon_sets.add(tuple(entry["horizons"]))
                index = entry["index"]
                if entry["censored"] and reading.g4_censored == "at":
                    index = len(entry["horizons"])  # Part 4.2: "censored at the largest horizon"
                indices[rec["seed"]] = index
            assert indices, f"{arm}: _absent above returns for an arm with no completed run"
            med[arm] = statistics.median(indices.values())
            seed_indices[budget][arm] = indices
        medians[budget] = med
    if len(horizon_sets) > 1:
        return Outcome(NOT_COMPUTABLE, f"the continuations have different horizons {sorted(horizon_sets)}; "
                                       "their indices are not comparable", {"horizons": sorted(horizon_sets)})
    one_horizon = bool(horizon_sets) and len(next(iter(horizon_sets))) == 1
    if one_horizon and (reading.g4_censored == "at" or reading.g4_within == "at_most_one"):
        # Part 6.1 cut 3 (one horizon): every index is 1 (reached) or censored; under these readings G4 is
        # decided by construction (every index 1 when censored at the largest horizon; a spread of at most
        # one always 'within one horizon'), so, as with Part 4.2's floor rule, it cannot separate the arms
        return Outcome(NOT_COMPUTABLE, "one horizon (Part 6.1 cut 3): under this reading of Q-g4-reading adaptation "
                                       "steps cannot separate the arms (the verdict would follow from the design, "
                                       "not the data)", {"horizons": sorted(horizon_sets)[0]})
    floor = B.fewshot_floor(reading, G4_ARMS)
    undecided = _undecided_floor(floor, R.UNSEEN_BUDGETS, "few-shot floor rule")
    if undecided:
        return Outcome(NOT_COMPUTABLE, undecided)
    budgets = [b for b in R.UNSEEN_BUDGETS if not floor[b]["floored"]]
    floored = [b for b in R.UNSEEN_BUDGETS if floor[b]["floored"]]
    vm_numbers, vm_notes = _vm_comparisons(B, reading, floored, "Dense", "Sparse", R.INTERVAL_LEVEL, "G4",
                                           _fewshot_vm, "few-shot violation magnitude at the largest horizon")
    if not budgets:
        return Outcome(NOT_COMPUTABLE, "every unseen budget is at the few-shot floor: adaptation steps cannot separate "
                                       "the arms (the violation-magnitude comparisons are reported)",
                       {**vm_numbers, "budgets_floored": floored}, [], vm_notes)
    for budget in budgets:
        med = medians[budget]
        for arm in G4_ARMS:
            for seed, index in seed_indices[budget][arm].items():
                by_seed[arm].setdefault(seed, []).append(index)
        ordered = med["Dense"] < med["Moderate"] < med["Sparse"]
        below = med["Sparse"] - med["Dense"] >= 1
        spread = max(med.values()) - min(med.values())
        within = spread < 1 if reading.g4_within == "less_than_one" else spread <= 1
        ordered_count += ordered
        below_count += below
        within_count += within
        numbers.update({f"b{budget:g}.median_index({a})": v for a, v in med.items()})
        numbers.update({f"b{budget:g}.ordered": ordered, f"b{budget:g}.dense_one_below_sparse": below,
                        f"b{budget:g}.within_one_horizon": within})
    # the arm statistic: the median over seeds of each seed's median over the budgets off the few-shot floor
    arm_median = {arm: statistics.median(statistics.median(v) for v in seeds.values())
                  for arm, seeds in by_seed.items()}
    arm_ordered = arm_median["Dense"] < arm_median["Moderate"] < arm_median["Sparse"]
    majority = len(budgets) / 2  # of the budgets off the few-shot floor
    ordering = arm_ordered if reading.g4_order == "arm_median" else ordered_count > majority
    notes: list[str] = []
    status = box_status(ordering and below_count > majority, within_count > majority, reading, notes)
    numbers.update({"budgets_ordered": ordered_count, "budgets_dense_one_below": below_count,
                    "budgets_within": within_count, "horizons": sorted(horizon_sets)[0] if horizon_sets else None,
                    **{f"arm_median_index({a})": v for a, v in arm_median.items()}, "arm_medians_ordered": arm_ordered,
                    "ordering_read": ("arm statistic (median over seeds of each seed's median over the budgets)"
                                      if reading.g4_order == "arm_median" else "per budget, majority"),
                    "within_read": ("spread of the per-budget medians below 1" if reading.g4_within == "less_than_one"
                                    else "spread of the per-budget medians at most 1"),
                    "censored_index": ("the largest horizon's" if reading.g4_censored == "at"
                                       else "after the largest horizon")})
    numbers.update({"budgets_on_adaptation_steps": budgets, "budgets_floored": floored, **vm_numbers})
    notes.append(f"adaptation steps are read on the {len(budgets)} budgets off the few-shot floor, the majority taken "
                 "over them; the few-shot floor per (budget, horizon) is in B_fewshot")
    notes += vm_notes
    if one_horizon and reading.g4_order == "per_budget":
        notes.append("one horizon (Part 6.1 cut 3): each seed's adaptation-steps index is binary (1 reached, 2 "
                     "censored), so a median over an odd number of seeds is 1 or 2 and the strict ordering Dense < "
                     "Moderate < Sparse cannot hold: with an odd seed count per arm G4 can be falsified or "
                     "inconclusive but not supported, by the design rather than the data")
    if status == FALSIFIED:
        notes.append("Part 5.7: no interval bound for G4 (medians of horizon indices); Q-falsification-calibration "
                     "does not apply")
    return Outcome(status, numbers=numbers, notes=notes)


# ---------------------------------------------------------------------------
# G5
# ---------------------------------------------------------------------------


def g5_outcome(B: StudyB, reading: Reading) -> Outcome:
    """Box G5: "Continuous minus Dense on the mean satisfaction over the four unseen budgets has a point
    estimate of at most 5 percentage points and an interval whose upper limit is at most 10" /
    falsified: "The lower limit of the interval for Continuous minus Dense exceeds 5 percentage points".
    G5 is falsified by an effect, not by its absence: its lower limit is itself the Part 5.7 annotation."""
    B = B.for_reading(reading)
    if _absent(B, ("Continuous", "Dense")):
        return Outcome(NOT_COMPUTABLE, _absent(B, ("Continuous", "Dense")))
    floor = B.floor(reading, lambda b: ("Continuous", "Dense"))
    undecided = _undecided_floor(floor, R.UNSEEN_BUDGETS)
    if undecided:
        return Outcome(NOT_COMPUTABLE, undecided)
    budgets = [b for b in R.UNSEEN_BUDGETS if not floor[b]["floored"]]
    floored = [b for b in R.UNSEEN_BUDGETS if floor[b]["floored"]]
    if not budgets:
        vm_numbers, vm_notes = _vm_comparisons(B, reading, floored, "Continuous", "Dense", R.INTERVAL_LEVEL, "G5")
        return Outcome(NOT_COMPUTABLE, "every unseen budget is at the floor: satisfaction cannot separate the arms "
                                       "(the violation-magnitude comparisons are reported)",
                       {**vm_numbers, "budgets_floored": floored}, [], vm_notes)
    est, why = B.contrast("Continuous", "Dense", _mean_sr(budgets), "G5|Continuous-Dense", R.INTERVAL_LEVEL,
                          reading=reading)
    if est is None:
        return Outcome(NOT_COMPUTABLE, why)
    lo, hi = est.interval(reading.interval)
    supported = settled(est.diff) <= R.G5_POINT_MAX and settled(hi) <= R.G5_UPPER_MAX
    falsified = settled(lo) > R.MIN_EFFECT_STUDY_B  # "exceeds 5 percentage points" (the minimum effect of Part 1.5)
    notes: list[str] = []
    status = box_status(supported, falsified, reading, notes)
    numbers = {"budgets_on_satisfaction": budgets, "budgets_floored": floored, "support": supported,
               "falsification": falsified}
    numbers.update(_numbers("Continuous - Dense", est, reading))
    if status == FALSIFIED:
        numbers.update({"bound(Continuous - Dense)": lo, "bound(Continuous - Dense)_limit": "lower",
                        "bound(Continuous - Dense)_minimum_effect": R.MIN_EFFECT_STUDY_B,
                        "bound(Continuous - Dense)_reading": ("the lower limit exceeds the minimum effect: the "
                                                              "difference is at least of practical relevance")})
    vm_numbers, vm_notes = _vm_comparisons(B, reading, floored, "Continuous", "Dense", R.INTERVAL_LEVEL, "G5")
    numbers.update(vm_numbers)
    return Outcome(status, numbers=numbers, notes=notes + vm_notes,
                   estimates=[{"contrast": "Continuous - Dense", **est.row()}])


# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------


@dataclass
class Catalogue:
    """Study B's verdicts and tables."""

    verdicts: list[Verdict] = field(default_factory=list)
    tables: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


def _decide(**kwargs: Any) -> Verdict:
    """``verdict.decide`` with the sensitivity readings of its answered keys beside the verdict (``SENSITIVITY_KEYS``).

    Each is one of the key's ``READINGS`` that its answer names (Q-floor-rule's joint reading, which it does not
    name, is left out); its status is reported as the number ``sensitivity(key: reading)`` (with its reason when it
    has one) and never decides: the verdict's status is the decided reading's. While a key is open, ``decide``
    computes its other readings instead."""
    verdict = decide(**kwargs)
    keys = [k for k in SENSITIVITY_KEYS if k in questions.keys_for(verdict.table_key) and not R.is_open(k)]
    for key in keys:
        for overrides in READINGS[key]:
            if len(overrides) > 1:  # Q-floor-rule's joint reading ('comparison' and 'per-seed' together)
                continue
            out = kwargs["compute"](replace(PROPOSED, **overrides))
            name = f"sensitivity({key}: {', '.join(f'{k}={v}' for k, v in overrides.items())})"
            verdict.numbers[name] = out.status
            if out.reason:
                verdict.numbers[f"{name}_reason"] = out.reason
    if keys:
        verdict.notes.append(f"the sensitivity(...) numbers are the statuses under the readings that the answers of "
                             f"{', '.join(keys)} report beside the verdict as sensitivity readings; they never "
                             "decide it")
    return verdict


def _budget_claims(B: StudyB, outcome: str) -> list[Verdict]:
    """Satisfaction per budget (table key B-per-budget) or violation magnitude (B-violation): Holm over
    the four budgets (Part 5.2).

    A floored budget's satisfaction claim is made on violation magnitude (Part 4.2 floor rule); the
    cells are rebuilt under each reading of Q-floor-rule (under 'comparison', the two arms of the
    contrast count). Where the floor rule is undecided (an arm's rate unknown), the satisfaction claim
    at that budget is NOT_COMPUTABLE. The four budgets are each claim's only Holm family, so Holm applies
    under every reading of Q-holm-families ("Families shrink only to cells not tested"): a budget not tested
    (an arm with fewer than two seeds) leaves the family, but a budget whose data are still to come (a
    value missing, the floor rule undecided) does not, and the decisions it could change wait on it
    (``stats.holm_decisions``).
    """
    cache: dict[tuple, dict] = {}

    def cells_for(reading: Reading, x_arm: str, y_arm: str) -> dict:
        key = (reading.floor_arms, reading.floor_rate, reading.seeds, x_arm, y_arm)
        if key not in cache:
            B_r = B.for_reading(reading)
            floor = B_r.floor(reading, lambda b: (x_arm, y_arm))
            cells = {}
            incomplete: dict[float, str] = {}
            for budget in R.UNSEEN_BUDGETS:
                if _absent(B_r, (x_arm, y_arm)):
                    cells[budget] = ((None, _absent(B_r, (x_arm, y_arm))), False)
                    continue
                if outcome == "satisfaction" and floor[budget]["floored"] is None:
                    cells[budget] = ((None, _undecided_floor(floor, [budget])), False)
                    incomplete[budget] = cells[budget][0][1]
                    continue
                on_vm = outcome == "violation" or bool(floor[budget]["floored"])
                cells[budget] = (B_r.contrast(x_arm, y_arm, _vm(budget) if on_vm else _sr(budget),
                                              f"B-{outcome}|{x_arm}-{y_arm}|b{budget:g}|{'vm' if on_vm else 'sr'}",
                                              reading=reading), on_vm)
            members: dict[float, Any] = {}
            for b, ((est, why), _) in cells.items():
                if est is not None:
                    members[b] = est.welch.p
                elif b in incomplete or why.startswith("incomplete"):
                    members[b] = stats.Incomplete(f"budget {b:g}: {incomplete.get(b) or why}")
                else:
                    members[b] = None
            adjusted, survives, waiting = stats.holm_decisions(members, R.ALPHA)
            cache[key] = {"cells": cells, "adjusted": adjusted, "survives": survives,
                          "waiting": "; ".join(waiting.values())}
        return cache[key]

    verdicts = []
    name = "satisfaction" if outcome == "satisfaction" else "violation magnitude"
    for x_arm, y_arm in PER_BUDGET_CONTRASTS:
        for budget in R.UNSEEN_BUDGETS:
            def compute(reading: Reading, x_arm=x_arm, y_arm=y_arm, budget=budget) -> Outcome:
                entry = cells_for(reading, x_arm, y_arm)
                (est, why), on_vm = entry["cells"][budget]
                if est is None:
                    return Outcome(NOT_COMPUTABLE, why)
                adjusted = entry["adjusted"][budget]
                lo, hi = est.interval(reading.interval)
                # the four budgets are the claim's only family (Part 5.2): Holm under every reading of Q-holm-families
                holm_ok = entry["survives"][budget]
                notes = []
                if on_vm and outcome == "satisfaction":
                    notes.append(f"floor rule applied at budget {budget:g}: compared on violation magnitude")
                excludes = lo > 0 or hi < 0  # an interval including zero is INCONCLUSIVE whatever Holm gives
                status = claim_status(excludes, holm_ok if excludes else True)
                if status == SUPPORTED:
                    notes += detectable_notes({"the difference": est})
                reason = ""
                if status == NOT_COMPUTABLE:
                    reason = (f"incomplete: Holm over the four budgets cannot be decided while a budget is "
                              f"incomplete ({entry['waiting']}); the family shrinks only to the budgets not tested "
                              "(Q-holm-families)")
                return Outcome(status, reason, notes=notes, estimates=[est.row()], numbers={
                    "diff": est.diff, "welch_lo": est.welch.lo, "welch_hi": est.welch.hi, "boot_lo": est.bootstrap.lo,
                    "boot_hi": est.bootstrap.hi, "p": est.welch.p, "holm_adjusted_p": adjusted.adjusted,
                    "holm_family_size": adjusted.family_size, "holm_survives": holm_ok,
                    "outcome": "violation magnitude" if on_vm else "satisfaction",
                    "cohens_d": est.cohens_d, "detectable_d": est.detectable_d,
                    "below_detectable": est.below_detectable})
            verdicts.append(_decide(
                id=f"B-{outcome}[{x_arm}-{y_arm}/b{budget:g}]", title=f"{name.capitalize()} per budget (secondary)",
                statement=(f"{x_arm} minus {y_arm} in zero-shot {name} at budget {budget:g} is not zero (Holm over "
                           "the four budgets)"),
                label=SECONDARY, table_key="B-per-budget" if outcome == "satisfaction" else "B-violation",
                compute=compute, group=f"Study B: {name} per budget"))
    return verdicts


def _tables(B: StudyB) -> dict[str, list[dict[str, Any]]]:
    """Tables B_outcomes, B_floor and B_fewshot (the floor rule as Q-floor-rule decides it)."""
    tables: dict[str, list[dict[str, Any]]] = {}
    rows = []
    for rec in B.records:
        for budget in R.UNSEEN_BUDGETS:
            nearest = nearest_training_level(rec["arm"], budget)
            rows.append({
                "run_id": rec["run_id"], "arm": rec["arm"], "seed": rec["seed"], "budget": budget,
                "sr_zero": (rec["sr_zero"] or {}).get(budget), "vm_zero": (rec["vm_zero"] or {}).get(budget),
                "cost_zero": (rec["cost_zero"] or {}).get(budget), "distance_eq11": distance(budget, rec["arm"]),
                "extrapolation": budget < min(R.CONTINUOUS_RANGE) or budget > max(R.CONTINUOUS_RANGE),
                "nearest_training_level": nearest, "sr_at_nearest_level": (rec["sr_train"] or {}).get(nearest),
            })
    tables["B_outcomes"] = rows
    tables["B_floor"] = [{"budget": b, **{k: (v if not isinstance(v, (list, dict)) else str(v))
                                          for k, v in info.items()}}
                         for b, info in B.floor(PROPOSED).items()]
    few = []
    for arm in R.STUDY_B_ARMS:
        recs = B.arm(arm)
        for budget in R.UNSEEN_BUDGETS:
            by_h: dict[int, list[float]] = {}
            vm_h: dict[int, list[float]] = {}
            for rec in recs:
                entry = rec["fewshot"].get(budget) or {}
                for h, v in (entry.get("sr") or {}).items():
                    by_h.setdefault(h, []).append(v)
                for h, v in (entry.get("vm") or {}).items():
                    vm_h.setdefault(h, []).append(v)
            for h in sorted(by_h):
                # n counts the runs with a few-shot rate; n_vm those with a supplement record's magnitude
                few.append({"arm": arm, "budget": budget, "horizon": h, "n": len(by_h[h]),
                            "mean_sr": statistics.fmean(by_h[h]), "n_vm": len(vm_h.get(h, [])),
                            "mean_vm": statistics.fmean(vm_h[h]) if vm_h.get(h) else None})
    # Q-floor-rule (answered in Table 9.1), as G4 applies it (StudyB.fewshot_floor): per (budget, horizon) over all
    # seven arms, an arm without a few-shot rate counted as unknown, never dropped; True, False or None (undecided).
    few_floor = B.fewshot_floor(PROPOSED, G4_ARMS)
    for row in few:
        row["floored"] = (few_floor[row["budget"]]["by_horizon"].get(row["horizon"]) or {}).get("floored")
    tables["B_fewshot"] = few
    return tables


def analyse(dataset: Dataset, *, iqm: bool = True) -> Catalogue:
    """Every Study B verdict and table (``iqm=False`` leaves table B_iqm empty: unit tests of the verdicts)."""
    B = StudyB(dataset)
    cat = Catalogue()
    group = "Study B: hypotheses"
    cat.verdicts.append(_decide(
        id="G1", title="G1: more training levels, higher zero-shot satisfaction", label=CONFIRMATORY, table_key="G1",
        statement="mean satisfaction over the unseen budgets ordered Sparse < Moderate < Dense, and Dense minus Sparse "
                  "exceeds 5 percentage points with a 97.5 percent interval excluding zero (primary analysis)",
        compute=lambda r: g1_outcome(B, r), group="Study B: primary"))
    cat.verdicts.append(_decide(
        id="G2", title="G2: Dense beats the nearest single-level arm", label=CONFIRMATORY, table_key="G2",
        statement="on at least three of the four unseen budgets Dense minus the nearest single-level arm exceeds 5 "
                  "percentage points with a 97.5 percent interval excluding zero (primary analysis)",
        compute=lambda r: g2_outcome(B, r), group="Study B: primary"))
    cat.verdicts.append(_decide(
        id="G3", title="G3: stricter extrapolation degrades satisfaction more", label=SECONDARY, table_key="G3",
        statement="for every arm the drop at budget 5 exceeds the drop at budget 45 by more than 5 percentage points "
                  "with an interval excluding zero", compute=lambda r: g3_outcome(B, r), group=group))
    cat.verdicts.append(_decide(
        id="G4", title="G4: few-shot adaptation steps decrease with coverage", label=SECONDARY, table_key="G4",
        statement="median adaptation steps (the arm statistic) ordered Dense < Moderate < Sparse, and Dense at least "
                  "one horizon below Sparse on the majority of unseen budgets", compute=lambda r: g4_outcome(B, r),
        group=group))
    cat.verdicts.append(_decide(
        id="G5", title="G5: Dense comes within 5 points of Continuous", label=SECONDARY, table_key="G5",
        statement="Continuous minus Dense: point estimate at most 5 percentage points and upper limit at most 10",
        compute=lambda r: g5_outcome(B, r), group=group))
    cat.verdicts += _budget_claims(B, "satisfaction")
    cat.verdicts += _budget_claims(B, "violation")
    cat.tables.update(_tables(B))
    cat.tables["B_estimates"] = [{"key": k, **v} for k, v in sorted(B.estimate_rows.items())]
    cat.tables["B_iqm"] = iqm_rows(B.iqm_specs, B.iqm_uses, B.arm, lambda arm: arm) if iqm else []
    return cat

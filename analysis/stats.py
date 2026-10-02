"""Statistics of Part 5: estimands, intervals, rank trends, multiplicity and power.

Owner: Role 4, Analysis and results (Muhammad Abdullah). Implements:

* Part 5.1: "The unit of analysis is the seed in both studies, and every interval is computed by
  treating the seed as the only independent replicate." Every function takes one number per seed
  ({seed: value}); episodes enter only the descriptive interquartile mean.
* Part 5.3: "Each estimand is reported as the difference of seed means, as Cohen's d with the
  pooled seed standard deviation (equation 12; Cohen, 1988), and with two 95 percent intervals: a
  two-sample t-interval with Welch's degrees of freedom (Welch, 1947), which is the primary interval
  ...; and a percentile bootstrap interval (Efron and Tibshirani, 1993) from 10,000 resamples of
  seed indices with replacement ... The resampling unit is the seed index: a resample takes every
  outcome of the chosen seeds together". ``welch``, ``cohens_d``, ``bootstrap_difference``;
  ``estimate_two_arms`` assembles them with the paired analysis of Part 5.1 ("paired and unpaired
  analyses are both reported, the unpaired as primary").
* Part 5.3 (penultimate sentence): the interquartile mean "with a stratified bootstrap over seeds and
  evaluation episodes" (Agarwal et al., 2021, as the pre-registration cites it): ``iqm``,
  ``stratified_iqm``, ``iqm_contrast`` (descriptive; Q-iqm, answered in Table 9.1).
* Part 5.4: Spearman's rank correlation "with an interval from resampling seed indices, each
  resample carrying all of a seed's factor levels together": ``spearman``, ``spearman_bootstrap``.
* Part 5.2: "corrected for multiplicity within its family ... by the Holm procedure at alpha =
  0.05": ``holm``.
* Part 5.5: "a two-sample t-test at alpha = 0.05 (two-sided) has 80 percent power at a standardised
  difference of 2.02, 55 percent at 1.5 and 29 percent at 1.0; with eight seeds the 80-percent point
  falls to 1.51, with ten to 1.33, with twelve to 1.20 (all from the noncentral t distribution)":
  ``power_two_sample``, ``detectable_d``. The tests reproduce R.POWER_REGISTERED_N5 to the printed
  precision (0.798, 0.549, 0.286 round to 0.80, 0.55, 0.29) and R.POWER_REGISTERED_D80 within 0.01
  only: the 80-percent points at five, eight and twelve seeds (2.0244, 1.5066, 1.1968) round to
  nearest as printed, but the one at ten seeds is 1.3249, which rounds to nearest to 1.32; the
  registered 1.33 is that value rounded up (power at 1.33 is 0.803). No single rounding rule gives
  all four printed values. ``POWER_PRINTED_DECIMALS`` and ``report.power_rows`` say which value
  rounds how (reported to the integrator for HANDOVER / the paper's power statement of Part 5.5).
* Equation (12): "U80 = x + 1.886 s / sqrt(3)", the pilot's bound with three seeds, in its paired form
  (Q-g3-pairing, answered in Table 9.1: s is the SD of the per-seed differences, the pairs by seed id
  and, where a replacement seed leaves fewer than three, the unpaired runs in increasing seed order):
  ``g3_pairs``, ``u80``; ``u80_pooled`` (descriptive only), ``u80_t_quantile``; as
  pilot/go_decision.py computes them.

Comparisons with a margin, a threshold or another estimate (``settled``): the outcomes are means of
integer episode costs (a 100-episode mean is a multiple of 0.01) and satisfaction rates k/100
(equation 10), so a registered margin such as 5 percentage points can be met exactly, and plain
floating point then decides it by noise (0.55 - 0.50 is 0.050000000000000044 > 0.05). Such
comparisons are made on values rounded to ``COMPARISON_DECIMALS`` (an implementation choice that no
PENDING key decides; the answer to Q-h4-reading compares "after settled rounding"). Rounding to 1e-9
removes floating-point noise (about 1e-15 here) and cannot move a real boundary: with at most twelve
seeds per arm and four budgets, two distinct
exact values lie at least 1 / (400 x 12 x 12), about 1.7e-5, apart. The 1e-4 of
Q-threshold-arithmetic (rules 3 and 5) would be too coarse here: with seven and eight seeds a mean
over four budgets can exceed 0.05 by 4.5e-5. Whether an interval excludes zero is still read on the
unrounded limits (``interval_excludes_zero``).

Analysis questions as answered in Table 9.1: Q-interval (the Welch interval decides; callers read the
percentile interval as the other reading while the key is open), Q-bootstrap-details (``BOOTSTRAP_SEED``,
generator seeded from the analysis id, linear quantiles, joint resampling of seed indices when two arms
share their seed list and per arm otherwise, undefined resamples dropped and counted), Q-iqm,
Q-holm-families ("Families shrink only to cells not tested": ``holm`` leaves out a member without a
p-value; "an incomplete member is held at p = 1": ``holm_decisions``).
"""

from __future__ import annotations

import functools
import math
import statistics
import zlib
from dataclasses import asdict, dataclass
from typing import Any, Callable, Hashable, Mapping, Optional, Sequence, Union

import numpy as np
from scipy import optimize
from scipy import stats as st

from configs import registered as R

BOOTSTRAP_SEED = 0  # Q-bootstrap-details (answered in Table 9.1): first entropy word of every bootstrap generator
QUANTILE_METHOD = "linear"  # Q-bootstrap-details: numpy's default percentile rule
NAN_FLAG_SHARE = 0.01  # Q-bootstrap-details: flag an interval if over 1 percent of resamples are undefined
U80_COVERAGE = 0.80  # equation (12): U80 is the upper limit of the two-sided 80 percent t-interval
IQM_CHUNK = 500  # Q-bootstrap-details: resamples per block of the two-level bootstrap (bounds memory;
# the block size fixes the order of the random draws, so it is part of the recorded bootstrap)
COMPARISON_DECIMALS = 9  # an implementation choice (no PENDING key; see the module docstring): round before comparing
POWER_PRINTED_DECIMALS = 2  # Part 5.5 prints power and the 80-percent points to two decimals

SeedValues = Mapping[int, float]


class StatsError(ValueError):
    """The data cannot give the statistic (too few seeds, no common seeds, a missing value)."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resamples(resamples: Optional[int]) -> int:
    """Part 5.3: "10,000 resamples" (R.BOOTSTRAP_RESAMPLES), read when called; an explicit count overrides it."""
    return R.BOOTSTRAP_RESAMPLES if resamples is None else int(resamples)


def generator_seed(analysis_id: str) -> list[int]:
    """The recorded seed of an analysis's bootstrap generator: [BOOTSTRAP_SEED, crc32(analysis id)]."""
    return [BOOTSTRAP_SEED, zlib.crc32(analysis_id.encode("utf-8"))]


def generator(analysis_id: str) -> np.random.Generator:
    """A generator of its own for each analysis id, so a result never depends on the order of analyses."""
    return np.random.default_rng(generator_seed(analysis_id))


def _check(values: SeedValues, name: str, minimum: int = 1) -> tuple[list[int], np.ndarray]:
    seeds = sorted(values)
    data = [values[s] for s in seeds]
    if any(v is None for v in data):
        raise StatsError(f"{name}: a seed has no value")
    array = np.asarray(data, dtype=float)
    if not np.all(np.isfinite(array)):
        raise StatsError(f"{name}: a value is not finite")
    if len(seeds) < minimum:
        raise StatsError(f"{name}: {len(seeds)} seeds; at least {minimum} needed")
    # Settled before any statistic: seeds that differ only by floating-point noise (0.1 + 0.2 and 0.3) are
    # equal, so a noise-only spread is zero variance, never a Welch, paired or bootstrap interval off zero.
    return seeds, np.asarray([settled(v) for v in array], dtype=float)


def _mean(array: np.ndarray) -> float:
    return statistics.fmean(array.tolist())


def settled(value: float) -> float:
    """``value`` as compared with a margin, a threshold or another estimate: rounded to COMPARISON_DECIMALS.

    Every comparison of a point estimate or a limit with the minimum effect, the G5 limits, the floor
    rate, zero (for a point estimate) or another point estimate goes through this, so that a
    difference of exactly 5 percentage points is not read as exceeding 5 (module docstring).
    """
    return round(float(value), COMPARISON_DECIMALS)


def interval_excludes_zero(lo: Optional[float], hi: Optional[float]) -> Optional[bool]:
    """Box language: "an interval excluding zero" (unrounded limits)."""
    if lo is None or hi is None or math.isnan(lo) or math.isnan(hi):
        return None
    return lo > 0 or hi < 0


def percentile_interval(samples: np.ndarray, level: float) -> tuple[float, float, int]:
    """(lo, hi, number of undefined resamples) of the percentile interval at ``level``."""
    finite = samples[np.isfinite(samples)]
    n_nan = int(samples.size - finite.size)
    if finite.size == 0:
        return math.nan, math.nan, n_nan
    lo, hi = np.quantile(finite, [(1 - level) / 2, (1 + level) / 2], method=QUANTILE_METHOD)
    return float(lo), float(hi), n_nan


# ---------------------------------------------------------------------------
# Part 5.3: difference of seed means, Cohen's d, Welch
# ---------------------------------------------------------------------------


def mean_difference(x: SeedValues, y: SeedValues) -> float:
    """The estimand: difference of seed means, mean(x) - mean(y) (Part 5.3)."""
    _, xa = _check(x, "x")
    _, ya = _check(y, "y")
    return _mean(xa) - _mean(ya)


def pooled_sd(x: SeedValues, y: SeedValues) -> Optional[float]:
    """Cohen's pooled SD, sqrt(((n1-1)s1^2 + (n2-1)s2^2) / (n1+n2-2)) (INFERENCE: eq. 12 writes s_p only)."""
    _, xa = _check(x, "x")
    _, ya = _check(y, "y")
    dof = xa.size + ya.size - 2
    if dof <= 0:
        return None
    sx = statistics.variance(xa.tolist()) if xa.size > 1 else 0.0
    sy = statistics.variance(ya.tolist()) if ya.size > 1 else 0.0
    return math.sqrt(((xa.size - 1) * sx + (ya.size - 1) * sy) / dof)


def cohens_d(x: SeedValues, y: SeedValues) -> Optional[float]:
    """Equation (12): d = (mean1 - mean2) / s_p; None when the pooled SD is 0 or undefined."""
    sp = pooled_sd(x, y)
    if sp is None or sp == 0:
        return None
    return mean_difference(x, y) / sp


@dataclass(frozen=True)
class TInterval:
    """A t-interval: the estimate, its limits at ``level``, df, t statistic, two-sided p and SE, and whether
    both arms had zero variance (the interval is then the point)."""

    estimate: float
    lo: float
    hi: float
    level: float
    df: float
    t: float
    p: float
    se: float
    zero_variance: bool = False


def _t_interval(estimate: float, se: float, df: float, level: float) -> TInterval:
    if se == 0:
        # Every seed has the same value in both arms: the interval degenerates to the point. Two arms constant at
        # the same value can differ by a rounding residue of the means (fmean([0.7] * 6) != 0.7), so zero is
        # decided on the settled estimate: then p = 1 and the interval is [0, 0].
        if settled(estimate) == 0:
            return TInterval(0.0, 0.0, 0.0, level, math.nan, math.nan, 1.0, 0.0, True)
        return TInterval(estimate, estimate, estimate, level, math.nan, math.nan, 0.0, 0.0, True)
    quantile = float(st.t.ppf((1 + level) / 2, df))
    t = estimate / se
    return TInterval(estimate, estimate - quantile * se, estimate + quantile * se, level, df, t,
                     float(2 * st.t.sf(abs(t), df)), se)


def welch(x: SeedValues, y: SeedValues, level: float = R.INTERVAL_LEVEL) -> TInterval:
    """Welch's two-sample t-interval for mean(x) - mean(y) (the primary interval of Part 5.3)."""
    _, xa = _check(x, "x", 2)
    _, ya = _check(y, "y", 2)
    vx, vy = statistics.variance(xa.tolist()) / xa.size, statistics.variance(ya.tolist()) / ya.size
    se2 = vx + vy
    diff = _mean(xa) - _mean(ya)
    if se2 == 0:
        return _t_interval(diff, 0.0, math.nan, level)
    df = se2 ** 2 / (vx ** 2 / (xa.size - 1) + vy ** 2 / (ya.size - 1))
    return _t_interval(diff, math.sqrt(se2), df, level)


def relevel(interval: TInterval, level: float) -> TInterval:
    """The same t-interval (estimate, SE, df) at another level, e.g. the 95 percent bound of Part 5.7
    (Q-falsification-calibration) of a 97.5 percent G1 or G2 estimate."""
    if interval.zero_variance:
        return TInterval(interval.estimate, interval.lo, interval.hi, level, interval.df, interval.t, interval.p,
                         interval.se, True)
    return _t_interval(interval.estimate, interval.se, interval.df, level)


def one_sample(values: SeedValues, level: float = R.INTERVAL_LEVEL) -> TInterval:
    """One-sample t-interval of the seed mean (per-arm intervals; within-seed differences such as G3's)."""
    _, a = _check(values, "values", 2)
    se = math.sqrt(statistics.variance(a.tolist()) / a.size)
    return _t_interval(_mean(a), se, float(a.size - 1), level)


def common_seeds(x: SeedValues, y: SeedValues) -> list[int]:
    """Seeds present in both arms: the pairs of Part 5.1 (paired by seed id; Q-seed-collision)."""
    return sorted(set(x) & set(y))


def paired(x: SeedValues, y: SeedValues, level: float = R.INTERVAL_LEVEL) -> TInterval:
    """Paired t-interval over the common seeds (Part 5.1: reported beside the unpaired, never deciding)."""
    seeds = common_seeds(x, y)
    return one_sample({s: x[s] - y[s] for s in seeds}, level)


# ---------------------------------------------------------------------------
# Part 5.3: percentile bootstrap over seed indices
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapInterval:
    """A percentile bootstrap interval with its recorded mode, resamples, undefined count and seed."""

    lo: float
    hi: float
    level: float
    mode: str  # "joint" (one index draw for both arms), "per_arm", "one_sample", "paired"
    resamples: int
    n_nan: int
    generator_seed: tuple[int, int]

    @property
    def flagged(self) -> bool:
        return self.n_nan > NAN_FLAG_SHARE * self.resamples


def bootstrap_difference(x: SeedValues, y: SeedValues, *, analysis_id: str, level: float = R.INTERVAL_LEVEL,
                         resamples: Optional[int] = None) -> BootstrapInterval:
    """Percentile bootstrap of mean(x) - mean(y) from resamples of seed indices.

    When the arms share their seed list, one index draw serves both, so "a resample takes every
    outcome of the chosen seeds together"; otherwise each arm is resampled on its own indices
    (Q-bootstrap-details).
    """
    xs, xa = _check(x, "x")
    ys, ya = _check(y, "y")
    resamples = _resamples(resamples)
    rng = generator(analysis_id)
    if xs == ys:
        idx = rng.integers(0, xa.size, size=(resamples, xa.size))
        samples = xa[idx].mean(axis=1) - ya[idx].mean(axis=1)
        mode = "joint"
    else:
        ix = rng.integers(0, xa.size, size=(resamples, xa.size))
        iy = rng.integers(0, ya.size, size=(resamples, ya.size))
        samples = xa[ix].mean(axis=1) - ya[iy].mean(axis=1)
        mode = "per_arm"
    lo, hi, n_nan = percentile_interval(samples, level)
    return BootstrapInterval(lo, hi, level, mode, resamples, n_nan, tuple(generator_seed(analysis_id)))


def bootstrap_mean(values: SeedValues, *, analysis_id: str, level: float = R.INTERVAL_LEVEL,
                   resamples: Optional[int] = None, mode: str = "one_sample") -> BootstrapInterval:
    """Percentile bootstrap of a seed mean (one-sample, or of within-seed differences)."""
    _, a = _check(values, "values")
    resamples = _resamples(resamples)
    rng = generator(analysis_id)
    idx = rng.integers(0, a.size, size=(resamples, a.size))
    lo, hi, n_nan = percentile_interval(a[idx].mean(axis=1), level)
    return BootstrapInterval(lo, hi, level, mode, resamples, n_nan, tuple(generator_seed(analysis_id)))


# ---------------------------------------------------------------------------
# Part 5.3: interquartile mean (descriptive; Q-iqm)
# ---------------------------------------------------------------------------


def iqm(values: Sequence[float]) -> float:
    """The interquartile mean: the mean of the middle 50 percent (trim R.IQM_TRIM from each end)."""
    array = np.asarray(list(values), dtype=float)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise StatsError("the IQM needs finite values")
    return float(st.trim_mean(array, R.IQM_TRIM))


@dataclass(frozen=True)
class IqmResult:
    """A signed sum of IQMs with its stratified-bootstrap percentile interval and generator seed."""

    estimate: float
    lo: float
    hi: float
    level: float
    resamples: int
    generator_seed: tuple[int, int]


def stratified_iqm(groups: Sequence[Mapping[int, Sequence[float]]], signs: Sequence[float], *, analysis_id: str,
                   level: float = R.INTERVAL_LEVEL, resamples: Optional[int] = None) -> IqmResult:
    """sum_k sign_k * IQM(pooled episodes of group k), with a two-level stratified bootstrap.

    Each group maps seed -> that seed's per-episode values. A resample draws seeds with replacement,
    then episodes with replacement within each drawn seed (Q-iqm: "seeds drawn with replacement ..., then
    episodes drawn with replacement within each drawn seed"). Groups with the same seed list share their
    seed draws (a gap's C_cond and C_ID of one arm; two arms that share seeds). Descriptive only: Part
    5.1 forbids episode resampling for the registered intervals.
    """
    if not groups:
        raise StatsError("no groups")
    if len(groups) != len(signs):
        raise StatsError("one sign per group")
    arrays, seed_lists = [], []
    for group in groups:
        seeds = sorted(group)
        if not seeds:
            raise StatsError("a group has no seeds")
        lengths = {len(group[s]) for s in seeds}
        if len(lengths) != 1:
            raise StatsError("every seed of a group needs the same number of episodes")
        array = np.asarray([list(group[s]) for s in seeds], dtype=float)
        if not np.all(np.isfinite(array)):
            raise StatsError("episode values must be finite")
        arrays.append(array)
        seed_lists.append(tuple(seeds))
    estimate = sum(sign * iqm(a.ravel()) for sign, a in zip(signs, arrays))
    resamples = _resamples(resamples)
    rng = generator(analysis_id)
    clusters = sorted(set(seed_lists))
    # Each group's values as codes into its sorted distinct values: a resample's trimmed mean is then read
    # from counts (``_trim_mean_rows``), the same number as scipy's trim_mean without a partition per row.
    coded = [np.unique(a, return_inverse=True) for a in arrays]
    samples = np.empty(resamples)
    for start in range(0, resamples, IQM_CHUNK):
        b = min(IQM_CHUNK, resamples - start)
        draws = {c: rng.integers(0, len(c), size=(b, len(c))) for c in clusters}
        total = np.zeros(b)
        for sign, array, seeds, (distinct, codes) in zip(signs, arrays, seed_lists, coded):
            n_seeds, m = array.shape
            s_idx = draws[seeds]
            e_idx = rng.integers(0, m, size=(b, n_seeds, m))
            rows = codes.reshape(n_seeds, m)[s_idx[:, :, None], e_idx].reshape(b, n_seeds * m)
            total += sign * _trim_mean_rows(rows, distinct, R.IQM_TRIM)
        samples[start:start + b] = total
    lo, hi, _ = percentile_interval(samples, level)
    return IqmResult(float(estimate), lo, hi, level, resamples, tuple(generator_seed(analysis_id)))


def _trim_mean_rows(codes: np.ndarray, distinct: np.ndarray, proportion: float) -> np.ndarray:
    """scipy.stats.trim_mean(proportion) of every row of ``distinct[codes]``, by counting.

    ``distinct`` is sorted and ``codes`` index it, so a row's sorted values are runs of equal values;
    scipy cuts ``int(proportion * n)`` values from each end, and the kept positions [low, high) overlap
    each run by a count that is read from the cumulative counts. Exact up to summation order.
    """
    b, n = codes.shape
    v = distinct.size
    counts = np.bincount((codes + (np.arange(b) * v)[:, None]).ravel(), minlength=b * v).reshape(b, v)
    upto = np.cumsum(counts, axis=1)
    low = int(proportion * n)
    high = n - low
    kept = np.clip(np.minimum(upto, high) - np.maximum(upto - counts, low), 0, None)
    return kept @ distinct / (high - low)


EpisodeParts = Sequence[tuple[float, Sequence[float]]]


@dataclass(frozen=True)
class EpisodeOutcome:
    """A per-seed outcome that is a signed sum of episode means, with the episodes behind it (Q-iqm).

    ``value(record)`` is the outcome per seed, what every registered estimate reads (a callable like
    any getter). ``episodes(record)`` gives ``[(sign, per-episode values), ...]`` whose signed means
    sum to that outcome (a gap: ``[(+1, C_cond episodes), (-1, C_ID episodes)]``), or None when the
    per-episode records are missing. ``source`` names the episodes; with the two arms' ids it names
    the interquartile mean, so one pair of arms on one source gets one IQM whichever verdict reads it.
    """

    source: str
    value: Callable[[Any], Optional[float]]
    episodes: Callable[[Any], Optional[EpisodeParts]]

    def __call__(self, record: Any) -> Optional[float]:
        return self.value(record)


def _episode_groups(parts_by_seed: Mapping[int, EpisodeParts], flip: bool) -> tuple[list[dict], list[float]]:
    """One group per part (every seed must carry the same signed parts), for ``stratified_iqm``."""
    seeds = sorted(parts_by_seed)
    if not seeds:
        raise StatsError("an arm has no seeds")
    signs = [float(sign) for sign, _ in parts_by_seed[seeds[0]]]
    groups: list[dict] = [dict() for _ in signs]
    for seed in seeds:
        parts = parts_by_seed[seed]
        if [float(sign) for sign, _ in parts] != signs:
            raise StatsError("every seed of an arm needs the same signed episode parts")
        for k, (_, values) in enumerate(parts):
            groups[k][seed] = list(values)
    return groups, [-s if flip else s for s in signs]


def iqm_contrast(x: Mapping[int, EpisodeParts], y: Optional[Mapping[int, EpisodeParts]], *, analysis_id: str,
                 level: float = R.INTERVAL_LEVEL, resamples: Optional[int] = None) -> dict[str, Any]:
    """Part 5.3's interquartile mean of a two-arm estimand (or of one arm's within-seed outcome, y None).

    IQM per arm = sum over its parts of sign x IQM(the part's episodes pooled over seeds) (Q-iqm: "the
    IQM of a gap is IQM(C_cond episodes) - IQM(C_ID episodes)"); the difference x - y with the two-level
    stratified bootstrap of ``stratified_iqm`` (descriptive only).
    """
    groups, signs = _episode_groups(x, flip=False)
    n_x = len(groups)
    if y is not None:
        y_groups, y_signs = _episode_groups(y, flip=True)
        groups, signs = groups + y_groups, signs + y_signs
    result = stratified_iqm(groups, signs, analysis_id=analysis_id, level=level, resamples=resamples)
    per_group = [sign * iqm([v for values in group.values() for v in values]) for sign, group in zip(signs, groups)]
    return {"iqm_x": sum(per_group[:n_x]), "iqm_y": None if y is None else -sum(per_group[n_x:]),
            "iqm_diff": result.estimate, "iqm_lo": result.lo, "iqm_hi": result.hi, "iqm_level": result.level,
            "iqm_resamples": result.resamples, "iqm_generator_seed": " ".join(map(str, result.generator_seed))}


# ---------------------------------------------------------------------------
# Part 5.4: Spearman's rank correlation with a seed-resampling interval
# ---------------------------------------------------------------------------


def _settled_array(values: Sequence[float]) -> np.ndarray:
    """``settled`` applied to every value (the ranks of Part 5.4 compare values with each other)."""
    return np.asarray([settled(v) for v in np.asarray(values, dtype=float).ravel()], dtype=float)


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman's rho with average ranks for ties; NaN when either variable is constant.

    Values are ranked as compared (``settled``): equal values that differ by floating-point noise tie.
    """
    xa, ya = _settled_array(x), _settled_array(y)
    if xa.size != ya.size or xa.size < 2:
        raise StatsError("Spearman needs two paired sequences of at least two values")
    rx, ry = st.rankdata(xa), st.rankdata(ya)
    if np.all(rx == rx[0]) or np.all(ry == ry[0]):
        return math.nan
    return float(np.corrcoef(rx, ry)[0, 1])


@dataclass(frozen=True)
class SpearmanResult:
    """Part 5.4's rho with its seed-bootstrap percentile interval, bootstrap p and undefined-resample count."""

    rho: float
    lo: float
    hi: float
    level: float
    p_boot: float
    n_points: int
    n_seeds: int
    resamples: int
    n_nan: int
    generator_seed: tuple[int, int]

    @property
    def flagged(self) -> bool:
        return self.n_nan > NAN_FLAG_SHARE * self.resamples


def _weighted_ranks(weights: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Average ranks of every point in every resample, where point p appears weights[b, p] times.

    In a resample, the copies of p tie with each other and with every equal value, so p's average rank
    is (weight of smaller values) + (weight of equal values + 1) / 2: exactly scipy's 'average' ranks
    of the expanded sample.
    """
    less = (values[None, :] < values[:, None]).astype(float)  # less[p, q]: value q < value p
    equal = (values[None, :] == values[:, None]).astype(float)
    return weights @ less.T + (weights @ equal.T + 1.0) / 2.0


def spearman_bootstrap(points_by_seed: Mapping[Hashable, Sequence[tuple[float, float]]], *, analysis_id: str,
                       level: float = R.INTERVAL_LEVEL, resamples: Optional[int] = None) -> SpearmanResult:
    """Part 5.4: rho over every (factor, outcome) point, with a percentile interval from resampling seeds.

    A resample draws seed ids with replacement from the union of seeds, and each drawn seed carries
    all of its points (its factor levels) together: point p of seed s enters the resample as many
    times as s is drawn, and rho is computed on that expanded sample with average ranks. Undefined
    resamples (a constant variable) are dropped and counted. The bootstrap p-value used for Holm is
    min(1, 2 min(#rho* <= 0, #rho* >= 0) / B_defined) (Q-bootstrap-details, answered in Table 9.1):
    Part 5.2 asks for Holm on the secondary claims but registers no p-value for a correlation.
    """
    seeds = sorted(points_by_seed)
    if not seeds:
        raise StatsError("no seeds")
    xs, ys, owner = [], [], []
    for i, s in enumerate(seeds):
        for a, b in points_by_seed[s]:
            xs.append(float(a))
            ys.append(float(b))
            owner.append(i)
    X, Y, owner_index = np.asarray(xs), np.asarray(ys), np.asarray(owner, dtype=int)
    if X.size < 2:
        raise StatsError("Spearman needs at least two points")
    if not (np.all(np.isfinite(X)) and np.all(np.isfinite(Y))):
        raise StatsError("Spearman points must be finite")
    X, Y = _settled_array(X), _settled_array(Y)  # ranked as compared: noise-only differences tie
    rho = spearman(X, Y)
    resamples = _resamples(resamples)
    rng = generator(analysis_id)
    idx = rng.integers(0, len(seeds), size=(resamples, len(seeds)))
    counts = (idx[:, :, None] == np.arange(len(seeds))[None, None, :]).sum(axis=1).astype(float)
    weights = counts[:, owner_index]  # (resamples, points): how often each point is drawn
    rx, ry = _weighted_ranks(weights, X), _weighted_ranks(weights, Y)
    n = weights.sum(axis=1, keepdims=True)
    tiny = 1e-12 * np.maximum(n[:, 0], 1.0) ** 3  # rank variances are sums of squared half-integers
    with np.errstate(invalid="ignore", divide="ignore"):  # n = 0: a resample of seeds without points
        dx = rx - (weights * rx).sum(axis=1, keepdims=True) / n
        dy = ry - (weights * ry).sum(axis=1, keepdims=True) / n
        cov = (weights * dx * dy).sum(axis=1)
        vx, vy = (weights * dx * dx).sum(axis=1), (weights * dy * dy).sum(axis=1)
        samples = np.where((vx > tiny) & (vy > tiny), cov / np.sqrt(vx * vy), np.nan)
    lo, hi, n_nan = percentile_interval(samples, level)
    finite = samples[np.isfinite(samples)]
    if finite.size:
        p_boot = min(1.0, 2 * min(int(np.sum(finite <= 0)), int(np.sum(finite >= 0))) / finite.size)
    else:
        p_boot = math.nan
    return SpearmanResult(rho, lo, hi, level, p_boot, int(X.size), len(seeds), resamples, n_nan,
                          tuple(generator_seed(analysis_id)))


# ---------------------------------------------------------------------------
# Part 5.2: Holm
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HolmResult:
    """One family member's raw and Holm-adjusted p, whether it is rejected, and the family's size."""

    p: float
    adjusted: float
    reject: bool
    family_size: int


def holm(pvalues: Mapping[Hashable, Optional[float]], alpha: float = R.ALPHA) -> dict[Hashable, HolmResult]:
    """Holm's step-down procedure; members without a p-value are left out (the family shrinks)."""
    tested = {k: float(p) for k, p in pvalues.items() if p is not None and not math.isnan(float(p))}
    order = sorted(tested, key=lambda k: (tested[k], repr(k)))
    m = len(order)
    out: dict[Hashable, HolmResult] = {}
    running = 0.0
    for i, key in enumerate(order):
        running = max(running, min(1.0, (m - i) * tested[key]))
        out[key] = HolmResult(tested[key], running, running <= alpha, m)
    return out


@dataclass(frozen=True)
class Incomplete:
    """A family member whose p-value cannot be computed YET (its data are incomplete: an arm not yet
    matched, a value not yet evaluated), as against a member that is not tested (None: no arm by
    design, a recorded cut, an infeasible task, an arm rule 6 leaves out). Q-holm-families' "Families
    shrink only to cells not tested" applies to the second only: a family never shrinks on data still to
    come, or a claim could pass Holm only because a sibling's data are missing."""

    reason: str


def holm_decisions(pvalues: Mapping[Hashable, Union[None, float, Incomplete]], alpha: float = R.ALPHA
                   ) -> tuple[dict[Hashable, HolmResult], dict[Hashable, Optional[bool]], dict[Hashable, str]]:
    """Holm over the members with a p-value, with the family's incomplete members kept in view.

    Returns (``holm`` over the tested members, {tested member: survives}, {incomplete member: reason}).
    Without an incomplete member, survives is Holm's rejection. With incomplete members, a member's
    survival is decided only where every p-value they could bring gives the same answer: True if it is
    rejected with each incomplete member at p = 1 (the least favourable value: a member ranked after it
    only raises the divisors m - i + 1 of those before it, and one ranked before it at a p-value it
    rejects leaves the others' divisors as they were), False if it is not rejected without them (adding
    a member never lowers an adjusted p-value, and a member that turns out not tested leaves the family),
    otherwise None: the claim waits on the data still to come.
    """
    incomplete = {k: p.reason for k, p in pvalues.items() if isinstance(p, Incomplete)}
    results = holm({k: p for k, p in pvalues.items() if not isinstance(p, Incomplete)}, alpha)
    if not incomplete:
        return results, {k: h.reject for k, h in results.items()}, incomplete
    worst = holm({**{k: h.p for k, h in results.items()}, **dict.fromkeys(incomplete, 1.0)}, alpha)
    survives = {k: True if worst[k].reject else (None if h.reject else False) for k, h in results.items()}
    return results, survives, incomplete


# ---------------------------------------------------------------------------
# Part 5.5: power; equation (12)
# ---------------------------------------------------------------------------


def _nct_upper(x: float, df: int, nc: float) -> float:
    """P(T > x) for T noncentral t; a NaN from scipy (none seen for ``sf``) counts as 0."""
    value = float(st.nct.sf(x, df, nc))
    return 0.0 if math.isnan(value) else value


def power_two_sample(d: float, n: int, alpha: float = R.ALPHA) -> float:
    """Power of the two-sided two-sample t-test with n seeds per arm at standardised difference d.

    Noncentral t with df = 2n - 2 and noncentrality d sqrt(n / 2) (Part 5.5). The lower tail
    P(T < -crit) is computed as P(-T > crit), -T being noncentral t with noncentrality -nc:
    scipy 1.15.3's ``nct.cdf(-crit, df, nc)`` returns NaN for large nc (for example n = 2,
    alpha = 0.025, d = 8), while ``nct.sf(crit, df, -nc)`` is finite there and equal to it wherever
    both are finite (checked on n = 2..40, alpha in {0.05, 0.025, 0.01}, |d| <= 64).
    """
    if n < 2:
        raise StatsError("power needs at least two seeds per arm")
    df = 2 * n - 2
    nc = d * math.sqrt(n / 2)
    crit = float(st.t.ppf(1 - alpha / 2, df))
    return _nct_upper(crit, df, nc) + _nct_upper(crit, df, -nc)


@functools.lru_cache(maxsize=None)
def detectable_d(n: int, alpha: float = R.ALPHA, power: float = R.POWER_TARGET) -> Optional[float]:
    """The standardised difference at which the test reaches ``power`` with n seeds per arm.

    None when it cannot be found (not bracketed below d = 64, or the root finder fails): Part 5.5's
    detectable size is an annotation beside an estimate, never a decision, so its absence is reported
    ("not computable") and never stops the analysis. StatsError with fewer than two seeds per arm.
    """
    if n < 2:
        raise StatsError("power needs at least two seeds per arm")
    lo, hi = 1e-6, 1.0
    try:
        while power_two_sample(hi, n, alpha) < power:
            hi *= 2
            if hi > 64:
                return None
        return float(optimize.brentq(lambda d: power_two_sample(d, n, alpha) - power, lo, hi, xtol=1e-12))
    except (ValueError, RuntimeError):  # brentq: a NaN value or no convergence
        return None


def u80_t_quantile(n: int = len(R.PILOT_SEEDS)) -> float:
    """The t quantile behind U80: 0.90 quantile with n - 1 degrees of freedom (1.886 for three seeds)."""
    return float(st.t.ppf((1 + U80_COVERAGE) / 2, n - 1))


def g3_pairs(late: Mapping[int, float], reference: Mapping[int, float]) -> list[tuple[int, int]]:
    """The (late seed, reference seed) pairs of equation (12)'s paired form (Q-g3-pairing, answered in Table 9.1).

    "s is the sample SD of the three per-seed differences gap_hazard(N = 0.50) - gap_hazard(N = 0), paired by
    seed id. If a replacement seed leaves fewer than three seed-matched pairs, the runs left unpaired are paired
    in increasing seed order (each arm's remaining runs sorted by seed, then matched one to one)". The seed-matched
    pairs come first, by seed; a run left over when the arms hold different counts is not paired.
    """
    common = sorted(set(late) & set(reference))
    rest_late = sorted(set(late) - set(common))
    rest_reference = sorted(set(reference) - set(common))
    return [(s, s) for s in common] + list(zip(rest_late, rest_reference))


def u80(values: Sequence[float]) -> float:
    """Equation (12): U80 = mean + 1.886 s / sqrt(3), over exactly the pilot's three paired differences
    (``g3_pairs``)."""
    n = len(R.PILOT_SEEDS)
    if len(values) != n:
        raise StatsError(f"equation (12) is registered for exactly {n} seeds (R.PILOT_SEEDS); got {len(values)}")
    return statistics.fmean(values) + R.U80_T_QUANTILE * statistics.stdev(values) / math.sqrt(n)


def u80_pooled(late: Sequence[float], reference: Sequence[float]) -> float:
    """Equation (12) with the pooled s of the two arms, as the go report computes it (pilot/go_decision.py): a
    descriptive figure only, never deciding G3 (Q-g3-pairing, answered in Table 9.1)."""
    n = len(R.PILOT_SEEDS)
    delta = statistics.fmean(late) - statistics.fmean(reference)
    s = math.sqrt((statistics.variance(reference) + statistics.variance(late)) / 2)
    return delta + R.U80_T_QUANTILE * s / math.sqrt(n)


# ---------------------------------------------------------------------------
# The two-arm estimate of Part 5.3
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TwoArmEstimate:
    """Everything Part 5.3 reports for one two-arm estimand, plus the paired reading of Part 5.1."""

    analysis_id: str
    level: float
    seeds_x: tuple[int, ...]
    seeds_y: tuple[int, ...]
    mean_x: float
    mean_y: float
    diff: float
    welch: TInterval
    bootstrap: BootstrapInterval
    cohens_d: Optional[float]
    paired: Optional[TInterval]
    paired_seeds: tuple[int, ...]
    detectable_d: Optional[float]

    @property
    def below_detectable(self) -> Optional[bool]:
        """Part 5.5 annotation: |d| below the detectable size at this n (never a decision)."""
        if self.cohens_d is None or self.detectable_d is None:
            return None
        return abs(self.cohens_d) < self.detectable_d

    def interval(self, method: str) -> tuple[float, float]:
        """(lo, hi) of the primary Welch interval, or of the percentile interval (Q-interval's other reading)."""
        if method == "welch":
            return self.welch.lo, self.welch.hi
        if method == "percentile":
            return self.bootstrap.lo, self.bootstrap.hi
        raise ValueError(f"unknown interval {method!r}")

    def row(self) -> dict[str, Any]:
        """A flat record for tables."""
        return {
            "analysis_id": self.analysis_id, "level": self.level,
            "n_x": len(self.seeds_x), "n_y": len(self.seeds_y),
            "seeds_x": " ".join(map(str, self.seeds_x)), "seeds_y": " ".join(map(str, self.seeds_y)),
            "mean_x": self.mean_x, "mean_y": self.mean_y, "diff": self.diff,
            "welch_lo": self.welch.lo, "welch_hi": self.welch.hi, "welch_df": self.welch.df,
            "welch_t": self.welch.t, "welch_p": self.welch.p, "welch_zero_variance": self.welch.zero_variance,
            "boot_lo": self.bootstrap.lo, "boot_hi": self.bootstrap.hi, "boot_mode": self.bootstrap.mode,
            "boot_resamples": self.bootstrap.resamples, "boot_n_nan": self.bootstrap.n_nan,
            "boot_generator_seed": " ".join(map(str, self.bootstrap.generator_seed)),
            "cohens_d": self.cohens_d,
            "paired_n": len(self.paired_seeds),
            "paired_diff": None if self.paired is None else self.paired.estimate,
            "paired_lo": None if self.paired is None else self.paired.lo,
            "paired_hi": None if self.paired is None else self.paired.hi,
            "detectable_d": self.detectable_d, "below_detectable": self.below_detectable,
        }


def estimate_two_arms(x: SeedValues, y: SeedValues, *, analysis_id: str,
                      level: float = R.INTERVAL_LEVEL) -> TwoArmEstimate:
    """mean(x) - mean(y) with Welch (primary), the seed bootstrap, Cohen's d and the paired reading.

    StatsError with fewer than two seeds in an arm. The detectable size is computed at the smaller
    arm's n with alpha = 1 - level (Part 5.5; an annotation only).
    """
    xs, xa = _check(x, "x", 2)
    ys, ya = _check(y, "y", 2)
    common = common_seeds(x, y)
    pair = paired(x, y, level) if len(common) >= 2 else None
    n = min(len(xs), len(ys))
    mean_x, mean_y = _mean(xa), _mean(ya)
    return TwoArmEstimate(
        analysis_id=analysis_id, level=level, seeds_x=tuple(xs), seeds_y=tuple(ys),
        mean_x=mean_x, mean_y=mean_y, diff=mean_x - mean_y,
        welch=welch(x, y, level), bootstrap=bootstrap_difference(x, y, analysis_id=analysis_id, level=level),
        cohens_d=cohens_d(x, y), paired=pair, paired_seeds=tuple(common),
        detectable_d=detectable_d(n, alpha=round(1 - level, 12)),  # 1e-12 strips the noise of 1 - level
    )


def to_plain(value: Any) -> Any:
    """Dataclasses and numpy scalars as JSON-ready Python values (NaN and infinity become None)."""
    if hasattr(value, "__dataclass_fields__"):
        return to_plain(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(v) for v in value]
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value

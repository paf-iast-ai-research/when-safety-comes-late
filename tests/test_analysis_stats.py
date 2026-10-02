"""Statistics of Part 5 (analysis/stats.py): checked against scipy and closed forms (Role 4)."""

from __future__ import annotations

import math
import statistics
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest
from scipy import stats as st

sys.path.insert(0, str(Path(__file__).parent))
import test_analysis_synthetic as syn  # noqa: E402

from analysis import data, report  # noqa: E402
from analysis import stats as S  # noqa: E402
from configs import registered as R  # noqa: E402

X = {0: 3.1, 1: 4.7, 2: 2.2, 3: 5.9, 4: 4.0}
Y = {0: 1.0, 1: 2.5, 2: 0.4, 3: 1.9, 4: 3.3}


# ---------------------------------------------------------------------------
# Part 5.3
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("level", [R.INTERVAL_LEVEL, R.INTERVAL_LEVEL_PRIMARY_STUDY_B])
def test_welch_matches_scipy(level) -> None:
    ours = S.welch(X, Y, level)
    ref = st.ttest_ind(list(X.values()), list(Y.values()), equal_var=False)
    ci = ref.confidence_interval(level)
    assert ours.estimate == pytest.approx(statistics.fmean(X.values()) - statistics.fmean(Y.values()))
    assert ours.t == pytest.approx(ref.statistic) and ours.p == pytest.approx(ref.pvalue) and ours.df == pytest.approx(ref.df)
    assert (ours.lo, ours.hi) == (pytest.approx(ci.low), pytest.approx(ci.high))
    unequal = {**Y, 5: 9.0, 6: -1.0}
    ref = st.ttest_ind(list(X.values()), list(unequal.values()), equal_var=False)
    ours = S.welch(X, unequal, level)
    assert ours.df == pytest.approx(ref.df) and ours.lo == pytest.approx(ref.confidence_interval(level).low)


def test_one_sample_and_paired_match_scipy() -> None:
    ours = S.one_sample(X, R.INTERVAL_LEVEL)
    ref = st.ttest_1samp(list(X.values()), 0.0)
    ref_ci = ref.confidence_interval(R.INTERVAL_LEVEL)
    assert (ours.lo, ours.hi) == (pytest.approx(ref_ci.low), pytest.approx(ref_ci.high))
    pair = S.paired(X, {**Y, 9: 100.0}, R.INTERVAL_LEVEL)  # seed 9 has no partner: pairs use the common seeds
    ref = st.ttest_rel([X[s] for s in range(5)], [Y[s] for s in range(5)])
    assert pair.p == pytest.approx(ref.pvalue) and pair.lo == pytest.approx(ref.confidence_interval(R.INTERVAL_LEVEL).low)
    assert S.common_seeds(X, {**Y, 9: 1.0}) == [0, 1, 2, 3, 4]


def test_cohens_d_uses_the_pooled_seed_sd_of_equation_12() -> None:
    n1, n2 = len(X), len(Y)
    s1, s2 = statistics.variance(X.values()), statistics.variance(Y.values())
    sp = math.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    assert S.cohens_d(X, Y) == pytest.approx((statistics.fmean(X.values()) - statistics.fmean(Y.values())) / sp)
    assert S.pooled_sd(X, Y) == pytest.approx(math.sqrt((s1 + s2) / 2))  # equal n: the mean of the variances
    assert S.cohens_d({0: 1.0, 1: 1.0}, {0: 1.0, 1: 1.0}) is None  # zero pooled SD
    assert S.mean_difference(X, Y) == pytest.approx(S.welch(X, Y).estimate)


def test_welch_with_zero_variance_and_too_few_seeds() -> None:
    flat = S.welch({0: 2.0, 1: 2.0}, {0: 1.0, 1: 1.0})
    assert flat.zero_variance and flat.lo == flat.hi == 1.0 and flat.p == 0.0
    same = S.welch({0: 1.0, 1: 1.0}, {0: 1.0, 1: 1.0})
    assert same.lo == same.hi == 0.0 and same.p == 1.0
    # review round 5: constant arms at one value whose means differ by a rounding residue (fmean([0.7] * 6) != 0.7)
    residue = S.welch({s: 0.7 for s in range(6)}, {s: 0.7 for s in range(5)})
    assert residue.lo == residue.hi == residue.estimate == 0.0 and residue.p == 1.0
    assert not S.interval_excludes_zero(residue.lo, residue.hi)
    # seeds equal up to floating-point noise (every value settles to 0.3) are equal: no interval off zero
    noisy = {s: (30.30 - 30.00, 0.1 + 0.2)[s % 2] for s in range(5)}
    flat3 = {s: 0.3 for s in range(5)}
    for interval in (S.welch(noisy, flat3), S.paired(noisy, flat3)):
        assert interval.zero_variance and interval.lo == interval.hi == 0.0 and interval.p == 1.0
    boot = S.bootstrap_difference(noisy, flat3, analysis_id="test-noise", resamples=200)
    assert boot.lo == boot.hi == 0.0 and S.cohens_d(noisy, flat3) is None
    with pytest.raises(S.StatsError):
        S.welch({0: 1.0}, Y)
    with pytest.raises(S.StatsError):
        S.welch({0: 1.0, 1: math.nan}, Y)
    with pytest.raises(S.StatsError):
        S.welch({0: 1.0, 1: None}, Y)


def test_the_seed_bootstrap_is_deterministic_and_records_its_generator() -> None:
    a = S.bootstrap_difference(X, Y, analysis_id="H1|test")
    b = S.bootstrap_difference(X, Y, analysis_id="H1|test")
    c = S.bootstrap_difference(X, Y, analysis_id="H1|other")
    assert (a.lo, a.hi) == (b.lo, b.hi) and (a.lo, a.hi) != (c.lo, c.hi)
    assert a.resamples == R.BOOTSTRAP_RESAMPLES and a.mode == "joint" and a.n_nan == 0
    assert a.generator_seed == (S.BOOTSTRAP_SEED, zlib.crc32(b"H1|test"))
    # the same computation by hand: one index draw of the shared seed list serves both arms
    rng = np.random.default_rng([S.BOOTSTRAP_SEED, zlib.crc32(b"H1|test")])
    idx = rng.integers(0, 5, size=(R.BOOTSTRAP_RESAMPLES, 5))
    xs, ys = np.array([X[s] for s in range(5)]), np.array([Y[s] for s in range(5)])
    samples = xs[idx].mean(axis=1) - ys[idx].mean(axis=1)
    lo, hi = np.quantile(samples, [0.025, 0.975], method="linear")
    assert (a.lo, a.hi) == (lo, hi)
    per_arm = S.bootstrap_difference(X, {**Y, 7: 2.0}, analysis_id="H1|test")
    assert per_arm.mode == "per_arm"
    assert a.lo <= S.mean_difference(X, Y) <= a.hi


def test_joint_resampling_keeps_pairs_together() -> None:
    x = {s: 10.0 + s for s in range(5)}
    y = {s: 7.0 + s for s in range(5)}  # every pair differs by exactly 3
    boot = S.bootstrap_difference(x, y, analysis_id="pairs")
    assert boot.lo == pytest.approx(3.0) and boot.hi == pytest.approx(3.0)
    one = S.bootstrap_mean(x, analysis_id="one")
    assert one.lo < statistics.fmean(x.values()) < one.hi and one.mode == "one_sample"


def test_iqm_and_its_two_level_bootstrap() -> None:
    values = list(range(1, 101))
    assert S.iqm(values) == pytest.approx(st.trim_mean(values, 0.25)) == pytest.approx(50.5)
    assert S.iqm([0.0, 0.0, 0.0, 100.0]) == 0.0  # the tails are cut
    rng = np.random.default_rng(0)
    group = {s: list(rng.integers(0, 40, size=R.EVAL_EPISODES).astype(float)) for s in range(5)}
    base = {s: list(rng.integers(0, 30, size=R.EVAL_EPISODES).astype(float)) for s in range(5)}
    one = S.stratified_iqm([group], [1.0], analysis_id="iqm", resamples=2000)
    assert one.estimate == pytest.approx(S.iqm([v for s in group.values() for v in s]))
    assert one.lo <= one.estimate <= one.hi and one.resamples == 2000
    again = S.stratified_iqm([group], [1.0], analysis_id="iqm", resamples=2000)
    assert (one.lo, one.hi) == (again.lo, again.hi)
    gap = S.stratified_iqm([group, base], [1.0, -1.0], analysis_id="iqm-gap", resamples=2000)
    assert gap.estimate == pytest.approx(one.estimate - S.iqm([v for s in base.values() for v in s]))
    with pytest.raises(S.StatsError):
        S.stratified_iqm([{0: [1.0, 2.0], 1: [1.0]}], [1.0], analysis_id="bad")
    with pytest.raises(S.StatsError, match="no groups"):
        S.stratified_iqm([], [], analysis_id="bad")
    with pytest.raises(S.StatsError, match="one sign per group"):
        S.stratified_iqm([group], [1.0, -1.0], analysis_id="bad")
    with pytest.raises(S.StatsError):
        S.iqm([1.0, math.nan])


def test_the_iqm_bootstrap_draws_seeds_then_episodes_within_each_drawn_seed() -> None:
    """Q-iqm: "seeds drawn with replacement (arms with the same seed list share the draw), then episodes drawn with
    replacement within each drawn seed"; rejected: "a one-level bootstrap of episodes within fixed seeds (understates
    the uncertainty)". Episodes constant within a seed and different between seeds vary only with the seed draw; one
    seed's episodes vary only with the episode draw; one group taken twice with opposite signs cancels in every
    resample only when the two share their seed draw."""
    between = {s: [float(s)] * 20 for s in range(5)}
    seeds = S.stratified_iqm([between], [1.0], analysis_id="levels", resamples=500)
    assert seeds.hi - seeds.lo > 0  # a bootstrap within fixed seeds gives zero width here
    episodes = S.stratified_iqm([{0: [float(v) for v in range(20)]}], [1.0], analysis_id="levels", resamples=500)
    assert episodes.hi - episodes.lo > 0  # one seed: only the episode draw can move the IQM
    shared = S.stratified_iqm([between, between], [1.0, -1.0], analysis_id="levels", resamples=500)
    assert shared.estimate == shared.lo == shared.hi == 0.0


# ---------------------------------------------------------------------------
# Part 5.4
# ---------------------------------------------------------------------------


def test_spearman_matches_scipy_with_ties() -> None:
    x = [0.0, 0.0, 0.1, 0.1, 0.25, 0.25, 0.5, 0.5]
    y = [1.0, 3.0, 2.0, 2.0, 5.0, 4.0, 9.0, 9.0]
    assert S.spearman(x, y) == pytest.approx(st.spearmanr(x, y).statistic)
    assert math.isnan(S.spearman([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))


def test_spearman_bootstrap_resamples_whole_seeds() -> None:
    # each seed's points are perfectly ordered in N and seeds differ in level: any resample is ordered
    points = {s: [(n, 10 * n + 0.01 * s) for n in R.ONSET_FRACTIONS] for s in R.SEEDS}
    res = S.spearman_bootstrap(points, analysis_id="trend")
    flat = [p for s in R.SEEDS for p in points[s]]
    assert res.rho == pytest.approx(st.spearmanr([p[0] for p in flat], [p[1] for p in flat]).statistic)
    assert res.n_points == R.H1_TREND_POINTS and res.n_seeds == len(R.SEEDS)
    assert res.lo > 0 and res.p_boot == 0.0 and res.n_nan == 0 and res.resamples == R.BOOTSTRAP_RESAMPLES
    again = S.spearman_bootstrap(points, analysis_id="trend")
    assert (again.lo, again.hi) == (res.lo, res.hi)
    noisy = {s: [(n, (s * 7 + i * 3) % 5) for i, n in enumerate(R.ONSET_FRACTIONS)] for s in R.SEEDS}
    res = S.spearman_bootstrap(noisy, analysis_id="noise", level=R.INTERVAL_LEVEL_PRIMARY_STUDY_B)
    assert res.lo <= res.rho <= res.hi and 0 < res.p_boot <= 1 and res.level == R.INTERVAL_LEVEL_PRIMARY_STUDY_B


def test_spearman_bootstrap_a_resample_of_seeds_without_points_warns_nothing() -> None:
    """A seed with no points can be drawn alone (n = 0): that resample is undefined, dropped and counted, silently."""
    import warnings

    points = {0: [(0.0, 1.0), (0.5, 2.0)], 1: [], 2: [(0.0, 0.5), (0.5, 3.0)]}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = S.spearman_bootstrap(points, analysis_id="x", resamples=200)
    assert res.n_nan > 0 and res.n_seeds == 3 and math.isfinite(res.rho)


def test_the_holm_p_values_are_named_in_the_q_bootstrap_details_text() -> None:
    """Review round 6: Part 5.2 asks for Holm on secondary claims but registers no p-value for a Spearman
    correlation (Part 5.4 registers only a percentile interval). The bootstrap p that H3 (a)'s cells give Holm
    is part of Q-bootstrap-details (answered in Table 9.1), so the key's text names it, beside the Welch p of
    two-arm claims, and so does the code that computes it."""
    text = R.PENDING["Q-bootstrap-details"]
    assert "Welch p" in text and "H3 (a)" in text and "B_defined" in text
    assert "min(1, 2 min(#rho* <= 0, #rho* >= 0) / B_defined)" in S.spearman_bootstrap.__doc__
    assert "Q-bootstrap-details" in S.spearman_bootstrap.__doc__


def test_spearman_bootstrap_equals_spearman_on_each_expanded_resample() -> None:
    # every resample draws whole seeds; its rho is scipy's on the expanded sample (average ranks, ties)
    points = {s: [(n, round((s * 7 + i * 3) % 5 + 0.5 * n, 1)) for i, n in enumerate(R.ONSET_FRACTIONS)]
              for s in R.SEEDS}
    points[3].pop()  # a seed with fewer points
    B = 300
    res_samples = []
    rng = np.random.default_rng(S.generator_seed("expand"))
    idx = rng.integers(0, len(R.SEEDS), size=(B, len(R.SEEDS)))
    for row in idx:
        drawn = [p for i in row for p in points[R.SEEDS[i]]]
        x, y = [p[0] for p in drawn], [p[1] for p in drawn]
        res_samples.append(np.nan if len(set(x)) < 2 or len(set(y)) < 2 else st.spearmanr(x, y).statistic)
    expected = np.asarray(res_samples)
    lo, hi, n_nan = S.percentile_interval(expected, R.INTERVAL_LEVEL)
    ours = S.spearman_bootstrap(points, analysis_id="expand", resamples=B)
    assert ours.n_nan == n_nan and ours.lo == pytest.approx(lo, abs=1e-12) and ours.hi == pytest.approx(hi, abs=1e-12)


def test_spearman_ties_values_that_differ_only_by_floating_point_noise() -> None:
    """Final review round 3: 25.31 - 25.01 and 30.30 - 30.00 are the same gap (0.3) but unequal floats; ranked raw,
    the noise broke Part 5.4's average-rank tie and moved rho and its seed-resampling interval."""
    g1, g2 = 25.31 - 25.01, 30.30 - 30.00
    assert g1 != g2 and S.settled(g1) == S.settled(g2)
    noisy = {0: [(0.0, g1), (0.5, 1.0)], 1: [(0.0, g2), (0.5, 0.3)], 2: [(0.0, 0.1), (0.5, 0.8)]}
    exact = {0: [(0.0, 0.3), (0.5, 1.0)], 1: [(0.0, 0.3), (0.5, 0.3)], 2: [(0.0, 0.1), (0.5, 0.8)]}
    a = S.spearman_bootstrap(noisy, analysis_id="ties")
    b = S.spearman_bootstrap(exact, analysis_id="ties")
    assert (a.rho, a.lo, a.hi, a.p_boot, a.n_nan) == (b.rho, b.lo, b.hi, b.p_boot, b.n_nan)
    assert S.spearman([g1, g2, 0.1], [1.0, 2.0, 3.0]) == S.spearman([0.3, 0.3, 0.1], [1.0, 2.0, 3.0])


def test_spearman_bootstrap_counts_undefined_resamples() -> None:
    points = {0: [(0.0, 1.0)], 1: [(0.5, 2.0)], 2: [(0.5, 3.0)]}  # a resample of one seed has one point
    res = S.spearman_bootstrap(points, analysis_id="few")
    assert res.n_nan > 0 and res.flagged


# ---------------------------------------------------------------------------
# Part 5.2 Holm; Part 5.5 power; equation (12)
# ---------------------------------------------------------------------------


def test_holm_against_the_closed_form() -> None:
    res = S.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.005})
    # sorted: d 0.005*4 = 0.02; a 0.01*3 = 0.03; c 0.03*2 = 0.06; b max(0.06, 0.04*1) = 0.06
    assert {k: round(v.adjusted, 10) for k, v in res.items()} == {"d": 0.02, "a": 0.03, "c": 0.06, "b": 0.06}
    assert {k for k, v in res.items() if v.reject} == {"d", "a"}
    shrunk = S.holm({"a": 0.02, "b": None, "c": math.nan})  # the family shrinks to the cells tested
    assert set(shrunk) == {"a"} and shrunk["a"].adjusted == 0.02 and shrunk["a"].family_size == 1
    capped = S.holm({"a": 0.6, "b": 0.7})
    assert capped["a"].adjusted == 1.0 and capped["b"].adjusted == 1.0 and not capped["b"].reject


def test_power_against_the_registered_numbers_and_how_each_is_rounded() -> None:
    for d, registered in R.POWER_REGISTERED_N5.items():  # all three to the printed precision
        assert round(S.power_two_sample(d, 5), S.POWER_PRINTED_DECIMALS) == registered
    for n, registered in R.POWER_REGISTERED_D80.items():
        computed = S.detectable_d(n)
        assert abs(computed - registered) < 0.01
        assert round(S.power_two_sample(registered, n), S.POWER_PRINTED_DECIMALS) == R.POWER_TARGET
        # five, eight and twelve seeds round to nearest as printed; ten does not (reported to the integrator)
        assert (round(computed, 2) == registered) == (n != 10)
    # recorded fact: the exact 80-percent point at ten seeds is 1.3249, which rounds to nearest to 1.32; the
    # registered 1.33 is it rounded up (power at 1.33 is 0.803): no one rounding rule gives all four values
    assert S.detectable_d(10) == pytest.approx(1.32495, abs=1e-4)
    assert math.ceil(S.detectable_d(10) * 100) / 100 == R.POWER_REGISTERED_D80[10]
    assert math.ceil(S.detectable_d(5) * 100) / 100 != R.POWER_REGISTERED_D80[5]
    rows = {(r["what"], r.get("n")): r for r in report.power_rows(data.build_dataset([], data.Supplement(), mode="final"))}
    assert rows[("registered 80-percent point", 10)]["registered_is"] == "rounded up"
    assert all(r["registered_is"] == "rounded to nearest" for k, r in rows.items() if k[0].startswith("registered")
               and k != ("registered 80-percent point", 10))
    assert S.detectable_d(5, R.ALPHA_PRIMARY_STUDY_B) == pytest.approx(2.3278, abs=1e-3)
    assert S.power_two_sample(S.detectable_d(8), 8) == pytest.approx(R.POWER_TARGET)


def test_power_has_no_nan_where_scipys_lower_tail_has_one() -> None:
    """Review round 3: scipy 1.15.3's nct.cdf(-crit, df, nc) is NaN for large nc (n = 2, alpha = 0.025, d = 8), which
    stopped detectable_d; the lower tail is computed as nct.sf(crit, df, -nc), equal wherever both are finite."""
    crit = float(st.t.ppf(1 - 0.025 / 2, 2))
    assert math.isnan(float(st.nct.cdf(-crit, 2, 8.0)))  # the scipy behaviour this guards against
    nan_points = 0
    for n in (2, 3, 5, 12):
        df = 2 * n - 2
        for alpha in (R.ALPHA, R.ALPHA_PRIMARY_STUDY_B):
            crit = float(st.t.ppf(1 - alpha / 2, df))
            for d in np.linspace(0.0, 64.0, 257):
                power = S.power_two_sample(float(d), n, alpha)
                assert math.isfinite(power) and 0.0 <= power <= 1.0 + 1e-12
                nc = d * math.sqrt(n / 2)
                lower = float(st.nct.cdf(-crit, df, nc))
                if math.isnan(lower):
                    nan_points += 1
                else:
                    assert power == pytest.approx(float(st.nct.sf(crit, df, nc)) + lower, abs=1e-15)
    assert nan_points > 0  # the grid covers points where the old formula was NaN
    for n in range(2, 13):
        for alpha in (R.ALPHA, R.ALPHA_PRIMARY_STUDY_B):
            d80 = S.detectable_d(n, alpha)
            assert d80 is not None and S.power_two_sample(d80, n, alpha) == pytest.approx(R.POWER_TARGET)
    with pytest.raises(S.StatsError):
        S.detectable_d(1)


def test_the_detectable_size_is_none_when_it_cannot_be_found(monkeypatch) -> None:
    """Part 5.5's detectable size is an annotation: when the root finder cannot give it, it is None, never an error."""
    S.detectable_d.cache_clear()
    try:
        monkeypatch.setattr(S, "power_two_sample", lambda d, n, alpha=R.ALPHA: math.nan)
        assert S.detectable_d(4, 0.02) is None  # brentq refuses a NaN
        monkeypatch.setattr(S, "power_two_sample", lambda d, n, alpha=R.ALPHA: 0.1)
        assert S.detectable_d(4, 0.03) is None  # never reaches 80 percent below d = 64
    finally:
        S.detectable_d.cache_clear()


def test_two_seeds_per_arm_at_the_primary_study_b_level() -> None:
    """Review round 3: an arm with two completed seeds (a pilot run crashed, its replacement not done) is analysed."""
    x, y = {0: 1.0, 1: 3.0}, {0: 0.0, 1: 0.5}
    est = S.estimate_two_arms(x, y, analysis_id="two-seeds", level=R.INTERVAL_LEVEL_PRIMARY_STUDY_B)
    assert est.detectable_d == pytest.approx(S.detectable_d(2, R.ALPHA_PRIMARY_STUDY_B))
    assert est.detectable_d > S.detectable_d(2) > S.detectable_d(3)
    rows, _ = syn.b_world(seeds=(0, 1))
    ds = data.build_dataset(rows, data.Supplement(), mode="final")
    power = [r for r in report.power_rows(ds) if r["what"].startswith("80-percent point at the seed count used")]
    assert {(r["n"], r["alpha"]) for r in power} == {(2, R.ALPHA), (2, R.ALPHA_PRIMARY_STUDY_B)}
    assert all(r["computed"] is not None and "note" not in r for r in power)


def test_equation_12() -> None:
    from pilot import go_decision

    assert round(S.u80_t_quantile(), 3) == R.U80_T_QUANTILE
    diffs = [7.0, 5.0, 6.5]
    assert S.u80(diffs) == go_decision.u80(diffs)
    with pytest.raises(S.StatsError):
        S.u80([1.0, 2.0])
    late, ref = [8.0, 7.0, 9.0], [1.0, 2.0, 3.0]
    s = math.sqrt((statistics.variance(ref) + statistics.variance(late)) / 2)
    assert S.u80_pooled(late, ref) == statistics.fmean(late) - statistics.fmean(ref) + R.U80_T_QUANTILE * s / math.sqrt(len(R.PILOT_SEEDS))


def test_equation_12_pairs_by_seed_then_in_seed_order() -> None:
    """Q-g3-pairing (answered in Table 9.1): the per-seed differences are paired by seed id; a replacement seed that
    leaves fewer than three seed-matched pairs has the runs left unpaired paired in increasing seed order (each arm's
    remaining runs sorted by seed, then matched one to one), and equation (12) is applied unchanged."""
    assert S.g3_pairs({0: 8.0, 1: 7.0, 2: 9.0}, {2: 3.0, 0: 1.0, 1: 2.0}) == [(0, 0), (1, 1), (2, 2)]
    # the late arm's seed 1 crashed and was replaced by seed 6; the reference's seed 2 by seed 5
    assert S.g3_pairs({0: 8.0, 2: 9.0, 6: 7.0}, {0: 1.0, 1: 2.0, 5: 3.0}) == [(0, 0), (2, 1), (6, 5)]
    assert S.g3_pairs({0: 8.0, 7: 9.0, 6: 7.0}, {0: 1.0, 5: 2.0, 9: 3.0}) == [(0, 0), (6, 5), (7, 9)]
    assert S.g3_pairs({0: 1.0, 1: 2.0}, {0: 1.0, 1: 1.0, 2: 1.0}) == [(0, 0), (1, 1)]  # a run left over: unpaired


def test_the_two_arm_estimate_of_part_5_3() -> None:
    est = S.estimate_two_arms(X, Y, analysis_id="E2")
    assert est.diff == pytest.approx(S.mean_difference(X, Y)) and est.welch.level == R.INTERVAL_LEVEL
    assert est.bootstrap.resamples == R.BOOTSTRAP_RESAMPLES and est.paired_seeds == (0, 1, 2, 3, 4)
    assert est.detectable_d == pytest.approx(S.detectable_d(5))
    assert est.below_detectable == (abs(est.cohens_d) < est.detectable_d)
    assert est.interval("welch") == (est.welch.lo, est.welch.hi)
    assert est.interval("percentile") == (est.bootstrap.lo, est.bootstrap.hi)
    with pytest.raises(ValueError):
        est.interval("other")
    row = est.row()
    assert row["welch_p"] == est.welch.p and row["boot_mode"] == "joint" and row["n_x"] == 5
    primary = S.estimate_two_arms(X, Y, analysis_id="E2", level=R.INTERVAL_LEVEL_PRIMARY_STUDY_B)
    assert primary.welch.hi > est.welch.hi  # the 97.5 percent interval is wider
    assert S.interval_excludes_zero(0.1, 2.0) and not S.interval_excludes_zero(-0.1, 2.0)
    assert S.interval_excludes_zero(None, 1.0) is None
    assert S.to_plain({"a": (np.float64(1.5), math.nan), "b": np.int64(3), "c": np.bool_(True)}) == \
        {"a": [1.5, None], "b": 3, "c": True}


def test_the_counting_trimmed_mean_is_scipys() -> None:
    rng = np.random.default_rng(3)
    for values in (rng.integers(0, 60, size=(6, 100)).astype(float), rng.normal(size=(6, 100)),
                   (rng.random((6, 100)) < 0.3).astype(float)):
        distinct, codes = np.unique(values, return_inverse=True)
        rows = codes.reshape(-1)[rng.integers(0, values.size, size=(50, values.size))]
        np.testing.assert_allclose(S._trim_mean_rows(rows, distinct, R.IQM_TRIM),
                                   st.trim_mean(distinct[rows], R.IQM_TRIM, axis=1), rtol=0, atol=1e-12)


def test_the_iqm_of_a_two_arm_contrast() -> None:
    rng = np.random.default_rng(4)
    arm = {s: [(1.0, list(rng.integers(20, 40, 100).astype(float))), (-1.0, list(rng.integers(10, 30, 100).astype(float)))]
           for s in range(5)}
    ref = {s: [(1.0, list(rng.integers(5, 15, 100).astype(float))), (-1.0, list(rng.integers(0, 10, 100).astype(float)))]
           for s in range(5)}
    out = S.iqm_contrast(arm, ref, analysis_id="c", resamples=1_000)
    pooled = lambda parts, k: [v for s in sorted(parts) for v in parts[s][k][1]]  # noqa: E731
    assert out["iqm_x"] == pytest.approx(S.iqm(pooled(arm, 0)) - S.iqm(pooled(arm, 1)))
    assert out["iqm_y"] == pytest.approx(S.iqm(pooled(ref, 0)) - S.iqm(pooled(ref, 1)))
    assert out["iqm_diff"] == pytest.approx(out["iqm_x"] - out["iqm_y"]) and out["iqm_lo"] <= out["iqm_diff"] <= out["iqm_hi"]
    assert S.iqm_contrast(arm, ref, analysis_id="c", resamples=1_000) == out  # recorded generator: reproducible
    within = S.iqm_contrast(arm, None, analysis_id="w", resamples=500)
    assert within["iqm_y"] is None and within["iqm_diff"] == pytest.approx(out["iqm_x"])
    with pytest.raises(S.StatsError):
        S.iqm_contrast({0: [(1.0, [1.0] * 100)], 1: [(-1.0, [1.0] * 100)]}, None, analysis_id="bad")


def test_comparisons_with_a_margin_are_made_on_settled_values() -> None:
    assert statistics.fmean([0.55, 0.60, 0.50, 0.57, 0.53]) - statistics.fmean([0.50, 0.52, 0.48, 0.49, 0.51]) > 0.05
    assert S.settled(statistics.fmean([0.55, 0.60, 0.50, 0.57, 0.53]) - statistics.fmean([0.50, 0.52, 0.48, 0.49, 0.51])) == 0.05
    # the finest real separation from a margin (twelve seeds per arm, four budgets) survives the rounding
    assert S.settled(0.05 + 1 / (400 * 12 * 12)) > 0.05 and S.settled(0.05 - 1 / (400 * 12 * 12)) < 0.05
    rel = S.relevel(S.welch({0: 1.0, 1: 2.0, 2: 4.0}, {0: 0.0, 1: 0.5, 2: 0.2}, R.INTERVAL_LEVEL_PRIMARY_STUDY_B), R.INTERVAL_LEVEL)
    direct = S.welch({0: 1.0, 1: 2.0, 2: 4.0}, {0: 0.0, 1: 0.5, 2: 0.2}, R.INTERVAL_LEVEL)
    assert (rel.lo, rel.hi) == pytest.approx((direct.lo, direct.hi))


def test_the_power_rows_take_the_seed_count_from_the_registry(monkeypatch) -> None:
    """Final review (BRIEF rule 4): the power at five seeds reads the registered seed count (R.SEEDS), not a retyped 5."""
    empty = data.build_dataset([], data.Supplement(), mode="final")
    five = [r for r in report.power_rows(empty) if r["what"] == "registered power at five seeds"]
    assert five and all(r["n"] == len(R.SEEDS) == 5 for r in five)
    monkeypatch.setattr(R, "SEEDS", (0, 1, 2))  # a changed registry changes the row, so nothing is retyped
    rows = [r for r in report.power_rows(empty) if r["what"] == "registered power at five seeds"]
    assert all(r["n"] == 3 and r["computed"] == S.power_two_sample(r["d"], 3) for r in rows)

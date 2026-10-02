"""Study B: Part 4.2 and boxes G1 to G5 on synthetic datasets (analysis/study_b.py)."""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import Callable, Optional

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import test_analysis_synthetic as syn  # noqa: E402

from analysis import data, stats, study_b  # noqa: E402
from analysis import verdict as V  # noqa: E402
from configs import registered as R  # noqa: E402
from results import supplement_schema as S  # noqa: E402


@pytest.fixture
def keys_open(monkeypatch):
    """Every PENDING key open (configs.registered.ANSWERED_QUESTIONS empty): the test reads the other readings that
    analysis.verdict computes only while a key is open, whatever the amendment log has answered since."""
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset())


@pytest.fixture(autouse=True)
def _fewer_resamples(monkeypatch):
    """The verdict logic does not depend on the resample count (the statistics tests use 10,000)."""
    monkeypatch.setattr(R, "BOOTSTRAP_RESAMPLES", 2_000)


def run(kind: str = "supported", sr: Optional[Callable[[str, float, int], float]] = None, *,
        with_records: bool = True, rows_hook: Optional[Callable[[list], None]] = None, iqm: bool = False
        ) -> tuple[dict[str, V.Verdict], study_b.Catalogue, data.Dataset]:
    """Build a final-mode dataset from ``syn.b_world`` and analyse it: (verdicts by id, catalogue, dataset)."""
    rows, records = syn.b_world(kind, sr=sr)
    if rows_hook is not None:
        rows_hook(rows)
    sup = data.Supplement()
    if with_records:
        for run_id, recs in records.items():
            for kind_name, record in [("zero_shot", recs["zero_shot"])] + [("fewshot", f) for f in recs["fewshot"]]:
                model = S.validate_record(kind_name, record)
                sup.records.setdefault(kind_name, {})[(run_id, S.record_part(kind_name, model))] = model
    ds = data.build_dataset(rows, sup, mode="final")
    cat = study_b.analyse(ds, iqm=iqm)
    return {v.id: v for v in cat.verdicts}, cat, ds


ARMS = tuple(R.STUDY_B_ARMS)


def sr_table(table: dict[str, float], noise: float = 0.02, by_budget: Optional[dict] = None
             ) -> Callable[[str, float, int], float]:
    """Satisfaction per (arm, budget, seed); each arm has its own seed-noise pattern (mean 0 over seeds 0-4),
    so paired seed resamples are not degenerate."""
    def sr(arm: str, budget: float, seed: int) -> float:
        base = (by_budget or {}).get((arm, budget), table[arm])
        return round(min(1.0, max(0.0, base + noise * syn.noise((seed + ARMS.index(arm)) % len(R.SEEDS)))), 2)
    return sr


# ---------------------------------------------------------------------------
# Equation (11) and Part 4.2
# ---------------------------------------------------------------------------


def test_nearest_single_arms_by_equation_11_match_part_4_2() -> None:
    assert study_b.nearest_single_arms(5.0) == ("Single-10",)
    assert study_b.nearest_single_arms(15.0) == ("Single-10", "Single-20")
    assert study_b.nearest_single_arms(30.0) == ("Single-20", "Single-40")
    assert study_b.nearest_single_arms(45.0) == ("Single-40",)
    comparators = {a for b in R.UNSEEN_BUDGETS for a in study_b.nearest_single_arms(b)}
    assert comparators != {"Single-10"}  # "G2 never compares Dense only with Single-10"


def test_distance_and_the_nearest_training_level() -> None:
    assert study_b.distance(5.0, "Sparse") == 5.0 and study_b.distance(45.0, "Dense") == 5.0
    assert study_b.distance(15.0, "Dense") == 2.5 and study_b.distance(30.0, "Continuous") == 0.0
    for arm in ("Sparse", "Moderate", "Dense", "Continuous"):
        assert study_b.nearest_training_level(arm, 5.0) == 10.0
        assert study_b.nearest_training_level(arm, 45.0) == 40.0
    assert study_b.nearest_training_level("Single-20", 45.0) == 20.0
    assert study_b.nearest_training_level("Continuous", 30.0) == 30.0


def test_the_floor_rule_needs_more_than_half_of_the_arms() -> None:
    four = sr_table({"Single-10": 0.0, "Single-20": 0.0, "Single-40": 0.0, "Sparse": 0.02, "Moderate": 0.5,
                     "Dense": 0.6, "Continuous": 0.6}, noise=0.0)
    _, cat, ds = run(sr=four)
    floor = study_b.StudyB(ds).floor(V.PROPOSED)
    assert all(floor[b]["floored"] and len(floor[b]["below"]) == 4 for b in R.UNSEEN_BUDGETS)
    assert [r["floored"] for r in cat.tables["B_floor"]] == [True] * len(R.UNSEEN_BUDGETS)
    three = sr_table({"Single-10": 0.0, "Single-20": 0.0, "Single-40": 0.0, "Sparse": 0.3, "Moderate": 0.5,
                      "Dense": 0.6, "Continuous": 0.6}, noise=0.0)
    _, _, ds = run(sr=three)
    floor = study_b.StudyB(ds).floor(V.PROPOSED)
    assert not any(floor[b]["floored"] for b in R.UNSEEN_BUDGETS) and len(floor[5.0]["below"]) == 3


def test_the_other_readings_of_the_floor_rule_count_other_arms_or_every_seed() -> None:
    """Q-floor-rule asks "which arms count and which rate"; the rule itself is registered (Part 4.2)
    and applies under every reading."""
    comparison, per_seed, both = V.alternatives("Q-floor-rule")
    # three of seven arms at zero: not floored by the seven arms, but floored when only the comparison's arms count
    three = sr_table({"Single-10": 0.0, "Single-20": 0.0, "Single-40": 0.0, "Sparse": 0.3, "Moderate": 0.5,
                      "Dense": 0.6, "Continuous": 0.6}, noise=0.0)
    _, _, ds = run(sr=three)
    B = study_b.StudyB(ds)
    pair = lambda b: ("Single-10", "Single-20")  # noqa: E731
    assert not B.floor(V.PROPOSED, pair)[5.0]["floored"]  # the proposal counts the seven arms whatever is compared
    assert B.floor(comparison, pair)[5.0]["floored"] is True
    assert B.floor(comparison, pair)[5.0]["counted"] == list(pair(5.0))
    assert not B.floor(comparison, lambda b: ("Dense", "Single-10"))[5.0]["floored"]  # one of two: not more than half
    assert not B.floor(comparison)[5.0]["floored"]  # a comparison that names no arms: the seven count
    # per seed: 15 runs at zero and three of Sparse's five seeds below 5 percent (its mean 0.2 is not): 18 of 35 runs
    def split(rows):
        for row in rows:
            if row["arm"] == "Sparse":
                row["sr_zero"] = {**row["sr_zero"], 5.0: 0.0 if row["seed"] < 3 else 0.5}
    _, _, ds = run(sr=three, with_records=False, rows_hook=split)
    B = study_b.StudyB(ds)
    assert not B.floor(V.PROPOSED)[5.0]["floored"]
    assert B.floor(V.PROPOSED)[5.0]["arm_means"]["Sparse"] == pytest.approx(0.2)
    seeds = B.floor(per_seed)[5.0]
    assert seeds["floored"] is True and seeds["unit"] == "run" and len(seeds["below"]) == 18 and seeds["units"] == 35
    assert not B.floor(per_seed)[15.0]["floored"]  # 15 of 35 runs at the other budgets
    assert B.floor(both, pair)[5.0]["floored"] is True and B.floor(both, pair)[5.0]["units"] == 10


# ---------------------------------------------------------------------------
# G1
# ---------------------------------------------------------------------------


def test_g1_supported_falsified_and_inconclusive() -> None:
    verdicts, _, _ = run("supported")
    g1 = verdicts["G1"]
    assert g1.status == V.SUPPORTED and g1.label == V.CONFIRMATORY
    assert g1.numbers["ordered"] and g1.numbers["trend_rho"] > 0
    assert g1.numbers["trend_level"] == R.INTERVAL_LEVEL_PRIMARY_STUDY_B
    assert g1.numbers["Dense - Sparse_level"] == R.INTERVAL_LEVEL_PRIMARY_STUDY_B  # the 97.5 percent interval
    # Part 5.7 ("If H1 or G1 is falsified, the result is reported as an upper bound"): a supported
    # G1 carries no bound and no Part 5.7 reading ("inconclusive at this sample size" is Part 1.2's for neither)
    bound_keys = ("bound_95_upper_Dense_minus_Sparse", "bound_97_5_upper_Dense_minus_Sparse", "bound_reading")
    assert not any(k in g1.numbers for k in bound_keys) and not any("Part 5.7" in n for n in g1.notes)
    verdicts, _, _ = run("flat")
    g1 = verdicts["G1"]
    assert g1.status == V.FALSIFIED and g1.numbers["bound_95_upper_Dense_minus_Sparse"] is not None
    assert g1.numbers["bound_reading"] in (V.BOUND_BELOW, V.BOUND_ABOVE)
    assert any("Part 5.7" in n for n in g1.notes)
    noisy = sr_table({"Single-10": 0.3, "Single-20": 0.3, "Single-40": 0.3, "Sparse": 0.40, "Moderate": 0.45,
                      "Dense": 0.50, "Continuous": 0.5}, noise=0.2)
    verdicts, _, _ = run(sr=noisy)
    g1 = verdicts["G1"]
    assert g1.numbers["ordered"] and g1.numbers["Dense - Sparse"] == pytest.approx(0.10)
    assert g1.numbers["Dense - Sparse_welch_lo"] < 0
    assert g1.status == V.INCONCLUSIVE
    assert not any(k in g1.numbers for k in bound_keys)  # nor does an inconclusive one


@pytest.mark.usefixtures("keys_open")
def test_g1_leaves_floored_budgets_out_and_says_so() -> None:
    by_budget = {(a, 5.0): 0.0 for a in ("Single-10", "Single-20", "Single-40", "Sparse")}
    table = dict(Sparse=0.40, Moderate=0.55, Dense=0.75, Continuous=0.77, **{f"Single-{k}": 0.3 for k in (10, 20, 40)})
    sr = sr_table(table, by_budget=by_budget)
    verdicts, _, _ = run(sr=sr)
    g1 = verdicts["G1"]
    assert g1.numbers["budgets_floored"] == [5.0] and g1.numbers["budgets_on_satisfaction"] == [15.0, 30.0, 45.0]
    assert any("floor rule applied at budget 5" in n for n in g1.notes)
    assert "VM(Dense) - VM(Sparse) at 5" in g1.numbers
    assert g1.status == V.SUPPORTED and g1.readings["Q-floor-rule"] == V.SUPPORTED  # the proposal and every other reading of Q-floor-rule support here
    everything = sr_table({a: 0.0 for a in R.STUDY_B_ARMS}, noise=0.0)
    verdicts, _, _ = run(sr=everything)
    g1 = verdicts["G1"]
    assert g1.proposal_status == V.NOT_COMPUTABLE and "floor" in g1.reason
    # Every reading of Q-floor-rule floors every budget ("a rate at the floor cannot separate arms"),
    # so no reading reads G1 or G5 on satisfaction: they are not computable, not UNDECIDED(Q-floor-rule)
    assert g1.status == V.NOT_COMPUTABLE and g1.readings["Q-floor-rule"] == V.NOT_COMPUTABLE
    assert verdicts["G5"].status == V.NOT_COMPUTABLE and verdicts["G5"].readings["Q-floor-rule"] == V.NOT_COMPUTABLE
    # Part 4.2: "that budget's comparison is made on violation magnitude ... and the report says so", for G5 as for G1
    g5 = verdicts["G5"]
    assert "violation-magnitude comparisons are reported" in g5.reason
    assert g5.numbers["budgets_floored"] == list(R.UNSEEN_BUDGETS)
    assert all(f"VM(Continuous) - VM(Dense) at {b:g}" in g5.numbers for b in R.UNSEEN_BUDGETS)
    assert sum("floor rule" in n and "violation magnitude" in n for n in g5.notes) == len(R.UNSEEN_BUDGETS)


def test_g1_counts_and_flags_the_undefined_trend_resamples() -> None:
    """Q-bootstrap-details: undefined resamples (a constant variable) "are dropped and counted, and flagged above 1
    percent", as Study A's H1 trend reports them. Every G1 seed is at 1.0 but Dense's seed 0 and Sparse's seed 1, so a
    resample of seeds 2 to 4 alone has a constant outcome (about (3/5)^5 of the resamples)."""
    def sr(arm, budget, seed):
        if (arm, seed) in (("Dense", 0), ("Sparse", 1)):
            return 0.9
        return 1.0 if arm in study_b.G1_ARMS else 0.5
    g1 = run(sr=sr, with_records=False)[0]["G1"]
    n_nan = g1.numbers["trend_undefined_resamples"]
    assert n_nan > stats.NAN_FLAG_SHARE * R.BOOTSTRAP_RESAMPLES
    assert f"{n_nan} of {R.BOOTSTRAP_RESAMPLES} trend resamples were undefined (over 1%)" in g1.notes
    g1 = run("supported", with_records=False)[0]["G1"]
    assert g1.numbers["trend_undefined_resamples"] == 0 and not any("trend resamples" in n for n in g1.notes)


# ---------------------------------------------------------------------------
# G2
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("keys_open")
def test_g2_uses_the_least_favourable_nearest_single_and_reports_both() -> None:
    table = {"Single-10": 0.35, "Single-20": 0.30, "Single-40": 0.32, "Sparse": 0.4, "Moderate": 0.55,
             "Dense": 0.75, "Continuous": 0.77}
    verdicts, _, _ = run(sr=sr_table(table))
    g2 = verdicts["G2"]
    assert g2.status == V.SUPPORTED and g2.label == V.CONFIRMATORY and g2.numbers["budgets_won"] == 4
    assert g2.numbers["b15.comparator"] == "Single-10" and g2.numbers["b30.comparator"] == "Single-40"
    assert g2.numbers["b5.comparator"] == "Single-10" and g2.numbers["b45.comparator"] == "Single-40"
    for budget, pair in ((15, ("Single-10", "Single-20")), (30, ("Single-20", "Single-40"))):
        for single in pair:  # "both are reported"
            assert g2.numbers[f"b{budget}.Dense - {single}"] is not None
            assert g2.numbers[f"b{budget}.Dense - {single}_level"] == R.INTERVAL_LEVEL_PRIMARY_STUDY_B
    assert "b5.holm_adjusted_p" in g2.numbers
    composite = "composite: Dense - comparator, mean over budgets"
    assert g2.numbers[composite] > R.MIN_EFFECT_STUDY_B
    assert g2.numbers[f"{composite}_level"] == R.INTERVAL_LEVEL_PRIMARY_STUDY_B
    assert g2.readings["Q-g2-outcome"] == V.SUPPORTED  # the composite (Part 5.2's outcome) agrees here
    verdicts, _, _ = run("flat")
    assert verdicts["G2"].status == V.FALSIFIED and verdicts["G2"].numbers["budgets_lost"] == 4


def test_g2_breaks_a_comparator_tie_by_the_smaller_lower_limit() -> None:
    def sr(arm, budget, seed):
        if arm == "Single-20":
            return round(0.30 + 0.10 * syn.noise(seed), 2)  # same mean as Single-10, wider interval
        base = {"Single-10": 0.30, "Single-40": 0.30, "Sparse": 0.4, "Moderate": 0.55, "Dense": 0.75,
                "Continuous": 0.77}[arm]
        return round(base + 0.02 * syn.noise(seed), 2)
    verdicts, _, _ = run(sr=sr)
    g2 = verdicts["G2"]
    assert g2.numbers["b15.comparator"] == "Single-20"
    assert g2.numbers["b15.Dense - Single-20_welch_lo"] < g2.numbers["b15.Dense - Single-10_welch_lo"]
    assert any("tie" in n for n in g2.notes)


def test_g2_on_a_floored_budget_compares_violation_magnitude() -> None:
    by_budget = {(a, 5.0): 0.0 for a in ("Single-10", "Single-20", "Single-40", "Sparse")}
    table = dict(Sparse=0.40, Moderate=0.55, Dense=0.75, Continuous=0.77, **{f"Single-{k}": 0.3 for k in (10, 20, 40)})
    verdicts, _, _ = run(sr=sr_table(table, by_budget=by_budget))
    g2 = verdicts["G2"]
    assert g2.numbers["b5.outcome"] == "violation magnitude" and g2.numbers["b5.win"] is True
    assert any("floor rule applied at budget 5" in n for n in g2.notes)


def test_g2_chooses_the_comparator_on_violation_magnitude_on_a_floored_budget() -> None:
    """Q-g2-outcome (answered in Table 9.1): Part 4.2's comparator rule ('whichever of the two has the higher
    satisfaction on that budget, the choice least favourable to G2') meets the floor rule ('a rate at the floor cannot
    separate arms'): on a floored budget the comparator is the one with the lower arm-mean zero-shot violation
    magnitude, the outcome compared. At budget 15, Single-10 has the higher satisfaction (0.04 against 0.00) but
    Single-20 the lower violation magnitude (1.0 against 4.8): Single-20 is chosen, and Dense loses the budget."""
    floor15 = {"Single-10": 0.04, "Single-20": 0.0, "Single-40": 0.0, "Sparse": 0.0}
    table = {"Single-10": 0.30, "Single-20": 0.30, "Single-40": 0.30, "Sparse": 0.40, "Moderate": 0.55, "Dense": 0.75,
             "Continuous": 0.77}

    def sr(arm: str, budget: float, seed: int) -> float:
        if budget == 15.0 and arm in floor15:
            return floor15[arm]
        return round(table[arm] + 0.02 * syn.noise((seed + ARMS.index(arm)) % len(R.SEEDS)), 2)

    rows, records = syn.b_world("supported", sr=sr)
    for recs in records.values():
        if recs["zero_shot"]["arm"] != "Single-20":
            continue
        for block in recs["zero_shot"]["budgets"]:
            if block["budget"] == 15.0:  # every episode one cost unit over the budget: satisfaction 0, violation 1
                costs = [16.0] * R.EVAL_EPISODES
                block.update(episode_costs=costs, mean_cost=16.0, satisfaction=S._satisfaction(costs, 15.0),
                             violation=S._violation(costs, 15.0))
    sup = data.Supplement()
    for run_id, recs in records.items():
        for kind_name, record in [("zero_shot", recs["zero_shot"])] + [("fewshot", f) for f in recs["fewshot"]]:
            model = S.validate_record(kind_name, record)
            sup.records.setdefault(kind_name, {})[(run_id, S.record_part(kind_name, model))] = model
    ds = data.build_dataset(rows, sup, mode="final")
    assert study_b.StudyB(ds).floor(V.PROPOSED)[15.0]["floored"] is True
    g2 = {v.id: v for v in study_b.analyse(ds, iqm=False).verdicts}["G2"]
    assert g2.numbers["b15.outcome"] == "violation magnitude" and g2.numbers["b15.comparator"] == "Single-20"
    assert g2.numbers["b15.comparator_mean_violation"] == pytest.approx({"Single-10": 4.8, "Single-20": 1.0})
    assert "b15.comparator_mean_satisfaction" not in g2.numbers
    assert any("budget 15: comparator chosen on violation magnitude" in n for n in g2.notes)
    # both comparators are reported on the outcome compared; Single-20 is the harder comparator for Dense there
    assert g2.numbers["b15.Dense - Single-10"] < 0 < g2.numbers["b15.Dense - Single-20"]
    assert g2.numbers["b15.win"] is False and g2.numbers["b15.lose"] is True
    # an unfloored budget still chooses on satisfaction
    assert g2.numbers["b30.outcome"] == "satisfaction" and "b30.comparator_mean_satisfaction" in g2.numbers


# ---------------------------------------------------------------------------
# G3, G4, G5
# ---------------------------------------------------------------------------


def test_g3_supported_falsified_and_the_proportion_sentence() -> None:
    table = {a: 0.5 for a in R.STUDY_B_ARMS}
    by_budget = {**{(a, 5.0): 0.2 for a in R.STUDY_B_ARMS}, **{(a, 45.0): 0.8 for a in R.STUDY_B_ARMS}}
    verdicts, _, _ = run(sr=sr_table(table, by_budget=by_budget))
    g3 = verdicts["G3"]
    assert g3.status == V.SUPPORTED, (g3.reason, g3.numbers)
    assert g3.numbers["Dense.D"] == pytest.approx(0.6) and g3.numbers["Continuous.reference_levels"] == (10.0, 40.0)
    sentence = study_b.g3_proportion_sentence()
    assert sentence in g3.notes and "0.5 of the lowest" in sentence and "0.125 of the highest" in sentence
    verdicts, _, _ = run("flat")
    assert verdicts["G3"].status == V.FALSIFIED and sentence in verdicts["G3"].notes
    verdicts, _, _ = run("supported", with_records=False)
    assert verdicts["G3"].status == V.NOT_COMPUTABLE and "reference levels 10/40" in verdicts["G3"].reason


def test_g4_supported_and_falsified() -> None:
    verdicts, _, _ = run("supported")
    g4 = verdicts["G4"]
    assert g4.status == V.SUPPORTED and g4.numbers["b5.median_index(Sparse)"] == len(R.FEWSHOT_HORIZONS) + 1
    assert g4.numbers["b5.median_index(Dense)"] == 1 and g4.numbers["budgets_ordered"] == 4
    verdicts, _, _ = run("flat")
    assert verdicts["G4"].status == V.FALSIFIED and verdicts["G4"].numbers["budgets_within"] == 4


@pytest.mark.usefixtures("keys_open")
def test_g4_after_the_part_6_1_cut_to_one_horizon() -> None:
    horizon = R.FEWSHOT_HORIZONS[0]

    def cut(rows):
        for row in rows:
            if row["sr_fewshot"] is None:
                continue
            reached = row["arm"] == "Dense"
            row["sr_fewshot"] = {S.fewshot_key(b, horizon): 0.9 if reached else 0.2 for b in R.UNSEEN_BUDGETS}
            row["adapt_steps"] = {b: horizon if reached else horizon + 1 for b in R.UNSEEN_BUDGETS}  # 200,001
    verdicts, _, ds = run(with_records=False, rows_hook=cut)
    sparse = next(r for r in ds.study_b if r["arm"] == "Sparse")["fewshot"][5.0]
    assert sparse["horizons"] == [horizon] and sparse["censored"] is True and sparse["index"] == 2
    assert sparse["consistent"] is True
    g4 = verdicts["G4"]
    assert g4.numbers["b5.median_index(Moderate)"] == g4.numbers["b5.median_index(Sparse)"] == 2
    # dense one horizon below sparse, but moderate ties sparse: the arm statistic (1, 2, 2) is not ordered, and a
    # spread of exactly one is not 'within one horizon' (below one): INCONCLUSIVE under the decided reading of
    # Q-g4-reading. With one horizon the readings 'within one horizon' = a difference of at most one, and 'censored
    # at the largest horizon', decide G4 by construction (it used to come out FALSIFIED whatever the data): they are
    # NOT_COMPUTABLE; the per-budget ordering notes the binary indices
    assert g4.proposal_status == V.INCONCLUSIVE and g4.readings["Q-g4-reading"] == V.NOT_COMPUTABLE
    assert g4.numbers["arm_medians_ordered"] is False and g4.numbers["budgets_within"] == 0
    assert g4.status == V.UNDECIDED and g4.undecided_by == ("Q-g4-reading",)
    assert not any("cannot hold" in n for n in g4.notes)
    for i in (0, 2):  # g4_within='at_most_one', g4_censored='at'
        alt = study_b.g4_outcome(study_b.StudyB(ds), V.alternative("Q-g4-reading", i))
        assert alt.status == V.NOT_COMPUTABLE and "one horizon" in alt.reason
    per_budget = study_b.g4_outcome(study_b.StudyB(ds), V.alternative("Q-g4-reading", 1))
    assert per_budget.status == V.INCONCLUSIVE
    assert any("one horizon (Part 6.1 cut 3)" in n and "cannot hold" in n for n in per_budget.notes)


def test_g4_refuses_adaptation_steps_that_do_not_follow_from_the_rates() -> None:
    def wrong(rows):
        for row in rows:
            if row["arm"] == "Dense":
                row["adapt_steps"] = {b: R.FEWSHOT_HORIZONS[1] for b in R.UNSEEN_BUDGETS}  # rates reach at the first
    verdicts, _, ds = run(with_records=False, rows_hook=wrong)
    assert verdicts["G4"].status == V.NOT_COMPUTABLE
    assert any("does not follow from sr_fewshot" in p for p in ds.problems)


def _fewshot_floor_world(floored_budgets) -> data.Dataset:
    """Every arm's few-shot rate is 0 at every horizon on ``floored_budgets`` (all censored: 'within one horizon')
    and reaches the target at Dense < Moderate < Sparse elsewhere; zero-shot 0.5 (no zero-shot floor)."""
    first = {"Dense": 0, "Moderate": 1, "Sparse": 2}
    rows, sup = [], data.Supplement()
    for spec in syn.specs(groups=("study_b",)):
        few, adapt = {}, {}
        for b in R.UNSEEN_BUDGETS:
            k = first.get(spec.arm, 0)
            rates = {h: 0.0 if b in floored_budgets else (0.9 if i >= k else 0.2)
                     for i, h in enumerate(R.FEWSHOT_HORIZONS)}
            few.update({S.fewshot_key(b, h): v for h, v in rates.items()})
            reached = [h for h in R.FEWSHOT_HORIZONS if rates[h] >= R.SATISFACTION_TARGET]
            adapt[b] = reached[0] if reached else max(R.FEWSHOT_HORIZONS) + 1
            model = S.validate_record("fewshot", syn.fewshot_record(spec, b, rates, adapt[b]))
            sup.records.setdefault("fewshot", {})[(spec.run_id, S.record_part("fewshot", model))] = model
        rows.append(syn.study_b_row(spec, sr_zero=dict.fromkeys(R.UNSEEN_BUDGETS, 0.5), sr_fewshot=few,
                                    adapt_steps=adapt))
    return data.build_dataset(rows, sup, mode="final")


def test_g4_applies_the_floor_rule_to_the_few_shot_rates() -> None:
    """Part 4.2's floor rule is "for every comparison above" (G4 included): "a rate at the floor
    cannot separate arms". Budgets where every arm's few-shot rate is at the floor at every horizon (all censored)
    used to count as 'within one horizon' and falsify G4; they now leave G4's majority and are compared on the
    few-shot violation magnitude (Q-floor-rule's proposal)."""
    ds = _fewshot_floor_world((5.0, 15.0, 45.0))
    B = study_b.StudyB(ds)
    floor = B.fewshot_floor(V.PROPOSED, study_b.G4_ARMS)
    assert [b for b in R.UNSEEN_BUDGETS if floor[b]["floored"]] == [5.0, 15.0, 45.0]
    out = study_b.g4_outcome(B, V.PROPOSED)
    assert out.status == V.SUPPORTED, (out.reason, out.numbers)  # budget 30 alone, ordered: the majority of one
    assert out.numbers["budgets_floored"] == [5.0, 15.0, 45.0] and out.numbers["budgets_within"] == 0
    assert out.numbers["budgets_on_adaptation_steps"] == [30.0] and "b5.within_one_horizon" not in out.numbers
    assert any("floor rule" in n and "few-shot violation magnitude" in n and "budget 5" in n for n in out.notes)
    assert "VM_fewshot(Dense) - VM_fewshot(Sparse) at 5" in out.numbers
    every = study_b.g4_outcome(study_b.StudyB(_fewshot_floor_world(tuple(R.UNSEEN_BUDGETS))), V.PROPOSED)
    assert every.status == V.NOT_COMPUTABLE and "few-shot floor" in every.reason
    # an arm without a few-shot rate (the proposal counts all seven arms): the floor is undecided, never guessed
    for rec in ds.study_b:
        if rec["arm"] == "Continuous":
            rec["fewshot"] = {}
    B = study_b.StudyB(ds)
    floor = B.fewshot_floor(V.PROPOSED, study_b.G4_ARMS)
    assert floor[30.0]["floored"] is False  # 6 of 7 known off the floor at every horizon (4 suffice)
    assert floor[5.0]["floored"] is True  # 6 of 7 known below: more than half whatever Continuous's rate


def test_the_fewshot_table_floors_as_g4_does() -> None:
    """Table B_fewshot counted only the arms with few-shot data, so with 4 of 7 arms still
    without it the table showed budget 5 floored at every horizon while G4 left it undecided (Q-floor-rule's
    proposal counts all seven arms, unknown ones included)."""
    ds = _fewshot_floor_world((5.0,))
    for rec in ds.study_b:
        if rec["arm"] not in ("Dense", "Moderate", "Sparse"):
            rec["fewshot"] = {}
    B = study_b.StudyB(ds)
    floor = B.fewshot_floor(V.PROPOSED, study_b.G4_ARMS)
    assert floor[5.0]["floored"] is None
    table = study_b._tables(B)["B_fewshot"]
    assert {r["floored"] for r in table if r["budget"] == 5.0} == {None}
    for r in table:
        assert r["floored"] == floor[r["budget"]]["by_horizon"][r["horizon"]]["floored"]
    # with every arm's data the table and G4 agree on a decided floor
    B = study_b.StudyB(_fewshot_floor_world((5.0,)))
    table = study_b._tables(B)["B_fewshot"]
    assert {r["floored"] for r in table if r["budget"] == 5.0} == {True}
    assert {r["floored"] for r in table if r["budget"] == 30.0} == {False}


def test_g5_supported_falsified_and_inconclusive() -> None:
    verdicts, _, _ = run("supported")
    assert verdicts["G5"].status == V.SUPPORTED and verdicts["G5"].numbers["Continuous - Dense"] == pytest.approx(0.02)
    table = {"Single-10": 0.3, "Single-20": 0.3, "Single-40": 0.3, "Sparse": 0.4, "Moderate": 0.55, "Dense": 0.75}
    verdicts, _, _ = run(sr=sr_table({**table, "Continuous": 0.95}))
    assert verdicts["G5"].status == V.FALSIFIED  # "discrete coverage, however dense, does not substitute for the range"
    verdicts, _, _ = run(sr=sr_table({**table, "Continuous": 0.83}, noise=0.1))
    g5 = verdicts["G5"]
    assert g5.status == V.INCONCLUSIVE and g5.numbers["Continuous - Dense"] > R.G5_POINT_MAX


def test_per_budget_claims_take_holm_over_the_four_budgets() -> None:
    verdicts, cat, _ = run("supported")
    claim = verdicts["B-satisfaction[Dense-Sparse/b5]"]
    assert claim.status == V.SUPPORTED and claim.numbers["holm_family_size"] == 4 and claim.label == V.SECONDARY
    vm = verdicts["B-violation[Dense-Sparse/b5]"]
    assert vm.numbers["outcome"] == "violation magnitude" and vm.status == V.SUPPORTED
    assert not any(v.status == V.FALSIFIED for k, v in verdicts.items() if k.startswith("B-"))
    outcomes = cat.tables["B_outcomes"]
    row = next(r for r in outcomes if r["arm"] == "Dense" and r["budget"] == 45.0)
    assert row["distance_eq11"] == 5.0 and row["extrapolation"] is True and row["nearest_training_level"] == 40.0
    assert next(r for r in outcomes if r["budget"] == 15.0)["extrapolation"] is False
    few = [r for r in cat.tables["B_fewshot"] if r["arm"] == "Sparse"]
    assert few and all(r["mean_sr"] == 0.2 for r in few)
    assert cat.tables["B_estimates"]


def test_g2_and_g3_inconclusive() -> None:
    def sr(arm, budget, seed):
        e = syn.noise((seed + ARMS.index(arm)) % len(R.SEEDS))
        if arm == "Dense":
            spread = 0.15 if budget == 15.0 else 0.02
            return round({5.0: 0.75, 15.0: 0.45, 30.0: 0.36, 45.0: 0.75}[budget] + spread * e, 2)
        if arm.startswith("Single"):
            return round(0.33 + 0.02 * e, 2)
        return round({"Sparse": 0.4, "Moderate": 0.55, "Continuous": 0.77}[arm] + 0.02 * e, 2)
    verdicts, _, _ = run(sr=sr)
    g2 = verdicts["G2"]
    # wins at 5 and 45; at 15 Dense leads by 0.12 with an interval including zero; at 30 by 0.03 (a loss)
    assert (g2.numbers["b5.win"], g2.numbers["b45.win"], g2.numbers["b15.win"], g2.numbers["b15.lose"],
            g2.numbers["b30.lose"]) == (True, True, False, False, True)
    assert g2.proposal_status == V.INCONCLUSIVE
    table = {a: 0.5 for a in R.STUDY_B_ARMS}
    by_budget = {**{(a, 5.0): 0.2 for a in R.STUDY_B_ARMS}, **{(a, 45.0): 0.8 for a in R.STUDY_B_ARMS}}
    by_budget.update({("Sparse", 5.0): 0.78, ("Moderate", 5.0): 0.78})  # Sparse, Moderate: drops within 5 points
    verdicts, _, _ = run(sr=sr_table(table, by_budget=by_budget))
    g3 = verdicts["G3"]  # two of four arms within the margin: not a majority; two support: not every arm
    assert g3.numbers["Sparse.within_margin"] and g3.numbers["Dense.supports"] and not g3.numbers["Sparse.supports"]
    assert g3.proposal_status == V.INCONCLUSIVE
    # Part 5.7's bound belongs to a falsified box only; an inconclusive G3 carries none
    assert not any(k.startswith("bound") for k in g3.numbers) and "bounded_arms" not in g3.numbers
    assert not any(v == V.BOUND_BELOW for v in g3.numbers.values() if isinstance(v, str))


# ---------------------------------------------------------------------------
# Exact margins, the floor rule with a missing rate, the other readings, annotations
# ---------------------------------------------------------------------------


def test_a_difference_of_exactly_five_points_does_not_exceed_five_points() -> None:
    dense_dev = {0: 0.0, 1: 0.01, 2: -0.01, 3: 0.01, 4: -0.01}
    single_dev = {0: 0.0, 1: 0.01, 2: -0.01, 3: -0.01, 4: 0.01}

    def sr(arm, budget, seed):
        if arm == "Dense":
            return round(0.55 + dense_dev[seed], 2)
        if arm.startswith("Single"):
            return round(0.50 + single_dev[seed], 2)
        return {"Sparse": 0.40, "Moderate": 0.45, "Continuous": 0.55}[arm]
    verdicts, _, _ = run(sr=sr)
    g2 = verdicts["G2"]
    assert g2.numbers["b5.Dense - Single-10"] > R.MIN_EFFECT_STUDY_B  # 0.050000000000000044 in floating point
    assert stats.settled(g2.numbers["b5.Dense - Single-10"]) == R.MIN_EFFECT_STUDY_B
    for budget in R.UNSEEN_BUDGETS:  # "within 5 percentage points": a loss, never a win
        assert g2.numbers[f"b{budget:g}.win"] is False and g2.numbers[f"b{budget:g}.lose"] is True
    # the box conditions falsify G2; its losing budgets' 95 percent upper limits are above 5 points, so the
    # falsification is not confirmed (Q-falsification-calibration): inconclusive at this sample size
    assert g2.numbers["falsification"] is True and g2.numbers["bounded_losing_budgets"] == 0
    assert g2.proposal_status == V.INCONCLUSIVE
    assert stats.settled(0.1 + 0.2) == stats.settled(0.3) and stats.settled(0.050017) > 0.05  # 1e-9 keeps real gaps


@pytest.mark.usefixtures("keys_open")
def test_the_floor_rule_never_drops_an_arm_whose_rate_is_missing() -> None:
    by_budget = {(a, 45.0): 0.02 for a in ("Single-10", "Single-20", "Single-40", "Sparse")}
    table = dict(Sparse=0.40, Moderate=0.55, Dense=0.75, Continuous=0.77, **{f"Single-{k}": 0.3 for k in (10, 20, 40)})
    sr = sr_table(table, noise=0.0, by_budget=by_budget)
    _, _, ds = run(sr=sr, with_records=False)
    assert study_b.StudyB(ds).floor(V.PROPOSED)[45.0]["floored"] is True  # four of seven below 5 percent
    missing = {}

    def drop(rows):
        row = next(r for r in rows if r["arm"] == "Single-10")
        missing["run_id"] = row["run_id"]
        row["sr_zero"] = {b: v for b, v in row["sr_zero"].items() if b != 45.0}
    verdicts, _, ds = run(sr=sr, with_records=False, rows_hook=drop)
    floor = study_b.StudyB(ds).floor(V.PROPOSED)[45.0]
    assert floor["floored"] is None and "Single-10" in floor["unknown"]
    assert floor["below"] == ["Single-20", "Single-40", "Sparse"]
    g1 = verdicts["G1"]
    assert g1.proposal_status == V.NOT_COMPUTABLE and missing["run_id"] in g1.reason and "budget 45" in g1.reason
    # counting only G1's own arms (Q-floor-rule's other reading) budget 45 is not floored whatever Single-10's rate is
    assert g1.status == V.UNDECIDED and "Q-floor-rule" in g1.undecided_by and g1.readings["Q-floor-rule"] == V.SUPPORTED
    assert verdicts["G2"].proposal_status == V.NOT_COMPUTABLE and verdicts["G5"].proposal_status == V.NOT_COMPUTABLE
    assert verdicts["B-satisfaction[Dense-Sparse/b45]"].proposal_status == V.NOT_COMPUTABLE
    assert verdicts["B-satisfaction[Dense-Sparse/b5]"].proposal_status == V.SUPPORTED  # other budgets are decided
    # an unknown arm that cannot change the count leaves the budget decided
    by_budget[("Moderate", 45.0)] = 0.02  # five of seven below: floored whatever Single-10's rate is
    _, _, ds = run(sr=sr_table(table, noise=0.0, by_budget=by_budget), with_records=False, rows_hook=drop)
    assert study_b.StudyB(ds).floor(V.PROPOSED)[45.0]["floored"] is True


def _reach(indices: dict[str, dict[float, int]]):
    """rows_hook: each arm reaches the target at the given horizon index per budget (4 = never)."""
    def hook(rows):
        for row in rows:
            if row["arm"] not in indices:
                continue
            few, adapt = {}, {}
            for b in R.UNSEEN_BUDGETS:
                k = indices[row["arm"]][b]
                for i, h in enumerate(R.FEWSHOT_HORIZONS, start=1):
                    few[S.fewshot_key(b, h)] = 0.9 if i >= k else 0.2
                adapt[b] = R.FEWSHOT_HORIZONS[k - 1] if k <= len(R.FEWSHOT_HORIZONS) else max(R.FEWSHOT_HORIZONS) + 1
            row["sr_fewshot"], row["adapt_steps"] = few, adapt
    return hook


def test_g4_ordering_per_budget_is_a_reading_of_q_g4_reading(monkeypatch) -> None:
    """Q-g4-reading (answered in Table 9.1): the ordering clause is read on the arm statistic (the median over seeds of
    each seed's median over the budgets), here ordered 1 < 1.5 < 3, with Dense one horizon below Sparse on every
    budget: SUPPORTED. Per budget the medians are ordered on two budgets only (not a majority): the other reading
    gives INCONCLUSIVE."""
    indices = {"Dense": {5.0: 1, 15.0: 1, 30.0: 1, 45.0: 1}, "Moderate": {5.0: 2, 15.0: 2, 30.0: 1, 45.0: 1},
               "Sparse": {5.0: 3, 15.0: 3, 30.0: 3, 45.0: 3}}
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset())  # every key open: the other readings are computed
    verdicts, _, _ = run(with_records=False, rows_hook=_reach(indices))
    g4 = verdicts["G4"]
    assert g4.numbers["budgets_ordered"] == 2 and g4.numbers["arm_medians_ordered"] is True
    assert g4.numbers["arm_median_index(Moderate)"] == 1.5 and g4.numbers["budgets_dense_one_below"] == 4
    assert g4.numbers["ordering_read"].startswith("arm statistic")
    assert g4.proposal_status == V.SUPPORTED and g4.readings["Q-g4-reading"] == V.INCONCLUSIVE
    assert g4.status == V.UNDECIDED and g4.undecided_by == ("Q-g4-reading",)
    assert any("g4_order=per_budget gives INCONCLUSIVE" in n for n in g4.notes)
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset({"Q-g4-reading"}))
    verdicts, _, _ = run(with_records=False, rows_hook=_reach(indices))
    assert verdicts["G4"].status == V.SUPPORTED


def test_g4_within_one_horizon_is_a_spread_below_one() -> None:
    """Q-g4-reading (answered in Table 9.1): 'within one horizon of one another' holds on a budget when the largest
    minus the smallest of the three per-budget medians is below 1, the exact negation of 'at least one horizon'. Six
    seeds per arm give medians Dense 1, Moderate 1.5, Sparse 1.5 on three of the four budgets: a spread of 0.5, within
    one horizon (the 'same median index' reading left such a spread neither within nor apart). G4 is FALSIFIED."""
    rows, _ = syn.b_world("supported", seeds=tuple(range(6)))
    for row in rows:
        if row["arm"] not in study_b.G4_ARMS:
            continue
        index = {}
        for b in R.UNSEEN_BUDGETS:
            if b == 45.0:
                index[b] = {"Dense": 1, "Moderate": 2, "Sparse": 3}[row["arm"]]
            else:
                index[b] = 1 if row["arm"] == "Dense" or row["seed"] < 3 else 2
        row["sr_fewshot"] = {S.fewshot_key(b, h): 0.9 if i >= index[b] else 0.2 for b in R.UNSEEN_BUDGETS
                             for i, h in enumerate(R.FEWSHOT_HORIZONS, start=1)}
        row["adapt_steps"] = {b: R.FEWSHOT_HORIZONS[index[b] - 1] for b in R.UNSEEN_BUDGETS}
    ds = data.build_dataset(rows, data.Supplement(), mode="final", surplus_extra=1)
    out = study_b.g4_outcome(study_b.StudyB(ds), V.PROPOSED)
    assert [out.numbers[f"b5.median_index({a})"] for a in study_b.G4_ARMS] == [1.5, 1.5, 1]  # Sparse, Moderate, Dense
    assert out.numbers["b5.within_one_horizon"] is True and out.numbers["b45.within_one_horizon"] is False
    assert out.numbers["budgets_within"] == 3 and out.numbers["arm_medians_ordered"] is False
    assert out.status == V.FALSIFIED
    with pytest.raises(ValueError, match="within one horizon"):
        study_b.g4_outcome(study_b.StudyB(ds), dataclasses.replace(V.PROPOSED, g4_within="same_index"))


@pytest.mark.usefixtures("keys_open")
def test_g4_censored_at_the_largest_horizon_is_a_reading_of_q_g4_reading() -> None:
    """Part 4.2 registers "adaptation steps censored at the largest horizon"; the proposal counts a
    censored value as the index after it (Table 2.5 "above"). Dense reaches the target at the second horizon,
    Moderate at the third, Sparse never: after the largest horizon Sparse is last (ordered, SUPPORTED); at the
    largest horizon Sparse ties Moderate and the ordering fails."""
    k = len(R.FEWSHOT_HORIZONS)
    indices = {"Dense": dict.fromkeys(R.UNSEEN_BUDGETS, 2), "Moderate": dict.fromkeys(R.UNSEEN_BUDGETS, 3),
               "Sparse": dict.fromkeys(R.UNSEEN_BUDGETS, k + 1)}
    verdicts, _, ds = run(with_records=False, rows_hook=_reach(indices))
    assert next(r for r in ds.study_b if r["arm"] == "Sparse")["fewshot"][5.0]["censored"] is True
    g4 = verdicts["G4"]
    assert g4.numbers["b5.median_index(Sparse)"] == k + 1 and g4.proposal_status == V.SUPPORTED
    at = study_b.g4_outcome(study_b.StudyB(ds), V.alternative("Q-g4-reading", 2))
    assert at.numbers["b5.median_index(Sparse)"] == at.numbers["b5.median_index(Moderate)"] == k
    assert at.status == V.INCONCLUSIVE and at.numbers["censored_index"] == "the largest horizon's"
    assert g4.status == V.UNDECIDED and g4.undecided_by == ("Q-g4-reading",)
    assert any("g4_censored=at gives INCONCLUSIVE" in n for n in g4.notes)


def test_the_sensitivity_readings_are_reported_beside_the_verdicts(monkeypatch) -> None:
    """Answered in Table 9.1: "The all-seven-arms reading ... is reported beside as a sensitivity reading" (Q-g3-arms);
    "Three sensitivity readings are reported beside the verdict" (Q-g4-reading); "The readings 'only the arms of the
    comparison' and 'per-seed rates' are reported beside as sensitivity readings" (Q-floor-rule). Their statuses are
    in the numbers and never decide: per budget the medians are ordered on two budgets only, so the per-budget
    ordering gives INCONCLUSIVE beside a SUPPORTED G4; with a missing rate the floor at budget 45 is undecided, so G1
    is NOT_COMPUTABLE beside a SUPPORTED 'comparison' reading."""
    indices = {"Dense": {5.0: 1, 15.0: 1, 30.0: 1, 45.0: 1}, "Moderate": {5.0: 2, 15.0: 2, 30.0: 1, 45.0: 1},
               "Sparse": {5.0: 3, 15.0: 3, 30.0: 3, 45.0: 3}}
    verdicts, _, _ = run(with_records=False, rows_hook=_reach(indices))
    g4 = verdicts["G4"]
    assert g4.status == V.SUPPORTED and not g4.readings
    assert g4.numbers["sensitivity(Q-g4-reading: g4_order=per_budget)"] == V.INCONCLUSIVE
    floor = ["sensitivity(Q-floor-rule: floor_arms=comparison)", "sensitivity(Q-floor-rule: floor_rate=per_seed)"]

    def readings(key: str) -> list[str]:
        return sorted(k for k in verdicts[key].numbers if k.startswith("sensitivity(") and not k.endswith("_reason"))
    assert readings("G4") == sorted(floor + ["sensitivity(Q-g4-reading: g4_censored=at)",
                                             "sensitivity(Q-g4-reading: g4_order=per_budget)",
                                             "sensitivity(Q-g4-reading: g4_within=at_most_one)"])
    assert any("sensitivity readings" in n and "never decide" in n for n in g4.notes)
    assert readings("G3") == sorted(floor + ["sensitivity(Q-g3-arms: g3_arms=seven)"])
    for key in ("G1", "G2", "G5", "B-satisfaction[Dense-Sparse/b5]"):
        assert readings(key) == floor, key  # Q-floor-rule's joint reading is not one its answer names
    assert readings("B-violation[Dense-Sparse/b5]") == []  # the floor rule does not enter it
    by_budget = {(a, 45.0): 0.02 for a in ("Single-10", "Single-20", "Single-40", "Sparse")}
    table = dict(Sparse=0.40, Moderate=0.55, Dense=0.75, Continuous=0.77, **{f"Single-{k}": 0.3 for k in (10, 20, 40)})

    def drop(rows):
        row = next(r for r in rows if r["arm"] == "Single-10")
        row["sr_zero"] = {b: v for b, v in row["sr_zero"].items() if b != 45.0}
    g1 = run(sr=sr_table(table, noise=0.0, by_budget=by_budget), with_records=False, rows_hook=drop)[0]["G1"]
    assert g1.status == V.NOT_COMPUTABLE
    assert g1.numbers["sensitivity(Q-floor-rule: floor_arms=comparison)"] == V.SUPPORTED
    # while a key is open its other readings are decide's (UNDECIDED), not sensitivity readings beside the verdict
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset())
    g4 = run(with_records=False, rows_hook=_reach(indices))[0]["G4"]
    assert g4.status == V.UNDECIDED and not any(k.startswith("sensitivity(") for k in g4.numbers)


@pytest.mark.usefixtures("keys_open")
def test_per_budget_claims_keep_holm_under_every_reading_of_q_holm_families() -> None:
    """The four budgets are the claim's only family (Part 5.2), so Q-holm-families (how two families
    combine) cannot drop Holm: a claim whose uncorrected interval excludes zero but whose Holm-adjusted p is above
    0.05 is INCONCLUSIVE under every reading."""
    table = {a: 0.5 for a in R.STUDY_B_ARMS}
    verdicts, _, _ = run(sr=sr_table(table, noise=0.05, by_budget={("Dense", 5.0): 0.57}))
    claim = verdicts["B-satisfaction[Dense-Sparse/b5]"]
    assert claim.numbers["welch_lo"] > 0 and claim.numbers["p"] < R.ALPHA < claim.numbers["holm_adjusted_p"]
    assert claim.status == V.INCONCLUSIVE and claim.readings["Q-holm-families"] == V.INCONCLUSIVE
    # its answer covers Study B's four budgets too ("Claims with a single family ... use that family")
    assert "Q-holm-families" in claim.provisional_on


@pytest.mark.usefixtures("keys_open")
def test_g2_on_the_mean_over_the_budgets_is_a_reading_of_q_g2_outcome() -> None:
    def sr(arm, budget, seed):
        e = syn.noise((seed + ARMS.index(arm)) % len(R.SEEDS))
        if arm == "Dense":  # 20 points above the singles at 5 and 45, 3 points at 15 and 30
            return round((0.50 if budget in (5.0, 45.0) else 0.33) + 0.02 * e, 2)
        if arm.startswith("Single"):
            return round(0.30 + 0.02 * e, 2)
        return round({"Sparse": 0.2, "Moderate": 0.3, "Continuous": 0.45}[arm] + 0.02 * e, 2)
    verdicts, cat, _ = run(sr=sr)
    g2 = verdicts["G2"]
    assert g2.numbers["budgets_won"] == 2 and g2.numbers["budgets_lost"] == 2 and g2.numbers["falsification"] is True
    # two losses falsify the box, but fewer than two losing budgets are bounded below 5 points (Part 5.7):
    # inconclusive (Q-falsification-calibration)
    assert g2.numbers["bounded_losing_budgets"] < R.G2_MIN_BUDGETS_FALSIFY and g2.proposal_status == V.INCONCLUSIVE
    composite = "composite: Dense - comparator, mean over budgets"
    assert g2.numbers[composite] == pytest.approx(0.115, abs=0.01) and g2.numbers[f"{composite}_welch_lo"] > 0
    assert g2.readings["Q-g2-outcome"] == V.SUPPORTED and g2.status == V.UNDECIDED and "Q-g2-outcome" in g2.undecided_by
    assert g2.readings["Q-falsification-calibration"] == V.FALSIFIED
    # Part 5.7 on the two losing budgets, on the 95 percent interval (the 97.5 percent one decides G2)
    assert g2.numbers["bound(b15 Dense - Single-10, 95% Welch)"] < g2.numbers["b15.Dense - Single-10_welch_hi"]
    readings = [g2.numbers[f"bound(b{b:g} Dense - {g2.numbers[f'b{b:g}.comparator']}, 95% Welch)_reading"]
                for b in (15, 30)]
    assert g2.numbers["bounded_losing_budgets"] == sum(r == V.BOUND_BELOW for r in readings)
    assert any(r["key"].startswith("G2|composite|") for r in cat.tables["B_estimates"])


@pytest.mark.parametrize("keys", ["answered", "open"])
def test_the_estimate_table_holds_the_decided_readings_estimate(monkeypatch, keys) -> None:
    """Q-floor-rule's other readings run beside the decided one: as sensitivity readings once the key is answered
    (``study_b.SENSITIVITY_KEYS``) and as its other readings while it is open (``verdict.decide``). Budget 5 is
    floored under the decided reading (four of seven arm means at 0.048) but under none of the others ('comparison':
    Continuous and Dense, or Sparse, Moderate and Dense, are off the floor; 'per-seed': 8 of 35 runs below), so G1's
    and G5's readings estimate different means under one analysis id; only the decided reading's estimate is tabled."""
    if keys == "open":
        monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset())
    table = dict(Sparse=0.40, Moderate=0.55, Dense=0.75, Continuous=0.77, **{f"Single-{k}": 0.3 for k in (10, 20, 40)})
    others = sr_table(table, by_budget={("Dense", 5.0): 0.6, ("Continuous", 5.0): 0.6})

    def sr(arm: str, budget: float, seed: int) -> float:
        if budget == 5.0 and arm in ("Sparse", "Single-10", "Single-20", "Single-40"):
            return 0.0 if seed < 2 else 0.08  # arm mean 0.048, below 5 percent; two of its five runs below it
        return others(arm, budget, seed)
    verdicts, cat, ds = run(sr=sr)
    B = study_b.StudyB(ds)
    assert B.floor(V.PROPOSED)[5.0]["floored"] is True
    pair = lambda b: ("Continuous", "Dense")  # noqa: E731
    assert not any(B.floor(r, pair)[5.0]["floored"] for r in V.alternatives("Q-floor-rule"))
    g5 = verdicts["G5"]
    assert g5.numbers["budgets_floored"] == [5.0]
    row = next(r for r in cat.tables["B_estimates"] if r["key"].startswith("G5|Continuous-Dense|"))
    # not the four-budget mean of Q-floor-rule's other readings
    assert row["diff"] == pytest.approx(g5.numbers["Continuous - Dense"])
    g1 = verdicts["G1"]
    row = next(r for r in cat.tables["B_estimates"] if r["key"].startswith("G1|Dense-Sparse|"))
    assert row["diff"] == pytest.approx(g1.numbers["Dense - Sparse"])


def test_study_b_iqm_of_every_tabled_contrast() -> None:
    _, cat, _ = run("supported", iqm=True)
    rows = {(r["x"], r["y"], r["outcome"]): r for r in cat.tables["B_iqm"]}
    g1 = rows[("Dense", "Sparse", "sr_zero_mean|5,15,30,45")]
    assert g1["iqm_lo"] <= g1["iqm_diff"] <= g1["iqm_hi"] and "G1|Dense-Sparse" in g1["used_by"]
    assert ("Dense", "Single-10", "sr_zero|b5") in rows and ("Dense", "Sparse", "vm_zero|b5") in rows
    assert any(k[1] is None and k[2].startswith("G3|") for k in rows)  # G3's within-arm differences
    # Q-iqm: "Its 95 percent percentile interval", also for G1's and G2's 97.5 percent contrasts
    assert {r["level"] for r in cat.tables["B_iqm"]} == {R.INTERVAL_LEVEL} and len(rows) == len(cat.tables["B_iqm"])
    # budget 5 floored: G1 and the per-budget claims read Dense - Sparse on violation magnitude at budget 5 at 97.5 and
    # 95 percent, one IQM row for that pair and source (stats.EpisodeOutcome)
    by_budget = {(a, 5.0): 0.0 for a in ("Single-10", "Single-20", "Single-40", "Sparse")}
    table = dict(Sparse=0.40, Moderate=0.55, Dense=0.75, Continuous=0.77, **{f"Single-{k}": 0.3 for k in (10, 20, 40)})
    _, cat, _ = run(sr=sr_table(table, by_budget=by_budget), iqm=True)
    vm5 = [r for r in cat.tables["B_iqm"] if (r["x"], r["y"], r["outcome"]) == ("Dense", "Sparse", "vm_zero|b5")]
    assert len(vm5) == 1 and vm5[0]["level"] == R.INTERVAL_LEVEL
    assert {"G1|vm|b5", "B-violation|Dense-Sparse|b5|vm"} <= set(vm5[0]["used_by"].split())
    _, cat, _ = run("supported", iqm=True, with_records=False)  # no zero-shot records: no episodes
    assert cat.tables["B_iqm"] and all(r["note"].startswith("not computable") for r in cat.tables["B_iqm"])


@pytest.mark.usefixtures("keys_open")
def test_g3_falsified_carries_the_bound_of_each_arm() -> None:
    verdicts, _, _ = run("flat")
    g3 = verdicts["G3"]
    assert g3.proposal_status == V.FALSIFIED and g3.numbers["bounded_arms"] == len(study_b.G3_ARMS)
    assert g3.numbers["bound(Dense.D, 95% t)_reading"] == V.BOUND_BELOW
    assert g3.readings["Q-falsification-calibration"] == V.FALSIFIED


def test_a_budget_claim_waits_on_a_budget_whose_data_are_still_to_come() -> None:
    """Holm over the four budgets shrinks to the budgets not tested, never on data still to come: with one Dense
    seed's rate at budget 45 missing, Dense - Sparse at budget 5 (p about 0.03, second largest of four) cannot pass
    Holm as if the family had three members."""
    table = dict(Sparse=0.40, Moderate=0.55, Dense=0.75, Continuous=0.77, **{f"Single-{k}": 0.3 for k in (10, 20, 40)})
    sr = sr_table(table, noise=0.1, by_budget={("Dense", 5.0): 0.53, ("Dense", 45.0): 0.40})
    key = "B-satisfaction[Dense-Sparse/b5]"
    complete, _, _ = run(sr=sr, with_records=False)
    assert complete[key].proposal_status == V.INCONCLUSIVE  # p * 2 > R.ALPHA with budget 45's large p in the family
    assert complete[key].numbers["holm_family_size"] == 4 and R.ALPHA / 2 < complete[key].numbers["p"] < R.ALPHA
    missing = {}

    def drop(rows):
        row = next(r for r in rows if r["arm"] == "Dense" and r["seed"] == 0)
        missing["run_id"] = row["run_id"]
        row["sr_zero"] = {b: v for b, v in row["sr_zero"].items() if b != 45.0}
    waiting, _, _ = run(sr=sr, with_records=False, rows_hook=drop)
    claim = waiting[key]
    assert claim.proposal_status == V.NOT_COMPUTABLE and "Holm" in claim.reason and missing["run_id"] in claim.reason
    assert claim.numbers["holm_survives"] is None
    assert waiting["B-satisfaction[Dense-Sparse/b15]"].proposal_status == V.SUPPORTED  # decided whatever 45 brings


# ---------------------------------------------------------------------------
# Data still to come (Q-arm-complete)
# ---------------------------------------------------------------------------


def _drop(rows: list, keep) -> None:
    rows[:] = [r for r in rows if keep(r)]


@pytest.mark.usefixtures("keys_open")
def test_an_arm_below_its_seed_target_makes_its_comparisons_incomplete() -> None:
    """With Sparse seeds 2 to 4 still training, G1 was SUPPORTED on 5 Dense against 2 Sparse
    seeds. A Study B arm below its seed target is data still to come: the verdicts that need it are NOT_COMPUTABLE
    and its per-budget cells are incomplete Holm members, never decided on the seeds present."""
    v, _, ds = run("supported",
                   rows_hook=lambda rows: _drop(rows, lambda r: not (r["arm"] == "Sparse" and r["seed"] >= 2)))
    assert ds.awaited_b == {"Sparse": (2, len(R.SEEDS))}
    assert ds.awaited_b_lines() == [f"B-Sparse: 2 of its {len(R.SEEDS)} seeds completed (Q-arm-complete)"]
    for key in ("G1", "G3", "G4"):
        assert v[key].status == V.NOT_COMPUTABLE, (key, v[key].status)
        assert f"Sparse: 2 of its {len(R.SEEDS)} seeds completed (Q-arm-complete)" in v[key].reason, v[key].reason
    cell = v["B-satisfaction[Dense-Sparse/b15]"]
    assert cell.status == V.NOT_COMPUTABLE and cell.reason.startswith("incomplete")
    # G2 compares Dense with the single-level arms; the floor rule decides without Sparse's unknown rate here
    assert v["G2"].status == V.SUPPORTED
    assert "Q-arm-complete" in v["G1"].provisional_on and "Q-arm-complete" in v["G2"].provisional_on


def test_an_excluded_run_waits_for_its_replacement_and_the_surplus_seeds_count() -> None:
    """A crashed run (Part 5.6) is not a completed row: its arm waits for the replacement seed. With E surplus seeds
    (Part 5.5) the target is 5 + E for every Study B arm."""
    def hook(rows: list) -> None:
        for r in rows:
            if r["arm"] == "Dense" and r["seed"] == 4:
                r["completed"], r["failure_cause"] = False, "crash"
    v, _, ds = run("supported", rows_hook=hook)
    assert ds.awaited_b == {"Dense": (4, len(R.SEEDS))} and [e["run_id"] for e in ds.exclusions] == ["B-Dense-s4"]
    for key in ("G1", "G2", "G5"):
        assert v[key].status == V.NOT_COMPUTABLE, (key, v[key].status)
        assert f"Dense: {len(R.SEEDS) - 1} of its {len(R.SEEDS)} seeds" in v[key].reason, (key, v[key].reason)
    rows, _ = syn.b_world("supported")
    full = data.build_dataset(rows, data.Supplement(), mode="final", surplus_extra=2)
    assert full.awaited_b == {arm: (len(R.SEEDS), len(R.SEEDS) + 2) for arm in R.STUDY_B_ARMS}
    assert data.build_dataset(rows, data.Supplement(), mode="final").awaited_b == {}
    with pytest.raises(data.DataError, match="surplus_extra"):
        data.build_dataset(rows, data.Supplement(), mode="final", surplus_extra=-1)


def test_an_arm_above_its_seed_target_refuses_a_surplus_extra_that_is_too_small() -> None:
    """With Dense and Sparse at 7 completed seeds and the other arms at 5 (surplus seeds 5 and 6
    still training for most arms), ``--surplus-extra 0`` decided G1 on 7 Sparse against 5 Moderate seeds. A Part 5.6
    replacement never raises a completed count above the target, so a count above 5 + E means E is wrong: refused."""
    rows, _ = syn.b_world("supported")
    more, _ = syn.b_world("supported", seeds=range(5, 7))
    rows += [r for r in more if r["arm"] in ("Dense", "Sparse")]
    with pytest.raises(data.DataError, match=r"B-Sparse \(7 completed, seeds \[0, 1, 2, 3, 4, 5, 6\]; target 5\)"
                                             r".*B-Dense.*--surplus-extra must be the data root's stored "
                                             r"surplus_extra"):
        data.build_dataset(rows, data.Supplement(), mode="final", surplus_extra=0)
    ds = data.build_dataset(rows, data.Supplement(), mode="final", surplus_extra=2)
    assert ds.awaited_b == {arm: (5, 7) for arm in R.STUDY_B_ARMS if arm not in ("Dense", "Sparse")}


def test_the_study_b_verdicts_are_provisional_on_q_arm_complete() -> None:
    from analysis import questions

    assert "Q-arm-complete" in questions.STUDY_B_DATA


def test_g3_is_secondary_in_the_code_and_in_the_q_g3_arms_text() -> None:
    """Part 5.2 puts G3 among the secondary outcomes; the Q-g3-arms text called its four-arm
    reading 'confirmatory'. The proposal decides a secondary box."""
    v, _, _ = run("supported")
    assert v["G3"].label == V.SECONDARY
    assert "confirmatory" not in R.PENDING["Q-g3-arms"] and "secondary" in R.PENDING["Q-g3-arms"]


@pytest.mark.usefixtures("keys_open")
def test_g4_names_the_overlap_question_it_can_be_decided_by() -> None:
    """Six seeds per arm give median horizon indices Dense 1, Moderate 1.5,
    Sparse 2; under Q-g4-reading's 'at most one horizon' reading support and falsification both hold, so the
    verdict rests on Q-support-falsify-overlap's proposal, and G4 names that key."""
    from dataclasses import replace

    from analysis import questions

    rows, _ = syn.b_world("supported", seeds=tuple(range(6)))
    for row in rows:
        k = {"Dense": 1, "Sparse": 2}.get(row["arm"], 1 if row["arm"] == "Moderate" and row["seed"] < 3 else 2)
        few = {S.fewshot_key(b, h): 0.9 if i >= k else 0.2 for b in R.UNSEEN_BUDGETS
               for i, h in enumerate(R.FEWSHOT_HORIZONS, start=1)}
        row["sr_fewshot"], row["adapt_steps"] = few, {b: R.FEWSHOT_HORIZONS[k - 1] for b in R.UNSEEN_BUDGETS}
    ds = data.build_dataset(rows, data.Supplement(), mode="final", surplus_extra=1)
    B = study_b.StudyB(ds)
    within = V.alternative("Q-g4-reading", 0)
    assert within.g4_within == "at_most_one"
    assert study_b.g4_outcome(B, within).status == V.FALSIFIED
    assert study_b.g4_outcome(B, replace(within, overlap="supported")).status == V.SUPPORTED
    assert "Q-support-falsify-overlap" in questions.keys_for("G4")
    g4 = {v.id: v for v in study_b.analyse(ds, iqm=False).verdicts}["G4"]
    assert "Q-support-falsify-overlap" in g4.provisional_on


# ---------------------------------------------------------------------------
# Floor and bound edge cases
# ---------------------------------------------------------------------------


def test_the_per_seed_floor_counts_an_absent_arm_at_its_seed_target() -> None:
    """With two surplus seeds (target 7) and every other arm at 5 of 7, an arm with no completed run
    counted as 5 runs (the largest count present) instead of its target of 7."""
    rows, _ = syn.b_world("supported")
    rows = [r for r in rows if r["arm"] != "Sparse"]
    ds = data.build_dataset(rows, data.Supplement(), mode="final", surplus_extra=2)
    assert ds.awaited_b["Sparse"] == (0, len(R.SEEDS) + 2)
    floor = study_b.StudyB(ds).floor(V.alternative("Q-floor-rule", 1))[5.0]
    assert floor["unit"] == "run" and floor["units"] == len(R.STUDY_B_ARMS) * (len(R.SEEDS) + 2)
    assert floor["unknown"]["Sparse"] == "no completed run"


def test_g3_falsified_on_violation_magnitude_has_no_bound() -> None:
    """A G3 falsified on violation magnitude said both "no bound on violation magnitude" and
    "reported with its bound ... (bound(...) numbers)"; under the bound reading it gave the wrong reason. Without a
    bound the falsification is not confirmed (Q-falsification-calibration, answered in Table 9.1): INCONCLUSIVE."""
    table = {a: 0.5 for a in R.STUDY_B_ARMS}
    _, _, ds = run(sr=sr_table(table, by_budget={(a, 5.0): 0.0 for a in R.STUDY_B_ARMS}), with_records=False)
    for rec in ds.study_b:  # equal rises in violation magnitude at 5 and at 45: the differences straddle zero
        e = (-1) ** rec["seed"] * 0.01 * (rec["seed"] + 1)
        rec["vm_zero"], rec["vm_train"] = {5.0: 1.0 + e, 45.0: 1.0}, {10.0: 0.0, 40.0: 0.0}
    B = study_b.StudyB(ds)
    box = study_b.g3_outcome(B, V.alternative("Q-falsification-calibration"))  # the box conditions decide
    assert box.status == V.FALSIFIED and box.numbers["outcome"] == "violation magnitude"
    assert box.numbers["bounded_arms"] is None and not any("bound(...)" in n for n in box.notes)
    assert any("no bound on violation magnitude" in n for n in box.notes)
    out = study_b.g3_outcome(B, V.PROPOSED)
    assert out.status == V.INCONCLUSIVE and out.numbers["falsification"] is True
    assert any("no Part 5.7 bound exists on violation magnitude" in n for n in out.notes)
    assert not any("not below the minimum effect" in n for n in out.notes)


def test_g2_falsified_on_violation_magnitude_has_no_bound() -> None:
    """As for G3: a G2 whose losing budgets are all floored (violation magnitude) has no Part 5.7 bound, so the
    verdict neither cites "bound(...) numbers" nor says the bound is not below the effect; the falsification is not
    confirmed (Q-falsification-calibration, answered in Table 9.1): INCONCLUSIVE."""
    table = dict(Sparse=0.40, Moderate=0.55, Dense=0.75, Continuous=0.77, **{f"Single-{k}": 0.3 for k in (10, 20, 40)})
    by = {(a, b): 0.0 for a in ("Single-10", "Single-20", "Single-40", "Sparse") for b in (5.0, 15.0)}
    _, _, ds = run(sr=sr_table(table, by_budget=by), with_records=False)
    for rec in ds.study_b:  # Dense above the singles on violation magnitude at the floored budgets 5 and 15
        e = 0.01 * (-1) ** rec["seed"] * (rec["seed"] + 1)
        rec["vm_zero"] = {b: (2.0 if rec["arm"] == "Dense" else 1.0) + e for b in R.UNSEEN_BUDGETS}
    B = study_b.StudyB(ds)
    box = study_b.g2_outcome(B, V.alternative("Q-falsification-calibration"))  # the box conditions decide
    assert box.status == V.FALSIFIED and box.numbers["budgets_lost"] == 2
    assert not any("bound(...)" in n for n in box.notes)
    assert any("no bound on violation magnitude" in n for n in box.notes)
    assert "bound(b5 Dense - Single-10, violation magnitude)_reading" in box.numbers
    out = study_b.g2_outcome(B, V.PROPOSED)
    assert out.status == V.INCONCLUSIVE and out.numbers["falsification"] is True
    assert any("no Part 5.7 bound exists on violation magnitude" in n for n in out.notes)
    assert not any("not below the minimum effect" in n for n in out.notes)
    # budget 15 is floored: its comparator is chosen on violation magnitude (equal here: the tie-break)
    assert out.numbers["b15.comparator_mean_violation"]["Single-10"] == pytest.approx(
        out.numbers["b15.comparator_mean_violation"]["Single-20"])
    assert any("budget 15: comparator chosen on violation magnitude" in n for n in out.notes)


def test_the_arm_mean_floor_names_an_awaited_arm() -> None:
    """An arm with seeds still to come is unknown to the arm-mean floor as awaited, not as missing a rate."""
    table = dict(Sparse=0.40, Moderate=0.55, Dense=0.75, Continuous=0.77, **{f"Single-{k}": 0.3 for k in (10, 20, 40)})
    by = {(a, 5.0): 0.0 for a in ("Single-10", "Single-20", "Single-40")}
    _, _, ds = run(sr=sr_table(table, by_budget=by), with_records=False,
                   rows_hook=lambda rows: _drop(rows, lambda r: not (r["arm"] == "Sparse" and r["seed"] >= 2)))
    floor = study_b.StudyB(ds).floor(V.PROPOSED)[5.0]
    assert floor["unknown"] == {"Sparse": f"Sparse: 2 of its {len(R.SEEDS)} seeds completed (Q-arm-complete)"}

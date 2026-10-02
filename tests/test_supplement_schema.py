"""Supplement record models (results/supplement_schema.py; Role 4): good and bad records."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import test_analysis_synthetic as syn  # noqa: E402

from configs import registered as R  # noqa: E402
from results import supplement_schema as S  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
# The kinds of Q-ledger-v2 (docs/DECISIONS.md), in its order.
LEDGER_V2_KINDS = ("evaluation", "measurement", "battery", "final_battery", "sensitivity_battery", "continuation",
                   "zero_shot", "fewshot", "training")


def _row() -> dict:
    return syn.a_world("supported", groups=("main",))[-1]  # a late arm with onset > 0


def _spec(row: dict) -> Any:  # pilot.manifest.RunSpec
    return next(s for s in syn.specs(groups=("main",)) if s.run_id == row["run_id"])


def good_records() -> dict[str, dict]:
    row = _row()
    spec = _spec(row)
    selection = []
    for i, step in enumerate(sorted(c["step"] for c in row["checkpoints"])[-R.SELECTION_WINDOW_CHECKPOINTS:]):
        block = syn.episodes(25.0 + i, seed_set="selection", base=1_000_000)
        selection.append({"step": step, **block})
    _, b_recs = syn.b_world()
    b_run = next(iter(b_recs))
    dense_run = next(r for r in b_recs if r.startswith("B-Dense"))
    cont = {"kind": "continuation", "run_id": row["run_id"], "code_commit": syn.COMMIT, "condition": "finetune",
            "continuation_run_id": row["run_id"].replace("-s", "-finetune-s"), "parent_step": row["matched_checkpoint_step"],
            "continuation_steps": R.FINETUNE_STEPS, "step": R.FINETUNE_STEPS, "task": row["task"],
            "episodes": syn.episodes(27.0), "measurement_cost": row["measurement_cost"],
            "gap": 27.0 - row["measurement_cost"]}
    return {
        "evaluation": {"kind": "evaluation", "run_id": row["run_id"], "code_commit": syn.COMMIT,
                       "final_step": spec.total_steps, "final": syn.episodes(26.0), "selection": selection,
                       "short_episodes": 0},
        "measurement": syn.measurement_record(row),
        "battery": syn.battery_record(row, "hazard"),
        "final_battery": syn.battery_record(row, "dynamics", kind="final_battery", step=spec.total_steps),
        "sensitivity_battery": syn.battery_record(row, "hazard", kind="sensitivity_battery"),
        "continuation": cont,
        "zero_shot": b_recs[b_run]["zero_shot"],
        "fewshot": b_recs[dense_run]["fewshot"][0],
        "training": syn.training_record(row, spec=spec, dormant_check=0.3, rank_check=30.0),
    }


def test_the_kinds_are_those_of_q_ledger_v2() -> None:
    assert tuple(S.KIND_MODELS) == LEDGER_V2_KINDS == S.KINDS
    assert S.SUPPLEMENT_SCHEMA_VERSION == 1
    assert S.PART_KINDS == {"battery", "final_battery", "sensitivity_battery", "continuation", "fewshot"}


@pytest.mark.parametrize("kind", LEDGER_V2_KINDS)
def test_every_kind_validates_a_good_record_and_rewrites_byte_identically(kind: str) -> None:
    record = good_records()[kind]
    model = S.validate_record(kind, record)
    assert model.kind == kind and model.run_id == record["run_id"]
    text = S.canonical_json(kind, record)
    assert "utc" not in text and "timestamp" not in text  # no time in a record: rewrites are identical
    again = S.canonical_json(kind, json.loads(text))
    assert again == text  # a record read back and written again gives the same bytes
    assert S.canonical_json(kind, model) == text
    assert S.validate_record(kind, model) == model


def test_record_parts_name_the_record_within_its_run() -> None:
    goods = good_records()
    assert S.record_part("battery", goods["battery"]) == "hazard"
    assert S.record_part("final_battery", goods["final_battery"]) == "dynamics"
    assert S.record_part("continuation", goods["continuation"]) == "finetune"
    assert S.record_part("fewshot", goods["fewshot"]) == "b5"
    for kind in ("evaluation", "measurement", "zero_shot", "training"):
        assert S.record_part(kind, goods[kind]) is None
    # a built model is read as it is when it is of the kind, and refused (ValueError) when it is of another kind
    battery = S.validate_record("battery", goods["battery"])
    measurement = S.validate_record("measurement", goods["measurement"])
    assert S.record_part("battery", battery) == "hazard"
    with pytest.raises(ValueError, match="is not a battery record"):
        S.record_part("battery", measurement)
    with pytest.raises(ValueError, match="is not a measurement record"):
        S.record_part("measurement", battery)


def test_the_measurement_record_gives_measurement_cost_and_return() -> None:
    record = good_records()["measurement"]
    model = S.validate_record("measurement", record)
    assert model.measurement_cost == statistics.fmean(record["episodes"]["episode_costs"])
    assert model.measurement_return == statistics.fmean(record["episodes"]["episode_returns"])


def test_fewshot_key_is_the_pipelines() -> None:
    from pilot.contracts import fewshot_key, parse_fewshot_key

    for budget in R.UNSEEN_BUDGETS:
        for horizon in R.FEWSHOT_HORIZONS:
            assert S.fewshot_key(budget, horizon) == fewshot_key(budget, horizon)
            assert parse_fewshot_key(S.fewshot_key(budget, horizon)) == (budget, horizon)
    assert S.fewshot_key(5.0, 200_000) == "5.0_200000"  # the frozen schema test's form


def _refused(kind: str, record: dict, match: str | None = None) -> None:
    with pytest.raises(ValueError, match=match):
        S.validate_record(kind, record)


def test_unknown_kinds_fields_and_kind_mismatch_are_refused() -> None:
    goods = good_records()
    with pytest.raises(ValueError, match="unknown supplement kind"):
        S.validate_record("episodes", goods["measurement"])
    for unhashable in (["measurement"], {"kind": "measurement"}):  # a ValueError, never a TypeError
        with pytest.raises(ValueError, match="unknown supplement kind"):
            S.validate_record(unhashable, goods["measurement"])  # type: ignore[arg-type]
    _refused("battery", goods["measurement"])  # the record says it is a measurement
    for kind, record in goods.items():
        extra = dict(record, recorded_at="2026-10-01T00:00:00Z")  # extra='forbid': no timestamps either
        _refused(kind, extra)
    bad = dict(goods["measurement"], code_commit="abc")
    _refused("measurement", bad)
    bad = dict(goods["measurement"], run_id="A-PointGoal1-N0.00")  # no seed suffix
    _refused("measurement", bad)
    bad = dict(goods["measurement"], schema_version=2)
    _refused("measurement", bad)
    for version in (True, 1.0):  # integers refuse booleans and floats, the Literal-typed version too
        _refused("measurement", dict(goods["measurement"], schema_version=version), "must be an integer")
    with pytest.raises(ValueError):
        S.validate_record("measurement", [1, 2])


def test_episode_blocks_need_100_finite_values_with_matching_means() -> None:
    good = good_records()["measurement"]
    for mutate in (
        lambda b: b["episode_costs"].pop(),  # 99 episodes
        lambda b: b["episode_returns"].append(1.0),  # 101
        lambda b: b["episode_costs"].__setitem__(0, math.nan),
        lambda b: b["episode_returns"].__setitem__(0, math.inf),
        lambda b: b["episode_costs"].__setitem__(0, -1.0),  # a cost is never negative
        lambda b: b["episode_costs"].__setitem__(0, True),  # booleans are not numbers here
        lambda b: b["episode_costs"].__setitem__(0, "25"),
        lambda b: b.__setitem__("mean_cost", b["mean_cost"] + 0.01),  # the mean is recomputed
        lambda b: b.__setitem__("mean_return", b["mean_return"] - 0.5),
        lambda b: b["seeds"].__setitem__(1, b["seeds"][0]),  # seeds are distinct
        lambda b: b["seeds"].__setitem__(1, 2.0),  # seeds are integers
        lambda b: b.__setitem__("seed_set", "training"),
    ):
        record = copy.deepcopy(good)
        mutate(record["episodes"])
        _refused("measurement", record)
    record = copy.deepcopy(good)
    record["episodes"]["seed_set"] = "selection"
    _refused("measurement", record)  # the matched checkpoint is measured on the measurement set


def _replacement(slot: int, seed: int, reserve: int, step: int = 12) -> dict:
    return {"slot_index": slot, "seed": seed, "reserve_seed": reserve, "step": step, "warning": "mjWARN_BADQVEL x1"}


def test_unstable_replacements_are_optional_and_absent_from_a_record_without_them() -> None:
    """Q-mujoco-exception (Table 9.1): a block of one evaluation call keeps its planned seeds and lists the reserve
    episodes that replaced unstable ones; a record without any has the bytes it had before the field existed."""
    from pilot import contracts

    assert (S.RESERVE_SEED_OFFSET, S.MAX_UNSTABLE_EPISODES) == (contracts.RESERVE_SEED_OFFSET,
                                                                contracts.MAX_UNSTABLE_EPISODES)
    goods = good_records()
    for kind, record in goods.items():
        assert "unstable_replacements" not in S.canonical_json(kind, record)
    record = copy.deepcopy(goods["evaluation"])
    seeds = record["final"]["seeds"]
    base = min(seeds) + S.RESERVE_SEED_OFFSET
    record["final"]["unstable_replacements"] = [_replacement(4, seeds[4], base), _replacement(4, base, base + 1, step=0)]
    record["selection"][0]["unstable_replacements"] = []  # none: the same as absent
    text = S.canonical_json("evaluation", record)
    assert S.canonical_json("evaluation", json.loads(text)) == text
    data = json.loads(text)
    assert [u["reserve_seed"] for u in data["final"]["unstable_replacements"]] == [base, base + 1]
    assert "unstable_replacements" not in data["selection"][0] and data["final"]["seeds"] == seeds
    model = S.validate_record("evaluation", record)
    assert model.selection[0].unstable_replacements is None and model.final.unstable_replacements[1].step == 0


def test_unstable_replacements_follow_the_reserve_rule() -> None:
    """The planned seed first, each later one the previous reserve seed, the r-th reserve seed min(seeds) + 50,000
    + r, at most five per evaluation call; budget and horizon blocks against their record's seeds."""
    goods = good_records()
    seeds = goods["measurement"]["episodes"]["seeds"]
    base = min(seeds) + S.RESERVE_SEED_OFFSET
    bad = [
        [_replacement(4, seeds[4], base + 1)],  # r skipped
        [_replacement(4, seeds[5], base)],  # not the slot's planned seed
        [_replacement(9, seeds[9], base), _replacement(2, seeds[2], base + 1)],  # slots out of order
        [_replacement(100, seeds[0], base)],  # no episode 100
        [_replacement(i, seeds[i], base + i) for i in range(6)],  # more than five
        [_replacement(0, seeds[0], base, step=-1)],
        [dict(_replacement(0, seeds[0], base), step=True)],
        [dict(_replacement(0, seeds[0], base), warning="")],
        [dict(_replacement(0, seeds[0], base), recorded_at="now")],
    ]
    for replacements in bad:
        record = copy.deepcopy(goods["measurement"])
        record["episodes"]["unstable_replacements"] = replacements
        _refused("measurement", record)
    five = copy.deepcopy(goods["measurement"])
    five["episodes"]["unstable_replacements"] = [_replacement(i, seeds[i], base + i) for i in range(5)]
    S.validate_record("measurement", five)
    zero_shot = copy.deepcopy(goods["zero_shot"])
    zseeds = zero_shot["seeds"]
    zero_shot["budgets"][0]["unstable_replacements"] = [_replacement(3, zseeds[3], min(zseeds) + S.RESERVE_SEED_OFFSET)]
    S.validate_record("zero_shot", zero_shot)
    zero_shot["budgets"][0]["unstable_replacements"][0]["seed"] = zseeds[4]
    _refused("zero_shot", zero_shot, "replaces seed")
    fewshot = copy.deepcopy(goods["fewshot"])
    fseeds = fewshot["seeds"]
    fewshot["by_horizon"][-1]["unstable_replacements"] = [_replacement(0, fseeds[0], min(fseeds) + S.RESERVE_SEED_OFFSET)]
    S.validate_record("fewshot", fewshot)
    fewshot["by_horizon"][-1]["unstable_replacements"][0]["reserve_seed"] += 1
    _refused("fewshot", fewshot, "next unused")


def test_battery_records_check_the_condition_the_seed_set_and_equation_1() -> None:
    good = good_records()["battery"]
    _refused("battery", dict(good, gap=good["gap"] + 0.5))
    _refused("battery", dict(good, condition="dynamics"))  # hazard episodes and seeds under dynamics
    _refused("battery", dict(good, hazard=None))
    _refused("battery", dict(good, measurement_cost=None))
    bad = copy.deepcopy(good)
    bad["hazard"]["layout_seeds"] = bad["hazard"]["layout_seeds"][:-1]
    _refused("battery", bad)
    for episodes, message in ((4, "Input should be 5"), (5.0, "must be an integer"), (True, "must be an integer")):
        bad = copy.deepcopy(good)
        bad["hazard"]["episodes_per_layout"] = episodes  # five, an integer
        _refused("battery", bad, message)
    bad = copy.deepcopy(good)
    bad["hazard"]["form"] = "registered"  # Q-hazard: the registered-wording form enters no result
    _refused("battery", bad, "Q-hazard")
    final = good_records()["final_battery"]
    measurement = dict(final, condition="measurement", measurement_cost=None, gap=None)
    assert S.validate_record("final_battery", measurement).condition == "measurement"
    # measurement-set episodes, so only the battery's own conditions (hazard, dynamics) refuse it
    _refused("battery", dict(measurement, kind="battery"), "'hazard' or 'dynamics'")
    _refused("sensitivity_battery", dict(measurement, kind="sensitivity_battery"))


def test_evaluation_records_hold_the_ten_selection_checkpoints() -> None:
    good = good_records()["evaluation"]
    bad = copy.deepcopy(good)
    bad["selection"].pop()
    _refused("evaluation", bad)
    bad = copy.deepcopy(good)
    bad["selection"][0]["seed_set"] = "measurement"
    _refused("evaluation", bad)
    bad = copy.deepcopy(good)
    bad["selection"][0], bad["selection"][1] = bad["selection"][1], bad["selection"][0]
    _refused("evaluation", bad)
    _refused("evaluation", dict(good, final_step=good["final_step"] - R.CHECKPOINT_INTERVAL_STEPS))
    # Q-final-cost-set (Table 9.1): the final checkpoint on the measurement set, not the selection set
    bad = copy.deepcopy(good)
    bad["final"] = {k: v for k, v in copy.deepcopy(good["selection"][-1]).items() if k != "step"}
    assert bad["final"]["seed_set"] == "selection"
    _refused("evaluation", bad, "Q-final-cost-set")
    budgets = copy.deepcopy(good)
    budgets["final"]["budgets"] = [10.0] * R.EVAL_EPISODES
    assert S.validate_record("evaluation", budgets).final.budgets[0] == 10.0
    budgets["final"]["budgets"] = [10] * R.EVAL_EPISODES  # budgets are floats
    _refused("evaluation", budgets)


def test_continuation_records_check_their_gap_and_final_checkpoint() -> None:
    good = good_records()["continuation"]
    _refused("continuation", dict(good, gap=good["gap"] + 1.0))
    _refused("continuation", dict(good, step=good["step"] - R.CHECKPOINT_INTERVAL_STEPS))
    _refused("continuation", dict(good, condition="hazard"))
    _refused("continuation", dict(good, continuation_steps=0))


def test_zero_shot_records_hold_every_unseen_and_reference_budget() -> None:
    good = good_records()["zero_shot"]
    assert S.validate_record("zero_shot", good).sr_zero.keys() == set(R.UNSEEN_BUDGETS)
    bad = copy.deepcopy(good)
    bad["budgets"] = [b for b in bad["budgets"] if b["budget"] != 45.0]
    _refused("zero_shot", bad)
    bad = copy.deepcopy(good)
    bad["budgets"] = [b for b in bad["budgets"] if b["role"] != "reference"]
    _refused("zero_shot", bad)
    bad = copy.deepcopy(good)
    bad["budgets"][0]["budget"] = int(bad["budgets"][0]["budget"])  # 5, not 5.0
    _refused("zero_shot", bad)
    bad = copy.deepcopy(good)
    bad["budgets"][0]["satisfaction"] = min(1.0, bad["budgets"][0]["satisfaction"] + 0.01)  # equation (10)
    _refused("zero_shot", bad)
    bad = copy.deepcopy(good)
    bad["budgets"][0]["violation"] += 0.5
    _refused("zero_shot", bad)
    bad = copy.deepcopy(good)
    bad["budgets"][0]["distance"] += 1.0  # equation (11)
    _refused("zero_shot", bad)
    bad = copy.deepcopy(good)
    bad["arm"] = "Dense"  # Dense's reference budgets are its five levels
    _refused("zero_shot", bad)
    for seed_set in ("selection", "hazard"):  # Q-studyb-eval (Table 9.1): the measurement set
        _refused("zero_shot", dict(good, seed_set=seed_set), "Q-studyb-eval")


def test_zero_shot_satisfaction_is_inclusive_and_distance_is_equation_11() -> None:
    assert S._satisfaction([5.0, 6.0], 5.0) == 0.5  # "cost <= budget"
    assert S._violation([5.0, 7.0], 5.0) == 1.0
    assert S._training_set_distance("Single-10", 5.0) == 5.0
    assert S._training_set_distance("Dense", 30.0) == 2.5
    assert S._training_set_distance("Continuous", 30.0) == 0.0
    assert S._training_set_distance("Continuous", 45.0) == 5.0
    assert S.reference_budgets("Continuous") == tuple(R.CONTINUOUS_RANGE)
    assert S.reference_budgets("Moderate") == R.STUDY_B_ARMS["Moderate"]


def test_fewshot_records_encode_censoring_above_the_largest_horizon_of_that_continuation() -> None:
    good = good_records()["fewshot"]
    assert S.validate_record("fewshot", good).censored is False
    bad = dict(good, adapt_steps=good["adapt_steps"] + 1)
    _refused("fewshot", bad)
    bad = copy.deepcopy(good)
    bad["sr_fewshot"] = {k.replace(".0_", "_"): v for k, v in bad["sr_fewshot"].items()}  # '5_200000' is not the form
    _refused("fewshot", bad)
    # a training level, not an unseen budget (the record rebuilt consistently at 10.0, so only that rule breaks)
    blocks = [{**h, **syn._block_for_rate(10.0, h["satisfaction"])} for h in good["by_horizon"]]
    bad = dict(good, budget=10.0, by_horizon=blocks,
               sr_fewshot={S.fewshot_key(10.0, h["horizon"]): h["satisfaction"] for h in blocks})
    _refused("fewshot", bad, match="not an unseen budget")
    bad = dict(good, budget=5)
    _refused("fewshot", bad)
    # a continuation cut to 200,000 steps (Part 6.1) that never reaches the target: censored at 200,001
    cut = copy.deepcopy(good)
    first = cut["by_horizon"][0]
    low = syn._block_for_rate(cut["budget"], 0.2)
    cut.update(horizons=[R.FEWSHOT_HORIZONS[0]], by_horizon=[{**first, **low}],
               sr_fewshot={S.fewshot_key(cut["budget"], R.FEWSHOT_HORIZONS[0]): low["satisfaction"]},
               adapt_steps=R.FEWSHOT_HORIZONS[0] + 1)
    model = S.validate_record("fewshot", cut)
    assert model.censored is True and model.adapt_steps == 200_001
    _refused("fewshot", dict(cut, adapt_steps=max(R.FEWSHOT_HORIZONS) + 1))  # not the constant 1,000,001
    _refused("fewshot", dict(cut, horizons=[300_000]))  # not a registered horizon
    # Q-studyb-eval: Table 2.5's three horizons, or cut 3's first one; no other subset
    for horizons in ([R.FEWSHOT_HORIZONS[1]], [R.FEWSHOT_HORIZONS[0], R.FEWSHOT_HORIZONS[2]]):
        blocks = [b for b in good["by_horizon"] if b["horizon"] in horizons]
        rates = {b["horizon"]: b["satisfaction"] for b in blocks}
        reached = [h for h in horizons if rates[h] >= R.SATISFACTION_TARGET]
        bad = dict(good, horizons=horizons, by_horizon=blocks,
                   sr_fewshot={S.fewshot_key(good["budget"], h): rates[h] for h in horizons},
                   adapt_steps=reached[0] if reached else max(horizons) + 1)
        _refused("fewshot", bad, "cut 3")
    for seed_set in ("selection", "hazard"):  # Q-studyb-eval (Table 9.1): the measurement set
        _refused("fewshot", dict(good, seed_set=seed_set), "Q-studyb-eval")


def test_training_records_check_onset_rows_and_recovery() -> None:
    good = good_records()["training"]
    model = S.validate_record("training", good)
    assert model.plasticity_check.dormant_trainable == 0.3
    bad = copy.deepcopy(good)
    bad["plasticity_check"]["step"] += R.STEPS_PER_EPOCH
    _refused("training", bad)
    _refused("training", dict(good, recovery_censored=True))  # a censored recovery has no steps
    _refused("training", dict(good, overshoot=2.0))  # peak - final
    _refused("training", dict(good, onset_step=good["total_steps"] + R.STEPS_PER_EPOCH, recovery_steps=None,
                              recovery_censored=True, plasticity_check=None), match="onset_step lies after")
    bad = copy.deepcopy(good)
    bad["plasticity_check"]["dormant_trainable"] = 1.5  # a fraction
    _refused("training", bad)
    reference = dict(good, onset_step=0, recovery_steps=None, recovery_censored=None, plasticity_check=None)
    assert S.validate_record("training", reference).recovery_steps is None
    _refused("training", dict(reference, recovery_censored=False))  # recovery is for late arms only
    with_intervention = dict(good, intervention={
        "treatment": "injection", "step": good["onset_step"], "layers": ["mean.2", "mean.4"], "generator_seed": 7,
        "trainable_parameters_before": 100, "trainable_parameters_after": 100, "max_output_difference": 0.0})
    assert S.validate_record("training", with_intervention).intervention.treatment == "injection"
    bad = copy.deepcopy(with_intervention)
    bad["intervention"]["step"] = 0
    _refused("training", bad)


def test_training_records_carry_the_rate_clips_share_of_a_rate_limited_run() -> None:
    """Q-rate-limit (Table 9.1): "the report gives the share of constrained epochs in which the clip bound"; an
    optional fraction of a late run, absent from the JSON of every other record (whose bytes stay as they were)."""
    good = good_records()["training"]
    assert "rate_limit_clip_share" not in S.canonical_json("training", good)
    limited = dict(good, rate_limit_clip_share=0.2)
    text = S.canonical_json("training", limited)
    assert json.loads(text)["rate_limit_clip_share"] == 0.2 and S.canonical_json("training", json.loads(text)) == text
    for bad in (1.5, -0.1, True, "0.2", math.nan):
        _refused("training", dict(good, rate_limit_clip_share=bad))
    reference = dict(good, onset_step=0, recovery_steps=None, recovery_censored=None, plasticity_check=None)
    _refused("training", dict(reference, rate_limit_clip_share=0.2), match="late arm")


def test_fewshot_records_name_the_parent_checkpoint_they_adapted() -> None:
    # pilot/manifest.py: study_b_fewshot specs carry parent_step = parent.total_steps, the parent's final checkpoint,
    # from which Study B few-shot continuations start (Q-continuations, Table 9.1); the record names it.
    good = good_records()["fewshot"]
    assert S.validate_record("fewshot", good).parent_step == R.TOTAL_STEPS
    missing = copy.deepcopy(good)
    del missing["parent_step"]
    _refused("fewshot", missing)
    _refused("fewshot", dict(good, parent_step=-1))
    _refused("fewshot", dict(good, parent_step=float(R.TOTAL_STEPS)))  # a step is an integer


# ---------------------------------------------------------------------------
# The freeze (Q-ledger-v2, answered in Table 9.1)
# ---------------------------------------------------------------------------


def test_recorded_hash_matches_schema_file() -> None:
    """Q-ledger-v2: the supplement schema is frozen at version 1 with its SHA-256 in results/supplement_schema.sha256,
    as the ledger schema is (tests/test_ledger_schema.py): UTF-8 'hex  results/supplement_schema.py' with one LF, as
    `sha256sum -c` reads it from the repository root."""
    raw = (REPO_ROOT / "results" / "supplement_schema.sha256").read_bytes()
    assert raw.endswith(b"\n") and raw.count(b"\n") == 1
    assert b"\r" not in raw and not raw.startswith(b"\xef\xbb\xbf")
    digest, name = raw.decode("ascii").rstrip("\n").split("  ")
    assert name == "results/supplement_schema.py"
    assert hashlib.sha256((REPO_ROOT / name).read_bytes()).hexdigest() == digest
    assert S.SUPPLEMENT_SCHEMA_VERSION == 1


# The configs.registered values the supplement schema validates against, as recorded when the schema was frozen
# (Q-ledger-v2): the hash covers the schema's file only, so a change to one of these changes what the schema
# accepts without changing its hash. Such a change is an amendment of the supplement schema (and of
# configs/registered.py); update this snapshot in the same amendment.
REGISTERED_SNAPSHOT = {
    "CONTINUOUS_RANGE": (10.0, 40.0),
    "EPISODES_PER_LAYOUT": 5,
    "EVAL_EPISODES": 100,
    "FEWSHOT_HORIZONS": (200_000, 500_000, 1_000_000),
    "HAZARD_LAYOUTS": 20,
    "MANIPULATION_CHECK_STEPS_AFTER_ONSET": 200_000,
    "SATISFACTION_TARGET": 0.8,
    "SELECTION_WINDOW_CHECKPOINTS": 10,
    "STUDY_B_ARMS": {"Single-10": (10.0,), "Single-20": (20.0,), "Single-40": (40.0,), "Sparse": (10.0, 40.0),
                     "Moderate": (10.0, 20.0, 40.0), "Dense": (10.0, 17.5, 25.0, 32.5, 40.0), "Continuous": None},
    "UNSEEN_BUDGETS": (5.0, 15.0, 30.0, 45.0),
}


def test_the_registered_values_the_schema_reads_are_those_recorded_at_its_freeze() -> None:
    source = (REPO_ROOT / "results" / "supplement_schema.py").read_text(encoding="utf-8")
    assert set(re.findall(r"\bR\.([A-Za-z_][A-Za-z_0-9]*)", source)) == set(REGISTERED_SNAPSHOT)
    for name, value in REGISTERED_SNAPSHOT.items():
        current = getattr(R, name)
        assert (dict(current) if name == "STUDY_B_ARMS" else current) == value, name

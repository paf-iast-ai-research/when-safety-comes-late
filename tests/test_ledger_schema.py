"""
test_ledger_schema.py — Tests for the results ledger schema.

Every test that writes a ledger uses a temporary directory; no test writes
to the real ledger.
Run from the repository root with:

    pytest tests/test_ledger_schema.py -v
"""

# FROZEN at schema_version = 1 on 2026-09-26.
# Any change to the schema requires the corresponding test change in the
# same commit; both are covered by the same Part 9 amendment, and
# results/ledger_schema.sha256 is re-recorded with it. The corrections of
# 2026-10-01, decided 2026-10-02 (X-ledger-schema-amendment,
# docs/DECISIONS.md) and to be ratified with the other decisions, and those
# of 2026-10-02 that follow the decisions of that day are listed in
# docs/ledger_schema_memo.md, "Changes since the freeze".

from __future__ import annotations

import hashlib
import math
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import results.ledger_schema as schema
from results.ledger_schema import (
    CHECKPOINT_STRUCT,
    ENRICHMENT_FIELDS,
    LEDGER_SCHEMA,
    SCHEMA_VERSION,
    CheckpointRecord,
    LedgerRow,
    absolute_checkpoint_path,
    append_to_ledger,
    load_ledger,
    load_ledger_as_rows,
    relative_checkpoint_path,
    write_enrichment,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
STARTED = datetime(2026, 9, 26, 10, 0, 0, tzinfo=timezone.utc)
FINISHED = datetime(2026, 9, 26, 18, 0, 0, tzinfo=timezone.utc)
MATCHED_PATH = "PointGoal/seed0/step8400000.pt"


@pytest.fixture
def tmp_ledger(tmp_path: Path) -> Path:
    return tmp_path / "ledger.parquet"


def make_study_a_row(
    seed: int = 0,
    run_id: Optional[str] = None,
    N: float = 0.50,
    onset_step: Optional[int] = None,
) -> LedgerRow:
    """A Study A row for testing. Checkpoints are generated at every multiple
    of 200,000 plus the onset step if that is not already a multiple. The
    onset step follows N (N x 10,000,000 steps; N = 0 is onset at step 0)
    unless given.

    The selection window is the last ten checkpoints, (8M, 10M]. Before
    onset the multiplier is at its initial value, 0.001 (Table 2.1).
    """
    if onset_step is None:
        onset_step = int(round(N * 10_000_000))
    arm = f"{N:.2f}_abrupt_total_steps"
    if run_id is None:
        run_id = f"A_PointGoal_{arm}_seed{seed}"
    steps = sorted(set(range(0, 10_000_001, 200_000)) | {onset_step})
    return LedgerRow(
        run_id=run_id,
        study="A",
        task="SafetyPointGoal1-v0",
        arm=arm,
        N=N,
        onset_shape="abrupt",
        step_matching="total_steps",
        seed=seed,
        commit_hash="abc123def456",
        config_hash="cfg789",
        started=STARTED,
        finished=FINISHED,
        wall_clock_hours=8.0,
        machine="workstation-01",
        completed=True,
        checkpoints=[
            CheckpointRecord(
                path=f"PointGoal/seed{seed}/step{step}.pt",
                step=step,
                training_cost=30.0 + step / 1e6,
                training_return=20.0,
                dormant=0.05,
                rank=12.0,
                norm=3.5,
                selection_cost=25.0 if step > 8_000_000 else None,
                selection_return=22.0 if step > 8_000_000 else None,
                multiplier=0.42 if step >= onset_step else 0.001,
            )
            for step in steps
        ],
        multiplier_trace_path=f"PointGoal/seed{seed}/multiplier_trace.csv",
        final_cost=25.3,
        final_return=22.5,
        dormant_onset=0.03,
        rank_onset=14.0,
        norm_onset=3.2,
    )


# Few-shot rates of each unseen budget at 200,000, 500,000 and 1,000,000
# steps, and the adaptation steps they give (Q-adapt-censoring: the smallest
# horizon whose rate is at least 0.80, else the largest horizon + 1): 5.0
# reaches the target at the second horizon, 15.0 and 30.0 (exactly 0.80) at
# the first, and 45.0 never.
FEWSHOT_RATES = {
    5.0: (0.5, 0.85, 0.9),
    15.0: (0.82, 0.9, 0.95),
    30.0: (0.8, 0.85, 0.9),
    45.0: (0.3, 0.5, 0.7),
}
FEWSHOT_ADAPT_STEPS = {
    5.0: 500_000,
    15.0: 200_000,
    30.0: 200_000,
    45.0: 1_000_001,
}


def make_study_b_row(
    seed: int = 0,
    run_id: str = "B_Moderate_seed0",
) -> LedgerRow:
    return LedgerRow(
        run_id=run_id,
        study="B",
        task="SafetyPointGoal1-v0",
        arm="Moderate",
        training_levels=[10.0, 20.0, 40.0],
        seed=seed,
        commit_hash="abc123def456",
        config_hash="cfg789",
        started=STARTED,
        finished=FINISHED,
        wall_clock_hours=8.0,
        machine="workstation-01",
        completed=True,
        sr_zero={5.0: 0.42, 15.0: 0.61, 30.0: 0.78, 45.0: 0.88},
        sr_fewshot={
            f"{budget}_{horizon}": rate
            for budget, rates in FEWSHOT_RATES.items()
            for horizon, rate in zip((200_000, 500_000, 1_000_000), rates)
        },
        adapt_steps=dict(FEWSHOT_ADAPT_STEPS),
        per_level_multipliers={10.0: 0.42, 20.0: 0.61, 40.0: 0.88},
    )


def payload_a(**changes: Any) -> Dict[str, Any]:
    return {**make_study_a_row().model_dump(), **changes}


def payload_b(**changes: Any) -> Dict[str, Any]:
    return {**make_study_b_row().model_dump(), **changes}


def failed(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        **payload,
        "completed": False,
        "finished": None,
        "wall_clock_hours": None,
        "failure_cause": "crash",
    }


def matched_fields() -> Dict[str, Any]:
    return {
        "matched_checkpoint_path": MATCHED_PATH,
        "matched_checkpoint_step": 8_400_000,
        "training_age": 8_400_000,
    }


# ---------------------------------------------------------------------------
# Fixture and round-trip
# ---------------------------------------------------------------------------


def test_fixture_selection_window_is_the_last_ten_checkpoints() -> None:
    row = make_study_a_row()
    window = [cp.step for cp in row.checkpoints if cp.selection_cost]
    assert window == list(range(8_200_000, 10_000_001, 200_000))
    assert len(window) == 10


def test_round_trip_study_a(tmp_ledger: Path) -> None:
    row = make_study_a_row()
    append_to_ledger(row, tmp_ledger)

    df = load_ledger(tmp_ledger)
    assert len(df) == 1

    loaded = load_ledger_as_rows(tmp_ledger)[0]
    assert loaded == row
    assert loaded.checkpoints[0].step == 0
    assert loaded.checkpoints[-1].step == 10_000_000


def test_round_trip_study_b(tmp_ledger: Path) -> None:
    row = make_study_b_row()
    append_to_ledger(row, tmp_ledger)

    loaded = load_ledger_as_rows(tmp_ledger)[0]
    assert loaded == row
    assert loaded.sr_fewshot is not None and len(loaded.sr_fewshot) == 12


def test_round_trip_failed_run(tmp_ledger: Path) -> None:
    row = LedgerRow(**failed(payload_a()))
    append_to_ledger(row, tmp_ledger)
    assert load_ledger_as_rows(tmp_ledger)[0] == row


def test_untrained_checkpoint_may_hold_nan_training_cost(
    tmp_ledger: Path,
) -> None:
    payload = payload_a()
    payload["checkpoints"][0]["training_cost"] = math.nan
    payload["checkpoints"][0]["training_return"] = math.nan
    append_to_ledger(LedgerRow(**payload), tmp_ledger)
    loaded = load_ledger_as_rows(tmp_ledger)[0]
    assert math.isnan(loaded.checkpoints[0].training_cost)


def test_nested_checkpoints(tmp_ledger: Path) -> None:
    row = make_study_a_row()
    append_to_ledger(row, tmp_ledger)

    loaded = load_ledger_as_rows(tmp_ledger)[0]
    for cp in loaded.checkpoints:
        assert cp.path == f"PointGoal/seed0/step{cp.step}.pt"
        assert cp.training_cost == pytest.approx(30.0 + cp.step / 1e6)


def test_onset_checkpoint_at_non_multiple_step(tmp_ledger: Path) -> None:
    """Onset at N=0.25 lands on step 2,500,000, which is not a multiple of
    200,000. The schema must accept this."""
    row = make_study_a_row(N=0.25, onset_step=2_500_000)
    assert row.run_id == "A_PointGoal_0.25_abrupt_total_steps_seed0"
    append_to_ledger(row, tmp_ledger)

    loaded = load_ledger_as_rows(tmp_ledger)[0]
    steps = {cp.step for cp in loaded.checkpoints}
    assert 2_500_000 in steps
    assert len(steps) == 52  # 51 multiples of 200,000 plus onset


def test_float_keys_are_floats(tmp_ledger: Path) -> None:
    row = make_study_b_row()
    append_to_ledger(row, tmp_ledger)

    loaded = load_ledger_as_rows(tmp_ledger)[0]
    assert all(isinstance(k, float) for k in loaded.sr_zero)
    assert all(isinstance(k, float) for k in loaded.adapt_steps)
    assert all(isinstance(k, float) for k in loaded.per_level_multipliers)


def test_sr_fewshot_keys_are_strings(tmp_ledger: Path) -> None:
    row = make_study_b_row()
    append_to_ledger(row, tmp_ledger)

    loaded = load_ledger_as_rows(tmp_ledger)[0]
    assert loaded.sr_fewshot is not None
    assert all(isinstance(k, str) for k in loaded.sr_fewshot)


def test_load_ledger_normalises_maps_to_dicts_with_string_keys(
    tmp_ledger: Path,
) -> None:
    append_to_ledger(make_study_b_row(), tmp_ledger)
    df = load_ledger(tmp_ledger)
    assert df["sr_zero"][0] == {
        "5.0": 0.42,
        "15.0": 0.61,
        "30.0": 0.78,
        "45.0": 0.88,
    }
    assert isinstance(df["per_level_multipliers"][0], dict)


# ---------------------------------------------------------------------------
# Validation: categorical fields
# ---------------------------------------------------------------------------


def test_invalid_study() -> None:
    with pytest.raises(ValueError, match="study must be one of"):
        LedgerRow(**payload_a(study="C"))


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("task", "SafetyPointGoal2-v0", "task must be one of"),
        ("onset_shape", "linear", "onset_shape must be one of"),
        ("step_matching", "steps", "step_matching must be one of"),
        ("treatment", "shrink", "treatment must be one of"),
        ("controller_variant", "lqr", "controller_variant must be one of"),
        ("N", 0.3, "N must be one of"),
    ],
)
def test_invalid_categorical_value(
    field: str, value: Any, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        LedgerRow(**payload_a(**{field: value}))


def test_invalid_failure_cause() -> None:
    payload = failed(payload_a())
    payload["failure_cause"] = "timeout"
    with pytest.raises(ValueError, match="failure_cause must be one of"):
        LedgerRow(**payload)


def test_invalid_study_b_arm() -> None:
    with pytest.raises(ValueError, match="Study B arm must be one of"):
        LedgerRow(**payload_b(arm="Unknown"))


def test_study_a_arm_cannot_use_study_b_name() -> None:
    with pytest.raises(ValueError, match="cannot use a Study B arm name"):
        LedgerRow(**payload_a(arm="Moderate"))


@pytest.mark.parametrize(
    "field", ["run_id", "arm", "machine", "commit_hash", "config_hash"]
)
def test_empty_identifier_rejected(field: str) -> None:
    with pytest.raises(ValueError, match="at least 1 character"):
        LedgerRow(**payload_a(**{field: ""}))


# ---------------------------------------------------------------------------
# Validation: completion and time
# ---------------------------------------------------------------------------


def test_completed_requires_finished() -> None:
    with pytest.raises(ValueError, match="completed=True requires finished"):
        LedgerRow(**payload_a(finished=None))


def test_completed_requires_wall_clock_hours() -> None:
    with pytest.raises(ValueError, match="requires wall_clock_hours"):
        LedgerRow(**payload_a(wall_clock_hours=None))


def test_completed_cannot_have_failure_cause() -> None:
    with pytest.raises(ValueError, match="cannot have failure_cause"):
        LedgerRow(**payload_a(failure_cause="crash"))


def test_not_completed_requires_failure_cause() -> None:
    payload = failed(payload_a())
    payload["failure_cause"] = None
    with pytest.raises(
        ValueError, match="completed=False requires failure_cause"
    ):
        LedgerRow(**payload)


def test_finished_before_started_rejected() -> None:
    with pytest.raises(ValueError, match="is before started"):
        LedgerRow(**payload_a(finished=STARTED - timedelta(hours=1)))


def test_naive_datetime_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        LedgerRow(**payload_a(started=datetime(2026, 9, 26, 10, 0, 0)))


def test_non_utc_offset_rejected() -> None:
    plus_two = timezone(timedelta(hours=2))
    with pytest.raises(ValueError, match="must be in UTC"):
        LedgerRow(
            **payload_a(started=datetime(2026, 9, 26, 10, tzinfo=plus_two))
        )


def test_zero_offset_zone_is_normalised_to_utc() -> None:
    zero = timezone(timedelta(0), "GMT")
    started = datetime(2026, 9, 26, 10, tzinfo=zero)
    row = LedgerRow(**payload_a(started=started))
    assert row.started.tzinfo is timezone.utc
    assert row.started == STARTED


# ---------------------------------------------------------------------------
# Validation: numbers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field, value",
    [
        ("seed", True),
        ("N", False),
        ("schema_version", True),
        ("final_cost", True),
        ("seed", "1"),
        ("final_cost", "nan"),
        ("wall_clock_hours", "8"),
    ],
)
def test_bool_and_string_numbers_rejected(field: str, value: Any) -> None:
    with pytest.raises(ValueError, match="expected a number"):
        LedgerRow(**payload_a(**{field: value}))


@pytest.mark.parametrize(
    "field, value",
    [
        ("completed", "yes"),
        ("completed", 1),
        ("matched", 1),
        ("infeasible", "false"),
    ],
)
def test_non_boolean_flags_rejected(field: str, value: Any) -> None:
    with pytest.raises(ValueError, match="expected a boolean"):
        LedgerRow(**payload_a(**{field: value}))


def test_numpy_bool_accepted() -> None:
    import numpy as np

    row = LedgerRow(**payload_a(completed=np.bool_(True)))
    assert row.completed is True


@pytest.mark.parametrize(
    "changes",
    [{"started": 0}, {"finished": 10}, {"started": 1.5e9},
     {"started": "1790000000"}, {"finished": True}],
)
def test_numeric_datetimes_rejected(changes: Dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="expected a(n ISO 8601)? datetime"):
        LedgerRow(**payload_a(**changes))


def test_iso_datetime_string_accepted() -> None:
    row = LedgerRow(**payload_a(started="2026-09-26T10:00:00+00:00"))
    assert row.started == STARTED


def test_infinite_training_statistics_rejected() -> None:
    with pytest.raises(ValueError, match="finite number or NaN"):
        CheckpointRecord(path="a.pt", step=5, training_cost=math.inf,
                         training_return=1.0)
    with pytest.raises(ValueError, match="finite number or NaN"):
        CheckpointRecord(path="a.pt", step=5, training_cost=1.0,
                         training_return=-math.inf)
    # NaN: no finite statistic (step 0, no progress row, a failed run).
    cp = CheckpointRecord(path="a.pt", step=5, training_cost=math.nan,
                          training_return=math.nan)
    assert math.isnan(cp.training_cost)


def test_bool_checkpoint_step_rejected() -> None:
    with pytest.raises(ValueError, match="expected a number"):
        CheckpointRecord(
            path="a.pt", step=True, training_cost=1.0, training_return=1.0
        )


def test_schema_version_other_than_one_rejected() -> None:
    with pytest.raises(ValueError, match="Input should be 1"):
        LedgerRow(**payload_a(schema_version=2))


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize(
    "field",
    [
        "final_cost",
        "final_return",
        "wall_clock_hours",
        "measurement_cost",
        "selection_cost_at_match",
        "gap_hazard",
        "lambda_peak",
        "rank_onset",
    ],
)
def test_non_finite_values_rejected(field: str, value: float) -> None:
    with pytest.raises(ValueError, match="finite number"):
        LedgerRow(**payload_a(**{field: value}))


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize(
    "field", ["selection_cost", "selection_return", "multiplier", "rank"]
)
def test_non_finite_checkpoint_values_rejected(
    field: str, value: float
) -> None:
    with pytest.raises(ValueError, match="finite number"):
        CheckpointRecord(
            path="a.pt",
            step=0,
            training_cost=1.0,
            training_return=1.0,
            **{field: value},
        )


# Rates are bounded to [0, 1], and Pydantic reports the bound for a
# non-finite rate; the other values report "finite number".
RATE_BOUND = "greater than or equal to 0|less than or equal to 1"


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"sr_zero": {5.0: math.nan}}, RATE_BOUND),
        ({"sr_zero": {5.0: math.inf}}, RATE_BOUND),
        ({"sr_zero": {5.0: -math.inf}}, RATE_BOUND),
        ({"sr_fewshot": {"5.0_200000": math.nan}}, RATE_BOUND),
        ({"sr_fewshot": {"5.0_200000": math.inf}}, RATE_BOUND),
        ({"sr_fewshot": {"5.0_200000": -math.inf}}, RATE_BOUND),
        ({"per_level_multipliers": {10.0: math.inf}}, "finite number"),
        ({"per_level_multipliers": {10.0: -math.inf}}, "finite number"),
        ({"per_level_multipliers": {10.0: math.nan}}, "finite number"),
        # One level, which the Single-10 arm's check would accept were it
        # 10.0: only the non-finite value is refused.
        ({"arm": "Single-10", "training_levels": [math.inf],
          "per_level_multipliers": None}, "finite number"),
        ({"arm": "Single-10", "training_levels": [math.nan],
          "per_level_multipliers": None}, "finite number"),
    ],
)
def test_non_finite_map_values_and_levels_rejected(
    changes: Dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        LedgerRow(**payload_b(**changes))


def test_int_beyond_int64_rejected() -> None:
    with pytest.raises(
        ValueError, match="less than or equal to 9223372036854775807"
    ):
        LedgerRow(**payload_a(seed=2**70))


def test_fractional_int_rejected() -> None:
    with pytest.raises(ValueError, match="fractional part"):
        LedgerRow(**payload_a(settling_steps=1.5))


# ---------------------------------------------------------------------------
# Validation: study-specific fields
# ---------------------------------------------------------------------------


def test_study_a_requires_n() -> None:
    with pytest.raises(ValueError, match="Study A row requires N"):
        LedgerRow(**payload_a(N=None))


@pytest.mark.parametrize(
    "field, value",
    [
        ("training_levels", [10.0]),
        ("sr_zero", {5.0: 0.5}),
        ("sr_fewshot", {"5.0_200000": 0.5}),
        ("adapt_steps", {5.0: 200_000}),
        ("per_level_multipliers", {10.0: 0.5}),
    ],
)
def test_study_a_cannot_have_study_b_fields(field: str, value: Any) -> None:
    with pytest.raises(ValueError, match=f"Study A row cannot have {field}"):
        LedgerRow(**payload_a(**{field: value}))


@pytest.mark.parametrize(
    "field, value",
    [
        ("N", 0.5),
        ("onset_shape", "abrupt"),
        ("step_matching", "total_steps"),
        ("treatment", "reset"),
        ("controller_variant", "pid"),
    ],
)
def test_study_b_cannot_have_study_a_fields(field: str, value: Any) -> None:
    with pytest.raises(ValueError, match=f"Study B row cannot have {field}"):
        LedgerRow(**payload_b(**{field: value}))


@pytest.mark.parametrize(
    "field, value",
    [(f, 1.0 if f not in ("matched", "infeasible") else False)
     for f in schema.STUDY_A_ONLY_FIELDS
     if f not in ("matched_checkpoint_path", "matched_checkpoint_step",
                  "training_age", "settling_steps")]
    + [
        ("matched_checkpoint_path", "a.pt"),
        ("matched_checkpoint_step", 0),
        ("training_age", 0),
        ("settling_steps", 5),
    ],
)
def test_study_b_cannot_have_study_a_enrichment_fields(
    field: str, value: Any
) -> None:
    """The fields Table B.1 marks '(Study A)' and the added matched and
    infeasible (Part 4.1)."""
    with pytest.raises(ValueError, match=f"Study B row cannot have {field}"):
        LedgerRow(**payload_b(**{field: value}))


@pytest.mark.parametrize(
    "arm, levels",
    [
        ("Moderate", [1.0]),
        ("Moderate", [10.0, 10.0, 40.0]),
        ("Moderate", [10.0, 20.0, 40.0, 40.0]),
        ("Single-10", [-10.0]),
        ("Dense", [10.0, 20.0, 40.0]),
    ],
)
def test_training_levels_must_be_the_arms(arm: str, levels: Any) -> None:
    with pytest.raises(ValueError, match="are not the levels of arm"):
        LedgerRow(
            **payload_b(arm=arm, training_levels=levels,
                        per_level_multipliers=None)
        )


def test_training_levels_in_any_order_accepted() -> None:
    row = LedgerRow(**payload_b(training_levels=[40.0, 10.0, 20.0]))
    assert row.training_levels == [40.0, 10.0, 20.0]


def test_arm_training_levels_match_the_registered_design() -> None:
    from configs import registered as R

    assert dict(R.STUDY_B_ARMS) == schema.ARM_TRAINING_LEVELS
    assert set(schema.ARM_TRAINING_LEVELS) == schema.VALID_STUDY_B_ARMS
    assert set(R.STUDY_B_ARMS) == schema.VALID_STUDY_B_ARMS
    assert R.CONTINUOUS_RANGE == schema.CONTINUOUS_RANGE
    assert R.CONTINUOUS_BIN_WIDTH == schema.CONTINUOUS_BIN_WIDTH


def test_registered_values_match_the_registered_design() -> None:
    """The schema keeps its own copies of registered values (it imports
    nothing from the pipeline); they are those of configs/registered.py."""
    from configs import registered as R

    assert schema.VALID_TASKS == set(R.TASKS_STUDY_A) | set(R.TASKS_STUDY_B)
    assert schema.VALID_STUDY_A_NS == set(R.ONSET_FRACTIONS)
    assert schema.VALID_UNSEEN_BUDGETS == set(R.UNSEEN_BUDGETS)
    assert schema.VALID_FEWSHOT_HORIZONS == set(R.FEWSHOT_HORIZONS)
    assert schema.SATISFACTION_TARGET == R.SATISFACTION_TARGET
    # Q-adapt-censoring: the largest horizon + 1, of the registered horizons
    # or of Part 6.1 cut 3's one horizon (the first).
    assert schema.CENSORED_ADAPT_STEPS == {
        max(R.FEWSHOT_HORIZONS) + 1, R.FEWSHOT_HORIZONS[0] + 1
    }


def test_continuous_arm_has_no_training_levels() -> None:
    with pytest.raises(ValueError, match="Continuous arm has no fixed"):
        LedgerRow(
            **payload_b(arm="Continuous", per_level_multipliers=None)
        )


@pytest.mark.parametrize("key", [12.5, 5.0, 45.0, 40.0])
def test_continuous_multiplier_keys_are_bin_edges(key: float) -> None:
    """Q-continuous-bins: six bins [10, 15), ..., [35, 40], named by their
    lower edges; 40 lies in [35, 40] and names no bin."""
    with pytest.raises(ValueError, match="lower edge of a 5-unit bin"):
        LedgerRow(
            **payload_b(arm="Continuous", training_levels=None,
                        per_level_multipliers={key: 0.1})
        )


def test_continuous_multipliers_of_the_six_bins_accepted() -> None:
    edges = {10.0: 0.1, 15.0: 0.2, 20.0: 0.3, 25.0: 0.4, 30.0: 0.5, 35.0: 0.6}
    row = LedgerRow(
        **payload_b(arm="Continuous", training_levels=None,
                    per_level_multipliers=edges)
    )
    assert row.per_level_multipliers == edges


def test_study_b_requires_training_levels_except_continuous() -> None:
    with pytest.raises(ValueError, match="requires training_levels"):
        LedgerRow(**payload_b(training_levels=None))
    row = LedgerRow(
        **payload_b(
            arm="Continuous",
            training_levels=None,
            per_level_multipliers={10.0: 0.1, 15.0: 0.2},
        )
    )
    assert row.training_levels is None


def test_per_level_multiplier_keys_must_be_training_levels() -> None:
    with pytest.raises(ValueError, match="are not training levels"):
        LedgerRow(**payload_b(per_level_multipliers={99.0: 1.0}))


def test_non_finite_per_level_multiplier_key_rejected() -> None:
    with pytest.raises(ValueError, match="key must be finite"):
        LedgerRow(
            **payload_b(arm="Continuous", training_levels=None,
                        per_level_multipliers={"nan": 0.1})
        )


# ---------------------------------------------------------------------------
# Validation: maps
# ---------------------------------------------------------------------------


def test_invalid_sr_zero_budget() -> None:
    with pytest.raises(ValueError, match="not in unseen budgets"):
        LedgerRow(**payload_b(sr_zero={5.0: 0.4, 20.0: 0.5}))


def test_invalid_adapt_steps_budget() -> None:
    with pytest.raises(ValueError, match="not in unseen budgets"):
        LedgerRow(**payload_b(adapt_steps={20.0: 200_000}))


def test_negative_adapt_steps_rejected() -> None:
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        LedgerRow(**payload_b(adapt_steps={5.0: -1}))


@pytest.mark.parametrize("steps", [0, 123, 500_001, 1_000_000_000])
def test_adapt_steps_is_a_horizon_or_censored(steps: int) -> None:
    """Q-adapt-censoring: a few-shot horizon, or the largest horizon + 1
    (1,000,001, or 200,001 under Part 6.1 cut 3)."""
    with pytest.raises(ValueError, match="neither a few-shot horizon"):
        LedgerRow(**payload_b(sr_fewshot=None, adapt_steps={5.0: steps}))


@pytest.mark.parametrize("steps", [200_000, 500_000, 1_000_000, 200_001,
                                   1_000_001])
def test_adapt_steps_without_rates_accepted(steps: int) -> None:
    row = LedgerRow(**payload_b(sr_fewshot=None, adapt_steps={5.0: steps}))
    assert row.adapt_steps == {5.0: steps}


@pytest.mark.parametrize(
    "budget, steps",
    [
        (5.0, 200_000),  # 0.5 at 200,000: reached at 500,000
        (5.0, 1_000_001),  # reached, so not censored
        (15.0, 500_000),  # reached at the first horizon already
        (30.0, 1_000_001),  # 0.80 reaches the target ("at least 0.80")
        (45.0, 1_000_000),  # never reached: censored at 1,000,001
        (45.0, 200_001),  # censored above this continuation's horizons
    ],
)
def test_adapt_steps_follow_from_sr_fewshot(budget: float, steps: int) -> None:
    with pytest.raises(ValueError, match="does not follow from sr_fewshot"):
        LedgerRow(
            **payload_b(adapt_steps={**FEWSHOT_ADAPT_STEPS, budget: steps})
        )


def test_adapt_steps_of_a_cut_continuation() -> None:
    """Part 6.1 cut 3 leaves one horizon, 200,000: censored at 200,001."""
    row = LedgerRow(
        **payload_b(
            sr_fewshot={"5.0_200000": 0.5, "15.0_200000": 0.9},
            adapt_steps={5.0: 200_001, 15.0: 200_000},
        )
    )
    assert row.adapt_steps == {5.0: 200_001, 15.0: 200_000}
    with pytest.raises(ValueError, match="does not follow from sr_fewshot"):
        LedgerRow(
            **payload_b(sr_fewshot={"5.0_200000": 0.5},
                        adapt_steps={5.0: 1_000_001})
        )


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_satisfaction_rates_bounded(value: float) -> None:
    message = "less than or equal to 1" if value > 1 else "greater than or"
    with pytest.raises(ValueError, match=message):
        LedgerRow(**payload_b(sr_zero={5.0: value}))
    with pytest.raises(ValueError, match=message):
        LedgerRow(**payload_b(sr_fewshot={"5.0_200000": value}))


def test_invalid_sr_fewshot_horizon() -> None:
    with pytest.raises(ValueError, match="horizon must be one of"):
        LedgerRow(**payload_b(sr_fewshot={"5.0_999999": 0.5}))


def test_invalid_sr_fewshot_key_format() -> None:
    with pytest.raises(ValueError, match="must be 'budget_horizon'"):
        LedgerRow(**payload_b(sr_fewshot={"5.0": 0.5}))


@pytest.mark.parametrize(
    "key", ["20.0_200000", "nan_200000", "inf_200000", "-3.0_500000"]
)
def test_sr_fewshot_budget_must_be_unseen(key: str) -> None:
    with pytest.raises(ValueError, match="not in unseen budgets"):
        LedgerRow(**payload_b(sr_fewshot={key: 0.5}))


@pytest.mark.parametrize(
    "key", ["5_200000", "5.00_200000", "5.0_+200000", " 5.0_ 200000"]
)
def test_sr_fewshot_key_must_be_canonical(key: str) -> None:
    with pytest.raises(ValueError, match="canonical spelling"):
        LedgerRow(**payload_b(sr_fewshot={key: 0.5}))


def test_string_keys_for_float_fields_are_accepted() -> None:
    """A dict constructed with string keys is coerced to float keys."""
    row = LedgerRow(
        **payload_b(
            sr_zero={"5.0": 0.42, "15.0": 0.61, "30.0": 0.78, "45.0": 0.88}
        )
    )
    assert row.sr_zero == {5.0: 0.42, 15.0: 0.61, 30.0: 0.78, 45.0: 0.88}


def test_list_of_tuples_for_float_fields_are_accepted() -> None:
    """A list of (key, value) tuples, as PyArrow may return, is coerced."""
    row = LedgerRow(
        **payload_b(
            sr_zero=[
                ("5.0", 0.42),
                ("15.0", 0.61),
                ("30.0", 0.78),
                ("45.0", 0.88),
            ]
        )
    )
    assert row.sr_zero == {5.0: 0.42, 15.0: 0.61, 30.0: 0.78, 45.0: 0.88}


@pytest.mark.parametrize(
    "value",
    [
        [1, 2, 3],
        [(5.0, 0.1, 9)],
        # One dict of two entries: unpacking it would take its two keys as
        # the key and the value.
        [{5.0: 0.1, 45.0: 0.2}],
        ["ab"],
    ],
)
def test_malformed_list_map_is_a_validation_error(value: Any) -> None:
    for field in ("sr_zero", "sr_fewshot", "adapt_steps",
                  "per_level_multipliers"):
        with pytest.raises(
            ValueError, match="list of \\(key, value\\) pairs"
        ):
            LedgerRow(**payload_b(**{field: value}))


@pytest.mark.parametrize("field", ["sr_zero", "sr_fewshot", "adapt_steps"])
def test_unhashable_list_map_key_is_a_validation_error(field: str) -> None:
    """A raw TypeError would escape Pydantic and the callers' ValueError."""
    with pytest.raises(ValueError, match="is not hashable"):
        LedgerRow(**payload_b(**{field: [([5.0], 0.1)]}))


@pytest.mark.parametrize(
    "field, value",
    [
        ("per_level_multipliers", {b"1_5": 0.1}),
        ("sr_zero", {b"5.0": 0.5}),
        ("sr_zero", [(b" 15.0", 0.1)]),
        ("sr_fewshot", {b"5.0_200000": 0.5}),
    ],
)
def test_bytes_map_key_rejected(field: str, value: Any) -> None:
    """float() reads b'1_5' as 15.0, past the str spelling checks."""
    with pytest.raises(ValueError, match="must not be bytes"):
        LedgerRow(**payload_b(**{field: value}))


@pytest.mark.parametrize(
    "value", [{"5": 0.1, "5.0": 0.9}, [("5", 0.1), ("5.0", 0.9)]]
)
def test_keys_equal_after_coercion_rejected(value: Any) -> None:
    with pytest.raises(ValueError, match="equal after float coercion"):
        LedgerRow(**payload_b(sr_zero=value))


def test_duplicate_key_in_list_of_pairs_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate map key"):
        LedgerRow(**payload_b(sr_zero=[("5.0", 0.1), ("5.0", 0.9)]))


@pytest.mark.parametrize("key", ["1_5", " 5.0", "5.0 ", ""])
def test_float_key_spelling_with_underscore_or_space_rejected(
    key: str,
) -> None:
    """float() would read '1_5' as 15.0."""
    with pytest.raises(ValueError, match="keys must be numeric"):
        LedgerRow(**payload_b(sr_zero={key: 0.5}))


def test_bool_map_key_rejected() -> None:
    with pytest.raises(ValueError, match="keys must be numeric"):
        LedgerRow(**payload_b(per_level_multipliers={True: 0.3}))


# ---------------------------------------------------------------------------
# Validation: checkpoints, matching and paths
# ---------------------------------------------------------------------------


def test_checkpoint_steps_must_increase() -> None:
    payload = payload_a()
    payload["checkpoints"].append(payload["checkpoints"][-1])
    with pytest.raises(ValueError, match="strictly increasing"):
        LedgerRow(**payload)


def test_matched_checkpoint_is_one_of_the_checkpoints() -> None:
    assert LedgerRow(**payload_a(**matched_fields())).training_age == 8_400_000
    with pytest.raises(ValueError, match="is not a checkpoint step"):
        LedgerRow(
            **payload_a(
                **{**matched_fields(), "matched_checkpoint_step": 123,
                   "training_age": 123}
            )
        )
    with pytest.raises(ValueError, match="is not the path of the checkpoint"):
        LedgerRow(
            **payload_a(**{**matched_fields(),
                           "matched_checkpoint_path": "nope.pt"})
        )
    with pytest.raises(ValueError, match="are set together"):
        LedgerRow(**payload_a(matched_checkpoint_step=8_400_000))


def test_training_age_equals_matched_step() -> None:
    with pytest.raises(ValueError, match="training_age"):
        LedgerRow(**payload_a(**{**matched_fields(), "training_age": 1}))


def test_matched_and_infeasible_exclusive() -> None:
    with pytest.raises(ValueError, match="cannot both be True"):
        LedgerRow(
            **payload_a(**matched_fields(), matched=True, infeasible=True)
        )


def test_matched_requires_a_matched_checkpoint() -> None:
    with pytest.raises(ValueError, match="matched=True requires"):
        LedgerRow(**payload_a(matched=True))
    assert LedgerRow(**payload_a(**matched_fields(), matched=True)).matched
    # Unmatched or infeasible rows need no matched checkpoint.
    assert LedgerRow(**payload_a(matched=False, infeasible=True)).infeasible


@pytest.mark.parametrize(
    "field, value",
    [
        ("training_age", 8_400_000),
        ("selection_cost_at_match", 24.8),
        ("measurement_cost", 25.1),
        ("lambda_at_selection", 0.39),
    ],
)
def test_selection_quantities_require_a_matched_checkpoint(
    field: str, value: Any
) -> None:
    with pytest.raises(ValueError, match=f"{field} requires matched"):
        LedgerRow(**payload_a(**{field: value}))


@pytest.mark.parametrize(
    "field", ["gap_hazard", "gap_dynamics", "gap_finetune", "gap_transfer"]
)
def test_gaps_require_a_matched_checkpoint(field: str) -> None:
    """Table B.1: 'Robustness gaps of the matched checkpoint (Study A)'."""
    with pytest.raises(ValueError, match=f"{field} requires matched"):
        LedgerRow(**payload_a(**{field: 3.0}))
    assert getattr(
        LedgerRow(**payload_a(**matched_fields(), **{field: 3.0})), field
    ) == 3.0


def test_matched_checkpoint_must_be_in_the_selection_window() -> None:
    """Part 4.1 rule 1: one of the last ten checkpoints, which alone carry
    a selection cost."""
    with pytest.raises(ValueError, match="not in the selection window"):
        LedgerRow(
            **payload_a(
                matched_checkpoint_path="PointGoal/seed0/step0.pt",
                matched_checkpoint_step=0,
                training_age=0,
                matched=True,
            )
        )


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("selection_cost_at_match", 999.0, "is not the selection_cost"),
        ("lambda_at_selection", 0.39, "is not the multiplier"),
    ],
)
def test_selection_quantities_are_the_matched_checkpoints(
    field: str, value: float, message: str
) -> None:
    """selection_cost_at_match is the chosen checkpoint's selection cost;
    lambda_at_selection is copied from its multiplier (memo decision 6)."""
    with pytest.raises(ValueError, match=message):
        LedgerRow(**payload_a(**matched_fields(), **{field: value}))
    row = LedgerRow(
        **payload_a(**matched_fields(), selection_cost_at_match=25.0,
                    lambda_at_selection=0.42)
    )
    assert row.lambda_at_selection == 0.42


def test_lambda_at_selection_needs_the_checkpoints_multiplier() -> None:
    payload = payload_a(**matched_fields(), lambda_at_selection=0.42)
    for cp in payload["checkpoints"]:
        if cp["step"] == 8_400_000:
            cp["multiplier"] = None
    with pytest.raises(ValueError, match="is not the multiplier"):
        LedgerRow(**payload)


@pytest.mark.parametrize(
    "fields",
    [
        matched_fields(),
        {"matched": False},
        {"infeasible": True},
        {"gap_hazard": 3.0},
        {"gap_transfer": 1.0},
    ],
)
def test_failed_run_cannot_carry_matching_results(
    fields: Dict[str, Any],
) -> None:
    """A failed run is excluded and repeated with the next seed (Part 5);
    it never enters the matched comparison."""
    with pytest.raises(ValueError, match="completed=False cannot have"):
        LedgerRow(**failed(payload_a(**fields)))


def test_failed_study_b_run_cannot_carry_evaluations() -> None:
    with pytest.raises(ValueError, match="completed=False cannot have sr_"):
        LedgerRow(**failed(payload_b()))
    row = LedgerRow(
        **failed(payload_b(sr_zero=None, sr_fewshot=None, adapt_steps=None))
    )
    assert row.per_level_multipliers is not None


@pytest.mark.parametrize("path", [".", "/", " ", "//"])
def test_stored_path_must_name_a_file(path: str) -> None:
    with pytest.raises(ValueError, match="must name a file|must not be empty"):
        CheckpointRecord(
            path=path, step=0, training_cost=1.0, training_return=1.0
        )
    with pytest.raises(ValueError, match="must name a file|must not be empty"):
        LedgerRow(**payload_a(multiplier_trace_path=path))


@pytest.mark.parametrize(
    "path, message",
    [
        ("", "must not be empty"),
        ("C:\\a\\b.pt", "POSIX separators"),
        ("a/../../etc/x.pt", "must not contain '..'"),
        ("/data/checkpoints/../x", "must not contain '..'"),
        ("a/b/", "normal POSIX form"),
        ("a//b.pt", "normal POSIX form"),
        ("./a.pt", "normal POSIX form"),
        ("a/./b.pt", "normal POSIX form"),
        ("a.pt ", "surrounding whitespace"),
        (" a.pt", "surrounding whitespace"),
        ("a\x00.pt", "NUL character"),
        ("//data/checkpoints/x.pt", "normal POSIX form"),
    ],
)
def test_invalid_stored_paths_rejected(path: str, message: str) -> None:
    """A stored path is in the normal form relative_checkpoint_path writes,
    so that one file has one spelling (matched_checkpoint_path is compared
    with the checkpoint's path as a string)."""
    with pytest.raises(ValueError, match=message):
        CheckpointRecord(
            path=path, step=0, training_cost=1.0, training_return=1.0
        )
    with pytest.raises(ValueError, match=message):
        LedgerRow(**payload_a(multiplier_trace_path=path))


@pytest.mark.parametrize(
    "path", ["a.pt", "a/b.pt", "/other/x.pt", "PointGoal/seed0/step0.pt"]
)
def test_normal_form_paths_accepted(path: str) -> None:
    assert relative_checkpoint_path(path) == path
    assert CheckpointRecord(
        path=path, step=0, training_cost=1.0, training_return=1.0
    ).path == path
    assert LedgerRow(
        **payload_a(multiplier_trace_path=path)
    ).multiplier_trace_path == path


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def test_schema_version_present(tmp_ledger: Path) -> None:
    append_to_ledger(make_study_a_row(), tmp_ledger)

    df = load_ledger(tmp_ledger)
    assert (df["schema_version"] == SCHEMA_VERSION).all()


def test_append_and_duplicate(tmp_ledger: Path) -> None:
    row1 = make_study_a_row(seed=0, run_id="A_seed0")
    row2 = make_study_a_row(seed=1, run_id="A_seed1")

    append_to_ledger(row1, tmp_ledger)
    append_to_ledger(row2, tmp_ledger)

    df = load_ledger(tmp_ledger)
    assert len(df) == 2
    assert set(df["run_id"]) == {"A_seed0", "A_seed1"}

    before = tmp_ledger.read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        append_to_ledger(row1, tmp_ledger)
    assert tmp_ledger.read_bytes() == before


def test_append_revalidates_a_mutated_row(tmp_ledger: Path) -> None:
    append_to_ledger(make_study_a_row(), tmp_ledger)
    before = tmp_ledger.read_bytes()
    row = make_study_a_row(seed=1)
    row.seed = -5
    row.notes = None  # type: ignore[assignment]
    with pytest.raises(ValueError, match="seed|notes"):
        append_to_ledger(row, tmp_ledger)
    assert tmp_ledger.read_bytes() == before
    assert len(load_ledger_as_rows(tmp_ledger)) == 1


def test_refused_append_creates_no_directories(tmp_path: Path) -> None:
    ledger = tmp_path / "new" / "dir" / "ledger.parquet"
    row = make_study_a_row()
    row.seed = -5
    with pytest.raises(ValueError, match="seed"):
        append_to_ledger(row, ledger)
    assert not (tmp_path / "new").exists()


def test_nanosecond_timestamp_rejected() -> None:
    """The timestamp('us') column would truncate it: the stored row would
    not be the validated one."""
    with pytest.raises(ValueError, match="microsecond precision"):
        LedgerRow(
            **payload_a(started=pd.Timestamp("2026-09-26T10:00:00.000000001Z"))
        )


def test_microsecond_pandas_timestamp_round_trips(tmp_ledger: Path) -> None:
    row = LedgerRow(
        **payload_a(started=pd.Timestamp("2026-09-26T10:00:00.000001Z"))
    )
    assert type(row.started) is datetime
    append_to_ledger(row, tmp_ledger)
    assert load_ledger_as_rows(tmp_ledger)[0] == row


def test_reading_the_umask_leaves_it_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On Linux the umask is read from /proc/self/status, without the
    process-wide os.umask(...) round trip another thread could observe."""
    if not Path("/proc/self/status").exists():
        pytest.skip("no /proc/self/status")
    old = os.umask(0o027)
    try:
        def no_umask(_: int) -> int:
            raise AssertionError("os.umask called")

        monkeypatch.setattr(schema.os, "umask", no_umask)
        assert schema._current_umask() == 0o027
    finally:
        monkeypatch.undo()
        os.umask(old)


def test_append_rejects_schema_mismatch(tmp_ledger: Path) -> None:
    pq.write_table(pa.table({"run_id": ["x"]}), tmp_ledger)
    with pytest.raises(ValueError, match="does not match LEDGER_SCHEMA"):
        append_to_ledger(make_study_a_row(), tmp_ledger)


def test_writes_are_atomic(
    tmp_ledger: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash mid-write leaves the previous ledger whole and no temp file."""
    append_to_ledger(make_study_a_row(), tmp_ledger)
    before = tmp_ledger.read_bytes()

    def crash(table: pa.Table, where: Any, **kw: Any) -> None:
        where.write(b"PAR1 truncated")
        raise OSError("disk full")

    monkeypatch.setattr(schema.pq, "write_table", crash)
    with pytest.raises(OSError):
        append_to_ledger(make_study_a_row(seed=1), tmp_ledger)
    with pytest.raises(OSError):
        write_enrichment(
            make_study_a_row().run_id,
            {**matched_fields(), "gap_hazard": 1.0},
            tmp_ledger,
        )
    monkeypatch.undo()
    assert tmp_ledger.read_bytes() == before
    assert [p.name for p in tmp_ledger.parent.iterdir()] == [tmp_ledger.name]
    assert len(load_ledger_as_rows(tmp_ledger)) == 1


def test_pyarrow_schema_matches_pydantic_model() -> None:
    pydantic_fields = set(LedgerRow.model_fields.keys())
    pyarrow_fields = {field.name for field in LEDGER_SCHEMA}
    assert pydantic_fields == pyarrow_fields, (
        f"Fields only in Pydantic: {pydantic_fields - pyarrow_fields}; "
        f"fields only in PyArrow: {pyarrow_fields - pydantic_fields}"
    )
    struct_fields = [CHECKPOINT_STRUCT.field(i).name
                     for i in range(CHECKPOINT_STRUCT.num_fields)]
    assert struct_fields == list(CheckpointRecord.model_fields)


def full_study_a_payload() -> Dict[str, Any]:
    """A Study A row with every field a Study A row may hold set."""
    return payload_a(
        **matched_fields(),
        treatment="reset",
        controller_variant="pid",
        selection_cost_at_match=25.0,
        measurement_cost=25.1,
        lambda_at_selection=0.42,
        matched=True,
        infeasible=False,
        gap_hazard=1.0,
        gap_dynamics=2.0,
        gap_finetune=3.0,
        gap_transfer=4.0,
        lambda_peak=0.5,
        lambda_final=0.4,
        settling_steps=200_000,
        notes="n",
    )


def test_pyarrow_types_match_round_trip(tmp_ledger: Path) -> None:
    """Every field survives with its type: a Study A row with every field a
    Study A row may hold (checkpoint structs included) and a Study B row
    with every map."""
    rows = [
        LedgerRow(**full_study_a_payload()),
        LedgerRow(**payload_b(notes="n", multiplier_trace_path="B/t.csv",
                              final_cost=1.0, final_return=2.0)),
    ]
    a = rows[0].model_dump()
    # failure_cause is None in a completed row; the rest are Study B's.
    assert {f for f in a if a[f] is None} == {
        "training_levels", "failure_cause", *schema.ALL_MAP_FIELDS
    }
    assert all(
        v is not None
        for cp in a["checkpoints"]
        for k, v in cp.items()
        if k not in ("selection_cost", "selection_return")
    )
    b = rows[1].model_dump()
    assert all(b[f] is not None for f in ("training_levels",
                                          *schema.ALL_MAP_FIELDS))
    for row in rows:
        append_to_ledger(row, tmp_ledger)
    assert pq.read_table(tmp_ledger).schema == LEDGER_SCHEMA
    assert load_ledger_as_rows(tmp_ledger) == rows


def test_new_ledger_mode_follows_the_umask(tmp_ledger: Path) -> None:
    """The atomic write keeps the mode a plain write would give (mkstemp's
    0o600 would hide the ledger from other accounts)."""
    old = os.umask(0o022)
    try:
        append_to_ledger(make_study_a_row(), tmp_ledger)
    finally:
        os.umask(old)
    if os.name == "posix":
        assert tmp_ledger.stat().st_mode & 0o777 == 0o644


def test_rewrite_keeps_the_ledger_mode(tmp_ledger: Path) -> None:
    append_to_ledger(make_study_a_row(), tmp_ledger)
    os.chmod(tmp_ledger, 0o640)
    append_to_ledger(make_study_a_row(seed=1), tmp_ledger)
    write_enrichment(make_study_a_row().run_id,
                     {**matched_fields(), "gap_hazard": 1.0}, tmp_ledger)
    if os.name == "posix":
        assert tmp_ledger.stat().st_mode & 0o777 == 0o640


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def test_loaders_raise_file_not_found(tmp_ledger: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_ledger(tmp_ledger)
    with pytest.raises(FileNotFoundError):
        load_ledger_as_rows(tmp_ledger)


def _rewrite(tmp_ledger: Path, table: pa.Table) -> None:
    pq.write_table(table, tmp_ledger)


def test_loaders_reject_missing_column(tmp_ledger: Path) -> None:
    append_to_ledger(make_study_a_row(), tmp_ledger)
    _rewrite(tmp_ledger, pq.read_table(tmp_ledger).drop(["notes"]))
    with pytest.raises(ValueError, match="does not match LEDGER_SCHEMA"):
        load_ledger_as_rows(tmp_ledger)
    with pytest.raises(ValueError, match="does not match LEDGER_SCHEMA"):
        load_ledger(tmp_ledger)


def test_loaders_reject_changed_column_type(tmp_ledger: Path) -> None:
    append_to_ledger(make_study_a_row(), tmp_ledger)
    table = pq.read_table(tmp_ledger)
    i = table.schema.get_field_index("seed")
    table = table.set_column(
        i, pa.field("seed", pa.float64()), table["seed"].cast(pa.float64())
    )
    _rewrite(tmp_ledger, table)
    with pytest.raises(ValueError, match="does not match LEDGER_SCHEMA"):
        load_ledger_as_rows(tmp_ledger)


def test_load_rejects_duplicate_run_ids(tmp_ledger: Path) -> None:
    append_to_ledger(make_study_a_row(), tmp_ledger)
    table = pq.read_table(tmp_ledger)
    _rewrite(tmp_ledger, pa.concat_tables([table, table]))
    with pytest.raises(ValueError, match="more than once"):
        load_ledger_as_rows(tmp_ledger)


def test_load_names_the_invalid_row(tmp_ledger: Path) -> None:
    row = make_study_a_row()
    append_to_ledger(row, tmp_ledger)
    records = pq.read_table(tmp_ledger).to_pylist()
    records[0]["seed"] = -1
    _rewrite(tmp_ledger, pa.Table.from_pylist(records, schema=LEDGER_SCHEMA))
    with pytest.raises(ValueError, match=re.escape(row.run_id)):
        load_ledger_as_rows(tmp_ledger)


# ---------------------------------------------------------------------------
# Enrichment
# ---------------------------------------------------------------------------


def test_nullable_enrichment(tmp_ledger: Path) -> None:
    row = make_study_a_row()
    append_to_ledger(row, tmp_ledger)

    loaded = load_ledger_as_rows(tmp_ledger)[0]
    assert loaded.matched_checkpoint_path is None
    assert loaded.gap_hazard is None
    assert loaded.lambda_at_selection is None

    write_enrichment(
        row.run_id,
        {
            **matched_fields(),
            "selection_cost_at_match": 25.0,
            "measurement_cost": 25.1,
            "lambda_at_selection": 0.42,
            "matched": True,
            "infeasible": False,
            "gap_hazard": 7.2,
            "gap_dynamics": 3.1,
        },
        tmp_ledger,
    )

    loaded = load_ledger_as_rows(tmp_ledger)[0]
    assert loaded.matched_checkpoint_path == MATCHED_PATH
    assert loaded.matched_checkpoint_step == 8_400_000
    assert loaded.training_age == 8_400_000
    assert loaded.selection_cost_at_match == 25.0
    assert loaded.measurement_cost == pytest.approx(25.1)
    assert loaded.lambda_at_selection == 0.42
    assert loaded.matched is True
    assert loaded.infeasible is False
    assert loaded.gap_hazard == pytest.approx(7.2)
    assert loaded.gap_dynamics == pytest.approx(3.1)
    assert loaded.notes == row.notes


def test_enrichment_rejects_raw_field(tmp_ledger: Path) -> None:
    row = make_study_a_row()
    append_to_ledger(row, tmp_ledger)
    with pytest.raises(ValueError, match="Invalid enrichment fields"):
        write_enrichment(row.run_id, {"final_cost": 99.0}, tmp_ledger)
    loaded = load_ledger_as_rows(tmp_ledger)[0]
    assert loaded.final_cost == pytest.approx(25.3)


def test_notes_is_not_an_enrichment_field(tmp_ledger: Path) -> None:
    assert "notes" not in ENRICHMENT_FIELDS
    payload = payload_a(notes="train_hours=8.0")
    append_to_ledger(LedgerRow(**payload), tmp_ledger)
    for value in ("overwritten", None):
        with pytest.raises(ValueError, match="Invalid enrichment fields"):
            write_enrichment(payload["run_id"], {"notes": value}, tmp_ledger)
    assert load_ledger_as_rows(tmp_ledger)[0].notes == "train_hours=8.0"


def test_enrichment_preserves_schema(tmp_ledger: Path) -> None:
    row = make_study_b_row()
    append_to_ledger(row, tmp_ledger)

    write_enrichment(
        row.run_id,
        {"sr_zero": {5.0: 0.50, 15.0: 0.65, 30.0: 0.80, 45.0: 0.90}},
        tmp_ledger,
    )

    table = pq.read_table(tmp_ledger)
    assert table.schema == LEDGER_SCHEMA
    loaded = load_ledger_as_rows(tmp_ledger)[0]
    assert loaded.sr_zero == {5.0: 0.50, 15.0: 0.65, 30.0: 0.80, 45.0: 0.90}


def test_enrichment_unknown_run_id_and_missing_ledger(
    tmp_ledger: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        write_enrichment("x", {"gap_hazard": 1.0}, tmp_ledger)
    append_to_ledger(make_study_a_row(), tmp_ledger)
    with pytest.raises(KeyError, match="not found"):
        write_enrichment("x", {"gap_hazard": 1.0}, tmp_ledger)


def test_enrichment_rejects_schema_mismatch(tmp_ledger: Path) -> None:
    pq.write_table(pa.table({"run_id": ["x"]}), tmp_ledger)
    with pytest.raises(ValueError, match="does not match LEDGER_SCHEMA"):
        write_enrichment("x", {"gap_hazard": 1.0}, tmp_ledger)


@pytest.mark.parametrize(
    "enrichment, message",
    [
        ({"lambda_at_selection": -1.0}, "greater than or equal to 0"),
        ({"dormant_onset": 1.5}, "less than or equal to 1"),
        ({"gap_hazard": math.nan}, "finite number"),
        ({"lambda_peak": True}, "boolean"),
        ({"settling_steps": 2.5}, "fractional part"),
        ({"sr_zero": {20.0: 0.5}}, "not in unseen budgets"),
        ({"sr_zero": {5.0: 1.5}}, "less than or equal to 1"),
        ({"sr_zero": {"5": 0.1, "5.0": 0.9}}, "equal after float coercion"),
        ({"sr_zero": [1, 2, 3]}, "list of \\(key, value\\) pairs"),
        ({"sr_fewshot": {"5_200000": 0.5}}, "canonical spelling"),
        ({"adapt_steps": {5.0: 200_000}}, "does not follow from sr_fewshot"),
        ({**matched_fields(), "matched": True, "infeasible": True},
         "cannot both be True"),
        ({"matched_checkpoint_step": 8_400_000}, "are set together"),
        ({**matched_fields(), "matched_checkpoint_step": 8_400_000.5},
         "fractional part"),
        ({**matched_fields(), "training_age": 1}, "training_age"),
        ({**matched_fields(), "selection_cost_at_match": 1.0},
         "is not the selection_cost"),
    ],
)
def test_enrichment_validates_the_row(
    tmp_ledger: Path, enrichment: Dict[str, Any], message: str
) -> None:
    """An invalid value is refused and the ledger is left unchanged, so it
    never makes load_ledger_as_rows fail for the whole ledger."""
    study_b = bool(set(enrichment) & {"sr_zero", "sr_fewshot", "adapt_steps"})
    row = make_study_b_row() if study_b else make_study_a_row()
    append_to_ledger(row, tmp_ledger)
    before = tmp_ledger.read_bytes()
    with pytest.raises(ValueError, match=message):
        write_enrichment(row.run_id, enrichment, tmp_ledger)
    assert tmp_ledger.read_bytes() == before
    assert load_ledger_as_rows(tmp_ledger)[0] == row


def test_enrichment_canonicalises_map_keys(tmp_ledger: Path) -> None:
    row = make_study_b_row()
    append_to_ledger(row, tmp_ledger)
    write_enrichment(
        row.run_id,
        {"sr_zero": {5: 0.5, "15": 0.6, 30.00: 0.7, 45.0: 0.8}},
        tmp_ledger,
    )
    on_disk = pq.read_table(tmp_ledger).column("sr_zero").to_pylist()[0]
    assert [k for k, _ in on_disk] == ["5.0", "15.0", "30.0", "45.0"]


def test_enrichment_accepts_list_shaped_map(tmp_ledger: Path) -> None:
    row = make_study_b_row()
    append_to_ledger(row, tmp_ledger)
    write_enrichment(row.run_id, {"sr_zero": [(5.0, 0.5)]}, tmp_ledger)
    assert load_ledger_as_rows(tmp_ledger)[0].sr_zero == {5.0: 0.5}


def test_enrichment_input_not_mutated(tmp_ledger: Path) -> None:
    row = make_study_b_row()
    append_to_ledger(row, tmp_ledger)
    enrichment = {"sr_zero": {5.0: 0.5}, "adapt_steps": {5.0: 500_000}}
    copy = {"sr_zero": {5.0: 0.5}, "adapt_steps": {5.0: 500_000}}
    write_enrichment(row.run_id, enrichment, tmp_ledger)
    assert enrichment == copy


@pytest.mark.parametrize(
    "enrichment",
    [
        {"gap_hazard": 1.0},
        {"lambda_peak": 2.0},
        {"dormant_onset": 0.2},
        {"matched": False},
    ],
)
def test_enrichment_refuses_study_a_fields_on_a_study_b_row(
    tmp_ledger: Path, enrichment: Dict[str, Any]
) -> None:
    row = make_study_b_row()
    append_to_ledger(row, tmp_ledger)
    before = tmp_ledger.read_bytes()
    with pytest.raises(ValueError, match="Study B row cannot have"):
        write_enrichment(row.run_id, enrichment, tmp_ledger)
    assert tmp_ledger.read_bytes() == before


def test_enrichment_refuses_matching_a_failed_run(tmp_ledger: Path) -> None:
    row = LedgerRow(**failed(payload_a()))
    append_to_ledger(row, tmp_ledger)
    with pytest.raises(ValueError, match="completed=False cannot have"):
        write_enrichment(
            row.run_id,
            {**matched_fields(), "gap_hazard": 3.0, "matched": True},
            tmp_ledger,
        )
    assert load_ledger_as_rows(tmp_ledger)[0] == row


def test_enrichment_refuses_duplicate_run_ids(tmp_ledger: Path) -> None:
    row = make_study_a_row()
    append_to_ledger(row, tmp_ledger)
    table = pq.read_table(tmp_ledger)
    _rewrite(tmp_ledger, pa.concat_tables([table, table]))
    before = tmp_ledger.read_bytes()
    with pytest.raises(ValueError, match="more than once"):
        write_enrichment(row.run_id, {"gap_hazard": 1.0}, tmp_ledger)
    assert tmp_ledger.read_bytes() == before


def test_enrichment_names_an_invalid_stored_row(tmp_ledger: Path) -> None:
    row = make_study_a_row()
    append_to_ledger(row, tmp_ledger)
    records = pq.read_table(tmp_ledger).to_pylist()
    records[0]["seed"] = -1
    _rewrite(tmp_ledger, pa.Table.from_pylist(records, schema=LEDGER_SCHEMA))
    with pytest.raises(ValueError, match=re.escape(f"{row.run_id!r} is not")):
        write_enrichment(row.run_id, {"gap_hazard": 1.0}, tmp_ledger)


def test_enrichment_leaves_other_rows_unchanged(tmp_ledger: Path) -> None:
    rows = [make_study_a_row(seed=s) for s in (0, 1)]
    for row in rows:
        append_to_ledger(row, tmp_ledger)
    write_enrichment(
        rows[0].run_id, {**matched_fields(), "gap_hazard": 2.0}, tmp_ledger
    )
    loaded = load_ledger_as_rows(tmp_ledger)
    assert loaded[0].gap_hazard == 2.0
    assert loaded[1] == rows[1]


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def test_relative_checkpoint_path() -> None:
    abs_path = "/data/checkpoints/PointGoal/seed0/step0.pt"
    assert relative_checkpoint_path(abs_path) == "PointGoal/seed0/step0.pt"
    assert absolute_checkpoint_path("PointGoal/seed0/step0.pt") == abs_path


def test_relative_checkpoint_path_outside_root() -> None:
    abs_path = "/other/location/file.pt"
    assert relative_checkpoint_path(abs_path) == abs_path


def test_relative_checkpoint_path_normalises() -> None:
    assert relative_checkpoint_path("a//b/") == "a/b"


@pytest.mark.parametrize(
    "path, message",
    [
        ("/data/checkpoints/../etc/x.pt", "must not contain '..'"),
        ("/data/checkpoints", "is the checkpoint root"),
        ("/data/checkpoints/", "is the checkpoint root"),
        ("", "must name a file"),
        (".", "must name a file"),
        ("/", "must name a file"),
        ("//data/checkpoints/x.pt", "must not start with '//'"),
    ],
)
def test_relative_checkpoint_path_refuses_escape_and_root(
    path: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        relative_checkpoint_path(path)


@pytest.mark.parametrize(
    "path, message",
    [
        ("../etc/x", "must not contain '..'"),
        ("a/../../x.pt", "must not contain '..'"),
        ("", "must name a file"),
        (".", "must name a file"),
        ("/", "must name a file"),
        ("//data/checkpoints/x.pt", "must not start with '//'"),
    ],
)
def test_absolute_checkpoint_path_refuses_escape_and_root(
    path: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        absolute_checkpoint_path(path)


# ---------------------------------------------------------------------------
# Recorded hash
# ---------------------------------------------------------------------------


def test_recorded_hash_matches_schema_file() -> None:
    """results/ledger_schema.sha256 is UTF-8 'hex  results/ledger_schema.py'
    with one LF, as `sha256sum -c` reads it from the repository root."""
    raw = (REPO_ROOT / "results" / "ledger_schema.sha256").read_bytes()
    assert raw.endswith(b"\n") and raw.count(b"\n") == 1
    assert b"\r" not in raw and not raw.startswith(b"\xef\xbb\xbf")
    digest, name = raw.decode("ascii").rstrip("\n").split("  ")
    assert name == "results/ledger_schema.py"
    source = (REPO_ROOT / "results" / "ledger_schema.py").read_bytes()
    assert hashlib.sha256(source).hexdigest() == digest
    memo = (REPO_ROOT / "docs" / "ledger_schema_memo.md").read_text("utf-8")
    assert digest in memo

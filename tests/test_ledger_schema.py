"""
test_ledger_schema.py — Tests for the results ledger schema.

Every test uses a temporary directory; no test writes to the real ledger.
Run with:

    pytest test_ledger_schema.py -v
"""

# FROZEN at schema_version = 1 on 2026-09-26.
# Any change to the schema requires the corresponding test change in the
# same commit, and both are covered by the same amendment.



from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from results.ledger_schema import (
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


@pytest.fixture
def tmp_ledger(tmp_path: Path) -> Path:
    return tmp_path / "ledger.parquet"


def make_study_a_row(
    seed: int = 0,
    run_id: str = "A_PointGoal_0.50_abrupt_total_steps_seed0",
    N: float = 0.50,
    onset_step: int = 5_000_000,
) -> LedgerRow:
    """A Study A row for testing. Checkpoints are generated at every multiple
    of 200,000 plus the onset step if that is not already a multiple."""
    steps = sorted(set(range(0, 10_000_001, 200_000)) | {onset_step})
    return LedgerRow(
        run_id=run_id,
        study="A",
        task="SafetyPointGoal1-v0",
        arm=f"{N}_abrupt_total_steps",
        N=N,
        onset_shape="abrupt",
        step_matching="total_steps",
        seed=seed,
        commit_hash="abc123def456",
        config_hash="cfg789",
        started=datetime(2026, 9, 26, 10, 0, 0, tzinfo=timezone.utc),
        finished=datetime(2026, 9, 26, 18, 0, 0, tzinfo=timezone.utc),
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
                selection_cost=25.0 if step >= 8_000_000 else None,
                selection_return=22.0 if step >= 8_000_000 else None,
                multiplier=0.42,
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
        started=datetime(2026, 9, 26, 10, 0, 0, tzinfo=timezone.utc),
        finished=datetime(2026, 9, 26, 18, 0, 0, tzinfo=timezone.utc),
        wall_clock_hours=8.0,
        machine="workstation-01",
        completed=True,
        sr_zero={5.0: 0.42, 15.0: 0.61, 30.0: 0.78, 45.0: 0.88},
        sr_fewshot={
            "5.0_200000": 0.55,
            "5.0_500000": 0.68,
            "5.0_1000000": 0.79,
            "15.0_200000": 0.72,
            "15.0_500000": 0.80,
            "15.0_1000000": 0.85,
        },
        adapt_steps={5.0: 500_000, 15.0: 200_000, 30.0: 200_000, 45.0: 200_000},
        per_level_multipliers={10.0: 0.42, 20.0: 0.61, 40.0: 0.88},
    )


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_round_trip_study_a(tmp_ledger: Path) -> None:
    row = make_study_a_row()
    append_to_ledger(row, tmp_ledger)

    df = load_ledger(tmp_ledger)
    assert len(df) == 1

    loaded = load_ledger_as_rows(tmp_ledger)[0]
    assert loaded.run_id == row.run_id
    assert loaded.study == "A"
    assert loaded.N == row.N
    assert loaded.seed == row.seed
    assert loaded.completed is True
    assert loaded.checkpoints[0].step == 0
    assert loaded.checkpoints[-1].step == 10_000_000
    assert loaded.dormant_onset == pytest.approx(0.03)
    assert loaded.final_cost == pytest.approx(25.3)


def test_round_trip_study_b(tmp_ledger: Path) -> None:
    row = make_study_b_row()
    append_to_ledger(row, tmp_ledger)

    loaded = load_ledger_as_rows(tmp_ledger)[0]
    assert loaded.study == "B"
    assert loaded.arm == "Moderate"
    assert loaded.training_levels == [10.0, 20.0, 40.0]
    assert loaded.sr_zero == {5.0: 0.42, 15.0: 0.61, 30.0: 0.78, 45.0: 0.88}
    assert loaded.sr_fewshot is not None
    assert loaded.sr_fewshot["5.0_200000"] == pytest.approx(0.55)
    assert loaded.sr_fewshot["15.0_1000000"] == pytest.approx(0.85)
    assert loaded.adapt_steps == {
        5.0: 500_000,
        15.0: 200_000,
        30.0: 200_000,
        45.0: 200_000,
    }
    assert loaded.per_level_multipliers == {10.0: 0.42, 20.0: 0.61, 40.0: 0.88}


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
    append_to_ledger(row, tmp_ledger)

    loaded = load_ledger_as_rows(tmp_ledger)[0]
    steps = {cp.step for cp in loaded.checkpoints}
    assert 2_500_000 in steps
    assert len(steps) == 52  # 51 multiples of 200,000 plus onset


def test_float_keys_are_floats(tmp_ledger: Path) -> None:
    row = make_study_b_row()
    append_to_ledger(row, tmp_ledger)

    loaded = load_ledger_as_rows(tmp_ledger)[0]
    assert all(isinstance(k, float) for k in loaded.sr_zero.keys())
    assert all(isinstance(k, float) for k in loaded.adapt_steps.keys())
    assert all(isinstance(k, float) for k in loaded.per_level_multipliers.keys())


def test_sr_fewshot_keys_are_strings(tmp_ledger: Path) -> None:
    row = make_study_b_row()
    append_to_ledger(row, tmp_ledger)

    loaded = load_ledger_as_rows(tmp_ledger)[0]
    assert loaded.sr_fewshot is not None
    assert all(isinstance(k, str) for k in loaded.sr_fewshot.keys())


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_invalid_study() -> None:
    payload = make_study_a_row().model_dump()
    payload["study"] = "C"
    with pytest.raises(ValueError, match="study must be one of"):
        LedgerRow(**payload)


def test_invalid_failure_cause() -> None:
    payload = make_study_a_row().model_dump()
    payload["completed"] = False
    payload["finished"] = None
    payload["wall_clock_hours"] = None
    payload["failure_cause"] = "timeout"
    with pytest.raises(ValueError, match="failure_cause must be one of"):
        LedgerRow(**payload)


def test_invalid_study_b_arm() -> None:
    payload = make_study_b_row().model_dump()
    payload["arm"] = "Unknown"
    with pytest.raises(ValueError, match="Study B arm must be one of"):
        LedgerRow(**payload)


def test_completed_requires_finished() -> None:
    payload = make_study_a_row().model_dump()
    payload["finished"] = None
    with pytest.raises(ValueError, match="completed=True requires finished"):
        LedgerRow(**payload)


def test_not_completed_requires_failure_cause() -> None:
    payload = make_study_a_row().model_dump()
    payload["completed"] = False
    payload["finished"] = None
    payload["wall_clock_hours"] = None
    with pytest.raises(ValueError, match="completed=False requires failure_cause"):
        LedgerRow(**payload)


def test_invalid_sr_zero_budget() -> None:
    payload = make_study_b_row().model_dump()
    payload["sr_zero"] = {5.0: 0.4, 20.0: 0.5}
    with pytest.raises(ValueError, match="not in unseen budgets"):
        LedgerRow(**payload)


def test_invalid_sr_fewshot_horizon() -> None:
    payload = make_study_b_row().model_dump()
    payload["sr_fewshot"] = {"5.0_999999": 0.5}
    with pytest.raises(ValueError, match="horizon must be one of"):
        LedgerRow(**payload)


def test_invalid_sr_fewshot_key_format() -> None:
    payload = make_study_b_row().model_dump()
    payload["sr_fewshot"] = {"5.0": 0.5}
    with pytest.raises(ValueError, match="must be 'budget_horizon'"):
        LedgerRow(**payload)


def test_naive_datetime_rejected() -> None:
    payload = make_study_a_row().model_dump()
    payload["started"] = datetime(2026, 9, 26, 10, 0, 0)
    with pytest.raises(ValueError, match="timezone-aware"):
        LedgerRow(**payload)


def test_string_keys_for_float_fields_are_accepted() -> None:
    """A dict constructed with string keys is coerced to float keys."""
    row = LedgerRow(
        **{
            **make_study_b_row().model_dump(),
            "sr_zero": {"5.0": 0.42, "15.0": 0.61, "30.0": 0.78, "45.0": 0.88},
        }
    )
    assert row.sr_zero == {5.0: 0.42, 15.0: 0.61, 30.0: 0.78, 45.0: 0.88}


def test_list_of_tuples_for_float_fields_are_accepted() -> None:
    """A list of (key, value) tuples, as PyArrow may return, is coerced."""
    row = LedgerRow(
        **{
            **make_study_b_row().model_dump(),
            "sr_zero": [("5.0", 0.42), ("15.0", 0.61), ("30.0", 0.78), ("45.0", 0.88)],
        }
    )
    assert row.sr_zero == {5.0: 0.42, 15.0: 0.61, 30.0: 0.78, 45.0: 0.88}


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def test_schema_version_present(tmp_ledger: Path) -> None:
    row = make_study_a_row()
    append_to_ledger(row, tmp_ledger)

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

    with pytest.raises(ValueError, match="already exists"):
        append_to_ledger(row1, tmp_ledger)


def test_pyarrow_schema_matches_pydantic_model() -> None:
    pydantic_fields = set(LedgerRow.model_fields.keys())
    pyarrow_fields = {field.name for field in LEDGER_SCHEMA}
    assert pydantic_fields == pyarrow_fields, (
        f"Fields only in Pydantic: {pydantic_fields - pyarrow_fields}; "
        f"fields only in PyArrow: {pyarrow_fields - pydantic_fields}"
    )


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
            "matched_checkpoint_path": "PointGoal/seed0/step8400000.pt",
            "matched_checkpoint_step": 8_400_000,
            "training_age": 8_400_000,
            "selection_cost_at_match": 24.8,
            "measurement_cost": 25.1,
            "lambda_at_selection": 0.39,
            "matched": True,
            "infeasible": False,
            "gap_hazard": 7.2,
            "gap_dynamics": 3.1,
        },
        tmp_ledger,
    )

    loaded = load_ledger_as_rows(tmp_ledger)[0]
    assert loaded.matched_checkpoint_path == "PointGoal/seed0/step8400000.pt"
    assert loaded.matched_checkpoint_step == 8_400_000
    assert loaded.training_age == 8_400_000
    assert loaded.selection_cost_at_match == pytest.approx(24.8)
    assert loaded.measurement_cost == pytest.approx(25.1)
    assert loaded.lambda_at_selection == pytest.approx(0.39)
    assert loaded.matched is True
    assert loaded.infeasible is False
    assert loaded.gap_hazard == pytest.approx(7.2)
    assert loaded.gap_dynamics == pytest.approx(3.1)


def test_enrichment_rejects_raw_field(tmp_ledger: Path) -> None:
    row = make_study_a_row()
    append_to_ledger(row, tmp_ledger)
    with pytest.raises(ValueError, match="Invalid enrichment fields"):
        write_enrichment(row.run_id, {"final_cost": 99.0}, tmp_ledger)
    loaded = load_ledger_as_rows(tmp_ledger)[0]
    assert loaded.final_cost == pytest.approx(25.3)


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
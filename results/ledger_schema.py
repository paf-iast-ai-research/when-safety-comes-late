"""
ledger_schema.py — Results ledger schema for the When Safety Comes Late /
Constraint-Coverage Generalization project.

One row per finished run. Frozen at version 1 before the pilot. Any change
after the pilot is an amendment in Part 9 and a new schema version.

Design decisions (see companion memo):
  1. Storage: Parquet.
  2. Nested data: structs and lists inside the row.
  3. Enrichment fields: same row, nullable initially.
  4. Location: results/ledger.parquet in the repository.
  5. Schema versioning: schema_version field, Literal[1].
  6. lambda_at_selection: most recent epoch at or before the checkpoint step.

Checkpoint cadence. Checkpoints are saved every 200,000 steps, at onset,
and at the end of training. Onset at N=0.25 lands on step 2,500,000, which
is not a multiple of 200,000; the schema stores any non-negative integer step.

Parquet MAP key limitation. Parquet MAP keys must be STRING. Float-keyed
dict fields use string keys on disk. A mode="before" validator normalises
both dict and list-of-tuples shapes on read.

Path portability. Checkpoint paths stored in the ledger are POSIX strings,
not host-native paths, so the ledger is portable across operating systems.
Use PurePosixPath for anything stored in or read from the ledger. Use Path
only for local filesystem operations on the ledger file itself.

Write concurrency. The pilot owner is the sole writer. Concurrent appends
would race on the read-concat-rewrite and are not supported.
"""

# FROZEN at schema_version = 1 on 2026-09-26.
# Any change to this file requires an amendment recorded in Part 9 of the
# pre-registration and a bump of SCHEMA_VERSION.



from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Literal, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION: int = 1
LEDGER_PATH: str = "results/ledger.parquet"
CHECKPOINT_ROOT: str = "/data/checkpoints"

VALID_STUDIES = {"A", "B"}
VALID_TASKS = {"SafetyPointGoal1-v0", "SafetyCarGoal1-v0", "SafetyPointButton1-v0"}
VALID_ONSET_SHAPES = {"abrupt", "ramp"}
VALID_STEP_MATCHING = {"constrained_steps", "total_steps"}
VALID_TREATMENTS = {"reset", "injection", "additional_constrained"}
VALID_CONTROLLERS = {"warm_started", "rate_limited", "pid"}
VALID_FAILURE_CAUSES = {
    "crash",
    "incomplete",
    "non_finite_loss",
    "non_finite_multiplier",
}
VALID_STUDY_A_NS = {0.0, 0.10, 0.25, 0.50}
VALID_STUDY_B_ARMS = {
    "Single-10",
    "Single-20",
    "Single-40",
    "Sparse",
    "Moderate",
    "Dense",
    "Continuous",
}
VALID_UNSEEN_BUDGETS = {5.0, 15.0, 30.0, 45.0}
VALID_FEWSHOT_HORIZONS = {200_000, 500_000, 1_000_000}

FLOAT_KEYED_MAP_FIELDS = ("sr_zero", "adapt_steps", "per_level_multipliers")
STRING_KEYED_MAP_FIELDS = ("sr_fewshot",)
ALL_MAP_FIELDS = FLOAT_KEYED_MAP_FIELDS + STRING_KEYED_MAP_FIELDS


# ---------------------------------------------------------------------------
# Module-level helper (not a Pydantic validator; safe to call directly)
# ---------------------------------------------------------------------------


def _normalise_map(v: Any, *, coerce_keys_to_float: bool) -> Any:
    """Normalise a Parquet map value to a Python dict.

    PyArrow returns map columns as either a dict or a list of (key, value)
    tuples depending on the version and code path. Both shapes are accepted
    here. When ``coerce_keys_to_float`` is True, the keys are converted to
    float; otherwise they are left as-is.

    Returns None for None; returns the input unchanged if it is neither a
    dict nor a list (Pydantic will then raise the appropriate type error).
    """
    if v is None:
        return v
    if isinstance(v, list):
        v = dict(v)
    if not isinstance(v, dict):
        return v
    if not coerce_keys_to_float:
        return v
    out: Dict[float, Any] = {}
    for k, val in v.items():
        try:
            out[float(k)] = val
        except (TypeError, ValueError) as exc:
            raise ValueError(f"keys must be numeric, got {k!r}") from exc
    return out


# ---------------------------------------------------------------------------
# Nested models
# ---------------------------------------------------------------------------


class CheckpointRecord(BaseModel):
    """One saved checkpoint of a run."""

    model_config = ConfigDict(extra="forbid")

    path: str
    step: int = Field(ge=0)
    training_cost: float
    training_return: float
    dormant: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    rank: Optional[float] = Field(default=None, ge=0.0)
    norm: Optional[float] = Field(default=None, ge=0.0)
    selection_cost: Optional[float] = None
    selection_return: Optional[float] = None
    multiplier: Optional[float] = Field(default=None, ge=0.0)


# ---------------------------------------------------------------------------
# Main ledger row
# ---------------------------------------------------------------------------


class LedgerRow(BaseModel):
    """One row of the results ledger. One row per finished run."""

    model_config = ConfigDict(extra="forbid")

    # Identity and provenance
    schema_version: Literal[1] = Field(default=SCHEMA_VERSION)
    run_id: str
    study: str
    task: str
    arm: str
    N: Optional[float] = None
    onset_shape: Optional[str] = None
    step_matching: Optional[str] = None
    treatment: Optional[str] = None
    controller_variant: Optional[str] = None
    training_levels: Optional[List[float]] = None
    seed: int = Field(ge=0)
    commit_hash: str
    config_hash: str
    started: datetime
    finished: Optional[datetime] = None
    wall_clock_hours: Optional[float] = Field(default=None, ge=0.0)
    machine: str

    # Completion
    completed: bool
    failure_cause: Optional[str] = None

    # Checkpoints and multiplier trace
    checkpoints: List[CheckpointRecord] = Field(default_factory=list)
    multiplier_trace_path: Optional[str] = None

    # Final evaluation
    final_cost: Optional[float] = None
    final_return: Optional[float] = None

    # Matching rule enrichment
    matched_checkpoint_path: Optional[str] = None
    matched_checkpoint_step: Optional[int] = Field(default=None, ge=0)
    training_age: Optional[int] = Field(default=None, ge=0)
    selection_cost_at_match: Optional[float] = None
    measurement_cost: Optional[float] = None
    lambda_at_selection: Optional[float] = Field(default=None, ge=0.0)
    matched: Optional[bool] = None
    infeasible: Optional[bool] = None

    # Robustness gaps
    gap_hazard: Optional[float] = None
    gap_dynamics: Optional[float] = None
    gap_finetune: Optional[float] = None
    gap_transfer: Optional[float] = None

    # Plasticity at onset
    dormant_onset: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    rank_onset: Optional[float] = Field(default=None, ge=0.0)
    norm_onset: Optional[float] = Field(default=None, ge=0.0)

    # Controller quantities
    lambda_peak: Optional[float] = Field(default=None, ge=0.0)
    lambda_final: Optional[float] = Field(default=None, ge=0.0)
    settling_steps: Optional[int] = Field(default=None, ge=0)

    # Study B satisfaction
    sr_zero: Optional[Dict[float, float]] = None
    sr_fewshot: Optional[Dict[str, float]] = None
    adapt_steps: Optional[Dict[float, int]] = None
    per_level_multipliers: Optional[Dict[float, float]] = None

    notes: str = ""

    # -- Field validators ---------------------------------------------------

    @field_validator("study")
    @classmethod
    def _v_study(cls, v: str) -> str:
        if v not in VALID_STUDIES:
            raise ValueError(f"study must be one of {VALID_STUDIES}, got {v!r}")
        return v

    @field_validator("task")
    @classmethod
    def _v_task(cls, v: str) -> str:
        if v not in VALID_TASKS:
            raise ValueError(f"task must be one of {VALID_TASKS}, got {v!r}")
        return v

    @field_validator("onset_shape")
    @classmethod
    def _v_onset_shape(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in VALID_ONSET_SHAPES:
            raise ValueError(
                f"onset_shape must be one of {VALID_ONSET_SHAPES}, got {v!r}"
            )
        return v

    @field_validator("step_matching")
    @classmethod
    def _v_step_matching(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in VALID_STEP_MATCHING:
            raise ValueError(
                f"step_matching must be one of {VALID_STEP_MATCHING}, got {v!r}"
            )
        return v

    @field_validator("treatment")
    @classmethod
    def _v_treatment(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in VALID_TREATMENTS:
            raise ValueError(
                f"treatment must be one of {VALID_TREATMENTS}, got {v!r}"
            )
        return v

    @field_validator("controller_variant")
    @classmethod
    def _v_controller(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in VALID_CONTROLLERS:
            raise ValueError(
                f"controller_variant must be one of {VALID_CONTROLLERS}, got {v!r}"
            )
        return v

    @field_validator("failure_cause")
    @classmethod
    def _v_failure_cause(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in VALID_FAILURE_CAUSES:
            raise ValueError(
                f"failure_cause must be one of {VALID_FAILURE_CAUSES}, got {v!r}"
            )
        return v

    @field_validator("N")
    @classmethod
    def _v_N(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v not in VALID_STUDY_A_NS:
            raise ValueError(
                f"N must be one of {sorted(VALID_STUDY_A_NS)}; got {v!r}"
            )
        return v

    @field_validator("started", "finished")
    @classmethod
    def _v_utc(cls, v: Optional[datetime]) -> Optional[datetime]:
        if v is None:
            return v
        if v.tzinfo is None:
            raise ValueError("datetime must be timezone-aware (UTC)")
        if v.utcoffset() != timezone.utc.utcoffset(None):
            raise ValueError(f"datetime must be in UTC, got offset {v.utcoffset()}")
        return v

    @field_validator(*ALL_MAP_FIELDS, mode="before")
    @classmethod
    def _v_map_shape(cls, v: Any, info: ValidationInfo) -> Any:
        """Normalise map columns read from Parquet.

        PyArrow's to_pylist returns map columns as a dict or as a list of
        (key, value) tuples depending on the version. Both are normalised to
        dict. Float-keyed fields have their keys coerced to float; string-
        keyed fields are left untouched.
        """
        return _normalise_map(
            v, coerce_keys_to_float=info.field_name in FLOAT_KEYED_MAP_FIELDS
        )

    @field_validator("sr_zero", "adapt_steps", mode="after")
    @classmethod
    def _v_unseen_budget_keys(cls, v: Any) -> Any:
        """sr_zero and adapt_steps are indexed by unseen budgets only."""
        if v is None:
            return v
        for k in v:
            if k not in VALID_UNSEEN_BUDGETS:
                raise ValueError(
                    f"key {k} not in unseen budgets {sorted(VALID_UNSEEN_BUDGETS)}"
                )
        return v

    @field_validator("sr_fewshot")
    @classmethod
    def _v_sr_fewshot_keys(
        cls, v: Optional[Dict[str, float]]
    ) -> Optional[Dict[str, float]]:
        if v is None:
            return v
        for key in v:
            if not isinstance(key, str) or "_" not in key:
                raise ValueError(
                    f"sr_fewshot key must be 'budget_horizon', got {key!r}"
                )
            budget_str, horizon_str = key.rsplit("_", 1)
            try:
                float(budget_str)
            except ValueError as exc:
                raise ValueError(
                    f"sr_fewshot budget must be numeric, got {budget_str!r}"
                ) from exc
            try:
                horizon = int(horizon_str)
            except ValueError as exc:
                raise ValueError(
                    f"sr_fewshot horizon must be int, got {horizon_str!r}"
                ) from exc
            if horizon not in VALID_FEWSHOT_HORIZONS:
                raise ValueError(
                    f"sr_fewshot horizon must be one of "
                    f"{sorted(VALID_FEWSHOT_HORIZONS)}, got {horizon}"
                )
        return v

    # -- Model validators ---------------------------------------------------

    @model_validator(mode="after")
    def _v_arm_for_study(self) -> "LedgerRow":
        if self.study == "B" and self.arm not in VALID_STUDY_B_ARMS:
            raise ValueError(
                f"Study B arm must be one of {sorted(VALID_STUDY_B_ARMS)}, "
                f"got {self.arm!r}"
            )
        if self.study == "A" and self.arm in VALID_STUDY_B_ARMS:
            raise ValueError(
                f"Study A arm cannot use a Study B arm name, got {self.arm!r}"
            )
        return self

    @model_validator(mode="after")
    def _v_completion(self) -> "LedgerRow":
        if self.completed:
            if self.finished is None:
                raise ValueError("completed=True requires finished")
            if self.wall_clock_hours is None:
                raise ValueError("completed=True requires wall_clock_hours")
            if self.failure_cause is not None:
                raise ValueError("completed=True cannot have failure_cause")
        else:
            if self.failure_cause is None:
                raise ValueError("completed=False requires failure_cause")
        return self


# ---------------------------------------------------------------------------
# Parquet schema (frozen)
# ---------------------------------------------------------------------------

CHECKPOINT_STRUCT = pa.struct(
    [
        ("path", pa.string()),
        ("step", pa.int64()),
        ("training_cost", pa.float64()),
        ("training_return", pa.float64()),
        ("dormant", pa.float64()),
        ("rank", pa.float64()),
        ("norm", pa.float64()),
        ("selection_cost", pa.float64()),
        ("selection_return", pa.float64()),
        ("multiplier", pa.float64()),
    ]
)

LEDGER_SCHEMA = pa.schema(
    [
        ("schema_version", pa.int32()),
        ("run_id", pa.string()),
        ("study", pa.string()),
        ("task", pa.string()),
        ("arm", pa.string()),
        ("N", pa.float64()),
        ("onset_shape", pa.string()),
        ("step_matching", pa.string()),
        ("treatment", pa.string()),
        ("controller_variant", pa.string()),
        ("training_levels", pa.list_(pa.float64())),
        ("seed", pa.int64()),
        ("commit_hash", pa.string()),
        ("config_hash", pa.string()),
        ("started", pa.timestamp("us", tz="UTC")),
        ("finished", pa.timestamp("us", tz="UTC")),
        ("wall_clock_hours", pa.float64()),
        ("machine", pa.string()),
        ("completed", pa.bool_()),
        ("failure_cause", pa.string()),
        ("checkpoints", pa.list_(CHECKPOINT_STRUCT)),
        ("multiplier_trace_path", pa.string()),
        ("final_cost", pa.float64()),
        ("final_return", pa.float64()),
        ("matched_checkpoint_path", pa.string()),
        ("matched_checkpoint_step", pa.int64()),
        ("training_age", pa.int64()),
        ("selection_cost_at_match", pa.float64()),
        ("measurement_cost", pa.float64()),
        ("lambda_at_selection", pa.float64()),
        ("matched", pa.bool_()),
        ("infeasible", pa.bool_()),
        ("gap_hazard", pa.float64()),
        ("gap_dynamics", pa.float64()),
        ("gap_finetune", pa.float64()),
        ("gap_transfer", pa.float64()),
        ("dormant_onset", pa.float64()),
        ("rank_onset", pa.float64()),
        ("norm_onset", pa.float64()),
        ("lambda_peak", pa.float64()),
        ("lambda_final", pa.float64()),
        ("settling_steps", pa.int64()),
        ("sr_zero", pa.map_(pa.string(), pa.float64())),
        ("sr_fewshot", pa.map_(pa.string(), pa.float64())),
        ("adapt_steps", pa.map_(pa.string(), pa.int64())),
        ("per_level_multipliers", pa.map_(pa.string(), pa.float64())),
        ("notes", pa.string()),
    ]
)

ENRICHMENT_FIELDS = frozenset(
    {
        "matched_checkpoint_path",
        "matched_checkpoint_step",
        "training_age",
        "selection_cost_at_match",
        "measurement_cost",
        "lambda_at_selection",
        "matched",
        "infeasible",
        "gap_hazard",
        "gap_dynamics",
        "gap_finetune",
        "gap_transfer",
        "dormant_onset",
        "rank_onset",
        "norm_onset",
        "lambda_peak",
        "lambda_final",
        "settling_steps",
        "sr_zero",
        "sr_fewshot",
        "adapt_steps",
        "per_level_multipliers",
        "notes",
    }
)


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _float_keys_to_str(d: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallow copy of d with float-keyed fields string-keyed."""
    out = dict(d)
    for field in FLOAT_KEYED_MAP_FIELDS:
        if out.get(field) is not None:
            out[field] = {str(k): v for k, v in out[field].items()}
    return out


def _to_parquet_record(row: LedgerRow) -> Dict[str, Any]:
    """Convert a LedgerRow to a dict suitable for PyArrow.

    model_dump(mode="python") recurses into CheckpointRecord, so no explicit
    nested conversion is needed. Float keys are stringified for Parquet's
    map<string, ...> schema.
    """
    return _float_keys_to_str(row.model_dump(mode="python"))


def _from_parquet_record(record: Dict[str, Any]) -> LedgerRow:
    """Convert a PyArrow record back to a LedgerRow.

    PyArrow map columns may be dicts or lists of (key, value) tuples. The
    LedgerRow's mode="before" validator normalises both shapes, so no
    conversion is done here.
    """
    return LedgerRow(**record)


# ---------------------------------------------------------------------------
# Writer and reader
# ---------------------------------------------------------------------------


def append_to_ledger(
    row: LedgerRow,
    ledger_path: str | Path = LEDGER_PATH,
) -> None:
    """Append one row to the ledger.

    Raises:
        ValueError: if a row with the same run_id already exists, or if the
            existing ledger schema differs from LEDGER_SCHEMA.
    """
    ledger_path = Path(ledger_path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)

    record = _to_parquet_record(row)
    new_table = pa.Table.from_pylist([record], schema=LEDGER_SCHEMA)

    if ledger_path.exists():
        existing = pq.read_table(ledger_path)
        if existing.schema != LEDGER_SCHEMA:
            raise ValueError(
                "Existing ledger schema does not match LEDGER_SCHEMA. "
                "The schema is frozen at version 1."
            )
        existing_ids = existing.column("run_id").to_pylist()
        if row.run_id in existing_ids:
            raise ValueError(
                f"run_id {row.run_id!r} already exists in the ledger. "
                "A run is recorded once. Use the next unused seed for a repeat."
            )
        combined = pa.concat_tables([existing, new_table])
    else:
        combined = new_table

    pq.write_table(combined, ledger_path)


def load_ledger(ledger_path: str | Path = LEDGER_PATH) -> pd.DataFrame:
    """Load the ledger as a pandas DataFrame.

    Map columns are normalised to Python dicts. Float-keyed dict columns
    retain string keys in this DataFrame; use load_ledger_as_rows for typed
    access with float keys.
    """
    ledger_path = Path(ledger_path)
    if not ledger_path.exists():
        raise FileNotFoundError(f"Ledger not found at {ledger_path}")

    df = pd.read_parquet(ledger_path)
    for col in ALL_MAP_FIELDS:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda v: dict(v) if isinstance(v, list) else v
            )
    return df


def load_ledger_as_rows(ledger_path: str | Path = LEDGER_PATH) -> List[LedgerRow]:
    """Load the ledger as a list of validated LedgerRow objects."""
    ledger_path = Path(ledger_path)
    if not ledger_path.exists():
        raise FileNotFoundError(f"Ledger not found at {ledger_path}")
    table = pq.read_table(ledger_path)
    return [_from_parquet_record(r) for r in table.to_pylist()]


def write_enrichment(
    run_id: str,
    enrichment: Dict[str, Any],
    ledger_path: str | Path = LEDGER_PATH,
) -> None:
    """Write enrichment fields for one run.

    Only fields in ENRICHMENT_FIELDS are permitted. Raw fields are immutable
    after the run completes.

    Reads the whole table to Python, replaces the target row, and rebuilds
    via pa.Table.from_pylist with the frozen schema. Routes through a single
    serialization path and refuses to proceed if the rebuilt table does not
    match the frozen schema. The caller's ``enrichment`` dict is not mutated.

    Raises:
        ValueError: invalid enrichment field, schema mismatch, or rebuild
            failure.
        KeyError: run_id not in the ledger.
        FileNotFoundError: ledger missing.
    """
    invalid = set(enrichment) - ENRICHMENT_FIELDS
    if invalid:
        raise ValueError(
            f"Invalid enrichment fields: {sorted(invalid)}. "
            f"Allowed: {sorted(ENRICHMENT_FIELDS)}"
        )

    ledger_path = Path(ledger_path)
    if not ledger_path.exists():
        raise FileNotFoundError(f"Ledger not found at {ledger_path}")

    table = pq.read_table(ledger_path)
    if table.schema != LEDGER_SCHEMA:
        raise ValueError(
            "Ledger schema does not match LEDGER_SCHEMA. "
            "The schema is frozen at version 1."
        )

    records = table.to_pylist()
    run_ids = [r["run_id"] for r in records]
    if run_id not in run_ids:
        raise KeyError(f"run_id {run_id!r} not found in the ledger.")
    idx = run_ids.index(run_id)

    to_write = _float_keys_to_str(enrichment)
    records[idx].update(to_write)

    try:
        new_table = pa.Table.from_pylist(records, schema=LEDGER_SCHEMA)
    except (pa.ArrowInvalid, pa.ArrowTypeError) as exc:
        raise ValueError(
            f"Enrichment failed schema validation for run_id {run_id!r}: {exc}"
        ) from exc

    pq.write_table(new_table, ledger_path)


# ---------------------------------------------------------------------------
# Path helpers
#
# Checkpoint paths stored in the ledger are POSIX strings. They are portable
# references, not host-native paths. PurePosixPath is used to keep them
# operating-system-independent. Local filesystem operations on the ledger
# file itself continue to use Path.
# ---------------------------------------------------------------------------


def relative_checkpoint_path(absolute_path: str) -> str:
    """Convert an absolute POSIX checkpoint path to one relative to CHECKPOINT_ROOT."""
    abs_path = PurePosixPath(absolute_path)
    root = PurePosixPath(CHECKPOINT_ROOT)
    try:
        return str(abs_path.relative_to(root))
    except ValueError:
        return str(abs_path)


def absolute_checkpoint_path(relative_path: str) -> str:
    """Convert a ledger checkpoint path to an absolute POSIX path."""
    p = PurePosixPath(relative_path)
    if p.is_absolute():
        return str(p)
    return str(PurePosixPath(CHECKPOINT_ROOT) / p)
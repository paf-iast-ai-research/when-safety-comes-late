"""
ledger_schema.py — Results ledger schema for the When Safety Comes Late /
Constraint-Coverage Generalization project.

One row per launched run, finished or failed (completed=False with a
failure_cause). Frozen at version 1 before the pilot. Any change to this file
is an amendment in Part 9 of the pre-registration (see the freeze rule below).

Design decisions (see docs/ledger_schema_memo.md):
  1. Storage: Parquet.
  2. Nested data: structs and lists inside the row.
  3. Enrichment fields: same row, nullable initially.
  4. Location: results/ledger.parquet in the repository.
  5. Schema versioning: schema_version field, Literal[1].
  6. lambda_at_selection: most recent epoch at or before the checkpoint step.

Checkpoint cadence. Checkpoints are saved every 200,000 steps, at onset,
and at the end of training. Onset at N=0.25 lands on step 2,500,000 (total-
steps matched) or 3,340,000 (constrained-steps matched, Q-rounding), neither
a multiple of 200,000; the schema stores any non-negative integer step.

Map keys. Parquet and PyArrow can store float64 map keys; this schema uses
string keys on disk by design, so that the spelling of a float key is fixed
("5.0") and the composite sr_fewshot key ("5.0_200000") has the same map
type. A mode="before" validator, which runs on every construction (not only
on read), normalises both dict and list-of-(key, value) shapes and coerces
the keys of the float-keyed fields to float.

Path portability. Checkpoint paths stored in the ledger are POSIX strings,
not host-native paths, so the ledger is portable across operating systems.
Use PurePosixPath for anything stored in or read from the ledger. Use Path
only for local filesystem operations on the ledger file itself.

Write concurrency. Every write (append and enrichment) reads the whole file
and rewrites it. The rewrite goes through a temporary file and an atomic
replace, so a crash cannot truncate the ledger, but concurrent writers would
still lose rows: the pilot owner is the sole writer, and callers use
pilot.ledger_writer.ledger_transaction, which holds the ledger's lock.
"""

# FROZEN at schema_version = 1 on 2026-09-26.
# Any change to this file is an amendment recorded in Part 9 of the
# pre-registration, and results/ledger_schema.sha256 is re-recorded with it.
# A change to the stored layout (LEDGER_SCHEMA) bumps SCHEMA_VERSION, and so
# does a change that would make a row already in a ledger invalid. A
# validation correction made before any row exists, or shown to keep every
# existing row valid, stays at version 1.
# Corrections of 2026-10-01, made before the pilot wrote any row; decided
# 2026-10-02 (X-ledger-schema-amendment, docs/DECISIONS.md), to be ratified
# with the other decisions and approved by Role 4's pull request before the
# first pilot ledger row; with the corrections of 2026-10-02 that follow the
# decisions of that day. See "Changes since the freeze" in
# docs/ledger_schema_memo.md.

from __future__ import annotations

import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
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
VALID_TASKS = {
    "SafetyPointGoal1-v0",
    "SafetyCarGoal1-v0",
    "SafetyPointButton1-v0",
}
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
# Q-adapt-censoring (Table 9.1): a budget's adapt_steps is the smallest
# few-shot horizon whose satisfaction rate reaches SATISFACTION_TARGET ("at
# least 0.80"); a budget that never reaches it records its continuation's
# largest horizon + 1: 1,000,001, or 200,001 under Part 6.1 cut 3, whose
# continuation has the one horizon 200,000.
SATISFACTION_TARGET = 0.80
CENSORED_ADAPT_STEPS = {200_001, 1_000_001}
VALID_ADAPT_STEPS = VALID_FEWSHOT_HORIZONS | CENSORED_ADAPT_STEPS

FLOAT_KEYED_MAP_FIELDS = ("sr_zero", "adapt_steps", "per_level_multipliers")
STRING_KEYED_MAP_FIELDS = ("sr_fewshot",)
ALL_MAP_FIELDS = FLOAT_KEYED_MAP_FIELDS + STRING_KEYED_MAP_FIELDS

# The Study B arm whose training budgets are drawn per episode, so it has no
# fixed set of training levels (Table 2.5).
CONTINUOUS_ARM = "Continuous"

# Table 2.5 "Training set of levels" of each Study B arm (configs/registered.py
# STUDY_B_ARMS; Table 3.4). A Study B row's training_levels, sorted, equal its
# arm's set; the Continuous arm has none.
ARM_TRAINING_LEVELS: Dict[str, Optional[tuple]] = {
    "Single-10": (10.0,),
    "Single-20": (20.0,),
    "Single-40": (40.0,),
    "Sparse": (10.0, 40.0),
    "Moderate": (10.0, 20.0, 40.0),
    "Dense": (10.0, 17.5, 25.0, 32.5, 40.0),
    CONTINUOUS_ARM: None,
}
# The Continuous arm's per-level multipliers are named by the lower edge of
# their 5-unit bin of [10, 40] (Table 2.5; Table 8.2). There are six bins,
# [10, 15), ..., [35, 40], the last one closed, so the keys are 10, 15, ...,
# 35 and 40 is not one (Q-continuous-bins, Table 9.1).
CONTINUOUS_RANGE = (10.0, 40.0)
CONTINUOUS_BIN_WIDTH = 5.0

# Study A's fields: those Table B.1 marks "(Study A)" and the added matched
# and infeasible (Part 4.1). A Study B row leaves them None.
STUDY_A_ONLY_FIELDS = (
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
)
# Quantities of the matching rule that describe the matched checkpoint, and
# its robustness gaps (Table B.1: "Robustness gaps of the matched
# checkpoint"): they are set only together with it (Part 4.1 rule 1).
SELECTION_FIELDS = (
    "training_age",
    "selection_cost_at_match",
    "measurement_cost",
    "lambda_at_selection",
    "gap_hazard",
    "gap_dynamics",
    "gap_finetune",
    "gap_transfer",
)
# Results of the matching rule and the evaluations, which a failed run
# (completed=False; Part 5: excluded and repeated with the next seed) never
# receives.
COMPLETED_ONLY_FIELDS = (
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
    "sr_zero",
    "sr_fewshot",
    "adapt_steps",
)

INT64_MAX = 2**63 - 1  # the Parquet columns are int64


# ---------------------------------------------------------------------------
# Module-level helpers (not Pydantic validators; safe to call directly)
# ---------------------------------------------------------------------------


def _normalise_map(v: Any, *, coerce_keys_to_float: bool) -> Any:
    """Normalise a Parquet map value to a Python dict.

    PyArrow returns map columns as either a dict or a list of (key, value)
    tuples depending on the version and code path. Both shapes (and a tuple
    of pairs) are accepted here. When ``coerce_keys_to_float`` is True, the
    keys are converted to float; otherwise they are left as-is.

    Returns None for None, and the input unchanged if it is neither a dict
    nor a list or tuple (Pydantic then reports the type error). Raises
    ValueError for a list whose elements are not (key, value) pairs (each
    must be a tuple or list of length 2; a dict of two entries is not one),
    for an unhashable key, for a key that occurs twice in a list of pairs,
    for a bytes key (in every map field), and, under float coercion, for a
    key that is not numeric (a boolean key, and a string key holding '_' or
    whitespace, such as '1_5', included) and for two keys that are equal
    after coercion (for example '10' and '10.0'), so that no entry is
    silently dropped or misread.
    """
    if v is None:
        return v
    if isinstance(v, (list, tuple)):
        pairs: Dict[Any, Any] = {}
        for item in v:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ValueError(
                    "map must be a dict or a list of (key, value) pairs, "
                    f"got element {item!r}"
                )
            key, val = item
            try:
                if key in pairs:
                    raise ValueError(f"duplicate map key {key!r}")
                pairs[key] = val
            except TypeError as exc:
                raise ValueError(f"map key {key!r} is not hashable") from exc
        v = pairs
    if not isinstance(v, dict):
        return v
    for k in v:
        # float() reads b'1_5' as 15.0, past the spelling checks below.
        if isinstance(k, (bytes, bytearray)):
            raise ValueError(f"map keys must not be bytes, got {k!r}")
    if not coerce_keys_to_float:
        return v
    out: Dict[float, Any] = {}
    for k, val in v.items():
        if isinstance(k, (bool, np.bool_)):
            raise ValueError(f"keys must be numeric, got {k!r}")
        # float() would read '1_5' as 15.0 and ' 5 ' as 5.0.
        if isinstance(k, str) and (
            "_" in k or k != k.strip() or not k
        ):
            raise ValueError(f"keys must be numeric, got {k!r}")
        try:
            key = float(k)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"keys must be numeric, got {k!r}") from exc
        if key in out:
            raise ValueError(
                f"keys {k!r} and another key are equal after float "
                f"coercion ({key!r})"
            )
        out[key] = val
    return out


def _reject_bool_and_str(v: Any) -> Any:
    """Refuse booleans and strings where a number is expected.

    Pydantic's lax mode would turn True into 1 and 'nan' into nan.
    """
    if isinstance(v, (bool, np.bool_)):
        raise ValueError(f"expected a number, got the boolean {v!r}")
    if isinstance(v, (str, bytes)):
        raise ValueError(f"expected a number, got the string {v!r}")
    return v


def _check_posix_path(v: Optional[str]) -> Optional[str]:
    """A stored path: non-empty, POSIX separators, no '..' component, no NUL
    character or surrounding whitespace, in the normal form that
    relative_checkpoint_path writes ('a/b.pt', not 'a//b.pt', './a.pt',
    'a/b/' or '//a/b.pt'), and naming a file ('.', '/' and whitespace-only
    paths are refused). One file then has one spelling, so
    matched_checkpoint_path can be compared with the checkpoint's path as a
    string."""
    if v is None:
        return v
    if not v or not v.strip():
        raise ValueError("path must not be empty")
    if "\x00" in v:
        raise ValueError(f"path must not contain a NUL character, got {v!r}")
    if v != v.strip():
        raise ValueError(f"path must not have surrounding whitespace, "
                         f"got {v!r}")
    if "\\" in v:
        raise ValueError(f"path must use POSIX separators, got {v!r}")
    p = PurePosixPath(v)
    if ".." in p.parts:
        raise ValueError(f"path must not contain '..', got {v!r}")
    if not p.name:
        raise ValueError(f"path must name a file, got {v!r}")
    if str(p) != v:
        raise ValueError(
            f"path must be in normal POSIX form {str(p)!r}, got {v!r}"
        )
    # PurePosixPath keeps exactly two leading slashes ('//a.pt'), a second
    # spelling of the file one leading slash names on Linux.
    if p.root == "//":
        raise ValueError(
            f"path must be in normal POSIX form (one leading '/'), got {v!r}"
        )
    return v


def _reject_inf(v: float) -> float:
    """NaN is allowed (no finite statistic), +/-inf is not."""
    if math.isinf(v):
        raise ValueError(f"expected a finite number or NaN, got {v!r}")
    return v


def _strict_bool(v: Any) -> Any:
    """A boolean only: Pydantic's lax mode would turn 'yes' or 1 into True."""
    if isinstance(v, np.bool_):
        return bool(v)
    if v is not None and not isinstance(v, bool):
        raise ValueError(f"expected a boolean, got {v!r}")
    return v


def _datetime_or_iso(v: Any) -> Any:
    """A datetime or an ISO 8601 string: Pydantic's lax mode would read a
    number (or a numeric string) as seconds since 1970."""
    if v is None or isinstance(v, datetime):
        return v
    if isinstance(v, str):
        try:
            float(v)
        except ValueError:
            return v  # Pydantic parses the ISO 8601 string
        raise ValueError(f"expected an ISO 8601 datetime string, got {v!r}")
    raise ValueError(f"expected a datetime, got {v!r}")


# Number types. Every number refuses bool and str. A finite float also
# refuses NaN and +/-inf; a NaN-capable float refuses +/-inf; an int is
# bounded by int64 (the Parquet type).
NumFloat = Annotated[
    float, BeforeValidator(_reject_bool_and_str), AfterValidator(_reject_inf)
]
FiniteFloat = Annotated[
    float, BeforeValidator(_reject_bool_and_str), Field(allow_inf_nan=False)
]
Rate = Annotated[
    float,
    BeforeValidator(_reject_bool_and_str),
    Field(allow_inf_nan=False, ge=0.0, le=1.0),
]
NonNegInt64 = Annotated[
    int, BeforeValidator(_reject_bool_and_str), Field(ge=0, le=INT64_MAX)
]
StrictBool = Annotated[bool, BeforeValidator(_strict_bool)]
UtcDatetime = Annotated[datetime, BeforeValidator(_datetime_or_iso)]


# ---------------------------------------------------------------------------
# Nested models
# ---------------------------------------------------------------------------


class CheckpointRecord(BaseModel):
    """One saved checkpoint of a run."""

    model_config = ConfigDict(extra="forbid")

    path: str
    step: NonNegInt64
    # NaN-capable (never +/-inf): NaN where the epoch has no finite training
    # statistic, as at the untrained step-0 checkpoint, at a checkpoint
    # without a progress row, or in a failed run.
    training_cost: NumFloat
    training_return: NumFloat
    dormant: Optional[FiniteFloat] = Field(default=None, ge=0.0, le=1.0)
    rank: Optional[FiniteFloat] = Field(default=None, ge=0.0)
    norm: Optional[FiniteFloat] = Field(default=None, ge=0.0)
    selection_cost: Optional[FiniteFloat] = None
    selection_return: Optional[FiniteFloat] = None
    multiplier: Optional[FiniteFloat] = Field(default=None, ge=0.0)

    @field_validator("path")
    @classmethod
    def _v_path(cls, v: str) -> str:
        return _check_posix_path(v)


# ---------------------------------------------------------------------------
# Main ledger row
# ---------------------------------------------------------------------------


class LedgerRow(BaseModel):
    """One row of the results ledger: one launched run, finished or failed."""

    model_config = ConfigDict(extra="forbid")

    # Identity and provenance
    schema_version: Annotated[
        Literal[1], BeforeValidator(_reject_bool_and_str)
    ] = Field(default=SCHEMA_VERSION)
    run_id: str = Field(min_length=1)
    study: str
    task: str
    arm: str = Field(min_length=1)
    N: Optional[NumFloat] = None
    onset_shape: Optional[str] = None
    step_matching: Optional[str] = None
    treatment: Optional[str] = None
    controller_variant: Optional[str] = None
    training_levels: Optional[List[FiniteFloat]] = None
    seed: NonNegInt64
    commit_hash: str = Field(min_length=1)
    config_hash: str = Field(min_length=1)
    started: UtcDatetime
    finished: Optional[UtcDatetime] = None
    wall_clock_hours: Optional[FiniteFloat] = Field(default=None, ge=0.0)
    machine: str = Field(min_length=1)

    # Completion
    completed: StrictBool
    failure_cause: Optional[str] = None

    # Checkpoints and multiplier trace
    checkpoints: List[CheckpointRecord] = Field(default_factory=list)
    multiplier_trace_path: Optional[str] = None

    # Final evaluation
    final_cost: Optional[FiniteFloat] = None
    final_return: Optional[FiniteFloat] = None

    # Matching rule enrichment
    matched_checkpoint_path: Optional[str] = None
    matched_checkpoint_step: Optional[NonNegInt64] = None
    training_age: Optional[NonNegInt64] = None
    selection_cost_at_match: Optional[FiniteFloat] = None
    measurement_cost: Optional[FiniteFloat] = None
    lambda_at_selection: Optional[FiniteFloat] = Field(default=None, ge=0.0)
    matched: Optional[StrictBool] = None
    infeasible: Optional[StrictBool] = None

    # Robustness gaps
    gap_hazard: Optional[FiniteFloat] = None
    gap_dynamics: Optional[FiniteFloat] = None
    gap_finetune: Optional[FiniteFloat] = None
    gap_transfer: Optional[FiniteFloat] = None

    # Plasticity at onset
    dormant_onset: Optional[FiniteFloat] = Field(default=None, ge=0.0, le=1.0)
    rank_onset: Optional[FiniteFloat] = Field(default=None, ge=0.0)
    norm_onset: Optional[FiniteFloat] = Field(default=None, ge=0.0)

    # Controller quantities
    lambda_peak: Optional[FiniteFloat] = Field(default=None, ge=0.0)
    lambda_final: Optional[FiniteFloat] = Field(default=None, ge=0.0)
    settling_steps: Optional[NonNegInt64] = None

    # Study B satisfaction
    sr_zero: Optional[Dict[float, Rate]] = None
    sr_fewshot: Optional[Dict[str, Rate]] = None
    # Q-adapt-censoring (Table 9.1): a few-shot horizon, or the largest
    # horizon + 1 if the target is never reached ("above the largest horizon
    # if never reached", Table 2.5); checked against sr_fewshot where it
    # holds the budget's horizons.
    adapt_steps: Optional[Dict[float, NonNegInt64]] = None
    per_level_multipliers: Optional[
        Dict[float, Annotated[FiniteFloat, Field(ge=0.0)]]
    ] = None

    notes: str = ""

    # -- Field validators ---------------------------------------------------

    @field_validator("study")
    @classmethod
    def _v_study(cls, v: str) -> str:
        if v not in VALID_STUDIES:
            raise ValueError(
                f"study must be one of {sorted(VALID_STUDIES)}, got {v!r}"
            )
        return v

    @field_validator("task")
    @classmethod
    def _v_task(cls, v: str) -> str:
        if v not in VALID_TASKS:
            raise ValueError(
                f"task must be one of {sorted(VALID_TASKS)}, got {v!r}"
            )
        return v

    @field_validator("onset_shape")
    @classmethod
    def _v_onset_shape(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in VALID_ONSET_SHAPES:
            raise ValueError(
                f"onset_shape must be one of {sorted(VALID_ONSET_SHAPES)}, "
                f"got {v!r}"
            )
        return v

    @field_validator("step_matching")
    @classmethod
    def _v_step_matching(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in VALID_STEP_MATCHING:
            raise ValueError(
                f"step_matching must be one of {sorted(VALID_STEP_MATCHING)}, "
                f"got {v!r}"
            )
        return v

    @field_validator("treatment")
    @classmethod
    def _v_treatment(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in VALID_TREATMENTS:
            raise ValueError(
                f"treatment must be one of {sorted(VALID_TREATMENTS)}, "
                f"got {v!r}"
            )
        return v

    @field_validator("controller_variant")
    @classmethod
    def _v_controller(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in VALID_CONTROLLERS:
            raise ValueError(
                "controller_variant must be one of "
                f"{sorted(VALID_CONTROLLERS)}, got {v!r}"
            )
        return v

    @field_validator("failure_cause")
    @classmethod
    def _v_failure_cause(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in VALID_FAILURE_CAUSES:
            raise ValueError(
                "failure_cause must be one of "
                f"{sorted(VALID_FAILURE_CAUSES)}, got {v!r}"
            )
        return v

    @field_validator("N")
    @classmethod
    def _v_N(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v not in VALID_STUDY_A_NS:
            raise ValueError(
                f"N must be one of {sorted(VALID_STUDY_A_NS)}, got {v!r}"
            )
        return v

    @field_validator("started", "finished")
    @classmethod
    def _v_utc(cls, v: Optional[datetime]) -> Optional[datetime]:
        """Timezone-aware with a zero UTC offset; normalised to UTC."""
        if v is None:
            return v
        if v.tzinfo is None or v.utcoffset() is None:
            raise ValueError("datetime must be timezone-aware (UTC)")
        if v.utcoffset() != timezone.utc.utcoffset(None):
            raise ValueError(
                f"datetime must be in UTC, got offset {v.utcoffset()}"
            )
        # A pandas Timestamp may hold nanoseconds, which the timestamp('us')
        # column would truncate: the stored row would not be the validated
        # one.
        if getattr(v, "nanosecond", 0):
            raise ValueError(
                f"datetime must have at most microsecond precision, got {v!r}"
            )
        if isinstance(v, pd.Timestamp):
            v = v.to_pydatetime()
        return v.astimezone(timezone.utc)

    @field_validator("matched_checkpoint_path", "multiplier_trace_path")
    @classmethod
    def _v_paths(cls, v: Optional[str]) -> Optional[str]:
        return _check_posix_path(v)

    @field_validator(*ALL_MAP_FIELDS, mode="before")
    @classmethod
    def _v_map_shape(cls, v: Any, info: ValidationInfo) -> Any:
        """Normalise map values, on every construction.

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
                    f"key {k} not in unseen budgets "
                    f"{sorted(VALID_UNSEEN_BUDGETS)}"
                )
        return v

    @field_validator("per_level_multipliers", mode="after")
    @classmethod
    def _v_level_keys(cls, v: Any) -> Any:
        if v is None:
            return v
        for k in v:
            if not math.isfinite(k):
                raise ValueError(
                    f"per_level_multipliers key must be finite, got {k!r}"
                )
        return v

    @field_validator("adapt_steps", mode="after")
    @classmethod
    def _v_adapt_steps_values(cls, v: Any) -> Any:
        """A few-shot horizon or a censored value (Q-adapt-censoring)."""
        if v is None:
            return v
        for budget, steps in v.items():
            if steps not in VALID_ADAPT_STEPS:
                raise ValueError(
                    f"adapt_steps {steps} at budget {budget} is neither a "
                    f"few-shot horizon {sorted(VALID_FEWSHOT_HORIZONS)} nor "
                    "the largest horizon + 1 "
                    f"{sorted(CENSORED_ADAPT_STEPS)} (Q-adapt-censoring)"
                )
        return v

    @field_validator("sr_fewshot")
    @classmethod
    def _v_sr_fewshot_keys(
        cls, v: Optional[Dict[str, float]]
    ) -> Optional[Dict[str, float]]:
        """Keys are 'budget_horizon' in the canonical spelling '5.0_200000'
        (pilot.contracts.fewshot_key), with an unseen budget and a horizon of
        Table 2.5, so that one cell has one key."""
        if v is None:
            return v
        for key in v:
            if not isinstance(key, str) or "_" not in key:
                raise ValueError(
                    f"sr_fewshot key must be 'budget_horizon', got {key!r}"
                )
            budget_str, horizon_str = key.rsplit("_", 1)
            try:
                budget = float(budget_str)
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
            if budget not in VALID_UNSEEN_BUDGETS:
                raise ValueError(
                    f"sr_fewshot budget {budget_str!r} not in unseen budgets "
                    f"{sorted(VALID_UNSEEN_BUDGETS)}"
                )
            if horizon not in VALID_FEWSHOT_HORIZONS:
                raise ValueError(
                    f"sr_fewshot horizon must be one of "
                    f"{sorted(VALID_FEWSHOT_HORIZONS)}, got {horizon}"
                )
            canonical = f"{budget}_{horizon}"
            if key != canonical:
                raise ValueError(
                    f"sr_fewshot key must be in the canonical spelling "
                    f"{canonical!r}, got {key!r}"
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
    def _v_study_fields(self) -> "LedgerRow":
        """Study-specific fields: present where required, absent elsewhere."""
        if self.study == "A":
            if self.N is None:
                raise ValueError("a Study A row requires N")
            for name in ("training_levels",) + ALL_MAP_FIELDS:
                if getattr(self, name) is not None:
                    raise ValueError(f"a Study A row cannot have {name}")
        else:
            for name in (
                "N",
                "onset_shape",
                "step_matching",
                "treatment",
                "controller_variant",
            ):
                if getattr(self, name) is not None:
                    raise ValueError(f"a Study B row cannot have {name}")
            for name in STUDY_A_ONLY_FIELDS:
                if getattr(self, name) is not None:
                    raise ValueError(
                        f"a Study B row cannot have {name} (Study A: "
                        "Table B.1; Part 4.1)"
                    )
            if self.training_levels is None and self.arm != CONTINUOUS_ARM:
                raise ValueError(
                    f"Study B arm {self.arm!r} requires training_levels"
                )
            expected = ARM_TRAINING_LEVELS.get(self.arm)
            if self.arm == CONTINUOUS_ARM:
                if self.training_levels is not None:
                    raise ValueError(
                        "the Continuous arm has no fixed training_levels "
                        "(Table 2.5); leave it None"
                    )
            elif sorted(self.training_levels) != list(expected):
                raise ValueError(
                    f"training_levels {self.training_levels} are not the "
                    f"levels of arm {self.arm!r} {list(expected)} (Table 2.5)"
                )
            if (
                self.arm == CONTINUOUS_ARM
                and self.per_level_multipliers is not None
            ):
                lo, hi = CONTINUOUS_RANGE
                for k in self.per_level_multipliers:
                    if not (
                        lo <= k < hi
                        and ((k - lo) / CONTINUOUS_BIN_WIDTH).is_integer()
                    ):
                        raise ValueError(
                            f"per_level_multipliers key {k!r} is not the "
                            f"lower edge of a {CONTINUOUS_BIN_WIDTH:g}-unit "
                            f"bin of {list(CONTINUOUS_RANGE)} (Continuous arm)"
                        )
            if (
                self.training_levels is not None
                and self.per_level_multipliers is not None
            ):
                extra = set(self.per_level_multipliers) - set(
                    self.training_levels
                )
                if extra:
                    raise ValueError(
                        f"per_level_multipliers keys {sorted(extra)} are not "
                        f"training levels {self.training_levels}"
                    )
        return self

    @model_validator(mode="after")
    def _v_adapt_steps(self) -> "LedgerRow":
        """Where sr_fewshot holds a budget's horizons, its adapt_steps is the
        smallest of them whose rate reaches SATISFACTION_TARGET, else the
        largest + 1 (Table 2.5 "Adaptation steps"; Q-adapt-censoring)."""
        if self.adapt_steps is None or self.sr_fewshot is None:
            return self
        rates: Dict[float, Dict[int, float]] = {}
        for key, rate in self.sr_fewshot.items():
            budget, horizon = key.rsplit("_", 1)
            rates.setdefault(float(budget), {})[int(horizon)] = rate
        for budget, steps in self.adapt_steps.items():
            if budget not in rates:
                continue
            horizons = sorted(rates[budget])
            reached = [
                h for h in horizons if rates[budget][h] >= SATISFACTION_TARGET
            ]
            expected = reached[0] if reached else horizons[-1] + 1
            if steps != expected:
                raise ValueError(
                    f"adapt_steps {steps} at budget {budget} does not follow "
                    f"from sr_fewshot: {expected} is the smallest horizon "
                    f"whose rate reaches {SATISFACTION_TARGET}, else the "
                    "largest + 1 (Q-adapt-censoring)"
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
            for name in COMPLETED_ONLY_FIELDS:
                if getattr(self, name) is not None:
                    raise ValueError(
                        f"completed=False cannot have {name}: a failed run "
                        "is excluded and repeated with the next seed (Part 5)"
                    )
        if self.finished is not None and self.finished < self.started:
            raise ValueError(
                f"finished {self.finished} is before started {self.started}"
            )
        return self

    @model_validator(mode="after")
    def _v_checkpoints_and_match(self) -> "LedgerRow":
        steps = [cp.step for cp in self.checkpoints]
        if any(b <= a for a, b in zip(steps, steps[1:])):
            raise ValueError("checkpoint steps must be strictly increasing")
        path, step = self.matched_checkpoint_path, self.matched_checkpoint_step
        if (path is None) != (step is None):
            raise ValueError(
                "matched_checkpoint_path and matched_checkpoint_step are set "
                "together"
            )
        if step is not None:
            match = [cp for cp in self.checkpoints if cp.step == step]
            if not match:
                raise ValueError(
                    f"matched_checkpoint_step {step} is not a checkpoint step"
                )
            chosen = match[0]
            if chosen.path != path:
                raise ValueError(
                    f"matched_checkpoint_path {path!r} is not the path of the "
                    f"checkpoint at step {step} ({chosen.path!r})"
                )
            # Part 4.1 rule 1: the chosen checkpoint is one of the last ten,
            # the selection window, whose checkpoints alone carry a
            # selection cost.
            if chosen.selection_cost is None:
                raise ValueError(
                    f"matched_checkpoint_step {step} is not in the selection "
                    "window: its checkpoint has no selection_cost (Part 4.1 "
                    "rule 1: one of the last ten checkpoints)"
                )
            if (
                self.selection_cost_at_match is not None
                and self.selection_cost_at_match != chosen.selection_cost
            ):
                raise ValueError(
                    f"selection_cost_at_match {self.selection_cost_at_match} "
                    "is not the selection_cost of the matched checkpoint "
                    f"({chosen.selection_cost})"
                )
            if (
                self.lambda_at_selection is not None
                and self.lambda_at_selection != chosen.multiplier
            ):
                raise ValueError(
                    f"lambda_at_selection {self.lambda_at_selection} is not "
                    "the multiplier of the matched checkpoint "
                    f"({chosen.multiplier}; memo decision 6: copied from it)"
                )
        if (
            self.training_age is not None
            and step is not None
            and self.training_age != step
        ):
            raise ValueError(
                f"training_age {self.training_age} must equal "
                f"matched_checkpoint_step {step} (Table B.1: the chosen "
                "checkpoint's step index)"
            )
        if step is None:
            if self.matched:
                raise ValueError(
                    "matched=True requires matched_checkpoint_path and "
                    "matched_checkpoint_step"
                )
            for name in SELECTION_FIELDS:
                if getattr(self, name) is not None:
                    raise ValueError(
                        f"{name} requires matched_checkpoint_path and "
                        "matched_checkpoint_step"
                    )
        if self.matched and self.infeasible:
            raise ValueError("matched and infeasible cannot both be True")
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

# Fields write_enrichment may set. ``notes`` is not one of them: it holds the
# writer's provenance and is never replaced by an enrichment.
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
    }
)


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _float_keys_to_str(d: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallow copy of d with float-keyed fields string-keyed.

    Keys are written as str(float(key)) ('5.0'), whatever their input type,
    and map values may be dicts or lists of (key, value) pairs.
    """
    out = dict(d)
    for field in FLOAT_KEYED_MAP_FIELDS:
        if out.get(field) is not None:
            value = _normalise_map(out[field], coerce_keys_to_float=True)
            if not isinstance(value, dict):
                raise ValueError(f"{field} must be a map, got {value!r}")
            out[field] = {str(k): v for k, v in value.items()}
    return out


def _to_parquet_record(row: LedgerRow) -> Dict[str, Any]:
    """Convert a LedgerRow to a dict suitable for PyArrow.

    model_dump(mode="python") recurses into CheckpointRecord, so no explicit
    nested conversion is needed. Float keys are stringified for the
    map<string, ...> columns.
    """
    return _float_keys_to_str(row.model_dump(mode="python"))


def _from_parquet_record(record: Dict[str, Any]) -> LedgerRow:
    """Convert a PyArrow record back to a LedgerRow.

    PyArrow map columns may be dicts or lists of (key, value) tuples. The
    LedgerRow's mode="before" validator normalises both shapes, so no
    conversion is done here.
    """
    return LedgerRow(**record)


def _revalidated(row: LedgerRow) -> LedgerRow:
    """Validate a row again: the model has no validate_assignment, so a row
    changed after construction is checked here before it is written.
    Raises pydantic.ValidationError (a subclass of ValueError)."""
    return LedgerRow.model_validate(row.model_dump(mode="python"))


def _to_table(records: List[Dict[str, Any]]) -> pa.Table:
    try:
        return pa.Table.from_pylist(records, schema=LEDGER_SCHEMA)
    except (pa.ArrowInvalid, pa.ArrowTypeError, OverflowError) as exc:
        raise ValueError(f"row does not fit LEDGER_SCHEMA: {exc}") from exc


def _check_unique_run_ids(run_ids: List[str]) -> None:
    if len(set(run_ids)) != len(run_ids):
        dupes = sorted({r for r in run_ids if run_ids.count(r) > 1})
        raise ValueError(f"run_id(s) {dupes} occur more than once")


def _row_from_record(record: Dict[str, Any]) -> LedgerRow:
    """_from_parquet_record, with an error that names the row's run_id."""
    try:
        return _from_parquet_record(record)
    except ValueError as exc:  # pydantic.ValidationError included
        raise ValueError(
            f"ledger row {record['run_id']!r} is not a valid LedgerRow: "
            f"{exc}"
        ) from exc


def _read_checked(ledger_path: Path) -> pa.Table:
    """Read the ledger and check its schema against LEDGER_SCHEMA."""
    table = pq.read_table(ledger_path)
    if table.schema != LEDGER_SCHEMA:
        raise ValueError(
            "Ledger schema does not match LEDGER_SCHEMA. "
            "The schema is frozen at version 1."
        )
    return table


def _current_umask() -> int:
    """The process umask, read without changing it where the system allows.

    On Linux it is read from /proc/self/status ('Umask:'). Elsewhere it is
    read by setting it and setting it back (os.umask), which changes the
    process-wide umask for that moment: a file another thread creates then
    gets the wrong mode, so that fallback is not thread-safe."""
    try:
        with open("/proc/self/status", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("Umask:"):
                    return int(line.split()[1], 8)
    except (OSError, ValueError, IndexError):
        pass
    umask = os.umask(0o077)
    os.umask(umask)
    return umask


def _ledger_mode(ledger_path: Path) -> int:
    """The permission bits the rewritten ledger keeps: the existing ledger's,
    or for a new ledger the default for a new file (0o666 less the umask),
    as a plain write would give. mkstemp alone would leave 0o600. See
    _current_umask for the thread-safety of reading the umask."""
    try:
        return ledger_path.stat().st_mode & 0o7777
    except FileNotFoundError:
        return 0o666 & ~_current_umask()


def _write_atomic(table: pa.Table, ledger_path: Path) -> None:
    """Write ``table`` to a temporary file beside the ledger, fsync it, and
    replace the ledger with it, so a crash never leaves a truncated ledger.

    The data is written and fsynced through one writable handle (Windows'
    fsync needs write access), and the ledger keeps its permission bits."""
    mode = _ledger_mode(ledger_path)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(ledger_path.parent),
        prefix=f".{ledger_path.name}.",
        suffix=".partial",
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            pq.write_table(table, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, ledger_path)
    finally:
        if tmp.exists():
            tmp.unlink()
    try:
        dir_fd = os.open(str(ledger_path.parent), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    except OSError:
        pass
    finally:
        os.close(dir_fd)


# ---------------------------------------------------------------------------
# Writer and reader
# ---------------------------------------------------------------------------


def append_to_ledger(
    row: LedgerRow,
    ledger_path: str | Path = LEDGER_PATH,
) -> None:
    """Append one row to the ledger.

    The row is validated again first (a LedgerRow changed after construction
    is not otherwise checked). The ledger is replaced atomically.

    Raises:
        ValueError: if the row is invalid (pydantic.ValidationError, a
            ValueError) or does not fit the Parquet schema,
            if a row with the same run_id already exists, or if the existing
            ledger schema differs from LEDGER_SCHEMA.
    """
    ledger_path = Path(ledger_path)

    row = _revalidated(row)
    new_table = _to_table([_to_parquet_record(row)])
    # Only after the row is accepted: a refused row leaves no directories.
    ledger_path.parent.mkdir(parents=True, exist_ok=True)

    if ledger_path.exists():
        existing = _read_checked(ledger_path)
        existing_ids = existing.column("run_id").to_pylist()
        if row.run_id in existing_ids:
            raise ValueError(
                f"run_id {row.run_id!r} already exists in the ledger. "
                "A run is recorded once. Use the next unused seed for a "
                "repeat."
            )
        combined = pa.concat_tables([existing, new_table])
    else:
        combined = new_table

    _write_atomic(combined, ledger_path)


def load_ledger(ledger_path: str | Path = LEDGER_PATH) -> pd.DataFrame:
    """Load the ledger as a pandas DataFrame.

    Map columns are normalised to Python dicts. Float-keyed dict columns
    retain string keys in this DataFrame; use load_ledger_as_rows for typed
    access with float keys. The rows are not validated.

    Raises:
        FileNotFoundError: ledger missing.
        ValueError: the file's schema differs from LEDGER_SCHEMA.
    """
    ledger_path = Path(ledger_path)
    if not ledger_path.exists():
        raise FileNotFoundError(f"Ledger not found at {ledger_path}")

    df = _read_checked(ledger_path).to_pandas()
    for col in ALL_MAP_FIELDS:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda v: dict(v) if isinstance(v, list) else v
            )
    return df


def load_ledger_as_rows(
    ledger_path: str | Path = LEDGER_PATH,
) -> List[LedgerRow]:
    """Load the ledger as a list of validated LedgerRow objects.

    Raises:
        FileNotFoundError: ledger missing.
        ValueError: the file's schema differs from LEDGER_SCHEMA, a run_id
            occurs twice, or a row is invalid (the message names its run_id;
            the pydantic.ValidationError is the __cause__).
    """
    ledger_path = Path(ledger_path)
    if not ledger_path.exists():
        raise FileNotFoundError(f"Ledger not found at {ledger_path}")
    table = _read_checked(ledger_path)
    records = table.to_pylist()
    _check_unique_run_ids([r["run_id"] for r in records])
    return [_row_from_record(record) for record in records]


def write_enrichment(
    run_id: str,
    enrichment: Dict[str, Any],
    ledger_path: str | Path = LEDGER_PATH,
) -> None:
    """Write enrichment fields for one run.

    Only fields in ENRICHMENT_FIELDS are permitted. Raw fields (and notes)
    are immutable after the row is written.

    Checks the existing file's schema against LEDGER_SCHEMA and its run_ids
    for duplicates, validates the target row as stored (an error names its
    run_id), merges the enrichment into it and validates the merged row as a
    LedgerRow (every field and cross-field constraint; map keys are
    canonicalised), then rebuilds the table with the frozen schema through
    the same serialisation path as append_to_ledger (_to_parquet_record and
    _to_table) and replaces the ledger atomically. The caller's
    ``enrichment`` dict is not mutated.

    Raises:
        ValueError: invalid enrichment field, schema mismatch, a duplicate
            run_id, an invalid stored row, an invalid merged row
            (pydantic.ValidationError, a subclass of ValueError), or
            rebuild failure.
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

    table = _read_checked(ledger_path)

    records = table.to_pylist()
    run_ids = [r["run_id"] for r in records]
    _check_unique_run_ids(run_ids)
    if run_id not in run_ids:
        raise KeyError(f"run_id {run_id!r} not found in the ledger.")
    idx = run_ids.index(run_id)

    current = _row_from_record(records[idx])
    # A pydantic.ValidationError (a ValueError) propagates as it is.
    candidate = LedgerRow.model_validate(
        {**current.model_dump(mode="python"), **enrichment}
    )
    records[idx] = _to_parquet_record(candidate)

    _write_atomic(_to_table(records), ledger_path)


# ---------------------------------------------------------------------------
# Path helpers
#
# Checkpoint paths stored in the ledger are POSIX strings. They are portable
# references, not host-native paths. PurePosixPath is used to keep them
# operating-system-independent. Local filesystem operations on the ledger
# file itself continue to use Path.
# ---------------------------------------------------------------------------


def _check_helper_path(path: str) -> PurePosixPath:
    """The checks both path helpers share: no '..' component, which could
    leave the root while looking relative, a path that names a file (not
    '', '.' or '/'), and no two leading slashes, which PurePosixPath keeps
    ('//data/checkpoints/x.pt' would stay outside the root)."""
    p = PurePosixPath(path)
    if ".." in p.parts:
        raise ValueError(
            f"checkpoint path must not contain '..', got {path!r}"
        )
    if not path or not path.strip() or not p.name:
        raise ValueError(f"checkpoint path must name a file, got {path!r}")
    if p.root == "//":
        raise ValueError(
            f"checkpoint path must not start with '//', got {path!r}"
        )
    return p


def relative_checkpoint_path(absolute_path: str) -> str:
    """Convert a POSIX checkpoint path to one relative to CHECKPOINT_ROOT.

    A path outside CHECKPOINT_ROOT (or already relative) is returned in
    normalised POSIX form (``'a//b/'`` becomes ``'a/b'``). Raises ValueError
    for a path with a '..' component or two leading slashes, for an empty
    path, '.' or '/', and for CHECKPOINT_ROOT itself, none of which is a
    checkpoint.
    """
    abs_path = _check_helper_path(absolute_path)
    root = PurePosixPath(CHECKPOINT_ROOT)
    try:
        rel = abs_path.relative_to(root)
    except ValueError:
        return str(abs_path)
    if str(rel) == ".":
        raise ValueError(
            f"{absolute_path!r} is the checkpoint root, not a checkpoint"
        )
    return str(rel)


def absolute_checkpoint_path(relative_path: str) -> str:
    """Convert a ledger checkpoint path to an absolute POSIX path.

    Raises ValueError, as relative_checkpoint_path does, for a '..'
    component (which could leave CHECKPOINT_ROOT) or two leading slashes,
    and for an empty path, '.' or '/', none of which is a checkpoint.
    """
    p = _check_helper_path(relative_path)
    if p.is_absolute():
        return str(p)
    return str(PurePosixPath(CHECKPOINT_ROOT) / p)

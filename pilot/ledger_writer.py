"""Write one results-ledger row per run (Appendix B, Table B.1; Part 3.4; Part 5.6).

The pilot owner is the sole writer of the ledger (``results/ledger_schema.py``, write concurrency
note). This module turns a run directory into a validated ``LedgerRow`` and appends it with the
frozen schema's own ``append_to_ledger``; the schema file itself is not changed.

Every write to a ledger (an append here, an enrichment in pilot/enrichment.py) goes through
``ledger_transaction``: an exclusive lock, the change applied to a copy, then an atomic rename. A
row can never be lost to a concurrent write, and a crash in the middle of a write leaves the
previous ledger intact.

Where things come from:
  identity, arm fields ........ the RunSpec (pilot/manifest.py)
  commit, config hash, times .. train_result.json (pilot/launch.py)
  checkpoints ................. OmniSafe torch_save/epoch-{k}.pt, step = k x 20,000
  training cost and return .... progress.csv row of epoch k-1: Metrics/BatchEpCost and
                                Metrics/BatchEpRet, the epoch's training batch (Part 3.4), logged
                                by FullStateCheckpointMixin; OmniSafe's Metrics/EpCost and EpRet
                                (a 50-episode moving mean) only when those columns are absent
  multiplier at a checkpoint .. progress.csv Metrics/LagrangeMultiplier of epoch k-1 (the value in
                                effect at the checkpoint; ledger memo decision 6); for step 0 the
                                initial value of the algorithm (0.001 PPO-Lagrangian, 0.0 PID)
  multiplier trace ............ every multiplier column of every epoch (Part 3.4), including Study
                                B's per-level Metrics/LagrangeMultiplier/level_{budget}
  per_level_multipliers ....... Study B: each level's multiplier at the end of training, written with
                                the row although the schema lists it among the enrichment fields
                                (it is a training output); enrichment never writes it
  selection cost and return ... evaluation.json (Role 2 harness), last ten checkpoints only
  final cost and return ....... evaluation.json
  plasticity metrics .......... plasticity.csv (Role 3), when present; a non-finite value (a run that
                                failed with NaN weights) is left null and named in the notes, so that
                                the row, and with it the exclusion of Part 5.6, can still be written

Pilot runs go to ``results/pilot/ledger.parquet`` and never to the main ledger ("The pilot's runs
are not reused", Part 3.6). The Study B pilot's unconstrained PPO run cannot be represented in
schema version 1 (its Study B arm list has no unconstrained arm), so it is written to a JSON
sidecar with the same fields (NaN written as null) until the schema is amended. Smoke runs
(uncommitted code or open questions bypassed) are refused by the repository's ledgers, and so is
any row whose checkpoint paths are not relative to the schema's CHECKPOINT_ROOT (/data/checkpoints):
a host-specific absolute path in a row that can never be rewritten would break the ledger on every
other machine.
"""

from __future__ import annotations

import contextlib
import csv
import json
import math
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from configs import registered as R
from pilot import provenance
from pilot.algorithms import BATCH_COST_KEY, BATCH_RETURN_KEY
from pilot.contracts import (
    PLASTICITY_COLUMNS,
    PLASTICITY_FILE,
    check_canonical_seeds,
    checkpoint_steps,
    selection_window,
    validate_seed_set,
)
from pilot.launch import EVALUATION_RESULT, MULTIPLIER_COLUMN, TRAIN_RESULT, read_progress
from pilot.manifest import RunSpec

REPO_ROOT = provenance.REPO_ROOT
MULTIPLIER_TRACE_FILE = "multiplier_trace.csv"
TRAINING_COST_NOTE = "training_cost = OmniSafe Metrics/EpCost, the mean of the last 50 training episodes"
TRAINING_COST_NOTE_BATCH = "training_cost = mean episodic cost of the epoch's training batch (Metrics/BatchEpCost)"
LEVEL_PREFIX = MULTIPLIER_COLUMN + "/level_"
STATISTIC_SUFFIXES = ("/Min", "/Max", "/Std")  # OmniSafe's min_and_max / std columns beside a logged value
REPOSITORY_RESULTS = REPO_ROOT / "results"


class LedgerWriteError(RuntimeError):
    pass


@dataclass(frozen=True)
class LedgerPaths:
    """Where rows go. The defaults are the repository's ledger files (ledger memo, decision 4)."""

    main: Path = REPO_ROOT / "results" / "ledger.parquet"
    pilot: Path = REPO_ROOT / "results" / "pilot" / "ledger.parquet"
    sidecar_dir: Path = REPO_ROOT / "results" / "pilot" / "sidecar"

    def for_spec(self, spec: RunSpec) -> Path:
        return self.pilot if spec.pilot else self.main

    def seeds_registry(self, spec: RunSpec) -> Path:
        """The evaluation seed sets fixed by the first run (Table 2.1), beside its ledger."""
        ledger = self.for_spec(spec)
        return ledger.with_name(ledger.stem + ".seeds.json")


DEFAULT_PATHS = LedgerPaths()


def uses_repository_ledgers(paths: LedgerPaths) -> bool:
    """True if any of the paths lies in the repository's results/ directory."""
    root = REPOSITORY_RESULTS.resolve()
    for p in (paths.main, paths.pilot, paths.sidecar_dir):
        resolved = Path(os.path.abspath(p)).resolve()
        if resolved == root or root in resolved.parents:
            return True
    return False


def smoke_paths(data_root: Path) -> LedgerPaths:
    """Ledgers for smoke tests: under the data root, never in the repository."""
    base = Path(data_root) / "smoke"
    return LedgerPaths(main=base / "ledger.parquet", pilot=base / "pilot_ledger.parquet", sidecar_dir=base / "sidecar")


# ---------------------------------------------------------------------------
# Locked, atomic ledger writes
# ---------------------------------------------------------------------------


def _fsync_dir(directory: Path) -> None:
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:  # pragma: no cover - platforms without directory handles
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextlib.contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    """Exclusive lock beside the ledger so two writers can never interleave (POSIX)."""
    lock_path = path.with_name(path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w", encoding="utf-8") as fh:
        try:
            import fcntl
        except ImportError:  # pragma: no cover - Windows: the scheduler is the only writer
            yield
            return
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


@contextlib.contextmanager
def ledger_transaction(path: Path) -> Iterator[Path]:
    """Yield a working copy of the ledger; on success it atomically replaces the ledger.

    Use for every write: ``with ledger_transaction(p) as tmp: append_to_ledger(row, tmp)``.
    """
    path = Path(path)
    with _file_lock(path):
        tmp = path.with_name(path.name + ".tmp")
        if tmp.exists():
            tmp.unlink()
        if path.exists():
            shutil.copy2(path, tmp)
        try:
            yield tmp
            if tmp.exists():
                with open(tmp, "rb") as fh:
                    os.fsync(fh.fileno())
                os.replace(tmp, path)
                _fsync_dir(path.parent)
        finally:
            if tmp.exists():
                tmp.unlink()


# ---------------------------------------------------------------------------
# Building a row
# ---------------------------------------------------------------------------


def _representable(spec: RunSpec) -> bool:
    from results.ledger_schema import VALID_STUDY_B_ARMS

    return not (spec.study == "B" and spec.arm not in VALID_STUDY_B_ARMS)


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _progress_by_epoch(rows: list[dict[str, str]]) -> dict[int, dict[str, str]]:
    return {int(float(r["Train/Epoch"])): r for r in rows if r.get("Train/Epoch") not in (None, "")}


def _read_plasticity(omnisafe_dir: Path) -> dict[int, dict[str, float]]:
    path = omnisafe_dir / PLASTICITY_FILE
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = set(PLASTICITY_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise LedgerWriteError(f"{path} lacks the columns {sorted(missing)} (contract 2: {', '.join(PLASTICITY_COLUMNS)})")
        out: dict[int, dict[str, float]] = {}
        for r in reader:
            try:
                step = int(float(r["step"]))
            except (TypeError, ValueError):
                raise LedgerWriteError(f"{path}: step {r['step']!r} is not a number (contract 2)") from None
            values = {}
            for key in PLASTICITY_COLUMNS[1:]:
                try:
                    values[key] = float(r[key])  # "nan" and "inf" parse; they are left null later
                except (TypeError, ValueError):
                    raise LedgerWriteError(
                        f"{path}: {key} at step {step} is {r[key]!r}, not a number (contract 2)"
                    ) from None
            out[step] = values
        return out


def multiplier_columns(header: list[str] | tuple[str, ...]) -> list[str]:
    """The progress.csv columns that hold a multiplier (contract 3), without OmniSafe's statistics."""
    return [
        c for c in header
        if (c == MULTIPLIER_COLUMN or c.startswith(LEVEL_PREFIX)) and not c.endswith(STATISTIC_SUFFIXES)
    ]


def _level(column: str) -> float:
    """``Metrics/LagrangeMultiplier/level_17.5`` -> 17.5."""
    try:
        return float(column[len(LEVEL_PREFIX):])
    except ValueError:
        raise LedgerWriteError(f"{column!r}: a per-level multiplier column must end in the budget level (contract 3)") from None


def write_multiplier_trace(omnisafe_dir: Path, rows: list[dict[str, str]]) -> Path | None:
    """Per-epoch trace of every multiplier (Part 3.4: "every per-level multiplier ... at every epoch").

    Columns: epoch, step_end, then ``lagrange_multiplier`` for the single multiplier and
    ``level_{budget}`` for each of Study B's per-level multipliers.
    """
    columns = multiplier_columns(list(rows[0])) if rows else []
    if not columns:
        return None
    names = ["lagrange_multiplier" if c == MULTIPLIER_COLUMN else c[len(MULTIPLIER_COLUMN) + 1:] for c in columns]
    path = omnisafe_dir / MULTIPLIER_TRACE_FILE
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["epoch", "step_end", *names])
        for epoch, row in sorted(_progress_by_epoch(rows).items()):
            writer.writerow([epoch, (epoch + 1) * R.STEPS_PER_EPOCH, *(row[c] for c in columns)])
    return path


def per_level_multipliers(rows: list[dict[str, str]]) -> dict[float, float] | None:
    """Study B: each level's multiplier after the last epoch (None without per-level columns)."""
    columns = [c for c in multiplier_columns(list(rows[0])) if c.startswith(LEVEL_PREFIX)] if rows else []
    if not columns:
        return None
    last = max(_progress_by_epoch(rows).items())[1]
    values = {_level(c): _float(last[c]) for c in columns}
    return values if all(math.isfinite(v) and v >= 0 for v in values.values()) else None


def _initial_multiplier(spec: RunSpec) -> float | None:
    """The multiplier in effect at step 0, where the algorithm fixes it.

    Table 2.1: before onset the multiplier is held at its initial value, 0.001 for
    PPO-Lagrangian (Table A.1); the controller variants of Table 2.4 change it only at onset, and
    every controller arm has N > 0, so step 0 is before onset. OmniSafe's PID multiplier
    (``cost_penalty``) is 0.0 until its first update. PPO has no multiplier, and Study B's are per
    level.
    """
    if spec.study != "A":
        return None
    if spec.base_algo == "CPPOPID":
        return 0.0
    if spec.base_algo == "PPOLag" and (spec.controller_variant is None or (spec.onset_step or 0) > 0):
        return R.LAGRANGE_MULTIPLIER_INIT
    return None


def build_row_fields(spec: RunSpec, run_dir: Path, attempt: int | None = None) -> dict[str, Any]:
    """Collect the ledger fields of one run as plain Python values.

    The step-0 checkpoint (the untrained policy) has no training epoch behind it, so its training
    cost and return are NaN. Non-finite plasticity metrics are left null (noted); a non-finite
    selection-set cost is refused, since rule 1 cannot rank it.
    """
    from results.ledger_schema import relative_checkpoint_path

    run_dir = Path(os.path.abspath(run_dir))  # not resolve(): keep paths under CHECKPOINT_ROOT if it is a symlink
    train = json.loads((run_dir / TRAIN_RESULT).read_text(encoding="utf-8"))
    if train.get("run_id") != spec.run_id:
        raise LedgerWriteError(f"{run_dir} belongs to {train.get('run_id')!r}, not {spec.run_id!r}")
    if train.get("status") not in ("completed", "failed"):
        raise LedgerWriteError(f"{spec.run_id}: training status {train.get('status')!r} is not final")
    commit = train.get("commit_hash", "")
    if not provenance.is_full_commit_hash(commit):
        raise LedgerWriteError(f"{spec.run_id}: commit hash {commit!r} is not a full 40-character hash")
    completed = train["status"] == "completed"
    evaluation: dict[str, Any] | None = None
    eval_path = run_dir / EVALUATION_RESULT
    if completed:
        if not eval_path.exists():
            raise LedgerWriteError(f"{spec.run_id}: completed run has no {EVALUATION_RESULT}; evaluate before writing")
        evaluation = json.loads(eval_path.read_text(encoding="utf-8"))

    omnisafe_dir = run_dir / train["omnisafe_dir"] if train.get("omnisafe_dir") else None
    rows = read_progress(omnisafe_dir) if omnisafe_dir and (omnisafe_dir / "progress.csv").exists() else []
    by_epoch = _progress_by_epoch(rows)
    plasticity = _read_plasticity(omnisafe_dir) if omnisafe_dir else {}
    steps = checkpoint_steps(omnisafe_dir) if omnisafe_dir else []
    selection: dict[int, list[float]] = {}
    if evaluation is not None:
        selection = {int(k): v for k, v in evaluation["selection"].items()}
        if sorted(selection) != selection_window(steps, spec.total_steps):
            raise LedgerWriteError(f"{spec.run_id}: selection steps do not match the last ten checkpoints")

    # Part 3.4 "training-batch mean episodic cost and return": the per-epoch batch mean logged by
    # FullStateCheckpointMixin; OmniSafe's own 50-episode moving mean only when it is absent.
    batch = bool(rows) and BATCH_COST_KEY in rows[0] and BATCH_RETURN_KEY in rows[0]
    cost_col, return_col = (BATCH_COST_KEY, BATCH_RETURN_KEY) if batch else ("Metrics/EpCost", "Metrics/EpRet")
    checkpoints = []
    nonfinite_plasticity: list[int] = []  # a run with NaN weights has NaN metrics; the row is still written
    for step in steps:
        epoch_done = step // R.STEPS_PER_EPOCH
        row = by_epoch.get(epoch_done - 1)
        multiplier: float | None
        if row is not None and MULTIPLIER_COLUMN in row:
            value = _float(row[MULTIPLIER_COLUMN])
            multiplier = value if math.isfinite(value) and value >= 0 else None  # failed runs may hold NaN
        elif epoch_done == 0:
            multiplier = _initial_multiplier(spec)
        else:
            multiplier = None
        record: dict[str, Any] = {
            "path": relative_checkpoint_path(str(omnisafe_dir / "torch_save" / f"epoch-{epoch_done}.pt")),
            "step": step,
            "training_cost": _float(row[cost_col]) if row else math.nan,  # NaN for the untrained step-0 checkpoint
            "training_return": _float(row[return_col]) if row else math.nan,
            "multiplier": multiplier,
        }
        if step in plasticity:
            finite = {k: v for k, v in plasticity[step].items() if math.isfinite(v)}
            if len(finite) < len(plasticity[step]):
                nonfinite_plasticity.append(step)
            record.update(finite)
        if step in selection:
            cost, ret = _float(selection[step][0]), _float(selection[step][1])
            if not (math.isfinite(cost) and math.isfinite(ret)):
                raise LedgerWriteError(f"{spec.run_id}: selection[{step}] = {selection[step]!r} is not finite (rule 1 needs finite costs)")
            record["selection_cost"], record["selection_return"] = cost, ret
        checkpoints.append(record)

    if spec.study == "B" and spec.training_levels is not None and rows:
        logged = sorted(_level(c) for c in multiplier_columns(list(rows[0])) if c.startswith(LEVEL_PREFIX))
        if logged != sorted(float(b) for b in spec.training_levels):
            raise LedgerWriteError(
                f"{spec.run_id}: per-level multiplier columns for levels {logged}, but the arm trains on "
                f"{sorted(spec.training_levels)} (Part 3.4: every per-level multiplier; contract 3)"
            )
    trace = write_multiplier_trace(omnisafe_dir, rows) if omnisafe_dir else None
    levels = per_level_multipliers(rows) if spec.study == "B" else None
    train_hours = float(train.get("wall_clock_hours") or 0.0)
    eval_hours = float(evaluation.get("eval_wall_clock_hours", 0.0)) if evaluation else 0.0
    notes = [TRAINING_COST_NOTE_BATCH if batch else TRAINING_COST_NOTE, f"train_hours={train_hours:.4f}", f"eval_hours={eval_hours:.4f}"]
    if evaluation and evaluation.get("eval_commit_hash"):
        notes.append(f"eval_commit={evaluation['eval_commit_hash']}")
    if train.get("config_hash") is None:
        notes.append("config_hash unavailable: the run failed before its configuration was built")
    if attempt and attempt > 1:
        notes.append(f"attempt={attempt} (restarted from scratch with the same seed after an interruption)")
    if levels is not None:
        notes.append("per_level_multipliers = each level's multiplier after the last epoch")
    if nonfinite_plasticity:
        notes.append(f"non-finite plasticity metrics left null at steps {nonfinite_plasticity}")
    if train.get("detail"):
        notes.append("detail: " + str(train["detail"])[-500:])

    fields: dict[str, Any] = {
        "run_id": spec.run_id,
        "study": spec.study,
        "task": spec.task,
        "arm": spec.arm,
        "N": spec.N if spec.study == "A" else None,
        "onset_shape": spec.onset_shape,
        "step_matching": spec.step_matching,
        "treatment": spec.treatment,
        "controller_variant": spec.controller_variant,
        "training_levels": list(spec.training_levels) if spec.training_levels is not None else None,
        "seed": spec.seed,
        "commit_hash": commit,
        "config_hash": train.get("config_hash") or "unavailable",
        "started": datetime.fromisoformat(train["started"]),
        "finished": datetime.fromisoformat(train["finished"]) if train.get("finished") else None,
        "wall_clock_hours": train_hours + eval_hours,
        "machine": train["machine"],
        "completed": completed,
        "failure_cause": train.get("failure_cause"),
        "checkpoints": checkpoints,
        "multiplier_trace_path": relative_checkpoint_path(str(trace)) if trace else None,
        "final_cost": float(evaluation["final_cost"]) if evaluation else None,
        "final_return": float(evaluation["final_return"]) if evaluation else None,
        "per_level_multipliers": levels,
        "notes": "; ".join(notes),
    }
    if spec.study == "A" and spec.onset_step is not None and spec.onset_step in plasticity:
        onset = plasticity[spec.onset_step]
        fields.update({f"{k}_onset": v for k, v in onset.items() if math.isfinite(v)})  # dormant, rank, norm
    return fields


def _is_smoke(run_dir: Path) -> bool:
    train = json.loads((Path(run_dir) / TRAIN_RESULT).read_text(encoding="utf-8"))
    return bool(train.get("worktree_dirty") or train.get("allow_dirty") or train.get("allow_pending"))


def _absolute_paths(fields: dict[str, Any]) -> list[str]:
    paths = [c["path"] for c in fields["checkpoints"]] + [fields.get("multiplier_trace_path")]
    return [p for p in paths if p and p.startswith("/")]


def _json_safe(value: Any) -> Any:
    """``value`` with every non-finite float replaced by None (null): NaN is not JSON (RFC 8259)."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _write_sidecar(target: Path, fields: dict[str, Any]) -> None:
    """Write ``fields`` as JSON to ``target`` once: fsync, then link (fails if the target exists).

    Non-finite floats (the step-0 checkpoint's training cost and return) are written as null, so
    that the file is strict JSON; the ledger itself stores them as NaN.
    """
    def default(value: Any) -> Any:
        if isinstance(value, datetime):
            return value.isoformat()
        raise TypeError(f"{type(value).__name__} is not JSON serialisable")

    tmp = target.with_name(target.name + f".{os.getpid()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(_json_safe(fields), indent=2, default=default, sort_keys=True, allow_nan=False))
            fh.flush()
            os.fsync(fh.fileno())
        try:
            os.link(tmp, target)  # atomic, and never replaces an existing record
        except FileExistsError:
            raise LedgerWriteError(f"{target} already exists; a run is recorded once") from None
        _fsync_dir(target.parent)
    finally:
        tmp.unlink(missing_ok=True)


def write_run(spec: RunSpec, run_dir: Path, paths: LedgerPaths = DEFAULT_PATHS, attempt: int | None = None) -> Path:
    """Validate and append one run's row. Returns the file written to.

    The evaluation seeds are checked against the registry only after the row validates, under the
    ledger's lock, so a rejected row never fixes the seed sets (Table 2.1).
    """
    from results.ledger_schema import CheckpointRecord, LedgerRow, append_to_ledger

    repository = uses_repository_ledgers(paths)
    if repository and _is_smoke(run_dir):
        raise LedgerWriteError(
            f"{spec.run_id} ran with uncommitted code or bypassed open questions (a smoke run); "
            "it may not be written to the repository's ledgers"
        )
    fields = build_row_fields(spec, run_dir, attempt)
    if repository and _absolute_paths(fields):
        raise LedgerWriteError(
            f"{spec.run_id}: checkpoint paths {_absolute_paths(fields)[:2]} are not under {_checkpoint_root()}; "
            "keep the data root at /data (a symlink to the real storage) so that the ledger stays portable"
        )
    seeds = None
    if fields["completed"]:
        evaluation = json.loads((Path(run_dir) / EVALUATION_RESULT).read_text(encoding="utf-8"))
        seeds = validate_seed_set("selection_seeds", evaluation["selection_seeds"])
    if not _representable(spec):
        paths.sidecar_dir.mkdir(parents=True, exist_ok=True)
        target = paths.sidecar_dir / f"{spec.run_id}.json"
        if target.exists():
            raise LedgerWriteError(f"{target} already exists; a run is recorded once")
        fields["reason"] = "not representable in ledger schema version 1 (no unconstrained Study B arm)"
        with _file_lock(paths.for_spec(spec)):
            if seeds is not None:
                check_canonical_seeds("selection", seeds, paths.seeds_registry(spec))
            _write_sidecar(target, fields)
        return target
    fields["checkpoints"] = [CheckpointRecord(**c) for c in fields["checkpoints"]]
    row = LedgerRow(**fields)
    path = paths.for_spec(spec)
    with ledger_transaction(path) as tmp:
        append_to_ledger(row, tmp)  # into the working copy; the ledger changes only if the check below passes
        if seeds is not None:
            check_canonical_seeds("selection", seeds, paths.seeds_registry(spec))
    return path


def _checkpoint_root() -> str:
    from results.ledger_schema import CHECKPOINT_ROOT

    return CHECKPOINT_ROOT


def recorded_run_ids(ledger_path: Path) -> set[str]:
    """run_ids already in a ledger (empty if the ledger does not exist yet)."""
    if not Path(ledger_path).exists():
        return set()
    import pyarrow.parquet as pq

    return set(pq.read_table(ledger_path, columns=["run_id"]).column("run_id").to_pylist())


def recorded_anywhere(spec: RunSpec, paths: LedgerPaths = DEFAULT_PATHS) -> bool:
    if spec.run_id in recorded_run_ids(paths.for_spec(spec)):
        return True
    return (paths.sidecar_dir / f"{spec.run_id}.json").exists()

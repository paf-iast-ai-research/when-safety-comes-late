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
  selection cost and return ... evaluation.json (contract 1: Role 2's harness, Role 5's for plug-in
                                study_b), last ten checkpoints only
  final cost and return ....... evaluation.json
  plasticity metrics .......... plasticity.csv (Role 3), checked by pilot.contracts.validate_plasticity
                                (contract 2; required for the Study A plug-ins of a completed run); a
                                non-finite value (a run that failed with NaN weights) is left null and
                                named in the notes, so that the row, and with it the exclusion of Part
                                5.6, can still be written. dormant_onset, rank_onset and norm_onset are
                                written with the row although the schema lists them among the
                                enrichment fields
  per-episode evaluations ..... evaluation.json's per-episode costs and returns (the final checkpoint and
                                every selection checkpoint), written as the supplement record
                                ``evaluation`` BEFORE the row (pilot/supplement.py; HANDOVER.md section 8), so that a
                                reader that finds the row finds its record, with the reserve seeds that
                                replaced unstable episodes (``unstable_replacements``; Q-mujoco-exception):
                                the seed sets checked against the registry stay the planned ones

The supplement record is written inside the ledger's transaction, after the row and the seed sets
have been checked and before the ledger is replaced: a refused row writes no record and fixes no seed
set. A crash between the two leaves a record without its row; the retry accepts it when it is the same
record, or the same numbers reproduced by later code after a ``reevaluate`` (only ``code_commit``
differs: ``supplement.write``'s ``accept_reproduction``, safe here because the row is not in the
ledger yet), and keeps the record on disk unchanged.
``ensure_evaluation_supplement`` serves the scheduler's "row already present" path: it
writes the record still missing for a row already in the ledger, after checking it against the row and
the seed registry. Every ledger requires the record: a
completed run whose evaluation.json holds no per-episode arrays is refused wherever the ledger lives
(evaluation.json is written only by the launcher's evaluation stage, ``pilot.launch.evaluate``, which
stores the harnesses' contract-1 result with them; the tests' fake launcher writes the same form).

Pilot runs go to ``results/pilot/ledger.parquet`` and never to the main ledger ("The pilot's runs
are not reused", Part 3.6). The Study B pilot's unconstrained PPO run cannot be represented in
schema version 1 (its Study B arm list has no unconstrained arm), so it is written to a JSON
sidecar with the row's fields plus ``reason`` and ``evaluation`` (NaN written as null); Q-ledger-v2,
answered in Table 9.1: schema v1 stays frozen and this sidecar is the run's record. Smoke runs
(uncommitted code or open questions bypassed, in training or in evaluation: the smoke markers of
train_result.json and evaluation.json) are refused by every registered ledger (the repository's, by
location, and any ledger whose ``.mode`` marker says 'registered'; ``uses_registered_ledgers``) and
by their supplements (``uses_repository_ledgers`` covers the supplement directories derived from the
ledgers). The repository's ledgers also refuse any row whose
checkpoint paths are not relative to the schema's CHECKPOINT_ROOT (/data/checkpoints):
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
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Mapping, Sequence

from configs import registered as R
from pilot import contracts, provenance, supplement
from pilot.algorithms import BATCH_COST_KEY, BATCH_RETURN_KEY
from pilot.contracts import (
    PLASTICITY_COLUMNS,
    ContractError,
    check_canonical_seeds,
    checkpoint_steps,
    selection_window,
    validate_plasticity,
    validate_seed_set,
)
from pilot.launch import EVALUATION_RESULT, TRAIN_RESULT, read_progress
from pilot.manifest import RunSpec
from pilot.rundir import LEVEL_MULTIPLIER_PREFIX, MULTIPLIER_COLUMN

REPO_ROOT = provenance.REPO_ROOT
MULTIPLIER_TRACE_FILE = "multiplier_trace.csv"
TRAINING_COST_NOTE = "training_cost = OmniSafe Metrics/EpCost, the mean of the last 50 training episodes"
TRAINING_COST_NOTE_BATCH = "training_cost = mean episodic cost of the epoch's training batch (Metrics/BatchEpCost)"
STATISTIC_SUFFIXES = ("/Min", "/Max", "/Std")  # OmniSafe's min_and_max / std columns beside a logged value
REPOSITORY_RESULTS = REPO_ROOT / "results"


class LedgerWriteError(RuntimeError):
    """A run that may not be written to the ledger as it stands."""


@dataclass(frozen=True)
class LedgerPaths:
    """Where rows go. The defaults are the repository's ledger files (ledger memo, decision 4)."""

    main: Path = REPOSITORY_RESULTS / "ledger.parquet"
    pilot: Path = REPOSITORY_RESULTS / "pilot" / "ledger.parquet"
    sidecar_dir: Path = REPOSITORY_RESULTS / "pilot" / "sidecar"

    def for_spec(self, spec: RunSpec) -> Path:
        return self.pilot if spec.pilot else self.main

    def seeds_registry(self, spec: RunSpec) -> Path:
        """The evaluation seed sets fixed by the first run (Table 2.1), beside its ledger."""
        return seeds_registry(self.for_spec(spec))

    @property
    def supplement_dirs(self) -> tuple[Path, ...]:
        """The supplement directories of the main and the pilot ledger (the same one when they share a directory)."""
        dirs = (supplement.supplement_dir(self.main), supplement.supplement_dir(self.pilot))
        return dirs if dirs[0] != dirs[1] else dirs[:1]


DEFAULT_PATHS = LedgerPaths()


def seeds_registry(ledger_path: Path) -> Path:
    """``<ledger>.seeds.json``: the evaluation seed sets fixed by the first run (Table 2.1), beside the ledger."""
    ledger = Path(ledger_path)
    return ledger.with_name(ledger.stem + ".seeds.json")


def _in_repository_results(path: Path) -> bool:
    """True if ``path`` (symlinks resolved) is the repository's results/ directory or lies in it."""
    root = REPOSITORY_RESULTS.resolve()
    resolved = Path(os.path.abspath(path)).resolve()
    return resolved == root or root in resolved.parents


def uses_repository_ledgers(paths: LedgerPaths) -> bool:
    """True if any of the paths, or a supplement directory derived from a ledger, lies in the
    repository's results/ directory (the smoke guard of ``write_run`` covers all of them)."""
    return any(_in_repository_results(p)
               for p in (paths.main, paths.pilot, paths.sidecar_dir, *paths.supplement_dirs))


LEDGER_MODES = ("registered", "smoke")


def mode_marker(path: Path) -> Path:
    """``<ledger>.mode`` (or ``<sidecar_dir>.mode``): the mode of the data roots writing there, fixed by the first."""
    path = Path(os.path.abspath(path))
    return path.with_name(path.name + ".mode")


def ledger_mode(path: Path) -> str | None:
    """The mode a ledger (or sidecar directory) is marked with: 'registered' for the repository's
    results/ (by location) or for a marker saying so, 'smoke', or None (not marked yet). A marker that is
    empty or names no mode is a LedgerWriteError naming the file (markers are written whole, so a crash never
    leaves an empty marker; an empty one is not a mode)."""
    if _in_repository_results(path):
        return "registered"
    marker = mode_marker(path)
    try:
        found = marker.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    if found not in LEDGER_MODES:
        raise LedgerWriteError(f"{marker} holds {found!r}, not one of {LEDGER_MODES}: the ledger's mode is unknown; "
                               "the operator checks which data root wrote there and fixes or removes the marker")
    return found


def uses_registered_ledgers(paths: LedgerPaths) -> bool:
    """True if any ledger of ``paths`` (or its sidecar directory) is registered (repository or marker)."""
    return uses_repository_ledgers(paths) or any(
        ledger_mode(p) == "registered" for p in (paths.main, paths.pilot, paths.sidecar_dir))


def mark_ledger_mode(paths: LedgerPaths, mode: str) -> None:
    """Fix ``mode`` on the ledgers of ``paths`` and their sidecar directory, or refuse (LedgerWriteError).

    A data root's mode is fixed per data root (``pilot.scheduler``), but two data roots can name the same
    ledgers (``--ledger-dir``): the marker beside each ledger keeps a smoke root off a registered root's
    ledgers wherever they lie, and keeps a registered root off a smoke root's. The repository's ledgers are
    registered by location and carry no marker.
    """
    if mode not in LEDGER_MODES:
        raise ValueError(f"mode must be one of {LEDGER_MODES}, got {mode!r}")
    for p in (paths.main, paths.pilot, paths.sidecar_dir):
        found = ledger_mode(p)
        if found is None:
            try:  # the first data root fixes it; the name appears only with its content (fsync, then link)
                provenance.write_text_once(mode_marker(p), mode + "\n")
            except FileExistsError:
                pass
            found = ledger_mode(p)
        if found != mode:
            raise LedgerWriteError(f"{p} is a {found} ledger ({mode_marker(p)}); a {mode} data root may not write "
                                   "there (smoke and registered runs never share ledgers)")


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


def _finite_or_nan(value: float) -> float:
    """``value``, or NaN for +/-inf (the ledger schema stores a non-finite training statistic as NaN)."""
    return value if not math.isinf(value) else math.nan


def _progress_by_epoch(rows: list[dict[str, str]]) -> dict[int, dict[str, str]]:
    return {int(float(r["Train/Epoch"])): r for r in rows if r.get("Train/Epoch") not in (None, "")}


def _read_plasticity(spec: RunSpec, omnisafe_dir: Path, completed: bool) -> dict[int, dict[str, float]]:
    """{step: {dormant, rank, norm}} from plasticity.csv, checked by contract 2 (HANDOVER.md section 8).

    ``pilot.contracts.validate_plasticity`` checks the file (a completed run of a Study A plug-in
    must have a row for every checkpoint it saved; a failed run may lack rows). Only the three
    columns the ledger's checkpoint record holds are returned (ledger schema v1); the others stay
    in the run directory and reach the ``training`` supplement record (pilot/enrichment.py).
    """
    try:
        rows = validate_plasticity(omnisafe_dir, spec, completed=completed)
    except ContractError as exc:
        raise LedgerWriteError(str(exc)) from None
    return {step: {k: values[k] for k in PLASTICITY_COLUMNS[1:]} for step, values in rows.items()}


def multiplier_columns(header: list[str] | tuple[str, ...]) -> list[str]:
    """The progress.csv columns that hold a multiplier (contract 3), without OmniSafe's statistics."""
    return [
        c for c in header
        if (c == MULTIPLIER_COLUMN or c.startswith(LEVEL_MULTIPLIER_PREFIX)) and not c.endswith(STATISTIC_SUFFIXES)
    ]


def _level(column: str) -> float:
    """``Metrics/LagrangeMultiplier/level_17.5`` -> 17.5."""
    try:
        return float(column[len(LEVEL_MULTIPLIER_PREFIX):])
    except ValueError:
        raise LedgerWriteError(f"{column!r}: a per-level multiplier column must end in the budget level (contract 3)") from None


def write_multiplier_trace(omnisafe_dir: Path, rows: list[dict[str, str]]) -> Path | None:
    """Per-epoch trace of every multiplier (Part 3.4: "every per-level multiplier ... at every epoch").

    Columns: epoch, step_end, then ``lagrange_multiplier`` for the single multiplier and
    ``level_{budget}`` (for the Continuous arm, the lower edge of a 5-unit bin) for each of Study B's
    per-level multipliers. The file is written through a temporary file and renamed into place, so
    that a crash never leaves a partial trace.
    """
    columns = multiplier_columns(list(rows[0])) if rows else []
    if not columns:
        return None
    names = ["lagrange_multiplier" if c == MULTIPLIER_COLUMN else c[len(MULTIPLIER_COLUMN) + 1:] for c in columns]
    path = omnisafe_dir / MULTIPLIER_TRACE_FILE
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["epoch", "step_end", *names])
            for epoch, row in sorted(_progress_by_epoch(rows).items()):
                writer.writerow([epoch, (epoch + 1) * R.STEPS_PER_EPOCH, *(row[c] for c in columns)])
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def per_level_multipliers(rows: list[dict[str, str]]) -> dict[float, float] | None:
    """Study B: each level's multiplier after the last epoch.

    None without per-level columns, without a row that names its epoch (a truncated progress.csv), or
    when any level's last value is non-finite or negative, as in a failed run.
    """
    columns = [c for c in multiplier_columns(list(rows[0])) if c.startswith(LEVEL_MULTIPLIER_PREFIX)] if rows else []
    if not columns:
        return None
    by_epoch = _progress_by_epoch(rows)
    if not by_epoch:
        return None
    last = by_epoch[max(by_epoch)]
    values = {_level(c): _float(last[c]) for c in columns}
    return values if all(math.isfinite(v) and v >= 0 for v in values.values()) else None


def _initial_multiplier(spec: RunSpec) -> float | None:
    """The multiplier in effect at step 0, where the algorithm fixes it.

    Table 2.1 'Abrupt onset': at onset the multiplier is at its initial value, 0.001 for
    PPO-Lagrangian (Table A.1), so it is not updated before onset; the controller variants of
    Table 2.4 change it only at onset, and every controller arm has N > 0, so step 0 is before
    onset. OmniSafe's PID multiplier (``cost_penalty``) is 0.0 until its first update. PPO has no
    multiplier, and Study B's are per level.
    """
    if spec.study != "A":
        return None
    if spec.base_algo == "CPPOPID":
        return 0.0
    if spec.base_algo == "PPOLag" and (spec.controller_variant is None or (spec.onset_step or 0) > 0):
        return R.LAGRANGE_MULTIPLIER_INIT
    return None


def build_row_fields(spec: RunSpec, run_dir: Path, attempt: int | None = None,
                     extra_notes: Sequence[str] = ()) -> dict[str, Any]:
    """Collect the ledger fields of one run as plain Python values, and (re)write the run's
    multiplier_trace.csv (Part 3.4), whose path the row records.

    The trace is written even when the row is refused later; it is a function of progress.csv alone.
    The step-0 checkpoint (the untrained policy) has no training epoch behind it, so its training
    cost and return are NaN. Non-finite plasticity metrics are left null (noted); a non-finite
    selection-set cost or return is refused, since rule 1 cannot rank it.
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
    plasticity = _read_plasticity(spec, omnisafe_dir, completed) if omnisafe_dir else {}
    steps = checkpoint_steps(omnisafe_dir) if omnisafe_dir else []
    selection: dict[int, list[float]] = {}
    if evaluation is not None:
        selection = {_step_key(k, f"{spec.run_id}: {EVALUATION_RESULT} selection"): v
                     for k, v in evaluation["selection"].items()}
        try:
            window = selection_window(steps, spec.total_steps)
        except contracts.SelectionWindowPending:
            raise  # a refusal while Q-selection-window is open (pilot.errors), not a broken run
        except ContractError as exc:  # a broken checkpoint set: refused as contract 2's breaches are
            raise LedgerWriteError(f"{spec.run_id}: {exc}") from None
        if sorted(selection) != window:
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
            # NaN where there is no finite statistic (the untrained step-0 checkpoint, a diverged epoch): the
            # schema refuses +/-inf
            "training_cost": _finite_or_nan(_float(row[cost_col])) if row else math.nan,
            "training_return": _finite_or_nan(_float(row[return_col])) if row else math.nan,
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
        logged = sorted(_level(c) for c in multiplier_columns(list(rows[0])) if c.startswith(LEVEL_MULTIPLIER_PREFIX))
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
    notes.extend(extra_notes)  # e.g. manifest.AUXILIARY_NOTE (the scheduler's record of an auxiliary run)
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


SMOKE_MARKERS = ("worktree_dirty", "allow_dirty", "allow_pending")  # train_result.json (pilot/launch.py)
EVAL_SMOKE_MARKERS = ("eval_worktree_dirty", "eval_allow_dirty", "eval_allow_pending")  # evaluation.json


def _is_smoke(run_dir: Path) -> bool:
    """True when training or evaluation ran with uncommitted code or with open questions bypassed.

    ``pilot.launch.evaluate`` records its own smoke markers in evaluation.json, so a run trained
    cleanly but evaluated as a smoke run is a smoke run too.
    """
    train = json.loads((Path(run_dir) / TRAIN_RESULT).read_text(encoding="utf-8"))
    if any(train.get(k) for k in SMOKE_MARKERS):
        return True
    eval_path = Path(run_dir) / EVALUATION_RESULT
    if not eval_path.exists():
        return False
    evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
    return any(evaluation.get(k) for k in EVAL_SMOKE_MARKERS)


# ---------------------------------------------------------------------------
# The ``evaluation`` supplement record (HANDOVER.md section 8)
# ---------------------------------------------------------------------------


def _episode_list(value: Any, what: str) -> list[float]:
    if not isinstance(value, (list, tuple)):
        raise LedgerWriteError(f"{what} must be a list of the {R.EVAL_EPISODES} episodes, got {type(value).__name__}")
    return list(value)


def _step_key(key: Any, what: str) -> int:
    """A checkpoint step as JSON stores a map key ("200000") or as a harness returns it (200000)."""
    if isinstance(key, int) and not isinstance(key, bool) and key >= 0:
        return key
    if isinstance(key, str) and key.isascii() and key.isdigit():
        return int(key)
    raise LedgerWriteError(f"{what}: {key!r} is not a checkpoint step")


def _replacements(value: Any, seeds: list[int], what: str) -> list[dict[str, Any]] | None:
    """One evaluation's ``unstable_replacements``, checked against its planned seeds (Q-mujoco-exception):
    the replaced episodes' reserve seeds lie in the set's reserve range, are distinct and at most
    ``contracts.MAX_UNSTABLE_EPISODES``; None (the record's form) when no episode was replaced."""
    try:
        return contracts.validate_unstable_replacements(what, value, seeds) or None
    except ContractError as exc:
        raise LedgerWriteError(str(exc)) from None


def evaluation_record(spec: RunSpec, evaluation: Mapping[str, Any]) -> dict[str, Any] | None:
    """The ``evaluation`` supplement record of contract 1's result (HANDOVER.md section 8), or None without per-episode arrays.

    Table 2.1: the selection set on each of the last ten checkpoints (Part 4.1 rule 1) and the final
    checkpoint on the measurement set (Q-final-cost-set, Table 9.1: ``final_seed_set`` must name it;
    any other set is refused); Part 5.3 resamples the evaluation episodes, so every episode's cost and return is kept.
    Built from evaluation.json as ``pilot.launch.evaluate`` wrote it: ``episode_costs`` and
    ``episode_returns`` ({"final": [...], "selection": {step: [...]}}), ``final_seed_set``, the seed
    sets, ``eval_commit_hash`` (the code that evaluated), ``short_episodes``, for a
    budget-conditioned run ``studyb_budgets`` (the budget each episode carried; Q-studyb-eval), and
    ``unstable_replacements`` when present (Q-mujoco-exception: each evaluation's replacements,
    checked by ``pilot.contracts.validate_unstable_replacements`` against its planned seeds, which the
    record keeps). The record's means are the ledger's final_cost/final_return and selection costs,
    and the schema checks each against its episodes. LedgerWriteError when evaluation.json holds
    some of these but not in the form of contract 1.
    """
    costs, returns = evaluation.get("episode_costs"), evaluation.get("episode_returns")
    if costs is None and returns is None:
        return None
    what = f"{spec.run_id}: {EVALUATION_RESULT}"
    if not isinstance(costs, Mapping) or not isinstance(returns, Mapping):
        raise LedgerWriteError(f"{what}: episode_costs and episode_returns must both map 'final' and 'selection'")
    commit = evaluation.get("eval_commit_hash")
    if not isinstance(commit, str) or not provenance.is_full_commit_hash(commit):
        raise LedgerWriteError(f"{what}: eval_commit_hash {commit!r} is not a full commit hash (the code that evaluated)")
    final_set = evaluation.get("final_seed_set")
    if final_set != "measurement":
        raise LedgerWriteError(f"{what}: final_seed_set {final_set!r} is not the measurement set; final_cost and "
                               "final_return are the measurement set's (Q-final-cost-set, Table 9.1)")
    try:
        selection_seeds = validate_seed_set("selection_seeds", evaluation.get("selection_seeds"))
        final_seeds = validate_seed_set("measurement_seeds", evaluation.get("measurement_seeds"))
    except (ContractError, TypeError) as exc:
        raise LedgerWriteError(f"{what}: {exc}") from None
    if set(final_seeds) & set(selection_seeds):
        raise LedgerWriteError(f"{what}: the measurement and selection seeds overlap (Table 2.1 requires disjoint sets)")
    budgets = evaluation.get("studyb_budgets")
    if spec.plugin == "study_b":
        if not isinstance(budgets, (list, tuple)) or len(budgets) != R.EVAL_EPISODES:
            raise LedgerWriteError(f"{what}: a budget-conditioned run records the budget of each of its "
                                   f"{R.EVAL_EPISODES} episodes (studyb_budgets; Q-studyb-eval)")
        budgets = list(budgets)
    elif budgets is not None:
        raise LedgerWriteError(f"{what}: studyb_budgets given for {spec.run_id}, whose plug-in {spec.plugin!r} takes no budget")
    selection = evaluation.get("selection")
    if not isinstance(selection, Mapping) or not isinstance(costs.get("selection"), Mapping) \
            or not isinstance(returns.get("selection"), Mapping):
        raise LedgerWriteError(f"{what}: selection, episode_costs['selection'] and episode_returns['selection'] must map steps")
    by_step = {_step_key(k, f"{what} selection"): v for k, v in selection.items()}
    cost_by_step = {_step_key(k, f"{what} episode_costs"): v for k, v in costs["selection"].items()}
    return_by_step = {_step_key(k, f"{what} episode_returns"): v for k, v in returns["selection"].items()}
    if not (sorted(by_step) == sorted(cost_by_step) == sorted(return_by_step)):
        raise LedgerWriteError(f"{what}: the per-episode arrays are not given for exactly the selection checkpoints")
    final_step = evaluation.get("final_step", spec.total_steps)
    if isinstance(final_step, bool) or final_step != spec.total_steps:
        raise LedgerWriteError(f"{what}: final_step {final_step!r} is not the run's total {spec.total_steps}")
    replaced = evaluation.get("unstable_replacements") or {}
    if not isinstance(replaced, Mapping) or not isinstance(replaced.get("selection") or {}, Mapping):
        raise LedgerWriteError(f"{what}: unstable_replacements must map 'final' and 'selection' (Q-mujoco-exception)")
    replaced_by_step = {_step_key(k, f"{what} unstable_replacements"): v
                        for k, v in (replaced.get("selection") or {}).items()}
    if not set(replaced_by_step) <= set(by_step):
        raise LedgerWriteError(f"{what}: unstable_replacements at steps {sorted(set(replaced_by_step) - set(by_step))} "
                               "that are not selection checkpoints")
    blocks = []
    for step in sorted(by_step):
        pair = by_step[step]
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise LedgerWriteError(f"{what}: selection[{step}] must be [cost, return]")
        blocks.append({
            "step": step, "seed_set": "selection", "seeds": selection_seeds,
            "episode_costs": _episode_list(cost_by_step[step], f"{what} episode_costs[selection][{step}]"),
            "episode_returns": _episode_list(return_by_step[step], f"{what} episode_returns[selection][{step}]"),
            "mean_cost": pair[0], "mean_return": pair[1], "budgets": budgets,
            "unstable_replacements": _replacements(replaced_by_step.get(step), selection_seeds,
                                                   f"{what} selection[{step}]"),
        })
    return {
        "kind": "evaluation",
        "run_id": spec.run_id,
        "code_commit": commit,
        "final_step": spec.total_steps,
        "final": {
            "seed_set": final_set, "seeds": final_seeds,
            "episode_costs": _episode_list(costs.get("final"), f"{what} episode_costs[final]"),
            "episode_returns": _episode_list(returns.get("final"), f"{what} episode_returns[final]"),
            "mean_cost": evaluation.get("final_cost"), "mean_return": evaluation.get("final_return"),
            "budgets": budgets,
            "unstable_replacements": _replacements(replaced.get("final"), final_seeds, f"{what} final"),
        },
        "selection": blocks,
        "short_episodes": evaluation.get("short_episodes"),
    }


def _validated_record(kind: str, record: Mapping[str, Any]) -> Any:
    """``results.supplement_schema.validate_record`` with its errors as LedgerWriteError."""
    from results.supplement_schema import validate_record

    try:
        return validate_record(kind, record)
    except ValueError as exc:
        raise LedgerWriteError(f"{record.get('run_id')}: not a valid {kind} supplement record: {exc}") from exc


def _check_seeds_fit(kind: str, seeds: list[int], registry: Path) -> None:
    """``check_canonical_seeds`` without writing: raises what it would raise, changes nothing.

    Called before the supplement record is written, so that a row refused for its seeds leaves
    no record behind (the record is written once; a re-evaluated run would produce another one).
    """
    contracts.check_canonical_seeds(kind, seeds, registry, write=False)


def _seed_checks(record: Mapping[str, Any]) -> list[tuple[str, list[int]]]:
    """The (kind, seeds) pairs a completed run fixes in the seed registry (Table 2.1).

    ``record`` is a validated ``evaluation`` record (``evaluation_record`` has checked its seed sets).
    """
    return [("selection", list(record["selection"][0]["seeds"])), ("measurement", list(record["final"]["seeds"]))]


def _sidecar_evaluation(spec: RunSpec, evaluation: Mapping[str, Any], fields: dict[str, Any]
                        ) -> list[tuple[str, list[int]]]:
    """The seed checks of a sidecar run (the pilot's unconstrained run), and its evaluation in ``fields``.

    Its final_cost is G4's "unconstrained cost" (pilot go report), so it obeys Table 2.1 as a ledger row
    does: the selection set, and the measurement set its final cost comes from (Q-final-cost-set,
    Table 9.1), must be the canonical ones of the pilot ledger's seed registry. The
    sidecar records which seeds and episodes produced that cost (``fields["evaluation"]``, with the
    replacements of unstable episodes; Q-mujoco-exception), as the
    ``evaluation`` supplement record does for a ledger row. The evaluation is held to the checks of a
    ledger row's ``evaluation`` record (``evaluation_record`` and the supplement schema): final_step is
    the run's total, the final and selection checkpoints carry R.EVAL_EPISODES per-episode costs
    (finite, non-negative) and returns, and final_cost and final_return are the means of the final
    episodes. LedgerWriteError otherwise (no record is written; G4 never reads a contradicted cost).
    """
    what = f"{spec.run_id}: {EVALUATION_RESULT}"
    record = evaluation_record(spec, evaluation)
    if record is None:
        raise LedgerWriteError(
            f"{what} holds no per-episode costs and returns; the unconstrained cost (G4) must be checked against "
            "its episodes as a ledger row's is (Part 5.3): re-evaluate with the current harness"
        )
    _validated_record("evaluation", record)
    final = record["final"]
    fields["evaluation"] = {
        "final_seed_set": final["seed_set"], "final_seeds": final["seeds"], "final_step": record["final_step"],
        "eval_commit_hash": record["code_commit"],
        "final_episode_costs": final["episode_costs"], "final_episode_returns": final["episode_returns"],
        "final_unstable_replacements": final["unstable_replacements"] or [],  # Q-mujoco-exception
    }
    return _seed_checks(record)


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
    """Write ``fields`` as JSON to ``target`` once (``provenance.write_text_once``: fsync, then link);
    LedgerWriteError if the target exists.

    Non-finite floats (the step-0 checkpoint's training cost and return) are written as null, so
    that the file is strict JSON; the ledger itself stores them as NaN.
    """
    def default(value: Any) -> Any:
        if isinstance(value, datetime):
            return value.isoformat()
        raise TypeError(f"{type(value).__name__} is not JSON serialisable")

    text = json.dumps(_json_safe(fields), indent=2, default=default, sort_keys=True, allow_nan=False)
    try:
        provenance.write_text_once(target, text)  # atomic, and never replaces an existing record
    except FileExistsError:
        raise LedgerWriteError(f"{target} already exists; a run is recorded once") from None


def write_run(spec: RunSpec, run_dir: Path, paths: LedgerPaths = DEFAULT_PATHS, attempt: int | None = None,
              extra_notes: Sequence[str] = ()) -> Path:
    """Validate and append one run's row, with its ``evaluation`` supplement record. Returns the file written to.

    The evaluation seeds are checked against the registry only after the row validates, under the
    ledger's lock, so a rejected row never fixes the seed sets (Table 2.1). The supplement record
    (HANDOVER.md section 8) is built and validated before anything is written, and written inside the
    ledger's transaction after the row and the seeds passed, before the ledger is replaced: whoever
    finds the row finds its record. A record left by a crash before the ledger was replaced is
    accepted when it is this one or differs from it only in ``code_commit`` (a re-evaluation by later
    code; it is kept unchanged). Every completed row requires its record, wherever the ledger lives:
    an evaluation.json without per-episode arrays is refused.

    Raises LedgerWriteError for a run that may not be written as it stands (a broken checkpoint set
    included), ContractError (``pilot.contracts``) for evaluation seed sets that differ from or
    overlap the seed registry's (Table 2.1), ``pilot.contracts.SelectionWindowPending`` (a
    PendingQuestionError) for a total off the checkpoint grid while Q-selection-window is open
    (unless ``pilot.errors.allow_pending`` is in effect), pydantic's ValidationError for a row the schema refuses, and
    ``supplement.SupplementConflict`` for a differing ``evaluation`` record already on disk.
    """
    from results.ledger_schema import CHECKPOINT_ROOT, CheckpointRecord, LedgerRow, append_to_ledger

    repository = uses_repository_ledgers(paths)
    if _is_smoke(run_dir) and uses_registered_ledgers(paths):  # the repository's, or ones marked registered elsewhere
        raise LedgerWriteError(
            f"{spec.run_id} ran with uncommitted code or bypassed open questions (a smoke run); "
            "it may not be written to the repository's ledgers or to ledgers marked registered"
        )
    fields = build_row_fields(spec, run_dir, attempt, extra_notes)
    absolute = _absolute_paths(fields)
    if repository and absolute:
        raise LedgerWriteError(
            f"{spec.run_id}: checkpoint paths {absolute[:2]} are not under {CHECKPOINT_ROOT}; keep the data root "
            f"at {PurePosixPath(CHECKPOINT_ROOT).parent} (a symlink to the real storage) so that the ledger stays portable"
        )
    evaluation: dict[str, Any] | None = None
    if fields["completed"]:
        evaluation = json.loads((Path(run_dir) / EVALUATION_RESULT).read_text(encoding="utf-8"))
    if not _representable(spec):
        paths.sidecar_dir.mkdir(parents=True, exist_ok=True)
        target = paths.sidecar_dir / f"{spec.run_id}.json"
        fields["reason"] = "not representable in ledger schema version 1 (no unconstrained Study B arm)"
        sidecar_checks = _sidecar_evaluation(spec, evaluation, fields) if evaluation is not None else []
        registry = paths.seeds_registry(spec)
        with _file_lock(paths.for_spec(spec)):  # as for a ledger row: all seeds fit, the record, then the registry
            if target.exists():
                raise LedgerWriteError(f"{target} already exists; a run is recorded once")
            for kind, seeds in sidecar_checks:
                _check_seeds_fit(kind, seeds, registry)
            _write_sidecar(target, fields)
            for kind, seeds in sidecar_checks:
                check_canonical_seeds(kind, seeds, registry)
        return target
    record = evaluation_record(spec, evaluation) if evaluation is not None else None
    if record is not None:
        _validated_record("evaluation", record)
    elif evaluation is not None:
        raise LedgerWriteError(
            f"{spec.run_id}: {EVALUATION_RESULT} holds no per-episode costs and returns; every row needs its "
            "evaluation supplement record (Part 5.3): re-evaluate with the current harness"
        )
    checks = _seed_checks(record) if record is not None else []
    fields["checkpoints"] = [CheckpointRecord(**c) for c in fields["checkpoints"]]
    row = LedgerRow(**fields)
    path = paths.for_spec(spec)
    registry = paths.seeds_registry(spec)
    with ledger_transaction(path) as tmp:
        append_to_ledger(row, tmp)  # into the working copy; the ledger changes only if everything below passes
        for kind, seeds in checks:
            _check_seeds_fit(kind, seeds, registry)
        if record is not None:  # the row is not in the ledger (append_to_ledger refuses duplicates): a record on
            # disk is a crash's leftover, which a deterministic re-evaluation reproduces up to its code_commit
            supplement.write("evaluation", spec.run_id, record, ledger_path=path, accept_reproduction=True)
        for kind, seeds in checks:
            check_canonical_seeds(kind, seeds, registry)
    return path


def ensure_evaluation_supplement(spec: RunSpec, run_dir: Path, paths: LedgerPaths = DEFAULT_PATHS) -> Path | None:
    """Write the ``evaluation`` record of a run whose row is already in the ledger, if it is missing.

    For the scheduler's "row already present" path, e.g. a row written before the supplement existed,
    or a row whose ``write_run`` crashed after its record and was retried. The run directory must be the
    row's run (same run_id and training commit); the record's final cost and return, and its selection
    checkpoints with their costs and returns, must be the row's; and its seed sets must be those of the
    seed registry (Table 2.1). A record on disk that differs from this one only in ``code_commit`` is
    accepted and kept (``write_run``'s crash recovery may have kept one evaluated by earlier code; the
    numbers have just been checked against the row). Returns the record's path, or None for a row
    without an evaluation (a failed run) and for a run recorded in the sidecar. LedgerWriteError for a
    smoke run of a registered ledger (the repository's or one marked registered), for an
    evaluation.json without per-episode arrays (as ``write_run`` refuses it) and for any of the
    mismatches above; ``supplement.SupplementConflict`` for a record on disk with other numbers.
    """
    from results.ledger_schema import load_ledger_as_rows

    if not _representable(spec):
        return None
    run_dir = Path(os.path.abspath(run_dir))
    if _is_smoke(run_dir) and uses_registered_ledgers(paths):
        raise LedgerWriteError(f"{spec.run_id} is a smoke run; it may not reach a registered ledger's supplement")
    ledger = paths.for_spec(spec)
    with _file_lock(ledger):
        found = [r for r in load_ledger_as_rows(ledger) if r.run_id == spec.run_id] if ledger.exists() else []
        if not found:
            raise LedgerWriteError(f"{spec.run_id} is not in {ledger}")
        row = found[0]
        if not row.completed:
            return None
        train = json.loads((run_dir / TRAIN_RESULT).read_text(encoding="utf-8"))
        if train.get("run_id") != spec.run_id or train.get("commit_hash") != row.commit_hash:
            raise LedgerWriteError(f"{run_dir} is not the run the ledger row of {spec.run_id} records (run_id "
                                   f"{train.get('run_id')!r}, commit {train.get('commit_hash')!r})")
        evaluation = json.loads((run_dir / EVALUATION_RESULT).read_text(encoding="utf-8"))
        record = evaluation_record(spec, evaluation)
        if record is None:
            raise LedgerWriteError(f"{spec.run_id}: {EVALUATION_RESULT} holds no per-episode arrays (Part 5.3)")
        model = _validated_record("evaluation", record)
        if (model.final.mean_cost, model.final.mean_return) != (row.final_cost, row.final_return):
            raise LedgerWriteError(
                f"{spec.run_id}: {EVALUATION_RESULT} gives final cost and return {model.final.mean_cost!r}, "
                f"{model.final.mean_return!r}; the ledger row records {row.final_cost!r}, {row.final_return!r}"
            )
        row_selection = {c.step: (c.selection_cost, c.selection_return)
                         for c in row.checkpoints if c.selection_cost is not None}
        record_selection = {b.step: (b.mean_cost, b.mean_return) for b in model.selection}
        if record_selection != row_selection:
            raise LedgerWriteError(
                f"{spec.run_id}: {EVALUATION_RESULT} gives selection costs and returns {record_selection!r}; "
                f"the ledger row records {row_selection!r}"
            )
        try:
            for kind, seeds in _seed_checks(record):
                _check_seeds_fit(kind, seeds, paths.seeds_registry(spec))
        except ContractError as exc:
            raise LedgerWriteError(f"{spec.run_id}: {EVALUATION_RESULT}: {exc}") from None
        # the numbers are the row's: a record that differs only in code_commit is the same evaluation
        return supplement.write("evaluation", spec.run_id, record, ledger_path=ledger, accept_reproduction=True)


def recorded_run_ids(ledger_path: Path) -> set[str]:
    """run_ids already in a ledger (empty if the ledger does not exist yet)."""
    if not Path(ledger_path).exists():
        return set()
    import pyarrow.parquet as pq

    return set(pq.read_table(ledger_path, columns=["run_id"]).column("run_id").to_pylist())


def recorded_anywhere(spec: RunSpec, paths: LedgerPaths = DEFAULT_PATHS) -> bool:
    """True if the run has a ledger row or a sidecar record."""
    if spec.run_id in recorded_run_ids(paths.for_spec(spec)):
        return True
    return (paths.sidecar_dir / f"{spec.run_id}.json").exists()

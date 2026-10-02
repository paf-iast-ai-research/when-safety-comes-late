"""The plasticity hook: ``plasticity.csv`` at every checkpoint (Role 3, Metrics and interventions; owner Abdullah).

Implements Table 2.3 "Logging points" ("Every 200,000 steps, that is every ten epochs (decision), at
onset, and at the end of training") and Part 3.4 ("Each run saves a checkpoint every 200,000 steps,
at onset, and at the end of training, and logs at each checkpoint: ... the three plasticity metrics
on the fixed batch (Study A)"), as contract 2 of ``pilot/contracts.py``.

Installed once by ``pilot.algorithms.FullStateCheckpointMixin._init_log`` when
``cfgs.pilot_cfgs.plasticity`` is true (the wiring is Role 1's; contract 2 in HANDOVER.md section 8),
at the end of ``_init_log``, after the step-0 checkpoint exists. A row is written:

* at install: the step-0 row (the untrained state of ``epoch-0.pt``; for N = 0 it is the onset);
* after every later ``logger.torch_save()``, through a wrapper of the logger instance's
  ``torch_save`` attribute that chains whatever was there before: OmniSafe's cadence and final
  saves (``policy_gradient.py:286-289``) and every extra save of the mixin (onset, onset +
  200,000 steps; ``pilot.contracts.extra_checkpoint_steps``) look the attribute up at call time, so
  each checkpoint gets its row.

Nothing else writes a row (contract 2: one row per checkpoint, no "live" rows between checkpoints),
a step is never written twice (a set of written steps), and the step of a row is the checkpoint's:
``logger.current_epoch x algo_cfgs.steps_per_epoch`` of the run itself (``epoch-k.pt`` holds the state after k epochs,
``common/logger.py:178-189``). Each row is flushed when written, so a crashed run keeps the rows it
reached. The hook never transforms the actor and never guards the rollout (HANDOVER.md section 8,
``metrics.interventions``: the Study A plug-in applies the interventions after the onset row, so
onset metrics are pre-intervention).

Neutrality (HANDOVER.md section 10, "Instrumentation never changes training"): no random number is
drawn; ``Normalizer.normalize`` is never called (it would push the batch into the running
statistics); only ``actor.mean`` and the critics' MLPs are evaluated, under ``torch.no_grad``, with
forward hooks that are removed at once; no parameter, buffer, optimiser state or module mode
changes. Training is bit-for-bit that of a run without the hook.

Error policy: a problem with the batch or the configuration raises ``RunRefused`` at install,
before training (the launcher exits 5 and the run stays pending). A failure of the metric
computation during training never propagates into ``learn()`` (Part 5.6: an exception there would be
a crash, hence an exclusion): the row is written with NaN and the error is recorded in
``plasticity_meta.json``. A failure to write the records during training (an ``OSError`` appending
the row or rewriting ``plasticity_meta.json``) deliberately does propagate: a run that cannot keep
its records crashes rather than finish without them, and ``pilot.contracts.validate_plasticity``
relies on it (a completed run finished every append, so only a failed run's last line may be cut
short).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from configs import registered as R
from metrics.batches import (FIXED_BATCH_SHA256, REPO_RELATIVE_DIR, atomic_write_bytes, load_fixed_batch, npy_bytes,
                             sha256_hex)
from metrics.plasticity import COLUMNS, as_batch_tensor, format_row, measure, nan_row
from pilot.contracts import PLASTICITY_FILE
from pilot.errors import RunRefused

PLASTICITY_META_FILE = "plasticity_meta.json"
METRICS_VERSION = 1
RECORDER_ATTRIBUTE = "_plasticity_recorder"


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """``payload`` as strict JSON (sorted keys, no NaN, trailing newline), written atomically."""
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    atomic_write_bytes(path, text.encode("utf-8"))


class PlasticityRecorder:
    """Writes ``plasticity.csv`` and ``plasticity_meta.json`` in the OmniSafe run directory of one run."""

    def __init__(self, source: Any, batch: torch.Tensor, log_dir: Path, steps_per_epoch: int,
                 meta: dict[str, Any]) -> None:
        self._source = source
        self._batch = batch
        self.log_dir = Path(log_dir)
        self.path = self.log_dir / PLASTICITY_FILE
        self.meta_path = self.log_dir / PLASTICITY_META_FILE
        self.steps_per_epoch = int(steps_per_epoch)
        self.meta = dict(meta, errors=[])
        self.rows: list[dict[str, int | float]] = []
        self._written: set[int] = set()

    @property
    def written_steps(self) -> list[int]:
        """Steps that have a row in plasticity.csv, ascending."""
        return sorted(self._written)

    @property
    def errors(self) -> list[str]:
        """A copy of the measurement errors recorded in plasticity_meta.json."""
        return list(self.meta["errors"])

    def start(self, first_row: dict[str, int | float]) -> None:
        """Create both files (refusing to overwrite either) and write the first row.

        All or nothing: if any step fails, the files this call created are removed before the
        exception propagates, so a refused install leaves the run directory as it found it.
        """
        created: list[Path] = []
        try:
            open(self.meta_path, "x", encoding="utf-8").close()  # "x": never replace a meta file not ours
            created.append(self.meta_path)
            _write_json_atomic(self.meta_path, self.meta)
            with open(self.path, "x", newline="", encoding="utf-8") as fh:  # "x": never overwrite a run's rows
                created.append(self.path)
                csv.writer(fh, lineterminator="\n").writerow(COLUMNS)
            self._append(first_row)
        except BaseException:
            for path in created:
                try:
                    path.unlink()
                except OSError:
                    pass  # best effort; the original error is the one to report
            self._written.clear()
            self.rows.clear()
            raise

    def _append(self, row: dict[str, int | float]) -> None:
        with open(self.path, "a", newline="", encoding="utf-8") as fh:
            csv.writer(fh, lineterminator="\n").writerow(format_row(row))
            fh.flush()
        self._written.add(int(row["step"]))
        self.rows.append(dict(row))

    def record(self, step: int) -> dict[str, int | float] | None:
        """Measure the current state and write its row at ``step``, unless that step has a row already.

        A measurement failure gives a NaN row and an entry in ``errors``; an ``OSError`` while writing
        the row or the error record propagates (module docstring, "Error policy").
        """
        step = int(step)
        if step in self._written:
            return None
        try:
            row = measure(self._source, self._batch, step=step)
        except Exception as exc:  # noqa: BLE001 - Part 5.6: a metrics failure must not crash the run
            row = nan_row(step)
            self.meta["errors"].append(f"step {step}: {type(exc).__name__}: {exc}")
            _write_json_atomic(self.meta_path, self.meta)
        self._append(row)
        return row


def _pilot_cfgs(cfgs: Any) -> Any:
    getter = getattr(cfgs, "get", None)
    return getter("pilot_cfgs") if getter else None


def _checked_batch(batch: Any, obs_dim: int, what: str) -> np.ndarray:
    array = np.asarray(batch)
    if array.ndim != 2 or array.shape[0] < 1 or not np.issubdtype(array.dtype, np.number):
        raise RunRefused(
            f"{what}: a non-empty 2-D numeric array is required, got shape {array.shape} dtype {array.dtype}"
        )
    if array.shape[1] != obs_dim:
        raise RunRefused(f"{what} has {array.shape[1]} observation components; the run's actor takes {obs_dim}")
    if not np.isfinite(array).all():
        raise RunRefused(f"{what} contains non-finite values")
    return array


def install_plasticity_hook(algo: Any, *, batch: np.ndarray | None = None) -> PlasticityRecorder:
    """Start ``plasticity.csv`` for ``algo`` (step-0 row) and write a row after every later checkpoint.

    ``algo`` is a constructed OmniSafe on-policy algorithm (at the end of ``_init_log``, after the step-0
    checkpoint is saved; before ``learn``).
    ``batch`` is for tests only (tiny configurations); production passes nothing and measures on the
    committed fixed batch of ``algo._cfgs.env_id`` (``metrics.batches.load_fixed_batch``).
    Raises ``RunRefused`` (never anything else for a configuration or batch problem) if the hook is
    installed twice, training has started, the batch is missing, unreadable, differs from the
    record or does not fit the actor, the output files exist already or cannot be written (any
    ``OSError``; the files already created are removed), or the metrics cannot be computed.
    """
    if getattr(algo, RECORDER_ATTRIBUTE, None) is not None:
        raise RunRefused("the plasticity hook is installed already on this run (install it once)")
    cfgs, logger = algo._cfgs, algo._logger
    task = str(cfgs.env_id)
    steps_per_epoch = int(cfgs.algo_cfgs.steps_per_epoch)
    if steps_per_epoch <= 0:
        raise RunRefused(f"algo_cfgs.steps_per_epoch must be a positive integer, got {steps_per_epoch}")
    if int(logger.current_epoch) != 0:
        raise RunRefused(
            f"the plasticity hook must be installed before the first epoch (logger epoch {logger.current_epoch})"
        )
    actor = algo._actor_critic.actor
    obs_dim = getattr(actor, "_obs_dim", None)
    if obs_dim is None:
        obs_dim = algo._env.observation_space.shape[0]
    obs_dim = int(obs_dim)
    if batch is None:
        raw = load_fixed_batch(task)  # FixedBatchError is a RunRefused
        batch_file, digest = f"{REPO_RELATIVE_DIR}/{task}.npy", FIXED_BATCH_SHA256[task]
        _checked_batch(raw, obs_dim, f"the fixed batch of {task}")
    else:
        checked = _checked_batch(batch, obs_dim, "the supplied batch")  # numeric first, so the cast cannot fail
        with np.errstate(over="ignore"):  # a finite value beyond float32's range becomes inf: refused just below
            raw = _checked_batch(checked.astype(np.float32), obs_dim, "the supplied batch (as float32)")
        batch_file, digest = None, sha256_hex(npy_bytes(raw))
    x = as_batch_tensor(raw)
    log_dir = Path(logger.log_dir)
    for name in (PLASTICITY_FILE, PLASTICITY_META_FILE):
        if (log_dir / name).exists():
            raise RunRefused(f"{log_dir / name} exists already; the hook writes a run's rows once")
    pilot_cfgs = _pilot_cfgs(cfgs)
    normalised = bool(cfgs.algo_cfgs.obs_normalize)
    run_id, onset_step = (pilot_cfgs.get("run_id"), pilot_cfgs.get("onset_step")) if pilot_cfgs else (None, None)
    meta = {
        "metrics_version": METRICS_VERSION,
        "owner": "Metrics and interventions (Role 3)",
        "registered": "Table 2.3; equations (6) and (7); Part 1.2 H3 (c); Part 3.4",
        # The Table 9.1 keys whose answers apply (both answered); the field keeps its name (as intervention.json's).
        "proposal_keys": ["Q-plasticity-definitions", "Q-reset-injection"],
        "task": task,
        "run_id": None if run_id is None else str(run_id),
        "onset_step": None if onset_step is None else int(onset_step),
        "steps_per_epoch": steps_per_epoch,
        "batch_file": batch_file,
        "batch_source": "committed fixed batch" if batch_file else "supplied by the caller (tests only)",
        "batch_sha256": digest,
        "n_states": int(x.shape[0]),
        "obs_dim": obs_dim,
        "tau": R.DORMANT_THRESHOLD,
        "delta": R.EFFECTIVE_RANK_DELTA,
        "input": (
            "the fixed batch normalised by the run's observation normaliser at the measured checkpoint, "
            "frozen (identity while its count <= 1)" if normalised
            else "the raw fixed batch (the configuration does not normalise observations)"
        ),
        "columns": list(COLUMNS),
        "rows": "one per checkpoint saved through logger.torch_save (step = logger epoch x steps_per_epoch); "
                "step 0 at install; never a step twice",
    }
    recorder = PlasticityRecorder(algo, x, log_dir, steps_per_epoch, meta)
    try:
        first = measure(algo, x, step=0)
    except Exception as exc:  # noqa: BLE001 - before training: refuse, do not exclude
        raise RunRefused(
            f"the plasticity metrics cannot be computed for this run: {type(exc).__name__}: {exc}"
        ) from exc
    try:
        recorder.start(first)
    except FileExistsError as exc:
        raise RunRefused(
            f"{exc.filename or recorder.path} appeared while the hook was installed; the hook writes a run's rows once"
        ) from exc
    except OSError as exc:  # before training: a run that cannot keep its records is refused, not crashed
        raise RunRefused(f"the plasticity records cannot be written in {log_dir}: {type(exc).__name__}: {exc}") from exc
    original_torch_save = logger.torch_save  # the bound method, or a wrapper installed before (chained)

    def torch_save(*args: Any, **kwargs: Any) -> Any:
        result = original_torch_save(*args, **kwargs)
        recorder.record(int(logger.current_epoch) * recorder.steps_per_epoch)
        return result

    logger.torch_save = torch_save
    setattr(algo, RECORDER_ATTRIBUTE, recorder)
    return recorder

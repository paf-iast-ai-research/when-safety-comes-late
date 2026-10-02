"""Fixed evaluation batches of the plasticity metrics (Role 3, Metrics and interventions; owner Abdullah).

Implements Table 2.3 "Fixed evaluation batch": "2,048 states (decision) collected once per task by
a uniform-random policy under seed 0, stored in the repository, and used for every metric
computation of every run of that task" (also Table 8.2 "Fixed evaluation batch | 2,048 states";
Appendix A lists "the fixed evaluation batch for each task" among the items "copied verbatim from the
repository at the registration commit"; Q-appendix-a: they are recorded as added later).

The collection procedure follows First Tasks section 9 step 1 (guidance, not registered): "reset
the environment with seed 0, act with a uniform random policy within the action bounds, resetting
at the end of each 1,000-step episode, and store the first 2,048 observations". The stored vectors
are the environment's observations (the actor's input), not the MuJoCo simulator state; that
reading, the float32 storage and the later addition of the files are answered in Q-appendix-a
(docs/DECISIONS.md; HANDOVER.md section 9; not a PENDING key): the files are recorded in Table 9.1 as
added after the registration commit, with their SHA-256, and the committed batches are the fixed
batches.

Why the hashes live in this Python file: ``pilot.provenance`` refuses a run that imports an
uncommitted or modified repository module, but it does not look at data files. Checking every
``.npy`` against a committed constant here closes that gap, so a run can only measure on the
committed batch (``load_fixed_batch`` raises ``FixedBatchError``, a ``RunRefused``, otherwise).
"""

from __future__ import annotations

import hashlib
import io
import os
import tempfile
from pathlib import Path
from types import MappingProxyType

import numpy as np

from configs import registered as R
from pilot.errors import RunRefused

FIXED_BATCH_DIR = Path(__file__).resolve().parent / "fixed_batch"
MANIFEST_FILE = "MANIFEST.json"
REPO_RELATIVE_DIR = "metrics/fixed_batch"  # how the batch files are named in records (plasticity_meta.json)

# SHA-256 of each committed ``.npy`` file (the whole file: header and data), written by
# ``python -m metrics.collect_fixed_batch --all`` and checked by ``--check``. Must equal MANIFEST.json.
FIXED_BATCH_SHA256 = MappingProxyType({
    "SafetyCarGoal1-v0": "abc01be62d4937d456e95372281252f99cf38da64d8931520ee34edcea7ec618",
    "SafetyPointButton1-v0": "84ae724901954a6f8bfd96d55eebf7e6b7651536398e357175115e2570e98f0a",
    "SafetyPointGoal1-v0": "5340625cdd26c93fa4e610f7d5c93ae2a1f4d050ba64c1786a7a15928a182558",
})

# The procedure, as recorded in MANIFEST.json: First Tasks section 9 step 1 (module docstring), made exact.
PROCEDURE = (
    "1. env = safety_gymnasium.make(task) (the Safety-Gymnasium environment that OmniSafe wraps; no "
    "OmniSafe wrapper, because the batch holds raw observations and OmniSafe's action scaling is the "
    "identity for actions in [-1, 1]). "
    "2. rng = numpy.random.default_rng(seed) (PCG64) draws the actions; obs, _ = env.reset(seed=seed). "
    "3. Until n_states observations are stored: store obs (the observation the policy acts on); draw "
    "a = rng.uniform(env.action_space.low, env.action_space.high) (float64, one call per step); "
    "obs, reward, cost, terminated, truncated, info = env.step(a); if terminated or truncated, "
    "obs, _ = env.reset() (unseeded, so the layout stream continues from the seeded reset). The final "
    "observation of an episode is never stored. "
    "4. numpy.save(path, numpy.asarray(batch, dtype=numpy.float32), allow_pickle=False) (C order, "
    "little-endian float32, shape (n_states, obs_dim)): OmniSafe presents float32 observations to "
    "the normaliser and the actor. "
    "5. Row order is collection order; episode_starts lists the rows where an episode begins."
)


class FixedBatchError(RunRefused):
    """The fixed batch of a task is missing, unrecorded or differs from the committed record.

    A ``RunRefused``: a run that cannot measure on the committed batch must not start (the launcher
    exits 5 and the run stays pending), and it is never classified as a crash (Part 5.6).
    """


def batch_path(task: str) -> Path:
    """``metrics/fixed_batch/<task>.npy`` for a Study A task (Table 2.3 covers Study A only)."""
    if task not in R.TASKS_STUDY_A:
        raise FixedBatchError(f"no fixed batch for {task!r}: the batches are for the Study A tasks {R.TASKS_STUDY_A}")
    return FIXED_BATCH_DIR / f"{task}.npy"


def npy_bytes(batch: np.ndarray) -> bytes:
    """The exact bytes ``numpy.save(..., allow_pickle=False)`` writes for ``batch``."""
    buffer = io.BytesIO()
    np.save(buffer, batch, allow_pickle=False)
    return buffer.getvalue()


def sha256_hex(data: bytes) -> str:
    """Lower-case hex SHA-256 of ``data``."""
    return hashlib.sha256(data).hexdigest()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically: a temporary file beside it, fsync, mode 0644, ``os.replace``.

    The temporary file is removed if anything fails. ``mkstemp`` creates files with mode 0600; the
    files of this package (batches, MANIFEST.json, plasticity records, recomputed rows) are ordinary
    files that others may read.
    """
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def check_batch_array(batch: np.ndarray, what: str) -> None:
    """Shape (FIXED_BATCH_STATES, obs_dim), float32, every value finite; FixedBatchError otherwise."""
    if (not isinstance(batch, np.ndarray) or batch.ndim != 2 or batch.shape[0] != R.FIXED_BATCH_STATES
            or batch.shape[1] < 1):
        shape = getattr(batch, "shape", None)
        raise FixedBatchError(f"{what}: shape {shape}, expected ({R.FIXED_BATCH_STATES}, obs_dim) (Table 2.3)")
    if batch.dtype != np.float32:
        raise FixedBatchError(f"{what}: dtype {batch.dtype}, expected float32")
    if not np.isfinite(batch).all():
        raise FixedBatchError(f"{what}: contains non-finite values")


def collect_fixed_batch(
    task: str, n_states: int = R.FIXED_BATCH_STATES, seed: int = R.FIXED_BATCH_SEED
) -> tuple[np.ndarray, list[int]]:
    """Collect a batch by the recorded procedure (``PROCEDURE``); returns (float32 array, episode_starts).

    Deterministic: the same task, ``n_states`` and seed give the same bytes on the same machine and
    pinned versions (the ``--check`` mode of ``python -m metrics.collect_fixed_batch`` verifies it).
    The procedure is prefix-consistent: the first k rows of a batch equal a collection of k states.
    """
    if isinstance(n_states, bool) or not isinstance(n_states, int) or n_states < 1:
        raise ValueError(f"n_states must be a positive integer, got {n_states!r}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")
    batch_path(task)  # refuses a task that is not a Study A task
    import safety_gymnasium  # lazily: the loader and the metrics need only numpy

    env = safety_gymnasium.make(task)
    try:
        rng = np.random.default_rng(seed)
        low, high = env.action_space.low, env.action_space.high
        obs, _ = env.reset(seed=seed)
        rows: list[np.ndarray] = []
        episode_starts = [0]
        while True:
            rows.append(np.array(obs, dtype=np.float64, copy=True))  # copy: never alias an env buffer
            if len(rows) == n_states:
                break
            action = rng.uniform(low, high)
            obs, _reward, _cost, terminated, truncated, _info = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
                episode_starts.append(len(rows))
    finally:
        env.close()
    batch = np.asarray(rows, dtype=np.float32)
    if not np.isfinite(batch).all():
        raise FixedBatchError(f"the collected batch of {task} contains non-finite observations")
    return batch, episode_starts


def load_fixed_batch(task: str) -> np.ndarray:
    """The committed fixed batch of ``task``, verified; a read-only float32 array (2,048, obs_dim).

    Raises ``FixedBatchError`` if no hash is recorded for the task, the file is missing or cannot be
    read, its SHA-256 differs from ``FIXED_BATCH_SHA256[task]``, or its shape, dtype or values are
    wrong. The file is read once and parsed from the bytes that were hashed, so what is checked is
    what is used.
    """
    if task not in FIXED_BATCH_SHA256:
        raise FixedBatchError(f"no fixed batch recorded for {task!r} (Role 3; metrics/batches.py FIXED_BATCH_SHA256)")
    path = batch_path(task)
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        raise FixedBatchError(f"the fixed batch of {task} is missing: {path}") from None
    except OSError as exc:
        # Any other read failure (a directory in its place, no permission, an I/O error) is a batch
        # problem too: the run must be refused (exit 5), not started and then classified as a crash.
        raise FixedBatchError(f"the fixed batch of {task} cannot be read: {path}: {type(exc).__name__}: {exc}") from exc
    digest = sha256_hex(data)
    if digest != FIXED_BATCH_SHA256[task]:
        raise FixedBatchError(
            f"the fixed batch of {task} differs from the committed record: sha256 {digest}, "
            f"recorded {FIXED_BATCH_SHA256[task]} (Table 2.3: collected once; First Tasks section 9 step 1: "
            "never resampled)"
        )
    try:
        batch = np.load(io.BytesIO(data), allow_pickle=False)
    except ValueError as exc:
        raise FixedBatchError(f"the fixed batch of {task} is not a plain .npy array: {exc}") from exc
    check_batch_array(batch, f"fixed batch of {task}")
    batch.setflags(write=False)
    return batch

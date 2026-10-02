"""Launch one run: the training stage and the evaluation stage of a single ``RunSpec``.

Usage (normally called by the scheduler, one process per run):

    python -m pilot.launch train    --spec RUN_DIR/spec.json --run-dir RUN_DIR
    python -m pilot.launch evaluate --spec RUN_DIR/spec.json --run-dir RUN_DIR

Training reproduces what ``omnisafe.Agent`` (``AlgoWrapper``) does, step for step, but for any
algorithm class, because ``omnisafe.Agent`` accepts only built-in algorithm names and loads their
YAML from the installed package (so a subclass such as the Study A plug-in, envs.onset, cannot go through it):

  default YAML of the base algorithm -> validated registered overrides -> exp_name, env_id, algo
  -> epochs = total_steps // steps_per_epoch -> check_all_configs -> torch.set_num_threads(1)
  -> plug-in factory -> learn()

Registered settings applied here (Table 3.1; Part 3.4; Table 8.2): CPU, one torch thread, one
vectorised environment, one process, total steps of the arm, 20,000 steps per epoch (Table A.1),
and a checkpoint every ten epochs (200,000 steps; Table 2.3 "Logging points"). Every other
hyperparameter stays OmniSafe's default for the pinned version: ``verify_installed_yaml`` refuses
to run unless the installed default YAML is byte-for-byte the copy committed in
``configs/omnisafe`` (Appendix A, Table A.1), and ``verify_registered_defaults`` checks the values
the pre-registration names.

Refusals (open questions, existing output, a claimed directory, uncommitted code, a config that
differs from Table A.1, a dependency that is not a completed run, a library the plug-in's factory
cannot import, or a plug-in or hook that refuses) exit with EXIT_REFUSED (HANDOVER.md section 8), and a
plug-in (or a repository module that it imports, at its import, in its factory or during training)
that is not in the repository yet exits with EXIT_UNAVAILABLE. They are not run outcomes and never
become exclusions (Part 5.6): the run directory is left as it was found, so the run stays pending.
Most happen before the claim; one raised after it removes the claim and OmniSafe's output, including
epochs already trained when a plug-in or hook refuses during training (the train log records how
many and the traceback), unless training had already gone non-finite, which Part 5.6 excludes like
any other non-finite run. The evaluation stage exits the same way: a refusal leaves the run trained,
and a harness (or a repository module it imports) that is not written yet is EXIT_UNAVAILABLE. Any
other exception (an evaluation that breaks contract 1, a library missing from the environment)
propagates, so Python exits 1, which the scheduler records as eval_failed or launch_failed.

``--allow-pending`` (smoke tests only, on a smoke data root) lets both stages run with open
questions: it skips the spec's run gates and opens the result gates of ``pilot.errors`` for this
process (HANDOVER.md section 10). ``--allow-dirty`` (smoke tests only) allows uncommitted code; it also
leaves out the run gates the current code derives for a registered run, so only the spec's stored
pending keys are checked (``manifest.open_run_gates``).

Outputs in RUN_DIR: ``spec.json`` (written by the scheduler), ``.claim`` (the training claim),
``omnisafe/`` (OmniSafe's log directory), ``train_result.json`` and, after evaluation,
``evaluation.json`` (layout: ``pilot/rundir.py``, whose file names and progress readers this module
re-exports).

Owner: pilot owner (Role 1).
"""

from __future__ import annotations

import os

# One thread for every numerical library, set before torch or numpy is imported (Table 3.1).
# Forced, not defaulted: an inherited OMP_NUM_THREADS=8 must not reach a registered run.
THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
for _var in THREAD_VARIABLES:
    os.environ[_var] = "1"

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402
import signal  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

from configs import registered as R  # noqa: E402
from pilot import algorithms, contracts, dependencies, errors, manifest, provenance, rundir  # noqa: E402
from pilot.errors import RunRefused  # noqa: E402  # one class in every process (pilot/errors.py)
from pilot.manifest import CONTINUATION_GROUPS, RunSpec  # noqa: E402

# The run-directory layout lives in pilot/rundir.py; re-exported here, where the
# scheduler, the ledger writer, scripts/determinism_check.py and the tests import it from.
OMNISAFE_SUBDIR = rundir.OMNISAFE_SUBDIR
TRAIN_RESULT = rundir.TRAIN_RESULT
EVALUATION_RESULT = rundir.EVALUATION_RESULT
SPEC_FILE = rundir.SPEC_FILE
CLAIM_FILE = rundir.CLAIM_FILE
MULTIPLIER_COLUMN = rundir.MULTIPLIER_COLUMN
STEPS_COLUMN = rundir.STEPS_COLUMN
find_omnisafe_run_dir = rundir.find_omnisafe_run_dir
read_progress = rundir.read_progress

# progress.csv columns of OmniSafe's losses (Loss/Loss_pi, ...), checked for non-finite values (Part 5.6).
LOSS_PREFIX = "Loss/"

# Exit codes of both stages. Any other exception propagates, and Python exits 1 (module docstring).
EXIT_COMPLETED = 0
# Also argparse's exit status for a usage error; callers tell the two apart by train_result.json
# (written for every failed run, never for a usage error) and the claim, never by the code alone.
EXIT_FAILED = 2
EXIT_INTERRUPTED = 3
# Another role's code (a plug-in, a harness, or a module or name either imports) is not in the repository yet.
EXIT_UNAVAILABLE = 4
# A refusal in either stage (provenance, configuration, run directory, dependency, plug-in, hook or
# harness): not a run outcome. A refusal during training discards the attempt; the run stays pending.
EXIT_REFUSED = 5

# sha256 of the committed OmniSafe YAML copies (Table A.1).
CONFIG_SOURCE = provenance.REPO_ROOT / "configs" / "omnisafe" / "SOURCE.json"

# One rule for the plug-in loader, the harness loader and both stages (pilot/algorithms.py): a
# ModuleNotFoundError naming another role's package (or a submodule), or an ImportError of a name
# such a module does not define yet, means that code is not written yet, so the stage is
# unavailable (exit 4), not failed.
PluginUnavailableError = algorithms.PluginUnavailableError
REPOSITORY_PACKAGES = algorithms.REPOSITORY_PACKAGES
is_repository_module = algorithms.is_repository_module
is_missing_repository_import = algorithms.is_missing_repository_import


# The signals that stop a run from outside it (a shutdown, Ctrl-C, a closed terminal or ssh session): ``train``
# records them as an interruption (status "interrupted"), which the scheduler restarts without limit as one of
# machine origin (Table 9.1, Q-interrupted-run; pilot/scheduler.py).
INTERRUPT_SIGNALS = tuple(getattr(signal, name) for name in ("SIGTERM", "SIGINT", "SIGHUP") if hasattr(signal, name))


class RunInterrupted(BaseException):
    """The process received SIGTERM, SIGINT or SIGHUP: the machine, not the run, stopped it.

    A BaseException, as KeyboardInterrupt is, so that the ``except Exception`` of a plug-in, a hook
    (Role 3's measurement at a checkpoint) or ``pilot.dependencies.load_full_state`` lets it through:
    ``_SignalGate`` raises only once, and a signal swallowed there would be lost.
    """


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def registered_overrides(spec: RunSpec, log_dir: Path) -> dict[str, Any]:
    """The custom configuration of a run: the settings the pre-registration fixes, plus OmniSafe's log directory."""
    return {
        "seed": int(spec.seed),
        "train_cfgs": {
            "device": R.DEVICE,
            "torch_threads": R.TORCH_THREADS,
            "vector_env_nums": R.VECTOR_ENV_NUMS,
            "parallel": R.PARALLEL_PROCESSES,
            "total_steps": int(spec.total_steps),
        },
        "algo_cfgs": {"steps_per_epoch": R.STEPS_PER_EPOCH},
        "logger_cfgs": {
            "save_model_freq": R.CHECKPOINT_INTERVAL_EPOCHS,
            "log_dir": str(log_dir),
        },
    }


def verify_registered_defaults(cfgs: Any, base_algo: str) -> None:
    """Refuse to run if the installed OmniSafe defaults differ from Appendix A, Table A.1."""
    expected = {
        ("algo_cfgs", "steps_per_epoch"): R.STEPS_PER_EPOCH,
        ("algo_cfgs", "update_iters"): R.UPDATE_ITERS,
        ("algo_cfgs", "batch_size"): R.MINIBATCH_SIZE,
        ("algo_cfgs", "target_kl"): R.TARGET_KL,
        ("model_cfgs", "actor", "hidden_sizes"): list(R.HIDDEN_SIZES),
        ("model_cfgs", "actor", "activation"): R.ACTIVATION,
        ("model_cfgs", "critic", "hidden_sizes"): list(R.HIDDEN_SIZES),
        ("model_cfgs", "critic", "activation"): R.ACTIVATION,
    }
    if base_algo == "PPOLag":
        expected.update({
            ("lagrange_cfgs", "cost_limit"): R.COST_LIMIT,
            ("lagrange_cfgs", "lagrangian_multiplier_init"): R.LAGRANGE_MULTIPLIER_INIT,
            ("lagrange_cfgs", "lambda_lr"): R.LAGRANGE_MULTIPLIER_LR,
        })
    elif base_algo == "CPPOPID":
        expected.update({
            ("lagrange_cfgs", "cost_limit"): R.COST_LIMIT,
            ("lagrange_cfgs", "lagrangian_multiplier_init"): R.LAGRANGE_MULTIPLIER_INIT,
        })
    problems = []
    for path, want in expected.items():
        node: Any = cfgs
        for key in path:
            node = node[key]
        got = list(node) if isinstance(node, (list, tuple)) else node
        if got != want:
            problems.append(f"{'.'.join(path)} = {got!r}, registered {want!r}")
    if problems:
        raise RunRefused("installed OmniSafe defaults differ from Table A.1: " + "; ".join(problems))


def installed_yaml(base_algo: str, algo_type: str = "on-policy") -> Path:
    """The default YAML that ``get_default_kwargs_yaml`` reads from the installed OmniSafe."""
    import omnisafe

    return Path(omnisafe.__file__).parent / "configs" / algo_type / f"{base_algo}.yaml"


def verify_installed_yaml(base_algo: str, algo_type: str = "on-policy") -> str:
    """Refuse unless the installed default YAML is the committed copy (Table 3.1; Table A.1).

    Returns the sha256 digest, which the run records.
    """
    recorded = json.loads(CONFIG_SOURCE.read_text(encoding="utf-8"))["files"]
    name = f"{base_algo}.yaml"
    if name not in recorded:
        raise RunRefused(f"{name} is not among the committed OmniSafe configs {sorted(recorded)}")
    digest = hashlib.sha256(installed_yaml(base_algo, algo_type).read_bytes()).hexdigest()
    if digest != recorded[name]:
        raise RunRefused(
            f"the installed {name} (sha256 {digest}) is not the copy committed in configs/omnisafe "
            f"(sha256 {recorded[name]}); reinstall the locked environment"
        )
    return digest


def pilot_config(spec: RunSpec, deps: dict[str, Any] | None = None) -> dict[str, Any]:
    """``cfgs.pilot_cfgs``: the run facts the mixin, the Role 3 hook and the plug-ins read.

    Added after OmniSafe's validation, as OmniSafe allows; saved in config.json and included in the
    config hash (the dependencies' two paths excepted, ``provenance.VOLATILE_DEPENDENCY_KEYS``).
    ``deps`` are the resolved dependencies (``pilot.dependencies.resolve``); they must be exactly
    the spec's ``depends_on``. A spec whose extra checkpoints cannot be computed is refused.

    A continuation's ``treatment`` and ``controller_variant`` are None here: its spec copies them
    from the parent for bookkeeping only (``pilot.manifest.battery_continuation``, used by the cut
    order), but the run applies neither. Table 2.2 describes one fine-tuning condition ("Training
    continues for 1,000,000 steps ... with the cost term removed from the objective and the
    multiplier frozen at zero") and one transfer condition for the battery, without reference to
    the parent's treatment or controller; that every continuation, a PID-arm parent's included,
    trains with PPO-Lagrangian is answered in Table 9.1 (Q-continuations; ``manifest.CONTINUATION_BASE_ALGO``),
    not by Table 2.2. So a plug-in or hook reading these run facts must not see a treatment or a
    controller variant the continuation does not apply.
    """
    try:
        extra_steps = contracts.extra_checkpoint_steps(spec)
    except ValueError as exc:
        raise RunRefused(f"{spec.run_id}: {exc}") from exc
    continuation = spec.group in CONTINUATION_GROUPS
    return {
        "run_id": spec.run_id,
        "onset_step": int(spec.onset_step or 0),
        "study": spec.study,
        "plugin": spec.plugin,
        "group": spec.group,
        "treatment": None if continuation else spec.treatment,
        "controller_variant": None if continuation else spec.controller_variant,
        "extra_checkpoint_steps": extra_steps,  # Part 3.4 "at onset"; box H3 (c); Table 2.5 horizons
        "plasticity": contracts.plasticity_required(spec),  # Table 2.3
        "dependencies": dependencies.to_config(deps or {}, spec),
    }


def build_config(spec: RunSpec, log_dir: Path, deps: dict[str, Any] | None = None) -> Any:
    """Build and check the OmniSafe Config as AlgoWrapper._init_config and _init_algo's check_all_configs do.

    The installed YAML and the Table A.1 defaults are verified first, and ``pilot_cfgs`` is added.
    """
    return build_config_and_digest(spec, log_dir, deps)[0]


def build_config_and_digest(spec: RunSpec, log_dir: Path, deps: dict[str, Any] | None = None) -> tuple[Any, str]:
    """``build_config`` and the sha256 of the installed default YAML it verified (recorded by the run)."""
    from omnisafe.algorithms import ALGORITHM2TYPE
    from omnisafe.envs import support_envs
    from omnisafe.utils.config import Config, check_all_configs, get_default_kwargs_yaml
    from omnisafe.utils.tools import recursive_check_config

    algo_type = ALGORITHM2TYPE.get(spec.base_algo)
    if algo_type != "on-policy":
        raise RunRefused(f"{spec.base_algo} is not an on-policy algorithm of the installed OmniSafe")
    if spec.task not in support_envs():
        raise RunRefused(f"{spec.task} is not supported by the installed OmniSafe")
    yaml_digest = verify_installed_yaml(spec.base_algo, algo_type)
    cfgs = get_default_kwargs_yaml(spec.base_algo, spec.task, algo_type)
    verify_registered_defaults(cfgs, spec.base_algo)
    custom = registered_overrides(spec, log_dir)
    recursive_check_config(custom, cfgs)
    cfgs.recurisve_update(custom)  # (sic) OmniSafe's spelling
    cfgs.update({"exp_increment_cfgs": custom})
    cfgs.recurisve_update(
        {"exp_name": f"{spec.base_algo}-{{{spec.task}}}", "env_id": spec.task, "algo": spec.base_algo}
    )
    cfgs.train_cfgs.recurisve_update(
        {"epochs": cfgs.train_cfgs.total_steps // cfgs.algo_cfgs.steps_per_epoch}
    )
    check_all_configs(cfgs, algo_type)
    # pilot_cfgs: see pilot_config.
    cfgs["pilot_cfgs"] = Config.dict2config(pilot_config(spec, deps))
    if cfgs.train_cfgs.parallel != 1 or cfgs.train_cfgs.device != "cpu":
        raise RunRefused("registered runs use one process on the CPU (Table 3.1)")
    if cfgs.train_cfgs.epochs * cfgs.algo_cfgs.steps_per_epoch != spec.total_steps:
        raise RunRefused("total_steps is not a whole number of epochs")
    return cfgs, yaml_digest


# ---------------------------------------------------------------------------
# Output inspection
# ---------------------------------------------------------------------------


def _nonfinite(rows: list[dict[str, str]], prefix: str) -> list[str]:
    """Non-finite values in every column whose name starts with ``prefix``."""
    bad = []
    for i, row in enumerate(rows):
        for col in row:
            if col is None or not col.startswith(prefix):
                continue
            if row[col] not in ("", None):
                try:
                    value = float(row[col])
                except ValueError:
                    continue  # not a number at all (a row cut short by a full disk): not a non-finite value
                if not math.isfinite(value):
                    bad.append(f"epoch {i} {col}={row[col]}")
    return bad


def _read_log(run_dir: Path) -> tuple[Path | None, list[dict[str, str]]]:
    """OmniSafe's run directory under ``run_dir`` and its progress.csv rows; (None, []) without a log.

    A log that cannot be read (NUL bytes, bad UTF-8) gives no rows, never an error.
    """
    omnisafe_dir: Path | None = None
    try:
        omnisafe_dir = find_omnisafe_run_dir(run_dir)
        return omnisafe_dir, read_progress(omnisafe_dir)
    except Exception:  # noqa: BLE001 - no log, or one that cannot be read: no rows
        return omnisafe_dir, []


def _log_nonfinite(rows: list[dict[str, str]]) -> tuple[str, list[str]] | None:
    """The Part 5.6 cause a log shows and its non-finite values, or None; a multiplier before a loss."""
    for cause, prefix in (("non_finite_multiplier", MULTIPLIER_COLUMN), ("non_finite_loss", LOSS_PREFIX)):
        bad = _nonfinite(rows, prefix)
        if bad:
            return cause, bad
    return None


def nonfinite_state(algo: Any) -> str | None:
    """Cause of a non-finite state in a live algorithm object, or None (Part 5.6).

    A NaN or inf loss makes the next optimiser step write non-finite parameters, and the next
    forward pass then raises inside torch.distributions before the epoch's row is logged, so the
    state of the model is checked, not only the log. Every multiplier counts: OmniSafe's single
    ``_lagrange`` and Study B's per-level ``_level_lagranges`` (studyb/conditioning.py).
    """
    import torch

    multipliers = [getattr(algo, "_lagrange", None), *dict(getattr(algo, "_level_lagranges", None) or {}).values()]
    for lagrange in multipliers:
        if lagrange is None:
            continue
        value = getattr(lagrange, "lagrangian_multiplier", lagrange)
        value = float(value.detach()) if isinstance(value, torch.Tensor) else float(value)
        if not math.isfinite(value):
            return "non_finite_multiplier"
    actor_critic = getattr(algo, "_actor_critic", None)
    if actor_critic is not None:
        for param in actor_critic.parameters():
            if not bool(torch.isfinite(param).all()):
                return "non_finite_loss"
    return None


def classify_training(
    run_dir: Path, spec: RunSpec, exception: BaseException | None, state_cause: str | None = None
) -> tuple[str, str | None, str]:
    """Return (status, failure_cause, detail) by the exclusion rule of Part 5.6.

    status is 'completed' or 'failed'; failure_cause is one of the ledger's causes
    (crash, incomplete, non_finite_loss, non_finite_multiplier) or None. ``state_cause`` is
    ``nonfinite_state`` of the algorithm when training raised. An exception whose model and
    multiplier are finite is a crash, whatever its message: a NaN that reaches torch.distributions
    from an observation or the normaliser is not a non-finite loss. A run is complete only when
    every epoch is logged and the final checkpoint loads. A log that cannot be read counts as nothing
    logged; a last TotalEnvSteps that is not a number counts as zero steps (the run is incomplete).
    Neither is an error of this function.
    """
    omnisafe_dir, rows = _read_log(run_dir)
    logged = _log_nonfinite(rows)
    if logged is not None:
        cause, bad = logged
        return "failed", cause, "; ".join(bad[:5])
    trace = ""
    if exception is not None:
        trace = "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))[-4000:]
        if state_cause is not None:
            return "failed", state_cause, trace
    expected_epochs = spec.total_steps // R.STEPS_PER_EPOCH
    try:
        last_steps = int(float(rows[-1][STEPS_COLUMN])) if rows and rows[-1].get(STEPS_COLUMN) else 0
    except (ValueError, OverflowError):  # a value cut short ("4e") or "inf": the last row is not complete
        last_steps = 0
    complete = len(rows) == expected_epochs and last_steps == spec.total_steps
    final_saved = omnisafe_dir is not None and final_checkpoint_loads(
        rundir.checkpoint_file(omnisafe_dir, expected_epochs))
    if exception is not None:
        if complete and final_saved:
            # Every step ran and the final state is saved; an error while closing the logger or the
            # environment does not undo that. "A run that completes is kept" (Part 5.6).
            return "completed", None, "error after the last epoch (run kept):\n" + trace
        return "failed", "crash", trace
    if not (complete and final_saved):
        return "failed", "incomplete", (
            f"{len(rows)} epochs logged, {last_steps} steps; expected {expected_epochs}, {spec.total_steps}; "
            f"final checkpoint {'loads' if final_saved else 'missing or unreadable'}"
        )
    return "completed", None, ""


def final_checkpoint_loads(path: Path) -> bool:
    """True if the checkpoint exists and loads (a disk-full error can leave a truncated file).

    OmniSafe logs the last epoch's progress row before it saves the last checkpoint, so a run is
    complete only when both are there.
    """
    if not path.exists():
        return False
    import torch

    try:
        # weights_only=False: the run's own file, loaded only to prove it is complete; a plug-in's
        # extra state (for example a numpy scalar) must not turn a complete run into an incomplete one.
        torch.load(path, map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001 - any failure to load means the state is not saved
        return False
    return True


def plain(value: Any) -> Any:
    """``value`` with numpy and torch scalars turned into Python numbers, recursively.

    A harness may return ``np.float32`` costs or ``np.int64`` step keys (or a torch tensor or
    ``torch.nn.Parameter``); JSON must store numbers.
    """
    if isinstance(value, dict):
        return {plain(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if type(value).__module__.split(".")[0] in ("numpy", "torch") and hasattr(value, "tolist"):
        return plain(value.tolist())
    return value


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    """Serialise first (a TypeError leaves nothing behind), then write, fsync, rename and fsync the directory.

    The temporary file is removed when the write or the rename fails (for example a full disk).
    """
    text = json.dumps(plain(data), indent=2, sort_keys=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)  # already gone after a successful rename
    provenance.fsync_directory(path.parent)


def _versions() -> dict[str, str]:
    """Versions of Python and the pinned packages, for train_result.json."""
    import importlib.metadata as md
    import platform

    out = {"python": platform.python_version()}
    for pkg in ("omnisafe", "safety-gymnasium", "torch", "numpy", "mujoco", "gymnasium"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = "not installed"
    return out


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


class _SignalGate:
    """The handler of ``train`` for INTERRUPT_SIGNALS (SIGTERM, SIGINT and SIGHUP).

    While armed, the first signal disarms the gate and raises RunInterrupted; every other signal
    (a second one arriving at the same time, or any signal after training ended) is only recorded,
    and ``train`` re-delivers it to the original handler once train_result.json is written. Recording
    rather than blocking: a thread mask would not help, because OmniSafe's TensorBoard writer thread
    receives the signal and CPython still runs the Python handler in the main thread.
    """

    def __init__(self) -> None:
        self.armed = True
        self.deferred: list[int] = []

    def __call__(self, signum: int, frame: Any) -> None:
        if self.armed:
            self.armed = False
            raise RunInterrupted(f"signal {signum}")
        self.deferred.append(signum)


def _refuse_unverified_code() -> None:
    """Refuse if any imported repository module is not in the commit (provenance.py)."""
    unverified = provenance.unverified_imported_code()
    if unverified:
        raise RunRefused("the run would execute code that its commit does not contain: " + ", ".join(unverified))


def _check_threads(reset: bool = True) -> int:
    """Set (``reset``) and check one torch thread (Table 3.1); refuse if the count differs."""
    import torch

    if reset:
        torch.set_num_threads(R.TORCH_THREADS)
    threads = torch.get_num_threads()
    if threads != R.TORCH_THREADS:
        raise RunRefused(f"torch uses {threads} threads; Table 3.1 requires {R.TORCH_THREADS}")
    return threads


def train(spec: RunSpec, run_dir: Path, *, allow_dirty: bool = False, allow_pending: bool = False) -> int:
    """Run the training stage. Returns EXIT_COMPLETED, EXIT_FAILED or EXIT_INTERRUPTED.

    Raises ``RunRefused`` or ``provenance.DirtyWorktreeError`` (``main``: EXIT_REFUSED), or
    ``PluginUnavailableError`` (``main``: EXIT_UNAVAILABLE), when the run must not start or must not
    go on. Raised after the claim (by the plug-in's factory, a check after it, or a plug-in or hook
    during training), they first remove the claim and OmniSafe's output, so the run directory is
    left as it was found and the run stays pending (``_train_claimed``). A refusal during training
    that comes after a non-finite loss or multiplier is not discarded: the run fails (EXIT_FAILED)
    with that cause (Part 5.6).

    The first SIGTERM, SIGINT or SIGHUP (INTERRUPT_SIGNALS) from the installation of the handlers (right
    after the claim) to the end of training ends the run as an interruption (EXIT_INTERRUPTED,
    train_result.json status "interrupted"). Any other signal (a second one, or one after training
    ended) is only recorded.
    On every exit from here on, the caller's original handlers are restored and the recorded signals
    are re-delivered to them, after train_result.json is written when it can be.

    A spec with ``depends_on`` has its dependencies resolved before the claim
    (``pilot.dependencies.resolve``: each a completed run with the expected spec, a continuation's
    parent checkpoint present and the one the spec names, and not a smoke run unless this one is
    too; RunRefused otherwise). They are passed to the plug-in in ``cfgs.pilot_cfgs.dependencies``
    and recorded in train_result.json (pilot/dependencies.py).
    ``allow_pending`` (smoke tests only) also opens the result gates of ``pilot.errors`` while the
    plug-in is built and trained.
    """
    from pilot.algorithms import load_plugin

    run_dir = Path(os.path.abspath(run_dir))
    # the stored keys and, for a registered run, the run gates the current code derives
    open_questions = manifest.open_run_gates(spec, registered=not (allow_dirty or allow_pending))
    if open_questions and not allow_pending:
        raise RunRefused(f"{spec.run_id} depends on open questions {open_questions}; see configs/registered.py PENDING")
    if (run_dir / OMNISAFE_SUBDIR).exists() or (run_dir / TRAIN_RESULT).exists():
        raise RunRefused(f"{run_dir} already holds training output; a run is never trained twice in one directory")
    # Every check that needs no output runs before the claim, so a refusal leaves nothing behind,
    # and so does everything that could fail between the claim and the guarded training below.
    commit = provenance.commit_hash() if allow_dirty else provenance.require_clean_worktree()
    smoke = bool(allow_dirty or allow_pending or errors.pending_allowed())  # recorded below: allow_dirty/allow_pending
    deps = dependencies.resolve(spec, run_dir, smoke=smoke) if spec.depends_on else {}
    factory = load_plugin(spec.plugin)
    if not allow_dirty:
        _refuse_unverified_code()
    cfgs, yaml_digest = build_config_and_digest(spec, run_dir / OMNISAFE_SUBDIR, deps)
    threads = _check_threads()
    result: dict[str, Any] = {
        "run_id": spec.run_id,
        "commit_hash": commit,
        "worktree_dirty": bool(provenance.dirty_paths()),
        "machine": provenance.machine_description(),
        "versions": _versions(),
        "omnisafe_yaml_sha256": yaml_digest,
        "config_hash": provenance.config_hash(cfgs.todict()),
        "torch_threads": threads,
        "thread_variables": {v: os.environ.get(v) for v in THREAD_VARIABLES},
        "allow_dirty": bool(allow_dirty),
        "allow_pending": bool(allow_pending or errors.pending_allowed()),  # what training runs with (below)
        "dependencies": cfgs.pilot_cfgs.dependencies.todict(),
    }
    # Claim the directory atomically: a second process started for the same run stops here.
    try:
        fd = os.open(run_dir / CLAIM_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        raise RunRefused(f"{run_dir} is already claimed by another process; a run is never trained twice") from None
    try:
        try:
            os.write(fd, f"{os.getpid()}\n".encode())
        finally:
            os.close(fd)
    except BaseException:
        (run_dir / CLAIM_FILE).unlink(missing_ok=True)  # nothing has started: a stranded claim would block the run
        raise

    result["started"] = provenance.utc_now().isoformat()
    gate = _SignalGate()
    old_handlers: dict[int, Any] = {}
    try:
        for sig in INTERRUPT_SIGNALS:
            old_handlers[sig] = signal.signal(sig, gate)
        with errors.allow_pending(allow_pending or errors.pending_allowed()):
            return _train_claimed(spec, run_dir, cfgs, factory, result, gate, allow_dirty=allow_dirty)
    except RunInterrupted as exc:
        # The gate fired before _train_claimed's guarded block began, so nothing has trained yet. Inside
        # that block every RunInterrupted is caught, and the gate raises only once.
        now = provenance.utc_now().isoformat()
        result.update(status="interrupted", failure_cause=None, detail=str(exc), interruption=str(exc),
                      finished=now, wall_clock_hours=0.0, omnisafe_dir=None)
        _write_json_atomic(run_dir / TRAIN_RESULT, result)
        return EXIT_INTERRUPTED
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        for signum in gate.deferred:  # a recorded signal now meets the original handler
            signal.raise_signal(signum)


def _train_claimed(
    spec: RunSpec, run_dir: Path, cfgs: Any, factory: Any, result: dict[str, Any], gate: _SignalGate,
    *, allow_dirty: bool,
) -> int:
    """``train`` after the claim, with ``gate`` installed: train, classify, write train_result.json.

    These are not run outcomes (HANDOVER.md section 8), whether raised by the factory, a check after it,
    or a plug-in or hook during ``learn()``: RunRefused (exit 5) and a missing repository module, or
    a name a repository module does not define yet (``is_missing_repository_import``;
    PluginUnavailableError, exit 4). A library module, or a name missing from a library, is a refusal
    when the factory imports it (nothing trained); an import error of a library during training is
    classified like every other training error.
    One exception: a refusal or missing module raised during ``learn()`` after the log or the live
    state has gone non-finite (``_nonfinite_evidence``) is classified too, so the run is excluded as
    Part 5.6 requires ("non_finite_loss" or "non_finite_multiplier") instead of being discarded and
    relaunched into the same NaN.
    """
    t0 = time.monotonic()
    exception: BaseException | None = None
    state_cause: str | None = None
    interrupted = False
    refused_after: str | None = None  # a refusal that came after training had gone non-finite
    algo: Any = None
    learning = False  # set just before learn(): a refusal after it discards trained epochs (if finite)
    try:
        try:
            os.environ["OMNISAFE_DEVICE"] = cfgs.train_cfgs.device
            try:
                algo = factory(spec.task, cfgs, spec)
            except ImportError as exc:
                if isinstance(exc, PluginUnavailableError):
                    raise  # already classified (handled below)
                raise _factory_import_error(spec, exc) from exc
            if not allow_dirty:
                _refuse_unverified_code()  # modules the factory imported lazily
            _check_threads(reset=False)  # a factory must not change the thread count
            # A plug-in may add its own settings (for example onset_cfgs) after validation; hash the result.
            result["config_hash"] = provenance.config_hash(cfgs.todict())
            learning = True
            algo.learn()
        finally:
            # From here on a signal is only recorded. One that arrives before this line raises
            # RunInterrupted here, which the handler below catches like any other interruption.
            gate.armed = False
    except RunInterrupted as exc:
        interrupted = True
        result["interruption"] = str(exc)
    except (RunRefused, ImportError) as exc:
        # An ImportError other than PluginUnavailableError or another role's missing code (a module,
        # or a name in one) is a library's, raised during training (the factory's are converted above):
        # it is classified below like any other error (Part 5.6).
        library = not isinstance(exc, (RunRefused, PluginUnavailableError)) and not is_missing_repository_import(exc)
        nonfinite = _nonfinite_evidence(run_dir, algo) if learning and not library else None
        if library:
            exception = exc
        elif nonfinite is not None:
            # Training had already gone non-finite when the refusal came (for example the injection's
            # bit-identity check at onset meets NaN weights, metrics.interventions.plasticity_injection:
            # torch.equal(nan, nan) is False). Part 5.6 excludes that run; discarding it instead would relaunch it, reproduce
            # the NaN with the same seed (Table 3.1), and never let the exclusion (and its
            # replacement seed) start. So it is classified like any other training error, and the
            # refusal is kept in the detail.
            exception = exc
            refused_after = f"{type(exc).__name__}: {exc}"
            print(f"{refused_after}\nraised during training after a non-finite state ({nonfinite}); the run is "
                  "classified (Part 5.6), not discarded", file=sys.stderr)
        else:
            # The run stays pending (HANDOVER.md section 8), so what this process created is removed: the
            # scheduler relaunches a pending run only into a directory holding nothing but spec.json
            # (Scheduler._start_training), and this launcher refuses a directory with output. The
            # discarded attempt is not a result; its relaunch with the same seed reproduces it
            # (Table 3.1 determinism), on the commit that fixes the refusal. A record of the attempt
            # goes to the train log (stderr) first.
            _discard_attempt(run_dir, exc, learning=learning)
            if is_missing_repository_import(exc):  # a repository module, or a name of one, imported during training
                raise PluginUnavailableError(
                    f"training of {spec.run_id} imported {exc.name!r} (or a name from it), which is not in the "
                    f"repository yet: {exc}"
                ) from exc
            raise
    except Exception as exc:  # noqa: BLE001 - every training error is classified, never swallowed
        exception = exc
    if exception is not None and algo is not None:
        try:
            state_cause = nonfinite_state(algo)
        except Exception:  # noqa: BLE001 - classification falls back to the log
            state_cause = None
    finished = provenance.utc_now()
    result["finished"] = finished.isoformat()
    result["wall_clock_hours"] = (time.monotonic() - t0) / 3600.0
    status, cause, detail = classify_training(run_dir, spec, exception, state_cause)
    # A failure the log already proves (Part 5.6) stands; an interruption only replaces "not finished".
    if interrupted and status != "completed" and cause not in ("non_finite_loss", "non_finite_multiplier"):
        result.update(status="interrupted", failure_cause=None, detail=result.get("interruption", ""))
        code = EXIT_INTERRUPTED
    else:
        if interrupted and status == "completed":  # after the last epoch and the final checkpoint: kept
            detail = f"signal after the last epoch (run kept): {result['interruption']}"
        elif interrupted:  # the log already shows a non-finite loss or multiplier: the failure stands
            detail = f"{detail} (then {result['interruption']})"
        elif refused_after is not None:  # the traceback is in the detail only when the log did not decide
            detail = f"{detail} (then {refused_after})"
        result.update(status=status, failure_cause=cause, detail=detail)
        code = EXIT_COMPLETED if status == "completed" else EXIT_FAILED
    try:
        result["omnisafe_dir"] = str(find_omnisafe_run_dir(run_dir).relative_to(run_dir))
    except FileNotFoundError:
        result["omnisafe_dir"] = None
    _write_json_atomic(run_dir / TRAIN_RESULT, result)
    return code


def _nonfinite_evidence(run_dir: Path, algo: Any) -> str | None:
    """The Part 5.6 cause that an attempt refused during training already shows, or None.

    The log first (a non-finite multiplier or loss in progress.csv, as ``classify_training`` reads
    it), then the live state (``nonfinite_state``): a NaN update writes non-finite parameters before
    the next epoch is logged. Best effort: an unreadable log or state is no evidence.
    """
    logged = _log_nonfinite(_read_log(run_dir)[1])
    if logged is not None:
        return logged[0]
    if algo is None:
        return None
    try:
        return nonfinite_state(algo)
    except Exception:  # noqa: BLE001 - a state that cannot be read is no evidence
        return None


def _factory_import_error(spec: RunSpec, exc: ImportError) -> Exception:
    """What an import error from the plug-in's factory means; nothing has trained yet.

    A repository module, or a name a repository module does not define (another role's code not
    written yet, ``is_missing_repository_import``), makes the plug-in unavailable (exit 4).
    Any other import error is a library's: a module missing from the environment
    (ModuleNotFoundError) or a name missing from an installed library (a version mismatch) is a
    refusal (exit 5) that names the library: a problem of the installation, never an exclusion of the
    run (Part 5.6), and not reported as another role's unfinished code either.
    """
    if is_missing_repository_import(exc):
        return PluginUnavailableError(f"the factory of plug-in {spec.plugin!r} could not import {exc.name!r}: {exc}")
    return RunRefused(
        f"the factory of plug-in {spec.plugin!r} could not import the library {exc.name!r} ({exc}); "
        "install the locked environment (environment/requirements.lock.txt; scripts/setup_env.sh)"
    )


def _discard_attempt(run_dir: Path, exc: BaseException, *, learning: bool) -> None:
    """Remove the claim and OmniSafe's output of an attempt that ends refused or unavailable.

    When training had started, the train log first records how many epochs are discarded and the
    traceback, because the refusal was raised deep inside a plug-in or hook (for example an
    intervention refused at onset, HANDOVER.md section 8) and the output that would show it is removed.
    """
    if learning:
        try:
            epochs = len(read_progress(find_omnisafe_run_dir(run_dir)))
        except Exception:  # noqa: BLE001 - best effort, for the log line only
            epochs = 0
        print(
            f"{type(exc).__name__} raised during training after {epochs} logged epoch(s); the attempt's output is "
            "removed and the run stays pending (not a run outcome, Part 5.6). Traceback:\n"
            + "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            file=sys.stderr,
        )
    shutil.rmtree(run_dir / OMNISAFE_SUBDIR, ignore_errors=True)
    (run_dir / CLAIM_FILE).unlink(missing_ok=True)


def evaluate(spec: RunSpec, run_dir: Path, *, allow_dirty: bool = False, allow_pending: bool = False) -> int:
    """Run the evaluation stage through the run's evaluation harness (contract 1).

    ``pilot.contracts.evaluator_target(spec)`` names the harness: Role 5's
    ``studyb.evaluation:evaluate_run`` for a budget-conditioned run, Role 2's
    ``envs.evaluation:evaluate_run`` otherwise. It returns a dict with final_cost, final_return,
    selection {step: [cost, return]} for the run's last ten checkpoints (Part 3.4; Part 4.1 rule 1),
    episodes and selection_seeds, checked by ``validate_evaluation``, and the per-episode
    episode_costs and episode_returns that the ``evaluation`` supplement record needs (results/supplement_schema.py).
    As in the training stage, a harness that is not written yet, or that imports (at its top level
    or lazily) another role's module, or a name of one, that is not written yet, is unavailable
    (PluginUnavailableError, exit 4), not an evaluation failure; a missing library shows its real
    error. A continuation is never evaluated here (its results are read into its parent's row by
    enrichment). ``allow_pending`` (smoke tests only) opens the result gates of ``pilot.errors`` for
    the stage, from the harness call to the write of evaluation.json (the contract checks call
    result gates too, for example ``contracts.selection_window``).

    evaluation.json records, beside ``eval_commit_hash``, this stage's smoke markers as
    train_result.json records training's: ``eval_allow_dirty``, ``eval_allow_pending`` (the result
    gates were open) and ``eval_worktree_dirty`` (``provenance.dirty_paths()`` was not empty), so a
    run trained cleanly but evaluated as a smoke run can be kept out of the repository's ledgers.
    """
    run_dir = Path(os.path.abspath(run_dir))
    if spec.group in CONTINUATION_GROUPS:
        raise RunRefused(f"{spec.run_id} is a continuation ({spec.group}); it is not evaluated by this stage")
    commit = provenance.commit_hash() if allow_dirty else provenance.require_clean_worktree()
    worktree_dirty = bool(provenance.dirty_paths())  # with the commit: the code this stage ran
    train_result = json.loads((run_dir / TRAIN_RESULT).read_text(encoding="utf-8"))
    if train_result.get("status") != "completed":
        raise RunRefused(f"{spec.run_id}: training did not complete; nothing to evaluate")
    _refuse_existing_evaluation(spec, run_dir)
    target = contracts.evaluator_target(spec)
    evaluator = contracts.load_evaluator(spec)
    if not allow_dirty:
        _refuse_unverified_code()  # the harness itself must be committed
    t0 = time.monotonic()
    omnisafe_dir = run_dir / train_result["omnisafe_dir"]
    pending_open = bool(allow_pending or errors.pending_allowed())
    # The result gates are open for the whole stage: the contract checks below call them too
    # (contracts.selection_window), not only the harness.
    with errors.allow_pending(pending_open):
        try:
            evaluation = plain(dict(evaluator(str(omnisafe_dir), spec.to_dict())))
        except ImportError as exc:
            if not is_missing_repository_import(exc):
                # already classified (PluginUnavailableError: exit 4), or a library missing from the
                # environment: re-raised so its real error shows
                raise
            # A module (or a name of one) the harness imports lazily is not written yet: like a missing
            # harness, not a failure.
            raise PluginUnavailableError(
                f"the evaluation harness {target} could not import {exc.name!r}: {exc}") from exc
        evaluation["eval_wall_clock_hours"] = (time.monotonic() - t0) / 3600.0
        evaluation["eval_commit_hash"] = commit
        # The smoke markers of this stage, as train_result.json records those of training: an evaluation
        # run with open result gates or uncommitted code is a smoke result even when training was not,
        # and the ledger writer's smoke guard must be able to see it (HANDOVER.md section 10).
        evaluation["eval_allow_dirty"] = bool(allow_dirty)
        evaluation["eval_allow_pending"] = pending_open
        evaluation["eval_worktree_dirty"] = worktree_dirty
        contracts.validate_evaluation(evaluation, spec, omnisafe_dir)
        # One key type, as JSON stores it: valid int and digit-string step keys mixed in a step-keyed map
        # would make the sorted write below raise TypeError after the whole harness ran.
        evaluation["selection"] = _step_keyed(evaluation["selection"])
        _check_evaluation_record(spec, evaluation)
        # The per-episode maps too, once the record check has accepted their keys as steps.
        for name in ("episode_costs", "episode_returns", "unstable_replacements"):
            block = evaluation.get(name) or {}
            if block.get("selection"):
                block["selection"] = _step_keyed(block["selection"])
        if not allow_dirty:
            # Modules the harness imported lazily, and those the record check imported
            # (results.supplement_schema): checked after every import of this stage, never before one.
            _refuse_unverified_code()
        # Written once: an evaluation that finished at the same time cannot be replaced.
        try:
            provenance.write_text_once(run_dir / EVALUATION_RESULT,
                                       json.dumps(plain(evaluation), indent=2, sort_keys=True))
        except FileExistsError:
            _refuse_existing_evaluation(spec, run_dir)
            raise
    return EXIT_COMPLETED


def _step_keyed(mapping: dict[Any, Any]) -> dict[str, Any]:
    """``mapping`` with every checkpoint-step key as the digit string JSON stores (``contracts._step_key``)."""
    return {str(contracts._step_key(k)): v for k, v in mapping.items()}


def _refuse_existing_evaluation(spec: RunSpec, run_dir: Path) -> None:
    """Refuse if evaluation.json exists: a run is never evaluated twice in one directory.

    As it is never trained twice: its evaluation.json may be the one its ledger row and ``evaluation``
    supplement record were computed from; ``schedule resolve RUN reevaluate`` moves it aside before a
    new evaluation.
    """
    if (run_dir / EVALUATION_RESULT).exists():
        raise RunRefused(f"{spec.run_id}: {run_dir / EVALUATION_RESULT} exists; a run is never evaluated twice in "
                         f"one directory (`schedule resolve {spec.run_id} reevaluate` moves it aside first)")


def _check_evaluation_record(spec: RunSpec, evaluation: dict[str, Any]) -> None:
    """The ledger writer's check of the ``evaluation`` supplement record (results/supplement_schema.py), at this stage.

    A result the ledger would refuse (no per-episode costs and returns, or a record that is not valid)
    fails here, as an evaluation failure (ContractError: exit 1, ``eval_failed``, which ``schedule
    resolve RUN reevaluate`` repeats), never first at ledger time.
    """
    from pilot import ledger_writer  # lazily: the ledger writer imports this module

    try:
        record = ledger_writer.evaluation_record(spec, evaluation)
        if record is None:
            raise contracts.ContractError(f"{spec.run_id}: the evaluation holds no per-episode costs and returns "
                                          "(episode_costs, episode_returns; contract 1, results/supplement_schema.py)")
        ledger_writer._validated_record("evaluation", record)
    except ledger_writer.LedgerWriteError as exc:
        raise contracts.ContractError(str(exc)) from exc


def main(argv: list[str] | None = None) -> int:
    """The command line: parse ``argv``, run the stage, and return its exit code."""
    parser = argparse.ArgumentParser(prog="python -m pilot.launch",
                                     description=(__doc__ or "Launch one run").split("\n")[0].replace("``", ""))
    parser.add_argument("stage", choices=("train", "evaluate"),
                        help="train: the training stage; evaluate: the evaluation stage of a trained run")
    parser.add_argument("--spec", required=True, type=Path, help="RUN_DIR/spec.json written by the scheduler")
    parser.add_argument("--run-dir", required=True, type=Path, help="the run directory")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="smoke tests only: allow uncommitted code (the run's derived run gates are not "
                             "added; its stored pending keys still apply)")
    parser.add_argument("--allow-pending", action="store_true",
                        help="smoke tests only: ignore open questions (run gates and result gates)")
    args = parser.parse_args(argv)
    # First of all: the result gates of this process follow the flag (pilot/errors.py). Restored on
    # return, so an in-process caller (a test) is left as it was.
    previous = errors.pending_allowed()
    errors.set_pending_allowed(args.allow_pending)
    try:
        return _run_stage(args)
    finally:
        errors.set_pending_allowed(previous)


def _run_stage(args: argparse.Namespace) -> int:
    """Run the stage ``args`` names; a refusal or an unavailable plug-in or harness becomes its exit code."""
    try:
        spec = RunSpec.from_json(args.spec.read_text(encoding="utf-8"))
        if args.stage == "train":
            return train(spec, args.run_dir, allow_dirty=args.allow_dirty, allow_pending=args.allow_pending)
        return evaluate(spec, args.run_dir, allow_dirty=args.allow_dirty, allow_pending=args.allow_pending)
    except PluginUnavailableError as exc:
        print(f"UNAVAILABLE: {exc}", file=sys.stderr)
        return EXIT_UNAVAILABLE
    except (RunRefused, provenance.DirtyWorktreeError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())

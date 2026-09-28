"""Launch one run: the training stage and the evaluation stage of a single ``RunSpec``.

Usage (normally called by the scheduler, one process per run):

    python -m pilot.launch train    --spec RUN_DIR/spec.json --run-dir RUN_DIR
    python -m pilot.launch evaluate --spec RUN_DIR/spec.json --run-dir RUN_DIR

Training reproduces what ``omnisafe.Agent`` (``AlgoWrapper``) does, step for step, but for any
algorithm class, because ``omnisafe.Agent`` accepts only built-in algorithm names and loads their
YAML from the installed package (so a subclass such as the onset scheduler cannot go through it):

  default YAML of the base algorithm -> validated registered overrides -> exp_name, env_id, algo
  -> epochs = total_steps // steps_per_epoch -> check_all_configs -> torch.set_num_threads(1)
  -> plug-in factory -> learn()

Registered settings applied here (Table 3.1; Part 3.4; Table 8.2): CPU, one torch thread, one
vectorised environment, one process, total steps of the arm, a checkpoint every ten epochs
(200,000 steps). Every other hyperparameter stays OmniSafe's default for the pinned version:
``verify_installed_yaml`` refuses to run unless the installed default YAML is byte-for-byte the
copy committed in ``configs/omnisafe`` (Appendix A, Table A.1), and ``verify_registered_defaults``
checks the values the pre-registration names.

Refusals (open questions, existing output, a claimed directory, uncommitted code, a config that
differs from Table A.1) exit with EXIT_REFUSED, and a plug-in (or a module its factory imports) that
is not in the repository yet exits with EXIT_UNAVAILABLE. Both happen before training starts and
leave the run directory as they found it (one raised after the plug-in's factory ran removes the
claim and OmniSafe's output): they are not run outcomes and never become exclusions (Part 5.6).

Outputs in RUN_DIR: ``spec.json`` (written by the scheduler), ``omnisafe/`` (OmniSafe's log
directory), ``train_result.json`` and, after evaluation, ``evaluation.json``.
"""

from __future__ import annotations

import os

# One thread for every numerical library, set before torch or numpy is imported (Table 3.1).
# Forced, not defaulted: an inherited OMP_NUM_THREADS=8 must not reach a registered run.
THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
for _var in THREAD_VARIABLES:
    os.environ[_var] = "1"

import argparse  # noqa: E402
import csv  # noqa: E402
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
from pilot import provenance  # noqa: E402
from pilot.manifest import RunSpec  # noqa: E402

OMNISAFE_SUBDIR = "omnisafe"
TRAIN_RESULT = "train_result.json"
EVALUATION_RESULT = "evaluation.json"
SPEC_FILE = "spec.json"

LOSS_PREFIX = "Loss/"
MULTIPLIER_COLUMN = "Metrics/LagrangeMultiplier"  # plug-ins log every multiplier under this prefix (contract 3)
STEPS_COLUMN = "TotalEnvSteps"

EXIT_COMPLETED = 0
EXIT_FAILED = 2
EXIT_INTERRUPTED = 3
EXIT_UNAVAILABLE = 4  # a plug-in or the evaluation harness of another role is not in the repository yet
EXIT_REFUSED = 5  # refused before training started (provenance, configuration, or the run directory)
CLAIM_FILE = ".claim"
CONFIG_SOURCE = provenance.REPO_ROOT / "configs" / "omnisafe" / "SOURCE.json"


class RunInterrupted(Exception):
    """The process received SIGTERM or SIGINT: the machine, not the run, stopped it."""


class RunRefused(Exception):
    """The run must not start (or continue) from this checkout or configuration; nothing is excluded."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def registered_overrides(spec: RunSpec, log_dir: Path) -> dict[str, Any]:
    """The custom configuration of a run: only settings the pre-registration fixes."""
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


def build_config(spec: RunSpec, log_dir: Path) -> Any:
    """Build and check the OmniSafe Config exactly as AlgoWrapper._init_config does."""
    return build_config_and_digest(spec, log_dir)[0]


def build_config_and_digest(spec: RunSpec, log_dir: Path) -> tuple[Any, str]:
    """``build_config`` and the sha256 of the installed default YAML it verified (recorded by the run)."""
    from omnisafe.algorithms import ALGORITHM2TYPE
    from omnisafe.envs import support_envs
    from omnisafe.utils.config import check_all_configs, get_default_kwargs_yaml
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
    # Run facts the pilot owner's checkpoint mixin needs (onset checkpoint, Part 3.4). Added
    # after validation, as OmniSafe allows; saved in config.json and included in the config hash.
    from omnisafe.utils.config import Config

    cfgs["pilot_cfgs"] = Config.dict2config({"run_id": spec.run_id, "onset_step": int(spec.onset_step or 0)})
    if cfgs.train_cfgs.parallel != 1 or cfgs.train_cfgs.device != "cpu":
        raise RunRefused("registered runs use one process on the CPU (Table 3.1)")
    if cfgs.train_cfgs.epochs * cfgs.algo_cfgs.steps_per_epoch != spec.total_steps:
        raise RunRefused("total_steps is not a whole number of epochs")
    return cfgs, yaml_digest


# ---------------------------------------------------------------------------
# Output inspection
# ---------------------------------------------------------------------------


def find_omnisafe_run_dir(run_dir: Path) -> Path:
    """OmniSafe writes to RUN_DIR/omnisafe/<exp_name>/seed-XXX-<timestamp>/; return that directory."""
    candidates = sorted((run_dir / OMNISAFE_SUBDIR).glob("*/seed-*"))
    candidates = [c for c in candidates if c.is_dir()]
    if len(candidates) != 1:
        raise FileNotFoundError(f"expected one OmniSafe run directory under {run_dir}, found {len(candidates)}")
    return candidates[0]


def read_progress(omnisafe_dir: Path) -> list[dict[str, str]]:
    with open(omnisafe_dir / "progress.csv", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


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


def nonfinite_state(algo: Any) -> str | None:
    """Cause of a non-finite state in a live algorithm object, or None (Part 5.6).

    A NaN or inf loss makes the next optimiser step write non-finite parameters, and the next
    forward pass then raises inside torch.distributions before the epoch's row is logged, so the
    state of the model is checked, not only the log.
    """
    import torch

    lagrange = getattr(algo, "_lagrange", None)
    if lagrange is not None:
        value = lagrange.lagrangian_multiplier
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
    every epoch is logged and the final checkpoint loads.
    """
    rows: list[dict[str, str]] = []
    omnisafe_dir: Path | None = None
    try:
        omnisafe_dir = find_omnisafe_run_dir(run_dir)
        rows = read_progress(omnisafe_dir)
    except (FileNotFoundError, OSError):
        rows = []
    bad_mult = _nonfinite(rows, MULTIPLIER_COLUMN)
    bad_loss = _nonfinite(rows, LOSS_PREFIX)
    if bad_mult:
        return "failed", "non_finite_multiplier", "; ".join(bad_mult[:5])
    if bad_loss:
        return "failed", "non_finite_loss", "; ".join(bad_loss[:5])
    trace = ""
    if exception is not None:
        trace = "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))[-4000:]
        if state_cause is not None:
            return "failed", state_cause, trace
    expected_epochs = spec.total_steps // R.STEPS_PER_EPOCH
    last_steps = int(float(rows[-1][STEPS_COLUMN])) if rows and rows[-1].get(STEPS_COLUMN) else 0
    complete = len(rows) == expected_epochs and last_steps == spec.total_steps
    final_saved = omnisafe_dir is not None and final_checkpoint_loads(omnisafe_dir / "torch_save" / f"epoch-{expected_epochs}.pt")
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
    """Serialise first (a TypeError leaves nothing behind), then write, fsync and rename."""
    text = json.dumps(plain(data), indent=2, sort_keys=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _versions() -> dict[str, str]:
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
    """The SIGTERM and SIGINT handler of ``train``.

    While armed, the first signal disarms the gate and raises RunInterrupted; every other signal
    (a second one arriving at the same time, or any signal after training ended) is only recorded,
    and ``train`` re-delivers it to the original handler once train_result.json is written. Recording
    rather than blocking: a thread mask would not help, because OmniSafe's TensorBoard writer thread
    receives the signal and CPython still runs the Python handler in the main thread.
    """

    def __init__(self) -> None:
        self.armed = True
        self.deferred: list[int] = []

    def __call__(self, signum: int, frame: Any) -> None:  # pragma: no cover - signal path
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
    ``PluginUnavailableError`` (``main``: EXIT_UNAVAILABLE), when the run must not start. Raised after
    the claim (by the plug-in's factory, or a check after it), they first remove the claim and
    OmniSafe's output, so the run directory is left as it was found.

    The first SIGTERM or SIGINT from the installation of the handlers (right after the claim) to the
    end of training ends the run as an interruption (EXIT_INTERRUPTED, train_result.json status
    "interrupted"). Any other signal (a second one, or one after training ended) is only recorded.
    On every exit from here on, the caller's original handlers are restored and the recorded signals
    are re-delivered to them, after train_result.json is written when it can be.
    """
    from pilot.algorithms import load_plugin

    run_dir = Path(os.path.abspath(run_dir))
    if spec.open_questions and not allow_pending:
        raise RunRefused(f"{spec.run_id} depends on open questions {spec.open_questions}; see configs/registered.py PENDING")
    if (run_dir / OMNISAFE_SUBDIR).exists() or (run_dir / TRAIN_RESULT).exists():
        raise RunRefused(f"{run_dir} already holds training output; a run is never trained twice in one directory")
    # Every check that needs no output runs before the claim, so a refusal leaves nothing behind,
    # and so does everything that could fail between the claim and the guarded training below.
    commit = provenance.commit_hash() if allow_dirty else provenance.require_clean_worktree()
    factory = load_plugin(spec.plugin)
    if not allow_dirty:
        _refuse_unverified_code()
    cfgs, yaml_digest = build_config_and_digest(spec, run_dir / OMNISAFE_SUBDIR)
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
        "allow_pending": bool(allow_pending),
    }
    # Claim the directory atomically: a second process started for the same run stops here.
    try:
        fd = os.open(run_dir / CLAIM_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RunRefused(f"{run_dir} is already claimed by another process; a run is never trained twice") from None
    os.write(fd, f"{os.getpid()}\n".encode())
    os.close(fd)

    result["started"] = provenance.utc_now().isoformat()
    gate = _SignalGate()
    old_handlers: dict[int, Any] = {}
    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            old_handlers[sig] = signal.signal(sig, gate)
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
    """``train`` after the claim, with ``gate`` installed: train, classify, write train_result.json."""
    from pilot.algorithms import PluginUnavailableError

    t0 = time.monotonic()
    exception: BaseException | None = None
    state_cause: str | None = None
    interrupted = False
    algo: Any = None
    try:
        try:
            os.environ["OMNISAFE_DEVICE"] = cfgs.train_cfgs.device
            try:
                algo = factory(spec.task, cfgs, spec)
            except ModuleNotFoundError as exc:
                # A module the factory imports lazily is missing: like a missing plug-in, not a run outcome.
                raise PluginUnavailableError(
                    f"the factory of plug-in {spec.plugin!r} could not import {exc.name!r}: {exc}"
                ) from exc
            if not allow_dirty:
                _refuse_unverified_code()  # modules the factory imported lazily
            _check_threads(reset=False)  # a factory must not change the thread count
            # A plug-in may add its own settings (for example onset_cfgs) after validation; hash the result.
            result["config_hash"] = provenance.config_hash(cfgs.todict())
            algo.learn()
        finally:
            # From here on a signal is only recorded. One that arrives before this line raises
            # RunInterrupted here, which the handler below catches like any other interruption.
            gate.armed = False
    except RunInterrupted as exc:
        interrupted = True
        result["interruption"] = str(exc)
    except (RunRefused, PluginUnavailableError):
        # Raised before learn(): remove what this process created, so the run stays untouched.
        shutil.rmtree(run_dir / OMNISAFE_SUBDIR, ignore_errors=True)
        (run_dir / CLAIM_FILE).unlink(missing_ok=True)
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
        result.update(status=status, failure_cause=cause, detail=detail)
        code = EXIT_COMPLETED if status == "completed" else EXIT_FAILED
    try:
        result["omnisafe_dir"] = str(find_omnisafe_run_dir(run_dir).relative_to(run_dir))
    except FileNotFoundError:
        result["omnisafe_dir"] = None
    _write_json_atomic(run_dir / TRAIN_RESULT, result)
    return code


def evaluate(spec: RunSpec, run_dir: Path, *, allow_dirty: bool = False) -> int:
    """Run the evaluation stage through the evaluation harness of Role 2.

    Contract (pilot/contracts.py): ``envs.evaluation.evaluate_run(omnisafe_dir, spec_dict)``
    returns a dict with final_cost, final_return and selection {step: [cost, return]}
    for the run's last ten checkpoints (Part 3.4; Part 4.1 rule 1).
    """
    from pilot.contracts import load_evaluator, validate_evaluation

    run_dir = Path(os.path.abspath(run_dir))
    commit = provenance.commit_hash() if allow_dirty else provenance.require_clean_worktree()
    train_result = json.loads((run_dir / TRAIN_RESULT).read_text(encoding="utf-8"))
    if train_result.get("status") != "completed":
        raise RunRefused(f"{spec.run_id}: training did not complete; nothing to evaluate")
    evaluator = load_evaluator()
    if not allow_dirty:
        _refuse_unverified_code()  # the harness itself must be committed
    t0 = time.monotonic()
    omnisafe_dir = run_dir / train_result["omnisafe_dir"]
    evaluation = plain(dict(evaluator(str(omnisafe_dir), spec.to_dict())))
    if not allow_dirty:
        _refuse_unverified_code()  # modules the harness imported lazily
    evaluation["eval_wall_clock_hours"] = (time.monotonic() - t0) / 3600.0
    evaluation["eval_commit_hash"] = commit
    validate_evaluation(evaluation, spec, omnisafe_dir)
    _write_json_atomic(run_dir / EVALUATION_RESULT, evaluation)
    return EXIT_COMPLETED


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m pilot.launch", description=__doc__.split("\n")[0])
    parser.add_argument("stage", choices=("train", "evaluate"))
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--allow-dirty", action="store_true", help="smoke tests only: allow uncommitted code")
    parser.add_argument("--allow-pending", action="store_true", help="smoke tests only: ignore open questions")
    args = parser.parse_args(argv)
    spec = RunSpec.from_json(args.spec.read_text(encoding="utf-8"))
    from pilot.algorithms import PluginUnavailableError

    try:
        if args.stage == "train":
            return train(spec, args.run_dir, allow_dirty=args.allow_dirty, allow_pending=args.allow_pending)
        return evaluate(spec, args.run_dir, allow_dirty=args.allow_dirty)
    except PluginUnavailableError as exc:
        print(f"UNAVAILABLE: {exc}", file=sys.stderr)
        return EXIT_UNAVAILABLE
    except (RunRefused, provenance.DirtyWorktreeError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())

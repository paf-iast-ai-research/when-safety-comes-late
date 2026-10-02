"""Algorithm plug-ins run by the launcher, and the full-state checkpoint mixin.

The launcher (``pilot/launch.py``) builds the OmniSafe configuration and calls a plug-in
``make(env_id, cfgs, spec) -> BaseAlgo``. The plug-ins in this file are the pilot owner's:

* ``make_ppolag``: unmodified OmniSafe PPO-Lagrangian (the determinism check, Table 3.1).
* ``make_unconstrained_ppo``: unmodified OmniSafe PPO (the Study B pilot's unconstrained run,
  Part 3.6).

The Study A plug-ins, the battery's continuation plug-ins and the Study B plug-ins belong to Roles 2
and 5 (``pilot/manifest.py: PLUGINS``); they mix in ``FullStateCheckpointMixin`` the same way, and a
continuation restores its parent with ``pilot.dependencies.restore_learner`` in its ``_init``.

Why the mixin: OmniSafe 0.5.0 saves only the actor ('pi') and the observation normaliser in each
checkpoint (``PolicyGradient._init_log``). The battery's continuations (Table 2.2) and Study B's
few-shot continuations (Table 2.5) need the critics as well (restored); the optimiser states, the
learning-rate scheduler, the multiplier and the logger's 50-episode windows (``Metrics/EpCost`` is
J_C of the next multiplier update) are also saved, though the Q-continuations answer (Table 9.1,
``pilot.dependencies.restore_learner``) restores none of them: the optimisers, the schedule and the
windows start fresh, and the plug-in holds the multiplier at 0 (fine-tuning) or starts a fresh one
(transfer, few-shot). The mixin saves them in the same files, at the same epochs, without touching
any random number stream, so training is bit-for-bit unchanged. Random number generator states
(torch, numpy, the environments) are NOT saved: a continuation reseeds, so it is not a bit-for-bit
extension of the original run (``pilot.dependencies.restore_learner``: the continuation is seeded by
``cfgs.seed = spec.seed``).
"""

from __future__ import annotations

import collections
import importlib
import numbers
from typing import Any, Callable

from pilot.errors import RunRefused
from pilot.manifest import PLUGINS


class PluginUnavailableError(ImportError):
    """Code another role provides (a plug-in, an evaluation harness) is not in the repository yet (exit 4)."""


# The packages of the other roles (Roles 2 to 5) whose code may not be written yet; not ``results``
# (Role 4's schemas), which are frozen in this repository with their SHA-256 recorded beside them
# (results/*.sha256). A ModuleNotFoundError naming one of them, or a submodule of one, or an
# ImportError of a name such a module does not define, means that role's code is not written (or not
# merged) yet: the stage is unavailable (exit 4), not failed, in both launcher stages (HANDOVER.md
# section 8; pilot/launch.py).
REPOSITORY_PACKAGES = ("envs", "metrics", "studyb", "analysis")


def is_repository_module(name: str | None) -> bool:
    """True if ``name`` is one of REPOSITORY_PACKAGES or a submodule of one."""
    return bool(name) and str(name).split(".")[0] in REPOSITORY_PACKAGES


def is_missing_repository_import(exc: BaseException) -> bool:
    """True if ``exc`` is an import of another role's code that is not written yet (HANDOVER.md section 8; pilot/launch.py).

    Two forms, one meaning: a repository module that does not exist (ModuleNotFoundError, ``name``
    the missing module), and a repository module that exists but does not define the name imported
    from it yet (``from metrics.hook import install_plasticity_hook`` before Role 3 has written
    it: a plain ImportError whose ``name`` is the module, ``'metrics.hook'``). Both make the stage
    unavailable (exit 4), never a run outcome. An import of a library (either form) is not: the
    environment is at fault, and the caller keeps its own handling. PluginUnavailableError (an
    ImportError too) has already been classified and is excluded.
    """
    return (isinstance(exc, ImportError) and not isinstance(exc, PluginUnavailableError)
            and is_repository_module(exc.name))


def import_provided(module_name: str, description: str) -> Any:
    """Import a module another role provides; PluginUnavailableError while its code is not written yet.

    Unavailable (HANDOVER.md section 8; pilot/launch.py): the module or a parent package does not exist, or it
    exists but imports, at its top level, a repository module or a name of one that does not exist
    yet (for example a Role 5 harness importing Role 2's ``envs.evaluation``, or a function of it,
    before that is merged; ``is_missing_repository_import``). A missing library re-raises the
    ImportError: an environment problem is shown, not reported as another role's unfinished code.
    ``description`` names what was imported and who provides it, for the message.
    """
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        missing = exc.name or ""
        if isinstance(exc, ModuleNotFoundError) and missing and (
                module_name == missing or module_name.startswith(missing + ".")):
            raise PluginUnavailableError(f"{description} is not in the repository yet") from exc
        if is_missing_repository_import(exc):
            what = repr(missing) if isinstance(exc, ModuleNotFoundError) else f"a name from {missing!r}"
            raise PluginUnavailableError(
                f"{description} imports {what}, which is not in the repository yet: {exc}"
            ) from exc
        raise  # a library is missing from the environment, not another role's code: show the real error


class _MultiplierState:
    """Adapter giving the Lagrange multiplier a ``state_dict`` for OmniSafe's ``Logger.torch_save``.

    ``torch_save`` stores ``v.state_dict()`` for every entry that has one, so this reads the
    multiplier's current value at every save rather than a stale copy.
    """

    def __init__(self, lagrange: Any) -> None:
        self._lagrange = lagrange

    def state_dict(self) -> dict[str, Any]:
        import torch

        value = self._lagrange.lagrangian_multiplier
        if isinstance(value, torch.Tensor):
            tensor = value.detach().clone()
        else:  # PIDLagrangian exposes the multiplier as a Python float: keep full precision
            tensor = torch.tensor(float(value), dtype=torch.float64)
        state: dict[str, Any] = {"value": tensor, "class": type(self._lagrange).__name__}
        # Plain Python numbers only, so the checkpoint loads with torch.load(weights_only=True). numpy
        # scalars (np.int64, np.float32, ...) and one-element tensors are converted, not dropped.
        numeric = {}
        for k, v in vars(self._lagrange).items():
            number = _plain_number(v)
            if number is not None:
                numeric[k] = number
        state["numeric_attributes"] = numeric
        deques = {
            k: [float(x) for x in v] for k, v in vars(self._lagrange).items() if isinstance(v, collections.deque)
        }
        if deques:
            state["deque_attributes"] = deques
        return state


def _plain_number(value: Any) -> int | float | None:
    """``value`` as a Python int or float if it is a real number (not a bool), else None."""
    import numpy as np
    import torch

    if isinstance(value, torch.Tensor):
        if value.numel() != 1 or value.dtype == torch.bool:
            return None
        value = value.item()
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value)
    return None


class _WindowState:
    """The logger's moving-window episode statistics, as a ``state_dict`` for ``Logger.torch_save``.

    ``Metrics/EpCost`` is the mean over the last 50 finished episodes, and PPO-Lagrangian updates
    the multiplier from it (``ppo_lag.py``: ``Jc = self._logger.get_stats('Metrics/EpCost')[0]``),
    so a continuation that resumed the multiplier would need the window's contents, not only its mean
    (saved for Q-continuations; its answer in Table 9.1 does not restore them).
    """

    KEYS = ("Metrics/EpRet", "Metrics/EpCost", "Metrics/EpLen")

    def __init__(self, logger: Any) -> None:
        self._logger = logger

    def state_dict(self) -> dict[str, Any]:
        import torch

        data = self._logger._data  # OmniSafe 0.5.0: key -> deque(maxlen=window_length)
        return {k: torch.tensor([float(v) for v in data[k]], dtype=torch.float64) for k in self.KEYS if k in data}


BATCH_COST_KEY = "Metrics/BatchEpCost"
BATCH_RETURN_KEY = "Metrics/BatchEpRet"


class FullStateCheckpointMixin:
    """Checkpoints and logging that Part 3.4 requires and OmniSafe 0.5.0 does not provide.

    Use as ``class FullStatePPOLag(FullStateCheckpointMixin, PPOLag)``. None of this consumes a
    random number or changes an update, so training is bit-for-bit that of the base class (the
    Role 3 hook of item 5 measures without changing anything; interventions are applied by the
    Study A plug-in itself, envs.onset, HANDOVER.md section 8).

    1. Full state in every checkpoint ``torch_save/epoch-{k}.pt``: pi, obs_normalizer (if used),
       reward_critic, cost_critic (if present), actor_optimizer, reward_critic_optimizer,
       cost_critic_optimizer (if present), actor_scheduler, episode_windows (the contents of the
       50-episode windows of Metrics/EpRet, EpCost and EpLen), and for Lagrangian algorithms
       lagrange (value, numeric state) and lambda_optimizer (if present; not for PID). A PID
       multiplier's state is in ``lagrange['numeric_attributes']`` and ``['deque_attributes']``
       (not restored under the Q-continuations answer, Table 9.1).
       Not saved: random number generator states. A plug-in adds entries of its own through
       ``_extra_checkpoint_state`` (e.g. Study B's per-level multipliers), loadable with
       ``torch.load(weights_only=True)``.
    2. Checkpoints off OmniSafe's ``save_model_freq`` grid: every step of
       ``cfgs['pilot_cfgs']['extra_checkpoint_steps']`` (``pilot.contracts.extra_checkpoint_steps``,
       set by the launcher) and the onset (Part 3.4: "at onset"), e.g. N = 0.25 total-steps
       matched (epoch 125), its manipulation-check step onset + 200,000 (box H3 (c)) and the
       few-shot horizon 500,000 (Table 2.5).
    3. The training-batch mean episodic cost and return of each epoch (Part 3.4), logged as
       ``Metrics/BatchEpCost`` and ``Metrics/BatchEpRet``. OmniSafe's ``Metrics/EpCost`` is a
       moving mean over the last 50 episodes (2.5 epochs), not the epoch's batch.
    4. ``epoch-0.pt`` is rewritten with the full state after the plug-in's ``_init`` (a
       continuation restores its parent there, ``pilot.dependencies.restore_learner``), so it holds
       the learner training starts from.
    5. For the Study A plug-ins (``cfgs['pilot_cfgs']['plasticity']``, from
       ``pilot.contracts.plasticity_required``): Role 3's plasticity hook
       ``metrics.hook.install_plasticity_hook(self)`` (Table 2.3; contract 2 of pilot/contracts.py), installed
       last. If ``metrics.hook``, or the function in it, is missing, the ModuleNotFoundError or
       ImportError propagates and the launcher reports the plug-in as unavailable (exit 4;
       ``is_missing_repository_import``): a Study A run never trains without it.
    """

    def _init_log(self) -> None:
        """OmniSafe's hook: its own setup, then the mixin's saves and metrics (items 1 to 5 above)."""
        super()._init_log()  # type: ignore[misc]  # sets up the default saver and writes epoch-0.pt
        self._log_batch_metrics()
        self._save_at_steps()
        ac = self._actor_critic  # type: ignore[attr-defined]
        what: dict[str, Any] = {"pi": ac.actor}
        if self._cfgs.algo_cfgs.obs_normalize:  # type: ignore[attr-defined]
            what["obs_normalizer"] = self._env.save()["obs_normalizer"]  # type: ignore[attr-defined]
        what["reward_critic"] = ac.reward_critic
        if hasattr(ac, "cost_critic"):
            what["cost_critic"] = ac.cost_critic
        what["actor_optimizer"] = ac.actor_optimizer
        what["reward_critic_optimizer"] = ac.reward_critic_optimizer
        if hasattr(ac, "cost_critic_optimizer"):
            what["cost_critic_optimizer"] = ac.cost_critic_optimizer
        what["actor_scheduler"] = ac.actor_scheduler
        what["episode_windows"] = _WindowState(self._logger)  # type: ignore[attr-defined]
        lagrange = getattr(self, "_lagrange", None)
        if lagrange is not None:
            what["lagrange"] = _MultiplierState(lagrange)
            if hasattr(lagrange, "lambda_optimizer"):
                what["lambda_optimizer"] = lagrange.lambda_optimizer
        extra = dict(self._extra_checkpoint_state())
        clash = sorted(set(extra) & set(what))
        if clash:
            raise RunRefused(f"the plug-in's extra checkpoint state reuses the mixin's keys {clash}")
        what.update(extra)
        self._logger.setup_torch_saver(what)  # type: ignore[attr-defined]
        self._logger.torch_save()  # type: ignore[attr-defined]  # rewrite epoch-0.pt with the full state
        pilot_cfgs = self._pilot_cfgs()
        if pilot_cfgs.get("plasticity"):
            from metrics.hook import install_plasticity_hook  # Role 3; a missing module or name is exit 4, not a crash

            install_plasticity_hook(self)

    def _extra_checkpoint_state(self) -> dict[str, Any]:
        """Entries a plug-in adds to every checkpoint (default none), merged into the saver's ``what``.

        Each value is an object with ``state_dict()`` (read at every save) or a value that loads with
        ``torch.load(weights_only=True)`` (contract 3). A key the mixin already saves is refused.
        """
        return {}

    def _pilot_cfgs(self) -> Any:
        """``cfgs['pilot_cfgs']`` (set by the launcher, pilot.launch.pilot_config), or an empty mapping."""
        cfgs = self._cfgs  # type: ignore[attr-defined]
        pilot_cfgs = cfgs.get("pilot_cfgs") if hasattr(cfgs, "get") else None
        return pilot_cfgs if pilot_cfgs else {}

    def _log_batch_metrics(self) -> None:
        """Log Metrics/BatchEpCost and BatchEpRet per finished episode (Part 3.4; item 3 above)."""
        logger = self._logger  # type: ignore[attr-defined]
        adapter = self._env  # type: ignore[attr-defined]
        logger.register_key(BATCH_COST_KEY)  # no window: the mean of this epoch's finished episodes
        logger.register_key(BATCH_RETURN_KEY)
        original = adapter._log_metrics

        def _log_metrics(log: Any, idx: int) -> None:
            original(log, idx)
            log.store({BATCH_COST_KEY: adapter._ep_cost[idx], BATCH_RETURN_KEY: adapter._ep_ret[idx]})

        adapter._log_metrics = _log_metrics

    def _save_at_steps(self) -> None:
        """Save a checkpoint at every extra step that OmniSafe's own cadence does not save.

        The steps are ``pilot_cfgs.extra_checkpoint_steps`` and the onset (``pilot_cfgs.onset_step``),
        converted to epochs with the run's own ``algo_cfgs.steps_per_epoch``. OmniSafe saves after
        epoch e when e is a multiple of ``save_model_freq`` or the last epoch
        (``policy_gradient.py``, in short: ``if (epoch + 1) % save_model_freq == 0 or (epoch + 1) == epochs``);
        every other step is saved from ``dump_tabular``, which runs after the epoch's update and
        learning-rate step, so ``epoch-k.pt`` holds the state after k epochs, as OmniSafe's own
        saves do. Saving draws no random number and changes nothing that training reads. RunRefused
        for a step that is not a whole number, is negative, is not a whole epoch or lies beyond the run.
        """
        from pilot.contracts import whole_number

        cfgs = self._cfgs  # type: ignore[attr-defined]
        pilot_cfgs = self._pilot_cfgs()
        try:
            steps = {whole_number(s) for s in (pilot_cfgs.get("extra_checkpoint_steps") or ())}
            onset_step = whole_number(pilot_cfgs.get("onset_step") or 0)
        except TypeError as exc:
            raise RunRefused(f"extra checkpoint steps and the onset must be whole step counts: {exc}") from None
        if onset_step > 0:
            steps.add(onset_step)
        steps_per_epoch = int(cfgs.algo_cfgs.steps_per_epoch)
        epochs = int(cfgs.train_cfgs.epochs)
        save_model_freq = int(cfgs.logger_cfgs.save_model_freq)
        extra_epochs: set[int] = set()
        for step in sorted(steps):
            if step < 0:
                raise RunRefused(f"extra checkpoint step {step} is negative")
            if step % steps_per_epoch:
                raise RunRefused(f"extra checkpoint step {step} is not a whole epoch of {steps_per_epoch} steps")
            epoch = step // steps_per_epoch
            if epoch > epochs:
                raise RunRefused(f"extra checkpoint step {step} lies beyond the run's {epochs} epochs")
            if epoch == 0 or epoch == epochs or epoch % save_model_freq == 0:
                continue  # epoch-0.pt is written by _init_log; OmniSafe's cadence saves the rest
            extra_epochs.add(epoch)
        if not extra_epochs:
            return
        logger = self._logger  # type: ignore[attr-defined]
        original_dump = logger.dump_tabular

        def dump_tabular() -> None:
            original_dump()
            if logger.current_epoch in extra_epochs:  # the state after current_epoch epochs
                logger.torch_save()  # looked up at call time: a Role 3 wrapper of torch_save sees it

        logger.dump_tabular = dump_tabular


def _full_state_class(base: type) -> type:
    """``base`` with ``FullStateCheckpointMixin`` mixed in, named ``FullState<base>``."""
    return type(f"FullState{base.__name__}", (FullStateCheckpointMixin, base), {})


def make_ppolag(env_id: str, cfgs: Any, spec: Any) -> Any:
    """Unmodified OmniSafe PPO-Lagrangian with full-state checkpoints."""
    from omnisafe.algorithms.on_policy.naive_lagrange.ppo_lag import PPOLag

    return _full_state_class(PPOLag)(env_id=env_id, cfgs=cfgs)


def make_unconstrained_ppo(env_id: str, cfgs: Any, spec: Any) -> Any:
    """Unmodified OmniSafe PPO (no cost term) with full-state checkpoints."""
    from omnisafe.algorithms.on_policy.base.ppo import PPO

    return _full_state_class(PPO)(env_id=env_id, cfgs=cfgs)


def load_plugin(key: str) -> Callable[[str, Any, Any], Any]:
    """Import the factory registered for ``key`` in ``pilot.manifest.PLUGINS``.

    PluginUnavailableError (exit 4) when the plug-in's module, or a repository module (or a name
    of one) it imports at its top level, is not written yet (``import_provided``), or the module
    lacks the factory. KeyError for a key that is not in PLUGINS.
    """
    if key not in PLUGINS:
        raise KeyError(f"unknown plug-in {key!r}")
    target, owner = PLUGINS[key]
    module_name, func_name = target.split(":")
    module = import_provided(module_name, f"plug-in {key!r} ({target}, provided by {owner})")
    try:
        return getattr(module, func_name)
    except AttributeError as exc:
        raise PluginUnavailableError(f"{module_name} has no function {func_name!r} yet ({owner})") from exc

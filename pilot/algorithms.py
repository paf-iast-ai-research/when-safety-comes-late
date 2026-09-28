"""Algorithm plug-ins run by the launcher, and the full-state checkpoint mixin.

The launcher (``pilot/launch.py``) builds the OmniSafe configuration and calls a plug-in
``make(env_id, cfgs, spec) -> BaseAlgo``. The plug-ins in this file are the pilot owner's:

* ``make_ppolag``: unmodified OmniSafe PPO-Lagrangian (the determinism check, Table 3.1).
* ``make_unconstrained_ppo``: unmodified OmniSafe PPO (the Study B pilot's unconstrained run,
  Part 3.6).

The Study A and Study B plug-ins belong to Roles 2 and 5 (``pilot/manifest.py: PLUGINS``); they
should mix in ``FullStateCheckpointMixin`` the same way.

Why the mixin: OmniSafe 0.5.0 saves only the actor ('pi') and the observation normaliser in each
checkpoint (``PolicyGradient._init_log``). The battery's continuations (Table 2.2) and Study B's
few-shot continuations (Table 2.5) need the critics, the optimiser states, the learning-rate
scheduler, the multiplier and the logger's 50-episode windows (``Metrics/EpCost`` is J_C of the
next multiplier update) as well. The mixin saves them in the same files, at the same epochs,
without touching any random number stream, so training is bit-for-bit unchanged. Random number
generator states (torch, numpy, the environments) are NOT saved: a continuation reseeds, so it
is not a bit-for-bit extension of the original run (Q-continuations).
"""

from __future__ import annotations

import collections
import importlib
import numbers
from typing import Any, Callable

from pilot.manifest import PLUGINS


class PluginUnavailableError(ImportError):
    """The plug-in named by a run spec is not in the repository yet."""


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
    so a continuation needs the window's contents, not only its mean.
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
    random number or changes an update, so training is bit-for-bit that of the base class.

    1. Full state in every checkpoint ``torch_save/epoch-{k}.pt``: pi, obs_normalizer (if used),
       reward_critic, cost_critic, actor_optimizer, reward_critic_optimizer,
       cost_critic_optimizer (if present), actor_scheduler, episode_windows (the contents of the
       50-episode windows of Metrics/EpRet, EpCost and EpLen), and for Lagrangian algorithms
       lagrange (value, numeric state) and lambda_optimizer. For PID multipliers restore from
       ``lagrange['numeric_attributes']`` and ``['deque_attributes']``. Not saved: random number
       generator states.
    2. A checkpoint at onset (Part 3.4: "at onset") when the onset epoch is not on the
       ten-epoch grid, e.g. N = 0.25 total-steps matched (epoch 125). The onset step is read
       from ``cfgs['pilot_cfgs']['onset_step']``, set by the launcher.
    3. The training-batch mean episodic cost and return of each epoch (Part 3.4), logged as
       ``Metrics/BatchEpCost`` and ``Metrics/BatchEpRet``. OmniSafe's ``Metrics/EpCost`` is a
       moving mean over the last 50 episodes (2.5 epochs), not the epoch's batch.
    """

    def _init_log(self) -> None:  # noqa: D401 - OmniSafe hook
        super()._init_log()  # type: ignore[misc]  # sets up the default saver and writes epoch-0.pt
        self._log_batch_metrics()
        self._save_at_onset()
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
        self._logger.setup_torch_saver(what)  # type: ignore[attr-defined]
        self._logger.torch_save()  # type: ignore[attr-defined]  # rewrite epoch-0.pt with the full state

    def _log_batch_metrics(self) -> None:
        logger = self._logger  # type: ignore[attr-defined]
        adapter = self._env  # type: ignore[attr-defined]
        logger.register_key(BATCH_COST_KEY)  # no window: the mean of this epoch's finished episodes
        logger.register_key(BATCH_RETURN_KEY)
        original = adapter._log_metrics

        def _log_metrics(log: Any, idx: int) -> None:
            original(log, idx)
            log.store({BATCH_COST_KEY: adapter._ep_cost[idx], BATCH_RETURN_KEY: adapter._ep_ret[idx]})

        adapter._log_metrics = _log_metrics

    def _save_at_onset(self) -> None:
        pilot_cfgs = self._cfgs.get("pilot_cfgs") if hasattr(self._cfgs, "get") else None  # type: ignore[attr-defined]
        onset_step = int(pilot_cfgs.get("onset_step") or 0) if pilot_cfgs else 0
        if onset_step <= 0:
            return
        steps_per_epoch = int(self._cfgs.algo_cfgs.steps_per_epoch)  # type: ignore[attr-defined]
        if onset_step % steps_per_epoch:
            raise ValueError(f"onset step {onset_step} is not a whole epoch")
        onset_epoch = onset_step // steps_per_epoch
        if onset_epoch % int(self._cfgs.logger_cfgs.save_model_freq) == 0:  # type: ignore[attr-defined]
            return  # OmniSafe's own cadence saves it
        logger = self._logger  # type: ignore[attr-defined]
        original_dump = logger.dump_tabular

        def dump_tabular() -> None:
            original_dump()
            if logger.current_epoch == onset_epoch:  # state after onset_epoch epochs = at onset
                logger.torch_save()

        logger.dump_tabular = dump_tabular


def _full_state_class(base: type) -> type:
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
    """Import the factory registered for ``key`` in ``pilot.manifest.PLUGINS``."""
    if key not in PLUGINS:
        raise KeyError(f"unknown plugin {key!r}")
    target, owner = PLUGINS[key]
    module_name, func_name = target.split(":")
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if missing and (module_name == missing or module_name.startswith(missing + ".")):
            raise PluginUnavailableError(
                f"plug-in {key!r} needs {target}, which is provided by {owner} and is not in the repository yet"
            ) from exc
        raise  # the plug-in exists but one of its own imports failed: show the real error
    try:
        return getattr(module, func_name)
    except AttributeError as exc:
        raise PluginUnavailableError(f"{module_name} has no function {func_name!r} ({owner})") from exc

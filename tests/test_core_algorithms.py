"""FullStateCheckpointMixin: extra checkpoints (``pilot.contracts.extra_checkpoint_steps``), the extra-state
hook (used by Study B) and the Role 3 plasticity hook (``pilot.contracts.plasticity_required``), tested with
stand-ins for OmniSafe's logger, adapter and actor-critic (no training)."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from pilot import algorithms
from pilot.algorithms import FullStateCheckpointMixin
from pilot.errors import RunRefused


class Cfg(dict):
    """Attribute access like omnisafe.utils.config.Config (a dict subclass)."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


def cfg(data: dict) -> Cfg:
    return Cfg({k: cfg(v) if isinstance(v, dict) else v for k, v in data.items()})


class FakeLogger:
    def __init__(self) -> None:
        self.current_epoch = 0
        self.saves: list[tuple[int, list[str]]] = []
        self.what: dict | None = None
        self._data: dict = {}
        self.keys: list[str] = []

    def register_key(self, key: str) -> None:
        self.keys.append(key)

    def store(self, data: dict) -> None:
        pass

    def setup_torch_saver(self, what: dict) -> None:
        self.what = what

    def torch_save(self) -> None:
        self.saves.append((self.current_epoch, sorted(self.what or {})))

    def dump_tabular(self) -> None:
        self.current_epoch += 1  # as OmniSafe's Logger: the epoch counter moves when the row is written


class FakeAdapter:
    _ep_cost = [0.0]
    _ep_ret = [0.0]

    def _log_metrics(self, log: Any, idx: int) -> None:
        pass

    def save(self) -> dict:
        return {"obs_normalizer": "normaliser"}


class FakeActorCritic:
    actor = "actor"
    reward_critic = "reward_critic"
    cost_critic = "cost_critic"
    actor_optimizer = "actor_optimizer"
    reward_critic_optimizer = "reward_critic_optimizer"
    cost_critic_optimizer = "cost_critic_optimizer"
    actor_scheduler = "actor_scheduler"


class Base:
    """OmniSafe's PolicyGradient._init_log in short: the logger, a saver with pi, epoch-0.pt."""

    def _init_log(self) -> None:
        self._logger = FakeLogger()
        self._logger.setup_torch_saver({"pi": "actor"})
        self._logger.torch_save()


class Algo(FullStateCheckpointMixin, Base):
    def __init__(self, pilot_cfgs: dict | None, *, epochs: int = 10, steps_per_epoch: int = 2_000,
                 save_model_freq: int = 10, extra: dict | None = None) -> None:
        data = {"algo_cfgs": {"obs_normalize": True, "steps_per_epoch": steps_per_epoch},
                "train_cfgs": {"epochs": epochs}, "logger_cfgs": {"save_model_freq": save_model_freq}}
        if pilot_cfgs is not None:
            data["pilot_cfgs"] = pilot_cfgs
        self._cfgs = cfg(data)
        self._actor_critic = FakeActorCritic()
        self._env = FakeAdapter()
        self._extra = extra or {}
        self._init_log()

    def _extra_checkpoint_state(self) -> dict:
        return self._extra

    def run(self) -> None:
        """OmniSafe's learn loop, as far as saving goes (policy_gradient.py: dump_tabular, then torch_save)."""
        freq, epochs = int(self._cfgs.logger_cfgs.save_model_freq), int(self._cfgs.train_cfgs.epochs)
        for epoch in range(epochs):
            self._logger.dump_tabular()
            if (epoch + 1) % freq == 0 or epoch + 1 == epochs:
                self._logger.torch_save()


def _saved_epochs(algo: Algo) -> list[int]:
    return [epoch for epoch, _ in algo._logger.saves]


def test_extra_steps_off_the_cadence_are_saved_once_each() -> None:
    pilot_cfgs = {"onset_step": 8_000, "extra_checkpoint_steps": [4_000, 6_000, 20_000, 10_000, 0], "plasticity": False}
    algo = Algo(pilot_cfgs, epochs=10, save_model_freq=5)
    algo.run()
    # epoch-0.pt twice (OmniSafe's, then the full state), the extra epochs 2, 3 and the onset 4, the cadence 5 and 10
    assert _saved_epochs(algo) == [0, 0, 2, 3, 4, 5, 10]
    assert algo._logger.saves[-1][1] == algo._logger.saves[2][1]  # every save holds the full state
    assert {"reward_critic", "actor_optimizer", "episode_windows", "obs_normalizer"} <= set(algo._logger.saves[2][1])


def test_without_pilot_cfgs_only_omnisafes_cadence_saves() -> None:
    for pilot_cfgs in (None, {}, {"onset_step": 0, "extra_checkpoint_steps": []}):
        algo = Algo(pilot_cfgs, epochs=4, save_model_freq=10)
        algo.run()
        assert _saved_epochs(algo) == [0, 0, 4]


@pytest.mark.parametrize("pilot_cfgs, message", [
    ({"extra_checkpoint_steps": [3_000]}, "not a whole epoch"),
    ({"extra_checkpoint_steps": [-2_000]}, "is negative"),
    ({"extra_checkpoint_steps": [4_000.5]}, "whole step counts"),
    ({"extra_checkpoint_steps": [True]}, "whole step counts"),
    ({"onset_step": 4_000.5}, "whole step counts"),
    ({"extra_checkpoint_steps": [22_000]}, "beyond the run"),
    ({"onset_step": 5_000}, "not a whole epoch"),
])
def test_an_extra_step_the_run_cannot_save_is_refused(pilot_cfgs, message) -> None:
    with pytest.raises(RunRefused, match=message):
        Algo(pilot_cfgs, epochs=10)


def test_a_plug_ins_extra_state_is_saved_with_the_full_state() -> None:
    algo = Algo({}, epochs=2, extra={"level_lagranges": "levels", "studyb_window": "window"})
    assert {"level_lagranges", "studyb_window", "pi", "reward_critic"} <= set(algo._logger.what)
    assert algo._logger.saves[1][1] == sorted(algo._logger.what)  # epoch-0.pt is rewritten with it
    with pytest.raises(RunRefused, match=r"reuses the mixin's keys \['pi'\]"):
        Algo({}, extra={"pi": "another actor"})


def _fake_hook(monkeypatch, calls: list) -> None:
    module = types.ModuleType("metrics.hook")

    def install_plasticity_hook(algo) -> None:
        calls.append((algo, list(algo._logger.saves), algo._logger.dump_tabular))

    module.install_plasticity_hook = install_plasticity_hook
    monkeypatch.setitem(sys.modules, "metrics.hook", module)


def test_the_plasticity_hook_is_installed_last_for_plasticity_runs(monkeypatch) -> None:
    calls: list = []
    _fake_hook(monkeypatch, calls)
    algo = Algo({"plasticity": True, "extra_checkpoint_steps": [4_000]})
    (called_with, saves_before, dump_at_hook) = calls[0]
    assert len(calls) == 1 and called_with is algo
    assert [epoch for epoch, _ in saves_before] == [0, 0]  # after the full-state epoch-0.pt, before training
    # after the mixin wrapped dump_tabular (_save_at_steps): a hook wrapping it wraps the mixin's wrapper
    assert not hasattr(dump_at_hook, "__func__") and dump_at_hook.__closure__
    for pilot_cfgs in ({"plasticity": False}, {}, None):
        Algo(pilot_cfgs)
    assert len(calls) == 1  # never for other runs


def test_a_missing_hook_module_is_not_hidden(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "metrics.hook", None)  # as if metrics/hook.py were not written yet
    with pytest.raises(ModuleNotFoundError) as exc:
        Algo({"plasticity": True})  # the launcher turns this into exit 4 (unavailable), not a crash
    assert algorithms.is_missing_repository_import(exc.value)
    Algo({"plasticity": False})  # the other runs do not need it


def test_a_hook_module_without_the_function_is_missing_code_too(monkeypatch) -> None:
    """metrics/hook.py written but without install_plasticity_hook: a plain ImportError naming the
    module (not ModuleNotFoundError), which the launcher treats the same way (exit 4, pilot/launch.py)."""
    monkeypatch.setitem(sys.modules, "metrics.hook", types.ModuleType("metrics.hook"))
    with pytest.raises(ImportError) as exc:
        Algo({"plasticity": True})
    assert not isinstance(exc.value, ModuleNotFoundError) and exc.value.name == "metrics.hook"
    assert algorithms.is_missing_repository_import(exc.value)

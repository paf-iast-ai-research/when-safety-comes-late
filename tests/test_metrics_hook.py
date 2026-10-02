"""The plasticity hook (Role 3): plasticity.csv rows at install and after every checkpoint save (contract 2).

Fast tests use a duck-typed algorithm (the attributes the hook reads from OmniSafe's): no training.
The ``slow`` tests train tiny PPO-Lagrangian runs built directly (2,000 steps per epoch, at most four
epochs; the hook is installed by the test, or by ``FullStateCheckpointMixin._init_log`` itself when
``pilot_cfgs.plasticity`` is set): rows at exactly the saved checkpoints, composition with the
mixin's extra saves, bit-for-bit training with and without the hook, an injection applied as the
Study A plug-in applies it (HANDOVER.md section 8, ``metrics.interventions``), and offline
recomputation equal to the logged rows. One more starts the real launcher as ``__main__`` with a
damaged fixed batch and expects a refusal (exit 5) that leaves the run directory as found.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from configs import registered as R
from metrics import batches, hook as H
from metrics.interventions import intervention_generator, plasticity_injection
from metrics.plasticity import (COLUMNS, actor_layers, dormant_fraction, effective_rank, hidden_activations, measure,
                                nan_row, normalise_frozen, parse_row)
from pilot.contracts import PLASTICITY_COLUMNS, PLASTICITY_FILE, plasticity_required
from pilot.errors import RunRefused

REPO = Path(__file__).resolve().parents[1]
TASK = "SafetyPointGoal1-v0"


# ---------------------------------------------------------------------------------------------
# A duck-typed algorithm
# ---------------------------------------------------------------------------------------------


class Cfg(dict):
    """Attribute access like OmniSafe's Config (a dict subclass)."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


class Logger:
    def __init__(self, log_dir: Path) -> None:
        self._log_dir, self._epoch, self.saved = str(log_dir), 0, []

    @property
    def log_dir(self) -> str:
        return self._log_dir

    @property
    def current_epoch(self) -> int:
        return self._epoch

    def torch_save(self) -> None:
        self.saved.append(self._epoch)

    def dump_tabular(self) -> None:
        self._epoch += 1


class Norm:
    def __init__(self, dim: int, count: int = 50) -> None:
        g = torch.Generator().manual_seed(11)
        self._count = torch.tensor(count)
        self._mean = torch.randn(dim, generator=g)
        self._std = torch.rand(dim, generator=g) + 0.5
        self._clip = 5.0 * torch.ones(dim)

    def normalize(self, data):  # pragma: no cover - must never be called
        raise AssertionError("the hook must not call Normalizer.normalize")


def _mlp(sizes: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for j in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[j], sizes[j + 1]), nn.Tanh() if j < len(sizes) - 2 else nn.Identity()]
    return nn.Sequential(*layers)


def fake_algo(tmp_path: Path, *, steps_per_epoch: int = 2000, obs_normalize: bool = True, obs_dim: int = 60):
    torch.manual_seed(0)
    actor = nn.Module()
    actor.mean = _mlp([obs_dim, *R.HIDDEN_SIZES, 2])
    actor.log_std = nn.Parameter(torch.zeros(2))
    actor._obs_dim = obs_dim
    critics = []
    for _ in range(2):
        critic = nn.Module()
        critic.add_module("critic_0", _mlp([obs_dim, *R.HIDDEN_SIZES, 1]))
        critics.append(critic)
    normalizer = Norm(obs_dim)
    env = SimpleNamespace(save=lambda: {"obs_normalizer": normalizer},
                          observation_space=SimpleNamespace(shape=(obs_dim,)))
    cfgs = Cfg(env_id=TASK, seed=0, algo_cfgs=Cfg(steps_per_epoch=steps_per_epoch, obs_normalize=obs_normalize),
               pilot_cfgs=Cfg(run_id="A-test-s0", onset_step=2 * steps_per_epoch))
    return SimpleNamespace(_cfgs=cfgs, _logger=Logger(tmp_path), _env=env, _normalizer=normalizer,
                           _actor_critic=SimpleNamespace(actor=actor, reward_critic=critics[0], cost_critic=critics[1],
                                                         actor_optimizer=torch.optim.Adam(actor.parameters())))


def _rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        assert tuple(reader.fieldnames) == COLUMNS
        return [parse_row(r) for r in reader]


def _epoch(algo, n: int = 1) -> None:
    for _ in range(n):
        algo._logger.dump_tabular()


# ---------------------------------------------------------------------------------------------
# Fast tests
# ---------------------------------------------------------------------------------------------


def test_columns_start_with_contract_two() -> None:
    assert COLUMNS[: len(PLASTICITY_COLUMNS)] == PLASTICITY_COLUMNS  # the ledger writer reads these


def test_install_writes_the_step_zero_row_and_the_meta_file(tmp_path) -> None:
    algo = fake_algo(tmp_path)
    recorder = H.install_plasticity_hook(algo)
    assert algo._plasticity_recorder is recorder and recorder.written_steps == [0]
    text = (tmp_path / PLASTICITY_FILE).read_text(encoding="utf-8")
    assert text.startswith(",".join(COLUMNS) + "\n") and "\r" not in text
    rows = _rows(tmp_path / PLASTICITY_FILE)
    assert [r["step"] for r in rows] == [0]
    x = normalise_frozen(torch.from_numpy(np.array(batches.load_fixed_batch(TASK))), algo._normalizer)
    assert rows[0] == measure({"actor": algo._actor_critic.actor, "reward_critic": algo._actor_critic.reward_critic,
                                "cost_critic": algo._actor_critic.cost_critic}, x, step=0)
    meta = json.loads((tmp_path / H.PLASTICITY_META_FILE).read_text(encoding="utf-8"))
    assert meta["task"] == TASK and meta["batch_file"] == f"metrics/fixed_batch/{TASK}.npy"
    assert meta["batch_sha256"] == batches.FIXED_BATCH_SHA256[TASK]
    assert meta["n_states"] == R.FIXED_BATCH_STATES and meta["obs_dim"] == 60
    assert meta["tau"] == R.DORMANT_THRESHOLD and meta["delta"] == R.EFFECTIVE_RANK_DELTA
    assert meta["columns"] == list(COLUMNS) and meta["errors"] == [] and meta["steps_per_epoch"] == 2000
    assert meta["run_id"] == "A-test-s0" and meta["onset_step"] == 4000 and "normalised" in meta["input"]


def test_rows_only_after_saves_at_the_runs_own_steps(tmp_path) -> None:
    algo = fake_algo(tmp_path, steps_per_epoch=3000)
    recorder = H.install_plasticity_hook(algo)
    logger = algo._logger
    _epoch(algo)  # epoch 1 without a save: no row (contract 2: no live rows)
    assert recorder.written_steps == [0]
    _epoch(algo)
    logger.torch_save()
    logger.torch_save()  # a second save of the same state: never a second row
    _epoch(algo, 3)
    logger.torch_save()
    assert logger.saved == [2, 2, 5]  # the original saver ran every time
    assert [r["step"] for r in _rows(tmp_path / PLASTICITY_FILE)] == [0, 6000, 15000] == recorder.written_steps


def test_step_zero_save_after_install_adds_no_row_and_wrappers_chain(tmp_path) -> None:
    algo = fake_algo(tmp_path)
    logger = algo._logger
    calls = []
    inner = logger.torch_save

    def earlier_wrapper():  # e.g. another component's wrapper installed before the hook
        calls.append(("earlier", logger.current_epoch))
        inner()

    logger.torch_save = earlier_wrapper
    H.install_plasticity_hook(algo)
    hooked = logger.torch_save

    def later_wrapper():  # installed after the hook: must still reach it
        calls.append(("later", logger.current_epoch))
        hooked()

    logger.torch_save = later_wrapper
    logger.torch_save()  # epoch 0 again (e.g. the mixin's full-state rewrite): no new row
    _epoch(algo)
    logger.torch_save()
    assert calls == [("later", 0), ("earlier", 0), ("later", 1), ("earlier", 1)]
    assert logger.saved == [0, 1]
    assert [r["step"] for r in _rows(tmp_path / PLASTICITY_FILE)] == [0, 2000]


def test_the_hook_changes_nothing_and_draws_no_random_number(tmp_path) -> None:
    algo = fake_algo(tmp_path)
    ac = algo._actor_critic
    modules = (ac.actor, ac.reward_critic, ac.cost_critic)
    states = [copy.deepcopy(m.state_dict()) for m in modules]
    norm_before = {k: v.clone() for k, v in vars(algo._normalizer).items()}
    rng = (torch.get_rng_state(), np.random.get_state()[1].copy(), random.getstate())
    grad_mode = torch.is_grad_enabled()
    H.install_plasticity_hook(algo)
    _epoch(algo)
    algo._logger.torch_save()
    assert torch.equal(rng[0], torch.get_rng_state()) and np.array_equal(rng[1], np.random.get_state()[1])
    assert rng[2] == random.getstate() and torch.is_grad_enabled() == grad_mode
    for m, s in zip(modules, states):
        assert all(torch.equal(v, m.state_dict()[k]) for k, v in s.items())
        assert all(not sub._forward_hooks for sub in m.modules())
    assert all(torch.equal(v, vars(algo._normalizer)[k]) for k, v in norm_before.items())


def test_rows_after_an_injection_measure_the_injected_actor(tmp_path) -> None:
    algo = fake_algo(tmp_path)
    H.install_plasticity_hook(algo)
    ac = algo._actor_critic
    plasticity_injection(ac.actor, ac.actor_optimizer, intervention_generator(0))  # as the Study A plug-in does
    _epoch(algo)
    algo._logger.torch_save()
    first, second = _rows(tmp_path / PLASTICITY_FILE)
    assert first["norm"] == first["norm_all"] and second["norm_all"] > second["norm"]
    # the post-injection row measures the injected layout (Q-reset-injection): 256 hidden units, rank of 192 features
    mean = ac.actor.mean
    layers = actor_layers(mean)
    assert layers.hidden == ("trunk.1", "frozen.3", "new.3", "new_frozen.3")
    x = normalise_frozen(torch.from_numpy(np.array(batches.load_fixed_batch(TASK))), algo._normalizer)
    acts = hidden_activations(mean, x, layers.hidden)
    assert sum(a.shape[1] for a in acts.values()) == 256
    assert second["dormant"] == dormant_fraction(acts[n] for n in layers.hidden)
    phi = torch.cat([acts[n] for n in layers.penultimate], dim=1)
    assert phi.shape[1] == 192 and second["rank"] == effective_rank(phi)


def test_a_measurement_failure_writes_nan_and_is_recorded_not_raised(tmp_path, monkeypatch) -> None:
    algo = fake_algo(tmp_path)
    recorder = H.install_plasticity_hook(algo)

    def broken(*args, **kwargs):
        raise FloatingPointError("synthetic failure")

    monkeypatch.setattr(H, "measure", broken)
    _epoch(algo)
    algo._logger.torch_save()  # must not raise inside learn() (Part 5.6)
    rows = _rows(tmp_path / PLASTICITY_FILE)
    assert rows[1]["step"] == 2000 and all(math.isnan(rows[1][c]) for c in COLUMNS[1:])
    meta = json.loads((tmp_path / H.PLASTICITY_META_FILE).read_text(encoding="utf-8"))
    assert meta["errors"] == ["step 2000: FloatingPointError: synthetic failure"] == recorder.errors


@pytest.mark.parametrize("what", ["row", "error_record"])
def test_a_failure_to_write_the_records_during_training_propagates(tmp_path, monkeypatch, what) -> None:
    """Error policy (metrics/hook.py docstring): only the metric computation is contained.

    An OSError writing the row, or the error record of a failed measurement, raises out of
    ``torch_save``: a run that completed has therefore finished every append, as
    ``pilot.contracts.validate_plasticity`` relies on.
    """
    algo = fake_algo(tmp_path)
    recorder = H.install_plasticity_hook(algo)

    def no_space(*args, **kwargs):
        raise OSError(28, "No space left on device")

    def broken(*args, **kwargs):
        raise FloatingPointError("synthetic failure")

    if what == "row":
        monkeypatch.setattr(H.PlasticityRecorder, "_append", no_space)
    else:
        monkeypatch.setattr(H, "measure", broken)
        monkeypatch.setattr(H, "_write_json_atomic", no_space)
    _epoch(algo)
    with pytest.raises(OSError, match="No space left on device"):
        algo._logger.torch_save()
    assert algo._logger.saved == [1]  # the checkpoint itself was saved before the row
    assert recorder.written_steps == [0]


def test_raw_input_when_the_run_does_not_normalise(tmp_path) -> None:
    algo = fake_algo(tmp_path, obs_normalize=False)
    H.install_plasticity_hook(algo)
    meta = json.loads((tmp_path / H.PLASTICITY_META_FILE).read_text(encoding="utf-8"))
    assert "raw" in meta["input"]
    assert _rows(tmp_path / PLASTICITY_FILE)[0] == measure({"actor": algo._actor_critic.actor,
                                                              "reward_critic": algo._actor_critic.reward_critic,
                                                              "cost_critic": algo._actor_critic.cost_critic},
                                                             batches.load_fixed_batch(TASK), step=0)


def test_supplied_test_batch(tmp_path) -> None:
    algo = fake_algo(tmp_path, obs_dim=4)
    batch = np.random.default_rng(0).normal(size=(32, 4))
    H.install_plasticity_hook(algo, batch=batch)
    meta = json.loads((tmp_path / H.PLASTICITY_META_FILE).read_text(encoding="utf-8"))
    assert meta["batch_file"] is None and meta["n_states"] == 32 and meta["obs_dim"] == 4


@pytest.mark.parametrize("problem", ["twice", "epoch", "width", "nan", "float32_overflow", "one_d", "exists",
                                     "unmeasurable", "missing_batch", "unknown_task", "unreadable_batch",
                                     "meta_unwritable", "row_unwritable"])
def test_install_refusals(tmp_path, monkeypatch, problem) -> None:
    """Every configuration, batch or file problem at install is a RunRefused (exit 5), never a crash.

    A plain exception from here would reach the launcher from inside the factory and be classified
    as a training failure (Part 5.6 exclusion) instead of a refusal; and a refused install leaves
    the run directory as it found it.
    """
    algo = fake_algo(tmp_path)
    kwargs = {}
    if problem == "twice":
        H.install_plasticity_hook(algo)
    elif problem == "epoch":
        _epoch(algo)
    elif problem == "width":
        kwargs["batch"] = np.zeros((8, 59), dtype=np.float32)
    elif problem == "nan":
        kwargs["batch"] = np.full((8, 60), np.nan, dtype=np.float32)
    elif problem == "float32_overflow":  # finite as float64, inf once cast to the actor's float32
        kwargs["batch"] = np.full((8, 60), 1e39, dtype=np.float64)
    elif problem == "one_d":
        kwargs["batch"] = np.zeros(60, dtype=np.float32)
    elif problem == "exists":
        (tmp_path / PLASTICITY_FILE).write_text("step,dormant,rank,norm\n", encoding="utf-8")
    elif problem == "unmeasurable":
        algo._actor_critic.actor.mean = nn.Linear(60, 2)  # not an OmniSafe MLP
    elif problem == "missing_batch":
        monkeypatch.setattr(batches, "FIXED_BATCH_DIR", tmp_path / "nowhere")
    elif problem == "unknown_task":
        algo._cfgs["env_id"] = "SafetyCarButton1-v0"
    elif problem == "unreadable_batch":  # read fails with IsADirectoryError (not FileNotFoundError)
        monkeypatch.setattr(batches, "FIXED_BATCH_DIR", tmp_path / "fixed_batch")
        (tmp_path / "fixed_batch" / f"{TASK}.npy").mkdir(parents=True)
    elif problem == "meta_unwritable":
        def no_space(path, payload):
            raise OSError(28, "No space left on device")
        monkeypatch.setattr(H, "_write_json_atomic", no_space)
    elif problem == "row_unwritable":  # both files exist when the first row fails: both must go
        def no_space(self, row):
            raise OSError(28, "No space left on device")
        monkeypatch.setattr(H.PlasticityRecorder, "_append", no_space)
    before = sorted(p.name for p in tmp_path.iterdir())
    with pytest.raises(RunRefused):
        H.install_plasticity_hook(algo, **kwargs)
    if problem != "twice":
        assert sorted(p.name for p in tmp_path.iterdir()) == before  # nothing written by a refused install
        assert algo._logger.torch_save.__func__ is Logger.torch_save  # no wrapper left behind
        assert getattr(algo, H.RECORDER_ATTRIBUTE, None) is None


def test_a_refused_start_removes_only_the_files_it_created(tmp_path, monkeypatch) -> None:
    """A plasticity.csv that appears between the check and the start is someone else's: kept; ours removed."""
    algo = fake_algo(tmp_path)
    write_meta = H._write_json_atomic

    def meta_then_foreign_csv(path, payload):
        write_meta(path, payload)
        (tmp_path / PLASTICITY_FILE).write_text("foreign\n", encoding="utf-8")

    monkeypatch.setattr(H, "_write_json_atomic", meta_then_foreign_csv)
    with pytest.raises(RunRefused, match="appeared"):
        H.install_plasticity_hook(algo)
    assert sorted(p.name for p in tmp_path.iterdir()) == [PLASTICITY_FILE]
    assert (tmp_path / PLASTICITY_FILE).read_text(encoding="utf-8") == "foreign\n"


def test_a_refused_start_keeps_a_meta_file_that_appeared(tmp_path) -> None:
    """A plasticity_meta.json that appears between the check and the start is kept, not replaced."""
    recorder = H.PlasticityRecorder(None, None, tmp_path, 2000, {"task": TASK})
    recorder.meta_path.write_text("foreign\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        recorder.start(nan_row(0))
    assert sorted(p.name for p in tmp_path.iterdir()) == [recorder.meta_path.name]
    assert recorder.meta_path.read_text(encoding="utf-8") == "foreign\n"


# ---------------------------------------------------------------------------------------------
# Tiny real training (slow)
# ---------------------------------------------------------------------------------------------

SPE = 2000  # steps per epoch of the tiny test runs (never the registered 20,000)


def _build(log_dir: Path, *, epochs: int, save_model_freq: int, pilot: dict):
    from omnisafe.algorithms.on_policy.naive_lagrange.ppo_lag import PPOLag
    from omnisafe.utils.config import Config, check_all_configs, get_default_kwargs_yaml
    from omnisafe.utils.tools import recursive_check_config

    from pilot.algorithms import FullStateCheckpointMixin

    cfgs = get_default_kwargs_yaml("PPOLag", TASK, "on-policy")
    custom = {"seed": 0,
              "train_cfgs": {"device": "cpu", "torch_threads": 1, "vector_env_nums": 1, "parallel": 1,
                             "total_steps": SPE * epochs},
              "algo_cfgs": {"steps_per_epoch": SPE},
              "logger_cfgs": {"save_model_freq": save_model_freq, "log_dir": str(log_dir), "use_tensorboard": False}}
    recursive_check_config(custom, cfgs)
    cfgs.recurisve_update(custom)  # (sic) OmniSafe's spelling
    cfgs.update({"exp_increment_cfgs": custom})
    cfgs.recurisve_update({"exp_name": f"PPOLag-{{{TASK}}}", "env_id": TASK, "algo": "PPOLag"})
    cfgs.train_cfgs.recurisve_update({"epochs": epochs})
    check_all_configs(cfgs, "on-policy")
    # With "plasticity": True the mixin's _init_log installs the real hook, as in production;
    # without it the test installs the hook itself.
    cfgs["pilot_cfgs"] = Config.dict2config(pilot)
    cls = type("FullStatePPOLag", (FullStateCheckpointMixin, PPOLag), {})
    return cls(env_id=TASK, cfgs=cfgs)


def _progress(omnisafe_dir: Path) -> list[dict]:
    with open(omnisafe_dir / "progress.csv", newline="", encoding="utf-8") as fh:
        return [{k: v for k, v in row.items() if not k.startswith("Time/")} for row in csv.DictReader(fh)]


def _equal_nested(a, b) -> bool:
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_equal_nested(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_equal_nested(x, y) for x, y in zip(a, b))
    return a == b


@pytest.fixture(scope="module")
def one_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(R.TORCH_THREADS)  # as the launcher (Table 3.1): recomputation is then exact
    yield
    torch.set_num_threads(threads)


def _train(root: Path, *, epochs: int, save_model_freq: int, pilot: dict, hook: bool,
           inject_at_epoch: int | None = None):
    """``hook``: install the hook by hand (after a counting wrapper); with ``pilot["plasticity"]`` the mixin has."""
    algo = _build(root, epochs=epochs, save_model_freq=save_model_freq, pilot=pilot)
    logger = algo._logger
    saves = []
    inner = logger.torch_save

    def counting_torch_save():  # installed before a hand-installed hook, after the mixin's: both must work
        saves.append(logger.current_epoch)
        inner()

    logger.torch_save = counting_torch_save
    recorder = H.install_plasticity_hook(algo) if hook else getattr(algo, H.RECORDER_ATTRIBUTE, None)
    records = []
    if inject_at_epoch is not None:  # stand-in for the Study A plug-in: at the start of the onset rollout
        rollout = algo._env.rollout

        def rollout_with_onset(*args, **kwargs):
            if logger.current_epoch == inject_at_epoch and not records:
                normalizer = algo._env.save()["obs_normalizer"]
                x = normalise_frozen(torch.from_numpy(np.array(batches.load_fixed_batch(TASK))), normalizer)
                ac = algo._actor_critic
                records.append(plasticity_injection(ac.actor, ac.actor_optimizer, intervention_generator(0), batch=x))
            return rollout(*args, **kwargs)

        algo._env.rollout = rollout_with_onset
    algo.learn()
    return SimpleNamespace(dir=Path(logger.log_dir), recorder=recorder, saves=saves, records=records)


@pytest.fixture(scope="module")
def run_hooked(tmp_path_factory, one_thread):
    return _train(tmp_path_factory.mktemp("hooked"), epochs=3, save_model_freq=1,
                  pilot={"run_id": "T-s0", "onset_step": 0}, hook=True)


@pytest.fixture(scope="module")
def run_plain(tmp_path_factory, one_thread):
    return _train(tmp_path_factory.mktemp("plain"), epochs=3, save_model_freq=1,
                  pilot={"run_id": "T-s0", "onset_step": 0}, hook=False)


@pytest.fixture(scope="module")
def run_injected(tmp_path_factory, one_thread):
    # onset after epoch 1 (off the save_model_freq 3 grid): the mixin saves epoch-1.pt from dump_tabular.
    # The hook is installed by FullStateCheckpointMixin._init_log (pilot_cfgs.plasticity), the production path.
    return _train(tmp_path_factory.mktemp("injected"), epochs=4, save_model_freq=3,
                  pilot={"run_id": "T-inj-s0", "onset_step": SPE, "extra_checkpoint_steps": [SPE], "plasticity": True},
                  hook=False, inject_at_epoch=1)


def _checkpoint_steps(omnisafe_dir: Path) -> list[int]:
    return sorted(int(p.stem.split("-")[1]) * SPE for p in (omnisafe_dir / "torch_save").glob("epoch-*.pt"))


@pytest.mark.omnisafe
@pytest.mark.slow
def test_rows_at_every_checkpoint_of_a_tiny_run(run_hooked) -> None:
    rows = _rows(run_hooked.dir / PLASTICITY_FILE)
    steps = [r["step"] for r in rows]
    assert steps == _checkpoint_steps(run_hooked.dir) == [0, 2000, 4000, 6000]
    assert len(set(steps)) == len(steps)
    assert all(0.0 <= r["dormant"] <= 1.0 and 1 <= r["rank"] <= 64 and r["norm"] > 0 for r in rows)
    assert all(math.isfinite(r[c]) for r in rows for c in COLUMNS)
    assert run_hooked.saves == [1, 2, 3]  # the wrapper installed before the hook saw every later save
    meta = json.loads((run_hooked.dir / H.PLASTICITY_META_FILE).read_text(encoding="utf-8"))
    assert meta["errors"] == []
    from pilot.contracts import validate_plasticity  # Role 1's check of contract 2

    spec = {"run_id": "T-s0", "study": "A", "plugin": "study_a"}
    validated = validate_plasticity(run_hooked.dir, spec, completed=True)
    assert sorted(validated) == steps and all(validated[r["step"]]["rank"] == r["rank"] for r in rows)


@pytest.mark.omnisafe
@pytest.mark.slow
def test_training_is_bit_for_bit_the_same_with_and_without_the_hook(run_hooked, run_plain) -> None:
    assert _progress(run_hooked.dir) == _progress(run_plain.dir)
    final = "torch_save/epoch-3.pt"
    a = torch.load(run_hooked.dir / final, weights_only=True, map_location="cpu")
    b = torch.load(run_plain.dir / final, weights_only=True, map_location="cpu")
    assert a.keys() == b.keys() and _equal_nested(a, b)
    assert not (run_plain.dir / PLASTICITY_FILE).exists()


@pytest.mark.omnisafe
@pytest.mark.slow
def test_mixin_extra_saves_compose_with_the_hook_and_the_injection(run_injected, run_hooked) -> None:
    assert isinstance(run_injected.recorder, H.PlasticityRecorder)  # installed by the mixin itself
    rows = {r["step"]: r for r in _rows(run_injected.dir / PLASTICITY_FILE)}
    assert sorted(rows) == run_injected.recorder.written_steps
    assert sorted(rows) == _checkpoint_steps(run_injected.dir) == [0, 2000, 6000, 8000]  # no row at 4000: no save
    assert run_injected.saves == [1, 3, 4]  # epoch 1 by the mixin (dump_tabular), 3 by the cadence, 4 at the end
    record = run_injected.records[0]
    assert record["output_identical"] is True and record["output_max_abs_change"] == 0.0
    # the onset row is pre-intervention: the state after one epoch equals the uninjected run's
    onset = rows[2000]
    assert onset == _rows(run_hooked.dir / PLASTICITY_FILE)[1]
    assert onset["norm"] == onset["norm_all"]
    for step in (6000, 8000):  # later rows measure the injected actor
        assert rows[step]["norm_all"] > rows[step]["norm"]
    onset_pi = torch.load(run_injected.dir / "torch_save/epoch-1.pt", weights_only=True)["pi"]
    final_pi = torch.load(run_injected.dir / "torch_save/epoch-4.pt", weights_only=True)["pi"]
    for layer in ("2", "4"):  # the frozen head kept the learned function
        for kind in ("weight", "bias"):
            assert torch.equal(final_pi[f"mean.frozen.{layer}.{kind}"], onset_pi[f"mean.{layer}.{kind}"])
    assert not torch.equal(final_pi["mean.new.2.weight"], final_pi["mean.new_frozen.2.weight"])  # the new copy trained
    assert not torch.equal(final_pi["mean.trunk.0.weight"], onset_pi["mean.0.weight"])  # Q-reset-injection: the trunk trains


@pytest.mark.omnisafe
@pytest.mark.slow
@pytest.mark.parametrize("which", ["run_hooked", "run_injected"])
def test_offline_recomputation_equals_the_logged_rows(which, request, tmp_path) -> None:
    from metrics import recompute

    run = request.getfixturevalue(which)
    logged = recompute.read_logged_rows(run.dir)
    recomputed = recompute.recompute_rows(run.dir)
    assert recompute.compare(recomputed, logged) == []
    out = tmp_path / "recomputed.csv"
    result = subprocess.run([sys.executable, "-m", "metrics.recompute", "--run-dir", str(run.dir), "--out", str(out)],
                            cwd=REPO, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "equal the logged plasticity.csv exactly" in result.stdout
    assert out.read_text(encoding="utf-8") == (run.dir / PLASTICITY_FILE).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------------------------
# A batch problem through the real launcher
# ---------------------------------------------------------------------------------------------

# Runs ``pilot.launch`` as ``__main__`` exactly as ``python -m pilot.launch`` does (the scheduler's way,
# pilot/scheduler.py), after two test-only substitutions in the child process: the fixed-batch
# directory (a corrupted copy) and the factory of plug-in "study_a" (envs.onset:make_algorithm, Role 2),
# replaced by the plain FullStateCheckpointMixin + PPO-Lagrangian factory so that the test exercises only
# the hook's refusal, independent of the Study A plug-in. The launcher sets pilot_cfgs.plasticity from the
# spec (contracts.plasticity_required), so the mixin's _init_log installs the real hook, which must
# refuse inside the factory: exit 5 (refused), the claim and OmniSafe's output removed.
_LAUNCH_AS_MAIN = """
import runpy, sys
from pathlib import Path
import metrics.batches
import pilot.algorithms
metrics.batches.FIXED_BATCH_DIR = Path(sys.argv[1])
pilot.algorithms.PLUGINS = {**pilot.algorithms.PLUGINS, "study_a": ("pilot.algorithms:make_ppolag", "test stand-in")}
sys.argv = ["pilot.launch", *sys.argv[2:]]
runpy.run_module("pilot.launch", run_name="__main__", alter_sys=True)
"""


@pytest.mark.omnisafe
@pytest.mark.slow
@pytest.mark.parametrize("damage", ["changed_byte", "unreadable"])
def test_a_batch_problem_makes_the_launcher_refuse_and_leave_the_run_as_found(tmp_path, damage) -> None:
    from pilot import launch, manifest

    spec = next(s for s in manifest.pilot() if s.study == "A" and s.plugin == "study_a" and s.onset_step == 0)
    spec = manifest.RunSpec.from_dict({**spec.to_dict(), "pending": []})  # no smoke flag needed for the gates
    assert plasticity_required(spec)
    run_dir, fixed = tmp_path / "run", tmp_path / "fixed_batch"
    run_dir.mkdir()
    fixed.mkdir()
    (run_dir / "spec.json").write_text(spec.to_json(), encoding="utf-8")
    for task in R.TASKS_STUDY_A:
        (fixed / f"{task}.npy").write_bytes(batches.batch_path(task).read_bytes())
    target = fixed / f"{spec.task}.npy"
    if damage == "changed_byte":
        data = bytearray(target.read_bytes())
        data[-1] ^= 0x01
        target.write_bytes(bytes(data))
    else:  # an OSError other than FileNotFoundError when the batch is read
        target.unlink()
        target.mkdir()
    before = sorted(p.relative_to(run_dir) for p in run_dir.rglob("*"))
    result = subprocess.run([sys.executable, "-c", _LAUNCH_AS_MAIN, str(fixed), "train", "--spec",
                             str(run_dir / "spec.json"), "--run-dir", str(run_dir), "--allow-dirty"],
                            cwd=REPO, capture_output=True, text=True, timeout=600)
    assert result.returncode == launch.EXIT_REFUSED, result.stdout + result.stderr
    reason = "differs from the committed record" if damage == "changed_byte" else "cannot be read"
    assert "REFUSED: the fixed batch of" in result.stderr and reason in result.stderr, result.stderr
    assert sorted(p.relative_to(run_dir) for p in run_dir.rglob("*")) == before == [Path("spec.json")]

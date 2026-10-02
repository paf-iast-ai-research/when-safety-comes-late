"""The full-state mixin and continuations with a tiny real PPO-Lagrangian (2,000 steps per epoch, at most
3 epochs per run).

* Extra checkpoints (``pilot.contracts.extra_checkpoint_steps``): a run saves one off the
  ``save_model_freq`` grid, with the plug-in's extra state, and trains exactly as a run without it.
* Part 3.4: the training-batch means ``Metrics/BatchEpCost`` and ``Metrics/BatchEpRet`` are each
  epoch's own episodes, beside OmniSafe's 50-episode windows, whose contents every checkpoint saves.
* Continuations: a continuation resolves its parent, restores the parent's learner in ``_init``
  (``pilot.dependencies.restore_learner``) and starts with fresh optimisers and a fresh schedule
  over its own epochs; ``epoch-0.pt`` holds the restored learner; the normaliser keeps its
  statistics and keeps updating.
* A parent that received plasticity injection is rebuilt through
  ``metrics.interventions.rebuild_injected`` and trains only the parameters the injected layout
  trains.

The algorithms are built directly (``launch.build_config`` shrunk to tiny epochs), never through
20,000-step epochs. Run with ``pytest -m slow``.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from configs import registered as R
from pilot import dependencies, launch, manifest, provenance, rundir
from pilot.errors import RunRefused
from pilot.manifest import RunSpec

pytestmark = [pytest.mark.omnisafe, pytest.mark.slow]

TINY = 2_000  # steps per epoch
COMMIT = "a" * 40


def _tiny(cfgs: Any, *, epochs: int, save_model_freq: int, extra: list[int] | None = None) -> Any:
    cfgs.algo_cfgs.steps_per_epoch = TINY
    cfgs.train_cfgs.total_steps = TINY * epochs
    cfgs.train_cfgs.epochs = epochs
    cfgs.logger_cfgs.save_model_freq = save_model_freq
    cfgs.logger_cfgs.use_tensorboard = False
    cfgs.pilot_cfgs.plasticity = False  # the Role 3 hook has its own tests (tests/test_metrics_*)
    if extra is not None:
        cfgs.pilot_cfgs.extra_checkpoint_steps = extra
    return cfgs


def _rows(omni: Path) -> list[dict[str, str]]:
    return [{k: v for k, v in r.items() if not k.startswith("Time/")} for r in rundir.read_progress(omni)]


def _finish(run_dir: Path, spec: RunSpec, cfgs: Any) -> None:
    """What the launcher writes after a completed run (the fields ``dependencies.resolve`` reads)."""
    (run_dir / "spec.json").write_text(spec.to_json())
    (run_dir / "train_result.json").write_text(json.dumps({
        "run_id": spec.run_id, "status": "completed", "commit_hash": COMMIT,
        "config_hash": provenance.config_hash(cfgs.todict()),
        "omnisafe_dir": str(rundir.find_omnisafe_run_dir(run_dir).relative_to(run_dir)),
    }))


@pytest.fixture(scope="module")
def parent_run(tmp_path_factory):
    """A 2-epoch parent with an extra checkpoint after epoch 1, and the same run without it."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("omnisafe")
    from omnisafe.algorithms.on_policy.naive_lagrange.ppo_lag import PPOLag

    from pilot.algorithms import FullStateCheckpointMixin, make_ppolag

    class ExtraStatePPOLag(FullStateCheckpointMixin, PPOLag):
        def _extra_checkpoint_state(self):
            return {"extra_marker": {"value": torch.tensor([7.0])}}

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        root = tmp_path_factory.mktemp("data") / "checkpoints"
        parent = next(s for s in manifest.design("main") if s.task == R.PRIMARY_TASK and s.N == 0.0 and s.seed == 0)
        run_dir = root / parent.run_id
        cfgs = _tiny(launch.build_config(parent, run_dir / "omnisafe"), epochs=2, save_model_freq=10, extra=[TINY])
        algo = ExtraStatePPOLag(env_id=parent.task, cfgs=cfgs)
        algo.learn()
        _finish(run_dir, parent, cfgs)
        plain_dir = tmp_path_factory.mktemp("plain")
        plain_cfgs = _tiny(launch.build_config(parent, plain_dir / "omnisafe"), epochs=2, save_model_freq=10, extra=[])
        make_ppolag(parent.task, plain_cfgs, parent).learn()
        yield SimpleNamespace(spec=parent, root=root, run_dir=run_dir, omni=rundir.find_omnisafe_run_dir(run_dir),
                              algo=algo, cfgs=cfgs, plain=rundir.find_omnisafe_run_dir(plain_dir))
    finally:
        torch.set_num_threads(threads)


def test_an_extra_checkpoint_is_saved_and_training_is_unchanged(parent_run) -> None:
    import torch

    saved = sorted(p.name for p in (parent_run.omni / "torch_save").glob("*.pt"))
    assert saved == ["epoch-0.pt", "epoch-1.pt", "epoch-2.pt"]  # epoch 1 is off the save_model_freq grid (10)
    assert sorted(p.name for p in (parent_run.plain / "torch_save").glob("*.pt")) == ["epoch-0.pt", "epoch-2.pt"]
    assert _rows(parent_run.omni) == _rows(parent_run.plain)  # every logged value, bit for bit
    ours = torch.load(parent_run.omni / "torch_save" / "epoch-2.pt", weights_only=True)
    theirs = torch.load(parent_run.plain / "torch_save" / "epoch-2.pt", weights_only=True)
    for key in ("pi", "reward_critic", "cost_critic", "obs_normalizer"):
        assert ours[key].keys() == theirs[key].keys()
        assert all(torch.equal(ours[key][k], theirs[key][k]) for k in theirs[key]), key
    extra = torch.load(parent_run.omni / "torch_save" / "epoch-1.pt", weights_only=True)
    assert set(dependencies.FULL_STATE_KEYS) <= set(extra) and torch.equal(extra["extra_marker"]["value"], torch.tensor([7.0]))
    assert extra["actor_scheduler"]["last_epoch"] == 1  # the state after one epoch
    logged = float(_rows(parent_run.omni)[0]["Metrics/LagrangeMultiplier"])
    assert float(extra["lagrange"]["value"]) == pytest.approx(logged, rel=1e-6)
    assert "extra_marker" not in theirs


def test_the_batch_metrics_are_each_epochs_episodes_and_the_windows_are_saved(parent_run) -> None:
    """Part 3.4 logs the "training-batch mean episodic cost and return"; OmniSafe's ``Metrics/EpCost``
    is the mean of its last 50 episodes, and each checkpoint saves those windows (``episode_windows``).

    SafetyPointGoal1-v0 episodes last 1,000 steps, so each 2,000-step epoch finishes two: after
    epoch 0 the window holds the batch, after epoch 1 all four episodes, the mean of the two batch
    means.
    """
    import torch

    rows = _rows(parent_run.omni)
    assert len(rows) == 2 and [float(r["Metrics/EpLen"]) for r in rows] == [TINY / 2, TINY / 2]
    for batch, window in (("Metrics/BatchEpCost", "Metrics/EpCost"), ("Metrics/BatchEpRet", "Metrics/EpRet")):
        b0, b1 = float(rows[0][batch]), float(rows[1][batch])
        assert b0 == pytest.approx(float(rows[0][window]), rel=1e-6)
        assert (b0 + b1) / 2 == pytest.approx(float(rows[1][window]), rel=1e-5)
    assert float(rows[1]["Metrics/BatchEpRet"]) != float(rows[1]["Metrics/EpRet"])  # a batch, not the window
    for epoch, episodes in ((1, 2), (2, 4)):  # the extra checkpoint after epoch 1, OmniSafe's final one
        windows = torch.load(parent_run.omni / "torch_save" / f"epoch-{epoch}.pt", weights_only=True)["episode_windows"]
        assert set(windows) == {"Metrics/EpRet", "Metrics/EpCost", "Metrics/EpLen"}
        for key, values in windows.items():
            assert values.dtype == torch.float64 and len(values) == episodes
            assert float(values.mean()) == pytest.approx(float(rows[epoch - 1][key]), rel=1e-5)


def _continuation(parent_spec: RunSpec, omni: Path, step: int) -> RunSpec:
    """The fine-tuning continuation of ``parent_spec`` from its tiny checkpoint at ``step``."""
    from test_core_dependencies import TINY_EPOCH, _continuation as continuation

    assert TINY_EPOCH == TINY
    return continuation(parent_spec, omni, step, commit=COMMIT)


def _child_class(spec: RunSpec, record: dict) -> type:
    import numpy as np
    import torch
    from omnisafe.algorithms.on_policy.naive_lagrange.ppo_lag import PPOLag

    from pilot.algorithms import FullStateCheckpointMixin

    class ContinuationPPOLag(FullStateCheckpointMixin, PPOLag):
        """What a continuation plug-in does in _init (pilot.dependencies.restore_learner), without its objective changes."""

        def _init(self) -> None:
            super()._init()
            dep = dependencies.from_cfgs(self._cfgs)[spec.params["parent_run_id"]]
            state = dependencies.load_full_state(dep.checkpoint(spec.params["parent_step"]))
            rng = (torch.get_rng_state(), np.random.get_state()[1].copy())
            record["summary"] = dependencies.restore_learner(self, state, spec=spec)
            record["rng_untouched"] = torch.equal(rng[0], torch.get_rng_state()) and bool(
                (rng[1] == np.random.get_state()[1]).all())
            with pytest.raises(RunRefused, match="already restored"):
                dependencies.restore_learner(self, state, spec=spec)
            record["state"] = state

    return ContinuationPPOLag


def _build_child(parent_run: SimpleNamespace, spec: RunSpec, epochs: int, record: dict) -> tuple[Any, Path, Any]:
    child_dir = parent_run.root / spec.run_id
    deps = dependencies.resolve(spec, child_dir)
    cfgs = _tiny(launch.build_config(spec, child_dir / "omnisafe", deps), epochs=epochs, save_model_freq=1)
    algo = _child_class(spec, record)(env_id=spec.task, cfgs=cfgs)
    return algo, child_dir, cfgs


def test_a_continuation_restores_the_parent_learner(parent_run) -> None:
    import torch

    spec = _continuation(parent_run.spec, parent_run.omni, step=TINY)
    record: dict = {}
    algo, child_dir, cfgs = _build_child(parent_run, spec, epochs=3, record=record)
    parent_state = torch.load(parent_run.omni / "torch_save" / "epoch-1.pt", weights_only=True)
    summary = record["summary"]
    assert summary["parent_run_id"] == parent_run.spec.run_id and summary["parent_step"] == TINY
    assert summary["parent_checkpoint_sha256"] == cfgs.pilot_cfgs.dependencies[parent_run.spec.run_id]["checkpoint_sha256"]
    assert summary["parent_checkpoint"] == spec.params["parent_checkpoint"]  # the ledger's spelling (portable)
    assert summary["restored"] == ["pi", "reward_critic", "cost_critic", "obs_normalizer"]
    assert summary["fresh"] == ["actor_optimizer", "actor_scheduler", "reward_critic_optimizer", "cost_critic_optimizer"]
    assert "extra_marker" in summary["not_restored"] and "lagrange" in summary["not_restored"]
    assert summary["injected_parent"] is False and summary["parameters"]["actor"] == summary["parameters"]["actor_trainable"]
    assert json.loads(json.dumps(summary)) == summary  # the plug-in writes it to continuation.json
    assert record["rng_untouched"]
    ac = algo._actor_critic
    assert list(ac.actor_optimizer.param_groups[0]["params"]) == list(ac.actor.parameters())
    assert all(p.requires_grad for p in ac.actor.parameters())
    # The first push after the restore continues the parent's statistics. A normaliser still taking
    # its "first" batch (OmniSafe's Normalizer._push while _first) would replace them: count 1, mean x.
    probe = copy.deepcopy(algo._env.save()["obs_normalizer"])  # a copy: the run's own normaliser is untouched
    parent_norm = parent_state["obs_normalizer"]
    n = int(parent_norm["_count"])
    assert n > 1 and torch.equal(probe._mean, parent_norm["_mean"]) and torch.equal(probe._sumsq, parent_norm["_sumsq"])
    probe.normalize(parent_norm["_mean"] + 1.0)  # one observation, exactly 1 above the parent's mean
    assert int(probe._count) == n + 1
    assert torch.allclose(probe._mean - parent_norm["_mean"], torch.full_like(probe._mean, 1.0 / (n + 1)), atol=1e-5)
    # epoch-0.pt of the continuation holds the restored learner, with fresh optimisers and schedule
    omni = rundir.find_omnisafe_run_dir(child_dir)
    start = torch.load(omni / "torch_save" / "epoch-0.pt", weights_only=True)
    for key in ("pi", "reward_critic", "cost_critic", "obs_normalizer"):
        assert start[key].keys() == parent_state[key].keys()
        assert all(torch.equal(start[key][k], parent_state[key][k]) for k in parent_state[key]), key
    for key in ("actor_optimizer", "reward_critic_optimizer", "cost_critic_optimizer"):
        assert start[key]["state"] == {} and parent_state[key]["state"] != {}, key
    assert start["actor_optimizer"]["param_groups"][0]["lr"] == cfgs.model_cfgs.actor.lr
    assert start["reward_critic_optimizer"]["param_groups"][0]["lr"] == cfgs.model_cfgs.critic.lr
    scheduler = start["actor_scheduler"]
    assert (scheduler["total_iters"], scheduler["last_epoch"], scheduler["base_lrs"]) == (3, 0, [cfgs.model_cfgs.actor.lr])
    assert (parent_state["actor_scheduler"]["total_iters"], parent_state["actor_scheduler"]["last_epoch"]) == (2, 1)
    # training continues from there: the learning rate decays over the continuation's 3 epochs, and
    # the normaliser keeps the parent's statistics and keeps adding to them
    count = int(parent_state["obs_normalizer"]["_count"])
    algo.learn()
    rows = _rows(omni)
    assert [float(r["Train/LR"]) for r in rows] == pytest.approx([cfgs.model_cfgs.actor.lr * f for f in (2 / 3, 1 / 3, 0.0)])
    end = torch.load(omni / "torch_save" / "epoch-3.pt", weights_only=True)
    assert int(end["obs_normalizer"]["_count"]) >= count + 3 * TINY
    assert not torch.equal(end["pi"]["mean.0.weight"], parent_state["pi"]["mean.0.weight"])


def test_a_continuation_of_an_injection_parent_trains_the_injected_layout(parent_run, tmp_path) -> None:
    import torch

    interventions = pytest.importorskip("metrics.interventions")
    # An injection parent: the parent's final learner with plasticity injection applied (Role 3's function,
    # as the Study A plug-in applies it at onset), saved as a full-state checkpoint of its own run.
    treated = next(s for s in manifest.design("treatment") if s.treatment == "injection" and s.task == R.PRIMARY_TASK
                   and s.seed == 0 and s.N == 0.5)
    run_dir = parent_run.root / treated.run_id
    omni = run_dir / parent_run.omni.relative_to(parent_run.run_dir)
    (omni / "torch_save").mkdir(parents=True)
    (omni / "config.json").write_bytes((parent_run.omni / "config.json").read_bytes())
    (omni / "progress.csv").write_bytes((parent_run.omni / "progress.csv").read_bytes())
    from omnisafe.models.actor.actor_builder import ActorBuilder

    model_cfgs = parent_run.cfgs.model_cfgs
    with torch.random.fork_rng():  # a fresh actor to hold the parent's weights, off every global stream
        actor = ActorBuilder(obs_space=parent_run.algo._env.observation_space, act_space=parent_run.algo._env.action_space,
                             hidden_sizes=model_cfgs.actor.hidden_sizes, activation=model_cfgs.actor.activation,
                             weight_initialization_mode=model_cfgs.weight_initialization_mode,
                             ).build_actor(actor_type=model_cfgs.actor_type)
    state = torch.load(parent_run.omni / "torch_save" / "epoch-2.pt", weights_only=True)
    actor.load_state_dict(state["pi"])
    optimizer = torch.optim.Adam(actor.parameters(), lr=model_cfgs.actor.lr)
    interventions.plasticity_injection(actor, optimizer, interventions.intervention_generator(treated.seed))
    state["pi"] = {k: v.detach().clone() for k, v in actor.state_dict().items()}
    torch.save(state, omni / "torch_save" / "epoch-2.pt")
    _finish(run_dir, treated, parent_run.cfgs)

    spec = _continuation(treated, omni, step=2 * TINY)
    record: dict = {}
    algo, child_dir, cfgs = _build_child(parent_run, spec, epochs=1, record=record)
    assert record["summary"]["injected_parent"] is True
    ac = algo._actor_critic
    assert isinstance(ac.actor.mean, interventions.InjectedHead)
    trainable = [n for n, p in ac.actor.named_parameters() if p.requires_grad]
    # registration order, as OmniSafe's Adam(actor.parameters()): the actor's own log_std, then actor.mean
    assert trainable == ["log_std", "mean.trunk.0.weight", "mean.trunk.0.bias", "mean.new.2.weight", "mean.new.2.bias",
                         "mean.new.4.weight", "mean.new.4.bias"]
    assert list(ac.actor_optimizer.param_groups[0]["params"]) == [p for p in ac.actor.parameters() if p.requires_grad]
    assert record["summary"]["parameters"]["actor_trainable"] == sum(p.numel() for p in ac.actor.parameters() if p.requires_grad)
    child_omni = rundir.find_omnisafe_run_dir(child_dir)
    start = torch.load(child_omni / "torch_save" / "epoch-0.pt", weights_only=True)
    assert start["pi"].keys() == state["pi"].keys()
    assert all(torch.equal(start["pi"][k], state["pi"][k]) for k in state["pi"])
    algo.learn()
    end = torch.load(child_omni / "torch_save" / "epoch-1.pt", weights_only=True)["pi"]
    for key, before in state["pi"].items():
        frozen = key.startswith(dependencies.INJECTED_FROZEN_PREFIXES)
        assert torch.equal(end[key], before) == frozen, key  # frozen parts never move; the rest trains

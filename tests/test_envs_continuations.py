"""Role 2's battery continuations ``envs.continuations`` without training (fast; the tiny runs are in
``tests/test_envs_continuations_slow.py``).

* The observation layouts of the Goal and Button tasks, read from Safety-Gymnasium, and the
  component-name mapping of transfer (Q-transfer-obs) for SafetyPointGoal1-v0 <-> SafetyPointButton1-v0
  and SafetyCarGoal1-v0 -> SafetyCarButton1-v0.
* ``map_state_for_transfer`` on real OmniSafe networks: target shapes, zero target-only columns, the
  mapped networks equal the parent's on shared inputs and ignore target-only inputs; the injected
  layout too.
* The normaliser mapping: shared statistics copied; a target-only dimension keeps variance 1 after
  OmniSafe's next push (envs/continuations.py).
* The continuation specs of ``pilot.manifest.battery_continuation`` pass; every other spec and every
  configuration problem is refused with ``RunRefused`` before anything is built.
"""

from __future__ import annotations

import copy
import json
import os
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from configs import registered as R
from pilot import dependencies, manifest
from pilot.errors import RunRefused
from pilot.manifest import RunSpec

pytestmark = pytest.mark.omnisafe

torch = pytest.importorskip("torch")
pytest.importorskip("omnisafe")

from envs import continuations as C  # noqa: E402  # after the skip: it imports OmniSafe
from envs.onset import SMOKE_PREFIX  # noqa: E402

E = R.STEPS_PER_EPOCH
COMMIT = "b" * 40
PARENT_STEP = 480 * E  # the parent's matched checkpoint step of these tests (epoch 480)
SENSORS = [("accelerometer", 3), ("velocimeter", 3), ("gyro", 3), ("magnetometer", 3)]
CAR = [("ballangvel_rear", 3), ("ballquat_rear", 9)]
# Safety-Gymnasium 0.4.1's obs_space_dict order (safety_gymnasium/bases/base_task.py:214-249; Q-transfer-obs)
LAYOUTS = {
    "SafetyPointGoal1-v0": SENSORS + [("goal_lidar", 16), ("hazards_lidar", 16), ("vases_lidar", 16)],
    "SafetyPointButton1-v0": SENSORS + [("buttons_lidar", 16), ("goal_lidar", 16), ("hazards_lidar", 16),
                                        ("gremlins_lidar", 16)],
    "SafetyCarGoal1-v0": SENSORS + CAR + [("goal_lidar", 16), ("hazards_lidar", 16), ("vases_lidar", 16)],
    "SafetyCarButton1-v0": SENSORS + CAR + [("buttons_lidar", 16), ("goal_lidar", 16), ("hazards_lidar", 16),
                                            ("gremlins_lidar", 16)],
}
PAIRS = [("SafetyPointGoal1-v0", "SafetyPointButton1-v0"), ("SafetyPointButton1-v0", "SafetyPointGoal1-v0"),
         ("SafetyCarGoal1-v0", "SafetyCarButton1-v0")]


# ---------------------------------------------------------------------------
# Layouts and the transfer map
# ---------------------------------------------------------------------------


def _same_numpy_state(a: tuple, b: tuple) -> bool:
    """Every field of two ``np.random.get_state()`` tuples: the key array, position and Gaussian cache."""
    return a[0] == b[0] and bool((a[1] == b[1]).all()) and tuple(a[2:]) == tuple(b[2:])


def test_the_observation_layouts_are_read_from_safety_gymnasium_without_touching_a_random_stream() -> None:
    C.observation_layout.cache_clear()
    random.seed(3)
    np.random.seed(3)
    torch.manual_seed(3)
    states = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    for task, layout in LAYOUTS.items():
        assert list(C.observation_layout(task)) == layout
    assert random.getstate() == states[0] and _same_numpy_state(np.random.get_state(), states[1])
    assert torch.equal(torch.get_rng_state(), states[2])
    assert [sum(s for _, s in LAYOUTS[t]) for t in LAYOUTS] == [60, 76, 72, 88]
    with pytest.raises(RunRefused):
        C.observation_layout("SafetyPointGoal9-v0")
    # a refusal restores the random states too
    assert random.getstate() == states[0] and _same_numpy_state(np.random.get_state(), states[1])
    assert torch.equal(torch.get_rng_state(), states[2])


@pytest.mark.parametrize("source,target", PAIRS)
def test_the_transfer_map_shares_the_sensors_and_the_goal_and_hazard_lidars(source, target) -> None:
    tmap = C.transfer_map(source, target)
    assert (tmap.source_dim, tmap.target_dim) == (sum(s for _, s in LAYOUTS[source]), sum(s for _, s in LAYOUTS[target]))
    car = [c for c, _ in CAR] if "Car" in source else []
    assert tmap.shared == tuple([c for c, _ in SENSORS] + car + ["goal_lidar", "hazards_lidar"])
    goal_only, button_only = ("vases_lidar",), ("buttons_lidar", "gremlins_lidar")
    assert (tmap.source_only, tmap.target_only) == ((goal_only, button_only) if "Goal" in source else (button_only, goal_only))
    pairs = tmap.column_pairs()
    shared_dim = sum(s for c, s in LAYOUTS[source] if c in tmap.shared)
    assert len(pairs) == shared_dim and len(set(t for t, _ in pairs)) == shared_dim
    assert sorted(tmap.target_only_columns() + tuple(t for t, _ in pairs)) == list(range(tmap.target_dim))
    assert sorted(tmap.source_only_columns() + tuple(s for _, s in pairs)) == list(range(tmap.source_dim))
    # by name, not by offset: goal_lidar sits at different offsets when Button inserts buttons_lidar first
    src, tgt = tmap._offsets(tmap.source_layout), tmap._offsets(tmap.target_layout)
    assert list(zip(tgt["goal_lidar"], src["goal_lidar"])) == [p for p in pairs if p[0] in tgt["goal_lidar"]]
    data = tmap.to_dict()
    assert json.loads(json.dumps(data)) == data and C.TransferMap.from_dict(data) == tmap
    as_tuples = {k: (tuple(tuple(x) if isinstance(x, list) else x for x in v) if isinstance(v, list) else v)
                 for k, v in data.items()}
    assert C.TransferMap.from_dict(as_tuples) == tmap  # tuples, at any depth, are read as JSON's lists


def test_a_transfer_map_that_does_not_fit_is_refused() -> None:
    tmap = C.transfer_map(*PAIRS[0])
    tampered = {**tmap.to_dict(), "shared": ["goal_lidar"]}
    with pytest.raises(RunRefused, match="derived fields"):
        C.TransferMap.from_dict(tampered)
    with pytest.raises(RunRefused, match="not a transfer map"):
        C.TransferMap.from_dict({"source_task": "x"})
    with pytest.raises(RunRefused, match="twice"):
        C.transfer_map("SafetyPointGoal1-v0", "SafetyPointGoal1-v0")
    with pytest.raises(RunRefused, match="different sizes"):
        C.TransferMap("a", "b", (("gyro", 3), ("goal_lidar", 16)), (("gyro", 3), ("goal_lidar", 8)))
    with pytest.raises(RunRefused, match="share no"):
        C.TransferMap("a", "b", (("gyro", 3),), (("goal_lidar", 16),))
    with pytest.raises(RunRefused, match="repeats"):
        C.TransferMap("a", "b", (("gyro", 3), ("gyro", 3)), (("gyro", 3),))
    for size in ("3", True, 3.0):
        with pytest.raises(RunRefused, match="not an integer"):
            C.TransferMap("a", "b", (("gyro", size),), (("gyro", 3),))
        data = tmap.to_dict()  # from_dict refuses it too: its sizes reach the check unconverted
        data["source_layout"][0][1] = size
        with pytest.raises(RunRefused, match="not an integer"):
            C.TransferMap.from_dict(data)


# ---------------------------------------------------------------------------
# Mapping the parent's networks and normaliser
# ---------------------------------------------------------------------------


def _model_cfgs() -> dict:
    return {"actor_type": C.ACTOR_TYPE, "weight_initialization_mode": "kaiming_uniform",
            "actor": {"hidden_sizes": list(R.HIDDEN_SIZES), "activation": R.ACTIVATION},
            "critic": {"hidden_sizes": list(R.HIDDEN_SIZES), "activation": R.ACTIVATION}}


def _networks(obs_dim: int, seed: int) -> tuple[Any, Any, Any]:
    """An OmniSafe actor and two critics for ``obs_dim`` inputs (built as training builds them)."""
    from gymnasium import spaces
    from omnisafe.models.actor.actor_builder import ActorBuilder
    from omnisafe.models.critic.critic_builder import CriticBuilder

    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    act_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        actor = ActorBuilder(obs_space, act_space, list(R.HIDDEN_SIZES), activation=R.ACTIVATION,
                             weight_initialization_mode="kaiming_uniform").build_actor(C.ACTOR_TYPE)
        critics = [CriticBuilder(obs_space, act_space, list(R.HIDDEN_SIZES), activation=R.ACTIVATION,
                                 weight_initialization_mode="kaiming_uniform", num_critics=1, use_obs_encoder=False,
                                 ).build_critic("v") for _ in range(2)]
    return actor, critics[0], critics[1]


def _normaliser_state(obs_dim: int, pushes: int = 3, seed: int = 0) -> dict:
    from omnisafe.common.normalizer import Normalizer

    norm = Normalizer((obs_dim,), clip=5)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(pushes):
        norm.normalize(torch.rand(40, obs_dim, generator=gen) * 3 + 1)
    return {k: v.clone() for k, v in norm.state_dict().items()}


def _parent_state(tmap: C.TransferMap) -> dict:
    actor, reward, cost = _networks(tmap.source_dim, seed=5)
    return {"pi": actor.state_dict(), "reward_critic": reward.state_dict(), "cost_critic": cost.state_dict(),
            "obs_normalizer": _normaliser_state(tmap.source_dim),
            "actor_optimizer": {"state": {}, "param_groups": []}, "lagrange": {"value": torch.tensor(0.2)}}


def _as_source(tmap: C.TransferMap, x_target: torch.Tensor) -> torch.Tensor:
    """The source observation with the target's shared components and source-only components at 0."""
    x = torch.zeros(x_target.shape[0], tmap.source_dim)
    for t, s in tmap.column_pairs():
        x[:, s] = x_target[:, t]
    return x


@pytest.mark.parametrize("source,target", PAIRS)
def test_the_mapped_networks_are_the_parents_on_shared_inputs_and_ignore_target_only_inputs(source, target) -> None:
    tmap = C.transfer_map(source, target)
    state = _parent_state(tmap)
    original = copy.deepcopy(state)
    mapped = C.map_state_for_transfer(state, tmap)
    for key in ("pi", "reward_critic", "cost_critic", "obs_normalizer"):  # the input is not modified
        assert state[key].keys() == original[key].keys()
        assert all(torch.equal(torch.as_tensor(state[key][k]), torch.as_tensor(original[key][k])) for k in original[key])
    assert mapped["actor_optimizer"] is state["actor_optimizer"] and mapped["lagrange"] is state["lagrange"]
    actor, reward, cost = _networks(tmap.target_dim, seed=9)
    actor.load_state_dict(mapped["pi"])  # strict: every shape fits the target's networks
    reward.load_state_dict(mapped["reward_critic"])
    cost.load_state_dict(mapped["cost_critic"])
    source_actor, source_reward, _ = _networks(tmap.source_dim, seed=9)
    source_actor.load_state_dict(state["pi"])
    source_reward.load_state_dict(state["reward_critic"])
    for key in ("pi", "reward_critic", "cost_critic"):
        first = C.ACTOR_FIRST_WEIGHTS[0] if key == "pi" else C.CRITIC_FIRST_WEIGHT
        weight = mapped[key][first]
        assert weight.shape == (R.HIDDEN_SIZES[0], tmap.target_dim)
        assert torch.equal(weight[:, list(tmap.target_only_columns())], torch.zeros(R.HIDDEN_SIZES[0], len(tmap.target_only_columns())))
        for t, s in tmap.column_pairs():
            assert torch.equal(weight[:, t], state[key][first][:, s])
        for name, tensor in state[key].items():
            if name != first:
                assert mapped[key][name] is tensor  # every other tensor unchanged
    gen = torch.Generator().manual_seed(2)
    x = torch.randn(32, tmap.target_dim, generator=gen)
    other = x.clone()
    other[:, list(tmap.target_only_columns())] = torch.randn(32, len(tmap.target_only_columns()), generator=gen) * 10
    with torch.no_grad():
        assert torch.equal(actor.mean(x), actor.mean(other))  # target-only inputs have no effect at the start
        assert torch.equal(_value(reward, x), _value(reward, other))
        assert torch.allclose(actor.mean(x), source_actor.mean(_as_source(tmap, x)), atol=1e-5)
        assert torch.allclose(_value(reward, x), _value(source_reward, _as_source(tmap, x)), atol=1e-5)
    assert torch.equal(actor.log_std, source_actor.log_std)


def _value(critic: Any, x: torch.Tensor) -> torch.Tensor:
    out = critic(x)
    return out[0] if isinstance(out, (list, tuple)) else out


def test_an_injected_parent_is_mapped_through_its_trunk() -> None:
    from metrics.interventions import build_actor, intervention_generator, plasticity_injection

    tmap = C.transfer_map(*PAIRS[0])
    actor, _, _ = _networks(tmap.source_dim, seed=5)
    optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
    plasticity_injection(actor, optimizer, intervention_generator(0))
    state = {"pi": {k: v.detach().clone() for k, v in actor.state_dict().items()},
             "reward_critic": _networks(tmap.source_dim, seed=6)[1].state_dict()}
    mapped = C.map_state_for_transfer(state, tmap)
    assert mapped["pi"]["mean.trunk.0.weight"].shape == (R.HIDDEN_SIZES[0], tmap.target_dim)
    rebuilt = build_actor(_model_cfgs(), tmap.target_dim, 2, mapped["pi"])
    x = torch.randn(8, tmap.target_dim, generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        assert torch.allclose(rebuilt.mean(x), actor.mean(_as_source(tmap, x)), atol=1e-5)


@pytest.mark.parametrize("source,target", PAIRS)
def test_target_only_normaliser_dimensions_keep_variance_one_after_the_next_push(source, target) -> None:
    """envs/continuations.py: ``_mean`` 0, ``_var`` = ``_std`` = 1 and ``_sumsq`` = ``_count`` - 1 for target-only
    dimensions, ``_count`` from the source: OmniSafe's next push recomputes the variance from ``_sumsq``."""
    from omnisafe.common.normalizer import Normalizer

    tmap = C.transfer_map(source, target)
    parent = _normaliser_state(tmap.source_dim)
    mapped = C.map_state_for_transfer({"pi": _networks(tmap.source_dim, 1)[0].state_dict(),
                                       "reward_critic": _networks(tmap.source_dim, 1)[1].state_dict(),
                                       "obs_normalizer": parent}, tmap)["obs_normalizer"]
    count = int(parent["_count"])
    only = list(tmap.target_only_columns())
    tcols, scols = [t for t, _ in tmap.column_pairs()], [s for _, s in tmap.column_pairs()]
    for key in ("_mean", "_var", "_std", "_sumsq", "_clip"):
        assert mapped[key].shape == (tmap.target_dim,) and torch.equal(mapped[key][tcols], parent[key][scols])
    assert torch.equal(mapped["_mean"][only], torch.zeros(len(only)))
    assert torch.equal(mapped["_var"][only], torch.ones(len(only))) and torch.equal(mapped["_std"][only], torch.ones(len(only)))
    assert torch.equal(mapped["_sumsq"][only], torch.full((len(only),), float(count - 1)))
    assert torch.equal(mapped["_clip"], torch.full((tmap.target_dim,), 5.0)) and int(mapped["_count"]) == count
    target_norm = Normalizer((tmap.target_dim,), clip=5)
    target_norm.load_state_dict(mapped)  # strict; OmniSafe's load sets _first False
    source_norm = Normalizer((tmap.source_dim,), clip=5)
    source_norm.load_state_dict(parent)
    obs = torch.rand(1, tmap.target_dim, generator=torch.Generator().manual_seed(7))  # raw lidar values lie in [0, 1]
    target_norm.normalize(obs)  # one push, as the first environment step of the transfer run
    source_norm.normalize(_as_source(tmap, obs))
    assert int(target_norm._count) == count + 1
    var = target_norm._var[only]
    assert torch.all((var - 1).abs() <= 2.0 / count) and torch.equal(target_norm._std[only], torch.sqrt(var))
    assert torch.allclose(target_norm._mean[tcols], source_norm._mean[scols])
    assert torch.allclose(target_norm._var[tcols], source_norm._var[scols])


def test_the_target_only_start_comes_from_the_q_transfer_obs_constants(monkeypatch) -> None:
    """HANDOVER.md section 10: Table 9.1's values are the named constants, not literals in the mapping. Other
    values put in the constants reach every target-only weight and statistic, and the next push keeps the
    variance the constant sets."""
    from omnisafe.common.normalizer import Normalizer

    assert (C.TARGET_ONLY_WEIGHT, C.TARGET_ONLY_MEAN, C.TARGET_ONLY_VARIANCE) == (0.0, 0.0, 1.0)  # Q-transfer-obs
    monkeypatch.setattr(C, "TARGET_ONLY_WEIGHT", 0.25)
    monkeypatch.setattr(C, "TARGET_ONLY_MEAN", 0.5)
    monkeypatch.setattr(C, "TARGET_ONLY_VARIANCE", 4.0)
    tmap = C.transfer_map(*PAIRS[0])
    mapped = C.map_state_for_transfer(_parent_state(tmap), tmap)
    only = list(tmap.target_only_columns())
    for key, first in (("pi", C.ACTOR_FIRST_WEIGHTS[0]), ("reward_critic", C.CRITIC_FIRST_WEIGHT),
                       ("cost_critic", C.CRITIC_FIRST_WEIGHT)):
        assert torch.all(mapped[key][first][:, only] == 0.25), key
    norm = mapped["obs_normalizer"]
    count = int(norm["_count"])
    assert torch.all(norm["_mean"][only] == 0.5) and torch.all(norm["_var"][only] == 4.0)
    assert torch.all(norm["_std"][only] == 2.0) and torch.all(norm["_sumsq"][only] == 4.0 * (count - 1))
    target_norm = Normalizer((tmap.target_dim,), clip=5)
    target_norm.load_state_dict(norm)
    target_norm.normalize(torch.rand(1, tmap.target_dim, generator=torch.Generator().manual_seed(1)))
    assert torch.all((target_norm._var[only] - 4.0).abs() <= 5.0 / count)  # |change| <= (4 + 1) / count


@pytest.mark.parametrize("damage", ["count one", "clip differs", "clip shape", "normaliser key", "normaliser shape",
                                    "normaliser not a mapping", "no pi", "actor not a mapping", "actor width",
                                    "two first layers", "critic width", "count nan", "count float",
                                    "count not numeric", "mean not numeric", "weight not numeric"])
def test_a_state_that_is_not_the_source_tasks_is_refused(damage) -> None:
    tmap = C.transfer_map(*PAIRS[0])
    state = _parent_state(tmap)
    if damage == "count one":
        state["obs_normalizer"]["_count"] = torch.tensor(1)
    elif damage == "clip differs":
        state["obs_normalizer"]["_clip"][3] = 4.0
    elif damage == "clip shape":
        state["obs_normalizer"]["_clip"] = torch.ones(0)
    elif damage == "normaliser key":
        del state["obs_normalizer"]["_sumsq"]
    elif damage == "normaliser shape":
        state["obs_normalizer"]["_var"] = torch.ones(tmap.target_dim)
    elif damage == "normaliser not a mapping":
        state["obs_normalizer"] = torch.zeros(3)
    elif damage == "no pi":
        del state["pi"]
    elif damage == "actor not a mapping":
        state["pi"] = torch.zeros(3)
    elif damage == "actor width":
        state["pi"] = _networks(tmap.target_dim, 1)[0].state_dict()
    elif damage == "two first layers":
        state["pi"]["mean.trunk.0.weight"] = state["pi"]["mean.0.weight"]
    elif damage == "critic width":
        state["cost_critic"] = _networks(tmap.target_dim, 1)[2].state_dict()
    elif damage == "count nan":
        state["obs_normalizer"]["_count"] = torch.tensor(float("nan"))
    elif damage == "count float":
        state["obs_normalizer"]["_count"] = torch.tensor(float("inf"))
    elif damage == "count not numeric":
        state["obs_normalizer"]["_count"] = "many"
    elif damage == "mean not numeric":
        state["obs_normalizer"]["_mean"] = "abc"
    elif damage == "weight not numeric":
        state["pi"]["mean.0.weight"] = "abc"
    with pytest.raises(RunRefused):
        C.map_state_for_transfer(state, tmap)


# ---------------------------------------------------------------------------
# Continuation specs
# ---------------------------------------------------------------------------


def _parents() -> list[RunSpec]:
    """Parents of every kind: N = 0, a late main arm, a treatment, a controller, the PID check, each task."""
    main = manifest.design("main")
    picks = [next(s for s in main if s.task == task and s.N == 0.0 and s.seed == 0) for task in R.TASKS_STUDY_A]
    picks.append(next(s for s in main if s.N == 0.5 and s.onset_shape == "ramp" and s.seed == 1))
    for group, field in (("treatment", "treatment"), ("controller", "controller_variant"), ("pid", "controller_variant")):
        picks.append(next(s for s in manifest.design(group) if getattr(s, field) and s.N == 0.5 and s.seed == 2))
    return picks


def _row(parent: RunSpec, step: int = PARENT_STEP, path: str = f"checkpoints/x/epoch-{PARENT_STEP // E}.pt") -> dict:
    return {"run_id": parent.run_id, "matched": True, "completed": True, "matched_checkpoint_step": step,
            "matched_checkpoint_path": path, "commit_hash": COMMIT}


@pytest.mark.parametrize("condition", C.CONDITIONS)
def test_every_battery_continuation_of_the_manifest_passes(condition) -> None:
    for parent in _parents():
        spec = manifest.battery_continuation(parent, _row(parent), condition)
        assert C.validate_continuation_spec(spec, condition, env_id=spec.task) is spec
        assert C.validate_continuation_spec(spec.to_dict(), condition) == spec
        smoke = replace(spec, run_id=SMOKE_PREFIX + spec.run_id, total_steps=2 * E)
        assert C.validate_continuation_spec(smoke, condition) == smoke  # a smoke copy's run is shrunk
        other = C.CONDITIONS[1 - C.CONDITIONS.index(condition)]
        with pytest.raises(RunRefused):
            C.validate_continuation_spec(spec, other)
        if condition == "transfer":
            assert spec.task == manifest.transfer_task(parent.task)
            assert spec.base_algo == "PPOLag"  # a PID parent's too (Table 9.1, Q-continuations)
        assert "lr_schedule_steps" not in spec.params  # its schedule spans its own registered length


def _refusals() -> list[tuple[str, str, RunSpec]]:
    parent = _parents()[0]
    fine = manifest.battery_continuation(parent, _row(parent), "finetune")
    trans = manifest.battery_continuation(parent, _row(parent), "transfer")

    def w(spec, **changes):
        data = spec.to_dict()
        params = changes.pop("params", None)
        if params is not None:
            data["params"] = {**spec.params, **params}
        data.update(changes)
        return RunSpec.from_dict(data)

    return [
        ("finetune", "total one epoch short", w(fine, total_steps=R.FINETUNE_STEPS - E)),
        ("finetune", "trained on another task", w(fine, task="SafetyPointButton1-v0")),
        ("finetune", "multiplier not frozen", w(fine, params={"multiplier": "fresh"})),
        ("finetune", "cost limit 30", w(fine, params={"cost_limit": 30.0})),
        ("finetune", "cost limit True", w(fine, params={"cost_limit": True})),
        ("finetune", "condition param", w(fine, params={"condition": "transfer"})),
        ("finetune", "onset", w(fine, onset_step=E)),
        ("finetune", "CPPOPID", w(fine, base_algo="CPPOPID")),
        ("finetune", "wrong plugin", w(fine, plugin="study_a")),
        ("finetune", "wrong group", w(fine, group="main")),
        ("finetune", "Study B", w(fine, study="B")),
        ("finetune", "no dependency", w(fine, depends_on=())),
        ("finetune", "other dependency", w(fine, depends_on=("A-PointGoal1-N0.00-s1",))),
        ("finetune", "parent step negative", w(fine, params={"parent_step": -E})),
        ("finetune", "parent step bool", w(fine, params={"parent_step": True})),
        ("finetune", "parent step missing", w(fine, params={"parent_step": None})),
        ("finetune", "source task outside Study A", w(fine, params={"source_task": "SafetyCarButton1-v0"})),
        ("finetune", "parent checkpoint missing", w(fine, params={"parent_checkpoint": None})),
        ("finetune", "parent checkpoint empty", w(fine, params={"parent_checkpoint": ""})),
        ("finetune", "parent commit missing", w(fine, params={"parent_commit": None})),
        ("transfer", "parent commit empty", w(trans, params={"parent_commit": ""})),
        ("transfer", "same task", w(trans, task=parent.task)),
        ("transfer", "another held-out task", w(trans, task="SafetyCarButton1-v0")),
        ("transfer", "fresh multiplier 0.01", w(trans, params={"fresh_multiplier_init": 0.01})),
        ("transfer", "fresh multiplier missing", w(trans, params={"fresh_multiplier_init": None})),
        ("transfer", "total", w(trans, total_steps=2 * R.TRANSFER_STEPS)),
        ("transfer", "a fine-tuning spec", fine),
        # only a few-shot continuation shortened by Part 6.1 cut 3 keeps a longer schedule (Q-continuations)
        ("finetune", "longer schedule", w(fine, params={"lr_schedule_steps": 2 * R.FINETUNE_STEPS})),
        ("transfer", "own schedule length named", w(trans, params={"lr_schedule_steps": R.TRANSFER_STEPS})),
    ]


REFUSALS = _refusals()


@pytest.mark.parametrize("condition,case,spec", REFUSALS, ids=[c for _, c, _ in REFUSALS])
def test_a_spec_that_is_not_a_battery_continuation_is_refused(condition, case, spec) -> None:
    with pytest.raises(RunRefused):
        C.validate_continuation_spec(spec, condition)


def test_a_wrong_environment_or_a_malformed_spec_is_refused() -> None:
    parent = _parents()[0]
    spec = manifest.battery_continuation(parent, _row(parent), "transfer")
    with pytest.raises(RunRefused, match="env_id"):
        C.validate_continuation_spec(spec, "transfer", env_id=parent.task)
    with pytest.raises(RunRefused):
        C.validate_continuation_spec("not a spec", "transfer")
    with pytest.raises(RunRefused, match="not a valid run spec"):  # never another exception
        C.validate_continuation_spec({**spec.to_dict(), "run_id": 5}, "transfer")


# ---------------------------------------------------------------------------
# The factories refuse configuration problems before building anything
# ---------------------------------------------------------------------------


def _fake_parent(root: Path, parent: RunSpec, epoch: int, state: dict) -> Path:
    run_dir = root / parent.run_id
    omni = run_dir / "omnisafe" / f"PPOLag-{{{parent.task}}}" / "seed-000-2026-09-30-00-00-00"
    (omni / "torch_save").mkdir(parents=True)
    (run_dir / "spec.json").write_text(parent.to_json(), encoding="utf-8")
    (omni / "config.json").write_text(json.dumps({"algo_cfgs": {"steps_per_epoch": E}}), encoding="utf-8")
    (omni / "progress.csv").write_text("Train/Epoch,TotalEnvSteps,Metrics/LagrangeMultiplier\n", encoding="utf-8")
    torch.save(state, omni / "torch_save" / f"epoch-{epoch}.pt")
    (run_dir / "train_result.json").write_text(json.dumps({
        "run_id": parent.run_id, "status": "completed", "commit_hash": COMMIT, "config_hash": "d" * 64,
        "omnisafe_dir": str(omni.relative_to(run_dir)),
    }), encoding="utf-8")
    return omni


def _registered_continuation(tmp_path: Path, condition: str, parent: RunSpec | None = None,
                             state: dict | None = None) -> tuple[RunSpec, Any]:
    """A registered continuation spec whose (fake) parent is resolved, and the launcher's configuration."""
    from pilot import launch
    from results.ledger_schema import relative_checkpoint_path

    parent = parent or _parents()[0]
    if state is None:
        state = {k: {"w": torch.zeros(1)} for k in dependencies.FULL_STATE_KEYS}
    epoch = PARENT_STEP // E
    omni = _fake_parent(tmp_path, parent, epoch, state)
    path = relative_checkpoint_path(os.path.abspath(omni / "torch_save" / f"epoch-{epoch}.pt"))
    spec = manifest.battery_continuation(parent, _row(parent, PARENT_STEP, path), condition)
    deps = dependencies.resolve(spec, tmp_path / spec.run_id)
    return spec, launch.build_config(spec, tmp_path / spec.run_id / "omnisafe", deps)


@pytest.mark.parametrize("condition", C.CONDITIONS)
@pytest.mark.parametrize("damage", ["env_id", "tiny epochs", "total", "seed", "use_cost", "penalty", "actor",
                                    "multiplier init", "multiplier lr", "cost limit", "no dependency record",
                                    "dependencies not a mapping", "another checkpoint step", "plasticity",
                                    "already built", "not a config", "plain dict", "infinite total",
                                    "infinite seed", "fractional seed", "fractional epoch", "parent spec",
                                    "pilot parent"])
def test_the_factory_refuses_every_configuration_problem_before_building(tmp_path, condition, damage) -> None:
    spec, cfgs = _registered_continuation(tmp_path, condition)
    factory = C.make_finetune_algorithm if condition == "finetune" else C.make_transfer_algorithm
    env_id = spec.task
    if damage == "env_id":
        env_id = "SafetyCarGoal1-v0"
    elif damage == "tiny epochs":
        cfgs.algo_cfgs.steps_per_epoch = 2_000
    elif damage == "total":
        cfgs.train_cfgs.epochs = 10
    elif damage == "seed":
        cfgs.seed = 3
    elif damage == "use_cost":
        cfgs.algo_cfgs.use_cost = False
    elif damage == "penalty":
        cfgs.algo_cfgs.penalty_coef = 0.2
    elif damage == "actor":
        cfgs.model_cfgs.actor_type = "mlp"
    elif damage == "multiplier init":
        cfgs.lagrange_cfgs.lagrangian_multiplier_init = 0.002
    elif damage == "multiplier lr":
        cfgs.lagrange_cfgs.lambda_lr = 0.05
    elif damage == "cost limit":
        cfgs.lagrange_cfgs.cost_limit = 20.0
    elif damage == "no dependency record":
        cfgs.pilot_cfgs.dependencies = {}
    elif damage == "dependencies not a mapping":
        cfgs.pilot_cfgs.dependencies = "resolved"
    elif damage == "another checkpoint step":
        next(iter(cfgs.pilot_cfgs.dependencies.values()))["checkpoint_step"] = 0
    elif damage == "plasticity":
        cfgs.pilot_cfgs.plasticity = True
    elif damage == "already built":
        cfgs["continuation_cfgs"] = {}
    elif damage == "not a config":
        cfgs = None
    elif damage == "plain dict":  # OmniSafe reads its Config by attribute: a dict is refused, not a crash
        cfgs = cfgs.todict()
    elif damage == "infinite total":
        cfgs.train_cfgs.total_steps = float("inf")
    elif damage == "infinite seed":
        cfgs.seed = float("inf")
    elif damage == "fractional seed":  # int() would truncate it to the spec's seed
        cfgs.seed = spec.seed + 0.5
    elif damage == "fractional epoch":
        cfgs.algo_cfgs.steps_per_epoch = E + 0.5
    elif damage == "parent spec":
        parent_dir = tmp_path / spec.params["parent_run_id"]
        other = next(s for s in manifest.design("main") if s.task == "SafetyCarGoal1-v0" and s.N == 0.0 and s.seed == 0)
        (parent_dir / "spec.json").write_text(replace(other, run_id=spec.params["parent_run_id"]).to_json(), encoding="utf-8")
    elif damage == "pilot parent":
        parent_dir = tmp_path / spec.params["parent_run_id"]
        parent = RunSpec.from_json((parent_dir / "spec.json").read_text(encoding="utf-8"))
        (parent_dir / "spec.json").write_text(replace(parent, group="pilot", pilot=True).to_json(), encoding="utf-8")
    with pytest.raises(RunRefused):
        factory(env_id, cfgs, spec)
    if cfgs is not None and damage != "already built":
        assert "continuation_cfgs" not in cfgs
    assert not (tmp_path / spec.run_id / "omnisafe").exists()  # nothing was built


@pytest.mark.parametrize("condition", C.CONDITIONS)
def test_a_parent_checkpoint_that_does_not_fit_the_networks_is_refused(tmp_path, condition) -> None:
    """The mapping (transfer, ``map_state_for_transfer``) or restore_learner's strict load (fine-tuning)
    refuses, naming the run; never a crash."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        spec, cfgs = _registered_continuation(tmp_path, condition)
        factory = C.make_finetune_algorithm if condition == "finetune" else C.make_transfer_algorithm
        reason = "does not fit" if condition == "finetune" else "first-layer weight"
        with pytest.raises(RunRefused, match=f"^{spec.run_id}: .*{reason}"):
            factory(spec.task, cfgs, spec)
        saved = cfgs.continuation_cfgs
        assert saved.condition == condition and saved.parent_step == PARENT_STEP
        assert (saved.transfer_map is None) == (condition == "finetune")
    finally:
        torch.set_num_threads(threads)


@pytest.mark.parametrize("condition", C.CONDITIONS)
@pytest.mark.parametrize("damage", ["no multiplier", "no transfer_map", "transfer_map of the other condition"])
def test_continuation_cfgs_without_the_factorys_settings_are_refused(condition, damage) -> None:
    """``_continuation_settings`` refuses continuation_cfgs the factory did not write; never a KeyError."""
    parent = _parents()[0]
    spec = manifest.battery_continuation(parent, _row(parent), condition)
    cls = C.FinetunePPOLag if condition == "finetune" else C.TransferPPOLag
    algo = object.__new__(cls)  # only the settings check: nothing is built
    algo._continuation_spec = spec
    settings = {"condition": condition, "parent_run_id": parent.run_id, "parent_step": PARENT_STEP,
                "source_task": parent.task, "target_task": spec.task, "multiplier": {"rule": "x"},
                "transfer_map": None if condition == "finetune" else C.transfer_map(parent.task, spec.task).to_dict()}
    algo._cfgs = {"continuation_cfgs": settings}
    assert algo._continuation_settings() == settings
    if damage == "no multiplier":
        del settings["multiplier"]
    elif damage == "no transfer_map":
        del settings["transfer_map"]
    else:
        settings["transfer_map"] = (C.transfer_map(parent.task, manifest.transfer_task(parent.task)).to_dict()
                                    if condition == "finetune" else None)
    with pytest.raises(RunRefused, match=f"^{spec.run_id}: continuation_cfgs"):
        algo._continuation_settings()

"""Partial reset and plasticity injection (Role 3): Table 2.4, equation (8), HANDOVER.md section 8.

Built on OmniSafe's own ``GaussianLearningActor`` with the pinned configuration (``omnisafe``
marker), an Adam optimiser and LinearLR schedule as ``ConstraintActorCritic`` builds them
(``models/actor_critic/actor_critic.py:91-113``), and one synthetic update so that Adam has state.
"""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pytest
import torch
from torch import nn

from configs import registered as R
from metrics import interventions as I
from metrics.batches import load_fixed_batch

pytestmark = pytest.mark.omnisafe

INJECTED_KEYS = [  # HANDOVER.md section 8: the checkpoint layout Role 2 relies on
    "log_std",
    "mean.trunk.0.weight", "mean.trunk.0.bias",
    "mean.frozen.2.weight", "mean.frozen.2.bias", "mean.frozen.4.weight", "mean.frozen.4.bias",
    "mean.new.2.weight", "mean.new.2.bias", "mean.new.4.weight", "mean.new.4.bias",
    "mean.new_frozen.2.weight", "mean.new_frozen.2.bias", "mean.new_frozen.4.weight", "mean.new_frozen.4.bias",
]
# Hand-computed for R.HIDDEN_SIZES (64, 64), the pinned configuration the tests build (60 inputs, 2 actions).
ACTOR_PARAMETERS = (60 * 64 + 64) + (64 * 64 + 64) + (64 * 2 + 2) + 2  # 8196: mean.0, mean.2, mean.4, log_std


def _model_cfgs():
    from omnisafe.utils.config import get_default_kwargs_yaml

    return get_default_kwargs_yaml("PPOLag", "SafetyPointGoal1-v0", "on-policy").model_cfgs


def _actor(seed: int = 0):
    from gymnasium import spaces
    from omnisafe.models.actor.actor_builder import ActorBuilder

    cfg = _model_cfgs()
    torch.manual_seed(seed)
    obs = spaces.Box(-np.inf, np.inf, (60,), np.float32)
    act = spaces.Box(-1.0, 1.0, (2,), np.float32)
    return ActorBuilder(obs, act, list(cfg.actor.hidden_sizes), activation=cfg.actor.activation,
                        weight_initialization_mode=cfg.weight_initialization_mode).build_actor(cfg.actor_type)


def _update(actor, optimizer, x: torch.Tensor, seed: int = 1) -> None:
    """One synthetic actor step (maximum likelihood of fixed actions), so that Adam has state.

    Draws only from its own generator.
    """
    g = torch.Generator().manual_seed(seed)
    actions = torch.rand(x.shape[0], 2, generator=g) * 2 - 1
    optimizer.zero_grad()
    loss = -actor(x).log_prob(actions).sum(-1).mean()
    loss.backward()
    optimizer.step()


def _trained():
    actor = _actor()
    optimizer = torch.optim.Adam(actor.parameters(), lr=_model_cfgs().actor.lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=10)
    x = torch.from_numpy(np.array(load_fixed_batch("SafetyPointGoal1-v0")))
    _update(actor, optimizer, x)
    scheduler.step()
    return actor, optimizer, scheduler, x


@pytest.fixture()
def trained():
    return _trained()


def _snapshot(module: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


# ---------------------------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------------------------


def test_intervention_seed_and_generator() -> None:
    expected = int.from_bytes(hashlib.sha256(b"wscl-intervention-3").digest()[:8], "little") & (2**63 - 1)
    assert I.intervention_seed(3) == expected == I.intervention_seed(np.int64(3))
    assert len({I.intervention_seed(s) for s in R.SEEDS}) == len(R.SEEDS)
    assert all(0 <= I.intervention_seed(s) < 2**63 for s in R.SEEDS)
    assert I.intervention_generator(3).initial_seed() == expected
    a = torch.rand(5, generator=I.intervention_generator(3))
    assert torch.equal(a, torch.rand(5, generator=I.intervention_generator(3)))
    for bad in (-1, True, 1.5, "0"):
        with pytest.raises(ValueError):
            I.intervention_seed(bad)


# ---------------------------------------------------------------------------------------------
# Partial reset
# ---------------------------------------------------------------------------------------------


def test_partial_reset_changes_only_the_last_two_layers_and_their_adam_state(trained) -> None:
    actor, optimizer, scheduler, x = trained
    before = _snapshot(actor)
    params = dict(actor.named_parameters())
    adam_before = {n: {k: v.clone() if torch.is_tensor(v) else v for k, v in optimizer.state[p].items()}
                   for n, p in params.items()}
    group_before = list(optimizer.param_groups[0]["params"])
    lr_before = optimizer.param_groups[0]["lr"]
    rng = torch.get_rng_state()
    record = I.partial_reset(actor, optimizer, I.intervention_generator(0), batch=x)
    assert torch.equal(rng, torch.get_rng_state())
    reset = {"mean.2.weight", "mean.2.bias", "mean.4.weight", "mean.4.bias"}
    after = _snapshot(actor)
    for name in before:
        assert torch.equal(before[name], after[name]) == (name not in reset), name
    for name, p in params.items():  # "the optimiser state for those parameters is cleared"
        if name in reset:
            assert p not in optimizer.state or not optimizer.state[p]
        else:
            assert all(torch.equal(v, adam_before[name][k]) if torch.is_tensor(v) else v == adam_before[name][k]
                       for k, v in optimizer.state[p].items())
    assert optimizer.param_groups[0]["params"] == group_before and optimizer.param_groups[0]["lr"] == lr_before
    bound = 1 / math.sqrt(64)
    for name in reset:
        assert after[name].abs().max() <= bound
    assert record["layers"] == ["mean.2", "mean.4"] and record["depth"] == R.INTERVENTION_LAYERS
    assert sorted(record["reinitialised"]) == sorted(reset) and record["adam_entries_cleared"] == 4
    assert record["generator_seed"] == I.intervention_seed(0) and record["global_rng_untouched"] is True
    assert record["parameter_count_before"] == record["parameter_count_after"] == ACTOR_PARAMETERS
    assert record["trainable_parameter_count_after"] == ACTOR_PARAMETERS and record["output_max_abs_change"] > 0
    json.dumps(record, allow_nan=False)
    _update(actor, optimizer, x, seed=2)  # Adam restarts the reset parameters' state
    assert int(optimizer.state[params["mean.2.weight"]]["step"]) == 1
    assert int(optimizer.state[params["mean.0.weight"]]["step"]) == 2


def test_partial_reset_draws_the_default_initialiser_from_the_generator(trained) -> None:
    actor, optimizer, _, _ = trained
    I.partial_reset(actor, optimizer, I.intervention_generator(4))
    g = I.intervention_generator(4)
    for index in (2, 4):  # layer by layer, weight then bias
        weight = torch.empty_like(actor.mean[index].weight)
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5), generator=g)
        bias = torch.empty_like(actor.mean[index].bias)
        nn.init.uniform_(bias, -1 / math.sqrt(64), 1 / math.sqrt(64), generator=g)
        assert torch.equal(actor.mean[index].weight, weight) and torch.equal(actor.mean[index].bias, bias)


def test_partial_reset_depth_one_is_the_output_layer_only(trained) -> None:
    actor, optimizer, _, _ = trained
    before = _snapshot(actor)
    record = I.partial_reset(actor, optimizer, I.intervention_generator(0), depth=1)
    after = _snapshot(actor)
    changed = {n for n in before if not torch.equal(before[n], after[n])}
    assert changed == {"mean.4.weight", "mean.4.bias"} and record["adam_entries_cleared"] == 2


def test_partial_reset_counts_only_the_adam_entries_that_existed(trained) -> None:
    actor, optimizer, _, _ = trained
    optimizer.state.pop(actor.mean[2].weight)  # one reset parameter without Adam state
    record = I.partial_reset(actor, optimizer, I.intervention_generator(0))
    assert len(record["reinitialised"]) == 4 and record["adam_entries_cleared"] == 3
    assert record["adam_entries_cleared_names"] == ["mean.2.bias", "mean.4.weight", "mean.4.bias"]
    assert all(p not in optimizer.state for layer in (actor.mean[2], actor.mean[4]) for p in layer.parameters())


# ---------------------------------------------------------------------------------------------
# Plasticity injection
# ---------------------------------------------------------------------------------------------


def test_injection_keeps_the_output_bit_identical(trained) -> None:
    actor, optimizer, _, x = trained
    randoms = torch.randn(256, 60, generator=torch.Generator().manual_seed(9)) * 4
    with torch.no_grad():
        before = [actor.mean(x), actor.mean(randoms)]
        dist = actor(x)
        mean_before, std_before = dist.mean.clone(), dist.stddev.clone()
    rng = torch.get_rng_state()
    record = I.plasticity_injection(actor, optimizer, I.intervention_generator(0), batch=x)
    assert torch.equal(rng, torch.get_rng_state())
    assert isinstance(actor.mean, I.InjectedHead)
    with torch.no_grad():
        assert torch.equal(actor.mean(x), before[0]) and torch.equal(actor.mean(randoms), before[1])
        dist = actor(x)
        assert torch.equal(dist.mean, mean_before) and torch.equal(dist.stddev, std_before)
        assert torch.equal(actor.predict(x, deterministic=True), before[0])
    assert list(actor.state_dict()) == INJECTED_KEYS
    assert record["output_max_abs_change"] == 0.0 and record["output_identical"] is True
    assert record["layers"] == ["mean.2", "mean.4"] and record["global_rng_untouched"] is True
    assert record["added_to_optimizer"] == ["mean.new.2.weight", "mean.new.2.bias",
                                            "mean.new.4.weight", "mean.new.4.bias"]
    assert record["trainable"] == ["log_std", "mean.trunk.0.weight", "mean.trunk.0.bias", *record["added_to_optimizer"]]
    assert record["trainable_parameter_count_before"] == record["trainable_parameter_count_after"] == ACTOR_PARAMETERS
    assert record["parameter_count_after"] == ACTOR_PARAMETERS + 2 * (64 * 64 + 64 + 2 * 64 + 2)
    # the reading of Table 9.1 (Q-reset-injection) and its source note, as intervention.json records them
    assert record["proposal_key"] == "Q-reset-injection"
    assert record["reading"] == (
        "Table 9.1 (Q-reset-injection): the trunk keeps training, as in Nikishin et al. (2023), "
        "arXiv:2305.15555, Section 3, per two independent search-engine extracts (paper PDF not opened from the "
        "sandbox); log_std keeps training too, a symmetric choice fixed before data that no source gives; the "
        "literal Table 2.4 reading ('only the new copy is trained thereafter') would freeze them too")
    json.dumps(record, allow_nan=False)


def test_injection_trains_only_trunk_new_head_and_log_std(trained) -> None:
    actor, optimizer, scheduler, x = trained
    I.plasticity_injection(actor, optimizer, I.intervention_generator(0))
    before = _snapshot(actor)
    head = actor.mean
    assert all(not p.requires_grad for p in (*head.frozen.parameters(), *head.new_frozen.parameters()))
    assert all(p.requires_grad for p in (*head.trunk.parameters(), *head.new.parameters(), actor.log_std))
    group = optimizer.param_groups[0]["params"]
    assert len(group) == 11 and group[-4:] == list(head.new.parameters())
    frozen_state = {id(p): optimizer.state[p]["exp_avg"].clone() for p in head.frozen.parameters()}
    _update(actor, optimizer, x, seed=3)
    grads = {n: p.grad is not None for n, p in actor.named_parameters()}
    trainable = {"log_std", "mean.trunk.0.weight", "mean.trunk.0.bias",
                 "mean.new.2.weight", "mean.new.2.bias", "mean.new.4.weight", "mean.new.4.bias"}
    assert {n for n, g in grads.items() if g} == trainable
    after = _snapshot(actor)
    for name in INJECTED_KEYS:
        assert torch.equal(before[name], after[name]) == (name not in trainable), name
    for p in head.frozen.parameters():  # frozen parameters keep their Adam state and are never updated
        assert torch.equal(optimizer.state[p]["exp_avg"], frozen_state[id(p)])
    lr = optimizer.param_groups[0]["lr"]
    scheduler.step()  # LinearLR decays the one group, new head included
    assert optimizer.param_groups[0]["lr"] < lr


def test_reset_and_injection_of_a_seed_draw_the_same_fresh_values(trained) -> None:
    actor, optimizer, _, _ = trained
    # A twin by the same deterministic construction: copy.deepcopy of an OmniSafe actor fails after a
    # forward pass with gradients (its _current_dist holds non-leaf tensors).
    twin, twin_optimizer, _, _ = _trained()
    assert all(torch.equal(a, b) for a, b in zip(actor.parameters(), twin.parameters()))
    I.partial_reset(actor, optimizer, I.intervention_generator(2))
    I.plasticity_injection(twin, twin_optimizer, I.intervention_generator(2))
    for index in ("2", "4"):
        for kind in ("weight", "bias"):
            fresh = getattr(actor.mean[int(index)], kind)
            assert torch.equal(getattr(getattr(twin.mean.new, index), kind), fresh)  # slices keep their keys
            assert torch.equal(getattr(getattr(twin.mean.new_frozen, index), kind), fresh)


@pytest.mark.parametrize("treatment", ["reset", "injection"])
@pytest.mark.parametrize("with_batch", [True, False])
def test_the_record_maps_onto_the_training_supplement_model(trained, treatment, with_batch) -> None:
    """results/supplement_schema.py: the ``training`` supplement's InterventionSummary (Role 4's model) accepts the summary."""
    from results.supplement_schema import InterventionSummary

    actor, optimizer, _, x = trained
    intervene = I.partial_reset if treatment == "reset" else I.plasticity_injection
    record = intervene(actor, optimizer, I.intervention_generator(1), batch=x if with_batch else None)
    record = json.loads(json.dumps(record, allow_nan=False))  # as read back from intervention.json
    onset = 5 * R.CHECKPOINT_INTERVAL_STEPS
    summary = I.intervention_summary(record, step=onset)
    assert tuple(summary) == I.SUMMARY_FIELDS == tuple(InterventionSummary.model_fields)
    model = InterventionSummary.model_validate(summary)
    assert model.treatment == treatment and model.step == onset and model.layers == ["mean.2", "mean.4"]
    assert model.generator_seed == I.intervention_seed(1)
    assert model.trainable_parameters_before == record["trainable_parameter_count_before"]
    assert model.trainable_parameters_after == record["trainable_parameter_count_after"]
    if not with_batch:
        assert model.max_output_difference is None
    elif treatment == "injection":
        assert model.max_output_difference == 0.0  # equation (8) at injection
    else:
        assert model.max_output_difference == record["output_max_abs_change"] > 0


def test_intervention_summary_refuses_what_is_not_an_intervention_record() -> None:
    good = {"intervention": "reset", "layers": ["mean.2"], "generator_seed": 1, "trainable_parameter_count_before": 3,
            "trainable_parameter_count_after": 3, "output_max_abs_change": None}
    assert I.intervention_summary(good, step=0)["max_output_difference"] is None
    for change in ({"intervention": "none"}, {"layers": []}, {"layers": [2]}, {"generator_seed": -1},
                   {"trainable_parameter_count_after": 1.5}, {"output_max_abs_change": float("nan")},
                   {"output_max_abs_change": -0.1}, {"output_max_abs_change": True}):
        with pytest.raises(ValueError):
            I.intervention_summary({**good, **change}, step=0)
    for step in (-1, True, 2.0):
        with pytest.raises(ValueError):
            I.intervention_summary(good, step=step)
    for record in (["x"], "reset", None):
        with pytest.raises(ValueError, match="not an intervention record"):
            I.intervention_summary(record, step=0)


def test_injection_of_the_last_layer_only(trained) -> None:
    actor, optimizer, _, x = trained
    with torch.no_grad():
        reference = actor.mean(x)
    I.plasticity_injection(actor, optimizer, I.intervention_generator(0), depth=1, batch=x)
    assert [k for k in actor.state_dict() if k.startswith("mean.")] == [
        "mean.trunk.0.weight", "mean.trunk.0.bias", "mean.trunk.2.weight", "mean.trunk.2.bias",
        "mean.frozen.4.weight", "mean.frozen.4.bias", "mean.new.4.weight", "mean.new.4.bias",
        "mean.new_frozen.4.weight", "mean.new_frozen.4.bias"]
    with torch.no_grad():
        assert torch.equal(actor.mean(x), reference)


def test_refusals_leave_the_actor_unchanged(trained, monkeypatch) -> None:
    actor, optimizer, _, x = trained
    before = _snapshot(actor)
    group = list(optimizer.param_groups[0]["params"])
    for depth in (0, 4, True):
        with pytest.raises(I.InterventionError):
            I.plasticity_injection(actor, optimizer, I.intervention_generator(0), depth=depth)
    with pytest.raises(I.InterventionError, match="no trunk"):
        I.plasticity_injection(actor, optimizer, I.intervention_generator(0), depth=3)
    with pytest.raises(I.InterventionError):
        I.partial_reset(actor, optimizer, 12345)  # a seed is not a dedicated generator
    with pytest.raises(I.InterventionError, match="optimiser"):
        I.partial_reset(actor, torch.optim.Adam([actor.log_std]), I.intervention_generator(0))
    with pytest.raises(I.InterventionError):
        I.plasticity_injection(actor, optimizer, I.intervention_generator(0), batch=np.zeros(5))
    used = I.intervention_generator(0)
    torch.rand(3, generator=used)  # its values are no longer those its seed names in the record
    for intervene in (I.partial_reset, I.plasticity_injection):
        with pytest.raises(I.InterventionError, match="already been drawn from"):
            intervene(actor, optimizer, used)
        # a refusal (RunRefused), never a crash (Part 5.6); 10**400 overflows float32
        for bad in (["a"] * 3, [[1.0] * 60, [1.0] * 59], [[10**400] * 60]):
            with pytest.raises(I.InterventionError, match="not a numeric array"):
                intervene(actor, optimizer, I.intervention_generator(0), batch=bad)
    # The output-check batch is checked before anything changes: inside learn() a torch shape error
    # would be a crash (Part 5.6), and a NaN would fail the identity check with a misleading message.
    nan_batch = x.clone()
    nan_batch[3, 7] = float("nan")
    for intervene in (I.partial_reset, I.plasticity_injection):
        with pytest.raises(I.InterventionError, match="observation components; the actor takes 60"):
            intervene(actor, optimizer, I.intervention_generator(0), batch=np.zeros((16, 59), dtype=np.float32))
        with pytest.raises(I.InterventionError, match="non-finite"):
            intervene(actor, optimizer, I.intervention_generator(0), batch=nan_batch)
        with pytest.raises(I.InterventionError, match="non-finite"):
            intervene(actor, optimizer, I.intervention_generator(0), batch=np.full((4, 60), np.inf))
    monkeypatch.setattr(actor, "_weight_initialization_mode", "orthogonal")
    with pytest.raises(I.InterventionError, match="weight_initialization_mode"):
        I.partial_reset(actor, optimizer, I.intervention_generator(0))
    monkeypatch.undo()
    monkeypatch.setattr(I.InjectedHead, "forward", lambda self, x: self.frozen(self.trunk(x)) + 1.0)
    with pytest.raises(I.InterventionError, match="unchanged at injection"):
        I.plasticity_injection(actor, optimizer, I.intervention_generator(0), batch=x)
    monkeypatch.undo()
    assert isinstance(actor.mean, nn.Sequential) and all(p.requires_grad for p in actor.parameters())
    assert optimizer.param_groups[0]["params"] == group
    assert all(torch.equal(v, actor.state_dict()[k]) for k, v in before.items())
    I.plasticity_injection(actor, optimizer, I.intervention_generator(0))
    with pytest.raises(I.InterventionError, match="already injected"):  # once per run (HANDOVER.md section 8)
        I.plasticity_injection(actor, optimizer, I.intervention_generator(0))
    with pytest.raises(I.InterventionError, match="already injected"):
        I.partial_reset(actor, optimizer, I.intervention_generator(0))


# ---------------------------------------------------------------------------------------------
# Loading: rebuild_injected and build_actor
# ---------------------------------------------------------------------------------------------


def _saved(tmp_path, actor) -> dict:
    path = tmp_path / "epoch-1.pt"
    torch.save({"pi": actor.state_dict()}, path)
    return torch.load(path, weights_only=True, map_location="cpu")["pi"]


@pytest.mark.parametrize("depth", [1, 2])
def test_rebuild_injected_is_rng_free_and_loads_strictly(trained, tmp_path, depth) -> None:
    actor, optimizer, _, x = trained
    assert not I.is_injected(actor.state_dict())
    I.plasticity_injection(actor, optimizer, I.intervention_generator(0), depth=depth)
    pi = _saved(tmp_path, actor)
    assert I.is_injected(pi)
    fresh = _actor(seed=7)
    rng = torch.get_rng_state()
    I.rebuild_injected(fresh, pi)
    assert torch.equal(rng, torch.get_rng_state())
    fresh.load_state_dict(pi, strict=True)
    assert [n for n, p in fresh.named_parameters() if p.requires_grad] == [
        n for n, p in actor.named_parameters() if p.requires_grad]
    with torch.no_grad():
        assert torch.equal(fresh.mean(x), actor.mean(x))
    with pytest.raises(I.InterventionError, match="not injected"):
        I.rebuild_injected(_actor(), _saved(tmp_path, _actor()))
    wrong = dict(pi)
    wrong["mean.new.4.weight"] = torch.zeros(3, 3)
    plain = _actor()
    with pytest.raises(I.InterventionError, match="shapes differ"):
        I.rebuild_injected(plain, wrong)
    assert isinstance(plain.mean, nn.Sequential)  # refused before any change


@pytest.mark.parametrize("injected", [False, 1, 2])  # plain, or injected at depth 1 or 2
@pytest.mark.parametrize("form", ["model_cfgs", "config"])
def test_build_actor_round_trips_both_layouts(trained, tmp_path, injected, form) -> None:
    actor, optimizer, _, x = trained
    if injected:
        I.plasticity_injection(actor, optimizer, I.intervention_generator(1), depth=injected)
        _update(actor, optimizer, x, seed=5)  # the heads diverge after injection
    pi = _saved(tmp_path, actor)
    cfg = _model_cfgs() if form == "model_cfgs" else json.loads(json.dumps({"model_cfgs": _model_cfgs().todict()}))
    rng = torch.get_rng_state()
    built = I.build_actor(cfg, 60, 2, pi)
    assert torch.equal(rng, torch.get_rng_state())
    assert not built.training and isinstance(built.mean, I.InjectedHead) == bool(injected)
    with torch.no_grad():
        assert torch.equal(built.mean(x), actor.mean(x)) and torch.equal(built.log_std, actor.log_std)
    with pytest.raises(I.InterventionError):
        I.build_actor(cfg, 61, 2, pi)


def test_build_actor_refuses_another_actor_type(trained) -> None:
    actor, _, _, _ = trained
    cfg = _model_cfgs().todict()
    cfg["actor_type"] = "mlp"
    with pytest.raises(I.InterventionError, match="actor_type"):
        I.build_actor(cfg, 60, 2, actor.state_dict())

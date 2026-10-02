"""Study B's budget conditioning (``studyb.conditioning``; Table 2.5; Table 3.1 "Networks"; equations (3) and (4)).

Pure helpers first (levels, the budget feature, the level of a feature, the budget draws), then the
OmniSafe-based classes built directly with a small configuration (``omnisafe`` marker; nothing here
trains: tests/test_studyb_training.py trains tiny runs under ``slow``).
"""

from __future__ import annotations

import copy
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from configs import registered as R
from envs.evaluation import append_budget
from pilot import launch, manifest
from pilot.errors import RunRefused
from pilot.ledger_writer import _level
from studyb import conditioning as C

TASK = R.TASKS_STUDY_B[0]


def _arm(arm: str, seed: int = 0) -> manifest.RunSpec:
    spec = next(s for s in manifest.study_b() if s.arm == arm and s.seed == 0)
    return spec if seed == 0 else spec.with_seed(seed)


def _fewshot(budget: float = 5.0) -> manifest.RunSpec:
    return next(s for s in manifest.study_b_fewshot([_arm("Moderate")]) if s.params["budget"] == budget)


def _cut_short(spec: manifest.RunSpec) -> manifest.RunSpec:
    """``spec`` as Part 6.1 cut 3 shortens it (pilot.manifest._cut_fewshot_short)."""
    (short,) = manifest.apply_cuts([spec], manifest.CUTS[:manifest.CUTS.index("fewshot_short") + 1])
    return short


# ---------------------------------------------------------------------------
# Levels, columns and the budget feature (Study B spec 6.1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arm", list(R.STUDY_B_ARMS))
def test_level_keys_and_columns_of_every_arm(arm) -> None:
    spec = _arm(arm)
    keys = C.level_keys(spec)
    if R.STUDY_B_ARMS[arm] is None:
        assert C.level_kind(spec) == C.CONTINUOUS
        assert keys == (10.0, 15.0, 20.0, 25.0, 30.0, 35.0)  # six 5-unit bins by lower edge (Q-continuous-bins)
    else:
        assert C.level_kind(spec) == C.DISCRETE
        assert keys == tuple(sorted(R.STUDY_B_ARMS[arm]))
    columns = [C.level_column(k) for k in keys]
    assert [_level(c) for c in columns] == list(keys)  # the ledger writer reads each level back (contract 3)
    assert all(c.startswith(launch.MULTIPLIER_COLUMN + "/level_") for c in columns)
    assert C.level_keys(spec.to_dict()) == keys  # the dict form is accepted too


def test_column_names_follow_contract_3() -> None:
    assert C.level_column(17.5) == "Metrics/LagrangeMultiplier/level_17.5"
    assert C.level_column(10.0) == "Metrics/LagrangeMultiplier/level_10"
    assert C.level_column(5.0) == "Metrics/LagrangeMultiplier/level_5"
    with pytest.raises(ValueError):
        C.level_column(12.34567891)  # would not survive the column name


def test_diagnostics_escape_the_launchers_multiplier_and_loss_checks() -> None:
    for key in (10.0, 17.5, 35.0, 45.0):
        for column in C.diagnostic_columns(key):
            assert not column.startswith(launch.MULTIPLIER_COLUMN) and not column.startswith(launch.LOSS_PREFIX)


def test_a_number_too_large_for_a_float_is_a_value_error() -> None:
    """``_real`` turns the OverflowError of float(10**400) into the ValueError the factory refuses on."""
    with pytest.raises(ValueError, match="finite number"):
        C._real(10**400, "params['budget']")
    assert C._real(45, "params['budget']") == 45.0


def test_fewshot_keys_are_its_unseen_budget_and_its_multiplier_starts_fresh() -> None:
    spec = _fewshot(45.0)
    assert C.level_kind(spec) == C.FEWSHOT and C.level_keys(spec) == (45.0,)
    with pytest.raises(ValueError):
        C.level_kind(next(s for s in manifest.pilot() if s.plugin == "unconstrained_ppo"))
    assert spec.params["fresh_multiplier_init"] == R.LAGRANGE_MULTIPLIER_INIT  # the factory's exact rule


def test_the_budget_feature_is_append_budgets_last_column_bit_for_bit() -> None:
    rng = np.random.default_rng(1)
    budgets = [*R.UNSEEN_BUDGETS, 10.0, 17.5, 25.0, 32.5, 40.0, *rng.uniform(10.0, 40.0, 200).tolist()]
    obs = torch.randn(3, 60)
    for b in budgets:
        feature = C.budget_feature(b)
        assert feature.dtype == torch.float32 and feature.shape == ()
        appended = append_budget(obs, b)
        assert appended.shape == (3, 61) and torch.equal(appended[:, :-1], obs)
        assert all(torch.equal(appended[i, -1], feature) for i in range(3))
        assert float(feature) == float(np.float32(b / R.BUDGET_OBSERVATION_DIVISOR))


def test_level_index_of_discrete_features() -> None:
    keys = tuple(R.STUDY_B_ARMS["Dense"])
    features = torch.stack([C.budget_feature(b) for b in keys])
    assert C.level_index(features, keys, C.DISCRETE).tolist() == [0, 1, 2, 3, 4]
    assert C.level_index(features.reshape(5, 1), keys, C.DISCRETE).tolist() == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError, match="not a level"):
        C.level_index(C.budget_feature(20.0), keys, C.DISCRETE)  # 20 is not a Dense level
    with pytest.raises(ValueError, match="not a level"):
        C.level_index(C.budget_feature(10.01), keys, C.DISCRETE)  # beyond LEVEL_TOLERANCE
    assert C.level_index(C.budget_feature(30.0), (30.0,), C.FEWSHOT).tolist() == [0]
    with pytest.raises(ValueError):
        C.level_index(C.budget_feature(15.0), (30.0,), C.FEWSHOT)
    with pytest.raises(ValueError, match="not finite"):
        C.level_index(torch.tensor([float("nan")]), keys, C.DISCRETE)
    assert C.level_index(torch.zeros(0), keys, C.DISCRETE).numel() == 0


def test_level_index_of_continuous_features_at_the_bin_edges() -> None:
    """Q-continuous-bins: [10, 15), ..., [30, 35), [35, 40], read from the float32 feature (Study B spec E5)."""
    edges = C.continuous_edges()
    just_above = float(np.nextafter(np.float32(0.15), np.float32(1))) * R.BUDGET_OBSERVATION_DIVISOR
    assert float(C.budget_feature(just_above)) > float(C.budget_feature(15.0))  # a feature above the edge's
    budgets = [10.0, 14.999999, 15.0, just_above, 19.99, 20.0, 25.0, 29.999, 30.0, 34.9, 35.0, 39.999, 40.0]
    expected = [0, 0, 1, 1, 1, 2, 3, 3, 4, 4, 5, 5, 5]
    assert float(C.budget_feature(35.0)) * R.BUDGET_OBSERVATION_DIVISOR < 35.0  # float32 rounds this edge down
    features = torch.stack([C.budget_feature(b) for b in budgets])
    assert C.level_index(features, edges, C.CONTINUOUS).tolist() == expected
    for bad in (9.9, 40.1, 5.0, 45.0):
        with pytest.raises(ValueError, match="outside the Continuous range"):
            C.level_index(C.budget_feature(bad), edges, C.CONTINUOUS)
    with pytest.raises(ValueError, match="bin edges"):
        C.level_index(C.budget_feature(12.0), (10.0, 20.0), C.CONTINUOUS)
    with pytest.raises(ValueError, match="bin edges"):  # refused for empty features too
        C.level_index(torch.zeros(0), (10.0, 20.0), C.CONTINUOUS)


def test_attribution_and_the_stored_feature_meet_the_same_multiplier() -> None:
    """A finished episode's J_C level (from its budget) and its samples' penalty level (from the stored
    observation) come from one function of one float32 feature, even at a bin edge."""
    rng = np.random.default_rng(7)
    budgets = [*rng.uniform(10.0, 40.0, 500).tolist(), 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
    stored = torch.stack([append_budget(torch.zeros(1, 60), b)[0, -1] for b in budgets])
    from_budget = torch.stack([C.budget_feature(b) for b in budgets])
    edges = C.continuous_edges()
    assert torch.equal(C.level_index(stored, edges, C.CONTINUOUS), C.level_index(from_budget, edges, C.CONTINUOUS))


# ---------------------------------------------------------------------------
# Budget draws (Study B spec 6.2): a generator of their own
# ---------------------------------------------------------------------------


def _draws(source, n: int = 200) -> list[float]:
    return [source() for _ in range(n)]


def test_budget_draws_are_reproducible_and_leave_the_global_streams_alone() -> None:
    levels = R.STUDY_B_ARMS["Moderate"]
    np.random.seed(123)
    torch.manual_seed(123)
    np.random.random()  # the key array is regenerated: a later draw moves only the position
    global_numpy, global_torch = np.random.get_state(), torch.get_rng_state().clone()
    first = _draws(C.make_budget_source(C.DISCRETE, levels, 3))
    after = np.random.get_state()
    assert after[0] == global_numpy[0] and (after[1] == global_numpy[1]).all() and after[2:] == global_numpy[2:]
    assert torch.equal(torch.get_rng_state(), global_torch)
    np.random.seed(999)  # another global state gives the same draws
    torch.manual_seed(999)
    assert _draws(C.make_budget_source(C.DISCRETE, levels, 3)) == first
    assert _draws(C.make_budget_source(C.DISCRETE, levels, 4)) != first  # the run seed matters
    assert set(first) == set(levels)  # uniform over the training set: every level is drawn
    counts = [first.count(b) for b in levels]
    assert min(counts) > len(first) / len(levels) / 2


def test_continuous_draws_cover_the_range_and_the_fixed_budget_is_fixed() -> None:
    draws = _draws(C.make_budget_source(C.CONTINUOUS, C.continuous_edges(), 0), 2000)
    assert min(draws) >= 10.0 and max(draws) < 40.0
    bins = C.level_index(torch.stack([C.budget_feature(b) for b in draws]), C.continuous_edges(), C.CONTINUOUS)
    assert sorted(set(bins.tolist())) == [0, 1, 2, 3, 4, 5]  # every bin is drawn
    assert draws == _draws(C.make_budget_source(C.CONTINUOUS, C.continuous_edges(), 0), 2000)
    assert _draws(C.make_budget_source(C.FEWSHOT, (15.0,), 0), 5) == [15.0] * 5
    assert C.budget_generator(0).integers(1 << 30) == np.random.default_rng([0, C.BUDGET_STREAM]).integers(1 << 30)
    with pytest.raises(ValueError):
        C.make_budget_source(C.FEWSHOT, (15.0, 30.0), 0)
    with pytest.raises(ValueError):
        C.budget_generator(-1)


# ---------------------------------------------------------------------------
# The OmniSafe-based classes, built without training
# ---------------------------------------------------------------------------


def _cfgs(spec, log_dir: Path, *, seed: int | None = None):
    cfgs = launch.build_config(spec, log_dir)
    cfgs.algo_cfgs.steps_per_epoch = 2_000
    cfgs.train_cfgs.total_steps = 4_000
    cfgs.train_cfgs.epochs = 2
    cfgs.logger_cfgs.use_tensorboard = False
    if seed is not None:
        cfgs.seed = seed
    return cfgs


def _build(spec, tmp_path: Path):
    threads = torch.get_num_threads()
    torch.set_num_threads(R.TORCH_THREADS)
    try:
        return C.make_algorithm(spec.task, _cfgs(spec, tmp_path / spec.run_id / "omnisafe"), spec)
    finally:
        torch.set_num_threads(threads)


@pytest.fixture
def moderate(tmp_path):
    return _build(_arm("Moderate"), tmp_path)


@pytest.mark.omnisafe
def test_the_classes_are_the_mixin_over_omnisafes_ppo() -> None:
    from omnisafe.adapter.onpolicy_adapter import OnPolicyAdapter
    from omnisafe.algorithms.on_policy.base.ppo import PPO
    from omnisafe.algorithms.on_policy.naive_lagrange.ppo_lag import PPOLag

    from pilot.algorithms import FullStateCheckpointMixin

    algo = C.BudgetConditionedPPOLag
    assert issubclass(algo, FullStateCheckpointMixin) and issubclass(algo, PPO) and not issubclass(algo, PPOLag)
    assert issubclass(C.FewShotPPOLag, algo) and issubclass(C.BudgetConditionedAdapter, OnPolicyAdapter)
    assert algo.__module__ == "studyb.conditioning" and algo.__name__ == "BudgetConditionedPPOLag"
    with pytest.raises(AttributeError):
        C.NoSuchClass  # noqa: B018


@pytest.mark.omnisafe
def test_the_budget_widens_the_networks_and_the_buffer_but_not_the_normaliser(moderate) -> None:
    """Table 3.1 "Networks": one more input for the actor and both critics; Q-budget-normalisation: the
    normaliser keeps the task's 60 features."""
    ac = moderate._actor_critic
    assert moderate._env.observation_space.shape == (61,)
    assert ac.actor.mean[0].in_features == 61
    assert ac.reward_critic.net_lst[0][0].in_features == 61 and ac.cost_critic.net_lst[0][0].in_features == 61
    assert moderate._buf.buffers[0].data["obs"].shape[1] == 61
    assert tuple(moderate._env.save()["obs_normalizer"].mean.shape) == (60,)
    assert moderate._cfgs.studyb_cfgs.kind == C.DISCRETE and list(moderate._cfgs.studyb_cfgs.level_keys) == [10.0, 20.0, 40.0]
    assert not hasattr(moderate, "_lagrange") and list(moderate._level_lagranges) == [10.0, 20.0, 40.0]
    # Table 3.1: lr 0.035 from 0.001; Adam (OmniSafe's Lagrange, Q-multiplier-adam); d = the level
    for key, lag in moderate._level_lagranges.items():
        assert lag.cost_limit == key and lag.lambda_lr == R.LAGRANGE_MULTIPLIER_LR
        assert lag.lagrangian_multiplier.item() == pytest.approx(R.LAGRANGE_MULTIPLIER_INIT)
        assert type(lag.lambda_optimizer).__name__ == "Adam"


@pytest.mark.omnisafe
def test_seed_k_builds_the_same_networks_in_every_arm(tmp_path) -> None:
    """Part 5.1: "seed k of one arm and seed k of the other share their network initialisation"; the
    budget draws come from their own generator, so no arm consumes a global draw before the model."""
    states = {}
    for arm in ("Single-10", "Moderate", "Dense", "Continuous"):
        algo = _build(_arm(arm, seed=3), tmp_path)
        states[arm] = {k: v.clone() for k, v in algo._actor_critic.state_dict().items()}
    reference = states.pop("Single-10")
    for arm, state in states.items():
        assert state.keys() == reference.keys()
        assert all(torch.equal(state[k], reference[k]) for k in reference), arm
    other = _build(_arm("Single-10", seed=4), tmp_path)
    assert not all(torch.equal(other._actor_critic.state_dict()[k], reference[k]) for k in reference)


@pytest.mark.omnisafe
def test_reset_and_step_append_the_budget_after_the_wrapper_chain(tmp_path) -> None:
    """The observation without its last feature is exactly the plain adapter's (same seed, same
    actions), whatever the normaliser's running statistics; the last feature is float32(b / 100)."""
    from omnisafe.adapter.onpolicy_adapter import OnPolicyAdapter

    spec = _arm("Continuous", seed=2)
    algo = _build(spec, tmp_path)
    adapter = algo._env
    plain = OnPolicyAdapter(TASK, 1, 2, _cfgs(spec, tmp_path / "plain"))
    obs, _ = adapter.reset()
    plain_obs, _ = plain.reset()
    budget = adapter.episode_budget
    assert 10.0 <= budget < 40.0 and adapter.budget_draws == 1
    assert torch.equal(obs[:, :-1], plain_obs) and torch.equal(obs[:, -1], C.budget_feature(budget).reshape(1))
    actions = torch.from_numpy(np.random.default_rng(0).uniform(-1, 1, (40, 1, 2)).astype(np.float32))
    for action in actions:
        obs, *_ = adapter.step(action)
        plain_obs, *_ = plain.step(action)
        assert torch.equal(obs[:, :-1], plain_obs) and float(obs[0, -1]) == float(C.budget_feature(budget))
    assert adapter.budget_draws == 1  # one episode, one draw
    plain.close()


@pytest.mark.omnisafe
def test_an_episode_that_takes_no_step_passes_its_draw_on(moderate) -> None:
    adapter = moderate._env
    adapter.reset()
    first = adapter.episode_budget
    adapter.reset()  # the episode begun above took no step: its budget is not used up
    assert adapter.episode_budget == first and adapter.budget_draws == 1
    adapter.step(torch.zeros(1, 2))
    adapter.reset()
    assert adapter.budget_draws == 2


@pytest.mark.omnisafe
def test_the_final_observation_carries_the_finished_episodes_budget(tmp_path) -> None:
    """At an episode's end AutoReset returns the next episode's first observation and the finished one
    in info['final_observation'] (the time-out bootstrap); each carries its own episode's budget."""
    algo = _build(_arm("Continuous", seed=5), tmp_path)
    adapter = algo._env
    adapter.reset()
    budget = adapter.episode_budget
    for step in range(R.EPISODE_LENGTH):
        obs, _, _, terminated, truncated, info = adapter.step(torch.zeros(1, 2))
        if bool(terminated.any() or truncated.any()):
            break
    assert step == R.EPISODE_LENGTH - 1 and bool(truncated.any())
    assert info["final_observation"].shape == (1, 61)
    assert float(info["final_observation"][0, -1]) == float(C.budget_feature(budget))
    assert adapter.episode_budget != budget and float(obs[0, -1]) == float(C.budget_feature(adapter.episode_budget))
    assert adapter.budget_draws == 2


def _replica(key: float, cfgs):
    from omnisafe.common.lagrange import Lagrange

    return Lagrange(**{**cfgs.lagrange_cfgs.todict(), "cost_limit": key})


@pytest.mark.omnisafe
def test_only_a_levels_own_episodes_move_its_multiplier(moderate) -> None:
    """Table 2.5: "updated only from episodes run at that level"; Q-level-jc: a level with no episode
    in the J_C window is not updated (its value and its Adam state stay as they are)."""
    lagranges = moderate._level_lagranges
    before = {k: lag.lagrangian_multiplier.item() for k, lag in lagranges.items()}
    moderate._update_level_multipliers([(20.0, 30.0), (20.0, 16.0)])
    replica = _replica(20.0, moderate._cfgs)
    replica.update_lagrange_multiplier(23.0)  # J_C = (30 + 16) / 2, not the level: the update moves it
    assert lagranges[20.0].lagrangian_multiplier.item() == replica.lagrangian_multiplier.item() != before[20.0]
    assert lagranges[20.0].lambda_optimizer.state
    for key in (10.0, 40.0):
        assert lagranges[key].lagrangian_multiplier.item() == before[key] and not lagranges[key].lambda_optimizer.state
    moderate._update_level_multipliers([(10.0, 12.0)])  # level 20 is updated again from its window episodes
    replica.update_lagrange_multiplier(23.0)
    assert lagranges[20.0].lagrangian_multiplier.item() == replica.lagrangian_multiplier.item()
    # 60 more episodes at 10 push the two episodes at 20 out of the 50-episode window: 20 is left alone
    def adam_steps(key: float) -> list[float]:
        return [float(s["step"]) for s in lagranges[key].lambda_optimizer.state.values()]

    value, steps = lagranges[20.0].lagrangian_multiplier.item(), adam_steps(20.0)
    assert steps == [2.0]
    moderate._update_level_multipliers([(10.0, 0.0)] * 60)
    assert len(moderate._level_window) == 50 == moderate._logger._data["Metrics/EpCost"].maxlen
    assert lagranges[20.0].lagrangian_multiplier.item() == value and adam_steps(20.0) == steps
    with pytest.raises(ValueError):
        moderate._update_level_multipliers([(17.5, 3.0)])  # not a Moderate level


@pytest.mark.omnisafe
def test_a_single_levels_jc_is_omnisafes_window_mean_bit_for_bit(tmp_path) -> None:
    """Q-level-jc with one level: the same episodes and the same float32 arithmetic as the J_C
    PPOLag reads (``get_stats('Metrics/EpCost')``), also once the 50-episode window has filled."""
    algo = _build(_arm("Single-10"), tmp_path)
    rng = np.random.default_rng(3)
    for _ in range(4):
        episodes = [(10.0, float(c)) for c in rng.integers(0, 200, 17)]
        for _, cost in episodes:
            algo._logger.store({"Metrics/EpCost": torch.tensor(cost, dtype=torch.float32)})
        replica_state = copy.deepcopy(algo._level_lagranges[10.0].lambda_optimizer.state_dict())
        before = algo._level_lagranges[10.0].lagrangian_multiplier.item()
        algo._update_level_multipliers(episodes)
        jc = algo._logger.get_stats("Metrics/EpCost")[0]
        window = [c for _, c, _ in algo._level_window]
        assert C._omnisafe_mean(window) == jc
        replica = _replica(10.0, algo._cfgs)
        replica.lagrangian_multiplier.data.fill_(before)
        replica.lambda_optimizer.load_state_dict(replica_state)
        replica.update_lagrange_multiplier(jc)
        assert algo._level_lagranges[10.0].lagrangian_multiplier.item() == replica.lagrangian_multiplier.item()
    assert len(algo._level_window) == 50


@pytest.mark.omnisafe
def test_a_continuous_bins_budget_is_the_mean_budget_of_its_window_episodes(tmp_path) -> None:
    """Q-continuous-bins: d of bin [10, 15) is the mean budget of its episodes, so the update
    minimises the loss -lambda * mean(C_e - b_e); an episode at 40 falls in the last bin."""
    algo = _build(_arm("Continuous"), tmp_path)
    algo._update_level_multipliers([(12.0, 20.0), (14.0, 10.0), (40.0, 50.0)])
    first, last = algo._level_lagranges[10.0], algo._level_lagranges[35.0]
    assert first.cost_limit == 13.0 and last.cost_limit == 40.0
    replica = _replica(13.0, algo._cfgs)
    replica.update_lagrange_multiplier(15.0)
    assert first.lagrangian_multiplier.item() == replica.lagrangian_multiplier.item()
    untouched = algo._level_lagranges[20.0]
    assert untouched.cost_limit == 22.5 and not untouched.lambda_optimizer.state  # the bin's centre, never updated


@pytest.mark.omnisafe
def test_the_surrogate_uses_each_samples_level_and_equals_ppolags_with_one_value(moderate) -> None:
    """Equation (3) per sample (the multiplier of the level whose feature the sample carries); with all
    multipliers equal it is PPOLag's _compute_adv_surrogate bit for bit (ppo_lag.py:101-102)."""
    values = {10.0: 0.25, 20.0: 1.5, 40.0: 0.0}
    for key, value in values.items():
        moderate._level_lagranges[key].lagrangian_multiplier.data.fill_(value)
    budgets = [10.0, 20.0, 40.0, 20.0, 10.0, 40.0] * 4
    obs = torch.stack([append_budget(torch.randn(60), b) for b in budgets])
    adv_r, adv_c = torch.randn(len(budgets)), torch.randn(len(budgets))
    captured = {}

    def loss_pi(obs, act, logp, adv):
        captured["adv"] = adv.detach().clone()
        return sum((p * 0.0).sum() for p in moderate._actor_critic.actor.parameters())

    moderate._loss_pi = loss_pi
    moderate._update_actor(obs, torch.zeros(len(budgets), 2), torch.zeros(len(budgets)), adv_r, adv_c)
    lam = torch.tensor([float(np.float32(values[b])) for b in budgets])
    assert torch.equal(captured["adv"], (adv_r - lam * adv_c) / (1 + lam))
    assert moderate._batch_penalty is None
    with pytest.raises(RuntimeError, match="only inside _update_actor"):
        moderate._compute_adv_surrogate(adv_r, adv_c)
    for lag in moderate._level_lagranges.values():
        lag.lagrangian_multiplier.data.fill_(0.3721)
    moderate._update_actor(obs, torch.zeros(len(budgets), 2), torch.zeros(len(budgets)), adv_r, adv_c)
    penalty = moderate._level_lagranges[10.0].lagrangian_multiplier.item()
    assert torch.equal(captured["adv"], (adv_r - penalty * adv_c) / (1 + penalty))


@pytest.mark.omnisafe
def test_every_checkpoint_holds_the_level_state_loadable_with_weights_only(moderate) -> None:
    omni = Path(moderate._logger.log_dir)
    state = torch.load(omni / "torch_save" / "epoch-0.pt", weights_only=True)
    assert {"level_lagranges", "level_lambda_optimizers", "level_window"} <= set(state)
    assert "lagrange" not in state and "lambda_optimizer" not in state  # no single multiplier
    assert state["level_lagranges"]["kind"] == C.DISCRETE
    assert state["level_lagranges"]["keys"].tolist() == [10.0, 20.0, 40.0]
    assert state["level_window"]["index"].numel() == 0 and state["level_window"]["maxlen"] == 50
    moderate._update_level_multipliers([(40.0, 70.0), (10.0, 1.0)])
    moderate._logger.torch_save()  # epoch-0.pt again (the logger's epoch is still 0)
    state = torch.load(omni / "torch_save" / "epoch-0.pt", weights_only=True)
    values = [lag.lagrangian_multiplier.item() for lag in moderate._level_lagranges.values()]
    assert state["level_lagranges"]["values"].tolist() == values
    assert state["level_lagranges"]["values"].dtype == torch.float64
    assert state["level_window"]["index"].tolist() == [2, 0] and state["level_window"]["cost"].tolist() == [70.0, 1.0]
    assert state["level_window"]["budget"].tolist() == [40.0, 10.0]
    assert set(state["level_lambda_optimizers"]) == {"10", "20", "40"}
    assert state["level_lambda_optimizers"]["40"]["state"] and not state["level_lambda_optimizers"]["20"]["state"]


@pytest.mark.omnisafe
def test_a_nonfinite_level_multiplier_is_seen_by_the_launcher(moderate) -> None:
    assert launch.nonfinite_state(moderate) is None
    moderate._level_lagranges[20.0].lagrangian_multiplier.data.fill_(float("nan"))
    assert launch.nonfinite_state(moderate) == "non_finite_multiplier"


# ---------------------------------------------------------------------------
# The factories refuse, and only refuse (HANDOVER.md section 8)
# ---------------------------------------------------------------------------


def _replace(spec, **changes):
    return manifest.RunSpec.from_dict({**spec.to_dict(), **changes})


SPEC_PROBLEMS = {
    "study": lambda s: _replace(s, study="A"),
    "plugin": lambda s: _replace(s, plugin="ppolag"),
    "base_algo": lambda s: _replace(s, base_algo="CPPOPID"),
    "task": lambda s: _replace(s, task="SafetyCarGoal1-v0"),
    "arm": lambda s: _replace(s, arm="Huge"),
    "levels": lambda s: _replace(s, training_levels=[10.0, 25.0, 40.0]),
    "onset": lambda s: _replace(s, onset_step=200_000),
    "divisor": lambda s: _replace(s, params={"budget_observation_divisor": 10.0}),
    "group": lambda s: _replace(s, group="study_b_fewshot"),
    "schedule": lambda s: _replace(s, params={**s.params, "lr_schedule_steps": R.FEWSHOT_STEPS}),  # cut 3's only
}
CFG_PROBLEMS = {  # a change of the configuration -> the words of its refusal
    "vector envs": (lambda c: c.train_cfgs.update({"vector_env_nums": 2}), "vector_env_nums"),
    "no cost": (lambda c: c.algo_cfgs.update({"use_cost": False}), "use_cost"),
    "seed": (lambda c: c.update({"seed": 7}), "cfgs.seed"),
    "multiplier init": (lambda c: c.lagrange_cfgs.update({"lagrangian_multiplier_init": 0.1}),
                        "lagrangian_multiplier_init"),
    "multiplier lr": (lambda c: c.lagrange_cfgs.update({"lambda_lr": 0.01}), "lambda_lr"),
    "set twice": (lambda c: c.update({"studyb_cfgs": {"kind": "discrete"}}), "already holds studyb_cfgs"),
}


@pytest.mark.omnisafe
@pytest.mark.parametrize("problem", sorted(SPEC_PROBLEMS))
def test_make_algorithm_refuses_a_spec_that_is_not_a_study_b_arm(tmp_path, problem) -> None:
    spec = _arm("Moderate")
    cfgs = _cfgs(spec, tmp_path / "omnisafe")
    with pytest.raises(RunRefused):
        C.make_algorithm(spec.task, cfgs, SPEC_PROBLEMS[problem](spec))
    assert "studyb_cfgs" not in cfgs and not (tmp_path / "omnisafe").exists()


@pytest.mark.omnisafe
@pytest.mark.parametrize("problem", sorted(CFG_PROBLEMS))
def test_make_algorithm_refuses_a_configuration_it_cannot_run(tmp_path, problem) -> None:
    spec = _arm("Moderate")
    cfgs = _cfgs(spec, tmp_path / "omnisafe")
    change, words = CFG_PROBLEMS[problem]
    change(cfgs)
    with pytest.raises(RunRefused, match=re.escape(words)):
        C.make_algorithm(spec.task, cfgs, spec)
    if problem != "set twice":
        assert "studyb_cfgs" not in cfgs


@pytest.mark.omnisafe
def test_make_algorithm_refuses_the_wrong_env_and_continuous_params(tmp_path) -> None:
    spec = _arm("Continuous")
    with pytest.raises(RunRefused, match="env_id"):
        C.make_algorithm("SafetyPointButton1-v0", _cfgs(spec, tmp_path / "a"), spec)
    for params in ({**spec.params, "continuous_range": [10.0, 45.0]}, {**spec.params, "continuous_bin_width": 2.5},
                   {"budget_observation_divisor": R.BUDGET_OBSERVATION_DIVISOR}):
        with pytest.raises(RunRefused, match="continuous"):
            C.make_algorithm(spec.task, _cfgs(spec, tmp_path / "b"), _replace(spec, params=params))


@pytest.mark.omnisafe
def test_make_algorithm_accepts_the_repilot_and_a_smoke_copy(tmp_path) -> None:
    """Unused params are accepted (the re-pilot adds pilot_revision); the total is not checked (smoke copies)."""
    repilot = next(s for s in manifest.pilot(1) if s.arm == "Moderate")
    assert repilot.params.get("pilot_revision") == 1
    algo = _build(repilot, tmp_path)
    assert algo._level_keys == (10.0, 20.0, 40.0)
    smoke = _replace(_arm("Sparse"), run_id="SMOKE-B-Sparse-s0", total_steps=R.STEPS_PER_EPOCH)
    assert _build(smoke, tmp_path)._level_keys == (10.0, 40.0)


FEWSHOT_PROBLEMS = {  # a change of the spec -> the words of its refusal
    "group": (lambda s: _replace(s, group="study_b"), "is not study_b_fewshot"),
    "plugin": (lambda s: _replace(s, plugin="study_b"), "plug-in 'study_b' is not 'study_b_fewshot'"),
    "params": (lambda s: _replace(s, params={k: v for k, v in s.params.items() if k != "horizons"}), "params lack"),
    "seen budget": (lambda s: _replace(s, params={**s.params, "budget": 20.0}), "not an unseen budget"),
    "horizon order": (lambda s: _replace(s, params={**s.params, "horizons": [500_000, 200_000, 1_000_000]}),
                      "not strictly increasing"),
    "unregistered horizon": (lambda s: _replace(s, params={**s.params, "horizons": [100_000, 1_000_000]}),
                             "are not few-shot horizons"),
    "last horizon": (lambda s: _replace(s, params={**s.params, "horizons": [200_000, 500_000]}),
                     "is not the continuation's total"),
    "fresh multiplier": (lambda s: _replace(s, params={**s.params, "fresh_multiplier_init": 0.0}), "a fresh multiplier"),
    "depends_on": (lambda s: _replace(s, depends_on=["B-Dense-s0"]), "is not the parent"),
    "parent step": (lambda s: _replace(s, params={**s.params, "parent_step": "final"}), "not a positive whole number"),
    # Table 9.1 (Q-continuations): only a continuation shortened by cut 3 keeps the 1,000,000-step schedule
    "schedule uncut": (lambda s: _replace(s, params={**s.params, "lr_schedule_steps": R.FEWSHOT_STEPS}),
                       "only a continuation shortened by Part 6.1 cut 3"),
    "schedule length": (lambda s: _replace(_cut_short(s), params={**_cut_short(s).params, "lr_schedule_steps": 500_000}),
                        "only a continuation shortened by Part 6.1 cut 3"),
    "schedule float": (lambda s: _replace(_cut_short(s), params={**_cut_short(s).params,
                                                                 "lr_schedule_steps": float(R.FEWSHOT_STEPS)}),
                       "only a continuation shortened by Part 6.1 cut 3"),
}


@pytest.mark.omnisafe
@pytest.mark.parametrize("problem", sorted(FEWSHOT_PROBLEMS))
def test_make_fewshot_algorithm_refuses_a_bad_continuation_spec(tmp_path, problem) -> None:
    spec = _fewshot()
    cfgs = _cfgs(_arm("Moderate"), tmp_path / "omnisafe")  # a configuration without the parent (never reached)
    cfgs.pilot_cfgs.extra_checkpoint_steps = [500_000]
    change, words = FEWSHOT_PROBLEMS[problem]
    with pytest.raises(RunRefused, match=re.escape(words)):
        C.make_fewshot_algorithm(spec.task, cfgs, change(spec))
    assert "studyb_cfgs" not in cfgs


@pytest.mark.omnisafe
def test_make_fewshot_algorithm_accepts_the_continuation_cut_3_shortens(tmp_path) -> None:
    """Part 6.1 cut 3 (Table 9.1, Q-continuations): the shortened spec, with its first horizon only and the full
    continuation's schedule length, passes every check of the spec and is refused only for the missing parent."""
    spec = _cut_short(_fewshot())
    assert (spec.total_steps, spec.params["horizons"], spec.params["lr_schedule_steps"]) == (
        R.FEWSHOT_HORIZONS[0], [R.FEWSHOT_HORIZONS[0]], R.FEWSHOT_STEPS)
    cfgs = _cfgs(_arm("Moderate"), tmp_path / "omnisafe")
    cfgs.pilot_cfgs.extra_checkpoint_steps = []  # its only horizon, 200,000, is on OmniSafe's cadence
    with pytest.raises(RunRefused, match="no entry for the parent"):  # past the spec's checks
        C.make_fewshot_algorithm(spec.task, cfgs, spec)
    assert "studyb_cfgs" not in cfgs


@pytest.mark.omnisafe
def test_make_fewshot_algorithm_needs_the_horizon_checkpoints_and_the_resolved_parent(tmp_path) -> None:
    spec = _fewshot()
    cfgs = _cfgs(_arm("Moderate"), tmp_path / "omnisafe")
    cfgs.pilot_cfgs.extra_checkpoint_steps = []
    with pytest.raises(RunRefused, match="lacks the horizons \\[500000\\]"):
        C.make_fewshot_algorithm(spec.task, cfgs, spec)
    cfgs.pilot_cfgs.extra_checkpoint_steps = [500_000]
    with pytest.raises(RunRefused, match="no entry for the parent"):
        C.make_fewshot_algorithm(spec.task, cfgs, spec)
    assert "studyb_cfgs" not in cfgs


@pytest.mark.omnisafe
@pytest.mark.parametrize("steps", [["x"], 5, [None], [500_000.0], [True]])
def test_make_fewshot_algorithm_refuses_malformed_extra_checkpoint_steps(tmp_path, steps) -> None:
    """A malformed ``pilot_cfgs.extra_checkpoint_steps`` is a RunRefused (HANDOVER.md section 8), not a crash."""
    spec = _fewshot()
    cfgs = _cfgs(_arm("Moderate"), tmp_path / "omnisafe")
    cfgs.pilot_cfgs.extra_checkpoint_steps = steps
    with pytest.raises(RunRefused, match="are not whole numbers of steps"):
        C.make_fewshot_algorithm(spec.task, cfgs, spec)
    assert "studyb_cfgs" not in cfgs

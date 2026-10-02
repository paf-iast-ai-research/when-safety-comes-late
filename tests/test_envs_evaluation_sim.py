"""The evaluation harness ``envs.evaluation`` (Role 2) in the pinned simulator (``omnisafe`` marker).

Each test runs at most a few 1,000-step episodes or resets (about 2.6 s per episode on the sandbox,
harness spec section 0). Policies come from synthetic OmniSafe run directories: config.json from OmniSafe's own PPOLag
defaults, ``epoch-k.pt`` holding the ``pi`` state of a fresh OmniSafe actor and the
``obs_normalizer`` state of an OmniSafe ``Normalizer`` pushed with random observations. Covered:
the frozen normaliser (spec 7.3), ``load_policy`` and its refusals (7.3; HANDOVER.md section 8), the wrapper
chain and bit-for-bit reproduction across calls, episode order and processes (7.4, 7.5; Table 3.1
"Determinism check"), the dynamics perturbation after every reset and the no-op of the literal
floor-only reading (Q-dynamics; 7.7.2), hazard layouts in both forms and hazard pinning (Q-hazard;
7.7.3), injected and budget-conditioned actors (Table 2.4, Table 2.5), contract 6 on a
budget-conditioned checkpoint (pilot/scheduler.py; Q-studyb-eval), missing, unreadable and ill-fitting
checkpoints as failures read before any episode (HANDOVER.md section 8; harness spec 7.10), the
recorded definitions of both battery conditions (7.7.2, 7.7.3), the line citations of the wrapper
chain, and the MuJoCo-instability detection (Q-mujoco-exception) in both its forms: MuJoCo's own
warnings (what the pinned stack does) and Safety-Gymnasium's dead exception branch, with the
reserve-seed replacement of an unstable episode (Table 9.1; its cap and bookkeeping are tested fast
in ``test_envs_evaluation.py``).
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import pickle
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from configs import registered as R
from envs import evaluation as E
from pilot import contracts, errors, manifest
from pilot.errors import RunRefused

pytestmark = pytest.mark.omnisafe

REPO = Path(__file__).resolve().parents[1]
TASK = "SafetyPointGoal1-v0"
CAR = "SafetyCarGoal1-v0"
SPE = 2_000  # the tiny test configurations' steps per epoch; checkpoints are named in the run's own epochs
TOTAL = 40_000  # a whole number of registered epochs, so that a RunSpec accepts it
A, B = E.selection_seeds()[:2]


@pytest.fixture(autouse=True)
def _restore_threads():
    threads = torch.get_num_threads()
    yield
    torch.set_num_threads(threads)


def _config(task: str = TASK, *, obs_normalize: bool = True, run_id: str | None = None, algo: str = "PPOLag") -> dict:
    from omnisafe.utils.config import get_default_kwargs_yaml

    cfg = get_default_kwargs_yaml("PPOLag", task, "on-policy").todict()
    cfg["train_cfgs"].update({"total_steps": TOTAL, "epochs": TOTAL // SPE, "device": "cpu", "torch_threads": 1})
    cfg["algo_cfgs"].update({"steps_per_epoch": SPE, "obs_normalize": obs_normalize})
    cfg.update({"env_id": task, "algo": algo, "exp_name": f"{algo}-{{{task}}}"})
    if run_id is not None:
        cfg["pilot_cfgs"] = {"run_id": run_id, "onset_step": 0}
    return cfg


def _fresh_actor(in_dim: int, act_dim: int = 2, seed: int = 0) -> torch.nn.Module:
    from gymnasium import spaces
    from omnisafe.models.actor.actor_builder import ActorBuilder

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return ActorBuilder(spaces.Box(-np.inf, np.inf, (in_dim,), np.float32), spaces.Box(-1.0, 1.0, (act_dim,), np.float32),
                            [64, 64], activation="tanh", weight_initialization_mode="kaiming_uniform",
                            ).build_actor("gaussian_learning")


def _normaliser_state(dim: int, *, count: int = 500, seed: int = 0) -> dict:
    from omnisafe.common import Normalizer

    norm = Normalizer((dim,), clip=5)
    if count:
        generator = torch.Generator().manual_seed(seed)
        norm.normalize(torch.randn(count, dim, generator=generator) * 2.0 + 0.5)
    return norm.state_dict()


def _write_run(root: Path, checkpoints: dict[int, dict], cfg: dict) -> Path:
    directory = root / "omnisafe" / f"{cfg['algo']}-{{{cfg['env_id']}}}" / "seed-000-synthetic"
    (directory / "torch_save").mkdir(parents=True)
    (directory / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    for epoch, state in checkpoints.items():
        torch.save(state, directory / "torch_save" / f"epoch-{epoch}.pt")
    return directory


def _spec(**changes) -> manifest.RunSpec:
    fields = {"run_id": "T-A-s0", "study": "A", "task": TASK, "arm": "T-A", "seed": 0, "total_steps": TOTAL,
              "base_algo": "PPOLag", "plugin": "study_a", "group": "main", "onset_step": 0}
    fields.update(changes)
    return manifest.RunSpec(**fields)


@pytest.fixture(scope="module")
def plain_run(tmp_path_factory) -> Path:
    """epoch-0.pt: a fresh normaliser (count 0, as OmniSafe saves it before training); epoch-1.pt: a pushed one."""
    actor = _fresh_actor(60)
    checkpoints = {0: {"pi": actor.state_dict(), "obs_normalizer": _normaliser_state(60, count=0)},
                   1: {"pi": actor.state_dict(), "obs_normalizer": _normaliser_state(60)}}
    return _write_run(tmp_path_factory.mktemp("plain"), checkpoints, _config(run_id="T-A-s0"))


def _streams() -> tuple:
    return pickle.dumps(np.random.get_state()), torch.get_rng_state(), random.getstate()


def _obs_normalize(eval_env: E.EvalEnv):
    from omnisafe.envs.wrapper import ObsNormalize

    wrapper = eval_env.env
    while not isinstance(wrapper, ObsNormalize):
        wrapper = wrapper._env
    return wrapper._obs_normalizer


@pytest.fixture(scope="module")
def episodes(plain_run) -> SimpleNamespace:
    """Six 1,000-step episodes shared by the reproduction tests.

    Seeds A and B in both orders, then A again through run_episodes and through EvalEnv.run_episode.
    """
    policy = E.load_policy(plain_run, SPE)
    before = _streams()
    ab = E.run_episodes(policy, [A, B])
    ba = E.run_episodes(policy, [B, A])
    again = E.run_episodes(policy, [A])
    after = _streams()
    env = E.make_eval_env(policy)
    try:
        norm = _obs_normalize(env)
        state_before = {k: v.clone() for k, v in norm.state_dict().items()}
        direct = env.run_episode(policy.actor, A)
        state_after = norm.state_dict()
        wrappers = _wrappers(env)
        chain = [type(w).__name__ for w in wrappers] + [type(wrappers[-1]._env).__name__]
        inner = type(env._base._env).__name__
        time_limit = next(w for w in wrappers if type(w).__name__ == "TimeLimit")._time_limit
    finally:
        env.close()
    return SimpleNamespace(policy=policy, ab=ab, ba=ba, again=again, direct=direct, before=before, after=after,
                           state_before=state_before, state_after=state_after, chain=chain, inner=inner,
                           time_limit=time_limit, norm_type=type(norm))


def _wrappers(eval_env: E.EvalEnv) -> list:
    """OmniSafe's wrappers around the evaluation env, outermost first."""
    from omnisafe.envs.core import Wrapper

    out, wrapper = [], eval_env.env
    while isinstance(wrapper, Wrapper):
        out.append(wrapper)
        wrapper = wrapper._env
    return out


# ---------------------------------------------------------------------------------------------
# FrozenNormalizer (spec 4.4, 7.3)
# ---------------------------------------------------------------------------------------------


def test_frozen_normalizer_never_changes_and_equals_omnisafes_normalisation() -> None:
    from omnisafe.common import Normalizer

    state = _normaliser_state(60)
    frozen = E.frozen_normalizer(state)
    assert type(frozen) is E.FrozenNormalizer and isinstance(frozen, Normalizer)
    before = {k: v.clone() for k, v in frozen.state_dict().items()}
    x = torch.randn(7, 60, generator=torch.Generator().manual_seed(1)) * 10.0
    out = frozen.normalize(x)
    frozen.normalize(x[0])
    frozen(x)
    after = frozen.state_dict()
    assert all(torch.equal(before[k], after[k]) and before[k].dtype == after[k].dtype for k in before)
    reference = Normalizer((60,), clip=5)
    reference.load_state_dict(state)
    reference._push = lambda raw_data: None  # OmniSafe's own normalize, without its update (normalizer.py:102-107)
    assert torch.equal(out, reference.normalize(x))
    assert torch.equal(out, torch.clamp((x - state["_mean"]) / state["_std"], -state["_clip"], state["_clip"]))
    assert bool((out.abs() <= 5.0).all()) and bool((out.abs() == 5.0).any())  # the saved clip applies
    updating = Normalizer((60,), clip=5)  # the contrast: OmniSafe's normaliser learns from what it sees
    updating.load_state_dict(state)
    updating.normalize(x)
    assert int(updating._count) == int(state["_count"]) + 7 and not torch.equal(updating._mean, state["_mean"])
    with pytest.raises(RuntimeError, match="never updates"):
        frozen._push(x)


def test_frozen_normalizer_is_the_identity_until_two_observations() -> None:
    from omnisafe.common import Normalizer

    x = torch.randn(3, 60, generator=torch.Generator().manual_seed(2))
    empty = _normaliser_state(60, count=0)  # epoch-0.pt: count 0, std 0 (spec 4.2)
    assert int(empty["_count"]) == 0 and torch.equal(E.frozen_normalizer(empty).normalize(x), x)
    one = Normalizer((60,), clip=5)
    one.normalize(torch.ones(1, 60))
    assert int(one._count) == 1 and torch.equal(E.frozen_normalizer(one.state_dict()).normalize(x), x)


# ---------------------------------------------------------------------------------------------
# load_policy (spec 7.3; HANDOVER.md section 8)
# ---------------------------------------------------------------------------------------------


def test_load_policy_reads_the_checkpoint_weights_only_with_one_thread(plain_run, monkeypatch) -> None:
    calls = []
    real = torch.load

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(torch, "load", spy)
    torch.set_num_threads(3)
    policy = E.load_policy(plain_run, SPE)
    assert calls == [{"map_location": "cpu", "weights_only": True}]  # contract 3: no pickled code runs
    assert torch.get_num_threads() == R.TORCH_THREADS
    assert policy.step == SPE and policy.checkpoint.name == "epoch-1.pt" and policy.task == TASK
    assert (policy.obs_dim, policy.act_dim) == (60, 2) and not policy.budget_conditioned and not policy.injected
    assert not policy.actor.training and not any(p.requires_grad for p in policy.actor.parameters())
    saved = real(policy.checkpoint, map_location="cpu", weights_only=True)
    assert all(torch.equal(v, saved["pi"][k]) for k, v in policy.actor.state_dict().items())
    assert set(policy.normalizer_state) == E.NORMALISER_KEYS
    assert all(torch.equal(v, saved["obs_normalizer"][k]) for k, v in policy.normalizer_state.items())


def test_load_policy_fails_when_torch_does_not_run_one_thread(plain_run, monkeypatch) -> None:
    monkeypatch.setattr(torch, "get_num_threads", lambda: 4)
    with pytest.raises(RuntimeError, match="Table 3.1"):
        E.load_policy(plain_run, SPE)


def test_load_policy_follows_the_runs_obs_normalize_flag(tmp_path) -> None:
    actor = _fresh_actor(60)
    directory = _write_run(tmp_path, {1: {"pi": actor.state_dict()}}, _config(obs_normalize=False))
    assert E.load_policy(directory, SPE).normalizer_state is None


def _refusal(tmp_path, *, cfg: dict | None = None, state: dict | None = None, step: int = SPE, spec=None) -> str:
    actor = _fresh_actor(60)
    ck = {"pi": actor.state_dict(), "obs_normalizer": _normaliser_state(60)} if state is None else state
    directory = _write_run(tmp_path, {1: ck}, _config(run_id="T-A-s0") if cfg is None else cfg)
    with pytest.raises(E.EvaluationRefused) as info:
        E.load_policy(directory, step, spec)
    assert isinstance(info.value, RunRefused) and isinstance(info.value, ValueError)
    return str(info.value)


def test_load_policy_refuses_a_spec_that_does_not_describe_the_run(tmp_path) -> None:
    assert "is not the spec's task" in _refusal(tmp_path / "a", spec=_spec(task=CAR))
    assert "is not the spec's run_id" in _refusal(tmp_path / "b", spec=_spec(run_id="T-B-s0", arm="T-B"))
    assert "is not the spec's total_steps" in _refusal(tmp_path / "c", spec=_spec(total_steps=2 * TOTAL))
    assert "is not the spec's base_algo" in _refusal(tmp_path / "d", spec=_spec(base_algo="CPPOPID"))
    assert "but the spec's plug-in is 'study_b'" in _refusal(tmp_path / "e", spec=_spec(plugin="study_b", study="B"))


def test_load_policy_refuses_a_configuration_the_evaluation_cannot_use(tmp_path) -> None:
    for i, (edit, message) in enumerate([
        (("algo_cfgs", "reward_normalize", True), "reward_normalize"),
        (("algo_cfgs", "cost_normalize", True), "cost_normalize"),
        (("env_cfgs", None, {"num_envs": 1}), "default layout distribution"),
    ]):
        cfg = _config(run_id="T-A-s0")
        section, key, value = edit
        if key is None:
            cfg[section] = value
        else:
            cfg[section][key] = value
        assert message in _refusal(tmp_path / str(i), cfg=cfg)
    no_model = {k: v for k, v in _config(run_id="T-A-s0").items() if k != "model_cfgs"}
    assert "model_cfgs.actor is missing" in _refusal(tmp_path / "no-model", cfg=no_model)
    unknown = _config(run_id="T-A-s0")
    unknown["model_cfgs"]["actor"]["activation"] = "nosuch"
    assert "model_cfgs.actor.activation is 'nosuch'" in _refusal(tmp_path / "activation", cfg=unknown)


def _failure(tmp_path, *, state: dict | None = None, step: int = SPE, spec=None) -> str:
    """A problem of the run's saved output: CheckpointInvalid, exit 1 (eval_failed), never a refusal."""
    actor = _fresh_actor(60)
    ck = {"pi": actor.state_dict(), "obs_normalizer": _normaliser_state(60)} if state is None else state
    directory = _write_run(tmp_path, {1: ck}, _config(run_id="T-A-s0"))
    with pytest.raises(E.CheckpointInvalid) as info:
        E.load_policy(directory, step, spec)
    assert not isinstance(info.value, RunRefused)
    assert isinstance(info.value, contracts.ContractError) and isinstance(info.value, ValueError)
    return str(info.value)


def test_load_policy_refuses_bad_steps_and_a_normaliser_covering_the_budget(tmp_path) -> None:
    assert "whole number" in _refusal(tmp_path / "a", step=SPE + 1)
    assert "non-negative integer" in _refusal(tmp_path / "b", step=-SPE)
    assert "beyond the run's 40000 steps" in _refusal(tmp_path / "c", step=TOTAL + SPE, spec=_spec())
    # the harness implements Q-budget-normalisation's answer (Table 9.1): a change would change the code, not the files
    assert "Q-budget-normalisation" in _refusal(tmp_path / "d", state={"pi": _fresh_actor(61).state_dict(),
                                                                       "obs_normalizer": _normaliser_state(61)})


def test_a_missing_or_ill_fitting_checkpoint_is_a_failure_not_a_refusal(tmp_path) -> None:
    """Classified as a truncated file is (``CheckpointUnreadable``): only restoring the files mends them."""
    assert "epoch-2.pt is missing" in _failure(tmp_path / "a", step=2 * SPE)
    assert "epoch-2.pt is missing" in _failure(tmp_path / "b", step=2 * SPE, spec=_spec())
    assert "holds no actor state" in _failure(tmp_path / "c", state={"obs_normalizer": _normaliser_state(60)})
    no_weight = {k: v for k, v in _fresh_actor(60).state_dict().items() if k != E.PLAIN_INPUT_KEY}
    assert "no first-layer weight" in _failure(tmp_path / "d", state={"pi": no_weight,
                                                                      "obs_normalizer": _normaliser_state(60)})
    assert "no complete 'obs_normalizer'" in _failure(tmp_path / "e", state={"pi": _fresh_actor(60).state_dict()})
    assert "takes 59 inputs" in _failure(tmp_path / "f", state={"pi": _fresh_actor(59).state_dict(),
                                                                "obs_normalizer": _normaliser_state(59)})
    assert "has shape (59,)" in _failure(tmp_path / "g", state={"pi": _fresh_actor(60).state_dict(),
                                                                "obs_normalizer": _normaliser_state(59)})
    bad_count = dict(_normaliser_state(60), _count=torch.tensor([1.0, 2.0]))  # a strict load refuses the shape
    assert "does not load" in _failure(tmp_path / "h", state={"pi": _fresh_actor(60).state_dict(),
                                                              "obs_normalizer": bad_count})
    wide = _fresh_actor(60).state_dict()  # the config's actor is [64, 64]; this state has a 65-unit second layer
    wide["mean.2.weight"] = torch.zeros(65, 64)
    assert "does not fit the actor of the run's config.json" in _failure(
        tmp_path / "i", state={"pi": wide, "obs_normalizer": _normaliser_state(60)})
    # a plain actor with a normaliser of obs_dim + 1 features is inconsistent, not the Q-budget-normalisation case
    assert "has shape (61,)" in _failure(tmp_path / "j", state={"pi": _fresh_actor(60).state_dict(),
                                                                "obs_normalizer": _normaliser_state(61)})


def test_an_unreadable_checkpoint_is_a_failure_not_a_refusal(tmp_path) -> None:
    """Every way torch.load(weights_only=True) can fail gives one class: exit 1 (eval_failed), never exit 5."""
    good = _write_run(tmp_path / "good", {1: {"pi": _fresh_actor(60).state_dict(), "obs_normalizer": _normaliser_state(60)}},
                      _config())
    raw = (good / "torch_save" / "epoch-1.pt").read_bytes()
    cases = {"empty": b"", "truncated": raw[: len(raw) // 2], "tail cut": raw[:-10], "garbage": b"\x80\x02garbage" * 10,
             "text": b"hello world"}  # the text file makes torch.load raise KeyError
    for name, data in cases.items():
        directory = _write_run(tmp_path / name, {}, _config())
        (directory / "torch_save" / "epoch-1.pt").write_bytes(data)
        with pytest.raises(E.CheckpointUnreadable, match="weights_only=True") as info:
            E.load_policy(directory, SPE)
        assert isinstance(info.value, E.CheckpointInvalid), name
        assert not isinstance(info.value, RunRefused), name
        assert isinstance(info.value, contracts.ContractError) and isinstance(info.value, ValueError), name
    pickled = _write_run(tmp_path / "pickled", {1: {"pi": _fresh_actor(60).state_dict(), "code": SimpleNamespace(a=1)}},
                         _config())
    with pytest.raises(E.CheckpointUnreadable, match="weights_only=True"):  # contract 3: no pickled objects
        E.load_policy(pickled, SPE)


def test_evaluate_run_reads_every_window_checkpoint_before_the_first_episode(tmp_path, monkeypatch) -> None:
    """A pilot run whose final checkpoint is truncated fails before any episode.

    Not after the 900 selection-set episodes of the earlier window checkpoints.
    """
    spec = next(s for s in manifest.pilot() if s.study == "A" and s.onset_step == 0 and s.seed == 0)
    directory = tmp_path / "omnisafe" / f"PPOLag-{{{spec.task}}}" / "seed-000-synthetic"
    (directory / "torch_save").mkdir(parents=True)
    cfg = _config(run_id=spec.run_id)
    cfg["train_cfgs"].update({"total_steps": spec.total_steps, "epochs": spec.total_steps // R.STEPS_PER_EPOCH})
    cfg["algo_cfgs"]["steps_per_epoch"] = R.STEPS_PER_EPOCH
    (directory / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    good = tmp_path / "good.pt"
    torch.save({"pi": _fresh_actor(60).state_dict(), "obs_normalizer": _normaliser_state(60)}, good)
    for step in contracts.expected_checkpoint_steps(spec):
        path = directory / "torch_save" / f"epoch-{step // R.STEPS_PER_EPOCH}.pt"
        if step == spec.total_steps:
            path.write_bytes(good.read_bytes()[: good.stat().st_size // 2])  # an interrupted copy
        else:
            path.symlink_to(good)
    calls = []
    monkeypatch.setattr(E, "run_episodes", lambda policy, seeds, **kw: calls.append(policy.step))
    with pytest.raises(E.CheckpointUnreadable, match="epoch-500.pt"):
        E.evaluate_run(str(directory), spec.to_dict())
    assert calls == []


# ---------------------------------------------------------------------------------------------
# The wrapper chain and reproduction (spec 7.4, 7.5; Table 3.1 "Determinism check")
# ---------------------------------------------------------------------------------------------


def test_the_eval_envs_line_citations_name_the_wrappers_of_the_installed_online_adapter() -> None:
    """EvalEnv's docstring cites omnisafe/adapter/online_adapter.py by line: each range must hold its wrapper."""
    import importlib.util
    import re

    package = Path(importlib.util.find_spec("omnisafe").submodule_search_locations[0])
    source = (package / "adapter" / "online_adapter.py").read_text(encoding="utf-8").splitlines()
    doc = " ".join(E.EvalEnv.__doc__.split())
    cited = re.findall(r"``(TimeLimit|ObsNormalize|ActionScale|Unsqueeze|AutoReset)[^`]*``[^()]*?"
                       r"\((?:online_adapter\.py)?:(\d+)(?:-(\d+))?", doc)
    assert {name for name, _, _ in cited} == {"TimeLimit", "ObsNormalize", "ActionScale", "Unsqueeze", "AutoReset"}
    for name, first, last in cited:
        lines = source[int(first) - 1:int(last or first)]
        assert any(f"{name}(" in line for line in lines), (name, first, last, lines)
        assert not any(f"{other}(" in line for line in lines for other in ("AutoReset", "ObsNormalize") if other != name)


def test_the_chain_is_trainings_without_autoreset(episodes) -> None:
    assert episodes.chain == ["Unsqueeze", "ActionScale", "ObsNormalize", "TimeLimit", "SafetyGymnasiumEnv"]
    assert episodes.inner != "SafeAutoResetWrapper"  # Safety-Gymnasium's inner auto-reset is stripped
    assert episodes.time_limit == R.EPISODE_LENGTH  # Table 2.1 "Episode": 1,000 steps
    assert episodes.norm_type is E.FrozenNormalizer


def test_every_episode_runs_1000_steps_with_raw_undiscounted_sums(episodes) -> None:
    for result in episodes.ab + episodes.ba + episodes.again + [episodes.direct]:
        assert result.length == R.EPISODE_LENGTH and result.budget is None
        assert result.cost == int(result.cost) and 0 <= result.cost <= R.EPISODE_LENGTH  # sum of 0/1 indicators


def test_an_episode_reproduces_bit_for_bit_across_calls_and_episode_order(episodes) -> None:
    a, b = episodes.ab
    assert [r.seed for r in episodes.ab] == [A, B] and [r.seed for r in episodes.ba] == [B, A]
    assert a == episodes.again[0] == episodes.ba[1] == episodes.direct  # dataclass equality: every float exactly
    assert b == episodes.ba[0]
    assert (a.cost, a.ret) != (b.cost, b.ret)  # two seeds, two different episodes


def test_the_frozen_normaliser_and_the_global_streams_are_untouched(episodes) -> None:
    assert all(torch.equal(episodes.state_before[k], episodes.state_after[k]) for k in episodes.state_before)
    np_state, torch_state, py_state = episodes.before
    assert episodes.after[0] == np_state and torch.equal(episodes.after[1], torch_state) and episodes.after[2] == py_state


def test_an_episode_reproduces_bit_for_bit_in_another_process(plain_run, episodes) -> None:
    code = (
        "import warnings; warnings.filterwarnings('ignore')\n"
        "from envs import evaluation as E\n"
        f"policy = E.load_policy({str(plain_run)!r}, {SPE})\n"
        f"print(repr(E.run_episodes(policy, [{A}])[0]))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().splitlines()[-1] == repr(episodes.ab[0])


def test_the_stripped_chain_equals_trainings_chain_with_its_auto_resets(episodes) -> None:
    from omnisafe.envs.core import make
    from omnisafe.envs.wrapper import ActionScale, AutoReset, ObsNormalize, TimeLimit, Unsqueeze

    cpu = torch.device("cpu")
    policy = episodes.policy
    with E._global_numpy_stream_kept():
        base = make(TASK, num_envs=1, device=cpu)
    env = TimeLimit(base, time_limit=base.max_episode_steps, device=cpu)  # online_adapter.py:117-146, as training
    env = AutoReset(env, device=cpu)
    env = ObsNormalize(env, device=cpu, norm=E.frozen_normalizer(policy.normalizer_state))
    env = Unsqueeze(ActionScale(env, low=-1.0, high=1.0, device=cpu), device=cpu)
    obs, _ = env.reset(seed=A)
    rewards, costs = [], []
    while True:
        with torch.no_grad():
            obs, reward, cost, terminated, truncated, _ = env.step(policy.actor.predict(obs, deterministic=True))
        rewards.append(float(reward.item()))
        costs.append(float(cost.item()))
        if bool(terminated.item()) or bool(truncated.item()):
            break
    env.close()
    assert (math.fsum(costs), math.fsum(rewards), len(costs)) == (episodes.ab[0].cost, episodes.ab[0].ret,
                                                                   episodes.ab[0].length)


def test_the_untrained_checkpoint_evaluates_with_the_identity_normaliser(plain_run) -> None:
    policy = E.load_policy(plain_run, 0)  # epoch-0.pt: count 0 (OmniSafe's rule: the identity)
    assert int(policy.normalizer_state["_count"]) == 0
    cost = E.evaluate_checkpoint(str(plain_run), 0, 1)
    assert type(cost) is float and 0.0 <= cost <= R.EPISODE_LENGTH


def test_a_policy_is_evaluated_only_on_a_task_with_its_observation_size(plain_run) -> None:
    policy = E.load_policy(plain_run, SPE)
    with pytest.raises(E.EvaluationRefused, match=r"SafetyCarGoal1-v0 observations have shape \(72,\)"):
        E.run_episodes(policy, [A], task=CAR)
    with pytest.raises(E.EvaluationRefused, match="cannot make 'SafetyNoSuchTask-v0'"):
        E.run_episodes(policy, [A], task="SafetyNoSuchTask-v0")
    with pytest.raises(E.EvaluationRefused, match="has no hazards"):  # Goal level 0 has none to relocate
        E.draw_hazard_layout("SafetyPointGoal0-v0", E.hazard_layout_seeds()[0])


# ---------------------------------------------------------------------------------------------
# Budget-conditioned and injected actors (Table 2.5; Table 2.4; HANDOVER.md section 8)
# ---------------------------------------------------------------------------------------------


def test_a_budget_conditioned_actor_gets_the_budget_after_normalisation(tmp_path) -> None:
    directory = _write_run(tmp_path, {1: {"pi": _fresh_actor(61).state_dict(), "obs_normalizer": _normaliser_state(60)}},
                           _config())
    policy = E.load_policy(directory, SPE)
    assert policy.budget_conditioned and policy.obs_dim == 60 and policy.normalizer_state["_mean"].shape == (60,)
    assert E.load_policy(directory, SPE, _spec(plugin="study_b", study="B", group="study_b")).budget_conditioned
    with pytest.raises(E.EvaluationRefused, match="spec's plug-in is 'study_a'"):
        E.load_policy(directory, SPE, _spec())
    with pytest.raises(ValueError, match="needs one budget per episode"):
        E.run_episodes(policy, [A])
    seen = []
    predict = policy.actor.predict

    def spy(obs, deterministic=False):
        seen.append(obs.clone())
        return predict(obs, deterministic=deterministic)

    policy.actor.predict = spy
    (result,) = E.run_episodes(policy, [A], budgets=[25.0])
    assert result.budget == 25.0 and result.length == R.EPISODE_LENGTH and len(seen) == R.EPISODE_LENGTH
    feature = torch.tensor(25.0 / R.BUDGET_OBSERVATION_DIVISOR, dtype=torch.float32)
    assert all(x.shape == (1, 61) and torch.equal(x[0, -1], feature) for x in seen)
    assert all(bool((x[0, :-1].abs() <= 5.0).all()) for x in seen)  # the first 60 features went through the normaliser


def test_evaluate_checkpoint_runs_a_budget_conditioned_checkpoint_on_its_arms_levels_in_turn(tmp_path, monkeypatch) -> None:
    """pilot/scheduler.py's study_b determinism report: contract 6 on a Moderate-like checkpoint (Q-studyb-eval)."""
    spec = _spec(run_id="DET-B-Moderate-s0", study="B", arm="Moderate", plugin="study_b", group="determinism",
                 onset_step=None, training_levels=R.STUDY_B_ARMS["Moderate"],
                 params={"budget_observation_divisor": R.BUDGET_OBSERVATION_DIVISOR})
    checkpoint = {"pi": _fresh_actor(61).state_dict(), "obs_normalizer": _normaliser_state(60)}
    directory = _write_run(tmp_path, {1: checkpoint}, _config(run_id=spec.run_id))
    (tmp_path / "spec.json").write_text(spec.to_json(), encoding="utf-8")  # RUN_DIR/spec.json (pilot/rundir.py)
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset())  # the key open, whatever the repository answers
    with pytest.raises(errors.PendingQuestionError, match="Q-studyb-eval"):
        E.evaluate_checkpoint(str(directory), SPE, 2)
    with errors.allow_pending():  # smoke only; a registered report needs Q-studyb-eval answered
        cost = E.evaluate_checkpoint(str(directory), SPE, 2)
    policy = E.load_policy(directory, SPE, spec)
    direct = E.run_episodes(policy, [A, B], budgets=[10.0, 20.0])  # Moderate's first two levels, in turn
    assert [r.budget for r in direct] == [10.0, 20.0] and type(cost) is float
    assert cost == E.summarise(direct)[0]


def test_the_hazard_definition_refuses_a_task_without_hazards() -> None:
    with pytest.raises(E.EvaluationRefused, match="no hazards to relocate"):
        E.hazard_definition("SafetyPointGoal0-v0")  # level 0 has none (every Study A task has, tested below)


def test_an_injected_actor_evaluates_exactly_as_the_plain_actor_it_came_from(tmp_path) -> None:
    from metrics.interventions import InjectedHead, intervention_generator, plasticity_injection

    actor = _fresh_actor(60)
    optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
    plain = copy.deepcopy(actor.state_dict())
    plasticity_injection(actor, optimizer, intervention_generator(0))
    injected = actor.state_dict()
    assert E.INJECTED_KEY in injected and E.PLAIN_INPUT_KEY not in injected
    norm = _normaliser_state(60)
    directory = _write_run(tmp_path, {1: {"pi": plain, "obs_normalizer": norm}, 2: {"pi": injected, "obs_normalizer": norm}},
                           _config())
    # detected by the pi state's key, never by the spec's treatment (HANDOVER.md section 8)
    p_plain = E.load_policy(directory, SPE, _spec(treatment="injection", group="treatment"))
    p_injected = E.load_policy(directory, 2 * SPE, _spec())
    assert not p_plain.injected and p_injected.injected and isinstance(p_injected.actor.mean, InjectedHead)
    x = torch.randn(256, 60, generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        assert torch.equal(p_plain.actor.predict(x, deterministic=True), p_injected.actor.predict(x, deterministic=True))
    assert E.run_episodes(p_plain, [A]) == E.run_episodes(p_injected, [A])


# ---------------------------------------------------------------------------------------------
# Dynamics perturbation (Table 2.2; Q-dynamics; spec 5.6, 7.7.2)
# ---------------------------------------------------------------------------------------------

ROBOT_BODIES = {TASK: ["agent"], CAR: ["agent", "left", "right", "rear"]}  # spec 5.6 (MEASURED exp3)
UNCHANGED_FIELDS = ("body_pos", "body_quat", "body_ipos", "body_iquat", "geom_size", "geom_pos", "geom_type",
                    "geom_solref", "geom_solimp", "dof_damping", "dof_armature", "actuator_gear", "jnt_range")


@pytest.mark.parametrize("task", [TASK, CAR])
def test_dynamics_scales_exactly_the_intended_model_fields_after_every_reset(task) -> None:
    base, perturbed = E.EvalEnv(task, None), E.EvalEnv(task, None, "dynamics")
    try:
        for i, seed in enumerate(E.measurement_seeds()[:2]):
            base.env.reset(seed=seed)
            perturbed.env.reset(seed=seed)
            mb, mp = base.task.model, perturbed.task.model
            agent = mb.body("agent").id
            bodies = [b for b in range(mb.nbody) if mb.body_rootid[b] == agent]
            assert [mb.body(b).name for b in bodies] == ROBOT_BODIES[task]
            others = [b for b in range(mb.nbody) if b not in bodies]
            for field in ("body_mass", "body_inertia"):  # x 1.3 exactly, once (not 1.69 after the second reset)
                assert np.array_equal(getattr(mp, field)[bodies], getattr(mb, field)[bodies] * R.BODY_MASS_SCALE)
                assert np.array_equal(getattr(mp, field)[others], getattr(mb, field)[others])
            assert mp.body_subtreemass[agent] == pytest.approx(mb.body_subtreemass[agent] * R.BODY_MASS_SCALE, rel=1e-12)
            assert not np.array_equal(mp.dof_invweight0, mb.dof_invweight0)  # mj_setConst ran
            geoms = sorted({g for g in range(mb.ngeom) if mb.geom_bodyid[g] in bodies} | {mb.geom("floor").id})
            rest = [g for g in range(mb.ngeom) if g not in geoms]
            assert np.array_equal(mp.geom_friction[geoms], mb.geom_friction[geoms] * R.GROUND_FRICTION_SCALE)
            assert np.array_equal(mp.geom_friction[rest], mb.geom_friction[rest])  # vase, goal, hazards untouched
            for field in UNCHANGED_FIELDS:
                assert np.array_equal(getattr(mp, field), getattr(mb, field)), field
            # the perturbation changes dynamics only: the same layout and initial state for the seed (Q-dynamics,
            # docs/DECISIONS.md: the episodes use the measurement seeds, paired with the measurement-set C_ID)
            layout_b, layout_p = base.task.world_info.layout, perturbed.task.world_info.layout
            assert layout_b.keys() == layout_p.keys() and all(np.array_equal(layout_b[k], layout_p[k]) for k in layout_b)
            assert np.array_equal(base.task.data.qpos, perturbed.task.data.qpos)
            assert np.array_equal(base.task.data.qvel, perturbed.task.data.qvel)
            assert perturbed.perturbation.applications == i + 1
            assert perturbed.perturbation.record["bodies"] == ROBOT_BODIES[task]
            geom_names = [mb.geom(g).name or f"geom#{g}" for g in geoms]
            assert perturbed.perturbation.record["friction_geoms"] == geom_names and "floor" in geom_names
    finally:
        base.close()
        perturbed.close()
    # The recorded definition (spec 7.7.2) names exactly what was scaled, read from the task itself.
    definition = E.dynamics_definition(task)
    assert definition["task"] == task and definition["bodies"] == ROBOT_BODIES[task]
    assert definition["friction_geoms"] == geom_names
    assert definition["body_mass_scale"] == R.BODY_MASS_SCALE and definition["ground_friction_scale"] == R.GROUND_FRICTION_SCALE
    assert definition["episode_seeds"] == "measurement" and json.loads(json.dumps(definition)) == definition


def test_the_dynamics_hook_refuses_a_reset_that_scales_other_bodies(monkeypatch) -> None:
    """The recorded names hold for every episode: a reset that scaled a different set is refused."""
    env = E.EvalEnv(TASK, None, "dynamics")
    try:
        env.env.reset(seed=E.measurement_seeds()[0])
        scale = E.apply_dynamics_perturbation
        monkeypatch.setattr(E, "apply_dynamics_perturbation", lambda task: dict(scale(task), bodies=["vase0"]))
        with pytest.raises(RuntimeError, match="same bodies and geoms in every episode"):
            env.env.reset(seed=E.measurement_seeds()[1])
    finally:
        env.close()


def test_the_dynamics_hook_refuses_a_model_that_was_not_recompiled() -> None:
    env = E.EvalEnv(TASK, None, "dynamics")
    try:
        env.env.reset(seed=E.measurement_seeds()[0])
        env.perturbation._original_build = lambda: None  # a reset that keeps the scaled model
        with pytest.raises(RuntimeError, match="compound"):
            env.env.reset(seed=E.measurement_seeds()[1])
    finally:
        env.close()


def _open_loop(task: str, hook=None, steps: int = 300) -> np.ndarray:
    """qpos and observation after each of ``steps`` fixed random actions from one seeded reset."""
    import safety_gymnasium

    with E._global_numpy_stream_kept():
        env = safety_gymnasium.make(task, autoreset=False)
    try:
        sim = env.unwrapped.task
        if hook is not None:
            build = sim._build

            def rebuild():
                build()
                hook(sim)

            sim._build = rebuild
        env.reset(seed=E.measurement_seeds()[0])
        actions = np.random.RandomState(0)
        rows = []
        for _ in range(steps):
            obs, *_ = env.step(np.clip(actions.normal(0.3, 0.6, env.action_space.shape), -1.0, 1.0))
            rows.append(np.concatenate([sim.data.qpos.copy(), obs]))
    finally:
        env.close()
    return np.array(rows)


@pytest.mark.parametrize("task", [TASK, CAR])
def test_scaling_only_the_floors_friction_changes_nothing(task) -> None:
    """The finding behind Q-dynamics: the literal "ground friction by 0.7" on the floor geom is a no-op.

    MuJoCo combines two geoms' friction by the element-wise maximum (spec 5.6, exp4).

    The hook changes the friction only. Contact friction is read at collision time, so no
    ``mj_forward`` is needed, and an extra ``mj_forward`` after the build would by itself move the
    Car's trajectory by about 5e-13 (its constraint solver is warm-started; measured in this work
    package's scratch check), which would hide what the friction change does.
    """

    def floor_only(sim) -> None:
        model = sim.model
        model.geom_friction[model.geom("floor").id] *= R.GROUND_FRICTION_SCALE

    baseline = _open_loop(task)
    assert np.array_equal(_open_loop(task, floor_only), baseline)  # bit-identical trajectory
    assert not np.array_equal(_open_loop(task, E.apply_dynamics_perturbation), baseline)  # the answer acts


# ---------------------------------------------------------------------------------------------
# Hazard relocation (Table 2.2; Q-hazard; spec 5.2, 7.7.3)
# ---------------------------------------------------------------------------------------------


def test_the_registered_form_is_the_layout_seeds_own_draw() -> None:
    import safety_gymnasium

    seed = E.hazard_layout_seeds()[0]
    with E._global_numpy_stream_kept():
        env = safety_gymnasium.make(TASK, autoreset=False)
    try:
        env.reset(seed=seed)
        sim = env.unwrapped.task
        expected = [(float(sim.world_info.layout[f"hazard{i}"][0]), float(sim.world_info.layout[f"hazard{i}"][1]))
                    for i in range(sim.hazards.num)]
    finally:
        env.close()
    assert E.draw_hazard_layout(TASK, seed, "registered") == expected  # the task's own generator (reading B)
    central = E.draw_hazard_layout(TASK, seed, "central")
    assert E.draw_hazard_layout(TASK, seed) == central  # the default is HAZARD_FORM; the same seed, the same layout
    assert central != expected


HAZARDS = {"SafetyPointGoal1-v0": 8, "SafetyCarGoal1-v0": 8, "SafetyPointButton1-v0": 4}  # goal/button_level1.py:33


@pytest.mark.parametrize("task", R.TASKS_STUDY_A)
def test_the_central_form_keeps_every_hazard_inside_the_recorded_bounds(task) -> None:
    """The recorded definition (spec 7.7.3) says where hazard centres lie.

    The placement square [-0.75, 0.75]^2 shrunk by the keepout 0.18, so within 0.57, with discs of
    size 0.2 reaching 0.77 (not "|x|, |y| <= 0.75").
    """
    central, registered = E._HazardDrawer(task, "central"), E._HazardDrawer(task, "registered")
    try:
        keepout, num, size = central._task.hazards.keepout, central._task.hazards.num, central._task.hazards.size
        drawn = [central.draw(seed) for seed in E.hazard_layout_seeds()[:2]]
        wide = registered.draw(E.hazard_layout_seeds()[0])
    finally:
        central.close()
        registered.close()
    a = E.HAZARD_CENTRAL_HALF_WIDTH
    definition = E.hazard_definition(task)
    assert (definition["form"], definition["task"], definition["placements"]) == ("central", task, [[-a, -a, a, a]])
    assert (definition["hazards"], definition["size"], definition["keepout"]) == (HAZARDS[task], 0.2, 0.18) == (num, size, keepout)
    bound = definition["centre_half_width"]
    assert bound == pytest.approx(a - keepout, abs=1e-12) and bound == pytest.approx(0.57, abs=1e-12)
    assert definition["centre_bounds"] == [[-bound, -bound, bound, bound]]
    reach = definition["reach_half_width"]
    assert reach == pytest.approx(0.77, abs=1e-12) and reach > a  # the discs reach past the square
    assert definition["layout_seeds"] == E.hazard_layout_seeds() and definition["episode_seeds"] == E.hazard_episode_seeds()
    assert json.loads(json.dumps(definition)) == definition
    for layout in drawn:
        assert len(layout) == num and len(set(layout)) == num
        assert all(abs(x) <= bound + 1e-12 and abs(y) <= bound + 1e-12 for x, y in layout)
    assert drawn[0] != drawn[1]
    literal = E.hazard_definition(task, "registered")  # reading B: the task's own placements, its extents
    assert literal["form"] == "registered" and literal["placements"] is None and literal["extents"] == [-1.5, -1.5, 1.5, 1.5]
    assert literal["centre_half_width"] == pytest.approx(1.5 - keepout, abs=1e-12)
    wide_max = max(max(abs(x), abs(y)) for x, y in wide)
    assert a < wide_max <= literal["centre_half_width"] + 1e-12  # the training extents are wider


def test_pinned_hazards_stay_while_each_episode_seed_redraws_the_rest(plain_run) -> None:
    policy = E.load_policy(plain_run, SPE)
    hazards = E.draw_hazard_layout(TASK, E.hazard_layout_seeds()[0])
    env = E.make_eval_env(policy, condition="hazard")
    try:
        env.pin_hazards(hazards)
        others = []
        for seed in E.hazard_episode_seeds()[0][:3]:
            obs, _ = env.env.reset(seed=seed)
            assert obs.shape == (1, 60)  # the observation size is unchanged
            layout = env.task.world_info.layout
            for i, (x, y) in enumerate(hazards):
                assert np.allclose(layout[f"hazard{i}"], (x, y), rtol=0.0, atol=1e-8)  # pinned (to 1e-9)
                assert np.allclose(env.task.data.body(f"hazard{i}").xpos[:2], (x, y), rtol=0.0, atol=1e-8)
            others.append((tuple(layout["agent"]), tuple(layout["goal"]), tuple(layout["vase0"])))
        assert len(set(others)) == 3  # agent, goal and vase differ between the episode seeds
    finally:
        env.close()
    extra = E.EvalEnv(TASK, None, "hazard")
    try:
        with pytest.raises(ValueError, match="7 hazard locations for 8 hazards"):
            extra.pin_hazards(hazards[:-1])
    finally:
        extra.close()


# ---------------------------------------------------------------------------------------------
# A MuJoCo instability (Q-mujoco-exception)
# ---------------------------------------------------------------------------------------------


def _faulty(policy, fault: str, env_of, at: int = 10):
    """``policy.actor.predict`` that injects ``fault`` before step ``at``; returns the step counter.

    The fault is a non-finite velocity (MuJoCo's velocity check fires in that step's ``mj_step``) or a
    non-finite control.
    """
    predict = policy.actor.predict
    steps: list[int] = []

    def faulty(obs, deterministic=False):
        steps.append(1)
        action = predict(obs, deterministic=deterministic)
        if len(steps) == at:
            if fault == "qvel":
                env_of().task.data.qvel[:] = np.inf
            else:
                action = torch.full_like(action, float("nan"))
        return action

    policy.actor.predict = faulty
    return steps


@pytest.mark.parametrize("fault, warning", [("qvel", "mjWARN_BADQVEL"), ("ctrl", "mjWARN_BADCTRL")])
def test_an_unstable_simulation_is_never_scored_although_mujoco_raises_nothing(
        plain_run, tmp_path, monkeypatch, fault, warning) -> None:
    """MuJoCo's warning counters stop an unstable episode before it is scored.

    In the pinned stack MuJoCo only warns, resets (or ignores the controls) and goes on: without the
    counters the episode would be scored as a normal 1,000-step one. Seeds outside the canonical sets
    have no reserve sequence, so here the unstable episode fails the call (Q-mujoco-exception).
    """
    monkeypatch.chdir(tmp_path)  # MuJoCo appends its warning to MUJOCO_LOG.TXT in the working directory
    policy = E.load_policy(plain_run, SPE)
    envs: list = []
    make = E.make_eval_env
    monkeypatch.setattr(E, "make_eval_env", lambda *a, **k: envs.append(make(*a, **k)) or envs[-1])
    steps = _faulty(policy, fault, lambda: envs[-1])
    with pytest.raises(E.MujocoInstabilityError) as info:  # through the public loop
        E.run_episodes(policy, [7, 8])
    message = str(info.value)
    assert warning in message and "seed 7" in message and "at step 10" in message and "Q-mujoco-exception" in message
    assert "no reserve" in message and (info.value.seed, info.value.step) == (7, 10) and warning in info.value.warning
    assert len(steps) == 10  # stopped at the step, not scored
    assert not isinstance(info.value, RunRefused)  # an evaluation failure (exit 1), never a refusal
    assert (tmp_path / "MUJOCO_LOG.TXT").is_file()  # MuJoCo's own file (the docstrings say so)


def test_an_unstable_episode_is_replaced_by_the_sets_next_reserve_seed(plain_run, tmp_path, monkeypatch) -> None:
    """Q-mujoco-exception (Table 9.1) in the simulator: a real non-finite velocity at step 10 of selection seed A's
    episode; ``run_episodes`` scores the selection set's first reserve seed (1,050,000) in its place, a clean
    episode equal to a fresh one, keeps B and the planned order, and records the replacement."""
    monkeypatch.chdir(tmp_path)
    policy = E.load_policy(plain_run, SPE)
    envs: list = []
    make = E.make_eval_env
    monkeypatch.setattr(E, "make_eval_env", lambda *a, **k: envs.append(make(*a, **k)) or envs[-1])
    predict = policy.actor.predict
    steps = _faulty(policy, "qvel", lambda: envs[-1])
    first, second = E.run_episodes(policy, [A, B])
    assert (first.seed, second.seed) == (1_050_000, B) and second.replaced == ()
    (replacement,) = first.replaced
    assert (replacement.seed, replacement.reserve_seed, replacement.step) == (A, 1_050_000, 10)
    assert "mjWARN_BADQVEL" in replacement.warning
    assert len(steps) == 10 + 2 * R.EPISODE_LENGTH  # the unstable episode was cut at step 10 and never scored
    assert E._checked([first, second], [A, B], "selection") == [first, second]
    policy.actor.predict = predict
    (fresh,) = E.run_episodes(policy, [1_050_000])
    assert dataclasses.replace(first, replaced=()) == fresh  # the reserve episode carries nothing of the unstable one


def test_the_instability_counters_belong_to_one_episode(plain_run, tmp_path, monkeypatch) -> None:
    """``MjData`` is rebuilt at every reset: after a failed episode, the next one on the same env is scored."""
    monkeypatch.chdir(tmp_path)
    policy = E.load_policy(plain_run, SPE)
    env = E.make_eval_env(policy)
    try:
        predict = policy.actor.predict
        _faulty(policy, "qvel", lambda: env)
        with pytest.raises(E.MujocoInstabilityError, match="mjWARN_BADQVEL"):
            env.run_episode(policy.actor, A)
        policy.actor.predict = predict
        result = env.run_episode(policy.actor, A)
        assert result.length == R.EPISODE_LENGTH and result == E.run_episodes(policy, [A])[0]
    finally:
        env.close()


def test_safety_gymnasiums_dead_exception_branch_is_a_second_guard(plain_run) -> None:
    """``MujocoException`` is raised nowhere in the pinned libraries; its dead branch fails too.

    Were its branch taken, ``Builder.step`` would raise ``UnboundLocalError`` (sg/builder.py:201-206,
    249), which fails the evaluation too.
    """
    policy = E.load_policy(plain_run, SPE)
    env = E.make_eval_env(policy)
    try:
        env.task.simulation_forward = lambda action: True  # what Underlying.simulation_forward returns after a MujocoException
        with pytest.raises(E.MujocoInstabilityError, match="Q-mujoco-exception") as info:
            env.run_episode(policy.actor, A)
        assert isinstance(info.value.__cause__, UnboundLocalError) and "'cost'" in str(info.value.__cause__)
        assert not isinstance(info.value, RunRefused)  # an evaluation failure (exit 1), never a refusal

        def other(action):
            raise UnboundLocalError("local variable 'x' referenced before assignment")

        env.task.simulation_forward = other
        with pytest.raises(UnboundLocalError) as plain_error:
            env.run_episode(policy.actor, A)
        assert type(plain_error.value) is UnboundLocalError  # any other error propagates unchanged
    finally:
        env.close()

"""The evaluation harness ``envs.evaluation`` (Role 2) on a real, tiny training run (``slow`` marker).

One PPO-Lagrangian run of one 2,000-step epoch (``FullStateCheckpointMixin`` over OmniSafe's
PPOLag, built directly with a small configuration as the slow tests do) gives a real
checkpoint: an actor after one update and a normaliser that has seen the rollout. On it:

* contract 6 (Table 3.1 "Determinism check"): ``evaluate_checkpoint(dir, step, 2)`` equals itself on
  a second call in the same process and in another process, and writes nothing;
* contract 1: a registered-looking run directory (config.json of a pilot spec's total, every Part 3.4
  checkpoint linked to the real one) passes ``pilot.contracts.validate_evaluation`` with
  ``run_episodes`` replaced by a fast stub, ``load_policy`` running for real;
* contract 5: the same directory gives ``pilot.enrichment.gaps_from_costs`` what it checks;
* ``evaluate_continuation`` loads a continuation's final checkpoint;
* hazard and dynamics episodes of the trained policy reproduce exactly;
* the central hazard form gives 100 distinct complete layouts, hazards pinned per layout (Q-hazard).
"""

from __future__ import annotations

import contextlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from configs import registered as R
from envs import evaluation as E
from pilot import contracts, manifest

pytestmark = [pytest.mark.omnisafe, pytest.mark.slow]

REPO = Path(__file__).resolve().parents[1]
TASK = "SafetyPointGoal1-v0"
SPE = 2_000


@pytest.fixture(autouse=True)
def _restore_threads():
    threads = torch.get_num_threads()
    yield
    torch.set_num_threads(threads)


@pytest.fixture(scope="module")
def trained(tmp_path_factory) -> Path:
    """The OmniSafe directory of a one-epoch, 2,000-step PPO-Lagrangian run (epoch-0.pt and epoch-1.pt)."""
    from omnisafe.algorithms.on_policy.naive_lagrange.ppo_lag import PPOLag
    from omnisafe.utils.config import Config, check_all_configs, get_default_kwargs_yaml
    from omnisafe.utils.tools import recursive_check_config

    from pilot.algorithms import FullStateCheckpointMixin

    threads = torch.get_num_threads()
    torch.set_num_threads(R.TORCH_THREADS)
    try:
        log_dir = tmp_path_factory.mktemp("trained")
        cfgs = get_default_kwargs_yaml("PPOLag", TASK, "on-policy")
        custom = {"seed": 0,
                  "train_cfgs": {"device": "cpu", "torch_threads": 1, "vector_env_nums": 1, "parallel": 1,
                                 "total_steps": SPE},
                  "algo_cfgs": {"steps_per_epoch": SPE},
                  "logger_cfgs": {"save_model_freq": 1, "log_dir": str(log_dir), "use_tensorboard": False}}
        recursive_check_config(custom, cfgs)
        cfgs.recurisve_update(custom)  # (sic) OmniSafe's spelling
        cfgs.update({"exp_increment_cfgs": custom})
        cfgs.recurisve_update({"exp_name": f"PPOLag-{{{TASK}}}", "env_id": TASK, "algo": "PPOLag"})
        cfgs.train_cfgs.recurisve_update({"epochs": 1})
        check_all_configs(cfgs, "on-policy")
        cfgs["pilot_cfgs"] = Config.dict2config({"run_id": "T-s0", "onset_step": 0})
        algo = type("FullStatePPOLag", (FullStateCheckpointMixin, PPOLag), {})(env_id=TASK, cfgs=cfgs)
        algo.learn()
        directory = Path(algo._logger.log_dir)
    finally:
        torch.set_num_threads(threads)
    assert sorted(p.name for p in (directory / "torch_save").glob("epoch-*.pt")) == ["epoch-0.pt", "epoch-1.pt"]
    return directory


def _tree(root: Path) -> dict[str, tuple[int, int]]:
    return {str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(root.rglob("*"))}


def _linked_run(trained: Path, root: Path, spec: manifest.RunSpec, steps: list[int], steps_per_epoch: int) -> Path:
    """A run directory for ``spec`` whose every checkpoint in ``steps`` is the trained epoch-1.pt."""
    directory = root / "omnisafe" / f"PPOLag-{{{spec.task}}}" / "seed-000-fixture"
    (directory / "torch_save").mkdir(parents=True)
    cfg = json.loads((trained / "config.json").read_text(encoding="utf-8"))
    cfg["train_cfgs"].update({"total_steps": spec.total_steps, "epochs": spec.total_steps // steps_per_epoch})
    cfg["algo_cfgs"]["steps_per_epoch"] = steps_per_epoch
    cfg["pilot_cfgs"] = {"run_id": spec.run_id, "onset_step": int(spec.onset_step or 0)}
    (directory / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    source = trained / "torch_save" / "epoch-1.pt"
    for step in steps:
        (directory / "torch_save" / f"epoch-{step // steps_per_epoch}.pt").symlink_to(source)
    return directory


@pytest.fixture(scope="module")
def registered_looking(trained, tmp_path_factory):
    spec = next(s for s in manifest.pilot() if s.study == "A" and s.onset_step == 5_000_000 and s.seed == 0)
    steps = contracts.expected_checkpoint_steps(spec)
    return spec, _linked_run(trained, tmp_path_factory.mktemp("registered"), spec, steps, R.STEPS_PER_EPOCH)


class RecordingStub:
    """``run_episodes`` replaced by a fast, deterministic stub that records the real policies it gets."""

    def __init__(self) -> None:
        self.calls: list[tuple[E.Policy, list[int], dict]] = []

    def __call__(self, policy, seeds, **kwargs):
        self.calls.append((policy, list(seeds), kwargs))
        return [E.EpisodeResult(seed=s, cost=float(s % 9), ret=(s % 4) / 8.0, length=R.EPISODE_LENGTH) for s in seeds]


# ---------------------------------------------------------------------------------------------
# Contract 6: the determinism evaluation
# ---------------------------------------------------------------------------------------------


def test_evaluate_checkpoint_is_bit_identical_across_calls_and_processes_and_writes_nothing(
        trained, tmp_path, monkeypatch) -> None:
    before = _tree(trained)
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    first = E.evaluate_checkpoint(str(trained), SPE, 2)
    second = E.evaluate_checkpoint(str(trained), SPE, 2)
    assert type(first) is float and first == second
    code = (
        "import warnings; warnings.filterwarnings('ignore')\n"
        "from envs import evaluation as E\n"
        f"print(repr(E.evaluate_checkpoint({str(trained)!r}, {SPE}, 2)))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().splitlines()[-1] == repr(first)
    assert _tree(trained) == before  # nothing written into the run directory
    assert list(cwd.iterdir()) == []  # nor into the working directory


# ---------------------------------------------------------------------------------------------
# Contracts 1 and 5 on a registered-looking directory of the real checkpoint
# ---------------------------------------------------------------------------------------------


def test_evaluate_run_on_a_registered_looking_run_passes_contract_one(trained, registered_looking, monkeypatch) -> None:
    spec, directory = registered_looking
    stub = RecordingStub()
    monkeypatch.setattr(E, "run_episodes", stub)
    result = E.evaluate_run(str(directory), spec.to_dict())
    contracts.validate_evaluation(result, spec, directory)  # the launcher's check (pilot/launch.py evaluate)
    window = contracts.selection_window(contracts.checkpoint_steps(directory), spec.total_steps)
    assert sorted(result["selection"]) == window
    saved = torch.load(trained / "torch_save" / "epoch-1.pt", map_location="cpu", weights_only=True)
    policies = [policy for policy, _, _ in stub.calls]
    assert [p.step for p in policies] == window + [spec.total_steps]
    for policy in policies:  # load_policy ran for real on the trained checkpoint
        assert isinstance(policy, E.Policy) and policy.task == TASK and policy.obs_dim == 60
        assert not policy.budget_conditioned and not policy.injected
        assert all(torch.equal(v, saved["pi"][k]) for k, v in policy.actor.state_dict().items())
        assert int(policy.normalizer_state["_count"]) == int(saved["obs_normalizer"]["_count"]) > 1
    assert [seeds for _, seeds, _ in stub.calls] == [E.selection_seeds()] * R.SELECTION_WINDOW_CHECKPOINTS + [
        E.measurement_seeds()]
    assert result["final_cost"] == statistics.fmean(float(s % 9) for s in E.measurement_seeds())


def test_evaluate_battery_on_the_registered_looking_run_feeds_the_enrichment(registered_looking, monkeypatch) -> None:
    from pilot import enrichment

    spec, directory = registered_looking
    stub = RecordingStub()
    monkeypatch.setattr(E, "run_episodes", stub)
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset({"Q-hazard", "Q-dynamics"}))
    step = 9_400_000  # a checkpoint of the selection window
    costs = E.evaluate_battery(str(directory), spec.to_dict(), step, ["hazard", "dynamics"])
    gaps = enrichment.gaps_from_costs(costs, ["hazard", "dynamics"])  # equation (1)
    assert gaps["measurement_cost"] == costs["measurement"]
    seeds = contracts.validate_seed_set("measurement_seeds", costs["measurement_seeds"])
    assert not set(seeds) & set(E.selection_seeds())
    assert [(p.step, kw.get("condition", "in_distribution")) for p, _, kw in stub.calls] == [
        (step, "in_distribution"), (step, "hazard"), (step, "dynamics")]
    # the recorded definitions name what the conditions relocate and scale on this task (spec 7.7.2, 7.7.3)
    hazard, dynamics = costs["conditions"]["hazard"], costs["conditions"]["dynamics"]
    assert (hazard["task"], hazard["form"], hazard["hazards"], hazard["size"]) == (TASK, E.HAZARD_FORM, 8, 0.2)
    assert hazard["centre_half_width"] == pytest.approx(0.57, abs=1e-12)
    assert (dynamics["task"], dynamics["bodies"], dynamics["friction_geoms"]) == (TASK, ["agent"], ["floor", "agent", "pointarrow"])
    assert json.loads(json.dumps(costs["conditions"])) == costs["conditions"]


def test_evaluate_continuation_loads_the_continuations_final_checkpoint(trained, tmp_path, monkeypatch) -> None:
    parent = next(s for s in manifest.pilot() if s.study == "A" and s.onset_step == 0 and s.seed == 0)
    spec = manifest.RunSpec(
        run_id=f"{parent.arm_id}-finetune-s0", study="A", task=TASK, arm=parent.arm, seed=0, total_steps=20_000,
        base_algo="PPOLag", plugin="battery_finetune", group="battery_finetune", depends_on=(parent.run_id,))
    directory = _linked_run(trained, tmp_path, spec, [0, 20_000], SPE)  # tiny: 10 epochs of 2,000 steps
    stub = RecordingStub()
    monkeypatch.setattr(E, "run_episodes", stub)
    result = E.evaluate_continuation(str(directory), spec.to_dict())
    ((policy, seeds, kwargs),) = stub.calls
    assert isinstance(policy, E.Policy) and policy.step == 20_000 and policy.checkpoint.name == "epoch-10.pt"
    assert seeds == E.measurement_seeds() and kwargs == {"task": TASK}
    assert result["step"] == 20_000 and result["cost"] == statistics.fmean(result["episode_costs"])


# ---------------------------------------------------------------------------------------------
# Battery conditions on the trained policy
# ---------------------------------------------------------------------------------------------


def test_hazard_and_dynamics_episodes_of_a_trained_policy_reproduce(trained, monkeypatch) -> None:
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", R.ANSWERED_QUESTIONS | {"Q-hazard", "Q-dynamics"})  # the gates pass
    policy = E.load_policy(trained, SPE)
    layout, (s0, s1, *_) = E.hazard_layout_seeds()[0], E.hazard_episode_seeds()[0]
    hazard = E.run_episodes(policy, [s0, s1], condition="hazard", hazard_layout_seed=layout)
    hazard_again = E.run_episodes(policy, [s1], condition="hazard", hazard_layout_seed=[layout])
    registered = E.run_episodes(policy, [s0], condition="hazard", hazard_layout_seed=layout, hazard_form="registered")
    dynamics = E.run_episodes(policy, E.measurement_seeds()[:1], condition="dynamics")
    dynamics_again = E.run_episodes(policy, E.measurement_seeds()[:1], condition="dynamics")
    in_distribution = E.run_episodes(policy, E.measurement_seeds()[:1])
    assert hazard[1] == hazard_again[0] and dynamics == dynamics_again
    assert all(r.length == R.EPISODE_LENGTH for r in hazard + registered + dynamics + in_distribution)
    assert registered[0] != hazard[0]  # the two readings of Table 2.2 give different episodes
    assert dynamics[0] != in_distribution[0]  # the perturbation changes the episode of the same seed


def test_the_central_form_gives_100_distinct_layouts_with_hazards_pinned_per_layout(trained) -> None:
    policy = E.load_policy(trained, SPE)
    bound = E.hazard_definition(TASK)["centre_half_width"]  # 0.57: the recorded bound, not the square's 0.75
    complete, hazard_sets = [], set()
    with contextlib.ExitStack() as stack:  # whatever was built is closed, even if building the next one fails
        env = E.make_eval_env(policy, condition="hazard")
        stack.callback(env.close)
        drawer = E._HazardDrawer(TASK, "central")
        stack.callback(drawer.close)
        for layout_seed, episode_seeds in zip(E.hazard_layout_seeds(), E.hazard_episode_seeds()):
            hazards = drawer.draw(layout_seed)
            hazard_sets.add(tuple(hazards))
            env.pin_hazards(hazards)
            for seed in episode_seeds:
                env.env.reset(seed=seed)
                layout = env.task.world_info.layout
                placed = [tuple(layout[f"hazard{i}"]) for i in range(len(hazards))]
                assert np.allclose(placed, hazards, rtol=0.0, atol=1e-8)  # pinned for the layout's five episodes
                assert all(abs(x) <= bound + 1e-8 and abs(y) <= bound + 1e-8 for x, y in placed)
                complete.append((tuple(hazards), tuple(layout["agent"]), tuple(layout["goal"]), tuple(layout["vase0"])))
    assert len(hazard_sets) == R.HAZARD_LAYOUTS  # twenty hazard layouts
    assert len(complete) == len(set(complete)) == R.EVAL_EPISODES  # 100 distinct complete layouts
    reached = max(max(abs(x), abs(y)) for hazards in hazard_sets for x, y in hazards)
    assert bound - 0.01 < reached <= bound + 1e-8 < E.HAZARD_CENTRAL_HALF_WIDTH  # 0.5696 measured over the 20 seeds

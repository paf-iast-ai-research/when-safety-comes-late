"""Study B's plug-ins on tiny real runs (``slow``; 2,000 steps per epoch, at most 3 epochs per run).

* A Single-10 arm: the per-level J_C of its one level is OmniSafe's ``Metrics/EpCost`` window every
  epoch, and its multiplier follows PPOLag's update of that J_C exactly (Q-level-jc, Q-jc-window).
* A Moderate arm (the pilot's Study B arm): one ``Metrics/LagrangeMultiplier/level_{b:g}`` column per
  training level and no plain multiplier column (contract 3; Part 3.4), the launcher's non-finite
  check and the ledger writer's per-level reading pass, every checkpoint holds the per-level state
  and loads with ``weights_only=True``, and each episode that entered the buffer carried its own
  budget draw (Table 2.5).
* A few-shot continuation of that Moderate run (Table 2.5; pilot.dependencies.restore_learner;
  pilot.contracts.extra_checkpoint_steps): ``epoch-0.pt`` holds the parent's final learner, the
  multiplier starts fresh at 0.001 at the unseen budget, optimisers and schedule are fresh, the extra
  checkpoints are saved, and every observation carries the budget.
* The evaluations of ``studyb.evaluation`` on these real checkpoints (``load_policy`` for real), in
  registered-looking run directories whose checkpoints link to the tiny ones; ``run_episodes`` is
  stubbed except for a few real episodes.

The algorithms are built directly (``pilot.launch.build_config`` shrunk to tiny epochs). Run with ``pytest -m slow``.
"""

from __future__ import annotations

import contextlib
import csv
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from configs import registered as R
from pilot import contracts, dependencies, errors, launch, ledger_writer, manifest, provenance

pytestmark = [pytest.mark.omnisafe, pytest.mark.slow]

TINY = 2_000  # steps per epoch
COMMIT = "b" * 40


@contextlib.contextmanager
def _one_torch_thread():
    """Table 3.1 "one torch thread", as the launcher sets it (pilot/launch.py), restored afterwards.

    Entered by the module-scoped fixtures that train: pytest sets up a module-scoped fixture before
    any function-scoped one, so the autouse fixture below does not cover them, and the tiny runs
    would otherwise train with the process's default thread count (several times slower on a loaded
    machine).
    """
    import torch

    threads = torch.get_num_threads()
    torch.set_num_threads(R.TORCH_THREADS)
    try:
        yield
    finally:
        torch.set_num_threads(threads)


@pytest.fixture(autouse=True)
def _one_thread():
    """Each test body with one thread too; ``envs.evaluation`` sets it for good, this restores it."""
    with _one_torch_thread():
        yield


def _tiny(cfgs, *, epochs: int, extra: list[int]):
    cfgs.algo_cfgs.steps_per_epoch = TINY
    cfgs.train_cfgs.total_steps = TINY * epochs
    cfgs.train_cfgs.epochs = epochs
    cfgs.logger_cfgs.save_model_freq = 10  # OmniSafe saves epoch 0 and the last; the rest are extra saves
    cfgs.logger_cfgs.use_tensorboard = False
    cfgs.pilot_cfgs.extra_checkpoint_steps = extra
    return cfgs


def _omnisafe_dir(run_dir: Path) -> Path:
    (path,) = [p for p in (run_dir / "omnisafe").glob("*/seed-*") if p.is_dir()]
    return path


def _rows(omni: Path) -> list[dict[str, str]]:
    with open(omni / "progress.csv", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _finish(run_dir: Path, spec, cfgs) -> None:
    """What the launcher writes after a completed run (the fields ``dependencies.resolve`` reads)."""
    (run_dir / "spec.json").write_text(spec.to_json(), encoding="utf-8")
    (run_dir / "train_result.json").write_text(json.dumps({
        "run_id": spec.run_id, "status": "completed", "commit_hash": COMMIT,
        "config_hash": provenance.config_hash(cfgs.todict()),
        "omnisafe_dir": str(_omnisafe_dir(run_dir).relative_to(run_dir)),
    }), encoding="utf-8")


def _study_b(arm: str, seed: int = 0):
    return next(s for s in manifest.study_b() if s.arm == arm and s.seed == seed)


def _learn(algo) -> list[int]:
    """``algo.learn()``, recording the torch thread count at every epoch's update (inside training)."""
    import torch

    threads: list[int] = []
    update = algo._update

    def recorded() -> None:
        threads.append(torch.get_num_threads())
        update()

    algo._update = recorded  # an instance attribute: learn() calls self._update() once per epoch
    try:
        algo.learn()
    finally:
        del algo._update
    return threads


def _train(spec, run_dir: Path, *, epochs: int, extra: list[int]):
    """Build and train a tiny run with one torch thread; returns the algorithm, its cfgs and the thread
    counts recorded during training (one per epoch)."""
    from studyb import conditioning

    with _one_torch_thread():
        cfgs = _tiny(launch.build_config(spec, run_dir / "omnisafe"), epochs=epochs, extra=extra)
        algo = conditioning.make_algorithm(spec.task, cfgs, spec)
        threads = _learn(algo)
    return algo, cfgs, threads


@pytest.fixture(scope="module")
def single10(tmp_path_factory):
    spec = _study_b("Single-10")
    run_dir = tmp_path_factory.mktemp("single10") / spec.run_id
    algo, _, threads = _train(spec, run_dir, epochs=2, extra=[])
    return SimpleNamespace(spec=spec, algo=algo, omni=_omnisafe_dir(run_dir), threads=threads)


@pytest.fixture(scope="module")
def moderate(tmp_path_factory):
    """B-Moderate-s0 under a data root's checkpoints/ directory, as the scheduler lays runs out."""
    root = tmp_path_factory.mktemp("data") / "checkpoints"
    spec = _study_b("Moderate")
    run_dir = root / spec.run_id
    algo, cfgs, threads = _train(spec, run_dir, epochs=2, extra=[TINY])
    _finish(run_dir, spec, cfgs)
    return SimpleNamespace(spec=spec, algo=algo, cfgs=cfgs, root=root, run_dir=run_dir, omni=_omnisafe_dir(run_dir),
                           threads=threads)


@pytest.fixture(scope="module")
def fewshot(moderate):
    """The budget-5 continuation of the tiny Moderate run from its final checkpoint (after 2 epochs).

    The spec is the registered continuation after Part 6.1 cut 3 (one horizon; its total equals it,
    so the launcher's extra checkpoints are none), with ``parent_step`` the tiny parent's final step;
    the tiny configuration asks for extra checkpoints (two here), as the launcher does for the off-grid
    horizon (500,000) of an uncut continuation (pilot.contracts.extra_checkpoint_steps).
    """
    import torch

    from studyb import conditioning

    registered = next(s for s in manifest.study_b_fewshot([moderate.spec]) if s.params["budget"] == 5.0)
    cut = manifest.apply_cuts([registered], manifest.CUTS[:3])[0]
    spec = manifest.RunSpec.from_dict({**cut.to_dict(), "params": {**cut.params, "parent_step": 2 * TINY}})
    run_dir = moderate.root / spec.run_id
    deps = dependencies.resolve(spec, run_dir, smoke=False)
    with _one_torch_thread():
        cfgs = _tiny(launch.build_config(spec, run_dir / "omnisafe", deps), epochs=3, extra=[TINY, 2 * TINY])
        algo = conditioning.make_fewshot_algorithm(spec.task, cfgs, spec)
        omni = Path(algo._logger.log_dir)
        start = torch.load(omni / "torch_save" / "epoch-0.pt", weights_only=True)  # before any training
        fresh = {"lr": algo._actor_critic.actor_scheduler.get_last_lr()[0],
                 "states": [len(o.state) for o in (algo._actor_critic.actor_optimizer,
                                                   algo._actor_critic.reward_critic_optimizer,
                                                   algo._actor_critic.cost_critic_optimizer)]}
        threads = _learn(algo)
    return SimpleNamespace(spec=spec, registered=registered, algo=algo, cfgs=cfgs, omni=omni, start=start, fresh=fresh,
                           threads=threads)


def test_the_tiny_runs_train_with_one_torch_thread(single10, moderate, fewshot) -> None:
    """Table 3.1 "one torch thread" at every epoch's update of every module-scoped training fixture
    (they are set up before any function-scoped fixture, so the autouse one cannot provide it)."""
    one = R.TORCH_THREADS
    assert [single10.threads, moderate.threads, fewshot.threads] == [[one] * 2, [one] * 2, [one] * 3]


# ---------------------------------------------------------------------------
# One level: PPOLag's J_C and PPOLag's multiplier
# ---------------------------------------------------------------------------


def test_a_single_level_updates_from_omnisafes_window_as_ppolag_does(single10) -> None:
    from omnisafe.common.lagrange import Lagrange

    rows = _rows(single10.omni)
    assert len(rows) == 2
    column = "Metrics/LagrangeMultiplier/level_10"
    assert [c for c in rows[0] if c.startswith(launch.MULTIPLIER_COLUMN)] == [column]
    replica = Lagrange(**{**single10.algo._cfgs.lagrange_cfgs.todict(), "cost_limit": 10.0})
    for row in rows:
        assert row["Metrics/LevelJc/level_10"] == row["Metrics/EpCost"]  # the same float, printed the same way
        assert float(row["Metrics/LevelWindow/level_10"]) == 2 * (float(row["Train/Epoch"]) + 1)  # two episodes per epoch
        replica.update_lagrange_multiplier(float(row["Metrics/EpCost"]))  # PPOLag: Jc = get_stats('Metrics/EpCost')[0]
        assert float(row[column]) == replica.lagrangian_multiplier.item()
    assert ledger_writer.per_level_multipliers(rows) == {10.0: float(rows[-1][column])}
    assert launch._nonfinite(rows, launch.MULTIPLIER_COLUMN) == [] and launch._nonfinite(rows, launch.LOSS_PREFIX) == []


# ---------------------------------------------------------------------------
# The Moderate arm
# ---------------------------------------------------------------------------


def test_the_moderate_run_logs_exactly_its_levels(moderate) -> None:
    rows = _rows(moderate.omni)
    header = list(rows[0])
    assert ledger_writer.multiplier_columns(header) == [f"Metrics/LagrangeMultiplier/level_{b}" for b in (10, 20, 40)]
    assert launch.MULTIPLIER_COLUMN not in header  # no plain column: nothing would be left NaN (Study B spec E2 and E3)
    levels = sorted(ledger_writer._level(c) for c in ledger_writer.multiplier_columns(header))
    assert levels == sorted(moderate.spec.training_levels)  # the ledger writer's check (ledger_writer.py)
    per_level = ledger_writer.per_level_multipliers(rows)
    assert set(per_level) == {10.0, 20.0, 40.0} and all(v >= 0 for v in per_level.values())
    assert launch._nonfinite(rows, launch.MULTIPLIER_COLUMN) == [] and launch.nonfinite_state(moderate.algo) is None
    for row in rows:  # the window of each level holds exactly that level's episodes
        episodes = sum(float(row[f"Metrics/LevelEpisodes/level_{b}"]) for b in (10, 20, 40))
        assert episodes == 2.0
        for b in (10, 20, 40):
            if float(row[f"Metrics/LevelWindow/level_{b}"]):
                assert float(row[f"Metrics/LevelBudget/level_{b}"]) == b


def test_each_episode_in_the_buffer_carried_its_own_draw(moderate) -> None:
    """Four episodes in two epochs, four draws used; the episode the last auto-reset began took no step
    and holds the next draw (no draw is lost at an epoch's end)."""
    from studyb import conditioning

    adapter = moderate.algo._env
    assert adapter.budget_draws == 5
    source = conditioning.make_budget_source(conditioning.DISCRETE, (10.0, 20.0, 40.0), moderate.spec.seed)
    draws = [source() for _ in range(5)]
    windows = [e[2] for e in moderate.algo._level_window]
    assert windows == draws[:4] and adapter.episode_budget == draws[4]
    feature = moderate.algo._buf.buffers[0].data["obs"][:, -1]  # the last epoch: episodes 3 and 4
    assert feature.numel() == 2 * R.EPISODE_LENGTH
    assert bool((feature[:R.EPISODE_LENGTH] == conditioning.budget_feature(draws[2])).all())
    assert bool((feature[R.EPISODE_LENGTH:] == conditioning.budget_feature(draws[3])).all())


def test_every_checkpoint_holds_the_level_state(moderate) -> None:
    import torch

    names = sorted(p.name for p in (moderate.omni / "torch_save").glob("*.pt"))
    assert names == ["epoch-0.pt", "epoch-1.pt", "epoch-2.pt"]  # epoch 1 is the extra checkpoint (pilot.contracts.extra_checkpoint_steps)
    rows = _rows(moderate.omni)
    for epoch in (1, 2):
        state = torch.load(moderate.omni / "torch_save" / f"epoch-{epoch}.pt", weights_only=True)
        assert set(dependencies.FULL_STATE_KEYS) <= set(state)
        levels = state["level_lagranges"]
        logged = [float(rows[epoch - 1][f"Metrics/LagrangeMultiplier/level_{b}"]) for b in (10, 20, 40)]
        assert levels["keys"].tolist() == [10.0, 20.0, 40.0] and levels["values"].tolist() == logged
        window = state["level_window"]
        assert window["cost"].tolist() == state["episode_windows"]["Metrics/EpCost"].tolist()  # the same episodes
        assert len(window["index"]) == 2 * epoch and window["maxlen"] == 50
        assert set(state["level_lambda_optimizers"]) == {"10", "20", "40"}
        assert state["pi"]["mean.0.weight"].shape[1] == 61 and state["obs_normalizer"]["_mean"].shape == (60,)


# ---------------------------------------------------------------------------
# The few-shot continuation
# ---------------------------------------------------------------------------


def test_the_continuation_starts_from_the_parents_final_learner(moderate, fewshot) -> None:
    import torch

    parent = torch.load(moderate.omni / "torch_save" / "epoch-2.pt", weights_only=True)
    for key in ("pi", "reward_critic", "cost_critic", "obs_normalizer"):
        assert fewshot.start[key].keys() == parent[key].keys()
        assert all(torch.equal(fewshot.start[key][k], parent[key][k]) for k in parent[key]), key
    assert fewshot.fresh == {"lr": fewshot.algo._cfgs.model_cfgs.actor.lr, "states": [0, 0, 0]}  # fresh optimisers, schedule
    assert fewshot.start["actor_scheduler"]["last_epoch"] == 0
    levels = fewshot.start["level_lagranges"]
    assert levels["kind"] == "fewshot" and levels["keys"].tolist() == [5.0] and levels["cost_limits"].tolist() == [5.0]
    assert levels["values"].tolist() == [pytest.approx(R.LAGRANGE_MULTIPLIER_INIT)]  # "a fresh multiplier at 0.001"
    record = json.loads((fewshot.omni / "continuation.json").read_text())
    assert record["restore"]["parent_run_id"] == moderate.spec.run_id and record["restore"]["parent_step"] == 2 * TINY
    assert record["restore"]["restored"] == ["pi", "reward_critic", "cost_critic", "obs_normalizer"]
    sha = fewshot.cfgs.pilot_cfgs.dependencies[moderate.spec.run_id]["checkpoint_sha256"]
    assert record["restore"]["parent_checkpoint_sha256"] == sha
    assert record["budget"] == 5.0 and record["horizons"] == [200_000]
    # cut 3 keeps the full continuation's 1,000,000-step schedule (Table 9.1, Q-continuations), in this run's epochs
    assert record["restore"]["actor_schedule"] == {"type": "LinearLR", "total_iters": R.FEWSHOT_STEPS // TINY,
                                                   "schedule_steps": R.FEWSHOT_STEPS,
                                                   "lr": fewshot.algo._cfgs.model_cfgs.actor.lr}
    assert record["multiplier"] == {"init": pytest.approx(R.LAGRANGE_MULTIPLIER_INIT), "cost_limit": 5.0,
                                    "lambda_lr": R.LAGRANGE_MULTIPLIER_LR, "optimizer": "Adam"}


def test_the_continuation_trains_under_its_one_budget(fewshot) -> None:
    import torch

    rows = _rows(fewshot.omni)
    assert len(rows) == 3
    assert ledger_writer.multiplier_columns(list(rows[0])) == ["Metrics/LagrangeMultiplier/level_5"]
    assert all(float(r["Metrics/LevelBudget/level_5"]) == 5.0 for r in rows)
    assert launch._nonfinite(rows, launch.MULTIPLIER_COLUMN) == []
    assert sorted(p.name for p in (fewshot.omni / "torch_save").glob("*.pt")) == [f"epoch-{k}.pt" for k in range(4)]
    adapter = fewshot.algo._env
    assert adapter.budget_draws == 7 and adapter.episode_budget == 5.0  # six episodes and the pending seventh, all at 5
    feature = fewshot.algo._buf.buffers[0].data["obs"][:, -1]
    assert bool((feature == torch.tensor(5.0 / R.BUDGET_OBSERVATION_DIVISOR, dtype=torch.float32)).all())
    assert {e[2] for e in fewshot.algo._level_window} == {5.0} and len(fewshot.algo._level_window) == 6


def _continuation_of(moderate, budget: float, parent_step: int = 2 * TINY):
    """The cut continuation of the tiny Moderate run at ``budget`` from its final checkpoint, with the
    launcher's configuration (its parent resolved); ``parent_step`` another step for a refusal."""
    registered = next(s for s in manifest.study_b_fewshot([moderate.spec]) if s.params["budget"] == budget)
    cut = manifest.apply_cuts([registered], manifest.CUTS[:3])[0]
    spec = manifest.RunSpec.from_dict({**cut.to_dict(), "params": {**cut.params, "parent_step": parent_step}})
    run_dir = moderate.root / spec.run_id
    deps = dependencies.resolve(spec, run_dir, smoke=False)
    return spec, run_dir, _tiny(launch.build_config(spec, run_dir / "omnisafe", deps), epochs=1, extra=[])


def test_a_continuation_refuses_a_parent_step_that_is_not_the_final_checkpoint(moderate) -> None:
    from studyb import conditioning

    # epoch-1.pt exists: the launcher would resolve it
    spec, run_dir, cfgs = _continuation_of(moderate, 15.0, parent_step=TINY)
    with pytest.raises(errors.RunRefused, match="is not the parent's final checkpoint"):
        conditioning.make_fewshot_algorithm(spec.task, cfgs, spec)
    assert not (run_dir / "omnisafe").exists()


PARENT_PROBLEMS = {  # a change of the parent's spec.json or config.json -> the words of the refusal
    "plugin": (lambda s, c: ({**s, "plugin": "ppolag"}, c), "is plug-in 'ppolag', not a Study B training run"),
    "arm": (lambda s, c: ({**s, "arm": "Dense"}, c), "has arm 'Dense'"),
    "seed": (lambda s, c: ({**s, "seed": 1, "run_id": "B-Moderate-s1"}, c), "has seed 1"),
    "task": (lambda s, c: ({**s, "task": "SafetyPointButton1-v0"}, c), "has task 'SafetyPointButton1-v0'"),
    "levels": (lambda s, c: ({**s, "training_levels": [10.0, 40.0]}, c), "trains on (10.0, 40.0)"),
    "no studyb_cfgs": (lambda s, c: (s, {k: v for k, v in c.items() if k != "studyb_cfgs"}),
                       "was not trained by studyb.conditioning"),
    "divisor": (lambda s, c: (s, {**c, "studyb_cfgs": {k: v for k, v in c["studyb_cfgs"].items()
                                                       if k != "budget_divisor"}}),
                "was not trained by studyb.conditioning"),
    "spec.json": (lambda s, c: (None, c), "its spec.json or config.json cannot be read"),
    "config.json": (lambda s, c: (s, None), "its spec.json or config.json cannot be read"),
    "config.json a list": (lambda s, c: (s, []), "config.json is not a JSON object"),
    "train_cfgs a number": (lambda s, c: (s, {**c, "train_cfgs": 5}), "is not the parent's final checkpoint"),
}


@pytest.mark.parametrize("problem", sorted(PARENT_PROBLEMS))
def test_a_continuation_refuses_a_parent_that_is_not_its_study_b_run(moderate, tmp_path, problem) -> None:
    """``_parent_problems``: the resolved parent's spec.json and config.json, altered copies in a sibling
    directory that the configuration's dependency entry points to."""
    from studyb import conditioning

    spec, run_dir, cfgs = _continuation_of(moderate, 15.0)
    parent_spec = json.loads((moderate.run_dir / "spec.json").read_text())
    parent_config = json.loads((moderate.omni / "config.json").read_text())
    change, words = PARENT_PROBLEMS[problem]
    new_spec, new_config = change(parent_spec, parent_config)
    fake = tmp_path / "parent"
    (fake / "omni").mkdir(parents=True)
    if new_spec is not None:
        (fake / "spec.json").write_text(json.dumps(new_spec))
    (fake / "omni" / "config.json").write_text("{not json" if new_config is None else json.dumps(new_config))
    entry = cfgs.pilot_cfgs.dependencies[spec.params["parent_run_id"]]
    entry["run_dir"], entry["omnisafe_dir"] = str(fake), str(fake / "omni")
    with pytest.raises(errors.RunRefused, match=re.escape(words)):
        conditioning.make_fewshot_algorithm(spec.task, cfgs, spec)
    assert not (run_dir / "omnisafe").exists()


def test_a_continuation_refuses_at_init_without_its_spec_or_its_parent(moderate, monkeypatch) -> None:
    """The run-time refusals of ``FewShotPPOLag._init``: no spec (not built by make_fewshot_algorithm)
    and no dependency entry for the parent."""
    from studyb import conditioning

    spec, _, cfgs = _continuation_of(moderate, 15.0)
    monkeypatch.setattr(conditioning, "_parent_problems", lambda run, cfgs: [])  # reach _init
    cfgs.pilot_cfgs.dependencies = {}
    with pytest.raises(errors.RunRefused, match="has no entry for the parent"):
        conditioning.make_fewshot_algorithm(spec.task, cfgs, spec)
    assert cfgs.studyb_cfgs.kind == conditioning.FEWSHOT  # set before the algorithm was built
    with pytest.raises(errors.RunRefused, match="is built by studyb.conditioning.make_fewshot_algorithm"):
        conditioning._classes()["FewShotPPOLag"](env_id=spec.task, cfgs=cfgs)


# ---------------------------------------------------------------------------
# The evaluations on the real checkpoints
# ---------------------------------------------------------------------------


def _linked(source: Path, root: Path, spec, steps: list[int], checkpoint: Path) -> Path:
    """A run directory for ``spec`` (registered epochs and total) whose checkpoints all link to ``checkpoint``."""
    directory = root / spec.run_id / "omnisafe" / f"PPOLag-{{{spec.task}}}" / "seed-000-fixture"
    (directory / "torch_save").mkdir(parents=True)
    cfg = json.loads((source / "config.json").read_text(encoding="utf-8"))
    cfg["train_cfgs"].update({"total_steps": spec.total_steps, "epochs": spec.total_steps // R.STEPS_PER_EPOCH})
    cfg["algo_cfgs"]["steps_per_epoch"] = R.STEPS_PER_EPOCH
    cfg["pilot_cfgs"]["run_id"] = spec.run_id
    (directory / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    for step in steps:
        (directory / "torch_save" / f"epoch-{step // R.STEPS_PER_EPOCH}.pt").symlink_to(checkpoint)
    return directory


class Stub:
    """``run_episodes`` replaced by a fast stub that records the real policies it is given."""

    def __init__(self) -> None:
        self.calls = []

    def __call__(self, policy, seeds, *, budgets=None, **kwargs):
        from envs import evaluation as E

        self.calls.append((policy, list(seeds), list(budgets)))
        return [E.EpisodeResult(seed=s, cost=float(s % 13), ret=0.5, length=R.EPISODE_LENGTH, budget=float(b))
                for s, b in zip(seeds, budgets)]


@pytest.fixture
def stub(monkeypatch):
    from envs import evaluation as E

    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset({"Q-studyb-eval"}))
    recorder = Stub()
    monkeypatch.setattr(E, "run_episodes", recorder)
    return recorder


def test_contract_1_on_real_checkpoints(moderate, stub, tmp_path) -> None:
    from studyb import evaluation as SE

    spec = next(s for s in manifest.pilot() if s.arm == "Moderate")
    directory = _linked(moderate.omni, tmp_path, spec, contracts.expected_checkpoint_steps(spec),
                        moderate.omni / "torch_save" / "epoch-2.pt")
    result = SE.evaluate_run(str(directory), spec.to_dict())
    contracts.validate_evaluation(launch.plain(result), spec, directory)
    policies = [call[0] for call in stub.calls]
    assert all(p.budget_conditioned and p.actor.mean[0].in_features == 61 for p in policies)
    assert [p.step for p in policies] == list(range(8_200_000, 10_000_001, 200_000)) + [10_000_000]


def test_zero_shot_training_budgets_and_fewshot_on_real_checkpoints(moderate, fewshot, stub, tmp_path) -> None:
    from studyb import evaluation as SE

    parent = _linked(moderate.omni, tmp_path / "p", moderate.spec, [moderate.spec.total_steps],
                     moderate.omni / "torch_save" / "epoch-2.pt")
    zero = SE.evaluate_zero_shot(str(parent), moderate.spec.to_dict())
    assert sorted(zero["sr_zero"]) == [5.0, 15.0, 30.0, 45.0] and zero["reference_budgets"] == [10.0, 20.0, 40.0]
    g4 = SE.evaluate_training_budgets(str(parent), moderate.spec.to_dict())
    assert sorted(g4["mean_cost"]) == [10.0, 20.0, 40.0]
    registered = fewshot.registered  # the uncut continuation: three horizons, 500,000 off the grid
    child = _linked(fewshot.omni, tmp_path / "c", registered, contracts.expected_checkpoint_steps(registered),
                    fewshot.omni / "torch_save" / "epoch-3.pt")
    result = SE.evaluate_fewshot(str(child), registered.to_dict())
    assert result["horizons"] == list(R.FEWSHOT_HORIZONS) and list(result["sr_fewshot"]) == [
        contracts.fewshot_key(5.0, h) for h in R.FEWSHOT_HORIZONS]
    assert [c[0].checkpoint.name for c in stub.calls[-3:]] == ["epoch-10.pt", "epoch-25.pt", "epoch-50.pt"]
    assert all(set(c[2]) == {5.0} for c in stub.calls[-3:])


def test_real_episodes_carry_the_budget_and_repeat_exactly(moderate, monkeypatch) -> None:
    """Role 2's episode loop with a Study B policy: the budget reaches the actor, the frozen normaliser
    and seeded resets make a second run identical (in the spirit of Table 3.1 "Determinism check")."""
    import torch

    from envs import evaluation as E

    policy = E.load_policy(moderate.omni, 2 * TINY)
    assert policy.budget_conditioned and policy.normalizer_state["_mean"].shape == (60,)
    before = {k: v.clone() for k, v in policy.normalizer_state.items()}
    seeds = E.measurement_seeds()[:1]
    first = E.run_episodes(policy, seeds, budgets=[45.0])
    again = E.run_episodes(policy, seeds, budgets=[45.0])
    assert first == again and first[0].budget == 45.0 and first[0].length == R.EPISODE_LENGTH
    assert all(torch.equal(before[k], policy.normalizer_state[k]) for k in before)
    seen = []
    forward = policy.actor.predict

    def recording(obs, deterministic=False):
        seen.append(obs[..., -1].clone())
        return forward(obs, deterministic=deterministic)

    monkeypatch.setattr(policy.actor, "predict", recording)
    E.run_episodes(policy, seeds, budgets=[15.0])
    assert len(seen) == R.EPISODE_LENGTH and all(float(x) == float(torch.tensor(0.15, dtype=torch.float32)) for x in seen)

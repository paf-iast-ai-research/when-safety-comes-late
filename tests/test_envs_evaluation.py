"""The evaluation harness ``envs.evaluation`` (Role 2), fast tests without the simulation stack.

Seed sets (Table 2.1, Table 2.2; Q-eval-seeds), ``append_budget`` (Table 2.5), the result gates
(Q-hazard, Q-dynamics; HANDOVER.md section 10), argument checks, and the orchestration of the contract
functions (pilot/contracts.py 1, 5, 6; HANDOVER.md section 8; pilot/scheduler.py) with ``load_policy``, ``run_episodes``,
``hazard_definition`` and ``dynamics_definition`` replaced by stubs on a fake OmniSafe run directory
(config.json and empty checkpoint files), the classes of refusals and failures (HANDOVER.md section 8;
harness spec 7.10), and the Study B budget schedule (Q-studyb-eval). The tests that run the
simulator are in ``test_envs_evaluation_sim.py`` (``omnisafe``) and ``test_envs_evaluation_slow.py``
(``slow``).
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
import statistics
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from configs import registered as R
from envs import evaluation as E
from pilot import contracts, errors, manifest
from pilot.errors import PendingQuestionError, RunRefused

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _restore_threads():
    threads = torch.get_num_threads()
    yield
    torch.set_num_threads(threads)


def _pilot_spec(onset: int = 5_000_000, seed: int = 0) -> manifest.RunSpec:
    return next(s for s in manifest.pilot() if s.study == "A" and s.onset_step == onset and s.seed == seed)


def _fake_run(root: Path, spec: manifest.RunSpec, *, steps: list[int] | None = None, **overrides) -> Path:
    """An OmniSafe run directory as the launcher leaves it: config.json and torch_save/epoch-k.pt names."""
    directory = root / "omnisafe" / f"{spec.base_algo}-{{{spec.task}}}" / "seed-000-fixture"
    (directory / "torch_save").mkdir(parents=True)
    cfg = {
        "env_id": spec.task,
        "algo": spec.base_algo,
        "seed": spec.seed,
        "train_cfgs": {"total_steps": spec.total_steps, "epochs": spec.total_steps // R.STEPS_PER_EPOCH},
        "algo_cfgs": {"steps_per_epoch": R.STEPS_PER_EPOCH, "obs_normalize": True, "reward_normalize": False,
                      "cost_normalize": False},
        "env_cfgs": {},
        "model_cfgs": {"actor": {"hidden_sizes": list(R.HIDDEN_SIZES), "activation": R.ACTIVATION}},
        "pilot_cfgs": {"run_id": spec.run_id, "onset_step": int(spec.onset_step or 0)},
    }
    for key, value in overrides.items():  # "algo_cfgs.reward_normalize" -> cfg["algo_cfgs"]["reward_normalize"]
        node = cfg
        *path, last = key.split(".")
        for part in path:
            node = node[part]
        node[last] = value
    (directory / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    for step in contracts.expected_checkpoint_steps(spec) if steps is None else steps:
        (directory / "torch_save" / f"epoch-{step // R.STEPS_PER_EPOCH}.pt").write_bytes(b"")
    return directory


class Stubs:
    """``load_policy``, ``run_episodes`` and the two condition definitions replaced by recorders.

    Every episode runs 1,000 steps. ``fail_at``: steps whose checkpoint ``load_policy`` finds
    unreadable (``CheckpointUnreadable``). ``definitions`` records ``(condition, task, run_episodes
    calls made so far)`` for each definition built.
    """

    def __init__(self, monkeypatch, *, budget_conditioned: bool = False, fail_at: tuple[int, ...] = ()) -> None:
        self.loads: list[tuple[Path, int, object]] = []
        self.calls: list[dict] = []
        self.definitions: list[tuple[str, str, int]] = []
        self.budget_conditioned = budget_conditioned
        self.fail_at = fail_at
        monkeypatch.setattr(E, "load_policy", self.load_policy)
        monkeypatch.setattr(E, "run_episodes", self.run_episodes)
        monkeypatch.setattr(E, "hazard_definition", self.hazard_definition)
        monkeypatch.setattr(E, "dynamics_definition", self.dynamics_definition)

    def hazard_definition(self, task):
        self.definitions.append(("hazard", task, len(self.calls)))
        return {"proposal_key": "Q-hazard", "form": E.HAZARD_FORM, "task": task, "hazards": 8, "size": 0.2,
                "layout_seeds": E.hazard_layout_seeds()}

    def dynamics_definition(self, task):
        self.definitions.append(("dynamics", task, len(self.calls)))
        return {"proposal_key": "Q-dynamics", "task": task, "bodies": ["agent"], "friction_geoms": ["floor", "agent"],
                "body_mass_scale": R.BODY_MASS_SCALE, "ground_friction_scale": R.GROUND_FRICTION_SCALE}

    def load_policy(self, omnisafe_dir, step, spec=None):
        self.loads.append((Path(omnisafe_dir), step, spec))
        if step in self.fail_at:
            raise E.CheckpointUnreadable(f"epoch-{step // R.STEPS_PER_EPOCH}.pt cannot be loaded")
        config = Path(omnisafe_dir) / "config.json"
        return SimpleNamespace(step=step, budget_conditioned=self.budget_conditioned, task="SafetyPointGoal1-v0",
                               normalizer_state={},
                               config=json.loads(config.read_text()) if config.is_file() else {},
                               checkpoint=Path(omnisafe_dir) / "torch_save" / f"epoch-{step // R.STEPS_PER_EPOCH}.pt")

    def run_episodes(self, policy, seeds, **kwargs):
        self.calls.append({"step": policy.step, "seeds": list(seeds), **kwargs})
        bump = {"in_distribution": 0.0, "hazard": 3.0, "dynamics": 1.0}[kwargs.get("condition", "in_distribution")]
        budgets = kwargs.get("budgets") or [None] * len(seeds)
        return [E.EpisodeResult(seed=s, cost=float(s % 7) + bump + policy.step / 1e6, ret=(s % 5) / 4.0,
                                length=R.EPISODE_LENGTH, budget=b) for s, b in zip(seeds, budgets)]


# ---------------------------------------------------------------------------------------------
# Seed sets (Table 2.1; Table 2.2; Q-eval-seeds)
# ---------------------------------------------------------------------------------------------


def test_seed_sets_are_fixed_disjoint_and_of_the_registered_sizes() -> None:
    selection, measurement = E.selection_seeds(), E.measurement_seeds()
    layouts, episodes = E.hazard_layout_seeds(), E.hazard_episode_seeds()
    flat = [s for row in episodes for s in row]
    assert selection == list(range(1_000_000, 1_000_100)) and measurement == list(range(2_000_000, 2_000_100))
    assert layouts == list(range(3_000_000, 3_000_020))
    assert episodes[0] == [3_100_000, 3_100_001, 3_100_002, 3_100_003, 3_100_004] and episodes[-1][-1] == 3_100_099
    assert len(selection) == len(set(selection)) == R.EVAL_EPISODES
    assert len(measurement) == len(set(measurement)) == R.EVAL_EPISODES
    assert len(layouts) == len(set(layouts)) == R.HAZARD_LAYOUTS
    assert [len(row) for row in episodes] == [R.EPISODES_PER_LAYOUT] * R.HAZARD_LAYOUTS
    assert len(flat) == len(set(flat)) == R.HAZARD_LAYOUTS * R.EPISODES_PER_LAYOUT == R.EVAL_EPISODES  # Table 2.2
    sets = [set(selection), set(measurement), set(layouts), set(flat)]
    assert sum(len(s) for s in sets) == len(set().union(*sets))  # pairwise disjoint (Table 2.1)
    training = set(range(0, 1000))  # run seeds 0-4, surplus/replacement seeds from 5 (Parts 5.5, 5.6), the fixed batch's 0
    assert not training & set().union(*sets)
    assert max(set().union(*sets)) < 2**32  # np.random.RandomState
    assert all(type(s) is int for s in selection + measurement + layouts + flat)


def test_seed_sets_pass_the_pipeline_checks(tmp_path) -> None:
    assert contracts.validate_seed_set("selection_seeds", E.selection_seeds()) == E.selection_seeds()
    assert contracts.validate_seed_set("measurement_seeds", E.measurement_seeds()) == E.measurement_seeds()
    flat = [s for row in E.hazard_episode_seeds() for s in row]
    assert contracts.validate_seed_set("hazard episode seeds", flat) == flat
    registry = tmp_path / "ledger.seeds.json"
    contracts.check_canonical_seeds("selection", E.selection_seeds(), registry)
    contracts.check_canonical_seeds("measurement", E.measurement_seeds(), registry)  # disjoint: accepted
    contracts.check_canonical_seeds("selection", E.selection_seeds(), registry)  # the same set again: accepted
    assert json.loads(registry.read_text()) == {"selection": E.selection_seeds(), "measurement": E.measurement_seeds()}


def test_seed_functions_return_fresh_lists() -> None:
    first = E.selection_seeds()
    first.append(-1)
    E.hazard_episode_seeds()[0].append(-1)
    assert E.selection_seeds() == list(range(1_000_000, 1_000_100))
    assert E.hazard_episode_seeds()[0] == list(range(3_100_000, 3_100_005))


# ---------------------------------------------------------------------------------------------
# append_budget (Table 2.5 "Budget conditioning"; Q-budget-normalisation)
# ---------------------------------------------------------------------------------------------


def test_append_budget_appends_budget_over_the_registered_divisor_as_the_last_feature() -> None:
    obs = torch.arange(6, dtype=torch.float32).reshape(1, 6)
    out = E.append_budget(obs, 25)
    assert out.shape == (1, 7) and out.dtype == torch.float32
    assert torch.equal(out[:, :6], obs)
    assert out[0, 6].item() == torch.tensor(25 / R.BUDGET_OBSERVATION_DIVISOR, dtype=torch.float32).item()
    batch = torch.zeros(3, 4, 60, dtype=torch.float64)
    wide = E.append_budget(batch, 17.5)
    assert wide.shape == (3, 4, 61) and wide.dtype == torch.float64 and torch.all(wide[..., -1] == 0.175)
    for bad in (float("nan"), float("inf"), -1.0, True, "25"):
        with pytest.raises(ValueError):
            E.append_budget(obs, bad)


# ---------------------------------------------------------------------------------------------
# Module constants and imports
# ---------------------------------------------------------------------------------------------


DECIDED_CONSTANT = re.compile(r"^([A-Z][A-Z0-9_]*)\s*(?::[^=\n]+)?=[^\n#]*#\s*answered in Table 9\.1 \((Q-[A-Za-z0-9-]+)\)",
                              re.M)  # as tests/test_envs_onset.py checks envs/onset.py and envs/continuations.py


def test_constants_match_the_other_roles_and_the_answered_questions() -> None:
    from metrics import interventions
    from pilot import enrichment

    assert set(E.BATTERY_CONDITIONS) == set(enrichment.CONDITION_FIELDS)  # contract 5's conditions
    assert E.INJECTED_KEY == interventions.INJECTED_KEY  # HANDOVER.md section 8: injection detected by this key only
    assert E.HAZARD_FORM == "central" and E.HAZARD_CENTRAL_HALF_WIDTH == 0.75  # the Q-hazard answer (Table 9.1)
    assert E.HAZARD_FORMS == ("registered", "central")
    assert "0.75" in R.PENDING["Q-hazard"]  # the answer the constant implements
    for key in ("Q-hazard", "Q-dynamics"):
        assert key in R.PENDING and not R.is_open(key)  # answered (docs/DECISIONS.md, 2026-10-02)
    # HANDOVER.md section 10: a value Table 9.1 decides is a named constant commented "# answered in Table 9.1 (Q-key)"
    source = (REPO / "envs" / "evaluation.py").read_text(encoding="utf-8")
    found = {m.group(1): m.group(2) for m in DECIDED_CONSTANT.finditer(source)}
    assert found == {"HAZARD_FORM": "Q-hazard", "HAZARD_CENTRAL_HALF_WIDTH": "Q-hazard"}
    assert "# PROPOSAL (" not in source


def test_the_seed_bases_are_marked_as_the_q_eval_seeds_answer() -> None:
    """HANDOVER.md section 10: a decided value is a named constant commented with its key (Table 9.1)."""
    lines = (REPO / "envs" / "evaluation.py").read_text(encoding="utf-8").splitlines()
    marker = "# Q-eval-seeds (Table 9.1; not a PENDING key)"
    for name, value in [("SELECTION_SEED_BASE", 1_000_000), ("MEASUREMENT_SEED_BASE", 2_000_000),
                        ("HAZARD_LAYOUT_SEED_BASE", 3_000_000), ("HAZARD_EPISODE_SEED_BASE", 3_100_000)]:
        (line,) = [ln for ln in lines if ln.startswith(f"{name} = ")]
        assert line.endswith(marker), line
        assert getattr(E, name) == value  # the harness spec's section 10 values
    assert "Q-eval-seeds" not in R.PENDING or not R.is_open("Q-eval-seeds")  # never an open gate


def test_the_module_imports_without_the_simulation_stack() -> None:
    code = (
        "import sys\n"
        "for name in ('omnisafe', 'safety_gymnasium', 'mujoco', 'metrics', 'gymnasium'):\n"
        "    sys.modules[name] = None  # an import of any of them now raises ImportError\n"
        "from envs import evaluation as E\n"
        "assert len(E.selection_seeds()) == 100 and E.HAZARD_FORM == 'central'\n"
        "print('ok')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0 and result.stdout.strip() == "ok", result.stderr


def test_module_getattr_invents_no_other_names() -> None:
    with pytest.raises(AttributeError):
        E.NoSuchName  # noqa: B018  (module __getattr__ must not invent names)


# ---------------------------------------------------------------------------------------------
# Result gates (HANDOVER.md section 10; Q-hazard, Q-dynamics)
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("condition", ["hazard", "dynamics"])
def test_battery_gates_refuse_while_open_before_touching_anything(tmp_path, monkeypatch, condition) -> None:
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset())  # the keys open, whatever the repository answers
    stubs = Stubs(monkeypatch)
    assert not errors.pending_allowed()
    missing = tmp_path / "no-such-run"
    with pytest.raises(PendingQuestionError) as info:
        E.evaluate_battery(str(missing), _pilot_spec().to_dict(), 10_000_000, [condition])
    assert isinstance(info.value, RunRefused) and isinstance(info.value, ValueError)  # exit 5; CLI without traceback
    assert ("Q-hazard" if condition == "hazard" else "Q-dynamics") in str(info.value)
    assert stubs.loads == [] and stubs.calls == [] and stubs.definitions == []


@pytest.mark.parametrize("condition", ["hazard", "dynamics"])
def test_battery_gates_pass_when_answered_or_under_allow_pending(tmp_path, monkeypatch, condition) -> None:
    spec = _pilot_spec()
    directory = _fake_run(tmp_path, spec)
    stubs = Stubs(monkeypatch)
    with errors.allow_pending():
        smoke = E.evaluate_battery(str(directory), spec.to_dict(), 10_000_000, [condition])
    assert math.isfinite(smoke[condition])
    key = "Q-hazard" if condition == "hazard" else "Q-dynamics"
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset({key}))
    answered = E.evaluate_battery(str(directory), spec.to_dict(), 10_000_000, [condition])
    assert answered == smoke and len(stubs.calls) == 4


def test_run_episodes_gates_its_conditions_before_building_anything(monkeypatch) -> None:
    def no_env(*args, **kwargs):
        raise AssertionError("an environment was built")

    monkeypatch.setattr(E, "make_eval_env", no_env)
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset())  # the keys open, whatever the repository answers
    policy = SimpleNamespace(budget_conditioned=False, task="SafetyPointGoal1-v0", obs_dim=60)
    with pytest.raises(PendingQuestionError, match="Q-hazard"):
        E.run_episodes(policy, [1], condition="hazard", hazard_layout_seed=3_000_000)
    with pytest.raises(PendingQuestionError, match="Q-dynamics"):
        E.run_episodes(policy, [1], condition="dynamics")
    with errors.allow_pending():
        assert E.run_episodes(policy, [], condition="dynamics") == []  # no seeds: nothing is built


def test_run_episodes_argument_checks(monkeypatch) -> None:
    monkeypatch.setattr(E, "make_eval_env", lambda *a, **k: (_ for _ in ()).throw(AssertionError("built")))
    plain = SimpleNamespace(budget_conditioned=False, task="SafetyPointGoal1-v0", obs_dim=60)
    budgeted = SimpleNamespace(budget_conditioned=True, task="SafetyPointGoal1-v0", obs_dim=60)
    with pytest.raises(ValueError, match="unknown condition"):
        E.run_episodes(plain, [1], condition="finetune")
    with pytest.raises(ValueError, match="budget-conditioned"):
        E.run_episodes(plain, [1], budgets=[25.0])  # a plain policy refuses budgets
    with pytest.raises(ValueError, match="needs one budget per episode"):
        E.run_episodes(budgeted, [1])  # a budget-conditioned policy refuses to run without
    with pytest.raises(ValueError, match="2 episodes need 2 budgets"):
        E.run_episodes(budgeted, [1, 2], budgets=[25.0])
    with pytest.raises(ValueError):
        E.run_episodes(budgeted, [1], budgets=[float("nan")])
    for bad in ([-1], [2**32], [True], [1.5], "12"):
        with pytest.raises(ValueError):
            E.run_episodes(plain, bad)
    with pytest.raises(ValueError, match="hazard condition only"):
        E.run_episodes(plain, [1], hazard_layout_seed=3_000_000)
    with pytest.raises(ValueError, match="hazard condition only"):
        E.run_episodes(plain, [1], condition="dynamics", hazard_form="central")
    with errors.allow_pending():
        with pytest.raises(ValueError, match="needs hazard_layout_seed"):
            E.run_episodes(plain, [1], condition="hazard")
        with pytest.raises(ValueError, match="3 episodes need 3 hazard layout seeds"):
            E.run_episodes(plain, [1, 2, 3], condition="hazard", hazard_layout_seed=[3_000_000, 3_000_001])
        with pytest.raises(ValueError, match="unknown hazard form"):
            E.run_episodes(plain, [1], condition="hazard", hazard_layout_seed=3_000_000, hazard_form="ring")
    assert E.run_episodes(budgeted, [], budgets=[]) == []


# ---------------------------------------------------------------------------------------------
# Unstable simulations: reserve-seed replacement (Q-mujoco-exception, Table 9.1)
# ---------------------------------------------------------------------------------------------


class UnstableEnv:
    """An ``EvalEnv`` whose detection reports an unstable simulation for chosen seeds (``unstable``: seed ->
    step), as ``EvalEnv.run_episode`` raises it; any other seed gives a deterministic episode. Records each
    episode it runs with the hazards pinned at the time and its budget."""

    def __init__(self, unstable: dict[int, int], condition: str = E.IN_DISTRIBUTION) -> None:
        self.unstable = dict(unstable)
        self.task_id, self.condition = "SafetyPointGoal1-v0", condition
        self.pinned: list | None = None
        self.ran: list[tuple[int, list | None, float | None]] = []
        self.closed = False

    def pin_hazards(self, locations) -> None:
        self.pinned = locations

    def run_episode(self, actor, seed, budget=None) -> E.EpisodeResult:
        self.ran.append((seed, self.pinned, budget))
        if seed in self.unstable:
            raise E.MujocoInstabilityError(f"unstable episode {seed}", seed=seed, step=self.unstable[seed],
                                           warning="mjWARN_BADQACC x1")
        return E.EpisodeResult(seed=seed, cost=float(seed % 7), ret=(seed % 5) / 4.0, length=R.EPISODE_LENGTH,
                               budget=budget)

    def close(self) -> None:
        self.closed = True


def _unstable_env(monkeypatch, unstable: dict[int, int]) -> UnstableEnv:
    env = UnstableEnv(unstable)

    def make(policy, *, task=None, condition=E.IN_DISTRIBUTION):
        env.condition = condition
        return env

    monkeypatch.setattr(E, "make_eval_env", make)
    return env


PLAIN = SimpleNamespace(budget_conditioned=False, actor=None)


def test_the_reserve_sequences_are_disjoint_from_every_seed_set() -> None:
    """Table 9.1 (Q-mujoco-exception): selection 1,050,000 + r, measurement 2,050,000 + r, hazard episodes
    3,150,000 + r, r < 5, disjoint from the canonical sets, the hazard layouts, the training seeds and the fixed
    batch's seed 0 (as docs/DECISIONS.md, Q-eval-seeds, requires of the canonical sets themselves)."""
    from results import supplement_schema as S

    assert E.RESERVE_SEED_OFFSET == contracts.RESERVE_SEED_OFFSET == S.RESERVE_SEED_OFFSET == 50_000
    assert E.MAX_UNSTABLE_EPISODES == contracts.MAX_UNSTABLE_EPISODES == S.MAX_UNSTABLE_EPISODES == 5
    flat = [s for row in E.hazard_episode_seeds() for s in row]
    reserves = {"selection": E.reserve_seeds(E.selection_seeds()), "measurement": E.reserve_seeds(E.measurement_seeds()),
                "hazard": E.reserve_seeds(flat)}
    assert reserves == {"selection": list(range(1_050_000, 1_050_005)),
                        "measurement": list(range(2_050_000, 2_050_005)), "hazard": list(range(3_150_000, 3_150_005))}
    canonical = set(E.selection_seeds()) | set(E.measurement_seeds()) | set(E.hazard_layout_seeds()) | set(flat)
    every_reserve = [s for seq in reserves.values() for s in seq]
    assert len(every_reserve) == len(set(every_reserve)) and not set(every_reserve) & canonical
    training = set(range(0, R.MAX_SEEDS_PER_ARM + 1000))  # run seeds, surplus and replacement seeds, the fixed batch's 0
    assert not set(every_reserve) & training and max(every_reserve) < 2**32
    # a part of a set has its set's sequence (contract 6's first n selection seeds; one dynamics episode)
    assert E.reserve_seeds(E.selection_seeds()[:3]) == reserves["selection"]
    assert E.reserve_seeds([E.measurement_seeds()[7]]) == reserves["measurement"]
    assert E.reserve_seeds([7, 8]) == E.reserve_seeds([E.selection_seeds()[0], E.measurement_seeds()[0]]) == []


def test_run_episodes_replaces_an_unstable_episode_by_the_next_reserve_seed(monkeypatch) -> None:
    """One unstable measurement episode: 100 episodes, that one on 2,050,000, the seeds asked for unchanged."""
    seeds = E.measurement_seeds()
    env = _unstable_env(monkeypatch, {seeds[3]: 412})
    results = E.run_episodes(PLAIN, seeds)
    assert len(results) == 100 and env.closed
    assert [r.seed for r in results] == seeds[:3] + [2_050_000] + seeds[4:]
    assert results[3].replaced == (E.Replacement(seed=seeds[3], reserve_seed=2_050_000, step=412,
                                                 warning="mjWARN_BADQACC x1"),)
    assert results[3].cost == float(2_050_000 % 7)  # the reserve episode is scored, the unstable one never
    assert [seed for seed, _, _ in env.ran] == seeds[:4] + [2_050_000] + seeds[4:]
    assert E.unstable_replacements(results) == [{"slot_index": 3, "seed": seeds[3], "reserve_seed": 2_050_000,
                                                 "step": 412, "warning": "mjWARN_BADQACC x1"}]
    assert E._checked(results, seeds, "measurement") == results  # the planned seeds are still the ones checked
    assert E.unstable_replacements(E.run_episodes(PLAIN, seeds[4:])) == []  # no instability, nothing recorded


def test_an_unstable_reserve_episode_is_replaced_and_counts_toward_the_cap(monkeypatch) -> None:
    """Every unstable episode, planned or reserve, counts toward the cap of 5 per evaluation call; a sixth
    fails the call (MujocoInstabilityError, held for the group), and r runs on across the call's episodes."""
    seeds = E.selection_seeds()
    env = _unstable_env(monkeypatch, {seeds[1]: 0, 1_050_000: 5, seeds[50]: 7, seeds[60]: 9, seeds[99]: 11})
    results = E.run_episodes(PLAIN, seeds)
    assert [r.seed for r in results if r.replaced] == [1_050_001, 1_050_002, 1_050_003, 1_050_004]
    assert [(u["slot_index"], u["seed"], u["reserve_seed"]) for u in E.unstable_replacements(results)] == [
        (1, seeds[1], 1_050_000), (1, 1_050_000, 1_050_001), (50, seeds[50], 1_050_002), (60, seeds[60], 1_050_003),
        (99, seeds[99], 1_050_004)]
    E._checked(results, seeds, "selection")
    env.unstable[seeds[70]] = 3  # a sixth unstable episode in the same call
    with pytest.raises(E.MujocoInstabilityError, match=r"6 episodes of one evaluation call .* more than the 5") as info:
        E.run_episodes(PLAIN, seeds)
    assert not isinstance(info.value, RunRefused) and "held for the group" in str(info.value)
    assert "not a Part 5.6 exclusion" in str(info.value) and len(info.value.unstable) == 5
    assert (info.value.seed, info.value.step) == (seeds[99], 11)  # seeds[70] took the fifth reserve seed
    assert env.closed


def test_six_unstable_planned_episodes_fail_the_call_and_five_do_not(monkeypatch) -> None:
    seeds = E.measurement_seeds()
    env = _unstable_env(monkeypatch, {s: 1 for s in seeds[:5]})
    assert [r.seed for r in E.run_episodes(PLAIN, seeds)[:6]] == [*range(2_050_000, 2_050_005), seeds[5]]
    env.unstable[seeds[5]] = 1
    with pytest.raises(E.MujocoInstabilityError, match="more than the 5"):
        E.run_episodes(PLAIN, seeds)


def test_seeds_outside_the_canonical_sets_have_no_reserve(monkeypatch) -> None:
    _unstable_env(monkeypatch, {8: 2})
    assert [r.seed for r in E.run_episodes(PLAIN, [7, 9])] == [7, 9]
    with pytest.raises(E.MujocoInstabilityError, match="no canonical evaluation set .* no reserve") as info:
        E.run_episodes(PLAIN, [7, 8])
    assert (info.value.seed, info.value.step, info.value.unstable) == (8, 2, ())


def test_a_replacement_keeps_the_pinned_hazard_layout_and_the_budget(monkeypatch) -> None:
    """The reserve episode of a hazard episode runs with that episode's pinned layout (Table 9.1: "with the same
    hazard layout or budget"), on the hazard set's reserve sequence 3,150,000 + r."""
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset({"Q-hazard"}))
    flat = [s for row in E.hazard_episode_seeds() for s in row]
    layouts = [lay for lay in E.hazard_layout_seeds() for _ in range(R.EPISODES_PER_LAYOUT)]

    class Drawer:
        def __init__(self, task_id, form):
            pass

        def draw(self, layout_seed):
            return [(float(layout_seed - E.HAZARD_LAYOUT_SEED_BASE), 0.0)]

        def close(self):
            pass

    monkeypatch.setattr(E, "_HazardDrawer", Drawer)
    unstable = flat[7]  # the third episode of the second layout
    env = _unstable_env(monkeypatch, {unstable: 30})
    results = E.run_episodes(PLAIN, flat, condition="hazard", hazard_layout_seed=layouts)
    assert results[7].seed == 3_150_000 and results[7].replaced[0].seed == unstable
    ran = {seed: pinned for seed, pinned, _ in env.ran}
    assert ran[unstable] == ran[3_150_000] == [(1.0, 0.0)]  # layout 3,000,001, pinned for both
    assert ran[flat[8]] == [(1.0, 0.0)] and ran[flat[10]] == [(2.0, 0.0)]
    budgeted = SimpleNamespace(budget_conditioned=True, actor=None)
    env = _unstable_env(monkeypatch, {E.measurement_seeds()[1]: 4})
    results = E.run_episodes(budgeted, E.measurement_seeds()[:3], budgets=[10.0, 20.0, 40.0])
    assert [(r.seed, r.budget) for r in results] == [(2_000_000, 10.0), (2_050_000, 20.0), (2_000_002, 40.0)]
    assert [(seed, budget) for seed, _, budget in env.ran] == [
        (2_000_000, 10.0), (2_000_001, 20.0), (2_050_000, 20.0), (2_000_002, 40.0)]
    E._checked(results, E.measurement_seeds()[:3], "budgets", budgets=[10.0, 20.0, 40.0])


def test_checked_refuses_a_replacement_off_the_rule(monkeypatch) -> None:
    """``_checked`` accepts a reserve seed only as the rule gives it: the planned seed first, the next unused
    seed of the set's reserve sequence each time, the episode run on the last reserve seed."""
    seeds = E.measurement_seeds()
    _unstable_env(monkeypatch, {seeds[0]: 3})
    results = E.run_episodes(PLAIN, seeds)

    def with_first(**changes):
        first = results[0]
        replaced = (E.Replacement(**{**dataclasses.asdict(first.replaced[0]), **changes.pop("replacement", {})}),)
        return [dataclasses.replace(first, replaced=replaced, **changes), *results[1:]]

    E._checked(results, seeds, "ok")
    cases = [
        (with_first(replacement={"reserve_seed": 2_050_001}, seed=2_050_001), "not the next unused seed"),
        (with_first(replacement={"reserve_seed": 1_050_000}, seed=1_050_000), "not the next unused seed"),
        (with_first(replacement={"seed": seeds[1]}), "replaces seed"),
        (with_first(seed=seeds[0]), "where seed 2050000 was asked for"),  # scored the unstable episode
        (with_first(replacement={"warning": ""}), "warning"),
    ]
    for bad, message in cases:
        with pytest.raises(contracts.ContractError, match=message):
            E._checked(bad, seeds, "bad")


# ---------------------------------------------------------------------------------------------
# evaluate_run (contract 1)
# ---------------------------------------------------------------------------------------------


def test_evaluate_run_passes_contract_one_on_the_selection_window(tmp_path, monkeypatch) -> None:
    spec = _pilot_spec()
    directory = _fake_run(tmp_path, spec)
    stubs = Stubs(monkeypatch)
    result = E.evaluate_run(str(directory), spec.to_dict())
    contracts.validate_evaluation(result, spec, directory)  # the launcher's check
    from pilot import launch

    launch._check_evaluation_record(spec, {**result, "eval_commit_hash": "a" * 40})  # and the ledger's, at that stage
    window = contracts.selection_window(contracts.checkpoint_steps(directory), spec.total_steps)
    assert window == list(range(8_200_000, 10_000_001, 200_000))
    assert sorted(result["selection"]) == window
    assert [load[1] for load in stubs.loads] == window  # one load per window checkpoint (the final one reused)
    assert all(isinstance(load[2], manifest.RunSpec) and load[2].run_id == spec.run_id for load in stubs.loads)
    assert [c["step"] for c in stubs.calls] == window + [spec.total_steps]
    assert all(c["seeds"] == E.selection_seeds() for c in stubs.calls[:-1])
    assert stubs.calls[-1]["seeds"] == E.measurement_seeds()  # Q-final-cost-set: the final cost on the measurement set
    assert all(set(c) == {"step", "seeds"} for c in stubs.calls)  # in distribution, no budget
    final_costs = [float(s % 7) + 10.0 for s in E.measurement_seeds()]
    assert result["final_cost"] == statistics.fmean(final_costs)
    assert result["episode_costs"]["final"] == final_costs
    assert result["final_selection_cost"] == result["selection"][spec.total_steps][0]
    # the selection-set value is kept beside it and decides nothing (Q-final-cost-set)
    assert result["final_selection_cost"] != result["final_cost"]
    assert result["final_step"] == spec.total_steps and result["final_seed_set"] == "measurement"
    assert result["episodes"] == R.EVAL_EPISODES == 100
    assert result["selection_seeds"] == E.selection_seeds() and result["measurement_seeds"] == E.measurement_seeds()
    for step in window:
        costs, returns = result["episode_costs"]["selection"][step], result["episode_returns"]["selection"][step]
        assert len(costs) == len(returns) == 100
        assert result["selection"][step] == [statistics.fmean(costs), statistics.fmean(returns)]
    assert result["short_episodes"] == 0 and result["harness"]["torch_threads"] == R.TORCH_THREADS


def test_evaluate_run_result_survives_the_launchers_json_writer(tmp_path, monkeypatch) -> None:
    from pilot import launch

    spec = _pilot_spec(onset=0)
    directory = _fake_run(tmp_path, spec)
    Stubs(monkeypatch)
    result = E.evaluate_run(str(directory), spec.to_dict())
    assert launch.plain(result) == result  # plain Python types only
    path = tmp_path / "evaluation.json"
    launch._write_json_atomic(path, result)  # sort_keys: a dict mixing int and str keys would raise here
    reloaded = json.loads(path.read_text(encoding="utf-8"))
    contracts.validate_evaluation(reloaded, spec, directory)
    assert reloaded["final_cost"] == result["final_cost"]


def _is_failure(exc: BaseException) -> bool:
    """A problem of the run's saved output: exit 1 (eval_failed), reported by enrich without a traceback."""
    return (isinstance(exc, E.CheckpointInvalid) and isinstance(exc, contracts.ContractError)
            and isinstance(exc, ValueError) and not isinstance(exc, RunRefused))


@pytest.mark.parametrize("damage", ["onset", "final", "window"])
def test_a_missing_part_3_4_checkpoint_fails_the_evaluation_before_any_episode(tmp_path, monkeypatch, damage) -> None:
    """A deleted checkpoint is classified as a truncated one is: a failure, never a refusal.

    A failure exits 1 (eval_failed); a refusal exits 5 and is relaunched at every scheduler start. The
    launcher's validate_evaluation agrees.
    """
    spec = _pilot_spec()
    gone = {"onset": 5_000_000, "final": spec.total_steps, "window": 9_000_000}[damage]
    steps = [s for s in contracts.expected_checkpoint_steps(spec) if s != gone]
    directory = _fake_run(tmp_path, spec, steps=steps)
    stubs = Stubs(monkeypatch)
    with pytest.raises(E.CheckpointInvalid, match=rf"missing at steps \[{gone}\]") as info:
        E.evaluate_run(str(directory), spec.to_dict())
    assert _is_failure(info.value)
    assert stubs.loads == [] and stubs.calls == []
    with pytest.raises(contracts.ContractError, match="missing"):  # the launcher's own check: the same class
        contracts.validate_evaluation({"final_cost": 0.0, "final_return": 0.0, "selection": {}, "episodes": 100,
                                       "selection_seeds": E.selection_seeds()}, spec, directory)


def test_a_checkpoint_beyond_the_total_fails_the_evaluation_before_any_episode(tmp_path, monkeypatch) -> None:
    spec = _pilot_spec()
    directory = _fake_run(tmp_path, spec, steps=contracts.expected_checkpoint_steps(spec) + [spec.total_steps + 200_000])
    stubs = Stubs(monkeypatch)
    with pytest.raises(E.CheckpointInvalid, match=r"beyond the run's 10000000 steps at steps \[10200000\]") as info:
        E.evaluate_run(str(directory), spec.to_dict())
    assert _is_failure(info.value) and stubs.loads == [] and stubs.calls == []


def test_an_off_grid_checkpoint_inside_the_window_fails_the_evaluation_before_any_episode(tmp_path, monkeypatch) -> None:
    """A stray in-window checkpoint is a broken checkpoint set: a failure (exit 1), not a refusal."""
    spec = _pilot_spec()
    stray = 9_100_000  # off the 200,000-step grid, inside the final 2,000,000 steps
    directory = _fake_run(tmp_path, spec, steps=contracts.expected_checkpoint_steps(spec) + [stray])
    stubs = Stubs(monkeypatch)
    with pytest.raises(E.CheckpointInvalid, match=r"off the grid .*broken checkpoint set") as info:
        E.evaluate_run(str(directory), spec.to_dict())
    assert _is_failure(info.value) and stubs.loads == [] and stubs.calls == []


def test_a_hazard_layout_seed_is_named_as_one_in_errors() -> None:
    with pytest.raises(ValueError, match="a hazard layout seed must be a non-negative integer"):
        E.draw_hazard_layout("SafetyPointGoal1-v0", -1)
    with pytest.raises(ValueError, match="an episode seed must be a non-negative integer"):
        E._seed_list([-1])


def test_evaluate_run_loads_every_window_checkpoint_before_the_first_episode(tmp_path, monkeypatch) -> None:
    """A damaged final checkpoint stops the call before any episode, as a failure (exit 1), not a refusal."""
    spec = _pilot_spec()
    directory = _fake_run(tmp_path, spec)
    stubs = Stubs(monkeypatch, fail_at=(spec.total_steps,))
    with pytest.raises(E.CheckpointUnreadable) as info:
        E.evaluate_run(str(directory), spec.to_dict())
    assert stubs.calls == []  # no episode ran (it used to be 9 selection sets, about 39 minutes)
    window = contracts.selection_window(contracts.checkpoint_steps(directory), spec.total_steps)
    assert [load[1] for load in stubs.loads] == window
    assert _is_failure(info.value)  # pilot.launch evaluate exits 1: eval_failed, not relaunched as refused


def test_evaluate_run_refuses_an_off_grid_total_while_q_selection_window_is_open(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset())  # the key open, whatever the repository answers
    spec = next(s for s in manifest.design("all") if s.total_steps % R.CHECKPOINT_INTERVAL_STEPS and s.study == "A")
    assert "Q-selection-window" in spec.pending
    directory = _fake_run(tmp_path, spec)
    stubs = Stubs(monkeypatch)
    with pytest.raises(E.EvaluationRefused, match="Q-selection-window"):
        E.evaluate_run(str(directory), spec.to_dict())
    assert stubs.calls == []


@pytest.mark.parametrize("total, message", [
    (1_020_000, "shorter than the 10-checkpoint end-relative window"),  # off the grid: 12 checkpoints saved
    (1_600_000, "only 9 checkpoints"),  # on the grid
])
def test_evaluate_run_refuses_a_total_shorter_than_the_selection_window(tmp_path, monkeypatch, total, message) -> None:
    """A total below 1,800,000 steps has no ten-checkpoint window: a problem of the spec, refused (exit 5), not a
    failure, even when the run saved every checkpoint it should (the end-relative steps of an off-grid total)."""
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", R.ANSWERED_QUESTIONS | {"Q-selection-window"})  # the window's answer
    spec = dataclasses.replace(_pilot_spec(onset=0), total_steps=total)
    expected = contracts.expected_checkpoint_steps(spec)
    assert len(expected) == {1_020_000: 12, 1_600_000: 9}[total]
    directory = _fake_run(tmp_path, spec)  # every expected checkpoint present
    stubs = Stubs(monkeypatch)
    with pytest.raises(E.EvaluationRefused, match=message) as info:
        E.evaluate_run(str(directory), spec.to_dict())
    assert isinstance(info.value, RunRefused) and not _is_failure(info.value)
    assert stubs.loads == [] and stubs.calls == []


@pytest.mark.parametrize("key, value, message", [
    ("env_id", "SafetyCarGoal1-v0", "is not the spec's task"),
    ("algo", "CPPOPID", "is not the spec's base_algo"),
    ("train_cfgs.total_steps", 4_000_000, "is not the spec's total_steps"),
    ("pilot_cfgs.run_id", "P-A-PointGoal1-N0.50-abrupt-total-s1", "is not the spec's run_id"),
    ("algo_cfgs.reward_normalize", True, "reward_normalize"),
    ("algo_cfgs.cost_normalize", True, "cost_normalize"),
    ("algo_cfgs.obs_normalize", None, "obs_normalize"),
    ("env_cfgs", {"num_envs": 2}, "default layout distribution"),
    ("algo_cfgs.steps_per_epoch", 10_000, "steps_per_epoch"),
    ("pilot_cfgs", "not-a-mapping", "pilot_cfgs.run_id"),
])
def test_evaluate_run_refuses_a_config_that_does_not_match(tmp_path, monkeypatch, key, value, message) -> None:
    spec = _pilot_spec()
    directory = _fake_run(tmp_path, spec, **{key: value})
    stubs = Stubs(monkeypatch)
    with pytest.raises(E.EvaluationRefused, match=message) as info:
        E.evaluate_run(str(directory), spec.to_dict())
    assert isinstance(info.value, RunRefused) and isinstance(info.value, ValueError)
    assert stubs.calls == []


def test_evaluate_run_refuses_a_directory_without_config_and_an_invalid_spec(tmp_path, monkeypatch) -> None:
    Stubs(monkeypatch)
    spec = _pilot_spec()
    with pytest.raises(E.EvaluationRefused, match="not an OmniSafe run directory"):
        E.evaluate_run(str(tmp_path), spec.to_dict())
    (tmp_path / "config.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(E.EvaluationRefused, match="cannot be read"):
        E.evaluate_run(str(tmp_path), spec.to_dict())
    with pytest.raises(E.EvaluationRefused, match="not a valid run spec"):
        E.evaluate_run(str(tmp_path), {**spec.to_dict(), "total_steps": 123})
    with pytest.raises(E.EvaluationRefused, match="run spec"):
        E.evaluate_run(str(tmp_path), "P-A-PointGoal1-N0.00-s0")


def test_evaluate_run_leaves_budget_conditioned_runs_and_continuations_to_their_owners(tmp_path, monkeypatch) -> None:
    stubs = Stubs(monkeypatch)
    moderate = next(s for s in manifest.pilot() if s.plugin == "study_b")
    with pytest.raises(E.EvaluationRefused, match="studyb.evaluation:evaluate_run"):
        E.evaluate_run(str(_fake_run(tmp_path / "b", moderate)), moderate.to_dict())
    fewshot = next(s for s in manifest.design("all") if s.group == "study_b_fewshot")
    with pytest.raises(E.EvaluationRefused, match="budget-conditioned.*studyb.evaluation.evaluate_fewshot evaluates it"):
        E.evaluate_run(str(_fake_run(tmp_path / "f", fewshot)), fewshot.to_dict())
    finetune = _finetune_spec()  # a battery continuation: not budget-conditioned, so the continuation check refuses it
    directory = _fake_run(tmp_path / "c", finetune)
    with pytest.raises(E.EvaluationRefused, match="is a continuation"):
        E.evaluate_run(str(directory), finetune.to_dict())
    with pytest.raises(E.EvaluationRefused, match="is a continuation"):
        E.evaluate_battery(str(directory), finetune.to_dict(), R.FINETUNE_STEPS, [])
    with pytest.raises(E.EvaluationRefused, match="evaluate_fewshot"):
        E.evaluate_continuation(str(_fake_run(tmp_path / "f2", fewshot)), fewshot.to_dict())
    assert stubs.calls == [] and stubs.loads == []


@pytest.mark.parametrize("key, value, message", [
    ("model_cfgs", None, "model_cfgs.actor is missing"),
    ("model_cfgs.actor", "tanh", "model_cfgs.actor is missing"),
    ("model_cfgs.actor.hidden_sizes", [64, 0], "hidden_sizes"),
    ("model_cfgs.actor.hidden_sizes", None, "hidden_sizes"),
    ("model_cfgs.actor.activation", "nosuch", "activation"),
    ("model_cfgs.actor_type", "mlp", "actor_type"),
    ("model_cfgs.weight_initialization_mode", "nosuch", "weight_initialization_mode"),
])
def test_load_policy_refuses_a_model_cfgs_that_cannot_build_the_actor(tmp_path, key, value, message) -> None:
    """Refused before the checkpoint is read (the fake run's checkpoint files are empty, so reading one would fail)."""
    spec = _pilot_spec()
    directory = _fake_run(tmp_path, spec, **{key: value})
    with pytest.raises(E.EvaluationRefused, match=message) as info:
        E.load_policy(directory, spec.total_steps, spec)
    assert isinstance(info.value, RunRefused) and not isinstance(info.value, E.CheckpointInvalid)
    assert "the actor cannot be built" in str(info.value)


def test_the_harness_record_lists_the_normaliser_only_when_the_run_has_one(tmp_path, monkeypatch) -> None:
    assert E._harness_record(True)["wrappers"] == ["TimeLimit", "ObsNormalize(frozen)", "ActionScale", "Unsqueeze"]
    plain = E._harness_record(False)
    assert plain["wrappers"] == ["TimeLimit", "ActionScale", "Unsqueeze"] and plain["normaliser"].startswith("none")
    spec = _pilot_spec()
    directory = _fake_run(tmp_path, spec)
    stubs = Stubs(monkeypatch)
    real = stubs.load_policy
    monkeypatch.setattr(E, "load_policy", lambda *a: SimpleNamespace(**{**vars(real(*a)), "normalizer_state": None}))
    assert E.evaluate_run(str(directory), spec.to_dict())["harness"] == plain
    assert E.evaluate_battery(str(directory), spec.to_dict(), spec.total_steps, [])["harness"] == plain


def test_evaluate_run_evaluates_the_unconstrained_pilot_run_like_study_a(tmp_path, monkeypatch) -> None:
    spec = next(s for s in manifest.pilot() if s.plugin == "unconstrained_ppo")  # study B, base_algo PPO
    directory = _fake_run(tmp_path, spec)
    Stubs(monkeypatch)
    contracts.validate_evaluation(E.evaluate_run(str(directory), spec.to_dict()), spec, directory)


def test_evaluate_run_checks_what_run_episodes_returns(tmp_path, monkeypatch) -> None:
    spec = _pilot_spec()
    directory = _fake_run(tmp_path, spec)
    stubs = Stubs(monkeypatch)
    real = stubs.run_episodes
    monkeypatch.setattr(E, "run_episodes", lambda policy, seeds, **kw: real(policy, seeds, **kw)[:-1])
    with pytest.raises(contracts.ContractError, match="99 episodes for 100 seeds"):
        E.evaluate_run(str(directory), spec.to_dict())
    monkeypatch.setattr(E, "run_episodes", lambda policy, seeds, **kw: list(reversed(real(policy, seeds, **kw))))
    with pytest.raises(contracts.ContractError, match="where seed 1000000 was asked for"):
        E.evaluate_run(str(directory), spec.to_dict())
    monkeypatch.setattr(E, "run_episodes", lambda policy, seeds, **kw: [
        E.EpisodeResult(seed=s, cost=float("nan"), ret=0.0, length=1000) for s in seeds])
    with pytest.raises(contracts.ContractError, match="nan"):
        E.evaluate_run(str(directory), spec.to_dict())


def _first_episode_replaced(run_episodes):
    """``run_episodes`` whose first episode of every call was unstable and ran the call's first reserve seed."""

    def replaced(policy, seeds, **kwargs):
        results = run_episodes(policy, seeds, **kwargs)
        reserve = E.reserve_seeds(seeds)[0]
        unstable = E.Replacement(seed=results[0].seed, reserve_seed=reserve, step=12, warning="mjWARN_BADQVEL x1")
        results[0] = dataclasses.replace(results[0], seed=reserve, replaced=(unstable,))
        return results

    return replaced


def test_the_contract_results_report_replacements_beside_the_planned_seeds(tmp_path, monkeypatch) -> None:
    """Q-mujoco-exception (Table 9.1; docs/DECISIONS.md: "Seed-set checks compare the planned canonical seeds"):
    every seed list stays the canonical planned one and each evaluation's replacements are reported; the
    launcher's and the ledger's checks accept them."""
    from pilot import launch

    spec = _pilot_spec()
    directory = _fake_run(tmp_path, spec)
    stubs = Stubs(monkeypatch)
    monkeypatch.setattr(E, "run_episodes", _first_episode_replaced(stubs.run_episodes))
    result = E.evaluate_run(str(directory), spec.to_dict())
    assert result["selection_seeds"] == E.selection_seeds() and result["measurement_seeds"] == E.measurement_seeds()
    replaced = result["unstable_replacements"]
    assert replaced["final"] == [{"slot_index": 0, "seed": 2_000_000, "reserve_seed": 2_050_000, "step": 12,
                                  "warning": "mjWARN_BADQVEL x1"}]
    assert sorted(replaced["selection"]) == sorted(result["selection"])
    assert all(entries[0]["reserve_seed"] == 1_050_000 for entries in replaced["selection"].values())
    contracts.validate_evaluation(result, spec, directory)
    path = tmp_path / "evaluation.json"
    launch._write_json_atomic(path, result)  # step keys become strings
    reloaded = json.loads(path.read_text(encoding="utf-8"))
    contracts.validate_evaluation(reloaded, spec, directory)
    launch._check_evaluation_record(spec, {**reloaded, "eval_commit_hash": "a" * 40})
    reloaded["unstable_replacements"]["final"][0]["reserve_seed"] = 2_050_001
    with pytest.raises(contracts.ContractError, match="not the next unused seed"):
        contracts.validate_evaluation(reloaded, spec, directory)

    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset({"Q-hazard", "Q-dynamics"}))
    battery = E.evaluate_battery(str(directory), spec.to_dict(), 9_000_000, ["hazard", "dynamics"])
    flat = [s for row in E.hazard_episode_seeds() for s in row]
    assert battery["seeds"] == {"measurement": E.measurement_seeds(), "hazard": flat, "dynamics": E.measurement_seeds()}
    assert {name: [u["reserve_seed"] for u in entries] for name, entries in battery["unstable_replacements"].items()} == {
        "measurement": [2_050_000], "hazard": [3_150_000], "dynamics": [2_050_000]}
    continuation = E.evaluate_continuation(str(tmp_path), _finetune_spec().to_dict())
    assert continuation["measurement_seeds"] == E.measurement_seeds()
    assert [u["reserve_seed"] for u in continuation["unstable_replacements"]] == [2_050_000]


# ---------------------------------------------------------------------------------------------
# evaluate_battery (contract 5)
# ---------------------------------------------------------------------------------------------


def test_evaluate_battery_measures_c_id_first_then_each_requested_condition(tmp_path, monkeypatch) -> None:
    from pilot import enrichment

    spec = _pilot_spec()
    directory = _fake_run(tmp_path, spec)
    stubs = Stubs(monkeypatch)
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset({"Q-hazard", "Q-dynamics"}))
    step = 9_000_000
    costs = E.evaluate_battery(str(directory), spec.to_dict(), step, ["dynamics", "hazard"])
    assert [load[1] for load in stubs.loads] == [step]
    # both definitions were read from the run's task before the first episode (C_ID)
    assert stubs.definitions == [("dynamics", "SafetyPointGoal1-v0", 0), ("hazard", "SafetyPointGoal1-v0", 0)]
    measurement, dynamics, hazard = stubs.calls
    assert measurement == {"step": step, "seeds": E.measurement_seeds()}
    assert dynamics == {"step": step, "seeds": E.measurement_seeds(), "condition": "dynamics"}  # paired with C_ID
    flat = [s for row in E.hazard_episode_seeds() for s in row]
    layouts = [lay for lay in E.hazard_layout_seeds() for _ in range(R.EPISODES_PER_LAYOUT)]
    assert hazard == {"step": step, "seeds": flat, "condition": "hazard", "hazard_layout_seed": layouts}
    assert set(costs) >= {"measurement", "measurement_return", "hazard", "hazard_return", "dynamics", "dynamics_return",
                          "episodes", "measurement_seeds", "step", "episode_costs", "episode_returns", "seeds",
                          "conditions", "harness", "short_episodes"}
    assert costs["step"] == step and costs["episodes"] == 100
    assert costs["seeds"] == {"measurement": E.measurement_seeds(), "dynamics": E.measurement_seeds(), "hazard": flat}
    for name in ("measurement", "hazard", "dynamics"):
        assert costs[name] == statistics.fmean(costs["episode_costs"][name])
        assert costs[f"{name}_return"] == statistics.fmean(costs["episode_returns"][name])
    assert costs["conditions"] == {"hazard": stubs.hazard_definition("SafetyPointGoal1-v0"),
                                   "dynamics": stubs.dynamics_definition("SafetyPointGoal1-v0")}
    gaps = enrichment.gaps_from_costs(costs, ["hazard", "dynamics"])  # equation (1)
    assert gaps["measurement_cost"] == costs["measurement"]
    assert gaps["gap_hazard"] == costs["hazard"] - costs["measurement"]
    assert gaps["gap_dynamics"] == costs["dynamics"] - costs["measurement"]
    measurement_seeds = contracts.validate_seed_set("measurement_seeds", costs["measurement_seeds"])
    assert not set(measurement_seeds) & set(E.selection_seeds())  # Table 2.1: disjoint sets


def test_evaluate_battery_checks_the_hazard_task_before_measuring_c_id(tmp_path, monkeypatch) -> None:
    spec = _pilot_spec()
    directory = _fake_run(tmp_path, spec)
    stubs = Stubs(monkeypatch)

    hazard_checks = []

    def no_hazards(task):
        hazard_checks.append(task)
        raise E.EvaluationRefused(f"{task} has no hazards to relocate")

    monkeypatch.setattr(E, "hazard_definition", no_hazards)
    with errors.allow_pending(), pytest.raises(E.EvaluationRefused, match="no hazards"):
        E.evaluate_battery(str(directory), spec.to_dict(), spec.total_steps, ["dynamics", "hazard"])
    assert hazard_checks == ["SafetyPointGoal1-v0"] and stubs.calls == []  # refused before the 100 C_ID episodes
    with errors.allow_pending():
        E.evaluate_battery(str(directory), spec.to_dict(), spec.total_steps, ["dynamics"])
    assert hazard_checks == ["SafetyPointGoal1-v0"]  # only a hazard condition needs hazards


def test_evaluate_battery_with_no_condition_is_the_measurement_only(tmp_path, monkeypatch) -> None:
    spec = _pilot_spec()
    directory = _fake_run(tmp_path, spec)
    stubs = Stubs(monkeypatch)
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset())
    assert R.is_open("Q-hazard") and R.is_open("Q-dynamics")  # no gate applies to C_ID alone (pilot/enrichment.py)
    costs = E.evaluate_battery(str(directory), spec.to_dict(), spec.total_steps, [])
    assert "hazard" not in costs and "dynamics" not in costs and costs["conditions"] == {}
    assert stubs.definitions == []  # nothing of a condition is built for the measurement alone
    assert len(stubs.calls) == 1 and costs["measurement"] == statistics.fmean(costs["episode_costs"]["measurement"])


def test_evaluate_battery_argument_and_run_checks(tmp_path, monkeypatch) -> None:
    spec = _pilot_spec()
    directory = _fake_run(tmp_path, spec)
    stubs = Stubs(monkeypatch)
    for bad in (["finetune"], ["hazard", "hazard"], "hazard", None):
        with pytest.raises(ValueError):
            E.evaluate_battery(str(directory), spec.to_dict(), spec.total_steps, bad)
    for step in (9_990_000, 10_200_000, -200_000, True, 9e6):  # off the epoch grid, beyond the run, negative, not ints
        with pytest.raises(E.EvaluationRefused):  # bad arguments: refused
            E.evaluate_battery(str(directory), spec.to_dict(), step, [])
    (directory / "torch_save" / "epoch-450.pt").unlink()  # a window checkpoint of the run, deleted
    for step in (9_000_000, 220_000):  # a deleted checkpoint, and a step inside the run that was never saved
        with pytest.raises(E.CheckpointInvalid, match="is missing") as info:
            E.evaluate_battery(str(directory), spec.to_dict(), step, [])
        assert _is_failure(info.value)
    moderate = next(s for s in manifest.pilot() if s.plugin == "study_b")
    with pytest.raises(E.EvaluationRefused, match="Study A"):
        E.evaluate_battery(str(directory), moderate.to_dict(), spec.total_steps, [])
    with pytest.raises(E.EvaluationRefused, match="is not the spec's run_id"):
        E.evaluate_battery(str(directory), _pilot_spec(seed=1).to_dict(), spec.total_steps, [])
    assert stubs.calls == []
    E.evaluate_battery(str(directory), spec.to_dict(), 200_000, [])  # any saved checkpoint of the run
    assert stubs.loads[-1][1] == 200_000


# ---------------------------------------------------------------------------------------------
# evaluate_checkpoint (contract 6) and evaluate_continuation
# ---------------------------------------------------------------------------------------------


def test_evaluate_checkpoint_runs_the_first_selection_seeds(tmp_path, monkeypatch) -> None:
    stubs = Stubs(monkeypatch)
    cost = E.evaluate_checkpoint(str(tmp_path), 200_000, 7)
    assert stubs.loads == [(tmp_path, 200_000, None)]  # no spec: the determinism check has none
    assert stubs.calls == [{"step": 200_000, "seeds": E.selection_seeds()[:7]}]
    assert type(cost) is float and cost == statistics.fmean(float(s % 7) + 0.2 for s in E.selection_seeds()[:7])
    for bad in (0, 101, True, 2.0, -1):
        with pytest.raises(ValueError, match="episodes must be an integer from 1 to 100"):
            E.evaluate_checkpoint(str(tmp_path), 200_000, bad)
    assert len(stubs.loads) == 1


def _moderate_determinism_spec() -> manifest.RunSpec:
    """What pilot/scheduler.py's ``--plugin study_b`` check trains: the Moderate arm, to the first scheduled checkpoint."""
    moderate = next(s for s in manifest.pilot() if s.plugin == "study_b")
    return dataclasses.replace(moderate, run_id="DET-B-Moderate-s0", total_steps=R.CHECKPOINT_INTERVAL_STEPS,
                               group="determinism", pilot=False)


def test_evaluate_checkpoint_of_a_budget_conditioned_run_is_gated_by_q_studyb_eval(tmp_path, monkeypatch) -> None:
    spec = _moderate_determinism_spec()
    directory = _fake_run(tmp_path, spec, steps=[0, R.CHECKPOINT_INTERVAL_STEPS])
    (tmp_path / "spec.json").write_text(spec.to_json(), encoding="utf-8")
    stubs = Stubs(monkeypatch, budget_conditioned=True)
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset())  # the key open, whatever the repository answers
    with pytest.raises(PendingQuestionError, match="Q-studyb-eval"):
        E.evaluate_checkpoint(str(directory), R.CHECKPOINT_INTERVAL_STEPS, 2)
    assert stubs.calls == []
    with errors.allow_pending():  # smoke tests only
        assert math.isfinite(E.evaluate_checkpoint(str(directory), R.CHECKPOINT_INTERVAL_STEPS, 2))


def test_evaluate_checkpoint_gives_a_budget_conditioned_checkpoint_its_arms_training_budgets(tmp_path, monkeypatch) -> None:
    """pilot/scheduler.py: the study_b determinism report goes through contract 6 like every other plug-in's."""
    spec = _moderate_determinism_spec()
    step = R.CHECKPOINT_INTERVAL_STEPS  # scripts/determinism_check.py: the first scheduled checkpoint
    directory = _fake_run(tmp_path, spec, steps=[0, step])
    stubs = Stubs(monkeypatch, budget_conditioned=True)
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset({"Q-studyb-eval"}))
    with pytest.raises(E.EvaluationRefused, match="pass spec="):  # no spec given and no spec.json beside the run
        E.evaluate_checkpoint(str(directory), step, 7)
    assert stubs.calls == []
    (tmp_path / "spec.json").write_text(spec.to_json(), encoding="utf-8")  # RUN_DIR/spec.json, as train_once writes it
    found = E.evaluate_checkpoint(str(directory), step, 7)  # the script's three-argument call
    given = E.evaluate_checkpoint(str(directory), step, 7, spec=spec.to_dict())
    budgets = [10.0, 20.0, 40.0, 10.0, 20.0, 40.0, 10.0]  # Moderate's levels in turn (Q-studyb-eval, Table 9.1)
    assert stubs.calls == [{"step": step, "seeds": E.selection_seeds()[:7], "budgets": budgets}] * 2
    assert type(found) is float and found == given == statistics.fmean(float(s % 7) + 0.2 for s in E.selection_seeds()[:7])
    assert stubs.loads[-1][2] == spec  # a spec passed in is checked by load_policy against config.json
    real = stubs.run_episodes
    monkeypatch.setattr(E, "run_episodes", lambda policy, seeds, **kw: [
        dataclasses.replace(r, budget=40.0) for r in real(policy, seeds, **kw)])
    with pytest.raises(contracts.ContractError, match="ran with budget 40.0, not 10.0"):
        E.evaluate_checkpoint(str(directory), step, 7)


@pytest.mark.parametrize("case", ["other run", "plain plug-in", "unreadable"])
def test_evaluate_checkpoint_checks_the_spec_found_beside_the_run(tmp_path, monkeypatch, case) -> None:
    spec = _moderate_determinism_spec()
    directory = _fake_run(tmp_path, spec, steps=[0, R.CHECKPOINT_INTERVAL_STEPS])
    stubs = Stubs(monkeypatch, budget_conditioned=True)
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset({"Q-studyb-eval"}))
    beside, message = {
        "other run": (dataclasses.replace(spec, run_id="DET-B-Sparse-s0").to_json(), "is not the spec's run_id"),
        "plain plug-in": (manifest.RunSpec(run_id=spec.run_id, study="A", task=spec.task, arm="x", seed=0,
                                           total_steps=spec.total_steps, base_algo="PPOLag", plugin="ppolag",
                                           group="determinism").to_json(), "the spec's plug-in is 'ppolag'"),
        "unreadable": ("{not json", "cannot be read"),
    }[case]
    (tmp_path / "spec.json").write_text(beside, encoding="utf-8")
    with pytest.raises(E.EvaluationRefused, match=message):
        E.evaluate_checkpoint(str(directory), R.CHECKPOINT_INTERVAL_STEPS, 2)
    assert stubs.calls == []


def test_training_budget_schedule_is_the_arms_training_budgets_in_turn() -> None:
    specs = {s.arm: s for s in manifest.design("all") if s.plugin == "study_b" and s.seed == 0}
    assert set(specs) == set(R.STUDY_B_ARMS)
    for arm, levels in R.STUDY_B_ARMS.items():
        full = E.training_budget_schedule(specs[arm].to_dict())
        assert len(full) == R.EVAL_EPISODES and all(type(b) is float for b in full)
        assert E.training_budget_schedule(specs[arm], 7) == full[:7]  # episode i's budget does not depend on n
        if levels is None:  # Continuous: midpoints of 100 equal parts of [10, 40]
            lo, hi = R.CONTINUOUS_RANGE
            assert full == [lo + (hi - lo) * (i + 0.5) / R.EVAL_EPISODES for i in range(R.EVAL_EPISODES)]
            assert full == sorted(full) and lo < full[0] < full[-1] < hi
            assert full[0] == pytest.approx(10.15) and statistics.fmean(full) == pytest.approx((lo + hi) / 2)
        else:
            k = len(levels)
            assert full == [sorted(levels)[i % k] for i in range(R.EVAL_EPISODES)]
            assert all(full.count(b) in (R.EVAL_EPISODES // k, R.EVAL_EPISODES // k + 1) for b in levels)
    assert E.training_budget_schedule(specs["Moderate"], 0) == []
    for n in (101, -1, True, 2.5):
        with pytest.raises(ValueError):
            E.training_budget_schedule(specs["Moderate"], n)


def test_training_budget_schedule_refuses_what_is_not_a_study_b_training_run() -> None:
    fewshot = next(s for s in manifest.design("all") if s.group == "study_b_fewshot")
    with pytest.raises(E.EvaluationRefused, match="evaluate_fewshot"):  # one budget, Role 5's evaluation
        E.training_budget_schedule(fewshot)
    with pytest.raises(E.EvaluationRefused, match="no training budget schedule"):
        E.training_budget_schedule(_pilot_spec())
    continuous = next(s for s in manifest.design("all") if s.plugin == "study_b" and s.training_levels is None)
    for bad in ([5.0, 45.0], None, "10-40", [10.0]):
        params = {k: v for k, v in continuous.params.items() if k != "continuous_range"}
        if bad is not None:
            params["continuous_range"] = bad
        with pytest.raises(E.EvaluationRefused, match="continuous_range"):
            E.training_budget_schedule(dataclasses.replace(continuous, params=params))
    with pytest.raises(E.EvaluationRefused, match="training_levels"):
        E.training_budget_schedule(dataclasses.replace(_moderate_determinism_spec(), training_levels=(10.0, float("nan"))))


def test_training_budget_schedule_quotes_q_studyb_eval_verbatim() -> None:
    """The docstring quotes Q-studyb-eval's answer (Table 9.1) verbatim.

    The key is answered (docs/DECISIONS.md, 2026-10-02): its PENDING text states the question and then
    the answer, of which the quotation is a verbatim part.
    """

    def norm(text: str) -> str:
        return re.sub(r"\s+", " ", text)

    doc = norm(E.training_budget_schedule.__doc__)
    quoted = doc[doc.index('"For the contract-1 fields') + 1:]
    quoted = quoted[:quoted.index('"')]
    assert "sorted(levels)[i mod k]" in quoted and "10.15, 10.45, ..., 39.85" in doc
    assert not R.is_open("Q-studyb-eval")  # answered (docs/DECISIONS.md, 2026-10-02)
    assert " Answered: " in R.PENDING["Q-studyb-eval"] and quoted in norm(R.PENDING["Q-studyb-eval"])


def _finetune_spec() -> manifest.RunSpec:
    parent = _pilot_spec()
    return manifest.RunSpec(
        run_id=f"{parent.arm_id}-finetune-s{parent.seed}", study="A", task=parent.task, arm=parent.arm, seed=parent.seed,
        total_steps=R.FINETUNE_STEPS, base_algo="PPOLag", plugin="battery_finetune", group="battery_finetune",
        depends_on=(parent.run_id,), params={"parent_run_id": parent.run_id, "parent_step": 9_000_000})


def test_evaluate_continuation_measures_the_final_checkpoint_on_the_measurement_set(tmp_path, monkeypatch) -> None:
    spec = _finetune_spec()
    stubs = Stubs(monkeypatch)
    result = E.evaluate_continuation(str(tmp_path), spec.to_dict())
    assert [load[1] for load in stubs.loads] == [R.FINETUNE_STEPS]
    assert stubs.calls == [{"step": R.FINETUNE_STEPS, "seeds": E.measurement_seeds(), "task": spec.task}]
    assert result["cost"] == statistics.fmean(result["episode_costs"]) and len(result["episode_returns"]) == 100
    assert result["step"] == R.FINETUNE_STEPS and result["task"] == spec.task and result["episodes"] == 100
    assert result["measurement_seeds"] == E.measurement_seeds()
    with pytest.raises(E.EvaluationRefused, match="not a continuation"):
        E.evaluate_continuation(str(tmp_path), _pilot_spec().to_dict())


def test_summarise_uses_fmean() -> None:
    results = [E.EpisodeResult(seed=i, cost=float(c), ret=r, length=1000) for i, (c, r) in enumerate([(1, 0.1), (2, 0.2), (4, 0.4)])]
    assert E.summarise(results) == (statistics.fmean([1.0, 2.0, 4.0]), statistics.fmean([0.1, 0.2, 0.4]))
    with pytest.raises(ValueError):
        E.summarise([])

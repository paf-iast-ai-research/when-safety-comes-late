"""Role 2's Study A plug-in ``envs.onset`` on tiny real runs (2,000 steps per epoch, at most 6 epochs).

Run with ``pytest -m slow``. Each run is built directly from the launcher's configuration shrunk to
tiny epochs (the registered factory path needs 20,000-step epochs; the N = 0 comparison goes through
``make_algorithm`` on a smoke copy). A spec's onset step is on the 20,000-step grid, so the tiny
runs set ``onset_cfgs`` and ``pilot_cfgs.onset_step`` to the tiny onset epoch themselves.

The abrupt run is deliberately shorter than the test First Tasks (Role 2, Tests) describes (20,000
steps of 2,000 per epoch, onset at step 10,000): 8,000 steps with onset at step 4,000, to keep the
slow suite's runtime down. It covers the same properties: two unconstrained epochs (the multiplier
held and both critics trained from one epoch to the next before onset), the onset epoch (the
multiplier updated first, then the actor with the new value) and one later constrained epoch.

* N = 0 is OmniSafe's PPO-Lagrangian bit for bit, with Role 3's plasticity hook installed.
* Abrupt onset at epoch 2 of 4: before onset the actor's surrogate is A_R exactly, the multiplier is
  its stored initial value and never updated, and both critics train; the onset epoch updates the
  multiplier first and the actor with the new value (First Tasks, Role 2 tests).
* The ramp's budget per epoch (equation 5), the warm start from an N = 0 run directory
  (Table 2.4; Q-warm-start), the rate limit (Q-rate-limit), the PID check skipping ``pid_update``
  before onset (Q-pid-eq9).
* Partial reset and plasticity injection: applied once at the start of the onset epoch's rollout,
  after the onset checkpoint and its plasticity row, which equal the untreated run's; later
  checkpoints load with ``weights_only=True`` and rebuild through ``metrics.interventions.build_actor``.
* Additional constrained training (Table 9.1, Q-data-control-lr): one run of T' + N·T' epochs whose
  first T' epochs are the untreated abrupt run's, bit for bit, apart from the learning-rate schedule's
  own state and the rate of the next epoch; then it trains on at the restarted rate (1 − N)·lr.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from configs import registered as R
from pilot import dependencies, launch, manifest
from pilot.errors import RunRefused
from pilot.manifest import RunSpec

pytestmark = [pytest.mark.omnisafe, pytest.mark.slow]

torch = pytest.importorskip("torch")
pytest.importorskip("omnisafe")

from envs import onset  # noqa: E402  # after the skip: it imports OmniSafe

TINY = 2_000  # steps per epoch
E = R.STEPS_PER_EPOCH
COMMIT = "a" * 40


# ---------------------------------------------------------------------------
# Building tiny runs
# ---------------------------------------------------------------------------


def _spec(group: str, **match) -> RunSpec:
    specs = manifest.pilot() if group == "pilot" else manifest.design(group)
    match.setdefault("task", R.PRIMARY_TASK)
    match.setdefault("seed", 0)
    return next(s for s in specs if all(getattr(s, k) == v for k, v in match.items()))


def _late_copy(spec: RunSpec, *, keep_dependencies: bool = False) -> RunSpec:
    """A SMOKE- copy of two registered epochs with the onset after the first (a valid unregistered spec)."""
    return replace(spec, run_id=onset.SMOKE_PREFIX + spec.run_id, total_steps=2 * E, onset_step=E,
                   depends_on=spec.depends_on if keep_dependencies else ())


def _tiny_cfgs(spec: RunSpec, run_dir: Path, *, epochs: int, onset_epoch: int, plasticity: bool = True, deps=None,
               vector_env_nums: int = 1):
    cfgs = launch.build_config(spec, run_dir / "omnisafe", deps)
    cfgs.algo_cfgs.steps_per_epoch = TINY
    cfgs.train_cfgs.vector_env_nums = vector_env_nums  # the launcher pins 1 (R.VECTOR_ENV_NUMS)
    cfgs.train_cfgs.total_steps = TINY * epochs
    cfgs.train_cfgs.epochs = epochs
    cfgs.logger_cfgs.save_model_freq = 1
    cfgs.logger_cfgs.use_tensorboard = False
    cfgs.pilot_cfgs.plasticity = plasticity
    cfgs.pilot_cfgs.extra_checkpoint_steps = []
    cfgs.pilot_cfgs.onset_step = onset_epoch * TINY
    return cfgs


def _tiny_onset(spec: RunSpec, onset_epoch: int) -> dict[str, Any]:
    """``build_onset_cfgs`` of the (unregistered) spec, moved to the tiny onset epoch."""
    oc = onset.build_onset_cfgs(spec, steps_per_epoch=E, registered=False)
    oc.update(onset_step=onset_epoch * TINY, onset_epoch=onset_epoch, steps_per_epoch=TINY)
    return oc


def _omnisafe_dir(run_dir: Path) -> Path:
    (path,) = [p for p in (run_dir / "omnisafe").glob("*/seed-*") if p.is_dir()]
    return path


def _rows(omni: Path) -> list[dict[str, str]]:
    with open(omni / "progress.csv", newline="") as fh:
        return [{k: v for k, v in r.items() if not k.startswith("Time/")} for r in csv.DictReader(fh)]


def _column(omni: Path, key: str) -> list[float]:
    return [float(r[key]) for r in _rows(omni)]


def _checkpoint(omni: Path, epoch: int) -> dict[str, Any]:
    return torch.load(omni / "torch_save" / f"epoch-{epoch}.pt", weights_only=True, map_location="cpu")


def _plasticity_lines(omni: Path) -> dict[int, str]:
    lines = (omni / "plasticity.csv").read_text().splitlines()[1:]
    return {int(line.split(",")[0]): line for line in lines}


def _same(a: Any, b: Any) -> bool:
    """Deep, exact equality of checkpoint entries (tensors bit for bit)."""
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def _f32(x: float) -> float:
    return float(np.float32(x))


def _multiplier(algo) -> float:
    value = algo._lagrange.lagrangian_multiplier
    return value.item() if isinstance(value, torch.Tensor) else float(value)


def _instrument(algo) -> list[tuple]:
    """Record every multiplier update and every surrogate call (instance attributes; no random draw).

    ("update", epoch, before, after) and ("surrogate", epoch, multiplier, is A_R, is equation 3).
    """
    events: list[tuple] = []
    lagrange = algo._lagrange
    name = "pid_update" if hasattr(lagrange, "pid_update") else "update_lagrange_multiplier"
    original_update = getattr(lagrange, name)

    def update(cost):
        before = _multiplier(algo)
        original_update(cost)
        events.append(("update", int(algo._logger.current_epoch), before, _multiplier(algo)))

    setattr(lagrange, name, update)
    original_surrogate = algo._compute_adv_surrogate

    def surrogate(adv_r, adv_c):
        out = original_surrogate(adv_r, adv_c)
        m = _multiplier(algo)
        events.append(("surrogate", int(algo._logger.current_epoch), m, bool(torch.equal(out, adv_r)),
                       bool(torch.equal(out, (adv_r - m * adv_c) / (1 + m)))))
        return out

    algo._compute_adv_surrogate = surrogate
    return events


def _train(root: Path, spec: RunSpec, *, epochs: int, onset_epoch: int, cls=None, plasticity: bool = True,
           deps=None, warm_start=None, vector_env_nums: int = 1, run_name: str | None = None) -> SimpleNamespace:
    from omnisafe.utils.config import Config

    run_dir = root / (run_name or spec.run_id)
    cfgs = _tiny_cfgs(spec, run_dir, epochs=epochs, onset_epoch=onset_epoch, plasticity=plasticity, deps=deps,
                      vector_env_nums=vector_env_nums)
    oc = _tiny_onset(spec, onset_epoch)
    if warm_start is not None:
        oc["warm_start"] = warm_start(cfgs)
    cfgs["onset_cfgs"] = Config.dict2config(oc)
    algo = (cls or onset.OnsetPPOLag)(env_id=spec.task, cfgs=cfgs)
    initial = algo._lagrange.lagrangian_multiplier
    initial = initial.detach().clone() if isinstance(initial, torch.Tensor) else float(initial)
    events = _instrument(algo)
    algo.learn()
    return SimpleNamespace(spec=spec, algo=algo, cfgs=cfgs, omni=_omnisafe_dir(run_dir), events=events, initial=initial)


@pytest.fixture(scope="module")
def data_root(tmp_path_factory):
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield tmp_path_factory.mktemp("onset")
    torch.set_num_threads(threads)


@pytest.fixture(scope="module")
def abrupt(data_root):
    """The pilot's abrupt N = 0.50 arm, onset at epoch 2 of 4: also the untreated partner of the treatments."""
    return _train(data_root, _late_copy(_spec("pilot", N=0.5)), epochs=4, onset_epoch=2)


# ---------------------------------------------------------------------------
# N = 0 is OmniSafe's PPO-Lagrangian
# ---------------------------------------------------------------------------


def test_n0_through_make_algorithm_is_omnisafes_ppolag_bit_for_bit(data_root) -> None:
    """N = 0 through ``make_algorithm`` logs and saves what ``make_ppolag`` does, bit for bit, with Role 3's hook
    installed through pilot_cfgs.plasticity."""
    from pilot.algorithms import make_ppolag

    base = _spec("main", N=0.0)
    smoke = replace(base, run_id=onset.SMOKE_PREFIX + base.run_id, total_steps=E)  # as scripts/smoke_run.py shortens it
    runs = {}
    for name, plasticity in (("ppolag", False), ("study_a", True)):
        run_dir = data_root / f"n0-{name}"
        cfgs = _tiny_cfgs(smoke, run_dir, epochs=3, onset_epoch=0, plasticity=plasticity)
        algo = (make_ppolag if name == "ppolag" else onset.make_algorithm)(smoke.task, cfgs, smoke)
        algo.learn()
        runs[name] = SimpleNamespace(omni=_omnisafe_dir(run_dir), cfgs=cfgs, algo=algo)
    ours, theirs = runs["study_a"], runs["ppolag"]
    assert type(ours.algo) is onset.OnsetPPOLag and ours.cfgs.onset_cfgs.onset_epoch == 0
    rows, reference = _rows(ours.omni), _rows(theirs.omni)
    assert len(rows) == len(reference) == 3
    for row, ref in zip(rows, reference):
        assert {k: row[k] for k in ref} == ref  # every logged value, bit for bit
        assert set(row) - set(ref) == set(onset.ONSET_KEYS)
    assert _column(ours.omni, onset.ACTIVE_KEY) == [1.0, 1.0, 1.0]
    assert _column(ours.omni, onset.BUDGET_KEY) == [R.COST_LIMIT] * 3
    for epoch in (1, 3):
        mine, ref = _checkpoint(ours.omni, epoch), _checkpoint(theirs.omni, epoch)
        assert mine.keys() == ref.keys()
        for key in ref:
            assert _same(mine[key], ref[key]), (epoch, key)
    assert sorted(_plasticity_lines(ours.omni)) == [0, TINY, 2 * TINY, 3 * TINY]
    assert not (theirs.omni / "plasticity.csv").exists()
    saved = json.loads((ours.omni / "config.json").read_text())
    assert saved["onset_cfgs"]["onset_step"] == 0 and saved["onset_cfgs"]["registered"] is False


# ---------------------------------------------------------------------------
# Abrupt onset
# ---------------------------------------------------------------------------


def test_before_onset_the_actor_uses_a_r_and_the_multiplier_is_held_while_both_critics_train(abrupt) -> None:
    """First Tasks (Role 2): the multiplier equals its stored initial value exactly and its update is never
    called before onset; the surrogate equals the reward advantage exactly; the cost critic's parameters
    change between the first and second epochs before onset."""
    initial = abrupt.initial
    assert initial.dtype == torch.float32 and initial.item() == _f32(R.LAGRANGE_MULTIPLIER_INIT)
    before = [e for e in abrupt.events if e[1] < 2]
    surrogates = [e for e in before if e[0] == "surrogate"]
    assert surrogates and all(e[3] for e in surrogates)  # A_R exactly, every minibatch
    assert not any(e[4] and not e[3] for e in surrogates)
    assert not [e for e in before if e[0] == "update"]
    assert _column(abrupt.omni, onset.MULTIPLIER_KEY)[:2] == [initial.item()] * 2
    assert _column(abrupt.omni, onset.START_KEY)[:3] == [initial.item()] * 3
    for epoch in (0, 1, 2):
        state = _checkpoint(abrupt.omni, epoch)
        assert torch.equal(state["lagrange"]["value"], initial)
        assert state["lambda_optimizer"]["state"] == {}  # never stepped: the onset starts from a fresh Adam state
    assert _column(abrupt.omni, onset.ACTIVE_KEY) == [0.0, 0.0, 1.0, 1.0]
    budgets = _column(abrupt.omni, onset.BUDGET_KEY)
    assert math.isnan(budgets[0]) and math.isnan(budgets[1]) and budgets[2:] == [R.COST_LIMIT] * 2
    proposed = _column(abrupt.omni, onset.PROPOSED_KEY)
    assert all(math.isnan(v) for v in proposed[:2]) and proposed[2:] == _column(abrupt.omni, onset.MULTIPLIER_KEY)[2:]
    states = [_checkpoint(abrupt.omni, epoch) for epoch in (0, 1, 2)]
    for critic in ("cost_critic", "reward_critic"):
        for a, b in zip(states, states[1:]):
            assert not _same(a[critic], b[critic]), critic  # trained in each unconstrained epoch
    losses = _column(abrupt.omni, "Loss/Loss_cost_critic")
    assert len(losses) == 4 and all(math.isfinite(v) for v in losses)


def test_the_onset_epoch_updates_the_multiplier_first_then_the_actor_with_it(abrupt) -> None:
    """From onset, OmniSafe's own order (ppo_lag.py:73-80): the multiplier from J_C, then the actor with the new
    multiplier; one update per constrained epoch."""
    updates = [e for e in abrupt.events if e[0] == "update"]
    assert [e[1] for e in updates] == [2, 3]
    onset_epoch = [e for e in abrupt.events if e[1] == 2]
    assert onset_epoch[0][0] == "update" and onset_epoch[0][2] == abrupt.initial.item()
    after = onset_epoch[0][3]
    assert after > abrupt.initial.item()
    surrogates = [e for e in onset_epoch if e[0] == "surrogate"]
    assert surrogates and all(e[2] == after and e[4] and not e[3] for e in surrogates)  # equation (3) with the new value
    multipliers = _column(abrupt.omni, onset.MULTIPLIER_KEY)
    assert multipliers[2] == after and multipliers[3] == updates[1][3]
    assert _column(abrupt.omni, onset.START_KEY)[3] == after
    assert float(_checkpoint(abrupt.omni, 4)["lambda_optimizer"]["state"][0]["step"]) == 2
    assert not (abrupt.omni / onset.INTERVENTION_FILE).exists()


# ---------------------------------------------------------------------------
# Ramp, warm start, rate limit, PID
# ---------------------------------------------------------------------------


def test_the_ramp_sets_the_budget_of_each_update_by_equation_5(data_root) -> None:
    spec = _late_copy(_spec("main", N=0.5, onset_shape="ramp", step_matching="total_steps"))
    run = _train(data_root, spec, epochs=3, onset_epoch=1, plasticity=False)
    expected = [onset.effective_budget(e * TINY, onset_step=TINY, cost_limit=R.COST_LIMIT,
                                       d_loose=R.D_LOOSE[spec.task], window_steps=R.RAMP_WINDOW_STEPS) for e in (1, 2)]
    assert expected[0] == R.D_LOOSE[spec.task] and expected[1] == pytest.approx(
        R.D_LOOSE[spec.task] - (R.D_LOOSE[spec.task] - R.COST_LIMIT) * TINY / R.RAMP_WINDOW_STEPS)
    budgets = _column(run.omni, onset.BUDGET_KEY)
    assert math.isnan(budgets[0]) and budgets[1:] == [_f32(b) for b in expected]  # progress.csv holds float32
    final = _checkpoint(run.omni, 3)["lagrange"]["numeric_attributes"]
    assert final["cost_limit"] == expected[1]
    assert [e[1] for e in run.events if e[0] == "update"] == [1, 2]


def test_two_environments_train_and_the_ramp_counts_whole_epoch_steps(data_root) -> None:
    """With vector_env_nums = 2, OmniSafe rolls out E / 2 steps per environment into a buffer of that
    size (policy_gradient.py:73-77, 121, 251). The run completes, and the ramp still evaluates
    equation 5 at e · E, the steps of every environment together (TotalEnvSteps,
    policy_gradient.py:270). A regression test: overwriting OmniSafe's per-environment length with E
    made learn() fail with "No more space in the buffer!"."""
    spec = _late_copy(_spec("main", N=0.5, onset_shape="ramp", step_matching="total_steps"))
    run = _train(data_root, spec, epochs=2, onset_epoch=1, plasticity=False, vector_env_nums=2, run_name="two-envs")
    assert run.algo._steps_per_epoch == TINY // 2 and run.algo._onset_steps_per_epoch == TINY
    assert [b.max_size for b in run.algo._buf.buffers] == [TINY // 2, TINY // 2]
    assert _column(run.omni, "TotalEnvSteps") == [TINY, 2 * TINY]
    budgets = _column(run.omni, onset.BUDGET_KEY)
    assert math.isnan(budgets[0]) and budgets[1] == _f32(R.D_LOOSE[spec.task])  # onset epoch: t = N·T gives d_loose
    assert [e[1] for e in run.events if e[0] == "update"] == [1]


def _fake_n0_run(root: Path, spec: RunSpec, values: list[str]) -> Path:
    """A completed N = 0 run directory at the tiny epoch length (progress.csv only)."""
    run_dir = root / spec.run_id
    omni = run_dir / "omnisafe" / f"PPOLag-{{{spec.task}}}" / "seed-000-2026-09-30-00-00-00"
    (omni / "torch_save").mkdir(parents=True)
    (run_dir / "spec.json").write_text(spec.to_json())
    (omni / "config.json").write_text(json.dumps({"algo_cfgs": {"steps_per_epoch": TINY}}))
    with open(omni / "progress.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Train/Epoch", "TotalEnvSteps", "Metrics/LagrangeMultiplier"])
        writer.writerows([epoch, (epoch + 1) * TINY, value] for epoch, value in enumerate(values))
    (run_dir / "train_result.json").write_text(json.dumps({
        "run_id": spec.run_id, "status": "completed", "commit_hash": COMMIT, "config_hash": "e" * 64,
        "omnisafe_dir": str(omni.relative_to(run_dir)),
    }))
    return omni


def test_the_warm_start_sets_the_n0_value_before_the_onset_epochs_update(data_root) -> None:
    root = data_root / "warm"
    source = _spec("main", N=0.0)
    value = "0.06747621297836304"
    _fake_n0_run(root, source, ["0.03599999472498894", value, "0.0955224186182022"])
    spec = _late_copy(_spec("controller", controller_variant="warm_started", N=0.5), keep_dependencies=True)
    assert spec.depends_on == (source.run_id,)
    deps = dependencies.resolve(spec, root / spec.run_id)
    run = _train(root, spec, epochs=3, onset_epoch=2, plasticity=False, deps=deps,
                 warm_start=lambda cfgs: onset.warm_start_source(cfgs, spec, 2 * TINY))
    assert run.cfgs.onset_cfgs.warm_start.value == float(value)
    assert run.algo._warm_start_applied
    multipliers = _column(run.omni, onset.MULTIPLIER_KEY)
    assert multipliers[:2] == [run.initial.item()] * 2  # held at 0.001 before onset
    assert torch.equal(_checkpoint(run.omni, 2)["lagrange"]["value"], run.initial)  # the onset state is pre-warm-start
    (update,) = [e for e in run.events if e[0] == "update"]
    assert update[1] == 2 and update[2] == float(value)  # set before the onset epoch's first update
    assert _column(run.omni, onset.START_KEY)[2] == float(value)
    assert multipliers[2] == update[3] != float(value)
    assert float(_checkpoint(run.omni, 3)["lambda_optimizer"]["state"][0]["step"]) == 1  # a fresh Adam state at onset


def test_the_rate_limit_clips_every_constrained_update(data_root) -> None:
    spec = _late_copy(_spec("controller", controller_variant="rate_limited", N=0.5))
    run = _train(data_root, spec, epochs=3, onset_epoch=1, plasticity=False)
    assert isinstance(run.algo._lagrange, onset.RateLimitedLagrange)
    updates = [e for e in run.events if e[0] == "update"]
    assert [e[1] for e in updates] == [1, 2]
    proposed = _column(run.omni, onset.PROPOSED_KEY)
    costs = _column(run.omni, "Metrics/EpCost")
    for _, epoch, before, after in updates:
        bound = R.RATE_LIMIT_RELATIVE * before + R.RATE_LIMIT_ABSOLUTE
        assert after == _f32(onset.rate_limit(before, proposed[epoch], relative=R.RATE_LIMIT_RELATIVE,
                                              absolute=R.RATE_LIMIT_ABSOLUTE))
        assert abs(after - before) <= bound + 1e-7
    # the first constrained epoch's J_C (the 50-episode window) exceeds d, so Adam proposes about +0.035: clipped
    assert costs[1] > R.COST_LIMIT and proposed[1] > updates[0][2] + R.RATE_LIMIT_ABSOLUTE
    assert updates[0][3] == _f32(updates[0][2] + R.RATE_LIMIT_RELATIVE * updates[0][2] + R.RATE_LIMIT_ABSOLUTE)
    saved = _checkpoint(run.omni, 3)["lagrange"]
    assert saved["class"] == "RateLimitedLagrange"
    assert saved["numeric_attributes"]["rate_limit_relative"] == R.RATE_LIMIT_RELATIVE
    assert saved["numeric_attributes"]["rate_limit_absolute"] == R.RATE_LIMIT_ABSOLUTE
    assert saved["numeric_attributes"]["last_proposed"] == proposed[2]


def test_the_pid_check_skips_pid_update_before_onset(data_root) -> None:
    spec = _late_copy(_spec("pid", N=0.5))
    run = _train(data_root, spec, epochs=3, onset_epoch=2, cls=onset.OnsetCPPOPID, plasticity=False)
    assert run.initial == 0.0  # OmniSafe's PID multiplier before its first update (pid_lagrange.py:85-90)
    updates = [e for e in run.events if e[0] == "update"]
    assert [e[1] for e in updates] == [2]
    surrogates = [e for e in run.events if e[0] == "surrogate"]
    before = [e for e in surrogates if e[1] < 2]
    at_onset = [e for e in surrogates if e[1] == 2]
    assert before and at_onset  # not vacuous: the surrogate was recorded before onset and in the onset epoch
    assert all(e[3] for e in before) and all(e[4] for e in at_onset)  # A_R, then equation (3)
    multipliers = _column(run.omni, onset.MULTIPLIER_KEY)
    assert multipliers[:2] == [0.0, 0.0] and multipliers[2] == _f32(updates[0][3])
    assert _column(run.omni, onset.BUDGET_KEY)[2] == R.COST_LIMIT
    saved = _checkpoint(run.omni, 2)["lagrange"]
    assert saved["class"] == "PIDLagrangian" and saved["numeric_attributes"]["_cost_penalty"] == 0.0
    final = _checkpoint(run.omni, 3)["lagrange"]
    assert final["value"].item() == updates[0][3]  # float64 in the checkpoint


# ---------------------------------------------------------------------------
# Additional constrained training (Table 2.4; Table 9.1, Q-data-control-lr)
# ---------------------------------------------------------------------------


def test_the_data_control_is_the_untreated_run_for_t_epochs_then_trains_at_the_restarted_rate(data_root,
                                                                                                abrupt) -> None:
    """The data control of the abrupt run (T' = 4 epochs, onset at epoch 2) is one run of T' + N·T' = 6 epochs. Its
    first T' epochs are the untreated run's: every progress row, except the last row's Train/LR (the rate of the
    next epoch: 0 for the untreated run, (1 − N)·lr after the restart), and every checkpoint entry up to step T',
    except the schedule's own state and, at T', the actor optimiser's rate of the next epoch. Then the actor trains
    on at the restarted rate, which decays to 0 at the end of the run."""
    base = _spec("treatment", treatment="additional_constrained", N=0.5)
    spec = replace(base, run_id=onset.SMOKE_PREFIX + base.run_id, total_steps=6 * E, onset_step=E)
    run = _train(data_root, spec, epochs=6, onset_epoch=2)
    assert run.cfgs.onset_cfgs[onset.SCHEDULE_KEY] == {"decay_epochs": 4, "tail_epochs": 2, "tail_start_factor": 0.5}
    assert isinstance(run.algo._actor_critic.actor_scheduler, onset.DataControlLR)
    lr = float(run.cfgs.model_cfgs.actor.lr)
    rows, untreated = _rows(run.omni), _rows(abrupt.omni)
    assert len(rows) == 6 and len(untreated) == 4 and rows[0].keys() == untreated[0].keys()
    for epoch in range(4):
        differ = {key for key in untreated[epoch] if rows[epoch][key] != untreated[epoch][key]}
        assert differ == (set() if epoch < 3 else {"Train/LR"}), epoch
    assert _column(abrupt.omni, "Train/LR")[3] == 0.0
    assert _column(run.omni, "Train/LR")[3:] == [_f32(0.5 * lr), _f32(0.25 * lr), 0.0]
    assert _column(run.omni, onset.ACTIVE_KEY) == [0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    for epoch in range(5):  # epoch-4.pt: the state after T' epochs
        mine, partner = _checkpoint(run.omni, epoch), _checkpoint(abrupt.omni, epoch)
        assert mine.keys() == partner.keys()
        differ = {key for key in partner if not _same(mine[key], partner[key])}
        assert differ == ({"actor_scheduler"} if epoch < 4 else {"actor_scheduler", "actor_optimizer"}), epoch
        if epoch < 4:  # the inner LinearLR is the untreated run's schedule, state for state
            assert _same(mine["actor_scheduler"]["decay"], partner["actor_scheduler"])
    mine, partner = _checkpoint(run.omni, 4)["actor_optimizer"], _checkpoint(abrupt.omni, 4)["actor_optimizer"]
    assert _same(mine["state"], partner["state"])  # the Adam moments continue
    assert [g["lr"] for g in mine["param_groups"]] == [0.5 * lr] and [g["lr"] for g in partner["param_groups"]] == [0.0]
    assert _same([{k: v for k, v in g.items() if k != "lr"} for g in mine["param_groups"]],
                 [{k: v for k, v in g.items() if k != "lr"} for g in partner["param_groups"]])
    assert not _same(_checkpoint(run.omni, 5)["pi"], _checkpoint(run.omni, 4)["pi"])  # the actor trains on
    assert _checkpoint(run.omni, 6)["actor_scheduler"]["last_epoch"] == 6


# ---------------------------------------------------------------------------
# Partial reset and plasticity injection (Table 2.4)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def treated(data_root):
    runs = {}
    for treatment in ("reset", "injection"):
        spec = _late_copy(_spec("treatment", treatment=treatment, N=0.5))
        runs[treatment] = _train(data_root, spec, epochs=4, onset_epoch=2)
    return runs


@pytest.mark.parametrize("treatment", ["reset", "injection"])
def test_the_intervention_is_applied_once_after_the_onset_checkpoint_and_its_plasticity_row(abrupt, treated,
                                                                                            treatment) -> None:
    from metrics.interventions import intervention_seed

    run = treated[treatment]
    record = json.loads((run.omni / onset.INTERVENTION_FILE).read_text())
    assert record["intervention"] == treatment and record["depth"] == R.INTERVENTION_LAYERS
    assert (record["onset_step"], record["onset_epoch"], record["run_seed"]) == (2 * TINY, 2, 0)
    assert record["generator_seed"] == intervention_seed(0) and record["global_rng_untouched"] is True
    assert record["run_id"] == run.spec.run_id and record["output_check_batch"]["normalised"] is True
    assert record["output_check_batch"]["states"] == R.FIXED_BATCH_STATES
    # the onset checkpoint and everything before it are the untreated run's, bit for bit
    for epoch in (0, 1, 2):
        mine, partner = _checkpoint(run.omni, epoch), _checkpoint(abrupt.omni, epoch)
        for key in partner:
            assert _same(mine[key], partner[key]), (epoch, key)
    lines, partner_lines = _plasticity_lines(run.omni), _plasticity_lines(abrupt.omni)
    assert sorted(lines) == [0, TINY, 2 * TINY, 3 * TINY, 4 * TINY]
    for step in (0, TINY, 2 * TINY):  # the onset row is pre-intervention
        assert lines[step] == partner_lines[step]
    assert lines[3 * TINY] != partner_lines[3 * TINY]
    assert not _same(_checkpoint(run.omni, 3)["pi"], _checkpoint(abrupt.omni, 3)["pi"])
    with pytest.raises(RunRefused, match="once"):
        run.algo._apply_intervention()  # a second application is refused


def test_partial_reset_reinitialises_the_last_two_layers_and_clears_their_adam_state(abrupt, treated) -> None:
    run = treated["reset"]
    record = json.loads((run.omni / onset.INTERVENTION_FILE).read_text())
    assert record["layers"] == ["mean.2", "mean.4"] and record["adam_entries_cleared"] == 4
    onset_state, after = _checkpoint(run.omni, 2), _checkpoint(run.omni, 3)
    steps_before = {i: float(s["step"]) for i, s in onset_state["actor_optimizer"]["state"].items()}
    steps_after = {i: float(s["step"]) for i, s in after["actor_optimizer"]["state"].items()}
    epoch_steps = steps_after[0] - steps_before[0]  # log_std: never reset
    assert epoch_steps > 0
    # parameter order (OmniSafe's Adam(actor.parameters())): log_std, mean.0.*, mean.2.*, mean.4.*
    assert [steps_after[i] - steps_before[i] for i in (0, 1, 2)] == [epoch_steps] * 3
    assert [steps_after[i] for i in (3, 4, 5, 6)] == [epoch_steps] * 4  # restarted at the reset
    assert after["pi"].keys() == onset_state["pi"].keys()  # the plain layout
    for key in ("mean.2.weight", "mean.2.bias", "mean.4.weight", "mean.4.bias"):
        assert not torch.equal(after["pi"][key], _checkpoint(abrupt.omni, 3)["pi"][key])
    from metrics.interventions import build_actor

    config = json.loads((run.omni / "config.json").read_text())
    pi = _checkpoint(run.omni, 4)["pi"]
    actor = build_actor(config, pi["mean.0.weight"].shape[1], pi["mean.4.weight"].shape[0], pi)
    x = torch.randn(16, pi["mean.0.weight"].shape[1], generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        assert torch.equal(actor.mean(x), run.algo._actor_critic.actor.mean(x))  # the final checkpoint is the learner


def test_injection_keeps_the_output_and_trains_only_the_trainable_layout(treated) -> None:
    from metrics.interventions import InjectedHead, build_actor

    run = treated["injection"]
    record = json.loads((run.omni / onset.INTERVENTION_FILE).read_text())
    assert record["output_identical"] is True and record["output_max_abs_change"] == 0.0
    assert record["trainable_parameter_count_after"] == record["trainable_parameter_count_before"]
    onset_pi = _checkpoint(run.omni, 2)["pi"]
    states = {epoch: _checkpoint(run.omni, epoch)["pi"] for epoch in (3, 4)}
    for epoch, pi in states.items():
        assert dependencies.INJECTED_KEY in pi
        for layer in ("2", "4"):
            for part in ("weight", "bias"):
                assert torch.equal(pi[f"mean.frozen.{layer}.{part}"], onset_pi[f"mean.{layer}.{part}"])  # frozen head
                assert torch.equal(pi[f"mean.new_frozen.{layer}.{part}"], states[3][f"mean.new_frozen.{layer}.{part}"])
    assert not torch.equal(states[4]["mean.new.4.weight"], states[4]["mean.new_frozen.4.weight"])  # the new copy trains
    assert not torch.equal(states[4]["mean.trunk.0.weight"], onset_pi["mean.0.weight"])  # the trunk trains (Q-reset-injection)
    assert not torch.equal(states[4]["log_std"], onset_pi["log_std"])
    config = json.loads((run.omni / "config.json").read_text())
    obs_dim, act_dim = onset_pi["mean.0.weight"].shape[1], onset_pi["mean.4.weight"].shape[0]
    live = run.algo._actor_critic.actor
    x = torch.randn(16, obs_dim, generator=torch.Generator().manual_seed(1))
    for epoch in (3, 4):
        actor = build_actor(config, obs_dim, act_dim, _checkpoint(run.omni, epoch)["pi"])
        assert isinstance(actor.mean, InjectedHead)
        if epoch == 4:  # the final checkpoint is the live learner
            with torch.no_grad():
                assert torch.equal(actor.mean(x), live.mean(x))
    assert isinstance(live.mean, InjectedHead)

"""Role 2's battery continuations on a tiny real parent (2,000 steps per epoch, 2 epochs per run).

Run with ``pytest -m slow``. The parent is a registered N = 0 spec trained by the pipeline's own
PPO-Lagrangian with full-state checkpoints; each continuation is a smoke copy of the spec
``pilot.manifest.battery_continuation`` builds (a smoke copy's run is shrunk), resolved and
configured by the launcher's own functions, and built through the plug-in's factory.

* Fine-tuning (Table 2.2 "Reward-only fine-tuning"): ``epoch-0.pt`` holds the parent's actor,
  critics and normaliser bit for bit with fresh optimisers; the multiplier is 0.0 in every epoch and
  never updated; the actor trains on A_R alone; the cost critic keeps training.
* Transfer (Table 2.2 "Transfer"; Q-transfer-obs): SafetyPointGoal1-v0 -> SafetyPointButton1-v0; the
  restored learner is the parent's mapped by component name; the multiplier starts fresh at 0.001
  and is updated once per epoch with d = 25; a target-only normaliser dimension keeps variance 1
  after the first push.
* ``Onset/Active`` is 0 in every fine-tuning epoch and 1 in every transfer epoch (a diagnostic of
  contract 3, pilot/contracts.py, as envs.onset logs it).
"""

from __future__ import annotations

import copy
import csv
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from configs import registered as R
from pilot import dependencies, launch, manifest, provenance
from pilot.manifest import RunSpec

pytestmark = [pytest.mark.omnisafe, pytest.mark.slow]

torch = pytest.importorskip("torch")
pytest.importorskip("omnisafe")

from envs import continuations as C  # noqa: E402  # after the skip: it imports OmniSafe
from envs.onset import SMOKE_PREFIX  # noqa: E402

TINY = 2_000
E = R.STEPS_PER_EPOCH
COMMIT = "a" * 40
PARENT_STEP = 480 * E  # a registered matched checkpoint step; overridden below by the tiny parent's


def _tiny(cfgs: Any, *, epochs: int) -> Any:
    cfgs.algo_cfgs.steps_per_epoch = TINY
    cfgs.train_cfgs.total_steps = TINY * epochs
    cfgs.train_cfgs.epochs = epochs
    cfgs.logger_cfgs.save_model_freq = 1
    cfgs.logger_cfgs.use_tensorboard = False
    cfgs.pilot_cfgs.plasticity = False
    cfgs.pilot_cfgs.extra_checkpoint_steps = []
    return cfgs


def _omnisafe_dir(run_dir: Path) -> Path:
    (path,) = [p for p in (run_dir / "omnisafe").glob("*/seed-*") if p.is_dir()]
    return path


def _rows(omni: Path) -> list[dict[str, str]]:
    with open(omni / "progress.csv", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _checkpoint(omni: Path, epoch: int) -> dict:
    return torch.load(omni / "torch_save" / f"epoch-{epoch}.pt", weights_only=True, map_location="cpu")


def _equal_states(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(torch.equal(torch.as_tensor(a[k]), torch.as_tensor(b[k])) for k in a)


@pytest.fixture(scope="module")
def parent(tmp_path_factory):
    from pilot.algorithms import make_ppolag

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        root = tmp_path_factory.mktemp("data") / "checkpoints"
        spec = next(s for s in manifest.design("main") if s.task == R.PRIMARY_TASK and s.N == 0.0 and s.seed == 0)
        run_dir = root / spec.run_id
        cfgs = _tiny(launch.build_config(spec, run_dir / "omnisafe"), epochs=2)
        make_ppolag(spec.task, cfgs, spec).learn()
        omni = _omnisafe_dir(run_dir)
        (run_dir / "spec.json").write_text(spec.to_json(), encoding="utf-8")
        (run_dir / "train_result.json").write_text(json.dumps({  # what the launcher writes (dependencies.resolve reads it)
            "run_id": spec.run_id, "status": "completed", "commit_hash": COMMIT,
            "config_hash": provenance.config_hash(cfgs.todict()), "omnisafe_dir": str(omni.relative_to(run_dir)),
        }), encoding="utf-8")
        yield SimpleNamespace(spec=spec, root=root, omni=omni, state=_checkpoint(omni, 2))
    finally:
        torch.set_num_threads(threads)


def _child(parent: SimpleNamespace, condition: str) -> SimpleNamespace:
    """The smoke copy of the parent's ``condition`` continuation from its checkpoint after 2 tiny epochs, built."""
    from results.ledger_schema import relative_checkpoint_path

    row = {"run_id": parent.spec.run_id, "matched": True, "completed": True, "matched_checkpoint_step": PARENT_STEP,
           "matched_checkpoint_path": "x", "commit_hash": COMMIT}
    spec = manifest.battery_continuation(parent.spec, row, condition)
    path = relative_checkpoint_path(os.path.abspath(parent.omni / "torch_save" / "epoch-2.pt"))
    spec = RunSpec.from_dict({**spec.to_dict(), "run_id": SMOKE_PREFIX + spec.run_id, "total_steps": 2 * E,
                              "params": {**spec.params, "parent_step": 2 * TINY, "parent_checkpoint": path}})
    run_dir = parent.root / spec.run_id
    deps = dependencies.resolve(spec, run_dir)
    cfgs = _tiny(launch.build_config(spec, run_dir / "omnisafe", deps), epochs=2)
    factory = C.make_finetune_algorithm if condition == "finetune" else C.make_transfer_algorithm
    algo = factory(spec.task, cfgs, spec)
    events: list[tuple] = []
    original_update = algo._lagrange.update_lagrange_multiplier
    original_surrogate = algo._compute_adv_surrogate

    def update(cost):
        original_update(cost)
        events.append(("update", int(algo._logger.current_epoch)))

    def surrogate(adv_r, adv_c):
        out = original_surrogate(adv_r, adv_c)
        m = algo._lagrange.lagrangian_multiplier.item()
        events.append(("surrogate", int(algo._logger.current_epoch), bool(torch.equal(out, adv_r)),
                       bool(torch.equal(out, (adv_r - m * adv_c) / (1 + m)))))
        return out

    algo._lagrange.update_lagrange_multiplier = update
    algo._compute_adv_surrogate = surrogate
    return SimpleNamespace(spec=spec, algo=algo, cfgs=cfgs, omni=_omnisafe_dir(run_dir), events=events)


def test_fine_tuning_restores_the_parent_and_trains_on_the_reward_advantage_with_the_multiplier_at_zero(parent) -> None:
    run = _child(parent, "finetune")
    assert type(run.algo) is C.FinetunePPOLag
    start = _checkpoint(run.omni, 0)
    for key in ("pi", "reward_critic", "cost_critic", "obs_normalizer"):
        assert _equal_states(start[key], parent.state[key]), key  # the restored learner, bit for bit
    for key in ("actor_optimizer", "reward_critic_optimizer", "cost_critic_optimizer", "lambda_optimizer"):
        assert start[key]["state"] == {}, key  # fresh (Q-continuations)
    assert start["lagrange"]["value"].item() == 0.0
    summary = json.loads((run.omni / C.CONTINUATION_FILE).read_text(encoding="utf-8"))
    assert summary["condition"] == "finetune" and summary["parent_step"] == 2 * TINY
    assert summary["restored"] == ["pi", "reward_critic", "cost_critic", "obs_normalizer"]
    assert summary["observation_mapping"] is False and summary["transfer_map"] is None
    assert summary["multiplier"]["value"] == 0.0
    saved = json.loads((run.omni / "config.json").read_text(encoding="utf-8"))["continuation_cfgs"]
    assert saved["condition"] == "finetune" and saved["parent_run_id"] == parent.spec.run_id
    run.algo.learn()
    rows = _rows(run.omni)
    assert [float(r["Metrics/LagrangeMultiplier"]) for r in rows] == [0.0, 0.0]  # stored every epoch (contract 3)
    assert [float(r["Onset/Active"]) for r in rows] == [0.0, 0.0]  # never constrained (envs.continuations)
    assert all(math.isfinite(float(r[k])) for r in rows for k in r if k.startswith(("Loss/", "Metrics/Lagrange")))
    assert not [e for e in run.events if e[0] == "update"]  # never updated (the multiplier update is never called)
    surrogates = [e for e in run.events if e[0] == "surrogate"]
    assert surrogates and all(e[2] for e in surrogates)  # A_R exactly
    end = _checkpoint(run.omni, 2)
    assert end["lagrange"]["value"].item() == 0.0 and end["lambda_optimizer"]["state"] == {}
    assert not _equal_states(end["cost_critic"], start["cost_critic"])  # the cost critic keeps training
    assert not _equal_states(end["pi"], start["pi"])
    assert int(end["obs_normalizer"]["_count"]) >= int(parent.state["obs_normalizer"]["_count"]) + 2 * TINY  # it keeps updating


def test_transfer_restores_the_mapped_parent_with_a_fresh_multiplier_on_the_held_out_task(parent) -> None:
    run = _child(parent, "transfer")
    assert type(run.algo) is C.TransferPPOLag and run.spec.task == "SafetyPointButton1-v0"
    tmap = C.transfer_map(parent.spec.task, run.spec.task)
    assert run.algo._env.observation_space.shape == (tmap.target_dim,) == (76,)
    mapped = C.map_state_for_transfer(parent.state, tmap)
    start = _checkpoint(run.omni, 0)
    for key in ("pi", "reward_critic", "cost_critic", "obs_normalizer"):
        assert _equal_states(start[key], mapped[key]), key
    only = list(tmap.target_only_columns())
    assert torch.equal(start["pi"]["mean.0.weight"][:, only], torch.zeros(R.HIDDEN_SIZES[0], len(only)))
    assert start["lagrange"]["value"].item() == float(np.float32(R.LAGRANGE_MULTIPLIER_INIT))  # fresh
    assert start["lagrange"]["numeric_attributes"]["cost_limit"] == R.COST_LIMIT
    summary = json.loads((run.omni / C.CONTINUATION_FILE).read_text(encoding="utf-8"))
    assert summary["observation_mapping"] is True and summary["transfer_map"] == tmap.to_dict()
    assert summary["multiplier"]["init"] == R.LAGRANGE_MULTIPLIER_INIT
    # the live normaliser: a target-only dimension keeps variance 1 after one push (envs/continuations.py)
    probe = copy.deepcopy(run.algo._env.save()["obs_normalizer"])  # a copy: the run's own normaliser is untouched
    count = int(probe._count)
    assert count == int(parent.state["obs_normalizer"]["_count"]) and probe._first is False
    probe.normalize(torch.rand(1, tmap.target_dim, generator=torch.Generator().manual_seed(0)))
    assert int(probe._count) == count + 1
    assert torch.all((probe._var[only] - 1).abs() <= 2.0 / count)
    run.algo.learn()
    assert [e[1] for e in run.events if e[0] == "update"] == [0, 1]  # PPO-Lagrangian from the first epoch
    surrogates = [e for e in run.events if e[0] == "surrogate"]
    assert surrogates and all(e[3] for e in surrogates)  # equation (3) with the current multiplier
    rows = _rows(run.omni)
    assert len(rows) == 2 and all(math.isfinite(float(r["Metrics/LagrangeMultiplier"])) for r in rows)
    assert [float(r["Onset/Active"]) for r in rows] == [1.0, 1.0]  # constrained from the first epoch
    end = _checkpoint(run.omni, 2)
    assert float(end["lambda_optimizer"]["state"][0]["step"]) == 2

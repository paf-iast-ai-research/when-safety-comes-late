"""The ledger writer and the enrichment on the REAL harness output of a real, tiny training run (``slow``).

One PPO-Lagrangian epoch of 2,000 steps (``FullStateCheckpointMixin`` over OmniSafe's PPOLag, built
directly with a small configuration, never through 20,000-step epochs) gives a real checkpoint. A
registered-looking run directory links every checkpoint of a main-study spec to it (as
tests/test_envs_evaluation_slow.py does), and a fine-tuning continuation's directory links its final
checkpoint. Then, with only
``envs.evaluation.run_episodes`` replaced by a fast deterministic stub (the policies are loaded,
checked and built for real):

* ``envs.evaluation.evaluate_run`` (contract 1) gives evaluation.json as ``pilot.launch.evaluate``
  writes it, and ``pilot.ledger_writer.write_run`` turns it into the row and its ``evaluation``
  supplement record;
* the Study A enrichment commands select, measure, battery, controller, training, final-battery and
  continuations run with the harness and the metrics they load themselves (``pilot.enrichment._load``:
  ``analysis.matching``'s checkpoint selection, ``envs.evaluation.evaluate_battery`` and
  ``evaluate_continuation``, ``metrics.controller``, ``metrics.recovery``), so the records are built
  from the real result shapes and validated by ``results.supplement_schema``. Matching is written by
  hand (``match``, rules 2 to 6, compares whole arms of five seeds; this test has one run) and the
  sensitivity battery is not run;
* the ledger and its supplement then read as one dataset without a data problem (``analysis.data``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
import fake_launcher  # noqa: E402

from configs import registered as R  # noqa: E402
from pilot import contracts, enrichment, manifest, supplement  # noqa: E402
from pilot.ledger_writer import LedgerPaths, write_run  # noqa: E402

pytestmark = [pytest.mark.omnisafe, pytest.mark.slow]

TASK = "SafetyPointGoal1-v0"
SPE = 2_000


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
                  "train_cfgs": {"device": "cpu", "torch_threads": R.TORCH_THREADS, "vector_env_nums": 1, "parallel": 1,
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
    assert (directory / "torch_save" / "epoch-1.pt").is_file()
    return directory


def _linked_run(trained: Path, root: Path, spec: manifest.RunSpec) -> Path:
    """``root/<run_id>``: the fake launcher's files, every checkpoint linked to the trained epoch-1.pt,
    and the trained config.json made the spec's (total steps, run_id, the registered epoch length)."""
    run_dir = root / spec.run_id
    run_dir.mkdir(parents=True)
    (run_dir / "spec.json").write_text(spec.to_json())
    omni = run_dir / fake_launcher.write_outputs(spec.to_dict(), run_dir)
    for checkpoint in (omni / "torch_save").glob("epoch-*.pt"):
        checkpoint.unlink()
        checkpoint.symlink_to(trained / "torch_save" / "epoch-1.pt")
    cfg = json.loads((trained / "config.json").read_text(encoding="utf-8"))
    cfg["train_cfgs"].update({"total_steps": spec.total_steps, "epochs": spec.total_steps // R.STEPS_PER_EPOCH})
    cfg["algo_cfgs"]["steps_per_epoch"] = R.STEPS_PER_EPOCH
    cfg["pilot_cfgs"] = {"run_id": spec.run_id, "onset_step": int(spec.onset_step or 0)}
    (omni / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    deps = fake_launcher.resolve_dependencies(spec.to_dict(), run_dir) if spec.depends_on else {}
    (run_dir / "train_result.json").write_text(json.dumps(
        fake_launcher.train_result(spec.to_dict(), str(omni.relative_to(run_dir)), "completed", None, deps)))
    return run_dir


class Episodes:
    """``run_episodes`` replaced: deterministic costs by seed and checkpoint, and the checkpoint step of each real
    policy recorded (``steps``)."""

    def __init__(self) -> None:
        self.steps: list[int] = []

    def __call__(self, policy, seeds, **kwargs):
        from envs import evaluation as E

        self.steps.append(policy.step)
        shift = policy.step // R.CHECKPOINT_INTERVAL_STEPS
        return [E.EpisodeResult(seed=s, cost=float((s + shift) % 9) + 20.0, ret=(s % 4) / 8.0,
                                length=R.EPISODE_LENGTH) for s in seeds]


def test_real_harness_output_through_the_ledger_writer_and_the_study_a_commands(trained, tmp_path, monkeypatch) -> None:
    from analysis.data import load_dataset
    from envs import evaluation as E
    from results.ledger_schema import load_ledger_as_rows

    from pilot.launch import plain

    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset(R.PENDING))  # every key answered, as decided
    episodes = Episodes()
    monkeypatch.setattr(E, "run_episodes", episodes)
    spec = next(s for s in manifest.study_a_main((0,)) if s.task == TASK and s.N == 0.0)
    root = tmp_path / "data" / "checkpoints"
    run_dir = _linked_run(trained, root, spec)
    omni = run_dir / json.loads((run_dir / "train_result.json").read_text())["omnisafe_dir"]

    # contract 1, then evaluation.json as pilot.launch.evaluate writes it
    evaluation = plain(dict(E.evaluate_run(str(omni), spec.to_dict())))
    evaluation.update(eval_wall_clock_hours=0.01, eval_commit_hash=fake_launcher.COMMIT, eval_allow_dirty=False,
                      eval_allow_pending=False, eval_worktree_dirty=False)
    contracts.validate_evaluation(evaluation, spec, omni)
    (run_dir / "evaluation.json").write_text(json.dumps(evaluation))
    paths = LedgerPaths(main=tmp_path / "ledger" / "ledger.parquet", pilot=tmp_path / "pilot" / "ledger.parquet",
                        sidecar_dir=tmp_path / "pilot" / "sidecar")
    ledger = write_run(spec, run_dir, paths)
    record = supplement.read("evaluation", spec.run_id, ledger_path=ledger)
    (row,) = load_ledger_as_rows(ledger)
    assert record.final.mean_cost == row.final_cost
    assert record.selection[0].seeds == E.selection_seeds() and record.final.seeds == E.measurement_seeds()

    run_dir_of = root.joinpath
    assert enrichment.apply_selection(ledger) == [spec.run_id]  # analysis.matching.select_checkpoint, for real
    assert enrichment.apply_measurement(ledger, run_dir_of) == [spec.run_id]
    enrichment._write(ledger, spec.run_id, {"matched": True, "infeasible": False})  # one run: rule 5 needs arms
    assert enrichment.apply_battery(ledger, run_dir_of, ["hazard", "dynamics"]) == [spec.run_id]
    assert enrichment.apply_controller(ledger, run_dir_of) == [spec.run_id]
    assert enrichment.apply_training(ledger, run_dir_of) == [spec.run_id]
    assert enrichment.apply_final_battery(ledger, run_dir_of, ["hazard"]) == [spec.run_id]
    (row,) = load_ledger_as_rows(ledger)
    hazard = supplement.read("battery", spec.run_id, ledger_path=ledger, part="hazard")
    assert hazard.hazard.form == E.HAZARD_FORM and hazard.hazard.layout_seeds == E.hazard_layout_seeds()
    assert hazard.gap == row.gap_hazard and row.measurement_cost == hazard.measurement_cost

    # the fine-tuning continuation of the matched checkpoint (Table 2.2), evaluated by the real harness
    cont = manifest.battery_continuation(spec, row.model_dump(mode="python"), "finetune")
    _linked_run(trained, root, cont)
    assert enrichment.apply_continuations(ledger, run_dir_of, ["finetune"]) == [spec.run_id]
    (row,) = load_ledger_as_rows(ledger)
    continuation = supplement.read("continuation", spec.run_id, ledger_path=ledger, part="finetune")
    assert continuation.step == cont.total_steps and row.gap_finetune == continuation.gap
    assert episodes.steps[-1] == cont.total_steps  # the continuation's final checkpoint

    dataset = load_dataset(ledger, mode="final", cuts=())
    assert dataset.problems == [] and dataset.study_a[0]["final_gap_hazard"] is not None

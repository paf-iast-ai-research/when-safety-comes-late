"""The determinism check of pilot/scheduler.py with tiny real trainings (2,000 steps per epoch, 2 epochs per run).

``scripts/determinism_check.py`` compares two runs of one configuration bit for bit (``compare``) and,
in its registered form, by the evaluation cost of a checkpoint (``registered_form``, through Role 2's
``envs.evaluation.evaluate_checkpoint``). Every key is answered (docs/DECISIONS.md), so no check waits on
an open question (tests/test_integration_sched_scripts.py). Here ``compare`` is applied to tiny runs built
from ``pilot.scheduler.determinism_spec`` for the plug-ins ppolag, unconstrained_ppo, study_a and study_b,
and ``registered_form`` to the first three. Left out, because a tiny run cannot carry them:

* study_b's registered form: the budget-conditioned harness needs the spec, and checks config.json's
  total_steps against it, which a tiny run's 4,000 steps cannot match (a RunSpec's total is a multiple
  of 20,000 steps);
* the study_a_pid check: its changed onset (``pilot.scheduler.DETERMINISM_LATE_ONSET``, N x the check's
  length) lies at step 100,000, beyond a tiny run, and the plug-in refuses an onset that leaves no
  constrained epoch (the PID check's onset path is trained in tests/test_envs_onset_slow.py).

The OmniSafe configurations are made directly (``launch.build_config`` shrunk to tiny epochs, as
tests/test_core_restore.py does), never through 20,000-step epochs. Run with ``pytest -m slow``.
"""

from __future__ import annotations

import importlib.util
import types
from dataclasses import replace
from pathlib import Path

import pytest

from configs import registered as R
from pilot import algorithms, launch
from pilot import scheduler as S
from pilot.manifest import RunSpec

pytestmark = [pytest.mark.omnisafe, pytest.mark.slow]

REPO = Path(__file__).resolve().parents[1]
TINY = 2_000  # steps per epoch
EPOCHS = 2


def _det() -> types.ModuleType:
    """scripts/determinism_check.py, loaded as a module."""
    spec = importlib.util.spec_from_file_location("determinism_check", REPO / "scripts" / "determinism_check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _train(spec: RunSpec, run_dir: Path) -> Path:
    """Train ``spec`` in-process with tiny epochs; its OmniSafe run directory."""
    cfgs = launch.build_config(spec, run_dir / "omnisafe")
    cfgs.algo_cfgs.steps_per_epoch = TINY
    cfgs.train_cfgs.total_steps = TINY * EPOCHS
    cfgs.train_cfgs.epochs = EPOCHS
    cfgs.logger_cfgs.save_model_freq = 1
    cfgs.logger_cfgs.use_tensorboard = False
    cfgs.pilot_cfgs.plasticity = False
    cfgs.pilot_cfgs.extra_checkpoint_steps = []
    algorithms.load_plugin(spec.plugin)(spec.task, cfgs, spec).learn()
    (path,) = [p for p in (run_dir / "omnisafe").glob("*/seed-*") if p.is_dir()]
    return path


@pytest.fixture(scope="module")
def one_thread():
    torch = pytest.importorskip("torch")
    pytest.importorskip("omnisafe")
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


@pytest.mark.parametrize("plugin", ["ppolag", "unconstrained_ppo", "study_a", "study_b"])
def test_the_determinism_check_finds_two_runs_of_a_configuration_identical(plugin, one_thread, tmp_path) -> None:
    from envs.evaluation import evaluate_checkpoint

    det = _det()
    spec = S.determinism_spec(plugin, R.CHECKPOINT_INTERVAL_STEPS)
    a = _train(spec, tmp_path / "runA")
    b = _train(spec, tmp_path / "runB")
    result = det.compare(a, b)
    assert result["identical"], result["differences"]
    assert result["epochs_compared"] == EPOCHS and result["tensors_compared"] > 0
    if plugin != "study_b":  # its harness needs the spec, which a tiny run cannot match (module docstring)
        step = TINY * EPOCHS
        registered = det.registered_form(a, b, evaluate_checkpoint, allow_dirty=True, step=step, episodes=2)
        assert registered["identical"] and registered["checkpoint_step"] == step
    # and it tells two different runs apart: another seed, in a smoke copy of the spec (a plug-in refuses a config seed
    # that is not its spec's, and a DET- spec that is not determinism_spec's)
    seed = spec.seed + 1
    other = replace(spec, seed=seed, run_id=f"SMOKE-{spec.run_id.removesuffix(f'-s{spec.seed}')}-s{seed}")
    c = _train(other, tmp_path / "runC")
    assert not det.compare(a, c)["identical"]

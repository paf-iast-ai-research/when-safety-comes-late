"""The launcher against the pinned OmniSafe (Table 3.1; Appendix A, Table A.1).

Tests marked ``omnisafe`` need the pinned stack and are skipped without it; the test marked ``slow``
trains two epochs twice (about ten minutes on one CPU thread) and runs with ``pytest -m slow``.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from configs import registered as R
from pilot.manifest import RunSpec

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs" / "omnisafe"


def _det_spec(total_steps: int = R.STEPS_PER_EPOCH) -> RunSpec:
    return RunSpec(run_id="DET-PPOLag-PointGoal1-s0", study="A", task=R.PRIMARY_TASK, arm="determinism-check",
                   seed=0, total_steps=total_steps, base_algo="PPOLag", plugin="ppolag", group="determinism")


# ---------------------------------------------------------------------------
# Committed configuration copies (no OmniSafe needed)
# ---------------------------------------------------------------------------


def test_committed_configs_match_their_recorded_hashes() -> None:
    source = json.loads((CONFIGS / "SOURCE.json").read_text())
    assert source["omnisafe_version"] == "0.5.0"
    for name, digest in source["files"].items():
        assert hashlib.sha256((CONFIGS / name).read_bytes()).hexdigest() == digest


def test_appendix_a_values_are_in_the_copied_configs() -> None:
    yaml = pytest.importorskip("yaml")
    ppolag = yaml.safe_load((CONFIGS / "PPOLag.yaml").read_text())["defaults"]
    assert ppolag["train_cfgs"]["total_steps"] == R.TOTAL_STEPS
    assert ppolag["algo_cfgs"]["steps_per_epoch"] == R.STEPS_PER_EPOCH
    assert ppolag["algo_cfgs"]["update_iters"] == R.UPDATE_ITERS
    assert ppolag["algo_cfgs"]["batch_size"] == R.MINIBATCH_SIZE
    assert ppolag["algo_cfgs"]["target_kl"] == R.TARGET_KL
    assert ppolag["lagrange_cfgs"]["cost_limit"] == R.COST_LIMIT
    assert ppolag["lagrange_cfgs"]["lagrangian_multiplier_init"] == R.LAGRANGE_MULTIPLIER_INIT
    assert ppolag["lagrange_cfgs"]["lambda_lr"] == R.LAGRANGE_MULTIPLIER_LR
    for net in ("actor", "critic"):
        assert ppolag["model_cfgs"][net]["hidden_sizes"] == list(R.HIDDEN_SIZES)
        assert ppolag["model_cfgs"][net]["activation"] == R.ACTIVATION
    assert ppolag["train_cfgs"]["device"] == R.DEVICE
    assert ppolag["train_cfgs"]["vector_env_nums"] == R.VECTOR_ENV_NUMS
    assert ppolag["train_cfgs"]["parallel"] == R.PARALLEL_PROCESSES
    # Documented deviations handled by the launcher's registered overrides:
    assert ppolag["train_cfgs"]["torch_threads"] == 16  # Table 3.1 requires 1
    assert ppolag["logger_cfgs"]["save_model_freq"] == 100  # Part 3.4 requires every 10 epochs
    pid = yaml.safe_load((CONFIGS / "CPPOPID.yaml").read_text())["defaults"]
    assert pid["lagrange_cfgs"]["cost_limit"] == R.COST_LIMIT
    assert pid["lagrange_cfgs"]["lagrangian_multiplier_init"] == R.LAGRANGE_MULTIPLIER_INIT
    for key in ("train_cfgs", "algo_cfgs", "model_cfgs"):
        assert pid[key] == ppolag[key]  # the PID check differs from PPO-Lagrangian only in the multiplier


# ---------------------------------------------------------------------------
# Launcher with the installed OmniSafe
# ---------------------------------------------------------------------------


@pytest.mark.omnisafe
def test_committed_configs_equal_the_installed_package() -> None:
    omnisafe = pytest.importorskip("omnisafe")
    installed = Path(omnisafe.__file__).parent / "configs" / "on-policy"
    for name in ("PPOLag.yaml", "CPPOPID.yaml", "PPO.yaml"):
        assert (installed / name).read_bytes() == (CONFIGS / name).read_bytes()


@pytest.mark.omnisafe
def test_build_config_applies_exactly_the_registered_overrides(tmp_path) -> None:
    pytest.importorskip("omnisafe")
    from omnisafe.utils.config import get_default_kwargs_yaml

    from pilot.launch import build_config

    cfgs = build_config(_det_spec(10_000_000), tmp_path)
    assert cfgs.seed == 0
    assert cfgs.train_cfgs.torch_threads == 1 and cfgs.train_cfgs.device == "cpu"
    assert cfgs.train_cfgs.epochs == 500 and cfgs.train_cfgs.total_steps == 10_000_000
    assert cfgs.logger_cfgs.save_model_freq == 10 and cfgs.logger_cfgs.log_dir == str(tmp_path)
    assert cfgs.exp_name == "PPOLag-{SafetyPointGoal1-v0}" and cfgs.algo == "PPOLag"
    default = get_default_kwargs_yaml("PPOLag", R.PRIMARY_TASK, "on-policy").todict()
    built = cfgs.todict()
    changed = {k for k in default if default[k] != built[k]}
    assert changed == {"train_cfgs", "logger_cfgs"}  # seed 0 and steps_per_epoch equal their defaults
    train_changed = {k for k in built["train_cfgs"] if built["train_cfgs"][k] != default["train_cfgs"].get(k)}
    assert train_changed == {"torch_threads", "epochs"}  # device, vector envs, parallel and total steps are the defaults
    logger_changed = {k for k in built["logger_cfgs"] if built["logger_cfgs"][k] != default["logger_cfgs"].get(k)}
    assert logger_changed == {"save_model_freq", "log_dir"}
    for key, value in default["algo_cfgs"].items():
        assert built["algo_cfgs"][key] == value
    assert built["model_cfgs"] == default["model_cfgs"] and built["lagrange_cfgs"] == default["lagrange_cfgs"]


@pytest.mark.omnisafe
def test_config_hash_does_not_depend_on_the_run_directory(tmp_path) -> None:
    pytest.importorskip("omnisafe")
    from pilot.launch import build_config
    from pilot.provenance import config_hash

    a = build_config(_det_spec(10_000_000), tmp_path / "a")
    b = build_config(_det_spec(10_000_000), tmp_path / "b")
    assert config_hash(a.todict()) == config_hash(b.todict())
    assert a.pilot_cfgs.onset_step == 0 and a.pilot_cfgs.run_id == "DET-PPOLag-PointGoal1-s0"


@pytest.mark.omnisafe
def test_tampered_defaults_are_refused(tmp_path) -> None:
    pytest.importorskip("omnisafe")
    from omnisafe.utils.config import get_default_kwargs_yaml

    from pilot.launch import RunRefused, verify_registered_defaults

    cfgs = get_default_kwargs_yaml("PPOLag", R.PRIMARY_TASK, "on-policy")
    verify_registered_defaults(cfgs, "PPOLag")
    cfgs.lagrange_cfgs.lambda_lr = 0.05
    with pytest.raises(RunRefused, match="lambda_lr"):
        verify_registered_defaults(cfgs, "PPOLag")
    pid = get_default_kwargs_yaml("CPPOPID", R.PRIMARY_TASK, "on-policy")
    verify_registered_defaults(pid, "CPPOPID")
    pid.lagrange_cfgs.lagrangian_multiplier_init = 0.5
    with pytest.raises(RunRefused, match="lagrangian_multiplier_init"):
        verify_registered_defaults(pid, "CPPOPID")


@pytest.mark.omnisafe
def test_an_installed_yaml_that_differs_from_the_committed_copy_is_refused(tmp_path, monkeypatch) -> None:
    pytest.importorskip("omnisafe")
    from pilot import launch

    assert launch.verify_installed_yaml("PPOLag") == json.loads((CONFIGS / "SOURCE.json").read_text())["files"]["PPOLag.yaml"]
    tampered = tmp_path / "PPOLag.yaml"
    tampered.write_bytes((CONFIGS / "PPOLag.yaml").read_bytes() + b"# edited\n")
    monkeypatch.setattr(launch, "installed_yaml", lambda algo, algo_type="on-policy": tampered)
    with pytest.raises(launch.RunRefused, match="not the copy committed"):
        launch.verify_installed_yaml("PPOLag")


@pytest.mark.omnisafe
def test_refusals_happen_before_the_claim_and_leave_nothing(tmp_path, monkeypatch, capsys) -> None:
    pytest.importorskip("omnisafe")
    from pilot import launch

    spec = _det_spec(2 * R.STEPS_PER_EPOCH)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "spec.json").write_text(spec.to_json())
    argv = ["train", "--spec", str(run_dir / "spec.json"), "--run-dir", str(run_dir), "--allow-dirty"]
    # a configuration that differs from Table A.1 is a refusal, not a crash (never an exclusion)
    monkeypatch.setattr(R, "UPDATE_ITERS", R.UPDATE_ITERS + 1)
    assert launch.main(argv) == launch.EXIT_REFUSED
    assert "REFUSED" in capsys.readouterr().err and sorted(p.name for p in run_dir.iterdir()) == ["spec.json"]
    monkeypatch.undo()
    # an open question refuses without --allow-pending
    pending = RunSpec.from_dict({**spec.to_dict(), "pending": ["Q-rounding"]})
    (run_dir / "spec.json").write_text(pending.to_json())
    assert launch.main(argv) == launch.EXIT_REFUSED
    assert sorted(p.name for p in run_dir.iterdir()) == ["spec.json"]
    # a directory another process has claimed is refused, and its claim is left in place
    (run_dir / "spec.json").write_text(spec.to_json())
    (run_dir / launch.CLAIM_FILE).write_text("12345\n")
    assert launch.main(argv) == launch.EXIT_REFUSED
    assert (run_dir / launch.CLAIM_FILE).read_text() == "12345\n"


def test_json_results_store_numbers_not_strings(tmp_path) -> None:
    np = pytest.importorskip("numpy")
    from pilot.launch import _write_json_atomic, plain

    data = {"final_cost": np.float32(3.25), "selection": {np.int64(8_200_000): [np.float64(24.0), 1.0]}}
    assert plain(data) == {"final_cost": 3.25, "selection": {8_200_000: [24.0, 1.0]}}
    path = tmp_path / "evaluation.json"
    _write_json_atomic(path, data)
    assert json.loads(path.read_text()) == {"final_cost": 3.25, "selection": {"8200000": [24.0, 1.0]}}
    with pytest.raises(TypeError):
        _write_json_atomic(tmp_path / "bad.json", {"x": object()})
    assert sorted(p.name for p in tmp_path.iterdir()) == ["evaluation.json"]  # nothing half-written


def test_an_evaluator_module_without_its_function_is_unavailable(monkeypatch) -> None:
    from pilot import contracts
    from pilot.algorithms import PluginUnavailableError

    monkeypatch.setattr(contracts, "EVALUATOR_TARGET", "json:evaluate_run")
    with pytest.raises(PluginUnavailableError, match="no function"):
        contracts.load_evaluator()


def test_classify_training(tmp_path) -> None:
    torch = pytest.importorskip("torch")  # classify_training loads the final checkpoint with torch
    from pilot.launch import classify_training

    spec = _det_spec(2 * R.STEPS_PER_EPOCH)
    omni = tmp_path / "omnisafe" / "PPOLag-{x}" / "seed-000-t"
    omni.mkdir(parents=True)
    header = "Train/Epoch,TotalEnvSteps,Loss/Loss_pi,Metrics/LagrangeMultiplier\n"
    (omni / "progress.csv").write_text(header + "0,20000,0.1,0.036\n1,40000,0.2,0.07\n")
    # every epoch logged but no final checkpoint (a signal between OmniSafe's last log row and its save)
    status, cause, detail = classify_training(tmp_path, spec, None)
    assert (status, cause) == ("failed", "incomplete") and "missing or unreadable" in detail
    (omni / "torch_save").mkdir()
    (omni / "torch_save" / "epoch-2.pt").write_bytes(b"truncated by a full disk")
    assert classify_training(tmp_path, spec, RuntimeError("No space left on device"))[1] == "crash"
    torch.save({"pi": {"w": torch.zeros(1)}}, omni / "torch_save" / "epoch-2.pt")
    assert classify_training(tmp_path, spec, None) == ("completed", None, "")
    (omni / "progress.csv").write_text(header + "0,20000,0.1,0.036\n")
    assert classify_training(tmp_path, spec, None)[1] == "incomplete"
    (omni / "progress.csv").write_text(header + "0,20000,nan,0.036\n1,40000,0.2,0.07\n")
    assert classify_training(tmp_path, spec, None)[1] == "non_finite_loss"
    (omni / "progress.csv").write_text(header + "0,20000,0.1,inf\n1,40000,0.2,0.07\n")
    assert classify_training(tmp_path, spec, None)[1] == "non_finite_multiplier"
    (omni / "progress.csv").write_text(header + "0,20000,0.1,0.036\n")
    assert classify_training(tmp_path, spec, RuntimeError("boom"))[1] == "crash"
    # a non-finite model state found after an exception is not a crash
    assert classify_training(tmp_path, spec, RuntimeError("boom"), "non_finite_multiplier")[1] == "non_finite_multiplier"
    # a NaN reaching torch.distributions with a finite model and multiplier is a crash, not a non-finite loss
    nan_msg = ValueError("Expected parameter loc (Tensor) of distribution Normal to satisfy the constraint Real(), but found invalid values")
    assert classify_training(tmp_path, spec, nan_msg)[1] == "crash"
    # every Loss/ and Metrics/LagrangeMultiplier* column is checked (per-level multipliers of Study B)
    (omni / "progress.csv").write_text(
        "Train/Epoch,TotalEnvSteps,Loss/Loss_extra,Metrics/LagrangeMultiplier/level_10\n0,20000,0.1,nan\n1,40000,0.2,0.1\n"
    )
    assert classify_training(tmp_path, spec, None)[1] == "non_finite_multiplier"
    # an error after the last epoch, with the final checkpoint saved, does not exclude a complete run
    (omni / "progress.csv").write_text(header + "0,20000,0.1,0.036\n1,40000,0.2,0.07\n")
    status, cause, detail = classify_training(tmp_path, spec, RuntimeError("env.close failed"))
    assert (status, cause) == ("completed", None) and "run kept" in detail
    # a row cut short by a full disk is not a non-finite loss: the run crashed (OmniSafe saves the
    # final checkpoint after logging the last row, so a full disk leaves none)
    (omni / "torch_save" / "epoch-2.pt").unlink()
    (omni / "progress.csv").write_text(header + "0,20000,0.1,0.036\n1,40000,1.25e")
    assert classify_training(tmp_path, spec, OSError(28, "No space left on device"))[1] == "crash"


def test_selection_window_and_required_checkpoints() -> None:
    from pilot.contracts import ContractError, expected_checkpoint_steps, selection_window

    grid = list(range(0, 10_000_001, 200_000))
    assert selection_window(grid, 10_000_000) == list(range(8_200_000, 10_000_001, 200_000))
    off_grid = list(range(0, 11_000_001, 200_000)) + [11_120_000]  # constrained-steps N = 0.10 total
    with pytest.raises(ContractError, match="Q-selection-window"):
        selection_window(off_grid, 11_120_000)
    spec = RunSpec(run_id="X-s0", study="A", task=R.PRIMARY_TASK, arm="N0.25-abrupt-total", seed=0, total_steps=10_000_000,
                   base_algo="PPOLag", plugin="study_a", group="main", N=0.25, onset_step=2_500_000)
    steps = expected_checkpoint_steps(spec)
    assert 2_500_000 in steps and steps[0] == 0 and steps[-1] == 10_000_000 and len(steps) == 52


def test_off_grid_totals_wait_for_the_selection_window_question() -> None:
    from pilot import manifest

    for s in manifest.design("main") + manifest.design("treatment"):
        off = s.total_steps % R.CHECKPOINT_INTERVAL_STEPS != 0
        assert ("Q-selection-window" in s.pending) == off


@pytest.mark.omnisafe
@pytest.mark.slow
def test_launcher_reproduces_omnisafe_agent_bit_for_bit(tmp_path) -> None:
    """Two epochs through pilot.launch equal two epochs through omnisafe.Agent (same seed, 1 thread),
    and the pilot owner's mixin adds the onset checkpoint and the per-epoch batch metrics."""
    import csv
    import subprocess
    import sys

    pytest.importorskip("omnisafe")
    spec = RunSpec.from_dict({**_det_spec(2 * R.STEPS_PER_EPOCH).to_dict(), "onset_step": R.STEPS_PER_EPOCH})
    run_dir = tmp_path / "launch"
    run_dir.mkdir()
    (run_dir / "spec.json").write_text(spec.to_json())
    code = subprocess.run(
        [sys.executable, "-m", "pilot.launch", "train", "--spec", str(run_dir / "spec.json"), "--run-dir", str(run_dir), "--allow-dirty"],
        cwd=REPO,
    ).returncode
    result = json.loads((run_dir / "train_result.json").read_text())
    assert code == 0 and result["status"] == "completed" and result["torch_threads"] == 1
    agent_dir = tmp_path / "agent"
    script = (
        "import omnisafe, sys\n"
        "cfg={'seed':0,'train_cfgs':{'total_steps':40000,'device':'cpu','torch_threads':1,'vector_env_nums':1,'parallel':1},"
        "'logger_cfgs':{'save_model_freq':10,'log_dir':sys.argv[1]}}\n"
        "omnisafe.Agent('PPOLag','SafetyPointGoal1-v0',custom_cfgs=cfg).learn()\n"
    )
    subprocess.run([sys.executable, "-c", script, str(agent_dir)], check=True, cwd=tmp_path,
                   env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})

    def rows(base: Path) -> list[dict]:
        (path,) = list(base.glob("**/progress.csv"))
        with open(path, newline="") as fh:
            return [{k: v for k, v in r.items() if not k.startswith("Time/")} for r in csv.DictReader(fh)]

    ours_rows, their_rows = rows(run_dir / "omnisafe"), rows(agent_dir)
    assert len(ours_rows) == len(their_rows) == 2
    assert [{k: r[k] for k in t} for r, t in zip(ours_rows, their_rows)] == their_rows  # every OmniSafe column
    # batch metrics: epoch 0 has 20 episodes (< 50), so the batch mean equals OmniSafe's window mean;
    # in epoch 1 the window holds 40 episodes, so window = (batch0 + batch1) / 2
    b0, b1 = float(ours_rows[0]["Metrics/BatchEpCost"]), float(ours_rows[1]["Metrics/BatchEpCost"])
    assert b0 == pytest.approx(float(ours_rows[0]["Metrics/EpCost"]))
    assert (b0 + b1) / 2 == pytest.approx(float(ours_rows[1]["Metrics/EpCost"]), rel=1e-5)
    import torch

    saved = sorted(p.name for p in (run_dir / "omnisafe").glob("**/torch_save/*.pt"))
    assert saved == ["epoch-0.pt", "epoch-1.pt", "epoch-2.pt"]  # epoch-1.pt is the onset checkpoint
    assert sorted(p.name for p in agent_dir.glob("**/torch_save/*.pt")) == ["epoch-0.pt", "epoch-2.pt"]
    (ours,) = list((run_dir / "omnisafe").glob("**/torch_save/epoch-2.pt"))
    (theirs,) = list(agent_dir.glob("**/torch_save/epoch-2.pt"))
    a, b = torch.load(ours, weights_only=False), torch.load(theirs, weights_only=False)
    for key, tensor in b["pi"].items():
        assert torch.equal(a["pi"][key], tensor)
    # the full-state checkpoint adds the rest of the learner state and loads without pickled code
    full = torch.load(ours, weights_only=True)
    assert {"pi", "obs_normalizer", "reward_critic", "cost_critic", "actor_optimizer", "reward_critic_optimizer",
            "cost_critic_optimizer", "actor_scheduler", "lagrange", "lambda_optimizer", "episode_windows"} <= set(full)
    window = full["episode_windows"]["Metrics/EpCost"]  # 40 episodes after two epochs; J_C is its mean
    assert len(window) == 40 and float(window.mean()) == pytest.approx(float(ours_rows[1]["Metrics/EpCost"]), rel=1e-5)
    assert float(full["lagrange"]["value"]) == pytest.approx(float(their_rows[1]["Metrics/LagrangeMultiplier"]))


def test_multiplier_state_stores_plain_numbers(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    from collections import deque

    from pilot.algorithms import _MultiplierState

    class FakePID:
        def __init__(self) -> None:
            self.lagrangian_multiplier = 0.25
            self.cost_limit = np.float64(30.0)  # e.g. a ramp plug-in's np.interp result
            self._pid_i = 0.001
            self._cost_ds = deque([np.float64(1.5), 2.5])
            self._pid_d_delay = np.int64(10)  # numpy integers and float32 are not Python numbers
            self._ramp_limit = np.float32(27.5)
            self._onset_scale = torch.tensor(0.5)  # a one-element tensor
            self._diff_norm = np.bool_(False)  # flags are configuration, not numeric state
            self._sum_norm = True
            self._weights = torch.zeros(3)

    path = tmp_path / "epoch-1.pt"
    torch.save({"lagrange": _MultiplierState(FakePID()).state_dict()}, path)
    loaded = torch.load(path, weights_only=True)["lagrange"]  # no pickled numpy objects
    assert loaded["numeric_attributes"] == {"lagrangian_multiplier": 0.25, "cost_limit": 30.0, "_pid_i": 0.001, "_pid_d_delay": 10,
                                            "_ramp_limit": 27.5, "_onset_scale": 0.5}
    assert type(loaded["numeric_attributes"]["_pid_d_delay"]) is int
    assert loaded["deque_attributes"]["_cost_ds"] == [1.5, 2.5]


def test_the_thread_check_after_the_factory_does_not_reset_the_count() -> None:
    torch = pytest.importorskip("torch")
    from pilot.launch import RunRefused, _check_threads

    before = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        with pytest.raises(RunRefused, match="threads"):
            _check_threads(reset=False)
        assert _check_threads() == 1
    finally:
        torch.set_num_threads(before)


# ---------------------------------------------------------------------------
# Regression tests of round 6 (signals, classification, plug-in factories, contracts)
# ---------------------------------------------------------------------------


def _fake_run(tmp_path: Path, monkeypatch, learn, total_steps: int = R.STEPS_PER_EPOCH) -> tuple[RunSpec, Path]:
    """A run directory whose plug-in is a fake algorithm with the given ``learn(run_dir)``."""
    import pilot.algorithms as algorithms

    class FakeAlgo:
        def learn(self) -> None:
            learn(run_dir)

    spec = _det_spec(total_steps)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "spec.json").write_text(spec.to_json())
    monkeypatch.setattr(algorithms, "load_plugin", lambda key: (lambda task, cfgs, s: FakeAlgo()))
    return spec, run_dir


def _omnisafe_dir(run_dir: Path) -> Path:
    omni = run_dir / "omnisafe" / "PPOLag-{x}" / "seed-000-t"
    omni.mkdir(parents=True)
    return omni


@pytest.mark.omnisafe
def test_a_signal_while_the_result_is_written_is_deferred_even_with_another_thread(tmp_path, monkeypatch) -> None:
    """OmniSafe's TensorBoard writer thread outlives learn(): a SIGTERM delivered to it must neither
    cut train_result.json short nor raise in the launcher; it reaches the original handler afterwards."""
    import signal
    import threading
    import time

    torch = pytest.importorskip("torch")
    from pilot import launch

    stop = threading.Event()
    thread = threading.Thread(target=stop.wait, daemon=True)  # stands in for the TensorBoard writer

    def learn(run_dir: Path) -> None:
        thread.start()
        omni = _omnisafe_dir(run_dir)
        (omni / "progress.csv").write_text(f"TotalEnvSteps,Loss/Loss_pi\n{R.STEPS_PER_EPOCH},0.1\n")
        (omni / "torch_save").mkdir()
        torch.save({"pi": {"w": torch.zeros(1)}}, omni / "torch_save" / "epoch-1.pt")

    spec, run_dir = _fake_run(tmp_path, monkeypatch, learn)
    real_classify = launch.classify_training

    def classify_during_a_signal(*args, **kwargs):
        # Process-directed, like the scheduler's SIGTERM: the kernel may hand it to any thread that
        # does not block it (here the extra thread), and CPython runs the Python handler in this one.
        os.kill(os.getpid(), signal.SIGTERM)
        for _ in range(20):
            time.sleep(0.01)  # let the handler run while the result is being assembled
        return real_classify(*args, **kwargs)

    monkeypatch.setattr(launch, "classify_training", classify_during_a_signal)
    received: list[int] = []

    def original(signum, frame) -> None:
        received.append(signum)

    previous = signal.signal(signal.SIGTERM, original)
    try:
        code = launch.train(spec, run_dir, allow_dirty=True)
        assert signal.getsignal(signal.SIGTERM) is original  # the caller's handler is back
    finally:
        signal.signal(signal.SIGTERM, previous)
        stop.set()
    result = json.loads((run_dir / "train_result.json").read_text())
    assert code == launch.EXIT_COMPLETED and result["status"] == "completed"
    assert received == [signal.SIGTERM]  # re-delivered once, after the result was written


@pytest.mark.omnisafe
def test_an_interruption_does_not_hide_a_non_finite_loss_already_logged(tmp_path, monkeypatch) -> None:
    from pilot import launch

    def learn(run_dir: Path) -> None:
        omni = _omnisafe_dir(run_dir)
        step = R.STEPS_PER_EPOCH
        (omni / "progress.csv").write_text(
            f"TotalEnvSteps,Loss/Loss_reward_critic,Metrics/LagrangeMultiplier\n{step},0.1,0.0\n{2 * step},nan,0.0\n"
        )
        raise launch.RunInterrupted("signal 15")  # SIGTERM during the next epoch's rollout

    spec, run_dir = _fake_run(tmp_path, monkeypatch, learn, total_steps=4 * R.STEPS_PER_EPOCH)
    assert launch.train(spec, run_dir, allow_dirty=True) == launch.EXIT_FAILED
    result = json.loads((run_dir / "train_result.json").read_text())
    assert (result["status"], result["failure_cause"]) == ("failed", "non_finite_loss")
    assert "signal 15" in result["detail"]


@pytest.mark.omnisafe
def test_a_module_missing_in_the_factory_is_unavailable_not_a_crash(tmp_path, monkeypatch, capsys) -> None:
    import signal

    import pilot.algorithms as algorithms
    from pilot import launch

    spec, run_dir = _fake_run(tmp_path, monkeypatch, lambda run_dir: None)

    def factory(task, cfgs, s):
        (run_dir / "omnisafe" / "PPOLag-{x}").mkdir(parents=True)  # OmniSafe's logger may already exist
        raise ModuleNotFoundError("No module named 'envs.onset_helpers'", name="envs.onset_helpers")

    monkeypatch.setattr(algorithms, "load_plugin", lambda key: factory)
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    argv = ["train", "--spec", str(run_dir / "spec.json"), "--run-dir", str(run_dir), "--allow-dirty"]
    assert launch.main(argv) == launch.EXIT_UNAVAILABLE
    assert "envs.onset_helpers" in capsys.readouterr().err
    assert sorted(p.name for p in run_dir.iterdir()) == ["spec.json"]  # claim and output removed
    assert {sig: signal.getsignal(sig) for sig in handlers} == handlers


def test_plain_converts_torch_parameters() -> None:
    torch = pytest.importorskip("torch")
    from pilot.launch import plain

    assert json.dumps(plain({"x": torch.nn.Parameter(torch.tensor(1.5))})) == '{"x": 1.5}'


def test_malformed_evaluations_break_the_contract_not_the_launcher(tmp_path) -> None:
    from pilot.contracts import ContractError, validate_evaluation

    spec = _det_spec(R.TOTAL_STEPS)
    omni = tmp_path / "omni"
    (omni / "torch_save").mkdir(parents=True)
    for step in range(0, R.TOTAL_STEPS + 1, R.CHECKPOINT_INTERVAL_STEPS):
        (omni / "torch_save" / f"epoch-{step // R.STEPS_PER_EPOCH}.pt").write_bytes(b"")
    window = range(R.TOTAL_STEPS - 9 * R.CHECKPOINT_INTERVAL_STEPS, R.TOTAL_STEPS + 1, R.CHECKPOINT_INTERVAL_STEPS)
    good = {"final_cost": 20.0, "final_return": 5.0, "episodes": R.EVAL_EPISODES,
            "selection_seeds": list(range(R.EVAL_EPISODES)), "selection": {s: [24.0, 1.0] for s in window}}
    validate_evaluation(good, spec, omni)
    for bad in ({"episodes": "x"}, {"episodes": 100.5}, {"selection": [1, 2]},
                {"selection": {**good["selection"], R.TOTAL_STEPS: 3.0}},
                {"selection": {**{str(s): [24.0, 1.0] for s in window}, "last": [24.0, 1.0]}}):
        with pytest.raises(ContractError):
            validate_evaluation({**good, **bad}, spec, omni)


def _determinism_script():
    import importlib.util

    spec = importlib.util.spec_from_file_location("determinism_check", REPO / "scripts" / "determinism_check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_determinism_check_refuses_before_training(tmp_path, monkeypatch) -> None:
    import pilot.enrichment as enrichment
    from pilot.algorithms import PluginUnavailableError

    det = _determinism_script()

    def no_training(*args, **kwargs):
        raise AssertionError("trained before the refusal")

    monkeypatch.setattr(det, "train_once", no_training)
    # --keep into a directory that already holds a run: a usage error, not a FileExistsError traceback
    (tmp_path / "runA").mkdir()
    with pytest.raises(SystemExit) as exc:
        det.main(["--allow-dirty", "--keep", str(tmp_path)])
    assert exc.value.code == 2
    # the registered form without the evaluation harness stops before training anything

    def missing(target):
        raise PluginUnavailableError("envs.evaluation.evaluate_checkpoint is not in the repository yet")

    monkeypatch.setattr(enrichment, "_load", missing)
    with pytest.raises(SystemExit, match="UNAVAILABLE"):
        det.main(["--allow-dirty", "--registered-form"])


# ---------------------------------------------------------------------------
# Regression tests of round 7 (two signals at once, handler restoration, seed sets)
# ---------------------------------------------------------------------------


def _recording_handlers(received: list[int]) -> dict:
    """Install a recording handler for SIGTERM and SIGINT; return the handlers they replaced."""
    import signal

    def original(signum, frame) -> None:
        received.append(signum)

    return {sig: signal.signal(sig, original) for sig in (signal.SIGTERM, signal.SIGINT)}


@pytest.mark.omnisafe
def test_two_signals_at_once_are_one_interruption_and_one_redelivery(tmp_path, monkeypatch) -> None:
    """The second of two simultaneous signals must not escape train() (no result, handlers not restored)."""
    import signal

    from pilot import launch

    both = {signal.SIGTERM, signal.SIGINT}

    def learn(run_dir: Path) -> None:
        signal.pthread_sigmask(signal.SIG_BLOCK, both)
        os.kill(os.getpid(), signal.SIGTERM)
        os.kill(os.getpid(), signal.SIGINT)
        signal.pthread_sigmask(signal.SIG_UNBLOCK, both)  # both handlers become pending together
        for _ in range(1000):
            pass

    spec, run_dir = _fake_run(tmp_path, monkeypatch, learn)
    received: list[int] = []
    previous = _recording_handlers(received)
    try:
        ours = {sig: signal.getsignal(sig) for sig in both}
        code = launch.train(spec, run_dir, allow_dirty=True)
        assert {sig: signal.getsignal(sig) for sig in both} == ours  # the caller's handlers are back
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    result = json.loads((run_dir / "train_result.json").read_text())
    assert code == launch.EXIT_INTERRUPTED and result["status"] == "interrupted"
    assert len(received) == 1 and received[0] in both  # one ended training, the other is re-delivered
    assert f"signal {int(received[0])}" != result["interruption"]


@pytest.mark.omnisafe
def test_a_failed_result_write_restores_the_handlers_and_redelivers_the_signal(tmp_path, monkeypatch) -> None:
    import signal

    from pilot import launch

    spec, run_dir = _fake_run(tmp_path, monkeypatch, lambda run_dir: None)

    def full_disk(path, data) -> None:
        os.kill(os.getpid(), signal.SIGINT)  # Ctrl-C while the result is written: recorded only
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(launch, "_write_json_atomic", full_disk)
    received: list[int] = []
    previous = _recording_handlers(received)
    try:
        ours = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
        with pytest.raises(OSError, match="No space left"):
            launch.train(spec, run_dir, allow_dirty=True)
        assert {sig: signal.getsignal(sig) for sig in ours} == ours
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    assert received == [signal.SIGINT]


def test_seed_sets_must_be_whole_numbers() -> None:
    from pilot.contracts import ContractError, validate_seed_set

    assert validate_seed_set("s", [float(i) for i in range(R.EVAL_EPISODES)])[:2] == [0, 1]
    for bad in ([i + 0.5 for i in range(R.EVAL_EPISODES)], [True] + list(range(1, R.EVAL_EPISODES)),
                [float("inf")] + list(range(1, R.EVAL_EPISODES)), None):
        with pytest.raises(ContractError):
            validate_seed_set("s", bad)


# ---------------------------------------------------------------------------
# Regression tests of round 8 (a signal before training is guarded; 64-bit seeds)
# ---------------------------------------------------------------------------


@pytest.mark.omnisafe
def test_a_signal_between_the_handlers_and_training_is_an_interruption(tmp_path, monkeypatch) -> None:
    """A signal after the gate is installed but before _train_claimed's guarded block must not escape train()."""
    import signal
    import time

    from pilot import launch

    def learn(run_dir: Path) -> None:
        raise AssertionError("trained after the interruption")

    class SignalAtFirstClock:  # launch.time: the first monotonic() is _train_claimed's t0, before its try
        calls = 0

        def monotonic(self) -> float:
            SignalAtFirstClock.calls += 1
            if SignalAtFirstClock.calls == 1:
                os.kill(os.getpid(), signal.SIGTERM)
                for _ in range(1000):
                    pass
            return time.monotonic()

    spec, run_dir = _fake_run(tmp_path, monkeypatch, learn)
    monkeypatch.setattr(launch, "time", SignalAtFirstClock())
    received: list[int] = []
    previous = _recording_handlers(received)
    try:
        ours = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
        code = launch.train(spec, run_dir, allow_dirty=True)
        assert {sig: signal.getsignal(sig) for sig in ours} == ours  # the caller's handlers are back
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    result = json.loads((run_dir / "train_result.json").read_text())
    assert code == launch.EXIT_INTERRUPTED
    assert (result["status"], result["failure_cause"], result["omnisafe_dir"]) == ("interrupted", None, None)
    assert result["interruption"] == f"signal {int(signal.SIGTERM)}"
    assert received == []  # the one signal ended the run; nothing was recorded for re-delivery


def test_seed_sets_keep_64_bit_seeds_and_reject_numpy_bools_and_strings() -> None:
    np = pytest.importorskip("numpy")
    from pilot.contracts import ContractError, validate_seed_set

    big = [2**53 + i for i in range(1, R.EVAL_EPISODES + 1)]  # not exactly representable as floats
    assert validate_seed_set("s", big) == big
    assert validate_seed_set("s", [np.uint64(x) for x in big]) == big
    assert validate_seed_set("s", [np.int64(i) for i in range(R.EVAL_EPISODES)])[:2] == [0, 1]
    for bad in ([np.True_] + list(range(2, R.EVAL_EPISODES + 1)), [str(i) for i in range(R.EVAL_EPISODES)],
                [float("nan")] + list(range(1, R.EVAL_EPISODES))):
        with pytest.raises(ContractError):
            validate_seed_set("s", bad)

"""Offline recomputation of plasticity rows (Role 3): ``python -m metrics.recompute``.

The end-to-end equality with the rows logged during training is in ``tests/test_metrics_hook.py``
(slow). Here: locating the run, comparing, refusing to write inside a run, and recomputation from
synthetic checkpoints (plain and injected actors, full-state and OmniSafe-only files).
"""

from __future__ import annotations

import io
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from metrics import recompute as RC
from metrics.plasticity import COLUMNS, NetworkParts, measure, nan_row

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _restore_threads():
    threads = torch.get_num_threads()  # RC.main sets one thread for the process, as training does
    yield
    torch.set_num_threads(threads)


def test_find_omnisafe_dir_in_every_layout(tmp_path) -> None:
    omni = tmp_path / "run" / "omnisafe" / "PPOLag-{X}" / "seed-000-2026"
    (omni / "torch_save").mkdir(parents=True)
    assert RC.find_omnisafe_dir(omni) == omni
    assert RC.find_omnisafe_dir(tmp_path / "run") == omni  # the launcher's layout
    record = {"omnisafe_dir": "omnisafe/PPOLag-{X}/seed-000-2026"}
    (tmp_path / "run" / "train_result.json").write_text(json.dumps(record))
    assert RC.find_omnisafe_dir(tmp_path / "run") == omni
    (tmp_path / "other" / "omnisafe").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        RC.find_omnisafe_dir(tmp_path / "other")


def test_checkpoint_files_in_epoch_order(tmp_path) -> None:
    (tmp_path / "torch_save").mkdir()
    for name in ("epoch-10.pt", "epoch-2.pt", "epoch-0.pt", "epoch-x.pt", "notes.txt"):
        (tmp_path / "torch_save" / name).write_bytes(b"")
    assert [k for k, _ in RC.checkpoint_files(tmp_path)] == [0, 2, 10]


def _row(step: int, value: float) -> dict:
    return {c: (step if c == "step" else value) for c in COLUMNS}


def test_compare_is_exact_and_names_every_difference() -> None:
    a = {0: _row(0, 0.5), 2000: nan_row(2000)}
    assert RC.compare(a, {0: _row(0, 0.5), 2000: nan_row(2000)}) == []  # NaN equals NaN here
    changed = _row(0, 0.5)
    changed["rank"] = math.nextafter(0.5, 1.0)
    problems = RC.compare(a, {0: changed, 4000: _row(4000, 1.0)})
    assert any(p.startswith("step 0: rank") for p in problems)
    assert "step 2000: checkpoint without a logged row" in problems
    assert "step 4000: logged row without a checkpoint" in problems


def test_read_logged_rows_drops_a_last_line_cut_short(tmp_path) -> None:
    text = RC.rows_csv({0: _row(0, 0.25), 2000: _row(2000, 0.25)})
    for cut in (text[:-2], text[:-20]):  # inside the last value ("0.2" of "0.25"); whole fields lost
        (tmp_path / "plasticity.csv").write_text(cut, encoding="utf-8")
        with pytest.warns(RuntimeWarning, match="no line end"):
            rows = RC.read_logged_rows(tmp_path)
        assert sorted(rows) == [0] and rows[0]["dormant"] == 0.25


def test_an_empty_or_header_cut_file_holds_no_logged_row(tmp_path) -> None:
    """A failed run's 0-byte file, or one whose header was cut, is no row (as pilot.contracts.validate_plasticity
    reads it): every checkpoint is a row the log lacks (EXIT_DIFFERENT), never EXIT_ERROR."""
    (tmp_path / "plasticity.csv").write_text("", encoding="utf-8")
    assert RC.read_logged_rows(tmp_path) == {}
    (tmp_path / "plasticity.csv").write_text("step,dormant,ra", encoding="utf-8")
    with pytest.warns(RuntimeWarning, match="no line end"):
        assert RC.read_logged_rows(tmp_path) == {}
    assert RC.compare({0: _row(0, 0.25)}, {}) == ["step 0: checkpoint without a logged row"]


def test_read_logged_rows_refuses_duplicates_and_missing_columns(tmp_path) -> None:
    assert RC.read_logged_rows(tmp_path) is None
    (tmp_path / "plasticity.csv").write_text(RC.rows_csv({0: _row(0, 0.25)}), encoding="utf-8")
    assert RC.read_logged_rows(tmp_path)[0]["dormant"] == 0.25
    text = RC.rows_csv({0: _row(0, 0.25)}) + RC.rows_csv({0: _row(0, 0.5)}).split("\n", 1)[1]
    (tmp_path / "plasticity.csv").write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="twice"):
        RC.read_logged_rows(tmp_path)
    (tmp_path / "plasticity.csv").write_text("step,dormant,rank,norm\n0,0,1,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="lacks"):
        RC.read_logged_rows(tmp_path)


def test_main_refuses_to_write_inside_the_run(tmp_path) -> None:
    (tmp_path / "torch_save").mkdir()
    out = io.StringIO()
    assert RC.main(["--run-dir", str(tmp_path), "--out", str(tmp_path / "rows.csv")], out=out) == RC.EXIT_REFUSED
    assert "inside the run directory" in out.getvalue() and not (tmp_path / "rows.csv").exists()
    existing = tmp_path.parent / f"{tmp_path.name}-existing.csv"
    existing.write_text("x", encoding="utf-8")
    out = io.StringIO()
    assert RC.main(["--run-dir", str(tmp_path), "--out", str(existing)], out=out) == RC.EXIT_REFUSED
    assert existing.read_text(encoding="utf-8") == "x"


def test_the_out_file_is_written_atomically_and_readable(tmp_path) -> None:
    """``--out`` is an ordinary file (mode 0644, as the package's other writers), in a directory created if needed."""
    path = tmp_path / "audit" / "rows.csv"
    text = RC.rows_csv({0: _row(0, 0.25)})
    RC._write_out(path, text)
    assert path.read_text(encoding="utf-8") == text and (path.stat().st_mode & 0o777) == 0o644
    assert sorted(p.name for p in path.parent.iterdir()) == ["rows.csv"]  # no temporary file left


def test_exit_codes_are_distinct_and_usage_is_argparses() -> None:
    codes = [RC.EXIT_OK, RC.EXIT_DIFFERENT, RC.EXIT_USAGE, RC.EXIT_REFUSED, RC.EXIT_ERROR]
    assert len(set(codes)) == len(codes) and RC.EXIT_USAGE == 2
    with pytest.raises(SystemExit) as exc:
        RC.main([], out=io.StringIO(), err=io.StringIO())
    assert exc.value.code == RC.EXIT_USAGE


def test_a_failure_to_recompute_is_an_error_not_a_difference(tmp_path) -> None:
    """No config.json, no OmniSafe directory, a checkpoint that does not load: EXIT_ERROR, never EXIT_DIFFERENT."""
    (tmp_path / "noconfig" / "torch_save").mkdir(parents=True)
    (tmp_path / "nothing").mkdir()
    broken = tmp_path / "broken"
    (broken / "torch_save").mkdir(parents=True)
    (broken / "config.json").write_text(json.dumps({"env_id": "SafetyPointGoal1-v0",
                                                    "algo_cfgs": {"steps_per_epoch": 2000, "obs_normalize": True}}))
    (broken / "torch_save" / "epoch-0.pt").write_bytes(b"not a checkpoint")
    for run, what in (("noconfig", "config.json"), ("nothing", "expected one OmniSafe run directory"),
                      ("broken", "epoch-0.pt")):
        out, err = io.StringIO(), io.StringIO()
        assert RC.main(["--run-dir", str(tmp_path / run)], out=out, err=err) == RC.EXIT_ERROR, run
        assert "error: could not recompute" in err.getvalue() and what in err.getvalue(), err.getvalue()
        assert not out.getvalue()


def test_the_command_exits_with_the_error_code(tmp_path) -> None:
    """Run as ``python -m metrics.recompute``: a failure exits EXIT_ERROR (4), never EXIT_DIFFERENT (1), and prints nothing to stdout."""
    (tmp_path / "torch_save").mkdir()
    result = subprocess.run([sys.executable, "-m", "metrics.recompute", "--run-dir", str(tmp_path)], cwd=REPO,
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == RC.EXIT_ERROR, result.stdout + result.stderr
    assert "FileNotFoundError" in result.stderr and not result.stdout


@pytest.mark.omnisafe
def test_recompute_rows_from_synthetic_checkpoints(tmp_path) -> None:
    """Plain and injected actors, a full-state and an OmniSafe-only checkpoint, the saved normaliser."""
    from gymnasium import spaces
    from omnisafe.common.normalizer import Normalizer
    from omnisafe.models.actor.actor_builder import ActorBuilder
    from omnisafe.models.critic.critic_builder import CriticBuilder
    from omnisafe.utils.config import get_default_kwargs_yaml

    from metrics.batches import load_fixed_batch
    from metrics.interventions import intervention_generator, plasticity_injection

    task = "SafetyPointGoal1-v0"
    cfgs = get_default_kwargs_yaml("PPOLag", task, "on-policy")
    config = {"env_id": task, "algo_cfgs": {"steps_per_epoch": 3000, "obs_normalize": True},
              "model_cfgs": cfgs.model_cfgs.todict()}
    omni = tmp_path / "omni"
    (omni / "torch_save").mkdir(parents=True)
    (omni / "config.json").write_text(json.dumps(config), encoding="utf-8")
    torch.manual_seed(0)
    obs, act = spaces.Box(-np.inf, np.inf, (60,), np.float32), spaces.Box(-1.0, 1.0, (2,), np.float32)
    actor_sizes, critic_sizes = list(cfgs.model_cfgs.actor.hidden_sizes), list(cfgs.model_cfgs.critic.hidden_sizes)
    actor = ActorBuilder(obs, act, actor_sizes, activation="tanh").build_actor("gaussian_learning")
    critics = [CriticBuilder(obs, act, critic_sizes, activation="tanh").build_critic("v") for _ in range(2)]
    norm = Normalizer((60,), clip=5)
    batch = np.array(load_fixed_batch(task))
    norm.normalize(torch.from_numpy(batch[:500]))
    full = {"pi": actor.state_dict(), "obs_normalizer": norm.state_dict(), "reward_critic": critics[0].state_dict(),
            "cost_critic": critics[1].state_dict()}
    torch.save(full, omni / "torch_save" / "epoch-0.pt")
    expected = {0: measure(NetworkParts(actor, critics[0], critics[1], norm), batch, step=0)}
    optimizer = torch.optim.Adam(actor.parameters())
    plasticity_injection(actor, optimizer, intervention_generator(0))
    with torch.no_grad():
        actor.mean.new[0].weight.mul_(1.5)  # the heads have diverged since injection
    torch.save({"pi": actor.state_dict(), "obs_normalizer": norm.state_dict()}, omni / "torch_save" / "epoch-2.pt")
    expected[6000] = measure(NetworkParts(actor, None, None, norm), batch, step=6000)
    rng = torch.get_rng_state()
    rows = RC.recompute_rows(omni)
    assert torch.equal(rng, torch.get_rng_state())
    assert sorted(rows) == [0, 6000] and RC.compare(rows, expected) == []
    assert math.isnan(rows[6000]["norm_reward_critic"]) and rows[6000]["norm_all"] > rows[6000]["norm"]
    (omni / "plasticity.csv").write_text(RC.rows_csv(expected), encoding="utf-8")
    out = io.StringIO()
    assert RC.main(["--run-dir", str(omni)], out=out) == RC.EXIT_OK, out.getvalue()
    expected[0]["dormant"] = 0.5
    (omni / "plasticity.csv").write_text(RC.rows_csv(expected), encoding="utf-8")
    out = io.StringIO()
    assert RC.main(["--run-dir", str(omni)], out=out) == RC.EXIT_DIFFERENT and "step 0: dormant" in out.getvalue()

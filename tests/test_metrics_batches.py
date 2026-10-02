"""Fixed evaluation batches (Role 3): the committed files, their records, the loader and the collection script.

Table 2.3 "Fixed evaluation batch": "2,048 states (decision) collected once per task by a
uniform-random policy under seed 0, stored in the repository". The re-collection tests need the
pinned Safety-Gymnasium stack (``omnisafe`` marker); the full ``--check`` of all three tasks is
``slow`` (about 25 s).
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from configs import registered as R
from metrics import batches, collect_fixed_batch as cli
from pilot.errors import RunRefused

REPO = Path(__file__).resolve().parents[1]
# Observation sizes measured on the pinned stack (Q-transfer-obs in configs/registered.py states 60, 72 and 76 too).
OBS_DIMS = {"SafetyPointGoal1-v0": 60, "SafetyCarGoal1-v0": 72, "SafetyPointButton1-v0": 76}


def _manifest() -> dict:
    return json.loads((batches.FIXED_BATCH_DIR / batches.MANIFEST_FILE).read_text(encoding="utf-8"))


def test_three_records_agree_for_every_study_a_task() -> None:
    manifest = _manifest()
    assert set(batches.FIXED_BATCH_SHA256) == set(R.TASKS_STUDY_A) == set(manifest["tasks"])
    for task in R.TASKS_STUDY_A:
        digest = hashlib.sha256(batches.batch_path(task).read_bytes()).hexdigest()
        assert batches.FIXED_BATCH_SHA256[task] == manifest["tasks"][task]["sha256"] == digest


def test_manifest_contents() -> None:
    manifest = _manifest()
    assert manifest["procedure"] == batches.PROCEDURE
    assert manifest["script"] == "metrics/collect_fixed_batch.py"
    assert set(manifest["versions"]) == {"python", "numpy", "gymnasium", "safety-gymnasium", "mujoco"}
    for task in R.TASKS_STUDY_A:
        entry = manifest["tasks"][task]
        assert entry == {
            "file": f"{task}.npy",
            "sha256": batches.FIXED_BATCH_SHA256[task],
            "shape": [R.FIXED_BATCH_STATES, OBS_DIMS[task]],
            "dtype": "float32",
            "n_states": R.FIXED_BATCH_STATES,
            "seed": R.FIXED_BATCH_SEED,
            "episode_starts": [0, 1000, 2000],
        }
    raw = (batches.FIXED_BATCH_DIR / batches.MANIFEST_FILE).read_bytes()
    assert raw == cli.manifest_bytes(manifest)  # sorted keys, LF, trailing newline: byte-stable


@pytest.mark.parametrize("task", R.TASKS_STUDY_A)
def test_load_fixed_batch(task: str) -> None:
    batch = batches.load_fixed_batch(task)
    assert batch.shape == (R.FIXED_BATCH_STATES, OBS_DIMS[task])
    assert batch.dtype == np.float32 and np.isfinite(batch).all()
    assert not batch.flags.writeable
    assert np.array_equal(batch, batches.load_fixed_batch(task))
    assert batches.npy_bytes(np.array(batch)) == batches.batch_path(task).read_bytes()


def test_loader_refuses_changed_missing_and_unknown(tmp_path, monkeypatch) -> None:
    task = "SafetyPointGoal1-v0"
    for t in R.TASKS_STUDY_A:
        shutil.copy(batches.batch_path(t), tmp_path / f"{t}.npy")
    monkeypatch.setattr(batches, "FIXED_BATCH_DIR", tmp_path)
    assert batches.load_fixed_batch(task).shape[1] == 60  # the copy loads
    data = bytearray((tmp_path / f"{task}.npy").read_bytes())
    data[-1] ^= 0x01  # one bit of the last float
    (tmp_path / f"{task}.npy").write_bytes(bytes(data))
    with pytest.raises(batches.FixedBatchError, match="differs from the committed record"):
        batches.load_fixed_batch(task)
    (tmp_path / f"{task}.npy").unlink()
    with pytest.raises(batches.FixedBatchError, match="missing"):
        batches.load_fixed_batch(task)
    (tmp_path / f"{task}.npy").mkdir()  # any other read failure (here IsADirectoryError) is a refusal too
    with pytest.raises(batches.FixedBatchError, match="cannot be read"):
        batches.load_fixed_batch(task)
    with pytest.raises(batches.FixedBatchError, match="no fixed batch recorded"):
        batches.load_fixed_batch("SafetyCarButton1-v0")
    with pytest.raises(batches.FixedBatchError):
        batches.batch_path("../../etc/passwd")
    assert issubclass(batches.FixedBatchError, RunRefused)  # a refusal (exit 5), never a crash


@pytest.mark.parametrize("bad", ["shape", "dtype", "nan"])
def test_loader_checks_shape_dtype_and_values_even_with_a_matching_hash(tmp_path, monkeypatch, bad) -> None:
    task = "SafetyPointGoal1-v0"
    array = np.zeros((R.FIXED_BATCH_STATES, 60), dtype=np.float32)
    if bad == "shape":
        array = array[:-1]
    elif bad == "dtype":
        array = array.astype(np.float64)
    else:
        array[5, 5] = np.nan
    data = batches.npy_bytes(array)
    (tmp_path / f"{task}.npy").write_bytes(data)
    monkeypatch.setattr(batches, "FIXED_BATCH_DIR", tmp_path)
    monkeypatch.setattr(batches, "FIXED_BATCH_SHA256", {task: batches.sha256_hex(data)})
    with pytest.raises(batches.FixedBatchError):
        batches.load_fixed_batch(task)


def test_collect_fixed_batch_argument_checks() -> None:
    with pytest.raises(ValueError):
        batches.collect_fixed_batch("SafetyPointGoal1-v0", n_states=0)
    with pytest.raises(ValueError):
        batches.collect_fixed_batch("SafetyPointGoal1-v0", seed=-1)
    with pytest.raises(batches.FixedBatchError):
        batches.collect_fixed_batch("SafetyCarButton1-v0", n_states=4)


# ---------------------------------------------------------------------------------------------
# The script's logic, with the collection replaced by the committed arrays (fast)
# ---------------------------------------------------------------------------------------------


VERSIONS = {"python": "3.10", "numpy": "1", "gymnasium": "0", "safety-gymnasium": "0.4.1", "mujoco": "2.3.0"}


@pytest.fixture()
def fake_collection(monkeypatch):
    committed = {t: (np.array(batches.load_fixed_batch(t)), [0, 1000, 2000]) for t in R.TASKS_STUDY_A}
    monkeypatch.setattr(batches, "collect_fixed_batch",
                        lambda task: (committed[task][0].copy(), list(committed[task][1])))
    monkeypatch.setattr(cli, "_versions", lambda: dict(VERSIONS))
    return committed


def test_collect_writes_once_and_check_compares_bytes(tmp_path, fake_collection) -> None:
    out = io.StringIO()
    assert cli.collect(list(R.TASKS_STUDY_A), directory=tmp_path, out=out) == cli.EXIT_OK
    for task in R.TASKS_STUDY_A:
        assert (tmp_path / f"{task}.npy").read_bytes() == batches.batch_path(task).read_bytes()
        assert f'"{task}": "{batches.FIXED_BATCH_SHA256[task]}",' in out.getvalue()
    manifest = cli.read_manifest(tmp_path)
    assert manifest["tasks"] == _manifest()["tasks"] and manifest["versions"] == VERSIONS
    # collected once: a second collection is refused and changes nothing
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    out = io.StringIO()
    assert cli.collect(["SafetyPointGoal1-v0"], directory=tmp_path, out=out) == cli.EXIT_REFUSED
    assert "refused" in out.getvalue() and before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    # --check: identical bytes and records pass
    out = io.StringIO()
    assert cli.check(list(R.TASKS_STUDY_A), directory=tmp_path, out=out) == cli.EXIT_OK, out.getvalue()
    # a changed byte fails
    path = tmp_path / "SafetyCarGoal1-v0.npy"
    data = bytearray(path.read_bytes())
    data[200] ^= 0x10
    path.write_bytes(bytes(data))
    out = io.StringIO()
    assert cli.check(list(R.TASKS_STUDY_A), directory=tmp_path, out=out) == cli.EXIT_MISMATCH
    assert "SafetyCarGoal1-v0: re-collection byte-identical: False" in out.getvalue()


def test_check_fails_when_a_record_disagrees(tmp_path, fake_collection) -> None:
    assert cli.collect(["SafetyPointGoal1-v0"], directory=tmp_path, out=io.StringIO()) == cli.EXIT_OK
    out = io.StringIO()
    wrong = {"SafetyPointGoal1-v0": "0" * 64}
    assert cli.check(["SafetyPointGoal1-v0"], directory=tmp_path, out=out, recorded=wrong) == cli.EXIT_MISMATCH
    assert "FIXED_BATCH_SHA256: False" in out.getvalue()
    out = io.StringIO()
    assert cli.check(["SafetyPointButton1-v0"], directory=tmp_path, out=out) == cli.EXIT_MISMATCH  # missing file


def test_check_writes_its_result_as_json_for_the_workstation_record(tmp_path, fake_collection, monkeypatch) -> None:
    """``--check --json PATH`` (scripts/workstation.py, Table 9.1 Q-appendix-a): per task the committed and the
    re-collected sha256, whether the committed records agree and whether the re-collection reproduced the batch; a
    mismatch of the re-collection is a result (EXIT_MISMATCH with its JSON), the committed files are never touched."""
    before = {t: batches.batch_path(t).read_bytes() for t in R.TASKS_STUDY_A}
    path = tmp_path / "check.json"
    assert cli.main(["--all", "--check", "--json", str(path)], out=io.StringIO()) == cli.EXIT_OK
    result = json.loads(path.read_text(encoding="utf-8"))
    assert (result["exit_code"], result["passed"], result["problems"]) == (cli.EXIT_OK, True, [])
    for task in R.TASKS_STUDY_A:
        entry = result["tasks"][task]
        assert entry["committed_sha256"] == entry["recollected_sha256"] == batches.FIXED_BATCH_SHA256[task]
        assert entry["committed_consistent"] is True and entry["reproduced"] is True
    changed = np.array(fake_collection["SafetyCarGoal1-v0"][0])
    changed[0, 0] += 1.0  # this machine re-collects another first state
    monkeypatch.setattr(batches, "collect_fixed_batch", lambda task: (
        (changed if task == "SafetyCarGoal1-v0" else fake_collection[task][0]).copy(), list(fake_collection[task][1])))
    assert cli.main(["--all", "--check", "--json", str(path)], out=io.StringIO()) == cli.EXIT_MISMATCH
    result = json.loads(path.read_text(encoding="utf-8"))
    assert (result["exit_code"], result["passed"]) == (cli.EXIT_MISMATCH, False)
    car = result["tasks"]["SafetyCarGoal1-v0"]
    assert car["committed_sha256"] == batches.FIXED_BATCH_SHA256["SafetyCarGoal1-v0"] != car["recollected_sha256"]
    assert car["committed_consistent"] is True and car["reproduced"] is False
    assert car["checks"]["re-collection byte-identical"] is False
    assert all(result["tasks"][t]["reproduced"] for t in R.TASKS_STUDY_A if t != "SafetyCarGoal1-v0")
    assert {t: batches.batch_path(t).read_bytes() for t in R.TASKS_STUDY_A} == before  # never rewritten
    with pytest.raises(SystemExit) as exc:
        cli.main(["--all", "--json", str(path)])  # --json belongs to --check
    assert exc.value.code == cli.EXIT_USAGE


def test_a_check_that_cannot_run_writes_no_json(tmp_path, fake_collection, monkeypatch) -> None:
    def broken(task):
        raise RuntimeError("synthetic environment failure")

    monkeypatch.setattr(batches, "collect_fixed_batch", broken)
    path = tmp_path / "check.json"
    assert cli.main(["--all", "--check", "--json", str(path)], out=io.StringIO(), err=io.StringIO()) == cli.EXIT_ERROR
    assert not path.exists()


def test_collect_refuses_a_manifest_of_another_environment(tmp_path, fake_collection, monkeypatch) -> None:
    assert cli.collect(["SafetyPointGoal1-v0"], directory=tmp_path, out=io.StringIO()) == cli.EXIT_OK
    monkeypatch.setattr(cli, "_versions", lambda: dict(VERSIONS, mujoco="3.0.0"))
    out = io.StringIO()
    assert cli.collect(["SafetyCarGoal1-v0"], directory=tmp_path, out=out) == cli.EXIT_REFUSED
    assert not (tmp_path / "SafetyCarGoal1-v0.npy").exists()


def test_cli_arguments() -> None:
    for argv in ([], ["--task", "SafetyPointGoal1-v0", "--all"], ["--task", "SafetyCarButton1-v0"]):
        with pytest.raises(SystemExit) as exc:
            cli.main(argv)
        assert exc.value.code == cli.EXIT_USAGE  # argparse's own code, used by no other outcome
    codes = [cli.EXIT_OK, cli.EXIT_MISMATCH, cli.EXIT_USAGE, cli.EXIT_REFUSED, cli.EXIT_ERROR]
    assert len(set(codes)) == len(codes) and cli.EXIT_USAGE == 2


def test_cli_reports_a_failure_to_run_as_an_error_not_a_result(tmp_path, fake_collection, monkeypatch) -> None:
    """A crash of the collection or the check exits EXIT_ERROR, never EXIT_MISMATCH (or EXIT_OK)."""
    monkeypatch.setattr(batches, "FIXED_BATCH_DIR", tmp_path)
    assert cli.main(["--task", "SafetyPointGoal1-v0"], out=io.StringIO(), err=io.StringIO()) == cli.EXIT_OK
    (tmp_path / "SafetyCarGoal1-v0.npy").mkdir()  # a committed file that cannot be read
    err = io.StringIO()
    assert cli.main(["--all", "--check"], out=io.StringIO(), err=err) == cli.EXIT_ERROR
    assert "error: could not check" in err.getvalue() and "IsADirectoryError" in err.getvalue()

    def broken(task):
        raise RuntimeError("synthetic environment failure")

    monkeypatch.setattr(batches, "collect_fixed_batch", broken)
    err = io.StringIO()
    assert cli.main(["--task", "SafetyPointButton1-v0"], out=io.StringIO(), err=err) == cli.EXIT_ERROR
    assert "synthetic environment failure" in err.getvalue()
    assert not (tmp_path / "SafetyPointButton1-v0.npy").exists()
    assert "SafetyPointButton1-v0" not in cli.read_manifest(tmp_path)["tasks"]  # nothing recorded


def test_npy_files_are_binary_and_not_ignored_by_git() -> None:
    assert "*.npy binary" in (REPO / ".gitattributes").read_text(encoding="utf-8").splitlines()
    files = [str(batches.batch_path(t).relative_to(REPO)) for t in R.TASKS_STUDY_A]
    files.append(str((batches.FIXED_BATCH_DIR / batches.MANIFEST_FILE).relative_to(REPO)))
    ignored = subprocess.run(["git", "check-ignore", *files], cwd=REPO, capture_output=True, text=True)
    assert ignored.returncode == 1 and not ignored.stdout, ignored.stdout  # 1: none of them is ignored
    attrs = subprocess.run(["git", "check-attr", "binary", files[0]], cwd=REPO, capture_output=True, text=True)
    assert attrs.stdout.strip().endswith("binary: set")


# ---------------------------------------------------------------------------------------------
# Re-collection in the pinned environment
# ---------------------------------------------------------------------------------------------


@pytest.mark.omnisafe
@pytest.mark.parametrize("task", R.TASKS_STUDY_A)
def test_recollection_reproduces_the_committed_prefix(task: str) -> None:
    """The procedure is prefix-consistent: 64 states equal the first 64 rows of the committed batch."""
    batch, starts = batches.collect_fixed_batch(task, n_states=64)
    assert batch.dtype == np.float32 and starts == [0]
    assert np.array_equal(batch, batches.load_fixed_batch(task)[:64])


@pytest.mark.omnisafe
@pytest.mark.slow
def test_check_mode_reproduces_every_committed_batch() -> None:
    result = subprocess.run([sys.executable, "-m", "metrics.collect_fixed_batch", "--all", "--check"], cwd=REPO,
                            capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "check passed" in result.stdout

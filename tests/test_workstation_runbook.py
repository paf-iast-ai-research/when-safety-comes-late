"""scripts/workstation.py: the workstation runbook (HANDOVER.md section 4; one command per remaining step).

Fast: nothing is trained, installed or committed. HEAD's facts are a ``FakeHead`` (for "not a Git
repository", the real ``Head`` on an empty directory), the working tree is a temporary directory
(``ws.REPO``), the data root's scheduler state is a real SQLite file made with the scheduler's own
schema, and every called command is recorded by ``Recorder`` instead of run (the test sets its exit
status and its side effects). Every command line the runbook builds is parsed by the called tool's own
parser (``python -m pilot``, ``python -m analysis``, ``python -m metrics.collect_fixed_batch``,
``python -m studyb.search``, scripts/determinism_check.py), so a flag that does not exist fails here.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shlex
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import pytest

from configs import registered as R
from pilot import manifest, provenance
from pilot import scheduler as S

REPO = Path(__file__).resolve().parents[1]
COMMIT = "c" * 40
REGISTERED = "735b394d18d1bb046c7a74f900af00a83f1fa316"
MANIFEST = {"tasks": {"SafetyPointGoal1-v0": {"sha256": "a" * 64}, "SafetyCarGoal1-v0": {"sha256": "b" * 64}}}
HASHES = {"SafetyPointGoal1-v0": "a" * 64, "SafetyCarGoal1-v0": "b" * 64}
# The stack the fixed-batch step finds installed (``ws.package_versions`` is patched) and the lock that pins it
VERSIONS = {"omnisafe": "0.5.0", "safety-gymnasium": "0.4.1", "mujoco": "2.3.0", "gymnasium": "0.28.1",
            "numpy": "1.26.4", "torch": "2.14.0"}
LOCK_TEXT = "".join(f"{name.replace('-', '_')}=={version}\n" for name, version in VERSIONS.items()).encode()
# The two registration files at the registration commit, and as Tables 9.0 and 9.1 fill them (task 3)
REGISTERED_FILES = {"prereg/Preregistration.docx": b"docx 735b394", "prereg/Preregistration.pdf": b"pdf 735b394"}
FILLED_FILES = {"prereg/Preregistration.docx": b"docx, Table 9.0 filled", "prereg/Preregistration.pdf": b"pdf, filled"}
PASSED_ANALYSIS = {"pilot": {"mismatches": [], "compared": 4}}  # what python -m analysis --mode pilot writes (exit 0)


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve the module's string annotations through sys.modules
    spec.loader.exec_module(module)
    return module


ws = _load(REPO / "scripts" / "workstation.py", "workstation_runbook")
STEP_NAMES = [step.name for step in ws.STEPS]


class FakeHead(ws.Head):
    """HEAD's facts without git: committed files, the registration tag, determinism reports, the search record."""

    def __init__(self, root: Path, files: dict[str, Any] | None = None, tag: str | None = None,
                 determinism: tuple[str, ...] = (), search: tuple[bool, str] = (False, "the record is not complete"),
                 annotated: bool = True):
        self.root, self.commit, self.problem = Path(root), COMMIT, ""
        self._files = {rel: v if isinstance(v, bytes) else json.dumps(v).encode() for rel, v in (files or {}).items()}
        self._tag, self._annotated = tag, annotated
        self._determinism = {p: f"ledger/determinism/determinism-{p}-20261001T000000Z.json" for p in determinism}
        self._search = search

    def file(self, rel: str) -> bytes | None:
        return self._files.get(rel)

    def file_at(self, commit: str, rel: str) -> bytes | None:
        return REGISTERED_FILES.get(rel) if commit == REGISTERED else None

    def files(self, directory: str) -> list[str]:
        prefix = directory.rstrip("/") + "/"
        return sorted(p for p in self._files if p.startswith(prefix))

    def tag(self, name: str) -> str | None:
        return self._tag if name == ws.TAG else None

    def tag_annotated(self, name: str) -> bool:
        return self.tag(name) is not None and self._annotated

    def determinism_plugins(self) -> dict[str, str]:
        return dict(self._determinism)

    def search_record(self) -> tuple[bool, str]:
        return self._search


class Recorder:
    """Stands in for ``ws.call``: records each command; the exit status and side effects are the test's."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.codes: dict[str, int] = {}
        self.effects: dict[str, Callable[[], None]] = {}

    def __call__(self, cmd: list[str]) -> int:
        cmd = [str(c) for c in cmd]
        self.calls.append(cmd)
        line = self.line(cmd)
        for needle, effect in self.effects.items():
            if needle in line:
                effect()
        return next((code for needle, code in self.codes.items() if needle in line), 0)

    @staticmethod
    def line(cmd: list[str]) -> str:
        return " ".join("python" if c == sys.executable else c for c in cmd)

    @property
    def lines(self) -> list[str]:
        return [self.line(c) for c in self.calls]


@pytest.fixture
def tree(tmp_path, monkeypatch) -> tuple[Path, Recorder]:
    """A working tree (``ws.REPO``) in a temporary directory; every called command recorded, never run."""
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(ws, "REPO", root)
    recorder = Recorder()
    monkeypatch.setattr(ws, "call", recorder)
    return root, recorder


def write(root: Path, rel: str, value: Any) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())
    return path


def default_report(**changes: Any) -> dict[str, Any]:
    """A report as ``python scripts/determinism_check.py`` writes it when it passed (the fields the step reads)."""
    report = {"check": ws.DEFAULT_FORM, "plugin": "ppolag", "passed": True, "rehearsal_only": False,
              "allow_pending": False, "worktree_dirty": False, "commit_hash": COMMIT, "run_commits": [COMMIT, COMMIT],
              "result": {"identical": True}}
    report.update(changes)
    return report


def head_through(root: Path, last: str | None, **overrides: Any) -> FakeHead:
    """A HEAD at which every step up to ``last`` (inclusive; None: none) is done, except pilot-runs (the data root)."""
    done = STEP_NAMES[: STEP_NAMES.index(last) + 1] if last else []
    files: dict[str, Any] = {ws.TAG_SCRIPT: f'#!/usr/bin/env bash\nCOMMIT="{REGISTERED}"\n'.encode(),
                             **(FILLED_FILES if "registration-record" in done else REGISTERED_FILES)}
    facts: dict[str, Any] = {"tag": REGISTERED if "registration" in done else None,
                             "determinism": ws.DETERMINISM_ORDER if "determinism" in done else (),
                             "search": (True, "the search record committed at HEAD is complete (outcome 'excluded')")
                             if "search" in done else (False, "the search record committed at HEAD is not complete")}
    if "env" in done:
        files.update({ws.LOCK: LOCK_TEXT, ws.WORKSTATION: {"cpu": "test"}})
    if "fixed-batch" in done:
        files.update({ws.FIXED_BATCH_MANIFEST: MANIFEST,
                      ws.FIXED_BATCH_CHECK: {"passed": True, "manifest": HASHES, "commit": COMMIT}})
    if "determinism-default" in done:
        files["ledger/determinism/determinism-20261001T000000Z.json"] = default_report()
    if "allocation" in done:
        files[ws.ALLOCATION] = {"machine_hours_allocated": 12000.0, "decided_by": "the group",
                                "meeting_date": "2026-10-05"}
    if "pilot-enrich" in done:
        files.update({ws.PILOT_LEDGER: b"PAR1", ws.G4_FILE: {"mean_cost": {}}})
    if "go" in done:
        files["results/pilot/go_report-20261020T000000Z.json"] = {"generated": {"revision": 0, "commit": COMMIT}}
        files["results/analysis/pilot-20261020T000000Z/analysis.json"] = PASSED_ANALYSIS
    files.update(overrides.pop("files", {}))
    facts.update(overrides)
    return FakeHead(root, files, **facts)


def use(monkeypatch, head: ws.Head) -> ws.Head:
    monkeypatch.setattr(ws, "read_head", lambda: head)
    return head


def pilot_state(data_root: Path, default: str = "ledgered", statuses: dict[str, str] | None = None,
                replaced_by: dict[str, str] | None = None, extra: tuple[manifest.RunSpec, ...] = (),
                unwritten: tuple[str, ...] = ()) -> None:
    """The data root's scheduler state, made with the scheduler's own schema (pilot.scheduler.SCHEMA); a ledgered or
    excluded run is in the ledger (``ledger_written``) unless ``unwritten`` names it."""
    path = data_root / "scheduler" / "state.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.executescript(S.SCHEMA)
    for spec in [*manifest.design("pilot"), *extra]:
        status = (statuses or {}).get(spec.run_id, default)
        in_ledger = int(status in ("ledgered", "excluded") and spec.run_id not in unwritten)
        con.execute("INSERT INTO runs (run_id, arm_id, seed, spec, status, replaced_by, ledger_written, updated) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (spec.run_id, spec.arm_id, spec.seed, spec.to_json(), status, (replaced_by or {}).get(spec.run_id),
                     in_ledger, "2026-10-02T00:00:00+00:00"))
    con.commit()
    con.close()


def verdict(name: str, head: ws.Head, *argv: str) -> Any:
    args = ws.build_parser().parse_args(["next", *argv])
    return ws.judge(ws.STEP_BY_NAME[name], head, args)


# ---------------------------------------------------------------------------
# status: read-only, never fails
# ---------------------------------------------------------------------------


def test_status_lists_every_step_in_order_and_names_the_next_command(tree, monkeypatch, capsys) -> None:
    root, recorder = tree
    use(monkeypatch, head_through(root, "registration"))
    assert ws.main(["status", "--data-root", str(root / "data")]) == 0
    out = capsys.readouterr().out
    rows = re.findall(r"^\s*(\d+)\. (\S+)\s+task (\S+)\s+(DONE|TODO|BLOCKED) ", out, flags=re.M)
    # X-allocation (Table 9.1): the allocation right after the env step, before any check that runs on the workstation
    # X-registration: the registration record (task 3) is complete before any pilot run
    assert [r[1] for r in rows] == STEP_NAMES == ["registration", "env", "allocation", "fixed-batch",
                                                  "determinism-default", "determinism", "search",
                                                  "registration-record", "pilot-runs", "pilot-enrich", "go"]
    assert [r[2] for r in rows] == ["1", "8", "12", "8a", "9", "12a", "16", "3", "20", "21-23",
                                    "24-26"]  # HANDOVER tasks
    assert dict((r[1], r[3]) for r in rows)["registration"] == "DONE"
    assert dict((r[1], r[3]) for r in rows)["search"] == "BLOCKED"
    assert "next: python scripts/workstation.py env" in out and "which runs: bash scripts/setup_env.sh" in out
    assert recorder.calls == []  # status runs nothing


def test_status_outside_a_git_repository_says_so_and_exits_0(tree, monkeypatch, capsys) -> None:
    root, recorder = tree
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(root.parent))  # git never finds a repository above it
    assert ws.main(["status", "--data-root", str(root / "no-data-root")]) == 0
    out = capsys.readouterr().out
    assert "cannot read HEAD" in out and "none can be judged here" in out
    assert out.count("BLOCKED") >= len(STEP_NAMES) and recorder.calls == []
    assert ws.main(["next"]) == ws.EXIT_BLOCKED and recorder.calls == []  # nothing runs without HEAD


def test_status_reports_unreadable_facts_instead_of_failing(tree, monkeypatch, capsys) -> None:
    root, _ = tree
    data_root = root / "data"
    write(data_root, "scheduler/state.sqlite", b"not a database, a truncated copy")
    head = use(monkeypatch, head_through(root, "registration"))

    def broken() -> dict[str, str]:
        raise RuntimeError("git ls-tree failed")

    monkeypatch.setattr(head, "determinism_plugins", broken)
    assert ws.main(["status", "--data-root", str(data_root)]) == 0
    out = capsys.readouterr().out
    assert re.search(r"pilot-runs\s+task 20\s+BLOCKED\s+cannot read the scheduler state", out)
    assert re.search(r"determinism\s+task 12a\s+BLOCKED\s+cannot tell \(RuntimeError: git ls-tree failed\)", out)


# ---------------------------------------------------------------------------
# registration (task 1), env (task 8)
# ---------------------------------------------------------------------------


def test_registration_reads_the_commit_of_the_committed_tag_script(tree, monkeypatch, capsys) -> None:
    root, recorder = tree
    real = FakeHead(root, {ws.TAG_SCRIPT: (REPO / ws.TAG_SCRIPT).read_bytes()})
    assert ws.registration_commit(real) == REGISTERED  # the commit scripts/tag_registration.sh verifies
    assert verdict("registration", head_through(root, None)).state == "TODO"
    assert verdict("registration", head_through(root, "registration")).state == "DONE"
    moved = verdict("registration", head_through(root, None, tag="d" * 40))
    assert moved.state == "BLOCKED" and "never moved" in moved.why
    done = verdict("registration", head_through(root, "registration"))
    assert "local tag" in done.why and "cannot see the remote" in done.why  # pushing it is the operator's
    lightweight = verdict("registration", head_through(root, "registration", annotated=False))
    assert lightweight.state == "BLOCKED" and "annotated" in lightweight.why  # X-registration: an annotated tag
    use(monkeypatch, head_through(root, None))
    assert ws.main(["registration"]) == 0
    assert recorder.lines == ["bash scripts/tag_registration.sh"]
    assert "git push origin registration" in capsys.readouterr().out.splitlines()  # printed, never run


def test_the_tag_script_lists_the_root_commits_files_and_refuses_a_lightweight_tag(tmp_path) -> None:
    """scripts/tag_registration.sh in a clone of this repository (``git clone --shared``: nothing of it changes),
    with the user setting log.showRoot=false, under which ``git show`` lists no file of a root commit."""
    if subprocess.run(["git", "cat-file", "-e", f"{REGISTERED}^{{commit}}"], cwd=REPO, capture_output=True).returncode:
        pytest.skip("the registration commit is not in this clone")
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", "--shared", "--no-checkout", str(REPO), str(clone)], check=True)
    (clone / "scripts").mkdir()
    shutil.copy(REPO / ws.TAG_SCRIPT, clone / ws.TAG_SCRIPT)
    env = {**os.environ, "GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "log.showRoot", "GIT_CONFIG_VALUE_0": "false",
           "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "test@example.org"}

    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", *args], cwd=clone, env=env, capture_output=True, text=True)

    def tag_script() -> subprocess.CompletedProcess:
        return subprocess.run(["bash", ws.TAG_SCRIPT], cwd=clone, env=env, capture_output=True, text=True)

    git("tag", "-d", ws.TAG)  # the clone copies the tag when this repository has one
    created = tag_script()
    assert created.returncode == 0 and "created tag 'registration'" in created.stdout, created.stderr
    assert git("cat-file", "-t", f"refs/tags/{ws.TAG}").stdout.strip() == "tag"  # annotated
    again = tag_script()
    assert again.returncode == 0 and "already points at" in again.stdout
    git("tag", "-d", ws.TAG)
    git("tag", ws.TAG, REGISTERED)  # a lightweight tag on the right commit
    lightweight = tag_script()
    assert lightweight.returncode == 1 and "lightweight" in lightweight.stderr


def test_the_registration_record_is_a_peoples_step_before_the_pilot_runs(tree, monkeypatch, capsys) -> None:
    """X-registration: Tables 9.0 and 9.1 are filled in the docx and its regenerated PDF, committed, and the tag is
    pushed before any pilot run; the scheduler does not check it, so ``next`` stops there."""
    root, recorder = tree
    head = head_through(root, "search")
    blocked = verdict("registration-record", head)
    assert blocked.state == "BLOCKED" and not blocked.skippable and "Tables 9.0 and 9.1" in blocked.why
    docx_only = head_through(root, "search", files={"prereg/Preregistration.docx": FILLED_FILES[
        "prereg/Preregistration.docx"]})
    assert "prereg/Preregistration.pdf unchanged at HEAD" in verdict("registration-record", docx_only).why
    assert verdict("registration-record", head_through(root, "registration-record")).state == "DONE"
    use(monkeypatch, head)
    assert ws.main(["registration-record"]) == ws.EXIT_BLOCKED and recorder.calls == []
    assert "git push origin registration" in capsys.readouterr().out
    assert ws.main(["next", "--max-concurrent", "3", "--data-root", str(root / "data")]) == ws.EXIT_BLOCKED
    out = capsys.readouterr().out
    assert recorder.calls == [] and "next: registration-record (HANDOVER task 3" in out  # no pilot run is queued


def test_env_runs_setup_once_and_refuses_when_the_lock_exists(tree, monkeypatch, capsys) -> None:
    root, recorder = tree
    use(monkeypatch, head_through(root, "registration"))
    assert ws.main(["env"]) == 0
    out = capsys.readouterr().out
    assert recorder.lines == ["bash scripts/setup_env.sh"]
    assert f"git add {ws.LOCK} {ws.WORKSTATION}" in out.splitlines()
    assert any(line.startswith('git commit -m "Pinned environment') for line in out.splitlines())
    write(root, ws.LOCK, b"numpy==1.26.4\n")
    recorder.calls.clear()
    assert ws.main(["env"]) == ws.EXIT_BLOCKED and recorder.calls == []
    out = capsys.readouterr().out
    assert "--locked` is for the others" in out and f"git add {ws.LOCK} {ws.WORKSTATION}" in out
    assert verdict("env", head_through(root, "env")).state == "DONE"


# ---------------------------------------------------------------------------
# fixed-batch (task 8a)
# ---------------------------------------------------------------------------


RECOLLECTED = "f" * 64  # the sha256 of a re-collection that differs from the committed batch


def simulate_check(recorder: Recorder, *, differs: tuple[str, ...] = (), consistent: bool = True,
                   code: int | None = None) -> None:
    """``python -m metrics.collect_fixed_batch --all --check --json PATH`` as the recorder runs it: the result JSON
    written to PATH (``check``'s report) and the tool's exit code (1, EXIT_MISMATCH, when a batch differs)."""
    code = (1 if differs or not consistent else 0) if code is None else code

    def effect() -> None:
        cmd = recorder.calls[-1]
        tasks = {task: {"committed_sha256": sha, "recollected_sha256": RECOLLECTED if task in differs else sha,
                        "committed_consistent": consistent, "reproduced": task not in differs, "checks": {}}
                 for task, sha in HASHES.items()}
        problems = [f"{task}: re-collection byte-identical: False" for task in differs]
        write(Path("/"), cmd[cmd.index("--json") + 1].lstrip("/"),
              {"exit_code": code, "passed": code == 0, "problems": problems, "tasks": tasks})

    recorder.effects["collect_fixed_batch"] = effect
    recorder.codes["collect_fixed_batch"] = code


@pytest.fixture
def batch_tree(tree, monkeypatch) -> tuple[Path, Recorder]:
    """A clean tree at COMMIT with the lock committed and the batches' MANIFEST.json in the working tree; the check
    passes unless a test simulates another outcome."""
    root, recorder = tree
    use(monkeypatch, head_through(root, "env"))
    write(root, ws.FIXED_BATCH_MANIFEST, MANIFEST)
    monkeypatch.setattr(ws, "package_versions", lambda: dict(VERSIONS))  # the stack HEAD's lock pins
    monkeypatch.setattr(provenance, "require_clean_worktree", lambda cwd=None: COMMIT)
    monkeypatch.setattr(provenance, "commit_hash", lambda cwd=None: COMMIT)
    simulate_check(recorder)
    return root, recorder


def test_the_check_runs_with_its_json_result_outside_the_tree(batch_tree) -> None:
    root, recorder = batch_tree
    import metrics.collect_fixed_batch as collect_cli

    assert ws.FIXED_BATCH_MISMATCH == collect_cli.EXIT_MISMATCH
    assert ws.main(["fixed-batch"]) == 0
    (cmd,) = recorder.calls
    assert recorder.line(cmd).startswith("python -m metrics.collect_fixed_batch --all --check --json ")
    result = Path(cmd[cmd.index("--json") + 1])
    assert root not in result.parents and not result.exists()  # a temporary file, removed after the step
    assert sorted(p.name for p in (root / ws.FIXED_BATCH_DIR).iterdir()) == ["MANIFEST.json"]  # batches untouched


@pytest.mark.parametrize("outcome", ["cannot run", "no result", "records disagree", "other tasks"])
def test_a_check_without_a_result_or_with_disagreeing_records_writes_nothing_and_says_stop(batch_tree, capsys,
                                                                                         outcome) -> None:
    root, recorder = batch_tree
    if outcome == "cannot run":
        recorder.effects.clear()
        recorder.codes["collect_fixed_batch"] = 4  # EXIT_ERROR: the collection could not run; no JSON
    elif outcome == "no result":
        recorder.effects.clear()  # exit 0 but no result to record
    elif outcome == "records disagree":
        simulate_check(recorder, consistent=False)  # a committed record does not match its batch: not a finding
    else:
        simulate_check(recorder)
        recorder.effects["collect_fixed_batch"] = lambda: write(
            Path("/"), recorder.calls[-1][-1].lstrip("/"),
            {"exit_code": 0, "passed": True, "problems": [], "tasks": {"SafetyPointGoal1-v0": {
                "committed_sha256": "a" * 64, "recollected_sha256": "a" * 64, "committed_consistent": True,
                "reproduced": True}}})
    assert ws.main(["fixed-batch"]) != 0
    out = capsys.readouterr().out
    assert not (root / ws.FIXED_BATCH_CHECK).exists() and "git add" not in out
    assert "STOP" in out and "raise it with the group" in out


def test_a_recollection_that_differs_is_recorded_reported_and_counts_as_done(batch_tree, monkeypatch, capsys) -> None:
    """Table 9.1 (Q-appendix-a): the committed batches are the fixed batches; a re-collection that differs is a
    reproducibility finding, recorded with both hashes and reported in Table 9.1, never a replacement."""
    root, recorder = batch_tree
    simulate_check(recorder, differs=("SafetyCarGoal1-v0",))
    assert ws.main(["fixed-batch"]) == 0
    record = json.loads((root / ws.FIXED_BATCH_CHECK).read_text(encoding="utf-8"))
    assert record["passed"] is False and record["not_reproduced"] == ["SafetyCarGoal1-v0"]
    assert record["manifest"] == HASHES and record["commit"] == COMMIT
    assert record["recollected_sha256"] == {**HASHES, "SafetyCarGoal1-v0": RECOLLECTED}
    assert record["reading"] == ws.FIXED_BATCH_READING and "Q-appendix-a" in record["reading"]
    assert record["problems"] == ["SafetyCarGoal1-v0: re-collection byte-identical: False"]
    out = capsys.readouterr().out
    assert "WARNING: the re-collection on the workstation differs" in out
    assert f"committed {'b' * 64}, re-collected {RECOLLECTED}" in out
    (row,) = [line for line in out.splitlines() if line.startswith("Table 9.1 row: Fixed-batch reproducibility check |")]
    assert "SafetyCarGoal1-v0" in row and "remain the fixed batches" in row and row.endswith("| the group")
    assert f"git add {ws.FIXED_BATCH_DIR}/ {ws.FIXED_BATCH_CHECK}" in out.splitlines()
    assert sorted(p.name for p in (root / ws.FIXED_BATCH_DIR).iterdir()) == ["MANIFEST.json"]  # nothing rewritten
    # written, not committed: a second run checks nothing again and keeps the mismatch in the commit message
    assert ws.main(["fixed-batch"]) == 0 and len(recorder.calls) == 1
    assert shlex.split(capsys.readouterr().out.splitlines()[-1])[-1] == ws.FIXED_BATCH_MISMATCH_MESSAGE
    # done, with a warning, once HEAD holds the record and the manifest with the same hashes
    committed = {ws.FIXED_BATCH_MANIFEST: MANIFEST, ws.FIXED_BATCH_CHECK: record}
    done = verdict("fixed-batch", head_through(root, "allocation", files=committed))
    assert done.state == "DONE" and "SafetyCarGoal1-v0" in done.warning and "Table 9.1" in done.warning
    other = {**committed, ws.FIXED_BATCH_MANIFEST: {"tasks": {"SafetyPointGoal1-v0": {"sha256": "f" * 64}}}}
    assert verdict("fixed-batch", head_through(root, "allocation", files=other)).state == "BLOCKED"
    for broken in ({**record, "not_reproduced": []}, {**record, "passed": None}):
        files = {**committed, ws.FIXED_BATCH_CHECK: broken}
        assert verdict("fixed-batch", head_through(root, "allocation", files=files)).state == "BLOCKED"
    use(monkeypatch, head_through(root, "allocation", files=committed))
    assert ws.main(["status", "--data-root", str(root / "data")]) == 0
    out = capsys.readouterr().out
    assert re.search(r"fixed-batch\s+task 8a\s+DONE ", out)
    assert "    warning: the re-collection on the workstation differs for SafetyCarGoal1-v0" in out
    assert ws.main(["fixed-batch"]) == 0 and "warning: the re-collection" in capsys.readouterr().out


def test_a_committed_check_record_that_blocks_the_step_is_refused_with_its_reason(batch_tree, monkeypatch,
                                                                                capsys) -> None:
    """A record at HEAD without an outcome, or with other batch hashes than MANIFEST.json, is BLOCKED: the step says
    why and exits 3; it neither checks again nor prints commit lines (the record is written once)."""
    root, recorder = batch_tree
    for record in ({"manifest": HASHES, "commit": COMMIT},
                   {"passed": True, "manifest": {"SafetyPointGoal1-v0": "f" * 64}, "commit": COMMIT}, [1]):
        committed = {ws.FIXED_BATCH_MANIFEST: MANIFEST, ws.FIXED_BATCH_CHECK: record}
        use(monkeypatch, head_through(root, "allocation", files=committed))
        write(root, ws.FIXED_BATCH_CHECK, record)  # the working tree holds the committed file
        assert verdict("fixed-batch", ws.read_head()).state == "BLOCKED"
        assert ws.main(["fixed-batch"]) == ws.EXIT_BLOCKED and recorder.calls == []
        out = capsys.readouterr().out
        assert "refused:" in out and "raise it with the group" in out and "git add" not in out


def test_the_fixed_batch_check_runs_only_in_the_stack_heads_lock_pins(batch_tree, monkeypatch, capsys) -> None:
    """The record names HEAD's lock (its sha256) as the stack that re-collected the batches, so another interpreter's
    re-collection is never recorded, least of all as a reproducibility finding."""
    root, recorder = batch_tree
    for installed in ({**VERSIONS, "numpy": "2.1.0"}, {**VERSIONS, "mujoco": None}):
        monkeypatch.setattr(ws, "package_versions", lambda installed=installed: dict(installed))
        assert ws.main(["fixed-batch"]) == ws.EXIT_BLOCKED and recorder.calls == []
        out = capsys.readouterr().out
        assert "does not run the stack" in out and ("numpy 2.1.0 (the lock: 1.26.4)" in out
                                                    or "mujoco not installed (the lock: 2.3.0)" in out)
        assert not (root / ws.FIXED_BATCH_CHECK).exists()
    monkeypatch.setattr(ws, "package_versions", lambda: dict(VERSIONS))
    unpinned = head_through(root, "env", files={ws.LOCK: b"numpy==1.26.4\ntorch @ file:///wheels/torch.whl\n"})
    use(monkeypatch, unpinned)
    assert ws.main(["fixed-batch"]) == ws.EXIT_BLOCKED and "torch 2.14.0 (the lock: not pinned)" in (
        capsys.readouterr().out)
    use(monkeypatch, head_through(root, "env"))  # names as pip freeze writes them (safety_gymnasium): the same stack
    assert ws.main(["fixed-batch"]) == 0 and len(recorder.calls) == 1


def test_a_passing_fixed_batch_check_is_recorded_once_and_read_back_from_head(batch_tree, capsys) -> None:
    root, recorder = batch_tree
    assert ws.main(["fixed-batch"]) == 0
    record = json.loads((root / ws.FIXED_BATCH_CHECK).read_text(encoding="utf-8"))
    assert record["passed"] is True and record["commit"] == COMMIT and record["manifest"] == HASHES
    assert record["recollected_sha256"] == HASHES and "not_reproduced" not in record
    assert set(record["packages"]) == set(ws.FIXED_BATCH_PACKAGES) and record["python"]
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT.*\+00:00", record["checked_at"])
    out = capsys.readouterr().out
    assert f"git add {ws.FIXED_BATCH_DIR}/ {ws.FIXED_BATCH_CHECK}" in out.splitlines()
    # written once: a second run checks nothing again and only prints the commit lines
    assert ws.main(["fixed-batch"]) == 0 and len(recorder.calls) == 1
    # done once HEAD holds the record and the manifest with the same hashes
    committed = {ws.FIXED_BATCH_MANIFEST: MANIFEST, ws.FIXED_BATCH_CHECK: record}
    passed = verdict("fixed-batch", head_through(root, "env", files=committed))
    assert passed.state == "DONE" and passed.warning == ""
    other = {**committed, ws.FIXED_BATCH_MANIFEST: {"tasks": {"SafetyPointGoal1-v0": {"sha256": "f" * 64}}}}
    assert verdict("fixed-batch", head_through(root, "env", files=other)).state == "BLOCKED"
    assert verdict("fixed-batch", head_through(root, "env")).state == "TODO"


def test_the_fixed_batch_check_needs_a_clean_tree_and_the_committed_lock(batch_tree, monkeypatch, capsys) -> None:
    root, recorder = batch_tree

    def dirty(cwd=None) -> str:
        raise provenance.DirtyWorktreeError("tracked files have uncommitted changes: metrics/batches.py")

    monkeypatch.setattr(provenance, "require_clean_worktree", dirty)
    assert ws.main(["fixed-batch"]) == ws.EXIT_BLOCKED and recorder.calls == []
    assert "metrics/batches.py" in capsys.readouterr().out
    monkeypatch.setattr(provenance, "require_clean_worktree", lambda cwd=None: COMMIT)
    write(root, "metrics/batches.py", b"FIXED_BATCH_SHA256 = {}\n")  # untracked: the clean-tree check ignores it
    assert ws.main(["fixed-batch"]) == ws.EXIT_BLOCKED and recorder.calls == []
    assert "code that HEAD does not hold as it is: metrics/batches.py" in capsys.readouterr().out
    use(monkeypatch, head_through(root, "env", files={"metrics/batches.py": b"FIXED_BATCH_SHA256 = {}\n"}))
    assert ws.main(["fixed-batch"]) == 0 and len(recorder.calls) == 1  # committed as it is: the check runs
    (root / ws.FIXED_BATCH_CHECK).unlink()
    recorder.calls.clear()
    use(monkeypatch, head_through(root, "registration"))  # no lock at HEAD
    assert ws.main(["fixed-batch"]) == ws.EXIT_BLOCKED and recorder.calls == []
    assert not (root / ws.FIXED_BATCH_CHECK).exists()


# ---------------------------------------------------------------------------
# determinism-default (task 9), determinism (task 12a)
# ---------------------------------------------------------------------------


def test_only_a_passing_bitwise_report_of_ppolag_counts_as_task_9(tree) -> None:
    root, _ = tree
    det = "ledger/determinism/determinism-20261001T000000Z.json"
    assert verdict("determinism-default", head_through(root, "determinism-default")).state == "DONE"
    for change in ({"passed": False}, {"rehearsal_only": True}, {"worktree_dirty": True}, {"allow_pending": True},
                   {"check": S.DETERMINISM_REGISTERED_FORM}, {"plugin": "study_a"}, {"run_commits": [COMMIT, "d" * 40]},
                   {"result": {"identical": False}}):
        head = head_through(root, "fixed-batch", files={det: default_report(**change)})
        assert verdict("determinism-default", head).state == "TODO", change


def test_the_default_check_runs_once_and_prints_the_reports_to_commit(tree, monkeypatch, capsys) -> None:
    root, recorder = tree
    use(monkeypatch, head_through(root, "fixed-batch"))
    stamp = "ledger/determinism/determinism-20261002T120000Z"
    recorder.effects["determinism_check.py"] = lambda: (write(root, stamp + ".md", b"# report\n"),
                                                        write(root, stamp + ".json", default_report()))
    assert ws.main(["determinism-default"]) == 0
    assert recorder.lines == ["python scripts/determinism_check.py"]
    assert f"git add {stamp}.json {stamp}.md" in capsys.readouterr().out.splitlines()
    assert ws.main(["determinism-default"]) == 0 and len(recorder.calls) == 1  # written, not committed: not run again


def test_a_failed_determinism_check_is_raised_with_the_group_and_never_retried_on_its_commit(tree, monkeypatch,
                                                                                            capsys) -> None:
    """Table 3.1: two runs with the same seed and configuration must produce identical results. A check whose runs
    differed holds its step (not skippable) until a later commit: the group's fix, or the commit recording it."""
    root, recorder = tree
    use(monkeypatch, head_through(root, "fixed-batch"))
    stamp = "ledger/determinism/determinism-20261002T120000Z"
    failed = default_report(passed=False, same_commit=True, result={"identical": False})  # as the script writes it
    recorder.codes["determinism_check.py"] = 1
    recorder.effects["determinism_check.py"] = lambda: (write(root, stamp + ".md", b"# report\n"),
                                                        write(root, stamp + ".json", failed))
    assert ws.main(["next"]) == 1 and "Stop and raise it with the group" in capsys.readouterr().out
    recorder.calls.clear()
    held = verdict("determinism-default", head_through(root, "fixed-batch"))
    assert held.state == "BLOCKED" and not held.skippable and stamp + ".json" in held.why
    assert ws.main(["next"]) == ws.EXIT_BLOCKED and ws.main(["determinism-default"]) == ws.EXIT_BLOCKED
    assert recorder.calls == [] and "raise it with the group" in capsys.readouterr().out  # not run again
    later = head_through(root, "fixed-batch")
    later.commit = "d" * 40  # a later commit: the group's fix, or the commit that records the failed report
    assert verdict("determinism-default", later).state == "TODO"
    for change in ({"rehearsal_only": True}, {"same_commit": False}):  # a rehearsal, or HEAD moved: not a failure
        write(root, stamp + ".json", {**failed, **change})
        assert verdict("determinism-default", head_through(root, "fixed-batch")).state == "TODO", change
    # the registered form, per plug-in: a failed study_b check holds the step too (not skippable), and is not rerun
    monkeypatch.setattr(ws, "determinism_waiting", lambda p: ())
    write(root, "ledger/determinism/determinism-study_b-20261002T130000Z.json",
          {**failed, "check": S.DETERMINISM_REGISTERED_FORM, "plugin": "study_b"})
    head = use(monkeypatch, head_through(root, "determinism-default", determinism=("study_a", "unconstrained_ppo")))
    registered = verdict("determinism", head)
    assert registered.state == "BLOCKED" and not registered.skippable and "study_b: ledger/determinism/" in (
        registered.why)
    assert ws.main(["determinism"]) == ws.EXIT_BLOCKED and recorder.calls == []


def test_determinism_checks_run_in_order_and_refuse_a_plugin_that_waits_naming_its_keys(tree, monkeypatch,
                                                                                         capsys) -> None:
    root, recorder = tree
    use(monkeypatch, head_through(root, "determinism-default", determinism=("study_a",)))
    monkeypatch.setattr(ws, "determinism_waiting", lambda p: ("Q-studyb-eval",) if p == "study_b" else ())
    assert ws.main(["determinism"]) == ws.EXIT_BLOCKED
    assert recorder.lines == [f"python scripts/determinism_check.py --registered-form --plugin {p}"
                              for p in ("unconstrained_ppo", "study_a_pid")]
    out = capsys.readouterr().out
    assert "study_b: refused before anything is trained: the check waits on Q-studyb-eval" in out
    recorder.calls.clear()
    assert ws.main(["determinism", "--plugins", "study_a_pid", "unconstrained_ppo"]) == 0
    assert [c[-1] for c in recorder.calls] == ["unconstrained_ppo", "study_a_pid"]  # in the runbook's order
    recorder.codes["--plugin unconstrained_ppo"] = 1
    recorder.calls.clear()
    assert ws.main(["determinism"]) == 1 and [c[-1] for c in recorder.calls] == ["unconstrained_ppo"]


def test_the_determinism_refusal_is_the_scripts_own_and_lifts_with_the_amendments(monkeypatch) -> None:
    assert sorted(ws.DETERMINISM_ORDER) == sorted(S.DETERMINISM_PLUGINS)
    assert ws.DETERMINISM_ORDER[0] == ws.PILOT_STUDY_A_PLUGIN == next(
        s.plugin for s in manifest.pilot() if s.study == "A")
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset())
    assert "Q-jc-window" in ws.determinism_waiting("study_a")
    assert "Q-studyb-eval" in ws.determinism_waiting("study_b")  # its evaluation gate (scripts/determinism_check.py)
    assert S.DETERMINISM_LATE_ONSET_KEY in ws.determinism_waiting("study_a_pid")
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset(R.PENDING))
    assert all(ws.determinism_waiting(p) == () for p in ws.DETERMINISM_ORDER)


def test_a_blocked_determinism_step_holds_next_only_while_study_a_waits(tree, monkeypatch) -> None:
    root, _ = tree
    monkeypatch.setattr(ws, "determinism_waiting", lambda p: ("Q-jc-window",))
    blocked = verdict("determinism", head_through(root, "determinism-default"))
    assert blocked.state == "BLOCKED" and not blocked.skippable and "study_a (waiting on Q-jc-window)" in blocked.why
    only_others = verdict("determinism", head_through(root, "determinism-default", determinism=("study_a",)))
    assert only_others.state == "BLOCKED" and only_others.skippable  # the scheduler holds those plug-ins' runs


# ---------------------------------------------------------------------------
# allocation (task 12): the group's number
# ---------------------------------------------------------------------------


BASIS = "24 x 500 days x 0.9 availability"  # how the group derived H (python -m pilot allocate --basis)


def test_the_allocation_is_never_invented(tree, monkeypatch, capsys) -> None:
    root, recorder = tree
    use(monkeypatch, head_through(root, "env"))
    assert ws.main(["allocation"]) == ws.EXIT_USAGE and recorder.calls == []
    out = capsys.readouterr().out
    assert "Part 6 G2" in out and "Machine-hours allocated to the registered runs" in out
    assert "before the pilot's throughput is read" in out and "--basis TEXT" in out
    assert "before the determinism checks and the smoke runs on the workstation" in out  # X-allocation
    for partial in (["--hours", "12000", "--date", "2026-10-05"], ["--basis", BASIS, "--date", "2026-10-05"],
                    ["--hours", "12000", "--basis", "", "--date", "2026-10-05"]):  # H and its basis are the group's
        assert ws.main(["allocation", *partial]) == ws.EXIT_USAGE and recorder.calls == []
        assert "Machine-hours allocated to the registered runs" in capsys.readouterr().out  # what the group decides
    assert ws.main(["allocation", "--hours", "12000", "--basis", BASIS]) == ws.EXIT_USAGE  # no --date
    assert recorder.calls == [] and "--date YYYY-MM-DD" in capsys.readouterr().out
    assert ws.main(["allocation", "--hours", "12000", "--basis", BASIS, "--date", "2026-10-05"]) == 0
    assert recorder.calls == [[sys.executable, "-m", "pilot", "allocate", "--hours", "12000", "--basis", BASIS,
                               "--by", "the group", "--date", "2026-10-05"]]
    lines = capsys.readouterr().out.splitlines()
    assert "git add pilot/allocation.json" in lines and any("X-allocation" in line for line in lines)
    assert any(line.startswith('git commit -m "Machine-hour allocation: 12000 machine-hours') for line in lines)
    assert verdict("allocation", head_through(root, "allocation")).state == "DONE"
    # X-allocation's own options pass through: an H above two years, and the smoke roots used on the workstation
    recorder.calls.clear()
    assert ws.main(["allocation", "--hours", "20000", "--basis", BASIS, "--date", "2026-10-05", "--confirm",
                    "--smoke-root", "/scratch/a", "--smoke-root", "/scratch/b"]) == 0
    assert recorder.calls[0][-5:] == ["--confirm", "--smoke-root", "/scratch/a", "--smoke-root", "/scratch/b"]


def test_the_allocation_text_states_g2_as_amended_and_hours_are_passed_unrounded(tree, monkeypatch, capsys) -> None:
    """Table 9.1 (Q-g2-run-equivalents): the corrected run-equivalents decide G2; the registered 484 is reported."""
    from pilot import budget

    root, recorder = tree
    use(monkeypatch, head_through(root, "env"))
    assert ws.main(["allocation"]) == ws.EXIT_USAGE
    out = " ".join(capsys.readouterr().out.split())
    assert "Q-g2-run-equivalents" in out and "627.13 for the uncut design" in out  # docs/DECISIONS.md
    assert f"{budget.corrected_run_equivalents()['total']:.2f} for the uncut design" in out  # read, never retyped
    assert f"the registered {R.G2_RUN_EQUIVALENTS} run-equivalents are reported beside it and decide nothing" in out
    for hours in (12000.0, 12000.5, 0.1, 1e20, 8765.4321098765432):
        assert float(ws.number(hours)) == hours  # never rounded
    assert ws.number(12000.0) == "12000"
    assert ws.main(["allocation", "--hours", "8765.4321098765432", "--basis", BASIS, "--date", "2026-10-05"]) == 0
    assert recorder.calls[0][recorder.calls[0].index("--hours") + 1] == "8765.432109876543"


def test_the_determinism_checks_wait_for_the_committed_allocation(tree, monkeypatch, capsys) -> None:
    """X-allocation (Table 9.1): a determinism check reveals the workstation's speed, so it runs only once HEAD holds
    the allocation."""
    root, recorder = tree
    monkeypatch.setattr(ws, "determinism_waiting", lambda p: ())
    head = head_through(root, "fixed-batch")
    del head._files[ws.ALLOCATION]
    use(monkeypatch, head)
    for step in ("determinism-default", "determinism"):
        assert ws.main([step]) == ws.EXIT_BLOCKED and recorder.calls == []
        assert "the allocation is recorded before the determinism checks" in capsys.readouterr().out
    use(monkeypatch, head_through(root, "fixed-batch"))
    ws.main(["determinism-default"])
    assert recorder.lines == ["python scripts/determinism_check.py"]


# ---------------------------------------------------------------------------
# search (task 16) and next
# ---------------------------------------------------------------------------


def test_search_is_a_peoples_step_that_runs_only_the_read_only_check(tree, monkeypatch, capsys) -> None:
    root, recorder = tree
    use(monkeypatch, head_through(root, "determinism"))
    assert ws.main(["search"]) == ws.EXIT_BLOCKED
    assert recorder.lines == ["python -m studyb.search check"]
    out = capsys.readouterr().out
    assert "studyb/search/README.md" in out and "python -m studyb.search queries" in out
    assert verdict("search", head_through(root, "search")).state == "DONE"


def test_next_reports_the_search_and_goes_on_to_the_study_a_pilot_runs(tree, monkeypatch, capsys) -> None:
    root, recorder = tree
    data_root = root / "data"
    use(monkeypatch, head_through(root, "determinism", files=FILLED_FILES))  # the registration record is complete
    assert ws.main(["next", "--max-concurrent", "3", "--data-root", str(data_root)]) == ws.EXIT_BLOCKED
    assert recorder.lines == [f"python -m pilot schedule add --design pilot --data-root {data_root}",
                              f"python -m pilot schedule run --max-concurrent 3 --data-root {data_root}",
                              f"python -m pilot schedule status --data-root {data_root}"]
    out = capsys.readouterr().out
    assert "search (task 16): BLOCKED, going on" in out and "pilot-runs: not finished" in out


def test_next_runs_the_first_step_not_done_and_stops_at_a_peoples_step(tree, monkeypatch, capsys) -> None:
    root, recorder = tree
    use(monkeypatch, head_through(root, None))
    assert ws.main(["next"]) == 0 and recorder.lines == ["bash scripts/tag_registration.sh"]
    recorder.calls.clear()
    use(monkeypatch, head_through(root, "env"))
    assert ws.main(["next"]) == ws.EXIT_USAGE and recorder.calls == []  # the allocation waits for the group
    assert "Part 6 G2" in capsys.readouterr().out  # before any check that runs on the workstation (X-allocation)
    monkeypatch.setattr(ws, "determinism_waiting", lambda p: () if p == "study_a" else ("Q-pilot-unconstrained-seed",))
    use(monkeypatch, head_through(root, "determinism-default", determinism=("study_a",), files=FILLED_FILES))
    assert ws.main(["next", "--data-root", str(root / "data")]) == ws.EXIT_USAGE and recorder.calls == []
    out = capsys.readouterr().out  # past the determinism and search steps, pilot-runs needs --max-concurrent
    assert "determinism (task 12a): BLOCKED, going on" in out and "pilot-runs needs --max-concurrent" in out
    monkeypatch.setattr(ws, "determinism_waiting", lambda p: ("Q-jc-window",))
    use(monkeypatch, head_through(root, "determinism-default"))
    assert ws.main(["next"]) == ws.EXIT_BLOCKED and recorder.calls == []  # study_a's report holds the pilot
    assert "nothing was run" in capsys.readouterr().out
    data_root = root / "data"
    pilot_state(data_root)
    use(monkeypatch, head_through(root, "go"))
    assert ws.main(["next", "--data-root", str(data_root)]) == 0 and recorder.calls == []
    assert "task 27" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# pilot-runs (task 20)
# ---------------------------------------------------------------------------


def test_the_pilot_runs_need_the_operators_concurrency(tree, monkeypatch, capsys) -> None:
    root, recorder = tree
    use(monkeypatch, head_through(root, "search"))
    assert ws.main(["pilot-runs", "--data-root", str(root / "data")]) == ws.EXIT_USAGE and recorder.calls == []
    out = capsys.readouterr().out
    assert f"--max-concurrent {ws.suggested_concurrency()}" in out and "physical cores" in out
    assert ws.suggested_concurrency() == max(1, ws.physical_cores() - 1) >= 1


def test_queued_pilot_runs_are_not_queued_again_and_the_data_root_is_passed_through(tree, monkeypatch) -> None:
    root, recorder = tree
    data_root = root / "data"
    pilot_state(data_root, default="pending")
    use(monkeypatch, head_through(root, "search"))
    assert ws.main(["pilot-runs", "--max-concurrent", "7", "--data-root", str(data_root)]) == ws.EXIT_BLOCKED
    assert recorder.lines == [f"python -m pilot schedule run --max-concurrent 7 --data-root {data_root}",
                              f"python -m pilot schedule status --data-root {data_root}"]
    pilot_state(data_root / "done")
    recorder.calls.clear()
    assert ws.main(["pilot-runs", "--max-concurrent", "7", "--data-root", str(data_root / "done")]) == 0  # done
    assert recorder.calls == []
    recorder.codes["schedule run"] = 130  # Ctrl-C: its own exit status, nothing more is run
    assert ws.main(["pilot-runs", "--max-concurrent", "7", "--data-root", str(data_root)]) == 130
    assert recorder.lines[-1].startswith("python -m pilot schedule run")


def test_the_pilot_runs_are_done_when_every_pilot_run_finished_without_a_pending_decision(tree) -> None:
    root, _ = tree
    head = head_through(root, "search")
    ids = [s.run_id for s in manifest.design("pilot")]
    pilot_state(root / "all")
    assert verdict("pilot-runs", head, "--data-root", str(root / "all")).state == "DONE"
    pilot_state(root / "one-training", statuses={ids[0]: "training"})
    assert verdict("pilot-runs", head, "--data-root", str(root / "one-training")).state == "TODO"
    pilot_state(root / "excluded", statuses={ids[0]: "excluded"})
    excluded = verdict("pilot-runs", head, "--data-root", str(root / "excluded"))
    # the circuit breaker: an operational pause (Q-exclusion-breaker); the operator diagnoses, then `schedule resolve
    # RUN replace`
    assert excluded.state == "BLOCKED" and "decision" in excluded.why
    replacement = manifest.design("pilot")[0].with_seed(5)
    pilot_state(root / "replaced", statuses={ids[0]: "excluded"}, replaced_by={ids[0]: replacement.run_id},
                extra=(replacement,))
    assert verdict("pilot-runs", head, "--data-root", str(root / "replaced")).state == "DONE"
    # replaced, but its exclusion is not in the ledger yet (Part 5.6): `schedule status` lists it among the decisions
    pilot_state(root / "unwritten", statuses={ids[0]: "excluded"}, replaced_by={ids[0]: replacement.run_id},
                extra=(replacement,), unwritten=(ids[0],))
    unwritten = verdict("pilot-runs", head, "--data-root", str(root / "unwritten"))
    assert unwritten.state == "BLOCKED" and ids[0] in unwritten.why
    decisions = S.Scheduler(S.SchedulerConfig(data_root=root / "unwritten")).status_report()["decisions"]
    assert [run_id for run_id, _ in decisions] == [ids[0]]  # the scheduler's own list
    pilot_state(root / "held", statuses={ids[1]: "interrupted"})
    assert verdict("pilot-runs", head, "--data-root", str(root / "held")).state == "BLOCKED"
    assert verdict("pilot-runs", head, "--data-root", str(root / "none")).state == "TODO"


def test_pending_pilot_runs_held_by_peoples_steps_are_blocked_with_the_gate(tree, monkeypatch) -> None:
    root, _ = tree
    pilot_state(root / "data", default="pending")
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset(R.PENDING))
    head = head_through(root, "determinism")
    del head._files[ws.ALLOCATION]
    no_allocation = verdict("pilot-runs", head, "--data-root", str(root / "data"))
    assert no_allocation.state == "BLOCKED" and "pilot/allocation.json not committed (8)" in no_allocation.why
    no_search = verdict("pilot-runs", head_through(root, "determinism"), "--data-root", str(root / "data"))
    assert no_search.state == "TODO"  # the six Study A runs can launch; the scheduler holds the Study B ones
    study_a_ledgered = {s.run_id: "ledgered" for s in manifest.design("pilot") if s.study == "A"}
    pilot_state(root / "b", default="pending", statuses=study_a_ledgered)
    held = verdict("pilot-runs", head_through(root, "determinism"), "--data-root", str(root / "b"))
    assert held.state == "BLOCKED" and "search_record_missing (2)" in held.why


# ---------------------------------------------------------------------------
# pilot-enrich (tasks 21 to 23), go (tasks 24 to 26)
# ---------------------------------------------------------------------------


def test_the_enrichment_chain_runs_in_order_and_stops_at_a_failure(tree, monkeypatch, capsys) -> None:
    root, recorder = tree
    data_root = str(root / "data")
    use(monkeypatch, head_through(root, "search"))
    assert ws.main(["pilot-enrich", "--data-root", data_root]) == ws.EXIT_BLOCKED and recorder.calls == []  # no ledger
    write(root, ws.PILOT_LEDGER, b"PAR1")
    recorder.codes["enrich measure"] = 3  # refused (a refusing step, or an open question)
    assert ws.main(["pilot-enrich", "--data-root", data_root]) == 3
    assert recorder.lines == [f"python -m pilot enrich {a} --ledger {ws.PILOT_LEDGER} --data-root {data_root}"
                              for a in ("select", "measure")]
    assert "git add" not in capsys.readouterr().out
    del recorder.codes["enrich measure"]
    recorder.calls.clear()
    assert ws.main(["pilot-enrich", "--data-root", data_root]) == 0
    assert [line.split(" --")[0] for line in recorder.lines] == [
        "python -m pilot enrich select", "python -m pilot enrich measure", "python -m pilot enrich battery",
        "python -m pilot g4-measure"]
    assert "git add results/pilot/" in capsys.readouterr().out.splitlines()
    write(root, ws.G4_FILE, {"mean_cost": {}})  # taken once: never measured again
    recorder.calls.clear()
    assert ws.main(["pilot-enrich", "--data-root", data_root]) == 0
    assert not any("g4-measure" in line for line in recorder.lines)
    assert verdict("pilot-enrich", head_through(root, "pilot-enrich")).state == "DONE"


def test_the_go_step_writes_the_report_and_the_analysis_test_run_once(tree, monkeypatch, capsys) -> None:
    root, recorder = tree
    use(monkeypatch, head_through(root, "pilot-enrich"))
    assert ws.main(["go", "--hours-per-run", "12.5"]) == ws.EXIT_USAGE and recorder.calls == []  # both or neither
    assert ws.main(["go", "--hours-per-run", "12.5", "--concurrent", "7"]) == 0
    assert recorder.lines == ["python -m pilot go", f"python -m analysis --mode pilot --ledger {ws.PILOT_LEDGER}",
                              "python -m pilot surplus --hours-per-run 12.5 --concurrent 7"]
    out = capsys.readouterr().out
    assert "git add results/pilot/ results/analysis/" in out.splitlines() and "task 27" in out
    write(root, "results/pilot/go_report-20261020T000000Z.json", {"generated": {"revision": 0}})
    write(root, "results/analysis/pilot-20261020T000000Z/analysis.json", PASSED_ANALYSIS)
    recorder.calls.clear()
    assert ws.main(["go"]) == 0 and recorder.calls == []  # neither is written twice (Part 5.8: run once)
    recorder.codes["analysis"] = 3
    (root / "results/analysis/pilot-20261020T000000Z/analysis.json").unlink()
    assert ws.main(["go"]) == 3 and "disagree on a number" in capsys.readouterr().out
    # python -m analysis wrote its output and exited 3 (a mismatch) or 4 (nothing compared): that output holds the
    # step, here and once committed, until the error is fixed and the analysis run again (Part 5.8)
    recorder.calls.clear()
    for code, pilot in ((3, {"mismatches": ["go condition G1: x is 1 in the analysis and 2 in the go report"],
                             "compared": 4}), (4, {"mismatches": [], "compared": 0})):
        write(root, "results/analysis/pilot-20261020T000000Z/analysis.json", {"pilot": pilot})
        assert ws.main(["go"]) == code and recorder.calls == []
        out = capsys.readouterr().out
        assert "fix the error" in out and "git add" not in out and "task 27" not in out
        committed = {"results/pilot/go_report-20261020T000000Z.json": {"generated": {"revision": 0}},
                     "results/analysis/pilot-20261020T000000Z/analysis.json": {"pilot": pilot}}
        failed = verdict("go", head_through(root, "pilot-enrich", files=committed))
        assert failed.state == "BLOCKED" and not failed.skippable and f"exit {code}" in failed.why
    fixed = {**committed, "results/analysis/pilot-20261021T000000Z/analysis.json": PASSED_ANALYSIS}
    assert verdict("go", head_through(root, "pilot-enrich", files=fixed)).state == "DONE"  # the latest run decides
    repilot = {"results/pilot/go_report-20261120T000000Z.json": {"generated": {"argv": ["go", "--repilot", "G1"]}},
               "results/analysis/pilot-20261120T000000Z/analysis.json": PASSED_ANALYSIS}
    assert verdict("go", head_through(root, "pilot-enrich", files=repilot)).state == "TODO"  # a re-pilot's report
    assert verdict("go", head_through(root, "go")).state == "DONE"


def test_the_go_reports_revision_is_read_as_python_m_pilot_freeze_reads_it(tmp_path) -> None:
    from pilot import enrichment

    variants = [{"generated": {"revision": 0}}, {"generated": {"revision": 1}}, {"generated": {"revision": True}},
                {"generated": {"revision": False}}, {"generated": {"argv": ["go", "--repilot", "G1"]}},
                {"generated": {"argv": ["go"]}}, {"generated": {}}, {}, {"generated": "x"}]
    for i, report in enumerate(variants):
        (tmp_path / f"go_report-2026102{i}T000000Z.json").write_text(json.dumps(report), encoding="utf-8")
    listed = enrichment.list_go_reports(tmp_path, read=lambda p: p.read_bytes())
    assert [r["revision"] if r["readable"] else None for r in listed] == [ws.go_revision(v) for v in variants]


# ---------------------------------------------------------------------------
# The command lines are the tools' real ones; the script never commits, tags or pushes
# ---------------------------------------------------------------------------


def _every_call(root: Path, recorder: Recorder, monkeypatch) -> list[list[str]]:
    """Run every step once on its happy path; every command the runbook calls."""
    data_root = str(root / "data")
    monkeypatch.setattr(ws, "determinism_waiting", lambda p: ())
    monkeypatch.setattr(ws, "package_versions", lambda: dict(VERSIONS))
    monkeypatch.setattr(provenance, "require_clean_worktree", lambda cwd=None: COMMIT)
    monkeypatch.setattr(provenance, "commit_hash", lambda cwd=None: COMMIT)
    write(root, ws.FIXED_BATCH_MANIFEST, MANIFEST)
    write(root, ws.PILOT_LEDGER, b"PAR1")
    argv = {"allocation": ["--hours", "12000", "--basis", BASIS, "--date", "2026-10-05", "--confirm",
                           "--smoke-root", data_root],
            "pilot-runs": ["--max-concurrent", "7", "--data-root", data_root],
            "pilot-enrich": ["--data-root", data_root], "go": ["--hours-per-run", "12.5", "--concurrent", "7"]}
    for i, name in enumerate(STEP_NAMES):
        use(monkeypatch, head_through(root, STEP_NAMES[i - 1] if i else None, files={ws.LOCK: LOCK_TEXT}))
        ws.main([name, *argv.get(name, [])])
    return recorder.calls


class _Parsed(Exception):
    """Raised by a stub right after a script parsed its command line."""


def test_every_command_the_runbook_calls_parses_with_the_tools_own_command_line(tree, monkeypatch) -> None:
    import analysis.__main__ as analysis_cli
    import metrics.collect_fixed_batch as collect_cli
    import pilot.__main__ as pilot_cli
    import studyb.search.__main__ as search_cli

    root, recorder = tree
    calls = _every_call(root, recorder, monkeypatch)
    assert {c[1] for c in calls if c[0] == "bash"} == {"scripts/tag_registration.sh", "scripts/setup_env.sh"}
    assert all((REPO / c[1]).is_file() for c in calls if c[0] == "bash")
    det = _load(REPO / "scripts" / "determinism_check.py", "determinism_check_cli")
    checked: list[list[str]] = []
    monkeypatch.setattr(collect_cli, "check", lambda tasks, out=None, report=None: checked.append(list(tasks)) or 0)
    monkeypatch.setattr(search_cli, "check_record", lambda root: type("S", (), {"complete": True, "outcome": "excluded",
                                                                                "problems": []})())

    def stop(total: int, plugin: str) -> None:
        raise _Parsed((total, plugin))

    monkeypatch.setattr(det, "make_spec", stop)
    seen: set[str] = set()
    for cmd in (c for c in calls if c[0] == sys.executable):
        tool, rest = (cmd[2], cmd[3:]) if cmd[1] == "-m" else (cmd[1], cmd[2:])
        seen.add(tool)
        if tool == "pilot":
            args = pilot_cli.build_parser().parse_args(rest)
            assert getattr(args, "func", None) is not None
            if rest[:2] == ["schedule", "run"]:
                assert args.max_concurrent == 7 and args.data_root == str(root / "data")
        elif tool == "analysis":
            args = analysis_cli.parser().parse_args(rest)
            assert args.mode == "pilot" and args.ledger == ws.PILOT_LEDGER
        elif tool == "metrics.collect_fixed_batch":
            assert rest[:3] == ["--all", "--check", "--json"] and len(rest) == 4  # the runbook's temporary result file
            rest = [*rest[:3], str(root.parent / "check.json")]  # that directory is gone after the step
            assert collect_cli.main(rest) == 0 and checked == [list(R.TASKS_STUDY_A)]  # --check of every task
        elif tool == "studyb.search":
            assert search_cli.main(rest) == 0
        elif tool == ws.DETERMINISM_SCRIPT:
            with pytest.raises(_Parsed) as parsed:
                det.main(rest)
            total, plugin = parsed.value.args[0]
            registered = "--registered-form" in rest
            assert total == (R.CHECKPOINT_INTERVAL_STEPS if registered else 2 * R.STEPS_PER_EPOCH)
            assert plugin == (rest[rest.index("--plugin") + 1] if registered else "ppolag")
        else:
            raise AssertionError(f"unexpected command {cmd}")
    assert seen == {"pilot", "analysis", "metrics.collect_fixed_batch", "studyb.search", ws.DETERMINISM_SCRIPT}
    assert ws.DEFAULT_FORM in (REPO / ws.DETERMINISM_SCRIPT).read_text(encoding="utf-8")  # the report's "check"


def test_the_runbook_never_commits_tags_or_pushes(tree, monkeypatch) -> None:
    root, recorder = tree
    calls = _every_call(root, recorder, monkeypatch)
    assert calls and not any(c[0] == "git" for c in calls)
    source = (REPO / "scripts" / "workstation.py").read_text(encoding="utf-8")
    assert set(re.findall(r'\["git", "([a-z-]+)"', source)) == {"rev-parse"}  # its only own git command reads
    assert ws.commit_lines(["a b"], 'say "x"') == ["git add 'a b'", "git commit -m 'say \"x\"'"]  # quoted for a shell
    assert shlex.split(ws.commit_lines(["p"], "Plain message (task 1)")[1]) == ["git", "commit", "-m",
                                                                                 "Plain message (task 1)"]


def test_call_prints_the_command_and_runs_it_from_the_repository_root(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(ws, "REPO", tmp_path)
    seen: dict[str, Any] = {}

    def run(cmd, cwd=None):
        seen.update(cmd=cmd, cwd=cwd)
        return type("Done", (), {"returncode": 4})()

    monkeypatch.setattr(ws.subprocess, "run", run)
    assert ws.call([sys.executable, "-m", "pilot", "go"]) == 4
    assert seen == {"cmd": [sys.executable, "-m", "pilot", "go"], "cwd": tmp_path}
    assert capsys.readouterr().out == "+ " + shlex.join([sys.executable, "-m", "pilot", "go"]) + "\n"


def test_every_cited_quotation_is_in_the_pre_registration() -> None:
    quotes = _load(REPO / "tests" / "test_core_quotes.py", "core_quotes_workstation")
    source = (REPO / "scripts" / "workstation.py").read_text(encoding="utf-8")
    found = quotes.cited_quotes(source)
    assert len(found) >= 5 and any("before the pilot's throughput is read" in q for q in found)
    assert not {q: m for q in found if (m := quotes.missing_pieces(q))}

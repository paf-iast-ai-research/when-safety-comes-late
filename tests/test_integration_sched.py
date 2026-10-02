"""The scheduler side of the pipeline integration (pilot.contracts.evaluator_target; pilot/manifest.py; pilot/scheduler.py; pilot.scheduler.CUTS_PATH).

* battery continuations (Table 2.2; Q-continuations): the fine-tuning and transfer continuations are
  queued from matched ledger rows only (gate Q-arm-complete), end ``continued`` without a ledger row,
  and an excluded one is left to the group like an excluded few-shot continuation;
* evaluation harnesses by target: a missing harness holds only the runs it evaluates (keyed by
  ``contracts.evaluator_target``), and ``schedule status`` names it;
* launch gates (``Scheduler.launch_gates``): each holds the runs it names and releases them only when
  HEAD holds what it requires (a temporary git repository; uncommitted files never count); none applies
  in smoke mode;
* the cut order of Part 6.1: ``add(..., cuts=K)`` applies the first K cuts of the committed ``pilot/cuts.json``.

Runs are "trained" by tests/fake_launcher.py (no OmniSafe).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from configs import registered as R
from pilot import errors, manifest
from pilot import scheduler as S
from pilot.ledger_writer import LedgerPaths
from pilot.manifest import RunSpec
from pilot.scheduler import Scheduler, SchedulerConfig, SchedulerError

FAKE = Path(__file__).with_name("fake_launcher.py")
SHORT = 2_000_000  # 100 epochs: the fake launcher's ten-checkpoint selection window fits


def shortened(spec: RunSpec, total: int = SHORT) -> RunSpec:
    return RunSpec.from_dict({**spec.to_dict(), "total_steps": total})


def plain_spec(seed: int = 0, name: str = "T-A-PointGoal1-N0.00", *, pilot: bool = False, group: str = "main",
               plugin: str = "ppolag") -> RunSpec:
    return RunSpec(run_id=f"{name}-s{seed}", study="A", task=R.PRIMARY_TASK, arm="N0.00", seed=seed, total_steps=SHORT,
                   base_algo="PPOLag", plugin=plugin, group=group, pilot=pilot, N=0.0, onset_step=0)


def study_a_parent(seed: int = 0) -> RunSpec:
    """The N = 0 main-sweep arm on the primary task, shortened (plug-in study_a: the fake writes plasticity.csv)."""
    return shortened(next(s for s in manifest.study_a_main((seed,)) if s.task == R.PRIMARY_TASK and s.N == 0.0))


def study_b_run(arm: str = "Moderate", seed: int = 0, *, pilot: bool = False) -> RunSpec:
    if pilot:
        return shortened(next(s for s in manifest.pilot() if s.arm == arm))
    return shortened(next(s for s in manifest.study_b((seed,)) if s.arm == arm))


@pytest.fixture
def env(tmp_path, monkeypatch):
    scenarios: dict[str, str] = {}
    scen_file = tmp_path / "scenarios.json"

    def set_scenarios(**kw):
        scenarios.update(kw)
        scen_file.write_text(json.dumps(scenarios), encoding="utf-8")

    set_scenarios()
    monkeypatch.setenv("FAKE_SCENARIOS", str(scen_file))
    monkeypatch.setenv("FAKE_RELEASE", str(tmp_path / "release"))
    paths = LedgerPaths(main=tmp_path / "ledger.parquet", pilot=tmp_path / "pilot.parquet", sidecar_dir=tmp_path / "sidecar")

    def make(**kw) -> Scheduler:
        cfg = SchedulerConfig(
            data_root=kw.pop("data_root", tmp_path / "data"), poll_seconds=0.05, ledger_paths=kw.pop("ledger_paths", paths),
            command_builder=lambda stage, spec_path, run_dir: [sys.executable, str(FAKE), stage, str(spec_path), str(run_dir)],
            **{"allow_dirty": True, "allow_pending": True, **kw},  # smoke mode unless a test asks for the registered gates
        )
        return Scheduler(cfg)

    return make, set_scenarios, paths, tmp_path


def ledger_rows(path: Path) -> list:
    from results.ledger_schema import load_ledger_as_rows

    return load_ledger_as_rows(path) if path.exists() else []


def match(paths: LedgerPaths, run_id: str, step: int = 1_800_000, matched: bool | None = True) -> None:
    """What `enrich select` and `enrich match` write: the matched checkpoint and the arm's flag."""
    from results.ledger_schema import write_enrichment

    row = next(r for r in ledger_rows(paths.main) if r.run_id == run_id)
    path = next(c.path for c in row.checkpoints if c.step == step)
    write_enrichment(run_id, {"matched_checkpoint_step": step, "matched_checkpoint_path": path, "matched": matched},
                     paths.main)


# ---------------------------------------------------------------------------
# Battery continuations (Table 2.2; Q-continuations)
# ---------------------------------------------------------------------------


def test_battery_continuations_are_queued_from_matched_rows_and_end_continued(env) -> None:
    make, _, paths, _ = env
    parent = study_a_parent()
    sched = make(max_concurrent=2)
    sched.add([parent])
    sched.run(max_passes=200)
    assert sched.row(parent.run_id)["status"] == "ledgered"
    with pytest.MonkeyPatch.context() as keys:
        keys.setattr(R, "ANSWERED_QUESTIONS", frozenset(R.ANSWERED_QUESTIONS) - {"Q-arm-complete"})
        with pytest.raises(errors.PendingQuestionError, match="Q-arm-complete"):
            sched.add_continuations(paths.main)  # while it is open, which arms are matched waits on Q-arm-complete
        with errors.allow_pending():
            assert sched.add_continuations(paths.main) == []  # matched is None: the arm waits for `enrich match`
    # answered (Table 9.1): no gate holds the plan
    assert sched.add_continuations(paths.main) == []  # matched is None: the arm waits for `enrich match`
    _, skipped = sched.plan_continuations(paths.main)
    assert skipped == [(parent.run_id, "its arm is not matched yet (`enrich match`)")]
    match(paths, parent.run_id)
    added = sched.add_continuations(paths.main)
    assert sched.add_continuations(paths.main) == []  # queued once
    finetune, transfer = (f"{parent.arm_id}-{c}-s0" for c in manifest.BATTERY_CONDITIONS)
    assert added == [finetune, transfer]
    for run_id, condition in ((finetune, "finetune"), (transfer, "transfer")):
        spec = sched.spec(run_id)
        assert spec == manifest.battery_continuation(parent, next(iter(ledger_rows(paths.main))), condition)
        assert spec.params["parent_step"] == 1_800_000 and spec.depends_on == (parent.run_id,)
    assert sched.spec(transfer).task == R.TRANSFER_TASKS[R.PRIMARY_TASK]
    sched.run(max_passes=200)
    assert {sched.row(r)["status"] for r in (finetune, transfer)} == {"continued"}
    assert [r.run_id for r in ledger_rows(paths.main)] == [parent.run_id]  # a continuation is never a row
    assert any("`enrich continuations`" in (e["detail"] or "") for e in sched.events(finetune) if e["event"] == "continued")
    # the launcher's record of the parent checkpoint the continuation started from (pilot/dependencies.py)
    train = json.loads((sched.cfg.run_dir(finetune) / "train_result.json").read_text())
    dep = train["dependencies"][parent.run_id]
    assert dep["checkpoint_step"] == 1_800_000 and dep["checkpoint_sha256"] == hashlib.sha256(b"").hexdigest()


def test_an_excluded_battery_continuation_is_left_to_the_group(env) -> None:
    make, set_scenarios, paths, _ = env
    parent = study_a_parent()
    finetune = f"{parent.arm_id}-finetune-s0"
    set_scenarios(**{finetune: "crash"})
    sched = make()
    sched.add([parent])
    sched.run(max_passes=200)
    match(paths, parent.run_id)
    with errors.allow_pending():
        sched.add_continuations(paths.main, conditions=("finetune",))
    sched.run(max_passes=200)
    row = sched.row(finetune)
    assert (row["status"], row["replaced_by"], row["ledger_written"]) == ("excluded", None, 1)
    assert not [r for r in sched.rows() if r["run_id"].startswith(f"{parent.arm_id}-finetune-") and r["run_id"] != finetune]
    assert any("amendment" in (e["detail"] or "") for e in sched.events(finetune) if e["event"] == "needs_decision")
    with pytest.raises(SchedulerError, match="amendment"):
        sched.resolve(finetune, "replace")  # with_seed refuses a continuation: no seed of its own
    assert [r.run_id for r in ledger_rows(paths.main)] == [parent.run_id]  # no exclusion row for a continuation
    decisions = dict(sched.status_report()["decisions"])
    assert "amendment" in decisions[finetune]


def test_continuations_skip_unmatched_foreign_and_already_measured_rows(env, monkeypatch) -> None:
    from results.ledger_schema import write_enrichment

    make, _, paths, tmp = env
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset({"Q-arm-complete"}))
    parents = [study_a_parent(seed) for seed in (0, 1, 2, 3)]
    sched = make(max_concurrent=4)
    sched.add(parents)
    sched.run(max_passes=200)
    match(paths, parents[0].run_id, matched=False)  # rule 6: unmatched, left out of the battery
    match(paths, parents[1].run_id)
    write_enrichment(parents[1].run_id, {"gap_finetune": 1.5}, paths.main)  # its fine-tuning continuation was read
    match(paths, parents[2].run_id)
    match(paths, parents[3].run_id)
    # a ledger of another data root: the run directory records another commit
    train_path = sched.cfg.run_dir(parents[3].run_id) / "train_result.json"
    train_path.write_text(json.dumps({**json.loads(train_path.read_text()), "commit_hash": "f" * 40}))
    specs, skipped = sched.plan_continuations(paths.main)
    reasons = dict(skipped)
    assert "unmatched" in reasons[parents[0].run_id] and "Part 4.1 rule 6" in reasons[parents[0].run_id]
    assert [s.run_id for s in specs] == [f"{parents[1].arm_id}-transfer-s1", f"{parents[2].arm_id}-finetune-s2",
                                         f"{parents[2].arm_id}-transfer-s2"]
    assert "gap_finetune" in [r for rid, r in skipped if rid == parents[1].run_id][0]
    assert "another" not in reasons[parents[0].run_id] and "commit" in reasons[parents[3].run_id]
    # a run already in a ledger (or sidecar) is never queued again
    paths.sidecar_dir.mkdir(parents=True, exist_ok=True)
    (paths.sidecar_dir / f"{parents[2].arm_id}-finetune-s2.json").write_text("{}")
    added = sched.add_continuations(paths.main)
    assert added == [f"{parents[1].arm_id}-transfer-s1", f"{parents[2].arm_id}-transfer-s2"]
    assert [e["event"] for e in sched.events(f"{parents[2].arm_id}-finetune-s2")] == ["already_in_ledger"]
    # a row whose run this data root never queued
    other = make(data_root=tmp / "other")
    _, skipped_other = other.plan_continuations(paths.main)
    assert "the run is not queued on this data root" in dict(skipped_other).values()


def test_continuations_refuse_the_pilot_ledger_and_bad_conditions(env, monkeypatch) -> None:
    make, _, paths, _ = env
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset({"Q-arm-complete"}))
    sched = make()
    with pytest.raises(SchedulerError, match="Part 3.6"):
        sched.add_continuations(paths.pilot)
    with pytest.raises(SchedulerError, match="conditions"):
        sched.plan_continuations(paths.main, conditions=("hazard",))
    with pytest.raises(SchedulerError, match="conditions"):
        sched.plan_continuations(paths.main, conditions=("finetune", "finetune"))
    with pytest.raises(SchedulerError, match="does not exist"):
        sched.plan_continuations(paths.main)


def test_a_replaced_parent_supersedes_every_kind_of_continuation(env, monkeypatch) -> None:
    """_replacement_sql's supersede loop covers every continuation group (pilot/manifest.py)."""
    make, set_scenarios, _, _ = env
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset({"Q-seed-collision"}))
    parent = study_a_parent()
    row = {"run_id": parent.run_id, "completed": True, "matched": True, "matched_checkpoint_step": 1_800_000,
           "matched_checkpoint_path": "x/epoch-90.pt", "commit_hash": "a" * 40}
    cont = manifest.battery_continuation(parent, row, "finetune")
    set_scenarios(**{parent.run_id: "crash"})
    sched = make()
    sched.add([parent, cont])  # (a continuation is normally queued only after its parent is ledgered)
    sched.run(max_passes=5)
    assert sched.row(parent.run_id)["status"] == "excluded" and sched.row(parent.run_id)["replaced_by"]
    assert sched.row(cont.run_id)["status"] == "superseded"
    with pytest.raises(SchedulerError, match="amendment"):
        sched.resolve(cont.run_id, "replace")
    assert not [r for r in sched.rows() if sched.spec(r["run_id"]).group in manifest.CONTINUATION_GROUPS
                and r["status"] != "superseded"]  # the replacement is continued once matched, not here


def test_status_offers_a_blocked_battery_continuation_only_what_resolve_accepts(env) -> None:
    """`schedule status` offers a blocked battery continuation only what `resolve` accepts, not "replace or
    unblock" (both refused): a continuation whose parent was excluded without a replacement is the group's."""
    make, set_scenarios, _, _ = env
    first, parent = study_a_parent(0), study_a_parent(1)
    row = {"run_id": parent.run_id, "completed": True, "matched": True, "matched_checkpoint_step": 1_800_000,
           "matched_checkpoint_path": "x/epoch-90.pt", "commit_hash": "a" * 40}
    cont = manifest.battery_continuation(parent, row, "finetune")
    set_scenarios(**{first.run_id: "crash", parent.run_id: "crash"})  # the second crash trips the circuit breaker
    sched = make()
    sched.add([first, parent, cont])
    sched.run(max_passes=300)
    assert (sched.row(parent.run_id)["status"], sched.row(parent.run_id)["replaced_by"]) == ("excluded", None)
    assert sched.row(cont.run_id)["status"] == "blocked"
    text = dict(sched.status_report()["decisions"])[cont.run_id]
    assert "amendment (Table 9.1)" in text and "unblock" not in text and "`schedule resolve`" not in text
    for action in ("replace", "unblock"):
        with pytest.raises(SchedulerError, match="amendment"):
            sched.resolve(cont.run_id, action)


def test_a_study_b_run_queued_without_continuations_is_replaced_without_them(env, monkeypatch) -> None:
    """A replacement is continued exactly as the run it replaces (a smoke copy's replacement queues no
    registered-length few-shot continuations); continuations queued after the exclusion follow the replacement."""
    make, set_scenarios, _, _ = env
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset({"Q-seed-collision"}))
    parent, other = study_b_run("Moderate"), study_b_run("Sparse")
    set_scenarios(**{parent.run_id: "crash"})
    sched = make()
    sched.add([parent, other])
    sched.run(max_passes=300)
    replacement = sched.row(parent.run_id)["replaced_by"]
    assert replacement == parent.with_seed(R.NEXT_UNUSED_SEED_START).run_id
    assert not [r for r in sched.rows() if sched.spec(r["run_id"]).group == "study_b_fewshot"]
    # `schedule add --design study_b_fewshot` after the exclusion: the excluded parent's continuations never run
    added = sched.add(manifest.study_b_fewshot([parent, other]))
    assert added == 2 * len(R.UNSEEN_BUDGETS)
    for old in manifest.study_b_fewshot([parent]):
        assert sched.row(old.run_id)["status"] == "superseded"
        assert replacement in [e["detail"] for e in sched.events(old.run_id) if e["event"] == "superseded"][0]
    new = [sched.spec(r["run_id"]) for r in sched.rows() if sched.spec(r["run_id"]).depends_on == (replacement,)]
    assert [s.params["budget"] for s in new] == list(R.UNSEEN_BUDGETS)
    assert new == manifest.study_b_fewshot([sched.spec(replacement)])
    assert all(sched.row(s.run_id)["status"] == "pending" for s in manifest.study_b_fewshot([other]))
    assert sched.add(manifest.study_b_fewshot([parent, other])) == 0  # queued once
    sched.run(max_passes=300)
    assert {sched.row(s.run_id)["status"] for s in new} == {"continued"}


# ---------------------------------------------------------------------------
# Evaluation harnesses by target (pilot.contracts.evaluator_target)
# ---------------------------------------------------------------------------


def test_a_missing_harness_holds_only_the_runs_it_evaluates(env) -> None:
    make, set_scenarios, _, _ = env
    a_run, b1, b2 = study_a_parent(), study_b_run("Moderate"), study_b_run("Sparse")
    set_scenarios(**{b1.run_id: "evalunavailable"})
    sched = make()
    sched.add([a_run, b1, b2])
    sched.run(max_passes=300)
    assert sched.row(a_run.run_id)["status"] == "ledgered"  # Role 2's harness is there
    assert sched.row(b1.run_id)["status"] == "trained" and sched.row(b2.run_id)["status"] == "trained"
    assert "eval_started" not in [e["event"] for e in sched.events(b2.run_id)]  # never tried: its harness is missing
    target = "studyb.evaluation:evaluate_run"
    assert S.contracts.evaluator_target(b1) == target
    detail = [e["detail"] for e in sched.events(b1.run_id) if e["event"] == "evaluator_unavailable"][0]
    assert target in detail and "Role 5" in detail
    report = sched.status_report()
    assert report["evaluators_unavailable"] == {target: [b1.run_id, b2.run_id]}
    assert any(f"evaluation harness {target} is missing: 2 trained runs wait" in line for line in sched.status_lines())
    # another process (`schedule status`) reads it back from the event log, for every run the scheduler holds
    # (not only the run that met the missing harness, with the others listed as ready)
    other = make().status_report()
    assert other["evaluators_unavailable"] == {target: [b1.run_id, b2.run_id]}
    assert other["waiting"] == {"evaluation: evaluator_unavailable": [b1.run_id, b2.run_id]}


def test_status_in_another_process_lists_every_run_held_for_a_missing_plugin(env) -> None:
    """The running scheduler holds every run of a missing plug-in without trying it; `schedule status` (another
    process) says so for each of them, until a new scheduler process tries them again."""
    make, set_scenarios, _, _ = env
    runs = [study_b_run(arm) for arm in ("Single-10", "Single-20", "Single-40")]
    ids = [r.run_id for r in runs]
    set_scenarios(**{r: "unavailable" for r in ids})
    sched = make()
    sched.add(runs)
    sched.run(max_passes=20)
    assert [e["event"] for e in sched.events(ids[1])] == ["added"]  # never tried: its plug-in is missing
    for report in (sched.status_report(), make().status_report()):
        assert report["plugins_unavailable"] == {"study_b": ids}
        assert report["waiting"] == {"plugin_unavailable": ids}
    assert any("plug-in 'study_b'" in line and "3 runs wait" in line for line in make().status_lines())
    # a new scheduler process tries them again: only the run that met the missing plug-in says so (its last event)
    sched._event(None, "scheduler_start", "max_concurrent=1 smoke=True")
    report = make().status_report()
    assert report["plugins_unavailable"] == {"study_b": ids[:1]} and report["waiting"]["ready"] == ids[1:]


def test_the_smoke_launcher_gets_allow_pending_in_both_stages(tmp_path) -> None:
    for stage in ("train", "evaluate"):
        cmd = S.default_command(stage, tmp_path / "spec.json", tmp_path, allow_dirty=False, allow_pending=True)
        assert "--allow-pending" in cmd and "--allow-dirty" not in cmd
        assert "--allow-pending" not in S.default_command(stage, tmp_path / "spec.json", tmp_path, allow_dirty=True,
                                                          allow_pending=False)


# ---------------------------------------------------------------------------
# Launch gates (Scheduler.launch_gates), read from HEAD of a temporary repository
# ---------------------------------------------------------------------------


class Repo:
    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir()
        self.git("init", "-q")
        self.git("config", "user.email", "t@example.com")
        self.git("config", "user.name", "t")
        self.commit({"README.md": "repository\n"})

    def git(self, *args: str) -> str:
        return subprocess.run(["git", *args], cwd=self.root, check=True, capture_output=True, text=True).stdout

    def write(self, files: dict[str, str | bytes]) -> None:
        for rel, content in files.items():
            path = self.root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content if isinstance(content, bytes) else content.encode())

    def commit(self, files: dict[str, str | bytes]) -> str:
        self.write(files)
        self.git("add", *files)
        self.git("commit", "-qm", "c")
        return self.git("rev-parse", "HEAD").strip()


@pytest.fixture
def repo(tmp_path, monkeypatch) -> Repo:
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset(R.PENDING))  # the run gates are not under test here
    return Repo(tmp_path / "repo")


def passing_report(plugin: str, **changes) -> str:
    """A report as scripts/determinism_check.py --registered-form writes it (tests/test_integration_sched_scripts.py
    checks that the script's own report passes the same predicate)."""
    report = {
        "check": S.DETERMINISM_REGISTERED_FORM, "plugin": plugin, "passed": True, "same_config_hash": True,
        "commit_hash": "a" * 40, "worktree_dirty": False, "rehearsal_only": False, "allow_pending": False,
        "run_commits": ["a" * 40, "a" * 40],
        "spec": S.determinism_spec(plugin, R.CHECKPOINT_INTERVAL_STEPS).to_dict(),
        "result": {"checkpoint_step": R.CHECKPOINT_INTERVAL_STEPS, "evaluation_cost_a": 1.0, "evaluation_cost_b": 1.0,
                   "identical": True},
    }
    report.update(changes)
    return json.dumps(report)


def fake_search_module(calls: list | None = None, *, fail: bool = False) -> types.ModuleType:
    """A stand-in for Role 5's studyb.search: complete when the committed decision.json says so."""
    module = types.ModuleType(S.SEARCH_MODULE)

    def check_record(root, read=None):
        if fail:
            raise RuntimeError("malformed record")
        decision = read("decision.json")
        record = read("search_record.csv")
        if calls is not None:
            calls.append((Path(root), decision))
        if decision is None or record is None:
            return types.SimpleNamespace(complete=False, outcome=None, problems=["decision.json or search_record.csv missing"])
        data = json.loads(decision)
        return types.SimpleNamespace(complete=bool(data.get("complete")), outcome=data.get("outcome"), problems=[])

    module.check_record = check_record
    return module


def record(outcome: str = "none", complete: bool = True) -> dict[str, str]:
    return {f"{S.SEARCH_RECORD_DIR}/decision.json": json.dumps({"outcome": outcome, "complete": complete}),
            f"{S.SEARCH_RECORD_DIR}/search_record.csv": "citation,meets_inclusion\n"}


def gates(sched: Scheduler, spec: RunSpec) -> list[str]:
    """The gates holding ``spec`` at a new pass."""
    return list(gate_reasons(sched, spec))


def gate_reasons(sched: Scheduler, spec: RunSpec) -> dict[str, str]:
    """The gates holding ``spec`` at a new pass, with their reasons."""
    sched._refresh_git()  # a new pass: HEAD and the main sweep's progress are read again
    return dict(sched.launch_gates(spec))


def test_analysis_missing_holds_non_pilot_runs_until_head_holds_the_entry_point(env, repo) -> None:
    make, _, _, _ = env
    run = plain_spec(0)
    sched = make(repo_root=repo.root, allow_dirty=False, allow_pending=False)
    sched.add([run])
    sched.run(max_passes=5)
    assert sched.row(run.run_id)["status"] == "pending"
    detail = [e["detail"] for e in sched.events(run.run_id) if e["event"] == "analysis_missing"][0]
    assert S.ANALYSIS_ENTRY in detail and "Part 5.8" in detail
    assert gates(sched, plain_spec(1, "P-A-PointGoal1-N0.00", pilot=True, group="pilot")) == []  # the pilot is not held
    repo.write({S.ANALYSIS_ENTRY: "print('analysis')\n"})  # untracked: not at HEAD
    assert gates(sched, run) == ["analysis_missing"]
    repo.commit({S.ANALYSIS_ENTRY: "print('analysis')\n"})
    assert gates(sched, run) == []
    sched.run(max_passes=300)
    assert sched.row(run.run_id)["status"] == "ledgered"


def test_determinism_missing_needs_a_committed_passing_registered_report_of_that_plugin(env, repo) -> None:
    make, _, _, _ = env
    pilot_a = shortened(next(s for s in manifest.pilot() if s.plugin == "study_a" and s.N == 0.0))
    sched = make(repo_root=repo.root, allow_dirty=False, allow_pending=False, require_allocation_for_pilot=False)
    assert gates(sched, pilot_a) == ["determinism_missing"]
    assert gates(sched, plain_spec(0, "P-A-PointGoal1-N0.00", pilot=True, group="pilot")) == []  # ppolag: not gated
    d = S.DETERMINISM_DIR
    not_enough = {
        f"{d}/other-plugin.json": passing_report("study_b"),
        f"{d}/rehearsal.json": passing_report("study_a", rehearsal_only=True),
        f"{d}/failed.json": passing_report("study_a", passed=False),
        f"{d}/dirty.json": passing_report("study_a", worktree_dirty=True),
        f"{d}/pending.json": passing_report("study_a", allow_pending=True),
        f"{d}/bitwise.json": passing_report("study_a", check="bitwise form (First Tasks, Role 1 step 10)"),
        f"{d}/other-config.json": passing_report("study_a", spec=S.determinism_spec("study_a", 2 * R.STEPS_PER_EPOCH).to_dict()),
        f"{d}/not-identical.json": passing_report("study_a", result={"identical": False}),
        f"{d}/broken.json": "{not json",
    }
    repo.commit(not_enough)
    assert gates(sched, pilot_a) == ["determinism_missing"]
    assert gates(sched, shortened(next(s for s in manifest.pilot() if s.plugin == "study_b"))) == [
        "search_record_missing"]  # study_b's report is there; its search record is not
    repo.write({f"{d}/good.json": passing_report("study_a")})  # uncommitted: not at HEAD
    assert gates(sched, pilot_a) == ["determinism_missing"]
    sched.add([pilot_a])
    sched.run(max_passes=5)
    detail = [e["detail"] for e in sched.events(pilot_a.run_id) if e["event"] == "determinism_missing"][0]
    assert "--plugin study_a" in detail and "Table 3.1" in detail
    repo.commit({f"{d}/good.json": passing_report("study_a")})
    assert gates(sched, pilot_a) == []
    assert S.committed_determinism_reports(repo.git("rev-parse", "HEAD").strip(), repo.root) == {
        "study_a": f"{d}/good.json", "study_b": f"{d}/other-plugin.json"}
    sched.run(max_passes=300)
    assert sched.row(pilot_a.run_id)["status"] == "ledgered"


def test_the_report_predicate_accepts_a_report_without_plugin_field_as_its_spec_says() -> None:
    legacy = json.loads(passing_report("ppolag"))
    del legacy["plugin"], legacy["allow_pending"]
    assert S.determinism_report_plugin(legacy) == "ppolag"
    mislabelled = {**json.loads(passing_report("study_b")), "plugin": "study_a"}
    assert S.determinism_report_plugin(mislabelled) is None  # its spec is study_b's
    assert S.determinism_report_plugin(None) is None and S.determinism_report_plugin([]) is None
    # both runs must have trained on the commit the report names
    for commits in (None, ["a" * 40, "b" * 40], ["b" * 40, "b" * 40]):
        assert S.determinism_report_plugin(json.loads(passing_report("study_a", run_commits=commits))) is None
    assert S.determinism_report_plugin(json.loads(passing_report("study_a", commit_hash=None))) is None


def test_search_record_missing_reads_only_the_committed_record(env, repo, monkeypatch) -> None:
    make, _, _, _ = env
    calls: list = []
    monkeypatch.setitem(sys.modules, S.SEARCH_MODULE, fake_search_module(calls))
    sched = make(repo_root=repo.root, allow_dirty=False, allow_pending=False)
    unconstrained = shortened(next(s for s in manifest.pilot() if s.plugin == "unconstrained_ppo"))
    fewshot = manifest.study_b_fewshot([study_b_run()])[0]
    repo.commit({f"{S.DETERMINISM_DIR}/u.json": passing_report("unconstrained_ppo")})
    for spec in (unconstrained, fewshot):
        assert "search_record_missing" in gates(sched, spec)
    assert gates(sched, study_a_parent()) == ["determinism_missing", "analysis_missing"]  # Study A is not held by it
    repo.write(record())  # untracked: the working tree does not count
    assert gates(sched, unconstrained) == ["search_record_missing"]
    assert calls[-1] == (repo.root / S.SEARCH_RECORD_DIR, None)
    repo.commit(record(complete=False))
    reasons = gate_reasons(sched, unconstrained)
    assert "search_record_missing" in reasons and "not complete" in reasons["search_record_missing"]
    repo.commit(record(outcome="included"))
    reasons = gate_reasons(sched, unconstrained)
    assert "search_record_missing" in reasons and "withdrawn by amendment" in reasons["search_record_missing"]
    head = repo.commit(record(outcome="none"))
    assert gates(sched, unconstrained) == [] and gates(sched, fewshot) == ["analysis_missing"]
    # the facts are read once per HEAD
    n = len(calls)
    assert gates(sched, unconstrained) == [] and len(calls) == n
    assert S.search_record_status(head, repo.root)[0] is True


def test_search_record_missing_when_the_checker_cannot_be_used(repo, monkeypatch) -> None:
    head = repo.commit(record())
    monkeypatch.setitem(sys.modules, S.SEARCH_MODULE, None)  # import fails: Role 5's code not written yet
    released, reason = S.search_record_status(head, repo.root)
    assert not released and "ModuleNotFoundError" in reason
    monkeypatch.setitem(sys.modules, S.SEARCH_MODULE, fake_search_module(fail=True))
    released, reason = S.search_record_status(head, repo.root)
    assert not released and "malformed record" in reason
    # the checker itself must be the committed code
    module = fake_search_module()
    module.__file__ = str(repo.root / "studyb" / "search" / "__init__.py")
    repo.write({"studyb/search/__init__.py": "# uncommitted checker\n"})
    monkeypatch.setitem(sys.modules, S.SEARCH_MODULE, module)
    released, reason = S.search_record_status(head, repo.root)
    assert not released and "studyb/search/__init__.py" in reason
    head = repo.commit({"studyb/search/__init__.py": "# uncommitted checker\n"})
    assert S.search_record_status(head, repo.root)[0] is True


def test_a_checker_changed_at_head_after_it_was_loaded_holds_until_the_scheduler_restarts(repo, monkeypatch) -> None:
    """The gate compares HEAD's checker with the source as the process loaded it (noted at import), not with the
    file on disk, so after a pull that changes the committed checker a running scheduler does not judge the
    record with its stale code in memory."""
    checker = "studyb/search/__init__.py"
    v1 = "def check_record(root, read=None):\n    return 'v1'\n"
    head = repo.commit({**record(), checker: v1})
    loaded = fake_search_module()
    loaded.__file__ = str(repo.root / checker)
    monkeypatch.setitem(sys.modules, S.SEARCH_MODULE, loaded)
    assert S.search_record_status(head, repo.root)[0] is True  # loaded as HEAD holds it
    head = repo.commit({checker: v1.replace("v1", "v2")})  # a pull while the scheduler runs: disk == HEAD now
    released, reason = S.search_record_status(head, repo.root)
    assert not released and checker in reason and "restart the scheduler" in reason
    assert S._uncommitted_modules("studyb", head, repo.root) == [checker]
    restarted = fake_search_module()  # a new process loads HEAD's code
    restarted.__file__ = str(repo.root / checker)
    monkeypatch.setitem(sys.modules, S.SEARCH_MODULE, restarted)
    assert S.search_record_status(head, repo.root)[0] is True


def test_a_checker_package_loaded_at_one_head_is_caught_at_the_next(repo, monkeypatch) -> None:
    """The same with a real package imported from the repository, through the scheduler's import path."""
    import importlib
    import uuid

    name = f"wscl_checker_{uuid.uuid4().hex[:12]}"  # a package no other test imports
    source = "def check_record(root, read=None):\n    return {!r}\n"
    head = repo.commit({f"{name}/__init__.py": "", f"{name}/search.py": source.format("v1")})
    monkeypatch.syspath_prepend(str(repo.root))
    monkeypatch.setattr(S, "SEARCH_MODULE", f"{name}.search")
    try:
        module = S._import_search_module()
        assert module.check_record(None) == "v1" and S._uncommitted_modules(name, head, repo.root) == []
        head = repo.commit({f"{name}/search.py": source.format("v2")})
        assert importlib.import_module(f"{name}.search").check_record(None) == "v1"  # still loaded
        assert S._uncommitted_modules(name, head, repo.root) == [f"{name}/search.py"]
    finally:
        for key in [k for k in sys.modules if k == name or k.startswith(name + ".")]:
            del sys.modules[key]


def test_a_module_loaded_by_a_failed_import_of_the_checker_is_caught_at_the_next_head(repo, monkeypatch) -> None:
    """The sources are noted at a failed import of the checker too: a submodule loaded before the failure stays
    in sys.modules and is reused by the next import, so noting HEAD's file on disk then would let the gate release
    Study B with stale code in memory, a checker that is not HEAD's."""
    import uuid

    name = f"wscl_failing_{uuid.uuid4().hex[:12]}"  # a package no other test imports
    search = f"{name}/search"
    check = ("from types import SimpleNamespace\nfrom {0}.search import protocol\n"
             "def check_record(root, read=None):\n    missing = [n for n in protocol.REQUIRED if read(n) is None]\n"
             "    return SimpleNamespace(complete=not missing, outcome='none', problems=missing)\n").format(name)
    head = repo.commit({f"{name}/__init__.py": "",
                        f"{search}/__init__.py": f"from {name}.search import protocol\nfrom {name}.search.record import "
                                                 "check_record\n",
                        f"{search}/protocol.py": "REQUIRED = []\n",  # v1: nothing required
                        f"{search}/record.py": "raise RuntimeError('v1 is broken')\n"})
    monkeypatch.syspath_prepend(str(repo.root))
    monkeypatch.setattr(S, "SEARCH_MODULE", f"{name}.search")
    try:
        released, reason = S.search_record_status(head, repo.root)
        assert not released and "v1 is broken" in reason
        assert f"{name}.search.protocol" in sys.modules  # loaded before the failure, and kept
        # the fix: record.py works, and protocol.py now requires the decision (HEAD commits none)
        head = repo.commit({f"{search}/record.py": check, f"{search}/protocol.py": "REQUIRED = ['decision.json']\n"})
        released, reason = S.search_record_status(head, repo.root)
        assert sys.modules[f"{name}.search.protocol"].REQUIRED == []  # the stale v1 is what the checker would use
        assert not released and f"{search}/protocol.py" in reason and "restart the scheduler" in reason
        assert S._uncommitted_modules(name, head, repo.root) == [f"{search}/protocol.py"]
    finally:
        for key in [k for k in sys.modules if k == name or k.startswith(name + ".")]:
            del sys.modules[key]


def test_role_5s_checker_reads_the_committed_template_as_incomplete(repo) -> None:
    """The real studyb.search.check_record through the gate's reader: its empty template fails
    (studyb/search/README.md). The template is written here, not read from the working tree, whose record the
    searcher fills in and commits complete before the pilot's Study B runs."""
    search = pytest.importorskip(S.SEARCH_MODULE)
    d = S.SEARCH_RECORD_DIR
    template = {  # the record's files as Role 5 commits them before the search: headers and placeholders only
        f"{d}/search_log.csv": "source,query_id,query_string,date,searcher,results_reported,results_screened,"
                               "export_file,notes\n",
        f"{d}/citations_log.csv": "list_id,kind,of_work,date,searcher,items_reported,items_screened,export_file,notes\n",
        f"{d}/search_record.csv": "citation,known_work,what_it_varies,what_it_evaluates,meets_inclusion,reason,"
                                  "exclusion_criterion,found_in,date,searcher\n",
        f"{d}/search_log.md": "# Targeted search for Study B: search log\n\n"
                              "Screening depth: (to be fixed before the first query)\n"
                              "Result order: (relevance; fixed before the first query)\n"
                              "Depth fixed on: (YYYY-MM-DD)\nFixed by: (the searcher's name)\n",
    }
    head = repo.commit(template)
    root = repo.root / S.SEARCH_RECORD_DIR
    read_paths: list[str] = []
    reader = S.committed_reader(head, root, repo.root)

    def spy(path):
        read_paths.append(str(path))
        return reader(path)

    status = search.check_record(root, read=spy)
    assert status.complete is False and status.problems
    assert read_paths and all(reader(p) is None or isinstance(reader(p), bytes) for p in read_paths)
    assert any(reader(p) is not None for p in read_paths)  # it found the committed files through the reader
    released, reason = S.search_record_status(head, repo.root)
    assert not released and "not complete" in reason


def test_the_committed_reader_reads_head_in_every_path_form(repo) -> None:
    head = repo.commit({"studyb/search/decision.json": "committed", "outside.txt": "x"})
    repo.write({"studyb/search/decision.json": "working copy", "studyb/search/new.csv": "untracked"})
    root = repo.root / "studyb" / "search"
    read = S.committed_reader(head, root, repo.root)
    assert read("decision.json") == b"committed"  # relative to the record's root
    assert read("studyb/search/decision.json") == b"committed"  # relative to the repository
    assert read(root / "decision.json") == b"committed"  # absolute
    assert read("new.csv") is None  # not committed
    assert read("../../outside.txt") == b"x"  # inside the repository
    for escape in ("../../../etc/passwd", "/etc/passwd"):
        assert read(escape) is None


def test_the_committed_reader_reads_a_committed_directory_as_absent(repo) -> None:
    """``git show <commit>:<dir>`` exits 0 with a tree listing; the committed reader reads a directory as absent,
    so a search_log.csv row whose export_file names a directory of the record (``screened``) is not a committed
    raw export, as ``python -m studyb.search check`` (IsADirectoryError) also judges it."""
    head = repo.commit({"studyb/search/screened/a.csv": "x"})
    read = S.committed_reader(head, repo.root / "studyb" / "search", repo.root)
    assert read("screened/a.csv") == b"x"
    assert read("screened") is None and read("screened/") is None
    assert S.provenance.file_committed_at(head, "studyb/search/screened", repo.root) is None
    assert S.provenance.file_committed_at(head, "studyb/search/screened/a.csv", repo.root) == b"x"


def main_sweep() -> list[RunSpec]:
    """Every run of the main sweep (manifest.design("main")); the first (N = 0, no onset) shortened for the fake
    launcher, which trains only that one (the gate reads run_ids and statuses, never the runs' lengths)."""
    sweep = manifest.design("main")
    assert sweep[0].onset_step == 0
    return [shortened(sweep[0]), *sweep[1:]]


def set_status(sched: Scheduler, status: str, run_ids) -> None:
    with sched.db:
        sched.db.executemany("UPDATE runs SET status = ? WHERE run_id = ?", [(status, r) for r in run_ids])


def test_main_sweep_unfinished_holds_study_b_while_a_main_run_trains(env, repo, monkeypatch) -> None:
    make, set_scenarios, _, tmp = env
    sweep, moderate = main_sweep(), study_b_run()
    main_run = sweep[0]
    sched = make(repo_root=repo.root, allow_dirty=False, allow_pending=False, max_concurrent=2)
    sched.add([moderate, *sweep])
    set_status(sched, "ledgered", [s.run_id for s in sweep])
    assert "main_sweep_unfinished" not in gates(sched, moderate)
    for status in S.UNFINISHED_TRAINING:
        set_status(sched, status, [main_run.run_id])
        assert "main_sweep_unfinished" in gates(sched, moderate), status
    for status in ("trained", "evaluated", "ledgered", "excluded", "eval_failed"):
        set_status(sched, status, [main_run.run_id])
        assert "main_sweep_unfinished" not in gates(sched, moderate), status
    pilot_moderate = study_b_run(pilot=True)
    assert "main_sweep_unfinished" not in gates(sched, pilot_moderate)
    assert "main_sweep_unfinished" not in gates(sched, manifest.study_b_fewshot([moderate])[0])
    # a replacement or surplus seed of a main arm belongs to the main sweep too
    sched.add([main_run.with_seed(R.NEXT_UNUSED_SEED_START)])
    assert "main_sweep_unfinished" in gates(sched, moderate)
    set_status(sched, "ledgered", [main_run.with_seed(R.NEXT_UNUSED_SEED_START).run_id])
    # end to end, with every other gate satisfied
    set_status(sched, "pending", [main_run.run_id])
    set_scenarios(**{main_run.run_id: "hang"})
    d = S.DETERMINISM_DIR
    repo.commit({S.ANALYSIS_ENTRY: "", f"{d}/b.json": passing_report("study_b"), f"{d}/a.json": passing_report("study_a"),
                 **record()})
    monkeypatch.setitem(sys.modules, S.SEARCH_MODULE, fake_search_module())
    sched.run(max_passes=3)
    assert sched.row(main_run.run_id)["status"] == "training"
    assert sched.row(moderate.run_id)["status"] == "pending"
    assert "main_sweep_unfinished" in [e["event"] for e in sched.events(moderate.run_id)]
    (tmp / "release").write_text("go")
    sched.run(max_passes=400)
    assert sched.row(main_run.run_id)["status"] == "ledgered" and sched.row(moderate.run_id)["status"] == "ledgered"


def test_main_sweep_unfinished_holds_study_b_queued_before_the_main_sweep(env, repo) -> None:
    """Study B queued before the main sweep (or on a data root without it) waits: every run of the main sweep
    must be queued and past training, not only the runs already queued."""
    make, _, _, _ = env
    sweep, moderate = main_sweep(), study_b_run()
    sched = make(repo_root=repo.root, allow_dirty=False, allow_pending=False)
    sched.add([moderate])
    reasons = gate_reasons(sched, moderate)
    assert "main_sweep_unfinished" in reasons
    reason = reasons["main_sweep_unfinished"]
    assert f"{len(sweep)} main-sweep runs are not queued on this data root yet (e.g. {sweep[0].run_id}" in reason
    assert "have not finished training" not in reason and "Q-studyb-order" in reason
    sched.add(sweep[:-1])
    set_status(sched, "ledgered", [s.run_id for s in sweep[:-1]])
    reasons = gate_reasons(sched, moderate)
    assert "main_sweep_unfinished" in reasons
    reason = reasons["main_sweep_unfinished"]
    assert f"1 main-sweep runs are not queued on this data root yet (e.g. {sweep[-1].run_id}" in reason
    sched.add(sweep[-1:])
    assert "main_sweep_unfinished" in gates(sched, moderate)  # queued, not trained yet
    assert "have not finished training (e.g. " + sweep[-1].run_id in dict(sched.launch_gates(moderate))[
        "main_sweep_unfinished"]
    set_status(sched, "trained", [sweep[-1].run_id])
    assert "main_sweep_unfinished" not in gates(sched, moderate)
    report = sched.status_report()
    assert "main_sweep_unfinished" not in [g["gate"] for g in report["gates"]]


def test_main_sweep_unfinished_follows_the_data_roots_cuts(env, repo) -> None:
    """The main sweep of a cut data root (Part 6.1; pilot.scheduler.CUTS_PATH) is the cut design: the cut runs are never queued."""
    make, _, _, _ = env
    repo.commit(cuts_file(["pid", "car"]))
    moderate = study_b_run()
    sched = make(repo_root=repo.root, allow_dirty=False, allow_pending=False)
    sched.add([moderate], cuts=2)
    kept = manifest.apply_cuts(main_sweep(), ("pid", "car"))
    assert len(kept) < len(main_sweep()) and not [s for s in kept if s.task in manifest.CAR_TASKS]
    sched.add(main_sweep())  # cut on the way in, as every add of this data root
    assert sorted(r["run_id"] for r in sched.rows() if sched.spec(r["run_id"]).group == "main") == sorted(
        s.run_id for s in kept)
    set_status(sched, "ledgered", [s.run_id for s in kept])
    assert "main_sweep_unfinished" not in gates(sched, moderate)


def test_no_launch_gate_applies_in_smoke_mode(env, repo) -> None:
    make, _, _, _ = env
    sched = make(repo_root=repo.root)  # smoke: allow_dirty and allow_pending
    for spec in (plain_spec(0), study_a_parent(), study_b_run(), study_b_run(pilot=True)):
        assert sched.launch_gates(spec) == []
    sched.add([study_a_parent()])
    sched.run(max_passes=300)
    assert sched.row(study_a_parent().run_id)["status"] == "ledgered"
    assert "launch gates: none in smoke mode" in sched.status_lines()


def test_the_pid_check_releases_no_run_while_its_changed_onset_waits_on_its_key(env, repo, monkeypatch) -> None:
    """The study_a_pid check runs a changed onset (DETERMINISM_LATE_ONSET, the rule of Q-determinism-late-onset), so
    while that key is open (reopened here) a committed passing report of it releases no PID-check run."""
    make, _, _, _ = env
    # (the registered length: launch_gates only reads the spec, and a shortened run would lose its onset)
    pid = next(s for s in manifest.study_a_pid((R.SEEDS[0],)) if s.N == max(R.LATE_ONSET_FRACTIONS))
    d, key = S.DETERMINISM_DIR, S.DETERMINISM_LATE_ONSET_KEY
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset(R.PENDING) - {key})  # every key but the check's own
    repo.commit({f"{d}/pid.json": passing_report("study_a_pid"), S.ANALYSIS_ENTRY: "print('analysis')\n"})
    sched = make(repo_root=repo.root, allow_dirty=False, allow_pending=False)
    assert S.determinism_report_plugin(json.loads(passing_report("study_a_pid"))) == "study_a_pid"
    held = dict(sched.launch_gates(pid))
    assert list(held) == ["determinism_missing"]  # the report is at HEAD, the check's own key is open
    assert key in held["determinism_missing"] and "--plugin study_a_pid" in held["determinism_missing"]
    assert gates(sched, pid) == ["determinism_missing"]
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset({key}))
    assert gates(sched, pid) == []  # answered: the committed report of that configuration releases the run
    assert gates(sched, shortened(next(s for s in manifest.pilot() if s.plugin == "study_a" and s.N == 0.0))) == [
        "determinism_missing"]  # another plug-in's runs are not released by it


def test_status_shows_each_gate_with_its_reason(env, repo, monkeypatch) -> None:
    make, _, _, _ = env
    monkeypatch.setitem(sys.modules, S.SEARCH_MODULE, None)
    sched = make(repo_root=repo.root, allow_dirty=False, allow_pending=False)
    sched.add([study_a_parent(), plain_spec(1), study_b_run()])
    report = sched.status_report()
    assert [g["gate"] for g in report["gates"]] == [
        "search_record_missing", "main_sweep_unfinished", "determinism_missing", "determinism_missing", "analysis_missing"]
    shown = {(g["gate"], g["reason"]): g["runs"] for g in report["gates"]}
    reasons = {gate: reason for gate, reason in shown}
    assert shown["analysis_missing", reasons["analysis_missing"]] == [
        study_a_parent().run_id, plain_spec(1).run_id, study_b_run().run_id]
    assert shown["main_sweep_unfinished", reasons["main_sweep_unfinished"]] == [study_b_run().run_id]
    assert "Q-studyb-order" in reasons["main_sweep_unfinished"]
    assert "ModuleNotFoundError" in reasons["search_record_missing"]
    # determinism: one entry per plug-in, each naming its own
    determinism = {reason: runs for (gate, reason), runs in shown.items() if gate == "determinism_missing"}
    assert sorted(determinism.values()) == [[study_a_parent().run_id], [study_b_run().run_id]]
    for reason, runs in determinism.items():
        assert f"--plugin {sched.spec(runs[0]).plugin}" in reason
    assert report["waiting"]["search_record_missing"] == [study_b_run().run_id]
    lines = sched.status_lines()
    assert "launch gates:" in lines
    assert any(line.strip().startswith("analysis_missing") and "holds 3 runs" in line for line in lines)


def test_status_judges_a_smoke_data_root_as_its_scheduler_does(env, monkeypatch, capsys) -> None:
    """`python -m pilot schedule status` (no smoke flags) judges a smoke data root's runs as its smoke scheduler
    does: it does not list pilot/scheduler.py's gates ("not in smoke mode") or open questions as holding them."""
    make, _, _, tmp = env
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset())
    runs = [study_b_run(pilot=True), study_a_parent()]
    smoke = make()  # --allow-dirty --allow-pending, as scripts/smoke_run.py
    smoke.fix_mode()
    smoke.add(runs)
    assert all(r.open_questions for r in runs)
    status = make(allow_dirty=False, allow_pending=False)  # the status command's own (registered) flags
    report = status.status_report()
    assert (report["mode"], report["config_mode"], report["judged_mode"]) == ("smoke", "registered", "smoke")
    assert report["gates"] == [] and report["waiting"] == {"ready": [r.run_id for r in runs]}
    assert "no scheduler has run it yet" in report["judged"]
    lines = status.status_lines(events=0)
    assert "launch gates: none in smoke mode" in lines
    assert any(line.startswith("note: this data root holds smoke runs; the reasons below are judged in smoke mode")
               for line in lines)
    # the flags of the last `schedule run` on the data root: without --allow-pending its open questions hold
    dirty_only = make(allow_pending=False)
    monkeypatch.setattr(S.Scheduler, "step", lambda self: None)  # record the flags, launch nothing
    dirty_only.run(once=True)
    report = status.status_report()
    assert report["waiting"] == {"open_questions": [r.run_id for r in runs]} and report["gates"] == []
    assert report["judged"] == "judged in smoke mode with the flags of its last `schedule run` (--allow-dirty)"
    # the command line's `schedule status`, which takes no smoke flags
    from pilot.__main__ import main

    assert main(["schedule", "status", "--data-root", str(tmp / "data"), "--events", "0"]) == 0
    out = capsys.readouterr().out
    assert "launch gates: none in smoke mode" in out
    assert not any(gate in out for gate in S.LAUNCH_GATES)


def test_status_judges_a_registered_data_root_as_registered(env, repo) -> None:
    """The converse: a command with smoke flags judges a registered data root's runs as registered."""
    make, _, _, _ = env
    registered = make(repo_root=repo.root, allow_dirty=False, allow_pending=False)
    registered.fix_mode()
    registered.add([plain_spec(0)])
    report = make(repo_root=repo.root).status_report()  # smoke flags
    assert (report["mode"], report["config_mode"], report["judged_mode"]) == ("registered", "smoke", "registered")
    assert [g["gate"] for g in report["gates"]] == ["analysis_missing"]  # HEAD of the fixture holds no analysis
    assert report["judged"].startswith("judged as registered")
    assert report == {**registered.status_report(), "config_mode": "smoke", "judged": report["judged"]}


# ---------------------------------------------------------------------------
# The cut order of Part 6.1, from the committed pilot/cuts.json
# ---------------------------------------------------------------------------


def cuts_file(cuts: list[str], amendment: str = "Table 9.1 row 3") -> dict[str, str]:
    return {S.CUTS_PATH: json.dumps({"cuts": cuts, "amendment": amendment})}


def test_add_applies_the_first_k_committed_cuts(env, repo) -> None:
    make, _, _, _ = env
    repo.commit(cuts_file(["pid", "car"]))
    specs = manifest.design("pid") + manifest.design("main")
    sched = make(repo_root=repo.root)
    with pytest.raises(SchedulerError, match="exceeds the 2 cuts"):
        sched.add(specs, cuts=3)
    expected = manifest.apply_cuts(specs, ("pid", "car"))
    assert sched.add(specs, cuts=2) == len(expected)
    queued = [sched.spec(r["run_id"]) for r in sched.rows()]
    assert queued == expected
    assert not [s for s in queued if s.group == "pid" or s.task in manifest.CAR_TASKS]
    assert sched.setting("cuts") == "2"
    # fixed per data root: later adds are cut the same way, and the count never changes
    before = len(sched.rows())
    sched.add(manifest.design("controller"))
    assert not [r for r in sched.rows()[before:] if sched.spec(r["run_id"]).task in manifest.CAR_TASKS]
    for other in (1, 0):
        with pytest.raises(SchedulerError, match="first 2 cuts"):
            sched.add([], cuts=other)
    assert sched.add(manifest.design("main"), cuts=2) == 0  # the same cuts again: nothing new


def test_cuts_are_refused_unless_committed_as_they_are(env, repo) -> None:
    make, _, _, _ = env
    sched = make(repo_root=repo.root)
    repo.write(cuts_file(["pid"]))  # untracked
    with pytest.raises(SchedulerError, match="not committed at HEAD"):
        sched.add(manifest.design("pid"), cuts=1)
    repo.commit(cuts_file(["pid"]))
    repo.write(cuts_file(["pid", "car"]))  # an edit not committed
    with pytest.raises(SchedulerError, match="uncommitted changes"):
        sched.add(manifest.design("pid"), cuts=1)
    repo.commit(cuts_file(["car"]))
    with pytest.raises(SchedulerError, match="not a prefix"):
        sched.add(manifest.design("pid"), cuts=1)
    repo.commit(cuts_file(["pid"], amendment=" "))
    with pytest.raises(SchedulerError, match="amendment"):
        sched.add(manifest.design("pid"), cuts=1)
    repo.commit({S.CUTS_PATH: "[1, 2]"})
    with pytest.raises(SchedulerError, match="must hold"):
        sched.add(manifest.design("pid"), cuts=1)
    assert sched.rows() == [] and sched.setting("cuts") is None
    with pytest.raises(SchedulerError, match="whole number"):
        S.committed_cuts(-1, repo.root)


def test_cuts_are_refused_when_queued_runs_would_be_cut(env, repo) -> None:
    make, _, _, _ = env
    repo.commit(cuts_file(["pid"]))
    sched = make(repo_root=repo.root)
    sched.add(manifest.design("pid"))
    with pytest.raises(SchedulerError, match="already holds 15 runs"):
        sched.add(manifest.design("main"), cuts=1)
    assert sched.setting("cuts") is None


def test_the_fewshot_cut_follows_surplus_seeds_and_replacements(env, repo, monkeypatch) -> None:
    make, set_scenarios, _, _ = env
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset({"Q-seed-collision"}))
    repo.commit(cuts_file(["pid", "car", "fewshot_short"]))
    parent = study_b_run()
    sched = make(repo_root=repo.root)
    sched.add([parent, *manifest.study_b_fewshot([parent])], cuts=3)
    first = R.FEWSHOT_HORIZONS[0]
    fewshots = [sched.spec(r["run_id"]) for r in sched.rows() if sched.spec(r["run_id"]).group == "study_b_fewshot"]
    assert len(fewshots) == len(R.UNSEEN_BUDGETS)
    assert {(s.total_steps, tuple(s.params["horizons"])) for s in fewshots} == {(first, (first,))}
    sched.add_surplus(1)
    surplus = [sched.spec(r["run_id"]) for r in sched.rows() if r["surplus"] and sched.spec(r["run_id"]).group == "study_b_fewshot"]
    assert surplus and {s.total_steps for s in surplus} == {first}
    set_scenarios(**{parent.run_id: "crash"})
    sched.run(max_passes=3)
    replacement = sched.row(parent.run_id)["replaced_by"]
    assert replacement
    conts = [sched.spec(r["run_id"]) for r in sched.rows() if replacement in sched.spec(r["run_id"]).depends_on]
    assert conts and {s.total_steps for s in conts} == {first}

"""The run scheduler: launch, resume, never duplicate, exclusion and replacement (Part 5.6), surplus seeds
(Part 5.5), modes and the command line, launch gates, the interruption policy, the operator's decisions
(requeue included) and the checks of the code that writes the ledger."""

from __future__ import annotations

import json
import sqlite3
import sys
import time
import types
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable

import pytest

from configs import registered as R
from pilot.errors import PendingQuestionError
from pilot.ledger_writer import LedgerPaths
from pilot.manifest import RunSpec
from pilot.scheduler import AUXILIARY_EXCLUDED, Scheduler, SchedulerConfig, SchedulerError

if TYPE_CHECKING:
    from results.ledger_schema import LedgerRow

FAKE = Path(__file__).with_name("fake_launcher.py")
BASE = "T-A-PointGoal1-N0.00"  # the N = 0 arm of the test specs; its run ids are f"{BASE}-s{seed}"
PILOT_BASE = "P-A-PointGoal1-N0.00"  # the same arm in the pilot
S0, S1, S5, S6 = (f"{BASE}-s{s}" for s in (0, 1, 5, 6))
WARM = "T-A-PointGoal1-N0.50-abrupt-total-warm_started"  # a warm-started run (controller_variant="warm_started")
# A run that depends on the N = 0 run of its seed and is neither warm-started nor a continuation (``spec`` sets no
# controller_variant): no such run is in the design, so the tests of the generic dependency rules use it
DEPENDENT = "T-A-PointGoal1-N0.50-abrupt-total-dependent"


def spec(seed: int = 0, name: str = BASE, pilot: bool = False, depends: Iterable[str] = (),
         group: str = "main") -> RunSpec:
    return RunSpec(
        run_id=f"{name}-s{seed}", study="A", task="SafetyPointGoal1-v0", arm="N0.00", seed=seed,
        total_steps=2_000_000, base_algo="PPOLag", plugin="ppolag", group=group, pilot=pilot,
        N=0.0, onset_step=0, depends_on=tuple(depends),
    )


@pytest.fixture
def analysis_at_head(monkeypatch) -> None:
    """HEAD holds the analysis entry point, so the analysis_missing gate (pilot/scheduler.py) releases non-pilot runs
    in the registered-mode tests that are about something else."""
    import pilot.scheduler as S

    real = S.provenance.file_committed_at

    def committed(commit, rel_path, *a, **kw):
        return b"# analysis entry point\n" if rel_path == S.ANALYSIS_ENTRY else real(commit, rel_path, *a, **kw)

    monkeypatch.setattr(S.provenance, "file_committed_at", committed)


@pytest.fixture
def env(tmp_path, monkeypatch) -> tuple[Callable[..., Scheduler], Callable[..., None], LedgerPaths, Path]:
    """(make, set_scenarios, paths, tmp_path): ``make(**cfg)`` builds a Scheduler on ``tmp_path/data`` (smoke
    mode, allow_dirty, unless a test passes other flags) whose launcher is tests/fake_launcher.py;
    ``set_scenarios(RUN_ID=scenario)`` tells the fake launcher how each run ends (``__`` stands for ``-``);
    ``paths`` are the test's ledgers."""
    scenarios: dict[str, str] = {}
    scen_file = tmp_path / "scenarios.json"

    def set_scenarios(**kw):
        scenarios.update({k.replace("__", "-"): v for k, v in kw.items()})
        scen_file.write_text(json.dumps(scenarios), encoding="utf-8")

    set_scenarios()
    monkeypatch.setenv("FAKE_SCENARIOS", str(scen_file))
    monkeypatch.setenv("FAKE_RELEASE", str(tmp_path / "release"))
    paths = LedgerPaths(main=tmp_path / "ledger.parquet", pilot=tmp_path / "pilot.parquet",
                        sidecar_dir=tmp_path / "sidecar")

    def make(**kw) -> Scheduler:
        cfg = SchedulerConfig(
            data_root=kw.pop("data_root", tmp_path / "data"), poll_seconds=0.05,
            ledger_paths=kw.pop("ledger_paths", paths),
            command_builder=lambda stage, spec_path, run_dir: [sys.executable, str(FAKE), stage, str(spec_path),
                                                               str(run_dir)],
            **{"allow_dirty": True, **kw},  # smoke mode unless a test asks for the registered gates
        )
        return Scheduler(cfg)

    return make, set_scenarios, paths, tmp_path


def ledger_rows(path: Path) -> list[LedgerRow]:
    """The rows of the ledger at ``path``."""
    from results.ledger_schema import load_ledger_as_rows

    return load_ledger_as_rows(path)


def run_to_end(sched: Scheduler, passes: int = 400) -> None:
    """Run the scheduler until it is idle with nothing left to launch (at most ``passes`` passes)."""
    sched.run(max_passes=passes)


def clean_tree(monkeypatch, commit: str = "c" * 40) -> None:
    """The command line's queueing actions run from a clean, committed tree (as on the workstation)."""
    from pilot import provenance

    monkeypatch.setattr(provenance, "dirty_paths", lambda *a, **kw: [])
    monkeypatch.setattr(provenance, "commit_hash", lambda *a, **kw: commit)
    monkeypatch.setattr(provenance, "unverified_imported_code", lambda *a, **kw: [])


def clean_git_repo(root: Path) -> str:
    """A stand-in repository with one committed tracked file; returns HEAD."""
    import subprocess

    root.mkdir()

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout.strip()

    git("init", "-q")
    (root / "code.py").write_text("x = 1\n", encoding="utf-8")
    git("add", ".")
    git("-c", "user.email=t@example.org", "-c", "user.name=t", "commit", "-qm", "c")
    return git("rev-parse", "HEAD")


def with_new_gate(monkeypatch, run_id: str, key: str = "Q-cost-critic") -> None:
    """The code checked out now derives one more run gate for ``run_id`` than when it was queued."""
    from pilot import manifest

    real = manifest.run_gate_keys
    monkeypatch.setattr(manifest, "run_gate_keys",
                        lambda spec: (*real(spec), key) if spec.run_id == run_id else real(spec))


# -- launch, ledger, exclusion, interruption and dependencies ----------------------------------------


def test_success_goes_to_ledger_once(env) -> None:
    make, _, paths, _ = env
    sched = make(max_concurrent=2)
    sched.add([spec(0), spec(1)])
    run_to_end(sched)
    assert sched.summary() == {"ledgered": 2}
    rows = ledger_rows(paths.main)
    assert sorted(r.run_id for r in rows) == [S0, S1]
    row = rows[0]
    assert row.completed and row.final_cost == 24.0
    assert [c.step for c in row.checkpoints] == list(range(0, 2_000_001, 200_000))
    window = [c for c in row.checkpoints if c.selection_cost is not None]
    assert [c.step for c in window] == list(range(200_000, 2_000_001, 200_000))
    assert row.checkpoints[0].multiplier == pytest.approx(0.001)  # initial value, Table A.1
    assert row.wall_clock_hours == pytest.approx(1.25)  # training + evaluation
    # adding the same runs again changes nothing, and running again launches nothing
    assert sched.add([spec(0), spec(1)]) == 0
    run_to_end(sched)
    assert len(ledger_rows(paths.main)) == 2


def test_pilot_runs_go_to_the_pilot_ledger(env) -> None:
    make, _, paths, _ = env
    sched = make()
    sched.add([spec(0, name=PILOT_BASE, pilot=True, group="pilot")])
    run_to_end(sched)
    assert not paths.main.exists()
    assert [r.run_id for r in ledger_rows(paths.pilot)] == [f"{PILOT_BASE}-s0"]


def test_crash_is_excluded_recorded_and_replaced_with_next_unused_seed(env) -> None:
    make, set_scenarios, paths, _ = env
    set_scenarios(**{S1: "crash"})
    sched = make(max_concurrent=3)
    sched.add([spec(0), spec(1), spec(2)])
    run_to_end(sched)
    assert sched.row(S1)["status"] == "excluded"
    assert sched.row(S1)["replaced_by"] == S5
    assert sched.row(S5)["replaces"] == S1
    rows = {r.run_id: r for r in ledger_rows(paths.main)}
    assert rows[S1].completed is False
    assert rows[S1].failure_cause == "crash"
    assert rows[S5].completed is True


def test_non_finite_multiplier_is_excluded(env) -> None:
    make, set_scenarios, paths, _ = env
    set_scenarios(**{S0: "nonfinite"})
    sched = make()
    sched.add([spec(0)])
    run_to_end(sched)
    row = {r.run_id: r for r in ledger_rows(paths.main)}[S0]
    assert row.failure_cause == "non_finite_multiplier"
    assert row.checkpoints[-1].multiplier is None  # NaN is not written as a multiplier value


def test_second_replacement_with_another_cause_takes_the_next_seed(env) -> None:
    make, set_scenarios, _, _ = env
    set_scenarios(**{S0: "crash", S5: "nonfinite"})
    sched = make()
    sched.add([spec(0)])
    run_to_end(sched)
    assert sched.row(S5)["replaced_by"] == S6
    assert sched.row(S6)["status"] == "ledgered"


def test_repeated_same_cause_stops_replacing(env) -> None:
    make, set_scenarios, paths, _ = env
    set_scenarios(**{S0: "crash", S5: "crash"})
    sched = make()
    sched.add([spec(0)])
    run_to_end(sched)
    row = sched.row(S5)
    assert row["status"] == "excluded" and row["replaced_by"] is None
    assert any(e["event"] == "needs_decision" for e in sched.events(S5))
    assert sorted(r.run_id for r in ledger_rows(paths.main)) == [S0, S5]


def test_segfault_counts_as_crash_not_interruption(env) -> None:
    make, set_scenarios, _, _ = env
    set_scenarios(**{S0: "segv"})
    sched = make(on_interrupt="hold")
    sched.add([spec(0)])
    run_to_end(sched)
    assert sched.row(S0)["failure_cause"] == "crash"
    assert sched.row(S5)["status"] == "ledgered"


def test_interruption_is_held_by_default_then_restarted_by_the_operator(env) -> None:
    make, set_scenarios, paths, tmp = env
    set_scenarios(**{S0: "interrupt"})
    sched = make()
    sched.add([spec(0)])
    run_to_end(sched)
    assert sched.row(S0)["status"] == "interrupted"
    assert not paths.main.exists()  # nothing recorded, nothing replaced
    set_scenarios(**{S0: "success"})
    sched.resolve(S0, "restart")
    run_to_end(sched)
    row = sched.row(S0)
    assert row["status"] == "ledgered" and row["attempt"] == 2
    assert (tmp / "data" / "scheduler" / "interrupted" / f"{S0}-attempt1").is_dir()


def test_interruption_policy_is_fixed_per_data_root(env) -> None:
    make, _, _, _ = env
    make().run(once=True)
    with pytest.raises(SchedulerError, match="set-policy"):
        make(on_interrupt="restart").run(once=True)


def test_set_policy_applies_to_every_held_run(env) -> None:
    make, set_scenarios, _, _ = env
    set_scenarios(**{S0: "interrupt", S1: "interrupt"})
    sched = make(max_concurrent=2)
    sched.add([spec(0), spec(1)])
    run_to_end(sched)
    assert {sched.row(f"{BASE}-s{s}")["status"] for s in (0, 1)} == {"interrupted"}
    with pytest.raises(ValueError):
        sched.resolve(S0, "exclude")  # one-by-one exclusion is not offered
    set_scenarios(**{S0: "success", S1: "success"})
    sched.set_policy("restart")
    assert {sched.row(f"{BASE}-s{s}")["status"] for s in (0, 1)} == {"pending"}


def test_interruption_policy_exclude(env) -> None:
    make, set_scenarios, paths, _ = env
    set_scenarios(**{S0: "interrupt"})
    sched = make(on_interrupt="exclude")
    sched.add([spec(0)])
    run_to_end(sched)
    rows = {r.run_id: r for r in ledger_rows(paths.main)}
    assert rows[S0].failure_cause == "incomplete"
    assert rows[S5].completed


def test_unavailable_plugin_leaves_run_pending_without_exclusion(env) -> None:
    make, set_scenarios, paths, _ = env
    set_scenarios(**{S0: "unavailable"})
    sched = make()
    sched.add([spec(0)])
    run_to_end(sched)
    assert sched.row(S0)["status"] == "pending"
    assert not paths.main.exists()


def test_dependent_is_blocked_when_the_dependency_is_excluded(env) -> None:
    make, set_scenarios, _, _ = env
    base = spec(0)
    dependent = spec(0, name=DEPENDENT, depends=[base.run_id], group="controller")
    set_scenarios(**{base.run_id: "crash"})
    sched = make(max_concurrent=2)
    sched.add([dependent, base])
    run_to_end(sched)
    assert sched.row(dependent.run_id)["status"] == "blocked"


def test_dependency_order_is_respected(env) -> None:
    make, _, _, _ = env
    base = spec(0)
    dependent = spec(0, name=DEPENDENT, depends=[base.run_id], group="controller")
    sched = make(max_concurrent=2)
    sched.add([dependent, base])
    sched.run(once=True)
    assert sched.row(dependent.run_id)["status"] == "pending"
    assert sched.row(base.run_id)["status"] == "training"
    run_to_end(sched)
    assert sched.summary() == {"ledgered": 2}


def test_concurrency_limit_and_adoption_after_scheduler_restart(env) -> None:
    make, set_scenarios, _, tmp = env
    set_scenarios(**{f"{BASE}-s{s}": "hang" for s in range(3)})
    sched = make(max_concurrent=2)
    sched.add([spec(s) for s in range(3)])
    sched.run(once=True)
    assert len(sched.rows("training")) == 2
    # a new scheduler instance (e.g. after the scheduler process restarted) adopts the live runs
    sched2 = make(max_concurrent=2)
    sched2.run(once=True)
    assert len(sched2.rows("training")) == 2
    assert any(e["event"] == "adopted" for e in sched2.events())
    (tmp / "release").write_text("go")
    run_to_end(sched2)
    assert sched2.summary() == {"ledgered": 3}


def test_never_reuses_a_run_directory_with_output(env) -> None:
    make, _, _, tmp = env
    sched = make()
    sched.add([spec(0)])
    run_dir = tmp / "data" / "checkpoints" / S0
    (run_dir / "omnisafe").mkdir(parents=True)
    with pytest.raises(SchedulerError, match="already holds output"):
        sched.run(once=True)


def test_eval_failed_can_be_reevaluated_and_does_not_block_dependents(env) -> None:
    make, set_scenarios, _, _ = env
    base = spec(0)
    dependent = spec(0, name=DEPENDENT, depends=[base.run_id], group="controller")
    set_scenarios(**{base.run_id: "evalfail"})
    sched = make(max_concurrent=1)
    sched.add([base, dependent])
    run_to_end(sched)
    assert sched.row(base.run_id)["status"] == "eval_failed"
    assert sched.row(dependent.run_id)["status"] == "ledgered"  # it needed only the base run's training
    set_scenarios(**{base.run_id: "success"})
    sched.resolve(base.run_id, "reevaluate")
    run_to_end(sched)
    assert sched.row(base.run_id)["status"] == "ledgered"


def test_a_ledger_failed_run_whose_evaluation_must_be_redone_can_be_reevaluated(env) -> None:
    """The ledger writer refuses an evaluation.json without per-episode arrays and says "re-evaluate", so a
    ledger_failed run in no ledger can be re-evaluated (before, `reevaluate` took only eval_failed runs and
    `retry-ledger` failed the same way again); one already recorded keeps its recorded evaluation."""
    make, _, _, _ = env
    sched = make()
    run_id = spec(0).run_id
    sched.add([spec(0)])
    sched._write_ledger = lambda run_id: None  # train and evaluate, stop before the ledger write
    for _ in range(400):
        sched.run(once=True)
        if sched.row(run_id)["status"] == "evaluated":
            break
        time.sleep(0.05)
    del sched._write_ledger
    path = sched.cfg.run_dir(run_id) / "evaluation.json"
    evaluation = json.loads(path.read_text())
    path.write_text(json.dumps({k: v for k, v in evaluation.items() if k not in ("episode_costs", "episode_returns")}))
    sched.step()
    assert sched.row(run_id)["status"] == "ledger_failed"
    assert "reevaluate" in sched.events(run_id)[0]["detail"]
    sched.resolve(run_id, "reevaluate")
    assert sched.row(run_id)["status"] == "trained"
    run_to_end(sched)
    assert sched.row(run_id)["status"] == "ledgered"
    with sched._write_txn():  # a recorded run's evaluation is the recorded one
        sched._set_sql(run_id, status="ledger_failed")
    with pytest.raises(SchedulerError, match="already recorded in a ledger or sidecar"):
        sched.resolve(run_id, "reevaluate")


def test_a_failed_reevaluation_is_never_taken_for_the_earlier_evaluation(env) -> None:
    """`reevaluate` moves the earlier evaluation.json aside, and an evaluate stage that exits nonzero is eval_failed
    whatever file is there, so a stale evaluation never reaches the ledger writer again."""
    make, set_scenarios, _, _ = env
    sched = make()
    run_id = spec(0).run_id
    sched.add([spec(0)])
    sched._write_ledger = lambda run_id: None
    for _ in range(400):
        sched.run(once=True)
        if sched.row(run_id)["status"] == "evaluated":
            break
        time.sleep(0.05)
    del sched._write_ledger
    path = sched.cfg.run_dir(run_id) / "evaluation.json"
    evaluation = json.loads(path.read_text())
    stale = json.dumps({k: v for k, v in evaluation.items() if k not in ("episode_costs", "episode_returns")})
    path.write_text(stale)
    sched.step()
    assert sched.row(run_id)["status"] == "ledger_failed"
    sched.resolve(run_id, "reevaluate")
    assert not path.exists()
    kept = sched.cfg.state_dir / "reevaluated" / f"{run_id}-1.json"
    assert kept.read_text() == stale and str(kept) in sched.events(run_id)[0]["detail"]
    set_scenarios(**{run_id: "evalfail"})  # the re-evaluation itself fails (exit 1)
    for _ in range(200):
        sched.run(once=True)
        if sched.row(run_id)["status"] not in ("trained", "evaluating"):
            break
        time.sleep(0.05)
    events = [e["event"] for e in sched.events(run_id, limit=6)]
    assert sched.row(run_id)["status"] == "eval_failed", events
    assert "evaluated" not in events[:3] and "ledger_failed" not in events[:1]
    # an evaluate stage that exits nonzero after writing a file is still a failure
    path.write_text(json.dumps(evaluation))
    sched._exit_code = lambda run_id: 1
    with sched._write_txn():
        sched._set_sql(run_id, status="evaluating")
    sched._finish_evaluation(run_id)
    assert sched.row(run_id)["status"] == "eval_failed"


def test_run_directory_with_a_claim_is_not_relaunched(env) -> None:
    make, _, _, tmp = env
    sched = make()
    sched.add([spec(0)])
    run_dir = tmp / "data" / "checkpoints" / S0
    run_dir.mkdir(parents=True)
    (run_dir / ".claim").write_text("123")
    with pytest.raises(SchedulerError, match="already holds output"):
        sched.run(once=True)


def test_smoke_runs_cannot_write_the_repository_ledgers(tmp_path) -> None:
    from pilot.ledger_writer import DEFAULT_PATHS

    with pytest.raises(ValueError, match="smoke"):
        SchedulerConfig(data_root=tmp_path, allow_dirty=True, ledger_paths=DEFAULT_PATHS)
    cfg = SchedulerConfig(data_root=tmp_path, allow_pending=True)
    assert cfg.ledger_paths.main == tmp_path / "smoke" / "ledger.parquet"


def test_smoke_mode_is_refused_on_the_registered_data_root(tmp_path, capsys, monkeypatch) -> None:
    """Smoke mode is refused on a registered data root and inside it, as scripts/smoke_run.py refuses it, so
    `schedule run --allow-pending` without --data-root never fixes the default root (/data) as a smoke root."""
    import pilot.scheduler as S
    from pilot.__main__ import main

    for root in (Path("/data"), Path("/data/smoke")):  # refused before anything is created
        for flag in ("allow_dirty", "allow_pending"):
            with pytest.raises(ValueError, match="registered data root /data"):
                SchedulerConfig(data_root=root, **{flag: True})
    assert SchedulerConfig(data_root=Path("/data")).mode == "registered"  # registered mode keeps its root
    registered = tmp_path / "registered"
    monkeypatch.setattr(S, "REGISTERED_DATA_ROOTS", (str(registered),))
    for flag in ("--allow-pending", "--allow-dirty"):
        assert main(["schedule", "run", "--max-concurrent", "1", "--once", flag, "--data-root", str(registered)]) == 1
        assert "registered data root" in capsys.readouterr().err
    assert not (registered / "scheduler").exists()
    SchedulerConfig(data_root=tmp_path / "scratch", allow_pending=True)  # a scratch root stays a smoke root


def test_surplus_seeds_follow_part_5_5(env) -> None:
    from pilot import manifest

    make, _, _, _ = env
    sched = make()
    primary = [s for s in manifest.study_a_main((0,)) if s.task == R.PRIMARY_TASK and s.N in (0.0, 0.50)]
    other = [s for s in manifest.study_a_main((0,)) if s.N == 0.25][:1]
    b = manifest.study_b((0,))[:1]
    sched.add(primary + other + b)
    added = sched.add_surplus(2)
    assert sorted(added) == sorted([f"{s.arm_id}-s{k}" for s in primary + b for k in (5, 6)])
    assert all(sched.row(r)["surplus"] == 1 for r in added)
    assert sched.exists(f"B-{b[0].arm}-fewshot-b5-s6")  # Study B surplus runs are continued too
    with pytest.raises(ValueError):
        sched.add_surplus(8)


def test_run_already_in_ledger_is_never_launched(env) -> None:
    make, _, paths, _ = env
    sched = make()
    sched.add([spec(0)])
    run_to_end(sched)
    other = make()  # same data root with its state emptied, same ledger
    other.db.execute("DELETE FROM runs")
    other.db.commit()
    other.add([spec(0)])
    other.run(once=True)
    assert other.row(S0)["status"] == "ledgered"
    assert len(ledger_rows(paths.main)) == 1


def test_a_row_already_present_gets_its_missing_evaluation_record(env) -> None:
    """results/supplement_schema.py: the "row already present" path writes the evaluation supplement record if it is
    missing (pilot.ledger_writer.ensure_evaluation_supplement); an error there ends ledger_failed."""
    from pilot import supplement

    make, _, paths, _ = env
    run_id = S0
    sched = make()
    sched.add([spec(0)])
    run_to_end(sched)
    record = supplement.record_path("evaluation", run_id, ledger_path=paths.main)
    text = record.read_text()  # the fake launcher's evaluation.json carries the per-episode arrays
    recorded = supplement.read("evaluation", run_id, ledger_path=paths.main)
    assert recorded.final.mean_cost == ledger_rows(paths.main)[0].final_cost
    record.unlink()  # e.g. a row written before the supplement existed, or a scheduler stopped in between
    sched._set(run_id, status="evaluated", ledger_written=0)
    sched.run(once=True)
    assert sched.row(run_id)["status"] == "ledgered" and record.read_text() == text
    assert sched.events(run_id)[0]["detail"].startswith("row already present; evaluation record")
    assert len(ledger_rows(paths.main)) == 1
    # an evaluation.json that is not the row's evaluation: refused, the run waits for the operator
    run_dir = sched.cfg.run_dir(run_id)
    ev = json.loads((run_dir / "evaluation.json").read_text())
    ev["final_cost"], ev["episode_costs"]["final"] = 30.0, [30.0] * 100
    (run_dir / "evaluation.json").write_text(json.dumps(ev))
    record.unlink()
    sched._set(run_id, status="evaluated", ledger_written=0)
    sched.run(once=True)
    assert sched.row(run_id)["status"] == "ledger_failed" and not record.exists()
    assert "the ledger row records" in sched.events(run_id)[0]["detail"]


def test_changed_spec_for_an_existing_run_is_refused(env) -> None:
    make, _, _, _ = env
    sched = make()
    sched.add([spec(0)])
    changed = RunSpec.from_dict({**spec(0).to_dict(), "total_steps": 4_000_000})
    with pytest.raises(SchedulerError, match="different spec"):
        sched.add([changed])


def test_pending_questions_block_launch_unless_allowed(env, monkeypatch) -> None:
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", R.ANSWERED_QUESTIONS - {"Q-rounding"})  # an open question
    make, _, _, tmp = env
    held = RunSpec.from_dict({**spec(0).to_dict(), "pending": ["Q-rounding"]})
    sched = make()
    sched.add([held])
    run_to_end(sched)
    assert sched.row(held.run_id)["status"] == "pending"
    waiting = [e for e in sched.events(held.run_id) if e["event"] == "open_questions"]
    assert len(waiting) == 1 and waiting[0]["detail"] == "Q-rounding"  # `schedule status` shows why it waits
    # with --allow-pending (smoke) the open question holds nothing
    allowed = make(data_root=tmp / "d2", allow_pending=True)
    allowed.add([held])
    run_to_end(allowed)
    assert allowed.row(held.run_id)["status"] == "ledgered"


def test_only_one_scheduler_per_data_root(env) -> None:
    make, *_ = env
    import fcntl

    sched = make()
    with open(sched.cfg.state_dir / "scheduler.lock", "w", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(SchedulerError, match="another scheduler"):
            sched.run(once=True)


def test_invalid_config_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError):
        SchedulerConfig(data_root=tmp_path, on_interrupt="ignore")
    with pytest.raises(ValueError):
        SchedulerConfig(data_root=tmp_path, max_concurrent=0)
    for poll in (-1.0, float("nan"), float("inf")):  # time.sleep refuses NaN and overflows on inf
        with pytest.raises(ValueError, match="poll_seconds must be a finite, non-negative number"):
            SchedulerConfig(data_root=tmp_path, poll_seconds=poll)


def test_replacement_waits_for_the_seed_rule_to_be_recorded(env, monkeypatch) -> None:
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", R.ANSWERED_QUESTIONS - {"Q-seed-collision"})  # while it is open
    make, set_scenarios, _, _ = env
    set_scenarios(**{S0: "crash"})
    sched = make()
    sched.add([spec(0)])
    run_to_end(sched)
    replacement = sched.spec(S5)
    assert "Q-seed-collision" in replacement.pending
    assert sched.row(replacement.run_id)["status"] == "pending"


def test_fewshot_continuation_ends_as_continued_without_a_row(env, monkeypatch) -> None:
    from pilot import manifest

    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset({"Q-seed-collision", "Q-continuations", "Q-jc-window",
                                                            "Q-budget-normalisation", "Q-level-jc", "Q-studyb-order"}))
    make, _, paths, _ = env
    parent = RunSpec.from_dict({**manifest.study_b((0,))[4].to_dict(), "total_steps": 2_000_000})
    cont = RunSpec.from_dict({**manifest.study_b_fewshot([parent])[0].to_dict(), "total_steps": 200_000})
    sched = make(max_concurrent=2)
    sched.add([parent, cont])
    run_to_end(sched)
    assert sched.row(parent.run_id)["status"] == "ledgered" and sched.row(cont.run_id)["status"] == "continued"
    assert [r.run_id for r in ledger_rows(paths.main)] == [parent.run_id]


def test_excluded_study_b_run_gets_its_replacement_continued(env, monkeypatch) -> None:
    from pilot import manifest

    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset({"Q-seed-collision", "Q-jc-window", "Q-budget-normalisation",
                                                            "Q-level-jc", "Q-studyb-order"}))
    make, set_scenarios, _, _ = env
    parent = RunSpec.from_dict({**manifest.study_b((0,))[4].to_dict(), "total_steps": 2_000_000})  # Moderate, short
    assert parent.arm == "Moderate"
    continuations = manifest.study_b_fewshot([parent])
    set_scenarios(**{parent.run_id: "crash"})
    sched = make(max_concurrent=2)
    sched.add([parent, *continuations])
    sched.run(max_passes=3)
    assert {sched.row(c.run_id)["status"] for c in continuations} == {"superseded"}
    new_ids = [f"B-Moderate-fewshot-b{b:g}-s5" for b in R.UNSEEN_BUDGETS]
    for rid in new_ids:
        assert sched.spec(rid).depends_on == ("B-Moderate-s5",)


def test_failed_exclusion_record_is_retried_within_the_same_run(env, monkeypatch) -> None:
    import pilot.scheduler as S

    make, set_scenarios, paths, _ = env
    set_scenarios(**{S1: "crash"})
    real = S.write_run
    calls = {"n": 0}

    def flaky(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("disk briefly unavailable")
        return real(*a, **kw)

    monkeypatch.setattr(S, "write_run", flaky)
    sched = make()
    sched.add([spec(1)])
    run_to_end(sched)
    ids = [r.run_id for r in ledger_rows(paths.main)]
    assert S1 in ids and sched.row(S1)["ledger_written"] == 1


def test_orphaned_child_is_adopted_not_relaunched(env) -> None:
    import subprocess

    make, set_scenarios, _, tmp = env
    set_scenarios(**{S0: "hang"})
    sched = make()
    sched.add([spec(0)])
    run_dir = tmp / "data" / "checkpoints" / S0
    run_dir.mkdir(parents=True)
    (run_dir / "spec.json").write_text(spec(0).to_json())
    # the scheduler died after committing 'training' and starting the child, before recording its PID
    child = subprocess.Popen([sys.executable, str(FAKE), "train", str(run_dir / "spec.json"), str(run_dir)])
    with sched.db:
        sched.db.execute("UPDATE runs SET status='training', pid=NULL, host=? WHERE run_id=?", (sched._host, S0))
    fresh = make()
    fresh.run(once=True)
    assert fresh.row(S0)["pid"] == child.pid
    assert not any(e["event"] == "train_started" for e in fresh.events(S0))
    (tmp / "release").write_text("go")
    child.wait(timeout=30)
    run_to_end(fresh)
    assert fresh.row(S0)["status"] == "ledgered"


# -- process identification and restarts -------------------------------------------------------------


def test_a_run_is_not_taken_for_a_run_whose_id_extends_it(env) -> None:
    make, set_scenarios, _, tmp = env
    set_scenarios(**{f"{BASE}-s10": "hang"})
    sched = make()
    sched.add([spec(10)])
    sched.run(once=True)
    s10_pid = sched.row(f"{BASE}-s10")["pid"]
    sched.add([spec(1)])
    # the scheduler committed 'training' for s1 and died before starting it, while s10 runs
    with sched.db:
        sched.db.execute("UPDATE runs SET status='training', pid=NULL, host=? WHERE run_id=?", (sched._host, S1))
    fresh = make()
    fresh.recover()
    assert fresh.row(S1)["pid"] != s10_pid and fresh.row(S1)["status"] == "pending"
    assert any(e["event"] == "never_started" for e in fresh.events(S1))
    (tmp / "release").write_text("go")
    run_to_end(fresh)
    assert fresh.summary() == {"ledgered": 2}


def test_a_live_run_is_adopted_not_restarted_when_the_host_name_changed(env) -> None:
    make, set_scenarios, _, tmp = env
    set_scenarios(**{S0: "hang"})
    sched = make(on_interrupt="restart")
    sched.add([spec(0)])
    sched.run(once=True)
    pid = sched.row(S0)["pid"]
    fresh = make()
    fresh._host = "renamed-host"
    fresh.run(once=True)
    row = fresh.row(S0)
    assert (row["status"], row["attempt"], row["pid"], row["host"]) == ("training", 1, pid, "renamed-host")
    with pytest.raises(SchedulerError, match="only an interrupted run"):
        fresh.restart(S0)  # an operator restarts only a held run
    with pytest.raises(SchedulerError, match="live run is never restarted"):
        fresh.restart(S0, expected=("training",))  # as the policy would, had the attempt seemed to end
    (tmp / "release").write_text("go")
    run_to_end(fresh)
    assert fresh.row(S0)["status"] == "ledgered" and fresh.row(S0)["attempt"] == 1


@pytest.mark.parametrize("policy", ["hold", "exclude"])
def test_a_run_that_never_started_goes_back_to_pending(env, monkeypatch, policy) -> None:
    make, _, paths, _ = env
    sched = make(on_interrupt=policy)
    sched.add([spec(0)])

    def killed(*a, **kw):
        raise KeyboardInterrupt  # the scheduler dies between the 'training' commit and the spawn

    monkeypatch.setattr(sched, "_spawn", killed)
    with pytest.raises(KeyboardInterrupt):
        sched.run(once=True)
    assert sched.row(S0)["status"] == "training"
    fresh = make(on_interrupt=policy)
    run_to_end(fresh)
    row = fresh.row(S0)
    assert row["status"] == "ledgered" and row["failure_cause"] is None and not fresh.exists(S5)
    assert [r.completed for r in ledger_rows(paths.main)] == [True]


def test_a_launcher_error_before_the_claim_is_neither_held_nor_excluded(env) -> None:
    make, set_scenarios, paths, _ = env
    set_scenarios(**{S0: "noclaim"})
    sched = make(on_interrupt="exclude")
    sched.add([spec(0)])
    run_to_end(sched)
    assert sched.row(S0)["status"] == "pending" and not paths.main.exists()
    events = [e["event"] for e in sched.events(S0)]
    assert "launch_failed" in events and events.count("train_started") == 1  # not relaunched by this process


def test_a_refused_launch_stays_pending_and_nothing_is_deleted(env) -> None:
    make, set_scenarios, paths, tmp = env
    set_scenarios(**{S0: "refused"})
    sched = make()
    sched.add([spec(0)])
    run_to_end(sched)
    assert sched.row(S0)["status"] == "pending" and not paths.main.exists()
    assert [e["event"] for e in sched.events(S0)].count("launch_refused") == 1
    assert sorted(p.name for p in (tmp / "data" / "checkpoints" / S0).iterdir()) == ["spec.json"]


def test_a_missing_evaluator_leaves_the_run_trained(env) -> None:
    make, set_scenarios, paths, _ = env
    set_scenarios(**{S0: "evalunavailable"})
    sched = make()
    sched.add([spec(0)])
    run_to_end(sched)
    assert sched.row(S0)["status"] == "trained" and not paths.main.exists()
    assert any(e["event"] == "evaluator_unavailable" for e in sched.events(S0))


def test_a_run_stopped_the_same_way_after_max_restarts_is_excluded_incomplete_and_replaced(env) -> None:
    """Table 9.1 (Q-interrupted-run): an end without a final result of process origin (here SIGKILL, e.g. the OOM
    killer) is restarted MAX_RESTARTS times; the next such end "fails to complete its steps" (Part 5.6): excluded
    with cause 'incomplete', not by the policy, and replaced by the next unused seed."""
    from pilot.scheduler import MAX_RESTARTS

    make, set_scenarios, paths, _ = env
    set_scenarios(**{S0: "interrupt"})
    sched = make(on_interrupt="restart")
    sched.add([spec(0)])
    run_to_end(sched, passes=200)
    row = sched.row(S0)
    assert (row["status"], row["attempt"], row["failure_cause"]) == ("excluded", MAX_RESTARTS + 1, "incomplete")
    assert row["replaced_by"] == S5 and sched.row(S5)["status"] == "ledgered"
    assert sched._interruption_origins(S0) == ["process"] * (MAX_RESTARTS + 1)
    excluded = next(e["detail"] for e in sched.events(S0) if e["event"] == "excluded")
    assert f"{MAX_RESTARTS + 1} times (more than MAX_RESTARTS = {MAX_RESTARTS})" in excluded
    assert "fails to complete its steps (Part 5.6" in excluded
    assert not any(e["event"] in ("excluded_by_policy", "restart_limit") for e in sched.events(S0))
    rows = {r.run_id: r for r in ledger_rows(paths.main)}
    assert rows[S0].failure_cause == "incomplete" and rows[S5].completed


def test_exclusions_at_the_restart_cap_count_towards_the_same_cause_breaker(env) -> None:
    make, set_scenarios, _, _ = env
    set_scenarios(**{S0: "interrupt", S5: "interrupt"})  # the arm's runs die the same way: lower the concurrency
    sched = make(on_interrupt="restart")
    sched.add([spec(0)])
    run_to_end(sched, passes=300)
    assert sched.row(S0)["replaced_by"] == S5
    last = sched.row(S5)
    assert (last["status"], last["failure_cause"], last["replaced_by"]) == ("excluded", "incomplete", None)
    decision = next(e["detail"] for e in sched.events(S5) if e["event"] == "needs_decision")
    assert decision.startswith(f"2 exclusions of arm {spec(0).arm_id} with cause 'incomplete' (circuit breaker")
    assert "its seed target: 5" in decision and "never their costs or returns" in decision
    assert "erratum" in decision and f"`schedule resolve {S5} replace`" in decision


@pytest.mark.parametrize("scenario", ["sigterm", "sighup"])
def test_an_interruption_of_machine_origin_is_restarted_without_limit(env, scenario) -> None:
    """Stopped from outside (the launcher recorded SIGTERM, or the child died of SIGHUP): restarted every time, never
    capped (Table 9.1, Q-interrupted-run)."""
    from pilot.scheduler import MAX_RESTARTS

    make, set_scenarios, _, _ = env
    set_scenarios(**{S0: scenario})
    sched = make(on_interrupt="restart")
    sched.add([spec(0)])
    deadline = time.monotonic() + 300  # each attempt is a launcher process: generous on a loaded machine
    while sched.row(S0)["attempt"] <= MAX_RESTARTS + 3 and time.monotonic() < deadline:
        sched.run(once=True)
        time.sleep(0.02)
    assert sched.row(S0)["attempt"] > MAX_RESTARTS + 3 and sched.row(S0)["status"] in ("pending", "training")
    set_scenarios(**{S0: "success"})
    run_to_end(sched)
    assert sched.row(S0)["status"] == "ledgered" and not sched.exists(S5)
    assert set(sched._interruption_origins(S0)) == {"machine"}
    if scenario == "sighup":
        assert any("stopped by signal 1 before the launcher recorded it" in (e["detail"] or "")
                   for e in sched.events(S0, limit=1000) if e["event"] == "interrupted")


def test_an_end_no_scheduler_watched_is_restarted_without_limit(env) -> None:
    """An attempt that ended while no scheduler was its parent (exit status unknown) is of machine origin."""
    from pilot.launch import CLAIM_FILE
    from pilot.scheduler import MAX_RESTARTS

    make, _, _, _ = env
    sched = make(on_interrupt="restart")
    sched.add([spec(0)])
    for attempt in range(1, MAX_RESTARTS + 4):
        run_dir = sched.cfg.run_dir(S0)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / CLAIM_FILE).write_text("1\n")  # claimed, then it ended while no scheduler watched it
        sched._set(S0, status="training")
        sched._finish_training(S0)  # no child of this process: the exit code is None
        assert (sched.row(S0)["status"], sched.row(S0)["attempt"]) == ("pending", attempt + 1)
    assert sched._interruption_origins(S0) == ["machine"] * (MAX_RESTARTS + 3)


def test_set_policy_restart_applies_the_restart_rule_to_held_runs(env) -> None:
    """`schedule set-policy restart` settles every held run by the rule: S0, held after its third end of process
    origin (two operator restarts in between), is excluded 'incomplete' and replaced; S1, held once, is restarted.
    The cap binds the operator too: `schedule resolve S0 restart` refuses S0 past it, and `schedule status` names
    only `set-policy restart` for it."""
    from pilot.scheduler import MAX_RESTARTS

    make, set_scenarios, _, _ = env
    set_scenarios(**{S0: "interrupt", S1: "interrupt"})
    sched = make(max_concurrent=2)  # hold
    sched.add([spec(0), spec(1)])
    run_to_end(sched)
    for _ in range(MAX_RESTARTS):
        sched.resolve(S0, "restart")
        run_to_end(sched)
    assert (sched.row(S0)["status"], sched.row(S0)["attempt"]) == ("interrupted", MAX_RESTARTS + 1)
    assert (sched.row(S1)["status"], sched.row(S1)["attempt"]) == ("interrupted", 1)
    with pytest.raises(SchedulerError, match=f"{MAX_RESTARTS + 1} times .*set-policy restart` excludes it"):
        sched.resolve(S0, "restart")  # past the cap: never restarted a fourth time
    assert (sched.row(S0)["status"], sched.row(S0)["attempt"]) == ("interrupted", MAX_RESTARTS + 1)
    decisions = dict(sched.status_report()["decisions"])
    assert "`schedule set-policy restart` excludes it" in decisions[S0] and "resolve" not in decisions[S0]
    assert f"`schedule resolve {S1} restart`" in decisions[S1]  # below the cap: either action
    set_scenarios(**{S1: "success"})
    sched.set_policy("restart")
    s0 = sched.row(S0)
    assert (s0["status"], s0["failure_cause"], s0["replaced_by"]) == ("excluded", "incomplete", S5)
    assert (sched.row(S1)["status"], sched.row(S1)["attempt"]) == ("pending", 2)
    run_to_end(sched)
    assert sched.row(S1)["status"] == "ledgered" and sched.row(S5)["status"] == "ledgered"


def test_an_operator_restart_past_the_cap_is_refused_on_a_registered_root_once_the_key_is_answered(
        tmp_path, pin_open) -> None:
    """On a registered root a held run past Q-interrupted-run's cap is refused by `schedule resolve RUN restart` once
    the key is answered (Table 9.1), as the 'restart' policy would exclude it; while the key is open the cap is not in
    force and the operator's restart goes ahead."""
    from pilot.scheduler import MAX_RESTARTS, ORIGIN_PREFIX, ORIGIN_PROCESS

    sched = Scheduler(SchedulerConfig(data_root=tmp_path / "root"))
    sched.fix_mode(ledgers=False)
    sched.add([spec(0)])
    with sched._write_txn():  # held under 'hold' after three ends of process origin, as _interrupted logs them
        for _ in range(MAX_RESTARTS + 1):
            sched._event_sql(S0, "interrupted", f"{ORIGIN_PREFIX}{ORIGIN_PROCESS}: no final training result (exit "
                                                "code -9); policy=hold")
        sched._set_sql(S0, status="interrupted", attempt=MAX_RESTARTS + 1)
    with pytest.raises(SchedulerError, match=r"more than MAX_RESTARTS = 2\): it fails to complete its steps"):
        sched.resolve(S0, "restart")
    assert "`schedule set-policy restart` excludes it" in dict(sched.status_report()["decisions"])[S0]
    pin_open("Q-interrupted-run")
    sched.resolve(S0, "restart")
    assert (sched.row(S0)["status"], sched.row(S0)["attempt"]) == ("pending", MAX_RESTARTS + 2)


def test_a_policy_restart_of_a_watched_attempt_that_changed_meanwhile_says_so(env) -> None:
    """``restart(expected=('training',))`` (the policy settling the attempt it watched end) names the status it
    expected, not the operator's rule, in the restart_skipped event."""
    from pilot.scheduler import StatusChanged

    make, set_scenarios, _, _ = env
    set_scenarios(**{S0: "interrupt"})
    sched = make()  # hold
    sched.add([spec(0)])
    run_to_end(sched)
    with pytest.raises(StatusChanged, match="is interrupted, not training; another command acted on it first"):
        sched.restart(S0, expected=("training",))
    sched._restart_or_hold(S0, "training")
    assert sched.events(S0)[0]["event"] == "restart_skipped" and "not training" in sched.events(S0)[0]["detail"]
    assert sched.row(S0)["status"] == "interrupted"


def restart_cut_short_after_the_archive_move(env, monkeypatch) -> Scheduler:
    """A held S0 whose `set-policy restart` died right after moving the attempt's directory to its archive."""
    import pilot.scheduler as S

    make, _, _, _ = env
    sched = make()  # hold
    sched.add([spec(0)])
    run_to_end(sched)
    assert sched.row(S0)["status"] == "interrupted"
    real_move = S.shutil.move

    def move_then_die(src, dst):
        real_move(src, dst)
        raise KeyboardInterrupt

    monkeypatch.setattr(S.shutil, "move", move_then_die)
    with pytest.raises(KeyboardInterrupt):
        sched.set_policy("restart")
    monkeypatch.setattr(S.shutil, "move", real_move)
    assert (sched.row(S0)["status"], sched.row(S0)["attempt"]) == ("interrupted", 1)
    assert not sched.cfg.run_dir(S0).exists() and sched._archive(S0, 1).is_dir()
    return sched


def test_a_set_policy_restart_cut_short_after_its_archive_move_is_completed_at_the_next_start(env, monkeypatch) -> None:
    make, set_scenarios, _, _ = env
    set_scenarios(**{S0: "interrupt"})
    restart_cut_short_after_the_archive_move(env, monkeypatch)
    set_scenarios(**{S0: "success"})
    fresh = make()
    run_to_end(fresh)
    assert (fresh.row(S0)["status"], fresh.row(S0)["attempt"]) == ("ledgered", 2)
    assert any(e["event"] == "restart_resumed" for e in fresh.events(S0, limit=1000))


def test_an_exclusion_after_a_restart_cut_short_keeps_the_launchers_record(env, monkeypatch) -> None:
    _, set_scenarios, paths, _ = env
    set_scenarios(**{S0: "sigterm"})  # the launcher records the interruption itself
    sched = restart_cut_short_after_the_archive_move(env, monkeypatch)
    sched.set_policy("exclude")  # the smoke root switches to 'exclude' before any scheduler started
    assert sched.row(S0)["status"] == "excluded" and not sched._archive(S0, 1).exists()
    assert any(e["event"] == "archive_restored" for e in sched.events(S0, limit=1000))
    record = json.loads((sched.cfg.run_dir(S0) / "train_result.json").read_text(encoding="utf-8"))
    assert (record["failure_cause"], record["interruption"]) == ("incomplete", "signal 15")
    assert "status 'interrupted'" in record["written_by"]
    assert {r.run_id: r.failure_cause for r in ledger_rows(paths.main)}[S0] == "incomplete"


def test_set_policy_exclude_logs_the_commit_of_the_replacements_code(env) -> None:
    """`schedule set-policy exclude` from a fresh process reads HEAD before it queues replacements, as `resolve`
    does: the replacement's `queued` event names the commit, never 'code at commit None'."""
    from pilot import provenance

    make, set_scenarios, _, _ = env
    set_scenarios(**{S0: "interrupt"})
    sched = make()
    sched.add([spec(0)])
    run_to_end(sched)
    fresh = make()  # as the command line: a new Scheduler for `schedule set-policy exclude --allow-dirty`
    fresh.set_policy("exclude")
    queued = [e["detail"] for e in fresh.events(S5) if e["event"] == "queued"]
    head = provenance.commit_hash(fresh.cfg.repo_root)
    assert head and queued == [f"replacement of {S0} (code at commit {head}, uncommitted changes allowed: smoke)"]


def test_the_launchers_own_interruption_record_is_kept(env) -> None:
    make, set_scenarios, paths, tmp = env
    set_scenarios(**{S0: "sigterm"})
    sched = make()
    sched.add([spec(0)])
    run_to_end(sched)
    assert sched.row(S0)["status"] == "interrupted"
    assert any("the launcher recorded an interruption" in (e["detail"] or "") for e in sched.events(S0))
    sched.set_policy("exclude")
    record = json.loads((tmp / "data" / "checkpoints" / S0 / "train_result.json").read_text(encoding="utf-8"))
    assert (record["status"], record["failure_cause"], record["interruption"]) == ("failed", "incomplete", "signal 15")
    assert "status 'interrupted'" in record["written_by"]
    assert {r.run_id: r.failure_cause for r in ledger_rows(paths.main)}[S0] == "incomplete"


# -- exclusion, replacement and the operator's decisions ---------------------------------------------


def test_exclusions_applied_by_the_interruption_policy_do_not_trip_the_breaker(env) -> None:
    make, set_scenarios, _, _ = env
    set_scenarios(**{f"{BASE}-s{s}": "interrupt" for s in range(3)})  # one outage stops three runs
    sched = make(max_concurrent=3)
    sched.add([spec(s) for s in range(3)])
    run_to_end(sched)
    sched.set_policy("exclude")
    assert [sched.row(f"{BASE}-s{s}")["replaced_by"] for s in range(3)] == [
        f"{BASE}-s{s}" for s in (5, 6, 7)]


def test_replacement_seeds_are_chosen_inside_the_write_transaction(env, monkeypatch) -> None:
    make, set_scenarios, _, _ = env
    set_scenarios(**{S0: "crash"})
    sched = make()
    in_txn: list[bool] = []
    real = sched.next_unused_seed

    def spy(arm_id, also_used=()):
        in_txn.append(sched.db.in_transaction)
        return real(arm_id, also_used)

    monkeypatch.setattr(sched, "next_unused_seed", spy)
    sched.add([spec(0)])
    sched.run(max_passes=40)
    assert sched.row(S0)["replaced_by"] == S5 and in_txn and all(in_txn)
    other = make()  # an operator command beside the running scheduler waits for its write transaction
    other.db.execute("PRAGMA busy_timeout = 50")
    with sched._write_txn():
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            other.add([spec(9)])


def test_a_dependency_queued_later_is_waited_for(env) -> None:
    make, _, _, _ = env
    base = spec(0)
    dependent = spec(0, name=DEPENDENT, depends=[base.run_id], group="controller")
    sched = make(max_concurrent=2)
    sched.add([dependent])  # `controller` queued before `main`
    sched.run(once=True)
    assert sched.row(dependent.run_id)["status"] == "pending"
    assert any(e["event"] == "dependency_missing" for e in sched.events(dependent.run_id))
    sched.add([base])
    run_to_end(sched)
    assert sched.summary() == {"ledgered": 2}


def test_the_operator_can_replace_a_run_the_breaker_left_without_replacement(env) -> None:
    make, set_scenarios, _, _ = env
    set_scenarios(**{S0: "crash", S5: "crash"})
    sched = make()
    sched.add([spec(0)])
    run_to_end(sched)
    assert sched.row(S5)["replaced_by"] is None
    with pytest.raises(SchedulerError, match="without a replacement"):
        sched.resolve(S0, "replace")  # already replaced by s5
    sched.resolve(S5, "replace")
    run_to_end(sched)
    assert sched.row(S5)["replaced_by"] == S6
    assert sched.row(S6)["status"] == "ledgered"


def test_a_blocked_dependent_is_replaced_by_the_operator(env) -> None:
    make, set_scenarios, _, _ = env
    base = spec(0)
    dependent = spec(0, name=DEPENDENT, depends=[base.run_id], group="controller")
    set_scenarios(**{base.run_id: "crash"})
    sched = make(max_concurrent=2)
    sched.add([base, dependent])
    run_to_end(sched)
    assert sched.row(dependent.run_id)["status"] == "blocked"
    blocked = [e["detail"] for e in sched.events(dependent.run_id) if e["event"] == "blocked"]
    assert blocked == [f"dependency {base.run_id} excluded; see `schedule status` for the action"]
    assert dict(sched.status_report()["decisions"])[dependent.run_id] == (
        f"a dependency was excluded, blocked or superseded: `schedule resolve {dependent.run_id} replace` (its arm's "
        f"next unused seed runs instead), or `schedule resolve {dependent.run_id} unblock` once its dependencies are "
        "fine again")
    with pytest.raises(SchedulerError, match="still excluded"):
        sched.resolve(dependent.run_id, "unblock")
    sched.resolve(dependent.run_id, "replace")
    new = f"{DEPENDENT}-s5"
    assert sched.spec(new).depends_on == (S5,)
    assert [e["detail"] for e in sched.events(dependent.run_id) if e["event"] == "superseded"] == [
        f"replaced by {new} (operator)"]
    run_to_end(sched)
    assert sched.row(new)["status"] == "ledgered" and sched.row(S5)["status"] == "ledgered"


def test_a_warm_started_run_whose_n0_run_was_excluded_is_superseded_by_the_next_unused_seed(env) -> None:
    """Table 9.1 (Q-warm-start): no value is taken from an excluded N = 0 run. The warm-started run, which never
    trained (it waits for its dependency), is not excluded: `schedule status` names `replace`, which queues the arm's
    next unused seed with the N = 0 arm's run of that seed as its source, and the blocked row ends superseded."""
    from dataclasses import replace

    make, set_scenarios, _, _ = env
    base = spec(0)
    warm = replace(spec(0, name=WARM, depends=[base.run_id], group="controller"), controller_variant="warm_started")
    set_scenarios(**{base.run_id: "crash"})
    sched = make(max_concurrent=2)
    sched.add([base, warm])
    run_to_end(sched)
    assert sched.row(base.run_id)["status"] == "excluded" and sched.row(base.run_id)["replaced_by"] == S5
    assert sched.row(warm.run_id)["status"] == "blocked"
    assert any(e["event"] == "train_started" for e in sched.events(base.run_id))
    assert not any(e["event"] == "train_started" for e in sched.events(warm.run_id))  # it never trained
    hint = dict(sched.status_report()["decisions"])[warm.run_id]
    assert hint == (f"a dependency was excluded, blocked or superseded: `schedule resolve {warm.run_id} replace` "
                    "(Q-warm-start: no value is taken from an excluded run; this run never trained and is superseded "
                    "by its arm's next unused seed)")
    assert "unblock" not in hint
    sched.resolve(warm.run_id, "replace")
    new = f"{WARM}-s5"
    assert sched.row(warm.run_id)["status"] == "superseded" and sched.row(warm.run_id)["replaced_by"] == new
    assert [e["detail"] for e in sched.events(warm.run_id) if e["event"] == "superseded"] == [
        f"replaced by {new} (Table 9.1, Q-warm-start: no value is taken from an excluded run)"]
    assert sched.spec(new).depends_on == (S5,) and sched.spec(new).controller_variant == "warm_started"
    run_to_end(sched)
    assert sched.row(new)["status"] == "ledgered" and sched.row(S5)["status"] == "ledgered"
    assert sched.row(warm.run_id)["status"] == "superseded"  # never excluded (Part 5.6 counts no failure)


def test_unblock_returns_a_blocked_run_whose_dependencies_are_fine(env) -> None:
    make, _, _, _ = env
    base = spec(0)
    dependent = spec(0, name=DEPENDENT, depends=[base.run_id], group="controller")
    sched = make()
    sched.add([base, dependent])
    with sched.db:  # e.g. blocked by an earlier version that blocked runs whose dependency was not queued yet
        sched.db.execute("UPDATE runs SET status='blocked' WHERE run_id=?", (dependent.run_id,))
    sched.resolve(dependent.run_id, "unblock")
    run_to_end(sched)
    assert sched.summary() == {"ledgered": 2}


def test_an_exclusion_found_in_the_ledger_is_not_marked_ledgered(env) -> None:
    make, set_scenarios, _, _ = env
    set_scenarios(**{S0: "crash"})
    sched = make()
    sched.add([spec(0)])
    run_to_end(sched)
    other = make()  # the state was restored from a backup older than the exclusion
    other.db.execute("DELETE FROM runs")
    other.db.commit()
    other.add([spec(0)])
    other.run(once=True)
    row = other.row(S0)
    assert (row["status"], row["ledger_written"], row["failure_cause"]) == ("excluded", 1, "crash")
    need = [e["detail"] for e in other.events(S0) if e["event"] == "needs_decision"]
    assert need == [f"the exclusion was found in the ledger: check whether its replacement ran; if not, `schedule "
                    f"resolve {S0} replace`"]
    # `schedule status` shows that text, not the circuit breaker's rule (no breaker paused this run)
    assert dict(other.status_report()["decisions"])[S0] == need[0]
    other.resolve(S0, "replace")
    assert other.row(S5)["replaces"] == S0


# -- surplus seeds (Part 5.5) ------------------------------------------------------------------------


def test_surplus_is_a_per_arm_target_fixed_per_data_root(env) -> None:
    from collections import Counter

    from pilot import manifest

    make, _, _, _ = env
    sched = make()
    study_a = [s for s in manifest.study_a_main((0,)) if s.task == R.PRIMARY_TASK and s.N in (0.0, 0.50)]
    study_b = manifest.study_b((0,))
    sched.add(study_a)
    first = sched.add_surplus(2)
    sched.add(study_b)  # Study B is queued after Study A's surplus (HANDOVER Phase 8)
    second = sched.add_surplus(2)
    assert sorted(second) == sorted(f"{s.arm_id}-s{k}" for s in study_b for k in (5, 6))
    assert sched.add_surplus(2) == []
    with pytest.raises(SchedulerError, match="fixed at 2"):
        sched.add_surplus(3)
    per_arm = Counter(r["arm_id"] for r in sched.rows() if r["surplus"] and "fewshot" not in r["run_id"])
    assert len(per_arm) == len(study_a) + len(study_b) and set(per_arm.values()) == {2}
    assert not any("Q-seed-collision" in sched.spec(r).pending for r in first + second)


def test_the_replacement_of_a_surplus_run_is_surplus(env) -> None:
    from pilot import manifest

    make, set_scenarios, paths, _ = env
    set_scenarios(**{S5: "crash"})
    sched = make()
    sched.add([spec(0)])
    assert sched.add_surplus(1) == [S5]
    run_to_end(sched)
    replacement = sched.row(S6)
    assert replacement["surplus"] == 1 and replacement["replaces"] == S5
    assert sched.add_surplus(1) == []  # the replacement does not count as a second surplus seed
    rows = {r.run_id: r.model_dump() for r in ledger_rows(paths.main)}
    assert manifest.seed_role(rows[S6]) == "surplus"  # not a replacement of a registered seed


def test_ledger_notes_tell_a_replacement_from_a_surplus_seed(env) -> None:
    """A replacement (Part 5.6) and a surplus seed (Part 5.5) both take the arm's next unused seed, so the seed
    alone cannot tell them apart; the ledger notes say which (analysis.data.five_seed_view)."""
    from pilot import manifest

    make, set_scenarios, paths, _ = env
    set_scenarios(**{S0: "crash", S5: "crash"})
    sched = make()
    sched.add([spec(0)])
    assert sched.add_surplus(1) == [S5]
    run_to_end(sched)
    repl_registered = sched.row(S0)["replaced_by"]
    rows = {r.run_id: r.model_dump() for r in ledger_rows(paths.main)}
    assert manifest.seed_role(rows[repl_registered]) == "replacement"
    assert manifest.replacement_note(S0) in rows[repl_registered]["notes"]
    assert manifest.seed_role(rows[S5]) == "surplus"  # its exclusion row too
    assert manifest.seed_role(rows[S0]) is None


# -- modes, ledger locations and the interruption policy on a registered root ------------------------


def test_a_smoke_data_root_refuses_registered_decisions(env) -> None:
    make, set_scenarios, _, tmp = env
    set_scenarios(**{S0: "interrupt"})
    sched = make()  # smoke mode
    sched.add([spec(0)])
    run_to_end(sched)
    registered = Scheduler(SchedulerConfig(data_root=tmp / "data"))  # `set-policy` / `resolve` without --allow-dirty
    with pytest.raises(SchedulerError, match="smoke"):
        registered.set_policy("exclude")
    with pytest.raises(SchedulerError, match="smoke"):
        registered.resolve(S0, "restart")
    assert registered.row(S0)["status"] == "interrupted" and registered.setting("on_interrupt") == "hold"


def test_smoke_ledgers_may_not_lie_inside_the_repository(tmp_path) -> None:
    from pilot.provenance import REPO_ROOT

    inside = LedgerPaths(main=REPO_ROOT / "results" / "ledger.parquet", pilot=tmp_path / "p.parquet",
                         sidecar_dir=tmp_path / "s")
    with pytest.raises(ValueError, match="inside the repository"):
        SchedulerConfig(data_root=tmp_path, allow_dirty=True, ledger_paths=inside)


def test_registered_mode_gates_hold_pilot_runs_and_restarts(env, monkeypatch) -> None:
    import pilot.scheduler as S

    make, _, _, _ = env
    monkeypatch.setattr(S.provenance, "dirty_paths", lambda *a, **kw: [])
    monkeypatch.setattr(S.provenance, "file_committed_at", lambda *a, **kw: None)
    sched = make(allow_dirty=False)
    pilot_run = spec(0, name=PILOT_BASE, pilot=True, group="pilot")
    sched.add([pilot_run, spec(1)])
    with sched.db:  # attempt 1 of s1 ran on another commit
        sched.db.execute("UPDATE runs SET attempt=2, launch_commit='deadbeef' WHERE run_id=?", (S1,))
    run_to_end(sched)
    assert sched.row(pilot_run.run_id)["status"] == "pending" and sched.row(S1)["status"] == "pending"
    assert any(e["event"] == "allocation_missing" for e in sched.events(pilot_run.run_id))
    assert any(e["event"] == "restart_commit_mismatch" for e in sched.events(S1))


def test_registered_mode_waits_for_a_clean_worktree(env, monkeypatch) -> None:
    import pilot.scheduler as S

    make, _, _, _ = env
    monkeypatch.setattr(S.provenance, "dirty_paths", lambda *a, **kw: ["pilot/scheduler.py"])
    sched = make(allow_dirty=False)
    sched.add([spec(0)])
    run_to_end(sched)
    assert sched.row(S0)["status"] == "pending"
    assert any(e["event"] == "dirty_worktree" for e in sched.events(S0))


def test_schedule_run_uses_the_stored_policy(tmp_path) -> None:
    """`schedule run` without --on-interrupt runs under the policy stored for the data root, also one other than the
    default of a first run (a registered root's first run stores 'restart'; 'hold' is set here)."""
    from pilot.__main__ import main

    root = str(tmp_path / "root")
    assert main(["schedule", "run", "--max-concurrent", "1", "--once", "--data-root", root]) == 0
    assert Scheduler(SchedulerConfig(data_root=Path(root))).setting("on_interrupt") == "restart"
    assert main(["schedule", "set-policy", "hold", "--data-root", root]) == 0
    assert main(["schedule", "run", "--max-concurrent", "1", "--once", "--data-root", root]) == 0  # HANDOVER's command
    stored = Scheduler(SchedulerConfig(data_root=Path(root))).setting("on_interrupt")
    assert stored == "hold"


def test_the_interruption_policy_waits_for_q_interrupted_run_on_a_registered_root(tmp_path, monkeypatch) -> None:
    """'restart' applies the answer to Q-interrupted-run and 'exclude' is refused once it is answered; while the key
    is open a registered data root keeps 'hold' (exit 3, nothing changed), and a smoke root may use any policy."""
    from pilot.__main__ import main

    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", R.ANSWERED_QUESTIONS - {"Q-interrupted-run"})
    clean_tree(monkeypatch)  # `set-policy exclude` writes exclusion rows: committed code first
    root = str(tmp_path / "root")
    run = ["schedule", "run", "--max-concurrent", "1", "--once", "--data-root", root]
    assert main([*run[:-2], "--on-interrupt", "exclude", *run[-2:]]) == 3
    assert main(run) == 0
    for policy in ("restart", "exclude"):
        assert main(["schedule", "set-policy", policy, "--data-root", root]) == 3
    with pytest.raises(PendingQuestionError, match="Q-interrupted-run"):  # the first run of a data root
        Scheduler(SchedulerConfig(data_root=tmp_path / "other", on_interrupt="restart")).run(once=True)
    assert Scheduler(SchedulerConfig(data_root=Path(root))).setting("on_interrupt") == "hold"
    smoke = Scheduler(SchedulerConfig(data_root=tmp_path / "smoke", allow_dirty=True, allow_pending=True,
                                      on_interrupt="restart"))
    smoke.run(once=True)
    assert smoke.setting("on_interrupt") == "restart"
    assert "MAX_RESTARTS" in R.PENDING["Q-interrupted-run"]


def test_a_registered_root_runs_under_restart_once_q_interrupted_run_is_answered(tmp_path, monkeypatch) -> None:
    """Table 9.1 (Q-interrupted-run): registered data roots run under 'restart' (the default of a first run once the
    key is answered); 'exclude' is refused on them; 'hold' stays an operational pause; smoke roots keep 'hold' as
    their default and may use any policy."""
    from pilot.__main__ import main

    clean_tree(monkeypatch)  # `set-policy exclude` writes exclusion rows: committed code first
    root = tmp_path / "root"
    run = ["schedule", "run", "--max-concurrent", "1", "--once", "--data-root", str(root)]
    assert main([*run[:-2], "--on-interrupt", "exclude", *run[-2:]]) == 1  # SchedulerError: nothing fixed
    assert main(run) == 0
    assert Scheduler(SchedulerConfig(data_root=root)).setting("on_interrupt") == "restart"
    assert main(["schedule", "set-policy", "exclude", "--data-root", str(root)]) == 1
    with pytest.raises(SchedulerError, match="contradicts the answer to Q-interrupted-run"):
        Scheduler(SchedulerConfig(data_root=root)).set_policy("exclude")
    assert main(["schedule", "set-policy", "hold", "--data-root", str(root)]) == 0
    assert main(["schedule", "set-policy", "restart", "--data-root", str(root)]) == 0
    assert Scheduler(SchedulerConfig(data_root=root)).setting("on_interrupt") == "restart"
    smoke = Scheduler(SchedulerConfig(data_root=tmp_path / "smoke", allow_dirty=True))
    smoke.run(once=True)
    assert smoke.setting("on_interrupt") == "hold"
    smoke.set_policy("exclude")
    assert smoke.setting("on_interrupt") == "exclude"
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", R.ANSWERED_QUESTIONS - {"Q-interrupted-run"})
    fresh = tmp_path / "fresh"  # while the key is open a registered root's first run keeps 'hold'
    assert main(["schedule", "run", "--max-concurrent", "1", "--once", "--data-root", str(fresh)]) == 0
    assert Scheduler(SchedulerConfig(data_root=fresh)).setting("on_interrupt") == "hold"


def commit_files(repo: Path, files: dict[str, str]) -> str:
    """Commit ``files`` (repository-relative path: text) to the stand-in repository ``repo``; returns HEAD."""
    import subprocess

    for rel, text in files.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text, encoding="utf-8")

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()

    git("add", *files)
    git("-c", "user.email=t@example.org", "-c", "user.name=t", "commit", "-qm", "c")
    return git("rev-parse", "HEAD")


def test_replace_needs_a_fixed_defect_once_the_arms_exclusions_reach_its_seed_target(env) -> None:
    """Table 9.1 (Q-exclusion-breaker): the breaker is an operational pause (diagnose from the logs, fix a defect with
    an erratum, replace); once an arm's same-cause exclusions reach its seed target (every pilot arm, the one-seed
    Study B pilot arms included: the pilot's three seeds, so no target lies below the pause), `resolve RUN replace`
    is refused unless --defect-fixed names an entry of analysis/ERRATA.json committed at HEAD."""
    import pilot.scheduler as S
    from pilot import manifest

    make, set_scenarios, _, tmp = env
    repo = tmp / "repo"
    clean_git_repo(repo)
    p0, p5, p6, p7 = (f"{PILOT_BASE}-s{s}" for s in (0, 5, 6, 7))
    set_scenarios(**{r: "crash" for r in (p0, p5, p6, p7)})
    run = spec(0, name=PILOT_BASE, pilot=True, group="pilot")
    sched = make(repo_root=repo)
    assert sched.seed_target(run) == len(R.PILOT_SEEDS) == 3 and sched.seed_target(spec(0)) == len(R.SEEDS)
    targets = {sched.seed_target(s) for s in manifest.pilot()}  # the Study B pilot arms (one seed each) included
    assert targets == {len(R.PILOT_SEEDS)} and min(targets) > S.MAX_SAME_CAUSE_EXCLUSIONS  # no stop before the pause
    sched.add([run])
    run_to_end(sched)
    assert sched.row(p5)["replaced_by"] is None  # paused after two exclusions with one cause
    paused = sched._last_decision(p5)
    assert paused.startswith(f"2 exclusions of arm {run.arm_id} with cause 'crash' (circuit breaker")
    assert "diagnose from the failed runs' logs and tracebacks, never their costs or returns" in paused
    sched.resolve(p5, "replace")  # below the seed target: the operator replaces once diagnosed
    run_to_end(sched)
    assert sched.row(p6)["status"] == "excluded" and sched.row(p6)["replaced_by"] is None
    assert "it did not complete its seeds (crash, 3) and enters no comparison (Part 4.1)" in sched._last_decision(p6)
    assert any(r == p6 and "did not complete its seeds (crash, 3)" in text
               for r, text in sched.status_report()["decisions"])
    with pytest.raises(SchedulerError, match=r"did not complete its seeds \(crash, 3\).*--defect-fixed ERRATUM_ID"):
        sched.resolve(p6, "replace")
    with pytest.raises(SchedulerError, match="names no entry of analysis/ERRATA.json committed at HEAD"):
        sched.resolve(p6, "replace", defect_fixed="E1")
    commit_files(repo, {S.ERRATA_PATH: json.dumps({"errata": [{"id": "E0"}]})})
    with pytest.raises(SchedulerError, match="names no entry"):
        sched.resolve(p6, "replace", defect_fixed="E1")
    head = commit_files(repo, {S.ERRATA_PATH: json.dumps({"errata": [{"id": "E0"}, {"id": "E1"}]})})
    with pytest.raises(ValueError, match="erratum of a `replace`"):
        sched.resolve(p6, "retry-ledger", defect_fixed="E1")
    sched.resolve(p6, "replace", defect_fixed="E1")
    assert sched.row(p6)["replaced_by"] == p7
    replace = next(e["detail"] for e in sched.events(p6) if e["event"] == "replace")
    assert replace == f"operator: {p7}; defect fixed, erratum E1 ({S.ERRATA_PATH} at {head})"


def test_defect_fixed_is_a_usage_error_with_any_other_decision(tmp_path, capsys) -> None:
    from pilot.__main__ import main

    root = tmp_path / "root"
    assert main(["schedule", "resolve", S0, "restart", "--defect-fixed", "E1", "--data-root", str(root)]) == 2
    assert "erratum of a `replace`" in capsys.readouterr().err and not root.exists()


def search_module(*, with_amendment: bool = True) -> types.ModuleType:
    """A stand-in for Role 5's studyb.search whose SearchStatus is read from the committed decision.json (its
    'amendment' too, unless ``with_amendment`` is False: a checker whose status does not report one)."""
    import pilot.scheduler as S

    module = types.ModuleType(S.SEARCH_MODULE)

    def check_record(root, read=None):
        data = json.loads(read("decision.json") or b'{"complete": false}')
        status = {"complete": bool(data.get("complete")), "outcome": data.get("outcome"), "problems": []}
        if with_amendment:
            status["amendment"] = data.get("amendment")
        return types.SimpleNamespace(**status)

    module.check_record = check_record
    return module


def search_decision(outcome: str, amendment: str | None = None, complete: bool = True) -> dict[str, str]:
    import pilot.scheduler as S

    return {f"{S.SEARCH_RECORD_DIR}/decision.json": json.dumps({"outcome": outcome, "amendment": amendment,
                                                                 "complete": complete})}


def amendment_log(*ids: str) -> dict[str, str]:
    import pilot.scheduler as S

    return {S.AMENDMENTS_PATH: json.dumps({"amendments": [{"id": i, "keys": []} for i in ids]})}


def test_a_partial_search_outcome_waits_for_its_amendment_committed_with_it(tmp_path, monkeypatch) -> None:
    """Table 9.1 (Q-search-before-pilot): a 'partial' outcome releases Study B only when the commit also holds the
    narrowing amendment in analysis/AMENDMENTS.json, the entry whose id the decision names; 'none' needs none and
    'included' stays held."""
    import pilot.scheduler as S

    monkeypatch.setitem(sys.modules, S.SEARCH_MODULE, search_module())
    repo = tmp_path / "repo"
    clean_git_repo(repo)
    head = commit_files(repo, search_decision("partial", "A2"))
    released, reason = S.search_record_status(head, repo)
    assert not released and reason == (
        "the search outcome is 'partial': its narrowing amendment 'A2' is not committed in analysis/AMENDMENTS.json "
        "(Table C.1 'Decision': recorded before the first run)")
    (repo / "analysis").mkdir()
    (repo / S.AMENDMENTS_PATH).write_text(amendment_log("A2")[S.AMENDMENTS_PATH])  # uncommitted: not at HEAD
    assert not S.search_record_status(head, repo)[0]
    for log in (amendment_log("A1"), {S.AMENDMENTS_PATH: "{not json"}, {S.AMENDMENTS_PATH: '{"amendments": {}}'}):
        head = commit_files(repo, log)
        assert not S.search_record_status(head, repo)[0]
    head = commit_files(repo, amendment_log("A1", "A2"))
    released, reason = S.search_record_status(head, repo)
    assert released and "outcome 'partial'; its narrowing amendment 'A2' is committed" in reason
    monkeypatch.setitem(sys.modules, S.SEARCH_MODULE, search_module(with_amendment=False))
    assert not S.search_record_status(head, repo)[0]  # a status that names no amendment holds a partial outcome
    monkeypatch.setitem(sys.modules, S.SEARCH_MODULE, search_module())
    assert S.search_record_status(commit_files(repo, search_decision("none")), repo)[0]
    released, reason = S.search_record_status(commit_files(repo, search_decision("included", "A2")), repo)
    assert not released and "withdrawn by amendment" in reason


def test_the_amendment_log_is_the_one_the_search_checker_names() -> None:
    import pilot.scheduler as S

    record = pytest.importorskip("studyb.search.record")
    assert record.AMENDMENT_LOG == S.AMENDMENTS_PATH


def test_the_pilots_study_b_runs_wait_for_the_search_record_and_its_study_a_runs_do_not(env, monkeypatch) -> None:
    """Table 9.1 (Q-search-before-pilot): the pilot's two Study B runs (the unconstrained PPO run included) are Study
    B's first runs, held until the complete record is committed; the pilot's Study A runs do not wait for it."""
    import pilot.scheduler as S
    from pilot import manifest

    monkeypatch.setitem(sys.modules, S.SEARCH_MODULE, search_module())
    make, _, _, tmp = env
    repo = tmp / "repo"
    clean_git_repo(repo)
    sched = make(repo_root=repo, allow_dirty=False)

    def held_by_search() -> set[str]:
        sched._refresh_git()  # a new pass: HEAD is read again
        return {s.run_id for s in manifest.pilot() if "search_record_missing" in dict(sched.launch_gates(s))}

    study_b = {s.run_id for s in manifest.pilot() if s.study == "B"}
    assert study_b == {"P-B-unconstrained-s0", "P-B-Moderate-s0"} and held_by_search() == study_b
    commit_files(repo, search_decision("none", complete=False))
    assert held_by_search() == study_b
    commit_files(repo, search_decision("none"))
    assert held_by_search() == set()


def test_schedule_commands_on_a_missing_data_root_create_nothing(tmp_path) -> None:
    from pilot.__main__ import main

    root = tmp_path / "typo"
    for argv in (["status"], ["resolve", S0, "restart"], ["set-policy", "restart"]):
        assert main(["schedule", *argv, "--data-root", str(root)]) == 2
    assert not root.exists()


def test_the_ledger_location_is_fixed_per_data_root(env, tmp_path) -> None:
    make, set_scenarios, _, _ = env
    set_scenarios(**{S0: "interrupt"})
    sched = make()
    sched.add([spec(0)])
    run_to_end(sched)
    elsewhere = LedgerPaths(main=tmp_path / "L" / "ledger.parquet", pilot=tmp_path / "L" / "pilot.parquet",
                            sidecar_dir=tmp_path / "L" / "sidecar")
    other = make(ledger_paths=elsewhere)  # e.g. `set-policy exclude` without the first run's --ledger-dir
    with pytest.raises(SchedulerError, match="--ledger-dir"):
        other.set_policy("exclude")
    with pytest.raises(SchedulerError, match="--ledger-dir"):
        other.run(once=True)
    assert other.row(S0)["status"] == "interrupted" and not (tmp_path / "L").exists()


# -- Study B replacements, restart commits, data-root spellings --------------------------------------


def test_the_operator_replaces_a_study_b_parent_and_its_continuations_follow(env, monkeypatch) -> None:
    from pilot import manifest

    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset({"Q-seed-collision", "Q-jc-window", "Q-budget-normalisation",
                                                            "Q-level-jc", "Q-studyb-order"}))
    make, set_scenarios, _, _ = env
    # Q-continuations stays open, so no continuation launches in this test.
    parent = RunSpec.from_dict({**manifest.study_b((0,))[4].to_dict(), "total_steps": 2_000_000})
    set_scenarios(**{parent.run_id: "crash", "B-Moderate-s5": "crash"})
    sched = make(max_concurrent=2)
    sched.add([parent, *manifest.study_b_fewshot([parent])])
    run_to_end(sched)
    old_conts = [f"B-Moderate-fewshot-b{b:g}-s5" for b in R.UNSEEN_BUDGETS]
    assert sched.row("B-Moderate-s5")["replaced_by"] is None  # the circuit breaker
    assert {sched.row(c)["status"] for c in old_conts} == {"blocked"}
    # `schedule status` offers only what `resolve` accepts (it once offered replace or unblock as well)
    decisions = dict(sched.status_report()["decisions"])
    assert decisions[old_conts[0]] == ("a dependency was excluded, blocked or superseded: `schedule resolve "
                                       "B-Moderate-s5 replace` replaces its parent, which supersedes it")
    with pytest.raises(SchedulerError, match="to repeat it, `schedule resolve B-Moderate-s5 replace`"):
        sched.resolve(old_conts[0], "replace")  # a continuation never takes a seed of its own
    with pytest.raises(SchedulerError, match="resolve B-Moderate-s5 replace"):
        sched.resolve(old_conts[0], "unblock")
    sched.resolve("B-Moderate-s5", "replace")
    assert sched.row("B-Moderate-s5")["replaced_by"] == "B-Moderate-s6"
    assert {sched.row(c)["status"] for c in old_conts} == {"superseded"}
    for b in R.UNSEEN_BUDGETS:
        assert sched.spec(f"B-Moderate-fewshot-b{b:g}-s6").depends_on == ("B-Moderate-s6",)
    assert not any(r["run_id"].endswith("-s7") for r in sched.rows())


def test_an_excluded_continuation_is_left_to_the_group(env, monkeypatch) -> None:
    from pilot import manifest

    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset({"Q-seed-collision", "Q-continuations", "Q-jc-window",
                                                            "Q-budget-normalisation", "Q-level-jc", "Q-studyb-order"}))
    make, set_scenarios, _, _ = env
    parent = RunSpec.from_dict({**manifest.study_b((0,))[4].to_dict(), "total_steps": 2_000_000})
    cont = RunSpec.from_dict({**manifest.study_b_fewshot([parent])[0].to_dict(), "total_steps": 200_000})
    set_scenarios(**{cont.run_id: "crash"})
    sched = make()
    sched.add([parent, cont])
    run_to_end(sched)
    assert sched.row(cont.run_id)["status"] == "excluded"
    assert any("amendment" in (e["detail"] or "") for e in sched.events(cont.run_id) if e["event"] == "needs_decision")
    with pytest.raises(SchedulerError, match="amendment"):
        sched.resolve(cont.run_id, "replace")


def test_a_restart_uses_the_commit_the_interrupted_attempt_ran_on(env, monkeypatch, analysis_at_head) -> None:
    import pilot.scheduler as S

    make, set_scenarios, _, _ = env
    head = {"h": "a" * 40}
    monkeypatch.setattr(S.provenance, "commit_hash", lambda *a, **kw: head["h"])
    monkeypatch.setattr(S.provenance, "dirty_paths", lambda *a, **kw: [])
    # the loaded code is committed (registered ledger writes check it; this checkout may be uncommitted)
    monkeypatch.setattr(S.provenance, "unverified_imported_code", lambda *a, **kw: [])
    set_scenarios(**{S0: "unavailable"})  # queued before its plug-in was committed
    sched = make(allow_dirty=False, on_interrupt="hold")  # the operator restarts the held run
    sched.add([spec(0)])
    run_to_end(sched, passes=5)
    assert sched.row(S0)["status"] == "pending"
    head["h"] = "b" * 40  # the plug-in is committed; attempt 1 really trains on this commit
    monkeypatch.setenv("FAKE_COMMIT", head["h"])
    set_scenarios(**{S0: "sigterm"})
    later = make(allow_dirty=False)  # the next scheduler start
    run_to_end(later, passes=20)
    assert (later.row(S0)["status"], later.row(S0)["launch_commit"]) == ("interrupted", "b" * 40)
    head["h"] = "c" * 40  # HEAD moved on: the restart must wait for b, the commit attempt 1 ran on
    set_scenarios(**{S0: "success"})
    later.resolve(S0, "restart")
    run_to_end(later, passes=5)
    assert later.row(S0)["status"] == "pending" and later.row(S0)["launch_commit"] == "b" * 40
    assert "ran on " + "b" * 40 in [e["detail"] for e in later.events(S0) if e["event"] == "restart_commit_mismatch"][0]
    head["h"] = "b" * 40
    fresh = make(allow_dirty=False)
    run_to_end(fresh)
    assert (fresh.row(S0)["status"], fresh.row(S0)["attempt"]) == ("ledgered", 2)


def test_the_data_root_is_absolute(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert SchedulerConfig(data_root=Path("root")).data_root == tmp_path / "root"


def test_a_live_run_is_found_under_another_spelling_of_the_data_root(env) -> None:
    make, set_scenarios, _, tmp = env
    (tmp / "data").mkdir()
    link = tmp / "link"
    link.symlink_to(tmp / "data", target_is_directory=True)
    set_scenarios(**{S0: "hang"})
    sched = make(data_root=link, on_interrupt="restart")
    sched.add([spec(0)])
    sched.run(once=True)
    pid = sched.row(S0)["pid"]
    fresh = make()  # restarted with the real path of the same data root
    fresh.run(once=True)
    row = fresh.row(S0)
    assert (row["status"], row["attempt"], row["pid"]) == ("training", 1, pid)
    (tmp / "release").write_text("go")
    run_to_end(fresh)
    assert (fresh.row(S0)["status"], fresh.row(S0)["attempt"]) == ("ledgered", 1)


def test_a_live_run_with_relative_arguments_is_found(env) -> None:
    import subprocess

    make, set_scenarios, _, tmp = env
    set_scenarios(**{S0: "hang"})
    sched = make(on_interrupt="restart")
    sched.add([spec(0)])
    run_dir = tmp / "data" / "checkpoints" / S0
    run_dir.mkdir(parents=True)
    (run_dir / "spec.json").write_text(spec(0).to_json(), encoding="utf-8")
    child = subprocess.Popen([sys.executable, str(FAKE), "train", f"{S0}/spec.json", S0], cwd=run_dir.parent)
    with sched.db:
        sched.db.execute("UPDATE runs SET status='training', pid=NULL, host=? WHERE run_id=?", (sched._host, S0))
    fresh = make(on_interrupt="restart")
    fresh.run(once=True)
    assert fresh.row(S0)["pid"] == child.pid and fresh.row(S0)["attempt"] == 1
    (tmp / "release").write_text("go")
    child.wait(timeout=30)
    run_to_end(fresh)
    assert fresh.row(S0)["status"] == "ledgered"


def test_the_ledger_location_accepts_another_spelling_of_the_same_root(env, tmp_path) -> None:
    make, _, _, _ = env
    root = tmp_path / "d3"
    first = make(data_root=root, ledger_paths=None)  # smoke ledgers under the data root
    first.set_policy("hold")
    (tmp_path / "d3link").symlink_to(root)
    again = make(data_root=tmp_path / "d3link", ledger_paths=None)
    again.set_policy("hold")  # the same ledgers through a symlink: accepted
    stored = json.loads(again.db.execute("SELECT value FROM settings WHERE key = 'ledgers'").fetchone()[0])
    assert stored[0] == str((root / "smoke" / "ledger.parquet").resolve())


# -- the policy breaker, the operator's decisions, held ends and ledger retries ----------------------


def test_replacements_stopped_by_the_exclude_policy_again_and_again_wait_for_the_operator(env) -> None:
    make, set_scenarios, _, _ = env
    set_scenarios(**{f"{BASE}-s{s}": "interrupt" for s in [0, *range(5, 20)]})  # e.g. OOM every time
    sched = make(on_interrupt="exclude")
    sched.add([spec(0)])
    run_to_end(sched)
    assert sorted(r["seed"] for r in sched.rows()) == [0, 5, 6]
    last = sched.row(S6)
    assert last["status"] == "excluded" and last["replaced_by"] is None
    assert any("interruption policy" in (e["detail"] or "") for e in sched.events(last["run_id"])
               if e["event"] == "needs_decision")


def test_two_replace_decisions_never_queue_two_seeds(env) -> None:
    make, set_scenarios, _, _ = env
    set_scenarios(**{S0: "crash", S5: "crash"})
    sched = make()
    sched.add([spec(0)])
    run_to_end(sched)
    first, second = make(), make()
    real = first._fix_mode
    calls = []

    def meanwhile(**kw) -> None:
        real(**kw)
        calls.append(S5)
        second.resolve(S5, "replace")  # the other operator command commits first

    first._fix_mode = meanwhile
    with pytest.raises(SchedulerError, match="without a replacement"):
        first.resolve(S5, "replace")  # refused by its own re-check inside the write transaction
    assert calls == [S5]  # the other command ran once and succeeded, so the refusal is first's
    assert [r["run_id"] for r in first.rows() if r["replaces"] == S5] == [S6]


def test_a_surplus_run_keeps_its_arms_open_question_and_waits(env, monkeypatch) -> None:
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", R.ANSWERED_QUESTIONS - {"Q-surplus-arm-set"})  # while it is open
    make, _, _, _ = env
    held = RunSpec.from_dict({**spec(0).to_dict(), "pending": ["Q-surplus-arm-set"]})
    sched = make()
    sched.add([held])
    assert sched.add_surplus(1) == [S5]
    assert "Q-surplus-arm-set" in sched.spec(S5).pending
    run_to_end(sched)
    assert sched.row(S5)["status"] == "pending"
    assert [e["detail"] for e in sched.events(S5) if e["event"] == "open_questions"] == ["Q-surplus-arm-set"]


def test_a_blocked_run_replaced_by_the_operator_is_superseded(env) -> None:
    make, set_scenarios, _, _ = env
    base = spec(0)
    dependent = spec(0, name=DEPENDENT, depends=[base.run_id], group="controller")
    set_scenarios(**{base.run_id: "crash"})
    sched = make(max_concurrent=2)
    sched.add([base, dependent])
    run_to_end(sched)
    sched.resolve(dependent.run_id, "replace")
    run_to_end(sched)
    assert sched.row(dependent.run_id)["status"] == "superseded"
    assert "blocked" not in sched.summary()


def test_a_restart_that_cannot_happen_holds_the_run_and_the_scheduler_goes_on(env) -> None:
    make, set_scenarios, _, tmp = env
    set_scenarios(**{S0: "interrupt", S1: "success"})
    sched = make(on_interrupt="restart")
    sched.add([spec(0), spec(1)])
    (tmp / "data" / "scheduler" / "interrupted" / f"{S0}-attempt1").mkdir(parents=True)  # left by an earlier crash
    run_to_end(sched)
    assert sched.row(S0)["status"] == "interrupted" and sched.row(S0)["attempt"] == 1
    assert any(e["event"] == "restart_failed" for e in sched.events(S0))
    assert sched.row(S1)["status"] == "ledgered"


def test_a_policy_exclusion_interrupted_by_a_scheduler_crash_stays_a_policy_exclusion(env, monkeypatch) -> None:
    make, set_scenarios, _, tmp = env
    set_scenarios(**{S0: "interrupt"})
    sched = make(on_interrupt="exclude")
    sched.add([spec(0)])

    def killed(*a, **kw):
        raise KeyboardInterrupt  # the scheduler dies after rewriting train_result.json, before the transaction commits

    monkeypatch.setattr(sched, "_exclude_sql", killed)
    with pytest.raises(KeyboardInterrupt):
        run_to_end(sched)
    record = json.loads((tmp / "data" / "checkpoints" / S0 / "train_result.json").read_text(encoding="utf-8"))
    assert record["excluded_by_policy"] is True
    fresh = make(on_interrupt="exclude")
    fresh.run(once=True)
    assert fresh.row(S0)["status"] == "excluded"
    assert any(e["event"] == "excluded_by_policy" for e in fresh.events(S0))


def test_evaluation_waits_for_a_clean_worktree(env, monkeypatch, analysis_at_head) -> None:
    import pilot.scheduler as S

    make, set_scenarios, _, tmp = env
    dirty: list[str] = []
    monkeypatch.setattr(S.provenance, "dirty_paths", lambda *a, **kw: list(dirty))
    # the loaded code is committed (registered ledger writes check it; this checkout may be uncommitted)
    monkeypatch.setattr(S.provenance, "unverified_imported_code", lambda *a, **kw: [])
    set_scenarios(**{S0: "hang"})
    sched = make(allow_dirty=False)
    sched.add([spec(0)])
    sched.run(once=True)
    assert sched.row(S0)["status"] == "training"
    dirty.append("pilot/scheduler.py")  # someone edits tracked code while the run trains
    (tmp / "release").write_text("go")
    run_to_end(sched)
    assert sched.row(S0)["status"] == "trained"
    events = [e["event"] for e in sched.events(S0)]
    assert "dirty_worktree" in events and "eval_started" not in events
    dirty.clear()
    run_to_end(sched)
    assert sched.row(S0)["status"] == "ledgered"


@pytest.mark.parametrize("scenario, code", [("interrupt", -9), ("exit1", 1)])
def test_an_end_without_a_final_result_is_held(env, scenario, code) -> None:
    make, set_scenarios, paths, _ = env
    set_scenarios(**{S0: scenario})
    sched = make()
    sched.add([spec(0)])
    run_to_end(sched)
    assert sched.row(S0)["status"] == "interrupted" and not paths.main.exists()
    assert any(f"exit code {code})" in (e["detail"] or "") for e in sched.events(S0) if e["event"] == "interrupted")


def test_retrying_a_recorded_exclusion_says_it_was_already_present(env) -> None:
    make, set_scenarios, _, _ = env
    set_scenarios(**{S0: "crash"})
    sched = make()
    sched.add([spec(0)])
    run_to_end(sched)
    sched.resolve(S0, "retry-ledger")
    events = [e["event"] for e in sched.events(S0)]
    assert events.count("ledger_exclusion_written") == 1 and events[0] == "ledger_exclusion_present"


def test_retrying_an_exclusion_that_cannot_be_written_is_refused(env, monkeypatch) -> None:
    """`schedule resolve RUN retry-ledger` on an excluded run whose exclusion cannot be written raises, so the
    command line never prints `RUN: retry-ledger` and exits 0 with nothing written (the failure was once logged only
    as ledger_write_failed)."""
    import pilot.scheduler as S

    make, set_scenarios, _, _ = env
    set_scenarios(**{S0: "crash"})
    sched = make()
    sched.add([spec(0)])
    run_to_end(sched)
    assert sched.row(S0)["status"] == "excluded"
    sched._set(S0, ledger_written=0)
    monkeypatch.setattr(S, "recorded_anywhere", lambda *a, **kw: False)

    def refuse(*a, **kw):
        raise S.LedgerWriteError("train_result.json cannot be read")

    monkeypatch.setattr(S, "write_run", refuse)
    with pytest.raises(SchedulerError, match="still not in the ledger.*train_result.json cannot be read"):
        sched.resolve(S0, "retry-ledger")
    assert sched.row(S0)["ledger_written"] == 0
    monkeypatch.setattr(sched, "_ledger_code_problem", lambda run_id: "tracked code has uncommitted changes")
    with pytest.raises(SchedulerError, match="uncommitted changes"):
        sched.resolve(S0, "retry-ledger")


def test_a_refused_decision_fixes_no_ledger_location(tmp_path, monkeypatch) -> None:
    """`set-policy restart|exclude` (refused, exit 3, while Q-interrupted-run is open) and `resolve` (refused by its
    action's gate or the run's status) fix neither the data root's ledger location nor its registered `.mode`
    markers, so a later `schedule run` with another --ledger-dir is not refused."""
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", R.ANSWERED_QUESTIONS - {"Q-interrupted-run"})  # while it is open
    root = tmp_path / "data"
    queued = Scheduler(SchedulerConfig(data_root=root))  # as `schedule add`: the mode is fixed, the ledgers not
    queued.fix_mode(ledgers=False)
    queued.add([spec(0)])
    queued.db.close()
    ledgers = tmp_path / "L"
    paths = LedgerPaths(main=ledgers / "ledger.parquet", pilot=ledgers / "pilot.parquet",
                        sidecar_dir=ledgers / "sidecar")
    sched = Scheduler(SchedulerConfig(data_root=root, ledger_paths=paths))
    for policy in ("restart", "exclude"):
        with pytest.raises(PendingQuestionError):
            sched.set_policy(policy)
    for action in ("restart", "add-auxiliary", "reevaluate", "unblock"):
        with pytest.raises((SchedulerError, PendingQuestionError)):
            sched.resolve(S0, action)
    assert sched.setting("ledgers") is None and not ledgers.exists()
    assert [e["event"] for e in sched.events(None) if e["run_id"] is None] == ["mode"]


@pytest.mark.parametrize("action", [["add", "--design", "pilot"], ["add-surplus", "--extra", "1"], ["status"]])
def test_commands_that_write_no_ledger_take_no_ledger_dir(tmp_path, action) -> None:
    from pilot.__main__ import main

    with pytest.raises(SystemExit) as exc:
        main(["schedule", *action, "--data-root", str(tmp_path / "root"), "--ledger-dir", str(tmp_path / "L")])
    assert exc.value.code == 2
    assert not (tmp_path / "root").exists()


def test_surplus_seeds_of_the_ramp_arms_wait_on_the_surplus_arm_set_question(env, monkeypatch) -> None:
    from pilot import manifest

    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset({"Q-cost-critic", "Q-jc-window", "Q-ramp-step",
                                                            "Q-plasticity-definitions"}))
    make, _, _, _ = env
    arms = [s for s in manifest.study_a_main((0,)) if s.task == R.PRIMARY_TASK and s.N == 0.50]
    sched = make(max_concurrent=len(arms))
    sched.add(arms)
    added = sched.add_surplus(1)
    assert sorted(added) == sorted(f"{s.arm_id}-s5" for s in arms)
    for s in arms:  # only the surplus runs are under test here
        with sched.db:
            sched.db.execute("UPDATE runs SET status='ledgered' WHERE run_id=?", (s.run_id,))
    run_to_end(sched)
    for rid in added:
        if "-ramp-" in rid:  # Part 5.5 names no onset shape: the ramp arms wait for the group's answer
            assert "Q-surplus-arm-set" in sched.spec(rid).pending and sched.row(rid)["status"] == "pending"
            assert [e["detail"] for e in sched.events(rid) if e["event"] == "open_questions"] == ["Q-surplus-arm-set"]
        else:
            assert "Q-surplus-arm-set" not in sched.spec(rid).pending and sched.row(rid)["status"] == "ledgered"


# -- cut-short restarts and exclusions, concurrent operator commands ---------------------------------


def test_a_restart_cut_short_after_its_archive_move_is_completed_at_the_next_start(env, monkeypatch,
                                                                                   analysis_at_head) -> None:
    import shutil

    import pilot.scheduler as S

    make, set_scenarios, _, tmp = env
    head = {"h": "a" * 40}
    monkeypatch.setattr(S.provenance, "commit_hash", lambda *a, **kw: head["h"])
    monkeypatch.setattr(S.provenance, "dirty_paths", lambda *a, **kw: [])
    # the loaded code is committed (registered ledger writes check it; this checkout may be uncommitted)
    monkeypatch.setattr(S.provenance, "unverified_imported_code", lambda *a, **kw: [])
    # the stand-in commits a and b differ in no code this process loaded (a ledger write after HEAD moved checks it)
    monkeypatch.setattr(S.provenance, "imported_code_changed_between", lambda *a, **kw: [])
    set_scenarios(**{S0: "interrupt"})
    sched = make(on_interrupt="restart", allow_dirty=False)
    sched.add([spec(0)])
    real_move = shutil.move

    def move_then_die(src, dst):
        real_move(src, dst)
        raise KeyboardInterrupt  # the scheduler is killed after the move, before the state commit

    monkeypatch.setattr(S.shutil, "move", move_then_die)
    with pytest.raises(KeyboardInterrupt):
        run_to_end(sched)
    monkeypatch.setattr(S.shutil, "move", real_move)
    assert (sched.row(S0)["status"], sched.row(S0)["attempt"]) == ("training", 1)
    head["h"] = "b" * 40  # HEAD moved on meanwhile: the restart must wait for a, the commit attempt 1 ran on
    set_scenarios(**{S0: "success"})
    fresh = make(on_interrupt="restart", allow_dirty=False)
    run_to_end(fresh, passes=5)
    events = [e["event"] for e in fresh.events(S0)]
    assert "never_started" not in events and "restart_resumed" in events and "restart" in events
    row = fresh.row(S0)
    assert (row["status"], row["attempt"], row["launch_commit"]) == ("pending", 2, "a" * 40)
    assert (tmp / "data" / "scheduler" / "interrupted" / f"{S0}-attempt1").is_dir()
    head["h"] = "a" * 40
    run_to_end(fresh)
    assert (fresh.row(S0)["status"], fresh.row(S0)["attempt"]) == ("ledgered", 2)


def test_a_restart_that_fails_with_an_os_error_holds_the_run_and_the_scheduler_goes_on(env, monkeypatch) -> None:
    import pilot.scheduler as S

    make, set_scenarios, _, _ = env
    set_scenarios(**{S0: "interrupt", S1: "success"})
    sched = make(on_interrupt="restart")
    sched.add([spec(0), spec(1)])

    def no_space(src, dst):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(S.shutil, "move", no_space)
    run_to_end(sched)
    assert (sched.row(S0)["status"], sched.row(S0)["attempt"]) == ("interrupted", 1)
    assert any(e["event"] == "restart_failed" and "No space" in e["detail"] for e in sched.events(S0))
    assert sched.row(S1)["status"] == "ledgered"


def test_an_operator_restart_never_archives_an_attempt_launched_meanwhile(env) -> None:
    make, set_scenarios, _, tmp = env
    set_scenarios(**{S0: "interrupt"})
    sched = make()  # hold
    sched.add([spec(0)])
    run_to_end(sched)
    assert sched.row(S0)["status"] == "interrupted"
    op = make()
    real_fix = op._fix_mode

    def meanwhile(**kw):
        real_fix(**kw)
        make().set_policy("restart")  # another operator carries out the group's decision
        set_scenarios(**{S0: "success"})
        sched.run(once=True)  # the scheduler launches attempt 2 ...
        sched._procs[S0].wait(timeout=30)  # ... which completes before the scheduler polls it

    op._fix_mode = meanwhile
    with pytest.raises(SchedulerError, match="only an interrupted run"):
        op.resolve(S0, "restart")  # checked 'interrupted' before; re-checked inside the transaction
    assert not (tmp / "data" / "scheduler" / "interrupted" / f"{S0}-attempt2").exists()
    run_to_end(sched)
    assert (sched.row(S0)["status"], sched.row(S0)["attempt"]) == ("ledgered", 2)


def test_one_outage_stopping_unrelated_replacements_does_not_trip_the_policy_breaker(env) -> None:
    import os
    import signal

    make, set_scenarios, _, tmp = env
    s2 = f"{BASE}-s2"
    set_scenarios(**{S1: "crash", s2: "nonfinite", S5: "hang", S6: "hang"})
    sched = make(max_concurrent=3)
    sched.add([spec(1), spec(2)])
    for _ in range(200):  # s5 replaces a crash of s1, s6 a non-finite multiplier of s2; both claim and train
        sched.run(once=True)
        if all(sched.exists(r) and (tmp / "data" / "checkpoints" / r / ".claim").exists() for r in (S5, S6)):
            break
        time.sleep(0.05)
    for rid in (S5, S6):  # one outage stops both
        os.kill(sched.row(rid)["pid"], signal.SIGKILL)
    for _ in range(200):
        sched.run(once=True)
        if {sched.row(r)["status"] for r in (S5, S6)} == {"interrupted"}:
            break
        time.sleep(0.05)
    sched.set_policy("exclude")
    assert sched.row(S5)["replaced_by"] == f"{BASE}-s7"
    assert sched.row(S6)["replaced_by"] == f"{BASE}-s8"
    assert not any(e["event"] == "needs_decision" for e in sched.events(S6))


def test_a_set_policy_exclude_cut_short_is_completed_at_the_next_start(env, monkeypatch) -> None:
    make, set_scenarios, paths, tmp = env
    set_scenarios(**{S0: "interrupt"})
    sched = make()
    sched.add([spec(0)])
    run_to_end(sched)
    assert sched.row(S0)["status"] == "interrupted"
    real = sched._exclude_sql

    def killed(*a, **kw):
        raise KeyboardInterrupt  # the operator's set-policy dies after rewriting train_result.json

    monkeypatch.setattr(sched, "_exclude_sql", killed)
    with pytest.raises(KeyboardInterrupt):
        sched.set_policy("exclude")
    monkeypatch.setattr(sched, "_exclude_sql", real)
    fresh = make()
    run_to_end(fresh)
    assert fresh.row(S0)["status"] == "excluded" and fresh.row(S0)["replaced_by"] == S5
    assert any(e["event"] == "excluded_by_policy" for e in fresh.events(S0))
    record = json.loads((tmp / "data" / "checkpoints" / S0 / "train_result.json").read_text(encoding="utf-8"))
    assert record["written_by"] == "scheduler (the process left no final result)"
    assert record["excluded_by_policy"] is True
    assert {r.run_id: r.failure_cause for r in ledger_rows(paths.main)}[S0] == "incomplete"


def test_unblock_and_replace_decisions_never_both_apply(env) -> None:
    make, _, _, _ = env
    base = spec(0)
    dependent = spec(0, name=DEPENDENT, depends=[base.run_id], group="controller")
    sched = make()
    sched.add([base, dependent])
    with sched.db:  # blocked, with its dependency fine again (e.g. blocked by an earlier version)
        sched.db.execute("UPDATE runs SET status='blocked' WHERE run_id=?", (dependent.run_id,))
    first, second = make(), make()
    real = first._fix_mode

    def meanwhile(**kw):
        real(**kw)
        second.resolve(dependent.run_id, "replace")  # another operator's decision commits first

    first._fix_mode = meanwhile
    with pytest.raises(SchedulerError, match="without a replacement can be unblocked"):
        first.resolve(dependent.run_id, "unblock")
    assert first.row(dependent.run_id)["status"] == "superseded"


# -- the policy leaves a run another operator restarted as it is -------------------------------------


def held_then_restarted_and_launched_meanwhile(env, hook_owner: Scheduler, name: str) -> Scheduler:
    """A held S0; while ``hook_owner`` is about to act on it, another operator restarts it and the scheduler
    launches attempt 2, which hangs (alive) until released."""
    make, set_scenarios, _, _ = env
    set_scenarios(**{S0: "interrupt"})
    sched = make()  # hold
    sched.add([spec(0)])
    run_to_end(sched)
    assert sched.row(S0)["status"] == "interrupted"
    set_scenarios(**{S0: "hang"})
    real = getattr(hook_owner, name)

    def meanwhile(*a, **kw):
        make().resolve(S0, "restart")  # another operator restarts the held run first ...
        sched.run(once=True)  # ... and the scheduler launches attempt 2
        return real(*a, **kw)

    setattr(hook_owner, name, meanwhile)
    return sched


def test_set_policy_restart_leaves_a_run_another_operator_restarted_as_it_is(env) -> None:
    make, _, _, tmp = env
    op = make()
    sched = held_then_restarted_and_launched_meanwhile(env, op, "restart")
    try:
        op.set_policy("restart")  # checked 'interrupted' from its snapshot; re-checked inside the transaction
        row = op.row(S0)
        assert (row["status"], row["attempt"], row["pid"]) == ("training", 2, sched._procs[S0].pid)
        assert sched._procs[S0].poll() is None
        assert [e["event"] for e in op.events(S0)][0] == "restart_skipped"
    finally:
        (tmp / "release").write_text("go")
    run_to_end(sched)
    assert (sched.row(S0)["status"], sched.row(S0)["attempt"]) == ("ledgered", 2)
    assert not (tmp / "data" / "scheduler" / "interrupted" / f"{S0}-attempt2").exists()


def test_set_policy_exclude_never_excludes_a_run_another_operator_restarted(env) -> None:
    make, _, paths, tmp = env
    op = make()
    sched = held_then_restarted_and_launched_meanwhile(env, op, "_exclude_without_result")
    try:
        op.set_policy("exclude")
        row = op.row(S0)
        assert (row["status"], row["attempt"], row["replaced_by"]) == ("training", 2, None)
        # the live run's directory is untouched
        assert not (tmp / "data" / "checkpoints" / S0 / "train_result.json").exists()
        assert not op.exists(S5)
        assert [e["event"] for e in op.events(S0)][0] == "exclusion_skipped"
    finally:
        (tmp / "release").write_text("go")
    run_to_end(sched)
    assert (sched.row(S0)["status"], sched.row(S0)["attempt"]) == ("ledgered", 2)
    assert [(r.run_id, r.completed) for r in ledger_rows(paths.main)] == [(S0, True)]


# -- operator decisions re-checked inside the write transaction --------------------------------------


def test_two_reevaluate_decisions_never_launch_two_evaluations(env) -> None:
    make, set_scenarios, _, _ = env
    set_scenarios(**{S0: "evalfail"})
    sched = make(max_concurrent=1)
    sched.add([spec(0)])
    run_to_end(sched)
    assert sched.row(S0)["status"] == "eval_failed"
    set_scenarios(**{S0: "success"})
    op1, op2 = make(), make()
    real = op2._fix_mode

    def meanwhile(**kw) -> None:
        real(**kw)
        op1.resolve(S0, "reevaluate")  # another operator carries out the same decision first
        sched.run(once=True)  # and the scheduler launches the evaluation

    op2._fix_mode = meanwhile
    with pytest.raises(SchedulerError, match="only an eval_failed or ledger_failed run can be re-evaluated"):
        op2.resolve(S0, "reevaluate")
    run_to_end(sched)
    assert sched.row(S0)["status"] == "ledgered"
    assert [e["event"] for e in sched.events(S0)].count("eval_started") == 2  # the failed one and one retry


# -- the smoke script and the commands that queue runs -----------------------------------------------


def load_smoke_run() -> types.ModuleType:
    """scripts/smoke_run.py, loaded as a module."""
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "smoke_run.py"
    module_spec = importlib.util.spec_from_file_location("smoke_run", path)
    smoke_run = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(smoke_run)
    return smoke_run


def test_smoke_script_refuses_a_registered_data_root_before_queueing(tmp_path, capsys) -> None:
    root = tmp_path / "registered"
    Scheduler(SchedulerConfig(data_root=root)).fix_mode()  # a registered data root
    assert load_smoke_run().main(["--data-root", str(root)]) == 1
    assert "registered runs" in capsys.readouterr().err
    assert Scheduler(SchedulerConfig(data_root=root)).rows() == []  # nothing was queued


def test_smoke_script_refuses_a_data_root_with_queued_registered_runs(tmp_path, capsys, monkeypatch) -> None:
    from pilot import manifest

    # were the refusal to regress, fail at once instead of launching the queued 10M-step runs
    monkeypatch.setattr(Scheduler, "run", lambda self, **kw: pytest.fail("smoke_run.py launched runs"))

    root = tmp_path / "queued"
    Scheduler(SchedulerConfig(data_root=root)).add(manifest.design("pilot"))  # Scheduler.add fixes no mode
    assert load_smoke_run().main(["--data-root", str(root)]) == 1
    assert "holds 8 registered runs" in capsys.readouterr().err
    after = Scheduler(SchedulerConfig(data_root=root))
    assert after.setting("mode") is None and all(r["status"] == "pending" for r in after.rows())


def test_the_first_mode_is_fixed_under_the_write_lock(tmp_path) -> None:
    root = tmp_path / "fresh"
    smoke = Scheduler(SchedulerConfig(data_root=root, allow_dirty=True))
    registered = Scheduler(SchedulerConfig(data_root=root))
    real = smoke.setting

    def setting_then_registered(key):  # the registered command tries to fix the mode meanwhile
        value = real(key)
        if key == "ledgers":
            with pytest.raises(sqlite3.OperationalError, match="locked"):  # blocked by smoke's write lock
                registered.db.execute("BEGIN IMMEDIATE")
        return value

    registered.db.execute("PRAGMA busy_timeout = 0")
    smoke.setting = setting_then_registered
    smoke.fix_mode()
    with pytest.raises(SchedulerError, match="holds smoke runs"):
        registered.fix_mode()


def test_schedule_add_refuses_a_smoke_data_root(tmp_path, capsys, monkeypatch) -> None:
    from pilot.__main__ import main

    clean_tree(monkeypatch)
    root = tmp_path / "smoke"
    Scheduler(SchedulerConfig(data_root=root, allow_dirty=True, allow_pending=True)).fix_mode()
    assert main(["schedule", "add", "--design", "pilot", "--data-root", str(root)]) == 2
    assert "smoke data root" in capsys.readouterr().err
    assert Scheduler(SchedulerConfig(data_root=root, allow_dirty=True)).rows() == []


@pytest.mark.parametrize("action", [["add", "--design", "pilot"], ["add-surplus", "--extra", "1"],
                                    ["add-continuations", "--ledger", "x.parquet"]])
def test_a_refused_queueing_command_creates_no_scheduler_state(tmp_path, capsys, monkeypatch, action) -> None:
    """Exit 2 is "a usage error or a refused provenance check (a CliError)" (pilot/__main__.py, EXIT_USAGE): on a
    dirty tree the queueing actions refuse before they create checkpoints/, scheduler/logs or state.sqlite on the
    data root."""
    from pilot import provenance
    from pilot.__main__ import main

    monkeypatch.setattr(provenance, "require_clean_worktree", lambda *a, **kw: (_ for _ in ()).throw(
        provenance.DirtyWorktreeError("dirty")))
    monkeypatch.setattr(provenance, "dirty_paths", lambda *a, **kw: ["pilot/x.py"])
    root = tmp_path / "typo"
    assert main(["schedule", *action, "--data-root", str(root)]) == 2
    assert "uncommitted changes" in capsys.readouterr().err and not root.exists()


def test_a_replacement_queued_by_the_command_line_logs_the_commit_of_its_code(tmp_path, monkeypatch) -> None:
    """`schedule resolve RUN replace` reads HEAD first, so the replacement's `queued` event names its commit, never
    'code at commit None' (HEAD was once read only by a scheduling pass)."""
    from pilot.__main__ import main

    clean_tree(monkeypatch, commit="e" * 40)
    root = tmp_path / "root"
    assert main(["schedule", "add", "--design", "pilot", "--data-root", str(root)]) == 0
    victim = Scheduler(SchedulerConfig(data_root=root)).rows()[0]["run_id"]
    Scheduler(SchedulerConfig(data_root=root))._set(victim, status="excluded")
    assert main(["schedule", "resolve", victim, "replace", "--data-root", str(root)]) == 0
    sched = Scheduler(SchedulerConfig(data_root=root))
    queued = [e["detail"] for e in sched.events(sched.row(victim)["replaced_by"]) if e["event"] == "queued"]
    assert queued == [f"replacement of {victim} (code at commit {'e' * 40})"]


def test_schedule_add_fixes_the_root_as_registered_so_smoke_flags_are_refused(tmp_path, capsys, monkeypatch) -> None:
    from pilot.__main__ import main

    clean_tree(monkeypatch)

    def run_without_training(self, **kw):  # run() checks the mode first; it must never get further here
        self.fix_mode()
        pytest.fail("launched the queued registered runs")

    monkeypatch.setattr(Scheduler, "run", run_without_training)
    root = tmp_path / "queued"
    assert main(["schedule", "add", "--design", "pilot", "--data-root", str(root)]) == 0
    capsys.readouterr()
    for argv in (["run", "--max-concurrent", "1"], ["set-policy", "hold"], ["resolve", "P-B-Moderate-s0", "restart"]):
        assert main(["schedule", *argv, "--data-root", str(root), "--allow-dirty", "--allow-pending"]) != 0
        assert "holds registered runs" in capsys.readouterr().err
    assert Scheduler(SchedulerConfig(data_root=root)).setting("mode") == "registered"


def test_schedule_add_leaves_the_ledger_location_to_the_first_run(tmp_path, monkeypatch) -> None:
    from pilot.__main__ import main

    clean_tree(monkeypatch)

    before, after, ledgers = tmp_path / "run_first", tmp_path / "add_first", str(tmp_path / "L")
    # a registered data root whose first run chose --ledger-dir can still queue runs (add takes no --ledger-dir)
    Scheduler(SchedulerConfig(data_root=before, ledger_paths=LedgerPaths(
        main=tmp_path / "L" / "ledger.parquet", pilot=tmp_path / "L" / "pilot_ledger.parquet",
        sidecar_dir=tmp_path / "L" / "sidecar"))).fix_mode()
    assert main(["schedule", "add", "--design", "pilot", "--data-root", str(before)]) == 0
    assert main(["schedule", "add-surplus", "--extra", "1", "--data-root", str(before)]) == 0
    # and `schedule add` first leaves the choice to the first run
    assert main(["schedule", "add", "--design", "pilot", "--data-root", str(after)]) == 0
    assert Scheduler(SchedulerConfig(data_root=after)).setting("ledgers") is None
    assert main(["schedule", "set-policy", "hold", "--data-root", str(after), "--ledger-dir", ledgers]) == 0
    assert Scheduler(SchedulerConfig(data_root=after)).setting("mode") == "registered"


@pytest.mark.parametrize("argv", [["add", "--design", "pilot"], ["add-surplus", "--extra", "1"]])
def test_queueing_commands_refuse_uncommitted_code_and_record_the_commit(tmp_path, capsys, monkeypatch, argv) -> None:
    """The stored spec is what the launcher trains (it never re-derives it from the design), so a registered run is
    never queued from uncommitted code (pilot/manifest.py, configs/registered.py) that its recorded commit does not
    contain; the commit that computed the queued runs is logged."""
    from pilot import provenance
    from pilot.__main__ import main

    root = tmp_path / "root"
    clean_tree(monkeypatch)
    monkeypatch.setattr(provenance, "dirty_paths", lambda *a, **kw: ["pilot/manifest.py"])
    assert main(["schedule", *argv, "--data-root", str(root)]) == 2
    err = capsys.readouterr().err
    assert "uncommitted changes" in err and "pilot/manifest.py" in err
    assert Scheduler(SchedulerConfig(data_root=root)).rows() == []
    monkeypatch.setattr(provenance, "dirty_paths", lambda *a, **kw: [])
    monkeypatch.setattr(provenance, "unverified_imported_code",
                        lambda *a, **kw: ["pilot/manifest_extra.py (untracked)"])
    assert main(["schedule", *argv, "--data-root", str(root)]) == 2
    assert "manifest_extra.py" in capsys.readouterr().err
    assert Scheduler(SchedulerConfig(data_root=root)).rows() == []
    clean_tree(monkeypatch, commit="d" * 40)
    assert main(["schedule", "add", "--design", "pilot", "--data-root", str(root)]) == 0
    assert main(["schedule", *argv, "--data-root", str(root)]) == 0
    queued = [e for e in Scheduler(SchedulerConfig(data_root=root)).events(limit=1000) if e["event"] == "queued"]
    assert queued and all(f"code at commit {'d' * 40}" in e["detail"] for e in queued)


def test_a_warm_started_replacement_whose_n0_seed_is_not_queued_is_a_decision(env, monkeypatch) -> None:
    """A warm-started replacement (seed 5) depends on the N = 0 run of seed 5, which the N = 0 arm never received
    (no surplus seed; on the other tasks it never would). `schedule status` names the decision instead of the run
    waiting silently, and `schedule resolve RUN add-auxiliary` (Table 9.1, Q-warm-start, refused while it is
    open) queues that N = 0 run, whose ledger row is marked auxiliary (manifest.AUXILIARY_NOTE) so it is not a seed
    of its arm."""
    from dataclasses import replace

    from pilot import manifest

    make, set_scenarios, paths, _ = env
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset({"Q-seed-collision"}))
    base = spec(0)
    warm = replace(spec(0, name="T-A-PointGoal1-N0.10-abrupt-total-warm_started", depends=[base.run_id],
                           group="controller"), controller_variant="warm_started")
    set_scenarios(**{warm.run_id: "crash"})
    sched = make(max_concurrent=2)
    sched.add([base, warm])
    run_to_end(sched)
    repl = sched.row(warm.run_id)["replaced_by"]
    dep = sched.spec(repl).depends_on[0]
    assert dep == f"{base.arm_id}-s5" and sched.row(repl)["status"] == "pending" and not sched.exists(dep)
    decisions = dict(sched.status_report()["decisions"])
    assert decisions[repl] == (f"dependency {dep} not queued on this data root: queue its auxiliary N = 0 run: "
                               f"`schedule resolve {repl} add-auxiliary` (Q-warm-start)")  # Table 9.1's rule
    with pytest.raises(PendingQuestionError, match="Q-warm-start"):  # refused while it is open (command line: exit 3)
        sched.resolve(repl, "add-auxiliary")
    assert not sched.exists(dep)
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset({"Q-seed-collision", "Q-warm-start"}))
    with pytest.raises(SchedulerError, match="warm-started"):
        sched.resolve(base.run_id, "add-auxiliary")
    sched.resolve(repl, "add-auxiliary")
    assert sched.exists(dep) and sched.is_auxiliary(dep) and not sched.is_auxiliary(base.run_id)
    assert sched.spec(dep) == sched._with_seed_rule(base.with_seed(5))  # the N = 0 arm's run of that seed, unchanged
    assert repl not in dict(sched.status_report()["decisions"])
    with pytest.raises(SchedulerError, match="nothing to add"):
        sched.resolve(repl, "add-auxiliary")
    assert sched.next_unused_seed(base.arm_id) == 6  # a later N = 0 replacement never collides with it
    run_to_end(sched)
    assert sched.row(dep)["status"] == "ledgered" and sched.row(repl)["status"] == "ledgered"
    rows = {r.run_id: r for r in ledger_rows(paths.main)}
    assert manifest.is_auxiliary_row(rows[dep].model_dump())
    assert not any(manifest.is_auxiliary_row(r.model_dump()) for k, r in rows.items() if k != dep)


def test_an_excluded_auxiliary_run_is_not_replaced_and_its_exclusion_row_is_marked(env, monkeypatch) -> None:
    """An auxiliary N = 0 run serves only the warm-started run of its own seed; excluding it queues no next unused
    seed of the N = 0 arm (which would become an extra seed of that arm and serve no one): Table 9.1 (Q-warm-start)
    never repeats or replaces it, `schedule resolve RUN replace` refuses it, its exclusion row carries
    manifest.AUXILIARY_NOTE, and its warm-started dependent is superseded by that arm's next unused seed."""
    from dataclasses import replace

    from pilot import manifest

    make, set_scenarios, paths, _ = env
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset({"Q-seed-collision", "Q-warm-start"}))
    base = spec(0)
    warm = replace(spec(0, name="T-A-PointGoal1-N0.10-abrupt-total-warm_started", depends=[base.run_id],
                           group="controller"), controller_variant="warm_started")
    set_scenarios(**{warm.run_id: "crash"})
    sched = make(max_concurrent=2)
    sched.add([base, warm])
    run_to_end(sched)
    repl = sched.row(warm.run_id)["replaced_by"]
    dep = sched.spec(repl).depends_on[0]
    set_scenarios(**{dep: "crash"})
    sched.resolve(repl, "add-auxiliary")
    run_to_end(sched)
    row = sched.row(dep)
    assert row["status"] == "excluded" and row["replaced_by"] is None and row["ledger_written"]
    assert [r["run_id"] for r in sched.rows() if r["arm_id"] == base.arm_id] == [base.run_id, dep]  # no new N = 0 seed
    assert sched.row(repl)["status"] == "blocked"
    decisions = dict(sched.status_report()["decisions"])
    assert decisions[dep] == f"an excluded auxiliary run: {AUXILIARY_EXCLUDED}"
    assert "never repeated or replaced" in decisions[dep] and "schedule resolve DEPENDENT replace" in decisions[dep]
    assert "Q-warm-start" in decisions[repl] and f"`schedule resolve {repl} replace`" in decisions[repl]
    assert any(e["event"] == "needs_decision" and "auxiliary" in (e["detail"] or "") for e in sched.events(limit=1000)
               if e["run_id"] == dep)
    with pytest.raises(SchedulerError, match="never repeated or replaced"):
        sched.resolve(dep, "replace")
    rows = {r.run_id: r for r in ledger_rows(paths.main)}
    assert rows[dep].completed is False and manifest.is_auxiliary_row(rows[dep].model_dump())
    assert not manifest.is_auxiliary_row(rows[base.run_id].model_dump())


def test_an_excluded_auxiliary_run_counts_towards_no_breaker_of_its_arm(env) -> None:
    """An auxiliary N = 0 run is not a seed of its arm (Table 9.1, Q-warm-start; manifest.AUXILIARY_NOTE): its
    exclusion counts neither towards the N = 0 arm's same-cause circuit breaker nor towards its seed-target stop, so
    the first crash of a real N = 0 seed after it is still replaced at once."""
    from pilot.scheduler import AUXILIARY_EVENT

    make, set_scenarios, _, _ = env
    set_scenarios(**{S5: "crash", S1: "crash"})
    sched = make()
    with sched._write_txn():  # queued as `schedule resolve RUN add-auxiliary` queues it (_add_auxiliary), first
        sched._insert_sql(Scheduler._with_seed_rule(spec(5)))
        sched._event_sql(S5, AUXILIARY_EVENT, "for a warm-started run of seed 5")
    sched.add([spec(1)])
    run_to_end(sched)
    assert (sched.row(S5)["status"], sched.row(S5)["failure_cause"], sched.row(S5)["replaced_by"]) == (
        "excluded", "crash", None)  # never replaced itself
    assert sched.row(S1)["status"] == "excluded" and sched.row(S1)["replaced_by"] == S6  # no breaker
    assert sched.row(S6)["status"] == "ledgered"
    assert sched._same_cause_exclusions(spec(1).arm_id, "crash") == 1


def test_resolve_decisions_that_queue_runs_refuse_uncommitted_code_and_record_the_commit(tmp_path, capsys,
                                                                                         monkeypatch) -> None:
    """`resolve replace` (a replacement and its few-shot continuations) and `resolve add-auxiliary` queue specs
    computed by the loaded code, as `schedule add` does; they need a clean, committed tree and log the commit.
    `add-auxiliary` with Q-warm-start open is refused as an open question (exit 3)."""
    from pilot import provenance
    from pilot.__main__ import main

    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", R.ANSWERED_QUESTIONS - {"Q-warm-start"})
    root = tmp_path / "root"
    clean_tree(monkeypatch, commit="e" * 40)
    assert main(["schedule", "add", "--design", "pilot", "--data-root", str(root)]) == 0
    sched = Scheduler(SchedulerConfig(data_root=root))
    victim = sched.rows()[0]["run_id"]
    sched._set(victim, status="excluded")  # excluded without a replacement (the circuit breaker's case)
    capsys.readouterr()
    monkeypatch.setattr(provenance, "dirty_paths", lambda *a, **kw: ["pilot/manifest.py"])
    assert main(["schedule", "resolve", victim, "replace", "--data-root", str(root)]) == 2
    assert "uncommitted changes" in capsys.readouterr().err
    assert Scheduler(SchedulerConfig(data_root=root)).row(victim)["replaced_by"] is None
    monkeypatch.setattr(provenance, "dirty_paths", lambda *a, **kw: [])
    monkeypatch.setattr(provenance, "unverified_imported_code", lambda *a, **kw: ["pilot/extra.py (untracked)"])
    assert main(["schedule", "resolve", victim, "replace", "--data-root", str(root)]) == 2
    assert "extra.py" in capsys.readouterr().err
    clean_tree(monkeypatch, commit="e" * 40)
    assert main(["schedule", "resolve", victim, "add-auxiliary", "--data-root", str(root)]) == 3
    assert "Q-warm-start" in capsys.readouterr().err
    assert main(["schedule", "resolve", victim, "replace", "--data-root", str(root)]) == 0
    sched = Scheduler(SchedulerConfig(data_root=root))
    assert sched.row(victim)["replaced_by"]
    queued = [e for e in sched.events(limit=1000) if e["event"] == "queued"]
    assert any(f"resolve {victim} replace" in e["detail"] and f"code at commit {'e' * 40}" in e["detail"]
               for e in queued)


def test_a_run_gate_added_after_queueing_holds_the_queued_run(env, monkeypatch, tmp_path) -> None:
    """The scheduler and the launcher hold a registered run on the run gates the current code derives
    (manifest.open_run_gates), not only those frozen into the stored spec, so a run gate added (or an answer
    changed) after `schedule add` holds the queued run."""
    import pilot.scheduler as S
    from pilot import launch, manifest
    from pilot.errors import RunRefused

    make, _, _, _ = env
    monkeypatch.setattr(S.provenance, "dirty_paths", lambda *a, **kw: [])
    sched = make(allow_dirty=False)
    spec_ = next(s for s in manifest.design("pilot") if s.plugin == "unconstrained_ppo")
    sched.add([spec_])
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset(spec_.pending))
    sched._refresh_git()
    assert sched._launch_blocker(sched.row(spec_.run_id)) != "open_questions"
    with_new_gate(monkeypatch, spec_.run_id)
    assert sched.spec(spec_.run_id).open_questions == ()  # the stored tuple alone would let it launch
    assert sched._launch_blocker(sched.row(spec_.run_id)) == "open_questions"
    with pytest.raises(RunRefused, match="Q-cost-critic"):
        launch.train(sched.spec(spec_.run_id), tmp_path / "run")
    assert not (tmp_path / "run").exists()
    # a smoke launch, a test's spec and a determinism check keep their stored keys only
    assert manifest.open_run_gates(sched.spec(spec_.run_id), registered=False) == ()
    assert manifest.open_run_gates(spec(0)) == ()


def test_requeue_stores_the_current_spec_of_a_run_that_never_trained(env, monkeypatch) -> None:
    """`schedule resolve RUN requeue` stores the current design's spec of a pending run that never trained, which no
    other command can do (`schedule add` refuses a changed spec)."""
    from dataclasses import replace

    import pilot.scheduler as S
    from pilot import __main__ as cli
    from pilot import manifest

    assert cli.RESOLVE_CHOICES == S.RESOLVE_ACTIONS and "requeue" in cli.QUEUEING_DECISIONS
    make, _, _, _ = env
    sched = make(allow_dirty=False)
    pilot_specs = manifest.design("pilot")
    sched.add(pilot_specs)
    target = next(s for s in pilot_specs if s.plugin == "unconstrained_ppo")
    for queued in pilot_specs:  # unchanged code rebuilds every queued spec as it is
        assert sched.current_spec(queued.run_id) == queued
    with pytest.raises(SchedulerError, match="already the current design's"):
        sched.resolve(target.run_id, "requeue")
    real = manifest.pilot
    monkeypatch.setattr(manifest, "pilot", lambda revision=0: [
        replace(s, pending=(*s.pending, "Q-cost-critic")) if s.run_id == target.run_id else s for s in real(revision)])
    changed = sched.current_spec(target.run_id)
    assert changed.pending == (*target.pending, "Q-cost-critic")
    sched.resolve(target.run_id, "requeue")
    assert sched.spec(target.run_id) == changed and sched.row(target.run_id)["status"] == "pending"
    assert any(e["event"] == S.REQUEUE_EVENT and "Q-cost-critic" in e["detail"] for e in sched.events(target.run_id))
    # a run that started (output in its run directory) or ran before is never changed
    other = next(s for s in pilot_specs if s.plugin == "study_b")
    monkeypatch.setattr(manifest, "pilot", lambda revision=0: [
        replace(s, pending=(*s.pending, "Q-cost-critic")) for s in real(revision)])
    run_dir = sched.cfg.run_dir(other.run_id)
    run_dir.mkdir(parents=True)
    (run_dir / "train_result.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SchedulerError, match="holds output"):
        sched.resolve(other.run_id, "requeue")
    third = pilot_specs[0]
    with sched.db:
        sched.db.execute("UPDATE runs SET attempt=2 WHERE run_id=?", (third.run_id,))
    with pytest.raises(SchedulerError, match="never trained"):
        sched.resolve(third.run_id, "requeue")
    assert sched.spec(other.run_id) == other and sched.spec(third.run_id) == third


def test_an_operator_command_between_the_read_and_the_launch_is_never_overwritten(env) -> None:
    """``_start_training`` reads the pending row again under the write lock after checking its blockers (slow: git,
    launch gates), and leaves a changed row for the next pass, so an operator command that commits in between is
    never overwritten: a requeued run never trains its old spec, and a superseded continuation never becomes
    'blocked'."""
    from dataclasses import replace

    from pilot import manifest
    from pilot.launch import SPEC_FILE

    make, _, _, _ = env
    sched, operator = make(allow_pending=True), make(allow_pending=True)
    target = next(s for s in manifest.design("pilot") if s.plugin == "unconstrained_ppo")
    stale = replace(target, pending=(*target.pending, "Q-cost-critic"))  # queued by older code
    sched.add([stale])
    real = sched._launch_blocker

    def blocker(row):
        out = real(row)
        if out is None and sched.spec(row["run_id"]) == stale:  # the requeue lands during the check
            operator.resolve(row["run_id"], "requeue")
        return out

    sched._launch_blocker = blocker
    sched._fix_mode()
    sched._fix_policy()
    sched.step()
    assert sched.row(target.run_id)["status"] == "pending" and sched.spec(target.run_id) == target
    assert not (sched.cfg.run_dir(target.run_id) / SPEC_FILE).exists()
    assert any(e["event"] == "launch_skipped" for e in sched.events(target.run_id))
    sched.step()  # the next pass launches the stored (requeued) spec
    assert sched.row(target.run_id)["status"] == "training"
    assert RunSpec.from_json((sched.cfg.run_dir(target.run_id) / SPEC_FILE).read_text()) == target
    run_to_end(sched)
    assert sched.row(target.run_id)["status"] == "ledgered"

    # a pending continuation superseded by `resolve PARENT replace` while its dependencies were checked
    other = make(allow_pending=True, data_root=sched.cfg.data_root.parent / "data2",
                 ledger_paths=LedgerPaths(main=sched.cfg.data_root.parent / "l2.parquet",
                                          pilot=sched.cfg.data_root.parent / "p2.parquet",
                                          sidecar_dir=sched.cfg.data_root.parent / "side2"))
    op2 = Scheduler(other.cfg)
    parent = next(x for x in manifest.design("study_b") if x.seed == 0)
    cont = manifest.study_b_fewshot([parent])[0]
    other.add([parent])
    with other.db:  # excluded and left without a replacement: an operational pause (Q-exclusion-breaker)
        other.db.execute("UPDATE runs SET status='excluded', failure_cause='crash', ledger_written=1 WHERE run_id=?",
                         (parent.run_id,))
    other.add([cont])
    deps = other._dependencies_state

    def decided(spec_):
        out = deps(spec_)
        if spec_.run_id == cont.run_id:
            op2.resolve(parent.run_id, "replace")
        return out

    other._dependencies_state = decided
    other._fix_mode()
    other._refresh_git()
    assert other._start_training(other.row(cont.run_id)) is False
    assert other.row(cont.run_id)["status"] == "superseded"
    assert [e["event"] for e in other.events(cont.run_id)][0] == "launch_skipped"


def test_schedule_run_never_queues_a_replacement_from_uncommitted_code(env, monkeypatch) -> None:
    """Settling an excluded run queues its replacement (and a Study B run's few-shot continuations) computed by the
    code this scheduler loaded. In registered mode that code must be the commit's, as for `schedule add` and
    `resolve replace`: on a dirty tree, with uncommitted modules loaded, or after HEAD moved, the replacement is
    left to the operator (needs_decision); otherwise a `queued` event names the commit."""
    import pilot.scheduler as S
    from pilot import manifest
    from pilot.launch import CLAIM_FILE, SPEC_FILE, TRAIN_RESULT

    make, _, _, _ = env
    head = {"h": "a" * 40}
    state = {"dirty": ["pilot/manifest.py"], "unverified": []}
    monkeypatch.setattr(S.provenance, "commit_hash", lambda *a, **kw: head["h"])
    monkeypatch.setattr(S.provenance, "dirty_paths", lambda *a, **kw: list(state["dirty"]))
    monkeypatch.setattr(S.provenance, "unverified_imported_code", lambda *a, **kw: list(state["unverified"]))

    def crashed(sched, parent):
        rd = sched.cfg.run_dir(parent.run_id)
        rd.mkdir(parents=True)
        (rd / SPEC_FILE).write_text(parent.to_json())
        (rd / CLAIM_FILE).write_text("1\n")
        (rd / TRAIN_RESULT).write_text(json.dumps({
            "run_id": parent.run_id, "status": "failed", "failure_cause": "crash", "detail": "x",
            "commit_hash": "0" * 40,
            "omnisafe_dir": None, "started": "2026-01-01T00:00:00+00:00", "finished": "2026-01-01T01:00:00+00:00",
            "wall_clock_hours": 1.0}))
        with sched.db:
            sched.db.execute("UPDATE runs SET status='training', pid=999999, attempt=1, launch_commit=? WHERE run_id=?",
                             ("0" * 40, parent.run_id))

    parents = [x for x in manifest.design("study_b") if x.seed == 0][:2]
    sched = make(allow_dirty=False)
    sched.add([*parents, *manifest.study_b_fewshot(parents)])
    before = {r["run_id"] for r in sched.rows()}
    crashed(sched, parents[0])
    sched.run(once=True)  # dirty tree: the first pass also fixes the commit the code was loaded from ("a" * 40)
    assert {r["run_id"] for r in sched.rows()} == before  # nothing queued by uncommitted code
    row = sched.row(parents[0].run_id)
    assert row["status"] == "excluded" and row["replaced_by"] is None
    need = [e["detail"] for e in sched.events(parents[0].run_id) if e["event"] == "needs_decision"]
    assert need and "uncommitted changes" in need[0] and f"schedule resolve {parents[0].run_id} replace" in need[0]
    # no group decision is needed: `schedule status` shows what the operator does (commit, then replace)
    assert need[0] in [text for run_id, text in sched.status_report()["decisions"] if run_id == parents[0].run_id]
    state.update(dirty=[], unverified=["pilot/x.py (untracked)"])
    sched._refresh_git()
    assert "not committed: pilot/x.py" in sched._queueing_code_problem()
    state.update(unverified=[])
    head["h"] = "b" * 40  # HEAD moved under a scheduler that loaded its code at "a" * 40
    sched._refresh_git()
    assert "restart the scheduler" in sched._queueing_code_problem()  # git cannot compare the stand-in commits
    head["h"] = "a" * 40  # committed code, HEAD unchanged: the replacement is queued, with its commit logged
    sched._refresh_git()
    assert sched._queueing_code_problem() == ""
    crashed(sched, parents[1])
    sched.run(once=True)
    replacement = sched.row(parents[1].run_id)["replaced_by"]
    assert replacement is not None
    queued = [e["detail"] for e in sched.events(replacement) if e["event"] == "queued"]
    assert queued and "a" * 40 in queued[0]


def test_current_spec_rebuilds_surplus_replacement_and_fewshot_runs(env) -> None:
    """``current_spec`` rebuilds each kind of queued run as it was queued (unchanged code: the stored spec)."""
    from pilot import manifest

    make, _, _, _ = env
    sched = make(allow_dirty=False)
    study_b = [s for s in manifest.design("study_b") if s.seed == 0]
    main = [s for s in manifest.design("main") if s.seed == 0 and s.task == R.PRIMARY_TASK]
    sched.add([*main, *study_b, *manifest.study_b_fewshot(study_b)])
    sched.add_surplus(1)
    with sched.db:
        sched.db.execute("UPDATE runs SET status='excluded' WHERE run_id=?", (study_b[0].run_id,))
    sched.resolve(study_b[0].run_id, "replace")
    rows = sched.rows()
    assert any(r["surplus"] for r in rows) and any(r["replaces"] for r in rows)
    for row in rows:
        assert sched.current_spec(row["run_id"]) == RunSpec.from_json(row["spec"]), row["run_id"]


def test_a_registered_scheduler_writes_no_ledger_row_from_a_dirty_tree(env, monkeypatch) -> None:
    """Ledger rows and their supplement records are write-once and computed by the loaded code: in registered mode
    an exclusion and an evaluated run wait unwritten (ledger_waiting) while the tree is dirty."""
    make, set_scenarios, paths, tmp = env
    repo = tmp / "repo"
    head = clean_git_repo(repo)
    monkeypatch.setenv("FAKE_COMMIT", head)
    run = spec(0, name=PILOT_BASE, pilot=True, group="pilot")
    crash = spec(1, name=PILOT_BASE, pilot=True, group="pilot")
    set_scenarios(**{crash.run_id: "crash"})
    sched = make(allow_dirty=False, repo_root=repo, require_allocation_for_pilot=False)

    def until(run_id: str, status: str) -> None:  # a run settles only inside step(): no race with the edit below
        for _ in range(400):
            if sched.row(run_id)["status"] == status:
                return
            sched.step()
        raise AssertionError(f"{run_id} never reached {status}")

    def edit(text: str) -> None:
        (repo / "code.py").write_text(text, encoding="utf-8")

    sched.add([crash])
    until(crash.run_id, "training")
    edit("x = 2  # an uncommitted edit\n")
    run_to_end(sched, passes=50)
    assert sched.row(crash.run_id)["status"] == "excluded" and not sched.row(crash.run_id)["ledger_written"]
    assert not paths.pilot.exists() and not paths.main.exists()
    edit("x = 1\n")
    sched.add([run])
    until(run.run_id, "evaluating")
    edit("x = 2  # an uncommitted edit\n")
    run_to_end(sched, passes=50)
    assert sched.row(run.run_id)["status"] == "evaluated" and not sched.row(run.run_id)["ledger_written"]
    assert [r.run_id for r in ledger_rows(paths.pilot)] == [crash.run_id]  # written once the tree was clean
    waiting = [e for e in sched.events(run.run_id) if e["event"] == "ledger_waiting"]
    assert len(waiting) == 1 and "uncommitted" in waiting[0]["detail"]  # logged once, not on every pass
    assert any(e["event"] == "ledger_waiting" for e in sched.events(crash.run_id))
    edit("x = 1\n")  # the tree is clean again
    run_to_end(sched, passes=5)
    assert sched.row(run.run_id)["status"] == "ledgered" and sched.row(crash.run_id)["ledger_written"]
    assert sorted(r.run_id for r in ledger_rows(paths.pilot)) == sorted([run.run_id, crash.run_id])
    # HEAD moves on: a commit of no loaded code leaves the loaded code the commit's; a commit of loaded code does not
    import pilot.scheduler as S

    commit_files(repo, {"notes.json": "{}\n"})
    sched._refresh_git()
    assert sched._head != head and sched._code_head == head
    assert sched._ledger_code_problem(run.run_id) == ""
    assert sched._queueing_code_problem() == ""  # a Part 5.6 replacement is still queued
    monkeypatch.setattr(S.provenance, "imported_code_changed_between", lambda *a, **kw: ["pilot/ledger_writer.py"])
    assert "restart the scheduler" in sched._ledger_code_problem(run.run_id)
    assert "restart the scheduler" in sched._queueing_code_problem()


def test_recover_reads_the_tree_as_it_is_now_not_as_an_earlier_pass_cached_it(env, monkeypatch) -> None:
    """``recover()`` (run() calls it at every start, before the first pass) reads HEAD and the tree first, as step()
    does, so a run that ended while the tree became dirty waits (ledger_waiting) instead of having its exclusion row
    written from the cleanliness an earlier step() of the same object cached."""
    make, set_scenarios, paths, tmp = env
    repo = tmp / "repo"
    monkeypatch.setenv("FAKE_COMMIT", clean_git_repo(repo))
    crash = spec(1, name=PILOT_BASE, pilot=True, group="pilot")
    set_scenarios(**{crash.run_id: "crash"})
    sched = make(allow_dirty=False, repo_root=repo, require_allocation_for_pilot=False)
    sched.add([crash])
    for _ in range(400):
        if sched.row(crash.run_id)["status"] == "training":
            break
        sched.step()
    assert sched.row(crash.run_id)["status"] == "training" and sched._clean
    result = sched.cfg.run_dir(crash.run_id) / "train_result.json"
    for _ in range(400):  # the crash ends before the edit (as on a slow pass or a loaded machine)
        if result.exists():
            break
        time.sleep(0.05)
    assert result.exists()
    time.sleep(0.3)  # the fake process exits just after writing its result
    (repo / "code.py").write_text("x = 2  # an uncommitted edit\n", encoding="utf-8")
    sched.recover()
    assert sched.row(crash.run_id)["status"] == "excluded" and not sched.row(crash.run_id)["ledger_written"]
    assert not paths.pilot.exists() and not paths.main.exists()
    events = [e["event"] for e in sched.events(crash.run_id)]
    assert "ledger_waiting" in events and "ledger_exclusion_written" not in events and "replaced" not in events


def test_a_restart_runs_at_a_head_that_changed_only_result_files(env) -> None:
    """A restarted run waits only while code or configuration differ between its interrupted attempt's commit and
    HEAD, so a commit of a ledger or a determinism report (results only) never holds it for good."""
    make, _, _, tmp = env
    repo = tmp / "repo"
    first = clean_git_repo(repo)
    sched = make(allow_dirty=False, allow_pending=True, repo_root=repo, require_allocation_for_pilot=False)
    sched.add([spec(0)])
    with sched.db:  # the state restart() leaves: attempt 2 on the interrupted attempt's commit
        sched.db.execute("UPDATE runs SET attempt=2, launch_commit=? WHERE run_id=?", (first, S0))
    sched._refresh_git()
    assert sched._launch_blocker(sched.row(S0)) != "restart_commit_mismatch"
    commit_files(repo, {"results/ledger.parquet": "rows"})
    commit_files(repo, {"ledger/determinism/determinism-x.json": "{}"})
    sched._refresh_git()
    assert sched._head != first and sched._launch_blocker(sched.row(S0)) != "restart_commit_mismatch"
    commit_files(repo, {"code.py": "x = 2\n"})  # code changed since the interrupted attempt: the restart waits
    sched._refresh_git()
    assert sched._launch_blocker(sched.row(S0)) == "restart_commit_mismatch"
    assert sched._restart_code_changes(first) == ["code.py"]


def test_a_smoke_restart_records_the_commit_it_runs_on(env) -> None:
    """A smoke data root (allow_dirty) runs a restarted attempt at HEAD whatever commit restart() recorded, so the
    attempt records HEAD, as its train_starting event says."""
    make, _, _, _ = env
    sched = make()  # smoke: allow_dirty
    sched.add([spec(0)])
    with sched.db:  # the state restart() leaves: attempt 2 on an older commit
        sched.db.execute("UPDATE runs SET attempt=2, launch_commit=? WHERE run_id=?", ("f" * 40, S0))
    sched.run(once=True)
    started = [e["detail"] for e in sched.events(S0) if e["event"] == "train_starting"]
    assert started == [f"attempt=2 commit={sched._head}"]
    assert sched.row(S0)["launch_commit"] == sched._head


def test_a_result_file_that_holds_no_json_object_reads_as_missing(env) -> None:
    """``_read_json`` returns only a JSON object: a list or a number in train_result.json or evaluation.json reads
    as no result, never as a value whose ``.get`` would stop the scheduler."""
    make, _, _, tmp = env
    sched = make()
    for text in ("[1, 2]", "3", '"text"', "null", "{"):
        (tmp / "result.json").write_text(text, encoding="utf-8")
        assert sched._read_json(tmp / "result.json") is None
    (tmp / "result.json").write_text('{"status": "completed"}', encoding="utf-8")
    assert sched._read_json(tmp / "result.json") == {"status": "completed"}


def test_imported_code_changed_between_names_only_loaded_modules(tmp_path, monkeypatch) -> None:
    import importlib

    from pilot import provenance

    repo = tmp_path / "repo"
    first = clean_git_repo(repo)
    loaded_at = commit_files(repo, {"loaded_mod_for_test.py": "X = 1\n"})
    monkeypatch.syspath_prepend(str(repo))
    monkeypatch.delitem(sys.modules, "loaded_mod_for_test", raising=False)
    importlib.import_module("loaded_mod_for_test")
    try:
        other = commit_files(repo, {"code.py": "x = 3\n"})  # tracked, but never imported
        assert provenance.imported_code_changed_between(loaded_at, other, repo) == []
        changed = commit_files(repo, {"loaded_mod_for_test.py": "X = 2\n"})
        assert provenance.imported_code_changed_between(loaded_at, changed, repo) == ["loaded_mod_for_test.py"]
        assert provenance.imported_code_changed_between(first, loaded_at, repo) == ["loaded_mod_for_test.py"]
    finally:
        sys.modules.pop("loaded_mod_for_test", None)


def test_ledger_writing_schedule_commands_refuse_a_dirty_tree(tmp_path, monkeypatch) -> None:
    """`schedule resolve RUN retry-ledger` and `schedule set-policy exclude` write ledger rows: like the queueing
    decisions they need a clean tree and committed code."""
    import pilot.__main__ as cli
    from pilot import provenance

    monkeypatch.setattr(provenance, "dirty_paths", lambda *a, **kw: ["pilot/ledger_writer.py"])
    root = str(tmp_path / "root")
    assert cli.main(["schedule", "run", "--max-concurrent", "1", "--once", "--data-root", root]) == 0
    assert cli.main(["schedule", "set-policy", "exclude", "--data-root", root]) == 2
    assert cli.main(["schedule", "resolve", S0, "retry-ledger", "--data-root", root]) == 2
    # no run is held here, so it writes no ledger row; a held run past MAX_RESTARTS would be excluded by the rule
    # ('incomplete') and its row would wait for committed code (ledger_waiting, Scheduler._ledger_code_problem)
    assert cli.main(["schedule", "set-policy", "restart", "--data-root", root]) == 0
    monkeypatch.setattr(provenance, "dirty_paths", lambda *a, **kw: [])
    monkeypatch.setattr(provenance, "unverified_imported_code", lambda *a, **kw: ["pilot/ledger_writer.py (modified)"])
    assert cli.main(["schedule", "set-policy", "exclude", "--data-root", root]) == 2


@pytest.mark.parametrize("argv, message", [
    (["add-surplus", "--extra", "99"], "--extra must be from 0 to"),
    (["add-surplus", "--extra", "-1"], "--extra must be from 0 to"),
    (["add", "--design", "pilot", "--cuts", "1"], "cuts.json"),
    (["run", "--max-concurrent", "0"], "--max-concurrent must be at least 1"),
    (["run", "--max-concurrent", "1", "--poll", "-1"], "--poll must be"),
])
def test_bad_schedule_values_are_usage_errors_that_leave_no_state(tmp_path, capsys, monkeypatch, argv, message) -> None:
    """`add-surplus --extra 99`, `add --cuts K` without a committed pilot/cuts.json and `run --max-concurrent 0` are
    usage errors (exit 2), refused before the fresh data root's state is created and fixed as registered, so later
    smoke use of that root is never refused for 'registered runs' it never held."""
    from pilot import scheduler
    from pilot.__main__ import main

    clean_tree(monkeypatch)

    def no_cuts(count, repo_root=None):
        raise scheduler.SchedulerError("pilot/cuts.json is not committed at HEAD; commit the group's cut decision "
                                       "first")

    monkeypatch.setattr(scheduler, "committed_cuts", no_cuts)
    root = tmp_path / "root"
    assert main(["schedule", *argv, "--data-root", str(root)]) == 2
    assert message in capsys.readouterr().err
    assert not SchedulerConfig(data_root=root).state_path.exists()
    assert not root.exists() or not any(root.iterdir())


def test_the_ledger_code_check_sees_the_modules_the_first_write_imports(env, monkeypatch) -> None:
    """``write_run`` imports results.ledger_schema and results.supplement_schema lazily, so ``_ledger_code_problem``
    imports them before it checks the loaded code: in a fresh scheduler process an uncommitted
    results/supplement_schema.py waits from the first row on, and no row is written (and attributed to HEAD) by
    uncommitted code."""
    import pilot.scheduler as S

    make, _, paths, tmp = env
    repo = tmp / "repo"
    head = clean_git_repo(repo)
    monkeypatch.setenv("FAKE_COMMIT", head)
    for name in ("results.supplement_schema", "results.ledger_schema"):
        monkeypatch.delitem(sys.modules, name, raising=False)  # as in a fresh scheduler process

    def unverified(*a, **kw):  # provenance's rule, for a supplement_schema.py missing from the commit
        return ["results/supplement_schema.py (untracked)"] if "results.supplement_schema" in sys.modules else []

    monkeypatch.setattr(S.provenance, "unverified_imported_code", unverified)
    runs = [spec(seed, name=PILOT_BASE, pilot=True, group="pilot") for seed in (0, 1)]
    sched = make(allow_dirty=False, repo_root=repo, require_allocation_for_pilot=False)
    for name in ("results.supplement_schema", "results.ledger_schema"):
        monkeypatch.delitem(sys.modules, name, raising=False)  # not even Scheduler() loaded them
    sched.add(runs)
    run_to_end(sched, passes=200)
    assert {sched.row(r.run_id)["status"] for r in runs} == {"evaluated"}
    assert not paths.pilot.exists() or not ledger_rows(paths.pilot)
    for r in runs:
        waiting = [e["detail"] for e in sched.events(r.run_id) if e["event"] == "ledger_waiting"]
        assert waiting and "results/supplement_schema.py (untracked)" in waiting[0]


def test_a_refused_run_fixes_neither_the_mode_nor_the_ledgers(tmp_path, monkeypatch) -> None:
    """`schedule run` checks its interruption policy before it fixes the data root's mode and ledger location or
    writes the ledgers' mode markers, so a run refused for its policy (exit 3 while Q-interrupted-run is open)
    leaves none of them behind and a later run with another --ledger-dir is not refused."""
    from pilot.__main__ import main

    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", R.ANSWERED_QUESTIONS - {"Q-interrupted-run"})
    root, led_x, led_y = tmp_path / "root", tmp_path / "ledX", tmp_path / "ledY"
    base = ["schedule", "run", "--max-concurrent", "1", "--once", "--data-root", str(root)]
    assert main([*base, "--ledger-dir", str(led_x), "--on-interrupt", "restart"]) == 3
    assert not led_x.exists() or not any(led_x.iterdir())
    sched = Scheduler(SchedulerConfig(data_root=root))
    assert sched.setting("mode") is None and sched.setting("ledgers") is None
    assert main([*base, "--ledger-dir", str(led_y)]) == 0
    stored = Scheduler(SchedulerConfig(data_root=root))
    assert str(led_y) in stored.setting("ledgers") and stored.setting("on_interrupt") == "hold"
    # a policy differing from the stored one is refused before anything is fixed as well
    assert main([*base, "--ledger-dir", str(led_x), "--on-interrupt", "exclude"]) != 0
    assert not led_x.exists() or not any(led_x.iterdir())

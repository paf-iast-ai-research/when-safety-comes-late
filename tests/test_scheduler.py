"""The run scheduler: launch, resume, never duplicate, exclusion and replacement (Part 5.6)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from configs import registered as R
from pilot.ledger_writer import LedgerPaths
from pilot.manifest import RunSpec
from pilot.scheduler import Scheduler, SchedulerConfig, SchedulerError

FAKE = Path(__file__).with_name("fake_launcher.py")


def spec(seed: int = 0, name: str = "T-A-PointGoal1-N0.00", pilot: bool = False, depends=(), group: str = "main") -> RunSpec:
    return RunSpec(
        run_id=f"{name}-s{seed}", study="A", task="SafetyPointGoal1-v0", arm="N0.00", seed=seed,
        total_steps=2_000_000, base_algo="PPOLag", plugin="ppolag", group=group, pilot=pilot,
        N=0.0, onset_step=0, depends_on=tuple(depends),
    )


@pytest.fixture
def adopt_seed_rule(monkeypatch):
    """The group has recorded Q-seed-collision, so replacements may launch."""
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset({"Q-seed-collision"}))


@pytest.fixture
def env(tmp_path, monkeypatch):
    scenarios: dict[str, str] = {}
    scen_file = tmp_path / "scenarios.json"

    def set_scenarios(**kw):
        scenarios.update({k.replace("__", "-"): v for k, v in kw.items()})
        scen_file.write_text(json.dumps(scenarios), encoding="utf-8")

    set_scenarios()
    monkeypatch.setenv("FAKE_SCENARIOS", str(scen_file))
    monkeypatch.setenv("FAKE_RELEASE", str(tmp_path / "release"))
    paths = LedgerPaths(main=tmp_path / "ledger.parquet", pilot=tmp_path / "pilot.parquet", sidecar_dir=tmp_path / "sidecar")

    def make(**kw) -> Scheduler:
        cfg = SchedulerConfig(
            data_root=kw.pop("data_root", tmp_path / "data"), poll_seconds=0.05, ledger_paths=kw.pop("ledger_paths", paths),
            command_builder=lambda stage, spec_path, run_dir: [sys.executable, str(FAKE), stage, str(spec_path), str(run_dir)],
            **{"allow_dirty": True, **kw},  # smoke mode unless a test asks for the registered gates
        )
        return Scheduler(cfg)

    return make, set_scenarios, paths, tmp_path


def ledger_rows(path: Path) -> list:
    from results.ledger_schema import load_ledger_as_rows

    return load_ledger_as_rows(path)


def run_to_end(sched: Scheduler, passes: int = 400) -> None:
    sched.run(max_passes=passes)


def test_success_goes_to_ledger_once(env) -> None:
    make, _, paths, _ = env
    sched = make(max_concurrent=2)
    sched.add([spec(0), spec(1)])
    run_to_end(sched)
    assert sched.summary() == {"ledgered": 2}
    rows = ledger_rows(paths.main)
    assert sorted(r.run_id for r in rows) == ["T-A-PointGoal1-N0.00-s0", "T-A-PointGoal1-N0.00-s1"]
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
    sched.add([spec(0, name="P-A-PointGoal1-N0.00", pilot=True, group="pilot")])
    run_to_end(sched)
    assert not paths.main.exists()
    assert [r.run_id for r in ledger_rows(paths.pilot)] == ["P-A-PointGoal1-N0.00-s0"]


def test_crash_is_excluded_recorded_and_replaced_with_next_unused_seed(env, adopt_seed_rule) -> None:
    make, set_scenarios, paths, _ = env
    set_scenarios(**{"T-A-PointGoal1-N0.00-s1": "crash"})
    sched = make(max_concurrent=3)
    sched.add([spec(0), spec(1), spec(2)])
    run_to_end(sched)
    assert sched.row("T-A-PointGoal1-N0.00-s1")["status"] == "excluded"
    assert sched.row("T-A-PointGoal1-N0.00-s1")["replaced_by"] == "T-A-PointGoal1-N0.00-s5"
    assert sched.row("T-A-PointGoal1-N0.00-s5")["replaces"] == "T-A-PointGoal1-N0.00-s1"
    rows = {r.run_id: r for r in ledger_rows(paths.main)}
    assert rows["T-A-PointGoal1-N0.00-s1"].completed is False
    assert rows["T-A-PointGoal1-N0.00-s1"].failure_cause == "crash"
    assert rows["T-A-PointGoal1-N0.00-s5"].completed is True


def test_non_finite_multiplier_is_excluded(env) -> None:
    make, set_scenarios, paths, _ = env
    set_scenarios(**{"T-A-PointGoal1-N0.00-s0": "nonfinite"})
    sched = make()
    sched.add([spec(0)])
    run_to_end(sched)
    row = {r.run_id: r for r in ledger_rows(paths.main)}["T-A-PointGoal1-N0.00-s0"]
    assert row.failure_cause == "non_finite_multiplier"
    assert row.checkpoints[-1].multiplier is None  # NaN is not written as a multiplier value


def test_second_replacement_with_another_cause_takes_the_next_seed(env, adopt_seed_rule) -> None:
    make, set_scenarios, _, _ = env
    set_scenarios(**{"T-A-PointGoal1-N0.00-s0": "crash", "T-A-PointGoal1-N0.00-s5": "nonfinite"})
    sched = make()
    sched.add([spec(0)])
    run_to_end(sched)
    assert sched.row("T-A-PointGoal1-N0.00-s5")["replaced_by"] == "T-A-PointGoal1-N0.00-s6"
    assert sched.row("T-A-PointGoal1-N0.00-s6")["status"] == "ledgered"


def test_repeated_same_cause_stops_replacing(env, adopt_seed_rule) -> None:
    make, set_scenarios, paths, _ = env
    set_scenarios(**{"T-A-PointGoal1-N0.00-s0": "crash", "T-A-PointGoal1-N0.00-s5": "crash"})
    sched = make()
    sched.add([spec(0)])
    run_to_end(sched)
    row = sched.row("T-A-PointGoal1-N0.00-s5")
    assert row["status"] == "excluded" and row["replaced_by"] is None
    assert any(e["event"] == "needs_decision" for e in sched.events("T-A-PointGoal1-N0.00-s5"))
    assert sorted(r.run_id for r in ledger_rows(paths.main)) == ["T-A-PointGoal1-N0.00-s0", "T-A-PointGoal1-N0.00-s5"]


def test_segfault_counts_as_crash_not_interruption(env, adopt_seed_rule) -> None:
    make, set_scenarios, paths, _ = env
    set_scenarios(**{"T-A-PointGoal1-N0.00-s0": "segv"})
    sched = make(on_interrupt="hold")
    sched.add([spec(0)])
    run_to_end(sched)
    assert sched.row("T-A-PointGoal1-N0.00-s0")["failure_cause"] == "crash"
    assert sched.row("T-A-PointGoal1-N0.00-s5")["status"] == "ledgered"


def test_interruption_is_held_by_default_then_restarted_by_the_operator(env) -> None:
    make, set_scenarios, paths, tmp = env
    set_scenarios(**{"T-A-PointGoal1-N0.00-s0": "interrupt"})
    sched = make()
    sched.add([spec(0)])
    run_to_end(sched)
    assert sched.row("T-A-PointGoal1-N0.00-s0")["status"] == "interrupted"
    assert not paths.main.exists()  # nothing recorded, nothing replaced
    set_scenarios(**{"T-A-PointGoal1-N0.00-s0": "success"})
    sched.resolve("T-A-PointGoal1-N0.00-s0", "restart")
    run_to_end(sched)
    row = sched.row("T-A-PointGoal1-N0.00-s0")
    assert row["status"] == "ledgered" and row["attempt"] == 2
    assert (tmp / "data" / "scheduler" / "interrupted" / "T-A-PointGoal1-N0.00-s0-attempt1").is_dir()


def test_interruption_policy_is_fixed_per_data_root(env) -> None:
    make, _, _, _ = env
    make().run(once=True)
    with pytest.raises(SchedulerError, match="set-policy"):
        make(on_interrupt="restart").run(once=True)


def test_set_policy_applies_to_every_held_run(env) -> None:
    make, set_scenarios, _, _ = env
    set_scenarios(**{"T-A-PointGoal1-N0.00-s0": "interrupt", "T-A-PointGoal1-N0.00-s1": "interrupt"})
    sched = make(max_concurrent=2)
    sched.add([spec(0), spec(1)])
    run_to_end(sched)
    assert {sched.row(f"T-A-PointGoal1-N0.00-s{s}")["status"] for s in (0, 1)} == {"interrupted"}
    with pytest.raises(ValueError):
        sched.resolve("T-A-PointGoal1-N0.00-s0", "exclude")  # one-by-one exclusion is not offered
    set_scenarios(**{"T-A-PointGoal1-N0.00-s0": "success", "T-A-PointGoal1-N0.00-s1": "success"})
    sched.set_policy("restart")
    assert {sched.row(f"T-A-PointGoal1-N0.00-s{s}")["status"] for s in (0, 1)} == {"pending"}


def test_interruption_policy_exclude(env, adopt_seed_rule) -> None:
    make, set_scenarios, paths, _ = env
    set_scenarios(**{"T-A-PointGoal1-N0.00-s0": "interrupt"})
    sched = make(on_interrupt="exclude")
    sched.add([spec(0)])
    run_to_end(sched)
    rows = {r.run_id: r for r in ledger_rows(paths.main)}
    assert rows["T-A-PointGoal1-N0.00-s0"].failure_cause == "incomplete"
    assert rows["T-A-PointGoal1-N0.00-s5"].completed


def test_unavailable_plugin_leaves_run_pending_without_exclusion(env) -> None:
    make, set_scenarios, paths, _ = env
    set_scenarios(**{"T-A-PointGoal1-N0.00-s0": "unavailable"})
    sched = make()
    sched.add([spec(0)])
    run_to_end(sched)
    assert sched.row("T-A-PointGoal1-N0.00-s0")["status"] == "pending"
    assert not paths.main.exists()


def test_dependent_is_blocked_when_the_dependency_is_excluded(env) -> None:
    make, set_scenarios, _, _ = env
    base = spec(0)
    dependent = spec(0, name="T-A-PointGoal1-N0.50-abrupt-total-warm_started", depends=[base.run_id], group="controller")
    set_scenarios(**{base.run_id: "crash"})
    sched = make(max_concurrent=2)
    sched.add([dependent, base])
    run_to_end(sched)
    assert sched.row(dependent.run_id)["status"] == "blocked"


def test_dependency_order_is_respected(env) -> None:
    make, _, _, _ = env
    base = spec(0)
    dependent = spec(0, name="T-A-PointGoal1-N0.50-abrupt-total-warm_started", depends=[base.run_id], group="controller")
    sched = make(max_concurrent=2)
    sched.add([dependent, base])
    sched.run(once=True)
    assert sched.row(dependent.run_id)["status"] == "pending"
    assert sched.row(base.run_id)["status"] == "training"
    run_to_end(sched)
    assert sched.summary() == {"ledgered": 2}


def test_concurrency_limit_and_adoption_after_scheduler_restart(env) -> None:
    make, set_scenarios, _, tmp = env
    set_scenarios(**{f"T-A-PointGoal1-N0.00-s{s}": "hang" for s in range(3)})
    sched = make(max_concurrent=2)
    sched.add([spec(s) for s in range(3)])
    sched.run(once=True)
    assert len(sched.rows("training")) == 2
    # a new scheduler instance (e.g. after the scheduler process restarted) adopts the live children
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
    run_dir = tmp / "data" / "checkpoints" / "T-A-PointGoal1-N0.00-s0"
    (run_dir / "omnisafe").mkdir(parents=True)
    with pytest.raises(SchedulerError, match="already holds output"):
        sched.run(once=True)


def test_eval_failed_can_be_reevaluated_and_does_not_block_dependents(env) -> None:
    make, set_scenarios, _, _ = env
    base = spec(0)
    dependent = spec(0, name="T-A-PointGoal1-N0.50-abrupt-total-warm_started", depends=[base.run_id], group="controller")
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


def test_run_directory_with_a_claim_is_not_relaunched(env) -> None:
    make, _, _, tmp = env
    sched = make()
    sched.add([spec(0)])
    run_dir = tmp / "data" / "checkpoints" / "T-A-PointGoal1-N0.00-s0"
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
    make, _, paths, tmp = env
    sched = make()
    sched.add([spec(0)])
    run_to_end(sched)
    other = make()  # same data root with its state emptied, same ledger
    other.db.execute("DELETE FROM runs")
    other.db.commit()
    other.add([spec(0)])
    other.run(once=True)
    assert other.row("T-A-PointGoal1-N0.00-s0")["status"] == "ledgered"
    assert len(ledger_rows(paths.main)) == 1


def test_changed_spec_for_an_existing_run_is_refused(env) -> None:
    make, _, _, _ = env
    sched = make()
    sched.add([spec(0)])
    changed = RunSpec.from_dict({**spec(0).to_dict(), "total_steps": 4_000_000})
    with pytest.raises(SchedulerError, match="different spec"):
        sched.add([changed])


def test_pending_questions_block_launch_unless_allowed(env) -> None:
    make, _, _, _ = env
    held = RunSpec.from_dict({**spec(0).to_dict(), "pending": ["Q-rounding"]})
    sched = make()
    sched.add([held])
    run_to_end(sched)
    assert sched.row(held.run_id)["status"] == "pending"
    waiting = [e for e in sched.events(held.run_id) if e["event"] == "open_questions"]
    assert len(waiting) == 1 and waiting[0]["detail"] == "Q-rounding"  # `schedule status` shows why it waits


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


def test_replacement_waits_for_the_seed_rule_to_be_recorded(env) -> None:
    make, set_scenarios, paths, _ = env
    set_scenarios(**{"T-A-PointGoal1-N0.00-s0": "crash"})
    sched = make()
    sched.add([spec(0)])
    run_to_end(sched)
    replacement = sched.spec("T-A-PointGoal1-N0.00-s5")
    assert "Q-seed-collision" in replacement.pending
    assert sched.row(replacement.run_id)["status"] == "pending"


def test_fewshot_continuation_ends_as_continued_without_a_row(env, adopt_seed_rule, monkeypatch) -> None:
    from pilot import manifest

    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset({"Q-seed-collision", "Q-continuations"}))
    make, _, paths, _ = env
    parent = RunSpec.from_dict({**manifest.study_b((0,))[4].to_dict(), "total_steps": 2_000_000})
    cont = RunSpec.from_dict({**manifest.study_b_fewshot([parent])[0].to_dict(), "total_steps": 200_000})
    sched = make(max_concurrent=2)
    sched.add([parent, cont])
    run_to_end(sched)
    assert sched.row(parent.run_id)["status"] == "ledgered" and sched.row(cont.run_id)["status"] == "continued"
    assert [r.run_id for r in ledger_rows(paths.main)] == [parent.run_id]


def test_excluded_study_b_run_gets_its_replacement_continued(env, adopt_seed_rule) -> None:
    from pilot import manifest

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


def test_failed_exclusion_record_is_retried_within_the_same_run(env, adopt_seed_rule, monkeypatch) -> None:
    import pilot.scheduler as S

    make, set_scenarios, paths, _ = env
    set_scenarios(**{"T-A-PointGoal1-N0.00-s1": "crash"})
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
    assert "T-A-PointGoal1-N0.00-s1" in ids and sched.row("T-A-PointGoal1-N0.00-s1")["ledger_written"] == 1


def test_orphaned_child_is_adopted_not_relaunched(env) -> None:
    import subprocess

    make, set_scenarios, _, tmp = env
    set_scenarios(**{"T-A-PointGoal1-N0.00-s0": "hang"})
    sched = make()
    sched.add([spec(0)])
    run_dir = tmp / "data" / "checkpoints" / "T-A-PointGoal1-N0.00-s0"
    run_dir.mkdir(parents=True)
    (run_dir / "spec.json").write_text(spec(0).to_json())
    # the scheduler died after committing 'training' and starting the child, before recording its PID
    child = subprocess.Popen([sys.executable, str(FAKE), "train", str(run_dir / "spec.json"), str(run_dir)])
    with sched.db:
        sched.db.execute("UPDATE runs SET status='training', pid=NULL, host=? WHERE run_id=?", (sched._host, "T-A-PointGoal1-N0.00-s0"))
    fresh = make()
    fresh.run(once=True)
    assert fresh.row("T-A-PointGoal1-N0.00-s0")["pid"] == child.pid
    assert not any(e["event"] == "train_started" for e in fresh.events("T-A-PointGoal1-N0.00-s0"))
    (tmp / "release").write_text("go")
    child.wait(timeout=30)
    run_to_end(fresh)
    assert fresh.row("T-A-PointGoal1-N0.00-s0")["status"] == "ledgered"


# -- process identification and restarts ------------------------------------------------------------

S0, S1, S5 = "T-A-PointGoal1-N0.00-s0", "T-A-PointGoal1-N0.00-s1", "T-A-PointGoal1-N0.00-s5"
WARM = "T-A-PointGoal1-N0.50-abrupt-total-warm_started"


def test_a_run_is_not_taken_for_a_run_whose_id_extends_it(env) -> None:
    make, set_scenarios, _, tmp = env
    set_scenarios(**{"T-A-PointGoal1-N0.00-s10": "hang"})
    sched = make()
    sched.add([spec(10)])
    sched.run(once=True)
    s10_pid = sched.row("T-A-PointGoal1-N0.00-s10")["pid"]
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
def test_a_run_that_never_started_goes_back_to_pending(env, adopt_seed_rule, monkeypatch, policy) -> None:
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


def test_restarts_under_the_restart_policy_are_limited(env) -> None:
    from pilot.scheduler import MAX_RESTARTS

    make, set_scenarios, _, _ = env
    set_scenarios(**{S0: "interrupt"})  # e.g. killed by the OOM killer every time
    sched = make(on_interrupt="restart")
    sched.add([spec(0)])
    run_to_end(sched, passes=100)
    row = sched.row(S0)
    assert row["status"] == "interrupted" and row["attempt"] == MAX_RESTARTS + 1
    assert any(e["event"] == "restart_limit" for e in sched.events(S0))


def test_the_launchers_own_interruption_record_is_kept(env, adopt_seed_rule) -> None:
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


# -- exclusion, replacement and the group's decisions ------------------------------------------------


def test_exclusions_applied_by_the_interruption_policy_do_not_trip_the_breaker(env, adopt_seed_rule) -> None:
    make, set_scenarios, _, _ = env
    set_scenarios(**{f"T-A-PointGoal1-N0.00-s{s}": "interrupt" for s in range(3)})  # one outage stops three runs
    sched = make(max_concurrent=3)
    sched.add([spec(s) for s in range(3)])
    run_to_end(sched)
    sched.set_policy("exclude")
    assert [sched.row(f"T-A-PointGoal1-N0.00-s{s}")["replaced_by"] for s in range(3)] == [
        "T-A-PointGoal1-N0.00-s5", "T-A-PointGoal1-N0.00-s6", "T-A-PointGoal1-N0.00-s7"]


def test_replacement_seeds_are_chosen_inside_the_write_transaction(env, adopt_seed_rule, monkeypatch) -> None:
    import sqlite3

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
    dependent = spec(0, name=WARM, depends=[base.run_id], group="controller")
    sched = make(max_concurrent=2)
    sched.add([dependent])  # `controller` queued before `main`
    sched.run(once=True)
    assert sched.row(dependent.run_id)["status"] == "pending"
    assert any(e["event"] == "dependency_missing" for e in sched.events(dependent.run_id))
    sched.add([base])
    run_to_end(sched)
    assert sched.summary() == {"ledgered": 2}


def test_the_group_can_replace_a_run_the_breaker_left_without_replacement(env, adopt_seed_rule) -> None:
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
    assert sched.row(S5)["replaced_by"] == "T-A-PointGoal1-N0.00-s6"
    assert sched.row("T-A-PointGoal1-N0.00-s6")["status"] == "ledgered"


def test_a_blocked_dependent_is_replaced_by_the_group_decision(env, adopt_seed_rule) -> None:
    make, set_scenarios, _, _ = env
    base = spec(0)
    dependent = spec(0, name=WARM, depends=[base.run_id], group="controller")
    set_scenarios(**{base.run_id: "crash"})
    sched = make(max_concurrent=2)
    sched.add([base, dependent])
    run_to_end(sched)
    assert sched.row(dependent.run_id)["status"] == "blocked"
    with pytest.raises(SchedulerError, match="still excluded"):
        sched.resolve(dependent.run_id, "unblock")
    sched.resolve(dependent.run_id, "replace")
    new = f"{WARM}-s5"
    assert sched.spec(new).depends_on == (S5,)
    run_to_end(sched)
    assert sched.row(new)["status"] == "ledgered" and sched.row(S5)["status"] == "ledgered"


def test_unblock_returns_a_blocked_run_whose_dependencies_are_fine(env) -> None:
    make, _, _, _ = env
    base = spec(0)
    dependent = spec(0, name=WARM, depends=[base.run_id], group="controller")
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
    assert any(e["event"] == "needs_decision" for e in other.events(S0))
    other.resolve(S0, "replace")
    assert other.row(S5)["replaces"] == S0


# -- surplus seeds (Part 5.5) -----------------------------------------------------------------------


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


def test_the_replacement_of_a_surplus_run_is_surplus(env, adopt_seed_rule) -> None:
    make, set_scenarios, _, _ = env
    set_scenarios(**{S5: "crash"})
    sched = make()
    sched.add([spec(0)])
    assert sched.add_surplus(1) == [S5]
    run_to_end(sched)
    replacement = sched.row("T-A-PointGoal1-N0.00-s6")
    assert replacement["surplus"] == 1 and replacement["replaces"] == S5
    assert sched.add_surplus(1) == []  # the replacement does not count as a second surplus seed


# -- modes and the command line --------------------------------------------------------------------


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

    inside = LedgerPaths(main=REPO_ROOT / "results" / "ledger.parquet", pilot=tmp_path / "p.parquet", sidecar_dir=tmp_path / "s")
    with pytest.raises(ValueError, match="inside the repository"):
        SchedulerConfig(data_root=tmp_path, allow_dirty=True, ledger_paths=inside)


def test_registered_mode_gates_hold_pilot_runs_and_restarts(env, monkeypatch) -> None:
    import pilot.scheduler as S

    make, _, _, _ = env
    monkeypatch.setattr(S.provenance, "dirty_paths", lambda *a, **kw: [])
    monkeypatch.setattr(S.provenance, "file_committed_at", lambda *a, **kw: None)
    sched = make(allow_dirty=False)
    pilot_run = spec(0, name="P-A-PointGoal1-N0.00", pilot=True, group="pilot")
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
    from pilot.__main__ import main

    root = str(tmp_path / "root")
    assert main(["schedule", "run", "--max-concurrent", "1", "--once", "--data-root", root]) == 0
    assert main(["schedule", "set-policy", "restart", "--data-root", root]) == 0
    assert main(["schedule", "run", "--max-concurrent", "1", "--once", "--data-root", root]) == 0  # HANDOVER's command
    stored = Scheduler(SchedulerConfig(data_root=Path(root))).setting("on_interrupt")
    assert stored == "restart"


def test_schedule_commands_on_a_missing_data_root_create_nothing(tmp_path) -> None:
    from pilot.__main__ import main

    root = tmp_path / "typo"
    for argv in (["status"], ["resolve", S0, "restart"], ["set-policy", "restart"]):
        assert main(["schedule", *argv, "--data-root", str(root)]) == 2
    assert not root.exists()


def test_the_ledger_location_is_fixed_per_data_root(env, tmp_path) -> None:
    make, set_scenarios, paths, _ = env
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


# -- review round 3: Study B replacements, restart commits, data-root spellings ---------------------


def test_the_group_replaces_a_study_b_parent_and_its_continuations_follow(env, adopt_seed_rule) -> None:
    from pilot import manifest

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
    with pytest.raises(SchedulerError, match="resolve B-Moderate-s5 replace"):
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

    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset({"Q-seed-collision", "Q-continuations"}))
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


def test_a_restart_uses_the_commit_the_interrupted_attempt_ran_on(env, monkeypatch) -> None:
    import pilot.scheduler as S

    make, set_scenarios, _, _ = env
    head = {"h": "a" * 40}
    monkeypatch.setattr(S.provenance, "commit_hash", lambda *a, **kw: head["h"])
    monkeypatch.setattr(S.provenance, "dirty_paths", lambda *a, **kw: [])
    set_scenarios(**{S0: "unavailable"})  # queued before its plug-in was committed
    sched = make(allow_dirty=False)
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


# -- review round 6 -----------------------------------------------------------------------------------


def test_replacements_stopped_by_the_exclude_policy_again_and_again_wait_for_the_group(env, adopt_seed_rule) -> None:
    make, set_scenarios, _, _ = env
    set_scenarios(**{f"T-A-PointGoal1-N0.00-s{s}": "interrupt" for s in [0, *range(5, 20)]})  # e.g. OOM every time
    sched = make(on_interrupt="exclude")
    sched.add([spec(0)])
    run_to_end(sched)
    assert sorted(r["seed"] for r in sched.rows()) == [0, 5, 6]
    last = sched.row("T-A-PointGoal1-N0.00-s6")
    assert last["status"] == "excluded" and last["replaced_by"] is None
    assert any("interruption policy" in (e["detail"] or "") for e in sched.events(last["run_id"]) if e["event"] == "needs_decision")


def test_two_replace_decisions_never_queue_two_seeds(env, adopt_seed_rule) -> None:
    import contextlib

    make, set_scenarios, _, _ = env
    set_scenarios(**{S0: "crash", S5: "crash"})
    sched = make()
    sched.add([spec(0)])
    run_to_end(sched)
    first, second = make(), make()
    real = first._write_txn

    @contextlib.contextmanager
    def after_the_other_command():
        second.resolve(S5, "replace")  # the other operator command commits first
        with real():
            yield

    first._write_txn = after_the_other_command
    with pytest.raises(SchedulerError, match="without a replacement"):
        first.resolve(S5, "replace")
    assert [r["run_id"] for r in first.rows() if r["replaces"] == S5] == ["T-A-PointGoal1-N0.00-s6"]


def test_a_surplus_run_keeps_its_arms_open_question_and_waits(env) -> None:
    make, _, _, _ = env
    held = RunSpec.from_dict({**spec(0).to_dict(), "pending": ["Q-surplus-arm-set"]})
    sched = make()
    sched.add([held])
    assert sched.add_surplus(1) == [S5]
    assert "Q-surplus-arm-set" in sched.spec(S5).pending
    run_to_end(sched)
    assert sched.row(S5)["status"] == "pending"
    assert [e["detail"] for e in sched.events(S5) if e["event"] == "open_questions"] == ["Q-surplus-arm-set"]


def test_a_blocked_run_replaced_by_the_group_is_superseded(env, adopt_seed_rule) -> None:
    make, set_scenarios, _, _ = env
    base = spec(0)
    dependent = spec(0, name=WARM, depends=[base.run_id], group="controller")
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


def test_a_policy_exclusion_interrupted_by_a_scheduler_crash_stays_a_policy_exclusion(env, adopt_seed_rule, monkeypatch) -> None:
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


def test_evaluation_waits_for_a_clean_worktree(env, monkeypatch) -> None:
    import pilot.scheduler as S

    make, set_scenarios, _, tmp = env
    dirty: list[str] = []
    monkeypatch.setattr(S.provenance, "dirty_paths", lambda *a, **kw: list(dirty))
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


@pytest.mark.parametrize(("scenario", "code"), [("interrupt", -9), ("exit1", 1)])
def test_an_end_without_a_final_result_is_held(env, scenario, code) -> None:
    make, set_scenarios, paths, _ = env
    set_scenarios(**{S0: scenario})
    sched = make()
    sched.add([spec(0)])
    run_to_end(sched)
    assert sched.row(S0)["status"] == "interrupted" and not paths.main.exists()
    assert any(f"exit code {code})" in (e["detail"] or "") for e in sched.events(S0) if e["event"] == "interrupted")


def test_retrying_a_recorded_exclusion_says_it_was_already_present(env, adopt_seed_rule) -> None:
    make, set_scenarios, _, _ = env
    set_scenarios(**{S0: "crash"})
    sched = make()
    sched.add([spec(0)])
    run_to_end(sched)
    sched.resolve(S0, "retry-ledger")
    events = [e["event"] for e in sched.events(S0)]
    assert events.count("ledger_exclusion_written") == 1 and events[0] == "ledger_exclusion_present"


@pytest.mark.parametrize("action", [["add", "--design", "pilot"], ["add-surplus", "--extra", "1"], ["status"]])
def test_commands_that_write_no_ledger_take_no_ledger_dir(tmp_path, action) -> None:
    from pilot.__main__ import main

    with pytest.raises(SystemExit) as exc:
        main(["schedule", *action, "--data-root", str(tmp_path / "root"), "--ledger-dir", str(tmp_path / "L")])
    assert exc.value.code == 2
    assert not (tmp_path / "root").exists()


def test_surplus_seeds_of_the_ramp_arms_wait_on_the_surplus_arm_set_question(env) -> None:
    from pilot import manifest

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


# -- review round 7 -----------------------------------------------------------------------------------


def test_a_restart_cut_short_after_its_archive_move_is_completed_at_the_next_start(env, monkeypatch) -> None:
    import shutil

    import pilot.scheduler as S

    make, set_scenarios, _, tmp = env
    head = {"h": "a" * 40}
    monkeypatch.setattr(S.provenance, "commit_hash", lambda *a, **kw: head["h"])
    monkeypatch.setattr(S.provenance, "dirty_paths", lambda *a, **kw: [])
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

    def meanwhile():
        real_fix()
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


def test_one_outage_stopping_unrelated_replacements_does_not_trip_the_policy_breaker(env, adopt_seed_rule) -> None:
    import os
    import signal
    import time

    make, set_scenarios, _, tmp = env
    s2, s6 = "T-A-PointGoal1-N0.00-s2", "T-A-PointGoal1-N0.00-s6"
    set_scenarios(**{S1: "crash", s2: "nonfinite", S5: "hang", s6: "hang"})
    sched = make(max_concurrent=3)
    sched.add([spec(1), spec(2)])
    for _ in range(200):  # s5 replaces a crash of s1, s6 a non-finite multiplier of s2; both claim and train
        sched.run(once=True)
        if all(sched.exists(r) and (tmp / "data" / "checkpoints" / r / ".claim").exists() for r in (S5, s6)):
            break
        time.sleep(0.05)
    for rid in (S5, s6):  # one outage stops both
        os.kill(sched.row(rid)["pid"], signal.SIGKILL)
    for _ in range(200):
        sched.run(once=True)
        if {sched.row(r)["status"] for r in (S5, s6)} == {"interrupted"}:
            break
        time.sleep(0.05)
    sched.set_policy("exclude")
    assert sched.row(S5)["replaced_by"] == "T-A-PointGoal1-N0.00-s7"
    assert sched.row(s6)["replaced_by"] == "T-A-PointGoal1-N0.00-s8"
    assert not any(e["event"] == "needs_decision" for e in sched.events(s6))


def test_a_set_policy_exclude_cut_short_is_completed_at_the_next_start(env, adopt_seed_rule, monkeypatch) -> None:
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
    assert record["written_by"] == "scheduler (the process left no final result)" and record["excluded_by_policy"] is True
    assert {r.run_id: r.failure_cause for r in ledger_rows(paths.main)}[S0] == "incomplete"


def test_unblock_and_replace_decisions_never_both_apply(env, adopt_seed_rule) -> None:
    make, set_scenarios, _, _ = env
    base = spec(0)
    dependent = spec(0, name=WARM, depends=[base.run_id], group="controller")
    sched = make()
    sched.add([base, dependent])
    with sched.db:  # blocked, with its dependency fine again (e.g. blocked by an earlier version)
        sched.db.execute("UPDATE runs SET status='blocked' WHERE run_id=?", (dependent.run_id,))
    first, second = make(), make()
    real = first._fix_mode

    def meanwhile():
        real()
        second.resolve(dependent.run_id, "replace")  # another operator's decision commits first

    first._fix_mode = meanwhile
    with pytest.raises(SchedulerError, match="without a replacement can be unblocked"):
        first.resolve(dependent.run_id, "unblock")
    assert first.row(dependent.run_id)["status"] == "superseded"


# -- review round 8 -----------------------------------------------------------------------------------


def _held_then_restarted_and_launched_meanwhile(env, hook_owner: Scheduler, name: str):
    """A held S0; while ``hook_owner`` is about to act on it, another operator restarts it and the
    scheduler launches attempt 2, which hangs (alive) until released."""
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
    sched = _held_then_restarted_and_launched_meanwhile(env, op, "restart")
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


def test_set_policy_exclude_never_excludes_a_run_another_operator_restarted(env, adopt_seed_rule) -> None:
    make, _, paths, tmp = env
    op = make()
    sched = _held_then_restarted_and_launched_meanwhile(env, op, "_exclude_without_result")
    try:
        op.set_policy("exclude")
        row = op.row(S0)
        assert (row["status"], row["attempt"], row["replaced_by"]) == ("training", 2, None)
        assert not (tmp / "data" / "checkpoints" / S0 / "train_result.json").exists()  # the live run's directory is untouched
        assert not op.exists(S5)
        assert [e["event"] for e in op.events(S0)][0] == "exclusion_skipped"
    finally:
        (tmp / "release").write_text("go")
    run_to_end(sched)
    assert (sched.row(S0)["status"], sched.row(S0)["attempt"]) == ("ledgered", 2)
    assert [(r.run_id, r.completed) for r in ledger_rows(paths.main)] == [(S0, True)]


# -- review round 9: operator decisions re-checked inside the write transaction ---------------------


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

    def meanwhile() -> None:
        real()
        op1.resolve(S0, "reevaluate")  # another operator carries out the same decision first
        sched.run(once=True)  # and the scheduler launches the evaluation

    op2._fix_mode = meanwhile
    with pytest.raises(SchedulerError, match="only an eval_failed run can be re-evaluated"):
        op2.resolve(S0, "reevaluate")
    run_to_end(sched)
    assert sched.row(S0)["status"] == "ledgered"
    assert [e["event"] for e in sched.events(S0)].count("eval_started") == 2  # the failed one and one retry


# -- review round 11: the smoke script never queues runs in a registered data root ------------------


def test_smoke_script_refuses_a_registered_data_root_before_queuing(tmp_path, capsys) -> None:
    import importlib.util

    root = tmp_path / "registered"
    Scheduler(SchedulerConfig(data_root=root)).fix_mode()  # a registered data root
    path = Path(__file__).resolve().parents[1] / "scripts" / "smoke_run.py"
    module_spec = importlib.util.spec_from_file_location("smoke_run", path)
    smoke_run = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(smoke_run)
    assert smoke_run.main(["--data-root", str(root)]) == 1
    assert "registered runs" in capsys.readouterr().err
    assert Scheduler(SchedulerConfig(data_root=root)).rows() == []  # nothing was queued


def test_smoke_script_refuses_a_data_root_with_queued_registered_runs(tmp_path, capsys, monkeypatch) -> None:
    import importlib.util

    from pilot import manifest

    # were the refusal to regress, fail at once instead of launching the queued 10M-step runs
    monkeypatch.setattr(Scheduler, "run", lambda self, **kw: pytest.fail("smoke_run.py launched runs"))

    root = tmp_path / "queued"
    Scheduler(SchedulerConfig(data_root=root)).add(manifest.design("pilot"))  # Scheduler.add fixes no mode
    path = Path(__file__).resolve().parents[1] / "scripts" / "smoke_run.py"
    module_spec = importlib.util.spec_from_file_location("smoke_run", path)
    smoke_run = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(smoke_run)
    assert smoke_run.main(["--data-root", str(root)]) == 1
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
            with pytest.raises(Exception):  # blocked by smoke's write lock (sqlite3 "database is locked")
                registered.db.execute("BEGIN IMMEDIATE")
        return value

    registered.db.execute("PRAGMA busy_timeout = 0")
    smoke.setting = setting_then_registered
    smoke.fix_mode()
    with pytest.raises(SchedulerError, match="holds smoke runs"):
        registered.fix_mode()


def test_schedule_add_refuses_a_smoke_data_root(tmp_path, capsys) -> None:
    from pilot.__main__ import main

    root = tmp_path / "smoke"
    Scheduler(SchedulerConfig(data_root=root, allow_dirty=True, allow_pending=True)).fix_mode()
    assert main(["schedule", "add", "--design", "pilot", "--data-root", str(root)]) == 2
    assert "smoke data root" in capsys.readouterr().err
    assert Scheduler(SchedulerConfig(data_root=root, allow_dirty=True)).rows() == []


def test_schedule_add_fixes_the_root_as_registered_so_smoke_flags_are_refused(tmp_path, capsys, monkeypatch) -> None:
    from pilot.__main__ import main

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


def test_schedule_add_leaves_the_ledger_location_to_the_first_run(tmp_path) -> None:
    from pilot.__main__ import main

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

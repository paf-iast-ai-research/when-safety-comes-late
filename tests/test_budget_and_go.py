"""Compute budget, surplus seeds (Part 5.5) and the go or no-go rule (Part 6; equation 12; pilot/go_decision.py)."""

from __future__ import annotations

import inspect
import json
import math
import os
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from configs import registered as R
from pilot import budget, go_decision


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


def test_registered_requirement_is_the_part6_formula() -> None:
    assert budget.registered_run_equivalents() == 484
    assert budget.registered_requirement(20.0, 10) == pytest.approx(20.0 / 10 * 484)
    with pytest.raises(ValueError):
        budget.registered_requirement(20.0, 0)


def test_corrected_run_equivalents_count_what_484_leaves_out() -> None:
    c = budget.corrected_run_equivalents()
    assert c["pid"] == pytest.approx(15) and c["controller"] == pytest.approx(90)
    assert c["study_b"] + c["study_b_fewshot"] == pytest.approx(49)
    # constrained-steps arms: per task and seed 2 shapes x (1.112 + 1.334 + 2.0) with the answered rounding (Q-rounding)
    assert c["main"] == pytest.approx(3 * 5 * (1 + 6 + 2 * (1.112 + 1.334 + 2.0)))
    assert c["treatment"] == pytest.approx(135 + 3 * 5 * (0.10 + 0.25 + 0.50))
    assert c["battery_continuations"] == pytest.approx(435 * 0.2)
    assert c["total"] == pytest.approx(627.13)
    assert c["total"] - c["battery_continuations"] == pytest.approx(540.13)


def test_surplus_plan_follows_part_5_5(monkeypatch) -> None:
    primary = budget.primary_comparison_specs(5)
    # registered units: Study A runs count 1 each (5 arms), Study B 7 x (1 + 4 x 0.1) = 9.8
    assert budget.registered_units(primary) == pytest.approx(5 + 9.8)
    # corrected: training steps (N=0.50 constrained arms train 2T) plus battery continuations 5 x 0.2
    assert budget.corrected_units(primary) == pytest.approx(7 + 9.8 + 1.0)
    none = budget.surplus_plan(machine_hours_allocated=1000, hours_per_run_equivalent=10, concurrent_runs=4)
    assert none.extra_seeds == 0 and none.seeds_per_primary_arm == 5  # capacity 400 < 627.13
    # Table 9.1 (Q-g2-run-equivalents): the surplus is the capacity beyond the corrected total, each seed priced in the
    # same units; the registered computation (484, registered units) is reported beside it and decides nothing
    some = budget.surplus_plan(machine_hours_allocated=(627.13 + 2.5 * 17.8) * 10 / 4, hours_per_run_equivalent=10,
                               concurrent_runs=4)
    assert some.extra_seeds == 2 and some.added_seeds == (5, 6) and some.seeds_per_primary_arm == 7
    assert some.surplus_run_equivalents == pytest.approx(2.5 * 17.8) and some.cost_per_extra_seed == pytest.approx(17.8)
    assert some.registered_surplus_run_equivalents == pytest.approx(627.13 + 2.5 * 17.8 - 484)
    assert some.registered_cost_per_extra_seed == pytest.approx(14.8) and some.registered_extra_seeds == 7
    only_registered = budget.surplus_plan(machine_hours_allocated=(484 + 2.5 * 14.8) * 10 / 4,
                                          hours_per_run_equivalent=10, concurrent_runs=4)
    assert only_registered.registered_extra_seeds == 2  # the registered 484 would give two seeds ...
    assert only_registered.extra_seeds == 0 and only_registered.added_seeds == ()  # ... the corrected total none
    # `python -m pilot surplus` prints these keys; the decisive ones keep their names
    printed = set(budget.as_dict(some))
    assert {"extra_seeds", "seeds_per_primary_arm", "added_seeds", "registered_extra_seeds"} <= printed
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset())  # Q-surplus-arm-set open: its warning
    capped = budget.surplus_plan(machine_hours_allocated=10**7, hours_per_run_equivalent=10, concurrent_runs=4)
    assert capped.seeds_per_primary_arm == R.MAX_SEEDS_PER_ARM and capped.added_seeds[-1] == 11
    assert "Q-surplus-arm-set is open" in capped.warning
    none = budget.surplus_plan(machine_hours_allocated=1000, hours_per_run_equivalent=10, concurrent_runs=4)
    assert none.extra_seeds == 0 and none.warning == ""  # no surplus, nothing waits
    for bad in ({"machine_hours_allocated": 100, "hours_per_run_equivalent": 0}, {"machine_hours_allocated": 0, "hours_per_run_equivalent": 10},
                {"machine_hours_allocated": -1, "hours_per_run_equivalent": 10}, {"machine_hours_allocated": 100, "hours_per_run_equivalent": float("nan")}):
        with pytest.raises(ValueError):
            budget.surplus_plan(concurrent_runs=4, **bad)
    for bad_concurrency in (float("nan"), 4.0, True):
        with pytest.raises(ValueError, match="concurrent_runs must be an integer"):
            budget.surplus_plan(machine_hours_allocated=1000, hours_per_run_equivalent=10, concurrent_runs=bad_concurrency)


def test_the_ramp_arms_surplus_seeds_are_released_once_the_arm_set_is_answered(monkeypatch) -> None:
    """Table 9.1 (Q-surplus-arm-set): every arm Part 5.5 lists, both onset shapes, gets the surplus seeds; the key the
    ramp arms' seeds carry holds them only while it is open."""
    primary = budget.primary_comparison_specs(5)
    tagged = {s.arm for s in primary if "Q-surplus-arm-set" in s.pending}
    assert tagged == {"N0.50-ramp-constrained", "N0.50-ramp-total"}
    assert {s.arm for s in primary if s.study == "A"} == {"N0.00", "N0.50-abrupt-constrained", "N0.50-abrupt-total",
                                                          "N0.50-ramp-constrained", "N0.50-ramp-total"}
    assert all(s.seed == 5 and s.run_id.endswith("-s5") for s in primary)
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset())
    assert {s.arm for s in primary if "Q-surplus-arm-set" in s.open_questions} == tagged  # held while open
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset({"Q-surplus-arm-set"}))
    assert not any("Q-surplus-arm-set" in s.open_questions for s in primary)  # released once answered
    capped = budget.surplus_plan(machine_hours_allocated=10**7, hours_per_run_equivalent=10, concurrent_runs=4)
    assert capped.warning == ""


BASIS = "24 x 120 days from 2026-11-01 to 2027-03-01 x 0.95 availability"


def test_allocation_is_recorded_once(tmp_path) -> None:
    path = tmp_path / "allocation.json"
    rec = budget.record_allocation(5000, "the group", "2026-10-01", BASIS, path)
    assert budget.load_allocation(path)["machine_hours_allocated"] == 5000.0 == rec["machine_hours_allocated"]
    with pytest.raises(FileExistsError):
        budget.record_allocation(6000, "the group", "2026-10-02", BASIS, path)
    with pytest.raises(ValueError):
        budget.record_allocation(-1, "x", "2026-10-01", BASIS, tmp_path / "other.json")
    with pytest.raises(ValueError):
        budget.record_allocation(5000, " ", "2026-10-01", BASIS, tmp_path / "other.json")
    for bad_date in ("1 October", None, 20261001):  # not YYYY-MM-DD, not a string: ValueError, never TypeError
        with pytest.raises(ValueError):
            budget.record_allocation(5000, "the group", bad_date, BASIS, tmp_path / "other.json")
    for no_basis in ("", "  ", None):
        with pytest.raises(ValueError, match="basis"):
            budget.record_allocation(5000, "the group", "2026-10-01", no_basis, tmp_path / "other.json")
    assert not (tmp_path / "other.json").exists()


def test_the_allocation_records_its_unit_basis_and_the_timings_that_existed(tmp_path) -> None:
    """X-allocation (Table 9.1): H is in workstation wall-clock hours (all concurrent runs together), derived from the
    calendar (its basis is stored), and the record lists the workstation timings that already existed."""
    path = tmp_path / "allocation.json"
    rec = budget.record_allocation(5000, "the group", "2026-10-01", BASIS, path,
                                   prior_workstation_timings=["ledger/determinism/study_a.json"])
    stored = budget.load_allocation(path)
    assert stored == rec
    assert stored["unit"] == budget.ALLOCATION_UNIT == "workstation wall-clock hours (all concurrent runs together)"
    assert stored["basis"] == BASIS and stored["prior_workstation_timings"] == ["ledger/determinism/study_a.json"]
    assert budget.record_allocation(5000, "the group", "2026-10-01", BASIS,
                                    tmp_path / "none.json")["prior_workstation_timings"] == []
    # above two years of elapsed time H is likely core-hours: refused unless confirmed
    assert budget.ALLOCATION_CONFIRM_ABOVE == 24 * 731
    with pytest.raises(ValueError, match="core-hours"):
        budget.record_allocation(24 * 731 + 1, "the group", "2026-10-01", BASIS, tmp_path / "big.json")
    assert not (tmp_path / "big.json").exists()
    big = budget.record_allocation(24 * 731 + 1, "the group", "2026-10-01", BASIS, tmp_path / "big.json", confirm=True)
    assert big["machine_hours_allocated"] == 24 * 731 + 1
    assert budget.record_allocation(24 * 731, "the group", "2026-10-01", BASIS,
                                    tmp_path / "two_years.json")["machine_hours_allocated"] == 24 * 731


def test_prior_workstation_timings_list_determinism_reports_and_smoke_ledgers(tmp_path, monkeypatch) -> None:
    from pilot.ledger_writer import smoke_paths

    repo = tmp_path / "repo"
    (repo / "ledger" / "determinism").mkdir(parents=True)
    monkeypatch.setattr(budget.provenance, "REPO_ROOT", repo)
    smoke_root = tmp_path / "smoke"
    assert budget.prior_workstation_timings([smoke_root]) == []
    (repo / "ledger" / "determinism" / "study_a.json").write_text("{}")
    (repo / "ledger" / "determinism" / "README.md").write_text("not a report")
    smoke = smoke_paths(smoke_root)
    smoke.pilot.parent.mkdir(parents=True)
    smoke.pilot.write_bytes(b"parquet")
    assert budget.prior_workstation_timings([smoke_root]) == ["ledger/determinism/study_a.json", str(smoke.pilot)]
    assert budget.prior_workstation_timings() == ["ledger/determinism/study_a.json"]  # smoke roots only as named


def test_prior_workstation_timings_list_a_smoke_roots_trained_runs_and_its_stored_ledgers(tmp_path, monkeypatch) -> None:
    """X-allocation: a smoke run too short to be evaluated (scripts/smoke_run.py's default designs) reaches no
    ledger, but its train_result.json holds its wall-clock hours; the ledgers of a smoke root are where its scheduler
    state stored them (`schedule run --ledger-dir`), not under <data-root>/smoke."""
    from pilot.ledger_writer import LedgerPaths, smoke_paths
    from pilot.scheduler import Scheduler, SchedulerConfig

    base = tmp_path.resolve()  # the stored ledgers are real paths
    root, elsewhere = base / "smoke", base / "elsewhere"
    paths = LedgerPaths(main=elsewhere / "ledger.parquet", pilot=elsewhere / "pilot.parquet",
                        sidecar_dir=elsewhere / "sidecar")
    sched = Scheduler(SchedulerConfig(data_root=root, ledger_paths=paths, allow_dirty=True))
    sched.fix_mode()  # the smoke scheduler stores its mode and its ledgers, as scripts/smoke_run.py does
    sched.db.close()
    monkeypatch.setattr(budget.provenance, "REPO_ROOT", base / "repo")  # no determinism report
    assert budget.prior_workstation_timings([root]) == []
    trained = SchedulerConfig(data_root=root).run_dir("SMOKE-DET-PPOLag-PointGoal1-s0") / "train_result.json"
    trained.parent.mkdir(parents=True)
    trained.write_text(json.dumps({"wall_clock_hours": [1.0]}))
    assert budget.prior_workstation_timings([root]) == [str(trained)]  # trained, never ledgered
    smoke = smoke_paths(root)  # not where this smoke root writes its ledgers
    smoke.pilot.parent.mkdir(parents=True, exist_ok=True)
    smoke.pilot.write_bytes(b"parquet")
    paths.pilot.write_bytes(b"parquet")
    paths.sidecar_dir.mkdir(parents=True, exist_ok=True)
    (paths.sidecar_dir / "SMOKE-DET-PPOLag-PointGoal1-s0.json").write_text("{}")
    assert budget.prior_workstation_timings([root]) == [str(paths.pilot), str(paths.sidecar_dir), str(trained)]


def test_registered_runs_started_reads_the_pilot_ledgers_and_the_registered_scheduler_state(tmp_path) -> None:
    from pilot.ledger_writer import LedgerPaths
    from pilot.manifest import pilot
    from pilot.scheduler import Scheduler, SchedulerConfig

    paths = LedgerPaths(main=tmp_path / "L" / "ledger.parquet", pilot=tmp_path / "L" / "pilot.parquet",
                        sidecar_dir=tmp_path / "L" / "sidecar")
    root = tmp_path / "data"
    assert budget.registered_runs_started(paths, [root]) == []  # no state, no ledger: nothing started
    sched = Scheduler(SchedulerConfig(data_root=root, ledger_paths=paths))
    run = pilot()[0]
    sched.add([run])  # queued, pending: nothing started
    assert budget.registered_runs_started(paths, [root]) == []
    sched.cfg.run_dir(run.run_id).mkdir(parents=True)  # a launch was attempted
    assert "1 run of the data root" in budget.registered_runs_started(paths, [root])[0]
    sched.cfg.run_dir(run.run_id).rmdir()
    sched._set(run.run_id, status="training")
    assert run.run_id in budget.registered_runs_started(paths, [root])[0]
    sched.db.close()
    paths.sidecar_dir.mkdir(parents=True)
    (paths.sidecar_dir / f"{run.run_id}.json").write_text("{}")
    paths.pilot.write_bytes(b"parquet")
    found = budget.registered_runs_started(paths, [])
    assert len(found) == 2 and "pilot ledger" in found[0] and "holds 1 record (e.g." in found[1]


def test_the_allocate_command_needs_a_basis_and_refuses_once_a_run_started(tmp_path, monkeypatch, capsys) -> None:
    from pilot.__main__ import main

    path = tmp_path / "allocation.json"
    monkeypatch.setattr(budget, "ALLOCATION_FILE", path)
    real = budget.record_allocation
    monkeypatch.setattr(budget, "record_allocation", lambda *a, **kw: real(*a, **{**kw, "path": path}))
    argv = ["allocate", "--hours", "2736", "--by", "the group", "--date", "2026-10-05"]
    with pytest.raises(SystemExit) as exc:  # --basis is required (usage error)
        main(argv)
    assert exc.value.code == 2 and not path.exists()
    monkeypatch.setattr(budget, "registered_runs_started", lambda: ["2 runs of the data root /data started"])
    assert main([*argv, "--basis", BASIS]) == 2 and not path.exists()
    assert "already started: 2 runs of the data root /data" in capsys.readouterr().err
    monkeypatch.setattr(budget, "registered_runs_started", lambda: [])
    monkeypatch.setattr(budget, "prior_workstation_timings", lambda roots=(): ["ledger/determinism/study_a.json"])
    assert main([*argv, "--basis", BASIS]) == 0
    out, err = capsys.readouterr()
    assert "warning: 1 record of runs on the workstation" in err and "ledger/determinism/study_a.json" in err
    record = budget.load_allocation(path)
    assert record["basis"] == BASIS and record["prior_workstation_timings"] == ["ledger/determinism/study_a.json"]
    assert ("Table 8.1, 'Machine-hours allocated to the registered runs': 2736 workstation wall-clock hours (all "
            "concurrent runs together), decided by the group on 2026-10-05") in out
    assert ("Table 9.1 row: Machine-hour allocation | 2026-10-05 | <the commit that adds pilot/allocation.json> | 2736 "
            f"workstation wall-clock hours (all concurrent runs together), basis: {BASIS} | Part 6 G2 | "
            "the group") in out
    path.unlink()
    big = ["allocate", "--hours", str(24 * 731 + 1), "--by", "the group", "--date", "2026-10-05", "--basis", BASIS]
    assert main(big) == 1 and not path.exists()  # likely core-hours
    assert "--confirm" in capsys.readouterr().err
    assert main([*big, "--confirm"]) == 0 and budget.load_allocation(path)["machine_hours_allocated"] == 24 * 731 + 1


def test_a_failed_allocation_write_leaves_no_partial_file_and_can_be_retried(tmp_path, monkeypatch) -> None:
    """record_allocation is once only AND atomic: a crash or a full disk between creating the file and
    writing it must not leave an empty file that refuses every retry."""
    path = tmp_path / "allocation.json"
    real_fsync = os.fsync

    def full_disk(fd):
        raise OSError("No space left on device")

    monkeypatch.setattr(os, "fsync", full_disk)
    with pytest.raises(OSError, match="No space left"):
        budget.record_allocation(5000, "the group", "2026-10-01", BASIS, path)
    assert sorted(p.name for p in tmp_path.iterdir()) == []  # no empty record, no temporary file
    monkeypatch.setattr(os, "fsync", real_fsync)

    def crash(*a, **kw):  # the process dies while the record is being produced
        raise KeyboardInterrupt

    monkeypatch.setattr(budget.json, "dumps", crash)
    with pytest.raises(KeyboardInterrupt):
        budget.record_allocation(5000, "the group", "2026-10-01", BASIS, path)
    assert sorted(p.name for p in tmp_path.iterdir()) == []
    monkeypatch.undo()
    budget.record_allocation(5000, "the group", "2026-10-01", BASIS, path)  # the retry is not refused
    assert budget.load_allocation(path)["machine_hours_allocated"] == 5000.0


def test_allocation_contained(tmp_path, monkeypatch) -> None:
    root = tmp_path.resolve()  # a stand-in repository root: nothing is written into the working tree
    monkeypatch.setattr(budget.provenance, "REPO_ROOT", root)
    (root / "pilot").mkdir()
    path = root / "pilot" / "allocation.json"
    rel = str(path.relative_to(root))
    committed = {"HEAD": b"A", "c1": b"A", "c2": b"A", "c3": None}
    monkeypatch.setattr(budget.provenance, "file_committed_at", lambda commit, name: committed[commit] if name == rel else None)
    monkeypatch.setattr(budget.provenance, "dirty_paths", lambda: set())
    ok, note = budget.allocation_contained([{"commit_hash": "c1"}], path)
    assert ok is False and "does not exist" in note
    path.write_bytes(b"A")
    assert budget.allocation_contained([{"commit_hash": "c1"}, {"commit_hash": "c2"}], path)[0] is True
    assert budget.allocation_contained([], path) == (False, "no pilot run records a launch commit yet")
    ok, note = budget.allocation_contained([{"commit_hash": "c1"}, {"commit_hash": "c3"}], path)
    assert ok is False and "c3" in note
    path.write_bytes(b"B")  # edited after the pilot runs were launched
    assert budget.allocation_contained([{"commit_hash": "c1"}], path)[0] is False
    monkeypatch.setattr(budget.provenance, "dirty_paths", lambda: {rel})
    ok, note = budget.allocation_contained([{"commit_hash": "c1"}], path)
    assert ok is False and "uncommitted" in note


def test_battery_continuations_are_counted_as_the_manifest_builds_them() -> None:
    from pilot import manifest

    specs = manifest.design("all")
    parents = budget.battery_parents(specs)
    assert len(parents) == 435 and all(s.group in manifest.STUDY_A_PARENT_GROUPS for s in parents)
    assert budget.battery_parents(manifest.pilot()) == []  # Part 3.6: the pilot's runs are not continued
    assert budget.BATTERY_CONTINUATION_STEPS == R.FINETUNE_STEPS + R.TRANSFER_STEPS
    # a continuation built by the manifest trains exactly these steps
    parent = next(s for s in parents if s.group == "main")
    row = {"run_id": parent.run_id, "matched": True, "completed": True, "matched_checkpoint_step": parent.total_steps,
           "matched_checkpoint_path": "x/epoch-500.pt", "commit_hash": "a" * 40}
    steps = sum(manifest.battery_continuation(parent, row, c).total_steps for c in manifest.BATTERY_CONDITIONS)
    assert steps == budget.BATTERY_CONTINUATION_STEPS


def test_corrected_run_equivalents_after_the_cut_order() -> None:
    from pilot import manifest

    full = budget.corrected_run_equivalents()
    pid = budget.corrected_run_equivalents(cuts=manifest.CUTS[:1])
    assert "pid" not in pid and pid["total"] == pytest.approx(full["total"] - 15 - 15 * 0.2)  # the runs and their continuations
    few = budget.corrected_run_equivalents(cuts=manifest.CUTS[:3])
    car = budget.corrected_run_equivalents(cuts=manifest.CUTS[:2])
    assert car["study_b_fewshot"] == pytest.approx(14.0)
    assert few["study_b_fewshot"] == pytest.approx(140 * 0.02)  # Part 6.1 cut 3: 200,000 steps each
    assert car["total"] - few["total"] == pytest.approx(11.2)  # "about 11 run-equivalents" (Part 6.1)
    with pytest.raises(ValueError, match="prefix"):
        budget.corrected_run_equivalents(cuts=("car",))


def test_cli_budget_prices_the_cuts(capsys) -> None:
    from pilot.__main__ import main

    assert main(["budget", "--cuts", "1"]) == 0
    out = capsys.readouterr().out
    assert "after the cuts pid" in out and "total                    609.13" in out
    assert main(["budget", "--cuts", "9"]) == 2


# ---------------------------------------------------------------------------
# Go or no-go
# ---------------------------------------------------------------------------


@pytest.fixture
def hazard_answered(monkeypatch):
    """Q-hazard answered: G3 is read from the numbers (while it is open G3 is UNDECIDED; pilot/go_decision.py)."""
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset({"Q-hazard"}))


def row(N: float, seed: int, cost: float, gap_h: float | None = None, gap_d: float | None = 1.0, final: float | None = None) -> dict:
    return {
        "run_id": f"P-A-{N}-s{seed}", "study": "A", "task": R.PILOT_STUDY_A_TASK, "N": N, "seed": seed,
        "completed": True, "onset_shape": None if N == 0 else R.PILOT_STUDY_A_SHAPE,
        "step_matching": None if N == 0 else R.PILOT_STUDY_A_CONTROL,
        "treatment": None, "controller_variant": None, "measurement_cost": cost, "final_cost": cost if final is None else final,
        "gap_hazard": gap_h, "gap_dynamics": gap_d, "wall_clock_hours": 10.0,
    }


def pilot_rows(ref=(24.0, 25.0, 26.0), late=(25.0, 25.5, 26.5), gh_ref=(1.0, 2.0, 3.0), gh_late=(8.0, 7.0, 9.0)) -> list[dict]:
    return [row(0.0, s, c, g) for s, (c, g) in enumerate(zip(ref, gh_ref))] + [
        row(0.5, s, c, g) for s, (c, g) in enumerate(zip(late, gh_late))
    ]


def test_g1_passes_and_fails_for_the_registered_reasons() -> None:
    ok = go_decision.g1_feasibility(pilot_rows())
    assert ok.passed is True and ok.values["reference_sd"] == pytest.approx(1.0)
    over_budget = go_decision.g1_feasibility(pilot_rows(late=(27.0, 28.0, 29.0)))
    assert over_budget.passed is False and over_budget.values["both_arms_within_budget"] is False
    spread = go_decision.g1_feasibility(pilot_rows(ref=(10.0, 25.0, 40.0), late=(25.0, 25.0, 25.0)))
    assert spread.values["reference_sd_ok"] is False and spread.passed is False
    unmatched = go_decision.g1_feasibility(pilot_rows(ref=(20.0, 20.0, 20.0), late=(23.0, 23.0, 23.0)))
    assert unmatched.values["arms_matched"] is False and unmatched.passed is False  # |23 - 20| > 2.5
    boundary = go_decision.g1_feasibility(pilot_rows(ref=(25.0, 25.0, 25.0), late=(27.5, 27.5, 27.5)))
    assert boundary.passed is True  # "at most" d + 2.5 and |difference| <= 2.5 are inclusive
    # the final-checkpoint SD of the reference arm is reported beside the selected checkpoint's SD
    finals = pilot_rows()
    for r, final in zip(finals, (20.0, 25.0, 30.0)):
        r["final_cost"] = final
    reported = go_decision.g1_feasibility(finals)
    assert reported.values["reference_final_checkpoint_sd"] == pytest.approx(5.0)
    assert reported.values["reference_sd"] == pytest.approx(1.0)


def test_g1_waits_on_q_g1_level_when_a_seed_is_over_budget_but_the_mean_is_not(monkeypatch) -> None:
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset())  # Q-g1-level open: its other reading is computed
    rows = pilot_rows(ref=(24.0, 24.0, 28.0), late=(25.0, 25.0, 27.0))  # means 25.33 and 25.67; one seed > 27.5
    split = go_decision.g1_feasibility(rows)
    assert split.values["both_arms_within_budget"] is True and split.values["every_seed_within_budget"] is False
    assert split.values["reference_costs"] == [24.0, 24.0, 28.0] and split.values["late_costs"] == [25.0, 25.0, 27.0]
    assert split.passed is None and any("Q-g1-level" in n and "disagree" in n for n in split.notes)
    assert split.undecided_by == "Q-g1-level"
    # the report says the condition waits on the question; it is not "not computable"
    result = go_decision.report(
        rows, hours_per_run=[10.0] * 8, concurrent_runs=8, machine_hours_allocated=1000.0,
        allocation_ok=True, allocation_note="ok", unconstrained_cost=55.0,
        moderate={"mean_cost": {10.0: 9.0, 20.0: 19.0, 40.0: 41.0}},
    )
    assert result["part6_reading"].startswith("UNDECIDED: G1 waits on Q-g1-level")
    assert result["conditions"]["G1 feasibility"]["undecided_by"] == "Q-g1-level"
    md = go_decision.to_markdown(result)
    assert "## Go condition G1 feasibility (Part 6): UNDECIDED (Q-g1-level)" in md and "NOT COMPUTABLE" not in md
    # while Q-g1-level is open: when another clause already fails, the reading does not matter and G1 fails
    assert R.is_open("Q-g1-level")
    unmatched = go_decision.g1_feasibility(pilot_rows(ref=(20.0, 20.0, 28.0), late=(26.0, 26.0, 26.0)))
    assert unmatched.values["every_seed_within_budget"] is False and unmatched.values["arms_matched"] is False
    assert unmatched.passed is False and unmatched.undecided_by is None
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset({"Q-g1-level"}))
    answered = go_decision.g1_feasibility(rows)
    assert answered.passed is True and answered.undecided_by is None  # the answer (the arm-mean reading)


def test_g1_not_computable_without_measurement_costs() -> None:
    rows = pilot_rows()
    rows[0]["measurement_cost"] = None
    g1 = go_decision.g1_feasibility(rows)
    assert g1.passed is None and any("missing measurement_cost in ['P-A-0.0-s0']" in n for n in g1.notes)
    short = go_decision.g1_feasibility(pilot_rows()[1:])  # only the run count is short: nothing is "missing"
    assert short.passed is None and not any("missing" in n for n in short.notes)


def test_g1_and_g3_need_three_completed_seeds_per_arm(hazard_answered) -> None:
    rows = pilot_rows()
    rows[2]["completed"] = False  # reference seed 2 excluded, replacement not finished
    assert go_decision.g1_feasibility(rows).passed is None
    g3 = go_decision.g3_signal(rows)
    assert g3.passed is None and g3.undecided_by is None and not any("missing" in n for n in g3.notes)
    replaced = pilot_rows()
    replaced[2]["seed"] = 5  # the replacement finished: three runs, but only two seed-matched pairs
    assert go_decision.g1_feasibility(replaced).passed is True
    g3 = go_decision.g3_signal(replaced)
    assert g3.passed is True and g3.undecided_by is None  # decided by the rule (Q-g3-pairing)
    weak = pilot_rows(gh_ref=(1.0, 1.0, 1.0), gh_late=(1.5, 2.0, 1.0))
    weak[2]["seed"] = 5
    g3 = go_decision.g3_signal(weak)
    assert g3.passed is False and g3.undecided_by is None and any("seed order" in n for n in g3.notes)


def test_seed_pairs_match_equal_ids_then_the_rest_in_seed_order() -> None:
    assert go_decision._seed_pairs([0, 1, 2], [2, 1, 0]) == [(0, 0), (1, 1), (2, 2)]
    assert go_decision._seed_pairs([0, 1, 5], [0, 1, 2]) == [(0, 0), (1, 1), (5, 2)]
    assert go_decision._seed_pairs([0, 6, 5], [7, 0, 1]) == [(0, 0), (5, 1), (6, 7)]


@pytest.mark.parametrize(("ref_seeds", "late_seeds"), [
    ((0, 1, 2), (0, 1, 2)), ((0, 1, 5), (0, 1, 2)), ((0, 1, 2), (5, 1, 0)), ((0, 5, 6), (7, 0, 1)), ((5, 6, 7), (0, 1, 2)),
])
def test_the_go_report_pairs_the_seeds_as_the_analysis_does(hazard_answered, ref_seeds, late_seeds) -> None:
    """Q-g3-pairing (Table 9.1): pilot/go_decision.py (which imports nothing from analysis/) and analysis.stats.g3_pairs
    apply the same rule, so analysis.pilot_check reproduces the go report's U80 with a replacement seed too."""
    from analysis import stats

    rows = pilot_rows(gh_ref=(1.0, 2.5, 4.0), gh_late=(8.0, 6.5, 9.25))
    for r, seed in zip(rows, (*ref_seeds, *late_seeds)):
        r["seed"] = seed
    ref = {r["seed"]: float(r["gap_hazard"]) for r in rows if r["N"] == 0.0}
    late = {r["seed"]: float(r["gap_hazard"]) for r in rows if r["N"] == 0.5}
    theirs = stats.g3_pairs(late, ref)  # (late seed, reference seed)
    assert go_decision._seed_pairs(ref, late) == [(r, l) for l, r in theirs]
    g3 = go_decision.g3_signal(rows)
    assert g3.values["U80"] == pytest.approx(stats.u80([late[l] - ref[r] for l, r in theirs]), rel=1e-12)
    assert g3.values["U80_pooled"] == pytest.approx(stats.u80_pooled(list(late.values()), list(ref.values())), rel=1e-12)


@pytest.mark.parametrize("answered", [frozenset({"Q-hazard"}), frozenset({"Q-hazard", "Q-g3-pairing"})])
def test_g3_with_a_replacement_seed_pairs_the_unmatched_runs_in_seed_order(monkeypatch, answered) -> None:
    """Table 9.1 (Q-g3-pairing): the runs a replacement seed leaves unpaired are paired in increasing seed order and
    equation (12) is applied unchanged to the three differences; G3 is decided by the rule, never by the group."""
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", answered)
    weak = pilot_rows(gh_ref=(1.0, 1.0, 1.0), gh_late=(1.5, 2.0, 1.0))
    weak[2]["seed"] = 5  # reference seeds 0, 1, 5; late seeds 0, 1, 2
    g3 = go_decision.g3_signal(weak)
    assert g3.values["pairs"] == [[0, 0], [1, 1], [5, 2]] and g3.values["paired_seeds"] == [0, 1]
    assert g3.values["paired_differences"] == [0.5, 1.0, 0.0]
    assert g3.values["U80"] == pytest.approx(go_decision.u80([0.5, 1.0, 0.0]))
    assert g3.passed is False and g3.undecided_by is None
    assert any("unmatched runs paired in seed order" in n and "the same under any pairing" in n for n in g3.notes)
    # the pooled-SD value is reported as a descriptive figure: delta 0.5, s = sqrt((0 + 0.25) / 2)
    assert g3.values["U80_pooled"] == pytest.approx(0.5 + 1.886 * math.sqrt(0.125) / math.sqrt(3))
    wide = pilot_rows(gh_ref=(0.0, 0.0, 0.0), gh_late=(0.0, 3.0, 9.0))
    wide[2]["seed"] = 5
    g3 = go_decision.g3_signal(wide)
    assert g3.values["delta_mean"] == pytest.approx(4.0) and g3.values["U80"] > 5 and g3.passed is True  # U80 decides


def test_u80_matches_equation_12_and_the_t_quantile() -> None:
    diffs = [7.0, 5.0, 6.0]
    expected = statistics.fmean(diffs) + 1.886 * statistics.stdev(diffs) / math.sqrt(3)
    assert go_decision.u80(diffs) == pytest.approx(expected)
    with pytest.raises(ValueError):
        go_decision.u80([1.0, 2.0])


def test_g3_signal_mean_or_upper_bound(hazard_answered) -> None:
    strong = go_decision.g3_signal(pilot_rows())
    assert strong.passed is True and strong.values["delta_mean"] == pytest.approx(6.0)
    # mean below 5 but U80 above 5 still passes (Part 6 G3: "or its upper bound U80 ... is at least 5")
    wide = go_decision.g3_signal(pilot_rows(gh_ref=(0.0, 0.0, 0.0), gh_late=(0.0, 3.0, 9.0)))
    assert wide.values["delta_mean"] == pytest.approx(4.0) and wide.values["U80"] > 5 and wide.passed is True
    weak = go_decision.g3_signal(pilot_rows(gh_ref=(1.0, 1.0, 1.0), gh_late=(1.5, 2.0, 1.0)))
    assert weak.passed is False


def test_g3_compares_at_least_5_after_rounding_as_g1_does(hazard_answered, monkeypatch) -> None:
    """Every paired difference below is exactly 5.00, but fmean gives 4.9999999999999964 and
    U80 4.999999999999998; compared unrounded, G3 failed on floating-point noise (a no-go for Study A)."""
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", R.ANSWERED_QUESTIONS | {"Q-g3-pairing"})
    exact = go_decision.g3_signal(pilot_rows(gh_ref=(11.58, 29.44, 11.9), gh_late=(16.58, 34.44, 16.9)))
    assert exact.values["delta_mean"] < 5 and exact.values["U80"] < 5  # the unrounded values are reported
    assert exact.passed is True
    below = go_decision.g3_signal(pilot_rows(gh_ref=(11.58, 29.44, 11.9), gh_late=(16.57, 34.43, 16.89)))
    assert below.passed is False  # every difference a real 0.01 below the bound still fails


def test_g3_is_provisional_on_q_threshold_arithmetic_while_it_is_open(hazard_answered, monkeypatch) -> None:
    """G3's rounding is Q-threshold-arithmetic's answer, so its PENDING text states the Part 6 reading; G3 carries the
    provisional note while the key is open (reopened here), and the report entry pilot-G3 names the key."""
    from analysis import questions

    text = " ".join(R.PENDING["Q-threshold-arithmetic"].split())
    assert "Part 6 G1 and G3" in text and "U80 and U80_pooled after rounding to 1e-9" in text
    assert "Q-threshold-arithmetic" in questions.keys_for(questions.PILOT_ENTRIES["G3"])
    rows = pilot_rows(gh_ref=(12.63, 28.19, 4.12), gh_late=(17.63, 33.19, 9.12))  # mean and U80 < 5 unrounded
    g3 = go_decision.g3_signal(rows)
    assert g3.values["delta_mean"] < 5
    note = next(n for n in g3.notes if "Q-threshold-arithmetic" in n)
    assert "provisional" in note and f"1e-{go_decision.THRESHOLD_DECIMALS}" in note and f"1e-{go_decision.SD_DECIMALS}" in note
    # while the key is open, a G3 that cannot be computed (no delta_mean) carries no provisional note
    assert R.is_open("Q-threshold-arithmetic")
    assert not any("Q-threshold-arithmetic" in n for n in go_decision.g3_signal(pilot_rows(gh_ref=(1.0, None, 2.0))).notes)
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", R.ANSWERED_QUESTIONS | {"Q-threshold-arithmetic"})
    assert not any("Q-threshold-arithmetic" in n for n in go_decision.g3_signal(rows).notes)


@pytest.mark.parametrize("answered", [frozenset({"Q-hazard"}), frozenset({"Q-hazard", "Q-g3-pairing"})])
def test_g3_is_decided_on_the_paired_form_and_the_pooled_value_is_descriptive(monkeypatch, answered) -> None:
    """Table 9.1 (Q-g3-pairing): equation (12) is the paired form; the pooled-SD value is reported only as a
    descriptive figure, never a reading of equation (12), so it never holds G3."""
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", answered)
    # paired: differences (2, 2, 2), U80 = 2 (fail); pooled: s = 10, 2 + 1.886 * 10 / sqrt(3) (descriptive)
    split = go_decision.g3_signal(pilot_rows(gh_ref=(0.0, 10.0, 20.0), gh_late=(2.0, 12.0, 22.0)))
    assert split.values["U80"] == pytest.approx(2.0)
    assert split.values["U80_pooled"] == pytest.approx(2.0 + 1.886 * 10 / math.sqrt(3))
    assert split.passed is False and split.undecided_by is None
    assert any("descriptive figure" in n for n in split.notes)
    provisional = [n for n in split.notes if "open question Q-g3-pairing" in n]
    assert bool(provisional) == R.is_open("Q-g3-pairing")  # provisional on the key only while it is open


def test_g4_conditioning() -> None:
    ok = go_decision.g4_conditioning(55.0, {10.0: 9.0, 20.0: 19.0, 40.0: 41.0})
    assert ok.passed is True and ok.values["largest_budget"] == 45.0
    unbound = go_decision.g4_conditioning(44.0, {10.0: 9.0, 20.0: 19.0, 40.0: 41.0})
    assert unbound.passed is False  # 45 is not below the unconstrained cost
    disordered = go_decision.g4_conditioning(55.0, {10.0: 21.0, 20.0: 19.0, 40.0: 41.0})
    assert disordered.values["ordered_with_budgets"] is False and disordered.passed is False
    over = go_decision.g4_conditioning(55.0, {10.0: 12.6, 20.0: 19.0, 40.0: 41.0})
    assert over.passed is False
    far_below = go_decision.g4_conditioning(55.0, {10.0: 1.0, 20.0: 5.0, 40.0: 12.0})
    assert far_below.passed is True  # Part 6 is one-sided; Part 3.6's two-sided reading is reported beside it
    assert any("Q-g4-level" in n for n in far_below.notes)
    # string keys (the g4-measure JSON after a round trip) are read as budgets
    assert go_decision.g4_conditioning(55.0, {"10.0": 9.0, "20.0": 19.0, "40.0": 41.0}).passed is True
    assert go_decision.g4_conditioning(None, None).passed is None


@pytest.mark.parametrize(("unconstrained", "moderate", "selection"), [
    (float("nan"), {10.0: 9.0, 20.0: 19.0, 40.0: 41.0}, None),
    (float("inf"), {10.0: 9.0, 20.0: 19.0, 40.0: 41.0}, None),
    (55.0, {10.0: float("nan"), 20.0: 19.0, 40.0: 41.0}, None),
    (55.0, {10.0: 9.0, 20.0: 19.0, 40.0: float("inf")}, None),
    (float("nan"), {10.0: 9.0, 20.0: 19.0, 40.0: 41.0}, 50.0),
])
def test_g4_is_not_computable_from_a_non_finite_cost(unconstrained, moderate, selection) -> None:
    """A NaN cost made "max(budgets) < nan" or the Moderate comparisons False: G4 read FAIL, not NOT COMPUTABLE."""
    cond = go_decision.g4_conditioning(unconstrained, moderate, unconstrained_selection_cost=selection)
    assert cond.passed is None and cond.undecided_by is None
    assert any(n.startswith("not computable") and "finite" in n for n in cond.notes)


@pytest.mark.parametrize("selection", [float("nan"), float("inf")])
def test_a_non_finite_selection_set_cost_leaves_only_that_reading_not_computed(pin_open, selection) -> None:
    """Table 9.1 (Q-final-cost-set): the selection-set cost of the final checkpoint decides nothing, so a non-finite
    one leaves that reading not computed, as G1's does, and G4 is decided on final_cost, even while the key is open."""
    for keys in ((), ("Q-final-cost-set",)):
        pin_open(*keys)
        cond = go_decision.g4_conditioning(55.0, {10.0: 9.0, 20.0: 19.0, 40.0: 41.0},
                                           unconstrained_selection_cost=selection)
        assert cond.passed is True and cond.undecided_by is None, keys
        assert cond.values["unconstrained_selection_set_cost"] is None
        assert cond.values["every_budget_below_unconstrained_selection_set"] is None
        assert any("selection-set reading of Q-final-cost-set is not computed" in n for n in cond.notes)
        assert not any(n.startswith("not computable") for n in cond.notes)


def test_g4_with_the_wrong_budgets_is_not_computable() -> None:
    cond = go_decision.g4_conditioning(55.0, {10.0: 9.0, 20.0: 19.0})
    assert cond.passed is None and any(n.startswith("not computable") for n in cond.notes)


def test_measured_concurrency() -> None:
    t0 = datetime(2026, 10, 1, tzinfo=timezone.utc)
    h = timedelta(hours=1)
    intervals = [(t0, t0 + 10 * h), (t0, t0 + 10 * h), (t0 + 10 * h, t0 + 20 * h)]
    m = go_decision.measured_concurrency(intervals)
    assert m["max"] == 2.0 and m["time_weighted_mean"] == pytest.approx(1.5)
    assert go_decision.measured_concurrency([]) == {"max": 0.0, "time_weighted_mean": 0.0}


def test_g2_needs_every_pilot_run() -> None:
    assert go_decision.g2_throughput([10.0] * 7, 8, 1000.0, True, "ok").passed is None


@pytest.mark.parametrize(("hours", "allocated"), [
    ([float("nan")] + [10.0] * 7, 1000.0),
    ([float("inf")] + [10.0] * 7, 1000.0),
    ([0.0] + [10.0] * 7, 1000.0),
    ([-1.0] + [10.0] * 7, 1000.0),
    ([10.0] * 8, float("nan")),
    ([10.0] * 8, float("inf")),
])
def test_g2_is_not_computable_from_non_finite_or_non_positive_values(hours, allocated) -> None:
    """A NaN wall-clock gave required = NaN and "NaN <= allocated" False: G2 read FAIL instead of NOT COMPUTABLE."""
    cond = go_decision.g2_throughput(hours, 8, allocated, True, "ok")
    assert cond.passed is None and any(n.startswith("not computable") for n in cond.notes)


def test_g2_is_decided_on_the_corrected_total() -> None:
    """Table 9.1 (Q-g2-run-equivalents): G2 is decided on the corrected run-equivalent total; the registered 484 is
    reported beside it and decides nothing."""
    # required 10/8 x 484 = 605 <= 700 < 10/8 x 627.13 = 783.9
    cond = go_decision.g2_throughput([10.0] * 8, 8, 700.0, True, "ok")
    assert cond.passed is False and any("decided on the corrected total" in n for n in cond.notes)
    assert cond.values["required_machine_hours_registered"] == pytest.approx(605.0)
    assert cond.values["required_machine_hours_corrected"] == pytest.approx(10 / 8 * 627.13)
    fits = go_decision.g2_throughput([10.0] * 8, 8, 784.0, True, "ok")
    assert fits.passed is True and not any("corrected" in n for n in fits.notes)


def test_g2_uses_the_allocation_only_if_it_came_first() -> None:
    passed = go_decision.g2_throughput([10.0] * 8, 8, 1000.0, True, "ok")
    assert passed.values["required_machine_hours_corrected"] == pytest.approx(10 / 8 * 627.13) and passed.passed is True
    late = go_decision.g2_throughput([10.0] * 8, 8, 1000.0, False, "after")
    assert late.passed is None
    short = go_decision.g2_throughput([10.0] * 8, 8, 500.0, True, "ok")
    assert short.passed is False


def test_part6_reading_branches() -> None:
    C = go_decision.Condition
    go = go_decision.part6_reading(C("G1 x", True), C("G2 x", True), C("G3 x", True), C("G4 x", True))
    assert go.startswith("GO")
    nogo_a = go_decision.part6_reading(C("G1 x", True), C("G2 x", True), C("G3 x", False), C("G4 x", True))
    assert "NO-GO for Study A" in nogo_a and "Study B becomes the main study" in nogo_a
    revise = go_decision.part6_reading(C("G1 x", False), C("G2 x", True), C("G3 x", True), C("G4 x", True))
    assert revise.startswith("REVISE ONCE for G1")
    incomplete = go_decision.part6_reading(C("G1 x", None), C("G2 x", True), C("G3 x", True), C("G4 x", True))
    assert incomplete.startswith("INCOMPLETE") and "UNDECIDED" not in incomplete
    g3_incomplete = go_decision.part6_reading(C("G1 x", True), C("G2 x", True), C("G3 x", None), C("G4 x", True))
    assert g3_incomplete == "INCOMPLETE: G3 cannot be evaluated yet."  # G1 and G2 hold, so G3 decides
    both = go_decision.part6_reading(C("G1 x", None, undecided_by="Q-g1-level"), C("G2 x", None), C("G3 x", True), C("G4 x", True))
    assert both.startswith("INCOMPLETE: G2 cannot be evaluated yet. UNDECIDED: G1 waits on Q-g1-level")
    again_g1 = go_decision.part6_reading(C("G1 x", False), C("G2 x", True), C("G3 x", True), C("G4 x", True), failed_before={"G1"})
    assert again_g1 == "NO-GO for Study A: G1 failed again after the revision."
    again_g4 = go_decision.part6_reading(C("G1 x", True), C("G2 x", True), C("G3 x", True), C("G4 x", False), failed_before={"G4"})
    assert "NO-GO for Study B" in again_g4
    again_g2 = go_decision.part6_reading(C("G1 x", True), C("G2 x", False), C("G3 x", True), C("G4 x", True), failed_before={"G2"})
    assert "cut order of Part 6.1" in again_g2 and "Never cut the number of seeds" in again_g2
    # only a second failure of the SAME condition is a no-go
    first_g1 = go_decision.part6_reading(C("G1 x", False), C("G2 x", True), C("G3 x", True), C("G4 x", True), failed_before={"G4"})
    assert "NO-GO" not in first_g1 and "G1 failed for the first time on the re-pilot" in first_g1
    repaired = go_decision.part6_reading(C("G1 x", True), C("G2 x", True), C("G3 x", True), C("G4 x", True), failed_before={"G1"})
    assert repaired.startswith("GO")
    with pytest.raises(ValueError):
        go_decision.part6_reading(C("G1 x", True), C("G2 x", True), C("G3 x", True), C("G4 x", True), failed_before={"G3"})


def test_part6_reading_reports_g3_alongside_other_failures() -> None:
    C = go_decision.Condition
    # first pilot: a revisable failure with G3 failing -> G3 is read again after the revision
    both = go_decision.part6_reading(C("G1 x", False), C("G2 x", True), C("G3 x", False), C("G4 x", True))
    assert both.startswith("REVISE ONCE for G1") and "G3 also failed; it is read again after the revision of G1." in both
    assert "NO-GO" not in both
    # first pilot: G4 fails, G1 and G2 hold, G3 fails -> revise G4 and no-go for Study A
    g4_g3 = go_decision.part6_reading(C("G1 x", True), C("G2 x", True), C("G3 x", False), C("G4 x", False))
    assert "REVISE ONCE for G4" in g4_g3 and "NO-GO for Study A" in g4_g3
    # re-pilot: only G3 fails -> no-go for Study A
    only_g3 = go_decision.part6_reading(C("G1 x", True), C("G2 x", True), C("G3 x", False), C("G4 x", True), failed_before={"G1"})
    assert only_g3.startswith("NO-GO for Study A") and "Study B becomes the main study" in only_g3
    # re-pilot: G2 fails again and G3 fails -> the cut order, and G3's failure is not dropped
    g2_g3 = go_decision.part6_reading(C("G1 x", True), C("G2 x", False), C("G3 x", False), C("G4 x", True), failed_before={"G2"})
    assert "cut order of Part 6.1" in g2_g3 and "G3 also failed on the re-pilot" in g2_g3
    # re-pilot: G1 fails again -> Study A is a no-go, so a G3 failure raises no question
    g1_g3 = go_decision.part6_reading(C("G1 x", False), C("G2 x", True), C("G3 x", False), C("G4 x", True), failed_before={"G1"})
    assert g1_g3 == "NO-GO for Study A: G1 failed again after the revision."


def test_an_undecided_g3_does_not_withhold_a_branch_that_g1_or_g2_already_fixes() -> None:
    # Part 6 reads G3 only when G1 and G2 hold, so once either fails an undecided G3 is moot for the branch
    C = go_decision.Condition
    g3 = C("G3 x", None, undecided_by="Q-hazard")
    again_g1 = go_decision.part6_reading(C("G1 x", False), C("G2 x", True), g3, C("G4 x", True), failed_before={"G1"})
    assert again_g1 == "NO-GO for Study A: G1 failed again after the revision."
    first_g1 = go_decision.part6_reading(C("G1 x", False), C("G2 x", True), g3, C("G4 x", True))
    assert first_g1.startswith("REVISE ONCE for G1") and "UNDECIDED" not in first_g1
    assert "G3 is undecided (Q-hazard); it is read again after the revision of G1." in first_g1
    first_g2 = go_decision.part6_reading(C("G1 x", True), C("G2 x", False), g3, C("G4 x", True))
    assert first_g2.startswith("REVISE ONCE for G2") and "G3 is undecided (Q-hazard)" in first_g2
    again_g2 = go_decision.part6_reading(C("G1 x", True), C("G2 x", False), g3, C("G4 x", True), failed_before={"G2"})
    assert "cut order of Part 6.1" in again_g2 and "G3 is undecided (Q-hazard) on the re-pilot" in again_g2
    # a G3 that cannot be computed is moot in the same way, and is called not computable, not undecided
    blank = C("G3 x", None)
    first_g1 = go_decision.part6_reading(C("G1 x", False), C("G2 x", True), blank, C("G4 x", True))
    assert first_g1.startswith("REVISE ONCE for G1") and "INCOMPLETE" not in first_g1
    assert "G3 is not computable; it is read again after the revision of G1." in first_g1
    again_g2 = go_decision.part6_reading(C("G1 x", True), C("G2 x", False), blank, C("G4 x", True), failed_before={"G2"})
    assert "G3 is not computable on the re-pilot" in again_g2 and "None" not in again_g2
    # while G1 and G2 hold, G3 decides the branch and the reading waits for the group
    held = go_decision.part6_reading(C("G1 x", True), C("G2 x", True), g3, C("G4 x", False))
    assert held.startswith("UNDECIDED: G3 waits on Q-hazard")


def test_a_replacement_seed_leaves_no_go_condition_to_the_group(monkeypatch) -> None:
    """Table 9.1 (Q-g3-pairing): with a replacement seed G3 is decided by the rule, so Part 6's branch follows
    from the numbers (here a no-go for Study A), never 'UNDECIDED: G3 waits on Q-g3-pairing'."""
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset({"Q-g3-pairing", "Q-hazard"}))
    weak = pilot_rows(gh_ref=(1.0, 1.0, 1.0), gh_late=(1.5, 2.0, 1.0))
    weak[2]["seed"] = 5
    C = go_decision.Condition
    reading = go_decision.part6_reading(C("G1 x", True), C("G2 x", True), go_decision.g3_signal(weak), C("G4 x", True))
    assert reading.startswith("NO-GO for Study A") and "Q-g3-pairing" not in reading


def test_report_and_table_8_1(tmp_path, hazard_answered) -> None:
    result = go_decision.report(
        pilot_rows(), hours_per_run=[10.0] * 8, concurrent_runs=8, machine_hours_allocated=1000.0,
        allocation_ok=True, allocation_note="ok", unconstrained_cost=55.0,
        moderate={"mean_cost": {10.0: 9.0, 20.0: 19.0, 40.0: 41.0}, "satisfaction": {10.0: 0.9}},
    )
    assert result["part6_reading"].startswith("GO")
    table = result["table_8_1"]
    assert table["Reference arm's cost under the dynamics perturbation (mean over seeds)"] == pytest.approx(26.0)
    # Table 9.1 (Q-p-list): the row Table 8.1 gains, under its name
    assert table["Moderate arm's satisfaction rates at each of its training budgets"] == {10.0: 0.9}
    # a mean over fewer than the pilot's seeds is not reported under a "mean over seeds" label
    partial = pilot_rows()
    partial[0]["gap_dynamics"] = None
    short = go_decision.report(
        partial, hours_per_run=[10.0] * 8, concurrent_runs=8, machine_hours_allocated=1000.0, allocation_ok=True,
        allocation_note="ok", unconstrained_cost=55.0, moderate=None,
    )
    assert short["table_8_1"]["Reference arm's cost under the dynamics perturbation (mean over seeds)"] is None
    # the perturbed cost is rebuilt from measurement_cost, never from the selection-set cost
    rows = pilot_rows()
    for r in rows:
        r["selection_cost_at_match"] = r["measurement_cost"] - 3.0
    other = go_decision.report(
        rows, hours_per_run=[10.0] * 8, concurrent_runs=8, machine_hours_allocated=1000.0, allocation_ok=True,
        allocation_note="ok", unconstrained_cost=55.0, moderate=None,
    )
    assert other["table_8_1"]["Reference arm's cost under the dynamics perturbation (mean over seeds)"] == pytest.approx(26.0)
    # G1 is read on the measurement set (the proposal); the selection-set means are reported beside it, and the
    # selection set can hold G1 UNDECIDED only while Q-matched-cost-set is open and its reading changes G1
    g1 = other["conditions"]["G1 feasibility"]["values"]
    assert g1["cost_field"] == "measurement_cost"
    assert g1["selection_set_means"] == [pytest.approx(22.0), pytest.approx(22.6666, rel=1e-3)]
    assert result["repilot"] is False and result["failed_before"] == []
    assert table["Seed-to-seed SD of final in-distribution cost, N = 0.50 arm"] == pytest.approx(statistics.stdev([25.0, 25.5, 26.5]))
    again = go_decision.report(
        pilot_rows(), hours_per_run=[10.0] * 8, concurrent_runs=8, machine_hours_allocated=500.0,
        allocation_ok=True, allocation_note="ok", unconstrained_cost=55.0,
        moderate={"mean_cost": {10.0: 9.0, 20.0: 19.0, 40.0: 41.0}}, failed_before=("G2", "G2"),
    )
    assert again["repilot"] is True and again["failed_before"] == ["G2"]
    assert again["part6_reading"].startswith("G2 failed again after the revision")
    js, md = go_decision.write_report(result, tmp_path, "20261001T000000Z")
    text = md.read_text()
    assert js.exists() and "## Go condition G1 feasibility (Part 6): PASS" in text
    assert text.startswith("# Pilot go or no-go report: go conditions G1 to G4 (Part 6; Table 8.1)")
    with pytest.raises(FileExistsError):
        go_decision.write_report(result, tmp_path, "20261001T000000Z")  # reports are never overwritten


def test_cli_reports_expected_errors_without_a_traceback(capsys, monkeypatch) -> None:
    from pilot.__main__ import main

    before = budget.ALLOCATION_FILE.read_bytes() if budget.ALLOCATION_FILE.exists() else None  # committed after task 12
    monkeypatch.setattr(budget, "registered_runs_started", lambda: [])  # whatever this machine's /data holds
    basis = ["--basis", BASIS]
    assert main(["allocate", "--hours", "-5", "--by", "the group", "--date", "2026-10-01", *basis]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ValueError") and "Traceback" not in err
    assert main(["allocate", "--hours", "5", "--by", "the group", "--date", "1 October", *basis]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ValueError") and "Traceback" not in err
    after = budget.ALLOCATION_FILE.read_bytes() if budget.ALLOCATION_FILE.exists() else None
    assert after == before  # a refused allocation writes nothing


# ---------------------------------------------------------------------------
# pilot/go_decision.py: G1, G3 and G4 wait on the open report keys whose readings disagree
# ---------------------------------------------------------------------------


def _with_selection(rows: list[dict], costs) -> list[dict]:
    for r, c in zip(rows, costs):
        r["selection_cost_at_match"] = c
    return rows


def test_g1_waits_on_q_matched_cost_set_when_the_selection_set_changes_it(monkeypatch) -> None:
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset())  # Q-matched-cost-set open
    rows = _with_selection(pilot_rows(), [29.0] * 6)  # selection-set means 29 > 27.5: G1 would fail
    g1 = go_decision.g1_feasibility(rows)
    assert g1.passed is None and g1.undecided_by == "Q-matched-cost-set"
    assert g1.values["selection_set_reading"]["passed"] is False and g1.values["reference_cost_mean"] == pytest.approx(25.0)
    assert any("Q-matched-cost-set" in n and "selection set" in n for n in g1.notes)
    result = go_decision.report(rows, hours_per_run=[10.0] * 8, concurrent_runs=8, machine_hours_allocated=1000.0,
                                allocation_ok=True, allocation_note="ok", unconstrained_cost=55.0,
                                moderate={"mean_cost": {10.0: 9.0, 20.0: 19.0, 40.0: 41.0}})
    assert "G1 waits on Q-matched-cost-set" in result["part6_reading"]
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset({"Q-matched-cost-set"}))
    assert go_decision.g1_feasibility(rows).passed is True  # the answer (the measurement set)


def test_g1_is_decided_when_both_cost_sets_agree(pin_open) -> None:
    pin_open("Q-matched-cost-set")  # the selection-set reading counts only while the key is open
    agree = go_decision.g1_feasibility(_with_selection(pilot_rows(), [24.0, 25.0, 26.0, 25.0, 25.5, 26.5]))
    assert agree.values["selection_set_reading"]["passed"] is True
    assert agree.passed is True and agree.undecided_by is None
    both_fail = go_decision.g1_feasibility(_with_selection(pilot_rows(late=(28.0, 29.0, 30.0)), [29.0] * 6))
    assert both_fail.passed is False
    missing = go_decision.g1_feasibility(pilot_rows())  # no selection-set cost: its reading is not computed
    assert missing.passed is True and missing.values["selection_set_reading"] is None
    assert any("not computed" in n for n in missing.notes)


def test_g1_waits_on_both_keys_when_only_their_combined_reading_changes_it(monkeypatch) -> None:
    """The measurement set passes on arm means and seed by seed, the selection set passes on arm means, but one
    selection-set seed is 28 > 27.5: only the two open keys read together fail G1, so both hold it."""
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset())
    rows = _with_selection(pilot_rows(), [24.0, 24.0, 28.0, 25.0, 25.0, 27.0])
    g1 = go_decision.g1_feasibility(rows)
    assert g1.values["every_seed_within_budget"] is True and g1.values["selection_set_reading"]["passed"] is True
    assert g1.values["selection_set_reading"]["passed_every_seed"] is False
    assert g1.passed is None and g1.undecided_by == "Q-g1-level, Q-matched-cost-set"
    assert any("selection set read seed by seed" in n for n in g1.notes)
    for answered in ({"Q-g1-level"}, {"Q-matched-cost-set"}):  # one key answered: its answer holds, G1 passes
        monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset(answered))
        g1 = go_decision.g1_feasibility(rows)
        assert g1.passed is True and g1.undecided_by is None


def test_g1_says_why_q_g1_level_holds_it_when_the_selection_set_readings_disagree(monkeypatch) -> None:
    """The measurement set fails on arm means and seed by seed (29 > 27.5); the selection set passes on arm means but
    one seed is 28 > 27.5. Both keys hold G1, and a note says why Q-g1-level is one of them."""
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset())
    rows = _with_selection(pilot_rows(ref=(29.0, 29.0, 29.0), late=(29.0, 29.0, 29.0)), [24.0, 24.0, 28.0, 25.0, 25.0, 27.0])
    g1 = go_decision.g1_feasibility(rows)
    assert g1.passed is None and g1.undecided_by == "Q-g1-level, Q-matched-cost-set"
    assert any("on the selection set the arm-mean and per-seed readings" in n and "Q-g1-level" in n for n in g1.notes)
    # the note is not given when the selection set's two readings agree
    agree = go_decision.g1_feasibility(_with_selection(pilot_rows(ref=(29.0, 29.0, 29.0), late=(29.0, 29.0, 29.0)),
                                                       [24.0, 25.0, 26.0, 25.0, 25.5, 26.5]))
    assert agree.undecided_by == "Q-matched-cost-set"
    assert not any("on the selection set the arm-mean and per-seed readings" in n for n in agree.notes)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_g1_is_not_computable_from_a_non_finite_cost(bad) -> None:
    """A NaN cost gave a NaN mean, and "round(nan) <= limit" is False: G1 read FAIL instead of NOT COMPUTABLE."""
    rows = pilot_rows()
    rows[0]["measurement_cost"] = bad
    g1 = go_decision.g1_feasibility(rows)
    assert g1.passed is None and g1.undecided_by is None
    assert any(n.startswith("not computable") and "must be finite" in n for n in g1.notes)
    # a non-finite selection-set cost leaves that reading not computed; it never holds G1
    sel = _with_selection(pilot_rows(), [24.0, 25.0, 26.0, 25.0, 25.5, bad])
    g1 = go_decision.g1_feasibility(sel)
    assert g1.passed is True and g1.values["selection_set_reading"] is None and g1.undecided_by is None
    assert g1.values["selection_set_means"][1] is None and any("not computed" in n for n in g1.notes)


def test_g1_names_every_key_that_holds_it(monkeypatch) -> None:
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset())
    rows = _with_selection(pilot_rows(ref=(24.0, 24.0, 28.0), late=(25.0, 25.0, 27.0)), [29.0] * 6)
    g1 = go_decision.g1_feasibility(rows)
    assert g1.passed is None and g1.undecided_by == "Q-g1-level, Q-matched-cost-set"
    md = go_decision.to_markdown({"part6_reading": "", "conditions": {"G1": {
        "passed": None, "undecided_by": g1.undecided_by, "values": g1.values, "notes": g1.notes}}, "table_8_1": {}})
    assert "## Go condition G1 (Part 6): UNDECIDED (Q-g1-level, Q-matched-cost-set)" in md


def test_g3_waits_on_q_hazard_while_it_is_open(monkeypatch) -> None:
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset())
    assert R.is_open("Q-hazard")
    strong = go_decision.g3_signal(pilot_rows())
    assert strong.passed is None and strong.undecided_by == "Q-hazard"
    assert strong.values["delta_mean"] == pytest.approx(6.0)  # the numbers are still reported
    split = go_decision.g3_signal(pilot_rows(gh_ref=(0.0, 10.0, 20.0), gh_late=(2.0, 12.0, 22.0)))
    assert split.undecided_by == "Q-hazard"  # the pooled-SD value is descriptive: Q-g3-pairing never holds G3
    missing = pilot_rows()
    missing[0]["gap_hazard"] = None  # the harness refuses hazard while the key is open
    assert go_decision.g3_signal(missing).undecided_by == "Q-hazard"
    C = go_decision.Condition
    reading = go_decision.part6_reading(C("G1 x", True), C("G2 x", True), strong, C("G4 x", True))
    assert reading.startswith("UNDECIDED: G3 waits on Q-hazard")
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset({"Q-hazard"}))
    assert go_decision.g3_signal(pilot_rows()).passed is True


def test_g4_waits_on_q_final_cost_set_when_the_selection_set_changes_it(monkeypatch) -> None:
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset())  # Q-final-cost-set open
    moderate = {10.0: 9.0, 20.0: 19.0, 40.0: 41.0}
    split = go_decision.g4_conditioning(55.0, moderate, unconstrained_selection_cost=44.0)  # 45 is not below 44
    assert split.passed is None and split.undecided_by == "Q-final-cost-set"
    # an UNDECIDED G4 carries no unqualified "passes" note for Q-g4-level
    far_below = go_decision.g4_conditioning(55.0, {10.0: 1.0, 20.0: 5.0, 40.0: 12.0}, unconstrained_selection_cost=44.0)
    assert far_below.passed is None
    assert all(n.startswith("on the measurement-set reading") for n in far_below.notes if "Q-g4-level" in n)
    assert split.values["unconstrained_selection_set_cost"] == 44.0
    assert split.values["every_budget_below_unconstrained_selection_set"] is False
    agree = go_decision.g4_conditioning(55.0, moderate, unconstrained_selection_cost=50.0)
    assert agree.passed is True and agree.undecided_by is None
    unknown = go_decision.g4_conditioning(55.0, moderate)
    assert unknown.passed is True and any("not computed" in n for n in unknown.notes)
    result = go_decision.report(pilot_rows(), hours_per_run=[10.0] * 8, concurrent_runs=8,
                                machine_hours_allocated=1000.0, allocation_ok=True, allocation_note="ok",
                                unconstrained_cost=55.0, unconstrained_selection_cost=44.0,
                                moderate={"mean_cost": moderate})
    assert result["conditions"]["G4 conditioning"]["undecided_by"] == "Q-final-cost-set"
    assert "G4 waits on Q-final-cost-set" in result["part6_reading"]
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset({"Q-final-cost-set"}))
    assert go_decision.g4_conditioning(55.0, moderate, unconstrained_selection_cost=44.0).passed is True


def test_the_values_analysis_pilot_check_reads_are_kept(hazard_answered) -> None:
    from analysis import pilot_check

    g1 = go_decision.g1_feasibility(_with_selection(pilot_rows(), [29.0] * 6))
    g3 = go_decision.g3_signal(pilot_rows())
    for key in pilot_check.G1_KEYS:
        assert key in g1.values
    for key in pilot_check.G3_KEYS:
        assert key in g3.values


@pytest.mark.parametrize(("ref", "late"), [
    ((24.9, 24.9, 24.9), (27.3, 27.3, 27.6)),  # arm means 24.90 and 27.40: 2.500000000000007 in binary floating point
    ((24.9, 24.9, 24.91), (27.3, 27.3, 27.61)),  # 2.5000000000000036
])
def test_g1_compares_after_rounding_as_part_4_does(monkeypatch, ref, late) -> None:
    """Part 6 G1: "the two arms can be matched under the tolerance of Part 4": rule 5 (inclusive, after
    rounding to 1e-4 under the answer of Q-threshold-arithmetic), not an unrounded float comparison."""
    from analysis import matching
    from pilot import enrichment

    assert go_decision.THRESHOLD_DECIMALS == enrichment.MATCH_DECIMALS == matching.THRESHOLD_DECIMALS
    difference = statistics.fmean(late) - statistics.fmean(ref)
    assert difference > R.MATCH_TOLERANCE and round(difference, 4) == R.MATCH_TOLERANCE  # the noise is real
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset(R.PENDING))  # every key answered: G1 is decided
    g1 = go_decision.g1_feasibility(pilot_rows(ref=ref, late=late))
    assert g1.values["arms_matched"] is True and g1.passed is True
    assert g1.values["match_difference"] == difference  # reported unrounded
    # the same clause as the Role 4 check computes it (analysis.pilot_check compares the two)
    assert g1.values["arms_matched"] == matching.within_tolerance(statistics.fmean(late), statistics.fmean(ref),
                                                                   R.MATCH_TOLERANCE)


def test_the_reference_sd_is_not_rounded_down_to_10() -> None:
    """The reference SD is a square root, not on the 0.01 grid of mean costs, so rounding it to
    1e-4 moves a real boundary: these costs have a sample SD of 10.000004, which "exceeds 10 cost units" (rule 5) and
    is not "at most 10" (G1). Rule 5 (both implementations) and G1 compare it at noise level only (1e-9)."""
    from analysis import matching
    from pilot import enrichment

    costs = [55.04, 34.19, 45.27, 32.03, 50.25]
    sd = statistics.stdev(costs)
    assert sd > R.INFEASIBLE_REFERENCE_SD == R.G1_REFERENCE_SD_MAX and round(sd, 4) == 10.0
    assert go_decision.SD_DECIMALS == enrichment.MATCH_SD_DECIMALS == matching.SD_DECIMALS
    assert matching.spread_feasible(sd) is False
    assert enrichment.PROPOSED_READING.rounded_sd(sd) > R.INFEASIBLE_REFERENCE_SD
    assert go_decision._g1_clauses(costs, costs)["reference_sd_ok"] is False
    # an SD of exactly 10 up to floating-point noise stays feasible (rule 5: "exceeds"; G1: "at most")
    assert matching.spread_feasible(10.0 + 1e-12) and go_decision._at_most(10.0 + 1e-12, 10.0, go_decision.SD_DECIMALS)
    assert enrichment.PROPOSED_READING.rounded_sd(10.0 + 1e-12) == 10.0


def test_g1_noise_in_the_selection_set_is_not_an_undecided_reading(monkeypatch) -> None:
    """The selection-set reading of Q-matched-cost-set is compared the same way: noise alone never makes G1 UNDECIDED."""
    rows = _with_selection(pilot_rows(ref=(25.0, 25.0, 25.0), late=(27.5, 27.5, 27.5)),
                           [24.9, 24.9, 24.9, 27.3, 27.3, 27.4])  # selection-set means 24.90 and 27.3333
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset({"Q-g1-level"}))
    assert R.is_open("Q-matched-cost-set") and R.is_open("Q-threshold-arithmetic")
    g1 = go_decision.g1_feasibility(rows)
    assert g1.values["selection_set_reading"]["passed"] is True and g1.passed is True and g1.undecided_by is None
    noisy = _with_selection(pilot_rows(ref=(25.0, 25.0, 25.0), late=(27.5, 27.5, 27.5)),
                            [24.9, 24.9, 24.9, 27.3, 27.3, 27.6])  # 2.500000000000007 on the selection set
    g1 = go_decision.g1_feasibility(noisy)
    assert g1.values["selection_set_reading"]["match_difference"] > R.MATCH_TOLERANCE
    assert g1.values["selection_set_reading"]["passed"] is True and g1.undecided_by is None
    assert any("Q-threshold-arithmetic" in n and "provisional" in n for n in g1.notes)
    # the note states the rounding the code applies (the SD at SD_DECIMALS, not at 1e-4)
    note = next(n for n in g1.notes if "Q-threshold-arithmetic" in n)
    assert f"1e-{go_decision.THRESHOLD_DECIMALS}" in note and f"SD after rounding to 1e-{go_decision.SD_DECIMALS}" in note
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset(R.PENDING))
    assert not any("Q-threshold-arithmetic" in n for n in go_decision.g1_feasibility(noisy).notes)


def test_the_go_reports_decided_values_carry_their_keys() -> None:
    """HANDOVER.md section 10: a value Table 9.1 decides is a named constant commented ``# answered in Table 9.1
    (Q-key)``. The measurement set of G1 is the answer of Q-matched-cost-set (its PENDING text), as in
    analysis/matching.py."""
    from analysis import matching

    source = inspect.getsource(go_decision).splitlines()
    for name, key in (("COST_FIELD", "Q-matched-cost-set"), ("THRESHOLD_DECIMALS", "Q-threshold-arithmetic")):
        at = next(i for i, line in enumerate(source) if line.startswith(f"{name} = "))
        above = at  # the constant's line and the comment block directly above or below it
        while above > 0 and source[above - 1].startswith("#"):
            above -= 1
        below = at + 1
        while below < len(source) and source[below].startswith("#"):
            below += 1
        assert f"# answered in Table 9.1 ({key})" in "\n".join(source[above:below]), f"{name} is not marked as {key}'s"
        assert key in R.PENDING and not R.is_open(key)
    assert "# PROPOSAL" not in "\n".join(source)
    assert go_decision.COST_FIELD == matching.COST_FIELD
    assert go_decision.SELECTION_COST_FIELD == matching.SELECTION_COST_FIELD


def test_the_go_report_is_written_whole_and_once(tmp_path, monkeypatch) -> None:
    """write_report opened both files with mode 'x' and then wrote them, so a crash or a full
    disk left an empty or truncated go_report under results/pilot/. Both go through provenance.write_text_once, the
    .json last; a failed .json write removes the .md."""
    from pilot import provenance

    result = {"conditions": {}, "table_8_1": {}}
    real = provenance.write_text_once

    def failing(path, text):
        if Path(path).suffix == ".json":
            raise OSError("no space left on device")
        real(path, text)

    monkeypatch.setattr(go_decision, "to_markdown", lambda r: "# go report\n")
    monkeypatch.setattr(provenance, "write_text_once", failing)
    with pytest.raises(OSError, match="no space"):
        go_decision.write_report(result, tmp_path, "20261001T000000Z")
    assert list(tmp_path.iterdir()) == []
    monkeypatch.setattr(provenance, "write_text_once", real)
    js, md = go_decision.write_report(result, tmp_path, "20261001T000000Z")
    assert json.loads(js.read_text()) == result and md.read_text() == "# go report\n"
    with pytest.raises(FileExistsError):
        go_decision.write_report(result, tmp_path, "20261001T000000Z")
    assert sorted(p.name for p in tmp_path.iterdir()) == [js.name, md.name]

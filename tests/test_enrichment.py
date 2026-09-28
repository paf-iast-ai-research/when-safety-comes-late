"""Enrichment: rule-1 selection and battery gaps written to the ledger (Part 4.1 rule 1; eq. 1)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import fake_launcher  # noqa: E402

from pilot import enrichment  # noqa: E402
from pilot.contracts import ContractError  # noqa: E402
from pilot.ledger_writer import LedgerPaths, write_run  # noqa: E402
from pilot.manifest import RunSpec  # noqa: E402


def _run(tmp: Path, seed: int) -> tuple[RunSpec, Path]:
    spec = RunSpec(run_id=f"T-A-PointGoal1-N0.00-s{seed}", study="A", task="SafetyPointGoal1-v0", arm="N0.00", seed=seed,
                   total_steps=2_000_000, base_algo="PPOLag", plugin="ppolag", group="main", N=0.0, onset_step=0)
    run_dir = tmp / "runs" / spec.run_id
    run_dir.mkdir(parents=True)
    (run_dir / "spec.json").write_text(spec.to_json())
    omni = fake_launcher.write_outputs(spec.to_dict(), run_dir)
    (run_dir / "train_result.json").write_text(json.dumps(fake_launcher.train_result(spec.to_dict(), omni, "completed", None)))
    steps = list(range(200_000, 2_000_001, 200_000))
    costs = [30, 29, 27, 26, 25.4, 24.9, 23, 22, 21, 20]  # closest to 25: 24.9 at step 1,200,000
    ev = {"final_cost": 20.0, "final_return": 5.0, "episodes": 100, "eval_wall_clock_hours": 0.1,
          "selection": {str(s): [c, 1.0] for s, c in zip(steps, costs)}, "selection_seeds": fake_launcher.SELECTION_SEEDS}
    (run_dir / "evaluation.json").write_text(json.dumps(ev))
    return spec, run_dir


def _closest_to_25(checkpoints: list[dict]) -> int:
    window = [c for c in checkpoints if c.get("selection_cost") is not None]
    return min(window, key=lambda c: (abs(c["selection_cost"] - 25.0), -c["step"]))["step"]


def test_selection_then_battery(tmp_path) -> None:
    paths = LedgerPaths(main=tmp_path / "ledger.parquet", pilot=tmp_path / "p.parquet", sidecar_dir=tmp_path / "s")
    for seed in (0, 1):
        spec, run_dir = _run(tmp_path, seed)
        write_run(spec, run_dir, paths)
    done = enrichment.apply_selection(paths.main, selector=_closest_to_25)
    assert sorted(done) == ["T-A-PointGoal1-N0.00-s0", "T-A-PointGoal1-N0.00-s1"]
    assert enrichment.apply_selection(paths.main, selector=_closest_to_25) == []  # never selected twice

    calls = []

    def fake_battery(omnisafe_dir, spec, step, conditions):
        calls.append((spec["run_id"], step, tuple(conditions)))
        return {"measurement": 25.1, "hazard": 31.1, "dynamics": 27.6, "episodes": 100,
                "measurement_seeds": fake_launcher.MEASUREMENT_SEEDS}

    run_dir_of = lambda rid: tmp_path / "runs" / rid  # noqa: E731
    done_first = enrichment.apply_battery(paths.main, run_dir_of, ["hazard"], evaluator=fake_battery)
    assert len(done_first) == 2 and calls[0][1] == 1_200_000 and calls[0][2] == ("hazard",)
    # a condition left out earlier is added later, and the measurement cost must reproduce
    done = enrichment.apply_battery(paths.main, run_dir_of, ["hazard", "dynamics"], evaluator=fake_battery)
    assert len(done) == 2 and calls[-1][2] == ("dynamics",)
    from results.ledger_schema import load_ledger_as_rows

    row = load_ledger_as_rows(paths.main)[0]
    assert row.matched_checkpoint_step == row.training_age == 1_200_000
    assert row.selection_cost_at_match == pytest.approx(24.9)
    assert row.lambda_at_selection == pytest.approx(0.001 + 0.035 * 60)  # multiplier after epoch 59
    assert row.measurement_cost == pytest.approx(25.1)
    assert row.gap_hazard == pytest.approx(6.0) and row.gap_dynamics == pytest.approx(2.5)
    assert enrichment.apply_battery(paths.main, run_dir_of, ["hazard", "dynamics"], evaluator=fake_battery) == []
    log = (tmp_path / "ledger.enrichment_log.jsonl").read_text().splitlines()
    assert [json.loads(line)["action"] for line in log] == ["select", "battery", "battery"]
    # the log records what each call actually evaluated, not only what it was asked for
    second = json.loads(log[2])
    assert second["conditions"] == ["hazard", "dynamics"]
    assert second["evaluated"] == {rid: ["dynamics"] for rid in done_first}


def test_selector_that_breaks_rule_one_is_refused(tmp_path) -> None:
    paths = LedgerPaths(main=tmp_path / "ledger.parquet", pilot=tmp_path / "p.parquet", sidecar_dir=tmp_path / "s")
    spec, run_dir = _run(tmp_path, 0)
    write_run(spec, run_dir, paths)
    with pytest.raises(ContractError, match="rule 1 gives 1200000"):
        enrichment.apply_selection(paths.main, selector=lambda cps: 2_000_000)  # in the window, but not closest to 25


def test_overlapping_measurement_seeds_are_refused(tmp_path) -> None:
    paths = LedgerPaths(main=tmp_path / "ledger.parquet", pilot=tmp_path / "p.parquet", sidecar_dir=tmp_path / "s")
    spec, run_dir = _run(tmp_path, 0)
    write_run(spec, run_dir, paths)
    enrichment.apply_selection(paths.main, selector=_closest_to_25)

    def leaky(omnisafe_dir, spec, step, conditions):
        return {"measurement": 25.0, "hazard": 30.0, "episodes": 100, "measurement_seeds": fake_launcher.SELECTION_SEEDS}

    with pytest.raises(ContractError, match="overlap"):
        enrichment.apply_battery(paths.main, lambda rid: tmp_path / "runs" / rid, ["hazard"], evaluator=leaky)


def test_selection_outside_the_window_is_refused(tmp_path) -> None:
    paths = LedgerPaths(main=tmp_path / "ledger.parquet", pilot=tmp_path / "p.parquet", sidecar_dir=tmp_path / "s")
    spec, run_dir = _run(tmp_path, 0)
    write_run(spec, run_dir, paths)
    with pytest.raises(ContractError, match="last ten"):
        enrichment.apply_selection(paths.main, selector=lambda cps: 0)


def test_gaps_need_100_episodes_and_finite_costs() -> None:
    with pytest.raises(ContractError, match="100 episodes"):
        enrichment.gaps_from_costs({"measurement": 1.0, "hazard": 2.0, "episodes": 50}, ["hazard"])
    with pytest.raises(ContractError, match="not finite"):
        enrichment.gaps_from_costs({"measurement": 1.0, "hazard": float("nan"), "episodes": 100}, ["hazard"])
    with pytest.raises(ValueError, match="continuations"):
        enrichment.gaps_from_costs({"measurement": 1.0, "finetune": 2.0, "episodes": 100}, ["finetune"])


def test_rule_one_ties_go_to_the_later_checkpoint_despite_float_noise() -> None:
    steps = list(range(200_000, 2_000_001, 200_000))
    costs = [40.0] * 8 + [32.01, 17.99]  # |32.01 - 25| and |17.99 - 25| differ in binary floating point
    assert abs(32.01 - 25) != abs(17.99 - 25)
    checkpoints = [{"step": s, "selection_cost": c} for s, c in zip(steps, costs)]
    assert enrichment.rule_one_step(checkpoints) == 2_000_000
    np = pytest.importorskip("numpy")  # a harness that averages in float32
    f32 = [{"step": s, "selection_cost": float(np.float32(c))} for s, c in zip(steps, [40.0] * 8 + [17.99, 32.01])]
    assert enrichment.rule_one_step(f32) == 2_000_000
    checkpoints[-1]["selection_cost"] = None
    with pytest.raises(ContractError, match="no selection-set cost"):
        enrichment.rule_one_step(checkpoints)
    with pytest.raises(ValueError, match="unknown condition"):
        enrichment.gaps_from_costs({"episodes": 100, "measurement": 1.0}, ["finetune"])


def test_runs_enriched_before_a_failure_are_logged(tmp_path) -> None:
    paths = LedgerPaths(main=tmp_path / "ledger.parquet", pilot=tmp_path / "p.parquet", sidecar_dir=tmp_path / "s")
    for seed in (0, 1):
        spec, run_dir = _run(tmp_path, seed)
        write_run(spec, run_dir, paths)
    calls = []

    def flaky(checkpoints):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("selector failed on the second run")
        return _closest_to_25(checkpoints)

    with pytest.raises(RuntimeError):
        enrichment.apply_selection(paths.main, selector=flaky)
    (entry,) = [json.loads(line) for line in (tmp_path / "ledger.enrichment_log.jsonl").read_text().splitlines()]
    assert entry["action"] == "select" and entry["run_ids"] == ["T-A-PointGoal1-N0.00-s0"]


def test_enrichment_checks_the_code_before_writing(tmp_path) -> None:
    paths = LedgerPaths(main=tmp_path / "ledger.parquet", pilot=tmp_path / "p.parquet", sidecar_dir=tmp_path / "s")
    spec, run_dir = _run(tmp_path, 0)
    write_run(spec, run_dir, paths)

    def refuse() -> None:
        raise RuntimeError("uncommitted code")

    with pytest.raises(RuntimeError, match="uncommitted"):
        enrichment.apply_selection(paths.main, selector=_closest_to_25, check_code=refuse)
    from results.ledger_schema import load_ledger_as_rows

    assert load_ledger_as_rows(paths.main)[0].matched_checkpoint_step is None  # nothing written


def _selected_ledger(tmp_path: Path) -> LedgerPaths:
    paths = LedgerPaths(main=tmp_path / "ledger.parquet", pilot=tmp_path / "p.parquet", sidecar_dir=tmp_path / "s")
    spec, run_dir = _run(tmp_path, 0)
    write_run(spec, run_dir, paths)
    enrichment.apply_selection(paths.main, selector=_closest_to_25)
    return paths


def _battery(omnisafe_dir, spec, step, conditions):
    return {"measurement": 25.1, "hazard": 31.1, "dynamics": 27.6, "episodes": 100,
            "measurement_seeds": fake_launcher.MEASUREMENT_SEEDS}


def test_battery_refuses_a_namesake_run_under_another_data_root(tmp_path) -> None:
    """Smoke and registered data roots share run_ids: the run directory must be the row's run."""
    import shutil

    paths = _selected_ledger(tmp_path)
    rid = "T-A-PointGoal1-N0.00-s0"
    other = tmp_path / "other" / rid
    shutil.copytree(tmp_path / "runs" / rid, other)  # same commit, but not the checkpoint the row records
    with pytest.raises(ContractError, match="matched checkpoint"):
        enrichment.apply_battery(paths.main, lambda r: tmp_path / "other" / r, ["hazard"], evaluator=_battery)
    train = json.loads((other / "train_result.json").read_text())
    train["commit_hash"] = "f" * 40  # e.g. a smoke run of later code
    (other / "train_result.json").write_text(json.dumps(train))
    with pytest.raises(ContractError, match="trained at commit"):
        enrichment.apply_battery(paths.main, lambda r: tmp_path / "other" / r, ["hazard"], evaluator=_battery)
    from results.ledger_schema import load_ledger_as_rows

    assert load_ledger_as_rows(paths.main)[0].gap_hazard is None
    done = enrichment.apply_battery(paths.main, lambda r: tmp_path / "runs" / r, ["hazard"], evaluator=_battery,
                                    data_root=tmp_path / "runs")
    assert done == [rid]
    entry = json.loads((tmp_path / "ledger.enrichment_log.jsonl").read_text().splitlines()[-1])
    assert entry["data_root"] == str(tmp_path / "runs")


def test_an_enrichment_that_breaks_the_schema_is_not_written(tmp_path) -> None:
    """write_enrichment checks only the Parquet types; the transaction validates every row."""
    from pydantic import ValidationError

    from results.ledger_schema import load_ledger_as_rows

    paths = _selected_ledger(tmp_path)
    before = paths.main.read_bytes()
    with pytest.raises(ValidationError):
        enrichment._write(paths.main, "T-A-PointGoal1-N0.00-s0", {"lambda_at_selection": -1.0})
    assert paths.main.read_bytes() == before
    assert load_ledger_as_rows(paths.main)[0].lambda_at_selection is not None


def test_rule_one_refuses_a_non_finite_cost() -> None:
    steps = list(range(200_000, 2_000_001, 200_000))
    costs = [float("nan")] + [40.0] * 8 + [25.0]  # NaN first once made rule 1 pick step 200,000
    with pytest.raises(ContractError, match="not finite"):
        enrichment.rule_one_step([{"step": s, "selection_cost": c} for s, c in zip(steps, costs)])


def test_a_selector_cannot_alter_the_checkpoints_that_are_checked(tmp_path) -> None:
    paths = LedgerPaths(main=tmp_path / "ledger.parquet", pilot=tmp_path / "p.parquet", sidecar_dir=tmp_path / "s")
    spec, run_dir = _run(tmp_path, 0)
    write_run(spec, run_dir, paths)

    def vandal(checkpoints):
        step = _closest_to_25(checkpoints)
        for c in checkpoints:
            c["selection_cost"] = 25.0  # would make every checkpoint tie and rule 1 pick the last one
        return step

    assert enrichment.apply_selection(paths.main, selector=vandal) == [spec.run_id]
    from results.ledger_schema import load_ledger_as_rows

    row = load_ledger_as_rows(paths.main)[0]
    assert row.matched_checkpoint_step == 1_200_000 and row.selection_cost_at_match == pytest.approx(24.9)


def test_an_unwritable_log_does_not_hide_the_enrichment_error(tmp_path, capsys) -> None:
    paths = LedgerPaths(main=tmp_path / "ledger.parquet", pilot=tmp_path / "p.parquet", sidecar_dir=tmp_path / "s")
    for seed in (0, 1):
        spec, run_dir = _run(tmp_path, seed)
        write_run(spec, run_dir, paths)
    (tmp_path / "ledger.enrichment_log.jsonl").mkdir()  # open(..., "a") raises IsADirectoryError
    calls = []

    def flaky(checkpoints):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("selector failed on the second run")
        return _closest_to_25(checkpoints)

    with pytest.raises(RuntimeError, match="second run"):
        enrichment.apply_selection(paths.main, selector=flaky)
    assert "enrichment log" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Command line: go and g4-measure
# ---------------------------------------------------------------------------


def test_go_rejects_a_non_positive_concurrency_with_its_own_message(capsys) -> None:
    from pilot.__main__ import main

    assert main(["go", "--concurrent", "0"]) == 2
    assert "--concurrent must be at least 1" in capsys.readouterr().err


def _dirty(monkeypatch) -> None:
    from pilot import provenance

    def refuse(*a, **k):
        raise provenance.DirtyWorktreeError("tracked files have uncommitted changes")

    monkeypatch.setattr(provenance, "require_clean_worktree", refuse)
    monkeypatch.setattr(provenance, "dirty_paths", lambda *a, **k: ["pilot/go_decision.py"])


def test_go_and_g4_refuse_uncommitted_code_with_exit_2(tmp_path, monkeypatch, capsys) -> None:
    from pilot import ledger_writer
    from pilot.__main__ import main

    _dirty(monkeypatch)
    assert main(["go", "--out-dir", str(tmp_path)]) == 2
    assert "commit before computing the go decision" in capsys.readouterr().err
    pilot_ledger = tmp_path / "pilot.parquet"
    pilot_ledger.write_bytes(b"")
    monkeypatch.setattr(ledger_writer, "DEFAULT_PATHS", LedgerPaths(main=tmp_path / "m.parquet", pilot=pilot_ledger,
                                                                    sidecar_dir=tmp_path / "s"))
    assert main(["g4-measure", "--data-root", str(tmp_path), "--out", str(tmp_path / "g4.json")]) == 2
    err = capsys.readouterr().err
    assert "commit before taking the G4 measurement" in err and "pilot/go_decision.py" in err
    assert not (tmp_path / "g4.json").exists() and sorted(p.name for p in tmp_path.iterdir()) == ["pilot.parquet"]


def test_go_report_records_the_commit_and_a_clean_worktree(tmp_path, monkeypatch) -> None:
    from datetime import datetime, timezone

    import pilot.__main__ as cli
    from pilot import budget, go_decision, provenance

    monkeypatch.setattr(provenance, "require_clean_worktree", lambda *a, **k: "a" * 40)
    monkeypatch.setattr(provenance, "dirty_paths", lambda *a, **k: [])
    monkeypatch.setattr(provenance, "unverified_imported_code", lambda *a, **k: [])
    record = {"run_id": "P-A-x", "completed": True, "wall_clock_hours": 24.0,
              "started": datetime(2026, 1, 1, tzinfo=timezone.utc), "finished": datetime(2026, 1, 2, tzinfo=timezone.utc)}
    monkeypatch.setattr(cli, "_load_pilot_records", lambda *a, **k: [record])
    monkeypatch.setattr(budget, "allocation_contained", lambda records: (True, "ok"))
    monkeypatch.setattr(go_decision, "report", lambda *a, **k: {})
    written = {}

    def capture(result, out_dir, stamp):
        written.update(result)
        md = out_dir / "r.md"
        md.write_text("report")
        return out_dir / "r.json", md

    monkeypatch.setattr(go_decision, "write_report", capture)
    assert cli.main(["go", "--out-dir", str(tmp_path)]) == 0
    assert written["generated"]["commit"] == "a" * 40 and written["generated"]["worktree_dirty"] is False


def test_write_once_is_atomic_and_never_overwrites(tmp_path, monkeypatch) -> None:
    import os

    import pilot.__main__ as cli

    out = tmp_path / "g4.json"
    cli._write_once(out, "{}\n", "taken once")
    with pytest.raises(cli.CliError, match="taken once"):
        cli._write_once(out, "{\"other\": 1}\n", "taken once")
    assert out.read_text() == "{}\n"

    def crash(fd):
        raise OSError("disk full")

    monkeypatch.setattr(os, "fsync", crash)
    with pytest.raises(OSError, match="disk full"):
        cli._write_once(tmp_path / "g4-rev1.json", "{}\n", "taken once")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["g4.json"]  # no partial file, no temporary file


def _moderate_pilot(tmp_path: Path, monkeypatch, result: dict) -> Path:
    """A ledgered Moderate pilot run under ``<tmp>/data``, a clean tree, and a harness returning ``result``."""
    from pilot import ledger_writer, provenance
    from pilot.scheduler import SchedulerConfig

    spec = RunSpec(run_id="P-B-Moderate-s0", study="B", task="SafetyPointGoal1-v0", arm="Moderate", seed=0,
                   total_steps=2_000_000, base_algo="PPOLag", plugin="study_b", group="study_b",
                   training_levels=(10.0, 20.0, 40.0), pilot=True)
    run_dir = SchedulerConfig(data_root=tmp_path / "data").run_dir(spec.run_id)
    run_dir.mkdir(parents=True)
    (run_dir / "spec.json").write_text(spec.to_json())
    omni = fake_launcher.write_outputs(spec.to_dict(), run_dir)
    (run_dir / "train_result.json").write_text(json.dumps(fake_launcher.train_result(spec.to_dict(), omni, "completed", None)))
    steps = list(range(200_000, 2_000_001, 200_000))
    ev = {"final_cost": 20.0, "final_return": 5.0, "episodes": 100, "eval_wall_clock_hours": 0.1,
          "selection": {str(s): [25.0, 1.0] for s in steps}, "selection_seeds": fake_launcher.SELECTION_SEEDS}
    (run_dir / "evaluation.json").write_text(json.dumps(ev))

    def levels(r: dict) -> dict:  # Study B logs one multiplier per training level
        r = {k: v for k, v in r.items() if not k.startswith("Metrics/LagrangeMultiplier")}
        return {**r, **{f"Metrics/LagrangeMultiplier/level_{b}": "0.1" for b in (10, 20, 40)}}

    import csv

    with open(run_dir / omni / "progress.csv", newline="") as fh:
        rows = [levels(dict(r)) for r in csv.DictReader(fh)]
    with open(run_dir / omni / "progress.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    paths = LedgerPaths(main=tmp_path / "m.parquet", pilot=tmp_path / "pilot.parquet", sidecar_dir=tmp_path / "s")
    write_run(spec, run_dir, paths)
    monkeypatch.setattr(ledger_writer, "DEFAULT_PATHS", paths)
    monkeypatch.setattr(provenance, "require_clean_worktree", lambda *a, **k: fake_launcher.COMMIT)
    monkeypatch.setattr(provenance, "unverified_imported_code", lambda *a, **k: [])
    monkeypatch.setattr(enrichment, "_load", lambda target: lambda omnisafe_dir, spec: dict(result))
    return run_dir


GOOD_G4 = {"episodes": 100, "mean_cost": {10.0: 9.0, 20.0: 19.0, 40.0: 38.0}, "satisfaction": {10.0: 0.9, 20.0: 0.9, 40.0: 0.9}}


def test_g4_measure_refuses_a_namesake_run_under_another_data_root(tmp_path, monkeypatch, capsys) -> None:
    """Same commit, another data root: the final checkpoint path differs from the ledger's, so nothing is written."""
    import shutil

    from pilot.__main__ import main

    run_dir = _moderate_pilot(tmp_path, monkeypatch, GOOD_G4)
    shutil.copytree(run_dir, tmp_path / "other" / "checkpoints" / run_dir.name)
    out = tmp_path / "g4.json"
    assert main(["g4-measure", "--data-root", str(tmp_path / "other"), "--out", str(out)]) == 2
    assert "final checkpoint" in capsys.readouterr().err and not out.exists()
    assert main(["g4-measure", "--data-root", str(tmp_path / "data"), "--out", str(out)]) == 0
    data = json.loads(out.read_text())
    assert data["mean_cost"] == {"10.0": 9.0, "20.0": 19.0, "40.0": 38.0} and data["provenance"]["run_id"] == run_dir.name


@pytest.mark.parametrize("mean_cost", [None, {10.0: 9.0, 40.0: 38.0}, {10.0: float("nan"), 20.0: 19.0, 40.0: 38.0},
                                       {10.0: "n/a", 20.0: 19.0, 40.0: 38.0}])
def test_g4_measure_refuses_a_result_without_finite_costs_at_every_training_budget(tmp_path, monkeypatch, capsys,
                                                                                  mean_cost) -> None:
    """Part 6 G4 decides on the mean cost at 10, 20 and 40; the file is taken once, so a bad result is never written."""
    from pilot.__main__ import main

    result = {**GOOD_G4, "mean_cost": mean_cost}
    _moderate_pilot(tmp_path, monkeypatch, result)
    out = tmp_path / "g4.json"
    assert main(["g4-measure", "--data-root", str(tmp_path / "data"), "--out", str(out)]) == 2
    assert "finite mean_cost for each training budget" in capsys.readouterr().err and not out.exists()


@pytest.mark.parametrize("satisfaction", [None, {10.0: 0.9, 40.0: 0.9}, {10.0: 1.5, 20.0: 0.9, 40.0: 0.9},
                                          {10.0: float("nan"), 20.0: 0.9, 40.0: 0.9},
                                          {10.0: True, 20.0: False, 40.0: True}])
def test_g4_measure_refuses_a_result_without_satisfaction_rates_at_every_training_budget(
    tmp_path, monkeypatch, capsys, satisfaction
) -> None:
    """Part 3.6 measures the Moderate arm's satisfaction rates too; a bad result is never written."""
    from pilot.__main__ import main

    _moderate_pilot(tmp_path, monkeypatch, {**GOOD_G4, "satisfaction": satisfaction})
    out = tmp_path / "g4.json"
    assert main(["g4-measure", "--data-root", str(tmp_path / "data"), "--out", str(out)]) == 2
    assert "satisfaction rate in [0, 1]" in capsys.readouterr().err and not out.exists()


def test_go_refuses_an_explicit_moderate_file_that_does_not_exist(tmp_path, capsys) -> None:
    import pilot.__main__ as cli

    assert cli.main(["go", "--out-dir", str(tmp_path), "--moderate", str(tmp_path / "typo.json")]) == 2
    assert "typo.json does not exist" in capsys.readouterr().err and list(tmp_path.iterdir()) == []

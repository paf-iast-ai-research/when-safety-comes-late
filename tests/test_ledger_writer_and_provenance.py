"""Ledger writer edge cases (Appendix B) and run provenance (Part 3.4)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import fake_launcher  # noqa: E402

from pilot import manifest, provenance  # noqa: E402
from pilot.ledger_writer import LedgerPaths, LedgerWriteError, build_row_fields, write_run  # noqa: E402
from pilot.manifest import RunSpec  # noqa: E402


def _spec(**kw) -> RunSpec:
    base = dict(run_id="T-A-PointGoal1-N0.00-s0", study="A", task="SafetyPointGoal1-v0", arm="N0.00", seed=0,
                total_steps=2_000_000, base_algo="PPOLag", plugin="ppolag", group="main", N=0.0, onset_step=0)
    base.update(kw)
    return RunSpec(**base)


def _make_run(tmp: Path, spec: RunSpec, *, status="completed", cause=None, evaluate=True, commit=fake_launcher.COMMIT) -> Path:
    run_dir = tmp / spec.run_id
    run_dir.mkdir(parents=True)
    omni = fake_launcher.write_outputs(spec.to_dict(), run_dir)
    result = fake_launcher.train_result(spec.to_dict(), omni, status, cause)
    result["commit_hash"] = commit
    (run_dir / "train_result.json").write_text(json.dumps(result))
    if evaluate:
        steps = list(range(200_000, spec.total_steps + 1, 200_000))[-10:]
        ev = {"final_cost": 24.0, "final_return": 20.0, "episodes": 100, "eval_wall_clock_hours": 0.5,
              "selection": {str(s): [25.0, 20.0] for s in steps}, "selection_seeds": fake_launcher.SELECTION_SEEDS}
        (run_dir / "evaluation.json").write_text(json.dumps(ev))
    return run_dir


def _paths(tmp: Path) -> LedgerPaths:
    return LedgerPaths(main=tmp / "main.parquet", pilot=tmp / "pilot.parquet", sidecar_dir=tmp / "sidecar")


def test_row_fields_and_trace(tmp_path) -> None:
    spec = _spec()
    run_dir = _make_run(tmp_path, spec)
    fields = build_row_fields(spec, run_dir)
    assert fields["commit_hash"] == fake_launcher.COMMIT and fields["completed"] is True
    assert fields["wall_clock_hours"] == pytest.approx(1.5)
    assert fields["checkpoints"][1]["training_cost"] == pytest.approx(30 - 9 * 0.05)  # progress row of epoch 9
    assert fields["checkpoints"][1]["multiplier"] == pytest.approx(0.001 + 0.035 * 10)
    assert "mean of the last 50 training episodes" in fields["notes"]
    trace = Path(run_dir / json.loads((run_dir / "train_result.json").read_text())["omnisafe_dir"]) / "multiplier_trace.csv"
    lines = trace.read_text().splitlines()
    assert lines[0] == "epoch,step_end,lagrange_multiplier" and lines[1].startswith("0,20000,")


def test_short_commit_hash_is_rejected(tmp_path) -> None:
    spec = _spec()
    run_dir = _make_run(tmp_path, spec, commit="abc123def456")  # 12 characters, as in the schema's own tests
    with pytest.raises(LedgerWriteError, match="40-character"):
        build_row_fields(spec, run_dir)


def test_completed_run_must_be_evaluated_first(tmp_path) -> None:
    spec = _spec()
    run_dir = _make_run(tmp_path, spec, evaluate=False)
    with pytest.raises(LedgerWriteError, match="evaluate before writing"):
        build_row_fields(spec, run_dir)


def test_selection_must_cover_exactly_the_last_ten_checkpoints(tmp_path) -> None:
    spec = _spec()
    run_dir = _make_run(tmp_path, spec)
    ev = json.loads((run_dir / "evaluation.json").read_text())
    ev["selection"].pop(next(iter(ev["selection"])))
    (run_dir / "evaluation.json").write_text(json.dumps(ev))
    with pytest.raises(LedgerWriteError, match="last ten"):
        build_row_fields(spec, run_dir)


def test_failed_run_is_written_without_evaluation(tmp_path) -> None:
    spec = _spec()
    run_dir = _make_run(tmp_path, spec, status="failed", cause="crash", evaluate=False)
    path = write_run(spec, run_dir, _paths(tmp_path))
    from results.ledger_schema import load_ledger_as_rows

    (row,) = load_ledger_as_rows(path)
    assert row.completed is False and row.failure_cause == "crash" and row.final_cost is None


def test_unconstrained_pilot_run_goes_to_a_sidecar(tmp_path) -> None:
    spec = next(s for s in manifest.pilot() if s.arm == "unconstrained")
    run_dir = _make_run(tmp_path, spec)
    target = write_run(spec, run_dir, _paths(tmp_path))
    assert target.parent == tmp_path / "sidecar"
    data = json.loads(target.read_text())
    assert data["final_cost"] == 24.0 and "not representable" in data["reason"]
    with pytest.raises(LedgerWriteError, match="recorded once"):
        write_run(spec, run_dir, _paths(tmp_path))


def test_every_run_must_use_the_same_selection_seeds(tmp_path) -> None:
    from pilot.contracts import ContractError

    paths = _paths(tmp_path)
    write_run(_spec(), _make_run(tmp_path, _spec()), paths)
    other = _spec(run_id="T-A-PointGoal1-N0.00-s1", seed=1)
    run_dir = _make_run(tmp_path, other)
    ev = json.loads((run_dir / "evaluation.json").read_text())
    ev["selection_seeds"] = list(range(5000, 5100))
    (run_dir / "evaluation.json").write_text(json.dumps(ev))
    with pytest.raises(ContractError, match="differ from those of earlier runs"):
        write_run(other, run_dir, paths)


def test_porcelain_parsing_keeps_every_path(tmp_path) -> None:
    repo = tmp_path / "r"
    repo.mkdir()
    git = lambda *a: subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True)  # noqa: E731
    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    (repo / "pilot").mkdir()
    (repo / "pilot" / "allocation.json").write_text("{}")
    (repo / "results" / "pilot").mkdir(parents=True)
    (repo / "results" / "pilot" / "g4.json").write_text("{}")
    git("add", ".")
    git("commit", "-qm", "init")
    (repo / "pilot" / "allocation.json").write_text('{"x": 1}')  # unstaged: porcelain line starts with a space
    assert provenance.dirty_paths(repo) == ["pilot/allocation.json"]
    git("checkout", "--", "pilot/allocation.json")
    git("mv", "results/pilot/g4.json", "pilot/evaluation.py")  # a rename that moves an output into code
    assert provenance.dirty_paths(repo) == ["pilot/evaluation.py"]


def test_plasticity_file_fills_checkpoint_and_onset_metrics(tmp_path) -> None:
    spec = _spec(run_id="T-A-PointGoal1-N0.50-abrupt-total-s0", arm="N0.50-abrupt-total", N=0.5, onset_step=1_000_000,
                 onset_shape="abrupt", step_matching="total_steps")
    run_dir = _make_run(tmp_path, spec)
    omni = run_dir / json.loads((run_dir / "train_result.json").read_text())["omnisafe_dir"]
    (omni / "plasticity.csv").write_text("step,dormant,rank,norm\n1000000,0.05,40,12.5\n2000000,0.07,38,13.0\n")
    fields = build_row_fields(spec, run_dir)
    assert fields["dormant_onset"] == 0.05 and fields["rank_onset"] == 40 and fields["norm_onset"] == 12.5
    last = fields["checkpoints"][-1]
    assert (last["dormant"], last["rank"], last["norm"]) == (0.07, 38.0, 13.0)


def test_config_hash_ignores_output_location_but_not_settings() -> None:
    a = {"seed": 0, "logger_cfgs": {"log_dir": "/data/x", "save_model_freq": 10}}
    b = {"seed": 0, "logger_cfgs": {"log_dir": "/elsewhere", "save_model_freq": 10}}
    c = {"seed": 1, "logger_cfgs": {"log_dir": "/data/x", "save_model_freq": 10}}
    assert provenance.config_hash(a) == provenance.config_hash(b) != provenance.config_hash(c)
    assert len(provenance.config_hash(a)) == 64
    # OmniSafe's second copy of the custom settings (exp_increment_cfgs) must not leak the location
    a2 = {**a, "exp_increment_cfgs": {"seed": 0, "logger_cfgs": {"log_dir": "/data/x", "save_model_freq": 10}}}
    b2 = {**b, "exp_increment_cfgs": {"seed": 0, "logger_cfgs": {"log_dir": "/elsewhere", "save_model_freq": 10}}}
    assert provenance.config_hash(a2) == provenance.config_hash(b2)


def test_commit_hash_shape_and_output_paths() -> None:
    assert provenance.is_full_commit_hash("0" * 40) and not provenance.is_full_commit_hash("abc123")
    assert provenance.is_output_path("results/ledger.parquet") and provenance.is_output_path("results/pilot/go_report.md")
    assert provenance.is_output_path("ledger/determinism/x.json")
    assert not provenance.is_output_path("results/ledger_schema.py") and not provenance.is_output_path("pilot/scheduler.py")


def test_dirty_worktree_blocks_registered_launch(tmp_path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    git = lambda *a: subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True)  # noqa: E731
    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    (repo / "code.py").write_text("x = 1\n")
    (repo / "results").mkdir()
    (repo / "results" / "ledger.parquet").write_bytes(b"1")
    git("add", ".")
    git("commit", "-qm", "init")
    assert provenance.is_full_commit_hash(provenance.require_clean_worktree(repo))
    (repo / "results" / "ledger.parquet").write_bytes(b"2")  # results may change between runs
    assert provenance.require_clean_worktree(repo)
    (repo / "code.py").write_text("x = 2\n")  # code may not
    with pytest.raises(provenance.DirtyWorktreeError):
        provenance.require_clean_worktree(repo)


def _rewrite_progress(run_dir: Path, transform) -> None:
    import csv

    omni = run_dir / json.loads((run_dir / "train_result.json").read_text())["omnisafe_dir"]
    with open(omni / "progress.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    rows = [transform(dict(r)) for r in rows]
    with open(omni / "progress.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_training_cost_comes_from_the_batch_metrics_when_logged(tmp_path) -> None:
    spec = _spec()
    run_dir = _make_run(tmp_path, spec)
    _rewrite_progress(run_dir, lambda r: {**r, "Metrics/BatchEpCost": "7.5", "Metrics/BatchEpRet": "3.5"})
    fields = build_row_fields(spec, run_dir)
    assert fields["checkpoints"][1]["training_cost"] == 7.5 and fields["checkpoints"][1]["training_return"] == 3.5
    assert "BatchEpCost" in fields["notes"]
    # the untrained step-0 checkpoint has no training batch; its multiplier is the registered initial value
    first = fields["checkpoints"][0]
    assert first["step"] == 0 and first["training_cost"] != first["training_cost"] and first["multiplier"] == 0.001


def test_study_b_per_level_multipliers_are_traced_and_recorded(tmp_path) -> None:
    spec = RunSpec(run_id="P-B-Moderate-s0", study="B", task="SafetyPointGoal1-v0", arm="Moderate", seed=0,
                   total_steps=2_000_000, base_algo="PPOLag", plugin="study_b", group="study_b", training_levels=(10.0, 20.0, 40.0))
    run_dir = _make_run(tmp_path, spec)

    def levels(r: dict) -> dict:
        r = {k: v for k, v in r.items() if not k.startswith("Metrics/LagrangeMultiplier")}
        e = float(r["Train/Epoch"])
        return {**r, "Metrics/LagrangeMultiplier/level_10": str(0.3 + e), "Metrics/LagrangeMultiplier/level_10/Max": "9",
                "Metrics/LagrangeMultiplier/level_20": str(0.2 + e), "Metrics/LagrangeMultiplier/level_40": str(0.1 + e)}

    _rewrite_progress(run_dir, levels)
    fields = build_row_fields(spec, run_dir)
    assert fields["per_level_multipliers"] == {10.0: 99.3, 20.0: 99.2, 40.0: 99.1}  # after the last epoch (99)
    from results.ledger_schema import absolute_checkpoint_path

    trace = Path(absolute_checkpoint_path(fields["multiplier_trace_path"]))
    header = trace.read_text().splitlines()[0]
    assert header == "epoch,step_end,level_10,level_20,level_40"  # OmniSafe's /Max column is not a multiplier


def test_repository_ledgers_refuse_smoke_runs_under_any_path_spelling(tmp_path, monkeypatch) -> None:
    from pilot import ledger_writer

    results = tmp_path / "repo" / "results"
    monkeypatch.setattr(ledger_writer, "REPOSITORY_RESULTS", results)
    spec = _spec()
    run_dir = _make_run(tmp_path, spec)
    result = json.loads((run_dir / "train_result.json").read_text())
    result["allow_pending"] = True
    (run_dir / "train_result.json").write_text(json.dumps(result))
    # the repository's default layout, and a --ledger-dir that points into results/ by another spelling
    for paths in (LedgerPaths(main=results / "ledger.parquet", pilot=results / "pilot" / "ledger.parquet", sidecar_dir=results / "pilot" / "sidecar"),
                  LedgerPaths(main=results / "sub" / ".." / "x.parquet", pilot=tmp_path / "p.parquet", sidecar_dir=tmp_path / "s")):
        with pytest.raises(LedgerWriteError, match="smoke run"):
            write_run(spec, run_dir, paths)
    assert not results.exists()


def test_repository_ledgers_refuse_host_specific_checkpoint_paths(tmp_path, monkeypatch) -> None:
    from pilot import ledger_writer

    results = tmp_path / "repo" / "results"
    monkeypatch.setattr(ledger_writer, "REPOSITORY_RESULTS", results)
    spec = _spec()
    run_dir = _make_run(tmp_path, spec)  # under a temporary directory, not /data/checkpoints
    paths = LedgerPaths(main=results / "ledger.parquet", pilot=results / "pilot.parquet", sidecar_dir=results / "sidecar")
    with pytest.raises(LedgerWriteError, match="not under /data/checkpoints"):
        write_run(spec, run_dir, paths)
    assert not (results / "ledger.parquet").exists()


def test_a_rejected_row_does_not_fix_the_seed_registry(tmp_path) -> None:
    from pydantic import ValidationError

    paths = _paths(tmp_path)
    spec = _spec()
    run_dir = _make_run(tmp_path, spec)
    omni = run_dir / json.loads((run_dir / "train_result.json").read_text())["omnisafe_dir"]
    (omni / "plasticity.csv").write_text("step,dormant,rank,norm\n2000000,1.5,38,13.0\n")  # a fraction above 1
    with pytest.raises(ValidationError):
        write_run(spec, run_dir, paths)
    assert not paths.seeds_registry(spec).exists() and not paths.main.exists()


def test_ledger_transaction_is_all_or_nothing(tmp_path) -> None:
    from pilot.ledger_writer import ledger_transaction

    ledger = tmp_path / "ledger.parquet"
    ledger.write_bytes(b"before")
    with pytest.raises(RuntimeError):
        with ledger_transaction(ledger) as tmp:
            tmp.write_bytes(b"half")
            raise RuntimeError("crash in the middle of a write")
    assert ledger.read_bytes() == b"before" and sorted(p.name for p in tmp_path.iterdir()) == ["ledger.parquet", "ledger.parquet.lock"]
    with ledger_transaction(ledger) as tmp:
        tmp.write_bytes(b"after")
    assert ledger.read_bytes() == b"after"


def test_sidecar_stores_iso_times(tmp_path) -> None:
    spec = next(s for s in manifest.pilot() if s.arm == "unconstrained")
    target = write_run(spec, _make_run(tmp_path, spec), _paths(tmp_path))
    data = json.loads(target.read_text())
    assert data["started"] == "2026-01-01T00:00:00+00:00"
    assert sorted(p.name for p in target.parent.iterdir()) == [target.name]  # no temporary file left


def test_study_b_multiplier_levels_must_match_the_training_levels(tmp_path) -> None:
    spec = RunSpec(run_id="P-B-Moderate-s0", study="B", task="SafetyPointGoal1-v0", arm="Moderate", seed=0,
                   total_steps=2_000_000, base_algo="PPOLag", plugin="study_b", group="study_b", training_levels=(10.0, 20.0, 40.0))
    run_dir = _make_run(tmp_path, spec)

    def two_levels(r: dict) -> dict:
        r = {k: v for k, v in r.items() if not k.startswith("Metrics/LagrangeMultiplier")}
        return {**r, "Metrics/LagrangeMultiplier/level_10": "0.1", "Metrics/LagrangeMultiplier/level_40": "0.2"}

    _rewrite_progress(run_dir, two_levels)
    with pytest.raises(LedgerWriteError, match="trains on"):
        build_row_fields(spec, run_dir)


def _plasticity(run_dir: Path, text: str) -> None:
    omni = run_dir / json.loads((run_dir / "train_result.json").read_text())["omnisafe_dir"]
    (omni / "plasticity.csv").write_text(text)


def test_non_finite_plasticity_is_left_null_so_an_exclusion_can_be_written(tmp_path) -> None:
    """Part 5.6: a run that failed with NaN weights (NaN metrics) must still reach the ledger."""
    from results.ledger_schema import load_ledger_as_rows

    spec = _spec(run_id="T-A-PointGoal1-N0.50-abrupt-total-s0", arm="N0.50-abrupt-total", N=0.5, onset_step=1_000_000,
                 onset_shape="abrupt", step_matching="total_steps")
    run_dir = _make_run(tmp_path, spec, status="failed", cause="non_finite_loss", evaluate=False)
    _plasticity(run_dir, "step,dormant,rank,norm\n200000,0.05,40,12.5\n1000000,nan,nan,inf\n")
    (row,) = load_ledger_as_rows(write_run(spec, run_dir, _paths(tmp_path)))
    by_step = {c.step: c for c in row.checkpoints}
    assert by_step[200_000].dormant == 0.05 and by_step[1_000_000].dormant is None and by_step[1_000_000].norm is None
    assert row.dormant_onset is None and row.norm_onset is None
    assert "non-finite plasticity metrics left null at steps [1000000]" in row.notes


def test_non_finite_selection_cost_is_refused(tmp_path) -> None:
    spec = _spec()
    run_dir = _make_run(tmp_path, spec)
    ev = json.loads((run_dir / "evaluation.json").read_text())
    ev["selection"]["2000000"] = [float("nan"), 20.0]
    (run_dir / "evaluation.json").write_text(json.dumps(ev))
    with pytest.raises(LedgerWriteError, match="not finite"):
        build_row_fields(spec, run_dir)


def test_sidecar_is_strict_json_with_nan_as_null(tmp_path) -> None:
    spec = next(s for s in manifest.pilot() if s.arm == "unconstrained")
    target = write_run(spec, _make_run(tmp_path, spec), _paths(tmp_path))

    def refuse(constant: str) -> None:
        raise ValueError(f"{constant} is not JSON")

    data = json.loads(target.read_text(), parse_constant=refuse)
    first = data["checkpoints"][0]
    assert first["step"] == 0 and first["training_cost"] is None and first["training_return"] is None


def test_a_malformed_plasticity_value_is_refused_not_written_as_null(tmp_path) -> None:
    spec = _spec()
    run_dir = _make_run(tmp_path, spec)
    omni = run_dir / json.loads((run_dir / "train_result.json").read_text())["omnisafe_dir"]
    (omni / "plasticity.csv").write_text("step,dormant,rank,norm\n1000000,oops,1,1\n")
    with pytest.raises(LedgerWriteError, match="not a number"):
        build_row_fields(spec, run_dir)
    (omni / "plasticity.csv").write_text("step,dormant,rank,norm\n,0.1,1,1\n")  # a missing step names the file too
    with pytest.raises(LedgerWriteError, match=r"plasticity\.csv: step '' is not a number"):
        build_row_fields(spec, run_dir)


def test_controller_arms_record_the_initial_multiplier_at_step_0() -> None:
    from pilot.ledger_writer import _initial_multiplier

    controllers = manifest.design("controller")
    assert controllers and all((s.onset_step or 0) > 0 for s in controllers)  # every controller arm has N > 0
    assert {_initial_multiplier(s) for s in controllers if s.base_algo == "PPOLag"} == {0.001}
    assert {_initial_multiplier(s) for s in manifest.design("pid")} == {0.0}

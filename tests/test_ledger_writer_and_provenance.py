"""Ledger writer edge cases (Appendix B), the ``evaluation`` supplement record (HANDOVER.md section 8), the
supplement writer (pilot/supplement.py) and run provenance (Part 3.4)."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import fake_launcher  # noqa: E402
import test_enrichment as T  # noqa: E402

from configs import registered as R  # noqa: E402
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
        ev = fake_launcher.evaluation_result(spec.to_dict(), steps, [25.0] * len(steps), commit=commit)
        ev["eval_wall_clock_hours"] = 0.5
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


def test_a_bad_selection_key_and_a_broken_checkpoint_set_are_the_writers_refusals(tmp_path) -> None:
    """A selection key that is not a checkpoint step, and a checkpoint missing from the final 2,000,000 steps
    (``pilot.contracts.selection_window``), are LedgerWriteErrors, as contract 2's breaches are."""
    spec = _spec()
    run_dir = _make_run(tmp_path, spec)
    ev = json.loads((run_dir / "evaluation.json").read_text())
    first = min(ev["selection"], key=int)
    signed = {("+" + k if k == first else k): v for k, v in ev["selection"].items()}  # int() would take "+200000"
    (run_dir / "evaluation.json").write_text(json.dumps({**ev, "selection": signed}))
    with pytest.raises(LedgerWriteError, match=r"'\+\d+' is not a checkpoint step"):
        build_row_fields(spec, run_dir)
    (run_dir / "evaluation.json").write_text(json.dumps(ev))
    omni = run_dir / json.loads((run_dir / "train_result.json").read_text())["omnisafe_dir"]
    (omni / "torch_save" / "epoch-50.pt").unlink()  # step 1,000,000, inside the final 2,000,000 steps
    with pytest.raises(LedgerWriteError, match=r"grid checkpoints at steps \[1000000\] are missing"):
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
    assert first["step"] == 0 and math.isnan(first["training_cost"])
    assert first["multiplier"] == R.LAGRANGE_MULTIPLIER_INIT


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
    # pilot.contracts.validate_plasticity (contract 2) checks the file; its ContractError is the writer's refusal
    with pytest.raises(LedgerWriteError, match=r"plasticity\.csv: line 2: step '' is not an integer"):
        build_row_fields(spec, run_dir)


def test_controller_arms_record_the_initial_multiplier_at_step_0() -> None:
    from pilot.ledger_writer import _initial_multiplier

    controllers = manifest.design("controller")
    assert controllers and all((s.onset_step or 0) > 0 for s in controllers)  # every controller arm has N > 0
    assert {_initial_multiplier(s) for s in controllers if s.base_algo == "PPOLag"} == {R.LAGRANGE_MULTIPLIER_INIT}
    assert {_initial_multiplier(s) for s in manifest.design("pid")} == {0.0}


# ---------------------------------------------------------------------------
# Contract 2 through pilot.contracts.validate_plasticity (HANDOVER.md section 8)
# ---------------------------------------------------------------------------


def test_a_completed_study_a_run_needs_a_plasticity_row_per_checkpoint(tmp_path) -> None:
    spec = _spec(plugin="study_a")
    run_dir = _make_run(tmp_path, spec)
    _plasticity(run_dir, "step,dormant,rank,norm\n0,0.1,1,1\n")
    with pytest.raises(LedgerWriteError, match="no row for the saved checkpoints"):
        build_row_fields(spec, run_dir)
    (Path(run_dir / json.loads((run_dir / "train_result.json").read_text())["omnisafe_dir"]) / "plasticity.csv").unlink()
    with pytest.raises(LedgerWriteError, match="lacks plasticity.csv"):
        build_row_fields(spec, run_dir)


# ---------------------------------------------------------------------------
# The evaluation supplement record (HANDOVER.md section 8)
# ---------------------------------------------------------------------------


def _full_run(tmp: Path, spec: RunSpec, **kw) -> Path:
    return T.make_run(tmp / "runs", spec, **kw)


def test_the_evaluation_record_is_written_with_the_row(tmp_path) -> None:
    from results.ledger_schema import load_ledger_as_rows

    from pilot import supplement

    spec = T.study_a_spec(0.5, 0)
    paths = _paths(tmp_path)
    ledger = write_run(spec, _full_run(tmp_path, spec), paths)
    (row,) = load_ledger_as_rows(ledger)
    record = supplement.read("evaluation", spec.run_id, ledger_path=ledger)
    assert record.final.mean_cost == row.final_cost and record.final.seed_set == "measurement"
    assert [c.step for c in record.selection] == [c.step for c in row.checkpoints if c.selection_cost is not None]
    assert [c.mean_cost for c in record.selection] == [c.selection_cost for c in row.checkpoints if c.selection_cost is not None]
    assert record.code_commit == fake_launcher.COMMIT and record.final_step == spec.total_steps
    assert supplement.exists("evaluation", spec.run_id, ledger_path=ledger)
    registry = json.loads(paths.seeds_registry(spec).read_text())
    assert registry == {"selection": T.SEL, "measurement": T.MEAS}  # both sets fixed by the first run (Table 2.1)
    assert supplement.supplement_dir(ledger) == tmp_path / "supplement"


def test_an_evaluation_with_a_replaced_episode_passes_the_seed_checks(tmp_path) -> None:
    """Q-mujoco-exception (Table 9.1): an evaluation whose final checkpoint and one selection
    checkpoint replaced unstable episodes by reserve seeds is written; the seed registry gets the planned sets
    (``check_canonical_seeds`` with the canonical lists) and the record carries the replacements."""
    from pilot import supplement

    spec = T.study_a_spec(0.5, 0)
    paths = _paths(tmp_path)
    run_dir = _full_run(tmp_path, spec)
    ev = json.loads((run_dir / "evaluation.json").read_text())
    step = sorted(ev["selection"], key=int)[0]
    replaced = {"slot_index": 10, "seed": T.MEAS[10], "reserve_seed": 2_050_000, "step": 412, "warning": "mjWARN_BADQACC x1"}
    ev["unstable_replacements"] = {"final": [replaced], "selection": {step: [
        {"slot_index": 0, "seed": T.SEL[0], "reserve_seed": 1_050_000, "step": 0, "warning": "mjWARN_BADQVEL x1"},
        {"slot_index": 0, "seed": 1_050_000, "reserve_seed": 1_050_001, "step": 3, "warning": "mjWARN_BADQVEL x1"}]}}
    (run_dir / "evaluation.json").write_text(json.dumps(ev))
    ledger = write_run(spec, run_dir, paths)
    assert json.loads(paths.seeds_registry(spec).read_text()) == {"selection": T.SEL, "measurement": T.MEAS}
    record = supplement.read("evaluation", spec.run_id, ledger_path=ledger)
    assert list(record.final.seeds) == T.MEAS and record.final.unstable_replacements[0].reserve_seed == 2_050_000
    assert [u.reserve_seed for u in record.selection[0].unstable_replacements] == [1_050_000, 1_050_001]
    assert all(block.unstable_replacements is None for block in record.selection[1:])
    other = T.study_a_spec(0.5, 1)
    run_dir = _full_run(tmp_path, other)
    ev = json.loads((run_dir / "evaluation.json").read_text())
    ev["unstable_replacements"] = {"final": [{**replaced, "reserve_seed": T.MEAS[50]}]}  # a seed of the set itself
    (run_dir / "evaluation.json").write_text(json.dumps(ev))
    with pytest.raises(LedgerWriteError, match="not the next unused seed"):
        write_run(other, run_dir, paths)
    assert not supplement.exists("evaluation", other.run_id, ledger_path=ledger)


def test_a_final_cost_on_the_selection_set_is_refused(tmp_path) -> None:
    """Q-final-cost-set (Table 9.1): final_cost and final_return are the measurement set's, for a ledger row and for
    the sidecar's unconstrained cost (G4) alike. An evaluation whose final checkpoint names the selection set is
    refused, with no row, record, sidecar or seed set written."""
    from pilot import supplement

    paths = _paths(tmp_path)
    study_a = T.study_a_spec(0.5, 0)
    unconstrained = next(s for s in manifest.pilot() if s.arm == "unconstrained")
    for spec, run_dir in ((study_a, _full_run(tmp_path, study_a)), (unconstrained, _make_run(tmp_path, unconstrained))):
        ev = json.loads((run_dir / "evaluation.json").read_text())
        (run_dir / "evaluation.json").write_text(json.dumps({**ev, "final_seed_set": "selection"}))
        with pytest.raises(LedgerWriteError, match=r"'selection' is not the measurement set.*Q-final-cost-set"):
            write_run(spec, run_dir, paths)
        assert not paths.seeds_registry(spec).exists() and not paths.for_spec(spec).exists()
        assert not supplement.exists("evaluation", spec.run_id, ledger_path=paths.for_spec(spec))
    assert not (paths.sidecar_dir / f"{unconstrained.run_id}.json").exists()


def test_a_budget_conditioned_run_records_the_budget_of_each_episode(tmp_path) -> None:
    from pilot import supplement

    spec = T.study_b_spec("Dense")
    ledger = write_run(spec, _full_run(tmp_path, spec), _paths(tmp_path))
    record = supplement.read("evaluation", spec.run_id, ledger_path=ledger)
    assert record.final.budgets == T.studyb_budgets(spec) and record.selection[0].budgets == T.studyb_budgets(spec)
    other = T.study_b_spec("Dense", seed=1)
    run_dir = _full_run(tmp_path, other)
    ev = json.loads((run_dir / "evaluation.json").read_text())
    (run_dir / "evaluation.json").write_text(json.dumps({**ev, "studyb_budgets": None}))  # the budgets are required
    with pytest.raises(LedgerWriteError, match="studyb_budgets"):
        write_run(other, run_dir, _paths(tmp_path))


def test_a_refused_row_writes_no_record_and_fixes_no_seed_set(tmp_path) -> None:
    from results.ledger_schema import load_ledger_as_rows

    from pilot import supplement
    from pilot.contracts import ContractError

    paths = _paths(tmp_path)
    first = T.study_a_spec(0.0, 0)
    write_run(first, _full_run(tmp_path, first), paths)
    second = T.study_a_spec(0.0, 1)
    run_dir = _full_run(tmp_path, second)
    ev = json.loads((run_dir / "evaluation.json").read_text())
    ev["measurement_seeds"] = list(range(7000, 7100))  # not the measurement set of the earlier run
    (run_dir / "evaluation.json").write_text(json.dumps(ev))
    with pytest.raises(ContractError, match="differ from those of earlier runs"):
        write_run(second, run_dir, paths)
    assert not supplement.exists("evaluation", second.run_id, ledger_path=paths.main)
    ev["measurement_seeds"] = T.MEAS
    ev["final_cost"] = 99.0  # not the mean of its episodes: the record is refused before anything is written
    (run_dir / "evaluation.json").write_text(json.dumps(ev))
    with pytest.raises(LedgerWriteError, match="not a valid evaluation supplement record"):
        write_run(second, run_dir, paths)
    assert [r.run_id for r in load_ledger_as_rows(paths.main)] == [first.run_id]
    assert json.loads(paths.seeds_registry(second).read_text())["measurement"] == T.MEAS


def test_a_record_left_by_a_crash_is_accepted_after_a_reevaluation(tmp_path, monkeypatch) -> None:
    """A crash after the evaluation record and before the ledger is replaced leaves the record without its row;
    the retry accepts it even after a deterministic re-evaluation by later code (only code_commit differs), keeps
    it unchanged, and still refuses a record with other numbers."""
    from results.ledger_schema import load_ledger_as_rows

    from pilot import ledger_writer, supplement

    paths = _paths(tmp_path)
    spec = T.study_a_spec(0.0, 0)
    run_dir = _full_run(tmp_path, spec)

    def crash(*a, **k):
        raise RuntimeError("killed")
    with monkeypatch.context() as m:
        m.setattr(ledger_writer, "check_canonical_seeds", crash)  # runs after the record, before the replace
        with pytest.raises(RuntimeError, match="killed"):
            write_run(spec, run_dir, paths)
    record = supplement.record_path("evaluation", spec.run_id, ledger_path=paths.main)
    text = record.read_text()
    assert not paths.main.exists() and not paths.seeds_registry(spec).exists()
    ev = json.loads((run_dir / "evaluation.json").read_text())
    (run_dir / "evaluation.json").write_text(json.dumps({**ev, "final_cost": 30.0, "episode_costs": {
        **ev["episode_costs"], "final": T.episodes(30.0)}, "eval_commit_hash": "f" * 40}))
    with pytest.raises(supplement.SupplementConflict):  # other numbers: never accepted
        write_run(spec, run_dir, paths)
    (run_dir / "evaluation.json").write_text(json.dumps({**ev, "eval_commit_hash": "f" * 40}))  # `reevaluate`
    assert write_run(spec, run_dir, paths) == paths.main
    assert [r.run_id for r in load_ledger_as_rows(paths.main)] == [spec.run_id] and record.read_text() == text


def test_a_sidecar_that_cannot_be_written_fixes_no_seed_set(tmp_path, monkeypatch) -> None:
    from pilot import ledger_writer

    spec = next(s for s in manifest.pilot() if s.arm == "unconstrained")
    run_dir = _make_run(tmp_path, spec)
    paths = _paths(tmp_path)

    def full_disk(*a, **k):
        raise OSError("no space left on device")
    with monkeypatch.context() as m:
        m.setattr(ledger_writer, "_write_sidecar", full_disk)
        with pytest.raises(OSError, match="no space"):
            write_run(spec, run_dir, paths)
    assert not paths.seeds_registry(spec).exists() and not (paths.sidecar_dir / f"{spec.run_id}.json").exists()
    target = write_run(spec, run_dir, paths)  # the retry is not refused
    assert target.exists() and paths.seeds_registry(spec).exists()


def test_malformed_progress_and_step_keys_are_refused_cleanly() -> None:
    from pilot import ledger_writer

    # a truncated progress.csv (no row names its epoch) has no last multiplier
    assert ledger_writer.per_level_multipliers([{"Metrics/LagrangeMultiplier/level_10": "1", "Train/Epoch": ""}]) is None
    assert ledger_writer._step_key("200000", "x") == 200_000
    with pytest.raises(LedgerWriteError, match="not a checkpoint step"):
        ledger_writer._step_key("\u00b2", "x")  # a non-ASCII digit: str.isdigit() is True, int() refuses it


def test_per_episode_arrays_are_required_by_every_ledger(tmp_path, monkeypatch) -> None:
    """A completed row needs its evaluation record wherever the ledger lives (Part 5.3): no leniency
    for ledgers outside the repository (evaluation.json is written only by the launcher's evaluation stage,
    pilot.launch.evaluate, which stores the harnesses' per-episode arrays; tests/fake_launcher.py writes the same
    form)."""
    from pilot import ledger_writer, supplement

    paths = _paths(tmp_path)
    # with or without the evaluation stage's marker, a ledger outside the repository refuses the row
    for seed, extra in ((1, {"eval_commit_hash": fake_launcher.COMMIT}), (2, {})):
        staged = T.study_a_spec(0.0, seed)
        staged_dir = _full_run(tmp_path, staged, per_episode=False)
        ev = {k: v for k, v in json.loads((staged_dir / "evaluation.json").read_text()).items() if k != "eval_commit_hash"}
        (staged_dir / "evaluation.json").write_text(json.dumps({**ev, **extra}))
        with pytest.raises(LedgerWriteError, match="per-episode"):
            write_run(staged, staged_dir, paths)
        assert not paths.main.exists() and not supplement.exists("evaluation", staged.run_id, ledger_path=paths.main)
    # the "row already present" path alike: a row whose run lost its arrays gets no record, and says so
    spec = T.study_a_spec(0.0, 0)
    run_dir = _full_run(tmp_path, spec)
    ledger = write_run(spec, run_dir, paths)
    supplement.record_path("evaluation", spec.run_id, ledger_path=ledger).unlink()
    ev = json.loads((run_dir / "evaluation.json").read_text())
    bare = {k: v for k, v in ev.items() if k not in ("episode_costs", "episode_returns")}
    (run_dir / "evaluation.json").write_text(json.dumps(bare))
    with pytest.raises(LedgerWriteError, match="per-episode"):
        ledger_writer.ensure_evaluation_supplement(spec, run_dir, paths)
    # and the repository's ledgers
    other = T.study_a_spec(0.0, 3)
    other_dir = _full_run(tmp_path, other, per_episode=False)
    results = tmp_path / "repo" / "results"
    monkeypatch.setattr(ledger_writer, "REPOSITORY_RESULTS", results)
    monkeypatch.setattr(ledger_writer, "_absolute_paths", lambda fields: [])  # the fixture is not under /data
    repo_paths = LedgerPaths(main=results / "ledger.parquet", pilot=results / "pilot" / "ledger.parquet",
                             sidecar_dir=results / "pilot" / "sidecar")
    with pytest.raises(LedgerWriteError, match="per-episode"):
        write_run(other, other_dir, repo_paths)
    assert not (results / "ledger.parquet").exists()


def test_an_evaluation_run_as_a_smoke_run_is_refused_by_the_repository(tmp_path, monkeypatch) -> None:
    from pilot import ledger_writer

    results = tmp_path / "repo" / "results"
    monkeypatch.setattr(ledger_writer, "REPOSITORY_RESULTS", results)
    spec = _spec()
    run_dir = _make_run(tmp_path, spec)
    ev = json.loads((run_dir / "evaluation.json").read_text())
    (run_dir / "evaluation.json").write_text(json.dumps({**ev, "eval_allow_pending": True}))
    paths = LedgerPaths(main=results / "ledger.parquet", pilot=results / "pilot.parquet", sidecar_dir=results / "s")
    with pytest.raises(LedgerWriteError, match="smoke run"):
        write_run(spec, run_dir, paths)
    assert ledger_writer._is_smoke(run_dir) and not results.exists()


def test_the_smoke_guard_covers_the_supplement_directories(tmp_path, monkeypatch) -> None:
    from pilot import ledger_writer

    results = tmp_path / "repo" / "results"
    monkeypatch.setattr(ledger_writer, "REPOSITORY_RESULTS", results)
    # a ledger outside results/ whose derived supplement directory is inside it (only through a symlink:
    # supplement_dir(ledger) is ledger.parent / "supplement")
    (results / "supplement").mkdir(parents=True)
    (tmp_path / "m").mkdir()
    (tmp_path / "m" / "supplement").symlink_to(results / "supplement", target_is_directory=True)
    inside = LedgerPaths(main=tmp_path / "m" / "l.parquet", pilot=tmp_path / "p" / "l.parquet", sidecar_dir=tmp_path / "s")
    assert ledger_writer.uses_repository_ledgers(inside)
    assert not any(ledger_writer._in_repository_results(p) for p in (inside.main, inside.pilot, inside.sidecar_dir))
    (tmp_path / "m" / "supplement").unlink()
    outside = LedgerPaths(main=tmp_path / "m" / "l.parquet", pilot=tmp_path / "p" / "l.parquet", sidecar_dir=tmp_path / "s")
    assert not ledger_writer.uses_repository_ledgers(outside)
    assert outside.supplement_dirs == (tmp_path / "m" / "supplement", tmp_path / "p" / "supplement")
    smoke = ledger_writer.smoke_paths(tmp_path / "data")
    assert smoke.supplement_dirs == (tmp_path / "data" / "smoke" / "supplement",)
    assert ledger_writer.DEFAULT_PATHS.supplement_dirs == (ledger_writer.REPO_ROOT / "results" / "supplement",
                                                          ledger_writer.REPO_ROOT / "results" / "pilot" / "supplement")


def test_ensure_evaluation_supplement_for_a_row_already_present(tmp_path) -> None:
    from pilot import ledger_writer, supplement

    spec = T.study_a_spec(0.0, 0)
    run_dir = _full_run(tmp_path, spec)
    paths = _paths(tmp_path)
    ledger = write_run(spec, run_dir, paths)
    record = supplement.record_path("evaluation", spec.run_id, ledger_path=ledger)
    text = record.read_text()
    record.unlink()  # e.g. a row written before the supplement existed
    assert ledger_writer.ensure_evaluation_supplement(spec, run_dir, paths) == record
    assert record.read_text() == text
    assert ledger_writer.ensure_evaluation_supplement(spec, run_dir, paths) == record  # identical: accepted
    ev = json.loads((run_dir / "evaluation.json").read_text())
    ev["final_cost"], ev["episode_costs"]["final"] = 30.0, T.episodes(30.0)  # not the row's evaluation
    (run_dir / "evaluation.json").write_text(json.dumps(ev))
    record.unlink()
    with pytest.raises(LedgerWriteError, match="the ledger row records"):
        ledger_writer.ensure_evaluation_supplement(spec, run_dir, paths)
    failed = T.study_a_spec(0.0, 1)
    failed_dir = _full_run(tmp_path, failed, status="failed", cause="crash")
    write_run(failed, failed_dir, paths)
    assert ledger_writer.ensure_evaluation_supplement(failed, failed_dir, paths) is None
    with pytest.raises(LedgerWriteError, match="not in"):
        ledger_writer.ensure_evaluation_supplement(T.study_a_spec(0.0, 2), failed_dir, paths)


def test_ensure_evaluation_supplement_checks_the_selection_and_the_seed_sets(tmp_path) -> None:
    """The record written for a row already present must agree with the row's selection costs and returns
    and with the seed registry (Table 2.1), not only with its final cost and return."""
    from pilot import ledger_writer, supplement

    spec = T.study_a_spec(0.0, 0)
    run_dir = _full_run(tmp_path, spec)
    paths = _paths(tmp_path)
    ledger = write_run(spec, run_dir, paths)
    record = supplement.record_path("evaluation", spec.run_id, ledger_path=ledger)
    record.unlink()
    ev = json.loads((run_dir / "evaluation.json").read_text())
    other = json.loads(json.dumps(ev))
    for step in other["selection"]:  # other selection numbers (consistent with their episodes), the same final
        other["selection"][step][0] = 5.0
        other["episode_costs"]["selection"][step] = T.episodes(5.0)
    (run_dir / "evaluation.json").write_text(json.dumps(other))
    with pytest.raises(LedgerWriteError, match="selection costs and returns"):
        ledger_writer.ensure_evaluation_supplement(spec, run_dir, paths)
    (run_dir / "evaluation.json").write_text(json.dumps({**ev, "selection_seeds": list(range(9000, 9100))}))
    with pytest.raises(LedgerWriteError, match="differ from those of earlier runs"):
        ledger_writer.ensure_evaluation_supplement(spec, run_dir, paths)
    assert not record.exists()
    (run_dir / "evaluation.json").write_text(json.dumps(ev))
    assert ledger_writer.ensure_evaluation_supplement(spec, run_dir, paths) == record


def test_ensure_evaluation_supplement_accepts_a_record_kept_by_the_crash_recovery(tmp_path, monkeypatch) -> None:
    """write_run crashes after its record, the run is re-evaluated by later code, the retried write_run keeps the
    first record, and the scheduler stops before its own bookkeeping: its next pass (the "row already present"
    path) accepts that record, which differs from the re-evaluation only in code_commit, and keeps it."""
    from pilot import ledger_writer, supplement

    paths = _paths(tmp_path)
    spec = T.study_a_spec(0.0, 0)
    run_dir = _full_run(tmp_path, spec)

    def crash(*a, **k):
        raise RuntimeError("killed")
    with monkeypatch.context() as m:
        m.setattr(ledger_writer, "check_canonical_seeds", crash)
        with pytest.raises(RuntimeError, match="killed"):
            write_run(spec, run_dir, paths)
    record = supplement.record_path("evaluation", spec.run_id, ledger_path=paths.main)
    text = record.read_text()
    ev = json.loads((run_dir / "evaluation.json").read_text())
    (run_dir / "evaluation.json").write_text(json.dumps({**ev, "eval_commit_hash": "f" * 40}))  # `reevaluate`
    assert write_run(spec, run_dir, paths) == paths.main
    assert ledger_writer.ensure_evaluation_supplement(spec, run_dir, paths) == record
    assert record.read_text() == text


# ---------------------------------------------------------------------------
# pilot/supplement.py: written once, validated, never overwritten (HANDOVER.md section 8)
# ---------------------------------------------------------------------------


def _measurement(run_id: str = "A-PointGoal1-N0.00-s0", commit: str = fake_launcher.COMMIT, mean: float = 25.1) -> dict:
    costs = T.episodes(mean)
    return {"kind": "measurement", "run_id": run_id, "code_commit": commit, "step": 1_200_000,
            "episodes": {"seed_set": "measurement", "seeds": T.MEAS, "episode_costs": costs,
                         "episode_returns": T.episodes(3.0), "mean_cost": T.fmean(costs), "mean_return": 3.0}}


def test_supplement_write_is_idempotent_and_refuses_a_different_record(tmp_path) -> None:
    from pilot import supplement

    ledger = tmp_path / "results" / "ledger.parquet"
    path = supplement.write("measurement", "A-PointGoal1-N0.00-s0", _measurement(), ledger_path=ledger)
    assert path == tmp_path / "results" / "supplement" / "measurement" / "A-PointGoal1-N0.00-s0.json"
    text = path.read_bytes()
    assert supplement.write("measurement", "A-PointGoal1-N0.00-s0", _measurement(), ledger_path=ledger) == path
    assert path.read_bytes() == text and sorted(p.name for p in path.parent.iterdir()) == [path.name]
    with pytest.raises(supplement.SupplementConflict, match="written once"):
        supplement.write("measurement", "A-PointGoal1-N0.00-s0", _measurement(mean=25.2), ledger_path=ledger)
    later = _measurement(commit="f" * 40)  # the same result reproduced by later code
    with pytest.raises(supplement.SupplementConflict):
        supplement.write("measurement", "A-PointGoal1-N0.00-s0", later, ledger_path=ledger)
    assert supplement.write("measurement", "A-PointGoal1-N0.00-s0", later, ledger_path=ledger,
                            accept_reproduction=True) == path
    assert path.read_bytes() == text  # the first record, with its commit, is kept
    assert supplement.read("measurement", "A-PointGoal1-N0.00-s0", ledger_path=ledger).measurement_cost == 25.1
    assert supplement.read("measurement", "A-PointGoal1-N0.00-s1", ledger_path=ledger) is None
    assert list(supplement.files(ledger)) == [path]


def test_supplement_write_validates_names_parts_and_content(tmp_path) -> None:
    from pilot import supplement

    ledger = tmp_path / "ledger.parquet"
    with pytest.raises(supplement.SupplementError, match="names run"):
        supplement.write("measurement", "A-PointGoal1-N0.00-s1", _measurement(), ledger_path=ledger)
    with pytest.raises(supplement.SupplementError, match="no part"):
        supplement.write("measurement", "A-PointGoal1-N0.00-s0", _measurement(), ledger_path=ledger, part="hazard")
    with pytest.raises(supplement.SupplementError, match="unknown supplement kind"):
        supplement.write("secondary", "A-PointGoal1-N0.00-s0", _measurement(), ledger_path=ledger)
    with pytest.raises(supplement.SupplementError, match="run_id"):
        supplement.record_path("measurement", "../../etc/passwd", ledger_path=ledger)
    with pytest.raises(supplement.SupplementError, match="needs a part"):
        supplement.record_path("battery", "A-PointGoal1-N0.00-s0", ledger_path=ledger)
    bad = _measurement()
    bad["episodes"]["mean_cost"] = 30.0  # not the mean of its episodes
    with pytest.raises(supplement.SupplementError, match="not a valid measurement record"):
        supplement.write("measurement", "A-PointGoal1-N0.00-s0", bad, ledger_path=ledger)
    battery = {**_measurement(), "kind": "battery", "condition": "dynamics", "measurement_cost": 25.0, "gap": 0.1}
    with pytest.raises(supplement.SupplementError, match="part 'dynamics' by its content"):
        supplement.write("battery", "A-PointGoal1-N0.00-s0", battery, ledger_path=ledger, part="hazard")
    assert supplement.write("battery", "A-PointGoal1-N0.00-s0", battery, ledger_path=ledger, part="dynamics").name == \
        "A-PointGoal1-N0.00-s0.dynamics.json"
    assert not (tmp_path / "supplement" / "measurement").exists()  # nothing written for the refused records


def test_supplement_dir_is_the_readers() -> None:
    from analysis.data import default_supplement_dir

    from pilot import supplement

    for ledger in (Path("results/ledger.parquet"), Path("results/pilot/ledger.parquet"), Path("/d/smoke/pilot_ledger.parquet")):
        assert supplement.supplement_dir(ledger) == default_supplement_dir(ledger)


def test_the_reader_loads_what_the_writer_wrote(tmp_path) -> None:
    from analysis.data import load_supplement

    from pilot import supplement

    ledger = tmp_path / "ledger.parquet"
    supplement.write("measurement", "A-PointGoal1-N0.00-s0", _measurement(), ledger_path=ledger)
    loaded = load_supplement(supplement.supplement_dir(ledger))
    assert loaded.get("measurement", "A-PointGoal1-N0.00-s0").measurement_cost == 25.1


def test_a_ledger_mode_marker_is_written_whole_and_an_empty_one_is_named(tmp_path, monkeypatch) -> None:
    """A marker created with open(..., 'x') and then written would be left empty by a crash in between, read as
    None and refuse every later command with 'is a None ledger'. It is written whole (fsync, then link), and an
    empty or unknown marker is an error naming the file."""
    import os

    from pilot import ledger_writer

    paths = LedgerPaths(main=tmp_path / "led" / "ledger.parquet", pilot=tmp_path / "led" / "pilot_ledger.parquet",
                        sidecar_dir=tmp_path / "led" / "sidecar")
    real_link = os.link

    def crash(*a, **k):
        raise OSError("no space left on device")
    monkeypatch.setattr(os, "link", crash)
    with pytest.raises(OSError, match="no space"):
        ledger_writer.mark_ledger_mode(paths, "smoke")
    assert not ledger_writer.mode_marker(paths.main).exists() and not list((tmp_path / "led").glob(".*tmp"))
    monkeypatch.setattr(os, "link", real_link)
    ledger_writer.mark_ledger_mode(paths, "smoke")  # the retry is not refused
    assert {ledger_writer.ledger_mode(p) for p in (paths.main, paths.pilot, paths.sidecar_dir)} == {"smoke"}
    ledger_writer.mode_marker(paths.main).write_text("")  # a marker left empty by older code
    with pytest.raises(LedgerWriteError, match=r"ledger\.parquet\.mode holds ''"):
        ledger_writer.ledger_mode(paths.main)
    with pytest.raises(LedgerWriteError, match="mode is unknown"):
        ledger_writer.mark_ledger_mode(paths, "smoke")


def test_a_smoke_root_never_reaches_a_registered_roots_ledgers_outside_the_repository(tmp_path) -> None:
    """A smoke data root can store the same --ledger-dir as a registered data root whose ledgers lie outside the
    repository; without a mark its rows would reach the registered ledger, and `enrich --allow-dirty
    --allow-pending --data-root <smoke root>` would accept that ledger. The ledgers carry their mode (``<ledger>.mode``)."""
    from pilot import ledger_writer
    from pilot.__main__ import CliError, _check_smoke_ledger
    from pilot.scheduler import Scheduler, SchedulerConfig, SchedulerError

    shared = tmp_path / "ledgers"
    paths = LedgerPaths(main=shared / "ledger.parquet", pilot=shared / "pilot_ledger.parquet", sidecar_dir=shared / "sidecar")
    registered = Scheduler(SchedulerConfig(data_root=tmp_path / "registered", ledger_paths=paths))
    registered.fix_mode()
    assert ledger_writer.ledger_mode(paths.main) == "registered" and ledger_writer.uses_registered_ledgers(paths)
    assert not ledger_writer.uses_repository_ledgers(paths)  # the repository guard alone would not see them
    smoke = Scheduler(SchedulerConfig(data_root=tmp_path / "smoke", ledger_paths=paths, allow_dirty=True, allow_pending=True))
    with pytest.raises(SchedulerError, match="registered ledger"):
        smoke.fix_mode()
    assert smoke.setting("mode") is None  # nothing fixed by the refused command
    # a smoke run is refused by those ledgers whoever writes it
    spec = _spec()
    run_dir = _make_run(tmp_path, spec)
    result = json.loads((run_dir / "train_result.json").read_text())
    (run_dir / "train_result.json").write_text(json.dumps({**result, "allow_dirty": True}))
    with pytest.raises(LedgerWriteError, match="smoke run"):
        write_run(spec, run_dir, paths)
    assert not paths.main.exists()
    # a smoke root fixed before the marker existed (its stored ledgers are the shared ones) is refused by the marker
    legacy = Scheduler(SchedulerConfig(data_root=tmp_path / "legacy", ledger_paths=paths, allow_dirty=True))
    for p in (paths.main, paths.pilot, paths.sidecar_dir):
        ledger_writer.mode_marker(p).unlink()
    legacy.fix_mode()
    for p in (paths.main, paths.pilot, paths.sidecar_dir):
        ledger_writer.mode_marker(p).write_text("registered\n")
    paths.main.write_bytes(b"")
    with pytest.raises(CliError, match="registered ledger"):
        _check_smoke_ledger(paths.main, tmp_path / "legacy")
    # and the reverse: a registered root never writes to ledgers a smoke root marked
    other = LedgerPaths(main=tmp_path / "s" / "ledger.parquet", pilot=tmp_path / "s" / "p.parquet", sidecar_dir=tmp_path / "s" / "sc")
    Scheduler(SchedulerConfig(data_root=tmp_path / "smoke2", ledger_paths=other, allow_dirty=True)).fix_mode()
    with pytest.raises(SchedulerError, match="smoke ledger"):
        Scheduler(SchedulerConfig(data_root=tmp_path / "registered2", ledger_paths=other)).fix_mode()

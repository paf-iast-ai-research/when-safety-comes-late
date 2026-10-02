"""End to end: ledger rows, every ``python -m pilot enrich`` command, then ``python -m analysis --mode final``.

A synthetic main-study data root (``tests/test_enrichment.py``'s fixtures: fake run directories of
ten Study A runs of the primary comparison, their battery continuations, and two Study B runs
(Moderate with its few-shot continuations, Continuous without)) is ledgered by the real ledger writer
and enriched through the real command line, with the other roles' harnesses replaced by the stand-ins
of ``Harness`` (their result shapes) and Role 3's and Role 4's functions running for real. After
every command the ledger must still load (``results.ledger_schema.load_ledger_as_rows``) and read as
an analysis dataset (``analysis.data``), and at the end the ledger and its supplement must agree (no
data problem) and the registered analysis must run on them (HANDOVER.md sections 7 and 8).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import test_enrichment as T  # noqa: E402

from configs import registered as R  # noqa: E402
from pilot import manifest, supplement  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

REF_COSTS = [24.0, 25.0, 26.0, 25.0, 25.0]
LATE_COSTS = [26.0, 27.0, 26.5, 27.5, 26.0]  # mean 26.6: matched at 2.5


@pytest.fixture
def stubbed(monkeypatch):
    """The harnesses of Roles 2 and 5 replaced where ``pilot.enrichment._load`` finds them."""
    from envs import evaluation as E
    from studyb import evaluation as SE

    harness = T.Harness()
    monkeypatch.setattr(E, "evaluate_battery", harness.battery)
    monkeypatch.setattr(E, "evaluate_continuation", harness.continuation)
    monkeypatch.setattr(SE, "evaluate_zero_shot", harness.zero_shot)
    monkeypatch.setattr(SE, "evaluate_fewshot", harness.fewshot)
    return harness


def _data_root(tmp_path: Path, harness: T.Harness) -> tuple[Path, Path, dict]:
    """The main study's ledger and data root: Study A rows (primary comparison, PointGoal1), Study B rows.

    The ledgers are the data root's smoke ledgers (``ledger_writer.smoke_paths``): the commands run with
    --allow-dirty, which the command line takes for smoke ledgers only (HANDOVER.md section 7). The data root has no
    scheduler state, so the continuations are judged by their own train_result.json."""
    from pilot.ledger_writer import smoke_paths

    data = tmp_path / "data"
    paths = smoke_paths(data)
    root = data / "checkpoints"
    specs = {}
    for N, costs in ((0.0, REF_COSTS), (0.5, LATE_COSTS)):
        for seed, cost in enumerate(costs):
            spec = T.study_a_spec(N, seed)
            harness.c_id[spec.run_id] = cost
            specs[spec.run_id] = spec
            T.write_run(spec, T.make_run(root, spec, batch_cost=lambda e: 40.0 if e < 70 else 20.0), paths)
    for arm in ("Moderate", "Continuous"):
        spec = T.study_b_spec(arm)
        specs[spec.run_id] = spec
        T.write_run(spec, T.make_run(root, spec), paths)
    for cont in manifest.study_b_fewshot([specs["B-Moderate-s0"]]):
        T.make_run(root, cont)
    return paths.main, data, specs


def _cli(*argv: str) -> int:
    from pilot.__main__ import main

    return main(list(argv))


def _check_loadable(ledger: Path) -> None:
    from analysis.data import load_dataset
    from results.ledger_schema import load_ledger_as_rows

    assert load_ledger_as_rows(ledger)
    load_dataset(ledger, mode="final", cuts=())


def test_every_enrichment_command_end_to_end_then_the_analysis(tmp_path, stubbed, monkeypatch, capsys) -> None:
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset(R.PENDING))  # every key answered, as decided
    ledger, data, specs = _data_root(tmp_path, stubbed)
    common = ["--ledger", str(ledger), "--data-root", str(data), "--allow-dirty"]
    _check_loadable(ledger)
    steps = [
        ["enrich", "select"],
        ["enrich", "measure"],
        ["enrich", "match", "--surplus-extra", "0"],
        ["enrich", "battery", "--conditions", "hazard", "dynamics"],
        ["enrich", "controller"],
        ["enrich", "training"],
        ["enrich", "final-battery", "--conditions", "hazard", "dynamics"],
        ["enrich", "sensitivity-battery", "--surplus-extra", "0"],
        ["enrich", "zeroshot"],
        ["enrich", "fewshot"],
    ]
    for argv in steps:
        assert _cli(*argv, *common) == 0, capsys.readouterr().err
        _check_loadable(ledger)
    rows = T.rows_of(ledger)
    late = rows["A-PointGoal1-N0.50-abrupt-total-s0"]
    assert late.matched is True and late.infeasible is False and late.measurement_cost == 26.0
    assert late.gap_hazard == pytest.approx(6.0) and late.lambda_final is not None
    # the battery continuations of the matched rows, then their gaps
    for spec in [s for s in specs.values() if s.study == "A"]:
        for condition in manifest.BATTERY_CONDITIONS:
            T.make_run(data / "checkpoints", manifest.battery_continuation(spec, rows[spec.run_id].model_dump(), condition))
    assert _cli("enrich", "continuations", *common) == 0
    _check_loadable(ledger)
    rows = T.rows_of(ledger)
    assert all(r.gap_finetune == pytest.approx(30.0 - r.measurement_cost) for r in rows.values() if r.study == "A")
    moderate = rows["B-Moderate-s0"]
    assert len(moderate.sr_fewshot) == len(R.UNSEEN_BUDGETS) * len(R.FEWSHOT_HORIZONS)
    assert moderate.adapt_steps == {b: 500_000 for b in R.UNSEEN_BUDGETS}
    assert rows["B-Continuous-s0"].sr_zero is not None and rows["B-Continuous-s0"].sr_fewshot is None
    # a second pass of every command changes nothing: every result is written once
    before = ledger.read_bytes()
    for argv in steps:
        assert _cli(*argv, *common) == 0
    assert _cli("enrich", "continuations", *common) == 0
    assert before == ledger.read_bytes()
    # the ledger and its supplement agree (analysis.data cross-checks every field against its record)
    from analysis.data import load_dataset

    dataset = load_dataset(ledger, mode="final", cuts=())
    assert dataset.problems == []
    rec = next(r for r in dataset.study_a if r["run_id"] == "A-PointGoal1-N0.50-abrupt-total-s0")
    assert rec["measurement_return"] == pytest.approx(3.0) and rec["final_gap_hazard"] == pytest.approx(6.0)
    assert rec["recovery_steps"] == 71 * R.STEPS_PER_EPOCH - specs[rec["run_id"]].onset_step and rec["battery_episodes"]["finetune"]
    assert rec["sens_gap_hazard"] is None  # matched at 2.5: rule 7 (a) reads its battery gap
    # completeness: only the Continuous arm's few-shot continuations are missing
    capsys.readouterr()
    assert _cli("completeness", "--ledger", str(ledger), "--data-root", str(data), "--json") == 0
    report = json.loads(capsys.readouterr().out)
    missing = {arm: e for arm, e in report["arms"].items() if e["fields_missing"] or e["records_missing"]}
    assert list(missing) == ["B-Continuous"]
    assert sorted(missing["B-Continuous"]["fields_missing"]) == ["adapt_steps", "sr_fewshot"]
    # the registered analysis runs on the enriched ledger and its supplement
    from analysis.__main__ import main as analysis_main

    out = tmp_path / "analysis-out"
    assert analysis_main(["--mode", "final", "--surplus-extra", "0", "--allow-dirty", "--ledger", str(ledger), "--out", str(out)]) == 0
    result = json.loads((out / "analysis.json").read_text())
    assert result["provenance"]["supplement_files"] == len(list(supplement.files(ledger)))


def test_the_pilot_flow_end_to_end(tmp_path, stubbed, monkeypatch, capsys) -> None:
    """The pilot: select, then the battery of every pilot Study A row (Part 3.6), with the smoke flags."""
    paths = T.paths_in(tmp_path)
    data = T.smoke_root(tmp_path, paths)  # --allow-dirty and --allow-pending: a smoke data root's ledgers only
    for N in (0.0, 0.5):
        for seed in R.PILOT_SEEDS:
            spec = T.study_a_spec(N, seed, pilot=True)
            T.write_run(spec, T.make_run(data / "checkpoints", spec), paths)
    common = ["--ledger", str(paths.pilot), "--data-root", str(data), "--allow-dirty"]
    assert _cli("enrich", "select", *common) == 0
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset())  # the keys open, whatever the repository answers
    assert _cli("enrich", "battery", *common) == 3  # Q-hazard and Q-dynamics are open: a clean refusal
    assert "refused: PendingQuestionError" in capsys.readouterr().err
    assert _cli("enrich", "battery", *common, "--allow-pending") == 0  # smoke ledgers only
    pilot_rows = T.rows_of(paths.pilot)
    assert all(r.gap_hazard is not None and r.gap_dynamics is not None for r in pilot_rows.values())
    assert _cli("enrich", "match", *common, "--surplus-extra", "0", "--allow-pending") == 1  # the pilot's matching is G1's
    assert "pilot's runs are not reused" in capsys.readouterr().err
    from analysis.data import load_dataset

    dataset = load_dataset(paths.pilot, mode="pilot", cuts=())
    assert dataset.problems == [] and len(dataset.study_a) == 6


def test_schedule_add_continuations_and_cuts_on_the_command_line(tmp_path, monkeypatch, capsys) -> None:
    from pilot.ledger_writer import LedgerPaths
    from pilot.scheduler import Scheduler, SchedulerConfig

    data = tmp_path / "data"
    ledger_dir = tmp_path / "L"
    paths = LedgerPaths(main=ledger_dir / "ledger.parquet", pilot=ledger_dir / "pilot_ledger.parquet",
                        sidecar_dir=ledger_dir / "sidecar")
    cfg = SchedulerConfig(data_root=data, ledger_paths=paths)
    sched = Scheduler(cfg)
    parent = T.study_a_spec(0.0, 0)
    sched.add([parent])
    T.write_run(parent, T.make_run(cfg.runs_root, parent), paths)
    from pilot import enrichment

    enrichment.apply_selection(paths.main, selector=T.closest_to_25)
    enrichment._write(paths.main, parent.run_id, {"measurement_cost": 25.0, "matched": True, "infeasible": False})
    sched.db.close()
    from pilot import provenance

    # the queueing commands run from a clean, committed tree (pilot.__main__ refuses a dirty one)
    monkeypatch.setattr(provenance, "dirty_paths", lambda *a, **kw: [])
    monkeypatch.setattr(provenance, "unverified_imported_code", lambda *a, **kw: [])
    argv = ["schedule", "add-continuations", "--ledger", str(paths.main), "--data-root", str(data)]
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset())  # the key open, whatever the repository answers
    assert _cli(*argv) == 3  # Q-arm-complete (the scheduler's gate) is open: a clean refusal
    assert "Q-arm-complete" in capsys.readouterr().err
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset({"Q-arm-complete"}))
    assert _cli(*argv, "--conditions", "finetune") == 0
    out = capsys.readouterr().out
    assert "added 1 battery continuations" in out and f"{parent.arm_id}-finetune-s0" in out
    assert Scheduler(SchedulerConfig(data_root=data)).exists(f"{parent.arm_id}-finetune-s0")
    # schedule add --cuts K reads the committed pilot/cuts.json: without it the command refuses cleanly
    other = tmp_path / "other"
    assert _cli("schedule", "add", "--design", "pid", "--cuts", "1", "--data-root", str(other)) in (1, 2)
    err = capsys.readouterr().err
    assert "cuts.json" in err and "Traceback" not in err
    assert Scheduler(SchedulerConfig(data_root=other)).rows() == []
    assert _cli("schedule", "add", "--design", "pid", "--cuts", "0", "--data-root", str(other)) == 0
    assert "added 15 of 15 runs" in capsys.readouterr().out

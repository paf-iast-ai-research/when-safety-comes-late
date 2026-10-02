"""``python -m analysis`` end to end (Part 5.8; analysis/__main__.py), pilot mode against the go report, and the
amendment and errata registries (Part 8.2; Part 5.8)."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import test_analysis_synthetic as syn  # noqa: E402

from analysis import __main__ as cli  # noqa: E402
from analysis import pilot_check, questions, records  # noqa: E402
from analysis import verdict as V  # noqa: E402
from configs import registered as R  # noqa: E402
from pilot import go_decision, manifest, provenance  # noqa: E402
from results.ledger_schema import load_ledger_as_rows  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SHA = "b" * 40


def _gfm_cells(line: str) -> int:
    """The number of cells of a GFM table row (a pipe escaped as ``\\|`` is cell content)."""
    return len(re.split(r"(?<!\\)\|", line.strip().strip("|")))


def _assert_gfm_tables(markdown: str) -> None:
    lines = markdown.splitlines()
    delimiters = [i for i, line in enumerate(lines) if re.fullmatch(r"\|(-+\|)+", line.strip())]
    assert delimiters
    for i in delimiters:
        width = _gfm_cells(lines[i])
        assert _gfm_cells(lines[i - 1]) == width, lines[i - 1]
        for row in lines[i + 1:]:
            if not row.startswith("|"):
                break
            assert _gfm_cells(row) == width, row


def _final_world(root: Path) -> Path:
    """A registered-style ledger (PointGoal1, every Study A group; Study B) with its supplement records."""
    rows = syn.a_world("supported")
    specs = {s.run_id: s for s in syn.specs(groups=("main", "treatment", "controller", "pid"))}
    sup = root / "supplement"
    for row in rows:
        syn.write_supplement(sup, "measurement", syn.measurement_record(row))
        for condition in ("hazard", "dynamics"):
            record = syn.battery_record(row, condition)
            row[f"gap_{condition}"] = record["gap"]  # the ledger holds the gap its record computed
            syn.write_supplement(sup, "battery", record)
        spec = specs[row["run_id"]]
        treated = spec.treatment in ("reset", "injection")
        syn.write_supplement(sup, "training", syn.training_record(
            row, spec=spec, dormant_check=(0.3 if treated else 0.5) + 0.01 * syn.noise(spec.seed),
            rank_check=(30.0 if treated else 20.0) + syn.noise(spec.seed)))
    b_rows, b_records = syn.b_world()
    for recs in b_records.values():
        syn.write_supplement(sup, "zero_shot", recs["zero_shot"])
        for f in recs["fewshot"]:
            syn.write_supplement(sup, "fewshot", f)
    return syn.write_ledger(root / "ledger.parquet", rows + b_rows)


def _pilot_world(root: Path, late_gaps=(8.0, 7.0, 9.5), ref_costs=(24.0, 25.0, 26.0)) -> Path:
    rows = []
    for spec in manifest.pilot():
        if spec.study == "A":
            late = spec.N == 0.5
            cost = (25.0, 25.5, 26.53)[spec.seed] if late else ref_costs[spec.seed]
            gap = late_gaps[spec.seed] if late else (1.0, 2.0, 3.0)[spec.seed]
            rows.append(syn.study_a_row(spec, cost=cost, gaps={"hazard": gap, "dynamics": 1.0 + spec.seed},
                                        final_cost=cost + 0.5))
        elif spec.arm == "Moderate":
            rows.append(syn.study_b_row(spec, sr_zero={b: 0.5 for b in R.UNSEEN_BUDGETS}))
    return syn.write_ledger(root / "ledger.parquet", rows)


def _as_on_disk(commit: str, rel: str, *a, **k):
    """``provenance.file_committed_at`` of a checkout whose commit holds every file as it is on disk."""
    path = Path(rel) if Path(rel).is_absolute() else REPO / rel
    return path.read_bytes() if path.is_file() else None


@pytest.fixture
def keys_open(monkeypatch):
    """Every PENDING key open (configs.registered.ANSWERED_QUESTIONS empty): the test reads what the go report and
    the analysis show only while a key is open, whatever the amendment log has answered since."""
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset())


@pytest.fixture
def clean_tree(monkeypatch):
    """A committed checkout: the CLI's provenance checks pass (the working tree of this build is not committed)."""
    monkeypatch.setattr(provenance, "commit_hash", lambda *a, **k: SHA)
    monkeypatch.setattr(provenance, "dirty_paths", lambda *a, **k: [])
    monkeypatch.setattr(provenance, "unverified_imported_code", lambda *a, **k: [])
    monkeypatch.setattr(provenance, "file_committed_at", _as_on_disk)


def test_end_to_end_final_analysis(tmp_path, clean_tree, capsys, monkeypatch) -> None:
    """The whole command on a synthetic ledger; 2,000 resamples keep it fast (the slow test below runs the
    registered 10,000: the interquartile means of every contrast are the costly part)."""
    monkeypatch.setattr(R, "BOOTSTRAP_RESAMPLES", 2_000)
    _check_final_analysis(tmp_path, capsys)


@pytest.mark.slow
def test_end_to_end_final_analysis_with_the_registered_resamples(tmp_path, clean_tree, capsys) -> None:
    assert R.BOOTSTRAP_RESAMPLES == 10_000
    _check_final_analysis(tmp_path, capsys)


def _check_final_analysis(tmp_path: Path, capsys) -> None:
    ledger = _final_world(tmp_path)
    out = tmp_path / "out"
    assert cli.main(["--mode", "final", "--surplus-extra", "0", "--ledger", str(ledger), "--out", str(out)]) == cli.EXIT_OK
    printed = capsys.readouterr().out
    assert "H1: SUPPORTED (confirmatory)" in printed and "G1: SUPPORTED (confirmatory)" in printed
    result = json.loads((out / "analysis.json").read_text())
    by_id = {v["id"]: v for v in result["verdicts"]}
    for key in ("H1", "H2", "H3a", "H3b", "H3c", "H3", "H4", "H0", "G1", "G2", "G3", "G4", "G5",
                "A-final-estimand", "A-rule7a"):
        assert key in by_id
    assert by_id["H1"]["status"] == V.SUPPORTED and by_id["H3"]["status"] == V.SUPPORTED
    assert by_id["G4"]["status"] == V.SUPPORTED and by_id["H0"]["status"] == V.FALSIFIED
    prov = result["provenance"]
    assert prov["commit"] == SHA and prov["worktree_dirty"] is False and prov["mode"] == "final"
    assert prov["ledger_sha256"] == hashlib.sha256(ledger.read_bytes()).hexdigest()
    assert prov["analysis_code_hash"] == records.code_hash() and prov["supplement_files"] > 700
    # the supplement hash is of the bytes read: each file's SHA-256, by relative path
    sup = ledger.parent / "supplement"
    digest = hashlib.sha256()
    for path in sorted(sup.rglob("*.json"), key=lambda q: q.relative_to(sup).as_posix()):
        digest.update(path.relative_to(sup).as_posix().encode() + b"\0"
                      + hashlib.sha256(path.read_bytes()).hexdigest().encode() + b"\0")
    assert prov["supplement_sha256"] == digest.hexdigest() and prov["supplement_dir_exists"] is True
    assert prov["inputs"]["analysis/AMENDMENTS.json"] == {
        "sha256": hashlib.sha256(records.AMENDMENTS_FILE.read_bytes()).hexdigest(), "committed": True}
    assert prov["answered_questions"] == sorted(R.ANSWERED_QUESTIONS) and prov["bootstrap"]["seed"] == 0
    assert prov["seeds_per_arm"]["A-PointGoal1-N0.00"] == list(R.SEEDS)
    assert result["open_keys"] == questions.open_keys() and result["problems"] == []
    report = (out / "report.md").read_text()
    for v in result["verdicts"]:  # every verdict with its status, label and open keys
        assert v["id"] in report and v["display_status"] in report
    assert "provisional_on: " + ", ".join(by_id["H1"]["provisional_on"]) in report
    assert "## Open questions" in report and all(f"- {k}:" in report for k in result["open_keys"])
    if not result["open_keys"]:  # the committed state: the section says so, never an empty list after a colon
        assert "## Open questions\n\nNone: every PENDING key is answered" in report
    assert "no effect" not in report.lower() and "no effect" not in (out / "analysis.json").read_text().lower()
    assert "consistent with mediation by plasticity" in report
    with open(out / "tables" / "verdicts.csv", newline="") as fh:
        table = list(csv.DictReader(fh))
    assert {r["id"] for r in table} == set(by_id)
    with open(out / "tables" / "A_estimates.csv", newline="") as fh:
        estimates = list(csv.DictReader(fh))
    assert estimates and {r["boot_resamples"] for r in estimates} == {str(R.BOOTSTRAP_RESAMPLES)}
    for name in ("A_matching", "A_selection", "A_training_age", "A_iqm", "B_outcomes", "B_floor", "B_fewshot",
                 "B_iqm", "power", "M11_exclusions", "seeds_per_arm", "A_gaps", "A_estimands_side_by_side", "A_norms"):
        assert (out / "tables" / f"{name}.csv").exists()
    # every table as CSV and as JSON (the same rows), and nothing else in tables/
    assert sorted(p.name for p in (out / "tables").iterdir()) == sorted(
        f"{name}.{ext}" for name in result["tables"] for ext in ("csv", "json"))
    for name in result["tables"]:
        with open(out / "tables" / f"{name}.csv", newline="") as fh:
            csv_rows = list(csv.DictReader(fh))
        json_rows = json.loads((out / "tables" / f"{name}.json").read_text())
        assert isinstance(json_rows, list) and len(json_rows) == len(csv_rows), name
    gaps = json.loads((out / "tables" / "A_gaps.json").read_text())  # equation (1) per seed (Part 5.1)
    study_a_runs = {r.run_id for r in load_ledger_as_rows(ledger) if r.study == "A"}
    assert {r["run_id"] for r in gaps} == study_a_runs and len(gaps) == len(study_a_runs)
    assert all(r["compared"] is True and r["gap_hazard"] is not None for r in gaps)
    # Part 5.7 on a falsified box only: the supported H1 and G1 print no bound and no Part 5.7 reading
    sections = {part.split(":", 1)[0].strip(): part for part in report.split("\n### ")[1:]}
    for key in ("H1", "G1"):
        assert by_id[key]["status"] == V.SUPPORTED
        assert "bound_reading" not in sections[key] and "Part 5.7" not in sections[key], key
    # the tables that rest on open questions carry them (analysis.questions.TABLE_ENTRIES)
    for name, entry in questions.TABLE_ENTRIES.items():
        with open(out / "tables" / f"{name}.csv", newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert all(r["provisional_on"] == ", ".join(questions.provisional_on(entry)) for r in rows), name
    assert "Q-iqm" in questions.keys_for("A-iqm")
    assert ("Q-iqm" in questions.provisional_on("A-iqm")) == R.is_open("Q-iqm")
    # Part 5.3: the interquartile mean of every tabled contrast with per-episode records (Q-iqm)
    for name in ("A_iqm", "B_iqm"):
        with open(out / "tables" / f"{name}.csv", newline="") as fh:
            iqm = list(csv.DictReader(fh))
        assert iqm and all(r["iqm_diff"] for r in iqm) and {r["iqm_resamples"] for r in iqm} == {str(R.BOOTSTRAP_RESAMPLES)}
    # Part 5.5: the detectable size beside every estimate, and how each registered power number is rounded
    assert "d80 at n (\\|d\\| below it)" in report and "\\|d\\| below d80" in report
    # GFM recognises a table only when its header has as many cells as its delimiter row (and a body row
    # with more is cut): every table of the report is one
    _assert_gfm_tables(report)
    with open(out / "tables" / "power.csv", newline="") as fh:
        power = {(r["what"], r["n"]): r for r in csv.DictReader(fh)}
    assert power[("registered 80-percent point", "10")]["registered_is"] == "rounded up"
    assert power[("registered 80-percent point", "5")]["registered_is"] == "rounded to nearest"
    # an analysis is never overwritten
    assert cli.main(["--mode", "final", "--surplus-extra", "0", "--ledger", str(ledger), "--out", str(out)]) == cli.EXIT_REFUSED
    assert "never overwritten" in capsys.readouterr().err


def test_the_cli_refuses_uncommitted_code(tmp_path, monkeypatch, capsys) -> None:
    ledger = _pilot_world(tmp_path)
    monkeypatch.setattr(provenance, "commit_hash", lambda *a, **k: SHA)
    monkeypatch.setattr(provenance, "file_committed_at", _as_on_disk)
    monkeypatch.setattr(provenance, "dirty_paths", lambda *a, **k: ["analysis/stats.py"])
    assert cli.main(["--mode", "pilot", "--ledger", str(ledger), "--out", str(tmp_path / "o1")]) == cli.EXIT_REFUSED
    assert "uncommitted changes" in capsys.readouterr().err and not (tmp_path / "o1").exists()
    monkeypatch.setattr(provenance, "dirty_paths", lambda *a, **k: [])
    monkeypatch.setattr(provenance, "unverified_imported_code", lambda *a, **k: ["analysis/study_a.py (modified)"])
    assert cli.main(["--mode", "pilot", "--ledger", str(ledger), "--out", str(tmp_path / "o2")]) == cli.EXIT_REFUSED
    assert "does not contain" in capsys.readouterr().err and not (tmp_path / "o2").exists()
    # --allow-dirty is for smoke and test outputs only, never the repository's results/
    target = REPO / "results" / "analysis" / "never-written"
    assert cli.main(["--mode", "pilot", "--ledger", str(ledger), "--out", str(target), "--allow-dirty"]) == cli.EXIT_REFUSED
    assert not target.exists()
    assert cli.main(["--mode", "pilot", "--ledger", str(tmp_path / "missing.parquet")]) == cli.EXIT_REFUSED


def test_the_cli_refuses_code_that_changed_while_it_ran(tmp_path, clean_tree, monkeypatch, capsys) -> None:
    """HEAD moving during the run to a commit that changes loaded code (the working tree is clean
    at the new HEAD, so unverified_imported_code passes), or analysis/*.py changing on disk, is refused: a report must
    not name the start commit with another commit's analysis_code_hash."""
    from analysis import report

    monkeypatch.setattr(R, "BOOTSTRAP_RESAMPLES", 200)
    ledger = _pilot_world(tmp_path)
    run = report.run
    state = {"head": SHA}

    def moving_run(*a, **k):
        result = run(*a, **k)
        state["head"] = "c" * 40  # a commit lands while the analysis runs
        return result
    monkeypatch.setattr(report, "run", moving_run)
    monkeypatch.setattr(provenance, "commit_hash", lambda *a, **k: state["head"])
    monkeypatch.setattr(provenance, "imported_code_changed_between",
                        lambda old, new, *a, **k: ["analysis/verdict.py"] if (old, new) == (SHA, "c" * 40) else [])
    assert cli.main(["--mode", "pilot", "--ledger", str(ledger), "--out", str(tmp_path / "o1")]) == cli.EXIT_REFUSED
    assert "HEAD moved" in capsys.readouterr().err and not (tmp_path / "o1").exists()
    # a commit that changes no loaded code leaves the report the start commit's
    monkeypatch.setattr(provenance, "imported_code_changed_between", lambda *a, **k: [])
    state["head"] = SHA
    code = cli.main(["--mode", "pilot", "--ledger", str(ledger), "--out", str(tmp_path / "o2")])
    assert (tmp_path / "o2").exists() and code != cli.EXIT_REFUSED, capsys.readouterr().err
    # analysis/*.py changing on disk while the analysis runs
    hashes = iter(["1" * 64, "2" * 64])
    monkeypatch.setattr(records, "code_hash", lambda *a, **k: next(hashes, "2" * 64))
    monkeypatch.setattr(report, "run", run)
    assert cli.main(["--mode", "pilot", "--ledger", str(ledger), "--out", str(tmp_path / "o3")]) == cli.EXIT_REFUSED
    assert "changed while the analysis ran" in capsys.readouterr().err and not (tmp_path / "o3").exists()


def test_final_mode_refuses_pilot_rows(tmp_path, clean_tree, capsys) -> None:
    ledger = _pilot_world(tmp_path)
    assert cli.main(["--mode", "final", "--surplus-extra", "0", "--ledger", str(ledger), "--out", str(tmp_path / "o")]) == cli.EXIT_REFUSED
    assert "not reused" in capsys.readouterr().err and not (tmp_path / "o").exists()


@pytest.mark.usefixtures("keys_open")
def test_pilot_mode_reproduces_the_go_reports_g1_and_g3_numbers(tmp_path, clean_tree) -> None:
    ledger = _pilot_world(tmp_path)
    out = tmp_path / "pilot"
    assert cli.main(["--mode", "pilot", "--ledger", str(ledger), "--out", str(out)]) == cli.EXIT_OK
    result = json.loads((out / "analysis.json").read_text())
    rows = [r.model_dump(mode="python") for r in load_ledger_as_rows(ledger)]
    g1, g3 = go_decision.g1_feasibility(rows), go_decision.g3_signal(rows)
    ours = result["pilot"]
    for key in ("reference_cost_mean", "late_cost_mean", "reference_sd", "match_difference", "arms_matched"):
        assert ours["g1_numbers"][key] == g1.values[key]  # exactly: the same numbers from the analysis's own code
    for key in ("delta_mean", "U80", "U80_pooled"):
        assert ours["g3_numbers"][key] == pytest.approx(g3.values[key], rel=1e-12)
    assert all(row["agree"] for row in ours["consistency"]) and ours["mismatches"] == []
    assert ours["g1_numbers"]["reference_final_checkpoint_sd"] == pytest.approx(g1.values["reference_final_checkpoint_sd"])
    report = (out / "report.md").read_text()
    assert "never decides go" in report and "Pilot consistency with the go report" in report
    # Part 5.8's test run marks every verdict NOT COMPUTABLE (pilot) and completes (Part 3.6: "The pilot's runs are
    # not reused"); every verdict's code ran, and what it computed is beside it, never a registered verdict
    assert result["verdicts"] and all(
        v["status"] == v["proposal_status"] == V.NOT_COMPUTABLE and v["reason"].startswith("pilot test run (Part 5.8)")
        and not v["undecided_by"] and not v["readings"] and "pilot_test_computed_status" in v["numbers"]
        for v in result["verdicts"])
    by_id = {v["id"]: v for v in result["verdicts"]}
    assert by_id["H1"]["numbers"]["pilot_test_computed_status"] == V.NOT_COMPUTABLE  # no N = 0.10, 0.25, controls
    computed = {v["numbers"]["pilot_test_computed_status"] for v in result["verdicts"]}
    assert computed - {V.NOT_COMPUTABLE}  # the code paths ran and decided something: the test run exercises them
    with open(out / "tables" / "verdicts.csv", newline="") as fh:
        assert {r["status"] for r in csv.DictReader(fh)} == {V.NOT_COMPUTABLE}
    # the go conditions' numbers carry the open questions they rest on
    for row in ours["consistency"]:
        entry = questions.PILOT_ENTRIES[row["condition"]]
        assert row["provisional_on"] == ", ".join(questions.provisional_on(entry))
    assert "Q-g1-level" in {k for r in ours["consistency"] if r["condition"] == "G1" for k in r["provisional_on"].split(", ")}
    estimates = {e["condition"]: e for e in ours["estimates"]}
    assert estimates["hazard"]["diff"] == pytest.approx(g3.values["delta_mean"])
    assert estimates["hazard"]["n_x"] == len(R.PILOT_SEEDS) and estimates["dynamics"]["diff"] == pytest.approx(0.0)


def test_pilot_mode_with_a_crashed_run_completes(tmp_path, clean_tree) -> None:
    """A pilot run that crashed (Part 5.6: completed False, its replacement not yet run) leaves an arm
    with two completed seeds; the detectable size at alpha 0.025 with two seeds (d = 8.01) must not stop the run."""
    rows = []
    for spec in manifest.pilot():
        if spec.study == "A":
            late = spec.N == 0.5
            rows.append(syn.study_a_row(spec, cost=25.0, gaps={"hazard": 8.0 if late else 2.0, "dynamics": 1.0},
                                        completed=not (late and spec.seed == 2)))
    ledger = syn.write_ledger(tmp_path / "ledger.parquet", rows)
    out = tmp_path / "pilot"
    # the go report needs three seeds per arm, so nothing is compared: written, but exit 4, never 0
    assert cli.main(["--mode", "pilot", "--ledger", str(ledger), "--out", str(out)]) == cli.EXIT_PILOT_UNTESTED
    power = json.loads((out / "tables" / "power.json").read_text())
    used = {(r["n"], r["alpha"]): r for r in power if r["what"].startswith("80-percent point at the seed count used")}
    assert used[(2, R.ALPHA_PRIMARY_STUDY_B)]["computed"] == pytest.approx(8.0104, abs=1e-3)
    pilot = json.loads((out / "analysis.json").read_text())["pilot"]
    assert pilot["g3_numbers"]["U80"] is None  # equation (12) is registered for three seeds: not computed with two
    # Table 8.1 under the go report's conditions: "three seeds" values and the SDs need every pilot seed of each arm
    table = pilot["table_8_1"]
    assert table["Delta(0.50) under hazard relocation, three seeds (mean; U80)"] == [None, None]
    assert table["Seed-to-seed SD of final in-distribution cost, N = 0 arm"] is None
    assert table["Seed-to-seed SD of final in-distribution cost, N = 0.50 arm"] is None
    assert table["Completed pilot runs (N = 0 arm; N = 0.50 arm)"] == [len(R.PILOT_SEEDS), len(R.PILOT_SEEDS) - 1]
    assert table["Reference arm's cost under the dynamics perturbation (mean over seeds)"] == 26.0  # 25 + 1, three seeds


def test_pilot_table_8_1_reports_a_mean_over_seeds_only_when_every_seed_contributes() -> None:
    """As the go report: "The mean over seeds is reported only when every pilot seed of the reference arm
    contributes" (pilot.go_decision.report); a reference seed without gap_dynamics leaves it None, never a mean over
    the other seeds."""
    from analysis import data

    rows = [syn.study_a_row(spec, cost=25.0, gaps={"hazard": 8.0 if spec.N == 0.5 else 2.0,
                                                   "dynamics": None if (spec.N == 0.0 and spec.seed == 1) else 1.0})
            for spec in manifest.pilot() if spec.study == "A"]
    dataset = data.build_dataset(rows, data.Supplement(), mode="pilot")
    g1, g3 = pilot_check.g1_numbers(dataset), pilot_check.g3_numbers(dataset)
    table = pilot_check.table_8_1(dataset, g1, g3)
    assert table["Reference arm's cost under the dynamics perturbation (mean over seeds)"] is None
    assert table["Delta(0.50) under hazard relocation, three seeds (mean; U80)"] == (6.0, g3["U80"])
    assert table["Seed-to-seed SD of final in-distribution cost, N = 0 arm"] == g1["reference_sd"] == 0.0


def test_a_pilot_test_run_that_tests_nothing_is_refused(tmp_path, clean_tree, capsys) -> None:
    """The Part 5.8 test run must test something. The re-pilot revision on the pilot ledger and pilot
    mode on a registered ledger are refused (exit 2, nothing written); pilot rows on which the go report computes no
    G1 or G3 quantity are analysed and written, but exit 4, never 0."""
    ledger = _pilot_world(tmp_path)
    out = tmp_path / "revision1"
    assert cli.main(["--mode", "pilot", "--revision", "1", "--ledger", str(ledger), "--out", str(out)]) == cli.EXIT_REFUSED
    assert "test nothing" in capsys.readouterr().err and not out.exists()
    (tmp_path / "final").mkdir()
    registered = syn.write_ledger(tmp_path / "final" / "ledger.parquet", syn.a_world("supported", groups=("main",)))
    out = tmp_path / "on_final"
    assert cli.main(["--mode", "pilot", "--ledger", str(registered), "--out", str(out)]) == cli.EXIT_REFUSED
    assert "test nothing" in capsys.readouterr().err and not out.exists()
    rows = [syn.study_a_row(spec, cost=25.0, gaps={}, measurement_cost=None) for spec in manifest.pilot() if spec.study == "A"]
    (tmp_path / "bare").mkdir()
    bare = syn.write_ledger(tmp_path / "bare" / "ledger.parquet", rows)  # not yet measured, no battery
    out = tmp_path / "bare_out"
    assert cli.main(["--mode", "pilot", "--ledger", str(bare), "--out", str(out)]) == cli.EXIT_PILOT_UNTESTED
    assert "compared nothing" in capsys.readouterr().err
    pilot = json.loads((out / "analysis.json").read_text())["pilot"]
    assert pilot["compared"] == 0 and all(r["agree"] is None for r in pilot["consistency"])
    assert "NOTHING WAS COMPARED" in (out / "report.md").read_text()
    good = tmp_path / "good_out"  # the pilot world compares every quantity
    assert cli.main(["--mode", "pilot", "--ledger", str(ledger), "--out", str(good)]) == cli.EXIT_OK
    compared = len(pilot_check.G1_KEYS) + len(pilot_check.G3_KEYS)
    assert json.loads((good / "analysis.json").read_text())["pilot"]["compared"] == compared


def test_pilot_mode_exits_3_when_the_go_report_disagrees(tmp_path, clean_tree, monkeypatch, capsys) -> None:
    ledger = _pilot_world(tmp_path)
    real = go_decision.g3_signal

    def shifted(rows):
        cond = real(rows)
        cond.values["delta_mean"] += 0.5
        return cond

    monkeypatch.setattr(go_decision, "g3_signal", shifted)
    out = tmp_path / "pilot"
    assert cli.main(["--mode", "pilot", "--ledger", str(ledger), "--out", str(out)]) == cli.EXIT_PILOT_MISMATCH
    assert "delta_mean" in capsys.readouterr().err
    assert "The analysis and the go report disagree" in (out / "report.md").read_text()


def test_pilot_mode_compares_each_go_flag_with_the_same_clause(tmp_path, clean_tree, monkeypatch) -> None:
    # the reference arm's SD is 15 (> 10): rule 5 makes the task infeasible, so the analysis's full rule set does not
    # match the arms, while Part 6 G1's matching clause (|difference| <= 2.5) holds in both implementations
    ledger = _pilot_world(tmp_path, ref_costs=(10.0, 25.0, 40.0))
    out = tmp_path / "pilot"
    assert cli.main(["--mode", "pilot", "--ledger", str(ledger), "--out", str(out)]) == cli.EXIT_OK
    ours = json.loads((out / "analysis.json").read_text())["pilot"]
    assert ours["g1_numbers"]["arms_matched"] is True and ours["g1_numbers"]["matched_by_rules"] is False
    assert ours["g1_numbers"]["reference_sd_ok"] is False and ours["mismatches"] == []
    rows = {r["quantity"]: r for r in ours["consistency"]}
    assert rows["arms_matched"]["agree"] and rows["reference_sd_ok"]["agree"] and not rows["arms_matched"]["note"]
    # a flag that really differs is an error (exit 3), not excused as rounding
    real = go_decision.g1_feasibility

    def flipped(rows):
        cond = real(rows)
        cond.values["arms_matched"] = not cond.values["arms_matched"]
        return cond

    monkeypatch.setattr(go_decision, "g1_feasibility", flipped)
    assert cli.main(["--mode", "pilot", "--ledger", str(ledger), "--out", str(tmp_path / "p2")]) == cli.EXIT_PILOT_MISMATCH
    ours = json.loads((tmp_path / "p2" / "analysis.json").read_text())["pilot"]
    assert any("arms_matched" in m for m in ours["mismatches"])
    # so is G1's first clause, "Both pilot arms reach ... at most d + 2.5" (a go report with a wrong limit or a strict <)
    assert rows["both_arms_within_budget"]["agree"] and ours["g1_numbers"]["both_arms_within_budget"] is True

    def over_budget(rows):
        cond = real(rows)
        cond.values["both_arms_within_budget"] = not cond.values["both_arms_within_budget"]
        return cond

    monkeypatch.setattr(go_decision, "g1_feasibility", over_budget)
    assert cli.main(["--mode", "pilot", "--ledger", str(ledger), "--out", str(tmp_path / "p3")]) == cli.EXIT_PILOT_MISMATCH
    ours = json.loads((tmp_path / "p3" / "analysis.json").read_text())["pilot"]
    assert [m for m in ours["mismatches"] if "both_arms_within_budget" in m] == ours["mismatches"] != []


def test_a_g1_flag_that_differs_at_the_rounding_boundary_is_a_mismatch(monkeypatch) -> None:
    """The go report and the matching round G1's clauses alike, so a disagreement on
    arms_matched at match_difference 2.500000000000007 (where unrounded and rounded comparisons differ) can only
    be a defect, never excused as a rounding difference."""
    from types import SimpleNamespace

    from analysis import pilot_check

    d = 2.500000000000007  # unrounded above the tolerance, rounded within it
    assert d > R.MATCH_TOLERANCE and go_decision._at_most(d, R.MATCH_TOLERANCE)
    ours = {"reference_cost_mean": 24.90, "late_cost_mean": 27.40, "reference_sd": 1.0, "match_difference": d,
            "arms_matched": True, "reference_sd_ok": True}
    theirs = go_decision.Condition("G1 feasibility")
    theirs.values.update({**ours, "arms_matched": False})  # a go report that compared unrounded
    monkeypatch.setattr(go_decision, "g1_feasibility", lambda rows: theirs)
    monkeypatch.setattr(go_decision, "g3_signal", lambda rows: go_decision.Condition("G3 signal"))
    rows, mismatches = pilot_check.compare_with_go_report(SimpleNamespace(rows=[]), ours, {})
    assert any("arms_matched" in m for m in mismatches)
    row = next(r for r in rows if r["quantity"] == "arms_matched")
    assert row["agree"] is False and row["note"] == ""


def test_data_the_analysis_cannot_read_is_a_refusal_not_a_traceback(tmp_path, clean_tree, capsys) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    from results.ledger_schema import LEDGER_SCHEMA

    rows, _ = syn.b_world("supported")
    ledger = syn.write_ledger(tmp_path / "ledger.parquet", rows)
    # The corrected schema refuses a non-canonical few-shot key on write, so the bad key is put
    # into the Parquet file directly, as a ledger written before the correction could hold it.
    records = pq.read_table(ledger).to_pylist()
    for record in records:
        if record["sr_fewshot"]:
            record["sr_fewshot"] = [(k.replace("5.0_", "5_"), v) for k, v in record["sr_fewshot"]]
            bad = record["run_id"]
            break
    pq.write_table(pa.Table.from_pylist(records, schema=LEDGER_SCHEMA), ledger)
    out = tmp_path / "o"
    assert cli.main(["--mode", "final", "--surplus-extra", "0", "--ledger", str(ledger), "--out", str(out)]) == cli.EXIT_REFUSED
    err = capsys.readouterr().err
    assert "refused" in err and bad in err and "canonical" in err and not out.exists()


def test_a_value_error_below_the_data_layer_is_a_refusal(tmp_path, clean_tree, capsys, monkeypatch) -> None:
    from analysis import report

    ledger = _pilot_world(tmp_path)

    def broken(dataset, **kw):
        raise ValueError("task X has more than one N = 0 arm")

    monkeypatch.setattr(report, "run", broken)
    out = tmp_path / "o"
    assert cli.main(["--mode", "pilot", "--ledger", str(ledger), "--out", str(out)]) == cli.EXIT_REFUSED
    assert "more than one N = 0 arm" in capsys.readouterr().err and not out.exists()


def test_an_explicit_supplement_directory_must_exist(tmp_path, clean_tree, capsys) -> None:
    ledger = _pilot_world(tmp_path)
    out = tmp_path / "o"
    code = cli.main(["--mode", "pilot", "--ledger", str(ledger), "--supplement", str(tmp_path / "supplemnt"),
                     "--out", str(out)])
    assert code == cli.EXIT_REFUSED and "not an existing directory" in capsys.readouterr().err and not out.exists()
    (tmp_path / "a_file").write_text("")
    assert cli.main(["--mode", "pilot", "--ledger", str(ledger), "--supplement", str(tmp_path / "a_file"),
                     "--out", str(out)]) == cli.EXIT_REFUSED
    # the default <ledger dir>/supplement may be absent (no records yet): said on stderr and in the provenance
    assert cli.main(["--mode", "pilot", "--ledger", str(ledger), "--out", str(out)]) == cli.EXIT_OK
    assert "no supplement directory" in capsys.readouterr().err
    prov = json.loads((out / "analysis.json").read_text())["provenance"]
    assert prov["supplement_dir_exists"] is False and prov["supplement_files"] == 0


def test_the_ledger_hash_is_of_the_rows_analysed(tmp_path, clean_tree, capsys, monkeypatch) -> None:
    from analysis import report

    ledger = _pilot_world(tmp_path)
    analysed = hashlib.sha256(ledger.read_bytes()).hexdigest()
    real_run = report.run

    def run_then_append(dataset, **kw):  # another process appends a row while the analysis runs
        assert dataset.ledger_sha256 == analysed
        result = real_run(dataset, **kw)
        extra = next(r for r in syn.a_world("supported", groups=("main",), seeds=(9,)) if r["N"] == 0.0)
        syn.write_ledger(ledger, [extra])
        return result

    monkeypatch.setattr(report, "run", run_then_append)
    out = tmp_path / "o"
    assert cli.main(["--mode", "pilot", "--ledger", str(ledger), "--out", str(out)]) == cli.EXIT_REFUSED
    assert "changed while the analysis ran" in capsys.readouterr().err and not out.exists()


def test_the_data_files_that_change_verdicts_must_be_committed(tmp_path, clean_tree, capsys, monkeypatch) -> None:
    """pilot/cuts.json, analysis/AMENDMENTS.json and analysis/ERRATA.json are compared with the commit byte for
    byte (git's status ignores untracked files), parsed from the bytes compared, and recorded with their hashes."""
    from analysis import data

    ledger = _pilot_world(tmp_path)
    for committed_as in (lambda rel: None, lambda rel: b'{"amendments": []}'):  # untracked; differs from the commit
        monkeypatch.setattr(provenance, "file_committed_at", lambda c, rel, *a, **k: (
            committed_as(rel) if rel == "analysis/AMENDMENTS.json" else _as_on_disk(c, rel)))
        out = tmp_path / "o"
        assert cli.main(["--mode", "pilot", "--ledger", str(ledger), "--out", str(out)]) == cli.EXIT_REFUSED
        err = capsys.readouterr().err
        assert "analysis/AMENDMENTS.json" in err and "commit" in err and not out.exists()
    # --allow-dirty (smoke and tests): it runs and says what was not committed
    assert cli.main(["--mode", "pilot", "--ledger", str(ledger), "--out", str(tmp_path / "d"), "--allow-dirty"]) == cli.EXIT_OK
    inputs = json.loads((tmp_path / "d" / "analysis.json").read_text())["provenance"]["inputs"]
    assert inputs["analysis/AMENDMENTS.json"]["committed"] is False and inputs["analysis/ERRATA.json"]["committed"]
    assert inputs["analysis/AMENDMENTS.json"]["sha256"] == hashlib.sha256(records.AMENDMENTS_FILE.read_bytes()).hexdigest()
    # a cut record: read through the same check, and the analysis uses the bytes that were compared
    cuts = tmp_path / "cuts.json"
    cuts.write_text(json.dumps({"cuts": list(manifest.CUTS[:1]), "amendment": "A9"}))
    monkeypatch.setattr(data, "CUTS_FILE", cuts)
    monkeypatch.setattr(provenance, "file_committed_at", lambda c, rel, *a, **k: None if rel == str(cuts) else _as_on_disk(c, rel))
    assert cli.main(["--mode", "pilot", "--ledger", str(ledger), "--out", str(tmp_path / "c1")]) == cli.EXIT_REFUSED
    assert str(cuts) in capsys.readouterr().err
    monkeypatch.setattr(provenance, "file_committed_at", _as_on_disk)
    assert cli.main(["--mode", "pilot", "--ledger", str(ledger), "--out", str(tmp_path / "c2")]) == cli.EXIT_OK
    prov = json.loads((tmp_path / "c2" / "analysis.json").read_text())["provenance"]
    assert prov["recorded_cuts"] == list(manifest.CUTS[:1])
    assert prov["inputs"][str(cuts)] == {"sha256": hashlib.sha256(cuts.read_bytes()).hexdigest(), "committed": True}


def test_the_module_runs_as_a_command(tmp_path) -> None:
    ledger = _pilot_world(tmp_path)
    out = tmp_path / "o"
    proc = subprocess.run([sys.executable, "-m", "analysis", "--mode", "pilot", "--ledger", str(ledger), "--out", str(out),
                           "--allow-dirty"], cwd=REPO, capture_output=True, text=True, timeout=600)
    assert proc.returncode == cli.EXIT_OK, proc.stderr
    assert (out / "report.md").exists() and "G1" in proc.stdout
    help_text = subprocess.run([sys.executable, "-m", "analysis", "--help"], cwd=REPO, capture_output=True, text=True)
    assert "--mode" in help_text.stdout and "--supplement" in help_text.stdout


# ---------------------------------------------------------------------------
# AMENDMENTS.json and ERRATA.json
# ---------------------------------------------------------------------------


def test_the_committed_registries_are_documented_and_hold_the_decisions_of_2026_10_02() -> None:
    """AMENDMENTS.json holds one entry, the decisions of 2026-10-02 (docs/DECISIONS.md): every PENDING key, made
    before any data were seen, its commit not yet recorded (null) until the group ratifies it in Table 9.1; it
    relabels nothing. ERRATA.json is empty."""
    (amendment,) = records.load_amendments()
    assert amendment["keys"] == sorted(R.PENDING) and set(amendment["keys"]) == set(R.ANSWERED_QUESTIONS)
    assert amendment["data_seen"] is False and amendment["affects"] == [] and amendment["date"] == "2026-10-02"
    assert amendment["commit"] is None and "analysis_code_hash" not in amendment  # no hash is invented
    assert "ratif" in amendment["approved_by"] and "docs/DECISIONS.md" in amendment["table_9_1_row"]
    assert records.load_errata() == []
    for path in (records.AMENDMENTS_FILE, records.ERRATA_FILE):
        doc = json.loads(path.read_text())
        assert doc["about"] and doc["fields"]
    assert "null while it is not yet recorded" in json.loads(records.AMENDMENTS_FILE.read_text())["fields"]["commit"]


def test_only_an_amendment_may_carry_a_commit_not_yet_recorded(tmp_path) -> None:
    """The commit that records an entry cannot name itself: an amendment's commit may be null (not yet recorded,
    as AMENDMENTS.json documents); a non-null commit must still be a 40-hex hash, and an erratum (the fix's own
    commit, recorded after it) needs its hash."""
    (entry,) = records.load_amendments(_registry(tmp_path, "a.json", "amendments", [_amendment(commit=None)]))
    assert entry["commit"] is None
    for bad in ("", "pending", 0):
        with pytest.raises(records.RegistryError, match="40-hex|commit must be a string"):
            records.load_amendments(_registry(tmp_path, "b.json", "amendments", [_amendment(commit=bad)]))
    erratum = {"id": "E1", "table_9_1_row": "Erratum 1", "date": "2027-02-01", "commit": None,
               "files": ["analysis/stats.py"], "description": "fix", "code_hash_before": "0" * 64,
               "code_hash_after": "1" * 64}
    with pytest.raises(records.RegistryError, match="commit must be a string"):
        records.load_errata(_registry(tmp_path, "e.json", "errata", [erratum]))


def _amendment(**kw) -> dict:
    entry = {"id": "A1", "table_9_1_row": "First amendment", "date": "2027-01-01", "commit": "c" * 40,
             "change": "answers Q-interval", "reason": "go decision", "approved_by": "Group",
             "keys": ["Q-interval"], "data_seen": True, "affects": []}
    entry.update(kw)
    return entry


def _registry(tmp_path: Path, name: str, list_key: str, entries: list[dict]) -> Path:
    source = records.AMENDMENTS_FILE if list_key == "amendments" else records.ERRATA_FILE
    doc = json.loads(source.read_text())
    doc[list_key] = entries
    path = tmp_path / name
    path.write_text(json.dumps(doc))
    return path


def test_an_amendment_made_after_data_were_seen_labels_what_it_affects_exploratory(tmp_path) -> None:
    path = _registry(tmp_path, "a.json", "amendments", [_amendment(), _amendment(id="A2", keys=[], affects=["G2*"])])
    amendments = records.load_amendments(path)

    def verdict(vid: str, table_key: str, label: str = V.CONFIRMATORY) -> V.Verdict:
        return V.Verdict(id=vid, title="", statement="", label=label, status=V.SUPPORTED, table_key=table_key)

    h1, g2 = verdict("H1", "H1"), verdict("G2", "G2")
    ctrl = verdict("A-controller", "A-controller", V.SECONDARY)  # depends on neither Q-interval nor the G2* pattern
    records.apply_amendments([h1, g2, ctrl], amendments)
    assert h1.label == V.EXPLORATORY and "['A1']" in h1.notes[-1]  # H1 depends on Q-interval
    assert g2.label == V.EXPLORATORY and "['A1', 'A2']" in g2.notes[-1]  # Q-interval and the pattern
    assert ctrl.label == V.SECONDARY
    # each rule on its own, on verdicts that depend on no data_seen key (A-controller's table entry)
    pattern = [_amendment(id="A2", keys=[], affects=["G2*"])]
    exact = [_amendment(id="A3", keys=[], affects=["H1"])]
    for vid, applied, hit in (("G2x", pattern, True), ("G1", pattern, False),
                              ("H1", exact, True), ("H1x", exact, False)):
        v = verdict(vid, "A-controller")
        records.apply_amendments([v], applied)
        assert (v.label == V.EXPLORATORY) is hit, vid
    before = verdict("H1", "H1")
    records.apply_amendments([before], [_amendment(data_seen=False)])
    assert before.label == V.CONFIRMATORY  # a change before the data were seen changes no label


@pytest.mark.parametrize("bad, message", [
    ({"date": "1 Jan 2027"}, "YYYY-MM-DD"),
    ({"date": "2027-02-30"}, "not a calendar date"),
    ({"commit": "abc"}, "40-hex"),
    ({"keys": ["Q-not-a-key"]}, "not PENDING keys"),
    ({"data_seen": "yes"}, "true or false"),
    ({"affects": "H1"}, "list"),
    ({"surprise": 1}, "undocumented"),
    ({"analysis_code_hash": "abc"}, "SHA-256"),
    # a trailing newline is no part of a hash or a date ("$" would match before it; the patterns end in \Z)
    ({"commit": "c" * 40 + "\n"}, "40-hex"),
    ({"analysis_code_hash": "0" * 64 + "\n"}, "SHA-256"),
    ({"date": "2027-01-01\n"}, "YYYY-MM-DD"),
    # wrong types are refusals too, never a TypeError or AttributeError that python -m analysis would show as a traceback
    ({"keys": None}, "keys must be a list of strings"),
    ({"id": ["A1"]}, "id must be a string"),
    ({"affects": [1]}, "affects must be a list of strings"),
    ({"date": 20270101}, "date must be a string"),
    ({"change": None}, "change must be a string"),
    ({"analysis_code_hash": 5}, "SHA-256"),
])
def test_a_malformed_amendment_is_refused(tmp_path, bad, message) -> None:
    path = _registry(tmp_path, "a.json", "amendments", [_amendment(**bad)])
    with pytest.raises(records.RegistryError, match=message):
        records.load_amendments(path)


def test_an_amendment_id_twice_is_refused(tmp_path) -> None:
    duplicate = _registry(tmp_path, "d.json", "amendments", [_amendment(), _amendment()])
    with pytest.raises(records.RegistryError, match="twice"):
        records.load_amendments(duplicate)


@pytest.mark.parametrize("fields", [None, {}, {"id": "the amendment's id"}])
def test_a_registry_that_does_not_document_its_fields_is_refused(tmp_path, fields) -> None:
    path = _registry(tmp_path, "a.json", "amendments", [_amendment()])
    doc = json.loads(path.read_text())
    doc["fields"] = fields
    path.write_text(json.dumps(doc))
    with pytest.raises(records.RegistryError, match="'fields' documents"):
        records.load_amendments(path)


def test_the_part_5_8_freeze_state(tmp_path) -> None:
    current = records.code_hash()
    assert "no go-decision hash" in records.freeze_state([], [], current)["state"]
    go = [_amendment(analysis_code_hash=current)]
    assert records.freeze_state(go, [], current)["state"] == "unchanged since the go decision"
    old = "0" * 64
    erratum = {"id": "E1", "table_9_1_row": "Erratum 1", "date": "2027-02-01", "commit": "d" * 40,
               "files": ["analysis/stats.py"], "description": "fix", "code_hash_before": old, "code_hash_after": current}
    errata = records.load_errata(_registry(tmp_path, "e.json", "errata", [erratum]))
    assert "E1" in records.freeze_state([_amendment(analysis_code_hash=old)], errata, current)["state"]
    flagged = records.freeze_state([_amendment(analysis_code_hash=old)], [], current)
    assert flagged["flagged"] and "without a recorded erratum" in flagged["state"]
    # the errata must chain from the go hash; an unrecorded edit before a recorded fix is flagged
    edited, mid = "b" * 64, "c" * 64
    gap = records.freeze_state([_amendment(analysis_code_hash=old)], [{**erratum, "code_hash_before": edited}], current)
    assert gap["flagged"] and "unrecorded edit" in gap["state"] and old in gap["state"], gap
    two = [{**erratum, "code_hash_after": mid}, {**erratum, "id": "E2", "code_hash_before": mid}]
    chained = records.freeze_state([_amendment(analysis_code_hash=old)], two, current)
    assert "flagged" not in chained and "['E1', 'E2']" in chained["state"]
    broken = records.freeze_state([_amendment(analysis_code_hash=old)], [two[0], {**two[1], "code_hash_before": edited}],
                                  current)
    assert broken["flagged"] and mid in broken["state"] and "['E1']" in broken["state"]
    for field, value in (("code_hash_after", "x"), ("code_hash_after", None), ("files", "analysis/stats.py"),
                         ("files", [1]), ("id", {"E": 1}), ("description", 3), ("code_hash_before", old + "\n")):
        bad = copy.deepcopy(erratum)
        bad[field] = value
        with pytest.raises(records.RegistryError):
            records.load_errata(_registry(tmp_path, "f.json", "errata", [bad]))
    assert len(current) == 64


# Imports every analysis module, then every module an analysis function imports when it runs (a function-level import
# is loaded only then), and prints the repository files in sys.modules: the analysis's in-repository import closure.
_CLOSURE_SCRIPT = r"""
import ast, importlib, json, pathlib, sys
repo = pathlib.Path.cwd().resolve()
modules = sorted(repo.glob("analysis/*.py"))
for path in modules:
    importlib.import_module("analysis" if path.stem == "__init__" else f"analysis.{path.stem}")
for path in modules:
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module] + [f"{node.module}.{alias.name}" for alias in node.names]
        for name in names:
            try:
                importlib.import_module(name)
            except ImportError:  # a name imported from a module, not a module
                pass
files = set()
for module in list(sys.modules.values()):
    path = pathlib.Path(getattr(module, "__file__", None) or "/").resolve()
    if path.is_file() and repo in path.parents:
        files.add(path.relative_to(repo).as_posix())
print(json.dumps(sorted(files)))
"""


def test_the_code_hash_covers_every_in_repository_module_the_analysis_loads() -> None:
    """Q-exploratory-labels (answered in Table 9.1): records.code_files covers analysis/*.py and every
    in-repository module the analysis loads, directly or transitively, function-level imports included. A fresh
    interpreter imports every analysis module and everything their functions import; every repository file it
    then holds is in code_files, and IMPORTED_CODE is exactly the files outside analysis/ (none hashed in vain)."""
    out = subprocess.run([sys.executable, "-c", _CLOSURE_SCRIPT], cwd=REPO, capture_output=True, text=True, check=True)
    loaded = set(json.loads(out.stdout))
    files = records.code_files()
    assert loaded - set(files) == set()
    assert set(records.IMPORTED_CODE) == {f for f in loaded if not f.startswith("analysis/")}
    assert {"configs/__init__.py", "configs/registered.py", "pilot/__init__.py", "pilot/manifest.py",
            "pilot/contracts.py", "pilot/provenance.py", "pilot/go_decision.py", "pilot/budget.py", "pilot/rundir.py",
            "pilot/errors.py", "results/ledger_schema.py", "results/supplement_schema.py"} <= set(records.IMPORTED_CODE)
    assert list(records.IMPORTED_CODE) == sorted(records.IMPORTED_CODE)


def test_the_code_hash_covers_the_modules_the_analysis_imports(tmp_path) -> None:
    """Q-exploratory-labels (answered in Table 9.1): records.code_hash, the Part 5.8 freeze, covers analysis/*.py and
    the modules the analysis loads from outside it (records.IMPORTED_CODE), each as (repository-relative path, NUL,
    bytes, NUL) in sorted path order: an edit to a registered constant after the go decision changes the hash, so it
    needs a recorded erratum."""
    files = records.code_files()
    assert set(records.IMPORTED_CODE) <= set(files) and "analysis/records.py" in files and files == sorted(files)
    for rel in files:  # a copy of the tree the hash reads
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes((REPO / rel).read_bytes())
    current = records.code_hash()
    assert records.code_hash(tmp_path) == current
    digest = hashlib.sha256()
    for rel in files:
        digest.update(rel.encode("utf-8") + b"\0" + (REPO / rel).read_bytes() + b"\0")
    assert digest.hexdigest() == current
    registered = tmp_path / "configs" / "registered.py"
    registered.write_bytes(registered.read_bytes() + b"\n# an edit after the go decision\n")
    assert records.code_hash(tmp_path) != current
    registered.unlink()  # a missing module is an error, never a hash without it
    with pytest.raises(OSError):
        records.code_hash(tmp_path)


def test_pilot_g3_pairs_the_runs_a_replacement_leaves_unpaired_in_seed_order() -> None:
    """Q-g3-pairing (answered in Table 9.1), as the go report decides G3: the late arm's seed 1 crashed and its
    replacement ran on seed 5, so only seeds 0 and 2 are seed-matched; the runs left unpaired (late 5, reference 1) are
    paired in increasing seed order, equation (12) is applied unchanged, and the pairing used is reported."""
    from analysis import data, stats

    late_gaps, ref_gaps = {0: 8.0, 2: 9.5, 5: 6.0}, {0: 1.0, 1: 2.0, 2: 3.0}
    rows = []
    for spec in manifest.pilot():
        if spec.study != "A":
            continue
        late = spec.N == 0.5
        notes = ""
        if late and spec.seed == 1:  # Part 5.6: the crashed run, then its replacement
            rows.append(syn.study_a_row(spec, cost=25.0, gaps={"hazard": None, "dynamics": None}, completed=False))
            notes, spec = manifest.replacement_note(spec.run_id), spec.with_seed(5)
        gap = (late_gaps if late else ref_gaps)[spec.seed]
        rows.append(syn.study_a_row(spec, cost=25.0, gaps={"hazard": gap, "dynamics": 1.0}, notes=notes))
    g3 = pilot_check.g3_numbers(data.build_dataset(rows, data.Supplement(), mode="pilot"))
    assert g3["paired_seeds"] == [0, 2] and g3["pairs"] == [(0, 0), (2, 2), (5, 1)]
    assert g3["pairing"].startswith("by seed id, then the runs left unpaired in increasing seed order")
    diffs = [8.0 - 1.0, 9.5 - 3.0, 6.0 - 2.0]
    assert g3["paired_differences"] == diffs and g3["U80"] == stats.u80(diffs)
    assert g3["delta_mean"] == pytest.approx(sum(diffs) / 3)


def test_the_final_analysis_needs_the_seed_targets_and_names_the_study_b_arms_still_to_come(
        tmp_path, clean_tree, capsys, monkeypatch) -> None:
    """The final analysis must not decide Study B on whatever seeds are present. It needs E (Part 5.5)
    for the seed targets of Q-arm-complete, and an arm below its target is listed and left incomplete."""
    monkeypatch.setattr(R, "BOOTSTRAP_RESAMPLES", 2_000)
    ledger = _final_world(tmp_path)
    assert cli.main(["--mode", "final", "--ledger", str(ledger), "--out", str(tmp_path / "a")]) == cli.EXIT_REFUSED
    assert "needs --surplus-extra E" in capsys.readouterr().err and not (tmp_path / "a").exists()
    assert cli.main(["--mode", "pilot", "--surplus-extra", "0", "--ledger", str(ledger),
                     "--out", str(tmp_path / "p")]) == cli.EXIT_REFUSED
    capsys.readouterr()
    # --revision selects the pilot or the re-pilot: in final mode it means nothing and is refused, never recorded
    assert cli.main(["--mode", "final", "--surplus-extra", "1", "--revision", "1", "--ledger", str(ledger),
                     "--out", str(tmp_path / "r")]) == cli.EXIT_REFUSED
    assert "takes no --revision" in capsys.readouterr().err and not (tmp_path / "r").exists()
    out = tmp_path / "b"
    assert cli.main(["--mode", "final", "--surplus-extra", "1", "--ledger", str(ledger), "--out", str(out)]) == cli.EXIT_OK
    captured = capsys.readouterr()
    assert "Study B arms are below their seed target" in captured.err and "G1: NOT_COMPUTABLE" in captured.out
    result = json.loads((out / "analysis.json").read_text())
    six = len(R.SEEDS) + 1
    assert f"B-Dense: {len(R.SEEDS)} of its {six} seeds completed (Q-arm-complete)" in result["awaited_study_b_arms"]
    assert result["provenance"]["surplus_extra"] == 1 and result["provenance"]["revision"] is None
    assert "Study B arms below their seed target" in (out / "report.md").read_text()

"""The ledger and supplement as per-seed tables (analysis/data.py; Part 5.1, 5.6; results/supplement_schema.py; analysis/__main__.py)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import test_analysis_synthetic as syn  # noqa: E402

from analysis import data  # noqa: E402
from configs import registered as R  # noqa: E402
from pilot import manifest  # noqa: E402
from results import supplement_schema as S  # noqa: E402


def _main_rows():
    return syn.a_world("supported", groups=("main",))


def test_the_supplement_reader_does_not_depend_on_the_layout(tmp_path) -> None:
    rows = _main_rows()
    row = rows[-1]
    measurement = syn.measurement_record(row)
    (tmp_path / "deep" / "er").mkdir(parents=True)
    (tmp_path / "deep" / "er" / "anything.json").write_text(S.canonical_json("measurement", measurement))
    battery = syn.battery_record(row, "hazard")
    del battery["kind"]  # no kind field: the first directory names it
    (tmp_path / "battery").mkdir()
    (tmp_path / "battery" / "x.json").write_text(json.dumps(battery))
    sup = data.load_supplement(tmp_path)
    assert sup.get("measurement", row["run_id"]).step == row["matched_checkpoint_step"]
    assert sup.get("battery", row["run_id"], "hazard").gap == battery["gap"]
    assert sorted(sup.files) == ["battery/x.json", "deep/er/anything.json"]
    assert data.load_supplement(tmp_path / "missing").records == {}
    assert data.load_supplement(None).records == {}


@pytest.mark.parametrize("content, message", [
    ("{not json", "not readable JSON"),
    ("[1, 2]", "does not hold a record"),
    (json.dumps({"run_id": "A-PointGoal1-N0.00-s0"}), "names no kind"),
    (json.dumps({"kind": "measurement", "run_id": "A-PointGoal1-N0.00-s0"}), "not a valid measurement record"),
    # a kind that is not a string is a refusal (DataError), not a TypeError shown as a traceback
    (json.dumps({"kind": ["battery"], "run_id": "A-PointGoal1-N0.00-s0"}), "names kind"),
    (json.dumps({"kind": {"battery": 1}, "run_id": "A-PointGoal1-N0.00-s0"}), "names kind"),
    (json.dumps({"kind": "episodes", "run_id": "A-PointGoal1-N0.00-s0"}), "names kind 'episodes'"),
])
def test_a_bad_supplement_file_is_refused(tmp_path, content, message) -> None:
    (tmp_path / "f.json").write_text(content)
    with pytest.raises(data.DataError, match=message):
        data.load_supplement(tmp_path)


def test_a_record_is_written_once(tmp_path) -> None:
    record = syn.measurement_record(_main_rows()[0])
    (tmp_path / "a.json").write_text(S.canonical_json("measurement", record))
    (tmp_path / "b.json").write_text(S.canonical_json("measurement", record))
    with pytest.raises(data.DataError, match="two measurement records"):
        data.load_supplement(tmp_path)


def test_final_mode_refuses_pilot_rows_and_pilot_mode_reads_one_pilot() -> None:
    main = _main_rows()
    pilot_rows = []
    for spec in manifest.pilot():
        if spec.study == "A":
            pilot_rows.append(syn.study_a_row(spec, cost=25.0, gaps={"hazard": 1.0}))
    with pytest.raises(data.DataError, match="not reused"):
        data.build_dataset(main + pilot_rows, data.Supplement(), mode="final")
    ds = data.build_dataset(main + pilot_rows, data.Supplement(), mode="pilot")
    assert ds.population == "P-" and {r["run_id"] for r in ds.study_a} == {r["run_id"] for r in pilot_rows}
    assert len(ds.ignored) == len(main)
    # a pilot population without a completed Study A run leaves the Part 5.8 test run nothing to test
    with pytest.raises(data.DataError, match="P1-.*test nothing"):
        data.build_dataset(main + pilot_rows, data.Supplement(), mode="pilot", revision=1)
    with pytest.raises(data.DataError, match="test nothing"):
        data.build_dataset(main, data.Supplement(), mode="pilot")  # a registered ledger in pilot mode
    failed = [dict(r, completed=False, failure_cause="crash") for r in pilot_rows]
    with pytest.raises(data.DataError, match="test nothing"):
        data.build_dataset(failed, data.Supplement(), mode="pilot")
    with pytest.raises(data.DataError):
        data.build_dataset(main, data.Supplement(), mode="other")


def test_only_failed_runs_leave_the_tables() -> None:
    rows = _main_rows()
    spec = syn.specs(groups=("main",))[3]
    failed = syn.study_a_row(spec.with_seed(5), cost=25.0, gaps={}, completed=False)
    ds = data.build_dataset(rows + [failed], data.Supplement(), mode="final")
    assert [e["run_id"] for e in ds.exclusions] == [failed["run_id"]] and ds.exclusions[0]["failure_cause"] == "crash"
    assert len(ds.study_a) == len(rows)  # every completed run is kept, whatever its numbers
    seeds = ds.seeds_per_arm()
    assert seeds[spec.arm_id] == list(R.SEEDS)  # the failed run's own arm keeps its completed seeds only


def test_records_join_the_ledger_and_disagreements_are_reported() -> None:
    rows = _main_rows()
    row = rows[-1]
    spec = next(s for s in syn.specs(groups=("main",)) if s.run_id == row["run_id"])
    good = syn.measurement_record(row)
    wrong = syn.battery_record(row, "hazard", gap=row["gap_hazard"] + 1.0)
    orphan = syn.measurement_record(dict(row, run_id="A-PointGoal1-N0.00-s9"))
    foreign = syn.measurement_record(dict(row, run_id="X-foo-s0"))  # a valid RunId of neither study
    training = syn.training_record(row, spec=spec, dormant_check=0.3, rank_check=30.0)
    sup = data.Supplement()
    for kind, rec in (("measurement", good), ("battery", wrong), ("measurement", orphan), ("measurement", foreign),
                      ("training", training)):
        model = S.validate_record(kind, rec)
        sup.records.setdefault(kind, {})[(model.run_id, S.record_part(kind, model))] = model
    ds = data.build_dataset(rows, sup, mode="final")
    rec = next(r for r in ds.study_a if r["run_id"] == row["run_id"])
    assert rec["measurement_return"] == good["episodes"]["mean_return"]
    assert rec["battery_episodes"]["hazard"] == wrong["episodes"]["episode_costs"]
    assert rec["onset_step"] == spec.onset_step and rec["onset_source"] == "supplement"
    assert rec["dormant_trainable_check"] == 0.3 and rec["recovery_steps"] == 400_000
    assert rec["constrained_steps_at_selection"] == rec["training_age"] - spec.onset_step
    assert any("gap_hazard" in p for p in ds.problems)
    assert any("without a ledger row" in p and "N0.00-s9" in p for p in ds.problems)
    assert any("not of study A or B" in p and "X-foo-s0" in p for p in ds.problems)
    other = next(r for r in ds.study_a if r["run_id"] != row["run_id"] and r["N"] == 0.25)
    assert other["onset_source"] == "design" and other["onset_step"] > 0  # pilot.manifest.onset_schedule


def _supplement(*records: tuple[str, dict]) -> data.Supplement:
    sup = data.Supplement()
    for kind, record in records:
        model = S.validate_record(kind, record)
        sup.records.setdefault(kind, {})[(model.run_id, S.record_part(kind, model))] = model
    return sup


def _continuation(row: dict, *, parent_step: int, measurement_cost: float) -> dict:
    return {"kind": "continuation", "run_id": row["run_id"], "code_commit": syn.COMMIT, "condition": "finetune",
            "continuation_run_id": row["run_id"].replace("-s", "-finetune-s"), "parent_step": parent_step,
            "continuation_steps": R.FINETUNE_STEPS, "step": R.FINETUNE_STEPS, "task": row["task"],
            "episodes": syn.episodes(27.0), "measurement_cost": measurement_cost, "gap": 27.0 - measurement_cost}


def test_a_record_of_another_checkpoint_or_c_id_is_a_problem_and_is_not_read() -> None:
    """Every record is read as the checkpoint it belongs to (Part 4.1): the matched checkpoint with the
    ledger's C_ID (battery, sensitivity battery of rule 7 (a), continuation, measurement) or the final checkpoint (final
    battery, rule 7 (b)). A record of another one is a data problem and is not read, never silently used."""
    rows = _main_rows()
    late = [r for r in rows if r["N"] == 0.25 and r["onset_shape"] == "abrupt" and r["step_matching"] == "total_steps"]
    bad, good = late[0], late[1]
    specs = {s.run_id: s for s in syn.specs(groups=("main",))}
    for row in (bad, good):
        row["measurement_cost"] = 29.0  # the C_ID the records below use (the arm stays matched: mean 26.75)
    matched = bad["matched_checkpoint_step"]
    earlier = matched - R.CHECKPOINT_INTERVAL_STEPS
    final = specs[bad["run_id"]].total_steps
    assert matched == final == max(c["step"] for c in bad["checkpoints"])
    measure_at = syn.measurement_record(bad)
    measure_at["step"] = earlier
    final_measure = syn.battery_record(bad, "dynamics", kind="final_battery", step=earlier)
    final_measure.update(condition="measurement", measurement_cost=None, gap=None, episodes=syn.episodes(29.0))
    sup = _supplement(
        ("sensitivity_battery", syn.battery_record(bad, "hazard", kind="sensitivity_battery", step=earlier)),
        ("sensitivity_battery", syn.battery_record(bad, "dynamics", kind="sensitivity_battery", measurement_cost=20.0)),
        ("battery", syn.battery_record(bad, "hazard", step=earlier)),
        ("battery", syn.battery_record(bad, "dynamics", measurement_cost=20.0, gap=bad["gap_dynamics"] + 9.0)),
        ("final_battery", syn.battery_record(bad, "hazard", kind="final_battery", step=earlier)),
        ("final_battery", final_measure),
        ("continuation", _continuation(bad, parent_step=earlier, measurement_cost=29.0)),
        ("measurement", measure_at),
        # the other run's records are of its matched and final checkpoints with its C_ID: read
        ("sensitivity_battery", syn.battery_record(good, "hazard", kind="sensitivity_battery")),
        ("battery", syn.battery_record(good, "hazard")),
        ("final_battery", syn.battery_record(good, "hazard", kind="final_battery", step=final)),
        ("continuation", _continuation(good, parent_step=good["matched_checkpoint_step"], measurement_cost=29.0)),
        ("measurement", syn.measurement_record(good)),
    )
    ds = data.build_dataset(rows, sup, mode="final")
    rec = next(r for r in ds.study_a if r["run_id"] == bad["run_id"])
    assert rec["sens_gap_hazard"] is None and rec["sens_gap_dynamics"] is None
    assert rec["battery_episodes"] == {} and rec["final_gap_hazard"] is None and rec["final_episodes"] == {}
    assert rec["measurement_return"] is None and rec["measurement_episodes"] is None
    mine = [p for p in ds.problems if p.startswith(bad["run_id"])]
    for what in ("sensitivity_battery (hazard) record is at step", "sensitivity_battery (dynamics) record uses C_ID 20.0",
                 "battery (hazard) record is at step", "battery (dynamics) record uses C_ID 20.0",
                 "final_battery (hazard) record is at step", "final_battery (measurement) record is at step",
                 "continuation (finetune) record is at step", "measurement record is at step"):
        assert any(what in p and "not read" in p for p in mine), what
    ok = next(r for r in ds.study_a if r["run_id"] == good["run_id"])
    assert ok["sens_gap_hazard"] is not None and set(ok["battery_episodes"]) == {"hazard", "finetune"}
    assert ok["final_gap_hazard"] is not None and ok["measurement_return"] is not None
    assert not [p for p in ds.problems if p.startswith(good["run_id"]) and "not read" in p]
    # the evaluation record's final step, when there is one, names the final checkpoint
    evaluation = {"kind": "evaluation", "run_id": good["run_id"], "code_commit": syn.COMMIT, "final_step": final,
                  "final": syn.episodes(good["final_cost"]), "short_episodes": 0,
                  "selection": [{"step": c["step"], **syn.episodes(25.0, seed_set="selection", base=1_000_000)}
                                for c in sorted(good["checkpoints"], key=lambda c: c["step"])[-R.SELECTION_WINDOW_CHECKPOINTS:]]}
    assert data._final_step(good, _supplement(("evaluation", evaluation))) == final


def test_a_measurement_or_final_battery_record_of_another_c_id_is_not_read() -> None:
    """The module docstring's "A record of another checkpoint or another C_ID is a data problem and is not read" also
    holds for a measurement record whose mean differs from the ledger's measurement_cost (its episodes are the C_ID
    episodes of every gap IQM; its return decides A-return) and for a final_battery condition record built against
    another C_ID than the final checkpoint's own measurement record (rule 7 (b), A-final-estimand)."""
    rows = _main_rows()
    late = [r for r in rows if r["N"] == 0.25 and r["onset_shape"] == "abrupt" and r["step_matching"] == "total_steps"]
    bad, good = late[0], late[1]
    final = {s.run_id: s for s in syn.specs(groups=("main",))}[bad["run_id"]].total_steps
    off = syn.measurement_record(bad)
    off["episodes"] = syn.episodes(bad["measurement_cost"] + 10.0)
    final_measure = syn.battery_record(bad, "dynamics", kind="final_battery", step=final)
    final_measure.update(condition="measurement", measurement_cost=None, gap=None, episodes=syn.episodes(25.0))
    final_hazard = syn.battery_record(bad, "hazard", kind="final_battery", step=final, measurement_cost=15.0)
    good_measure = syn.battery_record(good, "dynamics", kind="final_battery", step=final)
    good_measure.update(condition="measurement", measurement_cost=None, gap=None, episodes=syn.episodes(25.0))
    sup = _supplement(("measurement", off), ("final_battery", final_measure), ("final_battery", final_hazard),
                      ("measurement", syn.measurement_record(good)), ("final_battery", good_measure),
                      ("final_battery", syn.battery_record(good, "hazard", kind="final_battery", step=final,
                                                           measurement_cost=25.0)))
    ds = data.build_dataset(rows, sup, mode="final")
    rec = next(r for r in ds.study_a if r["run_id"] == bad["run_id"])
    assert rec["measurement_return"] is None and rec["measurement_episodes"] is None
    assert rec["final_gap_hazard"] is None and "hazard" not in rec["final_episodes"]
    assert rec["final_measurement_cost"] == 25.0  # the final checkpoint's own measurement record is read
    mine = [p for p in ds.problems if p.startswith(bad["run_id"])]
    assert any("measurement record uses C_ID" in p and "not read" in p for p in mine), mine
    assert any("final battery's hazard record uses C_ID 15.0" in p and "not read" in p for p in mine), mine
    ok = next(r for r in ds.study_a if r["run_id"] == good["run_id"])
    assert ok["measurement_return"] is not None and ok["final_gap_hazard"] is not None
    assert not [p for p in ds.problems if p.startswith(good["run_id"]) and "not read" in p]


def test_the_ledger_is_read_before_the_supplement(tmp_path, monkeypatch) -> None:
    """The ledger writer (pilot/ledger_writer.py) writes a run's supplement record BEFORE its row. Reading the ledger
    first and the supplement next means every row read has its records on disk; a writer that finishes while the
    supplement is scanned changes the ledger, which is hashed again after the scan, and the read is refused."""
    rows = _main_rows()
    first, last = rows[:-1], rows[-1]
    sup_dir = tmp_path / "supplement"
    for row in first:
        syn.write_supplement(sup_dir, "measurement", syn.measurement_record(row))
    ledger = syn.write_ledger(tmp_path / "ledger.parquet", first)
    order: list[str] = []
    real_rows, real_supplement = data.load_rows, data.load_supplement
    monkeypatch.setattr(data, "load_rows", lambda path: order.append("ledger") or real_rows(path))

    def scan_then_writer_finishes(path):
        order.append("supplement")
        out = real_supplement(path)
        syn.write_supplement(sup_dir, "measurement", syn.measurement_record(last))  # the record first ...
        syn.write_ledger(ledger, [last])  # ... then the row
        return out

    monkeypatch.setattr(data, "load_supplement", scan_then_writer_finishes)
    with pytest.raises(data.DataError, match="changed while"):
        data.load_dataset(ledger, mode="final")
    assert order == ["ledger", "supplement"]
    monkeypatch.setattr(data, "load_supplement", real_supplement)
    ds = data.load_dataset(ledger, mode="final")  # nothing writes now: every row with its record
    assert len(ds.study_a) == len(rows) and all(r["measurement_episodes"] is not None for r in ds.study_a)


def test_fewshot_censoring_is_read_above_the_largest_horizon_of_each_continuation() -> None:
    rows, records = syn.b_world()
    sup = data.Supplement()
    for run_id, recs in records.items():
        for f in recs["fewshot"]:
            model = S.validate_record("fewshot", f)
            sup.records.setdefault("fewshot", {})[(run_id, S.record_part("fewshot", model))] = model
    ds = data.build_dataset(rows, sup, mode="final")
    sparse = next(r for r in ds.study_b if r["arm"] == "Sparse")["fewshot"][45.0]
    assert sparse["adapt_steps"] == max(R.FEWSHOT_HORIZONS) + 1 and sparse["censored"] is True
    assert sparse["index"] == len(R.FEWSHOT_HORIZONS) + 1 and sparse["consistent"] is True
    assert sparse["vm"] is not None and sparse["horizons"] == list(R.FEWSHOT_HORIZONS)
    moderate = next(r for r in ds.study_b if r["arm"] == "Moderate")["fewshot"][15.0]
    assert moderate["censored"] is False and moderate["index"] == 2
    assert not ds.problems


def test_load_dataset_reads_a_parquet_ledger_and_its_default_supplement(tmp_path) -> None:
    rows = _main_rows()[:13]
    ledger = syn.write_ledger(tmp_path / "ledger.parquet", rows)
    syn.write_supplement(tmp_path / "supplement", "measurement", syn.measurement_record(rows[0]))
    ds = data.load_dataset(ledger, mode="final")
    assert len(ds.study_a) == 13 and ds.supplement.files == [f"measurement/{rows[0]['run_id']}.json"]
    assert data.default_supplement_dir(ledger) == tmp_path / "supplement"
    assert data.finite_or_none(float("nan")) is None and data.finite_or_none(2) == 2.0


def test_fewshot_records_count_before_the_ledger_is_enriched() -> None:
    rows, records = syn.b_world()
    for row in rows:
        row["sr_fewshot"] = row["adapt_steps"] = None  # the enrichment has not written the maps yet
    sup = data.Supplement()
    for run_id, recs in records.items():
        for f in recs["fewshot"]:
            model = S.validate_record("fewshot", f)
            sup.records.setdefault("fewshot", {})[(run_id, S.record_part("fewshot", model))] = model
    ds = data.build_dataset(rows, sup, mode="final")
    dense = next(r for r in ds.study_b if r["arm"] == "Dense")["fewshot"]
    assert set(dense) == set(R.UNSEEN_BUDGETS) and dense[5.0]["index"] == 1 and dense[5.0]["consistent"] is True


def test_the_recorded_cuts_are_read_and_checked(tmp_path) -> None:
    assert data.CUT_DROP_N010 in manifest.CUTS
    assert data.load_cuts(tmp_path / "absent.json") == ()
    record = tmp_path / "cuts.json"
    record.write_text(json.dumps({"cuts": list(manifest.CUTS[:2]), "amendment": "A3"}))
    assert data.load_cuts(record) == manifest.CUTS[:2]
    record.write_text(json.dumps({"cuts": [manifest.CUTS[1]], "amendment": "A3"}))  # not a prefix of the order
    with pytest.raises(data.DataError, match="prefix"):
        data.load_cuts(record)
    record.write_text("[]")
    with pytest.raises(data.DataError, match="cut record"):
        data.load_cuts(record)
    # checked as pilot.scheduler.committed_cuts checks it: a record the scheduler refuses never cuts an arm here
    for bad, message in (({"cuts": ["pid"]}, "names no amendment"),
                         ({"cuts": ["pid"], "amendment": " "}, "names no amendment"),
                         ({"cuts": "pid", "amendment": "A3"}, "cut record"),
                         ({"cuts": [1], "amendment": "A3"}, "cut record")):
        record.write_text(json.dumps(bad))
        with pytest.raises(data.DataError, match=message):
            data.load_cuts(record)
    ledger = syn.write_ledger(tmp_path / "ledger.parquet", _main_rows()[:13])
    record.write_text(json.dumps({"cuts": list(manifest.CUTS), "amendment": "A3"}))
    assert data.load_dataset(ledger, mode="final", cuts_file=record).cuts == manifest.CUTS


def test_the_cut_record_is_the_schedulers() -> None:
    """data.CUTS_FILE is the file ``schedule add --cuts K`` reads (pilot.scheduler.CUTS_PATH), and parse_cuts names it."""
    from pilot import scheduler

    assert data.CUTS_FILE == data.REPO_ROOT / scheduler.CUTS_PATH
    with pytest.raises(data.DataError, match=f"^{scheduler.CUTS_PATH} names no amendment"):
        data.parse_cuts(json.dumps({"cuts": []}).encode())


def _b_supplement(records: dict[str, dict]) -> data.Supplement:
    sup = data.Supplement()
    for run_id, recs in records.items():
        for kind, record in [("zero_shot", recs["zero_shot"])] + [("fewshot", f) for f in recs["fewshot"]]:
            model = S.validate_record(kind, record)
            sup.records.setdefault(kind, {})[(run_id, S.record_part(kind, model))] = model
    return sup


def test_a_study_b_record_of_another_checkpoint_or_arm_is_a_problem_and_is_not_read() -> None:
    """Study B runs are evaluated at their final checkpoint (Q-studyb-eval) and their few-shot continuations start
    from it (Q-continuations). A zero-shot record of another step or arm, and a few-shot record whose parent step or
    arm is another, is a data problem and is not read (as the Study A records of another checkpoint), never used."""
    rows, records = syn.b_world()
    dense = next(r for r in rows if r["arm"] == "Dense")
    sparse = next(r for r in rows if r["arm"] == "Sparse")
    final = max(c["step"] for c in dense["checkpoints"])
    earlier = final - R.CHECKPOINT_INTERVAL_STEPS
    records[dense["run_id"]]["zero_shot"]["step"] = earlier
    by_budget = {f["budget"]: f for f in records[dense["run_id"]]["fewshot"]}
    by_budget[5.0]["parent_step"] = earlier
    by_budget[15.0]["arm"] = "Moderate"
    moderate = next(s for s in syn.specs(groups=("study_b",)) if s.arm == "Moderate" and s.seed == sparse["seed"])
    records[sparse["run_id"]]["zero_shot"] = {  # a Moderate run's record filed under the Sparse run
        **syn.zero_shot_record(moderate, {b: 0.5 for b in R.UNSEEN_BUDGETS}), "run_id": sparse["run_id"]}
    ds = data.build_dataset(rows, _b_supplement(records), mode="final")
    rec = {r["run_id"]: r for r in ds.study_b}
    d, s = rec[dense["run_id"]], rec[sparse["run_id"]]
    # not read: the satisfactions at the training levels and the violations come from no record
    assert d["sr_train"] is None and d["vm_zero"] is None and d["zero_episodes"] is None
    assert s["sr_train"] is None and s["vm_zero"] is None  # another arm's training levels are never this run's
    assert d["sr_zero"] == dense["sr_zero"] and s["sr_zero"] == sparse["sr_zero"]  # the ledger's own values stay
    assert d["fewshot"][5.0]["vm"] is None and d["fewshot"][15.0]["vm"] is None and d["fewshot"][30.0]["vm"] is not None
    assert d["fewshot"][5.0]["adapt_steps"] == dense["adapt_steps"][5.0]  # the ledger's value, not the record's
    mine = [p for p in ds.problems if p.startswith(dense["run_id"])]
    for what in ("zero-shot record is at step", "fewshot (budget 5) record is at step",
                 "fewshot (budget 15) record is of arm Moderate"):
        assert any(what in p and "not read" in p for p in mine), (what, mine)
    assert any("zero-shot record is of arm Moderate" in p and "not read" in p
               for p in ds.problems if p.startswith(sparse["run_id"]))
    other = next(r for r in ds.study_b if r["arm"] == "Continuous")
    assert other["sr_train"] is not None and not [p for p in ds.problems if p.startswith(other["run_id"])]


@pytest.mark.parametrize("edit, message", [
    (lambda rows: rows.append(dict(rows[0])), "twice"),
    (lambda rows: rows.append(dict(rows[0], run_id="A-PointGoal1-N0.00-copy-s0")), "more than one N = 0 arm"),
    (lambda rows: rows.append(dict(rows[0], run_id="A-PointGoal1-N0.00-s7")), "does not end with its seed"),
    (lambda rows: rows.append(dict(rows[0], run_id="Z-PointGoal1-N0.00-s0")), "unexpected run_id"),
])
def test_rows_the_matching_cannot_read_are_refused_with_their_run(edit, message) -> None:
    rows = _main_rows()
    edit(rows)
    with pytest.raises(data.DataError, match=message):
        data.build_dataset(rows, data.Supplement(), mode="final")


def test_a_bool_surplus_extra_is_refused_even_after_the_cache_holds_1() -> None:
    rows = _main_rows()
    data.build_dataset(rows, data.Supplement(), mode="final", surplus_extra=1)  # warms the cached targets with 1
    with pytest.raises(data.DataError, match="surplus_extra must be a whole number"):
        data.build_dataset(rows, data.Supplement(), mode="final", surplus_extra=True)


def test_the_five_seed_view_keeps_the_registered_seeds_when_no_role_is_noted() -> None:
    rows = _main_rows()
    extra = [r for r in syn.a_world("supported", groups=("main",), seeds=range(5, 8)) if r["N"] == 0.5]
    b_rows, _ = syn.b_world(seeds=range(8))  # every Study B arm with three surplus seeds (5 to 7), no notes
    ds = data.build_dataset(rows + extra + b_rows, data.Supplement(), mode="final", surplus_extra=3)
    view = data.five_seed_view(ds)
    assert len(ds.study_a) == len(rows) + len(extra) and len(view.study_a) == len(rows)
    assert all(r["seed"] in R.SEEDS for r in view.study_a)
    assert len(ds.study_b) == len(b_rows) and len(view.study_b) == len(R.STUDY_B_ARMS) * len(R.SEEDS)
    assert all(r["seed"] in R.SEEDS for r in view.study_b)
    plain = data.build_dataset(rows, data.Supplement(), mode="final")
    assert data.five_seed_view(plain) is plain  # nothing to restrict: the same object


def test_the_five_seed_view_keeps_a_replacement_and_leaves_surplus_seeds_out() -> None:
    """Seed 2 failed after the surplus seeds 5 to 7 were queued, so its replacement took the
    next unused seed, 8 (Part 5.6). The "five seeds" reading keeps 0, 1, 3, 4 and 8, never surplus seed 5; the
    ledger notes (written from the scheduler's record) tell the replacement from the surplus seeds."""
    arm = "A-PointGoal1-N0.00"
    rows = [r for r in _main_rows() if not (r["arm"] == "N0.00" and r["seed"] == 2)]
    extra = [r for r in syn.a_world("supported", groups=("main",), seeds=range(5, 9)) if r["N"] == 0.0]
    for r in extra:
        r["notes"] = (manifest.replacement_note(f"{arm}-s2") if r["seed"] == 8 else manifest.SURPLUS_NOTE)
    assert manifest.seed_role(extra[-1]) == "replacement" and manifest.seed_role(extra[0]) == "surplus"
    ds = data.build_dataset(rows + extra, data.Supplement(), mode="final", surplus_extra=3)
    view = data.five_seed_view(ds)
    assert sorted(r["seed"] for r in view.study_a if r["arm_id"] == arm) == [0, 1, 3, 4, 8]


def test_an_auxiliary_run_is_not_a_seed_of_its_arm() -> None:
    """The N = 0 run queued for a warm-started replacement's seed (`schedule resolve RUN add-auxiliary`,
    Q-warm-start) carries manifest.AUXILIARY_NOTE in its ledger notes and is left out of its arm."""
    rows = _main_rows()
    ref = next(s for s in syn.specs(groups=("main",)) if s.N == 0.0)
    aux = syn.study_a_row(ref.with_seed(5), cost=25.0, gaps={})
    aux["notes"] = f"x; {manifest.AUXILIARY_NOTE}"
    assert manifest.is_auxiliary_row(aux) and not any(manifest.is_auxiliary_row(r) for r in rows)
    ds = data.build_dataset(rows + [aux], data.Supplement(), mode="final")
    assert ds.auxiliary == [aux["run_id"]] and aux["run_id"] not in {r["run_id"] for r in ds.study_a}
    assert ds.seeds_per_arm()[ref.arm_id] == list(R.SEEDS) and not ds.problems


def test_a_completed_study_a_row_without_n_is_a_refusal_not_a_crash() -> None:
    """The frozen schema accepts N = None; the analysis cannot tell the row's arm, so it is a DataError naming the run
    (``python -m analysis`` refuses it with exit 2), never a TypeError."""
    rows = _main_rows()
    rows[0]["N"] = None
    with pytest.raises(data.DataError, match=f"{rows[0]['run_id']}: a completed Study A row without N"):
        data.build_dataset(rows, data.Supplement(), mode="final")


def test_a_non_finite_selection_cost_does_not_decide_the_window_minimum() -> None:
    """min() with a NaN depends on the order of its arguments; only finite costs enter min_window_selection_cost."""
    row = _main_rows()[-1]
    window = data.matching.selection_window(row["checkpoints"])
    assert len(window) >= 2
    finite = min(float(c["selection_cost"]) for c in window)
    for position in (0, -1):
        checkpoints = [dict(c) for c in row["checkpoints"]]
        target = next(c for c in checkpoints if c["step"] == window[position]["step"])
        expected = min(float(c["selection_cost"]) for c in window if c["step"] != target["step"])
        target["selection_cost"] = float("nan")
        assert data._min_window_cost(checkpoints) == expected
    assert data._min_window_cost([dict(c, selection_cost=float("nan")) for c in row["checkpoints"]]) is None
    assert data._min_window_cost(row["checkpoints"]) == finite

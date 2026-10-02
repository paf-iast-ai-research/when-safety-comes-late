"""The contracts of pilot/contracts.py: evaluator dispatch and contract 1, extra checkpoints and the selection window,
plasticity (contract 2), reserve seeds and the seed registry (Q-mujoco-exception), and few-shot keys."""

from __future__ import annotations

import json
import math
import sys
import warnings
from collections import Counter
from fractions import Fraction
from pathlib import Path

import pytest

from configs import registered as R
from pilot import contracts, manifest
from pilot.contracts import ContractError
from pilot.manifest import RunSpec


def answer_only(monkeypatch, keys) -> None:
    """From here on in the test, ``keys`` are answered and every other PENDING key is open (conftest.py)."""
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset(keys))


def _spec(**changes) -> RunSpec:
    base = dict(run_id="X-A-s0", study="A", task=R.PRIMARY_TASK, arm="N0.25-abrupt-total", seed=0,
                total_steps=R.TOTAL_STEPS, base_algo="PPOLag", plugin="study_a", group="main", N=0.25,
                onset_step=2_500_000, onset_shape="abrupt", step_matching="total_steps")
    base.update(changes)
    return RunSpec(**base)


def _det_spec() -> RunSpec:
    return RunSpec(run_id="DET-PPOLag-PointGoal1-s0", study="A", task=R.PRIMARY_TASK, arm="determinism-check",
                   seed=0, total_steps=R.STEPS_PER_EPOCH, base_algo="PPOLag", plugin="ppolag", group="determinism")


def _matched_row(parent: RunSpec) -> dict:
    return {"run_id": parent.run_id, "matched": True, "completed": True, "matched_checkpoint_step": 9_000_000,
            "matched_checkpoint_path": f"{parent.run_id}/omnisafe/x/torch_save/epoch-450.pt", "commit_hash": "a" * 40}


# ---------------------------------------------------------------------------
# Contract 1: which harness evaluates a run
# ---------------------------------------------------------------------------


def test_the_evaluator_is_chosen_by_plug_in() -> None:
    by_plugin = {s.plugin: contracts.evaluator_target(s) for s in manifest.design("pilot") + manifest.design("all")}
    assert by_plugin["study_b"] == "studyb.evaluation:evaluate_run"
    for plugin in ("study_a", "study_a_pid", "unconstrained_ppo"):
        assert by_plugin[plugin] == "envs.evaluation:evaluate_run"
    # the spec's dict form (what the harness and the scheduler's JSON hold) gives the same answer
    moderate = next(s for s in manifest.pilot() if s.arm == "Moderate")
    assert contracts.evaluator_target(moderate.to_dict()) == contracts.evaluator_target(moderate)
    assert contracts.evaluator_target(_det_spec()) == contracts.EVALUATOR_TARGET


def test_a_missing_harness_is_unavailable_and_names_its_owner(monkeypatch) -> None:
    from pilot.algorithms import PluginUnavailableError

    moderate = next(s for s in manifest.pilot() if s.arm == "Moderate")
    monkeypatch.setattr(contracts, "STUDY_B_EVALUATOR_TARGET", "studyb.no_such_module_wpcore:evaluate_run")
    with pytest.raises(PluginUnavailableError, match=r"Study B and literature \(Role 5\)"):
        contracts.load_evaluator(moderate)
    monkeypatch.setattr(contracts, "EVALUATOR_TARGET", "envs.no_such_module_wpcore:evaluate_run")
    with pytest.raises(PluginUnavailableError, match=r"Environment and tests \(Role 2\)"):
        contracts.load_evaluator(_det_spec())


def test_a_harness_whose_own_import_fails_shows_the_real_error(tmp_path, monkeypatch) -> None:
    (tmp_path / "wpcore_fake_harness.py").write_text("import wpcore_library_that_is_not_installed\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(contracts, "EVALUATOR_TARGET", "wpcore_fake_harness:evaluate_run")
    with pytest.raises(ModuleNotFoundError) as exc:
        contracts.load_evaluator(_det_spec())
    assert exc.value.name == "wpcore_library_that_is_not_installed"
    sys.modules.pop("wpcore_fake_harness", None)


# ---------------------------------------------------------------------------
# Extra checkpoints and the selection window
# ---------------------------------------------------------------------------


def test_extra_checkpoint_steps_of_each_kind_of_run() -> None:
    main = {(s.task, s.arm, s.seed): s for s in manifest.design("main")}
    late = main[(R.PRIMARY_TASK, "N0.25-abrupt-total", 0)]
    assert contracts.extra_checkpoint_steps(late) == [2_500_000, 2_700_000]  # onset and onset + 200,000 (H3 (c))
    assert contracts.extra_checkpoint_steps(main[(R.PRIMARY_TASK, "N0.10-abrupt-total", 0)]) == []  # both on the grid
    assert contracts.extra_checkpoint_steps(main[(R.PRIMARY_TASK, "N0.50-abrupt-total", 0)]) == []
    # an off-grid total also saves the end-relative selection window (Q-selection-window's answer, Table 9.1)
    assert contracts.extra_checkpoint_steps(main[(R.PRIMARY_TASK, "N0.10-abrupt-constrained", 0)]) == [
        1_120_000, 1_320_000, *range(9_320_000, 11_000_000, 200_000)]
    assert contracts.extra_checkpoint_steps(main[(R.PRIMARY_TASK, "N0.00", 0)]) == []
    pid = next(s for s in manifest.design("pid") if s.N == 0.25)
    assert contracts.extra_checkpoint_steps(pid) == [2_500_000, 2_700_000]
    fewshot = manifest.design("study_b_fewshot")[0]
    assert contracts.extra_checkpoint_steps(fewshot) == [500_000]  # Table 2.5: the middle horizon is off the grid
    short = manifest.apply_cuts([fewshot], manifest.CUTS[:3])[0]  # cut 3: first horizon only
    assert contracts.extra_checkpoint_steps(short) == []
    # the determinism check has no manipulation step, but an onset (the launcher's slow test) is saved
    assert contracts.extra_checkpoint_steps(RunSpec.from_dict({**_det_spec().to_dict(), "total_steps": 40_000,
                                                               "onset_step": 20_000})) == [20_000]
    assert all(contracts.extra_checkpoint_steps(s) == [] for s in manifest.design("study_b"))
    # no continuation saves the end-relative window: rule 1 does not select from it
    parent = next(s for s in manifest.design("main") if s.N == 0.50)
    battery = manifest.battery_continuation(parent, _matched_row(parent), manifest.BATTERY_CONDITIONS[0])
    assert contracts.extra_checkpoint_steps({**battery.to_dict(), "total_steps": 11_120_000}) == []


def test_expected_checkpoints_hold_the_grid_the_end_and_the_extra_steps() -> None:
    fewshot = manifest.design("study_b_fewshot")[0]
    assert contracts.expected_checkpoint_steps(fewshot) == [0, 200_000, 400_000, 500_000, 600_000, 800_000, 1_000_000]
    rounded = next(s for s in manifest.design("main") if s.total_steps == 11_120_000 and s.onset_shape == "abrupt")
    steps = contracts.expected_checkpoint_steps(rounded)
    assert steps[-2:] == [11_000_000, 11_120_000] and {1_120_000, 1_320_000} <= set(steps)


def test_no_extra_checkpoint_lies_in_the_selection_window_of_any_registered_run(monkeypatch) -> None:
    """Rule 1's last ten checkpoints are the grid's; the onset and the manipulation-check step never
    enter them. For an off-grid total the window is the end-relative grid (Q-selection-window's
    answer, Table 9.1), saved as extra steps of its own.

    Continuations are not selected by rule 1 (they end as ``continued``); the few-shot horizon
    500,000 lies in its own last 2,000,000 steps, which is why they are left out here.
    """
    specs = manifest.design("all") + manifest.design("pilot") + manifest.design("repilot")
    with_extra = Counter()
    answer_only(monkeypatch, R.PENDING)
    for spec in specs:
        if spec.group in manifest.CONTINUATION_GROUPS:
            continue
        extra = contracts.extra_checkpoint_steps(spec)
        window = contracts.selection_window(contracts.expected_checkpoint_steps(spec), spec.total_steps)
        assert window == contracts.end_relative_window(spec.total_steps), spec.run_id
        other = [s for s in extra if s not in window]
        assert all(s <= spec.total_steps - R.SELECTION_WINDOW_STEPS for s in other), spec.run_id
        onset = spec.onset_step or 0
        assert onset not in window and onset + R.MANIPULATION_CHECK_STEPS_AFTER_ONSET not in window, spec.run_id
        assert set(extra) & set(window) == (set(window[:-1]) if spec.total_steps % R.CHECKPOINT_INTERVAL_STEPS else set())
        if other:
            with_extra[spec.group] += 1
    # Onsets off the grid: N = 0.25 total-steps (2,500,000) and the rounded constrained-steps onsets of
    # N = 0.10 and 0.25 (1,120,000; 3,340,000). Main: 3 such arms x 2 shapes x 3 tasks x 5 seeds = 90;
    # treatments and controllers (abrupt, total-steps): N = 0.25 only, 3 x 3 x 5 = 45 and 2 x 3 x 5 = 30;
    # PID: 5. N = 0.10 and 0.50 total-steps onsets and their + 200,000 steps are on the grid.
    assert with_extra == {"main": 90, "treatment": 45, "controller": 30, "pid": 5}


def test_every_study_a_run_is_evaluable_once_its_questions_are_answered(monkeypatch) -> None:
    """Q-selection-window: with every PENDING key answered, the checkpoints each Study A run saves give
    rule 1's window, so no run launched once its questions are answered is refused at evaluation (the
    75 off-grid runs: constrained-steps N = 0.10 and 0.25, the N = 0.25 data control)."""
    off_grid = Counter()
    answer_only(monkeypatch, R.PENDING)
    for spec in manifest.design("study_a"):
        assert spec.open_questions == (), spec.run_id
        window = contracts.selection_window(contracts.expected_checkpoint_steps(spec), spec.total_steps)
        assert len(window) == R.SELECTION_WINDOW_CHECKPOINTS and window[-1] == spec.total_steps, spec.run_id
        assert window[0] == spec.total_steps - (R.SELECTION_WINDOW_CHECKPOINTS - 1) * R.CHECKPOINT_INTERVAL_STEPS
        if spec.total_steps % R.CHECKPOINT_INTERVAL_STEPS:
            off_grid[spec.total_steps] += 1
    assert off_grid == {11_120_000: 30, 13_340_000: 30, 12_500_000: 15}


def test_an_off_grid_window_is_refused_while_the_question_is_open(monkeypatch) -> None:
    from pilot import errors

    spec = next(s for s in manifest.design("main") if s.total_steps == 11_120_000)
    steps = contracts.expected_checkpoint_steps(spec)
    assert contracts.end_relative_window(11_120_000) == list(range(9_320_000, 11_120_001, 200_000))
    answer_only(monkeypatch, ())  # the key open, whatever the repository answers
    assert R.is_open("Q-selection-window")
    with pytest.raises(ContractError, match="Q-selection-window") as info:
        contracts.selection_window(steps, spec.total_steps)
    # a refusal (exit 5), not an evaluation failure (HANDOVER.md section 8); still a ContractError for the harnesses
    assert isinstance(info.value, errors.PendingQuestionError) and isinstance(info.value, errors.RunRefused)
    with errors.allow_pending():  # smoke roots only
        assert contracts.selection_window(steps, spec.total_steps) == contracts.end_relative_window(11_120_000)
    answer_only(monkeypatch, {"Q-selection-window"})
    assert contracts.selection_window(steps, spec.total_steps) == contracts.end_relative_window(11_120_000)
    grid_only = sorted(set(range(0, 11_000_001, 200_000)) | {11_120_000})  # without the end-relative extra checkpoints
    with pytest.raises(ContractError, match=r"lacks the checkpoints at steps \[9320000"):
        contracts.selection_window(grid_only, spec.total_steps)


def test_an_off_grid_total_too_short_for_the_window_is_named(monkeypatch) -> None:
    """A total below 1,800,000 steps cannot hold ten end-relative checkpoints; the error says so
    instead of listing negative steps as missing."""
    steps = sorted(set(range(0, 1_000_001, 200_000)) | set(range(100_000, 1_100_001, 200_000)))
    answer_only(monkeypatch, {"Q-selection-window"})
    with pytest.raises(ContractError, match=r"the total 1,100,000 is shorter than the 10-checkpoint") as info:
        contracts.selection_window(steps, 1_100_000)
    assert "-700000" not in str(info.value)  # no negative step is named as missing


def test_the_selection_window_does_not_depend_on_the_order_of_the_steps() -> None:
    """``steps`` are sorted on entry, so a descending list gives the same window."""
    grid = list(range(0, 2_000_001, 200_000))
    assert contracts.selection_window(grid[::-1]) == grid[-10:]
    assert contracts.selection_window(grid[::-1], 2_000_000) == grid[-10:]


def test_an_on_grid_window_names_the_broken_checkpoint_set(monkeypatch) -> None:
    """For a total on the grid a missing or extra checkpoint is not Q-selection-window, and a checkpoint
    beyond the total is refused on and off the grid."""
    from pilot import errors

    grid = list(range(0, 10_000_001, 200_000))
    with pytest.raises(ContractError, match=r"grid checkpoints at steps \[10000000\] are missing") as info:
        contracts.selection_window(grid[:-1], 10_000_000)
    assert "Q-selection-window" not in str(info.value) and not isinstance(info.value, errors.RunRefused)
    with pytest.raises(ContractError, match=r"steps \[9100000\] are off the grid"):
        contracts.selection_window(sorted(grid + [9_100_000]), 10_000_000)
    with pytest.raises(ContractError, match=r"beyond the run's total 10,000,000 steps at steps \[10200000\]"):
        contracts.selection_window(grid + [10_200_000], 10_000_000)
    off_grid = sorted(set(range(0, 11_000_001, 200_000)) | set(contracts.end_relative_window(11_120_000)))
    answer_only(monkeypatch, {"Q-selection-window"})
    assert contracts.selection_window(off_grid, 11_120_000) == contracts.end_relative_window(11_120_000)
    with pytest.raises(ContractError, match=r"beyond the run's total 11,120,000 steps at steps \[11400000\]"):
        contracts.selection_window(off_grid + [11_400_000], 11_120_000)


def _evaluation_run(tmp_path: Path) -> tuple[RunSpec, Path, dict]:
    """A determinism-check spec over the registered total, its grid checkpoints and a valid contract-1 result."""
    spec = RunSpec.from_dict({**_det_spec().to_dict(), "total_steps": R.TOTAL_STEPS})
    omni = tmp_path / "omni"
    (omni / "torch_save").mkdir(parents=True)
    for step in range(0, R.TOTAL_STEPS + 1, R.CHECKPOINT_INTERVAL_STEPS):
        (omni / "torch_save" / f"epoch-{step // R.STEPS_PER_EPOCH}.pt").write_bytes(b"")
    window = contracts.end_relative_window(R.TOTAL_STEPS)
    good = {"final_cost": 20.0, "final_return": 5.0, "episodes": R.EVAL_EPISODES,
            "selection_seeds": list(range(R.EVAL_EPISODES)), "selection": {s: [24.0, 1.0] for s in window}}
    return spec, omni, good


def test_selection_keys_are_parsed_strictly(tmp_path) -> None:
    """A selection key is a whole step (int, integral float, or a string of digits); 8,200,000.7 is not
    truncated to 8,200,000, and True is not step 1."""
    spec, omni, good = _evaluation_run(tmp_path)
    contracts.validate_evaluation(good, spec, omni)
    contracts.validate_evaluation({**good, "selection": {str(s): v for s, v in good["selection"].items()}}, spec, omni)
    contracts.validate_evaluation({**good, "selection": {float(s): v for s, v in good["selection"].items()}}, spec, omni)
    contracts.validate_evaluation(good, spec.to_dict(), omni)  # the spec's dict form
    first = min(good["selection"])
    rest = {s: v for s, v in good["selection"].items() if s != first}
    for key in (first + 0.7, f"{first}\n", f"{first}.0", True, None):
        with pytest.raises(ContractError, match="selection keys must be checkpoint steps"):
            contracts.validate_evaluation({**good, "selection": {**rest, key: [24.0, 1.0]}}, spec, omni)


@pytest.mark.parametrize("bad", [{"final_cost": "20"}, {"final_return": True}, {"episodes": "100"},
                                 {"final_cost": None}, {"final_cost": math.nan}])
def test_contract_one_values_must_be_finite_numbers(tmp_path, bad) -> None:
    spec, omni, good = _evaluation_run(tmp_path)
    with pytest.raises(ContractError, match="not (a number|finite)"):
        contracts.validate_evaluation({**good, **bad}, spec, omni)


@pytest.mark.parametrize("pair", [[True, False], ["24.0", 1.0], [24.0, "1"]])
def test_selection_costs_and_returns_must_be_numbers(tmp_path, pair) -> None:
    spec, omni, good = _evaluation_run(tmp_path)
    with pytest.raises(ContractError, match="not a number"):
        contracts.validate_evaluation({**good, "selection": {s: pair for s in good["selection"]}}, spec, omni)


def test_checkpoint_steps_use_the_given_epoch_length(tmp_path) -> None:
    (tmp_path / "torch_save").mkdir()
    for k in (0, 1, 3):
        (tmp_path / "torch_save" / f"epoch-{k}.pt").write_bytes(b"")
    (tmp_path / "torch_save" / "epoch-x.pt").write_bytes(b"")
    (tmp_path / "torch_save" / "epoch-\u0663.pt").write_bytes(b"")  # an Arabic-Indic 3: not an ASCII step
    assert contracts.checkpoint_steps(tmp_path) == [0, R.STEPS_PER_EPOCH, 3 * R.STEPS_PER_EPOCH]
    assert contracts.checkpoint_steps(tmp_path, 2_000) == [0, 2_000, 6_000]


# ---------------------------------------------------------------------------
# Contract 2: plasticity
# ---------------------------------------------------------------------------


def test_plasticity_is_required_for_the_study_a_plug_ins_only() -> None:
    specs = manifest.design("all") + manifest.design("pilot")
    required = {s.plugin for s in specs if contracts.plasticity_required(s)}
    assert required == {"study_a", "study_a_pid"}
    assert all(contracts.plasticity_required(s) for s in specs if s.study == "A")  # pilot Study A included
    assert not contracts.plasticity_required(_det_spec())
    parent = next(s for s in manifest.design("main") if s.N == 0.50)
    for condition in manifest.BATTERY_CONDITIONS:
        assert not contracts.plasticity_required(manifest.battery_continuation(parent, _matched_row(parent), condition))
    assert contracts.plasticity_required(parent.to_dict())


def _plasticity_run(tmp_path: Path, epochs: tuple[int, ...], text: str | None, steps_per_epoch: int | None = None) -> Path:
    omni = tmp_path / "omni"
    (omni / "torch_save").mkdir(parents=True)
    for k in epochs:
        (omni / "torch_save" / f"epoch-{k}.pt").write_bytes(b"")
    if text is not None:
        (omni / contracts.PLASTICITY_FILE).write_text(text)
    if steps_per_epoch is not None:
        (omni / "config.json").write_text(json.dumps({"algo_cfgs": {"steps_per_epoch": steps_per_epoch}}))
    return omni


def test_a_complete_plasticity_file_returns_every_column(tmp_path) -> None:
    text = ("step,dormant,rank,norm,norm_reward_critic,dormant_trainable\n"
            "0,0.0,52,10.5,3.0,0.0\n20000,0.01,51,10.6,3.1,nan\n40000,nan,inf,10.7,3.2,0.02\n")
    omni = _plasticity_run(tmp_path, (0, 1, 2), text)
    rows = contracts.validate_plasticity(omni, _spec(), completed=True)
    assert sorted(rows) == [0, 20_000, 40_000]
    assert rows[20_000]["norm_reward_critic"] == 3.1 and math.isnan(rows[20_000]["dormant_trainable"])
    assert math.isnan(rows[40_000]["dormant"]) and rows[40_000]["rank"] == math.inf
    assert set(rows[0]) == {"dormant", "rank", "norm", "norm_reward_critic", "dormant_trainable"}


@pytest.mark.parametrize("text, message", [
    ("dormant,step,rank,norm\n0,0.1,1,1\n", "does not start with"),
    ("step,dormant,rank\n0,0.1,1\n", "does not start with"),
    ("step,dormant,rank,norm,norm\n0,0.1,1,1,1\n", "repeats a column"),
    ("step,dormant,rank,norm\n2500000.7,0.1,1,1\n", "not an integer"),
    ("step,dormant,rank,norm\n2e6,0.1,1,1\n", "not an integer"),
    ("step,dormant,rank,norm\n,0.1,1,1\n", "not an integer"),
    ("step,dormant,rank,norm\n-20000,0.1,1,1\n", "not an integer"),
    ('step,dormant,rank,norm\n"0\n",0.1,1,1\n', "not an integer"),
    ("step,dormant,rank,norm\n0,0.1,1,1\n0,0.1,1,1\n", "unique and ascending"),
    ("step,dormant,rank,norm\n20000,0.1,1,1\n0,0.1,1,1\n", "unique and ascending"),
    ("step,dormant,rank,norm\n0,oops,1,1\n", "not a number"),
    ("step,dormant,rank,norm\n0,,1,1\n", "not a number"),
    ("step,dormant,rank,norm\n0,0.1,1\n", "fields"),
    ("step,dormant,rank,norm\n0,0.1,1,1,9\n", "fields"),
])
def test_a_malformed_plasticity_file_breaks_contract_2(tmp_path, text, message) -> None:
    omni = _plasticity_run(tmp_path, (0,), text)
    for completed in (True, False):  # the format is checked for failed runs too
        with pytest.raises(ContractError, match=message):
            contracts.validate_plasticity(omni, _spec(), completed=completed)


def test_a_completed_study_a_run_needs_a_row_per_saved_checkpoint(tmp_path) -> None:
    omni = _plasticity_run(tmp_path, (0, 1, 2), "step,dormant,rank,norm\n0,0,1,1\n40000,0,1,1\n")
    with pytest.raises(ContractError, match=r"steps \[20000\]"):
        contracts.validate_plasticity(omni, _spec(), completed=True)
    assert sorted(contracts.validate_plasticity(omni, _spec(), completed=False)) == [0, 40_000]  # a failed run
    # extra rows (a step without a checkpoint) are accepted
    omni2 = _plasticity_run(tmp_path / "b", (0,), "step,dormant,rank,norm\n0,0,1,1\n30000,0,1,1\n")
    assert sorted(contracts.validate_plasticity(omni2, _spec(), completed=True)) == [0, 30_000]


@pytest.mark.parametrize("last", ["400000,0.2", "400000,0.2,48,3.4", "4000"])
def test_a_failed_runs_last_line_cut_short_is_dropped(tmp_path, last) -> None:
    """The hook ends every row with a line end: a last line without one is an append the failure cut
    short, possibly mid-number ("3.4" of "3.41"), so it is dropped, not read; the exclusion still
    reaches the ledger (Part 5.6)."""
    text = "step,dormant,rank,norm\n0,0.1,50,3.2\n200000,0.2,48,3.4\n" + last
    omni = _plasticity_run(tmp_path, (0, 10, 20), text)
    with pytest.warns(RuntimeWarning, match="cut short"):
        rows = contracts.validate_plasticity(omni, _spec(), completed=False)
    assert sorted(rows) == [0, 200_000] and rows[200_000]["norm"] == 3.4
    # a completed run gets no such allowance: its every append finished
    if last != "400000,0.2,48,3.4":
        with pytest.raises(ContractError):
            contracts.validate_plasticity(omni, _spec(), completed=True)
    else:  # complete but unterminated: a completed run reads the line, without a warning
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert sorted(contracts.validate_plasticity(omni, _spec(), completed=True)) == [0, 200_000, 400_000]


def test_only_the_last_line_of_a_failed_run_may_be_cut_short(tmp_path) -> None:
    # a malformed complete line is a breach whatever follows it
    omni = _plasticity_run(tmp_path, (0,), "step,dormant,rank,norm\n0,0.1,50\n200000,0.2,48,3.4\n400000,0.2")
    with pytest.warns(RuntimeWarning), pytest.raises(ContractError, match="line 2 has 3 fields"):
        contracts.validate_plasticity(omni, _spec(), completed=False)
    # the header itself cut short: the run failed before its first row
    omni = _plasticity_run(tmp_path / "b", (0,), "step,dorm")
    with pytest.warns(RuntimeWarning, match="cut short"):
        assert contracts.validate_plasticity(omni, _spec(), completed=False) == {}
    with pytest.raises(ContractError, match="does not start with"):
        contracts.validate_plasticity(omni, _spec(), completed=True)
    # a terminated file is read as before, without a warning
    omni = _plasticity_run(tmp_path / "c", (0,), "step,dormant,rank,norm\n0,0.1,50,3.2\n")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert sorted(contracts.validate_plasticity(omni, _spec(), completed=False)) == [0]


def test_checkpoint_rows_use_the_runs_own_epoch_length(tmp_path) -> None:
    text = "step,dormant,rank,norm\n0,0,1,1\n2000,0,1,1\n4000,0,1,1\n"
    omni = _plasticity_run(tmp_path, (0, 1, 2), text, steps_per_epoch=2_000)  # a tiny test run
    assert sorted(contracts.validate_plasticity(omni, _spec(), completed=True)) == [0, 2_000, 4_000]
    omni_registered = _plasticity_run(tmp_path / "b", (0, 1, 2), text)  # no config.json: the registered epoch
    with pytest.raises(ContractError, match=r"\[20000, 40000\]"):
        contracts.validate_plasticity(omni_registered, _spec(), completed=True)


def test_a_missing_plasticity_file(tmp_path) -> None:
    omni = _plasticity_run(tmp_path, (0, 1), None)
    with pytest.raises(ContractError, match="lacks plasticity.csv"):
        contracts.validate_plasticity(omni, _spec(), completed=True)
    assert contracts.validate_plasticity(omni, _spec(), completed=False) == {}  # failed before the hook existed
    # an empty file: the hook created it (open(path, "x")) and the run was killed before the header
    empty = _plasticity_run(tmp_path / "empty", (0, 1), "")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert contracts.validate_plasticity(empty, _spec(), completed=False) == {}  # its exclusion still reaches the ledger
    with pytest.raises(ContractError, match="does not start with"):
        contracts.validate_plasticity(empty, _spec(), completed=True)  # a completed run wrote every row
    assert contracts.validate_plasticity(omni, _det_spec(), completed=True) == {}  # not a Study A plug-in
    moderate = next(s for s in manifest.pilot() if s.arm == "Moderate")
    assert contracts.validate_plasticity(omni, moderate, completed=True) == {}


# ---------------------------------------------------------------------------
# Q-mujoco-exception: reserve seeds of unstable evaluation episodes (Table 9.1)
# ---------------------------------------------------------------------------

SELECTION = list(range(1_000_000, 1_000_100))
MEASUREMENT = list(range(2_000_000, 2_000_100))


def _replacement(slot: int, seed: int, reserve: int, step: int = 7) -> dict:
    return {"slot_index": slot, "seed": seed, "reserve_seed": reserve, "step": step, "warning": "mjWARN_BADQVEL x1"}


def test_reserve_seeds_are_the_sets_base_plus_50000_plus_r() -> None:
    assert contracts.RESERVE_SEED_OFFSET == 50_000 and contracts.MAX_UNSTABLE_EPISODES == 5
    assert contracts.reserve_seeds(SELECTION) == list(range(1_050_000, 1_050_005))
    assert contracts.reserve_seeds(list(reversed(MEASUREMENT))) == list(range(2_050_000, 2_050_005))
    assert contracts.reserve_seeds([]) == []


def test_unstable_replacements_follow_the_rule() -> None:
    """Planned seed first, the next unused reserve seed each time (so in the set's range and distinct), a reserve
    episode that is itself unstable replaced the same way, at most five per evaluation call."""
    chain = [_replacement(3, 1_000_003, 1_050_000), _replacement(3, 1_050_000, 1_050_001, step=0),
             _replacement(40, 1_000_040, 1_050_002)]
    assert contracts.validate_unstable_replacements("sel", chain, SELECTION) == chain
    assert contracts.validate_unstable_replacements("sel", None, SELECTION) == []
    assert contracts.validate_unstable_replacements("sel", [], SELECTION) == []
    five = [_replacement(i, 2_000_000 + i, 2_050_000 + i) for i in range(5)]
    assert len(contracts.validate_unstable_replacements("meas", five, MEASUREMENT)) == 5
    bad = {
        "at most 5 per evaluation call": [*five, _replacement(5, 2_000_005, 2_050_005)],
        "not the next unused seed": [_replacement(0, 2_000_000, 2_050_001)],  # r skipped
        "replaces seed": [_replacement(0, 2_000_001, 2_050_000)],  # not the slot's planned seed
        "after one of episode": [_replacement(9, 2_000_009, 2_050_000), _replacement(2, 2_000_002, 2_050_001)],
        "names episode 100": [_replacement(100, 2_000_000, 2_050_000)],
        "must hold exactly": [{**_replacement(0, 2_000_000, 2_050_000), "extra": 1}],
        "warning": [{**_replacement(0, 2_000_000, 2_050_000), "warning": ""}],
        "must be a list": "2050000",
    }
    for message, replacements in bad.items():
        with pytest.raises(ContractError, match=message):
            contracts.validate_unstable_replacements("meas", replacements, MEASUREMENT)
    with pytest.raises(ContractError, match="not the next unused seed"):  # a selection reserve on the measurement set
        contracts.validate_unstable_replacements("meas", [_replacement(0, 2_000_000, 1_050_000)], MEASUREMENT)
    with pytest.raises(ContractError, match="not the next unused seed"):  # seeds without a reserve sequence
        contracts.validate_unstable_replacements("x", [_replacement(0, 7, 50_007)], [7, 8], reserve=[])


def test_the_seed_registry_keeps_the_reserve_sequences_off_every_set(tmp_path) -> None:
    """check_canonical_seeds refuses a set that would run into a reserve sequence."""
    registry = tmp_path / "ledger.seeds.json"
    contracts.check_canonical_seeds("selection", SELECTION, registry)
    contracts.check_canonical_seeds("measurement", MEASUREMENT, registry)  # the registered sets: disjoint
    clash = tmp_path / "clash.seeds.json"
    contracts.check_canonical_seeds("selection", SELECTION, clash)
    with pytest.raises(ContractError, match=r"reserve seeds \[1050000.*selection set"):
        contracts.check_canonical_seeds("measurement", list(range(1_050_000, 1_050_100)), clash, write=False)
    with pytest.raises(ContractError, match=r"reserve seeds \[1000000.*measurement set"):
        contracts.check_canonical_seeds("measurement", list(range(950_000, 950_100)), clash, write=False)


def test_the_seed_registry_is_fsynced_around_its_rename(tmp_path, monkeypatch) -> None:
    """The registry is written as the ledger is: the file fsynced, renamed into place, then its directory fsynced,
    so a crash never leaves an empty or partial registry that every later ledger write would fail to read."""
    import os
    import stat

    events: list[str] = []
    real_fsync, real_replace = os.fsync, os.replace

    def fsync(fd: int) -> None:
        events.append("fsync dir" if stat.S_ISDIR(os.fstat(fd).st_mode) else "fsync file")
        real_fsync(fd)

    def replace(src, dst) -> None:
        events.append("replace")
        real_replace(src, dst)

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(os, "replace", replace)
    registry = tmp_path / "ledger.seeds.json"
    contracts.check_canonical_seeds("measurement", MEASUREMENT, registry, write=False)
    assert events == [] and not registry.exists()  # a check only
    contracts.check_canonical_seeds("selection", SELECTION, registry)
    assert events == ["fsync file", "replace", "fsync dir"]
    assert json.loads(registry.read_text()) == {"selection": SELECTION}
    assert not registry.with_name(registry.name + ".tmp").exists()


def test_final_replacements_are_checked_against_the_measurement_set(tmp_path) -> None:
    """Q-final-cost-set (Table 9.1): the final checkpoint runs the measurement set, so its replacements are checked
    against ``measurement_seeds``, whatever ``final_seed_set`` says."""
    spec, omni, good = _evaluation_run(tmp_path)
    good = {**good, "selection_seeds": SELECTION, "measurement_seeds": MEASUREMENT, "final_seed_set": "measurement"}
    contracts.validate_evaluation(
        {**good, "unstable_replacements": {"final": [_replacement(0, MEASUREMENT[0], 2_050_000)]}}, spec, omni)
    on_selection = {"final": [_replacement(0, SELECTION[0], 1_050_000)]}  # a selection-set replacement
    for final_set in ("measurement", "selection"):
        with pytest.raises(ContractError, match="final checkpoint: replacement 0 replaces seed 1000000"):
            contracts.validate_evaluation({**good, "final_seed_set": final_set, "unstable_replacements": on_selection},
                                          spec, omni)


# ---------------------------------------------------------------------------
# Few-shot keys
# ---------------------------------------------------------------------------


def test_fewshot_keys_have_one_spelling() -> None:
    assert contracts.fewshot_key(5, 200_000) == "5.0_200000"
    assert contracts.fewshot_key(17.5, 500_000.0) == "17.5_500000"
    assert contracts.parse_fewshot_key("5.0_200000") == (5.0, 200_000)
    assert contracts.fewshot_key(-0.0, 200_000) == contracts.fewshot_key(0.0, 200_000) == "0.0_200000"  # one key
    for budget in R.UNSEEN_BUDGETS:
        for horizon in R.FEWSHOT_HORIZONS:
            key = contracts.fewshot_key(budget, horizon)
            assert contracts.parse_fewshot_key(key) == (float(budget), horizon)
            assert contracts.fewshot_key(*contracts.parse_fewshot_key(key)) == key


@pytest.mark.parametrize("key", ["5_200000", "5.0-200000", "5.0_200000.0", "5.0_2e5", "x_200000", "5.0_", "_200000",
                                 "5.0_200000_1", "nan_200000", "inf_200000", 5.0, None, "5.0__200000",
                                 "-0.0_200000", "5.0_200000\n"])
def test_non_canonical_fewshot_keys_are_refused(key) -> None:
    with pytest.raises(ValueError):
        contracts.parse_fewshot_key(key)


@pytest.mark.parametrize("budget, horizon", [(True, 200_000), (float("nan"), 200_000), (float("inf"), 1), ("5", 200_000),
                                             (5.0, 0), (5.0, -200_000), (5.0, 200_000.5), (5.0, True), (5.0, "200000"),
                                             (10**400, 200_000), (5.0, Fraction(2**54 + 1, 2))])
def test_fewshot_key_refuses_bad_inputs(budget, horizon) -> None:
    with pytest.raises(ValueError):
        contracts.fewshot_key(budget, horizon)


def test_fewshot_keys_pass_the_frozen_schema_validator() -> None:
    pytest.importorskip("pydantic")
    from datetime import datetime, timezone

    from results.ledger_schema import LedgerRow

    rates = {contracts.fewshot_key(b, h): 0.5 for b in R.UNSEEN_BUDGETS for h in R.FEWSHOT_HORIZONS}
    row = LedgerRow(
        run_id="B-Moderate-s0", study="B", task=R.PRIMARY_TASK, arm="Moderate", training_levels=[10.0, 20.0, 40.0],
        seed=0, commit_hash="a" * 40, config_hash="c" * 64,
        started=datetime(2026, 9, 30, tzinfo=timezone.utc), finished=datetime(2026, 9, 30, 1, tzinfo=timezone.utc),
        wall_clock_hours=1.0, machine="m", completed=True, sr_fewshot=rates,
    )
    assert row.sr_fewshot == rates and len(rates) == len(R.UNSEEN_BUDGETS) * len(R.FEWSHOT_HORIZONS)
    assert all(contracts.parse_fewshot_key(k)[1] in R.FEWSHOT_HORIZONS for k in row.sr_fewshot)

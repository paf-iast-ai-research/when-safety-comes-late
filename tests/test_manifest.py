"""The registered design as runs (Part 3.2, Tables 3.2 to 3.4, Part 3.6)."""

from __future__ import annotations

from collections import Counter

import pytest

from configs import registered as R
from pilot import manifest
from pilot.manifest import RunSpec, onset_schedule


def test_counts_match_the_registration() -> None:
    assert len(manifest.design("main")) == R.MAIN_SWEEP_RUNS == 195
    assert len(manifest.design("treatment")) == R.TREATMENT_RUNS == 135
    assert len(manifest.design("controller")) == R.CONTROLLER_RUNS == 90
    assert len(manifest.design("pid")) == R.PID_RUNS == 15
    assert len(manifest.design("study_a")) == R.G2_RUN_EQUIVALENTS_STUDY_A == 435
    assert len(manifest.design("study_b")) == R.STUDY_B_RUNS == 35
    assert len(manifest.design("study_b_fewshot")) == R.STUDY_B_CONTINUATIONS == 140
    assert len(manifest.design("pilot")) == R.PILOT_RUNS == 8


def test_thirteen_main_arms_per_task_and_five_seeds_each() -> None:
    specs = manifest.design("main")
    arms_per_task = Counter((s.task, s.arm) for s in specs)
    for task in R.TASKS_STUDY_A:
        assert len({arm for t, arm in arms_per_task if t == task}) == R.MAIN_ARMS_PER_TASK
    assert set(arms_per_task.values()) == {len(R.SEEDS)}


def test_run_ids_are_unique_across_the_whole_design_and_the_pilot() -> None:
    ids = [s.run_id for s in manifest.design("all") + manifest.design("pilot")]
    assert len(ids) == len(set(ids))


def test_study_b_run_equivalents_are_49() -> None:
    specs = manifest.design("study_b") + manifest.design("study_b_fewshot")
    assert sum(s.run_equivalents for s in specs) == pytest.approx(R.STUDY_B_RUN_EQUIVALENTS)


def test_total_steps_matched_onsets_are_whole_epochs() -> None:
    assert onset_schedule(0.10, "total_steps") == (1_000_000, 10_000_000, ())
    assert onset_schedule(0.25, "total_steps") == (2_500_000, 10_000_000, ())
    assert onset_schedule(0.50, "total_steps") == (5_000_000, 10_000_000, ())


def test_constrained_steps_matched_onsets_follow_the_proposed_rounding_and_wait_for_it(monkeypatch) -> None:
    # First Tasks, Part I section 5: 11,120,000 and 13,340,000 steps; N = 0.50 needs no rounding.
    assert onset_schedule(0.10, "constrained_steps") == (1_120_000, 11_120_000, ("Q-rounding",))
    assert onset_schedule(0.25, "constrained_steps") == (3_340_000, 13_340_000, ("Q-rounding",))
    assert onset_schedule(0.50, "constrained_steps") == (10_000_000, 20_000_000, ())
    spec = next(s for s in manifest.design("main") if s.step_matching == "constrained_steps" and s.N == 0.10)
    assert spec.open_questions == ("Q-rounding", "Q-selection-window") and not spec.launchable
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset({"Q-rounding", "Q-selection-window"}))
    assert spec.launchable  # the stored spec is unchanged; answering the questions releases it


def test_every_run_has_exactly_T_constrained_or_total_steps() -> None:
    for s in manifest.design("main"):
        if s.N == 0:
            assert (s.onset_step, s.total_steps) == (0, R.TOTAL_STEPS)
        elif s.step_matching == "total_steps":
            assert s.total_steps == R.TOTAL_STEPS
        else:
            assert s.total_steps - s.onset_step == R.TOTAL_STEPS


def test_pilot_composition() -> None:
    specs = manifest.pilot()
    a = [s for s in specs if s.study == "A"]
    assert sorted((s.N, s.seed) for s in a) == [(0.0, 0), (0.0, 1), (0.0, 2), (0.5, 0), (0.5, 1), (0.5, 2)]
    assert all(s.task == R.PILOT_STUDY_A_TASK for s in a)
    assert all(s.onset_shape == "abrupt" and s.step_matching == "total_steps" for s in a if s.N)
    assert all(s.pilot and s.run_id.startswith("P-") for s in specs)
    b = {s.arm: s for s in specs if s.study == "B"}
    assert set(b) == {"unconstrained", "Moderate"}
    assert b["unconstrained"].base_algo == "PPO" and b["unconstrained"].pending == ("Q-pilot-unconstrained-seed",)
    assert b["Moderate"].training_levels == (10.0, 20.0, 40.0) and b["Moderate"].seed == 0


def test_warm_started_runs_depend_on_the_n0_run_of_the_same_task_and_seed() -> None:
    specs = {s.run_id: s for s in manifest.design("main")}
    for s in manifest.design("controller"):
        if s.controller_variant == "warm_started":
            (dep,) = s.depends_on
            assert dep in specs and specs[dep].N == 0 and specs[dep].seed == s.seed and specs[dep].task == s.task


def test_additional_constrained_training_trains_N_T_more_and_waits_on_its_question() -> None:
    for s in manifest.design("treatment"):
        if s.treatment == "additional_constrained":
            assert s.total_steps == R.TOTAL_STEPS + round(s.N * R.TOTAL_STEPS)
            assert "Q-data-control-lr" in s.pending


def test_pid_arms_use_cppopid_on_the_primary_task_only() -> None:
    specs = manifest.design("pid")
    assert {s.base_algo for s in specs} == {"CPPOPID"} and {s.task for s in specs} == {R.PRIMARY_TASK}


def test_ramp_arms_carry_d_loose_and_window() -> None:
    for s in manifest.design("main"):
        if s.onset_shape == "ramp":
            assert s.params["d_loose"] == R.D_LOOSE[s.task]
            assert s.params["ramp_window_steps"] == R.RAMP_WINDOW_STEPS


def test_fewshot_continuations_depend_on_their_parent() -> None:
    for s in manifest.design("study_b_fewshot"):
        assert s.depends_on == (s.params["parent_run_id"],) and s.pending == ("Q-continuations",)
        assert s.total_steps == R.FEWSHOT_STEPS and s.params["budget"] in R.UNSEEN_BUDGETS
    moved = manifest.design("study_b_fewshot")[0].with_seed(5)
    assert moved.depends_on == (moved.params["parent_run_id"],) and moved.depends_on[0].endswith("-s5")


def test_json_round_trip_and_reseeding() -> None:
    for s in manifest.design("controller")[:6] + manifest.pilot():
        assert RunSpec.from_json(s.to_json()) == s
    warm = next(s for s in manifest.design("controller") if s.controller_variant == "warm_started")
    moved = warm.with_seed(7)
    assert moved.run_id == f"{warm.arm_id}-s7" and moved.depends_on[0].endswith("-s7")
    assert moved.arm_id == warm.arm_id


def test_invalid_specs_are_rejected() -> None:
    good = manifest.pilot()[0].to_dict()
    with pytest.raises(ValueError, match="multiple"):
        RunSpec.from_dict({**good, "total_steps": 10_000})
    with pytest.raises(ValueError, match="must end with"):
        RunSpec.from_dict({**good, "run_id": "no-seed-suffix"})
    with pytest.raises(ValueError, match="plugin"):
        RunSpec.from_dict({**good, "plugin": "nope"})
    with pytest.raises(ValueError, match="whole epoch"):
        RunSpec.from_dict({**good, "onset_step": 5})


def test_the_repilot_is_the_pilot_under_its_own_run_ids() -> None:
    first, again = manifest.design("pilot"), manifest.design("repilot")
    assert len(again) == len(first) == 8
    assert [s.run_id for s in again] == ["P1-" + s.run_id[len("P-"):] for s in first]
    assert all(s.pilot and s.params["pilot_revision"] == 1 for s in again)
    assert [(s.arm, s.seed, s.total_steps, s.onset_step, s.pending) for s in again] == [
        (s.arm, s.seed, s.total_steps, s.onset_step, s.pending) for s in first
    ]
    assert not {s.run_id for s in again} & {s.run_id for s in manifest.design("all") + first}
    with pytest.raises(ValueError):
        manifest.pilot(revision=2)


def test_study_b_specs_carry_their_levels_and_conditioning() -> None:
    specs = manifest.design("study_b")
    for s in specs:
        assert s.training_levels == R.STUDY_B_ARMS[s.arm] and s.task == R.TASKS_STUDY_B[0]
        assert s.params["budget_observation_divisor"] == R.BUDGET_OBSERVATION_DIVISOR
        if s.arm == "Continuous":
            assert s.params["continuous_range"] == list(R.CONTINUOUS_RANGE)
            assert s.params["continuous_bin_width"] == R.CONTINUOUS_BIN_WIDTH
        else:
            assert "continuous_range" not in s.params


def test_rate_limited_arms_carry_the_registered_limits() -> None:
    rate = [s for s in manifest.design("controller") if s.controller_variant == "rate_limited"]
    assert len(rate) == 45
    for s in rate:
        assert s.params["rate_limit_relative"] == R.RATE_LIMIT_RELATIVE
        assert s.params["rate_limit_absolute"] == R.RATE_LIMIT_ABSOLUTE


def test_selection_window_question_only_on_totals_off_the_checkpoint_grid() -> None:
    for s in manifest.design("all"):
        off_grid = s.total_steps % R.CHECKPOINT_INTERVAL_STEPS != 0
        assert ("Q-selection-window" in s.pending) == off_grid, s.run_id
    data = {s.N: s.total_steps for s in manifest.design("treatment") if s.treatment == "additional_constrained"}
    assert data == {0.10: 11_000_000, 0.25: 12_500_000, 0.50: 15_000_000}


def test_surplus_seeds_of_arms_part_5_5_does_not_name_wait_on_their_question(monkeypatch) -> None:
    main = manifest.design("main")
    primary = [s for s in main if manifest.is_primary_comparison(s) and s.seed == 0]
    assert sorted(s.arm for s in primary) == [
        "N0.00", "N0.50-abrupt-constrained", "N0.50-abrupt-total", "N0.50-ramp-constrained", "N0.50-ramp-total",
    ]
    assert all(s.task == R.PRIMARY_TASK for s in primary)
    b = manifest.design("study_b")
    assert all(manifest.is_primary_comparison(s) for s in b)
    for base in primary + [s for s in b if s.seed == 0]:
        extra = manifest.surplus_spec(base, 5)
        assert extra.run_id == f"{base.arm_id}-s5"
        if base.onset_shape == "ramp":
            assert "Q-surplus-arm-set" in extra.open_questions and not extra.launchable
        else:
            assert "Q-surplus-arm-set" not in extra.pending and extra.launchable
    ramp = next(s for s in primary if s.onset_shape == "ramp")
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset({"Q-surplus-arm-set"}))
    assert manifest.surplus_spec(ramp, 6).launchable
    other = next(s for s in main if s.N == 0.10)
    with pytest.raises(ValueError, match="primary-comparison"):
        manifest.surplus_spec(other, 5)

"""The registered design as runs (Part 3.2, Tables 3.2 to 3.4, Part 3.6), its run gates (manifest.run_gate_keys),
the battery continuations (Table 2.2) and the cut order (Part 6.1), and the G2 input checks of pilot.budget."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from configs import registered as R
from pilot import budget, manifest
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


def test_constrained_steps_matched_onsets_follow_q_rounding_and_carry_its_key(monkeypatch) -> None:
    # First Tasks, section 5: 11,120,000 and 13,340,000 steps; N = 0.50 needs no rounding.
    assert onset_schedule(0.10, "constrained_steps") == (1_120_000, 11_120_000, ("Q-rounding",))
    assert onset_schedule(0.25, "constrained_steps") == (3_340_000, 13_340_000, ("Q-rounding",))
    assert onset_schedule(0.50, "constrained_steps") == (10_000_000, 20_000_000, ())
    spec = next(s for s in manifest.design("main") if s.step_matching == "constrained_steps" and s.N == 0.10
                and s.onset_shape == "abrupt")
    # the keys of its own arithmetic first, then the run gates (manifest.run_gate_keys)
    gates = ("Q-cost-critic", "Q-jc-window", "Q-plasticity-definitions")
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset())  # every key open
    assert spec.open_questions == ("Q-rounding", "Q-selection-window", *gates) and not spec.launchable
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset({"Q-rounding", "Q-selection-window"}))
    assert spec.open_questions == gates and not spec.launchable
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset({"Q-rounding", "Q-selection-window", *gates}))
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
    assert b["unconstrained"].base_algo == "PPO"
    assert b["unconstrained"].pending == ("Q-pilot-unconstrained-seed", "Q-search-before-pilot")
    assert b["Moderate"].training_levels == (10.0, 20.0, 40.0) and b["Moderate"].seed == 0


def test_the_pilot_unconstrained_seed_is_the_answered_constant_derived_from_the_moderate_seed() -> None:
    """Table 9.1 answers Q-pilot-unconstrained-seed with 'seed 0, the seed of the Moderate pilot run': a named
    constant marked with its key and derived from R.PILOT_MODERATE_SEED (HANDOVER.md section 10), not a retyped
    literal."""
    source = Path(manifest.__file__).read_text(encoding="utf-8")
    assert ("PILOT_UNCONSTRAINED_SEED = R.PILOT_MODERATE_SEED  # answered in Table 9.1 (Q-pilot-unconstrained-seed)"
            in source)
    assert manifest.PILOT_UNCONSTRAINED_SEED == R.PILOT_MODERATE_SEED == 0
    for revision in (0, 1):
        (run,) = [s for s in manifest.pilot(revision) if s.arm == "unconstrained"]
        assert run.seed == manifest.PILOT_UNCONSTRAINED_SEED
        assert run.run_id == f"{manifest.pilot_prefix(revision)}B-unconstrained-s{manifest.PILOT_UNCONSTRAINED_SEED}"
    assert 'seed=0,' not in source.split("def pilot(", 1)[1].split("\ndef ", 1)[0]


def test_warm_started_runs_depend_on_the_n0_run_of_the_same_task_and_seed() -> None:
    specs = {s.run_id: s for s in manifest.design("main")}
    for s in manifest.design("controller"):
        if s.controller_variant == "warm_started":
            (dep,) = s.depends_on
            assert dep in specs and specs[dep].N == 0 and specs[dep].seed == s.seed and specs[dep].task == s.task


def test_additional_constrained_training_trains_N_T_more_and_carries_q_data_control_lr() -> None:
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
    parents = {s.run_id: s for s in manifest.design("study_b")}
    for s in manifest.design("study_b_fewshot"):
        assert s.depends_on == (s.params["parent_run_id"],)
        bins = ("Q-continuous-bins",) if s.arm == "Continuous" else ()
        assert s.pending == ("Q-continuations", "Q-jc-window", "Q-budget-normalisation", "Q-level-jc", *bins)
        assert s.total_steps == R.FEWSHOT_STEPS and s.params["budget"] in R.UNSEEN_BUDGETS
        # pilot/manifest.py: the continuation starts from its parent's final checkpoint
        assert s.params["parent_step"] == parents[s.params["parent_run_id"]].total_steps == R.TOTAL_STEPS
    # a continuation follows its parent's seed: the scheduler builds the new parent's continuations instead
    with pytest.raises(ValueError, match="continuation"):
        manifest.design("study_b_fewshot")[0].with_seed(5)
    moved = manifest.study_b_fewshot([manifest.design("study_b")[0].with_seed(5)])[0]
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
    for key, value in (("total_steps", 10_000_000.0), ("seed", 0.0), ("seed", True), ("onset_step", 0.0)):
        with pytest.raises(ValueError, match=f"{key} must be an integer"):
            RunSpec.from_dict({**good, key: value})
    assert RunSpec.from_dict({**good, "onset_step": None}).onset_step is None
    with pytest.raises(TypeError):  # frozen but unhashable: params is a dict (documented on RunSpec)
        hash(manifest.pilot()[0])


def test_the_pilot_seeds_cannot_be_chosen_and_composites_are_checked_for_duplicates(monkeypatch) -> None:
    for name, prose in (("pilot", "pilot"), ("repilot", "re-pilot")):
        with pytest.raises(ValueError, match=f"the {prose} has its registered seeds"):
            manifest.design(name, seeds=(7,))
        assert len(manifest.design(name, seeds=R.SEEDS)) == R.PILOT_RUNS
    one = manifest.study_b((0,))
    monkeypatch.setattr(manifest, "study_b_fewshot", lambda parents: one)  # collides with the study_b runs
    with pytest.raises(AssertionError, match="duplicate run_id"):
        manifest.design("all", seeds=(0,))


@pytest.mark.parametrize("revision", [True, False, 1.0, 0.0, 2, -1, "1", None])
def test_the_pilot_revision_must_be_the_integer_0_or_1(revision) -> None:
    with pytest.raises(ValueError, match="one revision"):
        manifest.pilot_prefix(revision)
    with pytest.raises(ValueError, match="one revision"):
        manifest.pilot(revision=revision)
    assert (manifest.pilot_prefix(0), manifest.pilot_prefix(1)) == ("P-", "P1-")


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


def test_ramp_surplus_seeds_carry_q_surplus_arm_set(monkeypatch) -> None:
    # Only Q-surplus-arm-set is under test: every run gate (manifest.run_gate_keys) is answered here.
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset(manifest.RUN_GATE_ORDER))
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
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset({"Q-surplus-arm-set", *manifest.RUN_GATE_ORDER}))
    assert manifest.surplus_spec(ramp, 6).launchable
    other = next(s for s in main if s.N == 0.10)
    with pytest.raises(ValueError, match="primary-comparison"):
        manifest.surplus_spec(other, 5)


# ---------------------------------------------------------------------------
# Run gates, battery continuations, cut order
# ---------------------------------------------------------------------------


def _everything() -> list[RunSpec]:
    return manifest.design("all") + manifest.design("pilot") + manifest.design("repilot")


def _row(parent: RunSpec, **changes) -> dict:
    step = parent.total_steps - 2 * R.CHECKPOINT_INTERVAL_STEPS  # in rule 1's window (9,600,000 for a 10M-step run)
    row = {"run_id": parent.run_id, "matched": True, "completed": True, "matched_checkpoint_step": step,
           "matched_checkpoint_path": f"{parent.run_id}/omnisafe/PPOLag-x/seed-000-t/torch_save/epoch-480.pt",
           "commit_hash": "b" * 40, "seed": parent.seed}
    row.update(changes)
    return row


def test_every_key_a_spec_carries_is_a_pending_key() -> None:
    for s in _everything():
        assert set(s.pending) <= set(R.PENDING), s.run_id
        assert len(set(s.pending)) == len(s.pending), s.run_id
        assert set(manifest.run_gate_keys(s)) <= set(s.pending), s.run_id
    assert set(manifest.RUN_GATE_ORDER) <= set(R.PENDING)


def test_run_gate_keys_follow_their_docstring() -> None:
    specs = _everything()

    def holders(key: str) -> set[str]:
        return {s.run_id for s in specs if key in s.pending}

    def ids(pred) -> set[str]:
        return {s.run_id for s in specs if pred(s)}

    lagrangian = ("PPOLag", "CPPOPID")
    assert holders("Q-cost-critic") == ids(lambda s: s.study == "A" and (s.onset_step or 0) > 0)
    assert holders("Q-jc-window") == ids(lambda s: s.base_algo in lagrangian)  # the pilot's included
    assert holders("Q-ramp-step") == ids(lambda s: s.onset_shape == "ramp")
    assert holders("Q-warm-start") == ids(lambda s: s.controller_variant == "warm_started")
    assert holders("Q-rate-limit") == ids(lambda s: s.controller_variant == "rate_limited")
    assert holders("Q-pid-eq9") == ids(lambda s: s.group == "pid")
    assert holders("Q-reset-injection") == ids(lambda s: s.treatment in ("reset", "injection"))
    assert holders("Q-plasticity-definitions") == ids(lambda s: s.study == "A" and not s.pilot)
    assert holders("Q-budget-normalisation") == holders("Q-level-jc") == ids(lambda s: s.plugin in ("study_b", "study_b_fewshot"))
    assert holders("Q-continuous-bins") == ids(lambda s: s.run_id.startswith("B-Continuous-"))
    assert holders("Q-search-before-pilot") == ids(lambda s: s.pilot and s.study == "B")
    assert holders("Q-studyb-order") == ids(lambda s: s.group == "study_b")
    assert holders("Q-continuations") == ids(lambda s: s.group == "study_b_fewshot")
    assert not holders("Q-transfer-obs")  # only battery continuations, which the design does not list
    counts = {k: len(holders(k)) for k in manifest.RUN_GATE_ORDER}
    assert counts["Q-jc-window"] == 435 + 35 + 140 + 2 * 7  # Study A, Study B, few-shot, pilot and re-pilot
    assert counts["Q-cost-critic"] == 420 + 2 * 3  # every late Study A run; the N = 0.50 runs of the pilot and the re-pilot
    assert counts["Q-plasticity-definitions"] == 435
    assert counts["Q-pid-eq9"] == R.PID_RUNS and counts["Q-reset-injection"] == 90 and counts["Q-ramp-step"] == 90
    assert counts["Q-warm-start"] == counts["Q-rate-limit"] == 45
    assert counts["Q-budget-normalisation"] == 35 + 140 + 2 and counts["Q-continuous-bins"] == 5 + 20
    assert counts["Q-search-before-pilot"] == 4 and counts["Q-studyb-order"] == 35


def test_the_pilot_waits_on_the_questions_that_must_be_answered_before_it() -> None:
    pilot = {s.arm: s for s in manifest.pilot() if s.seed == 0}
    assert pilot["N0.00"].pending == ("Q-jc-window",)
    assert pilot["N0.50-abrupt-total"].pending == ("Q-cost-critic", "Q-jc-window")
    assert pilot["Moderate"].pending == ("Q-jc-window", "Q-budget-normalisation", "Q-level-jc", "Q-search-before-pilot")
    assert [s.pending for s in manifest.pilot(revision=1)] == [s.pending for s in manifest.pilot()]


def test_surplus_and_replacement_seeds_keep_their_run_gates() -> None:
    base = next(s for s in manifest.design("main") if s.task == R.PRIMARY_TASK and s.onset_shape == "ramp" and s.N == 0.5)
    extra = manifest.surplus_spec(base, 5)
    assert extra.pending == (*base.pending, "Q-surplus-arm-set")
    assert base.with_seed(7).pending == base.pending


def test_continuation_groups_follow_their_parent() -> None:
    assert set(manifest.CONTINUATION_GROUPS) == {"study_b_fewshot", "battery_finetune", "battery_transfer"}
    assert set(manifest.CONTINUATION_GROUPS) <= set(manifest.GROUPS)
    assert manifest.PLUGINS["battery_finetune"] == ("envs.continuations:make_finetune_algorithm", "Environment and tests (Role 2)")
    assert manifest.PLUGINS["battery_transfer"] == ("envs.continuations:make_transfer_algorithm", "Environment and tests (Role 2)")
    parent = next(s for s in manifest.design("main") if s.N == 0.5)
    for condition in manifest.BATTERY_CONDITIONS:
        with pytest.raises(ValueError, match="continuation"):
            manifest.battery_continuation(parent, _row(parent), condition).with_seed(5)


def test_the_transfer_task_of_each_study_a_task() -> None:
    assert dict(manifest.TRANSFER_TASKS_ANSWERED) == {"SafetyCarGoal1-v0": "SafetyCarButton1-v0"}
    assert not set(manifest.TRANSFER_TASKS_ANSWERED) & set(R.TRANSFER_TASKS)  # never overrides the registered pair
    assert {t: manifest.transfer_task(t) for t in R.TASKS_STUDY_A} == {
        "SafetyPointGoal1-v0": "SafetyPointButton1-v0", "SafetyPointButton1-v0": "SafetyPointGoal1-v0",
        "SafetyCarGoal1-v0": "SafetyCarButton1-v0",
    }
    with pytest.raises(ValueError):
        manifest.transfer_task("SafetyCarButton1-v0")


def test_battery_continuations_of_a_matched_arm() -> None:
    parent = next(s for s in manifest.design("main") if s.task == "SafetyCarGoal1-v0" and s.N == 0.5 and s.seed == 3
                  and s.onset_shape == "abrupt" and s.step_matching == "total_steps")
    row = _row(parent)
    finetune = manifest.battery_continuation(parent, row, "finetune")
    transfer = manifest.battery_continuation(parent, row, "transfer")
    assert finetune.run_id == "A-CarGoal1-N0.50-abrupt-total-finetune-s3"
    assert transfer.run_id == "A-CarGoal1-N0.50-abrupt-total-transfer-s3"
    assert (finetune.task, transfer.task) == ("SafetyCarGoal1-v0", "SafetyCarButton1-v0")
    assert (finetune.total_steps, transfer.total_steps) == (R.FINETUNE_STEPS, R.TRANSFER_STEPS)
    for spec, condition in ((finetune, "finetune"), (transfer, "transfer")):
        assert spec.depends_on == (parent.run_id,) and spec.seed == parent.seed and spec.onset_step is None
        assert (spec.plugin, spec.group, spec.base_algo) == (f"battery_{condition}", f"battery_{condition}", "PPOLag")
        assert {k: spec.params[k] for k in ("parent_run_id", "parent_step", "parent_checkpoint", "parent_commit",
                                            "condition", "source_task", "cost_limit")} == {
            "parent_run_id": parent.run_id, "parent_step": row["matched_checkpoint_step"],
            "parent_checkpoint": row["matched_checkpoint_path"], "parent_commit": row["commit_hash"],
            "condition": condition, "source_task": parent.task, "cost_limit": R.COST_LIMIT}
        assert RunSpec.from_json(spec.to_json()) == spec
        assert (spec.N, spec.treatment, spec.controller_variant, spec.arm) == (parent.N, None, None, parent.arm)
    assert finetune.params["multiplier"] == "frozen_zero" and "fresh_multiplier_init" not in finetune.params
    assert transfer.params["fresh_multiplier_init"] == R.LAGRANGE_MULTIPLIER_INIT
    assert finetune.pending == ("Q-continuations",)  # the multiplier is held at 0: no J_C window
    assert transfer.pending == ("Q-jc-window", "Q-continuations", "Q-transfer-obs")
    # the spec is a pure function of parent and row: queueing it again is a no-op
    assert manifest.battery_continuation(parent, dict(row), "finetune") == finetune
    # a PID parent's continuation trains PPO-Lagrangian, like every other arm's: Table 9.1 (Q-continuations;
    # Table 2.2 names no controller), kept in one labelled constant
    pid = manifest.design("pid")[0]
    assert pid.base_algo == "CPPOPID" and manifest.CONTINUATION_BASE_ALGO == "PPOLag"
    for condition in manifest.BATTERY_CONDITIONS:
        assert manifest.battery_continuation(pid, _row(pid), condition).base_algo == manifest.CONTINUATION_BASE_ALGO
    (line,) = [x for x in Path(manifest.__file__).read_text(encoding="utf-8").splitlines() if x.startswith("CONTINUATION_BASE_ALGO =")]
    assert line.endswith("# answered in Table 9.1 (Q-continuations)") and "Q-continuations" in R.PENDING
    assert "Q-pid-eq9" not in manifest.battery_continuation(pid, _row(pid), "transfer").pending


def test_battery_continuations_accept_a_ledger_row_model() -> None:
    parent = next(s for s in manifest.design("main") if s.N == 0.0)

    class Row:  # a pydantic LedgerRow offers model_dump()
        def model_dump(self) -> dict:
            return _row(parent)

    assert manifest.battery_continuation(parent, Row(), "finetune").params["parent_step"] == 9_600_000


@pytest.mark.parametrize("changes, message", [
    ({"matched": False}, "rule 6"), ({"matched": None}, "rule 6"), ({"matched": 1}, "rule 6"),
    ({"completed": False}, "did not complete"), ({"run_id": "A-other-s0"}, "does not belong"),
    ({"matched_checkpoint_step": None}, "not a checkpoint"), ({"matched_checkpoint_step": 12_345}, "not a checkpoint"),
    ({"matched_checkpoint_step": 30_000_000}, "not a checkpoint"), ({"matched_checkpoint_step": 9.6e6}, "not a checkpoint"),
    ({"matched_checkpoint_step": 9_620_000}, "not a checkpoint"),  # a whole epoch, but not a saved checkpoint
    ({"matched_checkpoint_step": 20_000}, "not a checkpoint"),  # a whole epoch off the 200,000-step grid (not a saved checkpoint)
    ({"matched_checkpoint_path": None}, "matched_checkpoint_path"), ({"commit_hash": ""}, "commit_hash"),
])
def test_battery_continuations_are_refused_for_invalid_or_unmatched_rows(changes, message) -> None:
    parent = next(s for s in manifest.design("main") if s.N == 0.5)
    with pytest.raises(ValueError, match=message):
        manifest.battery_continuation(parent, _row(parent, **changes), "finetune")


def test_battery_continuations_start_from_a_checkpoint_the_run_saves() -> None:
    parent = next(s for s in manifest.design("main") if s.N == 0.5 and s.total_steps == R.TOTAL_STEPS)
    for step in (8_200_000, 9_000_000, R.TOTAL_STEPS):  # grid checkpoints and the final one
        assert manifest.battery_continuation(parent, _row(parent, matched_checkpoint_step=step), "finetune").params[
            "parent_step"] == step
    off_grid = next(s for s in manifest.design("treatment") if s.total_steps % R.CHECKPOINT_INTERVAL_STEPS)
    step = off_grid.total_steps - R.CHECKPOINT_INTERVAL_STEPS  # the end-relative window (Q-selection-window)
    assert manifest.battery_continuation(off_grid, _row(off_grid, matched_checkpoint_step=step), "finetune").params[
        "parent_step"] == step
    with pytest.raises(ValueError, match="not a checkpoint"):  # a whole epoch the run does not save
        manifest.battery_continuation(off_grid, _row(off_grid, matched_checkpoint_step=step - 20_000), "finetune")


def test_battery_continuations_are_refused_for_other_parents_and_conditions() -> None:
    parent = next(s for s in manifest.design("main") if s.N == 0.5)
    with pytest.raises(ValueError, match="condition"):
        manifest.battery_continuation(parent, _row(parent), "hazard")
    for other in (manifest.pilot()[0], manifest.design("study_b")[0], manifest.design("study_b_fewshot")[0]):
        with pytest.raises(ValueError, match="main study"):
            manifest.battery_continuation(other, _row(other), "finetune")


def test_cuts_follow_part_6_1_in_order() -> None:
    specs = manifest.design("all")
    assert manifest.apply_cuts(specs, ()) == specs
    one = manifest.apply_cuts(specs, manifest.CUTS[:1])
    assert len(specs) - len(one) == R.PID_RUNS  # "Drop the PID check (15 runs)"
    two = manifest.apply_cuts(specs, manifest.CUTS[:2])
    dropped = Counter(s.group for s in one if s not in two)
    assert dropped == {"main": 65, "treatment": 45, "controller": 30}  # "(65 main-sweep runs, 45 treatment runs, 30 controller runs)"
    three = manifest.apply_cuts(specs, manifest.CUTS[:3])
    fewshot = [s for s in three if s.group == "study_b_fewshot"]
    assert len(fewshot) == R.STUDY_B_CONTINUATIONS
    assert all(s.total_steps == R.FEWSHOT_HORIZONS[0] and s.params["horizons"] == [R.FEWSHOT_HORIZONS[0]] for s in fewshot)
    # only total_steps and horizons change (the specs' rewrite, pilot.scheduler.CUTS_PATH), and the learning-rate
    # schedule keeps the full continuation's 1,000,000 steps (Table 9.1, Q-continuations; restore_learner reads it):
    # the shortened run is the full continuation's first 200,000 steps
    uncut = {s.run_id: s for s in two if s.group == "study_b_fewshot"}
    assert not any("lr_schedule_steps" in s.params for s in uncut.values())
    for s in fewshot:
        before = uncut[s.run_id].to_dict()
        assert s.to_dict() == {**before, "total_steps": R.FEWSHOT_HORIZONS[0],
                               "params": {**before["params"], "horizons": [R.FEWSHOT_HORIZONS[0]],
                                          "lr_schedule_steps": R.FEWSHOT_STEPS}}
        assert s.params["lr_schedule_steps"] == 1_000_000 and s.total_steps == 200_000
    assert manifest.apply_cuts(three, ()) == three and manifest._cut_fewshot_short(fewshot) == fewshot  # idempotent
    saved = sum(s.run_equivalents for s in two) - sum(s.run_equivalents for s in three)
    assert saved == pytest.approx(11.2)  # "about 11 run-equivalents": 140 x 800,000 / 10,000,000
    four = manifest.apply_cuts(specs, manifest.CUTS[:4])
    assert {s.N for s in four if s.group in ("treatment", "controller")} == {0.5}
    five = manifest.apply_cuts(specs, manifest.CUTS)
    main = [s for s in five if s.group == "main"]
    per_task = Counter(s.task for s in main)
    assert per_task == {"SafetyPointGoal1-v0": 45, "SafetyPointButton1-v0": 45}  # 20 runs of N = 0.10 dropped per task
    assert all(s.N != 0.10 for s in main)
    # never cut: seeds, Study B's zero-shot arms, the N = 0 and N = 0.50 arms on the two Point tasks
    for spec in specs:
        if manifest.is_never_cut(spec):
            assert spec in five
    assert [s for s in five if s.group == "study_b"] == [s for s in specs if s.group == "study_b"]
    arms_before = Counter(s.arm_id for s in specs)
    assert all(arms_before[a] == n for a, n in Counter(s.arm_id for s in five).items())


def test_cuts_must_be_a_prefix_of_the_registered_order() -> None:
    specs = manifest.design("pid")
    for cuts in (("car",), ("pid", "fewshot_short"), ("pid", "car", "fewshot_short", "drop_n010"), ("nope",)):
        with pytest.raises(ValueError, match="prefix"):
            manifest.apply_cuts(specs, cuts)


def test_cuts_follow_a_battery_continuation_to_its_parent() -> None:
    specs = []
    for parent in manifest.design("main")[:13] + manifest.design("pid")[:1] + manifest.design("treatment")[:1]:
        specs += [manifest.battery_continuation(parent, _row(parent), c) for c in manifest.BATTERY_CONDITIONS]
    car = next(s for s in manifest.design("main") if s.task == "SafetyCarGoal1-v0" and s.N == 0.1)
    specs += [manifest.battery_continuation(car, _row(car), "transfer")]  # trains on SafetyCarButton1-v0
    after = manifest.apply_cuts(specs, manifest.CUTS)
    assert not [s for s in after if s.controller_variant == R.PID_VARIANT]
    assert not [s for s in after if s.params["source_task"] == "SafetyCarGoal1-v0"]
    assert not [s for s in after if s.treatment is not None and s.N != 0.5]
    assert not [s for s in after if s.treatment is None and s.controller_variant is None and s.N == 0.1]
    assert [s for s in after if s.N in (0.0, 0.5)] == [s for s in specs if s.N in (0.0, 0.5) and s.group != "pid"
                                                      and s.controller_variant != R.PID_VARIANT
                                                      and s.params["source_task"] != "SafetyCarGoal1-v0"]


def test_the_never_cut_guard_refuses_a_cut_that_removes_the_primary_comparison(monkeypatch) -> None:
    specs = manifest.design("main") + manifest.design("study_b")
    monkeypatch.setitem(manifest._CUT_FUNCTIONS, "pid", lambda s: [x for x in s if x.N != 0.5])
    with pytest.raises(ValueError, match="never cuts"):
        manifest.apply_cuts(specs, ("pid",))
    monkeypatch.setitem(manifest._CUT_FUNCTIONS, "pid", lambda s: [x for x in s if x.group != "study_b"])
    with pytest.raises(ValueError, match="never cuts"):
        manifest.apply_cuts(specs, ("pid",))
    monkeypatch.setitem(manifest._CUT_FUNCTIONS, "pid", lambda s: [x for x in s if not (x.N == 0.25 and x.seed == 4)])
    with pytest.raises(ValueError, match="seeds"):
        manifest.apply_cuts(specs, ("pid",))


def test_never_cut_is_the_registered_list() -> None:
    # Part 6.1: "the N = 0 and N = 0.50 arms under both step-matching controls on SafetyPointGoal1-v0 and
    # SafetyPointButton1-v0, which carry the primary comparison"
    assert manifest.NEVER_CUT_TASKS == ("SafetyPointGoal1-v0", "SafetyPointButton1-v0")
    assert manifest.CAR_TASKS == ("SafetyCarGoal1-v0",)
    protected = [s for s in manifest.design("all") if manifest.is_never_cut(s)]
    assert Counter(s.group for s in protected) == {"study_b": 35, "main": 2 * 5 * len(R.SEEDS)}  # N0.00 + 4 N0.50 arms


@pytest.mark.parametrize("hours, concurrent", [
    (float("nan"), 2), (float("inf"), 2), (0.0, 2), (-1.0, 2), (True, 2), ("10", 2),
    (1.0, 1.5), (1.0, 0), (1.0, True), (1.0, float("nan")),
])
def test_g2_requirements_refuse_invalid_throughput(hours, concurrent) -> None:
    for requirement in (budget.registered_requirement, budget.corrected_requirement):
        with pytest.raises(ValueError):
            requirement(hours, concurrent)
    assert budget.registered_requirement(1.0, 2) == pytest.approx(R.G2_RUN_EQUIVALENTS / 2)


@pytest.mark.parametrize("hours, decided_by", [
    (True, "the group"), ("5000", "the group"), (float("nan"), "the group"), (0, "the group"),
    (5000, None), (5000, 42), (5000, ""),
])
def test_record_allocation_refuses_invalid_inputs_with_value_error(tmp_path, hours, decided_by) -> None:
    path = tmp_path / "allocation.json"
    with pytest.raises(ValueError):
        budget.record_allocation(hours, decided_by, "2026-10-01", "24 x 120 days x 0.95", path)
    assert not path.exists()


def test_allocation_paths_may_be_strings_and_outside_the_repository(tmp_path) -> None:
    path = tmp_path / "allocation.json"
    budget.record_allocation(5000, "the group", "2026-10-01", "24 x 120 days x 0.95", str(path))
    assert budget.load_allocation(str(path))["machine_hours_allocated"] == 5000.0
    for given in (path, str(path)):  # outside the repository: an answer, not an exception
        ok, note = budget.allocation_contained([], given)
        assert ok is False and "not inside the repository" in note

"""configs/registered.py against the registered text (First Tasks, habit 2).

Each constant is checked twice: its value is asserted, and the phrase of the pre-registration that
fixes it must appear verbatim in prereg/Preregistration.docx. A constant that drifts, or a
document that changes without the constant, fails here.
"""

from __future__ import annotations

import html
import math
import re
import zipfile
from functools import lru_cache
from pathlib import Path

import pytest

from configs import registered as R

DOCX = Path(__file__).resolve().parents[1] / "prereg" / "Preregistration.docx"


@lru_cache(maxsize=1)
def prereg_text() -> str:
    xml = zipfile.ZipFile(DOCX).read("word/document.xml").decode("utf-8")
    xml = re.sub(r"</w:p>", "\n", xml)
    return re.sub(r"[ \t]+", " ", html.unescape(re.sub(r"<[^>]+>", "", xml)))


CASES = [
    # (constant, expected value, phrase in the pre-registration)
    ("TOTAL_STEPS", 10_000_000, "10,000,000 per run: 500 epochs of 20,000 steps"),
    ("STEPS_PER_EPOCH", 20_000, "500 epochs of 20,000 steps"),
    ("EPOCHS", 500, "500 epochs of 20,000 steps"),
    ("UPDATE_ITERS", 40, "40 update iterations"),
    ("MINIBATCH_SIZE", 64, "minibatches of 64"),
    ("TARGET_KL", 0.02, "target KL of 0.02"),
    ("COST_LIMIT", 25.0, "d = 25 per episode"),
    ("LAGRANGE_MULTIPLIER_INIT", 0.001, "initial value 0.001"),
    ("LAGRANGE_MULTIPLIER_LR", 0.035, "learning rate 0.035"),
    ("HIDDEN_SIZES", (64, 64), "two hidden layers of 64 units and tanh"),
    ("ACTIVATION", "tanh", "two hidden layers of 64 units and tanh"),
    ("EPISODE_LENGTH", 1_000, "1,000 steps"),
    ("DEVICE", "cpu", "on CPU with one torch thread"),
    ("TORCH_THREADS", 1, "one torch thread"),
    ("VECTOR_ENV_NUMS", 1, "one vectorised environment"),
    ("PARALLEL_PROCESSES", 1, "one process each"),
    ("TASKS_STUDY_A", ("SafetyPointGoal1-v0", "SafetyCarGoal1-v0", "SafetyPointButton1-v0"),
     "Study A: SafetyPointGoal1-v0, SafetyCarGoal1-v0,"),
    ("PRIMARY_TASK", "SafetyPointGoal1-v0", "with SafetyPointGoal1-v0 as the primary task"),
    ("TASKS_STUDY_B", ("SafetyPointGoal1-v0",), "Study B: SafetyPointGoal1-v0 only"),
    ("CHECKPOINT_INTERVAL_STEPS", 200_000, "Every 200,000 steps, that is every ten epochs"),
    ("CHECKPOINT_INTERVAL_EPOCHS", 10, "every ten epochs"),
    ("SELECTION_WINDOW_CHECKPOINTS", 10, "last ten checkpoints (the final 2,000,000 steps"),
    ("SELECTION_WINDOW_STEPS", 2_000_000, "(the final 2,000,000 steps"),
    ("EVAL_EPISODES", 100, "Mean episodic cost over 100 evaluation episodes"),
    ("ONSET_FRACTIONS", (0.0, 0.10, 0.25, 0.50), "Levels: 0, 0.10, 0.25, 0.50"),
    ("SEEDS", (0, 1, 2, 3, 4), "Five per arm: 0, 1, 2, 3, 4"),
    ("MAIN_ARMS_PER_TASK", 13, "1 + 3 × 2 × 2 = 13 main arms"),
    ("MAIN_SWEEP_RUNS", 195, "the main sweep is 195 runs"),
    ("RAMP_WINDOW_FRACTION", 0.10, "W = 0.10 T"),
    ("RATE_LIMIT_RELATIVE", 0.05, "at most 5 percent of its current value plus 0.01 per epoch"),
    ("RATE_LIMIT_ABSOLUTE", 0.01, "plus 0.01 per epoch"),
    ("SETTLING_BAND", 0.10, "±10 percent of its final value"),
    ("OVERSHOOT_WINDOW_STEPS", 2_000_000, "the 2,000,000 steps after onset"),
    ("OVERSHOOT_FINAL_FRACTION", 0.10, "last 10 percent of training epochs"),
    ("HAZARD_LAYOUTS", 20, "twenty new layout seeds, five episodes per layout"),
    ("EPISODES_PER_LAYOUT", 5, "five episodes per layout"),
    ("BODY_MASS_SCALE", 1.3, "Body mass scaled by 1.3"),
    ("GROUND_FRICTION_SCALE", 0.7, "ground friction by 0.7"),
    ("FINETUNE_STEPS", 1_000_000, "Training continues for 1,000,000 steps, one tenth of T"),
    ("TRANSFER_STEPS", 1_000_000, "fine-tuned for 1,000,000 steps, one tenth of T"),
    ("FIXED_BATCH_STATES", 2_048, "2,048 states (decision)"),
    ("DORMANT_THRESHOLD", 0.025, "τ = 0.025"),
    ("EFFECTIVE_RANK_DELTA", 0.01, "δ = 0.01"),
    ("MANIPULATION_CHECK_STEPS_AFTER_ONSET", 200_000, "at the first logging point after onset (200,000 steps)"),
    ("CONTINUOUS_RANGE", (10.0, 40.0), "uniformly from [10, 40] at each episode"),
    ("CONTINUOUS_BIN_WIDTH", 5.0, "one multiplier per 5-unit bin"),
    ("UNSEEN_BUDGETS", (5.0, 15.0, 30.0, 45.0), "{5, 15, 30, 45}"),
    ("BUDGET_OBSERVATION_DIVISOR", 100.0, "The active budget divided by 100"),
    ("FEWSHOT_STEPS", 1_000_000, "One continuation of 1,000,000 steps per unseen budget"),
    ("FEWSHOT_HORIZONS", (200_000, 500_000, 1_000_000), "200,000, 500,000 and 1,000,000 steps"),
    ("SATISFACTION_TARGET", 0.80, "reaches 0.80 (decision)"),
    ("STUDY_B_RUNS", 35, "Seven arms with five seeds are 35 training runs"),
    ("STUDY_B_CONTINUATIONS", 140, "35 × 4 = 140 continuations"),
    ("STUDY_B_RUN_EQUIVALENTS", 49, "Study B's total is therefore 49 run-equivalents"),
    ("MATCH_TOLERANCE", 2.5, "≤ 2.5 cost units, one tenth of the budget"),
    ("INFEASIBLE_REFERENCE_SD", 10.0, "exceeds 10 cost units in a task"),
    ("SENSITIVITY_TOLERANCE", 5.0, "once with tolerance 5.0"),
    ("MIN_EFFECT_STUDY_A", 5.0, "5 cost units of robustness-gap difference"),
    ("MIN_EFFECT_STUDY_B", 0.05, "5 percentage points of constraint-satisfaction rate"),
    ("ALPHA_PRIMARY_STUDY_B", 0.025, "α = 0.025 (Bonferroni within the primary family of two)"),
    ("BOOTSTRAP_RESAMPLES", 10_000, "10,000 resamples of seed indices"),
    ("NEXT_UNUSED_SEED_START", 5, "repeated with the next unused seed (5, 6, and so on)"),
    ("SURPLUS_SEED_START", 5, "Seeds are added in the order 5, 6, 7, and so on"),
    ("MAX_SEEDS_PER_ARM", 12, "up to twelve"),
    ("PILOT_SEEDS", (0, 1, 2), "seeds 0, 1, 2"),
    ("PILOT_MODERATE_SEED", 0, "one Moderate arm run with seed 0"),
    ("PILOT_RUNS", 8, "eight runs, completed before the go decision"),
    ("G1_COST_MARGIN", 2.5, "of at most d + 2.5 within T"),
    ("G1_REFERENCE_SD_MAX", 10.0, "standard deviation of that cost is at most 10"),
    ("G2_RUN_EQUIVALENTS", 484, "484 run-equivalents (435 for Study A, 49 for Study B)"),
    ("G3_SIGNAL_MIN", 5.0, "is at least 5 cost units"),
    ("U80_T_QUANTILE", 1.886, "1.886"),
    ("G4_COST_MARGIN", 2.5, "each at most its budget plus 2.5"),
    ("LATE_ONSET_FRACTIONS", (0.10, 0.25, 0.50), "at N = 0.10, 0.25, 0.50, in three tasks"),
    ("ONSET_SHAPES", ("abrupt", "ramp"), "Abrupt; linear ramp"),
    ("STEP_MATCHING_CONTROLS", ("constrained_steps", "total_steps"), "Constrained-steps matched; total-steps matched"),
    ("TREATMENTS", ("reset", "injection", "additional_constrained"), "Reset, injection, additional constrained training"),
    ("CONTROLLER_VARIANTS", ("warm_started", "rate_limited"), "Warm-started and rate-limited multiplier"),
    ("PID_VARIANT", "pid", "PID multiplier, abrupt arms, SafetyPointGoal1-v0"),
    ("PID_TASKS", ("SafetyPointGoal1-v0",), "run as a further check on SafetyPointGoal1-v0 only"),
    ("TREATMENT_RUNS", 135, "3 × 3 × 3 × 5"),
    ("CONTROLLER_RUNS", 90, "2 × 3 × 3 × 5"),
    ("PID_RUNS", 15, "3 × 5 = 15"),
    ("G2_RUN_EQUIVALENTS_STUDY_A", 435, "(435 for Study A, 49 for Study B)"),
    ("G2_RUN_EQUIVALENTS_STUDY_B", 49, "(435 for Study A, 49 for Study B)"),
    ("ALPHA", 0.05, "by the Holm procedure at α = 0.05"),
    ("FIXED_BATCH_SEED", 0, "collected once per task by a uniform-random policy under seed 0"),
    ("PILOT_STUDY_A_TASK", "SafetyPointGoal1-v0", "Study A: SafetyPointGoal1-v0; arms N = 0 and N = 0.50"),
    ("PILOT_STUDY_A_ONSET_FRACTIONS", (0.0, 0.50), "arms N = 0 and N = 0.50, abrupt, total-steps matched"),
    ("PILOT_STUDY_A_SHAPE", "abrupt", "arms N = 0 and N = 0.50, abrupt, total-steps matched"),
    ("PILOT_STUDY_A_CONTROL", "total_steps", "abrupt, total-steps matched; seeds 0, 1, 2"),
    ("PILOT_BATTERY", ("hazard", "dynamics"), "battery conditions hazard relocation and dynamics perturbation"),
]


@pytest.mark.parametrize("name, expected, phrase", CASES, ids=[c[0] for c in CASES])
def test_registered_value_and_source(name: str, expected, phrase: str) -> None:
    assert getattr(R, name) == expected
    assert phrase in prereg_text(), f"phrase for {name} not found in the pre-registration: {phrase!r}"


def test_d_loose_is_twice_the_benchmark_unconstrained_cost() -> None:
    text = prereg_text()
    for task, loose in R.D_LOOSE.items():
        unconstrained = R.BENCHMARK_UNCONSTRAINED_PPO_COST[task]
        assert round(2 * unconstrained) == loose  # Table 2.1: twice the unconstrained cost
        assert f"{loose:g} for {task} (from {unconstrained:.2f})" in text


def test_study_b_arms_match_table_2_5() -> None:
    text = prereg_text()
    assert "Single: {10}, {20} or {40}. Sparse: {10, 40}. Moderate: {10, 20, 40}." in text
    assert "Dense: {10, 17.5, 25, 32.5, 40}." in text
    assert list(R.STUDY_B_ARMS) == ["Single-10", "Single-20", "Single-40", "Sparse", "Moderate", "Dense", "Continuous"]
    assert dict(R.STUDY_B_ARMS) == {
        "Single-10": (10.0,), "Single-20": (20.0,), "Single-40": (40.0,), "Sparse": (10.0, 40.0),
        "Moderate": (10.0, 20.0, 40.0), "Dense": (10.0, 17.5, 25.0, 32.5, 40.0), "Continuous": None,
    }


def test_ramp_window_is_a_tenth_of_T() -> None:
    assert R.RAMP_WINDOW_STEPS == R.RAMP_WINDOW_FRACTION * R.TOTAL_STEPS


def test_checkpoint_interval_is_ten_epochs() -> None:
    assert R.CHECKPOINT_INTERVAL_EPOCHS * R.STEPS_PER_EPOCH == R.CHECKPOINT_INTERVAL_STEPS
    assert R.EPOCHS * R.STEPS_PER_EPOCH == R.TOTAL_STEPS


def test_u80_quantile_is_the_t_quantile_with_two_degrees_of_freedom() -> None:
    # Student's t with 2 degrees of freedom has the closed-form quantile (2p - 1) / sqrt(2 p (1 - p)).
    p = 0.90
    assert round((2 * p - 1) / math.sqrt(2 * p * (1 - p)), 3) == R.U80_T_QUANTILE


def test_open_questions_are_named_and_answers_are_pending_keys() -> None:
    keys = {"Q-rounding", "Q-pilot-unconstrained-seed", "Q-interrupted-run", "Q-seed-collision",
            "Q-g1-level", "Q-g3-pairing", "Q-surplus-arm-set", "Q-data-control-lr", "Q-selection-window",
            "Q-g4-level", "Q-continuations", "Q-g2-run-equivalents"}
    assert set(R.PENDING) == keys
    # HANDOVER.md section 9 marks exactly the PENDING keys with a star ("Q-key\*" in Markdown), so a
    # key added to or dropped from PENDING without updating the handover fails here.
    handover = (Path(__file__).resolve().parents[1] / "HANDOVER.md").read_text(encoding="utf-8")
    section_9 = re.search(r"^## 9\..*?(?=^## )", handover, flags=re.S | re.M)
    assert section_9 is not None, "HANDOVER.md has no section 9"
    assert set(re.findall(r"(Q-[a-z0-9-]+)\\\*", section_9.group(0))) == keys
    # Only PENDING keys can be answered (HANDOVER.md task 11: answered by an amendment row of Table 9.1),
    # and a key is open exactly while it is not answered.
    assert R.ANSWERED_QUESTIONS <= keys
    assert all(R.is_open(key) == (key not in R.ANSWERED_QUESTIONS) for key in keys)
    with pytest.raises(KeyError):
        R.is_open("Q-unknown")

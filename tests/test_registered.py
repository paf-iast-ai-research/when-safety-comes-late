"""configs/registered.py against the registered text (First Tasks, habit 2).

Each registered value is asserted, and the phrase of the pre-registration that fixes it (or the
values it is derived from) must appear verbatim in prereg/Preregistration.docx. A constant that
drifts, or a document that changes without the constant, fails here.
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

ROOT = Path(__file__).resolve().parents[1]
DOCX = ROOT / "prereg" / "Preregistration.docx"


@lru_cache(maxsize=1)
def prereg_text() -> str:
    with zipfile.ZipFile(DOCX) as docx:
        xml = docx.read("word/document.xml").decode("utf-8")
    xml = re.sub(r"</w:p>", "\n", xml)
    xml = re.sub(r"</w:tc>", " | ", xml)  # a phrase cannot match across a table-cell boundary
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
    ("TREATMENTS", ("reset", "injection", "additional_constrained"),
     "Reset, injection, additional constrained training"),
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
    ("INTERVENTION_LAYERS", 2, "the last two layers of the actor (the output layer and the hidden layer before it)"),
    ("FLOOR_SATISFACTION_RATE", 0.05, "a satisfaction rate below 5 percent"),
    ("FLOOR_ARM_SHARE", 0.5, "If more than half of the arms have a satisfaction rate below 5 percent"),
    ("INTERVAL_LEVEL", 0.95, "with two 95 percent intervals"),
    ("INTERVAL_LEVEL_PRIMARY_STUDY_B", 0.975, "an interval that must exclude zero is the 97.5 percent interval"),
    ("IQM_TRIM", 0.25, "the interquartile mean"),
    ("H1_TREND_POINTS", 20, "twenty points: four onset fractions by five seeds"),
    ("POWER_TARGET", 0.80, "has 80 percent power at a standardised difference of 2.02"),
    ("G2_MIN_BUDGETS_SUPPORT", 3, "On at least three of the four unseen budgets"),
    ("G2_MIN_BUDGETS_FALSIFY", 2, "on at least two of the four unseen budgets"),
    ("G5_POINT_MAX", 0.05, "a point estimate of at most 5 percentage points"),
    ("G5_UPPER_MAX", 0.10, "an interval whose upper limit is at most 10"),
    ("SEARCH_SOURCES", ("Google Scholar", "Semantic Scholar", "arXiv", "OpenReview"),
     "Google Scholar, Semantic Scholar, arXiv, OpenReview"),
    ("SEARCH_CITATION_SEED", "Yao et al. (2023)",
     "the citation lists of Yao et al. (2023) and of every paper that cites it"),
    ("SEARCH_ADDED_PHRASES", ('"number of thresholds"', '"spacing"'),
     'each with and without "number of thresholds" and "spacing"'),
]


@pytest.mark.parametrize("name, expected, phrase", CASES, ids=[c[0] for c in CASES])
def test_registered_value_and_source(name: str, expected: object, phrase: str) -> None:
    value = getattr(R, name)
    assert value == expected and type(value) is type(expected), (name, value)
    if isinstance(expected, tuple):  # 1000.0 == 1000: compare the element types too
        assert all(type(a) is type(b) for a, b in zip(value, expected)), (name, value)
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


def test_power_statements_match_part_5_5() -> None:
    text = prereg_text()
    assert "80 percent power at a standardised difference of 2.02, 55 percent at 1.5 and 29 percent at 1.0" in text
    assert "with eight seeds the 80-percent point falls to 1.51, with ten to 1.33, with twelve to 1.20" in text
    assert dict(R.POWER_REGISTERED_N5) == {2.02: 0.80, 1.5: 0.55, 1.0: 0.29}
    assert dict(R.POWER_REGISTERED_D80) == {5: 2.02, 8: 1.51, 10: 1.33, 12: 1.20}


def test_search_queries_are_table_c1_verbatim() -> None:
    text = prereg_text()
    assert "; ".join(R.SEARCH_QUERIES) + "; each with and without" in text
    assert len(R.SEARCH_QUERIES) == 5


def test_transfer_tasks_are_the_registered_point_pair() -> None:
    # Table 2.2: "on the held-out task on the same robot (Goal to Button and Button to Goal)". The Car
    # pair's held-out task is not registered (Q-transfer-obs); it must not appear here.
    assert "the held-out task on the same robot (Goal to Button and Button to Goal)" in prereg_text()
    assert dict(R.TRANSFER_TASKS) == {"SafetyPointGoal1-v0": "SafetyPointButton1-v0",
                                      "SafetyPointButton1-v0": "SafetyPointGoal1-v0"}


RUN_GATE_KEYS = {"Q-rounding", "Q-pilot-unconstrained-seed", "Q-seed-collision", "Q-surplus-arm-set",
                 "Q-data-control-lr", "Q-selection-window", "Q-continuations", "Q-cost-critic", "Q-jc-window",
                 "Q-ramp-step", "Q-warm-start", "Q-rate-limit", "Q-pid-eq9", "Q-reset-injection",
                 "Q-plasticity-definitions", "Q-transfer-obs", "Q-budget-normalisation", "Q-level-jc",
                 "Q-continuous-bins", "Q-search-before-pilot", "Q-studyb-order", "Q-determinism-late-onset"}
RESULT_GATE_KEYS = {"Q-hazard", "Q-dynamics", "Q-studyb-eval", "Q-tie-break", "Q-matched-cost-set",
                    "Q-arm-complete", "Q-controller-quantities", "Q-adapt-censoring"}
REPORT_KEYS = {"Q-interrupted-run", "Q-g1-level", "Q-g3-pairing", "Q-g4-level", "Q-g2-run-equivalents",
               "Q-final-cost-set", "Q-interval", "Q-h1-shape", "Q-controls-reading", "Q-support-falsify-overlap",
               "Q-falsification-calibration", "Q-holm-families", "Q-threshold-arithmetic", "Q-iqm",
               "Q-bootstrap-details", "Q-surplus-in-analysis", "Q-h3-scope", "Q-h4-reading", "Q-h0-scope",
               "Q-g1-criteria", "Q-g2-outcome", "Q-g3-arms", "Q-g4-reading", "Q-floor-rule"}

# PENDING keys registered after HANDOVER.md was last updated: the integrator adds them to section 9 (with a
# star), after which they may be removed from here (a key both here and starred is accepted).
HANDOVER_STAR_PENDING: set[str] = set()
# PENDING keys whose answer the group changed at ratification: removed from ANSWERED_QUESTIONS until their code
# follows the new answer (configs/registered.py, above ANSWERED_QUESTIONS), and listed here meanwhile.
REOPENED_KEYS: set[str] = set()


def test_open_question_kinds_are_disjoint() -> None:
    assert not (RUN_GATE_KEYS & RESULT_GATE_KEYS or RUN_GATE_KEYS & REPORT_KEYS or RESULT_GATE_KEYS & REPORT_KEYS)


def test_open_questions_are_named_and_answers_are_pending_keys() -> None:
    keys = RUN_GATE_KEYS | RESULT_GATE_KEYS | REPORT_KEYS
    assert set(R.PENDING) == keys
    # HANDOVER.md section 9 marks exactly the PENDING keys with a star ("Q-key\*" in Markdown), so a
    # key added to or dropped from PENDING without updating the handover fails here.
    handover = (ROOT / "HANDOVER.md").read_text(encoding="utf-8")
    section_9 = re.search(r"^## 9\..*?(?=^## )", handover, flags=re.S | re.M)
    assert section_9 is not None, "HANDOVER.md has no section 9"
    starred = set(re.findall(r"(Q-[a-z0-9-]+)\\\*", section_9.group(0)))
    assert keys - HANDOVER_STAR_PENDING <= starred <= keys, sorted(starred ^ keys)
    # Only PENDING keys can be answered (HANDOVER.md task 11: answered by an amendment row of Table 9.1),
    # and a key is open exactly while it is not answered.
    assert R.ANSWERED_QUESTIONS <= keys
    assert all(R.is_open(key) == (key not in R.ANSWERED_QUESTIONS) for key in keys)
    with pytest.raises(KeyError):
        R.is_open("Q-unknown")


def test_every_key_is_answered_by_the_decisions_of_2026_10_02() -> None:
    """docs/DECISIONS.md (2026-10-02) answers every PENDING key, to be ratified in Table 9.1: ANSWERED_QUESTIONS is
    an explicit sorted literal of all of them but REOPENED_KEYS (a key the group changes is removed there, and
    listed in REOPENED_KEYS, until its code follows), and the decision record has a section for each."""
    assert REOPENED_KEYS <= set(R.PENDING), sorted(REOPENED_KEYS - set(R.PENDING))
    assert R.ANSWERED_QUESTIONS == frozenset(R.PENDING) - REOPENED_KEYS
    source = (ROOT / "configs" / "registered.py").read_text(encoding="utf-8")
    literal = source[source.index("ANSWERED_QUESTIONS: frozenset[str] = frozenset({"):]
    literal = literal[:literal.index("})")]
    listed = re.findall(r'"(Q-[a-z0-9-]+)"', literal)
    assert listed == sorted(R.ANSWERED_QUESTIONS)
    decisions = (ROOT / "docs" / "DECISIONS.md").read_text(encoding="utf-8")
    assert "2026-10-02" in decisions
    assert [key for key in R.PENDING if f"{key} (starred)" not in decisions] == []


def _pending_text(key: str) -> str:
    return " ".join(R.PENDING[key].split())


def test_pending_texts_carry_what_the_code_applies() -> None:
    """The group ratifies a key from its PENDING text: the text must name what the code implements.

    Every key states its question and then its answer (docs/DECISIONS.md, 2026-10-02), which
    ANSWERED_QUESTIONS accepts. Q-reset-injection: its claim about Nikishin et al. (2023) comes from two
    search-engine extracts of the paper, the PDF unread (metrics/interventions.py, envs/onset.py).
    Q-g4-reading: the ordering clause on the arm statistic (analysis.verdict.Reading.g4_order =
    "arm_median"), 'within one horizon' as a spread below 1 (g4_within = "less_than_one") and the three
    sensitivity readings beside it. Q-continuations: cut 3's shortened continuation keeps the full
    continuation's 1,000,000-step schedule (pilot/manifest.py _cut_fewshot_short).
    Q-controller-quantities: the analysis' censoring of recovery time (analysis.study_a).
    Q-data-control-lr: one run of T + N*T steps whose extension restarts the rate (envs.onset.DataControlLR).
    """
    reset = _pending_text("Q-reset-injection")
    assert "Nikishin et al. (2023, arXiv 2305.15555)" in reset
    assert "two independent search-engine extracts of the paper" in reset and "the PDF could not be opened" in reset
    g4 = _pending_text("Q-g4-reading")
    assert "arm statistic" in g4 and "more than half of those budgets" in g4
    assert "is less than 1" in g4 and "'within' as a spread of at most 1" in g4
    assert "the ordering on a majority of the per-budget medians" in g4
    continuations = _pending_text("Q-continuations")
    assert "cut 3 ('fewshot_short')" in continuations and "80 percent" in continuations
    assert "keeps the 1,000,000-step (50-epoch) schedule and stops after 200,000 steps" in continuations
    recovery = _pending_text("Q-controller-quantities")
    assert "plus one epoch" in recovery and "the shorter arm's horizon" in recovery
    data_control = _pending_text("Q-data-control-lr")
    assert "Answered: one run of T + N*T steps" in data_control and "envs.onset.DataControlLR" in data_control
    # No leftover proposal wording in any case ("proposed:", "Proposed:", "proposal"); the one legitimate use is
    # Q-rate-limit's logged "proposed (pre-clip) value" of the multiplier.
    proposal = re.compile(r"\bpropos\w*\b(?! \(pre-clip\))", re.I)
    probes = ("Proposed: x", "the proposal", "proposals", "The proposed (pre-clip) value")
    assert [bool(proposal.search(text)) for text in probes] == [True, True, True, False]
    assert all(" Answered: " in R.PENDING[key] and not proposal.search(R.PENDING[key]) for key in R.PENDING), sorted(
        key for key in R.PENDING if " Answered: " not in R.PENDING[key] or proposal.search(R.PENDING[key]))

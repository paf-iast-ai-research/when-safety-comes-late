"""Registered constants of the pre-registration, in one place.

Every number that code uses from the pre-registration lives here, with its
location in ``prereg/Preregistration.pdf`` on the same line (First Tasks,
habit 1). No registered number is typed anywhere else in the code; tests in
``tests/test_registered.py`` assert every value below against the text.

Marks (How to Read This Document):
  source      -- taken from a published source; cited beside the constant.
  (decision)  -- chosen by the group (register: Table 8.2); changed only by an
                 amendment recorded in Part 9 (Table 9.1).
  [P]         -- measured by the pilot (Table 8.1); never guessed here.

Values that the pre-registration leaves open are NOT chosen here. They are
listed in ``PENDING`` with the question that must be answered in the
amendment log (Part 9, Table 9.1) before code that depends on them runs.
"""

from __future__ import annotations

from types import MappingProxyType

# ---------------------------------------------------------------------------
# Training and software (Part 3.1, Table 3.1; Appendix A, Table A.1)
# ---------------------------------------------------------------------------

TOTAL_STEPS = 10_000_000  # T; Table 3.1 "Total steps T"; Table A.1 (OmniSafe on-policy benchmark)
STEPS_PER_EPOCH = 20_000  # Table 3.1 "Total steps T"; Table A.1
EPOCHS = 500  # Table 3.1; Table A.1
UPDATE_ITERS = 40  # Table 3.1 "Algorithm"; Table A.1 (published replication)
MINIBATCH_SIZE = 64  # Table 3.1 "Algorithm"; Table A.1 (published replication)
TARGET_KL = 0.02  # Table 3.1 "Algorithm"; Table A.1 (published replication)
COST_LIMIT = 25.0  # d; Table 2.1 "Cost budget d"; Table A.1 (OmniSafe documentation)
LAGRANGE_MULTIPLIER_INIT = 0.001  # Table 2.1 "Abrupt onset"; Table A.1 (OmniSafe documentation)
LAGRANGE_MULTIPLIER_LR = 0.035  # Table 3.1 "Algorithm"; Table A.1 (OmniSafe documentation)
HIDDEN_SIZES = (64, 64)  # Table 3.1 "Networks"; Table A.1
ACTIVATION = "tanh"  # Table 3.1 "Networks"; Table A.1
EPISODE_LENGTH = 1_000  # Table 2.1 "Episode"; Table A.1 (Ji et al., 2023)

DEVICE = "cpu"  # Table 3.1 "Hardware and device"; Table 8.2 "Device and parallelism" (decision)
TORCH_THREADS = 1  # Table 3.1 "Hardware and device"; Table 8.2 (decision); OmniSafe default is 16
VECTOR_ENV_NUMS = 1  # Table 3.1 "Hardware and device"; Table 8.2 (decision)
PARALLEL_PROCESSES = 1  # Table 3.1 "Hardware and device"; Table 8.2 (decision)

TASKS_STUDY_A = ("SafetyPointGoal1-v0", "SafetyCarGoal1-v0", "SafetyPointButton1-v0")  # Table 3.1 "Tasks"
PRIMARY_TASK = "SafetyPointGoal1-v0"  # Table 3.1 "Tasks"; Part 5.2
TASKS_STUDY_B = ("SafetyPointGoal1-v0",)  # Table 3.1 "Tasks"; Table 3.4

# ---------------------------------------------------------------------------
# Checkpoints and logging (Part 3.4; Table 2.3 "Logging points"; Table 8.2)
# ---------------------------------------------------------------------------

CHECKPOINT_INTERVAL_STEPS = 200_000  # Part 3.4; Table 2.3 "Logging points" (decision); Table 8.2
CHECKPOINT_INTERVAL_EPOCHS = 10  # Table 2.3 "Logging points": "Every 200,000 steps, that is every ten epochs"
SELECTION_WINDOW_CHECKPOINTS = 10  # Part 4.1 rule 1: "the run's last ten checkpoints"
SELECTION_WINDOW_STEPS = 2_000_000  # Part 4.1 rule 1: "(the final 2,000,000 steps; decision)"
EVAL_EPISODES = 100  # Table 2.1 "Evaluation cost of a checkpoint" (decision); Table 8.2

# ---------------------------------------------------------------------------
# Study A factors (Part 3.2, Tables 3.2 and 3.3; Table 2.1; Table 2.4)
# ---------------------------------------------------------------------------

ONSET_FRACTIONS = (0.0, 0.10, 0.25, 0.50)  # N; Table 2.1 "Onset fraction N"; Table 3.2
LATE_ONSET_FRACTIONS = (0.10, 0.25, 0.50)  # Table 3.2 (N > 0); Table 3.3
ONSET_SHAPES = ("abrupt", "ramp")  # Table 3.2 "Onset shape"
STEP_MATCHING_CONTROLS = ("constrained_steps", "total_steps")  # Table 3.2; Table 2.4 (last two rows)
SEEDS = (0, 1, 2, 3, 4)  # Table 3.2 "Seeds"
MAIN_ARMS_PER_TASK = 13  # Part 3.2: "1 + 3 x 2 x 2 = 13 main arms"
MAIN_SWEEP_RUNS = 195  # Part 3.2: "the main sweep is 195 runs"

RAMP_WINDOW_FRACTION = 0.10  # W = 0.10 T; Table 2.1 "Linear ramp" (decision); Table 8.2
RAMP_WINDOW_STEPS = 1_000_000  # W = 0.10 x T; Table 2.1 "Linear ramp" (decision); Table 8.2; First Tasks, Role 2 step 3
D_LOOSE = MappingProxyType({  # Table 2.1 "Linear ramp": twice the unconstrained PPO cost (decision)
    "SafetyPointGoal1-v0": 111.0,  # from 55.72 (Table 3.5)
    "SafetyCarGoal1-v0": 116.0,  # from 58.06 (Table 3.5)
    "SafetyPointButton1-v0": 305.0,  # from 152.48 (Table 3.5)
})

TREATMENTS = ("reset", "injection", "additional_constrained")  # Table 2.4; Table 3.3 "Mediation treatments"
CONTROLLER_VARIANTS = ("warm_started", "rate_limited")  # Table 2.4; Table 3.3 "Controller variants"
PID_VARIANT = "pid"  # Table 2.4 "PID multiplier"; Table 3.3 "PID check"
PID_TASKS = ("SafetyPointGoal1-v0",)  # Table 2.4 "PID multiplier"; Table 3.3 "PID check"
TREATMENT_RUNS = 135  # Table 3.3: 3 x 3 x 3 x 5
CONTROLLER_RUNS = 90  # Table 3.3: 2 x 3 x 3 x 5
PID_RUNS = 15  # Table 3.3: 3 x 5

RATE_LIMIT_RELATIVE = 0.05  # Table 2.4 "Rate-limited multiplier" (decision); Table 8.2
RATE_LIMIT_ABSOLUTE = 0.01  # Table 2.4 "Rate-limited multiplier" (decision); Table 8.2
SETTLING_BAND = 0.10  # Table 2.4 "Settling time" (decision); Table 8.2
OVERSHOOT_WINDOW_STEPS = 2_000_000  # Table 2.4 "Multiplier overshoot"
OVERSHOOT_FINAL_FRACTION = 0.10  # Table 2.4 "Multiplier overshoot": "mean over the last 10 percent"

# ---------------------------------------------------------------------------
# Robustness battery (Table 2.2)
# ---------------------------------------------------------------------------

HAZARD_LAYOUTS = 20  # Table 2.2 "Hazard relocation" (decision); Table 8.2
EPISODES_PER_LAYOUT = 5  # Table 2.2 "Hazard relocation" (decision); Table 8.2
BODY_MASS_SCALE = 1.3  # Table 2.2 "Dynamics perturbation" (decision); Table 8.2
GROUND_FRICTION_SCALE = 0.7  # Table 2.2 "Dynamics perturbation" (decision); Table 8.2
FINETUNE_STEPS = 1_000_000  # Table 2.2 "Reward-only fine-tuning" (decision); Table 8.2
TRANSFER_STEPS = 1_000_000  # Table 2.2 "Transfer" (decision); Table 8.2

# ---------------------------------------------------------------------------
# Plasticity metrics (Table 2.3; equations 6 and 7)
# ---------------------------------------------------------------------------

FIXED_BATCH_STATES = 2_048  # Table 2.3 "Fixed evaluation batch" (decision); Table 8.2
FIXED_BATCH_SEED = 0  # Table 2.3 "Fixed evaluation batch": "under seed 0"
DORMANT_THRESHOLD = 0.025  # tau; Table 2.3; equation (6) (Sokar et al., 2023)
EFFECTIVE_RANK_DELTA = 0.01  # delta; Table 2.3; equation (7) (Kumar et al., 2021)
MANIPULATION_CHECK_STEPS_AFTER_ONSET = 200_000  # Part 1.2, box H3 (c)

# ---------------------------------------------------------------------------
# Study B (Table 2.5; Part 3.3, Table 3.4)
# ---------------------------------------------------------------------------

STUDY_B_ARMS = MappingProxyType({  # Table 2.5 "Training set of levels"; Table 3.4 (decision)
    "Single-10": (10.0,),
    "Single-20": (20.0,),
    "Single-40": (40.0,),
    "Sparse": (10.0, 40.0),
    "Moderate": (10.0, 20.0, 40.0),
    "Dense": (10.0, 17.5, 25.0, 32.5, 40.0),
    "Continuous": None,  # uniform on CONTINUOUS_RANGE per episode
})
CONTINUOUS_RANGE = (10.0, 40.0)  # Table 2.5 "Training set of levels"
CONTINUOUS_BIN_WIDTH = 5.0  # Table 2.5: "one multiplier per 5-unit bin" (decision)
UNSEEN_BUDGETS = (5.0, 15.0, 30.0, 45.0)  # Table 2.5 "Unseen budgets" (decision); Table 8.2
BUDGET_OBSERVATION_DIVISOR = 100.0  # Table 2.5 "Budget conditioning": "The active budget divided by 100"
FEWSHOT_STEPS = 1_000_000  # Table 2.5 "Few-shot adaptation" (decision)
FEWSHOT_HORIZONS = (200_000, 500_000, 1_000_000)  # Table 2.5 (decision); Table 8.2
SATISFACTION_TARGET = 0.80  # Table 2.5 "Adaptation steps" (decision); Table 8.2
STUDY_B_RUNS = 35  # Part 3.3: "Seven arms with five seeds are 35 training runs"
STUDY_B_CONTINUATIONS = 140  # Part 3.3: "35 x 4 = 140 continuations"
STUDY_B_RUN_EQUIVALENTS = 49  # Part 3.3: "Study B's total is therefore 49 run-equivalents"

# ---------------------------------------------------------------------------
# Comparison rule (Part 4.1) and effects of interest (Part 1.5)
# ---------------------------------------------------------------------------

MATCH_TOLERANCE = 2.5  # Part 4.1 rule 5 (decision); Table 8.2
INFEASIBLE_REFERENCE_SD = 10.0  # Part 4.1 rule 5; Table 8.2
SENSITIVITY_TOLERANCE = 5.0  # Part 4.1 rule 7
MIN_EFFECT_STUDY_A = 5.0  # Part 1.5 (decision): 5 cost units of gap difference
MIN_EFFECT_STUDY_B = 0.05  # Part 1.5 (decision): 5 percentage points of satisfaction rate

# ---------------------------------------------------------------------------
# Analysis (Part 5)
# ---------------------------------------------------------------------------

ALPHA = 0.05  # Part 5.2 (Holm families at alpha = 0.05); Part 5.5
ALPHA_PRIMARY_STUDY_B = 0.025  # Part 5.2: Bonferroni within the primary family of two (G1, G2)
BOOTSTRAP_RESAMPLES = 10_000  # Part 5.3
NEXT_UNUSED_SEED_START = 5  # Part 5.6: "repeated with the next unused seed (5, 6, and so on)"
SURPLUS_SEED_START = 5  # Part 5.5: "Seeds are added in the order 5, 6, 7, and so on"
MAX_SEEDS_PER_ARM = 12  # Part 5.5: "up to twelve"

# ---------------------------------------------------------------------------
# Pilot (Part 3.6) and go or no-go rule (Part 6)
# ---------------------------------------------------------------------------

PILOT_SEEDS = (0, 1, 2)  # Part 3.6: "seeds 0, 1, 2"
PILOT_STUDY_A_TASK = "SafetyPointGoal1-v0"  # Part 3.6
PILOT_STUDY_A_ONSET_FRACTIONS = (0.0, 0.50)  # Part 3.6: "arms N = 0 and N = 0.50"
PILOT_STUDY_A_SHAPE = "abrupt"  # Part 3.6: "abrupt"
PILOT_STUDY_A_CONTROL = "total_steps"  # Part 3.6: "total-steps matched"
PILOT_BATTERY = ("hazard", "dynamics")  # Part 3.6: "hazard relocation and dynamics perturbation"
PILOT_MODERATE_SEED = 0  # Part 3.6: "one Moderate arm run with seed 0"
PILOT_RUNS = 8  # Part 3.6: "eight runs"

G1_COST_MARGIN = 2.5  # Part 6 G1: "at most d + 2.5"
G1_REFERENCE_SD_MAX = 10.0  # Part 6 G1: "standard deviation of that cost is at most 10"
G2_RUN_EQUIVALENTS = 484  # Part 6 G2: "484 run-equivalents (435 for Study A, 49 for Study B)"
G2_RUN_EQUIVALENTS_STUDY_A = 435  # Part 6 G2
G2_RUN_EQUIVALENTS_STUDY_B = 49  # Part 6 G2
G3_SIGNAL_MIN = 5.0  # Part 6 G3: "at least 5 cost units"
U80_T_QUANTILE = 1.886  # Equation (12): U80 = x + 1.886 s / sqrt(3) (0.90 quantile of t, 2 degrees of freedom)
G4_COST_MARGIN = 2.5  # Part 6 G4: "each at most its budget plus 2.5"

# ---------------------------------------------------------------------------
# Published reference values (Part 3.5, Table 3.5; Ji et al., 2024, Appendix B)
# ---------------------------------------------------------------------------

BENCHMARK_UNCONSTRAINED_PPO_COST = MappingProxyType({  # Table 3.5
    "SafetyPointGoal1-v0": 55.72,
    "SafetyCarGoal1-v0": 58.06,
    "SafetyPointButton1-v0": 152.48,
})

# ---------------------------------------------------------------------------
# Open questions: values the text does not fix (First Tasks, habit 4). A run that
# depends on one carries its key in ``RunSpec.pending`` and is not launched until the
# amendment log answers it. The go-report questions (Q-g1-level, Q-g3-pairing,
# Q-g4-level), Q-g2-run-equivalents and Q-interrupted-run (applied with `schedule set-policy`)
# gate no run. For the go-report keys the report states both readings, and Q-g1-level and
# Q-g3-pairing leave their condition UNDECIDED while the readings disagree (Q-g3-pairing also
# when a replacement seed leaves fewer than three pairs).
# Keys are stable identifiers used in error messages and in HANDOVER.md.
# ---------------------------------------------------------------------------

PENDING = MappingProxyType({
    "Q-rounding": "Table 2.4: onset step and total of constrained-steps-matched arms at N = 0.10 and 0.25 "
                  "(proposed: onset to the nearest whole epoch, then exactly 10,000,000 constrained steps).",
    "Q-pilot-unconstrained-seed": "Part 3.6: the seed of the Study B pilot's unconstrained PPO run is not stated "
                                  "(proposed: seed 0, as for the Moderate pilot run).",
    "Q-interrupted-run": "Part 5.6: whether a run stopped by the machine (power loss, reboot, kill) counts as a "
                         "crash, or is restarted from scratch with the same seed (proposed: restart; runs are "
                         "bit-for-bit deterministic, so the restart reproduces the uninterrupted run).",
    "Q-seed-collision": "Parts 5.5 and 5.6: replacement seeds and surplus seeds both start at 5, and pilot arms "
                        "use seeds 0-2 only (plain 'next unused' would be 3) (proposed: every arm, pilot "
                        "included, takes the smallest seed >= 5 it has not used, for either purpose).",
    "Q-g3-pairing": "Equation (12): U80 uses s / sqrt(3) with 2 degrees of freedom, which is the paired form "
                    "over the three seeds (proposed: s is the SD of the three per-seed differences). If a "
                    "replacement seed leaves fewer than three pairs, the paired U80 is undefined (proposed: "
                    "unless the mean clause decides G3, the group decides it; the report gives the pooled "
                    "form beside it).",
    "Q-surplus-arm-set": "Part 5.5: which N = 0.50 arms receive surplus seeds (onset shape is not named) "
                         "(proposed: both shapes, both controls, plus N = 0, on SafetyPointGoal1-v0). Surplus "
                         "seeds of the ramp arms, which the proposal adds, wait on this key.",
    "Q-selection-window": "Part 4.1 rule 1: 'the last ten checkpoints' and 'the final 2,000,000 steps' "
                          "differ when a run's total is off the 200,000-step grid (11,120,000; 13,340,000; "
                          "12,500,000 steps) (proposed: a checkpoint grid relative to the end of training).",
    "Q-g1-level": "Part 6 G1: 'Both pilot arms reach a final in-distribution cost ... of at most d + 2.5' is "
                  "read on each arm's mean over its seeds (as Part 4.1 defines an arm's matched cost) or on "
                  "every seed (proposed: the arm mean; the per-seed result is reported beside it).",
    "Q-g4-level": "Part 6 G4 says 'each at most its budget plus 2.5'; Part 3.6 says 'each within 2.5 of its "
                  "budget' (two-sided) (proposed: Part 6 decides; the two-sided result is reported beside it).",
    "Q-continuations": "Tables 2.2 and 2.5: learning rate and schedule, critics, optimiser states and normaliser of "
                       "every continuation run (fine-tuning, transfer, Study B few-shot); OmniSafe's actor learning "
                       "rate is 0 at the end of training.",
    "Q-g2-run-equivalents": "Part 6 G2: the 484 run-equivalents count Study A's 435 runs as one each, although "
                            "constrained-steps-matched arms train up to 2T and the data control T + N*T; with the "
                            "battery's continuations the total is about 627 (non-blocking: G2 is decided on 484, the "
                            "corrected figure is reported beside it until the group amends G2).",
    "Q-data-control-lr": "Table 2.4 'Additional constrained training': OmniSafe decays the actor learning rate "
                         "linearly to 0 over the run's own epochs (linear_lr_decay: True). Run as one run of "
                         "T + N*T steps (as the manifest specifies), its first T steps decay the rate more slowly "
                         "(a higher rate at every epoch after the first) than the untreated late arm's; run as a "
                         "continuation after T, the rate is 0 and the actor cannot move. Either way 'no change to the "
                         "network or optimiser' cannot hold; the schedule must be fixed.",
})

# Keys of PENDING that the amendment log (Table 9.1) has answered with the proposal stated in PENDING.
# Add a key here, in the same pull request that records the amendment, to release the runs that
# waited for it. If the group answers differently from the proposal, change the code that
# implements the proposal (and the PENDING text) to the answer, then add the key in the same pull
# request (a changed run spec is refused by `schedule add`; update the queue deliberately).
ANSWERED_QUESTIONS: frozenset[str] = frozenset()


def is_open(key: str) -> bool:
    """True while a PENDING question has not been answered in the amendment log."""
    if key not in PENDING:
        raise KeyError(f"unknown open question {key!r}")
    return key not in ANSWERED_QUESTIONS

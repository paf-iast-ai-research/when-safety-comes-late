"""Registered constants of the pre-registration, in one place.

Every number that code uses from the pre-registration lives here, with its
location in ``prereg/Preregistration.pdf`` on the same line (First Tasks,
habit 1). No registered number is typed anywhere else in the code, except in
the self-contained ledger schema ``results/ledger_schema.py`` (frozen by its
hash), which keeps its own copies of the values it validates
(``tests/test_ledger_schema.py`` asserts the Study B arms' levels and the
Continuous range and bin width against these); tests in
``tests/test_registered.py`` assert every value below against the text.

Marks (How to Read This Document):
  source      -- taken from a published source; cited beside the constant.
  (decision)  -- chosen by the group (register: Table 8.2); changed only by an
                 amendment recorded in Part 9 (Table 9.1).
  [P]         -- measured by the pilot (Table 8.1); never guessed here.

Values that the pre-registration leaves open and that gate a run or a result
are listed in ``PENDING`` with their question and the answer the code
implements. Every key is answered in docs/DECISIONS.md (2026-10-02), to be
ratified by the group in the amendment log (Part 9, Table 9.1);
``ANSWERED_QUESTIONS`` lists them, which releases the code that waited for
them. A few implementation-level readings (Q-eval-seeds, Q-onset-epoch,
Q-first-checkpoint, Q-mujoco-exception) gate nothing; they are answered as
notes of Table 9.1 in docs/DECISIONS.md and listed in HANDOVER.md section 9.
"""

from __future__ import annotations

from types import MappingProxyType

# ---------------------------------------------------------------------------
# Training and software (Part 3.1, Table 3.1; Appendix A, Table A.1)
# ---------------------------------------------------------------------------

TOTAL_STEPS = 10_000_000  # T; Table 3.1 "Total steps T"; Table A.1 (OmniSafe on-policy benchmark)
STEPS_PER_EPOCH = 20_000  # Table 3.1 "Total steps T" and "Algorithm"; Table A.1
EPOCHS = 500  # Table 3.1 "Total steps T" and "Algorithm"; Table A.1
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
SELECTION_WINDOW_CHECKPOINTS = 10  # Part 4.1 rule 1: "the run's last ten checkpoints"; Table 8.2
SELECTION_WINDOW_STEPS = 2_000_000  # Part 4.1 rule 1: "(the final 2,000,000 steps; decision)"; Table 8.2
EVAL_EPISODES = 100  # Table 2.1 "Evaluation cost of a checkpoint" (decision); Table 8.2

# ---------------------------------------------------------------------------
# Study A factors (Part 3.2, Tables 3.2 and 3.3; Table 2.1; Table 2.4)
# ---------------------------------------------------------------------------

ONSET_FRACTIONS = (0.0, 0.10, 0.25, 0.50)  # N; Table 2.1 "Onset fraction N"; Table 3.2
LATE_ONSET_FRACTIONS = (0.10, 0.25, 0.50)  # Table 3.2 (N > 0); Table 3.3
ONSET_SHAPES = ("abrupt", "ramp")  # Table 3.2 "Onset shape"
STEP_MATCHING_CONTROLS = ("constrained_steps", "total_steps")  # Table 3.2; Table 2.4 (last two rows)
SEEDS = (0, 1, 2, 3, 4)  # Table 3.2 "Seeds"
MAIN_ARMS_PER_TASK = 13  # Part 3.2: "1 + 3 × 2 × 2 = 13 main arms"
MAIN_SWEEP_RUNS = 195  # Part 3.2: "the main sweep is 195 runs"

RAMP_WINDOW_FRACTION = 0.10  # W = 0.10 T; Table 2.1 "Linear ramp" (decision); Table 8.2
RAMP_WINDOW_STEPS = 1_000_000  # W = 0.10 x T; Table 2.1 "Linear ramp" (decision); Table 8.2; First Tasks, Role 2 step 3
D_LOOSE = MappingProxyType({  # Table 2.1 "Linear ramp": twice the unconstrained PPO cost (decision); Table 8.2
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
# Table 2.2 "Transfer": "the held-out task on the same robot (Goal to Button and Button to Goal)". Only
# the Point pair is named by the tasks of Study A; SafetyCarGoal1-v0's held-out task is not registered:
# Q-transfer-obs answers it (SafetyCarButton1-v0; docs/DECISIONS.md, to be ratified in Table 9.1), and the
# answer lives in pilot/manifest.py (TRANSFER_TASK_PROPOSALS), not here.
TRANSFER_TASKS = MappingProxyType({  # Table 2.2 "Transfer"
    "SafetyPointGoal1-v0": "SafetyPointButton1-v0",
    "SafetyPointButton1-v0": "SafetyPointGoal1-v0",
})

# ---------------------------------------------------------------------------
# Plasticity metrics (Table 2.3; equations 6 and 7)
# ---------------------------------------------------------------------------

FIXED_BATCH_STATES = 2_048  # Table 2.3 "Fixed evaluation batch" (decision); Table 8.2
FIXED_BATCH_SEED = 0  # Table 2.3 "Fixed evaluation batch": "under seed 0"
DORMANT_THRESHOLD = 0.025  # tau; Table 2.3; equation (6) (Sokar et al., 2023)
EFFECTIVE_RANK_DELTA = 0.01  # delta; Table 2.3; equation (7) (Kumar et al., 2021)
MANIPULATION_CHECK_STEPS_AFTER_ONSET = 200_000  # Part 1.2, box H3 (c)
INTERVENTION_LAYERS = 2  # Table 2.4 "Partial reset" / "Plasticity injection": "the last two layers of the actor"

# ---------------------------------------------------------------------------
# Study B (Table 2.5; Part 3.3, Table 3.4)
# ---------------------------------------------------------------------------

STUDY_B_ARMS = MappingProxyType({  # Table 2.5 "Training set of levels"; Table 3.4 (decision); Table 8.2
    "Single-10": (10.0,),
    "Single-20": (20.0,),
    "Single-40": (40.0,),
    "Sparse": (10.0, 40.0),
    "Moderate": (10.0, 20.0, 40.0),
    "Dense": (10.0, 17.5, 25.0, 32.5, 40.0),
    "Continuous": None,  # uniform on CONTINUOUS_RANGE per episode
})
CONTINUOUS_RANGE = (10.0, 40.0)  # Table 2.5 "Training set of levels"
CONTINUOUS_BIN_WIDTH = 5.0  # Table 2.5: "one multiplier per 5-unit bin" (decision); Table 8.2
UNSEEN_BUDGETS = (5.0, 15.0, 30.0, 45.0)  # Table 2.5 "Unseen budgets" (decision); Table 8.2
BUDGET_OBSERVATION_DIVISOR = 100.0  # Table 2.5 "Budget conditioning": "The active budget divided by 100"; Table 8.2
FEWSHOT_STEPS = 1_000_000  # Table 2.5 "Few-shot adaptation" (decision)
FEWSHOT_HORIZONS = (200_000, 500_000, 1_000_000)  # Table 2.5 (decision); Table 8.2
SATISFACTION_TARGET = 0.80  # Table 2.5 "Adaptation steps" (decision); Table 8.2
STUDY_B_RUNS = 35  # Part 3.3: "Seven arms with five seeds are 35 training runs"
STUDY_B_CONTINUATIONS = 140  # Part 3.3: "35 × 4 = 140 continuations"
STUDY_B_RUN_EQUIVALENTS = 49  # Part 3.3: "Study B's total is therefore 49 run-equivalents"

# ---------------------------------------------------------------------------
# Comparison rule (Part 4.1) and effects of interest (Part 1.5)
# ---------------------------------------------------------------------------

MATCH_TOLERANCE = 2.5  # Part 4.1 rule 5 (decision); Table 8.2
INFEASIBLE_REFERENCE_SD = 10.0  # Part 4.1 rule 5; Table 8.2
SENSITIVITY_TOLERANCE = 5.0  # Part 4.1 rule 7
MIN_EFFECT_STUDY_A = 5.0  # Part 1.5 (decision): 5 cost units of gap difference
MIN_EFFECT_STUDY_B = 0.05  # Part 1.5 (decision): 5 percentage points of satisfaction rate
FLOOR_SATISFACTION_RATE = 0.05  # Part 4.2 floor rule: "a satisfaction rate below 5 percent"
FLOOR_ARM_SHARE = 0.5  # Part 4.2 floor rule: "more than half of the arms"

# ---------------------------------------------------------------------------
# Analysis (Part 5)
# ---------------------------------------------------------------------------

ALPHA = 0.05  # Part 5.2 (Holm families at alpha = 0.05); Part 5.5
ALPHA_PRIMARY_STUDY_B = 0.025  # Part 5.2: Bonferroni within the primary family of two (G1, G2)
BOOTSTRAP_RESAMPLES = 10_000  # Part 5.3
NEXT_UNUSED_SEED_START = 5  # Part 5.6: "repeated with the next unused seed (5, 6, and so on)"
SURPLUS_SEED_START = 5  # Part 5.5: "Seeds are added in the order 5, 6, 7, and so on"
MAX_SEEDS_PER_ARM = 12  # Part 5.5: "up to twelve"
INTERVAL_LEVEL = 0.95  # Part 5.3: "with two 95 percent intervals"
INTERVAL_LEVEL_PRIMARY_STUDY_B = 0.975  # Part 5.2: for G1 and G2 "the 97.5 percent interval"
IQM_TRIM = 0.25  # Part 5.3: "the interquartile mean" (Agarwal et al., 2021): the middle 50 percent
H1_TREND_POINTS = 20  # Part 5.4: "twenty points: four onset fractions by five seeds"
POWER_TARGET = 0.80  # Part 5.5: "80 percent power"
# Part 5.5: "80 percent power at a standardised difference of 2.02, 55 percent at 1.5 and 29 percent at 1.0";
# "with eight seeds the 80-percent point falls to 1.51, with ten to 1.33, with twelve to 1.20"
POWER_REGISTERED_N5 = MappingProxyType({2.02: 0.80, 1.5: 0.55, 1.0: 0.29})  # Part 5.5: d -> power at five seeds
POWER_REGISTERED_D80 = MappingProxyType({5: 2.02, 8: 1.51, 10: 1.33, 12: 1.20})  # Part 5.5: seeds -> d at 80 percent

# ---------------------------------------------------------------------------
# Hypothesis boxes of Study B (Part 1.4)
# ---------------------------------------------------------------------------

G2_MIN_BUDGETS_SUPPORT = 3  # box G2: "On at least three of the four unseen budgets"
G2_MIN_BUDGETS_FALSIFY = 2  # box G2: "on at least two of the four unseen budgets"
G5_POINT_MAX = 0.05  # box G5: "a point estimate of at most 5 percentage points"
G5_UPPER_MAX = 0.10  # box G5: "an interval whose upper limit is at most 10"

# ---------------------------------------------------------------------------
# Targeted search for Study B (Appendix C, Table C.1)
# ---------------------------------------------------------------------------

SEARCH_SOURCES = ("Google Scholar", "Semantic Scholar", "arXiv", "OpenReview")  # Table C.1 "Sources"
SEARCH_CITATION_SEED = "Yao et al. (2023)"  # Table C.1 "Sources": "the citation lists of Yao et al. (2023)
#   and of every paper that cites it"
SEARCH_QUERIES = (  # Table C.1 "Queries", verbatim
    '"constraint-conditioned" generalization threshold',
    '"cost threshold" zero-shot safe reinforcement learning',
    '"budget-conditioned" policy unseen',
    "versatile safe reinforcement learning coverage",
    '"cost limit" curriculum generalization unseen',
)
SEARCH_ADDED_PHRASES = ('"number of thresholds"', '"spacing"')  # Table C.1: "each with and without ... and ..."

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
# Open questions: values the text does not fix (First Tasks, habit 4). Every key is answered in
# docs/DECISIONS.md (2026-10-02), to be ratified by the group in Table 9.1 (ANSWERED_QUESTIONS below);
# each text states the question and its answer, which the code implements. The kind of a key says what
# it holds while it is open (a key the group changes is open again until its code follows the new answer):
#   run gates    -- a run that depends on the answer carries the key in ``RunSpec.pending``
#                   (pilot/manifest.py) and is not launched while the key is open;
#   result gates -- the function that produces a result depending on the answer calls
#                   ``pilot.errors.require_answered`` and refuses (PendingQuestionError) while the
#                   key is open (Q-hazard, Q-dynamics, Q-studyb-eval, Q-tie-break, Q-matched-cost-set,
#                   Q-arm-complete, Q-controller-quantities, Q-adapt-censoring);
#   report keys  -- they gate no run; while one is open, the go report or the analysis shows every
#                   verdict that depends on it as UNDECIDED or PROVISIONAL (the go-report keys and the
#                   analysis keys of Parts 1, 4 and 5, which freeze at the go decision), except
#                   Q-g4-level and Q-g2-run-equivalents (below).
# The go-report questions (Q-g1-level, Q-g3-pairing, Q-g4-level, Q-final-cost-set, and the result
# gate Q-matched-cost-set), Q-g2-run-equivalents and Q-interrupted-run (applied with
# `schedule set-policy`) gate no run. The go report states the other reading beside the answer (the per-seed
# G1, the selection-set costs, the two-sided G4 of Q-g4-level, the registered 484 of Q-g2-run-equivalents,
# the pooled U80); while a key is open, Q-g1-level, Q-matched-cost-set (G1) and Q-final-cost-set (G4) leave
# their condition UNDECIDED while the readings disagree, Q-threshold-arithmetic makes G1 and G3 provisional,
# Q-g3-pairing makes G3 provisional, Q-hazard leaves G3 UNDECIDED, and Q-g4-level and Q-g2-run-equivalents
# leave G4 and G2 decided by the answer the code implements, the other reading only reported beside it.
# Answered, every go condition is decided by its rule: Q-g3-pairing pairs the runs a replacement
# seed leaves unpaired in seed order (pilot.go_decision._seed_pairs), never the group.
# Keys are stable identifiers used in error messages and in HANDOVER.md.
# ---------------------------------------------------------------------------

PENDING = MappingProxyType({
    # --- First keys. Run gates: Q-rounding, Q-pilot-unconstrained-seed, Q-seed-collision, Q-surplus-arm-set,
    # --- Q-selection-window, Q-continuations, Q-data-control-lr. Report keys (they gate no run):
    # --- Q-interrupted-run, Q-g3-pairing, Q-g1-level, Q-g4-level, Q-g2-run-equivalents.
    "Q-rounding": "Table 2.4: onset step and total of constrained-steps-matched arms at N = 0.10 and 0.25. "
                  "Answered: the exact onset N*T/(1 - N) is rounded to the nearest whole 20,000-step epoch, and the "
                  "run then trains exactly T = 10,000,000 constrained steps: N = 0.10, onset 1,120,000 (epoch 56), "
                  "total 11,120,000; N = 0.25, onset 3,340,000 (epoch 167), total 13,340,000 (realised onset "
                  "fractions 0.1007 and 0.2504). The arms keep their registered labels N = 0.10 and 0.25 in every "
                  "analysis, so the H1 Spearman trend uses the registered levels; N = 0.50 is exact (onset "
                  "10,000,000, total 20,000,000).",
    "Q-pilot-unconstrained-seed": "Part 3.6: the seed of the Study B pilot's unconstrained PPO run is not stated. "
                                  "Answered: seed 0, the seed of the Moderate pilot run (P-B-unconstrained-s0), "
                                  "also in the re-pilot (P1-); fixed before any data exist and not chosen again "
                                  "after a result is seen.",
    "Q-interrupted-run": "Part 5.6: whether a run stopped by the machine (power loss, reboot, kill) counts as a "
                         "crash, or is restarted from scratch with the same seed. Answered: a run stopped from "
                         "outside (power loss, reboot, shutdown, SIGTERM, SIGINT or SIGHUP; or a run that ended while "
                         "no scheduler was its parent) is not an exclusion: it is restarted from scratch with the same "
                         "seed and configuration on the commit of its interrupted attempt, with no limit, every "
                         "attempt logged and counted in the run's ledger notes. A run that ends without a final "
                         "training result while a live scheduler watched it (SIGKILL, e.g. from the OOM killer, "
                         "another non-crash signal, or a non-zero exit without train_result) is also restarted with "
                         "the same seed; after MAX_RESTARTS = 2 such restarts (pilot.scheduler.MAX_RESTARTS) a third "
                         "such ending counts as 'fails to complete its steps' (Part 5.6, cause 'incomplete'): the run "
                         "is excluded, recorded with its cause and repeated with the next unused seed. Registered data "
                         "roots run under `schedule set-policy restart`, the policy 'exclude' is refused on them, and "
                         "no interrupted run waits for a group decision.",
    "Q-seed-collision": "Parts 5.5 and 5.6: replacement seeds and surplus seeds both start at 5, and pilot arms "
                        "use seeds 0-2 only (plain 'next unused' would be 3). Answered: every arm, pilot arms "
                        "included, takes for a replacement (Part 5.6) and for a surplus seed (Part 5.5) the smallest "
                        "seed >= 5 it has not used for any purpose (pilot.scheduler.Scheduler.next_unused_seed). "
                        "Surplus seeds are added with `schedule add-surplus` at the go decision, before any "
                        "main-study run of those arms can be excluded, so they are 5, 6, ... as Part 5.5 orders them "
                        "and later replacements follow them; arms pair on equal seed ids (Part 5.1).",
    "Q-g3-pairing": "Equation (12): U80 uses s / sqrt(3) with 2 degrees of freedom, which is the paired form "
                    "over the three seeds; a replacement seed can leave fewer than three seed-matched pairs. "
                    "Answered: s is the sample SD of the three per-seed differences gap_hazard(N = 0.50) - "
                    "gap_hazard(N = 0), paired by seed id (t quantile 0.90, 2 degrees of freedom). If a "
                    "replacement seed leaves fewer than three seed-matched pairs, the runs left unpaired are paired "
                    "in increasing seed order (each arm's remaining runs sorted by seed, then matched one to one) "
                    "and equation (12) is applied unchanged to the three differences (pilot.go_decision._seed_pairs, "
                    "analysis.stats.g3_pairs); the mean Delta(0.50) is the same under any pairing, the pairing "
                    "used is reported, and G3 is decided by the rule, never by the group. The pooled-SD value is "
                    "reported only as a descriptive figure, not as a reading of equation (12).",
    "Q-surplus-arm-set": "Part 5.5: which N = 0.50 arms receive surplus seeds (onset shape is not named). "
                         "Answered: every arm the passage lists, no onset shape excluded: on SafetyPointGoal1-v0 "
                         "the N = 0 arm and the four N = 0.50 arms (abrupt and ramp, each under total-steps and "
                         "constrained-steps matching), and the seven Study B arms with their few-shot "
                         "continuations. Every arm in the set receives the same number of surplus seeds, and its "
                         "seed target is 5 plus that number (Q-arm-complete). Gates the surplus seeds of the ramp "
                         "arms (pilot.manifest.surplus_spec).",
    "Q-selection-window": "Part 4.1 rule 1: 'the last ten checkpoints' and 'the final 2,000,000 steps' "
                          "differ when a run's total is off the 200,000-step grid (11,120,000; 13,340,000; "
                          "12,500,000 steps). Answered: the window is the ten checkpoints at total - k x 200,000 "
                          "steps, k = 0..9, for every run: on the grid, the grid's last ten checkpoints; off the "
                          "grid, an end-relative grid that the run saves in addition to its 200,000-step grid. They "
                          "are both 'the last ten checkpoints' and 'the final 2,000,000 steps'.",
    "Q-g1-level": "Part 6 G1: 'Both pilot arms reach a final in-distribution cost ... of at most d + 2.5' is "
                  "read on each arm's mean over its seeds (as Part 4.1 defines an arm's matched cost) or on "
                  "every seed. Answered: on each arm's mean over its three seeds of the selected checkpoints' "
                  "cost, as Part 4.1 defines an arm's matched cost and the reference cost, on the evaluation set "
                  "of Q-matched-cost-set, compared with 27.5 under the arithmetic of Q-threshold-arithmetic; the "
                  "per-seed result is reported beside it and decides nothing.",
    "Q-g4-level": "Part 6 G4 says 'each at most its budget plus 2.5'; Part 3.6 says 'each within 2.5 of its "
                  "budget' (two-sided). Answered: Part 6 decides: the Moderate arm's mean cost at each of its "
                  "training budgets (10, 20, 40) is at most that budget plus 2.5; the two-sided reading of Part "
                  "3.6 is reported beside it and decides nothing.",
    "Q-continuations": "Tables 2.2 and 2.5: parent checkpoint, learning rate and schedule, critics, optimiser states "
                       "and normaliser of every continuation run (fine-tuning, transfer, Study B few-shot); "
                       "OmniSafe's actor learning rate is 0 at the end of training; and under Part 6.1 cut 3 "
                       "('fewshot_short') a shortened few-shot continuation that decayed its rate over its own "
                       "200,000 steps would reach 0 at its only horizon, where the full 1,000,000-step continuation "
                       "is still at 80 percent. Answered: fine-tuning and transfer start from the parent's matched "
                       "checkpoint (ledger matched_checkpoint_step), few-shot from the parent's final checkpoint; "
                       "every continuation restores the parent's actor, both critics and observation normaliser, "
                       "which keeps updating, with fresh Adam states, and the actor learning rate decays linearly "
                       "(OmniSafe's LinearLR, 1.0 to 0.0) over the continuation's full registered length, 50 epochs "
                       "(1,000,000 steps). Fine-tuning holds the multiplier at 0 and trains the actor on the reward "
                       "advantage only; transfer and few-shot start a fresh multiplier at 0.001 (lr 0.035; d = 25 "
                       "for transfer, the unseen budget for few-shot); every battery continuation trains with "
                       "PPO-Lagrangian, a PID-check parent's included. Under cut 3 a shortened few-shot continuation "
                       "keeps the 1,000,000-step (50-epoch) schedule and stops after 200,000 steps (10 epochs), so "
                       "its only reading is the full continuation's first-horizon reading (pilot/manifest.py "
                       "_cut_fewshot_short, params lr_schedule_steps).",
    "Q-g2-run-equivalents": "Part 6 G2: the 484 run-equivalents count Study A's 435 runs as one each, although "
                            "constrained-steps-matched arms train up to 2T and the data control T + N*T; with the "
                            "battery's continuations the total is about 627. Answered: G2 is amended before the "
                            "pilot: the run-equivalents in G2 and in Part 5.5's surplus rule are training steps in "
                            "units of T = 10,000,000 (as Parts 3.3 and 6.1 count them), so the decisive requirement "
                            "is mean wall-clock hours per pilot run / runs sustained concurrently x the corrected "
                            "total of pilot.budget.corrected_run_equivalents() for the design in force (627.13 for "
                            "the uncut design: 540.13 of training runs plus 87.0 for the battery's fine-tuning and "
                            "transfer continuations). The registered 484 (435 + 49) is reported beside it and "
                            "decides nothing; the surplus seeds are computed from capacity beyond the corrected "
                            "total, each priced in the same units.",
    "Q-data-control-lr": "Table 2.4 'Additional constrained training': OmniSafe decays the actor learning rate "
                         "linearly to 0 over the run's own epochs (linear_lr_decay: True). Run as one run of "
                         "T + N*T steps with that decay, its first T steps would decay the rate more slowly than the "
                         "untreated late arm's; run as a continuation after T, the rate is 0 and the actor cannot "
                         "move. Either way 'no change to the network or optimiser' cannot hold; the schedule must be "
                         "fixed. Answered: one run of T + N*T steps whose first T steps are, bit for bit, the "
                         "untreated late arm of the same task and seed: the actor's rate follows OmniSafe's "
                         "LinearLR(start 1, end 0, total_iters = 500) for epochs 0-499, and in the N*T/20,000 further "
                         "epochs e = 500, ..., 500 + N*T/20,000 - 1 it is "
                         "3e-4 x (1 - N) x (1 - (e - 500)/(N*T/20,000)), "
                         "restarting at the rate at which the arm entered its constrained phase and decaying linearly "
                         "to 0 at the end of the run (envs.onset.DataControlLR; the schedule key is on the data "
                         "control's spec only). Nothing else changes: the network, the Adam moments, both critics "
                         "(constant LR), the multiplier and its Adam state, the normaliser and every random stream "
                         "continue, and the matched-checkpoint selection reads the run's last ten checkpoints like "
                         "every other arm's.",
    # --- Keys added with the code of Roles 2-5 (2026-10-01). Run gates: the specs that depend on the
    # --- answer carry the key in RunSpec.pending (pilot/manifest.py).
    "Q-cost-critic": "Table 2.1 'Constraint onset' does not say whether the cost critic trains before onset (First "
                     "Tasks, Table 9). Answered: both critics (reward and cost) train from the first epoch in every "
                     "Study A run (OmniSafe use_cost: True); at onset only the actor's surrogate (A_R to equation (3)) "
                     "and the multiplier update (equation (4)) change; before onset the cost advantage is computed "
                     "but does not enter the actor's loss. Gates every Study A run with an onset after step 0.",
    "Q-jc-window": "Equation (4) defines J_C as 'the mean episode cost of the current epoch'; Table 3.1 registers "
                   "'PPO-Lagrangian as implemented in OmniSafe', whose multiplier update reads the mean cost of the "
                   "last 50 finished episodes (2.5 epochs). Answered: J_C in the multiplier update (eq. 4) and the "
                   "error of the PID update (eq. 9) is OmniSafe's Metrics/EpCost, the mean episodic cost of the last "
                   "50 finished episodes (window_length = 50), read unmodified by PPOLag._update and "
                   "CPPOPID._update; the per-epoch batch mean over the epoch's own 20 episodes is logged beside it "
                   "as Metrics/BatchEpCost and is the quantity used for recovery time and for every training-batch "
                   "cost Part 3.4 requires. The gloss of J_C in Part 2.5 is corrected by amendment to this "
                   "definition. Gates every Lagrangian run.",
    "Q-ramp-step": "Equation (5) is written for a step t, but the multiplier is updated once per epoch. Answered: "
                   "the update of epoch e (0-based) evaluates equation (5) at t = e x 20,000, the first step of the "
                   "epoch whose data the update uses; N*T is the run's own onset step (the rounded onset for "
                   "constrained-steps arms, Q-rounding); W = 1,000,000 steps in every ramp arm, whatever its total. "
                   "The onset epoch's update uses d_loose, the following 49 updates fall linearly, and every update "
                   "from onset + 1,000,000 steps uses d = 25; only the budget in the multiplier update changes. "
                   "Gates the ramp runs.",
    "Q-warm-start": "Table 2.4 'Warm-started multiplier': copy only the multiplier or also its optimiser state, and "
                    "from which record of the N = 0 run. Answered: only the multiplier value, with a fresh "
                    "(never-stepped) Adam state: the value the N = 0 run of the same task and seed logged in "
                    "progress.csv after the update of epoch onset/E - 1 (cross-checked against that run's checkpoint "
                    "when one exists), set before the onset epoch's multiplier update. A replacement seed whose N = 0 "
                    "run the N = 0 arm never received gets an auxiliary N = 0 run of that seed (`schedule resolve RUN "
                    "add-auxiliary`), left out of the N = 0 arm by the matching and the analysis. If the N = 0 run a "
                    "warm-started run depends on is excluded (Part 5.6), no value is taken from it: the blocked "
                    "warm-started run, which never trained, is not excluded but superseded by its arm's next unused "
                    "seed (`schedule resolve RUN replace`); an excluded auxiliary run is never repeated or replaced "
                    "itself. Gates the warm-started runs.",
    "Q-rate-limit": "Table 2.4 'Rate-limited multiplier': how the clip combines with OmniSafe's Adam update, which "
                    "moves the multiplier by about 0.035 per epoch whatever the violation. Answered: OmniSafe's own "
                    "update runs unchanged (Adam, lr 0.035, on -lambda(J_C - d), clamped to [0, upper bound]); after "
                    "each step the result is clipped to [old - b, old + b], b = 0.05 x old + 0.01, old being the "
                    "value before that epoch's update, then clamped at 0; Adam's moments are not altered by the clip. "
                    "The proposed (pre-clip) value is logged each epoch as Onset/MultiplierProposed, and the report "
                    "gives the share of constrained epochs in which the clip bound "
                    "(metrics.controller.rate_limit_clip_share). While the violation is steady the clip binds only "
                    "while lambda is below about 0.5, and after a sudden rise in the violation (Adam's step about "
                    "2.5-3 x 0.035) also at larger lambda; the rate-limited arm stays the secondary H4 check, as "
                    "registered. Gates the rate-limited runs.",
    "Q-pid-eq9": "Equation (9) differs from OmniSafe's PIDLagrangian (moving-average P term, delayed D term, "
                 "penalty starting at 0.0), and the text does not say what the PID controller does before onset. "
                 "Answered: the PID arms run OmniSafe 0.5.0's CPPOPID/PIDLagrangian unmodified from onset with the "
                 "pinned CPPOPID.yaml gains (kp 0.1, ki 0.01, kd 0.01, d_delay 10, P and D EMA alpha 0.95, sum_norm "
                 "True, diff_norm False, penalty_max 100, init 0.001, cost limit 25), every other setting as "
                 "PPOLag.yaml, the error J_C - d with J_C of Q-jc-window; before onset pid_update is not called "
                 "(penalty 0.0, integral 0.001, both EMAs 0; the surrogate is A_R). The PID arms are abrupt and "
                 "total-steps matched on SafetyPointGoal1-v0 at N = 0.10, 0.25 and 0.50. The amendment rewrites "
                 "eq. (9) in OmniSafe's form: with e_k = J_C,k - d, P_k = 0.95 P_k-1 + 0.05 e_k, D_k = 0.95 D_k-1 "
                 "+ 0.05 J_C,k, I_k = max(0, I_k-1 + 0.01 e_k) (P_0 = D_0 = 0, I_0 = 0.001), lambda_k = max(0, "
                 "0.1 P_k + I_k + 0.01 max(0, D_k - D_max(k-10, 0))), with no penalty_max cap while sum_norm is "
                 "True, and the surrogate (A_R - lambda A_C)/(1 + lambda). Gates the PID runs.",
    "Q-reset-injection": "Table 2.4 'Partial reset' and 'Plasticity injection' leave open: whether log_std is "
                         "reset or frozen; whether the first hidden layer keeps training after injection (read "
                         "literally, 'only the new copy is trained thereafter' freezes it and log_std); how the "
                         "optimiser state is cleared and the new parameters enter the optimiser; the random "
                         "stream of the new weights. Answered: both act on the actor's output layer and the hidden "
                         "layer before it (actor.mean[2] and mean[4]); log_std is neither reset nor frozen and keeps "
                         "training with its Adam state. Reset overwrites those layers in place and removes their Adam "
                         "state; injection freezes the old head and adds a fresh head and its identical frozen copy "
                         "(output frozen(z) + new(z) - new_frozen(z), bit-identical at injection), the first hidden "
                         "layer (trunk) and log_std keep training through all three heads, so the number of trainable "
                         "parameters is unchanged, and the new head joins the actor optimiser's existing param group "
                         "with a fresh Adam state; the new weights come from OmniSafe's default initialiser, drawn "
                         "from a torch.Generator seeded from the run seed. After injection the actor's dormant "
                         "fraction covers all 256 hidden units (trunk, frozen head, new head, frozen copy) and its "
                         "effective rank is that of the three heads' concatenated last hidden layers (192 features). "
                         "Source: Nikishin et al. (2023, arXiv 2305.15555), Section 3, as two independent "
                         "search-engine extracts of the paper report it (the PDF could not be opened). Gates the reset "
                         "and injection runs.",
    "Q-plasticity-definitions": "Table 2.3 does not fix the input of the plasticity metrics or its details. "
                                "Answered: the task's fixed batch (2,048 states, seed 0), normalised by the run's "
                                "own observation normaliser as saved at that checkpoint and frozen; post-tanh "
                                "activations; the dormant score of equation (6) per layer against that layer's own "
                                "mean (tau = 0.025), the dormant fraction pooled over all hidden units of the actor "
                                "(128; 256 after injection); the effective rank of equation (7) (delta = 0.01) by "
                                "uncentred float64 SVD of the last hidden layer's activations (the three heads' "
                                "concatenated last hidden layers after injection); the parameter norm over all "
                                "trainable actor parameters, log_std included (frozen heads excluded after "
                                "injection; norm_all also logged). Edge cases: an all-zero layer counts as all "
                                "dormant; an all-zero matrix has rank 0; non-finite input gives NaN. The same three "
                                "metrics are logged for each critic, and for H3 (c) the trainable layer is the "
                                "second hidden layer (mean[3]) for the untreated and reset arms and the new head's "
                                "hidden layer (new.3) for injection. Gates the non-pilot Study A runs, whose ledger "
                                "rows hold these values.",
    "Q-transfer-obs": "Table 2.2 'Transfer': the observation sizes of the Goal and Button tasks differ (60 and 76 "
                      "for Point, 72 and 88 for Car), and SafetyCarGoal1-v0's held-out task is not named. "
                      "Answered: SafetyCarButton1-v0. The first-layer weights of the actor (plain or injected "
                      "trunk) and of both critics are mapped by Safety-Gymnasium observation component name, read "
                      "from both tasks at run time: shared components are copied (Point: accelerometer, "
                      "velocimeter, gyro, magnetometer, goal_lidar, hazards_lidar; Car: the same plus "
                      "ballangvel_rear and ballquat_rear), target-only components (buttons_lidar and gremlins_lidar "
                      "for Goal to Button; vases_lidar for Button to Goal) start at weight 0, source-only components "
                      "are dropped, and every other tensor is copied unchanged. The normaliser copies the shared "
                      "dimensions' statistics; target-only dimensions get mean 0 and variance 1 (_sumsq = _count - 1 "
                      "under the parent's count) and the parent's clip, so at the start of transfer the outputs "
                      "equal the parent's on the shared inputs. Gates the transfer continuations.",
    "Q-budget-normalisation": "Table 2.5 'Budget conditioning': OmniSafe's running observation normaliser would "
                              "map a constant budget feature to 0 and every unseen budget to +/-5. Answered: "
                              "budget/100 is appended to the observation after OmniSafe's observation normaliser "
                              "and the whole wrapper chain, as OmniSafe's Saute adapter appends its safety state; "
                              "the normaliser keeps the task's size (60 on SafetyPointGoal1-v0) and never sees the "
                              "budget, and the actor, both critics and the buffer take one more input (61). "
                              "Evaluation appends the budget the same way after the frozen normaliser "
                              "(envs.evaluation.append_budget) and refuses a checkpoint whose normaliser covers the "
                              "budget feature. Gates every budget-conditioned run, the pilot's Moderate run and the "
                              "few-shot continuations included.",
    "Q-level-jc": "Table 2.5 'Budget conditioning': 'updated only from episodes run at that level, by equation "
                  "(4)' does not say which episodes form a level's J_C, or what happens when a level has none. "
                  "Answered: a level's J_C is the mean episodic cost of that level's episodes among the episodes "
                  "that form J_C for a single multiplier under Q-jc-window, the last 50 finished episodes "
                  "(OmniSafe's Metrics/EpCost window, its length read from the logger), averaged as OmniSafe's "
                  "logger averages that window; a level with no episode in that set takes no step of equation (4) "
                  "in that epoch, its multiplier and Adam state unchanged. A Single arm thus updates its "
                  "multiplier from exactly the J_C that PPOLag uses, and the rule is the same for every arm and "
                  "every Continuous bin. Gates every budget-conditioned run.",
    "Q-continuous-bins": "Table 2.5 'Continuous': bin edges, the bin of the upper end 40, and the budget d in a "
                         "bin's update. Answered: six multipliers, one per 5-unit bin: [10, 15), [15, 20), [20, 25), "
                         "[25, 30), [30, 35), [35, 40], the last closed, so 40 lies in [35, 40]; an episode's or "
                         "sample's bin is read from its float32 budget feature (a feature at least an edge's feature "
                         "lies in the bin that edge opens) by one function, studyb.conditioning.level_index, for both "
                         "the per-sample penalty of equation (3) and the episodes of a bin's J_C. In each update of "
                         "equation (4) a bin's d is the mean budget of that bin's episodes in its J_C set "
                         "(Q-level-jc), and its multiplier takes one step through OmniSafe's "
                         "Lagrange.update_lagrange_multiplier (Adam) with J_C - d = mean(C_e) - mean(b_e), clamped "
                         "at 0; multipliers are named and logged by the bin's lower edge "
                         "(Metrics/LagrangeMultiplier/level_10, ..., level_35). Gates the Continuous arm.",
    "Q-search-before-pilot": "Part 7.2 and Appendix C: the targeted search is 'run before Study B's first run', "
                             "and the pilot contains two Study B runs. Answered: the pilot's two Study B runs, the "
                             "unconstrained PPO run included, are Study B's first runs: the search record must be "
                             "complete and committed in studyb/search/, and its path entered in Table 9.0, before "
                             "either starts; Study A's pilot runs and the determinism checks of plug-ins study_b "
                             "and unconstrained_ppo (software verification) do not wait for it. If the outcome is "
                             "'partial', the amendment narrowing Study B's claims (a Table 9.1 row with an "
                             "analysis/AMENDMENTS.json entry, whose id decision.json's 'amendment' field names) must "
                             "be ratified and committed before those runs start; if 'included', Study B is withdrawn "
                             "by amendment and none of its runs is launched (studyb/search/record.py, "
                             "pilot.scheduler.search_record_status). Gates the pilot's Study B runs.",
    "Q-studyb-order": "Part 3.3: 'Study B's runs are scheduled after Study A's main sweep'. Answered: a non-pilot "
                      "Study B training run (plug-in study_b, surplus and replacement seeds included) starts only "
                      "after every run of Study A's main sweep has finished training: every run of design 'main' "
                      "under the Part 6.1 cuts in force (195 runs without cuts) plus every queued group-'main' "
                      "replacement or surplus seed, an excluded run counting as finished. Pilot runs are outside "
                      "this rule, few-shot continuations follow their parents, and Study A's treatment, controller, "
                      "PID and battery runs may run alongside Study B and do not hold it. Gates the Study B training "
                      "runs.",
    "Q-determinism-late-onset": "Determinism check of a late arm (Table 3.1): Table 3.1 names no configuration "
                                "for the study_a_pid check (the N = 0.50 PID arm), whose registered onset "
                                "(5,000,000 steps) lies beyond the first checkpoint. Answered: the check trains a "
                                "copy of seed 0 of the N = 0.50 PID arm (DET- run_id) twice for the registered-form "
                                "200,000 steps (10 epochs), with the onset moved to N x the check's own length in "
                                "whole epochs, 100,000 steps (epoch 5), so both runs pass through the onset and five "
                                "PID updates before the first checkpoint is compared; nothing else changes from the "
                                "arm (pilot.scheduler.DETERMINISM_LATE_ONSET). Gates the study_a_pid determinism "
                                "check and, through its report, the 15 PID-check runs.",
    # --- Result gates: the function producing the result calls pilot.errors.require_answered.
    "Q-hazard": "Table 2.2 'Hazard relocation': training already draws a new layout from the task's generator at "
                "every reset, so re-sampling hazards with that generator is not a distribution shift, and 20 "
                "layouts x 5 deterministic episodes can repeat episodes. Answered: the hazard positions are "
                "re-sampled by the task's own layout generator with the hazards' placement square set to |x|, |y| "
                "<= 0.75 instead of the task's extents [-1.5, 1.5]^2; the generator shrinks it by the hazard keepout "
                "0.18, so hazard centres lie within 0.57 and the hazard discs (size 0.2) reach 0.77. Twenty layout "
                "seeds (3,000,000 + j) each fix the hazards, pinned for five episode seeds (3,100,000 + 5j + k) "
                "that re-draw every other object (robot, goal, vase, buttons, gremlins) with the generator's usual "
                "rejection, giving 100 distinct episodes; the policy is not updated. The registered-wording form "
                "(the unmodified generator) is not a distribution shift and enters no result. Gates every "
                "hazard-relocation evaluation (the primary outcome); while it is open the go report shows Part 6 "
                "G3 as UNDECIDED.",
    "Q-dynamics": "Table 2.2 'Dynamics perturbation': MuJoCo combines two geoms' friction by their maximum, so "
                  "scaling only the floor changes no contact, and 'body mass' names no bodies. Answered: the mass "
                  "and inertia of every body of the robot's kinematic tree (Point: agent; Car: agent, left, right, "
                  "rear) x 1.3, followed by mujoco.mj_setConst; all three friction coefficients of the floor geom "
                  "and of every robot geom x 0.7, so every robot-floor contact has 0.7 times its friction; applied "
                  "to the freshly compiled model after every reset, never compounded; no other body or geom is "
                  "changed. The robot's contacts with other objects keep their sliding friction (the object's "
                  "1.0), and for the Point robot the torsional and rolling coefficients of its object contacts fall "
                  "to 0.007, a side effect recorded (dynamics_definition contact_note). The 100 episodes use the "
                  "measurement seeds, paired with the measurement-set C_ID; the policy is not updated. Gates every "
                  "dynamics evaluation.",
    "Q-studyb-eval": "Study B's evaluations are not fully specified: which checkpoint, which seeds, and which "
                     "budget the observation carries for the final cost and selection-set costs that Part 3.4 and "
                     "Appendix B require of every run. Answered: Study B training runs are evaluated at their final "
                     "checkpoint, and each few-shot continuation at its checkpoints at 200,000, 500,000 and "
                     "1,000,000 steps (or at its only horizon under Part 6.1 cut 3); every Study B evaluation with a "
                     "budget in the observation uses Table 2.1's measurement set, with the same 100 episode seeds "
                     "at every budget and horizon and in every arm (zero-shot at the unseen and reference budgets, "
                     "few-shot, and the pilot's G4 measurement of the Moderate arm at 10, 20 and 40). For the "
                     "contract-1 fields Part 3.4 and Appendix B require of every run, episode i of the 100 carries "
                     "the arm's i-th training budget in turn, sorted(levels)[i mod k]. These fields are the "
                     "selection-set cost and return at the last ten checkpoints and final_cost/final_return on the "
                     "measurement set at the final checkpoint; for Continuous, episode i carries the midpoint "
                     "10 + 30 (i + 0.5)/100 (10.15, 10.45, ..., 39.85); they are descriptive and enter no Study B "
                     "hypothesis. Gates every evaluation of a budget-conditioned run, the pilot's Part 6 G4 "
                     "measurement included.",
    "Q-tie-break": "Part 4.1 rule 1: how a tie between two checkpoints equally close to 25 is broken (First Tasks, "
                   "Table 9). Answered: rule 1 selects the latest of the checkpoints of the window equally close "
                   "to d = 25, distances |selection_cost - 25| compared after rounding to 1e-4. Gates a selection "
                   "only when a tie occurs.",
    "Q-matched-cost-set": "Part 4.1 rules 2 to 5: which evaluation set gives an arm's matched cost (First Tasks, "
                          "Table 9). Answered: an arm's matched cost (rules 4 and 5), the reference cost and the "
                          "reference seed-to-seed SD (rules 2, 3 and 5), and G1's 'final in-distribution cost' all "
                          "use the measurement-set cost (100 episodes on the measurement seeds) of each seed's "
                          "selected checkpoint; the selection-set cost at the selected checkpoint is stored and "
                          "reported beside it, as Part 4.1.1 requires, and decides nothing. Gates the matched and "
                          "infeasible flags; while it is open the go report shows G1 as UNDECIDED when the two sets "
                          "disagree.",
    "Q-arm-complete": "Part 4.1 rules 2 to 5 say 'five seeds', but surplus and replacement seeds change an arm's "
                      "seeds. Answered: an arm is complete when its completed (non-excluded, non-auxiliary) runs "
                      "reach its seed target, 5 plus the Part 5.5 surplus count recorded in Part 8 for "
                      "primary-comparison arms. Rules 2, 3 and 5's SD condition wait until the task's reference "
                      "(N = 0) arm is complete; an arm's matched cost and flags (rules 4 to 6) and its battery "
                      "continuations wait until both the reference arm and that arm are complete; every completed "
                      "seed then counts (replacement seeds take the place of excluded ones, surplus seeds join the "
                      "mean and the SD). Gates the matched and infeasible flags and the battery continuations.",
    "Q-controller-quantities": "Table 2.4 overshoot and settling time and Table 2.1 recovery time at epoch "
                               "resolution. Answered: Controller quantities are computed at epoch resolution from "
                               "Metrics/LagrangeMultiplier, the value after each epoch's update. Peak: the maximum "
                               "over the epochs that end in (onset, onset + 2,000,000]. Final value: the mean over "
                               "the last ceil(0.10 x epochs) epochs of the run's own length. Overshoot = peak - "
                               "final. Settling time: the steps from onset to the end of the first post-onset epoch "
                               "e* such that the value of e* and of every later epoch lies within +/-10 percent of "
                               "the final value (inclusive). If the final value is 0, this means the multiplier "
                               "stays exactly 0 from e* on. It is none only if the last epoch's value lies outside "
                               "the band. Recovery time: the steps from onset to the end of the first constrained "
                               "epoch whose Metrics/BatchEpCost is at most 25; none if that never happens. In the "
                               "analysis a run that never recovers enters as its steps from onset to the end of "
                               "training plus one epoch, and a contrast of two arms censors both at the shorter "
                               "arm's horizon (metrics.controller, pilot.enrichment.controller_reference, "
                               "analysis.study_a). Gates the controller and recovery quantities.",
    "Q-adapt-censoring": "Table 2.5 'Adaptation steps': how 'recorded as above the largest horizon if never "
                         "reached' is written in the ledger's integer map. Answered: 'reaches 0.80' means a "
                         "satisfaction rate of at least 0.80 (80 or more of the 100 episodes within the budget); a "
                         "budget whose continuation never reaches 0.80 at any of its horizons records adapt_steps = "
                         "the continuation's largest horizon + 1 (1,000,001 for the registered horizons, 200,001 "
                         "under Part 6.1 cut 3), which readers decode as censored ('above the largest horizon'); "
                         "medians and the G4 readings rank a censored value above every observed horizon "
                         "(Q-g4-reading). Gates adaptation steps only when a budget never reaches the target.",
    # --- Go-report key: it gates no run; while it is open, pilot/go_decision.py shows G4 as UNDECIDED
    # --- when the two sets disagree.
    "Q-final-cost-set": "Appendix B 'final_cost': the evaluation set at the final checkpoint is not named. "
                        "Answered: final_cost and final_return are the mean cost and return over the 100 "
                        "measurement-set episodes at the run's final checkpoint; the final checkpoint's "
                        "selection-set cost (it is always in the selection window) is kept in the checkpoint record "
                        "as final_selection_cost and decides nothing. This applies to Study A rows, the "
                        "final-checkpoint estimand of rule 7(b), G1's final-checkpoint SD reported beside the "
                        "selected one, and the pilot's unconstrained cost for G4.",
    # --- Analysis keys (Parts 1, 4, 5): they gate no run; while one is open, every verdict that depends on
    # --- it is reported as PROVISIONAL or UNDECIDED by analysis/. Parts 1, 4 and 5 freeze at the go decision.
    "Q-interval": "Part 5.3 says an interval that must exclude zero is 'the primary interval' (Welch) and, in its "
                  "last sentence, 'the 95 percent percentile interval over seeds'. Answered: a box's interval "
                  "excluding zero is Part 5.3's primary interval, the two-sample Welch t-interval (95 percent; 97.5 "
                  "percent for G1 and G2 per Part 5.2); the percentile bootstrap interval over seed indices (10,000 "
                  "resamples) is reported beside every estimate and decides nothing. A box condition decided by a "
                  "correlation (Part 5.4: the H1 and G1 trends, H3 (a)) uses the registered seed-resampled "
                  "percentile interval at the box's level (95 percent; 97.5 percent for G1's trend). The Part 5.7 "
                  "bound is the upper (or lower) limit of the 95 percent Welch interval.",
    "Q-h1-shape": "H1 does not name the onset shape of the gap difference. Answered: box H1 (confirmatory, on the "
                  "primary outcome) reads Delta(N) of equation (2) on the abrupt-onset late arms, Delta(N) = mean "
                  "gap(abrupt, N, control) - mean gap(N = 0), under each step-matching control; the same box on the "
                  "ramp arms is a secondary reading, reported per (task, condition) cell as "
                  "H1[task/condition/ramp] and Holm-corrected in the ramp-shape families (Q-holm-families); the "
                  "shapes are never pooled.",
    "Q-controls-reading": "H1 is supported 'under both step-matching controls' but its falsification names no "
                          "control, and H2 names none. Answered: H1 and H2 are each read under both controls and "
                          "combined: SUPPORTED only if SUPPORTED under both; FALSIFIED only if FALSIFIED under both "
                          "(after Q-falsification-calibration); INCONCLUSIVE if the two controls disagree or either "
                          "is INCONCLUSIVE; NOT_COMPUTABLE only while a control that cannot be evaluated (data still "
                          "to come, or its N = 0.50 arm unmatched under rule 6) could still change the combined "
                          "status, with the reason given (analysis.verdict.combine_controls).",
    "Q-support-falsify-overlap": "The support and falsification conditions of H1 and G1 can hold at once (the "
                                 "trend test and the point ordering can disagree), and those of other boxes too (G4 "
                                 "under the 'at most one horizon' reading, now a sensitivity reading of "
                                 "Q-g4-reading, when an even seed count gives a median horizon index x.5). "
                                 "Answered: 'Delta increases with N' (H1) and G1's "
                                 "ordering are supported only when both Part 5.4's Spearman trend (rho > 0 with its "
                                 "seed-resampled interval excluding zero) and the ordering of the point estimates "
                                 "hold (H1: Delta(0.10) <= Delta(0.25) <= Delta(0.50); G1: Sparse < Moderate < "
                                 "Dense). If a box's support and falsification conditions both hold, the box is "
                                 "FALSIFIED with a note, and Q-falsification-calibration's bound then applies to it "
                                 "as to any falsified box.",
    "Q-falsification-calibration": "Part 1.2 reads 'falsified' as bounded below the minimum effect, while the boxes "
                                   "use intervals or point estimates, and Part 5.7's bound covers only H1 and G1. "
                                   "Answered: a box whose falsifying quantity carries a Part 1.5 minimum effect (H1, "
                                   "H2, H3 (b), H4; G1, G2, G3) is FALSIFIED only if its box conditions hold and its "
                                   "Part 5.7 bound lies below the minimum effect (5 cost units in Study A; 5 "
                                   "percentage points in Study B): the 95 percent Welch limit, in the direction the "
                                   "box claims, of each estimand its falsifying clause names (H1: the upper limit of "
                                   "Delta(0.50), per control; G1: the upper limit of Dense - Sparse; the others as "
                                   "analysis.verdict.bound_annotation computes them). Otherwise, and where the bound "
                                   "cannot be read, the box is INCONCLUSIVE ('inconclusive at this sample size') and "
                                   "its bound is reported. Boxes whose falsifying quantity has no minimum effect (H3 "
                                   "(a), H3 (c), G4), and G5, whose falsification is itself an interval limit "
                                   "against 5 percentage points, are decided by the box; whether the box conditions "
                                   "alone were met is reported in the verdict's numbers ('falsification') and in a "
                                   "note (analysis.verdict.calibrate).",
    "Q-holm-families": "Part 5.2's Holm families (four battery conditions, three tasks, four unseen budgets) overlap "
                       "for a cell that differs from the primary in task and condition. Answered: every secondary "
                       "Study A claim on a (task, condition) cell must survive Holm (alpha = 0.05) in both families "
                       "it belongs to, its task's four-condition family and its condition's three-task family: H1 "
                       "(each shape's own families) on every cell but the primary abrupt cell, and H2, H3 (a), H3 "
                       "(b) and H4 on every cell, SafetyPointGoal1-v0 x hazard included; the primary H1 claim is "
                       "uncorrected, but its p-value counts as a member of both abrupt H1 families. Families shrink "
                       "only to cells not tested (no arm by design, a recorded Part 6.1 cut, an infeasible task, an "
                       "arm left out by rule 6, a condition not yet evaluable while its key is open); an incomplete "
                       "member is held at p = 1, and any survival it could change stays undecided. Claims with a "
                       "single family (recovery time and return over the three tasks, H3 (c) over the three tasks, "
                       "Study B's four unseen budgets) use that family (analysis.study_a.StudyA.holm_survival).",
    "Q-threshold-arithmetic": "Part 4.1 rules 3 and 5: n or n - 1 in the standard deviation, strict or inclusive "
                              "comparisons. Answered: the sample standard deviation (n - 1) over the reference "
                              "arm's seeds; the task is infeasible if SD > 10 (SD = 10 is feasible); rules 3 and 5 "
                              "are inclusive; the means and their difference are compared after rounding to 1e-4, "
                              "the SD, which is not on the 0.01 grid of mean costs, after rounding to 1e-9. Part 6 "
                              "G1 and G3 compare in the same way: G1's 'at most d + 2.5', 'at most 10' and rule 5 "
                              "directly, and G3's 'at least 5' with the mean Delta(0.50) after rounding to 1e-4, U80 "
                              "and U80_pooled after rounding to 1e-9.",
    "Q-iqm": "Part 5.3's interquartile mean 'with a stratified bootstrap over seeds and evaluation episodes' resamples "
             "episodes, which Part 5.1 forbids for intervals. Answered: the IQM is descriptive and never decides a "
             "verdict: per arm and outcome, the 25-percent trimmed mean (scipy trim_mean 0.25) of the per-episode "
             "values pooled over the arm's seeds; the IQM of a gap is IQM(C_cond episodes) - IQM(C_ID episodes), "
             "and that of a two-arm estimand the difference of the arms' IQMs. Its 95 percent percentile interval "
             "comes from a two-level bootstrap with 10,000 resamples: seeds drawn with replacement (arms with the "
             "same seed list share the draw), then episodes drawn with replacement within each drawn seed. Where "
             "per-episode records are missing it is reported as not computable.",
    "Q-bootstrap-details": "Part 5.3's bootstrap leaves open the random stream, the quantile rule, joint or separate "
                           "resampling of arms, and resamples that are undefined. Answered: each analysis gets its "
                           "own generator, numpy.random.default_rng([0, crc32(analysis id)]), recorded with the "
                           "result; B = 10,000 resamples; percentile limits use numpy's 'linear' quantile rule. When "
                           "the two arms of a contrast have identical seed lists one index draw serves both (pairs "
                           "kept together); otherwise, partly overlapping lists included, each arm is resampled on "
                           "its own indices; a Spearman trend resamples seed ids from the union, each carrying all "
                           "its points. Undefined resamples (a constant variable) are dropped and counted, and "
                           "flagged above 1 percent. Holm (Part 5.2) uses the two-sided Welch p of a two-arm claim "
                           "and, for a Spearman claim (H3 (a)), for which Part 5.4 registers only a percentile "
                           "interval, the bootstrap p = min(1, 2 min(#rho* <= 0, #rho* >= 0) / B_defined).",
    "Q-surplus-in-analysis": "Parts 4.1 and 5.4 say 'five seeds' and 'twenty points', but surplus seeds give some arms "
                             "up to twelve. Answered: every completed seed of an arm (registered seeds, Part 5.6 "
                             "replacements and Part 5.5 surplus seeds) is used in matching (rules 2 to 5), every "
                             "two-arm estimate with its Cohen's d, Welch and bootstrap intervals and Holm p, the Part "
                             "5.7 bound and Study B's trend; paired analyses use the seeds common to both arms; the "
                             "power statement and the detectable size use the actual counts (the smaller arm's n). "
                             "The two exceptions are Study A's seed-index-resampled correlations of Part 5.4, the H1 "
                             "trend (the registered twenty points, four onset fractions by five seeds) and H3 (a) "
                             "('with the same interval'): both use each arm's registered seeds and their "
                             "replacements only (analysis.data.five_seed_view), because surplus seeds exist only at "
                             "N = 0 and N = 0.50 and would leave seed indices without all of their levels.",
    "Q-h3-scope": "H3: which arms form 'across late arms' in (a), and whether (b) and (c) are read at N = 0.50 only. "
                  "Answered: H3 (a) takes as points every completed run of the late main-sweep arms of the task (N = "
                  "0.10, 0.25, 0.50; abrupt and ramp; both step-matching controls) that is matched under Part 4.1, "
                  "an arm left out by rule 6 excluded and named; each point is the metric at onset against the gap "
                  "at the matched checkpoint, and treatment, controller-variant and PID arms are not points. H3 (b) "
                  "and (c) compare only the N = 0.50 abrupt, total-steps-matched arms (Table 3.3), treated minus "
                  "untreated. The absences in (b), '... and additional constrained training does not' [reduce the "
                  "gap], 'injection does not' and 'none of the three reduces it', mean no reduction with an "
                  "uncorrected 95 percent interval excluding zero; a claim of reduction needs the interval "
                  "excluding zero and Holm (Q-holm-families).",
    "Q-h4-reading": "H4: is 'the gap does not fall' read on the point estimate or on the interval. Answered: on the "
                    "point estimate: F = gap(warm-started, N = 0.50) - gap(untreated N = 0.50) >= 0, compared after "
                    "settled rounding; a fall whose 95 percent interval includes zero is INCONCLUSIVE, neither "
                    "support nor falsification. 'The difference from the N = 0 arm no longer excludes zero' is read "
                    "on the uncorrected 95 percent Welch interval of G = gap(warm-started) - gap(N = 0). The two "
                    "claims of support, 'falls' and 'remains above', each need the interval excluding zero and "
                    "Holm; a falsification also needs its Part 5.7 bound (Q-falsification-calibration: F's lower "
                    "limit above -5, or G's upper limit below 5).",
    "Q-h0-scope": "H0: on which outcome H1 and H2 must both be falsified. Answered: box H0 reads boxes H1 and H2 on "
                  "the primary outcome, the robustness gap under hazard relocation on SafetyPointGoal1-v0 (H1 on the "
                  "abrupt arms, Q-h1-shape), each combined over the two controls by Q-controls-reading and "
                  "calibrated by Q-falsification-calibration. H0 is SUPPORTED iff both are FALSIFIED, FALSIFIED iff "
                  "either is SUPPORTED, and otherwise INCONCLUSIVE (NOT_COMPUTABLE only while missing data could "
                  "still change it); when H0 is supported, H3 and H4 are reported as NOT_TESTED on every cell, with "
                  "their estimates still given.",
    "Q-g1-criteria": "G1: whether 'exceeds 5 percentage points with an interval excluding zero' is a point condition, "
                     "which intervals 'overlap', and the trend factor. Answered: two conditions on Dense minus "
                     "Sparse in the per-seed mean zero-shot satisfaction over the unfloored unseen budgets: the point "
                     "estimate exceeds 0.05, and the 97.5 percent Welch interval excludes zero (lower limit > 0; the "
                     "percentile interval reported beside it). In the falsification clause, 'overlapping intervals' "
                     "are the per-arm one-sample 97.5 percent t-intervals of the arm means of Sparse, Moderate and "
                     "Dense, compared pairwise, touching limits counting as overlapping, and 'within 5 percentage "
                     "points of one another' means that the largest minus the smallest of the three arm means is at "
                     "most 0.05. Part 5.4's Spearman trend uses the number of training levels (Sparse 2, Moderate 3, "
                     "Dense 5) as the factor, with a 97.5 percent percentile interval from resampling seed indices.",
    "Q-g2-outcome": "Part 5.2 tests G2 on the mean over the four budgets, while the G2 box counts budgets; and ties "
                    "between two equidistant comparators. Answered: G2 is decided by its box's per-budget count and "
                    "remains a confirmatory (primary) test. At each unseen budget Dense is compared with its nearest "
                    "single-level arm by equation (11): Single-10 at 5, Single-40 at 45, and at 15 and 30 whichever "
                    "of the two equidistant single-level arms has the higher arm-mean zero-shot satisfaction there "
                    "(on a budget floored by Part 4.2's floor rule, the one with the lower arm-mean zero-shot "
                    "violation magnitude; an exact tie goes to the one least favourable to G2 on the outcome "
                    "compared). A budget is won when Dense minus comparator exceeds 0.05 with the 97.5 percent Welch "
                    "interval excluding zero and lost when the point estimate is at most 0.05 (on a floored budget, "
                    "won when Dense's violation magnitude is lower with the interval excluding zero, lost when it is "
                    "not lower); G2 is supported with at least 3 of 4 budgets won and falsified with at least 2 "
                    "lost. The per-seed composite contrast over the unfloored budgets is descriptive, with "
                    "Holm-adjusted per-budget p-values beside it, and at 15 and 30 both candidates' contrasts are "
                    "reported.",
    "Q-g3-arms": "G3: 'for every arm', although only arms spanning [10, 40] have equal distances; how the arm's "
                 "satisfaction at its nearest training level is measured. Answered: G3, a secondary box, is decided "
                 "on Sparse, Moderate, Dense and Continuous, the arms whose training range is [10, 40] and whose "
                 "nearest training levels by equation (11) are 10 for budget 5 and 40 for budget 45. Each arm's "
                 "satisfaction at those reference levels uses the same final checkpoint, the same 100 "
                 "measurement-set episodes and the deterministic policy as the zero-shot evaluation, with the "
                 "reference budget in the observation (studyb.evaluation.reference_budgets); per seed D = (sr(10) - "
                 "sr(5)) - (sr(40) - sr(45)), and an arm is supported when mean D > 0.05 with its 95 percent "
                 "one-sample t-interval excluding zero. G3 is supported when all four arms are, and falsified when "
                 "|mean D| <= 0.05 for more than half of them (at least 3 of 4); if budget 5 or 45 is floored, D is "
                 "read on violation magnitude by direction and interval only (Q-floor-rule). The all-seven-arms "
                 "reading, each single-level arm with its own nearest level, is reported beside as a sensitivity "
                 "reading.",
    "Q-g4-reading": "G4: the median of adaptation steps over what, and whether 'within one horizon' is inclusive. "
                    "Answered: G4 is read on horizon indices (1 = 200,000, 2 = 500,000, 3 = 1,000,000 steps); a run "
                    "that never reaches 0.80 is censored at the index after the largest horizon, 'above the largest "
                    "horizon' (Table 2.5): 4 with the three registered horizons, 2 after Part 6.1 cut 3; the budgets "
                    "used are those off the few-shot floor (Q-floor-rule). The ordering clause 'dense < moderate < "
                    "sparse' is read on one arm statistic, the median over seeds of each seed's median index over "
                    "those budgets, strictly ordered; 'Dense is at least one horizon below sparse' holds on a budget "
                    "when Sparse's median index over seeds minus Dense's is at least 1 and must hold on more than "
                    "half of those budgets; 'within one horizon of one another' holds on a budget when the largest "
                    "minus the smallest of the three arms' per-budget medians is less than 1, and G4 is falsified "
                    "when it holds on more than half of those budgets. Three sensitivity readings are reported "
                    "beside the verdict: 'within' as a spread of at most 1, the ordering on a majority of the "
                    "per-budget medians, and a censored run placed at the largest horizon's index.",
    "Q-floor-rule": "Part 4.2 floor rule: which arms count and which rate. Answered: all seven Study B arms at each "
                    "unseen budget, each by its mean zero-shot satisfaction rate over its completed seeds; the budget "
                    "is floored when more than half of the arms (at least 4 of 7) have a mean strictly below 0.05. A "
                    "floored budget leaves the four-budget means of G1 and G5, and its comparisons (G1's Dense-Sparse "
                    "and G5's Continuous-Dense beside those verdicts, G2's win or loss there, the per-budget claims) "
                    "are made on zero-shot violation magnitude (Table 2.5) by direction and interval excluding zero, "
                    "with no 5-point margin, the report saying so; if budget 5 or 45 is floored, both of G3's drops "
                    "are read as rises in violation magnitude. For G4 the same count is made on the few-shot rates at "
                    "each horizon: a budget floored at every horizon leaves G4's counts and arm statistic, whose "
                    "majority is then taken over the remaining budgets (none left: not computable), and Dense and "
                    "Sparse are compared there on the few-shot violation magnitude at the largest horizon, by "
                    "direction and interval. When unknown rates could change the count, the dependent comparisons "
                    "are not computable. The readings 'only the arms of the comparison' and 'per-seed rates' are "
                    "reported beside as sensitivity readings.",
})

# Keys of PENDING whose answers the code implements: every key, answered in docs/DECISIONS.md (2026-10-02), to be
# ratified by the group in Table 9.1. The code implements each answer stated in PENDING above, so every run
# gate, result gate and report key is released. A key the group changes is removed here until its code
# follows the new answer (change the code and the PENDING text, then add the key back in the same pull request;
# a changed run spec is refused by `schedule add`; `python -m pilot schedule resolve RUN requeue` stores the
# committed code's spec of a queued run that never trained; the scheduler and the launcher hold every run on
# the run gates the current code derives, manifest.open_run_gates).
ANSWERED_QUESTIONS: frozenset[str] = frozenset({
    "Q-adapt-censoring", "Q-arm-complete", "Q-bootstrap-details", "Q-budget-normalisation", "Q-continuations",
    "Q-continuous-bins", "Q-controller-quantities", "Q-controls-reading", "Q-cost-critic", "Q-data-control-lr",
    "Q-determinism-late-onset", "Q-dynamics", "Q-falsification-calibration", "Q-final-cost-set", "Q-floor-rule",
    "Q-g1-criteria", "Q-g1-level", "Q-g2-outcome", "Q-g2-run-equivalents", "Q-g3-arms", "Q-g3-pairing", "Q-g4-level",
    "Q-g4-reading", "Q-h0-scope", "Q-h1-shape", "Q-h3-scope", "Q-h4-reading", "Q-hazard", "Q-holm-families",
    "Q-interrupted-run", "Q-interval", "Q-iqm", "Q-jc-window", "Q-level-jc", "Q-matched-cost-set", "Q-pid-eq9",
    "Q-pilot-unconstrained-seed", "Q-plasticity-definitions", "Q-ramp-step", "Q-rate-limit", "Q-reset-injection",
    "Q-rounding", "Q-search-before-pilot", "Q-seed-collision", "Q-selection-window", "Q-studyb-eval",
    "Q-studyb-order", "Q-support-falsify-overlap", "Q-surplus-arm-set", "Q-surplus-in-analysis",
    "Q-threshold-arithmetic", "Q-tie-break", "Q-transfer-obs", "Q-warm-start",
})


def is_open(key: str) -> bool:
    """True while a PENDING key is not in ANSWERED_QUESTIONS: the code does not implement its answer yet
    (docs/DECISIONS.md, to be ratified in Table 9.1).

    Raises KeyError for a key that is not in PENDING (a typo must not read as answered).
    """
    if key not in PENDING:
        raise KeyError(f"unknown open question {key!r}")
    return key not in ANSWERED_QUESTIONS

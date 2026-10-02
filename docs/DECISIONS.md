# Decisions on the open questions (draft amendment rows for Table 9.1)

## What this file is

This file answers every open question that the code of this repository had left to the group:

- the 54 keys of `PENDING` in `configs/registered.py`, marked *starred* below (while open, each held runs or results,
  or left verdicts undecided or provisional, except Q-g4-level and Q-g2-run-equivalents, whose other reading is only
  reported beside the verdict their answer decides);
- the 24 unstarred notes of HANDOVER.md §9, marked *note* (readings the code documented without holding anything);
- three extra items: the ledger-schema amendment (X-ledger-schema-amendment), and what can and cannot be decided
  about the machine-hour allocation (X-allocation) and the registration record (X-registration).

The answers were decided on 2026-10-02, before any pilot or registered run, so every result that rests on them stays
confirmatory (Part 8.2). Each section below is a draft row, or group of rows, for the amendment log (Part 9,
Table 9.1).

## How it was made

At the pilot owner's request, the answers were prepared by AI agents (Claude), not by a member of the group, in a
structured process:

1. a *decider* read the question, the registered text, the sources it cites, the pinned OmniSafe 0.5.0 and
   Safety-Gymnasium 0.4.1 code and the repository, and proposed an answer;
2. an *adversarial critic* tried to refute it, and then confirmed it, confirmed it with an edit, or disputed it;
3. an *independent judge* re-derived every answer that was not yet of high confidence (25 of them);
4. a *coordinator* settled the remaining disagreements (one dispute, on Q-surplus-arm-set) and checked the answers
   against each other.

The criteria, in this order: fidelity to the registered text (where two passages conflict, the one that defines a
quantity prevails over one that summarises it); the cited sources; the validity of the inference (arms treated
alike, the conservative choice where the text is open); feasibility; and the least change to what was already
proposed. "The earlier proposal" below means the text of the key in `PENDING`, or in HANDOVER.md §9, before this
decision. The full reasoning of every step, verbatim, is in `docs/DECISIONS_EVIDENCE.md`.

One limit: the agents worked in a sandbox that could not open some sources. The injection method of Nikishin et al.
(2023) is known from two search-engine extracts of the paper (Q-reset-injection), and the PPO-Lagrangian reward cell
of Ji et al. (2024) must be copied by the group (Q-appendix-a).

## Status

The code implements every answer, and `ANSWERED_QUESTIONS` in `configs/registered.py` lists all 54 starred keys, so
no run, result or verdict waits on a question any more. The answers are nevertheless not amendments until the group
ratifies them:

1. At one meeting the group reads this file, approves or changes each row, and enters the rows in Table 9.1 with the
   hash of the commit that records them (guidance at the end of this file).
2. Role 4 completes the analysis amendment `A1` already entered in `analysis/AMENDMENTS.json` (all 54 keys, `data_seen`
   false): it enters the hash of the commit that records the ratification (null, "not yet recorded", until then) and
   the approval, and removes a key the group changes until its code follows.
3. The code is merged only after that meeting. If the group changes an answer, the code that implements it changes in
   the same pull request (HANDOVER.md task 11).

## What stays with people

- The ratification itself, above.
- The targeted search of Appendix C, by Hamza Nisar. Q-search-before-pilot and the four search notes fix how it is
  run and recorded; the search is a person's work.
- The machine-hour allocation: one number that only the group can give (X-allocation).
- The registration tag and Table 9.0 (values at the end of this file; X-registration).
- The runs on the workstation, one command each in the runbook `scripts/workstation.py` (HANDOVER.md §4 and §7).

## Confidence

80 of the 81 entries are of high confidence: all 54 starred keys, 23 of the 24 notes and the three extra items (for
X-allocation, high confidence that no analysis can decide it). One is of medium confidence: **Q-detectable-size**.
Read literally, Part 5.5's sentence "Differences below the detectable size ... are not claimed as findings" would
hold back a SUPPORTED verdict whose observed effect size is small; the answer reads it as an annotation, because the
boxes' own interval rule and the registered power numbers govern. Only the group can say what the sentence was meant
to do, so please confirm this row at the meeting.

Several decisions fix a value that no published source gives. For these the confidence is high in the decision; the
value itself is a fixed design choice, made before data and the same for every arm:

- Q-hazard: the hazards' placement square |x|, |y| ≤ 0.75;
- Q-mujoco-exception: the cap of 5 unstable episodes per evaluation call, and the reserve seed bases;
- Q-data-control-lr: the level at which the extension's learning rate restarts, (1 − N) × 3e-4;
- Q-interrupted-run: `MAX_RESTARTS = 2` for endings of process origin;
- Q-exclusion-breaker: the stop at the arm's seed target;
- Q-search-depth: 50 results per search (the figure of First Tasks);
- Q-transfer-obs: the starting statistics of the normaliser for inputs only the target task has;
- Q-reset-injection: `log_std` keeps training after injection;
- Q-eval-seeds: the seed bases of the evaluation sets.

## Decisions that change an earlier proposal

24 of the 81 answers change what was proposed before (decision ADOPT_MODIFIED), and the code was changed to match.
X-allocation could not be decided by analysis. Every other answer adopts the earlier proposal; for Q-surplus-arm-set
it does so after a dispute.

| Key | What changed | Why |
|---|---|---|
| Q-warm-start | If the N = 0 run a warm-started run copies from is excluded, nothing is taken from it; the warm-started run, which never trained, is superseded by its arm's next unused seed. | Closes a case left to the group after data exist (Part 5.6). |
| Q-data-control-lr | The first T steps follow the untreated arm's schedule; the extension restarts at (1 − N) × 3e-4 and falls to 0 (was: one decay over T + N·T). | "Continues ... after T" (Table 2.4); a rate of 0 would make H3 (b) true by construction. |
| Q-continuations | Under cut 3 a shortened few-shot continuation keeps the 1,000,000-step schedule and stops at 200,000 steps (was: its own 200,000-step schedule). | Its reading must equal the full continuation's first horizon (Part 6.1, cut 3). |
| Q-controller-quantities | Settling time is defined also when the final multiplier is 0. | The registered definition is defined there (the band around 0 is {0}). |
| Q-mujoco-exception | An unstable episode is replaced from a reserve seed sequence, at most 5 per evaluation call (was: the evaluation fails and the group decides). | Keeps the registered 100 episodes; no choice after data. |
| Q-search-before-pilot | A "partial" outcome's narrowing amendment must be committed before the pilot's Study B runs; determinism checks are not Study B runs. | Table C.1: "recorded before the first run". |
| Q-search-depth | The result order is fixed with the depth (relevance; "Relevance" in arXiv); citation lists are screened in full with their length recorded. | A 50-deep cut means something only on a ranked list. |
| Q-search-citations | Both lists of every work citing Yao et al. are screened, not only those of works read in full; duplicates still count; a Google Scholar fallback. | "Every paper that cites it" (Table C.1). |
| Q-interrupted-run | Outages are restarted without limit; a third ending of process origin is an exclusion ('incomplete') with a replacement; SIGHUP counts as an outage; `exclude` is refused (was: held for the group after 2 restarts). | No decision about one run after data exist (Part 5.6). |
| Q-g3-pairing | With fewer than three seed-matched pairs, the leftover runs are paired in seed order and the rule decides G3 (was: the group decides); the pooled value is descriptive. | No go condition decided after the data are seen. |
| Q-g2-run-equivalents | The corrected total (627.13 for the uncut design) decides G2; 484 is reported beside it (was: 484 decides). | 484 counts runs, not steps (Parts 3.3 and 6.1). |
| Q-exclusion-breaker | The resolution is fixed: a diagnosis blind to outcomes, an erratum if a defect is found, then the replacement; a stop at the arm's seed target (was: the group decides). | Part 5.6 repeats every exclusion; no decision after data. |
| Q-g-names | An erratum makes Table 9.1's first row "G1 to G5"; the go conditions keep their names, qualified (was: renamed C1 to C4). | Keeps every registered identifier. |
| Q-ledger-v2 | The supplement schema is frozen at version 1 with `results/supplement_schema.sha256`. | The same protection the ledger schema has. |
| Q-appendix-a | The committed fixed batches stand; a mismatch of `--check` on the workstation is reported, never collected again (was: the mismatch goes to the group). | Table 2.3: "collected once per task". |
| X-registration | Tables 9.0 and 9.1 are filled in a commit that changes nothing else, and the registration is complete before any pilot run. | Until then the document is "the adopted draft". |
| Q-falsification-calibration | FALSIFIED only when the Part 5.7 bound lies below the minimum effect, otherwise INCONCLUSIVE (was: the boxes decide). | Part 1.2 and Part 5.7 define "falsified" so. |
| Q-holm-families | Every secondary claim must survive Holm in both its families (was: one family for cells that differ in one dimension). | Part 5.2 names both families. |
| Q-surplus-in-analysis | The H1 trend and H3 (a) keep each arm's registered seeds and their replacements (was: every seed everywhere). | Part 5.4's "twenty points". |
| Q-detectable-size | New wording of the note on a SUPPORTED verdict below the detectable size. | The old note contradicted its own verdict. |
| Q-exploratory-labels | The analysis code hash covers every in-repository module the analysis imports. | The freeze of Part 5.8. |
| Q-unmatched-reporting | "Never reaches the budget" means unmatched with a matched cost above d + tolerance (was: unmatched above the reference). | Rule 3's own test of the budget. |
| Q-training-age-systematic | Training ages are compared for every pair a verdict compares, not only against the reference. | "Between arms" (Part 4.1.1). |
| Q-g4-reading | An ordering at arm level; "within one horizon" is a spread below 1 (was: per-budget ordering; "within" as the same median index); a censored run takes the index after the largest horizon, as before. | Gives both clauses of the box force; G4 stays testable after cut 3. |

## 1. Study A training and run gates

### Q-rounding (starred)

**Answer.** Constrained-steps-matched arms at N = 0.10 and 0.25: the exact onset N·T/(1−N) is rounded to the nearest
whole 20,000-step epoch, and the run then trains exactly T = 10,000,000 constrained steps. N = 0.10: onset 1,120,000
(epoch 56), total 11,120,000. N = 0.25: onset 3,340,000 (epoch 167), total 13,340,000. The realised onset fractions
are 0.1007 and 0.2504. The arms keep their registered labels N = 0.10 and 0.25 in every analysis, so the H1 Spearman
trend uses the registered levels. N = 0.50 is exact (onset 10,000,000, total 20,000,000).

**Confidence.** High.

**Why.** Table 2.4 "Constrained-steps matched" defines the control first: every arm trains under the constraint for
T steps; the total T/(1 − N) follows from that definition, so the defining clause is kept exactly. T/(1 − N) is not
a whole number of 20,000-step epochs, and the multiplier updates once per epoch (Table 3.1), so the onset must fall
on an epoch boundary; the nearest epoch gives the smallest onset error (8,889 and 6,667 steps). Rounding the total
instead of the onset gives the same numbers.

**Rejected.** Flooring the onset (a larger error); an onset off the epoch grid; a variable epoch length (contradicts
Table 3.1); keeping the total at T/(1 − N) and giving up exactly T constrained steps.

**Implemented in.** No code change: `pilot.manifest.onset_schedule`; `envs.onset._check_registered_design`.

### Q-cost-critic (starred)

**Answer.** Both critics (reward and cost) train from the first epoch in every Study A run (OmniSafe use_cost:
True). At onset only two things change: the actor's surrogate switches from A_R to equation (3), and the multiplier
starts to update (equation 4). Before onset the cost critic is trained and the cost advantage is computed, but it
does not enter the actor's loss.

**Confidence.** High.

**Why.** Table 2.1 "Constraint onset" lists exactly two things that start at onset, the cost term in the surrogate
(equation 3) and the multiplier update (equation 4), and says that before onset "the cost advantage is not computed
into the loss", which presumes it exists. Freezing the cost critic would add a third change at onset and give only
the late arms an untrained cost critic, a confound for H1 and H4; it would also change OmniSafe's default `use_cost:
True` (Table 3.1). Training it before onset leaves the actor's pre-onset trajectory unchanged.

**Rejected.** Freezing or skipping the cost critic until onset.

**Implemented in.** No code change: `envs.onset.OnsetMixin._update` and `_compute_adv_surrogate`;
`envs.onset.check_training_config` refuses any `use_cost` other than True.

### Q-jc-window (starred)

**Answer.** J_C in the multiplier update (eq. 4) and the error of the PID update (eq. 9) is OmniSafe's
Metrics/EpCost: the mean episodic cost of the last 50 finished episodes (this epoch's 20 plus the previous 30;
window_length = 50). PPOLag._update and CPPOPID._update read it, unmodified. The per-epoch training-batch mean, over
the epoch's own 20 episodes, is logged beside it as Metrics/BatchEpCost. That is the quantity used for recovery time
and for every training-batch cost Part 3.4 requires. The gloss of J_C in Part 2.5 ('the mean episode cost of the
current epoch') is corrected by amendment to this definition.

**Confidence.** High.

**Why.** Table 3.1 registers PPO-Lagrangian "as implemented in OmniSafe", and Table A.1 verifies the multiplier
update against OmniSafe's PPOLag source. OmniSafe's `Metrics/EpCost` is a 50-episode window that its own source
labels "Average cost of the epoch", so the registered gloss repeats OmniSafe's inaccurate label rather than defining
another estimator. Keeping the window keeps the benchmark behind Table 3.5 and the controller dynamics that H4
studies; the critic noted that the gloss appears both in the symbols and beside equation (9), and the answer covers
both.

**Rejected.** Replacing J_C by the per-epoch batch mean: it modifies OmniSafe's update, departs from the benchmark,
and removes part of the lag behind the overshoot H4 studies.

**Implemented in.** No code change: OmniSafe's `PPOLag._update` and `CPPOPID._update`, called unmodified by
`envs.onset.OnsetMixin._update`; `pilot.algorithms.FullStateCheckpointMixin` logs `Metrics/BatchEpCost` and saves
the 50-episode windows.

### Q-ramp-step (starred)

**Answer.** The multiplier update of epoch e (0-based) evaluates equation (5) at t = e × 20,000, the first step of
the epoch whose data the update uses. N·T in eq. (5) is the run's own onset step (the rounded onset for
constrained-steps arms, Q-rounding). W = 1,000,000 steps (0.10 × the registered T = 10,000,000) in every ramp arm,
whatever its total. The onset epoch's update therefore uses d_loose. The following 49 updates fall linearly. Every
update from onset + 1,000,000 steps uses d = 25. Only the budget in the multiplier update changes; the surrogate has
no budget.

**Confidence.** High.

**Why.** Equation (5) is written for a step t on the half-open window from N·T to N·T + W, and Table 2.1 "Linear
ramp" says the budget falls "from d_loose" at onset. Evaluating it at t = e × 20,000 uses the convention that makes
the onset epoch the first constrained one (Q-onset-epoch), so the first constrained update uses exactly d_loose and
50 updates fall in the window. W is 0.10 of the registered T in every ramp arm, so the ramp has the same shape under
both step-matching controls, as H2's abrupt-against-ramp contrast needs; N·T is the run's own onset step, so the
ramp starts "from onset" in the constrained-steps arms too.

**Rejected.** t = (e + 1) × 20,000 (never uses d_loose); the middle of the epoch; W scaled to each run's total.

**Implemented in.** No code change: `envs.onset.effective_budget`, `OnsetMixin._update` (sets the cost limit of the
multiplier update) and `build_onset_cfgs`.

### Q-warm-start (starred)

**Answer.** Only the multiplier value is copied, with a fresh (never-stepped) Adam state. The value is the one the N
= 0 run of the same task and seed logged in progress.csv after the update of epoch onset/E − 1, i.e. the value it
had reached at the onset step; it is cross-checked against that run's checkpoint when one exists. It is set before
the onset epoch's multiplier update. A replacement seed whose N = 0 run the N = 0 arm never received gets an
auxiliary N = 0 run of that seed (`schedule resolve RUN add-auxiliary`), which is left out of the N = 0 arm by the
matching and the analysis. Added rule: if the N = 0 run a warm-started run depends on (an auxiliary run included) is
excluded under Part 5.6, no value is taken from the excluded run. The scheduler launches a warm-started run only
after its dependency has finished training, so that warm-started run is a blocked run that never trained. It is not
excluded: it is superseded by its arm's next unused seed (`schedule resolve RUN replace`), whose source is the N = 0
arm's run of that seed or, failing that, an auxiliary run. An excluded auxiliary run is never repeated or replaced
itself.

**Confidence.** High.

**Why.** Table 2.4 "Warm-started multiplier" names only the value ("set to the value the N = 0 arm of the same task
and seed had reached at the same step"), and the abrupt arm it is compared with starts with a fresh Adam state, so
copying only the value makes the starting value the single difference, a clean contrast for H4. The value "reached
at" the onset step is the one after the update of the epoch before onset, set before the onset epoch's update. The
change closes a case the code left to "the group decides": an excluded run supplies no value (Part 5.6), and the
warm-started run that waited for it never trained, so it is superseded by the arm's next unused seed and Part 5.6's
"a run that completes is kept" is respected.

**Rejected.** Copying the Adam moments too (imports another run's momentum); the value at the end of the onset epoch
(one update late); the mean over seeds for a replacement seed (breaks "same seed"); a value logged by an excluded
run (reuses an excluded run's data).

**Implemented in.** `envs.onset.warm_start_source` and `OnsetMixin._apply_warm_start`;
`pilot.dependencies.Dependency.multiplier_at`; `pilot.scheduler` (`schedule resolve RUN add-auxiliary`; `schedule
resolve RUN replace` for the blocked warm-started run, and the status hints).

### Q-rate-limit (starred)

**Answer.** The rate-limited multiplier runs OmniSafe's own update unchanged (Adam, lr 0.035, on −λ(J_C − d),
clamped to [0, upper bound]). After each such step the result is clipped to [old − b, old + b] with b = 0.05 × old +
0.01, where old is the value before that epoch's update. It is then clamped at 0. Adam's moments are not altered by
the clip. The proposed (pre-clip) value is logged each epoch as Onset/MultiplierProposed, and the report gives the
share of constrained epochs in which the clip bound. Adam's step is about its learning rate (0.035) while the sign
and size of J_C − d are steady, so the clip then binds only while λ is below about 0.5. Adam's step reaches about
2.5–3 × 0.035 after a sudden rise in the violation, and the clip also binds then at larger λ. The amendment states
this, and the rate-limited arm stays the secondary H4 check, as registered.

**Confidence.** High.

**Why.** Table 2.4 "Rate-limited multiplier" clips each multiplier update so that it changes by at most 5 percent of
its current value plus 0.01 per epoch. The update being clipped is the registered one, OmniSafe's Adam update
(Q-multiplier-adam), and "its current value" can only be the value before the update; Adam's moments are left alone
because the text defines no second change. The critic's simulation in the pinned torch showed that Adam's step grows
to about 2.5 to 3 times its learning rate after a sudden rise in violation, so the clip also binds then at larger
multipliers; the amendment states this, the logged pre-clip value makes the binding visible, and the arm stays the
secondary H4 check.

**Rejected.** Replacing Adam by the plain step of equation (4) (changes the registered optimiser); feeding the
clipped step back into Adam's moments (not in the text); reading "current value" as the value after the update
(circular).

**Implemented in.** No code change: `envs.onset.rate_limit` and `RateLimitedLagrange.update_lagrange_multiplier`
(logs `Onset/MultiplierProposed`).

### Q-pid-eq9 (starred)

**Answer.** The PID arms run OmniSafe 0.5.0's CPPOPID/PIDLagrangian unmodified from onset, with the pinned
CPPOPID.yaml gains: kp 0.1, ki 0.01, kd 0.01, d_delay 10, P and D EMA alpha 0.95, sum_norm True, diff_norm False,
penalty_max 100, init 0.001, cost limit 25. Every other setting is identical to PPOLag.yaml. The error is J_C − d,
with J_C from the 50-episode window (Q-jc-window). Before onset pid_update is not called: the penalty stays at
OmniSafe's initial 0.0, the integral at 0.001 and both EMAs at 0, and the actor's surrogate is A_R. The PID arms are
abrupt and total-steps matched on SafetyPointGoal1-v0 at N = 0.10, 0.25 and 0.50. The amendment rewrites eq. (9) in
OmniSafe's form. Let k = 1, 2, … count PID updates from onset, e_k = J_C,k − d, P_k = 0.95·P_{k−1} + 0.05·e_k, D_k =
0.95·D_{k−1} + 0.05·J_C,k and I_k = max(0, I_{k−1} + 0.01·e_k), with P_0 = D_0 = 0 and I_0 = 0.001. Then λ_k =
max(0, 0.1·P_k + I_k + 0.01·max(0, D_k − D_{max(k−10, 0)})), with no penalty_max cap while sum_norm is True. The
surrogate is (A_R − λ A_C)/(1 + λ).

**Confidence.** High.

**Why.** Table 2.4 "PID multiplier" registers "OmniSafe's PIDLagrangian update (equation 9), using the default gains
of OmniSafe's CPPOPID configuration", and Appendix A copies that configuration with its delays, moving-average
factors and penalty maximum. Equation (9) is itself headed as OmniSafe's form, so the specific passage that names
the software prevails: the text of equation (9) is corrected, not the code. Skipping the PID update before onset
keeps the multiplier "held at its initial value" (Table 2.1), and the PID arms are total-steps matched like the
other H4 variants (Table 3.3) and as the cut order's 15 runs imply. The critic checked that OmniSafe's delayed
derivative compares with 0 during the first ten updates after onset, and the rewritten equation states this exactly.

**Rejected.** Equation (9) implemented literally (not OmniSafe's PIDLagrangian); the PID update running before onset
(contradicts Table 2.1); constrained-steps-matched PID arms (not implied by Table 3.3).

**Implemented in.** No code change: `envs.onset.OnsetCPPOPID` and `OnsetMixin._update`;
`envs.onset.PID_STEP_MATCHING = "total_steps"`. The amendment rewrites the text of equation (9).

### Q-reset-injection (starred)

**Answer.** Both interventions act on the actor's last two layers: the output layer and the hidden layer before it
(actor.mean[2] and mean[4]). log_std is a free parameter, not a layer: it is neither reset nor frozen, and it keeps
training with its Adam state. Reset overwrites those layers in place, then removes their Adam state
(optimizer.state.pop). Injection works as follows. The old head is frozen. A fresh head θ'1 (trained) and its
identical frozen copy θ'2 are added. The output is frozen(z) + (new(z) − new_frozen(z)), bit-identical at injection.
The first hidden layer (trunk) and log_std keep training and receive gradients through all three heads, so the
number of trainable parameters is unchanged. The new head joins the actor optimiser's existing param group (same LR
and LinearLR schedule) with a fresh Adam state. The new weights come from OmniSafe's default initialiser
(kaiming_uniform weights, nn.Linear default biases), drawn from a dedicated torch.Generator seeded from the run
seed, so the global torch stream stays that of the untreated run of the same seed. After injection the actor's
dormant fraction covers all 256 hidden units (trunk, frozen head, new head, frozen copy), and its effective rank is
that of the three heads' concatenated last hidden layers (192 features). Source of this reading: Nikishin et al.
(2023), arXiv:2305.15555, Section 3, as two independent search-engine extracts of the paper report it (the PDF could
not be opened from the sandbox).

**Confidence.** High. Residual (judge): Keeping `log_std` training follows from no source (the networks of Nikishin
et al. have no state-independent standard deviation); it is a symmetric choice fixed before data. The paper was read
only through two search-engine extracts of its Section 3, because the PDF could not be opened from the sandbox; Role
3 reading it is still worth recording, but the judge found that it could not overturn the decision, which follows
from the paper's stated property that injection leaves the number of trainable parameters unchanged.

**Why.** Table 2.4 "Partial reset" names "the output layer and the hidden layer before it"; `log_std` is a free
parameter, not a layer, so it is not reset. "Plasticity injection" cites Nikishin et al. (2023), whose method
freezes the old head and adds a trained and a frozen copy of a fresh head while the encoder keeps training, so the
number of trainable parameters and the output at injection are unchanged; "only the new copy is trained thereafter"
contrasts the three heads, and H3 (c) calls "the last two layers" the trainable layers even of the untreated arm,
whose first layer plainly trains. Freezing the trunk and `log_std` would make injection the only arm with far fewer
trainable parameters and fixed exploration noise, which would bias H3 (b). The dedicated generator keeps the seed
pairing of Part 5.1.

**Rejected.** The literal reading (trunk and `log_std` frozen after injection); resetting or freezing `log_std`; new
weights from the global random stream; post-injection metrics on 128 units.

**Implemented in.** No code change: `metrics.interventions.partial_reset`, `plasticity_injection`, `InjectedHead`
and `intervention_generator`; `envs.onset.OnsetMixin._apply_intervention` (once, at the start of the onset epoch's
rollout).

### Q-plasticity-definitions (starred)

**Answer.** The metrics are computed on the task's fixed batch (2,048 states, seed 0), normalised by the run's own
observation normaliser as saved at that checkpoint and frozen (never pushed). Activations are post-tanh. Dormant
score: equation (6) per layer against that layer's own mean, τ = 0.025. The dormant fraction is pooled over all
hidden units of the actor (128; 256 after injection). Effective rank: equation (7) with δ = 0.01, by uncentred
float64 SVD of the last hidden layer's activations (the three heads' concatenated last hidden layers after
injection). Parameter norm: the Euclidean norm of all trainable actor parameters, log_std included (frozen heads
excluded after injection; norm_all also logged). Edge cases: an all-zero layer counts as all dormant; an all-zero
matrix has rank 0; non-finite input gives NaN. The same three metrics are logged for each critic. For H3(c) the
trainable layer is the second hidden layer (mean[3]) for the untreated and reset arms and the new head's hidden
layer (new.3) for injection.

**Confidence.** High.

**Why.** Table 2.3 fixes the actor as the network measured (the dormant fraction over all hidden units, the
effective rank over the penultimate features, the norm over all trainable actor parameters, the critics logged
separately), the fixed 2,048-state batch, Sokar et al.'s score with τ = 0.025 (equation 6) and Kumar et al.'s
effective rank with δ = 0.01 (equation 7). The open details follow the sources and the network: the actor only ever
sees normalised observations, so the batch goes through the run's own frozen normaliser; Sokar et al. score
activations; with two 64-unit layers the pooled fraction equals the mean of the per-layer fractions; `log_std` is a
trainable actor parameter. For H3 (c) the trainable units of "the last two layers" are the hidden units in them,
since the output layer has no activation.

**Rejected.** A raw batch; pre-activation scores; a centred SVD; a norm without `log_std`; critic norms only; output
units in H3 (c).

**Implemented in.** No code change: `metrics.plasticity.measure`, `normalise_frozen`, `dormant_scores`,
`dormant_fraction`, `effective_rank`, `parameter_norm` and `actor_layers`; `metrics.hook` writes `plasticity.csv`.

### Q-data-control-lr (starred)

**Answer.** 'Additional constrained training' is one run of T + N·T steps (N·T = 1,000,000 / 2,500,000 / 5,000,000).
Its first T steps are, bit for bit, the untreated late arm of the same task and seed. The actor's learning rate
follows OmniSafe's LinearLR(start 1, end 0, total_iters = 500) for epochs 0-499, exactly as the untreated arm's
does. In the N·T/20,000 further epochs e = 500, ..., 500 + N·T/20,000 − 1, the rate is 3e-4 × (1 − N) × (1 − (e −
500)/(N·T/20,000)). The rate thus restarts at the rate at which the arm entered its constrained phase and decays
linearly to 0 at the end of the run. At N = 0.50 the extension repeats the untreated arm's constrained-phase rates
exactly. At N = 0.10 and 0.25 the extension's mean rate per epoch equals the untreated constrained phase's to within
one epoch's discretisation (both ≈ (1 − N)·3e-4/2). Nothing else changes: the network, the Adam moments, both
critics (constant LR), the multiplier and its Adam state, the normaliser and every random stream continue.
Matched-checkpoint selection then reads the run's last ten checkpoints, at the end of a decayed schedule, like every
other arm.

**Confidence.** High. Residual (judge): The extension needs some non-zero learning rate, and no source fixes its
level: (1 − N) × 3e-4 falling to 0 is fixed before data, with the same rule for every N. A restart from 3e-4 would
be more conservative for H3 (b), but it adds a larger warm restart (itself a plasticity-like change) and breaks the
symmetry with the reset and injection arms.

**Why.** Table 2.4 "Additional constrained training" says the late arm "continues under the constraint for N·T
further steps after T, with no change to the network or optimiser", so its first T steps must be the untreated late
arm bit for bit. The earlier proposal (one decay over the whole T + N·T run) changes the learning rate from the
first epoch, and so the state at onset, which breaks the seed pairing of Part 5.1. Continuing the untreated schedule
literally gives a learning rate of exactly 0 after T, which would make H3 (b)'s "additional constrained training does
not" [reduce the gap] true by construction, so some non-zero rate is unavoidable. Restarting at (1 − N) × 3e-4, the
rate at which the arm entered its constrained phase and at which the reset and injection arms resume training, and
decaying to 0 repeats the arm's own constrained-phase schedule (exactly at N = 0.50) and ends at 0 like every arm, so
the selection window is treated alike.

**Rejected.** The earlier proposal (one decay over 750, 625 or 550 epochs); continuing the untreated schedule (rate
0); holding the final rate (effectively 0); a constant tail (a selection window at a non-zero rate); a fresh decay
from 3e-4 (a warm restart above the arm's constrained rates).

**Implemented in.** `envs.onset.actor_lr_schedule` and `DataControlLR`, set by `build_onset_cfgs` for the
additional-constrained-training arm only; `pilot.manifest` queues it as one run of T + N·T steps.

### Q-determinism-late-onset (starred)

**Answer.** The study_a_pid determinism check trains a copy of seed 0 of the N = 0.50 PID arm (DET- run_id) twice,
for the registered-form 200,000 steps (10 epochs). The onset is moved to N × the check's own length, in whole
epochs: 100,000 steps (epoch 5), so both runs pass through the onset and five PID updates before the first
checkpoint at 200,000 steps is compared. Nothing else changes from the arm. The check's report gates the 15
PID-check runs.

**Confidence.** High.

**Why.** Table 3.1 "Determinism check" asks that two runs with the same seed and configuration give identical
evaluation cost at the first checkpoint; it names no arm, and its purpose is to confirm that the code path an arm
runs is deterministic. With the registered onset of 5,000,000 steps a 200,000-step check would never call the PID
update. Scaling the onset by the arm's N (100,000 steps, epoch 5) keeps the arm's onset fraction and exercises five
PID updates within the registered-form length, at no extra cost.

**Rejected.** The registered onset (the PID path untested); training to the first checkpoint after the registered
onset (5.2 million steps, twice); the N = 0.10 arm (still costly, and another arm).

**Implemented in.** No code change: `pilot.scheduler.determinism_spec` and `determinism_arm`;
`envs.onset.is_determinism_spec`; `scripts/determinism_check.py`.

### Q-lr-decay (note)

**Answer.** Keep OmniSafe's default linear_lr_decay: True in every arm. The actor's Adam LR (3e-4) decays linearly
to 0 over each run's own epochs (total/E): 500 for T-step runs, and 556, 667 and 1,000 epochs for the
constrained-steps-matched runs at N = 0.10, 0.25 and 0.50 (totals from Q-rounding). The critics' LR stays constant
at 3e-4. The paper names this as part of what onset timing manipulates. A late arm enters its constrained phase at
(1 − N) × 3e-4 under the total-steps control, and at about that rate under the constrained-steps control (0.899,
0.750 and 0.500 × 3e-4). Train/LR is logged every epoch. The single exception is the data control's extension
(Q-data-control-lr). Any constant-LR comparison would be an exploratory ablation (Table 3.3 'Ablations').

**Confidence.** High.

**Why.** Table 3.1 makes every remaining hyperparameter OmniSafe's default for the pinned version, copied verbatim
into Appendix A, and the pinned `PPOLag.yaml` has `linear_lr_decay: True`; the benchmark behind Table 3.5, d_loose
and G1's feasibility expectation was run with the decay. In OmniSafe's pipeline a constraint added late is
necessarily added at a lower learning rate, which is part of the practitioner's question the matched estimand
answers; Part 4.1.1 asks that systematic differences be named, so the decay is named, not removed.

**Rejected.** `linear_lr_decay: False` in every arm (contradicts Table 3.1 and Appendix A, departs from the
benchmark and leaves the selection window at the full rate); a decay over the constrained phase only.

**Implemented in.** No code change: OmniSafe's default; `pilot.launch.verify_installed_yaml` keeps the installed
configuration equal to the committed copy.

### Q-multiplier-adam (note)

**Answer.** The multiplier is updated once per epoch by OmniSafe's Lagrange.update_lagrange_multiplier: one Adam
step (lr 0.035, betas 0.9/0.999) on the loss −λ(J_C − d), then clamped to [0, upper bound]. Equation (4) gives the
direction and the fixed point of this update, not its step size. Under Adam, λ moves by about 0.035 per epoch while
the sign and size of J_C − d are steady, largely independent of how large the violation is. After a sudden rise in
the violation it moves by up to about 3 × 0.035. Through Adam's momentum it keeps moving in the old direction for a
few epochs after J_C − d changes sign. The amendment states this beside eq. (4), and H4's controller quantities
(overshoot, settling) are read on this update.

**Confidence.** High.

**Why.** The text of equation (4) says the update is "applied there through the optimiser named in the
configuration", and Appendix A copies the multiplier optimiser (Adam, learning rate 0.035), so Adam is registered
and equation (4) gives the direction and the fixed point of the update. The critic's simulation in the pinned torch
showed that the step is about 0.035 only while the violation is steady, grows to about three times that after a
sudden rise, and continues in the old direction for a few epochs after the sign changes. That momentum is part of
the overshoot H4 studies, so the amendment states it.

**Rejected.** Replacing Adam by the plain step λ ← max(0, λ + 0.035 (J_C − d)), which contradicts the registered
configuration.

**Implemented in.** No code change: `envs.onset.OnsetMixin._update` calls OmniSafe's `PPOLag._update` unmodified
from onset; `pilot.launch.verify_registered_defaults`.

### Q-onset-epoch (note)

**Answer.** Epoch e (0-based) is constrained iff e × 20,000 ≥ the onset step, i.e. e ≥ onset/E. The onset epoch's
data (steps [onset, onset + 20,000)) are the first data used by a constrained update. In that update, as in
OmniSafe, the multiplier is updated first (from 0.001, or from the warm-start value) and then the actor uses the new
value. A total-steps-matched late arm thus trains exactly (1 − N)·T steps under the constraint, a
constrained-steps-matched arm exactly T, and an N = 0 run is OmniSafe's PPO-Lagrangian bit for bit. The checkpoint
'at onset' (epoch-{onset/E}.pt) is the last unconstrained state.

**Confidence.** High.

**Why.** Table 2.4 says a late arm trains (1 − N)·T steps under the constraint, and the constrained-steps control T
steps. Counting a step as constrained when the update that uses its data is constrained, only "epoch e is
constrained if and only if e × 20,000 ≥ onset" gives exactly these numbers and keeps N = 0 identical to OmniSafe's
PPO-Lagrangian. OmniSafe's own order within an update (multiplier first, then critics and actor) applies to every
arm alike.

**Rejected.** The update at the onset step (the end of the epoch before) as the first constrained one (one epoch too
many); the actor updated before the multiplier in the onset epoch.

**Implemented in.** No code change: `envs.onset.OnsetMixin._update` and `_compute_adv_surrogate`.

### Q-manipulation-point (note)

**Answer.** The H3(c) manipulation check is read at onset + 200,000 steps in every arm it compares (the treated arms
and the untreated late arm of the same N): 1,200,000, 2,700,000 and 5,200,000 steps at N = 0.10, 0.25 and 0.50
(total-steps matched). It uses the dormant_trainable and rank_trainable values of the plasticity row at that step.
The first grid checkpoint after onset (2,600,000 at N = 0.25) is not used for the check.

**Confidence.** High.

**Why.** Box H3 (c) reads the check "at the first logging point after onset (200,000 steps)" and gives the reason:
after 200,000 steps rather than at onset, because injection changes nothing at that moment. The defined quantity is
therefore a fixed 200,000 steps of training after onset; at N = 0.25 the first grid point would give only 100,000.
Treatments run only in the total-steps-matched abrupt arms (Table 3.3), and both checkpoints are saved, so no rerun
depends on this.

**Rejected.** The first grid checkpoint after onset (2,600,000 at N = 0.25).

**Implemented in.** No code change: `pilot.contracts.extra_checkpoint_steps` saves onset + 200,000;
`pilot.enrichment` and `analysis.study_a.h3c_outcome` read it.

### Q-first-checkpoint (note)

**Answer.** 'The first checkpoint' of the Table 3.1 determinism check is the first scheduled checkpoint, at 200,000
steps (epoch-10.pt; the registered-form check trains exactly 200,000 steps). It is not the untrained epoch-0.pt, nor
an extra checkpoint at onset (for example the 100,000-step onset checkpoint of the study_a_pid check).

**Confidence.** High.

**Why.** Table 3.1 asks for identical evaluation cost "at the first checkpoint"; the purpose is to show that
training is deterministic, and `epoch-0.pt` is written before any update. The checkpoint grid of Table 2.3 starts at
200,000 steps, and in the `study_a_pid` check only that checkpoint follows the onset.

**Rejected.** `epoch-0.pt` (tests no training); the onset checkpoint of the PID check (the state before the
constraint).

**Implemented in.** No code change: `scripts/determinism_check.py` (`registered_form`);
`pilot.scheduler.determinism_spec`.

## 2. Evaluation battery, continuations, matching and controller quantities

### Q-continuations (starred)

**Answer.** Reward-only fine-tuning and transfer start from the parent's matched checkpoint (ledger
matched_checkpoint_step). Study B few-shot continuations start from the parent's final checkpoint. Every
continuation restores the parent's actor, both critics and observation normaliser, and the normaliser keeps
updating. Adam states are fresh. The actor learning rate decays linearly (OmniSafe's default LinearLR, from 1.0 to
0.0) over the continuation's full registered length: 50 epochs (1,000,000 steps) for fine-tuning, transfer and
few-shot. Fine-tuning holds the multiplier at 0 and trains the actor on the reward advantage only. Transfer and
few-shot start a fresh multiplier at 0.001 (lr 0.035; d = 25 for transfer, the unseen budget for few-shot). Every
battery continuation trains with PPO-Lagrangian, including the continuations of a PID-check parent. Under Part 6.1
cut 3, a shortened few-shot continuation keeps the 1,000,000-step (50-epoch) schedule and stops after 200,000 steps
(10 epochs). Its only reading is therefore the full continuation's first-horizon reading.

**Confidence.** High.

**Why.** Equation (1), Table 2.1 "Robustness gap" and Appendix B ("Robustness gaps of the matched checkpoint") put
fine-tuning and transfer on the matched checkpoint; Study B has no checkpoint selection and each finished run is
"continued once" (Part 3.3), so few-shot starts from the final checkpoint. The parent's schedule cannot be reused
(it is 0 at the final checkpoint and depends on the training age at a matched one), so every continuation gets
OmniSafe's default schedule over its own registered length, fresh Adam states and PPO-Lagrangian, the same
instrument for every arm; transfer follows the registered few-shot multiplier rule so that no arm-specific
controller state enters a policy gap. The one change concerns Part 6.1 cut 3, whose reason is that "the three
horizons are read from one continuation": with its own 200,000-step schedule a shortened continuation's mean rate
would fall from about 0.91 to about 0.55 of the initial rate, so it keeps the 1,000,000-step schedule and stops at
200,000, and its reading equals the full continuation's first horizon.

**Rejected.** Continuing the parent's schedule; a constant learning rate; carrying the multiplier and its Adam state
into transfer; restoring Adam moments; fresh critics for transfer; cut 3 with its own 200,000-step schedule (the
earlier proposal).

**Implemented in.** `pilot.dependencies.restore_learner`; `pilot.manifest.battery_continuation`, `study_b_fewshot`
and `_cut_fewshot_short`; `envs.continuations.FinetunePPOLag` and `TransferPPOLag`;
`studyb.conditioning.make_fewshot_algorithm`.

### Q-transfer-obs (starred)

**Answer.** SafetyCarGoal1-v0's held-out task is SafetyCarButton1-v0; the Point pair is as registered. The
first-layer weights of the actor (plain or injected trunk) and of both critics are mapped by Safety-Gymnasium
observation component name, read from both tasks at run time. Columns of components both tasks observe are copied.
For Point these are accelerometer, velocimeter, gyro, magnetometer, goal_lidar and hazards_lidar. For Car they are
the same plus ballangvel_rear and ballquat_rear. Columns of target-only components (buttons_lidar and gremlins_lidar
for Goal to Button; vases_lidar for Button to Goal) start at weight 0. Source-only components are dropped. Every
other tensor is copied unchanged. The normaliser copies the shared dimensions' statistics. Target-only dimensions
get mean 0 and variance 1 (_sumsq = _count - 1 under the parent's count) and the parent's clip. At the start of
transfer the networks' outputs therefore equal the parent's on the shared inputs.

**Confidence.** High. Residual (judge): Starting the target-only normaliser statistics at mean 0 and variance 1
under the parent's count is a convention no source fixes; it does not change the policy at step 0 (those columns
have weight 0) and is the same for every arm.

**Why.** Table 2.2 "Transfer" names "the held-out task on the same robot (Goal to Button and Button to Goal)", which
for SafetyCarGoal1-v0 can only be SafetyCarButton1-v0. The observation sizes differ (60, 76, 72 and 88), so a
mapping is unavoidable; mapping the first layer by observation component name, with zero weights for target-only
inputs, keeps the parent's function on everything both tasks observe, so the transfer gap measures how the parent
policy fares and not a re-initialisation. Zero-weight columns still receive gradients and learn. The critic checked
the layouts in the pinned stack and added the two shared Car components to the text.

**Rejected.** A union observation for every run (changes the registered networks); a fresh first layer (measures
re-learning); a random initialisation of the new columns (perturbs the parent differently per seed); target-only
statistics from the target task's batch (an extra data source); another held-out task for CarGoal1.

**Implemented in.** No code change: `envs.continuations.transfer_map`, `TransferMap`, `_map_first_layer`,
`_map_normaliser` and `map_state_for_transfer`; `pilot.manifest.TRANSFER_TASK_PROPOSALS` and `transfer_task`.

### Q-hazard (starred)

**Answer.** Hazard relocation (Table 2.2; the primary outcome and G3) re-samples the hazard positions with the
task's own layout generator, with the hazards' placement square set to |x|, |y| ≤ 0.75 instead of the task's
extents [-1.5, 1.5]^2. The generator shrinks this square by the hazard keepout 0.18, so hazard centres lie within
|x|, |y| ≤ 0.57 and the hazard discs (size 0.2) reach 0.77. Twenty layout seeds (3,000,000 + j) each fix the hazard
positions, which are pinned for five episode seeds (3,100,000 + 5j + k). Each episode seed re-draws every other
object (robot, goal, vase, buttons, gremlins) with the generator's usual rejection against the pinned hazards,
giving 100 distinct episodes. The policy is not updated. The registered-wording form (hazards from the unmodified
generator) is not a distribution shift and enters no result.

**Confidence.** High. Residual (judge): The 0.75 half-width (hazard centres within 0.57) has no published source. It
is fixed before data and identical for every arm and task; choosing it from pilot data would make G3 and the primary
outcome non-confirmatory.

**Why.** Safety-Gymnasium draws a new layout of every object at every reset, so training already samples the task's
whole layout distribution, and the literal condition of Table 2.2 ("re-sampled with the task's own layout generator
using twenty new layout seeds") has an expected gap of 0 for every arm. That contradicts the definition of the
battery as distribution shifts (Table 2.1 "Robustness gap", after Kirk et al., 2023) and would make the primary
outcome (Part 5.2) and G3 fail by design. The answer keeps everything else Table 2.2 registers (the task's
generator, 20 layouts × 5 episodes, no policy update) and changes only where the generator may place hazards, a
configuration of about 1e-6 probability under training. Pinning the hazards per layout while the episode seeds
redraw the other objects also removes the five identical deterministic episodes per layout; the form was checked on
all three Study A tasks.

**Rejected.** The literal form (not a shift); hazards placed on the robot-goal path (not the task's generator); more
or larger hazards (not relocation); the literal form run beside it (a forking path that predictably reports about
0); other widths.

**Implemented in.** No code change: `envs.evaluation.HAZARD_FORM = "central"` and `HAZARD_CENTRAL_HALF_WIDTH =
0.75`; `_HazardDrawer`, `draw_hazard_layout`, `EvalEnv.pin_hazards`, `hazard_definition`, `run_episodes` and
`evaluate_battery`.

### Q-dynamics (starred)

**Answer.** The dynamics perturbation scales the mass and the inertia of every body of the robot's kinematic tree by
1.3 (Point: agent; Car: agent, left, right, rear), followed by mujoco.mj_setConst. It scales all three friction
coefficients of the floor geom and of every robot geom by 0.7, so every robot-floor contact has 0.7 times its
original friction. The scaling is applied to the freshly compiled model after every reset, never compounded. No
other body or geom is changed (vase, buttons, gremlins, hazards). Because MuJoCo combines two geoms' friction by the
element-wise maximum, the robot's contacts with other objects keep their sliding friction (the object's 1.0). For
the Point robot, whose own torsional and rolling coefficients (0.01) exceed the objects' (0.005 and 0.0001), those
two coefficients of its object contacts fall to 0.007. This side effect is recorded. The 100 episodes use the
measurement seeds, paired with the measurement-set C_ID. The policy is not updated.

**Confidence.** High.

**Why.** Table 2.2 "Dynamics perturbation" reads "Body mass scaled by 1.3 and ground friction by 0.7 in the
simulator's model file", "a change large enough to alter how actions move the robot". In the pinned models every
contact combines the two geoms' friction by the element-wise maximum, so scaling the floor alone changes no
robot-floor contact, and "body mass" names no bodies. Scaling the floor and the robot's geoms by 0.7 gives exactly 0.7
times the original robot-floor friction, the stated purpose points to the robot's bodies, and Safety-Gymnasium
recompiles the model at every reset, so the scaling is reapplied after each compile. The critic corrected the claim
that object contacts are untouched: the Point robot's torsional and rolling friction with objects falls from 0.01 to
0.007, a small side effect, the same for every arm, which the answer records.

**Rejected.** Floor friction only (changes nothing); a raised floor priority (not 0.7 times the original contact);
every body scaled (objects the text does not aim at); sliding friction only; fresh seeds instead of the measurement
seeds (loses the pairing with C_ID).

**Implemented in.** No code change: `envs.evaluation.apply_dynamics_perturbation`, `DynamicsPerturbation`,
`dynamics_definition` and `evaluate_battery`.

### Q-selection-window (starred)

**Answer.** Part 4.1 rule 1's window is the ten checkpoints at total − k × 200,000 steps, k = 0..9, for every run.
For a total on the 200,000-step grid these are the grid's last ten checkpoints. For a total off the grid
(11,120,000; 13,340,000; 12,500,000) they form an end-relative grid that the run saves in addition to its
200,000-step grid. They are both 'the last ten checkpoints' and 'the final 2,000,000 steps'.

**Confidence.** High.

**Why.** Part 4.1 rule 1 says "the run's last ten checkpoints (the final 2,000,000 steps; decision)", and Table 8.2
adds "every arm alike"; for a total off the 200,000-step grid the two descriptions disagree on that grid. The ten
checkpoints at total − k × 200,000 satisfy both, keep the registered spacing and treat every arm alike relative to
its end of training; the extra saves are whole epochs and change no update.

**Rejected.** The last ten checkpoints of the absolute grid (they span less than 2,000,000 steps); every checkpoint
in the final 2,000,000 steps (eleven); rounding the totals onto the grid (changes the registered totals).

**Implemented in.** No code change: `pilot.contracts.end_relative_window`, `selection_window` and
`extra_checkpoint_steps`; `analysis.matching.selection_window`.

### Q-tie-break (starred)

**Answer.** When two or more checkpoints of the selection window are equally close to d = 25, rule 1 selects the
latest of them. Distances |selection_cost − 25| are compared after rounding to 1e-4.

**Confidence.** High.

**Why.** Safety-Gymnasium binarises the cost of each step, so 100-episode means lie on a 0.01 grid and exact ties
(24.50 against 25.50) do occur; rounding the distance to 1e-4 removes only floating-point noise. "Later" depends
only on the step, treats every arm alike and picks the checkpoint closest to the final-checkpoint estimand of rule 7
(b); preferring a cost below 25 would add a criterion that rule 1 ("closest to the budget") does not have.

**Rejected.** The earlier checkpoint; a preference for a cost at or below 25; a random choice; no rounding.

**Implemented in.** No code change: `analysis.matching.select_checkpoint` (`TIE_DECIMALS = 4`);
`pilot.enrichment.rule_one_step` recomputes it independently.

### Q-matched-cost-set (starred)

**Answer.** An arm's matched cost (rules 4 and 5), the reference cost and the reference seed-to-seed SD (rules 2, 3
and 5), and G1's 'final in-distribution cost' all use the measurement-set cost (100 episodes on the measurement
seeds) of each seed's selected checkpoint. The selection-set cost at the selected checkpoint is stored and reported
beside it, as Part 4.1.1 requires, but decides nothing.

**Confidence.** High.

**Why.** Table 2.1 keeps the selection set "only to choose one of them in Part 4" and names the measurement set for
the reported in-distribution cost and the baseline of every gap, warning that choosing and measuring on the same
episodes biases the chosen checkpoint's cost downward; G1 cites the same definition. Matching on the selection set
would use a statistic selected for its closeness to 25, which makes matching too easy and the SD too small.

**Rejected.** The selection-set cost; pooling both sets.

**Implemented in.** No code change: `analysis.matching.COST_FIELD = "measurement_cost"` (`apply_rules`,
`match_arms`); `pilot.go_decision.COST_FIELD` (G1).

### Q-arm-complete (starred)

**Answer.** An arm is complete when its completed (non-excluded, non-auxiliary) runs reach its seed target. The
target is 5, plus the Part 5.5 surplus count recorded in Part 8 for primary-comparison arms. Rules 2, 3 and 5's SD
condition (reference cost, reference SD, feasibility) wait until the task's reference (N = 0) arm is complete. An
arm's matched cost and flags (rules 4 to 6), and its battery continuations, wait until both the reference arm and
that arm are complete. Every completed seed then counts: replacement seeds take the place of excluded ones, and
surplus seeds join the mean and the SD.

**Confidence.** High.

**Why.** Rules 2 and 4 say "the mean over the five seeds", but Part 5.5 adds surplus seeds to the primary-comparison
arms, Part 5.6 replaces an excluded run with the next unused seed, and Table 8.1 records the seeds actually used; so
"five" is the default count, and matching must describe the same seeds the analysis uses. Waiting until an arm is
complete prevents flags computed on part of an arm and revised later. The critic corrected the scope to the code's
per-arm rule: a flag computed once the reference arm and the arm itself are complete can never be changed by another
arm's later seeds.

**Rejected.** Matching on seeds 0 to 4 and adding surplus seeds afterwards; flagging an arm as soon as five seeds
complete; waiting for every arm of the task (delay without protection).

**Implemented in.** No code change: `pilot.enrichment.seed_targets` and `arm_readiness`;
`analysis.matching.match_arms`; `analysis.data.study_a_seed_targets`.

### Q-controller-quantities (starred)

**Answer.** Controller quantities are computed at epoch resolution from Metrics/LagrangeMultiplier, the value after
each epoch's update.

- Peak: the maximum over the epochs that end in (onset, onset + 2,000,000].
- Final value: the mean over the last ceil(0.10 × epochs) epochs of the run's own length. Overshoot = peak − final.
- Settling time: the steps from onset to the end of the first post-onset epoch e* such that the value of e* and of
  every later epoch lies within ±10 percent of the final value (inclusive). If the final value is 0, this means
  the multiplier stays exactly 0 from e* on. It is none only if the last epoch's value lies outside the band.
- Recovery time: the steps from onset to the end of the first constrained epoch whose Metrics/BatchEpCost is at most 25.
  It is none if that never happens.
- In the analysis, a run that never recovers enters as (its steps from onset to the end of training) + one epoch. A
  contrast of two arms censors both at the shorter arm's horizon.

**Confidence.** High.

**Why.** Table 2.4 defines overshoot and settling time on the multiplier and Table 2.1 recovery time on the
training-batch cost; the multiplier changes once per epoch, so epoch resolution is the only exact reading, and
`Metrics/BatchEpCost` is the true per-epoch batch mean that "training-batch mean" names. The change: with a final
value of 0 the earlier proposal gave no settling time, but the registered definition is defined there (OmniSafe
clamps the multiplier at exactly 0, so a band of ±10 percent around 0 is {0}), and the settling time is the epoch
from which the multiplier stays at 0. Censoring a run that never recovers at its horizon plus one epoch keeps
"recovered in the last epoch" and "never" apart, and censoring both arms of a contrast at the shorter horizon stops
the data control's extra steps from showing a recovery the untreated arm could not show.

**Rejected.** No settling time for a final value of 0 (the earlier proposal); an absolute band near 0 (not
registered); survival analysis (Part 5 registers differences of seed means); interpolation within epochs.

**Implemented in.** `metrics.controller.controller_quantities`; `metrics.recovery.recovery_steps`;
`pilot.enrichment.controller_reference`, the independent recomputation that `enrich controller` compares with;
`analysis.study_a._recovery` and `_recovery_getter`.

### Q-final-cost-set (starred)

**Answer.** final_cost and final_return (Appendix B) are the mean cost and return over the 100 measurement-set
episodes at the run's final checkpoint. The final checkpoint's selection-set cost (it is always in the selection
window) is kept in the checkpoint record as final_selection_cost and decides nothing. This applies to Study A rows,
the final-checkpoint estimand of rule 7(b), G1's final-checkpoint SD reported beside the selected one, and the
pilot's unconstrained cost for G4.

**Confidence.** High.

**Why.** Appendix B's `final_cost` is an "evaluation over 100 episodes at the final checkpoint" and names no set.
Rule 7 (b) puts the final checkpoint "in place of the selected one", whose in-distribution cost and gap baseline are
measurement-set quantities (Table 2.1), so the final checkpoint is measured the same way. The selection-set cost of
the final checkpoint is biased whenever that checkpoint was itself selected, while the measurement set keeps final
and selected costs on the same footing, as G1's "same statistic for the final checkpoints" needs.

**Rejected.** The selection-set cost at the final checkpoint; pooling both sets.

**Implemented in.** No code change: `envs.evaluation.evaluate_run` (`final_seed_set` "measurement",
`final_selection_cost` kept); `pilot.ledger_writer`; `pilot.go_decision.g4_conditioning`.

### Q-eval-seeds (note)

**Answer.** The evaluation reset seeds are fixed in code and shared by every run, arm, task and study:

- selection set: 1,000,000 + i (i = 0..99);
- measurement set: 2,000,000 + i;
- hazard layouts: 3,000,000 + j (j = 0..19);
- hazard episodes: 3,100,000 + 5j + k (k = 0..4).

The sets are pairwise disjoint, disjoint from every training seed (0-12 and replacements) and from the fixed batch's
seed 0, and below 2^32. The dynamics condition uses the measurement seeds.

**Confidence.** High.

**Why.** Table 2.1 requires two disjoint sets of 100 episodes with different evaluation seeds, and Table 2.2 twenty
layout seeds with five episodes each; the values themselves are unregistered. Common seeds give every arm the same
episodes (common random numbers), which reduces the variance of differences between arms without bias. The ranges
are disjoint by construction and below 2^32, as Safety-Gymnasium requires.

**Rejected.** Seeds derived from each run's seed (loses the pairing across arms); small bases such as 100 + i (risk
of collision with training seeds).

**Implemented in.** No code change: `envs.evaluation.selection_seeds`, `measurement_seeds`, `hazard_layout_seeds`
and `hazard_episode_seeds`; `pilot.contracts.check_canonical_seeds`.

### Q-mujoco-exception (note)

**Answer.** An evaluation episode in which MuJoCo reports an unstable simulation
(mjWARN_BADQPOS/BADQVEL/BADQACC/BADCTRL counters, or Safety-Gymnasium's MujocoException branch) is never scored. It
is replaced by an episode on the next unused seed of that set's reserve sequence, with the same hazard layout or
budget:

- selection: 1,050,000 + r;
- measurement and dynamics: 2,050,000 + r;
- hazard episodes: 3,150,000 + r.

r counts 0, 1, 2, ... within one evaluation call (one checkpoint, one condition). A reserve episode that is itself
unstable is replaced the same way. Every unstable episode, original or reserve, counts toward a cap of 5 per
evaluation call. Each replacement (original seed, reserve seed, step, warning) is recorded in the evaluation result
and reported. Seed-set checks compare the planned canonical seeds. If more than 5 episodes of one evaluation call
are unstable, the evaluation fails with MujocoInstabilityError and is held for the group. This is not a Part 5.6
exclusion: the run is kept.

**Confidence.** High. Residual (judge): The cap of 5 unstable episodes per evaluation call is a constant without a
source, the same for every arm. Beyond it the evaluation is held for the group, a decision after data, but only for
a failure of a whole condition that would need an erratum anyway; with actions clipped to [−1, 1] the event is
expected never to occur.

**Why.** In the pinned stack an unstable simulation raises nothing: MuJoCo 2.3.0 warns and resets the state, so such
an episode is not a valid measurement. The earlier proposal failed the evaluation and left the episode's scoring to
the group after the data exist; Part 5.6 makes exclusion a rule about runs, so the run is kept, and a fixed
replacement from a shared reserve sequence, blind to outcomes, keeps the registered 100 episodes (Table 2.1) and
keeps arms on common seeds. A re-run would reproduce the failure, since evaluation is deterministic. The critic
added that the planned seed lists stay as they are (the checks compare them) and the replacements are recorded
separately.

**Rejected.** Failing the evaluation and letting the group decide (the earlier proposal); scoring the episode as
returned; dropping it (99 episodes); imputing a maximum cost.

**Implemented in.** `envs.evaluation.reserve_seeds`, `unstable_replacements`, `EvalEnv._check_stable` and
`run_episode`, `Replacement`, `MujocoInstabilityError`; `pilot.contracts.RESERVE_SEED_OFFSET` and
`MAX_UNSTABLE_EPISODES`.

### Q-return-outcome (note)

**Answer.** The secondary outcome 'return' (Part 5.2) of Study A is the mean undiscounted episodic return of the
matched checkpoint on the 100 measurement-set episodes. The harness reports it as measurement_return, stored with
its per-episode returns in the 'measurement' supplement record, since Appendix B has no matched-checkpoint return
field. final_return (Appendix B) serves the final-checkpoint estimand.

**Confidence.** High.

**Why.** Part 5.2 lists return among the secondary outcomes, Part 5.1 compares arms at their matched checkpoints,
and the matched checkpoint's in-distribution evaluation uses the measurement set (Table 2.1). Appendix B has only
`final_return`, so the field is a recorded divergence held in the supplement (Q-ledger-v2).

**Rejected.** `final_return` (not the matched estimand); the selection-set return (selection bias).

**Implemented in.** No code change: the `measurement` record of `results/supplement_schema.py`;
`analysis.study_a.MEASUREMENT_RETURN`.

### Q-checkpoint-table (note)

**Answer.** The per-checkpoint selection-set costs and returns (and the step and path of every checkpoint) are
stored as a nested list inside each run's ledger row (schema v1), not in a separate checkpoint table joined by
run_id. Table 9.1 records this as a storage divergence from First Tasks' proposed second table; Appendix B's content
is unchanged.

**Confidence.** High.

**Why.** Appendix B lists "checkpoints | Paths and steps", and Part 3.4 requires the selection-set cost and return
of the final ten checkpoints; where they are stored has no effect on any estimate, verdict or seed set. Schema v1 is
written and tested, and a second table would add a join without any gain for inference.

**Rejected.** A separate checkpoint table joined by `run_id` (First Tasks' proposal).

**Implemented in.** No code change: the nested `checkpoints` list of each row in `results/ledger_schema.py`.

### Q-rule3-tolerance-sensitivity (note)

**Answer.** In the exploratory sensitivity analysis of Part 4.1 rule 7(a) ('once with tolerance 5.0'), the tolerance
is 5.0 wherever Part 4.1 uses it. Rule 5's matching criterion becomes |matched − reference| ≤ 5.0, and rule 3's
feasibility condition becomes reference cost ≤ d + 5.0 = 30. The SD threshold of 10 is unchanged. The result stays
labelled exploratory.

**Confidence.** High.

**Why.** Rule 3 ("reference cost ≤ d + tolerance") uses the term that rule 5 defines, and rule 7 (a) repeats the
analysis "with tolerance 5.0", which substitutes the defined quantity wherever it occurs; the SD threshold of 10 is
a separate number. "The tolerance is not widened" governs the primary analysis, while rule 7 is an exploratory
repetition that widens it by design.

**Rejected.** Widening rule 5 only.

**Implemented in.** No code change: `analysis.matching.apply_rules` passes the tolerance to `budget_feasible` (rule 3)
and `within_tolerance` (rule 5).

## 3. Study B training, evaluation and the targeted search

### Q-pilot-unconstrained-seed (starred)

**Answer.** The Study B pilot's unconstrained PPO run (P-B-unconstrained-s0) uses seed 0, the seed of the Moderate
pilot run (Part 3.6). In the re-pilot (P1-) it also uses seed 0. The seed is fixed now, before any data exist, and
is not chosen again after a result is seen.

**Confidence.** High.

**Why.** Part 3.6 gives seed 0 for the Moderate run but no seed for the unconstrained run. Every registered seed
list starts at 0, and seed 0 gives the unconstrained run the Moderate run's sequence of layouts. The run serves only
the unconstrained cost that G4 checks every budget against, so the only requirement is that the seed is fixed before
data.

**Rejected.** Another fixed seed such as 5 (sharing seed 0 with the Study A pilot is harmless, since pilot runs are
not reused, Part 3.6); several unconstrained seeds (not registered).

**Implemented in.** No code change: `pilot.manifest.PILOT_UNCONSTRAINED_SEED = R.PILOT_MODERATE_SEED`;
`pilot.manifest.pilot` gives the re-pilot the `P1-` prefix.

### Q-budget-normalisation (starred)

**Answer.** Budget/100 is appended to the observation after OmniSafe's observation normaliser, and after the whole
wrapper chain, as OmniSafe's own Sauté adapter appends its safety state. The normaliser keeps the task's size (60 on
SafetyPointGoal1-v0) and never sees the budget. The actor, the reward critic, the cost critic and the buffer take
one more input (61). Evaluation appends the budget the same way, after the frozen normaliser
(envs.evaluation.append_budget), and refuses a checkpoint whose normaliser covers the budget feature. This applies
to every budget-conditioned run, the pilot's Moderate run and the few-shot continuations included.

**Confidence.** High.

**Why.** Table 2.5 "Budget conditioning" says "the active budget divided by 100 is appended to the observation
vector", and Table 3.1 "Networks" calls it "the normalised budget", that is budget/100. OmniSafe's normaliser
standardises every feature with running statistics: a constant feature becomes 0, so in the three Single arms the
budget would always read 0 in training and every unseen budget would saturate at ±5, making zero-shot generalisation
untestable by construction. Appending after the normaliser delivers the registered value budget/100 and follows
OmniSafe's own Sauté adapter.

**Rejected.** Normalising the budget with the other features; standardising it with fixed constants (not "divided by
100"); no observation normalisation in Study B (changes a registered default).

**Implemented in.** No code change: `studyb.conditioning._AdapterMethods`; `envs.evaluation.append_budget`, which
refuses a normaliser that covers the budget feature.

### Q-level-jc (starred)

**Answer.** A level's J_C is the mean episodic cost of that level's episodes among the episodes that form J_C for a
single multiplier under Q-jc-window. Under Q-jc-window's answer, these are the last 50 finished episodes
(OmniSafe's Metrics/EpCost window, length read from the logger), averaged as OmniSafe's logger averages that window.
A level with no episode in that set takes no step of equation (4) in that epoch: its multiplier and its Adam state
stay unchanged. A Single arm therefore updates its multiplier from exactly the J_C that PPOLag uses. The rule is the
same for every arm and for every Continuous bin. If the group changes Q-jc-window to the per-epoch batch mean at
ratification, the set becomes the level's episodes that finished in the current epoch.

**Confidence.** High.

**Why.** Table 2.5 says each multiplier "is updated only from episodes run at that level, by equation (4)", whose
J_C is the single-multiplier J_C (Q-jc-window); taking that same set of episodes and filtering it by level stays
faithful whatever that key decides, and keeps a Single arm identical to PPO-Lagrangian. A window defined in time
gives every level and every arm the same lag, where a window of each level's own last 50 episodes would give Dense
five times Single's lag. A level with no episode takes no optimiser step, since under Adam a zero gradient still
moves the multiplier; this is rare (about 1e-5 per epoch for Dense) and logged.

**Rejected.** The level's per-epoch batch mean (that is Q-jc-window's question); a window of the level's own last 50
episodes (a lag that depends on the arm); an empty level updated with J_C = d.

**Implemented in.** No code change: `studyb.conditioning._PPOLagMethods._update_level_multipliers` and
`_ensure_level_window` (`Metrics/LevelWindow` logged).

### Q-continuous-bins (starred)

**Answer.** The Continuous arm has six multipliers, one per 5-unit bin: [10, 15), [15, 20), [20, 25), [25, 30), [30,
35), [35, 40]. The last bin is closed, so 40 lies in [35, 40]. An episode's or sample's bin is read from its float32
budget feature: a budget whose feature is at least the feature of an edge lies in the bin that edge opens, so 35
opens [35, 40]. The same function, studyb.conditioning.level_index, assigns both the per-sample penalty of equation
(3) and the episodes that enter a bin's J_C. In each update of equation (4), a bin's d is the mean budget of that
bin's episodes in its J_C set (Q-level-jc). The bin's multiplier then takes one step of equation (4) through the
configured optimiser (OmniSafe's Lagrange.update_lagrange_multiplier, Adam), with J_C − d = mean(C_e) − mean(b_e)
over those episodes, and is clamped at 0. Each multiplier is named and logged by its bin's lower edge:
Metrics/LagrangeMultiplier/level_10, level_15, ..., level_35.

**Confidence.** High.

**Why.** Table 2.5 gives "one multiplier per 5-unit bin of the range" [10, 40], hence six bins; half-open bins with
the last one closed are the standard histogram convention and cover the range once. Reading the bin from the float32
budget feature, the only budget information in the buffer, makes a sample's penalty and its episode's J_C use the
same multiplier. A bin's d is the mean budget of its episodes, so the update settles where each episode's cost
equals its own budget on average; the midpoint is noisier, and an edge would shift Continuous by 2.5 units against
the discrete arms and bias G5. The critic replaced a plain-gradient formula in the text by one step of the
configured optimiser, as the text of equation (4) requires.

**Rejected.** d at the bin midpoint; d at the lower edge; bins read from the float64 budget instead of the feature.

**Implemented in.** No code change: `studyb.conditioning.level_index`, `continuous_edges`, `level_column` and
`_PPOLagMethods._level_budget`.

### Q-search-before-pilot (starred)

**Answer.** Appendix C's targeted search counts the pilot's two Study B runs, the unconstrained PPO run included, as
Study B's first runs. The search record must be complete and committed in studyb/search/, and its path entered in
Table 9.0, before either of those runs starts. Study A's pilot runs do not wait for the search. The Table 3.1
determinism checks of plug-ins study_b and unconstrained_ppo are software verification, not Study B runs, and do not
wait for it either. If the outcome is 'partial', the amendment that narrows Study B's claims must be ratified and
committed (a Table 9.1 row with an analysis/AMENDMENTS.json entry) before those runs start. decision.json's
'amendment' field names that entry's id. If the outcome is 'included', Study B is withdrawn by amendment and none of
its runs, pilot runs included, is launched.

**Confidence.** High.

**Why.** Part 7.2 and Appendix C say the search is run, and its record committed, "before Study B's first run", and
Part 3.6 lists the pilot's two Study B runs, the earliest Study B runs. An included work withdraws Study B (Table
C.1 "Decision"), so running its pilot first would spend compute on a study that might be withdrawn and let the
novelty judgement be made with pilot data in view. The change: Table C.1 says a partial overlap narrows Study B's
claims "by amendment, recorded before the first run", so for "partial" that amendment must be ratified and committed
before the pilot's Study B runs, and the existing `amendment` field of `decision.json` names its entry. The Table
3.1 determinism checks are software verification, not Study B runs.

**Rejected.** "First run" read as the first main-study Study B run (the pilot's runs are Study B runs by Part 3.6);
holding the Study A pilot too (the text ties the search to Study B only).

**Implemented in.** `pilot.scheduler.Scheduler.launch_gates` and `pilot.scheduler.search_record_status` (which looks
the amendment up in the committed `analysis/AMENDMENTS.json`); `studyb.search.record` (`decision.json`'s `amendment`,
`SearchStatus.amendment`).

### Q-studyb-order (starred)

**Answer.** A non-pilot Study B training run (plug-in study_b, group study_b, surplus and replacement seeds
included) starts only after every run of Study A's main sweep has finished training. The main sweep means every run
of design 'main' under the Part 6.1 cuts in force (195 runs without cuts), plus every queued group-'main'
replacement or surplus seed. An excluded main-sweep run counts as finished. Pilot runs are outside this rule (Part
3.6 precedes the go decision). Few-shot continuations follow their parents and so come after the main sweep too.
Study A's treatment, controller, PID and battery runs may run alongside Study B and do not hold it.

**Confidence.** High.

**Why.** Part 3.3 says "Study B's runs are scheduled after Study A's main sweep"; read as an order in time, it stops
Study B from using compute before the main sweep, which carries the primary outcome, is secured. The text names only
the main sweep, so treatment, controller and PID runs do not hold Study B. The critic added that the main sweep is
the one left by the Part 6.1 cuts in force and that an excluded run counts as finished, so the hold cannot become
permanent; the order changes no result.

**Rejected.** Queue priority only (Study B could fill idle slots first); after all of Study A (the text says main
sweep); after every main-sweep run has started (does not protect replacements).

**Implemented in.** No code change: `pilot.scheduler.Scheduler._main_sweep_unfinished` and `launch_gates`
(`main_sweep_unfinished`).

### Q-studyb-eval (starred)

**Answer.** Study B training runs are evaluated at their final checkpoint, and each few-shot continuation at its
checkpoints at 200,000, 500,000 and 1,000,000 steps (or at its only horizon under Part 6.1 cut 3). Every Study B
evaluation with a budget in the observation uses Table 2.1's measurement set, with the same 100 episode seeds at
every budget and horizon and in every arm. This covers zero-shot at the unseen and reference budgets, few-shot, and
the pilot's G4 measurement of the Moderate arm at 10, 20 and 40. For the contract-1 fields Part 3.4 and Appendix B
require of every run, episode i of the 100 carries the arm's i-th training budget in turn, sorted(levels)[i mod k].
These fields are the selection-set cost and return at the last ten checkpoints and final_cost/final_return on the
measurement set at the final checkpoint. For Continuous, episode i carries the midpoint 10 + 30·(i + 0.5)/100
(10.15, 10.45, ..., 39.85). These contract-1 fields are descriptive and enter no Study B hypothesis.

**Confidence.** High.

**Why.** Part 3.3 evaluates "each finished run", Appendix B gives checkpoint selection to Study A only, and Part 4.2
defines Study B's comparisons without selection, so Study B uses the final checkpoint. Table 2.5's zero-shot
evaluation is "Decision, as in Table 2.1", where the selection set serves only to choose; Study B chooses nothing,
so its reported quantities use the measurement set, with common seeds across budgets, horizons and arms. A
budget-conditioned policy has no cost without a budget, so the contract-1 fields that Part 3.4 and Appendix B
require carry the arm's own training budgets in turn (for Continuous a stratified cover of [10, 40]); they are
deterministic and enter no hypothesis.

**Rejected.** Part 4.1's checkpoint selection for Study B; a separate seed set; random budget draws, or a single
budget such as 25, for the contract-1 costs.

**Implemented in.** No code change: `studyb.evaluation.evaluate_run`, `evaluate_training_budgets`,
`evaluate_zero_shot` and `evaluate_fewshot`; `envs.evaluation.training_budget_schedule`.

### Q-adapt-censoring (starred)

**Answer.** 'Reaches 0.80' means a satisfaction rate of at least 0.80 (80 or more of the 100 episodes within the
budget). A budget whose continuation never reaches 0.80 at any of its horizons records adapt_steps = the
continuation's largest horizon + 1: 1,000,001 for the registered horizons, or 200,001 under Part 6.1 cut 3. Readers
decode a value greater than the largest horizon as censored ('above the largest horizon'). Medians and the G4
readings treat a censored value as ranking above every observed horizon (Q-g4-reading).

**Confidence.** High.

**Why.** Table 2.5 "Adaptation steps", the passage that defines the quantity, says "recorded as above the largest
horizon if never reached"; Part 4.2's "censored at the largest horizon" summarises it. Recording the largest horizon
itself would merge "never" with "reached at 1,000,000"; the largest horizon + 1 is the smallest integer strictly
above it, fits the registered non-negative integer field, and keeps medians and orderings exact. "Reaches" is
inclusive in ordinary use.

**Rejected.** Recording the largest horizon; a null with a flag (drops censored budgets from medians); "strictly
greater than 0.80".

**Implemented in.** No code change: `studyb.evaluation.censored_steps` and `adaptation_steps`;
`pilot.enrichment.adaptation_steps`; `analysis.data` decodes a value above the largest horizon as censored.

### Q-search-depth (note)

**Answer.** Screen the first 50 results of each of the 80 searches (20 query variants × 4 sources), in each source's
relevance order. In arXiv this means choosing 'Relevance', not the default newest-first. A source that offers no
relevance order uses its default order, named in the search log's notes. Fix the depth and the order in
search_log.md before the first query. Record results_reported and results_screened = min(50, results_reported).
Citation lists (Q-search-citations) have no depth limit: each list is screened in full, and citations_log.csv
records both the list's length as the source shows it (items_reported) and the number screened, which must be equal.

**Confidence.** High. Residual (judge): The depth of 50 has no published source (only First Tasks' "for example the
first 50"); it is fixed before the first query. `items_reported` is the number of entries a list actually shows, not
Google Scholar's approximate "Cited by N".

**Why.** Table C.1 fixes the sources, queries and record but no screening depth, and Appendix C asks that the search
be judged against a fixed procedure; First Tasks gives 50. A 50-deep cut means something only on a list ranked by
relevance, and arXiv sorts by date unless asked, so the order is fixed with the depth (the change). Citation lists
are screened in full, and recording each list's length makes "in full" checkable.

**Rejected.** 100 or 200 per search (two to four times the effort for little gain on ranked lists, with citation
tracking as a backstop); a stopping rule decided during the search.

**Implemented in.** `studyb.search.protocol.SCREENING_DEPTH` and `RESULT_ORDER`; `studyb.search.record` (the `Result
order:` line of `search_log.md`; `items_reported` equal to `items_screened` in `citations_log.csv`).

### Q-search-queries (note)

**Answer.** Each of the five Table C.1 queries is run in four variants: as written, with "number of thresholds"
appended, with "spacing" appended, and with both appended. Each string is used verbatim (quotes kept) in each of the
four sources: 20 strings × 4 sources = 80 searches. Any source-specific syntax needed to express a string is
described in the search log's notes.

**Confidence.** High.

**Why.** Table C.1 lists five queries "each with and without 'number of thresholds' and 'spacing'", which can be
read as two, three or four variants. Running all four is the superset, so no reading's variant is skipped, at the
cost of 40 more searches.

**Rejected.** Two variants (neither phrase, both) or three (each phrase alone): each is a subset that leaves one
reading unmet.

**Implemented in.** No code change: `studyb.search.protocol.query_variants`; `python -m studyb.search queries`.

### Q-search-citations (note)

**Answer.** The citation lists screened are (1) Yao et al. (2023)'s reference list, (2) Yao et al.'s 'cited by'
lists in Google Scholar and in Semantic Scholar, and (3) for every work in the union of those two cited-by lists
(deduplicated, as of the search date), its reference list and its cited-by lists in Google Scholar and Semantic
Scholar. They also include (4) the reference list of every other work taken to full text, the Table C.1 known works
included. Every list is screened in full at title and abstract, with no depth limit. An item already screened
elsewhere is marked 'duplicate' and names, in duplicate_of, the exact citation text of its first screened
occurrence. Each list is logged in citations_log.csv, and its items are listed in
`screened/citations__<list_id>.csv`. If Google Scholar refuses access for a list in (3), its Semantic Scholar list
alone is screened, and the refusal is recorded in that row's notes.

**Confidence.** High. Residual (judge): Whether "citation list" means a work's references or its "cited by" list
cannot be settled by evidence; the superset satisfies both readings. If a citing work's reference list cannot be
obtained, the list is logged with 0 items and a note; the code accepts that, though the answer does not say so.

**Why.** Table C.1 "Sources" names "the citation lists of Yao et al. (2023) and of every paper that cites it"; the
earlier proposal took the lists only of the citing papers taken to full text, which narrows "every" by the
searcher's own first-stage judgement. Taking both kinds of list (references and "cited by") for every citing paper
satisfies the text under both readings of "citation list" and makes it harder, not easier, to claim novelty. The
critic closed two gaps: a citing paper first met in a database search is a duplicate that still counts
(`duplicate_of` names its first occurrence), and the fallback when Google Scholar refuses access is part of the rule
rather than an unregistered deviation.

**Rejected.** The earlier proposal (the lists of full-text works only); reference lists only (fails the forward
reading).

**Implemented in.** `studyb.search.protocol.CITING_WORK_LISTS` and `REFUSAL_NOTE`; `studyb.search.record`
(`citations_log`, and the `duplicate_of` column of the screened files).

### Q-search-record-scope (note)

**Answer.** Every screened result of every search and of every citation list is a 'hit'. Each is listed in its
screened/ file with its citation, its URL or DOI ('none' if the source gives neither), its title-and-abstract
decision (exclude, full_text or duplicate) and a non-empty reason. For an excluded result, the reason names the
element of the inclusion criterion it fails (no variation of budget levels, or no unseen budget evaluated) or the
Table C.1 exclusion clause it falls under. For a duplicate, the reason names its first screened occurrence. Every
work taken to full text, and each of the three Table C.1 known works, gets a search_record.csv row with all Table
C.1 fields: citation, what it varies, what it evaluates, whether it meets the inclusion criterion (yes, partial or
no) with the reason and the exclusion clause, where it was found, the date and the searcher's name. Results beyond
the screening depth are counted in results_reported but not listed.

**Confidence.** High. Residual (judge): Reading "every hit" in two tiers is an interpretation, anchored in First
Tasks ("one row per relevant hit"; "record every hit that reaches the full-text stage") and in two-stage reporting
as in PRISMA. That a reason names the failing element of the criterion rests on the searcher: the code can only
check that a reason is there.

**Why.** Table C.1 "Record" asks, "for every hit", for the citation, what it varies, what it evaluates, whether it
meets the inclusion criterion, the date and the searcher's name. Every screened result is listed with its citation,
URL or DOI, decision and a reason that carries its inclusion verdict; each work read in full, and each of the three
known works, gets every field. The critic found that the code did not require the reason to be filled; it now does.

**Rejected.** Every Table C.1 field for every screened result (no information for results excluded on title);
recording only full-text works (fails "for every hit").

**Implemented in.** `studyb.search.record` (`screened()` requires `url_or_doi`, where `none` is allowed, and a
reason on every row).

## 4. Operations, seeds, the go decision and the registration records

### Q-interrupted-run (starred)

**Answer.** A run stopped from outside (power loss, reboot, shutdown, SIGTERM, SIGINT or SIGHUP; or a run that ended
while no scheduler was its parent) is not an exclusion: it is restarted from scratch with the same seed and
configuration on the commit of its interrupted attempt, with no limit on the number of such restarts, and every
attempt is logged and counted in the run's ledger notes. A run that ends without a final training result while a
live scheduler watched it, through SIGKILL (for example from the OOM killer), another non-crash signal, or a
non-zero exit without train_result, is also restarted with the same seed. After MAX_RESTARTS = 2 such restarts, a
third such ending counts as 'fails to complete its steps' (Part 5.6, cause 'incomplete'): the run is excluded,
recorded in the ledger with its cause, and repeated with the next unused seed, like any other exclusion. Registered
data roots run under `schedule set-policy restart`. Once this key is answered, the policy 'exclude' is refused on a
registered root. No interrupted run waits for a group decision.

**Confidence.** High. Residual (judge): `MAX_RESTARTS = 2` for endings of process origin is a constant without a
source, fixed before data and the same for every arm. A restart from scratch with the same seed involves no choice
based on results.

**Why.** Part 5.6 excludes a run only for failures of the run itself (a crash, a failure to complete, a non-finite
loss or multiplier) and never on its results; a power cut or a reboot is not a property of the run, and a restart
with the same seed keeps the seed pairing of Part 5.1, where an exclusion would replace it with seed 5 at the same
cost. The change: the earlier proposal held a run "for the group" after two restarts of any kind, a decision about
one run after data exist that also counted genuine outages. Now only endings of process origin count (SIGKILL such
as the OOM killer, another non-crash signal, a non-zero exit without a result), and a third such ending is Part
5.6's "fails to complete its steps": an ordinary exclusion and replacement. The critic added SIGHUP (a closed
terminal) to the signals of machine origin and the refusal of the `exclude` policy once the key is answered.

**Rejected.** The earlier proposal (held for the group after two restarts); every interruption treated as
"incomplete" (breaks the pairing for nothing).

**Implemented in.** `pilot.scheduler.Scheduler._finish_training`, `_restart_by_rule`, `_default_policy` and
`_require_policy_answered`; `pilot.scheduler.MAX_RESTARTS`; `pilot.launch.INTERRUPT_SIGNALS` (SIGTERM, SIGINT,
SIGHUP).

### Q-seed-collision (starred)

**Answer.** Every arm, pilot arms included, takes for a replacement (Part 5.6) and for a surplus seed (Part 5.5) the
smallest seed ≥ 5 that the arm has not used for any purpose (pilot.scheduler.Scheduler.next_unused_seed). Surplus
seeds are added with `schedule add-surplus` at the go decision, before any main-study run of those arms can be
excluded, so they are 5, 6, ... as Part 5.5 orders them and later replacements follow them; arms pair on equal seed
ids (Part 5.1).

**Confidence.** High.

**Why.** Part 5.6 repeats an excluded run "with the next unused seed (5, 6, and so on)", and Part 5.5 adds surplus
seeds "in the order 5, 6, 7, and so on, the same count for every arm"; the smallest unused seed of 5 or more
satisfies both, never reuses a seed and keeps the count equal. Pilot arms are separate arms whose runs are not
reused (Part 3.6). The critic noted one condition for the order: queue the main sweep and run `add-surplus` before
`schedule run`.

**Rejected.** Plain "next unused" (3 for the pilot, against "(5, 6, and so on)"); a separate range for replacements
(against the same).

**Implemented in.** No code change: `pilot.scheduler.Scheduler.next_unused_seed` and `add_surplus`.

### Q-surplus-arm-set (starred)

**Answer.** The surplus seeds of Part 5.5 go to every arm the passage lists, with no onset shape excluded:

- on SafetyPointGoal1-v0, the N = 0 arm and the four N = 0.50 arms (abrupt and ramp, each under total-steps and
  constrained-steps matching);
- the seven Study B arms, with their few-shot continuations.

Every arm in the set receives the same number of surplus seeds, and its seed target is 5 plus that number
(Q-arm-complete).

**Confidence.** High. The coordinator upheld the critic's dispute (see Why).

**Why.** The decider first proposed giving surplus seeds only to the abrupt N = 0.50 arms, as the arms that carry
the primary comparison. The critic disputed this and the coordinator upheld the dispute: Part 5.5 lists "the N = 0
and N = 0.50 arms under both step-matching controls on SafetyPointGoal1-v0" with no onset shape, Part 6.1 uses the
same phrase, and treating both shapes alike keeps H2's comparison at N = 0.50 balanced. The answer is therefore the
earlier proposal.

**Rejected.** The abrupt arms only (narrows an explicit registered list by its purpose clause, and would depend on
Q-h1-shape); the N = 0 and abrupt total-steps arms only (against "both step-matching controls").

**Implemented in.** No code change: `pilot.manifest.is_primary_comparison` and `surplus_spec`;
`pilot.budget.primary_comparison_specs`.

### Q-g1-level (starred)

**Answer.** Part 6 G1's 'Both pilot arms reach a final in-distribution cost ... of at most d + 2.5' is read on each
arm's mean over its three seeds of the selected checkpoints' cost, as Part 4.1 defines an arm's matched cost and the
reference cost. The cost uses the evaluation set fixed by Q-matched-cost-set and is compared with 27.5 under the
arithmetic of Q-threshold-arithmetic. The per-seed result is reported beside it and decides nothing.

**Confidence.** High.

**Why.** Part 6 G1 refers to "the cost of the checkpoint selected by Part 4" and to its tolerance, and Part 4.1
defines the reference and matched costs as means over seeds; G1 is the pilot check of exactly that feasibility rule
("checked in the pilot (G1)"), so the arm mean decides, and the SD clause controls the spread between seeds. The
critic made the answer refer to Q-matched-cost-set and Q-threshold-arithmetic rather than restate them.

**Rejected.** Every seed at most 27.5 (a criterion the main study never applies).

**Implemented in.** No code change: `pilot.go_decision.g1_feasibility` and `_g1_clauses`.

### Q-g3-pairing (starred)

**Answer.** Equation (12)'s U80 = mean + 1.886 × s / √3 is the paired form: s is the sample SD of the three
per-seed differences gap_hazard(N = 0.50) − gap_hazard(N = 0), paired by seed id (t quantile 0.90 with 2 degrees of
freedom). If a replacement seed leaves fewer than three seed-matched pairs, the runs left unpaired are paired in
increasing seed order (each arm's remaining runs sorted by seed, then matched one to one) and equation (12) is
applied unchanged to the three differences; the mean Delta(0.50) is the same under any pairing, the pairing used is
reported, and G3 is decided by the rule, never by the group. The pooled-SD value is reported only as a descriptive
figure, not as a reading of equation (12).

**Confidence.** High. Residual (judge): Some rule for unmatched runs has to be chosen. Pairing them by seed order is
fixed before data and blind to results, and any fixed pairing gives the same mean.

**Why.** Equation (12) has a single mean, divides by √3 and uses t(0.90, 2 df) = 1.886, the one-sample bound on
three values; its prose speaks of three values, G3 and Table 8.1 of Δ(0.50) "over the three seeds", and Part 5.1
pairs seed k with seed k. A two-sample bound would need 4 degrees of freedom and s_p√(2/3), so the coded pooled
value understates the standard error and is reported only as a description. The change: the earlier proposal left G3
to the group when a replacement seed leaves fewer than three pairs, a go condition decided after the data are seen;
pairing the leftover runs in seed order keeps equation (12) exactly, and under the positive seed correlation the
design assumes it can only raise U80, which makes a no-go claim harder.

**Rejected.** The earlier proposal (the group decides with fewer than three pairs); a two-sample Welch bound
(departs from equation 12); the pooled form as decisive.

**Implemented in.** `pilot.go_decision._seed_pairs` and `_g3_signal`; `analysis.stats.g3_pairs` (the same pairing).

### Q-g4-level (starred)

**Answer.** G4's level clause is Part 6's: the Moderate arm's mean cost at each of its training budgets (10, 20, 40)
is at most that budget plus 2.5; the two-sided reading of Part 3.6 ('within 2.5 of its budget') is reported beside
it and decides nothing.

**Confidence.** High.

**Why.** Part 6 is the go rule that defines G4 ("each at most its budget plus 2.5 ... the level is the test that the
per-level multipliers work"); Part 3.6 summarises it with a reference back to G4. A one-sided clause is what a
Lagrangian constraint asks, as G1's "at most d + 2.5" is, and undershooting is caught by the ordering clause.

**Rejected.** Part 3.6's two-sided "within 2.5 of its budget".

**Implemented in.** No code change: `pilot.go_decision.g4_conditioning`.

### Q-g2-run-equivalents (starred)

**Answer.** Amend G2 before the pilot: the run-equivalents in G2 and in Part 5.5's surplus rule are training steps
in units of T = 10,000,000 (as Part 3.3 and Part 6.1 count them), so the decisive requirement is mean wall-clock
hours per pilot run / runs sustained concurrently × the corrected total of pilot.budget.corrected_run_equivalents()
for the design in force (627.13 for the uncut design: 540.13 of training runs plus 87.0 for the battery's
fine-tuning and transfer continuations). The registered 484 (435 + 49) is reported beside it and decides nothing;
the surplus seeds are computed from capacity beyond the corrected total, each priced in the same units.

**Confidence.** High. Residual (judge): The battery term (87) is an upper bound, because unmatched arms get no
battery; keeping it is the conservative choice. The amendment must be committed before any throughput of the pilot
is read.

**Why.** G2 multiplies by "484 run-equivalents (435 for Study A, 49 for Study B)", but the registered unit is
defined by training steps (Part 3.3 counts 140 continuations of 1,000,000 steps as "14 full runs"; Part 6.1 prices
cut 3 by steps), while 435 counts runs and leaves out the longer constrained-steps arms, the data control's T + N·T
and the battery's continuations. So 484 is an arithmetic inconsistency with the defining passages; deciding G2 on it
could pass a design that needs about 30 percent more compute, which would then be cut after data are seen. The judge
re-derived 627.13 independently; the corrected total is the conservative figure and changes no estimand.

**Rejected.** Keeping 484 decisive with the corrected total beside it (the earlier proposal); the corrected training
total without the battery (540.13, less conservative).

**Implemented in.** `pilot.go_decision.g2_throughput`; `pilot.budget.corrected_run_equivalents`,
`corrected_requirement` and `surplus_plan`.

### Q-exclusion-breaker (note)

**Answer.** This is an operational pause, not a decision point. After MAX_SAME_CAUSE_EXCLUSIONS = 2 exclusions of
one arm with the same cause, the scheduler stops adding replacements for that arm until the operator acts. The
resolution is fixed now:

- The operator reads the failed runs' logs and tracebacks, never their costs or returns.
- A defect in the code or environment is corrected by a commit recorded as an erratum (Part 5.8,
  analysis/ERRATA.json, Table 9.1). For a memory kill, the correction is to lower the number of concurrent runs.
- A correction may change no completed run's training. It must be bit-for-bit neutral for every configuration
  already run, which the operator checks with the determinism check on one completed configuration. If it is not
  neutral, every completed run it affects is rerun on the corrected code, so all arms are treated alike.
- With or without a defect, the operator then runs `schedule resolve RUN replace`, so every exclusion is repeated
  with the next unused seed as Part 5.6 requires.
- If no defect is found and an arm accumulates, for one cause, as many exclusions as its seed target (5, or 5 plus
  the surplus count), no further replacement is added. The arm is reported as 'did not complete its seeds (cause,
  count)' and enters no comparison, like an unmatched arm (Part 4.1).

Exclusion counts and causes are reported with each arm.

**Confidence.** High. Residual (judge): The stopping point (as many same-cause exclusions as the arm's seed target)
is a constant without a source, fixed before data and the same for every arm. Without it, repeating with the next
unused seed would not end when failure is near certain, and would give a survivor-selected seed set when failure is
merely likely.

**Why.** Part 5.6 requires every exclusion to be repeated with the next unused seed; the breaker only pauses
replacements, so that a deterministic bug does not use up the compute budget on identical failures. Fixing the
resolution now (a diagnosis blind to outcomes, an erratum under Part 5.8 if a defect is found, then the replacement)
leaves no decision to be taken after the data exist. The critic added a finite stop and the rule that a correction
must not change completed runs, or those runs are rerun, so that all arms are treated alike; an arm that cannot
complete is reported as Part 4.1 reports an unmatched arm.

**Rejected.** No breaker (a systematic bug uses compute unchecked); stopping the arm after a fixed number of
exclusions without a diagnosis (not in the text).

**Implemented in.** `pilot.scheduler.Scheduler.seed_target` and the `MAX_SAME_CAUSE_EXCLUSIONS` pause; `python -m
pilot schedule resolve RUN replace`, with `--defect-fixed ERRATUM_ID` once the arm's same-cause exclusions reach its
seed target.

### Q-p-list (note)

**Answer.** Table 8.1 gains a row 'Moderate arm's satisfaction rates at each of its training budgets' (reported,
used by no condition), and Part 3.6's list of measured quantities gains 'the Moderate arm's mean cost at each of its
training budgets' (used by G4). The go report gives both, as pilot.go_decision.report already does.

**Confidence.** High.

**Why.** Part 3.6 lists "the Moderate arm's satisfaction rates" among the [P] quantities of Part 8, but Table 8.1
has no such row and has the mean cost at each training budget, which G4 uses and Part 3.6 omits. Adding both changes
no decision, since G4 is decided on the mean costs.

**Rejected.** Dropping the satisfaction rates from Part 3.6 (loses a free descriptive quantity).

**Implemented in.** No code change: `pilot.go_decision.report` gives both.

### Q-g-names (note)

**Answer.** Record an erratum: the initial-registration row of Table 9.1 should read 'hypotheses G1 to G5' (Part 1.4
defines G1-G5). The go conditions keep their registered names G1-G4 (Part 6, cited in Parts 3.5, 3.6, 4.1, 5.5 and 8)
and are not renamed; every report and the paper qualify them as 'go condition G1 (Part 6)' versus 'Study B
hypothesis G1 (Part 1.4)'.

**Confidence.** High. Residual (judge): None that evidence could resolve: it is a choice between qualifying and
renaming, and qualifying keeps every registered identifier and cross-reference intact.

**Why.** Part 1.4 defines Study B's hypotheses G1 to G5, while the first row of Table 9.1 says "G1 to G4"; the
defining passage prevails, so the row is an erratum. Renaming the go conditions C1 to C4 (the earlier proposal)
would make the paper's labels differ from the registered text's many references to them; qualifying the names
removes the ambiguity and changes no registered identifier.

**Rejected.** Renaming the go conditions C1 to C4 (the earlier proposal).

**Implemented in.** `pilot.go_decision.GO_CONDITION_HEADING` ("Go condition Gk (Part 6)") in the go report; the
analysis already names them `pilot-G1` and `pilot-G3`.

### Q-ledger-v2 (note)

**Answer.** Keep ledger schema v1 frozen (one row per run, Table B.1, with the recorded divergences). Record the
supplement records instead of a v2:

- results/supplement_schema.py (SUPPLEMENT_SCHEMA_VERSION 1; kinds evaluation, measurement, battery, final_battery,
  sensitivity_battery, continuation, zero_shot, fewshot, training), written only by pilot.supplement;
- the pilot's JSON sidecar for the unconstrained Study B run.

Freeze the supplement schema at version 1 and record its SHA-256 in results/supplement_schema.sha256 (as for the
ledger schema), in the same commit as the Table 9.1 row and before the first pilot run. The registered constants it
imports from configs/registered.py are covered by the amendment rule for that file, not by this hash. Later changes
follow the ledger rule: an amendment, and a version bump if a stored record would become invalid.

**Confidence.** High. Residual (judge): Because Role 4 adopts `results/supplement_schema.py` in its own pull
request, "in the same commit as the Table 9.1 row" is read as "committed no later than that row, and before the
first pilot run".

**Why.** Appendix B fixes one row per run, but the registered analysis needs per-episode costs (the IQM of Part
5.3), the measurement-set return, the battery at the final checkpoint and at tolerance 5.0 (Part 4.1 rule 7) and
Study B's violation magnitude (Part 4.2). Records beside the frozen row change nothing registered, where a v2 would
re-freeze a reviewed schema before the pilot. The change: the supplement schema is frozen with its own SHA-256, the
same protection the ledger schema has; the critic noted that the registered constants it imports are governed by the
amendment rule for `configs/registered.py`, not by this hash.

**Rejected.** Ledger schema v2 with new columns (more change, and a new review of Role 4's frozen schema).

**Implemented in.** `results/supplement_schema.py` and `results/supplement_schema.sha256`, checked by
`tests/test_supplement_schema.py`; `pilot.supplement`, the only writer.

### Q-appendix-a (note)

**Answer.** Record in Table 9.1 that the Appendix A items were not in the registration commit (735b394 holds only
the prereg) and are added later, each with path, SHA-256 and source: configs/omnisafe/PPOLag.yaml, PPO.yaml and
CPPOPID.yaml copied byte for byte from OmniSafe 0.5.0's omnisafe/configs/on-policy (hashes in
configs/omnisafe/SOURCE.json), environment/requirements.in and requirements.lock.txt, environment/workstation.json,
the PPO-Lagrangian reward cell of Ji et al. (2024) copied by the group from the paper, and the fixed batches
metrics/fixed_batch/*.npy with MANIFEST.json (seed 0, 2,048 states; SHA-256 5340625c... SafetyPointGoal1-v0,
abc01be6... SafetyCarGoal1-v0, 84ae7249... SafetyPointButton1-v0). The committed batches are the fixed batches
(Table 2.3: collected once, never resampled): `python -m metrics.collect_fixed_batch --all --check` on the
workstation is recorded as a reproducibility check, and a mismatch is reported in Table 9.1 but does not replace
them. All are committed before the first pilot run.

**Confidence.** High. Residual (judge): Two facts are outstanding, and neither can change the decision: the group
copies the PPO-Lagrangian reward cell from Ji et al. (2024), which could not be read from the sandbox, and the
result of `--check` on the workstation is unknown (a mismatch is only reported).

**Why.** Appendix A says these settings were "copied verbatim from the repository at the registration commit", but
that commit holds only the pre-registration, so recording their later addition with hashes is the only truthful
option; the YAML files are byte for byte the pinned OmniSafe 0.5.0 files and change no setting. The change concerns
the fixed batches: Table 2.3 requires a batch "collected once per task", so the committed batches are the fixed
batches, and collecting them again after a mismatch on the workstation would be the resampling the table forbids.
The earlier proposal sent a mismatch to the group, which left a choice open.

**Rejected.** Collecting the batches again on the workstation if `--check` fails (the batch would depend on the
machine, and it needs an amendment).

**Implemented in.** `python -m metrics.collect_fixed_batch --all --check`; `metrics.batches` (the SHA-256
constants); `python scripts/workstation.py fixed-batch`, which writes `environment/fixed_batch_check.json` with the
check's outcome (passed, or the tasks whose re-collection differs, reported in Table 9.1 without replacing the
batches); `configs/omnisafe/SOURCE.json`.

### X-ledger-schema-amendment (extra item)

**Answer.** Approve the ledger-schema amendment LS-1 to LS-32 of docs/ledger_schema_memo.md ('Changes since the
freeze') as one Table 9.1 row: the Parquet layout (LEDGER_SCHEMA), field names, public functions and SCHEMA_VERSION
1 are unchanged; the recorded hash is re-recorded as sha256(results/ledger_schema.py) =
9d53851acd49e64d7aded82c0e5b7647147f27b804e117570423b5cce320be99 in results/ledger_schema.sha256 (UTF-8, one LF
line, verifiable with `sha256sum -c`); the divergences from Table B.1 listed in the memo are recorded with it. Role
4 (Muhammad Abdullah) approves it by pull request before the first pilot ledger row is written.

**Confidence.** High.

**Why.** The recorded hash of the frozen schema did not verify (a UTF-16 file holding the hash of a CRLF copy), and
the validation gaps listed in HANDOVER §3.3 would let invalid values into the ledger that the confirmatory analysis
reads; LS-1 to LS-32 close them. The decider and the critic loaded the frozen and the corrected schema in the pinned
environment: the Parquet layout, the field set, `SCHEMA_VERSION` and every public signature are unchanged, the only
change to `ENRICHMENT_FIELDS` is the removal of `notes` (which enforces the memo's "raw fields are immutable"),
`sha256sum -c` passes and the schema's tests pass. No ledger row exists yet, so no version bump is needed.

**Rejected.** Keeping the frozen file and correcting only the recorded hash (leaves the validation gaps).

**Implemented in.** `results/ledger_schema.py`, `results/ledger_schema.sha256`,
`results/ledger_schema_requirements.txt`, `tests/test_ledger_schema.py`; `docs/ledger_schema_memo.md`, section
"Changes since the freeze".

### X-allocation (extra item)

**Answer.** The allocation is the group's resource decision, and no analysis can supply it. The group must supply
one number, H: the wall-clock hours the workstation is dedicated to the registered runs of the whole design.

- Unit: elapsed hours of the machine with all its concurrent run slots together, not core-hours and not hours per
  run.
- Scope: Study A, Study B, their continuations, evaluations and replacements.
- Derivation: only from the calendar and the machine's availability (e.g. 24 × days from the planned start of the
  main sweep to the last date runs may finish × the fraction of time the workstation is available). Never from any
  timing of a run on the workstation, smoke runs and the determinism check included.

H is recorded once with `python -m pilot allocate --hours H --basis TEXT --by "the group" --date YYYY-MM-DD`,
committed, and entered in Table 8.1 with a Table 9.1 row before any pilot run. It should be recorded before the
determinism check and smoke runs on the workstation, which already reveal its speed.

**Confidence.** High that analysis cannot decide it. Coordinator: the number is a fact about the group's resources;
the code makes it one recorded input.

**Why.** Part 6 G2 compares hours per run ÷ concurrent runs × run-equivalents with "the machine-hours the group
allocates to the registered runs", written into Part 8 "before the pilot's throughput is read", and Table 8.1 calls
it a group decision recorded before the throughput measurement is read. The number is a fact about the group's
resources and deadline, so no analysis can supply it. What can be fixed is fixed: the unit (the left side of G2 is
elapsed workstation time, so core-hours would pass G2 spuriously), the scope (the whole design, as
Q-g2-run-equivalents counts it), and the basis (only the calendar and the machine's availability, recorded with
`--basis`, before any timing on the workstation exists; the critic's edit).

**Rejected.** Nothing: the value is not a matter of interpretation.

**Implemented in.** `pilot.budget.record_allocation` (`ALLOCATION_UNIT`, `ALLOCATION_CONFIRM_ABOVE`,
`prior_workstation_timings`, `registered_runs_started`); `python -m pilot allocate --hours H --basis TEXT --by "the
group" --date YYYY-MM-DD [--confirm] [--smoke-root DIR ...]`, refused once a registered run has started and warning
about earlier workstation timings.

### X-registration (extra item)

**Answer.** Everything in Table 9.0 except the search-record path is decidable now: Repository
github.com/paf-iast-ai-research/when-safety-comes-late (public); Registration commit hash
735b394d18d1bb046c7a74f900af00a83f1fa316; Commit timestamp 2026-09-18T23:22:45+05:00 (2026-09-18T18:22:45Z); Tag
registration (annotated, on 735b394). Table 9.1's first row takes the same date and hash, approved by the Group. The
Appendix C record path is studyb/search/ and is entered when the search record is committed. The tables are filled
in a later commit that changes only Tables 9.0 and 9.1 of the docx and its regenerated PDF, and the registration is
completed (tag pushed, tables committed) before any pilot run, since until then the document is 'the adopted draft
and not the registration record'.

**Confidence.** High. Residual (judge): The open items are acts: pushing the tag, which is not yet on the remote,
and committing the filled tables. Repository visibility was not re-checked here (HANDOVER §2 records it as public).
Git timestamps are asserted by the committer, so an independent time stamp (Zenodo or OSF, task 2) is recommended
but not needed.

**Why.** Table 9.0 and the registration status table make the record complete when Part 9 holds the hash, the
timestamp and the public address; until then the document is "the adopted draft and not the registration record".
The values were checked: 735b394 is the only root commit, holds exactly the two pre-registration files and has
author and committer dates of 2026-09-18T23:22:45+05:00, and a local annotated tag `registration` points at it. A
commit cannot contain its own hash, so the tables are filled in a later commit that changes only Tables 9.0 and 9.1
and the regenerated PDF; the change is that this is done, and the tag pushed, before any pilot run.

**Rejected.** Filling Table 9.0 without restricting that commit to the two tables (risks unnoticed edits of
registered text); starting the pilot before the record is complete. The optional tools first proposed (a checker of
`.docx` edits, a recorder of GitHub push events) were not adopted: they are not needed for the decision
(coordinator).

**Implemented in.** No code change: `scripts/tag_registration.sh`; `python3.10 scripts/workstation.py registration`.
The order (registration complete before any pilot run) is a rule for people: the scheduler does not check the tag or
the tables.

## 5. Study A analysis (Part 5) and hypothesis readings

### Q-interval (starred)

**Answer.** Where a hypothesis box says an interval excludes zero, it means the primary interval of Part 5.3: the
two-sample Welch t-interval (95 percent; 97.5 percent for G1 and G2 per Part 5.2). The percentile bootstrap interval
over seed indices (10,000 resamples) is reported beside every estimate and decides nothing. A box condition decided
by a correlation (Part 5.4: the H1 and G1 trends, H3 (a)) uses the registered seed-resampled percentile interval,
the only interval Part 5.4 registers for it, at the box's level (95 percent; 97.5 percent for G1's trend). The Part
5.7 bound is the upper (or lower) limit of the 95 percent Welch interval.

**Confidence.** High.

**Why.** Part 5.3 says an interval that must exclude zero is "the primary interval" and gives the reason (with five
seeds a percentile bootstrap has too few distinct resamples to hold its coverage); its last sentence calls it "the
95 percent percentile interval", which also contradicts Part 5.2 (97.5 percent for G1 and G2) and Part 5.7 ("the 95
percent interval of the primary estimand"). The reasoned, defining sentence prevails, and at n = 5 the Welch
interval is also the more conservative one. The critic added the level of the correlation intervals of Part 5.4 (95
percent; 97.5 percent for G1's trend).

**Rejected.** The percentile interval deciding (the literal last sentence); both intervals required to exclude zero
(a new rule).

**Implemented in.** No code change: `analysis.verdict.Reading.interval = "welch"`;
`analysis.stats.TwoArmEstimate.interval`; `analysis.stats.spearman_bootstrap`.

### Q-h1-shape (starred)

**Answer.** Box H1 (confirmatory, on the primary outcome) reads Δ(N) of equation (2) on the abrupt-onset late arms:
Δ(N) = mean gap(abrupt, N, control) − mean gap(N = 0), under each step-matching control. The same box applied to the
ramp arms is a secondary reading, reported per (task, condition) cell as H1[task/condition/ramp] and Holm-corrected
in the ramp-shape families (Q-holm-families). The shapes are never pooled.

**Confidence.** High.

**Why.** Table 3.2 makes onset fraction "the manipulation of H1" and onset shape "the manipulation of H2", so H1
holds the shape at its default, and Table 2.1's default onset is abrupt; every other registered "late arm" is abrupt
(the pilot behind G3, the H3 treatments and the H4 variants of Table 3.3). Reading H1 on the abrupt arms keeps the
H1 and G3 estimands the same, while pooling the shapes would mix H2's manipulation into H1.

**Rejected.** Pooling abrupt and ramp arms; requiring support under both shapes (a condition the box does not
state); ramp as primary.

**Implemented in.** No code change: `analysis.verdict.Reading.shape = "abrupt"`; `analysis.study_a.h1_outcome`, with
the ramp cells in their own Holm families.

### Q-controls-reading (starred)

**Answer.** Boxes H1 and H2 are each read under both step-matching controls and combined as follows. SUPPORTED only
if SUPPORTED under both controls. FALSIFIED only if FALSIFIED under both (after Q-falsification-calibration).
INCONCLUSIVE if the two controls disagree or either is INCONCLUSIVE. NOT_COMPUTABLE only while a control that cannot
be evaluated (data still to come, or its N = 0.50 arm unmatched under rule 6) could still change the combined
status, with the reason given (analysis.verdict.combine_controls).

**Confidence.** High.

**Why.** H1's support names both controls ("under both step-matching controls"), and Part 5.2 makes the primary
analysis H1 under both; its falsification and box H2 name no control. Letting one control falsify a hypothesis whose
support needs both would be asymmetric and would make H0 easier to support on half the evidence; under Part 1.2's
calibration a falsification under one control bounds the effect only there. The critic noted that if rule 6 leaves
one control's N = 0.50 arm unmatched, that control stays not computable and H1 can never be supported, which is
consistent with "supported under both".

**Rejected.** Falsified when falsified under either control; H2 read under one control only.

**Implemented in.** No code change: `analysis.verdict.combine_controls`; `analysis.study_a._combined`.

### Q-support-falsify-overlap (starred)

**Answer.** 'Δ increases with N' (H1), and G1's ordering, are supported only when both of the following hold: Part
5.4's Spearman trend (ρ > 0 with its seed-resampled interval excluding zero) and the ordering of the point estimates
(H1: Δ(0.10) ≤ Δ(0.25) ≤ Δ(0.50); G1: Sparse < Moderate < Dense). If a box's support and falsification conditions
both hold, the box is FALSIFIED with a note, and Q-falsification-calibration's bound then applies to it as to any
falsified box.

**Confidence.** High.

**Why.** H1's support says "Δ increases with N", Part 5.4 tests that by Spearman's ρ, and H1's falsification
includes "point estimates are not ordered with N"; requiring the ordering in the support makes the two conditions
consistent. Under the decided readings no box can meet both conditions, so the overlap rule is dormant; choosing
FALSIFIED, which is then calibrated, never rounds an outcome up to support (Part 1.2).

**Rejected.** SUPPORTED on overlap (rounds up); INCONCLUSIVE on overlap (ignores an observation the box names as
falsifying; it differs only when the effect is bounded).

**Implemented in.** No code change: `analysis.verdict.box_status`; `analysis.study_a._h1_control`;
`analysis.study_b.g1_outcome`.

### Q-falsification-calibration (starred)

**Answer.** Falsification follows Part 1.2's calibration and Part 5.7. A box whose falsifying quantity carries a
Part 1.5 minimum effect (H1, H2, H3 (b), H4; G1, G2, G3) is FALSIFIED only if two things hold: its box conditions
hold, and its Part 5.7 bound lies below the minimum effect (5 cost units in Study A; 5 percentage points in Study
B). The bound is the 95 percent Welch limit, in the direction the box claims, of each estimand its falsifying clause
names: H1, the upper limit of Δ(0.50), per control; G1, the upper limit of Dense − Sparse at 95 percent; the others
as analysis.verdict.bound_annotation computes them. Otherwise the box is INCONCLUSIVE ('inconclusive at this sample
size') and its bound is reported. Where the bound cannot be read (no interval, or a floored Study B budget compared on
violation magnitude), the falsification is not confirmed and the box is INCONCLUSIVE. Boxes whose falsifying quantity
has no minimum effect (H3 (a), H3 (c), G4), and G5, whose falsification is itself an interval limit against 5 pp, are
decided by the box. Whether the box conditions alone were met is reported in the verdict's numbers ('falsification')
and in a note.

**Confidence.** High. Residual (judge): Small and interpretive: whether Part 1.2's "every claim" also binds the
boxes other than H1 and G1, which Part 5.7 names. Calibrating all of them is the symmetric and conservative reading.

**Why.** Part 1.2 defines "falsified" for every claim, Study B's included, as an effect "bounded below the minimum
of interest", and Part 5.7 says a falsified H1 or G1 whose bound is above the minimum effect "is reported as
inconclusive at this sample size", the exact wording of the INCONCLUSIVE status. The earlier proposal let the boxes
decide and only annotated the bound, so it would report FALSIFIED where Part 5.7 says inconclusive; a box condition
such as "all intervals include zero" does not bound the effect. Applying the calibration to every box with a minimum
effect treats the boxes alike and is conservative; boxes without one, and G5, whose falsification is already an
interval limit, are decided by the box.

**Rejected.** The earlier proposal (the boxes decide, the bound annotated); calibrating H1 and G1 only (lets H2 to
H4, G2 and G3 be falsified by absence of evidence).

**Implemented in.** `analysis.verdict.calibrate` and `bound_annotation` (`Reading.calibration = "bound"`); the bound
of each box in `analysis.study_a` and `analysis.study_b`.

### Q-holm-families (starred)

**Answer.** Every secondary Study A claim on a (task, condition) cell must survive Holm (α = 0.05) in both families
it belongs to: its task's four-condition family and its condition's three-task family. This covers H1 (each shape's
own families) on every cell except the primary abrupt cell, and H2, H3 (a), H3 (b) and H4 on every cell including
SafetyPointGoal1-v0 × hazard. The primary H1 claim is uncorrected, but its p-value counts as a member of both abrupt
H1 families. Families shrink only to cells not tested (no arm by design, a recorded Part 6.1 cut, an infeasible
task, an arm left out by rule 6, a condition not yet evaluable while its key is open). An incomplete member is held
at p = 1, and any survival it could change stays undecided. Claims with a single family (recovery time and return
over the three tasks, H3 (c) over the three tasks, Study B's four unseen budgets) use that family.

**Confidence.** High. Residual (judge): Part 5.2 says "its family" in the singular and does not say what happens to
a cell that lies in a four-condition family and a three-task family; the conservative option is fixed before data
and applied to every cell alike.

**Why.** Part 5.2 names both families ("the four battery conditions form one family; the three tasks another"), and
every cell of the task × condition grid belongs to one of each. The earlier proposal decided a cell that differs
from the primary in one dimension by one family only, leaving the other family's error rate uncontrolled; requiring
survival in every family a claim belongs to is the only rule that corrects every named family, and it removes a
special case.

**Rejected.** The earlier proposal (one family for cells that differ in one dimension); survival in either family;
one family of twelve cells (not in the text).

**Implemented in.** `analysis.study_a.StudyA.holm_survival` (`Reading.holm = "both_families"`);
`analysis.stats.holm` and `holm_decisions`.

### Q-threshold-arithmetic (starred)

**Answer.** Part 4.1 rules 3 and 5 use the sample standard deviation (n − 1) over the reference arm's seeds. The
task is infeasible if SD > 10 (SD = 10 is feasible). Rule 3 (reference cost ≤ d + tolerance) and rule 5 (|matched
cost − reference cost| ≤ tolerance) are inclusive. The means and their difference are compared after rounding to
1e-4. The SD, which is not on the 0.01 grid of mean costs, is compared after rounding to 1e-9. Part 6 G1 and G3
compare in the same way: G1's 'at most d + 2.5', 'at most 10' and rule 5 directly; G3's 'at least 5' on the mean
Δ(0.50) after rounding to 1e-4, U80 and U80_pooled after rounding to 1e-9.

**Confidence.** High.

**Why.** The text fixes the directions: rule 3 "≤ d + tolerance", rule 5 "≤ 2.5" and "exceeds 10", G1 "at most", G3
"at least". The seed-to-seed SD is conventionally the sample SD, which is also the larger value and so the
conservative choice for a feasibility gate. Costs per step are binary in the pinned stack, so means lie on a fine
grid and rounding to 1e-4 cannot move a real boundary but stops floating-point noise from deciding a tie; the SD
gets only noise-level rounding. Since Q-g3-pairing, U80_pooled is descriptive and decides nothing.

**Rejected.** The population SD with strict comparisons (contradicts the printed ≤ and "exceeds"); one rounding for
everything (not needed).

**Implemented in.** No code change: `analysis.matching.Arithmetic`, `budget_feasible`, `within_tolerance` and
`spread_feasible`; `pilot.go_decision` (`THRESHOLD_DECIMALS`, `SD_DECIMALS`).

### Q-iqm (starred)

**Answer.** The interquartile mean of Part 5.3 is descriptive and never decides a verdict. For each arm and outcome
it is the 25-percent trimmed mean (scipy trim_mean 0.25) of the outcome's per-episode values pooled over the arm's
seeds. The IQM of a gap is IQM(C_cond episodes) − IQM(C_ID episodes), and the IQM of a two-arm estimand is the
difference of the arms' IQMs. Its 95 percent percentile interval comes from a two-level bootstrap with 10,000
resamples: seeds are drawn with replacement (arms with the same seed list share the draw), then episodes are drawn
with replacement within each drawn seed. Where per-episode records are missing it is reported as not computable.

**Confidence.** High.

**Why.** Part 5.3 reports the IQM "with a stratified bootstrap over seeds and evaluation episodes ... alongside the
mean", so it is descriptive, and Part 5.1 forbids resampling episodes as if they were independent. The nested
bootstrap (seeds, then episodes within a drawn seed) keeps the seed as the top-level replicate and includes the
variance between seeds. Battery and measurement episodes are different episodes, so a gap's IQM is a difference of
IQMs.

**Rejected.** An IQM over seed-level outcomes (ignores the episodes); a one-level bootstrap of episodes within fixed
seeds (understates the uncertainty); the IQM interval deciding a verdict (forbidden by Part 5.1).

**Implemented in.** No code change: `analysis.stats.stratified_iqm` and `iqm_contrast`.

### Q-bootstrap-details (starred)

**Answer.** Each analysis gets its own generator, numpy.random.default_rng([0, crc32(analysis id)]), recorded with
the result. B = 10,000 resamples. Percentile limits use numpy's 'linear' quantile rule. When the two arms of a
contrast have identical seed lists, one index draw serves both (pairs kept together). Otherwise, including partly
overlapping lists, each arm is resampled on its own indices. A Spearman trend resamples seed ids from the union,
each carrying all its points. Undefined resamples (a constant variable) are dropped and counted, and flagged above 1
percent. Holm (Part 5.2) uses the two-sided Welch p of a two-arm claim. For a Spearman claim (H3 (a)) it uses the
bootstrap p = min(1, 2·min(#ρ* ≤ 0, #ρ* ≥ 0) / B_defined).

**Confidence.** High.

**Why.** Part 5.3 fixes only "a percentile bootstrap ... from 10,000 resamples of seed indices with replacement"
that takes every outcome of the chosen seeds together; one draw for two identical seed lists is that sentence, and
Part 5.4 requires the trend resample to carry all of a seed's levels together. A generator per analysis makes every
interval reproducible and independent of the order in which analyses run. The bootstrap p of a Spearman claim
inverts the registered percentile interval (up to one resample at the boundary, as the critic noted) and respects
the clustering within seeds; the Welch p of a two-arm claim matches Q-interval.

**Rejected.** Resampling the union of seeds jointly for partly overlapping lists; an asymptotic Spearman p (ignores
the clustering); one random stream shared by all analyses.

**Implemented in.** No code change: `analysis.stats.generator`, `percentile_interval`, `bootstrap_difference` and
`spearman_bootstrap`.

### Q-surplus-in-analysis (starred)

**Answer.** Every completed seed of an arm (registered seeds, Part 5.6 replacements and Part 5.5 surplus seeds) is
used in: matching (rules 2 to 5; the reference and matched costs are means over all completed seeds); every two-arm
estimate, with its Cohen's d, Welch and bootstrap intervals and Holm p; the Part 5.7 bound; and Study B's trend
(Part 5.5 gives every Study B arm the same surplus count). Paired analyses use the seeds common to both arms. The
power statement and the detectable size use the actual counts (the smaller arm's n). The two exceptions are Study
A's seed-index-resampled correlations of Part 5.4, whose resampling unit is a seed index with all of its levels: the
H1 trend (the registered twenty points, four onset fractions by five seeds) and H3 (a) ('with the same interval',
across the late arms). Both use each arm's registered seeds and their replacements only
(analysis.data.five_seed_view), because surplus seeds exist only at N = 0 and N = 0.50 and would leave seed indices
without all of their levels.

**Confidence.** High. Residual (judge and coordinator): The judge left H3 (a) open at medium confidence; the
coordinator resolved it: H3 (a) uses the same resampling of seed indices as the trend ("with the same interval"),
which presupposes a complete grid of seeds by levels, so it also takes the five-seed view.

**Why.** Part 5.5 spends surplus seeds on "the arms that carry the primary comparison" and states the power from the
count actually used, and Part 6.1 separates that comparison (N = 0.50 against N = 0) from "the trend test of H1"; so
every completed seed counts in matching and in the two-arm estimates. The change: Part 5.4 defines the trend as
"twenty points: four onset fractions by five seeds", and surplus seeds exist only at N = 0 and 0.50, so using them
would weight the rank correlation toward the end levels, change the estimand and make support easier. The trend and
H3 (a) therefore use each arm's registered seeds and their replacements, a restriction fixed before data from the
scheduler's record of each seed's role; Study B gives every arm the same surplus count, so its trend keeps every
seed.

**Rejected.** Every seed in the trend too (the earlier proposal: an unbalanced trend leaning toward the primary
contrast); five seeds everywhere (wastes the surplus Part 5.5 spends).

**Implemented in.** `analysis.study_a._h1_control` and the H3 (a) points through `analysis.data.five_seed_view`
(from `pilot.manifest.seed_role`); `analysis.matching.apply_rules` uses every completed seed.

### Q-h3-scope (starred)

**Answer.** H3 (a): the points are every completed run of the task's late main-sweep arms (N = 0.10, 0.25, 0.50;
abrupt and ramp; both step-matching controls) that are matched under Part 4.1. An arm left out by rule 6 is excluded
and named. Each point is the metric at onset against the matched checkpoint's gap. Treatment, controller-variant and
PID arms are not points. H3 (b) and (c) compare only the N = 0.50 abrupt, total-steps-matched arms (Table 3.3):
treated minus untreated. The absences in (b), 'additional constrained training does not [reduce the gap]',
'injection does not' and 'none of the three reduces it', mean no reduction with an uncorrected 95 percent interval
excluding zero. A claim of reduction needs the interval excluding zero and Holm (Q-holm-families).

**Confidence.** High.

**Why.** Box H3 (a) reads "across late arms and seeds" and Part 5.4 "across all late arms of a task", which are the
late arms of the main sweep (Table 3.2); reset and injection are designed to break the link from the onset metric to
the gap (H3 (b)) and the controller variants change the dynamics after onset that H4 studies, so including them
would build those interventions into H3 (a)'s association. Rule 6 leaves an unmatched arm without a gap, so only
matched arms give points. Part 5.1 defines the treatment estimand at N = 0.50, which boxes (b) and (c) name; the
absences in (b) are an ellipsis of "reduce the gap ... with an interval excluding zero", read on the uncorrected
interval so that Holm cannot make an absence easier.

**Rejected.** Treatment and controller arms as points of (a); the absences read on the point estimate; (b) and (c)
pooled over N.

**Implemented in.** No code change: `analysis.study_a.h3a_outcome`, `_late_arm_records`, `_h3b_estimates`,
`h3b_outcome` and `h3c_outcome`.

### Q-h4-reading (starred)

**Answer.** In H4, 'the gap does not fall (overshoot explains none of it)' is read on the point estimate: F =
gap(warm-started, N = 0.50) − gap(untreated N = 0.50) ≥ 0, compared after settled rounding. A fall whose 95 percent
interval includes zero is INCONCLUSIVE, neither support nor falsification. 'The difference from the N = 0 arm no
longer excludes zero' is read on the uncorrected 95 percent Welch interval of G = gap(warm-started) − gap(N = 0).
The two claims of support, 'falls' and 'remains above', each need the interval excluding zero and Holm. A
falsification additionally needs its Part 5.7 bound (Q-falsification-calibration: F's lower limit above −5, or G's
upper limit below 5).

**Confidence.** High.

**Why.** Box H4's support needs the gap to fall "with an interval excluding zero", and its falsification, "the gap
does not fall (overshoot explains none of it)", describes no fall at all, a reading on the point estimate. Reading
it on the interval would make every fall that is not significant a falsification and leave no inconclusive outcome,
counting absence of evidence as evidence of absence, against Part 1.2; with Q-falsification-calibration a
falsification on the point estimate also needs its bound.

**Rejected.** The interval reading.

**Implemented in.** No code change: `analysis.study_a.h4_outcome`.

### Q-h0-scope (starred)

**Answer.** Box H0 reads boxes H1 and H2 on the primary outcome: the robustness gap under hazard relocation on
SafetyPointGoal1-v0 (H1 on the abrupt arms, per Q-h1-shape). Each box is combined over the two controls by
Q-controls-reading and calibrated by Q-falsification-calibration. H0 is SUPPORTED iff both are FALSIFIED, FALSIFIED
iff either is SUPPORTED, and otherwise INCONCLUSIVE (NOT_COMPUTABLE only while missing data could still change it).
When H0 is supported, H3 and H4 are reported as NOT_TESTED on every cell, with their estimates still given.

**Confidence.** High.

**Why.** Box H0 is supported when "H1 and H2 are both falsified as above", and Part 5.2 fixes the outcome on which
the main analysis is read: the gap under hazard relocation on SafetyPointGoal1-v0. H0 is one box, so it is read
once, on that outcome; reading it per cell would register twelve null hypotheses the text does not have. When H0
holds, H3 and H4 are "not evaluated" wherever they are read, the conservative gate.

**Rejected.** H0 per cell; H0 requiring H1 and H2 to be falsified in every cell (a much stronger null that would
never gate in practice).

**Implemented in.** No code change: `analysis.study_a.h0_from`, and the H0 gate of every H3 and H4 verdict in
`analysis.study_a.analyse`.

### Q-detectable-size (note)

**Answer.** Part 5.5's 'differences below the detectable size are reported with their intervals and are not claimed
as findings' is read as follows. An effect the study cannot resolve (interval including zero) is reported with its
interval as INCONCLUSIVE, never as a finding or a null. The detectable size (the 80-percent power point of the
two-sample t-test at the actual n, α = 1 − level) is an annotation beside every estimate and never a decision
criterion: box verdicts are unchanged. On a SUPPORTED verdict whose deciding |d| is below that point, the note
reads: 'Part 5.5: |d| = x is below the 80-percent detectable size y at n = k; the interval excludes zero and the
verdict stands; the test had under 80 percent power for an effect of this size, so its magnitude is imprecise (see
its interval; annotation only)'.

**Confidence.** Medium. Residual (judge and coordinator): read literally, Part 5.5's "Differences below the
detectable size ... are not claimed as findings" supports gating a SUPPORTED verdict on its observed |d|, and a
hostile reviewer can cite it. There is no drafting history to settle it, so only the group's statement at
ratification of what the sentence was meant to do could change this answer.

**Why.** Part 5.5 states the power of the boxes' t-test (80 percent at d = 2.02 with five seeds). Using the observed
|d| as a second gate would add an unregistered test and make those power numbers false: at a true d of 2.02 the
observed d falls below 2.02 about half the time. The sentence's continuation ("can establish a large effect ...
cannot resolve a medium one") and its "reported with their intervals" read as the INCONCLUSIVE case of Part 1.2. The
change is in the note: the earlier wording said a SUPPORTED difference "is not claimed as a finding", which
contradicts its own label, and the critic removed a claim that such an estimate is "likely to overstate" the effect,
which the data cannot back.

**Rejected.** Downgrading a SUPPORTED verdict whose |d| is below the detectable size (adds a second test); the
earlier wording of the note.

**Implemented in.** `analysis.verdict.detectable_notes`; `analysis.stats.detectable_d` and
`TwoArmEstimate.below_detectable`.

### Q-exploratory-labels (note)

**Answer.** The analysis learns about amendments from analysis/AMENDMENTS.json: one entry per Table 9.1 amendment
that answers a PENDING key or changes an analysis setting, with keys, data_seen and affects. Every verdict that an
entry with data_seen true affects (named in affects, or depending on one of its keys through analysis.questions) is
relabelled exploratory. The go amendment records analysis_code_hash (records.code_hash()). That hash covers
analysis/*.py and every in-repository module the analysis imports, directly or transitively, including those
imported inside functions: at present configs/\_\_init\_\_.py, configs/registered.py, pilot/\_\_init\_\_.py,
pilot/manifest.py, pilot/contracts.py, pilot/provenance.py, pilot/go_decision.py, pilot/budget.py, pilot/rundir.py,
pilot/errors.py, results/ledger_schema.py and results/supplement_schema.py. A test fails if an in-repository module
the analysis loads is missing from the hash. Error fixes after the go decision are recorded in analysis/ERRATA.json,
one per Table 9.1 row with code_hash_before/after. The report flags any hash that is neither the go hash nor one
reached from it by the chain of recorded errata. The answers of this amendment are entered with data_seen false,
since they were made before any data existed.

**Confidence.** High. Residual (judge): The judge revised the answer (REVISE, high): the analysis also loads
`pilot/budget.py`, `pilot/rundir.py` and `pilot/errors.py`, which the hash did not cover. No residual on the
principle; the list should be derived from the import closure and checked by a test, so that it cannot go stale.

**Why.** Part 8.2 and the registration status table label as exploratory every result affected by a change made
after data were seen, and Part 5.8 freezes the analysis script after the go decision except for recorded error
fixes. A machine-readable registry that mirrors Table 9.1 lets the frozen analysis apply these rules without editing
its code. The change concerns the freeze check: hashing `analysis/*.py` alone missed the registered constants and
pilot modules every verdict reads, so an edit after go to, say, the minimum effect would not have been flagged; the
hash now covers every in-repository module the analysis imports.

**Rejected.** Labels edited by hand in the report (not auditable); labels kept in `configs/` (mixes registered
settings with bookkeeping).

**Implemented in.** `analysis.records.load_amendments`, `apply_amendments`, `load_errata`, `code_files`, `code_hash`
(`IMPORTED_CODE`) and `freeze_state`; `analysis/AMENDMENTS.json` and `analysis/ERRATA.json`.

### Q-unmatched-reporting (note)

**Answer.** Rule 6's 'a late arm that never reaches the budget' is an arm that rule 5 leaves unmatched and whose
matched cost (rule 4: the mean over its completed seeds of the selected checkpoints' measurement-set cost) is above
d + tolerance (27.5; 30.0 in the rule 7 (a) repeat). This is the test rule 3 uses for 'satisfy the budget', compared
with the same arithmetic. Such an arm is reported as a finding about learning, not about robustness. Any other
unmatched arm (above the reference by more than the tolerance but within d + tolerance, or below the reference) is
reported with its cost and difference as unmatched, without the learning label. Table A_unmatched also gives each
arm's lowest window selection cost.

**Confidence.** High. Residual (judge): "Never" could also be read over the whole of training (on the training-batch
cost); the answer reads it on rule 4's evaluation cost, the quantity rule 6 works on, and reports each arm's lowest
window selection cost beside it.

**Why.** Rule 6 calls "a late arm that never reaches the budget" a finding about learning. The registration's own
test of satisfying the budget is rule 3's "reference cost ≤ d + tolerance", and G1 uses the same verb ("reach ... of
at most d + 2.5"). The earlier proposal (unmatched and above the reference) would give the learning label to an arm
at 26 against a reference at 22, which does meet the budget; the answer applies one test to every arm and changes
only a label, never which arms are compared.

**Rejected.** The earlier proposal (unmatched above the reference); no window checkpoint with a selection cost at or
below d (uses the selection set and ignores the tolerance; reported beside).

**Implemented in.** `analysis.study_a`, table `A_unmatched` (`reaches_budget` and `learning_finding` through
`analysis.matching.budget_feasible`).

### Q-training-age-systematic (note)

**Answer.** Selected training ages 'differ systematically between arms' when the 95 percent Welch interval of the
difference in mean training age (the matched checkpoint's step, over every completed seed) between two arms excludes
zero (unrounded limits). The difference is evaluated for every pair of arms that a Study A verdict compares on
matched checkpoints under the decided readings: each late arm against N = 0 (H1, and the rule 7 (a) repeat), abrupt
against ramp (H2), each treatment against the untreated N = 0.50 arm (H3 (b)), and each controller variant against the
untreated arm and against N = 0 (H4). Each verdict whose deciding contrast has systematically different ages carries a
note naming that difference as a limit on the causal interpretation. Table A_training_age keeps each arm against its
task's reference and adds these contrast rows. The final-checkpoint estimand (training age = total steps by
construction) and H3 (c) (measured at a fixed logging point, not at a matched checkpoint) carry no such note.

**Confidence.** High. Residual (judge): The registration gives no threshold for "systematically"; the Welch 95
percent interval is the registered primary interval, fixed before data and applied to every compared pair alike.

**Why.** Part 4.1.1 says that where selected training ages "differ systematically between arms", the difference is
named as a limit on the causal interpretation, and "between arms" refers to the arms being compared. The earlier
proposal compared each arm only with its task's reference, which cannot show whether abrupt and ramp (H2), treated
and untreated (H3 (b)) or the controller variants (H4) differ from each other. The critic left out the
final-checkpoint estimand (not a selected checkpoint) and H3 (c) (read at a fixed logging point); the
constrained-steps arms will be flagged by design, a genuine difference to name.

**Rejected.** Each arm against the reference only (the earlier proposal); any non-zero mean difference (flags
noise); a practical threshold such as one checkpoint interval (not registered).

**Implemented in.** `analysis.study_a.StudyA.age_note`; the contrast rows of table `A_training_age`.

### Q-treatment-other-N (note)

**Answer.** The treatment arms (reset, injection, additional constrained training) and the controller variants
(warm-started, rate-limited, PID) also run at N = 0.10 and 0.25 (Table 3.3). Their contrasts with the untreated
abrupt total-steps arm at those fractions are reported as descriptive estimates (table A_treatments_other_N:
difference, Welch and percentile intervals, d; label 'descriptive') with no verdict and no Holm. Boxes H3 (b), H3
(c) and H4, and the recovery-time claims for treated arms, are read at N = 0.50 only. These arms are not points of
H3 (a) (Q-h3-scope).

**Confidence.** High.

**Why.** Part 5.1 defines the treatment estimands only "at N = 0.50 for H3 and H4", and boxes H3 (b) and H4 name the
N = 0.50 arm; Table 3.3 runs the arms at all three fractions, and Part 6.1's fourth cut treats the other fractions
as dispensable. With no registered estimand there, any verdict would be unregistered; descriptive reporting avoids
forking paths without hiding data.

**Rejected.** Secondary verdicts at every N with Holm; pooling the fractions.

**Implemented in.** No code change: `analysis.study_a._other_n_rows` (table `A_treatments_other_N`).

## 6. Study B hypothesis readings and the floor rule

### Q-g1-criteria (starred)

**Answer.** G1's 'dense minus sparse exceeds 5 percentage points with an interval excluding zero' is read as two
conditions on Dense minus Sparse in the per-seed mean zero-shot satisfaction over the unfloored unseen budgets.
First, the point estimate exceeds 0.05. Second, the 97.5 percent Welch interval excludes zero (lower limit > 0); the
percentile interval is reported beside it. In the falsification clause, 'overlapping intervals' are the per-arm
one-sample 97.5 percent t-intervals of the arm means of Sparse, Moderate and Dense, compared pairwise, and touching
limits count as overlapping. 'Within 5 percentage points of one another' means max minus min of the three arm means
is at most 0.05. Part 5.4's Spearman trend uses the number of training levels (Sparse 2, Moderate 3, Dense 5) as the
factor, with a 97.5 percent percentile interval from resampling seed indices.

**Confidence.** High.

**Why.** The box's "exceeds 5 percentage points with an interval excluding zero" has the grammar of the H boxes'
"positive with an interval excluding zero": the first part is a property of the estimate and the interval is tested
against zero. Where the authors meant a limit to clear a margin they said so (G5: "an interval whose upper limit is
at most 10"), and Part 5.5's power is for a test against zero; the level is 97.5 percent because Part 5.2 tests G1
at α = 0.025. The overlapping intervals of the falsification belong to "the three ... arms' means", so they are
per-arm intervals, and the factor of the trend cannot change a rank correlation of three ordered arms.

**Rejected.** The lower limit of Dense − Sparse above 5 points (not powered by the design); intervals of the
differences for the overlap; 95 percent per-arm intervals; another factor for the trend (equivalent).

**Implemented in.** No code change: `analysis.study_b.g1_outcome`.

### Q-g2-outcome (starred)

**Answer.** G2 is decided by its box's per-budget count, and it remains a confirmatory (primary) test. At each
unseen budget, Dense is compared with its nearest single-level arm by equation (11): Single-10 at 5, Single-40 at
45, and at 15 and 30 whichever of the two equidistant single-level arms has the higher arm-mean zero-shot
satisfaction at that budget. On a budget floored by Part 4.2's floor rule, the choice is made on the outcome
compared instead: the candidate with the lower arm-mean zero-shot violation magnitude, again the choice least
favourable to G2. A budget is won when Dense minus comparator exceeds 0.05 with the 97.5 percent Welch interval
excluding zero. It is lost when the point estimate is at most 0.05. On a floored budget it is won when Dense's
violation magnitude is lower with the interval excluding zero, and lost when it is not lower. G2 is supported with
at least 3 of 4 budgets won and falsified with at least 2 lost. The per-seed composite contrast (Dense's mean over
the unfloored budgets minus the mean of each budget's comparator) is reported as descriptive, with Holm-adjusted
per-budget p-values beside it; neither decides. At 15 and 30 both candidate comparators' contrasts are reported. If
the two candidates tie exactly on the selecting outcome, the one least favourable to G2 on the outcome compared is
used.

**Confidence.** High. Residual (judge): Part 5.2 calls the primary outcome the mean over the four budgets, while box
G2 and Part 4.2 define G2 per budget; the defining passages prevail. Violation magnitude has no registered minimum
effect, so on a floored budget the thresholds come from Q-floor-rule.

**Why.** Box G2 counts budgets ("on at least three of the four"; "on at least two"), and Part 4.2 defines a
different nearest single-level comparator at each budget, so a composite has no registered construction and Part
5.2's mean is a summary. The count at 97.5 percent per budget keeps G2's confirmatory α = 0.025 without Holm: if G2
is false, a SUPPORTED verdict needs a false rejection on a truly null budget, with probability at most 0.025 by the
union bound (the critic's argument, checked by the judge). On a floored budget satisfaction "cannot separate arms"
(Part 4.2), so the comparator is chosen on violation magnitude, which keeps the choice least favourable to G2 (the
critic's edit).

**Rejected.** Deciding G2 on Part 5.2's four-budget mean (an unregistered construction); Holm over the per-budget
intervals inside the primary box; choosing the comparator by satisfaction on a floored budget.

**Implemented in.** `analysis.study_b.g2_outcome` (`Reading.g2_outcome = "count"`; `_g2_composite` descriptive).

### Q-g3-arms (starred)

**Answer.** G3, a secondary box, is decided on Sparse, Moderate, Dense and Continuous, the arms whose training range
is [10, 40]. Their nearest training levels by equation (11) are 10 for budget 5 and 40 for budget 45 (for
Continuous, the ends of its range). Each arm's satisfaction at those reference levels uses the same final
checkpoint, the same 100 measurement-set episodes and the deterministic policy as the zero-shot evaluation, with the
reference budget in the observation (studyb.evaluation.reference_budgets). Per seed, D = (sr(10) − sr(5)) − (sr(40) −
sr(45)). Each arm is supported when mean D > 0.05 with its 95 percent one-sample t-interval excluding zero. G3 is
supported when all four arms are. G3 is falsified when |mean D| ≤ 0.05 for more than half of them (at least 3 of
4). If budget 5 or 45 is floored, D is computed on violation magnitude by direction and interval only
(Q-floor-rule). The all-seven-arms reading, with each single-level arm's own nearest level, is reported beside as a
sensitivity reading.

**Confidence.** High.

**Why.** The G3 statement defines the estimand as extrapolation "at the same absolute distance of 5 cost units from
the training range (budget 5 below 10; budget 45 above 40)", which only the arms spanning [10, 40] meet; the
single-level arms would compare drops over unequal distances (Single-20: 15 and 25), which changes what is
estimated. Part 4.2's "its own nearest training level" is met by these four arms (10 and 40; the ends of the range
for Continuous). G3 is a secondary box (Part 5.2), and "for every arm" is an intersection rule that needs no
correction. The critic added the floor case and corrected the earlier §9 wording, which had called G3 confirmatory.

**Rejected.** All seven arms (break the equal-distance premise; reported beside as a sensitivity reading); a bin
midpoint or the range average as Continuous's reference.

**Implemented in.** No code change: `analysis.study_b.G3_ARMS` and `g3_outcome`;
`studyb.evaluation.reference_budgets`.

### Q-g4-reading (starred)

**Answer.** G4 is read on horizon indices: 1 = 200,000, 2 = 500,000, 3 = 1,000,000 steps. A run that never reaches
0.80 is censored and takes the index after the largest horizon, 'above the largest horizon' (Table 2.5): 4 with the
three registered horizons, 2 after Part 6.1 cut 3. The budgets used are those off the few-shot floor (Q-floor-rule).
The ordering clause 'median adaptation steps are ordered dense < moderate < sparse' is read on one statistic per
arm: the median over seeds of each seed's median index over those budgets, strictly ordered. 'Dense is at least one
horizon below sparse' holds on a budget when Sparse's median index over seeds minus Dense's median index over seeds
(a difference of medians) is at least 1. It must hold on more than half of those budgets. 'Within one horizon of one
another' holds on a budget when the largest minus the smallest of the three arms' per-budget medians is less than 1.
G4 is falsified when this holds on more than half of those budgets. Three sensitivity readings are reported beside
the verdict: 'within' as a spread of at most 1, the ordering on a majority of per-budget medians, and a censored run
placed at the largest horizon's index.

**Confidence.** High. Residual (judge): Interpretive only: the box does not say what the median is taken over. The
earlier readings are reported beside the verdict as sensitivity readings, so the choice stays visible.

**Why.** Box G4 attaches "on the majority of unseen budgets" to its second clause, and medians per budget lie on a
0.5 grid, so under the earlier per-budget ordering the clause "dense is at least one horizon below sparse" would add
nothing. An ordering at arm level (each seed's median index, then the median over seeds, as Part 5.1 summarises
within a seed first) gives both clauses force, mirrors G1's pooled ordering plus a size clause, and keeps G4
testable under Part 6.1 cut 3, where the per-budget ordering cannot hold. "Within one horizon" as a spread below 1
is the exact complement of "at least one horizon", so support and falsification exclude each other, and a censored
run takes the index after the largest horizon, following Table 2.5's "above the largest horizon". The critic noted
that the arm reading is not uniformly more conservative: its grounds are textual.

**Rejected.** The earlier proposal (ordering per budget on a majority); ordering on every budget; one median over
all seed × budget values; "within" as a spread of at most 1; censoring at the largest horizon's index.

**Implemented in.** `analysis.study_b.g4_outcome` (`Reading.g4_order = "arm_median"`, `g4_within = "less_than_one"`,
`g4_censored = "after"`).

### Q-floor-rule (starred)

**Answer.** Part 4.2's floor rule counts all seven Study B arms at each unseen budget, each by its mean zero-shot
satisfaction rate over its completed seeds. The budget is floored when more than half of the arms (at least 4 of 7)
have a mean strictly below 0.05. A floored budget is dropped from the mean over the four budgets used by G1 and G5.
Its comparisons (G1's Dense-Sparse and G5's Continuous-Dense, reported beside those verdicts; G2's win or loss at
that budget; the per-budget claims) are made on zero-shot violation magnitude (Table 2.5) by direction and interval
excluding zero, with no 5-point margin, and the report says so. If budget 5 or 45 is floored, both of G3's drops are
read as rises in violation magnitude. For G4 the same count is made on the few-shot rates at each horizon. A budget
floored at every horizon leaves G4's counts and arm statistic, whose majority is then taken over the remaining
budgets (if none remain, G4 is not computable). There, Dense and Sparse are compared on the few-shot violation
magnitude at the largest horizon, by direction and interval. When unknown rates could change the count, the
dependent comparisons are not computable. The readings 'only the arms of the comparison' and 'per-seed rates' are
reported beside as sensitivity readings.

**Confidence.** High.

**Why.** The floor rule is stated once, "for every comparison above", and refers to the arms of Study B, so one
count over the seven arms gives every box the same floored budgets and avoids forking paths (a two-arm count for G5
could floor other budgets than G1's count on the same data). "Arms have a satisfaction rate" is said of arms, so
each arm's mean over seeds counts. Violation magnitude has no registered minimum effect, so floored comparisons are
by direction and interval. For G4 the rule's reason ("a rate at the floor cannot separate arms") applies to the
few-shot rates, and flooring only at every horizon keeps the most data.

**Rejected.** Counting only the arms of each comparison; per-seed rates; switching a whole box to violation
magnitude; the zero-shot floor applied to G4; a few-shot floor at the largest horizon only.

**Implemented in.** No code change: `analysis.study_b.StudyB.floor`, `fewshot_floor` and `_vm_comparisons`, used by
`g1_outcome` to `g5_outcome`.

## Table 9.0 (to fill in Word)

From X-registration. Fill these in the `.docx`, regenerate the PDF, and commit both in a commit that changes nothing
else (HANDOVER.md task 3), after the tag is pushed and before any pilot run; the search record's field is filled
later, by its own Table 9.1 row when the record is committed (task 17).

| Field | Value |
|---|---|
| Repository (public) | `https://github.com/paf-iast-ai-research/when-safety-comes-late` |
| Registration commit hash | `735b394d18d1bb046c7a74f900af00a83f1fa316` |
| Commit timestamp | 2026-09-18T23:22:45+05:00 (2026-09-18T18:22:45Z) |
| Tag | `registration` (annotated, on 735b394; push it with `git push origin registration`) |
| Search record of Appendix C | `studyb/search/`, entered by its own Table 9.1 row when the record is committed (task 17) |

The first row of Table 9.1, the initial registration, takes the same date and hash and is approved by the group. By
the erratum of Q-g-names it reads "hypotheses G1 to G5", not "G1 to G4".

## Table 9.1 rows

Enter one row per amendment: the date of ratification (the meeting), the section of the pre-registration it amends,
the list of keys it answers, and "approved by the group"; the hash of the commit that records it goes with the row
(HANDOVER.md task 11). One row per section of this file keeps the log readable; the ledger schema, the allocation
and the search record are entered as rows of their own. Every row but the search record's is entered before the first
pilot run; the search record's row (with, for a 'partial' outcome, the amendment narrowing Study B's claims,
Q-search-before-pilot) is entered before the pilot's Study B runs (task 17).

| Row | Sections amended | Keys |
|---|---|---|
| Study A training and run gates | Tables 2.1, 2.3, 2.4 and 3.1; equations (4), (5) and (9) | Q-rounding, Q-cost-critic, Q-jc-window, Q-ramp-step, Q-warm-start, Q-rate-limit, Q-pid-eq9, Q-reset-injection, Q-plasticity-definitions, Q-data-control-lr, Q-determinism-late-onset, Q-lr-decay, Q-multiplier-adam, Q-onset-epoch, Q-manipulation-point, Q-first-checkpoint |
| Evaluation battery, continuations, matching and controller quantities | Tables 2.1, 2.2 and 2.5; Part 4.1; Part 5.6; Appendix B | Q-continuations, Q-transfer-obs, Q-hazard, Q-dynamics, Q-selection-window, Q-tie-break, Q-matched-cost-set, Q-arm-complete, Q-controller-quantities, Q-final-cost-set, Q-eval-seeds, Q-mujoco-exception, Q-return-outcome, Q-checkpoint-table, Q-rule3-tolerance-sensitivity |
| Study B training, evaluation and the targeted search | Table 2.5; Parts 3.3, 3.6 and 7.2; Appendix C | Q-pilot-unconstrained-seed, Q-budget-normalisation, Q-level-jc, Q-continuous-bins, Q-search-before-pilot, Q-studyb-order, Q-studyb-eval, Q-adapt-censoring, Q-search-depth, Q-search-queries, Q-search-citations, Q-search-record-scope |
| Operations, seeds, the go decision and the registration records | Parts 3.6, 5.5, 5.6 and 6; equation (12); Table 8.1; Table 9.1 (erratum); Appendices A and B | Q-interrupted-run, Q-seed-collision, Q-surplus-arm-set, Q-g1-level, Q-g3-pairing, Q-g4-level, Q-g2-run-equivalents, Q-exclusion-breaker, Q-p-list, Q-g-names, Q-ledger-v2, Q-appendix-a |
| Study A analysis and hypothesis readings | Parts 1.2, 4.1, 4.1.1, 5.1 to 5.5, 5.7, 5.8 and 8.2 | Q-interval, Q-h1-shape, Q-controls-reading, Q-support-falsify-overlap, Q-falsification-calibration, Q-holm-families, Q-threshold-arithmetic, Q-iqm, Q-bootstrap-details, Q-surplus-in-analysis, Q-h3-scope, Q-h4-reading, Q-h0-scope, Q-detectable-size, Q-exploratory-labels, Q-unmatched-reporting, Q-training-age-systematic, Q-treatment-other-N |
| Study B hypothesis readings and the floor rule | Parts 1.4 and 4.2 | Q-g1-criteria, Q-g2-outcome, Q-g3-arms, Q-g4-reading, Q-floor-rule |
| Ledger schema corrections | Appendix B (Table B.1) | X-ledger-schema-amendment: LS-1 to LS-32 of `docs/ledger_schema_memo.md`, approved by Role 4's pull request before the first pilot ledger row is written |
| Machine-hour allocation | Part 8 (Table 8.1); Part 6 G2 | X-allocation: the value H the group decides, recorded with `python -m pilot allocate` (task 12) |
| Search record | Table 9.0; Appendix C | The path `studyb/search/`, once the record is committed (task 17) |

X-registration is not an amendment: it fills Table 9.0 and the first row of Table 9.1 (above). In
`analysis/AMENDMENTS.json`, the ratification is the amendment `A1`, entered with the decisions: its `keys`
list all 54 starred keys, with `data_seen` false, since the answers were made before any data existed; Role 4
enters its commit hash (null until then) and the approval once the group has ratified it.

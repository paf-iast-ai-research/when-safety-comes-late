# envs/: Role 2, Environment and tests

**Owner: Muhammad Talha Jamil** (Environment and tests, Role 2). Second reviewer: the pilot owner,
Muhammad Umair Waseem (`.github/CODEOWNERS`).

## Status: written for you, not yet yours

The code in this folder was written by the pilot owner's build, which wrote first versions of every
role's modules so that the pipeline could be tested end to end. **Nothing of it is committed.** It
becomes Role 2's code only when you have read it, checked it against the pre-registration, and
adopted it through your own pull request (First Tasks, habit 3: the description cites the
pre-registration locations it implements). Change anything you disagree with; the docstrings say
where every choice comes from.

## What it implements

| Module | Pre-registration | What it does |
|---|---|---|
| `envs/onset.py` | Table 2.1 "Constraint onset", "Abrupt onset", "Linear ramp"; equations (3) to (5); Table 2.4 (controller variants, treatments, step-matching controls); Table 3.3 (PID check) | The Study A training plug-ins `study_a` (`make_algorithm`, on OmniSafe's `PPOLag`) and `study_a_pid` (`make_pid_algorithm`, on `CPPOPID`). Before onset the actor's surrogate is the reward advantage alone and the multiplier update is skipped; from onset OmniSafe's own update runs unmodified. Ramp, warm-started and rate-limited multipliers, partial reset and plasticity injection (timing only), additional constrained training. |
| `envs/continuations.py` | Table 2.2 "Reward-only fine-tuning" and "Transfer" | The battery plug-ins `battery_finetune` (`make_finetune_algorithm`) and `battery_transfer` (`make_transfer_algorithm`), restoring the parent's matched checkpoint with `pilot.dependencies.restore_learner`; the observation mapping between the Goal and Button tasks. |
| `envs/evaluation.py` | Table 2.1 "Evaluation cost of a checkpoint", "Episode", "Episodic cost", "Episodic return"; Table 2.2 "Hazard relocation", "Dynamics perturbation"; Part 3.4; Appendix B `final_cost`, `final_return`; Table 3.1 (one torch thread) | The seeded evaluation harness: selection and measurement sets, the battery conditions, contracts 1, 5 and 6 of `pilot/contracts.py`. It writes nothing to disk. |

`envs/__init__.py` imports none of the modules, so the pipeline and the harness never load OmniSafe's
algorithms through the package.

### Hook points in the pinned OmniSafe 0.5.0 (First Tasks, Role 2 step 1)

Checked against the installed `omnisafe` 0.5.0 package:

* Equation (4), the multiplier update: `omnisafe/algorithms/on_policy/naive_lagrange/ppo_lag.py:73-80`
  (`PPOLag._update`: J_C from `Metrics/EpCost`, then `update_lagrange_multiplier`, then the critics and
  the actor), which calls `omnisafe/common/lagrange.py:129-136` (`Lagrange.update_lagrange_multiplier`:
  Adam step, then clamp at 0) on the loss of `lagrange.py:112` (`-lambda * (J_C - cost_limit)`).
* Equation (3), the surrogate: `ppo_lag.py:101-102` (`_compute_adv_surrogate`:
  `(adv_r - penalty * adv_c) / (1 + penalty)`).
* PID check: `omnisafe/algorithms/on_policy/pid_lagrange/cppo_pid.py:75-81` (`_update`, calling
  `pid_update`) and `:102-103` (surrogate); the update itself is
  `omnisafe/common/pid_lagrange.py:92-125`.
* The onset itself changes only `_compute_adv_surrogate` and `_update` (the warm start is applied in
  `_update`, before the onset epoch's multiplier update), never `use_cost` (which in OmniSafe also
  decides whether cost advantages are computed and the cost critic trains; First Tasks, Role 2
  "Mistakes to avoid"). `_init` adds bookkeeping and puts `RateLimitedLagrange` in place of
  OmniSafe's `Lagrange` for the rate-limited variant and `DataControlLR` in place of the actor's
  LinearLR for the data control (Q-data-control-lr); `_init_log` registers the `Onset/` keys and, for
  reset and injection, wraps the rollout so that the intervention runs at the start of the onset
  epoch's rollout. Both critics train from the first epoch (Q-cost-critic, answered in
  `docs/DECISIONS.md`).

### Hook points of the evaluation harness (`envs/evaluation.py`)

Training's wrapper chain (`omnisafe/adapter/online_adapter.py:117-146`) without OmniSafe's
`AutoReset`: `SafetyGymnasiumEnv` with Safety-Gymnasium's inner `SafeAutoResetWrapper` removed,
`TimeLimit(1000)`, `ObsNormalize` with a `FrozenNormalizer` loaded from the same checkpoint,
`ActionScale(-1, 1)`, `Unsqueeze`. One seeded reset per episode, the deterministic mean action. The
dynamics perturbation hooks `task._build` (`DynamicsPerturbation`) so that the scaling is reapplied
after every model compile, then calls `mujoco.mj_setConst`. Hazard relocation (the central square of
Q-hazard) sets the hazards' `placements` to the square [-a, a]^2 and clears the task's cached
`placements_conf.placements`; the layout drawn for a layout seed is read from `task.world_info.layout`.

## Public API

Plug-ins (named in `pilot.manifest.PLUGINS`; the launcher calls `factory(spec.task, cfgs, spec)`):

* `envs.onset.make_algorithm`, `envs.onset.make_pid_algorithm`
* `envs.continuations.make_finetune_algorithm`, `envs.continuations.make_transfer_algorithm`

Helpers other code relies on:

* `envs.onset`: `build_onset_cfgs(spec, *, steps_per_epoch, registered=True)` (everything the factory
  adds goes into `cfgs['onset_cfgs']`, which is in `config.json` and in the configuration hash),
  `effective_budget(step, *, onset_step, cost_limit, d_loose=None, ...)` (equation 5),
  `rate_limit(old, proposed, *, relative, absolute, upper=None)`, `warm_start_source(cfgs, spec,
  onset_step)`, `RateLimitedLagrange`, `PARAM_KEYS` (must track `pilot/manifest.py` when the manifest's
  Study A parameters change), `is_determinism_spec`, `is_registered_spec`, `write_json_once` (also used
  by `studyb.conditioning`), and `INTERVENTION_FILE` and `PROPOSED_KEY` (mirrored in
  `pilot/enrichment.py`, where a test checks they agree).
* `envs.continuations`: `observation_layout(task)`, `transfer_map(source_task, target_task)`,
  `map_state_for_transfer(state, tmap)`, `validate_continuation_spec(spec, condition, *, env_id=None)`.
  The factories put the restore settings in `cfgs['continuation_cfgs']`.
* `envs.evaluation` (contracts in `pilot/contracts.py`):
  `evaluate_run(omnisafe_dir, spec)` (contract 1), `evaluate_battery(omnisafe_dir, spec, step,
  conditions)` (contract 5; `conditions=[]` measures C_ID only), `evaluate_checkpoint(omnisafe_dir,
  step, episodes, ...)` (contract 6, the determinism check), `evaluate_continuation(omnisafe_dir,
  spec)`; the seed sets `selection_seeds()`, `measurement_seeds()`, `hazard_layout_seeds()`,
  `hazard_episode_seeds()`; the primitives Study B reuses: `load_policy`, `run_episodes`,
  `append_budget`, `EpisodeResult`, `summarise`, `training_budget_schedule`; and `hazard_definition`,
  `dynamics_definition` (the condition definitions recorded in a battery result's `conditions`).
* Errors: `EvaluationRefused` (a `RunRefused`: exit 5, the run stays trained), `CheckpointInvalid` and
  `CheckpointUnreadable` (a `pilot.contracts.ContractError`: exit 1, `eval_failed`; restore the files,
  then `python -m pilot schedule resolve RUN reevaluate`), `MujocoInstabilityError` (exit 1,
  `eval_failed`, held for the group and never a Part 5.6 exclusion: the run is kept; raised when more
  than 5 episodes of one evaluation call, reserve episodes included, are unstable by MuJoCo's warning
  counters `mjWARN_BADQPOS`, `BADQVEL`, `BADQACC`, `BADCTRL`; up to that cap each unstable episode is
  never scored but replaced by one on the next seed of its set's reserve sequence (selection
  1,050,000 + r, measurement and dynamics 2,050,000 + r, hazard episodes 3,150,000 + r) and recorded in
  the result's `unstable_replacements`, the seed lists staying the canonical ones (Q-mujoco-exception,
  answered in Table 9.1; `pilot.contracts.validate_unstable_replacements`); `MUJOCO_LOG.TXT` is not
  read). Note the two classes for a bad checkpoint: in the harness an unreadable or unfitting
  checkpoint is `CheckpointInvalid` (the evaluation fails, exit 1), while a continuation's parent
  checkpoint that does not load is refused at launch by `pilot.dependencies.load_full_state`
  (`RunRefused`, exit 5, the run stays pending). Check that this split is what you want when you
  review.

No command line: the plug-ins run through `python -m pilot.launch train` and the harness through
`python -m pilot.launch evaluate` and `python -m pilot enrich ...` (see `pilot/README.md`).

## Files it writes

* `intervention.json` in the OmniSafe run directory of a reset or injection run (the record returned by
  `metrics.interventions`, with the run id, seed, onset step and epoch, and the output check on the
  fixed batch).
* `continuation.json` in the OmniSafe run directory of a fine-tuning or transfer run (the restore
  summary: parent checkpoint and its SHA-256, parameters restored, trainable counts).
* `progress.csv` columns under `Onset/` (`Onset/Active`, `Onset/EffectiveBudget`,
  `Onset/MultiplierBeforeUpdate`, `Onset/MultiplierProposed`), and `Metrics/LagrangeMultiplier` in every
  epoch, before onset too (contract 3).
* The harness writes nothing; its results are returned to the launcher and to `pilot.enrichment`.

## Decided questions it implements

Every question below is answered in `docs/DECISIONS.md` (2026-10-02), and the starred keys are in
`ANSWERED_QUESTIONS`, so nothing here waits on a question; the answers become amendments when the
group ratifies them. Each answer is a named constant or function commented with its key ("answered
in Table 9.1 (Q-key)"), or the behaviour the decision record describes.

* Former run gates (Q-rounding, Q-selection-window and Q-data-control-lr are set in
  `pilot/manifest.py` and checked here): Q-cost-critic, Q-ramp-step, Q-warm-start, Q-rate-limit,
  Q-pid-eq9 (`PID_STEP_MATCHING`), Q-reset-injection (timing), Q-rounding, Q-data-control-lr (the
  extension's schedule: `actor_lr_schedule`, `DataControlLR`), Q-continuations, Q-transfer-obs
  (`TARGET_ONLY_WEIGHT`, `TARGET_ONLY_MEAN`, `TARGET_ONLY_VARIANCE`), Q-selection-window,
  Q-budget-normalisation, Q-determinism-late-onset (the study_a_pid determinism check's copy of its
  arm, with the onset scaled into the check, accepted by `is_determinism_spec`).
* Former result gates: Q-hazard (`HAZARD_FORM = "central"`, `HAZARD_CENTRAL_HALF_WIDTH = 0.75`; the
  registered reading is implemented and tested beside it but enters no result), Q-dynamics,
  Q-studyb-eval (the budget branch of `evaluate_checkpoint` uses `training_budget_schedule`, which the
  registered `study_b` determinism report needs).
* Report key: Q-final-cost-set (`final_cost` from the measurement set; `final_selection_cost` kept).
* Answered notes, not `PENDING` keys: Q-eval-seeds (seed bases 1,000,000; 2,000,000; 3,000,000;
  3,100,000), Q-mujoco-exception (reserve seeds 1,050,000, 2,050,000 and 3,150,000 + r; at most 5
  unstable episodes per evaluation call), Q-onset-epoch (epoch e is constrained iff e >= onset_step /
  E; the multiplier update precedes the actor update), Q-first-checkpoint, Q-lr-decay (OmniSafe's
  linear decay of the actor's rate is kept, `check_training_config` requires it for the data control,
  whose `DataControlLR` continues it).

## Tests

Fast (default `pytest -q`; those marked `omnisafe` skip when OmniSafe is not installed):
`tests/test_envs_onset.py`, `tests/test_envs_continuations.py`, `tests/test_envs_evaluation.py`,
`tests/test_envs_evaluation_sim.py`.

Slow (`pytest -q -m slow`; tiny training runs of 2,000 steps per epoch, a few minutes each):
`tests/test_envs_onset_slow.py`, `tests/test_envs_continuations_slow.py`,
`tests/test_envs_evaluation_slow.py`.

## What the owner must still do

1. Review every module against Tables 2.1, 2.2, 2.4 and equations (3) to (5); adopt it by pull request
   (with `tests/test_envs_*`). Re-check the hook-point line numbers above on the workstation's pinned
   install.
2. Review the answers above in `docs/DECISIONS.md` before the group's ratification meeting; the
   ratified rows are entered in the amendment log (Table 9.1).
3. Run the slow tests on the workstation (`pytest -q -m slow`) and the registered-form determinism check
   of your plug-ins (`python scripts/determinism_check.py --registered-form --plugin study_a` and
   `--plugin study_a_pid`, whose onset follows Q-determinism-late-onset); see `pilot/README.md`.
4. Keep `PARAM_KEYS` and the arm names in `envs/onset.py` in step with `pilot/manifest.py` when either
   changes, and `INTERVENTION_FILE` and `PROPOSED_KEY` in step with their copies in
   `pilot/enrichment.py`.

Contracts with the pipeline: `pilot/contracts.py` and HANDOVER.md section 8. Questions and answers:
`docs/DECISIONS.md` and HANDOVER.md section 9.

# metrics/: Role 3, Metrics and interventions

**Owner: Abdullah** (Metrics and interventions, Role 3). Second reviewer: the pilot owner,
Muhammad Umair Waseem (`.github/CODEOWNERS`).

## Status: written for you, not yet yours

The code in this folder, and the three fixed batches, were written by the pilot owner's build, which
wrote first versions of every role's modules so that the pipeline could be tested end to end.
**Nothing of it is committed.** It becomes Role 3's code only when you have read it, checked it
against the pre-registration, and adopted it through your own pull request (First Tasks section 4,
habit 3: the description cites the pre-registration locations it implements).

## What it implements

| Module | Pre-registration | What it does |
|---|---|---|
| `metrics/batches.py`, `metrics/collect_fixed_batch.py`, `metrics/fixed_batch/` | Table 2.3 "Fixed evaluation batch" (2,048 states, uniform-random policy, seed 0, stored in the repository); Appendix A | The fixed batch of each Study A task: collection procedure (`batches.PROCEDURE`), loader with SHA-256 check (`load_fixed_batch`), the command that made the files and re-checks them. |
| `metrics/plasticity.py` | Table 2.3 (dormant-neuron fraction, equation 6, tau = 0.025; effective rank, equation 7, delta = 0.01; parameter norm; critics logged separately); Part 1.2 H3 (c) | `measure(...)`: every column of `plasticity.csv` from one forward pass of the batch through `actor.mean` and the critics' MLPs. |
| `metrics/hook.py` | Table 2.3 "Logging points"; Part 3.4 | `install_plasticity_hook(algo)`: one row at step 0 and one after every `logger.torch_save()` (cadence, onset, onset + 200,000, final). Installed by `pilot.algorithms.FullStateCheckpointMixin._init_log` when `cfgs.pilot_cfgs.plasticity` is true. Never draws a random number, never pushes the normaliser, never changes training. |
| `metrics/interventions.py` | Table 2.4 "Partial reset", "Plasticity injection"; equation (8) | `partial_reset`, `plasticity_injection`, `InjectedHead`, the generator seeded from the run seed, and the loaders of injected actors for evaluation and continuations. `envs.onset` (Role 2) decides when; this module decides what. |
| `metrics/controller.py` | Table 2.4 "Multiplier overshoot", "Settling time"; Appendix B `lambda_peak`, `lambda_final`, `settling_steps` | `controller_quantities(...)` from the multiplier trace in `progress.csv`, at epoch resolution; `rate_limit_clip_share(...)`, the share of constrained epochs in which the rate clip bound (Q-rate-limit). |
| `metrics/recovery.py` | Table 2.1 "Recovery time" | `recovery_steps(...)` from `Metrics/BatchEpCost`, at epoch resolution. |
| `metrics/recompute.py` | Table 2.3; Part 3.4 (an audit of contract 2) | Recomputes every row of a run's `plasticity.csv` from its saved checkpoints and compares. |

### First Tasks section 9 step 2: which input the metrics see

The pinned configurations set `algo_cfgs.obs_normalize: True` (`configs/omnisafe/PPOLag.yaml:48`, and
likewise `CPPOPID.yaml` and `PPO.yaml`). So the batch is passed through **the run's own observation
normaliser, frozen**: during training the hook reads the normaliser's current statistics without
updating them (`metrics.plasticity.normalise_frozen`; `Normalizer.normalize` is never called, since it
would push the batch into the running statistics); offline, `metrics.recompute` uses the
`obs_normalizer` saved in the same checkpoint. A run with `obs_normalize: False` would be measured on
raw observations. This is the answer to Q-plasticity-definitions (`docs/DECISIONS.md`).

## Public API

* `metrics.batches`: `load_fixed_batch(task)`, `FIXED_BATCH_SHA256`, `PROCEDURE`,
  `collect_fixed_batch(task, ...)`.
* `metrics.plasticity`: `measure(source, batch, *, step, ...)`, `dormant_fraction`, `effective_rank`,
  `parameter_norm`, `normalise_frozen`, `COLUMNS`.
* `metrics.hook`: `install_plasticity_hook(algo, *, batch=None) -> PlasticityRecorder`.
* `metrics.interventions`: `intervention_seed(run_seed)`, `intervention_generator(run_seed)`,
  `partial_reset(actor, optimizer, generator, depth=R.INTERVENTION_LAYERS, *, batch=None)`,
  `plasticity_injection(actor, optimizer, generator, depth=R.INTERVENTION_LAYERS, *, batch=None)`,
  `InjectedHead`, `is_injected(pi_state)`, `rebuild_injected(actor, pi_state)`,
  `build_actor(model_cfgs_or_config, obs_dim, act_dim, pi_state)`, `intervention_summary(record, *,
  step)` (the supplement's `InterventionSummary`).
* `metrics.controller.controller_quantities(...)`, `metrics.recovery.recovery_steps(...)` (both call
  `require_answered("Q-controller-quantities")`): `controller_quantities` is used by `python -m pilot
  enrich controller` and `enrich training`, `recovery_steps` by `enrich training` only.
* `metrics.controller.rate_limit_clip_share(epochs, multipliers, proposed, *, onset_step,
  steps_per_epoch, total_steps=None)`: the share of constrained epochs in which the rate clip bound,
  for the rate-limited arm (Q-rate-limit); no gate of its own (Q-rate-limit gates the rate-limited
  runs). Used by `enrich controller` and `enrich training`.

Commands (from the repository root):

```bash
python -m metrics.collect_fixed_batch --all            # collect once: refuses to overwrite an existing batch
python -m metrics.collect_fixed_batch --all --check    # re-collect and compare bytes with the committed files
python -m metrics.collect_fixed_batch --all --check --json PATH   # ... and write the result to PATH
python -m metrics.collect_fixed_batch --task TASK [--check]       # one task instead of --all
python -m metrics.recompute --run-dir RUN_DIR [--out FILE]   # FILE must lie outside the run directory
```

Exit codes: `collect_fixed_batch` 0 done / identical, 1 mismatch, 2 usage, 3 refused (nothing written),
4 error; `recompute` 0 identical (or no file), 1 different, 2 usage, 3 refused, 4 error. `recompute`
sets one torch thread, as training does, so that its floating-point results are the live ones.

Two cautions for anyone changing this code:

* Recomputation is bit-identical only with one torch thread (`R.TORCH_THREADS`).
* Do not `copy.deepcopy` an OmniSafe `GaussianLearningActor` after a forward pass with gradients: it
  keeps `_current_dist`, a non-leaf tensor, and deepcopy raises ("Only Tensors created explicitly by
  the user (graph leaves) support the deepcopy protocol"; checked on the pinned stack). Measure through
  `actor.mean` under `torch.no_grad`, as `hidden_activations` does.

## Files it writes

* `metrics/fixed_batch/<task>.npy` (float32, 2,048 x obs_dim) and `metrics/fixed_batch/MANIFEST.json`
  (procedure, versions, SHA-256), only by `collect_fixed_batch` without `--check`. `.gitignore` does not
  exclude them (`git check-ignore` finds no rule), and `.gitattributes` marks `*.npy` binary, so a plain
  `git add` commits them.
* `plasticity.csv` and `plasticity_meta.json` in the OmniSafe run directory of every plasticity run
  (contract 2). Columns, in order: `step, dormant, rank, norm, norm_reward_critic, norm_cost_critic,
  dormant_trainable, rank_trainable, dormant_reward_critic, rank_reward_critic, dormant_cost_critic,
  rank_cost_critic, norm_all` (13). The ledger writer reads the first four; the rest reach the analysis
  through the `training` supplement record.
* The record returned by the interventions is written by `envs.onset` to `intervention.json`.
* `metrics.recompute --out FILE`: a CSV of the recomputed rows.

## The fixed batches (Q-appendix-a)

Collected in the build sandbox with `python -m metrics.collect_fixed_batch --all` (seed 0, 2,048
states each; Python 3.10.21, numpy 1.26.4, gymnasium 0.28.1, mujoco 2.3.0, safety-gymnasium 0.4.1, as
recorded in `MANIFEST.json`):

| Task | Shape | SHA-256 |
|---|---|---|
| SafetyPointGoal1-v0 | 2,048 x 60 | `5340625cdd26c93fa4e610f7d5c93ae2a1f4d050ba64c1786a7a15928a182558` |
| SafetyCarGoal1-v0 | 2,048 x 72 | `abc01be62d4937d456e95372281252f99cf38da64d8931520ee34edcea7ec618` |
| SafetyPointButton1-v0 | 2,048 x 76 | `84ae724901954a6f8bfd96d55eebf7e6b7651536398e357175115e2570e98f0a` |

That the workstation's MuJoCo reproduces the same states has **not** been verified. Before the pilot,
run `python -m metrics.collect_fixed_batch --all --check` on the workstation after
`scripts/setup_env.sh` (or `python scripts/workstation.py fixed-batch`, which also writes
`environment/fixed_batch_check.json` with the outcome once the check runs to a result: a pass, or a
re-collection that differs (exit 1), recorded with the tasks that differ for the Table 9.1 row). The
committed batches are the fixed batches whatever the result (Q-appendix-a, answered in
`docs/DECISIONS.md`): the check is recorded as a reproducibility check. Exit 1 (a mismatch) is
reported to the group and in Table 9.1, but the batches are never collected again, since that would
be the resampling Table 2.3 forbids ("collected once per task"; First Tasks section 9 step 1: "never
resampled").

## Decided questions it implements

Every question below is answered in `docs/DECISIONS.md` (2026-10-02), and the starred keys are in
`ANSWERED_QUESTIONS`; the answers become amendments when the group ratifies them.

* Former run gates (set on the specs by `pilot/manifest.py`): **Q-plasticity-definitions** (the input
  above, the layers counted, the critics' metrics, the trainable-layer columns for H3 (c)) and
  **Q-reset-injection** (`INTERVENTION_SEED_PREFIX`; after injection the first hidden layer and
  `log_std` keep training, a departure from the literal "only the new copy is trained thereafter".
  For the first hidden layer its basis is Section 3 of Nikishin et al. (2023), arXiv 2305.15555, as
  two search-engine extracts report it; for `log_std` no source exists, and it is a symmetric choice
  fixed before data. The paper itself was not read, as the docstring of `metrics/interventions.py`
  says; Role 3 should read it before the ratification meeting and record what it finds).
* Former result gate: **Q-controller-quantities** (epoch-resolution overshoot, settling time and
  recovery time; the settling time is defined also when the final multiplier is 0, the epoch from
  which it stays 0).
* Answered note, not a `PENDING` key: Q-appendix-a (above).

The pilot's Study A runs never carried the Q-plasticity-definitions gate (its `PENDING` text gated
only the non-pilot Study A runs); the pilot records `plasticity.csv` under the same answer
(`pilot.contracts.plasticity_required`).

## Tests

Fast (default `pytest -q`): `tests/test_metrics_batches.py`, `tests/test_metrics_plasticity.py`,
`tests/test_metrics_hook.py`, `tests/test_metrics_interventions.py`, `tests/test_metrics_controller.py`,
`tests/test_metrics_recompute.py` (those marked `omnisafe` skip without the pinned stack).

Slow (`pytest -q -m slow`): the slow cases in `tests/test_metrics_hook.py` (tiny training runs: a row
at every checkpoint, training bit for bit the same with and without the hook, the mixin's extra saves
(onset off the save grid) and an injection composing with the hook, offline recomputation equal to the
logged rows, a batch problem refused by the launcher) and in
`tests/test_metrics_batches.py` (`--check` reproduces every committed batch, in this environment).

## What the owner must still do

1. Review every module against Table 2.3, equations (6) to (8) and Table 2.4; adopt it by pull request
   together with `metrics/fixed_batch/` and `tests/test_metrics_*`.
2. Run `python -m metrics.collect_fixed_batch --all --check` on the workstation (Q-appendix-a) and
   report the result in the pull request (and a mismatch in Table 9.1).
3. Review the answers to Q-plasticity-definitions, Q-reset-injection and Q-controller-quantities in
   `docs/DECISIONS.md` before the group's ratification meeting; read Nikishin et al. (2023) for
   Q-reset-injection.
4. Run the slow tests on the workstation (`pytest -q -m slow`).

Contracts with the pipeline: `pilot/contracts.py` (contract 2) and HANDOVER.md section 8.

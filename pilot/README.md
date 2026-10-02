# pilot/: Role 1, Pilot owner

**Owner: Muhammad Umair Waseem** (pilot owner, Role 1), who also owns `configs/`, `scripts/`,
`environment/`, `ledger/`, `prereg/`, `.github/` and the ledgers' contents. A second reviewer is
named in `.github/CODEOWNERS` once the group fills in the usernames.

## Status

This folder, and the first versions of every other role's modules, were written by the pilot owner's
build. `main` holds this folder as of 2026-09-28 (PR #2); **nothing written since is committed**: it
is in the pilot owner's working tree only. Each role owner reviews and adopts their own folder
through a pull request against `main` (role READMEs in `envs/`, `metrics/`, `analysis/`, `studyb/`;
HANDOVER.md, "How this repository reaches you"); this folder goes in through the pilot owner's own
pull request, reviewed by the second owner.

Review record of the build: 3 final adversarial review passes over 21 rounds; 112 findings handled
(each verified, then fixed). The last round still found 2 major findings, both fixed, so a further
review before the pilot is advised.

## What it implements

| Module | Pre-registration | What it does |
|---|---|---|
| `manifest.py` | Part 3.2 Table 3.2; Table 3.3; Part 3.3 Table 3.4; Part 3.6; Table 2.2; Part 6.1 | Every registered run as an immutable `RunSpec` with a deterministic `run_id` (610 in `design("all")`); `PLUGINS`; `run_gate_keys(spec)` and `open_run_gates(spec)`, the one place that says which open questions hold which run; `battery_continuation`; `apply_cuts`, `CUTS`. |
| `launch.py` | Table 3.1; Appendix A Table A.1; Part 3.4; Part 5.6 | One run: `train` (refusals before anything is written, the configuration as `omnisafe.Agent` builds it, one thread, claim, outcome classified by Part 5.6, `train_result.json`) and `evaluate` (the role's harness, checked, `evaluation.json`). |
| `algorithms.py` | Table 3.1; Part 3.4 | `make_ppolag`, `make_unconstrained_ppo`; `FullStateCheckpointMixin` (full learner state in every checkpoint, extra checkpoints at onset and onset + 200,000, at few-shot horizons and, for a total off the 200,000-step grid, at the end-relative selection window (Q-selection-window), the plasticity hook when `pilot_cfgs.plasticity`, per-epoch batch cost and return). |
| `dependencies.py` | Table 2.4 "Warm-started multiplier"; Tables 2.2 and 2.5 (continuations) | `Dependency`, `resolve` (before the claim), `to_config`, `from_cfgs`, `load_full_state`, `restore_learner` (the one restore rule of every continuation; Q-continuations). |
| `rundir.py` | Part 3.4 | Run-directory layout and readers (a leaf module; `launch.py` re-exports it). |
| `contracts.py` | Part 3.4; Part 4.1 rule 1; Table 2.1 | The interfaces other roles implement and their checks: `evaluator_target`, `load_evaluator(spec)`, checkpoint grid and `selection_window`, `extra_checkpoint_steps`, `plasticity_required`, `validate_plasticity`, `fewshot_key`, seed registries, `validate_evaluation`. |
| `errors.py` | (`configs/registered.py` `PENDING`) | `RunRefused`, `PendingQuestionError`, `require_answered`, `allow_pending` (smoke only; no environment variable). |
| `scheduler.py` | Part 3.6; Part 5.5; Part 5.6; Part 7.2; Table 3.1 | SQLite scheduler: queue, launch, adopt, settle, never duplicate; run gates and launch gates (`search_record_missing`, `main_sweep_unfinished`, `determinism_missing`, `analysis_missing`); operator actions. |
| `ledger_writer.py` | Appendix B Table B.1; Part 5.6 | The only writer of ledger rows: locked, atomic, validated by the frozen schema; the `evaluation` supplement record before the row. |
| `enrichment.py`, `supplement.py` | Part 4.1; Table 2.1; Table 2.2; Table 2.4; Table 2.5; Part 4.2; `results/supplement_schema.py` | Ledger fields after a row exists (written once; map fields only add keys) and the supplement records (`results/supplement/`, `results/pilot/supplement/`). |
| `budget.py` | Part 6 G2; Part 5.5 | Run-equivalents (the corrected 627.13 decides G2, the registered 484 beside it; Q-g2-run-equivalents), the allocation record (unit, basis, earlier workstation timings; X-allocation), the surplus-seed plan. |
| `go_decision.py` | Part 6 G1 to G4; equation (12); Part 8 Table 8.1 | Computes the four go conditions ("Go condition Gk (Part 6)") and the reading of Part 6; it does not take the decision. `Condition.values` keys read by `analysis/pilot_check.py` must keep their names (see `analysis/README.md`). |
| `provenance.py` | Appendix B; Part 3.4 | Commit hash, clean-tree rule, imported-code check, configuration hash, machine, UTC time. |
| `__main__.py` | | The command line. |

## Commands

```bash
python -m pilot design pilot -v                     # the 8 pilot runs, with any open question holding each (none now)
python -m pilot design all                          # 610 runs, 540.13 run-equivalents of training
python -m pilot budget [--cuts K]                   # corrected 627.13 run-equivalents (decides G2) against the registered 484
python -m pilot allocate --hours H --basis TEXT --by "the group" --date YYYY-MM-DD [--confirm] [--smoke-root DIR (repeatable)]   # then commit pilot/allocation.json
python -m pilot schedule add --design pilot         # also: add --cuts K, add-surplus --extra E, add-continuations --ledger L
python -m pilot schedule run --max-concurrent P     # long-running
python -m pilot schedule status [--events N]
python -m pilot schedule resolve RUN {restart,reevaluate,retry-ledger,unblock,replace,add-auxiliary,requeue}   # replace [--defect-fixed ERRATUM_ID]
python -m pilot schedule set-policy {hold,restart,exclude}   # registered data roots: restart (exclude is refused there)
python -m pilot enrich ACTION --ledger LEDGER [--surplus-extra E] [--conditions ...]
python -m pilot completeness [--ledger L]           # read-only
python -m pilot freeze [--ledger L]                 # read-only (Part 5.8)
python -m pilot g4-measure [--revision 0|1]         # writes results/pilot/g4_moderate[-rev1].json once
python -m pilot go [--repilot G ...]                # writes results/pilot/go_report-<UTC>.json and .md
python -m pilot surplus --hours-per-run X --concurrent P
python -m pilot.launch {train,evaluate} --spec RUN_DIR/spec.json --run-dir RUN_DIR    # normally the scheduler
python scripts/determinism_check.py --registered-form --plugin {ppolag,study_a,study_a_pid,study_b,unconstrained_ppo}
python scripts/smoke_run.py --data-root /scratch/smoke [--design determinism|pilot|sample]
python scripts/workstation.py status                # the runbook: each remaining step of HANDOVER.md section 4
```

`allocate` records H, the elapsed wall-clock hours of the workstation for the registered runs of the
whole design (all concurrent run slots together, not core-hours), with `--basis`, how H was derived
from the calendar and the machine's availability. Record it before any workstation timing exists: it
lists earlier timings in the record (`--smoke-root` names a smoke data root), is refused once a
registered run has started, and needs `--confirm` above 24 × 731 hours (X-allocation in
`docs/DECISIONS.md`).

Enrichment order (each step needs the ones before it): `select`, `measure`, `match --surplus-extra E`,
`battery`, `controller`, `training`, `final-battery`, `sensitivity-battery`; then `schedule
add-continuations` and `schedule run`, then `continuations`; for Study B `zeroshot`, `fewshot`;
`completeness` and `freeze` at any time.

Exit codes. `python -m pilot`: 0 done, 1 expected failure, 2 usage error or refused provenance check
(for example an uncommitted tree), 3 refused by an open question or a refusing step (nothing it
depends on is written), 130 interrupted (Ctrl-C). `pilot.launch`: 0 completed, 1 an uncaught
exception (the scheduler records `eval_failed` in the evaluation stage, `launch_failed` before the
claim of the training stage), 2 failed, 3 interrupted, 4 another role's code not in the repository
yet (unavailable), 5 refused before training (the run stays pending, or trained for `evaluate`).

`--allow-dirty` and `--allow-pending` are for smoke data roots only and are refused for the
repository's `results/`.

## Files it writes

* `<data root>/scheduler/state.sqlite`; run directories `<data root>/checkpoints/<run_id>/` with
  `spec.json`, `train_result.json`, `evaluation.json`, `omnisafe/...` (`config.json`, `progress.csv`,
  `torch_save/epoch-k.pt`); re-evaluated results moved to `<data root>/scheduler/reevaluated/`.
* `results/ledger.parquet` and `results/pilot/ledger.parquet` (the unconstrained pilot run in
  `results/pilot/sidecar/`), each with a seeds registry and an enrichment log beside it named from
  the ledger's stem (`results/ledger.seeds.json`, `results/ledger.enrichment_log.jsonl`; likewise
  under `results/pilot/`); supplement records under `results/supplement/` and
  `results/pilot/supplement/`. The repository's ledgers are registered by location and carry no mode
  marker; a ledger outside the repository's `results/` (a smoke root or `--ledger-dir`) also gets a
  marker named from the full file name (e.g. `ledger.parquet.mode`, and `sidecar.mode` for its
  sidecar directory).
* `pilot/allocation.json` (`allocate`); `results/pilot/g4_moderate[-rev1].json` (`g4-measure`);
  `results/pilot/go_report-<UTC>.{json,md}` (`go`); `ledger/determinism/determinism-*.{json,md}`
  (`scripts/determinism_check.py`); `environment/requirements.lock.txt` and
  `environment/workstation.json` (`scripts/setup_env.sh`, `scripts/record_workstation.py`).
* `pilot/cuts.json` is written by hand, by group decision (Part 6.1), and read by `schedule add --cuts`.

## Decided questions

The 54 `PENDING` keys of `configs/registered.py` are answered in `docs/DECISIONS.md` (2026-10-02) and
all are in `ANSWERED_QUESTIONS`, so no run, result or verdict waits on a question; the answers become
amendments when the group ratifies them (HANDOVER.md tasks 10 and 11). The pilot's runs waited on
Q-jc-window, Q-cost-critic, Q-budget-normalisation, Q-level-jc, Q-search-before-pilot and
Q-pilot-unconstrained-seed; they now wait only for the launch gates: the committed allocation, the
registered-form determinism reports and, for the two Study B runs, the search record.

The decided rules this folder applies:

* Q-interrupted-run: registered data roots run under `restart` (`schedule set-policy restart` once on
  a root whose policy was fixed before the key was answered; `exclude` is refused). An interruption
  of machine origin (SIGTERM, SIGINT, SIGHUP, or no scheduler as parent) is restarted without limit;
  a run that ends without a result a third time while the scheduler watched it (SIGKILL, a non-zero
  exit) is excluded as `incomplete` and replaced (`scheduler.MAX_RESTARTS`).
* Q-exclusion-breaker: after two same-cause exclusions of one arm the scheduler pauses its
  replacements; the operator diagnoses from logs, never results, records an erratum if a defect is
  fixed, then runs `schedule resolve RUN replace`, which needs `--defect-fixed ERRATUM_ID` once the
  arm's same-cause exclusions reach its seed target (`Scheduler.seed_target`). A fix must be
  bit-for-bit neutral for every configuration already run (checked with the determinism check on
  one completed configuration); if it is not, every completed run it affects is rerun on the
  corrected code, for which the scheduler has no action (HANDOVER.md task 32).
* Q-search-before-pilot: the pilot's two Study B runs wait for the complete search record, and a
  "partial" outcome also for its narrowing amendment in the committed `analysis/AMENDMENTS.json`.
* Q-g3-pairing: with fewer than three seed-matched pairs the leftover runs are paired in seed order
  (`go_decision._seed_pairs`), and the rule decides G3; the pooled value is descriptive.
* Q-g2-run-equivalents: G2 is decided on the corrected run-equivalents; the registered 484 is
  reported beside them.
* Q-g1-level, Q-g4-level, Q-matched-cost-set, Q-final-cost-set: the go report reads each condition by
  its answer and gives the other reading beside it; the headings say "Go condition Gk (Part 6)"
  (Q-g-names).
* Also: Q-rounding, Q-seed-collision, Q-surplus-arm-set (both onset shapes), Q-selection-window,
  Q-threshold-arithmetic, Q-tie-break, Q-arm-complete, Q-adapt-censoring, Q-determinism-late-onset
  (the study_a_pid determinism check, and through it the PID-check runs), Q-warm-start (`schedule
  resolve RUN add-auxiliary`; a warm-started run whose source is excluded is superseded by `schedule
  resolve RUN replace`) and Q-continuations (`dependencies.restore_learner`).

HANDOVER.md section 9 gives each question with a one-line answer.

## Tests

Fast (default `pytest -q`): `tests/test_manifest.py`, `tests/test_launch_and_configs.py`,
`tests/test_core_algorithms.py`, `tests/test_core_contracts.py`, `tests/test_core_dependencies.py`,
`tests/test_core_launch.py`, `tests/test_core_pending_keys.py`, `tests/test_core_quotes.py`,
`tests/test_core_refusals.py`, `tests/test_scheduler.py`, `tests/test_integration_sched.py`,
`tests/test_integration_sched_scripts.py`, `tests/test_ledger_writer_and_provenance.py`,
`tests/test_enrichment.py`, `tests/test_integration_enrich.py`, `tests/test_budget_and_go.py`,
`tests/test_registered.py`, `tests/test_workstation_runbook.py`; `tests/fake_launcher.py` simulates
runs, and `tests/conftest.py` holds the shared fixtures.

Slow (`pytest -q -m slow`): `tests/test_core_restore.py`, `tests/test_integration_sched_slow.py`,
`tests/test_integration_enrich_slow.py`, and the slow cases of `tests/test_core_refusals.py` and
`tests/test_launch_and_configs.py` (`test_launcher_reproduces_omnisafe_agent_bit_for_bit`: one
registered 20,000-step epoch, trained once through `pilot.launch` and once through `omnisafe.Agent`;
about six minutes).

## What the pilot owner must still do

1. Merge nothing until each folder is reviewed and the answers are ratified: commit this folder
   (with `configs/`, `scripts/`, `environment/`, the core tests) on a role branch cut from `main`,
   open its pull request, and ask each role owner to review and adopt theirs. Fill in
   `.github/CODEOWNERS` and turn on the ruleset.
2. On the workstation: `bash scripts/setup_env.sh`, commit the lock file and `workstation.json`; run the
   fast and slow suites; ask Role 3 for `python -m metrics.collect_fixed_batch --all --check`
   (Q-appendix-a).
3. Take `docs/DECISIONS.md` to the group for ratification at one meeting, enter its rows in Table 9.1
   and merge the code after that meeting (HANDOVER.md tasks 10 and 11). The pilot's Study B runs also
   wait until Hamza Nisar's search record is complete and committed.
4. Record the allocation the group decides (`allocate --basis`) before any workstation timing exists;
   run the registered-form determinism checks from a clean commit and commit the reports; a smoke run
   (`scripts/smoke_run.py`); then `schedule add --design pilot`, `schedule run`. The runbook
   `scripts/workstation.py` runs each of these steps (HANDOVER.md section 7).
5. Arrange the further review advised above before the pilot.

HANDOVER.md has the full plan, the contracts (section 8) and the questions with their answers
(section 9); `docs/DECISIONS.md` has the reasoning of each answer.

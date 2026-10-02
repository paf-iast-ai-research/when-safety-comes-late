# When Safety Comes Late, and Constraint-Coverage Generalization

Registered title: *Constraint-Onset Timing and Robustness in Constrained Reinforcement Learning
with the companion study Constraint-Coverage Generalization*.

Two pre-registered studies that share one pipeline (OmniSafe PPO-Lagrangian on Safety-Gymnasium).
**Study A, When Safety Comes Late**, asks whether a safety constraint added late in training gives
safety as robust as a constraint present from the start, when the policies compared are matched on
final in-distribution cost. It sweeps the onset fraction (0, 0.10, 0.25, 0.50), abrupt against
ramped onset, and two step-matching controls on three tasks. It tests whether loss of plasticity
(dormant neurons, effective rank) and overshoot of the Lagrange multiplier explain part of the effect.
**Study B, Constraint-Coverage Generalization**, asks whether the number and spacing of cost
budgets seen during training determine how well a budget-conditioned agent satisfies a budget it
has not seen, zero-shot and after a few-shot adaptation.

The registered plan is in [`prereg/`](prereg/). The repository's first commit
(`735b394d18d1bb046c7a74f900af00a83f1fa316`, 2026-09-18T23:22:45+05:00) is the registration, and
the plan governs every line of code here.

Research group: Muhammad Umair Waseem, Muhammad Talha Jamil, Hamza Nisar, Muhammad Abdullah,
Abdullah. Supervisor: Dr. Musadaq Mansoor. Pak-Austria Fachhochschule: Institute of Applied
Sciences and Technology, Haripur, Pakistan.

## State of the code (read this first)

* **The code was written by the pilot owner's build.** `main` holds the registration (`735b394`),
  Role 4's frozen ledger schema (PR #1) and the 2026-09-28 version of the pilot owner's part (PR #2:
  pinned environment, registered constants, run pipeline, scheduler, ledger writer and go/no-go
  tools). Everything written since, every other role's code included, is in the pilot owner's
  working tree only, not committed; HANDOVER.md ("How this repository reaches you") says how to
  bring it onto branches cut from `main`. Each owner reviews and adopts their folder through their
  own pull request; the role READMEs say what each folder holds and what its owner must still do.
* **The open pre-registration questions are answered** in [`docs/DECISIONS.md`](docs/DECISIONS.md)
  (2026-10-02, before any data; the reasoning in `docs/DECISIONS_EVIDENCE.md`): the 54 `PENDING`
  keys of `configs/registered.py`, the notes of HANDOVER.md section 9 and the ledger-schema
  amendment. The code implements every answer and `ANSWERED_QUESTIONS` lists all 54 keys. The
  answers were prepared by AI agents at the pilot owner's request; they become amendments only when
  the group ratifies them at one meeting (HANDOVER.md tasks 10 and 11), and the code is merged after
  that meeting.
* **The targeted search of Appendix C has not been run.** It is Hamza Nisar's to do by hand
  (`studyb/search/README.md`); no Study B run, the pilot's two included, starts before its record is
  complete and committed.
* **The fixed batches** in `metrics/fixed_batch/` were generated in the build sandbox. They are
  re-checked on the workstation with `python -m metrics.collect_fixed_batch --all --check` before the
  pilot; by the answer to Q-appendix-a they stand, and a mismatch is reported in Table 9.1.
* **Review record of the build:** 3 final adversarial review passes over 21 rounds; 112 findings
  handled (each verified, then fixed). The last round still found 2 major findings (fixed), so another
  review before the pilot is advised.
* **Tests:** `pytest -q` (the fast suite) in the build's trial environment (2026-10-01), on an
  earlier tree of 1,756 tests, ended `1705 passed, 51 deselected, 28 warnings in 527.92s (0:08:47)`
  (the duration varies between runs; an earlier run took 0:12:31); the 51 deselected were the slow
  tests (`pytest -q -m slow`). The tree has grown since (over 2,400 tests on 2026-10-02, more of
  them slow), so both suites have still to be run on the final tree and on the workstation.

## Structure

| Path | Owner | What it holds |
|---|---|---|
| `prereg/` | Group | The registered plan (PDF and DOCX). Changed only by amendment. |
| `configs/registered.py` | Muhammad Umair Waseem (Role 1) | Every registered number with its location; `PENDING` (the 54 questions); `ANSWERED_QUESTIONS` (all of them, answered in `docs/DECISIONS.md`). |
| `configs/omnisafe/` | Role 1 | Verbatim copies of OmniSafe 0.5.0's `PPOLag.yaml`, `CPPOPID.yaml`, `PPO.yaml`, with `SOURCE.json`. |
| [`pilot/`](pilot/README.md) | Muhammad Umair Waseem (Role 1) | Run manifest, launcher, scheduler, ledger writer, enrichment, supplement writer, budget, go or no-go report, command line `python -m pilot`. |
| [`envs/`](envs/README.md) | Muhammad Talha Jamil (Role 2) | Study A training plug-ins (onset, ramp, controller variants, treatments), battery continuations, evaluation harness. |
| [`metrics/`](metrics/README.md) | Abdullah (Role 3) | Fixed batches, plasticity metrics and their logging hook, partial reset and plasticity injection, controller quantities, recovery time. |
| [`analysis/`](analysis/README.md) | Muhammad Abdullah (Role 4) | The registered analysis of Parts 4 and 5, `python -m analysis`. |
| `results/ledger_schema.py` (frozen v1), `results/supplement_schema.py` | Muhammad Abdullah (Role 4) | The results ledger schema (Appendix B) and the supplement records' models. |
| `results/` ledgers and supplement | Role 1 writes, Role 4 reads | Created as runs complete; committed regularly. |
| `results/analysis/` | Role 4 writes (`python -m analysis`) | One new directory per analysis run (`<mode>-<UTC>/`). |
| [`studyb/`](studyb/README.md) | Hamza Nisar (Role 5) | Budget-conditioned training and few-shot continuations, Study B evaluations, the Appendix C search protocol and record. |
| `scripts/` | Role 1 | Environment setup, OmniSafe config copies, workstation record, determinism check, smoke run, registration tag, and the workstation runbook `workstation.py`. |
| `environment/` | Role 1 | Direct pins (`requirements.in`); the lock and workstation record are created on the workstation. |
| `ledger/determinism/` | Role 1 | Determinism reports. |
| `tests/` | Each role its own `test_<role>_*` files (Role 4 also `test_ledger_schema.py` and `test_supplement_schema.py`); Role 1 the rest | Fast suite by default; `-m slow` for the slow tests (tiny training runs, the full-resample analysis, the fixed-batch check). |
| `docs/ledger_schema_memo.md` | Role 4 | The ledger schema memo. |
| `docs/DECISIONS.md`, `docs/DECISIONS_EVIDENCE.md` | Group | The answers to the open questions (draft amendment rows for Table 9.1), and the verbatim reasoning behind them (generated). |
| `.github/CODEOWNERS` | Role 1 | Two owners per path; rules stay commented until the usernames are known. |

## Roles and owners

| Role | Owner | Folders |
|---|---|---|
| 1 Pilot owner | Muhammad Umair Waseem (`@Umair-Waseem`) | `pilot/`, `configs/`, `scripts/`, `environment/`, `ledger/`, `prereg/` (as code owner; changed only by group amendment), `.github/`; writer of the ledgers and supplement in `results/` |
| 2 Environment and tests | Muhammad Talha Jamil | `envs/` |
| 3 Metrics and interventions | Abdullah | `metrics/` |
| 4 Analysis and results | Muhammad Abdullah (`@Abdullah9712`) | `analysis/`, `results/ledger_schema.py`, `results/supplement_schema.py`, `docs/ledger_schema_memo.md`, `tests/test_ledger_schema.py`, `tests/test_supplement_schema.py` |
| 5 Study B and literature | Hamza Nisar | `studyb/` |

## Quick start

Linux (or WSL2 Ubuntu) with Python 3.10, from the repository root:

```bash
bash scripts/setup_env.sh                 # pilot owner, first time on the workstation (creates the lock)
bash scripts/setup_env.sh --locked        # everyone else: install exactly the committed lock
source .venv/bin/activate
pytest -q                                 # the fast suite
pytest -q -m slow                         # slow tests: tiny training runs, full-resample analysis, batch check
python -m pilot design pilot -v           # the 8 pilot runs, their steps and onsets
python -m pilot budget                    # registered 484 against corrected run-equivalents
python -m studyb.search check             # the Appendix C record: incomplete until the search is done
python -m metrics.collect_fixed_batch --all --check   # on the workstation (Q-appendix-a)
python scripts/workstation.py status      # pilot owner: each remaining step of HANDOVER.md section 4, DONE/TODO/BLOCKED
python scripts/workstation.py next        # run the next step; stops at a people's step (HANDOVER.md section 7)
```

On the workstation the pilot owner runs the remaining steps through the runbook,
`scripts/workstation.py`. It needs only the standard library, so `python3.10 scripts/workstation.py
next` also runs the first steps before `.venv` exists. It never commits or pushes; it prints the
`git add` and `git commit` lines to run.

## Where to read next

1. [`HANDOVER.md`](HANDOVER.md): the plan from now to the paper, the contracts between roles
   (section 8) and the questions with their answers (section 9).
2. [`docs/DECISIONS.md`](docs/DECISIONS.md): every answer, why it was chosen and where the code
   implements it; the group ratifies it at one meeting.
3. Your role's README: [`pilot/`](pilot/README.md), [`envs/`](envs/README.md),
   [`metrics/`](metrics/README.md), [`analysis/`](analysis/README.md), [`studyb/`](studyb/README.md).
4. `configs/registered.py`: the registered values, the `PENDING` questions and `ANSWERED_QUESTIONS`.
5. `pilot/contracts.py`: the interfaces between the pipeline and each role's code.

Every pull request cites the pre-registration location it implements (First Tasks, habit 3).

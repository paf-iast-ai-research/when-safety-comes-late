# studyb/: Role 5, Study B and literature

**Owner: Hamza Nisar** (Study B and literature, Role 5). Second reviewer: the pilot owner,
Muhammad Umair Waseem (`.github/CODEOWNERS`).

## Status: written for you, not yet yours

The code in this folder was written by the pilot owner's build, which wrote first versions of every
role's modules so that the pipeline could be tested end to end. **Nothing of it is committed.** It
becomes Role 5's code only when you have read it, checked it against the pre-registration, and adopted
it through your own pull request (First Tasks, habit 3).

**The targeted search of Appendix C has not been run.** No code runs it and no code or automated
agent has written a search result: the record files in `studyb/search/` (`search_log.md`,
`search_log.csv`, `search_record.csv`, `citations_log.csv`, `screened/`) are empty templates. The
search is yours to do by hand (`studyb/search/README.md`), and no Study B run, the pilot's two
included, starts until its record is complete and committed.

## What it implements

| Module | Pre-registration | What it does |
|---|---|---|
| `studyb/conditioning.py` | Table 2.5 "Budget conditioning", "Training set of levels", "Few-shot adaptation"; Table 3.1 "Networks"; Table 3.4; Part 3.3; equations (3) and (4) per level; Part 3.4 (per-level multipliers logged every epoch); Part 5.1 (seed pairing) | The plug-ins `study_b` (`make_algorithm`: one budget-conditioned PPO-Lagrangian arm) and `study_b_fewshot` (`make_fewshot_algorithm`: a few-shot continuation from the parent's final checkpoint (Q-continuations), one fresh multiplier at the unseen budget, checkpoints at every horizon). The budget / 100 is appended after the whole wrapper chain, so the observation normaliser never sees it. One OmniSafe `Lagrange` per level; the per-sample penalty is read from the budget feature. |
| `studyb/evaluation.py` | Table 2.5 "Zero-shot evaluation", "Constraint-satisfaction rate", "Violation magnitude", "Adaptation steps"; equations (10) and (11); Part 4.2; Part 6 G4; Part 3.4 and Appendix B (contract 1) | `evaluate_run` (contract 1 for `study_b` runs), `evaluate_training_budgets` (G4: the Moderate arm at each of its training budgets, read by `python -m pilot g4-measure`), `evaluate_zero_shot`, `evaluate_fewshot`, and the pure functions `satisfaction_rate`, `violation_magnitude`, `distance`, `nearest_training_level`, `nearest_single_arms`, `adaptation_steps`, `censored_steps` and `reference_budgets`. All use Role 2's harness primitives (`envs.evaluation`). |
| `studyb/search/` | Part 7.2; Appendix C, Table C.1; Part 9 Table 9.0 | `protocol.py` (sources, queries and added phrases from `configs/registered.py`, the three known works, the answers of Table 9.1 on depth, order and citation lists), `record.py` (`check_record(root, read=None) -> SearchStatus`), the empty record templates, and `studyb/search/README.md` with the steps. |

## Public API and commands

* Plug-ins (`pilot.manifest.PLUGINS`): `studyb.conditioning.make_algorithm`,
  `studyb.conditioning.make_fewshot_algorithm`. Everything the factory derives is in `cfgs.studyb_cfgs`
  (hashed): `kind`, `arm`, `level_keys`, `levels`, `continuous_range`, `bin_width`, `budget_divisor`,
  `budget_stream`, `jc_rule`, `window_source`, `budget_placement` and `penalty`; for a few-shot
  continuation also `budget`, `horizons`, `parent_run_id`, `parent_step` and
  `fresh_multiplier_init`; for the Continuous arm, `bins`.
* Budget draws: `budget_generator(seed) = numpy.random.default_rng([seed, 1])`, never a global stream,
  so seed k builds the same networks and layouts in every arm (Part 5.1).
* `studyb.evaluation` functions above; `GATES = ("Q-studyb-eval",)`; `SEED_SET = "measurement"`.
* `studyb.search`: `check_record`, `SearchStatus`, `RECORD_PATH = "studyb/search"`, `query_variants()`,
  `SCREENING_DEPTH`, `RESULT_ORDER`.

```bash
python -m studyb.search queries   # the 80 searches (4 sources x 5 queries x 4 variants) and their screened files
python -m studyb.search check     # exit 0 only when the record is complete; today it lists what is missing
```

## Files it writes

* `progress.csv`: `Metrics/LagrangeMultiplier/level_{b:g}` every epoch (the Continuous arm names each
  bin by its lower edge; there is no plain `Metrics/LagrangeMultiplier` column), and diagnostics
  `Metrics/LevelJc/level_*`, `Metrics/LevelBudget/level_*`, `Metrics/LevelEpisodes/level_*`,
  `Metrics/LevelWindow/level_*`, `Metrics/LevelBatchEpCost/level_*`.
* Checkpoints gain `level_lagranges`, `level_lambda_optimizers` and `level_window` (loadable with
  `weights_only=True`).
* `continuation.json` in the OmniSafe run directory of a few-shot continuation (the restore summary).
* The evaluations write nothing; `pilot.enrichment` turns them into ledger fields (`sr_zero`,
  `sr_fewshot`, `adapt_steps`) and `zero_shot` / `fewshot` supplement records.
* `studyb/search/` is written by hand only.

## Decided questions it implements

Every question below is answered in `docs/DECISIONS.md` (2026-10-02), and its `PENDING` keys
(starred in `docs/DECISIONS.md`) are in `ANSWERED_QUESTIONS`; the answers become amendments when the
group ratifies them.

* Former run gates (on the specs; `pilot.manifest.run_gate_keys`): **Q-budget-normalisation** (the
  feature after the normaliser), **Q-level-jc** (a level's J_C from its own episodes in OmniSafe's
  50-episode `Metrics/EpCost` window; a level with none there takes no step), **Q-continuous-bins**
  (bins [10, 15), ..., [30, 35), [35, 40], read from the float32 feature; a bin's d is the mean budget
  of its window episodes; one optimiser step per update), **Q-jc-window**, **Q-continuations**
  (few-shot restore from the final checkpoint; under Part 6.1 cut 3 a shortened continuation keeps
  the 1,000,000-step schedule and stops at 200,000 steps), **Q-studyb-order** (a non-pilot run starts
  when every main-sweep run under the cuts in force has finished training; an excluded run counts as
  finished), **Q-search-before-pilot** (the pilot's two Study B runs wait for the complete search
  record, and a "partial" outcome also for its narrowing amendment in the committed
  `analysis/AMENDMENTS.json`).
* Former result gate: **Q-studyb-eval** (final checkpoint, measurement seeds at every budget and
  horizon, training budgets in turn for contract 1; for the Continuous arm, the midpoints of 100
  equal sub-intervals of [10, 40]). It covers every evaluation of a budget-conditioned run, the
  pilot's G4 measurement and the registered `study_b` determinism report included.
* Implemented here (`adaptation_steps`, `censored_steps`; `evaluate_fewshot` reports `adapt_steps`),
  and checked again by `pilot.enrichment.adaptation_steps`: Q-adapt-censoring (the largest horizon + 1
  when never reached).
* Answered notes, not `PENDING` keys: Q-search-depth (`SCREENING_DEPTH = 50`, in each source's
  relevance order, `RESULT_ORDER`; citation lists in full), Q-search-queries (four variants of each
  query), Q-search-citations (both lists of every work that cites Yao et al.), Q-search-record-scope
  (every screened result listed with a reason), Q-multiplier-adam.

## Tests

Fast (default `pytest -q`): `tests/test_studyb_conditioning.py`, `tests/test_studyb_evaluation.py`,
`tests/test_studyb_quotes.py`, `tests/test_studyb_search.py`.

Slow (`pytest -q -m slow`): `tests/test_studyb_training.py` (tiny training runs of the plug-ins).

## What the owner must still do

1. **Run the targeted search** (Appendix C) by hand, following `studyb/search/README.md`: fix the
   screening depth and the result order first (the four lines at the top of `search_log.md`), enter
   the three known works, run all 80 searches and the citation lists, screen in two stages, apply the
   decision rule in `decision.json`, until `python -m studyb.search check` reports `complete: True`.
   Commit the record, report the outcome to the group, and give the path `studyb/search/` to the pilot
   owner for Table 9.0. The scheduler holds every Study B run (`search_record_missing`) until HEAD
   holds a complete record whose outcome is not `included` and, for a `partial` outcome, the
   amendment `decision.json` names is committed in `analysis/AMENDMENTS.json`.
2. Review `conditioning.py` and `evaluation.py` against Table 2.5 and equations (3), (4), (10), (11),
   and adopt them by pull request with `tests/test_studyb_*`.
3. Review the answers above in `docs/DECISIONS.md` before the group's ratification meeting, in
   particular Q-studyb-eval, Q-search-before-pilot and the four search notes.
4. Run the slow tests on the workstation (`pytest -q -m slow`).
5. Optional: a registered constant for Table C.1's "Already known" works in `configs/registered.py`,
   imported by `studyb/search/protocol.py` (today `KNOWN_WORKS` lives in `protocol.py`).

Contracts with the pipeline: `pilot/contracts.py` and HANDOVER.md section 8.

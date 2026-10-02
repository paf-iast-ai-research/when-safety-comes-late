# analysis/: Role 4, Analysis and results

**Owner: Muhammad Abdullah** (Analysis and results, Role 4; GitHub `@Abdullah9712`, author of PR #1,
the ledger schema). Second reviewer: the pilot owner, Muhammad Umair Waseem (`.github/CODEOWNERS`).
Role 4 also owns `results/ledger_schema.py` (frozen schema v1, with `results/ledger_schema.sha256`,
`results/ledger_schema_requirements.txt`, `tests/test_ledger_schema.py` and
`docs/ledger_schema_memo.md`) and the new `results/supplement_schema.py` with
`results/supplement_schema.sha256`.

## Status: written for you, not yet yours

Everything in this folder, and `results/supplement_schema.py`, was written by the pilot owner's
build, which wrote first versions of every role's modules so that the pipeline could be tested end
to end.
**Nothing of it is committed** except an earlier skeleton of this README (commit b527c84). It
becomes Role 4's code only when you have read it, checked it against Parts 4 and 5, and adopted it
through your own pull request (First Tasks, habit 3). Part 5.8
requires the analysis script to be committed before the first main-study run completes and to be run
once on the pilot data; from the go decision it is frozen except for recorded error fixes.

## What it implements

| Module | Pre-registration | What it does |
|---|---|---|
| `matching.py` | Part 4.1 rules 1 to 7 | `select_checkpoint` (rule 1; pipeline contract 4, called by `python -m pilot enrich select`), `match_arms(rows, *, tolerance, seed_targets)` (rules 2 to 6), the rule 7 tolerance. |
| `stats.py` | Part 5.3 to 5.5; equation (12) | Difference of seed means, Cohen's d (pooled), Welch t-interval, percentile bootstrap over seed indices, IQM with a two-level stratified bootstrap, Spearman with a seed-resampling interval, Holm, noncentral-t power, U80. |
| `study_a.py` | Part 1.2 boxes H0 to H4; Part 4.1.1; Part 5.2 secondary outcomes | Study A verdicts and tables. |
| `study_b.py` | Part 1.4 boxes G1 to G5; Part 4.2 (comparisons, floor rule) | Study B verdicts and tables (97.5 percent intervals for the primary family). |
| `verdict.py` | Part 1.2; Part 8.2 | SUPPORTED / FALSIFIED / INCONCLUSIVE / NOT_TESTED / NOT_COMPUTABLE, `UNDECIDED(key)` when the readings of an open question disagree, `provisional_on`, exploratory labels, the Part 5.7 bound that confirms a falsification (Q-falsification-calibration) and the Part 5.5 annotation. |
| `questions.py` | `configs/registered.py` `PENDING` | Which open questions each verdict depends on. |
| `data.py` | Part 5.1 | The ledger and the supplement records as per-seed tables; rule 1 re-checked; censored `adapt_steps` decoded as greater than the largest horizon; Part 6.1 cuts from `pilot/cuts.json`; auxiliary runs (`manifest.AUXILIARY_NOTE`) left out of their arm. |
| `records.py`, `AMENDMENTS.json`, `ERRATA.json` | Part 8.2; Part 5.8; Part 9 Table 9.1 | Amendments that relabel verdicts exploratory; errata after the go decision; the analysis code hash, over `analysis/*.py` and the in-repository modules the analysis imports (`records.IMPORTED_CODE`; Q-exploratory-labels). |
| `pilot_check.py` | Part 5.8 (the test run) | Recomputes the go report's G1 and G3 numbers on the pilot ledger and compares them. |
| `report.py`, `__main__.py` | Part 5; Part 5.8 | Tables, `analysis.json`, `report.md`, provenance; the command line. |
| `results/supplement_schema.py`, `results/supplement_schema.sha256` | Appendix B (what the frozen row cannot hold); Q-ledger-v2 | Pydantic models of the supplement records written by `pilot.supplement`, frozen at version 1 with their SHA-256 (`sha256sum -c results/supplement_schema.sha256` from the root). |

The verdicts' question keys (run gates, result gates and report keys alike; `configs/registered.py`
names the kinds) are reported per verdict: a verdict that depends on an open one carries it in
`provisional_on`, or is `UNDECIDED(key)` when its readings disagree. All of them are answered
(`docs/DECISIONS.md`, 2026-10-02), so with `ANSWERED_QUESTIONS` as committed every verdict is read
by its answer alone.

## Commands

```bash
python -m analysis --mode pilot --ledger results/pilot/ledger.parquet     # Part 5.8 test run on the pilot
python -m analysis --mode final --ledger results/ledger.parquet --surplus-extra E
```

Options: `--supplement DIR` (default `<ledger dir>/supplement`), `--out DIR` (default
`results/analysis/<mode>-<UTC>/`; must not exist), `--revision 0|1` (pilot mode: the pilot or the
re-pilot; `--revision 1` is refused in final mode), `--surplus-extra E` (final mode, required: surplus
seeds per primary-comparison arm, as for `python -m pilot enrich match`), `--allow-dirty` (smoke and
test runs only, from uncommitted code; its `--out` must lie outside the repository's `results/`).

Exit codes: **0** done; **2** refused (nothing written); **3** written, but in pilot mode the analysis
and the go report disagree on a number (the test run found an error); **4** written, but in pilot mode
nothing could be compared with the go report.

The command refuses a dirty tree, uncommitted imported code, HEAD moving during the run to a commit
that changes loaded code, the analysis code (`analysis/*.py` and the modules it imports,
`records.code_hash`) changing on disk during the run, and a `pilot/cuts.json`,
`AMENDMENTS.json` or `ERRATA.json` that differs from the commit.

## Files it writes

`<out>/analysis.json`, `<out>/report.md`, `<out>/tables/<name>.csv` and `.json`, in a new directory
(default `results/analysis/<mode>-<UTC>/`), never overwritten. It writes nothing else; the ledgers and
the supplement are written by the pilot owner's tools only.

### Supplement records (`results/supplement_schema.py`; read by `analysis.data`)

One JSON file per (kind, run, part) under `results/supplement/` (main study) or
`results/pilot/supplement/` (pilot): `evaluation`, `measurement`, `battery` (part = condition),
`final_battery` (part = measurement, hazard or dynamics), `sensitivity_battery` (part = condition),
`continuation` (part = finetune or transfer), `zero_shot`, `fewshot` (part = `b<budget>`), `training`.
The models forbid unknown fields, non-finite floats and mean values that disagree with their episodes.
The schema is frozen at `SUPPLEMENT_SCHEMA_VERSION` 1, with its SHA-256 in
`results/supplement_schema.sha256` (Q-ledger-v2); a change needs an amendment, and a version bump if a
stored record would become invalid. The registered constants it imports from `configs/registered.py`
follow that file's amendment rule.

## Contract with `pilot/go_decision.py`

`pilot_check` reads these keys of the go report's `Condition.values`, which must keep their names:
G1 `reference_cost_mean`, `late_cost_mean`, `reference_sd`, `match_difference`,
`both_arms_within_budget`, `arms_matched`, `reference_sd_ok`, `reference_final_checkpoint_sd`; G3
`delta_mean`, `U80`, `U80_pooled` (descriptive only, Q-g3-pairing). Both sides round G1's clauses
the same way (Q-threshold-arithmetic) and pair G3's seeds the same way (`stats.g3_pairs`,
`go_decision._seed_pairs`), so every disagreement is a mismatch (exit 3).

## Decided questions it implements

`analysis/questions.py` lists every key per verdict; every one is answered in `docs/DECISIONS.md`
(2026-10-02), and the answers become amendments when the group ratifies them. Implemented in this
folder: Q-tie-break, Q-threshold-arithmetic, Q-matched-cost-set and Q-selection-window (`matching.py`),
Q-interval, Q-iqm, Q-bootstrap-details, Q-holm-families, Q-g3-pairing (`stats.py`), Q-h0-scope,
Q-h1-shape, Q-controls-reading, Q-h3-scope, Q-h4-reading, Q-support-falsify-overlap,
Q-falsification-calibration, Q-surplus-in-analysis, Q-controller-quantities (Study A), Q-g1-criteria,
Q-g2-outcome, Q-g3-arms, Q-g4-reading, Q-floor-rule (Study B), Q-arm-complete (seed targets),
Q-adapt-censoring (`data.py`); and the notes that are not `PENDING` keys: Q-unmatched-reporting,
Q-return-outcome, Q-detectable-size, Q-exploratory-labels, Q-training-age-systematic,
Q-rule3-tolerance-sensitivity, Q-treatment-other-N, Q-manipulation-point (`study_a.h3c_outcome`),
Q-checkpoint-table, Q-ledger-v2 (the supplement).

The answers that changed an earlier proposal here: a box is FALSIFIED only when its Part 5.7 bound
lies below the minimum effect, otherwise INCONCLUSIVE (Q-falsification-calibration); a secondary
claim must survive Holm in both its families (Q-holm-families); the H1 trend and H3 (a) use each
arm's registered seeds and their replacements (`data.five_seed_view`; Q-surplus-in-analysis); G4
is read on arm-level medians, "within one horizon" as a spread below 1 (Q-g4-reading); with fewer than
three seed-matched pilot pairs, the runs a replacement leaves unpaired are paired in seed order and the
rule decides G3 (`stats.g3_pairs`, `pilot_check.g3_numbers`; Q-g3-pairing); "never reaches the budget"
is rule 3's test (Q-unmatched-reporting); training ages are compared for every compared pair
(Q-training-age-systematic); the note below the detectable size has new wording (Q-detectable-size);
and the code hash covers the imported modules (Q-exploratory-labels).

Points for the group:

* Part 5.5 prints the 80-percent point at ten seeds as 1.33; the noncentral t gives 1.3249, which rounds
  to nearest as 1.32 (power at 1.33 is 0.803). The tests accept the registered values within 0.01
  (`stats.py` docstring).
* H3 (b) "additional constrained training reduces the gap as much as reset" is read as requiring
  additional training to reduce the gap at all and by at least reset's amount; a reading that
  requires only "as much as" would drop the first half (`study_a.h3b_outcome` docstring).

## Tests

Fast (default `pytest -q`): `tests/test_analysis_matching.py`, `tests/test_analysis_stats.py`,
`tests/test_analysis_study_a.py`, `tests/test_analysis_study_b.py`, `tests/test_analysis_data.py`,
`tests/test_analysis_questions.py`, `tests/test_analysis_synthetic.py`, `tests/test_analysis_cli.py`,
`tests/test_supplement_schema.py`, `tests/test_ledger_schema.py`. One slow case in
`tests/test_analysis_cli.py` (`pytest -q -m slow`).

## What the owner must still do

1. Review every module against Parts 4 and 5 and adopt it by pull request, with
   `results/supplement_schema.py` and `tests/test_analysis_*`, `tests/test_supplement_schema.py`. A
   further targeted review of `analysis/` is advised before the go decision: in the build's review
   record, round 4 of this folder's review still found major findings (fixed), and the folder is large
   (over 7,000 lines of Python).
2. Review the answers above in `docs/DECISIONS.md` before the group's ratification meeting (Parts 1, 4
   and 5 freeze at the go decision). At ratification, complete the analysis amendment `A1` already in
   `AMENDMENTS.json` (all 54 keys, `data_seen` false; enter the commit hash, null until then, and the
   approval), and approve the ledger-schema amendment LS-1 to LS-32 by
   pull request before the first pilot ledger row is written (X-ledger-schema-amendment).
3. Run `python -m analysis --mode pilot --ledger results/pilot/ledger.parquet` on the pilot data (Part
   5.8) from a clean commit; exit 3 means an error to fix before the go meeting.
4. At the go decision, record `analysis_code_hash` in `AMENDMENTS.json` (`records.code_hash()`: the
   analysis code and every in-repository module it imports); after it, record every fix in
   `ERRATA.json` and Table 9.1.

Contracts: `pilot/contracts.py` (contract 4) and HANDOVER.md section 8. Questions and answers:
`docs/DECISIONS.md` and HANDOVER.md section 9.

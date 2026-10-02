# Targeted search for Study B (Appendix C)

Owner: Study B and literature (Role 5, Hamza Nisar).

Part 7.2 of the pre-registration: "A targeted search for prior work that varies the number or
spacing of training budgets as the independent variable is run by the protocol of Appendix C and its
record is committed to the repository before Study B's first run; the path of the record is entered
in Table 9.0." This directory is that record (its path, `studyb/search/`, goes to the pilot owner for
Table 9.0 once the search is run) together with the code that states the protocol and checks the
record.

**The search is the Study B owner's act.** It needs normal internet access and a person's
judgement. No code in this repository runs it, and no code or automated agent writes a search
result, a citation, a date, a hit or a classification into these files: they are written by hand by
the searcher. The CSV files committed here are empty templates (headers only), `search_log.md` is a
template to fill in, and `decision.json` is absent, so the record is incomplete until the search is
done, and the scheduler starts no Study B run before then
(pilot/scheduler.py: `studyb.search.check_record` on the files committed at HEAD; the gate holds the
pilot's two Study B runs too, Q-search-before-pilot below).

## What is fixed, and where

* `protocol.py`: the sources, queries and added phrases of Table C.1 (from `configs/registered.py`:
  `SEARCH_SOURCES`, `SEARCH_QUERIES`, `SEARCH_ADDED_PHRASES`, `SEARCH_CITATION_SEED`), the three works
  Table C.1 lists as already known, and the answers (Table 9.1) to what Table C.1 leaves open (below).
* `record.py`: `check_record(root, read=None) -> SearchStatus(complete, outcome, problems,
  amendment)`; its docstring gives the exact format of every file. The files follow the layout
  suggested by the Study B specification (section 8.1; a build document kept outside this repository)
  with six recorded departures (see that docstring): `citations_log.csv` names a list by `list_id`,
  `kind` and `of_work` instead of one `list` column and adds `items_reported`; the screened files
  allow `duplicate_of`; `search_record.csv` has a `known_work` column; `export_file` is optional in
  both logs; `citations_log.csv` also allows an optional `notes` column. First Tasks, Role 5 step 2
  lists every query in `search_log.md`; here that listing is `search_log.csv` (with the `screened/`
  files and `citations_log.csv`), and `search_log.md` is the narrative log, so the pull request
  contains all of these files.
* `python -m studyb.search queries` prints the 80 searches (4 sources x 5 queries x 4 variants) with
  the name of each one's screened-results file; `python -m studyb.search check` checks the record
  (exit 0 only when it is complete).

## Steps (after First Tasks, Role 5; numbered here, not as there)

1. Fix the screening depth and the result order before the first query: fill in the four lines at
   the top of `search_log.md` (`Screening depth:`, `Result order:`, `Depth fixed on:`, `Fixed by:`).
   The depth must equal `protocol.SCREENING_DEPTH` and the order must begin with
   `protocol.RESULT_ORDER` (Q-search-depth: the first 50 results of each search, in each source's
   relevance order; in arXiv choose "Relevance", not its default newest-first; a source that offers no
   relevance order is taken in its default order, named in the `notes` of its searches).
2. Enter the three already-known works of Table C.1 first in `search_record.csv`, one row each, with
   `known_work` set to its label in `protocol.KNOWN_WORKS` and every other column filled from your own
   reading (Table C.1 "Record": what it varies, what it evaluates, whether it meets the inclusion
   criterion; `exclusion_criterion`: the Table C.1 "Exclusion" clause the work falls under, one of
   `protocol.EXCLUSION_CRITERIA`, or `none`).
3. Run every search: each query variant in each source (`python -m studyb.search queries`; 20 strings
   in 4 sources, each used verbatim, quotes kept: Q-search-queries), its results taken in the order
   fixed in step 1. Log each in `search_log.csv` (the query string exactly as the protocol gives it;
   describe any source-specific syntax in `notes`; `results_reported` as the source shows it,
   `results_screened` = min(50, `results_reported`); where the source exports its results, commit the
   export in this directory and name it in `export_file`), and list every screened result in its file
   under `screened/` (Q-search-record-scope: every screened result is a hit). Fill `url_or_doi`
   (`none` if the source gives neither) and `reason` on every row: an excluded result's reason names
   the element of the inclusion criterion it fails (no variation of budget levels, or no unseen budget
   evaluated) or the Table C.1 "Exclusion" clause it falls under; a duplicate's names its first
   screened occurrence, whose exact citation text also goes in `duplicate_of`.
4. Follow the citations (Q-search-citations), screening every list in full at title and abstract, with
   no depth limit: (1) the reference list of Yao et al. (2023); (2) its "cited by" lists in Google
   Scholar and Semantic Scholar; (3) for every work in those two lists (deduplicated, as of the search
   date; excluded and duplicate items included, a duplicate standing for the work its `duplicate_of`
   names), its reference list and its "cited by" lists in Google Scholar and Semantic Scholar, each
   with `of_work` = that work's citation text; if Google Scholar refuses access to one, screen its
   Semantic Scholar list alone and write `Google Scholar refused` in that row's `notes`; (4) the
   reference list of every other work you take to full text, a known work of Table C.1 included (for
   Yao et al. that is the first list). Log each list in `citations_log.csv` (`items_reported`: the
   list's length as the source shows it; `items_screened`: the same number; `export_file` as in step
   3) and its screened items in `screened/citations__<list_id>.csv`.
5. Screen in two stages: title and abstract (every result, in `screened/`, its `stage1_decision`),
   then the full text of anything that might meet the inclusion criterion. Every work that reaches
   the full text gets a row of `search_record.csv`, whatever it shows, with the reason for its
   classification and the exclusion clause that applies (`none` if none does; never on a `yes` row).
6. Apply the decision rule (Table C.1 "Decision") in `decision.json`: `{"outcome": "none" |
   "partial" | "included", "amendment": "<id>" or null, "date": "YYYY-MM-DD", "searcher":
   "<name>"}`, dated on or after every search and every classification. A partial overlap or an
   included work is recorded by amendment: `amendment` is the `id` of the `analysis/AMENDMENTS.json`
   entry that records it (a Table 9.1 row ratified by the group), exactly as written there; it is null
   for "none" (Q-search-before-pilot). State the outcome in one sentence at the end of `search_log.md`.
7. Run `python -m studyb.search check` until it reports `complete: True`, commit the record, report
   the outcome to the group, and give the path to the pilot owner for Table 9.0, all before the
   pilot's Study B runs start.

The record reports what each paper does; the group decides what that means for Study B (First
Tasks, "Mistakes to avoid": no novelty claims in the record, every full-text hit recorded, preprints
included, no stopping at the first page).

## Questions answered in Table 9.1 (documented, not PENDING keys)

* Q-search-depth: the first 50 results of each of the 80 searches, in each source's relevance order
  (`SCREENING_DEPTH`, `RESULT_ORDER`; step 1); citation lists have no depth limit (step 4).
* Q-search-queries: four variants of each query (as written, with "number of thresholds", with
  "spacing", with both), the superset of every reading: 20 strings x 4 sources = 80 searches.
* Q-search-citations: the lists (1) to (4) of step 4, each screened in full; a duplicate names its
  first screened occurrence in `duplicate_of`.
* Q-search-record-scope: every screened result is a hit, listed in `screened/` with its citation,
  URL or DOI, title-and-abstract decision and reason (step 3); every full-text work and the three
  known works have a row with all Table C.1 fields in `search_record.csv`; results beyond the
  screening depth are counted in `results_reported` but not listed.

Q-search-before-pilot (a PENDING key of configs/registered.py) is answered in Table 9.1: the pilot's
two Study B runs (the unconstrained PPO run and the Moderate run) are Study B's first runs, so the
record is complete and committed here, and its path entered in Table 9.0, before either starts.
Study A's pilot runs do not wait for the search, and neither do the Table 3.1 determinism checks of
plug-ins `study_b` and `unconstrained_ppo` (software verification, not Study B runs:
scripts/determinism_check.py, outside the scheduler's gate). If the outcome is "partial", the
amendment that narrows Study B's claims is ratified and committed (a Table 9.1 row with an
`analysis/AMENDMENTS.json` entry whose `id` is decision.json's `amendment`) before those runs start;
the scheduler holds them until it finds that entry in the `analysis/AMENDMENTS.json` committed with
the record. If the outcome is "included", Study B is withdrawn by amendment and none of its runs,
pilot runs included, is launched.

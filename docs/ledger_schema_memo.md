# Memo: ledger schema design decisions

| Field | Value |
|---|---|
| To | The group |
| From | Role 4, Analysis and results owner (Muhammad Abdullah) |
| Date | 2026-09-26 (freeze of version 1); corrected 2026-10-01 and 2026-10-02 (see "Changes since the freeze") |
| Corrections | The "Changes since the freeze" section and the matching edits to this memo were made by the pilot owner (Role 1), not by Role 4: LS-1 to LS-32 on 2026-10-01, decided on 2026-10-02 (X-ledger-schema-amendment, `docs/DECISIONS.md`), and LS-33 to LS-36 on 2026-10-02, which follow that day's decisions; they are to be ratified with the other decisions and approved by Role 4's pull request before the first pilot ledger row is written |
| Re | Frozen decisions for the results ledger schema, version 1 |

Six design decisions, a list of divergences from Table B.1, and six
implementation notes (A to F). The code is `results/ledger_schema.py`; its
tests are `tests/test_ledger_schema.py`.

## Design decisions

1. **Storage format: Parquet.**
   Columnar, supports nested types natively, readable by pandas, polars,
   PyArrow and DuckDB, and self-describing. The pilot owner writes with
   PyArrow; the analysis reads with PyArrow or pandas.

2. **Nested data: structs and lists inside the row.**
   The ledger has one row per run. Table B.1 says only that `checkpoints`
   holds "Paths and steps"; this schema nests the checkpoints as a list of
   structs inside the row, and the Study B satisfaction dictionaries are
   Parquet maps. There are no separate tables joined by `run_id`. First
   Tasks proposes a second, checkpoint table instead; the nesting is a
   choice to be recorded in the amendment (Q-checkpoint-table).

3. **Enrichment fields: same row, nullable initially.**
   The ledger is written at the end of each run with the raw fields. The
   matching rule and the battery evaluations fill in the enrichment fields
   later. Raw fields, and `notes`, are immutable after the row is written:
   `write_enrichment` rejects any field not in `ENRICHMENT_FIELDS`, and
   `notes` is not one of them.

4. **Ledger location: `results/ledger.parquet` in the repository.**
   Checkpoints live at `/data/checkpoints/`, outside the repository. The
   ledger stores checkpoint paths relative to `/data/checkpoints/`, so the
   ledger is portable across machines (a path outside that root, or one
   already relative, is stored in normalised POSIX form, for example
   `/other//x.pt/` as `/other/x.pt`).

5. **Schema versioning: a `schema_version` field, frozen at 1.**
   Every row records the version it was written under. The field is typed
   `Literal[1]`, so any other version (and a boolean) is rejected. A change
   to the stored layout bumps the version, and so does a change that would
   make a row already in a ledger invalid; every change is an amendment.

6. **`lambda_at_selection`: the most recent epoch at or before the
   checkpoint step.**
   The multiplier is logged once per epoch (20,000 steps); a checkpoint is
   saved every ten epochs (200,000 steps). The value used is the one in the
   `progress.csv` row of the last epoch that ends at or before the
   checkpoint step; that row holds the state after the epoch
   (`pilot/rundir.py`, `EPOCH_COLUMN`), so at a checkpoint on an epoch
   boundary it is the value the next epoch starts from. The per-checkpoint
   multiplier (`checkpoints[].multiplier`) is read from `progress.csv` in
   `pilot/ledger_writer.py` (`build_row_fields`), and `lambda_at_selection`
   is copied from the matched checkpoint's multiplier in
   `pilot/enrichment.py`; the schema checks that the two are equal (LS-27).

## Divergences from Table B.1

To be recorded in the amendment with this memo.

- **Renamed.** Table B.1's `matched_checkpoint` is stored as
  `matched_checkpoint_path` and `matched_checkpoint_step`; its
  `selection_cost` is `selection_cost_at_match` (the per-checkpoint
  selection cost is `checkpoints[].selection_cost`).
- **Split.** Table B.1's `study; arm` is stored as `study`, `task`, `arm`,
  `N`, `onset_shape`, `step_matching`, `treatment`, `controller_variant`
  and `training_levels`; its `completed` with "the cause" is stored as
  `completed` and `failure_cause`.
- **Added.** `schema_version`; per-checkpoint training cost and return,
  plasticity metrics, selection cost and return, and multiplier;
  `multiplier_trace_path`; `matched` and `infeasible` (Part 4.1,
  Feasibility and Tolerance);
  `per_level_multipliers` (Study B).
- **Nesting.** Checkpoints are nested in the row (decision 2), not a second
  table.
- **Two-index map.** Table B.1's `sr_fewshot[budget, horizon]` is stored as
  one map with the string key `"budget_horizon"` in the canonical spelling
  `"5.0_200000"` (`pilot.contracts.fewshot_key`; note A).
- **Not validated here.** Table B.1's `seed` is an "Integer from the fixed
  list"; the schema accepts any non-negative int64 (the run specifications
  of `pilot/manifest.py` take their seeds from `configs/registered.py`).
  First Tasks Role 4 step 1 asks for a commit-hash length check; the schema
  requires only a non-empty `commit_hash`, and `pilot/ledger_writer.py`
  enforces a 40-character hexadecimal hash.

## Implementation notes

**A. Map keys.**
Parquet and PyArrow can store float64 map keys. This schema stores string
keys on disk by design: the spelling of a float key is then fixed (`"5.0"`),
and the composite `sr_fewshot` key (`"5.0_200000"`) has the same map type.
The float-keyed fields (`sr_zero`, `adapt_steps`, `per_level_multipliers`)
are converted back to float keys by the model.

**B. Enrichment write path.**
`write_enrichment` checks that the existing file's schema is
`LEDGER_SCHEMA`, merges the enrichment into the target row and validates
the merged row as a `LedgerRow` (every field and cross-field constraint;
map keys are canonicalised). It then rebuilds the table with the frozen
schema through the same serialisation path as `append_to_ledger`
(`_to_parquet_record` and `_to_table`), so Arrow type errors are also
caught, and replaces the ledger atomically. Like `load_ledger_as_rows`, it
refuses a ledger with a duplicate run_id, and an invalid stored target row
is reported by its run_id. Tests
verify schema preservation, that raw fields and `notes` cannot be modified
through this path, and that a refused enrichment leaves the file unchanged.

**C. Write concurrency and crash safety.**
Every write (append and enrichment) reads the whole file and rewrites it.
The rewrite goes through a temporary file in the same directory, which is
written and fsynced through one writable handle (Windows' fsync needs write
access), given the ledger's permission bits (those of the existing ledger,
or 0o666 less the umask for a new one, not `mkstemp`'s 0o600), and then
moved over the ledger with `os.replace`, so a crash cannot truncate the
ledger. The umask is read from `/proc/self/status` on Linux; elsewhere it
is read with `os.umask`, which changes the process-wide umask for a moment
and so is not thread-safe. Concurrent writers would still lose rows: the
pilot owner is the sole writer, and every caller wraps its writes in
`pilot.ledger_writer.ledger_transaction`, which holds the ledger's lock.

**D. Checkpoint cadence.**
Checkpoints are saved every 200,000 steps, at onset, and at the end of
training. Onset at N = 0.25 lands on step 2,500,000 (total-steps matched)
or 3,340,000 (constrained-steps matched, Q-rounding), neither a multiple of
200,000. The schema stores any non-negative integer step; the steps of a
row must be strictly increasing.

**E. Parquet map shape normalisation.**
PyArrow's `to_pylist` returns map columns as either a dict or a list of
(key, value) tuples, depending on the version and code path. A single
`mode="before"` validator on `LedgerRow` normalises both shapes to a dict
on every construction (not only on read), and applies float-key coercion
only to the float-keyed fields, using `ValidationInfo.field_name` to tell
them apart. This covers `sr_zero`, `sr_fewshot`, `adapt_steps` and
`per_level_multipliers`. A list whose elements are not tuples or lists of
length 2 (a dict of two entries is not a pair), an unhashable key, a key
repeated in a list of pairs, two keys that are equal after coercion (`"10"`
and `"10.0"`), a boolean key, a bytes key, and a string key holding `_` or
surrounding whitespace (`float` would read `"1_5"` as 15.0) are validation
errors.

**F. Path portability.**
Checkpoint paths stored in the ledger are POSIX strings, not host-native
paths. `relative_checkpoint_path` and `absolute_checkpoint_path` use
`PurePosixPath`, which is operating-system-independent. A stored path may
not be empty or whitespace only, have surrounding whitespace, contain a NUL
character, a backslash or a `..` component, name no file (`.`, `/`), or
differ from its normal form (`a//b.pt`, `./a.pt`, `a/b/`), so that one file
has one spelling. Both helpers refuse a `..`
component and an empty path, `.` or `/`, and `relative_checkpoint_path`
also refuses `CHECKPOINT_ROOT` itself; a path outside the root is returned
in normalised POSIX form. Local filesystem operations on the ledger file
itself use `Path`.

## Testing

`tests/test_ledger_schema.py` has 126 test functions (312 test cases with
the parametrised ones). They cover:

- round trips with whole-row equality (Study A, Study B, a failed run) and
  a NaN training cost at step 0, nested checkpoints, onset at a non-multiple
  step, float keys, string `sr_fewshot` keys, and `load_ledger`'s dict
  normalisation;
- every categorical field (study, task, onset shape, step matching,
  treatment, controller, N, failure cause, Study B arm, a Study A arm with
  a Study B name) and empty identifiers;
- completion consistency (both directions, `wall_clock_hours`,
  `failure_cause`), `finished` before `started`, naive and non-UTC
  datetimes, zero-offset normalisation, and nanosecond timestamps;
- booleans and strings refused as numbers, non-boolean flags and numeric
  datetimes refused, `schema_version`, NaN and ±inf refused (row fields;
  NaN, +inf and -inf in four checkpoint fields; NaN and ±inf in the values
  of `sr_zero`, `sr_fewshot` and `per_level_multipliers`; NaN and +inf in
  the training levels), ±inf refused in the training statistics, int64
  overflow and fractional integers;
- the study-specific fields (Study A needs N; neither study may carry the
  other's fields, Study A's enrichment fields included; Study B's training
  levels are its arm's, none for Continuous; per-level multiplier keys and
  the Continuous arm's bin edges, 10 to 35);
- the maps (unseen budgets, `adapt_steps` keys and values (a horizon or the
  largest horizon + 1, following from `sr_fewshot`), rates bounded to
  [0, 1], `sr_fewshot` horizon, format, budget and canonical spelling,
  string keys, list-of-tuples, malformed lists (a list holding a dict
  included), unhashable, colliding and repeated keys, boolean and bytes
  keys, keys spelled with `_` or spaces);
- checkpoints and matching (increasing steps, matched step and path,
  `training_age`, matched and infeasible, `matched`, the selection
  quantities and the gaps need a matched checkpoint, which lies in the
  selection window and supplies `selection_cost_at_match` and
  `lambda_at_selection`, a failed run carries no matching result) and
  stored paths (normal form, whitespace, NUL);
- the writer (schema version, duplicates leave the file unchanged, a row
  mutated after construction, a refused row creates no directory, a schema
  mismatch, atomic writes, the file mode, the umask read without changing
  it), the PyArrow-to-Pydantic field parity (row and checkpoint struct)
  and a fully populated round trip (a Study A row with every Study A field
  and a Study B row with every map);
- the loaders (missing file, missing column, changed column type, duplicate
  run_ids, an invalid row named by its run_id);
- enrichment (nullable fields, raw-field and `notes` protection, schema
  preservation, unknown run_id, missing file, schema mismatch, fifteen
  invalid enrichments that leave the file unchanged, Study A fields on a
  Study B row, a failed run, duplicate run_ids, an invalid stored row named
  by its run_id, key canonicalisation, list-shaped maps, the input not
  mutated, other rows unchanged);
- the path helpers, the arm levels and the other registered values
  (tasks, onset fractions, unseen budgets, few-shot horizons, the
  satisfaction target) against `configs/registered.py`, and the recorded
  hash below.

## Changes since the freeze (decided 2026-10-02, X-ledger-schema-amendment; to be ratified with the other decisions)

Corrections made on 2026-10-01 by the pilot owner (Role 1), before the
pilot wrote any ledger row, and decided on 2026-10-02 as one amendment
(X-ledger-schema-amendment, `docs/DECISIONS.md`), to be ratified with the
other decisions and approved by Role 4's pull request before the first
pilot ledger row is written; until then the frozen version 1 of PR #1
stays the registered one. None changes the Parquet layout
(`LEDGER_SCHEMA`), the field names, or the public functions and their
signatures. The rows the pipeline writes (`pilot/ledger_writer.py`,
`pilot/enrichment.py`) already meet them, as the pipeline's own tests show,
and no ledger exists yet, so the version stays 1.

1. **LS-1. Recorded hash.** `results/ledger_schema.sha256` held the hash of
   a CRLF copy of the file in a UTF-16 file that `sha256sum -c` cannot
   read. It is now UTF-8 with one LF-terminated line,
   `<hex>  results/ledger_schema.py`, re-recorded for the corrected file.
2. **LS-2. `write_enrichment` validates.** The merged row is validated as a
   `LedgerRow` before anything is written, so an out-of-range or ill-typed
   value is refused (a `pydantic.ValidationError`, which is a
   `ValueError`) instead of being truncated (8400000.5 to 8400000),
   converted (True to 1.0) or written and then breaking every later load.
   Map keys written by an enrichment are canonical (`"5.0"`), and a
   list-of-pairs map is accepted, as on read.
3. **LS-3. `notes` is not an enrichment field.** It holds the writer's
   provenance and can no longer be replaced or set to None by an
   enrichment.
4. **LS-4. Finite numbers.** NaN and ±inf are refused in every float
   field, in the map values and in the training levels, except a
   checkpoint's `training_cost` and `training_return`. The two training
   statistics may be NaN wherever the epoch has no finite statistic (the untrained step-0
   checkpoint, a checkpoint without a progress row, a failed run), but not
   ±inf (LS-20).
5. **LS-5. No booleans or strings as numbers.** `seed=True`, `N=False`,
   `schema_version=True`, a boolean checkpoint step and numeric strings
   such as `final_cost='nan'` are refused; integers are bounded by int64,
   so an overflow is a validation error, not a raw `OverflowError`.
6. **LS-6. Study B maps.** `sr_fewshot` budgets must be unseen budgets and
   its keys must be in the canonical spelling `f"{float(budget)}_{horizon}"`
   (`pilot.contracts.fewshot_key`, `"5.0_200000"`); `sr_zero` and
   `sr_fewshot` rates lie in [0, 1]; `adapt_steps` values are non-negative
   (LS-34 adds the encoding Q-adapt-censoring decided);
   `per_level_multipliers` keys are finite and, when the arm has fixed
   training levels, among them.
7. **LS-7. Map normalisation.** A malformed list is a validation error, not
   a raw `TypeError`; keys equal after float coercion (`"10"` and `"10.0"`)
   and boolean keys are refused instead of silently merged.
8. **LS-8. Cross-field checks.** Study A rows need N and carry no Study B
   field; Study B rows carry no N, onset shape, step matching, treatment or
   controller variant, and need training levels except the Continuous arm;
   `finished` is not before `started`; checkpoint steps strictly increase;
   `matched_checkpoint_path` and `matched_checkpoint_step` are set together
   and name one of the row's checkpoints; `training_age` equals
   `matched_checkpoint_step` (Table B.1: "its step index"); `matched` and
   `infeasible` are not both True; `run_id`, `arm`, `machine`,
   `commit_hash` and `config_hash` are not empty.
9. **LS-9. Datetimes.** A zone with a zero UTC offset is accepted and
   normalised to UTC (the error message says so).
10. **LS-10. Writes re-validate and are atomic.** `append_to_ledger`
    validates the row again (a row changed after construction is caught),
    and both write functions replace the ledger through a temporary file and
    `os.replace`.
11. **LS-11. Loaders check the file.** `load_ledger` and
    `load_ledger_as_rows` refuse a file whose schema is not `LEDGER_SCHEMA`;
    `load_ledger_as_rows` refuses duplicate run_ids and names the run_id of
    an invalid row (a `ValueError` whose cause is the validation error).
12. **LS-12. Paths.** Stored paths may not be empty, contain a backslash or a
    `..` component; `relative_checkpoint_path` refuses a `..` component and
    the root itself.
13. **LS-13. Requirements pinned.** `results/ledger_schema_requirements.txt`
    pins the versions of `environment/requirements.in` (numpy 1.26.4,
    pandas 2.0.3, pyarrow 25.0.1, pydantic 2.13.5, pytest 9.1.1).
14. **LS-14. Tests.** The fixture's selection window is the last ten
    checkpoints, (8M, 10M], not eleven; its run_id and arm follow N; its
    multiplier before onset is the initial 0.001; Study B's `sr_fewshot`
    covers all twelve cells; and the tests above were added.
15. **LS-15. Documentation.** The whole memo was rewritten (Markdown
    heading structure, counts, date and author, the metadata table, every
    design decision and implementation note, the new "Divergences from
    Table B.1", "Testing" and "Freeze and hash" sections and this list);
    the substantive changes are notes A to F, decisions 2 to 6 and the
    divergences. The module docstring (map keys as a design choice, one row
    per launched run, the freeze rule) was corrected.

Second round of corrections (2026-10-01, after a review of the first;
part of the amendment decided on 2026-10-02). The Parquet layout, field
names, public functions and `SCHEMA_VERSION` 1 are unchanged; the rows the
pipeline writes meet them (its tests pass), and no ledger row exists yet.

16. **LS-16. Ledger file mode and Windows fsync.** The atomic write (LS-10)
    left every ledger at `mkstemp`'s mode 0o600, unreadable to other
    accounts; the rewritten ledger now keeps the existing ledger's mode, or
    0o666 less the umask for a new one. The temporary file is written and
    fsynced through one writable handle, since Windows' fsync needs write
    access.
17. **LS-17. Study B rows carry no Study A field.** The fields Table B.1
    marks "(Study A)" (`matched_checkpoint_*`, `training_age`,
    `selection_cost_at_match`, `measurement_cost`, `lambda_at_selection`,
    `gap_*`, `dormant_onset`, `rank_onset`, `norm_onset`, `lambda_peak`,
    `lambda_final`, `settling_steps`) and the added `matched` and
    `infeasible` (Part 4.1) are refused in a Study B row (`STUDY_A_ONLY_FIELDS`), so an enrichment
    cannot write them there.
18. **LS-18. Training levels follow the arm.** A Study B row's
    `training_levels`, sorted, equal its arm's set of Table 2.5
    (`ARM_TRAINING_LEVELS`, the same as `configs/registered.py`
    `STUDY_B_ARMS`): no wrong, duplicated or negative levels; the
    Continuous arm has none, and its `per_level_multipliers` keys are lower
    edges of the 5-unit bins of [10, 40]: 10, 15, ..., 35 (LS-33;
    Q-continuous-bins).
19. **LS-19. Matching fields tied together.** `matched = True` needs a
    matched checkpoint; `training_age`, `selection_cost_at_match`,
    `measurement_cost` and `lambda_at_selection` are set only with one; a
    failed run (`completed = False`, excluded and repeated with the next
    seed, Part 5) carries no matched checkpoint, selection quantity,
    `matched`, `infeasible`, `gap_*`, `sr_zero`, `sr_fewshot` or
    `adapt_steps`, so it cannot enter the matched comparison.
20. **LS-20. Training statistics: NaN, not ±inf.** A checkpoint's
    `training_cost` and `training_return` refuse ±inf (LS-4).
    `pilot/ledger_writer.py` (`build_row_fields`) now records an infinite
    value from `progress.csv` as NaN.
21. **LS-21. Strict booleans and datetimes.** `completed`, `matched` and
    `infeasible` accept only booleans (`'yes'`, `1` and `'false'` are
    refused; a NumPy boolean is accepted); `started` and `finished` accept
    a datetime or an ISO 8601 string, not a number (which Pydantic would
    read as seconds since 1970).
22. **LS-22. Degenerate paths.** Stored paths that name no file (`.`, `/`,
    whitespace only) are refused; `absolute_checkpoint_path` refuses a `..`
    component and an empty path, `.` or `/`, as `relative_checkpoint_path`
    now also does (it returned `.` for an empty path).
23. **LS-23. Float map key spelling.** A string key holding `_` or
    surrounding whitespace is refused before `float` reads it (`"1_5"` was
    read as 15.0).
24. **LS-24. `write_enrichment` checks the ledger as the loader does.** It
    refuses a ledger with a duplicate run_id (it enriched only the first
    copy), names the run_id of an invalid stored target row, and rebuilds
    the table through `_to_table`, as `append_to_ledger` does.
25. **LS-25. Documentation and requirements.** Docstrings
    (`_normalise_map`, `_revalidated`, `write_enrichment`,
    `relative_checkpoint_path`), this memo (metadata table, decisions 4 and
    6, notes B, C, E and F, the divergences: `seed`, the commit-hash length
    and the two-index `sr_fewshot`), and
    `results/ledger_schema_requirements.txt` (Python 3.10, as
    `pyproject.toml` requires).
26. **LS-26. Tests.** The fixture's onset step follows N; the rejection
    tests name the expected error (`match=`); the coercion-collision test
    uses `[("5", …), ("5.0", …)]` and a repeated key has its own test; the
    run_id is matched literally (`re.escape`); the full round trip covers a
    Study A row with every Study A field (checkpoint structs included) and
    a Study B row with every map; and LS-16 to LS-24 are tested.

Third round of corrections (2026-10-01, after a review of the second, by
the pilot owner, Role 1; part of the amendment decided on 2026-10-02). The Parquet layout, field names, public functions and
`SCHEMA_VERSION` 1 are unchanged. The rows the pipeline writes meet them
(`pilot/ledger_writer.py` writes paths through `relative_checkpoint_path`,
and `pilot/enrichment.selection_fields` already writes the gaps only with a
matched checkpoint of the window and copies its selection cost and
multiplier; the pipeline's tests pass), and no ledger row exists yet, so
the version stays 1.

27. **LS-27. Matching fields tied to the matched checkpoint.** The four
    `gap_*` fields ("Robustness gaps of the matched checkpoint", Table B.1)
    are set only with a matched checkpoint, as the selection quantities are
    (`SELECTION_FIELDS`). The matched checkpoint must carry a
    `selection_cost`, so that it lies in the selection window (Part 4.1
    rule 1: one of the last ten checkpoints). `selection_cost_at_match`, when
    set, equals that checkpoint's `selection_cost`, and
    `lambda_at_selection`, when set, equals its `multiplier` (decision 6).
28. **LS-28. Map normalisation, completed.** In a list-shaped map, each
    element must be a tuple or list of length 2 (a list holding one dict of
    two entries was read as the pair of its two keys); an unhashable key is
    a validation error, not a raw `TypeError` (LS-7); a bytes key is refused
    in every map (`float(b"1_5")` is 15.0, past LS-23).
29. **LS-29. Stored paths in normal form.** A stored path must equal its
    `PurePosixPath` normal form (no `a//b.pt`, `./a.pt` or `a/b/`), have no
    surrounding whitespace and contain no NUL character, so that one file
    has one spelling (`matched_checkpoint_path` is compared with the
    checkpoint's path as a string) and every stored path can be opened.
30. **LS-30. Datetimes at microsecond precision.** A pandas `Timestamp`
    with nanoseconds is refused (the `timestamp('us')` column would
    truncate it, so the stored row would not be the validated one), and a
    `Timestamp` is stored as a plain `datetime`.
31. **LS-31. Writer details.** `append_to_ledger` creates the ledger's
    directory only after the row is accepted; the umask for a new ledger's
    mode is read from `/proc/self/status` where it exists, without the
    process-wide `os.umask` round trip (the fallback elsewhere is documented
    as not thread-safe); an unused import was removed and one line shortened.
32. **LS-32. Tests and memo.** Every rejection test names its expected error
    (`match=`), the enrichment cases included; the non-finite cases test
    what they claim (a level set the arm accepts, NaN and ±inf for
    `sr_fewshot` and the checkpoint fields); the accepted-path case has its
    own test; LS-27 to LS-31 are tested. In this memo: the corrections'
    attribution (metadata table), decision 4's example, the citation for
    `matched` and `infeasible`, the wording of LS-4 and the "Testing"
    section.

Callers changed with these corrections: `tests/test_analysis_cli.py`
(`test_data_the_analysis_cannot_read_is_a_refusal_not_a_traceback` now puts
its non-canonical key into the Parquet file directly, since the schema
refuses it on write), and docstrings only in `pilot/enrichment.py`,
`analysis/data.py` and `tests/test_enrichment.py`. With LS-16 to LS-26:
`pilot/ledger_writer.py` (an infinite training statistic is written as NaN,
LS-20) and the comment above the ledger pins in
`environment/requirements.in` (no stale test count). With LS-27 to LS-32:
none.

Fourth round of corrections (2026-10-02, by the pilot owner, Role 1, after
a review against the decisions of that day, before any ledger row was
written). They bring the schema in line with Q-continuous-bins and
Q-adapt-censoring and belong to the same amendment: when the group ratifies
X-ledger-schema-amendment, it names LS-1 to LS-36 and the hash below. The
Parquet layout, field names, public functions and `SCHEMA_VERSION` 1 are
unchanged; the rows the pipeline writes meet them (`pilot/ledger_writer.py`
reads the Continuous arm's multipliers from the progress columns that
`studyb.conditioning` names by the bins' lower edges, 10 to 35, and
`pilot/enrichment.py` writes a budget's `adapt_steps` with its `sr_fewshot`
rates, by the same rule), and no ledger row exists yet, so the version
stays 1.

33. **LS-33. The Continuous arm's six bins.** Q-continuous-bins decided six
    bins, [10, 15), ..., [35, 40], the last one closed and each named by its
    lower edge, so the Continuous arm's `per_level_multipliers` keys are 10,
    15, ..., 35; 40, which names no bin, is refused (it was accepted while
    the question was open).
34. **LS-34. Adaptation steps follow Q-adapt-censoring.** An `adapt_steps`
    value is a few-shot horizon (200,000, 500,000 or 1,000,000) or the
    largest horizon + 1 (1,000,001, or 200,001 under Part 6.1 cut 3), and
    where `sr_fewshot` holds a budget's horizons, its `adapt_steps` is the
    smallest of them whose rate is at least 0.80 (`SATISFACTION_TARGET`, the
    value of `configs/registered.py`), else the largest + 1 (Table 2.5
    "Adaptation steps": "recorded as above the largest horizon if never
    reached").
35. **LS-35. Two leading slashes.** `PurePosixPath` keeps exactly two
    leading slashes, so `//data/checkpoints/x.pt` passed as a stored path
    in normal form, a second spelling of `/data/checkpoints/x.pt`, and
    `relative_checkpoint_path` returned it unchanged. A stored path with two
    leading slashes is refused, and so is such a path in both helpers.
36. **LS-36. Documentation and tests.** The header comment (the
    amendment's status), the docstring's onset example (3,340,000 under
    constrained-steps matching, Q-rounding), and the comment and message of
    `STUDY_A_ONLY_FIELDS` (`matched` and `infeasible` are added fields, Part
    4.1, not Table B.1's). The test fixture's few-shot rates now give its
    `adapt_steps` (its rates were 0.5 at every horizon, yet its
    `adapt_steps` recorded the target as reached).
    LS-33 to LS-35 are tested, the schema's copies of registered values
    (tasks, onset fractions, unseen budgets, few-shot horizons, arms, the
    satisfaction target) are checked against `configs/registered.py`, and
    the non-finite map values the "Testing" section names are all tested
    (`sr_zero` -inf, `sr_fewshot` +inf and `per_level_multipliers` -inf were
    missing). In this memo: the metadata table, the quotation of Table B.1
    ("Paths and steps"), note D, the "Testing" section, LS-6, LS-17, LS-18
    and the status of each round.

Callers changed with LS-33 to LS-36: none.

## Freeze and hash

Frozen at schema_version = 1 on 2026-09-26; corrected 2026-10-01 (LS-1 to
LS-32, decided 2026-10-02 as X-ledger-schema-amendment) and 2026-10-02
(LS-33 to LS-36), to be ratified with the other decisions (see "Changes
since the freeze").

`sha256(results/ledger_schema.py) = f31b832bbe4802fe82f90b55070c3ef3b5885e23bfebabe0eedc03e3c4a017b9`

This must equal `results/ledger_schema.sha256`; check it from the
repository root with `sha256sum -c results/ledger_schema.sha256`.

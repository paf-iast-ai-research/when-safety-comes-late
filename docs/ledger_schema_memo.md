MEMO — Ledger Schema Design Decisions
To: The group
From: Analysis and Results owner
Date: [date of freeze]
Re: Frozen decisions for the results ledger schema, version 1

Six design decisions, plus four implementation notes.

1. STORAGE FORMAT: PARQUET.
   Columnar, supports nested types natively, readable by pandas, polars,
   PyArrow, and DuckDB, and self-describing. The pilot owner writes with
   PyArrow; the analysis reads with PyArrow or pandas.

2. NESTED DATA: STRUCTS AND LISTS INSIDE THE ROW.
   "One row per run" is the pre-registration's requirement. Nested
   checkpoints are a list of structs inside the row. Satisfaction
   dictionaries are Parquet maps. No separate tables joined by run_id.

3. ENRICHMENT FIELDS: SAME ROW, NULLABLE INITIALLY.
   The ledger is written at run completion with raw fields. The matching
   rule and battery evaluations fill in enrichment fields later. Raw fields
   are immutable after the run completes; write_enrichment rejects any
   field not in ENRICHMENT_FIELDS.

4. LEDGER LOCATION: results/ledger.parquet IN THE REPOSITORY.
   Checkpoints live at /data/checkpoints/ outside the repository. The
   ledger stores checkpoint paths relative to /data/checkpoints/ so the
   ledger is portable across machines.

5. SCHEMA VERSIONING: schema_version FIELD, FROZEN AT 1.
   If the schema changes after the pilot, every row is invalid. The version
   is recorded in every row and typed as Literal[1] so it cannot be
   overridden at construction. Any change is an amendment.

6. lambda_at_selection: MOST RECENT EPOCH AT OR BEFORE THE CHECKPOINT STEP.
   The multiplier is logged per epoch (20,000 steps). The checkpoint is
   saved per ten epochs (200,000 steps). The most recent epoch value is the
   one the run actually used at the checkpoint step.

IMPLEMENTATION NOTES.

A. Parquet map key limitation.
   Parquet MAP keys must be STRING. Float-keyed fields (sr_zero,
   adapt_steps, per_level_multipliers) use string keys on disk. A
   mode="before" validator on LedgerRow normalises keys to float and
   accepts both dicts and lists of (key, value) tuples.

B. Enrichment write path.
   write_enrichment reads the whole table to Python, replaces the target
   row, and rebuilds via pa.Table.from_pylist with the frozen schema.
   Routes through a single serialization path, preserves the schema
   byte-for-byte, and refuses to proceed if the rebuilt table does not
   match. Tests verify schema preservation and that raw fields cannot be
   modified through this path.

C. Write concurrency.
   The pilot owner is the sole writer. Concurrent appends would race on
   the read-concat-rewrite of Parquet and are not supported.

D. Checkpoint cadence.
   Checkpoints are saved every 200,000 steps, at onset, and at the end of
   training. Onset at N=0.25 lands on step 2,500,000, which is not a
   multiple of 200,000. The schema stores any non-negative integer step.
E. Parquet map shape normalisation.
   PyArrow's to_pylist returns map columns as either a dict or a list of
   (key, value) tuples depending on the version and code path. A single
   mode="before" validator on LedgerRow normalises both shapes to dict and
   applies float-key coercion only to the float-keyed fields, using
   ValidationInfo.field_name to distinguish. This covers sr_zero,
   sr_fewshot, adapt_steps, and per_level_multipliers.

F. Path portability.
   Checkpoint paths stored in the ledger are POSIX strings, not host-native
   paths. relative_checkpoint_path and absolute_checkpoint_path use
   PurePosixPath, which is operating-system-independent. Local filesystem
   operations on the ledger file itself continue to use Path.
TESTING.
   Twenty-two tests: round-trip, nested checkpoints, onset at a
   non-multiple step, float-key preservation, string-key preservation for
   sr_fewshot, invalid study, invalid failure cause, invalid Study B arm,
   completed consistency (both directions), invalid unseen-budget key,
   invalid sr_fewshot horizon, invalid sr_fewshot key format, naive-
   datetime rejection, string-key coercion, list-of-tuples coercion,
   schema version, append with duplicate detection, PyArrow-to-Pydantic
   field parity, nullable enrichment, raw-field protection, schema
   preservation after enrichment.

FROZEN at schema_version = 1 on 2026-09-26.
sha256(ledger_schema.py) = <44cf76c2d49aaac41aa00fe60cf5ce50f72a4da05ef13a8f1d8fa640e6465495 *results/ledger_schema.py>



# HANDOVER: Pilot owner (Role 1)

Prepared 2026-09-27, updated 2026-09-28, for Muhammad Umair Waseem, pilot owner. This is the
single guide to the pilot owner's code and duties. It says what the study is and what is wrong or
open in the plan today. It gives the ordered list of work from now to the paper, what every file
is for and how to run it, and what the other roles must deliver to you.

The pre-registration (`prereg/Preregistration.pdf`) governs everything here. Where this document or
the code differs from it, the pre-registration wins. The difference is then a question for the
group, answered in the amendment log (Part 9, Table 9.1) before the code that depends on it runs.

**How this repository reaches you.** It comes as an archive of the complete working tree, with its
`.git` history. Nothing below was committed to `main`. One earlier snapshot was pushed to the
feature branch `claude/research-project-analysis-7fhfza` (commit `b527c84`). After that, as you
asked, nothing more was committed: every later change is in the working tree only.

To bring it in:
1. Unpack the archive.
2. Review with `git status` / `git diff`.
3. Commit on a branch named after your role and component (First Tasks), for example
   `pilot/infrastructure`.
4. Open a pull request.

Delete or reuse the `claude/…` branch as you prefer.

---

## 1. The research in one page

**Study A, When Safety Comes Late.** Question: does a later constraint onset cause a larger
robustness gap, when the policies compared are matched on final in-distribution cost?
- X = onset fraction N ∈ {0, 0.10, 0.25, 0.50}: the point at which the cost term enters the
  PPO-Lagrangian surrogate (eq. 3) and the multiplier starts updating (eq. 4).
- Y = robustness gap = cost under a battery condition minus the in-distribution cost of the same
  matched checkpoint (eq. 1). The battery is hazard relocation, dynamics perturbation (mass ×1.3,
  friction ×0.7), reward-only fine-tuning (1M steps) and transfer Goal↔Button (1M steps).
- Z = matching: in every run, the checkpoint among the last ten whose selection-set cost is
  closest to d = 25 (Part 4.1). Arms are compared only if their mean cost is within 2.5 of the
  N = 0 reference.
- Hypotheses: H1 (later onset gives a larger gap, increasing in N), H2 (abrupt onset gives a
  larger gap than a linear ramp), H3 (plasticity: dormant-neuron fraction and effective rank
  correlate with the gap; reset and injection narrow it while extra constrained training does
  not; manipulation check), H4 (part of the abrupt gap is multiplier overshoot: a warm-started
  multiplier reduces but does not remove it), H0 (the null).
- Design: 13 main arms × 3 tasks × 5 seeds = 195 runs. Add 135 mediation runs, 90
  controller-variant runs and 15 PID runs: **435 runs**. Primary outcome: the hazard-relocation
  gap on SafetyPointGoal1-v0, with H1 under both step-matching controls.

**Study B, Constraint-Coverage Generalization.** Question: for a budget-conditioned PPO-Lagrangian
agent (budget/100 appended to the observation, one multiplier per training level), do the
number and spacing of training budgets determine zero-shot constraint satisfaction on unseen
budgets {5, 15, 30, 45}, and the steps needed to adapt?
- Arms: Single-10/20/40, Sparse {10,40}, Moderate {10,20,40}, Dense {10,17.5,25,32.5,40}, and
  Continuous U[10,40].
- Size: 7 arms × 5 seeds = 35 runs, plus 140 few-shot continuations of 1M steps each.
- Hypotheses G1–G5: coverage helps, coverage beats proximity, stricter is harder, coverage speeds
  adaptation, and a few levels approach the continuous case.
- Primary outcome: mean zero-shot satisfaction; G1 and G2 are tested at α = 0.025.

**Shared rules.**
- Software: OmniSafe PPO-Lagrangian defaults with T = 10M steps (500 epochs of 20,000).
- Hardware: CPU, one torch thread, one environment and one process per run.
- Statistics: the seed is the unit of analysis. The primary interval is Welch's t-interval, with
  a seed bootstrap beside it.
- Exclusions: a run is excluded only if it crashes, is incomplete, or produces a non-finite loss
  or multiplier. It is then repeated with the next unused seed. If H1 or G1 is falsified, the
  result is reported as an upper bound (Part 5.7).

**The pilot (Part 3.6), your responsibility.** Eight runs:
- Study A: SafetyPointGoal1-v0, N = 0 and N = 0.50 (abrupt, total-steps matched), seeds 0, 1, 2,
  with a hazard-relocation and dynamics battery.
- Study B: one unconstrained PPO run and one Moderate run (seed 0).

It measures the [P] values of Table 8.1. The group's go meeting then applies Part 6:
- **G1 feasibility**: both arms at most 27.5, reference SD at most 10, and the arms matched.
- **G2 throughput**: hours per run ÷ concurrent runs × 484 must not exceed the machine-hours
  allocated *before* the pilot.
- **G3 signal**: mean Δ(0.50) under hazard relocation of at least 5, or U80 of at least 5.
- **G4 conditioning**: every budget below the unconstrained cost, and the Moderate costs ordered
  and each at most its budget + 2.5 (Part 6; Part 3.6 says "within 2.5", see Q-g4-level).

Go freezes Parts 1, 4, 5 and 6. A failure of G1, G2 or G4 allows one revision and a re-pilot. If
G1 and G2 hold but G3 fails, Study A is a no-go (a negative pilot) and Study B becomes the main
study.

**Your role (Parts 9 and 10; First Tasks, Role 1).** You own:
- the repository and the registration record;
- the pinned environment and the determinism check;
- the run scheduler (launch, resume, never duplicate) and seed handling;
- the ledger writer (you are its only writer);
- running the pilot and every later run;
- throughput (G2) and the go or no-go report;
- Parts 1 and 6 of the document (hypotheses and go rule).

---

## 2. State of the repository (2026-09-27, verified)

| Item | State |
|---|---|
| Registration commit `735b394…` | Present. It holds only `prereg/Preregistration.{docx,pdf}`, and `prereg/` is unchanged since. |
| Tag `registration` | **Missing** (no tags on GitHub). |
| Table 9.0 / Table 9.1 first row | **Unfilled.** The "second commit" slot planned for it is gone: `main` has six later commits (four direct commits, the PR #1 commit and its merge). |
| Repository visibility | Public, as Table 9.0 requires. |
| `main` protection | **None.** Four commits went straight to `main`, and PR #1 was merged 20 minutes after its commit. |
| Ledger schema (Role 4, PR #1) | In `results/`. Its 25 tests pass on the pinned stack. See §3.3 for its problems. |
| Everything in §5 marked "You" | New or extended in this handover (`.gitignore` came with PR #1). |

---

## 3. Findings you must act on

Everything in this section was checked against the pinned sources (OmniSafe 0.5.0, Safety-Gymnasium
0.4.1) or reproduced by running code. The issue-ready wording of every question is in §9.

### 3.1 Blockers before the pilot

1. **Hazard relocation is not a distribution shift.** This hits the primary outcome and G3.
   - Safety-Gymnasium draws a new random layout of every object at every `reset()`
     (`bases/underlying.py: _build → random_generator.build_layout()`), and OmniSafe seeds the
     environment once.
   - So training already sees the full layout distribution, and "hazard positions re-sampled with
     the task's own layout generator using twenty new layout seeds" (Table 2.2) samples the same
     distribution. For every arm, the expected gap is ≈ 0 up to evaluation noise.
   - Confirmed by running it: hazard 0 moves every episode, and reseeding reproduces episode 0.
   - Also, with a deterministic policy, "20 layouts × 5 episodes" gives 5 identical copies of each
     episode.
   - Consequence: the pilot's G3 would fail by construction, and Study A would be declared a
     no-go for a design reason, not an empirical one.
   - The group must redefine the condition by amendment **before the pilot** (Q-hazard).
2. **The pinned install is broken without a numpy pin.**
   - `pip install omnisafe==0.5.0` on Python 3.10 installs numpy 2.2.6 next to OmniSafe's
     `pandas==2.0.3`, and `import omnisafe` then fails.
   - pip also silently falls back to safety-gymnasium 0.4.1 with mujoco 2.3.0.
   - Fixed in `environment/requirements.in`: `numpy==1.26.4` and explicit pins.
   - Python 3.10 on Linux is required.
3. **Study B's budget input would be destroyed by the observation normaliser.**
   - OmniSafe standardises every observation feature with running statistics (clip ±5, std
     floor 0.01).
   - In a single-level arm the budget feature is constant, so it always normalises to 0 in
     training. Every unseen budget then maps to ±5.
   - Role 5 must append the budget *after* normalisation, or the group must decide otherwise,
     before the pilot's Moderate run (Q-budget-normalisation).
4. **Registered settings that differ from OmniSafe's defaults.** The launcher applies them and
   verifies the rest of Table A.1 at every launch:
   - `torch_threads` is 16 in OmniSafe; Table 3.1 requires 1.
   - `save_model_freq` is 100 epochs; Part 3.4 requires every 10 epochs (200,000 steps).

### 3.2 To settle by amendment before the go decision or the main sweep

- **LR decay confound.** OmniSafe decays the actor learning rate linearly to 0 at the end of
  training (`linear_lr_decay: True`).
  - Late arms therefore train their constrained phase at (1−N) of the reference arm's learning
    rate.
  - The H3 "additional constrained training" control breaks "no change to the network or optimiser"
    either way. As one run of T + N·T steps (as the manifest queues it), its learning rate decays
    more slowly over the first T steps (a higher rate at every epoch after the first) than the
    untreated late arm's. As a continuation after T, its learning rate is **exactly 0**, so the
    actor cannot change, which would bias H3(b) toward support (Q-data-control-lr).
- **The multiplier optimiser is Adam, not eq. (4).**
  - λ rises about 0.035 per epoch whatever the size of the violation. The rehearsal measured
    0.001 → 0.036 → 0.071 while the cost was 84.6.
  - This matters for H4, for the ramp (λ stays near 0 for ~11 epochs after the ramp starts to
    bind) and for the rate limit (it binds only for λ < 0.5).
- **J_C is a 50-episode moving window.** OmniSafe's `Metrics/EpCost` is the mean of the last 50
  episodes (2.5 epochs), not "the mean episode cost of the current epoch". The code now also logs
  the true per-epoch batch mean (`Metrics/BatchEpCost`) for the ledger and for recovery time. The
  multiplier update still uses OmniSafe's window, and that must be stated or changed by amendment.
- **Continuations are undefined** (fine-tuning, transfer, Study B few-shot).
  - OmniSafe checkpoints hold only the actor and normaliser, and the learning rate is ~0 at the
    end of training.
  - The launcher's full-state checkpoints now save the critics, optimisers, scheduler and
    multiplier. The group must still fix the learning rate, the critics and the normaliser rule
    (Q-continuations). Few-shot runs wait on it.
- **Transfer Goal↔Button is impossible as written.** The observation sizes differ: PointGoal1 60,
  PointButton1 76, CarGoal1 72, CarButton1 88. Specify a mapping.
- **G2 undercounts compute.**
  - The registered 484 counts Study A's 435 *runs* as run-equivalents.
  - Constrained-steps-matched arms train up to 20M steps, and the data control trains T + N·T.
  - Real training is **540** run-equivalents, and **627** with the battery continuations.
    `python -m pilot budget` prints both.
- **Part 5.3 contradicts itself.** One sentence says "interval excludes zero" means the primary
  (Welch) interval; the last sentence says the 95% percentile interval. Part 5 freezes at go, so
  fix it before.
- **H1 does not name the onset shape of Δ(N)** (abrupt, ramp or both). The primary analysis and
  the surplus-seed arm set depend on it.
- **The ledger schema does not match Appendix B or the design (§3.3).**

### 3.3 The ledger schema (owned by Role 4; you are its only writer)

- **The recorded hash does not verify.**
  - `results/ledger_schema.sha256` records the hash of the file with Windows CRLF line endings
    (commit `7543f6f`). The committed file has LF endings, so the hash fails to verify although
    the content is identical.
  - The `.sha256` file is also UTF-16, so `sha256sum -c` cannot read it.
  - `.gitattributes` now stops this recurring. Re-record the hash by amendment.
- **The schema diverges from Table B.1 without an amendment.** It adds fields and renames
  `matched_checkpoint`, and it nests checkpoints in the row instead of the "second table" that
  First Tasks proposes.
- **The schema cannot hold things the analysis needs.**
  - It cannot record the Study B pilot's **unconstrained** run: the writer puts it in a JSON
    sidecar.
  - It has no satisfaction or cost at *training* budgets (needed for G3's drop and for the
    pilot's G4).
  - It has no violation magnitude (needed for the floor rule) and no recovery time.
- **Missing validation.** Nothing checks commit-hash length (First Tasks Role 4 step 1). The
  writer enforces 40-hex hashes instead.
- **Crash safety.** The schema's own write functions rewrite the file in place. Every write now
  goes through a lock and an atomic replace (`ledger_writer.ledger_transaction`), so a crash or a
  concurrent enrichment can no longer lose rows. The frozen file itself is unchanged.
- **Test collection.** Plain `pytest` failed to import `results`; `pyproject.toml` now fixes it.
- **`write_enrichment` does not validate.** It checks only the Parquet types, never the
  `LedgerRow` constraints, so an out-of-range value (a negative `lambda_at_selection`, a
  `dormant_onset` above 1, an `sr_zero` key that is not an unseen budget) is written and then
  makes `load_ledger_as_rows` fail for the whole ledger. The pilot's enrichment now validates the
  working copy inside its transaction; the schema should validate the rebuilt row itself (for
  example `_from_parquet_record(new_table.slice(idx, 1).to_pylist()[0])` before `pq.write_table`).
- **Study B maps are validated unevenly.** `sr_fewshot` budgets are not restricted to the unseen
  budgets (`nan_200000` and `-3_500000` are accepted), although `sr_zero` and `adapt_steps` are;
  satisfaction rates are not bounded to [0, 1]; `adapt_steps` values are unconstrained, and
  nothing fixes how Table 2.5's "recorded as above the largest horizon if never reached" is
  encoded; float-key coercion silently merges `'10'` and `'10.0'`.
- **No cross-field checks.** A Study A row without `N`, a Study B row with `N` or `treatment`, a
  `finished` before `started` and `seed = True` are all accepted (First Tasks Role 4 step 1 asks
  for a validation that rejects a row with a missing field).
- **Unbounded floats accept NaN and ±inf.** Selection cost and return, final cost and return,
  `selection_cost_at_match`, `measurement_cost` and the `gap_*` fields validate NaN and ±inf, and
  the fields bounded only below (`wall_clock_hours`, `lambda_*`, `rank`, `norm`, `rank_onset`,
  `norm_onset`, `multiplier`) validate +inf (the pilot code guards them upstream; the analysis
  relies on the schema). Add `Field(allow_inf_nan=False)` to them; keep
  `training_cost`/`training_return` able to hold NaN for the untrained step-0 checkpoint. The same
  holds for the values of `sr_zero`, `sr_fewshot` and `per_level_multipliers` and the items of
  `training_levels`; there the constraint goes on the value type
  (`Dict[float, Annotated[float, Field(allow_inf_nan=False)]]`; `Dict[str, …]` for `sr_fewshot`,
  `List[Annotated[…]]` for `training_levels`).
- **The test fixture's selection window has eleven checkpoints.** `make_study_a_row()` in
  `tests/test_ledger_schema.py` gives a selection cost to every checkpoint with `step >= 8_000_000`
  (eleven), where Parts 3.4 and 4.1 say the last ten, (8M, 10M]; use `step > 8_000_000`.
- **`notes` is an enrichment field and is replaced, not appended.** An enrichment can erase the
  provenance the writer keeps there (train and evaluation hours, evaluation commit, attempt,
  failure detail), against memo decision 3 ("Raw fields are immutable"). Remove it from
  `ENRICHMENT_FIELDS` or append.
- **A second rename from Table B.1.** Table B.1's `selection_cost` is `selection_cost_at_match` in
  the schema; record it with `matched_checkpoint` in the amendment.
- **`results/ledger_schema_requirements.txt` is unpinned.** The schema needs pydantic 2
  (`field_validator`, `ConfigDict`); pin the versions of `environment/requirements.in` (pandas
  2.0.3, pyarrow 25.0.1, pydantic 2.13.5), or at least `pydantic>=2`.
- **The memo (`docs/ledger_schema_memo.md`) is out of date.** Issues for Role 4, to fix with the
  schema amendment:
  - It announces "four implementation notes" but has six (A to F), and "Twenty-two tests" where
    `tests/test_ledger_schema.py` has 25.
  - Its "Date: [date of freeze]" placeholder is unfilled (the freeze was 2026-09-26), and the hash
    it quotes is the CRLF one above.
  - It does not render as Markdown: the header lines run into one paragraph, and note E and
    "TESTING." merge into the paragraphs before them (D and F) for lack of blank lines.
  - It spells "serialization" in an otherwise British text, and its sentence "Routes through a
    single serialization path, …" has no subject ("It routes through …").
  - Note A says Parquet map keys must be strings; Parquet and PyArrow store float64 keys, so the
    string keys are a design choice (record it as such).
  - Decision 2 ("No separate tables joined by run_id") contradicts the second table that First
    Tasks proposes; record the choice in the amendment.
  - The schema's docstring says a change "after the pilot" is an amendment; its header comment
    says that any change is.

---

## 4. The ordered task list, first to last

`[YOU]` marks the pilot owner's tasks; the others are the owners' tasks you depend on or
coordinate. Every pull request cites the pre-registration location it implements.

### Phase 0: registration record and repository

1. **[YOU] Create and push the tag.**
   - Run `bash scripts/tag_registration.sh`, then `git push origin registration`.
   - The script checks the commit, its two files and its timestamp. (Table 9.0; First Tasks 2a.)
2. **[YOU] Get an independent time stamp** (recommended).
   - Publish a GitHub release of the tag with the Zenodo integration on (a DOI), or deposit the
     PDF on OSF Registries.
   - Adding the DOI to Table 9.0 is an amendment in Table 9.1. (First Tasks 2b.)
3. **[YOU] Fill in the registration tables.**
   - In Word, fill Table 9.0: repository address, hash `735b394d18d1bb046c7a74f900af00a83f1fa316`,
     timestamp `2026-09-18T23:22:45+05:00`, tag `registration`.
   - Fill the first row of Table 9.1 with the same values.
   - Save the PDF from Word and commit both files. It is no longer the second commit; say so in
     the message.
4. **[YOU] Commit this handover** on a role branch and open a pull request citing Table 3.1,
   Appendix A, Part 3.4, Part 5.6 and Part 6.
5. **[YOU] Complete `.github/CODEOWNERS`** with the real GitHub usernames and uncomment the rules
   (only `@Abdullah9712` is known). Give every path two owners: GitHub never counts an author's
   own approval, so a path with a single owner could never be merged by that owner. Also fill in
   the owners' names in `envs/`, `metrics/` and `studyb/` `README.md`.
6. **[YOU] Protect `main`, only now:**
   - First check that every rule in `.github/CODEOWNERS` is active and names two owners
     (otherwise the ruleset below blocks that owner's own pull requests for good).
   - Settings → Rules → Rulesets, on the default branch, with no bypass list.
   - Block deletion and force pushes.
   - Require a pull request with one approval and code-owner review.

### Phase 1: reproducible environment (Table 3.1; Appendix A)

7. **[YOU] Choose and record the workstation.**
   - Linux (use WSL2 Ubuntu if the machine runs Windows); Python 3.10.
   - As many physical cores as possible.
   - At least 50 GB free under `/data` (a symlink to the real storage is fine; see §6).
8. **[YOU] Build the environment.**
   - Run `bash scripts/setup_env.sh` on the workstation. It creates a fresh `.venv`, installs the
     pins, writes `environment/workstation.json`, checks `configs/omnisafe/`, runs the fast
     tests, and only then writes `environment/requirements.lock.txt`.
   - Commit the lock and workstation files. Everyone else installs with
     `bash scripts/setup_env.sh --locked` (exactly the lock, `--no-deps`; it records their own
     machine in `.venv/workstation.local.json` and leaves the registered workstation file alone).
     The script refuses to overwrite an existing lock (`--relock` needs an amendment).
9. **[YOU] Run the determinism check.**
   - From a clean commit: `python scripts/determinism_check.py`.
   - It trains 2 epochs twice and compares every log column, every checkpoint tensor and the
     config hash.
   - Commit `ledger/determinism/determinism-*.{json,md}`.
   - Re-run it with `--registered-form` once Role 2's harness exists (until then that form stops
     at once with UNAVAILABLE, before training anything). `--keep DIR` keeps the two run
     directories; DIR must not already hold runA or runB.

### Phase 2: questions to the group and one amendment before the pilot

10. **[YOU] Raise every question of §9** as a GitHub issue titled
    `Pre-registration question: <key>`, in one batch. The group answers at one meeting.
11. **[YOU] Record the answers** in Table 9.1 with the commit hash. Then, in the same pull
    request, add each answered starred key (§9) to `ANSWERED_QUESTIONS` in
    `configs/registered.py` and run the fast tests: `tests/test_registered.py` checks that every
    answered key is a `PENDING` key. Unstarred keys are not in `PENDING` and are recorded in
    Table 9.1 only.
    - This releases the runs waiting on it; `python -m pilot design all -v` shows what is still
      open.
    - If the group answers differently from the proposal, change the code that implements the
      proposal (and the `PENDING` text) to the answer, then add the key in the same pull request.
      The scheduler refuses the changed specs already queued, so update the queue deliberately.
    - Decisions taken now, before any data, keep every result confirmatory.
12. **[YOU] Record the machine-hour allocation.**
    - The group decides it. Then run
      `python -m pilot allocate --hours H --by "the group" --date YYYY-MM-DD` and **commit**
      `pilot/allocation.json`.
    - Enter the same value in Table 8.1 ("Machine-hours allocated to the registered runs") with
      an amendment row in Table 9.1, before any pilot run: Part 6 G2 requires it "written into
      Part 8 before the pilot's throughput is read".
    - The scheduler starts no pilot run until HEAD contains that file.
    - The go report checks that every pilot run was launched from a commit holding exactly that
      allocation (Part 6 G2).

### Phase 3: components the pilot needs from the other roles (contracts in §8)

13. **Role 2 (Environment and tests):**
    - the onset plug-in `envs.onset:make_algorithm` (N = 0 and abrupt onset for the pilot);
    - the seeded evaluation harness `envs.evaluation`: `evaluate_run`, `evaluate_battery` and
      `evaluate_checkpoint`, with disjoint selection and measurement seed sets and a frozen
      normaliser. Not OmniSafe's `Evaluator`, which is unseeded and updates the normaliser;
    - hazard relocation (as amended) and dynamics perturbation.
14. **Role 3 (Metrics and interventions):** fixed batches, the three metrics, and the logging hook
    writing `plasticity.csv` from the pilot's first run.
15. **Role 4 (Analysis and results):** `analysis.matching.select_checkpoint` (rule 1; the pilot
    code re-checks it), rules 2–7, schema v2 if adopted, and the Part 5 statistics.
16. **Role 5 (Study B and literature):**
    - the **targeted search of Appendix C, finished before any Study B run**. The pilot contains
      two, so it must be done before the pilot;
    - budget conditioning `studyb.conditioning:make_algorithm` (budget after normalisation);
    - `studyb.evaluation.evaluate_training_budgets` (G4).
17. **[YOU] Enter the search record's path in Table 9.0** (`studyb/search/`, from Role 5) by an
    amendment row in Table 9.1, before the pilot's Study B runs (Part 7.2; Appendix C; First Tasks,
    Role 5 step 7).

### Phase 4: rehearsal

18. **[YOU] Run a single-seed smoke test of the whole pipeline** on short runs; the
    pre-registration allows these ("single-seed smoke tests of the pipeline") and they are not
    analysed. Use `python scripts/smoke_run.py --data-root /scratch/smoke` (one 1-epoch run; add
    `--epochs 100` to pass the evaluation stage, which needs the last ten checkpoints; the
    evaluation of shorter runs ends `eval_failed`, or leaves them `trained` until Role 2's harness
    exists).
    - `--design pilot --epochs 2` runs short copies of the eight pilot runs (three seeds of each
      Study A arm and the two Study B runs; their N = 0.50 onset rounds down to 0 below 20 epochs),
      on a scratch data root that holds no registered runs. Use it only to
      exercise the scheduler, never look at its costs, and say in the go report that it was run.
    - It runs the real scheduler and launcher in smoke mode: the ledgers go to
      `/scratch/smoke/smoke/`, never to the repository, and the data root is fixed as a smoke root
      (`schedule run`, `resolve` and `set-policy` on it need `--allow-dirty`/`--allow-pending` and
      no other `--ledger-dir`; registered runs need another data root).
    - The command line cannot queue short runs, because `schedule add` takes only the registered
      designs.
19. **Role 4: rehearse the analysis script once on synthetic or smoke data**, before the pilot's
    data exist (its registered test run on the pilot data is task 25).

### Phase 5: the pilot (Part 3.6)

20. **[YOU] Queue and run the eight pilot runs.**
    - `python -m pilot schedule add --design pilot`
    - `python -m pilot schedule run --max-concurrent K`
    - Choose K as the number of runs the workstation can hold at once, usually physical cores
      minus 1.
21. **[YOU] Select the checkpoints.**
    `python -m pilot enrich select --ledger results/pilot/ledger.parquet` applies rule 1.
22. **[YOU] Evaluate the battery.**
    `python -m pilot enrich battery --ledger results/pilot/ledger.parquet` runs the measurement
    set, hazard relocation and dynamics perturbation. Use the data root the ledger was written
    from (default `/data`): a run directory whose commit or matched checkpoint path differs from
    the row's is refused, and the log records the data root.
23. **[YOU] Measure the Moderate run.** `python -m pilot g4-measure` evaluates it at its three
    training budgets, from a clean commit, and writes `results/pilot/g4_moderate.json` once. Use the
    data root the ledger was written from: a run directory whose commit or final checkpoint path
    differs from the ledger row's is refused. The harness result must give a finite `mean_cost` and
    a satisfaction rate in [0, 1] at each of the three training budgets; otherwise nothing is
    written, so a faulty evaluation never uses up the once-only file.
24. **[YOU] Write the go report.**
    - `python -m pilot go`, from a clean commit, writes a timestamped
      `results/pilot/go_report-<UTC>.{json,md}` with G1–G4, the reading of Part 6 and the Table 8.1
      values; the report records the commit and `worktree_dirty`. `--moderate PATH` names the
      g4-measure output; a path that does not exist is refused (only the default may be absent, and
      G4 is then not computable).
    - Concurrency is measured from the pilot runs' own start and finish times. G2 waits until all
      eight pilot runs are complete.
    - A condition that waits on an open question (Q-g1-level, Q-g3-pairing) is shown as
      `UNDECIDED (<key>)`, not NOT COMPUTABLE; the group decides before the condition is read
      (answering the key, or, for G3 with fewer than three seed pairs, deciding G3 itself). The
      reading of Part 6 waits on an undecided G3 until G1 or G2 has failed; from then on it gives
      that branch (revision, no-go or cut order) and names G3 as undecided in it (not after a
      second G1 failure, which makes Study A a no-go).
25. **Role 4: commit the analysis script and run it once on the pilot data** (Part 5.8: it "is run
    once on the pilot data to test it"). From the go decision on, it is edited only to fix
    errors, each recorded in Part 9.
26. **[YOU] Prepare the surplus-seed plan** (Part 5.5) for the go meeting:
    - `python -m pilot surplus --hours-per-run H --concurrent K` gives the registered plan, with
      the corrected plan and a warning beside it.
    - The seed count is recorded in Part 8 at the go decision (task 27).

### Phase 6: the go meeting (Part 6)

27. **Group, with you presenting.**
    - Enter the Table 8.1 values in Part 8, including the seeds per primary-comparison arm from
      task 26, and record the decision.
    - Commit the amended document (the row Table 9.1 labels "First amendment"; the amendment
      rows of tasks 2, 11, 12 and 17 are entered above it) and freeze Parts 1, 4, 5 and 6.
28. **If a condition fails:**
    - G1, G2 or G4: one revision recorded in Part 9, then a re-pilot of the same size:
      `schedule add --design repilot` queues the same eight runs under run_ids `P1-…`, so they
      never collide with the first pilot's records. Then `g4-measure --revision 1` and
      `go --repilot G1` (name every condition that failed on the first pilot): only a second
      failure of the same condition is a no-go for that study, a second G2 failure leads to the
      cut order of Part 6.1 (task 29), and a condition that fails for the first time on the
      re-pilot is reported as a question for the group (Part 6 does not cover it). If G3 also
      fails, or is undecided, while G1 or G2 fails on the re-pilot, the report says so as a
      question for the group (Part 6 decides G3 only when G1 and G2 hold), unless a second G1
      failure has already made Study A a no-go.
    - G1 and G2 hold but G3 fails: a negative pilot with its U80 bound, and Study B becomes the
      main study.

### Phase 7: the main sweep of Study A (after go)

29. **Cut order first.** If G2 failed after the revision, decide the cuts in the order of Part 6.1
    (the `go --repilot` report prints it) before queuing anything, and queue only the groups that
    remain in tasks 31 and 34. Only the first cut (do not queue `pid`) is possible with the command
    line, which queues whole designs. The others (drop SafetyCarGoal1-v0, shorten the few-shot
    continuations to 200,000 steps, restrict treatments and controller variants to N = 0.50, drop
    N = 0.10) need a manifest change (a design filter or reduced designs), recorded with the
    amendment, before tasks 31 and 34.
30. **Role 4: confirm the analysis script is committed** before the first main-study run completes
    (Part 5.8; committed in task 25).
31. **[YOU] Queue the main sweep and the additional arms.**
    - `schedule add --design main`, then `treatment`, `controller` and `pid`.
    - Then `schedule add-surplus --extra E` for the surplus seeds of task 26. E is a per-arm
      target fixed for the data root (Part 5.5: the same count for every arm); running it again
      with the same E adds only the seeds an arm lacks. Surplus seeds of the N = 0.50 ramp arms
      wait on Q-surplus-arm-set.
    - Warm-started runs wait for their N = 0 run, even if it is queued later. Runs with open
      questions wait for `ANSWERED_QUESTIONS`; `schedule status` shows why a run waits.
32. **[YOU] Run and monitor.** `schedule run --max-concurrent K` and `schedule status`.
    - A held interruption: `schedule resolve RUN restart` (same seed, on the commit the
      interrupted attempt ran on; it reproduces the run bit for bit, and waits while HEAD differs).
    - Once the group rules on Q-interrupted-run, apply its decision to every run with
      `schedule set-policy restart|exclude`.
    - A failed evaluation: `schedule resolve RUN reevaluate`. A failed ledger write:
      `schedule resolve RUN retry-ledger`.
    - A group decision on an excluded run left without a replacement (after two same-cause
      exclusions), or on a run blocked by an excluded dependency: `schedule resolve RUN replace`
      queues the arm's next unused seed (a blocked run so replaced becomes superseded); `schedule
      resolve RUN unblock` returns a blocked run whose dependencies are fine. A Study B
      continuation follows its parent's seed: for a blocked one, replace its parent (`schedule
      resolve B-…-sK replace`), which supersedes the old continuations and queues the new
      parent's; an excluded continuation is decided by amendment (the scheduler has no action for
      it).
    - Never re-run a completed run and never exclude on results; the scheduler enforces both.
33. **[YOU] Enrich the ledger** (`enrich select`, `enrich battery`) as runs finish; it is safe
    while the scheduler runs, and it refuses to write results produced by uncommitted code
    (tracked changes, or an untracked selector or harness that it imported). An enrichment is
    refused if any ledger row would break the schema.
    - The fine-tuning and transfer continuations of the battery (87 run-equivalents) have no
      design or command yet: `enrich battery` covers only `hazard` and `dynamics`. Once
      Q-continuations and Q-transfer-obs are answered, add a manifest design and a plug-in for
      them, then queue them.

### Phase 8: Study B (after Study A's main sweep; after the targeted search)

34. **[YOU] Queue and run Study B.**
    - `schedule add --design study_b`, then `study_b_fewshot` (140 continuations, which wait for
      their parent and for Q-continuations).
    - Run `schedule add-surplus --extra E` again with the E of task 31: it adds the Study B arms'
      surplus seeds and their continuations.
    - A continuation ends as `continued`, and its results are written to the parent row
      (`sr_fewshot`, `adapt_steps`) by Role 5's evaluation. That enrichment command is still to be
      added once Role 5's contract is fixed.

### Phase 9: analysis and reporting

35. **Role 4: final analysis.** Primary and secondary estimands, both intervals, the sensitivity
    analyses of rule 7, Holm families, nulls as bounds, and exploratory labels.
36. **[YOU] Freeze and publish the data.** Commit the final ledgers, archive `/data/checkpoints`
    and the scheduler database outside the repository, and publish a release with a DOI.
37. **All: write the paper.** Report the matched and final-checkpoint estimands side by side, name
    every limit on causal interpretation (Part 4.1.1), and list every amendment with its date and
    whether data had been seen.

---

## 5. Repository map: what each file is for

| Path | Owner | What it is and what you do with it |
|---|---|---|
| `prereg/Preregistration.{docx,pdf}` | Group | The registered plan (commit `735b394`). Change only by amendment; fill Table 9.0/9.1 (task 3). |
| `README.md` | You | One-paragraph description and link to `prereg/` (First Tasks, Role 1 step 4). |
| `HANDOVER.md` | You | This file. |
| `pyproject.toml` | You | Python 3.10; pytest configuration (`pythonpath`, `-m 'not slow'` by default, markers `omnisafe`, `slow`). |
| `.gitattributes` | You | LF line endings in the repository, so file hashes are the same on every OS. |
| `.gitignore` | You | Excludes run outputs, checkpoints, virtual environments. |
| `.github/CODEOWNERS` | You | Fill usernames, uncomment (task 5). |
| `environment/requirements.in` | You | The direct pins, with the reason for each. Change only by amendment. |
| `environment/requirements.lock.txt`, `environment/workstation.json` | You | *Created on the workstation* by `scripts/setup_env.sh`. Commit them. |
| `configs/registered.py` | All (you created it) | Every registered number with its location; `PENDING` (open questions); `ANSWERED_QUESTIONS` (keys the amendment log has answered). Other roles add their values by PR. |
| `configs/omnisafe/{PPOLag,CPPOPID,PPO}.yaml`, `SOURCE.json` | You | Verbatim copies from omnisafe 0.5.0 with version and SHA-256. Never edit. |
| `pilot/manifest.py` | You | Every registered run as a `RunSpec` with a deterministic `run_id` (`A-PointGoal1-N0.50-abrupt-total-s3`, `B-Dense-s2`, `P-…` for the pilot); `PLUGINS`; open questions per spec. |
| `pilot/launch.py` | You | One run. `train`: refuses (exit 5, nothing written) on open questions, existing output, a claimed directory, uncommitted code, an installed YAML other than the committed copy, or defaults that differ from Table A.1; builds the config as `omnisafe.Agent` does with only the registered overrides; enforces one thread; claims the run directory; classifies the outcome by Part 5.6; writes `train_result.json`. `evaluate`: Role 2's harness, checked, writes `evaluation.json`. |
| `pilot/algorithms.py` | You | Plug-ins for plain PPO-Lagrangian and PPO. `FullStateCheckpointMixin`: the learner state in every checkpoint (networks, optimisers, scheduler, multiplier, the 50-episode windows; not the random number generators), the onset checkpoint, and the per-epoch batch cost and return (Part 3.4), without changing training. Roles 2 and 5 mix it into their classes. |
| `pilot/contracts.py` | You | The interfaces the other roles implement, and their checks (checkpoint set, selection window, evaluation seeds). |
| `pilot/scheduler.py` | You | SQLite-backed scheduler (see §7). |
| `pilot/ledger_writer.py` | You | The only writer of ledger rows, with locked atomic writes. Pilot runs go to `results/pilot/ledger.parquet`; the unconstrained pilot run to a sidecar; smoke runs are refused by the repository ledgers. |
| `pilot/enrichment.py` | You | Rule-1 selection (Role 4's function, re-checked) and battery gaps (Role 2's harness, seed-checked), with a provenance log. |
| `pilot/budget.py` | You | Run-equivalents (registered 484 and corrected 627), the allocation record and its containment check, the surplus-seed plan. |
| `pilot/go_decision.py` | You | G1–G4 (G1 on arm means with the per-seed reading beside it), equation (12) (paired, with the pooled reading beside it), measured concurrency, the reading of Part 6 (first pilot, and re-pilot given the conditions that failed first), the Table 8.1 values, timestamped reports that are never overwritten. |
| `pilot/provenance.py` | You | Commit hash, clean-tree rule (exact output files only), imported-code check, configuration hash, machine, UTC time. |
| `pilot/__main__.py` | You | The command line (§7). |
| `pilot/allocation.json` | You | *Created* by `python -m pilot allocate` (task 12). |
| `scripts/setup_env.sh`, `copy_omnisafe_configs.py`, `record_workstation.py`, `determinism_check.py`, `smoke_run.py`, `tag_registration.sh` | You | Environment, config copies, workstation record, determinism check, smoke test of the pipeline (task 18), registration tag. |
| `ledger/determinism/` | You | Determinism reports (First Tasks path). |
| `results/ledger_schema.py`, `.sha256`, `_requirements.txt`, `tests/test_ledger_schema.py`, `docs/ledger_schema_memo.md` | Role 4 | Frozen ledger schema v1. Do not edit without an amendment (§3.3). |
| `results/ledger.parquet`, `results/pilot/` | You (writer) | Ledgers, created as runs complete, with `*.seeds.json` (the evaluation seed sets, fixed by the first run) and `*.enrichment_log.jsonl`. Commit them regularly. |
| `envs/`, `metrics/`, `analysis/`, `studyb/` | Roles 2, 3, 4, 5 | Role folders with owner READMEs. |
| `tests/` | You | 339 tests: 338 fast and 1 slow (see §12); `tests/fake_launcher.py` simulates runs for the scheduler tests. |

---

## 6. Setting up

**Workstation requirements.**
- Linux (native or WSL2 Ubuntu) with Python 3.10 and Git.
- Registered runs are CPU-only with one thread each, so throughput scales with the number of
  physical cores.
- A full-state checkpoint is about 0.3 MB. A 10M-step run keeps 51 of them (52 when its onset
  falls off the ten-epoch grid), about 16 MB, and the whole study needs about 10 GB of
  checkpoints. Plan at least 50 GB under `/data`, with backups.

```bash
git clone https://github.com/paf-iast-ai-research/when-safety-comes-late.git   # or unpack the archive
cd when-safety-comes-late
bash scripts/setup_env.sh            # first time on the workstation (creates the lock file)
source .venv/bin/activate
pytest -q                            # the fast tests
pytest -q -m slow                    # 2-epoch training: launcher == omnisafe.Agent bit for bit (~10 min)
python scripts/determinism_check.py  # from a clean commit; writes ledger/determinism/
```

- Other members: `bash scripts/setup_env.sh --locked` (it does not touch
  `environment/workstation.json`).
- If `python3.10` is not on your PATH, set `PYTHON=/path/to/python3.10`.
- Run every command from the repository root.
- Data root: `/data` by default. For registered runs keep it at `/data` (make `/data` a symlink
  to the real storage if needed): the ledger stores checkpoint paths relative to the schema's
  `/data/checkpoints`, and the ledger writer refuses a row for the repository's ledgers whose
  paths lie elsewhere. `WSCL_DATA_ROOT` or `--data-root` point elsewhere only for smoke tests.

---

## 7. Running things

```bash
python -m pilot design pilot -v          # the 8 pilot runs, their steps, onsets, open questions
python -m pilot design all               # 610 runs, 540.13 run-equivalents of training
python -m pilot budget                   # registered 484 vs corrected 627.13 run-equivalents

python -m pilot allocate --hours 12000 --by "the group" --date 2026-10-05   # then COMMIT pilot/allocation.json

python -m pilot schedule add --design pilot
python -m pilot schedule run --max-concurrent 7            # long-running; use tmux or nohup
python -m pilot schedule status --events 30
python -m pilot schedule resolve P-A-PointGoal1-N0.00-s1 restart     # a held interruption
python -m pilot schedule resolve P-A-PointGoal1-N0.00-s1 reevaluate  # after a failed evaluation
python -m pilot schedule set-policy restart                          # the group's ruling, for every run
python -m pilot schedule resolve P-A-PointGoal1-N0.00-s5 replace     # the group's decision after the circuit breaker

python -m pilot enrich select  --ledger results/pilot/ledger.parquet
python -m pilot enrich battery --ledger results/pilot/ledger.parquet --conditions hazard dynamics
python -m pilot g4-measure
python -m pilot go                      # re-pilot: go --repilot G1 (the conditions that failed first)
python -m pilot surplus --hours-per-run 12.5 --concurrent 7
python -m pilot schedule add-surplus --extra 2             # per-arm target; repeat after queuing Study B
```

**How the scheduler behaves.**
- **Run layout.** Each run lives in `/data/checkpoints/<run_id>/`:
  - `spec.json` and `.claim`;
  - `omnisafe/<algo>-{<task>}/seed-XXX-<time>/` (`progress.csv`, `config.json`,
    `torch_save/epoch-{k}.pt` = state after k epochs);
  - `train_result.json`, `evaluation.json`.
- **State and logs.** Logs are in `/data/scheduler/logs/`; state (runs, events, settings) is in
  `/data/scheduler/state.sqlite`.
- **Never twice.** A run is never launched twice:
  - its state is committed as `training` before the process starts;
  - the launcher claims the directory atomically;
  - a run already in a ledger is skipped (an exclusion record found there marks it excluded);
  - only one scheduler can run per data root, and operator commands serialise with it on
    database write transactions;
  - a live run is recognised by its run directory on its command line (compared as real
    paths, so a relative, absolute or symlinked spelling of the data root still finds it) and is
    never restarted.
- **Resume.** Start the scheduler again. Live runs are adopted, finished runs are settled, runs that
  never started go back to pending, and exclusions not yet written to the ledger are retried. A
  restart by the policy cut short after its archive move is completed (it is not taken for a run
  that never started); one by `resolve` or `set-policy` leaves the run held, and repeating the
  command completes it. Under the `exclude` policy a run still held (a `set-policy exclude` cut
  short) is excluded. A run that dies from a signal while no scheduler is its parent leaves no exit
  status: it is held as an interruption, not excluded as a crash. One workstation per data root.
- **Interruptions.** A run stopped by the machine, or whose launcher ended without a final result,
  is **held** under the default policy. The policy is fixed per data root and applies to every run
  alike. A held run can be restarted (same seed, from scratch, on the commit its interrupted attempt
  ran on, as recorded in its `train_result.json`; it waits while HEAD differs), but never excluded
  one by one. `schedule run` uses the stored policy; change it only with `schedule set-policy`.
  Under `restart` a run is restarted at most twice (`MAX_RESTARTS`), then held; a restart that
  cannot happen (a live process, an archive already present, a file-system error while archiving)
  holds the run with a `restart_failed` event and the scheduler goes on. `schedule resolve RUN
  restart` restarts a run only if it is still held when its write transaction starts, and the policy
  (`set-policy`, or a scheduler start under `exclude`) restarts or excludes a held run only if it is
  still held then: a run another command acted on first (for example restarted and launched again)
  is left as it is (`restart_skipped`, `exclusion_skipped`). A signal that arrives after the last
  epoch and the final checkpoint does not undo a complete run. Only the first SIGTERM or SIGINT from
  the moment the launcher's handlers are installed (right after the claim) to the end of training
  interrupts it (exit code 3, `train_result.json` status `interrupted`); any other signal (a second
  one arriving at the same time, or one after training ended) is only recorded. On every exit of the
  launcher's training stage, even when writing `train_result.json` fails, the original handlers are
  restored and the recorded signals are delivered to them, after `train_result.json` is written when
  it can be (a thread mask cannot do this: OmniSafe's TensorBoard writer thread would receive the
  signal). An interruption never replaces a non-finite loss or multiplier that the log already
  shows: that run is excluded with that cause.
- **Crashes.** A crash (Python error, segfault), an incomplete run, or a non-finite loss or
  multiplier is **excluded**. The exclusion, its replacement and its dependents' new states are one
  database transaction; the ledger record of the exclusion is written right after and retried at
  every pass until it succeeds. The ledger writer leaves non-finite plasticity metrics null (named
  in the notes), so a NaN run's row and its exclusion are still written. The arm's next unused seed
  (5, 6, …) waits on Q-seed-collision until that is answered. After two exclusions of one arm with
  the same cause, no further replacement is added and the arm waits for the group (`schedule resolve
  RUN replace` carries out its decision; Q-exclusion-breaker); exclusions applied by the
  interruption policy do not count, except that after two successive replacements stopped by the
  policy (each replacing the last) the arm also waits for the group. A replaced Study B run's
  continuations are superseded by its replacement's. An excluded continuation is not replaced (it
  follows its parent's seed); the group decides by amendment.
- **Missing pieces.** A run whose plug-in or evaluator is not in the repository yet waits, and is
  never excluded for it.
- **Refusals.** The launcher refuses, before it writes anything, a run with open questions, a
  directory that already holds output or is claimed, uncommitted code (tracked changes, or
  untracked code the run imports), an installed OmniSafe YAML that is not the committed copy, or
  defaults that differ from Table A.1. A refused run stays pending and nothing in its directory is
  touched; a refusal is never an exclusion (Part 5.6 excludes only run outcomes). A launcher error
  before the claim leaves the run pending until the next scheduler start. In registered mode,
  training and evaluation both wait while tracked code has uncommitted changes.
- **Pilot gate.** Pilot runs start only when HEAD contains `pilot/allocation.json`.
- **Modes.** A data root's mode (smoke or registered) and ledger location (`--ledger-dir`, taken
  only by `run`, `resolve` and `set-policy`, or the default; compared as real paths, and relative to
  the clone for the repository's own ledgers, so a symlinked root or a fresh clone is the same
  location) are fixed by its first run or decision; `run`, `resolve` and `set-policy` refuse a
  different one. Smoke ledgers may not lie inside the repository. `schedule add` and `add-surplus`
  fix the data root's mode as registered (so `run`, `resolve` and `set-policy` with
  `--allow-dirty`/`--allow-pending` are then refused on it) and refuse a smoke data root. `status`,
  `resolve` and `set-policy` refuse a data root without scheduler state.

---

## 8. Contracts the other roles implement

| Function or file | Owner | Returns / writes |
|---|---|---|
| `envs.onset:make_algorithm(env_id, cfgs, spec) -> BaseAlgo` | Role 2 | Study A algorithm for any `spec` (N, onset_step, onset_shape, treatment, controller_variant, params). Subclass `PPOLag` + `pilot.algorithms.FullStateCheckpointMixin`, which saves the onset checkpoint and the batch metrics. Change only the actor surrogate (`_compute_adv_surrogate`) and the multiplier update (`_update`) at onset. Never touch `use_cost`. Anything added to checkpoints must load with `torch.load(weights_only=True)` (contract 3); numeric attributes of the multiplier object (Python, numpy or one-element tensor) are saved as plain numbers in `lagrange['numeric_attributes']`. A module the factory imports lazily that is missing makes the launcher exit 4 (unavailable) and clean up; it is not a crash. |
| `envs.onset:make_pid_algorithm(...)` | Role 2 | The same on `CPPOPID` (skip `pid_update` before onset). |
| `envs.evaluation.evaluate_run(omnisafe_dir, spec) -> dict` | Role 2 | `final_cost`, `final_return`, `selection` {step: [cost, return]} for exactly the last 10 checkpoints, `episodes` = 100, `selection_seeds` (100 distinct seeds, the same for every run). Deterministic mean action, a seeded reset per episode, frozen normaliser. |
| `envs.evaluation.evaluate_battery(omnisafe_dir, spec, step, conditions) -> dict` | Role 2 | `measurement` and one cost per condition (`hazard`, `dynamics`), `episodes` = 100, `measurement_seeds` (100, disjoint from the selection seeds, the same for every run). |
| `envs.evaluation.evaluate_checkpoint(omnisafe_dir, step, episodes) -> float` | Role 2 | The evaluation cost of one checkpoint over `episodes` (100) episodes, for the registered-form determinism check (contract 6 in `pilot/contracts.py`). |
| `plasticity.csv` in the OmniSafe run dir | Role 3 | Columns `step,dormant,rank,norm` (the actor), one row per checkpoint and at onset. Optional columns `norm_reward_critic,norm_cost_critic` hold the critics' norms, which Table 2.3 logs separately; they stay in the run directory, because ledger schema v1 has one `norm` per checkpoint (raise it with Role 4 for schema v2). |
| progress.csv column names | Roles 2, 5 | Every multiplier under a name starting `Metrics/LagrangeMultiplier`; Study B's per level as `Metrics/LagrangeMultiplier/level_{budget}` (e.g. `…/level_10`, `…/level_17.5`); the Continuous arm names each 5-unit bin's multiplier by its lower edge, `level_{lower edge}` (e.g. `…/level_10` for [10, 15)), and its per-level check is skipped because its `training_levels` is None; every loss under `Loss/`. All are checked for non-finite values; every multiplier is traced per epoch, and each level's final value goes to `per_level_multipliers`. |
| `analysis.matching.select_checkpoint(checkpoints) -> int` | Role 4 | Rule 1 on the ledger's checkpoint records (selection set only; ties to the later checkpoint, distances to 25 compared after rounding to 1e-4); the pilot code recomputes rule 1 and refuses a different answer. |
| `studyb.conditioning:make_algorithm(...)`, `make_fewshot_algorithm(...)` | Role 5 | Study B algorithms (budget appended **after** normalisation; per-level multipliers). |
| `studyb.evaluation.evaluate_training_budgets(omnisafe_dir, spec) -> dict` | Role 5 | `mean_cost` {10: c, 20: c, 40: c}, `satisfaction` {…}, `episodes` = 100 (G4). |

`omnisafe.Agent` rejects unknown algorithm names, so `pilot/launch.py` builds the config itself and
calls the plug-in. Plug-ins may add their own settings to `cfgs` after validation; they are then
included in the configuration hash.

---

## 9. Pre-registration questions to raise now (issue-ready)

Title each GitHub issue `Pre-registration question: <key>`. Quote the passage, give the proposal,
and record the answer in Table 9.1 before dependent code runs; then, for a starred key, add it to
`ANSWERED_QUESTIONS` (task 11). "Before" means the latest safe moment. Raising all of them now,
before any data, keeps every result confirmatory. Keys marked \* are in `configs/registered.py`
`PENDING`: runs that depend on one wait for it. The go-report keys (Q-g1-level, Q-g3-pairing,
Q-g4-level), Q-g2-run-equivalents and Q-interrupted-run (its ruling is applied with `schedule
set-policy`) gate no run. For the go-report keys the report gives both readings: G1 and G3 are
reported as UNDECIDED (key), not NOT COMPUTABLE, while their readings disagree; G3 also when a
replacement seed leaves fewer than three pairs (the paired U80 is then undefined and only the
pooled form is reported). The reading of Part 6 then says "UNDECIDED: … waits on" the key, for G3
only until G1 or G2 has failed; from then on it gives that branch and names G3 as undecided.

**Already raised by First Tasks (Table 9):**
- which evaluation set gives the matched cost (measurement set; the go report uses it);
- the checkpoint tie-break (the later checkpoint; the enrichment check uses it);
- where per-checkpoint selection costs are stored (the schema nests them, which contradicts the
  proposed second table; decide which);
- constrained-steps rounding (Q-rounding\*);
- search screening depth;
- whether the cost critic trains before onset (yes).

**New:**

| Key | Where | Question and proposal | Before |
|---|---|---|---|
| Q-hazard | Table 2.2; Part 5.2; G3 | Training already samples new layouts every episode from the same generator, so "re-sampled with the task's own layout generator" is not a shift, and 20 layouts × 5 deterministic episodes are 5 identical copies. Redefine as a real shift (e.g. more or larger hazards, hazards on the agent–goal path, placements outside the training region), with 100 distinct layouts. | **pilot** |
| Q-budget-normalisation | Table 2.5 | Append budget/100 after observation normalisation. | **pilot** |
| Q-search-before-pilot | Part 7.2; App. C | The search must finish before Study B's first run, which is in the pilot. | **pilot** |
| Q-lr-decay | Table 3.1; H1/H3 | Actor LR decays to 0 over each run; late arms have (1−N)× the LR when constrained. Keep and name it, or set `linear_lr_decay: False` for all arms. | pilot |
| Q-data-control-lr\* | Table 2.4 | OmniSafe decays the actor LR to 0 over the run's own epochs. As one T + N·T run (as specified) the LR decays more slowly over the first T steps (a higher LR at every epoch after the first) than in the untreated arm; as a continuation after T the LR is 0. Either way 'no change to the network or optimiser' fails: fix the schedule. | main sweep |
| Q-multiplier-adam | eq. (4) | λ is updated by Adam (lr 0.035) on −λ(J_C − d), clamped at 0; eq. (4) is the idealisation. | pilot |
| Q-jc-window | eq. (4); Part 3.4 | J_C is OmniSafe's 50-episode moving mean; the ledger uses the per-epoch batch mean. State both. | pilot |
| Q-first-checkpoint | Table 3.1 | "First checkpoint" = first scheduled checkpoint (200,000 steps), not the untrained `epoch-0.pt`. | pilot |
| Q-pilot-unconstrained-seed\* | Part 3.6 | Seed of the unconstrained pilot run: 0. | pilot |
| Q-interrupted-run\* | Part 5.6 | A machine interruption is restarted from scratch with the same seed (deterministic), not a "crash". | pilot |
| Q-seed-collision\* | Parts 5.5/5.6 | Replacement and surplus seeds both start at 5, and pilot arms use seeds 0–2: every arm takes its smallest unused seed ≥ 5. | pilot |
| Q-exclusion-breaker | Part 5.6 | After two exclusions of one arm with the same cause (crash, incomplete, non-finite loss or multiplier), or two successive replacements stopped by the interruption policy, the scheduler stops replacing and asks the group (`resolve replace` repeats the run); Part 5.6 says every exclusion is repeated. Confirm this operational stop. | pilot |
| Q-g1-level\* | Part 6 G1 | "Both pilot arms reach … at most d + 2.5": arm mean over seeds (as coded), or every seed? The go report gives the per-seed result beside it and reports G1 as UNDECIDED (Q-g1-level) while the two readings disagree. | go |
| Q-g3-pairing\* | eq. (12) | U80 with s/√3 and 2 df is the paired form over three seeds (as coded; eq. 12 writes s, not a pooled s). The go report gives the pooled reading beside it and reports G3 as UNDECIDED (Q-g3-pairing) while the two readings disagree. If a replacement seed leaves fewer than three pairs, the paired U80 is undefined: unless the mean clause decides G3, the group decides it, with the pooled form reported beside it (G3 stays UNDECIDED even after this key is answered). | go |
| Q-g4-level\* | Part 6 G4 vs 3.6 | "At most its budget plus 2.5" (Part 6, decisive as coded) or "within 2.5 of its budget" (Part 3.6, reported beside it). | go |
| Q-p-list | Part 3.6; Table 8.1 | Part 3.6 lists "the Moderate arm's satisfaction rates" as a [P] quantity of Part 8, but Table 8.1 has no row for them and has "mean cost at each of its training budgets" instead, which Part 3.6 omits. Add a Table 8.1 row for the satisfaction rates (reported, used by no condition) and add the mean costs to Part 3.6's list; the go report gives both. | go |
| Q-g2-run-equivalents\* | Part 6 G2 | 484 undercounts (540 training, 627 with continuations). Use the corrected total? | go |
| Q-surplus-arm-set\* | Part 5.5 | Which N = 0.50 arms get surplus seeds (both shapes and controls, as coded)? The ramp arms' surplus seeds wait on this key. | go |
| Q-interval | Part 5.3 | Two sentences disagree on which interval "excludes zero". Keep "the primary (Welch) interval". | go |
| Q-h1-shape | H1 | Which onset shape does Δ(N) use (abrupt, ramp, both)? | go |
| Q-g-names | Table 9.1; Part 6 | Table 9.1 says "G1 to G4" for Study B (Part 1.4 has G1–G5), and Part 6 reuses G1–G4 for go conditions. Rename the go conditions (C1–C4). | go |
| Q-selection-window\* | Part 4.1 rule 1 | "Last ten checkpoints" ≠ "final 2,000,000 steps" when a total is off the 200,000 grid (11.12M, 13.34M, 12.5M). Use a grid relative to the end. | main sweep |
| Q-continuations\* | Tables 2.2, 2.5 | LR, critics, optimiser state and normaliser of every continuation run. | main sweep |
| Q-transfer-obs | Table 2.2 | Observation dims differ (60/76/72/88): define the mapping and CarGoal1's held-out task. | main sweep |
| Q-warm-start | Table 2.4 | Copy only λ or also Adam's state; the source is the N = 0 arm's λ logged at epoch k0 − 1. | main sweep |
| Q-rate-limit | Table 2.4 | Under Adam (~0.035/epoch) the limit binds only for λ < 0.5 and does not stop overshoot. | main sweep |
| Q-pid-eq9 | eq. (9) | OmniSafe's PID differs from eq. (9) (EMA P term, delayed derivative of cost, start at 0). Correct the text. | main sweep |
| Q-reset-injection | Table 2.4 | `log_std` reset/injected or shared; Adam state cleared per parameter; injected params added to the optimiser. | main sweep |
| Q-ledger-v2 | App. B | Schema v2: unconstrained arm, costs/satisfaction at training budgets, violation magnitude, recovery time; record divergences from Table B.1; re-record the LF hash. | pilot / Study B |
| Q-appendix-a | App. A | Configs, environment file, workstation spec and fixed batches were "copied verbatim at the registration commit", but that commit holds only the prereg. Record that they are added later with their hashes. | pilot |

---

## 10. Working rules and mistakes to avoid

- **Numbers.** Every registered number lives in `configs/registered.py` with its location, and
  tests assert it against the registered `.docx`. Never type one elsewhere, and never choose a
  value by trying several (First Tasks habits 1, 2 and 5).
- **Launching.** Launch registered runs only from a clean commit; the launcher and scheduler
  refuse otherwise. `--allow-dirty` and `--allow-pending` are for smoke tests, whose ledgers never
  reach the repository.
- **Where runs go.** Never run on GPU. Never commit checkpoints or run folders. The pilot's runs
  never go to the main ledger.
- **Exclusions.** A run is excluded only by Part 5.6. The scheduler never looks at costs or
  returns, and neither should you when resolving a held run.
- **Allocation.** Commit `pilot/allocation.json` before any pilot run.
- **Backups.** Back up `/data/checkpoints` and `/data/scheduler/state.sqlite` regularly, and commit
  the ledgers after each batch of runs.

---

## 11. What was not possible from here, and what I need from you

- **Registration acts.** The tag push, the Zenodo/OSF deposit, the ruleset and Table 9.0 in Word
  are acts on the public registration record, so they are yours (tasks 1, 2, 3 and 6).
- **People.** GitHub usernames of all five members are needed for CODEOWNERS. The mapping of
  Roles 2, 3 and 5 to people is also needed.
- **Workstation.** Its OS, core count, RAM and disk decide K and so G2.
- **Literature.** Papers could not be read in this sandbox: arxiv.org, openreview.net,
  proceedings.mlr.press and readthedocs are blocked by its network policy. Everything above rests
  on the pinned source code, which is the stronger evidence for these questions. The literature
  claims (Sokar τ = 0.025, Kumar δ = 0.01, the benchmark table values, Yao/Sootla details) were
  not re-verified here. Role 5's search needs normal internet access.

---

## 12. Verification record

- **Environment.** Everything was run on the pinned stack: Python 3.10.21, omnisafe 0.5.0,
  safety-gymnasium 0.4.1, mujoco 2.3.0, torch 2.14.0 (the PyPI wheel, which bundles CUDA
  libraries; runs use the CPU only), numpy 1.26.4, pandas 2.0.3, pyarrow 25.0.1, pydantic 2.13.5.
- **Fast tests.** 338 fast tests pass, including the 25 original ledger tests (without OmniSafe
  installed, the tests that need it are skipped by `tests/conftest.py`). They cover every
  registered value against its phrase in the registered `.docx`, the design counts, the scheduler
  end to end with a fake launcher (crash, NaN, segfault, interruption, adoption, orphan recovery,
  circuit breaker, surplus seeds, continuations and their parent's replacement, never-started and
  refused launches, restart limit and restart commit, registered-mode gates, smoke mode and ledger
  location per data root, live runs under another spelling of the data root), the launcher's
  refusals, the ledger writer (batch metrics, per-level multipliers, portable paths, atomic
  writes), enrichment, and G1–G4 with both pilot readings of Part 6.
- **Slow test.** Two epochs through the launcher equal two epochs through `omnisafe.Agent` on
  every OmniSafe log column and on the final actor, bit for bit. The mixin adds the onset
  checkpoint and batch metrics that satisfy (batch₀ + batch₁)/2 = OmniSafe's window mean.
- **Determinism rehearsal** (sandbox, a rehearsal and not the registered check). Two runs matched
  on every log column and every checkpoint tensor, and after the fix on the config hash too.
- **End-to-end smoke.** The real scheduler with the real launcher trained a 1-epoch run on one
  thread. The run then waited correctly for Role 2's evaluation harness (status `trained`, event
  `evaluator_unavailable`), the scheduler went idle, and the smoke ledgers stayed under the data
  root.
- **Independent review.** Five reviewers covered pre-registration conformance, the OmniSafe API,
  adversarial testing, bias and integrity, and engineering practice. They reported 65 findings.
  Every confirmed defect was fixed and tested, and the rest are open questions listed in §9. The
  adversarial reviewer's own tests were then re-run against the fixed code. The ones that still
  fail do so only because they use earlier API names, or because they expect behaviour the
  registration leaves to the group; these are named in §9.

  The integrity fixes:
  - the allocation-before-throughput check is now tamper-proof (commit containment, not dates);
  - G1 is fixed to the measurement set (no operator choice);
  - concurrency is measured, not typed;
  - the interruption policy is uniform, and restarts use the same commit;
  - imported code must be committed;
  - smoke runs cannot reach the ledgers;
  - ledger writes are locked and atomic, and exclusions are transactional;
  - a circuit breaker stops systematic failures;
  - few-shot continuations have their own path;
  - surplus seeds can be queued;
  - rule 1 and the evaluation seeds are checked at the boundary.
- **Line-by-line convergence review** (after the first handover). Rounds of independent reviewers
  read every file of the repository line by line, each on one area (scheduler; launcher, OmniSafe
  and provenance; pre-registration fidelity of the design, budget and go rule; ledger,
  enrichment and command line; every document). Role 4's frozen files and the registered
  pre-registration were read too, but not edited: their defects are listed in §3.3 and §9. Each
  round's confirmed defects were fixed, with a regression test for every behavioural fix, before
  the next round started.
  - First series: round 2, 84 findings (the largest: setup errors could become exclusions, a live
    run could be trained twice, a re-pilot could not be scheduled, Part 6's re-pilot reading,
    Study B's per-level multipliers were never recorded, smoke rows could reach the repository's
    ledgers); round 3, 18; round 4, 4; round 5, none.
  - Second series, which also covered every Markdown document: round 6, about 100 findings (among
    them a signal during the result write could lose a completed run's result, `enrich battery`
    could write another data root's gaps into a row, NaN plasticity made a failed run's exclusion
    unwritable, the code-owner rules would have blocked every merge, and the task list lacked the
    Part 5.8 run of the analysis script on the pilot data); round 7, 36; round 8, 21; round 9, 9;
    round 10, 5; round 11, 2; rounds 12 to 15, 3, 2, 1 and 1 (all in how the data-root mode is
    fixed: smoke and registered runs can no longer meet on one data root in any order); round 16,
    no defects (one unused test argument, removed).

  The fast suite (338 tests; 321 pass and 17 are skipped without OmniSafe), pyflakes,
  `bash -n` and the slow bit-for-bit test pass on the final code, and `scripts/smoke_run.py` ran
  the real scheduler and launcher end to end (a 1-epoch run, trained, then waiting for Role 2's
  harness).

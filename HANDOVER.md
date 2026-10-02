# HANDOVER: Pilot owner (Role 1)

Prepared 2026-09-27, updated 2026-09-28, 2026-10-01 and 2026-10-02, for Muhammad Umair Waseem,
pilot owner. This is the single guide to the pilot owner's code and duties. It says what the study
is, what was wrong or open in the plan and how it is now decided. It gives the ordered list of work
from now to the paper, what every file is for and how to run it, and what the other roles must
review, adopt and deliver.

The pre-registration (`prereg/Preregistration.pdf`) governs everything here. Where this document or
the code differs from it, the pre-registration wins. The difference is then a question for the
group, answered in the amendment log (Part 9, Table 9.1) before the code that depends on it runs.

**What changed on 2026-10-01.** The code of every role now exists: Study A's training plug-ins and
the evaluation harness (`envs/`, Role 2), the plasticity metrics, fixed batches and interventions
(`metrics/`, Role 3), the registered analysis and the supplement schema (`analysis/`,
`results/supplement_schema.py`, Role 4), Study B's budget conditioning, evaluations and the
search protocol (`studyb/`, Role 5), and the pilot owner's integration of all of them (dependencies,
continuations, enrichment, supplement records, launch gates). **All of it was written by the pilot
owner's build, for every role.** It is a proposal to each owner, not their work: each owner must
review their part and adopt it by a pull request (task 4a) before it is used for a registered run.
Every behaviour that rested on an open question was written as a gated proposal that waited for the
amendment log (section 9).

**What changed on 2026-10-02.** Every open question is answered in `docs/DECISIONS.md` (the reasoning,
verbatim, in `docs/DECISIONS_EVIDENCE.md`). The code implements every answer and `ANSWERED_QUESTIONS`
lists all 54 starred keys, so no run or result waits on a question. The answers become amendments
only when the group ratifies them (tasks 10 and 11); §2 says what remains. The runbook
`scripts/workstation.py` runs each remaining step on the workstation with one command (§7).

**How this repository reaches you.** It comes as an archive of the complete working tree, which
you bring into your own clone of the GitHub repository (below). **The 2026-10-01 and 2026-10-02
work is not committed**, to `main` or anywhere else: as you asked, it is in the working tree only.
GitHub holds (checked on 2026-10-02): `main` at `0854221`, which contains your PR #2
(`pilot-owner/handover`, merged on 2026-09-28: the 2026-09-28 handover, with its `HANDOVER.md`,
`README.md`, `pilot/`, `configs/`, `scripts/`, `environment/`, `tests/` and `.github/CODEOWNERS`)
and, after that merge, your web edit of `.github/CODEOWNERS` (`0854221` itself), committed straight
to `main`. The build's feature branch `claude/research-project-analysis-7fhfza` (commit `b527c84`) is
deleted there. This tree was built on `b527c84`, not on `main`; it holds every file of PR #2, in the
same or a later version.

To bring it in:
1. Unpack the archive outside your clone. If it holds a `.git` (based on `b527c84`), do not commit
   or push from it: its branches predate PR #2 and conflict with `main`'s versions of the same
   files.
2. In your clone, `git fetch origin` and cut each role branch from `origin/main`, named after the
   role and component (First Tasks), for example `git switch -c pilot/infrastructure origin/main`;
   one pull request per role (task 4a).
3. Copy that role's files from the unpacked tree over the clone and review with `git status` /
   `git diff`: the diff is against `main`, so for the files of PR #2 it shows what changed since
   2026-09-28. Keep this tree's `.github/CODEOWNERS`: it names you as `@Umair-Waseem`, as `main`'s
   does, and corrects the `@Umair-WaseemE` typo of `main`'s `/pilot/` line (task 5).
4. Delete untracked files that are not part of the work (scratch data roots from a test run,
   downloaded wheels) before `git add`: `.gitignore` does not cover them, so `git add .` would pick
   them up.
5. Open the pull requests.

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
- **G2 throughput**: hours per run ÷ concurrent runs × the corrected run-equivalents (627.13 for
  the uncut design; Part 6 says 484, which is reported beside them, see Q-g2-run-equivalents) must
  not exceed the machine-hours allocated before the pilot's throughput is read (the code requires
  it before any pilot run, task 12).
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
- the ledger writer (you are its only writer) and the supplement records beside it;
- running the pilot and every later run;
- throughput (G2) and the go or no-go report;
- Parts 1 and 6 of the document (hypotheses and go rule).

**The people.** Role 1, pilot owner: Muhammad Umair Waseem (`pilot/`, `configs/`, `scripts/`,
`environment/`; GitHub `@Umair-Waseem`). Role 2, Environment and tests: Muhammad Talha Jamil
(`envs/`). Role 3, Metrics and interventions: Abdullah (`metrics/`). Role 4, Analysis and results:
Muhammad Abdullah (`analysis/`, `results/ledger_schema.py`, `results/supplement_schema.py`; GitHub
`@Abdullah9712`, the author of PR #1). Role 5, Study B and literature: Hamza Nisar (`studyb/`).

---

## 2. State of the repository (2026-10-02; GitHub checked that day)

| Item | State |
|---|---|
| Registration commit `735b394…` | Present. It holds only `prereg/Preregistration.{docx,pdf}`, and `prereg/` is unchanged since. |
| Tag `registration` | **Not on GitHub** (no tags there). The build's clone holds a local annotated tag `registration` on `735b394`, made by the build (tagger `Claude`) and never pushed: in a copy that holds it, `workstation.py registration` reports DONE, and only `git push origin registration` remains (task 1). |
| Table 9.0 / Table 9.1 first row | **Unfilled.** The "second commit" slot planned for it is gone: `main` has nine later commits (five direct commits, the PR #1 commit and its merge, the PR #2 commit and its merge). |
| Repository visibility | Public, as Table 9.0 requires. |
| `main` protection | **None.** Five commits went straight to `main` (the last, `0854221`, a web edit of `.github/CODEOWNERS` after PR #2's merge); PR #1 was merged 20 minutes after its commit and PR #2 two minutes after its commit. |
| Ledger schema (Role 4, PR #1) | In `results/`, frozen at v1 by PR #1 and **corrected in the working tree on 2026-10-01** (validation, atomic writes, recorded hash, memo; layout and API unchanged). The corrections are one amendment, decided on 2026-10-02 (X-ledger-schema-amendment in `docs/DECISIONS.md`) and to be ratified (§3.3, last list). Its tests (120 functions, 286 cases) pass on the pinned stack. |
| Code of every role | **Written, not committed**, apart from the 2026-09-28 version of your part on `main` (PR #2). `envs/`, `metrics/`, `analysis/`, `studyb/`, `results/supplement_schema.py` and the pilot integration were written by the pilot owner's build for all roles. Each owner must review and adopt their part by a pull request (task 4a). |
| Open questions | **All answered on 2026-10-02** in `docs/DECISIONS.md`: the 54 keys of `configs/registered.py` `PENDING` (22 run gates, 8 result gates and 24 report keys), the 24 unstarred notes of section 9 and the ledger-schema amendment. The code implements every answer and `ANSWERED_QUESTIONS` lists all 54 keys, so no run of `python -m pilot design all` waits on a question. The answers are amendments only once the group ratifies them (tasks 10 and 11). |
| Targeted search (Appendix C) | **Not run.** `studyb/search/` holds the protocol and empty record templates only. The search is Hamza Nisar's act (task 16); the scheduler starts no Study B run, the pilot's two included, until a complete record is committed. |
| Fixed batches (Table 2.3) | **Generated in the sandbox**, not on the workstation: `metrics/fixed_batch/*.npy` and `MANIFEST.json`. They are re-checked on the workstation with `--check` before they are committed (task 8a); under Q-appendix-a a mismatch is reported in Table 9.1, and the batches are never collected again. |
| Determinism reports | **None.** `ledger/determinism/` holds only `.gitkeep` (tasks 9 and 12a). |
| Machine-hour allocation | **None.** `pilot/allocation.json` does not exist (task 12; X-allocation: a number only the group can give). |
| Cut decision | None: `pilot/cuts.json` does not exist (needed only if G2 fails, task 29). |
| Environment lock and workstation record | Not yet created: `environment/` holds only `requirements.in` (task 8). |
| Tests | The fast suite and the slow suite (§12). |

**What remains (2026-10-02).** Every open question is decided in `docs/DECISIONS.md`, and the code
implements every answer. What remains is people's work, in this order:

1. Adopt the code by role pull requests and ratify `docs/DECISIONS.md` at one meeting, entering its
   rows in Table 9.1; Role 4 completes the analysis amendment `A1` in `analysis/AMENDMENTS.json`. Merge only
   after that meeting (tasks 4a, 10 and 11).
2. Complete the registration record: push the tag, fill Table 9.0 and the first row of Table 9.1
   (tasks 1 to 3).
3. Decide the machine-hour allocation at the same meeting and record it before any run is timed on
   the workstation (task 12).
4. On the workstation: the environment, the fixed-batch check and the determinism reports (tasks 8,
   8a, 9 and 12a).
5. The targeted search of Appendix C, by Hamza Nisar, before the pilot's two Study B runs (task 16).
6. The pilot, its enrichment, the go report and the analysis test run (tasks 20 to 26), then the go
   meeting (task 27).

Each step that runs code is one command of the runbook `scripts/workstation.py` (§7).

---

## 3. Findings you must act on

Everything in this section was checked against the pinned sources (OmniSafe 0.5.0, Safety-Gymnasium
0.4.1) or reproduced by running code. The question and answer of every key are in §9. Where the
code implements the answer to a finding, this is said under **In the code**, and a line **Decided:**
names the key answered in `docs/DECISIONS.md` on 2026-10-02; the answers wait only for the group's
ratification.

### 3.1 Blockers before the pilot

1. **Hazard relocation is not a distribution shift.** This hits the primary outcome and G3.
   - Safety-Gymnasium draws a new random layout of every object at every `reset()`
     (`bases/underlying.py: _build → random_generator.build_layout()`), and OmniSafe seeds the
     environment once.
   - So training already sees the full layout distribution, and "Hazard positions are re-sampled
     with the task's own layout generator using twenty new layout seeds" (Table 2.2) samples the same
     distribution. For every arm, the expected gap is ≈ 0 up to evaluation noise.
   - Confirmed by running it: hazard 0 moves every episode, and reseeding reproduces episode 0.
   - Also, with a deterministic policy, "20 layouts × 5 episodes" gives 5 identical copies of each
     episode.
   - Consequence: the pilot's G3 would fail by construction, and Study A would be declared a
     no-go for a design reason, not an empirical one.
   - The group had to redefine the condition by amendment **before the pilot** (Q-hazard).
   - **Decided:** Q-hazard — see `docs/DECISIONS.md`.
   - **In the code:** `envs/evaluation.py` implements both readings, and the battery uses the answer
     (`HAZARD_FORM = "central"`): hazards re-sampled by the task's generator inside the placement
     square |x|, |y| ≤ 0.75, which the generator shrinks by the hazard keepout, so hazard centres
     lie within 0.57; 20 layout seeds, each pinning the hazards, × 5 episode seeds. The
     registered-wording form enters no result.
2. **The pinned install is broken without a numpy pin.**
   - `pip install omnisafe==0.5.0` on Python 3.10 installs numpy 2.2.6 next to OmniSafe's
     `pandas==2.0.3`, and `import omnisafe` then fails.
   - pip also silently falls back to safety-gymnasium 0.4.1 with mujoco 2.3.0.
   - Fixed in `environment/requirements.in`: `numpy==1.26.4` and explicit pins; `scipy==1.15.3`
     is now pinned too (the analysis uses it; once scipy is installed `import omnisafe` also loads
     it through seaborn, an optional import, and loading a module draws no random number, so the
     pin does not change any random stream).
   - Python 3.10 on Linux is required.
3. **Study B's budget input would be destroyed by the observation normaliser.**
   - OmniSafe standardises every observation feature with running statistics (clip ±5, std
     floor 0.01).
   - In a single-level arm the budget feature is constant, so it always normalises to 0 in
     training. Every unseen budget then maps to ±5.
   - Role 5 had to append the budget *after* normalisation, or the group to decide otherwise,
     before the pilot's Moderate run (Q-budget-normalisation).
   - **Decided:** Q-budget-normalisation — see `docs/DECISIONS.md`.
   - **In the code:** `studyb.conditioning.BudgetConditionedAdapter` appends
     budget/100 after the whole wrapper chain (OmniSafe's own Sauté adapter is the precedent), so
     the normaliser stays at the task's size and never sees the budget; the actor, both critics and
     the buffer take one more input. The evaluation harness appends it the same way
     (`envs.evaluation.append_budget`) and refuses a normaliser that covers the budget feature.
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
    actor cannot change, which would bias H3(b) towards support (Q-data-control-lr).
  - **Decided:** Q-lr-decay, Q-data-control-lr — see `docs/DECISIONS.md`.
  - **In the code:** every arm keeps OmniSafe's decay, named in the paper (Q-lr-decay). The data
    control is one run of T + N·T steps whose first T steps follow the untreated arm's schedule bit
    for bit; its N·T further steps restart at (1 − N) × 3e-4 and decay to 0
    (`envs.onset.actor_lr_schedule`, `DataControlLR`; Q-data-control-lr). Only the data control's
    configuration changed: its `onset_cfgs` gain the key `actor_lr_schedule`, so its config and config
    hash differ from those of a build before this answer; no other arm's config or config hash changed.
- **The multiplier optimiser is Adam, not eq. (4).**
  - λ rises about 0.035 per epoch whatever the size of the violation. The rehearsal measured
    0.001 → 0.036 → 0.071 while the cost was 84.6.
  - This matters for H4, for the ramp (λ stays near 0 for ~11 epochs after the ramp starts to
    bind) and for the rate limit (while J_C − d is steady it binds only for λ below about 0.5;
    after a sudden rise in the violation, also above).
  - **Decided:** Q-multiplier-adam, Q-ramp-step, Q-rate-limit — see `docs/DECISIONS.md`.
  - **In the code:** the onset plug-in uses OmniSafe's own Adam update from onset, unmodified, and the
    amendment states its step size and momentum beside eq. (4) (Q-multiplier-adam); the ramp reads
    eq. (5) at the first step of each epoch (Q-ramp-step); the rate limit clips after each Adam step
    and logs the value before the clip (Q-rate-limit).
- **J_C is a 50-episode moving window.** OmniSafe's `Metrics/EpCost` is the mean of the last 50
  episodes (2.5 epochs), not "the mean episode cost of the current epoch". The code logs the true
  per-epoch batch mean (`Metrics/BatchEpCost`) for the ledger and for recovery time. The
  multiplier update still uses OmniSafe's window, and that had to be stated or changed by amendment.
  - **Decided:** Q-jc-window, Q-level-jc — see `docs/DECISIONS.md`.
  - **In the code:** the multiplier update keeps OmniSafe's 50-episode window, and the amendment
    corrects the registered gloss of J_C (Q-jc-window); Study B's per-level J_C filters the same
    window by level (Q-level-jc).
- **Continuations are undefined** (fine-tuning, transfer, Study B few-shot).
  - OmniSafe checkpoints hold only the actor and normaliser, and the learning rate is 0 at the
    end of training.
  - The launcher's full-state checkpoints save the critics, optimisers, scheduler and
    multiplier. The group had to fix the learning rate, the critics and the normaliser rule
    (Q-continuations).
  - **Decided:** Q-continuations — see `docs/DECISIONS.md`.
  - **In the code:** one restore rule for every continuation
    (`pilot.dependencies.restore_learner`): actor, both critics and the observation normaliser
    restored (the normaliser keeps updating), fresh Adam states and a fresh linear decay over the
    continuation's registered length; fine-tuning holds the multiplier at 0 with the reward advantage
    only, transfer and few-shot start a fresh multiplier at 0.001. Fine-tuning and transfer start
    from the matched checkpoint, few-shot from the final one. Under cut 3 a shortened few-shot
    continuation keeps the 1,000,000-step schedule and stops at 200,000 steps (only that spec carries
    `params["lr_schedule_steps"]`). The battery's continuations are a design
    (`manifest.battery_continuation`, queued by `schedule add-continuations`) with plug-ins
    (`envs.continuations`). Their `continuation_cfgs` strings changed with this answer, so their config
    hashes differ from those of a build before it; no such run exists yet.
- **Transfer Goal↔Button is impossible as written.** The observation sizes differ: PointGoal1 60,
  PointButton1 76, CarGoal1 72, CarButton1 88. A mapping had to be specified.
  - **Decided:** Q-transfer-obs — see `docs/DECISIONS.md`.
  - **In the code:** first-layer weights mapped by observation component
    name (`envs.continuations.transfer_map`): shared components copied, target-only components at
    zero weight, source-only components dropped; the normaliser copies the shared statistics and
    starts the others at mean 0 and variance 1. SafetyCarGoal1-v0's held-out task is
    SafetyCarButton1-v0 (`manifest.TRANSFER_TASK_PROPOSALS`; the registered constant
    `TRANSFER_TASKS` holds only the Point pair).
- **G2 undercounts compute.**
  - The registered 484 counts Study A's 435 *runs* as run-equivalents.
  - Constrained-steps-matched arms train up to 20M steps, and the data control trains T + N·T.
  - Real training is **540** run-equivalents, and **627** with the battery continuations.
    `python -m pilot design all` prints the 540 and `python -m pilot budget` the 627 (with the
    battery continuations as a separate line); `budget --cuts K` prices the design after the first
    K cuts of Part 6.1.
  - **Decided:** Q-g2-run-equivalents — see `docs/DECISIONS.md`.
- **Part 5.3 contradicts itself.** One sentence says "interval excludes zero" means the primary
  (Welch) interval; the last sentence says the 95% percentile interval. Part 5 freezes at go, so
  it had to be fixed before. (Q-interval, a report key.)
  - **Decided:** Q-interval — see `docs/DECISIONS.md`.
- **H1 does not name the onset shape of Δ(N)** (abrupt, ramp or both). The primary analysis and
  the surplus-seed arm set depend on it. (Q-h1-shape, a report key.)
  - **Decided:** Q-h1-shape, Q-surplus-arm-set — see `docs/DECISIONS.md`.
- **The ledger schema does not match Appendix B or the design (§3.3).**
  - **Decided:** X-ledger-schema-amendment, Q-ledger-v2 — see `docs/DECISIONS.md`.

### 3.3 The ledger schema (owned by Role 4; you are its only writer)

**Decided:** X-ledger-schema-amendment — see `docs/DECISIONS.md`. The corrections LS-1 to LS-32 below
are one amendment, decided on 2026-10-02 and to be ratified with the other decisions; Role 4 approves
it by pull request before the first pilot ledger row is written. The supplement schema is frozen at
version 1, its SHA-256 recorded in `results/supplement_schema.sha256` (Q-ledger-v2).

- **The recorded hash does not verify.**
  - `results/ledger_schema.sha256` records the hash of the file with Windows CRLF line endings
    (commit `7543f6f`). The committed file has LF endings, so the hash fails to verify although
    the content is identical.
  - The `.sha256` file is also UTF-16, so `sha256sum -c` cannot read it.
  - `.gitattributes` now stops this recurring. Re-record the hash by amendment. (Re-recorded in
    the working tree: LS-1 below, amendment decided.)
- **The schema diverges from Table B.1 without an amendment.** It adds fields and renames
  `matched_checkpoint`, and it nests checkpoints in the row instead of the "second table" that
  First Tasks proposes. (Recorded in the memo's "Divergences from Table B.1": LS-15 below,
  amendment decided.)
- **The schema cannot hold things the analysis needs.**
  - It cannot record the Study B pilot's **unconstrained** run: the writer puts it in a JSON
    sidecar.
  - It has no satisfaction or cost at *training* budgets (needed for G3's drop and for the
    pilot's G4).
  - It has no violation magnitude (needed for the floor rule) and no recovery time.
  - **In the code:** the schema is not changed. What the analysis needs beyond the row is written as
    **supplement records** (`results/supplement_schema.py`, Role 4; written only by
    `pilot/supplement.py`, Role 1): per-episode costs and returns, the measurement-set return, the
    battery at the final checkpoint and at tolerance 5.0, the continuations, Study B's evaluations
    at every budget and the training quantities (§8). The supplement is decided (Q-ledger-v2): its
    schema is frozen at version 1, with its hash in `results/supplement_schema.sha256`. The
    recorded definition of each battery condition (what was relocated
    or scaled) has no field there; the enrichment log records it.
  - **Decided:** Q-ledger-v2 — see `docs/DECISIONS.md`.
- **Missing validation.** Nothing checks commit-hash length (First Tasks Role 4 step 1). The
  writer enforces 40-hex hashes instead.
- **Crash safety.** The schema's own write functions rewrite the file in place. Every write now
  goes through a lock and an atomic replace (`ledger_writer.ledger_transaction`), so a crash or a
  concurrent enrichment can no longer lose rows. (The schema's own writes are now atomic too:
  LS-10 below, amendment decided.)
- **Test collection.** Plain `pytest` failed to import `results`; `pyproject.toml` now fixes it.
- **`write_enrichment` does not validate.** It checks only the Parquet types, never the
  `LedgerRow` constraints, so an out-of-range value (a negative `lambda_at_selection`, a
  `dormant_onset` above 1, an `sr_zero` key that is not an unseen budget) is written and then
  makes `load_ledger_as_rows` fail for the whole ledger. The pilot's enrichment now validates the
  working copy inside its transaction; the schema should validate the rebuilt row itself (for
  example `_from_parquet_record(new_table.slice(idx, 1).to_pylist()[0])` before `pq.write_table`).
  (Corrected in code: LS-2 below, amendment decided.)
- **Study B maps are validated unevenly.** `sr_fewshot` budgets are not restricted to the unseen
  budgets (`nan_200000` and `-3_500000` are accepted), although `sr_zero` and `adapt_steps` are;
  satisfaction rates are not bounded to [0, 1]; `adapt_steps` values are unconstrained, and
  nothing fixes how Table 2.5's "recorded as above the largest horizon if never reached" is
  encoded; float-key coercion silently merges `'10'` and `'10.0'`. The pipeline now writes
  few-shot keys only through `pilot.contracts.fewshot_key` (`"5.0_200000"`), and the analysis
  refuses a key not in that form; the censoring is decided by Q-adapt-censoring (the largest
  horizon + 1).
  (Corrected in code, censoring encoding aside: LS-6 and LS-7 below, amendment decided.)
  - **Decided:** Q-adapt-censoring — see `docs/DECISIONS.md`.
- **No cross-field checks.** A Study A row without `N`, a Study B row with `N` or `treatment`, a
  `finished` before `started` and `seed = True` are all accepted (First Tasks Role 4 step 1 asks
  for a validation that rejects a row with a missing field). (Corrected in code: LS-5 and LS-8
  below, amendment decided.)
- **Unbounded floats accept NaN and ±inf.** Selection cost and return, final cost and return,
  `selection_cost_at_match`, `measurement_cost` and the `gap_*` fields validate NaN and ±inf, and
  the fields bounded only below (`wall_clock_hours`, `lambda_*`, `rank`, `norm`, `rank_onset`,
  `norm_onset`, `multiplier`) validate +inf (the pilot code guards them upstream; the analysis
  relies on the schema). Add `Field(allow_inf_nan=False)` to them; keep
  `training_cost`/`training_return` able to hold NaN for the untrained step-0 checkpoint. The same
  holds for the values of `sr_zero`, `sr_fewshot` and `per_level_multipliers` and the items of
  `training_levels`; there the constraint goes on the value type
  (`Dict[float, Annotated[float, Field(allow_inf_nan=False)]]`; `Dict[str, …]` for `sr_fewshot`,
  `List[Annotated[…]]` for `training_levels`). (Corrected in code: LS-4 below, amendment decided.)
- **The test fixture's selection window has eleven checkpoints.** `make_study_a_row()` in
  `tests/test_ledger_schema.py` gives a selection cost to every checkpoint with `step >= 8_000_000`
  (eleven), where Parts 3.4 and 4.1 say the last ten, (8M, 10M]; use `step > 8_000_000`.
  (Corrected in code: LS-14 below, amendment decided.)
- **`notes` is an enrichment field and is replaced, not appended.** An enrichment can erase the
  provenance the writer keeps there (train and evaluation hours, evaluation commit, attempt,
  failure detail), against memo decision 3 ("Raw fields are immutable"). Remove it from
  `ENRICHMENT_FIELDS` or append. (Corrected in code: LS-3 below, amendment decided.)
- **A second rename from Table B.1.** Table B.1's `selection_cost` is `selection_cost_at_match` in
  the schema; record it with `matched_checkpoint` in the amendment. (Recorded in the memo: LS-15
  below, amendment decided.)
- **`results/ledger_schema_requirements.txt` is unpinned.** The schema needs pydantic 2
  (`field_validator`, `ConfigDict`); pin the versions of `environment/requirements.in` (pandas
  2.0.3, pyarrow 25.0.1, pydantic 2.13.5), or at least `pydantic>=2`. (Corrected in code: LS-13
  below, amendment decided.)
- **The memo (`docs/ledger_schema_memo.md`) is out of date.** Issues for Role 4, to fix with the
  schema amendment (all corrected in the working tree: LS-15 below, amendment decided):
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
    Tasks proposes; record the choice in the amendment (Q-checkpoint-table).
    - **Decided:** Q-checkpoint-table — see `docs/DECISIONS.md`.
  - The schema's docstring says a change "after the pilot" is an amendment; its header comment
    says that any change is.
- **Corrections of 2026-10-01** (the five files above; each is entry LS-n of "Changes since the
  freeze (decided 2026-10-02, X-ledger-schema-amendment; to be ratified with the other decisions)" in
  `docs/ledger_schema_memo.md`; the
  Parquet layout, field names, public functions and `SCHEMA_VERSION` 1 are unchanged; no ledger
  row existed yet):
  - LS-1 recorded hash re-recorded as UTF-8 `<hex>  results/ledger_schema.py` with LF
    (`sha256sum -c results/ledger_schema.sha256` from the root; now
    `9d53851acd49e64d7aded82c0e5b7647147f27b804e117570423b5cce320be99`): corrected in code,
    amendment decided.
  - LS-2 `write_enrichment` validates the merged row (no truncation, no bool-to-float, canonical
    map keys, list-shaped maps accepted): corrected in code, amendment decided.
  - LS-3 `notes` removed from `ENRICHMENT_FIELDS`: corrected in code, amendment decided.
  - LS-4 NaN and ±inf refused (except a checkpoint's `training_cost`/`training_return`), also in
    map values and training levels: corrected in code, amendment decided.
  - LS-5 booleans and strings refused as numbers (`seed = True`, `N = False`,
    `schema_version = True`, `'nan'`); integers bounded by int64: corrected in code, amendment
    decided.
  - LS-6 Study B maps: `sr_fewshot` budgets unseen and keys canonical (`"5.0_200000"`), rates in
    [0, 1], `adapt_steps` values ≥ 0 (censoring: Q-adapt-censoring, the largest horizon + 1),
    `per_level_multipliers` keys finite and among the training levels: corrected in code, amendment
    decided.
  - LS-7 malformed list maps are validation errors; keys equal after coercion (`'10'`, `'10.0'`)
    and boolean keys refused: corrected in code, amendment decided.
  - LS-8 cross-field checks (Study A needs N; study-specific fields; Study B needs training levels
    except Continuous; `finished` ≥ `started`; increasing checkpoint steps; matched path and step
    name one checkpoint; `training_age` = matched step; not matched and infeasible; non-empty
    identifiers): corrected in code, amendment decided.
  - LS-9 zero-offset zones normalised to UTC: corrected in code, amendment decided.
  - LS-10 `append_to_ledger` re-validates; both writes go through a temporary file and
    `os.replace`: corrected in code, amendment decided.
  - LS-11 loaders refuse a schema mismatch and duplicate run_ids, and name an invalid row's run_id:
    corrected in code, amendment decided.
  - LS-12 stored paths: no empty path, backslash or `..`; `relative_checkpoint_path` refuses `..`
    and the root: corrected in code, amendment decided.
  - LS-13 `results/ledger_schema_requirements.txt` pinned to `environment/requirements.in`:
    corrected in code, amendment decided.
  - LS-14 test fixture's selection window is ten checkpoints (`step > 8_000_000`), and the tests
    cover every correction: corrected in code, amendment decided.
  - LS-15 memo rewritten (Markdown, date and author, counts, notes A to F, decisions 2 to 6,
    divergences from Table B.1) and module docstring corrected: corrected in code, amendment
    decided.
  - LS-16 the atomic write keeps the ledger's mode (the existing file's, or 0o666 less the umask;
    `mkstemp` left 0o600, unreadable to analysts' accounts) and fsyncs through a writable handle
    (Windows): corrected in code, amendment decided.
  - LS-17 a Study B row refuses every field Table B.1 marks "(Study A)" (`matched*`,
    `training_age`, selection quantities, `infeasible`, `gap_*`, `*_onset`, `lambda_peak`,
    `lambda_final`, `settling_steps`): corrected in code, amendment decided.
  - LS-18 Study B `training_levels` equal the arm's set of Table 2.5 (none for Continuous, whose
    multiplier keys are 5-unit bin edges of [10, 40]): corrected in code, amendment decided.
  - LS-19 `matched = True` and the selection quantities need a matched checkpoint; a failed run
    carries no matching result, gap or Study B evaluation: corrected in code, amendment decided.
  - LS-20 a checkpoint's training cost and return may be NaN but not ±inf (the writer records an
    infinite value as NaN): corrected in code, amendment decided.
  - LS-21 `completed`, `matched`, `infeasible` accept only booleans; `started`/`finished` refuse
    numbers: corrected in code, amendment decided.
  - LS-22 stored paths `.`, `/` and whitespace refused; `absolute_checkpoint_path` refuses `..` and
    empty paths, as `relative_checkpoint_path` now does: corrected in code, amendment decided.
  - LS-23 float map keys spelled with `_` or surrounding spaces (`"1_5"`) refused: corrected in
    code, amendment decided.
  - LS-24 `write_enrichment` refuses duplicate run_ids, names an invalid stored row and rebuilds
    through `_to_table`: corrected in code, amendment decided.
  - LS-25 docstrings, memo (metadata table, decisions 4 and 6, notes B, C, E, F, divergences:
    `seed`, commit-hash length, two-index `sr_fewshot`) and requirements (Python 3.10): corrected
    in code, amendment decided.
  - LS-26 tests: onset follows N, `match=` on every rejection, literal run_id match, full Study A
    and Study B round trips, tests for LS-16 to LS-24: corrected in code, amendment decided.
  - LS-27 the `gap_*` fields need a matched checkpoint; the matched checkpoint lies in the
    selection window (carries a `selection_cost`); `selection_cost_at_match` and
    `lambda_at_selection` equal its selection cost and multiplier: corrected in code, amendment
    decided.
  - LS-28 list-shaped maps take only 2-element tuples or lists (a list holding a dict was read as
    its two keys); an unhashable key is a validation error, not a `TypeError`; bytes keys refused
    (`b"1_5"` read as 15.0): corrected in code, amendment decided.
  - LS-29 stored paths in `PurePosixPath` normal form (no `a//b.pt`, `./a.pt`, `a/b/`), without
    surrounding whitespace or NUL: corrected in code, amendment decided.
  - LS-30 a nanosecond pandas `Timestamp` is refused (the `us` column would truncate it):
    corrected in code, amendment decided.
  - LS-31 `append_to_ledger` creates directories only for an accepted row; the umask is read from
    `/proc/self/status` (the `os.umask` fallback documented as not thread-safe); unused import
    removed: corrected in code, amendment decided.
  - LS-32 tests (`match=` on every rejection, non-finite cases that test what they claim, tests for
    LS-27 to LS-31) and memo (attribution of the corrections to Role 1, decision 4's example,
    the `matched`/`infeasible` citation, LS-4 wording, "Testing"): corrected in code, amendment
    decided.
  - Callers: `tests/test_analysis_cli.py` writes its non-canonical few-shot key into the Parquet
    file directly (the schema now refuses it on write); docstrings only in `pilot/enrichment.py`,
    `analysis/data.py` and `tests/test_enrichment.py`; with LS-20, `pilot/ledger_writer.py`
    (`build_row_fields` writes an infinite training statistic as NaN); the test count in the
    comment of `environment/requirements.in` removed; with LS-27 to LS-32, no caller changed.

### 3.4 Found while writing the code of Roles 2 to 5 (2026-10-01)

Each is checked in the code named; the matching question is in §9.

- **Plasticity injection departs from the literal Table 2.4.** "Only the new copy is trained
  thereafter" would also freeze the first hidden layer and `log_std`. The answer keeps them
  training, which keeps the number of trainable parameters unchanged. Its basis is two agreeing
  search-engine extracts of Section 3 of Nikishin et al. (2023, arXiv 2305.15555); **the paper
  itself was not read** (arXiv is blocked in the sandbox). `metrics/interventions.py`,
  `envs/onset.py` and the Q-reset-injection text say so. Role 3 should still read the paper before
  the group ratifies the answer.
  - **Decided:** Q-reset-injection — see `docs/DECISIONS.md`.
- **Dynamics perturbation as written changes nothing.** MuJoCo combines the friction of two geoms by
  their element-wise maximum, so scaling the floor alone changes no contact, and "body mass" names
  no bodies (`envs/evaluation.py` docstring; Q-dynamics).
  - **Decided:** Q-dynamics — see `docs/DECISIONS.md`.
- **An unstable simulation raises nothing in the pinned stack.** MuJoCo 2.3.0 warns, resets the
  state and goes on; the harness reads MuJoCo's warning counters. Under the answer to
  Q-mujoco-exception an unstable episode is never scored: it is replaced by an episode on a reserve
  seed, and more than 5 in one evaluation call fail the evaluation with `MujocoInstabilityError`,
  held for the group. An evaluation failure is not a Part 5.6 exclusion cause.
  - **Decided:** Q-mujoco-exception — see `docs/DECISIONS.md`.
- **The registered 80-percent point at ten seeds is 1.3249**, which rounds to nearest to 1.32;
  Part 5.5 prints 1.33 (power at 1.33 is 0.803). No single rounding rule gives all four printed
  values (`analysis/stats.py` docstring; `tests/test_analysis_stats.py`). A note for the group; the
  code keeps the registered numbers in `configs/registered.py`.
- **H3 (b): "additional constrained training reduces the gap as much as reset"** is read as
  needing both halves: additional training itself reduces the gap, and by at least as much as reset
  (`analysis/study_a.py`, `h3b_outcome` docstring). A formula that compares only the two estimates
  would falsify the box when additional training widens the gap less than reset widens it.
- **G4 after Part 6.1 cut 3** (one horizon): under the readings "censored at the largest horizon"
  and "within at most one horizon", every index is 1 or censored, so G4 cannot separate the arms;
  the analysis reports those readings NOT_COMPUTABLE (`analysis/study_b.py`). The answer to
  Q-g4-reading reads G4 on an ordering of arm-level medians, which stays testable after cut 3, and
  keeps those two readings as sensitivity readings only.
  - **Decided:** Q-g4-reading — see `docs/DECISIONS.md`.
- **The determinism check of the PID arm needs a changed onset.** Its registered onset (5,000,000
  steps) lies beyond the check's first checkpoint, so the check moves the onset to N × the check's
  length (100,000 steps of the 200,000-step registered form). This is the answer to
  Q-determinism-late-onset; the 15 PID-check runs wait on that check's report.
  - **Decided:** Q-determinism-late-onset — see `docs/DECISIONS.md`.
- **The registered-form determinism check of Study B needs Q-studyb-eval.** Its evaluation cost
  depends on the budgets the observation carries (the arm's training budgets in turn), which is
  that key's answer.
  - **Decided:** Q-studyb-eval — see `docs/DECISIONS.md`.
- **The `analysis_missing` gate is stricter than Part 5.8.** Part 5.8 requires the analysis script
  to be committed "before the first main-study run completes"; the scheduler launches no non-pilot
  run until HEAD holds `analysis/__main__.py`, which always satisfies the text.

---

## 4. The ordered task list, first to last

`[YOU]` marks the pilot owner's tasks; the others are the owners' tasks you depend on or
coordinate. Every pull request cites the pre-registration location it implements. Task numbers are
kept from the first handover (the code and tests cite them); new tasks carry a letter.

**Done by code (2026-10-01), to be reviewed and adopted:** the onset, controller, treatment and PID
plug-ins; the evaluation harness with hazard relocation and dynamics perturbation; the battery's
fine-tuning and transfer continuations; the plasticity metrics, hook, fixed-batch collection and
interventions; Study B's budget conditioning, few-shot continuations and evaluations; the search
protocol and record check; the registered analysis with its supplement schema; and on Role 1's side
the dependencies, the continuation design and queue, every enrichment command, the supplement
writer, the cut order, the launch gates and the determinism check per plug-in. On 2026-10-02 every
open question was answered (`docs/DECISIONS.md`) and the code changed to implement each answer; all
54 starred keys are in `ANSWERED_QUESTIONS`. The runbook `scripts/workstation.py` runs each remaining
step on the workstation (§7).

**Still to do by people:** adopting the code by pull requests (task 4a); ratifying the answers of
`docs/DECISIONS.md` (tasks 10 and 11); the registration tag and tables (tasks 1 to 3); the allocation
(task 12); the fixed-batch check on the workstation (task 8a); the determinism reports per plug-in
(tasks 9 and 12a); the targeted search (task 16); then the runs, the enrichment and the analysis in
the order below.

### Phase 0: registration record and repository

*Runbook.* Task 1: or `python3.10 scripts/workstation.py registration`; task 3 is the people's step
`registration-record`, which holds `next` before the pilot runs;
`python3.10 scripts/workstation.py status` lists every remaining step (§7).

1. **[YOU] Create and push the tag.**
   - Run `bash scripts/tag_registration.sh`, then `git push origin registration`.
   - The script checks the commit, its two files and its timestamp. (Table 9.0; First Tasks 2a.)
   - The build's clone already holds a local annotated tag `registration` (tagger `Claude`, never
     pushed). In a copy that holds it, the script and `workstation.py registration` report it as
     done and only the push remains; to tag in your own name instead, run `git tag -d registration`
     first. The registration is complete only when the tag is pushed and the tables of task 3 are
     committed, before any pilot run (X-registration). The scheduler checks neither; the runbook's
     `registration-record` step checks only that the registered docx and PDF changed at HEAD, never
     the push.
2. **[YOU] Get an independent time stamp** (recommended).
   - Publish a GitHub release of the tag with the Zenodo integration on (a DOI), or deposit the
     PDF on OSF Registries.
   - Adding the DOI to Table 9.0 is an amendment in Table 9.1. (First Tasks 2b.)
3. **[YOU] Fill in the registration tables.**
   - In Word, fill Table 9.0: repository address, hash `735b394d18d1bb046c7a74f900af00a83f1fa316`,
     timestamp `2026-09-18T23:22:45+05:00`, tag `registration` (the values are at the end of
     `docs/DECISIONS.md`; X-registration).
   - Fill the first row of Table 9.1 with the same values; by the erratum of Q-g-names it reads
     "hypotheses G1 to G5".
   - The commit changes only Tables 9.0 and 9.1 and the regenerated PDF, and is made before any
     pilot run: until then the document is "the adopted draft and not the registration record".
   - Save the PDF from Word and commit both files. It is no longer the second commit; say so in
     the message.
4. **[YOU] Commit your own part** (`pilot/`, `configs/`, `scripts/`, `environment/`, the
   repository files and this handover) on a role branch and open a pull request citing Table 3.1,
   Appendix A, Part 3.4, Part 5.6 and Part 6.

   4a. **Every owner: review and adopt the code of your role by a pull request.** The code was
   written by the pilot owner's build, not by you. Read it against the pre-registration locations its
   docstrings cite, run your tests, and open one pull request per role on a branch named after your
   role and component:
   - Role 2 (Muhammad Talha Jamil): `envs/`, `tests/test_envs_*.py`.
   - Role 3 (Abdullah): `metrics/` (with `metrics/fixed_batch/` after task 8a),
     `tests/test_metrics_*.py`.
   - Role 4 (Muhammad Abdullah): `analysis/`, `results/supplement_schema.py`,
     `tests/test_analysis_*.py`, `tests/test_supplement_schema.py`.
   - Role 5 (Hamza Nisar): `studyb/`, `tests/test_studyb_*.py`.
   - Role 1 (you): the pilot integration (`pilot/`, `scripts/`, `tests/test_core_*.py`,
     `tests/test_integration_*.py` and the other pilot tests).
   The parts import each other at module level (`metrics` and `envs` import `pilot`, `studyb`
   imports `envs.evaluation`), so merge in the order Role 1, Role 3, Role 2, Role 5, Role 4. Some
   tests of one role exercise another role's code (the integration tests use all of it), and the
   fast suite was run only on the whole tree (§12): review the five pull requests together and run
   the fast tests again once all are merged. A change an owner wants is made in that owner's pull
   request; a change to an answer also changes its `PENDING` text (task 11).
5. **[YOU] Complete `.github/CODEOWNERS`** with the real GitHub usernames and uncomment the rules
   (`@Umair-Waseem` and `@Abdullah9712` are known; Muhammad Talha Jamil's, Abdullah's and Hamza
   Nisar's are missing, and a second reviewer for your own paths). `main`'s version (`0854221`)
   names `@Umair-WaseemE` on its `/pilot/` line, a typo that this tree's file corrects. Give every
   path two owners: GitHub never counts an author's own approval, so a path with a single owner
   could never be merged by that owner. Also check the owners' names in the `envs/`, `metrics/`,
   `analysis/` and `studyb/` `README.md` files.
6. **[YOU] Protect `main`, only now:**
   - First check that every rule in `.github/CODEOWNERS` is active and names two owners
     (otherwise the ruleset below blocks that owner's own pull requests for good).
   - Settings → Rules → Rulesets, on the default branch, with no bypass list.
   - Block deletion and force pushes.
   - Require a pull request with one approval and code-owner review.

### Phase 1: reproducible environment (Table 3.1; Appendix A)

*Runbook.* Tasks 8, 8a and 9: or `python3.10 scripts/workstation.py env`, then in `.venv` the
allocation (task 12, Phase 2), `python scripts/workstation.py fixed-batch` and
`python scripts/workstation.py determinism-default`, in the runbook's order. The runbook refuses
both determinism steps until `pilot/allocation.json` is committed (X-allocation).

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

   8a. **Role 3, with you: re-check the fixed batches on the workstation** (Table 2.3; Appendix A;
   Q-appendix-a). The files in `metrics/fixed_batch/` were generated in the sandbox. On the
   workstation, in the locked environment, run `python -m metrics.collect_fixed_batch --all --check`:
   it re-collects every batch and compares the bytes with the files. Commit them (with
   `MANIFEST.json`). `python scripts/workstation.py fixed-batch` also writes
   `environment/fixed_batch_check.json` whenever the check runs to a result: `"passed": true`, or
   `"passed": false` with the committed and the re-collected SHA-256 of each batch when the
   re-collection differs. On a mismatch the committed batches still stand (Q-appendix-a: collecting
   them again would be the resampling Table 2.3 forbids): the step prints the Table 9.1 row that
   reports it, and it is done, with a warning, once the record is committed. It writes nothing only
   when the check cannot run or the committed records disagree; then stop and raise it with the
   group. The SHA-256 constants in `metrics/batches.py` and every plasticity value depend on these
   bytes.
9. **[YOU] Run the determinism check (default form).**
   - From a clean commit: `python scripts/determinism_check.py` (plug-in `ppolag`, the First
     Tasks check; it waits on no open question).
   - It trains 2 epochs twice and compares every log column, every checkpoint tensor and the
     config hash.
   - Commit `ledger/determinism/determinism-*.{json,md}`.
   - `--keep DIR` keeps the two run directories; DIR must not already hold runA or runB. The
     registered-form reports the scheduler needs are task 12a.
   - Record the allocation (task 12) before this check: its run times already reveal the
     workstation's speed (X-allocation), and the runbook refuses this step until
     `pilot/allocation.json` is committed.

### Phase 2: ratifying the answers, and the allocation, before the pilot

*Runbook.* Tasks 12 and 12a: or
`python scripts/workstation.py allocation --hours H --basis TEXT --date YYYY-MM-DD` and
`python scripts/workstation.py determinism`.

10. **[YOU] Take `docs/DECISIONS.md` to the group for ratification.** It answers every question of
    §9 (the 54 starred `PENDING` keys and the 24 unstarred notes) and the ledger-schema amendment,
    decided on 2026-10-02 before any data; the code already implements every answer, and all 54
    starred keys are in `ANSWERED_QUESTIONS`. At one meeting the group reads it, approves or changes
    each row, and decides the machine-hour allocation of task 12. A row the group wants to discuss
    first can still get a GitHub issue titled `Pre-registration question: <key>`.
11. **[YOU] Record the ratified answers** in Table 9.1 with the commit hash (the suggested rows are
    at the end of `docs/DECISIONS.md`), and merge the code only after that meeting. Unstarred keys
    are not in `PENDING` and are recorded in Table 9.1 only.
    - Role 4 completes the analysis amendment already entered in `analysis/AMENDMENTS.json` (`A1`, all
      54 keys, `data_seen` false: decided before any data, Part 8.2): its `commit` is null ("not yet
      recorded", since a commit cannot name itself) until the hash of the commit that records the
      ratification is entered, with the approval; a key the group changes leaves its `keys` until its
      code follows.
    - If the group changes an answer, change the code that implements it (and its `PENDING` text)
      in the same pull request, and run the fast tests: `tests/test_registered.py` checks that every
      answered key is a `PENDING` key. The scheduler refuses the changed specs already queued, so
      update the queue deliberately (`schedule resolve RUN requeue` stores the committed code's spec
      of a queued run that never trained).
    - Decisions taken now, before any data, keep every result confirmatory.
    - **What the pilot needed answered** (all answered now). Every pilot run but the unconstrained
      one waited on Q-jc-window; the N = 0.50 runs also on Q-cost-critic; the Moderate run on
      Q-budget-normalisation, Q-level-jc and Q-search-before-pilot; the unconstrained run on
      Q-search-before-pilot and Q-pilot-unconstrained-seed. The pilot's results needed the result
      gates Q-hazard and Q-dynamics (the battery), Q-studyb-eval (the Moderate run's evaluation and
      G4) and Q-tie-break (only if a tie occurs). `python -m pilot design pilot -v` shows that none
      is open. `manifest.open_run_gates` is the one place that says which open questions hold a
      run: it combines the spec's `pending` keys (Q-rounding, Q-pilot-unconstrained-seed and the
      others a spec carries) with the run gates of `manifest.run_gate_keys`.
12. **[YOU] Record the machine-hour allocation** (X-allocation in `docs/DECISIONS.md`).
    - The group decides H at the ratification meeting: the elapsed wall-clock hours the workstation
      is dedicated to the registered runs of the whole design (all its concurrent run slots
      together, not core-hours and not hours per run), derived only from the calendar and the
      machine's availability, never from a timing of a run on the workstation.
    - Then run
      `python -m pilot allocate --hours H --basis TEXT --by "the group" --date YYYY-MM-DD` (or
      `python scripts/workstation.py allocation --hours H --basis TEXT --date YYYY-MM-DD`) and
      **commit** `pilot/allocation.json`; `--basis` records how H was derived. Record it before the
      determinism checks and smoke runs on the workstation, which already reveal its speed:
      `allocate` lists earlier workstation timings in the record (`--smoke-root DIR` names a smoke
      data root), is refused once a registered run has started, and needs `--confirm` above
      24 × 731 hours (likely core-hours).
    - Enter the same value in Table 8.1 ("Machine-hours allocated to the registered runs") with
      an amendment row in Table 9.1, before any pilot run: Part 6 G2 requires it "written into
      Part 8 before the pilot's throughput is read".
    - The scheduler starts no pilot run until HEAD contains that file.
    - The go report checks that every pilot run was launched from a commit holding exactly that
      allocation (Part 6 G2).

    12a. **[YOU] Commit a registered-form determinism report for every plug-in** (Table 3.1
    "Determinism check"; pilot/scheduler.py). The scheduler holds every registered run of plug-in
    `study_a`, `study_a_pid`, `study_b` or `unconstrained_ppo`, the pilot's included, until HEAD
    holds a passing registered-form report of that plug-in (`determinism_missing` in `schedule
    status`). From a clean commit (the keys each configuration waited on, named below, are
    answered):
    - `python scripts/determinism_check.py --registered-form --plugin study_a` (waited on
      Q-jc-window);
    - `--plugin unconstrained_ppo` (Q-pilot-unconstrained-seed);
    - `--plugin study_b` (Q-jc-window, Q-budget-normalisation, Q-level-jc, and Q-studyb-eval for
      its evaluation);
    - `--plugin study_a_pid` (Q-cost-critic, Q-jc-window, Q-pid-eq9 and Q-determinism-late-onset;
      the 15 PID-check runs wait on this report);
    - optionally `--registered-form` for `ppolag`.
    The pilot needs the first three. A configuration that waits on an open question is refused
    before anything trains. Commit each report under `ledger/determinism/`.

### Phase 3: what the pilot needs from the other roles (contracts in §8)

*Runbook.* Task 16: `python scripts/workstation.py search` shows what the record still lacks (a
people's step; the scheduler holds only the Study B runs until it is committed).

The code for tasks 13 to 15 and for Role 5's code in task 16 exists (task 4a); these tasks are now
the owners' review, adoption and duties.

13. **Role 2 (Environment and tests):** adopt `envs.onset` (the plug-ins `study_a`,
    `study_a_pid`), `envs.evaluation` (`evaluate_run`, `evaluate_battery`, `evaluate_checkpoint`,
    `evaluate_continuation`; disjoint selection and measurement seed sets, frozen normaliser; not
    OmniSafe's `Evaluator`, which is unseeded and updates the normaliser) and `envs.continuations`
    (`battery_finetune`, `battery_transfer`); confirm hazard relocation and dynamics perturbation
    as answered (Q-hazard, Q-dynamics).
14. **Role 3 (Metrics and interventions):** adopt `metrics/` (fixed batches, the three metrics,
    the hook writing `plasticity.csv` from the pilot's first Study A run, the interventions,
    controller quantities and recovery time); task 8a; read Nikishin et al. (2023) before the
    ratification meeting: the answer to Q-reset-injection rests on two search-engine extracts of it
    (§3.4).
15. **Role 4 (Analysis and results):** adopt `analysis/` (`analysis.matching.select_checkpoint`,
    rule 1, which the pilot code re-checks; rules 2–7; the Part 5 statistics) and
    `results/supplement_schema.py` (frozen at version 1 with `results/supplement_schema.sha256`,
    Q-ledger-v2); approve the ledger-schema amendment LS-1 to LS-32 by pull request before the first
    pilot ledger row is written (X-ledger-schema-amendment).
16. **Role 5 (Study B and literature):**
    - **Run the targeted search of Appendix C.** It has **not** been run: the files in
      `studyb/search/` are empty templates, and no code runs the search or writes a result. It needs
      normal internet access and Hamza Nisar's judgement. Follow `studyb/search/README.md`
      (`python -m studyb.search queries` prints the 80 searches; `python -m studyb.search check`
      exits 0 only when the record is complete). It must be finished and committed **before any
      Study B run, the pilot's two included** (Part 7.2; Q-search-before-pilot): the scheduler
      checks the record committed at HEAD (`search_record_missing`). An outcome of "included"
      withdraws Study B by amendment, so it holds the runs; a "partial" one holds them until its
      narrowing amendment is in the committed `analysis/AMENDMENTS.json` (`decision.json`'s
      `amendment` names its id).
    - Adopt `studyb.conditioning` (`study_b`, `study_b_fewshot`; budget after normalisation;
      per-level multipliers) and `studyb.evaluation` (`evaluate_run`, `evaluate_training_budgets`
      for G4, `evaluate_zero_shot`, `evaluate_fewshot`).
17. **[YOU] Enter the search record's path in Table 9.0** (`studyb/search/`, from Role 5) by an
    amendment row in Table 9.1, before the pilot's Study B runs (Part 7.2; Appendix C; First Tasks,
    Role 5 step 7).

### Phase 4: rehearsal

18. **[YOU] Run a single-seed smoke test of the whole pipeline** on short runs; the pre-registration
    allows these ("single-seed smoke tests of the pipeline") and they are not analysed. Use
    `python scripts/smoke_run.py --data-root /scratch/smoke` (one 1-epoch PPO-Lagrangian run),
    `--design sample` (2 epochs of one run of each training plug-in family: the N = 0 arm, an
    N = 0.50 arm with partial reset, a PID arm and the Moderate arm, so that the onset code, the
    reset at onset and the PID controller after onset are exercised) or `--design pilot --epochs 2`.
    Evaluation needs ten checkpoints in the final 2,000,000 steps, so runs shorter than 90 epochs
    end their evaluation `eval_failed` or refused; use `--epochs 90` or more to pass it.
    - The script exits 0 when every run ended as a smoke test expects, 3 when any run did not,
      and 1 when the data root is refused (for example `/data`).
    - It runs the real scheduler and launcher in smoke mode (`--allow-dirty --allow-pending` in
      both stages): the ledgers go to `/scratch/smoke/smoke/`, never to the repository, and the data
      root is fixed as a smoke root (`schedule run`, `resolve` and `set-policy` on it need
      `--allow-dirty`/`--allow-pending` and no other `--ledger-dir`; registered runs need another
      data root). It refuses `/data`.
    - Use the pilot design only to exercise the scheduler, never look at its costs, and say in the
      go report that it was run.
    - The command line cannot queue short runs, because `schedule add` takes only the registered
      designs.
19. **Role 4: rehearse the analysis script once on synthetic or smoke data**, before the pilot's
    data exist (its registered test run on the pilot data is task 25). The analysis tests already
    run it on synthetic ledgers with known answers (`tests/test_analysis_synthetic.py` builds them);
    a run on a smoke ledger needs `--allow-dirty` with an `--out` outside `results/`.

### Phase 5: the pilot (Part 3.6)

*Runbook.* Tasks 20 to 26: or `python scripts/workstation.py pilot-runs --max-concurrent K`, then
`pilot-enrich`, then `go [--hours-per-run H --concurrent K]`.

The pilot launches only when: the answers are ratified and merged (tasks 10 and 11); the
registration record is complete, the tag pushed and Tables 9.0 and 9.1 committed (tasks 1 and 3;
X-registration); `pilot/allocation.json` is committed (task 12); the registered-form determinism
reports of `study_a`, `study_b` and `unconstrained_ppo` are committed (task 12a); and, for its two
Study B runs, the complete search record is committed (task 16) and its path entered in Table 9.0
(task 17). `schedule status` names every gate that holds a run. The scheduler checks neither the
ratification nor the registration record, and the runbook checks only that the registered docx and
PDF changed at HEAD (`registration-record`), not the push or what the change holds. Smoke roots
with `--allow-pending` are the only way round the gates, and they are never registered data.

20. **[YOU] Queue and run the eight pilot runs.**
    - `python -m pilot schedule add --design pilot`
    - `python -m pilot schedule run --max-concurrent K`
    - Choose K as the number of runs the workstation can hold at once, usually physical cores
      minus 1.
21. **[YOU] Select the checkpoints.**
    `python -m pilot enrich select --ledger results/pilot/ledger.parquet` applies rule 1.
22. **[YOU] Measure and evaluate the battery.**
    `python -m pilot enrich measure --ledger results/pilot/ledger.parquet` writes the
    measurement-set cost of each matched checkpoint, then
    `python -m pilot enrich battery --ledger results/pilot/ledger.parquet` runs hazard relocation
    and dynamics perturbation (it also writes a missing measurement). Use the data root the ledger
    was written from (default `/data`): a run directory whose commit or matched checkpoint path
    differs from the row's is refused, and the log records the data root.
23. **[YOU] Measure the Moderate run.** `python -m pilot g4-measure` evaluates it at its three
    training budgets, from a clean commit, and writes `results/pilot/g4_moderate.json` once. Use the
    data root the ledger was written from: a run directory whose commit or final checkpoint path
    differs from the ledger row's is refused. The harness result must give a finite `mean_cost` and
    a satisfaction rate in [0, 1] at each of the three training budgets; otherwise nothing is
    written, so a faulty evaluation never uses up the once-only file. Its checkpoint and seeds
    follow Q-studyb-eval.
24. **[YOU] Write the go report.**
    - `python -m pilot go`, from a clean commit, writes a timestamped
      `results/pilot/go_report-<UTC>.{json,md}` with G1–G4, the reading of Part 6 and the Table 8.1
      values; the report records the commit and `worktree_dirty`. `--moderate PATH` names the
      g4-measure output; a path that does not exist is refused (only the default may be absent, and
      G4 is then not computable).
    - Concurrency is measured from the pilot runs' own start and finish times. G2 waits until all
      eight pilot runs are complete.
    - Each condition is read by its answer (`docs/DECISIONS.md`): G1 on each arm's mean
      (Q-g1-level) of the measurement-set cost (Q-matched-cost-set); G2 on the corrected
      run-equivalents (Q-g2-run-equivalents); G3 on the paired U80, the leftover runs paired in seed
      order when a replacement seed leaves fewer than three seed-matched pairs (Q-g3-pairing); G4 on
      Part 6's one-sided clause (Q-g4-level) and the measurement-set final cost (Q-final-cost-set).
      The other readings are reported beside them and decide nothing; a condition would show
      `UNDECIDED (<key>)` only while one of these keys was open. The headings read "Go condition G1
      (Part 6)" and so on, to keep them apart from Study B's hypotheses (Q-g-names).
25. **Role 4: commit the analysis script and run it once on the pilot data** (Part 5.8: it "is run
    once on the pilot data to test it"):
    `python -m analysis --mode pilot --ledger results/pilot/ledger.parquet` (from a clean commit;
    the output goes to a new `results/analysis/pilot-<UTC>/`). It never decides go: it recomputes the
    go report's G1 and G3 numbers and exits 3 if they disagree (an error to fix) and 4 if nothing
    could be compared. From the go decision on, it is edited only to fix errors, each recorded in
    Part 9 and in `analysis/ERRATA.json`.
26. **[YOU] Prepare the surplus-seed plan** (Part 5.5) for the go meeting:
    - `python -m pilot surplus --hours-per-run H --concurrent K` gives the surplus-seed plan on the
      corrected run-equivalents (`extra_seeds`, `seeds_per_primary_arm`, `added_seeds`: decisive,
      Q-g2-run-equivalents), with the plan under the registered 484 (`registered_*`) beside it,
      deciding nothing.
    - The seed count recorded in Part 8 at the go decision (task 27) is `seeds_per_primary_arm`;
      the E of task 31 is `extra_seeds`.

### Phase 6: the go meeting (Part 6)

*Runbook.* `python scripts/workstation.py status` shows every step DONE before the meeting; the
meeting itself is a people's step.

27. **Group, with you presenting.**
    - Enter the Table 8.1 values in Part 8, including the seeds per primary-comparison arm from
      task 26, and record the decision.
    - Commit the amended document (the row Table 9.1 labels "First amendment"; the amendment
      rows of tasks 2, 11, 12 and 17 are entered above it) and freeze Parts 1, 4, 5 and 6. Role 4
      records the go amendment in `analysis/AMENDMENTS.json` with `analysis_code_hash` (the Part 5.8
      freeze).
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
      failure has already made Study A a no-go. The analysis test run of the re-pilot is
      `python -m analysis --mode pilot --revision 1 --ledger results/pilot/ledger.parquet`.
    - G1 and G2 hold but G3 fails: a negative pilot with its U80 bound, and Study B becomes the
      main study.

### Phase 7: the main sweep of Study A (after go)

29. **Cut order first.** If G2 failed after the revision, decide the cuts in the order of Part 6.1
    (the `go --repilot` report prints it) before queuing anything. The cuts are
    `manifest.CUTS = ("pid", "car", "fewshot_short", "treatments_controllers_n050", "drop_n010")`,
    applied only as a prefix (never seeds, Study B's zero-shot arms, or the N = 0 / 0.50 arms of
    both controls on SafetyPointGoal1-v0 and SafetyPointButton1-v0). Commit the decision as
    `pilot/cuts.json` (`{"cuts": [...], "amendment": "..."}`) with its amendment; then every
    `schedule add` takes `--cuts K` (fixed per data root), and `python -m pilot budget --cuts K`
    prices it.
30. **Role 4: confirm the analysis script is committed** before the first main-study run completes
    (Part 5.8; committed in task 25). The scheduler holds every non-pilot run until HEAD holds
    `analysis/__main__.py` (`analysis_missing`).
31. **[YOU] Queue the main sweep and the additional arms.**
    - `schedule add --design main`, then `treatment`, `controller` and `pid`.
    - Then `schedule add-surplus --extra E` for the surplus seeds of task 26. E is a per-arm
      target fixed for the data root (Part 5.5: the same count for every arm); running it again
      with the same E adds only the seeds an arm lacks. Surplus seeds go to both onset shapes
      (Q-surplus-arm-set).
    - `schedule add`, `add-surplus` and `add-continuations` refuse an uncommitted tree or
      unverified imported code, and log a `queued` event with HEAD.
    - Warm-started runs wait for their N = 0 run, even if it is queued later. A warm-started
      replacement whose N = 0 run of that seed was never queued (seed 5 on a task without surplus
      seeds) waits for `schedule resolve RUN add-auxiliary`, which queues an auxiliary N = 0 run;
      `schedule status` lists it under the decisions. Auxiliary runs carry a note in the ledger and
      are left out of their arm by the matching and the analysis (Q-warm-start). If the N = 0 run a
      warm-started run copies from is excluded, the warm-started run, which never trained, is
      superseded by its arm's next unused seed: `schedule resolve RUN replace`.
    - The PID-check runs wait for the `study_a_pid` determinism report (Q-determinism-late-onset);
      `schedule status` shows why a run waits.
32. **[YOU] Run and monitor.** `schedule run --max-concurrent K` and `schedule status`.
    - A held interruption: `schedule resolve RUN restart` (same seed, on the commit the
      interrupted attempt ran on; it reproduces the run bit for bit, and waits while HEAD differs).
    - Registered data roots run under the `restart` policy (Q-interrupted-run): an interruption of
      machine origin is restarted without limit, and a run that ends without a result a third time
      while the scheduler watched it (SIGKILL, a non-zero exit) is excluded as incomplete and
      replaced. A data root whose policy was fixed before the key was answered needs
      `schedule set-policy restart` once; `exclude` is refused on a registered data root.
    - A failed evaluation: `schedule resolve RUN reevaluate` (the earlier `evaluation.json` is moved
      to `<data root>/scheduler/reevaluated/<run>-<n>.json`, named in the `reevaluate` event; an
      evaluate stage that exits nonzero is `eval_failed` even if it left an `evaluation.json`). A
      failed ledger write: `schedule resolve RUN retry-ledger`.
    - After two exclusions of one arm with the same cause the scheduler pauses that arm's
      replacements (Q-exclusion-breaker), an operational pause: read the failed runs' logs and
      tracebacks, never their costs or returns; fix a defect by a commit recorded as an erratum
      (`analysis/ERRATA.json`, Table 9.1; for a memory kill, lower K). The fix must be bit-for-bit
      neutral for every configuration already run, which you check with the determinism check on
      one completed configuration; if it is not, every completed run it affects is rerun on the
      corrected code, so that all arms are treated alike (Q-exclusion-breaker). The scheduler has no
      action for such a rerun (it never re-runs a completed run): raise it with the group before
      any affected result is used. Then `schedule resolve RUN replace` queues the arm's next unused
      seed. Once the arm's same-cause exclusions reach its
      seed target, `replace` needs `--defect-fixed ERRATUM_ID`; without a fixed defect the arm is
      reported as not having completed its seeds. A run blocked by an excluded dependency is
      replaced the same way (it becomes superseded); `schedule resolve RUN unblock` returns a
      blocked run whose dependencies are fine. A Study B
      continuation follows its parent's seed: for a blocked one, replace its parent (`schedule
      resolve B-…-sK replace`), which supersedes the old continuations and queues the new
      parent's; an excluded continuation is decided by amendment (the scheduler has no action for
      it).
    - Never re-run a completed run, except the reruns Q-exclusion-breaker requires after a
      non-neutral fix (above), and never exclude on results; the scheduler enforces both.
33. **[YOU] Enrich the ledger** as runs finish; it is safe while the scheduler runs, and it refuses
    to write results produced by uncommitted code (tracked changes, or an untracked file that it
    imported). An enrichment is refused if any ledger row would break the schema. Every command
    writes only empty fields and absent map keys; an existing value must be reproduced exactly.
    Run them in this order, each with `--ledger results/ledger.parquet` (E is the surplus target of
    task 31):
    1. `enrich select` (rule 1);
    2. `enrich measure` (C_ID of every matched checkpoint);
    3. `enrich match --surplus-extra E` (rules 2–6; an incomplete arm waits, Q-arm-complete);
    4. `enrich battery` (hazard and dynamics gaps, matched rows only, rule 6);
    5. `enrich controller` and `enrich training` (Table 2.4 quantities, recovery time, plasticity at
       onset and at onset + 200,000 steps, the intervention summary; Q-controller-quantities);
    6. `enrich final-battery` (rule 7 (b): every final checkpoint) and
       `enrich sensitivity-battery --surplus-extra E` (rule 7 (a): arms matched at 5.0 but not 2.5);
    7. `schedule add-continuations --ledger results/ledger.parquet` (the fine-tuning and transfer
       continuations of every matched row, once its arm is complete), then `schedule run` until they
       end `continued`;
    8. `enrich continuations` (gap_finetune, gap_transfer).
    `python -m pilot completeness` shows, per arm, what is still missing (read-only). A command
    that waits on an open question for some runs writes the others and exits 3 naming the runs
    that wait.

### Phase 8: Study B (after Study A's main sweep; after the targeted search)

34. **[YOU] Queue and run Study B.**
    - `schedule add --design study_b`, then `study_b_fewshot` (140 continuations, which wait for
      their parent). A non-pilot Study B run starts only when every run of
      the main sweep has finished training (`main_sweep_unfinished`; Q-studyb-order).
    - Run `schedule add-surplus --extra E` again with the E of task 31: it adds the Study B arms'
      surplus seeds and their continuations.
    - A continuation ends as `continued`. Then `enrich zeroshot` (sr_zero) and `enrich fewshot`
      (sr_fewshot and adapt_steps, from each budget's own continuation), both evaluated as
      Q-studyb-eval decides; a budget that never reaches the target records the largest horizon + 1
      (Q-adapt-censoring).

### Phase 9: analysis and reporting

35. **Role 4: final analysis.**
    `python -m analysis --mode final --ledger results/ledger.parquet --surplus-extra E`, from a
    clean commit: primary and secondary estimands, both intervals, the sensitivity analyses of rule
    7, Holm families, nulls as bounds, and exploratory labels (`analysis/AMENDMENTS.json`). A
    verdict is PROVISIONAL or UNDECIDED (key) only while a key it depends on is open; with every key
    answered, none is. The output goes to a new `results/analysis/final-<UTC>/`.
36. **[YOU] Freeze and publish the data.** `python -m pilot freeze` prints the manifest of the
    ledger, supplement and analysis file hashes and the analysis changes since the go report the
    decision was taken from (the earliest committed one of the last pilot revision; read-only; Part
    5.8). Commit the final ledgers and supplements, archive `/data/checkpoints` and the scheduler
    database outside the repository, and publish a release with a DOI.
37. **All: write the paper.** Report the matched and final-checkpoint estimands side by side, name
    every limit on causal interpretation (Part 4.1.1), and list every amendment with its date and
    whether data had been seen.

---

## 5. Repository map: what each file is for

Owners: You = Role 1 (Muhammad Umair Waseem); Role 2 = Muhammad Talha Jamil; Role 3 = Abdullah;
Role 4 = Muhammad Abdullah; Role 5 = Hamza Nisar. Every file marked *new* was written by the pilot
owner's build on 2026-10-01 or 2026-10-02 and is uncommitted; its owner adopts it (task 4a).

| Path | Owner | What it is and what you do with it |
|---|---|---|
| `prereg/Preregistration.{docx,pdf}` | Group | The registered plan (commit `735b394`). Change only by amendment; fill Table 9.0/9.1 (task 3). |
| `README.md` | You | Description, link to `prereg/`, state of the code, structure, owners and quick start (First Tasks, Role 1 step 4). |
| `HANDOVER.md` | You | This file. `tests/test_registered.py` checks its section 9 stars. |
| `docs/DECISIONS.md` (*new*) | Group (prepared for you) | The answers to every open question (2026-10-02), the draft rows of Table 9.1 and the values of Table 9.0; the group ratifies it at one meeting (tasks 10 and 11). |
| `docs/DECISIONS_EVIDENCE.md` (*new*) | Group | Generated: the verbatim reasoning behind each answer (decider, critic, judge, coordinator). Not edited by hand. |
| `pyproject.toml` | You | Python 3.10; pytest configuration (`pythonpath`, `-m 'not slow'` by default, markers `omnisafe`, `slow`). |
| `.gitattributes` | You | LF line endings in the repository, so file hashes are the same on every OS; `*.npy` binary. |
| `.gitignore` | You | Excludes run outputs, checkpoints, virtual environments (not `metrics/fixed_batch/`). |
| `.github/CODEOWNERS` | You | Fill usernames, uncomment (task 5). |
| `environment/requirements.in` | You | The direct pins, with the reason for each (scipy 1.15.3 added for the analysis). Change only by amendment. |
| `environment/requirements.lock.txt`, `environment/workstation.json` | You | *Created on the workstation* by `scripts/setup_env.sh`. Commit them. |
| `environment/fixed_batch_check.json` | You | *Created on the workstation* by `python scripts/workstation.py fixed-batch` whenever `--check` runs to a result: whether it passed, the SHA-256 of every batch and, if the re-collection differs, the re-collected ones (task 8a; Q-appendix-a). Commit it. |
| `configs/registered.py` | All (you created it) | Every registered number with its location; `PENDING` (the 54 questions of §9); `ANSWERED_QUESTIONS` (all 54, answered in `docs/DECISIONS.md`, to be ratified). Other roles add their values by PR. |
| `configs/omnisafe/{PPOLag,CPPOPID,PPO}.yaml`, `SOURCE.json` | You | Verbatim copies from omnisafe 0.5.0 with version and SHA-256. Never edit. |
| `pilot/manifest.py` | You | Every registered run as a `RunSpec` with a deterministic `run_id` (`A-PointGoal1-N0.50-abrupt-total-s3`, `B-Dense-s2`, `P-…` for the pilot); `PLUGINS`; `run_gate_keys` and `open_run_gates` (the one place that says which open questions hold a run: the spec's `pending` keys with `run_gate_keys`); the battery continuations (`battery_continuation`, `CONTINUATION_GROUPS`, `TRANSFER_TASK_PROPOSALS`); the cut order (`CUTS`, `apply_cuts`); `AUXILIARY_NOTE`. |
| `pilot/launch.py` | You | One run. `train`: resolves dependencies, then refuses (exit 5, nothing written) on open questions, existing output, a claimed directory, uncommitted code, an installed YAML other than the committed copy, or defaults that differ from Table A.1; builds the config as `omnisafe.Agent` does with only the registered overrides plus `pilot_cfgs`; enforces one thread; claims the run directory; classifies the outcome by Part 5.6; writes `train_result.json`. `evaluate`: the contract-1 harness for the run (`contracts.evaluator_target`), checked, writes `evaluation.json`. A missing repository module is exit 4 (unavailable) in both stages. |
| `pilot/rundir.py` (*new*) | You | Run-directory layout and readers (`read_progress`, `find_omnisafe_run_dir`, the file names); a leaf module that `pilot/launch.py` re-exports. |
| `pilot/errors.py` (*new*) | You | `RunRefused`, `PendingQuestionError`, the result gate `require_answered` and its smoke-only bypass (`allow_pending`). |
| `pilot/dependencies.py` (*new*) | You | Runs that start from other runs: `resolve` (before the claim), `to_config`/`from_cfgs`, `Dependency.multiplier_at` (warm start), `load_full_state`, and `restore_learner`, the one restore rule of every continuation (Q-continuations). |
| `pilot/algorithms.py` | You | Plug-ins for plain PPO-Lagrangian and PPO. `FullStateCheckpointMixin`: the learner state in every checkpoint (networks, optimisers, scheduler, multiplier, the 50-episode windows; not the random number generators), the extra checkpoints (onset, onset + 200,000 steps, few-shot horizons, an end-relative window), the per-epoch batch cost and return (Part 3.4), the plasticity hook's installation, without changing training. Roles 2 and 5 mix it into their classes. |
| `pilot/contracts.py` | You | The interfaces the other roles implement, and their checks (checkpoint set, extra checkpoints, selection window, evaluation seeds, `validate_plasticity`, `fewshot_key`). |
| `pilot/scheduler.py` | You | SQLite-backed scheduler with the launch gates (see §7). |
| `pilot/ledger_writer.py` | You | The only writer of ledger rows, with locked atomic writes, and of the `evaluation` supplement record. Pilot runs go to `results/pilot/ledger.parquet`; the unconstrained pilot run to a sidecar; smoke runs are refused by the repository ledgers. |
| `pilot/supplement.py` (*new*) | You | The supplement writer: one JSON file per (kind, run, part) beside a ledger, validated, written once, never overwritten. |
| `pilot/enrichment.py` | You | Every enrichment command (`select`, `measure`, `battery`, `match`, `controller`, `training`, `continuations`, `zeroshot`, `fewshot`, `final-battery`, `sensitivity-battery`), each result checked at the boundary and logged. |
| `pilot/budget.py` | You | Run-equivalents (registered 484 and corrected 627, after cuts too), the allocation record and its containment check, the surplus-seed plan. |
| `pilot/go_decision.py` | You | G1–G4 (G1 on arm means with the per-seed reading beside it), equation (12) (paired, with the pooled reading beside it), measured concurrency, the reading of Part 6 (first pilot, and re-pilot given the conditions that failed first), the Table 8.1 values, timestamped reports that are never overwritten. |
| `pilot/provenance.py` | You | Commit hash, clean-tree rule (exact output files only), imported-code check, configuration hash (dependency paths volatile), machine, UTC time. |
| `pilot/__main__.py` | You | The command line (§7). |
| `pilot/README.md` | You | Owner README. |
| `pilot/allocation.json` | You | *Created* by `python -m pilot allocate` (task 12). |
| `pilot/cuts.json` | You | *Created* only if the group decides cuts (task 29). |
| `scripts/setup_env.sh`, `copy_omnisafe_configs.py`, `record_workstation.py`, `determinism_check.py`, `smoke_run.py` (*new*), `tag_registration.sh` | You | Environment, config copies, workstation record, determinism check per plug-in, smoke test of the pipeline (task 18), registration tag. |
| `scripts/workstation.py` (*new*) | You | The runbook: one command per remaining step of §4, each step's state read from HEAD (§7). Standard library only. |
| `ledger/determinism/` | You | Determinism reports (First Tasks path; tasks 9 and 12a). |
| `envs/__init__.py`, `envs/onset.py` (*new*) | Role 2 | Study A plug-ins `study_a` (`make_algorithm`) and `study_a_pid` (`make_pid_algorithm`): onset, ramp, warm start, rate limit, PID, and the timing of reset and injection. |
| `envs/evaluation.py` (*new*) | Role 2 | The seeded evaluation harness (contracts 1, 5, 6; `evaluate_continuation`; hazard relocation and dynamics perturbation; the primitives Study B reuses). |
| `envs/continuations.py` (*new*) | Role 2 | The battery's continuation plug-ins `battery_finetune` and `battery_transfer` (with the observation mapping of transfer). |
| `envs/README.md` | Role 2 | Owner README. |
| `metrics/__init__.py`, `metrics/plasticity.py` (*new*) | Role 3 | The three metrics of Table 2.3 (and the trainable-layer and critic variants). |
| `metrics/hook.py` (*new*) | Role 3 | Writes `plasticity.csv` and `plasticity_meta.json` at every saved checkpoint. |
| `metrics/interventions.py` (*new*) | Role 3 | Partial reset, plasticity injection (`InjectedHead`), `rebuild_injected`, `build_actor`, `intervention_summary`. |
| `metrics/batches.py`, `metrics/collect_fixed_batch.py`, `metrics/fixed_batch/` (*new*) | Role 3 | The fixed batches: their SHA-256 constants, the collection script (`--check`), and the three `.npy` files with `MANIFEST.json` (generated in the sandbox; task 8a). |
| `metrics/controller.py`, `metrics/recovery.py` (*new*) | Role 3 | Overshoot and settling time (Table 2.4); recovery time (Table 2.1). |
| `metrics/recompute.py` (*new*) | Role 3 | Recomputes a run's plasticity rows from its checkpoints (an audit of contract 2). |
| `metrics/README.md` | Role 3 | Owner README. |
| `analysis/__init__.py`, `__main__.py` (*new*) | Role 4 | `python -m analysis --mode pilot|final` (Part 5.8). |
| `analysis/matching.py`, `stats.py`, `study_a.py`, `study_b.py` (*new*) | Role 4 | Part 4.1 rules 1–7 (contract 4); the Part 5 statistics; boxes H0–H4 and G1–G5 with the secondary outcomes. |
| `analysis/data.py`, `questions.py`, `verdict.py`, `records.py`, `report.py`, `pilot_check.py` (*new*) | Role 4 | Ledger and supplement loaders; which open keys each verdict depends on; verdicts and readings; the amendment and errata registries; the report; the Part 5.8 pilot consistency check. |
| `analysis/AMENDMENTS.json`, `analysis/ERRATA.json` (*new*) | Role 4 | Registries of amendments (Part 8.2 exploratory labels; the go freeze hash), holding `A1`, the decisions of 2026-10-02 (commit not yet recorded), and of post-go errata (Part 5.8; empty). |
| `analysis/README.md` | Role 4 | Owner README. |
| `results/ledger_schema.py`, `.sha256`, `_requirements.txt`, `tests/test_ledger_schema.py`, `docs/ledger_schema_memo.md` | Role 4 | Frozen ledger schema v1, with the corrections of 2026-10-01 (amendment LS-1 to LS-32, decided 2026-10-02 as X-ledger-schema-amendment, to be ratified; §3.3). Do not edit without an amendment. |
| `results/supplement_schema.py` (*new*) | Role 4 | Pydantic models of the supplement records (§8); never touches the frozen schema. |
| `results/supplement_schema.sha256` (*new*) | Role 4 | SHA-256 of the supplement schema, frozen at version 1 (Q-ledger-v2); `sha256sum -c results/supplement_schema.sha256` from the root. Change only by amendment. |
| `results/ledger.parquet`, `results/pilot/`, `results/supplement/`, `results/pilot/supplement/`, `results/analysis/` | You (writer); Role 4 (analysis outputs) | Ledgers, created as runs complete, with `*.seeds.json` (the evaluation seed sets, fixed by the first run) and `*.enrichment_log.jsonl`; the supplement records; the analysis outputs. Commit them regularly. |
| `studyb/__init__.py`, `studyb/conditioning.py` (*new*) | Role 5 | Plug-ins `study_b` and `study_b_fewshot`: budget conditioning after normalisation, per-level multipliers. |
| `studyb/evaluation.py` (*new*) | Role 5 | Contract 1 for budget-conditioned runs, G4's measurement, zero-shot and few-shot evaluations, equations (10) and (11). |
| `studyb/search/` (*new*) | Role 5 | `__init__.py`, `protocol.py`, `record.py`, `__main__.py`, `README.md` and the empty record templates (`search_log.md`, `search_log.csv`, `search_record.csv`, `citations_log.csv`, `screened/` with its `README.md`). The search is not run (task 16). |
| `studyb/README.md` | Role 5 | Owner README. |
| `tests/` | Each test's owner | The recorded passing runs in §12 are from an earlier tree of 1,756 tests; the tree has grown since (over 2,400 tests on 2026-10-02; `pytest --collect-only -q -m ''` counts them). `tests/test_core_*.py`, `tests/test_integration_*.py` and `tests/test_workstation_runbook.py` (You, *new*), `tests/test_envs_*.py` (Role 2, *new*), `tests/test_metrics_*.py` (Role 3, *new*), `tests/test_analysis_*.py` and `tests/test_supplement_schema.py` (Role 4, *new*), `tests/test_studyb_*.py` (Role 5, *new*), the earlier pilot tests (You), `tests/conftest.py` (You, *new*), `tests/test_ledger_schema.py` (Role 4); `tests/fake_launcher.py` simulates runs for the scheduler and enrichment tests. |

---

## 6. Setting up

**Workstation requirements.**
- Linux (native or WSL2 Ubuntu) with Python 3.10 and Git.
- Registered runs are CPU-only with one thread each, so throughput scales with the number of
  physical cores.
- A full-state checkpoint is about 0.3 MB. A 10M-step run keeps 51 of them (more when its onset or
  the manipulation-check step falls off the ten-epoch grid), about 16 MB, and the whole study needs
  about 10 GB of checkpoints. Plan at least 50 GB under `/data`, with backups.

```bash
# only after the role pull requests are merged; until then unpack the archive instead
git clone https://github.com/paf-iast-ai-research/when-safety-comes-late.git
cd when-safety-comes-late
bash scripts/setup_env.sh            # first time on the workstation (creates the lock file)
source .venv/bin/activate
pytest -q                            # the fast tests
pytest -q -m slow                    # the slow tests (short training; launcher == omnisafe.Agent bit for bit); on the workstation after the allocation (task 12)
python scripts/determinism_check.py  # from a clean commit, after the allocation (task 12); writes ledger/determinism/
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

Every command below was checked against its `--help` on 2026-10-02.

**The runbook: `scripts/workstation.py`.** Once the group has ratified the answers (tasks 10 and
11), the pilot owner can run every remaining step of §4 that runs code with one command each (the
people's steps it does not check are listed below). The runbook reads
whether a step is done from what is committed at HEAD (never the working tree), and calls the
commands below with their real flags. Each command is printed as `+ cmd` before it runs, from the
repository root, with the interpreter that runs the runbook. It never commits, tags or pushes: a
step whose output must be committed ends by printing the exact `git add` and `git commit -m "..."`
lines. It needs only the standard library, so it runs with `python3.10` before `.venv` exists. Run
every step after `env` in `.venv`.

```bash
python3.10 scripts/workstation.py status        # read-only: each step DONE / TODO / BLOCKED (why), and the next command
python3.10 scripts/workstation.py next          # run the first step not done; stops at a people's step (takes every step option)
python3.10 scripts/workstation.py registration  # task 1: bash scripts/tag_registration.sh; prints `git push origin registration`
python3.10 scripts/workstation.py env           # task 8: bash scripts/setup_env.sh; refused once the lock exists (--locked is for the others)
source .venv/bin/activate
python scripts/workstation.py allocation --hours H --basis TEXT --date YYYY-MM-DD [--by "the group"] [--confirm] [--smoke-root DIR]   # task 12; without --hours it prints what the group decides and exits 2
python scripts/workstation.py fixed-batch       # task 8a: collect_fixed_batch --all --check from a clean tree; writes environment/fixed_batch_check.json whenever the check runs to a result (a mismatch: done with a warning, Table 9.1)
python scripts/workstation.py determinism-default                 # task 9: determinism_check.py (ppolag, bitwise form); refused until pilot/allocation.json is committed
python scripts/workstation.py determinism [--plugins P ...]       # task 12a: study_a, unconstrained_ppo, study_b, study_a_pid; refused until pilot/allocation.json is committed; refuses a plug-in whose keys are open, naming them
python scripts/workstation.py search            # task 16 (Role 5's): the instructions, and `python -m studyb.search check` (read-only)
python scripts/workstation.py registration-record   # task 3 (a people's step): what Tables 9.0 and 9.1 and the tag push still need (read-only)
python scripts/workstation.py pilot-runs --max-concurrent K [--data-root /data]   # task 20: schedule add (skipped once queued), schedule run, schedule status; long-running: tmux
python scripts/workstation.py pilot-enrich [--data-root /data]                    # tasks 21-23: enrich select, measure, battery; g4-measure (taken once)
python scripts/workstation.py go [--hours-per-run H --concurrent K]               # tasks 24-26: go, analysis --mode pilot (run once), surplus; then the go meeting (task 27)
```

**When a step counts as done (read at HEAD):**
- the tag `registration` points at the commit in `scripts/tag_registration.sh` (a local tag: the
  runbook does not check that it is pushed);
- the lock and `environment/workstation.json` are committed;
- `pilot/allocation.json` is committed;
- `environment/fixed_batch_check.json` records the check's outcome (a pass, or a re-collection that
  differs: done with a warning, Q-appendix-a) with the sha256 values of
  `metrics/fixed_batch/MANIFEST.json`;
- a passing bitwise `ppolag` report is committed under `ledger/determinism/`;
- `pilot.scheduler.committed_determinism_reports` holds each plug-in;
- `search_record_status` releases the gate;
- `prereg/Preregistration.docx` and `.pdf` at HEAD differ from the registration commit's (Tables 9.0
  and 9.1 filled; the runbook cannot see whether the tag is pushed);
- every pilot run is finished on the data root (its scheduler state, read-only), with no decision
  pending;
- `results/pilot/ledger.parquet` and `g4_moderate.json` are committed;
- a go report and the analysis test run are committed, and the latest analysis output passed its
  test (`python -m analysis` exit 0).

**What the runbook does not check.** The push of the tag (tasks 1 and 3: it cannot see the remote);
what the committed change of the registered docx and PDF holds (Tables 9.0 and 9.1 and nothing
else, task 3); the search record's path in Table 9.0 (task 17); and the ratification and merge
(tasks 10 and 11). `next` stops at `registration-record` until the docx and PDF are committed
changed, then goes on to `pilot-runs`: do the rest first, since the registration is complete before
any pilot run (X-registration).

**Where `next` does not stop.** It stops at a people's step, except where the scheduler itself holds
the runs that wait: an incomplete search record holds only the pilot's two Study B runs, and a
determinism check other than study_a's that an open key refuses holds only that plug-in's runs. In
those cases it reports the step and goes on. With every key answered no check is refused, so `next`
runs every missing determinism check.

**Exit status of the runbook:** 0 done; 2 a usage error or a value only the group or operator can
give; 3 the step cannot go on now (a people's step, open keys, an uncommitted tree, an output that
must not be written twice); otherwise the exit status of the called command that failed; 130 Ctrl-C.

**The commands themselves.**

```bash
python -m pilot design pilot -v          # the 8 pilot runs, their steps, onsets, open questions
python -m pilot design all               # 610 runs, 540.13 run-equivalents of training
python -m pilot budget                   # registered 484 vs corrected 627.13 run-equivalents
python -m pilot budget --cuts 2          # what-if: the design after the first 2 cuts of Part 6.1

python -m pilot allocate --hours 12000 --basis "24 h x 500 days, workstation dedicated" --by "the group" --date 2026-10-05   # then COMMIT pilot/allocation.json

python -m metrics.collect_fixed_batch --all --check       # task 8a: re-collect and compare the bytes
python scripts/determinism_check.py                       # task 9 (plug-in ppolag)
python scripts/determinism_check.py --registered-form --plugin study_a   # task 12a; also study_b, study_a_pid, unconstrained_ppo

python -m pilot schedule add --design pilot
python -m pilot schedule run --max-concurrent 7            # long-running; use tmux or nohup
python -m pilot schedule status --events 30
python -m pilot schedule resolve P-A-PointGoal1-N0.00-s1 restart     # a held interruption
python -m pilot schedule resolve P-A-PointGoal1-N0.00-s1 reevaluate  # after a failed evaluation
python -m pilot schedule resolve P-A-PointGoal1-N0.00-s1 retry-ledger
python -m pilot schedule set-policy restart                          # once, on a data root made before Q-interrupted-run was answered
python -m pilot schedule resolve P-A-PointGoal1-N0.00-s5 replace     # after the circuit breaker (Q-exclusion-breaker); --defect-fixed ERRATUM_ID at the seed target
python -m pilot schedule resolve RUN add-auxiliary                   # Q-warm-start: queue the N = 0 run a warm-started replacement needs
python -m pilot schedule resolve RUN requeue                         # store the committed code's spec of a run that never trained

python -m pilot enrich select  --ledger results/pilot/ledger.parquet
python -m pilot enrich measure --ledger results/pilot/ledger.parquet
python -m pilot enrich battery --ledger results/pilot/ledger.parquet --conditions hazard dynamics
python -m pilot g4-measure
python -m pilot go                      # re-pilot: go --repilot G1 (the conditions that failed first)
python -m analysis --mode pilot --ledger results/pilot/ledger.parquet   # Role 4, Part 5.8 test run
python -m pilot surplus --hours-per-run 12.5 --concurrent 7

python -m pilot schedule add --design main                 # with --cuts K once pilot/cuts.json is committed
python -m pilot schedule add-surplus --extra 2             # per-arm target; repeat after queuing Study B
python -m pilot enrich select   --ledger results/ledger.parquet
python -m pilot enrich measure  --ledger results/ledger.parquet
python -m pilot enrich match    --ledger results/ledger.parquet --surplus-extra 2
python -m pilot enrich battery  --ledger results/ledger.parquet
python -m pilot enrich controller --ledger results/ledger.parquet
python -m pilot enrich training   --ledger results/ledger.parquet
python -m pilot enrich final-battery --ledger results/ledger.parquet
python -m pilot enrich sensitivity-battery --ledger results/ledger.parquet --surplus-extra 2
python -m pilot schedule add-continuations --ledger results/ledger.parquet   # then schedule run
python -m pilot enrich continuations --ledger results/ledger.parquet
python -m pilot enrich zeroshot --ledger results/ledger.parquet
python -m pilot enrich fewshot  --ledger results/ledger.parquet
python -m pilot completeness            # per arm: what is still missing (read-only; --json)
python -m analysis --mode final --ledger results/ledger.parquet --surplus-extra 2   # Role 4
python -m pilot freeze                  # hashes of ledger, supplement and analysis (read-only; Part 5.8)

python -m studyb.search queries         # Role 5: the 80 searches of the protocol
python -m studyb.search check           # exit 0 only when the record is complete
python -m metrics.recompute --run-dir /data/checkpoints/RUN --out /tmp/RUN-plasticity.csv   # audit of plasticity.csv
python scripts/smoke_run.py --data-root /scratch/smoke --design sample      # task 18
```

**Exit status of `python -m pilot`.** 0 done (`enrich` also prints what it left waiting for its
data); 1 an expected failure (a bad value, a missing file, a refused write, another role's code not
in the repository); 2 a usage error or a refused provenance check (for example an uncommitted tree; nothing more is
written, though `enrich` keeps the rows it wrote before the refusal); 3 refused by an open
pre-registration question or a refusing step: the step waits for the amendment log, and nothing that
depends on it is written; 130 interrupted (Ctrl-C).
`python -m analysis`: 0 done; 2 refused, nothing written; 3 pilot mode found the analysis and the go
report disagree on a number; 4 pilot mode compared nothing. `--allow-dirty` and `--allow-pending` of
`enrich` are for smoke ledgers only (and `--allow-pending` only with `--allow-dirty`);
`--allow-dirty` of `analysis` is refused for an output inside `results/`.

**How the scheduler behaves.**
- **Run layout.** Each run lives in `/data/checkpoints/<run_id>/`:
  - `spec.json` and `.claim`;
  - `omnisafe/<algo>-{<task>}/seed-XXX-<time>/` (`progress.csv`, `config.json`,
    `torch_save/epoch-{k}.pt` = state after k epochs, `plasticity.csv` and `plasticity_meta.json`
    for Study A training runs, `intervention.json` for reset and injection runs,
    `continuation.json` for continuations);
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
  command completes it. Under the `exclude` policy (smoke data roots only) a run still held (a
  `set-policy exclude` cut short) is excluded. A run that dies from a signal while no scheduler is
  its parent leaves no exit status: it counts as an interruption of machine origin, never as a
  crash. One workstation per data root.
- **Interruptions.** The interruption policy is fixed per data root and applies to every run alike;
  `schedule run` uses the stored policy, changed only with `schedule set-policy`. A registered data
  root runs under `restart` (Q-interrupted-run): it is the default of its first run once the key is
  answered, a root whose policy was fixed while the key was open (stored `hold`) needs `schedule
  set-policy restart` once, `exclude` is refused, and `hold` only pauses it; a smoke data root may
  use any policy (default `hold`). Under `restart` an interruption of machine origin (the launcher
  recorded SIGTERM, SIGINT or SIGHUP, the child died of one of those signals, or it ended while no
  scheduler was its parent) is restarted without limit; one of process origin (SIGKILL, e.g. the OOM
  killer, another non-crash signal, or a non-zero exit without a final result) is restarted at most
  `MAX_RESTARTS` = 2 times, and a third such ending "fails to complete its steps" (Part 5.6): the
  run is excluded with cause `incomplete` and replaced, and the exclusion counts towards the circuit
  breaker. A restart is from scratch, with the same seed, on the commit its interrupted attempt ran
  on, as recorded in its `train_result.json` (it waits while HEAD differs). A held run can be
  restarted with `schedule resolve RUN restart`, but never excluded one by one. A restart that
  cannot happen (a live process, an archive already present, a file-system error while archiving)
  holds the run with a `restart_failed` event and the scheduler goes on. `schedule resolve RUN
  restart` restarts a run only if it is still held when its write transaction starts, and the policy
  (`set-policy`, or a scheduler start under `exclude`) restarts or excludes a held run only if it is
  still held then: a run another command acted on first (for example restarted and launched again)
  is left as it is (`restart_skipped`, `exclusion_skipped`). A signal that arrives after the last
  epoch and the final checkpoint does not undo a complete run. Only the first SIGTERM, SIGINT or
  SIGHUP from the moment the launcher's handlers are installed (right after the claim) to the end of
  training interrupts it (exit code 3, `train_result.json` status `interrupted`); any other signal
  (a second one arriving at the same time, or one after training ended) is only recorded. On every
  exit of the launcher's training stage, even when writing `train_result.json` fails, the original
  handlers are restored and the recorded signals are delivered to them, after `train_result.json` is
  written when it can be (a thread mask cannot do this: OmniSafe's TensorBoard writer thread would
  receive the signal). An interruption never replaces a non-finite loss or multiplier that the log
  already shows: that run is excluded with that cause.
- **Crashes.** A crash (Python error, segfault), an incomplete run, or a non-finite loss or
  multiplier is **excluded**. The exclusion, its replacement and its dependents' new states are one
  database transaction; the ledger record of the exclusion is written right after and retried at
  every pass until it succeeds. The ledger writer leaves non-finite plasticity metrics null (named
  in the notes), so a NaN run's row and its exclusion are still written. The replacement takes the
  arm's smallest unused seed of 5 or more (Q-seed-collision). After two exclusions of one arm with
  the same cause, no further replacement is added until the operator acts (an operational pause,
  Q-exclusion-breaker; task 32); an exclusion for a third ending of process origin under `restart`
  counts. Exclusions applied by the `exclude` policy (smoke data roots only) do not count, except
  that after two successive replacements stopped by that policy (each replacing the last) the arm
  pauses too. A replaced Study B run's
  continuations are superseded by its replacement's. An excluded continuation is not replaced (it
  follows its parent's seed); the group decides by amendment.
- **Continuations.** Study B's few-shot continuations and the battery's fine-tuning and transfer
  continuations end as `continued`; their results reach the parent's row only through enrichment.
- **Missing pieces.** A run whose plug-in or evaluator is not in the repository waits, and is
  never excluded for it. Each evaluation harness is tracked by its target, so a missing Study B
  harness holds only the budget-conditioned runs.
- **Refusals.** The launcher refuses, before it writes anything, a run with open questions, a
  dependency that is not a completed run, a directory that already holds output or is claimed,
  uncommitted code (tracked changes, or untracked code the run imports), an installed OmniSafe YAML
  that is not the committed copy, or defaults that differ from Table A.1. A refused run stays
  pending and nothing in its directory is touched; a refusal is never an exclusion (Part 5.6
  excludes only run outcomes). A launcher error before the claim leaves the run pending until the
  next scheduler start. In registered mode, training and evaluation both wait while tracked code has
  uncommitted changes.
- **Launch gates** (registered mode; each read from HEAD and shown by `schedule status`):
  - pilot runs start only when HEAD contains `pilot/allocation.json`;
  - `search_record_missing`: every Study B run, the pilot's included, waits for a complete search
    record (`studyb.search.check_record`) whose outcome is not "included";
  - `main_sweep_unfinished`: a non-pilot Study B training run waits until every run of the main
    sweep has finished training (Q-studyb-order);
  - `determinism_missing`: a run of plug-in `study_a`, `study_a_pid`, `study_b` or
    `unconstrained_ppo` waits for a passing registered-form report of that plug-in under
    `ledger/determinism/`; the report must have trained both runs on its own commit
    (`run_commits`, `same_commit`);
  - `analysis_missing`: a non-pilot run waits for `analysis/__main__.py`.
- **Modes.** A data root's mode (smoke or registered) and ledger location (`--ledger-dir`, taken
  only by `run`, `resolve` and `set-policy`, or the default; compared as real paths, and relative to
  the clone for the repository's own ledgers, so a symlinked root or a fresh clone is the same
  location) are fixed by its first run or decision; `run`, `resolve` and `set-policy` refuse a
  different one. Smoke ledgers may not lie inside the repository. `schedule add`, `add-surplus`
  and `add-continuations` fix the data root's mode as registered (so `run`, `resolve` and
  `set-policy` with `--allow-dirty`/`--allow-pending` are then refused on it) and refuse a smoke
  data root.
  `status`, `resolve` and `set-policy` refuse a data root without scheduler state. A `<ledger>.mode`
  marker records the mode of each ledger outside the repository's `results/` (the repository's
  ledgers are registered by location and carry none); an empty or unknown marker is a ledger write
  error naming the file (remove or fix it).

---

## 8. Contracts between the roles

The code of every contract exists (task 4a); the checks on the pilot owner's side are in
`pilot/contracts.py` (numbered contracts 1 to 6), `pilot/ledger_writer.py`, `pilot/enrichment.py`
and `results/supplement_schema.py`.

**Plug-ins** (`pilot.manifest.PLUGINS`; the launcher calls `factory(spec.task, cfgs, spec)`):

| Plug-in | Function | Owner | Runs |
|---|---|---|---|
| `ppolag`, `unconstrained_ppo` | `pilot.algorithms:make_ppolag`, `make_unconstrained_ppo` | Role 1 | determinism check; the pilot's unconstrained run |
| `study_a`, `study_a_pid` | `envs.onset:make_algorithm`, `make_pid_algorithm` | Role 2 | main sweep, treatments, controller variants, pilot Study A; the PID check |
| `battery_finetune`, `battery_transfer` | `envs.continuations:make_finetune_algorithm`, `make_transfer_algorithm` | Role 2 | the battery's continuations (Table 2.2) |
| `study_b`, `study_b_fewshot` | `studyb.conditioning:make_algorithm`, `make_fewshot_algorithm` | Role 5 | Study B arms (pilot Moderate included); few-shot continuations |

Rules for every plug-in: subclass the OmniSafe algorithm with
`pilot.algorithms.FullStateCheckpointMixin`. Study A changes only the actor surrogate
(`_compute_adv_surrogate`) and the multiplier update (`_update`) at onset, never `use_cost`.
Anything added to checkpoints must load with `torch.load(weights_only=True)` (contract 3). A
configuration or spec problem raises `pilot.errors.RunRefused` (exit 5, the run stays pending); a
missing repository module is exit 4 (unavailable), never a crash. Everything a factory derives goes
into `cfgs` before construction, so it is in `config.json` and the configuration hash:
`cfgs.pilot_cfgs` (the launcher: `run_id`, `onset_step`, `study`, `plugin`, `group`, `treatment`,
`controller_variant`, `extra_checkpoint_steps`, `plasticity`, `dependencies`; the dependencies'
`run_dir` and `omnisafe_dir` are excluded from the hash), `cfgs.onset_cfgs` (Role 2, Study A),
`cfgs.continuation_cfgs` (Role 2, battery continuations) and `cfgs.studyb_cfgs` (Role 5: the kind
and arm, its levels or bins, the budget's divisor, placement and random stream, the J_C rule and
window, and for few-shot the parent, budget and horizons). The Study A factories refuse a registered
spec whose epoch length is not 20,000 steps (tests and smoke runs use `SMOKE-` copies). Plug-ins
read their dependencies only through `pilot.dependencies.from_cfgs`.

| Function or file | Owner | Returns / writes |
|---|---|---|
| Contract 1: `envs.evaluation.evaluate_run(omnisafe_dir, spec) -> dict`; for plug-in `study_b`, `studyb.evaluation.evaluate_run` (`contracts.evaluator_target`) | Role 2; Role 5 | `final_cost`, `final_return` (the measurement set at the final checkpoint, Q-final-cost-set), `selection` {step: [cost, return]} for exactly the last 10 checkpoints, `episodes` = 100, `selection_seeds` (100 distinct seeds, the same for every run). Extras: `final_selection_cost`, `final_step`, `final_seed_set` ("measurement"), `measurement_seeds`, `episode_costs` and `episode_returns` (`{"final": [...], "selection": {step: [...]}}`), `short_episodes`, `harness`; Study B adds `studyb_budgets` (the 100 per-episode budgets). Deterministic mean action, a seeded reset per episode, frozen normaliser. |
| `evaluation.json` (launcher) | Role 1 | The contract-1 result plus `eval_commit_hash` (the code that evaluated) and the evaluation stage's smoke markers. The ledger writer writes its per-episode arrays as the `evaluation` supplement record before the row. |
| Contract 2: `plasticity.csv` in the OmniSafe run dir | Role 3 | 13 columns: `step, dormant, rank, norm` (the actor; read by the ledger writer), `norm_reward_critic, norm_cost_critic, dormant_trainable, rank_trainable, dormant_reward_critic, rank_reward_critic, dormant_cost_critic, rank_cost_critic, norm_all`. One row per checkpoint saved through `logger.torch_save` (step 0 at installation, the onset and onset + 200,000 steps included), never a step twice; checked by `contracts.validate_plasticity`. `plasticity_meta.json` beside it records the batch file and SHA-256, τ, δ, the input rule and any error (a row that cannot be computed is NaN). Required for Study A training runs (`plasticity_required`). |
| Contract 3: `progress.csv` names; checkpoints | Roles 2, 5 | Every multiplier under a name starting `Metrics/LagrangeMultiplier`; Study B's per level as `Metrics/LagrangeMultiplier/level_{b:g}` (e.g. `…/level_10`, `…/level_17.5`), with no plain `Metrics/LagrangeMultiplier`; the Continuous arm names each 5-unit bin by its lower edge (`level_10` for [10, 15)), and its per-level check is skipped because its `training_levels` is None; every loss under `Loss/`. All are checked for non-finite values; each level's final value goes to `per_level_multipliers`. Diagnostics stay outside those prefixes: `Onset/Active`, `Onset/EffectiveBudget`, `Onset/MultiplierBeforeUpdate`, `Onset/MultiplierProposed` (Study A; NaN before onset) and `Metrics/LevelJc/`, `LevelBudget/`, `LevelEpisodes/`, `LevelWindow/`, `LevelBatchEpCost/level_…` (Study B). Study B checkpoints add `level_lagranges`, `level_lambda_optimizers` and `level_window`. Extra checkpoints (`contracts.extra_checkpoint_steps`): the onset, onset + 200,000 steps (Study A, box H3 (c)), the few-shot horizons (500,000 is off the grid) and, for a total off the grid, the end-relative window. |
| Contract 4: `analysis.matching.select_checkpoint(checkpoints) -> int` | Role 4 | Rule 1 on the ledger's checkpoint records (selection set only; ties to the later checkpoint, distances to 25 compared after rounding to 1e-4); the pilot code recomputes rule 1 and refuses a different answer. |
| Contract 5: `envs.evaluation.evaluate_battery(omnisafe_dir, spec, step, conditions) -> dict` | Role 2 | `measurement` (C_ID), `measurement_return`, one cost and `<condition>_return` per condition (`hazard`, `dynamics`; `conditions=[]` measures C_ID only), `episodes` = 100, `measurement_seeds`, `seeds` {set: [...]}, the per-episode arrays, `conditions` {name: definition} (what was relocated or scaled, and the seeds), `short_episodes`, `harness`. Gated inside by Q-hazard and Q-dynamics. |
| Contract 6: `envs.evaluation.evaluate_checkpoint(omnisafe_dir, step, episodes, *, spec=None) -> float` | Role 2 | The mean cost of one checkpoint over the first `episodes` selection seeds, for `scripts/determinism_check.py --registered-form` (a budget-conditioned checkpoint carries the arm's training budgets in turn; gated by Q-studyb-eval). |
| `envs.evaluation.evaluate_continuation(omnisafe_dir, spec) -> dict` | Role 2 | A battery continuation's final checkpoint on the measurement seeds, on `spec['task']` (`enrich continuations`). |
| `studyb.evaluation.evaluate_training_budgets(omnisafe_dir, spec) -> dict` | Role 5 | `mean_cost` {10: c, 20: c, 40: c}, `satisfaction` {…}, `episodes` = 100, `seeds` (the 100 measurement seeds, `envs.evaluation.measurement_seeds()`) (G4; `g4-measure`). |
| `studyb.evaluation.evaluate_zero_shot`, `evaluate_fewshot` | Role 5 | Zero-shot at the unseen budgets and the reference budgets (training levels, or 10 and 40 for Continuous); few-shot at each horizon of a continuation under its budget. Few-shot keys are `pilot.contracts.fewshot_key(budget, horizon)` = `"5.0_200000"`. All gated by Q-studyb-eval. |
| `metrics.interventions` | Role 3 | `intervention_seed`, `intervention_generator`, `partial_reset`, `plasticity_injection`, `InjectedHead` (submodules `trunk`, `frozen`, `new`, `new_frozen`), `rebuild_injected`, `build_actor`, `intervention_summary`. `envs.onset` decides when (once, at the start of the onset epoch's rollout, after the onset checkpoint and its plasticity row); `metrics.interventions` decides what. An injected actor is recognised by `'mean.trunk.0.weight'` in its state, never by the spec. |
| `intervention.json` in the OmniSafe run dir | Roles 3 and 2 | Written once by the Study A plug-in for reset and injection runs: `intervention`, `owner`, `registered`, `proposal_key`, `depth`, `generator_seed`, `weight_initialization_mode`, parameter and tensor counts before and after (all and trainable), `trainable` (names), `output_max_abs_change` (on the fixed batch), and from the plug-in `run_id`, `run_seed`, `onset_step`, `onset_epoch`, `applied`, `applied_by`, `output_check_batch` {task, states, source, normalised, normaliser_count}. |
| `continuation.json` in the OmniSafe run dir | Roles 1, 2, 5 | Written by every continuation from `pilot.dependencies.restore_learner`: `parent_run_id`, `parent_step`, `parent_checkpoint` (the ledger's spelling, so it is portable) and its `parent_checkpoint_sha256`, `parent_commit_hash`, `parent_config_hash`, `injected_parent`, `observation_mapping`, `restored`, `fresh`, `not_restored`, `actor_schedule`, `parameters`, `rule`. `enrich continuations` and `enrich fewshot` check that the parent checkpoint is the one the row matched (or the final one) and refuse a continuation with smoke markers or without a full commit hash. |
| `pilot.dependencies` | Role 1 | `resolve(spec, run_dir)` (before the claim), `to_config`, `from_cfgs(cfgs)`, `Dependency.checkpoint(step)`, `Dependency.multiplier_at(step)` (the value logged in progress.csv), `load_full_state(path)` (`weights_only=True`), `restore_learner(algo, state, *, spec, obs_map=None)`. A step is converted to an epoch with the dependency's own `steps_per_epoch`. |
| Supplement records (`results/supplement_schema.py`; written by `pilot/supplement.py`) | Role 4 (models); Role 1 (writer) | `<ledger dir>/supplement/<kind>/<run_id>[.<part>].json`, so `results/supplement/` and `results/pilot/supplement/`. Written once, validated, never overwritten (a byte-identical rewrite is accepted); no timestamps; `code_commit` records the producing code. Kinds: `evaluation` (contract 1), `measurement` (matched checkpoint, C_ID and its return), `battery` (part = condition), `final_battery` (part = measurement, hazard or dynamics; rule 7 (b)), `sensitivity_battery` (part = condition; rule 7 (a)), `continuation` (part = finetune or transfer), `zero_shot`, `fewshot` (part = `b{budget:g}`), `training` (recovery, controller quantities, plasticity rows at onset and at the check, intervention summary). Read by `analysis.data`. The schema is frozen at version 1, its SHA-256 in `results/supplement_schema.sha256` (Q-ledger-v2). |
| `pilot.go_decision` → `analysis.pilot_check` | Roles 1, 4 | The analysis's pilot mode reads the go report's `Condition.values`: `reference_cost_mean`, `late_cost_mean`, `reference_sd`, `match_difference`, `arms_matched`, `reference_sd_ok`, `delta_mean`, `U80`, `U80_pooled` (descriptive only, Q-g3-pairing). Keep these names. The analysis's code hash (`analysis.records.code_hash`, recorded at the go decision) covers `analysis/*.py` and the in-repository modules it imports (Q-exploratory-labels). |
| Determinism report (`ledger/determinism/`) | Role 1 | Records the plug-in, the arm and every change from it, `run_commits`, `runs_worktree_dirty`, `commit_at_end`, `same_commit`; `passed` needs both runs trained on the report's commit and HEAD unchanged at the end. The scheduler accepts only `run_commits == [commit_hash, commit_hash]`. |

`omnisafe.Agent` rejects unknown algorithm names, so `pilot/launch.py` builds the config itself and
calls the plug-in.

---

## 9. Pre-registration questions and their answers

Every key below is answered in `docs/DECISIONS.md` (2026-10-02), with the full reasoning in
`docs/DECISIONS_EVIDENCE.md`; the answers become amendments when the group ratifies them and enters
them in Table 9.1 (tasks 10 and 11). Keys marked \* are exactly the 54 keys of `configs/registered.py`
`PENDING` (`tests/test_registered.py` checks this section against it), and all 54 are in
`ANSWERED_QUESTIONS`. Unstarred keys are not in `PENDING`: they are documented in the code that
implements their answers and recorded in Table 9.1 only. "Before" means the latest safe moment for
the answer. In that column, **pilot** in bold marks a key that a planned pilot run or a pilot result
waited on; plain pilot means it had to be answered before the pilot but blocked none of them (at most
a replacement run). Answering all of them before any data keeps every result confirmatory.

The starred keys were of three kinds, and all three are now released in code:
- **run gates**: the runs that depend on an answer carry the key and waited for it;
- **result gates**: the function producing a result that depends on an answer refused while the key
  was open (Q-hazard, Q-dynamics, Q-studyb-eval, Q-tie-break, Q-matched-cost-set, Q-arm-complete,
  Q-controller-quantities, Q-adapt-censoring);
- **report keys**: they gate nothing; while one was open, the go report or the analysis showed every
  verdict that depends on it as UNDECIDED or PROVISIONAL.

The go-report keys (Q-g1-level, Q-g3-pairing, Q-g4-level, Q-matched-cost-set, Q-final-cost-set),
Q-g2-run-equivalents and Q-interrupted-run gate no run. With their answers the go report reads each
condition by its answer and reports the other reading beside it; G3 with fewer than three seed
pairs is decided by the seed-order pairing of Q-g3-pairing, not by the group. The interruption
policy of Q-interrupted-run is `restart` on a registered data root (`schedule set-policy restart`
for a root whose policy was fixed before the key was answered).

**Already raised by First Tasks (Table 9):**
- which evaluation set gives the matched cost (Q-matched-cost-set\*). **Answered:** the measurement
  set, for matching, the reference SD and G1; the selection-set cost is reported beside it
  (`docs/DECISIONS.md`).
- the checkpoint tie-break (Q-tie-break\*). **Answered:** the later checkpoint, distances compared
  after rounding to 1e-4 (`docs/DECISIONS.md`).
- where per-checkpoint selection costs are stored (the schema nests them, which contradicts the
  proposed second table) (Q-checkpoint-table). **Answered:** the nested list of schema v1, recorded
  in Table 9.1 as a divergence of storage (`docs/DECISIONS.md`).
- constrained-steps rounding (Q-rounding\*). **Answered:** the onset to the nearest whole epoch
  (1,120,000 and 3,340,000 steps), then exactly 10,000,000 constrained steps (`docs/DECISIONS.md`).
- whether the cost critic trains before onset (Q-cost-critic\*). **Answered:** yes: both critics
  train from the first epoch, and only the surrogate and the multiplier change at onset
  (`docs/DECISIONS.md`).

**New:**

| Key | Where | Question and answer | Before |
|---|---|---|---|
| Q-hazard\* | Table 2.2; Part 5.2; G3 | Training already samples new layouts every episode from the same generator, so "re-sampled with the task's own layout generator" is not a shift, and 20 layouts × 5 deterministic episodes can repeat episodes. **Answered:** Hazards re-sampled by the task's generator inside the square \|x\|, \|y\| ≤ 0.75 (centres within 0.57), 20 layout seeds that pin the hazards × 5 episode seeds that redraw the other objects; the registered-wording form enters no result (`docs/DECISIONS.md`). | **pilot** |
| Q-budget-normalisation\* | Table 2.5 | Where the budget/100 input enters relative to OmniSafe's observation normaliser, which would erase it in the Single arms. **Answered:** Appended after the normaliser and the whole wrapper chain; the actor, both critics and the buffer take one more input (`docs/DECISIONS.md`). | **pilot** |
| Q-search-before-pilot\* | Part 7.2; App. C | Whether "before Study B's first run" includes the pilot's two Study B runs. **Answered:** Yes: the record is complete and committed before either starts, and a "partial" outcome also needs its narrowing amendment committed; Study A's pilot and the determinism checks do not wait (`docs/DECISIONS.md`). | **pilot** |
| Q-lr-decay | Table 3.1; H1/H3 | Actor LR decays to 0 over each run; late arms have (1−N)× the LR when constrained. **Answered:** OmniSafe's decay is kept in every arm and named in the paper; the one exception is the data control's extension (`docs/DECISIONS.md`). | pilot |
| Q-data-control-lr\* | Table 2.4 | OmniSafe decays the actor LR to 0 over the run's own epochs, so 'no change to the network or optimiser' fails for the T + N·T data control either way. **Answered:** Its first T steps follow the untreated arm's schedule bit for bit; the N·T further steps restart at (1 − N) × 3e-4 and decay linearly to 0 (`docs/DECISIONS.md`). | main sweep |
| Q-multiplier-adam | eq. (4) | λ is updated by Adam (lr 0.035) on −λ(J_C − d), clamped at 0, not by the plain step of eq. (4). **Answered:** Adam, as the registered configuration names it; the amendment states its step size and momentum beside eq. (4) (`docs/DECISIONS.md`). | pilot |
| Q-jc-window\* | eq. (4); Part 3.4 | J_C is OmniSafe's 50-episode moving mean, not the mean of the current epoch. **Answered:** J_C stays OmniSafe's window, unmodified; the per-epoch batch mean (`Metrics/BatchEpCost`) is logged beside it for the ledger and recovery time, and the registered gloss is corrected (`docs/DECISIONS.md`). | **pilot** |
| Q-first-checkpoint | Table 3.1 | Which checkpoint is "the first checkpoint" of the determinism check. **Answered:** The first scheduled one, at 200,000 steps, not the untrained `epoch-0.pt` (`docs/DECISIONS.md`). | pilot |
| Q-pilot-unconstrained-seed\* | Part 3.6 | The seed of the unconstrained pilot run. **Answered:** Seed 0, in the re-pilot too (`docs/DECISIONS.md`). | **pilot** |
| Q-interrupted-run\* | Part 5.6 | Whether a run stopped by the machine counts as a crash. **Answered:** No: it is restarted from scratch with the same seed, without limit; a run that ends without a result three times while the scheduler watched it (SIGKILL, a non-zero exit) is excluded as incomplete and replaced; registered data roots run under `schedule set-policy restart` (`docs/DECISIONS.md`). | pilot |
| Q-seed-collision\* | Parts 5.5/5.6 | Replacement and surplus seeds both start at 5, and pilot arms use seeds 0–2. **Answered:** Every arm takes its smallest unused seed ≥ 5, for either purpose (`docs/DECISIONS.md`). | pilot |
| Q-exclusion-breaker | Part 5.6 | What follows two exclusions of one arm with the same cause, when the scheduler stops replacing. **Answered:** An operational pause: the operator diagnoses from logs, never results, fixes a defect by a recorded erratum, then replaces; at the arm's seed target of same-cause exclusions, the arm is reported as not completing its seeds (`docs/DECISIONS.md`). | pilot |
| Q-g1-level\* | Part 6 G1 | Arm mean over seeds, or every seed? **Answered:** The arm mean over its three seeds; the per-seed result is reported beside it and decides nothing (`docs/DECISIONS.md`). | go |
| Q-g3-pairing\* | eq. (12) | Paired or pooled U80, and what if a replacement seed leaves fewer than three pairs? **Answered:** The paired form; leftover runs are paired in seed order and the rule decides G3, never the group; the pooled value is descriptive only (`docs/DECISIONS.md`). | go |
| Q-g4-level\* | Part 6 G4 vs 3.6 | "At most its budget plus 2.5" (Part 6) or "within 2.5 of its budget" (Part 3.6)? **Answered:** Part 6's one-sided clause decides; the two-sided reading is reported beside it (`docs/DECISIONS.md`). | go |
| Q-p-list | Part 3.6; Table 8.1 | Part 3.6 and Table 8.1 list different [P] quantities of the Moderate arm. **Answered:** Each list gains the missing item: the satisfaction rates in Table 8.1 (reported only), the mean costs in Part 3.6 (used by G4) (`docs/DECISIONS.md`). | go |
| Q-g2-run-equivalents\* | Part 6 G2 | 484 counts Study A's runs, not their steps (540 training, 627 with continuations). **Answered:** The corrected total (627.13 for the uncut design) decides G2 and prices the surplus seeds; 484 is reported beside it (`docs/DECISIONS.md`). | go |
| Q-surplus-arm-set\* | Part 5.5 | Which N = 0.50 arms get surplus seeds (the onset shape is not named)? **Answered:** Both shapes and both controls, with N = 0, on SafetyPointGoal1-v0, and the seven Study B arms, the same count for each (`docs/DECISIONS.md`). | go |
| Q-interval\* | Part 5.3 | Two sentences disagree on which interval "excludes zero". **Answered:** The primary (Welch) interval; the percentile interval is reported beside it, and a correlation uses its registered percentile interval (`docs/DECISIONS.md`). | go |
| Q-h1-shape\* | H1 | Which onset shape does Δ(N) use? **Answered:** Abrupt is primary; ramp is a secondary reading, Holm-corrected in its own families and never pooled (`docs/DECISIONS.md`). | go |
| Q-g-names | Table 9.1; Part 6 | Table 9.1 says "G1 to G4" for Study B (Part 1.4 has G1–G5), and Part 6 reuses G1–G4 for the go conditions. **Answered:** An erratum makes the first row read "G1 to G5"; the go conditions keep their names, qualified as "go condition G1 (Part 6)" (`docs/DECISIONS.md`). | go |
| Q-selection-window\* | Part 4.1 rule 1 | "Last ten checkpoints" ≠ "final 2,000,000 steps" when a total is off the 200,000 grid (11.12M, 13.34M, 12.5M). **Answered:** The ten checkpoints at total − k × 200,000, saved as a grid relative to the end where needed (`docs/DECISIONS.md`). | main sweep |
| Q-continuations\* | Tables 2.2, 2.5 | Parent checkpoint, learning rate, critics, optimiser states and normaliser of every continuation. **Answered:** As proposed (matched checkpoint for fine-tuning and transfer, final for few-shot; actor, critics and normaliser restored; fresh Adam states and a fresh decay over the continuation's registered length), except that under cut 3 a shortened few-shot continuation keeps the 1,000,000-step schedule and stops at 200,000 steps (`docs/DECISIONS.md`). | main sweep |
| Q-transfer-obs\* | Table 2.2 | Observation dims differ (60/76/72/88). **Answered:** CarGoal1's held-out task is CarButton1; first layers mapped by component name (shared copied, target-only at zero weight, source-only dropped), and the normaliser likewise (`docs/DECISIONS.md`). | main sweep |
| Q-warm-start\* | Table 2.4 | What is copied, from which run and from when. **Answered:** Only λ, with a fresh Adam state, as logged after the epoch before onset by the N = 0 run of the same task and seed (an auxiliary run for a replacement seed); an excluded source gives nothing, and its blocked dependent is superseded by the next unused seed (`docs/DECISIONS.md`). | main sweep |
| Q-rate-limit\* | Table 2.4 | How the clip combines with OmniSafe's Adam update. **Answered:** After each Adam step, clip to the previous value ± (0.05 × previous + 0.01), then at 0, Adam's moments untouched; the value before the clip is logged, and the amendment states when the clip binds (`docs/DECISIONS.md`). | main sweep |
| Q-pid-eq9\* | eq. (9) | OmniSafe's PID differs from eq. (9) (EMA P term, delayed derivative of cost, start at 0). **Answered:** OmniSafe's PIDLagrangian unmodified from onset with the pinned gains, its update skipped before onset, total-steps matched; eq. (9) is rewritten in OmniSafe's form (`docs/DECISIONS.md`). | main sweep |
| Q-reset-injection\* | Table 2.4 | Which layers, what happens to `log_std` and the optimiser state, and what trains after injection. **Answered:** The output layer and the hidden layer before it; `log_std` neither reset nor frozen; after injection the first hidden layer and `log_std` keep training (Nikishin et al., 2023, Section 3, as two search-engine extracts report it) (`docs/DECISIONS.md`). | main sweep |
| Q-ramp-step\* | eq. (5) | Eq. (5) is written for a step t, but the multiplier updates once per epoch. **Answered:** Epoch e's update uses t = e × 20,000; N·T is the run's onset step; W = 1,000,000 in every ramp arm (`docs/DECISIONS.md`). | main sweep |
| Q-plasticity-definitions\* | Table 2.3 | The input of the metrics and their details. **Answered:** The fixed batch through the run's own frozen normaliser; post-tanh activations; a pooled dormant fraction; uncentred SVD of the last hidden layer; `log_std` in the norm; the same metrics for each critic; the H3 (c) trainable layer as proposed (`docs/DECISIONS.md`). | main sweep |
| Q-dynamics\* | Table 2.2 | MuJoCo combines friction by the maximum of the two geoms, so scaling the floor alone changes nothing, and 'body mass' names no bodies. **Answered:** The robot bodies' mass and inertia × 1.3; the floor's and the robot geoms' friction × 0.7; reapplied after every reset; the measurement seeds (`docs/DECISIONS.md`). | **pilot** |
| Q-level-jc\* | Table 2.5 | Which episodes form a level's J_C. **Answered:** The level's episodes among OmniSafe's last 50; a level with none takes no step (`docs/DECISIONS.md`). | **pilot** |
| Q-continuous-bins\* | Table 2.5 | Bin edges, the bin of 40, and a bin's d. **Answered:** [10,15) ... [35,40], read from the float32 budget feature; d = the mean budget of the bin's window episodes; one optimiser step per update (`docs/DECISIONS.md`). | Study B |
| Q-studyb-order\* | Part 3.3 | What "scheduled after Study A's main sweep" means. **Answered:** A non-pilot Study B run starts when every run of the main sweep under the cuts in force has finished training (an excluded run counts as finished) (`docs/DECISIONS.md`). | Study B |
| Q-studyb-eval\* | Parts 3.4, 3.6; App. B | Study B's checkpoint, evaluation seeds and contract-1 budgets. **Answered:** The final checkpoint; the measurement set at every budget and horizon; the training budgets in turn for contract 1 (Continuous: the midpoints of 100 equal sub-intervals of [10, 40]) (`docs/DECISIONS.md`). | **pilot** |
| Q-determinism-late-onset\* | Table 3.1 | The registered onset of the PID arm (5,000,000 steps) lies beyond the check's first checkpoint. **Answered:** The check moves the onset to N × its own length, in whole epochs (100,000 steps of the 200,000-step registered form) (`docs/DECISIONS.md`). | main sweep |
| Q-arm-complete\* | Part 4.1 | When is an arm complete, given surplus and replacement seeds? **Answered:** When its completed runs reach its seed target (5, plus the surplus count for primary arms); its flags wait for the reference arm and the arm itself, and every completed seed counts (`docs/DECISIONS.md`). | go |
| Q-controller-quantities\* | Tables 2.1, 2.4 | Overshoot, settling time and recovery time at epoch resolution. **Answered:** As proposed, with the settling time defined also when the final multiplier is 0 (the epoch from which it stays 0) (`docs/DECISIONS.md`). | main sweep |
| Q-adapt-censoring\* | Table 2.5 | How "recorded as above the largest horizon" is encoded. **Answered:** The largest horizon + 1; 'reaches' means >= 0.80 (`docs/DECISIONS.md`). | Study B |
| Q-final-cost-set\* | App. B | The evaluation set of `final_cost`. **Answered:** The measurement set at the final checkpoint; the selection-set value stays in the checkpoint record and decides nothing (`docs/DECISIONS.md`). | go |
| Q-controls-reading\* | H1, H2 | How the two step-matching controls combine. **Answered:** Supported only under both, falsified only under both, otherwise inconclusive (`docs/DECISIONS.md`). | go |
| Q-support-falsify-overlap\* | H1, G1 | What if support and falsification both hold? **Answered:** Support needs the trend and ordered point estimates; on overlap the box is FALSIFIED with a note, then calibrated (`docs/DECISIONS.md`). | go |
| Q-falsification-calibration\* | Part 1.2; 5.7 | Do the boxes or Part 1.2's calibration decide "falsified"? **Answered:** A box with a minimum effect is FALSIFIED only when its Part 5.7 bound lies below that minimum, otherwise INCONCLUSIVE (`docs/DECISIONS.md`). | go |
| Q-holm-families\* | Part 5.2 | Which of its two overlapping families must a claim survive? **Answered:** Every secondary claim must survive Holm in both its families (`docs/DECISIONS.md`). | go |
| Q-threshold-arithmetic\* | Part 4.1 | Sample or population SD, strict or inclusive comparisons, rounding. **Answered:** Sample SD; infeasible if SD > 10; rules 3 and 5 inclusive; means compared after rounding to 1e-4, the SD to 1e-9; G1 and G3 likewise (U80 to 1e-9; U80_pooled is now descriptive, Q-g3-pairing) (`docs/DECISIONS.md`). | go |
| Q-iqm\* | Parts 5.1, 5.3 | How the IQM's bootstrap over episodes squares with the seed as the only replicate. **Answered:** The IQM is descriptive, with a two-level (seed, then episode) bootstrap (`docs/DECISIONS.md`). | go |
| Q-bootstrap-details\* | Part 5.3 | Random stream, quantile rule, shared seeds, undefined resamples, and the Holm p of a Spearman claim. **Answered:** As proposed: a generator per analysis, numpy's linear rule, one draw for identical seed lists, undefined resamples dropped and counted, a bootstrap p for a Spearman claim (`docs/DECISIONS.md`). | go |
| Q-surplus-in-analysis\* | Parts 4.1, 5.4 | How surplus seeds enter the analysis. **Answered:** Every completed seed counts, except in the H1 trend and H3 (a), which keep each arm's registered seeds and their replacements (`docs/DECISIONS.md`). | go |
| Q-h3-scope\* | H3 | Which arms form H3 (a); at which N (b) and (c) are read; what 'does not reduce' means. **Answered:** (a) the matched late arms of the main sweep; (b) and (c) at N = 0.50; 'does not reduce' = no reduction with an uncorrected interval excluding zero (`docs/DECISIONS.md`). | go |
| Q-h4-reading\* | H4 | Is 'the gap does not fall' read on the point estimate or the interval? **Answered:** On the point estimate (`docs/DECISIONS.md`). | go |
| Q-h0-scope\* | H0 | On which outcome must H1 and H2 both be falsified? **Answered:** The primary outcome, read once; H3 and H4 are then not tested anywhere (`docs/DECISIONS.md`). | go |
| Q-g1-criteria\* | G1 | A point condition or a bound, and which intervals overlap? **Answered:** Point estimate > 5 pp and the 97.5% Welch interval excluding zero; overlap of the per-arm 97.5% t-intervals (`docs/DECISIONS.md`). | go |
| Q-g2-outcome\* | G2; Part 5.2 | The G2 box counts budgets while Part 5.2 tests the mean. **Answered:** The per-budget count decides, confirmatory at 97.5% per budget; the mean contrast is descriptive; on a floored budget the comparator is chosen by violation magnitude (`docs/DECISIONS.md`). | go |
| Q-g3-arms\* | G3 | 'For every arm', although only some arms are at equal distances. **Answered:** G3, a secondary box, is decided on the four arms spanning [10, 40] (nearest levels 10 and 40); the all-seven reading is reported beside it (`docs/DECISIONS.md`). | go |
| Q-g4-reading\* | G4 (hypothesis) | The median over what, and what 'within one horizon' means. **Answered:** An ordering of arm-level medians; 'one horizon below' and 'within one horizon' (a spread below 1) per budget, on a majority of budgets; a censored run takes the index after the largest horizon, so G4 stays testable after cut 3 (`docs/DECISIONS.md`). | go |
| Q-floor-rule\* | Part 4.2 | Which arms count and which rate. **Answered:** All seven arms, each by its mean rate; a floored budget is compared on violation magnitude, by direction and interval (`docs/DECISIONS.md`). | go |
| Q-ledger-v2 | App. B | Schema v2 or the supplement records (§8). **Answered:** Schema v1 stays frozen; the supplement records are kept, their schema frozen at version 1 with `results/supplement_schema.sha256` (`docs/DECISIONS.md`). | pilot / Study B |
| Q-appendix-a | App. A | Configs, environment file, workstation spec and fixed batches were "Copied verbatim from the repository at the registration commit", but that commit holds only the prereg. **Answered:** They are recorded as added later, each with path, SHA-256 and source; the committed fixed batches (SHA-256 `5340625c…`, `abc01be6…`, `84ae7249…`) stand, and a mismatch of `--check` on the workstation is reported in Table 9.1, never collected again (`docs/DECISIONS.md`). | pilot |
| Q-manipulation-point | H3 (c) | "The first logging point after onset (200,000 steps)" is off the grid at N = 0.25 (onset 2,500,000). **Answered:** Onset + 200,000 steps (`docs/DECISIONS.md`). | main sweep |
| Q-onset-epoch | Table 2.1; Table 2.4 | Which update is the first constrained one. **Answered:** Epoch e is constrained iff e ≥ onset/E, the multiplier updated before the actor (`docs/DECISIONS.md`). | main sweep |
| Q-eval-seeds | Table 2.1; Table 2.2 | Seed bases of the evaluation sets. **Answered:** Selection 1,000,000 + i, measurement 2,000,000 + i, hazard layouts 3,000,000 + j, hazard episodes 3,100,000 + 5j + k, shared by every run (`docs/DECISIONS.md`). | pilot |
| Q-mujoco-exception | Part 5.6 | In the pinned stack an unstable simulation raises nothing (MuJoCo 2.3.0 warns and resets). **Answered:** Such an episode is never scored and is replaced from a reserve seed sequence; more than 5 in one evaluation call fail the evaluation, held for the group; never an exclusion (`docs/DECISIONS.md`). | pilot |
| Q-search-depth | App. C | How deep each search is screened. **Answered:** The first 50 results in each source's relevance order ("Relevance" in arXiv), fixed before the first query; citation lists in full (`docs/DECISIONS.md`). | pilot |
| Q-search-queries | App. C, Table C.1 | What "each with and without 'number of thresholds' and 'spacing'" means. **Answered:** All four variants of each query: 80 searches (`docs/DECISIONS.md`). | pilot |
| Q-search-citations | App. C | Which citation lists. **Answered:** Yao et al.'s references and cited-by lists, both lists of every work citing it, and the references of every work read in full (`docs/DECISIONS.md`). | pilot |
| Q-search-record-scope | App. C | What "for every hit" requires. **Answered:** Every screened result listed with citation, URL or DOI, decision and reason; every work read in full and the three known works get every Table C.1 field (`docs/DECISIONS.md`). | pilot |
| Q-return-outcome | Part 5.2; App. B | The matched checkpoint's return has no ledger field. **Answered:** The measurement-set return of the matched checkpoint, kept in the `measurement` supplement record (`docs/DECISIONS.md`). | pilot |
| Q-detectable-size | Part 5.5 | Is "not claimed as findings" below the detectable size a decision criterion? **Answered:** An annotation only; verdicts are unchanged, and the note on a SUPPORTED verdict says its power was under 80 percent (medium confidence: confirm at ratification) (`docs/DECISIONS.md`). | go |
| Q-exploratory-labels | Part 8.2; Part 5.8 | How the analysis learns which results an amendment made after data were seen affects, and how edits after go are tracked. **Answered:** `analysis/AMENDMENTS.json` and `analysis/ERRATA.json`, with a code hash over the analysis and every in-repository module it imports (`docs/DECISIONS.md`). | go |
| Q-unmatched-reporting | Part 4.1 rule 6 | What "a late arm that never reaches the budget" is. **Answered:** An unmatched arm whose matched cost exceeds d + tolerance (rule 3's test) (`docs/DECISIONS.md`). | go |
| Q-training-age-systematic | Part 4.1.1 | When selected training ages "differ systematically". **Answered:** When the Welch 95% interval of the difference excludes 0, for every pair of arms a verdict compares (`docs/DECISIONS.md`). | go |
| Q-rule3-tolerance-sensitivity | Part 4.1 rules 3, 7 | Whether rule 7 (a)'s tolerance 5.0 also changes rule 3. **Answered:** Yes: rule 3 becomes d + 5 too (`docs/DECISIONS.md`). | go |
| Q-treatment-other-N | Part 5.1; Table 3.3 | Treatments and controller variants also run at N = 0.10 and 0.25, where Part 5.1 defines no estimand. **Answered:** Reported as descriptive estimates, with no verdict (`docs/DECISIONS.md`). | go |

---

## 10. Working rules and mistakes to avoid

- **Numbers.** Every registered number lives in `configs/registered.py` with its location, and
  tests assert it against the registered `.docx`. Never type one elsewhere, and never choose a
  value by trying several (First Tasks habits 1, 2 and 5). A value an open question decides lives in
  the module that uses it, as a named constant commented `# answered in Table 9.1 (Q-key)` (while a
  key is open again after the group changed it: `# PROPOSAL (Q-key)`), never in `configs/registered.py`.
- **Open questions.** Code never decides an open question: a run gate goes in
  `manifest.run_gate_keys` or in the spec's `pending` (both read by `manifest.open_run_gates`), a
  result gate is `pilot.errors.require_answered(...)` inside the producing function, and every key
  passed to a gate (`require_answered`, `open_keys`, `is_open`) must be a `PENDING` key
  (a test checks it).
- **Launching.** Launch registered runs only from a clean commit; the launcher and scheduler
  refuse otherwise. `--allow-dirty` and `--allow-pending` are for smoke tests, whose ledgers never
  reach the repository.
- **Where runs go.** Never run on GPU. Never commit checkpoints or run folders. The pilot's runs
  never go to the main ledger.
- **Exclusions.** A run is excluded only by Part 5.6. The scheduler never looks at costs or
  returns, and neither should you when resolving a held run.
- **Allocation.** Commit `pilot/allocation.json` before any pilot run.
- **Instrumentation never changes training.** Hooks, extra checkpoints and metrics draw from no
  global random stream; new random draws (budget sampling, intervention weights) use generators of
  their own, seeded from the run seed. `metrics.recompute` sets one torch thread itself, as in
  training, to reproduce the live rows bit for bit; keep it so. The metrics run on `actor.mean`,
  never `actor(x)` or `actor.predict`, which set OmniSafe's cached `_current_dist`; do not
  `copy.deepcopy` a `GaussianLearningActor` after a forward pass with gradients.
- **Coupled code.** `envs.onset.PARAM_KEYS` and its re-derivation of arm names follow
  `pilot/manifest.py`: change them together.
- **Backups.** Back up `/data/checkpoints` and `/data/scheduler/state.sqlite` regularly, and commit
  the ledgers and supplements after each batch of runs.

---

## 11. What was not possible from here, and what I need from you

- **Registration acts.** The tag push, the Zenodo/OSF deposit, the ruleset and Table 9.0 in Word
  are acts on the public registration record, so they are yours (tasks 1, 2, 3 and 6).
- **Adoption.** The code of every role was written by the pilot owner's build; apart from the
  2026-09-28 version of your part on `main` (PR #2), nothing is committed. Each owner reviews and
  adopts their part by a pull request (task 4a). Until then it is nobody's reviewed work.
- **The ratification.** The 54 `PENDING` keys, the 24 notes of §9 and the ledger-schema amendment
  are answered in `docs/DECISIONS.md`, prepared by AI agents at the pilot owner's request; only the
  group can ratify them (tasks 10 and 11) and give the machine-hour allocation (task 12).
- **The targeted search** (Appendix C) has not been run and could not be: it is Hamza Nisar's act,
  needs normal internet access, and gates every Study B run, the pilot's two included (task 16).
- **The workstation.** Its OS, core count, RAM and disk decide K and so G2. The environment lock,
  the workstation record, the fixed-batch check (`--check`, task 8a) and the determinism reports per
  plug-in (tasks 9 and 12a) can only be made there.
- **People.** The GitHub usernames of three members are needed for CODEOWNERS (Muhammad Talha
  Jamil, Abdullah and Hamza Nisar; `@Umair-Waseem` and `@Abdullah9712` are known), and a second
  reviewer for your own paths. The mapping of roles to people is in §1.
- **Literature.** Papers could not be read in this sandbox: arxiv.org, openreview.net,
  proceedings.mlr.press and readthedocs are blocked by its network policy. Everything above rests
  on the pinned source code, which is the stronger evidence for these questions. The literature
  claims (Sokar τ = 0.025, Kumar δ = 0.01, the benchmark table values, Yao/Sootla details, and
  Nikishin et al.'s injection method behind the Q-reset-injection answer, known only from two
  search-engine extracts) were not verified here.

---

## 12. Verification record

- **Environment.** Everything was run on the pinned stack: Python 3.10.21, omnisafe 0.5.0,
  safety-gymnasium 0.4.1, mujoco 2.3.0, torch 2.14.0 (the PyPI wheel, which bundles CUDA
  libraries; runs use the CPU only), numpy 1.26.4, pandas 2.0.3, pyarrow 25.0.1, pydantic 2.13.5,
  scipy 1.15.3.
- **Fast tests (2026-10-01).** `python -m pytest -q` in the pinned environment, on the working tree
  as it then was (1,756 tests): `1705 passed, 51 deselected, 28 warnings in 527.92s (0:08:47)` (the
  counts were the same in every run, but the duration varies: an earlier run took 0:12:31; the 51
  deselected are the slow tests; the 28 warnings are deprecation warnings from the pinned
  libraries). Later reviews and the decisions of 2026-10-02 added tests (over 2,400 on 2026-10-02),
  and no complete run of the final tree is recorded here: a fast run during the 2026-10-02 work
  stopped at its first failure (`1 failed, 2107 passed, 63 deselected`), a test the decisions had
  made stale, since corrected. Re-run both suites before relying on this record. The tests cover,
  besides the earlier coverage below, every role's code: the onset plug-ins, controllers and
  treatments, the evaluation harness and its seed sets, the continuations and their restore rule,
  the plasticity metrics, hook and recomputation, the interventions, the fixed batches, Study B's
  conditioning, evaluations and search record check, the analysis (matching, statistics, every box,
  the command line, synthetic ledgers), the supplement schema, the enrichment chain end to end with
  `tests/fake_launcher.py`, the scheduler's launch gates, and that every key passed to a gate
  (`require_answered`, `open_keys`, `is_open`) is a `PENDING` key.
- **Slow tests.** 51 tests were marked `slow` in that tree (more now): short training through the
  real launcher and plug-ins; the launcher test compares one registered epoch (20,000 steps) with
  `omnisafe.Agent` bit for bit (every OmniSafe log column and the actor).
  `python -m pytest -q -m slow` on 2026-10-01: `51 passed, 1705 deselected in 2387.22s (0:39:47)`.
- **Earlier record (first handover).** 338 fast tests covered every registered value against its
  phrase in the registered `.docx`, the design counts, the scheduler end to end with a fake launcher
  (crash, NaN, segfault, interruption, adoption, orphan recovery, circuit breaker, surplus seeds,
  continuations and their parent's replacement, never-started and refused launches, restart limit
  and restart commit, registered-mode gates, smoke mode and ledger location per data root, live runs
  under another spelling of the data root), the launcher's refusals, the ledger writer, enrichment,
  and G1–G4 with both pilot readings of Part 6. The slow test of that earlier handover showed that
  two epochs through the launcher equal two epochs through `omnisafe.Agent` on every OmniSafe log
  column and on the final actor, bit for bit, and that the batch metrics satisfy
  (batch₀ + batch₁)/2 = OmniSafe's window mean.
- **Determinism rehearsal** (sandbox, a rehearsal and not the registered check). Two runs matched
  on every log column and every checkpoint tensor, and after the fix on the config hash too.
- **End-to-end smoke (2026-10-01).** `python scripts/smoke_run.py --data-root <scratch> --design
  sample --max-concurrent 4` ran the real scheduler and launcher with the real plug-ins on a
  sandbox scratch data root: the N = 0 arm, the N = 0.50 arm with partial reset, the N = 0.50 PID
  arm and the Moderate arm each trained 2 epochs (about 0.14 h each, four at once); their evaluation
  was refused, as expected for runs without the last ten checkpoints; the script printed "smoke test
  passed" and exited 0. Its ledgers lay under the data root. (The first handover's 1-epoch smoke run
  of plug-in `ppolag` ended likewise, waiting for the then-missing harness.)
- **Independent review (first handover).** Five reviewers covered pre-registration conformance,
  the OmniSafe API, adversarial testing, bias and integrity, and engineering practice. They reported
  65 findings. Every confirmed defect was fixed and tested, and the rest are open questions listed
  in §9. The integrity fixes:
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
- **Line-by-line convergence review (first handover).** Rounds of independent reviewers read every
  file of the repository line by line, each on one area. Role 4's frozen files and the registered
  pre-registration were read too, but not edited: their defects are listed in §3.3 and §9. Each
  round's confirmed defects were fixed, with a regression test for every behavioural fix, before
  the next round started. First series: round 2, 84 findings; round 3, 18; round 4, 4; round 5,
  none. Second series, which also covered every Markdown document: round 6, about 100 findings;
  round 7, 36; round 8, 21; round 9, 9; round 10, 5; round 11, 2; rounds 12 to 15, 3, 2, 1 and 1;
  round 16, no defects.
- **Review of the 2026-10-01 code.** **3 final adversarial review passes, 21 rounds in all, handled
  112 findings**: each finding was first verified against the code, the pre-registration or the
  pinned sources, and only then fixed. **The last round still found 2 major defects** (both fixed),
  so the review had not run dry: **another review before the pilot is advised**, with an extra
  targeted review of `analysis/` (one of its earlier rounds still found majors).

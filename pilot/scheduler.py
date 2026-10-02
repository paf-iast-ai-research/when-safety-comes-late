"""Run scheduler: launch, resume, never duplicate (First Tasks, Role 1 "What comes next").

State lives in one SQLite file outside the repository (``<data_root>/scheduler/state.sqlite``);
run directories live under ``<data_root>/checkpoints/<run_id>`` so that the ledger's relative
checkpoint paths resolve against ``results.ledger_schema.CHECKPOINT_ROOT`` (``/data/checkpoints``
when ``data_root`` is ``/data``).

Life of a run:

  pending --train--> training --> trained --evaluate--> evaluating --> evaluated --ledger--> ledgered
     |                  |             |                      |                        |
     |                  |             |                      +--> eval_failed          +--> ledger_failed
     |                  |             |                      |    --reevaluate--> trained   --retry-ledger--> evaluated
     |                  |             |                      |                              --reevaluate--> trained
     |                  |             |                      |                                (in no ledger yet)
     |                  |             |                      +--> trained (evaluator unavailable, launcher refused,
     |                  |                                         or ended unobserved)
     |                  +--> continued (a continuation: Study B's few-shot continuations and the battery's
     |                  |    fine-tuning and transfer continuations, manifest.CONTINUATION_GROUPS; their
     |                  |    results are read into the parent's row by enrichment, never written as a row)
     |                  +--> excluded (crash, incomplete, non-finite loss or multiplier; Part 5.6), recorded
     |                  |    in the ledger, and replaced by the arm's next unused seed; 'incomplete' includes
     |                  |    the third end without a final result of process origin under 'restart'
     |                  +--> interrupted (the machine stopped it, or the launcher ended without a final
     |                  |    result; held under the policy 'hold', or when a restart fails)
     |                  |    --restart--> pending (same seed, from scratch; under the 'restart' policy
     |                  |                 directly from training)
     |                  |    --set-policy exclude, or a scheduler start under 'exclude'--> excluded
     |                  |                 (cause 'incomplete'; smoke data roots only)
     |                  +--> pending (plug-in unavailable, launcher refused, launcher failed before its
     |                       claim, or never started: nothing ran, so nothing is excluded)
     +--> ledgered or excluded (already in a ledger, e.g. the state was restored from a backup)
     +--> blocked (a dependency was excluded, blocked or superseded: a warm-started run is replaced
     |    by Table 9.1's rule, Q-warm-start: `schedule resolve RUN replace`; a blocked Study B
     |    continuation follows its parent: `schedule resolve PARENT replace`; a blocked battery
     |    continuation is left to the group's amendment, Table 9.1; `schedule resolve RUN unblock`
     |    returns a run whose dependencies are fine again)
     |    --unblock--> pending
     |    --replace--> superseded (the arm's next unused seed runs instead)
     +--> superseded (a continuation whose parent was excluded and replaced)

  An excluded run left without a replacement (by the circuit breaker, or deferred because the code
  is uncommitted or HEAD moved) waits for the operator's `schedule resolve RUN replace`, under the
  rule of the circuit breaker below. An excluded continuation (few-shot or battery) cannot take a new
  seed (it follows its parent's seed and checkpoint; ``RunSpec.with_seed`` refuses it), so the
  scheduler offers no action for it: the group decides by an amendment (Table 9.1). ledgered,
  continued and superseded are terminal.

Queueing: ``add`` (``schedule add``, with the cut order of Part 6.1 read from the committed
``pilot/cuts.json``, ``CUTS_PATH``), ``add_surplus`` (``schedule add-surplus``, Part 5.5) and
``add_continuations`` (``schedule add-continuations``: the battery's fine-tuning and transfer
continuations of every matched Study A row, Table 2.2; pilot/manifest.py). ``status_report`` and
``status_lines`` give what ``schedule status`` shows: the counts, the launch gates holding runs
with their reasons, each missing evaluation harness by its target, and the decisions left to the
group, each run judged as the data root's own scheduler judges it (its stored mode, and a smoke
root's flags of its last ``schedule run``), whatever the status command's flags.

Guarantees:
  * never duplicate: run_id is the primary key; a run already in a ledger is never launched; the
    state 'training' is committed before the process starts; the launcher claims the run directory
    atomically; one scheduler process per data root (exclusive lock), and operator commands
    serialise with it on SQLite write transactions; live runs are adopted on restart, identified by
    their run directory on their command line (compared as real paths, so another spelling of the
    same data root still finds them); a run whose process is still alive is never restarted.
  * no result-based decisions: the scheduler reads only completion status, step counts and
    finiteness, never costs or returns. The interruption policy is stored once per data root and
    applies to every run alike; held runs may be restarted, which reproduces them bit for bit, but
    never excluded one by one. Table 9.1 (Q-interrupted-run) answers it with 'restart', which is
    refused on a registered data root while the key is open ('hold' until then) and is then the
    default of a registered data root's first run; 'hold' stays an operational pause, and 'exclude'
    is refused on a registered data root (smoke data roots may use any policy). Under 'restart' an
    interruption of machine origin (``_finish_training``: the launcher recorded SIGTERM, SIGINT or
    SIGHUP, the child died of one of those signals, or it ended while no scheduler was its parent) is
    restarted without limit; one of process origin (SIGKILL, e.g. the OOM killer, another non-crash
    signal, or a non-zero exit without a final result) is restarted MAX_RESTARTS times, and a third
    such end is Part 5.6's "fails to complete its steps": excluded with cause 'incomplete' and
    replaced, counting towards the circuit breaker. The cap binds the operator's restarts of a run
    held under 'hold' as well: `schedule resolve RUN restart` refuses a run past it, which
    `schedule set-policy restart` excludes by the rule. No interrupted run waits for a group decision.
  * crash-safe exclusion: the exclusion, its replacement and the dependents' new states are one
    SQLite transaction; the ledger record of an exclusion is written after it and retried at every
    pass until written.
  * reproducible restarts: a restarted run uses the commit its interrupted attempt ran on (from
    that attempt's train_result.json), or a HEAD that differs from it in result files only
    (``provenance.is_output_path``: the same code and configuration), or waits. A restart cut
    short after its archive move is completed at the next start (``recover``; under 'exclude' the
    held run is excluded instead, its archived output moved back first), never taken for a run
    that never started. An operator restarts only a held run, and the policy only the attempt the
    scheduler saw end or a held run (the status is checked inside the write transaction); a run
    another command acted on first is left as that command left it, never restarted, held or
    excluded by the policy.
  * a systematic failure pauses: after MAX_SAME_CAUSE_EXCLUSIONS exclusions of one arm with the same
    cause no further replacement is added until the operator acts. This is an operational pause,
    not a decision point (Table 9.1, Q-exclusion-breaker): the operator reads the failed runs' logs
    and tracebacks, never their costs or returns; a defect in the code or environment is fixed by a
    commit recorded as an erratum (analysis/ERRATA.json; for a memory kill, fewer concurrent runs);
    then `schedule resolve RUN replace` repeats the run with the next unused seed (Part 5.6). Once
    the arm's exclusions with one cause reach its seed target (``seed_target``), it did not complete
    its seeds and enters no comparison (Part 4.1): `replace` is refused unless `--defect-fixed
    ERRATUM_ID` names the erratum, an entry of analysis/ERRATA.json committed at HEAD. Exclusions
    applied by the 'exclude' policy (smoke data roots only) do not count (one outage is not a
    property of the arm), except that after MAX_SAME_CAUSE_EXCLUSIONS successive replacements
    stopped by that policy (each replacing the last: a run stopped the same way again and again)
    the arm pauses as well.
  * smoke and registered runs never mix: the mode (smoke or registered) and the ledger location
    are fixed per data root at its first run or decision (the command line's ``schedule add``,
    ``add-surplus`` and ``add-continuations`` fix the mode, as registered), and smoke ledgers never
    lie inside the repository.
  * pilot runs start only when HEAD contains the machine-hour allocation (Part 6 G2).
  * registered preconditions (registered mode only, each read from the content committed
    at HEAD, never from the working tree):
      search_record_missing  every Study B run, the pilot's two included, waits for the complete
                             Appendix C search record (``studyb.search.check_record``, reading HEAD's
                             files) whose outcome is not "included"; a "partial" outcome also waits
                             for its narrowing amendment, the entry of analysis/AMENDMENTS.json at
                             HEAD whose id the decision names. Part 7.2: "its record is committed to
                             the repository before Study B's first run"; Table C.1 "Decision": "by
                             amendment, recorded before the first run" (answered in Table 9.1,
                             Q-search-before-pilot). Study A's pilot runs do not wait for it, nor do
                             the Table 3.1 determinism checks (software verification, not Study B runs).
      main_sweep_unfinished  a non-pilot Study B training run waits while any run of the main sweep
                             (``manifest.design("main")`` under the data root's cuts, and its queued
                             replacement and surplus seeds, never an auxiliary N = 0 run of Q-warm-start)
                             is not queued yet or has not finished training (pending, training,
                             interrupted or blocked); an excluded run has finished. Part 3.3: "Study
                             B's runs are scheduled after Study A's main sweep" (answered in Table
                             9.1, Q-studyb-order).
      determinism_missing    a run of plug-in study_a, study_a_pid, study_b or unconstrained_ppo waits
                             for a passing registered-form determinism report of that plug-in's
                             configuration under ledger/determinism/ (``determinism_spec``). Table 3.1
                             "Determinism check": "Two runs with the same seed and configuration must
                             produce identical evaluation cost at the first checkpoint before any arm
                             is launched."
      analysis_missing       a non-pilot run waits for the analysis entry point analysis/__main__.py.
                             Part 5.8: "The analysis script is committed to the repository before the
                             first main-study run completes"; gating the launch is stricter than the
                             text, so it always satisfies it.
  * each evaluation harness is tracked by its target (``contracts.evaluator_target``):
    Role 5's missing harness holds only the budget-conditioned runs, never Role 2's evaluations.

Limits (see HANDOVER.md): one workstation per data root; a run that dies from a signal while no
scheduler is its parent (the scheduler was restarted) leaves no exit status, so it cannot be told
apart from an interruption: it counts as one of machine origin, never as a crash and never towards
MAX_RESTARTS. The scheduler imports ``studyb.search`` once per process (the record's files are
re-read at every new HEAD); a HEAD that changes that code holds the Study B runs
(search_record_missing, "restart the scheduler") until the scheduler is restarted and loads it.

Owner: pilot owner (Role 1).
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import math
import os
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import time
import weakref
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from configs import registered as R
from pilot import contracts, errors, manifest, provenance
from pilot.launch import (
    CLAIM_FILE,
    EVALUATION_RESULT,
    EXIT_REFUSED,
    EXIT_UNAVAILABLE,
    INTERRUPT_SIGNALS,
    OMNISAFE_SUBDIR,
    SPEC_FILE,
    TRAIN_RESULT,
    _write_json_atomic,
)
from pilot.ledger_writer import (DEFAULT_PATHS, LedgerPaths, LedgerWriteError, ensure_evaluation_supplement,
                                 mark_ledger_mode, recorded_anywhere, smoke_paths, write_run)
from pilot.manifest import CONTINUATION_GROUPS, RunSpec

ACTIVE = ("training", "evaluating")
DONE_TRAINING = ("trained", "evaluating", "evaluated", "ledgered", "continued", "eval_failed", "ledger_failed")
# main_sweep_unfinished: a main-sweep run in one of these has not finished training.
UNFINISHED_TRAINING = ("pending", "training", "interrupted", "blocked")
INTERRUPT_POLICIES = ("hold", "restart", "exclude")
MAX_SAME_CAUSE_EXCLUSIONS = 2  # operational safety valve: independent failures do not repeat like this
# Q-interrupted-run (answered in Table 9.1, which names the cap): under the 'restart' policy a run that ends without a
# final result of process origin is restarted this many times; the next such end is Part 5.6's "fails to complete its
# steps" (cause 'incomplete'), excluded and replaced like any exclusion
MAX_RESTARTS = 2
POLICY_KEY = "Q-interrupted-run"  # 'restart' records its answer (refused on a registered root while open)
# The origin of an interruption, written at the start of its 'interrupted' event (ORIGIN_PREFIX + origin + ": "):
# 'machine' (stopped from outside the run) is restarted without limit, 'process' counts towards MAX_RESTARTS.
ORIGIN_PREFIX = "origin="
ORIGIN_MACHINE, ORIGIN_PROCESS = "machine", "process"
POLICY_EXCLUSION_DETAIL = "interruption; policy 'exclude' applied to all held runs"
ALLOCATION_PATH = "pilot/allocation.json"
# Modules ``write_run`` and ``ensure_evaluation_supplement`` import lazily (pilot.ledger_writer): imported before the
# check of this process's loaded code (``_ledger_code_problem``), so the code that validates and serialises a ledger
# row and its records is checked before the first write, never after it
LEDGER_WRITE_MODULES = ("results.ledger_schema", "results.supplement_schema")
RESOLVE_ACTIONS = ("restart", "reevaluate", "retry-ledger", "unblock", "replace", "add-auxiliary", "requeue")
# `resolve RUN requeue` replaces the stored spec of a run that never trained by the spec the code checked out
# now builds for its run_id (a run gate or an answer of Table 9.1 changed after it was queued).
REQUEUE_EVENT = "requeued"
# The start of a needs_decision event whose replacement waits only for committed code (``_queueing_code_problem``):
# no group decision is needed, the operator commits and carries it out (`schedule status` shows the event's text).
CODE_PROBLEM_DECISION = "the replacement was not queued: "
# The start of the needs_decision event of an exclusion found in the ledger, not seen by this data root's scheduler
# (``_settle_recorded``: e.g. the state was restored from an older backup); `schedule status` shows the event's text.
RECORDED_DECISION = "the exclusion was found in the ledger: "
# `resolve RUN add-auxiliary` queues the N = 0 run a warm-started replacement depends on when the N = 0 arm never
# received that seed (Table 9.1, Q-warm-start; refused while the key is open; manifest.AUXILIARY_NOTE).
AUXILIARY_KEY = "Q-warm-start"
AUXILIARY_EVENT = "auxiliary"
AUXILIARY_EXCLUDED = (  # an auxiliary run is not a seed of its arm: the next unused seed would serve no one
    "an auxiliary N = 0 run serves only the warm-started run of its own seed: it is never repeated or replaced "
    "(Table 9.1, Q-warm-start), and no value is taken from it; its warm-started dependent, which never trained, "
    "is superseded by its arm's next unused seed (`schedule resolve DEPENDENT replace`)")
# Waiting reasons that only the process which saw them knows; `schedule status` reads them back from
# the event log (the last event of a pending or trained run).
REMEMBERED_BLOCKERS = ("plugin_unavailable", "launch_refused", "launch_failed", "evaluator_unavailable")

# The launch gates, in the order `schedule status` lists them (registered mode only).
LAUNCH_GATES = ("search_record_missing", "main_sweep_unfinished", "determinism_missing", "analysis_missing")
SEARCH_RECORD_DIR = "studyb/search"  # the Appendix C record, the path entered in Table 9.0 (studyb/search/README.md)
# Role 5: check_record(root, read=None) -> SearchStatus(complete, outcome, problems, amendment)
SEARCH_MODULE = "studyb.search"
# Part 7.2: "If it finds such a paper, Study B is withdrawn by amendment"; Appendix C's decision "included".
SEARCH_WITHDRAWING_OUTCOME = "included"
# Table C.1 "Decision": "a partially overlapping work narrows Study B's claims by amendment, recorded before the first
# run"; the decision's amendment (SearchStatus.amendment) is the id of that amendment's entry in the amendment log,
# which must be committed at the same commit (Q-search-before-pilot).
SEARCH_NARROWING_OUTCOME = "partial"
# The amendment log (Part 9, Table 9.1): {"amendments": [{"id": ...}]} (studyb.search.record.AMENDMENT_LOG)
AMENDMENTS_PATH = "analysis/AMENDMENTS.json"
# The errata log (Part 5.8, Table 9.1): {"errata": [{"id": ...}]}; an arm's exclusions past its seed target are
# replaced only after a fixed defect recorded there (`schedule resolve RUN replace --defect-fixed ID`)
ERRATA_PATH = "analysis/ERRATA.json"
ANALYSIS_ENTRY = "analysis/__main__.py"  # `python -m analysis`, Part 5.8's analysis script
DETERMINISM_DIR = "ledger/determinism"  # where scripts/determinism_check.py writes its reports
DETERMINISM_REGISTERED_FORM = "registered form (Table 3.1)"  # a report's "check" for --registered-form
# The plug-ins whose runs the determinism_missing launch gate holds (Table 3.1)
DETERMINISM_PLUGINS = ("study_a", "study_a_pid", "study_b", "unconstrained_ppo")
DETERMINISM_CHECK_PLUGINS = ("ppolag", *DETERMINISM_PLUGINS)  # scripts/determinism_check.py --plugin
# Run gates of an arm that change no update of its training, so the determinism check of the arm's
# configuration does not wait for them. Q-studyb-order and Q-search-before-pilot decide when a Study B run
# may start, not how it trains (the search_record_missing and main_sweep_unfinished gates hold the runs
# themselves); Table 9.1 answers Q-search-before-pilot so: the determinism checks of plug-ins study_b and
# unconstrained_ppo are software verification (Table 3.1), not Study B runs, and do not wait for the search
# record. Q-plasticity-definitions decides only what the plasticity hook logs, and HANDOVER.md section 10
# keeps training bit for bit unchanged by the hook (metrics/hook.py, "Neutrality"); its PENDING text says
# it gates the non-pilot Study A runs only, so keeping it would hold the pilot's Study A runs (plug-in
# study_a, held by determinism_missing) on a key the pilot does not wait on.
DETERMINISM_EXEMPT_KEYS = ("Q-studyb-order", "Q-search-before-pilot", "Q-plasticity-definitions")
# The check of a late arm (study_a_pid: the N = 0.50 arm of the PID check) keeps the arm's onset FRACTION
# N, applied to the check's own length, because the arm's registered onset (N x T = 5,000,000 steps) lies
# beyond the check's first checkpoint and a RunSpec refuses an onset outside its run. So the check passes
# through the onset and the plug-in's controller, but the onset STEP it runs is not the arm's registered
# one (scripts/determinism_check.py records the rule and both steps in the report). The design says only
# "study_a_pid -> the N = 0.50 PID arm" and Table 3.1 names no configuration, so the rule is an answer of the
# amendment log (a PENDING key of configs/registered.py): DETERMINISM_LATE_ONSET_KEY holds the check
# (determinism_open_keys) and the determinism_missing gate of the runs it releases while it is open.
DETERMINISM_LATE_ONSET = (  # answered in Table 9.1 (Q-determinism-late-onset)
    "N x the check's total_steps, in whole epochs")
# The key is registered in configs/registered.PENDING (a test checks it), and the amendment log answers
# it, which releases the PID-check runs. Were it ever missing from PENDING it would count as open
# (fail closed): the check and the runs it releases would wait for it.
DETERMINISM_LATE_ONSET_KEY = "Q-determinism-late-onset"
CUTS_PATH = "pilot/cuts.json"  # the group's cut decision (Part 6.1): {"cuts": [...], "amendment": "..."}
# A child killed by one of these signals crashed (Part 5.6 'crash').
CRASH_SIGNALS = tuple(
    -int(getattr(signal, name)) for name in ("SIGSEGV", "SIGABRT", "SIGBUS", "SIGFPE", "SIGILL")
    if hasattr(signal, name)
)
# A child killed by one of the launcher's INTERRUPT_SIGNALS (SIGTERM, SIGINT, SIGHUP) before it recorded the
# interruption itself was stopped from outside: an interruption of machine origin. Any other signal (SIGKILL, e.g.
# the OOM killer) is one of process origin (Table 9.1, Q-interrupted-run).
STOP_SIGNALS = tuple(-int(s) for s in INTERRUPT_SIGNALS)

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    arm_id TEXT NOT NULL,
    seed INTEGER NOT NULL,
    spec TEXT NOT NULL,
    status TEXT NOT NULL,
    attempt INTEGER NOT NULL DEFAULT 1,
    pid INTEGER,
    host TEXT,
    stage_started TEXT,
    launch_commit TEXT,
    failure_cause TEXT,
    replaces TEXT,
    replaced_by TEXT,
    surplus INTEGER NOT NULL DEFAULT 0,
    ledger_written INTEGER NOT NULL DEFAULT 0,
    train_hours REAL,
    eval_hours REAL,
    updated TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    run_id TEXT,
    event TEXT NOT NULL,
    detail TEXT
);
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

CommandBuilder = Callable[[str, Path, Path], Sequence[str]]


def _count(count: int, noun: str) -> str:
    """``count`` and ``noun``, plural unless the count is 1 ('1 run', '2 runs')."""
    return f"{count} {noun}" + ("" if count == 1 else "s")


def _verb(count: int, singular: str, plural: str) -> str:
    """The verb form that agrees with ``count``."""
    return singular if count == 1 else plural


def default_command(stage: str, spec_path: Path, run_dir: Path, *, allow_dirty: bool, allow_pending: bool) -> list[str]:
    """The launcher's command line for one stage of one run.

    In smoke mode both stages get the smoke flags (pilot/launch.py): ``--allow-pending`` skips the run
    gates of the training stage and opens the result gates of ``pilot.errors`` in both, so a smoke
    evaluation is not refused by a result gate whose key is open (Q-studyb-eval for a
    budget-conditioned run, were it reopened).
    """
    cmd = [sys.executable, "-m", "pilot.launch", stage, "--spec", str(spec_path), "--run-dir", str(run_dir)]
    if allow_dirty:
        cmd.append("--allow-dirty")
    if allow_pending:
        cmd.append("--allow-pending")
    return cmd


# ------------------------------------------------------------------------------
# Content committed at HEAD (``CUTS_PATH``): read with git, never from the working tree
# ------------------------------------------------------------------------------


def committed_files(commit: str, directory: str, repo_root: Path = provenance.REPO_ROOT) -> list[str]:
    """Repository-relative paths of the files under ``directory`` in ``commit`` (none if it has no such tree)."""
    out = subprocess.run(["git", "ls-tree", "-r", "-z", "--name-only", commit, "--", directory.rstrip("/") + "/"],
                         cwd=repo_root, capture_output=True)
    if out.returncode != 0:
        return []
    return sorted(p for p in out.stdout.decode("utf-8", "surrogateescape").split("\0") if p)


def _committed_json(commit: str, rel_path: str, repo_root: Path) -> Any:
    """The JSON value of a file committed at ``commit`` (None if absent or not JSON)."""
    data = provenance.file_committed_at(commit, rel_path, repo_root)
    if data is None:
        return None
    try:
        return json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None


def committed_log_entry(commit: str, rel_path: str, entries: str, entry_id: Any,
                        repo_root: Path = provenance.REPO_ROOT) -> bool:
    """True if the JSON log committed at ``commit`` holds, in its list ``entries``, an entry whose "id" is
    ``entry_id`` (a non-empty string): the amendment log (``AMENDMENTS_PATH``, "amendments") or the errata
    (``ERRATA_PATH``, "errata"). False when the file is not committed, not JSON or not of that form."""
    log = _committed_json(commit, rel_path, repo_root)
    listed = log.get(entries) if isinstance(log, Mapping) else None
    return (isinstance(entry_id, str) and bool(entry_id.strip()) and isinstance(listed, list)
            and any(isinstance(e, Mapping) and e.get("id") == entry_id for e in listed))


def determinism_spec(plugin: str, total_steps: int) -> RunSpec:
    """The run whose determinism Table 3.1 requires before the plug-in's registered runs.

    Table 3.1 "Determinism check": "Two runs with the same seed and configuration must produce
    identical evaluation cost at the first checkpoint before any arm is launched."
    ``scripts/determinism_check.py --plugin P`` trains this spec twice; the scheduler's
    ``determinism_missing`` gate accepts only a report of exactly this spec (its registered form:
    ``total_steps = R.CHECKPOINT_INTERVAL_STEPS``). Each is seed 0 of a registered arm, except ppolag
    (the First Tasks check, no arm of the design):

    * ppolag: OmniSafe's PPO-Lagrangian unchanged on the primary task (the First Tasks check; the
      spec the script always used);
    * study_a: the N = 0 main-sweep arm on SafetyPointGoal1-v0;
    * study_a_pid: the N = 0.50 arm of the PID check (Table 3.3);
    * study_b: the main study's Moderate arm (Table 3.4);
    * unconstrained_ppo: the pilot's unconstrained PPO run (Part 3.6).

    Changed from the arm (``determinism_changes`` lists them; the report records them): the run_id
    (``DET-`` + the arm's), ``total_steps``, its open-question list (below), and the onset step of a
    late arm, which keeps the arm's fraction N of the check's length (DETERMINISM_LATE_ONSET, answered in
    Table 9.1 under DETERMINISM_LATE_ONSET_KEY: Table 3.1 names no arm, and the registered onset of 5,000,000 steps
    lies beyond the check), so the two runs pass through the onset and the plug-in's controller before
    the checkpoint compared. The study_a_pid check therefore runs a changed onset, not the arm's
    registered configuration, and waits on that key while it is open (``determinism_open_keys``). The spec keeps the
    arm's open questions except DETERMINISM_EXEMPT_KEYS, which change no update, and adds the
    check's own (``determinism_config_keys``): a check cannot run while its configuration waits on
    an open question (the launcher refuses it; the script says which), and an exempt key would hold
    it for no reason (for the pilot's Study A runs, on a key the pilot does not wait on).
    """
    if total_steps <= 0 or total_steps % R.STEPS_PER_EPOCH:
        raise ValueError(f"total_steps must be a positive multiple of {R.STEPS_PER_EPOCH}, got {total_steps}")
    arm = determinism_arm(plugin)
    if arm is None:  # ppolag
        seed = R.SEEDS[0]
        return RunSpec(run_id=f"DET-PPOLag-{manifest.short_task(R.PRIMARY_TASK)}-s{seed}", study="A",
                       task=R.PRIMARY_TASK, arm="determinism-check", seed=seed, total_steps=total_steps,
                       base_algo="PPOLag", plugin="ppolag", group="determinism")
    onset = arm.onset_step
    if onset:  # DETERMINISM_LATE_ONSET: the arm's fraction N of the check's own length, in whole epochs
        onset = int(Fraction(str(arm.N)) * total_steps) // R.STEPS_PER_EPOCH * R.STEPS_PER_EPOCH
    # The check's own key joins its run gates (the `k in R.PENDING` test fails closed should the key ever be
    # missing from PENDING: RunSpec.open_questions refuses an unknown key, and determinism_open_keys then
    # reports it as open).
    own = [k for k in determinism_config_keys(plugin) if k in R.PENDING and k not in arm.pending]
    return RunSpec.from_dict({
        **arm.to_dict(), "run_id": f"DET-{arm.run_id}", "total_steps": total_steps, "onset_step": onset,
        "pending": [k for k in arm.pending if k not in DETERMINISM_EXEMPT_KEYS] + own,
    })


def determinism_config_keys(plugin: str) -> tuple[str, ...]:
    """The PENDING keys of the determinism check's own configuration choices for ``plugin``.

    Only a late arm's check changes the arm's onset step (DETERMINISM_LATE_ONSET), so only study_a_pid
    carries DETERMINISM_LATE_ONSET_KEY; the other checks run their arm's registered configuration.
    """
    arm = determinism_arm(plugin)
    return (DETERMINISM_LATE_ONSET_KEY,) if arm is not None and arm.onset_step else ()


def determinism_open_keys(plugin: str) -> tuple[str, ...]:
    """The keys of ``determinism_config_keys(plugin)`` still open, a key not yet in PENDING included.

    Fail closed: a rule whose key the integrator has not registered yet has not been answered, so
    ``scripts/determinism_check.py`` refuses the check (it names the key) and the scheduler's
    ``determinism_missing`` gate holds the runs the check releases, whatever report HEAD holds.
    """
    return tuple(k for k in determinism_config_keys(plugin) if k not in R.PENDING or R.is_open(k))


def determinism_arm(plugin: str) -> RunSpec | None:
    """The registered arm (seed 0) whose configuration ``determinism_spec(plugin, ...)`` checks (None for ppolag)."""
    if plugin not in DETERMINISM_CHECK_PLUGINS:
        raise ValueError(f"plugin must be one of {DETERMINISM_CHECK_PLUGINS}, got {plugin!r}")
    if plugin == "ppolag":
        return None
    seed = R.SEEDS[0]
    if plugin == "study_a":
        return next(s for s in manifest.study_a_main((seed,))
                    if s.task == R.PRIMARY_TASK and s.N == R.ONSET_FRACTIONS[0])
    if plugin == "study_a_pid":
        late = max(R.LATE_ONSET_FRACTIONS)
        return next(s for s in manifest.study_a_pid((seed,)) if s.task == R.PRIMARY_TASK and s.N == late)
    if plugin == "study_b":
        moderate = next(s for s in manifest.pilot() if s.plugin == "study_b").arm  # Part 3.6's Moderate arm
        return next(s for s in manifest.study_b((seed,)) if s.arm == moderate)
    return next(s for s in manifest.pilot() if s.plugin == "unconstrained_ppo")


def determinism_changes(plugin: str, total_steps: int) -> dict[str, list[Any]]:
    """{field: [the arm's value, the check's value]} for every field the check changes.

    Recorded in the determinism report, so that it says which configuration was checked: the
    study_a_pid check runs its onset at N x total_steps (DETERMINISM_LATE_ONSET), not at the arm's
    registered 5,000,000 steps. ``pending`` is listed when DETERMINISM_EXEMPT_KEYS drop a key or the
    check's own keys (``determinism_config_keys``) are added.
    """
    arm = determinism_arm(plugin)
    if arm is None:
        return {}
    before = json.loads(json.dumps(arm.to_dict()))  # JSON values (lists, not tuples), as the report holds them
    after = json.loads(json.dumps(determinism_spec(plugin, total_steps).to_dict()))
    return {k: [v, after[k]] for k, v in before.items() if after[k] != v}


def _spec_without_gates(data: Any) -> Any:
    """The spec mapping without its open-question list (``data`` itself when it is not a mapping)."""
    return {k: v for k, v in data.items() if k != "pending"} if isinstance(data, Mapping) else data


def determinism_report_plugin(report: Any) -> str | None:
    """The plug-in whose registered determinism check ``report`` passed, or None.

    A report counts only if it is the registered form (``check`` DETERMINISM_REGISTERED_FORM, Table
    3.1: identical evaluation cost at the first checkpoint), passed, was not a rehearsal (no
    ``--allow-dirty`` or ``--allow-pending``), ran from a clean worktree, both of its runs trained on
    the commit it names (``run_commits``), names its plug-in (``plugin``; a report without one is
    the ppolag check the script always ran) and holds exactly ``determinism_spec(plugin,
    R.CHECKPOINT_INTERVAL_STEPS)`` apart from its open-question list, so a report of an older
    configuration does not release the runs of a changed one.
    """
    if not isinstance(report, Mapping):
        return None
    result = report.get("result")
    if (report.get("check") != DETERMINISM_REGISTERED_FORM or report.get("passed") is not True
            or report.get("rehearsal_only") is not False or report.get("worktree_dirty") is not False
            or report.get("allow_pending") is True
            or not isinstance(result, Mapping) or result.get("identical") is not True):
        return None
    commit = report.get("commit_hash")  # both runs trained on the commit the report names
    if not isinstance(commit, str) or not provenance.is_full_commit_hash(commit) \
            or report.get("run_commits") != [commit, commit]:
        return None
    spec = report.get("spec")
    plugin = report.get("plugin", spec.get("plugin") if isinstance(spec, Mapping) else None)
    if plugin not in DETERMINISM_CHECK_PLUGINS:
        return None
    expected = determinism_spec(plugin, R.CHECKPOINT_INTERVAL_STEPS).to_dict()
    if _spec_without_gates(spec) != _spec_without_gates(json.loads(json.dumps(expected))):
        return None
    return plugin


def committed_determinism_reports(commit: str, repo_root: Path = provenance.REPO_ROOT) -> dict[str, str]:
    """{plug-in: the first passing registered-form report committed at ``commit``} under DETERMINISM_DIR."""
    found: dict[str, str] = {}
    for path in committed_files(commit, DETERMINISM_DIR, repo_root):
        if path.endswith(".json"):
            plugin = determinism_report_plugin(_committed_json(commit, path, repo_root))
            if plugin is not None:
                found.setdefault(plugin, path)
    return found


def committed_reader(commit: str, root: Path, repo_root: Path = provenance.REPO_ROOT) -> Callable[[Any], bytes | None]:
    """``read(path) -> bytes | None``: a file's content as committed at ``commit`` (None if it is not).

    The reader ``studyb.search.check_record(root, read=...)`` is given (studyb/search/record.py), so the gate
    judges the record HEAD holds, never an uncommitted working copy. ``path`` may be relative to
    ``root`` (the record directory), repository-relative (``studyb/search/...``) or absolute inside
    the repository; anything outside the repository reads as absent.
    """
    repo = Path(os.path.abspath(repo_root))
    base = Path(os.path.abspath(root))
    base_rel = PurePosixPath(base.relative_to(repo).as_posix())

    def read(path: Any) -> bytes | None:
        given = PurePosixPath(os.fspath(path).replace(os.sep, "/"))
        if given.is_absolute():
            try:
                rel = PurePosixPath(Path(os.path.normpath(str(given))).relative_to(repo).as_posix())
            except ValueError:
                return None
        elif given.parts[: len(base_rel.parts)] == base_rel.parts:
            rel = given  # already relative to the repository
        else:
            rel = base_rel / given
        normal = os.path.normpath(str(rel)).replace(os.sep, "/")
        if normal == "." or normal == ".." or normal.startswith("../") or normal.startswith("/"):
            return None
        return provenance.file_committed_at(commit, normal, repo_root)

    return read


# {module: (its file, sha256 of its source)} as this process loaded it: noted when the scheduler first sees
# the module (``_note_loaded_sources``, right after each import of the checker, successful or not), so the
# gate compares the code in memory, not the file on disk now, with HEAD (a checker that a pull or commit
# changes while the scheduler runs stays loaded as it was; the gate then holds until the scheduler is restarted).
_LOADED_SOURCES: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _package_modules(package: str) -> list[Any]:
    return [m for name, m in list(sys.modules.items())
            if m is not None and (name == package or name.startswith(package + "."))]


def _note_loaded_sources(package: str) -> None:
    """Note the source digest of every loaded module of ``package`` not noted yet (its first sight)."""
    for module in _package_modules(package):
        file = getattr(module, "__file__", None)
        if not file or module in _LOADED_SOURCES:
            continue
        try:
            digest: str | None = hashlib.sha256(Path(file).read_bytes()).hexdigest()
        except OSError:
            digest = None
        _LOADED_SOURCES[module] = (os.path.abspath(file), digest)


def _import_search_module() -> Any:
    """``studyb.search`` (Role 5); ImportError when it cannot be imported (missing from the checkout, or broken).
    A separate function for tests.

    The sources of the modules it loaded are noted right after the import (``_LOADED_SOURCES``),
    whether it succeeded or failed: a failed import leaves the submodules it had loaded before the
    failure in ``sys.modules``, and the next import reuses them, so they are noted as they were loaded
    (else a later HEAD would note its own file for code still in memory from the failed attempt).
    Only a change on disk during the import itself would go unseen.
    """
    importlib.invalidate_caches()  # a package added after this process started is found
    try:
        return importlib.import_module(SEARCH_MODULE)
    finally:
        _note_loaded_sources(SEARCH_MODULE.split(".")[0])


def _uncommitted_modules(package: str, commit: str, repo_root: Path) -> list[str]:
    """Loaded modules of ``package`` inside the repository whose loaded source ``commit`` does not hold.

    The gate's own code must be the committed code, like a run's (``provenance.unverified_imported_code``).
    Compared is the source as this process loaded it (``_LOADED_SOURCES``), not the file on disk now:
    after a pull or commit that changes the checker, the file equals HEAD but the code in memory is
    the old one. A module first seen now (for example one the checker imported lazily) is noted as
    it is on disk now.
    """
    _note_loaded_sources(package)
    repo = Path(os.path.abspath(repo_root))
    problems = []
    for module in _package_modules(package):
        if module not in _LOADED_SOURCES:
            continue  # no file: nothing on disk to compare
        file, loaded = _LOADED_SOURCES[module]
        try:
            rel = Path(file).relative_to(repo).as_posix()
        except ValueError:
            continue  # not this repository's code
        committed = provenance.file_committed_at(commit, rel, repo_root)
        if committed is None or loaded is None or hashlib.sha256(committed).hexdigest() != loaded:
            problems.append(rel)
    return sorted(set(problems))


def search_record_status(commit: str, repo_root: Path = provenance.REPO_ROOT) -> tuple[bool, str]:
    """(released, reason) of the search_record_missing gate at ``commit`` (studyb/search/README.md).

    Part 7.2: "its record is committed to the repository before Study B's first run"; "If it finds
    such a paper, Study B is withdrawn by amendment". Released only when ``check_record`` (Role 5)
    finds the record committed at ``commit`` complete and its outcome is not "included"; a "partial"
    outcome only when ``commit`` also holds its narrowing amendment in the amendment log (Table C.1
    "Decision": "by amendment, recorded before the first run"; answered in Table 9.1,
    Q-search-before-pilot): an entry of ``AMENDMENTS_PATH`` whose id is the decision's amendment
    (``SearchStatus.amendment``; a status without one holds). Missing while ``studyb.search`` cannot
    be imported (missing from the checkout, or broken: a gate never crashes the scheduler), while that
    code is not committed as this process loaded it (``_uncommitted_modules``), or when ``check_record``
    fails.
    """
    try:
        check_record = _import_search_module().check_record
    except Exception as exc:  # noqa: BLE001 - ImportError when Role 5's code cannot be imported; any other alike
        return False, (f"{SEARCH_MODULE}.check_record is not available ({type(exc).__name__}: {exc}); Role 5 commits "
                       f"the Appendix C record and its checker under {SEARCH_RECORD_DIR}/")
    uncommitted = _uncommitted_modules(SEARCH_MODULE.split(".")[0], commit, repo_root)
    if uncommitted:
        return False, (f"the record's checker is not committed as loaded: {', '.join(uncommitted[:3])} (commit it; "
                       "if HEAD changed it after this scheduler loaded it, restart the scheduler)")
    root = Path(os.path.abspath(repo_root)) / SEARCH_RECORD_DIR
    try:
        status = check_record(root, read=committed_reader(commit, root, repo_root))
    except Exception as exc:  # noqa: BLE001 - a failing check holds the runs, it does not stop the scheduler
        return False, f"check_record failed on the committed record: {type(exc).__name__}: {exc}"
    complete, outcome = getattr(status, "complete", None), getattr(status, "outcome", None)
    problems = list(getattr(status, "problems", None) or [])
    if complete is not True:
        shown = "; ".join(str(p) for p in problems[:3]) or "no detail"
        return False, (f"the search record committed at HEAD is not complete ({_count(len(problems), 'problem')}: "
                       f"{shown}); Part 7.2: the record is committed before Study B's first run")
    if outcome == SEARCH_WITHDRAWING_OUTCOME:
        return False, ("the search record's outcome is 'included': Study B is withdrawn by amendment (Part 7.2); "
                       "its runs wait for the group's amendment")
    if outcome == SEARCH_NARROWING_OUTCOME:
        amendment = getattr(status, "amendment", None)
        if not committed_log_entry(commit, AMENDMENTS_PATH, "amendments", amendment, repo_root):
            return False, (f"the search outcome is 'partial': its narrowing amendment {amendment!r} is not "
                           f"committed in {AMENDMENTS_PATH} (Table C.1 'Decision': recorded before the first run)")
        return True, (f"the search record committed at HEAD is complete (outcome 'partial'; its narrowing amendment "
                      f"{amendment!r} is committed in {AMENDMENTS_PATH})")
    return True, f"the search record committed at HEAD is complete (outcome {outcome!r})"


def committed_cuts(count: int, repo_root: Path = provenance.REPO_ROOT) -> tuple[str, ...]:
    """The first ``count`` cuts of the group's decision in the committed ``pilot/cuts.json`` (``CUTS_PATH``).

    Serves ``schedule add --cuts K``. Part 6.1: "Cuts are made in this order, each applied only if
    the previous is insufficient." The file is ``{"cuts": [...], "amendment": "..."}``: a prefix of
    ``manifest.CUTS`` and the amendment that records the decision. Refused (SchedulerError) unless
    HEAD holds the file and the working copy equals it (an uncommitted decision is not a decision),
    or when ``count`` exceeds its list.
    """
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise SchedulerError(f"the number of cuts must be a whole number from 0, got {count!r}")
    if count == 0:
        return ()
    head = provenance.commit_hash(repo_root)
    committed = provenance.file_committed_at(head, CUTS_PATH, repo_root)
    if committed is None:
        raise SchedulerError(f"{CUTS_PATH} is not committed at HEAD; commit the group's cut decision (Part 6.1) first")
    working = Path(repo_root) / CUTS_PATH
    if not working.is_file() or working.read_bytes() != committed:
        raise SchedulerError(f"{CUTS_PATH} has uncommitted changes; commit them (or restore HEAD's copy) first")
    try:
        data = json.loads(committed.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise SchedulerError(f"{CUTS_PATH} is not JSON: {exc}") from exc
    cuts = data.get("cuts") if isinstance(data, dict) else None
    amendment = data.get("amendment") if isinstance(data, dict) else None
    if not isinstance(cuts, list) or not all(isinstance(c, str) for c in cuts):
        raise SchedulerError(f"{CUTS_PATH} must hold {{\"cuts\": [...], \"amendment\": \"...\"}}")
    if not isinstance(amendment, str) or not amendment.strip():
        raise SchedulerError(f"{CUTS_PATH} names no amendment; the cut decision is recorded by one (Table 9.1)")
    if tuple(cuts) != manifest.CUTS[: len(cuts)]:
        raise SchedulerError(f"{CUTS_PATH} lists {cuts}, not a prefix of the registered order "
                             f"{list(manifest.CUTS)} (Part 6.1)")
    if count > len(cuts):
        raise SchedulerError(f"--cuts {count} exceeds the {len(cuts)} cuts decided in {CUTS_PATH} ({cuts})")
    return tuple(cuts[:count])


# Registered runs keep their data root at /data (HANDOVER.md section 6: the ledger stores checkpoint paths relative
# to the schema's /data/checkpoints). Smoke mode (--allow-dirty / --allow-pending) is refused on it and inside it, so
# a smoke command that leaves out --data-root never fixes /data as a smoke root (as scripts/smoke_run.py, which reads
# these). $WSCL_DATA_ROOT pointing elsewhere names a smoke root.
REGISTERED_DATA_ROOTS = ("/data",)


def registered_root(data_root: Path, roots: tuple[str, ...] | None = None) -> str | None:
    """The registered data root (of ``roots``, default ``REGISTERED_DATA_ROOTS``) that ``data_root`` is, or lies in
    (None if neither)."""
    real = Path(os.path.realpath(data_root))
    for root in REGISTERED_DATA_ROOTS if roots is None else roots:
        registered = Path(os.path.realpath(root))
        if real == registered or registered in real.parents:
            return root
    return None


@dataclass
class SchedulerConfig:
    """Settings of one data root's scheduler (smoke mode when ``allow_dirty`` or ``allow_pending``)."""

    data_root: Path = field(default_factory=lambda: Path(os.environ.get("WSCL_DATA_ROOT", "/data")))
    max_concurrent: int = 1
    poll_seconds: float = 30.0
    # None: the policy stored for the data root, else ``Scheduler._default_policy`` ('restart' on a registered root
    # once Q-interrupted-run is answered, as it is; 'hold' on a smoke root)
    on_interrupt: str | None = None
    allow_dirty: bool = False  # smoke tests only
    allow_pending: bool = False  # smoke tests only
    ledger_paths: LedgerPaths | None = None  # None: the repository's ledgers, or smoke ledgers for smoke tests
    command_builder: CommandBuilder | None = None  # tests inject a fake launcher here
    require_allocation_for_pilot: bool = True
    # The repository whose HEAD the launch gates, the allocation and the cut decision are read from
    # (tests point it at a temporary repository; the launcher always runs from provenance.REPO_ROOT).
    repo_root: Path = provenance.REPO_ROOT

    def __post_init__(self) -> None:
        # Absolute (not resolved: /data stays /data for the ledger's portable paths), so that the
        # command lines of the runs name the same directory whatever the caller's working directory.
        self.data_root = Path(os.path.abspath(self.data_root))
        self.repo_root = Path(os.path.abspath(self.repo_root))
        if self.on_interrupt is not None and self.on_interrupt not in INTERRUPT_POLICIES:
            raise ValueError(f"on_interrupt must be one of {INTERRUPT_POLICIES}")
        if self.max_concurrent < 1:
            raise ValueError("max_concurrent must be at least 1")
        if not (math.isfinite(self.poll_seconds) and self.poll_seconds >= 0):
            raise ValueError("poll_seconds must be a finite, non-negative number")
        if self.smoke:
            registered = registered_root(self.data_root)
            if registered is not None:
                raise ValueError(f"{self.data_root} is (or lies in) the registered data root {registered}: smoke runs "
                                 "(--allow-dirty / --allow-pending) need a scratch --data-root")
        if self.ledger_paths is None:
            self.ledger_paths = smoke_paths(self.data_root) if self.smoke else DEFAULT_PATHS
        if self.smoke:
            repo = provenance.REPO_ROOT.resolve()
            lp = self.ledger_paths
            if any(Path(p).resolve().is_relative_to(repo) for p in (lp.main, lp.pilot, lp.sidecar_dir)):
                raise ValueError("smoke runs (--allow-dirty / --allow-pending) may not write ledgers inside the "
                                 "repository")

    @property
    def smoke(self) -> bool:
        return self.allow_dirty or self.allow_pending

    @property
    def mode(self) -> str:
        return "smoke" if self.smoke else "registered"

    @property
    def runs_root(self) -> Path:
        return self.data_root / "checkpoints"

    @property
    def state_dir(self) -> Path:
        return self.data_root / "scheduler"

    @property
    def state_path(self) -> Path:
        return self.state_dir / "state.sqlite"

    def run_dir(self, run_id: str) -> Path:
        return self.runs_root / run_id


class SchedulerError(RuntimeError):
    """A refusal of the scheduler or of an operator command."""


class StatusChanged(SchedulerError):
    """The run's status is no longer the one the caller checked: another command acted on it first."""


class Scheduler:
    """Launches, resumes and settles the runs of one data root (see the module docstring)."""

    def __init__(self, config: SchedulerConfig) -> None:
        self.cfg = config
        self.cfg.state_dir.mkdir(parents=True, exist_ok=True)
        (self.cfg.state_dir / "logs").mkdir(exist_ok=True)
        self.cfg.runs_root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.cfg.state_path, timeout=60.0)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self._procs: dict[str, subprocess.Popen] = {}
        self._load_ledger_code()  # with the rest of this process's code; a failure waits at the first write
        self._host = socket.gethostname()
        # Evaluation harnesses found missing, by contracts.evaluator_target: Role 5's
        # missing harness holds the budget-conditioned runs only.
        self._unavailable_evaluators: set[str] = set()
        self._unavailable_plugins: set[str] = set()
        self._refused: set[str] = set()
        self._launch_failed: set[str] = set()
        self._notified: set[tuple[str, str]] = set()
        self._head: str | None = None
        self._clean: bool | None = None
        # The commit this process loaded its code from (HEAD at its first pass): a replacement computed later by
        # that code must be the commit's (``_queueing_code_problem``).
        self._code_head: str | None = None
        # Facts of the launch gates: those read from HEAD are kept while HEAD stays the
        # same; the main sweep's progress is re-read at every pass (_refresh_git).
        self._facts_head: str | None = None
        self._facts: dict[str, Any] = {}
        self._main_unfinished: tuple[list[str], list[str]] | None = None
        # (allow_dirty, allow_pending) under which the waiting reasons are judged, when not the command's own:
        # `schedule status` judges a data root's runs as the scheduler that runs them does (_status_flags).
        self._judge: tuple[bool, bool] | None = None

    @property
    def _allow_dirty(self) -> bool:
        return self.cfg.allow_dirty if self._judge is None else self._judge[0]

    @property
    def _allow_pending(self) -> bool:
        return self.cfg.allow_pending if self._judge is None else self._judge[1]

    @property
    def _smoke(self) -> bool:
        return self._allow_dirty or self._allow_pending

    @contextlib.contextmanager
    def _judged_as(self, flags: tuple[bool, bool] | None) -> Iterator[None]:
        """Judge launch and evaluation blockers under ``flags`` (None: the command's own) inside the block."""
        previous, self._judge = self._judge, flags
        try:
            yield
        finally:
            self._judge = previous

    @property
    def paths(self) -> LedgerPaths:
        assert self.cfg.ledger_paths is not None
        return self.cfg.ledger_paths

    # -- bookkeeping -----------------------------------------------------------

    def _now(self) -> str:
        return provenance.utc_now().isoformat()

    @contextlib.contextmanager
    def _write_txn(self) -> Iterator[None]:
        """One write transaction that holds SQLite's write lock from its first read.

        Reads that decide a write (the next unused seed, the exclusion count) happen inside it, so
        an operator command running beside the scheduler can never pick the same seed.
        """
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self.db.rollback()
            raise
        else:
            self.db.commit()

    def _event_sql(self, run_id: str | None, event: str, detail: str = "") -> None:
        self.db.execute(
            "INSERT INTO events (ts, run_id, event, detail) VALUES (?, ?, ?, ?)", (self._now(), run_id, event, detail)
        )

    def _event(self, run_id: str | None, event: str, detail: str = "") -> None:
        with self.db:
            self._event_sql(run_id, event, detail)

    def _event_once(self, run_id: str, event: str, detail: str = "") -> None:
        """Log a waiting reason once per scheduler process, not on every poll."""
        if (run_id, event) not in self._notified:
            self._notified.add((run_id, event))
            self._event(run_id, event, detail)

    def _set_sql(self, run_id: str, **values: object) -> None:
        values["updated"] = self._now()
        cols = ", ".join(f"{k} = ?" for k in values)
        self.db.execute(f"UPDATE runs SET {cols} WHERE run_id = ?", (*values.values(), run_id))

    def _set(self, run_id: str, **values: object) -> None:
        with self.db:
            self._set_sql(run_id, **values)

    def row(self, run_id: str) -> sqlite3.Row:
        """The run's state row; KeyError if the run is not queued on this data root."""
        found = self.db.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if found is None:
            raise KeyError(run_id)
        return found

    def exists(self, run_id: str) -> bool:
        """True if the run is queued on this data root, whatever its status."""
        return self.db.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,)).fetchone() is not None

    def spec(self, run_id: str) -> RunSpec:
        """The run's stored spec; KeyError if the run is not queued."""
        return RunSpec.from_json(self.row(run_id)["spec"])

    def rows(self, status: str | Iterable[str] | None = None) -> list[sqlite3.Row]:
        """The state rows of the runs in ``status`` (one status or several; None: every run), in queue order."""
        if status is None:
            return list(self.db.execute("SELECT * FROM runs ORDER BY rowid"))
        statuses = (status,) if isinstance(status, str) else tuple(status)
        marks = ",".join("?" * len(statuses))
        return list(self.db.execute(f"SELECT * FROM runs WHERE status IN ({marks}) ORDER BY rowid", statuses))

    def setting(self, key: str) -> str | None:
        """The stored value of a per-data-root setting, or None if it is not set."""
        found = self.db.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return None if found is None else found["value"]

    def _ledger_location(self) -> str:
        """The ledger paths as stored per data root: real paths (any spelling of the data root is the
        same root), relative to the repository when inside it (a moved or fresh clone still writes
        the repository's ledgers)."""
        repo = Path(os.path.realpath(provenance.REPO_ROOT))

        def where(p: Path) -> str:
            real = Path(os.path.realpath(p))
            return "<repo>/" + real.relative_to(repo).as_posix() if real.is_relative_to(repo) else str(real)

        return json.dumps([where(p) for p in (self.paths.main, self.paths.pilot, self.paths.sidecar_dir)])

    def fix_mode(self, *, ledgers: bool = True) -> None:
        """Fix this data root's mode and ledger location now, or refuse (``SchedulerError``) if they differ.

        ``run``, ``resolve`` and ``set-policy`` do this themselves; a caller that queues runs of one
        mode (``scripts/smoke_run.py``) calls it first, so runs of the wrong mode are never queued.
        ``ledgers=False`` fixes and checks the mode only (``schedule add`` takes no ``--ledger-dir``).
        """
        self._fix_mode(ledgers=ledgers)

    def _check_mode(self) -> str | None:
        """The stored mode (None before the first run or decision), or SchedulerError if this command's differs."""
        stored = self.setting("mode")
        if stored is not None and stored != self.cfg.mode:
            hint = ("pass --allow-dirty or --allow-pending" if stored == "smoke"
                    else "drop --allow-dirty and --allow-pending")
            raise SchedulerError(f"{self.cfg.data_root} holds {stored} runs; {hint} (the mode is fixed per data root)")
        return stored

    def _fix_mode(self, *, ledgers: bool = True) -> None:
        """The mode (smoke or registered) and the ledger location are fixed by the first run or decision;
        the command line's ``schedule add``, ``add-surplus`` and ``add-continuations`` fix the mode only
        (``ledgers=False``).

        Smoke and registered runs never share a data root, and every command that writes ledger
        records (run, resolve, set-policy) writes them to the same ledgers.
        """
        with self._write_txn():  # read and fixed under one write lock: two first commands never both pass
            stored = self._check_mode()
            stored_ledgers = self.setting("ledgers") if ledgers else None
            if stored_ledgers is not None and stored_ledgers != self._ledger_location():
                main = json.loads(stored_ledgers)[0]
                raise SchedulerError(
                    f"{self.cfg.data_root} writes its ledgers beside {main}; this command would write them beside "
                    f"{json.loads(self._ledger_location())[0]}. Use the same --ledger-dir as its first run (none for "
                    "the default)"
                )
            if stored is None:
                self.db.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('mode', ?)", (self.cfg.mode,))
                self._event_sql(None, "mode", self.cfg.mode)
            if ledgers:  # the ledgers themselves carry the mode: two data roots may name the same --ledger-dir
                try:
                    mark_ledger_mode(self.paths, self.cfg.mode)
                except LedgerWriteError as exc:
                    raise SchedulerError(f"{self.cfg.data_root}: {exc}") from None
            if ledgers and stored_ledgers is None:
                self.db.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('ledgers', ?)",
                                (self._ledger_location(),))
                self._event_sql(None, "ledgers", json.loads(self._ledger_location())[0])

    # -- adding runs -----------------------------------------------------------

    def _insert_sql(self, spec: RunSpec, *, status: str = "pending", replaces: str | None = None,
                    surplus: bool = False) -> bool:
        existing = self.db.execute("SELECT spec FROM runs WHERE run_id = ?", (spec.run_id,)).fetchone()
        if existing is not None:
            if RunSpec.from_json(existing["spec"]) != spec:
                raise SchedulerError(f"{spec.run_id} is already scheduled with a different spec; a queued spec is "
                                     "never changed by `add` (a run that never trained: `schedule resolve RUN "
                                     "requeue`)")
            return False
        self.db.execute(
            "INSERT INTO runs (run_id, arm_id, seed, spec, status, replaces, surplus, updated) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (spec.run_id, spec.arm_id, spec.seed, spec.to_json(), status, replaces, int(surplus), self._now()),
        )
        detail = "; ".join(x for x in (f"replaces={replaces}" if replaces else "", "surplus" if surplus else "") if x)
        self._event_sql(spec.run_id, "added" if status == "pending" else f"added_{status}", detail)
        return True

    def record_queued(self, what: str, commit: str | None) -> None:
        """Log the commit whose code computed the runs a command queued (``schedule add``, ``add-surplus``,
        ``add-continuations``; the command line refuses them from an uncommitted tree)."""
        self._event(None, "queued", f"{what} (code at commit {commit})")

    def add(self, specs: Iterable[RunSpec], *, cuts: int | None = None) -> int:
        """Insert runs as pending (``schedule add``). A run_id already present must carry an identical spec.

        ``cuts``: the number K of the group's cuts to apply (``schedule add --cuts K``; Part 6.1;
        ``CUTS_PATH``): the first K cuts of the committed ``pilot/cuts.json`` (``committed_cuts``), applied by
        ``manifest.apply_cuts``. The count is fixed per data root at its first use with K >= 1 (K = 0
        applies no cut and fixes nothing), never undone or changed ("each applied only if the
        previous is insufficient": a data root holds one cut design), and refused while the data
        root holds a run the cuts would drop or change. Once fixed, every later add (``cuts`` None),
        surplus seed, replacement and continuation of this data root is cut the same way.
        """
        specs = list(specs)
        names = committed_cuts(cuts, self.cfg.repo_root) if cuts else ()
        added = 0
        with self._write_txn():
            stored = self.setting("cuts")
            if cuts is not None and stored is not None and int(stored) != cuts:
                raise SchedulerError(f"this data root is cut by the first {stored} cuts of {CUTS_PATH} "
                                     f"({', '.join(manifest.CUTS[:int(stored)])}); --cuts {cuts} would change that "
                                     "(Part 6.1)")
            if names and stored is None:
                self._fix_cuts_sql(names)
            before = len(specs)
            specs = self._cut(specs)
            if len(specs) < before:
                self._event_sql(None, "cuts_applied",
                                f"{before - len(specs)} of {before} runs left out by the cut order (Part 6.1)")
            for spec in specs:
                added += self._add_sql(spec)
        return added

    def _fix_cuts_sql(self, names: tuple[str, ...]) -> None:
        """Record the cut count for this data root (inside a write transaction), if nothing queued is cut."""
        queued = [RunSpec.from_json(r["spec"]) for r in self.rows()]
        try:
            kept = {s.run_id: s for s in manifest.apply_cuts(queued, names)}
        except ValueError as exc:
            raise SchedulerError(f"the cuts {names} cannot apply to the runs already queued: {exc}") from exc
        lost = [s.run_id for s in queued if kept.get(s.run_id) != s]
        if lost:
            raise SchedulerError(f"this data root already holds {_count(len(lost), 'run')} that the cuts {names} drop "
                                 f"or change (e.g. {lost[0]}); decide the cuts before queueing them (Part 6.1)")
        self.db.execute("INSERT INTO settings (key, value) VALUES ('cuts', ?)", (str(len(names)),))
        head = provenance.commit_hash(self.cfg.repo_root)
        self._event_sql(None, "cuts", f"{', '.join(names)} (Part 6.1; {CUTS_PATH} at {head})")

    def _cut(self, specs: list[RunSpec]) -> list[RunSpec]:
        """``specs`` under the cuts fixed for this data root (none until ``add(..., cuts=K)`` fixed them)."""
        stored = self.setting("cuts")
        if not stored or not specs:
            return specs
        try:
            return manifest.apply_cuts(specs, manifest.CUTS[: int(stored)])
        except ValueError as exc:
            raise SchedulerError(str(exc)) from exc

    def next_unused_seed(self, arm_id: str, also_used: Iterable[int] = ()) -> int:
        """The smallest seed >= 5 the arm has not used for any purpose (Parts 5.5 and 5.6; answered in Table 9.1,
        Q-seed-collision)."""
        used = {r["seed"] for r in self.db.execute("SELECT seed FROM runs WHERE arm_id = ?", (arm_id,))}
        used |= set(also_used)
        seed = R.NEXT_UNUSED_SEED_START
        while seed in used:
            seed += 1
        return seed

    @staticmethod
    def _with_seed_rule(spec: RunSpec) -> RunSpec:
        """Replacement and collision seeds follow the seed rule of Q-seed-collision (answered in Table 9.1); the key
        they carry holds them only while it is open."""
        if "Q-seed-collision" in spec.pending:
            return spec
        return RunSpec.from_dict({**spec.to_dict(), "pending": [*spec.pending, "Q-seed-collision"]})

    def add_surplus(self, extra: int) -> list[str]:
        """Part 5.5: ``extra`` surplus seeds for every scheduled arm of the primary-comparison set.

        ``extra`` is a per-arm target, fixed per data root ("the same count for every arm in the
        set"): calling again with the same value adds only the seeds an arm still lacks, for example
        for the Study B arms queued after Study A's surplus. The set is the N = 0 and N = 0.50 main
        arms on SafetyPointGoal1-v0, both onset shapes under both controls, and the seven Study B arms
        (answered in Table 9.1, Q-surplus-arm-set; ``manifest.is_primary_comparison``). Surplus seed i
        (0-based) is 5 + i, or the arm's next unused seed when a replacement used that one (answered in
        Table 9.1, Q-seed-collision; such a seed carries the key, as a ramp arm's carries
        Q-surplus-arm-set, which holds it only while the key is open).
        """
        if extra < 0 or extra > R.MAX_SEEDS_PER_ARM - len(R.SEEDS):
            raise ValueError(f"extra seeds must be between 0 and {R.MAX_SEEDS_PER_ARM - len(R.SEEDS)}")
        added: list[str] = []
        with self._write_txn():
            stored = self.setting("surplus_extra")
            if stored is not None and int(stored) != extra:
                raise SchedulerError(
                    f"the surplus is fixed at {stored} seeds per arm on this data root "
                    "(Part 5.5: the same count for every arm in the set)"
                )
            bases: dict[str, RunSpec] = {}
            for row in self.rows():
                spec = RunSpec.from_json(row["spec"])
                if manifest.is_primary_comparison(spec) and not row["surplus"] and spec.arm_id not in bases:
                    bases[spec.arm_id] = spec
            for arm_id, base in bases.items():
                have = self.db.execute(
                    "SELECT COUNT(*) FROM runs WHERE arm_id = ? AND surplus = 1 AND replaces IS NULL", (arm_id,)
                ).fetchone()[0]
                taken: list[int] = []
                for i in range(have, extra):
                    seed = self.next_unused_seed(arm_id, taken)
                    taken.append(seed)
                    spec = manifest.surplus_spec(base, seed)  # a ramp arm's carries Q-surplus-arm-set
                    if seed != R.SURPLUS_SEED_START + i:
                        spec = self._with_seed_rule(spec)
                    if self._insert_sql(spec, surplus=True):
                        added.append(spec.run_id)
                    if spec.group == "study_b":
                        for cont in self._cut(manifest.study_b_fewshot([spec])):  # cut 3 shortens them too
                            self._insert_sql(cont, surplus=True)
            if stored is None:
                self.db.execute("INSERT INTO settings (key, value) VALUES ('surplus_extra', ?)", (str(extra),))
                self._event_sql(None, "surplus", f"{extra} seeds per primary-comparison arm (Part 5.5)")
        return added

    def plan_continuations(self, ledger_path: Path,
                           conditions: Sequence[str] = manifest.BATTERY_CONDITIONS
                           ) -> tuple[list[RunSpec], list[tuple[str, str]]]:
        """The battery continuations ``add_continuations`` would queue, and every row or condition it
        skips with the reason (``schedule add-continuations`` prints them).

        Candidates are the ledger's completed Study A rows. Part 4.1 rule 6: "An arm that cannot be
        matched is reported with its cost and left out of the battery", so only a row with
        ``matched is True`` is continued (pilot/manifest.py); the others are skipped with the reason (None:
        the arm waits for ``enrich match``). A row whose run is not queued on this data root, whose
        run directory records another commit than the row (a ledger of another data root), or whose
        ``gap_<condition>`` the ledger already holds (its continuation ran and was read) is skipped
        too. Each spec is ``manifest.battery_continuation(parent, row, condition)``, cut like every
        other run of this data root. Refuses the pilot ledger (Part 3.6: "The pilot's runs are not
        reused").
        """
        from results.ledger_schema import load_ledger_as_rows

        conditions = tuple(conditions)
        if not conditions or len(set(conditions)) != len(conditions) or any(
                c not in manifest.BATTERY_CONDITIONS for c in conditions):
            raise SchedulerError(f"conditions must be distinct values of {manifest.BATTERY_CONDITIONS}, "
                                 f"got {conditions}")
        ledger_path = Path(ledger_path)
        if os.path.realpath(ledger_path) == os.path.realpath(self.paths.pilot):
            raise SchedulerError(f"{ledger_path} is the pilot ledger; the pilot's runs are not continued (Part 3.6)")
        if not ledger_path.exists():
            raise SchedulerError(f"{ledger_path} does not exist")
        specs: list[RunSpec] = []
        skipped: list[tuple[str, str]] = []
        for row in load_ledger_as_rows(ledger_path):
            if row.study != "A" or not row.completed:
                continue  # not a battery parent: Study B, or an exclusion
            if row.matched is not True:
                why = ("its arm is not matched yet (`enrich match`)" if row.matched is None
                       else "its arm is unmatched: left out of the battery (Part 4.1 rule 6)")
                skipped.append((row.run_id, why))
                continue
            if not self.exists(row.run_id):
                skipped.append((row.run_id, "the run is not queued on this data root"))
                continue
            parent = self.spec(row.run_id)
            train = self._read_json(self.cfg.run_dir(row.run_id) / TRAIN_RESULT) or {}
            if train.get("commit_hash") != row.commit_hash:
                skipped.append((row.run_id, f"its run directory records commit {train.get('commit_hash')!r}, the "
                                            f"ledger {row.commit_hash!r}: use the data root the ledger was written "
                                            "from"))
                continue
            for condition in conditions:
                if getattr(row, f"gap_{condition}") is not None:
                    skipped.append((row.run_id, f"gap_{condition} is in the ledger already: its continuation ran"))
                    continue
                try:
                    specs.append(manifest.battery_continuation(parent, row, condition))
                except ValueError as exc:
                    skipped.append((row.run_id, f"{condition}: {exc}"))
        return self._cut(specs), skipped

    def add_continuations(self, ledger_path: Path,
                          conditions: Sequence[str] = manifest.BATTERY_CONDITIONS) -> list[str]:
        """Queue the battery's fine-tuning and transfer continuations (``schedule add-continuations``).

        Table 2.2 "Reward-only fine-tuning" and "Transfer" continue a matched run for 1,000,000 steps
        (pilot/manifest.py). Gated by Q-arm-complete (``pilot.errors.require_answered``, first): which rows
        are matched depends on when an arm counts as complete. The specs are those of
        ``plan_continuations``; one already queued is not queued again (an identical spec is a no-op, a
        different one is refused), and a run already in a ledger is never queued again. Returns the
        run_ids queued now. The continuations carry their run gates (Q-continuations; Q-transfer-obs for
        transfer), which hold them only while a key is open, like every run's; they end ``continued``.
        """
        errors.require_answered("Q-arm-complete", what="queueing the battery's fine-tuning and transfer continuations "
                                                       "(Table 2.2; which arms are matched)")
        specs, skipped = self.plan_continuations(ledger_path, conditions)
        added: list[str] = []
        with self._write_txn():
            for spec in specs:
                if recorded_anywhere(spec, self.paths):
                    self._event_sql(spec.run_id, "already_in_ledger", "never queued twice")
                    continue
                if self._insert_sql(spec):
                    added.append(spec.run_id)
            self._event_sql(None, "continuations",
                            f"{len(added)} queued from {ledger_path} ({', '.join(conditions)}); {len(skipped)} skipped")
        return added

    # -- launching -------------------------------------------------------------

    def _command(self, stage: str, spec_path: Path, run_dir: Path) -> list[str]:
        if self.cfg.command_builder is not None:
            return list(self.cfg.command_builder(stage, spec_path, run_dir))
        return default_command(stage, spec_path, run_dir, allow_dirty=self.cfg.allow_dirty,
                               allow_pending=self.cfg.allow_pending)

    def _spawn(self, run_id: str, stage: str, attempt: int) -> subprocess.Popen:
        run_dir = self.cfg.run_dir(run_id)
        log = self.cfg.state_dir / "logs" / f"{run_id}.{stage}.attempt{attempt}.log"
        env = dict(os.environ)
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[var] = "1"  # Table 3.1: one thread per run
        env["PYTHONPATH"] = str(provenance.REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        with open(log, "ab") as fh:
            proc = subprocess.Popen(
                self._command(stage, run_dir / SPEC_FILE, run_dir),
                cwd=provenance.REPO_ROOT,
                env=env,
                stdout=fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,  # a scheduler restart does not kill running runs
            )
        self._procs[run_id] = proc
        return proc

    def _dependencies_state(self, spec: RunSpec) -> str:
        """'ready', 'waiting', 'blocked' (a dependency was excluded, blocked or superseded) or 'missing' (not
        queued yet)."""
        state = "ready"
        for dep in spec.depends_on:
            try:
                status = self.row(dep)["status"]
            except KeyError:
                state = "missing"
                continue
            if status in ("excluded", "blocked", "superseded"):
                return "blocked"
            if status not in DONE_TRAINING and state == "ready":
                state = "waiting"
        return state

    def _refresh_git(self) -> None:
        """HEAD and cleanliness, read once per scheduling pass; the gates' facts follow HEAD."""
        self._head = provenance.commit_hash(self.cfg.repo_root)
        self._clean = not provenance.dirty_paths(self.cfg.repo_root)
        if self._code_head is None:
            self._code_head = self._head
        self._main_unfinished = None  # the main sweep's progress changes between passes
        if self._facts_head != self._head:
            self._facts_head, self._facts = self._head, {}

    def _allocation_at_head(self) -> bool:
        return (self._head is not None and bool(self._clean)
                and provenance.file_committed_at(self._head, ALLOCATION_PATH, self.cfg.repo_root) is not None)

    # -- launch gates ---------------------------------------------

    def _head_fact(self, name: str, compute: Callable[[str], Any]) -> Any:
        """A fact of the content committed at HEAD, computed once per HEAD."""
        if self._head is None:
            self._refresh_git()
        assert self._head is not None
        if self._facts_head != self._head:
            self._facts_head, self._facts = self._head, {}
        if name not in self._facts:
            self._facts[name] = compute(self._head)
        return self._facts[name]

    def _main_sweep_unfinished(self) -> tuple[list[str], list[str]]:
        """(not queued, not finished training): the main-sweep runs that hold Study B, once per pass.

        Q-studyb-order, answered in Table 9.1: a non-pilot Study B training run starts only after every
        run of Study A's main sweep has finished training. Every run of the main sweep is every run of
        ``manifest.design("main")`` under this data root's cuts (``_cut``; Part 6.1), so a main-sweep run
        not queued yet holds Study B as well (queue order alone does not order the runs), and so does
        every queued group == "main" run (replacement and surplus seeds included) in UNFINISHED_TRAINING,
        except an auxiliary N = 0 run (``is_auxiliary``; Table 9.1, Q-warm-start: it serves a controller
        run and is not a seed of its arm). An excluded main-sweep run counts as finished (Table 9.1,
        Q-studyb-order); its replacement, once queued, holds Study B while it trains. One left without a
        replacement (an operational pause, Q-exclusion-breaker, listed by ``schedule status``) does not
        hold it: no command records a decision not to replace it, so such a hold could never be released.
        """
        if self._main_unfinished is None:
            queued = {r["run_id"] for r in self.rows()}
            not_queued = [s.run_id for s in self._cut(manifest.design("main")) if s.run_id not in queued]
            unfinished = [r["run_id"] for r in self.rows(UNFINISHED_TRAINING)
                          if RunSpec.from_json(r["spec"]).group == "main" and not self.is_auxiliary(r["run_id"])]
            self._main_unfinished = (not_queued, unfinished)
        return self._main_unfinished

    def launch_gates(self, spec: RunSpec) -> list[tuple[str, str]]:
        """The launch gates (LAUNCH_GATES) that hold ``spec`` now, as (gate, reason), in LAUNCH_GATES order.

        Registered mode only: a smoke scheduler, and ``schedule status`` of a smoke data root, find
        none. Each is read from the content committed at HEAD, never from the working tree (module
        docstring, "registered preconditions"). Serves the scheduler's launch decision and ``schedule
        status``.
        """
        if self._smoke:
            return []
        repo = self.cfg.repo_root
        held: list[tuple[str, str]] = []
        if spec.study == "B":
            released, reason = self._head_fact("search", lambda head: search_record_status(head, repo))
            if not released:
                held.append(("search_record_missing", reason))
        if spec.group == "study_b" and not spec.pilot:
            not_queued, unfinished = self._main_sweep_unfinished()
            if not_queued or unfinished:
                parts = [f"{len(not_queued)} main-sweep runs are not queued on this data root yet "
                         f"(e.g. {not_queued[0]}; `schedule add --design main`)" if not_queued else "",
                         f"{len(unfinished)} main-sweep runs have not finished training (e.g. {unfinished[0]})"
                         if unfinished else ""]
                held.append(("main_sweep_unfinished",
                             f"{'; '.join(p for p in parts if p)}; Part 3.3: \"Study B's runs are scheduled "
                             "after Study A's main sweep\" (Table 9.1, Q-studyb-order)"))
        if spec.plugin in DETERMINISM_PLUGINS:
            reports = self._head_fact("determinism", lambda head: committed_determinism_reports(head, repo))
            open_keys = determinism_open_keys(spec.plugin)
            if open_keys:  # the rule of the check's own configuration is open: no report releases it
                held.append(("determinism_missing",
                             f"the determinism check of plug-in {spec.plugin!r} runs the rule "
                             f"({DETERMINISM_LATE_ONSET}) of {', '.join(open_keys)}, which is open "
                             "(configs/registered.py PENDING); once it is answered, run `python "
                             f"scripts/determinism_check.py --registered-form --plugin {spec.plugin}` from a clean "
                             "commit and commit its report"))
            elif spec.plugin not in reports:
                held.append(("determinism_missing",
                             "HEAD holds no passing registered-form determinism report of plug-in "
                             f"{spec.plugin!r} under {DETERMINISM_DIR}/ (Table 3.1); run `python "
                             f"scripts/determinism_check.py --registered-form --plugin {spec.plugin}` from a clean "
                             "commit and commit its report"))
        if not spec.pilot:
            present = self._head_fact(
                "analysis", lambda head: provenance.file_committed_at(head, ANALYSIS_ENTRY, repo) is not None)
            if not present:
                held.append(("analysis_missing",
                             f"HEAD holds no {ANALYSIS_ENTRY}; Part 5.8: the analysis script is committed before the "
                             "first main-study run completes (Role 4)"))
        return held

    def _launch_blocker(self, row: sqlite3.Row) -> str | None:
        """Why a pending run cannot start now (None: it can). Never depends on any result."""
        spec = RunSpec.from_json(row["spec"])
        if spec.plugin in self._unavailable_plugins:
            return "plugin_unavailable"
        if row["run_id"] in self._refused:
            return "launch_refused"
        if row["run_id"] in self._launch_failed:
            return "launch_failed"
        # The keys stored in the spec and the run gates the current code derives (manifest.open_run_gates).
        if manifest.open_run_gates(spec, registered=not self._smoke) and not self._allow_pending:
            return "open_questions"
        deps = self._dependencies_state(spec)
        if deps != "ready":
            return f"dependency_{deps}"
        if not self._allow_dirty and not self._clean:
            return "dirty_worktree"
        if spec.pilot and self.cfg.require_allocation_for_pilot and not self._smoke and not self._allocation_at_head():
            return "allocation_missing"
        if (int(row["attempt"]) > 1 and row["launch_commit"] and row["launch_commit"] != self._head
                and not self._allow_dirty and self._restart_code_changes(row["launch_commit"])):
            return "restart_commit_mismatch"
        gates = self.launch_gates(spec)
        return gates[0][0] if gates else None

    def _restart_code_changes(self, launch_commit: str) -> list[str]:
        """Files other than results that differ between a restarted run's commit and HEAD (empty: the restart may
        run at HEAD). A commit of results only (a ledger, a determinism report: ``provenance.is_output_path``)
        leaves the code and configuration the interrupted attempt ran on, so it never holds the restart; any
        other change does, and so does a commit git cannot compare (named as such)."""
        def compute(head: str) -> list[str]:
            try:
                return provenance.non_output_changes_between(launch_commit, head, self.cfg.repo_root)
            except subprocess.CalledProcessError:
                return [f"(git cannot compare {launch_commit} with {head})"]

        return self._head_fact(f"restart_code_changes:{launch_commit}", compute)

    def _ledger_record(self, spec: RunSpec) -> dict | None:
        """The run's recorded ledger row (or sidecar record) as a dict, or None."""
        sidecar = self.paths.sidecar_dir / f"{spec.run_id}.json"
        if sidecar.exists():
            return self._read_json(sidecar)
        from results.ledger_schema import load_ledger_as_rows

        ledger = self.paths.for_spec(spec)
        for recorded in load_ledger_as_rows(ledger) if ledger.exists() else []:
            if recorded.run_id == spec.run_id:
                return {"completed": recorded.completed, "failure_cause": recorded.failure_cause}
        return None

    def _settle_recorded(self, run_id: str, spec: RunSpec) -> None:
        """A pending run already in a ledger (e.g. the state was restored from an older backup)."""
        record = self._ledger_record(spec) or {}
        if record.get("completed") is False:
            with self.db:
                self._set_sql(run_id, status="excluded", pid=None, failure_cause=record.get("failure_cause"),
                              ledger_written=1)
                self._event_sql(run_id, "exclusion_in_ledger",
                                f"cause={record.get('failure_cause')}; never launched twice")
                self._event_sql(run_id, "needs_decision", f"{RECORDED_DECISION}check whether its replacement ran; "
                                                          f"if not, `schedule resolve {run_id} replace`")
            return
        self._set(run_id, status="ledgered")
        self._event(run_id, "already_in_ledger", "never launched twice")

    def _changed_since(self, row: sqlite3.Row) -> bool:
        """Inside a write transaction: True (and a ``launch_skipped`` event) if the run's row is no longer the
        pending row ``row`` this pass checked (status, spec, attempt or launch commit changed by an operator
        command in between); the scheduler then writes nothing for it and reads it again on the next pass."""
        current = self.db.execute("SELECT status, spec, attempt, launch_commit FROM runs WHERE run_id = ?",
                                  (row["run_id"],)).fetchone()
        if current is not None and all(current[k] == row[k] for k in ("status", "spec", "attempt", "launch_commit")):
            return False
        now = "removed" if current is None else f"now {current['status']}"
        self._event_sql(row["run_id"], "launch_skipped", f"the row changed since this pass read it ({now}); "
                        "read again on the next pass")
        return True

    def _start_training(self, row: sqlite3.Row) -> bool:
        run_id = row["run_id"]
        spec = RunSpec.from_json(row["spec"])
        if recorded_anywhere(spec, self.paths):
            self._settle_recorded(run_id, spec)
            return False
        blocker = self._launch_blocker(row)
        if blocker == "dependency_blocked":
            with self._write_txn():  # an operator command may have settled the run since it was read
                if self._changed_since(row):
                    return False
                self._set_sql(run_id, status="blocked")
                self._event_sql(run_id, "blocked",
                                f"{blocker} for {', '.join(spec.depends_on)}; see `schedule status` for the action")
            return False
        if blocker is not None:
            detail = self._blocker_detail(blocker, row, spec)
            if detail is not None:
                self._event_once(run_id, blocker, detail)
            return False
        run_dir = self.cfg.run_dir(run_id)
        attempt = int(row["attempt"])
        # Committed before the process exists: a crash here can never launch it twice. The row is read again
        # under the write lock (operator commands serialise with the scheduler): a run requeued, superseded or
        # otherwise changed since this pass read it, while its blockers were checked, is left for the next pass,
        # and spec.json is written from the spec checked, which is the one stored.
        with self._write_txn():
            if self._changed_since(row):
                return False
            if run_dir.exists() and any(p.name != SPEC_FILE for p in run_dir.iterdir()):
                raise SchedulerError(f"{run_dir} already holds output; refusing to launch {run_id} again")
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / SPEC_FILE).write_text(spec.to_json(), encoding="utf-8")
            # Attempt 1 records the commit it starts on now (an earlier start that never trained, e.g. a
            # plug-in not yet committed, does not count); a restart keeps the commit restart() recorded, unless
            # that commit differs from HEAD in result files only (``_restart_code_changes``): it then runs at HEAD.
            # A smoke root (allow_dirty) runs at HEAD whatever the commit recorded, so it records HEAD.
            moved = (attempt > 1 and row["launch_commit"] and row["launch_commit"] != self._head
                     and not self._allow_dirty)
            commit = (self._head if attempt == 1 or moved or self._allow_dirty
                      else (row["launch_commit"] or self._head))
            self._set_sql(run_id, status="training", pid=None, host=self._host, stage_started=self._now(),
                          launch_commit=commit)
            self._event_sql(run_id, "train_starting", f"attempt={attempt} commit={self._head}" + (
                f"; the interrupted attempt ran on {row['launch_commit']}, which differs from HEAD in result files "
                "only (provenance.is_output_path): the same code and configuration" if moved else ""))
        proc = self._spawn(run_id, "train", attempt)
        self._set(run_id, pid=proc.pid)
        self._event(run_id, "train_started", f"pid={proc.pid}")
        return True

    def _blocker_detail(self, blocker: str, row: sqlite3.Row, spec: RunSpec) -> str | None:
        """The event detail of a launch blocker (None: the blocker is logged by no event); built only for it."""
        if blocker == "allocation_missing":
            return f"commit {ALLOCATION_PATH} before any pilot run (Part 6 G2)"
        if blocker == "restart_commit_mismatch":
            return (f"the interrupted attempt ran on {row['launch_commit']}; HEAD is {self._head}, which changes "
                    + ", ".join(self._restart_code_changes(row["launch_commit"])[:10]))
        if blocker == "dirty_worktree":
            return "commit the changes to tracked code before launching"
        if blocker == "open_questions":
            return ", ".join(manifest.open_run_gates(spec, registered=not self._smoke))
        if blocker == "dependency_missing":
            return f"waits for {', '.join(self._missing_dependencies(spec))}, not queued yet"
        if blocker in LAUNCH_GATES:
            return dict(self.launch_gates(spec)).get(blocker)
        return None

    def _evaluation_blocker(self, run_id: str) -> str | None:
        """Why a trained run cannot be evaluated now (None: it can); the launcher would refuse it.

        A missing harness holds only the runs it evaluates (``contracts.evaluator_target``).
        """
        if contracts.evaluator_target(self.spec(run_id)) in self._unavailable_evaluators:
            return "evaluator_unavailable"
        if run_id in self._refused:
            return "launch_refused"
        if not self._allow_dirty and not self._clean:
            return "dirty_worktree"
        return None

    def _start_evaluation(self, run_id: str) -> bool:
        blocker = self._evaluation_blocker(run_id)
        if blocker is not None:
            if blocker == "dirty_worktree":
                self._event_once(run_id, blocker, "commit the changes to tracked code before evaluating")
            return False
        with self.db:
            self._set_sql(run_id, status="evaluating", pid=None, host=self._host, stage_started=self._now())
        proc = self._spawn(run_id, "evaluate", int(self.row(run_id)["attempt"]))
        self._set(run_id, pid=proc.pid)
        self._event(run_id, "eval_started", f"pid={proc.pid}")
        return True

    # -- process state ---------------------------------------------------------

    def _names_run(self, argv: list[bytes], run_id: str, cwd: str | None) -> bool:
        """True if a command line carries this run's directory and spec file as whole arguments.

        Arguments are compared as real paths (a relative one against the process's working
        directory ``cwd``; skipped when that is unknown), so a scheduler started with another
        spelling of the same data root (relative, absolute, through a symlink) still finds the run.
        """
        needle = run_id.encode()
        candidates = [a for a in argv if needle in a]  # cheap filter before any filesystem call
        if not candidates:
            return False
        run_dir = self.cfg.run_dir(run_id)
        wanted = {os.path.realpath(run_dir), os.path.realpath(run_dir / SPEC_FILE)}
        found = set()
        for arg in candidates:
            path = os.fsdecode(arg)
            if not os.path.isabs(path):
                if cwd is None:
                    continue
                path = os.path.join(cwd, path)
            found.add(os.path.realpath(path))
        return wanted <= found

    @staticmethod
    def _proc_argv(pid: int | str, run_id: str) -> tuple[list[bytes], str | None]:
        """Command line and working directory of a process (Linux /proc); OSError if unreadable.

        The working directory is read only when the command line mentions ``run_id`` at all.
        """
        argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        if not any(run_id.encode() in a for a in argv):
            return argv, None
        try:
            cwd: str | None = os.readlink(f"/proc/{pid}/cwd")
        except OSError:  # another user's process: only absolute arguments can be compared
            cwd = None
        return argv, cwd

    def _process_runs(self, pid: int, run_id: str) -> bool:
        """True if process ``pid`` exists and is this run's launcher (guards against PID reuse)."""
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            pass  # it exists but belongs to another user: its command line decides
        if not Path("/proc").is_dir():  # pragma: no cover - no /proc: trust the PID (Linux only: HANDOVER.md section 6)
            return True
        try:
            argv, cwd = self._proc_argv(pid, run_id)
        except OSError:
            return False
        return self._names_run(argv, run_id, cwd)

    def _find_process(self, run_id: str) -> int | None:
        """PID of a live process whose command line names this run's directory (Linux /proc)."""
        proc_root = Path("/proc")
        if not proc_root.is_dir():
            return None
        for entry in proc_root.iterdir():
            if not entry.name.isdigit() or int(entry.name) == os.getpid():
                continue
            try:
                argv, cwd = self._proc_argv(entry.name, run_id)
            except OSError:
                continue
            if self._names_run(argv, run_id, cwd):
                return int(entry.name)
        return None

    def _running(self, row: sqlite3.Row, *, announce: bool = False) -> bool:
        """Is the run's process alive? Adopts a live process found by its run directory.

        The recorded PID is trusted only on the recording host and only if its command line still
        names the run; otherwise (PID not yet recorded, another host name, PID reused) the process
        is looked up by its run directory, so a live run is never taken for dead and restarted.
        """
        run_id = row["run_id"]
        proc = self._procs.get(run_id)
        if proc is not None:
            return proc.poll() is None
        pid = row["pid"]
        if pid is not None and row["host"] == self._host and self._process_runs(int(pid), run_id):
            if announce:
                self._event(run_id, "adopted", f"pid={pid}")
            return True
        found = self._find_process(run_id)
        if found is None:
            return False
        self._set(run_id, pid=found, host=self._host)
        note = "" if row["host"] in (None, self._host) else f"; recorded on host {row['host']}"
        self._event(run_id, "adopted", f"pid={found} (found by its run directory{note})")
        return True

    def _settle(self, row: sqlite3.Row) -> None:
        if row["status"] == "training":
            self._finish_training(row["run_id"])
        else:
            self._finish_evaluation(row["run_id"])

    def _exit_code(self, run_id: str) -> int | None:
        proc = self._procs.pop(run_id, None)
        return None if proc is None else proc.returncode

    def _read_json(self, path: Path) -> dict | None:
        """The JSON object in ``path``; None if it is missing, unreadable, malformed or not an object."""
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def _finish_training(self, run_id: str) -> None:
        code = self._exit_code(run_id)
        spec = self.spec(run_id)
        run_dir = self.cfg.run_dir(run_id)
        if code in (EXIT_UNAVAILABLE, EXIT_REFUSED):
            # The launcher refuses before it claims the directory, or removes its own claim and output:
            # nothing is cleaned up here (a refusal of a directory claimed by a live run must not touch it).
            if code == EXIT_UNAVAILABLE:
                self._unavailable_plugins.add(spec.plugin)
            else:
                self._refused.add(run_id)
            self._set(run_id, status="pending", pid=None)
            if code == EXIT_UNAVAILABLE:
                self._event(run_id, "plugin_unavailable",
                            f"plug-in {spec.plugin!r} is not in the repository yet; the run stays pending")
            else:
                self._event(run_id, "launch_refused", "the launcher refused to start the run; see the train log")
            return
        if not run_dir.exists() and self._archive(run_id, int(self.row(run_id)["attempt"])).exists():
            # A restart under the policy was cut short after its archive move: finish that restart
            # (not 'never started', which would relaunch the same attempt on HEAD).
            self._event(run_id, "restart_resumed", "the attempt was archived before the scheduler stopped")
            self._restart_or_hold(run_id, "training")
            return
        if not (run_dir / CLAIM_FILE).exists() and not (run_dir / TRAIN_RESULT).exists():
            # The launcher never claimed the directory: nothing trained, so nothing is excluded or held.
            self._set(run_id, status="pending", pid=None)
            if code is None:
                self._event(run_id, "never_started", "no claim in the run directory; back to pending")
            else:
                self._launch_failed.add(run_id)
                self._event(run_id, "launch_failed",
                            f"exit code {code} before the claim; see the train log. Retried at the next scheduler "
                            "start")
            return
        result = self._read_json(run_dir / TRAIN_RESULT)
        status = result.get("status") if result else None
        if status == "completed":
            if spec.group in CONTINUATION_GROUPS:
                # A continuation is not a ledger row: enrichment reads it into its parent's row
                # (pilot/enrichment.py; pilot/manifest.py).
                command = "enrich fewshot" if spec.group == "study_b_fewshot" else "enrich continuations"
                self._set(run_id, status="continued", pid=None, train_hours=result.get("wall_clock_hours"))
                self._event(run_id, "continued",
                            "its results are read into "
                            f"{spec.depends_on[0] if spec.depends_on else 'the parent'}'s row by `{command}`")
            else:
                self._set(run_id, status="trained", pid=None, train_hours=result.get("wall_clock_hours"))
                self._event(run_id, "trained", f"{result.get('wall_clock_hours') or 0.0:.3f} h")
        elif status == "failed":
            self._exclude(run_id, result["failure_cause"], result.get("detail", ""),
                          by_policy=bool(result.get("excluded_by_policy")))
        elif status == "interrupted":
            self._interrupted(run_id, f"the launcher recorded an interruption: {result.get('detail', '')}",
                              ORIGIN_MACHINE)
        elif code in CRASH_SIGNALS:
            self._exclude_without_result(run_id, "crash", f"process died with signal {-code}")
        elif code is None:
            self._interrupted(run_id, "ended while no scheduler was its parent; exit status unknown "
                                      "(a death by signal cannot be told apart from an interruption)", ORIGIN_MACHINE)
        elif code in STOP_SIGNALS:
            self._interrupted(run_id, f"stopped by signal {-code} before the launcher recorded it", ORIGIN_MACHINE)
        else:  # SIGKILL (e.g. the OOM killer), another non-crash signal, or a non-zero exit without train_result
            self._interrupted(run_id, f"no final training result (exit code {code})", ORIGIN_PROCESS)

    def _policy(self) -> str:
        """The interruption policy in force: the stored one, else this command's, else ``_default_policy``."""
        return self.setting("on_interrupt") or self.cfg.on_interrupt or self._default_policy()

    def _default_policy(self) -> str:
        """The policy of a data root whose first run names none: 'restart' on a registered data root once
        Q-interrupted-run is answered (Table 9.1: registered data roots run under `schedule set-policy restart`),
        'hold' while it is open and on a smoke data root."""
        return "restart" if not self._smoke and not R.is_open(POLICY_KEY) else "hold"

    def _interrupted(self, run_id: str, detail: str, origin: str) -> None:
        """Settle an attempt that ended without a final result, an interruption of ``origin`` (ORIGIN_MACHINE or
        ORIGIN_PROCESS, recorded at the start of its 'interrupted' event), under the policy in force."""
        policy = self._policy()
        self._event(run_id, "interrupted", f"{ORIGIN_PREFIX}{origin}: {detail}; policy={policy}")
        if policy == "hold":
            self._set(run_id, status="interrupted", pid=None)
        elif policy == "restart":
            self._restart_by_rule(run_id, "training")
        else:
            self._exclude_without_result(run_id, "incomplete", detail, by_policy=True)

    def _interruption_origins(self, run_id: str) -> list[str]:
        """The origin of each of the run's interruptions, oldest first, as its 'interrupted' events record it."""
        found = self.db.execute("SELECT detail FROM events WHERE run_id = ? AND event = 'interrupted' ORDER BY id",
                                (run_id,))
        return [detail[len(ORIGIN_PREFIX):].split(":", 1)[0] for (detail,) in found
                if (detail or "").startswith(ORIGIN_PREFIX)]

    def _restart_by_rule(self, run_id: str, expected: str) -> None:
        """The 'restart' policy (Table 9.1, Q-interrupted-run) for the run's last interruption, already logged.

        A run whose last interruption is of process origin and that ended so more than MAX_RESTARTS times (the
        operator's restarts of a held run included) "fails to complete its steps" (Part 5.6): it is excluded with
        cause 'incomplete', not by the policy (``by_policy=False``: it counts towards the circuit breaker), and
        replaced by the arm's next unused seed. Every other interrupted run is restarted (``_restart_or_hold``).
        ``expected``: 'training' for the attempt the scheduler watched end, 'interrupted' for a held run.
        """
        ends = self._ends_past_restart_cap(run_id)
        if ends:
            self._exclude_without_result(
                run_id, "incomplete",
                f"ended without a final result while the scheduler watched it (SIGKILL, another non-crash signal or a "
                f"non-zero exit) {ends} times (more than MAX_RESTARTS = {MAX_RESTARTS}): it fails to complete its "
                "steps (Part 5.6; Table 9.1, Q-interrupted-run)", expected=expected)
        else:
            self._restart_or_hold(run_id, expected)

    def _ends_past_restart_cap(self, run_id: str) -> int:
        """The run's ends of process origin if its last interruption is one and they number more than MAX_RESTARTS
        (Table 9.1, Q-interrupted-run: it "fails to complete its steps", Part 5.6), else 0."""
        origins = self._interruption_origins(run_id)
        ends = origins.count(ORIGIN_PROCESS)
        return ends if origins and origins[-1] == ORIGIN_PROCESS and ends > MAX_RESTARTS else 0

    def _restart_refused(self, run_id: str) -> int:
        """``_ends_past_restart_cap(run_id)`` where the cap refuses the operator's restart of a held run (`schedule
        resolve RUN restart`): on a smoke data root, and on a registered one once Q-interrupted-run is answered (where
        the 'restart' policy applies the cap); else 0. The cap binds the operator's restarts of a run held under
        'hold' as it binds the policy's."""
        return self._ends_past_restart_cap(run_id) if self._smoke or not R.is_open(POLICY_KEY) else 0

    def _restart_or_hold(self, run_id: str, expected: str) -> None:
        """Restart under the policy; a restart that cannot happen now holds the run (never stops the scheduler).

        A run another command acted on first (its status is no longer ``expected``) is left as that
        command left it: holding it could mark a live attempt as held.
        """
        try:
            self.restart(run_id, expected=(expected,))
        except StatusChanged as exc:
            self._event(run_id, "restart_skipped", str(exc))
        except (SchedulerError, OSError) as exc:
            self._set(run_id, status="interrupted", pid=None)
            self._event(run_id, "restart_failed", f"{exc}; held (`schedule resolve {run_id} restart` once fixed)")

    def _archive(self, run_id: str, attempt: int) -> Path:
        return self.cfg.state_dir / "interrupted" / f"{run_id}-attempt{attempt}"

    def restart(self, run_id: str, *, expected: tuple[str, ...] = ("interrupted",)) -> None:
        """Archive the interrupted run's partial output and queue it to train again from scratch.

        The run is set back to pending as its next attempt (same seed; the commit its interrupted attempt
        ran on), and a later pass of the scheduler trains it.

        ``expected``: the statuses the caller checked ('training' only for the scheduler settling
        the attempt it watched end), re-checked inside the write transaction.
        """
        run_dir = self.cfg.run_dir(run_id)
        # The status is checked and changed inside one write transaction, so two operator commands
        # can never restart the same attempt twice; the archive move happens while it is held.
        with self._write_txn():
            row = self.row(run_id)
            if row["status"] not in expected:
                raise StatusChanged(f"{run_id} is {row['status']}; only an interrupted run can be restarted"
                                    if expected == ("interrupted",) else
                                    f"{run_id} is {row['status']}, not {' or '.join(expected)}; another command "
                                    "acted on it first")
            live = self._find_process(run_id)
            if live is not None:
                raise SchedulerError(f"{run_id}: process {live} still uses {run_dir}; a live run is never restarted")
            attempt = int(row["attempt"])
            archive = self._archive(run_id, attempt)
            # The commit the interrupted attempt actually ran on, as its launcher recorded it; without a
            # result (a hard kill), the commit the scheduler recorded when it started that attempt.
            previous = self._read_json(run_dir / TRAIN_RESULT) or self._read_json(archive / TRAIN_RESULT) or {}
            commit = previous.get("commit_hash") or row["launch_commit"]
            if run_dir.exists():
                if archive.exists():
                    raise SchedulerError(f"{archive} already exists; move it aside before restarting {run_id}")
                archive.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(run_dir), str(archive))
            # else: a restart cut short after its move (before this commit) already archived it
            self._set_sql(run_id, status="pending", pid=None, attempt=attempt + 1, launch_commit=commit)
            self._event_sql(run_id, "restart", f"attempt {attempt + 1}, same seed and commit ({commit}), from scratch")

    def _exclude_without_result(self, run_id: str, cause: str, detail: str, *, by_policy: bool = False,
                                expected: str = "training") -> None:
        """Exclude a run whose train_result.json is not a final one: record what is known.

        The launcher's own record (for example of an interruption) is kept; the scheduler adds the
        exclusion's cause. ``expected`` is the status the caller saw ('training' for the attempt the
        scheduler watched end, 'interrupted' for a held run under the 'exclude' policy); it is
        re-checked inside the write transaction, and a run another command acted on first (for
        example restarted and launched again) is left alone: its directory is never written.
        """
        with self._write_txn():
            row = self.row(run_id)
            if row["status"] != expected:
                self._event_sql(run_id, "exclusion_skipped",
                                f"{run_id} is {row['status']}, not {expected}; another command acted on it first")
                return
            self._write_exclusion_record(run_id, row, cause, detail, by_policy)
            self._exclude_sql(run_id, cause, detail, by_policy=by_policy)
        self._record_exclusion(run_id)

    def _write_exclusion_record(self, run_id: str, row: sqlite3.Row, cause: str, detail: str, by_policy: bool) -> None:
        """Rewrite train_result.json as the exclusion's final record (inside the exclusion's transaction).

        A held run whose restart was cut short after its archive move (no run directory, its attempt's
        archive present) gets that output moved back first, so the record keeps the launcher's data and
        the ledger writer finds the run's output where it always is.
        """
        run_dir = self.cfg.run_dir(run_id)
        archive = self._archive(run_id, int(row["attempt"]))
        if not run_dir.exists() and archive.exists():
            shutil.move(str(archive), str(run_dir))
            self._event_sql(run_id, "archive_restored", f"a restart cut short after its archive move; {archive} moved "
                                                        "back before the exclusion")
        run_dir.mkdir(parents=True, exist_ok=True)
        previous = self._read_json(run_dir / TRAIN_RESULT) or {}
        omnisafe_dir = previous.get("omnisafe_dir")
        if omnisafe_dir is None and (run_dir / OMNISAFE_SUBDIR).exists():
            found = sorted(p for p in (run_dir / OMNISAFE_SUBDIR).glob("*/seed-*") if p.is_dir())
            omnisafe_dir = str(found[0].relative_to(run_dir)) if len(found) == 1 else None
        record = {
            **previous,
            "run_id": run_id,
            "status": "failed",
            "failure_cause": cause,
            "detail": detail,
            "commit_hash": previous.get("commit_hash") or row["launch_commit"] or provenance.commit_hash(),
            "machine": previous.get("machine") or provenance.machine_description(),
            "started": previous.get("started") or row["stage_started"] or self._now(),
            "finished": previous.get("finished") or self._now(),
            "wall_clock_hours": float(previous.get("wall_clock_hours") or 0.0),
            "omnisafe_dir": omnisafe_dir,
            "written_by": (previous["written_by"] if "excluded_by_policy" in previous  # its own record, re-applied
                           else f"scheduler, from the launcher's result with status {previous.get('status')!r}"
                           if previous else "scheduler (the process left no final result)"),
            # Kept in the record, so an exclusion re-read after a crash keeps its origin (circuit breaker).
            "excluded_by_policy": by_policy,
        }
        if self.cfg.smoke:
            record["allow_dirty"] = record.get("allow_dirty") or self.cfg.allow_dirty
            record["allow_pending"] = record.get("allow_pending") or self.cfg.allow_pending
        _write_json_atomic(run_dir / TRAIN_RESULT, record)

    def seed_target(self, spec: RunSpec) -> int:
        """The seed target of ``spec``'s arm on this data root: 5 plus this data root's surplus seeds for an arm of Part
        5.5's primary-comparison set, 5 otherwise (Q-arm-complete; as ``pilot.enrichment.seed_targets``), and for
        every pilot arm the pilot's registered seed count, len(R.PILOT_SEEDS) = 3 (Part 3.6: "seeds 0, 1, 2"), the
        one-seed Study B pilot arms included. Every target thus exceeds MAX_SAME_CAUSE_EXCLUSIONS, so an arm's stop
        (Q-exclusion-breaker: as many same-cause exclusions as its seed target) never comes before the circuit
        breaker's pause, which that resolution presumes (pause, diagnose, then replace or stop)."""
        if spec.pilot:
            return len(R.PILOT_SEEDS)
        extra = int(self.setting("surplus_extra") or 0)
        return len(R.SEEDS) + (extra if manifest.is_primary_comparison(spec) else 0)

    def _replace_rule(self, run_id: str, spec: RunSpec, cause: str) -> str:
        """What an excluded run left without a replacement asks of the operator (Table 9.1, Q-exclusion-breaker): an
        operational pause, never a decision on results. Below the arm's seed target: diagnose, fix a defect with
        an erratum, replace. At it: the arm did not complete its seeds; only a fixed defect resumes it."""
        count, target = self._same_cause_exclusions(spec.arm_id, cause), self.seed_target(spec)
        if count >= target:
            return (f"arm {spec.arm_id} has {count} exclusions with cause '{cause}', as many as its seed target "
                    f"({target}): it did not complete its seeds ({cause}, {count}) and enters no comparison "
                    f"(Part 4.1); a replacement follows only a fixed defect: `schedule resolve {run_id} replace "
                    f"--defect-fixed ERRATUM_ID`, its entry in {ERRATA_PATH} committed at HEAD")
        return ("diagnose from the failed runs' logs and tracebacks, never their costs or returns; fix a defect in the "
                f"code or environment by a commit recorded as an erratum ({ERRATA_PATH}, Table 9.1; for a memory kill, "
                f"fewer concurrent runs); then `schedule resolve {run_id} replace`")

    def _same_cause_exclusions(self, arm_id: str, cause: str) -> int:
        """Exclusions of the arm with this cause, not counting those applied by the 'exclude' interruption policy nor
        those of an auxiliary run (AUXILIARY_EVENT; Table 9.1, Q-warm-start: not a seed of its arm,
        manifest.AUXILIARY_NOTE), so neither the circuit breaker nor the seed-target stop counts them."""
        return self.db.execute(
            "SELECT COUNT(*) FROM runs WHERE arm_id = ? AND status = 'excluded' AND failure_cause = ? "
            "AND run_id NOT IN (SELECT run_id FROM events WHERE event = 'excluded_by_policy' AND run_id IS NOT NULL) "
            "AND run_id NOT IN (SELECT run_id FROM events WHERE event = ? AND run_id IS NOT NULL)",
            (arm_id, cause, AUXILIARY_EVENT),
        ).fetchone()[0]

    def _policy_replacement_chain(self, run_id: str) -> int:
        """Replacements stopped by the interruption policy in a row, ending at ``run_id`` (counted as one).

        Follows ``replaces`` back while the replaced run is itself a replacement excluded by the
        policy, so one outage that stops unrelated replacements of an arm counts once for each.
        """
        count, previous = 1, self.row(run_id)["replaces"]
        while previous is not None:
            row = self.row(previous)
            by_policy = self.db.execute(
                "SELECT 1 FROM events WHERE run_id = ? AND event = 'excluded_by_policy'", (previous,)
            ).fetchone()
            if row["replaces"] is None or by_policy is None:
                break
            count, previous = count + 1, row["replaces"]
        return count

    def _replacement_sql(self, run_id: str, spec: RunSpec, *, surplus: bool) -> RunSpec:
        """Queue the arm's next unused seed in place of ``run_id`` (inside a write transaction).

        The replacement repeats the run as it was queued (Part 5.6: "repeated with the next unused
        seed"), continuations included: the replaced run's own continuations never run (superseded
        here), and a Study B replacement's few-shot continuations (Part 3.3) are queued in their place
        exactly when the replaced run had them queued on this data root. So a run whose continuations
        were never queued (a smoke copy of a Study B run: scripts/smoke_run.py queues none) gets a
        replacement without them, never registered-length continuations it was not given;
        continuations queued later by ``add`` follow the replacement there. A battery continuation
        exists only for a matched, ledgered parent, which is never replaced; were one queued for a
        replaced parent, it would be superseded here too (the replacement is continued once it is
        matched, by ``add_continuations``).
        """
        replacement = self._with_seed_rule(spec.with_seed(self.next_unused_seed(spec.arm_id)))
        self._insert_sql(replacement, replaces=run_id, surplus=surplus)
        self._set_sql(run_id, replaced_by=replacement.run_id)
        self._event_sql(run_id, "replaced", replacement.run_id)
        # the commit whose code computed the replacement (and its continuations), as `record_queued` logs for the
        # queueing commands
        self._event_sql(replacement.run_id, "queued", f"replacement of {run_id} (code at commit {self._head}"
                        + (", uncommitted changes allowed: smoke)" if self._allow_dirty else ")"))
        missing = self._missing_dependencies(replacement)
        if missing:
            self._event_sql(replacement.run_id, "dependency_missing", f"waits for {', '.join(missing)}, not queued yet")
        continued = False
        for dependent in self.rows(("pending", "blocked")):
            dspec = RunSpec.from_json(dependent["spec"])
            if dspec.group in CONTINUATION_GROUPS and run_id in dspec.depends_on:
                continued = continued or dspec.group == "study_b_fewshot"
                self._set_sql(dependent["run_id"], status="superseded")
                self._event_sql(dependent["run_id"], "superseded",
                                f"parent {run_id} excluded and replaced by {replacement.run_id}; "
                                + ("its continuations run instead" if dspec.group == "study_b_fewshot" else
                                   "it is continued once matched (`schedule add-continuations`)"))
        if continued:
            for cont in self._cut(manifest.study_b_fewshot([replacement])):  # Part 3.3: each run is continued
                self._insert_sql(cont, surplus=surplus)
        return replacement

    def _missing_dependencies(self, spec: RunSpec) -> list[str]:
        """The runs ``spec`` depends on that are not queued on this data root."""
        return [d for d in spec.depends_on if not self.exists(d)]

    def _add_auxiliary(self, run_id: str) -> list[str]:
        """Queue the N = 0 runs a pending warm-started run depends on but this data root never queued.

        A warm-started replacement (``spec.with_seed``) depends on the N = 0 run of its own seed, which
        exists only if the N = 0 arm received that seed (a surplus seed on SafetyPointGoal1-v0);
        otherwise the replacement would wait forever, so it gets an auxiliary N = 0 run of that seed
        (Table 9.1, Q-warm-start; ``manifest.AUXILIARY_NOTE``). The run queued is the N = 0 arm's
        queued spec with the replacement's seed and the seed rule of Q-seed-collision (it trains
        exactly as that arm's run of that seed), recorded by an AUXILIARY_EVENT so its ledger row
        carries ``manifest.AUXILIARY_NOTE`` and is left out of its arm. Refused while Q-warm-start is
        open.
        """
        errors.require_answered(AUXILIARY_KEY, what=f"queueing an auxiliary N = 0 run ({AUXILIARY_KEY})")
        added: list[str] = []
        with self._write_txn():
            row = self.row(run_id)
            spec = RunSpec.from_json(row["spec"])
            if row["status"] != "pending" or spec.controller_variant != "warm_started":
                raise SchedulerError(f"{run_id} is {row['status']} ({spec.controller_variant or spec.group} run); "
                                     "only a pending warm-started run waits for an auxiliary N = 0 run")
            missing = self._missing_dependencies(spec)
            if not missing:
                raise SchedulerError(f"{run_id}: every dependency is queued; nothing to add")
            for dep in missing:
                arm = dep.rsplit("-s", 1)[0]
                source = self.db.execute("SELECT spec FROM runs WHERE arm_id = ? ORDER BY seed LIMIT 1",
                                         (arm,)).fetchone()
                if source is None:
                    raise SchedulerError(f"{run_id}: the arm {arm} of its dependency {dep} is not queued on this "
                                         "data root")
                aux = self._with_seed_rule(RunSpec.from_json(source["spec"]).with_seed(spec.seed))
                if aux.run_id != dep or aux.N != 0.0:
                    raise SchedulerError(f"{run_id}: its dependency {dep} is not an N = 0 run of the arm {arm}")
                self._insert_sql(aux)
                self._event_sql(aux.run_id, AUXILIARY_EVENT, f"for {run_id}: {manifest.AUXILIARY_NOTE}")
                added.append(aux.run_id)
        return added

    def current_spec(self, run_id: str) -> RunSpec:
        """The spec the code checked out now builds for the queued run ``run_id`` (``requeue``).

        Rebuilt as the run was queued: the design's arm (``manifest.design``, or the pilot's) with the
        run's seed (``with_seed``, as replacements and surplus seeds are made), ``manifest.surplus_spec``
        for a surplus seed, the seed rule of Q-seed-collision when the stored spec carries it (a
        replacement, a collision seed or an auxiliary run), and this data root's cuts. A few-shot
        continuation is rebuilt from its parent's current spec. A battery continuation is built from its
        parent's ledger row (``add_continuations``) and is not rebuilt here: SchedulerError.
        """
        row = self.row(run_id)
        stored = RunSpec.from_json(row["spec"])
        if stored.group == "study_b_fewshot":
            parent = self.current_spec(stored.depends_on[0])
            found = [c for c in self._cut(manifest.study_b_fewshot([parent])) if c.run_id == run_id]
        elif stored.group in CONTINUATION_GROUPS:
            raise SchedulerError(f"{run_id} is a battery continuation, built from its parent's ledger row "
                                 "(`schedule add-continuations`); it is not rebuilt here")
        else:
            designed = (*manifest.pilot(), *manifest.pilot(revision=1), *manifest.design("all", seeds=R.SEEDS[:1]))
            arms = [s for s in designed if s.arm_id == stored.arm_id and s.group not in CONTINUATION_GROUPS]
            if not arms:
                raise SchedulerError(f"{run_id}: the current design holds no arm {stored.arm_id}")
            spec = arms[0] if arms[0].seed == stored.seed else arms[0].with_seed(stored.seed)
            if row["surplus"] and manifest.is_primary_comparison(spec):
                spec = manifest.surplus_spec(spec, stored.seed)
            if "Q-seed-collision" in stored.pending:
                spec = self._with_seed_rule(spec)
            found = [c for c in self._cut([spec]) if c.run_id == run_id]
        if not found:
            raise SchedulerError(f"{run_id}: the current design, under this data root's cuts, holds no such run")
        return found[0]

    def _requeue(self, run_id: str) -> RunSpec:
        """``resolve RUN requeue``: store ``current_spec(run_id)`` for a pending or blocked run that never trained.

        Only attempt 1, never replaced, with no output in its run directory (its spec file aside) and no
        live process: nothing was trained under the stored spec. Logged with both specs' differences.
        """
        new = self.current_spec(run_id)
        with self._write_txn():  # checked inside the write lock: never swapped under a launch
            row = self.row(run_id)
            if row["status"] not in ("pending", "blocked") or int(row["attempt"]) != 1 or row["replaced_by"]:
                raise SchedulerError(f"{run_id} is {row['status']} (attempt {row['attempt']}, replaced_by="
                                     f"{row['replaced_by']}); only a pending or blocked run that never trained is "
                                     "requeued")
            run_dir = self.cfg.run_dir(run_id)
            if run_dir.exists() and any(p.name != SPEC_FILE for p in run_dir.iterdir()):
                raise SchedulerError(f"{run_dir} holds output; a run that started is never changed")
            if self._find_process(run_id) is not None:
                raise SchedulerError(f"{run_id} has a live process; it is never changed")
            old = RunSpec.from_json(row["spec"])
            if old == new:
                raise SchedulerError(f"{run_id}: its stored spec is already the current design's; nothing to requeue")
            before, after = old.to_dict(), new.to_dict()
            changed = {k: [before.get(k), after.get(k)] for k in sorted(set(before) | set(after))
                       if before.get(k) != after.get(k)}
            self._set_sql(run_id, spec=new.to_json(), arm_id=new.arm_id, seed=new.seed)
            self._event_sql(run_id, REQUEUE_EVENT, json.dumps(changed, default=str))
        return new

    def is_auxiliary(self, run_id: str) -> bool:
        """True for a run queued by ``_add_auxiliary`` (its ledger row carries ``manifest.AUXILIARY_NOTE``)."""
        return self.db.execute("SELECT 1 FROM events WHERE run_id = ? AND event = ? LIMIT 1",
                               (run_id, AUXILIARY_EVENT)).fetchone() is not None

    def _ledger_notes(self, run_id: str) -> tuple[str, ...]:
        """The notes the scheduler's record adds to a run's ledger row: an auxiliary run (``manifest.AUXILIARY_NOTE``),
        a surplus seed (``manifest.SURPLUS_NOTE``; Part 5.5, a surplus seed's replacement included) or the
        replacement of a registered seed (``manifest.replacement_note``; Part 5.6). Both of the last take the arm's
        next unused seed, so only this record tells them apart (Q-surplus-in-analysis: analysis.data.five_seed_view)."""
        if self.is_auxiliary(run_id):
            return (manifest.AUXILIARY_NOTE,)
        row = self.row(run_id)
        if row["surplus"]:
            return (manifest.SURPLUS_NOTE,)
        if row["replaces"]:
            return (manifest.replacement_note(row["replaces"]),)
        return ()

    def _replacement_of(self, run_id: str) -> str | None:
        """The run that replaces ``run_id`` in the end (following ``replaced_by``), or None if it was not replaced."""
        latest = None
        found = self.db.execute("SELECT replaced_by FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        while found is not None and found["replaced_by"]:
            latest = found["replaced_by"]
            found = self.db.execute("SELECT replaced_by FROM runs WHERE run_id = ?", (latest,)).fetchone()
        return latest

    def _add_sql(self, spec: RunSpec) -> int:
        """Queue one spec of ``add`` (inside a write transaction); the number of runs queued.

        A few-shot continuation of a parent already excluded and replaced (its continuations were
        queued after the exclusion, e.g. ``schedule add --design study_b_fewshot`` after a crash) never
        runs: it is recorded as superseded, and the continuation with the same budget of the run that
        replaces the parent in the end is queued in its place, as ``_replacement_sql`` does for
        continuations queued before the exclusion.
        """
        replacement = self._replacement_of(spec.depends_on[0]) if (
            spec.group == "study_b_fewshot" and spec.depends_on) else None
        if replacement is None:
            return int(self._insert_sql(spec))
        if self._insert_sql(spec, status="superseded"):
            self._event_sql(spec.run_id, "superseded", f"parent {spec.depends_on[0]} was excluded and replaced by "
                                                       f"{replacement} before this continuation was queued; that "
                                                       "run's continuation runs instead")
        surplus = bool(self.row(replacement)["surplus"])
        return sum(int(self._insert_sql(cont, surplus=surplus))
                   for cont in self._cut(manifest.study_b_fewshot([self.spec(replacement)]))
                   if cont.params["budget"] == spec.params["budget"])

    def _exclude(self, run_id: str, cause: str, detail: str, *, by_policy: bool = False) -> None:
        """Part 5.6: exclusion with its cause, recorded in the ledger, and repeated with the next unused seed.

        The state change, the replacement and the dependents' states are one transaction.
        ``by_policy`` marks an exclusion applied by the 'exclude' interruption policy (smoke data
        roots only): it does not count towards the same-cause circuit breaker, but a replacement
        stopped by the policy again and again does (MAX_SAME_CAUSE_EXCLUSIONS successive replacements,
        each replacing the last, stopped by the policy).
        """
        with self._write_txn():
            self._exclude_sql(run_id, cause, detail, by_policy=by_policy)
        self._record_exclusion(run_id)

    def _queueing_code_problem(self) -> str:
        """Why this process may not compute a registered run's replacement now ('' when it may).

        The replacement's spec (``spec.with_seed``, ``_with_seed_rule``) and a Study B replacement's few-shot
        continuations (``manifest.study_b_fewshot``, ``apply_cuts``) are trained as stored, never re-derived, so
        the code that computes them must be the commit's, as `schedule add` and `resolve replace` require
        (``_loaded_code_problem``: a clean tree, no uncommitted module loaded, and, if HEAD moved since this
        process loaded its code, no loaded module changed between the two commits; a commit of a ledger, a
        report or a record leaves the loaded code the commit's). Smoke data roots (``allow_dirty``) are exempt.
        """
        if self._allow_dirty:
            return ""
        return self._loaded_code_problem()

    def _loaded_code_problem(self) -> str:
        """Why this process's loaded code is not the commit's ('' when it is), checked in this order: a clean tree,
        no uncommitted module loaded, and, if HEAD moved since this process loaded its code, no loaded module
        changed between the two commits (``_queueing_code_problem``, ``_ledger_code_problem``)."""
        if self._head is None:
            self._refresh_git()
        if not self._clean:
            return "tracked code has uncommitted changes"
        unverified = provenance.unverified_imported_code(self.cfg.repo_root)
        if unverified:
            return "code loaded by this scheduler is not committed: " + ", ".join(unverified)
        if self._code_head is not None and self._head != self._code_head:
            try:
                moved = provenance.imported_code_changed_between(self._code_head, self._head, self.cfg.repo_root)
            except subprocess.CalledProcessError:
                moved = [f"(git cannot compare {self._code_head} with {self._head})"]
            if moved:
                return (f"code this scheduler loaded at {self._code_head} changed by HEAD {self._head}: "
                        + ", ".join(moved) + " (restart the scheduler)")
        return ""

    def _exclude_sql(self, run_id: str, cause: str, detail: str, *, by_policy: bool = False) -> None:
        """The exclusion, its replacement and the dependents' new states (inside the caller's write transaction).

        The replacement is deferred to the operator (``needs_decision``: `schedule resolve RUN replace`, which
        requires committed code) while ``_queueing_code_problem`` says this process may not compute it, and when
        the circuit breaker pauses the arm (its rule: ``_replace_rule``)."""
        spec = self.spec(run_id)
        row = self.row(run_id)
        reason = breaker = ""
        if spec.group in CONTINUATION_GROUPS:
            reason = ("a continuation follows its parent's seed and checkpoint and cannot take a new seed; the group "
                      "decides by an amendment (Table 9.1); the scheduler has no action for it")
        elif self.is_auxiliary(run_id):
            reason = AUXILIARY_EXCLUDED
        elif not by_policy and (same := self._same_cause_exclusions(spec.arm_id, cause) + 1) \
                >= MAX_SAME_CAUSE_EXCLUSIONS:
            breaker = (f"{same} exclusions of arm {spec.arm_id} with cause '{cause}' (circuit breaker: "
                       f"MAX_SAME_CAUSE_EXCLUSIONS = {MAX_SAME_CAUSE_EXCLUSIONS}; its seed target: "
                       f"{self.seed_target(spec)}): ")
        elif (by_policy and row["replaces"]
              and self._policy_replacement_chain(run_id) >= MAX_SAME_CAUSE_EXCLUSIONS):
            breaker = (f"{MAX_SAME_CAUSE_EXCLUSIONS} successive replacements stopped by the interruption policy "
                       "(likely systematic): ")
        if not reason and not breaker:
            problem = self._queueing_code_problem()
            if problem:
                reason = (f"{CODE_PROBLEM_DECISION}{problem}, and a queued spec is trained as stored, so the "
                          f"commit's code must compute it; commit, then `schedule resolve {run_id} replace`")
        self._set_sql(run_id, status="excluded", pid=None, failure_cause=cause, replaced_by=None)
        self._event_sql(run_id, "excluded", f"{cause}: {detail[-300:]}")
        if by_policy:
            self._event_sql(run_id, "excluded_by_policy", "the interruption policy, applied to every run alike")
        if breaker:  # after the exclusion is set: the rule counts it
            reason = breaker + self._replace_rule(run_id, spec, cause)
        # A replaced Study B parent's continuations are superseded inside _replacement_sql.
        replacement = None if reason else self._replacement_sql(run_id, spec, surplus=bool(row["surplus"]))
        for dependent in self.rows(("pending",)):
            if run_id in RunSpec.from_json(dependent["spec"]).depends_on:
                self._set_sql(dependent["run_id"], status="blocked")
                self._event_sql(dependent["run_id"], "blocked",
                                f"dependency {run_id} excluded; see `schedule status` for the action")
        if replacement is None:
            self._event_sql(run_id, "needs_decision", reason)

    @staticmethod
    def _load_ledger_code() -> str:
        """Import the modules the ledger writes import lazily (``LEDGER_WRITE_MODULES``), so that the check of the
        loaded code (``provenance.unverified_imported_code``) sees them before the first write, never after it;
        '' or why one cannot be imported (the writes wait, as for uncommitted code)."""
        for name in LEDGER_WRITE_MODULES:
            try:
                importlib.import_module(name)
            except ImportError as exc:
                return f"the ledger writer's module {name} cannot be imported ({exc})"
        return ""

    def _ledger_code_problem(self, run_id: str) -> str:
        """Why this process may not write a registered ledger row or supplement record now ('' when it may).

        Ledger rows and their records are write-once, computed by this process's code
        (``pilot.ledger_writer``, ``pilot.contracts``) and attributed to a commit, so that code must be the
        commit's: the ledger writer's modules imported (``_load_ledger_code``), then the checks of a
        replacement (``_loaded_code_problem``): a clean tree, no uncommitted module loaded, and, if HEAD
        moved since this process loaded its code, no loaded module changed between the two commits (a
        commit of a report or a record leaves the loaded code the commit's; a commit of code needs a
        restart). Otherwise the run waits, with nothing written (``ledger_waiting``, logged once per
        process). Smoke data roots (``allow_dirty``) are exempt.
        """
        if self._allow_dirty:
            return ""
        problem = self._load_ledger_code() or self._loaded_code_problem()
        if problem:
            self._event_once(run_id, "ledger_waiting", f"the ledger is not written while {problem}")
        return problem

    def _record_exclusion(self, run_id: str, quiet: bool = False) -> str:
        """Record an excluded run in the ledger; returns why it is not recorded ('' when it is)."""
        spec = self.spec(run_id)
        if spec.group in CONTINUATION_GROUPS:
            self._set(run_id, ledger_written=1)  # continuations are not ledger rows; the event log keeps the record
            return ""
        problem = self._ledger_code_problem(run_id)
        if problem:
            return problem  # ledger_written stays 0: retried at every pass, and on `schedule resolve RUN retry-ledger`
        try:
            if recorded_anywhere(spec, self.paths):
                event = "ledger_exclusion_present"
            else:
                notes = self._ledger_notes(run_id)  # as _write_ledger
                write_run(spec, self.cfg.run_dir(run_id), self.paths, attempt=int(self.row(run_id)["attempt"]),
                          extra_notes=notes)
                event = "ledger_exclusion_written"
            self._set(run_id, ledger_written=1)
            self._event(run_id, event, "")
            return ""
        except Exception as exc:  # noqa: BLE001 - retried at every pass; logged once per process when quiet
            if quiet:
                self._event_once(run_id, "ledger_write_failed", repr(exc))
            else:
                self._event(run_id, "ledger_write_failed", repr(exc))
            return repr(exc)

    def _finish_evaluation(self, run_id: str) -> None:
        code = self._exit_code(run_id)
        evaluation = self._read_json(self.cfg.run_dir(run_id) / EVALUATION_RESULT)
        if code == EXIT_UNAVAILABLE:
            target = contracts.evaluator_target(self.spec(run_id))
            owner = contracts.EVALUATOR_OWNERS.get(target.split(".")[0], "another role")
            # holds only the runs this harness evaluates (pilot.contracts.evaluator_target)
            self._unavailable_evaluators.add(target)
            self._set(run_id, status="trained", pid=None)
            self._event(run_id, "evaluator_unavailable",
                        f"the evaluation harness {target} ({owner}) is not in the repository yet; the run stays "
                        "trained")
        elif code == EXIT_REFUSED:
            self._set(run_id, status="trained", pid=None)
            self._refused.add(run_id)
            self._event(run_id, "launch_refused", "the launcher refused to evaluate the run; see the evaluate log")
        elif evaluation is not None and code in (0, None):
            # a failed evaluate stage (any other code) is eval_failed even if an evaluation.json is there: never
            # an earlier attempt's file taken for this one's (``resolve reevaluate`` moves that aside as well)
            self._set(run_id, status="evaluated", pid=None, eval_hours=evaluation.get("eval_wall_clock_hours"))
            self._event(run_id, "evaluated", "")
        elif code is None:
            # The evaluation ended while no scheduler watched it (for example a reboot). Evaluation
            # is deterministic and writes nothing until it succeeds, so it is simply run again.
            self._set(run_id, status="trained", pid=None)
            self._event(run_id, "eval_rerun", "evaluation ended unobserved; it will run again")
        else:
            self._set(run_id, status="eval_failed", pid=None)
            self._event(run_id, "eval_failed",
                        f"exit code {code}; see the evaluate log, then `schedule resolve {run_id} reevaluate`")

    def _write_ledger(self, run_id: str) -> None:
        """Append an evaluated run's row with its ``evaluation`` supplement record (``write_run``).

        A row already present (a scheduler stopped between the append and its own bookkeeping, or a
        row written before the supplement existed) gets its record if it is missing
        (``ensure_evaluation_supplement``, results/supplement_schema.py): the run must be the row's run, and its
        evaluation the row's. Any error of either (LedgerWriteError, ContractError, an unreadable
        file) ends the run ``ledger_failed`` until the operator retries.
        """
        spec = self.spec(run_id)
        if self._ledger_code_problem(run_id):
            return  # the run stays evaluated, with nothing written, until the code is committed
        try:
            if recorded_anywhere(spec, self.paths):
                record = ensure_evaluation_supplement(spec, self.cfg.run_dir(run_id), self.paths)
                self._event(run_id, "ledgered",
                            "row already present" + (f"; evaluation record {record}" if record else ""))
            else:
                notes = self._ledger_notes(run_id)
                written = write_run(spec, self.cfg.run_dir(run_id), self.paths,
                                    attempt=int(self.row(run_id)["attempt"]), extra_notes=notes)
                self._event(run_id, "ledgered", str(written))
        except Exception as exc:  # noqa: BLE001 - terminal until the operator retries
            self._set(run_id, status="ledger_failed")
            self._event(run_id, "ledger_failed",
                        f"{exc!r}; fix the cause, then `schedule resolve {run_id} retry-ledger` (`reevaluate` when "
                        "the cause is its evaluation.json)")
            return
        self._set(run_id, status="ledgered", ledger_written=1)

    # -- main loop -------------------------------------------------------------

    def recover(self) -> None:
        """On start: adopt live runs, settle runs that ended unobserved, retry unrecorded exclusions.

        Under the 'exclude' policy a run still held (a `set-policy exclude` cut short) is excluded
        now, as the policy applies to every run alike. Under 'hold' and 'restart' a held run whose
        restart (`set-policy restart` or `resolve RUN restart`) was cut short after its archive move
        (no run directory, its attempt's archive present) is restarted now, completing that restart.
        """
        # as in step(): the exclusions' ledger writes and replacements are gated on the tree as it is now, never on
        # the cleanliness cached by an earlier pass of this object
        self._refresh_git()
        for row in self.rows(ACTIVE):
            if not self._running(row, announce=True):
                self._settle(row)
        policy = self.setting("on_interrupt")
        for row in self.rows("interrupted"):
            run_id = row["run_id"]
            if policy == "exclude":
                self._exclude_without_result(run_id, "incomplete", POLICY_EXCLUSION_DETAIL, by_policy=True,
                                             expected="interrupted")
            elif (not self.cfg.run_dir(run_id).exists()
                  and self._archive(run_id, int(row["attempt"])).exists()):
                self._event(run_id, "restart_resumed", "the attempt was archived before the restart was recorded")
                self._restart_or_hold(run_id, "interrupted")
        for row in self.rows("excluded"):
            if not row["ledger_written"]:
                self._record_exclusion(row["run_id"])

    def step(self) -> None:
        """One pass: settle finished processes, write ledger rows, retry unrecorded exclusions, fill free slots."""
        self._refresh_git()  # before the ledger writes, which are gated on it (_ledger_code_problem)
        for row in self.rows(ACTIVE):
            if not self._running(row):
                self._settle(row)
        for row in self.rows("evaluated"):
            self._write_ledger(row["run_id"])
        for row in self.rows("excluded"):  # an exclusion whose ledger record failed earlier
            if not row["ledger_written"]:
                self._record_exclusion(row["run_id"], quiet=True)
        free = self.cfg.max_concurrent - len(self.rows(ACTIVE))
        for row in self.rows("trained"):  # evaluation first: finish runs before starting new ones
            if free <= 0:
                break
            if self._start_evaluation(row["run_id"]):
                free -= 1
        for row in self.rows("pending"):
            if free <= 0:
                break
            if self._start_training(row):
                free -= 1

    def idle(self) -> bool:
        """True if no run is training or evaluating."""
        return not self.rows(ACTIVE)

    def _require_policy_answered(self, policy: str) -> None:
        """The policies a registered data root may use (smoke data roots are exempt). 'restart' applies the group's
        answer to Q-interrupted-run (Part 5.6: whether an interrupted run counts as a crash or is restarted), so it
        is refused (PendingQuestionError, exit 3) while the key is open and the policy can only record a ruling of
        the amendment log. 'exclude' is refused likewise while the key is open, and always once it is answered
        (SchedulerError): Table 9.1 restarts an interrupted run, never excludes it for the interruption. 'hold'
        decides nothing (before the answer, and afterwards as an operational pause)."""
        if policy == "hold" or self._smoke:
            return
        errors.require_answered(POLICY_KEY, what=f"the interruption policy {policy!r} ({POLICY_KEY}: hold until "
                                                 "the group answers)")
        if policy == "exclude":
            raise SchedulerError(f"the interruption policy 'exclude' contradicts the answer to {POLICY_KEY} (Table "
                                 "9.1): an interrupted run is restarted, never excluded for the interruption; a "
                                 "registered data root runs under `schedule set-policy restart` ('hold' pauses it)")

    def _check_policy(self) -> str | None:
        """The stored policy (None before the first run), refused (PendingQuestionError or SchedulerError) when this
        command may not run under it; writes nothing, so ``run`` checks it before fixing the mode and the ledgers."""
        stored = self.setting("on_interrupt")
        self._require_policy_answered(stored if stored is not None else self.cfg.on_interrupt or self._default_policy())
        if stored is not None and self.cfg.on_interrupt is not None and stored != self.cfg.on_interrupt:
            raise SchedulerError(
                f"this data root uses on_interrupt={stored!r}; it applies to every run alike. "
                "Change it only with `schedule set-policy`, as Table 9.1 (Q-interrupted-run) rules."
            )
        return stored

    def _fix_policy(self) -> None:
        if self._check_policy() is None:
            policy = self.cfg.on_interrupt or self._default_policy()
            with self.db:
                self.db.execute("INSERT INTO settings (key, value) VALUES ('on_interrupt', ?)", (policy,))
                self._event_sql(None, "policy", f"on_interrupt={policy}")

    def set_policy(self, policy: str) -> None:
        """Change the interruption policy for every run and apply it to held runs.

        Under 'restart' each held run is settled by the rule of Q-interrupted-run (``_restart_by_rule``):
        a run whose last interruption is of process origin and that ended so more than MAX_RESTARTS
        times is excluded ('incomplete') and replaced; every other held run is restarted.
        """
        if policy not in INTERRUPT_POLICIES:
            raise ValueError(f"policy must be one of {INTERRUPT_POLICIES}")
        self._check_mode()  # a smoke data root refuses a registered decision as such
        self._require_policy_answered(policy)  # before anything is fixed: a refusal (exit 3) writes nothing
        self._fix_mode()
        # as resolve() and recover(): HEAD for the gates and for the `queued` event of a replacement, and the tree as
        # it is now (never the cleanliness cached by an earlier pass of this object)
        self._refresh_git()
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO settings (key, value) VALUES ('on_interrupt', ?)", (policy,))
            self._event_sql(None, "policy", f"on_interrupt={policy} (changed)")
        self.cfg.on_interrupt = policy
        for row in self.rows("interrupted"):
            run_id = row["run_id"]
            if policy == "restart":
                self._restart_by_rule(run_id, "interrupted")
            elif policy == "exclude":
                self._exclude_without_result(run_id, "incomplete", POLICY_EXCLUSION_DETAIL, by_policy=True,
                                             expected="interrupted")

    def run(self, *, once: bool = False, max_passes: int | None = None) -> None:
        """Hold the data root's lock, check and fix the mode and policy, recover, then step until idle.

        ``once`` stops after one pass and ``max_passes`` after that many; otherwise the scheduler polls every
        ``poll_seconds`` until nothing is running and nothing is left to launch.
        """
        with self._exclusive():
            # every refusal before anything is fixed, as set_policy: a refused run (exit 3, or a policy that differs
            # from the stored one) leaves no mode, ledger location or ledger mode markers behind
            self._check_mode()
            self._check_policy()
            self._fix_mode()
            self._fix_policy()
            with self.db:  # the flags `schedule status` judges this data root's runs under (_status_flags)
                self.db.execute("INSERT OR REPLACE INTO settings (key, value) VALUES ('run_flags', ?)", (json.dumps(
                    {"allow_dirty": self.cfg.allow_dirty, "allow_pending": self.cfg.allow_pending}),))
            self._event(None, "scheduler_start", f"max_concurrent={self.cfg.max_concurrent} smoke={self.cfg.smoke}")
            self.recover()
            passes = 0
            while True:
                self.step()
                passes += 1
                if once or (max_passes is not None and passes >= max_passes):
                    return
                if self.idle() and not self._launchable_left():
                    self._event(None, "idle", "nothing left to launch")
                    return
                time.sleep(self.cfg.poll_seconds)

    def _launchable_left(self) -> bool:
        if self.rows("evaluated"):
            return True
        self._refresh_git()
        if any(self._evaluation_blocker(r["run_id"]) is None for r in self.rows("trained")):
            return True
        return any(self._launch_blocker(row) is None for row in self.rows("pending"))

    @contextlib.contextmanager
    def _exclusive(self) -> Iterator[None]:
        lock_path = self.cfg.state_dir / "scheduler.lock"
        with open(lock_path, "w", encoding="utf-8") as fh:
            try:
                import fcntl
            except ImportError:  # pragma: no cover - no fcntl: POSIX only (Linux only: HANDOVER.md section 6)
                yield
                return
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise SchedulerError(f"another scheduler is running on {self.cfg.data_root}") from exc
            try:
                yield
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

    # -- reporting and operator actions ----------------------------------------

    def summary(self) -> dict[str, int]:
        """Runs per status, plus ``excluded_not_in_ledger`` when an exclusion is not yet in the ledger."""
        counts: dict[str, int] = {}
        for row in self.rows():
            counts[row["status"]] = counts.get(row["status"], 0) + 1
        unwritten = sum(1 for r in self.rows("excluded") if not r["ledger_written"])
        if unwritten:
            counts["excluded_not_in_ledger"] = unwritten
        return counts

    def events(self, run_id: str | None = None, limit: int = 50) -> list[sqlite3.Row]:
        """The latest ``limit`` events, newest first: of one run, or of the whole data root (None)."""
        if run_id is None:
            return list(self.db.execute("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)))
        return list(self.db.execute("SELECT * FROM events WHERE run_id = ? ORDER BY id DESC LIMIT ?", (run_id, limit)))

    def _last_events(self) -> dict[str, tuple[str, str]]:
        """Each run's most recent event: (event, detail)."""
        found = self.db.execute(
            "SELECT e.run_id, e.event, e.detail FROM events e JOIN (SELECT run_id, MAX(id) AS id FROM events "
            "WHERE run_id IS NOT NULL GROUP BY run_id) m ON e.id = m.id"
        )
        return {r["run_id"]: (r["event"], r["detail"] or "") for r in found}

    def _last_decision(self, run_id: str) -> str:
        """The detail of the run's most recent needs_decision event ('' if it has none)."""
        found = self.db.execute("SELECT detail FROM events WHERE run_id = ? AND event = 'needs_decision' "
                                "ORDER BY id DESC LIMIT 1", (run_id,)).fetchone()
        return (found["detail"] or "") if found is not None else ""

    def _known_unavailable(self) -> tuple[set[str], set[str]]:
        """(plug-ins, evaluation-harness targets) that the latest scheduler process found missing
        (pilot.contracts.evaluator_target).

        The running scheduler holds every run of such a plug-in or harness without trying it (sets that
        last as long as its process: ``_unavailable_plugins``, ``_unavailable_evaluators``). ``schedule
        status`` runs in another process, so it reads them back from the event log: the plug-in, or the
        ``contracts.evaluator_target``, of every run with a plugin_unavailable or evaluator_unavailable
        event since the last scheduler_start (a new scheduler process tries them again), together with
        this process's own sets.
        """
        plugins, targets = set(self._unavailable_plugins), set(self._unavailable_evaluators)
        start = self.db.execute("SELECT MAX(id) FROM events WHERE event = 'scheduler_start'").fetchone()[0] or 0
        found = self.db.execute(
            "SELECT e.event, r.spec FROM events e JOIN runs r ON r.run_id = e.run_id WHERE e.id > ? "
            "AND e.event IN ('plugin_unavailable', 'evaluator_unavailable')", (start,))
        for event, spec_json in found:
            spec = RunSpec.from_json(spec_json)
            if event == "plugin_unavailable":
                plugins.add(spec.plugin)
            else:
                targets.add(contracts.evaluator_target(spec))
        return plugins, targets

    def _status_flags(self, stored_mode: str | None) -> tuple[tuple[bool, bool] | None, str | None]:
        """The flags ``schedule status`` judges the waiting runs under (None: the command's own), and a note.

        The data root's runs are judged as the scheduler that runs them judges them, not by the
        status command's own flags (the command line's ``schedule status`` takes no smoke flags):
        the launch gates apply in registered mode only, so a smoke data root shows none, and its open
        questions hold nothing when its runs are run with ``--allow-pending``. A registered data root
        is judged as registered; a smoke one with the flags its last ``schedule run`` recorded
        (``run_flags``), or with both smoke flags before any run (scripts/smoke_run.py passes both).
        """
        if stored_mode is None or stored_mode == self.cfg.mode:
            return None, None
        if stored_mode == "registered":
            return (False, False), "judged as registered, the mode of its runs (not by this command's smoke flags)"
        try:
            recorded = json.loads(self.setting("run_flags") or "null")
            flags = (bool(recorded["allow_dirty"]), bool(recorded["allow_pending"]))
        except (ValueError, TypeError, KeyError):
            return (True, True), "judged in smoke mode with --allow-dirty --allow-pending (no scheduler has run it yet)"
        shown = " ".join(f for f, on in zip(("--allow-dirty", "--allow-pending"), flags) if on)
        return flags, f"judged in smoke mode with the flags of its last `schedule run` ({shown})"

    def status_report(self) -> dict[str, Any]:
        """What ``python -m pilot schedule status`` shows (pilot.contracts.evaluator_target; pilot/manifest.py).

        * ``head``: HEAD now;
        * ``mode``: the data root's stored mode (``config_mode``: this command's); ``judged``: how its
          runs are judged when the two differ (``_status_flags``; None: by this command's flags), and
          ``judged_mode`` the mode they are judged in;
        * ``counts``: runs per status (``summary``), plus ``excluded_not_in_ledger``: excluded runs
          whose exclusion is not yet in the ledger;
        * ``gates``: every launch gate (LAUNCH_GATES) holding a pending run, with its reason and the
          runs it holds (registered mode; each read from HEAD now);
        * ``evaluators_unavailable``: {evaluation harness target: trained runs waiting for it}: every
          trained run of a target the latest scheduler process found missing (``_known_unavailable``),
          as that process holds them all, not only the run that met the missing harness;
        * ``plugins_unavailable``: {plug-in: pending runs waiting for it}, likewise for every pending
          run of the plug-in;
        * ``waiting``: {reason: runs} for every pending and trained run ("ready" when nothing holds
          it; a trained run's reason is prefixed "evaluation: "). Reasons only a running scheduler
          knows (a plug-in or harness found missing, a refusal) are read back from the event log
          (``_known_unavailable``, then the run's last event);
        * ``decisions``: [(run_id, what the group or the operator must decide)]: excluded runs left
          without a replacement (an excluded continuation: an amendment, Table 9.1; an excluded
          auxiliary run: ``AUXILIARY_EXCLUDED``; a replacement deferred only for uncommitted or moved
          code: the operator commits, then replaces; an exclusion found in the ledger: check whether
          its replacement ran; any other: the rule of the circuit breaker, ``_replace_rule``), held
          interrupted runs (one past Q-interrupted-run's cap: `set-policy restart` only), blocked runs,
          failed evaluations and ledger writes, exclusions not yet in the ledger, and pending runs
          whose dependency is not queued (queue it, or `schedule resolve RUN add-auxiliary` for a
          warm-started replacement).
        Never depends on any result.
        """
        stored_mode = self.setting("mode")
        flags, judged = self._status_flags(stored_mode)
        with self._judged_as(flags):
            report = self._status_report()
            judged_mode = "smoke" if self._smoke else "registered"
        return {"mode": stored_mode, "config_mode": self.cfg.mode, "judged": judged, "judged_mode": judged_mode,
                **report}

    def _status_report(self) -> dict[str, Any]:
        """``status_report`` apart from the modes, judged under the current flags (``_judged_as``)."""
        self._refresh_git()
        last = self._last_events()
        missing_plugins, missing_targets = self._known_unavailable()
        gates: dict[tuple[str, str], list[str]] = {}
        waiting: dict[str, list[str]] = {}
        plugins: dict[str, list[str]] = {}
        evaluators: dict[str, list[str]] = {}
        for row in self.rows("pending"):
            run_id, spec = row["run_id"], RunSpec.from_json(row["spec"])
            for gate in self.launch_gates(spec):
                gates.setdefault(gate, []).append(run_id)
            # _launch_blocker's first reason, as the scheduler that found the plug-in missing sees it
            reason = "plugin_unavailable" if spec.plugin in missing_plugins else self._launch_blocker(row)
            remembered = last.get(run_id, ("", ""))[0]
            if reason is None and remembered in REMEMBERED_BLOCKERS:
                reason = remembered
            if reason == "plugin_unavailable":
                plugins.setdefault(spec.plugin, []).append(run_id)
            waiting.setdefault(reason or "ready", []).append(run_id)
        for row in self.rows("trained"):
            run_id = row["run_id"]
            target = contracts.evaluator_target(self.spec(run_id))
            # _evaluation_blocker's first reason, as the scheduler that found the harness missing sees it
            reason = "evaluator_unavailable" if target in missing_targets else self._evaluation_blocker(run_id)
            remembered = last.get(run_id, ("", ""))[0]
            if reason is None and remembered in REMEMBERED_BLOCKERS:
                reason = remembered
            if reason == "evaluator_unavailable":
                evaluators.setdefault(target, []).append(run_id)
            waiting.setdefault(f"evaluation: {reason or 'ready'}", []).append(run_id)
        decisions: list[tuple[str, str]] = []
        for row in self.rows():
            run_id, status = row["run_id"], row["status"]
            spec = RunSpec.from_json(row["spec"])
            if status == "excluded" and not row["replaced_by"]:
                deferred = self._last_decision(run_id)
                decisions.append((run_id, "an excluded continuation: the group decides by an amendment (Table 9.1)"
                                  if spec.group in CONTINUATION_GROUPS else
                                  f"an excluded auxiliary run: {AUXILIARY_EXCLUDED}" if self.is_auxiliary(run_id) else
                                  deferred if deferred.startswith((CODE_PROBLEM_DECISION, RECORDED_DECISION)) else
                                  "excluded without a replacement (an operational pause, Q-exclusion-breaker): "
                                  + self._replace_rule(run_id, spec, row["failure_cause"])))
            if status == "excluded" and not row["ledger_written"]:
                decisions.append((run_id, "the exclusion is not in the ledger yet "
                                          f"(`schedule resolve {run_id} retry-ledger`)"))
            elif status == "interrupted" and (ends := self._restart_refused(run_id)):
                decisions.append((run_id, f"held after {ends} ends without a final result of process origin, more "
                                          f"than MAX_RESTARTS = {MAX_RESTARTS}: it fails to complete its steps (Part "
                                          "5.6); `schedule set-policy restart` excludes it by Q-interrupted-run's rule "
                                          "(Table 9.1, cause 'incomplete') and queues its replacement"))
            elif status == "interrupted":
                decisions.append((run_id, "held (the policy 'hold', or a restart that failed): `schedule "
                                          "set-policy restart` applies Q-interrupted-run's rule (Table 9.1), or "
                                          f"`schedule resolve {run_id} restart`"))
            elif status == "blocked":  # a continuation takes neither `replace` nor `unblock`
                decisions.append((run_id, "a dependency was excluded, blocked or superseded: " + (
                    self._continuation_hint(spec) if spec.group in CONTINUATION_GROUPS
                    # Table 9.1 (Q-warm-start): no value is taken from an excluded N = 0 run; the warm-started run
                    # never trained, so it is not excluded but superseded by its arm's next unused seed
                    else f"`schedule resolve {run_id} replace` (Q-warm-start: no value is taken from an excluded "
                         "run; this run never trained and is superseded by its arm's next unused seed)"
                    if spec.controller_variant == "warm_started"
                    else f"`schedule resolve {run_id} replace` (its arm's next unused seed runs instead), or `schedule "
                         f"resolve {run_id} unblock` once its dependencies are fine again")))
            elif status == "eval_failed":
                decisions.append((run_id, f"see the evaluate log, then `schedule resolve {run_id} reevaluate`"))
            elif status == "ledger_failed":
                decisions.append((run_id, f"fix the cause, then `schedule resolve {run_id} retry-ledger` (or "
                                          "`reevaluate` when the cause is its evaluation.json)"))
            elif status == "pending" and (missing := self._missing_dependencies(spec)):
                decisions.append((run_id, f"dependency {', '.join(missing)} not queued on this data root: " + (
                    f"queue its auxiliary N = 0 run: `schedule resolve {run_id} add-auxiliary` ({AUXILIARY_KEY})"
                    if spec.controller_variant == "warm_started" else "queue it (`schedule add`)")))
        return {
            "head": self._head,
            "counts": self.summary(),
            "gates": [{"gate": g, "reason": reason, "runs": runs}
                      for (g, reason), runs in sorted(gates.items(), key=lambda kv: LAUNCH_GATES.index(kv[0][0]))],
            "evaluators_unavailable": evaluators,
            "plugins_unavailable": plugins,
            "waiting": waiting,
            "decisions": decisions,
        }

    def status_lines(self, *, events: int = 20) -> list[str]:
        """``status_report`` and the last ``events`` events as the lines ``schedule status`` prints."""
        report = self.status_report()
        lines = [f"data root {self.cfg.data_root} ({report['mode'] or 'no mode yet'}; HEAD {report['head']})"]
        if report["judged"]:
            lines.append(f"note: this data root holds {report['mode']} runs; the reasons below are {report['judged']}")
        lines += [f"{status:24s} {count}" for status, count in sorted(report["counts"].items())]
        if report["gates"]:
            lines.append("launch gates:")
            for gate in report["gates"]:
                lines.append(f"  {gate['gate']:22s} holds {_count(len(gate['runs']), 'run')} (e.g. {gate['runs'][0]}): "
                             f"{gate['reason']}")
        elif report["judged_mode"] == "smoke":
            lines.append("launch gates: none in smoke mode")
        for target, runs in sorted(report["evaluators_unavailable"].items()):
            lines.append(f"evaluation harness {target} is missing: {_count(len(runs), 'trained run')} "
                         f"{_verb(len(runs), 'waits', 'wait')} (e.g. {runs[0]})")
        for plugin, runs in sorted(report["plugins_unavailable"].items()):
            lines.append(f"plug-in {plugin!r} ({manifest.PLUGINS[plugin][0]}) is missing: {_count(len(runs), 'run')} "
                         f"{_verb(len(runs), 'waits', 'wait')} (e.g. {runs[0]})")
        if report["waiting"]:
            lines.append("waiting:")
            lines += [f"  {reason:40s} {len(runs)} (e.g. {runs[0]})"
                      for reason, runs in sorted(report["waiting"].items())]
        if report["decisions"]:
            lines.append("decisions:")
            lines += [f"  {run_id}: {text}" for run_id, text in report["decisions"]]
        for ev in self.events(limit=events):
            lines.append(f"{ev['ts']}  {ev['run_id'] or '-':60s} {ev['event']:24s} {ev['detail'] or ''}")
        return lines

    @staticmethod
    def _continuation_hint(spec: RunSpec) -> str:
        """What the group can do about a blocked continuation (``resolve`` refusals and ``schedule status``).

        A continuation takes no seed of its own (``RunSpec.with_seed`` refuses it), so neither ``replace``
        nor ``unblock`` applies to it: a few-shot continuation is repeated by replacing its parent, which
        supersedes it; a battery continuation's parent is a matched, ledgered run, which is never replaced.
        """
        if spec.group == "study_b_fewshot":
            return f"`schedule resolve {spec.depends_on[0]} replace` replaces its parent, which supersedes it"
        return "a battery continuation follows its parent's checkpoint; the group decides by an amendment (Table 9.1)"

    def resolve(self, run_id: str, action: str, *, defect_fixed: str | None = None) -> None:
        """Operator actions that never depend on a run's results.

        restart       an interrupted run, same seed, from scratch (reproduces it bit for bit); refused for a
                      run past Q-interrupted-run's cap (``_restart_refused``: more than MAX_RESTARTS ends of
                      process origin), which `set-policy restart` excludes by the rule (Table 9.1)
        reevaluate    an eval_failed run (evaluation is deterministic), or a ledger_failed run that is in
                      no ledger or sidecar (its evaluation.json must be redone: the ledger writer's
                      refusal says "re-evaluate"); the earlier evaluation.json is moved to
                      scheduler/reevaluated/<run>-<n>.json before the evaluate stage runs again
        retry-ledger  a ledger_failed run, or an excluded run whose exclusion is not yet in the ledger
        unblock       a blocked run whose dependencies are no longer excluded, blocked or superseded
        replace       repeats a run with the arm's next unused seed (Part 5.6): an excluded run left
                      without a replacement (circuit breaker, found in the ledger, or deferred for
                      uncommitted or moved code), or a blocked run whose dependency was excluded (for
                      a warm-started run, Table 9.1's rule, Q-warm-start). Once
                      the arm's exclusions with the run's cause reach its seed target (``seed_target``),
                      refused unless ``defect_fixed`` names the erratum of the defect found and fixed,
                      an entry of analysis/ERRATA.json committed at HEAD
                      (Table 9.1, Q-exclusion-breaker); the erratum is logged with the decision. Refused for
                      a continuation (few-shot or battery): it follows its parent's seed, so a
                      blocked few-shot continuation is repeated by replacing its parent (which
                      supersedes the parent's pending and blocked continuations); an excluded
                      continuation is left to the group's amendment. Refused likewise for an
                      auxiliary N = 0 run (``AUXILIARY_EXCLUDED``): its next unused seed would be
                      a seed of the N = 0 arm, which serves no warm-started run.
        add-auxiliary a pending warm-started run (a replacement) whose N = 0 dependency of its seed is
                      not queued: queues that N = 0 run as an auxiliary run (``_add_auxiliary``;
                      Table 9.1, Q-warm-start; refused while that key is open).
        requeue       a pending or blocked run that never trained (attempt 1, no output): its stored
                      spec is replaced by the one the committed code builds now (``current_spec``),
                      after a run gate or an answer changed (the command line requires a clean,
                      committed tree and logs the commit). A queued spec is otherwise never changed.
        Excluding an interrupted run one by one is not offered: Table 9.1 (Q-interrupted-run) restarts
        interrupted runs (`set-policy restart`); `set-policy exclude` remains for smoke data roots.
        """
        if action not in RESOLVE_ACTIONS:
            raise ValueError(f"action must be one of {RESOLVE_ACTIONS}")
        if defect_fixed is not None and action != "replace":
            raise ValueError(f"defect_fixed names the erratum of a `replace`, not of {action!r}")
        row = self.row(run_id)
        status = row["status"]
        # the mode only: the ledger location is fixed (and the ledgers marked) by the one action that writes ledger
        # records, retry-ledger, never by an action refused by its own gate or status
        self._fix_mode(ledgers=False)
        # HEAD for the gates and for the `queued` event of a replacement (_replacement_sql logs the commit whose code
        # computed it)
        self._refresh_git()
        if action == "restart":
            if status != "interrupted":
                raise SchedulerError(f"{run_id} is {status}; only an interrupted run can be restarted")
            ends = self._restart_refused(run_id)
            if ends:
                raise SchedulerError(f"{run_id} ended without a final result of process origin {ends} times (more "
                                     f"than MAX_RESTARTS = {MAX_RESTARTS}): it fails to complete its steps (Part 5.6; "
                                     "Table 9.1, Q-interrupted-run) and is not restarted again; `schedule set-policy "
                                     "restart` excludes it by that rule (cause 'incomplete') and queues its "
                                     "replacement")
            self.restart(run_id)
        elif action == "add-auxiliary":
            self._add_auxiliary(run_id)
        elif action == "requeue":
            self._requeue(run_id)
        elif action == "reevaluate":
            spec = self.spec(run_id)
            with self._write_txn():  # checked inside the write lock: never a second evaluation of a live one
                status = self.row(run_id)["status"]
                if status not in ("eval_failed", "ledger_failed"):
                    raise SchedulerError(f"{run_id} is {status}; only an eval_failed or ledger_failed run can be "
                                         "re-evaluated")
                # A ledger write refused for its evaluation.json (no per-episode arrays, an invalid record, seeds
                # that do not fit) needs a new evaluation; one already recorded keeps the evaluation it was
                # recorded with.
                if status == "ledger_failed" and recorded_anywhere(spec, self.paths):
                    raise SchedulerError(f"{run_id} is already recorded in a ledger or sidecar; its evaluation is the "
                                         f"recorded one: `schedule resolve {run_id} retry-ledger`")
                # The earlier attempt's evaluation.json is moved aside (kept for the record): a re-evaluation that
                # fails, or ends unobserved, must never find it and be taken for a success
                stale = self.cfg.run_dir(run_id) / EVALUATION_RESULT
                moved = ""
                if stale.exists():
                    aside = self.cfg.state_dir / "reevaluated"
                    aside.mkdir(parents=True, exist_ok=True)
                    n = 1
                    while (aside / f"{run_id}-{n}.json").exists():
                        n += 1
                    os.replace(stale, aside / f"{run_id}-{n}.json")
                    moved = f"; the earlier {EVALUATION_RESULT} is kept as {aside / f'{run_id}-{n}.json'}"
                self._set_sql(run_id, status="trained")
                self._event_sql(run_id, "reevaluate", "operator" + moved)
        elif action == "retry-ledger":
            if status in ("ledger_failed", "excluded"):
                self._fix_mode()
            if status == "ledger_failed":
                with self._write_txn():  # re-checked inside the write lock, like every other decision
                    status = self.row(run_id)["status"]
                    if status != "ledger_failed":
                        raise SchedulerError(f"{run_id} is {status}; nothing to retry")
                    self._set_sql(run_id, status="evaluated")
                    self._event_sql(run_id, "retry_ledger", "operator")
            elif status == "excluded":
                # the operator's retry must not read as a success when nothing was written
                problem = self._record_exclusion(run_id)
                if problem:
                    raise SchedulerError(f"{run_id}: the exclusion is still not in the ledger: {problem}")
            else:
                raise SchedulerError(f"{run_id} is {status}; nothing to retry")
        elif action == "unblock":
            spec = self.spec(run_id)
            with self._write_txn():  # checked inside the write lock: never unblocked and replaced at once
                row = self.row(run_id)
                if row["status"] != "blocked" or row["replaced_by"]:
                    raise SchedulerError(f"{run_id} is {row['status']} (replaced_by={row['replaced_by']}); "
                                         "only a blocked run without a replacement can be unblocked")
                if self._dependencies_state(spec) == "blocked":
                    hint = (self._continuation_hint(spec) if spec.group in CONTINUATION_GROUPS
                            else "`replace` queues the arm's next unused seed instead")
                    raise SchedulerError(f"{run_id}: a dependency is still excluded, blocked or superseded; {hint}")
                self._set_sql(run_id, status="pending")
                self._event_sql(run_id, "unblocked", "operator")
        else:  # replace
            spec = self.spec(run_id)
            if spec.group in CONTINUATION_GROUPS:
                # A continuation takes its parent's seed: a new seed of its own would continue a parent
                # that does not exist. Replacing a Study B parent queues the parent's new continuations;
                # a battery continuation's parent is a completed, ledgered run, which is never replaced.
                repeat = status == "blocked" and spec.group == "study_b_fewshot"
                raise SchedulerError(f"{run_id} is a continuation; it follows its parent's seed, not a new one"
                                     + (f"; to repeat it, `schedule resolve {spec.depends_on[0]} replace`"
                                        if repeat else "; the group decides by an amendment (Table 9.1)"))
            if self.is_auxiliary(run_id):
                raise SchedulerError(f"{run_id}: {AUXILIARY_EXCLUDED}")
            if defect_fixed is not None and not committed_log_entry(self._head, ERRATA_PATH, "errata", defect_fixed,
                                                                    self.cfg.repo_root):
                raise SchedulerError(f"--defect-fixed {defect_fixed!r} names no entry of {ERRATA_PATH} committed at "
                                     f"HEAD ({self._head}); record the fix as an erratum (Table 9.1) and commit it")
            with self._write_txn():  # checked inside the write lock: two decisions never queue two seeds
                row = self.row(run_id)
                status = row["status"]
                if status not in ("excluded", "blocked") or row["replaced_by"]:
                    raise SchedulerError(f"{run_id} is {status} (replaced_by={row['replaced_by']}); "
                                         "only an excluded or blocked run without a replacement can be replaced")
                if (status == "excluded" and defect_fixed is None
                        and self._same_cause_exclusions(spec.arm_id, row["failure_cause"]) >= self.seed_target(spec)):
                    raise SchedulerError(f"{run_id}: {self._replace_rule(run_id, spec, row['failure_cause'])}")
                replacement = self._replacement_sql(run_id, spec, surplus=bool(row["surplus"]))
                self._event_sql(run_id, "replace", f"operator: {replacement.run_id}" + (
                    f"; defect fixed, erratum {defect_fixed} ({ERRATA_PATH} at {self._head})" if defect_fixed else ""))
                if status == "blocked":  # the replacement runs instead of it
                    self._set_sql(run_id, status="superseded")
                    self._event_sql(run_id, "superseded", f"replaced by {replacement.run_id} " + (
                        "(Table 9.1, Q-warm-start: no value is taken from an excluded run)"
                        if spec.controller_variant == "warm_started" else "(operator)"))

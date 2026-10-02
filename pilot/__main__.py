"""Command line of the pilot owner's tools. Run ``python -m pilot --help`` from the repository root.

Owner: pilot owner (Role 1). Commands:
  design        list the runs of a design (pilot, repilot, main, treatment, controller, pid, study_a, study_b,
                study_b_fewshot, all)
  budget        run-equivalents: registered (Part 6 G2) and corrected, optionally after the first K cuts of Part 6.1
  schedule      add [--cuts K] | add-surplus | add-continuations | run | status | resolve | set-policy
                (the run scheduler; pilot/scheduler.py)
  enrich        select | measure | battery | match | controller | training | continuations | zeroshot |
                fewshot | final-battery | sensitivity-battery (pilot/enrichment.py)
  completeness  per arm: runs queued and their statuses, enrichment fields and supplement records missing (read-only)
  freeze        a manifest of the ledger, supplement and analysis file hashes and the analysis changes since
                the go report the decision was taken from (the earliest committed one of the last pilot
                revision; read-only; Part 5.8)
  g4-measure    evaluate the pilot Moderate run at its training budgets (Part 3.6; G4)
  allocate      record the machine-hour allocation H (before any pilot run is launched; refused once a registered run
                has started)
  go            compute go conditions G1 to G4 (Part 6) and the Table 8.1 values from the pilot
  surplus       the surplus-seed rule of Part 5.5

Exit status: 0 done (``enrich`` also prints what it left waiting for its data); 1 an expected
failure (a bad value, a missing file, a refused write, a scheduler state that cannot be read, a git
command that fails, an unstable simulation the harness reports, another role's code not in the
repository yet); 2 a usage error or a refused provenance check (a CliError; nothing more is written,
though ``enrich`` keeps the rows it wrote before the refusal); 3 refused by an open pre-registration
question or a refusing step (``pilot.errors.RunRefused``, e.g. ``PendingQuestionError``): the step
waits for the amendment log, nothing it depends on is written; 130 interrupted (Ctrl-C, e.g. to stop
``schedule run``). No expected failure prints a traceback.

``--allow-dirty`` and ``--allow-pending`` of ``enrich`` are for smoke ledgers only: never the
repository's results/, and only a ledger of a smoke data root (the mode its scheduler state stores,
and the ledgers it stores, else ``<data-root>/smoke``), so that a registered data root whose ledgers
lie elsewhere (``--ledger-dir``) is refused too. ``--allow-pending`` needs ``--allow-dirty`` and
opens the result gates of ``pilot.errors`` for the command (``set_pending_allowed``), never through
the environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import numbers
import os
import sqlite3
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from configs import registered as R
from pilot.provenance import REPO_ROOT

if TYPE_CHECKING:
    from pilot.ledger_writer import LedgerPaths
    from pilot.scheduler import Scheduler

PILOT_RESULTS = REPO_ROOT / "results" / "pilot"
DESIGNS = ("pilot", "repilot", "main", "treatment", "controller", "pid", "study_a", "study_b", "study_b_fewshot", "all")
DEFAULT_DATA_ROOT = os.environ.get("WSCL_DATA_ROOT", "/data")
MAX_POLL_SECONDS = 86_400.0  # one day: `schedule run --poll` above it is refused (time.sleep overflows on huge values)

EXIT_FAILURE = 1  # an expected failure; see the module docstring
EXIT_USAGE = 2  # a usage error or a refused provenance check (a CliError)
EXIT_REFUSED = 3  # an open question or a refusing step (pilot.errors.RunRefused)
EXIT_INTERRUPTED = 130  # stopped by Ctrl-C (KeyboardInterrupt), as a shell reports SIGINT


class CliError(Exception):
    """A usage error or a refused provenance check, reported without a traceback (exit 2)."""


def _ledger_paths(args: argparse.Namespace) -> LedgerPaths | None:
    from pilot.ledger_writer import LedgerPaths

    if not getattr(args, "ledger_dir", None):
        return None
    base = Path(args.ledger_dir)
    return LedgerPaths(main=base / "ledger.parquet", pilot=base / "pilot_ledger.parquet",
                       sidecar_dir=base / "sidecar")


def _in_repository_results(path: Path) -> bool:
    root = (REPO_ROOT / "results").resolve()
    resolved = Path(os.path.abspath(path)).resolve()
    return resolved == root or root in resolved.parents


def _cmd_design(args: argparse.Namespace) -> int:
    from pilot import manifest

    specs = manifest.design(args.name)
    if args.jsonl:
        Path(args.jsonl).write_text("\n".join(s.to_json() for s in specs) + "\n", encoding="utf-8")
    total = sum(s.run_equivalents for s in specs)
    waiting = [s for s in specs if s.open_questions]
    for s in specs if args.verbose else []:
        print(f"{s.run_id:60s} steps={s.total_steps:>11,d} onset={s.onset_step} "
              f"open={','.join(s.open_questions) or '-'}")
    print(f"{args.name}: {len(specs)} runs, {total:.2f} run-equivalents of training, "
          f"{len(waiting)} waiting on open questions")
    return 0


def _cmd_budget(args: argparse.Namespace) -> int:
    from pilot import budget, manifest

    if not 0 <= args.cuts <= len(manifest.CUTS):
        raise CliError(f"--cuts must be from 0 to {len(manifest.CUTS)} (Part 6.1 lists {len(manifest.CUTS)} cuts)")
    cuts = manifest.CUTS[: args.cuts]
    print(f"registered (Part 6 G2): {budget.registered_run_equivalents()} run-equivalents")
    if cuts:
        print(f"corrected after the cuts {', '.join(cuts)} (Part 6.1; a price, not a decision):")
    for group, value in budget.corrected_run_equivalents(cuts=cuts).items():
        print(f"corrected  {group:22s} {value:8.2f}")
    return 0


def _scheduler(args: argparse.Namespace) -> Scheduler:
    from pilot.scheduler import Scheduler, SchedulerConfig

    cfg = SchedulerConfig(
        data_root=Path(args.data_root),
        max_concurrent=getattr(args, "max_concurrent", 1),
        poll_seconds=getattr(args, "poll", 30.0),
        on_interrupt=getattr(args, "on_interrupt", None),  # None: the policy stored for the data root
        allow_dirty=getattr(args, "allow_dirty", False),
        allow_pending=getattr(args, "allow_pending", False),
        ledger_paths=_ledger_paths(args),
    )
    if args.action not in ("add", "add-surplus", "run") and not cfg.state_path.exists():
        raise CliError(f"no scheduler state at {cfg.state_path}; check --data-root "
                       "(only add, add-surplus and run create it)")
    return Scheduler(cfg)


QUEUEING = ("add", "add-surplus", "add-continuations")  # the actions that queue registered runs
# pilot.scheduler.RESOLVE_ACTIONS (not imported here: the parser is built without the scheduler;
# tests/test_scheduler.py checks that the two are equal)
RESOLVE_CHOICES = ("restart", "reevaluate", "retry-ledger", "unblock", "replace", "add-auxiliary", "requeue")
QUEUEING_DECISIONS = ("replace", "add-auxiliary", "requeue")  # `resolve` decisions that queue new specs (a
# replacement and its few-shot continuations; an auxiliary N = 0 run; a never-trained run's current spec), computed
# by the loaded code like those of QUEUEING
LEDGER_DECISIONS = ("retry-ledger",)  # `resolve` decisions that write ledger rows or supplement records (write-once,
# computed by the loaded pilot/ledger_writer.py and pilot/contracts.py): the commit's code must write them
# pilot.scheduler.INTERRUPT_POLICIES and pilot.manifest.BATTERY_CONDITIONS (not imported, as for RESOLVE_CHOICES;
# kept equal to them by hand: no test compares them)
POLICY_CHOICES = ("hold", "restart", "exclude")
CONTINUATION_CHOICES = ("finetune", "transfer")


def _check_schedule_values(args: argparse.Namespace) -> None:
    """The argument values ``schedule`` would refuse later, refused first as usage errors (CliError, exit 2).

    ``_scheduler`` creates the data root's state and the queueing actions fix its mode, so a value refused
    after that (``Scheduler.add_surplus``'s range, ``committed_cuts`` for ``--cuts K``, ``SchedulerConfig``'s
    ``max_concurrent`` and ``poll_seconds``) would leave a fresh data root fixed as registered with nothing
    queued.
    """
    from pilot import scheduler

    if args.action == "add" and args.cuts is not None and args.cuts < 0:
        raise CliError(f"--cuts must be a whole number from 0, got {args.cuts}")
    if args.action == "add-surplus" and not 0 <= args.extra <= R.MAX_SEEDS_PER_ARM - len(R.SEEDS):
        raise CliError(f"--extra must be from 0 to {R.MAX_SEEDS_PER_ARM - len(R.SEEDS)} (Part 5.5: at most "
                       f"{R.MAX_SEEDS_PER_ARM} seeds per arm), got {args.extra}")
    if args.action == "add" and args.cuts:
        try:
            scheduler.committed_cuts(args.cuts)
        except scheduler.SchedulerError as exc:
            raise CliError(str(exc)) from None
    if args.action == "run":
        if args.max_concurrent < 1:
            raise CliError(f"--max-concurrent must be at least 1, got {args.max_concurrent}")
        if not (math.isfinite(args.poll) and 0 <= args.poll <= MAX_POLL_SECONDS):
            raise CliError(f"--poll must be a number of seconds from 0 to {MAX_POLL_SECONDS:g}, got {args.poll}")
    if args.action == "status" and args.events < 0:
        raise CliError(f"--events must be a whole number from 0, got {args.events}")
    if args.action == "resolve" and args.defect_fixed is not None and args.decision != "replace":
        raise CliError(f"--defect-fixed names the erratum of a `replace`, not of {args.decision!r}")


def _cmd_schedule(args: argparse.Namespace) -> int:
    from pilot import manifest, provenance

    commit = None
    _check_schedule_values(args)  # bad values are usage errors, refused first, before any state exists
    if args.action in QUEUEING:
        # The stored spec (onset, total, params, dependencies, open questions) is what the launcher trains, and
        # it never re-derives it from the design: the code that computes it (pilot/manifest.py,
        # configs/registered.py, ...) must be the commit's, as for every run it trains. Checked before the
        # scheduler is built, so a refused command creates no state on the data root (exit 2: nothing written),
        # and again after the specs are computed (_check_queueing_code), before the queueing is logged.
        commit = _require_clean_tree(f"queueing registered runs (schedule {args.action})")
    specs: list[manifest.RunSpec] = []
    if args.action in QUEUEING:
        if args.action == "add":
            specs = manifest.design(args.design)
        # untracked or modified code imported to compute the specs (checked before anything is queued or fixed)
        unverified = provenance.unverified_imported_code()
        if unverified:
            raise CliError("the runs would be computed by code that the commit does not contain: "
                           + ", ".join(unverified))
    sched = _scheduler(args)
    if args.action in QUEUEING and sched.setting("mode") == "smoke":
        # registered runs queued here would train in smoke mode, without the allocation gate
        raise CliError(f"{sched.cfg.data_root} is a smoke data root; queue registered runs on another data root")
    if args.action in QUEUEING:
        # They queue registered runs only, so they fix the data root as registered: a later command with
        # --allow-dirty/--allow-pending is then refused instead of training these runs in smoke mode.
        sched.fix_mode(ledgers=False)  # the ledger location stays with the first run or decision
    if args.action == "add":
        # --cuts K: the first K cuts of the committed pilot/cuts.json (Part 6.1; pilot.scheduler.CUTS_PATH), fixed per data root
        added_count = sched.add(specs, cuts=args.cuts)
        _check_queueing_code(commit, args)
        sched.record_queued(f"schedule add --design {args.design}: {added_count} of {len(specs)} runs", commit)
        print(f"added {added_count} of {len(specs)} runs from design {args.design!r}")
    elif args.action == "add-surplus":
        added = sched.add_surplus(args.extra)
        _check_queueing_code(commit, args)
        sched.record_queued(f"schedule add-surplus --extra {args.extra}: {len(added)} runs", commit)
        print(f"added {len(added)} surplus training runs (Part 5.5): {', '.join(added) or '-'}")
    elif args.action == "add-continuations":
        conditions = tuple(args.conditions or manifest.BATTERY_CONDITIONS)
        _, skipped = sched.plan_continuations(Path(args.ledger), conditions)  # read-only: the reasons to print
        added = sched.add_continuations(Path(args.ledger), conditions)
        _check_queueing_code(commit, args)
        sched.record_queued(f"schedule add-continuations ({', '.join(conditions)}): {len(added)} runs", commit)
        print(f"added {len(added)} battery continuations (Table 2.2; {', '.join(conditions)}): "
              f"{', '.join(added) or '-'}")
        for run_id, why in skipped:
            print(f"  skipped {run_id}: {why}")
    elif args.action == "run":
        sched.run(once=args.once)
    elif args.action == "status":
        for line in sched.status_lines(events=args.events):
            print(line)
    elif args.action == "resolve":
        from pilot import errors

        if not sched.exists(args.run_id):  # checked before anything is written (exit 2: nothing written)
            raise CliError(f"unknown run_id {args.run_id!r}")
        queues = args.decision in QUEUEING_DECISIONS
        if queues and not args.allow_dirty:  # --allow-dirty: smoke roots only (Scheduler._fix_mode refuses it
            # elsewhere); the specs it queues are trained as stored, never re-derived: the commit's code must
            # compute them
            commit = _require_clean_tree(f"queueing runs (schedule resolve {args.decision})")
            unverified = provenance.unverified_imported_code()
            if unverified:
                raise CliError("the runs would be computed by code that the commit does not contain: "
                               + ", ".join(unverified))
        elif queues:  # --allow-dirty: the log names HEAD, and says that its code was not checked
            commit = f"{provenance.commit_hash()} (smoke: uncommitted code allowed)"
        if args.decision in LEDGER_DECISIONS and not args.allow_dirty:
            _require_committed_ledger_code(f"writing the ledger (schedule resolve {args.decision})")
        previous = errors.pending_allowed()
        errors.set_pending_allowed(args.allow_pending)  # smoke roots only (Scheduler._fix_mode refuses it elsewhere)
        try:
            # any other KeyError is main's (exit 1), not a usage error
            sched.resolve(args.run_id, args.decision, defect_fixed=args.defect_fixed)
        finally:
            errors.set_pending_allowed(previous)
        if queues:
            if not args.allow_dirty:
                _check_queueing_code(commit, args)
            sched.record_queued(f"schedule resolve {args.run_id} {args.decision}", commit)
        print(f"{args.run_id}: {args.decision}")
    elif args.action == "set-policy":
        if args.policy == "exclude" and not args.allow_dirty:  # it writes the held runs' exclusion rows
            _require_committed_ledger_code("writing exclusion rows to the ledger (schedule set-policy exclude)")
        sched.set_policy(args.policy)
        print(f"interruption policy is now {args.policy!r} for every run on {args.data_root}")
    return 0


def _check_queueing_code(commit: str | None, args: argparse.Namespace) -> None:
    """Re-check, after the specs were computed and before the queueing is logged, that the code they were
    computed by is ``commit``'s: the scheduler imports some of it lazily (``plan_continuations``, ``resolve``),
    and HEAD may move while the command runs (as ``enrich``, ``go`` and ``g4-measure`` re-check)."""
    from pilot import provenance

    if commit is None:
        return
    unverified = provenance.unverified_imported_code()
    if unverified:
        raise CliError("the runs were computed by code that the commit does not contain: " + ", ".join(unverified))
    _require_code_of(commit, f"schedule {args.action}")


ENRICH_ACTIONS = ("select", "measure", "battery", "match", "controller", "training", "continuations", "zeroshot",
                  "fewshot", "final-battery", "sensitivity-battery")
BATTERY_CHOICES = ("hazard", "dynamics")  # pilot.enrichment.CONDITION_FIELDS (Table 2.2 rows 1-2)


def _enrich_conditions(args: argparse.Namespace) -> list[str]:
    """``--conditions`` of the actions that take them, with each action's default."""
    from pilot import manifest

    allowed = manifest.BATTERY_CONDITIONS if args.action == "continuations" else BATTERY_CHOICES
    defaults = {"battery": R.PILOT_BATTERY, "continuations": manifest.BATTERY_CONDITIONS,
                "final-battery": ("hazard",), "sensitivity-battery": ("hazard",)}
    if args.action not in defaults:
        if args.conditions:
            raise CliError(f"enrich {args.action} takes no --conditions")
        return []
    conditions = list(args.conditions or defaults[args.action])
    wrong = [c for c in conditions if c not in allowed]
    if wrong or len(set(conditions)) != len(conditions):
        raise CliError(f"enrich {args.action} --conditions must be distinct values of {list(allowed)}, "
                       f"got {conditions}")
    return conditions


def _stored_ledgers(value: str) -> list[Path]:
    """The main and pilot ledgers of a data root's stored ``ledgers`` setting (``pilot.budget.stored_ledgers``)."""
    from pilot import budget

    return budget.stored_ledgers(value)[:2]


def _check_smoke_ledger(ledger: Path, data_root: str | Path) -> None:
    """``--allow-dirty`` (and so ``--allow-pending``) only on a ledger of a smoke data root (pilot/enrichment.py):
    --allow-pending only together with --allow-dirty, and on smoke roots only.

    A value is written once and never changed (pilot/enrichment.py), so one computed under uncommitted code or under a
    proposal whose key is open must never reach a registered ledger, wherever it lies (``schedule run
    --ledger-dir`` puts a registered data root's ledgers outside the repository). The data root's own
    scheduler state decides: a data root whose stored mode is not smoke is refused; a smoke data root's
    ledgers are those it stored (``--ledger-dir``), else ``ledger_writer.smoke_paths``; a data root with
    no mode yet (no scheduler state, or none fixed) is taken as smoke only for the ledgers of
    ``smoke_paths``, the one place no registered scheduler writes by default. A ledger marked registered
    (``ledger_writer.ledger_mode``: a registered scheduler fixed it) is refused whatever data root is named.
    """
    from pilot import enrichment
    from pilot.ledger_writer import ledger_mode, mode_marker, smoke_paths

    if ledger_mode(ledger) == "registered":  # a registered data root's ledger, whichever --data-root is named
        raise CliError(f"--allow-dirty and --allow-pending are for smoke ledgers only; {ledger} is a registered ledger "
                       f"({mode_marker(ledger)}), and a value written to it is never changed (pilot/enrichment.py)")
    state = enrichment.read_scheduler_state(data_root)
    mode = None if state is None else state.settings.get("mode")
    if mode is not None and mode != "smoke":
        raise CliError(f"--allow-dirty and --allow-pending are for smoke data roots only; {data_root} holds "
                       f"{mode} runs (its scheduler state), and a value written to its ledgers is never changed "
                       "(pilot/enrichment.py)")
    stored = state.settings.get("ledgers") if state is not None and mode == "smoke" else None
    defaults = smoke_paths(Path(data_root))
    ledgers = _stored_ledgers(stored) if stored else [defaults.main, defaults.pilot]
    if os.path.realpath(ledger) not in {os.path.realpath(p) for p in ledgers}:
        listed = ", ".join(str(p) for p in ledgers)
        where = (f"a ledger of the smoke data root {data_root} ({listed})" if mode == "smoke" else
                 f"a smoke ledger of {data_root}, which has no mode yet (a smoke scheduler fixes it; until then only "
                 f"{listed} count)")
        raise CliError(f"--allow-dirty and --allow-pending are for smoke ledgers only; {ledger} is not {where}")


def _cmd_enrich(args: argparse.Namespace) -> int:
    from pilot import enrichment, errors, provenance
    from pilot.scheduler import SchedulerConfig

    ledger = Path(args.ledger)
    if args.allow_dirty and _in_repository_results(ledger):
        raise CliError(f"--allow-dirty is for smoke ledgers only; {ledger} is in the repository's results/")
    if args.allow_pending and not args.allow_dirty:
        raise CliError("--allow-pending is for smoke ledgers only and needs --allow-dirty (pilot/enrichment.py)")
    conditions = _enrich_conditions(args)  # usage errors first: they read nothing
    if args.action in ("match", "sensitivity-battery") and args.surplus_extra is None:
        raise CliError(f"enrich {args.action} needs --surplus-extra E: the surplus seeds per primary-comparison arm "
                       "(Part 5.5; 0 without surplus), which set each arm's seed target (Q-arm-complete)")
    if args.surplus_extra is not None and args.action not in ("match", "sensitivity-battery"):
        raise CliError(f"enrich {args.action} takes no --surplus-extra")
    if not args.allow_dirty and provenance.dirty_paths():
        raise CliError("tracked code has uncommitted changes; commit before enriching the ledger "
                       "(the log records the commit)")
    start = None if args.allow_dirty else provenance.commit_hash()  # the commit the records and the log name
    if not ledger.is_file():
        raise CliError(f"the ledger {ledger} does not exist")
    if args.allow_dirty:  # --allow-pending needs it (above): both are checked against the data root here
        _check_smoke_ledger(ledger, args.data_root)

    def check_code() -> None:
        # The owners' code is imported lazily: check after it ran, before each write.
        unverified = [] if args.allow_dirty else provenance.unverified_imported_code()
        if unverified:
            raise CliError("the enrichment ran code that the commit does not contain: " + ", ".join(unverified))
        if start is not None:
            _require_code_of(start, "the enrichment")

    run_dir_of = SchedulerConfig(data_root=Path(args.data_root)).run_dir
    state = enrichment.read_scheduler_state(args.data_root) if args.action in (
        "match", "sensitivity-battery", "continuations", "fewshot") else None
    status_of = state.status_of if state is not None else None
    stored_extra = state.settings.get("surplus_extra") if state is not None else None
    common = {"check_code": check_code, "data_root": args.data_root}
    previous = errors.pending_allowed()
    errors.set_pending_allowed(args.allow_pending)  # smoke ledgers only (checked above); restored on return
    try:
        if args.action == "select":
            done = enrichment.apply_selection(ledger, check_code=check_code)
        elif args.action == "measure":
            done = enrichment.apply_measurement(ledger, run_dir_of, **common)
        elif args.action == "battery":
            done = enrichment.apply_battery(ledger, run_dir_of, conditions, **common)
        elif args.action == "match":
            done = enrichment.apply_matching(ledger, surplus_extra=args.surplus_extra,
                                             stored_surplus_extra=stored_extra, check_code=check_code)
        elif args.action == "controller":
            done = enrichment.apply_controller(ledger, run_dir_of, **common)
        elif args.action == "training":
            done = enrichment.apply_training(ledger, run_dir_of, **common)
        elif args.action == "continuations":
            done = enrichment.apply_continuations(ledger, run_dir_of, conditions, status_of=status_of, **common)
        elif args.action == "zeroshot":
            done = enrichment.apply_zero_shot(ledger, run_dir_of, **common)
        elif args.action == "fewshot":
            done = enrichment.apply_fewshot(ledger, run_dir_of, status_of=status_of, **common)
        elif args.action == "final-battery":
            done = enrichment.apply_final_battery(ledger, run_dir_of, conditions, **common)
        else:
            done = enrichment.apply_sensitivity_battery(ledger, run_dir_of, conditions,
                                                        surplus_extra=args.surplus_extra,
                                                        stored_surplus_extra=stored_extra, **common)
    finally:
        errors.set_pending_allowed(previous)
    print(f"{args.action}: {len(done)} runs enriched")
    # What waits for its data (an incomplete arm, a run not selected yet, a continuation not run yet): nothing
    # was written for it, and it is said here, not only in the enrichment log (pilot.enrichment.Done).
    waiting = getattr(done, "waiting", {})
    if waiting:
        print(f"{args.action}: {len(waiting)} not written, waiting for their data:")
        for what, why in sorted(waiting.items()):
            print(f"  {what}: {why}")
    # What an open report key's other readings would give differently (Q-threshold-arithmetic for the flags):
    # written under its proposal, which gates nothing; the analysis reports what depends on it as provisional.
    provisional = getattr(done, "provisional", {})
    if provisional:
        print(f"{args.action}: {len(provisional)} computed under the proposal of an open report key "
              "(it gates nothing; the analysis reports what depends on it as PROVISIONAL or UNDECIDED):")
        for what, why in sorted(provisional.items()):
            print(f"  {what}: {why}")
    return 0


# The statuses that count as finished: the same as the literal in pilot.enrichment.completeness_report that
# computes totals["not_finished"] (kept equal by hand).
FINISHED_STATUSES = ("ledgered", "continued", "excluded", "superseded")


def _cmd_completeness(args: argparse.Namespace) -> int:
    from pilot import enrichment

    ledger = Path(args.ledger)
    if not ledger.is_file():  # a mistyped path is not "nothing missing"
        raise CliError(f"the ledger {ledger} does not exist")
    report = enrichment.completeness_report(ledger, enrichment.read_scheduler_state(args.data_root))
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
        return 0
    totals = report["totals"]
    print(f"{ledger}: {totals['ledger_rows']} rows, {totals['arms']} arms, {totals['queued']} runs queued on "
          f"{args.data_root} ({'no scheduler state' if not report['scheduler_state'] else 'scheduler state read'})")
    print(f"missing: {totals['fields_missing']} enrichment fields, {totals['records_missing']} supplement records; "
          f"{totals['not_finished']} scheduled runs not finished")
    for arm, entry in report["arms"].items():
        unfinished = {s: n for s, n in entry["by_status"].items() if s not in FINISHED_STATUSES}
        if not (entry["fields_missing"] or entry["records_missing"] or unfinished):
            continue
        print(f"{arm}: queued {entry['queued']}, trained {entry['trained']}, ledgered {entry['ledgered']}; "
              f"ledger rows {entry['ledger_rows']} ({entry['completed']} completed)"
              + (f", statuses {unfinished}" if unfinished else ""))
        for name, runs in sorted(entry["fields_missing"].items()):
            print(f"  field {name:28s} missing in {len(runs)}: {', '.join(runs[:3])}{' ...' if len(runs) > 3 else ''}")
        for name, runs in sorted(entry["records_missing"].items()):
            print(f"  record {name:27s} missing for {len(runs)}: "
                  f"{', '.join(runs[:3])}{' ...' if len(runs) > 3 else ''}")
    for note in report["notes"]:
        print(f"note: {note}")
    return 0


def _cmd_freeze(args: argparse.Namespace) -> int:
    from pilot import enrichment

    ledger = Path(args.ledger)
    report = enrichment.freeze_report(ledger, state=enrichment.read_scheduler_state(args.data_root),
                                      data_root=args.data_root)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
        return 0
    print(f"HEAD {report['head']}; {len(report['files'])} files hashed (use --json for the manifest)")
    for entry in report["go_reports"]:
        print(f"go report {entry['path']}: revision {entry['revision']}, commit {entry['commit']}"
              + ("" if entry["readable"] else " (not committed at HEAD, or unreadable)"))
    go = report["go_report"]
    if go:
        print(f"go decision taken from: {go['path']} at {go['commit']} (the earliest committed report of revision "
              f"{go['revision']})")
    else:
        print("go decision taken from: no committed go report")
    changes = report["analysis_changes_since_go"]
    if changes is not None:
        print(f"analysis/ changed in {len(changes)} commits since the go report (each must be a Part 9 erratum):")
        for line in changes:
            print(f"  {line}")
    print("ready to freeze" if report["ready"] else "not ready to freeze:")
    for problem in report["problems"]:
        print(f"  {problem}")
    return 0


def _cmd_g4_measure(args: argparse.Namespace) -> int:
    from pilot import manifest, provenance
    from pilot.contracts import ContractError
    from pilot.enrichment import _load, check_same_commit, matched_checkpoint_path
    from pilot.launch import SPEC_FILE, TRAIN_RESULT, plain
    from pilot.ledger_writer import DEFAULT_PATHS
    from pilot.scheduler import SchedulerConfig
    from results.ledger_schema import load_ledger_as_rows

    out = Path(args.out) if args.out else _g4_path(args.revision)
    if out.exists():  # checked first: the evaluation is long and is taken once
        raise CliError(f"{out} exists; the G4 measurement is taken once")
    if not DEFAULT_PATHS.pilot.exists():
        raise CliError(f"{DEFAULT_PATHS.pilot} does not exist yet; the Moderate pilot run must be ledgered first")
    commit = _require_clean_tree("taking the G4 measurement")
    # The completed Moderate pilot run (seed 0, or its replacement seed after an exclusion).
    prefix = manifest.pilot_prefix(args.revision)
    moderate = [
        r for r in load_ledger_as_rows(DEFAULT_PATHS.pilot)
        if r.study == "B" and r.arm == "Moderate" and r.completed and r.run_id.startswith(prefix)
    ]
    if len(moderate) != 1:
        raise CliError(f"expected one completed Moderate pilot run in the pilot ledger, found {len(moderate)}")
    cfg = SchedulerConfig(data_root=Path(args.data_root))
    run_dir = cfg.run_dir(moderate[0].run_id)
    spec = manifest.RunSpec.from_json((run_dir / SPEC_FILE).read_text(encoding="utf-8"))
    levels = sorted(float(b) for b in spec.training_levels or ())
    registered = sorted(float(b) for b in R.STUDY_B_ARMS["Moderate"])
    if not levels or levels != registered:  # checked before the long evaluation; go_decision reads the same list
        raise CliError(f"{spec.run_id}: the stored training budgets {levels} are not the Moderate arm's registered "
                       f"{registered} (Part 3.6; Part 6 G4)")
    train = json.loads((run_dir / TRAIN_RESULT).read_text(encoding="utf-8"))
    try:  # a smoke data root holds runs with the same run_ids
        check_same_commit(spec.run_id, run_dir, train, moderate[0].commit_hash)
    except ContractError as exc:
        raise CliError(str(exc)) from None
    # The same run, not a namesake under another data root at the same commit: the final checkpoint
    # must be the one the ledger records.
    final = max(moderate[0].checkpoints, key=lambda c: c.step)
    path = matched_checkpoint_path(run_dir / train["omnisafe_dir"], final.step)
    if spec.run_id != moderate[0].run_id or path != final.path:
        raise CliError(
            f"{moderate[0].run_id}: {run_dir} gives the final checkpoint {path!r}, but the ledger records "
            f"{final.path!r}; use the data root the ledger was written from"
        )
    evaluate = _load(("studyb.evaluation", "evaluate_training_budgets", "Study B and literature (Role 5)"))
    # Q-studyb-eval gates the evaluation inside the harness: its PendingQuestionError is a refusal (exit 3)
    result = plain(dict(evaluate(str(run_dir / train["omnisafe_dir"]), spec.to_dict())))
    unverified = provenance.unverified_imported_code()
    if unverified:
        raise CliError("the evaluation ran code that the commit does not contain: " + ", ".join(unverified))
    _require_code_of(commit, "the G4 measurement")
    episodes = result.get("episodes")
    if isinstance(episodes, bool) or not isinstance(episodes, numbers.Integral) or int(episodes) != R.EVAL_EPISODES:
        raise CliError(f"the Moderate evaluation must use {R.EVAL_EPISODES} episodes per budget, got {episodes!r}")
    costs = _numbers_by_budget(result.get("mean_cost"))
    if sorted(costs) != levels or not all(math.isfinite(v) and v >= 0 for v in costs.values()):
        raise CliError(f"the Moderate evaluation must give a finite, non-negative mean_cost for each training budget "
                       f"{levels} (Part 6 G4; HANDOVER.md section 8), got {result.get('mean_cost')!r}")
    rates = _numbers_by_budget(result.get("satisfaction"))
    if sorted(rates) != levels or not all(0.0 <= v <= 1.0 for v in rates.values()):
        raise CliError(f"the Moderate evaluation must give a satisfaction rate in [0, 1] for each training budget "
                       f"{levels} (Part 3.6; HANDOVER.md section 8), got {result.get('satisfaction')!r}")
    _check_g4_seeds(result.get("seeds"), DEFAULT_PATHS.seeds_registry(spec))
    _check_g4_replacements(result.get("unstable_replacements"), levels)
    result["mean_cost"], result["satisfaction"] = costs, rates  # the checked numbers, not the harness's objects
    result["provenance"] = {"run_id": spec.run_id, "commit": commit, "utc": provenance.utc_now().isoformat(),
                            "data_root": os.path.abspath(args.data_root)}
    _write_once(out, json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
                "the G4 measurement is taken once")
    print(f"wrote {out}")
    return 0


def _check_g4_seeds(seeds: object, registry: Path) -> None:
    """The G4 measurement runs on the measurement set (Table 2.1; Q-studyb-eval, Table 9.1): ``R.EVAL_EPISODES``
    distinct seeds, exactly ``envs.evaluation.measurement_seeds()``, the set every run is measured on,
    and the measurement set the pilot ledger's runs fixed in its seed registry (``registry``; read only:
    a harness changed after those runs is refused, never a new set recorded)."""
    from envs.evaluation import measurement_seeds  # Role 2's seed sets; studyb.evaluation is built on them
    from pilot.contracts import ContractError, validate_seed_set
    from pilot.ledger_writer import _check_seeds_fit

    try:
        given = validate_seed_set("seeds", seeds)
    except ContractError as exc:
        raise CliError(f"the Moderate evaluation must record its episode seeds: {exc}") from None
    if given != list(measurement_seeds()):
        raise CliError("the Moderate evaluation did not run on the measurement seeds of envs.evaluation "
                       "(Table 2.1: the same measurement set for every run)")
    try:
        _check_seeds_fit("measurement", given, registry)
    except ContractError as exc:
        raise CliError(f"the Moderate evaluation's seeds are not the pilot runs' measurement set: {exc}") from None


def _check_g4_replacements(replacements: object, levels: list[float]) -> None:
    """Each training budget's replacements of unstable episodes (Q-mujoco-exception, Table 9.1): the reserve seeds
    of the measurement set's reserve sequence, at most ``MAX_UNSTABLE_EPISODES``, against the planned measurement
    seeds that ``_check_g4_seeds`` compared (``pilot.contracts.validate_unstable_replacements``). None: none."""
    from envs.evaluation import measurement_seeds
    from pilot.contracts import ContractError, validate_unstable_replacements

    if replacements is None:
        return
    if not isinstance(replacements, Mapping):
        raise CliError(f"the Moderate evaluation's unstable_replacements must map budgets to lists, got {replacements!r}")
    try:
        by_budget = {float(k): v for k, v in replacements.items()}
    except (TypeError, ValueError):
        raise CliError("the Moderate evaluation's unstable_replacements are not keyed by budgets") from None
    if len(by_budget) != len(replacements) or not set(by_budget) <= set(levels):
        raise CliError(f"the Moderate evaluation's unstable_replacements name budgets {list(replacements)}, not "
                       f"training budgets {levels}")
    for budget, entries in sorted(by_budget.items()):
        try:
            validate_unstable_replacements(f"the Moderate evaluation at budget {budget:g}", entries, measurement_seeds())
        except ContractError as exc:
            raise CliError(str(exc)) from None


def _numbers_by_budget(mapping: object) -> dict[float, float]:
    """``{budget: value}`` as floats, or ``{}`` (refused by the caller) if ``mapping`` is not a mapping, if
    any value is not a real number (a bool, a string or None never passes as a cost or a rate), or if two
    keys name the same budget (``'10'`` and ``10.0``)."""
    if not isinstance(mapping, Mapping):
        return {}
    items = list(mapping.items())
    if not all(isinstance(v, numbers.Real) and not isinstance(v, bool) for _, v in items):
        return {}
    try:
        converted = {float(k): float(v) for k, v in items}
    except (TypeError, ValueError):
        return {}
    return converted if len(converted) == len(items) else {}


def _require_clean_tree(purpose: str) -> str:
    """HEAD's hash; a CliError (exit 2) if tracked code has uncommitted changes."""
    from pilot import provenance

    try:
        return provenance.require_clean_worktree()
    except provenance.DirtyWorktreeError:
        raise CliError(f"tracked code has uncommitted changes; commit before {purpose}: "
                       + ", ".join(provenance.dirty_paths())) from None


def _require_committed_ledger_code(purpose: str) -> None:
    """A CliError unless the tree is clean and every loaded module is committed: ledger rows are write-once."""
    from pilot import provenance

    _require_clean_tree(purpose)
    unverified = provenance.unverified_imported_code()
    if unverified:
        raise CliError(f"the ledger would be written by code that the commit does not contain ({purpose}): "
                       + ", ".join(unverified))


def _require_code_of(start: str, what: str) -> None:
    """A CliError if HEAD moved from ``start`` (the commit the command read when it started, and that its output
    names) to a commit that changes a module this process loaded: ``unverified_imported_code`` compares the
    working tree with the new HEAD only, so code loaded before such a commit, or imported after it, would be
    attributed to a commit that did not compute it (the scheduler's ``_ledger_code_problem``)."""
    from pilot import provenance

    head = provenance.commit_hash()
    if head == start:
        return
    try:
        moved = provenance.imported_code_changed_between(start, head)
    except subprocess.CalledProcessError:
        moved = [f"(git cannot compare {start} with {head})"]
    if moved:
        raise CliError(f"HEAD moved from {start} to {head} while {what} ran, and that commit changes code this "
                       "command loaded: " + ", ".join(moved) + "; nothing more is written: run the command again")


def _write_once(path: Path, text: str, why: str) -> None:
    """Create ``path`` with ``text`` atomically and only once: a crash never leaves a partial file."""
    from pilot import provenance

    try:
        provenance.write_text_once(path, text)
    except FileExistsError:
        raise CliError(f"{path} exists; {why}") from None


def _cmd_allocate(args: argparse.Namespace) -> int:
    """Record H once (Part 6 G2; Table 8.1; X-allocation, Table 9.1): refused once a registered run has started
    (its throughput may be read), with a warning when workstation timings already exist (they are stored in the
    record), and printed with the Table 8.1 cell and the Table 9.1 row to enter."""
    from pilot import budget

    started = budget.registered_runs_started()
    if started:
        raise CliError("the allocation is recorded before the pilot's throughput is read (Part 6 G2; Table 8.1), but "
                       "a registered run has already started: " + "; ".join(started))
    timings = budget.prior_workstation_timings(args.smoke_root or ())
    record = budget.record_allocation(args.hours, args.by, args.date, args.basis, prior_workstation_timings=timings,
                                      confirm=args.confirm)
    print(json.dumps(record, indent=2))
    if timings:
        print(f"warning: {len(timings)} record{'' if len(timings) == 1 else 's'} of runs on the workstation existed "
              "when H was recorded, so its speed "
              "was already visible (Part 6 G2 asks for the allocation before throughput is read); the record lists "
              "them (prior_workstation_timings): " + ", ".join(timings), file=sys.stderr)
    hours = f"{record['machine_hours_allocated']:.10g} {budget.ALLOCATION_UNIT}"
    print(f"Table 8.1, 'Machine-hours allocated to the registered runs': {hours}, decided by {args.by} on {args.date}")
    print(f"Table 9.1 row: Machine-hour allocation | {args.date} | <the commit that adds pilot/allocation.json> | "
          f"{hours}, basis: {args.basis} | Part 6 G2 | {args.by}")
    print("Now commit pilot/allocation.json. The scheduler launches no pilot run until HEAD contains it (Part 6 G2).")
    return 0


def _g4_path(revision: int) -> Path:
    return PILOT_RESULTS / ("g4_moderate.json" if revision == 0 else f"g4_moderate-rev{revision}.json")


def _load_pilot_records(ledger: Path, sidecar_dir: Path, revision: int = 0) -> list[dict]:
    """The records of one pilot (revision 0) or of the re-pilot (revision 1), ledger and sidecar."""
    from pilot.manifest import pilot_prefix
    from results.ledger_schema import load_ledger_as_rows

    rows = [r.model_dump(mode="python") for r in load_ledger_as_rows(ledger)] if ledger.exists() else []
    for path in sorted(sidecar_dir.glob("*.json")) if sidecar_dir.exists() else []:
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    prefix = pilot_prefix(revision)
    return [r for r in rows if str(r.get("run_id", "")).startswith(prefix)]


def _pilot_inputs(ledger: Path, sidecar_dir: Path) -> dict:
    """The sha256 of the pilot ledger and of each sidecar file ``_load_pilot_records`` reads (None: absent). Both
    are output paths, exempt from the clean-tree check, so the go report records them (as ``python -m analysis``
    records ledger_sha256)."""

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    sidecars = sorted(sidecar_dir.glob("*.json")) if sidecar_dir.exists() else []
    return {"pilot_ledger": {"path": str(ledger), "sha256": digest(ledger) if ledger.exists() else None},
            "sidecar": {"dir": str(sidecar_dir), "sha256": {p.name: digest(p) for p in sidecars}}}


def _final_selection_cost(record: dict) -> float | None:
    """The selection-set cost of a record's final checkpoint (its checkpoint record), or None."""
    checkpoints = [c for c in record.get("checkpoints") or [] if isinstance(c, dict) and c.get("step") is not None]
    if not checkpoints:
        return None
    value = max(checkpoints, key=lambda c: c["step"]).get("selection_cost")
    return float(value) if isinstance(value, numbers.Real) and not isinstance(value, bool) else None


def _as_datetime(value: object) -> datetime:
    return value if isinstance(value, datetime) else datetime.fromisoformat(str(value))


def _cmd_go(args: argparse.Namespace) -> int:
    from pilot import budget, go_decision, provenance
    from pilot.ledger_writer import DEFAULT_PATHS

    if args.concurrent is not None and args.concurrent < 1:
        raise CliError(f"--concurrent must be at least 1, got {args.concurrent}")
    if args.moderate and not Path(args.moderate).exists():  # only the default may be absent (not measured yet)
        raise CliError(f"--moderate {args.moderate} does not exist")
    commit = _require_clean_tree("computing the go decision (Part 6; the report records the commit)")
    revision = 1 if args.repilot else 0
    inputs = _pilot_inputs(DEFAULT_PATHS.pilot, DEFAULT_PATHS.sidecar_dir)
    records = _load_pilot_records(DEFAULT_PATHS.pilot, DEFAULT_PATHS.sidecar_dir, revision)
    ledger_rows = [r for r in records if "reason" not in r]
    completed = [r for r in records if r.get("completed")]
    intervals = [(_as_datetime(r["started"]), _as_datetime(r["finished"]))
                 for r in completed if r.get("started") and r.get("finished")]
    measured = go_decision.measured_concurrency(intervals)
    concurrent = args.concurrent if args.concurrent is not None else int(measured["max"])
    if concurrent < 1:
        raise CliError("no completed pilot run: concurrency cannot be measured yet")
    if concurrent > measured["max"]:
        raise CliError(f"--concurrent {concurrent} exceeds the {int(measured['max'])} runs the pilot actually ran "
                       "at once")
    allocation_ok, note = budget.allocation_contained(records)
    allocated = budget.load_allocation()["machine_hours_allocated"] if budget.ALLOCATION_FILE.exists() else None
    unconstrained_runs = [r for r in completed if r.get("arm") == "unconstrained"]
    unconstrained = unconstrained_runs[0].get("final_cost") if len(unconstrained_runs) == 1 else None
    # Q-final-cost-set's other reading: the selection-set cost of the same (final) checkpoint
    unconstrained_selection = _final_selection_cost(unconstrained_runs[0]) if len(unconstrained_runs) == 1 else None
    moderate_path = Path(args.moderate) if args.moderate else _g4_path(revision)
    moderate = json.loads(moderate_path.read_text(encoding="utf-8")) if moderate_path.exists() else None
    g4_input = None if moderate is None else _check_g4_input(moderate, moderate_path, completed, revision)
    hours = [float(r["wall_clock_hours"]) for r in completed if r.get("wall_clock_hours")]
    result = go_decision.report(
        ledger_rows,
        hours_per_run=hours,
        concurrent_runs=concurrent,
        machine_hours_allocated=allocated,
        allocation_ok=allocation_ok,
        allocation_note=note,
        unconstrained_cost=unconstrained,
        moderate=moderate,
        failed_before=args.repilot,
        measured=measured,
        unconstrained_selection_cost=unconstrained_selection,
    )
    unverified = provenance.unverified_imported_code()
    if unverified:
        raise CliError("the go decision ran code that the commit does not contain: " + ", ".join(unverified))
    _require_code_of(commit, "the go decision")
    earlier = _go_reports_of_revision(Path(args.out_dir), revision)
    if earlier:  # Part 5.8's freeze reads the analysis changes from the earliest committed report of the revision
        print(f"note: go reports of this pilot revision exist already ({', '.join(p.name for p in earlier)}); "
              "`python -m pilot freeze` measures the analysis changes from the earliest one committed",
              file=sys.stderr)
    result["generated"] = {"utc": provenance.utc_now().isoformat(), "commit": commit,
                           "worktree_dirty": bool(provenance.dirty_paths()),
                           "argv": getattr(args, "argv", sys.argv[1:]),
                           "revision": revision, "g4_moderate": g4_input,
                           "earlier_reports": [p.name for p in earlier], "inputs": inputs,
                           "run_ids": sorted(str(r["run_id"]) for r in records)}
    if _pilot_inputs(DEFAULT_PATHS.pilot, DEFAULT_PATHS.sidecar_dir) != inputs:  # as the analysis CLI re-checks
        raise CliError(f"the pilot ledger {DEFAULT_PATHS.pilot} or its sidecar {DEFAULT_PATHS.sidecar_dir} changed "
                       "while the go decision was computed; run `go` again")
    stamp = provenance.utc_now().strftime("%Y%m%dT%H%M%SZ")
    js, md = go_decision.write_report(result, Path(args.out_dir), stamp)
    print(Path(md).read_text(encoding="utf-8"))
    print(f"wrote {js} and {md}")
    return 0


def _go_reports_of_revision(directory: Path, revision: int) -> list[Path]:
    """The go reports under ``directory`` taken for this pilot revision (``pilot.enrichment.list_go_reports``)."""
    from pilot import enrichment

    return [Path(r["file"]) for r in enrichment.list_go_reports(directory, read=enrichment.read_working_file)
            if r["revision"] == revision]


def _commit_in_history(commit: str) -> bool:
    """True when ``commit`` is HEAD or one of its ancestors (``git merge-base --is-ancestor``; read-only)."""
    from pilot import provenance

    if not provenance.is_full_commit_hash(commit):
        return False
    out = subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=REPO_ROOT, capture_output=True)
    return out.returncode == 0


def _check_g4_input(moderate: dict, path: Path, completed: list[dict], revision: int) -> dict:
    """The G4 Moderate measurement must be ``g4-measure``'s of this revision's completed Moderate pilot run, taken at
    a commit of HEAD's history (not any file with a mean_cost map, and the report says which); returns what the go
    report records of it (path, sha256, provenance)."""
    from pilot.manifest import pilot_prefix

    prov = moderate.get("provenance") if isinstance(moderate, dict) else None
    if (not isinstance(prov, dict) or not isinstance(prov.get("run_id"), str)
            or not isinstance(prov.get("commit"), str)):
        raise CliError(f"{path} records no provenance (run_id, commit): only `python -m pilot g4-measure` writes "
                       "the G4 measurement")
    prefix = pilot_prefix(revision)
    runs = sorted(str(r["run_id"]) for r in completed if r.get("study") == "B" and r.get("arm") == "Moderate")
    if not prov["run_id"].startswith(prefix) or runs != [prov["run_id"]]:
        raise CliError(f"{path} measures {prov['run_id']}, but the completed Moderate run of this pilot revision "
                       f"(run_ids {prefix}*) is {runs or 'none'}")
    if not _commit_in_history(prov["commit"]):
        raise CliError(f"{path} was measured at commit {prov['commit']}, which is not in HEAD's history")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "provenance": prov}


def _cmd_surplus(args: argparse.Namespace) -> int:
    from pilot import budget

    allocated = args.hours if args.hours is not None else budget.load_allocation()["machine_hours_allocated"]
    try:
        plan = budget.surplus_plan(allocated, args.hours_per_run, args.concurrent)
    except ValueError as exc:
        raise CliError(str(exc)) from None
    print(json.dumps(budget.as_dict(plan), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    from pilot import budget

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-root", default=DEFAULT_DATA_ROOT, help="run data (default: $WSCL_DATA_ROOT or /data)")
    ledger = argparse.ArgumentParser(add_help=False)
    ledger.add_argument("--ledger-dir", help="write ledgers here instead of the repository (smoke ledgers default to "
                        "<data-root>/smoke); fixed per data root by its first run or decision")

    p = argparse.ArgumentParser(prog="python -m pilot", description="Pilot owner's tools (Role 1).")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("design", help="list the runs of a design")
    d.add_argument("name", choices=DESIGNS)
    d.add_argument("--jsonl", help="write the specs as JSON lines")
    d.add_argument("-v", "--verbose", action="store_true", help="one line per run")
    d.set_defaults(func=_cmd_design)

    b = sub.add_parser("budget", help="run-equivalents, registered and corrected")
    b.add_argument("--cuts", type=int, default=0,
                   help="price the design after the first K cuts of Part 6.1 "
                        "(a what-if; the decision is pilot/cuts.json)")
    b.set_defaults(func=_cmd_budget)

    # The smoke flags of the commands that launch runs or write ledgers; the mode is fixed per data root.
    smoke = argparse.ArgumentParser(add_help=False)
    smoke.add_argument("--allow-dirty", action="store_true",
                       help="smoke tests only: run uncommitted code "
                            "(ledgers under <data-root>/smoke unless --ledger-dir)")
    smoke.add_argument("--allow-pending", action="store_true",
                       help="smoke tests only: train runs held by open pre-registration questions and open their "
                            "result gates (ledgers under <data-root>/smoke unless --ledger-dir)")
    s = sub.add_parser("schedule", help="the run scheduler")
    ssub = s.add_subparsers(dest="action", required=True)
    sa = ssub.add_parser("add", parents=[common], help="queue the runs of a registered design as pending")
    sa.add_argument("--design", required=True, choices=DESIGNS,
                    help="the design to queue (runs already queued with the same spec are skipped)")
    sa.add_argument("--cuts", type=int, default=None,
                    help="apply the first K cuts of the committed pilot/cuts.json (Part 6.1; fixed per data root)")
    sc = ssub.add_parser("add-continuations", parents=[common],
                         help="queue the battery's fine-tuning and transfer continuations of matched rows (Table 2.2)")
    sc.add_argument("--ledger", required=True, help="the main study's ledger (results/ledger.parquet)")
    sc.add_argument("--conditions", nargs="+", choices=CONTINUATION_CHOICES, default=None,
                    help="continuations to queue (default: both)")
    su = ssub.add_parser("add-surplus", parents=[common], help="Part 5.5 surplus seeds for the primary arms")
    su.add_argument("--extra", type=int, required=True,
                    help="surplus seeds per arm (a target fixed per data root; repeat it to cover arms queued later)")
    sr = ssub.add_parser("run", parents=[common, ledger, smoke],
                         help="launch, adopt and settle runs until nothing is left to launch (long-running)")
    sr.add_argument("--max-concurrent", type=int, required=True,
                    help="runs executed at once ([P] measured in the pilot)")
    sr.add_argument("--poll", type=float, default=30.0,
                    help=f"seconds between scheduling passes, from 0 to {MAX_POLL_SECONDS:g} (default 30)")
    sr.add_argument("--on-interrupt", choices=POLICY_CHOICES, default=None,
                    help="fixed at the first run of a data root (default: restart on a registered data root once "
                         "Q-interrupted-run is answered, else hold); change with set-policy")
    sr.add_argument("--once", action="store_true", help="make one scheduling pass and return")
    st = ssub.add_parser("status", parents=[common], help="run counts by status and the latest events")
    st.add_argument("--events", type=int, default=20, help="how many of the latest events to print (default 20)")
    rs = ssub.add_parser("resolve", parents=[common, ledger, smoke],
                         help="carry out an operator action or a group decision on one run")
    rs.add_argument("run_id", help="the run to act on (as `schedule status` prints it)")
    rs.add_argument("decision", choices=RESOLVE_CHOICES,
                    help="restart: an interrupted run; reevaluate: an eval_failed run, or a ledger_failed run in no "
                         "ledger whose evaluation.json must be redone; retry-ledger: a ledger_failed "
                         "run or an unrecorded exclusion; unblock: a blocked run whose dependencies cleared; "
                         "replace: an excluded run (after diagnosing it from its logs; Q-exclusion-breaker) or a "
                         "blocked warm-started run (Table 9.1, Q-warm-start: superseded by its arm's next unused "
                         "seed); "
                         "add-auxiliary: queue the N = 0 run a pending warm-started run waits for (Q-warm-start); "
                         "requeue: store the committed design's spec for a pending or blocked run that never trained")
    rs.add_argument("--defect-fixed", metavar="ERRATUM_ID", default=None,
                    help="replace only: the id of the analysis/ERRATA.json entry, committed at HEAD, that records the "
                         "fix of the defect behind the run's exclusion; required once the arm's exclusions with that "
                         "cause reach its seed target (Table 9.1, Q-exclusion-breaker)")
    sp_ = ssub.add_parser("set-policy", parents=[common, ledger, smoke],
                          help="interruption policy for every run (Table 9.1, Q-interrupted-run)")
    sp_.add_argument("policy", choices=POLICY_CHOICES,
                     help="applied at once to every held (interrupted) run, and to every later interruption; a "
                          "registered data root runs under restart (Table 9.1, Q-interrupted-run), hold pauses it, "
                          "exclude is for smoke data roots")
    s.set_defaults(func=_cmd_schedule)

    e = sub.add_parser("enrich", parents=[common], help="write enrichment fields and supplement records (pilot/enrichment.py)")
    e.add_argument("action", choices=ENRICH_ACTIONS, help=(
        "select: rule 1 (Part 4.1); measure: C_ID of the matched checkpoint (Table 2.1); battery: hazard and "
        "dynamics gaps (eq. 1; matched rows only, rule 6); match: rules 2-6; controller: Table 2.4 quantities; "
        "training: the training record; continuations: fine-tuning and transfer gaps; zeroshot, fewshot: Study B "
        "(Table 2.5); final-battery, sensitivity-battery: rule 7 (b) and (a)"))
    e.add_argument("--ledger", required=True, help="the ledger to enrich, e.g. results/pilot/ledger.parquet")
    e.add_argument("--conditions", nargs="+", default=None, choices=BATTERY_CHOICES + CONTINUATION_CHOICES,
                   help="battery: hazard, dynamics (default: both, the pilot's battery of Part 3.6); final-battery and "
                        "sensitivity-battery: "
                        "hazard (default) and dynamics; continuations: finetune transfer (default: both)")
    e.add_argument("--surplus-extra", type=int, default=None,
                   help="match, sensitivity-battery: surplus seeds per primary-comparison arm (Part 5.5; required)")
    e.add_argument("--allow-dirty", action="store_true",
                   help="smoke ledgers only: a ledger of a smoke --data-root (never the repository's results/)")
    e.add_argument("--allow-pending", action="store_true",
                   help="smoke ledgers only, with --allow-dirty: open the result gates of open questions")
    e.set_defaults(func=_cmd_enrich)

    cp = sub.add_parser("completeness", parents=[common], help="what is still missing, per arm (read-only)")
    cp.add_argument("--ledger", default=str(REPO_ROOT / "results" / "ledger.parquet"),
                    help="the ledger (default: the main one)")
    cp.add_argument("--json", action="store_true", help="print the report as JSON")
    cp.set_defaults(func=_cmd_completeness)

    fz = sub.add_parser("freeze", parents=[common], help="the data freeze manifest (read-only; Part 5.8)")
    fz.add_argument("--ledger", default=str(REPO_ROOT / "results" / "ledger.parquet"),
                    help="the ledger (default: the main one)")
    fz.add_argument("--json", action="store_true", help="print the manifest as JSON")
    fz.set_defaults(func=_cmd_freeze)

    g4 = sub.add_parser("g4-measure", parents=[common], help="Moderate pilot run at its training budgets")
    g4.add_argument("--revision", type=int, choices=(0, 1), default=0, help="0: the pilot; 1: the re-pilot (Part 6)")
    g4.add_argument("--out", help="written once, never overwritten (default: results/pilot/g4_moderate[-rev1].json)")
    g4.set_defaults(func=_cmd_g4_measure)

    a = sub.add_parser("allocate", help="record the machine-hour allocation, before any workstation timing exists")
    a.add_argument("--hours", type=float, required=True,
                   help="H, the machine-hours allocated to the registered runs of the whole design (Part 6 G2; Table "
                        "8.1), in workstation wall-clock hours (all concurrent runs together): elapsed hours of the "
                        "machine with all its run slots, not core-hours and not hours per run")
    a.add_argument("--basis", required=True,
                   help="how H was derived, from the calendar and the machine's availability only (e.g. 24 x the days "
                        "from the planned start of the main sweep to the last date runs may finish x the fraction of "
                        "time the workstation is available), never from a timing of a run on the workstation")
    a.add_argument("--by", required=True, help="who decided (the group)")
    a.add_argument("--date", required=True, help="meeting date, YYYY-MM-DD")
    a.add_argument("--confirm", action="store_true",
                   help=f"record an H above {budget.ALLOCATION_CONFIRM_ABOVE} hours (two years of elapsed time; "
                        "pilot.budget.ALLOCATION_CONFIRM_ABOVE), otherwise refused as likely core-hours")
    a.add_argument("--smoke-root", action="append", default=None, metavar="DIR",
                   help="a smoke data root used on the workstation (scripts/smoke_run.py --data-root); its ledgers, "
                        "its sidecar directory when it holds records and the train_result.json of every run trained "
                        "there are listed in the record's prior_workstation_timings (repeatable)")
    a.set_defaults(func=_cmd_allocate)

    g = sub.add_parser("go", help="go conditions G1 to G4 (Part 6) and Table 8.1 (from a clean commit)")
    g.add_argument("--concurrent", type=int,
                   help="runs sustained concurrently (default: measured from the pilot runs)")
    g.add_argument("--moderate", help="the g4-measure output (default: results/pilot/g4_moderate[-rev1].json)")
    g.add_argument("--out-dir", default=str(PILOT_RESULTS), help="where the timestamped report is written")
    g.add_argument(
        "--repilot", nargs="+", choices=("G1", "G2", "G4"), default=(), metavar="G",
        help="read the re-pilot (run_ids P1-...) after the one revision; "
             "name the conditions that failed on the first pilot",
    )
    g.set_defaults(func=_cmd_go)

    sp = sub.add_parser("surplus", help="Part 5.5 surplus seeds")
    sp.add_argument("--hours-per-run", type=float, required=True, help="measured wall-clock hours per 10M-step run")
    sp.add_argument("--concurrent", type=int, required=True, help="runs sustained concurrently (Table 8.1)")
    sp.add_argument("--hours", type=float, help="allocated machine-hours (default: pilot/allocation.json)")
    sp.set_defaults(func=_cmd_surplus)
    return p


def _harness_failure(exc: BaseException) -> bool:
    """A failure another role's harness reports for the operator, not a bug: MuJoCo's unstable simulation
    (``envs.evaluation.MujocoInstabilityError``, a RuntimeError). Q-mujoco-exception (Table 9.1): the
    harness replaces unstable episodes by reserve episodes and fails an evaluation call only when more than
    ``pilot.contracts.MAX_UNSTABLE_EPISODES`` of its episodes are unstable; evaluation is deterministic, so it fails the
    same way again, and the evaluation is held for the group (not a Part 5.6 exclusion: the run is kept).
    Matched only when the harness was imported (it raised it), so the command line never imports it for this."""
    evaluation = sys.modules.get("envs.evaluation")
    unstable = getattr(evaluation, "MujocoInstabilityError", None)
    return isinstance(unstable, type) and isinstance(exc, unstable)


def main(argv: list[str] | None = None) -> int:
    from pilot.algorithms import PluginUnavailableError, is_missing_repository_import
    from pilot.contracts import ContractError
    from pilot.errors import RunRefused
    from pilot.ledger_writer import LedgerWriteError
    from pilot.provenance import DirtyWorktreeError
    from pilot.scheduler import SchedulerError
    from pilot.supplement import SupplementError

    args = build_parser().parse_args(argv)
    args.argv = list(argv) if argv is not None else sys.argv[1:]  # the command line the go report records
    try:
        return int(args.func(args) or 0)
    except CliError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except PluginUnavailableError as exc:
        print(f"unavailable: {exc}", file=sys.stderr)
        return EXIT_FAILURE
    except RunRefused as exc:  # PendingQuestionError included (it is also a ValueError: caught first)
        print(f"refused: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    except ImportError as exc:
        # another role's module (or a name of one) imported lazily inside a harness call, not written yet:
        # unavailable, as the launcher classifies it (HANDOVER.md section 8; pilot/launch.py); a missing library keeps its traceback
        if not is_missing_repository_import(exc):
            raise
        print(f"unavailable: another role's code imported {exc.name!r}, which is not in the repository yet: {exc}",
              file=sys.stderr)
        return EXIT_FAILURE
    except subprocess.CalledProcessError as exc:  # git, run by provenance (no commits yet, dubious ownership, ...)
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr or ""
        command = " ".join(map(str, exc.cmd)) if isinstance(exc.cmd, (list, tuple)) else str(exc.cmd)
        print(f"error: git failed: `{command}` exited with {exc.returncode}: {stderr.strip() or '(no message)'}",
              file=sys.stderr)
        return EXIT_FAILURE
    # Expected failures (bad values, missing files, refused writes, a scheduler state that cannot be read
    # or stays locked) are reported without a traceback.
    except (ValueError, KeyError, OSError, ContractError, LedgerWriteError, DirtyWorktreeError, SchedulerError,
            SupplementError, sqlite3.Error) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_FAILURE
    except RuntimeError as exc:
        if not _harness_failure(exc):
            raise  # an unexpected failure keeps its traceback
        print(f"error: {type(exc).__name__}: {exc} (Q-mujoco-exception: the evaluation is held for the group, report "
              "it to the group; evaluation is deterministic, so a retry fails the same way; the run is kept, an "
              "evaluation failure is not a Part 5.6 exclusion)", file=sys.stderr)
        return EXIT_FAILURE
    except KeyboardInterrupt:  # Ctrl-C, e.g. stopping the long-running `schedule run`: no traceback
        print("interrupted", file=sys.stderr)
        return EXIT_INTERRUPTED


if __name__ == "__main__":
    sys.exit(main())

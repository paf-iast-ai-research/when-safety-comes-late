"""Command line of the pilot owner's tools. Run ``python -m pilot --help`` from the repository root.

Commands
  design      list the runs of a design (pilot, repilot, main, treatment, controller, pid, study_a, study_b, study_b_fewshot, all)
  budget      run-equivalents: registered (Part 6 G2) and corrected
  schedule    add | add-surplus | run | status | resolve | set-policy   (the run scheduler; pilot/scheduler.py)
  enrich      select | battery               (rule-1 selection and the battery, written to the ledger)
  g4-measure  evaluate the pilot Moderate run at its training budgets (Part 3.6; G4)
  allocate    record the machine-hour allocation (before any pilot run is launched)
  go          compute G1 to G4 and the Table 8.1 values from the pilot
  surplus     the surplus-seed rule of Part 5.5
"""

from __future__ import annotations

import argparse
import json
import math
import numbers
import os
import sys
from datetime import datetime
from pathlib import Path

from configs import registered as R
from pilot.provenance import REPO_ROOT

PILOT_RESULTS = REPO_ROOT / "results" / "pilot"
DESIGNS = ("pilot", "repilot", "main", "treatment", "controller", "pid", "study_a", "study_b", "study_b_fewshot", "all")
DEFAULT_DATA_ROOT = os.environ.get("WSCL_DATA_ROOT", "/data")


class CliError(Exception):
    """A usage error reported without a traceback."""


def _ledger_paths(args: argparse.Namespace):
    from pilot.ledger_writer import LedgerPaths

    if not getattr(args, "ledger_dir", None):
        return None
    base = Path(args.ledger_dir)
    return LedgerPaths(main=base / "ledger.parquet", pilot=base / "pilot_ledger.parquet", sidecar_dir=base / "sidecar")


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
        print(f"{s.run_id:60s} steps={s.total_steps:>11,d} onset={s.onset_step} open={','.join(s.open_questions) or '-'}")
    print(f"{args.name}: {len(specs)} runs, {total:.2f} run-equivalents of training, {len(waiting)} waiting on open questions")
    return 0


def _cmd_budget(args: argparse.Namespace) -> int:
    from pilot import budget

    print(f"registered (Part 6 G2): {budget.registered_run_equivalents()} run-equivalents")
    for group, value in budget.corrected_run_equivalents().items():
        print(f"corrected  {group:22s} {value:8.2f}")
    return 0


def _scheduler(args: argparse.Namespace):
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
        raise CliError(f"no scheduler state at {cfg.state_path}; check --data-root (only add, add-surplus and run create it)")
    return Scheduler(cfg)


def _cmd_schedule(args: argparse.Namespace) -> int:
    from pilot import manifest

    sched = _scheduler(args)
    if args.action in ("add", "add-surplus") and sched.setting("mode") == "smoke":
        # registered runs queued here would train in smoke mode, without the allocation gate
        raise CliError(f"{sched.cfg.data_root} is a smoke data root; queue registered runs on another data root")
    if args.action in ("add", "add-surplus"):
        # They queue registered runs only, so they fix the data root as registered: a later command with
        # --allow-dirty/--allow-pending is then refused instead of training these runs in smoke mode.
        sched.fix_mode(ledgers=False)  # the ledger location stays with the first run or decision
    if args.action == "add":
        specs = manifest.design(args.design)
        print(f"added {sched.add(specs)} of {len(specs)} runs from design {args.design!r}")
    elif args.action == "add-surplus":
        added = sched.add_surplus(args.extra)
        print(f"added {len(added)} surplus training runs (Part 5.5): {', '.join(added) or '-'}")
    elif args.action == "run":
        sched.run(once=args.once)
    elif args.action == "status":
        for status, count in sorted(sched.summary().items()):
            print(f"{status:24s} {count}")
        for ev in sched.events(limit=args.events):
            print(f"{ev['ts']}  {ev['run_id'] or '-':60s} {ev['event']:24s} {ev['detail'] or ''}")
    elif args.action == "resolve":
        try:
            sched.resolve(args.run_id, args.decision)
        except KeyError:
            raise CliError(f"unknown run_id {args.run_id!r}") from None
        print(f"{args.run_id}: {args.decision}")
    elif args.action == "set-policy":
        sched.set_policy(args.policy)
        print(f"interruption policy is now {args.policy!r} for every run on {args.data_root}")
    return 0


def _cmd_enrich(args: argparse.Namespace) -> int:
    from pilot import enrichment, provenance
    from pilot.scheduler import SchedulerConfig

    ledger = Path(args.ledger)
    if args.allow_dirty and _in_repository_results(ledger):
        raise CliError(f"--allow-dirty is for smoke ledgers only; {ledger} is in the repository's results/")
    if not args.allow_dirty and provenance.dirty_paths():
        raise CliError("tracked code has uncommitted changes; commit before enriching the ledger (the log records the commit)")

    def check_code() -> None:
        # The selector and the harness are imported lazily: check after they ran, before any write.
        unverified = [] if args.allow_dirty else provenance.unverified_imported_code()
        if unverified:
            raise CliError("the enrichment ran code that the commit does not contain: " + ", ".join(unverified))

    if args.action == "select":
        done = enrichment.apply_selection(ledger, check_code=check_code)
    else:
        cfg = SchedulerConfig(data_root=Path(args.data_root))
        done = enrichment.apply_battery(ledger, cfg.run_dir, args.conditions, check_code=check_code,
                                        data_root=args.data_root)
    print(f"{args.action}: {len(done)} runs enriched")
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
    result = plain(dict(evaluate(str(run_dir / train["omnisafe_dir"]), spec.to_dict())))
    unverified = provenance.unverified_imported_code()
    if unverified:
        raise CliError("the evaluation ran code that the commit does not contain: " + ", ".join(unverified))
    if int(result.get("episodes", -1)) != R.EVAL_EPISODES:
        raise CliError(f"the Moderate evaluation must use {R.EVAL_EPISODES} episodes per budget")
    levels = sorted(float(b) for b in spec.training_levels or ())
    costs = _numbers_by_budget(result.get("mean_cost"))
    if sorted(costs) != levels or not all(math.isfinite(v) for v in costs.values()):
        raise CliError(f"the Moderate evaluation must give a finite mean_cost for each training budget {levels} "
                       f"(Part 6 G4; HANDOVER.md section 8), got {result.get('mean_cost')!r}")
    rates = _numbers_by_budget(result.get("satisfaction"))
    if sorted(rates) != levels or not all(0.0 <= v <= 1.0 for v in rates.values()):
        raise CliError(f"the Moderate evaluation must give a satisfaction rate in [0, 1] for each training budget "
                       f"{levels} (Part 3.6; HANDOVER.md section 8), got {result.get('satisfaction')!r}")
    result["mean_cost"], result["satisfaction"] = costs, rates  # the checked numbers, not the harness's objects
    result["provenance"] = {"run_id": spec.run_id, "commit": commit, "utc": provenance.utc_now().isoformat(),
                            "data_root": os.path.abspath(args.data_root)}
    _write_once(out, json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", "the G4 measurement is taken once")
    print(f"wrote {out}")
    return 0


def _numbers_by_budget(mapping: object) -> dict[float, float]:
    """``{budget: value}`` as floats, or ``{}`` (refused by the caller) if any value is not a real
    number: a bool, a string or None never passes as a cost or a rate."""
    try:
        items = dict(mapping or {}).items()  # type: ignore[call-overload]
        if not all(isinstance(v, numbers.Real) and not isinstance(v, bool) for _, v in items):
            return {}
        return {float(k): float(v) for k, v in items}
    except (TypeError, ValueError):
        return {}


def _require_clean_tree(purpose: str) -> str:
    """HEAD's hash; a CliError (exit 2) if tracked code has uncommitted changes."""
    from pilot import provenance

    try:
        return provenance.require_clean_worktree()
    except provenance.DirtyWorktreeError:
        raise CliError(f"tracked code has uncommitted changes; commit before {purpose}: "
                       + ", ".join(provenance.dirty_paths())) from None


def _write_once(path: Path, text: str, why: str) -> None:
    """Create ``path`` with ``text`` atomically and only once: a crash never leaves a partial file."""
    from pilot.ledger_writer import _fsync_dir

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        try:
            os.link(tmp, path)  # atomic, and fails if the file exists
        except FileExistsError:
            raise CliError(f"{path} exists; {why}") from None
        _fsync_dir(path.parent)
    finally:
        tmp.unlink(missing_ok=True)


def _cmd_allocate(args: argparse.Namespace) -> int:
    from pilot import budget

    record = budget.record_allocation(args.hours, args.by, args.date)
    print(json.dumps(record, indent=2))
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


def _as_datetime(value) -> datetime:
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
    records = _load_pilot_records(DEFAULT_PATHS.pilot, DEFAULT_PATHS.sidecar_dir, revision)
    ledger_rows = [r for r in records if "reason" not in r]
    completed = [r for r in records if r.get("completed")]
    intervals = [(_as_datetime(r["started"]), _as_datetime(r["finished"])) for r in completed if r.get("started") and r.get("finished")]
    measured = go_decision.measured_concurrency(intervals)
    concurrent = args.concurrent if args.concurrent is not None else int(measured["max"])
    if concurrent < 1:
        raise CliError("no completed pilot run: concurrency cannot be measured yet")
    if concurrent > measured["max"]:
        raise CliError(f"--concurrent {concurrent} exceeds the {int(measured['max'])} runs the pilot actually ran at once")
    allocation_ok, note = budget.allocation_contained(records)
    allocated = budget.load_allocation()["machine_hours_allocated"] if budget.ALLOCATION_FILE.exists() else None
    unconstrained_runs = [r for r in completed if r.get("arm") == "unconstrained"]
    unconstrained = unconstrained_runs[0].get("final_cost") if len(unconstrained_runs) == 1 else None
    moderate_path = Path(args.moderate) if args.moderate else _g4_path(revision)
    moderate = json.loads(moderate_path.read_text(encoding="utf-8")) if moderate_path.exists() else None
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
    )
    unverified = provenance.unverified_imported_code()
    if unverified:
        raise CliError("the go decision ran code that the commit does not contain: " + ", ".join(unverified))
    result["generated"] = {"utc": provenance.utc_now().isoformat(), "commit": commit,
                           "worktree_dirty": bool(provenance.dirty_paths()), "argv": sys.argv[1:]}
    stamp = provenance.utc_now().strftime("%Y%m%dT%H%M%SZ")
    js, md = go_decision.write_report(result, Path(args.out_dir), stamp)
    print(Path(md).read_text(encoding="utf-8"))
    print(f"wrote {js} and {md}")
    return 0


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
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-root", default=DEFAULT_DATA_ROOT, help="run data (default: $WSCL_DATA_ROOT or /data)")
    ledger = argparse.ArgumentParser(add_help=False)
    ledger.add_argument("--ledger-dir", help="write ledgers here instead of the repository (smoke tests); "
                        "fixed per data root by its first run or decision")

    p = argparse.ArgumentParser(prog="python -m pilot", description="Pilot owner's tools (Role 1).")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("design", help="list the runs of a design")
    d.add_argument("name", choices=DESIGNS)
    d.add_argument("--jsonl", help="write the specs as JSON lines")
    d.add_argument("-v", "--verbose", action="store_true", help="one line per run")
    d.set_defaults(func=_cmd_design)

    b = sub.add_parser("budget", help="run-equivalents, registered and corrected")
    b.set_defaults(func=_cmd_budget)

    # The smoke flags of the commands that launch runs or write ledgers; the mode is fixed per data root.
    smoke = argparse.ArgumentParser(add_help=False)
    smoke.add_argument("--allow-dirty", action="store_true", help="smoke tests only (ledgers go under <data-root>/smoke)")
    smoke.add_argument("--allow-pending", action="store_true", help="smoke tests only (ledgers go under <data-root>/smoke)")
    s = sub.add_parser("schedule", help="the run scheduler")
    ssub = s.add_subparsers(dest="action", required=True)
    sa = ssub.add_parser("add", parents=[common], help="queue the runs of a registered design as pending")
    sa.add_argument("--design", required=True, choices=DESIGNS,
                    help="the design to queue (runs already queued with the same spec are skipped)")
    su = ssub.add_parser("add-surplus", parents=[common], help="Part 5.5 surplus seeds for the primary arms")
    su.add_argument("--extra", type=int, required=True,
                    help="surplus seeds per arm (a target fixed per data root; repeat it to cover arms queued later)")
    sr = ssub.add_parser("run", parents=[common, ledger, smoke],
                         help="launch, adopt and settle runs until nothing is left to launch (long-running)")
    sr.add_argument("--max-concurrent", type=int, required=True, help="runs executed at once ([P] measured in the pilot)")
    sr.add_argument("--poll", type=float, default=30.0, help="seconds between scheduling passes (default 30)")
    sr.add_argument("--on-interrupt", choices=("hold", "restart", "exclude"), default=None,
                    help="fixed at the first run of a data root (default hold); change with set-policy")
    sr.add_argument("--once", action="store_true", help="make one scheduling pass and return")
    st = ssub.add_parser("status", parents=[common], help="run counts by status and the latest events")
    st.add_argument("--events", type=int, default=20, help="how many of the latest events to print (default 20)")
    rs = ssub.add_parser("resolve", parents=[common, ledger, smoke],
                         help="carry out an operator action or a group decision on one run")
    rs.add_argument("run_id", help="the run to act on (as `schedule status` prints it)")
    rs.add_argument("decision", choices=("restart", "reevaluate", "retry-ledger", "unblock", "replace"),
                    help="restart: an interrupted run; reevaluate: an eval_failed run; retry-ledger: a ledger_failed "
                         "run or an unrecorded exclusion; unblock / replace: a blocked or excluded run (group decision)")
    sp_ = ssub.add_parser("set-policy", parents=[common, ledger, smoke],
                          help="interruption policy for every run (a group decision)")
    sp_.add_argument("policy", choices=("hold", "restart", "exclude"),
                     help="applied at once to every held (interrupted) run, and to every later interruption")
    s.set_defaults(func=_cmd_schedule)

    e = sub.add_parser("enrich", parents=[common], help="rule-1 selection and battery gaps")
    e.add_argument("action", choices=("select", "battery"), help="select: rule 1 (Part 4.1); battery: robustness gaps (eq. 1)")
    e.add_argument("--ledger", required=True, help="the ledger to enrich, e.g. results/pilot/ledger.parquet")
    e.add_argument("--conditions", nargs="+", default=list(R.PILOT_BATTERY), choices=("hazard", "dynamics"),
                   help="battery conditions evaluated here (default: the pilot's)")
    e.add_argument("--allow-dirty", action="store_true", help="smoke ledgers only (refused for the repository's results/)")
    e.set_defaults(func=_cmd_enrich)

    g4 = sub.add_parser("g4-measure", parents=[common], help="Moderate pilot run at its training budgets")
    g4.add_argument("--revision", type=int, choices=(0, 1), default=0, help="0: the pilot; 1: the re-pilot (Part 6)")
    g4.add_argument("--out", help="written once, never overwritten (default: results/pilot/g4_moderate[-rev1].json)")
    g4.set_defaults(func=_cmd_g4_measure)

    a = sub.add_parser("allocate", help="record the machine-hour allocation")
    a.add_argument("--hours", type=float, required=True)
    a.add_argument("--by", required=True, help="who decided (the group)")
    a.add_argument("--date", required=True, help="meeting date, YYYY-MM-DD")
    a.set_defaults(func=_cmd_allocate)

    g = sub.add_parser("go", help="G1 to G4 and Table 8.1 (from a clean commit)")
    g.add_argument("--concurrent", type=int, help="runs sustained concurrently (default: measured from the pilot runs)")
    g.add_argument("--moderate", help="the g4-measure output (default: results/pilot/g4_moderate[-rev1].json)")
    g.add_argument("--out-dir", default=str(PILOT_RESULTS), help="where the timestamped report is written")
    g.add_argument(
        "--repilot", nargs="+", choices=("G1", "G2", "G4"), default=(), metavar="G",
        help="read the re-pilot (run_ids P1-...) after the one revision; name the conditions that failed on the first pilot",
    )
    g.set_defaults(func=_cmd_go)

    sp = sub.add_parser("surplus", help="Part 5.5 surplus seeds")
    sp.add_argument("--hours-per-run", type=float, required=True, help="measured wall-clock hours per 10M-step run")
    sp.add_argument("--concurrent", type=int, required=True, help="runs sustained concurrently (Table 8.1)")
    sp.add_argument("--hours", type=float, help="allocated machine-hours (default: pilot/allocation.json)")
    sp.set_defaults(func=_cmd_surplus)
    return p


def main(argv: list[str] | None = None) -> int:
    from pilot.algorithms import PluginUnavailableError
    from pilot.contracts import ContractError
    from pilot.ledger_writer import LedgerWriteError
    from pilot.provenance import DirtyWorktreeError
    from pilot.scheduler import SchedulerError

    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except CliError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except PluginUnavailableError as exc:
        print(f"unavailable: {exc}", file=sys.stderr)
        return 1
    # Expected failures (bad values, missing files, refused writes) are reported without a traceback.
    except (ValueError, KeyError, OSError, ContractError, LedgerWriteError, DirtyWorktreeError, SchedulerError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

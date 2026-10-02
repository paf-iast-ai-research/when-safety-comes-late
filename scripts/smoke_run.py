"""Smoke test of the whole pipeline on short runs (HANDOVER.md, task 18; not analysed).

    python scripts/smoke_run.py --data-root /scratch/smoke                          # one 1-epoch PPO-Lagrangian run
    python scripts/smoke_run.py --data-root /scratch/smoke --design sample          # 2 epochs: through the onsets
    python scripts/smoke_run.py --data-root /scratch/smoke --design pilot --epochs 2

Queues short copies of registered runs (``--design``) with ``total_steps = epochs x 20,000`` and
runs them through the real scheduler and launcher, with the real plug-ins, in smoke mode
(``allow_dirty`` and ``allow_pending``: the launcher gets ``--allow-dirty --allow-pending`` in both
stages, so open questions neither hold the runs nor refuse their results): the ledgers go under
DATA_ROOT/smoke, never to the repository, and the data root is fixed as a smoke root. The command
line cannot queue short runs (``schedule add`` takes only the registered designs), which is why
this script exists. Designs:

  determinism  one PPO-Lagrangian run of OmniSafe unchanged (plug-in ppolag; the default)
  pilot        the eight runs of the pilot (Part 3.6)
  sample       one run of each onset or budget plug-in on SafetyPointGoal1-v0, seed 0: the N = 0
               main-sweep arm (study_a), the N = 0.50 abrupt total-steps arm with partial reset
               (study_a with a treatment), the N = 0.50 arm of the PID check (study_a_pid), and the
               Moderate arm of Study B (study_b)

A late onset is moved into the short run (``short``). The sample design exists to pass through the
plug-ins' onset code (the onset switch, the partial reset at onset, the PID controller after onset),
so it runs 2 epochs by default and refuses fewer (MIN_EPOCHS): in a 1-epoch run every late onset
moves to step 0 and the reset arm is refused by its plug-in. Another design run for 1 epoch with a
late onset prints a note that no onset is exercised. Runs whose plug-in or evaluation harness is
not in the repository yet wait, as in a real sweep (``plugin_unavailable``); the script prints the
scheduler's status and its last events. Evaluation needs ten checkpoints in the final 2,000,000
steps, as in a registered run (``evaluable``: at least 90 epochs; a total off the 200,000-step grid
uses the end-relative window the run also saves), so a shorter run, such as one of 1 to 3 epochs, is
a training smoke test: the scheduler still launches its evaluation, which ends eval_failed or refused
(the checkpoints are missing), or leaves the run trained while the harness is not in the repository.
The evaluate -> ledger -> enrich chain is tested with tests/fake_launcher.py.

On the workstation, the allocation is recorded first (X-allocation, HANDOVER task 12): a smoke run
there reveals the machine's speed, and ``python -m pilot allocate --smoke-root DATA_ROOT`` lists its
timings in the record.

Never on a registered data root: the script refuses /data (the only registered data root, HANDOVER.md
section 6: "`WSCL_DATA_ROOT` or `--data-root` point elsewhere only for smoke tests") and any directory
inside it, a data root whose mode is fixed as registered, and one that holds runs it did not queue
(for example runs queued by ``schedule add``).

Exit status (so that a runbook or CI step can tell a failing smoke test from a passing one): 0 when
every run ended as a smoke test expects: ledgered or continued, waiting for a plug-in or an evaluation
harness not yet in the repository (plugin_unavailable, evaluator_unavailable), or, in a run too short
to be evaluated, eval_failed or refused at evaluation; EXIT_RUNS_FAILED (3) when any run ended
otherwise (``failed_runs``), for example excluded, interrupted, blocked or ledger_failed, pending
after its launcher refused or failed, or eval_failed in a run long enough to be evaluated; 1 when the
data root is refused; 2 on a usage error (argparse; for example --epochs below the design's minimum).

Owner: pilot owner (Role 1).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from configs import registered as R  # noqa: E402
from pilot import contracts, errors, manifest  # noqa: E402
from pilot.manifest import RunSpec  # noqa: E402
from pilot import scheduler  # noqa: E402
from pilot.scheduler import Scheduler, SchedulerConfig, SchedulerError, determinism_spec  # noqa: E402

DESIGNS = ("determinism", "pilot", "sample")
# Epochs per run: the default and the least each design accepts. The sample design needs a late onset
# after step 0, which ``short`` places at epoch 1 of a 2-epoch run (the defaults are 1 or 2 epochs).
DEFAULT_EPOCHS = {"determinism": 1, "pilot": 1, "sample": 2}
MIN_EPOCHS = {"determinism": 1, "pilot": 1, "sample": 2}
RESET = "reset"  # one of R.TREATMENTS: Table 2.4 "Partial reset"
MODERATE = "Moderate"  # one of R.STUDY_B_ARMS (Table 3.4): Study B's arm in the sample design
SMOKE_PREFIX = "SMOKE-"
# Registered runs keep their data root at /data (HANDOVER.md section 6: the ledger stores checkpoint paths
# relative to the schema's /data/checkpoints); $WSCL_DATA_ROOT names a smoke root when it points elsewhere,
# so it is not refused. A registered root elsewhere is still refused by its stored mode and by the runs it
# holds (``main``).
REGISTERED_DATA_ROOTS = scheduler.REGISTERED_DATA_ROOTS  # SchedulerConfig refuses smoke mode on them too
EXIT_REFUSED_ROOT = 1
EXIT_RUNS_FAILED = 3
# 90: the fewest epochs whose final 2,000,000 steps hold ten checkpoints (for the messages; ``evaluable`` decides)
EVALUATION_EPOCHS = (R.SELECTION_WINDOW_CHECKPOINTS - 1) * R.CHECKPOINT_INTERVAL_EPOCHS
# How a run may end in a smoke test that passed (``failed_runs``): the terminal states, a run waiting for
# a plug-in or harness Roles 2 to 5 have not committed yet, and (only for runs too short to be evaluated) the
# evaluation stage's refusal or failure (their checkpoints are missing).
PASSED_STATUSES = ("ledgered", "continued")
WAITING_FOR_CODE = ("plugin_unavailable", "evaluation: evaluator_unavailable")
SHORT_RUN_EVALUATION = ("evaluation: launch_refused",)


def short(spec: RunSpec, epochs: int) -> RunSpec:
    """The same run with ``epochs`` epochs, a SMOKE- run_id, no dependencies, and a late onset moved inside.

    A late onset later than half the run moves to the last checkpoint at or before half the run
    (from 20 epochs on, it stays on the checkpoint grid, so it adds no off-grid checkpoint to the
    selection window: those are Q-selection-window's case, not a smoke test's); in a run too short
    for that (under 20 epochs) it moves to half the run in whole epochs, so the plug-in's onset (and
    a treatment's intervention) still happens during the smoke test. In a 1-epoch run there is no room for an
    onset: the arm trains from step 0, as N = 0 does, and a partial-reset or injection arm (Table
    2.4: "At onset, ...") is refused by its plug-in; use 2 or 3 epochs to exercise it.
    """
    steps = epochs * R.STEPS_PER_EPOCH
    onset = spec.onset_step
    if onset:
        on_grid = steps // 2 // R.CHECKPOINT_INTERVAL_STEPS * R.CHECKPOINT_INTERVAL_STEPS
        onset = min(onset, on_grid or epochs // 2 * R.STEPS_PER_EPOCH)
    return replace(spec, run_id=SMOKE_PREFIX + spec.run_id, total_steps=steps, onset_step=onset, depends_on=())


def sample_design() -> list[RunSpec]:
    """One run of each onset or budget plug-in (study_a, study_a with a treatment, study_a_pid, study_b), seed 0 on
    SafetyPointGoal1-v0."""
    seed, task, late = R.SEEDS[0], R.PRIMARY_TASK, max(R.LATE_ONSET_FRACTIONS)
    return [
        next(s for s in manifest.study_a_main((seed,)) if s.task == task and s.N == R.ONSET_FRACTIONS[0]),
        next(s for s in manifest.study_a_treatments((seed,)) if s.task == task and s.N == late and s.treatment == RESET),
        next(s for s in manifest.study_a_pid((seed,)) if s.task == task and s.N == late),
        next(s for s in manifest.study_b((seed,)) if s.arm == MODERATE),
    ]


def base_specs(name: str) -> list[RunSpec]:
    """The registered runs a smoke design copies (before ``short``)."""
    if name == "determinism":
        return [determinism_spec("ppolag", R.STEPS_PER_EPOCH)]
    if name == "pilot":
        return manifest.design("pilot")
    if name == "sample":
        return sample_design()
    raise ValueError(f"unknown design {name!r}; one of {DESIGNS}")


def design_specs(name: str, epochs: int) -> list[RunSpec]:
    """The smoke copies (``short``) of a design's registered runs."""
    return [short(s, epochs) for s in base_specs(name)]


def onsets_lost(name: str, epochs: int) -> list[str]:
    """The smoke copies of late-onset runs that ``short`` moves to step 0 (no onset is exercised)."""
    return [copy.run_id for base, copy in zip(base_specs(name), design_specs(name, epochs))
            if base.onset_step and not copy.onset_step]


def registered_root(data_root: Path) -> str | None:
    """The registered data root that ``data_root`` is, or lies in (None if neither)."""
    return scheduler.registered_root(data_root, REGISTERED_DATA_ROOTS)


def evaluable(spec: RunSpec) -> bool:
    """Whether the run saves the checkpoints its evaluation needs: the selection window of contract 1.

    The rule of the evaluation harnesses (``pilot.contracts.selection_window`` over
    ``pilot.contracts.expected_checkpoint_steps``), with pending questions allowed as in smoke mode: ten
    checkpoints in the final 2,000,000 steps, so at least 1,800,000 steps (90 epochs); a total off the
    200,000-step grid uses the end-relative window the run also saves (Q-selection-window). A
    continuation is evaluated at its horizons, not by this window, and counts as evaluable.
    """
    if spec.group in manifest.CONTINUATION_GROUPS:
        return True
    try:
        with errors.allow_pending():
            contracts.selection_window(contracts.expected_checkpoint_steps(spec), spec.total_steps)
    except contracts.ContractError:
        return False
    return True


def plural(count: int, one: str, many: str) -> str:
    """``one`` for a count of 1, else ``many`` (for the messages)."""
    return one if count == 1 else many


def failed_runs(sched: Scheduler) -> list[str]:
    """Every run whose end fails the smoke test, as "run_id (status[, reason])" (the exit status's rule).

    The reasons are those ``schedule status`` shows (``Scheduler.status_report``). A run passes when it
    ended ledgered or continued, waits for a plug-in or an evaluation harness not in the repository
    yet, or, when it is too short to be evaluated (``evaluable``), ended eval_failed or refused at
    evaluation.
    """
    waiting = {run: reason for reason, runs in sched.status_report()["waiting"].items() for run in runs}
    failed = []
    for row in sched.rows():
        status, reason = row["status"], waiting.get(row["run_id"])
        if (status in PASSED_STATUSES or reason in WAITING_FOR_CODE
                or (not evaluable(sched.spec(row["run_id"]))
                    and (status == "eval_failed" or reason in SHORT_RUN_EVALUATION))):
            continue
        failed.append(f"{row['run_id']} ({status}{', ' + reason if reason else ''})")
    return failed


def main(argv: list[str] | None = None) -> int:
    """Queue the smoke runs, run them through the scheduler, and return the exit status (module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-root", type=Path, required=True, help="a scratch directory outside the repository")
    parser.add_argument("--design", choices=DESIGNS, default="determinism",
                        help="the runs to queue (default determinism)")
    defaults = "; ".join(f"{d} {n}" for d, n in DEFAULT_EPOCHS.items())
    parser.add_argument("--epochs", type=int, default=None,
                        help=f"epochs per run ({R.STEPS_PER_EPOCH:,} steps each; default by design: {defaults}; at "
                             f"least {EVALUATION_EPOCHS} to reach evaluation)")
    parser.add_argument("--max-concurrent", type=int, default=1,
                        help="the most runs trained at once (default 1)")
    args = parser.parse_args(argv)
    if args.epochs is None:
        args.epochs = DEFAULT_EPOCHS[args.design]
    if args.epochs < MIN_EPOCHS[args.design]:
        parser.error(f"--epochs must be at least {MIN_EPOCHS[args.design]} for the {args.design} design"
                     + (" (its late onsets, the partial reset and the PID controller act after step 0)"
                        if args.design == "sample" else ""))
    registered = registered_root(args.data_root)
    if registered is not None:
        print(f"error: {args.data_root} is (or lies in) the registered data root {registered}; "
              "use a scratch data root for smoke tests", file=sys.stderr)
        return EXIT_REFUSED_ROOT

    specs = design_specs(args.design, args.epochs)
    collapsed = onsets_lost(args.design, args.epochs)
    if collapsed:
        n = len(collapsed)
        print(f"note: {n} {plural(n, 'run has no room for its', 'runs have no room for their')} late onset in "
              f"{args.epochs} epoch(s) and {plural(n, 'trains', 'train')} from step 0 (e.g. {collapsed[0]}); no onset "
              "is exercised. Use --epochs 2 or 3 to exercise it")
    short_runs = [s.run_id for s in specs if not evaluable(s)]
    if short_runs:
        print(f"note: {len(short_runs)} {plural(len(short_runs), 'run is', 'runs are')} too short to be evaluated "
              f"(evaluation needs the ten checkpoints of the final {R.SELECTION_WINDOW_STEPS:,} steps: at least "
              f"{EVALUATION_EPOCHS} epochs); "
              "the evaluation of these runs ends eval_failed or refused (the checkpoints are missing), or leaves "
              "them trained while the evaluation harness is not in the repository")
    cfg = SchedulerConfig(data_root=args.data_root, max_concurrent=args.max_concurrent, poll_seconds=5.0,
                          allow_dirty=True, allow_pending=True, require_allocation_for_pilot=False)
    sched = Scheduler(cfg)
    foreign = [r["run_id"] for r in sched.rows() if not r["run_id"].startswith(SMOKE_PREFIX)]
    if foreign:  # registered runs queued by `schedule add` would otherwise train here in smoke mode
        print(f"error: {cfg.data_root} holds {len(foreign)} registered {plural(len(foreign), 'run', 'runs')} "
              f"(e.g. {foreign[0]}); "
              "use a scratch data root for smoke tests", file=sys.stderr)
        return EXIT_REFUSED_ROOT
    try:
        sched.fix_mode()  # before queuing: a registered data root refuses the smoke runs instead of keeping them
    except SchedulerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_REFUSED_ROOT
    print(f"queued {sched.add(specs)} of {len(specs)} smoke runs; ledgers: {cfg.ledger_paths.main.parent}")
    sched.run()
    for line in sched.status_lines(events=20):
        print(line)
    failed = failed_runs(sched)
    if failed:
        print(f"smoke test FAILED: {len(failed)} {plural(len(failed), 'run', 'runs')} did not end as expected: "
              f"{'; '.join(failed)}", file=sys.stderr)
        return EXIT_RUNS_FAILED
    print("smoke test passed: every run ended as expected (runs waiting for code not yet in the repository included)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Smoke test of the whole pipeline on short runs (HANDOVER.md, task 18; not analysed).

    python scripts/smoke_run.py --data-root /scratch/smoke                 # one 1-epoch PPO-Lagrangian run
    python scripts/smoke_run.py --data-root /scratch/smoke --design pilot --epochs 2

Queues short copies of registered runs (``--design``, default: one determinism-style run) with
``total_steps = epochs x 20,000`` and runs them through the real scheduler and launcher in smoke
mode (``allow_dirty`` and ``allow_pending``): the ledgers go under DATA_ROOT/smoke, never to the
repository, and the data root is fixed as a smoke root. The command line cannot queue short runs
(``schedule add`` takes only the registered designs), which is why this script exists.

Runs whose plug-in or evaluation harness is not in the repository yet wait, as in a real sweep;
the script prints the scheduler's summary and its last events. Evaluation needs the last ten
checkpoints of the ten-epoch cadence, as in a registered run, so use ``--epochs 100`` (or another
multiple of ten from 100) to pass it; in a shorter smoke test the scheduler still launches the
evaluation, which ends eval_failed (the checkpoints are missing) or, before Role 2's harness is
in the repository, leaves the run trained.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from configs import registered as R  # noqa: E402
from pilot import manifest  # noqa: E402
from pilot.manifest import RunSpec  # noqa: E402
from pilot.scheduler import Scheduler, SchedulerConfig, SchedulerError  # noqa: E402


def short(spec: RunSpec, epochs: int) -> RunSpec:
    """The same run with ``epochs`` epochs, a SMOKE- run_id, and the onset at the last checkpoint at or
    before half the run (0 for runs under 20 epochs, which then train as N = 0 does)."""
    steps = epochs * R.STEPS_PER_EPOCH
    # The onset at half the run, rounded down to the checkpoint grid (off-grid onsets inside the
    # selection window are Q-selection-window's case, not a smoke test's).
    onset = None if spec.onset_step is None else min(
        spec.onset_step, steps // 2 // R.CHECKPOINT_INTERVAL_STEPS * R.CHECKPOINT_INTERVAL_STEPS
    )
    return replace(spec, run_id="SMOKE-" + spec.run_id, total_steps=steps, onset_step=onset, depends_on=())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-root", type=Path, required=True, help="a scratch directory outside the repository")
    parser.add_argument("--design", choices=("determinism", "pilot"), default="determinism")
    parser.add_argument("--epochs", type=int, default=1, help="epochs per run (20,000 steps each; a multiple of ten from 100 to reach evaluation)")
    parser.add_argument("--max-concurrent", type=int, default=1)
    args = parser.parse_args(argv)
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")

    if args.design == "determinism":
        specs = [RunSpec(run_id="DET-PPOLag-PointGoal1-s0", study="A", task=R.PRIMARY_TASK, arm="smoke", seed=0,
                         total_steps=R.STEPS_PER_EPOCH, base_algo="PPOLag", plugin="ppolag", group="determinism")]
    else:
        specs = manifest.design("pilot")
    specs = [short(s, args.epochs) for s in specs]
    grid = R.SELECTION_WINDOW_CHECKPOINTS * R.CHECKPOINT_INTERVAL_EPOCHS  # 100 epochs: ten checkpoints after step 0
    if args.epochs < grid or args.epochs % R.CHECKPOINT_INTERVAL_EPOCHS:
        print(f"note: evaluation needs a multiple of {R.CHECKPOINT_INTERVAL_EPOCHS} epochs from {grid}; "
              "their evaluation ends eval_failed (the checkpoints are missing), or leaves them trained "
              "while Role 2's harness is not in the repository")
    cfg = SchedulerConfig(data_root=args.data_root, max_concurrent=args.max_concurrent, poll_seconds=5.0,
                          allow_dirty=True, allow_pending=True, require_allocation_for_pilot=False)
    sched = Scheduler(cfg)
    foreign = [r["run_id"] for r in sched.rows() if not r["run_id"].startswith("SMOKE-")]
    if foreign:  # registered runs queued by `schedule add` would otherwise train here in smoke mode
        print(f"error: {cfg.data_root} holds {len(foreign)} registered runs (e.g. {foreign[0]}); "
              "use a scratch data root for smoke tests", file=sys.stderr)
        return 1
    try:
        sched.fix_mode()  # before queuing: a registered data root refuses the smoke runs instead of keeping them
    except SchedulerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"queued {sched.add(specs)} of {len(specs)} smoke runs; ledgers: {cfg.ledger_paths.main.parent}")
    sched.run()
    for status, count in sorted(sched.summary().items()):
        print(f"{status:24s} {count}")
    for ev in reversed(sched.events(limit=20)):
        print(f"{ev['ts']}  {ev['run_id'] or '-':60s} {ev['event']:24s} {ev['detail'] or ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Recompute plasticity rows from a run's saved checkpoints (Role 3, Metrics and interventions; owner Abdullah).

An audit of contract 2 (``plasticity.csv``; Table 2.3; Part 3.4): every ``torch_save/epoch-k.pt``
of a run is loaded with ``torch.load(weights_only=True)`` (contract 3), the actor (plain or
injected, ``metrics.interventions.build_actor``) and the critics are rebuilt without touching any
global random stream, the fixed batch is normalised with that checkpoint's own ``obs_normalizer``
(``metrics.plasticity.normalise_frozen``), and the row of step k x steps_per_epoch (the run's own,
from its OmniSafe ``config.json``) is computed exactly as the live hook computes it::

    python -m metrics.recompute --run-dir RUN_DIR [--out FILE]

RUN_DIR is a pipeline run directory (``train_result.json``, ``omnisafe/...``) or an OmniSafe run
directory (``torch_save/``). The rows are compared with the run's ``plasticity.csv``. ``--out``
writes the recomputed rows as a CSV outside the run directory (never inside it: a run's records are
written once, by the run). One torch thread, as in training (Table 3.1), so that the floating-point
results are the live ones.

Exit codes (each outcome its own, so a caller never mistakes a failure for a result):

* 0 (``EXIT_OK``): every recomputed row equals the logged one exactly, or the run has no
  ``plasticity.csv`` (the rows are only recomputed);
* 1 (``EXIT_DIFFERENT``): the rows were recomputed and differ from the logged ones;
* 2 (``EXIT_USAGE``): the arguments are wrong (argparse's own code);
* 3 (``EXIT_REFUSED``): ``--out`` is inside the run or exists; nothing was computed or written;
* 4 (``EXIT_ERROR``): the rows could not be recomputed or compared (no OmniSafe directory, no
  ``config.json``, a batch problem, a checkpoint that does not load, an unreadable
  ``plasticity.csv``, ...); the error is printed to stderr.

A failed run's ``plasticity.csv`` may end in a line without a line end: an append the failure itself
cut short. That line is dropped with a RuntimeWarning (on stderr), as ``pilot.contracts.validate_plasticity``
drops it, not read: its last value may be truncated ("0.2" of "0.25") and still parse. Its step then
counts as a recomputed row the log lacks. A file that is empty, or empty once that line is dropped (a
run killed before or while the hook wrote the header), holds no logged row, as ``validate_plasticity``
reads a failed run's: every checkpoint is then a row the log lacks (exit 1), not an error.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
import traceback
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO

import numpy as np
import torch

from configs import registered as R
from metrics.batches import atomic_write_bytes, load_fixed_batch
from metrics.interventions import INJECTED_KEY, build_actor, is_injected
from metrics.plasticity import COLUMNS, NetworkParts, format_row, measure, parse_row
from pilot.contracts import PLASTICITY_FILE
from pilot.rundir import (
    CHECKPOINT_SUBDIR,
    TRAIN_RESULT,
    checkpoint_files,
    find_omnisafe_run_dir,
    read_config,
    run_steps_per_epoch,
)

EXIT_OK = 0
EXIT_DIFFERENT = 1
EXIT_USAGE = 2  # argparse exits with 2 on a usage error; no other outcome uses it
EXIT_REFUSED = 3
EXIT_ERROR = 4


def find_omnisafe_dir(run_dir: Path) -> Path:
    """The OmniSafe run directory of ``run_dir`` (itself, the launcher's record, or the one ``omnisafe/*/seed-*``)."""
    run_dir = Path(run_dir)
    if (run_dir / CHECKPOINT_SUBDIR).is_dir():
        return run_dir
    train_result = run_dir / TRAIN_RESULT
    if train_result.exists():
        relative = json.loads(train_result.read_text(encoding="utf-8")).get("omnisafe_dir")
        if relative:
            return run_dir / relative
    return find_omnisafe_run_dir(run_dir)  # the launcher's layout (pilot/rundir.py)


def _spaces(obs_dim: int, act_dim: int) -> tuple[Any, Any]:
    from gymnasium import spaces

    return (spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32),
            spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32))


def build_critic(config: Mapping[str, Any], obs_dim: int, act_dim: int,
                 state: Mapping[str, Any] | None) -> torch.nn.Module | None:
    """An OmniSafe V critic holding ``state`` (None if the checkpoint has no such critic), built inside ``fork_rng``."""
    if state is None:
        return None
    model_cfgs = config["model_cfgs"]
    critic_cfg = model_cfgs["critic"]
    with torch.random.fork_rng(devices=[]):  # nn.Linear construction draws from the global torch RNG
        from omnisafe.models.critic.critic_builder import CriticBuilder

        obs_space, act_space = _spaces(obs_dim, act_dim)
        critic = CriticBuilder(obs_space, act_space, [int(n) for n in critic_cfg["hidden_sizes"]],
                               activation=critic_cfg["activation"],
                               weight_initialization_mode=model_cfgs.get("weight_initialization_mode",
                                                                         "kaiming_uniform"),
                               num_critics=1).build_critic("v")
    critic.load_state_dict(dict(state), strict=True)
    critic.eval()
    return critic


def recompute_rows(omnisafe_dir: Path, *, batch: np.ndarray | None = None) -> dict[int, dict[str, int | float]]:
    """The plasticity row of every saved checkpoint of an OmniSafe run directory, by step.

    ``batch`` is for tests only; by default the committed fixed batch of the run's task is used.
    """
    omnisafe_dir = Path(omnisafe_dir)
    steps_per_epoch = run_steps_per_epoch(omnisafe_dir)  # FileNotFoundError without config.json
    config = read_config(omnisafe_dir)
    normalised = bool(config["algo_cfgs"]["obs_normalize"])
    raw = load_fixed_batch(str(config["env_id"])) if batch is None else batch
    rows: dict[int, dict[str, int | float]] = {}
    for k, path in checkpoint_files(omnisafe_dir):
        try:
            state = torch.load(path, weights_only=True, map_location="cpu")
        except Exception as exc:  # noqa: BLE001 - name the file: torch's message does not
            raise ValueError(f"{path} does not load with torch.load(weights_only=True) (contract 3): "
                             f"{type(exc).__name__}: {exc}") from exc
        pi = state["pi"]
        obs_dim = int(pi[INJECTED_KEY if is_injected(pi) else "mean.0.weight"].shape[1])
        act_dim = int(pi["log_std"].shape[0])
        actor = build_actor(config, obs_dim, act_dim, pi)
        normalizer = state.get("obs_normalizer") if normalised else None
        if normalised and normalizer is None:
            raise ValueError(f"{path} has no obs_normalizer although the run normalises observations")
        parts = NetworkParts(
            actor,
            build_critic(config, obs_dim, act_dim, state.get("reward_critic")),
            build_critic(config, obs_dim, act_dim, state.get("cost_critic")),
            normalizer,
        )
        step = k * steps_per_epoch
        rows[step] = measure(parts, raw, step=step)
    return rows


def read_logged_rows(omnisafe_dir: Path) -> dict[int, dict[str, int | float]] | None:
    """The rows of the run's ``plasticity.csv`` by step (None if there is no file; {} if it holds no header)."""
    path = Path(omnisafe_dir) / PLASTICITY_FILE
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    if text and not text.endswith("\n"):
        kept, line_end, partial = text.rpartition("\n")
        warnings.warn(f"{path}: the last line {partial!r} has no line end (an append a failed run cut short); "
                      "dropped, not compared", RuntimeWarning, stacklevel=2)
        text = kept + line_end
    if not text:
        return {}  # no header: created, then killed before the header was whole (pilot.contracts.validate_plasticity)
    with io.StringIO(text, newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} lacks the columns {missing}")
        rows: dict[int, dict[str, int | float]] = {}
        for record in reader:
            row = parse_row(record)
            step = int(row["step"])
            if step in rows:
                raise ValueError(f"{path} has step {step} twice")
            rows[step] = row
    return rows


def _same(a: float, b: float) -> bool:
    return (math.isnan(a) and math.isnan(b)) or a == b


def compare(recomputed: Mapping[int, Mapping[str, Any]], logged: Mapping[int, Mapping[str, Any]]) -> list[str]:
    """Every difference between recomputed and logged rows (empty if they agree exactly)."""
    problems = []
    for step in sorted(set(recomputed) | set(logged)):
        if step not in logged:
            problems.append(f"step {step}: checkpoint without a logged row")
        elif step not in recomputed:
            problems.append(f"step {step}: logged row without a checkpoint")
        else:
            differ = [c for c in COLUMNS if not _same(float(recomputed[step][c]), float(logged[step][c]))]
            if differ:
                details = [f"{c} logged {logged[step][c]!r} recomputed {recomputed[step][c]!r}" for c in differ]
                problems.append(f"step {step}: {', '.join(details)}")
    return problems


def rows_csv(rows: Mapping[int, Mapping[str, Any]]) -> str:
    """The rows as ``plasticity.csv`` text (the header, then the rows by step)."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(COLUMNS)
    for step in sorted(rows):
        writer.writerow(format_row(rows[step]))
    return buffer.getvalue()


def _inside(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def _write_out(path: Path, text: str) -> None:
    """Write the ``--out`` CSV atomically (fsync, mode 0644), creating its directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(path, text.encode("utf-8"))


def main(argv: Sequence[str] | None = None, out: TextIO | None = None, err: TextIO | None = None) -> int:
    """The command line (module docstring): exit codes ``EXIT_*``."""
    out = sys.stdout if out is None else out
    err = sys.stderr if err is None else err
    parser = argparse.ArgumentParser(prog="python -m metrics.recompute", description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", required=True, type=Path,
                        help="pipeline run directory or OmniSafe run directory")
    parser.add_argument("--out", type=Path, default=None,
                        help="CSV file for the recomputed rows (outside the run directory)")
    args = parser.parse_args(argv)  # a usage error exits with EXIT_USAGE (2)
    torch.set_num_threads(R.TORCH_THREADS)  # as in training (Table 3.1): identical floating-point results
    try:
        return _run(args, out)
    except Exception as exc:  # noqa: BLE001 - a failure to recompute is reported as such, never as a result
        print(f"error: could not recompute the plasticity rows of {args.run_dir}: {type(exc).__name__}: {exc}",
              file=err)
        traceback.print_exc(file=err)
        return EXIT_ERROR


def _run(args: argparse.Namespace, out: TextIO) -> int:
    omnisafe_dir = find_omnisafe_dir(args.run_dir)
    if args.out is not None:
        if _inside(args.out, args.run_dir) or _inside(args.out, omnisafe_dir):
            print(f"refused: --out {args.out} is inside the run directory; a run's records are written by the run only",
                  file=out)
            return EXIT_REFUSED
        if args.out.exists():
            print(f"refused: {args.out} exists", file=out)
            return EXIT_REFUSED
    recomputed = recompute_rows(omnisafe_dir)
    if args.out is not None:
        _write_out(args.out, rows_csv(recomputed))
        print(f"wrote {len(recomputed)} rows to {args.out}", file=out)
    logged = read_logged_rows(omnisafe_dir)
    if logged is None:
        print(f"{omnisafe_dir}: no {PLASTICITY_FILE}; recomputed {len(recomputed)} rows at steps {sorted(recomputed)}",
              file=out)
        return EXIT_OK
    problems = compare(recomputed, logged)
    if problems:
        print(f"{omnisafe_dir}: {len(problems)} difference(s):\n  " + "\n  ".join(problems), file=out)
        return EXIT_DIFFERENT
    print(f"{omnisafe_dir}: all {len(recomputed)} rows equal the logged {PLASTICITY_FILE} exactly "
          f"(steps {sorted(recomputed)})", file=out)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())

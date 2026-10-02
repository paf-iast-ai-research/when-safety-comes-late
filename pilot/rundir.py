"""The layout of a run directory, and the readers that every stage shares.

Owner: pilot owner (Role 1). A leaf module (standard library only): ``pilot.launch`` re-exports
the run-directory file names, ``MULTIPLIER_COLUMN``, ``STEPS_COLUMN``, ``find_omnisafe_run_dir`` and
``read_progress``, so existing ``from pilot.launch import TRAIN_RESULT`` imports keep working, and
``pilot.dependencies``, ``pilot.contracts`` and ``pilot.enrichment`` read run directories without
importing the launcher. Keeping these names here breaks the import cycle ``pilot.launch`` ->
``pilot.dependencies`` -> ``pilot.launch``.

Layout. Part 3.4: "Each run saves a checkpoint every 200,000 steps, at onset, and at the end of
training" (OmniSafe's torch_save below) and "Every run writes one row to the results ledger
(Appendix B) at completion, including the commit hash, the configuration file hash, the seed, the
wall-clock time and the machine" (built by pilot/ledger_writer.py from train_result.json and
evaluation.json below):

    RUN_DIR/spec.json                                  the RunSpec, written by the scheduler
    RUN_DIR/.claim                                     the training claim (one process per run)
    RUN_DIR/omnisafe/<exp_name>/seed-XXX-<timestamp>/  OmniSafe's log directory: progress.csv,
                                                       config.json, torch_save/epoch-{k}.pt
    RUN_DIR/train_result.json                          the training stage's result
    RUN_DIR/evaluation.json                            the evaluation stage's result
"""

from __future__ import annotations

import csv
import json
import numbers
import re
from pathlib import Path
from typing import Any, Mapping

OMNISAFE_SUBDIR = "omnisafe"
TRAIN_RESULT = "train_result.json"
EVALUATION_RESULT = "evaluation.json"
SPEC_FILE = "spec.json"
CLAIM_FILE = ".claim"

# Inside OmniSafe's log directory (omnisafe/common/logger.py:108-112, 167-168, 182 in 0.5.0).
PROGRESS_FILE = "progress.csv"
CONFIG_FILE = "config.json"
CHECKPOINT_SUBDIR = "torch_save"
# Used with fullmatch ("$" would also match before a trailing newline); [0-9], not \d, which matches
# any Unicode decimal digit.
CHECKPOINT_NAME = re.compile(r"epoch-([0-9]+)\.pt")

# progress.csv columns that the pipeline reads (policy_gradient.py:268-281; contract 3 of pilot/contracts.py).
MULTIPLIER_COLUMN = "Metrics/LagrangeMultiplier"  # plug-ins log every multiplier under this prefix (contract 3)
LEVEL_MULTIPLIER_PREFIX = MULTIPLIER_COLUMN + "/level_"  # Study B: one multiplier per level, level_{b:g}
STEPS_COLUMN = "TotalEnvSteps"
EPOCH_COLUMN = "Train/Epoch"  # 0-based: the row of epoch e holds the state after e + 1 epochs


def row_epoch(row: Mapping[str, Any]) -> int | None:
    """The 0-based epoch of a progress.csv row (``EPOCH_COLUMN``), or None if it is not a whole number."""
    try:
        value = float(row.get(EPOCH_COLUMN) or "nan")
    except ValueError:
        return None
    return int(value) if value.is_integer() else None


def find_omnisafe_run_dir(run_dir: Path) -> Path:
    """OmniSafe writes to RUN_DIR/omnisafe/<exp_name>/seed-XXX-<timestamp>/; return that directory."""
    candidates = sorted((Path(run_dir) / OMNISAFE_SUBDIR).glob("*/seed-*"))
    candidates = [c for c in candidates if c.is_dir()]
    if len(candidates) != 1:
        raise FileNotFoundError(f"expected one OmniSafe run directory under {run_dir}, found {len(candidates)}")
    return candidates[0]


def read_progress(omnisafe_dir: Path) -> list[dict[str, str]]:
    """The rows of OmniSafe's progress.csv, one per epoch, as strings."""
    with open(Path(omnisafe_dir) / PROGRESS_FILE, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def read_config(omnisafe_dir: Path) -> dict[str, Any]:
    """The configuration OmniSafe saved for the run (``Logger.save_config``: config.json)."""
    return json.loads((Path(omnisafe_dir) / CONFIG_FILE).read_text(encoding="utf-8"))


def run_steps_per_epoch(omnisafe_dir: Path, default: int | None = None) -> int:
    """``algo_cfgs.steps_per_epoch`` of the run, read from its own config.json.

    A checkpoint ``epoch-k.pt`` holds the state after k epochs of the run's *own* length, which is
    ``configs.registered.STEPS_PER_EPOCH`` for every registered run (pilot/launch.py sets it) but 2,000 in the tiny
    test configurations; reading it from the run keeps one code path for both. ``default`` is
    returned only when config.json does not exist (a fixture without one); a config.json without a
    positive integer ``algo_cfgs.steps_per_epoch`` raises ValueError.
    """
    path = Path(omnisafe_dir) / CONFIG_FILE
    if not path.exists():
        if default is not None:
            return int(default)
        raise FileNotFoundError(f"{path} does not exist")
    try:
        config = read_config(omnisafe_dir)
    except ValueError as exc:  # json.JSONDecodeError or UnicodeDecodeError
        raise ValueError(f"{path} is not valid JSON") from exc
    try:
        value = config["algo_cfgs"]["steps_per_epoch"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{path} has no algo_cfgs.steps_per_epoch") from exc
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value <= 0:
        raise ValueError(f"{path}: algo_cfgs.steps_per_epoch = {value!r} is not a positive integer")
    return int(value)


def checkpoint_file(omnisafe_dir: Path, epoch: int) -> Path:
    """``torch_save/epoch-{epoch}.pt``: OmniSafe's name for the state after ``epoch`` epochs (logger.py:182)."""
    return Path(omnisafe_dir) / CHECKPOINT_SUBDIR / f"epoch-{int(epoch)}.pt"


def checkpoint_files(omnisafe_dir: Path) -> list[tuple[int, Path]]:
    """(k, path) of every ``torch_save/epoch-k.pt`` (the state after k epochs), in epoch order."""
    found = []
    for path in (Path(omnisafe_dir) / CHECKPOINT_SUBDIR).glob("epoch-*.pt"):
        match = CHECKPOINT_NAME.fullmatch(path.name)
        if match:
            found.append((int(match.group(1)), path))
    return sorted(found)

"""Collect (once) or re-check the fixed evaluation batches (Role 3, Metrics and interventions; owner Abdullah).

Table 2.3 "Fixed evaluation batch": "2,048 states (decision) collected once per task by a
uniform-random policy under seed 0, stored in the repository". This is the script that made the
files in ``metrics/fixed_batch/`` (First Tasks section 9 step 1: "store ... in metrics/fixed_batch/
with the script that made them"):

    python -m metrics.collect_fixed_batch --all            # writes <task>.npy and MANIFEST.json
    python -m metrics.collect_fixed_batch --all --check    # re-collects and compares the bytes
    python -m metrics.collect_fixed_batch --all --check --json PATH   # ... and writes the result to PATH

The default mode refuses to overwrite an existing batch or manifest entry (Table 2.3: "collected
once"; First Tasks section 9 step 1: "never resampled"): a new batch needs an amendment. It prints
the ``FIXED_BATCH_SHA256`` entries to commit in ``metrics/batches.py``. ``--check`` re-collects every
batch and compares it with the three committed records (the file, MANIFEST.json,
``FIXED_BATCH_SHA256``). Run it on the workstation after ``scripts/setup_env.sh``: that MuJoCo
reproduces the same states on another machine is not verified here. Table 9.1 (Q-appendix-a): the
committed batches are the fixed batches (Table 2.3: collected once; First Tasks section 9 step 1:
never resampled); ``--check`` on the workstation is a reproducibility check, and a mismatch is
reported in Table 9.1 but does not replace them. ``--json PATH`` writes the check's result for that report (``check``'s
``report``): per task the committed and the re-collected sha256, whether the three committed
records agree (``committed_consistent``) and whether the re-collection reproduced the batch
(``reproduced``). scripts/workstation.py reads it; nothing here ever rewrites a committed batch.

Exit codes (each outcome its own, so a caller never mistakes a failure for a result):

* 0 (``EXIT_OK``): collected; or (``--check``) every batch byte-identical and the records agree;
* 1 (``EXIT_MISMATCH``): ``--check`` re-collected and found a difference (or a committed file missing);
* 2 (``EXIT_USAGE``): the arguments are wrong (argparse's own code);
* 3 (``EXIT_REFUSED``): a batch exists already, or MANIFEST.json describes another environment;
  nothing was written;
* 4 (``EXIT_ERROR``): the collection or the check could not run (an import, the environment, a
  file that cannot be read or written, ...); the error is printed to stderr.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import traceback
from pathlib import Path
from typing import Any, Sequence, TextIO

from configs import registered as R
from metrics import batches

EXIT_OK = 0
EXIT_MISMATCH = 1
EXIT_USAGE = 2  # argparse exits with 2 on a usage error; no other outcome uses it
EXIT_REFUSED = 3
EXIT_ERROR = 4


def _versions() -> dict[str, str]:
    import gymnasium
    import mujoco
    import numpy
    import safety_gymnasium

    return {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "gymnasium": gymnasium.__version__,
        "safety-gymnasium": safety_gymnasium.__version__,
        "mujoco": mujoco.__version__,
    }


def manifest_path(directory: Path | None = None) -> Path:
    """``MANIFEST.json`` in ``directory`` (default: ``metrics/fixed_batch/``)."""
    return (directory or batches.FIXED_BATCH_DIR) / batches.MANIFEST_FILE


def read_manifest(directory: Path | None = None) -> dict[str, Any] | None:
    """The parsed MANIFEST.json, or None if absent."""
    path = manifest_path(directory)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def manifest_bytes(manifest: dict[str, Any]) -> bytes:
    """UTF-8, LF, sorted keys, strict JSON (no NaN), trailing newline."""
    return (json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def task_entry(task: str, data: bytes, shape: tuple[int, ...], episode_starts: Sequence[int]) -> dict[str, Any]:
    """The MANIFEST.json record of one task's batch file (``data``: its exact bytes)."""
    return {
        "file": f"{task}.npy",
        "sha256": batches.sha256_hex(data),
        "shape": [int(n) for n in shape],
        "dtype": "float32",
        "n_states": R.FIXED_BATCH_STATES,
        "seed": R.FIXED_BATCH_SEED,
        "episode_starts": [int(i) for i in episode_starts],
    }


def collect(tasks: Sequence[str], directory: Path | None = None, out: TextIO | None = None) -> int:
    """Collect every task's batch and write the files and the manifest.

    If any requested task has a batch or manifest entry already (or the manifest records other
    versions), the whole call is refused and nothing is written.
    """
    out = sys.stdout if out is None else out
    directory = directory or batches.FIXED_BATCH_DIR
    manifest = read_manifest(directory)
    versions = _versions()
    if manifest is not None and manifest.get("versions") != versions:
        print(f"refused: MANIFEST.json records versions {manifest.get('versions')}, this environment has {versions}; "
              "one manifest must describe one environment", file=out)
        return EXIT_REFUSED
    existing = [t for t in tasks if (directory / f"{t}.npy").exists() or t in ((manifest or {}).get("tasks") or {})]
    if existing:
        print(f"refused: a fixed batch exists already for {existing} (Table 2.3: collected once; a new batch "
              "needs an amendment). Use --check to verify it.", file=out)
        return EXIT_REFUSED
    if manifest is None:
        manifest = {
            "procedure": batches.PROCEDURE,
            "registered": "Table 2.3 'Fixed evaluation batch'; Appendix A",
            "script": "metrics/collect_fixed_batch.py",
            "command": "python -m metrics.collect_fixed_batch --all",
            "versions": versions,
            "tasks": {},
        }
    directory.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        batch, starts = batches.collect_fixed_batch(task)
        batches.check_batch_array(batch, f"collected batch of {task}")
        data = batches.npy_bytes(batch)
        batches.atomic_write_bytes(directory / f"{task}.npy", data)
        manifest["tasks"][task] = task_entry(task, data, batch.shape, starts)
        # rewritten right after each file: at most the last file can lack its record (--check reports it)
        batches.atomic_write_bytes(manifest_path(directory), manifest_bytes(manifest))
        print(f"{task}: shape {batch.shape} episode starts {starts} sha256 {batches.sha256_hex(data)}", file=out)
    print("FIXED_BATCH_SHA256 entries for metrics/batches.py:", file=out)
    for task in sorted(manifest["tasks"]):
        print(f'    "{task}": "{manifest["tasks"][task]["sha256"]}",', file=out)
    return EXIT_OK


# The checks of ``check`` that compare the committed records with each other (the committed batch is consistent), and
# those that compare the re-collection with them (the batch is reproduced on this machine).
CONSISTENCY_CHECKS = ("sha256 == metrics/batches.py FIXED_BATCH_SHA256", "sha256 == MANIFEST.json")
REPRODUCTION_CHECKS = ("re-collection byte-identical", "MANIFEST episode_starts reproduced",
                       "MANIFEST shape reproduced")


def check(tasks: Sequence[str], directory: Path | None = None, out: TextIO | None = None,
          recorded: dict[str, str] | None = None, report: dict[str, Any] | None = None) -> int:
    """Re-collect each task and compare bytes with the committed file and the two hash records.

    The manifest's shape and episode_starts are compared with the re-collection too; a missing
    committed file is a mismatch. ``report`` (optional) is filled with the result: ``passed``,
    ``problems`` and per task ``committed_sha256`` and ``recollected_sha256`` (None for a missing
    committed file, which is not re-collected), ``committed_consistent`` (the file exists and its
    sha256 is the one ``FIXED_BATCH_SHA256`` and MANIFEST.json record, ``CONSISTENCY_CHECKS``),
    ``reproduced`` (``REPRODUCTION_CHECKS``) and every check by name.
    """
    out = sys.stdout if out is None else out
    directory = directory or batches.FIXED_BATCH_DIR
    recorded = dict(batches.FIXED_BATCH_SHA256) if recorded is None else recorded
    manifest = read_manifest(directory) or {}
    entries = manifest.get("tasks") or {}
    problems: list[str] = []
    versions = _versions()
    if manifest and manifest.get("versions") != versions:
        print(f"note: MANIFEST.json versions {manifest.get('versions')} differ from this environment's {versions}",
              file=out)
    results: dict[str, Any] = {}
    for task in tasks:
        path = directory / f"{task}.npy"
        if not path.exists():
            problems.append(f"{task}: {path} is missing")
            results[task] = {"committed_sha256": None, "recollected_sha256": None, "committed_consistent": False,
                             "reproduced": False, "checks": {}}
            continue
        committed = path.read_bytes()
        digest = batches.sha256_hex(committed)
        batch, starts = batches.collect_fixed_batch(task)
        fresh = batches.npy_bytes(batch)
        same = fresh == committed
        entry = entries.get(task) or {}
        checks = {
            "re-collection byte-identical": same,
            "sha256 == metrics/batches.py FIXED_BATCH_SHA256": recorded.get(task) == digest,
            "sha256 == MANIFEST.json": entry.get("sha256") == digest,
            "MANIFEST episode_starts reproduced": entry.get("episode_starts") == list(starts),
            "MANIFEST shape reproduced": entry.get("shape") == list(batch.shape),
        }
        for name, ok in checks.items():
            if not ok:
                problems.append(f"{task}: {name}: False")
        results[task] = {"committed_sha256": digest, "recollected_sha256": batches.sha256_hex(fresh),
                         "committed_consistent": all(checks[name] for name in CONSISTENCY_CHECKS),
                         "reproduced": all(checks[name] for name in REPRODUCTION_CHECKS), "checks": checks}
        print(f"{task}: sha256 {digest} " + " ".join(f"[{name}: {ok}]" for name, ok in checks.items()), file=out)
    if report is not None:
        report.update(passed=not problems, problems=list(problems), tasks=results)
    if problems:
        print("CHECK FAILED:\n  " + "\n  ".join(problems), file=out)
        return EXIT_MISMATCH
    print("check passed: every re-collected batch is byte-identical to the committed file", file=out)
    return EXIT_OK


def main(argv: Sequence[str] | None = None, out: TextIO | None = None, err: TextIO | None = None) -> int:
    """The command line (module docstring): exit codes ``EXIT_*``."""
    out = sys.stdout if out is None else out
    err = sys.stderr if err is None else err
    parser = argparse.ArgumentParser(prog="python -m metrics.collect_fixed_batch", description=__doc__.split("\n\n")[0])
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--task", choices=R.TASKS_STUDY_A)
    which.add_argument("--all", action="store_true", help="every Study A task (configs.registered.TASKS_STUDY_A)")
    parser.add_argument("--check", action="store_true", help="re-collect and compare bytes with the committed files")
    parser.add_argument("--json", metavar="PATH", type=Path,
                        help="with --check: write the result (check's report, module docstring) as JSON to PATH")
    args = parser.parse_args(argv)  # a usage error exits with EXIT_USAGE (2)
    if args.json is not None and not args.check:
        parser.error("--json writes the result of --check")
    tasks = list(R.TASKS_STUDY_A) if args.all else [args.task]
    try:
        if not args.check:
            return collect(tasks, out=out)
        report: dict[str, Any] = {}
        code = check(tasks, out=out, report=report)
        if args.json is not None:  # written only when the check ran to its end (an error writes nothing)
            text = json.dumps({"exit_code": code, **report}, indent=2, sort_keys=True, allow_nan=False) + "\n"
            batches.atomic_write_bytes(args.json, text.encode("utf-8"))
        return code
    except Exception as exc:  # noqa: BLE001 - a failure to collect or check is reported as such, never as a result
        what = "check" if args.check else "collect"
        print(f"error: could not {what} the fixed batches of {tasks}: {type(exc).__name__}: {exc}", file=err)
        traceback.print_exc(file=err)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())

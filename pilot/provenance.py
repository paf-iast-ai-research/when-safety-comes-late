"""Provenance of a run: code commit, configuration hash, machine and UTC timestamps.

Also: the clean-worktree rule and its result-file exemptions (``is_output_path``), the checks that
the loaded code is the committed code and the comparisons of two commits, and ``write_text_once``.

Appendix B, Table B.1: ``commit_hash; config_hash`` (code and configuration at launch) and
``started; finished; wall_clock_hours; machine`` (timing and hardware). Part 3.4: every run
writes the commit hash, the configuration file hash, the seed, the wall-clock time and the machine.

Owner: pilot owner (Role 1).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
_FULL_SHA = re.compile(r"[0-9a-f]{40}")

# Keys excluded from the configuration hash: they name where outputs go, not how the run trains.
# OmniSafe keeps a second copy of the custom settings under exp_increment_cfgs (AlgoWrapper).
VOLATILE_CONFIG_KEYS = (
    ("logger_cfgs", "log_dir"),
    ("exp_increment_cfgs", "logger_cfgs", "log_dir"),
)
# Excluded from every entry of pilot_cfgs.dependencies (any dependency id): where the dependency lies
# under this data root. Its commit, config hash, steps and checkpoint digest stay hashed (pilot/dependencies.py),
# so the hash identifies the input checkpoint but not the data root.
VOLATILE_DEPENDENCY_KEYS = ("run_dir", "omnisafe_dir")

# The two ledgers (written as runs complete; exempt from the clean-worktree rule).
OUTPUT_FILES = ("results/ledger.parquet", "results/pilot/ledger.parquet")
# Result directories whose data files (at any depth) are written after runs complete: the supplement
# records beside each ledger (results/supplement_schema.py; Q-ledger-v2) and the analysis reports (analysis/__main__.py).
OUTPUT_DIRS = ("results/supplement/", "results/pilot/supplement/", "results/analysis/")
OUTPUT_DIR_SUFFIXES = (".json", ".md", ".csv", ".parquet")


class DirtyWorktreeError(RuntimeError):
    """Raised when a step that records HEAD as its code runs with uncommitted changes."""


def utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC datetime (the ledger requires UTC)."""
    return datetime.now(timezone.utc)


def _git_raw(*args: str, cwd: Path = REPO_ROOT) -> str:
    """Git's standard output, unmodified (porcelain formats depend on leading spaces)."""
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


def _git(*args: str, cwd: Path = REPO_ROOT) -> str:
    return _git_raw(*args, cwd=cwd).strip()


def is_full_commit_hash(value: object) -> bool:
    """True for a 40-character lowercase hexadecimal Git commit hash (False for any non-string)."""
    return isinstance(value, str) and _FULL_SHA.fullmatch(value) is not None


def commit_hash(cwd: Path = REPO_ROOT) -> str:
    """Return the full hash of HEAD."""
    sha = _git("rev-parse", "HEAD", cwd=cwd)
    if not is_full_commit_hash(sha):
        raise RuntimeError(f"unexpected commit hash {sha!r}")
    return sha


def is_output_path(path: str) -> bool:
    """Result files written while runs complete; they are data, not code or configuration.

    Exactly the ledgers, their enrichment logs and evaluation-seed registries, the sidecar, the
    pilot's reports, the determinism reports, and the .json, .md, .csv and .parquet files under
    results/supplement/, results/pilot/supplement/ and results/analysis/. Nothing else is exempt from
    the clean-worktree rule (code, configuration and pilot/allocation.json in particular are not,
    nor is a .py file under those directories).
    """
    if ".." in path.split("/"):  # git never reports one; a path that climbs out is never a result file
        return False
    if path in OUTPUT_FILES:
        return True
    if path.startswith(OUTPUT_DIRS) and path.endswith(OUTPUT_DIR_SUFFIXES):
        return True
    if path.startswith("results/") and path.endswith((".enrichment_log.jsonl", ".seeds.json")):
        return True
    if path.startswith("results/pilot/sidecar/") and path.endswith(".json"):
        return True
    if path.startswith("results/pilot/") and "/" not in path[len("results/pilot/"):] and path.endswith((".json", ".md")):
        return True
    return path.startswith("ledger/determinism/") and path.endswith((".json", ".md"))


def _porcelain_paths(cwd: Path) -> list[str]:
    """Every path named by ``git status --porcelain=v1 -z --untracked-files=no`` (both sides of a rename or copy)."""
    fields = _git_raw("status", "--porcelain=v1", "-z", "--untracked-files=no", cwd=cwd).split("\0")
    paths: list[str] = []
    i = 0
    while i < len(fields):
        entry = fields[i]
        i += 1
        if len(entry) < 4:
            continue
        status, path = entry[:2], entry[3:]
        paths.append(path)
        if "R" in status or "C" in status:  # the next field is the original path
            paths.append(fields[i])
            i += 1
    return paths


def dirty_paths(cwd: Path = REPO_ROOT) -> list[str]:
    """Tracked files with uncommitted changes (staged or not), other than result files.

    Code, configuration and documentation alike; only result files (``is_output_path``) are exempt.
    Untracked files are ignored here; ``unverified_imported_code`` catches untracked code that a
    run actually imports.
    """
    return [p for p in _porcelain_paths(cwd) if not is_output_path(p)]


def _loaded_repository_files(root: Path) -> set[str]:
    """Repository-relative POSIX paths of the files of ``root`` that the current process imported.

    Virtual environments inside the checkout are skipped.
    """
    loaded = set()
    for module in list(sys.modules.values()):
        file = getattr(module, "__file__", None)
        if not file:
            continue
        path = Path(file).resolve()
        try:
            rel = path.relative_to(root).as_posix()
        except ValueError:
            continue
        if rel.startswith((".venv/", "venv/", "env/")) or "site-packages" in path.parts:
            continue
        loaded.add(rel)
    return loaded


def unverified_imported_code(cwd: Path = REPO_ROOT) -> list[str]:
    """Python files of this repository, imported by the current process, that are not committed as is.

    Called after a run's plug-in is imported and again after its factory (or the evaluation
    harness) ran, so a run cannot execute code that its recorded commit does not contain (for
    example an untracked envs/onset.py). Virtual environments inside the checkout are skipped.
    """
    root = cwd.resolve()
    tracked = set(_git_raw("ls-files", "-z", cwd=cwd).split("\0"))
    changed = set(_git_raw("diff", "--name-only", "-z", "HEAD", cwd=cwd).split("\0"))
    problems = []
    for rel in _loaded_repository_files(root):
        if rel not in tracked:
            problems.append(f"{rel} (untracked)")
        elif rel in changed:
            problems.append(f"{rel} (modified)")
    return sorted(problems)


def imported_code_changed_between(old: str, new: str, cwd: Path = REPO_ROOT) -> list[str]:
    """Loaded repository files that differ between commits ``old`` and ``new``.

    These are the Python files of this repository, imported by the current process, whose code at
    ``old`` (what a long-running process loaded) is no longer the code of ``new``. A commit that
    changes only other files (a report, a record) leaves the loaded code identical to the code at
    ``new``. A renamed or moved file counts as changed under both its paths (``--no-renames``,
    whatever git's ``diff.renames`` setting). Raises ``subprocess.CalledProcessError`` if git cannot
    compare the commits.
    """
    changed = set(_git_raw("diff", "--name-only", "--no-renames", "-z", old, new, cwd=cwd).split("\0")) - {""}
    return sorted(changed & _loaded_repository_files(cwd.resolve()))


def non_output_changes_between(old: str, new: str, cwd: Path = REPO_ROOT) -> list[str]:
    """Files other than results (``is_output_path``) that differ between commits ``old`` and ``new``.

    Empty when ``new`` only added or changed result files (a ledger, a report, a record), so code and
    configuration are those of ``old``. Both paths of a renamed file are compared (``--no-renames``).
    Raises ``subprocess.CalledProcessError`` if git cannot compare the commits.
    """
    changed = set(_git_raw("diff", "--name-only", "--no-renames", "-z", old, new, cwd=cwd).split("\0")) - {""}
    return sorted(p for p in changed if not is_output_path(p))


def file_committed_at(commit: str, rel_path: str, cwd: Path = REPO_ROOT) -> bytes | None:
    """Content of ``rel_path`` in ``commit``, or None if the commit does not contain it as a file.

    ``git cat-file blob`` reads files only: a directory (a tree, which ``git show`` would list with exit 0)
    reads as absent, just as reading a directory of the working tree as a file fails.
    """
    out = subprocess.run(["git", "cat-file", "blob", f"{commit}:{rel_path}"], cwd=cwd, capture_output=True)
    return out.stdout if out.returncode == 0 else None


def require_clean_worktree(cwd: Path = REPO_ROOT) -> str:
    """Return HEAD's hash, or raise if tracked files other than result files (``is_output_path``) differ from it.

    A registered run must be reproducible from its recorded commit, so it may not be launched
    from modified code.
    """
    dirty = dirty_paths(cwd)
    if dirty:
        raise DirtyWorktreeError(
            "tracked files have uncommitted changes; commit them first: "
            + ", ".join(dirty)
        )
    return commit_hash(cwd)


def write_text_once(path: Path, text: str) -> None:
    """Create ``path`` with ``text`` atomically and only once (FileExistsError if it exists).

    The text goes to a temporary file that is fsynced and then hard-linked to ``path``: the link is
    atomic and fails if the file exists, so a crash or a full disk never leaves an empty or partial
    once-only record behind (which would refuse every retry). The temporary file is removed on every
    normal or exceptional exit; a hard kill can leave a stray hidden .tmp file, never a partial record.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.link(tmp, path)  # atomic, and FileExistsError if the file exists
        fsync_directory(path.parent)
    finally:
        tmp.unlink(missing_ok=True)


def fsync_directory(directory: Path) -> None:
    """fsync ``directory``, so that a file just renamed or linked into it survives a crash."""
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:  # pragma: no cover - platforms without directory handles
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _drop_volatile(config: Mapping[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = json.loads(json.dumps(config, sort_keys=True, default=str))
    for path in VOLATILE_CONFIG_KEYS:
        node: Any = cleaned
        for key in path[:-1]:
            node = node.get(key) if isinstance(node, dict) else None
            if node is None:
                break
        if isinstance(node, dict):
            node.pop(path[-1], None)
    pilot_cfgs = cleaned.get("pilot_cfgs")
    dependencies = pilot_cfgs.get("dependencies") if isinstance(pilot_cfgs, dict) else None
    if isinstance(dependencies, dict):
        for entry in dependencies.values():
            if isinstance(entry, dict):
                for key in VOLATILE_DEPENDENCY_KEYS:
                    entry.pop(key, None)
    return cleaned


def config_hash(config: Mapping[str, Any]) -> str:
    """SHA-256 of the canonical JSON of a run's full configuration, without its locations.

    Dropped: its output locations (``VOLATILE_CONFIG_KEYS``) and the paths of its dependencies under
    the data root (``VOLATILE_DEPENDENCY_KEYS``).
    """
    canonical = json.dumps(_drop_volatile(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown-cpu"


def machine_description() -> str:
    """One-line machine identity for the ledger's ``machine`` field."""
    return (
        f"{socket.gethostname()} | {platform.system()} {platform.release()} | "
        f"{_cpu_model()} | {os.cpu_count()} logical cores | Python {platform.python_version()}"
    )

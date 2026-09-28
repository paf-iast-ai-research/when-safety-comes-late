"""Provenance of a run: code commit, configuration hash, machine and UTC timestamps.

Appendix B, Table B.1: ``commit_hash; config_hash`` (code and configuration at launch) and
``started; finished; wall_clock_hours; machine`` (timing and hardware). Part 3.4: every run
writes the commit hash, the configuration file hash, the seed, the wall-clock time and the machine.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")

# Keys excluded from the configuration hash: they name where outputs go, not how the run trains.
# OmniSafe keeps a second copy of the custom settings under exp_increment_cfgs (AlgoWrapper).
VOLATILE_CONFIG_KEYS = (
    ("logger_cfgs", "log_dir"),
    ("exp_increment_cfgs", "logger_cfgs", "log_dir"),
)


class DirtyWorktreeError(RuntimeError):
    """Raised when a registered run is launched from uncommitted code."""


def utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC datetime (the ledger requires UTC)."""
    return datetime.now(timezone.utc)


def _git_raw(*args: str, cwd: Path = REPO_ROOT) -> str:
    """Git's standard output, unmodified (porcelain formats depend on leading spaces)."""
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


def _git(*args: str, cwd: Path = REPO_ROOT) -> str:
    return _git_raw(*args, cwd=cwd).strip()


def is_full_commit_hash(value: str) -> bool:
    """True for a 40-character lowercase hexadecimal Git commit hash."""
    return bool(_FULL_SHA.match(value))


def commit_hash(cwd: Path = REPO_ROOT) -> str:
    """Return the full hash of HEAD."""
    sha = _git("rev-parse", "HEAD", cwd=cwd)
    if not is_full_commit_hash(sha):
        raise RuntimeError(f"unexpected commit hash {sha!r}")
    return sha


OUTPUT_FILES = ("results/ledger.parquet", "results/pilot/ledger.parquet")


def is_output_path(path: str) -> bool:
    """Result files written while runs complete; they are data, not code or configuration.

    Exactly the ledgers, their enrichment logs and evaluation-seed registries, the sidecar, the
    pilot's reports and the determinism reports. Nothing else is exempt from the clean-worktree rule (code, configuration and
    pilot/allocation.json in particular are not).
    """
    if path in OUTPUT_FILES:
        return True
    if path.startswith("results/") and path.endswith((".enrichment_log.jsonl", ".seeds.json")):
        return True
    if path.startswith("results/pilot/sidecar/") and path.endswith(".json"):
        return True
    if path.startswith("results/pilot/") and "/" not in path[len("results/pilot/"):] and path.endswith((".json", ".md")):
        return True
    return path.startswith("ledger/determinism/") and path.endswith((".json", ".md"))


def _porcelain_paths(cwd: Path) -> list[str]:
    """Every path named by `git status --porcelain=v1 -z` (both sides of a rename or copy)."""
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
    """Tracked code or configuration files with uncommitted changes (staged or not).

    Untracked files are ignored here; ``unverified_imported_code`` catches untracked code that a
    run actually imports. Result files (``is_output_path``) are exempt.
    """
    return [p for p in _porcelain_paths(cwd) if not is_output_path(p)]


def unverified_imported_code(cwd: Path = REPO_ROOT) -> list[str]:
    """Python files of this repository, imported by the current process, that are not committed as is.

    Called after a run's plug-in is imported and again after its factory (or the evaluation
    harness) ran, so a run cannot execute code that its recorded commit does not contain (for
    example an untracked envs/onset.py). Virtual environments inside the checkout are skipped.
    """
    import sys

    root = cwd.resolve()
    tracked = set(_git_raw("ls-files", "-z", cwd=cwd).split("\0"))
    changed = set(_git_raw("diff", "--name-only", "-z", "HEAD", cwd=cwd).split("\0"))
    problems = []
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
        if rel not in tracked:
            problems.append(f"{rel} (untracked)")
        elif rel in changed:
            problems.append(f"{rel} (modified)")
    return sorted(problems)


def file_committed_at(commit: str, rel_path: str, cwd: Path = REPO_ROOT) -> bytes | None:
    """Content of ``rel_path`` in ``commit``, or None if the commit does not contain it."""
    out = subprocess.run(["git", "show", f"{commit}:{rel_path}"], cwd=cwd, capture_output=True)
    return out.stdout if out.returncode == 0 else None


def require_clean_worktree(cwd: Path = REPO_ROOT) -> str:
    """Return HEAD's hash, or raise if tracked files differ from it.

    A registered run must be reproducible from its recorded commit, so it may not be launched
    from modified code.
    """
    dirty = dirty_paths(cwd)
    if dirty:
        raise DirtyWorktreeError(
            "tracked files have uncommitted changes; commit them before launching a registered run: "
            + ", ".join(dirty)
        )
    return commit_hash(cwd)


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
    return cleaned


def config_hash(config: Mapping[str, Any]) -> str:
    """SHA-256 of the canonical JSON of a run's full configuration, without output locations."""
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

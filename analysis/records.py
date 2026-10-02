"""The amendment and errata registries of the analysis (Part 8.2; Part 5.8; Part 9, Table 9.1).

Owner: Role 4, Analysis and results (Muhammad Abdullah).

* ``AMENDMENTS.json``: Part 8.2, "a change made after data are in view labels every affected result
  exploratory"; the registration status table: "A result affected by a change made after the data
  were seen is reported as exploratory". An entry with ``data_seen`` true relabels every verdict it
  affects (named in ``affects``, or depending on one of its ``keys`` through analysis.questions).
  The go amendment records ``analysis_code_hash`` (Part 5.8 freeze).
* ``ERRATA.json``: Part 5.8, the script "is not edited between the go decision and the final analysis
  except to fix errors, each of which is recorded in Part 9". The report states whether the analysis
  code has the go hash or the hash reached from it by a chain of recorded errata (each
  ``code_hash_before`` the previous ``code_hash_after``); any other hash is flagged.
* ``code_hash`` (answered in Table 9.1, Q-exploratory-labels): the analysis code is analysis/*.py and
  every in-repository module the analysis loads, directly or transitively, function-level imports
  included (``IMPORTED_CODE``, the import closure: configs/registered.py, whose registered constants
  and ANSWERED_QUESTIONS every verdict reads, the package files configs/__init__.py and
  pilot/__init__.py, pilot/manifest.py, pilot/contracts.py, pilot/provenance.py, pilot/go_decision.py
  and the modules they import, pilot/budget.py, pilot/rundir.py and pilot/errors.py, and the ledger and
  supplement schemas results/ledger_schema.py and results/supplement_schema.py, which the data loader
  reads; tests/test_analysis_cli.py checks the closure). An edit to any of them after the go decision,
  an amendment's included, needs its recorded erratum (or its entry's hashes) to keep the chain. The
  ledger and supplement schemas are also frozen by their own SHA-256 files (results/).

Both files document their own fields; ``load_amendments`` and ``load_errata`` check them.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Optional

from configs import registered as R

from analysis import questions
from analysis.verdict import EXPLORATORY, Verdict

ANALYSIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = ANALYSIS_DIR.parent
# The in-repository modules outside analysis/ that the analysis loads, directly or transitively, function-level
# imports included, repository-relative and sorted (answered in Table 9.1, Q-exploratory-labels): the registered
# constants and answered questions, the design, cuts and seed roles, the pipeline contracts, the provenance checks,
# the go report's conditions, the ledger and supplement schemas and what they import change results as much as
# analysis/*.py does. tests/test_analysis_cli.py derives the import closure and checks this list against it.
IMPORTED_CODE = ("configs/__init__.py", "configs/registered.py", "pilot/__init__.py", "pilot/budget.py",
                 "pilot/contracts.py", "pilot/errors.py", "pilot/go_decision.py", "pilot/manifest.py",
                 "pilot/provenance.py", "pilot/rundir.py", "results/ledger_schema.py", "results/supplement_schema.py")
AMENDMENTS_FILE = ANALYSIS_DIR / "AMENDMENTS.json"
ERRATA_FILE = ANALYSIS_DIR / "ERRATA.json"
# \Z, not $: "$" also matches before a trailing newline, which would accept "<hash>\n" (never a code hash)
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}\Z")
_SHA1 = re.compile(r"^[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"^[0-9a-f]{64}\Z")


class RegistryError(ValueError):
    """AMENDMENTS.json or ERRATA.json does not follow its documented fields."""


def _entries(path: Path, list_key: str, fields: tuple[str, ...], content: Optional[bytes] = None, *,
             unrecorded_commit: bool = False) -> list[dict[str, Any]]:
    """The registry's entries, from ``content`` when given (bytes already read and checked), else from ``path``.

    ``unrecorded_commit``: an entry's ``commit`` may be null, "not yet recorded" (AMENDMENTS.json documents it: the
    commit that records an amendment cannot name itself, so the entry written before it carries null until Role 4
    enters the hash after the group's ratification).
    """
    try:
        raw = Path(path).read_bytes() if content is None else content
        data = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RegistryError(f"{path}: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get(list_key), list):
        raise RegistryError(f"{path}: needs a list {list_key!r}")
    fields_doc = data.get("fields")
    if not isinstance(fields_doc, dict) or not fields_doc or set(fields) - set(fields_doc):
        raise RegistryError(f"{path}: 'fields' documents each field by name (an object naming at least {list(fields)})")
    documented = set(fields_doc)
    entries = data[list_key]
    ids = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise RegistryError(f"{path}: every entry is an object")
        missing = [f for f in fields if f not in entry]
        unknown = sorted(set(entry) - documented)
        if missing or unknown:
            raise RegistryError(f"{path}: entry {entry.get('id')!r} misses {missing} or has undocumented fields {unknown}")
        # types first: a wrong type must be a refusal (RegistryError), never a TypeError further on
        unrecorded = unrecorded_commit and entry["commit"] is None  # a commit not yet recorded
        for name in ("id", "date") if unrecorded else ("id", "date", "commit"):
            _require_str(path, entry, name)
        if entry["id"] in ids:
            raise RegistryError(f"{path}: id {entry['id']!r} twice")
        ids.add(entry["id"])
        if not _DATE.match(entry["date"]):
            raise RegistryError(f"{path}: {entry['id']}: date must be YYYY-MM-DD")
        try:
            datetime.date.fromisoformat(entry["date"])
        except ValueError:
            raise RegistryError(f"{path}: {entry['id']}: date {entry['date']!r} is not a calendar date") from None
        if not unrecorded and not _SHA1.match(entry["commit"]):
            raise RegistryError(f"{path}: {entry['id']}: commit must be a 40-hex commit hash"
                                + (" or null (not yet recorded)" if unrecorded_commit else ""))
    return entries


def _require_str(path: Path, entry: dict[str, Any], name: str) -> None:
    if not isinstance(entry[name], str):
        raise RegistryError(f"{path}: entry {entry.get('id')!r}: {name} must be a string, "
                            f"got {type(entry[name]).__name__}")


def _require_str_list(path: Path, entry: dict[str, Any], name: str) -> None:
    value = entry[name]
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise RegistryError(f"{path}: {entry['id']}: {name} must be a list of strings")


def load_amendments(path: Path = AMENDMENTS_FILE, *, content: Optional[bytes] = None) -> list[dict[str, Any]]:
    """The amendments, checked: required fields and their types (strings; ``keys`` and ``affects`` lists of
    strings; ``commit`` a 40-hex hash or null while not yet recorded), known PENDING keys, a boolean ``data_seen``.

    ``content``: the file's bytes as ``python -m analysis`` read and checked them against the commit.
    """
    entries = _entries(path, "amendments", ("id", "table_9_1_row", "date", "commit", "change", "reason",
                                            "approved_by", "keys", "data_seen", "affects"), content,
                       unrecorded_commit=True)
    for entry in entries:
        if not isinstance(entry["data_seen"], bool):
            raise RegistryError(f"{path}: {entry['id']}: data_seen must be true or false")
        for name in ("table_9_1_row", "change", "reason", "approved_by"):
            _require_str(path, entry, name)
        _require_str_list(path, entry, "keys")
        _require_str_list(path, entry, "affects")  # verdict ids, or prefixes ending in '*'
        unknown = [k for k in entry["keys"] if k not in R.PENDING]
        if unknown:
            raise RegistryError(f"{path}: {entry['id']}: keys {unknown} are not PENDING keys")
        code_hash = entry.get("analysis_code_hash")
        if code_hash is not None and not (isinstance(code_hash, str) and _SHA256.match(code_hash)):
            raise RegistryError(f"{path}: {entry['id']}: analysis_code_hash must be a SHA-256")
    return entries


def load_errata(path: Path = ERRATA_FILE, *, content: Optional[bytes] = None) -> list[dict[str, Any]]:
    """The errata, checked: required fields, ``files`` a list of strings and SHA-256 code hashes (``content`` as in
    ``load_amendments``)."""
    entries = _entries(path, "errata", ("id", "table_9_1_row", "date", "commit", "files", "description",
                                        "code_hash_before", "code_hash_after"), content)
    for entry in entries:
        for name in ("table_9_1_row", "description"):
            _require_str(path, entry, name)
        _require_str_list(path, entry, "files")
        for key in ("code_hash_before", "code_hash_after"):
            if not (isinstance(entry[key], str) and _SHA256.match(entry[key])):
                raise RegistryError(f"{path}: {entry['id']}: {key} must be a SHA-256")
    return entries


def code_files(root: Path = REPO_ROOT) -> list[str]:
    """The repository-relative paths ``code_hash`` covers, sorted: analysis/*.py under ``root`` and ``IMPORTED_CODE``."""
    return sorted([f"analysis/{path.name}" for path in Path(root, "analysis").glob("*.py")] + list(IMPORTED_CODE))


def code_hash(root: Path = REPO_ROOT) -> str:
    """SHA-256 over the analysis code and the modules it imports (``code_files`` of the tree at ``root``), each file
    as (repository-relative path, NUL, bytes, NUL) in sorted path order (Part 5.8 freeze; Q-exploratory-labels). A
    file of ``IMPORTED_CODE`` missing under ``root`` is an error (OSError), never a hash without it."""
    digest = hashlib.sha256()
    for rel in code_files(root):
        digest.update(rel.encode("utf-8") + b"\0" + Path(root, rel).read_bytes() + b"\0")
    return digest.hexdigest()


def _affected(verdict: Verdict, amendment: dict[str, Any]) -> bool:
    for pattern in amendment["affects"]:
        if pattern.endswith("*") and verdict.id.startswith(pattern[:-1]):
            return True
        if verdict.id == pattern:
            return True
    keys = set(amendment["keys"])
    return bool(keys & set(questions.keys_for(verdict.table_key, conditions=verdict.conditions))) if verdict.table_key else False


def apply_amendments(verdicts: Iterable[Verdict], amendments: Iterable[dict[str, Any]]) -> None:
    """Part 8.2: relabel as exploratory every verdict an amendment made after data were seen affects."""
    amendments = [a for a in amendments if a["data_seen"]]
    for verdict in verdicts:
        hits = [a["id"] for a in amendments if _affected(verdict, a)]
        if hits and verdict.label != EXPLORATORY:
            verdict.notes.append(f"labelled exploratory (was {verdict.label}): affected by amendment(s) {hits} made after "
                                 "data were seen (Part 8.2)")
            verdict.label = EXPLORATORY


def freeze_state(amendments: list[dict[str, Any]], errata: list[dict[str, Any]],
                 current: Optional[str] = None) -> dict[str, Any]:
    """Part 5.8: is the analysis code (``code_hash``) the code of the go decision, or that code with recorded fixes
    only?"""
    current = current or code_hash()
    go_hashes = [a["analysis_code_hash"] for a in amendments if a.get("analysis_code_hash")]
    if not go_hashes:
        return {"code_hash": current, "state": "no go-decision hash recorded yet (AMENDMENTS.json analysis_code_hash)"}
    go_hash = go_hashes[-1]
    if current == go_hash:
        return {"code_hash": current, "go_hash": go_hash, "state": "unchanged since the go decision"}
    # the errata must chain from the go hash to the current one (E1.before == go hash, each later before == the
    # earlier after): an erratum whose 'after' is the current hash does not cover an unrecorded edit made before
    # it (Part 5.8: "not edited ... except to fix errors, each of which is recorded")
    chain: list[str] = []
    link, used = go_hash, set()
    while link != current:
        step = next((i for i, e in enumerate(errata) if i not in used and e["code_hash_before"] == link), None)
        if step is None:
            break
        used.add(step)
        chain.append(errata[step]["id"])
        link = errata[step]["code_hash_after"]
    if link == current:
        return {"code_hash": current, "go_hash": go_hash, "state": f"changed since the go decision by recorded errata {chain}"}
    if not chain and not any(e["code_hash_after"] == current for e in errata):
        return {"code_hash": current, "go_hash": go_hash, "flagged": True,
                "state": "CHANGED since the go decision without a recorded erratum (Part 5.8; ERRATA.json)"}
    return {"code_hash": current, "go_hash": go_hash, "flagged": True,
            "state": f"CHANGED since the go decision by an unrecorded edit: the recorded errata {chain} lead from the "
                     f"go hash to {link}, and no erratum has code_hash_before {link} (Part 5.8; ERRATA.json)"}

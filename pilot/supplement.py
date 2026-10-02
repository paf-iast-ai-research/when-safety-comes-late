"""Supplement records beside a ledger: written once, validated, never overwritten (HANDOVER.md section 8).

Owner: pilot owner (Role 1), the only writer of the ledgers and of their supplements. The models are
Role 4's (``results/supplement_schema.py``); the reader is ``analysis.data``. Q-ledger-v2 (HANDOVER.md
section 9, a note answered in Table 9.1; docs/DECISIONS.md): ledger schema v1 stays frozen, and what it
cannot hold, the per-episode costs and returns behind every mean (Part 5.3: "the interquartile mean
with a stratified bootstrap over seeds and evaluation episodes"), the battery at the final checkpoint
and at tolerance 5.0 (Part 4.1 rule 7; Part 4.1.1), the continuations (Table 2.2), Study B's
evaluations at every budget (Table 2.5; Part 4.2) and the training quantities (Tables 2.1, 2.3, 2.4),
is written here, one JSON file per (kind, run, part); the supplement schema is frozen at version 1
(results/supplement_schema.sha256).

Layout (HANDOVER.md section 8): ``supplement_dir(ledger) = ledger.parent / "supplement"`` (results/supplement
for the main ledger, results/pilot/supplement for the pilot's, ``<data root>/smoke/supplement`` for
the smoke ledgers), and in it ``<kind>/<run_id>.json`` or ``<kind>/<run_id>.<part>.json``. The part
of a record is a function of its content (``results.supplement_schema.record_part``: the condition,
or ``b<budget>`` for a few-shot continuation), so the file name can never disagree with the record.

Rules (HANDOVER.md section 8: "Written once, validated, never overwritten (a byte-identical rewrite is
accepted)"):

* ``write`` validates the record with ``validate_record`` and writes ``canonical_json`` (sorted keys,
  no timestamps), atomically: a temporary file is written, flushed, fsynced and hard-linked to its
  name, which fails if the name exists, so a crash never leaves a partial record and a record is
  never replaced.
* An existing record with the same bytes is accepted (a retry after a crash writes the same record).
  A differing one is refused (``SupplementConflict``), with one exception, which relaxes that rule
  and is named for the integrator: a writer recovering from a crash between a record and the ledger
  change it belongs to passes ``accept_reproduction=True``, and a record that differs from the
  existing one only in ``code_commit`` is then accepted: the same numbers reproduced by later code (evaluation is
  deterministic). The writers are an enrichment whose ledger field is still empty
  (``pilot.enrichment._commit`` decides that, per record), ``pilot.ledger_writer.write_run``, whose
  row is not in the ledger yet (its ``evaluation`` record), and
  ``pilot.ledger_writer.ensure_evaluation_supplement``, which first checks the record's numbers and
  seed sets against the row already in the ledger (``write_run``'s recovery may have kept a record
  of earlier code, and a later pass of the scheduler must accept it). The existing file, with the commit that
  first produced it, is kept unchanged. Without that exception a crash at that moment followed by a
  new commit would leave the field or row unwritable for good, since a record is never replaced.
* ``check`` answers what ``write`` would do without writing, so that a caller writing several
  records can refuse all of them before it writes the first (``pilot.enrichment._commit``).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterator

from pilot import provenance

SUPPLEMENT_SUBDIR = "supplement"
# A part names a record within its run: a condition ("hazard") or a few-shot budget ("b5", "b15").
_PART = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*-s[0-9]+$")  # a run_id ends with '-s<seed>'
_REPRODUCIBLE = "code_commit"  # the only field a reproduced record may differ in (accept_reproduction)


class SupplementError(ValueError):
    """A supplement record that may not be written or read as it stands."""


class SupplementConflict(SupplementError):
    """A record for this (kind, run, part) exists already with other content; records are written once."""


def supplement_dir(ledger_path: Path | str) -> Path:
    """The supplement of a ledger lives beside it, ``<ledger dir>/supplement`` (HANDOVER.md section 8).

    The same rule as ``analysis.data.default_supplement_dir`` (the reader); a test checks they agree.
    """
    return Path(ledger_path).parent / SUPPLEMENT_SUBDIR


def _schema():
    from results import supplement_schema

    return supplement_schema


def _check_names(kind: str, run_id: str, part: str | None) -> None:
    S = _schema()
    if not isinstance(kind, str) or kind not in S.KIND_MODELS:
        raise SupplementError(f"unknown supplement kind {kind!r}; choose from {list(S.KIND_MODELS)}")
    if not isinstance(run_id, str) or not _RUN_ID.match(run_id):
        raise SupplementError(f"{run_id!r} is not a run_id (it must end with '-s<seed>')")
    if kind in S.PART_KINDS:
        if not isinstance(part, str) or not _PART.match(part) or ".." in part:
            raise SupplementError(f"a {kind} record needs a part (its condition or budget), got {part!r}")
    elif part is not None:
        raise SupplementError(f"a {kind} record has one file per run and no part, got {part!r}")


def record_path(kind: str, run_id: str, *, ledger_path: Path | str, part: str | None = None) -> Path:
    """``<supplement>/<kind>/<run_id>[.<part>].json`` of the ledger ``ledger_path``."""
    _check_names(kind, run_id, part)
    name = f"{run_id}.{part}.json" if part is not None else f"{run_id}.json"
    return supplement_dir(ledger_path) / kind / name


def _validated(kind: str, run_id: str, record: Any, part: str | None) -> tuple[Any, str]:
    """The record's model and its canonical text; SupplementError unless it is a valid record of this run and part."""
    S = _schema()
    try:
        model = S.validate_record(kind, record)
    except ValueError as exc:  # pydantic's ValidationError is a ValueError
        raise SupplementError(f"not a valid {kind} record of {run_id}: {exc}") from exc
    if model.run_id != run_id:
        raise SupplementError(f"the {kind} record names run {model.run_id!r}, not {run_id!r}")
    expected = S.record_part(kind, model)
    if part != expected:
        raise SupplementError(f"the {kind} record of {run_id} is part {expected!r} by its content, not {part!r}")
    return model, S.canonical_json(kind, model)


def _same_but_commit(existing: str, text: str) -> bool:
    try:
        old, new = json.loads(existing), json.loads(text)
    except ValueError:
        return False
    if not isinstance(old, dict) or not isinstance(new, dict):
        return False
    old.pop(_REPRODUCIBLE, None)
    new.pop(_REPRODUCIBLE, None)
    return old == new


def check(kind: str, run_id: str, record: Any, *, ledger_path: Path | str,
          part: str | None = None, accept_reproduction: bool = False) -> bool:
    """What ``write`` would do, without writing anything: False when the record is new, True when an
    acceptable record is on disk already; ``SupplementError`` (``SupplementConflict`` for a differing
    record) when ``write`` would refuse it. The caller holds the ledger's lock, so nothing can change
    between ``check`` and ``write``."""
    path = record_path(kind, run_id, ledger_path=ledger_path, part=part)  # checks the names first
    _, text = _validated(kind, run_id, record, part)
    if not path.exists():
        return False
    _confirm(path, text, kind, run_id, part, accept_reproduction)
    return True


def write(kind: str, run_id: str, record: Any, *, ledger_path: Path | str,
          part: str | None = None, accept_reproduction: bool = False) -> Path:
    """Write one supplement record beside ``ledger_path`` once; returns its path (HANDOVER.md section 8).

    ``record`` is validated (``results.supplement_schema.validate_record``); its ``run_id`` must be
    ``run_id`` and ``part`` must be ``record_part(kind, record)``. The file is created atomically and
    never replaced: an existing file with the same bytes is accepted, a differing one raises
    ``SupplementConflict`` (with ``accept_reproduction``, for crash recovery only, one that differs
    only in ``code_commit`` is accepted and kept as it is; see the module docstring).
    """
    path = record_path(kind, run_id, ledger_path=ledger_path, part=part)  # checks the names first
    _, text = _validated(kind, run_id, record, part)
    if path.exists():
        return _confirm(path, text, kind, run_id, part, accept_reproduction)
    try:
        provenance.write_text_once(path, text)  # fsync, then link: atomic, and never replaces an existing record
    except FileExistsError:  # another writer got there first: its record must be this one
        return _confirm(path, text, kind, run_id, part, accept_reproduction)
    return path


def _confirm(path: Path, text: str, kind: str, run_id: str, part: str | None, accept_reproduction: bool) -> Path:
    existing = path.read_text(encoding="utf-8")
    if existing == text or (accept_reproduction and _same_but_commit(existing, text)):
        return path
    what = f"{kind} record of {run_id}" + (f" ({part})" if part else "")
    raise SupplementConflict(
        f"{path} holds a different {what}; a record is written once and never replaced. "
        "A deterministic re-evaluation must reproduce it exactly: find out why it differs"
    )


def read(kind: str, run_id: str, *, ledger_path: Path | str, part: str | None = None) -> Any | None:
    """The validated record of (kind, run, part), or None when it was not written.

    SupplementError when the file cannot be read or is not a valid record of that run and part.
    """
    path = record_path(kind, run_id, ledger_path=ledger_path, part=part)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SupplementError(f"{path} is not readable JSON: {exc}") from exc
    model, _ = _validated(kind, run_id, data, part)
    return model


def exists(kind: str, run_id: str, *, ledger_path: Path | str, part: str | None = None) -> bool:
    """True when the file of (kind, run, part) exists (its content is not read)."""
    return record_path(kind, run_id, ledger_path=ledger_path, part=part).exists()


def files(ledger_path: Path | str) -> Iterator[Path]:
    """Every record file of the ledger's supplement, sorted (temporary files of a writer excluded)."""
    root = supplement_dir(ledger_path)
    if not root.is_dir():
        return iter(())
    return iter(sorted(p for p in root.rglob("*.json") if p.is_file() and not p.name.startswith(".")))

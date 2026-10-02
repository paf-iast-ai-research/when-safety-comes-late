"""Exceptions shared by the pipeline and the other roles' code, and the open-question gate.

Kept in a module of their own so that every process sees one class: runs are started as
``python -m pilot.launch``, where a class defined in ``pilot/launch.py`` would exist twice
(``__main__.RunRefused`` and ``pilot.launch.RunRefused``), and a refusal raised by a plug-in, a
hook or a harness would not be recognised as one. ``pilot.launch.RunRefused`` is this class.

``require_answered`` is the result gate of ``configs.registered.PENDING``: a function whose result
depends on the answer of a PENDING key calls it first, and refuses while the key is open (HANDOVER.md
section 9; every key is answered in Table 9.1, docs/DECISIONS.md, so the gate holds only for a key
the group changes, until its code follows). Smoke tests bypass it only through
``set_pending_allowed(True)``, which the launcher and the command line call for their
``--allow-pending`` flag (smoke data roots only), or inside an ``allow_pending(True)`` block. There
is deliberately no environment variable: a stale export in an operator's shell must never open a
gate on registered data.
"""

from __future__ import annotations

import contextlib
from typing import Iterator

_PENDING_ALLOWED = False


class RunRefused(Exception):
    """The run (or the step) must not start from this checkout or configuration; nothing is excluded.

    The launcher exits 5 (EXIT_REFUSED) and the scheduler leaves the run as it was (pending, or
    trained for the evaluation stage). Plug-in factories, hooks and harnesses raise it for every
    problem of configuration or specification, so that it is never classified as a crash.
    """


class PendingQuestionError(RunRefused, ValueError):
    """An open pre-registration question (``configs.registered.PENDING``) blocks this step."""


def pending_allowed() -> bool:
    """True only after ``set_pending_allowed(True)`` or inside ``allow_pending(True)`` (smoke tests)."""
    return _PENDING_ALLOWED


def set_pending_allowed(flag: bool) -> None:
    """Set the smoke-test bypass for the rest of the process (the launcher and CLI entry points)."""
    global _PENDING_ALLOWED
    _PENDING_ALLOWED = bool(flag)


@contextlib.contextmanager
def allow_pending(flag: bool = True) -> Iterator[None]:
    """Within the block, ``require_answered`` passes when ``flag`` is true (smoke tests only)."""
    global _PENDING_ALLOWED
    previous = _PENDING_ALLOWED
    _PENDING_ALLOWED = bool(flag)
    try:
        yield
    finally:
        _PENDING_ALLOWED = previous


def open_keys(*keys: str) -> tuple[str, ...]:
    """The given PENDING keys that the amendment log has not answered (KeyError for an unknown key)."""
    from configs import registered as R

    return tuple(k for k in keys if R.is_open(k))  # R.is_open raises KeyError for a key not in PENDING


def require_answered(*keys: str, what: str) -> None:
    """Refuse ``what`` while any of ``keys`` is open, unless smoke tests allow pending questions.

    Unknown keys raise KeyError even in smoke tests: a misspelt key is a programming error.
    """
    still_open = open_keys(*keys)
    if still_open and not _PENDING_ALLOWED:
        raise PendingQuestionError(
            f"{what} depends on open pre-registration questions {', '.join(still_open)} "
            "(configs/registered.py PENDING; HANDOVER.md section 9); it waits for the amendment log"
        )

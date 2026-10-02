"""Workstation runbook: one command per remaining step of the pilot owner (HANDOVER.md section 4).

Once the group has ratified its answers to the open questions (HANDOVER.md tasks 10 and 11), the
pilot owner's remaining steps are runs on the workstation, each one command here, and the document
entries the steps print (Tables 8.1, 9.0 and 9.1). This script knows the order of the remaining steps,
reads whether each is done from what is committed at HEAD (never from the working tree), and runs a
step through the repository's own tools, with their real command lines:

    python scripts/workstation.py status            # read-only: each step DONE, TODO or BLOCKED (why); the next command
    python scripts/workstation.py next [OPTIONS]    # run the first step that is not done (stops at a people's step)
    python scripts/workstation.py STEP [OPTIONS]    # run one step

The steps, in order (HANDOVER.md section 4 task numbers; ``STEPS``):

  registration         task 1       ``bash scripts/tag_registration.sh``; the operator then pushes the tag (Table 9.0)
  env                  task 8       ``bash scripts/setup_env.sh`` (Table 3.1; Appendix A)
  allocation           task 12      ``python -m pilot allocate --hours H --basis TEXT --by BY --date YYYY-MM-DD``
                                    [``--confirm``] [``--smoke-root DIR``] (Part 6 G2; X-allocation: before any
                                    run on the workstation reveals its speed)
  fixed-batch          task 8a      ``python -m metrics.collect_fixed_batch --all --check``, then the record
                                    environment/fixed_batch_check.json (Table 2.3; Appendix A; Q-appendix-a)
  determinism-default  task 9       ``python scripts/determinism_check.py`` (plug-in ppolag, the bitwise form)
  determinism          task 12a     ``python scripts/determinism_check.py --registered-form --plugin P`` for
                                    study_a, unconstrained_ppo, study_b and study_a_pid, in this order (Table 3.1)
  search               task 16      a people's step (Role 5): the record of Appendix C (Part 7.2)
  registration-record  task 3       a people's step: Tables 9.0 and 9.1 filled and committed, the tag pushed,
                                    before any pilot run (X-registration)
  pilot-runs           task 20      ``python -m pilot schedule add --design pilot``, then ``python -m pilot
                                    schedule run --max-concurrent K`` (Part 3.6)
  pilot-enrich         tasks 21-23  ``python -m pilot enrich select``, ``enrich measure``, ``enrich battery``,
                                    ``python -m pilot g4-measure``
  go                   tasks 24-26  ``python -m pilot go``, ``python -m analysis --mode pilot``, and
                                    ``python -m pilot surplus`` when its two values are given

and then the go meeting (task 27), a people's step.

When a step is done, read from HEAD through git (``Head``):

  registration         the annotated tag registration exists (locally) and points at the commit
                       scripts/tag_registration.sh names;
  env                  environment/requirements.lock.txt and environment/workstation.json are committed;
  allocation           pilot/allocation.json is committed;
  fixed-batch          environment/fixed_batch_check.json is committed, records the check's outcome (passed, or a
                       re-collection that differs: done with a warning, Table 9.1, Q-appendix-a), and records
                       the sha256 of every batch that metrics/fixed_batch/MANIFEST.json at HEAD lists;
  determinism-default  a passing bitwise-form report of plug-in ppolag is committed under ledger/determinism/
                       (not a rehearsal, from a clean tree, both runs on the commit it names); the report in the
                       working tree of a check that failed on HEAD's commit blocks it, as it blocks the
                       determinism step for its plug-in (never retried on the same commit, Table 3.1);
  determinism          ``pilot.scheduler.committed_determinism_reports(HEAD)`` holds each plug-in, the
                       scheduler's own fact for its determinism_missing gate;
  search               ``pilot.scheduler.search_record_status(HEAD)`` releases the search_record_missing gate;
  registration-record  the two registration files at HEAD (the docx and its PDF) differ from the registration
                       commit's (Tables 9.0 and 9.1 filled); the runbook cannot see whether the tag is pushed;
  pilot-runs           the eight pilot runs are queued on the data root and every pilot run (run_ids P-) is
                       finished there (``pilot.__main__.FINISHED_STATUSES``, from its scheduler state, read-only),
                       none waiting for a decision (``schedule status``: an exclusion without its replacement
                       or not yet in the ledger, a held, blocked or failed run);
  pilot-enrich         results/pilot/ledger.parquet and results/pilot/g4_moderate.json are committed;
  go                   a go report of the pilot and the output of the analysis test run are committed, and the
                       latest such output passed its test (its go-report check found no mismatch and compared
                       something: ``python -m analysis`` exit 0, not 3 or 4).

``status`` runs nothing, and never fails on a missing file, a missing or unreadable data root, or a
directory that is not a Git repository: it says so. ``next`` runs the first step that is not done
and stops at a people's step, with two exceptions where the scheduler itself holds the runs that
wait: the search record holds only the pilot's Study B runs (search_record_missing), so ``next``
reports it and goes on to queue and run the pilot; and a determinism report other than study_a's
holds only that plug-in's runs (determinism_missing), so ``next`` goes on once study_a's is committed.

The script never commits, tags or pushes, and changes Git state only through the tools it calls (the
tag script creates the local tag). A step whose output must be committed ends by printing the exact
``git add`` and ``git commit`` lines. Every command is printed before it runs (``+ cmd``), from the
repository root, and every Python command with this interpreter (``sys.executable``: after the env
step, run this script with the pinned environment's Python, ``source .venv/bin/activate``); a
command that fails ends the step with its exit status.

Where the pre-registration decides what a step does:

* Table 3.1 "Determinism check": "Two runs with the same seed and configuration must produce
  identical evaluation cost at the first checkpoint before any arm is launched." The determinism
  step refuses a plug-in, naming the keys, while its configuration waits on an open question
  (scripts/determinism_check.py ``waiting_on``), before anything is trained.
* Table 2.3 "Fixed evaluation batch": the batch is "collected once per task by a uniform-random
  policy under seed 0, stored in the repository". The fixed-batch step re-collects it on the
  workstation from a clean tree, with the package versions HEAD's lock pins (the record names that
  lock; another interpreter is refused). Table 9.1 (Q-appendix-a): the committed batches are the fixed
  batches; the re-collection is a reproducibility check. A re-collection that differs is recorded
  (``"passed": false``, the committed and the re-collected sha256 of each batch), the step prints the
  Table 9.1 row that reports it, and the step is done, with a warning, once the record is committed;
  the batches are never rewritten. A check that cannot run, or committed records that disagree with
  each other, write nothing: the operator stops and raises it with the group (HANDOVER task 8a).
* Part 6 G2: "The allocation is written into Part 8 before the pilot's throughput is read, so that
  it cannot be set to fit the result." The number and how it was derived are the group's resource
  decision: the script never invents them, and without ``--hours`` or ``--basis`` the allocation step
  prints what the group must decide and exits 2. X-allocation (Table 9.1): it is recorded before the
  determinism checks and the smoke runs on the workstation, which already reveal its speed, so the
  step comes right after the env step and the determinism steps refuse to run before it is committed.
* Part 7.2: the search record "is committed to the repository before Study B's first run".
* Part 5.8: the analysis script "is run once on the pilot data to test it". The go step runs it once,
  and never again over an existing output; a test run that found an error (exit 3) or tested nothing
  (exit 4) holds the step until the error is fixed and the analysis run again by hand.

Exit status: 0 the step ran, or was done already (``status``: always 0); 2 a usage error, or a value
only the group or the operator can give is missing (``allocation`` without ``--hours`` or ``--basis``,
``pilot-runs`` without ``--max-concurrent``); 3 the step cannot go on now (a people's step, an open
question, a failed check, an uncommitted tree, HEAD that moved, an output that must not be written
twice): what was not run is said; 130 interrupted (Ctrl-C). Otherwise the exit status of the called command that
failed, printed with it (the tools' own codes: HANDOVER.md section 7); the go step repeats the
analysis test run's 3 or 4 while that output holds it.

Owner: pilot owner (Role 1).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import shlex
import sqlite3
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping, Sequence

SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parent  # the working tree the steps write to and the commands run in (tests point it elsewhere)
sys.path.insert(0, str(REPO))

from configs import registered as R  # noqa: E402
from pilot import budget, manifest, provenance  # noqa: E402
from pilot import scheduler as S  # noqa: E402
from pilot.__main__ import DEFAULT_DATA_ROOT, FINISHED_STATUSES  # noqa: E402
from pilot.manifest import RunSpec  # noqa: E402

EXIT_OK = 0
EXIT_USAGE = 2  # argparse's code; also a value only the group or the operator can give is missing
EXIT_BLOCKED = 3  # the step cannot go on now; nothing more is run
EXIT_INTERRUPTED = 130

DONE, TODO, BLOCKED = "DONE", "TODO", "BLOCKED"

TAG = "registration"
TAG_SCRIPT = "scripts/tag_registration.sh"
# The registration commit's two files (scripts/tag_registration.sh checks them): Tables 9.0 and 9.1 are filled in them
REGISTRATION_FILES = ("prereg/Preregistration.docx", "prereg/Preregistration.pdf")
SETUP_SCRIPT = "scripts/setup_env.sh"
DETERMINISM_SCRIPT = "scripts/determinism_check.py"
LOCK = "environment/requirements.lock.txt"
WORKSTATION = "environment/workstation.json"
FIXED_BATCH_DIR = "metrics/fixed_batch"
FIXED_BATCH_MANIFEST = "metrics/fixed_batch/MANIFEST.json"
FIXED_BATCH_CHECK = "environment/fixed_batch_check.json"
FIXED_BATCH_COMMAND = ("-m", "metrics.collect_fixed_batch", "--all", "--check")
FIXED_BATCH_MISMATCH = 1  # metrics.collect_fixed_batch.EXIT_MISMATCH: the check ran and a re-collection differs
# Table 9.1 (Q-appendix-a), recorded with a check whose re-collection differs
FIXED_BATCH_READING = ("Table 9.1 (Q-appendix-a): the committed batches are the fixed batches (Table 2.3: collected "
                       "once, never resampled); this check is a reproducibility check, and its mismatch is reported in "
                       "Table 9.1 without replacing them")
# The packages the record of the fixed-batch check names (the stack that produces the states, and torch)
FIXED_BATCH_PACKAGES = ("omnisafe", "safety-gymnasium", "mujoco", "gymnasium", "numpy", "torch")
# The "check" of a report scripts/determinism_check.py writes without --registered-form (tests compare the two)
DEFAULT_FORM = "bitwise form (First Tasks, Role 1 step 10)"
DEFAULT_FORM_PLUGIN = "ppolag"
# HANDOVER task 12a: the pilot needs the first three; the 15 PID-check runs (Table 3.3) wait on the last
DETERMINISM_ORDER = ("study_a", "unconstrained_ppo", "study_b", "study_a_pid")
PILOT_STUDY_A_PLUGIN = "study_a"  # the plug-in of the pilot's six Study A runs (manifest.pilot)
ALLOCATION = S.ALLOCATION_PATH
DEFAULT_BY = "the group"
PILOT_RESULTS = "results/pilot"
PILOT_LEDGER = "results/pilot/ledger.parquet"  # pilot.ledger_writer.DEFAULT_PATHS.pilot
G4_FILE = "results/pilot/g4_moderate.json"  # what `python -m pilot g4-measure` writes once (revision 0)
ANALYSIS_DIR = "results/analysis"
GO_REPORT = re.compile(r"^results/pilot/go_report-[^/]+\.json$")
PILOT_ANALYSIS = re.compile(r"^results/analysis/pilot-[^/]+/analysis\.json$")
ENRICH_ACTIONS = ("select", "measure", "battery")  # HANDOVER tasks 21 and 22, in this order
# Statuses of the scheduler that wait for an operator action or a group decision (`schedule status`, "decisions")
DECISION_STATUSES = ("interrupted", "blocked", "eval_failed", "ledger_failed")
# What `python -m analysis --mode pilot` exits with (analysis/__main__.py): 3 and 4 after writing its output
ANALYSIS_EXITS = {2: "refused, nothing written",
                  3: "the analysis and the go report disagree on a number: an error to fix before the go meeting",
                  4: "nothing could be compared with the go report: the test run tested nothing"}

ENV_MESSAGE = "Pinned environment: lock and workstation record (HANDOVER task 8; Table 3.1; Appendix A)"
FIXED_BATCH_MESSAGE = "Fixed batches re-checked on the workstation (HANDOVER task 8a; Table 2.3; Appendix A)"
FIXED_BATCH_MISMATCH_MESSAGE = ("Fixed batches re-checked on the workstation: the re-collection differs, the committed "
                                "batches stay (HANDOVER task 8a; Table 2.3; Q-appendix-a)")
DEFAULT_DETERMINISM_MESSAGE = "Determinism check, default form, plug-in ppolag (HANDOVER task 9; Table 3.1)"
ENRICH_MESSAGE = "Pilot ledger, enrichment and G4 measurement (HANDOVER tasks 20 to 23; Part 3.6; Part 6 G4)"
GO_MESSAGE = "Go report and the analysis test run on the pilot (HANDOVER tasks 24 and 25; Part 6; Part 5.8)"

ALLOCATION_DECISION = """\
The machine-hour allocation is the group's resource decision (HANDOVER task 12); this script never invents it.
  Part 6 G2: "The allocation is written into Part 8 before the pilot's throughput is read, so that it cannot be
  set to fit the result."
  Table 8.1, "Machine-hours allocated to the registered runs": "A group decision, recorded before the
  throughput measurement is read".
  G2 as amended (Table 9.1, Q-g2-run-equivalents) holds when the measured wall-clock time per run, divided by
  the runs sustained concurrently, times the corrected run-equivalents of the design in force ({corrected:.2f}
  for the uncut design, pilot.budget.corrected_run_equivalents) is at most this allocation; the registered
  {registered} run-equivalents are reported beside it and decide nothing (`python -m pilot budget` prints both).
The group decides H, in workstation wall-clock hours (all concurrent runs together), and its basis: how H was
derived from the calendar and the machine's availability only (e.g. 24 x the days from the planned start of the
main sweep to the last date runs may finish x the fraction of time the workstation is available), never from a
timing of a run on the workstation.
Once the group has decided, record both, before the determinism checks and the smoke runs on the workstation,
which already reveal its speed (X-allocation, Table 9.1), and so before any pilot run is launched:
  python scripts/workstation.py allocation --hours H --basis TEXT --date YYYY-MM-DD [--by "the group"]
(--confirm for an H above {confirm_above} hours; --smoke-root DIR for each smoke data root already used on the
workstation, whose timings the record lists), and enter the same value in Table 8.1 with an amendment row in
Table 9.1 before any pilot run."""

SEARCH_INSTRUCTIONS = (
    "Role 5 (Hamza Nisar) runs the targeted search of Appendix C by studyb/search/README.md: "
    "`python -m studyb.search queries` prints the searches, `python -m studyb.search check` exits 0 only when the "
    "record is complete; then studyb/search/ is committed and its path entered in Table 9.0 by an amendment row "
    "(HANDOVER tasks 16 and 17)")

REGISTRATION_RECORD = (
    "fill Table 9.0 and the first row of Table 9.1 (HANDOVER task 3; the values are at the end of docs/DECISIONS.md), "
    "commit the docx and its regenerated PDF in a commit that changes nothing else, and push the tag (git push origin "
    "registration), before any pilot run (X-registration)")

# When pilot.scheduler.search_record_status releases the search_record_missing gate (Q-search-before-pilot)
SEARCH_RELEASE = ("HEAD holds the complete record, with a 'partial' outcome also its committed narrowing amendment "
                  "(an 'included' outcome withdraws Study B and holds them for good)")

GO_MEETING = (
    "The go meeting (HANDOVER task 27) is a people's step: the group enters the Table 8.1 values in Part 8 (with the "
    "seeds per primary-comparison arm of the surplus plan, Part 5.5), records the decision, commits the amended "
    "document and freezes Parts 1, 4, 5 and 6; Role 4 records the go amendment in analysis/AMENDMENTS.json.")


# ---------------------------------------------------------------------------
# What is committed at HEAD (read with git, never from the working tree)
# ---------------------------------------------------------------------------


class Head:
    """The facts of the commit at HEAD that the steps read (``commit`` is None outside a Git repository).

    Files are read with ``git cat-file`` (``provenance.file_committed_at``) and listed with ``git
    ls-tree`` (``pilot.scheduler.committed_files``); the determinism reports and the search record are
    judged by the scheduler's own functions, so a step is done exactly when its launch gate is released.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.commit: str | None = None
        self.problem = ""
        try:
            self.commit = provenance.commit_hash(self.root)
        except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
            stderr = getattr(exc, "stderr", None)
            detail = stderr.strip() if isinstance(stderr, str) and stderr.strip() else str(exc)
            self.problem = (f"cannot read HEAD in {self.root}: not a Git repository, or git is unavailable "
                            f"({detail.splitlines()[0] if detail else type(exc).__name__})")

    def file(self, rel: str) -> bytes | None:
        """The bytes of ``rel`` as committed at HEAD (None: not committed)."""
        return None if self.commit is None else provenance.file_committed_at(self.commit, rel, self.root)

    def file_at(self, commit: str, rel: str) -> bytes | None:
        """The bytes of ``rel`` as committed at ``commit`` (None: not committed there, or no such commit here)."""
        return provenance.file_committed_at(commit, rel, self.root)

    def json(self, rel: str) -> Any:
        """The JSON value of ``rel`` as committed at HEAD (None: not committed, or not JSON)."""
        data = self.file(rel)
        if data is None:
            return None
        try:
            return json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            return None

    def files(self, directory: str) -> list[str]:
        """Repository-relative paths of the files under ``directory`` at HEAD (at any depth)."""
        return [] if self.commit is None else S.committed_files(self.commit, directory, self.root)

    def tag(self, name: str) -> str | None:
        """The commit the tag ``name`` points at (None: no such tag)."""
        return self._rev_parse(f"refs/tags/{name}^{{commit}}")

    def tag_annotated(self, name: str) -> bool:
        """True when the tag ``name`` is a tag object (annotated); a lightweight tag names its commit itself."""
        target = self._rev_parse(f"refs/tags/{name}")
        return target is not None and target != self.tag(name)

    def _rev_parse(self, ref: str) -> str | None:
        out = subprocess.run(["git", "rev-parse", "-q", "--verify", ref], cwd=self.root, capture_output=True, text=True)
        return out.stdout.strip() if out.returncode == 0 else None

    def determinism_plugins(self) -> dict[str, str]:
        """{plug-in: its first passing registered-form report at HEAD} (the determinism_missing gate's fact)."""
        return {} if self.commit is None else S.committed_determinism_reports(self.commit, self.root)

    def search_record(self) -> tuple[bool, str]:
        """(released, reason) of the search_record_missing gate at HEAD."""
        return (False, self.problem) if self.commit is None else S.search_record_status(self.commit, self.root)


def read_head() -> Head:
    """The facts of HEAD in the repository (``REPO``)."""
    return Head(REPO)


# ---------------------------------------------------------------------------
# Running commands
# ---------------------------------------------------------------------------


def python(*args: str) -> list[str]:
    """A command line of this interpreter (``sys.executable``)."""
    return [sys.executable, *args]


def call(cmd: Sequence[str]) -> int:
    """Run ``cmd`` from the repository root, printed first (``+ cmd``); its exit status."""
    print("+ " + shlex.join(cmd), flush=True)
    return subprocess.run(list(cmd), cwd=REPO).returncode


def commit_lines(paths: Sequence[str], message: str) -> list[str]:
    """The ``git add`` and ``git commit -m "..."`` lines that commit ``paths`` (the operator runs them)."""
    quoted = shlex.quote(message) if re.search(r'["$`\\!]', message) else f'"{message}"'
    return ["git add " + " ".join(shlex.quote(p) for p in paths), f"git commit -m {quoted}"]


def print_commit(paths: Sequence[str], message: str) -> None:
    """Print the lines that commit the step's output (on a role branch; main takes pull requests, task 6)."""
    print("Commit it (on your branch, then a pull request):")
    for line in commit_lines(paths, message):
        print(line)


def number(value: float) -> str:
    """A float as the command line gives it back, never rounded (12000.0 -> 12000; else its shortest exact form)."""
    text = repr(float(value))
    return text[:-2] if text.endswith(".0") else text


@lru_cache(maxsize=None)
def _script(name: str) -> ModuleType:
    """scripts/<name>.py, loaded as a module (the runbook applies what the script itself applies)."""
    spec = importlib.util.spec_from_file_location(f"_workstation_{name}", SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def physical_cores() -> int:
    """The physical cores of this machine as scripts/record_workstation.py records them, else the logical ones."""
    return _script("record_workstation").physical_cores() or os.cpu_count() or 1


def suggested_concurrency() -> int:
    """HANDOVER task 20: the runs the workstation holds at once, usually the physical cores minus 1."""
    return max(1, physical_cores() - 1)


def interpreter_note() -> str | None:
    """A warning when this interpreter is not the repository's pinned environment ``.venv`` (once it exists)."""
    venv = REPO / ".venv"
    try:
        if not venv.is_dir() or Path(sys.prefix).resolve() == venv.resolve():
            return None
    except OSError:
        return None
    return (f"the commands run with {sys.executable}, not the pinned environment {venv}: "
            "source .venv/bin/activate, then run this script again")


def data_root_of(args: argparse.Namespace) -> str:
    """The scheduler's data root (``--data-root``; default ``$WSCL_DATA_ROOT`` or /data, as ``python -m pilot``)."""
    return str(getattr(args, "data_root", None) or DEFAULT_DATA_ROOT)


def written(rel: str) -> bool:
    """True when the working tree holds the file ``rel``."""
    return (REPO / rel).is_file()


def files_in(directory: str, pattern: str = "*") -> set[str]:
    """Repository-relative paths of the working tree's files ``directory/pattern``."""
    root = REPO / directory
    return {f"{directory}/{p.name}" for p in root.glob(pattern) if p.is_file()} if root.is_dir() else set()


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Verdict:
    """A step's state (DONE, TODO or BLOCKED) and why.

    ``skippable``: BLOCKED, but what waits for it are runs the scheduler holds itself (a launch gate),
    so ``next`` reports it and goes on to the following steps. ``warning``: DONE, with something the
    operator reports (``status`` and the step print it).
    """

    state: str
    why: str
    skippable: bool = False
    warning: str = ""


@dataclass(frozen=True)
class Step:
    """One step of the runbook: its HANDOVER task, its done fact, what it runs, and its options."""

    name: str
    tasks: str
    title: str
    verdict: Callable[[Head, argparse.Namespace], Verdict]
    run: Callable[[Head, argparse.Namespace], int]
    command: Callable[[argparse.Namespace], str]  # the runbook's command line (``status`` prints it)
    runs: Callable[[argparse.Namespace], str]  # the tools it calls
    options: tuple[str, ...] = ()  # option groups of build_parser


def judge(step: Step, head: Head, args: argparse.Namespace) -> Verdict:
    """The step's verdict; a fact that cannot be read is reported, never raised (``status`` never fails)."""
    if head.commit is None:
        return Verdict(BLOCKED, head.problem)
    try:
        return step.verdict(head, args)
    except Exception as exc:  # noqa: BLE001 - an unreadable fact is reported as such
        return Verdict(BLOCKED, f"cannot tell ({type(exc).__name__}: {exc})")


# -- 1. registration (task 1; Table 9.0) --------------------------------------


def registration_commit(head: Head) -> str | None:
    """The registration commit that scripts/tag_registration.sh names (``COMMIT=...``) at HEAD."""
    found = re.search(rb'^COMMIT="([0-9a-f]{40})"', head.file(TAG_SCRIPT) or b"", re.M)
    return found.group(1).decode("ascii") if found else None


def registration_verdict(head: Head, args: argparse.Namespace) -> Verdict:
    expected = registration_commit(head)
    if expected is None:
        return Verdict(BLOCKED, f"{TAG_SCRIPT} is not committed at HEAD, or names no registration commit")
    actual = head.tag(TAG)
    if actual is None:
        return Verdict(TODO, f"no tag '{TAG}' yet (it must point at the registration commit {expected})")
    if actual != expected:
        return Verdict(BLOCKED, f"the tag '{TAG}' points at {actual}, not at the registration commit {expected}: "
                                "a published tag is never moved; raise it with the group")
    if not head.tag_annotated(TAG):
        return Verdict(BLOCKED, f"the tag '{TAG}' on {expected} is a lightweight tag, and X-registration names an "
                                f"annotated one: if it is not pushed yet, git tag -d {TAG} and run this step again; "
                                "otherwise raise it with the group")
    return Verdict(DONE, f"the local tag '{TAG}' points at {expected} (publish it with git push origin {TAG}, before "
                         "any pilot run; the runbook cannot see the remote)")


def registration_run(head: Head, args: argparse.Namespace) -> int:
    code = call(["bash", TAG_SCRIPT])
    if code:
        return code
    print("Now publish the tag (this script never pushes):")
    print(f"git push origin {TAG}")
    return EXIT_OK


# -- the registration record (task 3; Table 9.0; X-registration): a people's step before the pilot runs --


def registration_record_verdict(head: Head, args: argparse.Namespace) -> Verdict:
    expected = registration_commit(head)
    if expected is None:
        return Verdict(BLOCKED, f"{TAG_SCRIPT} is not committed at HEAD, or names no registration commit")
    registered = {rel: head.file_at(expected, rel) for rel in REGISTRATION_FILES}
    absent = [rel for rel, data in registered.items() if data is None]
    if absent:
        return Verdict(BLOCKED, f"cannot compare: the registration commit {expected} holds no {', '.join(absent)} in "
                                "this repository (a shallow clone?)")
    missing = [rel for rel in REGISTRATION_FILES if head.file(rel) is None]
    if missing:
        return Verdict(BLOCKED, "not committed at HEAD: " + ", ".join(missing) + ": raise it with the group")
    unchanged = [rel for rel in REGISTRATION_FILES if head.file(rel) == registered[rel]]
    if unchanged:
        # The status table, "Registration record": until Part 9 holds the hash, the timestamp and the address, the
        # document is "the adopted draft and not the registration record"
        return Verdict(BLOCKED, f"a people's step: {', '.join(unchanged)} unchanged at HEAD from the registration "
                                f"commit, so Tables 9.0 and 9.1 are not filled yet: {REGISTRATION_RECORD}")
    return Verdict(DONE, f"{' and '.join(REGISTRATION_FILES)} at HEAD differ from the registration commit's (Tables "
                         f"9.0 and 9.1 filled, task 3); the tag is published with git push origin {TAG} (the runbook "
                         "cannot see the remote)")


def registration_record_run(head: Head, args: argparse.Namespace) -> int:
    verdict = registration_record_verdict(head, args)
    print(f"registration-record (HANDOVER task 3) is a people's step, not done at HEAD: {verdict.why}")
    print("The scheduler does not check the tag or the tables (X-registration): no pilot run before they are done.")
    return EXIT_BLOCKED


# -- 2. env (task 8; Table 3.1; Appendix A) ------------------------------------


def env_verdict(head: Head, args: argparse.Namespace) -> Verdict:
    missing = [rel for rel in (LOCK, WORKSTATION) if head.file(rel) is None]
    if not missing:
        return Verdict(DONE, f"{LOCK} and {WORKSTATION} are committed")
    note = f"; {LOCK} is in the working tree, not committed" if written(LOCK) else ""
    return Verdict(TODO, "not committed at HEAD: " + ", ".join(missing) + note)


def env_run(head: Head, args: argparse.Namespace) -> int:
    if written(LOCK):
        print(f"refused: {LOCK} exists already. The pilot owner resolves the pinned environment once (task 8); "
              f"`bash {SETUP_SCRIPT} --locked` is for the others (it installs exactly the lock), and re-resolving "
              "it (`--relock`) needs an amendment.")
        if any(head.file(rel) is None for rel in (LOCK, WORKSTATION)):
            print_commit([LOCK, WORKSTATION], ENV_MESSAGE)
        return EXIT_BLOCKED
    code = call(["bash", SETUP_SCRIPT])
    if code:
        return code
    print_commit([LOCK, WORKSTATION], ENV_MESSAGE)
    print("Then run every later step in the pinned environment: source .venv/bin/activate")
    return EXIT_OK


# -- 3. allocation (task 12; Part 6 G2; Table 8.1; X-allocation) ---------------


def allocation_verdict(head: Head, args: argparse.Namespace) -> Verdict:
    if head.file(ALLOCATION) is not None:
        record = head.json(ALLOCATION)
        shown = (f": {record.get('machine_hours_allocated')} machine-hours, decided by {record.get('decided_by')} "
                 f"on {record.get('meeting_date')}") if isinstance(record, Mapping) else ""
        return Verdict(DONE, f"{ALLOCATION} is committed{shown}")
    if written(ALLOCATION):
        return Verdict(TODO, f"{ALLOCATION} is written, not committed")
    return Verdict(TODO, "the group decides the machine-hours and their basis (Part 6 G2; Table 8.1); then --hours H "
                         "--basis TEXT --date YYYY-MM-DD")


def allocation_run(head: Head, args: argparse.Namespace) -> int:
    if written(ALLOCATION):
        print(f"{ALLOCATION} exists already (recorded once; changed only by amendment).")
        print_commit([ALLOCATION], "Machine-hour allocation (HANDOVER task 12; Part 6 G2; Table 8.1)")
        return EXIT_OK
    hours, basis, date = getattr(args, "hours", None), getattr(args, "basis", None), getattr(args, "date", None)
    if hours is None or not basis:
        print(allocation_decision())
        return EXIT_USAGE
    if date is None:
        print("error: --date YYYY-MM-DD, the date of the group's decision, is needed with --hours and --basis")
        return EXIT_USAGE
    by = getattr(args, "by", None) or DEFAULT_BY
    options = ["--confirm"] if getattr(args, "confirm", False) else []
    for root in getattr(args, "smoke_root", None) or ():
        options += ["--smoke-root", root]
    code = call(python("-m", "pilot", "allocate", "--hours", number(hours), "--basis", basis, "--by", by,
                       "--date", date, *options))
    if code:
        return code
    print_commit([ALLOCATION], f"Machine-hour allocation: {number(hours)} machine-hours, decided by {by} on {date} "
                              "(HANDOVER task 12; Part 6 G2; Table 8.1)")
    print(f"Commit {ALLOCATION} before the determinism checks and any smoke run on the workstation (X-allocation); "
          "enter the same value in Table 8.1 with an amendment row in Table 9.1 before any pilot run (task 12).")
    return EXIT_OK


def allocation_decision() -> str:
    """What the group decides (``ALLOCATION_DECISION``), with G2's run-equivalents as pilot.budget gives them."""
    return ALLOCATION_DECISION.format(corrected=budget.corrected_run_equivalents()["total"],
                                      registered=budget.registered_run_equivalents(),
                                      confirm_above=budget.ALLOCATION_CONFIRM_ABOVE)


# -- 4. fixed-batch (task 8a; Table 2.3; Appendix A; Q-appendix-a) -------------


def manifest_hashes(manifest_value: Any) -> dict[str, str] | None:
    """{task: sha256} of a MANIFEST.json of metrics/collect_fixed_batch.py (None: none, or malformed)."""
    tasks = manifest_value.get("tasks") if isinstance(manifest_value, Mapping) else None
    if not isinstance(tasks, Mapping) or not tasks:
        return None
    hashes: dict[str, str] = {}
    for task, entry in tasks.items():
        sha = entry.get("sha256") if isinstance(entry, Mapping) else None
        if not isinstance(sha, str):
            return None
        hashes[str(task)] = sha
    return hashes


def package_versions() -> dict[str, str | None]:
    """The installed version of each of FIXED_BATCH_PACKAGES in this interpreter (None: not installed)."""
    versions: dict[str, str | None] = {}
    for name in FIXED_BATCH_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def canonical(name: str) -> str:
    """A package name as pip compares them (PEP 503: case and runs of ``-``, ``_`` and ``.`` do not matter)."""
    return re.sub(r"[-_.]+", "-", name.strip()).lower()


def unlocked_versions(lock: bytes) -> list[str]:
    """Each of FIXED_BATCH_PACKAGES whose version in this interpreter is not the one ``lock`` pins (its
    ``name==version`` lines, as ``pip freeze`` writes them), with both versions; empty: this is the locked stack."""
    pins: dict[str, str] = {}
    for line in lock.decode("utf-8", errors="replace").splitlines():
        name, sep, version = line.split("#", 1)[0].partition("==")
        if sep:
            pins[canonical(name)] = version.strip()
    installed = package_versions()
    return [f"{name} {installed[name] or 'not installed'} (the lock: {pins.get(canonical(name), 'not pinned')})"
            for name in FIXED_BATCH_PACKAGES if installed[name] is None or installed[name] != pins.get(canonical(name))]


def not_reproduced(record: Mapping[str, Any]) -> list[str] | None:
    """The tasks a recorded mismatch names (``fixed_batch_run``), or None when the record is not one."""
    tasks = record.get("not_reproduced")
    if record.get("passed") is not False or not isinstance(tasks, list) or not tasks \
            or not all(isinstance(t, str) for t in tasks):
        return None
    return list(tasks)


def fixed_batch_verdict(head: Head, args: argparse.Namespace) -> Verdict:
    expected = manifest_hashes(head.json(FIXED_BATCH_MANIFEST))
    if head.file(FIXED_BATCH_CHECK) is None:
        note = "; written in the working tree, not committed" if written(FIXED_BATCH_CHECK) else ""
        return Verdict(TODO, f"{FIXED_BATCH_CHECK} is not committed at HEAD{note}")
    record = head.json(FIXED_BATCH_CHECK)
    if not isinstance(record, Mapping):
        return Verdict(BLOCKED, f"{FIXED_BATCH_CHECK} at HEAD is not a JSON object: stop and raise it with the group "
                                "(task 8a)")
    differs = not_reproduced(record)
    if record.get("passed") is not True and differs is None:
        return Verdict(BLOCKED, f"{FIXED_BATCH_CHECK} at HEAD records no outcome of the check: stop and raise it "
                                "with the group (task 8a)")
    if expected is None:
        return Verdict(BLOCKED, f"{FIXED_BATCH_CHECK} is committed but {FIXED_BATCH_MANIFEST} is not: commit "
                                f"{FIXED_BATCH_DIR}/ with it")
    if record.get("manifest") != expected:
        return Verdict(BLOCKED, f"{FIXED_BATCH_CHECK} records other batch hashes than {FIXED_BATCH_MANIFEST} at "
                                "HEAD: a changed batch needs an amendment and a new check; raise it with the group")
    if differs is not None:
        return Verdict(DONE, f"the check ran at {record.get('commit')} for {len(expected)} batches; the committed "
                             "batches are the fixed batches (Table 9.1, Q-appendix-a)",
                       warning=f"the re-collection on the workstation differs for {', '.join(differs)}: report it in "
                               "Table 9.1 (Q-appendix-a); the committed batches are not replaced")
    return Verdict(DONE, f"the check passed at {record.get('commit')} for {len(expected)} batches")


def read_check_result(path: Path) -> dict[str, Any] | None:
    """The result ``python -m metrics.collect_fixed_batch --check --json PATH`` wrote (None: none, or malformed)."""
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    tasks = result.get("tasks") if isinstance(result, dict) else None
    if not isinstance(tasks, dict) or not tasks or not all(isinstance(t, dict) for t in tasks.values()):
        return None
    return result


def fixed_batch_run(head: Head, args: argparse.Namespace) -> int:
    if head.file(FIXED_BATCH_CHECK) is not None:  # written once, and not done (cmd_step): the verdict says why
        print(f"refused: {FIXED_BATCH_CHECK} is committed at HEAD already (written once, never checked again): "
              f"{fixed_batch_verdict(head, args).why}")
        return EXIT_BLOCKED
    if written(FIXED_BATCH_CHECK):
        print(f"{FIXED_BATCH_CHECK} exists already (written once, after the check); not checked again.")
        try:
            record = json.loads((REPO / FIXED_BATCH_CHECK).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            record = None
        mismatch = isinstance(record, Mapping) and not_reproduced(record) is not None
        print_commit([FIXED_BATCH_DIR + "/", FIXED_BATCH_CHECK],
                     FIXED_BATCH_MISMATCH_MESSAGE if mismatch else FIXED_BATCH_MESSAGE)
        return EXIT_OK
    lock = head.file(LOCK)
    if lock is None:
        print(f"refused: {LOCK} is not committed at HEAD; the check runs in the workstation's pinned environment "
              "(the env step, task 8, comes first)")
        return EXIT_BLOCKED
    unlocked = unlocked_versions(lock)
    if unlocked:  # the record names HEAD's lock as the stack that re-collected the batches
        print(f"refused: {sys.executable} does not run the stack {LOCK} at HEAD pins: {'; '.join(unlocked)}. The "
              "check runs in the pinned environment (source .venv/bin/activate; HANDOVER task 8a); nothing was "
              "written")
        return EXIT_BLOCKED
    manifest_path = REPO / FIXED_BATCH_MANIFEST
    before = manifest_path.read_bytes() if manifest_path.is_file() else None
    try:
        hashes = manifest_hashes(json.loads(before)) if before is not None else None
    except ValueError:
        hashes = None
    if hashes is None:
        print(f"refused: {FIXED_BATCH_MANIFEST} is missing or lists no batch; the batches are Role 3's (task 4a)")
        return EXIT_BLOCKED
    # The record names HEAD as the code that checked: require_clean_worktree ignores untracked files, so the
    # checking code (metrics/batches.py holds FIXED_BATCH_SHA256) must be HEAD's as it is.
    uncommitted = [rel for rel in sorted(files_in("metrics", "*.py")) if head.file(rel) != (REPO / rel).read_bytes()]
    if uncommitted:
        print("refused: the check would run code that HEAD does not hold as it is: " + ", ".join(uncommitted)
              + " (Role 3 adopts metrics/ by its pull request, task 4a)")
        return EXIT_BLOCKED
    try:
        commit = provenance.require_clean_worktree(REPO)
    except provenance.DirtyWorktreeError as exc:
        print(f"refused: {exc}")
        return EXIT_BLOCKED
    with tempfile.TemporaryDirectory(prefix="fixed-batch-check-") as tmp:  # outside the tree: it stays clean
        result_path = Path(tmp) / "check.json"
        code = call(python(*FIXED_BATCH_COMMAND, "--json", str(result_path)))
        result = read_check_result(result_path)
    tasks = result["tasks"] if result is not None else {}
    consistent = result is not None and set(tasks) == set(hashes) and all(
        entry.get("committed_consistent") is True and entry.get("committed_sha256") == hashes[task]
        for task, entry in tasks.items())
    if code not in (0, FIXED_BATCH_MISMATCH) or not consistent or (code == 0) != (result.get("passed") is True):
        print(f"STOP: the fixed-batch check did not run to a result, or the committed records disagree (exit {code}); "
              "nothing was written: raise it with the group (HANDOVER task 8a). The SHA-256 constants in "
              "metrics/batches.py and every plasticity value depend on these bytes.")
        return code or EXIT_BLOCKED
    if provenance.commit_hash(REPO) != commit or manifest_path.read_bytes() != before:
        print(f"refused: HEAD or {FIXED_BATCH_MANIFEST} changed while the check ran; nothing was written: run the "
              "step again")
        return EXIT_BLOCKED
    differs = sorted(task for task, entry in tasks.items() if entry.get("reproduced") is not True)
    record: dict[str, Any] = {
        "passed": code == 0,
        "commit": commit,
        "checked_at": provenance.utc_now().isoformat(),
        "python": platform.python_version(),
        "packages": package_versions(),
        "manifest": hashes,
        "recollected_sha256": {task: entry.get("recollected_sha256") for task, entry in tasks.items()},
        "lock_sha256": hashlib.sha256(lock).hexdigest(),
        "command": "python " + " ".join(FIXED_BATCH_COMMAND),
        "registered": "Table 2.3 'Fixed evaluation batch'; Appendix A; HANDOVER.md task 8a (Q-appendix-a)",
    }
    if code:
        record.update(not_reproduced=differs, problems=list(result.get("problems") or []), reading=FIXED_BATCH_READING)
    provenance.write_text_once(REPO / FIXED_BATCH_CHECK, json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"wrote {FIXED_BATCH_CHECK}")
    if code:
        shown = "; ".join(f"{task}: committed {hashes[task]}, re-collected {record['recollected_sha256'][task]}"
                          for task in differs)
        print(f"WARNING: the re-collection on the workstation differs from the committed batches ({shown}). The "
              "committed batches stay the fixed batches (Table 2.3; Table 9.1, Q-appendix-a); nothing was "
              "rewritten. Report it in Table 9.1:")
        print(f"Table 9.1 row: Fixed-batch reproducibility check | {record['checked_at'][:10]} | <the commit that adds "
              f"{FIXED_BATCH_CHECK}> | the re-collection on the workstation differs for {', '.join(differs)} "
              f"({shown}); the committed batches remain the fixed batches | Table 2.3; Appendix A (Q-appendix-a) | "
              "the group")
    message = FIXED_BATCH_MISMATCH_MESSAGE if code else FIXED_BATCH_MESSAGE
    print_commit([FIXED_BATCH_DIR + "/", FIXED_BATCH_CHECK], message)
    return EXIT_OK


# -- 5. determinism-default (task 9) and 6. determinism (task 12a; Table 3.1) --


def default_report_passed(report: Any) -> bool:
    """True for a passing report of ``python scripts/determinism_check.py`` (default form, plug-in ppolag):
    not a rehearsal, from a clean tree, and both runs trained on the commit it names."""
    if not isinstance(report, Mapping):
        return False
    result, commit = report.get("result"), report.get("commit_hash")
    return (report.get("check") == DEFAULT_FORM and report.get("plugin", DEFAULT_FORM_PLUGIN) == DEFAULT_FORM_PLUGIN
            and report.get("passed") is True and report.get("rehearsal_only") is False
            and report.get("allow_pending") is not True and report.get("worktree_dirty") is False
            and provenance.is_full_commit_hash(commit) and report.get("run_commits") == [commit, commit]
            and isinstance(result, Mapping) and result.get("identical") is True)


def failed_check(report: Any, form: str, plugin: str, commit: str | None) -> bool:
    """True for a determinism check of ``form`` and ``plugin`` that ran on ``commit`` and did not pass: not a
    rehearsal, and not explained by HEAD or the tree changing while it ran (``same_commit`` false), so its runs or
    their configurations differed (Table 3.1)."""
    return (isinstance(report, Mapping) and commit is not None and report.get("check") == form
            and report.get("plugin", DEFAULT_FORM_PLUGIN) == plugin and report.get("passed") is False
            and report.get("rehearsal_only") is False and report.get("same_commit") is not False
            and report.get("commit_hash") == commit)


def failed_checks(head: Head, form: str, plugins: Sequence[str]) -> dict[str, str]:
    """{plug-in: path} of the working tree's determinism reports of ``form`` that failed on HEAD's commit
    (``failed_check``): a failure is raised with the group, never retried on the same commit."""
    found: dict[str, str] = {}
    for rel in sorted(files_in(S.DETERMINISM_DIR, "determinism-*.json")):
        try:
            report = json.loads((REPO / rel).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for plugin in plugins:
            if failed_check(report, form, plugin, head.commit):
                found.setdefault(plugin, rel)
    return found


def failed_text(failed: Mapping[str, str]) -> str:
    """What a failed determinism check (``failed_checks``) holds, and what releases it."""
    return ("the determinism check failed on HEAD's commit (" + "; ".join(f"{p}: {rel}" for p, rel in failed.items())
            + "): raise it with the group (Table 3.1); it is run again only on a later commit (the group's fix, or "
              "the commit that records the failed report)")


def report_files() -> set[str]:
    """The determinism reports (.json and .md) in the working tree's ledger/determinism/."""
    return files_in(S.DETERMINISM_DIR, "determinism-*.json") | files_in(S.DETERMINISM_DIR, "determinism-*.md")


def with_markdown(rel: str) -> list[str]:
    """A report's .md (when written) and .json, as the script writes them."""
    md = rel[: -len(".json")] + ".md"
    return [md, rel] if written(md) else [rel]


def uncommitted_reports(head: Head) -> dict[str, Any]:
    """{path: report} of the determinism reports in the working tree that HEAD does not hold as they are."""
    reports: dict[str, Any] = {}
    for rel in sorted(files_in(S.DETERMINISM_DIR, "determinism-*.json")):
        data = (REPO / rel).read_bytes()
        if head.file(rel) == data:
            continue
        try:
            reports[rel] = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            reports[rel] = None
    return reports


def uncommitted_plugin_reports(head: Head) -> dict[str, str]:
    """{plug-in: path} of the passing registered-form reports written but not committed (the gate's own test)."""
    found: dict[str, str] = {}
    for rel, report in uncommitted_reports(head).items():
        plugin = S.determinism_report_plugin(report)
        if plugin is not None:
            found.setdefault(plugin, rel)
    return found


def determinism_waiting(plugin: str) -> tuple[str, ...]:
    """The open questions that hold the registered-form check of ``plugin``, before anything is trained.

    Those of scripts/determinism_check.py ``waiting_on``: the arm's own run gates (the spec's open
    questions), the check's own key (``pilot.scheduler.determinism_open_keys``, study_a_pid's changed
    onset; Table 9.1, Q-determinism-late-onset) and, for the budget-conditioned plug-in, Q-studyb-eval,
    which gates its evaluation.
    """
    check = _script("determinism_check")  # its ``waiting_on`` is the refusal it applies itself
    return tuple(check.waiting_on(check.make_spec(R.CHECKPOINT_INTERVAL_STEPS, plugin), True))


def selected_plugins(args: argparse.Namespace) -> tuple[str, ...]:
    """The plug-ins of the determinism step (``--plugins``; default all), in DETERMINISM_ORDER."""
    chosen = getattr(args, "plugins", None) or DETERMINISM_ORDER
    return tuple(p for p in DETERMINISM_ORDER if p in chosen)


def determinism_default_verdict(head: Head, args: argparse.Namespace) -> Verdict:
    found = [p for p in head.files(S.DETERMINISM_DIR) if p.endswith(".json") and default_report_passed(head.json(p))]
    if found:
        return Verdict(DONE, f"a passing default-form report is committed: {found[0]}")
    failed = failed_checks(head, DEFAULT_FORM, (DEFAULT_FORM_PLUGIN,))
    if failed:
        return Verdict(BLOCKED, failed_text(failed))
    pending = [rel for rel, report in uncommitted_reports(head).items() if default_report_passed(report)]
    note = f"; written, not committed: {pending[0]}" if pending else ""
    return Verdict(TODO, f"no passing default-form report of plug-in ppolag committed under {S.DETERMINISM_DIR}/{note}")


def allocation_missing(head: Head) -> bool:
    """True (and says so) while HEAD lacks the allocation: a determinism check reveals the workstation's speed."""
    if head.file(ALLOCATION) is not None:
        return False
    print(f"refused: {ALLOCATION} is not committed at HEAD; the allocation is recorded before the determinism "
          "checks, which already reveal the workstation's speed (X-allocation, Table 9.1; Part 6 G2): run the "
          "allocation step first")
    return True


def determinism_default_run(head: Head, args: argparse.Namespace) -> int:
    pending = [rel for rel, report in uncommitted_reports(head).items() if default_report_passed(report)]
    if pending:
        print(f"a passing default-form report is written already: {pending[0]}; not checked again.")
        print_commit(with_markdown(pending[0]), DEFAULT_DETERMINISM_MESSAGE)
        return EXIT_OK
    failed = failed_checks(head, DEFAULT_FORM, (DEFAULT_FORM_PLUGIN,))
    if failed:
        print(f"refused: {failed_text(failed)}")
        return EXIT_BLOCKED
    if allocation_missing(head):
        return EXIT_BLOCKED
    before = report_files()
    code = call(python(DETERMINISM_SCRIPT))
    new = sorted(report_files() - before)
    if code != 0:
        shown = f"; its report is written: {', '.join(new)}" if new else "; no report was written (see above)"
        print(f"the determinism check did not pass (exit {code}){shown}. Stop and raise it with the group (Table "
              "3.1) before any other step.")
        return code
    print_commit(new or [S.DETERMINISM_DIR + "/"], DEFAULT_DETERMINISM_MESSAGE)
    return EXIT_OK


def determinism_verdict(head: Head, args: argparse.Namespace) -> Verdict:
    plugins = selected_plugins(args)
    have = head.determinism_plugins()
    missing = [p for p in plugins if p not in have]
    if not missing:
        return Verdict(DONE, "passing registered-form reports committed: " + ", ".join(plugins))
    failed = failed_checks(head, S.DETERMINISM_REGISTERED_FORM, missing)
    if failed:
        return Verdict(BLOCKED, failed_text(failed))
    uncommitted = uncommitted_plugin_reports(head)
    waiting = {p: determinism_waiting(p) for p in missing if p not in uncommitted}
    parts = [f"{p} (written, not committed: {uncommitted[p]})" if p in uncommitted
             else f"{p} (waiting on {', '.join(waiting[p])})" if waiting[p] else f"{p} (can run now)"
             for p in missing]
    why = "missing: " + "; ".join(parts)
    if all(waiting.get(p) for p in missing):
        # Every missing check is refused: the keys are the group's (task 11). The scheduler's determinism_missing
        # gate holds each plug-in's runs, so only study_a's report holds the pilot's Study A runs.
        return Verdict(BLOCKED, why + "; answered by an amendment (task 11)",
                       skippable=PILOT_STUDY_A_PLUGIN not in missing)
    return Verdict(TODO, why)


def determinism_run(head: Head, args: argparse.Namespace) -> int:
    if allocation_missing(head):
        return EXIT_BLOCKED
    have = head.determinism_plugins()
    failed = failed_checks(head, S.DETERMINISM_REGISTERED_FORM, [p for p in selected_plugins(args) if p not in have])
    if failed:
        print(f"refused: {failed_text(failed)}")
        return EXIT_BLOCKED
    uncommitted = uncommitted_plugin_reports(head)
    to_commit: list[str] = []
    checked: list[str] = []
    refused: dict[str, tuple[str, ...]] = {}
    for plugin in selected_plugins(args):
        if plugin in have:
            print(f"{plugin}: done ({have[plugin]})")
            continue
        if plugin in uncommitted:
            print(f"{plugin}: a passing report is written already, not committed: {uncommitted[plugin]}")
            to_commit += with_markdown(uncommitted[plugin])
            checked.append(plugin)
            continue
        keys = determinism_waiting(plugin)
        if keys:
            print(f"{plugin}: refused before anything is trained: the check waits on {', '.join(keys)} "
                  "(configs/registered.py PENDING; answered by an amendment, HANDOVER task 11)")
            refused[plugin] = keys
            continue
        before = report_files()
        code = call(python(DETERMINISM_SCRIPT, "--registered-form", "--plugin", plugin))
        new = sorted(report_files() - before)
        if code != 0:
            shown = f"; its report is written: {', '.join(new)}" if new else "; no report was written (see above)"
            print(f"{plugin}: the determinism check did not pass (exit {code}){shown}. Stop and raise it with the "
                  "group (Table 3.1).")
            if to_commit:
                print("The reports that passed before it:")
                print_commit(to_commit, determinism_message(checked))
            return code
        to_commit += new
        checked.append(plugin)
    if to_commit:
        print_commit(to_commit, determinism_message(checked))
    if refused:
        print("The scheduler holds the runs of the refused plug-ins (determinism_missing) until their reports are "
              "committed.")
        return EXIT_BLOCKED
    return EXIT_OK


def determinism_message(plugins: Sequence[str]) -> str:
    return f"Determinism checks, registered form: {', '.join(plugins)} (HANDOVER task 12a; Table 3.1)"


# -- 7. search (task 16; Part 7.2; Appendix C): a people's step -----------------


def search_verdict(head: Head, args: argparse.Namespace) -> Verdict:
    released, reason = head.search_record()
    if released:
        return Verdict(DONE, reason)
    # The scheduler holds only the Study B runs (search_record_missing), so the Study A pilot runs go on.
    return Verdict(BLOCKED, f"a people's step: {reason}; the scheduler holds the pilot's two Study B runs until "
                            f"{SEARCH_RELEASE}", skippable=True)


def search_run(head: Head, args: argparse.Namespace) -> int:
    released, reason = head.search_record()
    if released:
        print(f"search: done: {reason}")
        return EXIT_OK
    print(f"search (HANDOVER task 16) is a people's step, not done at HEAD: {reason}")
    print(SEARCH_INSTRUCTIONS + ".")
    print(f"The scheduler holds the pilot's two Study B runs (search_record_missing) until {SEARCH_RELEASE}; the six "
          "Study A pilot runs do not wait for it. What the record in the working tree still lacks:")
    call(python("-m", "studyb.search", "check"))  # read-only (it never writes); the step stays blocked either way
    return EXIT_BLOCKED


# -- 8. pilot-runs (task 20; Part 3.6) -----------------------------------------


def pilot_run_ids() -> list[str]:
    """The run_ids of the eight pilot runs (``manifest.design("pilot")``)."""
    return [spec.run_id for spec in manifest.design("pilot")]


def read_runs(data_root: str | Path) -> dict[str, dict[str, Any]] | None:
    """{run_id: status, spec (JSON), replaced_by and ledger_written} of a data root's scheduler state, opened
    read-only (None: none).

    sqlite3.Error when the file is not the scheduler's state (``pilot.enrichment.read_scheduler_state``
    reads it the same way).
    """
    path = S.SchedulerConfig(data_root=Path(data_root)).state_path
    if not path.is_file():
        return None
    con = sqlite3.connect(f"{Path(os.path.abspath(path)).as_uri()}?mode=ro", uri=True, timeout=60.0)
    try:
        rows = con.execute("SELECT run_id, status, spec, replaced_by, ledger_written FROM runs "
                           "ORDER BY rowid").fetchall()
    finally:
        con.close()
    return {r[0]: {"status": r[1], "spec": r[2], "replaced_by": r[3], "ledger_written": r[4]} for r in rows}


def held_reasons(head: Head, rows: Sequence[Mapping[str, Any]]) -> str | None:
    """Why every one of ``rows`` (unfinished pilot runs) waits on a fact of HEAD, or None when one can move.

    The launch gates of pilot/scheduler.py that a people's step releases: the allocation, the
    determinism report of the run's plug-in, the search record (Study B) and the run's open questions.
    ``schedule status`` gives the scheduler's own reasons.
    """
    if any(row["status"] != "pending" for row in rows):
        return None
    allocation = head.file(ALLOCATION) is not None
    reports = head.determinism_plugins()
    search: bool | None = None
    reasons: Counter[str] = Counter()
    for row in rows:
        spec = RunSpec.from_json(row["spec"])
        why = [] if allocation else [f"{ALLOCATION} not committed"]
        if spec.plugin in S.DETERMINISM_PLUGINS and spec.plugin not in reports:
            why.append(f"determinism_missing ({spec.plugin})")
        if spec.study == "B":
            search = head.search_record()[0] if search is None else search
            if not search:
                why.append("search_record_missing")
        if spec.open_questions:
            why.append("waiting on " + ", ".join(spec.open_questions))
        if not why:
            return None
        reasons.update(why)
    return "; ".join(f"{why} ({count})" for why, count in reasons.items())


def pilot_runs_verdict(head: Head, args: argparse.Namespace) -> Verdict:
    data_root = data_root_of(args)
    try:
        runs = read_runs(data_root)
    except sqlite3.Error as exc:
        return Verdict(BLOCKED, f"cannot read the scheduler state of {data_root} ({type(exc).__name__}: {exc})")
    expected = pilot_run_ids()
    if runs is None:
        return Verdict(TODO, f"no scheduler state on {data_root}: the {len(expected)} pilot runs are not queued yet")
    queued = [r for r in expected if r in runs]
    if len(queued) < len(expected):
        return Verdict(TODO, f"{len(queued)} of the {len(expected)} pilot runs are queued on {data_root}")
    prefix = manifest.pilot_prefix(0)
    pilot = {run_id: row for run_id, row in runs.items() if run_id.startswith(prefix)}  # replacements included
    counts = ", ".join(f"{status} {n}" for status, n in sorted(Counter(r["status"] for r in pilot.values()).items()))
    # as Scheduler.status_report lists them: an exclusion without its replacement or not yet in the ledger
    decisions = [run_id for run_id, row in pilot.items() if row["status"] in DECISION_STATUSES
                 or (row["status"] == "excluded" and not (row["replaced_by"] and row["ledger_written"]))]
    if decisions:
        return Verdict(BLOCKED, f"{len(decisions)} pilot runs wait for a decision (e.g. {decisions[0]}): see "
                                f"`python -m pilot schedule status --data-root {data_root}` ({counts})")
    unfinished = [run_id for run_id, row in pilot.items() if row["status"] not in FINISHED_STATUSES]
    if not unfinished:
        return Verdict(DONE, f"all {len(pilot)} pilot runs finished on {data_root} ({counts})")
    held = held_reasons(head, [pilot[r] for r in unfinished])
    if held:
        return Verdict(BLOCKED, f"{len(unfinished)} pilot runs are held: {held} ({counts})")
    return Verdict(TODO, f"{len(pilot) - len(unfinished)} of {len(pilot)} pilot runs finished on {data_root} "
                         f"({counts})")


def pilot_runs_run(head: Head, args: argparse.Namespace) -> int:
    data_root = data_root_of(args)
    concurrent = getattr(args, "max_concurrent", None)
    if concurrent is None or concurrent < 1:
        print(f"pilot-runs needs --max-concurrent K (at least 1), the runs executed at once ([P], measured in the "
              f"pilot): usually the physical cores minus 1, here {suggested_concurrency()} "
              f"({physical_cores()} physical cores):")
        print(f"python scripts/workstation.py pilot-runs --max-concurrent {suggested_concurrency()} "
              f"--data-root {shlex.quote(data_root)}")
        return EXIT_USAGE
    try:
        runs = read_runs(data_root)
    except sqlite3.Error as exc:
        print(f"error: cannot read the scheduler state of {data_root} ({type(exc).__name__}: {exc})")
        return EXIT_BLOCKED
    expected = pilot_run_ids()
    if runs is not None and all(r in runs for r in expected):
        print(f"the {len(expected)} pilot runs are queued on {data_root} already: not queued again")
    else:  # Scheduler.add skips a run_id queued with the same spec and refuses one with another spec
        code = call(python("-m", "pilot", "schedule", "add", "--design", "pilot", "--data-root", data_root))
        if code:
            return code
    print("`schedule run` runs for days: keep it in tmux or under nohup. Ctrl-C stops it (exit 130); running this "
          "step again resumes (live runs are adopted).")
    code = call(python("-m", "pilot", "schedule", "run", "--max-concurrent", str(concurrent), "--data-root", data_root))
    if code:
        return code
    code = call(python("-m", "pilot", "schedule", "status", "--data-root", data_root))
    if code:
        return code
    verdict = judge(STEP_BY_NAME["pilot-runs"], read_head(), args)
    if verdict.state == DONE:
        print(f"pilot-runs: {verdict.why}. Next: python scripts/workstation.py pilot-enrich --data-root "
              f"{shlex.quote(data_root)}")
        return EXIT_OK
    print(f"pilot-runs: not finished: {verdict.why}")
    return EXIT_BLOCKED


# -- 9. pilot-enrich (tasks 21 to 23) ------------------------------------------


def pilot_enrich_verdict(head: Head, args: argparse.Namespace) -> Verdict:
    missing = [rel for rel in (PILOT_LEDGER, G4_FILE) if head.file(rel) is None]
    if not missing:
        return Verdict(DONE, f"{PILOT_LEDGER} and {G4_FILE} are committed")
    present = [rel for rel in missing if written(rel)]
    note = f"; in the working tree: {', '.join(present)}" if present else ""
    return Verdict(TODO, "not committed at HEAD: " + ", ".join(missing) + note)


def pilot_enrich_run(head: Head, args: argparse.Namespace) -> int:
    data_root = data_root_of(args)
    if not written(PILOT_LEDGER):
        print(f"refused: {PILOT_LEDGER} does not exist yet: the pilot runs write it (pilot-runs, task 20)")
        return EXIT_BLOCKED
    commands = [python("-m", "pilot", "enrich", action, "--ledger", PILOT_LEDGER, "--data-root", data_root)
                for action in ENRICH_ACTIONS]
    if written(G4_FILE):
        print(f"{G4_FILE} exists already (the G4 measurement is taken once): not measured again")
    else:
        commands.append(python("-m", "pilot", "g4-measure", "--data-root", data_root))
    for cmd in commands:
        code = call(cmd)
        if code:
            print(f"the chain stopped (exit {code}; 3: refused, a refusing step or an open question, see above). What "
                  "the earlier commands wrote stays (an existing value is never changed); fix the cause and run this "
                  "step again.")
            return code
    print_commit([PILOT_RESULTS + "/"], ENRICH_MESSAGE)
    return EXIT_OK


# -- 10. go (tasks 24 to 26; Part 6; Part 5.8; Part 5.5) -------------------------


def go_revision(report: Any) -> int | None:
    """The pilot revision a go report records, read as ``pilot.enrichment.list_go_reports`` reads it for ``python -m
    pilot freeze`` (None: unreadable). A copy: pilot.enrichment imports the ledger writer's libraries, and this
    script runs with the standard library before the env step."""
    generated = report.get("generated") if isinstance(report, Mapping) else None
    if not isinstance(generated, Mapping) or not generated:
        return None
    revision = generated.get("revision")
    if isinstance(revision, int):
        return revision
    argv = generated.get("argv")
    return 1 if isinstance(argv, list) and "--repilot" in argv else 0


def worktree_go_reports() -> list[str]:
    """The go reports of the pilot (revision 0) in the working tree's results/pilot/."""
    found = []
    for rel in sorted(files_in(PILOT_RESULTS, "go_report-*.json")):
        try:
            report = json.loads((REPO / rel).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if go_revision(report) == 0:
            found.append(rel)
    return found


def worktree_pilot_analyses() -> list[str]:
    """The output directories of pilot-mode analyses in the working tree's results/analysis/."""
    root = REPO / ANALYSIS_DIR
    if not root.is_dir():
        return []
    return sorted(f"{ANALYSIS_DIR}/{p.parent.name}" for p in root.glob("pilot-*/analysis.json"))


def analysis_failure(value: Any) -> tuple[int, str] | None:
    """(exit status, why) when a pilot-mode analysis.json records a test run that did not pass, as ``python -m
    analysis`` judged it after writing it: 3 its go-report check found a mismatch, 4 it compared nothing
    (analysis/__main__.py; ``pilot.mismatches`` and ``pilot.compared``, analysis/pilot_check.py); None: it passed."""
    pilot = value.get("pilot") if isinstance(value, Mapping) else None
    mismatches = pilot.get("mismatches") if isinstance(pilot, Mapping) else None
    compared = pilot.get("compared") if isinstance(pilot, Mapping) else None
    if not isinstance(mismatches, list) or isinstance(compared, bool) or not isinstance(compared, int):
        return EXIT_BLOCKED, "its check against the go report is missing or unreadable"
    if mismatches:
        return 3, f"{ANALYSIS_EXITS[3]} ({'; '.join(str(m) for m in mismatches)})"
    if compared < 1:
        return 4, ANALYSIS_EXITS[4]
    return None


def go_verdict(head: Head, args: argparse.Namespace) -> Verdict:
    reports = [p for p in head.files(PILOT_RESULTS) if GO_REPORT.match(p) and go_revision(head.json(p)) == 0]
    analyses = sorted(p for p in head.files(ANALYSIS_DIR) if PILOT_ANALYSIS.match(p))
    failed = analysis_failure(head.json(analyses[-1])) if analyses else None
    if failed:  # the latest test run (its UTC stamp) did not pass: Part 5.8, the error is fixed, then run again
        return Verdict(BLOCKED, f"the analysis test run {analyses[-1].rsplit('/', 1)[0]}/ at HEAD did not pass (exit "
                                f"{failed[0]}: {failed[1]}): fix the error (Part 5.8; recorded in Part 9), then python "
                                f"-m analysis --mode pilot --ledger {PILOT_LEDGER}")
    if reports and analyses:
        return Verdict(DONE, f"{reports[0]} and the analysis test run {analyses[0].rsplit('/', 1)[0]}/ are committed")
    missing = ([] if reports else ["a go report of the pilot"]) + ([] if analyses else ["the analysis test run"])
    return Verdict(TODO, "not committed at HEAD: " + " and ".join(missing))


def go_run(head: Head, args: argparse.Namespace) -> int:
    hours_per_run, concurrent = getattr(args, "hours_per_run", None), getattr(args, "concurrent", None)
    if (hours_per_run is None) != (concurrent is None):
        print("error: the surplus-seed plan (Part 5.5; task 26) needs both --hours-per-run H and --concurrent K")
        return EXIT_USAGE
    reports = worktree_go_reports()
    if reports:
        print(f"a go report of the pilot exists already ({', '.join(reports)}): not written again "
              "(`python -m pilot freeze` reads the earliest committed one)")
    else:
        code = call(python("-m", "pilot", "go"))
        if code:
            return code
    analyses = worktree_pilot_analyses()
    if analyses:
        try:
            latest = json.loads((REPO / analyses[-1] / "analysis.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            latest = None
        failed = analysis_failure(latest)
        if failed:
            print(f"the analysis test run {analyses[-1]}/ did not pass (exit {failed[0]}: {failed[1]}): Part 5.8, fix "
                  "the error (recorded in Part 9), then run it again: python -m analysis --mode pilot --ledger "
                  f"{PILOT_LEDGER}. Nothing more is run.")
            return failed[0]
        print(f"the analysis test run exists already ({', '.join(analyses)}): Part 5.8 runs it once on the pilot data; "
              "not run again (after fixing an error it found: python -m analysis --mode pilot --ledger "
              f"{PILOT_LEDGER})")
    else:
        code = call(python("-m", "analysis", "--mode", "pilot", "--ledger", PILOT_LEDGER))
        if code:
            meaning = ANALYSIS_EXITS.get(code, "")
            print(f"the analysis test run exited {code}" + (f": {meaning}" if meaning else ""))
            return code
    if hours_per_run is not None:
        code = call(python("-m", "pilot", "surplus", "--hours-per-run", number(hours_per_run),
                           "--concurrent", str(concurrent)))
        if code:
            return code
    else:
        print("For the surplus-seed plan of the go meeting (Part 5.5; task 26), with the go report's measured values: "
              "python scripts/workstation.py go --hours-per-run H --concurrent K")
    print_commit([PILOT_RESULTS + "/", ANALYSIS_DIR + "/"], GO_MESSAGE)
    print(GO_MEETING)
    return EXIT_OK


# -- the runbook ---------------------------------------------------------------


def _plain(command: str) -> Callable[[argparse.Namespace], str]:
    return lambda args: command


def _allocation_command(args: argparse.Namespace) -> str:
    hours, basis, date = getattr(args, "hours", None), getattr(args, "basis", None), getattr(args, "date", None)
    decided = hours is not None and bool(basis)
    return (f"python scripts/workstation.py allocation --hours {number(hours) if hours is not None else 'H'} "
            f"--basis {shlex.quote(basis) if basis else 'TEXT'} --date {date or 'YYYY-MM-DD'}"
            + ("" if decided else "  (once the group has decided H and its basis)"))


def _pilot_runs_command(args: argparse.Namespace) -> str:
    k = getattr(args, "max_concurrent", None) or suggested_concurrency()
    return (f"python scripts/workstation.py pilot-runs --max-concurrent {k} --data-root "
            f"{shlex.quote(data_root_of(args))}  (K: usually the physical cores minus 1)")


STEPS: tuple[Step, ...] = (
    Step("registration", "1", "create the tag of the registration commit (Table 9.0)",
         registration_verdict, registration_run, _plain("python scripts/workstation.py registration"),
         _plain(f"bash {TAG_SCRIPT}; then git push origin {TAG} (by you)")),
    Step("env", "8", "build the pinned environment and the workstation record (Table 3.1; Appendix A)",
         env_verdict, env_run, _plain("python scripts/workstation.py env"), _plain(f"bash {SETUP_SCRIPT}")),
    Step("allocation", "12", "record the group's machine-hour allocation (Part 6 G2; Table 8.1)",
         allocation_verdict, allocation_run, _allocation_command,
         _plain("python -m pilot allocate --hours H --basis TEXT --by BY --date YYYY-MM-DD [--confirm] "
                "[--smoke-root DIR ...]"), ("allocation",)),
    Step("fixed-batch", "8a", "re-check the fixed batches on the workstation (Table 2.3; Appendix A)",
         fixed_batch_verdict, fixed_batch_run, _plain("python scripts/workstation.py fixed-batch"),
         _plain("python " + " ".join(FIXED_BATCH_COMMAND) + f"; then the runbook writes {FIXED_BATCH_CHECK}")),
    Step("determinism-default", "9", "the determinism check, default form (plug-in ppolag)",
         determinism_default_verdict, determinism_default_run,
         _plain("python scripts/workstation.py determinism-default"), _plain(f"python {DETERMINISM_SCRIPT}")),
    Step("determinism", "12a", "registered-form determinism reports per plug-in (Table 3.1)",
         determinism_verdict, determinism_run, _plain("python scripts/workstation.py determinism"),
         _plain(f"python {DETERMINISM_SCRIPT} --registered-form --plugin P, for P in {', '.join(DETERMINISM_ORDER)}"),
         ("plugins",)),
    Step("search", "16", "the targeted search of Appendix C (Role 5; a people's step)",
         search_verdict, search_run, _plain("python scripts/workstation.py search"),
         _plain("python -m studyb.search check (read-only)")),
    Step("registration-record", "3", "Tables 9.0 and 9.1 filled and committed, the tag pushed (X-registration; a "
         "people's step)", registration_record_verdict, registration_record_run,
         _plain("python scripts/workstation.py registration-record"),
         _plain("nothing: your Word edit of Tables 9.0 and 9.1 (task 3), its commit, git push origin registration")),
    Step("pilot-runs", "20", "queue and run the eight pilot runs (Part 3.6)",
         pilot_runs_verdict, pilot_runs_run, _pilot_runs_command,
         _plain("python -m pilot schedule add --design pilot; python -m pilot schedule run --max-concurrent K; "
                "python -m pilot schedule status"), ("data", "runs")),
    Step("pilot-enrich", "21-23", "select, measure, the battery, and the G4 measurement",
         pilot_enrich_verdict, pilot_enrich_run,
         lambda args: f"python scripts/workstation.py pilot-enrich --data-root {shlex.quote(data_root_of(args))}",
         _plain(f"python -m pilot enrich select|measure|battery --ledger {PILOT_LEDGER}; python -m pilot g4-measure"),
         ("data",)),
    Step("go", "24-26", "the go report, the analysis test run and the surplus-seed plan (Part 6; Part 5.8; Part 5.5)",
         go_verdict, go_run, _plain("python scripts/workstation.py go [--hours-per-run H --concurrent K]"),
         _plain(f"python -m pilot go; python -m analysis --mode pilot --ledger {PILOT_LEDGER}; "
                "python -m pilot surplus --hours-per-run H --concurrent K"), ("go",)),
)
STEP_BY_NAME = {step.name: step for step in STEPS}


def choose(head: Head, args: argparse.Namespace) -> tuple[list[tuple[Step, Verdict]], tuple[Step, Verdict] | None]:
    """Every step with its verdict, and the next one: the first not done and not skippably blocked (None: all done)."""
    judged = [(step, judge(step, head, args)) for step in STEPS]
    for step, verdict in judged:
        if verdict.state == DONE or (verdict.state == BLOCKED and verdict.skippable):
            continue
        return judged, (step, verdict)
    return judged, None


def cmd_status(args: argparse.Namespace) -> int:
    """Print every step's verdict and the next command; run nothing (exit 0)."""
    head = read_head()
    print(f"workstation runbook (HANDOVER.md section 4): HEAD {head.commit or '-'}; data root {data_root_of(args)}; "
          f"interpreter {sys.executable}")
    if head.commit is None:
        print(f"note: {head.problem}; every step reads what is committed at HEAD, so none can be judged here")
    note = interpreter_note()
    if note:
        print(f"note: {note}")
    judged, chosen = choose(head, args)
    for i, (step, verdict) in enumerate(judged, 1):
        print(f"{i:2d}. {step.name:20s} task {step.tasks:6s} {verdict.state:8s} {verdict.why}")
        if verdict.warning:
            print(f"    warning: {verdict.warning}")
    if chosen is None:
        print("next: " + GO_MEETING)
    elif chosen[1].state == BLOCKED:
        print(f"next: {chosen[0].name} (task {chosen[0].tasks}) is BLOCKED: {chosen[1].why}")
    else:
        print(f"next: {chosen[0].command(args)}")
        print(f"      which runs: {chosen[0].runs(args)}")
    return EXIT_OK


def cmd_next(args: argparse.Namespace) -> int:
    """Run the first step that is not done; stop at a people's step (module docstring)."""
    head = read_head()
    if head.commit is None:
        print(f"refused: {head.problem}")
        return EXIT_BLOCKED
    note = interpreter_note()
    if note:
        print(f"note: {note}")
    judged, chosen = choose(head, args)
    for step, verdict in judged:
        if chosen is not None and step is chosen[0]:
            break
        if verdict.state == BLOCKED:  # skippable: the scheduler holds what waits for it
            print(f"{step.name} (task {step.tasks}): BLOCKED, going on (the scheduler holds the runs that wait for "
                  f"it): {verdict.why}")
    if chosen is None:
        print("Every step of the runbook is done. " + GO_MEETING)
        return EXIT_OK
    step, verdict = chosen
    print(f"next: {step.name} (HANDOVER task {step.tasks}, {step.title}): {verdict.state}: {verdict.why}")
    if verdict.state == BLOCKED:
        print("nothing was run: a people's step or a decision holds it")
        return EXIT_BLOCKED
    return step.run(head, args)


def cmd_step(step: Step, args: argparse.Namespace) -> int:
    """Run one step (a step already done only says so)."""
    head = read_head()
    if head.commit is None:
        print(f"refused: {head.problem}")
        return EXIT_BLOCKED
    verdict = judge(step, head, args)
    if verdict.state == DONE:
        print(f"{step.name} (HANDOVER task {step.tasks}): done: {verdict.why}")
        if verdict.warning:
            print(f"warning: {verdict.warning}")
        return EXIT_OK
    note = interpreter_note()
    if note:
        print(f"note: {note}")
    return step.run(head, args)


def build_parser() -> argparse.ArgumentParser:
    """The command line: ``status``, ``next`` (every step option) and one subcommand per step."""
    groups = {name: argparse.ArgumentParser(add_help=False, allow_abbrev=False)
              for name in ("data", "plugins", "allocation", "runs", "go")}
    groups["data"].add_argument("--data-root", default=DEFAULT_DATA_ROOT,
                                help="the scheduler's data root (default: $WSCL_DATA_ROOT or /data, as python -m "
                                     "pilot)")
    groups["plugins"].add_argument("--plugins", nargs="+", choices=DETERMINISM_ORDER, default=None,
                                   help="determinism: only these plug-ins (default: all, in the order "
                                        f"{', '.join(DETERMINISM_ORDER)})")
    groups["allocation"].add_argument("--hours", type=float, default=None,
                                      help="allocation: the machine-hours the group allocated (Part 6 G2; Table 8.1)")
    groups["allocation"].add_argument("--basis", default=None,
                                      help="allocation: how the group derived H, from the calendar and the machine's "
                                           "availability only, never from a timing of a run on the workstation "
                                           "(python -m pilot allocate --basis)")
    groups["allocation"].add_argument("--date", default=None, help="allocation: the date of the decision, YYYY-MM-DD")
    groups["allocation"].add_argument("--by", default=DEFAULT_BY,
                                      help=f"allocation: who decided (default: {DEFAULT_BY})")
    groups["allocation"].add_argument("--confirm", action="store_true",
                                      help=f"allocation: record an H above {budget.ALLOCATION_CONFIRM_ABOVE} hours "
                                           "(python -m pilot allocate --confirm)")
    groups["allocation"].add_argument("--smoke-root", action="append", default=None, metavar="DIR",
                                      help="allocation: a smoke data root already used on the workstation, whose "
                                           "timings the record lists (python -m pilot allocate --smoke-root; "
                                           "repeatable)")
    groups["runs"].add_argument("--max-concurrent", type=int, default=None,
                                help="pilot-runs: runs executed at once (usually the physical cores minus 1)")
    groups["go"].add_argument("--hours-per-run", type=float, default=None,
                              help="go: measured wall-clock hours per 10M-step run, for the surplus plan (Part 5.5)")
    groups["go"].add_argument("--concurrent", type=int, default=None,
                              help="go: runs sustained concurrently, for the surplus plan (Table 8.1)")
    p = argparse.ArgumentParser(prog="python scripts/workstation.py", description=__doc__.split("\n")[0],
                                allow_abbrev=False)
    sub = p.add_subparsers(dest="command", required=True, metavar="{status,next,STEP}")
    sub.add_parser("status", parents=[groups["data"]], allow_abbrev=False,
                   help="read-only: every step DONE, TODO or BLOCKED (why), and the next command")
    sub.add_parser("next", parents=list(groups.values()), allow_abbrev=False,
                   help="run the first step that is not done (stops at a people's step)")
    for step in STEPS:
        sub.add_parser(step.name, parents=[groups[name] for name in step.options], allow_abbrev=False,
                       help=f"task {step.tasks}: {step.title}")
    return p


def main(argv: list[str] | None = None) -> int:
    """Run the subcommand; its exit status (module docstring)."""
    args = build_parser().parse_args(argv)
    try:
        if args.command == "status":
            return cmd_status(args)
        if args.command == "next":
            return cmd_next(args)
        return cmd_step(STEP_BY_NAME[args.command], args)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return EXIT_INTERRUPTED


if __name__ == "__main__":
    sys.exit(main())

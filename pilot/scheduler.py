"""Run scheduler: launch, resume, never duplicate (First Tasks, Role 1 "What comes next").

State lives in one SQLite file outside the repository (``<data_root>/scheduler/state.sqlite``);
run directories live under ``<data_root>/checkpoints/<run_id>`` so that the ledger's relative
checkpoint paths resolve against ``results.ledger_schema.CHECKPOINT_ROOT`` (``/data/checkpoints``
when ``data_root`` is ``/data``).

Life of a run:

  pending --train--> training --> trained --evaluate--> evaluating --> evaluated --ledger--> ledgered
     |                  |             |                      |                        |
     |                  |             |                      +--> eval_failed          +--> ledger_failed
     |                  |             |                      |    --reevaluate--> trained   --retry-ledger--> evaluated
     |                  |             |                      +--> trained (evaluator unavailable, launcher refused,
     |                  |                                         or ended unobserved)
     |                  +--> continued (Study B few-shot continuations: their results are written to the
     |                  |    parent row by Role 5's evaluation, not as a row)
     |                  +--> excluded (crash, incomplete, non-finite loss or multiplier; Part 5.6), recorded
     |                  |    in the ledger, and replaced by the arm's next unused seed
     |                  +--> interrupted (the machine stopped it, or the launcher ended without a final
     |                  |    result; see the interruption policy)
     |                  |    --restart--> pending (same seed, from scratch; under the 'restart' policy
     |                  |                 directly from training)
     |                  |    --set-policy exclude, or a scheduler start under 'exclude'--> excluded
     |                  |                 (cause 'incomplete')
     |                  +--> pending (plug-in unavailable, launcher refused, launcher failed before its
     |                       claim, or never started: nothing ran, so nothing is excluded)
     +--> ledgered or excluded (already in a ledger, e.g. the state was restored from a backup)
     +--> blocked (a dependency was excluded; group decision: `resolve replace` or `resolve unblock`;
     |    a blocked Study B continuation follows its parent: `resolve PARENT replace`)
     |    --unblock--> pending
     |    --replace--> superseded (the arm's next unused seed runs instead)
     +--> superseded (a Study B continuation whose parent was excluded and replaced)

  An excluded run left without a replacement by the circuit breaker waits for the group's
  decision, carried out with `resolve replace`. An excluded Study B continuation cannot take a new
  seed (it follows its parent's), so the scheduler offers no action for it: the group decides by
  an amendment (Table 9.1). ledgered, continued and superseded are terminal.

Guarantees:
  * never duplicate: run_id is the primary key; a run already in a ledger is never launched; the
    state 'training' is committed before the process starts; the launcher claims the run directory
    atomically; one scheduler process per data root (exclusive lock), and operator commands
    serialize with it on SQLite write transactions; live runs are adopted on restart, identified by
    their run directory on their command line (compared as real paths, so another spelling of the
    same data root still finds them); a run whose process is still alive is never restarted.
  * no result-based decisions: the scheduler reads only completion status, step counts and
    finiteness, never costs or returns. The interruption policy is stored once per data root and
    applies to every run alike ('hold' until the group answers Q-interrupted-run; held runs may be
    restarted, which reproduces them bit for bit, but never excluded one by one). Under 'restart' a
    run is restarted at most MAX_RESTARTS times, then held.
  * crash-safe exclusion: the exclusion, its replacement and the dependents' new states are one
    SQLite transaction; the ledger record of an exclusion is written after it and retried at every
    pass until written.
  * reproducible restarts: a restarted run uses the commit its interrupted attempt ran on (from
    that attempt's train_result.json) or waits; a restart cut short after its archive move is
    completed at the next start, never taken for a run that never started. An operator restarts
    only a held run, and the policy only the attempt the scheduler saw end or a run still held
    (the status is checked inside the write transaction); a run another command acted on first is
    left as that command left it, never restarted, held or excluded by the policy.
  * a systematic failure stops: after MAX_SAME_CAUSE_EXCLUSIONS exclusions of one arm with the same
    cause no further replacement is added, and the arm waits for a group decision. Exclusions applied
    by the interruption policy do not count (one outage is not a property of the arm), except that
    after MAX_SAME_CAUSE_EXCLUSIONS successive replacements stopped by the policy (each replacing
    the last: a run stopped the same way again and again) the arm waits for the group as well.
  * smoke and registered runs never mix: the mode (smoke or registered) and the ledger location
    are fixed per data root at its first run or decision (the command line's ``schedule add`` and
    ``add-surplus`` fix the mode, as registered), and smoke ledgers never lie inside the repository.
  * pilot runs start only when HEAD contains the machine-hour allocation (Part 6 G2).

Limits (see HANDOVER.md): one workstation per data root; a run that dies from a signal while no
scheduler is its parent (the scheduler was restarted) leaves no exit status, so it cannot be told
apart from an interruption: it is held, not excluded as a crash.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

from configs import registered as R
from pilot import manifest, provenance
from pilot.launch import (
    CLAIM_FILE,
    EVALUATION_RESULT,
    EXIT_REFUSED,
    EXIT_UNAVAILABLE,
    OMNISAFE_SUBDIR,
    SPEC_FILE,
    TRAIN_RESULT,
    _write_json_atomic,
)
from pilot.ledger_writer import DEFAULT_PATHS, LedgerPaths, recorded_anywhere, smoke_paths, write_run
from pilot.manifest import RunSpec

ACTIVE = ("training", "evaluating")
DONE_TRAINING = ("trained", "evaluating", "evaluated", "ledgered", "continued", "eval_failed", "ledger_failed")
INTERRUPT_POLICIES = ("hold", "restart", "exclude")
MAX_SAME_CAUSE_EXCLUSIONS = 2  # operational safety valve: independent failures do not repeat like this
MAX_RESTARTS = 2  # under the 'restart' policy: a run killed the same way every time is then held
ALLOCATION_PATH = "pilot/allocation.json"
RESOLVE_ACTIONS = ("restart", "reevaluate", "retry-ledger", "unblock", "replace")
# A child killed by one of these signals crashed (Part 5.6 'crash'); other signals mean the machine stopped it.
CRASH_SIGNALS = tuple(
    -int(getattr(signal, name)) for name in ("SIGSEGV", "SIGABRT", "SIGBUS", "SIGFPE", "SIGILL") if hasattr(signal, name)
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    arm_id TEXT NOT NULL,
    seed INTEGER NOT NULL,
    spec TEXT NOT NULL,
    status TEXT NOT NULL,
    attempt INTEGER NOT NULL DEFAULT 1,
    pid INTEGER,
    host TEXT,
    stage_started TEXT,
    launch_commit TEXT,
    failure_cause TEXT,
    replaces TEXT,
    replaced_by TEXT,
    surplus INTEGER NOT NULL DEFAULT 0,
    ledger_written INTEGER NOT NULL DEFAULT 0,
    train_hours REAL,
    eval_hours REAL,
    updated TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    run_id TEXT,
    event TEXT NOT NULL,
    detail TEXT
);
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

CommandBuilder = Callable[[str, Path, Path], Sequence[str]]


def default_command(stage: str, spec_path: Path, run_dir: Path, *, allow_dirty: bool, allow_pending: bool) -> list[str]:
    cmd = [sys.executable, "-m", "pilot.launch", stage, "--spec", str(spec_path), "--run-dir", str(run_dir)]
    if allow_dirty:
        cmd.append("--allow-dirty")
    if stage == "train" and allow_pending:
        cmd.append("--allow-pending")
    return cmd


@dataclass
class SchedulerConfig:
    data_root: Path = field(default_factory=lambda: Path(os.environ.get("WSCL_DATA_ROOT", "/data")))
    max_concurrent: int = 1
    poll_seconds: float = 30.0
    on_interrupt: str | None = None  # None: the policy stored for the data root ('hold' on its first run)
    allow_dirty: bool = False  # smoke tests only
    allow_pending: bool = False  # smoke tests only
    ledger_paths: LedgerPaths | None = None  # None: the repository's ledgers, or smoke ledgers for smoke tests
    command_builder: CommandBuilder | None = None  # tests inject a fake launcher here
    require_allocation_for_pilot: bool = True

    def __post_init__(self) -> None:
        # Absolute (not resolved: /data stays /data for the ledger's portable paths), so that the
        # command lines of the runs name the same directory whatever the caller's working directory.
        self.data_root = Path(os.path.abspath(self.data_root))
        if self.on_interrupt is not None and self.on_interrupt not in INTERRUPT_POLICIES:
            raise ValueError(f"on_interrupt must be one of {INTERRUPT_POLICIES}")
        if self.max_concurrent < 1:
            raise ValueError("max_concurrent must be at least 1")
        if self.poll_seconds < 0:
            raise ValueError("poll_seconds must be non-negative")
        if self.ledger_paths is None:
            self.ledger_paths = smoke_paths(self.data_root) if self.smoke else DEFAULT_PATHS
        if self.smoke:
            repo = provenance.REPO_ROOT.resolve()
            lp = self.ledger_paths
            if any(Path(p).resolve().is_relative_to(repo) for p in (lp.main, lp.pilot, lp.sidecar_dir)):
                raise ValueError("smoke runs (--allow-dirty / --allow-pending) may not write ledgers inside the repository")

    @property
    def smoke(self) -> bool:
        return self.allow_dirty or self.allow_pending

    @property
    def mode(self) -> str:
        return "smoke" if self.smoke else "registered"

    @property
    def runs_root(self) -> Path:
        return self.data_root / "checkpoints"

    @property
    def state_dir(self) -> Path:
        return self.data_root / "scheduler"

    @property
    def state_path(self) -> Path:
        return self.state_dir / "state.sqlite"

    def run_dir(self, run_id: str) -> Path:
        return self.runs_root / run_id


class SchedulerError(RuntimeError):
    pass


class StatusChanged(SchedulerError):
    """The run's status is no longer the one the caller checked: another command acted on it first."""


class Scheduler:
    def __init__(self, config: SchedulerConfig) -> None:
        self.cfg = config
        self.cfg.state_dir.mkdir(parents=True, exist_ok=True)
        (self.cfg.state_dir / "logs").mkdir(exist_ok=True)
        self.cfg.runs_root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.cfg.state_path, timeout=60.0)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self._procs: dict[str, subprocess.Popen] = {}
        self._host = socket.gethostname()
        self._evaluator_unavailable = False
        self._unavailable_plugins: set[str] = set()
        self._refused: set[str] = set()
        self._launch_failed: set[str] = set()
        self._notified: set[tuple[str, str]] = set()
        self._head: str | None = None
        self._clean: bool | None = None

    @property
    def paths(self) -> LedgerPaths:
        assert self.cfg.ledger_paths is not None
        return self.cfg.ledger_paths

    # -- bookkeeping ---------------------------------------------------------

    def _now(self) -> str:
        return provenance.utc_now().isoformat()

    @contextlib.contextmanager
    def _write_txn(self) -> Iterator[None]:
        """One write transaction that holds SQLite's write lock from its first read.

        Reads that decide a write (the next unused seed, the exclusion count) happen inside it, so
        an operator command running beside the scheduler can never pick the same seed.
        """
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self.db.rollback()
            raise
        else:
            self.db.commit()

    def _event_sql(self, run_id: str | None, event: str, detail: str = "") -> None:
        self.db.execute(
            "INSERT INTO events (ts, run_id, event, detail) VALUES (?, ?, ?, ?)", (self._now(), run_id, event, detail)
        )

    def _event(self, run_id: str | None, event: str, detail: str = "") -> None:
        with self.db:
            self._event_sql(run_id, event, detail)

    def _event_once(self, run_id: str, event: str, detail: str = "") -> None:
        """Log a waiting reason once per scheduler process, not on every poll."""
        if (run_id, event) not in self._notified:
            self._notified.add((run_id, event))
            self._event(run_id, event, detail)

    def _set_sql(self, run_id: str, **values: object) -> None:
        values["updated"] = self._now()
        cols = ", ".join(f"{k} = ?" for k in values)
        self.db.execute(f"UPDATE runs SET {cols} WHERE run_id = ?", (*values.values(), run_id))

    def _set(self, run_id: str, **values: object) -> None:
        with self.db:
            self._set_sql(run_id, **values)

    def row(self, run_id: str) -> sqlite3.Row:
        found = self.db.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if found is None:
            raise KeyError(run_id)
        return found

    def exists(self, run_id: str) -> bool:
        return self.db.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,)).fetchone() is not None

    def spec(self, run_id: str) -> RunSpec:
        return RunSpec.from_json(self.row(run_id)["spec"])

    def rows(self, status: str | Iterable[str] | None = None) -> list[sqlite3.Row]:
        if status is None:
            return list(self.db.execute("SELECT * FROM runs ORDER BY rowid"))
        statuses = (status,) if isinstance(status, str) else tuple(status)
        marks = ",".join("?" * len(statuses))
        return list(self.db.execute(f"SELECT * FROM runs WHERE status IN ({marks}) ORDER BY rowid", statuses))

    def setting(self, key: str) -> str | None:
        found = self.db.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return None if found is None else found["value"]

    def _ledger_location(self) -> str:
        """The ledger paths as stored per data root: real paths (any spelling of the data root is the
        same root), relative to the repository when inside it (a moved or fresh clone still writes
        the repository's ledgers)."""
        repo = Path(os.path.realpath(provenance.REPO_ROOT))

        def where(p: Path) -> str:
            real = Path(os.path.realpath(p))
            return "<repo>/" + real.relative_to(repo).as_posix() if real.is_relative_to(repo) else str(real)

        return json.dumps([where(p) for p in (self.paths.main, self.paths.pilot, self.paths.sidecar_dir)])

    def fix_mode(self, *, ledgers: bool = True) -> None:
        """Fix this data root's mode and ledger location now, or refuse (``SchedulerError``) if they differ.

        ``run``, ``resolve`` and ``set-policy`` do this themselves; a caller that queues runs of one
        mode (``scripts/smoke_run.py``) calls it first, so runs of the wrong mode are never queued.
        ``ledgers=False`` fixes and checks the mode only (``schedule add`` takes no ``--ledger-dir``).
        """
        self._fix_mode(ledgers=ledgers)

    def _fix_mode(self, *, ledgers: bool = True) -> None:
        """The mode (smoke or registered) and the ledger location are fixed by the first run or decision;
        the command line's ``schedule add`` and ``add-surplus`` fix the mode only (``ledgers=False``).

        Smoke and registered runs never share a data root, and every command that writes ledger
        records (run, resolve, set-policy) writes them to the same ledgers.
        """
        with self._write_txn():  # read and fixed under one write lock: two first commands never both pass
            stored = self.setting("mode")
            if stored is not None and stored != self.cfg.mode:
                hint = "pass --allow-dirty or --allow-pending" if stored == "smoke" else "drop --allow-dirty and --allow-pending"
                raise SchedulerError(f"{self.cfg.data_root} holds {stored} runs; {hint} (the mode is fixed per data root)")
            stored_ledgers = self.setting("ledgers") if ledgers else None
            if stored_ledgers is not None and stored_ledgers != self._ledger_location():
                main = json.loads(stored_ledgers)[0]
                raise SchedulerError(
                    f"{self.cfg.data_root} writes its ledgers beside {main}; this command would write them beside "
                    f"{json.loads(self._ledger_location())[0]}. Use the same --ledger-dir as its first run (none for the default)"
                )
            if stored is None:
                self.db.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('mode', ?)", (self.cfg.mode,))
                self._event_sql(None, "mode", self.cfg.mode)
            if ledgers and stored_ledgers is None:
                self.db.execute("INSERT OR IGNORE INTO settings (key, value) VALUES ('ledgers', ?)", (self._ledger_location(),))
                self._event_sql(None, "ledgers", json.loads(self._ledger_location())[0])

    # -- adding runs -----------------------------------------------------------

    def _insert_sql(self, spec: RunSpec, *, status: str = "pending", replaces: str | None = None, surplus: bool = False) -> bool:
        existing = self.db.execute("SELECT spec FROM runs WHERE run_id = ?", (spec.run_id,)).fetchone()
        if existing is not None:
            if RunSpec.from_json(existing["spec"]) != spec:
                raise SchedulerError(f"{spec.run_id} is already scheduled with a different spec; a registered run is never changed")
            return False
        self.db.execute(
            "INSERT INTO runs (run_id, arm_id, seed, spec, status, replaces, surplus, updated) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (spec.run_id, spec.arm_id, spec.seed, spec.to_json(), status, replaces, int(surplus), self._now()),
        )
        detail = "; ".join(x for x in (f"replaces={replaces}" if replaces else "", "surplus" if surplus else "") if x)
        self._event_sql(spec.run_id, "added" if status == "pending" else f"added_{status}", detail)
        return True

    def add(self, specs: Iterable[RunSpec]) -> int:
        """Insert runs as pending. A run_id already present must carry an identical spec."""
        added = 0
        with self._write_txn():
            for spec in specs:
                added += int(self._insert_sql(spec))
        return added

    def next_unused_seed(self, arm_id: str, also_used: Iterable[int] = ()) -> int:
        """The smallest seed >= 5 the arm has not used for any purpose (Part 5.6; Q-seed-collision)."""
        used = {r["seed"] for r in self.db.execute("SELECT seed FROM runs WHERE arm_id = ?", (arm_id,))} | set(also_used)
        seed = R.NEXT_UNUSED_SEED_START
        while seed in used:
            seed += 1
        return seed

    @staticmethod
    def _with_seed_rule(spec: RunSpec) -> RunSpec:
        """Replacement and collision seeds follow the proposed answer to Q-seed-collision."""
        if "Q-seed-collision" in spec.pending:
            return spec
        return RunSpec.from_dict({**spec.to_dict(), "pending": [*spec.pending, "Q-seed-collision"]})

    def add_surplus(self, extra: int) -> list[str]:
        """Part 5.5: ``extra`` surplus seeds for every scheduled arm of the primary-comparison set.

        ``extra`` is a per-arm target, fixed per data root ("the same count for every arm in the
        set"): calling again with the same value adds only the seeds an arm still lacks, for example
        for the Study B arms queued after Study A's surplus. The set is the N = 0 and N = 0.50 main
        arms on SafetyPointGoal1-v0 and the seven Study B arms (proposed reading, Q-surplus-arm-set).
        Surplus seed i (0-based) is 5 + i; an arm whose seed differs (because a replacement used
        one) waits on Q-seed-collision.
        """
        if extra < 0 or extra > R.MAX_SEEDS_PER_ARM - len(R.SEEDS):
            raise ValueError(f"extra seeds must be between 0 and {R.MAX_SEEDS_PER_ARM - len(R.SEEDS)}")
        added: list[str] = []
        with self._write_txn():
            stored = self.setting("surplus_extra")
            if stored is not None and int(stored) != extra:
                raise SchedulerError(
                    f"the surplus is fixed at {stored} seeds per arm on this data root "
                    "(Part 5.5: the same count for every arm in the set)"
                )
            bases: dict[str, RunSpec] = {}
            for row in self.rows():
                spec = RunSpec.from_json(row["spec"])
                if manifest.is_primary_comparison(spec) and not row["surplus"] and spec.arm_id not in bases:
                    bases[spec.arm_id] = spec
            for arm_id, base in bases.items():
                have = self.db.execute(
                    "SELECT COUNT(*) FROM runs WHERE arm_id = ? AND surplus = 1 AND replaces IS NULL", (arm_id,)
                ).fetchone()[0]
                taken: list[int] = []
                for i in range(have, extra):
                    seed = self.next_unused_seed(arm_id, taken)
                    taken.append(seed)
                    spec = manifest.surplus_spec(base, seed)  # the ramp arms wait on Q-surplus-arm-set
                    if seed != R.SURPLUS_SEED_START + i:
                        spec = self._with_seed_rule(spec)
                    if self._insert_sql(spec, surplus=True):
                        added.append(spec.run_id)
                    if spec.group == "study_b":
                        for cont in manifest.study_b_fewshot([spec]):
                            self._insert_sql(cont, surplus=True)
            if stored is None:
                self.db.execute("INSERT INTO settings (key, value) VALUES ('surplus_extra', ?)", (str(extra),))
                self._event_sql(None, "surplus", f"{extra} seeds per primary-comparison arm (Part 5.5)")
        return added

    # -- launching -------------------------------------------------------------

    def _command(self, stage: str, spec_path: Path, run_dir: Path) -> list[str]:
        if self.cfg.command_builder is not None:
            return list(self.cfg.command_builder(stage, spec_path, run_dir))
        return default_command(stage, spec_path, run_dir, allow_dirty=self.cfg.allow_dirty, allow_pending=self.cfg.allow_pending)

    def _spawn(self, run_id: str, stage: str, attempt: int) -> subprocess.Popen:
        run_dir = self.cfg.run_dir(run_id)
        log = self.cfg.state_dir / "logs" / f"{run_id}.{stage}.attempt{attempt}.log"
        env = dict(os.environ)
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[var] = "1"  # Table 3.1: one thread per run
        env["PYTHONPATH"] = str(provenance.REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        with open(log, "ab") as fh:
            proc = subprocess.Popen(
                self._command(stage, run_dir / SPEC_FILE, run_dir),
                cwd=provenance.REPO_ROOT,
                env=env,
                stdout=fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,  # a scheduler restart does not kill running runs
            )
        self._procs[run_id] = proc
        return proc

    def _dependencies_state(self, spec: RunSpec) -> str:
        """'ready', 'waiting', 'blocked' (a dependency was excluded, blocked or superseded) or 'missing' (not scheduled yet)."""
        state = "ready"
        for dep in spec.depends_on:
            try:
                status = self.row(dep)["status"]
            except KeyError:
                state = "missing"
                continue
            if status in ("excluded", "blocked", "superseded"):
                return "blocked"
            if status not in DONE_TRAINING and state == "ready":
                state = "waiting"
        return state

    def _refresh_git(self) -> None:
        """HEAD and cleanliness, read once per scheduling pass."""
        self._head = provenance.commit_hash()
        self._clean = not provenance.dirty_paths()

    def _allocation_at_head(self) -> bool:
        return self._head is not None and provenance.file_committed_at(self._head, ALLOCATION_PATH) is not None and bool(self._clean)

    def _launch_blocker(self, row: sqlite3.Row) -> str | None:
        """Why a pending run cannot start now (None: it can). Never depends on any result."""
        spec = RunSpec.from_json(row["spec"])
        if spec.plugin in self._unavailable_plugins:
            return "plugin_unavailable"
        if row["run_id"] in self._refused:
            return "launch_refused"
        if row["run_id"] in self._launch_failed:
            return "launch_failed"
        if spec.open_questions and not self.cfg.allow_pending:
            return "open_questions"
        deps = self._dependencies_state(spec)
        if deps != "ready":
            return f"dependency_{deps}"
        if not self.cfg.allow_dirty and not self._clean:
            return "dirty_worktree"
        if spec.pilot and self.cfg.require_allocation_for_pilot and not self.cfg.smoke and not self._allocation_at_head():
            return "allocation_missing"
        if int(row["attempt"]) > 1 and row["launch_commit"] and row["launch_commit"] != self._head and not self.cfg.allow_dirty:
            return "restart_commit_mismatch"
        return None

    def _ledger_record(self, spec: RunSpec) -> dict | None:
        """The run's recorded ledger row (or sidecar record) as a dict, or None."""
        sidecar = self.paths.sidecar_dir / f"{spec.run_id}.json"
        if sidecar.exists():
            return self._read_json(sidecar)
        from results.ledger_schema import load_ledger_as_rows

        ledger = self.paths.for_spec(spec)
        for recorded in load_ledger_as_rows(ledger) if ledger.exists() else []:
            if recorded.run_id == spec.run_id:
                return {"completed": recorded.completed, "failure_cause": recorded.failure_cause}
        return None

    def _settle_recorded(self, run_id: str, spec: RunSpec) -> None:
        """A pending run already in a ledger (e.g. the state was restored from an older backup)."""
        record = self._ledger_record(spec) or {}
        if record.get("completed") is False:
            with self.db:
                self._set_sql(run_id, status="excluded", pid=None, failure_cause=record.get("failure_cause"), ledger_written=1)
                self._event_sql(run_id, "exclusion_in_ledger", f"cause={record.get('failure_cause')}; never launched twice")
                self._event_sql(run_id, "needs_decision",
                                f"check whether its replacement ran; if not, `schedule resolve {run_id} replace`")
            return
        self._set(run_id, status="ledgered")
        self._event(run_id, "already_in_ledger", "never launched twice")

    def _start_training(self, row: sqlite3.Row) -> bool:
        run_id = row["run_id"]
        spec = RunSpec.from_json(row["spec"])
        if recorded_anywhere(spec, self.paths):
            self._settle_recorded(run_id, spec)
            return False
        blocker = self._launch_blocker(row)
        if blocker == "dependency_blocked":
            self._set(run_id, status="blocked")
            self._event(run_id, "blocked", f"{blocker} for {spec.depends_on}; needs a group decision")
            return False
        if blocker is not None:
            details = {
                "allocation_missing": f"commit {ALLOCATION_PATH} before any pilot run (Part 6 G2)",
                "restart_commit_mismatch": f"the interrupted attempt ran on {row['launch_commit']}; HEAD is {self._head}",
                "dirty_worktree": "commit the changes to tracked code before launching",
                "open_questions": ", ".join(spec.open_questions),
                "dependency_missing": f"waits for {[d for d in spec.depends_on if not self.exists(d)]}, not scheduled yet",
            }
            if blocker in details:
                self._event_once(run_id, blocker, details[blocker])
            return False
        run_dir = self.cfg.run_dir(run_id)
        if run_dir.exists() and any(p.name != SPEC_FILE for p in run_dir.iterdir()):
            raise SchedulerError(f"{run_dir} already holds output; refusing to launch {run_id} again")
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / SPEC_FILE).write_text(spec.to_json(), encoding="utf-8")
        attempt = int(row["attempt"])
        with self.db:  # committed before the process exists: a crash here can never launch it twice
            # Attempt 1 records the commit it starts on now (an earlier start that never trained, e.g. a
            # plug-in not yet committed, does not count); a restart keeps the commit restart() recorded.
            self._set_sql(run_id, status="training", pid=None, host=self._host, stage_started=self._now(),
                          launch_commit=(row["launch_commit"] or self._head) if attempt > 1 else self._head)
            self._event_sql(run_id, "train_starting", f"attempt={attempt} commit={self._head}")
        proc = self._spawn(run_id, "train", attempt)
        self._set(run_id, pid=proc.pid)
        self._event(run_id, "train_started", f"pid={proc.pid}")
        return True

    def _evaluation_blocker(self, run_id: str) -> str | None:
        """Why a trained run cannot be evaluated now (None: it can); the launcher would refuse it."""
        if self._evaluator_unavailable:
            return "evaluator_unavailable"
        if run_id in self._refused:
            return "launch_refused"
        if not self.cfg.allow_dirty and not self._clean:
            return "dirty_worktree"
        return None

    def _start_evaluation(self, run_id: str) -> bool:
        blocker = self._evaluation_blocker(run_id)
        if blocker is not None:
            if blocker == "dirty_worktree":
                self._event_once(run_id, blocker, "commit the changes to tracked code before evaluating")
            return False
        with self.db:
            self._set_sql(run_id, status="evaluating", pid=None, host=self._host, stage_started=self._now())
        proc = self._spawn(run_id, "evaluate", int(self.row(run_id)["attempt"]))
        self._set(run_id, pid=proc.pid)
        self._event(run_id, "eval_started", f"pid={proc.pid}")
        return True

    # -- process state ---------------------------------------------------------

    def _names_run(self, argv: list[bytes], run_id: str, cwd: str | None) -> bool:
        """True if a command line carries this run's directory and spec file as whole arguments.

        Arguments are compared as real paths (a relative one against the process's working
        directory ``cwd``; skipped when that is unknown), so a scheduler started with another
        spelling of the same data root (relative, absolute, through a symlink) still finds the run.
        """
        needle = run_id.encode()
        candidates = [a for a in argv if needle in a]  # cheap filter before any filesystem call
        if not candidates:
            return False
        run_dir = self.cfg.run_dir(run_id)
        wanted = {os.path.realpath(run_dir), os.path.realpath(run_dir / SPEC_FILE)}
        found = set()
        for arg in candidates:
            path = os.fsdecode(arg)
            if not os.path.isabs(path):
                if cwd is None:
                    continue
                path = os.path.join(cwd, path)
            found.add(os.path.realpath(path))
        return wanted <= found

    @staticmethod
    def _proc_argv(pid: int | str, run_id: str) -> tuple[list[bytes], str | None]:
        """Command line and working directory of a process (Linux /proc); OSError if unreadable.

        The working directory is read only when the command line mentions ``run_id`` at all.
        """
        argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        if not any(run_id.encode() in a for a in argv):
            return argv, None
        try:
            cwd: str | None = os.readlink(f"/proc/{pid}/cwd")
        except OSError:  # another user's process: only absolute arguments can be compared
            cwd = None
        return argv, cwd

    def _process_runs(self, pid: int, run_id: str) -> bool:
        """True if process ``pid`` exists and is this run's launcher (guards against PID reuse)."""
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            pass  # it exists but belongs to another user: its command line decides
        if not Path("/proc").is_dir():  # pragma: no cover - no /proc: trust the PID (see HANDOVER.md)
            return True
        try:
            argv, cwd = self._proc_argv(pid, run_id)
        except OSError:
            return False
        return self._names_run(argv, run_id, cwd)

    def _find_process(self, run_id: str) -> int | None:
        """PID of a live process whose command line names this run's directory (Linux /proc)."""
        proc_root = Path("/proc")
        if not proc_root.is_dir():
            return None
        for entry in proc_root.iterdir():
            if not entry.name.isdigit() or int(entry.name) == os.getpid():
                continue
            try:
                argv, cwd = self._proc_argv(entry.name, run_id)
            except OSError:
                continue
            if self._names_run(argv, run_id, cwd):
                return int(entry.name)
        return None

    def _running(self, row: sqlite3.Row, *, announce: bool = False) -> bool:
        """Is the run's process alive? Adopts a live process found by its run directory.

        The recorded PID is trusted only on the recording host and only if its command line still
        names the run; otherwise (PID not yet recorded, another host name, PID reused) the process
        is looked up by its run directory, so a live run is never taken for dead and restarted.
        """
        run_id = row["run_id"]
        proc = self._procs.get(run_id)
        if proc is not None:
            return proc.poll() is None
        pid = row["pid"]
        if pid is not None and row["host"] == self._host and self._process_runs(int(pid), run_id):
            if announce:
                self._event(run_id, "adopted", f"pid={pid}")
            return True
        found = self._find_process(run_id)
        if found is None:
            return False
        self._set(run_id, pid=found, host=self._host)
        note = "" if row["host"] in (None, self._host) else f"; recorded on host {row['host']}"
        self._event(run_id, "adopted", f"pid={found} (found by its run directory{note})")
        return True

    def _settle(self, row: sqlite3.Row) -> None:
        if row["status"] == "training":
            self._finish_training(row["run_id"])
        else:
            self._finish_evaluation(row["run_id"])

    def _exit_code(self, run_id: str) -> int | None:
        proc = self._procs.pop(run_id, None)
        return None if proc is None else proc.returncode

    def _read_json(self, path: Path) -> dict | None:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _finish_training(self, run_id: str) -> None:
        code = self._exit_code(run_id)
        spec = self.spec(run_id)
        run_dir = self.cfg.run_dir(run_id)
        if code in (EXIT_UNAVAILABLE, EXIT_REFUSED):
            # The launcher refuses before it claims the directory, or removes its own claim and output:
            # nothing is cleaned up here (a refusal of a directory claimed by a live run must not touch it).
            if code == EXIT_UNAVAILABLE:
                self._unavailable_plugins.add(spec.plugin)
            else:
                self._refused.add(run_id)
            self._set(run_id, status="pending", pid=None)
            if code == EXIT_UNAVAILABLE:
                self._event(run_id, "plugin_unavailable", f"plug-in {spec.plugin!r} is not in the repository yet; the run stays pending")
            else:
                self._event(run_id, "launch_refused", "the launcher refused to start the run; see the train log")
            return
        if not run_dir.exists() and self._archive(run_id, int(self.row(run_id)["attempt"])).exists():
            # A restart under the policy was cut short after its archive move: finish that restart
            # (not 'never started', which would relaunch the same attempt on HEAD).
            self._event(run_id, "restart_resumed", "the attempt was archived before the scheduler stopped")
            self._restart_or_hold(run_id, "training")
            return
        if not (run_dir / CLAIM_FILE).exists() and not (run_dir / TRAIN_RESULT).exists():
            # The launcher never claimed the directory: nothing trained, so nothing is excluded or held.
            self._set(run_id, status="pending", pid=None)
            if code is None:
                self._event(run_id, "never_started", "no claim in the run directory; back to pending")
            else:
                self._launch_failed.add(run_id)
                self._event(run_id, "launch_failed",
                            f"exit code {code} before the claim; see the train log. Retried at the next scheduler start")
            return
        result = self._read_json(run_dir / TRAIN_RESULT)
        status = result.get("status") if result else None
        if status == "completed":
            if spec.group == "study_b_fewshot":
                self._set(run_id, status="continued", pid=None, train_hours=result.get("wall_clock_hours"))
                self._event(run_id, "continued", "few-shot results are written to the parent row by Role 5's evaluation")
            else:
                self._set(run_id, status="trained", pid=None, train_hours=result.get("wall_clock_hours"))
                self._event(run_id, "trained", f"{result.get('wall_clock_hours') or 0.0:.3f} h")
        elif status == "failed":
            self._exclude(run_id, result["failure_cause"], result.get("detail", ""),
                          by_policy=bool(result.get("excluded_by_policy")))
        elif status == "interrupted":
            self._interrupted(run_id, f"the launcher recorded an interruption: {result.get('detail', '')}")
        elif code in CRASH_SIGNALS:
            self._exclude_without_result(run_id, "crash", f"process died with signal {-code}")
        elif code is None:
            self._interrupted(run_id, "ended while no scheduler was its parent; exit status unknown "
                                      "(a death by signal cannot be told apart from an interruption)")
        else:
            self._interrupted(run_id, f"no final training result (exit code {code})")

    def _interrupted(self, run_id: str, detail: str) -> None:
        policy = self.setting("on_interrupt") or self.cfg.on_interrupt or "hold"
        restarts = int(self.row(run_id)["attempt"]) - 1
        if policy == "restart" and restarts >= MAX_RESTARTS:
            self._event(run_id, "restart_limit", f"restarted {restarts} times (MAX_RESTARTS); held for the group")
            policy = "hold"
        self._event(run_id, "interrupted", f"{detail}; policy={policy}")
        if policy == "hold":
            self._set(run_id, status="interrupted", pid=None)
        elif policy == "restart":
            self._restart_or_hold(run_id, "training")
        else:
            self._exclude_without_result(run_id, "incomplete", detail, by_policy=True)

    def _restart_or_hold(self, run_id: str, expected: str) -> None:
        """Restart under the policy; a restart that cannot happen now holds the run (never stops the scheduler).

        A run another command acted on first (its status is no longer ``expected``) is left as that
        command left it: holding it could mark a live attempt as held.
        """
        try:
            self.restart(run_id, expected=(expected,))
        except StatusChanged as exc:
            self._event(run_id, "restart_skipped", str(exc))
        except (SchedulerError, OSError) as exc:
            self._set(run_id, status="interrupted", pid=None)
            self._event(run_id, "restart_failed", f"{exc}; held (`schedule resolve {run_id} restart` once fixed)")

    def _archive(self, run_id: str, attempt: int) -> Path:
        return self.cfg.state_dir / "interrupted" / f"{run_id}-attempt{attempt}"

    def restart(self, run_id: str, *, expected: tuple[str, ...] = ("interrupted",)) -> None:
        """Archive the interrupted run's partial output and train it again from scratch, same seed.

        ``expected``: the statuses the caller checked ('training' only for the scheduler settling
        the attempt it watched end), re-checked inside the write transaction.
        """
        run_dir = self.cfg.run_dir(run_id)
        # The status is checked and changed inside one write transaction, so two operator commands
        # can never restart the same attempt twice; the archive move happens while it is held.
        with self._write_txn():
            row = self.row(run_id)
            if row["status"] not in expected:
                raise StatusChanged(f"{run_id} is {row['status']}; only an interrupted run can be restarted")
            live = self._find_process(run_id)
            if live is not None:
                raise SchedulerError(f"{run_id}: process {live} still uses {run_dir}; a live run is never restarted")
            attempt = int(row["attempt"])
            archive = self._archive(run_id, attempt)
            # The commit the interrupted attempt actually ran on, as its launcher recorded it; without a
            # result (a hard kill), the commit the scheduler recorded when it started that attempt.
            previous = self._read_json(run_dir / TRAIN_RESULT) or self._read_json(archive / TRAIN_RESULT) or {}
            commit = previous.get("commit_hash") or row["launch_commit"]
            if run_dir.exists():
                if archive.exists():
                    raise SchedulerError(f"{archive} already exists; move it aside before restarting {run_id}")
                archive.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(run_dir), str(archive))
            # else: a restart cut short after its move (before this commit) already archived it
            self._set_sql(run_id, status="pending", pid=None, attempt=attempt + 1, launch_commit=commit)
            self._event_sql(run_id, "restart", f"attempt {attempt + 1}, same seed and commit ({commit}), from scratch")

    def _exclude_without_result(self, run_id: str, cause: str, detail: str, *, by_policy: bool = False,
                                expected: str = "training") -> None:
        """Exclude a run whose train_result.json is not a final one: record what is known.

        The launcher's own record (for example of an interruption) is kept; the scheduler adds the
        exclusion's cause. ``expected`` is the status the caller saw ('training' for the attempt the
        scheduler watched end, 'interrupted' for a held run under the 'exclude' policy); it is
        re-checked inside the write transaction, and a run another command acted on first (for
        example restarted and launched again) is left alone: its directory is never written.
        """
        with self._write_txn():
            row = self.row(run_id)
            if row["status"] != expected:
                self._event_sql(run_id, "exclusion_skipped",
                                f"{run_id} is {row['status']}, not {expected}; another command acted on it first")
                return
            self._write_exclusion_record(run_id, row, cause, detail, by_policy)
            self._exclude_sql(run_id, cause, detail, by_policy=by_policy)
        self._record_exclusion(run_id)

    def _write_exclusion_record(self, run_id: str, row: sqlite3.Row, cause: str, detail: str, by_policy: bool) -> None:
        """Rewrite train_result.json as the exclusion's final record (inside the exclusion's transaction)."""
        run_dir = self.cfg.run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        previous = self._read_json(run_dir / TRAIN_RESULT) or {}
        omnisafe_dir = previous.get("omnisafe_dir")
        if omnisafe_dir is None and (run_dir / OMNISAFE_SUBDIR).exists():
            found = sorted(p for p in (run_dir / OMNISAFE_SUBDIR).glob("*/seed-*") if p.is_dir())
            omnisafe_dir = str(found[0].relative_to(run_dir)) if len(found) == 1 else None
        record = {
            **previous,
            "run_id": run_id,
            "status": "failed",
            "failure_cause": cause,
            "detail": detail,
            "commit_hash": previous.get("commit_hash") or row["launch_commit"] or provenance.commit_hash(),
            "machine": previous.get("machine") or provenance.machine_description(),
            "started": previous.get("started") or row["stage_started"] or self._now(),
            "finished": previous.get("finished") or self._now(),
            "wall_clock_hours": float(previous.get("wall_clock_hours") or 0.0),
            "omnisafe_dir": omnisafe_dir,
            "written_by": (previous["written_by"] if "excluded_by_policy" in previous  # its own record, re-applied
                           else f"scheduler, from the launcher's result with status {previous.get('status')!r}" if previous
                           else "scheduler (the process left no final result)"),
            # Kept in the record, so an exclusion re-read after a crash keeps its origin (circuit breaker).
            "excluded_by_policy": by_policy,
        }
        if self.cfg.smoke:
            record["allow_dirty"] = record.get("allow_dirty") or self.cfg.allow_dirty
            record["allow_pending"] = record.get("allow_pending") or self.cfg.allow_pending
        _write_json_atomic(run_dir / TRAIN_RESULT, record)

    def _same_cause_exclusions(self, arm_id: str, cause: str) -> int:
        """Exclusions of the arm with this cause, not counting those applied by the interruption policy."""
        return self.db.execute(
            "SELECT COUNT(*) FROM runs WHERE arm_id = ? AND status = 'excluded' AND failure_cause = ? AND run_id NOT IN "
            "(SELECT run_id FROM events WHERE event = 'excluded_by_policy' AND run_id IS NOT NULL)",
            (arm_id, cause),
        ).fetchone()[0]

    def _policy_replacement_chain(self, run_id: str) -> int:
        """Replacements stopped by the interruption policy in a row, ending at ``run_id`` (counted as one).

        Follows ``replaces`` back while the replaced run is itself a replacement excluded by the
        policy, so one outage that stops unrelated replacements of an arm counts once for each.
        """
        count, previous = 1, self.row(run_id)["replaces"]
        while previous is not None:
            row = self.row(previous)
            by_policy = self.db.execute(
                "SELECT 1 FROM events WHERE run_id = ? AND event = 'excluded_by_policy'", (previous,)
            ).fetchone()
            if row["replaces"] is None or by_policy is None:
                break
            count, previous = count + 1, row["replaces"]
        return count

    def _replacement_sql(self, run_id: str, spec: RunSpec, *, surplus: bool) -> RunSpec:
        """Queue the arm's next unused seed in place of ``run_id`` (inside a write transaction)."""
        replacement = self._with_seed_rule(spec.with_seed(self.next_unused_seed(spec.arm_id)))
        self._insert_sql(replacement, replaces=run_id, surplus=surplus)
        self._set_sql(run_id, replaced_by=replacement.run_id)
        self._event_sql(run_id, "replaced", replacement.run_id)
        missing = [d for d in replacement.depends_on if not self.exists(d)]
        if missing:
            self._event_sql(replacement.run_id, "dependency_missing", f"waits for {missing}, not scheduled yet")
        if spec.group == "study_b":
            for cont in manifest.study_b_fewshot([replacement]):  # Part 3.3: each run is continued
                self._insert_sql(cont, surplus=surplus)
            # The replaced parent's own continuations are never run: its replacement's are.
            for dependent in self.rows(("pending", "blocked")):
                dspec = RunSpec.from_json(dependent["spec"])
                if dspec.group == "study_b_fewshot" and run_id in dspec.depends_on:
                    self._set_sql(dependent["run_id"], status="superseded")
                    self._event_sql(dependent["run_id"], "superseded",
                                    f"parent {run_id} excluded and replaced by {replacement.run_id}; its continuations run instead")
        return replacement

    def _exclude(self, run_id: str, cause: str, detail: str, *, by_policy: bool = False) -> None:
        """Part 5.6: exclusion with its cause, recorded in the ledger, and repeated with the next unused seed.

        The state change, the replacement and the dependents' states are one transaction.
        ``by_policy`` marks an exclusion applied by the interruption policy: it does not count
        towards the same-cause circuit breaker, but a replacement stopped by the policy again and
        again does (MAX_SAME_CAUSE_EXCLUSIONS successive replacements, each replacing the last,
        stopped by the policy).
        """
        with self._write_txn():
            self._exclude_sql(run_id, cause, detail, by_policy=by_policy)
        self._record_exclusion(run_id)

    def _exclude_sql(self, run_id: str, cause: str, detail: str, *, by_policy: bool = False) -> None:
        """The exclusion, its replacement and the dependents' new states (inside the caller's write transaction)."""
        spec = self.spec(run_id)
        row = self.row(run_id)
        reason = ""
        if spec.group == "study_b_fewshot":
            reason = ("a continuation follows its parent's seed and cannot take a new one; the group decides "
                      "by an amendment (Table 9.1); the scheduler has no action for it")
        elif not by_policy and self._same_cause_exclusions(spec.arm_id, cause) + 1 >= MAX_SAME_CAUSE_EXCLUSIONS:
            reason = (f"{MAX_SAME_CAUSE_EXCLUSIONS} exclusions of this arm with cause '{cause}': likely systematic; "
                      f"no further replacement until the group decides (`schedule resolve {run_id} replace`)")
        elif (by_policy and row["replaces"]
              and self._policy_replacement_chain(run_id) >= MAX_SAME_CAUSE_EXCLUSIONS):
            reason = (f"{MAX_SAME_CAUSE_EXCLUSIONS} successive replacements stopped by the interruption policy: "
                      f"likely systematic; no further replacement until the group decides "
                      f"(`schedule resolve {run_id} replace`)")
        self._set_sql(run_id, status="excluded", pid=None, failure_cause=cause, replaced_by=None)
        self._event_sql(run_id, "excluded", f"{cause}: {detail[-300:]}")
        if by_policy:
            self._event_sql(run_id, "excluded_by_policy", "the interruption policy, applied to every run alike")
        # A replaced Study B parent's continuations are superseded inside _replacement_sql.
        replacement = None if reason else self._replacement_sql(run_id, spec, surplus=bool(row["surplus"]))
        for dependent in self.rows(("pending",)):
            if run_id in RunSpec.from_json(dependent["spec"]).depends_on:
                self._set_sql(dependent["run_id"], status="blocked")
                self._event_sql(dependent["run_id"], "blocked", f"dependency {run_id} excluded; needs a group decision")
        if replacement is None:
            self._event_sql(run_id, "needs_decision", reason)

    def _record_exclusion(self, run_id: str, quiet: bool = False) -> None:
        spec = self.spec(run_id)
        if spec.group == "study_b_fewshot":
            self._set(run_id, ledger_written=1)  # continuations are not ledger rows; the event log keeps the record
            return
        try:
            if recorded_anywhere(spec, self.paths):
                event = "ledger_exclusion_present"
            else:
                write_run(spec, self.cfg.run_dir(run_id), self.paths, attempt=int(self.row(run_id)["attempt"]))
                event = "ledger_exclusion_written"
            self._set(run_id, ledger_written=1)
            self._event(run_id, event, "")
        except Exception as exc:  # noqa: BLE001 - retried at every pass; logged once per process when quiet
            if quiet:
                self._event_once(run_id, "ledger_write_failed", repr(exc))
            else:
                self._event(run_id, "ledger_write_failed", repr(exc))

    def _finish_evaluation(self, run_id: str) -> None:
        code = self._exit_code(run_id)
        evaluation = self._read_json(self.cfg.run_dir(run_id) / EVALUATION_RESULT)
        if code == EXIT_UNAVAILABLE:
            self._evaluator_unavailable = True
            self._set(run_id, status="trained", pid=None)
            self._event(run_id, "evaluator_unavailable", "the evaluation harness (Role 2) is not in the repository yet")
        elif code == EXIT_REFUSED:
            self._set(run_id, status="trained", pid=None)
            self._refused.add(run_id)
            self._event(run_id, "launch_refused", "the launcher refused to evaluate the run; see the evaluate log")
        elif evaluation is not None:
            self._set(run_id, status="evaluated", pid=None, eval_hours=evaluation.get("eval_wall_clock_hours"))
            self._event(run_id, "evaluated", "")
        elif code is None:
            # The evaluation ended while no scheduler watched it (for example a reboot). Evaluation
            # is deterministic and writes nothing until it succeeds, so it is simply run again.
            self._set(run_id, status="trained", pid=None)
            self._event(run_id, "eval_rerun", "evaluation ended unobserved; it will run again")
        else:
            self._set(run_id, status="eval_failed", pid=None)
            self._event(run_id, "eval_failed", f"exit code {code}; see the evaluate log, then `schedule resolve {run_id} reevaluate`")

    def _write_ledger(self, run_id: str) -> None:
        spec = self.spec(run_id)
        try:
            if recorded_anywhere(spec, self.paths):
                self._event(run_id, "ledgered", "row already present")
            else:
                written = write_run(spec, self.cfg.run_dir(run_id), self.paths, attempt=int(self.row(run_id)["attempt"]))
                self._event(run_id, "ledgered", str(written))
        except Exception as exc:  # noqa: BLE001 - terminal until the operator retries
            self._set(run_id, status="ledger_failed")
            self._event(run_id, "ledger_failed", f"{exc!r}; fix the cause, then `schedule resolve {run_id} retry-ledger`")
            return
        self._set(run_id, status="ledgered", ledger_written=1)

    # -- main loop ---------------------------------------------------------------

    def recover(self) -> None:
        """On start: adopt live children, settle runs that ended unobserved, retry unrecorded exclusions.

        Under the 'exclude' policy a run still held (a `set-policy exclude` cut short) is excluded
        now, as the policy applies to every run alike.
        """
        for row in self.rows(ACTIVE):
            if not self._running(row, announce=True):
                self._settle(row)
        if self.setting("on_interrupt") == "exclude":
            for row in self.rows("interrupted"):
                self._exclude_without_result(row["run_id"], "incomplete", "interruption; policy 'exclude' applied to all held runs",
                                             by_policy=True, expected="interrupted")
        for row in self.rows("excluded"):
            if not row["ledger_written"]:
                self._record_exclusion(row["run_id"])

    def step(self) -> None:
        """One pass: settle finished processes, write ledger rows, fill free slots."""
        for row in self.rows(ACTIVE):
            if not self._running(row):
                self._settle(row)
        for row in self.rows("evaluated"):
            self._write_ledger(row["run_id"])
        for row in self.rows("excluded"):  # an exclusion whose ledger record failed earlier
            if not row["ledger_written"]:
                self._record_exclusion(row["run_id"], quiet=True)
        self._refresh_git()
        free = self.cfg.max_concurrent - len(self.rows(ACTIVE))
        for row in self.rows("trained"):  # evaluation first: finish runs before starting new ones
            if free <= 0:
                break
            if self._start_evaluation(row["run_id"]):
                free -= 1
        for row in self.rows("pending"):
            if free <= 0:
                break
            if self._start_training(row):
                free -= 1

    def idle(self) -> bool:
        return not self.rows(ACTIVE)

    def _fix_policy(self) -> None:
        stored = self.setting("on_interrupt")
        if stored is None:
            policy = self.cfg.on_interrupt or "hold"
            with self.db:
                self.db.execute("INSERT INTO settings (key, value) VALUES ('on_interrupt', ?)", (policy,))
                self._event_sql(None, "policy", f"on_interrupt={policy}")
        elif self.cfg.on_interrupt is not None and stored != self.cfg.on_interrupt:
            raise SchedulerError(
                f"this data root uses on_interrupt={stored!r}; it applies to every run alike. "
                "Change it only with `schedule set-policy` after the group's decision."
            )

    def set_policy(self, policy: str) -> None:
        """Change the interruption policy for every run (a recorded group decision) and apply it to held runs."""
        if policy not in INTERRUPT_POLICIES:
            raise ValueError(f"policy must be one of {INTERRUPT_POLICIES}")
        self._fix_mode()
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO settings (key, value) VALUES ('on_interrupt', ?)", (policy,))
            self._event_sql(None, "policy", f"on_interrupt={policy} (changed)")
        self.cfg.on_interrupt = policy
        for row in self.rows("interrupted"):
            if policy == "restart":
                self._restart_or_hold(row["run_id"], "interrupted")
            elif policy == "exclude":
                self._exclude_without_result(row["run_id"], "incomplete", "interruption; policy 'exclude' applied to all held runs",
                                             by_policy=True, expected="interrupted")

    def run(self, *, once: bool = False, max_passes: int | None = None) -> None:
        with self._exclusive():
            self._fix_mode()
            self._fix_policy()
            self._event(None, "scheduler_start", f"max_concurrent={self.cfg.max_concurrent} smoke={self.cfg.smoke}")
            self.recover()
            passes = 0
            while True:
                self.step()
                passes += 1
                if once or (max_passes is not None and passes >= max_passes):
                    return
                if self.idle() and not self._launchable_left():
                    self._event(None, "idle", "nothing left to launch")
                    return
                time.sleep(self.cfg.poll_seconds)

    def _launchable_left(self) -> bool:
        if self.rows("evaluated"):
            return True
        self._refresh_git()
        if any(self._evaluation_blocker(r["run_id"]) is None for r in self.rows("trained")):
            return True
        return any(self._launch_blocker(row) is None for row in self.rows("pending"))

    @contextlib.contextmanager
    def _exclusive(self) -> Iterator[None]:
        lock_path = self.cfg.state_dir / "scheduler.lock"
        with open(lock_path, "w", encoding="utf-8") as fh:
            try:
                import fcntl
            except ImportError:  # pragma: no cover - POSIX only; see HANDOVER.md
                yield
                return
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise SchedulerError(f"another scheduler is running on {self.cfg.data_root}") from exc
            try:
                yield
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

    # -- reporting and operator actions --------------------------------------------

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self.rows():
            counts[row["status"]] = counts.get(row["status"], 0) + 1
        unwritten = sum(1 for r in self.rows("excluded") if not r["ledger_written"])
        if unwritten:
            counts["excluded_not_in_ledger"] = unwritten
        return counts

    def events(self, run_id: str | None = None, limit: int = 50) -> list[sqlite3.Row]:
        if run_id is None:
            return list(self.db.execute("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)))
        return list(self.db.execute("SELECT * FROM events WHERE run_id = ? ORDER BY id DESC LIMIT ?", (run_id, limit)))

    def resolve(self, run_id: str, action: str) -> None:
        """Operator actions that never depend on a run's results.

        restart       an interrupted run, same seed, from scratch (reproduces it bit for bit)
        reevaluate    an eval_failed run (evaluation is deterministic)
        retry-ledger  a ledger_failed run, or an excluded run whose exclusion is not yet in the ledger
        unblock       a blocked run whose dependencies are no longer excluded, blocked or superseded
        replace       carries out the group's decision to repeat a run with the arm's next unused
                      seed: an excluded run left without a replacement (circuit breaker, or found
                      in the ledger), or a blocked run whose dependency was excluded. Refused for
                      a Study B continuation: it follows its parent's seed, so replace the parent
                      (which supersedes the parent's pending and blocked continuations); an
                      excluded continuation is left to the group's amendment.
        Excluding an interrupted run one by one is not offered: use `set-policy exclude`, which
        applies to every held run alike, once the group has decided (Q-interrupted-run).
        """
        if action not in RESOLVE_ACTIONS:
            raise ValueError(f"action must be one of {RESOLVE_ACTIONS}")
        row = self.row(run_id)
        status = row["status"]
        self._fix_mode()
        if action == "restart":
            if status != "interrupted":
                raise SchedulerError(f"{run_id} is {status}; only an interrupted run can be restarted")
            self.restart(run_id)
        elif action == "reevaluate":
            with self._write_txn():  # checked inside the write lock: never a second evaluation of a live one
                status = self.row(run_id)["status"]
                if status != "eval_failed":
                    raise SchedulerError(f"{run_id} is {status}; only an eval_failed run can be re-evaluated")
                self._set_sql(run_id, status="trained")
                self._event_sql(run_id, "reevaluate", "operator")
        elif action == "retry-ledger":
            if status == "ledger_failed":
                with self._write_txn():  # re-checked inside the write lock, like every other decision
                    status = self.row(run_id)["status"]
                    if status != "ledger_failed":
                        raise SchedulerError(f"{run_id} is {status}; nothing to retry")
                    self._set_sql(run_id, status="evaluated")
                    self._event_sql(run_id, "retry_ledger", "operator")
            elif status == "excluded":
                self._record_exclusion(run_id)
            else:
                raise SchedulerError(f"{run_id} is {status}; nothing to retry")
        elif action == "unblock":
            spec = self.spec(run_id)
            with self._write_txn():  # checked inside the write lock: never unblocked and replaced at once
                row = self.row(run_id)
                if row["status"] != "blocked" or row["replaced_by"]:
                    raise SchedulerError(f"{run_id} is {row['status']} (replaced_by={row['replaced_by']}); "
                                         "only a blocked run without a replacement can be unblocked")
                if self._dependencies_state(spec) == "blocked":
                    hint = (f"`resolve {spec.depends_on[0]} replace` replaces its parent, which supersedes it"
                            if spec.group == "study_b_fewshot" else "`replace` queues the arm's next unused seed instead")
                    raise SchedulerError(f"{run_id}: a dependency is still excluded, blocked or superseded; {hint}")
                self._set_sql(run_id, status="pending")
                self._event_sql(run_id, "unblocked", "operator")
        else:  # replace
            spec = self.spec(run_id)
            if spec.group == "study_b_fewshot":
                # A continuation takes its parent's seed: a new seed of its own would continue a parent
                # that does not exist. Replacing the parent queues the parent's new continuations.
                raise SchedulerError(f"{run_id} is a continuation; it follows its parent's seed, not a new one"
                                     + (f"; to repeat it, `resolve {spec.depends_on[0]} replace`" if status == "blocked" else
                                        "; the group decides by an amendment (Table 9.1)"))
            with self._write_txn():  # checked inside the write lock: two decisions never queue two seeds
                row = self.row(run_id)
                status = row["status"]
                if status not in ("excluded", "blocked") or row["replaced_by"]:
                    raise SchedulerError(f"{run_id} is {status} (replaced_by={row['replaced_by']}); "
                                         "only an excluded or blocked run without a replacement can be replaced")
                replacement = self._replacement_sql(run_id, spec, surplus=bool(row["surplus"]))
                self._event_sql(run_id, "replace", f"operator (group decision): {replacement.run_id}")
                if status == "blocked":  # decided: the replacement runs instead of it
                    self._set_sql(run_id, status="superseded")
                    self._event_sql(run_id, "superseded", f"replaced by {replacement.run_id} (group decision)")

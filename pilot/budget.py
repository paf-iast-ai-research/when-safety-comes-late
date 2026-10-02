"""Compute budget, machine-hour allocation and the surplus-seed rule (owner: pilot owner, Role 1).

* Part 6 G2, amended before the pilot (Table 9.1, Q-g2-run-equivalents): the run-equivalents of G2
  and of Part 5.5's surplus rule are training steps in units of T = 10,000,000, as Parts 3.3 and 6.1
  count them, so the decisive requirement is wall-clock per run / concurrent runs x the corrected
  total of ``corrected_run_equivalents`` for the design in force (``corrected_requirement``: 627.13
  for the uncut design). That total counts what the registered 484 leaves out (constrained-steps-
  matched arms train up to 2T; the data control trains T + N*T; the battery's fine-tuning and
  transfer continuations). ``registered_requirement`` applies the registered formula with 484
  (435 + 49); it is reported beside the corrected one and decides nothing. The continuations are
  counted as the manifest builds them (``manifest.battery_continuation``): one per condition of
  ``manifest.BATTERY_CONDITIONS`` for every Study A run of the main study's parent groups, an upper
  bound, since Part 4.1 rule 6 leaves unmatched arms out of the battery. ``cuts`` prices the design
  after the first cuts of Part 6.1 (``manifest.apply_cuts``; a parent that is cut takes its
  continuations with it).
* Part 6 G2 / Part 8: the allocation is written before the pilot's throughput is read, "so that it
  cannot be set to fit the result". ``record_allocation`` writes ``pilot/allocation.json`` with the
  unit and the basis of H (Table 9.1, X-allocation) and the workstation timings that already
  existed (``prior_workstation_timings``); ``registered_runs_started`` says whether a registered run
  has started (``pilot allocate`` then refuses); ``allocation_contained`` checks from Git that every
  pilot run was launched from a commit that already contained exactly this allocation.
* Part 5.5: surplus compute goes to seeds first, in the order 5, 6, 7, ..., the same count for every
  arm in the primary-comparison set, up to twelve seeds; the surplus is the capacity beyond the
  corrected total, each seed priced in the same units (``surplus_plan``).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

from configs import registered as R
from pilot import manifest, provenance
from pilot.manifest import RunSpec

if TYPE_CHECKING:
    from pilot.ledger_writer import LedgerPaths

ALLOCATION_FILE = provenance.REPO_ROOT / "pilot" / "allocation.json"
# X-allocation (Table 9.1): H is elapsed time of the workstation with all its concurrent run slots together, the unit
# of G2's left side (hours per run / runs sustained concurrently x run-equivalents), never core-hours or hours per run
ALLOCATION_UNIT = "workstation wall-clock hours (all concurrent runs together)"
# Two years of elapsed time: a larger H is more likely core-hours than the workstation's elapsed hours, so it is
# recorded only when confirmed (`pilot allocate --confirm`)
ALLOCATION_CONFIRM_ABOVE = 24 * 731
# Table 2.2 "Reward-only fine-tuning" and "Transfer": the training steps of one parent's continuations,
# one per condition of manifest.BATTERY_CONDITIONS (manifest.battery_continuation sets these totals).
CONTINUATION_STEPS = MappingProxyType({"finetune": R.FINETUNE_STEPS, "transfer": R.TRANSFER_STEPS})
BATTERY_CONTINUATION_STEPS = sum(CONTINUATION_STEPS[c] for c in manifest.BATTERY_CONDITIONS)


def battery_parents(specs: Iterable[RunSpec]) -> list[RunSpec]:
    """The runs whose matched checkpoint the battery continues (``manifest.battery_continuation``
    accepts them): Study A runs of the main study's groups, never the pilot's (Part 3.6)."""
    return [s for s in specs if s.study == "A" and not s.pilot and s.group in manifest.STUDY_A_PARENT_GROUPS]


def registered_run_equivalents() -> int:
    """Part 6 G2: 484 = 435 (Study A) + 49 (Study B)."""
    return R.G2_RUN_EQUIVALENTS


def training_run_equivalents(specs: Iterable[RunSpec]) -> float:
    """Training steps of ``specs`` in units of T (no battery continuations)."""
    return sum(s.run_equivalents for s in specs)


def corrected_run_equivalents(include_battery_continuations: bool = True, cuts: Sequence[str] = ()) -> dict[str, float]:
    """Training steps of the whole registered design in units of T, by group.

    Constrained-steps-matched arms at N = 0.10 and 0.25 use the rounding answered in Table 9.1 (Q-rounding);
    the difference from the exact T/(1-N) is below 0.01 run-equivalents per run. ``cuts``: a
    prefix of ``manifest.CUTS`` (Part 6.1), applied to the design first.
    """
    groups: dict[str, float] = {}
    specs = manifest.apply_cuts(manifest.design("all"), cuts) if cuts else manifest.design("all")
    for spec in specs:
        groups[spec.group] = groups.get(spec.group, 0.0) + spec.run_equivalents
    if include_battery_continuations:
        groups["battery_continuations"] = len(battery_parents(specs)) * BATTERY_CONTINUATION_STEPS / R.TOTAL_STEPS
    groups["total"] = sum(groups.values())
    return groups


def _check_positive_hours(name: str, value: float) -> None:
    """ValueError unless ``value`` is a finite number > 0 (a bool or a string is not a number here)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")


def _check_throughput(hours_per_run: float, concurrent_runs: int) -> None:
    """ValueError unless hours_per_run is a finite number > 0 and concurrent_runs an integer >= 1."""
    _check_positive_hours("hours_per_run", hours_per_run)
    if isinstance(concurrent_runs, bool) or not isinstance(concurrent_runs, int):
        raise ValueError("concurrent_runs must be an integer")
    if concurrent_runs < 1:
        raise ValueError("concurrent_runs must be at least 1")


def registered_requirement(hours_per_run: float, concurrent_runs: int) -> float:
    """Part 6 G2, as registered: hours per run / concurrent runs x 484 (reported beside the corrected requirement;
    it decides nothing, Table 9.1, Q-g2-run-equivalents)."""
    _check_throughput(hours_per_run, concurrent_runs)
    return hours_per_run / concurrent_runs * registered_run_equivalents()


def corrected_requirement(hours_per_run: float, concurrent_runs: int) -> float:
    """Part 6 G2 as amended (Table 9.1, Q-g2-run-equivalents): the same formula with the corrected run-equivalent
    total of the uncut design (decisive)."""
    _check_throughput(hours_per_run, concurrent_runs)
    return hours_per_run / concurrent_runs * corrected_run_equivalents()["total"]


# ---------------------------------------------------------------------------
# Allocation (recorded before throughput is read)
# ---------------------------------------------------------------------------


def record_allocation(
    machine_hours: float, decided_by: str, meeting_date: str, basis: str, path: Path = ALLOCATION_FILE, *,
    prior_workstation_timings: Sequence[str] = (), confirm: bool = False,
) -> dict[str, Any]:
    """Write pilot/allocation.json once (Part 6 G2; Table 8.1); FileExistsError if it exists.

    X-allocation (Table 9.1): H (``machine_hours``) is the group's resource decision, the elapsed hours
    the workstation, with all its concurrent run slots together (``ALLOCATION_UNIT``, stored), is
    dedicated to the registered runs of the whole design; ``basis`` says how it was derived, from the
    calendar and the machine's availability only, never from a timing of a run on the workstation.
    ``prior_workstation_timings`` (``prior_workstation_timings()``) are stored as the record of which
    timings already existed. An H above ALLOCATION_CONFIRM_ABOVE (two years) is refused unless
    ``confirm``: it is likely core-hours. ValueError for any invalid input; nothing is written then.
    """
    _check_positive_hours("machine_hours", machine_hours)
    if not isinstance(decided_by, str) or not decided_by.strip():
        raise ValueError("decided_by must name who decided the allocation")
    if not isinstance(basis, str) or not basis.strip():
        raise ValueError("basis must say how the allocation was derived (dates and the machine's availability)")
    if not isinstance(meeting_date, str):
        raise ValueError(f"meeting_date must be a date YYYY-MM-DD, got {meeting_date!r}")
    date.fromisoformat(meeting_date)  # ValueError unless YYYY-MM-DD
    if machine_hours > ALLOCATION_CONFIRM_ABOVE and not confirm:
        raise ValueError(f"{machine_hours:.10g} hours exceed {ALLOCATION_CONFIRM_ABOVE} (two years of elapsed time); "
                         f"the allocation is in {ALLOCATION_UNIT}, not core-hours or hours per run: confirm it if it "
                         "is (`pilot allocate --confirm`)")
    record = {
        "machine_hours_allocated": float(machine_hours),
        "unit": ALLOCATION_UNIT,
        "basis": basis,
        "decided_by": decided_by,
        "meeting_date": meeting_date,
        "prior_workstation_timings": list(prior_workstation_timings),
        "recorded_utc": provenance.utc_now().isoformat(),
        "prereg": "Part 6 G2; Part 8 Table 8.1 'Machine-hours allocated to the registered runs'",
    }
    try:  # atomic and once only: never overwrites an earlier allocation, never leaves a partial one
        provenance.write_text_once(Path(path), json.dumps(record, indent=2, sort_keys=True) + "\n")
    except FileExistsError:
        raise FileExistsError(f"{path} exists; the allocation is recorded once (change it only by amendment)") from None
    return record


def stored_ledgers(value: str) -> list[Path]:
    """The paths of a data root's stored ``ledgers`` setting (``Scheduler._ledger_location``): the main ledger, the
    pilot ledger and the sidecar directory, real paths, or ``<repo>/``-relative inside the repository."""
    prefix = "<repo>/"
    root = Path(provenance.REPO_ROOT)
    return [root / p[len(prefix):] if p.startswith(prefix) else Path(p) for p in json.loads(value)]


def prior_workstation_timings(smoke_roots: Iterable[str | Path] = ()) -> list[str]:
    """The records whose wall-clock times already show the workstation's throughput: every determinism report
    under ledger/determinism/ (Table 3.1's check, run before any arm; repository-relative) and, for each of
    ``smoke_roots`` (scripts/smoke_run.py --data-root), its ledgers (those its scheduler state stored, ``schedule
    run --ledger-dir``, else ``ledger_writer.smoke_paths``), its sidecar directory when it holds records, and the
    train_result.json (wall_clock_hours) of every run trained there, ledgered or not (a smoke run too short to be
    evaluated reaches no ledger), those that exist. G2 fixes the allocation before throughput is read, so
    ``record_allocation`` stores this list (X-allocation)."""
    from pilot.enrichment import read_scheduler_state
    from pilot.ledger_writer import smoke_paths
    from pilot.scheduler import DETERMINISM_DIR, SchedulerConfig

    root = Path(provenance.REPO_ROOT)
    found = sorted(p.relative_to(root).as_posix() for p in (root / DETERMINISM_DIR).glob("*.json"))
    for data_root in smoke_roots:
        state = read_scheduler_state(data_root)
        stored = None if state is None else state.settings.get("ledgers")
        smoke = smoke_paths(Path(data_root))
        main, pilot, sidecar = stored_ledgers(stored) if stored else (smoke.main, smoke.pilot, smoke.sidecar_dir)
        found += [str(p) for p in (main, pilot) if p.exists()]
        if sidecar.is_dir() and any(sidecar.iterdir()):
            found.append(str(sidecar))
        runs = SchedulerConfig(data_root=Path(data_root)).runs_root
        found += sorted(str(p) for p in runs.glob("*/train_result.json"))
    return found


def registered_runs_started(paths: LedgerPaths | None = None,
                            data_roots: Iterable[str | Path] | None = None) -> list[str]:
    """Why a registered run has already started (empty if none has): the pilot ledger exists, its sidecar
    directory holds records, or a run of a registered data root's scheduler state (``data_roots``, default
    ``scheduler.REGISTERED_DATA_ROOTS``) has left 'pending' or has a run directory. Its wall-clock time may then
    be read, and Part 6 G2 fixes the allocation before that ("recorded before the throughput measurement is
    read", Table 8.1), so ``pilot allocate`` refuses (X-allocation). ``paths``: the ledgers (default
    ``ledger_writer.DEFAULT_PATHS``)."""
    from pilot.enrichment import read_scheduler_state
    from pilot.ledger_writer import DEFAULT_PATHS
    from pilot.scheduler import REGISTERED_DATA_ROOTS, SchedulerConfig

    paths = DEFAULT_PATHS if paths is None else paths
    found = [f"the pilot ledger {paths.pilot} exists"] if Path(paths.pilot).exists() else []
    sidecar = Path(paths.sidecar_dir)
    records = sorted(sidecar.glob("*.json")) if sidecar.is_dir() else []
    if records:
        found.append(f"{sidecar} holds {_counted(len(records), 'record')} (e.g. {records[0].name})")
    for data_root in REGISTERED_DATA_ROOTS if data_roots is None else data_roots:
        state = read_scheduler_state(data_root)
        config = SchedulerConfig(data_root=Path(data_root))
        started = [run_id for run_id, status in (state.statuses.items() if state is not None else ())
                   if status != "pending" or config.run_dir(run_id).exists()]
        if started:
            found.append(f"{_counted(len(started), 'run')} of the data root {data_root} started (e.g. {started[0]})")
    return found


def _counted(n: int, noun: str) -> str:
    """``n`` and ``noun``, in the plural unless ``n`` is 1 ("1 record", "2 records")."""
    return f"{n} {noun}{'' if n == 1 else 's'}"


def load_allocation(path: Path = ALLOCATION_FILE) -> dict[str, Any]:
    """The allocation record written by ``record_allocation``."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def allocation_contained(records: Iterable[dict], path: Path = ALLOCATION_FILE) -> tuple[bool, str]:
    """Part 6 G2: was this exact allocation fixed before any pilot run was launched?

    Dates in Git can be set by the committer, so containment is checked instead: every pilot run
    records the commit it was launched from, and each of those commits must already contain
    pilot/allocation.json with exactly the content in use now. The scheduler refuses to launch a
    pilot run from a commit without the file, so a later edit cannot pass this check.
    """
    path = Path(path)
    try:
        rel = str(path.resolve().relative_to(provenance.REPO_ROOT))
    except ValueError:
        return False, f"{path} is not inside the repository; the allocation must be committed in it"
    if not path.exists():
        return False, f"{rel} does not exist; record the allocation with `python -m pilot allocate` and commit it"
    if rel in provenance.dirty_paths() or provenance.file_committed_at("HEAD", rel) is None:
        return False, f"{rel} has uncommitted changes or is not committed"
    current = path.read_bytes()
    commits = {r.get("commit_hash") for r in records if r.get("commit_hash")}
    if not commits:
        return False, "no pilot run records a launch commit yet"
    bad = sorted(c for c in commits if provenance.file_committed_at(c, rel) != current)
    if bad:
        return False, f"pilot runs were launched from commits without this allocation: {bad}"
    return True, f"every pilot run was launched from a commit that already held this allocation ({len(commits)} commits checked)"


# ---------------------------------------------------------------------------
# Surplus seeds (Part 5.5)
# ---------------------------------------------------------------------------


def primary_comparison_specs(seed: int) -> list[RunSpec]:
    """Arms that receive surplus seeds, for one seed (answered in Table 9.1, Q-surplus-arm-set).

    Part 5.5: "the N = 0 and N = 0.50 arms under both step-matching controls on
    SafetyPointGoal1-v0, and the seven arms of Study B": no onset shape is excluded, so both shapes
    of the N = 0.50 arms are included, and the seven Study B arms with their few-shot continuations.
    The ramp arms' surplus seeds carry ``Q-surplus-arm-set`` in ``pending`` (``manifest.surplus_spec``),
    which holds them only while the key is open.
    """
    first = R.SEEDS[0]
    study_a = [manifest.surplus_spec(s, seed) for s in manifest.study_a_main((first,)) if manifest.is_primary_comparison(s)]
    b = [manifest.surplus_spec(s, seed) for s in manifest.study_b((first,))]
    return study_a + b + manifest.study_b_fewshot(b)


def registered_units(specs: Iterable[RunSpec]) -> float:
    """Run-equivalents in the units of the registered 484: each Study A run counts 1 (Part 6 G2's
    435 is a count of runs) and Study B counts training steps / T (Part 3.3's 49); reported only."""
    return sum(1.0 if s.study == "A" else s.run_equivalents for s in specs)


def corrected_units(specs: Iterable[RunSpec]) -> float:
    """Training steps / T plus the battery's continuations of every battery parent (``battery_parents``): the
    units of the corrected total (Table 9.1, Q-g2-run-equivalents)."""
    specs = list(specs)
    battery = len(battery_parents(specs)) * BATTERY_CONTINUATION_STEPS / R.TOTAL_STEPS
    return training_run_equivalents(specs) + battery


@dataclass(frozen=True)
class SurplusPlan:
    """The result of ``surplus_plan``.

    Capacities, surpluses and costs are in run-equivalents (one 10M-step run each): the decisive fields
    in training steps / T plus battery continuations (``corrected_units``, beyond the corrected total;
    Table 9.1, Q-g2-run-equivalents), the ``registered_`` fields in the units of the registered 484
    (``registered_units``), reported only. ``extra_seeds``, ``seeds_per_primary_arm`` and
    ``registered_extra_seeds`` count seeds per primary-comparison arm; ``added_seeds`` are the seed
    numbers added; ``warning`` is empty, or says that the ramp arms' surplus seeds wait while Q-surplus-arm-set
    is open.
    """

    capacity_run_equivalents: float
    surplus_run_equivalents: float
    cost_per_extra_seed: float
    extra_seeds: int
    seeds_per_primary_arm: int
    added_seeds: tuple[int, ...]
    registered_surplus_run_equivalents: float
    registered_cost_per_extra_seed: float
    registered_extra_seeds: int
    warning: str


def surplus_plan(
    machine_hours_allocated: float,
    hours_per_run_equivalent: float,
    concurrent_runs: int,
) -> SurplusPlan:
    """How many seeds beyond five the primary-comparison arms get (Part 5.5).

    ``hours_per_run_equivalent`` is the ``hours_per_run`` of ``corrected_requirement``: the measured
    wall-clock hours of one 10M-step run (one run-equivalent).

    Decisive (Table 9.1, Q-g2-run-equivalents): the capacity beyond the corrected total of
    ``corrected_run_equivalents``, with each extra seed priced in the same units (``corrected_units``).
    Reported beside it: the same computation with the registered 484 and ``registered_units``.
    """
    _check_positive_hours("machine_hours_allocated", machine_hours_allocated)
    _check_throughput(hours_per_run_equivalent, concurrent_runs)
    capacity = machine_hours_allocated * concurrent_runs / hours_per_run_equivalent
    max_extra = R.MAX_SEEDS_PER_ARM - len(R.SEEDS)
    primary = primary_comparison_specs(R.SURPLUS_SEED_START)

    def extra_for(surplus: float, per_seed: float) -> int:
        return 0 if surplus <= 0 else min(max_extra, math.floor(surplus / per_seed))

    surplus = capacity - corrected_run_equivalents()["total"]
    per_seed = corrected_units(primary)
    extra = extra_for(surplus, per_seed)
    registered_surplus = capacity - registered_run_equivalents()
    registered_per_seed = registered_units(primary)
    registered_extra = extra_for(registered_surplus, registered_per_seed)
    warning = ""
    if extra and R.is_open("Q-surplus-arm-set"):
        warning = ("Q-surplus-arm-set is open: the surplus seeds of the N = 0.50 ramp arms are priced in but wait "
                   "on the group's answer (Part 5.5 names no onset shape)")
    added = tuple(range(R.SURPLUS_SEED_START, R.SURPLUS_SEED_START + extra))
    return SurplusPlan(capacity, surplus, per_seed, extra, len(R.SEEDS) + extra, added,
                       registered_surplus, registered_per_seed, registered_extra, warning)


def as_dict(plan: SurplusPlan) -> Mapping[str, object]:
    """The plan as a JSON-serialisable mapping (for the ``surplus`` command)."""
    return asdict(plan)

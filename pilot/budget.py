"""Compute budget, machine-hour allocation and the surplus-seed rule.

* Part 6 G2: required machine-hours = wall-clock per run / concurrent runs x 484 run-equivalents.
  ``registered_requirement`` applies that formula exactly. ``corrected_run_equivalents`` also
  counts what the 484 leaves out (constrained-steps-matched arms train up to 2T; the data control
  trains T + N*T; the battery's fine-tuning and transfer continuations), and is reported beside
  the registered figure, never in its place, until the group amends G2.
* Part 6 G2 / Part 8: the allocation is written before the pilot's throughput is read, "so that it
  cannot be set to fit the result". ``record_allocation`` writes ``pilot/allocation.json``;
  ``allocation_contained`` checks from Git that every pilot run was launched from a commit that
  already contained exactly this allocation.
* Part 5.5: surplus compute goes to seeds first, in the order 5, 6, 7, ..., the same count for every
  arm in the primary-comparison set, up to twelve seeds.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Mapping

from configs import registered as R
from pilot import manifest, provenance
from pilot.manifest import RunSpec

ALLOCATION_FILE = provenance.REPO_ROOT / "pilot" / "allocation.json"
# Table 2.2; counted for every Study A run (an upper bound: Part 4.1 leaves unmatched arms out of the battery)
BATTERY_CONTINUATION_STEPS = R.FINETUNE_STEPS + R.TRANSFER_STEPS


def registered_run_equivalents() -> int:
    """Part 6 G2: 484 = 435 (Study A) + 49 (Study B)."""
    return R.G2_RUN_EQUIVALENTS


def training_run_equivalents(specs: Iterable[RunSpec]) -> float:
    return sum(s.run_equivalents for s in specs)


def corrected_run_equivalents(include_battery_continuations: bool = True) -> dict[str, float]:
    """Training steps of the whole registered design in units of T, by group.

    Constrained-steps-matched arms at N = 0.10 and 0.25 use the proposed rounding (Q-rounding);
    the difference from the exact T/(1-N) is below 0.01 run-equivalents per run.
    """
    groups: dict[str, float] = {}
    specs = manifest.design("all")
    for spec in specs:
        groups[spec.group] = groups.get(spec.group, 0.0) + spec.run_equivalents
    if include_battery_continuations:
        study_a_runs = sum(1 for s in specs if s.study == "A")
        groups["battery_continuations"] = study_a_runs * BATTERY_CONTINUATION_STEPS / R.TOTAL_STEPS
    groups["total"] = sum(groups.values())
    return groups


def registered_requirement(hours_per_run: float, concurrent_runs: int) -> float:
    """Part 6 G2, as registered: hours per run / concurrent runs x 484."""
    if concurrent_runs < 1 or hours_per_run <= 0:
        raise ValueError("hours per run must be positive and concurrent runs at least 1")
    return hours_per_run / concurrent_runs * registered_run_equivalents()


def corrected_requirement(hours_per_run: float, concurrent_runs: int) -> float:
    """The same formula with the corrected run-equivalent total (reported, not decisive)."""
    if concurrent_runs < 1 or hours_per_run <= 0:
        raise ValueError("hours per run must be positive and concurrent runs at least 1")
    return hours_per_run / concurrent_runs * corrected_run_equivalents()["total"]


# ---------------------------------------------------------------------------
# Allocation (recorded before throughput is read)
# ---------------------------------------------------------------------------


def record_allocation(machine_hours: float, decided_by: str, meeting_date: str, path: Path = ALLOCATION_FILE) -> dict:
    if not math.isfinite(machine_hours) or machine_hours <= 0:
        raise ValueError("machine_hours must be a positive number")
    if not decided_by.strip():
        raise ValueError("decided_by must name who decided the allocation")
    date.fromisoformat(meeting_date)  # ValueError unless YYYY-MM-DD
    record = {
        "machine_hours_allocated": float(machine_hours),
        "decided_by": decided_by,
        "meeting_date": meeting_date,
        "recorded_utc": provenance.utc_now().isoformat(),
        "prereg": "Part 6 G2; Part 8 Table 8.1 'Machine-hours allocated to the registered runs'",
    }
    try:
        with open(path, "x", encoding="utf-8") as fh:  # "x": never overwrite an earlier allocation
            fh.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
    except FileExistsError:
        raise FileExistsError(f"{path} exists; the allocation is recorded once (change it only by amendment)") from None
    return record


def load_allocation(path: Path = ALLOCATION_FILE) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def allocation_contained(records: Iterable[dict], path: Path = ALLOCATION_FILE) -> tuple[bool, str]:
    """Part 6 G2: was this exact allocation fixed before any pilot run was launched?

    Dates in Git can be set by the committer, so containment is checked instead: every pilot run
    records the commit it was launched from, and each of those commits must already contain
    pilot/allocation.json with exactly the content in use now. The scheduler refuses to launch a
    pilot run from a commit without the file, so a later edit cannot pass this check.
    """
    rel = str(path.resolve().relative_to(provenance.REPO_ROOT))
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
    """Arms that receive surplus seeds, for one seed (proposed reading of Q-surplus-arm-set).

    Part 5.5: "the N = 0 and N = 0.50 arms under both step-matching controls on
    SafetyPointGoal1-v0, and the seven arms of Study B". The onset shape is not named; both shapes
    are included here, and the ramp arms' surplus seeds carry ``Q-surplus-arm-set`` in ``pending``
    (``manifest.surplus_spec``) until the group answers.
    """
    first = R.SEEDS[0]
    study_a = [manifest.surplus_spec(s, seed) for s in manifest.study_a_main((first,)) if manifest.is_primary_comparison(s)]
    b = [manifest.surplus_spec(s, seed) for s in manifest.study_b((first,))]
    return study_a + b + manifest.study_b_fewshot(b)


def registered_units(specs: Iterable[RunSpec]) -> float:
    """Run-equivalents in the units of the registered 484: each Study A run counts 1 (Part 6 G2's
    435 is a count of runs) and Study B counts training steps / T (Part 3.3's 49)."""
    return sum(1.0 if s.study == "A" else s.run_equivalents for s in specs)


def corrected_units(specs: Iterable[RunSpec]) -> float:
    """Training steps / T plus the battery's continuations for every Study A run."""
    specs = list(specs)
    battery = sum(1 for s in specs if s.study == "A") * BATTERY_CONTINUATION_STEPS / R.TOTAL_STEPS
    return training_run_equivalents(specs) + battery


@dataclass(frozen=True)
class SurplusPlan:
    capacity_run_equivalents: float
    surplus_run_equivalents: float
    cost_per_extra_seed: float
    extra_seeds: int
    seeds_per_primary_arm: int
    added_seeds: tuple[int, ...]
    corrected_surplus_run_equivalents: float
    corrected_cost_per_extra_seed: float
    corrected_extra_seeds: int
    warning: str


def surplus_plan(
    machine_hours_allocated: float,
    hours_per_run_equivalent: float,
    concurrent_runs: int,
) -> SurplusPlan:
    """How many seeds beyond five the primary-comparison arms get (Part 5.5).

    Decisive, as registered: capacity beyond the 484 run-equivalents, with each extra seed priced
    in the same units. Reported beside it: the same computation with the corrected totals, and a
    warning when it would allow fewer seeds.
    """
    for name, value in (("machine_hours_allocated", machine_hours_allocated), ("hours_per_run_equivalent", hours_per_run_equivalent)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
    if isinstance(concurrent_runs, bool) or not isinstance(concurrent_runs, int):
        raise ValueError("concurrent_runs must be an integer")
    if hours_per_run_equivalent <= 0 or concurrent_runs < 1 or machine_hours_allocated <= 0:
        raise ValueError("hours per run-equivalent and the allocation must be positive, concurrent runs at least 1")
    capacity = machine_hours_allocated * concurrent_runs / hours_per_run_equivalent
    max_extra = R.MAX_SEEDS_PER_ARM - len(R.SEEDS)
    primary = primary_comparison_specs(R.SURPLUS_SEED_START)

    def extra_for(surplus: float, per_seed: float) -> int:
        return 0 if surplus <= 0 else min(max_extra, int(math.floor(surplus / per_seed)))

    surplus = capacity - registered_run_equivalents()
    per_seed = registered_units(primary)
    extra = extra_for(surplus, per_seed)
    corrected_surplus = capacity - corrected_run_equivalents()["total"]
    corrected_per_seed = corrected_units(primary)
    corrected_extra = extra_for(corrected_surplus, corrected_per_seed)
    warnings = []
    if corrected_extra < extra:
        warnings.append(
            f"the registered computation allows {extra} extra seeds but the corrected totals allow only "
            f"{corrected_extra}; raise Q-g2-run-equivalents (HANDOVER.md, section 9) with the group before adding seeds"
        )
    if extra and R.is_open("Q-surplus-arm-set"):
        warnings.append(
            "Q-surplus-arm-set is open: the surplus seeds of the N = 0.50 ramp arms are priced in but wait "
            "on the group's answer (Part 5.5 names no onset shape)"
        )
    warning = "; ".join(warnings)
    added = tuple(range(R.SURPLUS_SEED_START, R.SURPLUS_SEED_START + extra))
    return SurplusPlan(capacity, surplus, per_seed, extra, len(R.SEEDS) + extra, added,
                       corrected_surplus, corrected_per_seed, corrected_extra, warning)


def as_dict(plan: SurplusPlan) -> Mapping[str, object]:
    """The plan as a JSON-serialisable mapping (for the ``surplus`` command)."""
    return asdict(plan)

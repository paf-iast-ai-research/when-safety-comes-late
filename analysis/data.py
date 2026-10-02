"""The ledger and the supplement records as per-seed tables (Part 5.1; results/supplement_schema.py; analysis/__main__.py).

Owner: Role 4, Analysis and results (Muhammad Abdullah).

Part 5.1: "The unit of analysis is the seed in both studies ... Within a seed, the 100 evaluation
episodes, the four battery conditions, and in Study B the four unseen budgets are repeated measures;
they are summarised into one number per seed per outcome before any interval is computed". Each
completed run is one seed of one arm, so the tables here hold one record per run.

Sources:
* the ledger, read with ``results.ledger_schema.load_ledger_as_rows`` (validated rows; the float
  map keys restored, which plain pandas would leave as strings);
* the supplement records of ``results.supplement_schema`` (``<ledger dir>/supplement`` by
  results/supplement_schema.py), found by walking the directory: a record names its own kind and run, so the reader does not
  depend on the writer's file layout.

Part 5.6: "A run is excluded only if it crashes, fails to complete its steps, or produces a
non-finite loss or multiplier ... No run is excluded on the basis of its results." Of the rows of the
population analysed, only those with ``completed`` false leave the tables as exclusions (listed in
``exclusions`` with their cause); auxiliary runs (Q-warm-start) are not seeds of any arm and are listed in
``Dataset.auxiliary``; rows of other populations are listed in ``Dataset.ignored``; and a row of neither
study, or of a Study B arm that is not registered, is not read. A missing field makes the affected
analysis incomplete, never a silent drop.

Part 3.6: "The pilot's runs are not reused": the final mode refuses pilot rows (run_ids ``P-`` or
``P1-``); the pilot mode reads the rows of one pilot (revision 0 or 1) only.

Few-shot censoring (Table 2.5 "Adaptation steps": "recorded as above the largest horizon if never
reached"; docs/DECISIONS.md, Q-adapt-censoring: "Readers decode a value greater than the largest horizon as
censored"; ``studyb.evaluation.censored_steps``): a value is censored when it lies above the largest horizon
of THAT continuation (from its supplement record, else the horizons its ``sr_fewshot`` keys hold),
never by equality with a constant, so a continuation cut to 200,000 steps (Part 6.1) is read right.

Rule 1 is re-checked (Part 4.1 rule 1; pipeline contract 4 of pilot/contracts.py, whose step
``pilot.enrichment.rule_one_step`` recomputes when it is stored): every completed Study A row with a
stored matched step is run through ``analysis.matching.select_checkpoint`` on its own checkpoint
records, and a different step (or a window rule 1 cannot rank) is listed among the data problems.

A row the frozen schema accepts but the analysis cannot read (a few-shot key not in the canonical
form, a run_id that does not end with its seed, a seed twice in an arm, two N = 0 arms in a task, a
completed Study A row without N) is a ``DataError`` naming the run: ``python -m analysis`` refuses
it (exit 2), it never crashes. (The schema corrections of 2026-10-01, an amendment decided on 2026-10-02 and to
be ratified with the other decisions, docs/ledger_schema_memo.md, make the schema itself refuse the first and the
last; the checks here stay for a ledger read with the schema as frozen.)

The recorded Part 6.1 cuts (pilot.scheduler.CUTS_PATH: the committed ``pilot/cuts.json``, {"cuts": [...],
"amendment": "..."}) are read so that an arm a cut removed is told apart from one that is missing
(``Dataset.cuts``); boxes H1 and H2 read their N = 0.10 clauses on the arms that remain only after the
cut "drop_n010" (analysis.study_a). ``python -m analysis`` passes the cuts it read from the bytes it
checked against the commit (``parse_cuts``). In final mode the registered Study A arms the cuts leave
(``awaited_arms``) are expected: one without a completed run is data still to come
(``Dataset.awaited``), so a Holm family keeps it as an incomplete member and never shrinks on it
(analysis.study_a; Q-holm-families); only a cut arm, or one the design does not have, is not tested.
Likewise a Study B arm with fewer completed runs than its seed target (``study_b_seed_targets``;
Q-arm-complete) is data still to come (``Dataset.awaited_b``): its comparisons are incomplete
(analysis.study_b), never decided on the seeds that happen to be present. So is a Study A arm with some but
fewer completed runs than its seed target (``study_a_seed_targets``; ``Dataset.awaited_a``; analysis.study_a).
An arm with MORE completed runs than its target (5, or 5 + E for a primary-comparison arm) means the
given E is not the data root's surplus (a Part 5.6 replacement takes the place of a run that is not completed):
``build_dataset`` refuses it (DataError) rather than decide on whichever surplus seeds happen to be completed.

Provenance: the ledger's SHA-256 is taken before it is read and again after the supplement is read
(``Dataset.ledger_sha256``; a ledger written meanwhile is refused), and each supplement file is parsed
from the bytes whose SHA-256 is recorded (``Supplement.digests``), so the report identifies the data it
analysed. The order follows the writer's: pilot/ledger_writer.py writes a run's supplement record BEFORE its row,
so every row the ledger held when it was read has its records on disk when the supplement is scanned
after it; a row appended during the scan changes the ledger's hash and is refused.

A supplement record is read only as the checkpoint it belongs to (Part 4.1): the battery, sensitivity
battery (rule 7 (a): the selected checkpoint, matched at tolerance 5.0), continuation and measurement
records at the ledger's matched_checkpoint_step with its measurement_cost as C_ID, and the final
battery at the run's final checkpoint (rule 7 (b)); Study B's zero-shot record, and each few-shot record's parent
step, at the run's final checkpoint (Q-studyb-eval; Q-continuations), each of the row's arm. A record of another
checkpoint, another C_ID or another arm is a data problem and is not read (the analyses that need it are
incomplete), never silently used.

Pilot mode is the Part 5.8 test run ("is run once on the pilot data to test it"): a selected pilot
population with no completed Study A run is refused (DataError), so the test run cannot succeed
without testing anything (and ``python -m analysis`` exits 4 after writing a pilot report in which nothing
could be compared with the go report; analysis.pilot_check).
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional

from configs import registered as R

from analysis import matching

MODES = ("pilot", "final")
CONDITIONS = ("hazard", "dynamics", "finetune", "transfer")  # Table 2.2; the ledger's gap_* fields
REPO_ROOT = Path(__file__).resolve().parents[1]
CUTS_FILE = REPO_ROOT / "pilot" / "cuts.json"  # pilot.scheduler.CUTS_PATH: the committed record of the Part 6.1 cuts
CUT_DROP_N010 = "drop_n010"  # Part 6.1 cut 5, "Drop N = 0.10 from the main sweep" (pilot.manifest.CUTS; a test checks it)


class DataError(ValueError):
    """The ledger or the supplement cannot be analysed as it stands (a refusal, not a result)."""


def default_supplement_dir(ledger_path: Path) -> Path:
    """results/supplement_schema.py: the supplement of a ledger lives beside it, ``<ledger dir>/supplement``."""
    return Path(ledger_path).parent / "supplement"


def sha256_bytes(content: bytes) -> str:
    """SHA-256 of bytes (hex)."""
    return hashlib.sha256(content).hexdigest()


def file_sha256(path: Path) -> str:
    """SHA-256 of a file's bytes."""
    return sha256_bytes(Path(path).read_bytes())


# ---------------------------------------------------------------------------
# Supplement
# ---------------------------------------------------------------------------


@dataclass
class Supplement:
    """Validated supplement records: {kind: {(run_id, part): model}}."""

    root: Optional[Path] = None
    records: dict[str, dict[tuple[str, Optional[str]], Any]] = field(default_factory=dict)
    files: list[str] = field(default_factory=list)
    digests: dict[str, str] = field(default_factory=dict)  # relative path -> SHA-256 of the bytes parsed

    def sha256(self) -> str:
        """One SHA-256 over every file read: its relative path and the SHA-256 of its bytes, by path."""
        digest = hashlib.sha256()
        for rel in sorted(self.digests):
            digest.update(rel.encode("utf-8") + b"\0" + self.digests[rel].encode("ascii") + b"\0")
        return digest.hexdigest()

    def get(self, kind: str, run_id: str, part: Optional[str] = None) -> Any:
        return self.records.get(kind, {}).get((run_id, part))

    def of_kind(self, kind: str) -> dict[tuple[str, Optional[str]], Any]:
        return self.records.get(kind, {})


def load_supplement(root: Optional[Path]) -> Supplement:
    """Every ``*.json`` under ``root`` validated as a supplement record (DataError on any bad file).

    A record's kind is its ``kind`` field, else the first directory below ``root`` when that is a
    kind; its run and part come from its content (``results.supplement_schema.record_part``). A
    second record for the same (kind, run, part) is refused: a record is written once.
    """
    from results import supplement_schema as S

    out = Supplement(root=None if root is None else Path(root))
    if root is None or not Path(root).exists():
        return out
    root = Path(root)
    for path in sorted(root.rglob("*.json")):
        rel = path.relative_to(root).as_posix()
        try:
            content = path.read_bytes()  # parsed and hashed from the same bytes (provenance)
            data = json.loads(content.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DataError(f"supplement file {rel} is not readable JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise DataError(f"supplement file {rel} does not hold a record")
        top = rel.split("/", 1)[0] if "/" in rel else None
        kind = data.get("kind") or (top if top in S.KIND_MODELS else None)
        if kind is None:
            raise DataError(f"supplement file {rel} names no kind")
        if not isinstance(kind, str) or kind not in S.KIND_MODELS:  # a list or object is a refusal, not a TypeError
            raise DataError(f"supplement file {rel} names kind {kind!r}, not one of {list(S.KIND_MODELS)}")
        try:
            model = S.validate_record(kind, data)
        except ValueError as exc:
            raise DataError(f"supplement file {rel} is not a valid {kind} record: {exc}") from exc
        key = (model.run_id, S.record_part(kind, model))
        bucket = out.records.setdefault(kind, {})
        if key in bucket:
            raise DataError(f"two {kind} records for {key} (the second is {rel})")
        bucket[key] = model
        out.files.append(rel)
        out.digests[rel] = sha256_bytes(content)
    return out


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


def load_rows(ledger_path: Path) -> list[dict[str, Any]]:
    """The ledger's rows as validated dicts (``LedgerRow.model_dump``)."""
    from results.ledger_schema import load_ledger_as_rows

    return [row.model_dump(mode="python") for row in load_ledger_as_rows(Path(ledger_path))]


def pilot_population(revision: int) -> str:
    """The run_id prefix of the pilot (0) or the re-pilot (1) (pilot.manifest.pilot_prefix)."""
    from pilot.manifest import pilot_prefix

    return pilot_prefix(revision)


def group_of(row: Mapping[str, Any]) -> str:
    """The design group of a Study A row (Tables 3.2 and 3.3), from its factors."""
    if row.get("treatment") is not None:
        return "treatment"
    variant = row.get("controller_variant")
    if variant == R.PID_VARIANT:
        return "pid"
    if variant is not None:
        return "controller"
    return "main"


def _onset_from_design(row: Mapping[str, Any]) -> Optional[int]:
    """Onset step implied by the design (pilot.manifest.onset_schedule) when no training record gives it."""
    from pilot.manifest import onset_schedule

    if row.get("N") is None:
        return None
    if row["N"] == R.ONSET_FRACTIONS[0]:
        return 0
    if row.get("step_matching") is None:
        return None
    return onset_schedule(float(row["N"]), str(row["step_matching"]))[0]


def _min_window_cost(checkpoints: list[Mapping[str, Any]]) -> Optional[float]:
    window = matching.selection_window(checkpoints)  # rule 1's window (end-relative off the grid: Q-selection-window)
    costs = [cost for cost in (finite_or_none(c.get("selection_cost")) for c in window) if cost is not None]
    return min(costs) if costs else None


# The parameter norms of Table 2.3 ("Parameter norm", "the two critics logged separately"): the
# actor norm over trainable parameters, over all parameters, and each critic's norm (exploratory, descriptive).
NORM_FIELDS = ("norm", "norm_all", "norm_reward_critic", "norm_cost_critic")


def finite_or_none(value: Any) -> Optional[float]:
    """A float if finite, else None."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _checkpoint_norms(row: Mapping[str, Any]) -> tuple[Optional[float], Optional[float]]:
    """The actor norm of the ledger's checkpoint record at the matched checkpoint and at the final one."""
    by_step = {int(c["step"]): c for c in row.get("checkpoints") or []}
    matched_step = row.get("matched_checkpoint_step")
    matched = by_step.get(int(matched_step)) if matched_step is not None else None
    final = by_step[max(by_step)] if by_step else None
    return (None if matched is None else finite_or_none(matched.get("norm")),
            None if final is None else finite_or_none(final.get("norm")))


def _parse_fewshot_key(key: str, run_id: str) -> tuple[float, int]:
    from pilot.contracts import parse_fewshot_key

    try:
        return parse_fewshot_key(key)
    except ValueError as exc:
        raise DataError(f"{run_id}: sr_fewshot: {exc}") from None


def _arm_id(row: Mapping[str, Any]) -> str:
    try:
        return matching.arm_id(row)
    except ValueError as exc:
        raise DataError(f"{row.get('run_id')}: {exc}") from None


def _population(run_id: str) -> str:
    try:
        return matching.population(run_id)
    except ValueError as exc:
        raise DataError(str(exc)) from None


def _population_or_none(run_id: str) -> Optional[str]:
    """The population of a run id, or None for an id matching.population cannot read."""
    try:
        return matching.population(run_id)
    except ValueError:
        return None


def parse_cuts(content: Optional[bytes], source: str = CUTS_FILE.relative_to(REPO_ROOT).as_posix()) -> tuple[str, ...]:
    """The Part 6.1 cuts in a cut record's bytes (pilot.scheduler.CUTS_PATH); () when there is no record (``content`` None).

    The record is checked as ``pilot.scheduler.committed_cuts`` checks it before ``schedule add --cuts K`` applies
    it: a list of cut names, a prefix of the registered order, and the amendment that records the decision. A
    record the scheduler refuses is refused here too (DataError), so the analysis never reads as cut an arm the
    scheduler queued uncut.
    """
    from pilot.manifest import CUTS

    if content is None:
        return ()
    try:
        record = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise DataError(f"{source} does not hold a cut record {{'cuts': [...], 'amendment': ...}}: {exc}") from None
    cuts = record.get("cuts") if isinstance(record, dict) else None
    amendment = record.get("amendment") if isinstance(record, dict) else None
    if not isinstance(cuts, list) or not all(isinstance(c, str) for c in cuts):
        raise DataError(f"{source} does not hold a cut record {{'cuts': [...], 'amendment': ...}}")
    if not isinstance(amendment, str) or not amendment.strip():
        raise DataError(f"{source} names no amendment; the cut decision is recorded by one (Table 9.1)")
    cuts = tuple(cuts)
    if cuts != CUTS[: len(cuts)]:
        raise DataError(f"recorded cuts {cuts} are not a prefix of the registered order {CUTS} (Part 6.1)")
    return cuts


def load_cuts(path: Optional[Path] = None) -> tuple[str, ...]:
    """The Part 6.1 cuts the group recorded (pilot.scheduler.CUTS_PATH ``pilot/cuts.json``); () when there is no record."""
    path = CUTS_FILE if path is None else Path(path)
    if not path.exists():
        return ()
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise DataError(f"{path} cannot be read: {exc}") from None
    return parse_cuts(content, str(path))


def _rule_one_recheck(row: Mapping[str, Any], problems: list[str]) -> Optional[bool]:
    """Rule 1 re-run on the row's checkpoints; True when it gives the stored matched step (Part 4.1 rule 1)."""
    stored = row.get("matched_checkpoint_step")
    if stored is None:
        return None
    try:
        chosen = matching.select_checkpoint(list(row.get("checkpoints") or []))
    except ValueError as exc:
        problems.append(f"{row['run_id']}: rule 1 cannot be re-checked on the stored checkpoints ({exc})")
        return False
    if chosen != stored:
        problems.append(f"{row['run_id']}: rule 1 on the stored checkpoints selects step {chosen}, the ledger's "
                        f"matched_checkpoint_step is {stored} (Part 4.1 rule 1; the measured cost and gaps are "
                        "of another checkpoint than the rule chooses)")
        return False
    return True


@dataclass
class Dataset:
    """The per-seed tables of one analysis run."""

    mode: str
    population: str
    rows: list[dict[str, Any]]
    study_a: list[dict[str, Any]]
    study_b: list[dict[str, Any]]
    exclusions: list[dict[str, Any]]
    supplement: Supplement
    problems: list[str] = field(default_factory=list)
    ignored: list[str] = field(default_factory=list)
    cuts: tuple[str, ...] = ()  # the recorded Part 6.1 cuts (load_cuts)
    ledger_sha256: Optional[str] = None  # the SHA-256 of the ledger bytes read (load_dataset)
    auxiliary: list[str] = field(default_factory=list)  # auxiliary runs left out of their arms (Q-warm-start)
    # final mode: the registered Study A arms the recorded cuts leave that have no completed run yet (``awaited_arms``)
    awaited: list[str] = field(default_factory=list)
    # final mode: each Study B arm below its seed target (``study_b_seed_targets``; Q-arm-complete) -> (completed
    # seeds, target); its comparisons are incomplete (analysis.study_b), never decided on the seeds present
    awaited_b: dict[str, tuple[int, int]] = field(default_factory=dict)
    surplus_extra: Optional[int] = None  # final mode: E of the seed targets (Part 5.5)
    # final mode: each Study A arm with some completed runs but fewer than its seed target (``study_a_seed_targets``;
    # Q-arm-complete) -> (completed seeds, target); its comparisons are incomplete (analysis.study_a)
    awaited_a: dict[str, tuple[int, int]] = field(default_factory=dict)

    def awaited_a_lines(self) -> list[str]:
        """``awaited_a`` as the report states it."""
        return [awaited_a_line(arm, n, target) for arm, (n, target) in sorted(self.awaited_a.items())]

    def awaited_b_lines(self) -> list[str]:
        """``awaited_b`` as the report states it."""
        return [f"B-{arm}: {n} of its {target} seeds completed (Q-arm-complete)"
                for arm, (n, target) in sorted(self.awaited_b.items())]

    def seeds_per_arm(self) -> dict[str, list[int]]:
        out: dict[str, list[int]] = {}
        for rec in self.study_a + self.study_b:
            out.setdefault(rec["arm_id"], []).append(int(rec["seed"]))
        return {k: sorted(v) for k, v in sorted(out.items())}


def _final_step(row: Mapping[str, Any], sup: Supplement) -> Optional[int]:
    """The run's final checkpoint: the evaluation record's final_step, else the last checkpoint of the ledger row."""
    evaluation = sup.get("evaluation", str(row["run_id"]))
    if evaluation is not None:
        return int(evaluation.final_step)
    steps = [int(c["step"]) for c in row.get("checkpoints") or []]
    return max(steps) if steps else None


def _same_cost(a: Optional[float], b: Optional[float]) -> bool:
    """Two C_ID values of one checkpoint agree (to the schema's tolerance for a mean of the same episodes)."""
    if a is None or b is None:
        return True  # nothing to compare
    from results import supplement_schema as S

    return math.isclose(float(a), float(b), rel_tol=S.MEAN_REL_TOL, abs_tol=S.MEAN_ABS_TOL)


def _of_checkpoint(run_id: str, what: str, step: int, expected: Optional[int], label: str, problems: list[str],
                   c_id: Optional[float] = None, expected_c_id: Optional[float] = None) -> bool:
    """True when a supplement record is of the checkpoint (and the C_ID) the analysis reads it as.

    Otherwise the difference is a data problem and the record is not read: a gap of another checkpoint
    would enter rule 7 or the final-checkpoint estimand as if it were this one's.
    """
    ok = True
    if expected is not None and int(step) != int(expected):
        problems.append(f"{run_id}: the {what} record is at step {step}, the {label} at {expected}; it is not read")
        ok = False
    if not _same_cost(c_id, expected_c_id):
        problems.append(f"{run_id}: the {what} record uses C_ID {c_id!r}, the ledger's measurement_cost is "
                        f"{expected_c_id!r}; it is not read")
        ok = False
    return ok


def _study_a_record(row: Mapping[str, Any], sup: Supplement, problems: list[str]) -> dict[str, Any]:
    run_id = str(row["run_id"])
    rec: dict[str, Any] = {k: row.get(k) for k in (
        "run_id", "study", "completed", "task", "arm", "N", "onset_shape", "step_matching", "treatment", "controller_variant", "seed",
        "measurement_cost", "selection_cost_at_match", "matched_checkpoint_step", "training_age",
        "lambda_at_selection", "final_cost", "final_return", "gap_hazard", "gap_dynamics", "gap_finetune",
        "gap_transfer", "dormant_onset", "rank_onset", "norm_onset", "notes",
    )}
    rec.update(arm_id=_arm_id(row), population=_population(run_id), group=group_of(row),
               stored_matched=row.get("matched"), stored_infeasible=row.get("infeasible"),
               min_window_selection_cost=_min_window_cost(row.get("checkpoints") or []),
               rule_one_rechecked=_rule_one_recheck(row, problems))
    steps = [c["step"] for c in row.get("checkpoints") or []]
    rec["norm_matched_checkpoint"], rec["norm_final_checkpoint"] = _checkpoint_norms(row)
    training = sup.get("training", run_id)
    # Table 2.3's norms of the plasticity.csv rows at onset and at onset + 200,000 (training supplement record)
    rec["plasticity_norms"] = {
        when: {name: None if snapshot is None else finite_or_none(getattr(snapshot, name)) for name in NORM_FIELDS}
        | {"step": None if snapshot is None else snapshot.step}
        for when, snapshot in (("onset", None if training is None else training.plasticity_onset),
                               ("check", None if training is None else training.plasticity_check))
    }
    if training is not None:
        rec.update(onset_step=training.onset_step, total_steps=training.total_steps, onset_source="supplement",
                   recovery_steps=training.recovery_steps, recovery_censored=training.recovery_censored,
                   lambda_peak=training.lambda_peak, lambda_final=training.lambda_final,
                   overshoot=training.overshoot, settling_steps=training.settling_steps,
                   rate_limit_clip_share=training.rate_limit_clip_share,
                   dormant_trainable_check=getattr(training.plasticity_check, "dormant_trainable", None),
                   rank_trainable_check=getattr(training.plasticity_check, "rank_trainable", None),
                   intervention=None if training.intervention is None else training.intervention.model_dump())
    else:
        rec.update(onset_step=_onset_from_design(row), total_steps=max(steps) if steps else None,
                   onset_source="design", recovery_steps=None, recovery_censored=None,
                   lambda_peak=row.get("lambda_peak"), lambda_final=row.get("lambda_final"),
                   overshoot=(None if row.get("lambda_peak") is None or row.get("lambda_final") is None
                              else row["lambda_peak"] - row["lambda_final"]),
                   settling_steps=row.get("settling_steps"), rate_limit_clip_share=None,
                   dormant_trainable_check=None, rank_trainable_check=None, intervention=None)
    if rec["training_age"] is not None and rec["onset_step"] is not None:
        rec["constrained_steps_at_selection"] = int(rec["training_age"]) - int(rec["onset_step"])
    else:
        rec["constrained_steps_at_selection"] = None
    matched_step = row.get("matched_checkpoint_step")
    c_id = row.get("measurement_cost")
    final_step = _final_step(row, sup)
    measurement = sup.get("measurement", run_id)
    # a measurement record of another C_ID is not read: its episodes are the C_ID episodes of every gap IQM and its
    # return decides the A-return claims
    if measurement is not None and not _of_checkpoint(run_id, "measurement", measurement.step, matched_step,
                                                      "ledger's matched checkpoint", problems,
                                                      measurement.measurement_cost, c_id):
        measurement = None
    rec["measurement_return"] = None if measurement is None else measurement.measurement_return
    rec["measurement_episodes"] = None if measurement is None else list(measurement.episodes.episode_costs)
    rec["measurement_return_episodes"] = None if measurement is None else list(measurement.episodes.episode_returns)
    # Per-episode costs behind each gap (Part 5.3's interquartile mean; Q-iqm): the condition's episodes
    # at the matched checkpoint (battery, continuation), at tolerance 5.0 (sensitivity battery) and at
    # the final checkpoint (final battery, with its own measurement episodes as C_ID).
    rec["battery_episodes"] = {}
    rec["sens_episodes"] = {}
    rec["final_episodes"] = {}
    for condition in ("hazard", "dynamics"):
        battery = sup.get("battery", run_id, condition)
        if battery is None:
            continue
        ledger_gap = row.get(f"gap_{condition}")
        if ledger_gap is not None and battery.gap != ledger_gap:
            problems.append(f"{run_id}: gap_{condition} {ledger_gap!r} in the ledger, {battery.gap!r} in the supplement")
        if _of_checkpoint(run_id, f"battery ({condition})", battery.step, matched_step, "ledger's matched checkpoint",
                          problems, battery.measurement_cost, c_id):
            rec["battery_episodes"][condition] = list(battery.episodes.episode_costs)
    for condition in ("finetune", "transfer"):
        cont = sup.get("continuation", run_id, condition)
        if cont is None:
            continue
        ledger_gap = row.get(f"gap_{condition}")
        if ledger_gap is not None and cont.gap != ledger_gap:
            problems.append(f"{run_id}: gap_{condition} {ledger_gap!r} in the ledger, {cont.gap!r} in the supplement")
        if _of_checkpoint(run_id, f"continuation ({condition})", cont.parent_step, matched_step,
                          "ledger's matched checkpoint (the continuation's parent step)", problems,
                          cont.measurement_cost, c_id):
            rec["battery_episodes"][condition] = list(cont.episodes.episode_costs)
    final_measure = sup.get("final_battery", run_id, "measurement")
    if final_measure is not None and not _of_checkpoint(run_id, "final_battery (measurement)", final_measure.step,
                                                        final_step, "run's final checkpoint", problems):
        final_measure = None
    rec["final_measurement_cost"] = None if final_measure is None else final_measure.episodes.mean_cost
    if final_measure is not None:
        rec["final_episodes"]["measurement"] = list(final_measure.episodes.episode_costs)
    for condition in ("hazard", "dynamics"):
        final = sup.get("final_battery", run_id, condition)
        if final is not None and not _of_checkpoint(run_id, f"final_battery ({condition})", final.step, final_step,
                                                    "run's final checkpoint", problems):
            final = None
        if (final is not None and final_measure is not None
                and not _same_cost(final.measurement_cost, final_measure.episodes.mean_cost)):
            # a gap against another C_ID than the final checkpoint's own is not read (rule 7 (b), A-final-estimand)
            problems.append(f"{run_id}: the final battery's {condition} record uses C_ID {final.measurement_cost!r}, "
                            f"its measurement record {final_measure.episodes.mean_cost!r}; it is not read")
            final = None
        rec[f"final_gap_{condition}"] = None if final is None else final.gap
        if final is not None:
            rec["final_episodes"][condition] = list(final.episodes.episode_costs)
        sens = sup.get("sensitivity_battery", run_id, condition)
        # rule 7 (a) re-reads rule 1's selected checkpoint at tolerance 5.0: the same step and C_ID as the ledger's
        if sens is not None and not _of_checkpoint(run_id, f"sensitivity_battery ({condition})", sens.step, matched_step,
                                                   "ledger's matched checkpoint", problems, sens.measurement_cost, c_id):
            sens = None
        rec[f"sens_gap_{condition}"] = None if sens is None else sens.gap
        if sens is not None:
            rec["sens_episodes"][condition] = list(sens.episodes.episode_costs)
    evaluation = sup.get("evaluation", run_id)
    if evaluation is not None and row.get("final_cost") is not None and evaluation.final.mean_cost != row["final_cost"]:
        problems.append(f"{run_id}: final cost {row['final_cost']!r} in the ledger, {evaluation.final.mean_cost!r} "
                        "in the evaluation record")
    return rec


def _fewshot_entries(row: Mapping[str, Any], sup: Supplement, problems: list[str]) -> dict[float, dict[str, Any]]:
    """{budget: {horizons, sr {h: v}, vm {h: v} or None, adapt_steps, censored, index, consistent}} of one Study B run.

    The horizons of a continuation come from its supplement record (its spec's horizons); without a
    record, from the horizons its ``sr_fewshot`` keys hold. Ledger and record are cross-checked. A record whose
    parent step is not the run's final checkpoint (Q-continuations: the continuations "start from the parent's
    final checkpoint"), or whose arm is not the row's, is a data problem and is not read.
    """
    run_id = str(row["run_id"])
    final_step = _final_step(row, sup)
    sr_by_budget: dict[float, dict[int, float]] = {}
    for key, value in (row.get("sr_fewshot") or {}).items():
        budget, horizon = _parse_fewshot_key(key, run_id)
        sr_by_budget.setdefault(budget, {})[horizon] = float(value)
    adapt = {float(k): int(v) for k, v in (row.get("adapt_steps") or {}).items()}
    recorded = {model.budget for (owner, _), model in sup.of_kind("fewshot").items() if owner == run_id}
    out: dict[float, dict[str, Any]] = {}
    for budget in sorted(set(sr_by_budget) | set(adapt) | recorded):
        record = sup.get("fewshot", run_id, f"b{budget:g}")
        if record is not None:
            same_arm = record.arm == row["arm"]
            if not same_arm:
                problems.append(f"{run_id}: the fewshot (budget {budget:g}) record is of arm {record.arm}, the ledger "
                                f"row of {row['arm']}; it is not read")
            at_final = _of_checkpoint(run_id, f"fewshot (budget {budget:g})", record.parent_step, final_step,
                                      "run's final checkpoint (the continuation's parent step)", problems)
            if not (same_arm and at_final):
                record = None
        sr = sr_by_budget.get(budget, {})
        vm = None
        if record is not None:
            horizons = list(record.horizons)
            vm = {b.horizon: b.violation for b in record.by_horizon}
            record_sr = {b.horizon: b.satisfaction for b in record.by_horizon}
            if sr and sr != record_sr:
                problems.append(f"{run_id}: sr_fewshot at budget {budget:g} differs between the ledger and the supplement")
            if budget in adapt and adapt[budget] != record.adapt_steps:
                problems.append(f"{run_id}: adapt_steps at budget {budget:g} differs between the ledger and the supplement")
            sr = sr or record_sr
            steps = adapt.get(budget, record.adapt_steps)
        else:
            horizons = sorted(sr)
            steps = adapt.get(budget)
        entry: dict[str, Any] = {"horizons": horizons, "sr": dict(sorted(sr.items())), "vm": vm, "adapt_steps": steps,
                                 "censored": None, "index": None, "consistent": None}
        if horizons and steps is not None:
            entry["censored"] = steps > max(horizons)  # never by equality with a constant (Q-adapt-censoring)
            entry["index"] = len(horizons) + 1 if entry["censored"] else (
                horizons.index(steps) + 1 if steps in horizons else None)
            if entry["index"] is None:
                problems.append(f"{run_id}: adapt_steps {steps} at budget {budget:g} is neither a horizon "
                                f"{horizons} nor above the largest")
            if set(sr) == set(horizons):
                reached = [h for h in horizons if sr[h] >= R.SATISFACTION_TARGET]
                recomputed = reached[0] if reached else None
                entry["consistent"] = (recomputed == steps) if recomputed is not None else bool(entry["censored"])
                if not entry["consistent"]:
                    problems.append(f"{run_id}: adapt_steps {steps} at budget {budget:g} does not follow from "
                                    f"sr_fewshot {entry['sr']} (target {R.SATISFACTION_TARGET})")
        out[budget] = entry
    return out


def _study_b_record(row: Mapping[str, Any], sup: Supplement, problems: list[str]) -> dict[str, Any]:
    run_id = str(row["run_id"])
    rec: dict[str, Any] = {
        "run_id": run_id, "arm_id": _arm_id(row), "population": _population(run_id),
        "arm": row["arm"], "seed": int(row["seed"]), "notes": row.get("notes"), "training_levels": row.get("training_levels"),
        "sr_zero": None if row.get("sr_zero") is None else {float(k): float(v) for k, v in row["sr_zero"].items()},
        "vm_zero": None, "cost_zero": None, "sr_train": None, "vm_train": None,
        "zero_episodes": None, "train_episodes": None,
    }
    zero = sup.get("zero_shot", run_id)
    if zero is not None:
        same_arm = zero.arm == row["arm"]
        if not same_arm:
            problems.append(f"{run_id}: the zero-shot record is of arm {zero.arm}, the ledger row of {row['arm']}; "
                            "it is not read")
        # Q-studyb-eval: "Study B training runs are evaluated at their final checkpoint"
        at_final = _of_checkpoint(run_id, "zero-shot", zero.step, _final_step(row, sup), "run's final checkpoint",
                                  problems)
        if not (same_arm and at_final):
            zero = None
    if zero is not None:
        unseen = [b for b in zero.budgets if b.role == "unseen"]
        reference = [b for b in zero.budgets if b.role == "reference"]
        rec["vm_zero"] = {b.budget: b.violation for b in unseen}
        rec["cost_zero"] = {b.budget: b.mean_cost for b in unseen}
        rec["sr_train"] = {b.budget: b.satisfaction for b in reference}
        rec["vm_train"] = {b.budget: b.violation for b in reference}
        rec["zero_episodes"] = {b.budget: list(b.episode_costs) for b in unseen}  # per-episode costs (Q-iqm)
        rec["train_episodes"] = {b.budget: list(b.episode_costs) for b in reference}
        sr_record = {b.budget: b.satisfaction for b in unseen}
        if rec["sr_zero"] is None:
            rec["sr_zero"] = sr_record
        elif rec["sr_zero"] != sr_record:
            problems.append(f"{run_id}: sr_zero differs between the ledger and the zero-shot record")
    rec["fewshot"] = _fewshot_entries(row, sup, problems)
    return rec


def _check_arms(rows: list[dict[str, Any]]) -> None:
    """Refuse what rules 2 to 6 cannot read: a seed twice in one arm, two N = 0 arms in one task (DataError)."""
    seen: dict[tuple[str, int], str] = {}
    references: dict[str, set[str]] = {}
    for row in rows:
        if not row.get("completed") or row.get("study") not in ("A", "B"):
            continue
        key = (_arm_id(row), int(row["seed"]))
        if key in seen:
            raise DataError(f"{row['run_id']} and {seen[key]}: arm {key[0]} has seed {key[1]} twice; each seed is one "
                            "run (Part 5.1)")
        seen[key] = str(row["run_id"])
        if matching.is_reference_row(row):
            references.setdefault(str(row["task"]), set()).add(key[0])
    for task, arms in sorted(references.items()):
        if len(arms) > 1:
            raise DataError(f"task {task} has more than one N = 0 arm {sorted(arms)} (rule 2 needs one reference)")


ArmFactors = tuple[str, float, Optional[str], Optional[str], Optional[str], Optional[str]]


def arm_factors(item: Any) -> ArmFactors:
    """(task, N, onset_shape, step_matching, treatment, controller_variant) of a record (dict) or a RunSpec."""
    get = item.get if isinstance(item, Mapping) else (lambda k: getattr(item, k))
    return (str(get("task")), float(get("N")), get("onset_shape"), get("step_matching"), get("treatment"),
            get("controller_variant"))


@functools.lru_cache(maxsize=None)
def awaited_arms(cuts: tuple[str, ...]) -> Mapping[ArmFactors, str]:
    """{factors: arm_id} of every registered Study A arm (``pilot.manifest.design("study_a")``) that the
    recorded Part 6.1 cuts leave (``pilot.manifest.apply_cuts``): the arms the final analysis expects.

    An arm here without a completed run is data still to come (a missing field makes the analysis incomplete, never a
    silent drop; Part 5.6 excludes runs only for the causes it lists), so a Holm family keeps it as an incomplete member (Q-holm-families:
    the family shrinks to the cells tested, never on data still to come); an arm a cut removed is not tested.
    """
    from pilot.manifest import apply_cuts, design

    specs = apply_cuts(design("study_a", R.SEEDS[:1]), cuts)
    return MappingProxyType({arm_factors(s): s.arm_id for s in specs})


def awaited_a_line(arm_id: str, n: int, target: int) -> str:
    """Why a Study A arm below its seed target waits (``Dataset.awaited_a``). The surplus seeds of the ramp arms
    exist under Q-surplus-arm-set's answer (Table 9.1; ``pilot.manifest.surplus_spec``), so their line names that key
    too."""
    surplus_key = ", Q-surplus-arm-set" if "-ramp-" in arm_id and target > len(R.SEEDS) else ""
    return f"{arm_id}: {n} of its {target} seeds completed (Q-arm-complete{surplus_key})"


def _check_surplus_extra(surplus_extra: Any) -> None:
    if isinstance(surplus_extra, bool) or not isinstance(surplus_extra, int) \
            or not 0 <= surplus_extra <= R.MAX_SEEDS_PER_ARM - len(R.SEEDS):
        raise DataError(f"surplus_extra must be a whole number from 0 to {R.MAX_SEEDS_PER_ARM - len(R.SEEDS)}, "
                        f"got {surplus_extra!r}")


@functools.lru_cache(maxsize=None)
def study_a_seed_targets(cuts: tuple[str, ...], surplus_extra: int) -> Mapping[ArmFactors, tuple[str, int]]:
    """{factors: (arm_id, seed target)} of every registered Study A arm the recorded cuts leave (``awaited_arms``),
    under the answer of Q-arm-complete (Table 9.1): "an arm is complete when its completed (non-excluded,
    non-auxiliary) runs reach its seed target, 5 plus the Part 5.5 surplus count recorded in Part 8 for
    primary-comparison arms" (``pilot.manifest.is_primary_comparison``; as
    ``pilot.enrichment.seed_targets``), E = ``surplus_extra``. A run excluded under Part 5.6 is not a completed
    row, so its arm waits for the replacement seed (and is never decided on the seeds that happen to be present)."""
    from pilot.manifest import apply_cuts, design, is_primary_comparison

    _check_surplus_extra(surplus_extra)
    counts: dict[str, int] = {}
    for s in apply_cuts(design("study_a", R.SEEDS), cuts):
        counts[s.arm_id] = counts.get(s.arm_id, 0) + 1
    specs = apply_cuts(design("study_a", R.SEEDS[:1]), cuts)
    return MappingProxyType({arm_factors(s): (s.arm_id, counts[s.arm_id] + (surplus_extra if is_primary_comparison(s)
                                                                             else 0)) for s in specs})


@functools.lru_cache(maxsize=None)
def study_b_seed_targets(cuts: tuple[str, ...], surplus_extra: int) -> Mapping[str, int]:
    """{arm: seed target} of every registered Study B arm (``pilot.manifest.design("study_b")``) the recorded cuts
    leave, under the answer of Q-arm-complete (Table 9.1): "an arm is complete when its completed (non-excluded,
    non-auxiliary) runs reach its seed target, 5 plus the Part 5.5 surplus count recorded in Part 8 for
    primary-comparison arms" (``pilot.manifest.is_primary_comparison``: Part 5.5 gives
    surplus seeds to "the seven arms of Study B"), E = ``surplus_extra`` (as ``pilot.enrichment.seed_targets`` for
    Study A). A run excluded under Part 5.6 is not a completed row, so its arm waits for the replacement seed."""
    from pilot.manifest import apply_cuts, design, is_primary_comparison

    _check_surplus_extra(surplus_extra)
    specs = apply_cuts(design("study_b", R.SEEDS[:1]), cuts)
    return MappingProxyType({s.arm: len(R.SEEDS) + (surplus_extra if is_primary_comparison(s) else 0) for s in specs})


def build_dataset(rows: Iterable[Mapping[str, Any]], supplement: Supplement, *, mode: str,
                  revision: int = 0, cuts: Iterable[str] = (), surplus_extra: int = 0) -> Dataset:
    """The per-seed tables of the rows of one population (final: registered runs; pilot: one pilot).

    In final mode each Study B arm whose completed runs are fewer than its seed target (``study_b_seed_targets``,
    E = ``surplus_extra``: ``python -m analysis --mode final`` requires it) is data still to come
    (``Dataset.awaited_b``: never decided on the seeds of an arm whose other seeds still train), and so is each
    Study A arm with some completed runs but fewer than its target (``study_a_seed_targets``; ``Dataset.awaited_a``).
    An arm of either study with more completed runs than its target is refused (DataError): E is then not the data
    root's surplus.
    """
    if mode not in MODES:
        raise DataError(f"mode must be one of {MODES}, got {mode!r}")
    if mode == "final":
        # checked here, not only in the cached helpers: lru_cache takes True for 1 and would skip their check
        _check_surplus_extra(surplus_extra)
    rows = [dict(r) for r in rows]
    populations = {str(r["run_id"]): _population(str(r["run_id"])) for r in rows}
    if mode == "final":
        pilots = sorted(run_id for run_id, pop in populations.items() if pop)
        if pilots:
            raise DataError(f"the final analysis refuses pilot rows (Part 3.6: 'The pilot's runs are not reused'): "
                            f"{pilots[:5]}{' ...' if len(pilots) > 5 else ''}")
        population = ""
    else:
        population = pilot_population(revision)
    selected = [r for r in rows if populations[str(r["run_id"])] == population]
    ignored = sorted(str(r["run_id"]) for r in rows if populations[str(r["run_id"])] != population)
    # an auxiliary run (pilot.manifest.AUXILIARY_NOTE: the N = 0 run of a warm-started replacement's seed, which
    # its arm never received; answered in Table 9.1, Q-warm-start) is not a seed of its arm (the matching leaves it
    # out too)
    from pilot.manifest import is_auxiliary_row

    auxiliary = sorted(str(r["run_id"]) for r in selected if is_auxiliary_row(r))
    selected = [r for r in selected if not is_auxiliary_row(r)]
    _check_arms(selected)
    problems: list[str] = []
    study_a, study_b, exclusions = [], [], []
    for row in sorted(selected, key=lambda r: str(r["run_id"])):
        if not row.get("completed"):
            exclusions.append({"run_id": row["run_id"], "study": row.get("study"), "arm": row.get("arm"),
                               "task": row.get("task"), "seed": row.get("seed"), "failure_cause": row.get("failure_cause")})
            continue
        if row.get("study") == "A":
            if row.get("N") is None:
                raise DataError(f"{row['run_id']}: a completed Study A row without N (its arm cannot be told)")
            study_a.append(_study_a_record(row, supplement, problems))
        elif row.get("study") == "B" and row.get("arm") in R.STUDY_B_ARMS:
            study_b.append(_study_b_record(row, supplement, problems))
    if mode == "pilot" and not study_a:
        # Part 5.8: the test run "is run once on the pilot data to test it"; with no pilot run to read it tests nothing
        raise DataError(f"pilot mode (revision {revision}) found no completed Study A run of the pilot population "
                        f"{population!r} ({len(ignored)} rows of other populations ignored): the Part 5.8 test run would "
                        "test nothing; check --revision and --ledger")
    known = {r["run_id"] for r in selected} | set(auxiliary)
    unreadable = sorted({run_id for kind in supplement.records.values() for run_id, _ in kind
                         if _population_or_none(run_id) is None})
    if unreadable:
        problems.append(f"supplement records whose run_id is not of study A or B: {unreadable[:10]}"
                        f"{' ...' if len(unreadable) > 10 else ''}")
    orphans = sorted({run_id for kind in supplement.records.values() for run_id, _ in kind if run_id not in known
                      and _population_or_none(run_id) == population})
    if orphans:
        problems.append(f"supplement records without a ledger row: {orphans[:10]}{' ...' if len(orphans) > 10 else ''}")
    cuts = tuple(cuts)
    awaited: list[str] = []
    awaited_b: dict[str, tuple[int, int]] = {}
    awaited_a: dict[str, tuple[int, int]] = {}
    if mode == "final":
        present = {arm_factors(r) for r in study_a}
        awaited = sorted(arm_id for factors, arm_id in awaited_arms(cuts).items() if factors not in present)
        over: list[str] = []
        for factors, (arm_id, target) in study_a_seed_targets(cuts, surplus_extra).items():
            recs = [r for r in study_a if arm_factors(r) == factors]
            if recs and len(recs) < target:
                awaited_a[recs[0]["arm_id"]] = (len(recs), target)
            elif len(recs) > target:
                over.append(f"{recs[0]['arm_id']} ({len(recs)} completed, seeds {sorted(int(r['seed']) for r in recs)}; "
                            f"target {target})")
        for arm, target in study_b_seed_targets(cuts, surplus_extra).items():
            recs = [r for r in study_b if r["arm"] == arm]
            if len(recs) < target:
                awaited_b[arm] = (len(recs), target)
            elif len(recs) > target:
                over.append(f"B-{arm} ({len(recs)} completed, seeds {sorted(int(r['seed']) for r in recs)}; "
                            f"target {target})")
        if over:
            raise DataError(f"arms hold more completed runs than their seed target (5, or 5 + E for "
                            f"primary-comparison arms; E = --surplus-extra {surplus_extra}; Q-arm-complete): "
                            f"{'; '.join(over)}. A Part 5.6 replacement takes the place of a run that is not completed, "
                            "so --surplus-extra must be the data root's stored "
                            "surplus_extra (the value `schedule add-surplus` fixed; Part 5.5: 'the same count for "
                            "every arm in the set')")
    return Dataset(mode=mode, population=population, rows=selected, study_a=study_a, study_b=study_b,
                   exclusions=exclusions, supplement=supplement, problems=problems, ignored=ignored, cuts=cuts,
                   auxiliary=auxiliary, awaited=awaited, awaited_b=awaited_b, awaited_a=awaited_a,
                   surplus_extra=surplus_extra if mode == "final" else None)


def load_dataset(ledger_path: Path, *, mode: str, supplement_dir: Optional[Path] = None, revision: int = 0,
                 cuts_file: Optional[Path] = None, cuts: Optional[Iterable[str]] = None,
                 surplus_extra: int = 0) -> Dataset:
    """Read the ledger, its supplement (default ``<ledger dir>/supplement``) and the recorded cuts into per-seed tables.

    ``cuts`` (already parsed, e.g. by ``parse_cuts`` from checked bytes) replaces reading ``cuts_file``.
    The ledger is hashed and read first, the supplement scanned next, and the ledger hashed again:
    the ledger writer (pilot/ledger_writer.py) writes a run's records before its row, so every row read has its records on disk by the
    scan, and a row appended meanwhile is refused (DataError). ``Dataset.ledger_sha256`` is the hash of
    the rows analysed.
    """
    before = file_sha256(ledger_path)
    rows = load_rows(ledger_path)
    supplement = load_supplement(default_supplement_dir(ledger_path) if supplement_dir is None else supplement_dir)
    if file_sha256(ledger_path) != before:
        raise DataError(f"the ledger {ledger_path} changed while it and its supplement were read (another process "
                        "wrote it); run again")
    dataset = build_dataset(rows, supplement, mode=mode, revision=revision,
                            cuts=load_cuts(cuts_file) if cuts is None else tuple(cuts), surplus_extra=surplus_extra)
    dataset.ledger_sha256 = before
    return dataset


def five_seed_view(dataset: Dataset) -> Dataset:
    """The dataset as Q-surplus-in-analysis's other reading reads it: "five seeds" per arm; also, under every reading
    (answered in Table 9.1), the seeds of Part 5.4's seed-index-resampled correlations: the H1 trend ("twenty points:
    four onset fractions by five seeds"; ``analysis.study_a._h1_control``) and H3 (a) ("with the same interval";
    ``analysis.study_a._h3a_points``).

    Each arm keeps the registered seeds and the replacements of failed ones (Part 5.6), and leaves its
    surplus seeds (Part 5.5) out. A replacement and a surplus seed both take the arm's next unused seed
    >= 5, so the seed does not tell them apart: the ledger notes do (``pilot.manifest.seed_role``, written
    from the scheduler's record). A row of seed >= 5 whose notes say neither (written before the notes
    were recorded) ranks after the registered seeds and the noted replacements, lowest seed first, and at
    most ``len(R.SEEDS)`` records are kept per arm. The same object when nothing is left out, so that
    reading costs nothing then.
    """
    from pilot.manifest import seed_role

    limit = len(R.SEEDS)

    def rank(rec: dict[str, Any]) -> tuple[int, int]:
        known = int(rec["seed"]) in R.SEEDS or seed_role(rec) == "replacement"
        return (0 if known else 1, int(rec["seed"]))

    def keep(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        by_arm: dict[str, list[dict[str, Any]]] = {}
        for rec in records:
            if seed_role(rec) != "surplus":
                by_arm.setdefault(rec["arm_id"], []).append(rec)
        kept = {id(r) for recs in by_arm.values() for r in sorted(recs, key=rank)[:limit]}
        return [r for r in records if id(r) in kept]

    study_a, study_b = keep(dataset.study_a), keep(dataset.study_b)
    if len(study_a) == len(dataset.study_a) and len(study_b) == len(dataset.study_b):
        return dataset
    return replace(dataset, study_a=study_a, study_b=study_b)

"""Enrichment of ledger rows after a run is recorded (ledger memo, decision 3; results/supplement_schema.py).

Owner: pilot owner (Role 1), the only writer of the ledgers and their supplements. The rules come
from the owners of each rule (Roles 2 to 5); every result is checked at the boundary before it is
written, and re-checked here by an independent implementation where the rule is Role 1's to verify.

Commands (``python -m pilot enrich <action>``), each logged to ``<ledger>.enrichment_log.jsonl``:

* ``select`` (``apply_selection``): Part 4.1 rule 1, per run. The Analysis and results owner's
  ``analysis.matching.select_checkpoint`` (contract 4) chooses, and ``rule_one_step`` re-checks
  (closest selection-set cost to d = 25 among the last ten checkpoints; distances compared after
  rounding to 1e-4; a non-finite cost is refused). A tie goes to the later checkpoint, the answer of
  Q-tie-break in Table 9.1: ``require_answered("Q-tie-break")`` is called only when more than one
  window checkpoint attains the minimum distance, so untied runs never waited for the answer.
  Written: matched_checkpoint_path, matched_checkpoint_step, training_age, selection_cost_at_match,
  lambda_at_selection (the multiplier recorded with that checkpoint; ledger memo decision 6).
* ``measure`` (``apply_measurement``): Table 2.1, "a measurement set, run on the chosen checkpoint
  only and used for the reported in-distribution cost and as the baseline of every robustness gap":
  ``envs.evaluation.evaluate_battery(..., conditions=[])`` (contract 5) at the matched checkpoint;
  measurement_cost and the ``measurement`` supplement record (its episodes and return).
* ``battery`` (``apply_battery``): Table 2.2 hazard relocation and dynamics perturbation at the
  matched checkpoint; equation (1) gap = C_cond - C_ID. Part 4.1 rule 6, "An arm that cannot be
  matched is reported with its cost and left out of the battery": a row of the main study gets the
  battery only when ``matched is True``; the pilot's rows all get it, as Part 3.6 runs the pilot's
  battery and G1 decides its matching. One ``battery`` record per condition.
* ``match`` (``apply_matching``): Part 4.1 rules 2 to 6 by ``analysis.matching.match_arms``, re-checked
  by ``rule_matching`` (an independent implementation under the answer of Q-matched-cost-set, the
  measurement set, and the answer of Q-threshold-arithmetic). Gated by Q-matched-cost-set and
  Q-arm-complete (both answered in Table 9.1). An arm is complete when its completed runs reach its
  seed target, 5, or 5 + E for the primary-comparison arms of Part 5.5
  (``manifest.is_primary_comparison``), E the surplus seeds per arm (``--surplus-extra``, checked
  against the data root's stored value, 0 when it stores none); flags are written only for complete
  arms of a task whose reference (N = 0) arm is complete, and an incomplete arm waits.
  Q-threshold-arithmetic (the SD's n or n - 1, <= or <, rounding to 1e-4; answered in Table 9.1) is
  a report key, not a gate: the flags follow its answer, and while it is open the arms whose flags
  another of its readings would change are named in the log and by the command line
  (``Done.provisional``; ``_flags``); the analysis reports what depends on them as PROVISIONAL or
  UNDECIDED.
* ``controller`` (``apply_controller``): Table 2.4 multiplier overshoot and settling time: lambda_peak,
  lambda_final and settling_steps by ``metrics.controller.controller_quantities`` (gated inside by
  Q-controller-quantities) over the run's progress.csv at the run's own steps per epoch, re-checked
  by ``controller_reference``; for a rate-limited run, the share of constrained epochs in which the
  clip bound (Q-rate-limit, Table 9.1: ``metrics.controller.rate_limit_clip_share``, re-checked by
  ``rate_limit_clip_reference``), which has no ledger field and is logged.
* ``training`` (``apply_training``): the ``training`` supplement record: recovery time (Table 2.1;
  ``metrics.recovery``), the controller quantities and the overshoot (Table 2.4), the rate clip's share
  of a rate-limited run (``rate_limit_clip_share``; Q-rate-limit), the plasticity.csv
  rows at onset and at onset + 200,000 steps with the critic norms (Table 2.3; box H3 (c)), and the
  intervention summary of a reset or injection run (Table 2.4;
  ``metrics.interventions.intervention_summary`` over the plug-in's intervention.json).
* ``continuations`` (``apply_continuations``): Table 2.2 reward-only fine-tuning and transfer. Each
  battery continuation that ended ``continued`` (``manifest.battery_continuation``; pilot/manifest.py) is
  evaluated on the measurement set at its final checkpoint by
  ``envs.evaluation.evaluate_continuation``; gap_finetune / gap_transfer = C_after - the parent's
  measurement_cost; the ``continuation`` record. The continuation must have started from the
  checkpoint the row matched (its train_result.json digest of the parent checkpoint).
* ``zeroshot`` (``apply_zero_shot``): Table 2.5 zero-shot evaluation at the unseen budgets
  (``studyb.evaluation.evaluate_zero_shot``, gated inside by Q-studyb-eval); sr_zero, equation (10),
  and the ``zero_shot`` record (the reference budgets included, for box G3).
* ``fewshot`` (``apply_fewshot``): Table 2.5 few-shot adaptation and adaptation steps
  (``studyb.evaluation.evaluate_fewshot``) of every continuation that ended ``continued``;
  sr_fewshot keyed by ``pilot.contracts.fewshot_key``; adapt_steps per budget from that budget's own
  continuation and its own horizons (Part 6.1 cut 3 may leave one); "recorded as above the largest
  horizon if never reached" is the largest horizon + 1, the answer of Q-adapt-censoring in Table 9.1,
  which is therefore required only for a budget that never reaches the target, and only for its
  adapt_steps and record (the key "Gates adaptation steps only"): while the key is open that budget's
  sr_fewshot rates are written.
* ``final-battery`` (``apply_final_battery``): Part 4.1 rule 7 (b) and Part 4.1.1, "every arm's final
  checkpoint in place of the selected one": the measurement set and hazard (and, optionally,
  dynamics) at the final checkpoint of every completed Study A row of the main study; the
  ``final_battery`` records; no ledger field (schema v1 has none).
* ``sensitivity-battery`` (``apply_sensitivity_battery``): Part 4.1 rule 7 (a), "once with tolerance
  5.0": the battery at the matched checkpoint of the arms that match at 5.0 but not at 2.5
  (``match_arms`` at ``R.SENSITIVITY_TOLERANCE``); the ``sensitivity_battery`` records, never the
  ledger's gap fields of an unmatched row (rule 6).

The battery commands are gated only by the conditions' own keys (Q-hazard, Q-dynamics), which
``envs.evaluation.evaluate_battery`` checks before it runs anything. Each condition's recorded
definition (HANDOVER.md section 8, contract 5: "what was relocated or scaled, and the seeds";
``result["conditions"]``, from ``envs.evaluation.hazard_definition`` and ``dynamics_definition``)
has no field in the supplement schema, so the three battery commands log it (``definitions``: each
distinct definition once, with the runs it applies to).

Writing. Everything is written under the ledger's lock and atomically
(``ledger_writer.ledger_transaction``), so an enrichment can run while the scheduler appends rows:
the fields are merged into a working copy, every row of the copy is validated as a ``LedgerRow``
(the frozen ``write_enrichment`` checked only the Parquet types; the corrected one validates the
row too), the seed sets are checked (the measurement set, and the hazard condition's episode and
layout seeds: Q-eval-seeds, the same for every run), the supplement records are written
(pilot/supplement.py; before the ledger changes, results/supplement_schema.py), and only then is
the ledger replaced. A scalar field is written only when it is None; a map field (sr_zero,
sr_fewshot, adapt_steps) only gains absent keys.
An existing value is never changed: a re-evaluation must reproduce it exactly (evaluation is
deterministic) or the enrichment is refused. The run directory must be the run the row records
(the same training commit, and the matched or final checkpoint at the path the row stores), so a
smoke data root with the same run_ids is refused.

An open question that only some runs depend on (Q-tie-break, Q-adapt-censoring) holds those runs:
the others are written, and the command then refuses with ``PendingQuestionError`` naming the runs
that wait. A question every run depends on refuses at once, before anything is written. A report key
(configs/registered.py; Q-threshold-arithmetic here) holds nothing: while it is open, what depends
on it is computed under its answer (docs/DECISIONS.md; Table 9.1) and named as provisional
(``Done.provisional``). What waits for its data instead (a run not selected yet, an incomplete arm,
a continuation not ``continued`` yet) is returned beside the run_ids written (``Done.waiting``) and
printed by the command line, so that "nothing written" is never read as "done".
"""


from __future__ import annotations

import copy
import hashlib
import json
import math
import numbers
import os
import statistics
import sys
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from configs import registered as R
from pilot import manifest, provenance, supplement
from pilot.algorithms import BATCH_COST_KEY
from pilot.contracts import (
    ContractError,
    check_canonical_seeds,
    fewshot_key,
    validate_plasticity,
    validate_seed_set,
    validate_unstable_replacements,
)
from pilot.errors import PendingQuestionError, require_answered
from pilot.ledger_writer import SMOKE_MARKERS, _is_smoke, ledger_transaction, seeds_registry
from pilot.manifest import RunSpec
from pilot.rundir import (
    CHECKPOINT_SUBDIR,
    EPOCH_COLUMN,
    EVALUATION_RESULT,
    MULTIPLIER_COLUMN,
    SPEC_FILE,
    TRAIN_RESULT,
    checkpoint_file,
    read_progress,
    run_steps_per_epoch,
)

CONDITION_FIELDS = {"hazard": "gap_hazard", "dynamics": "gap_dynamics"}  # Table 2.2 rows 1-2 (evaluations)
# envs.onset.PROPOSED_KEY (tested equal): the rate-limited multiplier's value before the clip, logged each epoch
PROPOSED_MULTIPLIER_COLUMN = "Onset/MultiplierProposed"
CONTINUATION_FIELDS = {"finetune": "gap_finetune", "transfer": "gap_transfer"}  # Table 2.2 rows 3-4 (continuation runs)
MAP_FIELDS = ("sr_zero", "sr_fewshot", "adapt_steps")  # Appendix B: keys are only ever added
TIE_DECIMALS = 4  # Q-tie-break (Table 9.1): rule 1 distances are compared at this precision: costs are
# means of 100 episodes of 0/1 per-step indicators (multiples of 0.01), so 1e-4 absorbs float32 and float64 noise alike
# answered in Table 9.1 (Q-threshold-arithmetic): rules 3 and 5 compare after rounding to 1e-4, with the sample SD
# and inclusive bounds (the answer ``analysis.matching.PROPOSED_ARITHMETIC`` implements; re-checked independently here).
MATCH_DECIMALS = 4
# rule 5's SD is a square root, not on the 0.01 grid: compared after rounding to 1e-9 (noise only), so an SD just
# above 10 is never rounded down to 10 (analysis.matching.SD_DECIMALS; a test checks they agree)
MATCH_SD_DECIMALS = 9
MATCH_SD_DDOF = 1  # answered in Table 9.1 (Q-threshold-arithmetic): rule 5's seed-to-seed SD is the sample SD (n - 1)
MATCH_INCLUSIVE = True  # answered in Table 9.1 (Q-threshold-arithmetic): rules 3 and 5 inclusive (<=), as printed
INTERVENTION_FILE = "intervention.json"  # envs.onset.INTERVENTION_FILE (HANDOVER.md section 8; a test checks they agree)
PLASTICITY_ROW_FIELDS = (  # results.supplement_schema.PlasticityRow (a test checks they agree)
    "dormant", "rank", "norm", "norm_reward_critic", "norm_cost_critic", "dormant_trainable", "rank_trainable",
    "dormant_reward_critic", "rank_reward_critic", "dormant_cost_critic", "rank_cost_critic", "norm_all",
)

SELECT_TARGET = ("analysis.matching", "select_checkpoint", "Analysis and results (Role 4)")
MATCH_TARGET = ("analysis.matching", "match_arms", "Analysis and results (Role 4)")
BATTERY_TARGET = ("envs.evaluation", "evaluate_battery", "Environment and tests (Role 2)")
CONTINUATION_TARGET = ("envs.evaluation", "evaluate_continuation", "Environment and tests (Role 2)")
CONTROLLER_TARGET = ("metrics.controller", "controller_quantities", "Metrics and interventions (Role 3)")
CLIP_SHARE_TARGET = ("metrics.controller", "rate_limit_clip_share", "Metrics and interventions (Role 3)")
RECOVERY_TARGET = ("metrics.recovery", "recovery_steps", "Metrics and interventions (Role 3)")
INTERVENTION_TARGET = ("metrics.interventions", "intervention_summary", "Metrics and interventions (Role 3)")
ZERO_SHOT_TARGET = ("studyb.evaluation", "evaluate_zero_shot", "Study B and literature (Role 5)")
FEWSHOT_TARGET = ("studyb.evaluation", "evaluate_fewshot", "Study B and literature (Role 5)")


def _load(target: tuple[str, str, str]) -> Callable[..., Any]:
    """Import ``module.function`` of another role; PluginUnavailableError when it is not written yet.

    One rule with the launcher (``pilot.algorithms.import_provided``; HANDOVER.md section 8; pilot/launch.py): the
    module, or a repository module (or a name of one) it imports, not written yet is unavailable; a
    missing library shows its real error."""
    from pilot.algorithms import PluginUnavailableError, import_provided

    module_name, func, owner = target
    module = import_provided(module_name, f"{module_name}.{func} (provided by {owner})")
    try:
        return getattr(module, func)
    except AttributeError as exc:
        raise PluginUnavailableError(f"{module_name} has no function {func!r} yet ({owner})") from exc


def _check_condition(condition: str) -> None:
    if condition not in CONDITION_FIELDS:
        raise ValueError(
            f"unknown condition {condition!r}; choose from {sorted(CONDITION_FIELDS)} "
            "(fine-tuning and transfer are continuations, scheduled as runs)"
        )


def _conditions(conditions: Iterable[str], allowed: Mapping[str, str], what: str) -> list[str]:
    names = list(conditions)
    unknown = [c for c in names if c not in allowed]
    if unknown:
        raise ValueError(f"unknown {what} {unknown}; choose from {sorted(allowed)}")
    if len(set(names)) != len(names):
        raise ValueError(f"{what} repeat a name: {names}")
    return names


def _code_commit() -> str:
    """HEAD: the commit of the code that computes the records of this call (``code_commit``)."""
    return provenance.commit_hash()


def _abs(path: str | Path | None) -> str | None:
    """The absolute spelling of ``path`` for the enrichment log (None stays None)."""
    return None if path is None else os.path.abspath(path)


def _plain(value: Any) -> Any:
    from pilot.launch import plain

    return plain(value)


def _finite(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(float(value)):
        raise ContractError(f"{what} is not a finite number: {value!r}")
    return float(value)


def _check_episodes(raw: Mapping[str, Any], message: str) -> None:
    """A harness result's ``episodes`` is the integer R.EVAL_EPISODES (not a flag, a float or a string)."""
    value = raw.get("episodes")
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value != R.EVAL_EPISODES:
        raise ContractError(f"{message} (the result has episodes={value!r})")


def _real_or_nan(value: Any) -> float:
    """``value`` as a float when it is a real number (not a flag), else NaN (which equals nothing)."""
    return float(value) if isinstance(value, numbers.Real) and not isinstance(value, bool) else math.nan


def _is_pilot(run_id: str) -> bool:
    """A run of the pilot or of the re-pilot (run_ids ``P-`` or ``P1-``; Part 3.6, Part 6)."""
    return run_id.startswith((manifest.pilot_prefix(0), manifest.pilot_prefix(1)))


def _refuse_pilot_rows(rows: Sequence[Any], ledger_path: Path, action: str) -> None:
    pilot = [r.run_id for r in rows if _is_pilot(r.run_id)]
    if pilot:
        raise ContractError(
            f"{action} is for the main study's ledger; {ledger_path} holds pilot runs ({pilot[0]}, ...), and the "
            "pilot's runs are not reused (Part 3.6)"
        )


def _rows(ledger_path: Path) -> list[Any]:
    from results.ledger_schema import load_ledger_as_rows

    return load_ledger_as_rows(ledger_path)


# ---------------------------------------------------------------------------
# Rule 1 (Part 4.1), re-checked
# ---------------------------------------------------------------------------


def _window_steps(steps: Iterable[int]) -> list[int]:
    """Rule 1's window: the last ten checkpoint steps; for a run whose total (its last step) is off the
    200,000-step grid, the ten steps total - k x 200,000 (Q-selection-window, Table 9.1; the steps
    ``pilot.contracts.selection_window`` evaluated). ContractError if one of those is missing."""
    ordered = sorted(steps)
    if not ordered or ordered[-1] % R.CHECKPOINT_INTERVAL_STEPS == 0:
        return ordered[-R.SELECTION_WINDOW_CHECKPOINTS:]
    total = ordered[-1]
    window = [total - k * R.CHECKPOINT_INTERVAL_STEPS for k in range(R.SELECTION_WINDOW_CHECKPOINTS - 1, -1, -1)]
    missing = sorted(set(window) - set(ordered))
    if missing:
        raise ContractError(f"the end-relative selection window {window} (Q-selection-window) lacks steps {missing}")
    return window


def _window(checkpoints: list[dict]) -> list[dict]:
    steps = set(_window_steps(c["step"] for c in checkpoints))
    window = sorted((c for c in checkpoints if c["step"] in steps), key=lambda c: c["step"])
    if any(c.get("selection_cost") is None for c in window):
        raise ContractError("a checkpoint of the selection window has no selection-set cost")
    if not all(math.isfinite(float(c["selection_cost"])) for c in window):
        raise ContractError("a selection-set cost of the window is not finite; rule 1 cannot rank it")
    return window


def _distance(checkpoint: Mapping[str, Any]) -> float:
    return round(abs(float(checkpoint["selection_cost"]) - R.COST_LIMIT), TIE_DECIMALS)


def rule_one_step(checkpoints: list[dict]) -> int:
    """Part 4.1 rule 1, computed independently: among the last ten checkpoints, the one whose
    selection-set cost is closest to d = 25; a tie goes to the later checkpoint (Q-tie-break)."""
    return min(_window(checkpoints), key=lambda c: (_distance(c), -c["step"]))["step"]


def rule_one_ties(checkpoints: list[dict]) -> list[int]:
    """The steps of the window checkpoints at the minimum rounded distance (more than one: a tie)."""
    window = _window(checkpoints)
    best = min(_distance(c) for c in window)
    return sorted(c["step"] for c in window if _distance(c) == best)


def selection_fields(row_data: dict[str, Any], chosen_step: int) -> dict[str, Any]:
    """Ledger fields describing the chosen checkpoint (Appendix B), after checking rule 1."""
    window = _window_steps(c["step"] for c in row_data["checkpoints"])
    if chosen_step not in window:
        raise ContractError(f"{row_data['run_id']}: chosen step {chosen_step} is not among the last ten checkpoints {window}")
    expected = rule_one_step(row_data["checkpoints"])
    if chosen_step != expected:
        raise ContractError(
            f"{row_data['run_id']}: the selector chose {chosen_step}, but rule 1 gives {expected} "
            f"(closest selection-set cost to d = {R.COST_LIMIT:g}; ties to the later checkpoint)"
        )
    record = next(c for c in row_data["checkpoints"] if c["step"] == chosen_step)
    return {
        "matched_checkpoint_path": record["path"],
        "matched_checkpoint_step": chosen_step,
        "training_age": chosen_step,
        "selection_cost_at_match": record["selection_cost"],
        "lambda_at_selection": record.get("multiplier"),
    }


# ---------------------------------------------------------------------------
# Logging, waiting runs, and the one way of writing
# ---------------------------------------------------------------------------


def enrichment_log_path(ledger_path: Path) -> Path:
    """``<ledger>.enrichment_log.jsonl`` beside the ledger."""
    return Path(ledger_path).with_name(Path(ledger_path).stem + ".enrichment_log.jsonl")


WAITING_KEYS = ("waiting", "arms_waiting", "not_ready")  # a call that only left runs waiting is logged too


def _head_or_unavailable() -> str:
    """HEAD, or "unavailable" if git cannot tell (the log's ``head_at_end``)."""
    try:
        return provenance.commit_hash()
    except Exception:  # noqa: BLE001 - the log records what it can; the command's own checks decide
        return "unavailable"


def _log(ledger_path: Path, action: str, run_ids: list[str], extra: dict[str, Any] | None = None, *,
         commit: str) -> None:
    """Append the provenance of an enrichment (commit, time, runs, runs waiting) beside the ledger.

    ``commit`` is the one the command read when it started (``_code_commit``), the commit its records
    name: the log is the only provenance of the ledger's scalar fields, so it never reads HEAD again at
    the end, when a commit landing during a long command would name code that did not compute them.
    ``head_at_end`` is added when HEAD moved meanwhile (or is 'unavailable' when git cannot tell).
    Nothing is appended for a call that enriched nothing and left nothing waiting.
    """
    extra = dict(extra or {})
    if not run_ids and not any(extra.get(k) for k in WAITING_KEYS):
        return
    log = enrichment_log_path(ledger_path)
    end = _head_or_unavailable()
    try:
        dirty: Any = bool(provenance.dirty_paths())
    except Exception:  # noqa: BLE001 - called from ``finally``: never hide the enrichment's own error
        dirty = "unavailable"
    entry = {"utc": provenance.utc_now().isoformat(), "action": action, "commit": commit,
             "worktree_dirty": dirty, "run_ids": run_ids, **extra}
    if end != commit:
        entry["head_at_end"] = end
    try:
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, sort_keys=True, default=str) + "\n")
    except OSError as exc:  # called from ``finally``: never replace the enrichment's own error
        print(f"warning: enrichment log {log} not written ({exc}); record it by hand: {json.dumps(entry, default=str)}",
              file=sys.stderr)


class Done(list):
    """What an enrichment command wrote: the run_ids, in order (it is that list), and ``waiting``,
    what it left unwritten because its data is not there yet ({arm, run or continuation: why}; e.g. an
    incomplete arm, which ``enrich match`` leaves unflagged, or a continuation not yet ``continued``). The
    command line prints both, so that "nothing written" never reads as "done"; the enrichment log
    records both too. What waits on an open question is not here: the command refuses for it
    (``_Waiting``) after writing the rest. ``provisional`` names what was computed under the answer
    (Table 9.1) of a report key while the key is open and its other readings would give it differently
    ({arm: what each reading gives}; Q-threshold-arithmetic for the matching flags): it gates nothing,
    and the command line and the log say so."""

    def __init__(self, run_ids: Iterable[str] = (), waiting: Mapping[str, str] | None = None) -> None:
        super().__init__(run_ids)
        self.waiting: dict[str, str] = dict(waiting or {})
        self.provisional: dict[str, str] = {}


@dataclass
class _Waiting:
    """Runs held by an open question that only they depend on (the other runs go on)."""

    items: dict[str, str] = field(default_factory=dict)  # what waits -> the refusal of ``require_answered``

    def refuse(self, action: str, done: list[str], not_ready: Mapping[str, str] | None = None) -> None:
        """PendingQuestionError naming what the open questions hold, once the others were written;
        ``not_ready`` (what waits for its data, the ``Done.waiting`` the refusal replaces) is counted in it."""
        if self.items:
            names = ", ".join(sorted(self.items))
            first = next(iter(self.items.values()))
            more = (f" {len(not_ready)} more wait for their data ({', '.join(sorted(not_ready)[:5])}"
                    f"{', ...' if len(not_ready) > 5 else ''}; see the enrichment log)." if not_ready else "")
            raise PendingQuestionError(
                f"{action}: {len(self.items)} item(s) held by open questions ({names}); {len(done)} run(s) written."
                f"{more} {first}"
            )


def _same(a: Any, b: Any) -> bool:
    """Equal values; a flag equals only a flag (True is not 1), and NaN equals NaN."""
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def merge_fields(current: Mapping[str, Any], fields: Mapping[str, Any], run_id: str) -> dict[str, Any]:
    """The enrichment update of one row under this module's writing rules; ContractError if it would change a value.

    A scalar field is set only when it is None (the same value again changes nothing); a map field
    (``MAP_FIELDS``) gains only its absent keys, and an existing key must keep its value. A value
    None in ``fields`` writes nothing. Returns only what changes.
    """
    update: dict[str, Any] = {}
    for name, value in fields.items():
        have = current.get(name)
        if name in MAP_FIELDS:
            if value is None:
                continue
            merged = dict(have or {})
            for key, item in dict(value).items():
                if key in merged:
                    if not _same(merged[key], item):
                        raise ContractError(
                            f"{run_id}: {name}[{key!r}] is {merged[key]!r} in the ledger, the new result gives {item!r}; "
                            "an existing value is never changed (a re-evaluation must reproduce it exactly)"
                        )
                else:
                    merged[key] = item
            if merged != dict(have or {}):
                update[name] = merged
        elif value is None or (have is not None and _same(have, value)):
            continue
        elif have is not None:
            raise ContractError(
                f"{run_id}: {name} is {have!r} in the ledger, the new result gives {value!r}; an existing value is "
                "never changed (evaluation is deterministic: find out why it differs)"
            )
        else:
            update[name] = value
    return update


@dataclass(frozen=True)
class _Record:
    kind: str
    run_id: str
    record: Mapping[str, Any]
    part: str | None = None


def _validated(record: _Record) -> Any:
    from results.supplement_schema import validate_record

    try:
        return validate_record(record.kind, record.record)
    except ValueError as exc:
        raise ContractError(f"{record.run_id}: the result is not a valid {record.kind} supplement record: {exc}") from exc


def _check_seeds_fit(kind: str, seeds: list[int], registry: Path) -> None:
    """``check_canonical_seeds`` without writing (so that a refused result fixes no seed set)."""
    from pilot.ledger_writer import _check_seeds_fit as check

    check(kind, seeds, registry)


def _ledger_home(record: _Record) -> tuple[str, Any] | None:
    """The ledger field (and, for a map field, its key) that holds the result a record carries, or None
    for the kinds whose numbers have no ledger field (training, final_battery, sensitivity_battery)."""
    if record.kind == "measurement":
        return "measurement_cost", None
    if record.kind == "battery":
        return CONDITION_FIELDS[str(record.part)], None
    if record.kind == "continuation":
        return CONTINUATION_FIELDS[str(record.part)], None
    if record.kind == "zero_shot":
        return "sr_zero", None
    if record.kind == "fewshot":
        return "adapt_steps", float(record.record["budget"])
    return None


def _recovering(record: _Record, row: Mapping[str, Any]) -> bool:
    """True when the record may be on disk from an interrupted earlier call: the ledger field it belongs
    to is still empty. Only then may an existing record that differs in ``code_commit`` alone be kept
    (``supplement.write(accept_reproduction=True)``); otherwise a record must be byte-identical (results/supplement_schema.py)."""
    home = _ledger_home(record)
    if home is None:
        return False
    name, key = home
    value = row.get(name)
    if key is None:
        return value is None
    return value is None or all(float(k) != key for k in value)


def _commit(
    ledger_path: Path,
    run_id: str,
    fields: Mapping[str, Any] | None = None,
    *,
    records: Sequence[_Record] = (),
    seeds: Sequence[tuple[str, list[int]]] = (),
    check: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Write one run's enrichment: ledger fields, supplement records and seed sets, all or nothing.

    Under the ledger's lock: the fields are merged (``merge_fields``) into the working copy, every
    row of the copy is validated as a ``LedgerRow``, the seed sets are checked against the
    registry, ``check`` runs, every supplement record is checked (``supplement.check``: a record
    already on disk must be this one) before the first is written, so that one conflicting record
    leaves no other record behind; then the records are written, the seed sets are fixed, and the
    ledger is replaced. Every record is validated before the lock is taken. A record on disk that
    differs only in ``code_commit`` is accepted only while the ledger field it belongs to is still
    empty (``_recovering``: a crash between the record and the ledger, retried at a later commit).
    Only a crash in the middle of the writes can leave some records without their fields, and the
    retry then recovers them. Returns the fields that changed.
    """
    from results.ledger_schema import load_ledger_as_rows, write_enrichment

    for record in records:
        _validated(record)
    registry = seeds_registry(ledger_path)
    with ledger_transaction(ledger_path) as tmp:
        current = next((r for r in load_ledger_as_rows(tmp) if r.run_id == run_id), None)
        if current is None:
            raise KeyError(f"run_id {run_id!r} not found in the ledger {ledger_path}")
        before = current.model_dump(mode="python")
        update = merge_fields(before, dict(fields or {}), run_id)
        if update:
            write_enrichment(run_id, update, tmp)
            load_ledger_as_rows(tmp)  # raises (and the ledger stays as it was) if any row breaks the schema
        for kind, values in seeds:
            _check_seeds_fit(kind, values, registry)
        if check is not None:
            check()
        recovering = [_recovering(record, before) for record in records]
        for record, accept in zip(records, recovering):  # every record first: none is written if one conflicts
            supplement.check(record.kind, record.run_id, record.record, ledger_path=ledger_path, part=record.part,
                             accept_reproduction=accept)
        for record, accept in zip(records, recovering):
            supplement.write(record.kind, record.run_id, record.record, ledger_path=ledger_path, part=record.part,
                             accept_reproduction=accept)
        for kind, values in seeds:
            check_canonical_seeds(kind, values, registry)
    return update


def _write(ledger_path: Path, run_id: str, fields: dict[str, Any]) -> None:
    """Write ``fields`` to one row (``_commit`` without records); used by the tests to set fields directly."""
    _commit(ledger_path, run_id, fields)


# ---------------------------------------------------------------------------
# The run directory of a row
# ---------------------------------------------------------------------------


def check_same_commit(run_id: str, run_dir: Path, train: dict[str, Any], ledger_commit: str) -> None:
    """The run directory holds the run the ledger row records: the same training commit.

    Smoke and registered data roots use the same run_ids, so the run_id alone cannot tell a run
    from its namesake under another data root.
    """
    if train.get("commit_hash") != ledger_commit:
        raise ContractError(
            f"{run_id}: {run_dir} was trained at commit {train.get('commit_hash')!r}, but the ledger row records "
            f"{ledger_commit!r}; is --data-root the data root this ledger was written from?"
        )


def matched_checkpoint_path(omnisafe_dir: Path, step: int) -> str:
    """The ledger's spelling of the checkpoint saved at ``step`` (as ``ledger_writer`` writes it)."""
    from results.ledger_schema import relative_checkpoint_path

    return relative_checkpoint_path(os.path.abspath(Path(omnisafe_dir) / CHECKPOINT_SUBDIR / f"epoch-{step // R.STEPS_PER_EPOCH}.pt"))


def _read_json(path: Path, what: str) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ContractError(f"{what}: {path} does not exist; is --data-root the data root this ledger was written from?") from None
    except (OSError, ValueError) as exc:
        raise ContractError(f"{what}: cannot read {path}: {exc}") from None
    if not isinstance(data, dict):
        raise ContractError(f"{what}: {path} does not hold a JSON object")
    return data


@dataclass(frozen=True)
class RunFiles:
    """The run directory of a ledger row, checked to be that row's run."""

    run_dir: Path
    spec: RunSpec
    train: dict[str, Any]
    omnisafe_dir: Path

    def evaluation(self) -> dict[str, Any]:
        return _read_json(self.run_dir / EVALUATION_RESULT, self.spec.run_id)

    def steps_per_epoch(self) -> int:
        try:
            return run_steps_per_epoch(self.omnisafe_dir, default=R.STEPS_PER_EPOCH)
        except (OSError, ValueError) as exc:
            raise ContractError(f"{self.spec.run_id}: {exc}") from None


def open_run(row: Any, run_dir_of: Callable[[str], Path]) -> RunFiles:
    """The run directory of ``row``: its spec (same run_id), its training result (same commit)."""
    run_dir = Path(run_dir_of(row.run_id))
    spec_data = _read_json(run_dir / SPEC_FILE, row.run_id)
    try:
        spec = RunSpec.from_dict(spec_data)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{run_dir / SPEC_FILE} is not a run spec: {exc}") from None
    if spec.run_id != row.run_id:
        raise ContractError(f"{run_dir} holds the spec of {spec.run_id}, not {row.run_id}")
    train = _read_json(run_dir / TRAIN_RESULT, row.run_id)
    check_same_commit(row.run_id, run_dir, train, row.commit_hash)
    if not train.get("omnisafe_dir"):
        raise ContractError(f"{row.run_id}: {run_dir / TRAIN_RESULT} names no OmniSafe directory")
    return RunFiles(run_dir, spec, train, run_dir / train["omnisafe_dir"])


def _check_checkpoint(row: Any, files: RunFiles, step: int, stored: str | None, label: str) -> None:
    """The checkpoint at ``step`` of the run directory is the one the row stores (``label``)."""
    path = matched_checkpoint_path(files.omnisafe_dir, step)
    if path != stored:
        raise ContractError(
            f"{row.run_id}: {files.run_dir} gives the checkpoint path {path!r}, but the ledger's {label} is "
            f"{stored!r}; use the data root the ledger was written from"
        )


def _final_checkpoint(row: Any, files: RunFiles) -> int:
    """The run's final step, checked against the ledger's last checkpoint record."""
    if not row.checkpoints:
        raise ContractError(f"{row.run_id}: the ledger row records no checkpoint")
    last = max(row.checkpoints, key=lambda c: c.step)
    if last.step != files.spec.total_steps:
        raise ContractError(f"{row.run_id}: the ledger's last checkpoint is step {last.step}, the run's total is "
                            f"{files.spec.total_steps}")
    _check_checkpoint(row, files, last.step, last.path, "final checkpoint")
    return last.step


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _selection_seeds(files: RunFiles) -> set[int]:
    evaluation = files.evaluation()
    return set(validate_seed_set("selection_seeds", evaluation.get("selection_seeds")))


# ---------------------------------------------------------------------------
# Harness results as supplement records (contract 5; HANDOVER.md section 8; results/supplement_schema.py)
# ---------------------------------------------------------------------------


def _episode_arrays(result: Mapping[str, Any], name: str, what: str) -> tuple[list[Any], list[Any]]:
    costs = (result.get("episode_costs") or {}).get(name)
    returns = (result.get("episode_returns") or {}).get(name)
    if not isinstance(costs, (list, tuple)) or not isinstance(returns, (list, tuple)):
        raise ContractError(f"{what}: the harness result holds no per-episode costs and returns of {name!r} "
                            "(HANDOVER.md section 8; results/supplement_schema.py: every result carries them)")
    return list(costs), list(returns)


def _replacements(value: Any, seeds: Any, what: str) -> list[dict[str, Any]] | None:
    """One evaluation's ``unstable_replacements`` (Q-mujoco-exception, Table 9.1), checked against its planned
    seeds, which the result and the record keep: the reserve seeds lie in the set's reserve range, are
    distinct and at most ``MAX_UNSTABLE_EPISODES`` (``pilot.contracts.validate_unstable_replacements``).
    None, the record's form, when no episode was replaced (or the result predates the field)."""
    return validate_unstable_replacements(what, value, list(seeds or ())) or None


def _replacement_map(raw: Mapping[str, Any], what: str) -> Mapping[Any, Any]:
    """A harness result's ``unstable_replacements`` map ({evaluation: [...]}), empty when absent."""
    value = raw.get("unstable_replacements") or {}
    if not isinstance(value, Mapping):
        raise ContractError(f"{what}: unstable_replacements must map each evaluation to its replacements")
    return value


def _block(seed_set: str, seeds: Any, costs: list[Any], returns: list[Any], mean_cost: Any, mean_return: Any,
           budgets: list[float] | None = None, replaced: Any = None, what: str = "") -> dict[str, Any]:
    """An ``EpisodeBlock`` of results.supplement_schema (validated with the record), with the evaluation's
    replacements of unstable episodes (``replaced``, checked against ``seeds``; ``what`` names it)."""
    return {"seed_set": seed_set, "seeds": list(seeds or ()), "episode_costs": costs, "episode_returns": returns,
            "mean_cost": mean_cost, "mean_return": mean_return, "budgets": budgets,
            "unstable_replacements": _replacements(replaced, seeds, what)}


@dataclass(frozen=True)
class BatteryResult:
    """Contract 5's result at one checkpoint, checked: C_ID, its episodes, and each condition's."""

    step: int
    c_id: float
    measurement_seeds: list[int]
    measurement: dict[str, Any]  # an EpisodeBlock
    conditions: dict[str, dict[str, Any]]  # {condition: {"cost", "episodes", "hazard", "definition"}}
    raw: dict[str, Any]

    def fields(self) -> dict[str, Any]:
        """Equation (1): measurement_cost and gap = C_cond - C_ID for every condition evaluated."""
        return gaps_from_costs(self.raw, list(self.conditions))

    def definitions(self) -> dict[str, dict[str, Any]]:
        """{condition: the harness's recorded definition} (HANDOVER.md section 8, contract 5), for the log."""
        return {c: entry["definition"] for c, entry in self.conditions.items()}

    def seed_sets(self) -> list[tuple[str, list[int]]]:
        """The seed sets this result fixes in the seed registry (``_commit``): the measurement set and, when the
        hazard condition was evaluated, its episode and layout seeds, which Q-eval-seeds (Table 9.1) fixes "in
        code and shared by every run, arm, task and study" (``check_canonical_seeds``: the first run fixes them)."""
        sets = [("measurement", list(self.measurement_seeds))]
        hazard = self.conditions.get("hazard")
        if hazard is not None:
            # the layout seeds as recorded (the record's schema, checked first in ``_commit``, refuses a bad list)
            sets += [("hazard", list(hazard["episodes"]["seeds"])),
                     ("hazard_layouts", list(hazard["hazard"]["layout_seeds"] or ()))]
        return sets

    def condition_record(self, kind: str, run_id: str, commit: str, condition: str, c_id: float) -> _Record:
        entry = self.conditions[condition]
        return _Record(kind, run_id, {
            "kind": kind, "run_id": run_id, "code_commit": commit, "step": self.step, "condition": condition,
            "episodes": entry["episodes"], "hazard": entry["hazard"], "measurement_cost": c_id,
            "gap": entry["cost"] - c_id,
        }, condition)


def check_battery_result(run_id: str, result: Mapping[str, Any], *, step: int, conditions: Sequence[str],
                         selection_seeds: Iterable[int]) -> BatteryResult:
    """Contract 5 checked at the boundary: 100 episodes, the measurement seeds (disjoint from the
    selection seeds, Table 2.1), finite costs, the step asked for, per-episode arrays for C_ID and for
    every condition asked for (hazard on its 20 x 5 layout seeds, dynamics paired with C_ID), each
    condition's recorded definition (``result["conditions"]``; HANDOVER.md section 8, contract 5: "what
    was relocated or scaled, and the seeds", which differs by robot), and each evaluation's
    replacements of unstable episodes (``unstable_replacements``; Q-mujoco-exception), checked against
    its planned seeds, which stay the canonical ones. The writers check the seed sets against the
    registry (``BatteryResult.seed_sets``)."""
    what = f"{run_id} step {step}"
    raw = _plain(dict(result))
    _check_episodes(raw, f"{what}: battery evaluation must use {R.EVAL_EPISODES} episodes per condition")
    if "step" in raw and raw["step"] != step:
        raise ContractError(f"{what}: the harness evaluated step {raw['step']!r}")
    measurement_seeds = validate_seed_set("measurement_seeds", raw.get("measurement_seeds"))
    if set(measurement_seeds) & set(selection_seeds):
        raise ContractError(f"{run_id}: measurement and selection seeds overlap (Table 2.1 requires disjoint sets)")
    c_id = _finite(raw.get("measurement"), f"{what}: the measurement cost")
    seeds = raw.get("seeds") or {}
    replaced = _replacement_map(raw, what)
    costs, returns = _episode_arrays(raw, "measurement", what)
    if list(seeds.get("measurement", measurement_seeds)) != measurement_seeds:
        raise ContractError(f"{what}: the measurement episodes were not run on the measurement seeds")
    measurement = _block("measurement", measurement_seeds, costs, returns, c_id, raw.get("measurement_return"),
                         replaced=replaced.get("measurement"), what=f"{what} measurement")
    evaluated: dict[str, dict[str, Any]] = {}
    for condition in conditions:
        _check_condition(condition)
        value = _finite(raw.get(condition), f"{what}: the cost under {condition}")
        costs, returns = _episode_arrays(raw, condition, what)
        definition = (raw.get("conditions") or {}).get(condition)
        if not isinstance(definition, Mapping) or not definition:
            raise ContractError(f"{what}: the harness result holds no recorded definition of {condition!r} "
                                "(result['conditions']; HANDOVER.md section 8, contract 5)")
        hazard = None
        if condition == "hazard":
            hazard = {k: definition.get(k) for k in ("form", "layout_seeds", "episodes_per_layout")}
            block = _block("hazard", validate_seed_set("hazard seeds", seeds.get("hazard")), costs, returns, value,
                           raw.get("hazard_return"), replaced=replaced.get("hazard"), what=f"{what} hazard")
        else:
            if list(seeds.get(condition, measurement_seeds)) != measurement_seeds:
                raise ContractError(f"{what}: the {condition} episodes were not run on the measurement seeds")
            block = _block("measurement", measurement_seeds, costs, returns, value, raw.get(f"{condition}_return"),
                           replaced=replaced.get(condition), what=f"{what} {condition}")
        evaluated[condition] = {"cost": value, "episodes": block, "hazard": hazard, "definition": dict(definition)}
    return BatteryResult(step, c_id, measurement_seeds, measurement, evaluated, raw)


def _definitions_entry(by_run: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> list[dict[str, Any]]:
    """The enrichment log's ``definitions``: each distinct recorded definition of a condition once, with
    the runs evaluated under it (``by_run``: {run_id: {condition: definition}}, ``BatteryResult.definitions``).

    The supplement schema has no field for a definition (results/supplement_schema.py ``HazardSeeds``
    keeps the form and the seeds; nothing is kept of dynamics), so the log is where what was relocated
    or scaled, per robot, is found (Q-hazard, Q-dynamics); it is also recomputable from code_commit.
    """
    groups: dict[str, dict[str, Any]] = {}
    for run_id, conditions in sorted(by_run.items()):
        for condition, definition in sorted(conditions.items()):
            key = json.dumps([condition, definition], sort_keys=True, default=str)
            groups.setdefault(key, {"condition": condition, "definition": definition, "run_ids": []})["run_ids"].append(run_id)
    return list(groups.values())


def gaps_from_costs(costs: dict[str, Any], conditions: Iterable[str]) -> dict[str, float]:
    """Equation (1): gap = C_cond - C_ID for every requested condition."""
    _check_episodes(costs, f"battery evaluation must use {R.EVAL_EPISODES} episodes per condition")
    c_id = float(costs["measurement"])
    if not math.isfinite(c_id):
        raise ContractError("measurement cost is not finite")
    out = {"measurement_cost": c_id}
    for condition in conditions:
        _check_condition(condition)
        value = float(costs[condition])
        if not math.isfinite(value):
            raise ContractError(f"cost under {condition} is not finite")
        out[CONDITION_FIELDS[condition]] = value - c_id
    return out


def _measurement_record(run_id: str, commit: str, result: BatteryResult) -> _Record:
    return _Record("measurement", run_id, {"kind": "measurement", "run_id": run_id, "code_commit": commit,
                                           "step": result.step, "episodes": result.measurement})


def _same_episodes(existing: Any, block: Mapping[str, Any]) -> bool:
    """A stored EpisodeBlock and a new one hold the same episodes (a deterministic re-evaluation)."""
    return (list(existing.seeds) == list(block["seeds"])
            and [float(c) for c in existing.episode_costs] == [float(c) for c in block["episode_costs"]]
            and [float(r) for r in existing.episode_returns] == [float(r) for r in block["episode_returns"]])


# ---------------------------------------------------------------------------
# select, measure, battery
# ---------------------------------------------------------------------------


def apply_selection(
    ledger_path: Path,
    *,
    selector: Callable[[list[dict]], int] | None = None,
    check_code: Callable[[], None] | None = None,
) -> Done:
    """Write rule-1 selections for completed Study A runs that have none yet. Returns run_ids done.

    ``check_code`` runs after each call of the selector and before its result is written (the
    command line refuses code that the commit does not contain; ``provenance``). A run whose window
    has a tie waits while Q-tie-break is open (``require_answered``); the others are written, and
    the call then raises ``PendingQuestionError`` naming the runs that wait.
    """
    ledger_path = Path(ledger_path)
    select = selector or _load(SELECT_TARGET)
    done = Done()
    commit = _code_commit()
    waiting = _Waiting()
    try:
        for row in _rows(ledger_path):
            if row.study != "A" or not row.completed or row.matched_checkpoint_step is not None:
                continue
            data = row.model_dump(mode="python")
            step = int(select(copy.deepcopy(data["checkpoints"])))  # the selector cannot alter what is checked
            if check_code is not None:
                check_code()
            fields = selection_fields(data, step)
            ties = rule_one_ties(data["checkpoints"])
            if len(ties) > 1:  # only a tie depends on how ties are broken
                try:
                    require_answered("Q-tie-break", what=f"{row.run_id}: rule 1 with checkpoints {ties} equally close "
                                                         f"to d = {R.COST_LIMIT:g} (Part 4.1; the later one by Q-tie-break)")
                except PendingQuestionError as exc:
                    waiting.items[row.run_id] = str(exc)
                    continue
            _commit(ledger_path, row.run_id, fields)
            done.append(row.run_id)
    finally:
        _log(ledger_path, "select", done, {"waiting": waiting.items}, commit=commit)  # also the runs written before a failure
    waiting.refuse("select", done)
    return done


def _open_matched(row: Any, run_dir_of: Callable[[str], Path]) -> RunFiles:
    files = open_run(row, run_dir_of)
    _check_checkpoint(row, files, int(row.matched_checkpoint_step), row.matched_checkpoint_path, "matched checkpoint")
    return files


def apply_measurement(
    ledger_path: Path,
    run_dir_of: Callable[[str], Path],
    *,
    evaluator: Callable[..., dict[str, Any]] | None = None,
    check_code: Callable[[], None] | None = None,
    data_root: str | Path | None = None,
) -> Done:
    """``enrich measure``: C_ID of every selected Study A checkpoint (Table 2.1), with its episodes.

    ``evaluate_battery(omnisafe_dir, spec, matched step, conditions=[])`` (contract 5) for each
    completed Study A row with a matched checkpoint whose measurement_cost or ``measurement``
    record is missing; writes measurement_cost (an existing value must be reproduced exactly) and
    the ``measurement`` record (the episodes and the measurement-set return, Part 5.2 "return").
    The measurement seeds must be the ledger's canonical set (Table 2.1: the same for every run).
    """
    ledger_path = Path(ledger_path)
    evaluate = evaluator or _load(BATTERY_TARGET)
    commit = _code_commit()
    done = Done()
    try:
        for row in _rows(ledger_path):
            if row.study != "A" or not row.completed:
                continue
            if row.matched_checkpoint_step is None:
                done.waiting[row.run_id] = "no matched checkpoint yet (`enrich select`)"
                continue
            if row.measurement_cost is not None and supplement.exists("measurement", row.run_id, ledger_path=ledger_path):
                continue
            files = _open_matched(row, run_dir_of)
            step = int(row.matched_checkpoint_step)
            costs = evaluate(str(files.omnisafe_dir), files.spec.to_dict(), step, [])
            if check_code is not None:
                check_code()
            result = check_battery_result(row.run_id, costs, step=step, conditions=[], selection_seeds=_selection_seeds(files))
            _commit(ledger_path, row.run_id, {"measurement_cost": result.c_id},
                    records=[_measurement_record(row.run_id, commit, result)],
                    seeds=[("measurement", result.measurement_seeds)])
            done.append(row.run_id)
    finally:
        _log(ledger_path, "measure", done, {"not_ready": done.waiting,
                                            "data_root": _abs(data_root)}, commit=commit)
    return done


def apply_battery(
    ledger_path: Path,
    run_dir_of: Callable[[str], Path],
    conditions: Iterable[str],
    *,
    evaluator: Callable[..., dict[str, Any]] | None = None,
    check_code: Callable[[], None] | None = None,
    data_root: str | Path | None = None,
) -> Done:
    """``enrich battery``: evaluate each selected checkpoint that lacks a requested gap or record, and write them.

    Part 4.1 rule 6: a row of the main study enters the battery only when ``matched is True`` (an
    auxiliary row, which ``enrich match`` never flags, is left out and does not wait); the pilot's rows
    (``P-``/``P1-``) all do (Part 3.6; G1 decides the pilot's matching). The spec of each
    run is read from its own run directory (``spec.json``), so replacement runs are handled like any
    other. The run directory must hold the run the row records: the same training commit, and the
    matched checkpoint at the path the row stores (otherwise, for example, a smoke data root with
    the same run_ids would be evaluated). Evaluation is deterministic, so a measurement cost or a
    gap already in the ledger must be reproduced exactly. Writes measurement_cost and the gaps, one
    ``battery`` record per condition and, when missing, the ``measurement`` record. ``check_code``
    runs after each evaluation and before its result is written. ``data_root`` and each condition's
    recorded definition (``definitions``) are recorded in the enrichment log. The result gates of the
    conditions (Q-hazard, Q-dynamics) are the harness's.
    """
    ledger_path = Path(ledger_path)
    conditions = _conditions(conditions, CONDITION_FIELDS, "conditions")
    evaluate = evaluator or _load(BATTERY_TARGET)
    commit = _code_commit()
    done = Done()
    evaluated: dict[str, list[str]] = {}  # the conditions actually evaluated and written, per run
    definitions: dict[str, dict[str, Any]] = {}  # the harness's recorded definitions, per run and condition
    unmatched: list[str] = []
    try:
        for row in _rows(ledger_path):
            if row.study != "A" or not row.completed:
                continue
            if row.matched_checkpoint_step is None:
                done.waiting[row.run_id] = "no matched checkpoint yet (`enrich select`)"
                continue
            if not _is_pilot(row.run_id) and row.matched is not True:
                if manifest.is_auxiliary_row(row.model_dump()):  # manifest.AUXILIARY_NOTE; Q-warm-start
                    continue  # never matched (``arm_readiness``): no battery, and nothing to wait for
                unmatched.append(row.run_id)  # rule 6 (None: waits for `enrich match`)
                if row.matched is None:
                    done.waiting[row.run_id] = "no matched flag yet (`enrich match`; Part 4.1 rule 6)"
                continue
            missing = [c for c in conditions if getattr(row, CONDITION_FIELDS[c]) is None
                       or not supplement.exists("battery", row.run_id, ledger_path=ledger_path, part=c)]
            if not missing:
                continue
            files = _open_matched(row, run_dir_of)
            step = int(row.matched_checkpoint_step)
            costs = evaluate(str(files.omnisafe_dir), files.spec.to_dict(), step, missing)
            if check_code is not None:
                check_code()
            result = check_battery_result(row.run_id, costs, step=step, conditions=missing,
                                          selection_seeds=_selection_seeds(files))
            records = [result.condition_record("battery", row.run_id, commit, c, result.c_id) for c in missing]
            if not supplement.exists("measurement", row.run_id, ledger_path=ledger_path):
                records.append(_measurement_record(row.run_id, commit, result))
            _commit(ledger_path, row.run_id, result.fields(), records=records, seeds=result.seed_sets())
            done.append(row.run_id)
            evaluated[row.run_id] = missing
            definitions[row.run_id] = result.definitions()
    finally:
        _log(ledger_path, "battery", done,
             {"conditions": conditions, "evaluated": evaluated, "not_matched": unmatched, "not_ready": done.waiting,
              "definitions": _definitions_entry(definitions),
              "data_root": _abs(data_root)}, commit=commit)
    return done


# ---------------------------------------------------------------------------
# match (Part 4.1 rules 2 to 6), re-checked
# ---------------------------------------------------------------------------


def _arm_id(row: Any) -> str:
    run_id, seed = str(row["run_id"]), int(row["seed"])
    if not run_id.endswith(f"-s{seed}"):
        raise ContractError(f"run_id {run_id!r} does not end with its seed '-s{seed}'")
    return run_id[: -len(f"-s{seed}")]


def _is_reference(row: Mapping[str, Any]) -> bool:
    """Rule 2: the N = 0 arm ("no onset shape and no step-matching variant", Part 3.2), untreated."""
    return row.get("N") == R.ONSET_FRACTIONS[0] and row.get("treatment") is None and row.get("controller_variant") is None


@dataclass(frozen=True)
class ThresholdReading:
    """One reading of the arithmetic of Part 4.1 rules 3 and 5 (Q-threshold-arithmetic, answered in Table 9.1;
    its question: "n or n - 1 in the standard deviation, strict or inclusive comparisons")."""

    ddof: int  # 1: the sample SD (n - 1); 0: the population SD (n)
    inclusive: bool  # rule 3 and rule 5's tolerance compare with <= (True) or < (False); under every reading the
    # task is infeasible when the reference SD > 10
    decimals: int | None  # compare after rounding to this many decimals, or unrounded (None)

    def rounded(self, value: float) -> float:
        return value if self.decimals is None else round(value, self.decimals)

    def within(self, value: float, limit: float) -> bool:
        value = self.rounded(value)
        return value <= limit if self.inclusive else value < limit

    def sd(self, values: Sequence[float]) -> float:
        return statistics.stdev(values) if self.ddof == 1 else statistics.pstdev(values)

    def rounded_sd(self, value: float) -> float:
        """Rule 5's SD, rounded at noise level only (``MATCH_SD_DECIMALS``) when the reading rounds."""
        return value if self.decimals is None else round(value, MATCH_SD_DECIMALS)


# The answer of Q-threshold-arithmetic (Table 9.1; docs/DECISIONS.md), named when it was still the proposal
# (tests/test_budget_and_go.py imports the name).
PROPOSED_READING = ThresholdReading(MATCH_SD_DDOF, MATCH_INCLUSIVE, MATCH_DECIMALS)
# Every reading the question names (n - 1 or n in the SD, <= or <), each with and without the answer's
# rounding; the answer (Table 9.1) first.
THRESHOLD_READINGS = (PROPOSED_READING, *(
    ThresholdReading(ddof, inclusive, decimals)
    for ddof in (1, 0) for inclusive in (True, False) for decimals in (MATCH_DECIMALS, None)
    if ThresholdReading(ddof, inclusive, decimals) != PROPOSED_READING
))


def rule_matching(rows: Iterable[Mapping[str, Any]], *, tolerance: float,
                  reading: ThresholdReading = PROPOSED_READING) -> dict[str, dict[str, bool]]:
    """Part 4.1 rules 2 to 5, computed independently of ``analysis.matching``: {run_id: {matched, infeasible}}.

    Per task, over the completed Study A rows given (every row counts: Q-arm-complete and
    Q-surplus-in-analysis; the caller passes complete arms only): rule 2, the reference cost is the
    mean of the N = 0 arm's measurement costs (Q-matched-cost-set); rule 3, the task is feasible when
    reference cost <= d + tolerance; rule 5, and when the reference arm's seed-to-seed SD does not
    exceed 10; an arm is matched when |its mean cost - the reference cost| <= tolerance. ``reading``
    is the arithmetic of Q-threshold-arithmetic (default: its answer in Table 9.1, the sample SD, inclusive
    bounds, compared after rounding to ``MATCH_DECIMALS``; ``THRESHOLD_READINGS`` lists the others).
    The reference is matched exactly when its task is feasible; ``infeasible`` holds for every row of
    an infeasible task. ContractError for a task without exactly one reference arm of at least two
    seeds, or a missing or non-finite cost.
    """
    if tolerance not in (R.MATCH_TOLERANCE, R.SENSITIVITY_TOLERANCE):
        raise ValueError(f"tolerance {tolerance!r} is neither rule 5's {R.MATCH_TOLERANCE} nor rule 7's {R.SENSITIVITY_TOLERANCE}")
    by_task: dict[str, dict[str, list[Mapping[str, Any]]]] = {}
    for row in rows:
        if row.get("study") != "A" or row.get("completed") is not True:
            continue
        by_task.setdefault(str(row["task"]), {}).setdefault(_arm_id(row), []).append(row)
    flags: dict[str, dict[str, bool]] = {}
    for task, arms in sorted(by_task.items()):
        refs = [key for key, arm_rows in arms.items() if _is_reference(arm_rows[0])]
        if len(refs) != 1:
            raise ContractError(f"task {task}: {len(refs)} reference (N = 0) arms among the rows; rule 2 needs one")

        def cost(row: Mapping[str, Any]) -> float:
            return _finite(row.get("measurement_cost"), f"{row['run_id']}: measurement_cost")

        ref_costs = [cost(r) for r in arms[refs[0]]]
        if len(ref_costs) < 2:
            raise ContractError(f"task {task}: rule 5's seed-to-seed SD needs at least two reference seeds")
        ref_cost = statistics.fmean(ref_costs)
        feasible = (reading.within(ref_cost, R.COST_LIMIT + tolerance)
                    and not reading.rounded_sd(reading.sd(ref_costs)) > R.INFEASIBLE_REFERENCE_SD)
        for key, arm_rows in arms.items():
            if key == refs[0]:
                matched = feasible
            else:
                difference = abs(statistics.fmean(cost(r) for r in arm_rows) - ref_cost)
                matched = feasible and reading.within(difference, tolerance)
            for row in arm_rows:
                flags[str(row["run_id"])] = {"matched": bool(matched), "infeasible": not feasible}
    return flags


def seed_targets(arm_ids: Iterable[str], surplus_extra: int) -> dict[str, int]:
    """Each arm's seed target under the answer of Q-arm-complete (Table 9.1): 5, or 5 + E for the arms of Part
    5.5's primary-comparison set (``manifest.is_primary_comparison``), E the surplus seeds per arm."""
    if isinstance(surplus_extra, bool) or not isinstance(surplus_extra, int) \
            or not 0 <= surplus_extra <= R.MAX_SEEDS_PER_ARM - len(R.SEEDS):
        raise ValueError(f"surplus_extra must be a whole number from 0 to {R.MAX_SEEDS_PER_ARM - len(R.SEEDS)}, "
                         f"got {surplus_extra!r}")
    design = {s.arm_id: s for s in manifest.design("study_a", seeds=(R.SEEDS[0],))}
    targets = {}
    for arm in sorted(set(arm_ids)):
        if arm not in design:
            raise ContractError(f"arm {arm} is not an arm of the registered Study A design")
        targets[arm] = len(R.SEEDS) + (surplus_extra if manifest.is_primary_comparison(design[arm]) else 0)
    return targets


def _check_surplus(surplus_extra: int, stored: Any) -> None:
    """``surplus_extra`` must be the data root's stored surplus seeds per arm; none stored is 0, as the scheduler
    reads it (``Scheduler.seed_target``)."""
    have = int(stored or 0)
    if have != surplus_extra:
        raise ContractError(
            f"--surplus-extra {surplus_extra} differs from the data root's {have} surplus seeds per arm (Part 5.5; "
            f"`schedule add-surplus`{'' if stored is not None else ' stored none'}); the seed targets of "
            "Q-arm-complete follow the data root"
        )


@dataclass
class ArmReadiness:
    """Which arms rules 2 to 5 can decide now, per task, and why the others wait."""

    ready: dict[str, list[Mapping[str, Any]]] = field(default_factory=dict)  # task -> rows (reference first)
    targets: dict[str, int] = field(default_factory=dict)  # arm -> seed target (the ready arms)
    waiting: dict[str, str] = field(default_factory=dict)  # arm -> reason


def arm_readiness(rows: Iterable[Mapping[str, Any]], surplus_extra: int) -> ArmReadiness:
    """The complete arms (Q-arm-complete): completed runs reaching the target, each with a measurement cost.

    A task is decided only when its reference arm is complete; an incomplete arm waits.
    """
    # an auxiliary run (manifest.AUXILIARY_NOTE; Q-warm-start) is not a seed of its arm
    completed = [r for r in rows if r.get("study") == "A" and r.get("completed") is True
                 and not manifest.is_auxiliary_row(r)]
    arms: dict[str, list[Mapping[str, Any]]] = {}
    for row in completed:
        arms.setdefault(_arm_id(row), []).append(row)
    targets = seed_targets(arms, surplus_extra)
    out = ArmReadiness()

    def problem(arm: str) -> str | None:
        arm_rows = arms[arm]
        if len(arm_rows) < targets[arm]:
            return f"{len(arm_rows)} of its {targets[arm]} seeds completed (Q-arm-complete)"
        missing = [r["run_id"] for r in arm_rows if r.get("measurement_cost") is None]
        return f"no measurement_cost yet for {missing} (`enrich measure`)" if missing else None

    by_task: dict[str, list[str]] = {}
    for arm, arm_rows in arms.items():
        by_task.setdefault(str(arm_rows[0]["task"]), []).append(arm)
    for task, task_arms in sorted(by_task.items()):
        refs = [a for a in task_arms if _is_reference(arms[a][0])]
        if len(refs) != 1:
            for arm in task_arms:
                out.waiting[arm] = f"task {task} has {len(refs)} completed reference (N = 0) arms; rule 2 needs one"
            continue
        why = problem(refs[0])
        if why is not None:
            for arm in task_arms:
                out.waiting[arm] = f"the reference arm {refs[0]}: {why}" if arm != refs[0] else why
            continue
        ready = [refs[0]]
        for arm in sorted(task_arms):
            if arm == refs[0]:
                continue
            why = problem(arm)
            if why is None:
                ready.append(arm)
            else:
                out.waiting[arm] = why
        out.ready[task] = [r for arm in ready for r in arms[arm]]
        out.targets.update({arm: targets[arm] for arm in ready})
    return out


def _reading_label(reading: ThresholdReading) -> str:
    sd = "sample SD" if reading.ddof == 1 else "population SD"
    bound = "<=" if reading.inclusive else "<"
    rounding = "unrounded" if reading.decimals is None else f"rounded to 1e-{reading.decimals}"
    return f"{sd}, {bound}, {rounding}"


def _flags(readiness: ArmReadiness, tolerance: float, matcher: Callable[..., Mapping[str, Any]],
           check_code: Callable[[], None] | None) -> tuple[dict[str, dict[str, bool]], dict[str, str]]:
    """``match_arms`` on the ready arms of each task, re-checked by ``rule_matching``: (flags, provisional).

    The flags follow the answer of Q-threshold-arithmetic (Table 9.1), which is a report key, not a result gate
    (configs/registered.py: "they gate no run; while one is open, every verdict that depends on it is
    reported as PROVISIONAL or UNDECIDED by analysis/"); this module gates ``match`` by Q-matched-cost-set
    and Q-arm-complete only, and the sensitivity battery only by its conditions' own keys (Q-hazard,
    Q-dynamics; the module docstring). So every ready arm is flagged by the answer. While the key is
    open, an arm whose flags another of its readings (``THRESHOLD_READINGS``) would give differently
    is returned in ``provisional`` (arm -> what each reading gives), for the enrichment log and the
    command line: the flags are written once, and the group sees which of them its answer decides
    (``analysis`` recomputes the matching under each reading and reports the verdicts that depend on
    it as PROVISIONAL or UNDECIDED).
    """
    flags: dict[str, dict[str, bool]] = {}
    provisional: dict[str, str] = {}
    for task, task_rows in sorted(readiness.ready.items()):
        targets = {a: readiness.targets[a] for a in {_arm_id(r) for r in task_rows}}
        given = matcher([dict(r) for r in task_rows], tolerance=tolerance, seed_targets=targets)
        if check_code is not None:
            check_code()
        expected = rule_matching(task_rows, tolerance=tolerance)
        got = {str(k): {"matched": v.get("matched"), "infeasible": v.get("infeasible")} for k, v in dict(given).items()}
        if got != expected:
            wrong = sorted(k for k in set(got) | set(expected) if got.get(k) != expected.get(k))
            raise ContractError(
                f"task {task}: the matcher's flags differ from rules 2 to 5 at tolerance {tolerance} for {wrong[:5]} "
                f"(matcher {[got.get(k) for k in wrong[:2]]}, rules {[expected.get(k) for k in wrong[:2]]})"
            )
        flags.update(expected)
        if not R.is_open("Q-threshold-arithmetic"):
            continue
        others = {r: rule_matching(task_rows, tolerance=tolerance, reading=r) for r in THRESHOLD_READINGS[1:]}
        first_run: dict[str, str] = {}
        for r in task_rows:
            first_run.setdefault(_arm_id(r), str(r["run_id"]))
        for arm, run_id in sorted(first_run.items()):
            differ = {r: o[run_id] for r, o in others.items() if o[run_id] != expected[run_id]}
            if differ:
                mine = expected[run_id]
                provisional[arm] = (
                    f"task {task}, tolerance {tolerance:g}: written under the answer of Q-threshold-arithmetic "
                    f"({_reading_label(PROPOSED_READING)}: matched {mine['matched']}, infeasible {mine['infeasible']}); "
                    + "; ".join(f"{_reading_label(r)}: matched {v['matched']}, infeasible {v['infeasible']}"
                                for r, v in differ.items())
                    + " (Part 4.1 rules 3 and 5; the key is open, a report key)"
                )
    return flags, provisional


def apply_matching(
    ledger_path: Path,
    *,
    surplus_extra: int,
    stored_surplus_extra: Any = None,
    matcher: Callable[..., Mapping[str, Any]] | None = None,
    check_code: Callable[[], None] | None = None,
) -> Done:
    """``enrich match``: the matched and infeasible flags of Part 4.1 rules 2 to 6, for complete arms.

    Gated by Q-matched-cost-set (the measurement set gives the matched cost) and Q-arm-complete
    (when an arm is complete), checked first. The main study's ledger only (Part 3.6: the pilot's
    matching is G1's). ``surplus_extra`` is E of the seed targets (``seed_targets``) and must equal
    the data root's stored value ``stored_surplus_extra`` (``schedule add-surplus``; None, none
    stored, is 0, as ``Scheduler.seed_target`` reads it). The flags of
    ``analysis.matching.match_arms`` (Role 4) must equal ``rule_matching`` exactly; a flag already in
    the ledger is never changed. An incomplete arm waits: it gets no flag and is returned in
    ``Done.waiting`` with the reason (the command line prints it; the log records it). The flags
    follow the answer of Q-threshold-arithmetic (Table 9.1), a report key: while it is open, the arms whose
    flags another of its readings would change are written like the others and named in
    ``Done.provisional`` and in the log (``_flags``).
    """
    require_answered("Q-matched-cost-set", "Q-arm-complete",
                     what="the matched and infeasible flags (Part 4.1 rules 2 to 6)")
    ledger_path = Path(ledger_path)
    _check_surplus(surplus_extra, stored_surplus_extra)
    rows = _rows(ledger_path)
    _refuse_pilot_rows(rows, ledger_path, "enrich match")
    match = matcher or _load(MATCH_TARGET)
    done = Done()
    commit = _code_commit()
    try:
        readiness = arm_readiness([r.model_dump(mode="python") for r in rows], surplus_extra)
        done.waiting.update(readiness.waiting)
        flags, done.provisional = _flags(readiness, R.MATCH_TOLERANCE, match, check_code)
        current = {r.run_id: r for r in rows}
        for run_id, value in sorted(flags.items()):
            row = current[run_id]
            if (row.matched, row.infeasible) == (value["matched"], value["infeasible"]):
                continue
            _commit(ledger_path, run_id, value)
            done.append(run_id)
    finally:
        _log(ledger_path, "match", done, {"surplus_extra": surplus_extra, "arms_waiting": done.waiting,
                                          "provisional": done.provisional}, commit=commit)
    return done


# ---------------------------------------------------------------------------
# controller and training (Tables 2.1, 2.3, 2.4)
# ---------------------------------------------------------------------------


def controller_reference(epochs: Sequence[int], values: Sequence[float], *, onset_step: int, total_steps: int,
                         steps_per_epoch: int) -> dict[str, Any]:
    """Table 2.4 at epoch resolution, computed independently of ``metrics.controller`` (Q-controller-quantities,
    answered in Table 9.1).

    The value logged at epoch e is the multiplier after (e + 1) x E steps. ``lambda_peak``: the
    maximum over the epochs ending in (onset, onset + 2,000,000]; ``lambda_final``: the ``math.fsum``
    mean over the last ceil(0.10 x epochs) epochs, the count computed exactly; ``settling_steps``:
    (e* + 1) x E - onset for the first epoch e* at or after onset from which every value lies within
    +/-10 percent of lambda_final (inclusive; with lambda_final 0 only an exact 0 does), None only if
    the last value lies outside. A non-finite or negative multiplier is refused, as Role 3 refuses it.
    """
    bad = _bad_multipliers(values)
    if bad:
        raise ContractError(f"the multiplier trace has non-finite or negative values {bad}")
    n = len(values)
    if list(epochs) != list(range(n)) or n * steps_per_epoch != total_steps:
        raise ContractError(f"the multiplier trace has {n} epochs of {steps_per_epoch} steps; the run has {total_steps}")
    first = onset_step // steps_per_epoch
    window = [values[e] for e in range(first, n)
              if onset_step < (e + 1) * steps_per_epoch <= onset_step + R.OVERSHOOT_WINDOW_STEPS]
    share = Fraction(str(R.OVERSHOOT_FINAL_FRACTION))
    count = -((-share.numerator * n) // share.denominator)  # ceil(share x n) in integers: no float product
    final = math.fsum(values[n - count:]) / count
    settling = None
    band = R.SETTLING_BAND * final  # 0 for a final value of 0: every value but an exact 0 lies outside
    outside = [e for e in range(first, n) if abs(values[e] - final) > band]
    start = (outside[-1] + 1) if outside else first
    if start < n:
        settling = (start + 1) * steps_per_epoch - onset_step
    return {"lambda_peak": max(window) if window else None, "lambda_final": final, "settling_steps": settling}


def _bad_multipliers(values: Sequence[float]) -> list[float]:
    """The values of a multiplier trace that are not finite or are negative (the first five)."""
    return [v for v in values if not math.isfinite(v) or v < 0.0][:5]


def _progress_trace(files: RunFiles, column: str, *, required: bool) -> tuple[list[int], list[float] | None]:
    """(epochs, values of ``column``) of the run's progress.csv; values None when the column is absent."""
    try:
        rows = read_progress(files.omnisafe_dir)
    except OSError as exc:
        raise ContractError(f"{files.spec.run_id}: cannot read progress.csv: {exc}") from None
    epochs = []
    for r in rows:
        try:
            value = float(r.get(EPOCH_COLUMN) or "nan")
        except ValueError:
            value = math.nan
        if not value.is_integer():
            raise ContractError(f"{files.spec.run_id}: progress.csv has a row whose {EPOCH_COLUMN} is {r.get(EPOCH_COLUMN)!r}")
        epochs.append(int(value))
    if not rows or column not in rows[0]:
        if required:
            raise ContractError(f"{files.spec.run_id}: progress.csv has no column {column!r}")
        return epochs, None
    values = []
    for r in rows:
        try:
            values.append(float(r[column]))
        except (TypeError, ValueError):
            raise ContractError(f"{files.spec.run_id}: {column} of epoch {r.get(EPOCH_COLUMN)} is {r[column]!r}") from None
    return epochs, values


def _controller(files: RunFiles, quantities: Callable[..., Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """(Role 3's quantities, this module's reference) over the run's multiplier trace."""
    spe = files.steps_per_epoch()
    epochs, values = _progress_trace(files, MULTIPLIER_COLUMN, required=False)
    onset = int(files.spec.onset_step or 0)
    bad = _bad_multipliers(values) if values is not None else []
    if bad:
        raise ContractError(f"{files.spec.run_id}: the multiplier trace has non-finite or negative values {bad}")
    given = dict(quantities(epochs, values, onset_step=onset, total_steps=files.spec.total_steps, steps_per_epoch=spe))
    if values is None:
        return given, {"lambda_peak": None, "lambda_final": None, "settling_steps": None}
    return given, controller_reference(epochs, values, onset_step=onset, total_steps=files.spec.total_steps,
                                       steps_per_epoch=spe)


def _check_controller(run_id: str, given: Mapping[str, Any], expected: Mapping[str, Any]) -> dict[str, Any]:
    out = {k: given.get(k) for k in ("lambda_peak", "lambda_final", "settling_steps")}
    if out != dict(expected) or any(isinstance(v, bool) for v in out.values()):
        raise ContractError(f"{run_id}: metrics.controller gives {out}, Table 2.4 at epoch resolution gives {dict(expected)}")
    return out


def rate_limit_clip_reference(epochs: Sequence[int], multipliers: Sequence[float], proposed: Sequence[float], *,
                              onset_step: int, total_steps: int, steps_per_epoch: int) -> float:
    """Q-rate-limit (Table 9.1), computed independently of ``metrics.controller``: "the share of constrained
    epochs in which the clip bound".

    The constrained epochs start at or after onset (epoch e starts at e x E); the clip bound where the
    value before the clip (``Onset/MultiplierProposed``) is not the logged multiplier. A trace that
    does not cover the run, or a non-finite or negative value in a constrained epoch, is refused.
    """
    n = len(multipliers)
    if list(epochs) != list(range(n)) or len(proposed) != n or n * steps_per_epoch != total_steps:
        raise ContractError(f"the multiplier traces have {n} and {len(proposed)} epochs of {steps_per_epoch} steps; "
                            f"the run has {total_steps}")
    constrained = [e for e in range(n) if e * steps_per_epoch >= onset_step]
    bad = _bad_multipliers([v for e in constrained for v in (multipliers[e], proposed[e])])
    if bad or not constrained:
        raise ContractError(f"the constrained epochs' multiplier or proposed values are non-finite or negative {bad}"
                            if bad else "the multiplier trace has no constrained epoch")
    return sum(1 for e in constrained if proposed[e] != multipliers[e]) / len(constrained)


def _clip_share(files: RunFiles, share: Callable[..., Any] | None) -> float | None:
    """The rate clip's share of a rate-limited run (Role 3's ``rate_limit_clip_share``, which must equal
    ``rate_limit_clip_reference`` exactly), None for every other run. ``share`` None loads Role 3's."""
    if files.spec.controller_variant != "rate_limited":
        return None
    epochs, values = _progress_trace(files, MULTIPLIER_COLUMN, required=True)
    _, proposed = _progress_trace(files, PROPOSED_MULTIPLIER_COLUMN, required=True)
    kw = {"onset_step": int(files.spec.onset_step or 0), "total_steps": files.spec.total_steps,
          "steps_per_epoch": files.steps_per_epoch()}
    try:
        given = (share or _load(CLIP_SHARE_TARGET))(epochs, values, proposed, **kw)
    except ValueError as exc:
        raise ContractError(f"{files.spec.run_id}: {exc}") from None
    expected = rate_limit_clip_reference(epochs, values, proposed, **kw)
    if isinstance(given, bool) or given != expected:
        raise ContractError(f"{files.spec.run_id}: metrics.controller gives the rate clip's share {given!r}, the share "
                            f"of constrained epochs in which the clip bound is {expected!r} (Q-rate-limit)")
    return expected


def apply_controller(
    ledger_path: Path,
    run_dir_of: Callable[[str], Path],
    *,
    quantities: Callable[..., Mapping[str, Any]] | None = None,
    clip_share: Callable[..., Any] | None = None,
    check_code: Callable[[], None] | None = None,
    data_root: str | Path | None = None,
) -> Done:
    """``enrich controller``: lambda_peak, lambda_final and settling_steps of every completed Study A row.

    ``metrics.controller.controller_quantities`` (Role 3; its result gate Q-controller-quantities)
    over the ``Metrics/LagrangeMultiplier`` trace of the run's progress.csv, with the run's own steps
    per epoch (config.json), must equal ``controller_reference`` exactly. The run directory must hold
    the row's run: the same training commit and the final checkpoint at the path the row stores
    (``_final_checkpoint``), so a namesake under another data root is never read. A run without a
    multiplier column gets nothing. An existing value is never changed. For a rate-limited run, the
    share of constrained epochs in which the clip bound (``clip_share``, default Role 3's
    ``rate_limit_clip_share``; Q-rate-limit) is checked against ``rate_limit_clip_reference`` and
    logged (``rate_limit_clip_share``): the ledger has no field for it, and the ``training`` record
    carries it to the analysis (H4).
    """
    ledger_path = Path(ledger_path)
    compute = quantities or _load(CONTROLLER_TARGET)
    done = Done()
    commit = _code_commit()
    shares: dict[str, float] = {}
    try:
        for row in _rows(ledger_path):
            if row.study != "A" or not row.completed or row.lambda_final is not None:
                continue
            files = open_run(row, run_dir_of)
            _final_checkpoint(row, files)  # the row's run, not a namesake under another data root
            given, expected = _controller(files, compute)
            share = _clip_share(files, clip_share)
            if check_code is not None:
                check_code()
            fields = _check_controller(row.run_id, given, expected)
            if fields["lambda_final"] is None:
                continue  # no multiplier (Table B.1's controller quantities need one)
            _commit(ledger_path, row.run_id, fields)
            done.append(row.run_id)
            if share is not None:
                shares[row.run_id] = share
    finally:
        _log(ledger_path, "controller", done, {"rate_limit_clip_share": shares, "data_root": _abs(data_root)},
             commit=commit)
    return done


def _plasticity_row(step: int, values: Mapping[str, float] | None) -> dict[str, Any] | None:
    """A ``PlasticityRow`` of the supplement: the known columns, a non-finite value as None."""
    if values is None:
        return None
    row: dict[str, Any] = {"step": step}
    for name in PLASTICITY_ROW_FIELDS:
        value = values.get(name)
        row[name] = None if value is None or not math.isfinite(value) else float(value)
    return row


def training_record(
    row: Any,
    files: RunFiles,
    commit: str,
    *,
    recovery: Callable[..., Any],
    quantities: Callable[..., Mapping[str, Any]],
    summary: Callable[..., Mapping[str, Any]],
    clip_share: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """The ``training`` supplement record of one completed Study A run (results/supplement_schema.py).

    * recovery (Table 2.1 "Recovery time", "late arms only"): ``metrics.recovery.recovery_steps`` over
      the run's ``Metrics/BatchEpCost``; None and censored when the cost never fell to d; None for N = 0;
    * the controller quantities and the overshoot lambda_peak - lambda_final (Table 2.4), checked
      as in ``enrich controller`` and against the ledger's values when present;
    * for a rate-limited run, the share of constrained epochs in which the clip bound
      (``rate_limit_clip_share``; Q-rate-limit, Table 9.1; ``clip_share`` as in ``enrich controller``),
      None for every other run;
    * the plasticity.csv rows at onset and at onset + 200,000 steps (Table 2.3; box H3 (c)), every
      column of contract 2 including the critics' norms;
    * the intervention summary (Table 2.4; ``metrics.interventions.intervention_summary`` at the
      onset step) from intervention.json, required for a reset or injection run and refused for any other.
    """
    spec = files.spec
    onset = int(spec.onset_step or 0)
    spe = files.steps_per_epoch()
    recovery_steps = censored = None
    if onset > 0:
        epochs, costs = _progress_trace(files, BATCH_COST_KEY, required=True)
        recovery_steps = recovery(epochs, costs, onset_step=onset, steps_per_epoch=spe, total_steps=spec.total_steps)
        if recovery_steps is not None and (isinstance(recovery_steps, bool) or not isinstance(recovery_steps, int)):
            raise ContractError(f"{row.run_id}: metrics.recovery gives {recovery_steps!r}, not a whole number of steps")
        censored = recovery_steps is None
    given, expected = _controller(files, quantities)
    controller = _check_controller(row.run_id, given, expected)
    for name, value in controller.items():
        stored = getattr(row, name)
        if stored is not None and stored != value:
            raise ContractError(f"{row.run_id}: {name} is {stored!r} in the ledger, {value!r} now")
    peak, final = controller["lambda_peak"], controller["lambda_final"]
    share = _clip_share(files, clip_share)
    try:
        plasticity = validate_plasticity(files.omnisafe_dir, spec, completed=True)
    except ContractError as exc:
        raise ContractError(f"{row.run_id}: {exc}") from None
    check = onset + R.MANIPULATION_CHECK_STEPS_AFTER_ONSET
    intervention_path = files.omnisafe_dir / INTERVENTION_FILE
    intervention = None
    if intervention_path.exists():
        if spec.treatment not in ("reset", "injection"):
            raise ContractError(f"{row.run_id}: {INTERVENTION_FILE} exists, but the run's treatment is {spec.treatment!r}")
        record = _read_json(intervention_path, row.run_id)
        try:
            intervention = _plain(dict(summary(record, step=onset)))
        except ValueError as exc:
            raise ContractError(f"{row.run_id}: {INTERVENTION_FILE}: {exc}") from None
        if intervention.get("treatment") != spec.treatment:
            raise ContractError(f"{row.run_id}: {INTERVENTION_FILE} records {intervention.get('treatment')!r}, the "
                                f"spec {spec.treatment!r}")
    elif spec.treatment in ("reset", "injection"):
        raise ContractError(f"{row.run_id}: a {spec.treatment} run has no {INTERVENTION_FILE} (HANDOVER.md section 8)")
    return {
        "kind": "training", "run_id": row.run_id, "code_commit": commit,
        "onset_step": onset, "total_steps": spec.total_steps, "steps_per_epoch": spe,
        "recovery_steps": recovery_steps, "recovery_censored": censored,
        "lambda_peak": peak, "lambda_final": final,
        "overshoot": None if peak is None or final is None else peak - final,
        "settling_steps": controller["settling_steps"],
        "rate_limit_clip_share": share,
        "plasticity_onset": _plasticity_row(onset, plasticity.get(onset)),
        "plasticity_check": _plasticity_row(check, plasticity.get(check)),
        "intervention": intervention,
    }


def apply_training(
    ledger_path: Path,
    run_dir_of: Callable[[str], Path],
    *,
    recovery: Callable[..., Any] | None = None,
    quantities: Callable[..., Mapping[str, Any]] | None = None,
    summary: Callable[..., Mapping[str, Any]] | None = None,
    clip_share: Callable[..., Any] | None = None,
    check_code: Callable[[], None] | None = None,
    data_root: str | Path | None = None,
) -> Done:
    """``enrich training``: the ``training`` supplement record of every completed Study A row lacking one.

    No ledger field is written (schema v1 has none for these; the controller fields are
    ``enrich controller``'s). Role 3's functions are gated inside by Q-controller-quantities;
    ``clip_share`` (default Role 3's ``rate_limit_clip_share``) gives a rate-limited run's
    ``rate_limit_clip_share`` (Q-rate-limit). The run directory must hold the row's run, as for
    ``enrich controller`` (``_final_checkpoint``).
    """
    ledger_path = Path(ledger_path)
    recover = recovery or _load(RECOVERY_TARGET)
    compute = quantities or _load(CONTROLLER_TARGET)
    summarise = summary or _load(INTERVENTION_TARGET)
    commit = _code_commit()
    done = Done()
    try:
        for row in _rows(ledger_path):
            if row.study != "A" or not row.completed or supplement.exists("training", row.run_id, ledger_path=ledger_path):
                continue
            files = open_run(row, run_dir_of)
            _final_checkpoint(row, files)  # the row's run, not a namesake under another data root
            record = training_record(row, files, commit, recovery=recover, quantities=compute, summary=summarise,
                                     clip_share=clip_share)
            if check_code is not None:
                check_code()
            _commit(ledger_path, row.run_id, records=[_Record("training", row.run_id, record)])
            done.append(row.run_id)
    finally:
        _log(ledger_path, "training", done, {"data_root": _abs(data_root)}, commit=commit)
    return done


# ---------------------------------------------------------------------------
# Continuation runs: battery fine-tuning and transfer, Study B few-shot (pilot/manifest.py)
# ---------------------------------------------------------------------------


def _continuation_waits(run_id: str, run_dir_of: Callable[[str], Path],
                        status_of: Callable[[str], str | None] | None) -> str | None:
    """None when the continuation ended ``continued`` (without scheduler state: its training
    completed); otherwise why it is not ready yet (not an error: it has not run)."""
    if status_of is not None:
        status = status_of(run_id)
        if status is None:
            return "not queued on this data root"
        if status != "continued":
            return f"its status is {status!r}, not 'continued'"
    path = Path(run_dir_of(run_id)) / TRAIN_RESULT
    if not path.exists():
        return "not trained yet"
    train = _read_json(path, run_id)
    if train.get("status") != "completed":
        return f"its training is {train.get('status')!r}"
    return None


def _open_continuation(row: Any, cont_id: str, allowed: Sequence[RunSpec], run_dir_of: Callable[[str], Path], *,
                       parent_checkpoint: Path, parent_step: int) -> RunFiles:
    """The continuation's run directory, checked to continue ``row``'s run from ``parent_checkpoint``.

    Its spec.json is the design's (``allowed``); its training completed, at a full commit hash; and its
    train_result.json records the parent (pilot/dependencies.py): the row's commit, the checkpoint step, and the SHA-256 of the
    parent's checkpoint file as it is now, so a continuation never reads into a row it did not start from.
    A continuation trained as a smoke run (``ledger_writer.SMOKE_MARKERS``: uncommitted code, or open questions
    bypassed with --allow-pending) is read only into the row of a smoke run (a smoke ledger's): its values are
    written once and never changed, so it may never reach a registered row (as ``write_run`` and
    ``dependencies.resolve`` refuse smoke runs).
    """
    run_dir = Path(run_dir_of(cont_id))
    try:
        spec = RunSpec.from_dict(_read_json(run_dir / SPEC_FILE, cont_id))
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{run_dir / SPEC_FILE} is not a run spec: {exc}") from None
    if spec not in allowed:
        raise ContractError(f"{cont_id}: {run_dir / SPEC_FILE} is not the continuation the design builds from the ledger "
                            f"row of {row.run_id} (pilot/manifest.py)")
    train = _read_json(run_dir / TRAIN_RESULT, cont_id)
    if train.get("status") != "completed" or not train.get("omnisafe_dir"):
        raise ContractError(f"{cont_id}: its training did not complete")
    if not provenance.is_full_commit_hash(str(train.get("commit_hash") or "")):
        raise ContractError(f"{cont_id}: {TRAIN_RESULT} records no full commit hash ({train.get('commit_hash')!r})")
    markers = [k for k in SMOKE_MARKERS if train.get(k)]
    if markers and not _is_smoke(Path(run_dir_of(row.run_id))):
        raise ContractError(f"{cont_id} was trained as a smoke run ({', '.join(markers)}: uncommitted code or open "
                            f"questions bypassed); it may not be read into the row of {row.run_id}, a registered run")
    entry = (train.get("dependencies") or {}).get(row.run_id)
    if not isinstance(entry, Mapping):
        raise ContractError(f"{cont_id}: {TRAIN_RESULT} records no dependency on {row.run_id}")
    if entry.get("commit_hash") != row.commit_hash:
        raise ContractError(f"{cont_id}: its parent was trained at {entry.get('commit_hash')!r}, the ledger row of "
                            f"{row.run_id} records {row.commit_hash!r}")
    if entry.get("checkpoint_step") != parent_step:
        raise ContractError(f"{cont_id} started from step {entry.get('checkpoint_step')!r} of {row.run_id}, not {parent_step}")
    if not parent_checkpoint.is_file():
        raise ContractError(f"{row.run_id}: the checkpoint {parent_checkpoint} does not exist")
    if entry.get("checkpoint_sha256") != _sha256(parent_checkpoint):
        raise ContractError(f"{cont_id} started from a checkpoint of {row.run_id} whose SHA-256 is not that of "
                            f"{parent_checkpoint} now")
    return RunFiles(run_dir, spec, train, run_dir / train["omnisafe_dir"])


def _parent_checkpoint(files: RunFiles, step: int) -> Path:
    """The parent's checkpoint file at ``step``: the one ``_check_checkpoint`` verified against the ledger.

    The ledger's step-to-path mapping (``matched_checkpoint_path``, as ``ledger_writer`` writes it)
    assumes R.STEPS_PER_EPOCH; a run of another epoch length saved that step under another name, so
    it is refused rather than hashed under a name the ledger never verified.
    """
    spe = files.steps_per_epoch()
    if spe != R.STEPS_PER_EPOCH:
        raise ContractError(f"{files.spec.run_id}: its epochs are {spe} steps, but the ledger's checkpoint paths "
                            f"assume {R.STEPS_PER_EPOCH} (ledger_writer); its checkpoint at step {step} is not the "
                            "one the ledger records")
    return checkpoint_file(files.omnisafe_dir, step // spe)


def continuation_record(row: Any, cont: RunFiles, result: Mapping[str, Any], commit: str, *,
                        selection_seeds: Iterable[int]) -> tuple[_Record, list[int]]:
    """``envs.evaluation.evaluate_continuation``'s result checked, as a ``continuation`` record, with its seeds.

    Table 2.2: the cost is "measured after fine-tuning on the training task" (transfer: "on the
    held-out task"), at the continuation's final checkpoint, on the measurement set (Table 2.1;
    HANDOVER.md section 8, ``evaluate_continuation``); the gap is C_after - C_ID with C_ID the parent's
    measurement_cost (equation 1).
    """
    raw = _plain(dict(result))
    what = cont.spec.run_id
    _check_episodes(raw, f"{what}: the continuation must be evaluated on {R.EVAL_EPISODES} episodes")
    if raw.get("step") != cont.spec.total_steps:
        raise ContractError(f"{what}: evaluated at step {raw.get('step')!r}, not the final {cont.spec.total_steps}")
    if raw.get("task") != cont.spec.task:
        raise ContractError(f"{what}: evaluated on {raw.get('task')!r}, not its task {cont.spec.task!r}")
    seeds = validate_seed_set("measurement_seeds", raw.get("measurement_seeds"))
    if set(seeds) & set(selection_seeds):
        raise ContractError(f"{what}: measurement and selection seeds overlap (Table 2.1 requires disjoint sets)")
    cost = _finite(raw.get("cost"), f"{what}: the cost")
    costs, returns = raw.get("episode_costs"), raw.get("episode_returns")
    if not isinstance(costs, list) or not isinstance(returns, list):
        raise ContractError(f"{what}: the result holds no per-episode costs and returns (results/supplement_schema.py)")
    condition = str(cont.spec.params.get("condition"))
    record = {
        "kind": "continuation", "run_id": row.run_id, "code_commit": commit, "condition": condition,
        "continuation_run_id": cont.spec.run_id, "parent_step": int(cont.spec.params["parent_step"]),
        "continuation_steps": cont.spec.total_steps, "step": raw["step"], "task": raw["task"],
        "episodes": _block("measurement", seeds, costs, returns, cost, raw.get("return"),
                           replaced=raw.get("unstable_replacements"), what=what),
        "measurement_cost": row.measurement_cost, "gap": cost - row.measurement_cost,
    }
    return _Record("continuation", row.run_id, record, condition), seeds


def apply_continuations(
    ledger_path: Path,
    run_dir_of: Callable[[str], Path],
    conditions: Iterable[str] = manifest.BATTERY_CONDITIONS,
    *,
    evaluator: Callable[..., dict[str, Any]] | None = None,
    status_of: Callable[[str], str | None] | None = None,
    check_code: Callable[[], None] | None = None,
    data_root: str | Path | None = None,
) -> Done:
    """``enrich continuations``: gap_finetune and gap_transfer from the battery's continuation runs (Table 2.2).

    For every matched row of the main study (Part 4.1 rule 6) lacking a gap or its ``continuation``
    record: the continuation ``manifest.battery_continuation(parent, row, condition)`` must have ended
    ``continued`` (``status_of``: the scheduler's status; without it, a completed training) and
    started from the row's matched checkpoint (``_open_continuation``); it is evaluated by
    ``envs.evaluation.evaluate_continuation`` and gap = C_after - measurement_cost. Continuations
    that have not run yet are returned in ``Done.waiting`` and logged (``not_ready``). The continuations' run_ids,
    commits and configuration hashes are logged (the ledger has no field for them).
    """
    ledger_path = Path(ledger_path)
    conditions = _conditions(conditions, CONTINUATION_FIELDS, "conditions")
    rows = _rows(ledger_path)
    _refuse_pilot_rows(rows, ledger_path, "enrich continuations")
    evaluate = evaluator or _load(CONTINUATION_TARGET)
    commit = _code_commit()
    done = Done()
    not_ready = done.waiting  # returned with the run_ids: the command line prints what is not ready
    written: dict[str, dict[str, Any]] = {}
    try:
        for row in rows:
            if row.study != "A" or not row.completed or row.matched is not True:
                continue
            todo = [c for c in conditions if getattr(row, CONTINUATION_FIELDS[c]) is None
                    or not supplement.exists("continuation", row.run_id, ledger_path=ledger_path, part=c)]
            if not todo:
                continue
            if row.measurement_cost is None:
                not_ready[row.run_id] = "no measurement_cost yet (`enrich measure`)"
                continue
            parent = _open_matched(row, run_dir_of)
            step = int(row.matched_checkpoint_step)
            checkpoint = _parent_checkpoint(parent, step)
            selection = _selection_seeds(parent)
            for condition in todo:
                expected = manifest.battery_continuation(parent.spec, row.model_dump(mode="python"), condition)
                why = _continuation_waits(expected.run_id, run_dir_of, status_of)
                if why is not None:
                    not_ready[expected.run_id] = why
                    continue
                cont = _open_continuation(row, expected.run_id, [expected], run_dir_of,
                                          parent_checkpoint=checkpoint, parent_step=step)
                result = evaluate(str(cont.omnisafe_dir), cont.spec.to_dict())
                if check_code is not None:
                    check_code()
                record, seeds = continuation_record(row, cont, result, commit, selection_seeds=selection)
                _commit(ledger_path, row.run_id, {CONTINUATION_FIELDS[condition]: record.record["gap"]},
                        records=[record], seeds=[("measurement", seeds)])
                if row.run_id not in done:
                    done.append(row.run_id)
                written.setdefault(row.run_id, {})[condition] = {
                    "run_id": cont.spec.run_id, "commit": cont.train.get("commit_hash"),
                    "config_hash": cont.train.get("config_hash")}
    finally:
        _log(ledger_path, "continuations", done, {"conditions": conditions, "continuations": written,
                                                  "not_ready": not_ready,
                                                  "data_root": _abs(data_root)}, commit=commit)
    return done


# ---------------------------------------------------------------------------
# Study B: zero-shot and few-shot (Table 2.5; equation 10)
# ---------------------------------------------------------------------------


def _by_budget(mapping: Any, what: str) -> dict[float, Any]:
    if not isinstance(mapping, Mapping):
        raise ContractError(f"{what} must map budgets to values")
    try:
        return {float(k): v for k, v in mapping.items()}
    except (TypeError, ValueError):
        raise ContractError(f"{what}: a key is not a budget") from None


def _study_b_seeds(raw: Mapping[str, Any], what: str) -> tuple[str, list[int]]:
    """A Study B result's seed set and seeds: the measurement set only (Q-studyb-eval, Table 9.1: "Every Study B
    evaluation with a budget in the observation uses Table 2.1's measurement set")."""
    seed_set = raw.get("seed_set")
    if seed_set != "measurement":
        raise ContractError(f"{what}: seed_set {seed_set!r} is not the measurement set (Q-studyb-eval, Table 9.1)")
    return seed_set, validate_seed_set("seeds", raw.get("seeds"))


def zero_shot_record(row: Any, files: RunFiles, result: Mapping[str, Any], commit: str
                     ) -> tuple[_Record, str, list[int], dict[float, float]]:
    """``studyb.evaluation.evaluate_zero_shot``'s result checked, as a ``zero_shot`` record.

    The final checkpoint (Q-studyb-eval, Table 9.1), 100 episodes at each unseen budget of Table
    2.5 and at the arm's reference budgets (box G3; Q-g3-arms); the schema recomputes equation (10)'s
    satisfaction, the violation magnitude and equation (11)'s distance from the episodes. Returns the
    record, its seed set and seeds, and the validated record's sr_zero.
    """
    raw = _plain(dict(result))
    what = row.run_id
    _check_episodes(raw, f"{what}: the zero-shot evaluation must use {R.EVAL_EPISODES} episodes per budget")
    if raw.get("step") != files.spec.total_steps:
        raise ContractError(f"{what}: evaluated at step {raw.get('step')!r}, not the final {files.spec.total_steps}")
    seed_set, seeds = _study_b_seeds(raw, what)
    unseen = [float(b) for b in raw.get("unseen_budgets") or ()]
    if sorted(unseen) != sorted(float(b) for b in R.UNSEEN_BUDGETS):
        raise ContractError(f"{what}: unseen budgets {unseen}, Table 2.5 has {list(R.UNSEEN_BUDGETS)}")
    reference = [float(b) for b in raw.get("reference_budgets") or ()]
    maps = {name: _by_budget(raw.get(name), f"{what}: {name}") for name in
            ("mean_cost", "mean_return", "satisfaction", "violation", "episode_costs", "episode_returns")}
    distance = _by_budget(raw.get("distance"), f"{what}: distance")
    replaced = _by_budget(raw.get("unstable_replacements") or {}, f"{what}: unstable_replacements")
    blocks = []
    for budget in unseen + reference:
        try:
            block = {name: maps[name][budget] for name in maps}
        except KeyError:
            raise ContractError(f"{what}: the result has no values at budget {budget:g}") from None
        blocks.append({"budget": budget, "role": "unseen" if budget in unseen else "reference", **block,
                       "distance": distance.get(budget) if budget in unseen else None,
                       "unstable_replacements": _replacements(replaced.get(budget), seeds, f"{what} budget {budget:g}")})
    record = {"kind": "zero_shot", "run_id": row.run_id, "code_commit": commit, "arm": files.spec.arm,
              "step": raw["step"], "seed_set": seed_set, "seeds": seeds, "budgets": blocks}
    rec = _Record("zero_shot", row.run_id, record)
    model = _validated(rec)
    given = _by_budget(raw.get("sr_zero"), f"{what}: sr_zero")
    if given != model.sr_zero:
        raise ContractError(f"{what}: sr_zero {given} differs from the unseen budgets' satisfaction {model.sr_zero}")
    return rec, seed_set, seeds, dict(model.sr_zero)


def apply_zero_shot(
    ledger_path: Path,
    run_dir_of: Callable[[str], Path],
    *,
    evaluator: Callable[..., dict[str, Any]] | None = None,
    check_code: Callable[[], None] | None = None,
    data_root: str | Path | None = None,
) -> Done:
    """``enrich zeroshot``: sr_zero and the ``zero_shot`` record of every completed Study B row.

    Part 3.3: "Each finished run is evaluated zero-shot on the four unseen budgets". The main study
    only; the harness is ``studyb.evaluation.evaluate_zero_shot`` (Role 5; gated inside by
    Q-studyb-eval), at the run's final checkpoint, checked to be the one the row records."""
    ledger_path = Path(ledger_path)
    rows = _rows(ledger_path)
    _refuse_pilot_rows(rows, ledger_path, "enrich zeroshot")
    evaluate = evaluator or _load(ZERO_SHOT_TARGET)
    commit = _code_commit()
    done = Done()
    try:
        for row in rows:
            if row.study != "B" or not row.completed or row.arm not in R.STUDY_B_ARMS:
                continue
            if row.sr_zero is not None and supplement.exists("zero_shot", row.run_id, ledger_path=ledger_path):
                continue
            files = open_run(row, run_dir_of)
            _final_checkpoint(row, files)
            result = evaluate(str(files.omnisafe_dir), files.spec.to_dict())
            if check_code is not None:
                check_code()
            record, seed_set, seeds, sr_zero = zero_shot_record(row, files, result, commit)
            _commit(ledger_path, row.run_id, {"sr_zero": sr_zero}, records=[record], seeds=[(seed_set, seeds)])
            done.append(row.run_id)
    finally:
        _log(ledger_path, "zeroshot", done, {"data_root": _abs(data_root)}, commit=commit)
    return done


def adaptation_steps(rates: Mapping[int, float], horizons: Sequence[int]) -> tuple[int, bool]:
    """Table 2.5 "Adaptation steps": (the smallest horizon whose rate reaches ``R.SATISFACTION_TARGET``,
    False), or (the largest horizon + 1, True) when none does: "recorded as above the largest horizon
    if never reached", the answer of Q-adapt-censoring (Table 9.1), with the continuation's own horizons."""
    for horizon in sorted(horizons):
        if rates[horizon] >= R.SATISFACTION_TARGET:
            return horizon, False
    return max(horizons) + 1, True  # Q-adapt-censoring (Table 9.1): one step above the largest horizon


def fewshot_record(row: Any, cont: RunFiles, budget: float, result: Mapping[str, Any], commit: str
                   ) -> tuple[_Record, str, list[int], dict[str, float], int, bool]:
    """``studyb.evaluation.evaluate_fewshot``'s result checked, as a ``fewshot`` record.

    The horizons are the continuation's own (``spec.params["horizons"]``; Part 6.1 cut 3 may leave
    one), each horizon's block is the checkpoint of that horizon, and the adaptation steps follow
    ``adaptation_steps``. Returns the record, its seed set and seeds, the sr_fewshot entries, the
    adaptation steps and whether they are censored.
    """
    raw = _plain(dict(result))
    what = cont.spec.run_id
    horizons = [int(h) for h in cont.spec.params.get("horizons") or ()]
    if [int(h) for h in raw.get("horizons") or ()] != horizons:
        raise ContractError(f"{what}: evaluated at horizons {raw.get('horizons')!r}, its spec has {horizons}")
    if _real_or_nan(raw.get("budget")) != budget or _real_or_nan(cont.spec.params.get("budget")) != budget:
        raise ContractError(f"{what}: the result's budget {raw.get('budget')!r} is not the continuation's {budget:g}")
    if raw.get("parent_run_id") != row.run_id or raw.get("parent_step") != cont.spec.params.get("parent_step"):
        raise ContractError(f"{what}: the result names parent {raw.get('parent_run_id')!r} at step {raw.get('parent_step')!r}")
    _check_episodes(raw, f"{what}: the few-shot evaluation must use {R.EVAL_EPISODES} episodes per horizon")
    seed_set, seeds = _study_b_seeds(raw, what)
    by_horizon = raw.get("by_horizon")
    if not isinstance(by_horizon, Mapping):
        raise ContractError(f"{what}: the result has no by_horizon map")
    blocks = {int(h): b for h, b in by_horizon.items()}
    if sorted(blocks) != horizons:
        raise ContractError(f"{what}: by_horizon holds {sorted(blocks)}, not the horizons {horizons}")
    horizon_blocks = []
    for h in horizons:
        block = blocks[h]
        if block.get("step", h) != h:
            raise ContractError(f"{what}: the block of horizon {h} evaluated step {block.get('step')!r}")
        horizon_blocks.append({"horizon": h, **{k: block.get(k) for k in (
            "episode_costs", "episode_returns", "mean_cost", "mean_return", "satisfaction", "violation")},
            "unstable_replacements": _replacements(block.get("unstable_replacements"), seeds, f"{what} horizon {h}")})
    rates = {b["horizon"]: _finite(b["satisfaction"], f"{what}: a satisfaction rate") for b in horizon_blocks}
    steps, censored = adaptation_steps(rates, horizons)
    sr = {fewshot_key(budget, h): rates[h] for h in horizons}
    if dict(raw.get("sr_fewshot") or {}) != sr:
        raise ContractError(f"{what}: sr_fewshot {raw.get('sr_fewshot')!r} differs from the horizons' rates {sr}")
    given = _by_budget(raw.get("adapt_steps"), f"{what}: adapt_steps")
    if given != {budget: steps}:
        raise ContractError(f"{what}: adapt_steps {given} differs from Table 2.5's rule, {steps}")
    record = {"kind": "fewshot", "run_id": row.run_id, "code_commit": commit, "continuation_run_id": cont.spec.run_id,
              "parent_step": int(cont.spec.params["parent_step"]), "arm": cont.spec.arm, "budget": budget,
              "horizons": horizons, "seed_set": seed_set, "seeds": seeds, "by_horizon": horizon_blocks,
              "sr_fewshot": sr, "adapt_steps": steps}
    return _Record("fewshot", row.run_id, record, f"b{budget:g}"), seed_set, seeds, sr, steps, censored


def _fewshot_specs(parent: RunSpec, budget: float) -> list[RunSpec]:
    """The continuation of ``parent`` at ``budget`` as the design builds it, and as Part 6.1 cut 3 shortens it."""
    full = next(s for s in manifest.study_b_fewshot([parent]) if float(s.params["budget"]) == budget)
    cut = manifest.apply_cuts([full], manifest.CUTS[: manifest.CUTS.index("fewshot_short") + 1])
    return [full, *cut]


def apply_fewshot(
    ledger_path: Path,
    run_dir_of: Callable[[str], Path],
    *,
    evaluator: Callable[..., dict[str, Any]] | None = None,
    status_of: Callable[[str], str | None] | None = None,
    check_code: Callable[[], None] | None = None,
    data_root: str | Path | None = None,
) -> Done:
    """``enrich fewshot``: sr_fewshot and adapt_steps of every completed Study B row, per unseen budget.

    Each budget is read from its own continuation (Table 2.5: "The three horizons are read from one
    continuation, not three"), once it ended ``continued`` and only if it started from the row's
    final checkpoint (pilot/manifest.py; ``_open_continuation``). The harness is
    ``studyb.evaluation.evaluate_fewshot`` (Role 5; gated inside by Q-studyb-eval). Q-adapt-censoring
    "Gates adaptation steps only when a budget never reaches the target": while it is open, such a
    budget's sr_fewshot rates (equation 10 at each horizon, which do not depend on how censoring is
    written) are written, and its adapt_steps entry and ``fewshot`` record (which carries adapt_steps)
    wait; the other budgets are written whole, and the call then refuses naming what waits.
    Continuations that have not run yet are returned in ``Done.waiting``.
    """
    ledger_path = Path(ledger_path)
    rows = _rows(ledger_path)
    _refuse_pilot_rows(rows, ledger_path, "enrich fewshot")
    evaluate = evaluator or _load(FEWSHOT_TARGET)
    commit = _code_commit()
    done = Done()
    not_ready = done.waiting  # returned with the run_ids: the command line prints what is not ready
    waiting = _Waiting()
    try:
        for row in rows:
            if row.study != "B" or not row.completed or row.arm not in R.STUDY_B_ARMS:
                continue
            parent: RunFiles | None = None
            for budget in (float(b) for b in R.UNSEEN_BUDGETS):
                part = f"b{budget:g}"
                if (budget in (row.adapt_steps or {})
                        and supplement.exists("fewshot", row.run_id, ledger_path=ledger_path, part=part)):
                    continue
                if parent is None:
                    parent = open_run(row, run_dir_of)
                    _final_checkpoint(row, parent)
                specs = _fewshot_specs(parent.spec, budget)
                why = _continuation_waits(specs[0].run_id, run_dir_of, status_of)
                if why is not None:
                    not_ready[specs[0].run_id] = why
                    continue
                final = parent.spec.total_steps
                cont = _open_continuation(row, specs[0].run_id, specs, run_dir_of,
                                          parent_checkpoint=_parent_checkpoint(parent, final), parent_step=final)
                result = evaluate(str(cont.omnisafe_dir), cont.spec.to_dict())
                if check_code is not None:
                    check_code()
                record, seed_set, seeds, sr, steps, censored = fewshot_record(row, cont, budget, result, commit)
                if censored:  # only a budget that never reaches the target depends on how it is recorded
                    try:
                        require_answered("Q-adapt-censoring", what=f"{row.run_id}: adapt_steps at budget {budget:g}, which "
                                                                   f"never reaches {R.SATISFACTION_TARGET} (recorded as {steps}, "
                                                                   "the largest horizon + 1)")
                    except PendingQuestionError as exc:
                        waiting.items[f"{row.run_id}/{part}"] = str(exc)
                        # The rates do not depend on the answer: written now, after the record is validated
                        # (the schema recomputes each rate from its episodes) and the seeds checked; an
                        # existing key must be reproduced exactly. adapt_steps and the record, which carries it, wait.
                        _validated(record)
                        if _commit(ledger_path, row.run_id, {"sr_fewshot": sr}, seeds=[(seed_set, seeds)]) \
                                and row.run_id not in done:
                            done.append(row.run_id)
                        continue
                _commit(ledger_path, row.run_id, {"sr_fewshot": sr, "adapt_steps": {budget: steps}},
                        records=[record], seeds=[(seed_set, seeds)])
                if row.run_id not in done:
                    done.append(row.run_id)
    finally:
        _log(ledger_path, "fewshot", done, {"waiting": waiting.items, "not_ready": not_ready,
                                            "data_root": _abs(data_root)}, commit=commit)
    waiting.refuse("fewshot", done, not_ready)
    return done


# ---------------------------------------------------------------------------
# Part 4.1 rule 7: the final checkpoint (b) and tolerance 5.0 (a)
# ---------------------------------------------------------------------------


def apply_final_battery(
    ledger_path: Path,
    run_dir_of: Callable[[str], Path],
    conditions: Iterable[str] = ("hazard",),
    *,
    evaluator: Callable[..., dict[str, Any]] | None = None,
    check_code: Callable[[], None] | None = None,
    data_root: str | Path | None = None,
) -> Done:
    """``enrich final-battery``: Part 4.1 rule 7 (b) at the final checkpoint of every completed Study A row.

    Part 4.1.1: "The difference in robustness gap at the end of training, whatever the
    in-distribution cost" (no matching filter). ``evaluate_battery`` at the final step with the
    conditions still missing (hazard by default; dynamics on request); the ``final_battery`` records
    ``measurement`` (C_ID of the final checkpoint) and one per condition. The final checkpoint's
    measurement-set cost is the ledger's final_cost when evaluation.json says it was measured on
    that set (Q-final-cost-set), and must then equal it exactly. The main study only; no ledger field.
    Each condition's recorded definition is logged (``definitions``).
    """
    ledger_path = Path(ledger_path)
    conditions = _conditions(conditions, CONDITION_FIELDS, "conditions")
    rows = _rows(ledger_path)
    _refuse_pilot_rows(rows, ledger_path, "enrich final-battery")
    evaluate = evaluator or _load(BATTERY_TARGET)
    commit = _code_commit()
    done = Done()
    definitions: dict[str, dict[str, Any]] = {}  # the harness's recorded definitions, per run and condition
    try:
        for row in rows:
            if row.study != "A" or not row.completed:
                continue
            missing = [c for c in conditions
                       if not supplement.exists("final_battery", row.run_id, ledger_path=ledger_path, part=c)]
            existing = supplement.read("final_battery", row.run_id, ledger_path=ledger_path, part="measurement")
            if not missing and existing is not None:
                continue
            files = open_run(row, run_dir_of)
            final = _final_checkpoint(row, files)
            costs = evaluate(str(files.omnisafe_dir), files.spec.to_dict(), final, missing)
            if check_code is not None:
                check_code()
            result = check_battery_result(row.run_id, costs, step=final, conditions=missing,
                                          selection_seeds=_selection_seeds(files))
            if existing is not None and not _same_episodes(existing.episodes, result.measurement):
                raise ContractError(f"{row.run_id}: the final checkpoint's measurement episodes differ from those recorded; "
                                    "evaluation is not deterministic")
            if files.evaluation().get("final_seed_set") == "measurement" and result.c_id != row.final_cost:
                raise ContractError(f"{row.run_id}: the final checkpoint's measurement-set cost {result.c_id!r} is not the "
                                    f"ledger's final_cost {row.final_cost!r} (the same episodes; Q-final-cost-set)")
            records = [result.condition_record("final_battery", row.run_id, commit, c, result.c_id) for c in missing]
            if existing is None:
                records.append(_Record("final_battery", row.run_id, {
                    "kind": "final_battery", "run_id": row.run_id, "code_commit": commit, "step": final,
                    "condition": "measurement", "episodes": result.measurement, "hazard": None,
                    "measurement_cost": None, "gap": None}, "measurement"))
            _commit(ledger_path, row.run_id, records=records, seeds=result.seed_sets())
            done.append(row.run_id)
            definitions[row.run_id] = result.definitions()
    finally:
        _log(ledger_path, "final-battery", done, {"conditions": conditions,
                                                  "definitions": _definitions_entry(definitions),
                                                  "data_root": _abs(data_root)}, commit=commit)
    return done


def _flagged(readiness: ArmReadiness, rows: Mapping[str, Mapping[str, Any]]) -> ArmReadiness:
    """The ready arms whose flags ``enrich match`` has written, per task whose reference is flagged."""
    out = ArmReadiness(waiting=dict(readiness.waiting))
    for task, task_rows in readiness.ready.items():
        arms: dict[str, list[Mapping[str, Any]]] = {}
        for r in task_rows:
            arms.setdefault(_arm_id(r), []).append(r)
        order = list(arms)  # the reference first (arm_readiness)
        flagged = [a for a in order if all(rows[str(r["run_id"])].get("matched") is not None for r in arms[a])]
        if order[0] not in flagged:
            for arm in order:
                out.waiting[arm] = ("no flags yet (`enrich match`)" if arm == order[0]
                                    else f"the reference arm {order[0]} has no flags yet (`enrich match`)")
            continue
        for arm in order:
            if arm not in flagged:
                out.waiting[arm] = "no flags yet (`enrich match`)"
        out.ready[task] = [r for a in flagged for r in arms[a]]
        out.targets.update({a: readiness.targets[a] for a in flagged})
    return out


def apply_sensitivity_battery(
    ledger_path: Path,
    run_dir_of: Callable[[str], Path],
    conditions: Iterable[str] = ("hazard",),
    *,
    surplus_extra: int,
    stored_surplus_extra: Any = None,
    matcher: Callable[..., Mapping[str, Any]] | None = None,
    evaluator: Callable[..., dict[str, Any]] | None = None,
    check_code: Callable[[], None] | None = None,
    data_root: str | Path | None = None,
) -> Done:
    """``enrich sensitivity-battery``: Part 4.1 rule 7 (a), "once with tolerance 5.0".

    The arms matched at ``R.SENSITIVITY_TOLERANCE`` (``match_arms``, re-checked by ``rule_matching``;
    the tolerance applies to rule 3 as to rule 5) but not at 2.5 (the ledger's ``matched`` False),
    among the arms ``enrich match`` has flagged, get the battery at their matched checkpoint as
    ``sensitivity_battery`` records (hazard by default). Their ledger gap fields stay empty: rule 6
    leaves an unmatched arm out of the battery. The measurement cost must reproduce the ledger's.
    The arms not flagged yet are returned in ``Done.waiting``. The tolerance-5.0 flags follow the
    answer of Q-threshold-arithmetic (Table 9.1), a report key: while it is open, the arms whose flags another
    of its readings would change are evaluated like the others and named in ``Done.provisional`` and
    in the log (``_flags``). Each condition's recorded definition (``result["conditions"]``) goes to the
    log (``definitions``).
    """
    ledger_path = Path(ledger_path)
    conditions = _conditions(conditions, CONDITION_FIELDS, "conditions")
    _check_surplus(surplus_extra, stored_surplus_extra)
    rows = _rows(ledger_path)
    _refuse_pilot_rows(rows, ledger_path, "enrich sensitivity-battery")
    match = matcher or _load(MATCH_TARGET)
    evaluate = evaluator or _load(BATTERY_TARGET)
    commit = _code_commit()
    done = Done()
    definitions: dict[str, dict[str, Any]] = {}  # the harness's recorded definitions, per run and condition
    try:
        dumps = {r.run_id: r.model_dump(mode="python") for r in rows}
        readiness = _flagged(arm_readiness(list(dumps.values()), surplus_extra), dumps)
        done.waiting.update(readiness.waiting)
        flags, done.provisional = _flags(readiness, R.SENSITIVITY_TOLERANCE, match, check_code)
        for row in rows:
            if row.matched is not False or flags.get(row.run_id, {}).get("matched") is not True:
                continue
            missing = [c for c in conditions
                       if not supplement.exists("sensitivity_battery", row.run_id, ledger_path=ledger_path, part=c)]
            if not missing:
                continue
            files = _open_matched(row, run_dir_of)
            step = int(row.matched_checkpoint_step)
            costs = evaluate(str(files.omnisafe_dir), files.spec.to_dict(), step, missing)
            if check_code is not None:
                check_code()
            result = check_battery_result(row.run_id, costs, step=step, conditions=missing,
                                          selection_seeds=_selection_seeds(files))
            if result.c_id != row.measurement_cost:
                raise ContractError(f"{row.run_id}: the measurement cost {result.c_id!r} is not the ledger's "
                                    f"{row.measurement_cost!r}; evaluation is not deterministic")
            records = [result.condition_record("sensitivity_battery", row.run_id, commit, c, row.measurement_cost)
                       for c in missing]
            if not supplement.exists("measurement", row.run_id, ledger_path=ledger_path):
                records.append(_measurement_record(row.run_id, commit, result))
            _commit(ledger_path, row.run_id, records=records, seeds=result.seed_sets())
            done.append(row.run_id)
            definitions[row.run_id] = result.definitions()
    finally:
        _log(ledger_path, "sensitivity-battery", done, {"conditions": conditions, "surplus_extra": surplus_extra,
                                                        "arms_waiting": done.waiting, "provisional": done.provisional,
                                                        "definitions": _definitions_entry(definitions),
                                                        "data_root": _abs(data_root)}, commit=commit)
    return done


# ---------------------------------------------------------------------------
# Read-only reports: the scheduler's state, completeness, freeze (HANDOVER.md section 4, tasks 33 and 36; Part 5.8)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SchedulerState:
    """A data root's scheduler state, read without writing (``state.sqlite`` opened read-only)."""

    statuses: dict[str, str]
    specs: dict[str, RunSpec]
    settings: dict[str, str]

    def status_of(self, run_id: str) -> str | None:
        return self.statuses.get(run_id)


def read_scheduler_state(data_root: str | Path) -> SchedulerState | None:
    """The scheduler state of ``data_root``, or None when it has none (nothing is created).

    ContractError, naming the file, when it cannot be read as the scheduler's state (not a database,
    e.g. a truncated copy; locked by a writer for more than 60 s; tables missing).
    """
    import sqlite3

    from pilot.scheduler import SchedulerConfig

    path = SchedulerConfig(data_root=Path(data_root)).state_path
    if not path.is_file():
        return None
    try:
        con = sqlite3.connect(f"{Path(os.path.abspath(path)).as_uri()}?mode=ro", uri=True, timeout=60.0)
        try:
            runs = con.execute("SELECT run_id, status, spec FROM runs ORDER BY rowid").fetchall()
            settings = dict(con.execute("SELECT key, value FROM settings").fetchall())
        finally:
            con.close()
    except sqlite3.Error as exc:
        raise ContractError(f"cannot read the scheduler state {path} ({type(exc).__name__}: {exc}); "
                            "is --data-root a data root of `python -m pilot schedule`?") from None
    return SchedulerState({r[0]: r[1] for r in runs}, {r[0]: RunSpec.from_json(r[2]) for r in runs}, settings)


def expected_records(row: Any) -> list[tuple[str, str | None]]:
    """The supplement records (kind, part) a ledger row should have once enriched (results/supplement_schema.py).

    Every completed row: ``evaluation``. Study A: ``training``; with a matched checkpoint,
    ``measurement``; in the battery (the pilot's rows, or ``matched is True``), ``battery`` per
    condition of Part 3.6 / Table 2.2; in the main study, ``final_battery`` (measurement, hazard) and,
    when matched, ``continuation`` per condition. Study B of the main study: ``zero_shot`` and
    ``fewshot`` per unseen budget. (``sensitivity_battery`` depends on the tolerance-5.0 matching and
    is not listed.) An auxiliary row (``manifest.AUXILIARY_NOTE``) is never matched (``arm_readiness``).
    """
    if not row.completed:
        return []
    out: list[tuple[str, str | None]] = [("evaluation", None)]
    pilot = _is_pilot(row.run_id)
    if row.study == "A":
        out.append(("training", None))
        if row.matched_checkpoint_step is not None:
            out.append(("measurement", None))
        if pilot or row.matched is True:
            out += [("battery", c) for c in CONDITION_FIELDS]
        if not pilot:
            out += [("final_battery", "measurement"), ("final_battery", "hazard")]
            # an auxiliary run (manifest.AUXILIARY_NOTE) is never matched: no battery or continuation records
            if row.matched is True and not manifest.is_auxiliary_row(row.model_dump()):
                out += [("continuation", c) for c in CONTINUATION_FIELDS]
    elif row.study == "B" and row.arm in R.STUDY_B_ARMS and not pilot:
        out.append(("zero_shot", None))
        out += [("fewshot", f"b{float(b):g}") for b in R.UNSEEN_BUDGETS]
    return out


def expected_fields(row: Any) -> list[str]:
    """The enrichment fields a completed ledger row should hold once enriched (Appendix B)."""
    if not row.completed:
        return []
    pilot = _is_pilot(row.run_id)
    if row.study == "A":
        fields = ["matched_checkpoint_step", "measurement_cost"]
        if any(c.multiplier is not None for c in row.checkpoints):
            fields.append("lambda_final")
        if pilot:
            fields += list(CONDITION_FIELDS.values())
        elif manifest.is_auxiliary_row(row.model_dump()):
            pass  # an auxiliary run is left out of its arm (arm_readiness): `enrich match` never flags it
        else:
            fields += ["matched", "infeasible"]
            if row.matched is True:
                fields += [*CONDITION_FIELDS.values(), *CONTINUATION_FIELDS.values()]
        return fields
    if row.study == "B" and row.arm in R.STUDY_B_ARMS and not pilot:
        return ["sr_zero", "sr_fewshot", "adapt_steps"]
    return []


def _missing_field(row: Any, name: str) -> bool:
    value = getattr(row, name)
    if value is None:
        return True
    if name in ("sr_zero", "adapt_steps"):
        return sorted(float(k) for k in value) != sorted(float(b) for b in R.UNSEEN_BUDGETS)
    return False


def completeness_report(ledger_path: Path, state: SchedulerState | None = None) -> dict[str, Any]:
    """Per arm: runs queued and their statuses, ledger rows, enrichment fields still missing and
    supplement records still missing (HANDOVER.md section 4, task 33; the operator's tool before the freeze).

    Read-only: the ledger, its supplement and (``state``) the data root's scheduler state are read,
    nothing is written. Continuations are arms of their own in the scheduler's counts.
    """
    from pilot.scheduler import DONE_TRAINING

    ledger_path = Path(ledger_path)
    rows = _rows(ledger_path) if ledger_path.exists() else []
    arms: dict[str, dict[str, Any]] = {}

    def arm(key: str) -> dict[str, Any]:
        return arms.setdefault(key, {"queued": 0, "trained": 0, "ledgered": 0, "by_status": {}, "ledger_rows": 0,
                                     "completed": 0, "fields_missing": {}, "records_missing": {}})

    for run_id, spec in (state.specs.items() if state else ()):
        entry = arm(spec.arm_id)
        entry["queued"] += 1
        status = state.statuses[run_id]
        entry["by_status"][status] = entry["by_status"].get(status, 0) + 1
        entry["trained"] += int(status in DONE_TRAINING)  # training finished (continuations: "continued")
        entry["ledgered"] += int(status == "ledgered")
    for row in rows:
        entry = arm(_arm_id(row.model_dump(mode="python")))
        entry["ledger_rows"] += 1
        entry["completed"] += int(bool(row.completed))
        for name in expected_fields(row):
            if _missing_field(row, name):
                entry["fields_missing"].setdefault(name, []).append(row.run_id)
        for kind, part in expected_records(row):
            if not supplement.exists(kind, row.run_id, ledger_path=ledger_path, part=part):
                entry["records_missing"].setdefault(kind if part is None else f"{kind}/{part}", []).append(row.run_id)
    totals = {
        "arms": len(arms), "queued": sum(a["queued"] for a in arms.values()), "ledger_rows": len(rows),
        "fields_missing": sum(len(v) for a in arms.values() for v in a["fields_missing"].values()),
        "records_missing": sum(len(v) for a in arms.values() for v in a["records_missing"].values()),
        "not_finished": sum(n for a in arms.values() for s, n in a["by_status"].items()
                            if s not in ("ledgered", "continued", "excluded", "superseded")),
    }
    return {"ledger": str(ledger_path), "scheduler_state": state is not None, "totals": totals,
            "arms": dict(sorted(arms.items())),
            "notes": ["sensitivity_battery records (Part 4.1 rule 7 (a)) depend on the tolerance-5.0 matching and are "
                      "not counted here; run `enrich sensitivity-battery`"]}


def _repo_rel(path: Path) -> str:
    try:
        return Path(os.path.abspath(path)).relative_to(provenance.REPO_ROOT).as_posix()
    except ValueError:
        return str(Path(os.path.abspath(path)))


def read_working_file(path: Path) -> bytes | None:
    """The file as it is in the working tree (None: absent)."""
    try:
        return Path(path).read_bytes()
    except OSError:
        return None


def list_go_reports(directory: Path, *, read: Callable[[Path], bytes | None],
                    also: Iterable[Path] = ()) -> list[dict[str, Any]]:
    """Every ``go_report-<UTC>.json`` of ``directory``, oldest first: ``file``, ``path`` (repository-relative), the
    ``commit`` (None unless a full commit hash) and pilot ``revision`` (0, or 1 for the re-pilot) it records, read
    through ``read`` (None for a file ``read`` does not give, e.g. one not committed at HEAD), and ``readable``.
    ``also`` adds reports that are not in the working tree (``freeze_report``: those committed at HEAD)."""
    out: list[dict[str, Any]] = []
    listed = {Path(os.path.abspath(p)) for p in Path(directory).glob("go_report-*.json")}
    listed |= {Path(os.path.abspath(p)) for p in also if Path(p).match("go_report-*.json")}
    for path in sorted(listed, key=lambda p: (p.name, str(p))):
        content = read(path)
        try:
            generated = json.loads(content).get("generated", {}) if content is not None else None
        except (ValueError, AttributeError):
            generated = None
        generated = generated if isinstance(generated, dict) and generated else None
        commit = generated.get("commit") if generated is not None else None
        revision = generated.get("revision") if generated is not None else None
        if generated is not None and not isinstance(revision, int):  # reports written before the revision was recorded
            argv = generated.get("argv")
            revision = 1 if isinstance(argv, list) and "--repilot" in argv else 0
        out.append({"file": str(path), "path": _repo_rel(path),
                    "commit": commit if provenance.is_full_commit_hash(commit) else None,
                    "revision": revision, "readable": generated is not None})
    return out


def decision_go_report(reports: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The report the go decision was taken from, for Part 5.8's freeze: of the last pilot revision, the EARLIEST
    readable report (`go` may be re-run at any time, and a later report must never move the
    baseline past edits to analysis/; a later report of the same revision only adds to the changes listed)."""
    readable = [r for r in reports if r["readable"] and isinstance(r["revision"], int)]
    if not readable:
        return None
    last = max(r["revision"] for r in readable)
    return next(r for r in readable if r["revision"] == last)


def analysis_changes_since(commit: str) -> list[str]:
    """``git log --format='%H %s' --end-of-options <commit>..HEAD -- analysis/ <IMPORTED_CODE>`` (read-only): each change
    to the analysis code after the go decision, to be matched with a Table 9.1 erratum (Part 5.8). The analysis code
    is analysis/ and every in-repository module it imports (``analysis.records.IMPORTED_CODE``, the closure its
    ``code_hash`` covers; Q-exploratory-labels, Table 9.1). ``commit`` must be a full commit hash, so that git never
    reads it as an option."""
    import subprocess

    from analysis.records import IMPORTED_CODE

    if not provenance.is_full_commit_hash(commit):
        raise ContractError(f"the go report's commit {commit!r} is not a full commit hash")
    out = subprocess.run(["git", "log", "--format=%H %s", "--end-of-options", f"{commit}..HEAD", "--", "analysis/",
                          *IMPORTED_CODE], cwd=provenance.REPO_ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        raise ContractError(f"git log {commit}..HEAD failed: {out.stderr.strip()}")
    return [line for line in out.stdout.splitlines() if line.strip()]


def freeze_report(ledger_path: Path, *, state: SchedulerState | None = None, data_root: str | Path | None = None,
                  go_reports: Path | None = None, analysis_outputs: Path | None = None,
                  read_go_report: Callable[[Path], bytes | None] | None = None) -> dict[str, Any]:
    """A manifest of the data as it stands (HANDOVER.md section 4, task 36; Part 5.8), read-only: nothing is written
    or tagged.

    SHA-256 of the ledger, its seed registry and enrichment log, every supplement record, every file
    under the analysis outputs (``results/analysis``), the analysis code (``analysis/`` and the modules
    it imports, ``analysis.records.IMPORTED_CODE``; Q-exploratory-labels) and the data root's
    scheduler state; HEAD and the uncommitted tracked files; every go report committed at HEAD (read
    through ``read_go_report``, by default from HEAD), the one the decision was taken from
    (``decision_go_report``: the earliest of the last pilot revision) and every commit that changed
    the analysis code since the one it records (``analysis_changes_since``; Part 5.8: "not edited
    between the go decision and the final analysis except to fix errors, each of which is recorded
    in Part 9"); and the completeness totals.
    ``ready`` is True when nothing is missing, nothing is unfinished and the tree is clean; ``problems``
    says why not.
    """
    from analysis.records import IMPORTED_CODE
    from pilot.scheduler import SchedulerConfig

    ledger_path = Path(ledger_path)
    go_reports_dir = Path(go_reports) if go_reports is not None else provenance.REPO_ROOT / "results" / "pilot"
    analysis_outputs = Path(analysis_outputs) if analysis_outputs is not None else provenance.REPO_ROOT / "results" / "analysis"
    files: dict[str, str] = {}
    candidates = [ledger_path, seeds_registry(ledger_path), enrichment_log_path(ledger_path), *supplement.files(ledger_path)]
    candidates += sorted(p for p in analysis_outputs.rglob("*") if p.is_file()) if analysis_outputs.is_dir() else []
    code = provenance.REPO_ROOT / "analysis"
    candidates += sorted(p for p in code.rglob("*") if p.is_file() and p.suffix in (".py", ".json")
                         and "__pycache__" not in p.parts)
    candidates += [provenance.REPO_ROOT / rel for rel in IMPORTED_CODE]  # the modules the analysis imports
    if data_root is not None:
        candidates.append(SchedulerConfig(data_root=Path(data_root)).state_path)
    for path in candidates:
        if path.is_file():
            files[_repo_rel(path)] = _sha256(path)
    problems: list[str] = []
    head = provenance.commit_hash()
    dirty = provenance.dirty_paths()
    if dirty:
        problems.append(f"uncommitted tracked files: {dirty[:5]}")
    if not ledger_path.exists():
        problems.append(f"{ledger_path} does not exist")

    def _read_committed(path: Path) -> bytes | None:  # the reports committed at HEAD, never a working-tree file
        rel = _repo_rel(path)
        return None if Path(rel).is_absolute() else provenance.file_committed_at(head, rel)

    read = read_go_report or _read_committed
    committed_reports: list[Path] = []
    go_rel = _repo_rel(go_reports_dir)
    if not Path(go_rel).is_absolute():  # every report committed at HEAD counts, even one deleted from the working
        # tree (an output path: the deletion does not make the tree dirty)
        from pilot.scheduler import committed_files

        committed_reports = [provenance.REPO_ROOT / rel for rel in committed_files(head, go_rel)
                             if rel.rpartition("/")[0] == go_rel.rstrip("/")]
        for path in committed_reports:
            if path.match("go_report-*.json") and not path.is_file():
                problems.append(f"{_repo_rel(path)} is committed at HEAD but missing from the working tree: restore it "
                                "(`git checkout -- <file>`)")
    reports = list_go_reports(go_reports_dir, read=read, also=committed_reports)
    for report in reports:
        if not report["readable"]:
            problems.append(f"{report['path']} is not committed at HEAD (or unreadable): the go decision is read "
                            "from committed reports only")
    go = decision_go_report(reports)
    changes: list[str] | None = None
    if go is None:
        problems.append(f"no committed go report under {go_reports_dir}")
    elif go["commit"] is None:
        problems.append(f"{go['path']} records no commit")
    else:
        try:
            changes = analysis_changes_since(go["commit"])
        except ContractError as exc:  # e.g. the go report's commit is not in this clone's history
            problems.append(str(exc))
    completeness = completeness_report(ledger_path, state) if ledger_path.exists() else None
    if completeness is not None:
        totals = completeness["totals"]
        if totals["fields_missing"] or totals["records_missing"]:
            problems.append(f"{totals['fields_missing']} enrichment fields and {totals['records_missing']} supplement "
                            "records are missing (`python -m pilot completeness`)")
        if totals["not_finished"]:
            problems.append(f"{totals['not_finished']} scheduled runs have not finished")
    if state is None:
        problems.append("no scheduler state was read (pass the data root)")
    return {
        "head": head, "worktree_dirty": dirty, "ledger": str(ledger_path), "files": files,
        "go_report": None if go is None else {k: go[k] for k in ("path", "commit", "revision")},
        "go_reports": [{k: r[k] for k in ("path", "commit", "revision", "readable")} for r in reports],
        "analysis_changes_since_go": changes,
        "completeness": None if completeness is None else completeness["totals"],
        "ready": not problems, "problems": problems,
    }

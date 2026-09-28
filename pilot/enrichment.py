"""Enrichment of ledger rows after a run is recorded (ledger memo, decision 3).

The pilot owner remains the only writer; the rules come from the owners of each rule, and every
result is checked at the boundary before it is written:

1. ``apply_selection``: Part 4.1 rule 1 (per run, independent of other arms). For each completed
   Study A run, the checkpoint among its last ten whose selection-set cost is closest to d = 25 is
   chosen by the Analysis and results owner's function (contract 4 in pilot/contracts.py)
       ``analysis.matching.select_checkpoint(checkpoints: list[dict]) -> int``  (the chosen step)
   and verified here against rule 1 computed independently (ties: the later checkpoint, as First
   Tasks proposes; distances to 25 are compared after rounding to 1e-4, so costs such as 32.01 and
   17.99 tie despite floating-point noise, also from a float32 harness; a non-finite cost is
   refused). It is written as matched_checkpoint_path, matched_checkpoint_step, training_age,
   selection_cost_at_match and lambda_at_selection (the multiplier recorded with that checkpoint;
   ledger memo decision 6).

2. ``apply_battery``: Table 2.2 / Part 3.4. The chosen checkpoint is evaluated on the measurement
   set and under the requested battery conditions by the Environment and tests owner's harness
   (contract 5)
       ``envs.evaluation.evaluate_battery(omnisafe_dir: str, spec: dict, step: int,
                                          conditions: list[str]) -> dict``
   returning {"measurement": cost, "<condition>": cost, ..., "episodes": 100,
   "measurement_seeds": [100 seeds]}. The measurement seeds must be disjoint from the run's
   selection seeds and the same for every run (Table 2.1). Robustness gaps are equation (1):
   gap = C_cond - C_ID, with C_ID the measurement-set cost of the same checkpoint. The run
   directory must be the one the row describes (same commit, same matched checkpoint path), so a
   wrong data root, such as a smoke root with the same run_ids, is refused.

Every write takes the ledger's lock and is atomic (``ledger_writer.ledger_transaction``), so an
enrichment can run while the scheduler appends rows. Inside the transaction every row of the
working copy is validated against ``LedgerRow`` before the ledger is replaced: the schema's
``write_enrichment`` checks only the Parquet types, and a row that breaks a constraint would make
the whole ledger unloadable. The arm-level flags ``matched`` and ``infeasible`` (Part 4.1 rules 2
to 6) are the analysis owner's and are not written here.
"""

from __future__ import annotations

import copy
import importlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

from configs import registered as R
from pilot import provenance
from pilot.contracts import ContractError, check_canonical_seeds, validate_seed_set
from pilot.launch import EVALUATION_RESULT, SPEC_FILE, TRAIN_RESULT
from pilot.ledger_writer import ledger_transaction
from pilot.manifest import RunSpec

CONDITION_FIELDS = {"hazard": "gap_hazard", "dynamics": "gap_dynamics"}
TIE_DECIMALS = 4  # rule 1 distances are compared at this precision: costs are means of 100 episodes of
# 0/1 per-step indicators (multiples of 0.01), so 1e-4 absorbs float32 and float64 noise alike
SELECT_TARGET = ("analysis.matching", "select_checkpoint", "Analysis and results (Role 4)")
BATTERY_TARGET = ("envs.evaluation", "evaluate_battery", "Environment and tests (Role 2)")


def _load(target: tuple[str, str, str]) -> Callable[..., Any]:
    from pilot.algorithms import PluginUnavailableError

    module_name, func, owner = target
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if missing and (module_name == missing or module_name.startswith(missing + ".")):
            raise PluginUnavailableError(f"{module_name}.{func} is provided by {owner} and is not in the repository yet") from exc
        raise  # the module exists but one of its own imports failed: show the real error
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


def rule_one_step(checkpoints: list[dict]) -> int:
    """Part 4.1 rule 1, computed independently: among the last ten checkpoints, the one whose
    selection-set cost is closest to d = 25; a tie goes to the later checkpoint (First Tasks Table 9)."""
    window = sorted(checkpoints, key=lambda c: c["step"])[-R.SELECTION_WINDOW_CHECKPOINTS:]
    if any(c.get("selection_cost") is None for c in window):
        raise ContractError("a checkpoint of the selection window has no selection-set cost")
    if not all(math.isfinite(float(c["selection_cost"])) for c in window):
        raise ContractError("a selection-set cost of the window is not finite; rule 1 cannot rank it")
    return min(
        window, key=lambda c: (round(abs(float(c["selection_cost"]) - R.COST_LIMIT), TIE_DECIMALS), -c["step"])
    )["step"]


def selection_fields(row_data: dict[str, Any], chosen_step: int) -> dict[str, Any]:
    """Ledger fields describing the chosen checkpoint (Appendix B), after checking rule 1."""
    window = sorted(c["step"] for c in row_data["checkpoints"])[-R.SELECTION_WINDOW_CHECKPOINTS:]
    if chosen_step not in window:
        raise ContractError(f"{row_data['run_id']}: chosen step {chosen_step} is not among the last ten checkpoints {window}")
    expected = rule_one_step(row_data["checkpoints"])
    if chosen_step != expected:
        raise ContractError(
            f"{row_data['run_id']}: the selector chose {chosen_step}, but rule 1 gives {expected} "
            "(closest selection-set cost to 25; ties to the later checkpoint)"
        )
    record = next(c for c in row_data["checkpoints"] if c["step"] == chosen_step)
    return {
        "matched_checkpoint_path": record["path"],
        "matched_checkpoint_step": chosen_step,
        "training_age": chosen_step,
        "selection_cost_at_match": record["selection_cost"],
        "lambda_at_selection": record.get("multiplier"),
    }


def _log(ledger_path: Path, action: str, run_ids: list[str], extra: dict[str, Any] | None = None) -> None:
    """Append the provenance of an enrichment (commit, time, runs) beside the ledger."""
    if not run_ids:
        return
    log = Path(ledger_path).with_name(Path(ledger_path).stem + ".enrichment_log.jsonl")
    try:
        commit: Any = provenance.commit_hash()
        dirty: Any = bool(provenance.dirty_paths())
    except Exception:  # noqa: BLE001 - called from ``finally``: never hide the enrichment's own error
        commit = dirty = "unavailable"
    entry = {"utc": provenance.utc_now().isoformat(), "action": action, "commit": commit,
             "worktree_dirty": dirty, "run_ids": run_ids, **(extra or {})}
    try:
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, sort_keys=True, default=str) + "\n")
    except OSError as exc:  # called from ``finally``: never replace the enrichment's own error
        print(f"warning: enrichment log {log} not written ({exc}); record it by hand: {json.dumps(entry, default=str)}",
              file=sys.stderr)


def _write(ledger_path: Path, run_id: str, fields: dict[str, Any], check: Callable[[], None] | None = None) -> None:
    """Write ``fields`` to one row; ``check`` runs under the same lock, after the row validated.

    ``write_enrichment`` checks only the Parquet types, so every row of the working copy is loaded
    as a ``LedgerRow`` (the schema's constraints) before the ledger is replaced.
    """
    from results.ledger_schema import load_ledger_as_rows, write_enrichment

    with ledger_transaction(ledger_path) as tmp:
        write_enrichment(run_id, fields, tmp)
        load_ledger_as_rows(tmp)  # raises (and the ledger stays as it was) if any row breaks the schema
        if check is not None:
            check()


def apply_selection(
    ledger_path: Path,
    *,
    selector: Callable[[list[dict]], int] | None = None,
    check_code: Callable[[], None] | None = None,
) -> list[str]:
    """Write rule-1 selections for completed Study A runs that have none yet. Returns run_ids done.

    ``check_code`` runs after each call of the selector and before its result is written (the
    command line refuses code that the commit does not contain; ``provenance``).
    """
    from results.ledger_schema import load_ledger_as_rows

    select = selector or _load(SELECT_TARGET)
    done: list[str] = []
    try:
        for row in load_ledger_as_rows(ledger_path):
            if row.study != "A" or not row.completed or row.matched_checkpoint_step is not None:
                continue
            data = row.model_dump(mode="python")
            step = int(select(copy.deepcopy(data["checkpoints"])))  # the selector cannot alter what is checked
            if check_code is not None:
                check_code()
            _write(ledger_path, row.run_id, selection_fields(data, step))
            done.append(row.run_id)
    finally:
        _log(ledger_path, "select", done)  # also the runs written before a failure
    return done


def gaps_from_costs(costs: dict[str, Any], conditions: Iterable[str]) -> dict[str, float]:
    """Equation (1): gap = C_cond - C_ID for every requested condition."""
    if int(costs.get("episodes", -1)) != R.EVAL_EPISODES:
        raise ContractError(f"battery evaluation must use {R.EVAL_EPISODES} episodes per condition")
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

    return relative_checkpoint_path(os.path.abspath(Path(omnisafe_dir) / "torch_save" / f"epoch-{step // R.STEPS_PER_EPOCH}.pt"))


def apply_battery(
    ledger_path: Path,
    run_dir_of: Callable[[str], Path],
    conditions: Iterable[str],
    *,
    evaluator: Callable[..., dict[str, Any]] | None = None,
    check_code: Callable[[], None] | None = None,
    data_root: str | Path | None = None,
) -> list[str]:
    """Evaluate each selected checkpoint that lacks a requested gap, and write its gaps.

    The spec of each run is read from its own run directory (``spec.json``), so replacement runs
    are handled like any other. The run directory must hold the run the row records: the same
    training commit, and the matched checkpoint at the path the row stores (otherwise, for
    example, a smoke data root with the same run_ids would be evaluated). Evaluation is
    deterministic, so a measurement cost already in the ledger must be reproduced exactly when a
    missing condition is added later. ``check_code`` runs after each evaluation and before its
    result is written, as in ``apply_selection``. ``data_root`` is recorded in the enrichment log.
    """
    from results.ledger_schema import load_ledger_as_rows

    conditions = list(conditions)
    for condition in conditions:
        _check_condition(condition)
    evaluate = evaluator or _load(BATTERY_TARGET)
    registry = Path(ledger_path).with_name(Path(ledger_path).stem + ".seeds.json")
    done: list[str] = []
    evaluated: dict[str, list[str]] = {}  # the conditions actually evaluated and written, per run
    try:
        for row in load_ledger_as_rows(ledger_path):
            if row.study != "A" or row.matched_checkpoint_step is None:
                continue
            missing = [c for c in conditions if getattr(row, CONDITION_FIELDS[c]) is None]
            if not missing:
                continue
            run_dir = Path(run_dir_of(row.run_id))
            spec = RunSpec.from_json((run_dir / SPEC_FILE).read_text(encoding="utf-8"))
            if spec.run_id != row.run_id:
                raise ContractError(f"{run_dir} holds the spec of {spec.run_id}, not {row.run_id}")
            train = json.loads((run_dir / TRAIN_RESULT).read_text(encoding="utf-8"))
            check_same_commit(row.run_id, run_dir, train, row.commit_hash)
            evaluation = json.loads((run_dir / EVALUATION_RESULT).read_text(encoding="utf-8"))
            omnisafe_dir = run_dir / train["omnisafe_dir"]
            path = matched_checkpoint_path(omnisafe_dir, int(row.matched_checkpoint_step))
            if path != row.matched_checkpoint_path:
                raise ContractError(
                    f"{row.run_id}: {run_dir} gives the checkpoint path {path!r}, but the ledger's matched checkpoint is "
                    f"{row.matched_checkpoint_path!r}; use the data root the ledger was written from"
                )
            costs = evaluate(str(omnisafe_dir), spec.to_dict(), int(row.matched_checkpoint_step), missing)
            if check_code is not None:
                check_code()
            measurement_seeds = validate_seed_set("measurement_seeds", costs.get("measurement_seeds"))
            if set(measurement_seeds) & set(int(s) for s in evaluation["selection_seeds"]):
                raise ContractError(f"{row.run_id}: measurement and selection seeds overlap (Table 2.1 requires disjoint sets)")
            fields = gaps_from_costs(costs, missing)
            if row.measurement_cost is not None and fields["measurement_cost"] != row.measurement_cost:
                raise ContractError(f"{row.run_id}: the measurement cost changed between evaluations; evaluation is not deterministic")
            # The seed registry is fixed only once the row has validated, under the ledger's lock.
            _write(ledger_path, row.run_id, fields,
                   check=lambda seeds=measurement_seeds: check_canonical_seeds("measurement", seeds, registry))
            done.append(row.run_id)
            evaluated[row.run_id] = missing
    finally:
        _log(ledger_path, "battery", done,
             {"conditions": conditions, "evaluated": evaluated,
              "data_root": None if data_root is None else os.path.abspath(data_root)})
    return done

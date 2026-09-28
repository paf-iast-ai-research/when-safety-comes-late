"""Interfaces between the pilot owner's pipeline and the other roles' code.

The scheduler and ledger writer (Role 1) call code owned by other roles through the functions
named here. Each contract is checked when its result comes back, so a mistake in one component is
caught at the boundary instead of reaching the ledger.

1. Evaluation harness, owned by Environment and tests (Role 2):
   ``envs.evaluation.evaluate_run(omnisafe_dir: str, spec: dict) -> dict`` with keys
     final_cost, final_return   mean episodic cost and return over EVAL_EPISODES (100) episodes of the
                                deterministic policy at the final checkpoint (Appendix B: final_cost; final_return)
     selection                  {step: [cost, return]} on the selection set for exactly the run's last
                                SELECTION_WINDOW_CHECKPOINTS (10) checkpoints (Table 2.1; Part 3.4; Part 4.1 rule 1)
     episodes                   the number of episodes per evaluation (must equal 100)
     selection_seeds            the 100 episode reset seeds of the selection set (Table 2.1: the
                                selection and measurement sets are disjoint sets of evaluation seeds;
                                every run uses the same sets)
2. Plasticity metrics, owned by Metrics and interventions (Role 3): an optional file
   ``plasticity.csv`` in the OmniSafe run directory with columns step, dormant, rank, norm (the
   actor), one row per checkpoint and at onset (Table 2.3; Appendix B: dormant_onset, rank_onset,
   norm_onset). Optional columns norm_reward_critic, norm_cost_critic hold the two critics' parameter
   norms, which Table 2.3 logs separately; they stay in the run directory, because ledger schema v1
   has one ``norm`` per checkpoint. The ledger writer reads only step, dormant, rank and norm.
3. Logging names, for every algorithm plug-in (Roles 2 and 5): every Lagrange multiplier is
   logged each epoch in a progress.csv column whose name starts with ``Metrics/LagrangeMultiplier``
   (Study B per level, e.g. ``Metrics/LagrangeMultiplier/level_10``), and every loss in a column
   starting with ``Loss/``. The launcher checks all of them for non-finite values (Part 5.6).
   A per-level column ends in the budget level (``level_10``, ``level_17.5``); the Continuous arm
   names each 5-unit bin's multiplier by the bin's lower edge (``level_10`` for [10, 15)), and its
   per-level check against ``training_levels`` is skipped (its ``training_levels`` is None). The
   ledger writer traces every multiplier column and records each level's final value
   (per_level_multipliers).
   Checkpoints: plug-ins mix in ``pilot.algorithms.FullStateCheckpointMixin`` and keep what they
   add to a checkpoint loadable with ``torch.load(weights_only=True)`` (tensors, state dicts and
   plain Python numbers), so continuations restore it without executing pickled code.
4. Checkpoint selector, owned by Analysis and results (Role 4):
   ``analysis.matching.select_checkpoint(checkpoints: list[dict]) -> int``, the step chosen by
   Part 4.1 rule 1 (closest selection-set cost to d = 25; ties to the later checkpoint, distances
   compared after rounding to 1e-4). Verified by ``pilot.enrichment.rule_one_step``.
5. Battery harness, owned by Environment and tests (Role 2):
   ``envs.evaluation.evaluate_battery(omnisafe_dir: str, spec: dict, step: int, conditions: list[str]) -> dict``
   with the measurement-set cost, the cost under each condition, ``episodes`` (100) and the
   ``measurement_seeds`` (disjoint from the selection seeds; Table 2.1). See pilot/enrichment.py.
6. Determinism evaluation, owned by Environment and tests (Role 2):
   ``envs.evaluation.evaluate_checkpoint(omnisafe_dir: str, step: int, episodes: int) -> float``,
   the evaluation cost of one checkpoint over ``episodes`` (100) episodes (Table 3.1 "Determinism
   check"; ``scripts/determinism_check.py --registered-form``).
"""

from __future__ import annotations

import importlib
import math
import numbers
import re
from pathlib import Path
from typing import Any, Callable, Mapping

from configs import registered as R

EVALUATOR_TARGET = "envs.evaluation:evaluate_run"
EVALUATOR_OWNER = "Environment and tests (Role 2)"
PLASTICITY_FILE = "plasticity.csv"
PLASTICITY_COLUMNS = ("step", "dormant", "rank", "norm")

_CKPT = re.compile(r"^epoch-(\d+)\.pt$")


class ContractError(ValueError):
    """A component returned something that breaks its interface."""


def load_evaluator() -> Callable[[str, Mapping[str, Any]], Mapping[str, Any]]:
    from pilot.algorithms import PluginUnavailableError

    module_name, func = EVALUATOR_TARGET.split(":")
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if missing and (module_name == missing or module_name.startswith(missing + ".")):
            raise PluginUnavailableError(
                f"the evaluation harness {EVALUATOR_TARGET} is provided by {EVALUATOR_OWNER} and is not in the repository yet"
            ) from exc
        raise  # the harness exists but one of its own imports failed: show the real error
    try:
        return getattr(module, func)
    except AttributeError as exc:
        raise PluginUnavailableError(f"{module_name} has no function {func!r} yet ({EVALUATOR_OWNER})") from exc


def checkpoint_steps(omnisafe_dir: Path) -> list[int]:
    """Steps of the checkpoints OmniSafe saved: torch_save/epoch-{k}.pt holds the state after k epochs."""
    steps = []
    for path in (Path(omnisafe_dir) / "torch_save").glob("epoch-*.pt"):
        match = _CKPT.match(path.name)
        if match:
            steps.append(int(match.group(1)) * R.STEPS_PER_EPOCH)
    return sorted(steps)


def selection_window(steps: list[int], total_steps: int | None = None) -> list[int]:
    """The run's last ten checkpoints, "the final 2,000,000 steps" (Part 4.1 rule 1).

    With ``total_steps`` given, the two descriptions must agree: exactly ten checkpoints lie in
    (total - 2,000,000, total]. They disagree when the total is off the 200,000-step grid; that
    case is open question Q-selection-window and is refused.
    """
    if len(steps) < R.SELECTION_WINDOW_CHECKPOINTS:
        raise ContractError(f"only {len(steps)} checkpoints; the selection rule needs {R.SELECTION_WINDOW_CHECKPOINTS}")
    window = steps[-R.SELECTION_WINDOW_CHECKPOINTS:]
    if total_steps is not None:
        inside = [s for s in steps if total_steps - R.SELECTION_WINDOW_STEPS < s <= total_steps]
        if inside != window:
            raise ContractError(
                f"the last ten checkpoints {window} are not the checkpoints of the final "
                f"{R.SELECTION_WINDOW_STEPS:,} steps {inside}; open question Q-selection-window"
            )
    return window


def expected_checkpoint_steps(spec: Any) -> list[int]:
    """Part 3.4: every 200,000 steps, at onset, and at the end of training (and the initial state)."""
    steps = set(range(0, spec.total_steps + 1, R.CHECKPOINT_INTERVAL_STEPS))
    steps.add(spec.total_steps)
    if spec.onset_step:
        steps.add(int(spec.onset_step))
    return sorted(steps)


def _whole_number(x: Any) -> int:
    """``x`` as an int if it is an integer, or a float with an integral value; else TypeError.

    Not ``int(x)``, which truncates 0.5 to seed 0, and not ``float(x) == int(x)``, which rejects
    64-bit seeds above 2**53. bool and numpy.bool_ are not seeds.
    """
    if isinstance(x, numbers.Integral) and not isinstance(x, bool):
        return int(x)
    if isinstance(x, numbers.Real) and math.isfinite(x) and float(x).is_integer():
        return int(x)
    raise TypeError(f"not an integer: {x!r}")


def validate_seed_set(name: str, seeds: Any) -> list[int]:
    """A set of evaluation seeds: exactly EVAL_EPISODES distinct non-negative integers."""
    try:
        values = [_whole_number(x) for x in seeds]
    except TypeError as exc:
        raise ContractError(f"{name} must be a list of integers") from exc
    if len(values) != R.EVAL_EPISODES or len(set(values)) != R.EVAL_EPISODES or min(values) < 0:
        raise ContractError(f"{name} must hold {R.EVAL_EPISODES} distinct non-negative seeds")
    return values


def check_canonical_seeds(kind: str, seeds: list[int], registry: Path) -> None:
    """Every run must use the same seed set of a kind (selection or measurement); the first run
    fixes it in ``registry`` (a JSON file beside the ledger) and later runs are compared with it."""
    import json

    registry = Path(registry)
    data = json.loads(registry.read_text(encoding="utf-8")) if registry.exists() else {}
    if kind in data:
        if sorted(data[kind]) != sorted(seeds):
            raise ContractError(f"the {kind} seeds differ from those of earlier runs ({registry})")
        return
    other = "measurement" if kind == "selection" else "selection"
    if other in data and set(data[other]) & set(seeds):
        raise ContractError(f"the {kind} and {other} seed sets overlap (Table 2.1 requires disjoint sets)")
    data[kind] = sorted(seeds)
    registry.parent.mkdir(parents=True, exist_ok=True)
    tmp = registry.with_name(registry.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    tmp.replace(registry)


def _finite(name: str, value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{name} is not a number: {value!r}") from exc
    if not math.isfinite(number):
        raise ContractError(f"{name} is not finite: {value!r}")
    return number


def validate_evaluation(evaluation: Mapping[str, Any], spec: Any, omnisafe_dir: Path) -> None:
    """Check the evaluation harness result against contract 1."""
    for key in ("final_cost", "final_return", "selection", "episodes", "selection_seeds"):
        if key not in evaluation:
            raise ContractError(f"{spec.run_id}: evaluation lacks {key!r}")
    validate_seed_set("selection_seeds", evaluation["selection_seeds"])
    _finite("final_cost", evaluation["final_cost"])
    _finite("final_return", evaluation["final_return"])
    if _finite("episodes", evaluation["episodes"]) != R.EVAL_EPISODES:
        raise ContractError(f"{spec.run_id}: {evaluation['episodes']} episodes per evaluation; Table 2.1 fixes {R.EVAL_EPISODES}")
    saved = checkpoint_steps(omnisafe_dir)
    missing = sorted(set(expected_checkpoint_steps(spec)) - set(saved))
    if missing:
        raise ContractError(f"{spec.run_id}: checkpoints required by Part 3.4 are missing at steps {missing}")
    expected = selection_window(saved, spec.total_steps)
    if not isinstance(evaluation["selection"], Mapping):
        raise ContractError(f"{spec.run_id}: selection must map checkpoint steps to [cost, return]")
    try:
        got = sorted(int(k) for k in evaluation["selection"])
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{spec.run_id}: selection keys must be checkpoint steps") from exc
    if got != expected:
        raise ContractError(f"{spec.run_id}: selection steps {got} are not the last ten checkpoints {expected}")
    for step, pair in evaluation["selection"].items():
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ContractError(f"{spec.run_id}: selection[{step}] must be [cost, return]")
        _finite(f"selection[{step}].cost", pair[0])
        _finite(f"selection[{step}].return", pair[1])

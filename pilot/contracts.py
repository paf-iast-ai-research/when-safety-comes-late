"""Interfaces between the pilot owner's pipeline and the other roles' code.

Owner: pilot owner (Role 1). The scheduler and ledger writer (Role 1) call code owned by other
roles through the functions named here. Each contract is checked when its result comes back, so a
mistake in one component is caught at the boundary instead of reaching the ledger.

1. Evaluation harness, owned by Environment and tests (Role 2) and, for budget-conditioned runs,
   by Study B and literature (Role 5). ``evaluator_target(spec)`` names the function:
   ``studyb.evaluation:evaluate_run`` for plug-in ``study_b`` (the budget must be appended to the
   observation), ``envs.evaluation:evaluate_run`` for every other run. Continuations (few-shot and
   battery) are never evaluated by the launcher's evaluation stage. Both return a dict with keys
     final_cost, final_return   mean episodic cost and return over EVAL_EPISODES (100) episodes of the
                                deterministic policy at the final checkpoint (Appendix B: final_cost; final_return),
                                on the measurement set (Q-final-cost-set, answered in Table 9.1)
     selection                  {step: [cost, return]} on the selection set for exactly the run's last
                                SELECTION_WINDOW_CHECKPOINTS (10) checkpoints (Table 2.1; Part 3.4; Part 4.1 rule 1)
                                (``selection_window``: for a total off the 200,000-step grid, the
                                end-relative window of Q-selection-window)
     episodes                   the number of episodes per evaluation (must equal 100)
     selection_seeds            the 100 episode reset seeds of the selection set (Table 2.1: the
                                selection and measurement sets are disjoint sets of evaluation seeds;
                                that every run uses the same sets is this pipeline's rule,
                                ``check_canonical_seeds``)
   and, when an episode was unstable, ``unstable_replacements`` {"final": [...], "selection": {step:
   [...]}}: the reserve seeds that replaced unstable episodes (Q-mujoco-exception, answered in Table
   9.1; ``validate_unstable_replacements``). The seed lists stay the canonical planned ones. The
   ledger writer also requires ``final_seed_set`` ("measurement") and ``measurement_seeds`` (the 100
   seeds of the measurement set), with the per-episode arrays (``pilot.ledger_writer.evaluation_record``).
2. Plasticity metrics, owned by Metrics and interventions (Role 3): a file ``plasticity.csv`` in the
   OmniSafe run directory whose columns start step, dormant, rank, norm (the actor), one row per
   saved checkpoint, the onset included (Table 2.3; Appendix B: dormant_onset, rank_onset,
   norm_onset). Required for the Study A plug-ins (``plasticity_required``), whose
   ``FullStateCheckpointMixin`` installs ``metrics.hook.install_plasticity_hook``; checked by
   ``validate_plasticity``. Further columns hold what Table 2.3 logs beside the actor
   (norm_reward_critic, norm_cost_critic) and the trainable-layer metrics of box H3 (c)'s
   manipulation check (dormant_trainable, rank_trainable, ...). Ledger schema v1 has one ``norm`` per
   checkpoint, so they stay in the run directory, and their rows at onset and at the manipulation
   check reach the ``training`` supplement record (pilot/enrichment.py).
3. Logging names and checkpoints, for every algorithm plug-in (Roles 1, 2 and 5): every Lagrange multiplier is
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
   plain Python numbers), so continuations restore it without executing pickled code. Besides the
   ten-epoch grid, a run saves the steps of ``extra_checkpoint_steps``.
4. Checkpoint selector, owned by Analysis and results (Role 4):
   ``analysis.matching.select_checkpoint(checkpoints: list[dict]) -> int``, the step chosen by
   Part 4.1 rule 1 (closest selection-set cost to d = 25; a tie goes to the later checkpoint,
   distances compared after rounding to 1e-4, Q-tie-break as answered in Table 9.1). Verified by
   ``pilot.enrichment.rule_one_step``.
5. Battery harness, owned by Environment and tests (Role 2):
   ``envs.evaluation.evaluate_battery(omnisafe_dir: str, spec: dict, step: int, conditions: Sequence[str]) -> dict``
   with the measurement-set cost, the cost under each condition, ``episodes`` (100) and the
   ``measurement_seeds`` (disjoint from the selection seeds; Table 2.1). See pilot/enrichment.py.
6. Determinism evaluation, owned by Environment and tests (Role 2):
   ``envs.evaluation.evaluate_checkpoint(omnisafe_dir: str, step: int, episodes: int, *, spec=None) -> float``,
   the evaluation cost of one checkpoint over ``episodes`` (100) episodes (Table 3.1 "Determinism
   check"; ``scripts/determinism_check.py --registered-form``).

Few-shot results (Table 2.5; Appendix B ``sr_fewshot[budget, horizon]``) are keyed by
``fewshot_key(budget, horizon)`` (``"5.0_200000"``), the one spelling every writer and reader uses
(``studyb.evaluation.evaluate_fewshot`` and pilot/enrichment.py write it; analysis/data.py reads it);
``parse_fewshot_key`` reads it back.
"""

from __future__ import annotations

import csv
import io
import json
import math
import numbers
import os
import re
import warnings
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from configs import registered as R
from pilot import errors
from pilot.errors import PendingQuestionError
from pilot.manifest import CONTINUATION_GROUPS, ROLE_OWNERS
from pilot.rundir import checkpoint_files, run_steps_per_epoch

EVALUATOR_TARGET = "envs.evaluation:evaluate_run"  # contract 1 for every run that is not budget-conditioned
STUDY_B_EVALUATOR_TARGET = "studyb.evaluation:evaluate_run"  # contract 1 for plug-in study_b (``evaluator_target``)
EVALUATOR_OWNERS = {package: ROLE_OWNERS[package] for package in ("envs", "studyb")}
PLASTICITY_FILE = "plasticity.csv"
PLASTICITY_COLUMNS = ("step", "dormant", "rank", "norm")
PLASTICITY_PLUGINS = ("study_a", "study_a_pid")  # the Study A training plug-ins (pilot.launch.pilot_config)

SELECTION_WINDOW_KEY = "Q-selection-window"

# Q-mujoco-exception (Table 9.1): an evaluation episode that MuJoCo reports unstable is replaced by an
# episode on the next unused seed of its set's reserve sequence, the set's base (its smallest seed) +
# RESERVE_SEED_OFFSET + r, r = 0, 1, ... within one evaluation call (one checkpoint, one condition);
# more than MAX_UNSTABLE_EPISODES unstable episodes, planned or reserve, fail the call. The harness
# (envs.evaluation.run_episodes) uses these two numbers; the pipeline checks every result against them.
RESERVE_SEED_OFFSET = 50_000  # answered in Table 9.1 (Q-mujoco-exception)
MAX_UNSTABLE_EPISODES = 5  # answered in Table 9.1 (Q-mujoco-exception)
REPLACEMENT_FIELDS = ("slot_index", "seed", "reserve_seed", "step", "warning")

# Used with fullmatch ("$" would also match before a trailing newline); [0-9], not \d, which
# matches any Unicode decimal digit.
_DIGITS = re.compile(r"[0-9]+")


class ContractError(ValueError):
    """A component returned something that breaks its interface."""


class SelectionWindowPending(ContractError, PendingQuestionError):
    """The selection window waits for open question Q-selection-window: a refusal, not a failure.

    A PendingQuestionError (a RunRefused: the launcher's exit 5, HANDOVER.md section 8), and a
    ContractError for the harnesses that turn either into their own refusal.
    """


def _field(spec: Any, name: str) -> Any:
    """``spec.name`` for a RunSpec, ``spec[name]`` for its dict form (the harness receives the dict)."""
    return spec[name] if isinstance(spec, Mapping) else getattr(spec, name)


def evaluator_target(spec: Any) -> str:
    """The ``module:function`` of contract 1 that evaluates ``spec``.

    Plug-in ``study_b`` needs the budget appended to every observation, which only Role 5's harness
    does; every other training run (Study A, the pilot's unconstrained PPO run) goes to Role 2's.
    Few-shot and battery continuations end as ``continued`` and never reach the evaluation stage.
    """
    return STUDY_B_EVALUATOR_TARGET if _field(spec, "plugin") == "study_b" else EVALUATOR_TARGET


def load_evaluator(spec: Any) -> Callable[[str, Mapping[str, Any]], Mapping[str, Any]]:
    """Import the contract-1 function for ``spec``.

    PluginUnavailableError (the launcher's exit 4, HANDOVER.md section 8) when the harness module, or
    a repository module (or a name of one) it imports at its top level (Role 5's harness is built
    on Role 2's ``envs.evaluation``, studyb/conditioning.py), is not written yet, or the module lacks the
    function; a missing library re-raises its ImportError (``pilot.algorithms.import_provided``).
    """
    from pilot.algorithms import PluginUnavailableError, import_provided

    target = evaluator_target(spec)
    module_name, func = target.split(":")
    owner = EVALUATOR_OWNERS.get(module_name.split(".")[0], "another role")
    module = import_provided(module_name, f"the evaluation harness {target} (provided by {owner})")
    try:
        return getattr(module, func)
    except AttributeError as exc:
        raise PluginUnavailableError(f"{module_name} has no function {func!r} yet ({owner})") from exc


def checkpoint_steps(omnisafe_dir: Path, steps_per_epoch: int = R.STEPS_PER_EPOCH) -> list[int]:
    """Steps of the checkpoints OmniSafe saved: torch_save/epoch-{k}.pt holds the state after k epochs.

    ``steps_per_epoch`` is the run's own epoch length; registered runs use R.STEPS_PER_EPOCH
    (pilot/launch.py), the tiny test configurations pass theirs (``rundir.run_steps_per_epoch``).
    """
    return [k * int(steps_per_epoch) for k, _ in checkpoint_files(omnisafe_dir)]


def on_checkpoint_grid(total_steps: int) -> bool:
    """True when the run's total is a multiple of R.CHECKPOINT_INTERVAL_STEPS (200,000 steps)."""
    return int(total_steps) % R.CHECKPOINT_INTERVAL_STEPS == 0


def end_relative_window(total_steps: int) -> list[int]:
    """The ten steps total - 9 x 200,000, ..., total - 200,000, total, ascending.

    Q-selection-window, answered in Table 9.1: for a total off the 200,000-step grid (11,120,000;
    13,340,000; 12,500,000 steps), the window is an end-relative grid that the run saves in addition
    to its 200,000-step grid, so that it is both the run's last ten checkpoints and its final
    2,000,000 steps (Part 4.1 rule 1). The run saves them (``extra_checkpoint_steps``). For a total
    on the grid they are the grid's last ten. For a total below 1,800,000 steps the first of them are
    negative; ``selection_window`` refuses such a total, and ``extra_checkpoint_steps`` keeps only its
    positive steps.
    """
    total = int(total_steps)
    return [total - k * R.CHECKPOINT_INTERVAL_STEPS for k in range(R.SELECTION_WINDOW_CHECKPOINTS - 1, -1, -1)]


def selection_window(steps: list[int], total_steps: int | None = None) -> list[int]:
    """The run's last ten checkpoints, "the final 2,000,000 steps" (Part 4.1 rule 1).

    ``steps`` are the saved checkpoint steps (as ``checkpoint_steps`` returns them); they are
    sorted on entry, so their order does not matter. With ``total_steps`` given, the two
    descriptions must agree: exactly ten checkpoints lie in (total - 2,000,000, total], and no
    checkpoint lies beyond the total. They disagree on the saved 200,000-step grid when the total is
    off that grid; that case is question Q-selection-window (a run gate). While it is open it is
    refused (SelectionWindowPending, a PendingQuestionError), unless pending questions are allowed
    (smoke roots only, ``pilot.errors.allow_pending``); once answered (Table 9.1), the window is the
    answer's end-relative grid (``end_relative_window``), which the run saved
    (``extra_checkpoint_steps``), and a missing step of it is refused. For a total on the grid a
    disagreement is a broken checkpoint set (a missing or an extra checkpoint), refused with the
    steps it concerns.
    """
    steps = sorted(steps)
    total = None if total_steps is None else int(total_steps)
    if len(steps) < R.SELECTION_WINDOW_CHECKPOINTS:
        raise ContractError(f"only {len(steps)} checkpoints; the selection rule needs {R.SELECTION_WINDOW_CHECKPOINTS}")
    if total is not None:
        beyond = [s for s in steps if s > total]
        if beyond:
            raise ContractError(f"checkpoints beyond the run's total {total:,} steps at steps {beyond}")
    if total is not None and not on_checkpoint_grid(total):
        window = end_relative_window(total)
        if errors.open_keys(SELECTION_WINDOW_KEY) and not errors.pending_allowed():
            raise SelectionWindowPending(
                f"the total {total:,} is off the {R.CHECKPOINT_INTERVAL_STEPS:,}-step grid: the last "
                f"{R.SELECTION_WINDOW_CHECKPOINTS} checkpoints and the final {R.SELECTION_WINDOW_STEPS:,} steps "
                "differ; "
                f"open question {SELECTION_WINDOW_KEY} (its proposal, the end-relative window {window}, waits for "
                "the answer)"
            )
        if window[0] < 0:
            raise ContractError(
                f"the total {total:,} is shorter than the {R.SELECTION_WINDOW_CHECKPOINTS}-checkpoint "
                "end-relative window"
            )
        missing = sorted(set(window) - set(steps))
        if missing:
            raise ContractError(f"the end-relative selection window {window} ({SELECTION_WINDOW_KEY}) lacks the "
                                f"checkpoints at steps {missing}")
        return window
    window = steps[-R.SELECTION_WINDOW_CHECKPOINTS:]
    if total is not None:
        inside = [s for s in steps if total - R.SELECTION_WINDOW_STEPS < s <= total]
        if inside != window:
            grid = end_relative_window(total)  # on the grid: the grid's last ten
            missing = sorted(set(grid) - set(steps))
            extra = sorted(set(inside) - set(grid))
            problems = ([f"the grid checkpoints at steps {missing} are missing"] if missing else []) + (
                [f"the checkpoints at steps {extra} are off the grid"] if extra else [])
            raise ContractError(
                f"the last {R.SELECTION_WINDOW_CHECKPOINTS} checkpoints {window} are not the checkpoints of the final "
                f"{R.SELECTION_WINDOW_STEPS:,} steps {inside}: " + "; ".join(problems) + " (a broken checkpoint set)"
            )
    return window


def extra_checkpoint_steps(spec: Any) -> list[int]:
    """Steps off the ten-epoch grid that the run must also save, ascending.

    * the onset, when it is after step 0 (Part 3.4: a checkpoint "at onset");
    * for the Study A plug-ins with a late onset, onset + 200,000 steps, when that is within the run
      (Part 1.2, box H3 (c): "Manipulation check, at the first logging point after onset (200,000
      steps)"; R.MANIPULATION_CHECK_STEPS_AFTER_ONSET), the H3 (c) reading point (Q-manipulation-point,
      answered in Table 9.1). When the onset is off the grid (N = 0.25 total-steps: 2,500,000) that
      step is 2,700,000, off the grid as well, and the grid checkpoint 2,600,000 is not used for the
      check; the extra save changes no update (pilot.algorithms.FullStateCheckpointMixin);
    * for a few-shot continuation, its horizons (Table 2.5: "satisfaction is measured at 200,000,
      500,000 and 1,000,000 steps ... read from one continuation"); 500,000 is off the grid;
    * for a total off the grid, the end-relative selection window total - k x 200,000 (k = 1..9;
      ``end_relative_window``, Q-selection-window's answer in Table 9.1; whole epochs, so the mixin
      saves them and training is unchanged). The run itself waits on the key while it is open
      (pilot/manifest.py).
      Not for a continuation (``manifest.CONTINUATION_GROUPS``), which rule 1 does not select from.

    Steps on the grid (every R.CHECKPOINT_INTERVAL_STEPS), step 0 and the final step are saved by
    OmniSafe's own cadence and are not listed; neither is a step beyond the run.
    """
    total = int(_field(spec, "total_steps"))
    onset = int(_field(spec, "onset_step") or 0)
    plugin = _field(spec, "plugin")
    candidates: set[int] = set()
    if onset > 0:
        candidates.add(onset)
        if plugin in PLASTICITY_PLUGINS:
            candidates.add(onset + R.MANIPULATION_CHECK_STEPS_AFTER_ONSET)
    if plugin == "study_b_fewshot":
        params = _field(spec, "params") or {}
        try:
            candidates.update(whole_number(h) for h in params.get("horizons") or ())
        except TypeError as exc:
            raise ValueError(f"{_field(spec, 'run_id')}: few-shot horizons must be whole step counts") from exc
    if not on_checkpoint_grid(total) and _field(spec, "group") not in CONTINUATION_GROUPS:  # not selected by rule 1
        candidates.update(end_relative_window(total)[:-1])
    return sorted(s for s in candidates if 0 < s < total and s % R.CHECKPOINT_INTERVAL_STEPS)


def expected_checkpoint_steps(spec: Any) -> list[int]:
    """The checkpoint steps a run must save, ascending.

    Part 3.4: every 200,000 steps, at onset, and at the end of training (and the initial state),
    plus the other extra steps of ``extra_checkpoint_steps``.
    """
    total = int(_field(spec, "total_steps"))
    steps = set(range(0, total + 1, R.CHECKPOINT_INTERVAL_STEPS))
    steps.add(total)
    steps.update(extra_checkpoint_steps(spec))
    return sorted(steps)


def whole_number(x: Any) -> int:
    """``x`` as an int if it is an integer, or a real number with an integral value; else TypeError.

    Not ``int(x)``, which truncates 0.5 to 0 (a seed, a step or a horizon), and not
    ``float(x) == int(x)``, which rejects 64-bit seeds above 2**53; the value is never truncated.
    A bool or numpy.bool_ is not a whole number here.
    """
    if isinstance(x, bool):
        raise TypeError(f"not an integer: {x!r}")  # bool is an Integral and a Real; numpy.bool_ is neither
    if isinstance(x, numbers.Integral):
        return int(x)
    if isinstance(x, numbers.Real) and math.isfinite(x):
        n = int(x)
        if n == x:  # exact for Fraction and numpy.longdouble, unlike float(x).is_integer()
            return n
    raise TypeError(f"not an integer: {x!r}")


# ---------------------------------------------------------------------------
# Contract 2: plasticity metrics (Table 2.3; equations 6 and 7)
# ---------------------------------------------------------------------------


def plasticity_required(spec: Any) -> bool:
    """True for the runs whose checkpoints carry plasticity metrics: the Study A training plug-ins.

    Table 2.3 logs dormant fraction, effective rank and parameter norm at its "Logging points": "Every
    200,000 steps, that is every ten epochs (decision), at onset, and at the end of training" for
    Study A; the ppolag determinism check (OmniSafe unchanged), the Study B plug-ins and the battery
    continuations are not Study A training runs (pilot.launch.pilot_config). The determinism checks
    of the Study A plug-ins (``pilot.scheduler.determinism_spec``) are, and log plasticity.
    """
    return _field(spec, "study") == "A" and _field(spec, "plugin") in PLASTICITY_PLUGINS


def validate_plasticity(omnisafe_dir: Path, spec: Any, *, completed: bool) -> dict[int, dict[str, float]]:
    """Check ``plasticity.csv`` against contract 2 and return {step: {column: value}}.

    * The header starts with step, dormant, rank, norm; column names are unique.
    * Every step is written as a non-negative integer; steps are strictly ascending (so unique: a
      step written twice would let one row silently replace another in the ledger).
    * Every other value parses as a float; ``nan`` and ``inf`` are allowed (a run with non-finite
      weights has non-finite metrics, and the ledger writer leaves them null with a note).
    * A completed run of a plug-in that must log plasticity (``plasticity_required``) has the file,
      with a row for every checkpoint it saved (step 0, the grid, the onset, the extra steps and the
      end). A failed run may lack trailing rows or the whole file (it may have failed before the
      hook was installed); its exclusion must still reach the ledger (Part 5.6). An empty file
      counts as none for a failed run: the hook creates the file (``open(path, "x")``) before it
      writes the header, so a run killed between the two leaves 0 bytes (``metrics/hook.py``
      ``start``). A completed run's empty file breaks the contract.
    * A failed run's last line may also be cut short. The hook writes every row whole, ending in a
      newline (``metrics/hook.py``: one ``csv.writer(..., lineterminator="\\n").writerow`` per
      append), so a last line without a line end is an append that the failure itself interrupted
      (a full disk, a kill). It is dropped, with a RuntimeWarning, not read: its last value may be
      truncated ("0.2" of "0.25") and still parse. Every other line is checked as usual, so a
      malformed line that is complete still breaks the contract. A completed run gets no such
      allowance: an append that fails raises inside training (``record`` calls ``_append`` outside
      its error handler), so a run that completed finished every append.

    Checkpoint steps are counted in the run's own ``steps_per_epoch`` (its config.json; the
    registered value when a fixture has none), as the hook writes them. Every column except
    ``step`` is returned, the optional ones included. ContractError on any violation.
    """
    omnisafe_dir = Path(omnisafe_dir)
    path = omnisafe_dir / PLASTICITY_FILE
    run_id = _field(spec, "run_id")
    required = plasticity_required(spec)
    if not path.exists():
        if required and completed:
            raise ContractError(
                f"{run_id}: the completed run lacks {PLASTICITY_FILE} (contract 2; Table 2.3; Part 3.4)"
            )
        return {}
    text = path.read_text(encoding="utf-8")
    if not text and not completed:
        return {}  # created, then killed before the header was written: like no file at all
    if text and not text.endswith("\n") and not completed:
        kept, line_end, partial = text.rpartition("\n")
        warnings.warn(
            f"{path}: the failed run's last line {partial!r} has no line end (an append cut short); "
            "dropped (contract 2)",
            RuntimeWarning, stacklevel=2,
        )
        if not line_end:
            return {}  # the header itself was cut short: the run failed before its first row
        text = kept + line_end
    rows: dict[int, dict[str, float]] = {}
    with io.StringIO(text, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None or tuple(header[: len(PLASTICITY_COLUMNS)]) != PLASTICITY_COLUMNS:
            raise ContractError(
                f"{path}: the header {header!r} does not start with {', '.join(PLASTICITY_COLUMNS)} (contract 2)"
            )
        if len(set(header)) != len(header):
            raise ContractError(f"{path}: the header {header!r} repeats a column (contract 2)")
        previous = -1
        for line, fields in enumerate(reader, start=2):
            if len(fields) != len(header):
                raise ContractError(
                    f"{path}: line {line} has {len(fields)} fields, the header {len(header)} (contract 2)"
                )
            if not _DIGITS.fullmatch(fields[0]):
                raise ContractError(f"{path}: line {line}: step {fields[0]!r} is not an integer (contract 2)")
            step = int(fields[0])
            if step <= previous:
                raise ContractError(
                    f"{path}: line {line}: step {step} after step {previous}; steps must be unique and "
                    "ascending (contract 2)"
                )
            values: dict[str, float] = {}
            for name, cell in zip(header[1:], fields[1:]):
                try:
                    values[name] = float(cell)  # "nan" and "inf" parse; the ledger writer leaves them null
                except ValueError:
                    raise ContractError(
                        f"{path}: {name} at step {step} is {cell!r}, not a number (contract 2)"
                    ) from None
            rows[step] = values
            previous = step
    if required and completed:
        steps_per_epoch = run_steps_per_epoch(omnisafe_dir, default=R.STEPS_PER_EPOCH)
        missing = sorted(set(checkpoint_steps(omnisafe_dir, steps_per_epoch)) - set(rows))
        if missing:
            raise ContractError(
                f"{run_id}: {PLASTICITY_FILE} has no row for the saved checkpoints at steps {missing} "
                "(contract 2; Table 2.3)"
            )
    return rows


# ---------------------------------------------------------------------------
# Few-shot keys (Table 2.5; Appendix B sr_fewshot[budget, horizon]; written by studyb/evaluation.py
# evaluate_fewshot and pilot/enrichment.py, read by analysis/data.py)
# ---------------------------------------------------------------------------


def fewshot_key(budget: Any, horizon: Any) -> str:
    """The ``sr_fewshot`` key of an unseen budget and a horizon: ``"5.0_200000"``.

    The form of the frozen schema's own test data (tests/test_ledger_schema.py: "5.0_200000"), which
    its validator accepts ("budget_horizon", numeric budget, a horizon of Table 2.5). ValueError for a
    non-finite budget or a horizon that is not a positive whole number of steps. A budget of -0.0
    is written as 0.0, so that it has one key.
    """
    if isinstance(budget, bool) or not isinstance(budget, numbers.Real):
        raise ValueError(f"few-shot budget must be a finite number, got {budget!r}")
    try:
        value = float(budget)
    except OverflowError:
        raise ValueError(f"few-shot budget must be a finite number, got {budget!r}") from None
    if not math.isfinite(value):
        raise ValueError(f"few-shot budget must be a finite number, got {budget!r}")
    try:
        steps = whole_number(horizon)
    except TypeError:
        raise ValueError(f"few-shot horizon must be a whole number of steps, got {horizon!r}") from None
    if steps <= 0:
        raise ValueError(f"few-shot horizon must be positive, got {horizon!r}")
    return f"{value + 0.0}_{steps}"  # + 0.0 turns -0.0 into 0.0


def parse_fewshot_key(key: Any) -> tuple[float, int]:
    """(budget, horizon) of a key written by ``fewshot_key``; ValueError for any other spelling.

    Only the canonical form is accepted ("5_200000" is refused), so that one result can never be
    stored under two keys.
    """
    if not isinstance(key, str) or key.count("_") != 1:
        raise ValueError(f"few-shot key must be 'budget_horizon', got {key!r}")
    budget_text, horizon_text = key.split("_")
    try:
        budget = float(budget_text)
    except ValueError:
        raise ValueError(f"few-shot key {key!r}: budget {budget_text!r} is not a number") from None
    if not _DIGITS.fullmatch(horizon_text):
        raise ValueError(f"few-shot key {key!r}: horizon {horizon_text!r} is not a whole number")
    horizon = int(horizon_text)
    if fewshot_key(budget, horizon) != key:
        raise ValueError(f"few-shot key {key!r} is not in the canonical form {fewshot_key(budget, horizon)!r}")
    return budget, horizon


# ---------------------------------------------------------------------------
# Evaluation seeds and contract 1
# ---------------------------------------------------------------------------


def validate_seed_set(name: str, seeds: Any) -> list[int]:
    """A set of evaluation seeds: exactly EVAL_EPISODES distinct non-negative integers."""
    try:
        values = [whole_number(x) for x in seeds]
    except TypeError as exc:
        raise ContractError(f"{name} must be a list of integers") from exc
    if len(values) != R.EVAL_EPISODES or len(set(values)) != R.EVAL_EPISODES or min(values) < 0:
        raise ContractError(f"{name} must hold {R.EVAL_EPISODES} distinct non-negative seeds")
    return values


def reserve_seeds(canonical: Sequence[int]) -> list[int]:
    """The reserve sequence of a canonical evaluation seed set (Q-mujoco-exception, Table 9.1).

    Its smallest seed (the set's base: 1,000,000 selection, 2,000,000 measurement, 3,100,000 hazard
    episodes) + ``RESERVE_SEED_OFFSET`` + r for r < ``MAX_UNSTABLE_EPISODES``, in the order the reserve
    seeds are used; empty for an empty set.
    """
    if not canonical:
        return []
    base = min(whole_number(s) for s in canonical) + RESERVE_SEED_OFFSET
    return [base + r for r in range(MAX_UNSTABLE_EPISODES)]


def validate_unstable_replacements(name: str, replacements: Any, seeds: Sequence[int], *,
                                   reserve: Sequence[int] | None = None) -> list[dict[str, Any]]:
    """The replacements of one evaluation call's unstable episodes, checked; ContractError otherwise.

    Q-mujoco-exception (Table 9.1): each entry is {slot_index, seed, reserve_seed, step, warning},
    in the order the replacements were made. ``seeds`` are the call's planned (canonical) seeds, which
    the result keeps; ``reserve`` its set's reserve sequence (default ``reserve_seeds(seeds)``, for a
    whole canonical set). Checked: at most ``MAX_UNSTABLE_EPISODES`` entries; the slots in the order
    the episodes ran; each slot's first entry replaces its planned seed and each later one the
    previous reserve seed (a reserve episode that is itself unstable is replaced the same way); the
    reserve seeds are the sequence's first ones, in order, so they lie in the set's reserve range
    and are distinct within the call; the step a non-negative integer and the warning a non-empty
    string. None (a result from before the replacements existed) is no replacement.
    """
    if replacements is None:
        return []
    if not isinstance(replacements, (list, tuple)):
        raise ContractError(f"{name}: unstable_replacements must be a list, got {type(replacements).__name__}")
    if len(replacements) > MAX_UNSTABLE_EPISODES:
        raise ContractError(f"{name}: {len(replacements)} unstable episodes were replaced; at most "
                            f"{MAX_UNSTABLE_EPISODES} per evaluation call (Q-mujoco-exception)")
    allowed = list(reserve_seeds(seeds) if reserve is None else reserve)
    planned = list(seeds)
    out: list[dict[str, Any]] = []
    for r, entry in enumerate(replacements):
        if not isinstance(entry, Mapping) or set(entry) != set(REPLACEMENT_FIELDS):
            raise ContractError(f"{name}: replacement {r} must hold exactly {', '.join(REPLACEMENT_FIELDS)}")
        try:
            slot, seed, reserve_seed, step = (whole_number(entry[k]) for k in REPLACEMENT_FIELDS[:4])
        except TypeError as exc:
            raise ContractError(f"{name}: replacement {r}: {exc}") from None
        warning = entry["warning"]
        if not isinstance(warning, str) or not warning or step < 0:
            raise ContractError(f"{name}: replacement {r} needs MuJoCo's warning and a non-negative step")
        if not 0 <= slot < len(planned):
            raise ContractError(f"{name}: replacement {r} names episode {slot} of {len(planned)}")
        previous = out[-1] if out else None
        if previous is not None and slot < previous["slot_index"]:
            raise ContractError(f"{name}: replacement {r} of episode {slot} after one of episode {previous['slot_index']}")
        same_slot = previous is not None and previous["slot_index"] == slot
        expected = previous["reserve_seed"] if same_slot else planned[slot]
        if seed != expected:
            raise ContractError(f"{name}: replacement {r} replaces seed {seed}; episode {slot} ran seed {expected}")
        if r >= len(allowed) or reserve_seed != allowed[r]:
            raise ContractError(f"{name}: replacement {r} ran reserve seed {reserve_seed}, not the next unused seed of "
                                f"the set's reserve sequence {allowed[:MAX_UNSTABLE_EPISODES]} (Q-mujoco-exception)")
        out.append({"slot_index": slot, "seed": seed, "reserve_seed": reserve_seed, "step": step, "warning": warning})
    return out


def check_canonical_seeds(kind: str, seeds: list[int], registry: Path, *, write: bool = True) -> None:
    """Every run must use the same seed set of a kind (selection or measurement).

    The first run fixes it in ``registry`` (a JSON file beside the ledger) and later runs are
    compared with it. ``write=False`` only checks: it raises what the call would raise and changes
    nothing. A new set must also keep the reserve sequences (``reserve_seeds``; Q-mujoco-exception)
    disjoint from every registered set, its own included, so that a replaced episode never runs a
    seed of a canonical set.
    """
    registry = Path(registry)
    data = json.loads(registry.read_text(encoding="utf-8")) if registry.exists() else {}
    if kind in data:
        if sorted(data[kind]) != sorted(seeds):
            raise ContractError(f"the {kind} seeds differ from those of earlier runs ({registry})")
        return
    other = "measurement" if kind == "selection" else "selection"
    if other in data and set(data[other]) & set(seeds):
        raise ContractError(f"the {kind} and {other} seed sets overlap (Table 2.1 requires disjoint sets)")
    sets = {**data, kind: list(seeds)}
    for owner, values in sets.items():
        clash = sorted(set(reserve_seeds(values)) & {s for v in sets.values() for s in v})
        if clash:
            raise ContractError(f"the reserve seeds {clash} of the {owner} set are seeds of a canonical set "
                                "(Q-mujoco-exception: the reserve sequences must be disjoint from every set)")
    if not write:
        return
    data[kind] = sorted(seeds)
    registry.parent.mkdir(parents=True, exist_ok=True)
    # Written as the ledger is (ledger_writer.ledger_transaction): fsync, rename, fsync of the directory, so a crash
    # never leaves an empty or partial registry, which every later ledger write would fail to read.
    tmp = registry.with_name(registry.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(data, indent=1))
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, registry)
    try:
        fd = os.open(registry.parent, os.O_RDONLY)
    except OSError:  # pragma: no cover - platforms without directory handles
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _finite(name: str, value: Any) -> float:
    """``value`` as a float if it is a finite real number (not a bool, not a string); else ContractError."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ContractError(f"{name} is not a number: {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ContractError(f"{name} is not finite: {value!r}")
    return number


def _step_key(key: Any) -> int:
    """A ``selection`` key as a checkpoint step; TypeError for anything else, never truncated.

    An int (not a bool), a float with an integral value, or a string of digits (JSON object keys
    arrive as strings).
    """
    if isinstance(key, str):
        if not _DIGITS.fullmatch(key):
            raise TypeError(f"not a step: {key!r}")
        return int(key)
    return whole_number(key)


def validate_evaluation(evaluation: Mapping[str, Any], spec: Any, omnisafe_dir: Path) -> None:
    """Check the evaluation harness result against contract 1; ContractError on any violation.

    Besides the result's own keys and values, the run directory must hold every checkpoint the run
    must save (Part 3.4 and the extra steps of ``extra_checkpoint_steps``, ``expected_checkpoint_steps``), and
    ``selection`` must cover exactly the selection window (``selection_window``). The replacements of
    unstable episodes, when the result has them (``unstable_replacements``), must follow
    Q-mujoco-exception's rule against the planned seed sets (``validate_unstable_replacements``).
    """
    run_id = _field(spec, "run_id")
    for key in ("final_cost", "final_return", "selection", "episodes", "selection_seeds"):
        if key not in evaluation:
            raise ContractError(f"{run_id}: evaluation lacks {key!r}")
    validate_seed_set("selection_seeds", evaluation["selection_seeds"])
    _finite("final_cost", evaluation["final_cost"])
    _finite("final_return", evaluation["final_return"])
    if _finite("episodes", evaluation["episodes"]) != R.EVAL_EPISODES:
        raise ContractError(
            f"{run_id}: {evaluation['episodes']} episodes per evaluation; Table 2.1 fixes {R.EVAL_EPISODES}"
        )
    saved = checkpoint_steps(omnisafe_dir)
    missing = sorted(set(expected_checkpoint_steps(spec)) - set(saved))
    if missing:
        raise ContractError(f"{run_id}: checkpoints required by Part 3.4 and ``extra_checkpoint_steps`` are missing at steps {missing}")
    expected = selection_window(saved, _field(spec, "total_steps"))
    if not isinstance(evaluation["selection"], Mapping):
        raise ContractError(f"{run_id}: selection must map checkpoint steps to [cost, return]")
    try:
        got = sorted(_step_key(k) for k in evaluation["selection"])
    except TypeError as exc:
        raise ContractError(f"{run_id}: selection keys must be checkpoint steps") from exc
    if got != expected:
        raise ContractError(f"{run_id}: selection steps {got} are not the last {R.SELECTION_WINDOW_CHECKPOINTS} "
                            f"checkpoints {expected}")
    for step, pair in evaluation["selection"].items():
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ContractError(f"{run_id}: selection[{step}] must be [cost, return]")
        _finite(f"selection[{step}].cost", pair[0])
        _finite(f"selection[{step}].return", pair[1])
    if evaluation.get("unstable_replacements") is not None:
        _check_replacements(evaluation, run_id, expected)


def _check_replacements(evaluation: Mapping[str, Any], run_id: str, window: list[int]) -> None:
    """Contract 1's ``unstable_replacements`` {"final": [...], "selection": {step: [...]}}, each list
    checked against the planned seeds of its evaluation (``validate_unstable_replacements``)."""
    replacements = evaluation["unstable_replacements"]
    if (not isinstance(replacements, Mapping) or not set(replacements) <= {"final", "selection"}
            or not isinstance(replacements.get("selection", {}), Mapping)):
        raise ContractError(f"{run_id}: unstable_replacements must map 'final' to a list and 'selection' to "
                            "{step: list} (Q-mujoco-exception)")
    selection_seeds = validate_seed_set("selection_seeds", evaluation["selection_seeds"])
    try:
        by_step = {_step_key(k): v for k, v in replacements.get("selection", {}).items()}
    except TypeError as exc:
        raise ContractError(f"{run_id}: unstable_replacements['selection'] keys must be checkpoint steps") from exc
    outside = sorted(set(by_step) - set(window))
    if outside:
        raise ContractError(f"{run_id}: unstable_replacements at steps {outside} outside the selection window {window}")
    for step, entries in sorted(by_step.items()):
        validate_unstable_replacements(f"{run_id} step {step} selection set", entries, selection_seeds)
    if replacements.get("final"):  # the final checkpoint runs the measurement set (Q-final-cost-set, Table 9.1)
        final_seeds = validate_seed_set("measurement_seeds", evaluation.get("measurement_seeds"))
        validate_unstable_replacements(f"{run_id} final checkpoint", replacements["final"], final_seeds)

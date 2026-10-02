"""Evaluations of Study B's budget-conditioned runs, and the pure functions of equations (10) and (11).

Owner: Study B and literature (Role 5, Hamza Nisar). Contract 1 of plug-in ``study_b`` is
``studyb.evaluation:evaluate_run`` (``pilot.contracts.evaluator_target``).

What the pre-registration fixes (prereg/Preregistration.pdf):

* Table 2.5 "Zero-shot evaluation" (p. 10): "The unseen budget is placed in the observation and 100
  episodes are run with the deterministic policy; no update." "Unseen budgets": {5, 15, 30, 45}
  (``R.UNSEEN_BUDGETS``).
* Table 2.5 "Constraint-satisfaction rate": "The fraction of evaluation episodes whose episodic cost
  is at most the unseen budget (equation 10)." "Violation magnitude": "The mean over evaluation
  episodes of max(cost - budget, 0)." "Adaptation steps": "The smallest few-shot horizon at which the
  satisfaction rate reaches 0.80 (decision); recorded as above the largest horizon if never reached."
* Equation (11) (Part 2.5): the distance of an unseen budget from the training set,
  min over the training set of |d_test - d| (the Continuous arm's set is the interval [10, 40]).
* Part 4.2 (p. 15): "Budget 5 has one nearest arm, Single-10; budget 45 has one, Single-40. Budgets 15
  and 30 are each equidistant from two single-level arms" (``nearest_single_arms``); G3's drops are
  "measured from the arm's satisfaction at its own nearest training level" (the reference budgets of
  ``evaluate_zero_shot``; Q-g3-arms).
* Part 6 G4 (p. 18) and Part 3.6 (p. 14): the Moderate arm "evaluated with each of its three training
  budgets in the observation" (``evaluate_training_budgets``, read by ``python -m pilot g4-measure``).
* Part 3.4 (p. 13) and Appendix B: contract 1 of every run (``evaluate_run``).

How: every evaluation uses Role 2's primitives (``envs.evaluation``: ``load_policy``,
``run_episodes`` with ``budgets=``, which appends the budget with ``append_budget`` after the frozen
normaliser, ``measurement_seeds``, ``selection_seeds``, ``summarise``, ``training_budget_schedule``),
so Study B shares Study A's seeding, frozen normaliser, deterministic mean action and action scaling,
and a budget-conditioned checkpoint gives one cost in contract 1 and in contract 6. Every result
carries per-episode costs and returns (Part 5.3's interquartile mean resamples episodes). Every
checkpoint a call needs is checked before the first episode: a missing or unfit one raises
``CheckpointInvalid`` (a failure), a configuration that does not describe the run
``EvaluationRefused``. Five private helpers of ``envs.evaluation`` are used deliberately, so the two
contract-1 harnesses check and record a run alike: ``_read_config`` and ``_check_config``
(config.json against the spec), ``_checked`` (the episodes against the seeds and budgets asked for),
``_short`` (episodes that ended early) and ``_harness_record`` (what an episode of the harness is).

Result gate (HANDOVER.md section 10): every evaluation first calls
``require_answered("Q-studyb-eval")`` (which holds only while the key is open), whose answer in
Table 9.1 fixes the checkpoint (the final one, or a continuation's horizon checkpoints), the seeds
(``SEED_SET``, the measurement set, at every budget and horizon and in every arm) and, for contract
1, the budgets in the observation (the arm's training budgets in turn,
``envs.evaluation.training_budget_schedule``). Q-adapt-censoring (the censored value of
``adaptation_steps``, largest horizon + 1, answered in Table 9.1) is gated where the ledger receives
``adapt_steps`` (``pilot.enrichment``, pilot/enrichment.py), not here: ``evaluate_fewshot`` reports
``adapt_censored``.

Unstable simulations (Q-mujoco-exception, Table 9.1): ``run_episodes`` replaces an episode that
MuJoCo reports unstable by a reserve episode with the same budget. Every result keeps its canonical
seed lists and reports, per evaluation call (one budget, one horizon at one budget, or one checkpoint
in contract 1), the replacements as ``unstable_replacements``
(``envs.evaluation.unstable_replacements``).
"""

from __future__ import annotations

import math
import numbers
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from configs import registered as R
from envs import evaluation as E
from pilot import contracts
from pilot.errors import require_answered
from pilot.manifest import CONTINUATION_GROUPS, RunSpec
from pilot.rundir import run_steps_per_epoch
from studyb import conditioning

GATES = ("Q-studyb-eval",)  # the result gate of every Study B evaluation (configs/registered.py PENDING)
SEED_SET = "measurement"  # Q-studyb-eval (Table 9.1): Table 2.1's measurement set at every budget and horizon
STUDYB_EVALUATION_VERSION = "1"  # recorded in every result beside the harness's own version
# Why a negative episodic cost cannot be one: the definition (Table 2.1) and the library that makes each
# step's cost 0 or 1 (CostConf.constrain_indicator, True by default, applied in builder.py's step).
NEGATIVE_COST = (
    'an episodic cost is negative: Table 2.1 defines it as "The sum over an episode of the per-step cost the task'
    ' returns", and Safety-Gymnasium 0.4.1 makes each step\'s cost 0 or 1 (CostConf.constrain_indicator, True by'
    " default, safety_gymnasium/bases/base_task.py; applied in safety_gymnasium/builder.py)"
)


# ---------------------------------------------------------------------------
# Pure functions: equations (10) and (11), violation magnitude, adaptation steps
# ---------------------------------------------------------------------------


def _number(value: Any, what: str) -> float:
    try:  # an int too large for a float raises OverflowError: refused as any other non-finite number
        number = float(value) if isinstance(value, numbers.Real) and not isinstance(value, bool) else math.nan
    except OverflowError:
        number = math.nan
    if not math.isfinite(number):
        raise ValueError(f"{what} must be a finite number, got {value!r}")
    return number


def _costs(costs: Sequence[float]) -> list[float]:
    if isinstance(costs, (str, bytes)):
        raise ValueError(f"costs must be a sequence of numbers, got {costs!r}")
    values = [_number(c, "an episodic cost") for c in costs]
    if not values:
        raise ValueError("no episodes: a rate over zero episodes is undefined")
    if min(values) < 0.0:
        # Table 2.1 defines the episodic cost only as a sum of the task's per-step cost; that each step's
        # cost is 0 or 1 (so a sum cannot be negative) is Safety-Gymnasium 0.4.1's, not the pre-registration's.
        raise ValueError(NEGATIVE_COST)
    return values


def satisfaction_rate(costs: Sequence[float], budget: float) -> float:
    """Equation (10): SR(d) = (1/n) sum over the n episodes of 1[C_e <= d] (a cost equal to d satisfies)."""
    values = _costs(costs)
    d = _number(budget, "a budget")
    return sum(1 for c in values if c <= d) / len(values)


def violation_magnitude(costs: Sequence[float], budget: float) -> float:
    """Table 2.5 "Violation magnitude": the mean over the episodes of max(cost - budget, 0)."""
    values = _costs(costs)
    d = _number(budget, "a budget")
    return float(statistics.fmean(max(c - d, 0.0) for c in values))


def _training_set(arm: str) -> tuple[float, ...] | None:
    if arm not in R.STUDY_B_ARMS:
        raise ValueError(f"unknown Study B arm {arm!r}; choose from {tuple(R.STUDY_B_ARMS)}")
    levels = R.STUDY_B_ARMS[arm]
    return None if levels is None else tuple(float(b) for b in levels)


def distance(d_test: float, arm: str) -> float:
    """Equation (11): min over the arm's training set of |d_test - d|; for Continuous the set is the
    interval ``R.CONTINUOUS_RANGE``, so a budget inside it is at distance 0."""
    d = _number(d_test, "a budget")
    levels = _training_set(arm)
    if levels is None:
        lo, hi = (float(x) for x in R.CONTINUOUS_RANGE)
        return max(lo - d, 0.0, d - hi)
    return min(abs(d - level) for level in levels)


def nearest_training_level(arm: str, d_test: float) -> float:
    """The arm's training level nearest to ``d_test`` by equation (11) (the lower one on a tie);
    for Continuous, ``d_test`` clipped to its range (the nearest point of the interval)."""
    d = _number(d_test, "a budget")
    levels = _training_set(arm)
    if levels is None:
        lo, hi = (float(x) for x in R.CONTINUOUS_RANGE)
        return min(max(d, lo), hi)
    return min(levels, key=lambda level: (abs(d - level), level))


def nearest_single_arms(d_test: float) -> tuple[str, ...]:
    """The single-level arms at the least equation (11) distance from ``d_test``, by increasing level.

    Part 4.2: 5 -> (Single-10,); 15 -> (Single-10, Single-20); 30 -> (Single-20, Single-40); 45 ->
    (Single-40,). Where two are equidistant the comparator of G2 is chosen by the analysis ("whichever
    of the two has the higher satisfaction"); both are returned here.
    """
    d = _number(d_test, "a budget")
    singles = {arm: levels[0] for arm, levels in R.STUDY_B_ARMS.items() if levels is not None and len(levels) == 1}
    best = min(abs(d - level) for level in singles.values())
    return tuple(sorted((arm for arm, level in singles.items() if abs(d - level) == best), key=singles.get))


def _horizons(horizons: Sequence[int]) -> list[int]:
    if isinstance(horizons, (str, bytes)):
        raise ValueError(f"horizons must be a sequence of step counts, got {horizons!r}")
    try:
        items = list(horizons)
    except TypeError:  # an int or another non-iterable: refused as ValueError, as documented
        raise ValueError(f"horizons must be a sequence of step counts, got {horizons!r}") from None
    values = []
    for h in items:
        if isinstance(h, bool) or not isinstance(h, numbers.Integral) or int(h) <= 0:
            raise ValueError(f"a horizon must be a positive whole number of steps, got {h!r}")
        values.append(int(h))
    if not values or values != sorted(set(values)):
        raise ValueError(f"horizons must be distinct and increasing, got {values}")
    return values


def censored_steps(horizons: Sequence[int]) -> int:
    """The adaptation steps recorded when the target is never reached: the largest horizon + 1.

    Q-adapt-censoring (Table 9.1): Table 2.5 records it "as above the largest horizon", Part 4.2
    censors it "at the largest horizon"; one step above the largest horizon evaluated (1,000,001 for
    the registered horizons, 200,001 under Part 6.1 cut 3) keeps it distinct from reaching the target
    at the largest horizon. Readers decode censoring as ``steps > max(horizons)``.
    """
    return max(_horizons(horizons)) + 1


def _horizon_key(h: Any) -> int:
    """A key of ``rates_by_horizon`` as a horizon: a whole number of steps, or its text or float form
    (JSON keys are read back as text); a bool, a fraction or anything else is refused."""
    if isinstance(h, bool):
        raise ValueError(f"a horizon must be a whole number of steps, got {h!r}")
    if isinstance(h, numbers.Integral):
        return int(h)
    try:
        value = float(h) if isinstance(h, (str, numbers.Real)) else math.nan
    except ValueError:
        value = math.nan
    if not math.isfinite(value) or value != int(value):
        raise ValueError(f"a horizon must be a whole number of steps, got {h!r}")
    return int(value)


def adaptation_steps(rates_by_horizon: Mapping[int, float], horizons: Sequence[int]) -> int:
    """Table 2.5 "Adaptation steps": the smallest horizon whose satisfaction rate reaches
    ``R.SATISFACTION_TARGET`` ("reaches 0.80": a rate of at least 0.80); ``censored_steps(horizons)``
    if none does. ``rates_by_horizon`` must hold a rate in [0, 1] for exactly ``horizons``. There is
    no horizon 0: a continuation that already satisfies the target records its first horizon.
    """
    order = _horizons(horizons)
    if not isinstance(rates_by_horizon, Mapping):
        raise ValueError(f"rates_by_horizon must map horizons to rates, got {rates_by_horizon!r}")
    rates: dict[int, float] = {}
    try:
        for h, r in rates_by_horizon.items():
            key = _horizon_key(h)
            if key in rates:
                raise ValueError(f"horizon {key} is given twice (as {h!r} too)")
            rates[key] = _number(r, f"the rate at horizon {h}")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"rates_by_horizon: {exc}") from None
    if sorted(rates) != order:
        raise ValueError(f"rates are given for horizons {sorted(rates)}, not for {order}")
    outside = {h: r for h, r in rates.items() if not 0.0 <= r <= 1.0}
    if outside:
        raise ValueError(f"satisfaction rates must lie in [0, 1], got {outside}")
    for h in order:
        if rates[h] >= R.SATISFACTION_TARGET:
            return h
    return censored_steps(order)


def reference_budgets(arm: str) -> tuple[float, ...]:
    """The seen budgets that zero-shot evaluation adds for G3's drops (answered in Table 9.1, Q-g3-arms): the arm's
    training levels, or the ends of ``R.CONTINUOUS_RANGE`` for Continuous (the nearest training
    levels of 5 and 45 by equation (11))."""
    levels = _training_set(arm)
    return tuple(float(x) for x in R.CONTINUOUS_RANGE) if levels is None else tuple(sorted(levels))


# ---------------------------------------------------------------------------
# Loading and running (envs.evaluation primitives)
# ---------------------------------------------------------------------------


def _gate(what: str) -> None:
    require_answered(*GATES, what=what)


def _run(spec: Any, plugin: str, what: str) -> RunSpec:
    """The spec as a RunSpec of ``plugin``; EvaluationRefused otherwise (a configuration problem)."""
    if isinstance(spec, RunSpec):
        run = spec
    elif isinstance(spec, Mapping):
        try:
            run = RunSpec.from_dict(spec)
        except (TypeError, ValueError, KeyError) as exc:
            raise E.EvaluationRefused(f"not a valid run spec: {exc}") from exc
    else:
        raise E.EvaluationRefused(f"a run spec (RunSpec.to_dict()) is required, got {type(spec).__name__}")
    if run.study != "B" or run.plugin != plugin:
        raise E.EvaluationRefused(f"{run.run_id}: {what} is for Study B plug-in {plugin!r}, not study {run.study!r} "
                                  f"plug-in {run.plugin!r}")
    if run.arm not in R.STUDY_B_ARMS:
        raise E.EvaluationRefused(f"{run.run_id}: arm {run.arm!r} is not a Study B arm (Table 3.4)")
    return run


def _check_studyb_cfgs(config: Mapping[str, Any], directory: Path, run: RunSpec) -> None:
    """The run's ``studyb_cfgs`` must use the budget divisor the harness appends with and the multiplier
    keys of the spec (else the feature the policy reads would not be the budget it was trained on)."""
    studyb = config.get("studyb_cfgs")
    if not isinstance(studyb, Mapping):
        raise E.EvaluationRefused(f"{directory}: config.json has no studyb_cfgs; the run was not trained by "
                                  "studyb.conditioning")
    if studyb.get("budget_divisor") != R.BUDGET_OBSERVATION_DIVISOR:
        raise E.EvaluationRefused(f"{directory}: the run appended budget / {studyb.get('budget_divisor')!r}; the harness "
                                  f"appends budget / {R.BUDGET_OBSERVATION_DIVISOR} (Table 2.5)")
    try:
        kind, keys = conditioning.level_kind(run), [float(k) for k in conditioning.level_keys(run)]
    except ValueError as exc:
        raise E.EvaluationRefused(f"{run.run_id}: {exc}") from exc
    try:
        given = [float(k) for k in studyb.get("level_keys") or ()]
    except (TypeError, ValueError) as exc:
        raise E.EvaluationRefused(f"{directory}: config.json studyb_cfgs.level_keys {studyb.get('level_keys')!r} "
                                  "are not numbers") from exc
    if studyb.get("kind") != kind or given != keys:
        raise E.EvaluationRefused(f"{directory}: config.json studyb_cfgs ({studyb.get('kind')!r}, keys "
                                  f"{studyb.get('level_keys')!r}) do not describe {run.run_id} ({kind!r}, keys {keys})")


def _check_run_config(directory: Path, run: RunSpec) -> None:
    """config.json checked against the spec before any checkpoint is looked for, in the order of Role 2's
    ``envs.evaluation.evaluate_run`` (its ``_read_config`` and ``_check_config``, which ``load_policy``
    repeats for every checkpoint), then ``_check_studyb_cfgs``. A directory whose configuration
    describes another run is so refused (``EvaluationRefused``, exit 5: the run stays trained) by both
    contract-1 harnesses alike, never reported as missing checkpoints (``CheckpointInvalid``, a failure).
    """
    config = E._read_config(directory)
    E._check_config(config, directory, run)
    _check_studyb_cfgs(config, directory, run)


def _load(directory: Path, step: int, run: RunSpec) -> E.Policy:
    """``envs.evaluation.load_policy`` for a budget-conditioned run trained by ``studyb.conditioning``.

    Besides the harness's own checks (config.json against the spec, the checkpoint file, an actor with
    one more input than the task's observation, the normaliser of the task's size), the run's
    ``studyb_cfgs`` are checked (``_check_studyb_cfgs``).
    """
    policy = E.load_policy(directory, step, run)
    if not policy.budget_conditioned:  # load_policy already ties this to the plug-in; kept as the contract's guard
        raise E.EvaluationRefused(f"{policy.checkpoint}: the actor takes no budget input")
    _check_studyb_cfgs(policy.config, directory, run)
    return policy


def _episodes(policy: E.Policy, seeds: Sequence[int], budgets: Sequence[float], what: str) -> list[E.EpisodeResult]:
    """``run_episodes`` with one budget per seed, its episodes checked against the seeds and budgets
    asked for by Role 2's own check (``envs.evaluation._checked``; a ContractError otherwise)."""
    seeds, budgets = list(seeds), list(budgets)
    return E._checked(E.run_episodes(policy, seeds, budgets=budgets), seeds, what, budgets=budgets)


def _at_budget(policy: E.Policy, budget: float, what: str) -> dict[str, Any]:
    """100 episodes of the measurement set with ``budget`` in every observation: equation (10) and the
    rest, and the replacements of unstable episodes (Q-mujoco-exception)."""
    seeds = E.measurement_seeds()
    results = _episodes(policy, seeds, [budget] * len(seeds), what)
    costs = [r.cost for r in results]
    mean_cost, mean_return = E.summarise(results)
    return {
        "mean_cost": mean_cost,
        "mean_return": mean_return,
        "satisfaction": satisfaction_rate(costs, budget),
        "violation": violation_magnitude(costs, budget),
        "episode_costs": costs,
        "episode_returns": [r.ret for r in results],
        "unstable_replacements": E.unstable_replacements(results),
        "short_episodes": E._short(results),
    }


def _harness(policy: E.Policy, checkpoint_rule: str, seed_rule: str = SEED_SET) -> dict[str, Any]:
    """How the result was measured: Role 2's record of an episode (``envs.evaluation._harness_record``,
    whose normaliser follows the policy's) and Study B's own fields; the checkpoint rule is prose, the
    seed rule the set's name (``SEED_SET``) unless given (contract 1 gives prose)."""
    normalised = policy.normalizer_state is not None
    where = "after the normaliser" if normalised else "to the raw observation (no normaliser)"
    return {
        "harness": "envs.evaluation",
        **E._harness_record(normalised),
        "studyb_version": STUDYB_EVALUATION_VERSION,
        "budget_feature": f"budget / {R.BUDGET_OBSERVATION_DIVISOR} appended {where} (append_budget)",
        "checkpoint_rule": checkpoint_rule,
        "seed_rule": seed_rule,
    }


def _relative(policy: E.Policy) -> str:
    return policy.checkpoint.relative_to(policy.omnisafe_dir).as_posix()


def _maps(blocks: Mapping[Any, Mapping[str, Any]]) -> dict[str, Any]:
    """{budget: block} as one map per quantity, the form g4-measure reads."""
    names = ("mean_cost", "satisfaction", "violation", "mean_return", "episode_costs", "episode_returns",
             "unstable_replacements")
    return {name: {key: block[name] for key, block in blocks.items()} for name in names}


# ---------------------------------------------------------------------------
# Go condition G4 (Part 6; Part 3.6)
# ---------------------------------------------------------------------------


def evaluate_training_budgets(omnisafe_dir: str | Path, spec: Mapping[str, Any] | RunSpec) -> dict[str, Any]:
    """Go condition G4: the final checkpoint with each training budget in the observation.

    Part 6 G4: "the Moderate arm, evaluated with each of its three training budgets in the
    observation, shows mean costs that are ordered with the budgets (lowest at 10, highest at 40) and
    each at most its budget plus 2.5"; Part 3.6 measures "the Moderate arm's satisfaction rates". For
    each level b of ``spec['training_levels']``: 100 episodes of the measurement set with b in every
    observation, at the final checkpoint (``spec['total_steps']``; Q-studyb-eval).

    Returns what ``python -m pilot g4-measure`` validates (pilot/__main__.py ``_cmd_g4_measure``):
    ``episodes`` == ``R.EVAL_EPISODES``, ``mean_cost`` and ``satisfaction`` keyed by exactly the float
    training levels, every value a finite real number (rates in [0, 1]), every map keyed by one type
    (it is written with ``sort_keys=True, allow_nan=False``), ``seeds`` ==
    ``envs.evaluation.measurement_seeds()`` and the pilot registry's measurement set; besides them
    ``violation``, ``mean_return``, the per-episode ``episode_costs`` and ``episode_returns``, the
    ``unstable_replacements`` of each budget (Q-mujoco-exception), ``run_id``, ``arm``, ``step``,
    ``checkpoint``, ``seed_set``, ``budgets``, ``short_episodes`` and ``harness``. Ordering and the
    2.5 margin are judged by the go report (Q-g4-level). EvaluationRefused for a Continuous arm (it
    has no training levels) or a spec that does not describe the run; CheckpointInvalid if the final
    checkpoint is missing or does not fit.
    """
    _gate("go condition G4's measurement of the Moderate arm at its training budgets (Part 6 G4; Part 3.6)")
    run = _run(spec, "study_b", "evaluate_training_budgets")
    if run.training_levels is None:
        raise E.EvaluationRefused(f"{run.run_id}: the Continuous arm has no training levels to evaluate (Table 2.5)")
    levels = sorted(float(b) for b in run.training_levels)
    _check_run_config(Path(omnisafe_dir), run)  # a configuration problem is a refusal, whatever the checkpoints hold
    policy = _load(Path(omnisafe_dir), run.total_steps, run)
    blocks = {b: _at_budget(policy, b, f"{run.run_id} final checkpoint at training budget {b:g}") for b in levels}
    return {
        "run_id": run.run_id,
        "arm": run.arm,
        "episodes": R.EVAL_EPISODES,
        "step": policy.step,
        "checkpoint": _relative(policy),
        "seed_set": SEED_SET,
        "seeds": E.measurement_seeds(),
        "budgets": levels,
        **_maps(blocks),
        "short_episodes": sum(block["short_episodes"] for block in blocks.values()),
        "harness": _harness(policy, "final (Q-studyb-eval)"),
    }


# ---------------------------------------------------------------------------
# Zero-shot (Table 2.5; Part 4.2)
# ---------------------------------------------------------------------------


def evaluate_zero_shot(omnisafe_dir: str | Path, spec: Mapping[str, Any] | RunSpec) -> dict[str, Any]:
    """Zero-shot satisfaction of a Study B run: the final checkpoint at the unseen and reference budgets.

    Table 2.5 "Zero-shot evaluation": each unseen budget of ``R.UNSEEN_BUDGETS`` "is placed in the
    observation and 100 episodes are run with the deterministic policy; no update" (measurement
    seeds, the final checkpoint; Q-studyb-eval). The reference budgets (``reference_budgets``: the
    arm's training levels, or 10 and 40 for Continuous; Q-g3-arms) are run the same way, for G3's
    drops "measured from the arm's satisfaction at its own nearest training level" (Part 4.2).

    Returns, keyed by float budget, ``mean_cost``, ``satisfaction`` (equation (10)), ``violation``,
    ``mean_return``, ``episode_costs``, ``episode_returns`` and ``unstable_replacements``
    (Q-mujoco-exception) for every budget; ``distance`` (equation (11)) and
    ``nearest_training_level`` for the unseen budgets; ``sr_zero`` {unseen budget: rate} in the
    ledger's form (Appendix B ``sr_zero[budget]``); ``unseen_budgets``,
    ``reference_budgets``, ``episodes``, ``step``, ``checkpoint``, ``seed_set``, ``seeds``; ``run_id``,
    ``arm``, ``short_episodes`` and ``harness``.
    """
    _gate("the zero-shot evaluation of a Study B run (Table 2.5; sr_zero)")
    run = _run(spec, "study_b", "evaluate_zero_shot")
    unseen = [float(b) for b in R.UNSEEN_BUDGETS]
    reference = [float(b) for b in reference_budgets(run.arm)]
    if set(unseen) & set(reference):  # impossible for the registered sets; the roles below would collide
        raise E.EvaluationRefused(f"{run.run_id}: budgets {sorted(set(unseen) & set(reference))} are both unseen and seen")
    _check_run_config(Path(omnisafe_dir), run)  # a configuration problem is a refusal, whatever the checkpoints hold
    policy = _load(Path(omnisafe_dir), run.total_steps, run)
    blocks = {}
    for b in unseen + reference:
        role = "unseen" if b in unseen else "reference"
        blocks[b] = _at_budget(policy, b, f"{run.run_id} final checkpoint at {role} budget {b:g}")
    return {
        "run_id": run.run_id,
        "arm": run.arm,
        "episodes": R.EVAL_EPISODES,
        "step": policy.step,
        "checkpoint": _relative(policy),
        "seed_set": SEED_SET,
        "seeds": E.measurement_seeds(),
        "unseen_budgets": unseen,
        "reference_budgets": reference,
        **_maps(blocks),
        "distance": {b: distance(b, run.arm) for b in unseen},
        "nearest_training_level": {b: nearest_training_level(run.arm, b) for b in unseen},
        "sr_zero": {b: blocks[b]["satisfaction"] for b in unseen},
        "short_episodes": sum(block["short_episodes"] for block in blocks.values()),
        "harness": _harness(policy, "final (Q-studyb-eval)"),
    }


# ---------------------------------------------------------------------------
# Few-shot (Table 2.5 "Few-shot adaptation", "Adaptation steps")
# ---------------------------------------------------------------------------


def evaluate_fewshot(omnisafe_dir: str | Path, spec: Mapping[str, Any] | RunSpec) -> dict[str, Any]:
    """One few-shot continuation at each of its horizons, with its unseen budget in the observation.

    Table 2.5: "satisfaction is measured at 200,000, 500,000 and 1,000,000 steps ... The three
    horizons are read from one continuation, not three." ``omnisafe_dir`` is the continuation's own
    OmniSafe directory and ``spec`` its spec; the horizons and the budget are read from
    ``spec['params']`` (Part 6.1 cut 3 may leave one horizon). At each horizon h the checkpoint saved
    after h continuation steps is run for 100 episodes of the measurement set with the budget in every
    observation (Q-studyb-eval). Every horizon's checkpoint is loaded and checked before the first
    episode.

    Returns ``by_horizon`` {h: {step, checkpoint, mean_cost, mean_return, satisfaction, violation,
    episode_costs, episode_returns, unstable_replacements, short_episodes}}; ``sr_fewshot``
    {``pilot.contracts.fewshot_key(budget, h)``: rate} (Appendix B ``sr_fewshot[budget, horizon]``,
    e.g. "5.0_200000"); ``adapt_steps`` {budget: ``adaptation_steps``} and ``adapt_censored`` (True
    when no horizon reached the target, so the value is ``censored_steps``: the enrichment gates that
    case on Q-adapt-censoring); ``budget``, ``horizons``, ``parent_run_id``, ``parent_step``,
    ``episodes``, ``seed_set``, ``seeds``; ``run_id``, ``arm``, ``short_episodes`` and ``harness``.
    """
    _gate("the few-shot evaluation of a Study B continuation (Table 2.5; sr_fewshot, adapt_steps)")
    run = _run(spec, "study_b_fewshot", "evaluate_fewshot")
    if run.group not in CONTINUATION_GROUPS:
        raise E.EvaluationRefused(f"{run.run_id} is not a continuation (group {run.group!r})")
    params = dict(run.params or {})
    try:
        budget = _number(params.get("budget"), "params['budget']")
        horizons = _horizons(params.get("horizons") or ())
    except ValueError as exc:
        raise E.EvaluationRefused(f"{run.run_id}: {exc}") from None
    if budget not in R.UNSEEN_BUDGETS:
        raise E.EvaluationRefused(f"{run.run_id}: budget {budget:g} is not an unseen budget {list(R.UNSEEN_BUDGETS)}")
    unknown = [h for h in horizons if h not in R.FEWSHOT_HORIZONS]
    if unknown:  # the ledger's sr_fewshot accepts the registered horizons only (results/ledger_schema.py)
        raise E.EvaluationRefused(f"{run.run_id}: horizons {unknown} are not few-shot horizons {list(R.FEWSHOT_HORIZONS)}")
    if horizons[-1] > run.total_steps:
        raise E.EvaluationRefused(f"{run.run_id}: horizon {horizons[-1]} lies beyond the continuation's {run.total_steps} steps")
    directory = Path(omnisafe_dir)
    _check_run_config(directory, run)  # a configuration problem is a refusal, whatever the checkpoints hold
    policies = {h: _load(directory, h, run) for h in horizons}  # every checkpoint checked before the first episode
    blocks = {}
    for h in horizons:
        block = _at_budget(policies[h], budget, f"{run.run_id} horizon {h} at budget {budget:g}")
        blocks[h] = {"step": policies[h].step, "checkpoint": _relative(policies[h]), **block}
    rates = {h: blocks[h]["satisfaction"] for h in horizons}
    steps = adaptation_steps(rates, horizons)
    return {
        "run_id": run.run_id,
        "parent_run_id": params.get("parent_run_id"),
        "parent_step": params.get("parent_step"),
        "arm": run.arm,
        "budget": budget,
        "horizons": horizons,
        "episodes": R.EVAL_EPISODES,
        "seed_set": SEED_SET,
        "seeds": E.measurement_seeds(),
        "by_horizon": blocks,
        "sr_fewshot": {contracts.fewshot_key(budget, h): rates[h] for h in horizons},
        "adapt_steps": {budget: steps},
        "adapt_censored": steps > max(horizons),
        "short_episodes": sum(block["short_episodes"] for block in blocks.values()),
        "harness": _harness(policies[horizons[-1]], "the continuation's checkpoint at each horizon (Q-studyb-eval)"),
    }


# ---------------------------------------------------------------------------
# Contract 1 for plug-in study_b (pilot.contracts.evaluator_target)
# ---------------------------------------------------------------------------


def evaluate_run(omnisafe_dir: str | Path, spec: Mapping[str, Any] | RunSpec) -> dict[str, Any]:
    """Contract 1 (pilot/contracts.py) for a Study B training run: what Part 3.4 asks of every run.

    ``selection[step] = [mean cost, mean return]`` over ``selection_seeds()`` for exactly the run's
    last ten checkpoints (``pilot.contracts.selection_window``), and ``final_cost``/``final_return``
    at the final checkpoint over ``measurement_seeds()`` (Q-final-cost-set, Table 9.1, as Role 2's
    harness; ``final_selection_cost`` keeps the selection-set value there). Episode i of every set
    carries ``envs.evaluation.training_budget_schedule(spec)[i]`` in its observation (Q-studyb-eval:
    "the arm's i-th training budget in turn"; Continuous: the midpoints of 100 equal parts of
    [10, 40]), so a checkpoint's selection-set cost equals contract 6's ``evaluate_checkpoint`` cost
    of it. No registered Study B analysis reads these fields (Q-studyb-eval: they "are descriptive and
    enter no Study B hypothesis"; Study B's outcomes are ``evaluate_zero_shot`` and
    ``evaluate_fewshot``); the ledger needs them for every run.

    Returns contract 1's keys plus ``final_selection_cost``, ``final_step``, ``final_seed_set``,
    ``measurement_seeds``, ``studyb_budgets`` (the 100 per-episode budgets), the per-episode
    ``episode_costs`` and ``episode_returns`` ({'final': [...], 'selection': {step: [...]}}),
    ``unstable_replacements`` in the same form (Q-mujoco-exception), ``short_episodes`` and
    ``harness``. Before any checkpoint is looked for, config.json is checked against the spec
    (``_check_run_config``; else ``EvaluationRefused``). Every Part 3.4 checkpoint must exist, the
    last ten must be the window's (a broken checkpoint set, such as an off-grid checkpoint inside the
    final 2,000,000 steps, is the run's saved output), and every window checkpoint is loaded and
    checked before the first episode (else ``CheckpointInvalid``, as Role 2's harness); a spec too
    short for the window, or a total off the grid while Q-selection-window is open, is refused.
    """
    _gate("contract 1 of a budget-conditioned run (Part 3.4; the arm's training budgets in turn)")
    run = _run(spec, "study_b", "studyb.evaluation.evaluate_run")
    if run.group in CONTINUATION_GROUPS:
        raise E.EvaluationRefused(f"{run.run_id} is a continuation ({run.group}); it is not evaluated by contract 1")
    directory = Path(omnisafe_dir)
    _check_run_config(directory, run)  # a configuration problem is a refusal, whatever the checkpoints hold
    budgets = E.training_budget_schedule(run)  # Q-studyb-eval (Table 9.1): the arm's training budgets in turn
    try:
        steps_per_epoch = run_steps_per_epoch(directory)
    except (OSError, ValueError) as exc:
        raise E.EvaluationRefused(f"{directory}: {exc}") from exc
    if steps_per_epoch != R.STEPS_PER_EPOCH:
        raise E.EvaluationRefused(f"{directory}: steps_per_epoch {steps_per_epoch} is not the registered "
                                  f"{R.STEPS_PER_EPOCH} (Table 3.1), which contract 1's checkpoint steps assume")
    steps = contracts.checkpoint_steps(directory, R.STEPS_PER_EPOCH)
    missing = sorted(set(contracts.expected_checkpoint_steps(run)) - set(steps))
    if missing:
        raise E.CheckpointInvalid(f"{run.run_id}: checkpoints required by Part 3.4 are missing at steps {missing}")
    beyond = [s for s in steps if s > run.total_steps]
    if beyond:
        raise E.CheckpointInvalid(f"{run.run_id}: checkpoints beyond the run's {run.total_steps} steps at steps {beyond}")
    try:
        window = contracts.selection_window(steps, run.total_steps)
    except contracts.ContractError as exc:
        # a total off the grid while Q-selection-window is open, or too short: the spec, refused
        # (as Role 2's envs.evaluation.evaluate_run)
        if (isinstance(exc, contracts.SelectionWindowPending)
                or len(contracts.expected_checkpoint_steps(run)) < R.SELECTION_WINDOW_CHECKPOINTS):
            raise E.EvaluationRefused(f"{run.run_id}: {exc}") from exc
        # an off-grid checkpoint inside the window (a broken checkpoint set): the run's saved output
        raise E.CheckpointInvalid(f"{run.run_id}: {exc}") from exc
    final = run.total_steps
    policies = {step: _load(directory, step, run) for step in window}

    selection: dict[int, list[float]] = {}
    costs: dict[int, list[float]] = {}
    returns: dict[int, list[float]] = {}
    replaced: dict[int, list[dict[str, Any]]] = {}
    short = 0
    selection_set, measurement_set = E.selection_seeds(), E.measurement_seeds()
    for step in window:
        results = _episodes(policies[step], selection_set, budgets, f"{run.run_id} step {step} selection set")
        selection[step] = list(E.summarise(results))
        costs[step] = [r.cost for r in results]
        returns[step] = [r.ret for r in results]
        replaced[step] = E.unstable_replacements(results)
        short += E._short(results)
    final_results = _episodes(policies[final], measurement_set, budgets, f"{run.run_id} final checkpoint measurement set")
    final_cost, final_return = E.summarise(final_results)
    return {
        "final_cost": final_cost,
        "final_return": final_return,
        "final_selection_cost": selection[final][0],
        "final_step": final,
        "final_seed_set": "measurement",  # Q-final-cost-set
        "selection": selection,
        "episodes": R.EVAL_EPISODES,
        "selection_seeds": selection_set,
        "measurement_seeds": measurement_set,
        "studyb_budgets": list(budgets),
        "episode_costs": {"final": [r.cost for r in final_results], "selection": costs},
        "episode_returns": {"final": [r.ret for r in final_results], "selection": returns},
        "unstable_replacements": {"final": E.unstable_replacements(final_results), "selection": replaced},
        "short_episodes": short + E._short(final_results),
        "harness": _harness(policies[final], "the last ten checkpoints and the final one (contract 1)",
                            "selection at the last ten checkpoints; measurement at the final one (Q-final-cost-set)"),
    }

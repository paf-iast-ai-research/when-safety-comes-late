"""Budget-conditioned PPO-Lagrangian for Study B, and its few-shot continuation.

Owner: Study B and literature (Role 5, Hamza Nisar). This module provides the plug-ins ``study_b``
(``make_algorithm``, the seven arms of Table 3.4) and ``study_b_fewshot`` (``make_fewshot_algorithm``,
the continuations of Part 3.3) of ``pilot.manifest.PLUGINS``; the launcher calls
``factory(spec.task, cfgs, spec)``.
A reference of the form Study B spec n.m is to the build's Study B specification, which is kept
outside this repository.

What the pre-registration fixes (prereg/Preregistration.pdf):

* Table 2.5 "Budget conditioning" (p. 10): "The active budget divided by 100 is appended to the
  observation vector. During training each episode draws its budget uniformly at random from the
  arm's training set. One multiplier is kept per training level and is updated only from episodes
  run at that level, by equation (4)."
* Table 2.5 "Training set of levels": the arms of ``R.STUDY_B_ARMS``; Continuous: "a budget drawn
  uniformly from [10, 40] at each episode, with one multiplier per 5-unit bin of the range".
* Table 3.1 "Networks" (p. 12): "Study B appends one input, the normalised budget, to the actor and
  critic inputs."
* Equation (3) (Part 2.5), the surrogate (A_R - lambda A_C) / (1 + lambda), here per sample with
  the multiplier of the sample's level; equation (4), lambda <- max(0, lambda + eta (J_C - d)), per
  level, "applied there through the optimiser named in the configuration" (OmniSafe's ``Lagrange``,
  lr 0.035, from 0.001; the optimiser is taken from the pinned PPOLag configuration, Adam, and only
  checked to be named; Table 9.1, Q-multiplier-adam).
* Part 3.4 (p. 13): "The multiplier, or in Study B every per-level multiplier, is logged at every
  epoch." Contract 3 (pilot/contracts.py): ``Metrics/LagrangeMultiplier/level_{b:g}``; the
  Continuous arm names each bin by its lower edge.
* Part 5.1 (p. 15): "seed k of one arm and seed k of the other share their network initialisation
  and layout sequence and are treated as a pair". The budget draws therefore come from a generator
  of their own (``budget_generator``), never from a global random stream (HANDOVER.md section 10), so
  seed k builds the same networks and environment in every arm.
* Table 2.5 "Few-shot adaptation": "One continuation of 1,000,000 steps per unseen budget, under that
  budget with a fresh multiplier at 0.001; satisfaction is measured at 200,000, 500,000 and 1,000,000
  steps". The horizons are read from the spec (Part 6.1 cut 3 may shorten them; the shortened
  continuation keeps the full one's actor schedule, ``params['lr_schedule_steps']``, Table 9.1,
  Q-continuations).

How (Study B spec sections 6.1 to 6.8):

* ``BudgetConditionedAdapter`` appends the feature with ``envs.evaluation.append_budget`` (the
  function the evaluation harness uses, so training and evaluation see bit-identical features)
  after the whole wrapper chain, OmniSafe's own Sauté precedent (omnisafe/adapter/saute_adapter.py):
  the observation normaliser stays at the task's size and never sees the budget
  (Q-budget-normalisation); the actor, both critics and the buffer take one more input, because
  OmniSafe sizes them from the adapter's ``observation_space``. Its ``rollout`` is OmniSafe
  0.5.0's (omnisafe/adapter/onpolicy_adapter.py) with every change marked ``# CHANGE``.
* ``BudgetConditionedPPOLag`` is ``FullStateCheckpointMixin`` over OmniSafe's PPO with PPOLag's four
  hooks rewritten per level: one OmniSafe ``Lagrange`` per level in ``_level_lagranges`` (a PPOLag
  subclass would register the plain ``Metrics/LagrangeMultiplier`` column, which a per-level
  algorithm leaves NaN and the launcher then excludes as a non-finite multiplier, Study B spec E2
  and E3); the per-sample penalty is read from the float32 budget feature stored in the buffer's
  observations, because OmniSafe's ``_compute_adv_surrogate`` never sees the observations.
* J_C of a level (Q-level-jc): the mean cost of that level's episodes among OmniSafe's
  ``Metrics/EpCost`` window (its length read from the logger: 50 in OmniSafe 0.5.0), averaged
  exactly as OmniSafe's logger averages that window; a level with no episode there is not updated.
  A single-level arm therefore updates from the same number PPOLag does (Q-jc-window).
* Continuous (Q-continuous-bins): six bins [10, 15), ..., [30, 35), [35, 40], read from the float32
  feature; a bin's budget d is the mean budget of its window episodes, so its update minimises the
  loss -lambda * mean(C_e - b_e).

Answered questions (docs/DECISIONS.md, 2026-10-02): run gates carried by the specs
(``pilot.manifest.run_gate_keys``), each holding only while its key is open: Q-budget-normalisation,
Q-level-jc, Q-continuous-bins, Q-jc-window, Q-continuations (few-shot), and the scheduling gates
Q-studyb-order (non-pilot training runs) and Q-search-before-pilot (the pilot's Study B runs), which
this module does not implement. Nothing here calls a result gate; training runs only when the scheduler
launched the spec.
"""

from __future__ import annotations

import collections
import functools
import math
import numbers
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from configs import registered as R
from envs.evaluation import append_budget
from pilot import contracts, dependencies, rundir
from pilot.algorithms import FullStateCheckpointMixin
from pilot.errors import RunRefused
from pilot.manifest import CONTINUATION_GROUPS, RunSpec

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The spawn key of the budget generator: numpy.random.default_rng([run seed, BUDGET_STREAM]). An
# implementation constant, not a registered number and not an open question: any fixed stream draws
# the i.i.d. uniform budgets Table 2.5 asks for; fixing it makes the draws reproducible.
BUDGET_STREAM = 1
# Cost units. A discrete budget feature (float32 of b / 100) recovers b within about 3e-6; a feature
# farther than this from every level cannot come from the arm's own draws, so it is a bug, refused.
LEVEL_TOLERANCE = 1e-3

DISCRETE = "discrete"  # the Single, Sparse, Moderate and Dense arms: one multiplier per level
CONTINUOUS = "continuous"  # the Continuous arm: one multiplier per 5-unit bin
FEWSHOT = "fewshot"  # a few-shot continuation: one fresh multiplier at its unseen budget
KINDS = (DISCRETE, CONTINUOUS, FEWSHOT)

# progress.csv names (contract 3). The multiplier of each level, every epoch:
LEVEL_MULTIPLIER_PREFIX = rundir.LEVEL_MULTIPLIER_PREFIX
# Diagnostics, deliberately outside the prefixes the launcher checks for non-finite values
# (``Metrics/LagrangeMultiplier`` and ``Loss/``) and outside the ledger writer's level prefix: a level
# with no episode leaves its J_C undefined, which OmniSafe writes as NaN (Study B spec E2).
LEVEL_JC_PREFIX = "Metrics/LevelJc/level_"  # the J_C that updated the level's multiplier
LEVEL_BUDGET_PREFIX = "Metrics/LevelBudget/level_"  # the d of that update (the mean window budget for a bin)
LEVEL_EPISODES_PREFIX = "Metrics/LevelEpisodes/level_"  # the level's episodes finished this epoch
LEVEL_WINDOW_PREFIX = "Metrics/LevelWindow/level_"  # the level's episodes in the J_C window
LEVEL_BATCH_COST_PREFIX = "Metrics/LevelBatchEpCost/level_"  # this epoch's mean episodic cost at the level
DIAGNOSTIC_PREFIXES = (LEVEL_JC_PREFIX, LEVEL_BUDGET_PREFIX, LEVEL_EPISODES_PREFIX, LEVEL_WINDOW_PREFIX,
                       LEVEL_BATCH_COST_PREFIX)
EPCOST_KEY = "Metrics/EpCost"  # OmniSafe's 50-episode window, J_C of PPOLag (ppo_lag.py _update)

CONTINUATION_FILE = "continuation.json"  # the few-shot restore summary, in the OmniSafe run directory
FEWSHOT_PARAMS = ("parent_run_id", "parent_step", "budget", "horizons", "fresh_multiplier_init")
# Part 6.1 cut 3 (pilot.manifest._cut_fewshot_short): the steps of the actor schedule a shortened few-shot continuation
# keeps (R.FEWSHOT_STEPS; Table 9.1, Q-continuations; read by pilot.dependencies.restore_learner). No other spec has it.
SCHEDULE_PARAM = "lr_schedule_steps"

# The OmniSafe-based class names (the adapter and the two algorithms), built on first use (OmniSafe is
# imported lazily, as in pilot/algorithms.py).
_LAZY_CLASSES = ("BudgetConditionedAdapter", "BudgetConditionedPPOLag", "FewShotPPOLag")


# ---------------------------------------------------------------------------
# Levels, the budget feature and the columns (Study B spec 6.1)
# ---------------------------------------------------------------------------


def _as_run(spec: Any) -> RunSpec:
    """A ``RunSpec`` from a RunSpec or its dict (``RunSpec.to_dict()``); ValueError otherwise."""
    if isinstance(spec, RunSpec):
        return spec
    if not isinstance(spec, Mapping):
        raise ValueError(f"a run spec (RunSpec or RunSpec.to_dict()) is required, got {type(spec).__name__}")
    try:
        return RunSpec.from_dict(spec)
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError(f"not a valid run spec: {exc}") from exc


def _real(value: Any, what: str) -> float:
    """``value`` as a finite float; ValueError for a bool, a string, None or a non-finite number."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f"{what} must be a finite number, got {value!r}")
    try:
        x = float(value)
    except (OverflowError, TypeError, ValueError):
        x = math.inf  # an integer too large for a float
    if not math.isfinite(x):
        raise ValueError(f"{what} must be a finite number, got {value!r}")
    return x


def level_kind(spec: Any) -> str:
    """``fewshot`` for plug-in study_b_fewshot; for plug-in study_b ``continuous`` (the Continuous arm) or
    ``discrete``; ValueError for any other plug-in (not budget-conditioned)."""
    run = _as_run(spec)
    if run.plugin == "study_b_fewshot":
        return FEWSHOT
    if run.plugin != "study_b":
        raise ValueError(f"{run.run_id}: plug-in {run.plugin!r} is not budget-conditioned (Study B)")
    return CONTINUOUS if run.training_levels is None else DISCRETE


def continuous_edges() -> tuple[float, ...]:
    """The lower edges of the Continuous arm's bins: 10, 15, ..., 35 (Table 2.5; Q-continuous-bins)."""
    lo, hi = (float(x) for x in R.CONTINUOUS_RANGE)
    width = float(R.CONTINUOUS_BIN_WIDTH)
    count = round((hi - lo) / width)
    if count < 1 or not math.isclose(count * width, hi - lo):
        raise ValueError(f"the range {R.CONTINUOUS_RANGE} is not a whole number of {width}-unit bins")
    return tuple(lo + i * width for i in range(count))


def level_keys(spec: Any) -> tuple[float, ...]:
    """The keys of a run's multipliers, ascending: its levels, the Continuous bins' lower edges, or
    a few-shot continuation's one budget (``spec.params['budget']``)."""
    run = _as_run(spec)
    kind = level_kind(run)
    if kind == FEWSHOT:
        return (_real((run.params or {}).get("budget"), f"{run.run_id}: params['budget']"),)
    if kind == CONTINUOUS:
        return continuous_edges()
    return tuple(sorted(_real(b, f"{run.run_id}: a training level") for b in run.training_levels or ()))


def budget_feature(budget: float) -> torch.Tensor:
    """The float32 feature a budget becomes in the observation: ``append_budget``'s own last element."""
    return append_budget(torch.zeros(0, dtype=torch.float32), budget)[-1]


def level_index(feature: Any, keys: Sequence[float], kind: str) -> torch.Tensor:
    """The multiplier index of each budget feature (a tensor of any shape, flattened; a 1-D long CPU
    tensor with one index per element).

    Features are float32 (as stored in the observations); the budget is read back as
    ``float64(feature) * R.BUDGET_OBSERVATION_DIVISOR``. Discrete arms and few-shot continuations: the
    nearest key, which must lie within ``LEVEL_TOLERANCE`` (else ValueError). Continuous
    (Q-continuous-bins: "an episode's or sample's bin is read from its float32 budget feature"): the
    number of inner edges 15, 20, ..., 35 whose own feature ``budget_feature(edge)`` is at most the
    feature, so the bins are [10, 15), ..., [30, 35) and [35, 40] in feature space. Comparing features
    with features keeps a budget at an edge in the bin it opens although float32 rounds some edges
    down (35 / 100 becomes 0.34999999...; a bin of ``floor((b - 10) / 5)`` would put 35 in [30, 35)),
    and 40 falls in the last bin. A budget outside [10, 40] is refused. Training uses this one
    function both for the stored feature of each sample (the penalty of equation (3)) and for
    ``budget_feature(b)`` of each finished episode (the level whose J_C it enters), so every budget
    meets the same multiplier in both. The features are moved to the CPU, where the key tables live.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown level kind {kind!r}; choose from {KINDS}")
    keys = tuple(float(k) for k in keys)
    if not keys:
        raise ValueError("no level keys")
    if kind == CONTINUOUS and keys != continuous_edges():
        raise ValueError(f"continuous keys must be the bin edges {continuous_edges()}, got {keys}")
    features = torch.as_tensor(feature).detach().reshape(-1).cpu().to(torch.float32).to(torch.float64)
    if features.numel() == 0:
        return torch.zeros(0, dtype=torch.long)
    if not bool(torch.isfinite(features).all()):
        raise ValueError("a budget feature is not finite")
    budgets = features * R.BUDGET_OBSERVATION_DIVISOR
    if kind == CONTINUOUS:
        lo, hi = (float(x) for x in R.CONTINUOUS_RANGE)
        outside = (budgets < lo - LEVEL_TOLERANCE) | (budgets > hi + LEVEL_TOLERANCE)
        if bool(outside.any()):
            raise ValueError(f"budget {float(budgets[outside][0]):g} lies outside the Continuous range [{lo:g}, {hi:g}]")
        inner = _edge_features(keys[1:])  # the features of 15, ..., 35
        return (features.unsqueeze(1) >= inner.unsqueeze(0)).sum(dim=1).to(torch.long)
    table = torch.tensor(keys, dtype=torch.float64)
    nearest, index = (budgets.unsqueeze(1) - table.unsqueeze(0)).abs().min(dim=1)
    if float(nearest.max()) > LEVEL_TOLERANCE:
        worst = int(nearest.argmax())
        raise ValueError(f"budget {float(budgets[worst]):g} is not a level of {keys} (within {LEVEL_TOLERANCE})")
    return index


@functools.lru_cache(maxsize=None)
def _edge_features(edges: tuple[float, ...]) -> torch.Tensor:
    """The float32 features of the bin edges, widened to float64 (exact); computed once, as
    ``level_index`` runs on every minibatch of the policy update."""
    return torch.stack([budget_feature(edge) for edge in edges]).to(torch.float64)


def _key_text(key: float) -> str:
    text = f"{float(key):g}"
    if float(text) != float(key):  # the ledger writer reads the level back with float() (contract 3)
        raise ValueError(f"level {key!r} does not survive the column name {text!r}")
    return text


def level_column(key: float) -> str:
    """``Metrics/LagrangeMultiplier/level_{key:g}`` (contract 3), e.g. ``level_17.5``, ``level_10``."""
    return LEVEL_MULTIPLIER_PREFIX + _key_text(key)


def diagnostic_columns(key: float) -> tuple[str, ...]:
    """The diagnostic progress.csv columns of one level (none is a multiplier or a loss column)."""
    return tuple(prefix + _key_text(key) for prefix in DIAGNOSTIC_PREFIXES)


# ---------------------------------------------------------------------------
# Budget sources (Study B spec 6.2): dedicated generators, never a global stream
# ---------------------------------------------------------------------------


def budget_generator(seed: int) -> np.random.Generator:
    """The run's own budget stream: ``numpy.random.default_rng([seed, BUDGET_STREAM])``.

    Drawing from it touches neither numpy's global state nor torch's, so the network initialisation
    and the environment's layout sequence of seed k are the same in every arm (Part 5.1).
    """
    seed = int(seed)
    if seed < 0:
        raise ValueError(f"a run seed is non-negative, got {seed}")
    return np.random.default_rng([seed, BUDGET_STREAM])


class DiscreteBudgets:
    """Table 2.5: "each episode draws its budget uniformly at random from the arm's training set"."""

    def __init__(self, levels: Sequence[float], rng: np.random.Generator) -> None:
        self.levels = tuple(float(b) for b in levels)
        if not self.levels:
            raise ValueError("an arm has at least one training level")
        self._rng = rng

    def __call__(self) -> float:
        return self.levels[int(self._rng.integers(len(self.levels)))]


class ContinuousBudgets:
    """Table 2.5 Continuous: "a budget drawn uniformly from [10, 40] at each episode".

    numpy's ``uniform`` draws from [low, high); the difference from the closed interval has
    probability zero (Study B spec 6.2).
    """

    def __init__(self, low: float, high: float, rng: np.random.Generator) -> None:
        self.low, self.high = float(low), float(high)
        if not self.low < self.high:
            raise ValueError(f"an empty budget range [{low}, {high}]")
        self._rng = rng

    def __call__(self) -> float:
        return float(self._rng.uniform(self.low, self.high))


class FixedBudget:
    """A few-shot continuation trains "under that budget" (Table 2.5): every episode carries it."""

    def __init__(self, budget: float) -> None:
        self.budget = _real(budget, "a few-shot budget")

    def __call__(self) -> float:
        return self.budget


def make_budget_source(kind: str, keys: Sequence[float], seed: int) -> Callable[[], float]:
    """The budget source of a run of ``kind`` with multiplier ``keys`` and run seed ``seed``."""
    if kind == DISCRETE:
        return DiscreteBudgets(keys, budget_generator(seed))
    if kind == CONTINUOUS:
        lo, hi = R.CONTINUOUS_RANGE
        return ContinuousBudgets(lo, hi, budget_generator(seed))
    if kind == FEWSHOT:
        if len(keys) != 1:
            raise ValueError(f"a few-shot continuation has one budget, got {tuple(keys)}")
        return FixedBudget(keys[0])
    raise ValueError(f"unknown level kind {kind!r}; choose from {KINDS}")


# ---------------------------------------------------------------------------
# Per-level state in every checkpoint (Study B spec 6.7; contract 3: weights_only-loadable)
# ---------------------------------------------------------------------------


class _LevelLagrangeState:
    """``level_lagranges``: kind, keys, multiplier values and cost limits, read at every save."""

    def __init__(self, algo: Any) -> None:
        self._algo = algo

    def state_dict(self) -> dict[str, Any]:
        lagranges = self._algo._level_lagranges
        return {
            "kind": str(self._algo._level_kind),
            "keys": torch.tensor([float(k) for k in lagranges], dtype=torch.float64),
            # float32 parameters widened to float64: exact
            "values": torch.tensor([float(lag.lagrangian_multiplier.detach()) for lag in lagranges.values()],
                                   dtype=torch.float64),
            "cost_limits": torch.tensor([float(lag.cost_limit) for lag in lagranges.values()], dtype=torch.float64),
        }


class _LevelOptimizerState:
    """``level_lambda_optimizers``: each level's multiplier optimiser (Adam) state, keyed ``f"{key:g}"``.

    An object read at save time, not a dict put in the saver: ``Logger.torch_save`` calls
    ``state_dict()`` only on entries that have one, so a plain dict would be a stale snapshot.
    """

    def __init__(self, algo: Any) -> None:
        self._algo = algo

    def state_dict(self) -> dict[str, Any]:
        return {_key_text(key): lag.lambda_optimizer.state_dict() for key, lag in self._algo._level_lagranges.items()}


class _LevelWindowState:
    """``level_window``: the J_C window (level index, episodic cost, budget of each finished episode)."""

    def __init__(self, algo: Any) -> None:
        self._algo = algo

    def state_dict(self) -> dict[str, Any]:
        window = self._algo._ensure_level_window()  # the mixin saves epoch-0.pt before the algorithm's _init_log returns
        entries = list(window)
        return {
            "index": torch.tensor([int(e[0]) for e in entries], dtype=torch.int64),
            "cost": torch.tensor([float(e[1]) for e in entries], dtype=torch.float64),
            "budget": torch.tensor([float(e[2]) for e in entries], dtype=torch.float64),
            "maxlen": int(window.maxlen),
        }


def _omnisafe_mean(values: Sequence[float]) -> float:
    """The mean exactly as OmniSafe's ``Logger.get_stats`` computes ``Metrics/EpCost`` (logger.py:
    ``torch.tensor(vals)`` in float32, then ``dist_statistics_scalar``), so a single-level arm's J_C is
    the float PPOLag passes to its multiplier update."""
    from omnisafe.utils.distributed import dist_statistics_scalar

    data = torch.tensor([float(v) for v in values]).to(os.getenv("OMNISAFE_DEVICE", "cpu"))
    return float(dist_statistics_scalar(data)[0].item())


# ---------------------------------------------------------------------------
# The adapter (Study B spec 6.3)
# ---------------------------------------------------------------------------


class _AdapterMethods:
    """``BudgetConditionedAdapter``: OmniSafe's on-policy adapter with the budget in the observation.

    The class is ``BudgetConditionedAdapter(_AdapterMethods, OnPolicyAdapter)``, built on first use.

    * ``observation_space`` is the wrapper chain's plus one feature, so OmniSafe builds the actor,
      both critics and the buffer with 61 inputs on SafetyPointGoal1-v0 (Table 3.1 "Networks"); the
      observation normaliser (``ObsNormalize``, inside the chain) keeps the task's 60.
    * ``reset`` and ``step`` append the current episode's budget with ``append_budget`` after the
      whole chain, and ``info['final_observation']`` carries the finished episode's budget (it
      bootstraps that episode's value at a time-out).
    * Each episode that takes a step carries its own draw (Table 2.5). A draw is made when an
      episode begins (a reset, or OmniSafe's ``AutoReset`` after an episode's end); an episode that
      begins but takes no step before the next reset (the auto-reset at the end of an epoch, which
      the next epoch's ``reset`` discards) passes its draw on to the episode that replaces it, so no
      draw is lost and the k-th episode that enters the buffer carries the k-th draw of the stream.
    * ``rollout`` records each finished episode's (budget, undiscounted cost) in
      ``finished_episodes``, which the algorithm drains once per epoch (the per-level J_C).
    """

    def __init__(self, env_id: str, num_envs: int, seed: int, cfgs: Any, budget_source: Callable[[], float]) -> None:
        if int(num_envs) != 1:
            # AutoReset and Unsqueeze require one environment (omnisafe/envs/wrapper.py), and so does
            # the bookkeeping of one current budget; Table 3.1 fixes one vectorised environment.
            raise RunRefused(f"the budget-conditioned adapter runs one environment, not {num_envs} (Table 3.1)")
        super().__init__(env_id, num_envs, seed, cfgs)  # type: ignore[call-arg]  # builds, wraps and seeds the env
        from gymnasium.spaces import Box

        inner = self._env.observation_space  # type: ignore[attr-defined]
        if not isinstance(inner, Box) or len(inner.shape) != 1:
            raise RunRefused(f"{env_id}: a flat Box observation is required to append the budget, got {inner}")
        # As omnisafe/adapter/saute_adapter.py builds its augmented space; constructing a Box draws nothing.
        self._observation_space = Box(low=-np.inf, high=np.inf, shape=(int(inner.shape[0]) + 1,), dtype=np.float32)
        self._budget_source = budget_source
        self._episode_budget: float | None = None
        self._episode_stepped = False
        self._finished_budget: float | None = None
        self.budget_draws = 0
        self.finished_episodes: list[tuple[float, float]] = []

    @property
    def observation_space(self) -> Any:
        """The wrapper chain's observation space plus the budget feature."""
        return self._observation_space

    @property
    def episode_budget(self) -> float | None:
        """The budget of the current episode (None before the first reset)."""
        return self._episode_budget

    def _begin_episode(self) -> None:
        if self._episode_budget is None or self._episode_stepped:
            self._episode_budget = float(self._budget_source())
            self.budget_draws += 1
        # else: the episode begun last took no step and is discarded; its draw passes to this one
        self._episode_stepped = False

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
        """The wrapper chain's reset; a new episode begins, and its budget is appended."""
        obs, info = super().reset(seed=seed, options=options)  # type: ignore[misc]
        self._begin_episode()
        return append_budget(obs, self._episode_budget), info

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                                   torch.Tensor, dict[str, Any]]:
        """The wrapper chain's step with the budget appended (the finished episode's to
        ``info['final_observation']``, the next episode's to the returned observation)."""
        self._episode_stepped = True  # this episode's draw is used
        next_obs, reward, cost, terminated, truncated, info = super().step(action)  # type: ignore[misc]
        ended = bool(torch.as_tensor(terminated).any()) or bool(torch.as_tensor(truncated).any())
        if ended != ("final_observation" in info):
            # AutoReset adds final_observation exactly when an episode ends (omnisafe/envs/wrapper.py);
            # anything else means the wrapper chain is not the one this adapter was written for.
            raise RuntimeError("the wrapper chain's episode end and its final_observation disagree")
        if ended:
            info["final_observation"] = append_budget(info["final_observation"], self._episode_budget)
            self._finished_budget = self._episode_budget
            self._begin_episode()  # AutoReset has begun the next episode
        return append_budget(next_obs, self._episode_budget), reward, cost, terminated, truncated, info

    def drain_finished(self) -> list[tuple[float, float]]:
        """The (budget, cost) of every episode finished since the last call, in order; then empty."""
        finished, self.finished_episodes = self.finished_episodes, []
        return finished

    def rollout(
        self,
        steps_per_epoch: int,
        agent: Any,
        buffer: Any,
        logger: Any,
    ) -> None:
        """OmniSafe 0.5.0's ``OnPolicyAdapter.rollout`` (omnisafe/adapter/onpolicy_adapter.py:58-131).

        Changes are marked ``# CHANGE``: every finished episode's budget and undiscounted cost are
        recorded where OmniSafe logs ``Metrics/EpCost``, so the per-level J_C window sees the same
        episodes, in the same order, as OmniSafe's window. The observations come from this class's
        ``reset`` and ``step`` (budget appended), so the buffer stores the feature each action was
        taken on and the time-out bootstrap uses the finished episode's budget. Nothing else differs:
        the same calls in the same order, so the same random draws.
        """
        from rich.progress import track

        self._reset_log()  # type: ignore[attr-defined]

        obs, _ = self.reset()
        for step in track(
            range(steps_per_epoch),
            description=f'Processing rollout for epoch: {logger.current_epoch}...',
        ):
            act, value_r, value_c, logp = agent.step(obs)
            next_obs, reward, cost, terminated, truncated, info = self.step(act)

            self._log_value(reward=reward, cost=cost, info=info)  # type: ignore[attr-defined]

            if self._cfgs.algo_cfgs.use_cost:  # type: ignore[attr-defined]
                logger.store({'Value/cost': value_c})
            logger.store({'Value/reward': value_r})

            buffer.store(
                obs=obs,
                act=act,
                reward=reward,
                cost=cost,
                value_r=value_r,
                value_c=value_c,
                logp=logp,
            )

            obs = next_obs
            epoch_end = step >= steps_per_epoch - 1
            for idx, (done, time_out) in enumerate(zip(terminated, truncated)):
                if epoch_end or done or time_out:
                    last_value_r = torch.zeros(1)
                    last_value_c = torch.zeros(1)
                    if not done:
                        if epoch_end:
                            logger.log(
                                f'Warning: trajectory cut off when rollout by epoch at {self._ep_len[idx]} steps.',  # type: ignore[attr-defined]
                            )
                            _, last_value_r, last_value_c, _ = agent.step(obs[idx])
                        if time_out:
                            _, last_value_r, last_value_c, _ = agent.step(
                                info['final_observation'][idx],
                            )
                        last_value_r = last_value_r.unsqueeze(0)
                        last_value_c = last_value_c.unsqueeze(0)

                    if done or time_out:
                        self._log_metrics(logger, idx)  # type: ignore[attr-defined]
                        # CHANGE: the finished episode's budget and undiscounted cost (the value OmniSafe
                        # has just logged as Metrics/EpCost), for the per-level J_C (Q-level-jc).
                        self.finished_episodes.append((float(self._finished_budget), float(self._ep_cost[idx])))  # type: ignore[attr-defined]
                        self._reset_log(idx)  # type: ignore[attr-defined]

                        self._ep_ret[idx] = 0.0  # type: ignore[attr-defined]
                        self._ep_cost[idx] = 0.0  # type: ignore[attr-defined]
                        self._ep_len[idx] = 0.0  # type: ignore[attr-defined]

                    buffer.finish_path(last_value_r, last_value_c, idx)


# ---------------------------------------------------------------------------
# The algorithm (Study B spec 6.4 to 6.7)
# ---------------------------------------------------------------------------


class _PPOLagMethods:
    """``BudgetConditionedPPOLag``: PPO-Lagrangian with one multiplier per training level.

    The class is ``BudgetConditionedPPOLag(_PPOLagMethods, FullStateCheckpointMixin, PPO)``, built on
    first use, so its method order is that of a ``(FullStateCheckpointMixin, PPO)`` subclass that
    overrides these methods. It reads its run facts from ``cfgs.studyb_cfgs`` (set by the factory).

    * ``_level_lagranges``: {key: OmniSafe ``Lagrange``} with the configured optimiser, learning rate
      and initial value (``cfgs.lagrange_cfgs``) and cost limit d of the level (the unseen budget for
      a continuation; for a Continuous bin, the mean window budget, set before each update). There
      is no ``_lagrange``: nothing is logged as the plain ``Metrics/LagrangeMultiplier``, and
      ``pilot.launch.nonfinite_state`` reads these (pilot/launch.py).
    * ``_update``: once per epoch, before the policy update as in PPOLag (ppo_lag.py ``_update``),
      each level's multiplier takes one step of equation (4) from its own J_C (Q-level-jc); after the
      policy update every level's value is logged (Part 3.4).
    * ``_update_actor`` / ``_compute_adv_surrogate``: equation (3) per sample, with the multiplier of
      the sample's level read from the feature stored in the observation; with one multiplier value
      the result is PPOLag's bit for bit.
    """

    _studyb_spec: RunSpec | None = None  # the continuation's spec (set by make_fewshot_algorithm)

    def _studyb_cfgs(self) -> Mapping[str, Any]:
        cfg = self._cfgs.get("studyb_cfgs") if hasattr(self._cfgs, "get") else None  # type: ignore[attr-defined]
        if not isinstance(cfg, Mapping) or cfg.get("kind") not in KINDS:
            raise RunRefused("cfgs.studyb_cfgs is missing: build the algorithm with studyb.conditioning.make_algorithm "
                             "or make_fewshot_algorithm")
        return cfg

    def _init_env(self) -> None:
        """PolicyGradient's ``_init_env`` with the budget-conditioned adapter (policy_gradient.py:48-77)."""
        from omnisafe.utils import distributed

        cfg = self._studyb_cfgs()
        self._level_kind = str(cfg["kind"])
        self._level_keys = tuple(float(k) for k in cfg["level_keys"])
        source = make_budget_source(self._level_kind, self._level_keys, self._seed)  # type: ignore[attr-defined]
        env_nums = int(self._cfgs.train_cfgs.vector_env_nums)  # type: ignore[attr-defined]
        self._env = _classes()["BudgetConditionedAdapter"](
            self._env_id, env_nums, self._seed, self._cfgs, source,  # type: ignore[attr-defined]
        )
        steps = int(self._cfgs.algo_cfgs.steps_per_epoch)  # type: ignore[attr-defined]
        if steps % (distributed.world_size() * env_nums):
            raise RunRefused(f"steps_per_epoch {steps} is not divisible by world size x {env_nums} environments "
                             "(policy_gradient.py _init_env)")
        self._steps_per_epoch = steps // distributed.world_size() // env_nums

    def _initial_cost_limit(self, key: float) -> float:
        if self._level_kind == CONTINUOUS:  # the bin's centre until its first update sets the window mean
            return float(key) + float(R.CONTINUOUS_BIN_WIDTH) / 2.0
        return float(key)

    def _init(self) -> None:
        super()._init()  # type: ignore[misc]  # PolicyGradient: the buffer, with the budget feature
        from omnisafe.common.lagrange import Lagrange

        lagrange_cfgs = dict(self._cfgs.lagrange_cfgs.todict())  # type: ignore[attr-defined]
        # lagrange_cfgs.cost_limit (Table 2.1's d = 25) is Study A's; each level has its own d.
        self._level_lagranges = {
            key: Lagrange(**{**lagrange_cfgs, "cost_limit": self._initial_cost_limit(key)}) for key in self._level_keys
        }
        self._batch_penalty: torch.Tensor | None = None

    def _init_log(self) -> None:
        # The mixin: OmniSafe's keys, the batch metrics, the extra checkpoint steps (pilot.contracts.extra_checkpoint_steps), the
        # full-state saver with _extra_checkpoint_state, and epoch-0.pt.
        super()._init_log()  # type: ignore[misc]
        logger = self._logger  # type: ignore[attr-defined]
        for key in self._level_keys:
            logger.register_key(level_column(key))  # one column per level (contract 3), every epoch
            for column in diagnostic_columns(key):
                logger.register_key(column)
        self._ensure_level_window()

    def _ensure_level_window(self) -> collections.deque[tuple[int, float, float]]:
        """The J_C window: (level index, cost, budget) of the last W finished episodes, created once.

        W is the length of OmniSafe's own ``Metrics/EpCost`` window, read from the logger (not
        retyped; 50 in OmniSafe 0.5.0), so a level's J_C is taken over its episodes among exactly the
        episodes OmniSafe's J_C would average (Q-level-jc).
        """
        window = getattr(self, "_level_window", None)
        if window is None:
            window = collections.deque(maxlen=self._logger._data[EPCOST_KEY].maxlen)  # type: ignore[attr-defined]
            self._level_window = window
        return window

    def _extra_checkpoint_state(self) -> dict[str, Any]:
        return {
            "level_lagranges": _LevelLagrangeState(self),
            "level_lambda_optimizers": _LevelOptimizerState(self),
            "level_window": _LevelWindowState(self),
        }

    def _level_budget(self, key: float, budgets: Sequence[float]) -> float:
        """The d of a level's update: the level itself, or for a Continuous bin the mean window budget."""
        if self._level_kind == CONTINUOUS:
            return math.fsum(budgets) / len(budgets)
        far = [b for b in budgets if abs(b - key) > LEVEL_TOLERANCE]
        if far:
            raise RuntimeError(f"episodes at budget {far[0]:g} were attributed to level {key:g}")
        return float(key)

    def _update_level_multipliers(self, finished: Sequence[tuple[float, float]]) -> None:
        """Equation (4) for every level that has episodes in the J_C window (Q-level-jc)."""
        logger = self._logger  # type: ignore[attr-defined]
        window = self._ensure_level_window()
        batch: list[list[float]] = [[] for _ in self._level_keys]
        for budget, cost in finished:
            index = int(level_index(budget_feature(budget), self._level_keys, self._level_kind)[0])
            window.append((index, float(cost), float(budget)))
            batch[index].append(float(cost))
        for index, (key, lagrange) in enumerate(self._level_lagranges.items()):
            suffix = _key_text(key)
            entries = [(cost, budget) for i, cost, budget in window if i == index]
            logger.store({LEVEL_EPISODES_PREFIX + suffix: float(len(batch[index])),
                          LEVEL_WINDOW_PREFIX + suffix: float(len(entries))})
            if batch[index]:
                logger.store({LEVEL_BATCH_COST_PREFIX + suffix: _omnisafe_mean(batch[index])})
            if not entries:
                continue  # no episode of this level in the window: its multiplier and optimiser stay as they are
            jc = _omnisafe_mean([cost for cost, _ in entries])
            if not math.isfinite(jc):  # PPOLag asserts J_C is not NaN (ppo_lag.py:74); an infinite J_C is refused too
                raise RuntimeError(f"cost for updating the multiplier of level {suffix} is not finite")
            lagrange.cost_limit = self._level_budget(key, [budget for _, budget in entries])
            lagrange.update_lagrange_multiplier(jc)
            logger.store({LEVEL_JC_PREFIX + suffix: jc, LEVEL_BUDGET_PREFIX + suffix: float(lagrange.cost_limit)})

    def _update(self) -> None:
        """PPOLag's ``_update`` (ppo_lag.py:52-80), one multiplier per level."""
        self._update_level_multipliers(self._env.drain_finished())  # type: ignore[attr-defined]
        super()._update()  # type: ignore[misc]  # PolicyGradient: critics and actor
        for key, lagrange in self._level_lagranges.items():
            # stored as PPOLag stores its multiplier (the Parameter; the logger keeps .mean().item())
            self._logger.store({level_column(key): lagrange.lagrangian_multiplier})  # type: ignore[attr-defined]

    def _update_actor(self, obs: torch.Tensor, act: torch.Tensor, logp: torch.Tensor, adv_r: torch.Tensor,
                      adv_c: torch.Tensor) -> None:
        """PolicyGradient's ``_update_actor`` with each sample's multiplier set for the surrogate."""
        index = level_index(obs[:, -1], self._level_keys, self._level_kind)
        values = torch.stack([lag.lagrangian_multiplier.detach() for lag in self._level_lagranges.values()])
        self._batch_penalty = values.to(dtype=adv_r.dtype, device=adv_r.device)[index.to(adv_r.device)]
        try:
            super()._update_actor(obs, act, logp, adv_r, adv_c)  # type: ignore[misc]
        finally:
            self._batch_penalty = None

    def _compute_adv_surrogate(self, adv_r: torch.Tensor, adv_c: torch.Tensor) -> torch.Tensor:
        """Equation (3) per sample: (A_R - lambda_i A_C) / (1 + lambda_i) (ppo_lag.py:101-102 with a vector)."""
        penalty = self._batch_penalty
        if penalty is None:
            raise RuntimeError("the per-sample penalty is set only inside _update_actor")
        return (adv_r - penalty * adv_c) / (1 + penalty)


class _FewShotMethods:
    """``FewShotPPOLag``: a few-shot continuation (Table 2.5 "Few-shot adaptation"; Part 3.3).

    ``BudgetConditionedPPOLag`` with one fresh multiplier at the continuation's unseen budget and the
    budget fixed at it, after the parent's learner is restored in ``_init`` by
    ``pilot.dependencies.restore_learner`` (Table 9.1, Q-continuations: actor, both critics and
    normaliser from the parent's final checkpoint; fresh optimisers and a fresh schedule over the
    continuation's full registered length, 1,000,000 steps, which a continuation shortened by Part 6.1
    cut 3 keeps), so ``epoch-0.pt`` holds the restored learner. The restore summary is written to
    ``continuation.json`` in the OmniSafe run directory. The checkpoint at a horizon off OmniSafe's
    200,000-step cadence (500,000) is saved by the mixin from the launcher's
    ``pilot_cfgs.extra_checkpoint_steps``; the others are on the cadence.
    """

    def _init(self) -> None:
        super()._init()  # type: ignore[misc]  # the buffer and the fresh multiplier at d_test
        spec = self._studyb_spec  # type: ignore[attr-defined]
        if spec is None:
            raise RunRefused("a few-shot continuation is built by studyb.conditioning.make_fewshot_algorithm")
        parent_id = spec.params["parent_run_id"]
        parent = dependencies.from_cfgs(self._cfgs).get(parent_id)  # type: ignore[attr-defined]
        if parent is None:
            raise RunRefused(f"{spec.run_id}: pilot_cfgs.dependencies has no entry for the parent {parent_id!r}")
        state = dependencies.load_full_state(parent.checkpoint(int(spec.params["parent_step"])))
        self._restore_summary = dependencies.restore_learner(self, state, spec=spec)

    def _init_log(self) -> None:
        super()._init_log()  # type: ignore[misc]
        spec = self._studyb_spec  # type: ignore[attr-defined]
        lagrange = next(iter(self._level_lagranges.values()))  # type: ignore[attr-defined]
        record = {
            "run_id": spec.run_id,
            "restore": self._restore_summary,
            "budget": float(spec.params["budget"]),
            "horizons": [int(h) for h in spec.params["horizons"]],
            "multiplier": {
                "init": float(lagrange.lagrangian_multiplier.detach()),
                "cost_limit": float(lagrange.cost_limit),
                "lambda_lr": float(lagrange.lambda_lr),
                "optimizer": type(lagrange.lambda_optimizer).__name__,
            },
            "rule": "Table 2.5 few-shot adaptation; Table 9.1, Q-continuations (pilot.dependencies.restore_learner)",
        }
        from envs.onset import write_json_once

        write_json_once(Path(self._logger.log_dir) / CONTINUATION_FILE, record)  # type: ignore[attr-defined]


@functools.lru_cache(maxsize=None)
def _classes() -> dict[str, type]:
    """The three OmniSafe-based classes, built once on first use (OmniSafe is imported lazily)."""
    from omnisafe.adapter.onpolicy_adapter import OnPolicyAdapter
    from omnisafe.algorithms.on_policy.base.ppo import PPO

    def build(name: str, bases: tuple[type, ...], body: type) -> type:
        return type(name, bases, {"__module__": __name__, "__qualname__": name, "__doc__": body.__doc__})

    adapter = build("BudgetConditionedAdapter", (_AdapterMethods, OnPolicyAdapter), _AdapterMethods)
    algo = build("BudgetConditionedPPOLag", (_PPOLagMethods, FullStateCheckpointMixin, PPO), _PPOLagMethods)
    fewshot = build("FewShotPPOLag", (_FewShotMethods, algo), _FewShotMethods)
    return {"BudgetConditionedAdapter": adapter, "BudgetConditionedPPOLag": algo, "FewShotPPOLag": fewshot}


def __getattr__(name: str) -> Any:
    """``BudgetConditionedAdapter``, ``BudgetConditionedPPOLag`` and ``FewShotPPOLag``, built on first use."""
    if name in _LAZY_CLASSES:
        return _classes()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# The factories (Study B spec 6.6 and 6.8). Every problem is a RunRefused (HANDOVER.md section 8).
# ---------------------------------------------------------------------------


def _refuse(run_id: str, problems: list[str]) -> None:
    if problems:
        raise RunRefused(f"{run_id}: " + "; ".join(problems))


def _section(cfgs: Any, name: str) -> Mapping[str, Any]:
    value = cfgs.get(name) if hasattr(cfgs, "get") else None
    return value if isinstance(value, Mapping) else {}


def _same_levels(given: Any, registered: Any) -> bool:
    if given is None or registered is None:
        return given is None and registered is None
    try:
        return sorted(float(b) for b in given) == sorted(float(b) for b in registered)
    except (TypeError, ValueError):
        return False


def _common_problems(env_id: str, cfgs: Any, run: RunSpec, plugin: str) -> list[str]:
    """What both plug-ins require of the spec and of the launcher's configuration."""
    problems: list[str] = []
    if run.study != "B":
        problems.append(f"study {run.study!r} is not Study B")
    if run.plugin != plugin:
        problems.append(f"plug-in {run.plugin!r} is not {plugin!r}")
    if run.base_algo != "PPOLag":
        problems.append(f"base_algo {run.base_algo!r} is not PPOLag (Table 3.1: PPO-Lagrangian)")
    if env_id != run.task:
        problems.append(f"env_id {env_id!r} is not the spec's task {run.task!r}")
    if run.task not in R.TASKS_STUDY_B:
        problems.append(f"task {run.task!r} is not a Study B task {R.TASKS_STUDY_B} (Table 3.1 'Tasks')")
    if run.arm not in R.STUDY_B_ARMS:
        problems.append(f"arm {run.arm!r} is not a Study B arm {tuple(R.STUDY_B_ARMS)} (Table 3.4)")
    elif not _same_levels(run.training_levels, R.STUDY_B_ARMS[run.arm]):
        problems.append(f"training_levels {run.training_levels!r} are not arm {run.arm}'s {R.STUDY_B_ARMS[run.arm]!r} "
                        "(Table 2.5 'Training set of levels')")
    if run.onset_step not in (None, 0):
        problems.append(f"onset_step {run.onset_step} (Study B has no onset)")
    train_cfgs, algo_cfgs, lagrange_cfgs = (_section(cfgs, n) for n in ("train_cfgs", "algo_cfgs", "lagrange_cfgs"))
    if train_cfgs.get("vector_env_nums") != R.VECTOR_ENV_NUMS:
        problems.append(f"train_cfgs.vector_env_nums is {train_cfgs.get('vector_env_nums')!r}, not {R.VECTOR_ENV_NUMS} "
                        "(Table 3.1 'Hardware and device')")
    if algo_cfgs.get("use_cost") is not True:
        problems.append("algo_cfgs.use_cost is not True (the cost critic trains from the first step)")
    seed = cfgs.get("seed") if hasattr(cfgs, "get") else None
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral) or int(seed) != run.seed:
        problems.append(f"cfgs.seed {seed!r} is not the spec's seed {run.seed} (Part 5.1 pairs seed k across arms)")
    if lagrange_cfgs.get("lagrangian_multiplier_init") != R.LAGRANGE_MULTIPLIER_INIT:
        problems.append(f"lagrange_cfgs.lagrangian_multiplier_init is {lagrange_cfgs.get('lagrangian_multiplier_init')!r}, "
                        f"not {R.LAGRANGE_MULTIPLIER_INIT} (Table 3.1 'Algorithm')")
    if lagrange_cfgs.get("lambda_lr") != R.LAGRANGE_MULTIPLIER_LR:
        problems.append(f"lagrange_cfgs.lambda_lr is {lagrange_cfgs.get('lambda_lr')!r}, not {R.LAGRANGE_MULTIPLIER_LR} "
                        "(Table 3.1 'Algorithm')")
    if not isinstance(lagrange_cfgs.get("lambda_optimizer"), str):
        problems.append("lagrange_cfgs.lambda_optimizer names no optimiser")
    if hasattr(cfgs, "get") and cfgs.get("studyb_cfgs") is not None:
        problems.append("cfgs already holds studyb_cfgs; the launcher builds a fresh configuration for each run")
    return problems


def _set_studyb_cfgs(cfgs: Any, values: dict[str, Any]) -> None:
    """Add the plug-in's settings to the configuration: saved in config.json and hashed by the
    launcher after the factory returns (pilot/launch.py; OmniSafe allows keys after validation)."""
    from omnisafe.utils.config import Config

    cfgs["studyb_cfgs"] = Config.dict2config(values)


def _base_studyb_cfgs(run: RunSpec, kind: str, keys: tuple[float, ...]) -> dict[str, Any]:
    return {
        "kind": kind,
        "arm": run.arm,
        "level_keys": [float(k) for k in keys],
        "budget_divisor": float(R.BUDGET_OBSERVATION_DIVISOR),
        "budget_placement": "appended after the wrapper chain (envs.evaluation.append_budget); the observation "
                            "normaliser keeps the task's size (Q-budget-normalisation)",
        "budget_stream": BUDGET_STREAM,
        "jc_rule": "per_level_in_omnisafe_epcost_window",  # Q-level-jc
        "window_source": EPCOST_KEY,
        "penalty": "per sample, the multiplier of the level of the float32 budget feature (equation (3))",
    }


def make_algorithm(env_id: str, cfgs: Any, spec: Any) -> Any:
    """Plug-in ``study_b``: the budget-conditioned PPO-Lagrangian of one Study B arm (Table 3.4).

    RunRefused (exit 5, the run stays pending; HANDOVER.md section 8) unless the spec is a Study B
    training run of a registered arm with its registered levels (Table 2.5), the budget divisor of
    ``R.BUDGET_OBSERVATION_DIVISOR`` and, for Continuous, ``R.CONTINUOUS_RANGE`` and
    ``R.CONTINUOUS_BIN_WIDTH``, and the configuration has one environment, the cost critic, the
    spec's seed and the registered multiplier settings (the optimiser is only checked to be named),
    among other checks (env_id is the task, the task and base_algo, no onset, not a continuation group,
    ``studyb_cfgs`` not yet set). ``spec.total_steps`` is not compared with
    ``R.TOTAL_STEPS`` (smoke copies are shorter, scripts/smoke_run.py) and params keys this plug-in
    does not use are accepted (the re-pilot adds ``pilot_revision``), but not ``SCHEDULE_PARAM``,
    which only a shortened few-shot continuation has. Sets ``cfgs.studyb_cfgs``.
    """
    try:
        run = _as_run(spec)
    except ValueError as exc:
        raise RunRefused(str(exc)) from exc
    problems = _common_problems(env_id, cfgs, run, "study_b")
    if run.group in CONTINUATION_GROUPS:
        problems.append(f"group {run.group!r} is a continuation; plug-in study_b trains a Study B arm from scratch")
    params = dict(run.params or {})
    if params.get("budget_observation_divisor") != R.BUDGET_OBSERVATION_DIVISOR:
        problems.append(f"params['budget_observation_divisor'] is {params.get('budget_observation_divisor')!r}, not "
                        f"{R.BUDGET_OBSERVATION_DIVISOR} (Table 2.5 'Budget conditioning')")
    if SCHEDULE_PARAM in params:
        problems.append(f"params['{SCHEDULE_PARAM}'] belongs to a few-shot continuation shortened by Part 6.1 cut 3 "
                        "(Q-continuations); a Study B training run's schedule spans its own epochs")
    if run.arm in R.STUDY_B_ARMS and R.STUDY_B_ARMS[run.arm] is None:
        given_range = params.get("continuous_range")
        if not isinstance(given_range, (list, tuple)) or list(given_range) != list(R.CONTINUOUS_RANGE):
            problems.append(f"params['continuous_range'] is {given_range!r}, not {list(R.CONTINUOUS_RANGE)} (Table 2.5)")
        if params.get("continuous_bin_width") != R.CONTINUOUS_BIN_WIDTH:
            problems.append(f"params['continuous_bin_width'] is {params.get('continuous_bin_width')!r}, not "
                            f"{R.CONTINUOUS_BIN_WIDTH} (Table 2.5)")
    _refuse(run.run_id, problems)
    kind = level_kind(run)
    keys = level_keys(run)
    values = _base_studyb_cfgs(run, kind, keys)
    if kind == CONTINUOUS:
        values.update({"levels": None, "continuous_range": [float(x) for x in R.CONTINUOUS_RANGE],
                       "bin_width": float(R.CONTINUOUS_BIN_WIDTH), "bins": "lower edge; 40 in the last bin (Q-continuous-bins)"})
    else:
        values.update({"levels": [float(k) for k in keys], "continuous_range": None, "bin_width": None})
    _set_studyb_cfgs(cfgs, values)
    return _classes()["BudgetConditionedPPOLag"](env_id=env_id, cfgs=cfgs)


def _whole(value: Any) -> int:
    """``value`` as an int if it is an integer (not a bool); TypeError otherwise.

    Deliberately stricter than ``pilot.contracts.whole_number``, which also accepts an integral float
    such as 500000.0: the launcher writes specs and configurations as JSON ints, so a float here is
    refused rather than read.
    """
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise TypeError(f"not an integer: {value!r}")
    return int(value)


def _horizon_problems(run: RunSpec, horizons: Any) -> list[str]:
    try:
        values = [_whole(h) for h in horizons]
    except TypeError:
        return [f"params['horizons'] {horizons!r} are not whole numbers of steps"]
    problems: list[str] = []
    if not values:
        problems.append("params['horizons'] is empty")
    elif values != sorted(set(values)):
        problems.append(f"params['horizons'] {values} are not strictly increasing")
    unknown = [h for h in values if h not in R.FEWSHOT_HORIZONS]
    if unknown:
        problems.append(f"params['horizons'] {unknown} are not few-shot horizons {list(R.FEWSHOT_HORIZONS)} (Table 2.5)")
    off_grid = [h for h in values if h % R.STEPS_PER_EPOCH]
    if off_grid:
        problems.append(f"params['horizons'] {off_grid} are not whole epochs of {R.STEPS_PER_EPOCH} steps")
    if values and max(values) != run.total_steps:
        problems.append(f"the last horizon {max(values)} is not the continuation's total {run.total_steps} (Table 2.5: "
                        "the horizons are read from one continuation)")
    return problems


def _parent_problems(run: RunSpec, cfgs: Any) -> list[str]:
    """The parent (``pilot_cfgs.dependencies``, resolved by the launcher, pilot/dependencies.py) must be this arm's
    Study B training run of the same seed, and ``parent_step`` its final checkpoint (Q-continuations:
    "few-shot from the parent's final checkpoint")."""
    params = run.params
    parent_id = params.get("parent_run_id")
    try:
        deps = dependencies.from_cfgs(cfgs)
    except RunRefused as exc:
        return [str(exc)]
    parent = deps.get(parent_id)
    if parent is None:
        return [f"pilot_cfgs.dependencies has no entry for the parent {parent_id!r}"]
    problems: list[str] = []
    try:
        parent_spec = RunSpec.from_json((parent.run_dir / rundir.SPEC_FILE).read_text(encoding="utf-8"))
        parent_config = rundir.read_config(parent.omnisafe_dir)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return [f"parent {parent_id}: its spec.json or config.json cannot be read: {exc}"]
    if not isinstance(parent_config, Mapping):
        return [f"parent {parent_id}: config.json is not a JSON object"]
    if parent_spec.plugin != "study_b":
        problems.append(f"parent {parent_id} is plug-in {parent_spec.plugin!r}, not a Study B training run (study_b)")
    for field_name in ("arm", "seed", "task"):
        if getattr(parent_spec, field_name) != getattr(run, field_name):
            problems.append(f"parent {parent_id} has {field_name} {getattr(parent_spec, field_name)!r}, the continuation "
                            f"{getattr(run, field_name)!r}")
    if not _same_levels(parent_spec.training_levels, run.training_levels):
        problems.append(f"parent {parent_id} trains on {parent_spec.training_levels!r}, the spec names {run.training_levels!r}")
    parent_studyb = parent_config.get("studyb_cfgs")
    if not isinstance(parent_studyb, Mapping) or parent_studyb.get("budget_divisor") != R.BUDGET_OBSERVATION_DIVISOR:
        problems.append(f"parent {parent_id} was not trained by studyb.conditioning with the budget divisor "
                        f"{R.BUDGET_OBSERVATION_DIVISOR} (config.json studyb_cfgs)")
    train = parent_config.get("train_cfgs")
    final = train.get("total_steps") if isinstance(train, Mapping) else None
    if params.get("parent_step") != final:
        problems.append(f"parent_step {params.get('parent_step')!r} is not the parent's final checkpoint (its "
                        f"train_cfgs.total_steps {final!r}; Q-continuations: few-shot starts from the final checkpoint)")
    return problems


def make_fewshot_algorithm(env_id: str, cfgs: Any, spec: Any) -> Any:
    """Plug-in ``study_b_fewshot``: one few-shot continuation of a Study B run (Table 2.5; Part 3.3).

    Restores the parent's final checkpoint (``params['parent_step']``; Table 9.1, Q-continuations)
    by ``pilot.dependencies.restore_learner`` and trains under the unseen budget
    ``params['budget']`` with one fresh multiplier: initial value ``R.LAGRANGE_MULTIPLIER_INIT``
    ("a fresh multiplier at 0.001", which ``params['fresh_multiplier_init']`` must equal), learning
    rate ``R.LAGRANGE_MULTIPLIER_LR`` and cost limit d_test; the budget feature is fixed at d_test.
    The checkpoints at the horizons off OmniSafe's cadence come from the launcher's
    ``pilot_cfgs.extra_checkpoint_steps`` (``pilot.contracts.extra_checkpoint_steps``: 500,000 for
    the registered horizons), which must hold them. ``params['lr_schedule_steps']`` (``SCHEDULE_PARAM``)
    is accepted only on a continuation shortened by Part 6.1 cut 3 (``pilot.manifest._cut_fewshot_short``:
    ``total_steps`` the first horizon), with the value ``R.FEWSHOT_STEPS`` (Table 9.1, Q-continuations).
    RunRefused for every problem of the spec, the configuration or the parent.
    """
    try:
        run = _as_run(spec)
    except ValueError as exc:
        raise RunRefused(str(exc)) from exc
    problems = _common_problems(env_id, cfgs, run, "study_b_fewshot")
    if run.group != "study_b_fewshot":
        problems.append(f"group {run.group!r} is not study_b_fewshot")
    params = dict(run.params or {})
    missing = [k for k in FEWSHOT_PARAMS if k not in params]
    if missing:
        problems.append(f"params lack {missing}")
    _refuse(run.run_id, problems)
    try:
        budget = _real(params["budget"], "params['budget']")
    except ValueError as exc:
        budget = None
        problems.append(str(exc))
    if budget is not None and budget not in R.UNSEEN_BUDGETS:
        problems.append(f"budget {budget:g} is not an unseen budget {list(R.UNSEEN_BUDGETS)} (Table 2.5)")
    problems += _horizon_problems(run, params["horizons"])
    if SCHEDULE_PARAM in params:
        schedule = params[SCHEDULE_PARAM]
        if isinstance(schedule, bool) or not isinstance(schedule, numbers.Integral) or schedule != R.FEWSHOT_STEPS \
                or run.total_steps != R.FEWSHOT_HORIZONS[0]:
            problems.append(f"params['{SCHEDULE_PARAM}'] {schedule!r} with total_steps {run.total_steps}: only a "
                            f"continuation shortened by Part 6.1 cut 3 to {R.FEWSHOT_HORIZONS[0]} steps keeps the "
                            f"{R.FEWSHOT_STEPS}-step schedule (Table 9.1, Q-continuations)")
    if params["fresh_multiplier_init"] != R.LAGRANGE_MULTIPLIER_INIT:
        problems.append(f"params['fresh_multiplier_init'] is {params['fresh_multiplier_init']!r}, not "
                        f"{R.LAGRANGE_MULTIPLIER_INIT} (Table 2.5: 'a fresh multiplier at 0.001')")
    if run.depends_on != (params["parent_run_id"],):
        problems.append(f"depends_on {run.depends_on} is not the parent ({params['parent_run_id']!r},)")
    if isinstance(params["parent_step"], bool) or not isinstance(params["parent_step"], numbers.Integral) \
            or params["parent_step"] <= 0:
        problems.append(f"params['parent_step'] {params['parent_step']!r} is not a positive whole number of steps")
    required = set(contracts.extra_checkpoint_steps(run)) if not problems else set()
    raw = _section(cfgs, "pilot_cfgs").get("extra_checkpoint_steps") or ()
    try:
        saved = {_whole(s) for s in raw}
    except TypeError:
        problems.append(f"pilot_cfgs.extra_checkpoint_steps {raw!r} are not whole numbers of steps")
        saved = set()
    if required - saved:
        problems.append(f"pilot_cfgs.extra_checkpoint_steps {sorted(saved)} lacks the horizons {sorted(required - saved)} "
                        "(a checkpoint at every horizon)")
    _refuse(run.run_id, problems)
    _refuse(run.run_id, _parent_problems(run, cfgs))
    keys = (float(budget),)
    values = _base_studyb_cfgs(run, FEWSHOT, keys)
    values.update({
        "levels": None, "continuous_range": None, "bin_width": None,
        "budget": float(budget),
        "horizons": [int(h) for h in params["horizons"]],
        "parent_run_id": params["parent_run_id"],
        "parent_step": int(params["parent_step"]),
        "fresh_multiplier_init": float(params["fresh_multiplier_init"]),
    })
    _set_studyb_cfgs(cfgs, values)
    cls = _classes()["FewShotPPOLag"]
    # The spec is set before BaseAlgo.__init__ runs _init, where restore_learner needs it.
    algo = cls.__new__(cls)
    algo._studyb_spec = run
    algo.__init__(env_id=env_id, cfgs=cfgs)
    return algo

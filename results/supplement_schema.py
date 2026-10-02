"""Supplement records: per-run data that the frozen ledger (schema v1) cannot hold.

Owner: Role 4, Analysis and results (Muhammad Abdullah); a new file beside the frozen
``results/ledger_schema.py``, which it never imports or changes (Q-ledger-v2 of HANDOVER.md
section 9, answered in Table 9.1: the ledger schema stays at v1, and these records are kept beside
it). Frozen at ``SUPPLEMENT_SCHEMA_VERSION`` 1, with its SHA-256 in results/supplement_schema.sha256
(as for the ledger schema; ``sha256sum -c`` from the repository root); a later change is an
amendment, with a version bump if a stored record would become invalid. The registered constants it
reads from configs/registered.py are governed by the amendment rule for that file, not by the hash
(tests/test_supplement_schema.py holds their values as recorded at the freeze).

Why a supplement. Appendix B, Table B.1 fixes one row per run, and schema v1 froze that row before
the pilot. The registered analysis needs more than the row holds (analysis spec section 3.2,
inputs L1-L14): the per-episode costs and returns behind every mean (Part 5.3: "the interquartile
mean with a stratified bootstrap over seeds and evaluation episodes"), the measurement-set return
of the matched checkpoint (Part 5.2: "return" is a secondary outcome), the battery at the final
checkpoint and at tolerance 5.0 (Part 4.1 rule 7; Part 4.1.1), the fine-tuning and transfer
continuations (Table 2.2), Study B's satisfaction and violation magnitude at every budget (Part
4.2; Table 2.5), the few-shot horizons of each continuation (Table 2.5 "Adaptation steps"), and
the training quantities (Table 2.1 "Recovery time"; Table 2.4 overshoot and settling time; Table
2.3 metrics at onset and at onset + 200,000 steps for box H3 (c); the intervention summary). A
reference to the "analysis spec" or the "metrics spec" is to the build's specification of that name,
which is kept outside this repository.

One record per (kind, run_id, part). The writer is ``pilot.supplement.write`` (Role 1), which
validates with ``validate_record`` before writing; the reader is ``analysis.data`` (Role 4). The
producers are the harnesses and computations of Roles 2, 3 and 5 (``envs.evaluation``,
``studyb.evaluation``, ``metrics``), mapped into these models by the enrichment commands. A record
carries no timestamp, so re-writing it reproduces the same bytes (``canonical_json``); where it
was produced is recorded by ``code_commit``.

Kinds:
  evaluation           contract 1 (evaluate_run): the final checkpoint and the ten selection checkpoints
  measurement          the matched checkpoint on the measurement set (C_ID and its return)
  battery              one battery condition at the matched checkpoint (part = the condition)
  final_battery        the final checkpoint, rule 7 (b) (part = measurement | hazard | dynamics)
  sensitivity_battery  arms matched at tolerance 5.0 but not 2.5, rule 7 (a) (part = the condition)
  continuation         fine-tuning or transfer continuation of a matched checkpoint (part = the condition)
  zero_shot            a Study B run at the unseen and the reference budgets
  fewshot              one Study B few-shot continuation (part = f"b{budget:g}")
  training             recovery, controller quantities, plasticity rows at onset and at the check, intervention

Strictness: every model forbids unknown fields; every float is finite; floats reject booleans and
strings; integers reject booleans and floats; budgets must be floats (5.0, never 5); a registered
evaluation holds exactly ``R.EVAL_EPISODES`` per-episode values, and every stored mean is checked
against its episodes.

Amendments at version 1, which invalidate no stored record. Each adds an optional field that is left
out of the record's JSON while it is None (``_Amended``), so a record without it has the bytes it
had before the field existed:

* Unstable simulations (Q-mujoco-exception, answered in Table 9.1): an evaluation episode that
  MuJoCo reports unstable is replaced by an episode on the next unused seed of its set's reserve
  sequence. A block of one evaluation call (``EpisodeBlock``, ``BudgetBlock``, ``HorizonBlock``)
  keeps the planned seeds and lists its replacements in ``unstable_replacements``.
* The rate-limited arm (Q-rate-limit, answered in Table 9.1: "the report gives the share of
  constrained epochs in which the clip bound"): the training record's ``rate_limit_clip_share``.
"""

from __future__ import annotations

import json
import math
import statistics
from types import MappingProxyType
from typing import Annotated, Any, ClassVar, Literal, Mapping, Optional

from pydantic import (BaseModel, BeforeValidator, ConfigDict, Field, StrictBool, StrictInt, field_validator,
                      model_serializer, model_validator)

from configs import registered as R

SUPPLEMENT_SCHEMA_VERSION: int = 1

# Means are recomputed from the stored episodes; a producer may sum in another order (numpy rather
# than statistics.fmean), so equality is checked to this relative precision, far below 0.01 (one
# episode's integer cost moves a 100-episode mean by 0.01).
MEAN_REL_TOL = 1e-9
MEAN_ABS_TOL = 1e-12

# Q-mujoco-exception (Table 9.1): the reserve sequence of a seed set is its smallest seed (its base) +
# RESERVE_SEED_OFFSET + r, r = 0, 1, ... within one evaluation call, and at most MAX_UNSTABLE_EPISODES
# episodes of a call are replaced. Local copies of pilot.contracts' numbers (which envs.evaluation
# uses), so that the schema checks a record without importing the pipeline; a test asserts they agree.
RESERVE_SEED_OFFSET = 50_000
MAX_UNSTABLE_EPISODES = 5


# ---------------------------------------------------------------------------
# Field types
# ---------------------------------------------------------------------------


def _real(value: Any) -> Any:
    """A real number: ``int`` or ``float`` (numpy.float64 is a float); never bool, str or numpy ints."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"must be a real number, got {type(value).__name__} {value!r}")
    return value


def _int_only(value: Any) -> Any:
    """An ``int`` for a Literal-typed integer, which pydantic would also match with True or 1.0."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"must be an integer, got {type(value).__name__} {value!r}")
    return value


def _float_only(value: Any) -> Any:
    """A float such as 5.0: budgets are floats in the ledger's maps and in these records."""
    if isinstance(value, bool) or not isinstance(value, float):
        raise ValueError(f"must be a float (write 5.0, not 5), got {type(value).__name__} {value!r}")
    return value


Real = Annotated[float, BeforeValidator(_real)]
NonNegative = Annotated[float, BeforeValidator(_real), Field(ge=0.0)]
Fraction01 = Annotated[float, BeforeValidator(_real), Field(ge=0.0, le=1.0)]
Budget = Annotated[float, BeforeValidator(_float_only), Field(gt=0.0)]
Step = Annotated[StrictInt, Field(ge=0)]
RunId = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._+-]*-s[0-9]+$")]
Commit = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
Episodes = Annotated[list[Real], Field(min_length=R.EVAL_EPISODES, max_length=R.EVAL_EPISODES)]
NonNegativeEpisodes = Annotated[list[NonNegative], Field(min_length=R.EVAL_EPISODES, max_length=R.EVAL_EPISODES)]
Seeds = Annotated[list[Annotated[StrictInt, Field(ge=0)]], Field(min_length=R.EVAL_EPISODES, max_length=R.EVAL_EPISODES)]

STUDY_B_ARM = Literal["Single-10", "Single-20", "Single-40", "Sparse", "Moderate", "Dense", "Continuous"]
SEED_SET = Literal["selection", "measurement", "hazard"]
BATTERY_CONDITION = Literal["hazard", "dynamics"]


def _same_mean(name: str, stored: float, values: list[float]) -> None:
    expected = statistics.fmean(values)
    if not math.isclose(stored, expected, rel_tol=MEAN_REL_TOL, abs_tol=MEAN_ABS_TOL):
        raise ValueError(f"{name} {stored!r} is not the mean of its {len(values)} episodes ({expected!r})")


def _matches_episodes(name: str, stored: float, expected: float) -> None:
    if not math.isclose(stored, expected, rel_tol=MEAN_REL_TOL, abs_tol=MEAN_ABS_TOL):
        raise ValueError(f"{name} {stored!r} does not match its episodes ({expected!r})")


def _distinct(name: str, seeds: list[int]) -> None:
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{name} must be distinct")


class _Model(BaseModel):
    """Strict base: unknown fields are refused and every float must be finite."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, validate_default=True)


class _Amended(_Model):
    """A model with optional fields added after the freeze (``ADDED_FIELDS``), each left out of the record's
    JSON while it is None, so that a record without them has the bytes it had before they existed."""

    ADDED_FIELDS: ClassVar[tuple[str, ...]] = ()

    @model_serializer(mode="wrap")
    def _absent_when_none(self, handler: Any) -> Any:
        data = handler(self)
        for name in self.ADDED_FIELDS:
            if getattr(self, name) is None:
                data.pop(name, None)
        return data


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class UnstableReplacement(_Model):
    """An episode MuJoCo reported unstable and the reserve episode that replaced it (Q-mujoco-exception)."""

    slot_index: Annotated[StrictInt, Field(ge=0, lt=R.EVAL_EPISODES)]
    """The planned episode's index in the block's seeds."""
    seed: Annotated[StrictInt, Field(ge=0)]
    """The unstable episode's seed: the planned one, or the reserve seed of an earlier replacement of the slot."""
    reserve_seed: Annotated[StrictInt, Field(ge=0)]
    """The reserve seed that replaced it: the next unused one of the set's reserve sequence."""
    step: Step
    """The step at which MuJoCo warned (0: at the reset)."""
    warning: Annotated[str, Field(min_length=1)]
    """What MuJoCo reported (e.g. 'mjWARN_BADQVEL x1')."""


def _check_replacements(name: str, replacements: Optional[list[UnstableReplacement]], seeds: list[int]) -> None:
    """The replacements of one evaluation call follow Q-mujoco-exception's rule against its planned seeds.

    In the order made: the slots in the order the episodes ran; each slot's first replacement
    replaces its planned seed and each later one the previous reserve seed; the r-th reserve seed is
    min(seeds) + ``RESERVE_SEED_OFFSET`` + r (so distinct, and in the set's reserve range).
    """
    previous: Optional[UnstableReplacement] = None
    for r, item in enumerate(replacements or ()):
        if previous is not None and item.slot_index < previous.slot_index:
            raise ValueError(f"{name}: replacement {r} of episode {item.slot_index} comes after episode "
                             f"{previous.slot_index}'s")
        same_slot = previous is not None and previous.slot_index == item.slot_index
        expected = previous.reserve_seed if same_slot else seeds[item.slot_index]
        if item.seed != expected:
            raise ValueError(f"{name}: replacement {r} replaces seed {item.seed}; episode {item.slot_index} ran {expected}")
        if item.reserve_seed != min(seeds) + RESERVE_SEED_OFFSET + r:
            raise ValueError(f"{name}: replacement {r} ran reserve seed {item.reserve_seed}, not the next unused one "
                             f"{min(seeds) + RESERVE_SEED_OFFSET + r} (Q-mujoco-exception)")
        previous = item


class _Replaced(_Amended):
    """A block of one evaluation call, with the replacements of its unstable episodes (Q-mujoco-exception)."""

    ADDED_FIELDS: ClassVar[tuple[str, ...]] = ("unstable_replacements",)

    unstable_replacements: Optional[Annotated[list[UnstableReplacement],
                                              Field(min_length=1, max_length=MAX_UNSTABLE_EPISODES)]] = None
    """The replacements in the order made; None (written as absent) when no episode was unstable."""

    @field_validator("unstable_replacements", mode="before")
    @classmethod
    def _none_for_empty(cls, value: Any) -> Any:
        return None if isinstance(value, (list, tuple)) and not value else value


class EpisodeBlock(_Replaced):
    """One evaluation of one checkpoint: 100 episodes of the deterministic policy (Table 2.1).

    ``mean_cost`` and ``mean_return`` are the means the ledger and the other records store (for
    example ``final_cost``, ``measurement_cost``, a battery condition's cost); they are checked
    against the episodes. Episodic cost is the undiscounted sum of the 0/1 per-step cost indicator
    (Safety-Gymnasium 0.4.1 ``builder.py:283-285``), so it is never negative.
    """

    seed_set: SEED_SET
    """Which registered seed set was run: 'selection' and 'measurement' (Table 2.1, disjoint), or
    'hazard' (Table 2.2 hazard relocation: 20 layout seeds x 5 episode seeds, see ``HazardSeeds``)."""
    seeds: Seeds
    """The 100 planned episode reset seeds in the order the episodes were run (distinct, non-negative); an
    episode replaced for an unstable simulation ran the last reserve seed of its ``unstable_replacements``."""
    episode_costs: NonNegativeEpisodes
    """Per-episode undiscounted cost, in the order of ``seeds``."""
    episode_returns: Episodes
    """Per-episode undiscounted return, in the order of ``seeds``."""
    mean_cost: NonNegative
    """Mean of ``episode_costs`` (statistics.fmean)."""
    mean_return: Real
    """Mean of ``episode_returns`` (statistics.fmean)."""
    budgets: Optional[Annotated[list[Budget], Field(min_length=R.EVAL_EPISODES, max_length=R.EVAL_EPISODES)]] = None
    """Per-episode budget in the observation of a budget-conditioned policy (Study B); None otherwise."""

    @model_validator(mode="after")
    def _check(self) -> "EpisodeBlock":
        _distinct("seeds", self.seeds)
        _check_replacements("unstable_replacements", self.unstable_replacements, self.seeds)
        _same_mean("mean_cost", self.mean_cost, self.episode_costs)
        _same_mean("mean_return", self.mean_return, self.episode_returns)
        return self


class CheckpointEpisodes(EpisodeBlock):
    """An ``EpisodeBlock`` of one saved checkpoint (``torch_save/epoch-{step // steps_per_epoch}.pt``)."""

    step: Step
    """The checkpoint's training step."""


class HazardSeeds(_Model):
    """How the hazard-relocation episodes were drawn (Table 2.2; Q-hazard, answered in Table 9.1)."""

    form: Literal["registered", "central"]
    """'central' = the Q-hazard answer of Table 9.1, ``envs.evaluation``'s form (hazards inside the central
    square); 'registered' = the literal Table 2.2 reading, which enters no result and is refused."""
    layout_seeds: Annotated[list[Annotated[StrictInt, Field(ge=0)]], Field(min_length=R.HAZARD_LAYOUTS, max_length=R.HAZARD_LAYOUTS)]
    """The 20 layout seeds (Table 2.2: "twenty new layout seeds"), distinct."""
    episodes_per_layout: Annotated[Literal[R.EPISODES_PER_LAYOUT],  # type: ignore[valid-type]
                                   BeforeValidator(_int_only)]
    """Episodes per layout (Table 2.2: five); the block's 100 seeds are layout-major."""

    @model_validator(mode="after")
    def _check(self) -> "HazardSeeds":
        _distinct("layout_seeds", self.layout_seeds)
        if self.form != "central":
            raise ValueError(f"the hazard layouts are drawn in the 'central' form, not {self.form!r} (Q-hazard, "
                             "Table 9.1: the registered-wording form enters no result)")
        return self


class PlasticityRow(_Model):
    """One row of the run's ``plasticity.csv`` (contract 2; Table 2.3; metrics spec section 5.4).

    A non-finite value in the file is stored as None (the ledger writer's convention).
    """

    step: Step
    """The row's training step."""
    dormant: Optional[Fraction01] = None
    """Actor dormant-neuron fraction, equation (6)."""
    rank: Optional[NonNegative] = None
    """Actor effective rank of the penultimate features, equation (7)."""
    norm: Optional[NonNegative] = None
    """Actor parameter norm over trainable parameters (Table 2.3)."""
    norm_reward_critic: Optional[NonNegative] = None
    """Reward critic parameter norm (Table 2.3: "the two critics logged separately")."""
    norm_cost_critic: Optional[NonNegative] = None
    """Cost critic parameter norm."""
    dormant_trainable: Optional[Fraction01] = None
    """Dormant fraction of the trainable layers' units (box H3 (c))."""
    rank_trainable: Optional[NonNegative] = None
    """Effective rank of the trainable layers' features (box H3 (c))."""
    dormant_reward_critic: Optional[Fraction01] = None
    """Reward critic dormant fraction (Q-plasticity-definitions, Table 9.1: "The same three metrics are logged
    for each critic")."""
    rank_reward_critic: Optional[NonNegative] = None
    """Reward critic effective rank."""
    dormant_cost_critic: Optional[Fraction01] = None
    """Cost critic dormant fraction."""
    rank_cost_critic: Optional[NonNegative] = None
    """Cost critic effective rank."""
    norm_all: Optional[NonNegative] = None
    """Actor norm over all parameters, trainable or not (differs from ``norm`` after injection)."""
    checkpoint: Optional[StrictBool] = None
    """True if a checkpoint was saved at this step."""


class InterventionSummary(_Model):
    """The summary of ``intervention.json`` of a reset or injection run (Table 2.4; HANDOVER.md section 8)."""

    treatment: Literal["reset", "injection"]
    """Which intervention was applied (Table 2.4 "Partial reset" / "Plasticity injection")."""
    step: Step
    """The step at which it was applied (the onset step)."""
    layers: Annotated[list[str], Field(min_length=1)]
    """Names of the actor layers reinitialised (reset) or duplicated into the new head (injection)."""
    generator_seed: Annotated[StrictInt, Field(ge=0)]
    """Seed of the dedicated generator that drew the fresh weights (Q-reset-injection)."""
    trainable_parameters_before: Annotated[StrictInt, Field(ge=0)]
    """Number of trainable actor parameters before the intervention."""
    trainable_parameters_after: Annotated[StrictInt, Field(ge=0)]
    """Number of trainable actor parameters after the intervention."""
    max_output_difference: Optional[NonNegative] = None
    """Max |actor output after - before| on the fixed batch at the moment of injection (0 by equation (8))."""


# ---------------------------------------------------------------------------
# Record kinds
# ---------------------------------------------------------------------------


class _Record(_Model):
    schema_version: Annotated[Literal[SUPPLEMENT_SCHEMA_VERSION],  # type: ignore[valid-type]
                              BeforeValidator(_int_only)] = SUPPLEMENT_SCHEMA_VERSION
    """``SUPPLEMENT_SCHEMA_VERSION``; a change of these models is an amendment (Part 9)."""
    run_id: RunId
    """The ledger row the record belongs to (for continuations: the parent run)."""
    code_commit: Commit
    """The 40-hex commit of the code that produced the record (the evaluating or computing code)."""


class EvaluationRecord(_Record):
    """Contract 1 (``evaluate_run``), per episode: written by the ledger writer before the row.

    Table 2.1: the selection set is run on each of the last ten checkpoints (Part 4.1 rule 1), and
    ``final`` is the final checkpoint on the set of Q-final-cost-set (Table 9.1: the measurement set).
    """

    kind: Literal["evaluation"] = "evaluation"
    """The record kind."""
    final_step: Step
    """The final checkpoint's step (the run's total steps)."""
    final: EpisodeBlock
    """The final checkpoint's evaluation; its means are the ledger's final_cost and final_return."""
    selection: Annotated[list[CheckpointEpisodes], Field(min_length=R.SELECTION_WINDOW_CHECKPOINTS,
                                                         max_length=R.SELECTION_WINDOW_CHECKPOINTS)]
    """The selection set on each of the last ten checkpoints, in increasing step order."""
    short_episodes: Optional[Annotated[StrictInt, Field(ge=0)]] = None
    """Episodes that ended before the 1,000-step limit (harness diagnostic)."""

    @model_validator(mode="after")
    def _check(self) -> "EvaluationRecord":
        steps = [c.step for c in self.selection]
        if steps != sorted(set(steps)):
            raise ValueError("selection checkpoints must have distinct steps in increasing order")
        if any(c.seed_set != "selection" for c in self.selection):
            raise ValueError("the selection checkpoints must be evaluated on the selection set (Table 2.1)")
        if self.final_step != steps[-1]:
            raise ValueError("the final checkpoint must be the last checkpoint of the selection window")
        if self.final.seed_set != "measurement":
            raise ValueError(f"the final checkpoint is evaluated on the measurement set (Q-final-cost-set, Table 9.1), "
                             f"not the {self.final.seed_set} set")
        return self


class MeasurementRecord(_Record):
    """The matched checkpoint on the measurement set (Table 2.1; Part 4.1 rules 4 and 5).

    ``measurement_cost`` is C_ID of equation (1) and the ledger's measurement_cost;
    ``measurement_return`` is the return outcome of Part 5.2 (Q-return-outcome).
    """

    kind: Literal["measurement"] = "measurement"
    """The record kind."""
    step: Step
    """The matched checkpoint's step (ledger matched_checkpoint_step)."""
    episodes: EpisodeBlock
    """The measurement-set episodes."""

    @property
    def measurement_cost(self) -> float:
        return self.episodes.mean_cost

    @property
    def measurement_return(self) -> float:
        return self.episodes.mean_return

    @model_validator(mode="after")
    def _check(self) -> "MeasurementRecord":
        if self.episodes.seed_set != "measurement":
            raise ValueError("the matched checkpoint is measured on the measurement set (Table 2.1)")
        return self


class _ConditionRecord(_Record):
    """One condition at one checkpoint, with C_ID of the same checkpoint for equation (1)."""

    step: Step
    """The evaluated checkpoint's step."""
    condition: Literal["measurement", "hazard", "dynamics"]
    """The battery condition (Table 2.2), or 'measurement' for C_ID itself."""
    episodes: EpisodeBlock
    """The condition's episodes (hazard: the hazard set; dynamics and measurement: the measurement set)."""
    hazard: Optional[HazardSeeds] = None
    """How the hazard layouts were drawn; required for the hazard condition, None otherwise."""
    measurement_cost: Optional[NonNegative] = None
    """C_ID of the same checkpoint (measurement set); None for the measurement condition itself."""
    gap: Optional[Real] = None
    """Equation (1): gap = C_cond - C_ID = episodes.mean_cost - measurement_cost; None for measurement."""

    @model_validator(mode="after")
    def _check(self) -> "_ConditionRecord":
        expected_set = "hazard" if self.condition == "hazard" else "measurement"
        if self.episodes.seed_set != expected_set:
            raise ValueError(f"the {self.condition} condition is run on the {expected_set} set, not {self.episodes.seed_set}")
        if (self.hazard is None) == (self.condition == "hazard"):
            raise ValueError("the hazard seeds are given for the hazard condition and only for it")
        if self.condition == "measurement":
            if self.measurement_cost is not None or self.gap is not None:
                raise ValueError("the measurement condition has no gap")
        else:
            if self.measurement_cost is None or self.gap is None:
                raise ValueError(f"the {self.condition} condition needs measurement_cost and gap (equation 1)")
            expected = self.episodes.mean_cost - self.measurement_cost
            if not math.isclose(self.gap, expected, rel_tol=MEAN_REL_TOL, abs_tol=MEAN_ABS_TOL):
                raise ValueError(f"gap {self.gap!r} is not C_cond - C_ID = {expected!r} (equation 1)")
        return self


class BatteryRecord(_ConditionRecord):
    """A battery condition at the matched checkpoint (Table 2.2; Part 4.1 rule 6: matched arms only)."""

    kind: Literal["battery"] = "battery"
    """The record kind."""
    condition: BATTERY_CONDITION  # type: ignore[assignment]
    """'hazard' or 'dynamics' (fine-tuning and transfer are ``continuation`` records)."""


class FinalBatteryRecord(_ConditionRecord):
    """Part 4.1 rule 7 (b) and Part 4.1.1: the final checkpoint "in place of the selected one"."""

    kind: Literal["final_battery"] = "final_battery"
    """The record kind."""


class SensitivityBatteryRecord(_ConditionRecord):
    """Part 4.1 rule 7 (a): the battery of an arm matched at tolerance 5.0 but not at 2.5."""

    kind: Literal["sensitivity_battery"] = "sensitivity_battery"
    """The record kind."""
    condition: BATTERY_CONDITION  # type: ignore[assignment]
    """'hazard' or 'dynamics'."""


class ContinuationRecord(_Record):
    """Fine-tuning or transfer from the matched checkpoint (Table 2.2; pilot/manifest.py).

    ``run_id`` is the parent Study A run; the gap is C_after - C_ID with C_ID the parent's
    measurement cost at the matched checkpoint (``gap_finetune`` / ``gap_transfer``).
    """

    kind: Literal["continuation"] = "continuation"
    """The record kind."""
    condition: Literal["finetune", "transfer"]
    """Table 2.2 "Reward-only fine-tuning" or "Transfer"."""
    continuation_run_id: RunId
    """The continuation run's own run_id."""
    parent_step: Step
    """The parent's matched checkpoint step the continuation started from."""
    continuation_steps: Annotated[StrictInt, Field(gt=0)]
    """Training steps of the continuation (registered: 1,000,000; Table 2.2)."""
    step: Step
    """The evaluated checkpoint of the continuation (its final one)."""
    task: str
    """The task evaluated (the parent's task for fine-tuning, the held-out task for transfer)."""
    episodes: EpisodeBlock
    """The continuation's final checkpoint on the measurement set."""
    measurement_cost: NonNegative
    """C_ID: the parent's measurement cost at the matched checkpoint."""
    gap: Real
    """episodes.mean_cost - measurement_cost (equation 1)."""

    @model_validator(mode="after")
    def _check(self) -> "ContinuationRecord":
        if self.episodes.seed_set != "measurement":
            raise ValueError("continuations are evaluated on the measurement set")
        if self.step != self.continuation_steps:
            raise ValueError("the continuation is evaluated at its final checkpoint")
        expected = self.episodes.mean_cost - self.measurement_cost
        if not math.isclose(self.gap, expected, rel_tol=MEAN_REL_TOL, abs_tol=MEAN_ABS_TOL):
            raise ValueError(f"gap {self.gap!r} is not C_after - C_ID = {expected!r}")
        return self


def _satisfaction(costs: list[float], budget: float) -> float:
    """Equation (10): the share of episodes whose cost is within the budget (inclusive)."""
    return sum(1 for c in costs if c <= budget) / len(costs)


def _violation(costs: list[float], budget: float) -> float:
    """Table 2.5 "Violation magnitude": the mean over episodes of max(cost - budget, 0)."""
    return statistics.fmean(max(c - budget, 0.0) for c in costs)


def _training_set_distance(arm: str, budget: float) -> float:
    """Equation (11): distance of a budget from the arm's training set (Continuous: its interval)."""
    levels = R.STUDY_B_ARMS[arm]
    if levels is None:
        lo, hi = R.CONTINUOUS_RANGE
        return max(lo - budget, 0.0, budget - hi)
    return min(abs(budget - level) for level in levels)


def reference_budgets(arm: str) -> tuple[float, ...]:
    """The arm's own training levels (Continuous: the range ends), for box G3's drops (Q-g3-arms)."""
    levels = R.STUDY_B_ARMS[arm]
    return tuple(R.CONTINUOUS_RANGE) if levels is None else tuple(levels)


class BudgetBlock(_Replaced):
    """A budget-conditioned policy at one budget: 100 episodes with that budget in the observation."""

    budget: Budget
    """The budget d_test in the observation (a float)."""
    role: Literal["unseen", "reference"]
    """'unseen' = Table 2.5's unseen budgets; 'reference' = the arm's training levels (box G3)."""
    episode_costs: NonNegativeEpisodes
    """Per-episode cost, in the order of the record's seeds."""
    episode_returns: Episodes
    """Per-episode return."""
    mean_cost: NonNegative
    """Mean episode cost."""
    mean_return: Real
    """Mean episode return."""
    satisfaction: Fraction01
    """Equation (10): share of episodes with cost <= budget."""
    violation: NonNegative
    """Table 2.5 violation magnitude: mean of max(cost - budget, 0)."""
    distance: Optional[NonNegative] = None
    """Equation (11) distance from the arm's training set (unseen budgets only)."""

    @model_validator(mode="after")
    def _check(self) -> "BudgetBlock":
        _same_mean("mean_cost", self.mean_cost, self.episode_costs)
        _same_mean("mean_return", self.mean_return, self.episode_returns)
        _matches_episodes("satisfaction", self.satisfaction, _satisfaction(self.episode_costs, self.budget))
        _matches_episodes("violation", self.violation, _violation(self.episode_costs, self.budget))
        if (self.distance is None) != (self.role == "reference"):
            raise ValueError("an unseen budget carries its equation (11) distance; a reference budget does not")
        return self


class ZeroShotRecord(_Record):
    """A Study B run's final checkpoint at the unseen budgets and at its own training levels.

    Table 2.5 "Unseen budgets"; equation (10); Part 4.2 (floor rule and G3's drops "measured from
    the arm's satisfaction at its own nearest training level", Q-g3-arms). The ledger's ``sr_zero``
    holds the four unseen satisfactions.
    """

    kind: Literal["zero_shot"] = "zero_shot"
    """The record kind."""
    arm: STUDY_B_ARM
    """The Study B arm (Table 3.4)."""
    step: Step
    """The evaluated checkpoint (the final one; Q-studyb-eval)."""
    seed_set: SEED_SET
    """The episode seed set, the same at every budget (Q-studyb-eval, Table 9.1: the measurement set)."""
    seeds: Seeds
    """The 100 episode seeds, shared by every budget."""
    budgets: Annotated[list[BudgetBlock], Field(min_length=1)]
    """One block per budget: the four unseen budgets and the arm's reference budgets."""

    @model_validator(mode="after")
    def _check(self) -> "ZeroShotRecord":
        _distinct("seeds", self.seeds)
        for block in self.budgets:
            _check_replacements(f"unstable_replacements at budget {block.budget:g}", block.unstable_replacements,
                                self.seeds)
        if self.seed_set != "measurement":
            raise ValueError(f"zero-shot evaluations use the measurement set (Q-studyb-eval, Table 9.1), not the "
                             f"{self.seed_set} set")
        values = [b.budget for b in self.budgets]
        if len(values) != len(set(values)):
            raise ValueError("each budget appears once")
        unseen = sorted(b.budget for b in self.budgets if b.role == "unseen")
        reference = sorted(b.budget for b in self.budgets if b.role == "reference")
        if unseen != sorted(R.UNSEEN_BUDGETS):
            raise ValueError(f"the unseen budgets must be exactly {list(R.UNSEEN_BUDGETS)}, got {unseen}")
        if reference != sorted(reference_budgets(self.arm)):
            raise ValueError(f"the reference budgets of {self.arm} are {list(reference_budgets(self.arm))}, got {reference}")
        for block in self.budgets:
            if block.role == "unseen":
                expected = _training_set_distance(self.arm, block.budget)
                if not math.isclose(block.distance, expected, abs_tol=MEAN_ABS_TOL):
                    raise ValueError(f"distance of budget {block.budget} is {expected} by equation (11), not {block.distance}")
        return self

    @property
    def sr_zero(self) -> dict[float, float]:
        """The ledger's ``sr_zero`` map: satisfaction at each unseen budget."""
        return {b.budget: b.satisfaction for b in self.budgets if b.role == "unseen"}


def fewshot_key(budget: float, horizon: int) -> str:
    """The frozen ledger's ``sr_fewshot`` key form, e.g. '5.0_200000' (results/ledger_schema.py,
    ``_v_sr_fewshot_keys``).

    ``pilot.contracts.fewshot_key`` is the pipeline's helper; this local copy only lets the schema
    check a record without importing the pipeline, and a test asserts the two agree.
    """
    return f"{float(budget)}_{int(horizon)}"


class HorizonBlock(_Replaced):
    """A few-shot continuation's checkpoint at one horizon; satisfaction and violation are at the continuation's
    budget (``FewshotRecord.budget``)."""

    horizon: Annotated[StrictInt, Field(gt=0)]
    """Continuation steps at this checkpoint (Table 2.5: 200,000, 500,000, 1,000,000)."""
    episode_costs: NonNegativeEpisodes
    """Per-episode cost."""
    episode_returns: Episodes
    """Per-episode return."""
    mean_cost: NonNegative
    """Mean episode cost."""
    mean_return: Real
    """Mean episode return."""
    satisfaction: Fraction01
    """Equation (10) at the continuation's budget."""
    violation: NonNegative
    """Violation magnitude at the continuation's budget."""


class FewshotRecord(_Record):
    """One few-shot continuation of a Study B run at one unseen budget (Table 2.5; Part 3.3).

    ``run_id`` is the parent Study B run. "The three horizons are read from one continuation, not
    three" (Table 2.5). ``adapt_steps`` is "The smallest few-shot horizon at which the satisfaction
    rate reaches 0.80 (decision); recorded as above the largest horizon if never reached": the
    censored value is ``max(horizons) + 1`` of THIS continuation (``studyb.evaluation.censored_steps``;
    Q-adapt-censoring, Table 9.1), so a reader decodes censoring as ``adapt_steps > max(horizons)``, never by
    equality with a constant.
    """

    kind: Literal["fewshot"] = "fewshot"
    """The record kind."""
    continuation_run_id: RunId
    """The continuation run's own run_id."""
    parent_step: Step
    """The parent run's checkpoint step the continuation restored, from its spec's ``parent_step``
    (pilot/manifest.py: ``parent.total_steps``, the parent's final checkpoint, from which Study B few-shot
    continuations start (Q-continuations, Table 9.1)); the record names the checkpoint the continuation restored."""
    arm: STUDY_B_ARM
    """The parent's Study B arm."""
    budget: Budget
    """The unseen budget d_test of the continuation (a float)."""
    horizons: Annotated[list[Annotated[StrictInt, Field(gt=0)]], Field(min_length=1)]
    """The continuation's horizons, from its spec: Table 2.5's three, or 200,000 only under Part 6.1 cut 3
    (Q-studyb-eval, Table 9.1)."""
    seed_set: SEED_SET
    """The episode seed set (Q-studyb-eval, Table 9.1: the measurement set)."""
    seeds: Seeds
    """The 100 episode seeds, shared by every horizon."""
    by_horizon: Annotated[list[HorizonBlock], Field(min_length=1)]
    """One block per horizon, in increasing horizon order."""
    sr_fewshot: dict[str, Fraction01]
    """The ledger's sr_fewshot entries of this continuation, keyed '5.0_200000' (``fewshot_key``)."""
    adapt_steps: Annotated[StrictInt, Field(gt=0)]
    """Smallest horizon whose satisfaction reaches R.SATISFACTION_TARGET, else max(horizons) + 1."""

    @field_validator("horizons")
    @classmethod
    def _horizons(cls, value: list[int]) -> list[int]:
        allowed = (list(R.FEWSHOT_HORIZONS), [R.FEWSHOT_HORIZONS[0]])
        if value not in allowed:
            raise ValueError(f"horizons {value} are neither Table 2.5's {allowed[0]} nor Part 6.1 cut 3's {allowed[1]} "
                             "(Q-studyb-eval, Table 9.1)")
        return value

    @model_validator(mode="after")
    def _check(self) -> "FewshotRecord":
        _distinct("seeds", self.seeds)
        if self.seed_set != "measurement":
            raise ValueError(f"few-shot evaluations use the measurement set (Q-studyb-eval, Table 9.1), not the "
                             f"{self.seed_set} set")
        if self.budget not in R.UNSEEN_BUDGETS:
            raise ValueError(f"few-shot budget {self.budget} is not an unseen budget {list(R.UNSEEN_BUDGETS)}")
        if [b.horizon for b in self.by_horizon] != self.horizons:
            raise ValueError("by_horizon must hold exactly the continuation's horizons, in order")
        for block in self.by_horizon:
            _check_replacements(f"unstable_replacements at horizon {block.horizon}", block.unstable_replacements,
                                self.seeds)
            _same_mean("mean_cost", block.mean_cost, block.episode_costs)
            _same_mean("mean_return", block.mean_return, block.episode_returns)
            _matches_episodes("satisfaction", block.satisfaction, _satisfaction(block.episode_costs, self.budget))
            _matches_episodes("violation", block.violation, _violation(block.episode_costs, self.budget))
        expected_keys = {fewshot_key(self.budget, h): b.satisfaction for h, b in zip(self.horizons, self.by_horizon)}
        if set(self.sr_fewshot) != set(expected_keys):
            raise ValueError(f"sr_fewshot keys must be {sorted(expected_keys)}, got {sorted(self.sr_fewshot)}")
        for key, value in expected_keys.items():
            if self.sr_fewshot[key] != value:
                raise ValueError(f"sr_fewshot[{key!r}] differs from the horizon's satisfaction")
        reached = [b.horizon for b in self.by_horizon if b.satisfaction >= R.SATISFACTION_TARGET]
        expected = reached[0] if reached else max(self.horizons) + 1
        if self.adapt_steps != expected:
            raise ValueError(f"adapt_steps must be {expected} (smallest horizon reaching {R.SATISFACTION_TARGET}, "
                             f"else the largest horizon + 1), got {self.adapt_steps}")
        return self

    @property
    def censored(self) -> bool:
        """True when the target was never reached: adapt_steps lies above the largest horizon."""
        return self.adapt_steps > max(self.horizons)


class TrainingRecord(_Record, _Amended):
    """Training quantities of one Study A run: those the ledger does not hold (recovery, overshoot, the
    plasticity rows, the intervention, the rate clip's share) and the controller quantities it does.

    Table 2.1 "Recovery time" ("Training steps from onset until the training-batch mean episodic
    cost first falls to or below d. Secondary outcome, late arms only"); Table 2.4 "Multiplier
    overshoot" and "Settling time"; Table 2.3 metrics at onset and at onset + 200,000 steps (box H3
    (c) manipulation check); the intervention of Table 2.4. Computed by Role 3's
    ``metrics.recovery`` / ``metrics.controller`` under Q-controller-quantities.
    """

    ADDED_FIELDS: ClassVar[tuple[str, ...]] = ("rate_limit_clip_share",)

    kind: Literal["training"] = "training"
    """The record kind."""
    onset_step: Step
    """The run's onset step (0 for N = 0; the spec's onset_step)."""
    total_steps: Annotated[StrictInt, Field(gt=0)]
    """The run's total training steps (spec.total_steps)."""
    steps_per_epoch: Annotated[StrictInt, Field(gt=0)]
    """The run's own steps per epoch (config.json algo_cfgs.steps_per_epoch)."""
    recovery_steps: Optional[Step] = None
    """Steps from onset to the end of the first constrained epoch with batch mean cost <= d; None
    if never reached (then ``recovery_censored``) or for N = 0."""
    recovery_censored: Optional[StrictBool] = None
    """True if the batch cost never fell to d after onset (right-censored at the end of training);
    None for N = 0 (recovery is defined for late arms only)."""
    lambda_peak: Optional[NonNegative] = None
    """Maximum multiplier over epochs ending in (onset, onset + 2,000,000] (Table 2.4)."""
    lambda_final: Optional[NonNegative] = None
    """Mean multiplier over the last 10 percent of training epochs (Table 2.4)."""
    overshoot: Optional[Real] = None
    """lambda_peak - lambda_final (Table 2.4 "Multiplier overshoot")."""
    settling_steps: Optional[Step] = None
    """Steps from onset until the multiplier stays within +/-10 percent of lambda_final (Table 2.4
    "Settling time"; with lambda_final 0, until it stays exactly 0); None only if the last epoch's value
    lies outside the band (Q-controller-quantities, Table 9.1)."""
    rate_limit_clip_share: Optional[Fraction01] = None
    """The rate-limited arm only: the share of constrained epochs in which the rate clip bound
    (Onset/MultiplierProposed differs from the multiplier; Q-rate-limit, Table 9.1); None (written as
    absent) for every other run."""
    plasticity_onset: Optional[PlasticityRow] = None
    """The plasticity.csv row at onset (box H3 (a): "the metrics at onset")."""
    plasticity_check: Optional[PlasticityRow] = None
    """The plasticity.csv row at onset + 200,000 steps (box H3 (c) manipulation check)."""
    intervention: Optional[InterventionSummary] = None
    """The intervention summary of a reset or injection run; None otherwise."""

    @model_validator(mode="after")
    def _check(self) -> "TrainingRecord":
        if self.onset_step > self.total_steps:
            raise ValueError("onset_step lies after the end of training")
        late = self.onset_step > 0
        if not late and (self.recovery_steps is not None or self.recovery_censored is not None):
            raise ValueError("recovery time is defined for late arms only (Table 2.1)")
        if late and self.recovery_censored is not None:
            if self.recovery_censored != (self.recovery_steps is None):
                raise ValueError("recovery_steps is None exactly when recovery is censored")
        if self.recovery_steps is not None and self.onset_step + self.recovery_steps > self.total_steps:
            raise ValueError("recovery ends after the end of training")
        if self.overshoot is not None:
            if self.lambda_peak is None or self.lambda_final is None:
                raise ValueError("overshoot needs lambda_peak and lambda_final")
            if not math.isclose(self.overshoot, self.lambda_peak - self.lambda_final,
                                rel_tol=MEAN_REL_TOL, abs_tol=MEAN_ABS_TOL):
                raise ValueError("overshoot must equal lambda_peak - lambda_final (Table 2.4)")
        if self.plasticity_onset is not None and self.plasticity_onset.step != self.onset_step:
            raise ValueError("plasticity_onset must be the row at the onset step")
        if self.plasticity_check is not None:
            check = self.onset_step + R.MANIPULATION_CHECK_STEPS_AFTER_ONSET
            if self.plasticity_check.step != check:
                raise ValueError(f"plasticity_check must be the row at onset + "
                                 f"{R.MANIPULATION_CHECK_STEPS_AFTER_ONSET} = {check}")
        if self.intervention is not None and self.intervention.step != self.onset_step:
            raise ValueError("the intervention is applied at onset (Table 2.4)")
        if self.rate_limit_clip_share is not None and not late:
            raise ValueError("the rate clip's share is a late arm's (Table 3.3: the rate-limited arm has an onset)")
        return self


KIND_MODELS: Mapping[str, type[_Record]] = MappingProxyType({
    "evaluation": EvaluationRecord,
    "measurement": MeasurementRecord,
    "battery": BatteryRecord,
    "final_battery": FinalBatteryRecord,
    "sensitivity_battery": SensitivityBatteryRecord,
    "continuation": ContinuationRecord,
    "zero_shot": ZeroShotRecord,
    "fewshot": FewshotRecord,
    "training": TrainingRecord,
})
KINDS: tuple[str, ...] = tuple(KIND_MODELS)
# Kinds with several records per run; the part names the record within the run.
PART_KINDS: frozenset[str] = frozenset({"battery", "final_battery", "sensitivity_battery", "continuation", "fewshot"})


def validate_record(kind: str, record: Mapping[str, Any] | BaseModel) -> _Record:
    """Validate ``record`` as a supplement record of ``kind``; returns the model.

    Raises ValueError for an unknown kind, a model of another kind, a record that is not a mapping, or a
    record whose own ``kind`` differs, and pydantic's ValidationError (a ValueError) for a record that breaks
    the model.
    """
    if not isinstance(kind, str) or kind not in KIND_MODELS:  # an unhashable kind is refused, not a TypeError
        raise ValueError(f"unknown supplement kind {kind!r}; choose from {list(KIND_MODELS)}")
    model = KIND_MODELS[kind]
    if isinstance(record, BaseModel):
        if not isinstance(record, model):
            raise ValueError(f"a {type(record).__name__} is not a {kind} record")
        record = record.model_dump(mode="python")
    if not isinstance(record, Mapping):
        raise ValueError(f"a {kind} record must be a mapping, got {type(record).__name__}")
    if record.get("kind", kind) != kind:
        raise ValueError(f"the record says kind {record.get('kind')!r}, not {kind!r}")
    return model.model_validate(dict(record))


def record_part(kind: str, record: Mapping[str, Any] | BaseModel) -> Optional[str]:
    """The part that names a record within its run, from the record's own content.

    battery / final_battery / sensitivity_battery / continuation: the condition; fewshot:
    f"b{budget:g}"; the other kinds have one record per run (None).
    """
    # an already-built model of this kind is used as it is; any other record (a model of another kind too) is validated
    model = record if isinstance(kind, str) and isinstance(record, KIND_MODELS.get(kind, ())) \
        else validate_record(kind, record)
    if kind not in PART_KINDS:
        return None
    if kind == "fewshot":
        return f"b{model.budget:g}"  # type: ignore[attr-defined]
    return str(model.condition)  # type: ignore[attr-defined]


def canonical_json(kind: str, record: Mapping[str, Any] | BaseModel) -> str:
    """The record as canonical JSON text (validated, sorted keys, no NaN), so rewrites are byte-identical."""
    # an already-built model of this kind is used as it is (as in record_part); any other record is validated
    model = record if isinstance(kind, str) and isinstance(record, KIND_MODELS.get(kind, ())) \
        else validate_record(kind, record)
    return json.dumps(model.model_dump(mode="json"), sort_keys=True, indent=1, allow_nan=False, ensure_ascii=True) + "\n"

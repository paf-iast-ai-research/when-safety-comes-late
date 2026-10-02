"""The robustness battery's continuation plug-ins: reward-only fine-tuning and transfer (Table 2.2).

Owner: Environment and tests (Role 2; Muhammad Talha Jamil). ``pilot.manifest.PLUGINS`` names them:
``battery_finetune -> make_finetune_algorithm`` and ``battery_transfer -> make_transfer_algorithm``;
the specs come from ``pilot.manifest.battery_continuation`` (pilot/manifest.py).

What the pre-registration fixes (Table 2.2, PDF p. 8):

* Table 2.2 "Reward-only fine-tuning": "Training continues for 1,000,000 steps, one tenth of T
  (decision), with the cost term removed from the objective and the multiplier frozen at zero;
  evaluation cost is measured after fine-tuning on the training task."
* Table 2.2 "Transfer": "The policy is fine-tuned for 1,000,000 steps, one tenth of T (decision), on
  the held-out task on the same robot (Goal to Button and Button to Goal) under the same budget d;
  evaluation cost is measured on the held-out task after fine-tuning."

What Table 9.1 decides (each continuation spec carries the keys as run gates):

* Q-continuations: the continuation starts from the parent's matched checkpoint
  (``spec.params['parent_step']``); the parent's actor, both critics and observation normaliser are
  restored and the optimisers and the actor's learning-rate schedule (OmniSafe's LinearLR from 1 to
  0) start fresh over the continuation's full registered length, 50 epochs, by the one restore rule
  of every continuation, ``pilot.dependencies.restore_learner``, called in ``_init`` after OmniSafe
  has built the networks and before ``_init_log``, so ``epoch-0.pt`` holds the restored learner.
  Fine-tuning holds the multiplier at 0 and trains the actor on the reward advantage alone; transfer
  starts a fresh multiplier at 0.001 (Table 2.5's few-shot rule, borrowed) with the registered
  d = 25; both train with PPO-Lagrangian, a PID-arm parent's continuation included
  (``pilot.manifest.CONTINUATION_BASE_ALGO``).
* Q-transfer-obs: the Goal and Button tasks observe different components (60 against 76 inputs for
  Point, 72 against 88 for Car), and Table 2.2 does not name SafetyCarGoal1-v0's held-out task (Table
  9.1: SafetyCarButton1-v0, ``pilot.manifest.TRANSFER_TASK_PROPOSALS``). The mapping is by component
  name (``transfer_map``, read at run time from Safety-Gymnasium's observation dictionary, never
  from hard-coded offsets): the first layer's columns of a shared component are copied, a
  component only the target has starts at zero weight (``TARGET_ONLY_WEIGHT``), a component only
  the source has is dropped (``map_state_for_transfer``). The target-sized normaliser copies the
  shared statistics; a target-only dimension gets mean 0 and variance 1 (``TARGET_ONLY_MEAN``,
  ``TARGET_ONLY_VARIANCE``) with ``_sumsq = _count - 1``, under the parent's single ``_count``
  (OmniSafe's ``Normalizer`` keeps one count for every dimension, ``omnisafe/common/normalizer.py:56``,
  and recomputes the variance from ``_sumsq`` at every push, ``omnisafe/common/normalizer.py:137``),
  so its variance stays near 1 after the next push. These three values are Table 9.1's, named
  constants of this module (HANDOVER.md section 10). A shared name is not a shared meaning:
  ``goal_lidar`` points at the goal location in Goal and at the goal button in Button
  (``safety_gymnasium/tasks/button/button_level0.py:68-74``).

Every problem of configuration or specification raises ``pilot.errors.RunRefused`` (exit 5, the run
stays pending), never another exception. The plug-ins write ``continuation.json`` (the restore
summary) into the OmniSafe run directory. They log ``Metrics/LagrangeMultiplier`` (0.0 for
fine-tuning) in every epoch (Part 3.4: "The multiplier, or in Study B every per-level multiplier, is
logged at every epoch"; pilot/contracts.py contract 3), and the diagnostic ``Onset/Active`` (0 for
fine-tuning, 1 for transfer) in every epoch, as ``envs.onset`` logs it (contract 3; HANDOVER.md
section 8). They install no plasticity hook (Table 2.3's logging points are those of the training
runs, ``pilot.contracts.plasticity_required``).
"""

from __future__ import annotations

import functools
import math
import numbers
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import safety_gymnasium
import torch
from omnisafe.algorithms.on_policy.naive_lagrange.ppo_lag import PPOLag
from omnisafe.utils.config import Config

from configs import registered as R
# The helpers and names this module shares with the Study A plug-in (one definition each):
from envs.onset import (ACTIVE_KEY, ACTOR_TYPE, OWNER, SMOKE_PREFIX, _as_spec, _config_value, _plain, _refuse,
                        _whole_number, write_json_once)
from pilot import dependencies, rundir
from pilot.algorithms import FullStateCheckpointMixin
from pilot.errors import RunRefused
from pilot.manifest import BATTERY_CONDITIONS, CONTINUATION_BASE_ALGO, STUDY_A_PARENT_GROUPS, RunSpec, transfer_task

CONTINUATION_FILE = "continuation.json"
CONDITIONS = BATTERY_CONDITIONS  # Table 2.2: ("finetune", "transfer")
CONDITION_STEPS = {"finetune": R.FINETUNE_STEPS, "transfer": R.TRANSFER_STEPS}  # Table 2.2; Table 8.2
# ``Onset/Active`` in every epoch (a diagnostic of contract 3, pilot/contracts.py): 0 for fine-tuning (no cost term,
# the multiplier frozen at zero), 1 for transfer (constrained from its first epoch), as envs.onset logs it.
CONDITION_ACTIVE = {"finetune": 0.0, "transfer": 1.0}
# The first Linear layer of each network, the only tensors whose shape depends on the observation size.
ACTOR_FIRST_WEIGHTS = ("mean.0.weight", "mean.trunk.0.weight")  # plain and injected actor (HANDOVER.md section 8)
CRITIC_FIRST_WEIGHT = "critic_0.0.weight"  # omnisafe/models/critic/v_critic.py:73: add_module(f'critic_{idx}', net)
NORMALISER_VECTORS = ("_mean", "_sumsq", "_var", "_std", "_clip")  # omnisafe/common/normalizer.py:52-57
# How transfer starts an observation component only the held-out task has (Table 9.1, Q-transfer-obs): its
# first-layer columns start at weight 0, its normaliser dimensions at mean 0 and variance 1.
TARGET_ONLY_WEIGHT = 0.0  # answered in Table 9.1 (Q-transfer-obs): its first-layer columns, no effect at the start
TARGET_ONLY_MEAN = 0.0  # answered in Table 9.1 (Q-transfer-obs): its normaliser mean
TARGET_ONLY_VARIANCE = 1.0  # answered in Table 9.1 (Q-transfer-obs): its normaliser variance (std = its square root)


# ---------------------------------------------------------------------------
# The observation mapping of transfer (Q-transfer-obs)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def observation_layout(task: str) -> tuple[tuple[str, int], ...]:
    """The flat observation of ``task`` as ``((component, size), ...)`` in the order OmniSafe sees it.

    Safety-Gymnasium 0.4.1 builds ``obs_space_dict`` once in the builder
    (``safety_gymnasium/bases/base_task.py:214-249``: the agent's sensors, then each observed
    obstacle's lidar, in an ``OrderedDict``) and flattens every observation in that order
    (``safety_gymnasium/bases/base_task.py:392``:
    ``gymnasium.spaces.utils.flatten(self.obs_info.obs_space_dict, obs)``). Read from a freshly made
    environment; the global random states (Python, numpy, torch) are restored afterwards, whether
    or not the environment can be made, so asking for a layout changes no stream. RunRefused if the
    task cannot be made or its layout does not add up to its flat observation space.
    """
    states = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    try:
        try:
            env = safety_gymnasium.make(task)
        except Exception as exc:  # noqa: BLE001 - an unknown task is a refusal, never a crash
            raise RunRefused(f"{task} cannot be made by Safety-Gymnasium: {exc}") from exc
        try:
            spaces = env.unwrapped.obs_space_dict.spaces
            layout = tuple((str(name), int(np.prod(space.shape))) for name, space in spaces.items())
            flat = int(np.prod(env.observation_space.shape))
        finally:
            env.close()
    finally:
        random.setstate(states[0])
        np.random.set_state(states[1])
        torch.set_rng_state(states[2])
    if sum(size for _, size in layout) != flat:
        raise RunRefused(f"{task}: the observation components {layout} do not add up to its {flat} inputs")
    return layout


def _as_json(value: Any) -> Any:
    """``value`` as JSON reads it back: tuples, at every depth, become lists."""
    if isinstance(value, Mapping):
        return {k: _as_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_json(v) for v in value]
    return value


@dataclass(frozen=True)
class TransferMap:
    """Which source observation columns feed which target columns (Q-transfer-obs), by component name.

    ``source_layout`` and ``target_layout`` are ``((component, size), ...)`` in flat order
    (``observation_layout``). Shared components (same name) must have the same size; at least one
    must be shared. Columns are 0-based indices into the flat observation.
    """

    source_task: str
    target_task: str
    source_layout: tuple[tuple[str, int], ...]
    target_layout: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        for name, layout in (("source", self.source_layout), ("target", self.target_layout)):
            if any(isinstance(size, bool) or not isinstance(size, int) for _, size in layout):
                raise RunRefused(f"the {name} observation layout {layout} has a size that is not an integer")
            names = [c for c, _ in layout]
            if len(set(names)) != len(names) or any(size <= 0 for _, size in layout):
                raise RunRefused(f"the {name} observation layout {layout} repeats a component or has an empty one")
        sizes = dict(self.source_layout)
        shared = [(c, s) for c, s in self.target_layout if c in sizes]
        if not shared:
            raise RunRefused(f"{self.source_task} and {self.target_task} share no observation component")
        wrong = [c for c, s in shared if sizes[c] != s]
        if wrong:
            raise RunRefused(f"components {wrong} have different sizes in {self.source_task} and {self.target_task}")

    @staticmethod
    def _offsets(layout: tuple[tuple[str, int], ...]) -> dict[str, range]:
        out, start = {}, 0
        for name, size in layout:
            out[name] = range(start, start + size)
            start += size
        return out

    @property
    def source_dim(self) -> int:
        """The number of inputs of the source task's flat observation."""
        return sum(size for _, size in self.source_layout)

    @property
    def target_dim(self) -> int:
        """The number of inputs of the target task's flat observation."""
        return sum(size for _, size in self.target_layout)

    @property
    def shared(self) -> tuple[str, ...]:
        """Components both tasks observe, in the target's order."""
        source = dict(self.source_layout)
        return tuple(c for c, _ in self.target_layout if c in source)

    @property
    def source_only(self) -> tuple[str, ...]:
        """Components only the source observes, in the source's order."""
        target = dict(self.target_layout)
        return tuple(c for c, _ in self.source_layout if c not in target)

    @property
    def target_only(self) -> tuple[str, ...]:
        """Components only the target observes, in the target's order."""
        source = dict(self.source_layout)
        return tuple(c for c, _ in self.target_layout if c not in source)

    def column_pairs(self) -> tuple[tuple[int, int], ...]:
        """``(target column, source column)`` for every column of a shared component, in target order."""
        src, tgt = self._offsets(self.source_layout), self._offsets(self.target_layout)
        return tuple(pair for name in self.shared for pair in zip(tgt[name], src[name]))

    def target_only_columns(self) -> tuple[int, ...]:
        """The target columns of the target-only components, in the target's order."""
        tgt = self._offsets(self.target_layout)
        return tuple(i for name in self.target_only for i in tgt[name])

    def source_only_columns(self) -> tuple[int, ...]:
        """The source columns of the source-only components (dropped by the mapping), in the source's order."""
        src = self._offsets(self.source_layout)
        return tuple(i for name in self.source_only for i in src[name])

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready (for ``cfgs['continuation_cfgs']`` and ``continuation.json``); ``from_dict`` inverts it."""
        return {
            "source_task": self.source_task,
            "target_task": self.target_task,
            "source_layout": [[c, s] for c, s in self.source_layout],
            "target_layout": [[c, s] for c, s in self.target_layout],
            "source_dim": self.source_dim,
            "target_dim": self.target_dim,
            "shared": list(self.shared),
            "source_only": list(self.source_only),
            "target_only": list(self.target_only),
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> TransferMap:
        """The map ``to_dict`` wrote; RunRefused if a derived field disagrees with the layouts."""
        try:
            tmap = TransferMap(
                source_task=str(data["source_task"]),
                target_task=str(data["target_task"]),
                source_layout=tuple((str(c), s) for c, s in data["source_layout"]),
                target_layout=tuple((str(c), s) for c, s in data["target_layout"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RunRefused(f"not a transfer map: {exc}") from exc
        if tmap.to_dict() != _as_json(_plain(data)):
            raise RunRefused("the transfer map's derived fields disagree with its layouts")
        return tmap


def transfer_map(source_task: str, target_task: str) -> TransferMap:
    """The component-name mapping from ``source_task``'s observation to ``target_task``'s (Q-transfer-obs).

    Both layouts are read at run time (``observation_layout``). For the registered pairs of Table 2.2
    ("Goal to Button and Button to Goal"), and SafetyCarGoal1-v0 -> SafetyCarButton1-v0 (Table 9.1,
    Q-transfer-obs), the sensors (accelerometer, velocimeter, gyro, magnetometer, and for the Car
    robot also ``ballangvel_rear`` and ``ballquat_rear``), ``goal_lidar`` and ``hazards_lidar`` are
    shared; ``vases_lidar`` is Goal's alone and ``buttons_lidar`` and ``gremlins_lidar`` are Button's
    alone.
    """
    if source_task == target_task:
        raise RunRefused(f"transfer maps one task to another, got {source_task} twice")
    return TransferMap(source_task, target_task, observation_layout(source_task), observation_layout(target_task))


def _column_index(tmap: TransferMap) -> tuple[torch.Tensor, torch.Tensor]:
    """The target and the source columns of ``tmap.column_pairs()``, as index tensors."""
    pairs = tmap.column_pairs()
    return (torch.tensor([t for t, _ in pairs], dtype=torch.long),
            torch.tensor([s for _, s in pairs], dtype=torch.long))


def _as_tensor(value: Any, what: str) -> torch.Tensor:
    """``value`` as a tensor; RunRefused for anything that is not numeric."""
    try:
        return torch.as_tensor(value)
    except (TypeError, ValueError, OverflowError, RuntimeError) as exc:
        raise RunRefused(f"{what} is not a numeric tensor: {exc}") from exc


def _map_first_layer(weight: Any, tmap: TransferMap, what: str) -> torch.Tensor:
    """``weight`` (``(hidden, source_dim)``) as ``(hidden, target_dim)``: shared columns copied."""
    weight = _as_tensor(weight, what)
    if weight.ndim != 2 or weight.shape[1] != tmap.source_dim:
        raise RunRefused(f"{what} has shape {tuple(weight.shape)}; the source {tmap.source_task} has "
                         f"{tmap.source_dim} inputs")
    mapped = torch.full((weight.shape[0], tmap.target_dim), TARGET_ONLY_WEIGHT, dtype=weight.dtype)
    target_cols, source_cols = _column_index(tmap)
    mapped[:, target_cols] = weight[:, source_cols]
    return mapped


def _map_network(state: Mapping[str, Any], keys: tuple[str, ...], tmap: TransferMap, what: str) -> dict[str, Any]:
    """``state`` with its one first-layer weight (exactly one of ``keys``) mapped; RunRefused otherwise."""
    if not isinstance(state, Mapping):
        raise RunRefused(f"{what}: a state dict is required, got {type(state).__name__}")
    present = [k for k in keys if k in state]
    if len(present) != 1:
        raise RunRefused(f"{what}: expected exactly one first-layer weight of {keys}, found {present}")
    out = dict(state)
    out[present[0]] = _map_first_layer(state[present[0]], tmap, f"{what} {present[0]}")
    return out


def _map_normaliser(norm: Mapping[str, Any], tmap: TransferMap) -> dict[str, Any]:
    """The target-sized normaliser state: shared statistics copied; target-only dimensions at
    ``TARGET_ONLY_MEAN`` and ``TARGET_ONLY_VARIANCE``.

    OmniSafe recomputes ``_var = _sumsq / (_count - 1)`` at every push
    (``omnisafe/common/normalizer.py:137``), so a target-only dimension gets ``_sumsq =
    TARGET_ONLY_VARIANCE · (_count - 1)`` under the parent's ``_count``; its variance then stays near
    ``TARGET_ONLY_VARIANCE`` after the next push.
    """
    if not isinstance(norm, Mapping):
        raise RunRefused(f"the parent's observation normaliser must be a state dict, got {type(norm).__name__}")
    missing = [k for k in (*NORMALISER_VECTORS, "_count") if k not in norm]
    if missing:
        raise RunRefused(f"the parent's observation normaliser lacks {missing}")
    count = _as_tensor(norm["_count"], "the parent's normaliser _count")
    if count.ndim != 0 or count.dtype.is_floating_point or count.dtype.is_complex or count.dtype == torch.bool \
            or int(count) < 2:
        raise RunRefused(f"the parent's observation normaliser has count {count.tolist()}; it holds no statistics to map")
    clip = _as_tensor(norm["_clip"], "the parent's normaliser _clip")
    if tuple(clip.shape) != (tmap.source_dim,):
        raise RunRefused(f"the parent's normaliser _clip has shape {tuple(clip.shape)}; the source has "
                         f"{tmap.source_dim} inputs")
    if not bool((clip == clip[0]).all()):
        raise RunRefused("the parent's normaliser clips its dimensions differently; the target-only clip is undefined")
    target_cols, source_cols = _column_index(tmap)
    fills = {"_mean": TARGET_ONLY_MEAN, "_var": TARGET_ONLY_VARIANCE, "_std": math.sqrt(TARGET_ONLY_VARIANCE),
             "_sumsq": TARGET_ONLY_VARIANCE * (int(count) - 1), "_clip": float(clip[0])}
    out = dict(norm)
    for key in NORMALISER_VECTORS:
        vector = _as_tensor(norm[key], f"the parent's normaliser {key}")
        if tuple(vector.shape) != (tmap.source_dim,):
            raise RunRefused(f"the parent's normaliser {key} has shape {tuple(vector.shape)}; the source has "
                             f"{tmap.source_dim} inputs")
        mapped = torch.full((tmap.target_dim,), float(fills[key]), dtype=vector.dtype)
        mapped[target_cols] = vector[source_cols]
        out[key] = mapped
    out["_count"] = count.clone()
    return out


def map_state_for_transfer(state: Mapping[str, Any], tmap: TransferMap) -> dict[str, Any]:
    """The parent's full state mapped onto the target task's observation layout (Q-transfer-obs).

    The first-layer weight of the actor (``mean.0.weight``, or ``mean.trunk.0.weight`` of an injected
    actor) and of each critic (``critic_0.0.weight``) becomes ``(hidden, target_dim)``: the columns
    of shared components are the source columns of the same name, the columns of target-only
    components are ``TARGET_ONLY_WEIGHT`` (zero), source-only columns are dropped. Every other
    tensor of those networks is unchanged. The observation normaliser (``_mean``, ``_sumsq``,
    ``_var``, ``_std``, ``_clip``) becomes target-sized: shared dimensions copied; target-only
    dimensions ``_mean`` ``TARGET_ONLY_MEAN`` (0), ``_var`` ``TARGET_ONLY_VARIANCE`` (1) and ``_std``
    its square root, ``_sumsq = TARGET_ONLY_VARIANCE · (_count - 1)`` (so the next push,
    ``omnisafe/common/normalizer.py:129-139``, keeps their variance near
    ``TARGET_ONLY_VARIANCE``) and the parent's clip;
    ``_count`` is the parent's. At the start of transfer the networks' outputs therefore depend only
    on the shared components, with the parent's weights.

    Entries other than ``pi``, the critics and ``obs_normalizer`` (optimiser states, scheduler,
    multiplier, windows) are passed through unchanged: ``pilot.dependencies.restore_learner`` never
    loads them. The input is not modified; nothing random is drawn. RunRefused for a state that does
    not come from the source task's networks.
    """
    if not isinstance(state, Mapping) or "pi" not in state or "reward_critic" not in state:
        raise RunRefused("a full-state checkpoint with 'pi' and 'reward_critic' is required")
    out = dict(state)
    out["pi"] = _map_network(state["pi"], ACTOR_FIRST_WEIGHTS, tmap, "the parent's actor")
    for critic in ("reward_critic", "cost_critic"):
        if critic in state:
            out[critic] = _map_network(state[critic], (CRITIC_FIRST_WEIGHT,), tmap, f"the parent's {critic}")
    if "obs_normalizer" in state:
        out["obs_normalizer"] = _map_normaliser(state["obs_normalizer"], tmap)
    return out


def _map_state_of_run(state: Mapping[str, Any], tmap: TransferMap, run_id: str) -> dict[str, Any]:
    """``map_state_for_transfer`` for the run ``run_id``: its refusals name the run."""
    try:
        return map_state_for_transfer(state, tmap)
    except RunRefused as exc:
        raise _refuse(run_id, str(exc)) from exc


# ---------------------------------------------------------------------------
# The algorithms
# ---------------------------------------------------------------------------


def _same_number(value: Any, expected: float) -> bool:
    """``value`` is a finite real number (not a bool) equal to ``expected``."""
    return (not isinstance(value, bool) and isinstance(value, numbers.Real) and math.isfinite(float(value))
            and float(value) == float(expected))


def is_registered_spec(spec: RunSpec) -> bool:
    """False for a smoke copy (``SMOKE-`` run_id, scripts/smoke_run.py), whose run length is shrunk."""
    return not spec.run_id.startswith(SMOKE_PREFIX)


def validate_continuation_spec(spec: Any, condition: str, *, env_id: str | None = None) -> RunSpec:
    """``spec`` as a battery continuation of ``condition`` (``pilot.manifest.battery_continuation``).

    RunRefused unless: the spec is Study A, group and plug-in ``battery_<condition>``, base algorithm
    ``CONTINUATION_BASE_ALGO`` (PPO-Lagrangian, a PID-arm parent's continuation included; Table 9.1,
    Q-continuations), without an onset; it depends on exactly its parent ``params['parent_run_id']`` and
    names ``parent_step`` (a whole, non-negative step), ``parent_checkpoint`` and ``parent_commit``
    (non-empty strings, which ``pilot.dependencies.resolve`` checks against the parent), the
    ``condition`` and the parent's task ``source_task`` (a Study A task); ``params['cost_limit']`` is
    ``R.COST_LIMIT`` (Table 2.2 "Transfer": "under the same budget d"); fine-tuning trains on the source
    task with ``params['multiplier'] == 'frozen_zero'`` (Table 2.2: "the multiplier frozen at zero");
    transfer trains on the source task's held-out task (``pilot.manifest.transfer_task``: Table 2.2's
    Point pair, or Table 9.1's for Car) with ``params['fresh_multiplier_init'] ==
    R.LAGRANGE_MULTIPLIER_INIT``; ``params`` has no ``lr_schedule_steps`` (the actor schedule spans the
    continuation's own registered length; only a few-shot continuation shortened by Part 6.1 cut 3
    carries that key, Table 9.1, Q-continuations); ``env_id`` (when given) is the spec's task; and,
    unless the spec is a smoke copy, the total is ``R.FINETUNE_STEPS`` or ``R.TRANSFER_STEPS`` (Table
    2.2; Table 8.2).
    """
    spec = _as_spec(spec)
    what = spec.run_id
    if condition not in CONDITIONS:
        raise _refuse(what, f"condition {condition!r} is not one of {CONDITIONS}")
    plugin = f"battery_{condition}"
    if (spec.study, spec.plugin, spec.group) != ("A", plugin, plugin):
        raise _refuse(what, f"study {spec.study!r}, plugin {spec.plugin!r}, group {spec.group!r}: not a {plugin} continuation")
    if spec.base_algo != CONTINUATION_BASE_ALGO:
        raise _refuse(what, f"base_algo {spec.base_algo!r}; every battery continuation trains with "
                            f"{CONTINUATION_BASE_ALGO} (Table 9.1, Q-continuations)")
    if spec.onset_step not in (None, 0):
        raise _refuse(what, f"a continuation has no onset, got onset_step {spec.onset_step!r}")
    params = dict(spec.params)
    parent = params.get("parent_run_id")
    if not isinstance(parent, str) or spec.depends_on != (parent,):
        raise _refuse(what, f"depends on {spec.depends_on}; a continuation depends on exactly its parent {parent!r}")
    step = params.get("parent_step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise _refuse(what, f"parent_step {step!r} is not a whole, non-negative step")
    for key in ("parent_checkpoint", "parent_commit"):
        if not isinstance(params.get(key), str) or not params[key]:
            raise _refuse(what, f"params {key} {params.get(key)!r} is not a non-empty string (pilot/manifest.py)")
    if params.get("condition") != condition:
        raise _refuse(what, f"params condition {params.get('condition')!r} is not {condition!r}")
    source = params.get("source_task")
    if source not in R.TASKS_STUDY_A:
        raise _refuse(what, f"source_task {source!r} is not a Study A task {R.TASKS_STUDY_A}")
    if not _same_number(params.get("cost_limit"), R.COST_LIMIT):
        raise _refuse(what, f"params cost_limit {params.get('cost_limit')!r} is not the registered d = {R.COST_LIMIT}")
    if "lr_schedule_steps" in params:
        raise _refuse(what, "a battery continuation's actor schedule spans its own registered length; only a few-shot "
                            "continuation shortened by Part 6.1 cut 3 has params lr_schedule_steps (Table 9.1, "
                            "Q-continuations)")
    if condition == "finetune":
        if spec.task != source:
            raise _refuse(what, f"fine-tuning trains on the training task {source}, not {spec.task} (Table 2.2)")
        if params.get("multiplier") != "frozen_zero":
            raise _refuse(what, f"params multiplier {params.get('multiplier')!r}; Table 2.2 freezes it at zero")
    else:
        try:
            held_out = transfer_task(source)
        except ValueError as exc:
            raise _refuse(what, str(exc)) from exc
        if spec.task != held_out:
            raise _refuse(what, f"transfer from {source} trains on its held-out task {held_out}, not {spec.task}")
        if not _same_number(params.get("fresh_multiplier_init"), R.LAGRANGE_MULTIPLIER_INIT):
            raise _refuse(what, f"fresh_multiplier_init {params.get('fresh_multiplier_init')!r} is not "
                                f"{R.LAGRANGE_MULTIPLIER_INIT} (Table 9.1, Q-continuations)")
    if env_id is not None and env_id != spec.task:
        raise _refuse(what, f"env_id {env_id!r} is not the spec's task {spec.task!r}")
    if is_registered_spec(spec) and spec.total_steps != CONDITION_STEPS[condition]:
        raise _refuse(what, f"total_steps {spec.total_steps} is not the registered {CONDITION_STEPS[condition]} (Table 2.2)")
    return spec


def _check_config(cfgs: Any, spec: RunSpec) -> None:
    """The configuration is this continuation's: RunRefused otherwise (see ``make_finetune_algorithm``)."""
    what = spec.run_id
    if not isinstance(cfgs, Config):
        raise _refuse(what, "the configuration must be OmniSafe's Config")
    if "continuation_cfgs" in cfgs:
        raise _refuse(what, "the configuration already has continuation_cfgs; the factory adds them once")
    try:
        seed = _whole_number(_config_value(cfgs, "seed"))
        steps_per_epoch = _whole_number(_config_value(cfgs, "algo_cfgs", "steps_per_epoch"))
        total = _whole_number(_config_value(cfgs, "train_cfgs", "total_steps"))
        epochs = _whole_number(_config_value(cfgs, "train_cfgs", "epochs"))
        use_cost = _config_value(cfgs, "algo_cfgs", "use_cost")
        penalty = float(_config_value(cfgs, "algo_cfgs", "penalty_coef"))
        actor_type = _config_value(cfgs, "model_cfgs", "actor_type")
        lagrange = {k: _config_value(cfgs, "lagrange_cfgs", k)
                    for k in ("cost_limit", "lagrangian_multiplier_init", "lambda_lr")}
    except (TypeError, ValueError, OverflowError) as exc:
        raise _refuse(what, f"the configuration is malformed: {exc}") from exc
    except RunRefused as exc:
        raise _refuse(what, str(exc)) from exc
    if seed != spec.seed:
        raise _refuse(what, f"cfgs.seed {seed} is not the spec's seed {spec.seed} (the parent's seed, "
                            "pilot.manifest.battery_continuation)")
    if use_cost is not True:
        raise _refuse(what, "algo_cfgs.use_cost must stay True: the cost critic keeps training")
    if penalty != 0.0:
        raise _refuse(what, f"algo_cfgs.penalty_coef is {penalty}; the reward advantage must not contain a cost term")
    if actor_type != ACTOR_TYPE:
        raise _refuse(what, f"model_cfgs.actor_type {actor_type!r}; the registered actor is {ACTOR_TYPE!r}")
    registered_lagrange = {"cost_limit": R.COST_LIMIT, "lagrangian_multiplier_init": R.LAGRANGE_MULTIPLIER_INIT,
                           "lambda_lr": R.LAGRANGE_MULTIPLIER_LR}
    wrong = {k: v for k, v in lagrange.items() if not _same_number(v, registered_lagrange[k])}
    if wrong:
        raise _refuse(what, f"lagrange_cfgs {wrong} differ from Table A.1 {registered_lagrange}")
    if is_registered_spec(spec) and (steps_per_epoch, total, epochs * steps_per_epoch) != (
            R.STEPS_PER_EPOCH, spec.total_steps, spec.total_steps):
        raise _refuse(what, f"the configuration trains {epochs} x {steps_per_epoch} of {total} steps; a registered continuation "
                            f"trains {spec.total_steps} in epochs of {R.STEPS_PER_EPOCH} (Table 3.1)")
    pilot_cfgs = cfgs.get("pilot_cfgs")
    parent = spec.params["parent_run_id"]
    records = pilot_cfgs.get("dependencies") if isinstance(pilot_cfgs, Mapping) else None
    record = records.get(parent) if isinstance(records, Mapping) else None
    if not isinstance(record, Mapping) or record.get("checkpoint_step") != spec.params["parent_step"]:
        raise _refuse(what, f"the launcher resolved no checkpoint of {parent} at step {spec.params['parent_step']} "
                            "(cfgs.pilot_cfgs.dependencies; pilot/dependencies.py)")
    if pilot_cfgs.get("plasticity"):
        raise _refuse(what, "a continuation logs no plasticity metrics (pilot.contracts.plasticity_required)")


def _check_parent_spec(cfgs: Any, spec: RunSpec) -> None:
    """RunRefused unless the parent's spec.json is a Study A training run of the main study (Part 3.6:
    "The pilot's runs are not reused"; ``pilot.manifest.STUDY_A_PARENT_GROUPS``) on the spec's source
    task, with the spec's seed."""
    what = spec.run_id
    parent = spec.params["parent_run_id"]
    dep = dependencies.from_cfgs(cfgs).get(parent)
    if dep is None:
        raise _refuse(what, f"the launcher resolved no dependency {parent!r}")
    try:
        parent_spec = RunSpec.from_json((dep.run_dir / rundir.SPEC_FILE).read_text(encoding="utf-8"))
    except (AttributeError, OSError, TypeError, ValueError) as exc:
        raise _refuse(what, f"the spec of the parent {parent} cannot be read: {exc}") from exc
    if (parent_spec.run_id, parent_spec.study, parent_spec.task, parent_spec.seed) != (
            parent, "A", spec.params["source_task"], spec.seed):
        raise _refuse(what, f"the parent {parent} is not the Study A run of {spec.params['source_task']} with seed {spec.seed}")
    if parent_spec.pilot or parent_spec.group not in STUDY_A_PARENT_GROUPS:
        raise _refuse(what, f"the parent {parent} (group {parent_spec.group!r}) is not a Study A run of the main study "
                            f"{STUDY_A_PARENT_GROUPS}")


class ContinuationMixin:
    """Restore the parent's learner in ``_init`` and record it in ``continuation.json`` (pilot.dependencies.restore_learner).

    Use as ``class FinetunePPOLag(ContinuationMixin, FullStateCheckpointMixin, PPOLag)``, built with
    the spec: ``FinetunePPOLag(env_id=..., cfgs=..., spec=spec)``. ``cfgs['continuation_cfgs']``
    (set by the factory, hashed) holds the condition, the parent and, for transfer, the transfer map.
    ``pilot.dependencies.restore_learner`` runs after OmniSafe's ``_init_env``, ``_init_model`` and
    the base ``_init`` (``omnisafe/algorithms/base_algo.py:48-53``) and before ``_init_log``, so the
    mixin's saver holds the restored objects and ``epoch-0.pt`` holds the restored learner. Nothing
    here draws a random number; the continuation is seeded by ``cfgs.seed``, the parent's seed.
    """

    _CONDITION = ""

    def __init__(self, env_id: str, cfgs: Any, *, spec: RunSpec) -> None:
        self._continuation_spec = validate_continuation_spec(spec, self._CONDITION, env_id=env_id)
        self._continuation_summary: dict[str, Any] | None = None
        super().__init__(env_id=env_id, cfgs=cfgs)  # type: ignore[call-arg]

    def _continuation_settings(self) -> dict[str, Any]:
        """The factory's ``continuation_cfgs``, checked against the spec; RunRefused otherwise."""
        spec = self._continuation_spec
        raw = self._cfgs.get("continuation_cfgs")  # type: ignore[attr-defined]
        if not isinstance(raw, Mapping):
            raise _refuse(spec.run_id, "the configuration has no continuation_cfgs: build the algorithm with its factory")
        settings = _plain(raw)
        expected = {"condition": self._CONDITION, "parent_run_id": spec.params["parent_run_id"],
                    "parent_step": spec.params["parent_step"], "source_task": spec.params["source_task"],
                    "target_task": spec.task}
        wrong = {k: settings.get(k) for k, v in expected.items() if settings.get(k) != v}
        if wrong:
            raise _refuse(spec.run_id, f"continuation_cfgs {wrong} do not describe this spec")
        if not isinstance(settings.get("multiplier"), Mapping):
            raise _refuse(spec.run_id, f"continuation_cfgs multiplier {settings.get('multiplier')!r} is not the "
                                       "factory's multiplier rule")
        if "transfer_map" not in settings or (settings["transfer_map"] is None) != (self._CONDITION == "finetune"):
            raise _refuse(spec.run_id, "continuation_cfgs must hold a transfer_map for transfer, and None for fine-tuning")
        return settings

    def _init(self) -> None:
        """OmniSafe's ``_init``, then the condition's multiplier and the parent's learner, restored."""
        super()._init()  # type: ignore[misc]  # PPOLag._init: buffer and a fresh multiplier from lagrange_cfgs
        spec = self._continuation_spec
        settings = self._continuation_settings()
        self._prepare_multiplier()
        obs_map = None
        if settings["transfer_map"] is not None:
            try:
                tmap = TransferMap.from_dict(settings["transfer_map"])
            except RunRefused as exc:
                raise _refuse(spec.run_id, str(exc)) from exc
            obs_dim = int(self._env.observation_space.shape[0])  # type: ignore[attr-defined]
            if (tmap.source_task, tmap.target_task, tmap.target_dim) != (spec.params["source_task"], spec.task, obs_dim):
                raise _refuse(spec.run_id, f"the transfer map {tmap.source_task} -> {tmap.target_task} "
                                           f"({tmap.target_dim} inputs) does not fit this run's {spec.task} "
                                           f"({obs_dim} inputs)")
            obs_map = functools.partial(_map_state_of_run, tmap=tmap, run_id=spec.run_id)
        parent = dependencies.from_cfgs(self._cfgs).get(spec.params["parent_run_id"])  # type: ignore[attr-defined]
        if parent is None:
            raise _refuse(spec.run_id, f"the launcher resolved no parent {spec.params['parent_run_id']!r}")
        state = dependencies.load_full_state(parent.checkpoint(int(spec.params["parent_step"])))
        summary = dependencies.restore_learner(self, state, spec=spec, obs_map=obs_map)
        self._continuation_summary = {
            **summary,
            "run_id": spec.run_id,
            "condition": self._CONDITION,
            "source_task": spec.params["source_task"],
            "target_task": spec.task,
            "multiplier": settings["multiplier"],
            "transfer_map": settings["transfer_map"],
            "owner": OWNER,
        }

    def _prepare_multiplier(self) -> None:
        """The condition's multiplier, set before the saver captures it (default: PPOLag's fresh one)."""

    def _init_log(self) -> None:
        super()._init_log()  # type: ignore[misc]  # FullStateCheckpointMixin: saver, epoch-0.pt (restored learner)
        if self._continuation_summary is None:
            raise _refuse(self._continuation_spec.run_id, "the continuation's learner was not restored")
        self._logger.register_key(ACTIVE_KEY)  # type: ignore[attr-defined]  # stored once per epoch by _store_active
        write_json_once(Path(self._logger.log_dir) / CONTINUATION_FILE, self._continuation_summary)  # type: ignore[attr-defined]

    def _store_active(self) -> None:
        """``Onset/Active`` of this epoch: ``CONDITION_ACTIVE`` of the condition (as envs.onset logs it)."""
        self._logger.store({ACTIVE_KEY: CONDITION_ACTIVE[self._CONDITION]})  # type: ignore[attr-defined]


class FinetunePPOLag(ContinuationMixin, FullStateCheckpointMixin, PPOLag):
    """Reward-only fine-tuning (Table 2.2): the actor on A_R alone, the multiplier held at 0.

    "with the cost term removed from the objective and the multiplier frozen at zero": the surrogate
    is ``adv_r`` itself (not ``(adv_r - 0 · adv_c) / 1``, which an infinite cost advantage would turn
    into NaN), the multiplier update is never called and J_C is never read, and the multiplier (0.0)
    is stored as ``Metrics/LagrangeMultiplier`` in every epoch (contract 3; the launcher checks the
    column), with ``Onset/Active`` 0. Both critics keep training (``use_cost`` is never touched).
    """

    _CONDITION = "finetune"

    def _prepare_multiplier(self) -> None:
        self._lagrange.lagrangian_multiplier.data.fill_(0.0)

    def _update(self) -> None:
        if float(self._lagrange.lagrangian_multiplier.detach()) != 0.0:
            raise _refuse(self._continuation_spec.run_id, "the fine-tuning multiplier must stay at zero (Table 2.2)")
        super(PPOLag, self)._update()  # PolicyGradient._update: both critics, the actor on A_R; no multiplier update
        self._logger.store({rundir.MULTIPLIER_COLUMN: self._lagrange.lagrangian_multiplier})
        self._store_active()

    def _compute_adv_surrogate(self, adv_r: torch.Tensor, adv_c: torch.Tensor) -> torch.Tensor:
        return adv_r


class TransferPPOLag(ContinuationMixin, FullStateCheckpointMixin, PPOLag):
    """Transfer (Table 2.2): OmniSafe's PPO-Lagrangian, unmodified, on the held-out task under d = 25.

    The multiplier is the fresh one ``PPOLag._init`` builds from ``lagrange_cfgs`` (0.001, learning
    rate 0.035, cost limit 25; Table A.1), with a fresh Adam state (Table 9.1, Q-continuations, which
    borrows Table 2.5's few-shot rule "with a fresh multiplier at 0.001"). The parent's learner
    is mapped onto the held-out task's observation layout before it is restored (Q-transfer-obs).
    Every epoch is constrained, so ``Onset/Active`` is 1 in every epoch.
    """

    _CONDITION = "transfer"

    def _update(self) -> None:
        super()._update()  # PPOLag._update, unmodified: the multiplier from J_C first, then the critics and the actor
        self._store_active()


# ---------------------------------------------------------------------------
# The factories (pilot.manifest.PLUGINS)
# ---------------------------------------------------------------------------


def _make(env_id: str, cfgs: Any, spec: Any, condition: str) -> Any:
    """The ``condition`` continuation of ``spec``, checked and built (``make_finetune_algorithm``)."""
    spec = validate_continuation_spec(spec, condition, env_id=env_id)
    _check_config(cfgs, spec)
    _check_parent_spec(cfgs, spec)
    finetune = condition == "finetune"
    tmap = None if finetune else transfer_map(spec.params["source_task"], spec.task)
    settings = {
        "owner": OWNER,
        "condition": condition,
        "parent_run_id": spec.params["parent_run_id"],
        "parent_step": int(spec.params["parent_step"]),
        "source_task": spec.params["source_task"],
        "target_task": spec.task,
        "restore": "pilot.dependencies.restore_learner (Table 9.1, Q-continuations)",
        "multiplier": (
            {"rule": "frozen at zero; the actor on the reward advantage only (Table 2.2)", "value": 0.0}
            if finetune else
            {"rule": "fresh (Table 9.1, Q-continuations)", "init": float(R.LAGRANGE_MULTIPLIER_INIT),
             "lr": float(R.LAGRANGE_MULTIPLIER_LR), "cost_limit": float(R.COST_LIMIT)}
        ),
        "transfer_map": None if tmap is None else tmap.to_dict(),
        "registered": is_registered_spec(spec),
    }
    cfgs["continuation_cfgs"] = Config.dict2config(settings)  # before construction: in config.json and the hash
    cls = FinetunePPOLag if finetune else TransferPPOLag
    return cls(env_id=env_id, cfgs=cfgs, spec=spec)


def make_finetune_algorithm(env_id: str, cfgs: Any, spec: Any) -> FinetunePPOLag:
    """``PLUGINS['battery_finetune']``: reward-only fine-tuning of the parent's matched checkpoint.

    Validates the spec (``validate_continuation_spec``) and the configuration (OmniSafe's ``Config``
    without ``continuation_cfgs`` yet, the parent's seed, ``use_cost`` True, no reward penalty, the
    registered actor, the registered multiplier settings, the launcher's record of the parent
    checkpoint, no plasticity logging, and for a registered spec the registered run length, in whole
    numbers), checks the parent's spec.json, stores ``cfgs['continuation_cfgs']`` and builds
    ``FinetunePPOLag``, which restores the parent (Q-continuations). RunRefused for every problem.
    """
    return _make(env_id, cfgs, spec, "finetune")


def make_transfer_algorithm(env_id: str, cfgs: Any, spec: Any) -> TransferPPOLag:
    """``PLUGINS['battery_transfer']``: transfer of the parent's matched checkpoint to the held-out task.

    As ``make_finetune_algorithm``, plus the observation mapping ``transfer_map(source_task,
    spec.task)`` (Q-transfer-obs), stored in ``cfgs['continuation_cfgs']['transfer_map']`` and
    applied by ``map_state_for_transfer`` when ``TransferPPOLag`` restores the parent.
    """
    return _make(env_id, cfgs, spec, "transfer")

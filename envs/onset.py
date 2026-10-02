"""The Study A training plug-ins: constraint onset, ramp, controller variants and treatments.

Owner: Environment and tests (Role 2; Muhammad Talha Jamil). ``pilot.manifest.PLUGINS`` names the
two factories: ``study_a -> make_algorithm`` (PPO-Lagrangian: the main sweep, the mediation
treatments, the controller variants and the pilot's Study A runs) and ``study_a_pid ->
make_pid_algorithm`` (the PID check). The launcher calls ``factory(spec.task, cfgs, spec)``.

What the pre-registration fixes (Table 2.1, PDF p. 7; equations 3 to 5, Part 2.5, p. 10):

* Table 2.1 "Constraint onset": "The training step at which the cost term enters the surrogate
  objective (equation 3) and the multiplier begins to update (equation 4). Before onset the cost
  advantage is not computed into the loss and the multiplier is held at its initial value."
* Table 2.1 "Abrupt onset": "At onset the budget d = 25 applies at once, with the multiplier at its
  initial value 0.001 (OmniSafe default)."
* Table 2.1 "Linear ramp": "From onset the effective budget in the multiplier update falls linearly
  from d_loose to d over a window W = 0.10 T (decision), then stays at d (equation 5)."
* Table 3.1 runs "PPO-Lagrangian as implemented in OmniSafe", so from onset the update is
  OmniSafe's own ``PPOLag._update`` (``ppo_lag.py:73-80``: the multiplier from J_C first, then the
  critics and the actor with the new multiplier) and its surrogate ``(adv_r - penalty * adv_c) / (1
  + penalty)`` (``ppo_lag.py:101-102``), unmodified.

How this module realises it (the plug-in changes the training math only through
``_compute_adv_surrogate`` and ``_update``, plus the rate-limited multiplier object, the one-off
intervention at onset and, for the data control only, the actor's learning-rate schedule
(``DataControlLR``; Table 9.1, Q-data-control-lr), never ``use_cost``, as HANDOVER.md section 8
requires):

* Epoch e (OmniSafe's 0-based loop index, ``logger.current_epoch`` during its rollout and update)
  is constrained iff ``e >= onset_step / E``. A total-steps-matched late arm then trains exactly
  the steps [N·T, T) under the constraint (Table 2.4 "Total-steps matched": "a late arm trains
  (1 − N)·T steps under the constraint"), an N = 0 run is OmniSafe's PPO-Lagrangian bit for bit,
  and ``epoch-{k0}.pt`` (the checkpoint "at onset", Part 3.4) is the last unconstrained state. This
  reading of which update is the first constrained one is answered in Table 9.1 (Q-onset-epoch).
* Before onset: the actor's surrogate is ``adv_r`` itself (not ``(adv_r - 0.001 adv_c) / 1.001``);
  the multiplier update is never called, so the multiplier keeps its stored initial value (float32
  0.001 for PPO-Lagrangian, 0.0 for OmniSafe's PID) and its Adam state stays empty; J_C is not
  read; ``Metrics/LagrangeMultiplier`` is still stored once every epoch (Part 3.4: "The multiplier,
  or in Study B every per-level multiplier, is logged at every epoch"; a key not stored in an epoch
  becomes NaN and the launcher would exclude the run as ``non_finite_multiplier``). Both critics
  train from the first epoch (Table 9.1, Q-cost-critic).
* Ramp (equation 5, reconstructed from the PDF glyphs: d_eff(t) = d_loose − (d_loose − d)(t − N·T)/W
  for N·T <= t < N·T + W, then d): the multiplier update of epoch e uses ``Lagrange.cost_limit =
  effective_budget(e · E)``, with N·T the run's onset step and W = R.RAMP_WINDOW_STEPS (Table 9.1,
  Q-ramp-step). Only the multiplier update changes; the surrogate has no budget.
* Warm-started multiplier (Table 2.4: "At onset the multiplier is set to the value the N = 0 arm of
  the same task and seed had reached at the same step, instead of 0.001"): before the onset
  epoch's multiplier update, the multiplier is set to the N = 0 run's logged value after
  ``onset_step`` steps (``pilot.dependencies.Dependency.multiplier_at``), with the fresh (never
  stepped) Adam state (Table 9.1, Q-warm-start). The factory reads it, so it is in config.json and
  in the configuration hash.
* Rate-limited multiplier (Table 2.4: "Each multiplier update is clipped so that the multiplier
  changes by at most 5 percent of its current value plus 0.01 per epoch"): ``RateLimitedLagrange``
  clips after each Adam step to ``old ± (R.RATE_LIMIT_RELATIVE · |old| + R.RATE_LIMIT_ABSOLUTE)``,
  then to [0, the multiplier's upper bound]; Adam's moments are untouched (Table 9.1, Q-rate-limit).
* PID check (Table 2.4 "PID multiplier": "OmniSafe's PIDLagrangian update (equation 9), using the
  default gains of OmniSafe's CPPOPID configuration for the pinned version"): ``pid_update`` is
  skipped before onset and the surrogate is ``adv_r``; from onset OmniSafe's CPPOPID runs
  unmodified (Table 9.1, Q-pid-eq9).
* Partial reset and plasticity injection (Table 2.4; equation 8): applied exactly once, at the start
  of the rollout of the onset epoch, after ``epoch-{k0}.pt`` and its plasticity row are written, so
  the onset checkpoint and the onset metrics are pre-intervention (HANDOVER.md section 8). This
  module decides WHEN; ``metrics.interventions`` (Role 3) decides WHAT. Injection keeps the first
  hidden layer and ``log_std`` training (Table 9.1, Q-reset-injection), which departs from the
  literal reading of Table 2.4 ("only the new copy is trained thereafter"); its source is Nikishin
  et al. (2023), arXiv:2305.15555, Section 3, per two independent search-engine extracts (paper PDF
  not opened from the sandbox; see ``metrics/interventions.py``). The record goes to
  ``intervention.json`` in the OmniSafe run directory.
* Additional constrained training (Table 2.4: "The late arm continues under the constraint for N·T
  further steps after T, with no change to the network or optimiser"): the abrupt schedule over
  the spec's T + N·T steps, as one run (the manifest's total) whose first T steps are, bit for bit,
  the untreated late arm of the same task and seed (Table 9.1, Q-data-control-lr). OmniSafe would
  decay the actor's learning rate over the run's own epochs, which changes it from the first epoch;
  ``DataControlLR`` replaces that schedule before ``_init_log`` (the key
  ``onset_cfgs['actor_lr_schedule']``, which only the data control carries, set by the function
  ``actor_lr_schedule``): OmniSafe's own LinearLR over the first T/E epochs, then a restart at the
  rate at which the arm entered its constrained phase, (1 − N)·lr, decaying linearly to 0 over the
  N·T/E further epochs.

Every setting the factory derives goes into ``cfgs['onset_cfgs']`` before the algorithm is built,
so it is in config.json and in the configuration hash (``pilot/launch.py``). Diagnostics are logged
under ``Onset/`` only: the launcher checks every ``Metrics/LagrangeMultiplier*`` and ``Loss/``
column for finiteness, and ``Onset/EffectiveBudget`` and ``Onset/MultiplierProposed`` are NaN in
unconstrained epochs. Every configuration or specification problem raises
``pilot.errors.RunRefused`` (the launcher exits 5 and the run stays pending), never another
exception, which the launcher would classify as a crash (Part 5.6). A missing Role 3 module or
name raises ``ImportError`` from inside the factory, which the launcher reports as unavailable
(exit 4).

The full-state checkpoints, the extra checkpoints at onset and at onset + 200,000 steps
(``pilot_cfgs.extra_checkpoint_steps``) and the plasticity hook (``pilot_cfgs.plasticity``,
pilot/contracts.py contract 2) come from ``pilot.algorithms.FullStateCheckpointMixin``, which each
class keeps in its method resolution order; this module installs none of them again.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import os
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from omnisafe.algorithms.on_policy.naive_lagrange.ppo_lag import PPOLag
from omnisafe.algorithms.on_policy.pid_lagrange.cppo_pid import CPPOPID
from omnisafe.common.lagrange import Lagrange
from omnisafe.utils.config import Config
from torch.optim.lr_scheduler import LinearLR, LRScheduler

from configs import registered as R
from pilot import dependencies, rundir
from pilot.algorithms import FullStateCheckpointMixin
from pilot.errors import RunRefused
from pilot.manifest import PLUGINS, RunSpec, onset_schedule, short_task

OWNER = PLUGINS["study_a"][1]  # "Environment and tests (Role 2)", as pilot.manifest.PLUGINS names it
SMOKE_PREFIX = "SMOKE-"  # scripts/smoke_run.py ``short``: shrunk totals and onsets, never registered data
# pilot.scheduler.determinism_spec names its copy of an arm "DET-" + the arm's run_id and keeps the arm's
# group (pilot/scheduler.py); only a spec equal to that function's output is accepted as the check (is_determinism_spec).
DETERMINISM_PREFIX = "DET-"
STUDY_A_GROUPS = ("main", "treatment", "controller", "pid", "pilot")
# Table 3.3's PID row ("PID multiplier, abrupt arms, SafetyPointGoal1-v0 only, N = 0.10, 0.25, 0.50") names
# no step-matching control; the manifest's PID arms are total-steps matched (pilot/manifest.py
# ``study_a_pid``), the choice answered in Table 9.1 (Q-pid-eq9).
PID_STEP_MATCHING = "total_steps"  # answered in Table 9.1 (Q-pid-eq9)
CONTROL_SHORT = {"total_steps": "total", "constrained_steps": "constrained"}  # arm names (pilot/manifest.py)
# The spec parameters the manifest writes for Study A arms (pilot/manifest.py ``_study_a_spec`` and ``pilot``).
PARAM_KEYS = frozenset({"cost_limit", "d_loose", "ramp_window_steps", "rate_limit_relative", "rate_limit_absolute",
                        "additional_constrained_steps", "pilot_revision"})
ACTOR_TYPE = "gaussian_learning"  # the pinned configuration's actor (PPOLag.yaml; metrics.interventions)
INTERVENTION_FUNCTIONS = {"reset": "partial_reset", "injection": "plasticity_injection"}  # metrics.interventions (HANDOVER.md section 8)
INTERVENTION_FILE = "intervention.json"
# The data control's actor learning-rate schedule in onset_cfgs (Table 9.1, Q-data-control-lr; ``actor_lr_schedule``,
# ``DataControlLR``): present on that arm only, so no other arm's config.json or configuration hash changes.
SCHEDULE_KEY = "actor_lr_schedule"
SCHEDULE_FIELDS = ("decay_epochs", "tail_epochs", "tail_start_factor")
MULTIPLIER_KEY = rundir.MULTIPLIER_COLUMN  # "Metrics/LagrangeMultiplier" (contract 3)
# Diagnostics, one value per epoch. NaN is allowed here and only here (see the module docstring).
ACTIVE_KEY = "Onset/Active"  # 1.0 in constrained epochs, 0.0 before onset
BUDGET_KEY = "Onset/EffectiveBudget"  # the cost limit of the epoch's multiplier update; NaN before onset
# The multiplier when the epoch's _update starts (after a warm start; the held value before onset).
START_KEY = "Onset/MultiplierBeforeUpdate"
PROPOSED_KEY = "Onset/MultiplierProposed"  # OmniSafe's update before the rate clip (else the new value); NaN before onset
ONSET_KEYS = (ACTIVE_KEY, BUDGET_KEY, START_KEY, PROPOSED_KEY)


def _refuse(what: str, message: str) -> RunRefused:
    return RunRefused(f"{what}: {message}")


def _as_spec(spec: Any) -> RunSpec:
    """The launcher passes a ``RunSpec``; a mapping (spec.json) is converted, and a bad one refused."""
    if isinstance(spec, RunSpec):
        return spec
    if isinstance(spec, Mapping):
        try:
            return RunSpec.from_dict(spec)
        except (AttributeError, TypeError, ValueError) as exc:  # AttributeError: e.g. a run_id that is not a string
            raise RunRefused(f"not a valid run spec: {exc}") from exc
    raise RunRefused(f"a run spec is required, got {type(spec).__name__}")


def _whole(value: Any, what: str, name: str) -> int:
    """A spec value or argument that must be an integer (not a bool, not a float); RunRefused otherwise."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise _refuse(what, f"{name} must be a whole number, got {value!r}")
    return int(value)


def _whole_number(value: Any) -> int:
    """A configuration value as a whole number (an integer, or a finite float with no fraction; not a bool).

    ValueError otherwise, never a truncation: ``int(0.5)`` would pass a fractional seed as seed 0.
    """
    if isinstance(value, bool) or not (isinstance(value, numbers.Integral) or (
            isinstance(value, float) and math.isfinite(value) and value.is_integer())):
        raise ValueError(f"{value!r} is not a whole number")
    return int(value)


def _number(value: Any, what: str, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real) or not math.isfinite(float(value)):
        raise _refuse(what, f"{name} must be a finite number, got {value!r}")
    return float(value)


def is_determinism_spec(spec: Any) -> bool:
    """True iff ``spec`` is the determinism check's run of its plug-in (pilot/scheduler.py).

    That is ``pilot.scheduler.determinism_spec(spec.plugin, spec.total_steps)`` exactly, apart from
    its open-question list (as the scheduler's ``determinism_report_plugin`` compares a report's
    spec): the arm's copy with a ``DET-`` run_id, the check's total, the arm's group, and a late
    onset scaled into the check. ``scripts/determinism_check.py --plugin study_a|study_a_pid`` trains
    it through the launcher (Table 3.1 "Determinism check"). The scheduler is imported here, lazily:
    training never needs it otherwise.
    """
    spec = _as_spec(spec)
    if not spec.run_id.startswith(DETERMINISM_PREFIX):
        return False
    # the scheduler's own comparison rule (_spec_without_gates), so the two cannot diverge
    from pilot.scheduler import DETERMINISM_CHECK_PLUGINS, _spec_without_gates, determinism_spec

    if spec.plugin not in DETERMINISM_CHECK_PLUGINS:
        return False
    try:
        expected = determinism_spec(spec.plugin, spec.total_steps)
    except ValueError:  # a total the check never uses (not a positive whole number of registered epochs)
        return False
    return _spec_without_gates(spec.to_dict()) == _spec_without_gates(expected.to_dict())


def is_registered_spec(spec: Any) -> bool:
    """True unless the spec is a smoke copy (``SMOKE-`` run_id) or the determinism check's run.

    A registered spec must match the registered design exactly (``build_onset_cfgs`` with
    ``registered=True``); a smoke copy has shrunk totals and onsets (scripts/smoke_run.py), and the
    determinism check (Table 3.1 "Determinism check") trains the copy of one arm that
    ``pilot.scheduler.determinism_spec`` makes, with the check's total and onset
    (``is_determinism_spec``).
    """
    spec = _as_spec(spec)
    return not spec.run_id.startswith(SMOKE_PREFIX) and not is_determinism_spec(spec)


# ---------------------------------------------------------------------------
# Pure pieces: equation (5), the rate limit, the arm's settings
# ---------------------------------------------------------------------------


def effective_budget(step: int, *, onset_step: int, cost_limit: float, d_loose: float | None = None,
                     window_steps: int | None = None) -> float:
    """The budget of the multiplier update at ``step`` (equation 5; Table 2.1 "Linear ramp").

    d_eff(t) = d_loose − (d_loose − d)(t − N·T)/W for N·T <= t < N·T + W, then d (the PDF's
    equation 5 as reconstructed from its glyphs). ``d_loose`` None is the abrupt onset (Table 2.1
    "Abrupt onset": "At onset the budget d = 25 applies at once"): ``cost_limit`` at every step. The
    plug-in evaluates it at ``t = e · E``, the first step of the epoch whose data the update uses
    (Table 9.1, Q-ramp-step), so the onset epoch gets d_loose and every epoch from onset + W gets d.
    ValueError for a step before onset or an invalid window (the plug-in validates both earlier).
    """
    if step < onset_step:
        raise ValueError(f"step {step} is before the onset step {onset_step}: no multiplier update happens there")
    if d_loose is None:
        if window_steps is not None:
            raise ValueError("an abrupt onset has no ramp window")
        return float(cost_limit)
    if window_steps is None or window_steps <= 0:
        raise ValueError(f"a ramp needs a positive window, got {window_steps!r}")
    if step >= onset_step + window_steps:
        return float(cost_limit)
    return float(d_loose) - (float(d_loose) - float(cost_limit)) * (step - onset_step) / window_steps


def rate_limit(old: float, proposed: float, *, relative: float, absolute: float, upper: float | None = None) -> float:
    """The rate-limited multiplier (Table 2.4 "Rate-limited multiplier"; Table 9.1, Q-rate-limit).

    "clipped so that the multiplier changes by at most 5 percent of its current value plus 0.01 per
    epoch": ``proposed`` (OmniSafe's Adam step, already clamped to ``[0, upper]``) clipped to ``old ± b`` with
    ``b = relative · |old| + absolute`` and "its current value" read as the value before the update
    (Table 9.1), then clamped to ``[0, upper]`` (the lower end of the band is negative while
    ``old < absolute / (1 − relative)``).
    """
    bound = relative * abs(old) + absolute
    value = min(max(proposed, old - bound), old + bound)
    value = max(value, 0.0)
    if upper is not None:
        value = min(value, float(upper))
    return value


def _arm(N: float, shape: str | None, control: str | None, suffix: str | None) -> str:
    """The arm name the manifest gives these factors (``pilot.manifest._study_a_arm``), re-derived."""
    arm = "N0.00" if N == 0 else f"N{N:.2f}-{shape}-{CONTROL_SHORT[control]}"
    return f"{arm}-{suffix}" if suffix else arm


def _nt(N: float) -> int:
    """N·T in whole steps (exact: the manifest's own Fraction arithmetic)."""
    return int(Fraction(str(N)) * R.TOTAL_STEPS)


def actor_lr_schedule(epochs: int, N: float) -> dict[str, Any]:
    """The data control's actor learning-rate schedule over its run of ``epochs`` epochs (Q-data-control-lr).

    Table 2.4 "Additional constrained training": "The late arm continues under the constraint for N·T
    further steps after T, with no change to the network or optimiser". Table 9.1 (Q-data-control-lr):
    the run's first ``decay_epochs`` epochs follow OmniSafe's LinearLR from 1 to 0 over T/E epochs, the
    untreated late arm's own schedule; the ``tail_epochs`` = N·T/E further epochs restart at
    ``tail_start_factor`` = 1 − N times the actor's rate, the rate at which the arm entered its
    constrained phase, and decay linearly to 0 (``DataControlLR``). ``decay_epochs`` is the run's
    epochs times T/(T + N·T) = 1/(1 + N) to the nearest whole epoch (at least one): exactly
    ``R.EPOCHS`` (500), with 250, 125 or 50 tail epochs, for the registered totals T + N·T; a smoke
    copy keeps the proportion in its shrunk run (scripts/smoke_run.py ``short`` keeps the spec's N·T,
    which is longer than that run). ValueError unless ``epochs`` is a positive whole number and N a
    late onset fraction.
    """
    if isinstance(epochs, bool) or not isinstance(epochs, numbers.Integral) or epochs < 1:
        raise ValueError(f"the data control's run needs a positive whole number of epochs, got {epochs!r}")
    n = Fraction(str(N))
    if not 0 < n < 1:
        raise ValueError(f"the data control has a late onset fraction, got N = {N!r}")
    decay = max(1, math.floor(Fraction(int(epochs)) / (1 + n) + Fraction(1, 2)))  # no exact half at N = 0.10, 0.25, 0.50
    return {"decay_epochs": decay, "tail_epochs": int(epochs) - decay, "tail_start_factor": float(1 - n)}


def build_onset_cfgs(spec: Any, *, steps_per_epoch: int, registered: bool = True) -> dict[str, Any]:
    """The plug-in's settings for ``spec`` (``cfgs['onset_cfgs']``), validated against configs.registered.

    Pure (reads no file). Refuses (``RunRefused``) unless:

    * the spec is a Study A run of plug-in ``study_a`` on PPOLag or ``study_a_pid`` on CPPOPID, the
      PID one exactly when ``controller_variant`` is ``R.PID_VARIANT`` and only on ``R.PID_TASKS``
      (Table 3.3 "PID check"), on a task of ``R.TASKS_STUDY_A`` (Table 3.1);
    * N is one of ``R.ONSET_FRACTIONS`` (Table 2.1 "Onset fraction N"); N = 0 has no shape, control,
      treatment or controller and onset 0; a late arm has a shape of ``R.ONSET_SHAPES`` and a control
      of ``R.STEP_MATCHING_CONTROLS`` (Table 3.2);
    * the onset is a whole epoch of ``steps_per_epoch`` and, when after step 0, before the end of
      the run (at least one constrained epoch);
    * a treatment (``R.TREATMENTS``) or a controller variant is on an abrupt arm at a late N (Table
      3.3), never both; a treatment, the warm start and the rate limit on a total-steps-matched arm
      (Table 3.3: "applied at onset to the abrupt arm under the total-steps-matched control";
      "abrupt arms, total-steps matched"); the PID check on a ``PID_STEP_MATCHING`` arm, the
      manifest's choice answered in Table 9.1 (Q-pid-eq9), not Table 3.3's (its PID row names no
      control); reset, injection and the warm start need an onset after step 0 (they act "At
      onset", Table 2.4);
    * the parameters are the registered values: ``cost_limit == R.COST_LIMIT`` (Table 2.1 "Cost
      budget d"); a ramp has ``d_loose == R.D_LOOSE[task]`` and ``ramp_window_steps ==
      R.RAMP_WINDOW_STEPS`` (Table 2.1 "Linear ramp") and nothing else has them; a rate-limited arm
      has ``R.RATE_LIMIT_RELATIVE`` and ``R.RATE_LIMIT_ABSOLUTE`` (Table 2.4; Table 8.2) and nothing
      else has them; the data control has ``additional_constrained_steps`` = N·T and nothing else
      has it, and its run is a whole number of epochs; a warm-started arm has exactly one
      dependency and nothing else has one.

    ``registered=True`` (every spec of ``pilot.manifest.design('study_a')`` and the pilot's Study A
    runs) also requires the registered design: ``steps_per_epoch == R.STEPS_PER_EPOCH``; onset and
    total as ``pilot.manifest.onset_schedule(N, control)`` (Table 2.4; T + N·T for additional
    constrained training): registered values except for the constrained-steps arms at N = 0.10
    and 0.25, whose registered total T/(1 − N) is not a whole epoch: there the schedule's onset
    rounded to the nearest whole epoch is Table 9.1's (Q-rounding), and those specs carry that run
    gate; a late arm's onset after step 0; the group, run_id, arm name and warm-start dependency
    the manifest gives these factors; only the manifest's parameter keys.
    ``registered=False`` (smoke copies and the determinism check, ``is_registered_spec``) skips
    exactly these, so a smoke copy's shrunk run, onset 0 included, and the determinism check's
    copy of an arm (its total and scaled onset, ``pilot.scheduler.determinism_spec``) are accepted.

    The result holds the onset (step and epoch), the factors, the registered values in use, the
    intervention to apply (Role 3's function, depth ``R.INTERVENTION_LAYERS`` and generator),
    ``warm_start`` None (the factory fills it from the dependency) and, for the data control only,
    the key ``actor_lr_schedule`` (the function ``actor_lr_schedule`` of the run's epochs; Table 9.1,
    Q-data-control-lr): no other arm's settings, config.json or configuration hash carry that key.
    """
    spec = _as_spec(spec)
    what = spec.run_id
    if spec.study != "A":
        raise _refuse(what, f"study {spec.study!r} is not Study A")
    if spec.task not in R.TASKS_STUDY_A:
        raise _refuse(what, f"{spec.task} is not a Study A task {R.TASKS_STUDY_A} (Table 3.1)")
    pid = spec.controller_variant == R.PID_VARIANT
    expected_algo = ("CPPOPID", "study_a_pid") if pid else ("PPOLag", "study_a")
    if (spec.base_algo, spec.plugin) != expected_algo:
        raise _refuse(what, f"base_algo {spec.base_algo!r} with plugin {spec.plugin!r}; this arm needs {expected_algo}")
    if pid and spec.task not in R.PID_TASKS:
        raise _refuse(what, f"the PID check runs on {R.PID_TASKS} only (Table 3.3), not {spec.task}")
    if spec.group not in STUDY_A_GROUPS:  # a smoke copy and the determinism check keep their arm's group
        raise _refuse(what, f"group {spec.group!r} is not a Study A training group {STUDY_A_GROUPS}")
    E = _whole(steps_per_epoch, what, "steps_per_epoch")
    if E <= 0:
        raise _refuse(what, f"steps_per_epoch must be positive, got {E}")
    if registered and E != R.STEPS_PER_EPOCH:
        raise _refuse(what, f"a registered run has {R.STEPS_PER_EPOCH} steps per epoch (Table 3.1), not {E}")

    N = spec.N
    if isinstance(N, bool) or not isinstance(N, numbers.Real) or float(N) not in R.ONSET_FRACTIONS:
        raise _refuse(what, f"N = {N!r} is not a registered onset fraction {R.ONSET_FRACTIONS} (Table 2.1)")
    N = float(N)
    late = N > 0
    shape, control = spec.onset_shape, spec.step_matching
    treatment, controller = spec.treatment, spec.controller_variant
    total = _whole(spec.total_steps, what, "total_steps")
    onset = 0 if spec.onset_step is None and not late else spec.onset_step
    onset = _whole(onset, what, "onset_step")
    if not late:
        if (shape, control, treatment, controller) != (None, None, None, None) or onset != 0:
            raise _refuse(what, "the N = 0 arm has no onset shape, control, treatment or controller and onset 0 (Part 3.2)")
    else:
        if shape not in R.ONSET_SHAPES:
            raise _refuse(what, f"onset shape {shape!r} is not one of {R.ONSET_SHAPES} (Table 3.2)")
        if control not in R.STEP_MATCHING_CONTROLS:
            raise _refuse(what, f"step-matching control {control!r} is not one of {R.STEP_MATCHING_CONTROLS} (Table 3.2)")
    if treatment is not None and treatment not in R.TREATMENTS:
        raise _refuse(what, f"treatment {treatment!r} is not one of {R.TREATMENTS} (Table 3.3)")
    if controller is not None and controller not in (*R.CONTROLLER_VARIANTS, R.PID_VARIANT):
        raise _refuse(what, f"controller variant {controller!r} is not one of {(*R.CONTROLLER_VARIANTS, R.PID_VARIANT)} "
                                f"(Tables 2.4 and 3.3)")
    if treatment is not None and controller is not None:
        raise _refuse(what, "an arm has a treatment or a controller variant, never both (Table 3.3)")
    if treatment is not None or controller is not None:
        if N not in R.LATE_ONSET_FRACTIONS or shape != "abrupt":
            raise _refuse(what, f"treatments and controller variants apply to abrupt arms at N in "
                                f"{R.LATE_ONSET_FRACTIONS} (Table 3.3)")
        if controller == R.PID_VARIANT:
            if control != PID_STEP_MATCHING:
                raise _refuse(what, f"the PID check is {PID_STEP_MATCHING}-matched: Table 3.3's PID row names no "
                                    "control; the manifest's choice is answered in Table 9.1 (Q-pid-eq9)")
        elif control != "total_steps":
            raise _refuse(what, "treatments and the warm-started and rate-limited multipliers apply under the "
                                "total-steps-matched control (Table 3.3)")
    if onset < 0 or onset % E:
        raise _refuse(what, f"onset_step {onset} is not a whole epoch of {E} steps")
    if onset > 0 and onset >= total:
        raise _refuse(what, f"onset_step {onset} leaves no constrained epoch in a run of {total} steps")
    if (treatment in INTERVENTION_FUNCTIONS or controller == "warm_started") and onset == 0:
        raise _refuse(what, f"{treatment or controller} acts at onset (Table 2.4), which must come after step 0")

    params = dict(spec.params)
    cost_limit = params.get("cost_limit")
    if isinstance(cost_limit, bool) or cost_limit != R.COST_LIMIT:
        raise _refuse(what, f"params cost_limit {cost_limit!r} is not the registered d = {R.COST_LIMIT} (Table 2.1)")
    ramp = shape == "ramp"
    d_loose = params.get("d_loose")
    window = params.get("ramp_window_steps")
    if ramp:
        if isinstance(d_loose, bool) or d_loose != R.D_LOOSE[spec.task]:
            raise _refuse(what, f"d_loose {d_loose!r} is not the registered {R.D_LOOSE[spec.task]} of {spec.task} (Table 2.1)")
        if isinstance(window, bool) or window != R.RAMP_WINDOW_STEPS:
            raise _refuse(what, f"ramp_window_steps {window!r} is not the registered W = {R.RAMP_WINDOW_STEPS} (Table 2.1)")
    elif "d_loose" in params or "ramp_window_steps" in params:
        raise _refuse(what, f"an arm with onset shape {shape!r} has no ramp parameters")
    rate_limited = controller == "rate_limited"
    if rate_limited:
        for key, value in (("rate_limit_relative", R.RATE_LIMIT_RELATIVE), ("rate_limit_absolute", R.RATE_LIMIT_ABSOLUTE)):
            if isinstance(params.get(key), bool) or params.get(key) != value:
                raise _refuse(what, f"{key} {params.get(key)!r} is not the registered {value} (Table 2.4; Table 8.2)")
    elif "rate_limit_relative" in params or "rate_limit_absolute" in params:
        raise _refuse(what, "only the rate-limited arm has rate limits")
    data_control = treatment == "additional_constrained"
    extra_steps = params.get("additional_constrained_steps")
    if data_control:
        if isinstance(extra_steps, bool) or extra_steps != _nt(N):
            raise _refuse(what, f"additional_constrained_steps {extra_steps!r} is not N·T = {_nt(N)} (Table 2.4)")
        if total % E:
            raise _refuse(what, f"total_steps {total} is not a whole number of {E}-step epochs: the data control's "
                                "learning-rate schedule counts epochs (Q-data-control-lr)")
    elif "additional_constrained_steps" in params:
        raise _refuse(what, "only additional constrained training has additional_constrained_steps")
    warm = controller == "warm_started"
    if warm and len(spec.depends_on) != 1:
        raise _refuse(what, f"a warm-started run depends on exactly its N = 0 run, got {spec.depends_on}")
    if not warm and spec.depends_on:
        raise _refuse(what, f"only a warm-started run has a dependency, got {spec.depends_on}")

    if registered:
        _check_registered_design(spec, N=N, onset=onset, total=total, params=params)

    settings: dict[str, Any] = {
        "owner": OWNER,
        "onset_step": onset,
        "onset_epoch": onset // E,
        "steps_per_epoch": E,
        "N": N,
        "shape": shape,
        "step_matching": control,
        "treatment": treatment,
        "controller_variant": controller,
        "multiplier_update": "CPPOPID" if pid else "PPOLag",
        "cost_limit": float(R.COST_LIMIT),
        "d_loose": float(d_loose) if ramp else None,
        "ramp_window_steps": int(window) if ramp else None,
        "rate_limit_relative": float(R.RATE_LIMIT_RELATIVE) if rate_limited else None,
        "rate_limit_absolute": float(R.RATE_LIMIT_ABSOLUTE) if rate_limited else None,
        "additional_constrained_steps": int(extra_steps) if data_control else None,
        "intervention": {
            "module": "metrics.interventions",
            "function": INTERVENTION_FUNCTIONS[treatment],
            "depth": int(R.INTERVENTION_LAYERS),
            "generator": "metrics.interventions.intervention_generator(seed)",
            "output_check_batch": "metrics.batches.load_fixed_batch(task), normalised frozen by the run's observation "
                                  "normaliser at onset (metrics.plasticity.normalise_frozen)",
            "when": "start of the rollout of the onset epoch, after its checkpoint and plasticity row",
        } if treatment in INTERVENTION_FUNCTIONS else None,
        "warm_start": None,
        "registered": bool(registered),
    }
    if data_control:
        settings[SCHEDULE_KEY] = actor_lr_schedule(total // E, N)
    return settings


def _check_registered_design(spec: RunSpec, *, N: float, onset: int, total: int, params: Mapping[str, Any]) -> None:
    """The parts of ``build_onset_cfgs`` that tie a spec to the registered design (``registered=True``)."""
    what = spec.run_id
    late = N > 0
    schedule_pending: tuple[str, ...] = ()
    if late:
        expected_onset, expected_total, schedule_pending = onset_schedule(N, spec.step_matching)
    else:
        expected_onset, expected_total = 0, R.TOTAL_STEPS
    if spec.treatment == "additional_constrained":
        expected_total = R.TOTAL_STEPS + _nt(N)  # Table 2.4: N·T further steps after T
    if (onset, total) != (expected_onset, expected_total):
        # Table 2.4 registers a constrained-steps arm's total as T/(1 − N), which is not a whole epoch at N = 0.10
        # and 0.25: onset_schedule's whole-epoch values there are Table 9.1's (Q-rounding), not Table 2.4's numbers.
        if "Q-rounding" in schedule_pending:
            source = ("the values of pilot.manifest.onset_schedule (Table 2.4, rounded to whole epochs: Table 9.1, "
                      "Q-rounding)")
        elif late:
            source = "the registered values (Table 2.4)"
        else:
            source = "the registered values (Table 3.1 \"Total steps T\")"
        raise _refuse(what, f"onset {onset} and total {total} are not {expected_onset} and {expected_total}, "
                            f"{source}, for N = {N}, {spec.step_matching}")
    if late and onset <= 0:
        raise _refuse(what, "a registered late arm has its onset after step 0")
    unknown = sorted(set(params) - PARAM_KEYS)
    if unknown:
        raise _refuse(what, f"unknown spec parameters {unknown}")
    treatment, controller = spec.treatment, spec.controller_variant
    if spec.pilot != (spec.group == "pilot"):
        raise _refuse(what, f"pilot {spec.pilot} does not match group {spec.group!r}")
    if spec.group == "pilot":
        if (treatment, controller) != (None, None) or spec.task != R.PILOT_STUDY_A_TASK \
                or N not in R.PILOT_STUDY_A_ONSET_FRACTIONS \
                or (late and (spec.onset_shape, spec.step_matching) != (R.PILOT_STUDY_A_SHAPE, R.PILOT_STUDY_A_CONTROL)):
            raise _refuse(what, f"the pilot's Study A arms are N in {R.PILOT_STUDY_A_ONSET_FRACTIONS}, a late one "
                                f"{R.PILOT_STUDY_A_SHAPE} under the {R.PILOT_STUDY_A_CONTROL} control, on "
                                f"{R.PILOT_STUDY_A_TASK} (Part 3.6)")
    if treatment is not None:
        expected_group = "treatment"
    elif controller == R.PID_VARIANT:
        expected_group = "pid"
    elif controller is not None:
        expected_group = "controller"
    else:
        expected_group = "main"
    if spec.group != "pilot" and spec.group != expected_group:
        raise _refuse(what, f"group {spec.group!r}; this arm belongs to {expected_group!r} (Tables 3.2 and 3.3)")
    arm = _arm(N, spec.onset_shape, spec.step_matching, treatment or controller)
    revision = params.get("pilot_revision")
    if revision is not None and not spec.pilot:
        raise _refuse(what, "only a pilot run has a pilot revision")
    prefix = "" if not spec.pilot else ("P-" if revision is None else f"P{revision}-")
    expected_id = f"{prefix}A-{short_task(spec.task)}-{arm}-s{spec.seed}"
    if spec.arm != arm or spec.run_id != expected_id:
        raise _refuse(what, f"arm {spec.arm!r} and run_id do not name these factors (expected {arm!r}, {expected_id!r})")
    if controller == "warm_started":
        source = f"A-{short_task(spec.task)}-N0.00-s{spec.seed}"
        if spec.depends_on != (source,):
            raise _refuse(what, f"a warm-started run depends on {source}, the N = 0 run of the same task and seed "
                                f"(Table 2.4), not {spec.depends_on}")


# ---------------------------------------------------------------------------
# Warm start: the N = 0 run's multiplier at the onset step (Table 2.4; Q-warm-start)
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def warm_start_source(cfgs: Any, spec: Any, onset_step: int) -> dict[str, Any]:
    """The multiplier the N = 0 run of the same task and seed had reached after ``onset_step`` steps.

    Table 2.4 "Warm-started multiplier": "At onset the multiplier is set to the value the N = 0 arm of
    the same task and seed had reached at the same step, instead of 0.001". Table 9.1 (Q-warm-start):
    the multiplier only (a fresh Adam state), read from the N = 0 run's progress.csv
    (``pilot.dependencies.from_cfgs(cfgs)[dep].multiplier_at(onset_step)``: the row of 0-based epoch
    ``onset_step // E - 1``, which OmniSafe stores after that epoch's update, ``ppo_lag.py:73-80``);
    the N = 0 main runs do not have a checkpoint at every onset step (epoch 125 is off their grid), so
    progress.csv is the one source for every N, and the checkpoint ``epoch-{onset_step // E}.pt``
    cross-checks it when it exists.

    RunRefused unless: the spec has exactly one dependency, resolved by the launcher
    (``cfgs.pilot_cfgs.dependencies``); its spec.json is the N = 0 PPO-Lagrangian Study A run of
    the same task, seed and pilot flag; the value is finite, at least 0 and a float32 number (as
    OmniSafe logs a multiplier); and it equals the checkpoint's multiplier when that checkpoint
    exists. Returns ``{value, source_run_id, source_step, source_progress_row, source_progress_sha256,
    source_commit, source_config_hash, checked_against_checkpoint}``, which the factory puts in
    ``onset_cfgs.warm_start`` (hashed; no absolute path).
    """
    spec = _as_spec(spec)
    what = spec.run_id
    if len(spec.depends_on) != 1:
        raise _refuse(what, f"a warm-started run depends on exactly one N = 0 run, got {spec.depends_on}")
    dep_id = spec.depends_on[0]
    deps = dependencies.from_cfgs(cfgs)
    if dep_id not in deps:
        raise _refuse(what, f"the launcher resolved no dependency {dep_id!r} (cfgs.pilot_cfgs.dependencies)")
    dep = deps[dep_id]
    try:
        source = RunSpec.from_json((dep.run_dir / rundir.SPEC_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise _refuse(what, f"the spec of {dep_id} cannot be read: {exc}") from exc
    if (source.run_id, source.study, source.plugin, source.base_algo) != (dep_id, "A", "study_a", "PPOLag") \
            or source.N != 0.0 or (source.onset_step or 0) != 0 or source.task != spec.task \
            or source.seed != spec.seed or source.pilot != spec.pilot:
        raise _refuse(what, f"{dep_id} is not the N = 0 PPO-Lagrangian run of {spec.task}, seed {spec.seed} (Table 2.4)")
    onset_step = _whole(onset_step, what, "onset_step")
    if onset_step <= 0:
        raise _refuse(what, "the warm start acts at an onset after step 0")
    value = dep.multiplier_at(onset_step)  # RunRefused unless one finite value is logged there
    if value < 0.0:
        raise _refuse(what, f"{dep_id} logged a negative multiplier {value} at step {onset_step}")
    if float(np.float32(value)) != value:
        raise _refuse(what, f"{dep_id} logged {value!r} at step {onset_step}, which is not a float32 multiplier")
    epoch = onset_step // dep.steps_per_epoch()
    checkpoint = rundir.checkpoint_file(dep.omnisafe_dir, epoch)
    checked = checkpoint.is_file()
    if checked:
        state = dependencies.load_full_state(checkpoint)
        lagrange = state.get("lagrange")
        try:
            saved = float(lagrange["value"])  # type: ignore[index]
        except (KeyError, TypeError, ValueError) as exc:
            raise _refuse(what, f"{checkpoint} holds no multiplier value: {exc}") from exc
        if saved != value:
            raise _refuse(what, f"{dep_id}: progress.csv gives {value!r} at step {onset_step}, its checkpoint {saved!r}")
    try:
        progress_sha256 = _sha256(dep.omnisafe_dir / rundir.PROGRESS_FILE)
    except OSError as exc:
        raise _refuse(what, f"the progress.csv of {dep_id} cannot be read: {exc}") from exc
    return {
        "value": value,
        "source_run_id": dep_id,
        "source_step": onset_step,
        "source_progress_row": epoch - 1,  # 0-based Train/Epoch
        "source_progress_sha256": progress_sha256,
        "source_commit": dep.commit_hash,
        "source_config_hash": dep.config_hash,
        "checked_against_checkpoint": checked,
    }


# ---------------------------------------------------------------------------
# The data control's learning-rate schedule (Table 2.4; Q-data-control-lr)
# ---------------------------------------------------------------------------


class DataControlLR(LRScheduler):
    """The actor's learning-rate schedule of additional constrained training (Table 9.1, Q-data-control-lr).

    ``OnsetMixin._init`` puts it in place of OmniSafe's ``actor_scheduler`` (a LinearLR over the run's
    own epochs, ``actor_critic.py:99-113``) before ``_init_log``, so OmniSafe's loop (``step()`` after
    each epoch's update, ``get_last_lr()`` for ``Train/LR``; ``policy_gradient.py:265-278``) and the
    full-state saver (pilot/algorithms.py) use this object. The rate of epoch k (after k steps):

    * k < ``decay_epochs``: the rate of a torch ``LinearLR(start_factor=1.0, end_factor=0.0,
      total_iters=decay_epochs)`` on the same optimiser, which this object steps: bit for bit the
      untreated late arm's rates (LinearLR updates the rate recursively, so its floats differ from
      any closed form in most epochs). Built after OmniSafe's own LinearLR, which already set each
      param group's rate to its ``initial_lr`` times 1.0, it starts from that same rate.
    * k >= ``decay_epochs``: ``initial_lr · tail_start_factor · (1 − (k − decay_epochs) / tail_epochs)``
      per param group (``initial_lr`` is ``model_cfgs.actor.lr``), computed exactly with
      ``fractions.Fraction`` and rounded once to float; 0 from k = decay_epochs + tail_epochs on.

    ``state_dict`` holds plain numbers only, the inner LinearLR's state nested, so every full-state
    checkpoint loads with ``torch.load(weights_only=True)`` (contract 3). Draws no random number.
    ValueError for a schedule that is not ``actor_lr_schedule``'s shape.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, *, decay_epochs: int, tail_epochs: int,
                 tail_start_factor: float) -> None:
        for name, value, least in (("decay_epochs", decay_epochs, 1), ("tail_epochs", tail_epochs, 0)):
            if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < least:
                raise ValueError(f"{name} must be a whole number of at least {least}, got {value!r}")
        if isinstance(tail_start_factor, bool) or not isinstance(tail_start_factor, numbers.Real) \
                or not 0 < float(tail_start_factor) <= 1:
            raise ValueError(f"tail_start_factor must lie in (0, 1], got {tail_start_factor!r}")
        self.decay_epochs = int(decay_epochs)
        self.tail_epochs = int(tail_epochs)
        self.tail_start_factor = float(tail_start_factor)
        self._decay = LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=self.decay_epochs)
        super().__init__(optimizer)  # base_lrs from each group's initial_lr; the initial step (k = 0) changes no rate

    def tail_lr(self, initial_lr: float, k: int) -> float:
        """The rate after ``k >= decay_epochs`` steps of a param group starting at ``initial_lr``."""
        done = k - self.decay_epochs
        if done >= self.tail_epochs:
            return 0.0
        factor = Fraction(str(self.tail_start_factor)) * (1 - Fraction(done, self.tail_epochs))
        return float(Fraction(initial_lr) * factor)

    def step(self, epoch: int | None = None) -> None:
        """Advance one epoch (OmniSafe steps the schedule after each epoch's update)."""
        if epoch is not None:
            raise ValueError("DataControlLR advances one epoch per step; it takes no epoch argument")
        self._step_count += 1
        self.last_epoch += 1
        k = self.last_epoch
        if 0 < k < self.decay_epochs:
            self._decay.step()
        elif k >= self.decay_epochs:
            for group, initial_lr in zip(self.optimizer.param_groups, self.base_lrs):
                group["lr"] = self.tail_lr(float(initial_lr), k)
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self) -> dict[str, Any]:
        """The schedule, its position and the inner LinearLR's state, as plain numbers."""
        return {
            "decay_epochs": self.decay_epochs,
            "tail_epochs": self.tail_epochs,
            "tail_start_factor": self.tail_start_factor,
            "base_lrs": [float(lr) for lr in self.base_lrs],
            "last_epoch": int(self.last_epoch),
            "_step_count": int(self._step_count),
            "_last_lr": [float(lr) for lr in self._last_lr],
            "decay": self._decay.state_dict(),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Resume from ``state_dict``; ValueError if it is another schedule's."""
        saved = tuple(state_dict.get(name) for name in SCHEDULE_FIELDS)
        if saved != (self.decay_epochs, self.tail_epochs, self.tail_start_factor):
            raise ValueError(f"the saved schedule {dict(zip(SCHEDULE_FIELDS, saved))} is not this one "
                             f"{dict(zip(SCHEDULE_FIELDS, (self.decay_epochs, self.tail_epochs, self.tail_start_factor)))}")
        self.base_lrs = [float(lr) for lr in state_dict["base_lrs"]]
        self.last_epoch = int(state_dict["last_epoch"])
        self._step_count = int(state_dict["_step_count"])
        self._last_lr = [float(lr) for lr in state_dict["_last_lr"]]
        self._decay.load_state_dict(dict(state_dict["decay"]))


# ---------------------------------------------------------------------------
# The rate-limited multiplier (Table 2.4; Q-rate-limit)
# ---------------------------------------------------------------------------


class RateLimitedLagrange(Lagrange):
    """OmniSafe's ``Lagrange`` whose every update is clipped by ``rate_limit`` after the Adam step.

    ``update_lagrange_multiplier`` runs OmniSafe's own update (``common/lagrange.py:129-136``: Adam
    on ``-lambda (J_C - d)``, then clamped to ``[0, lagrangian_upper_bound]``), records the value it proposed in
    ``last_proposed`` (logged as ``Onset/MultiplierProposed``), and writes the clipped value into
    the same Parameter. Adam's moments are not changed by the clip (Table 9.1, Q-rate-limit). The
    numeric attributes (the two limits, ``last_proposed``) go into every checkpoint's
    ``lagrange['numeric_attributes']`` through the mixin.
    """

    def __init__(self, *, rate_limit_relative: float, rate_limit_absolute: float, **lagrange_cfgs: Any) -> None:
        super().__init__(**lagrange_cfgs)
        self.rate_limit_relative = float(rate_limit_relative)
        self.rate_limit_absolute = float(rate_limit_absolute)
        self.last_proposed: float | None = None

    def update_lagrange_multiplier(self, Jc: float) -> None:
        old = float(self.lagrangian_multiplier.detach())
        super().update_lagrange_multiplier(Jc)
        proposed = float(self.lagrangian_multiplier.detach())
        self.last_proposed = proposed
        limited = rate_limit(old, proposed, relative=self.rate_limit_relative, absolute=self.rate_limit_absolute,
                             upper=self.lagrangian_upper_bound)
        self.lagrangian_multiplier.data.fill_(limited)


# ---------------------------------------------------------------------------
# The algorithms
# ---------------------------------------------------------------------------


def _plain(value: Any) -> Any:
    """An OmniSafe ``Config`` (a dict subclass) as plain dicts."""
    if hasattr(value, "todict"):
        return value.todict()
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    return value


def _config_value(cfgs: Any, *path: str) -> Any:
    node = cfgs
    for key in path:
        if not isinstance(node, Mapping) or key not in node:
            raise RunRefused(f"the configuration has no {'.'.join(path)}")
        node = node[key]
    return node


ONSET_CFG_KEYS = ("onset_step", "onset_epoch", "steps_per_epoch", "treatment", "controller_variant",
                  "multiplier_update", "cost_limit", "d_loose", "ramp_window_steps", "rate_limit_relative",
                  "rate_limit_absolute", "intervention", "warm_start")


def check_training_config(cfgs: Any, *, pid: bool) -> dict[str, Any]:
    """``cfgs['onset_cfgs']`` as a plain dict, checked against the configuration the algorithm runs.

    RunRefused unless: ``onset_cfgs`` is present with every key of ``ONSET_CFG_KEYS``; the onset is a whole epoch of
    ``algo_cfgs.steps_per_epoch`` equal to ``onset_cfgs.steps_per_epoch`` and, when after step 0,
    at or before the last epoch (at least one constrained epoch); ``pilot_cfgs.onset_step`` (the
    launcher's, used by the mixin's onset checkpoint), when present, is the same step, and is present
    for reset and injection (whose intervention requires that checkpoint to exist, so that the onset
    state is saved before it); the steps per epoch and the epochs are whole numbers (never truncated);
    ``algo_cfgs.use_cost`` is True (the cost critic trains in every epoch, Q-cost-critic; First
    Tasks warns against switching it off); ``algo_cfgs.penalty_coef`` is 0.0
    (otherwise OmniSafe's buffer subtracts ``penalty_coef · cost`` from the reward, so the pre-onset
    surrogate would not be A_R, ``onpolicy_buffer.py:185``); the actor is
    ``gaussian_learning``; the multiplier update is the class's (PID exactly for ``OnsetCPPOPID``);
    ``lagrange_cfgs.cost_limit`` is the onset settings' cost limit; and the arm's own settings are
    complete: a ramp has a budget of at least d and a positive window and is never on the PID check
    (``_update`` applies no ramp to OmniSafe's PID update; Table 3.3: "abrupt arms"), a
    rate-limited arm its two limits, reset and injection their Role 3 function and depth, a
    warm-started arm a non-negative value, and no other arm has any of them. The data control, and only it, has ``SCHEDULE_KEY``
    (``actor_lr_schedule``; Table 9.1, Q-data-control-lr): whole numbers of decay epochs (at least
    one) and tail epochs (at least zero) that sum to ``train_cfgs.epochs`` and a start factor in
    (0, 1], over OmniSafe's linear decay of a set actor rate (``model_cfgs.linear_lr_decay`` True and
    ``model_cfgs.actor.lr`` set), the schedule ``DataControlLR`` continues. Any other arm with the
    key, even as None, is refused: its configuration is that of every arm without it.
    """
    raw = cfgs.get("onset_cfgs") if isinstance(cfgs, Mapping) else None
    if not isinstance(raw, Mapping):
        raise RunRefused("the configuration has no onset_cfgs: build the algorithm with envs.onset.make_algorithm "
                         "or make_pid_algorithm")
    onset = _plain(raw)
    missing = [k for k in ONSET_CFG_KEYS if k not in onset]
    if missing:
        raise RunRefused(f"onset_cfgs lacks {missing}")
    try:
        E = _whole_number(_config_value(cfgs, "algo_cfgs", "steps_per_epoch"))
        epochs = _whole_number(_config_value(cfgs, "train_cfgs", "epochs"))
        use_cost = _config_value(cfgs, "algo_cfgs", "use_cost")
        penalty = float(_config_value(cfgs, "algo_cfgs", "penalty_coef"))
        actor_type = _config_value(cfgs, "model_cfgs", "actor_type")
        cost_limit = float(_config_value(cfgs, "lagrange_cfgs", "cost_limit"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise RunRefused(f"the configuration is malformed: {exc}") from exc
    step = onset["onset_step"]
    if isinstance(step, bool) or not isinstance(step, int) or step < 0 or E <= 0 or step % E:
        raise RunRefused(f"onset_cfgs.onset_step {step!r} is not a whole epoch of {E} steps")
    if onset["steps_per_epoch"] != E or onset["onset_epoch"] != step // E:
        raise RunRefused(f"onset_cfgs was built for {onset['steps_per_epoch']} steps per epoch and epoch "
                         f"{onset['onset_epoch']}; the run has {E} steps per epoch, onset epoch {step // E}")
    if step > 0 and step // E >= epochs:
        raise RunRefused(f"the onset epoch {step // E} leaves no constrained epoch in {epochs} epochs")
    pilot_cfgs = cfgs.get("pilot_cfgs") or {}
    launcher_onset = pilot_cfgs.get("onset_step") if isinstance(pilot_cfgs, Mapping) else None
    if launcher_onset is not None and (isinstance(launcher_onset, bool) or launcher_onset != step):
        raise RunRefused(f"pilot_cfgs.onset_step {launcher_onset!r} is not onset_cfgs.onset_step {step}")
    if onset["treatment"] in INTERVENTION_FUNCTIONS and launcher_onset is None:
        raise RunRefused(f"{onset['treatment']} needs pilot_cfgs.onset_step {step}: without it no onset checkpoint is saved")
    if use_cost is not True:
        raise RunRefused("algo_cfgs.use_cost must stay True: the cost critic trains from the first epoch (Q-cost-critic)")
    if penalty != 0.0:
        raise RunRefused(f"algo_cfgs.penalty_coef is {penalty}; with a reward penalty the surrogate before onset is not A_R")
    if actor_type != ACTOR_TYPE:
        raise RunRefused(f"model_cfgs.actor_type {actor_type!r}; the registered actor is {ACTOR_TYPE!r}")
    if (onset["multiplier_update"] == "CPPOPID") != pid or (onset["controller_variant"] == R.PID_VARIANT) != pid:
        raise RunRefused(f"onset_cfgs describes a {onset['multiplier_update']} multiplier; this algorithm's is "
                         f"{'CPPOPID' if pid else 'PPOLag'}")
    if cost_limit != onset["cost_limit"]:
        raise RunRefused(f"lagrange_cfgs.cost_limit {cost_limit} is not onset_cfgs.cost_limit {onset['cost_limit']}")
    d_loose, window = onset["d_loose"], onset["ramp_window_steps"]
    if (d_loose is None) != (window is None):
        raise RunRefused("onset_cfgs has a ramp budget without a window, or a window without a budget")
    if d_loose is not None:
        if pid:  # OnsetMixin._update gives OmniSafe's PID update no ramp: it would train abrupt
            raise RunRefused("the PID check has no ramp (Table 3.3: \"abrupt arms\"); onset_cfgs has a ramp budget")
        _number(d_loose, "onset_cfgs", "d_loose")
        if isinstance(window, bool) or not isinstance(window, int) or window <= 0 or d_loose < onset["cost_limit"]:
            raise RunRefused(f"onset_cfgs ramp from {d_loose!r} over {window!r} steps is not a ramp down to the budget")
    limited = onset["controller_variant"] == "rate_limited"
    for key in ("rate_limit_relative", "rate_limit_absolute"):
        if limited:
            if _number(onset[key], "onset_cfgs", key) < 0:
                raise RunRefused(f"onset_cfgs.{key} is negative")
        elif onset[key] is not None:
            raise RunRefused(f"only the rate-limited arm has onset_cfgs.{key}")
    treatment = onset["treatment"]
    intervention = onset["intervention"]
    if treatment in INTERVENTION_FUNCTIONS:
        if step == 0:
            raise RunRefused(f"{treatment} acts at an onset after step 0")
        depth = intervention.get("depth") if isinstance(intervention, Mapping) else None
        if not isinstance(intervention, Mapping) or intervention.get("function") != INTERVENTION_FUNCTIONS[treatment] \
                or isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
            raise RunRefused(f"onset_cfgs.intervention {intervention!r} is not the {treatment} of metrics.interventions")
    elif intervention is not None:
        raise RunRefused(f"only reset and injection have onset_cfgs.intervention, not {treatment!r}")
    if onset["controller_variant"] == "warm_started":
        warm = onset["warm_start"]
        if not isinstance(warm, Mapping) or "value" not in warm or step == 0:
            raise RunRefused("a warm-started run needs onset_cfgs.warm_start (envs.onset.warm_start_source) and a late onset")
        if _number(warm["value"], "onset_cfgs.warm_start", "value") < 0:
            raise RunRefused("the warm-start multiplier is negative")
    elif onset["warm_start"] is not None:
        raise RunRefused("only a warm-started run has onset_cfgs.warm_start")
    if treatment == "additional_constrained":
        _check_actor_lr_schedule(cfgs, onset.get(SCHEDULE_KEY), epochs)
    elif SCHEDULE_KEY in onset:
        raise RunRefused(f"only additional constrained training has onset_cfgs.{SCHEDULE_KEY} (Q-data-control-lr), "
                         f"not {treatment!r}")
    return onset


def _check_actor_lr_schedule(cfgs: Any, schedule: Any, epochs: int) -> None:
    """The data control's ``onset_cfgs.actor_lr_schedule`` against the run (``check_training_config``)."""
    if not isinstance(schedule, Mapping) or set(schedule) != set(SCHEDULE_FIELDS):
        raise RunRefused(f"the data control needs onset_cfgs.{SCHEDULE_KEY} with {list(SCHEDULE_FIELDS)} "
                         f"(envs.onset.actor_lr_schedule; Q-data-control-lr), got {schedule!r}")
    decay, tail, factor = (schedule[name] for name in SCHEDULE_FIELDS)
    for name, value, least in (("decay_epochs", decay, 1), ("tail_epochs", tail, 0)):
        if isinstance(value, bool) or not isinstance(value, int) or value < least:
            raise RunRefused(f"onset_cfgs.{SCHEDULE_KEY}.{name} {value!r} is not a whole number of at least {least}")
    if not 0 < _number(factor, f"onset_cfgs.{SCHEDULE_KEY}", "tail_start_factor") <= 1:
        raise RunRefused(f"onset_cfgs.{SCHEDULE_KEY}.tail_start_factor {factor!r} is not in (0, 1]")
    if decay + tail != epochs:
        raise RunRefused(f"onset_cfgs.{SCHEDULE_KEY} spans {decay} + {tail} epochs; the run has {epochs}")
    try:
        decays = _config_value(cfgs, "model_cfgs", "linear_lr_decay")
        actor_lr = _config_value(cfgs, "model_cfgs", "actor", "lr")
    except RunRefused as exc:
        raise RunRefused(f"the data control's schedule continues OmniSafe's actor schedule: {exc}") from exc
    if decays is not True or actor_lr is None:
        raise RunRefused(f"the data control's schedule continues OmniSafe's linear decay of the actor's rate "
                         f"(model_cfgs.linear_lr_decay True, Q-lr-decay), not linear_lr_decay {decays!r} with actor lr "
                         f"{actor_lr!r}")


def write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    """Write ``payload`` to ``path`` atomically; RunRefused if the file exists or the payload is not plain JSON.

    The file is published with ``os.link``, which never replaces an existing file, so a file that
    appears between the check and the write is refused too.
    """
    try:
        text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    except (TypeError, ValueError) as exc:
        raise RunRefused(f"{path.name} would not be plain JSON: {exc}") from exc
    if path.exists():
        raise RunRefused(f"{path} exists already; it is written once per run")
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, 0o644)
        try:
            os.link(tmp, path)  # exclusive: FileExistsError if the path exists
        except FileExistsError as exc:
            raise RunRefused(f"{path} exists already; it is written once per run") from exc
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class OnsetMixin:
    """Constraint onset for OmniSafe's PPO-Lagrangian and CPPOPID (see the module docstring).

    Use as ``class OnsetPPOLag(OnsetMixin, FullStateCheckpointMixin, PPOLag)``: ``_init_log`` then
    runs the mixin's (full-state saver, ``epoch-0.pt``, extra checkpoints, plasticity hook) before
    this class adds its keys and its rollout wrapper, and ``super()._update()`` is the base
    algorithm's. Reads everything from ``cfgs['onset_cfgs']`` (``build_onset_cfgs``), checked by
    ``check_training_config`` in ``_init`` (the factory already checked it before construction).
    Draws no random number: the interventions use Role 3's dedicated generator, so every other
    stream stays that of the untreated run of the same seed (HANDOVER.md section 10).
    """

    _LAGRANGIAN_BASE: type = PPOLag  # the class whose own _update holds the multiplier update

    # -- construction ------------------------------------------------------------------------

    def _init(self) -> None:
        super()._init()  # type: ignore[misc]  # buffer; the base algorithm's multiplier (self._lagrange)
        pid = issubclass(self._LAGRANGIAN_BASE, CPPOPID)
        self._onset = check_training_config(self._cfgs, pid=pid)  # type: ignore[attr-defined]
        self._onset_epoch = int(self._onset["onset_epoch"])
        # E, the steps of one whole epoch over every environment and process (the ramp's step count, e · E). Never
        # OmniSafe's own ``_steps_per_epoch``: that is the per-environment, per-process rollout length
        # (policy_gradient.py:73-77), which sizes the buffer (:121) and every rollout (:251), and equals E only
        # when vector_env_nums and world_size are 1.
        self._onset_steps_per_epoch = int(self._onset["steps_per_epoch"])
        self._constraint_active = False
        self._intervention_record: dict[str, Any] | None = None
        self._fixed_batch: Any = None
        self._intervention_api: dict[str, Any] = {}
        self._warm_start_applied = False
        if self._onset["controller_variant"] == "rate_limited":
            # Replaced here, before _init_log, so the mixin's saver holds this object (pilot/algorithms.py).
            self._lagrange = RateLimitedLagrange(
                rate_limit_relative=self._onset["rate_limit_relative"],
                rate_limit_absolute=self._onset["rate_limit_absolute"],
                **self._cfgs.lagrange_cfgs,  # type: ignore[attr-defined]
            )
        schedule = self._onset.get(SCHEDULE_KEY)
        if schedule is not None:
            # The data control (Q-data-control-lr): replaced before _init_log too, so OmniSafe's loop steps and logs
            # this object and the mixin's saver captures it (pilot/algorithms.py FullStateCheckpointMixin item 1).
            ac = self._actor_critic  # type: ignore[attr-defined]
            ac.actor_scheduler = DataControlLR(ac.actor_optimizer, **schedule)

    def _init_log(self) -> None:
        super()._init_log()  # type: ignore[misc]  # FullStateCheckpointMixin: saver, epoch-0.pt, extra saves, hook
        for key in ONSET_KEYS:
            self._logger.register_key(key)  # type: ignore[attr-defined]
        if self._onset["treatment"] in INTERVENTION_FUNCTIONS:
            self._prepare_intervention()

    # -- equation (4): the multiplier update --------------------------------------------------

    def _multiplier_value(self) -> float:
        value = self._lagrange.lagrangian_multiplier  # type: ignore[attr-defined]
        return float(value.detach()) if isinstance(value, torch.Tensor) else float(value)

    def _update(self) -> None:
        epoch = int(self._logger.current_epoch)  # type: ignore[attr-defined]  # the epoch whose rollout is in the buffer
        active = epoch >= self._onset_epoch
        self._constraint_active = active  # read by _compute_adv_surrogate in this epoch's actor update
        budget = proposed = math.nan
        if active:
            if epoch == self._onset_epoch and self._onset["warm_start"] is not None:
                self._apply_warm_start()
            lagrange = self._lagrange  # type: ignore[attr-defined]
            if self._onset["multiplier_update"] == "CPPOPID":
                budget = float(lagrange._cost_limit)  # OmniSafe's PID has no ramp (Table 3.3: abrupt arms)
            else:
                budget = effective_budget(
                    epoch * self._onset_steps_per_epoch, onset_step=int(self._onset["onset_step"]),
                    cost_limit=float(self._onset["cost_limit"]), d_loose=self._onset["d_loose"],
                    window_steps=self._onset["ramp_window_steps"],
                )
                lagrange.cost_limit = budget  # read by Lagrange.compute_lambda_loss (lagrange.py:112) at this update
            start = self._multiplier_value()
            super()._update()  # type: ignore[misc]  # PPOLag._update / CPPOPID._update, unmodified
            last = getattr(lagrange, "last_proposed", None)
            proposed = self._multiplier_value() if last is None else float(last)
        else:
            start = self._multiplier_value()
            # PolicyGradient._update: both critics, and the actor on A_R; no J_C, no multiplier update.
            super(self._LAGRANGIAN_BASE, self)._update()  # type: ignore[misc]
            self._logger.store({MULTIPLIER_KEY: self._lagrange.lagrangian_multiplier})  # type: ignore[attr-defined]
        self._logger.store({  # type: ignore[attr-defined]
            ACTIVE_KEY: float(active), BUDGET_KEY: budget, START_KEY: start, PROPOSED_KEY: proposed,
        })

    def _apply_warm_start(self) -> None:
        """Set the multiplier to the N = 0 run's value before the onset epoch's first update (Q-warm-start)."""
        if self._warm_start_applied:
            raise RunRefused("the warm start is applied once, before the onset epoch's multiplier update")
        lagrange = self._lagrange  # type: ignore[attr-defined]
        optimizer = getattr(lagrange, "lambda_optimizer", None)
        if optimizer is not None and len(optimizer.state):
            raise RunRefused("the multiplier's Adam state is not fresh at onset; the warm start copies the value only")
        value = float(self._onset["warm_start"]["value"])
        lagrange.lagrangian_multiplier.data.fill_(value)
        if self._multiplier_value() != value:
            raise RunRefused(f"the warm-start value {value!r} is not representable in the multiplier's float32")
        self._warm_start_applied = True
        self._logger.log(  # type: ignore[attr-defined]
            f"Onset: warm-started multiplier {value!r} from {self._onset['warm_start'].get('source_run_id')} "
            f"at step {self._onset['onset_step']} (Table 2.4; Q-warm-start)"
        )

    # -- equation (3): the surrogate ------------------------------------------------------------

    def _compute_adv_surrogate(self, adv_r: torch.Tensor, adv_c: torch.Tensor) -> torch.Tensor:
        """The actor's surrogate: ``adv_r`` before onset, the base algorithm's equation (3) from onset.

        Table 2.1: "Before onset the cost advantage is not computed into the loss".
        """
        if not self._constraint_active:
            return adv_r
        return super()._compute_adv_surrogate(adv_r, adv_c)  # type: ignore[misc]

    # -- the treatments (Table 2.4; HANDOVER.md section 8) --------------------------------------------------

    def _prepare_intervention(self) -> None:
        """Load the output-check batch and wrap the adapter's rollout for the intervention at onset.

        A missing batch refuses the run before training; the wrapped rollout applies the intervention
        at the start of the onset epoch's rollout.
        """
        # Role 3's code, imported here, inside the factory: a missing module or name is exit 4, not a crash.
        from metrics.batches import load_fixed_batch
        from metrics.interventions import InjectedHead, intervention_generator, partial_reset, plasticity_injection
        from metrics.plasticity import as_batch_tensor, normalise_frozen

        self._intervention_api = {
            "function": partial_reset if self._onset["treatment"] == "reset" else plasticity_injection,
            "generator": intervention_generator,
            "injected_head": InjectedHead,
            "as_batch_tensor": as_batch_tensor,
            "normalise_frozen": normalise_frozen,
        }
        # HANDOVER.md section 8: an actor that is injected already, or a record of an intervention, refuses the run
        # before anything trains (a fresh OmniSafe run directory never holds either).
        if isinstance(self._actor_critic.actor.mean, InjectedHead) \
                or (Path(self._logger.log_dir) / INTERVENTION_FILE).exists():  # type: ignore[attr-defined]
            raise RunRefused("this run already received an intervention; it is applied once per run (HANDOVER.md section 8)")
        task = str(self._env_id)  # type: ignore[attr-defined]
        batch = load_fixed_batch(task)  # FixedBatchError is a RunRefused
        obs_dim = int(self._env.observation_space.shape[0])  # type: ignore[attr-defined]
        if batch.ndim != 2 or batch.shape[1] != obs_dim:
            raise RunRefused(f"the fixed batch of {task} has shape {batch.shape}; the actor takes {obs_dim}")
        self._fixed_batch = batch
        adapter = self._env  # type: ignore[attr-defined]
        original_rollout = adapter.rollout

        def rollout(*args: Any, **kwargs: Any) -> Any:
            epoch = int(self._logger.current_epoch)  # type: ignore[attr-defined]
            if self._intervention_record is None:
                if epoch == self._onset_epoch:
                    self._apply_intervention()
                elif epoch > self._onset_epoch:
                    raise RunRefused(f"the rollout of epoch {epoch} started without the intervention of onset epoch "
                                     f"{self._onset_epoch}")
            return original_rollout(*args, **kwargs)

        adapter.rollout = rollout

    def _apply_intervention(self) -> dict[str, Any]:
        """Partial reset or plasticity injection, once (Table 2.4; ``metrics.interventions``, Role 3).

        Refused (RunRefused) if it was applied already (a record, an injected head, or
        ``intervention.json``), or if the onset checkpoint or its plasticity row is missing: the onset
        metrics must be pre-intervention (HANDOVER.md section 8). The output check uses the task's fixed batch as
        the actor sees it now (normalised by the run's normaliser, frozen), and the generator seeded
        from the run seed. The returned record, with the onset and the batch, is written to
        ``intervention.json`` in the OmniSafe run directory.
        """
        api = self._intervention_api
        treatment = self._onset["treatment"]
        ac = self._actor_critic  # type: ignore[attr-defined]
        actor = ac.actor
        log_dir = Path(self._logger.log_dir)  # type: ignore[attr-defined]
        path = log_dir / INTERVENTION_FILE
        if self._intervention_record is not None or isinstance(actor.mean, api["injected_head"]) or path.exists():
            raise RunRefused(f"the {treatment} intervention is applied once per run (HANDOVER.md section 8); it was applied already")
        epoch = int(self._logger.current_epoch)  # type: ignore[attr-defined]
        if epoch != self._onset_epoch:
            raise RunRefused(f"the {treatment} intervention belongs to onset epoch {self._onset_epoch}, not {epoch}")
        checkpoint = rundir.checkpoint_file(log_dir, epoch)
        if not checkpoint.is_file():
            raise RunRefused(f"the onset checkpoint {checkpoint.name} does not exist before the intervention")
        onset_step = int(self._onset["onset_step"])
        pilot_cfgs = self._pilot_cfgs()  # type: ignore[attr-defined]
        if pilot_cfgs.get("plasticity"):
            from metrics.hook import RECORDER_ATTRIBUTE

            recorder = getattr(self, RECORDER_ATTRIBUTE, None)
            if recorder is None or onset_step not in recorder.written_steps:
                raise RunRefused(f"the plasticity row at onset step {onset_step} is not written before the intervention")
        normalise = bool(self._cfgs.algo_cfgs.obs_normalize)  # type: ignore[attr-defined]
        normalizer = self._env.save()["obs_normalizer"] if normalise else None  # type: ignore[attr-defined]
        batch = api["normalise_frozen"](api["as_batch_tensor"](self._fixed_batch), normalizer)  # never pushes
        seed = int(self._cfgs.seed)  # type: ignore[attr-defined]
        record = dict(api["function"](actor, ac.actor_optimizer, api["generator"](seed),
                                      int(self._onset["intervention"]["depth"]), batch=batch))
        extra = {
            "run_id": pilot_cfgs.get("run_id"),
            "run_seed": seed,
            "onset_step": onset_step,
            "onset_epoch": self._onset_epoch,
            "applied": "at the start of the rollout of the onset epoch, after the onset checkpoint and its "
                       "plasticity row (HANDOVER.md section 8)",
            "applied_by": "envs.onset (Role 2)",
            "output_check_batch": {
                "task": str(self._env_id),  # type: ignore[attr-defined]
                "states": int(batch.shape[0]),
                "source": "metrics.batches.load_fixed_batch",
                "normalised": normalise,
                "normaliser_count": None if normalizer is None else int(normalizer._count),
            },
        }
        clash = sorted(set(extra) & set(record))
        if clash:
            raise RunRefused(f"the intervention record already has the keys {clash}")
        record.update(extra)
        write_json_once(path, record)
        self._intervention_record = record
        self._logger.log(  # type: ignore[attr-defined]
            f"Onset: {treatment} applied at the start of epoch {epoch} (step {onset_step}); record {path.name}"
        )
        return record


class OnsetPPOLag(OnsetMixin, FullStateCheckpointMixin, PPOLag):
    """Study A's PPO-Lagrangian: every arm but the PID check (``make_algorithm``)."""

    _LAGRANGIAN_BASE = PPOLag


class OnsetCPPOPID(OnsetMixin, FullStateCheckpointMixin, CPPOPID):
    """Study A's PID check on OmniSafe's CPPOPID (``make_pid_algorithm``; Table 3.3 "PID check")."""

    _LAGRANGIAN_BASE = CPPOPID


# ---------------------------------------------------------------------------
# The factories (pilot.manifest.PLUGINS)
# ---------------------------------------------------------------------------


def _check_factory_config(cfgs: Any, spec: RunSpec, settings: Mapping[str, Any], *, registered: bool) -> None:
    """What the factory checks beyond ``check_training_config``: the configuration is this spec's run."""
    what = spec.run_id
    if "onset_cfgs" in cfgs:
        raise _refuse(what, "the configuration already has onset_cfgs; the factory adds them once")
    try:
        seed = _whole_number(_config_value(cfgs, "seed"))
        E = _whole_number(_config_value(cfgs, "algo_cfgs", "steps_per_epoch"))
        total = _whole_number(_config_value(cfgs, "train_cfgs", "total_steps"))
        epochs = _whole_number(_config_value(cfgs, "train_cfgs", "epochs"))
    except (TypeError, ValueError) as exc:
        raise _refuse(what, f"the configuration is malformed: {exc}") from exc
    if seed != spec.seed:
        raise _refuse(what, f"cfgs.seed {seed} is not the spec's seed {spec.seed}")
    pilot_cfgs = cfgs.get("pilot_cfgs")
    if registered:
        if not isinstance(pilot_cfgs, Mapping):
            raise _refuse(what, "a registered run carries the launcher's pilot_cfgs")
        if (total, epochs * E) != (spec.total_steps, spec.total_steps):
            raise _refuse(what, f"the configuration trains {epochs} x {E} = {epochs * E} of {total} steps; the spec has "
                                f"{spec.total_steps}")
        for key, expected in (("run_id", spec.run_id), ("onset_step", settings["onset_step"]),
                              ("treatment", spec.treatment), ("controller_variant", spec.controller_variant),
                              ("plasticity", True)):
            if pilot_cfgs.get(key) != expected:
                raise _refuse(what, f"pilot_cfgs.{key} is {pilot_cfgs.get(key)!r}, expected {expected!r} "
                                    "(pilot.launch.pilot_config; pilot/contracts.py contract 2)")


def _make(env_id: str, cfgs: Any, spec: Any, *, pid: bool) -> Any:
    spec = _as_spec(spec)
    what = spec.run_id
    if (spec.controller_variant == R.PID_VARIANT) != pid:
        raise _refuse(what, f"the {'PID' if spec.controller_variant == R.PID_VARIANT else 'PPO-Lagrangian'} arm is built "
                            f"by {'make_pid_algorithm' if not pid else 'make_algorithm'}")
    if env_id != spec.task:
        raise _refuse(what, f"env_id {env_id!r} is not the spec's task {spec.task!r}")
    if spec.run_id.startswith(DETERMINISM_PREFIX) and not is_determinism_spec(spec):
        raise _refuse(what, f"a {DETERMINISM_PREFIX} run is the determinism check's; this spec is not "
                            f"pilot.scheduler.determinism_spec({spec.plugin!r}, {spec.total_steps}) (pilot/scheduler.py)")
    registered = is_registered_spec(spec)
    if not isinstance(cfgs, Config):  # a plain dict would fail OmniSafe's own checks (base_algo.py:39), a crash
        raise _refuse(what, "the configuration must be OmniSafe's Config")
    try:
        steps_per_epoch = _config_value(cfgs, "algo_cfgs", "steps_per_epoch")
    except RunRefused as exc:
        raise _refuse(what, str(exc)) from exc
    settings = build_onset_cfgs(spec, steps_per_epoch=steps_per_epoch, registered=registered)
    _check_factory_config(cfgs, spec, settings, registered=registered)
    if settings["controller_variant"] == "warm_started":
        settings["warm_start"] = warm_start_source(cfgs, spec, settings["onset_step"])
    cfgs["onset_cfgs"] = Config.dict2config(settings)  # before construction: in config.json and the config hash
    try:
        check_training_config(cfgs, pid=pid)  # before the environment is built: refuse early
    except RunRefused as exc:
        del cfgs["onset_cfgs"]
        raise _refuse(what, str(exc)) from exc
    cls = OnsetCPPOPID if pid else OnsetPPOLag
    return cls(env_id=env_id, cfgs=cfgs)


def make_algorithm(env_id: str, cfgs: Any, spec: Any) -> OnsetPPOLag:
    """``PLUGINS['study_a']``: the PPO-Lagrangian Study A run of ``spec`` (every arm but the PID check).

    Validates the spec (``build_onset_cfgs``; registered unless a smoke copy or the determinism
    check's ``pilot.scheduler.determinism_spec``, ``is_registered_spec``; a ``DET-`` spec that is
    not that function's output is refused) and the configuration, reads the warm-start value for a
    warm-started arm (``warm_start_source``), stores the settings in ``cfgs['onset_cfgs']`` and
    builds ``OnsetPPOLag``. RunRefused for every configuration or specification problem.
    """
    return _make(env_id, cfgs, spec, pid=False)


def make_pid_algorithm(env_id: str, cfgs: Any, spec: Any) -> OnsetCPPOPID:
    """``PLUGINS['study_a_pid']``: the PID check (Table 3.3) on OmniSafe's CPPOPID, as ``make_algorithm``."""
    return _make(env_id, cfgs, spec, pid=True)

"""The registered design as a list of runs.

Every run of the study is one immutable ``RunSpec`` with a deterministic ``run_id``. The design
is generated from ``configs/registered.py`` and the constants below that Table 9.1 decides (HANDOVER.md section 10):

* Study A main sweep: Part 3.2, Table 3.2 (13 arms x 3 tasks x 5 seeds = 195 runs).
* Study A additional arms: Table 3.3 (treatments 135, controller variants 90, PID check 15).
* Study B: Part 3.3, Table 3.4 (7 arms x 5 seeds = 35 runs; 140 few-shot continuations).
* The pilot: Part 3.6 (8 runs).
* The re-pilot: Part 6 (8 runs, ``P1-``).
* The battery's continuations (Table 2.2 "Reward-only fine-tuning" and "Transfer"): built from a
  matched ledger row by ``battery_continuation``, never part of ``design("all")``.

Owner: pilot owner (Role 1). Where the text leaves a value open (``configs.registered.PENDING``),
the spec is still generated so that it can be listed and costed, but it carries ``pending`` keys
and the scheduler refuses to launch it until the amendment log answers the question (First Tasks,
habit 4). ``run_gate_keys`` is the one place that decides which open question gates which run.
``apply_cuts`` applies the cut order of Part 6.1.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from fractions import Fraction
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence

from configs import registered as R

# The role that owns each top-level package, named in messages about a module not written yet.
ROLE_OWNERS: Mapping[str, str] = MappingProxyType({
    "pilot": "Pilot owner (Role 1)",
    "envs": "Environment and tests (Role 2)",
    "metrics": "Metrics and interventions (Role 3)",
    "analysis": "Analysis and results (Role 4)",
    "studyb": "Study B and literature (Role 5)",
})

# Algorithm plug-ins. The launcher imports "module:function" and calls it as
# function(env_id, cfgs, spec) -> omnisafe BaseAlgo. Owners are named so that a missing
# plug-in fails with a message saying who provides it.
PLUGINS: Mapping[str, tuple[str, str]] = {
    "ppolag": ("pilot.algorithms:make_ppolag", ROLE_OWNERS["pilot"]),
    "unconstrained_ppo": ("pilot.algorithms:make_unconstrained_ppo", ROLE_OWNERS["pilot"]),
    "study_a": ("envs.onset:make_algorithm", ROLE_OWNERS["envs"]),
    "study_a_pid": ("envs.onset:make_pid_algorithm", ROLE_OWNERS["envs"]),
    "study_b": ("studyb.conditioning:make_algorithm", ROLE_OWNERS["studyb"]),
    "study_b_fewshot": ("studyb.conditioning:make_fewshot_algorithm", ROLE_OWNERS["studyb"]),
    # Table 2.2 "Reward-only fine-tuning" and "Transfer" (envs/continuations.py)
    "battery_finetune": ("envs.continuations:make_finetune_algorithm", ROLE_OWNERS["envs"]),
    "battery_transfer": ("envs.continuations:make_transfer_algorithm", ROLE_OWNERS["envs"]),
}

BASE_ALGOS = ("PPOLag", "CPPOPID", "PPO")
GROUPS = (
    "pilot", "main", "treatment", "controller", "pid", "study_b", "study_b_fewshot", "determinism",
    "battery_finetune", "battery_transfer",
)
# Runs that continue a finished parent from one of its checkpoints. They follow the parent's seed
# (``with_seed`` refuses them), end as ``continued`` and write no ledger row of their own: their
# results are read into the parent's row by enrichment.
CONTINUATION_GROUPS = ("study_b_fewshot", "battery_finetune", "battery_transfer")
BATTERY_CONDITIONS = ("finetune", "transfer")  # Table 2.2 rows 3 and 4, in the table's order
STUDY_A_PARENT_GROUPS = ("main", "treatment", "controller", "pid")  # Part 3.6: "The pilot's runs are not reused"

# Table 2.2 "Transfer": "the held-out task on the same robot (Goal to Button and Button to Goal)".
# Only the Point pair is registered (R.TRANSFER_TASKS); SafetyCarGoal1-v0's held-out task is not named there.
# Table 9.1 (Q-transfer-obs) names the same robot's Button task at the same level.
TRANSFER_TASKS_ANSWERED: Mapping[str, str] = MappingProxyType({
    "SafetyCarGoal1-v0": "SafetyCarButton1-v0",  # answered in Table 9.1 (Q-transfer-obs)
})
# Table 2.2 names no controller for "Reward-only fine-tuning" or "Transfer". Every battery
# continuation, a PID-arm (CPPOPID) parent's included, trains with PPO-Lagrangian's update; for transfer
# that means a fresh multiplier updated by naive PPO-Lagrangian, not by the parent's PID controller.
CONTINUATION_BASE_ALGO = "PPOLag"  # answered in Table 9.1 (Q-continuations)


def short_task(task: str) -> str:
    """'SafetyPointGoal1-v0' -> 'PointGoal1'."""
    if not (task.startswith("Safety") and task.endswith("-v0")):
        raise ValueError(f"unexpected task id {task!r}")
    return task[len("Safety"):-len("-v0")]


_CONTROL_SHORT = {"total_steps": "total", "constrained_steps": "constrained"}


@dataclass(frozen=True)
class RunSpec:
    """One run: one full training of one agent with one seed (How to Read This Document).

    Frozen, but not hashable: ``params`` is a dict, so ``hash(spec)`` raises TypeError (key specs
    by ``run_id``).
    """

    run_id: str
    study: str
    task: str
    arm: str
    seed: int
    total_steps: int
    base_algo: str
    plugin: str
    group: str
    pilot: bool = False
    N: float | None = None
    onset_step: int | None = None
    onset_shape: str | None = None
    step_matching: str | None = None
    treatment: str | None = None
    controller_variant: str | None = None
    training_levels: tuple[float, ...] | None = None
    depends_on: tuple[str, ...] = ()
    pending: tuple[str, ...] = ()
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.study not in ("A", "B"):
            raise ValueError(f"study must be 'A' or 'B', got {self.study!r}")
        if self.base_algo not in BASE_ALGOS:
            raise ValueError(f"base_algo must be one of {BASE_ALGOS}, got {self.base_algo!r}")
        if self.plugin not in PLUGINS:
            raise ValueError(f"unknown plugin {self.plugin!r}")
        if self.group not in GROUPS:
            raise ValueError(f"unknown group {self.group!r}")
        for name in ("total_steps", "seed", "onset_step"):
            value = getattr(self, name)
            if not (isinstance(value, int) and not isinstance(value, bool)) and not (name == "onset_step" and value is None):
                raise ValueError(f"{self.run_id}: {name} must be an integer, got {value!r}")
        if self.total_steps <= 0 or self.total_steps % R.STEPS_PER_EPOCH:
            raise ValueError(
                f"{self.run_id}: total_steps {self.total_steps} is not a positive multiple of "
                f"{R.STEPS_PER_EPOCH} (OmniSafe floor-divides steps into epochs)"
            )
        if self.onset_step is not None and (
            self.onset_step < 0 or self.onset_step % R.STEPS_PER_EPOCH or self.onset_step > self.total_steps
        ):
            raise ValueError(f"{self.run_id}: onset_step {self.onset_step} is not a whole epoch within the run")
        if not self.run_id.endswith(f"-s{self.seed}"):
            raise ValueError(f"run_id {self.run_id!r} must end with '-s{self.seed}'")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")

    @property
    def arm_id(self) -> str:
        """The run_id without its seed: identifies the arm for seed bookkeeping (Part 5.6)."""
        return self.run_id[: -len(f"-s{self.seed}")]

    @property
    def epochs(self) -> int:
        """Training epochs of R.STEPS_PER_EPOCH steps each."""
        return self.total_steps // R.STEPS_PER_EPOCH

    @property
    def run_equivalents(self) -> float:
        """Training steps in units of one registered run of T = 10,000,000 steps."""
        return self.total_steps / R.TOTAL_STEPS

    @property
    def open_questions(self) -> tuple[str, ...]:
        """The spec's pending keys that the amendment log has not answered yet."""
        return tuple(k for k in self.pending if R.is_open(k))

    @property
    def launchable(self) -> bool:
        """True when no stored pending key is still open (the scheduler uses ``open_run_gates``)."""
        return not self.open_questions

    def with_seed(self, seed: int) -> RunSpec:
        """The same arm with another seed (replacement or surplus seed).

        A continuation has no seed of its own to change: it continues its parent's checkpoint with
        the parent's seed, so it is refused (build the new parent's continuations instead).
        """
        if self.group in CONTINUATION_GROUPS:
            raise ValueError(
                f"{self.run_id} is a continuation ({self.group}); it follows its parent's seed and checkpoint"
            )
        data = self.to_dict()
        data["seed"] = seed
        data["run_id"] = f"{self.arm_id}-s{seed}"
        data["depends_on"] = [_reseed_dependency(dep, self.seed, seed) for dep in self.depends_on]
        if "parent_run_id" in data["params"]:
            data["params"]["parent_run_id"] = _reseed_dependency(data["params"]["parent_run_id"], self.seed, seed)
        return RunSpec.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        """The spec as a plain mapping (tuples kept; ``params`` copied)."""
        data = asdict(self)
        data["params"] = dict(self.params)
        return data

    def to_json(self) -> str:
        """The spec as JSON with sorted keys."""
        return json.dumps(self.to_dict(), sort_keys=True)

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> RunSpec:
        """The spec of a mapping written by ``to_dict`` (or read back from JSON)."""
        d = dict(data)
        for key in ("depends_on", "pending"):
            d[key] = tuple(d.get(key) or ())
        if d.get("training_levels") is not None:
            d["training_levels"] = tuple(float(x) for x in d["training_levels"])
        d["params"] = dict(d.get("params") or {})
        return RunSpec(**d)

    @staticmethod
    def from_json(text: str) -> RunSpec:
        """The spec of a JSON text written by ``to_json``."""
        return RunSpec.from_dict(json.loads(text))


# An auxiliary run: a run queued only because a replacement depends on it (the N = 0 run of a warm-started
# replacement's seed, which the N = 0 arm never received; ``schedule resolve RUN add-auxiliary``, Table 9.1,
# Q-warm-start). The ledger writer puts AUXILIARY_NOTE into its row's notes, and the arm's matching
# (pilot.enrichment.arm_readiness) and the analysis (analysis.data.build_dataset) leave the row out of its arm.
AUXILIARY_NOTE = "auxiliary run (Q-warm-start): not a seed of its arm"


def is_auxiliary_row(row: Mapping[str, Any]) -> bool:
    """True for the ledger row of an auxiliary run (its notes carry AUXILIARY_NOTE)."""
    return AUXILIARY_NOTE in str(row.get("notes") or "").split("; ")


# Why a run of seed >= 5 exists (the ledger writer puts the note into its row's notes, from the scheduler's
# record): a surplus seed (Part 5.5; also a replacement of a surplus seed) or the replacement of a failed
# registered seed (Part 5.6: "repeated with the next unused seed"). Both take the arm's next unused seed, so the
# seed alone does not tell them apart; Q-surplus-in-analysis's "five seeds" reading keeps the registered seeds
# and their replacements and leaves the surplus seeds out (analysis.data.five_seed_view).
SURPLUS_NOTE = "surplus seed (Part 5.5)"
REPLACEMENT_NOTE_PREFIX = "replacement (Part 5.6) of "


def replacement_note(replaced_run_id: str) -> str:
    """The note of a run that replaces the registered-seed run ``replaced_run_id`` (Part 5.6)."""
    return f"{REPLACEMENT_NOTE_PREFIX}{replaced_run_id}"


def seed_role(row: Mapping[str, Any]) -> str | None:
    """The seed role in a ledger row's notes: "surplus", "replacement" or None.

    Read from ``SURPLUS_NOTE`` and ``replacement_note``; None when the notes say neither (a registered
    seed, or a row written before the notes were recorded).
    """
    notes = str(row.get("notes") or "").split("; ")
    if SURPLUS_NOTE in notes:
        return "surplus"
    if any(n.startswith(REPLACEMENT_NOTE_PREFIX) for n in notes):
        return "replacement"
    return None


def _reseed_dependency(dep: str, old_seed: int, new_seed: int) -> str:
    suffix = f"-s{old_seed}"
    if not dep.endswith(suffix):
        raise ValueError(f"dependency {dep!r} does not carry seed {old_seed}")
    return dep[: -len(suffix)] + f"-s{new_seed}"


# ---------------------------------------------------------------------------
# Run gates: which open question holds which run
# ---------------------------------------------------------------------------

# The run-gate keys of ``run_gate_keys``, in canonical order. ``_gated`` appends any a spec lacks after
# the keys it already carries (its own arithmetic: Q-rounding, Q-data-control-lr, Q-selection-window,
# Q-pilot-unconstrained-seed; Q-continuations for a few-shot spec). ``surplus_spec`` appends
# Q-surplus-arm-set and the scheduler appends Q-seed-collision after the run gates.
RUN_GATE_ORDER = (
    "Q-cost-critic", "Q-jc-window", "Q-ramp-step", "Q-warm-start", "Q-rate-limit", "Q-pid-eq9",
    "Q-reset-injection", "Q-plasticity-definitions", "Q-continuations", "Q-transfer-obs",
    "Q-budget-normalisation", "Q-level-jc", "Q-continuous-bins", "Q-search-before-pilot", "Q-studyb-order",
)
LAGRANGIAN_ALGOS = ("PPOLag", "CPPOPID")


def run_gate_keys(spec: RunSpec) -> tuple[str, ...]:
    """The open questions whose answers the training of ``spec`` implements (run gates; each answered in
    Table 9.1, so a gate holds only while its key is open).

    Each key's text in
    ``configs.registered.PENDING`` states the question, its answer and the runs it gates:

    * Q-cost-critic: every Study A training run with an onset after step 0 (the pilot's N = 0.50
      runs included);
    * Q-jc-window: every PPO-Lagrangian or PID run of the design (pilot, main, treatment,
      controller, pid, study_b, study_b_fewshot, battery_transfer), but not the fine-tuning
      continuation (its multiplier is held at 0), not unconstrained PPO and not the ppolag
      determinism check (group determinism; it runs OmniSafe unchanged, Table 3.1; the other
      determinism checks keep their arm's group and stored keys, see ``open_run_gates``);
    * Q-ramp-step (ramp arms), Q-warm-start and Q-rate-limit (the controller variants), Q-pid-eq9
      (the PID check), Q-reset-injection (partial reset and plasticity injection);
    * Q-plasticity-definitions: the non-pilot Study A training runs, whose ledger rows hold the
      plasticity metrics;
    * Q-continuations: every continuation; Q-transfer-obs: the transfer continuations;
    * Q-budget-normalisation and Q-level-jc: every budget-conditioned run (the pilot's Moderate
      run and the few-shot continuations included); Q-continuous-bins: the Continuous arm's runs;
    * Q-search-before-pilot: the Study B runs of the pilot and of the re-pilot; Q-studyb-order: the Study B training
      runs of the main study.

    A battery continuation copies its parent's N, shape, control, treatment and controller for
    bookkeeping only (not its onset step); it trains PPO-Lagrangian from the parent's checkpoint
    (fine-tuning with the multiplier held at 0), so the Study A training keys do not apply to it.
    """
    keys: set[str] = set()
    continuation = spec.group in CONTINUATION_GROUPS
    if spec.base_algo in LAGRANGIAN_ALGOS and spec.group != "determinism" and spec.plugin != "battery_finetune":
        keys.add("Q-jc-window")
    if spec.study == "A" and not continuation:
        if (spec.onset_step or 0) > 0:
            keys.add("Q-cost-critic")
        if spec.onset_shape == "ramp":
            keys.add("Q-ramp-step")
        if spec.controller_variant == "warm_started":
            keys.add("Q-warm-start")
        if spec.controller_variant == "rate_limited":
            keys.add("Q-rate-limit")
        if spec.controller_variant == R.PID_VARIANT:
            keys.add("Q-pid-eq9")
        if spec.treatment in ("reset", "injection"):
            keys.add("Q-reset-injection")
        if not spec.pilot and spec.plugin in ("study_a", "study_a_pid"):
            keys.add("Q-plasticity-definitions")
    if continuation:
        keys.add("Q-continuations")
    if spec.plugin == "battery_transfer":
        keys.add("Q-transfer-obs")
    if spec.plugin in ("study_b", "study_b_fewshot"):
        keys.update(("Q-budget-normalisation", "Q-level-jc"))
        if spec.arm == "Continuous":
            keys.add("Q-continuous-bins")
    if spec.study == "B" and spec.pilot:
        keys.add("Q-search-before-pilot")
    if spec.group == "study_b":
        keys.add("Q-studyb-order")
    return tuple(k for k in RUN_GATE_ORDER if k in keys)


_REGISTERED_ARMS: dict[str, str] | None = None  # built once, on first use (the design does not change in a process)


def _registered_arms() -> dict[str, str]:
    """{arm_id: plug-in} of the registered design (the pilot, its re-pilot and every arm of ``design("all")``)."""
    global _REGISTERED_ARMS
    if _REGISTERED_ARMS is None:
        _REGISTERED_ARMS = {s.arm_id: s.plugin for s in (*pilot(), *pilot(revision=1), *design("all", R.SEEDS[:1]))}
    return _REGISTERED_ARMS


def is_registered_run(spec: RunSpec) -> bool:
    """A run of the registered design: an arm of it (any seed) trained by that arm's plug-in, or a battery
    continuation whose ``parent_run_id`` names a seed of a registered arm."""
    if spec.group in ("battery_finetune", "battery_transfer"):
        parent = str(spec.params.get("parent_run_id") or "")
        return parent.rsplit("-s", 1)[0] in _registered_arms()
    return _registered_arms().get(spec.arm_id) == spec.plugin


def open_run_gates(spec: RunSpec, *, registered: bool = True) -> tuple[str, ...]:
    """The open questions that hold ``spec``'s launch now: its stored ``pending`` keys and, for a run of
    the registered design, the run gates the code checked out now derives for it (``run_gate_keys``),
    those the amendment log has not answered.

    A spec is stored when it is queued, so its ``pending`` tuple can predate a run gate added since;
    the scheduler and the launcher read this, never the stored tuple alone. A
    determinism check's spec keeps its stored keys only (``pilot.scheduler.determinism_spec`` leaves out
    the keys that change no update), and so does a run outside the design (a test's or a SMOKE- copy,
    which carries its arm's keys) or one launched in smoke mode (``registered=False``: the launcher's
    ``--allow-dirty``/``--allow-pending``, the scheduler's smoke mode).
    """
    gated = registered and spec.group != "determinism" and is_registered_run(spec)
    keys = (*spec.pending, *run_gate_keys(spec)) if gated else spec.pending
    return tuple(k for k in dict.fromkeys(keys) if R.is_open(k))


def _gated(spec: RunSpec) -> RunSpec:
    """``spec`` with its run-gate keys appended to the keys it already carries (order kept)."""
    missing = tuple(k for k in run_gate_keys(spec) if k not in spec.pending)
    return replace(spec, pending=(*spec.pending, *missing)) if missing else spec


# ---------------------------------------------------------------------------
# Onset arithmetic (Table 2.1 "Onset fraction N"; Table 2.4 last two rows)
# ---------------------------------------------------------------------------


def _fraction(n: float) -> Fraction:
    return Fraction(str(n))


def onset_schedule(N: float, control: str) -> tuple[int, int, tuple[str, ...]]:
    """Return (onset_step, total_steps, pending) for onset fraction N under a step-matching control.

    Total-steps matched: every arm trains T steps; onset at N*T (Table 2.4).
    Constrained-steps matched: every arm trains T steps under the constraint; total T/(1-N)
    (Table 2.4). For N = 0.10 and 0.25 this is not a whole number of epochs; Table 9.1 (Q-rounding;
    First Tasks, section 5) rounds the onset to the nearest whole epoch and keeps exactly T
    constrained steps after it. For these the returned pending is ``('Q-rounding',)``, so the spec
    carries Q-rounding (with its other keys) as a run gate.
    """
    n = _fraction(N)
    T = R.TOTAL_STEPS
    E = R.STEPS_PER_EPOCH
    if not 0 <= n < 1:
        raise ValueError(f"onset fraction must be in [0, 1), got {N}")
    if control == "total_steps":
        onset = n * T
        if onset.denominator != 1 or int(onset) % E:
            raise ValueError(f"N*T = {onset} is not a whole epoch")
        return int(onset), T, ()
    if control == "constrained_steps":
        exact_onset = n * T / (1 - n)
        if exact_onset.denominator == 1 and int(exact_onset) % E == 0:
            onset = int(exact_onset)
            return onset, onset + T, ()
        epochs = exact_onset / E
        if epochs - int(epochs) == Fraction(1, 2):
            raise ValueError("onset lies exactly between two epochs; the rounding of Q-rounding is undefined")
        rounded_epochs = int(epochs + Fraction(1, 2))  # nearest whole epoch (exact halves refused above)
        onset = rounded_epochs * E
        return onset, onset + T, ("Q-rounding",)
    raise ValueError(f"unknown step-matching control {control!r}")


# ---------------------------------------------------------------------------
# Study A
# ---------------------------------------------------------------------------


def _study_a_arm(N: float, shape: str | None, control: str | None, suffix: str | None) -> str:
    if N == 0:
        arm = "N0.00"
    else:
        arm = f"N{N:.2f}-{shape}-{_CONTROL_SHORT[control]}"
    return f"{arm}-{suffix}" if suffix else arm


def _study_a_spec(
    task: str,
    N: float,
    seed: int,
    *,
    shape: str | None,
    control: str | None,
    group: str,
    treatment: str | None = None,
    controller: str | None = None,
    pilot: bool = False,
) -> RunSpec:
    suffix = treatment or controller
    arm = _study_a_arm(N, shape, control, suffix)
    prefix = "P-" if pilot else ""
    run_id = f"{prefix}A-{short_task(task)}-{arm}-s{seed}"
    if N == 0:
        onset, total, pending = 0, R.TOTAL_STEPS, ()
    else:
        onset, total, pending = onset_schedule(N, control)
    params: dict[str, Any] = {"cost_limit": R.COST_LIMIT}
    if shape == "ramp":
        params["d_loose"] = R.D_LOOSE[task]
        params["ramp_window_steps"] = R.RAMP_WINDOW_STEPS
    depends: tuple[str, ...] = ()
    base_algo, plugin = "PPOLag", "study_a"
    if treatment == "additional_constrained":
        # Table 2.4: the late arm continues under the constraint for N*T further steps after T.
        total = R.TOTAL_STEPS + int(_fraction(N) * R.TOTAL_STEPS)
        params["additional_constrained_steps"] = int(_fraction(N) * R.TOTAL_STEPS)
        pending = pending + ("Q-data-control-lr",)
    if controller == "warm_started":
        # Table 2.4: the multiplier is set to the value the N = 0 arm of the same task and seed
        # had reached at the same step, so that run must exist first.
        depends = (f"{prefix}A-{short_task(task)}-N0.00-s{seed}",)
    if controller == "rate_limited":
        params["rate_limit_relative"] = R.RATE_LIMIT_RELATIVE
        params["rate_limit_absolute"] = R.RATE_LIMIT_ABSOLUTE
    if controller == R.PID_VARIANT:
        base_algo, plugin = "CPPOPID", "study_a_pid"
    if total % R.CHECKPOINT_INTERVAL_STEPS:
        pending = pending + ("Q-selection-window",)
    return _gated(RunSpec(
        run_id=run_id,
        study="A",
        task=task,
        arm=arm,
        seed=seed,
        total_steps=total,
        base_algo=base_algo,
        plugin=plugin,
        group=group,
        pilot=pilot,
        N=float(N),
        onset_step=onset,
        onset_shape=shape,
        step_matching=control,
        treatment=treatment,
        controller_variant=controller,
        depends_on=depends,
        pending=pending,
        params=params,
    ))


def study_a_main(seeds: Iterable[int] = R.SEEDS) -> list[RunSpec]:
    """Table 3.2: per task, the N = 0 arm plus 3 fractions x 2 shapes x 2 controls."""
    specs: list[RunSpec] = []
    for task in R.TASKS_STUDY_A:
        for seed in seeds:
            specs.append(_study_a_spec(task, 0.0, seed, shape=None, control=None, group="main"))
            for N in R.LATE_ONSET_FRACTIONS:
                for shape in R.ONSET_SHAPES:
                    for control in R.STEP_MATCHING_CONTROLS:
                        specs.append(_study_a_spec(task, N, seed, shape=shape, control=control, group="main"))
    return specs


def study_a_treatments(seeds: Iterable[int] = R.SEEDS) -> list[RunSpec]:
    """Table 3.3 "Mediation treatments": abrupt, total-steps matched, N in {0.10, 0.25, 0.50}."""
    return [
        _study_a_spec(task, N, seed, shape="abrupt", control="total_steps", group="treatment", treatment=t)
        for task in R.TASKS_STUDY_A
        for N in R.LATE_ONSET_FRACTIONS
        for t in R.TREATMENTS
        for seed in seeds
    ]


def study_a_controllers(seeds: Iterable[int] = R.SEEDS) -> list[RunSpec]:
    """Table 3.3 "Controller variants": warm-started and rate-limited, abrupt, total-steps matched."""
    return [
        _study_a_spec(task, N, seed, shape="abrupt", control="total_steps", group="controller", controller=c)
        for task in R.TASKS_STUDY_A
        for N in R.LATE_ONSET_FRACTIONS
        for c in R.CONTROLLER_VARIANTS
        for seed in seeds
    ]


def study_a_pid(seeds: Iterable[int] = R.SEEDS) -> list[RunSpec]:
    """Table 3.3 "PID check": SafetyPointGoal1-v0 only, abrupt; total-steps matched is answered in
    Table 9.1 (Q-pid-eq9; Table 3.3 names no step-matching control for the PID row)."""
    return [
        _study_a_spec(task, N, seed, shape="abrupt", control="total_steps", group="pid", controller=R.PID_VARIANT)
        for task in R.PID_TASKS
        for N in R.LATE_ONSET_FRACTIONS
        for seed in seeds
    ]


# ---------------------------------------------------------------------------
# Study B (Table 2.5; Table 3.4)
# ---------------------------------------------------------------------------


def _study_b_spec(arm: str, seed: int, *, pilot: bool = False) -> RunSpec:
    levels = R.STUDY_B_ARMS[arm]
    params: dict[str, Any] = {"budget_observation_divisor": R.BUDGET_OBSERVATION_DIVISOR}
    if levels is None:
        params["continuous_range"] = list(R.CONTINUOUS_RANGE)
        params["continuous_bin_width"] = R.CONTINUOUS_BIN_WIDTH
    prefix = "P-" if pilot else ""
    return _gated(RunSpec(
        run_id=f"{prefix}B-{arm}-s{seed}",
        study="B",
        task=R.TASKS_STUDY_B[0],
        arm=arm,
        seed=seed,
        total_steps=R.TOTAL_STEPS,
        base_algo="PPOLag",
        plugin="study_b",
        group="pilot" if pilot else "study_b",
        pilot=pilot,
        training_levels=levels,
        params=params,
    ))


def is_primary_comparison(spec: RunSpec) -> bool:
    """Part 5.5: an arm that receives surplus seeds (answered in Table 9.1, Q-surplus-arm-set).

    "the N = 0 and N = 0.50 arms under both step-matching controls on SafetyPointGoal1-v0, and the
    seven arms of Study B". No onset shape is excluded: the four N = 0.50 arms, abrupt and ramp, each
    under both controls.
    """
    late_n = max(R.LATE_ONSET_FRACTIONS)
    if spec.group == "main":
        return spec.task == R.PRIMARY_TASK and spec.N in (R.ONSET_FRACTIONS[0], late_n)
    return spec.group == "study_b"


def surplus_spec(base: RunSpec, seed: int) -> RunSpec:
    """A surplus seed of a primary-comparison arm (Part 5.5).

    The ramp arms at N = 0.50, which Part 5.5's list includes without naming the onset shape (answered
    in Table 9.1, Q-surplus-arm-set), carry the key in ``pending``: it holds them only while the key is
    open.
    """
    if not is_primary_comparison(base):
        raise ValueError(f"{base.run_id} is not in the primary-comparison set of Part 5.5")
    spec = base.with_seed(seed)
    if spec.onset_shape == "ramp" and "Q-surplus-arm-set" not in spec.pending:
        spec = replace(spec, pending=(*spec.pending, "Q-surplus-arm-set"))
    return spec


def study_b(seeds: Iterable[int] = R.SEEDS) -> list[RunSpec]:
    """Table 3.4: the seven Study B arms x seeds."""
    return [_study_b_spec(arm, seed) for arm in R.STUDY_B_ARMS for seed in seeds]


def study_b_fewshot(parents: Iterable[RunSpec]) -> list[RunSpec]:
    """Part 3.3: each finished run is continued once under each unseen budget for 1,000,000 steps.

    Each continuation starts from its parent's final checkpoint, ``parent_step = parent.total_steps``
    ("Each finished run ... continued once"; Table 9.1, Q-continuations, which gates these runs).
    """
    specs = []
    for parent in parents:
        if parent.group != "study_b":
            raise ValueError(f"{parent.run_id} is not a Study B training run")
        for budget in R.UNSEEN_BUDGETS:
            specs.append(_gated(RunSpec(
                run_id=f"B-{parent.arm}-fewshot-b{budget:g}-s{parent.seed}",
                study="B",
                task=parent.task,
                arm=parent.arm,
                seed=parent.seed,
                total_steps=R.FEWSHOT_STEPS,
                base_algo="PPOLag",
                plugin="study_b_fewshot",
                group="study_b_fewshot",
                training_levels=parent.training_levels,
                depends_on=(parent.run_id,),
                pending=("Q-continuations",),
                params={
                    "parent_run_id": parent.run_id,
                    "parent_step": parent.total_steps,
                    "budget": budget,
                    "horizons": list(R.FEWSHOT_HORIZONS),
                    "fresh_multiplier_init": R.LAGRANGE_MULTIPLIER_INIT,
                },
            )))
    return specs


# ---------------------------------------------------------------------------
# Battery continuations (Table 2.2 "Reward-only fine-tuning" and "Transfer")
# ---------------------------------------------------------------------------


def transfer_task(task: str) -> str:
    """The held-out task of Table 2.2 "Transfer" for a Study A task.

    The registered Point pair (R.TRANSFER_TASKS), else Table 9.1's for SafetyCarGoal1-v0
    (TRANSFER_TASKS_ANSWERED; Q-transfer-obs, a run gate through ``run_gate_keys``).
    """
    if task in R.TRANSFER_TASKS:
        return R.TRANSFER_TASKS[task]
    if task in TRANSFER_TASKS_ANSWERED:
        return TRANSFER_TASKS_ANSWERED[task]
    raise ValueError(f"{task} has no held-out task for transfer (Table 2.2)")


def battery_continuation(parent: RunSpec, row: Any, condition: str) -> RunSpec:
    """The fine-tuning or transfer continuation of a matched Study A run (Table 2.2).

    Table 2.2: "Training continues for 1,000,000 steps ... with the cost term removed from the
    objective and the multiplier frozen at zero" (fine-tuning) and "The policy is fine-tuned for
    1,000,000 steps ... on the held-out task on the same robot ... under the same budget d"
    (transfer). The table names no checkpoint to start from.

    ``row`` is the parent's ledger row: a mapping, or a model with ``model_dump()`` such as ``LedgerRow``.
    Only a row with ``matched is True`` gets continuations: "An arm that cannot be matched is reported
    with its cost and left out of the battery" (Part 4.1 rule 6), and rule 7's sensitivity analysis
    needs no continuation. The pilot's runs are not continued (Part 3.6: "The pilot's runs are not reused").
    The spec is a pure function of the parent and the row, so queueing it again is a no-op.
    ValueError for any other parent, row or condition.

    Table 9.1 (Q-continuations), not Table 2.2: the continuation starts from the parent's matched
    checkpoint (``parent_step`` is the row's ``matched_checkpoint_step``, which Part 4.1 rule 1 picks from
    the run's "last ten checkpoints", so it must be a checkpoint the run saves: a step of the 200,000-step
    grid, the final step or, for a total off the grid, a step of ``pilot.contracts.end_relative_window``;
    that it lies among the last ten is not checked here), the checkpoint whose robustness gap the
    condition measures (Table 2.1 "Robustness gap"; eq. 1); every continuation trains with
    ``CONTINUATION_BASE_ALGO`` (PPO-Lagrangian), a PID-arm parent's included (Table 2.2 names no
    controller); and transfer starts a fresh multiplier at 0.001, the few-shot rule of Table 2.5.
    """
    if condition not in BATTERY_CONDITIONS:
        raise ValueError(f"condition must be one of {BATTERY_CONDITIONS}, got {condition!r}")
    if parent.study != "A" or parent.pilot or parent.group not in STUDY_A_PARENT_GROUPS:
        raise ValueError(f"{parent.run_id} is not a Study A run of the main study ({STUDY_A_PARENT_GROUPS})")
    data = dict(row.model_dump() if hasattr(row, "model_dump") else row)
    if data.get("run_id") != parent.run_id:
        raise ValueError(f"the ledger row of {data.get('run_id')!r} does not belong to {parent.run_id}")
    if data.get("matched") is not True:
        raise ValueError(
            f"{parent.run_id}: matched is {data.get('matched')!r}; only a matched arm enters the battery (Part 4.1 rule 6)"
        )
    if data.get("completed") is not True:
        raise ValueError(f"{parent.run_id}: the run did not complete")
    from pilot.contracts import end_relative_window  # deferred: pilot.contracts imports this module

    step = data.get("matched_checkpoint_step")
    if isinstance(step, bool) or not isinstance(step, int) or not 0 <= step <= parent.total_steps or not (
        step % R.CHECKPOINT_INTERVAL_STEPS == 0 or step == parent.total_steps
        or step in end_relative_window(parent.total_steps)  # saved for a total off the grid (Q-selection-window)
    ):
        raise ValueError(f"{parent.run_id}: matched_checkpoint_step {step!r} is not a checkpoint of the run (rule 1)")
    for key in ("matched_checkpoint_path", "commit_hash"):
        if not isinstance(data.get(key), str) or not data[key]:
            raise ValueError(f"{parent.run_id}: the ledger row has no {key}")
    finetune = condition == "finetune"
    params: dict[str, Any] = {
        "parent_run_id": parent.run_id,
        "parent_step": step,
        "parent_checkpoint": data["matched_checkpoint_path"],
        "parent_commit": data["commit_hash"],
        "condition": condition,
        "source_task": parent.task,
        "cost_limit": R.COST_LIMIT,
    }
    if finetune:
        params["multiplier"] = "frozen_zero"  # Table 2.2: "the multiplier frozen at zero"
    else:
        # Table 9.1 (Q-continuations), borrowed from Table 2.5's few-shot rule ("with a fresh multiplier at 0.001").
        params["fresh_multiplier_init"] = R.LAGRANGE_MULTIPLIER_INIT
    return _gated(RunSpec(
        run_id=f"{parent.arm_id}-{condition}-s{parent.seed}",
        study="A",
        task=parent.task if finetune else transfer_task(parent.task),
        arm=parent.arm,
        seed=parent.seed,
        total_steps=R.FINETUNE_STEPS if finetune else R.TRANSFER_STEPS,
        base_algo=CONTINUATION_BASE_ALGO,  # a PID parent's too: Table 9.1's, not Table 2.2's (see the constant)
        plugin=f"battery_{condition}",
        group=f"battery_{condition}",
        # N, shape, control, treatment and controller: the parent's, for bookkeeping only (the cut order
        # follows a continuation to its parent's arm). The continuation applies none of them: the
        # launcher's run facts (cfgs.pilot_cfgs) carry treatment and controller_variant None for it
        # (pilot.launch.pilot_config), and run_gate_keys gives it no Study A training key.
        N=parent.N,
        onset_step=None,
        onset_shape=parent.onset_shape,
        step_matching=parent.step_matching,
        treatment=parent.treatment,
        controller_variant=parent.controller_variant,
        depends_on=(parent.run_id,),
        params=params,
    ))


# ---------------------------------------------------------------------------
# The pilot (Part 3.6)
# ---------------------------------------------------------------------------


PILOT_PREFIX = "P-"
# Part 3.6 does not state the seed of the Study B pilot's unconstrained PPO run; Table 9.1 gives it the Moderate pilot
# run's seed (seed 0, in the re-pilot too), fixed before any data exist, so it is derived from that seed (an amendment
# of the Moderate seed carries over to it).
PILOT_UNCONSTRAINED_SEED = R.PILOT_MODERATE_SEED  # answered in Table 9.1 (Q-pilot-unconstrained-seed)


def pilot_prefix(revision: int = 0) -> str:
    """run_id prefix of the pilot (``P-``) and of the re-pilot after the revision (``P1-``)."""
    if isinstance(revision, bool) or not isinstance(revision, int) or revision not in (0, 1):
        raise ValueError("Part 6 permits one revision: revision is 0 (the pilot) or 1 (the re-pilot)")
    return PILOT_PREFIX if revision == 0 else f"P{revision}-"


def pilot(revision: int = 0) -> list[RunSpec]:
    """Part 3.6: six Study A runs and two Study B runs.

    ``revision=1`` gives the re-pilot of Part 6 ("a re-pilot of the same size" after the one
    permitted revision): the same runs under run_ids starting ``P1-``, so they never collide with
    the first pilot's records, and the go report reads each pilot separately.
    """
    prefix = pilot_prefix(revision)
    specs = [
        _study_a_spec(
            R.PILOT_STUDY_A_TASK, N, seed,
            shape=None if N == 0 else R.PILOT_STUDY_A_SHAPE,
            control=None if N == 0 else R.PILOT_STUDY_A_CONTROL,
            group="pilot", pilot=True,
        )
        for N in R.PILOT_STUDY_A_ONSET_FRACTIONS
        for seed in R.PILOT_SEEDS
    ]
    specs.append(_gated(RunSpec(
        run_id=f"{PILOT_PREFIX}B-unconstrained-s{PILOT_UNCONSTRAINED_SEED}",
        study="B",
        task=R.TASKS_STUDY_B[0],
        arm="unconstrained",
        seed=PILOT_UNCONSTRAINED_SEED,
        total_steps=R.TOTAL_STEPS,
        base_algo="PPO",
        plugin="unconstrained_ppo",
        group="pilot",
        pilot=True,
        pending=("Q-pilot-unconstrained-seed",),
        params={"purpose": "Part 3.6 / Part 6 G4: every Study B budget lies below the unconstrained cost"},
    )))
    specs.append(_study_b_spec("Moderate", R.PILOT_MODERATE_SEED, pilot=True))
    if revision:
        specs = [
            replace(s, run_id=prefix + s.run_id[len(PILOT_PREFIX):], params={**dict(s.params or {}), "pilot_revision": revision})
            for s in specs
        ]
    return specs


# ---------------------------------------------------------------------------
# Whole design
# ---------------------------------------------------------------------------


def design(name: str, seeds: Iterable[int] = R.SEEDS) -> list[RunSpec]:
    """Named subsets: pilot, repilot, main, treatment, controller, pid, study_a, study_b, study_b_fewshot, all.

    The pilot and the re-pilot have their registered seeds (Part 3.6); ``seeds`` other than the default is
    refused for them (ValueError).
    """
    seeds = tuple(seeds)
    if name in ("pilot", "repilot") and seeds != tuple(R.SEEDS):
        prose = {"pilot": "pilot", "repilot": "re-pilot"}[name]
        raise ValueError(f"the {prose} has its registered seeds {R.PILOT_SEEDS} (Part 3.6); seeds cannot be chosen")
    builders = {
        "pilot": lambda: pilot(),
        "repilot": lambda: pilot(revision=1),
        "main": lambda: study_a_main(seeds),
        "treatment": lambda: study_a_treatments(seeds),
        "controller": lambda: study_a_controllers(seeds),
        "pid": lambda: study_a_pid(seeds),
        "study_b": lambda: study_b(seeds),
        "study_b_fewshot": lambda: study_b_fewshot(study_b(seeds)),
    }
    composites = {
        "study_a": ("main", "treatment", "controller", "pid"),
        "all": ("main", "treatment", "controller", "pid", "study_b", "study_b_fewshot"),
    }
    if name not in builders and name not in composites:
        raise ValueError(f"unknown design {name!r}")
    specs = [s for part in composites.get(name, (name,)) for s in builders[part]()]
    ids = [s.run_id for s in specs]
    if len(ids) != len(set(ids)):
        raise AssertionError("duplicate run_id in design")
    return specs


# ---------------------------------------------------------------------------
# Cut order (Part 6.1; pilot.scheduler.CUTS_PATH)
# ---------------------------------------------------------------------------

CUTS = ("pid", "car", "fewshot_short", "treatments_controllers_n050", "drop_n010")
BATTERY_GROUPS = ("battery_finetune", "battery_transfer")
CAR_TASKS = tuple(t for t in R.TASKS_STUDY_A if short_task(t).startswith("Car"))  # SafetyCarGoal1-v0
# Part 6.1 "Never cut: ... on SafetyPointGoal1-v0 and SafetyPointButton1-v0": the Study A tasks but Car.
NEVER_CUT_TASKS = tuple(t for t in R.TASKS_STUDY_A if t not in CAR_TASKS)


def _parent_group(spec: RunSpec) -> str:
    """The Study A group a spec belongs to; a battery continuation belongs to its parent's."""
    if spec.group not in BATTERY_GROUPS:
        return spec.group
    if spec.treatment is not None:
        return "treatment"
    if spec.controller_variant == R.PID_VARIANT:
        return "pid"
    return "controller" if spec.controller_variant is not None else "main"


def _source_task(spec: RunSpec) -> str:
    """The Study A task a spec trains on; for a continuation, its parent's task."""
    return str(spec.params.get("source_task", spec.task)) if spec.group in BATTERY_GROUPS else spec.task


def _cut_pid(specs: list[RunSpec]) -> list[RunSpec]:
    """Cut 1: "Drop the PID check (15 runs)"."""
    return [s for s in specs if _parent_group(s) != "pid"]


def _cut_car(specs: list[RunSpec]) -> list[RunSpec]:
    """Cut 2: "Drop SafetyCarGoal1-v0 from every group of Study A"."""
    return [s for s in specs if not (s.study == "A" and _source_task(s) in CAR_TASKS)]


def _cut_fewshot_short(specs: list[RunSpec]) -> list[RunSpec]:
    """Cut 3: "Shorten Study B's few-shot continuations from 1,000,000 to 200,000 steps, keeping only
    the first horizon": each continuation trains to its first horizon and is read there only. The
    spec's ``total_steps`` and ``params["horizons"]`` change, and ``params["lr_schedule_steps"]``
    keeps the full continuation's schedule; nothing else changes. Applying it twice changes nothing.

    Table 9.1 (Q-continuations): a shortened few-shot continuation keeps the 1,000,000-step (50-epoch)
    actor schedule (``R.FEWSHOT_STEPS``; ``pilot.dependencies.restore_learner`` reads
    ``lr_schedule_steps``) and stops after 200,000 steps (10 epochs), where the rate is still 80
    percent of the initial one, as in the full continuation. Its only reading is therefore the full
    continuation's first-horizon reading, as Part 6.1's reason for the cut ("the three horizons are
    read from one continuation") requires.
    """
    first = R.FEWSHOT_HORIZONS[0]
    out = []
    for s in specs:
        if s.group == "study_b_fewshot":
            params = {**s.params, "horizons": [first], "lr_schedule_steps": R.FEWSHOT_STEPS}
            s = RunSpec.from_dict({**s.to_dict(), "total_steps": first, "params": params})
        out.append(s)
    return out


def _cut_treatments_controllers(specs: list[RunSpec]) -> list[RunSpec]:
    """Cut 4: "Restrict Study A's mediation treatments and controller variants to N = 0.50"."""
    late = max(R.LATE_ONSET_FRACTIONS)
    return [s for s in specs if not (_parent_group(s) in ("treatment", "controller") and s.N != late)]


def _cut_drop_n010(specs: list[RunSpec]) -> list[RunSpec]:
    """Cut 5: "Drop N = 0.10 from the main sweep"."""
    early = min(R.LATE_ONSET_FRACTIONS)
    return [s for s in specs if not (s.study == "A" and _parent_group(s) == "main" and s.N == early)]


_CUT_FUNCTIONS: dict[str, Callable[[list[RunSpec]], list[RunSpec]]] = {
    "pid": _cut_pid,
    "car": _cut_car,
    "fewshot_short": _cut_fewshot_short,
    "treatments_controllers_n050": _cut_treatments_controllers,
    "drop_n010": _cut_drop_n010,
}


def is_never_cut(spec: RunSpec) -> bool:
    """A run that no cut may remove or change (Part 6.1 "Never cut"): Study B's zero-shot arms (its
    training runs) and the main sweep's N = 0 and N = 0.50 arms on the tasks other than Car."""
    if spec.group == "study_b":
        return True
    late = max(R.LATE_ONSET_FRACTIONS)
    return spec.group == "main" and spec.task in NEVER_CUT_TASKS and spec.N in (R.ONSET_FRACTIONS[0], late)


def apply_cuts(specs: Iterable[RunSpec], cuts: Sequence[str]) -> list[RunSpec]:
    """The specs left after the cuts of Part 6.1, applied in the registered order.

    Part 6.1 "Cut order if G2 fails after the revision" (PDF p. 18): "Cuts are made in this order,
    each applied only if the previous is insufficient. 1. Drop the PID check (15 runs). 2. Drop
    SafetyCarGoal1-v0 from every group of Study A (65 main-sweep runs, 45 treatment runs, 30
    controller runs). 3. Shorten Study B's few-shot continuations from 1,000,000 to 200,000 steps,
    keeping only the first horizon; the three horizons are read from one continuation, so this, not
    dropping horizons, is what saves compute (about 11 run-equivalents). 4. Restrict Study A's
    mediation treatments and controller variants to N = 0.50. 5. Drop N = 0.10 from the main sweep
    (4 arms per remaining task, 20 runs per task). ... Never cut: the number of seeds; Study B's
    seven zero-shot arms; the N = 0 and N = 0.50 arms under both step-matching controls on
    SafetyPointGoal1-v0 and SafetyPointButton1-v0, which carry the primary comparison."

    ``cuts`` must be a prefix of CUTS ("each applied only if the previous is insufficient"). A
    battery continuation follows its parent's arm. The guard refuses (ValueError) any result that
    lost or changed a never-cut run, or lost some but not all seeds of an arm.
    """
    cuts = tuple(cuts)
    if cuts != CUTS[: len(cuts)]:
        raise ValueError(f"cuts {cuts} are not a prefix of the registered order {CUTS} (Part 6.1)")
    before = list(specs)
    after = list(before)
    for name in cuts:
        after = _CUT_FUNCTIONS[name](after)
    kept = {s.run_id: s for s in after}
    lost = [s.run_id for s in before if is_never_cut(s) and kept.get(s.run_id) != s]
    if lost:
        raise ValueError(f"cuts {cuts} would cut or change runs Part 6.1 never cuts: {lost[:5]}")
    seeds_before: dict[str, set[int]] = {}
    seeds_after: dict[str, set[int]] = {}
    for s in before:
        seeds_before.setdefault(s.arm_id, set()).add(s.seed)
    for s in after:
        seeds_after.setdefault(s.arm_id, set()).add(s.seed)
    thinned = sorted(a for a in seeds_after if seeds_after[a] != seeds_before.get(a))
    if thinned:
        raise ValueError(f"cuts {cuts} would change the seeds of arms {thinned[:5]} (Part 6.1: never the number of seeds)")
    return after

"""The registered design as a list of runs.

Every run of the study is one immutable ``RunSpec`` with a deterministic ``run_id``. The design
is generated from ``configs/registered.py`` only:

* Study A main sweep: Part 3.2, Table 3.2 (13 arms x 3 tasks x 5 seeds = 195 runs).
* Study A additional arms: Table 3.3 (treatments 135, controller variants 90, PID check 15).
* Study B: Part 3.3, Table 3.4 (7 arms x 5 seeds = 35 runs; 140 few-shot continuations).
* The pilot: Part 3.6 (8 runs).

Where the text leaves a value open (``configs.registered.PENDING``), the spec is still generated
so that it can be listed and costed, but it carries ``pending`` keys and the scheduler refuses to
launch it until the amendment log answers the question (First Tasks, habit 4).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from fractions import Fraction
from typing import Any, Iterable, Mapping

from configs import registered as R

# Algorithm plug-ins. The launcher imports "module:function" and calls it as
# function(env_id, cfgs, spec) -> omnisafe BaseAlgo. Owners are named so that a missing
# plug-in fails with a message saying who provides it.
PLUGINS: Mapping[str, tuple[str, str]] = {
    "ppolag": ("pilot.algorithms:make_ppolag", "Pilot owner (Role 1)"),
    "unconstrained_ppo": ("pilot.algorithms:make_unconstrained_ppo", "Pilot owner (Role 1)"),
    "study_a": ("envs.onset:make_algorithm", "Environment and tests (Role 2)"),
    "study_a_pid": ("envs.onset:make_pid_algorithm", "Environment and tests (Role 2)"),
    "study_b": ("studyb.conditioning:make_algorithm", "Study B and literature (Role 5)"),
    "study_b_fewshot": ("studyb.conditioning:make_fewshot_algorithm", "Study B and literature (Role 5)"),
}

BASE_ALGOS = ("PPOLag", "CPPOPID", "PPO")
GROUPS = (
    "pilot", "main", "treatment", "controller", "pid", "study_b", "study_b_fewshot", "determinism",
)


def short_task(task: str) -> str:
    """'SafetyPointGoal1-v0' -> 'PointGoal1'."""
    if not (task.startswith("Safety") and task.endswith("-v0")):
        raise ValueError(f"unexpected task id {task!r}")
    return task[len("Safety"):-len("-v0")]


_CONTROL_SHORT = {"total_steps": "total", "constrained_steps": "constrained"}


@dataclass(frozen=True)
class RunSpec:
    """One run: one training of one agent with one seed (How to Read This Document)."""

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
        return not self.open_questions

    def with_seed(self, seed: int) -> "RunSpec":
        """The same arm with another seed (replacement or surplus seed)."""
        data = self.to_dict()
        data["seed"] = seed
        data["run_id"] = f"{self.arm_id}-s{seed}"
        data["depends_on"] = [_reseed_dependency(dep, self.seed, seed) for dep in self.depends_on]
        if "parent_run_id" in data["params"]:
            data["params"]["parent_run_id"] = _reseed_dependency(data["params"]["parent_run_id"], self.seed, seed)
        return RunSpec.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["params"] = dict(self.params)
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "RunSpec":
        d = dict(data)
        for key in ("depends_on", "pending"):
            d[key] = tuple(d.get(key) or ())
        if d.get("training_levels") is not None:
            d["training_levels"] = tuple(float(x) for x in d["training_levels"])
        d["params"] = dict(d.get("params") or {})
        return RunSpec(**d)

    @staticmethod
    def from_json(text: str) -> "RunSpec":
        return RunSpec.from_dict(json.loads(text))


def _reseed_dependency(dep: str, old_seed: int, new_seed: int) -> str:
    suffix = f"-s{old_seed}"
    if not dep.endswith(suffix):
        raise ValueError(f"dependency {dep!r} does not carry seed {old_seed}")
    return dep[: -len(suffix)] + f"-s{new_seed}"


# ---------------------------------------------------------------------------
# Onset arithmetic (Table 2.1 "Onset fraction N"; Table 2.4 last two rows)
# ---------------------------------------------------------------------------


def _fraction(n: float) -> Fraction:
    return Fraction(str(n))


def onset_schedule(N: float, control: str) -> tuple[int, int, tuple[str, ...]]:
    """Return (onset_step, total_steps, pending) for onset fraction N under a step-matching control.

    Total-steps matched: every arm trains T steps; onset at N*T (Table 2.4).
    Constrained-steps matched: every arm trains T steps under the constraint; total T/(1-N)
    (Table 2.4). For N = 0.10 and 0.25 this is not a whole number of epochs; the proposed rule
    (First Tasks, Part I Section 5) rounds the onset to the nearest whole epoch and keeps exactly
    T constrained steps after it. Such specs always carry ``pending = ('Q-rounding',)``; the
    scheduler launches them once the key is in ``configs.registered.ANSWERED_QUESTIONS``.
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
        rounded_epochs = int(epochs + Fraction(1, 2))  # nearest whole epoch; exact halves do not occur here
        if epochs - int(epochs) == Fraction(1, 2):
            raise ValueError("onset lies exactly between two epochs; the proposed rounding is undefined")
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
    return RunSpec(
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
    )


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
    """Table 3.3 'Mediation treatments': abrupt, total-steps matched, N in {0.10, 0.25, 0.50}."""
    return [
        _study_a_spec(task, N, seed, shape="abrupt", control="total_steps", group="treatment", treatment=t)
        for task in R.TASKS_STUDY_A
        for N in R.LATE_ONSET_FRACTIONS
        for t in R.TREATMENTS
        for seed in seeds
    ]


def study_a_controllers(seeds: Iterable[int] = R.SEEDS) -> list[RunSpec]:
    """Table 3.3 'Controller variants': warm-started and rate-limited, abrupt, total-steps matched."""
    return [
        _study_a_spec(task, N, seed, shape="abrupt", control="total_steps", group="controller", controller=c)
        for task in R.TASKS_STUDY_A
        for N in R.LATE_ONSET_FRACTIONS
        for c in R.CONTROLLER_VARIANTS
        for seed in seeds
    ]


def study_a_pid(seeds: Iterable[int] = R.SEEDS) -> list[RunSpec]:
    """Table 3.3 'PID check': SafetyPointGoal1-v0 only, abrupt, total-steps matched."""
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
    return RunSpec(
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
    )


def is_primary_comparison(spec: RunSpec) -> bool:
    """Part 5.5: an arm that receives surplus seeds, under the proposed reading of Q-surplus-arm-set.

    "the N = 0 and N = 0.50 arms under both step-matching controls on SafetyPointGoal1-v0, and the
    seven arms of Study B". The onset shape is not named; both shapes are included (proposal).
    """
    late_n = max(R.LATE_ONSET_FRACTIONS)
    if spec.group == "main":
        return spec.task == R.PRIMARY_TASK and spec.N in (R.ONSET_FRACTIONS[0], late_n)
    return spec.group == "study_b"


def surplus_spec(base: RunSpec, seed: int) -> RunSpec:
    """A surplus seed of a primary-comparison arm (Part 5.5).

    Arms that only the proposed reading of Q-surplus-arm-set adds (the ramp arms at N = 0.50; Part
    5.5 names no onset shape) carry the key in ``pending``, so the scheduler holds them until the
    amendment log answers it.
    """
    if not is_primary_comparison(base):
        raise ValueError(f"{base.run_id} is not in the primary-comparison set of Part 5.5")
    spec = base.with_seed(seed)
    if spec.onset_shape == "ramp" and "Q-surplus-arm-set" not in spec.pending:
        spec = replace(spec, pending=(*spec.pending, "Q-surplus-arm-set"))
    return spec


def study_b(seeds: Iterable[int] = R.SEEDS) -> list[RunSpec]:
    return [_study_b_spec(arm, seed) for arm in R.STUDY_B_ARMS for seed in seeds]


def study_b_fewshot(parents: Iterable[RunSpec]) -> list[RunSpec]:
    """Part 3.3: each finished run is continued once under each unseen budget for 1,000,000 steps."""
    specs = []
    for parent in parents:
        if parent.group != "study_b":
            raise ValueError(f"{parent.run_id} is not a Study B training run")
        for budget in R.UNSEEN_BUDGETS:
            specs.append(RunSpec(
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
                    "budget": budget,
                    "horizons": list(R.FEWSHOT_HORIZONS),
                    "fresh_multiplier_init": R.LAGRANGE_MULTIPLIER_INIT,
                },
            ))
    return specs


# ---------------------------------------------------------------------------
# The pilot (Part 3.6)
# ---------------------------------------------------------------------------


PILOT_PREFIX = "P-"


def pilot_prefix(revision: int = 0) -> str:
    """run_id prefix of the pilot (``P-``) and of the re-pilot after the revision (``P1-``)."""
    if revision not in (0, 1):
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
    specs.append(RunSpec(
        run_id="P-B-unconstrained-s0",
        study="B",
        task=R.TASKS_STUDY_B[0],
        arm="unconstrained",
        seed=0,
        total_steps=R.TOTAL_STEPS,
        base_algo="PPO",
        plugin="unconstrained_ppo",
        group="pilot",
        pilot=True,
        pending=("Q-pilot-unconstrained-seed",),
        params={"purpose": "Part 3.6 / Part 6 G4: every Study B budget lies below the unconstrained cost"},
    ))
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
    """Named subsets: pilot, repilot, main, treatment, controller, pid, study_a, study_b, study_b_fewshot, all."""
    seeds = tuple(seeds)
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
    if name == "study_a":
        return design("main", seeds) + design("treatment", seeds) + design("controller", seeds) + design("pid", seeds)
    if name == "all":
        return design("study_a", seeds) + design("study_b", seeds) + design("study_b_fewshot", seeds)
    if name not in builders:
        raise ValueError(f"unknown design {name!r}")
    specs = builders[name]()
    ids = [s.run_id for s in specs]
    if len(ids) != len(set(ids)):
        raise AssertionError("duplicate run_id in design")
    return specs

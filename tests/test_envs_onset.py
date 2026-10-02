"""Role 2's Study A plug-in ``envs.onset`` without training (fast).

The tiny training runs are in ``tests/test_envs_onset_slow.py``.

* Equation (5) and the rate limit of Table 2.4 as pure functions.
* ``build_onset_cfgs`` accepts every Study A spec the manifest generates (the design and both pilots)
  with the registered values, and refuses (``RunRefused``, never another exception) every spec that
  departs from the registered design; the copies scripts/smoke_run.py really queues (``short``,
  ``design_specs``) and the determinism check's ``pilot.scheduler.determinism_spec``
  pass unregistered, through the factories; any other ``DET-`` spec is refused.
* The PID check's total-steps matching is Table 9.1's (Q-pid-eq9; Table 3.3's PID row names no
  control); values Table 9.1 decides are named constants under their key.
* The data control's learning-rate schedule (Table 9.1, Q-data-control-lr): ``DataControlLR`` gives the
  untreated late arm's rates bit for bit for T/E epochs, then the restart at (1 − N)·lr decaying to 0;
  only the data control carries ``onset_cfgs['actor_lr_schedule']``; the factory swaps OmniSafe's
  schedule before the full-state saver captures it, and its state loads with ``weights_only=True``.
* The warm-start source (Table 2.4; Q-warm-start) read from a fixture N = 0 run directory.
* The surrogate before and after onset on synthetic advantages (First Tasks, Role 2 test 2), the
  rate-limited multiplier object, and the configuration checks of the algorithm.
* The factories build the registered arms (the environment and the networks, no training), put
  their settings in ``cfgs['onset_cfgs']`` before construction, and refuse every configuration
  problem before anything is built.
* Every quotation of the pre-registration in ``envs/__init__.py``, ``envs/onset.py`` and
  ``envs/continuations.py`` is verbatim (the checker of ``tests/test_core_quotes.py``).

Everything here imports OmniSafe (``envs.onset`` subclasses its algorithms), hence the marker.
"""

from __future__ import annotations

import csv
import functools
import importlib.util
import json
import math
import re
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import yaml

from configs import registered as R
from pilot import dependencies, manifest, provenance
from pilot.errors import RunRefused
from pilot.manifest import RunSpec

pytestmark = pytest.mark.omnisafe

torch = pytest.importorskip("torch")
pytest.importorskip("omnisafe")

from envs import onset  # noqa: E402  # after the skip: it imports OmniSafe

REPO = Path(__file__).resolve().parents[1]
E = R.STEPS_PER_EPOCH
COMMIT = "c" * 40


def _study_a_specs() -> list[RunSpec]:
    """Every Study A spec the manifest generates: the design, the pilot and the re-pilot."""
    pilots = [s for rev in (0, 1) for s in manifest.pilot(rev) if s.study == "A"]
    return manifest.design("study_a") + pilots


def _spec(group: str = "main", **match) -> RunSpec:
    """The first seed-0 PointGoal1 spec of ``group`` with the given field values."""
    specs = manifest.pilot() if group == "pilot" else manifest.design(group)
    match.setdefault("task", R.PRIMARY_TASK)
    match.setdefault("seed", 0)
    return next(s for s in specs if all(getattr(s, k) == v for k, v in match.items()))


def _with(spec: RunSpec, **changes) -> RunSpec:
    data = spec.to_dict()
    params = changes.pop("params", None)
    if params is not None:
        data["params"] = params
    data.update(changes)
    return RunSpec.from_dict(data)


@functools.lru_cache(maxsize=None)
def _smoke_run():
    """scripts/smoke_run.py itself, loaded as a module.

    Loaded as tests/test_integration_sched_scripts.py loads it (it is a script), so these tests use
    the smoke copies the smoke run really queues, never a re-implementation of them."""
    module_spec = importlib.util.spec_from_file_location("_envs_smoke_run", REPO / "scripts" / "smoke_run.py")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def _smoke(spec: RunSpec, epochs: int) -> RunSpec:
    """scripts/smoke_run.py ``short``: a SMOKE- copy of ``epochs`` epochs without dependencies.

    A late onset later than half the run moves to the last checkpoint at or before half the run, or
    to half the run in whole epochs under 20 epochs (0 in a 1-epoch run)."""
    return _smoke_run().short(spec, epochs)


# ---------------------------------------------------------------------------
# Equation (5) and the rate limit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task,midpoint", [("SafetyPointGoal1-v0", 68.0), ("SafetyCarGoal1-v0", 70.5),
                                           ("SafetyPointButton1-v0", 165.0)])
def test_the_ramp_falls_from_the_loose_budget_to_d_over_the_window(task, midpoint) -> None:
    """First Tasks (Role 2): "The ramp returns the loose budget at onset, 25 at onset plus W and after, and
    the value of equation (5) halfway"; evaluated at the first step of each epoch (Q-ramp-step)."""
    d_loose, W, d = R.D_LOOSE[task], R.RAMP_WINDOW_STEPS, R.COST_LIMIT
    for onset_step in (1_000_000, 2_500_000, 3_340_000, 5_000_000):
        budget = lambda t: onset.effective_budget(t, onset_step=onset_step, cost_limit=d, d_loose=d_loose,  # noqa: E731
                                                  window_steps=W)
        assert budget(onset_step) == d_loose
        assert budget(onset_step + W // 2) == midpoint == (d_loose + d) / 2
        assert budget(onset_step + W - E) == pytest.approx(d + (d_loose - d) / (W // E))  # the last ramp epoch
        assert budget(onset_step + W) == d and budget(onset_step + 3 * W) == d
        steps = range(onset_step, onset_step + W + 2 * E, E)
        values = [budget(t) for t in steps]
        assert all(a > b for a, b in zip(values[: W // E], values[1: W // E + 1]))  # strictly falling in the window
        assert values[W // E:] == [d, d]
    assert onset.effective_budget(123 * E, onset_step=E, cost_limit=d) == d  # abrupt: d at once
    with pytest.raises(ValueError, match="before the onset"):
        onset.effective_budget(0, onset_step=E, cost_limit=d, d_loose=d_loose, window_steps=W)
    with pytest.raises(ValueError, match="positive window"):
        onset.effective_budget(E, onset_step=E, cost_limit=d, d_loose=d_loose, window_steps=0)
    with pytest.raises(ValueError, match="no ramp window"):
        onset.effective_budget(E, onset_step=E, cost_limit=d, window_steps=W)


def test_the_rate_limit_clips_each_change_to_five_percent_plus_a_hundredth() -> None:
    """Table 2.4 "Rate-limited multiplier" with the registered 0.05 and 0.01 (Table 8.2)."""
    limit = lambda old, new, **kw: onset.rate_limit(old, new, relative=R.RATE_LIMIT_RELATIVE,  # noqa: E731
                                                    absolute=R.RATE_LIMIT_ABSOLUTE, **kw)
    assert limit(0.001, 0.036) == pytest.approx(0.01105, abs=1e-15)  # Adam's 0.035 step, clipped
    assert limit(1.0, 1.035) == 1.035  # within 0.06: not binding
    assert limit(1.0, 0.9) == pytest.approx(0.94)
    assert limit(0.005, 0.0) == 0.0  # inside the band [-0.00525, 0.01525]
    assert limit(0.005, -0.004) == 0.0  # the band reaches below zero: clamped at 0
    assert limit(0.0, 0.5) == pytest.approx(0.01)
    assert limit(10.0, 20.0, upper=10.2) == 10.2
    assert limit(2.0, 2.0) == 2.0
    for old in (0.0, 0.001, 0.3, 4.0):
        for new in (0.0, old / 2, old + 0.035, 3 * old + 1):
            got = limit(old, new)
            assert got >= 0.0 and abs(got - old) <= R.RATE_LIMIT_RELATIVE * old + R.RATE_LIMIT_ABSOLUTE + 1e-12


def test_the_rate_limited_multiplier_clips_after_omnisafes_adam_step() -> None:
    lagrange = onset.RateLimitedLagrange(rate_limit_relative=R.RATE_LIMIT_RELATIVE,
                                         rate_limit_absolute=R.RATE_LIMIT_ABSOLUTE,
                                         cost_limit=R.COST_LIMIT, lagrangian_multiplier_init=R.LAGRANGE_MULTIPLIER_INIT,
                                         lambda_lr=R.LAGRANGE_MULTIPLIER_LR, lambda_optimizer="Adam")
    assert lagrange.last_proposed is None and len(lagrange.lambda_optimizer.state) == 0
    old = lagrange.lagrangian_multiplier.item()
    lagrange.update_lagrange_multiplier(150.0)
    assert lagrange.last_proposed == pytest.approx(old + R.LAGRANGE_MULTIPLIER_LR, rel=1e-4)  # Adam's first step
    new = lagrange.lagrangian_multiplier.item()
    assert new == float(np.float32(onset.rate_limit(old, lagrange.last_proposed, relative=R.RATE_LIMIT_RELATIVE,
                                                    absolute=R.RATE_LIMIT_ABSOLUTE)))
    assert new == pytest.approx(old + R.RATE_LIMIT_RELATIVE * old + R.RATE_LIMIT_ABSOLUTE, rel=1e-6)
    assert new < lagrange.last_proposed  # binding
    trace = [new]
    for cost in (150.0, 150.0, 0.0, 0.0, 0.0):
        lagrange.update_lagrange_multiplier(cost)
        trace.append(lagrange.lagrangian_multiplier.item())
    for a, b in zip(trace, trace[1:]):
        assert b >= 0.0 and abs(b - a) <= R.RATE_LIMIT_RELATIVE * a + R.RATE_LIMIT_ABSOLUTE + 1e-7
    assert int(lagrange.lambda_optimizer.state[lagrange.lagrangian_multiplier]["step"]) == 6  # Adam keeps its moments


# ---------------------------------------------------------------------------
# build_onset_cfgs
# ---------------------------------------------------------------------------


def test_every_study_a_spec_of_the_manifest_passes_with_the_registered_values() -> None:
    specs = _study_a_specs()
    assert len(specs) == (R.MAIN_SWEEP_RUNS + R.TREATMENT_RUNS + R.CONTROLLER_RUNS + R.PID_RUNS
                          + 2 * len(R.PILOT_STUDY_A_ONSET_FRACTIONS) * len(R.PILOT_SEEDS))  # pilot and re-pilot
    onset_epochs: dict[tuple, set[int]] = {}
    for spec in specs:
        oc = onset.build_onset_cfgs(spec, steps_per_epoch=E)
        json.dumps(oc, allow_nan=False)  # plain JSON: config.json and the configuration hash
        assert oc["registered"] is True and oc["warm_start"] is None and oc["steps_per_epoch"] == E
        assert oc["onset_step"] == spec.onset_step and oc["onset_epoch"] == spec.onset_step // E
        assert oc["N"] == spec.N and oc["cost_limit"] == R.COST_LIMIT
        assert (oc["shape"], oc["step_matching"], oc["treatment"], oc["controller_variant"]) == (
            spec.onset_shape, spec.step_matching, spec.treatment, spec.controller_variant)
        ramp = spec.onset_shape == "ramp"
        assert (oc["d_loose"], oc["ramp_window_steps"]) == ((R.D_LOOSE[spec.task], R.RAMP_WINDOW_STEPS) if ramp
                                                            else (None, None))
        limited = spec.controller_variant == "rate_limited"
        assert (oc["rate_limit_relative"], oc["rate_limit_absolute"]) == (
            (R.RATE_LIMIT_RELATIVE, R.RATE_LIMIT_ABSOLUTE) if limited else (None, None))
        assert oc["multiplier_update"] == ("CPPOPID" if spec.controller_variant == R.PID_VARIANT else "PPOLag")
        if spec.treatment in ("reset", "injection"):
            assert oc["intervention"]["function"] == {"reset": "partial_reset", "injection": "plasticity_injection"}[
                spec.treatment] and oc["intervention"]["depth"] == R.INTERVENTION_LAYERS
        else:
            assert oc["intervention"] is None
        if spec.treatment == "additional_constrained":
            assert oc["additional_constrained_steps"] == round(spec.N * R.TOTAL_STEPS)
            assert spec.total_steps == R.TOTAL_STEPS + oc["additional_constrained_steps"]
            assert oc[onset.SCHEDULE_KEY] == {"decay_epochs": R.EPOCHS, "tail_epochs": oc["additional_constrained_steps"] // E,
                                              "tail_start_factor": float(1 - Fraction(str(spec.N)))}
        else:
            assert onset.SCHEDULE_KEY not in oc  # Q-data-control-lr changes no other arm's configuration
        onset_epochs.setdefault((spec.N, spec.step_matching), set()).add(oc["onset_epoch"])
    # the onset epochs of the design (pilot.manifest.onset_schedule): constrained-steps arms rounded (Q-rounding)
    assert onset_epochs == {(0.0, None): {0}, (0.1, "total_steps"): {50}, (0.25, "total_steps"): {125},
                            (0.5, "total_steps"): {250}, (0.1, "constrained_steps"): {56},
                            (0.25, "constrained_steps"): {167}, (0.5, "constrained_steps"): {500}}


def test_replacement_and_surplus_seeds_pass_too() -> None:
    warm = _spec("controller", controller_variant="warm_started", N=0.25)
    for seed in (5, 11):
        spec = warm.with_seed(seed)
        assert onset.build_onset_cfgs(spec, steps_per_epoch=E)["controller_variant"] == "warm_started"
    surplus = manifest.surplus_spec(_spec("main", N=0.5, onset_shape="ramp", step_matching="total_steps"), 6)
    assert onset.build_onset_cfgs(surplus, steps_per_epoch=E)["d_loose"] == R.D_LOOSE[R.PRIMARY_TASK]


def _refusal_cases() -> list[tuple[str, RunSpec]]:
    ramp = _spec("main", N=0.5, onset_shape="ramp", step_matching="total_steps")
    abrupt = _spec("main", N=0.5, onset_shape="abrupt", step_matching="total_steps")
    n0 = _spec("main", N=0.0)
    reset = _spec("treatment", treatment="reset", N=0.25)
    data = _spec("treatment", treatment="additional_constrained", N=0.5)
    rate = _spec("controller", controller_variant="rate_limited", N=0.5)
    warm = _spec("controller", controller_variant="warm_started", N=0.5)
    pid = _spec("pid", N=0.5)
    car = next(s for s in manifest.design("main") if s.task == "SafetyCarGoal1-v0" and s.N == 0.5
               and s.onset_shape == "abrupt" and s.step_matching == "total_steps" and s.seed == 0)
    return [
        ("d_loose 110", _with(ramp, params={**ramp.params, "d_loose": 110.0})),
        ("ramp window 900,000", _with(ramp, params={**ramp.params, "ramp_window_steps": 900_000})),
        ("cost limit 30", _with(abrupt, params={**abrupt.params, "cost_limit": 30.0})),
        ("cost limit missing", _with(abrupt, params={})),
        ("cost limit True", _with(abrupt, params={"cost_limit": True})),
        ("ramp parameters on an abrupt arm", _with(abrupt, params={**ramp.params})),
        ("rate 0.06", _with(rate, params={**rate.params, "rate_limit_relative": 0.06})),
        ("rate absolute 0.02", _with(rate, params={**rate.params, "rate_limit_absolute": 0.02})),
        ("rate limits on an abrupt arm", _with(abrupt, params={**abrupt.params, **rate.params})),
        ("onset one epoch late", _with(abrupt, onset_step=abrupt.onset_step + E)),
        ("onset one epoch early", _with(abrupt, onset_step=abrupt.onset_step - E)),
        ("total one epoch short", _with(abrupt, total_steps=abrupt.total_steps - E)),
        ("treatment with a ramp", _with(reset, onset_shape="ramp", params={**reset.params, "d_loose": R.D_LOOSE[R.PRIMARY_TASK],
                                                                            "ramp_window_steps": R.RAMP_WINDOW_STEPS})),
        ("treatment under constrained-steps matching", _with(reset, step_matching="constrained_steps")),
        ("treatment and controller", _with(reset, controller_variant="warm_started")),
        ("treatment on N = 0", _with(n0, treatment="reset")),
        ("unknown treatment", _with(reset, treatment="shrink_perturb")),
        ("unknown controller", _with(rate, controller_variant="clipped")),
        ("N = 0 with a shape", _with(n0, onset_shape="abrupt")),
        ("N = 0 with an onset", _with(n0, onset_step=E)),
        ("N not registered", _with(abrupt, N=0.3)),
        ("N missing", _with(abrupt, N=None)),
        ("late arm without onset", _with(abrupt, onset_step=None)),
        ("late arm without shape", _with(abrupt, onset_shape=None)),
        ("unknown shape", _with(abrupt, onset_shape="step")),
        ("unknown control", _with(abrupt, step_matching="both")),
        ("PID on CarGoal1", _with(car, run_id=car.run_id.replace("-s0", "-pid-s0"), group="pid",
                                  controller_variant="pid", base_algo="CPPOPID", plugin="study_a_pid")),
        ("PID on PPOLag", _with(pid, base_algo="PPOLag", plugin="study_a")),
        ("PPOLag arm on the PID plug-in", _with(abrupt, base_algo="CPPOPID", plugin="study_a_pid")),
        ("a Study B spec", manifest.design("study_b")[0]),
        ("a task outside Study A", _with(abrupt, task="SafetyCarButton1-v0",
                                         run_id="A-CarButton1-N0.50-abrupt-total-s0")),
        ("group of another arm", _with(reset, group="main")),
        ("pilot flag on a main run", _with(abrupt, pilot=True)),
        ("arm label of another arm", _with(abrupt, arm="N0.50-ramp-total")),
        ("run_id of another arm", _with(abrupt, run_id="A-PointGoal1-N0.50-ramp-total-s0")),
        ("unknown parameter", _with(abrupt, params={**abrupt.params, "lr": 0.1})),
        ("data control steps off", _with(data, params={**data.params, "additional_constrained_steps": 4_000_000})),
        ("data control total T", _with(data, total_steps=R.TOTAL_STEPS)),
        ("data control steps on another arm", _with(abrupt, params={**abrupt.params,
                                                                    "additional_constrained_steps": 5_000_000})),
        ("warm start without its N = 0 run", _with(warm, depends_on=())),
        ("warm start from another seed", _with(warm, depends_on=("A-PointGoal1-N0.00-s1",))),
        ("warm start from a pilot run", _with(warm, depends_on=("P-A-PointGoal1-N0.00-s0",))),
        ("dependency without a warm start", _with(abrupt, depends_on=("A-PointGoal1-N0.00-s0",))),
        ("determinism group as registered", _with(n0, group="determinism")),
    ]


@pytest.mark.parametrize("case,spec", _refusal_cases(), ids=[c for c, _ in _refusal_cases()])
def test_a_spec_off_the_registered_design_is_refused(case, spec) -> None:
    with pytest.raises(RunRefused):
        onset.build_onset_cfgs(spec, steps_per_epoch=E)


def test_the_registered_epoch_length_is_required_unless_unregistered() -> None:
    spec = _spec("pilot", N=0.5)
    with pytest.raises(RunRefused, match=f"{E} steps per epoch"):
        onset.build_onset_cfgs(spec, steps_per_epoch=2_000)
    oc = onset.build_onset_cfgs(spec, steps_per_epoch=2_000, registered=False)
    assert (oc["onset_epoch"], oc["registered"]) == (spec.onset_step // 2_000, False)
    for bad in (0, -E, 2_000.0, True, "20000"):
        with pytest.raises(RunRefused):
            onset.build_onset_cfgs(spec, steps_per_epoch=bad)


def test_a_mapping_is_read_as_a_spec_and_a_bad_one_refused() -> None:
    spec = _spec("pilot", N=0.5)
    assert onset.build_onset_cfgs(spec.to_dict(), steps_per_epoch=E) == onset.build_onset_cfgs(spec, steps_per_epoch=E)
    not_a_string = {**spec.to_dict(), "run_id": 5}  # RunSpec raises AttributeError for it: refused, never a crash
    for bad in ({**spec.to_dict(), "total_steps": 123}, {"run_id": "x"}, not_a_string, "A-PointGoal1-N0.50", None):
        with pytest.raises(RunRefused):
            onset.build_onset_cfgs(bad, steps_per_epoch=E)
    for check in (onset.is_registered_spec, onset.is_determinism_spec):
        with pytest.raises(RunRefused, match="not a valid run spec"):
            check(not_a_string)


def test_smoke_copies_are_accepted_as_unregistered() -> None:
    """The copies scripts/smoke_run.py ``short`` really makes: a late onset moves inside the short run."""
    pilot_late = _spec("pilot", N=0.5)
    short = _smoke(pilot_late, 2)  # half the run in whole epochs: the switch happens in the smoke test
    assert short.onset_step == E and not onset.is_registered_spec(short)
    oc = onset.build_onset_cfgs(short, steps_per_epoch=E, registered=False)
    assert (oc["onset_step"], oc["onset_epoch"], oc["shape"], oc["registered"]) == (E, 1, "abrupt", False)
    with pytest.raises(RunRefused):  # the same copy is not the registered design
        onset.build_onset_cfgs(short, steps_per_epoch=E)
    one = _smoke(pilot_late, 1)  # no room for an onset: the arm trains from step 0, as N = 0 does
    assert onset.build_onset_cfgs(one, steps_per_epoch=E, registered=False)["onset_epoch"] == 0
    long = _smoke(pilot_late, 30)
    assert onset.build_onset_cfgs(long, steps_per_epoch=E, registered=False)["onset_step"] == R.CHECKPOINT_INTERVAL_STEPS
    for spec in _study_a_specs():
        for epochs in (1, 2, 3):
            smoke_copy = _smoke(spec, epochs)
            acts_at_onset = spec.treatment in onset.INTERVENTION_FUNCTIONS and not smoke_copy.onset_step
            if spec.controller_variant == "warm_started" or acts_at_onset:
                # a smoke copy has no dependency, which a warm start reads; reset and injection act "At onset"
                with pytest.raises(RunRefused):
                    onset.build_onset_cfgs(smoke_copy, steps_per_epoch=E, registered=False)
                continue
            oc = onset.build_onset_cfgs(smoke_copy, steps_per_epoch=E, registered=False)
            assert oc["onset_step"] == smoke_copy.onset_step == (min(spec.onset_step, epochs // 2 * E) if spec.N else 0)
            if spec.treatment == "additional_constrained":  # the schedule spans the shrunk run (actor_lr_schedule)
                schedule = oc[onset.SCHEDULE_KEY]
                assert schedule["decay_epochs"] >= 1 and schedule["decay_epochs"] + schedule["tail_epochs"] == epochs
    assert onset.is_registered_spec(pilot_late) and onset.is_registered_spec(_spec("pid", N=0.1))
    assert not onset.is_determinism_spec(pilot_late) and not onset.is_determinism_spec(short)


class _Built:
    """Stands in for OnsetPPOLag / OnsetCPPOPID: the factory's every check runs, nothing is constructed."""

    def __init__(self, env_id, cfgs):
        self.env_id, self.cfgs = env_id, cfgs


@pytest.mark.parametrize("design", ["pilot", "sample"])
def test_every_study_a_run_of_the_smoke_designs_passes_its_factory(tmp_path, monkeypatch, design) -> None:
    """scripts/smoke_run.py ``design_specs`` (1 to 3 epochs, --allow-pending) through the launcher's configuration
    and the plug-in's factory; refused only where a reset or injection has no onset after step 0 (1 epoch)."""
    from pilot import launch
    from pilot.algorithms import load_plugin

    monkeypatch.setattr(onset, "OnsetPPOLag", _Built)
    monkeypatch.setattr(onset, "OnsetCPPOPID", _Built)
    seen = 0
    for epochs in (1, 2, 3):
        for spec in _smoke_run().design_specs(design, epochs):
            if spec.plugin not in ("study_a", "study_a_pid"):
                continue
            seen += 1
            cfgs = launch.build_config(spec, tmp_path / spec.run_id / "omnisafe")
            factory = load_plugin(spec.plugin)
            if spec.treatment in onset.INTERVENTION_FUNCTIONS and epochs == 1:
                with pytest.raises(RunRefused, match="acts at onset"):
                    factory(spec.task, cfgs, spec)
                continue
            built = factory(spec.task, cfgs, spec)
            assert isinstance(built, _Built) and built.env_id == spec.task
            oc = built.cfgs.onset_cfgs
            assert (oc.registered, oc.onset_step, oc.onset_epoch) == (False, spec.onset_step, spec.onset_step // E)
            assert oc.multiplier_update == ("CPPOPID" if spec.plugin == "study_a_pid" else "PPOLag")
    assert seen >= 9  # the loop is not vacuous: at least 9 Study A runs over the three run lengths


def test_the_two_epoch_sample_smoke_runs_are_built(tmp_path, one_thread) -> None:
    """The sample design's Study A runs at 2 epochs, built for real (no training): N = 0, the reset and the PID
    check, the last two with their onset at epoch 1."""
    from pilot import launch
    from pilot.algorithms import load_plugin

    specs = [s for s in _smoke_run().design_specs("sample", 2) if s.plugin in ("study_a", "study_a_pid")]
    assert [(s.treatment, s.controller_variant, s.onset_step) for s in specs] == [
        (None, None, 0), ("reset", None, E), (None, R.PID_VARIANT, E)]
    for spec in specs:
        algo = load_plugin(spec.plugin)(spec.task, launch.build_config(spec, tmp_path / spec.run_id / "omnisafe"), spec)
        assert algo._onset_epoch == spec.onset_step // E and algo._onset["registered"] is False
        assert (algo._fixed_batch is not None) == (spec.treatment == "reset")
        algo._env.close()


@pytest.mark.parametrize("plugin", ["study_a", "study_a_pid"])
def test_the_determinism_checks_run_is_built_as_unregistered(tmp_path, one_thread, plugin) -> None:
    """``scripts/determinism_check.py --plugin study_a|study_a_pid`` trains
    ``pilot.scheduler.determinism_spec`` (the arm's copy: DET- run_id, the arm's group, the check's total and a
    late onset scaled into it) through the launcher; the scheduler holds every registered run of the plug-in
    until a passing report of exactly this spec is committed, so the factory must build it."""
    from pilot import launch
    from pilot.scheduler import determinism_spec

    factory = onset.make_pid_algorithm if plugin == "study_a_pid" else onset.make_algorithm
    for total in (R.CHECKPOINT_INTERVAL_STEPS, 2 * E):  # the registered form and the default form
        spec = determinism_spec(plugin, total)
        assert spec.run_id.startswith(onset.DETERMINISM_PREFIX) and spec.group in onset.STUDY_A_GROUPS
        assert onset.is_determinism_spec(spec) and not onset.is_registered_spec(spec)
        assert onset.is_determinism_spec(spec.to_dict())
        oc = onset.build_onset_cfgs(spec, steps_per_epoch=E, registered=False)
        assert oc["onset_step"] == spec.onset_step and oc["registered"] is False
        with pytest.raises(RunRefused):  # its total and onset are not the registered design's
            onset.build_onset_cfgs(spec, steps_per_epoch=E)
        if total != R.CHECKPOINT_INTERVAL_STEPS:
            continue
        cfgs = launch.build_config(spec, tmp_path / spec.run_id / "omnisafe")
        algo = factory(spec.task, cfgs, spec)
        assert type(algo).__name__ == ("OnsetCPPOPID" if plugin == "study_a_pid" else "OnsetPPOLag")
        assert algo._onset_epoch == spec.onset_step // E
        # only the PID check's copy has a late onset; study_a's is the N = 0 arm
        assert (0 < spec.onset_step < spec.total_steps) == (plugin == "study_a_pid")
        assert algo._onset["registered"] is False and cfgs.pilot_cfgs.run_id == spec.run_id
        algo._env.close()


@pytest.mark.parametrize("change", ["seed", "onset", "group determinism", "N", "task", "other plugin"])
def test_a_det_spec_that_is_not_the_schedulers_determinism_run_is_refused(tmp_path, change) -> None:
    from pilot import launch
    from pilot.scheduler import determinism_spec

    pid = change in ("onset", "other plugin")
    spec = determinism_spec("study_a_pid" if pid else "study_a", R.CHECKPOINT_INTERVAL_STEPS)
    if change == "seed":
        spec = replace(spec, seed=1, run_id=spec.run_id.removesuffix("-s0") + "-s1")  # the check runs seed 0 only
    elif change == "onset":
        spec = replace(spec, onset_step=spec.onset_step - E)
    elif change == "group determinism":  # a plausible group the scheduler never gives its determinism run
        spec = replace(spec, group="determinism")
    elif change == "N":
        spec = replace(spec, N=0.1, onset_shape="abrupt", step_matching="total_steps", onset_step=E)
    elif change == "task":
        spec = replace(spec, task="SafetyCarGoal1-v0")
    elif change == "other plugin":
        spec = replace(spec, plugin="study_a", base_algo="PPOLag", controller_variant=None, group="main")
    assert not onset.is_determinism_spec(spec) and onset.is_registered_spec(spec)
    factory = onset.make_pid_algorithm if spec.plugin == "study_a_pid" else onset.make_algorithm
    with pytest.raises(RunRefused, match="determinism check's"):
        factory(spec.task, launch.build_config(spec, tmp_path / "omnisafe"), spec)
    with pytest.raises(RunRefused):
        onset.build_onset_cfgs(spec, steps_per_epoch=E)
    assert not (tmp_path / "omnisafe").exists()


@pytest.mark.parametrize("group,match", [("treatment", {"treatment": "reset"}), ("treatment", {"treatment": "injection"}),
                                         ("controller", {"controller_variant": "warm_started"})])
def test_what_acts_at_onset_needs_an_onset_after_step_0(group, match) -> None:
    registered = _spec(group, N=0.5, **match)
    spec = replace(_smoke(registered, 2), depends_on=registered.depends_on)  # onset at epoch 1 of 2
    assert onset.build_onset_cfgs(spec, steps_per_epoch=E, registered=False)["onset_epoch"] == 1
    for early in (replace(spec, onset_step=0), replace(_smoke(registered, 1), depends_on=registered.depends_on)):
        with pytest.raises(RunRefused, match="acts at onset"):
            onset.build_onset_cfgs(early, steps_per_epoch=E, registered=False)


def test_an_onset_that_leaves_no_constrained_epoch_is_refused() -> None:
    spec = replace(_smoke(_spec("pilot", N=0.5), 2), onset_step=2 * E)
    with pytest.raises(RunRefused, match="no constrained epoch"):
        onset.build_onset_cfgs(spec, steps_per_epoch=E, registered=False)
    with pytest.raises(RunRefused, match="whole epoch"):
        onset.build_onset_cfgs(replace(spec, onset_step=E), steps_per_epoch=3_000, registered=False)


def test_the_pid_checks_step_matching_is_answered_by_q_pid_eq9_not_table_3_3() -> None:
    """Table 3.3's PID row ("PID multiplier, abrupt arms, SafetyPointGoal1-v0 only, N = 0.10, 0.25, 0.50") names
    no control; the treatments' and controller variants' rows name the total-steps-matched one."""
    pid = _spec("pid", N=0.5)
    assert pid.step_matching == onset.PID_STEP_MATCHING == "total_steps"
    constrained = _with(pid, step_matching="constrained_steps")
    for registered in (True, False):
        with pytest.raises(RunRefused, match="Q-pid-eq9") as info:
            onset.build_onset_cfgs(constrained, steps_per_epoch=E, registered=registered)
        assert "Table 3.3's PID row names no control" in str(info.value)
    for other in (_spec("treatment", treatment="reset", N=0.5), _spec("controller", controller_variant="rate_limited",
                                                                      N=0.5)):
        with pytest.raises(RunRefused, match=r"total-steps-matched control \(Table 3.3\)") as info:
            onset.build_onset_cfgs(_with(other, step_matching="constrained_steps"), steps_per_epoch=E, registered=False)
        assert "Q-pid-eq9" not in str(info.value)
    with pytest.raises(RunRefused, match="abrupt arms"):
        onset.build_onset_cfgs(_with(pid, onset_shape="ramp"), steps_per_epoch=E, registered=False)


def test_diagnostic_columns_are_never_checked_as_multipliers_or_losses() -> None:
    """The launcher excludes a run with a non-finite ``Metrics/LagrangeMultiplier*`` or ``Loss/`` column;
    the Onset/ columns hold NaN before onset."""
    for key in onset.ONSET_KEYS:
        assert key.startswith("Onset/")
        assert not key.startswith(onset.MULTIPLIER_KEY) and not key.startswith("Loss/")


# ---------------------------------------------------------------------------
# The warm start (Table 2.4; Q-warm-start)
# ---------------------------------------------------------------------------


def _fake_run(root: Path, spec: RunSpec, rows: list[tuple[int, str]], *, steps_per_epoch: int = E,
              checkpoint: tuple[int, float] | None = None) -> Path:
    """A completed run directory of ``spec``: progress.csv rows (epoch, multiplier text), optional checkpoint."""
    run_dir = root / spec.run_id
    omni = run_dir / "omnisafe" / f"PPOLag-{{{spec.task}}}" / "seed-000-2026-09-30-00-00-00"
    (omni / "torch_save").mkdir(parents=True)
    (run_dir / "spec.json").write_text(spec.to_json())
    (omni / "config.json").write_text(json.dumps({"algo_cfgs": {"steps_per_epoch": steps_per_epoch}}))
    with open(omni / "progress.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Train/Epoch", "TotalEnvSteps", "Metrics/LagrangeMultiplier"])
        for epoch, value in rows:
            writer.writerow([epoch, (epoch + 1) * steps_per_epoch, value])
    if checkpoint is not None:
        epoch, value = checkpoint
        state = {k: {} for k in dependencies.FULL_STATE_KEYS}
        state["lagrange"] = {"value": torch.tensor(value, dtype=torch.float32), "class": "Lagrange"}
        torch.save(state, omni / "torch_save" / f"epoch-{epoch}.pt")
    (run_dir / "train_result.json").write_text(json.dumps({
        "run_id": spec.run_id, "status": "completed", "commit_hash": COMMIT, "config_hash": "f" * 64,
        "omnisafe_dir": str(omni.relative_to(run_dir)),
    }))
    return omni


def _warm_cfgs(root: Path, spec: RunSpec) -> dict:
    deps = dependencies.resolve(spec, root / spec.run_id)
    return {"pilot_cfgs": {"dependencies": dependencies.to_config(deps, spec)}}


VALUE = "0.06747621297836304"  # a float32 multiplier as OmniSafe writes it to progress.csv


def test_the_warm_start_value_is_the_n0_runs_logged_multiplier_bit_for_bit(tmp_path) -> None:
    warm = _spec("controller", controller_variant="warm_started", N=0.25)  # onset 2,500,000: epoch 125
    source = _spec("main", N=0.0)
    k0 = warm.onset_step // E
    rows = [(e, "0.5") for e in range(k0 - 3, k0 - 1)] + [(k0 - 1, VALUE), (k0, "0.9")]
    omni = _fake_run(tmp_path, source, rows)
    ws = onset.warm_start_source(_warm_cfgs(tmp_path, warm), warm, warm.onset_step)
    assert ws["value"] == float(VALUE) and ws["checked_against_checkpoint"] is False  # no epoch-125.pt on the grid
    assert (ws["source_run_id"], ws["source_step"], ws["source_progress_row"]) == (source.run_id, warm.onset_step, k0 - 1)
    assert ws["source_commit"] == COMMIT and ws["source_config_hash"] == "f" * 64
    assert ws["source_progress_sha256"] == onset._sha256(omni / "progress.csv")
    assert all(not str(v).startswith("/") for v in ws.values())  # no path: the config hash ignores the data root


def test_the_warm_start_is_cross_checked_against_the_onset_checkpoint(tmp_path) -> None:
    warm = _spec("controller", controller_variant="warm_started", N=0.5)  # onset epoch 250: on the N = 0 grid
    k0 = warm.onset_step // E
    _fake_run(tmp_path, _spec("main", N=0.0), [(k0 - 1, VALUE)], checkpoint=(k0, float(VALUE)))
    assert onset.warm_start_source(_warm_cfgs(tmp_path, warm), warm, warm.onset_step)["checked_against_checkpoint"]
    other = tmp_path / "other"
    other.mkdir()
    _fake_run(other, _spec("main", N=0.0), [(k0 - 1, VALUE)], checkpoint=(k0, 0.07))
    with pytest.raises(RunRefused, match="its checkpoint"):
        onset.warm_start_source(_warm_cfgs(other, warm), warm, warm.onset_step)


def test_an_unreadable_n0_progress_file_is_refused_not_a_crash(tmp_path, monkeypatch) -> None:
    warm = _spec("controller", controller_variant="warm_started", N=0.25)
    _fake_run(tmp_path, _spec("main", N=0.0), [(warm.onset_step // E - 1, VALUE)])

    def unreadable(path):
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(onset, "_sha256", unreadable)
    with pytest.raises(RunRefused, match="cannot be read"):
        onset.warm_start_source(_warm_cfgs(tmp_path, warm), warm, warm.onset_step)


def test_write_json_once_writes_once_and_never_replaces_a_file(tmp_path, monkeypatch) -> None:
    path = tmp_path / onset.INTERVENTION_FILE
    onset.write_json_once(path, {"a": 1})
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1}
    with pytest.raises(RunRefused, match="exists already"):
        onset.write_json_once(path, {"a": 2})
    with pytest.raises(RunRefused, match="plain JSON"):
        onset.write_json_once(tmp_path / "nan.json", {"a": math.nan})
    # a file that appears between the existence check and the publication is refused, not overwritten
    raced = tmp_path / "raced.json"
    real_fdopen = onset.os.fdopen

    def fdopen(*args, **kwargs):
        raced.write_text("other writer\n", encoding="utf-8")
        return real_fdopen(*args, **kwargs)

    monkeypatch.setattr(onset.os, "fdopen", fdopen)
    with pytest.raises(RunRefused, match="exists already"):
        onset.write_json_once(raced, {"a": 3})
    monkeypatch.undo()
    assert raced.read_text(encoding="utf-8") == "other writer\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == [onset.INTERVENTION_FILE, "raced.json"]  # no temporary left


@pytest.mark.parametrize("damage", ["nan", "negative", "not float32", "row missing", "two rows", "not a number"])
def test_a_warm_start_value_that_cannot_be_the_n0_multiplier_is_refused(tmp_path, damage) -> None:
    warm = _spec("controller", controller_variant="warm_started", N=0.1)
    k0 = warm.onset_step // E
    rows = {"nan": [(k0 - 1, "nan")], "negative": [(k0 - 1, "-0.5")], "not float32": [(k0 - 1, "0.1")],
            "row missing": [(k0 - 2, VALUE)], "two rows": [(k0 - 1, VALUE), (k0 - 1, VALUE)],
            "not a number": [(k0 - 1, "")]}[damage]
    _fake_run(tmp_path, _spec("main", N=0.0), rows)
    with pytest.raises(RunRefused):
        onset.warm_start_source(_warm_cfgs(tmp_path, warm), warm, warm.onset_step)


def test_the_warm_start_source_must_be_the_n0_run_of_the_same_task_and_seed(tmp_path) -> None:
    warm = _spec("controller", controller_variant="warm_started", N=0.1)
    k0 = warm.onset_step // E
    late = _spec("main", N=0.1, onset_shape="abrupt", step_matching="total_steps")
    impostor = _with(late, run_id="A-PointGoal1-N0.00-s0")  # the N = 0 run's name on a late arm's spec
    _fake_run(tmp_path, impostor, [(k0 - 1, VALUE)])
    with pytest.raises(RunRefused, match="is not the N = 0"):
        onset.warm_start_source(_warm_cfgs(tmp_path, warm), warm, warm.onset_step)
    seed1 = tmp_path / "seed1"
    seed1.mkdir()
    _fake_run(seed1, _spec("main", N=0.0, seed=1), [(k0 - 1, VALUE)])
    other_seed = _with(warm, depends_on=("A-PointGoal1-N0.00-s1",))
    with pytest.raises(RunRefused, match="is not the N = 0"):
        onset.warm_start_source(_warm_cfgs(seed1, other_seed), other_seed, warm.onset_step)
    with pytest.raises(RunRefused, match="resolved no dependency"):
        onset.warm_start_source({"pilot_cfgs": {"dependencies": {}}}, warm, warm.onset_step)
    with pytest.raises(RunRefused, match="exactly one"):
        onset.warm_start_source({}, _with(warm, controller_variant=None, group="main", depends_on=()), warm.onset_step)


# ---------------------------------------------------------------------------
# The surrogate (equation 3) and the algorithm's configuration checks
# ---------------------------------------------------------------------------


def _bare(cls, **attributes):
    """An instance without OmniSafe's construction (no environment), for the surrogate alone."""
    algo = object.__new__(cls)
    for key, value in attributes.items():
        setattr(algo, key, value)
    return algo


def test_the_surrogate_is_the_reward_advantage_exactly_before_onset_and_equation_3_after() -> None:
    """First Tasks (Role 2): "The surrogate before onset equals the reward advantage exactly, on synthetic
    advantages" (not (A_R - 0.001 A_C) / 1.001)."""
    from omnisafe.common.lagrange import Lagrange
    from omnisafe.common.pid_lagrange import PIDLagrangian

    gen = torch.Generator().manual_seed(0)
    adv_r, adv_c = torch.randn(64, generator=gen), torch.randn(64, generator=gen) * 50
    lagrange = Lagrange(cost_limit=R.COST_LIMIT, lagrangian_multiplier_init=R.LAGRANGE_MULTIPLIER_INIT,
                        lambda_lr=R.LAGRANGE_MULTIPLIER_LR, lambda_optimizer="Adam")
    algo = _bare(onset.OnsetPPOLag, _constraint_active=False, _lagrange=lagrange)
    before = algo._compute_adv_surrogate(adv_r, adv_c)
    assert before is adv_r and torch.equal(before, adv_r)
    penalty = lagrange.lagrangian_multiplier.item()
    assert not torch.equal((adv_r - penalty * adv_c) / (1 + penalty), adv_r)  # the mistake First Tasks warns of
    algo._constraint_active = True
    assert torch.equal(algo._compute_adv_surrogate(adv_r, adv_c), (adv_r - penalty * adv_c) / (1 + penalty))
    pinned = yaml.safe_load((REPO / "configs" / "omnisafe" / "CPPOPID.yaml").read_text())["defaults"]["lagrange_cfgs"]
    pid = PIDLagrangian(**{**pinned, "lagrangian_multiplier_init": R.LAGRANGE_MULTIPLIER_INIT,
                           "cost_limit": R.COST_LIMIT})
    pid._cost_penalty = 0.4
    algo = _bare(onset.OnsetCPPOPID, _constraint_active=False, _lagrange=pid)
    assert algo._compute_adv_surrogate(adv_r, adv_c) is adv_r
    algo._constraint_active = True
    assert torch.equal(algo._compute_adv_surrogate(adv_r, adv_c), (adv_r - 0.4 * adv_c) / (1 + 0.4))


def _cfgs(spec: RunSpec, tmp_path: Path, deps=None):
    from pilot import launch

    return launch.build_config(spec, tmp_path / spec.run_id / "omnisafe", deps)


def _with_onset(cfgs, spec: RunSpec):
    from omnisafe.utils.config import Config

    cfgs["onset_cfgs"] = Config.dict2config(onset.build_onset_cfgs(spec, steps_per_epoch=E))
    return cfgs


@pytest.mark.parametrize("damage", ["no onset_cfgs", "use_cost", "penalty", "actor", "epoch length", "pilot onset",
                                    "missing key", "cost limit", "no constrained epoch", "warm start without value",
                                    "warm start on another arm", "negative warm start", "ramp without window",
                                    "ramp upwards", "rate limit on another arm", "rate limit missing",
                                    "intervention missing", "intervention of another treatment", "intervention depth",
                                    "intervention not a mapping", "intervention without pilot onset",
                                    "intervention on another arm", "ramp on the PID check", "fractional epochs"])
def test_the_algorithm_refuses_a_configuration_its_behaviour_depends_on(tmp_path, damage) -> None:
    spec = _spec("pilot", N=0.5)
    if damage == "rate limit missing":
        spec = _spec("controller", controller_variant="rate_limited", N=0.5)
    elif damage.startswith("intervention") and damage != "intervention on another arm":
        spec = _spec("treatment", treatment="reset", N=0.5)
    elif damage == "negative warm start":
        spec = _spec("controller", controller_variant="warm_started", N=0.5)
    elif damage in ("ramp without window", "ramp upwards"):
        spec = _spec("main", N=0.5, onset_shape="ramp", step_matching="total_steps")
    elif damage == "ramp on the PID check":
        spec = _spec("pid", N=0.5)
    pid = spec.controller_variant == R.PID_VARIANT
    cfgs = _with_onset(_cfgs(spec, tmp_path, _fixture_deps(spec, tmp_path)), spec)
    if damage == "no onset_cfgs":
        del cfgs["onset_cfgs"]
    elif damage == "use_cost":
        cfgs.algo_cfgs.use_cost = False
    elif damage == "penalty":
        cfgs.algo_cfgs.penalty_coef = 0.5
    elif damage == "actor":
        cfgs.model_cfgs.actor_type = "gaussian_sac"
    elif damage == "epoch length":
        cfgs.algo_cfgs.steps_per_epoch = 10_000
    elif damage == "pilot onset":
        cfgs.pilot_cfgs.onset_step = spec.onset_step + E
    elif damage == "missing key":
        del cfgs.onset_cfgs["warm_start"]
    elif damage == "cost limit":
        cfgs.lagrange_cfgs.cost_limit = 30.0
    elif damage == "no constrained epoch":
        cfgs.train_cfgs.epochs = spec.onset_step // E
    elif damage == "warm start without value":
        cfgs.onset_cfgs.controller_variant = "warm_started"
    elif damage == "warm start on another arm":
        cfgs.onset_cfgs.warm_start = {"value": 0.5}
    elif damage == "negative warm start":
        cfgs.onset_cfgs.warm_start = {"value": -0.5}
    elif damage == "ramp without window":
        cfgs.onset_cfgs.ramp_window_steps = None
    elif damage == "ramp upwards":
        cfgs.onset_cfgs.d_loose = 10.0
    elif damage == "rate limit on another arm":
        cfgs.onset_cfgs.rate_limit_relative = R.RATE_LIMIT_RELATIVE
    elif damage == "rate limit missing":
        cfgs.onset_cfgs.rate_limit_absolute = None
    elif damage == "intervention missing":
        cfgs.onset_cfgs.intervention = None
    elif damage == "intervention of another treatment":
        cfgs.onset_cfgs.intervention.function = "plasticity_injection"
    elif damage == "intervention depth":
        cfgs.onset_cfgs.intervention.depth = 0
    elif damage == "intervention not a mapping":
        cfgs.onset_cfgs.intervention = "partial_reset"
    elif damage == "intervention without pilot onset":
        del cfgs.pilot_cfgs["onset_step"]  # no onset checkpoint would be saved, which the intervention requires
    elif damage == "intervention on another arm":
        cfgs.onset_cfgs.intervention = {"function": "partial_reset", "depth": 2}
    elif damage == "ramp on the PID check":  # OnsetMixin._update applies no ramp to OmniSafe's PID update
        onset.check_training_config(cfgs, pid=True)  # accepted before the damage
        cfgs.onset_cfgs.update(d_loose=R.D_LOOSE[spec.task], ramp_window_steps=R.RAMP_WINDOW_STEPS)
    elif damage == "fractional epochs":  # int() would truncate it
        cfgs.train_cfgs.epochs = R.EPOCHS + 0.5
    with pytest.raises(RunRefused):
        onset.check_training_config(cfgs, pid=pid)


def _fixture_deps(spec: RunSpec, tmp_path: Path):
    """A resolved (fixture) N = 0 dependency for a warm-started spec, else None."""
    if not spec.depends_on:
        return None
    _fake_run(tmp_path, _spec("main", N=0.0), [(spec.onset_step // E - 1, VALUE)])
    return dependencies.resolve(spec, tmp_path / spec.run_id)


def test_the_algorithm_accepts_the_launchers_configuration_and_refuses_the_other_base(tmp_path) -> None:
    for spec in (_spec("pilot", N=0.5), _spec("treatment", treatment="injection", N=0.1),
                 _spec("controller", controller_variant="rate_limited", N=0.25),
                 _spec("main", N=0.1, onset_shape="ramp", step_matching="constrained_steps")):
        oc = onset.check_training_config(_with_onset(_cfgs(spec, tmp_path), spec), pid=False)
        assert oc["onset_epoch"] == spec.onset_step // E and isinstance(oc, dict)
    spec = _spec("pilot", N=0.5)
    with pytest.raises(RunRefused, match="multiplier"):
        onset.check_training_config(_with_onset(_cfgs(spec, tmp_path / "b"), spec), pid=True)


# ---------------------------------------------------------------------------
# The data control's learning-rate schedule (Table 2.4; Table 9.1, Q-data-control-lr)
# ---------------------------------------------------------------------------

ACTOR_LR = 3e-4  # model_cfgs.actor.lr of the pinned PPOLag.yaml


def _rates(make_scheduler, steps: int) -> list[float]:
    """The actor's rate of epochs 0 to ``steps`` (after k steps), as OmniSafe's loop sets and logs it."""
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.Adam([param], lr=ACTOR_LR)
    scheduler = make_scheduler(optimizer)
    rates = [optimizer.param_groups[0]["lr"]]
    for _ in range(steps):
        optimizer.step()  # no gradient, no change; OmniSafe steps the optimiser before the schedule
        scheduler.step()
        rates.append(optimizer.param_groups[0]["lr"])
        assert scheduler.get_last_lr() == [rates[-1]]  # Train/LR (policy_gradient.py:278)
    return rates


def _untreated(epochs: int):
    """OmniSafe's own actor schedule of a run of ``epochs`` epochs (actor_critic.py:99-113)."""
    from torch.optim.lr_scheduler import LinearLR

    return lambda optimizer: LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=epochs)


def _data_control(schedule: dict):
    """OmniSafe's LinearLR over the data control's own epochs, replaced as OnsetMixin._init replaces it."""
    def build(optimizer):
        _untreated(schedule["decay_epochs"] + schedule["tail_epochs"])(optimizer)
        return onset.DataControlLR(optimizer, **schedule)
    return build


@pytest.mark.parametrize("N", R.LATE_ONSET_FRACTIONS)
def test_the_data_controls_first_t_steps_have_the_untreated_arms_rates_bit_for_bit(N) -> None:
    """Epochs 0 to 499 have torch LinearLR(total_iters = 500)'s rates exactly (==); epochs 500 to 500 + N·T/E − 1
    have 3e-4 × (1 − N) × (1 − (e − 500)/(N·T/E)); the rate is 0 after the last step. At N = 0.50 the extension
    repeats the untreated constrained phase's rates (up to LinearLR's recursive rounding, about 2e-15 relative); at
    every N its mean rate equals the untreated constrained phase's to within one epoch's discretisation."""
    data = _spec("treatment", treatment="additional_constrained", N=N)
    schedule = onset.build_onset_cfgs(data, steps_per_epoch=E)[onset.SCHEDULE_KEY]
    tail = schedule["tail_epochs"]
    assert (schedule["decay_epochs"], tail * E) == (R.EPOCHS, data.params["additional_constrained_steps"])
    rng = torch.get_rng_state()
    rates = _rates(_data_control(schedule), R.EPOCHS + tail + 3)
    assert torch.equal(rng, torch.get_rng_state())  # no random number drawn
    untreated = _rates(_untreated(R.EPOCHS), R.EPOCHS)
    assert rates[:R.EPOCHS] == untreated[:R.EPOCHS] and untreated[R.EPOCHS] == 0.0
    start = 1 - Fraction(str(N))
    assert rates[R.EPOCHS:R.EPOCHS + tail] == [float(Fraction(ACTOR_LR) * start * (1 - Fraction(j, tail)))
                                               for j in range(tail)]
    assert rates[R.EPOCHS + tail:] == [0.0] * 4
    k0 = data.onset_step // E
    if N == 0.5:
        assert all(math.isclose(a, b, rel_tol=1e-14, abs_tol=0.0) for a, b in zip(rates[R.EPOCHS:R.EPOCHS + tail],
                                                                                   untreated[k0:R.EPOCHS]))
    extension = sum(rates[R.EPOCHS:R.EPOCHS + tail]) / tail
    constrained = sum(untreated[k0:R.EPOCHS]) / (R.EPOCHS - k0)
    assert abs(extension - constrained) <= ACTOR_LR * float(start) / (2 * tail)


def test_the_schedule_splits_the_run_at_t_and_a_smoke_copy_keeps_the_proportion() -> None:
    for N, tail in ((0.1, 50), (0.25, 125), (0.5, 250)):
        assert onset.actor_lr_schedule(R.EPOCHS + tail, N) == {"decay_epochs": R.EPOCHS, "tail_epochs": tail,
                                                              "tail_start_factor": float(1 - Fraction(str(N)))}
    assert onset.actor_lr_schedule(6, 0.5) == {"decay_epochs": 4, "tail_epochs": 2, "tail_start_factor": 0.5}
    assert onset.actor_lr_schedule(3, 0.5)["decay_epochs"] == 2 and onset.actor_lr_schedule(2, 0.5)["decay_epochs"] == 1
    assert onset.actor_lr_schedule(1, 0.1) == {"decay_epochs": 1, "tail_epochs": 0, "tail_start_factor": 0.9}
    data = _spec("treatment", treatment="additional_constrained", N=0.5)
    assert onset.build_onset_cfgs(_smoke(data, 3), steps_per_epoch=E, registered=False)[onset.SCHEDULE_KEY] == {
        "decay_epochs": 2, "tail_epochs": 1, "tail_start_factor": 0.5}
    with pytest.raises(RunRefused, match="whole number of"):  # 3 epochs of E are 1.5 epochs of 2E: no schedule
        onset.build_onset_cfgs(replace(_smoke(data, 3), onset_step=2 * E), steps_per_epoch=2 * E, registered=False)
    for epochs, N in ((0, 0.5), (True, 0.5), (3.0, 0.5), (3, 0.0), (3, 1.0)):
        with pytest.raises(ValueError):
            onset.actor_lr_schedule(epochs, N)
    # without tail epochs (a smoke copy too short for one) the schedule is the untreated arm's
    no_tail = {"decay_epochs": 3, "tail_epochs": 0, "tail_start_factor": 0.9}
    assert _rates(_data_control(no_tail), 5) == _rates(_untreated(3), 5)
    optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=ACTOR_LR)
    for bad in ({**no_tail, "decay_epochs": 0}, {**no_tail, "tail_epochs": -1}, {**no_tail, "decay_epochs": 2.0},
                {**no_tail, "tail_start_factor": 0.0}, {**no_tail, "tail_start_factor": 1.5},
                {**no_tail, "tail_start_factor": True}):
        with pytest.raises(ValueError):
            onset.DataControlLR(optimizer, **bad)
    with pytest.raises(ValueError, match="no epoch argument"):
        onset.DataControlLR(optimizer, **no_tail).step(2)


@pytest.mark.parametrize("stop", [3, 8])  # inside the LinearLR part, inside the restart
def test_the_data_control_schedule_resumes_from_a_weights_only_checkpoint(tmp_path, stop) -> None:
    schedule = {"decay_epochs": 6, "tail_epochs": 4, "tail_start_factor": 0.75}
    rates = _rates(_data_control(schedule), 12)
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.Adam([param], lr=ACTOR_LR)
    scheduler = _data_control(schedule)(optimizer)
    for _ in range(stop):
        optimizer.step()
        scheduler.step()
    torch.save({"actor_optimizer": optimizer.state_dict(), "actor_scheduler": scheduler.state_dict()},
               tmp_path / "epoch.pt")
    state = torch.load(tmp_path / "epoch.pt", weights_only=True)  # plain numbers only (contract 3)
    assert state["actor_scheduler"] == scheduler.state_dict()
    resumed_optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=ACTOR_LR)
    resumed = _data_control(schedule)(resumed_optimizer)  # built before the optimiser state is loaded (torch's order)
    resumed_optimizer.load_state_dict(state["actor_optimizer"])
    resumed.load_state_dict(state["actor_scheduler"])
    assert resumed_optimizer.param_groups[0]["lr"] == rates[stop] and resumed.get_last_lr() == [rates[stop]]
    for k in range(stop + 1, 13):
        resumed_optimizer.step()
        resumed.step()
        assert resumed_optimizer.param_groups[0]["lr"] == rates[k]
    other = onset.DataControlLR(torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=ACTOR_LR),
                                **{**schedule, "tail_epochs": 5})
    with pytest.raises(ValueError, match="not this one"):
        other.load_state_dict(state["actor_scheduler"])


@pytest.mark.parametrize("damage", ["schedule missing", "schedule None", "schedule on another arm",
                                    "schedule None on another arm", "epochs off", "no decay epoch", "negative tail",
                                    "fractional epochs", "start factor 0", "start factor above 1", "extra field",
                                    "no linear decay", "no actor rate"])
def test_only_the_data_control_carries_a_schedule_and_it_must_fit_the_run(tmp_path, damage) -> None:
    data = _spec("treatment", treatment="additional_constrained", N=0.5)
    other = damage.endswith("on another arm")
    spec = _spec("main", N=0.5, onset_shape="abrupt", step_matching="total_steps") if other else data
    cfgs = _with_onset(_cfgs(spec, tmp_path), spec)
    checked = onset.check_training_config(cfgs, pid=False)
    assert (onset.SCHEDULE_KEY in checked) == (not other)
    schedule = cfgs.onset_cfgs.get(onset.SCHEDULE_KEY)
    if damage == "schedule missing":
        del cfgs.onset_cfgs[onset.SCHEDULE_KEY]
    elif damage in ("schedule None", "schedule None on another arm"):
        cfgs.onset_cfgs[onset.SCHEDULE_KEY] = None
    elif damage == "schedule on another arm":
        cfgs.onset_cfgs[onset.SCHEDULE_KEY] = {"decay_epochs": R.EPOCHS, "tail_epochs": 0, "tail_start_factor": 0.5}
    elif damage == "epochs off":
        cfgs.train_cfgs.epochs = R.EPOCHS + 249
    elif damage == "no decay epoch":
        schedule.update(decay_epochs=0, tail_epochs=R.EPOCHS + 250)
    elif damage == "negative tail":
        schedule.update(decay_epochs=R.EPOCHS + 251, tail_epochs=-1)
    elif damage == "fractional epochs":
        schedule.update(decay_epochs=float(R.EPOCHS))
    elif damage == "start factor 0":
        schedule.update(tail_start_factor=0.0)
    elif damage == "start factor above 1":
        schedule.update(tail_start_factor=1.5)
    elif damage == "extra field":
        schedule.update(lr=ACTOR_LR)
    elif damage == "no linear decay":
        cfgs.model_cfgs.linear_lr_decay = False
    elif damage == "no actor rate":
        cfgs.model_cfgs.actor.lr = None
    with pytest.raises(RunRefused):
        onset.check_training_config(cfgs, pid=False)


def test_the_factory_swaps_the_data_controls_schedule_before_the_saver_captures_it(tmp_path, one_thread) -> None:
    """OnsetMixin._init replaces OmniSafe's LinearLR over the run's 625 epochs by DataControlLR before _init_log, so
    the full-state saver holds the new object, epoch-0.pt saves its state (weights_only), and OmniSafe's loop steps
    it: the untreated partner's rates for 500 epochs, then the restart at 0.75 × 3e-4, 0 at the end."""
    data = _spec("treatment", treatment="additional_constrained", N=0.25)
    untreated = _spec("main", N=0.25, onset_shape="abrupt", step_matching="total_steps")
    algos = {}
    try:
        for spec in (data, untreated):
            algos[spec.run_id] = onset.make_algorithm(spec.task, _cfgs(spec, tmp_path), spec)
        algo, partner = algos[data.run_id], algos[untreated.run_id]
        scheduler = algo._actor_critic.actor_scheduler
        assert isinstance(scheduler, onset.DataControlLR) and algo._cfgs.train_cfgs.epochs == R.EPOCHS + 125
        assert algo._logger._what_to_save["actor_scheduler"] is scheduler  # FullStateCheckpointMixin item 1
        assert scheduler.optimizer is algo._actor_critic.actor_optimizer
        assert (scheduler.decay_epochs, scheduler.tail_epochs, scheduler.tail_start_factor) == (R.EPOCHS, 125, 0.75)
        assert scheduler.base_lrs == [algo._cfgs.model_cfgs.actor.lr]
        state = torch.load(_omnisafe_dir(tmp_path / data.run_id) / "torch_save" / "epoch-0.pt", weights_only=True)
        assert state["actor_scheduler"] == scheduler.state_dict()
        assert state["actor_scheduler"]["decay"]["total_iters"] == R.EPOCHS
        reference = partner._actor_critic.actor_scheduler
        assert type(reference).__name__ == "LinearLR" and reference.total_iters == R.EPOCHS
        rates, untreated_rates = [scheduler.get_last_lr()[0]], [reference.get_last_lr()[0]]
        for k in range(1, R.EPOCHS + 126):
            for each in (algo, partner):
                each._actor_critic.actor_optimizer.step()  # no gradient: changes nothing
            scheduler.step()
            rates.append(scheduler.get_last_lr()[0])
            if k <= R.EPOCHS:
                reference.step()
                untreated_rates.append(reference.get_last_lr()[0])
        assert rates[:R.EPOCHS] == untreated_rates[:R.EPOCHS] and untreated_rates[R.EPOCHS] == 0.0
        assert rates[R.EPOCHS] == 0.75 * ACTOR_LR and rates[-1] == 0.0 and rates[-2] > 0.0
    finally:
        for each in algos.values():
            each._env.close()


# ---------------------------------------------------------------------------
# The factories
# ---------------------------------------------------------------------------


@pytest.fixture
def one_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def _omnisafe_dir(run_dir: Path) -> Path:
    (path,) = [p for p in (run_dir / "omnisafe").glob("*/seed-*") if p.is_dir()]
    return path


@pytest.mark.parametrize("group,match,cls,factory", [
    ("pilot", {"N": 0.5}, "OnsetPPOLag", "make_algorithm"),
    ("main", {"N": 0.25, "onset_shape": "ramp", "step_matching": "total_steps"}, "OnsetPPOLag", "make_algorithm"),
    ("controller", {"controller_variant": "rate_limited", "N": 0.1}, "OnsetPPOLag", "make_algorithm"),
    ("treatment", {"treatment": "injection", "N": 0.5}, "OnsetPPOLag", "make_algorithm"),
    ("treatment", {"treatment": "additional_constrained", "N": 0.1}, "OnsetPPOLag", "make_algorithm"),
    ("pid", {"N": 0.25}, "OnsetCPPOPID", "make_pid_algorithm"),
])
def test_the_factory_builds_a_registered_arm_with_its_settings_in_the_config(tmp_path, one_thread, group, match, cls,
                                                                            factory) -> None:
    """The launcher's configuration and the factory, up to the start of training (nothing is trained)."""
    spec = _spec(group, **match)
    cfgs = _cfgs(spec, tmp_path)
    algo = getattr(onset, factory)(spec.task, cfgs, spec)
    assert type(algo).__name__ == cls and algo._onset_epoch == spec.onset_step // E
    omni = _omnisafe_dir(tmp_path / spec.run_id)
    saved = json.loads((omni / "config.json").read_text())
    assert saved["onset_cfgs"] == onset.build_onset_cfgs(spec, steps_per_epoch=E)  # written before construction
    provenance.config_hash(cfgs.todict())  # the launcher hashes it after the factory
    assert (omni / "plasticity.csv").read_text().splitlines()[1].startswith("0,")  # the mixin installed Role 3's hook
    assert [p.name for p in (omni / "torch_save").iterdir()] == ["epoch-0.pt"]
    assert all(key in algo._logger._current_row for key in onset.ONSET_KEYS)
    lagrange = type(algo._lagrange).__name__
    assert lagrange == {"rate_limited": "RateLimitedLagrange", "pid": "PIDLagrangian"}.get(spec.controller_variant,
                                                                                          "Lagrange")
    data_control = spec.treatment == "additional_constrained"
    assert (onset.SCHEDULE_KEY in saved["onset_cfgs"]) == data_control  # no other arm's configuration has the key
    assert type(algo._actor_critic.actor_scheduler).__name__ == ("DataControlLR" if data_control else "LinearLR")
    if spec.treatment == "injection":
        assert algo._fixed_batch.shape == (R.FIXED_BATCH_STATES, algo._env.observation_space.shape[0])
        assert algo._intervention_record is None and not (omni / onset.INTERVENTION_FILE).exists()
    algo._env.close()


def test_the_factory_reads_the_warm_start_before_construction(tmp_path, one_thread) -> None:
    warm = _spec("controller", controller_variant="warm_started", N=0.5)
    k0 = warm.onset_step // E
    _fake_run(tmp_path, _spec("main", N=0.0), [(k0 - 1, VALUE)], checkpoint=(k0, float(VALUE)))
    deps = dependencies.resolve(warm, tmp_path / warm.run_id)
    cfgs = _cfgs(warm, tmp_path, deps)
    algo = onset.make_algorithm(warm.task, cfgs, warm)
    saved = json.loads((_omnisafe_dir(tmp_path / warm.run_id) / "config.json").read_text())
    assert saved["onset_cfgs"]["warm_start"]["value"] == float(VALUE)
    assert saved["onset_cfgs"]["warm_start"]["checked_against_checkpoint"] is True
    assert algo._lagrange.lagrangian_multiplier.item() == float(np.float32(R.LAGRANGE_MULTIPLIER_INIT))  # not yet: at onset
    algo._env.close()


def test_omnisafes_per_environment_rollout_length_is_left_alone(tmp_path, one_thread) -> None:
    """With two environments OmniSafe's own ``_steps_per_epoch`` is E / 2 per environment
    (policy_gradient.py:73-77), and it sizes the buffer (:121) and every rollout (:251). The plug-in
    keeps the whole epoch's E for the ramp (e · E) in its own attribute and never overwrites
    OmniSafe's. Overwriting it made every Study A run with vector_env_nums > 1 overflow its buffer in
    learn(), a crash; this is the regression test."""
    from pilot.algorithms import make_ppolag

    base = _spec("main", N=0.5, onset_shape="ramp", step_matching="total_steps")
    smoke = replace(base, run_id=onset.SMOKE_PREFIX + base.run_id, total_steps=2 * E, onset_step=E)
    built = {}
    for name, factory in (("ppolag", make_ppolag), ("study_a", onset.make_algorithm)):
        cfgs = _cfgs(smoke, tmp_path / name)
        cfgs.train_cfgs.vector_env_nums = 2
        algo = factory(smoke.task, cfgs, smoke)
        try:  # two asynchronous environments are worker processes: always closed
            built[name] = (algo._steps_per_epoch, [buffer.max_size for buffer in algo._buf.buffers])
            if name == "study_a":
                assert algo._onset_steps_per_epoch == E == algo._onset["steps_per_epoch"]
        finally:
            algo._env.close()
    assert built["study_a"] == built["ppolag"] == (E // 2, [E // 2, E // 2])


@pytest.mark.parametrize("run_id,source", [
    ("A-PointGoal1-N0.10-abrupt-constrained-s0", "Table 9.1, Q-rounding"),
    ("A-PointGoal1-N0.25-ramp-constrained-s0", "Table 9.1, Q-rounding"),
    ("A-PointGoal1-N0.50-abrupt-constrained-s0", "the registered values (Table 2.4)"),
    ("A-PointGoal1-N0.25-abrupt-total-s0", "the registered values (Table 2.4)"),
    ("A-PointGoal1-N0.25-abrupt-total-additional_constrained-s0", "the registered values (Table 2.4)"),
    ("A-PointGoal1-N0.00-s0", "the registered values (Table 3.1"),
])
def test_a_schedule_refusal_names_its_source_registered_or_the_rounding_answer(run_id, source) -> None:
    """Table 2.4 registers a constrained-steps arm's total as T/(1 − N), not a whole epoch at N = 0.10
    and 0.25: there onset_schedule's whole-epoch values are Table 9.1's (Q-rounding; its specs carry
    that gate), and a refusal must not call them registered."""
    spec = next(s for s in _study_a_specs() if s.run_id == run_id)
    assert ("Q-rounding" in spec.pending) == ("Q-rounding" in source)
    with pytest.raises(RunRefused) as info:
        onset.build_onset_cfgs(replace(spec, total_steps=spec.total_steps + E), steps_per_epoch=E)
    message = str(info.value)
    assert source in message and f"not {spec.onset_step} and {spec.total_steps}," in message
    assert ("registered" in message) != ("Q-rounding" in source)


@pytest.mark.parametrize("damage", ["env_id", "pid factory", "ppolag factory", "use_cost", "penalty", "actor",
                                    "tiny epochs", "seed", "pilot onset", "plasticity off", "run_id", "treatment",
                                    "onset_cfgs present", "cost limit", "total", "not a config", "plain dict",
                                    "fractional seed", "no pilot_cfgs", "bad spec"])
def test_the_factory_refuses_every_configuration_problem_before_building(tmp_path, damage) -> None:
    spec = _spec("pilot", N=0.5)
    pid_spec = _spec("pid", N=0.5)
    cfgs = _cfgs(spec, tmp_path)
    factory, env_id, arg, built = onset.make_algorithm, spec.task, spec, spec  # built: whose configuration
    if damage == "env_id":
        env_id = "SafetyPointButton1-v0"
    elif damage == "pid factory":
        factory = onset.make_pid_algorithm
    elif damage == "ppolag factory":
        cfgs, arg, built = _cfgs(pid_spec, tmp_path), pid_spec, pid_spec
    elif damage == "use_cost":
        cfgs.algo_cfgs.use_cost = False
    elif damage == "penalty":
        cfgs.algo_cfgs.penalty_coef = 0.1
    elif damage == "actor":
        cfgs.model_cfgs.actor_type = "mlp"
    elif damage == "tiny epochs":
        cfgs.algo_cfgs.steps_per_epoch = 2_000
    elif damage == "seed":
        cfgs.seed = 1
    elif damage == "pilot onset":
        cfgs.pilot_cfgs.onset_step = 0
    elif damage == "plasticity off":
        cfgs.pilot_cfgs.plasticity = False
    elif damage == "run_id":
        cfgs.pilot_cfgs.run_id = "P-A-PointGoal1-N0.50-abrupt-total-s1"
    elif damage == "treatment":
        cfgs.pilot_cfgs.treatment = "reset"
    elif damage == "onset_cfgs present":
        cfgs["onset_cfgs"] = {}
    elif damage == "cost limit":
        cfgs.lagrange_cfgs.cost_limit = 30.0
    elif damage == "total":
        cfgs.train_cfgs.epochs = 10
    elif damage == "not a config":
        cfgs = None
    elif damage == "plain dict":  # OmniSafe's own assertion would fail on it (base_algo.py:39): a crash, not a refusal
        cfgs = cfgs.todict()
    elif damage == "fractional seed":  # int() would truncate it to the spec's seed
        cfgs.seed = spec.seed + 0.5
    elif damage == "no pilot_cfgs":
        del cfgs["pilot_cfgs"]
    elif damage == "bad spec":
        arg = {**spec.to_dict(), "onset_step": 7}
    with pytest.raises(RunRefused):
        factory(env_id, cfgs, arg)
    if cfgs is not None and damage != "onset_cfgs present":
        assert "onset_cfgs" not in cfgs  # nothing was added: the launcher's configuration is as it was
    assert not (tmp_path / built.run_id / "omnisafe").exists()  # refused before OmniSafe's logger wrote anything


def test_a_missing_role_3_module_propagates_as_an_import_error(tmp_path, one_thread, monkeypatch) -> None:
    """The launcher reports a missing repository module as unavailable (exit 4), never a crash."""
    import sys

    from pilot.algorithms import is_missing_repository_import

    monkeypatch.setitem(sys.modules, "metrics.interventions", None)  # "import metrics.interventions" fails
    spec = _spec("treatment", treatment="reset", N=0.5)
    with pytest.raises(ImportError) as info:
        onset.make_algorithm(spec.task, _cfgs(spec, tmp_path), spec)
    assert is_missing_repository_import(info.value)


# ---------------------------------------------------------------------------
# Quotations of the pre-registration
# ---------------------------------------------------------------------------

ENVS_FILES = ("envs/__init__.py", "envs/onset.py", "envs/continuations.py")


def _quote_checker():
    path = REPO / "tests" / "test_core_quotes.py"
    if not path.exists() or not (REPO / "prereg" / "Preregistration.docx").exists():
        pytest.skip("the quotation checker or the pre-registration is not in this checkout")
    module_spec = importlib.util.spec_from_file_location("_envs_quote_checker", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("path", ENVS_FILES)
def test_every_cited_quotation_in_envs_is_in_the_pre_registration(path) -> None:
    checker = _quote_checker()
    source = (REPO / path).read_text(encoding="utf-8")
    quotes = checker.cited_quotes(source)
    bad = {q: checker.missing_pieces(q) for q in quotes}
    assert not {q: m for q, m in bad.items() if m}
    assert not checker.unlabelled_continuation_starts(source)  # a starting checkpoint is Table 9.1's answer (Q-continuations)
    if path != "envs/__init__.py":
        assert len(quotes) >= 5


DECIDED_CONSTANT = re.compile(r"^([A-Z][A-Z0-9_]*)\s*(?::[^=\n]+)?=[^\n#]*#\s*answered in Table 9\.1 \((Q-[A-Za-z0-9-]+)\)",
                              re.M)
DECIDED_CONSTANTS = {  # a value Table 9.1 decides is a named constant commented "# answered in Table 9.1 (Q-key)"
    "envs/onset.py": {"PID_STEP_MATCHING": "Q-pid-eq9"},
    "envs/continuations.py": {"TARGET_ONLY_WEIGHT": "Q-transfer-obs", "TARGET_ONLY_MEAN": "Q-transfer-obs",
                              "TARGET_ONLY_VARIANCE": "Q-transfer-obs"},
}


@pytest.mark.parametrize("path", sorted(DECIDED_CONSTANTS))
def test_decided_values_are_named_constants_under_their_key(path) -> None:
    source = (REPO / path).read_text(encoding="utf-8")
    found = {m.group(1): m.group(2) for m in DECIDED_CONSTANT.finditer(source)}
    assert found == DECIDED_CONSTANTS[path]
    assert all(key in R.PENDING for key in found.values())  # a key of configs.registered, never decided here alone
    assert "# PROPOSAL (" not in source  # every value of these modules is answered (Table 9.1)


def test_the_package_docstring_names_the_owner_and_the_modules() -> None:
    import envs

    doc = envs.__doc__
    assert "Muhammad Talha Jamil" in doc and "Role 2" in doc
    for name in ("envs.onset", "envs.continuations", "envs.evaluation"):
        assert name in doc

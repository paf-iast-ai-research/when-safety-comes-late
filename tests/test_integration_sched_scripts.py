"""The scheduler's scripts: scripts/determinism_check.py --plugin (pilot/scheduler.py) and scripts/smoke_run.py.

Also tests/fake_launcher.py against the contracts it repeats, the quotations of the pre-registration
in pilot/scheduler.py and these scripts (``WP_FILES``), the registration of the late-onset key, and
scripts/record_workstation.py and scripts/copy_omnisafe_configs.py.

No training: the determinism check's training and harness are replaced by stand-ins, and the smoke
script's scheduler launches tests/fake_launcher.py. The tiny real trainings are in
tests/test_integration_sched_slow.py.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import pytest

from configs import registered as R
from pilot import contracts, errors, launch, manifest, provenance
from pilot import scheduler as S

REPO = Path(__file__).resolve().parents[1]
FAKE = Path(__file__).with_name("fake_launcher.py")
sys.path.insert(0, str(Path(__file__).parent))
import fake_launcher  # noqa: E402


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def det():
    return _load(REPO / "scripts" / "determinism_check.py", "determinism_check")


@pytest.fixture
def smoke():
    return _load(REPO / "scripts" / "smoke_run.py", "smoke_run")


# ---------------------------------------------------------------------------
# scripts/determinism_check.py --plugin
# ---------------------------------------------------------------------------


def _arm(plugin: str) -> manifest.RunSpec:
    late = max(R.LATE_ONSET_FRACTIONS)
    return {
        "study_a": lambda: next(s for s in manifest.study_a_main((0,)) if s.task == R.PRIMARY_TASK and s.N == 0.0),
        "study_a_pid": lambda: next(s for s in manifest.study_a_pid((0,)) if s.N == late),
        "study_b": lambda: next(s for s in manifest.study_b((0,)) if s.arm == "Moderate"),
        "unconstrained_ppo": lambda: next(s for s in manifest.pilot() if s.plugin == "unconstrained_ppo"),
    }[plugin]()


@pytest.mark.parametrize("total", [2 * R.STEPS_PER_EPOCH, R.CHECKPOINT_INTERVAL_STEPS])
def test_each_plugin_builds_its_registered_configuration(det, total) -> None:
    legacy = manifest.RunSpec(run_id="DET-PPOLag-PointGoal1-s0", study="A", task=R.PRIMARY_TASK, arm="determinism-check",
                              seed=0, total_steps=total, base_algo="PPOLag", plugin="ppolag", group="determinism")
    assert det.make_spec(total) == legacy == S.determinism_spec("ppolag", total)  # unchanged for ppolag
    for plugin in S.DETERMINISM_PLUGINS:
        spec, arm = det.make_spec(total, plugin), _arm(plugin)
        assert spec == S.determinism_spec(plugin, total)
        assert (spec.run_id, spec.plugin, spec.seed, spec.total_steps) == (f"DET-{arm.run_id}", plugin, 0, total)
        same = {k: v for k, v in spec.to_dict().items() if k not in ("run_id", "total_steps", "onset_step", "pending")}
        assert same == {k: v for k, v in arm.to_dict().items() if k not in ("run_id", "total_steps", "onset_step", "pending")}
        assert spec.pending == (tuple(k for k in arm.pending if k not in S.DETERMINISM_EXEMPT_KEYS)
                                + S.determinism_config_keys(plugin))  # the check's own key (registered)
        assert S.determinism_arm(plugin) == arm
        launch.pilot_config(spec)  # the launcher accepts it (extra checkpoints computable)
    pid = det.make_spec(total, "study_a_pid")
    assert (pid.N, pid.controller_variant, pid.base_algo, pid.onset_step) == (0.5, R.PID_VARIANT, "CPPOPID", total // 2)
    assert contracts.extra_checkpoint_steps(pid) == ([total // 2] if (total // 2) % R.CHECKPOINT_INTERVAL_STEPS else [])
    assert det.make_spec(total, "study_a").onset_step == 0
    moderate = det.make_spec(total, "study_b")
    assert moderate.training_levels == R.STUDY_B_ARMS["Moderate"] and "Q-studyb-order" in _arm("study_b").pending
    assert "Q-studyb-order" not in moderate.pending
    unconstrained = det.make_spec(total, "unconstrained_ppo")
    assert unconstrained.base_algo == "PPO" and unconstrained.pending == ("Q-pilot-unconstrained-seed",)


def test_an_unknown_plugin_is_a_usage_error(det) -> None:
    with pytest.raises(SystemExit) as exc:
        det.main(["--plugin", "battery_finetune"])  # continuations are exempt from the gate: no check of their own
    assert exc.value.code == 2


class Trained(Exception):
    """Raised by the stand-in training: the check got as far as training."""


@pytest.fixture
def no_training(det, monkeypatch, tmp_path):
    def refuse(*a, **kw):
        raise Trained()

    monkeypatch.setattr(det, "train_once", refuse)
    monkeypatch.setattr(det, "OUT_DIR", tmp_path / "reports")
    return tmp_path / "reports"


def test_a_configuration_waiting_on_open_questions_is_refused_before_training(det, no_training, monkeypatch,
                                                                              keys_open) -> None:
    with pytest.raises(SystemExit, match="waiting on Q-jc-window: the study_a configuration"):
        det.main(["--allow-dirty", "--plugin", "study_a"])  # not on Q-plasticity-definitions: it changes no update
    moderate = det.make_spec(R.CHECKPOINT_INTERVAL_STEPS, "study_b")
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset(moderate.pending))
    with pytest.raises(SystemExit, match="waiting on Q-studyb-eval"):
        det.main(["--allow-dirty", "--registered-form", "--plugin", "study_b"])
    assert not no_training.exists()  # no report


def test_every_check_trains_once_its_keys_are_answered(det, no_training) -> None:
    """Every key is answered (docs/DECISIONS.md, 2026-10-02): no determinism check waits on an open question."""
    assert R.ANSWERED_QUESTIONS == frozenset(R.PENDING)
    for plugin in S.DETERMINISM_CHECK_PLUGINS:
        assert S.determinism_open_keys(plugin) == (), plugin
        assert det.waiting_on(det.make_spec(R.CHECKPOINT_INTERVAL_STEPS, plugin), True) == (), plugin
        with pytest.raises(Trained):
            det.main(["--allow-dirty", "--registered-form", "--plugin", plugin])
    with pytest.raises(Trained):  # the bitwise form compares training only: Q-studyb-eval does not hold it
        det.main(["--allow-dirty", "--plugin", "study_b"])
    with pytest.raises(Trained):  # a rehearsal may bypass them
        det.main(["--allow-dirty", "--allow-pending", "--registered-form", "--plugin", "study_a"])


def test_the_pilots_own_keys_release_the_checks_that_hold_the_pilot(det, no_training, monkeypatch) -> None:
    """Once the pilot's own keys are answered, the checks that hold the pilot's runs (determinism_missing) no
    longer wait: the study_a check does not wait on Q-plasticity-definitions, a key of the non-pilot runs only
    (its PENDING text). Only keys that change no update are dropped (pilot/scheduler.py DETERMINISM_EXEMPT_KEYS)."""
    assert "Q-plasticity-definitions" in S.DETERMINISM_EXEMPT_KEYS
    assert "Gates the non-pilot Study A runs" in R.PENDING["Q-plasticity-definitions"]
    held = {}
    for spec in manifest.pilot():
        if spec.plugin in S.DETERMINISM_PLUGINS:
            held.setdefault(spec.plugin, set()).update(spec.pending)
    assert set(held) == {"study_a", "study_b", "unconstrained_ppo"}
    for plugin, keys in held.items():
        monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset(keys))  # the pilot's run gates of that plug-in only
        check = det.make_spec(R.CHECKPOINT_INTERVAL_STEPS, plugin)
        assert check.open_questions == (), plugin
        assert det.waiting_on(check, False) == ()
        # the registered form of the budget-conditioned check also waits on the harness's result gate
        assert det.waiting_on(check, True) == (("Q-studyb-eval",) if plugin == "study_b" else ()), plugin
        with pytest.raises(Trained):
            det.main(["--allow-dirty", "--plugin", plugin])
    # a key that changes an update still holds the check (the PID check keeps Q-pid-eq9 and Q-cost-critic)
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset())
    assert {"Q-pid-eq9", "Q-cost-critic", "Q-jc-window"} <= set(det.make_spec(R.CHECKPOINT_INTERVAL_STEPS,
                                                                               "study_a_pid").open_questions)


def test_the_pid_checks_changed_onset_is_a_named_decided_constant_and_recorded(det) -> None:
    """The study_a_pid check does not run the arm's registered onset; the rule is a named constant marked with its
    key, answered in Table 9.1 (HANDOVER.md section 10), and every change from the arm is listed for the report."""
    source = (REPO / "pilot" / "scheduler.py").read_text(encoding="utf-8")
    line = next(x for x in source.splitlines() if x.startswith("DETERMINISM_LATE_ONSET = "))
    assert line.endswith(f"# answered in Table 9.1 ({S.DETERMINISM_LATE_ONSET_KEY})")  # a key, not just a marker
    arm, total = _arm("study_a_pid"), R.CHECKPOINT_INTERVAL_STEPS
    changes = S.determinism_changes("study_a_pid", total)
    assert changes["onset_step"] == [arm.onset_step, total // 2] and arm.onset_step == arm.total_steps // 2
    assert changes["total_steps"] == [arm.total_steps, total] and changes["run_id"] == [arm.run_id, f"DET-{arm.run_id}"]
    assert set(changes) == {"run_id", "total_steps", "onset_step", "pending"}
    for plugin in ("study_a", "study_b", "unconstrained_ppo"):
        assert "onset_step" not in S.determinism_changes(plugin, total), plugin
    assert S.determinism_changes("ppolag", total) == {} and S.determinism_arm("ppolag") is None
    json.dumps(changes)  # the report holds it as it is


@pytest.fixture
def fake_check(det, monkeypatch, tmp_path):
    """Stand-ins for training and the harness; returns (evaluator calls, a function giving a fresh report dir)."""
    import pilot.enrichment as enrichment

    calls: list = []

    def train_once(spec, run_dir, allow_dirty, allow_pending=False):
        omni = run_dir / "omnisafe" / "exp" / "seed-000"
        omni.mkdir(parents=True)
        (run_dir / "train_result.json").write_text(json.dumps({
            "commit_hash": provenance.commit_hash(), "worktree_dirty": False,
            "status": "completed", "config_hash": "c" * 64, "versions": {"omnisafe": "0.5.0"}, "torch_threads": 1,
            "wall_clock_hours": 0.1, "omnisafe_dir": "omnisafe/exp/seed-000", "allow_pending": allow_pending}))
        return omni

    def evaluate_checkpoint(omnisafe_dir, step, episodes, *, spec=None):
        calls.append((step, episodes, spec))
        if spec is not None and spec["plugin"] == "study_b":  # as envs.evaluation does for a budget-conditioned run
            errors.require_answered("Q-studyb-eval", what="a budget-conditioned checkpoint")
        return 3.25

    monkeypatch.setattr(det, "train_once", train_once)
    monkeypatch.setattr(enrichment, "_load", lambda target: evaluate_checkpoint)
    monkeypatch.setattr(provenance, "require_clean_worktree", lambda *a, **kw: "a" * 40)
    monkeypatch.setattr(provenance, "commit_hash", lambda *a, **kw: "a" * 40)
    monkeypatch.setattr(provenance, "dirty_paths", lambda *a, **kw: [])
    monkeypatch.setattr(provenance, "unverified_imported_code", lambda *a, **kw: [])
    counter = iter(range(100))

    def out_dir() -> Path:
        path = tmp_path / f"reports{next(counter)}"
        monkeypatch.setattr(det, "OUT_DIR", path)
        return path

    return calls, out_dir


def _report(directory: Path) -> tuple[str, dict]:
    (path,) = directory.glob("*.json")
    return path.name, json.loads(path.read_text())


def test_the_pid_checks_changed_onset_waits_on_its_own_key(det, no_training, monkeypatch) -> None:
    """The study_a_pid check's changed onset (DETERMINISM_LATE_ONSET) is the rule of its own key,
    Q-determinism-late-onset, so while that key is open (reopened here) answering the arm's own keys alone does not
    let its report release the PID-check runs: an open question gates, it never decides. The key (registered in configs/registered.py PENDING) holds the check, and would
    hold it too, fail closed, were its PENDING entry missing."""
    pid = _arm("study_a_pid")
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset(pid.pending))  # the arm's own keys only
    assert S.determinism_config_keys("study_a_pid") == (S.DETERMINISM_LATE_ONSET_KEY,)
    for plugin in ("ppolag", "study_a", "study_b", "unconstrained_ppo"):  # they run their arm's configuration
        assert S.determinism_config_keys(plugin) == () == S.determinism_open_keys(plugin), plugin
    assert S.DETERMINISM_LATE_ONSET_KEY in det.make_spec(R.CHECKPOINT_INTERVAL_STEPS, "study_a_pid").pending
    registered = R.PENDING
    unregistered = MappingProxyType({k: v for k, v in registered.items() if k != S.DETERMINISM_LATE_ONSET_KEY})
    for pending in (unregistered, registered):  # fail closed without its entry; an ordinary open question with it
        monkeypatch.setattr(R, "PENDING", pending)
        assert S.determinism_open_keys("study_a_pid") == (S.DETERMINISM_LATE_ONSET_KEY,)
        spec = det.make_spec(R.CHECKPOINT_INTERVAL_STEPS, "study_a_pid")
        assert det.waiting_on(spec, True) == (S.DETERMINISM_LATE_ONSET_KEY,)
        with pytest.raises(SystemExit, match=f"waiting on {S.DETERMINISM_LATE_ONSET_KEY}: the study_a_pid"):
            det.main(["--allow-dirty", "--plugin", "study_a_pid"])
        assert not no_training.exists()
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset(pid.pending) | {S.DETERMINISM_LATE_ONSET_KEY})
    assert S.determinism_open_keys("study_a_pid") == ()
    with pytest.raises(Trained):
        det.main(["--allow-dirty", "--plugin", "study_a_pid"])


def test_the_registered_form_records_the_plugin_and_passes_the_schedulers_gate(det, fake_check, monkeypatch) -> None:
    calls, out_dir = fake_check
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset(R.PENDING))
    for plugin in S.DETERMINISM_CHECK_PLUGINS:
        directory = out_dir()
        assert det.main(["--registered-form", "--plugin", plugin]) == 0
        name, report = _report(directory)
        assert name.startswith("determinism-20" if plugin == "ppolag" else f"determinism-{plugin}-20")
        assert (report["plugin"], report["rehearsal_only"], report["allow_pending"]) == (plugin, False, False)
        assert S.determinism_report_plugin(report) == plugin
        spec = det.make_spec(R.CHECKPOINT_INTERVAL_STEPS, plugin)
        assert calls[-1] == (R.CHECKPOINT_INTERVAL_STEPS, R.EVAL_EPISODES, spec.to_dict())
        md = (directory / name.replace(".json", ".md")).read_text()
        assert f"plug-in: {plugin}" in md
        # the configuration checked, against the registered arm it stands for
        assert report["changed_from_arm"] == S.determinism_changes(plugin, R.CHECKPOINT_INTERVAL_STEPS)
        if plugin == "ppolag":
            assert (report["arm"], report["changed_from_arm"], report["late_onset_rule"]) == (None, {}, None)
        else:
            assert report["arm"] == _arm(plugin).run_id and f"- arm: {report['arm']}" in md
        if plugin == "study_a_pid":
            assert report["late_onset_rule"] == S.DETERMINISM_LATE_ONSET
            assert report["changed_from_arm"]["onset_step"] == [_arm(plugin).onset_step, R.CHECKPOINT_INTERVAL_STEPS // 2]
            assert f"onset_step {_arm(plugin).onset_step} -> {R.CHECKPOINT_INTERVAL_STEPS // 2}" in md
            assert "not the arm's registered onset" in md
        else:
            assert report["late_onset_rule"] is None
    for flags in (["--allow-dirty"], ["--allow-pending"]):  # a rehearsal never releases the gate
        directory = out_dir()
        assert det.main(["--registered-form", "--plugin", "study_a", *flags]) == 0
        report = _report(directory)[1]
        assert report["rehearsal_only"] is True and S.determinism_report_plugin(report) is None


def test_a_commit_landing_between_the_two_runs_fails_the_check(det, fake_check, monkeypatch) -> None:
    """runA and runB are separate launcher processes, so a commit can land between them. The report records both
    runs' commits and passes (releasing the scheduler's gate) only if both are the commit it names."""
    _, out_dir = fake_check
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset(R.PENDING))
    trained = iter(["a" * 40, "b" * 40])
    real = det.train_once

    def moving(spec, run_dir, allow_dirty, allow_pending=False):
        monkeypatch.setattr(provenance, "commit_hash", lambda *a, **kw: next(trained, "b" * 40))
        return real(spec, run_dir, allow_dirty, allow_pending)

    monkeypatch.setattr(det, "train_once", moving)
    directory = out_dir()
    assert det.main(["--registered-form", "--plugin", "study_a"]) == 1
    report = _report(directory)[1]
    assert report["passed"] is False and report["same_commit"] is False
    assert report["commit_hash"] == "a" * 40 and report["run_commits"] == ["a" * 40, "b" * 40]
    assert report["result"]["identical"] is True and S.determinism_report_plugin(report) is None


def test_a_pending_question_raised_by_the_harness_writes_no_report(det, fake_check, monkeypatch) -> None:
    _, out_dir = fake_check
    moderate = det.make_spec(R.CHECKPOINT_INTERVAL_STEPS, "study_b")
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset(moderate.pending) | {"Q-studyb-eval"})
    import pilot.enrichment as enrichment

    def harness_still_waits(omnisafe_dir, step, episodes, *, spec=None):
        raise errors.PendingQuestionError("the evaluation cost depends on open pre-registration questions Q-studyb-eval")

    monkeypatch.setattr(enrichment, "_load", lambda target: harness_still_waits)
    directory = out_dir()
    with pytest.raises(SystemExit, match="waiting on Q-studyb-eval.*no report written"):
        det.main(["--registered-form", "--plugin", "study_b"])
    assert not directory.exists()
    # a rehearsal opens the harness's result gate, and says so
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset(moderate.pending))
    monkeypatch.setattr(enrichment, "_load", lambda target: lambda d, s, e, *, spec=None: (
        errors.require_answered("Q-studyb-eval", what="x"), 2.0)[1])
    directory = out_dir()
    assert det.main(["--registered-form", "--plugin", "study_b", "--allow-dirty", "--allow-pending"]) == 0
    report = _report(directory)[1]
    assert report["allow_pending"] is True and report["result"]["identical"] is True
    assert not errors.pending_allowed()  # the bypass ended with the check


def test_compare_reads_every_multiplier_column(det, tmp_path) -> None:
    torch = pytest.importorskip("torch")

    def run(name: str, weight: float, levels: bool) -> Path:
        omni = tmp_path / name
        (omni / "torch_save").mkdir(parents=True)
        cols = "Metrics/LagrangeMultiplier/level_10,Metrics/LagrangeMultiplier/level_10/Max" if levels else "Metrics/LagrangeMultiplier"
        vals = "0.5,0.6" if levels else "0.5"
        (omni / "progress.csv").write_text(f"Metrics/EpCost,Metrics/EpRet,Time/Total,{cols}\n1.0,2.0,9.9,{vals}\n")
        torch.save({"pi": {"w": torch.tensor([weight])}}, omni / "torch_save" / "epoch-0.pt")
        return omni

    same = det.compare(run("a", 1.0, True), run("b", 1.0, True))
    assert same["identical"] and same["per_epoch_run_a"][0]["LagrangeMultiplier/level_10"] == "0.5"
    assert "LagrangeMultiplier/level_10/Max" not in same["per_epoch_run_a"][0]
    assert det.compare(run("c", 1.0, False), run("d", 1.0, False))["per_epoch_run_a"][0]["LagrangeMultiplier"] == "0.5"
    differs = det.compare(run("e", 1.0, True), run("f", 2.0, True))
    assert not differs["identical"] and "tensors differ" in differs["differences"][0]
    # a tensor in one run and a plain value (or the reverse) under the same key is a difference, not an error
    plain_value = run("g", 1.0, True)
    torch.save({"pi": {"w": 1.0}}, plain_value / "torch_save" / "epoch-0.pt")
    for pair in ((run("h", 1.0, True), plain_value), (plain_value, run("i", 1.0, True))):
        mixed = det.compare(*pair)
        assert not mixed["identical"] and "types differ" in mixed["differences"][0]
    multi = run("j", 1.0, True)
    torch.save({"pi": {"w": torch.tensor([1.0, 2.0])}}, multi / "torch_save" / "epoch-0.pt")
    assert "types differ (float vs Tensor)" in det.compare(plain_value, multi)["differences"][0]


# ---------------------------------------------------------------------------
# scripts/smoke_run.py
# ---------------------------------------------------------------------------


def test_the_sample_design_has_one_run_of_each_plugin_family(smoke) -> None:
    specs = smoke.design_specs("sample", 2)
    assert [s.plugin for s in specs] == ["study_a", "study_a", "study_a_pid", "study_b"]
    assert [s.treatment for s in specs] == [None, "reset", None, None]
    assert [s.N for s in specs] == [0.0, 0.5, 0.5, None]
    assert [s.onset_shape for s in specs] == [None, "abrupt", "abrupt", None]
    assert all(s.run_id.startswith("SMOKE-") and s.task == R.PRIMARY_TASK and s.seed == 0 for s in specs)
    assert specs[3].arm == "Moderate" and not any(s.depends_on for s in specs)
    assert {s.total_steps for s in specs} == {2 * R.STEPS_PER_EPOCH}
    # a late onset moves inside the short run, so the onset (and the reset) happens during the smoke test
    assert [s.onset_step for s in specs] == [0, R.STEPS_PER_EPOCH, R.STEPS_PER_EPOCH, None]
    assert [s.onset_step for s in smoke.design_specs("sample", 1)] == [0, 0, 0, None]
    assert [s.onset_step for s in smoke.design_specs("sample", 20)][1] == R.CHECKPOINT_INTERVAL_STEPS
    assert [s.onset_step for s in smoke.design_specs("sample", 100)][1] == 50 * R.STEPS_PER_EPOCH
    assert [s.plugin for s in smoke.design_specs("determinism", 1)] == ["ppolag"]
    assert len(smoke.design_specs("pilot", 1)) == len(manifest.design("pilot"))


def test_the_sample_design_always_passes_through_an_onset(smoke, tmp_path, monkeypatch, capsys) -> None:
    """At 1 epoch every late onset of the sample design would move to step 0 (the PID arm training as N = 0,
    the reset arm refused), so the sample design runs 2 epochs by default and refuses fewer; another design
    run for 1 epoch says that no onset is exercised."""
    assert smoke.onsets_lost("sample", 1) == [s.run_id for s in smoke.design_specs("sample", 1)][1:3]
    assert smoke.onsets_lost("sample", smoke.DEFAULT_EPOCHS["sample"]) == []
    assert smoke.DEFAULT_EPOCHS["sample"] == smoke.MIN_EPOCHS["sample"] == 2
    assert len(smoke.onsets_lost("pilot", 1)) == 3 and smoke.onsets_lost("pilot", 2) == []
    with pytest.raises(SystemExit) as exc:
        smoke.main(["--data-root", str(tmp_path / "root"), "--design", "sample", "--epochs", "1"])
    assert exc.value.code == 2 and "at least 2 for the sample design" in capsys.readouterr().err
    assert not (tmp_path / "root").exists()  # refused before anything is queued
    queued: list = []
    monkeypatch.setattr(smoke.Scheduler, "add", lambda self, specs: queued.extend(specs) or len(queued))
    monkeypatch.setattr(smoke.Scheduler, "run", lambda self, **kw: None)
    assert smoke.main(["--data-root", str(tmp_path / "root"), "--design", "sample"]) == 0
    assert queued == smoke.design_specs("sample", 2)
    assert [s.onset_step for s in queued] == [0, R.STEPS_PER_EPOCH, R.STEPS_PER_EPOCH, None]
    assert "no onset is exercised" not in capsys.readouterr().out
    queued.clear()
    assert smoke.main(["--data-root", str(tmp_path / "pilot-root"), "--design", "pilot"]) == 0
    assert "3 runs have no room for their late onset in 1 epoch(s)" in capsys.readouterr().out


@pytest.mark.parametrize("design", ["determinism", "sample"])
def test_a_smoke_run_is_evaluable_exactly_when_the_harnesses_contract_accepts_it(smoke, design) -> None:
    """``evaluable`` is the selection window of contract 1 (smoke mode allows pending questions): ten
    checkpoints in the final 2,000,000 steps, so from 90 epochs on, off-grid totals included."""
    for epochs, expected in ((1, False), (3, False), (89, False), (90, True), (91, True), (95, True),
                             (100, True), (105, True), (110, True)):
        assert [smoke.evaluable(s) for s in smoke.design_specs(design, epochs)] == [expected] * len(
            smoke.base_specs(design)), epochs
    assert smoke.EVALUATION_EPOCHS == 90
    continuation = next(s for s in manifest.design("all") if s.group in manifest.CONTINUATION_GROUPS)
    assert smoke.evaluable(replace(continuation, total_steps=R.STEPS_PER_EPOCH))  # evaluated at its horizons


def test_the_smoke_script_never_touches_the_registered_data_root(smoke, tmp_path, monkeypatch, capsys) -> None:
    assert smoke.registered_root(Path("/data")) == "/data" and smoke.registered_root(Path("/data/smoke")) == "/data"
    assert smoke.registered_root(tmp_path) is None
    registered = tmp_path / "registered"
    monkeypatch.setattr(smoke, "REGISTERED_DATA_ROOTS", (str(registered),))
    monkeypatch.setattr(S.Scheduler, "run", lambda self, **kw: pytest.fail("smoke_run.py launched runs"))
    for root in (registered, registered / "inner", tmp_path / "link"):
        if root.name == "link":
            registered.mkdir()
            root.symlink_to(registered, target_is_directory=True)
        assert smoke.main(["--data-root", str(root)]) == 1
        assert "registered data root" in capsys.readouterr().err
    assert not (registered / "scheduler").exists() and not (registered / "inner").exists()


def test_the_smoke_script_runs_the_sample_design_through_the_scheduler(smoke, tmp_path, monkeypatch, capsys,
                                                                       keys_open) -> None:
    scenarios = tmp_path / "scenarios.json"
    runs = smoke.design_specs("sample", 2)
    scenarios.write_text(json.dumps({s.run_id: "evalunavailable" for s in runs}))
    monkeypatch.setenv("FAKE_SCENARIOS", str(scenarios))
    real = S.SchedulerConfig

    def with_fake_launcher(**kw):
        return real(**kw, command_builder=lambda stage, spec_path, run_dir: [
            sys.executable, str(FAKE), stage, str(spec_path), str(run_dir)])

    monkeypatch.setattr(smoke, "SchedulerConfig", with_fake_launcher)
    root = tmp_path / "smoke-root"
    assert smoke.main(["--data-root", str(root), "--design", "sample", "--epochs", "2"]) == 0
    out = capsys.readouterr().out
    # waiting for an evaluation harness not yet in the repository does not fail the smoke test
    assert "smoke test passed" in out
    sched = S.Scheduler(real(data_root=root, allow_dirty=True, allow_pending=True))
    assert {r["run_id"]: r["status"] for r in sched.rows()} == {s.run_id: "trained" for s in runs}
    assert sched.setting("mode") == "smoke"
    assert all(s.open_questions for s in runs)  # keys pinned open: --allow-pending let them train (smoke only)
    assert "launch gates: none in smoke mode" in out
    assert "evaluation harness envs.evaluation:evaluate_run is missing" in out
    # the fake launcher saved the onset checkpoint and a plasticity row for each checkpoint of a Study A run
    reset = runs[1]
    omni = next((sched.cfg.run_dir(reset.run_id) / "omnisafe").glob("*/seed-*"))
    steps = [int(line.split(",")[0]) for line in (omni / "plasticity.csv").read_text().splitlines()[1:]]
    assert steps == fake_launcher.checkpoint_steps(reset.to_dict()) == [0, R.STEPS_PER_EPOCH, 2 * R.STEPS_PER_EPOCH]


def test_wscl_data_root_pointing_elsewhere_names_a_smoke_root(tmp_path, monkeypatch) -> None:
    """$WSCL_DATA_ROOT pointing away from /data names a smoke root, which is not refused (HANDOVER.md section 6:
    it points elsewhere only for smoke tests; registered runs keep /data). /data stays refused."""
    scratch = tmp_path / "scratch"
    monkeypatch.setenv("WSCL_DATA_ROOT", str(scratch))
    smoke = _load(REPO / "scripts" / "smoke_run.py", "smoke_run_env")
    assert smoke.REGISTERED_DATA_ROOTS == ("/data",)
    assert smoke.registered_root(scratch) is None and smoke.registered_root(Path("/data/smoke")) == "/data"
    queued: list = []
    monkeypatch.setattr(smoke.Scheduler, "add", lambda self, specs: queued.extend(specs) or len(queued))
    monkeypatch.setattr(smoke.Scheduler, "run", lambda self, **kw: None)
    assert smoke.main(["--data-root", str(scratch)]) == 0
    assert queued == smoke.design_specs("determinism", 1)


@pytest.fixture
def fake_smoke(smoke, tmp_path, monkeypatch, capsys):
    """scripts/smoke_run.py through tests/fake_launcher.py: run(scenarios, *argv) -> (exit code, stdout, stderr);
    ``launched`` lists every (stage, run_id, total_steps) the scheduler launched."""
    real, scen, launched = S.SchedulerConfig, tmp_path / "scenarios.json", []
    monkeypatch.setenv("FAKE_SCENARIOS", str(scen))

    def builder(stage, spec_path, run_dir):
        spec = json.loads(Path(spec_path).read_text())
        launched.append((stage, spec["run_id"], spec["total_steps"]))
        return [sys.executable, str(FAKE), stage, str(spec_path), str(run_dir)]

    monkeypatch.setattr(smoke, "SchedulerConfig", lambda **kw: real(**{**kw, "poll_seconds": 0.02}, command_builder=builder))

    def run(scenarios: dict, *argv: str):
        scen.write_text(json.dumps(scenarios))
        code = smoke.main(list(argv))
        captured = capsys.readouterr()
        return code, captured.out, captured.err

    run.launched = launched
    run.state = lambda root: S.Scheduler(real(data_root=root, allow_dirty=True, allow_pending=True))
    return run


def test_a_crashed_smoke_run_is_replaced_by_a_smoke_copy_only(smoke, fake_smoke, tmp_path) -> None:
    """The replacement of a crashed smoke copy of a Study B run is a smoke copy, continued only as the run it
    replaces was: no registered few-shot continuation (1,000,000 steps each, no SMOKE- prefix) is queued or
    launched on the smoke root. The crash fails the smoke test."""
    runs = smoke.design_specs("sample", 2)
    moderate = runs[3].run_id
    replacement = moderate[: -len("-s0")] + f"-s{R.NEXT_UNUSED_SEED_START}"
    scenarios = {**{s.run_id: "evalrefused" for s in runs}, replacement: "evalrefused", moderate: "crash"}
    root = tmp_path / "root"
    code, _, err = fake_smoke(scenarios, "--data-root", str(root), "--design", "sample")
    sched = fake_smoke.state(root)
    assert sched.row(moderate)["status"] == "excluded" and sched.row(moderate)["replaced_by"] == replacement
    assert all(r["run_id"].startswith(smoke.SMOKE_PREFIX) for r in sched.rows())
    assert {sched.spec(r["run_id"]).total_steps for r in sched.rows()} == {2 * R.STEPS_PER_EPOCH}
    assert {(rid.startswith(smoke.SMOKE_PREFIX), steps) for _, rid, steps in fake_smoke.launched} == {
        (True, 2 * R.STEPS_PER_EPOCH)}
    assert code == smoke.EXIT_RUNS_FAILED and f"{moderate} (excluded)" in err and "1 run did not end" in err
    # the data root is still a smoke root: a second start is not refused for holding registered runs
    code, _, err = fake_smoke(scenarios, "--data-root", str(root), "--design", "sample")
    assert code == smoke.EXIT_RUNS_FAILED and "registered run" not in err


@pytest.mark.parametrize("scenario, epochs, expected", [
    ("evalrefused", 1, 0),       # the real launcher refuses to evaluate a run too short for evaluation
    ("evalfail", 1, 0),          # or its evaluation fails: both expected of a training smoke test
    ("evalunavailable", 1, 0),   # Role 2's harness not in the repository yet
    ("unavailable", 1, 0),       # the plug-in not in the repository yet
    ("evalrefused", 100, 3),     # long enough to be evaluated: a refusal is a failure
    ("evalfail", 100, 3),
    ("evalfail", 91, 3),         # an off-grid total is evaluated too (its end-relative window is saved)
    ("evalrefused", 89, 0),      # one epoch short of ten checkpoints in the final 2,000,000 steps
    ("refused", 1, 3),           # the launcher refused to train it
    ("crash", 1, 3),             # excluded (and its replacements, until the circuit breaker stops them)
    ("sigterm", 1, 3),           # interrupted, held
])
def test_the_smoke_exit_status_tells_a_failing_smoke_test_from_a_passing_one(smoke, fake_smoke, tmp_path,
                                                                            scenario, epochs, expected) -> None:
    """scripts/smoke_run.py exits 0 only when every run ended as a smoke test expects (``failed_runs``)."""
    run_id = smoke.design_specs("determinism", epochs)[0].run_id
    # seed 0 and every replacement seed the circuit breaker can add (MAX_SAME_CAUSE_EXCLUSIONS), with one to spare
    seeds = (0, *range(R.NEXT_UNUSED_SEED_START, R.NEXT_UNUSED_SEED_START + S.MAX_SAME_CAUSE_EXCLUSIONS + 1))
    scenarios = {run_id[: -len("-s0")] + f"-s{seed}": scenario for seed in seeds}
    code, out, err = fake_smoke(scenarios, "--data-root", str(tmp_path / "root"), "--epochs", str(epochs))
    assert code == (smoke.EXIT_RUNS_FAILED if expected else 0), (out, err)
    assert ("smoke test FAILED" in err) == bool(expected) and ("smoke test passed" in out) == (not expected)
    if expected:
        assert run_id in err


def test_the_fake_launcher_saves_what_the_contracts_expect() -> None:
    """tests/fake_launcher.py repeats the rule of pilot.contracts.extra_checkpoint_steps and the plasticity header (it imports nothing of the package)."""
    from metrics.plasticity import COLUMNS

    assert fake_launcher.PLASTICITY_COLUMNS == COLUMNS
    assert fake_launcher.CONTINUATION_GROUPS == manifest.CONTINUATION_GROUPS
    assert fake_launcher.PLASTICITY_PLUGINS == contracts.PLASTICITY_PLUGINS
    assert fake_launcher.SELECTION_WINDOW_CHECKPOINTS == R.SELECTION_WINDOW_CHECKPOINTS
    assert fake_launcher.CHECKPOINT_INTERVAL_STEPS == R.CHECKPOINT_INTERVAL_STEPS
    assert fake_launcher.STEPS_PER_EPOCH == R.STEPS_PER_EPOCH
    assert fake_launcher.MANIPULATION_CHECK_STEPS_AFTER_ONSET == R.MANIPULATION_CHECK_STEPS_AFTER_ONSET
    assert fake_launcher.CLAIM_FILE == launch.CLAIM_FILE
    exits = ("EXIT_COMPLETED", "EXIT_FAILED", "EXIT_INTERRUPTED", "EXIT_UNAVAILABLE", "EXIT_REFUSED")
    assert [getattr(fake_launcher, e) for e in exits] == [getattr(launch, e) for e in exits]
    from envs.evaluation import measurement_seeds, selection_seeds

    assert (fake_launcher.SELECTION_SEEDS, fake_launcher.MEASUREMENT_SEEDS) == (selection_seeds(), measurement_seeds())
    specs = manifest.design("all") + manifest.pilot()
    off_grid = next(s for s in manifest.study_a_pid((0,)) if s.N == max(R.LATE_ONSET_FRACTIONS))
    off_grid = replace(off_grid, total_steps=2_100_000, onset_step=1_960_000)  # an off-grid total and onset
    parent = next(s for s in manifest.study_a_main((0,)) if s.N == 0.0)
    step = contracts.end_relative_window(parent.total_steps)[0]
    matched = {"run_id": parent.run_id, "completed": True, "matched": True, "matched_checkpoint_step": step,
               "matched_checkpoint_path": f"x/epoch-{step // R.STEPS_PER_EPOCH}.pt", "commit_hash": "a" * 40}
    # off-grid continuations: no end-relative selection window (a continuation is not selected from)
    off_grid_continuations = [replace(manifest.battery_continuation(parent, matched, condition), total_steps=1_020_000)
                              for condition in manifest.BATTERY_CONDITIONS]
    short = [replace(specs[0], total_steps=total) for total in (40_000, 400_000, 1_000_000)]  # too short to evaluate
    windows = refused = 0
    for spec in [*specs, off_grid, *off_grid_continuations, *short]:
        extra = contracts.extra_checkpoint_steps(spec)
        assert fake_launcher.extra_checkpoint_steps(spec.to_dict()) == extra, spec.run_id
        steps = fake_launcher.checkpoint_steps(spec.to_dict())
        assert steps == contracts.expected_checkpoint_steps(spec), spec.run_id
        if spec.group in manifest.CONTINUATION_GROUPS:
            continue
        with errors.allow_pending():
            try:
                window = contracts.selection_window(steps, spec.total_steps)
            except contracts.ContractError:
                with pytest.raises(fake_launcher.IncompleteWindow):  # the fake's evaluate stage refuses it (exit 5)
                    fake_launcher.selection_window(spec.to_dict(), steps)
                refused += 1
                continue
        assert fake_launcher.selection_window(spec.to_dict(), steps) == window, spec.run_id
        windows += 1
    assert windows > len(specs) // 2 and refused >= len(short)
    assert fake_launcher.selection_window(off_grid.to_dict(), fake_launcher.checkpoint_steps(off_grid.to_dict())) == [
        300_000 + k * R.CHECKPOINT_INTERVAL_STEPS for k in range(R.SELECTION_WINDOW_CHECKPOINTS)]


@pytest.mark.parametrize("total", [2 * R.STEPS_PER_EPOCH, 1_000_000])
def test_the_fake_launcher_refuses_to_evaluate_a_run_too_short_for_its_selection_window(tmp_path, monkeypatch,
                                                                                       total) -> None:
    """As the real evaluation does (contracts.selection_window), the fake writes no evaluation.json for a run
    without the ten checkpoints of its selection window: its evaluate stage exits with EXIT_REFUSED."""
    spec = replace(manifest.design("main")[0], total_steps=total)
    run_dir = tmp_path / spec.run_id
    run_dir.mkdir()
    (run_dir / "spec.json").write_text(spec.to_json())
    scenarios = tmp_path / "scenarios.json"
    scenarios.write_text("{}")
    monkeypatch.setenv("FAKE_SCENARIOS", str(scenarios))
    codes = [subprocess.run([sys.executable, str(FAKE), stage, str(run_dir / "spec.json"), str(run_dir)],
                            capture_output=True, text=True) for stage in ("train", "evaluate")]
    assert [c.returncode for c in codes] == [fake_launcher.EXIT_COMPLETED, fake_launcher.EXIT_REFUSED]
    assert "REFUSED" in codes[1].stderr and not (run_dir / "evaluation.json").exists()


def test_the_fake_launchers_evaluation_is_contract_1_with_its_supplement_record() -> None:
    """evaluation.json of tests/fake_launcher.py carries what the real harnesses return (HANDOVER.md section 8; studyb/conditioning.py),
    so that the scheduler's tests write each row's evaluation record (results/supplement_schema.py)."""
    from envs.evaluation import training_budget_schedule
    from results.supplement_schema import validate_record

    from pilot import ledger_writer

    assert fake_launcher.EVAL_EPISODES == R.EVAL_EPISODES and fake_launcher.CONTINUOUS_RANGE == R.CONTINUOUS_RANGE
    for arm in R.STUDY_B_ARMS:
        spec = next(s for s in manifest.design("all") if s.plugin == "study_b" and s.arm == arm)
        assert fake_launcher.studyb_budgets(spec.to_dict()) == training_budget_schedule(spec), arm
    for spec in (next(s for s in manifest.design("all") if s.plugin == p) for p in ("study_a", "study_b")):
        window = contracts.selection_window(contracts.expected_checkpoint_steps(spec), spec.total_steps)
        ev = fake_launcher.evaluation_result(spec.to_dict(), window, [25.0 + i * 0.1 for i in range(len(window))])
        assert set(ev) >= {"final_step", "final_seed_set", "measurement_seeds", "episode_costs", "episode_returns",
                           "short_episodes", "eval_commit_hash", "final_selection_cost"}
        assert ("studyb_budgets" in ev) == (spec.plugin == "study_b")
        assert ev["final_selection_cost"] == ev["selection"][str(window[-1])][0]
        record = validate_record("evaluation", ledger_writer.evaluation_record(spec, json.loads(json.dumps(ev))))
        assert record.final.mean_cost == ev["final_cost"] == 24.0 and len(record.selection) == R.SELECTION_WINDOW_CHECKPOINTS


# ---------------------------------------------------------------------------
# Quotations of the pre-registration in pilot/scheduler.py and its scripts (verbatim; tests/test_core_quotes.py)
# ---------------------------------------------------------------------------

WP_FILES = ("pilot/scheduler.py", "scripts/smoke_run.py", "scripts/determinism_check.py", "tests/fake_launcher.py")


@pytest.mark.parametrize("path", WP_FILES)
def test_every_cited_quotation_is_in_the_pre_registration(path) -> None:
    quotes = _load(REPO / "tests" / "test_core_quotes.py", "core_quotes")
    source = (REPO / path).read_text(encoding="utf-8")
    bad = {q: quotes.missing_pieces(q) for q in quotes.cited_quotes(source)}
    assert not {q: m for q, m in bad.items() if m}
    assert not quotes.unlabelled_continuation_starts(source)


def test_the_quotation_check_sees_this_packages_quotations() -> None:
    quotes = _load(REPO / "tests" / "test_core_quotes.py", "core_quotes")
    found = [q for path in WP_FILES for q in quotes.cited_quotes((REPO / path).read_text(encoding="utf-8"))]
    assert len(found) >= 8
    assert any("scheduled after Study A's main sweep" in q for q in found)


def test_the_late_onset_key_is_a_registered_pending_key() -> None:
    """The late-onset key is in PENDING, so the amendment log can answer it: an unregistered key counts as open
    for good (determinism_open_keys fails closed), which would hold the 15 PID-check runs forever."""
    assert S.DETERMINISM_LATE_ONSET_KEY in R.PENDING
    assert not R.is_open(S.DETERMINISM_LATE_ONSET_KEY)  # answered (docs/DECISIONS.md, 2026-10-02); no KeyError
    for plugin in ("ppolag", "study_a", "study_a_pid", "study_b", "unconstrained_ppo"):
        assert set(S.determinism_config_keys(plugin)) <= set(R.PENDING), plugin


def test_the_determinism_report_is_written_whole_and_once(det, fake_check, monkeypatch) -> None:
    """Both report files go through provenance.write_text_once (whole and once-only, unlike Path.write_text), the
    .json last; a failed .json write removes the .md, so no partial or orphaned report is left in
    ledger/determinism/."""
    _, out_dir = fake_check
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset(R.PENDING))
    real = provenance.write_text_once
    written: list[str] = []

    def failing(path, text):
        written.append(Path(path).suffix)
        if Path(path).suffix == ".json":
            raise OSError("no space left on device")
        real(path, text)

    monkeypatch.setattr(provenance, "write_text_once", failing)
    directory = out_dir()
    with pytest.raises(OSError, match="no space"):
        det.main(["--registered-form", "--plugin", "study_a"])
    assert written == [".md", ".json"] and not any(p for p in directory.iterdir() if not p.name.startswith("run"))
    monkeypatch.setattr(provenance, "write_text_once", real)
    directory = out_dir()
    assert det.main(["--registered-form", "--plugin", "study_a"]) == 0
    name, report = _report(directory)
    assert report["plugin"] == "study_a" and (directory / name.replace(".json", ".md")).exists()


# ---------------------------------------------------------------------------
# scripts/record_workstation.py and scripts/copy_omnisafe_configs.py
# ---------------------------------------------------------------------------


def test_the_workstation_record_lists_every_direct_pin() -> None:
    record = _load(REPO / "scripts" / "record_workstation.py", "record_workstation")
    lines = (REPO / "environment" / "requirements.in").read_text(encoding="utf-8").splitlines()
    pins = [line.split("==")[0].strip() for line in lines if "==" in line and not line.lstrip().startswith("#")]
    assert "scipy" in pins and sorted(record.PACKAGES) == sorted(pins)


def test_the_workstation_record_is_written_whole_into_a_new_directory(tmp_path, capsys) -> None:
    """--out may name a file in a directory that does not exist yet (another machine); the record is written
    through a temporary file and a rename, and what is printed is what was written."""
    record = _load(REPO / "scripts" / "record_workstation.py", "record_workstation")
    out = tmp_path / "new" / "dir" / "workstation.json"
    assert record.main(["--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert capsys.readouterr().out == text and set(json.loads(text)["packages"]) == set(record.PACKAGES)
    assert [p.name for p in out.parent.iterdir()] == ["workstation.json"]  # no temporary file left


def test_the_config_check_reports_a_missing_source_record(tmp_path, monkeypatch, capsys) -> None:
    copier = _load(REPO / "scripts" / "copy_omnisafe_configs.py", "copy_omnisafe_configs")
    installed, dest = tmp_path / "installed", tmp_path / "dest"
    installed.mkdir()
    dest.mkdir()
    for name in copier.FILES:
        (installed / name).write_text(f"{name}: 1\n", encoding="utf-8")
        (dest / name).write_text(f"{name}: 1\n", encoding="utf-8")
    monkeypatch.setattr(copier, "installed_dir", lambda: installed)
    monkeypatch.setattr(copier, "DEST", dest)
    monkeypatch.setattr(copier.importlib.metadata, "version", lambda name: "0.5.0")
    assert copier.main(["--check"]) == 1  # no traceback: a mismatch like any other
    assert "SOURCE.json missing" in capsys.readouterr().err
    assert copier.main([]) == 0  # the copy writes it
    assert copier.main(["--check"]) == 0
    assert "matches the installed package" in capsys.readouterr().out

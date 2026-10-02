"""The launcher (pilot/launch.py): exit codes, refusals and --allow-pending; the run-directory layout
(pilot/rundir.py) without an import cycle; the evaluation stage; dependencies and ``pilot_cfgs``; output
paths; interruptions by signal."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from configs import registered as R
from pilot import dependencies, errors, launch, manifest, provenance, rundir
from pilot.manifest import RunSpec

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fake_launcher  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
PILOT_MODULES = sorted(p.stem for p in (REPO / "pilot").glob("*.py") if p.stem != "__init__")


def _det_spec(total_steps: int = R.STEPS_PER_EPOCH, **changes) -> RunSpec:
    base = dict(run_id="DET-PPOLag-PointGoal1-s0", study="A", task=R.PRIMARY_TASK, arm="determinism-check", seed=0,
                total_steps=total_steps, base_algo="PPOLag", plugin="ppolag", group="determinism")
    base.update(changes)
    return RunSpec(**base)


# ---------------------------------------------------------------------------
# One leaf module for the run-directory layout (pilot/rundir.py); no import cycle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module", PILOT_MODULES)
def test_every_pilot_module_imports_first_in_a_fresh_process(module) -> None:
    code = f"import sys, pilot.{module}; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    if module in ("rundir", "errors", "manifest", "contracts", "dependencies", "launch", "scheduler", "provenance"):
        assert out.stdout.strip() == "False"  # importing the pipeline does not import torch


def test_dependencies_and_launch_import_in_either_order() -> None:
    for code in ("import pilot.dependencies, pilot.launch", "import pilot.launch, pilot.dependencies",
                 "from pilot.launch import RunRefused, read_progress, TRAIN_RESULT; import pilot.scheduler"):
        out = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
    out = subprocess.run([sys.executable, "-m", "pilot.launch", "--help"], cwd=REPO, capture_output=True, text=True)
    assert out.returncode == 0 and "--allow-pending" in out.stdout


def test_the_launcher_re_exports_the_run_directory_layout() -> None:
    assert launch.RunRefused is errors.RunRefused
    assert issubclass(errors.PendingQuestionError, launch.RunRefused)
    for name in ("OMNISAFE_SUBDIR", "TRAIN_RESULT", "EVALUATION_RESULT", "SPEC_FILE", "CLAIM_FILE",
                 "MULTIPLIER_COLUMN", "STEPS_COLUMN", "find_omnisafe_run_dir", "read_progress"):
        assert getattr(launch, name) is getattr(rundir, name), name
    assert (launch.TRAIN_RESULT, launch.EVALUATION_RESULT, launch.SPEC_FILE, launch.CLAIM_FILE, launch.OMNISAFE_SUBDIR) == (
        "train_result.json", "evaluation.json", "spec.json", ".claim", "omnisafe")


# ``python -m pilot.launch`` with Q-rounding pinned open (every key is answered, docs/DECISIONS.md of 2026-10-02):
# runpy runs the module as __main__, as ``-m`` does, after the interpreter has opened the key.
PYTHON_M_LAUNCH_ROUNDING_OPEN = (
    "import runpy, sys\n"
    "from configs import registered as R\n"
    "R.ANSWERED_QUESTIONS = frozenset(R.ANSWERED_QUESTIONS) - {'Q-rounding'}\n"
    "sys.argv[0] = 'pilot.launch'\n"
    "runpy.run_module('pilot.launch', run_name='__main__', alter_sys=True)\n"
)


def test_a_refusal_under_python_m_is_the_shared_class(tmp_path) -> None:
    """``python -m pilot.launch`` runs the module as __main__; its own refusal exits 5. A refusal raised
    by another module under ``python -m``: tests/test_core_refusals.py."""
    spec = RunSpec.from_dict({**_det_spec().to_dict(), "pending": ["Q-rounding"]})
    (tmp_path / "spec.json").write_text(spec.to_json())
    out = subprocess.run([sys.executable, "-c", PYTHON_M_LAUNCH_ROUNDING_OPEN, "train", "--spec",
                          str(tmp_path / "spec.json"), "--run-dir", str(tmp_path), "--allow-dirty"],
                         cwd=REPO, capture_output=True, text=True)
    assert out.returncode == launch.EXIT_REFUSED and "REFUSED" in out.stderr and "Q-rounding" in out.stderr
    assert sorted(p.name for p in tmp_path.iterdir()) == ["spec.json"]


def test_run_steps_per_epoch_reads_the_runs_config(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        rundir.run_steps_per_epoch(tmp_path)
    assert rundir.run_steps_per_epoch(tmp_path, default=R.STEPS_PER_EPOCH) == R.STEPS_PER_EPOCH
    for bad in ({}, {"algo_cfgs": {}}, {"algo_cfgs": {"steps_per_epoch": 0}}, {"algo_cfgs": {"steps_per_epoch": 2000.0}},
                {"algo_cfgs": {"steps_per_epoch": True}}, []):
        (tmp_path / "config.json").write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            rundir.run_steps_per_epoch(tmp_path, default=R.STEPS_PER_EPOCH)
    (tmp_path / "config.json").write_text('{"algo_cfgs": ')
    with pytest.raises(ValueError, match="not valid JSON"):
        rundir.run_steps_per_epoch(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({"algo_cfgs": {"steps_per_epoch": 2000}}))
    assert rundir.run_steps_per_epoch(tmp_path) == 2000
    assert rundir.checkpoint_file(tmp_path, 3) == tmp_path / "torch_save" / "epoch-3.pt"


# ---------------------------------------------------------------------------
# Non-finite state, the pending flag and the evaluation stage
# ---------------------------------------------------------------------------


def test_a_non_finite_level_multiplier_is_a_non_finite_multiplier() -> None:
    torch = pytest.importorskip("torch")

    class Level:
        def __init__(self, value) -> None:
            self.lagrangian_multiplier = value

    class StudyB:  # Study B keeps one multiplier per level and no _lagrange (studyb/conditioning.py)
        def __init__(self, values) -> None:
            self._level_lagranges = {10.0: Level(torch.nn.Parameter(torch.tensor(0.1))), 20.0: Level(values)}
            self._actor_critic = torch.nn.Linear(2, 2)

    assert launch.nonfinite_state(StudyB(torch.tensor(0.2))) is None
    assert launch.nonfinite_state(StudyB(torch.nn.Parameter(torch.tensor(math.nan)))) == "non_finite_multiplier"
    assert launch.nonfinite_state(StudyB(math.inf)) == "non_finite_multiplier"  # a plain float (PID-like)
    broken = StudyB(0.3)
    with torch.no_grad():
        broken._actor_critic.weight[0, 0] = math.nan
    assert launch.nonfinite_state(broken) == "non_finite_loss"


@pytest.mark.parametrize("body", [b"\x00\x00", b"1,\xff\xfe", b"1,4e"])
@pytest.mark.parametrize("exception", [None, RuntimeError("x")])
def test_a_corrupt_progress_log_is_classified_not_raised(body, exception, tmp_path) -> None:
    """A progress.csv with NUL bytes, bad UTF-8 or a value cut short (a full disk) is no evidence: the run is
    classified ("incomplete", or "crash" after an exception, Part 5.6), so train_result.json is still written."""
    spec = _det_spec()
    omni = tmp_path / "omnisafe" / "PPOLag-{x}" / "seed-000-t"
    omni.mkdir(parents=True)
    (omni / "progress.csv").write_bytes(f"Train/Epoch,{launch.STEPS_COLUMN}\n".encode() + body)
    status, cause, _ = launch.classify_training(tmp_path, spec, exception)
    assert (status, cause) == ("failed", "incomplete" if exception is None else "crash")


def test_an_atomic_json_write_that_fails_leaves_no_temporary_file(tmp_path, monkeypatch) -> None:
    def full_disk(src, dst):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(launch.os, "replace", full_disk)
    with pytest.raises(OSError, match="No space"):
        launch._write_json_atomic(tmp_path / "train_result.json", {"status": "failed"})
    assert list(tmp_path.iterdir()) == []
    monkeypatch.undo()
    launch._write_json_atomic(tmp_path / "train_result.json", {"status": "failed"})
    assert [p.name for p in tmp_path.iterdir()] == ["train_result.json"]
    assert json.loads((tmp_path / "train_result.json").read_text()) == {"status": "failed"}


def test_repository_packages_are_the_other_roles() -> None:
    from pilot import algorithms

    # one rule for the plug-in loader, the harness loader and both launcher stages
    assert launch.is_repository_module is algorithms.is_repository_module
    assert launch.REPOSITORY_PACKAGES == algorithms.REPOSITORY_PACKAGES == ("envs", "metrics", "studyb", "analysis")
    for name in ("envs", "envs.evaluation", "metrics.interventions", "studyb.evaluation", "analysis.matching"):
        assert launch.is_repository_module(name)
    for name in (None, "", "scipy", "omnisafe.envs", "envsx", "pilot.supplement"):
        assert not launch.is_repository_module(name)


def _trained_run(tmp_path: Path, spec: RunSpec) -> Path:
    run_dir = tmp_path / "run"
    omni = run_dir / "omnisafe" / "PPOLag-{x}" / "seed-000-t"
    (omni / "torch_save").mkdir(parents=True)
    for step in range(0, spec.total_steps + 1, R.CHECKPOINT_INTERVAL_STEPS):
        (omni / "torch_save" / f"epoch-{step // R.STEPS_PER_EPOCH}.pt").write_bytes(b"")
    (run_dir / "spec.json").write_text(spec.to_json())
    (run_dir / "train_result.json").write_text(json.dumps({"run_id": spec.run_id, "status": "completed",
                                                           "omnisafe_dir": str(omni.relative_to(run_dir))}))
    return run_dir


def _good_evaluation(spec: RunSpec) -> dict:
    """A harness's contract-1 result with its per-episode arrays (tests/fake_launcher.py builds it)."""
    window = range(spec.total_steps - (R.SELECTION_WINDOW_CHECKPOINTS - 1) * R.CHECKPOINT_INTERVAL_STEPS,
                   spec.total_steps + 1, R.CHECKPOINT_INTERVAL_STEPS)
    evaluation = fake_launcher.evaluation_result(spec.to_dict(), list(window), [24.0] * R.SELECTION_WINDOW_CHECKPOINTS,
                                                 final_cost=20.0, final_return=5.0, selection_return=1.0)
    return {k: v for k, v in evaluation.items() if not k.startswith("eval_")}  # the launcher adds those


def _argv(run_dir: Path, stage: str = "evaluate", *extra: str) -> list[str]:
    return [stage, "--spec", str(run_dir / "spec.json"), "--run-dir", str(run_dir), "--allow-dirty", *extra]


# Missing repository modules and libraries in both stages, with real module files and through
# ``python -m pilot.launch``: tests/test_core_refusals.py.


def test_the_evaluation_stage_honours_allow_pending_and_restores_the_flag(tmp_path, monkeypatch, capsys) -> None:
    from pilot import contracts

    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", R.ANSWERED_QUESTIONS - {"Q-hazard"})  # its result gate is closed
    spec = _det_spec(R.TOTAL_STEPS)
    run_dir = _trained_run(tmp_path, spec)
    seen: list[bool] = []

    def harness(omnisafe_dir, spec_dict):
        seen.append(errors.pending_allowed())
        errors.require_answered("Q-hazard", what="a hazard evaluation")  # a result gate inside the harness
        return _good_evaluation(spec)

    monkeypatch.setattr(contracts, "load_evaluator", lambda s: harness)
    assert not errors.pending_allowed()
    assert launch.main(_argv(run_dir)) == launch.EXIT_REFUSED  # PendingQuestionError is a refusal
    assert "Q-hazard" in capsys.readouterr().err and not (run_dir / "evaluation.json").exists()
    assert launch.main(_argv(run_dir, "evaluate", "--allow-pending")) == launch.EXIT_COMPLETED
    assert seen == [False, True] and not errors.pending_allowed()  # restored for this (test) process
    evaluation = json.loads((run_dir / "evaluation.json").read_text())
    assert evaluation["final_cost"] == 20.0 and "eval_commit_hash" in evaluation
    # the function form: allow_pending opens the gates only for its own call
    (run_dir / "evaluation.json").unlink()
    assert launch.evaluate(spec, run_dir, allow_dirty=True, allow_pending=True) == launch.EXIT_COMPLETED
    assert not errors.pending_allowed()


def test_the_function_form_opens_the_result_gates_for_the_contract_checks_too(tmp_path, monkeypatch) -> None:
    """An off-grid total (Q-selection-window): ``validate_evaluation`` calls a result gate after the
    harness, so evaluate(allow_pending=True) must complete as ``main(... --allow-pending)`` does."""
    from pilot import contracts

    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", R.ANSWERED_QUESTIONS - {"Q-selection-window"})  # open
    spec = next(s for s in manifest.design("main") + manifest.design("treatment") if "Q-selection-window" in s.pending)
    run_dir = tmp_path / "run"
    omni = run_dir / "omnisafe" / "PPOLag-{x}" / "seed-000-t"
    (omni / "torch_save").mkdir(parents=True)
    for step in contracts.expected_checkpoint_steps(spec):
        (omni / "torch_save" / f"epoch-{step // R.STEPS_PER_EPOCH}.pt").write_bytes(b"")
    (run_dir / "spec.json").write_text(spec.to_json())
    (run_dir / "train_result.json").write_text(json.dumps({"run_id": spec.run_id, "status": "completed",
                                                           "omnisafe_dir": str(omni.relative_to(run_dir))}))
    window = contracts.end_relative_window(spec.total_steps)
    evaluation = fake_launcher.evaluation_result(spec.to_dict(), window, [24.0] * R.SELECTION_WINDOW_CHECKPOINTS,
                                                 final_cost=20.0, final_return=5.0, selection_return=1.0)
    evaluation = {k: v for k, v in evaluation.items() if not k.startswith("eval_")}
    monkeypatch.setattr(contracts, "load_evaluator", lambda s: lambda omnisafe_dir, spec_dict: dict(evaluation))
    with pytest.raises(contracts.SelectionWindowPending):  # gates closed: the open question refuses
        launch.evaluate(spec, run_dir, allow_dirty=True)
    assert launch.evaluate(spec, run_dir, allow_dirty=True, allow_pending=True) == launch.EXIT_COMPLETED
    assert json.loads((run_dir / "evaluation.json").read_text())["eval_allow_pending"] is True
    assert not errors.pending_allowed()


def test_selection_keys_of_mixed_types_are_written_as_one_type(tmp_path, monkeypatch) -> None:
    """int and digit-string step keys both meet contract 1; mixed in ``selection`` or in a per-episode map
    (``episode_costs``, ``episode_returns``, ``unstable_replacements``), they once broke the sorted JSON write."""
    from pilot import contracts

    spec = _det_spec(R.TOTAL_STEPS)
    run_dir = _trained_run(tmp_path, spec)
    evaluation = _good_evaluation(spec)
    window = sorted(int(k) for k in evaluation["selection"])
    replacement = {"slot_index": 3, "seed": 1_000_003, "reserve_seed": 1_050_000, "step": 7,
                   "warning": "mjWARN_BADQVEL x1"}  # Q-mujoco-exception: the selection set's first reserve seed
    evaluation["unstable_replacements"] = {"final": [], "selection": {str(step): [replacement] for step in window[:2]}}
    maps = [evaluation, evaluation["episode_costs"], evaluation["episode_returns"], evaluation["unstable_replacements"]]
    for block in maps:
        selection = {str(k): v for k, v in block["selection"].items()}
        first = next(iter(selection))
        selection[int(first)] = selection.pop(first)
        block["selection"] = selection
    monkeypatch.setattr(contracts, "load_evaluator", lambda s: lambda omnisafe_dir, spec_dict: evaluation)
    assert launch.evaluate(spec, run_dir, allow_dirty=True) == launch.EXIT_COMPLETED
    written = json.loads((run_dir / "evaluation.json").read_text())
    for block in (written, written["episode_costs"], written["episode_returns"]):
        assert sorted(int(k) for k in block["selection"]) == window
    assert written["unstable_replacements"]["selection"] == {str(step): [replacement] for step in window[:2]}


def test_a_claim_that_cannot_be_written_is_removed(tmp_path, monkeypatch) -> None:
    """A failed write of the claim (a full disk) must not strand it: nothing has started yet."""
    spec = _det_spec()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "spec.json").write_text(spec.to_json())
    monkeypatch.setattr(launch.dependencies, "resolve", lambda *a, **k: {})
    cfgs = SimpleNamespace(todict=dict, pilot_cfgs=SimpleNamespace(dependencies=SimpleNamespace(todict=dict)))
    monkeypatch.setattr(launch, "build_config_and_digest", lambda *a, **k: (cfgs, "y" * 64))
    monkeypatch.setattr(launch, "_check_threads", lambda reset=True: 1)
    monkeypatch.setattr(launch.provenance, "commit_hash", lambda *a, **k: "c" * 40)
    import pilot.algorithms as algorithms

    monkeypatch.setattr(algorithms, "load_plugin", lambda key: None)

    def full_disk(fd, data):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(launch.os, "write", full_disk)
    with pytest.raises(OSError, match="No space left"):
        launch.train(spec, run_dir, allow_dirty=True, allow_pending=True)
    monkeypatch.undo()
    assert sorted(p.name for p in run_dir.iterdir()) == ["spec.json"]


@pytest.mark.parametrize("flags, dirty, expected", [
    ((), False, (False, False, False)),
    (("--allow-dirty",), False, (True, False, False)),
    (("--allow-pending",), False, (False, True, False)),
    (("--allow-dirty",), True, (True, False, True)),
])
def test_evaluation_json_records_the_smoke_markers_of_the_evaluation_stage(flags, dirty, expected, tmp_path,
                                                                           monkeypatch) -> None:
    """A run trained cleanly but evaluated with open result gates or uncommitted code is a smoke result:
    evaluation.json says so, as train_result.json does for training (HANDOVER.md section 10)."""
    from pilot import contracts

    spec = _det_spec(R.TOTAL_STEPS)
    run_dir = _trained_run(tmp_path, spec)
    monkeypatch.setattr(contracts, "load_evaluator", lambda s: lambda omnisafe_dir, spec_dict: _good_evaluation(spec))
    monkeypatch.setattr(provenance, "require_clean_worktree", lambda *a, **k: "c" * 40)  # a clean checkout
    monkeypatch.setattr(provenance, "unverified_imported_code", lambda *a, **k: [])
    monkeypatch.setattr(provenance, "dirty_paths", lambda *a, **k: ["pilot/launch.py"] if dirty else [])
    argv = ["evaluate", "--spec", str(run_dir / "spec.json"), "--run-dir", str(run_dir), *flags]
    assert launch.main(argv) == launch.EXIT_COMPLETED
    evaluation = json.loads((run_dir / "evaluation.json").read_text())
    assert (evaluation["eval_allow_dirty"], evaluation["eval_allow_pending"], evaluation["eval_worktree_dirty"]) == expected
    if not flags:
        assert evaluation["eval_commit_hash"] == "c" * 40


def test_an_evaluation_the_ledger_would_refuse_fails_at_the_evaluation_stage(tmp_path, monkeypatch) -> None:
    """A result without per-episode arrays once passed this stage and was refused only at ledger time,
    where ``schedule resolve RUN reevaluate`` did not apply. It is an evaluation failure (exit 1, eval_failed)."""
    from pilot import contracts

    spec = _det_spec(R.TOTAL_STEPS)
    run_dir = _trained_run(tmp_path, spec)
    bare = {k: v for k, v in _good_evaluation(spec).items() if k not in ("episode_costs", "episode_returns")}
    monkeypatch.setattr(contracts, "load_evaluator", lambda s: lambda omnisafe_dir, spec_dict: bare)
    with pytest.raises(contracts.ContractError, match="no per-episode costs and returns"):
        launch.evaluate(spec, run_dir, allow_dirty=True)
    wrong = _good_evaluation(spec)
    wrong["final_cost"] = 21.0  # not the mean of its final episodes: the supplement schema refuses the record
    monkeypatch.setattr(contracts, "load_evaluator", lambda s: lambda omnisafe_dir, spec_dict: wrong)
    with pytest.raises(contracts.ContractError, match="not a valid evaluation supplement record"):
        launch.evaluate(spec, run_dir, allow_dirty=True)
    assert not (run_dir / "evaluation.json").exists()


def test_an_evaluation_json_is_written_once(tmp_path, monkeypatch) -> None:
    """A second evaluate stage on a run once silently replaced its evaluation.json (the one its
    ledger row and supplement record were computed from). It is refused (exit 5), before and after the harness."""
    from pilot import contracts

    spec = _det_spec(R.TOTAL_STEPS)
    run_dir = _trained_run(tmp_path, spec)
    calls: list[int] = []

    def harness(omnisafe_dir, spec_dict):
        calls.append(1)
        return _good_evaluation(spec)

    monkeypatch.setattr(contracts, "load_evaluator", lambda s: harness)
    assert launch.evaluate(spec, run_dir, allow_dirty=True) == launch.EXIT_COMPLETED
    first = (run_dir / "evaluation.json").read_text()
    with pytest.raises(launch.RunRefused, match=f"evaluated twice.*schedule resolve {spec.run_id} reevaluate"):
        launch.evaluate(spec, run_dir, allow_dirty=True)
    assert launch.main(_argv(run_dir)) == launch.EXIT_REFUSED
    assert calls == [1] and (run_dir / "evaluation.json").read_text() == first

    # an evaluation that finished first while this one ran: the write itself refuses, the file is kept
    (run_dir / "evaluation.json").unlink()

    def racing(omnisafe_dir, spec_dict):
        (run_dir / "evaluation.json").write_text('{"marker": "the other one"}')
        return _good_evaluation(spec)

    monkeypatch.setattr(contracts, "load_evaluator", lambda s: racing)
    with pytest.raises(launch.RunRefused, match="evaluated twice"):
        launch.evaluate(spec, run_dir, allow_dirty=True)
    assert json.loads((run_dir / "evaluation.json").read_text()) == {"marker": "the other one"}
    assert [p.name for p in run_dir.iterdir() if p.name.endswith(".tmp")] == []


def test_continuations_are_never_evaluated_by_the_evaluation_stage(tmp_path) -> None:
    fewshot = manifest.design("study_b_fewshot")[0]
    run_dir = _trained_run(tmp_path, fewshot)
    with pytest.raises(launch.RunRefused, match="continuation"):
        launch.evaluate(fewshot, run_dir, allow_dirty=True)


def test_the_training_stage_sets_the_flag_for_its_process(tmp_path, monkeypatch) -> None:
    seen: list[bool] = []

    def fake_train(spec, run_dir, *, allow_dirty, allow_pending):
        seen.append(errors.pending_allowed())
        return launch.EXIT_COMPLETED

    monkeypatch.setattr(launch, "train", fake_train)
    (tmp_path / "spec.json").write_text(_det_spec().to_json())
    assert launch.main(_argv(tmp_path, "train", "--allow-pending")) == launch.EXIT_COMPLETED
    assert launch.main(_argv(tmp_path, "train")) == launch.EXIT_COMPLETED
    assert seen == [True, False] and not errors.pending_allowed()


# ---------------------------------------------------------------------------
# Dependencies before the claim; pilot_cfgs
# ---------------------------------------------------------------------------


def test_pilot_cfgs_hold_the_run_facts() -> None:
    late = next(s for s in manifest.design("treatment") if s.treatment == "injection" and s.N == 0.25)
    cfg = launch.pilot_config(late)
    assert cfg == {
        "run_id": late.run_id, "onset_step": 2_500_000, "study": "A", "plugin": "study_a", "group": "treatment",
        "treatment": "injection", "controller_variant": None, "extra_checkpoint_steps": [2_500_000, 2_700_000],
        "plasticity": True, "dependencies": {},
    }
    fewshot = manifest.design("study_b_fewshot")[0]
    with pytest.raises(launch.RunRefused, match="dependencies"):
        launch.pilot_config(fewshot)  # a continuation's parent must be resolved first
    assert launch.pilot_config(_det_spec())["plasticity"] is False
    assert json.loads(json.dumps(cfg)) == cfg


@pytest.mark.parametrize("group, variant", [("treatment", "injection"), ("treatment", "reset"),
                                            ("controller", "rate_limited"), ("controller", "warm_started"),
                                            ("pid", R.PID_VARIANT)])
def test_a_continuation_applies_no_treatment_or_controller_of_its_parent(group, variant, tmp_path) -> None:
    """Its spec copies the parent's treatment and controller for bookkeeping (the cut order); the run
    facts that plug-ins and hooks read say None (Table 2.2: one condition for every matched arm)."""
    from test_core_dependencies import COMMIT, _continuation, make_run

    field = "treatment" if group == "treatment" else "controller_variant"
    parent = next(s for s in manifest.design(group) if getattr(s, field) == variant and s.seed == 0)
    omni = make_run(tmp_path, parent)
    tiny = _continuation(parent, omni).params  # the fixture checkpoint's step (2 * TINY_EPOCH) and path
    row = {"run_id": parent.run_id, "matched": True, "completed": True, "matched_checkpoint_step": 9_600_000,
           "matched_checkpoint_path": "x", "commit_hash": COMMIT}
    for condition in ("finetune", "transfer"):
        built = manifest.battery_continuation(parent, row, condition)
        spec = RunSpec.from_dict({**built.to_dict(), "params": {
            **built.params, "parent_step": tiny["parent_step"], "parent_checkpoint": tiny["parent_checkpoint"]}})
        assert getattr(spec, field) == variant  # bookkeeping: the spec keeps the parent's
        cfg = launch.pilot_config(spec, dependencies.resolve(spec, tmp_path / spec.run_id))
        assert (cfg["treatment"], cfg["controller_variant"], cfg["onset_step"]) == (None, None, 0), condition
        assert (cfg["plasticity"], cfg["extra_checkpoint_steps"], cfg["group"]) == (False, [], f"battery_{condition}")
    if not parent.depends_on:  # the parent itself keeps its run facts
        assert launch.pilot_config(parent)[field] == variant


def test_a_dependency_that_is_not_ready_is_refused_before_the_claim(tmp_path, capsys) -> None:
    warm = next(s for s in manifest.design("controller") if s.controller_variant == "warm_started")
    run_dir = tmp_path / "checkpoints" / warm.run_id
    run_dir.mkdir(parents=True)
    (run_dir / "spec.json").write_text(warm.to_json())
    assert launch.main(_argv(run_dir, "train", "--allow-pending")) == launch.EXIT_REFUSED
    assert warm.depends_on[0] in capsys.readouterr().err
    assert sorted(p.name for p in run_dir.iterdir()) == ["spec.json"]  # no claim, no output


def test_a_smoke_dependency_is_refused_before_the_claim_of_a_registered_run(tmp_path, monkeypatch, capsys) -> None:
    """A clean run must not start from a smoke run: the smoke markers would not reach its own record."""
    from test_core_dependencies import make_run

    parent = next(s for s in manifest.design("main") if s.task == R.PRIMARY_TASK and s.N == 0.0 and s.seed == 0)
    warm = next(s for s in manifest.design("controller") if s.controller_variant == "warm_started" and s.seed == 0
                and s.task == R.PRIMARY_TASK and s.N == 0.25)
    root = tmp_path / "checkpoints"
    make_run(root, parent)
    result = json.loads((root / parent.run_id / "train_result.json").read_text())
    (root / parent.run_id / "train_result.json").write_text(json.dumps({**result, "allow_pending": True}))
    run_dir = root / warm.run_id
    run_dir.mkdir()
    (run_dir / "spec.json").write_text(warm.to_json())
    monkeypatch.setattr(R, "is_open", lambda key: False)  # the questions are answered: a registered launch
    monkeypatch.setattr(provenance, "require_clean_worktree", lambda *a, **k: "c" * 40)
    argv = ["train", "--spec", str(run_dir / "spec.json"), "--run-dir", str(run_dir)]
    assert launch.main(argv) == launch.EXIT_REFUSED
    assert "smoke run" in capsys.readouterr().err
    assert sorted(p.name for p in run_dir.iterdir()) == ["spec.json"]  # no claim, no output


@pytest.mark.omnisafe
def test_the_launcher_passes_and_records_the_dependencies(tmp_path, monkeypatch) -> None:
    pytest.importorskip("omnisafe")
    import pilot.algorithms as algorithms
    from test_core_dependencies import TINY_EPOCH, make_run

    parent = next(s for s in manifest.design("main") if s.task == R.PRIMARY_TASK and s.N == 0.0 and s.seed == 0)
    warm = next(s for s in manifest.design("controller") if s.controller_variant == "warm_started" and s.seed == 0
                and s.task == R.PRIMARY_TASK and s.N == 0.25)
    seen = {}

    class FakeAlgo:  # stands in for the warm-started plug-in: reads its dependency, trains nothing
        def learn(self) -> None:
            pass

    def factory(task, cfgs, spec):
        seen["cfgs"] = cfgs
        seen["multiplier"] = dependencies.from_cfgs(cfgs)[parent.run_id].multiplier_at(TINY_EPOCH)
        return FakeAlgo()

    monkeypatch.setattr(algorithms, "load_plugin", lambda key: factory)
    hashes = []
    for root in (tmp_path / "a", tmp_path / "b"):
        make_run(root / "checkpoints", parent)
        if root.name == "b":  # a smoke parent: accepted, because this run is a smoke run too (flags below)
            path = root / "checkpoints" / parent.run_id / "train_result.json"
            path.write_text(json.dumps({**json.loads(path.read_text()), "allow_dirty": True, "worktree_dirty": True}))
        run_dir = root / "checkpoints" / warm.run_id
        run_dir.mkdir()
        (run_dir / "spec.json").write_text(warm.to_json())
        launch.train(warm, run_dir, allow_dirty=True, allow_pending=True)
        result = json.loads((run_dir / "train_result.json").read_text())
        block = result["dependencies"]
        assert block == seen["cfgs"].pilot_cfgs.dependencies.todict()
        assert block[parent.run_id]["run_dir"] == str(root / "checkpoints" / parent.run_id)
        assert seen["multiplier"] == 0.036
        resolved = dependencies.resolve(warm, run_dir, smoke=True)
        assert seen["cfgs"].pilot_cfgs.todict() == launch.pilot_config(warm, resolved)
        assert result["status"] == "failed"  # the fake trained nothing; the record is written all the same
        hashes.append(result["config_hash"])
    assert hashes[0] == hashes[1]  # the data root is not part of the configuration


# ---------------------------------------------------------------------------
# Result files that do not make the worktree dirty
# ---------------------------------------------------------------------------


def test_supplement_and_analysis_results_are_output_paths() -> None:
    for path in ("results/supplement/evaluation/A-x-s0.json", "results/pilot/supplement/battery/P-A-s0/hazard.json",
                 "results/analysis/final-20270101T000000Z/report.md", "results/analysis/pilot-x/tables/h1.csv",
                 "results/supplement/x.parquet"):
        assert provenance.is_output_path(path), path
    for path in ("results/analysis/run.py", "results/supplement/schema.py", "results/supplement_schema.py",
                 "results/analysis.json", "analysis/report.md", "results/supplement/../../pilot/x.json",
                 "results/pilot/supplement.py", "pilot/allocation.json"):
        assert not provenance.is_output_path(path), path
    # unchanged exemptions
    assert provenance.is_output_path("results/ledger.parquet") and provenance.is_output_path("results/pilot/go_report.md")


def test_dependency_paths_are_volatile_for_the_config_hash() -> None:
    def config(run_dir: str, sha: str = "1" * 64) -> dict:
        entry = {"run_dir": run_dir, "omnisafe_dir": run_dir + "/omnisafe/x", "commit_hash": "a" * 40,
                 "config_hash": "b" * 64, "total_steps": 10, "checkpoint_step": 0, "checkpoint_sha256": sha}
        return {"logger_cfgs": {"log_dir": run_dir}, "pilot_cfgs": {"dependencies": {"P-s0": entry, "Q-s0": dict(entry)}}}

    assert provenance.config_hash(config("/data/a")) == provenance.config_hash(config("/smoke/b"))
    assert provenance.config_hash(config("/data/a")) != provenance.config_hash(config("/data/a", sha="2" * 64))
    assert provenance.config_hash({"pilot_cfgs": {"dependencies": {}}}) != provenance.config_hash({"pilot_cfgs": {}})
    # a pilot_cfgs of None does not raise, and is neither a missing nor an empty block
    assert provenance.config_hash({"pilot_cfgs": None}) not in (provenance.config_hash({}),
                                                                provenance.config_hash({"pilot_cfgs": {}}))


def test_the_evaluation_stage_checks_the_code_after_the_record_check_imported_its_schema(tmp_path, monkeypatch) -> None:
    """The last check of the loaded code once ran before ``_check_evaluation_record`` imported
    results.supplement_schema, so an uncommitted schema validated the record unseen. The check now runs after it."""
    from pilot import contracts

    spec = _det_spec(R.TOTAL_STEPS)
    run_dir = _trained_run(tmp_path, spec)
    monkeypatch.setattr(contracts, "load_evaluator", lambda s: lambda omnisafe_dir, spec_dict: _good_evaluation(spec))
    monkeypatch.setattr(provenance, "require_clean_worktree", lambda *a, **k: "c" * 40)
    monkeypatch.setattr(provenance, "dirty_paths", lambda *a, **k: [])
    monkeypatch.delitem(sys.modules, "results.supplement_schema", raising=False)  # as in a fresh launcher process
    monkeypatch.setattr(provenance, "unverified_imported_code", lambda *a, **k: (
        ["results/supplement_schema.py (untracked)"] if "results.supplement_schema" in sys.modules else []))
    with pytest.raises(launch.RunRefused, match="supplement_schema"):
        launch.evaluate(spec, run_dir)
    assert not (run_dir / "evaluation.json").exists()


# ---------------------------------------------------------------------------
# Q-interrupted-run (Table 9.1): SIGHUP stops a run from outside, like SIGTERM and SIGINT
# ---------------------------------------------------------------------------


@pytest.mark.omnisafe
def test_sighup_ends_training_as_an_interruption(tmp_path, monkeypatch) -> None:
    """A closed terminal or ssh session (SIGHUP) is a machine event: the launcher records it as an interruption,
    which the scheduler restarts without limit, and restores the caller's handler."""
    import os
    import signal

    from test_launch_and_configs import _fake_run

    assert set(launch.INTERRUPT_SIGNALS) == {signal.SIGTERM, signal.SIGINT, signal.SIGHUP}

    def learn(run_dir: Path) -> None:
        os.kill(os.getpid(), signal.SIGHUP)
        for _ in range(1000):
            pass

    spec, run_dir = _fake_run(tmp_path, monkeypatch, learn)
    previous = signal.getsignal(signal.SIGHUP)
    code = launch.train(spec, run_dir, allow_dirty=True)
    assert signal.getsignal(signal.SIGHUP) is previous
    result = json.loads((run_dir / "train_result.json").read_text())
    assert code == launch.EXIT_INTERRUPTED
    assert (result["status"], result["interruption"]) == ("interrupted", f"signal {int(signal.SIGHUP)}")


@pytest.mark.omnisafe
def test_a_signal_inside_an_except_exception_still_interrupts_the_run(tmp_path, monkeypatch) -> None:
    """RunInterrupted once subclassed Exception: a SIGTERM that arrived inside a plug-in's or a hook's
    ``except Exception`` (Role 3's measurement at a checkpoint) was swallowed, the gate was spent, and the run
    trained on and was kept. It is a BaseException, so the run ends as an interruption (Q-interrupted-run)."""
    import os
    import signal

    from test_launch_and_configs import _fake_run

    assert not issubclass(launch.RunInterrupted, Exception)
    swallowed: list[str] = []
    trained_on: list[bool] = []

    def learn(run_dir: Path) -> None:
        try:  # as metrics.hook's record(): a measurement failure must not crash the run
            os.kill(os.getpid(), signal.SIGTERM)
            for _ in range(1000):
                pass
        except Exception as exc:  # noqa: BLE001 - the handler under test
            swallowed.append(repr(exc))
        trained_on.append(True)

    spec, run_dir = _fake_run(tmp_path, monkeypatch, learn)
    previous = signal.getsignal(signal.SIGTERM)
    code = launch.train(spec, run_dir, allow_dirty=True)
    assert signal.getsignal(signal.SIGTERM) is previous
    result = json.loads((run_dir / "train_result.json").read_text())
    assert (swallowed, trained_on) == ([], [])
    assert code == launch.EXIT_INTERRUPTED
    assert (result["status"], result["interruption"]) == ("interrupted", f"signal {int(signal.SIGTERM)}")

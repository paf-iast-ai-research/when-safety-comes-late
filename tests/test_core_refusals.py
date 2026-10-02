"""Refusals and missing modules are not run outcomes, in both launcher stages
(HANDOVER.md section 8; pilot/launch.py).

* A repository module (``envs``, ``metrics``, ``studyb``, ``analysis``) that is missing, or that
  exists but lacks the name imported from it (a plain ImportError, not ModuleNotFoundError: Role 3's
  ``metrics.hook`` without ``install_plasticity_hook`` yet), when a harness or a plug-in module is
  imported (at its top level or lazily), when the factory runs, or during training makes the stage
  unavailable (exit 4) and leaves the run as found. A missing library shows its real error in
  the evaluation stage and when a plug-in module is imported, is a refusal (exit 5) when the
  factory imports it (nothing has trained; a name missing from a library too), and is a crash like
  any other error during training (Part 5.6).
* A refusal raised by a plug-in or hook during training (an intervention refused at onset,
  HANDOVER.md section 8; a result gate inside a hook) exits 5 and leaves the run directory as found, so the run stays
  pending; the train log (stderr) records how many epochs were discarded and the traceback.
* Unless training had already gone non-finite (the log or the live state): then the run fails with
  that cause and keeps its output, as Part 5.6 excludes it (for example Role 3's injection refusing
  NaN weights at onset, metrics.interventions.plasticity_injection: torch.equal(nan, nan) is False).

The harnesses and plug-ins are real module files on a temporary ``sys.path`` entry, loaded through
the real ``pilot.contracts.load_evaluator`` and ``pilot.algorithms.load_plugin``; only the target
they are registered under changes. The ``python -m pilot.launch`` tests run the launcher as
``__main__``, as the scheduler starts it. The slow tests train a tiny PPO-Lagrangian (2,000 steps
per epoch, 2 epochs) that refuses after one trained epoch.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest

from configs import registered as R
from pilot import contracts, launch, manifest
from pilot.manifest import RunSpec
from test_core_launch import _det_spec, _trained_run

REPO = Path(__file__).resolve().parents[1]
MISSING_METRICS = "metrics.wpcore_not_written_yet"  # a submodule of Role 3's package, never written
MISSING_ENVS = "envs.wpcore_not_written_yet"
MISSING_LIBRARY = "wpcore_library_not_installed"
# A Role 3 module that exists but defines none of the names imported from it (autouse fixture below):
# ``from STUB_METRICS import x`` raises ImportError with ``name == STUB_METRICS``, as when a function
# another role is writing in parallel is not merged yet (HANDOVER.md section 8: metrics.interventions.rebuild_injected).
STUB_METRICS = "metrics.wpcore_stub"
OWNER = "Test fixture (no role)"


@pytest.fixture(autouse=True)
def stub_module(monkeypatch):
    monkeypatch.setitem(sys.modules, STUB_METRICS, types.ModuleType(STUB_METRICS))


def test_the_stub_raises_what_a_missing_name_raises() -> None:
    with pytest.raises(ImportError) as exc:
        exec(f"from {STUB_METRICS} import helper")
    assert not isinstance(exc.value, ModuleNotFoundError) and exc.value.name == STUB_METRICS
    assert launch.is_missing_repository_import(exc.value)
    with pytest.raises(ImportError) as library:
        exec("from json import wpcore_name_not_there")
    assert library.value.name == "json" and not launch.is_missing_repository_import(library.value)
    assert not launch.is_missing_repository_import(launch.PluginUnavailableError("already classified"))


@pytest.fixture
def modules(tmp_path, monkeypatch):
    """Write importable modules: ``modules(name, source)``; they are forgotten after the test."""
    root = tmp_path / "modules"
    root.mkdir()
    monkeypatch.syspath_prepend(str(root))
    written: list[str] = []

    def write(name: str, source: str) -> str:
        (root / f"{name}.py").write_text(textwrap.dedent(source))
        written.append(name)
        return name

    write.root = root  # type: ignore[attr-defined]
    yield write
    for name in written:
        sys.modules.pop(name, None)


def _untrained_run(tmp_path: Path, spec: RunSpec) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "spec.json").write_text(spec.to_json())
    return run_dir


def _argv(run_dir: Path, stage: str) -> list[str]:
    return [stage, "--spec", str(run_dir / "spec.json"), "--run-dir", str(run_dir), "--allow-dirty"]


def _left(run_dir: Path) -> list[str]:
    return sorted(p.name for p in run_dir.iterdir())


def _python_m_launch(modules_root: Path, argv: list[str], *, evaluator: str | None = None,
                     plugin: str | None = None) -> subprocess.CompletedProcess:
    """``python -m pilot.launch ARGV`` with the evaluator or the ppolag plug-in pointing at a test module."""
    driver = textwrap.dedent(f"""
        import runpy, sys, types
        sys.path.insert(0, {str(modules_root)!r})
        sys.modules[{STUB_METRICS!r}] = types.ModuleType({STUB_METRICS!r})
        from pilot import contracts, manifest
        if {evaluator!r}:
            contracts.EVALUATOR_TARGET = {evaluator!r}
        if {plugin!r}:
            manifest.PLUGINS["ppolag"] = ({plugin!r}, {OWNER!r})
        sys.argv = ["pilot.launch", *{argv!r}]
        runpy.run_module("pilot.launch", run_name="__main__", alter_sys=True)  # as python -m pilot.launch
    """)
    return subprocess.run([sys.executable, "-c", driver], cwd=REPO, capture_output=True, text=True)


# ---------------------------------------------------------------------------
# The evaluation stage
# ---------------------------------------------------------------------------

HARNESSES = {
    "imports it at its top level": (f"""
        from {MISSING_METRICS} import helper

        def evaluate_run(omnisafe_dir, spec):
            raise AssertionError("not reached")
        """, MISSING_METRICS),
    "imports it when called": (f"""
        def evaluate_run(omnisafe_dir, spec):
            from {MISSING_ENVS} import run_episodes
        """, MISSING_ENVS),
    "imports a name it lacks at its top level": (f"""
        from {STUB_METRICS} import helper

        def evaluate_run(omnisafe_dir, spec):
            raise AssertionError("not reached")
        """, STUB_METRICS),
    "imports a name it lacks when called": (f"""
        def evaluate_run(omnisafe_dir, spec):
            from {STUB_METRICS} import build_actor
            return build_actor
        """, STUB_METRICS),
}


@pytest.mark.parametrize("case", sorted(HARNESSES))
def test_a_harness_needing_a_repository_module_not_written_yet_is_unavailable(case, tmp_path, modules, monkeypatch,
                                                                               capsys) -> None:
    source, missing = HARNESSES[case]
    name = modules("wpcore_harness_" + case.replace(" ", "_"), source)
    spec = _det_spec(R.TOTAL_STEPS)
    run_dir = _trained_run(tmp_path, spec)
    monkeypatch.setattr(contracts, "EVALUATOR_TARGET", f"{name}:evaluate_run")
    assert launch.main(_argv(run_dir, "evaluate")) == launch.EXIT_UNAVAILABLE
    err = capsys.readouterr().err
    assert missing in err and ("cannot import name" in err) == ("name" in case)
    assert not (run_dir / "evaluation.json").exists()  # the run stays trained, not eval_failed


@pytest.mark.parametrize("when", ["top level", "when called"])
def test_a_harness_missing_a_library_shows_the_real_error(when, tmp_path, modules, monkeypatch) -> None:
    source = (f"import {MISSING_LIBRARY}\n\ndef evaluate_run(omnisafe_dir, spec):\n    pass\n" if when == "top level"
              else f"def evaluate_run(omnisafe_dir, spec):\n    import {MISSING_LIBRARY}\n")
    name = modules("wpcore_harness_library_" + when.replace(" ", "_"), source)
    run_dir = _trained_run(tmp_path, _det_spec(R.TOTAL_STEPS))
    monkeypatch.setattr(contracts, "EVALUATOR_TARGET", f"{name}:evaluate_run")
    with pytest.raises(ModuleNotFoundError) as exc:  # an environment problem is shown, not hidden as "unavailable"
        launch.main(_argv(run_dir, "evaluate"))
    assert exc.value.name == MISSING_LIBRARY


@pytest.mark.parametrize("case", ["imports it at its top level", "imports a name it lacks when called"])
def test_python_m_launch_evaluate_exits_4_for_a_harness_importing_a_missing_repository_module(case, tmp_path,
                                                                                              modules) -> None:
    source, missing = HARNESSES[case]
    name = modules("wpcore_harness_python_m_" + case.replace(" ", "_"), source)
    run_dir = _trained_run(tmp_path, _det_spec(R.TOTAL_STEPS))
    out = _python_m_launch(modules.root, _argv(run_dir, "evaluate"), evaluator=f"{name}:evaluate_run")
    assert out.returncode == launch.EXIT_UNAVAILABLE, out.stderr
    assert "UNAVAILABLE" in out.stderr and missing in out.stderr
    assert not (run_dir / "evaluation.json").exists()


def test_python_m_launch_exits_5_for_a_refusal_raised_outside_main(tmp_path, modules) -> None:
    """A RunRefused raised by another module (here a harness) is the class ``__main__`` catches
    (pilot/errors.py): were it defined again in pilot/launch.py, this would be a traceback."""
    name = modules("wpcore_harness_python_m_refuses", """
        from pilot.errors import RunRefused

        def evaluate_run(omnisafe_dir, spec):
            raise RunRefused("wpcore harness refusal")
        """)
    run_dir = _trained_run(tmp_path, _det_spec(R.TOTAL_STEPS))
    out = _python_m_launch(modules.root, _argv(run_dir, "evaluate"), evaluator=f"{name}:evaluate_run")
    assert out.returncode == launch.EXIT_REFUSED, out.stderr
    assert "REFUSED: wpcore harness refusal" in out.stderr
    assert not (run_dir / "evaluation.json").exists()


# ---------------------------------------------------------------------------
# The training stage: importing the plug-in, and its factory
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("missing", [MISSING_METRICS, STUB_METRICS])  # the module, or a name in it
def test_a_plug_in_importing_a_repository_module_not_written_yet_is_unavailable(missing, tmp_path, modules, monkeypatch,
                                                                                capsys) -> None:
    name = modules("wpcore_plugin_top_" + missing.replace(".", "_"), f"""
        from {missing} import helper

        def make(env_id, cfgs, spec):
            raise AssertionError("not reached")
        """)
    monkeypatch.setitem(manifest.PLUGINS, "ppolag", (f"{name}:make", OWNER))
    run_dir = _untrained_run(tmp_path, _det_spec())
    assert launch.main(_argv(run_dir, "train")) == launch.EXIT_UNAVAILABLE
    assert missing in capsys.readouterr().err
    assert _left(run_dir) == ["spec.json"]


def test_a_plug_in_importing_a_missing_library_shows_the_real_error(tmp_path, modules, monkeypatch) -> None:
    name = modules("wpcore_plugin_library", f"import {MISSING_LIBRARY}\n\ndef make(env_id, cfgs, spec):\n    pass\n")
    monkeypatch.setitem(manifest.PLUGINS, "ppolag", (f"{name}:make", OWNER))
    run_dir = _untrained_run(tmp_path, _det_spec())
    with pytest.raises(ModuleNotFoundError) as exc:  # before the claim: the scheduler's launch_failed, not a run outcome
        launch.main(_argv(run_dir, "train"))
    assert exc.value.name == MISSING_LIBRARY and _left(run_dir) == ["spec.json"]


@pytest.mark.parametrize("missing", [MISSING_ENVS, STUB_METRICS])
def test_python_m_launch_train_exits_4_for_a_plug_in_importing_a_missing_repository_module(missing, tmp_path,
                                                                                          modules) -> None:
    name = modules("wpcore_plugin_python_m_" + missing.replace(".", "_"), f"from {missing} import OnsetMixin\n")
    run_dir = _untrained_run(tmp_path, _det_spec())
    out = _python_m_launch(modules.root, _argv(run_dir, "train"), plugin=f"{name}:make")
    assert out.returncode == launch.EXIT_UNAVAILABLE, out.stderr
    assert "UNAVAILABLE" in out.stderr and missing in out.stderr
    assert _left(run_dir) == ["spec.json"]


@pytest.mark.omnisafe
@pytest.mark.parametrize("statement, missing, code", [
    (f"import {MISSING_METRICS}", MISSING_METRICS, launch.EXIT_UNAVAILABLE),
    (f"from {STUB_METRICS} import rebuild_injected", STUB_METRICS, launch.EXIT_UNAVAILABLE),  # a name it lacks
    (f"import {MISSING_LIBRARY}", MISSING_LIBRARY, launch.EXIT_REFUSED),
    ("from json import wpcore_name_not_there", "json", launch.EXIT_REFUSED),  # a library's version mismatch
])
def test_a_module_the_factory_cannot_import(statement, missing, code, tmp_path, modules, monkeypatch, capsys) -> None:
    """Nothing has trained: a repository module, or a name it lacks, is unavailable (exit 4), a library
    module, or a name it lacks, a refusal (exit 5)."""
    pytest.importorskip("omnisafe")
    name = modules("wpcore_plugin_factory_" + missing.replace(".", "_"), f"""
        from pathlib import Path

        def make(env_id, cfgs, spec):
            Path(cfgs.logger_cfgs.log_dir, "PPOLag-{{x}}").mkdir(parents=True)  # OmniSafe's logger may exist already
            {statement}
        """)
    monkeypatch.setitem(manifest.PLUGINS, "ppolag", (f"{name}:make", OWNER))
    run_dir = _untrained_run(tmp_path, _det_spec())
    assert launch.main(_argv(run_dir, "train")) == code
    err = capsys.readouterr().err
    assert missing in err and ("UNAVAILABLE" if code == launch.EXIT_UNAVAILABLE else "REFUSED") in err
    assert _left(run_dir) == ["spec.json"]  # claim and output removed: the run stays pending


@pytest.mark.omnisafe
@pytest.mark.parametrize("hook", ["module missing", "function missing"])
def test_a_study_a_run_whose_plasticity_hook_is_not_written_yet_is_unavailable(hook, tmp_path, monkeypatch,
                                                                               capsys) -> None:
    """The mixin's own lazy import (``_init_log``: ``from metrics.hook import install_plasticity_hook``,
    pilot/contracts.py contract 2) with the real ``make_ppolag``: Role 3's module not written yet, or written without the
    function, is exit 4 and leaves the run as found; never a crash that the scheduler would exclude
    (Part 5.6) and replace with the next seed."""
    pytest.importorskip("omnisafe")
    import pilot.algorithms as algorithms

    spec = next(s for s in manifest.pilot() if s.study == "A" and s.N == 0.0)
    assert launch.pilot_config(spec)["plasticity"]
    monkeypatch.setitem(sys.modules, "metrics.hook",
                        None if hook == "module missing" else types.ModuleType("metrics.hook"))
    monkeypatch.setattr(algorithms, "load_plugin", lambda key: algorithms.make_ppolag)  # any plug-in with the mixin
    run_dir = _untrained_run(tmp_path, spec)
    assert launch.main(_argv(run_dir, "train") + ["--allow-pending"]) == launch.EXIT_UNAVAILABLE
    err = capsys.readouterr().err
    assert "UNAVAILABLE" in err and "metrics.hook" in err
    assert ("cannot import name 'install_plasticity_hook'" in err) == (hook == "function missing")
    assert _left(run_dir) == ["spec.json"]


# ---------------------------------------------------------------------------
# The training stage: during learn()
# ---------------------------------------------------------------------------

# A plug-in whose learn() trains and logs two epochs, then reaches "the start of the onset epoch's rollout"
# (HANDOVER.md section 8).
LEARNING_PLUGIN = """
    from pathlib import Path


    class Algo:
        def __init__(self, cfgs):
            self._omni = Path(cfgs.logger_cfgs.log_dir) / "PPOLag-{{x}}" / "seed-000-t"

        def learn(self):
            (self._omni / "torch_save").mkdir(parents=True)
            (self._omni / "progress.csv").write_text("Train/Epoch,TotalEnvSteps\\n0,2000\\n1,4000\\n")
            (self._omni / "torch_save" / "epoch-0.pt").write_bytes(b"")
            at_onset()


    def at_onset():
    {body}


    def make(env_id, cfgs, spec):
        return Algo(cfgs)
    """

ONSET_BODIES = {
    "refusal": ("from pilot.errors import RunRefused\nraise RunRefused('an intervention record exists already (HANDOVER.md section 8)')",
                launch.EXIT_REFUSED, "RunRefused"),
    "result gate": ("from pilot.errors import require_answered\nrequire_answered('Q-hazard', what='a result gate in a hook')",
                    launch.EXIT_REFUSED, "PendingQuestionError"),
    "repository module": (f"from {MISSING_METRICS} import plasticity_injection", launch.EXIT_UNAVAILABLE,
                          "ModuleNotFoundError"),
    "repository name": (f"from {STUB_METRICS} import plasticity_injection", launch.EXIT_UNAVAILABLE, "ImportError"),
}


def _example_gate_open(monkeypatch) -> None:
    """The "result gate" case uses Q-hazard as its example: every key is answered (docs/DECISIONS.md, 2026-10-02), so
    the test pins it open, as a key the group changed would be."""
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", frozenset(R.ANSWERED_QUESTIONS) - {"Q-hazard"})


def _learning_plugin(modules, key: str, body: str) -> str:
    source = textwrap.dedent(LEARNING_PLUGIN).replace("{body}", textwrap.indent(body, "    "))
    return modules("wpcore_learning_" + key.replace(" ", "_"), source.replace("{{x}}", "{x}"))


@pytest.mark.omnisafe
@pytest.mark.parametrize("case", sorted(ONSET_BODIES))
def test_a_refusal_during_training_leaves_the_run_pending_and_logs_the_discarded_epochs(case, tmp_path, modules,
                                                                                         monkeypatch, capsys) -> None:
    pytest.importorskip("omnisafe")
    _example_gate_open(monkeypatch)
    body, code, raised = ONSET_BODIES[case]
    name = _learning_plugin(modules, case, body)
    monkeypatch.setitem(manifest.PLUGINS, "ppolag", (f"{name}:make", OWNER))
    run_dir = _untrained_run(tmp_path, _det_spec())
    assert launch.main(_argv(run_dir, "train")) == code
    err = capsys.readouterr().err
    assert f"{raised} raised during training after 2 logged epoch(s)" in err
    assert "Traceback" in err and "at_onset" in err  # where the refusal came from, although the output is gone
    assert _left(run_dir) == ["spec.json"]  # no claim, no output, no train_result.json: the run stays pending


@pytest.mark.omnisafe
@pytest.mark.parametrize("statement, missing", [(f"import {MISSING_LIBRARY}", MISSING_LIBRARY),
                                                ("from json import wpcore_name_not_there", "wpcore_name_not_there")])
def test_a_library_missing_during_training_is_still_a_crash(statement, missing, tmp_path, modules, monkeypatch) -> None:
    """pilot/launch.py exempts only repository modules (and names in them): any other error during training,
    a library module or a name missing from one included, is classified (Part 5.6)."""
    pytest.importorskip("omnisafe")
    name = _learning_plugin(modules, "library_" + missing, statement)
    monkeypatch.setitem(manifest.PLUGINS, "ppolag", (f"{name}:make", OWNER))
    run_dir = _untrained_run(tmp_path, _det_spec())
    assert launch.main(_argv(run_dir, "train")) == launch.EXIT_FAILED
    result = json.loads((run_dir / "train_result.json").read_text())
    assert (result["status"], result["failure_cause"]) == ("failed", "crash") and missing in result["detail"]


# ---------------------------------------------------------------------------
# The training stage: a refusal after training has gone non-finite is not discarded (Part 5.6)
# ---------------------------------------------------------------------------

# A plug-in whose second logged epoch (or live state) is non-finite when a refusal, a result gate or a
# missing repository module comes at "the start of the onset epoch's rollout" (HANDOVER.md section 8).
# Part 5.6 excludes such a run; discarding it would relaunch it into the same NaN (same seed, Table 3.1)
# and never exclude it.
NONFINITE_PLUGIN = """
    import math
    from pathlib import Path
    from types import SimpleNamespace

    import torch

    LOSS = {loss!r}
    MULTIPLIER = {multiplier!r}
    STATE = {state!r}


    class Algo:
        def __init__(self, cfgs):
            self._omni = Path(cfgs.logger_cfgs.log_dir) / "PPOLag-{{x}}" / "seed-000-t"
            self._actor_critic = torch.nn.Module()
            self._actor_critic.weight = torch.nn.Parameter(torch.zeros(3))  # no draw from any random stream

        def learn(self):
            (self._omni / "torch_save").mkdir(parents=True)
            (self._omni / "progress.csv").write_text(
                "Train/Epoch,TotalEnvSteps,Loss/Loss_pi,Metrics/LagrangeMultiplier\\n"
                f"0,2000,0.1,0.001\\n1,4000,{{LOSS}},{{MULTIPLIER}}\\n")
            (self._omni / "torch_save" / "epoch-0.pt").write_bytes(b"")
            if STATE == "weights":  # the NaN update reached the weights before the next epoch was logged
                with torch.no_grad():
                    self._actor_critic.weight.fill_(math.nan)
            elif STATE == "level multiplier":  # Study B's per-level multipliers count too (pilot/launch.py)
                self._level_lagranges = {{10.0: SimpleNamespace(lagrangian_multiplier=torch.tensor(math.nan))}}
            at_onset()


    def at_onset():
    {body}


    def make(env_id, cfgs, spec):
        return Algo(cfgs)
    """

NONFINITE_CASES = {
    # case: (loss logged in epoch 1, multiplier logged in epoch 1, live state, expected failure cause)
    "nan loss in the log": ("nan", "0.002", None, "non_finite_loss"),
    "inf multiplier in the log": ("0.1", "inf", None, "non_finite_multiplier"),
    "nan weights after a finite log": ("0.1", "0.002", "weights", "non_finite_loss"),
    "nan level multiplier after a finite log": ("0.1", "0.002", "level multiplier", "non_finite_multiplier"),
}


def _nonfinite_plugin(modules, key: str, onset: str, loss: str, multiplier: str, state: str | None) -> str:
    source = textwrap.dedent(NONFINITE_PLUGIN).format(loss=loss, multiplier=multiplier, state=state, body="{body}")
    source = source.replace("{body}", textwrap.indent(ONSET_BODIES[onset][0], "    "))
    return modules("wpcore_nonfinite_" + (key + "_" + onset).replace(" ", "_"), source)


@pytest.mark.omnisafe
@pytest.mark.parametrize("onset", sorted(ONSET_BODIES))
@pytest.mark.parametrize("case", sorted(NONFINITE_CASES))
def test_a_refusal_after_training_went_non_finite_excludes_the_run(case, onset, tmp_path, modules, monkeypatch,
                                                                    capsys) -> None:
    pytest.importorskip("omnisafe")
    _example_gate_open(monkeypatch)
    loss, multiplier, state, cause = NONFINITE_CASES[case]
    name = _nonfinite_plugin(modules, case, onset, loss, multiplier, state)
    monkeypatch.setitem(manifest.PLUGINS, "ppolag", (f"{name}:make", OWNER))
    run_dir = _untrained_run(tmp_path, _det_spec())
    assert launch.main(_argv(run_dir, "train")) == launch.EXIT_FAILED  # not EXIT_REFUSED or EXIT_UNAVAILABLE
    result = json.loads((run_dir / "train_result.json").read_text())
    assert (result["status"], result["failure_cause"]) == ("failed", cause)
    assert ONSET_BODIES[onset][2] in result["detail"]  # the refusal is kept in the record
    assert "omnisafe" in _left(run_dir) and result["omnisafe_dir"]  # the output is kept, not discarded
    assert "not discarded" in capsys.readouterr().err


@pytest.mark.omnisafe
def test_the_injection_refusing_nan_weights_at_onset_excludes_the_run(tmp_path, modules, monkeypatch) -> None:
    """Role 3's injection (metrics.interventions.plasticity_injection) compares the actor's output before and after
    with torch.equal, which is False for NaN, so it refuses (InterventionError, a RunRefused) at onset."""
    pytest.importorskip("omnisafe")
    interventions = pytest.importorskip("metrics.interventions")
    assert issubclass(interventions.InterventionError, launch.RunRefused)
    body = textwrap.dedent("""
        import numpy as np
        from gymnasium.spaces import Box
        from omnisafe.models.actor.actor_builder import ActorBuilder
        from metrics import interventions
        with torch.random.fork_rng():
            actor = ActorBuilder(obs_space=Box(-1, 1, (4,), np.float32), act_space=Box(-1, 1, (2,), np.float32),
                                 hidden_sizes=[8, 8], activation="tanh").build_actor(actor_type="gaussian_learning")
        with torch.no_grad():
            actor.mean[0].weight.fill_(math.nan)  # the NaN update of the last epoch reached the weights
        optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
        interventions.plasticity_injection(actor, optimizer, interventions.intervention_generator(0),
                                           batch=np.zeros((3, 4), np.float32))
        """)
    source = textwrap.dedent(NONFINITE_PLUGIN).format(loss="nan", multiplier="0.002", state=None, body="{body}")
    name = modules("wpcore_nonfinite_injection", source.replace("{body}", textwrap.indent(body, "    ")))
    monkeypatch.setitem(manifest.PLUGINS, "ppolag", (f"{name}:make", OWNER))
    run_dir = _untrained_run(tmp_path, _det_spec())
    assert launch.main(_argv(run_dir, "train")) == launch.EXIT_FAILED
    result = json.loads((run_dir / "train_result.json").read_text())
    assert (result["status"], result["failure_cause"]) == ("failed", "non_finite_loss")
    assert "InterventionError" in result["detail"]


# ---------------------------------------------------------------------------
# Slow: a tiny real PPO-Lagrangian that refuses after one trained epoch
# ---------------------------------------------------------------------------


@pytest.mark.omnisafe
@pytest.mark.slow
@pytest.mark.parametrize("case", ["refusal", "repository module", "repository name"])
def test_a_tiny_run_refused_at_its_second_epoch_leaves_nothing_behind(case, tmp_path, monkeypatch, capsys) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("omnisafe")
    from omnisafe.algorithms.on_policy.naive_lagrange.ppo_lag import PPOLag

    import pilot.algorithms as algorithms
    from pilot.errors import RunRefused

    assert STUB_METRICS == "metrics.wpcore_stub"  # the module the "repository name" case imports from
    epochs_logged: list[int] = []

    class RefusingAtOnset(algorithms.FullStateCheckpointMixin, PPOLag):
        def _update(self) -> None:
            if self._logger.current_epoch == 1:  # at the update of the second epoch, after one trained and logged
                # epoch (stands in for onset)
                epochs_logged.append(len(launch.read_progress(launch.find_omnisafe_run_dir(run_dir))))
                if case == "refusal":
                    raise RunRefused("an intervention record exists already (HANDOVER.md section 8)")
                if case == "repository name":  # the module exists, the function is not written yet
                    from metrics.wpcore_stub import plasticity_injection  # STUB_METRICS (asserted above)

                    plasticity_injection()
                importlib.import_module(MISSING_METRICS)  # a lazy import at onset
            super()._update()

    def factory(env_id, cfgs, spec):
        cfgs.algo_cfgs.steps_per_epoch = 2_000
        cfgs.train_cfgs.total_steps = 4_000
        cfgs.train_cfgs.epochs = 2
        cfgs.logger_cfgs.use_tensorboard = False
        return RefusingAtOnset(env_id=env_id, cfgs=cfgs)

    monkeypatch.setattr(algorithms, "load_plugin", lambda key: factory)
    threads = torch.get_num_threads()
    run_dir = _untrained_run(tmp_path, _det_spec())
    try:
        code = launch.main(_argv(run_dir, "train"))
    finally:
        torch.set_num_threads(threads)
    assert code == (launch.EXIT_REFUSED if case == "refusal" else launch.EXIT_UNAVAILABLE)
    assert epochs_logged == [1]  # one epoch was trained and logged before the refusal
    err = capsys.readouterr().err
    assert "raised during training after 1 logged epoch(s)" in err and "_update" in err
    assert _left(run_dir) == ["spec.json"]

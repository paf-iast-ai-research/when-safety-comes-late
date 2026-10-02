"""Stand-in for ``python -m pilot.launch``: the scheduler, launcher, ledger-writer, enrichment and integration tests.

Writes the same files the real launcher writes (the ``.claim`` file, OmniSafe-like progress.csv
and torch_save files, plasticity.csv for the Study A plug-ins, train_result.json, evaluation.json)
without training anything, and uses the real launcher's exit codes. The outcome of each run is
read from the JSON file named by $FAKE_SCENARIOS: {run_id: scenario}; the recorded commit is
$FAKE_COMMIT if set. Scenarios:

  success          training and evaluation complete
  crash            training fails with an exception (failure_cause 'crash')
  nonfinite        the multiplier becomes NaN (failure_cause 'non_finite_multiplier')
  interrupt        the process is killed without a final result (SIGKILL, e.g. the OOM killer)
  exit1            the process exits with code 1 after its claim, without a final result (an
                   uncaught launcher error)
  sigterm          the launcher is stopped by SIGTERM and records status 'interrupted' (exit 3)
  sighup           the process dies of SIGHUP without a final result (stopped from outside before the
                   launcher recorded it)
  unavailable      exit code 4 before the claim: the plug-in is not in the repository yet
  refused          exit code 5 before the claim: the launcher refuses to start the run
  noclaim          exit code 1 before the claim: an unexpected launcher error
  segv             the process kills itself with SIGSEGV after claiming the directory
  hang             sleeps until the file $FAKE_RELEASE exists, then succeeds
  evalfail         training completes; evaluation exits with an error
  evalunavailable  training completes; evaluation exits with code 4 (no evaluation harness yet)
  evalrefused      training completes; evaluation exits with code 5 (refused by the launcher)

In every scenario whose evaluation would complete, a run too short to have the ten checkpoints of its
selection window (``selection_window`` raises IncompleteWindow, as ``pilot.contracts.selection_window``
raises ContractError) is not evaluated: the evaluate stage exits with code 5, one of the ends a real
smoke run that short may have (scripts/smoke_run.py: eval_failed or refused).

evaluation.json is what ``pilot.launch.evaluate`` writes from the harnesses' contract-1 result
(``envs.evaluation.evaluate_run``, ``studyb.evaluation.evaluate_run``; HANDOVER.md section 8; studyb/conditioning.py): the means,
and the per-episode ``episode_costs`` and ``episode_returns`` ({"final": [100], "selection": {step:
[100]}}, integer costs whose means are the means given), ``final_step``, ``final_seed_set``
("measurement", Q-final-cost-set, answered in Table 9.1), ``measurement_seeds``, ``short_episodes``,
``final_selection_cost``, ``studyb_budgets`` for plug-in study_b (Q-studyb-eval, answered in Table 9.1;
``envs.evaluation.training_budget_schedule``), and the launcher's ``eval_commit_hash`` and smoke
markers, so that the ledger writer writes each row's ``evaluation`` supplement record (results/supplement_schema.py).

As the real launcher does (pilot/launch.py, pilot/dependencies.py; pilot.contracts.extra_checkpoint_steps; pilot/manifest.py):

* a run with ``depends_on`` is refused (exit 5, before the claim) unless each dependency's sibling
  run directory holds a completed train_result.json; train_result.json records the dependencies
  (``dependencies``: commit, config hash, total steps, and for a continuation's parent the
  checkpoint step and the sha256 of that checkpoint file);
* checkpoints are saved at step 0, every ten epochs, at the end, and at the extra steps of
  ``pilot.contracts.extra_checkpoint_steps`` (the onset, onset + 200,000 for the Study A plug-ins,
  the few-shot horizons, the end-relative selection window of an off-grid total that is not a
  continuation), with a plasticity.csv row for every one of them for the Study A runs of plug-ins
  study_a and study_a_pid (contract 2; ``pilot.contracts.plasticity_required``);
* a continuation (``CONTINUATION_GROUPS``) is never evaluated by this stage (exit 5); the scheduler
  ends it ``continued`` after training. The fine-tuning continuation logs its multiplier held at 0
  (Table 2.2: "the multiplier frozen at zero").
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import statistics
import sys
import time
from pathlib import Path

STEPS_PER_EPOCH = 20_000  # R.STEPS_PER_EPOCH
CHECKPOINT_INTERVAL_STEPS = 200_000  # R.CHECKPOINT_INTERVAL_STEPS; repeated so the fake imports nothing of the package
SELECTION_WINDOW_CHECKPOINTS = 10  # R.SELECTION_WINDOW_CHECKPOINTS; repeated likewise
MANIPULATION_CHECK_STEPS_AFTER_ONSET = 200_000  # R.MANIPULATION_CHECK_STEPS_AFTER_ONSET
COMMIT = "0123456789abcdef0123456789abcdef01234567"
EVAL_EPISODES = 100  # R.EVAL_EPISODES
SELECTION_SEEDS = [1_000_000 + i for i in range(EVAL_EPISODES)]  # envs.evaluation.selection_seeds()
# envs.evaluation.measurement_seeds(): the final checkpoint's set (Q-final-cost-set); the battery's measurement set
MEASUREMENT_SEEDS = [2_000_000 + i for i in range(EVAL_EPISODES)]
CONTINUOUS_RANGE = (10.0, 40.0)  # R.CONTINUOUS_RANGE
# The real launcher's exit codes (pilot.launch); repeated here so that the fake imports nothing of the package.
EXIT_COMPLETED, EXIT_FAILED, EXIT_INTERRUPTED, EXIT_UNAVAILABLE, EXIT_REFUSED = 0, 2, 3, 4, 5
CLAIM_FILE = ".claim"
CONTINUATION_GROUPS = ("study_b_fewshot", "battery_finetune", "battery_transfer")  # pilot.manifest.CONTINUATION_GROUPS
PLASTICITY_PLUGINS = ("study_a", "study_a_pid")  # pilot.contracts.PLASTICITY_PLUGINS
# metrics.plasticity.COLUMNS: the header the Role 3 hook writes (contract 2 requires its first four).
PLASTICITY_COLUMNS = (
    "step", "dormant", "rank", "norm", "norm_reward_critic", "norm_cost_critic", "dormant_trainable",
    "rank_trainable", "dormant_reward_critic", "rank_reward_critic", "dormant_cost_critic", "rank_cost_critic",
    "norm_all",
)


def extra_checkpoint_steps(spec: dict) -> list[int]:
    """``pilot.contracts.extra_checkpoint_steps`` for a spec dict (tests check the two agree)."""
    total = spec["total_steps"]
    onset = spec.get("onset_step") or 0
    candidates = set()
    if onset > 0:
        candidates.add(onset)
        if spec["plugin"] in PLASTICITY_PLUGINS:
            candidates.add(onset + MANIPULATION_CHECK_STEPS_AFTER_ONSET)
    if spec["plugin"] == "study_b_fewshot":
        candidates.update(int(h) for h in (spec.get("params") or {}).get("horizons") or ())
    # the end-relative selection window (Q-selection-window, answered in Table 9.1); a continuation is not selected from
    if total % CHECKPOINT_INTERVAL_STEPS and spec.get("group") not in CONTINUATION_GROUPS:
        candidates.update(total - k * CHECKPOINT_INTERVAL_STEPS for k in range(1, SELECTION_WINDOW_CHECKPOINTS))
    return sorted(s for s in candidates if 0 < s < total and s % CHECKPOINT_INTERVAL_STEPS)


def checkpoint_steps(spec: dict) -> list[int]:
    """Every step the run saves: 0, the ten-epoch grid, the end, and the extra steps (pilot.contracts.extra_checkpoint_steps)."""
    total = spec["total_steps"]
    steps = set(range(0, total + 1, CHECKPOINT_INTERVAL_STEPS)) | {total} | set(extra_checkpoint_steps(spec))
    return sorted(steps)


class IncompleteWindow(Exception):
    """The run did not save the ten checkpoints of its selection window (``pilot.contracts.ContractError``)."""


def selection_window(spec: dict, steps: list[int]) -> list[int]:
    """``pilot.contracts.selection_window`` of the saved ``steps`` (tests check the two agree).

    For a total off the checkpoint grid, the end-relative window (Q-selection-window as answered in
    Table 9.1, total - k x 200,000); otherwise the last ten saved checkpoints. IncompleteWindow if fewer than ten
    checkpoints were saved or a step of the end-relative window was not.
    """
    total = spec["total_steps"]
    if len(steps) < SELECTION_WINDOW_CHECKPOINTS:
        raise IncompleteWindow(f"only {len(steps)} checkpoints; the selection rule needs {SELECTION_WINDOW_CHECKPOINTS}")
    if total % CHECKPOINT_INTERVAL_STEPS:
        window = [total - k * CHECKPOINT_INTERVAL_STEPS for k in range(SELECTION_WINDOW_CHECKPOINTS - 1, -1, -1)]
        missing = sorted(set(window) - set(steps))
        if missing:
            raise IncompleteWindow(f"the end-relative selection window {window} lacks the checkpoints at steps {missing}")
        return window
    return steps[-SELECTION_WINDOW_CHECKPOINTS:]


def plasticity_row(step: int) -> list[str]:
    """Deterministic, plausible metrics (a fraction, a rank, norms) for one checkpoint."""
    k = step // STEPS_PER_EPOCH
    values = {
        "dormant": 0.05 + 0.0005 * (k % 100), "rank": 40 - (k % 10), "norm": 12.5 + 0.01 * k,
        "norm_reward_critic": 20.0 + 0.01 * k, "norm_cost_critic": 21.0 + 0.01 * k,
        "dormant_trainable": 0.04 + 0.0005 * (k % 100), "rank_trainable": 30 - (k % 10),
        "dormant_reward_critic": 0.03, "rank_reward_critic": 50, "dormant_cost_critic": 0.02, "rank_cost_critic": 51,
        "norm_all": 12.5 + 0.01 * k,
    }
    return [str(step)] + [f"{values[c]:g}" for c in PLASTICITY_COLUMNS[1:]]


def write_outputs(spec: dict, run_dir: Path, nan_multiplier: bool = False) -> str:
    """Write the run's OmniSafe-like output (progress.csv, checkpoints, plasticity.csv); return its path in run_dir."""
    epochs = spec["total_steps"] // STEPS_PER_EPOCH
    omni = run_dir / "omnisafe" / f"{spec['base_algo']}-{{{spec['task']}}}" / "seed-000-2026-01-01-00-00-00"
    (omni / "torch_save").mkdir(parents=True, exist_ok=True)
    # Study B logs one multiplier per training level (contract 3); every other run logs one.
    levels = spec.get("training_levels") if spec.get("study") == "B" else None
    names = [f"Metrics/LagrangeMultiplier/level_{float(b):g}" for b in levels] if levels else ["Metrics/LagrangeMultiplier"]
    frozen_zero = spec["plugin"] == "battery_finetune"  # Table 2.2: "the multiplier frozen at zero"
    with open(omni / "progress.csv", "w", encoding="utf-8") as fh:
        fh.write("Metrics/EpRet,Metrics/EpCost,Train/Epoch,TotalEnvSteps,Loss/Loss_pi," + ",".join(names) + "\n")
        for e in range(epochs):
            lam = "0.0" if frozen_zero else f"{0.001 + 0.035 * (e + 1):.6f}"
            lam = "nan" if nan_multiplier and e == epochs - 1 else lam
            fh.write(f"{10 + e * 0.1},{30 - e * 0.05},{e},{(e + 1) * STEPS_PER_EPOCH},0.1," + ",".join([lam] * len(names)) + "\n")
    steps = checkpoint_steps(spec)
    for step in steps:
        (omni / "torch_save" / f"epoch-{step // STEPS_PER_EPOCH}.pt").write_bytes(b"")
    if spec["plugin"] in PLASTICITY_PLUGINS and spec.get("study") == "A":
        with open(omni / "plasticity.csv", "w", encoding="utf-8") as fh:
            fh.write(",".join(PLASTICITY_COLUMNS) + "\n")
            for step in steps:
                fh.write(",".join(plasticity_row(step)) + "\n")
    return str(omni.relative_to(run_dir))


def episodes(mean: float, n: int = EVAL_EPISODES) -> list[float]:
    """n integer episode values whose mean is ``mean`` (a multiple of 1/n), as a 0/1 per-step cost gives."""
    total = round(mean * n)
    q, r = divmod(total, n)
    return [float(q + 1)] * r + [float(q)] * (n - r)


def fmean(values: list[float]) -> float:
    """``statistics.fmean``, the mean the supplement schema checks (results/supplement_schema.py)."""
    return statistics.fmean(values)


def studyb_budgets(spec: dict) -> list[float]:
    """``envs.evaluation.training_budget_schedule`` for a spec dict (Q-studyb-eval, answered in Table 9.1)."""
    levels = spec.get("training_levels")
    if levels is None:
        lo, hi = CONTINUOUS_RANGE
        return [lo + (hi - lo) * (i + 0.5) / EVAL_EPISODES for i in range(EVAL_EPISODES)]
    levels = sorted(float(b) for b in levels)
    return [levels[i % len(levels)] for i in range(EVAL_EPISODES)]


def evaluation_result(spec: dict, window: list[int], selection_costs: list[float], *, final_cost: float = 24.0,
                      final_return: float = 20.0, selection_return: float = 20.0, commit: str | None = None) -> dict:
    """evaluation.json as ``pilot.launch.evaluate`` writes it from a harness's contract-1 result.

    ``window`` are the selection window's ten checkpoint steps, ``selection_costs`` their selection-set mean costs
    (multiples of 0.01). Every mean is the ``fmean`` of its 100 episodes, as the harnesses compute it.
    """
    window = list(window)
    sel_costs = {str(s): episodes(c) for s, c in zip(window, selection_costs)}
    sel_returns = {str(s): episodes(selection_return) for s in window}
    final_costs, final_returns = episodes(final_cost), episodes(final_return)
    evaluation = {
        "final_cost": fmean(final_costs), "final_return": fmean(final_returns),
        "final_selection_cost": fmean(sel_costs[str(window[-1])]) if window else None,
        "final_step": spec["total_steps"], "final_seed_set": "measurement",
        "selection": {s: [fmean(sel_costs[s]), fmean(sel_returns[s])] for s in sel_costs},
        "episodes": EVAL_EPISODES, "selection_seeds": SELECTION_SEEDS, "measurement_seeds": MEASUREMENT_SEEDS,
        "episode_costs": {"final": final_costs, "selection": sel_costs},
        "episode_returns": {"final": final_returns, "selection": sel_returns},
        "short_episodes": 0, "harness": {"fake": True}, "eval_wall_clock_hours": 0.25,
        "eval_commit_hash": commit or os.environ.get("FAKE_COMMIT", COMMIT),
        "eval_allow_dirty": False, "eval_allow_pending": False, "eval_worktree_dirty": False,
    }
    if spec["plugin"] == "study_b":
        evaluation["studyb_budgets"] = studyb_budgets(spec)
    return evaluation


def train_result(spec: dict, omni: str | None, status: str, cause: str | None, dependencies: dict | None = None) -> dict:
    """train_result.json after training: the fields of the real launcher's record (pilot/launch.py ``train``)
    that the scheduler, enrichment and the ledger writer read. The smoke markers (``worktree_dirty``,
    ``allow_dirty``, ``allow_pending``) are left out, which every reader takes as false."""
    return {
        "run_id": spec["run_id"], "status": status, "failure_cause": cause, "detail": "fake",
        "commit_hash": os.environ.get("FAKE_COMMIT", COMMIT), "config_hash": "c" * 64, "machine": "fake-machine",
        "started": "2026-01-01T00:00:00+00:00", "finished": "2026-01-01T01:00:00+00:00",
        "wall_clock_hours": 1.0, "omnisafe_dir": omni, "versions": {}, "dependencies": dependencies or {},
    }


def resolve_dependencies(spec: dict, run_dir: Path) -> dict | None:
    """``pilot.dependencies.to_config`` for the fake's layout, or None when a dependency is not a completed run."""
    out = {}
    params = spec.get("params") or {}
    for dep_id in spec.get("depends_on") or ():
        dep_dir = run_dir.parent / dep_id
        try:
            train = json.loads((dep_dir / "train_result.json").read_text(encoding="utf-8"))
            dep_spec = json.loads((dep_dir / "spec.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if train.get("status") != "completed" or not train.get("omnisafe_dir"):
            return None
        omni = dep_dir / train["omnisafe_dir"]
        entry = {"run_dir": str(dep_dir), "omnisafe_dir": str(omni), "commit_hash": train.get("commit_hash"),
                 "config_hash": train.get("config_hash"), "total_steps": dep_spec["total_steps"]}
        if dep_id == params.get("parent_run_id") and params.get("parent_step") is not None:
            checkpoint = omni / "torch_save" / f"epoch-{int(params['parent_step']) // STEPS_PER_EPOCH}.pt"
            if not checkpoint.is_file():
                return None
            entry["checkpoint_step"] = int(params["parent_step"])
            entry["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        out[dep_id] = entry
    return out


def main() -> int:
    """Run one stage (``train`` or ``evaluate``) as ``pilot.launch`` does: argv is stage, spec.json, run_dir."""
    stage, spec_path, run_dir = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    scenarios = json.loads(Path(os.environ["FAKE_SCENARIOS"]).read_text(encoding="utf-8"))
    scenario = scenarios.get(spec["run_id"], "success")
    if stage == "train":
        if scenario == "unavailable":
            return EXIT_UNAVAILABLE
        if scenario == "refused":
            return EXIT_REFUSED
        if scenario == "noclaim":
            return 1
        dependencies = resolve_dependencies(spec, run_dir)
        if dependencies is None:  # the real launcher resolves its dependencies before the claim
            print(f"REFUSED: a dependency of {spec['run_id']} is not a completed run", file=sys.stderr)
            return EXIT_REFUSED
        # Claim the directory atomically, as the real launcher does.
        os.close(os.open(run_dir / CLAIM_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        if scenario == "segv":
            os.kill(os.getpid(), signal.SIGSEGV)
        if scenario == "hang":
            while not Path(os.environ["FAKE_RELEASE"]).exists():
                time.sleep(0.05)
        omni = write_outputs(spec, run_dir, nan_multiplier=scenario == "nonfinite")
        if scenario == "interrupt":
            os.kill(os.getpid(), signal.SIGKILL)
        if scenario == "sighup":
            signal.signal(signal.SIGHUP, signal.SIG_DFL)
            os.kill(os.getpid(), signal.SIGHUP)
        if scenario == "exit1":
            return 1
        if scenario == "crash":
            result = train_result(spec, omni, "failed", "crash", dependencies)
        elif scenario == "nonfinite":
            result = train_result(spec, omni, "failed", "non_finite_multiplier", dependencies)
        elif scenario == "sigterm":
            result = {**train_result(spec, omni, "interrupted", None, dependencies), "detail": "signal 15",
                      "interruption": "signal 15"}
        else:
            result = train_result(spec, omni, "completed", None, dependencies)
        (run_dir / "train_result.json").write_text(json.dumps(result), encoding="utf-8")
        return {"completed": EXIT_COMPLETED, "interrupted": EXIT_INTERRUPTED}.get(result["status"], EXIT_FAILED)
    if spec["group"] in CONTINUATION_GROUPS:  # pilot.launch.evaluate refuses a continuation
        print(f"REFUSED: {spec['run_id']} is a continuation; it is not evaluated by this stage", file=sys.stderr)
        return EXIT_REFUSED
    if scenario == "evalfail":
        return 1
    if scenario == "evalunavailable":
        return EXIT_UNAVAILABLE
    if scenario == "evalrefused":
        return EXIT_REFUSED
    train = json.loads((run_dir / "train_result.json").read_text(encoding="utf-8"))
    steps = sorted(int(p.stem.split("-")[1]) * STEPS_PER_EPOCH for p in (run_dir / train["omnisafe_dir"] / "torch_save").glob("epoch-*.pt"))
    try:
        window = selection_window(spec, steps)
    except IncompleteWindow as exc:  # too short to be evaluated
        print(f"REFUSED: {spec['run_id']}: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    evaluation = evaluation_result(spec, window, [25.0 + i * 0.1 for i in range(len(window))])
    (run_dir / "evaluation.json").write_text(json.dumps(evaluation), encoding="utf-8")
    return EXIT_COMPLETED


if __name__ == "__main__":
    sys.exit(main())

"""Determinism check (Table 3.1 "Determinism check"; First Tasks, Role 1 step 10; pilot/scheduler.py).

Table 3.1: "Two runs with the same seed and configuration must produce identical evaluation cost at
the first checkpoint before any arm is launched." Trains one configuration with seed 0 twice, each
in a fresh process, through the same launcher as every registered run (CPU, one torch thread, one
environment, one process), and compares. ``--plugin`` chooses the configuration
(``pilot.scheduler.determinism_spec``):

  ppolag             PPO-Lagrangian unchanged on SafetyPointGoal1-v0 (the default; the First Tasks check)
  study_a            the N = 0 main-sweep arm on SafetyPointGoal1-v0 (Role 2's onset plug-in)
  study_a_pid        the N = 0.50 arm of the PID check (Table 3.3), with a CHANGED onset: N x the check's
                     length (100,000 steps in the registered form), not the arm's registered 5,000,000
                     steps, which lie beyond the check (pilot.scheduler.DETERMINISM_LATE_ONSET; Table 9.1,
                     Q-determinism-late-onset)
  study_b            the Moderate arm of Study B (Role 5's budget-conditioned plug-in, Table 3.4)
  unconstrained_ppo  the pilot's unconstrained PPO run (Part 3.6)

Each is seed 0 of the arm with its run_id, length, open questions (the exempt keys below) and
(study_a_pid) onset changed; the report records every change from the arm ("arm", "changed_from_arm",
"late_onset_rule"). The open questions of the arm that change no update
(pilot.scheduler.DETERMINISM_EXEMPT_KEYS: the Study B ordering keys and Q-plasticity-definitions) do
not hold the check.

The scheduler holds every registered run of plug-in study_a, study_a_pid, study_b or
unconstrained_ppo until HEAD holds a passing registered-form report of that plug-in's configuration
(``determinism_missing``). A run of study_a_pid is also held by Q-determinism-late-onset, a gate that
fails closed: the key is answered (Table 9.1), so it holds only if the key were reopened.
The comparisons:

  default form (First Tasks, stronger): every non-timing column of progress.csv for every epoch
      (episode cost, return, losses, multiplier, ...) and every tensor of every checkpoint
      (actor, critics, optimiser states, normaliser, multiplier), bit for bit; 2 epochs = 40,000 steps.
  --registered-form (Table 3.1 as written; through envs.evaluation.evaluate_checkpoint, Role 2's
      harness, contract 6 in pilot/contracts.py): identical evaluation cost at the first scheduled
      checkpoint (200,000 steps = epoch-10.pt; epoch-0.pt is the untrained initial policy, and
      study_a_pid's onset checkpoint is not compared; Table 9.1, Q-first-checkpoint). The harness is
      loaded before anything is trained: if it cannot be imported, this form stops at once with
      "UNAVAILABLE".

A configuration that waits on an open question (configs/registered.py PENDING) is refused before
anything is trained ("waiting on ..."), and so is the registered form of the budget-conditioned
plug-in while Q-studyb-eval is open (its evaluation cost depends on that key's answer; the harness
raises PendingQuestionError, reported the same way): nothing is written and the exit code is not 0.
Both refusals fail closed: every key is answered (ANSWERED_QUESTIONS), so they act only on a key
that is reopened.
``--allow-pending`` (rehearsal only) opens both; the report says so and the scheduler ignores it.

Writes ledger/determinism/determinism-<UTC time>.json and .md (determinism-<plugin>-<UTC time> for a
plug-in other than ppolag); the report records the plug-in. Exit code 0 only if the check passed:
identical results, the same config hash, and both runs trained on the commit the report names, which
is still HEAD at the end, from a clean tree (not required with --allow-dirty).

    python scripts/determinism_check.py                 # on the workstation, from a clean commit
    python scripts/determinism_check.py --registered-form --plugin study_a
    python scripts/determinism_check.py --allow-dirty   # rehearsal only; the report says so

Owner: pilot owner (Role 1).
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from configs import registered as R  # noqa: E402
from pilot import errors, provenance  # noqa: E402
from pilot.errors import PendingQuestionError  # noqa: E402
from pilot.launch import MULTIPLIER_COLUMN, plain  # noqa: E402
from pilot.manifest import RunSpec  # noqa: E402
from pilot.scheduler import (  # noqa: E402
    DETERMINISM_CHECK_PLUGINS,
    DETERMINISM_LATE_ONSET,
    DETERMINISM_REGISTERED_FORM,
    determinism_arm,
    determinism_changes,
    determinism_open_keys,
    determinism_spec,
)

OUT_DIR = REPO / "ledger" / "determinism"
TIME_COLUMNS = ("Time/Total", "Time/Rollout", "Time/Update", "Time/Epoch", "Time/FPS")
BUDGET_PLUGINS = ("study_b",)  # budget-conditioned: its evaluation depends on Q-studyb-eval (envs/evaluation.py)


def make_spec(total_steps: int, plugin: str = "ppolag") -> RunSpec:
    """The configuration checked for ``plugin`` (``pilot.scheduler.determinism_spec``, the one the gate accepts)."""
    return determinism_spec(plugin, total_steps)


def waiting_on(spec: RunSpec, registered_form: bool) -> tuple[str, ...]:
    """The open questions that hold this check (before anything is trained).

    The spec's own (the launcher refuses a run with open questions); the check's own key
    (study_a_pid's changed onset, Q-determinism-late-onset, Table 9.1; fail closed while
    unregistered); and for the registered form of a budget-conditioned plug-in Q-studyb-eval, which
    gates its evaluation cost (a result gate of ``envs.evaluation.evaluate_checkpoint``).
    """
    keys = list(spec.open_questions)
    # The check's own key (a late arm's onset, Q-determinism-late-onset): open until answered, and while it is
    # not yet registered in PENDING (pilot.scheduler.determinism_open_keys fails closed).
    keys += [k for k in determinism_open_keys(spec.plugin) if k not in keys]
    if registered_form and spec.plugin in BUDGET_PLUGINS:
        keys += [k for k in errors.open_keys("Q-studyb-eval") if k not in keys]
    return tuple(keys)


def train_once(spec: RunSpec, run_dir: Path, allow_dirty: bool, allow_pending: bool = False) -> Path:
    """Train ``spec`` once through the launcher in a fresh process; the run's OmniSafe directory."""
    run_dir.mkdir(parents=True)
    (run_dir / "spec.json").write_text(spec.to_json(), encoding="utf-8")
    cmd = [sys.executable, "-m", "pilot.launch", "train", "--spec", str(run_dir / "spec.json"), "--run-dir", str(run_dir)]
    if allow_dirty:
        cmd.append("--allow-dirty")
    if allow_pending:
        cmd.append("--allow-pending")
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    with open(run_dir / "launch.log", "wb") as log:
        code = subprocess.run(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    if not (run_dir / "train_result.json").exists():
        raise SystemExit(f"the launcher exited {code} without a train_result.json; see {run_dir / 'launch.log'}")
    result = json.loads((run_dir / "train_result.json").read_text(encoding="utf-8"))
    if code != 0 or result["status"] != "completed":
        raise SystemExit(f"training failed in {run_dir}: {result.get('failure_cause')} {result.get('detail', '')[-500:]}")
    return run_dir / result["omnisafe_dir"]


def progress_rows(omnisafe_dir: Path) -> list[dict[str, str]]:
    """The rows of progress.csv without the timing columns (TIME_COLUMNS), which differ between runs."""
    with open(omnisafe_dir / "progress.csv", newline="", encoding="utf-8") as fh:
        return [{k: v for k, v in row.items() if k not in TIME_COLUMNS} for row in csv.DictReader(fh)]


def _flatten(prefix: str, obj: Any, out: dict) -> None:
    """Add every leaf of ``obj`` (tensors and plain values in nested dicts, lists and tuples) to ``out`` by path."""
    import torch

    if isinstance(obj, torch.Tensor):
        out[prefix] = obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(f"{prefix}/{k}", v, out)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _flatten(f"{prefix}/{i}", v, out)
    else:
        out[prefix] = obj


def checkpoint_contents(omnisafe_dir: Path) -> dict[str, dict]:
    """Every checkpoint file of the run, flattened (``_flatten``), by file name."""
    import torch

    contents = {}
    for path in sorted((omnisafe_dir / "torch_save").glob("epoch-*.pt")):
        flat: dict = {}
        _flatten("", torch.load(path, map_location="cpu", weights_only=False), flat)
        contents[path.name] = flat
    return contents


def compare(a_dir: Path, b_dir: Path) -> dict:
    """The bitwise form: every non-timing progress.csv value and every checkpoint value of two runs, compared."""
    import torch

    rows_a, rows_b = progress_rows(a_dir), progress_rows(b_dir)
    differences = []
    if len(rows_a) != len(rows_b):
        differences.append(f"epoch count {len(rows_a)} vs {len(rows_b)}")
    for i, (ra, rb) in enumerate(zip(rows_a, rows_b)):
        for key in sorted(set(ra) | set(rb)):
            if ra.get(key) != rb.get(key):
                differences.append(f"epoch {i} {key}: {ra.get(key)} vs {rb.get(key)}")
    ck_a, ck_b = checkpoint_contents(a_dir), checkpoint_contents(b_dir)
    if sorted(ck_a) != sorted(ck_b):
        differences.append(f"checkpoint files {sorted(ck_a)} vs {sorted(ck_b)}")
    digest = hashlib.sha256()
    tensors = 0
    for name in sorted(set(ck_a) & set(ck_b)):
        fa, fb = ck_a[name], ck_b[name]
        if sorted(fa) != sorted(fb):
            differences.append(f"{name}: different keys")
            continue
        for key in sorted(fa):
            va, vb = fa[key], fb[key]
            if type(va) is not type(vb):
                differences.append(f"{name}{key}: types differ ({type(va).__name__} vs {type(vb).__name__})")
            elif isinstance(va, torch.Tensor):
                tensors += 1
                if va.dtype != vb.dtype or va.shape != vb.shape or not torch.equal(va, vb):
                    differences.append(f"{name}{key}: tensors differ")
                digest.update(va.detach().cpu().contiguous().numpy().tobytes())
            elif va != vb:
                differences.append(f"{name}{key}: {va!r} vs {vb!r}")
    # Every multiplier column (Study B logs one per training level; contract 3), besides cost and return.
    per_epoch = [
        {"epoch": i, "EpCost": r.get("Metrics/EpCost"), "EpRet": r.get("Metrics/EpRet"),
         **{("LagrangeMultiplier" + k[len(MULTIPLIER_COLUMN):]): v for k, v in r.items() if k.startswith(MULTIPLIER_COLUMN)
            and not k.endswith(("/Min", "/Max", "/Std"))}}
        for i, r in enumerate(rows_a)
    ]
    return {
        "identical": not differences,
        "differences": differences[:50],
        "epochs_compared": len(rows_a),
        "columns_compared": sorted(rows_a[0]) if rows_a else [],
        "checkpoints_compared": sorted(ck_a),
        "tensors_compared": tensors,
        "tensor_digest_sha256": digest.hexdigest(),
        "per_epoch_run_a": per_epoch,
    }


def load_checkpoint_evaluator(allow_dirty: bool) -> Callable[..., Any]:
    """The harness of contract 6 (pilot/contracts.py), loaded before anything is trained."""
    from pilot.algorithms import PluginUnavailableError
    from pilot.enrichment import _load

    try:
        evaluate = _load(("envs.evaluation", "evaluate_checkpoint", "Environment and tests (Role 2)"))
    except PluginUnavailableError as exc:
        raise SystemExit(f"UNAVAILABLE: {exc}; the registered form needs it (run the default form until then)") from None
    unverified = [] if allow_dirty else provenance.unverified_imported_code()
    if unverified:
        raise SystemExit("REFUSED: the evaluation would execute code that the commit does not contain: " + ", ".join(unverified))
    return evaluate


def registered_form(a_dir: Path, b_dir: Path, evaluate: Callable[..., Any], allow_dirty: bool, *,
                    spec: RunSpec | None = None, allow_pending: bool = False,
                    step: int = R.CHECKPOINT_INTERVAL_STEPS, episodes: int = R.EVAL_EPISODES) -> dict:
    """Table 3.1 as written: the evaluation cost of each run's first scheduled checkpoint (contract 6).

    ``spec`` is passed to the harness (``evaluate_checkpoint(..., spec=...)``), which checks each run's
    config.json against it and reads a budget-conditioned arm's training budgets from it. A
    PendingQuestionError of the harness (Q-studyb-eval, unless ``allow_pending``) stops the check:
    "waiting on ...", nothing written. ``step`` and ``episodes`` are the registered ones (tests
    shrink them on tiny runs).
    """
    kwargs = {} if spec is None else {"spec": spec.to_dict()}
    try:
        with errors.allow_pending(allow_pending):
            cost_a = plain(evaluate(str(a_dir), step, episodes, **kwargs))  # numbers, not numpy scalars
            cost_b = plain(evaluate(str(b_dir), step, episodes, **kwargs))
    except PendingQuestionError as exc:
        keys = [k for k in R.PENDING if k in str(exc)] or ["an open question"]
        raise SystemExit(f"waiting on {', '.join(keys)}: {exc}; no report written") from None
    unverified = [] if allow_dirty else provenance.unverified_imported_code()  # modules the harness imported lazily
    if unverified:
        raise SystemExit("REFUSED: the evaluation executed code that the commit does not contain: " + ", ".join(unverified))
    return {"checkpoint_step": step, "evaluation_cost_a": cost_a, "evaluation_cost_b": cost_b, "identical": cost_a == cost_b}


def main(argv: list[str] | None = None) -> int:
    """Train the configuration twice, compare, write the report, and return the exit code (module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--registered-form", action="store_true",
                        help="Table 3.1 as written: evaluation cost at the first scheduled checkpoint "
                             "(needs the evaluation harness)")
    parser.add_argument("--plugin", choices=DETERMINISM_CHECK_PLUGINS, default="ppolag",
                        help="the configuration checked (pilot.scheduler.determinism_spec; default ppolag)")
    parser.add_argument("--allow-dirty", action="store_true", help="rehearsal only")
    parser.add_argument("--allow-pending", action="store_true",
                        help="rehearsal only: run a configuration that waits on open questions (the report says so)")
    parser.add_argument("--keep", type=Path, help="keep the two run directories here (must not hold runA or runB)")
    args = parser.parse_args(argv)
    if args.keep and any((args.keep / name).exists() for name in ("runA", "runB")):
        parser.error(f"{args.keep} already holds runA or runB; choose an empty directory")

    total = R.CHECKPOINT_INTERVAL_STEPS if args.registered_form else 2 * R.STEPS_PER_EPOCH
    spec = make_spec(total, args.plugin)
    arm = determinism_arm(args.plugin)
    waiting = waiting_on(spec, args.registered_form)
    if waiting and not args.allow_pending:  # before anything is trained: the check could not pass
        raise SystemExit(f"waiting on {', '.join(waiting)}: the {args.plugin} configuration depends on open questions "
                         "(configs/registered.py PENDING); no report written. Rehearse with --allow-dirty --allow-pending")
    started = provenance.utc_now()
    try:
        commit = provenance.commit_hash() if args.allow_dirty else provenance.require_clean_worktree()
    except provenance.DirtyWorktreeError as exc:
        raise SystemExit(f"REFUSED: {exc}") from None
    evaluate = load_checkpoint_evaluator(args.allow_dirty) if args.registered_form else None
    workspace = (
        contextlib.nullcontext(str(args.keep.resolve()))  # absolute: the launcher runs with cwd = the repository
        if args.keep
        else tempfile.TemporaryDirectory(prefix="determinism-")
    )
    with workspace as tmp:
        base = Path(tmp)
        a_dir = train_once(spec, base / "runA", args.allow_dirty, args.allow_pending)
        b_dir = train_once(spec, base / "runB", args.allow_dirty, args.allow_pending)
        if evaluate is not None:
            result = registered_form(a_dir, b_dir, evaluate, args.allow_dirty, spec=spec, allow_pending=args.allow_pending)
        else:
            result = compare(a_dir, b_dir)
        train_a = json.loads((base / "runA" / "train_result.json").read_text(encoding="utf-8"))
        train_b = json.loads((base / "runB" / "train_result.json").read_text(encoding="utf-8"))
    same_config = train_a.get("config_hash") is not None and train_a.get("config_hash") == train_b.get("config_hash")
    # Both runs must have trained on the commit the report names, from a clean tree, and HEAD must still be that
    # commit: a commit landing between runA and runB (separate launcher processes) would otherwise give a pass for
    # code neither run, or only one, trained on.
    run_commits = [train_a.get("commit_hash"), train_b.get("commit_hash")]
    runs_dirty = [train_a.get("worktree_dirty"), train_b.get("worktree_dirty")]
    end_commit = provenance.commit_hash()
    same_code = (run_commits == [commit, commit] and end_commit == commit
                 and (args.allow_dirty or runs_dirty == [False, False]))
    report = {
        "check": DETERMINISM_REGISTERED_FORM if args.registered_form else "bitwise form (First Tasks, Role 1 step 10)",
        "plugin": args.plugin,
        "passed": bool(result["identical"] and same_config and same_code),
        "same_config_hash": same_config,
        "same_commit": same_code,
        "commit_hash": commit,
        "run_commits": run_commits,
        "runs_worktree_dirty": runs_dirty,
        "commit_at_end": end_commit,
        "worktree_dirty": bool(provenance.dirty_paths()),
        # A rehearsal (uncommitted code, or open questions bypassed) never releases the scheduler's gate.
        "rehearsal_only": bool(args.allow_dirty or args.allow_pending),
        "allow_pending": bool(args.allow_pending),
        "started_utc": started.isoformat(),
        "finished_utc": provenance.utc_now().isoformat(),
        "machine": provenance.machine_description(),
        "versions": train_a.get("versions"),
        "torch_threads": [train_a.get("torch_threads"), train_b.get("torch_threads")],
        "config_hash": [train_a.get("config_hash"), train_b.get("config_hash")],
        "wall_clock_hours": [train_a.get("wall_clock_hours"), train_b.get("wall_clock_hours")],
        "spec": spec.to_dict(),
        # The configuration checked, against the registered arm it stands for (pilot/scheduler.py): study_a_pid's
        # onset is not the arm's (DETERMINISM_LATE_ONSET; Table 9.1, Q-determinism-late-onset).
        "arm": None if arm is None else arm.run_id,
        "changed_from_arm": determinism_changes(args.plugin, total),
        "late_onset_rule": DETERMINISM_LATE_ONSET if arm is not None and arm.onset_step else None,
        "result": result,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    name = f"determinism-{stamp}" if args.plugin == "ppolag" else f"determinism-{args.plugin}-{stamp}"
    js = OUT_DIR / f"{name}.json"
    md = OUT_DIR / f"{name}.md"
    changes = "; ".join(f"{k} {v[0]} -> {v[1]}" for k, v in report["changed_from_arm"].items())
    arm_lines = "" if arm is None else f"- arm: {arm.run_id}; changed from it: {changes}\n"
    if report["late_onset_rule"]:
        arm_lines += (f"- onset: {report['late_onset_rule']} (Table 9.1, Q-determinism-late-onset; not the arm's "
                      "registered onset)\n")
    # each report file whole and only once (provenance.write_text_once), the .json last: a crash never leaves an
    # empty or partial report, and a readable .json always has its .md
    provenance.write_text_once(md, (
        f"# Determinism check {stamp}\n\n- check: {report['check']}\n- plug-in: {args.plugin} ({spec.run_id})\n"
        f"{arm_lines}"
        f"- passed: **{report['passed']}**\n"
        f"- commit: {commit} (dirty: {report['worktree_dirty']}; rehearsal: {report['rehearsal_only']})\n"
        f"- commits the runs trained on: {run_commits} (dirty: {runs_dirty}); HEAD at the end: {end_commit}\n"
        f"- machine: {report['machine']}\n- versions: {json.dumps(report['versions'])}\n"
        f"- torch threads: {report['torch_threads']}\n- config hashes: {report['config_hash']}\n"
        f"- details: see {js.name}\n"))
    try:
        provenance.write_text_once(js, json.dumps(plain(report), indent=2, sort_keys=True) + "\n")
    except BaseException:
        md.unlink(missing_ok=True)
        raise
    print(md.read_text(encoding="utf-8"))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

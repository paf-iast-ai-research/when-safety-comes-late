"""Determinism check (Table 3.1 "Determinism check"; First Tasks, Role 1 step 10).

Trains PPO-Lagrangian on SafetyPointGoal1-v0 with seed 0 twice, each in a fresh process, through
the same launcher as every registered run (CPU, one torch thread, one environment, one process),
and compares:

  default form (First Tasks, stronger): every non-timing column of progress.csv for every epoch
      (episode cost, return, losses, multiplier, ...) and every tensor of every checkpoint
      (actor, critics, optimiser states, normaliser, multiplier), bit for bit; 2 epochs = 40,000 steps.
  --registered-form (Table 3.1 as written, once the evaluation harness exists): identical
      evaluation cost at the first scheduled checkpoint (200,000 steps = epoch-10.pt; epoch-0.pt is
      the untrained initial policy), through envs.evaluation.evaluate_checkpoint (Role 2; contract 6
      in pilot/contracts.py). The harness is loaded before anything is trained: while it does not
      exist, this form stops at once with "UNAVAILABLE".

Writes ledger/determinism/determinism-<UTC time>.json and .md. Exit code 0 only if identical.

    python scripts/determinism_check.py                 # on the workstation, from a clean commit
    python scripts/determinism_check.py --allow-dirty   # rehearsal only; the report says so
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
from pilot import provenance  # noqa: E402
from pilot.launch import plain  # noqa: E402
from pilot.manifest import RunSpec  # noqa: E402

OUT_DIR = REPO / "ledger" / "determinism"
TIME_COLUMNS = ("Time/Total", "Time/Rollout", "Time/Update", "Time/Epoch", "Time/FPS")


def make_spec(total_steps: int) -> RunSpec:
    return RunSpec(
        run_id="DET-PPOLag-PointGoal1-s0",
        study="A",
        task=R.PRIMARY_TASK,
        arm="determinism-check",
        seed=0,
        total_steps=total_steps,
        base_algo="PPOLag",
        plugin="ppolag",
        group="determinism",
    )


def train_once(spec: RunSpec, run_dir: Path, allow_dirty: bool) -> Path:
    run_dir.mkdir(parents=True)
    (run_dir / "spec.json").write_text(spec.to_json(), encoding="utf-8")
    cmd = [sys.executable, "-m", "pilot.launch", "train", "--spec", str(run_dir / "spec.json"), "--run-dir", str(run_dir)]
    if allow_dirty:
        cmd.append("--allow-dirty")
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    with open(run_dir / "launch.log", "wb") as log:
        code = subprocess.run(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    if not (run_dir / "train_result.json").exists():
        raise SystemExit(f"the launcher exited {code} before training; see {run_dir / 'launch.log'}")
    result = json.loads((run_dir / "train_result.json").read_text(encoding="utf-8"))
    if code != 0 or result["status"] != "completed":
        raise SystemExit(f"training failed in {run_dir}: {result.get('failure_cause')} {result.get('detail', '')[-500:]}")
    return run_dir / result["omnisafe_dir"]


def progress_rows(omnisafe_dir: Path) -> list[dict[str, str]]:
    with open(omnisafe_dir / "progress.csv", newline="", encoding="utf-8") as fh:
        return [{k: v for k, v in row.items() if k not in TIME_COLUMNS} for row in csv.DictReader(fh)]


def _flatten(prefix: str, obj, out: dict) -> None:
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
    import torch

    contents = {}
    for path in sorted((omnisafe_dir / "torch_save").glob("epoch-*.pt")):
        flat: dict = {}
        _flatten("", torch.load(path, map_location="cpu", weights_only=False), flat)
        contents[path.name] = flat
    return contents


def compare(a_dir: Path, b_dir: Path) -> dict:
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
            if isinstance(va, torch.Tensor):
                tensors += 1
                if va.dtype != vb.dtype or va.shape != vb.shape or not torch.equal(va, vb):
                    differences.append(f"{name}{key}: tensors differ")
                digest.update(va.detach().cpu().contiguous().numpy().tobytes())
            elif va != vb:
                differences.append(f"{name}{key}: {va!r} vs {vb!r}")
    per_epoch = [
        {"epoch": i, "EpCost": r.get("Metrics/EpCost"), "EpRet": r.get("Metrics/EpRet"), "LagrangeMultiplier": r.get("Metrics/LagrangeMultiplier")}
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


def load_checkpoint_evaluator(allow_dirty: bool) -> Callable[[str, int, int], Any]:
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


def registered_form(a_dir: Path, b_dir: Path, evaluate: Callable[[str, int, int], Any], allow_dirty: bool) -> dict:
    step = R.CHECKPOINT_INTERVAL_STEPS
    cost_a = plain(evaluate(str(a_dir), step, R.EVAL_EPISODES))  # numbers, not numpy scalars
    cost_b = plain(evaluate(str(b_dir), step, R.EVAL_EPISODES))
    unverified = [] if allow_dirty else provenance.unverified_imported_code()  # modules the harness imported lazily
    if unverified:
        raise SystemExit("REFUSED: the evaluation executed code that the commit does not contain: " + ", ".join(unverified))
    return {"checkpoint_step": step, "evaluation_cost_a": cost_a, "evaluation_cost_b": cost_b, "identical": cost_a == cost_b}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--registered-form", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true", help="rehearsal only")
    parser.add_argument("--keep", type=Path, help="keep the two run directories here (must not hold runA or runB)")
    args = parser.parse_args(argv)
    if args.keep and any((args.keep / name).exists() for name in ("runA", "runB")):
        parser.error(f"{args.keep} already holds runA or runB; choose an empty directory")

    total = R.CHECKPOINT_INTERVAL_STEPS if args.registered_form else 2 * R.STEPS_PER_EPOCH
    spec = make_spec(total)
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
        a_dir = train_once(spec, base / "runA", args.allow_dirty)
        b_dir = train_once(spec, base / "runB", args.allow_dirty)
        if evaluate is not None:
            result = registered_form(a_dir, b_dir, evaluate, args.allow_dirty)
        else:
            result = compare(a_dir, b_dir)
        train_a = json.loads((base / "runA" / "train_result.json").read_text(encoding="utf-8"))
        train_b = json.loads((base / "runB" / "train_result.json").read_text(encoding="utf-8"))
    same_config = train_a.get("config_hash") == train_b.get("config_hash") is not None
    report = {
        "check": "registered form (Table 3.1)" if args.registered_form else "bitwise form (First Tasks, Role 1 step 10)",
        "passed": bool(result["identical"] and same_config),
        "same_config_hash": same_config,
        "commit_hash": commit,
        "worktree_dirty": bool(provenance.dirty_paths()),
        "rehearsal_only": bool(args.allow_dirty),
        "started_utc": started.isoformat(),
        "finished_utc": provenance.utc_now().isoformat(),
        "machine": provenance.machine_description(),
        "versions": train_a.get("versions"),
        "torch_threads": [train_a.get("torch_threads"), train_b.get("torch_threads")],
        "config_hash": [train_a.get("config_hash"), train_b.get("config_hash")],
        "wall_clock_hours": [train_a.get("wall_clock_hours"), train_b.get("wall_clock_hours")],
        "spec": spec.to_dict(),
        "result": result,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    js = OUT_DIR / f"determinism-{stamp}.json"
    js.write_text(json.dumps(plain(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = OUT_DIR / f"determinism-{stamp}.md"
    md.write_text(
        f"# Determinism check {stamp}\n\n- check: {report['check']}\n- passed: **{report['passed']}**\n"
        f"- commit: {commit} (dirty: {report['worktree_dirty']}; rehearsal: {report['rehearsal_only']})\n"
        f"- machine: {report['machine']}\n- versions: {json.dumps(report['versions'])}\n"
        f"- torch threads: {report['torch_threads']}\n- config hashes: {report['config_hash']}\n"
        f"- details: see {js.name}\n",
        encoding="utf-8",
    )
    print(md.read_text(encoding="utf-8"))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

"""Stand-in for ``python -m pilot.launch`` used by tests/test_scheduler.py.

Writes the same files the real launcher writes (the ``.claim`` file, OmniSafe-like progress.csv
and torch_save files, train_result.json, evaluation.json) without training anything, and uses the
real launcher's exit codes. The outcome of each run is read from the JSON file named by
$FAKE_SCENARIOS: {run_id: scenario}; the recorded commit is $FAKE_COMMIT if set. Scenarios:

  success          training and evaluation complete
  crash            training fails with an exception (failure_cause 'crash')
  nonfinite        the multiplier becomes NaN (failure_cause 'non_finite_multiplier')
  interrupt        the process is killed without a final result (SIGKILL, e.g. the OOM killer)
  exit1            the process exits with code 1 after its claim, without a final result (an
                   uncaught launcher error)
  sigterm          the launcher is stopped by SIGTERM and records status 'interrupted' (exit 3)
  unavailable      exit code 4 before the claim: the plug-in is not in the repository yet
  refused          exit code 5 before the claim: the launcher refuses to start the run
  noclaim          exit code 1 before the claim: an unexpected launcher error
  segv             the process kills itself with SIGSEGV after claiming the directory
  hang             sleeps until the file $FAKE_RELEASE exists, then succeeds
  evalfail         training completes; evaluation exits with an error
  evalunavailable  training completes; evaluation exits with code 4 (no evaluation harness yet)
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path

STEPS_PER_EPOCH = 20_000
COMMIT = "0123456789abcdef0123456789abcdef01234567"
SELECTION_SEEDS = list(range(1000, 1100))
MEASUREMENT_SEEDS = list(range(2000, 2100))  # used by tests/test_enrichment.py (the battery's measurement set)
# The real launcher's exit codes (pilot.launch); repeated here so that the fake imports nothing of the package.
EXIT_COMPLETED, EXIT_FAILED, EXIT_INTERRUPTED, EXIT_UNAVAILABLE, EXIT_REFUSED = 0, 2, 3, 4, 5
CLAIM_FILE = ".claim"


def write_outputs(spec: dict, run_dir: Path, nan_multiplier: bool = False) -> str:
    epochs = spec["total_steps"] // STEPS_PER_EPOCH
    omni = run_dir / "omnisafe" / f"{spec['base_algo']}-{{{spec['task']}}}" / "seed-000-2026-01-01-00-00-00"
    (omni / "torch_save").mkdir(parents=True, exist_ok=True)
    # Study B logs one multiplier per training level (contract 3); every other run logs one.
    levels = spec.get("training_levels") if spec.get("study") == "B" else None
    names = [f"Metrics/LagrangeMultiplier/level_{float(b):g}" for b in levels] if levels else ["Metrics/LagrangeMultiplier"]
    with open(omni / "progress.csv", "w", encoding="utf-8") as fh:
        fh.write("Metrics/EpRet,Metrics/EpCost,Train/Epoch,TotalEnvSteps,Loss/Loss_pi," + ",".join(names) + "\n")
        for e in range(epochs):
            lam = "nan" if nan_multiplier and e == epochs - 1 else f"{0.001 + 0.035 * (e + 1):.6f}"
            fh.write(f"{10 + e * 0.1},{30 - e * 0.05},{e},{(e + 1) * STEPS_PER_EPOCH},0.1," + ",".join([lam] * len(names)) + "\n")
    for k in range(0, epochs + 1, 10):
        (omni / "torch_save" / f"epoch-{k}.pt").write_bytes(b"")
    if epochs % 10:
        (omni / "torch_save" / f"epoch-{epochs}.pt").write_bytes(b"")
    return str(omni.relative_to(run_dir))


def train_result(spec: dict, omni: str | None, status: str, cause: str | None) -> dict:
    return {
        "run_id": spec["run_id"], "status": status, "failure_cause": cause, "detail": "fake",
        "commit_hash": os.environ.get("FAKE_COMMIT", COMMIT), "config_hash": "c" * 64, "machine": "fake-machine",
        "started": "2026-01-01T00:00:00+00:00", "finished": "2026-01-01T01:00:00+00:00",
        "wall_clock_hours": 1.0, "omnisafe_dir": omni, "versions": {},
    }


def main() -> int:
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
        if scenario == "exit1":
            return 1
        if scenario == "crash":
            result = train_result(spec, omni, "failed", "crash")
        elif scenario == "nonfinite":
            result = train_result(spec, omni, "failed", "non_finite_multiplier")
        elif scenario == "sigterm":
            result = {**train_result(spec, omni, "interrupted", None), "detail": "signal 15", "interruption": "signal 15"}
        else:
            result = train_result(spec, omni, "completed", None)
        (run_dir / "train_result.json").write_text(json.dumps(result), encoding="utf-8")
        return {"completed": EXIT_COMPLETED, "interrupted": EXIT_INTERRUPTED}.get(result["status"], EXIT_FAILED)
    if scenario == "evalfail":
        return 1
    if scenario == "evalunavailable":
        return EXIT_UNAVAILABLE
    train = json.loads((run_dir / "train_result.json").read_text(encoding="utf-8"))
    steps = sorted(int(p.stem.split("-")[1]) * STEPS_PER_EPOCH for p in (run_dir / train["omnisafe_dir"] / "torch_save").glob("epoch-*.pt"))
    evaluation = {
        "final_cost": 24.0, "final_return": 20.0, "episodes": 100, "eval_wall_clock_hours": 0.25,
        "selection": {str(s): [25.0 + i * 0.1, 20.0] for i, s in enumerate(steps[-10:])},
        "selection_seeds": SELECTION_SEEDS,
    }
    (run_dir / "evaluation.json").write_text(json.dumps(evaluation), encoding="utf-8")
    return EXIT_COMPLETED


if __name__ == "__main__":
    sys.exit(main())

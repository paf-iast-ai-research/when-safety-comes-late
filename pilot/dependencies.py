"""Runs that depend on other runs: locating the dependency, and restoring a continuation's learner.

Owner: pilot owner (Role 1). Dependencies, and the continuation restore policy (``restore_learner``).

Three kinds of runs start from another run (``RunSpec.depends_on``):

* the warm-started multiplier (Table 2.4 "Warm-started multiplier": "At onset the multiplier is set
  to the value the N = 0 arm of the same task and seed had reached at the same step"), which reads
  the N = 0 run's logged multiplier (``Dependency.multiplier_at``; Table 9.1, Q-warm-start);
* the battery's fine-tuning and transfer continuations (Table 2.2), from the parent's matched
  checkpoint (``params["parent_step"]``; Table 9.1, Q-continuations: Table 2.2 names no starting
  checkpoint);
* Study B's few-shot continuations (Table 2.5 "Few-shot adaptation"), from the parent's final
  checkpoint (Table 9.1, Q-continuations).

The launcher calls ``resolve`` before it claims the run directory (a dependency that is not a
completed run whose spec.json names the expected run_id is a refusal, exit 5; the run stays
pending, never excluded), stores ``to_config(...)`` in ``cfgs.pilot_cfgs.dependencies`` (hashed,
apart from the two absolute paths, so the configuration hash identifies the exact input checkpoint
but not the data root), and records the same block in train_result.json. Plug-ins read their
dependencies only through ``from_cfgs(cfgs)``, never by guessing the scheduler's directory layout.

``restore_learner`` applies the one restore rule of every continuation (Table 9.1, Q-continuations,
a run gate of every continuation spec): the parent's actor, both critics and observation normaliser
are restored; the optimisers start fresh, and so does the actor's learning-rate schedule, over the
continuation's full registered length (a few-shot continuation shortened by Part 6.1 cut 3 keeps
the 1,000,000-step schedule, ``params["lr_schedule_steps"]``); the multiplier is the plug-in's (held
at 0 for fine-tuning; a fresh one at 0.001 for transfer and few-shot).

Top-level imports are only ``pilot.rundir``, ``pilot.errors`` and ``pilot.manifest`` (pilot/rundir.py:
no import cycle with ``pilot.launch``, and importing the scheduler does not import torch); torch,
``pilot.provenance``, ``results.ledger_schema`` and ``metrics`` are imported inside the functions
that need them.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from pilot import rundir
from pilot.errors import RunRefused
from pilot.manifest import RunSpec

# Keys a full-state checkpoint of FullStateCheckpointMixin always holds (pilot/algorithms.py).
FULL_STATE_KEYS = ("pi", "reward_critic", "actor_optimizer", "reward_critic_optimizer", "actor_scheduler",
                   "episode_windows")
# The first weight of an injected actor's trunk: its presence marks a parent that received plasticity
# injection (HANDOVER.md section 8: InjectedHead with submodules trunk, frozen, new, new_frozen). It mirrors
# metrics.interventions.INJECTED_KEY, kept local so that importing this module imports neither torch
# nor metrics; tests/test_core_dependencies.py holds the two equal.
INJECTED_KEY = "mean.trunk.0.weight"
# Parameters that the injected layout keeps frozen (the old head and the frozen copy of the new one).
INJECTED_FROZEN_PREFIXES = ("mean.frozen.", "mean.new_frozen.")
_DEPENDENCY_FIELDS = ("run_dir", "omnisafe_dir", "commit_hash", "config_hash", "total_steps")
# The markers of a smoke run in train_result.json (pilot/launch.py ``train``), the ones the ledger
# writer's smoke guard reads: uncommitted code, or open questions bypassed.
SMOKE_MARKERS = ("worktree_dirty", "allow_dirty", "allow_pending")


@dataclass(frozen=True)
class Dependency:
    """A completed run that another run starts from.

    ``run_dir`` and ``omnisafe_dir`` are absolute. Steps are converted to checkpoint epochs with the
    dependency's own ``algo_cfgs.steps_per_epoch`` (its config.json; E below), which is
    R.STEPS_PER_EPOCH for every registered run and 2,000 in the tiny test configurations.
    """

    run_id: str
    run_dir: Path
    omnisafe_dir: Path
    commit_hash: str
    config_hash: str
    total_steps: int

    def steps_per_epoch(self) -> int:
        """The dependency's own epoch length (config.json ``algo_cfgs.steps_per_epoch``)."""
        try:
            return rundir.run_steps_per_epoch(self.omnisafe_dir)
        except (OSError, ValueError) as exc:
            raise RunRefused(f"dependency {self.run_id}: {exc}") from exc

    def _epoch(self, step: int) -> int:
        epoch_length = self.steps_per_epoch()
        if isinstance(step, bool) or not isinstance(step, int) or step < 0 or step % epoch_length:
            raise RunRefused(
                f"dependency {self.run_id}: step {step!r} is not a non-negative whole number of its "
                f"{epoch_length}-step epochs"
            )
        return step // epoch_length

    def checkpoint(self, step: int) -> Path:
        """``torch_save/epoch-{step // E}.pt``, the state after ``step`` steps; RunRefused if it was not saved."""
        path = rundir.checkpoint_file(self.omnisafe_dir, self._epoch(step))
        if not path.is_file():
            raise RunRefused(f"dependency {self.run_id} saved no checkpoint at step {step} ({path})")
        return path

    def progress(self) -> list[dict[str, str]]:
        """The dependency's progress.csv rows (one per epoch)."""
        try:
            return rundir.read_progress(self.omnisafe_dir)
        except (OSError, ValueError, csv.Error) as exc:  # unreadable, not UTF-8, or not a CSV file
            raise RunRefused(f"dependency {self.run_id}: cannot read its progress.csv: {exc}") from exc

    def multiplier_at(self, step: int) -> float:
        """The multiplier the dependency had reached after ``step`` steps, exactly as it logged it.

        OmniSafe stores ``Metrics/LagrangeMultiplier`` after the epoch's multiplier update
        (``ppo_lag.py`` ``_update``), in the row whose 0-based ``Train/Epoch`` is ``step // E - 1``
        (``policy_gradient.py`` ``learn``); the ledger writer reads checkpoints the same way. The
        value is the float that progress.csv holds (Table 9.1, Q-warm-start, copies it).
        RunRefused unless exactly one row has that epoch and its value is a finite number.
        """
        epoch = self._epoch(step)
        if epoch == 0:
            raise RunRefused(f"dependency {self.run_id}: no multiplier is logged before the first epoch (step 0)")
        rows = [r for r in self.progress() if rundir.row_epoch(r) == epoch - 1]
        if len(rows) != 1:
            raise RunRefused(
                f"dependency {self.run_id}: {len(rows)} progress rows for epoch {epoch - 1} (step {step}); expected one"
            )
        text = rows[0].get(rundir.MULTIPLIER_COLUMN)
        try:
            value = float(text)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            raise RunRefused(
                f"dependency {self.run_id}: {rundir.MULTIPLIER_COLUMN} at step {step} is {text!r}, not a number"
            ) from None
        if not math.isfinite(value):
            raise RunRefused(f"dependency {self.run_id}: the multiplier at step {step} is {value}")
        return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, what: str) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RunRefused(f"{what}: cannot read {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RunRefused(f"{what}: {path} does not hold a JSON object")
    return data


def _whole_step(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RunRefused(f"{what} must be a non-negative whole number of steps, got {value!r}")
    return value


def _read_dependency(dep_id: str, dep_dir: Path, *, smoke: bool = False) -> Dependency:
    """The completed run ``dep_id`` in ``dep_dir``; RunRefused for anything else.

    A dependency trained as a smoke run (any of SMOKE_MARKERS true in its train_result.json) is
    refused unless the dependent run is itself launched as one (``smoke``): otherwise a clean child
    would carry a smoke parent's learner or multiplier into the registered ledgers with no trace in
    its own train_result.json (HANDOVER.md section 10 for --allow-pending; ``pilot.launch.train`` for
    --allow-dirty: both bypasses are for smoke roots only).
    """
    what = f"dependency {dep_id}"
    if not dep_dir.is_dir():
        raise RunRefused(f"{what} has no run directory {dep_dir}; it is not trained yet")
    try:
        dep_spec = RunSpec.from_json((dep_dir / rundir.SPEC_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise RunRefused(f"{what}: {dep_dir / rundir.SPEC_FILE} is missing or not a run spec: {exc}") from exc
    if dep_spec.run_id != dep_id:
        raise RunRefused(f"{what}: {dep_dir} holds the spec of {dep_spec.run_id!r}")
    train = _read_json(dep_dir / rundir.TRAIN_RESULT, what)
    if train.get("run_id") != dep_id:
        raise RunRefused(f"{what}: {rundir.TRAIN_RESULT} belongs to {train.get('run_id')!r}")
    if train.get("status") != "completed":
        raise RunRefused(f"{what} is not a completed run (training status {train.get('status')!r})")
    markers = [k for k in SMOKE_MARKERS if train.get(k)]
    if markers and not smoke:
        raise RunRefused(
            f"{what} was trained as a smoke run ({', '.join(markers)} in {rundir.TRAIN_RESULT}); only a smoke run "
            "(--allow-dirty or --allow-pending) may start from it"
        )
    from pilot.provenance import is_full_commit_hash

    commit = train.get("commit_hash")
    if not isinstance(commit, str) or not is_full_commit_hash(commit):
        raise RunRefused(f"{what}: commit hash {commit!r} is not a full 40-character hash")
    config_hash = train.get("config_hash")
    if not isinstance(config_hash, str) or not config_hash:
        raise RunRefused(f"{what}: {rundir.TRAIN_RESULT} records no config_hash")
    relative = train.get("omnisafe_dir")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise RunRefused(f"{what}: {rundir.TRAIN_RESULT} records no relative OmniSafe directory ({relative!r})")
    omnisafe_dir = Path(os.path.normpath(dep_dir / relative))
    if not omnisafe_dir.is_dir() or dep_dir not in omnisafe_dir.parents:
        raise RunRefused(f"{what}: {omnisafe_dir} is not an OmniSafe directory inside {dep_dir}")
    dependency = Dependency(run_id=dep_id, run_dir=dep_dir, omnisafe_dir=omnisafe_dir, commit_hash=commit,
                            config_hash=config_hash, total_steps=int(dep_spec.total_steps))
    dependency.steps_per_epoch()  # config.json must say how its checkpoints are numbered
    return dependency


def load_full_state(path: Path) -> dict[str, Any]:
    """A full-state checkpoint, loaded without executing pickled code (contract 3).

    ``torch.load(path, weights_only=True, map_location="cpu")``. RunRefused if the file does not
    load or is not a full-state checkpoint of ``FullStateCheckpointMixin`` (FULL_STATE_KEYS).
    """
    import torch

    try:
        state = torch.load(Path(path), weights_only=True, map_location="cpu")
    except Exception as exc:  # noqa: BLE001 - any failure to load is a refusal, never a crash
        raise RunRefused(f"{path} does not load with weights_only=True: {exc}") from exc
    if not isinstance(state, dict):
        raise RunRefused(f"{path} is not a checkpoint dictionary")
    missing = [k for k in FULL_STATE_KEYS if k not in state]
    if missing:
        raise RunRefused(f"{path} is not a full-state checkpoint: it lacks {missing}")
    return state


def _ledger_spelling(path: Path) -> str:
    """``path`` as the ledger spells checkpoints (``results.ledger_schema.relative_checkpoint_path``).

    Relative to CHECKPOINT_ROOT when it lies below it (not ``resolve()``d, so a /data symlink to the
    real storage keeps the relative form), else absolute.
    """
    from results.ledger_schema import relative_checkpoint_path

    return relative_checkpoint_path(os.path.abspath(path))


def _check_parent(spec: RunSpec, dependency: Dependency) -> None:
    """A continuation's parent: the checkpoint it starts from exists, loads, and is the one the spec names.

    The spec's ``parent_commit`` must be the parent's training commit and ``parent_checkpoint`` the
    ledger's spelling of that checkpoint (``results.ledger_schema.relative_checkpoint_path``), so a
    continuation never starts from a namesake run under another data root.
    """
    params = spec.params
    what = f"{spec.run_id}: parent {dependency.run_id}"
    if "parent_commit" in params and params["parent_commit"] != dependency.commit_hash:
        raise RunRefused(f"{what} was trained at {dependency.commit_hash}, but the spec names {params['parent_commit']!r}")
    if params.get("parent_step") is None:
        if "parent_checkpoint" in params:
            raise RunRefused(f"{what}: the spec names a parent checkpoint but no parent_step")
        return
    step = _whole_step(params["parent_step"], f"{spec.run_id}: parent_step")
    path = dependency.checkpoint(step)
    if "parent_checkpoint" in params:
        spelled = _ledger_spelling(path)
        if params["parent_checkpoint"] != spelled:
            raise RunRefused(f"{what}: the spec names checkpoint {params['parent_checkpoint']!r}, the run saved {spelled!r}")
    load_full_state(path)


def resolve(spec: RunSpec, run_dir: Path, *, smoke: bool = False) -> dict[str, Dependency]:
    """Every dependency of ``spec``, found beside its run directory.

    The scheduler puts each run in ``<data_root>/checkpoints/<run_id>``, so a dependency is the
    sibling ``run_dir.parent / dep_id``. Each must hold ``spec.json`` with that run_id and a
    ``train_result.json`` of a completed run with its commit, config hash and OmniSafe directory.
    For a continuation's parent (``params["parent_run_id"]``) the checkpoint at
    ``params["parent_step"]`` must exist and load as a full state, and the parent's commit and the
    checkpoint path must be those the spec names. A dependency trained as a smoke run is refused
    unless ``smoke`` (the dependent run is launched with --allow-dirty or --allow-pending, so its own
    train_result.json carries the smoke marker). RunRefused otherwise: the launcher calls this
    before its claim, so the run stays pending and nothing is written.
    """
    run_dir = Path(os.path.abspath(run_dir))
    parent_id = spec.params.get("parent_run_id")
    if parent_id is not None and parent_id not in spec.depends_on:
        raise RunRefused(f"{spec.run_id}: parent {parent_id!r} is not among its dependencies {spec.depends_on}")
    out: dict[str, Dependency] = {}
    for dep_id in spec.depends_on:
        dependency = _read_dependency(dep_id, run_dir.parent / dep_id, smoke=smoke)
        if dep_id == parent_id:
            _check_parent(spec, dependency)
        out[dep_id] = dependency
    return out


def to_config(deps: Mapping[str, Dependency], spec: RunSpec) -> dict[str, dict[str, Any]]:
    """The ``pilot_cfgs.dependencies`` block: JSON-ready, one entry per dependency.

    ``{dep: {run_dir, omnisafe_dir, commit_hash, config_hash, total_steps[, checkpoint_step,
    checkpoint_sha256]}}``; the checkpoint entries for a continuation's parent. The two paths are
    volatile for the configuration hash (``pilot.provenance``); everything else is hashed, so the
    hash changes with the parent's checkpoint bytes.
    """
    if set(deps) != set(spec.depends_on):
        raise RunRefused(f"{spec.run_id}: dependencies {sorted(deps)} are not the spec's {list(spec.depends_on)}")
    out: dict[str, dict[str, Any]] = {}
    for dep_id in spec.depends_on:
        dep = deps[dep_id]
        entry: dict[str, Any] = {
            "run_dir": str(dep.run_dir),
            "omnisafe_dir": str(dep.omnisafe_dir),
            "commit_hash": dep.commit_hash,
            "config_hash": dep.config_hash,
            "total_steps": int(dep.total_steps),
        }
        if dep_id == spec.params.get("parent_run_id") and spec.params.get("parent_step") is not None:
            step = _whole_step(spec.params["parent_step"], f"{spec.run_id}: parent_step")
            entry["checkpoint_step"] = step
            entry["checkpoint_sha256"] = _sha256(dep.checkpoint(step))
        out[dep_id] = entry
    return out


def _pilot_cfgs(cfgs: Any) -> Mapping[str, Any]:
    pilot_cfgs = cfgs.get("pilot_cfgs") if isinstance(cfgs, Mapping) else getattr(cfgs, "pilot_cfgs", None)
    return pilot_cfgs if isinstance(pilot_cfgs, Mapping) else {}


def from_cfgs(cfgs: Any) -> dict[str, Dependency]:
    """The dependencies the launcher resolved for this run (``cfgs.pilot_cfgs.dependencies``)."""
    block = _pilot_cfgs(cfgs).get("dependencies") or {}
    if not isinstance(block, Mapping):
        raise RunRefused(f"pilot_cfgs.dependencies is not a mapping: {block!r}")
    out: dict[str, Dependency] = {}
    for dep_id, entry in block.items():
        missing = [k for k in _DEPENDENCY_FIELDS if not isinstance(entry, Mapping) or k not in entry]
        if missing:
            raise RunRefused(f"pilot_cfgs.dependencies[{dep_id!r}] lacks {missing}; the launcher writes every field")
        total_steps = _whole_step(entry["total_steps"], f"pilot_cfgs.dependencies[{dep_id!r}].total_steps")
        out[dep_id] = Dependency(
            run_id=dep_id, run_dir=Path(entry["run_dir"]), omnisafe_dir=Path(entry["omnisafe_dir"]),
            commit_hash=str(entry["commit_hash"]), config_hash=str(entry["config_hash"]),
            total_steps=total_steps,
        )
    return out


# ---------------------------------------------------------------------------
# Restoring a continuation's learner (``restore_learner``; Table 9.1, Q-continuations)
# ---------------------------------------------------------------------------


def _count(parameters: Any) -> int:
    return int(sum(p.numel() for p in parameters))


def actor_schedule_length(cfgs: Any, spec: RunSpec) -> tuple[int, int]:
    """(epochs, steps) of a continuation's fresh actor schedule (``restore_learner`` item 4).

    Table 9.1 (Q-continuations): the continuation's full registered length. That is its own run,
    ``train_cfgs.epochs`` of ``algo_cfgs.steps_per_epoch`` steps (the launcher derives the epochs from
    ``spec.total_steps``, pilot/launch.py), unless the spec names ``params["lr_schedule_steps"]``: a
    few-shot continuation shortened by Part 6.1 cut 3 (``pilot.manifest._cut_fewshot_short``) keeps
    the 1,000,000-step schedule and stops after its 200,000 steps. RunRefused unless that value is a
    whole number of the run's epochs and at least as long as the run.
    """
    steps_per_epoch = int(cfgs.algo_cfgs.steps_per_epoch)
    epochs = int(cfgs.train_cfgs.epochs)
    steps = spec.params.get("lr_schedule_steps")
    if steps is None:
        return epochs, epochs * steps_per_epoch
    if isinstance(steps, bool) or not isinstance(steps, int) or steps % steps_per_epoch or steps < epochs * steps_per_epoch:
        raise RunRefused(
            f"{spec.run_id}: params lr_schedule_steps {steps!r} is not a whole number of the run's {steps_per_epoch}-step "
            f"epochs at least as long as its {epochs} epochs ({epochs * steps_per_epoch} steps; Q-continuations)"
        )
    return steps // steps_per_epoch, steps


def restore_learner(
    algo: Any,
    state: Mapping[str, Any],
    *,
    spec: RunSpec,
    obs_map: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Restore the parent's learner into a freshly built continuation algorithm.

    Called in the continuation's ``_init``, after OmniSafe's ``_init_env`` and ``_init_model`` and
    before ``_init_log`` (``base_algo.py`` ``__init__``: ``_init_env``, ``_init_model``, ``_init``,
    ``_init_log``), so the mixin's saver captures the restored objects and ``epoch-0.pt`` holds the
    restored learner. ``state`` is ``load_full_state(dep.checkpoint(spec.params["parent_step"]))``.
    Table 9.1, Q-continuations (one rule for fine-tuning, transfer and few-shot):

    1. Actor. A parent with plasticity injection (``INJECTED_KEY`` in ``state["pi"]``) first gets its
       head rebuilt as ``metrics.interventions.InjectedHead`` by
       ``metrics.interventions.rebuild_injected(actor, pi_state)`` (Role 3; deepcopy-based, no
       global random draws). Then ``actor.load_state_dict(state["pi"])`` (strict), and
       ``requires_grad`` as in the parent: the injected layout's old head and frozen copy
       (``INJECTED_FROZEN_PREFIXES``) False, everything else True.
    2. ``reward_critic`` and ``cost_critic``: strict ``load_state_dict`` (both sides must agree on
       whether a cost critic exists).
    3. The observation normaliser (when ``algo_cfgs.obs_normalize``): strict ``load_state_dict`` into
       ``algo._env.save()["obs_normalizer"]``; it keeps updating as training does. For transfer,
       ``obs_map`` (Role 2, ``envs.continuations``; Table 9.1, Q-transfer-obs) returns the parent
       state mapped onto the target task's observation layout, first-layer weights and
       normaliser statistics included, before anything is loaded.
    4. Fresh optimisers and schedule, exactly as OmniSafe 0.5.0's actor-critic builds them
       (``models/actor_critic/actor_critic.py:91-113``: ``optim.Adam(self.actor.parameters(),
       lr=model_cfgs.actor.lr)``, ``optim.Adam(self.reward_critic.parameters(),
       lr=model_cfgs.critic.lr)``, then ``LinearLR(start_factor=1.0, end_factor=0.0,
       total_iters=epochs)`` if ``linear_lr_decay`` else ``ConstantLR(factor=1.0,
       total_iters=epochs)``; ``constraint_actor_critic.py:77-82`` for the cost critic): the actor's
       optimiser over its trainable parameters in registration order, and the schedule over the
       continuation's full registered length (``actor_schedule_length``): its own
       ``train_cfgs.epochs``, 50 for fine-tuning, transfer and few-shot, except that a few-shot
       continuation shortened by Part 6.1 cut 3 keeps the 50-epoch schedule of
       ``params["lr_schedule_steps"]`` and stops after 10 epochs, at 80 percent of the rate, as the
       full continuation is there (``pilot.manifest._cut_fewshot_short``). The parent's schedule is
       not usable: with ``linear_lr_decay`` (the registered setting) OmniSafe's rate is exactly 0 at
       the end of the parent's training, and either schedule has already run over the parent's
       epochs, not the continuation's.

    Not restored: the optimiser states, the parent's schedule, the multiplier and its optimiser
    (the plug-in holds it at 0 or starts a fresh one), the episode windows, and the random number
    generators (the continuation is seeded by ``cfgs.seed = spec.seed``). Nothing here draws a
    random number. Returns the summary that the plug-in writes to ``continuation.json`` in its
    OmniSafe directory. RunRefused for any mismatch (the launcher removes the attempt; the run
    stays pending).
    """
    from torch import optim
    from torch.optim.lr_scheduler import ConstantLR, LinearLR

    if getattr(algo, "_pilot_restore_summary", None) is not None:
        raise RunRefused(f"{spec.run_id}: the learner was already restored once")
    cfgs = algo._cfgs
    parent_id = spec.params.get("parent_run_id")
    parent_step = spec.params.get("parent_step")
    if parent_id is None or parent_step is None or spec.depends_on != (parent_id,):
        raise RunRefused(
            f"{spec.run_id} is not a continuation: it needs parent_run_id and parent_step and depends on exactly "
            f"that parent (parent_run_id {parent_id!r}, parent_step {parent_step!r}, depends_on {spec.depends_on})"
        )
    if int(cfgs.seed) != int(spec.seed):
        raise RunRefused(f"{spec.run_id}: cfgs.seed {cfgs.seed} is not the spec's seed {spec.seed}")
    schedule_epochs, schedule_steps = actor_schedule_length(cfgs, spec)
    resolved = from_cfgs(cfgs)  # refuses a block that is not a mapping, or an entry without its fields
    block = _pilot_cfgs(cfgs).get("dependencies") or {}
    record = block.get(parent_id) if isinstance(block, Mapping) else None
    if not isinstance(record, Mapping) or record.get("checkpoint_step") != parent_step or not record.get("checkpoint_sha256"):
        raise RunRefused(f"{spec.run_id}: pilot_cfgs.dependencies records no checkpoint of {parent_id} at step {parent_step}")
    path = resolved[parent_id].checkpoint(int(parent_step))
    digest = _sha256(path)
    if digest != record["checkpoint_sha256"]:
        raise RunRefused(f"{spec.run_id}: {path} changed since the launcher recorded it (sha256 {digest})")
    if obs_map is not None:
        state = obs_map(state)
    if not isinstance(state, Mapping):
        raise RunRefused(f"{spec.run_id}: the parent state is not a mapping ({type(state).__name__})")
    missing = [k for k in ("pi", "reward_critic") if k not in state]
    if missing:
        raise RunRefused(f"{spec.run_id}: the parent state lacks {missing}")
    # torch's load_state_dict raises TypeError, not RuntimeError, for an entry that is not dict-like
    not_dicts = [k for k in ("pi", "reward_critic", "cost_critic", "obs_normalizer")
                 if k in state and not isinstance(state[k], Mapping)]
    if not_dicts:
        raise RunRefused(f"{spec.run_id}: the parent state's {not_dicts} are not state dictionaries")

    ac = algo._actor_critic
    actor = ac.actor
    pi_state = state["pi"]
    injected = INJECTED_KEY in pi_state
    restored: list[str] = []
    try:
        if injected:
            from metrics.interventions import rebuild_injected  # Role 3 (HANDOVER.md section 8); a missing module or name -> exit 4

            rebuild_injected(actor, pi_state)
        actor.load_state_dict(pi_state)
        restored.append("pi")
        for name, param in actor.named_parameters():
            param.requires_grad_(not name.startswith(INJECTED_FROZEN_PREFIXES))
        ac.reward_critic.load_state_dict(state["reward_critic"])
        restored.append("reward_critic")
        if hasattr(ac, "cost_critic") != ("cost_critic" in state):
            raise RunRefused(f"{spec.run_id}: the continuation and its parent disagree on a cost critic")
        if hasattr(ac, "cost_critic"):
            ac.cost_critic.load_state_dict(state["cost_critic"])
            restored.append("cost_critic")
        normalise = bool(cfgs.algo_cfgs.obs_normalize)
        if normalise != ("obs_normalizer" in state):
            raise RunRefused(f"{spec.run_id}: the continuation and its parent disagree on observation normalisation")
        if normalise:
            normalizer = algo._env.save()["obs_normalizer"]
            normalizer.load_state_dict(state["obs_normalizer"])
            # OmniSafe's Normalizer replaces its statistics with the first batch it sees while its
            # plain attribute ``_first`` is True (common/normalizer.py:119-127 in 0.5.0), so a restored
            # normaliser must leave it False. The pinned load_state_dict already sets it
            # (normalizer.py:157); this assignment only repeats that, so that continuing the parent's
            # statistics does not rest on a detail of the pinned version. The slow test checks the
            # behaviour: the first push after the restore updates the parent's statistics.
            normalizer._first = False
            restored.append("obs_normalizer")
    except RuntimeError as exc:  # a strict load_state_dict with other keys or shapes
        raise RunRefused(f"{spec.run_id}: the parent state does not fit the continuation's networks: {exc}") from exc

    model_cfgs = cfgs.model_cfgs
    fresh: list[str] = []
    schedule: dict[str, Any] | None = None
    if model_cfgs.actor.lr is not None:
        trainable = [p for p in actor.parameters() if p.requires_grad]
        ac.actor_optimizer = optim.Adam(trainable, lr=model_cfgs.actor.lr)
        if model_cfgs.linear_lr_decay:
            ac.actor_scheduler = LinearLR(ac.actor_optimizer, start_factor=1.0, end_factor=0.0,
                                          total_iters=schedule_epochs)
        else:
            ac.actor_scheduler = ConstantLR(ac.actor_optimizer, factor=1.0, total_iters=schedule_epochs)
        schedule = {"type": type(ac.actor_scheduler).__name__, "total_iters": schedule_epochs,
                    "schedule_steps": schedule_steps, "lr": float(model_cfgs.actor.lr)}
        fresh += ["actor_optimizer", "actor_scheduler"]
    if model_cfgs.critic.lr is not None:
        ac.reward_critic_optimizer = optim.Adam(ac.reward_critic.parameters(), lr=model_cfgs.critic.lr)
        fresh.append("reward_critic_optimizer")
        if hasattr(ac, "cost_critic"):
            ac.cost_critic_optimizer = optim.Adam(ac.cost_critic.parameters(), lr=model_cfgs.critic.lr)
            fresh.append("cost_critic_optimizer")

    parameters = {
        "actor": _count(actor.parameters()),
        "actor_trainable": _count(p for p in actor.parameters() if p.requires_grad),
        "reward_critic": _count(ac.reward_critic.parameters()),
    }
    if hasattr(ac, "cost_critic"):
        parameters["cost_critic"] = _count(ac.cost_critic.parameters())
    summary = {
        "parent_run_id": parent_id,
        "parent_step": int(parent_step),
        # the ledger's spelling (relative to the schema's CHECKPOINT_ROOT when the data root is /data),
        # so that continuation.json is portable; the sha256 identifies the file wherever it lies
        "parent_checkpoint": _ledger_spelling(path),
        "parent_checkpoint_sha256": digest,
        "parent_commit_hash": record.get("commit_hash"),
        "parent_config_hash": record.get("config_hash"),
        "injected_parent": injected,
        "observation_mapping": obs_map is not None,
        "restored": restored,
        "fresh": fresh,
        "not_restored": sorted(k for k in state if k not in restored),
        "actor_schedule": schedule,
        "parameters": parameters,
        "rule": "Table 9.1, Q-continuations (pilot.dependencies.restore_learner)",
    }
    algo._pilot_restore_summary = summary
    return summary

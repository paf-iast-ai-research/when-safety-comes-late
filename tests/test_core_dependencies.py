"""Locating dependency runs (``pilot.dependencies``), without training.

Fixture run directories are laid out as the scheduler and the launcher write them
(``<root>/<run_id>/{spec.json, train_result.json, omnisafe/<exp_name>/seed-000-<timestamp>/}``,
pilot/rundir.py); checkpoints are small full-state dictionaries saved with torch.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from configs import registered as R
from pilot import dependencies, manifest, provenance
from pilot.dependencies import Dependency
from pilot.errors import RunRefused
from pilot.manifest import RunSpec

torch = pytest.importorskip("torch")

COMMIT = "c" * 40
TINY_EPOCH = 2_000


class _Opaque:
    """An object that only unpickling arbitrary code could restore (refused by weights_only=True)."""


def _full_state(scale: float = 1.0) -> dict:
    return {key: {"w": torch.full((2,), scale)} for key in dependencies.FULL_STATE_KEYS} | {
        "cost_critic": {"w": torch.zeros(2)}, "obs_normalizer": {"_count": torch.tensor(5)}}


def make_run(root: Path, spec: RunSpec, *, steps_per_epoch: int = TINY_EPOCH,
             multipliers: tuple[str, ...] = ("0.036000", "0.071000")) -> Path:
    """A finished run directory of ``spec`` under ``root``; returns its OmniSafe directory."""
    run_dir = root / spec.run_id
    omni = run_dir / "omnisafe" / "PPOLag-{SafetyPointGoal1-v0}" / "seed-000-2026-09-30-00-00-00"
    (omni / "torch_save").mkdir(parents=True)
    (run_dir / "spec.json").write_text(spec.to_json())
    (omni / "config.json").write_text(json.dumps({"algo_cfgs": {"steps_per_epoch": steps_per_epoch}}))
    rows = "".join(f"{e},{(e + 1) * steps_per_epoch},{m}\n" for e, m in enumerate(multipliers))
    (omni / "progress.csv").write_text("Train/Epoch,TotalEnvSteps,Metrics/LagrangeMultiplier\n" + rows)
    for k in (0, 1, 2):
        torch.save(_full_state(k), omni / "torch_save" / f"epoch-{k}.pt")
    (run_dir / "train_result.json").write_text(json.dumps({
        "run_id": spec.run_id, "status": "completed", "commit_hash": COMMIT, "config_hash": "f" * 64,
        "omnisafe_dir": str(omni.relative_to(run_dir)),
    }))
    return omni


def _parent() -> RunSpec:
    return next(s for s in manifest.design("main") if s.task == R.PRIMARY_TASK and s.N == 0.0 and s.seed == 0)


def _warm() -> RunSpec:
    return next(s for s in manifest.design("controller") if s.controller_variant == "warm_started" and s.seed == 0
                and s.task == R.PRIMARY_TASK and s.N == 0.25)


def _continuation(parent: RunSpec, omni: Path, step: int = 2 * TINY_EPOCH, commit: str = COMMIT) -> RunSpec:
    """The fine-tuning continuation of ``parent`` (trained at ``commit``) from its tiny checkpoint at ``step``
    (also tests/test_core_restore.py)."""
    from results.ledger_schema import relative_checkpoint_path

    row = {"run_id": parent.run_id, "matched": True, "completed": True, "matched_checkpoint_step": 9_600_000,
           "matched_checkpoint_path": "x", "commit_hash": commit}
    spec = manifest.battery_continuation(parent, row, "finetune")
    path = relative_checkpoint_path(os.path.abspath(omni / "torch_save" / f"epoch-{step // TINY_EPOCH}.pt"))
    return RunSpec.from_dict({**spec.to_dict(), "params": {**spec.params, "parent_step": step,
                                                           "parent_checkpoint": path}})


def test_resolve_finds_a_completed_dependency_beside_the_run(tmp_path) -> None:
    parent = _parent()
    omni = make_run(tmp_path, parent)
    warm = _warm()
    assert warm.depends_on == (parent.run_id,)
    deps = dependencies.resolve(warm, tmp_path / warm.run_id)  # the run directory need not exist yet
    dep = deps[parent.run_id]
    assert dep == Dependency(run_id=parent.run_id, run_dir=tmp_path / parent.run_id, omnisafe_dir=omni,
                             commit_hash=COMMIT, config_hash="f" * 64, total_steps=parent.total_steps)
    assert dep.run_dir.is_absolute() and dep.omnisafe_dir.is_absolute()
    assert dependencies.resolve(parent, tmp_path / parent.run_id) == {}  # nothing to resolve


@pytest.mark.parametrize("damage, message", [
    ("no_dir", "not trained yet"),
    ("spec_other", "holds the spec"),
    ("spec_missing", "not a run spec"),
    ("failed", "not a completed run"),
    ("interrupted", "not a completed run"),
    ("result_other", "belongs to"),
    ("short_commit", "not a full 40-character hash"),
    ("no_config_hash", "no config_hash"),
    ("no_omnisafe", "no relative OmniSafe directory"),
    ("absolute_omnisafe", "no relative OmniSafe directory"),
    ("escaping_omnisafe", "is not an OmniSafe directory"),
    ("no_config_json", "config.json"),
    ("bad_json", "cannot read"),
])
def test_resolve_refuses_a_dependency_that_is_not_ready(tmp_path, damage, message) -> None:
    parent = _parent()
    omni = make_run(tmp_path, parent)
    run_dir = tmp_path / parent.run_id
    result = json.loads((run_dir / "train_result.json").read_text())
    if damage == "no_dir":
        shutil.rmtree(run_dir)
    elif damage == "spec_other":
        (run_dir / "spec.json").write_text(parent.with_seed(1).to_json())
    elif damage == "spec_missing":
        (run_dir / "spec.json").unlink()
    elif damage == "bad_json":
        (run_dir / "train_result.json").write_text("{not json")
    elif damage == "no_config_json":
        (omni / "config.json").unlink()
    else:
        result.update({
            "failed": {"status": "failed"}, "interrupted": {"status": "interrupted"},
            "result_other": {"run_id": "A-other-s0"}, "short_commit": {"commit_hash": "abc123"},
            "no_config_hash": {"config_hash": ""}, "no_omnisafe": {"omnisafe_dir": None},
            "absolute_omnisafe": {"omnisafe_dir": str(omni)}, "escaping_omnisafe": {"omnisafe_dir": "../elsewhere"},
        }[damage])
        (run_dir / "train_result.json").write_text(json.dumps(result))
    if damage == "escaping_omnisafe":
        # a real OmniSafe directory outside the run, so that only the containment check refuses it
        elsewhere = tmp_path / "elsewhere"
        (elsewhere / "torch_save").mkdir(parents=True)
        shutil.copy(omni / "config.json", elsewhere / "config.json")
    with pytest.raises(RunRefused, match=message):
        dependencies.resolve(_warm(), tmp_path / _warm().run_id)


def test_a_continuation_parent_must_be_the_checkpoint_the_spec_names(tmp_path) -> None:
    parent = _parent()
    omni = make_run(tmp_path, parent)
    child = _continuation(parent, omni)
    deps = dependencies.resolve(child, tmp_path / child.run_id)
    assert list(deps) == [parent.run_id]
    bad = {
        "parent_commit": ("d" * 40, "was trained at"),
        "parent_checkpoint": ("elsewhere/epoch-2.pt", "names checkpoint"),
        "parent_step": (3 * TINY_EPOCH, "saved no checkpoint"),
    }
    for key, (value, message) in bad.items():
        spec = RunSpec.from_dict({**child.to_dict(), "params": {**child.params, key: value}})
        with pytest.raises(RunRefused, match=message):
            dependencies.resolve(spec, tmp_path / child.run_id)
    odd = RunSpec.from_dict({**child.to_dict(), "params": {**child.params, "parent_step": TINY_EPOCH + 1}})
    with pytest.raises(RunRefused, match="whole number"):
        dependencies.resolve(odd, tmp_path / child.run_id)
    orphan = RunSpec.from_dict({**child.to_dict(), "params": {**child.params, "parent_run_id": "A-other-s0"}})
    with pytest.raises(RunRefused, match="not among its dependencies"):
        dependencies.resolve(orphan, tmp_path / child.run_id)
    # the checkpoint must load without pickled code and be a full state
    (omni / "torch_save" / "epoch-2.pt").write_bytes(b"truncated")
    with pytest.raises(RunRefused, match="weights_only"):
        dependencies.resolve(child, tmp_path / child.run_id)
    torch.save({"pi": {"w": torch.zeros(1)}}, omni / "torch_save" / "epoch-2.pt")
    with pytest.raises(RunRefused, match="not a full-state checkpoint"):
        dependencies.resolve(child, tmp_path / child.run_id)


def test_checkpoints_are_numbered_by_the_dependencys_own_epoch_length(tmp_path) -> None:
    parent = _parent()
    omni = make_run(tmp_path, parent)
    dep = dependencies.resolve(_warm(), tmp_path / _warm().run_id)[parent.run_id]
    assert dep.steps_per_epoch() == TINY_EPOCH
    assert dep.checkpoint(0) == omni / "torch_save" / "epoch-0.pt"
    assert dep.checkpoint(2 * TINY_EPOCH) == omni / "torch_save" / "epoch-2.pt"
    for step in (TINY_EPOCH // 2, -TINY_EPOCH, 1.0, True):
        with pytest.raises(RunRefused, match="whole number"):
            dep.checkpoint(step)
    with pytest.raises(RunRefused, match="saved no checkpoint"):
        dep.checkpoint(5 * TINY_EPOCH)
    # a registered run: 20,000 steps per epoch
    registered = make_run(tmp_path / "reg", parent, steps_per_epoch=R.STEPS_PER_EPOCH)
    dep = dependencies.resolve(_warm(), tmp_path / "reg" / _warm().run_id)[parent.run_id]
    assert dep.checkpoint(2 * R.STEPS_PER_EPOCH) == registered / "torch_save" / "epoch-2.pt"


def test_multiplier_at_returns_the_logged_value_exactly(tmp_path) -> None:
    parent = _parent()
    make_run(tmp_path, parent, multipliers=("0.03599999472498894", "0.07100000232458115"))
    dep = dependencies.resolve(_warm(), tmp_path / _warm().run_id)[parent.run_id]
    # the value after k epochs is the row of Train/Epoch k - 1, bit for bit
    assert dep.multiplier_at(TINY_EPOCH) == 0.03599999472498894
    assert dep.multiplier_at(2 * TINY_EPOCH).hex() == float("0.07100000232458115").hex()
    assert len(dep.progress()) == 2
    for step, message in ((0, "before the first epoch"), (3 * TINY_EPOCH, "0 progress rows"), (TINY_EPOCH + 1, "whole number")):
        with pytest.raises(RunRefused, match=message):
            dep.multiplier_at(step)


@pytest.mark.parametrize("csv_text, message", [
    ("Train/Epoch,TotalEnvSteps,Metrics/LagrangeMultiplier\n0,2000,nan\n", "is nan"),
    ("Train/Epoch,TotalEnvSteps,Metrics/LagrangeMultiplier\n0,2000,inf\n", "is inf"),
    ("Train/Epoch,TotalEnvSteps,Metrics/LagrangeMultiplier\n0,2000,\n", "not a number"),
    ("Train/Epoch,TotalEnvSteps,Metrics/LagrangeMultiplier/level_10\n0,2000,0.1\n", "not a number"),
    ("Train/Epoch,TotalEnvSteps,Metrics/LagrangeMultiplier\n0,2000,0.1\n0.0,2000,0.1\n", "2 progress rows"),
])
def test_multiplier_at_refuses_a_log_it_cannot_read(tmp_path, csv_text, message) -> None:
    parent = _parent()
    omni = make_run(tmp_path, parent)
    (omni / "progress.csv").write_text(csv_text)
    dep = dependencies.resolve(_warm(), tmp_path / _warm().run_id)[parent.run_id]
    with pytest.raises(RunRefused, match=message):
        dep.multiplier_at(TINY_EPOCH)
    (omni / "progress.csv").unlink()
    with pytest.raises(RunRefused, match="progress.csv"):
        dep.progress()


def test_a_progress_log_that_is_not_utf8_text_is_a_refusal(tmp_path) -> None:
    parent = _parent()
    omni = make_run(tmp_path, parent)
    dep = dependencies.resolve(_warm(), tmp_path / _warm().run_id)[parent.run_id]
    (omni / "progress.csv").write_bytes(b"\xff\xfeTrain/Epoch\n")
    with pytest.raises(RunRefused, match="cannot read its progress.csv"):
        dep.multiplier_at(TINY_EPOCH)


def test_to_config_records_the_parent_checkpoint_digest(tmp_path) -> None:
    parent = _parent()
    omni = make_run(tmp_path, parent)
    child = _continuation(parent, omni)
    deps = dependencies.resolve(child, tmp_path / child.run_id)
    block = dependencies.to_config(deps, child)
    digest = hashlib.sha256((omni / "torch_save" / "epoch-2.pt").read_bytes()).hexdigest()
    assert block == {parent.run_id: {
        "run_dir": str(tmp_path / parent.run_id), "omnisafe_dir": str(omni), "commit_hash": COMMIT,
        "config_hash": "f" * 64, "total_steps": parent.total_steps, "checkpoint_step": 2 * TINY_EPOCH,
        "checkpoint_sha256": digest,
    }}
    assert json.loads(json.dumps(block)) == block  # config.json and train_result.json hold it as is
    # a warm start reads the log, not a checkpoint: no digest
    warm_block = dependencies.to_config(dependencies.resolve(_warm(), tmp_path / _warm().run_id), _warm())
    assert set(warm_block[parent.run_id]) == set(dependencies._DEPENDENCY_FIELDS)
    with pytest.raises(RunRefused, match="not the spec's"):
        dependencies.to_config({}, child)
    assert dependencies.to_config({}, parent) == {}
    # from_cfgs gives back what resolve found
    assert dependencies.from_cfgs({"pilot_cfgs": {"dependencies": block}}) == deps
    assert dependencies.from_cfgs({"pilot_cfgs": {"dependencies": {}}}) == {} == dependencies.from_cfgs({})
    broken = {parent.run_id: {k: v for k, v in block[parent.run_id].items() if k != "omnisafe_dir"}}
    with pytest.raises(RunRefused, match="lacks"):
        dependencies.from_cfgs({"pilot_cfgs": {"dependencies": broken}})
    with pytest.raises(RunRefused, match="not a mapping"):
        dependencies.from_cfgs({"pilot_cfgs": {"dependencies": [block]}})
    for bad_steps in ("many", -1, 2.5, True):
        bad = {parent.run_id: {**block[parent.run_id], "total_steps": bad_steps}}
        with pytest.raises(RunRefused, match="total_steps"):
            dependencies.from_cfgs({"pilot_cfgs": {"dependencies": bad}})


def _hashable_config(block: dict) -> dict:
    return {"seed": 0, "logger_cfgs": {"log_dir": "/somewhere"}, "pilot_cfgs": {"run_id": "x-s0", "dependencies": block}}


def test_the_config_hash_names_the_input_checkpoint_but_not_the_data_root(tmp_path) -> None:
    parent = _parent()
    hashes = []
    children = []
    omnis = []
    for root in (tmp_path / "data_a", tmp_path / "data_b"):
        omni = make_run(root, parent)
        child = _continuation(parent, omni)
        block = dependencies.to_config(dependencies.resolve(child, root / child.run_id), child)
        hashes.append(provenance.config_hash(_hashable_config(block)))
        children.append(child)
        omnis.append(omni)
    assert hashes[0] == hashes[1]  # the same parent under two data roots
    omni_b, child_b = omnis[1], children[1]
    torch.save(_full_state(7.0), omni_b / "torch_save" / "epoch-2.pt")  # other checkpoint bytes under data_b
    changed = dependencies.to_config(dependencies.resolve(child_b, tmp_path / "data_b" / child_b.run_id), child_b)
    assert provenance.config_hash(_hashable_config(changed)) != hashes[1]
    other_commit = {parent.run_id: {**changed[parent.run_id], "commit_hash": "e" * 40}}
    assert provenance.config_hash(_hashable_config(other_commit)) != provenance.config_hash(_hashable_config(changed))


def test_load_full_state_refuses_what_is_not_a_full_state(tmp_path) -> None:
    path = tmp_path / "epoch-1.pt"
    torch.save(_full_state(), path)
    state = dependencies.load_full_state(path)
    assert set(dependencies.FULL_STATE_KEYS) <= set(state) and torch.equal(state["pi"]["w"], torch.ones(2))
    torch.save([1, 2], path)
    with pytest.raises(RunRefused, match="not a checkpoint dictionary"):
        dependencies.load_full_state(path)

    torch.save({"pi": _Opaque()}, path)
    with pytest.raises(RunRefused, match="weights_only"):
        dependencies.load_full_state(path)
    with pytest.raises(RunRefused, match="weights_only"):
        dependencies.load_full_state(tmp_path / "missing.pt")


def test_an_interruption_while_a_parent_loads_is_not_a_refusal(tmp_path, monkeypatch) -> None:
    """A continuation loads its parent in ``_init``, with the launcher's signal gate armed: a SIGTERM there
    (RunInterrupted, pilot/launch.py) once became a RunRefused (exit 5, the attempt discarded) instead of
    an interruption (Table 9.1, Q-interrupted-run)."""
    from pilot import launch

    path = tmp_path / "epoch-1.pt"
    torch.save(_full_state(), path)

    def interrupted(*args, **kwargs):
        raise launch.RunInterrupted("signal 15")

    monkeypatch.setattr(torch, "load", interrupted)
    with pytest.raises(launch.RunInterrupted, match="signal 15"):
        dependencies.load_full_state(path)


def test_restore_learner_refuses_a_run_that_is_not_a_continuation_before_touching_it() -> None:
    class Algo:
        _cfgs = {"seed": 0, "pilot_cfgs": {"dependencies": {}}}

    with pytest.raises(RunRefused, match="not a continuation"):
        dependencies.restore_learner(Algo(), {}, spec=_parent())
    # parent_run_id and parent_step named, but another dependency besides the parent
    parent = _parent()
    child = _continuation(parent, Path("/nowhere"))
    extra = RunSpec.from_dict({**child.to_dict(), "depends_on": [*child.depends_on, _warm().run_id]})
    with pytest.raises(RunRefused, match=re.escape("depends on exactly that parent")):
        dependencies.restore_learner(Algo(), {}, spec=extra)


def test_the_smoke_markers_are_those_the_ledger_writer_reads() -> None:
    from pilot import ledger_writer

    assert dependencies.SMOKE_MARKERS == ledger_writer.SMOKE_MARKERS


def test_is_full_commit_hash_accepts_only_exact_lowercase_40_hex_strings() -> None:
    assert provenance.is_full_commit_hash(COMMIT)
    for value in (COMMIT + "\n", COMMIT.upper(), COMMIT[:-1], COMMIT + "c", None, 40, b"c" * 40):
        assert not provenance.is_full_commit_hash(value)


@pytest.mark.parametrize("marker", dependencies.SMOKE_MARKERS)
def test_a_smoke_dependency_is_refused_unless_the_run_is_a_smoke_run_too(tmp_path, marker) -> None:
    """A clean run that started from a smoke run would carry the smoke parent's learner or multiplier
    into the registered ledgers with no smoke marker of its own (the ledger writer reads only the
    run's own train_result.json)."""
    parent = _parent()
    make_run(tmp_path, parent)
    path = tmp_path / parent.run_id / "train_result.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), marker: True}))
    with pytest.raises(RunRefused, match=re.escape(f"smoke run ({marker} in train_result.json)")):
        dependencies.resolve(_warm(), tmp_path / _warm().run_id)
    deps = dependencies.resolve(_warm(), tmp_path / _warm().run_id, smoke=True)  # a smoke run may start from it
    assert list(deps) == [parent.run_id]
    path.write_text(json.dumps({**json.loads(path.read_text()), marker: False}))
    assert dependencies.resolve(_warm(), tmp_path / _warm().run_id) == deps  # recorded false: not a smoke run


def test_ledger_spelling_is_relative_to_the_checkpoint_root_without_resolving_symlinks(tmp_path, monkeypatch) -> None:
    """The helper that ``restore_learner`` uses for continuation.json's parent_checkpoint: relative to the
    schema's CHECKPOINT_ROOT when the checkpoint lies below it (portable), absolute otherwise; the
    sha256 beside it identifies the file. tests/test_core_restore.py checks the summary itself."""
    import results.ledger_schema as schema

    root = tmp_path / "data" / "checkpoints"
    checkpoint = root / "A-PointGoal1-N0.00-s0" / "omnisafe" / "x" / "torch_save" / "epoch-500.pt"
    monkeypatch.setattr(schema, "CHECKPOINT_ROOT", str(root))
    assert dependencies._ledger_spelling(checkpoint) == "A-PointGoal1-N0.00-s0/omnisafe/x/torch_save/epoch-500.pt"
    assert dependencies._ledger_spelling(tmp_path / "elsewhere" / "epoch-1.pt") == str(tmp_path / "elsewhere" / "epoch-1.pt")
    link = tmp_path / "link"
    link.symlink_to(root.parent)  # a symlink is not resolved: /data -> real storage keeps the relative form
    monkeypatch.setattr(schema, "CHECKPOINT_ROOT", str(link / "checkpoints"))
    assert dependencies._ledger_spelling(link / "checkpoints" / "r-s0" / "epoch-1.pt") == "r-s0/epoch-1.pt"


def test_injected_key_mirrors_the_one_of_metrics_interventions() -> None:
    interventions = pytest.importorskip("metrics.interventions")
    assert dependencies.INJECTED_KEY == interventions.INJECTED_KEY


# ---------------------------------------------------------------------------
# restore_learner's refusals, with a stub algorithm (no OmniSafe, no training)
# ---------------------------------------------------------------------------


class _Weights(torch.nn.Module):
    """A network whose state is one parameter ``w``, as in ``_full_state``."""

    def __init__(self, size: int = 2) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(size))


class _Normalizer(torch.nn.Module):
    """A normaliser whose state is one buffer ``_count``, as in ``_full_state``."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("_count", torch.tensor(0))
        self._first = True


def _stub_algo(block: dict, *, seed: int = 0, cost_critic: bool = True, obs_normalize: bool = True,
               actor_size: int = 2, epochs: int = 3, steps_per_epoch: int = TINY_EPOCH) -> SimpleNamespace:
    """The attributes of an OmniSafe algorithm that ``restore_learner`` reads and replaces."""
    ac = SimpleNamespace(actor=_Weights(actor_size), reward_critic=_Weights())
    if cost_critic:
        ac.cost_critic = _Weights()
    normalizer = _Normalizer()
    cfgs = SimpleNamespace(
        seed=seed, pilot_cfgs={"dependencies": block},
        algo_cfgs=SimpleNamespace(obs_normalize=obs_normalize, steps_per_epoch=steps_per_epoch),
        train_cfgs=SimpleNamespace(epochs=epochs),
        model_cfgs=SimpleNamespace(actor=SimpleNamespace(lr=1e-3), critic=SimpleNamespace(lr=1e-3), linear_lr_decay=True),
    )
    return SimpleNamespace(_cfgs=cfgs, _actor_critic=ac, _env=SimpleNamespace(save=lambda: {"obs_normalizer": normalizer}))


def test_restore_learner_refuses_every_mismatch(tmp_path) -> None:
    parent = _parent()
    omni = make_run(tmp_path, parent)
    child = _continuation(parent, omni)
    block = dependencies.to_config(dependencies.resolve(child, tmp_path / child.run_id), child)
    state = dependencies.load_full_state(omni / "torch_save" / "epoch-2.pt")
    entry = block[parent.run_id]

    summary = dependencies.restore_learner(_stub_algo(block, seed=child.seed), state, spec=child)  # the stub fits
    assert summary["restored"] == ["pi", "reward_critic", "cost_critic", "obs_normalizer"]
    assert summary["parent_checkpoint_sha256"] == entry["checkpoint_sha256"]
    # an uncut continuation's schedule spans its own run (Table 9.1, Q-continuations)
    assert summary["actor_schedule"] == {"type": "LinearLR", "total_iters": 3, "schedule_steps": 3 * TINY_EPOCH, "lr": 1e-3}

    def refused(algo: SimpleNamespace, message: str, given: object = state, **kwargs: object) -> None:
        with pytest.raises(RunRefused, match=message):
            dependencies.restore_learner(algo, given, spec=child, **kwargs)  # type: ignore[arg-type]

    refused(_stub_algo(block, seed=child.seed + 1), "is not the spec's seed")
    unrecorded = {k: v for k, v in entry.items() if k not in ("checkpoint_step", "checkpoint_sha256")}
    refused(_stub_algo({parent.run_id: unrecorded}, seed=child.seed), "records no checkpoint of")
    for not_a_mapping in ([1], "abc"):
        refused(_stub_algo(not_a_mapping, seed=child.seed), "not a mapping")  # type: ignore[arg-type]
    refused(_stub_algo({parent.run_id: {**entry, "checkpoint_sha256": "0" * 64}}, seed=child.seed),
            "changed since the launcher")
    refused(_stub_algo(block, seed=child.seed, cost_critic=False), "disagree on a cost critic")
    refused(_stub_algo(block, seed=child.seed, obs_normalize=False), "disagree on observation normalisation")
    refused(_stub_algo(block, seed=child.seed, actor_size=3), "does not fit the continuation's networks")
    for key in ("pi", "reward_critic", "cost_critic", "obs_normalizer"):
        refused(_stub_algo(block, seed=child.seed), "are not state dictionaries", given={**state, key: [1, 2]})
    refused(_stub_algo(block, seed=child.seed), "lacks", given={k: v for k, v in state.items() if k != "pi"})
    refused(_stub_algo(block, seed=child.seed), "is not a mapping", obs_map=lambda s: [s])


def _fewshot_cut_short(tmp_path: Path) -> tuple[RunSpec, RunSpec, dict]:
    """A few-shot continuation as Part 6.1 cut 3 shortens it, its uncut form, and its dependencies block.

    The parent is a fixture run of the Study B training run, so ``parent_step`` names a fixture checkpoint."""
    parent = manifest.design("study_b")[0]
    make_run(tmp_path, parent)
    full = manifest.study_b_fewshot([parent])[0]
    (short,) = manifest.apply_cuts([full], manifest.CUTS[:manifest.CUTS.index("fewshot_short") + 1])
    full, short = (RunSpec.from_dict({**s.to_dict(), "params": {**s.params, "parent_step": 2 * TINY_EPOCH}})
                   for s in (full, short))
    block = dependencies.to_config(dependencies.resolve(short, tmp_path / short.run_id), short)
    return short, full, block


def test_a_shortened_fewshot_continuation_keeps_the_full_continuations_schedule(tmp_path) -> None:
    """Table 9.1 (Q-continuations): cut 3's continuation keeps the 1,000,000-step (50-epoch) LinearLR and stops after
    10 epochs, where the rate is 0.8 x lr; its 10 rates are the full 50-epoch continuation's first 10, bit for bit."""
    short, full, block = _fewshot_cut_short(tmp_path)
    assert (short.total_steps, short.params["lr_schedule_steps"]) == (R.FEWSHOT_HORIZONS[0], R.FEWSHOT_STEPS)
    state = dependencies.load_full_state(make_run(tmp_path / "copy", manifest.design("study_b")[0])
                                         / "torch_save" / "epoch-2.pt")
    rates = {}
    for name, spec, epochs in (("shortened", short, 10), ("uncut", full, 50)):
        algo = _stub_algo(block, seed=spec.seed, epochs=epochs, steps_per_epoch=R.STEPS_PER_EPOCH)
        summary = dependencies.restore_learner(algo, state, spec=spec)
        assert summary["actor_schedule"] == {"type": "LinearLR", "total_iters": 50, "schedule_steps": R.FEWSHOT_STEPS,
                                             "lr": 1e-3}
        scheduler = algo._actor_critic.actor_scheduler
        assert scheduler.total_iters == 50
        rates[name] = [scheduler.get_last_lr()[0]]
        for _ in range(epochs):
            algo._actor_critic.actor_optimizer.step()  # no gradient: changes nothing
            scheduler.step()
            rates[name].append(scheduler.get_last_lr()[0])
    shortened, uncut = rates["shortened"], rates["uncut"]
    assert shortened[10] == pytest.approx(0.8 * 1e-3, rel=1e-12) and uncut[-1] == 0.0
    assert shortened == uncut[:11]


@pytest.mark.parametrize("value", [R.FEWSHOT_STEPS + 1, R.FEWSHOT_HORIZONS[0] - R.STEPS_PER_EPOCH, True,
                                   float(R.FEWSHOT_STEPS), "1000000"])
def test_a_schedule_length_that_is_not_whole_epochs_or_shorter_than_the_run_is_refused(tmp_path, value) -> None:
    short, _, block = _fewshot_cut_short(tmp_path)
    spec = RunSpec.from_dict({**short.to_dict(), "params": {**short.params, "lr_schedule_steps": value}})
    algo = _stub_algo(block, seed=spec.seed, epochs=10, steps_per_epoch=R.STEPS_PER_EPOCH)
    with pytest.raises(RunRefused, match="lr_schedule_steps"):
        dependencies.restore_learner(algo, {}, spec=spec)
    assert not hasattr(algo, "_pilot_restore_summary")  # refused before anything was restored


def test_a_renamed_loaded_module_counts_as_changed(tmp_path, monkeypatch) -> None:
    """``git diff`` reports a rename under its new path only (diff.renames); the loaded path is the old one.

    This belongs beside tests/test_scheduler.py's imported_code_changed_between checks; it lives here
    because this unit's tests cover ``pilot.provenance``.
    """
    import importlib

    repo = tmp_path / "repo"
    pkg = repo / "renamed_pkg_for_test"
    pkg.mkdir(parents=True)

    def git(*args: str) -> str:
        return subprocess.run(["git", "-c", "user.email=t@example.org", "-c", "user.name=t", *args], cwd=repo,
                              check=True, capture_output=True, text=True).stdout.strip()

    git("init", "-q")
    git("config", "diff.renames", "true")  # the default since git 2.9; set so the test does not rest on it
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "mod_a.py").write_text("X = 1\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-qm", "first")
    old = git("rev-parse", "HEAD")
    monkeypatch.syspath_prepend(str(repo))
    for name in ("renamed_pkg_for_test", "renamed_pkg_for_test.mod_a"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    try:
        importlib.import_module("renamed_pkg_for_test.mod_a")  # loaded at ``old``, then renamed by ``new``
        git("mv", "renamed_pkg_for_test/mod_a.py", "renamed_pkg_for_test/mod_b.py")
        git("commit", "-qm", "rename")
        new = git("rev-parse", "HEAD")
        assert provenance.imported_code_changed_between(old, new, repo) == ["renamed_pkg_for_test/mod_a.py"]
        assert provenance.non_output_changes_between(old, new, repo) == [
            "renamed_pkg_for_test/mod_a.py", "renamed_pkg_for_test/mod_b.py"]
    finally:
        for name in ("renamed_pkg_for_test", "renamed_pkg_for_test.mod_a"):
            sys.modules.pop(name, None)

"""Partial reset and plasticity injection of the actor (Role 3, Metrics and interventions; owner Abdullah).

Implements Table 2.4 (PDF p. 9) and equation (8) (Part 2.5, p. 11):

* Partial reset: "At onset, the last two layers of the actor (the output layer and the hidden layer
  before it) are reinitialised with the network's default initialiser, and the optimiser state for
  those parameters is cleared. The critics are not reset. Reset depth of one layer is an ablation."
* Plasticity injection: "At onset, the last two layers of the actor are frozen; two fresh copies of
  those layers are created with identical initialisation; the actor's output becomes the sum in
  equation (8); only the new copy is trained thereafter. The output is unchanged at the moment of
  injection." Equation (8): f(x) = f_theta(x) + f_theta'1(x) - f_theta'2(x).

``R.INTERVENTION_LAYERS`` (2) is the registered depth ("the last two layers of the actor"); the
``depth`` argument serves the ablations of Table 3.3 ("reset depth one layer; injection on the last
layer only"), which are exploratory unless registered by amendment.

WHEN is decided by the Study A plug-in (``envs.onset``, Role 2): once, at the start of the rollout
of the onset epoch, after the onset checkpoint and its plasticity row are written (HANDOVER.md section 8), so
the onset metrics are pre-intervention. This module decides WHAT. The plasticity hook
(``metrics.hook``) never transforms the actor.

Answered in Table 9.1 (Q-reset-injection, a run gate of the reset and injection runs; its text in
``configs/registered.py``):

* The layers are the last ``depth`` ``nn.Linear`` modules of ``actor.mean``: for depth 2,
  ``mean[2]`` (the hidden layer before the output) and ``mean[4]`` (the output layer) of OmniSafe's
  ``GaussianLearningActor`` (``models/actor/gaussian_learning_actor.py:57-62``). ``log_std`` is a
  free parameter, not a layer: neither reset nor frozen, it keeps training and its Adam state.
* "The network's default initialiser" is OmniSafe's: ``nn.init.kaiming_uniform_(weight,
  a=sqrt(5))`` (``utils/model.py:35-36``, config ``weight_initialization_mode: kaiming_uniform``)
  and ``nn.Linear``'s own bias initialisation U(-1/sqrt(fan_in), 1/sqrt(fan_in)) (OmniSafe
  initialises only the weight), drawn layer by layer, weight then bias, from a dedicated
  ``torch.Generator`` seeded by ``intervention_seed(run seed)``. The global torch stream (action
  sampling, minibatch order) therefore stays that of the untreated partner of the same seed (Part
  5.1 pairs seed k of two arms), and the reset and injection arms of a seed receive the same fresh
  values.
* "Cleared" means ``optimizer.state.pop(p)``: Adam restarts step, exp_avg and exp_avg_sq at the
  next update; the param group, its learning rate and the LinearLR schedule are unchanged.
* Injection keeps the trunk (for depth 2 the first hidden layer) and ``log_std`` training, and appends the new
  head to the actor optimiser's existing param group (same learning rate, same LinearLR decay).
  **This departs from the literal reading of Table 2.4**: "only the new copy is trained thereafter"
  would also freeze the trunk and ``log_std``. The trunk's source is Nikishin et al. (2023),
  arXiv:2305.15555, Section 3, per two independent search-engine extracts (paper PDF not opened from
  the sandbox): the network splits into an encoder and a head; injection freezes the old head and
  adds a trained and a frozen copy of a fresh head; the gradient of every output component reaches
  the encoder, which keeps training, and the number of trainable parameters is unchanged. ``log_std``
  has no source (the networks of Nikishin et al. have no state-independent standard deviation): it
  keeps training as a symmetric choice fixed before data (``docs/DECISIONS.md``, Q-reset-injection).
  Role 3 reads the paper on the workstation before the group's ratification meeting and records any
  contradiction as an erratum.
* Equation (8) is evaluated as ``frozen(z) + (new(z) - new_frozen(z))``: with new == new_frozen the
  bracket is exactly zero, so the output is bit-identical at injection (``(f + f1) - f2`` need not be,
  because of rounding).

Module construction draws from the global torch RNG (``nn.Linear.reset_parameters``); on the live
path new modules are therefore made by ``copy.deepcopy`` of existing ones, and ``build_actor``
(offline use) constructs inside ``torch.random.fork_rng`` (HANDOVER.md section 10).
"""

from __future__ import annotations

import copy
import hashlib
import math
import numbers
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from configs import registered as R
from pilot.errors import RunRefused

# The generator's seed string (``intervention_seed``).
INTERVENTION_SEED_PREFIX = "wscl-intervention-"  # answered in Table 9.1 (Q-reset-injection)
SUPPORTED_INITIALISATION = "kaiming_uniform"  # the pinned configuration's weight_initialization_mode (PPOLag.yaml:96)
INJECTED_KEY = "mean.trunk.0.weight"  # present exactly in the pi state of an injected actor (HANDOVER.md section 8)


class InterventionError(RunRefused, ValueError):
    """An intervention or an injected-actor load cannot be applied to this actor, optimiser or state.

    A ``RunRefused``: the launcher discards the attempt and keeps the run pending (exit 5); it is
    never classified as a crash (Part 5.6).
    """


class InjectedHead(nn.Module):
    """``actor.mean`` after plasticity injection: equation (8) on top of the trainable trunk.

    ``trunk`` = the old ``mean[:split]`` and ``frozen`` = the old ``mean[split:]`` (slices of the old
    ``nn.Sequential``: the same modules and ``Parameter`` objects, with their original indices, so
    the checkpoint keys are ``mean.trunk.0.*``, ``mean.frozen.2.*``, ``mean.frozen.4.*``, ...);
    ``new`` is the trained fresh copy theta'1 and ``new_frozen`` its frozen twin theta'2.
    """

    def __init__(self, trunk: nn.Sequential, frozen: nn.Sequential, new: nn.Sequential,
                 new_frozen: nn.Sequential) -> None:
        super().__init__()
        self.trunk = trunk
        self.frozen = frozen
        self.new = new
        self.new_frozen = new_frozen

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.trunk(x)
        return self.frozen(z) + (self.new(z) - self.new_frozen(z))  # parenthesised: exact at injection


def is_injected(pi_state: Mapping[str, Any]) -> bool:
    """True if a checkpoint's ``pi`` state dict holds an injected actor (HANDOVER.md section 8: by this key only)."""
    return INJECTED_KEY in pi_state


def _whole(value: Any, what: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or int(value) < 0:
        raise ValueError(f"{what} must be a non-negative integer, got {value!r}")
    return int(value)


def intervention_seed(run_seed: int) -> int:
    """Seed of the intervention generator: the low 63 bits of sha256("wscl-intervention-{seed}"), little-endian.

    Table 9.1 (Q-reset-injection): Table 3.2 "Seeds": "The seed sets network initialisation, layout
    sampling, action sampling and minibatch order"; the pre-registration does not name the stream
    of the fresh copies. Derived from the run seed, deterministic, and independent of the global
    streams. Masked to 63 bits so that it is a non-negative signed 64-bit integer (fits torch's int64 and
    numpy's ``SeedSequence``); it is not a valid seed for numpy's legacy ``np.random.seed`` (< 2**32).
    """
    seed = _whole(run_seed, "run seed")
    digest = hashlib.sha256(f"{INTERVENTION_SEED_PREFIX}{seed}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & (2**63 - 1)


def intervention_generator(run_seed: int) -> torch.Generator:
    """A CPU ``torch.Generator`` seeded with ``intervention_seed(run_seed)``."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(intervention_seed(run_seed))
    return generator


# ---------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------


def _plain_mean(actor: nn.Module) -> nn.Sequential:
    mean = getattr(actor, "mean", None)
    if isinstance(mean, InjectedHead):
        raise InterventionError("the actor is already injected; an intervention is applied once per run "
                                "(HANDOVER.md section 8)")
    if not isinstance(mean, nn.Sequential):
        raise InterventionError(f"actor.mean must be OmniSafe's MLP (nn.Sequential), got {type(mean).__name__}")
    return mean


def _split_index(mean: nn.Sequential, depth: int) -> int:
    """Index in ``mean`` of the first of its last ``depth`` Linear layers."""
    if isinstance(depth, bool) or not isinstance(depth, numbers.Integral):
        raise InterventionError(f"depth must be an integer, got {depth!r}")
    linear = [i for i, module in enumerate(mean) if isinstance(module, nn.Linear)]
    if not 1 <= int(depth) <= len(linear):
        raise InterventionError(
            f"depth {depth} is not between 1 and {len(linear)}, the number of the actor's Linear layers")
    return linear[-int(depth)]


def _initialisation_mode(actor: nn.Module) -> str:
    mode = getattr(actor, "_weight_initialization_mode", SUPPORTED_INITIALISATION)
    if mode != SUPPORTED_INITIALISATION:
        raise InterventionError(
            f"weight_initialization_mode {mode!r}: only the pinned configuration's {SUPPORTED_INITIALISATION!r} "
            "is implemented (configs/omnisafe/PPOLag.yaml)"
        )
    return mode


def _reinitialise_(layer: nn.Linear, generator: torch.Generator) -> None:
    """OmniSafe's default initialisation of one Linear layer, drawn from ``generator`` (weight, then bias)."""
    with torch.no_grad():
        nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5), generator=generator)  # utils/model.py:35-36
        if layer.bias is not None:
            fan_in = layer.weight.shape[1]
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0  # torch nn.Linear.reset_parameters
            nn.init.uniform_(layer.bias, -bound, bound, generator=generator)


def _optimizer_params(optimizer: torch.optim.Optimizer) -> set[int]:
    return {id(p) for group in optimizer.param_groups for p in group["params"]}


def _check_optimizer(actor: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    held = _optimizer_params(optimizer)
    missing = [name for name, p in actor.named_parameters() if p.requires_grad and id(p) not in held]
    if missing:
        raise InterventionError(f"the optimiser does not hold the actor's trainable parameters {missing}")


def _group_holding(optimizer: torch.optim.Optimizer, parameter: torch.Tensor) -> dict[str, Any]:
    for group in optimizer.param_groups:
        if any(p is parameter for p in group["params"]):
            return group
    raise InterventionError("the actor optimiser does not hold the layers to be injected")


def _counts(actor: nn.Module) -> dict[str, int]:
    params = list(actor.parameters())
    return {
        "parameters": int(sum(p.numel() for p in params)),
        "trainable_parameters": int(sum(p.numel() for p in params if p.requires_grad)),
        "tensors": len(params),
        "trainable_tensors": sum(1 for p in params if p.requires_grad),
    }


def _as_input(batch: Any, mean: nn.Sequential) -> torch.Tensor | None:
    """The output-check batch as a float32 tensor, checked against ``mean`` before anything changes.

    The checks run here, not in the forward pass: the plug-in applies an intervention inside
    ``learn()`` (HANDOVER.md section 8), where a torch error would be classified as a crash (an exclusion,
    Part 5.6) instead of a refusal, and a non-finite batch would fail the bit-identity check with
    a misleading "output would change" message.
    """
    if batch is None:
        return None
    try:
        if isinstance(batch, torch.Tensor):
            x = batch.detach().to(torch.float32)
        else:
            x = torch.from_numpy(np.array(batch, dtype=np.float32, copy=True))
    except (TypeError, ValueError, OverflowError, RuntimeError) as exc:
        raise InterventionError(f"the output-check batch is not a numeric array: {exc}") from exc
    if x.ndim != 2 or x.shape[0] < 1:
        raise InterventionError(f"the output-check batch must be 2-D and non-empty (states x obs_dim), "
                                f"got shape {tuple(x.shape)}")
    first = next((m for m in mean if isinstance(m, nn.Linear)), None)
    if first is not None and x.shape[1] != first.in_features:
        raise InterventionError(
            f"the output-check batch has {x.shape[1]} observation components; the actor takes {first.in_features}"
        )
    if not bool(torch.isfinite(x).all()):
        raise InterventionError("the output-check batch contains non-finite values")
    return x


def _generator_seed(generator: torch.Generator) -> int:
    """The seed of a fresh, never drawn-from generator; the record's ``generator_seed`` names its values.

    A generator that was drawn from before would give other fresh values than its seed implies: the
    audit trail would be wrong, and the reset and injection arms of a seed would differ.
    """
    if not isinstance(generator, torch.Generator):
        raise InterventionError(f"a dedicated torch.Generator is required (HANDOVER.md section 10), "
                                f"got {type(generator).__name__}")
    seed = int(generator.initial_seed())
    fresh = torch.Generator(device=generator.device).manual_seed(seed)
    if not torch.equal(generator.get_state(), fresh.get_state()):
        raise InterventionError(
            f"the generator (seed {seed}) has already been drawn from; pass a fresh intervention_generator(run seed)"
        )
    return seed


def _linear_layers(mean: nn.Sequential, split: int) -> list[tuple[str, nn.Linear]]:
    """The Linear layers of ``mean[split:]`` with their names (indices) in ``mean``, in forward order."""
    return [(name, module) for name, module in list(mean.named_children())[split:] if isinstance(module, nn.Linear)]


def _report(kind: str, actor: nn.Module, depth: int, seed: int, before: dict[str, int], mode: str) -> dict[str, Any]:
    after = _counts(actor)
    return {
        "intervention": kind,
        "owner": "Metrics and interventions (Role 3)",
        "registered": "Table 2.4; equation (8)" if kind == "injection" else "Table 2.4",
        # The Table 9.1 key whose answer applies (answered); the field keeps the name HANDOVER.md section 8 gives it.
        "proposal_key": "Q-reset-injection",
        "depth": int(depth),
        "generator_seed": seed,
        "weight_initialization_mode": mode,
        "parameter_count_before": before["parameters"],
        "parameter_count_after": after["parameters"],
        "trainable_parameter_count_before": before["trainable_parameters"],
        "trainable_parameter_count_after": after["trainable_parameters"],
        "parameter_tensors_before": before["tensors"],
        "parameter_tensors_after": after["tensors"],
        "trainable_tensors_before": before["trainable_tensors"],
        "trainable_tensors_after": after["trainable_tensors"],
        "trainable": [name for name, p in actor.named_parameters() if p.requires_grad],
    }


# ---------------------------------------------------------------------------------------------
# The two interventions (Table 2.4)
# ---------------------------------------------------------------------------------------------


def partial_reset(actor: nn.Module, optimizer: torch.optim.Optimizer, generator: torch.Generator,
                  depth: int = R.INTERVENTION_LAYERS, *, batch: Any = None) -> dict[str, Any]:
    """Reinitialise the actor's last ``depth`` layers in place and clear their optimiser state.

    Table 2.4 "Partial reset" (see the module docstring for Table 9.1's answer, Q-reset-injection).
    The same ``Parameter`` objects are overwritten (``torch.no_grad``), so the optimiser's param
    group is untouched; only the reset parameters' Adam entries are removed. ``log_std``, the layers
    before the reset ones (at the registered depth 2, the first hidden layer) and the critics are
    untouched. ``generator`` must be fresh (never drawn from). ``batch`` (optional; the plug-in passes the fixed
    batch as the metrics see it) gives ``output_max_abs_change``, the largest change of
    ``actor.mean`` on it. Returns the record the plug-in writes to ``intervention.json``.
    """
    seed = _generator_seed(generator)
    mean = _plain_mean(actor)
    mode = _initialisation_mode(actor)
    _check_optimizer(actor, optimizer)
    split = _split_index(mean, depth)
    x = _as_input(batch, mean)
    rng_before = torch.get_rng_state()
    before = _counts(actor)
    with torch.no_grad():
        output_before = mean(x) if x is not None else None
    layers = _linear_layers(mean, split)
    reset_parameters: list[str] = []
    cleared: list[str] = []
    for name, layer in layers:  # forward order (depth 2: the hidden layer before the output, then the output layer)
        _reinitialise_(layer, generator)
        for pname, p in layer.named_parameters():
            reset_parameters.append(f"mean.{name}.{pname}")
            p.grad = None  # a stale gradient of the old weights must never be applied to the new ones
            if optimizer.state.pop(p, None) is not None:
                cleared.append(f"mean.{name}.{pname}")
    with torch.no_grad():
        output_after = mean(x) if x is not None else None
    untouched = torch.equal(rng_before, torch.get_rng_state())
    if not untouched:  # a programming error: the treated run would drift from its untreated partner
        raise InterventionError("partial reset drew from the global torch RNG")
    record = _report("reset", actor, depth, seed, before, mode)
    record.update({
        "layers": [f"mean.{name}" for name, _ in layers],
        "reinitialised": reset_parameters,
        "adam_entries_cleared": len(cleared),
        "adam_entries_cleared_names": cleared,
        "output_max_abs_change": None if x is None else float((output_after - output_before).abs().max()),
        "global_rng_untouched": untouched,
    })
    return record


def plasticity_injection(actor: nn.Module, optimizer: torch.optim.Optimizer, generator: torch.Generator,
                         depth: int = R.INTERVENTION_LAYERS, *, batch: Any = None) -> dict[str, Any]:
    """Replace ``actor.mean`` in place by an ``InjectedHead`` (equation (8)) and extend the optimiser.

    Table 2.4 "Plasticity injection". ``frozen`` (theta) and ``new_frozen`` (theta'2) get
    ``requires_grad=False``; ``new`` (theta'1, fresh weights from ``generator``) is appended to the
    optimiser's param group that holds the copied layers (OmniSafe has one), so it shares its
    learning rate and LinearLR decay. The trunk and ``log_std`` keep training (Table 9.1,
    Q-reset-injection; departs from the literal Table 2.4 reading, see the module docstring). Frozen
    parameters stay in the optimiser and keep their Adam state, but receive no gradient, so Adam
    never updates them. ``generator`` must be fresh (never drawn from).

    ``batch`` (optional): the output of ``actor.mean`` before and after is compared on it, and the
    injection is refused (``InterventionError``) before anything changes unless it is bit-identical
    ("The output is unchanged at the moment of injection"). Returns the ``intervention.json`` record.
    """
    seed = _generator_seed(generator)
    mean = _plain_mean(actor)
    mode = _initialisation_mode(actor)
    _check_optimizer(actor, optimizer)
    split = _split_index(mean, depth)
    if split == 0:  # the injected layout is recognised by its trunk (mean.trunk.0.weight, HANDOVER.md section 8)
        raise InterventionError(f"injection of depth {depth} would leave no trunk; "
                                f"at most {depth - 1} layers can be injected")
    group = _group_holding(optimizer, next(mean[split].parameters()))
    x = _as_input(batch, mean)
    rng_before = torch.get_rng_state()
    before = _counts(actor)
    trunk, frozen = mean[:split], mean[split:]  # slices keep the modules, their Parameters and their indices
    new = copy.deepcopy(frozen)  # deepcopy, not construction: no draw from the global torch RNG
    for module in new:
        if isinstance(module, nn.Linear):
            _reinitialise_(module, generator)
    new_frozen = copy.deepcopy(new)  # "two fresh copies of those layers ... with identical initialisation"
    head = InjectedHead(trunk, frozen, new, new_frozen)
    output_change = None
    if x is not None:
        with torch.no_grad():
            reference, injected = mean(x), head(x)
        if not torch.equal(reference, injected):
            raise InterventionError(
                "injection would change the actor's output (max |difference| "
                f"{float((injected - reference).abs().max())}); Table 2.4: the output is unchanged at injection"
            )
        output_change = 0.0
    frozen.requires_grad_(False)
    new_frozen.requires_grad_(False)
    new.requires_grad_(True)
    for p in (*frozen.parameters(), *new.parameters(), *new_frozen.parameters()):
        p.grad = None  # no stale gradient of a frozen parameter can reach the optimiser
    actor.mean = head  # same actor object: forward, predict, log_prob and KL early stopping keep working
    group["params"].extend(new.parameters())  # the group of the layers it copies: same lr and LinearLR decay
    untouched = torch.equal(rng_before, torch.get_rng_state())
    if not untouched:
        raise InterventionError("plasticity injection drew from the global torch RNG")
    record = _report("injection", actor, depth, seed, before, mode)
    record.update({
        "layers": [f"mean.{name}" for name, _ in _linear_layers(mean, split)],
        "frozen": [f"mean.{n}" for n, p in head.named_parameters() if n.startswith(("frozen.", "new_frozen."))],
        "added_to_optimizer": [f"mean.new.{n}" for n, _ in new.named_parameters()],
        "output_max_abs_change": output_change,
        "output_identical": None if x is None else True,
        "global_rng_untouched": untouched,
        "reading": "Table 9.1 (Q-reset-injection): the trunk keeps training, as in Nikishin et al. (2023), "
                   "arXiv:2305.15555, Section 3, per two independent search-engine extracts (paper PDF not opened "
                   "from the sandbox); log_std keeps training too, a symmetric choice fixed before data that no "
                   "source gives; the literal Table 2.4 reading ('only the new copy is trained thereafter') would "
                   "freeze them too",
    })
    return record


# The fields of results/supplement_schema.py InterventionSummary (the "training" record), in order.
SUMMARY_FIELDS = ("treatment", "step", "layers", "generator_seed", "trainable_parameters_before",
                  "trainable_parameters_after", "max_output_difference")


def intervention_summary(record: Mapping[str, Any], *, step: int) -> dict[str, Any]:
    """The ``InterventionSummary`` of the ``training`` supplement record (results/supplement_schema.py) for a record.

    The record returned by ``partial_reset`` / ``plasticity_injection`` (and written by the Study A
    plug-in, HANDOVER.md section 8) names its fields for the audit; the supplement model
    (``results/supplement_schema.py``, Role 4) names them differently, and this is the one mapping:
    ``intervention`` -> ``treatment``, ``trainable_parameter_count_before``/``_after`` ->
    ``trainable_parameters_before``/``_after``, ``output_max_abs_change`` ->
    ``max_output_difference``; ``layers`` and ``generator_seed`` keep their names. ``step`` is not
    in the record (the functions do not know the schedule): the caller passes the step at which
    the plug-in applied it, the spec's ``onset_step`` (Table 2.4 "At onset"). Returns exactly
    ``SUMMARY_FIELDS``; raises ``ValueError`` for a record that is not an intervention record.
    """
    if not isinstance(record, Mapping):
        raise ValueError(f"not an intervention record: {type(record).__name__}")
    treatment = record.get("intervention")
    if treatment not in ("reset", "injection"):
        raise ValueError(f"not an intervention record: 'intervention' is {treatment!r}")
    layers = record.get("layers")
    if not isinstance(layers, (list, tuple)) or not layers or not all(isinstance(n, str) for n in layers):
        raise ValueError(f"the record's layers must be a non-empty list of names, got {layers!r}")
    change = record.get("output_max_abs_change")
    if change is not None:
        if (isinstance(change, (bool, np.bool_)) or not isinstance(change, numbers.Real)
                or not math.isfinite(float(change)) or float(change) < 0.0):
            raise ValueError(f"the record's output_max_abs_change must be a finite non-negative number, got {change!r}")
        change = float(change)
    return {
        "treatment": treatment,
        "step": _whole(step, "step"),
        "layers": list(layers),
        "generator_seed": _whole(record.get("generator_seed"), "generator_seed"),
        "trainable_parameters_before": _whole(record.get("trainable_parameter_count_before"),
                                              "trainable_parameter_count_before"),
        "trainable_parameters_after": _whole(record.get("trainable_parameter_count_after"),
                                             "trainable_parameter_count_after"),
        "max_output_difference": change,
    }


# ---------------------------------------------------------------------------------------------
# Loading injected actors (evaluation harness, continuations, offline metrics)
# ---------------------------------------------------------------------------------------------


def _injected_split(pi_state: Mapping[str, Any]) -> int:
    indices: set[int] = set()
    for key in pi_state:
        parts = key.split(".")
        if len(parts) >= 3 and parts[0] == "mean" and parts[1] == "frozen" and parts[2].isdigit():
            indices.add(int(parts[2]))
    if not indices:
        raise InterventionError("injected pi state has no mean.frozen.* entries")
    return min(indices)


def rebuild_injected(actor: nn.Module, pi_state: Mapping[str, Any]) -> None:
    """Restructure a plain actor in place so that ``actor.load_state_dict(pi_state)`` accepts an injected state.

    Used by continuations of injection arms (``pilot.dependencies.restore_learner``) and, through
    ``build_actor``, by the evaluation harness (HANDOVER.md section 8). Built by ``copy.deepcopy`` of
    the actor's own modules (no draw from any global random stream); the
    values are those of the fresh actor until the caller loads ``pi_state``. ``requires_grad``
    follows the parent layout: ``frozen`` and ``new_frozen`` False, trunk and ``new`` trainable.
    Refused (``InterventionError``) unless the rebuilt keys and shapes equal ``pi_state``'s.
    """
    if not is_injected(pi_state):
        raise InterventionError(f"pi state is not injected (no {INJECTED_KEY!r}); load it into the plain actor")
    mean = _plain_mean(actor)
    split = _injected_split(pi_state)
    if split not in range(1, len(mean)) or not isinstance(mean[split], nn.Linear):
        raise InterventionError(
            f"injected pi state splits actor.mean at {split}: not a Linear layer of it after the first "
            "(the trunk would be empty or the index is out of range)"
        )
    trunk, frozen = mean[:split], mean[split:]
    head = InjectedHead(trunk, frozen, copy.deepcopy(frozen), copy.deepcopy(frozen))
    expected = {f"mean.{k}": tuple(v.shape) for k, v in head.state_dict().items()}
    expected.update({k: tuple(v.shape) for k, v in actor.state_dict().items() if not k.startswith("mean.")})
    got = {k: tuple(torch.as_tensor(v).shape) for k, v in pi_state.items()}
    if expected != got:
        missing = sorted(set(expected) - set(got))
        unexpected = sorted(set(got) - set(expected))
        wrong = sorted(k for k in set(expected) & set(got) if expected[k] != got[k])
        raise InterventionError(
            f"injected pi state does not fit this actor: missing {missing}, unexpected {unexpected}, "
            f"shapes differ {wrong}"
        )
    head.frozen.requires_grad_(False)
    head.new_frozen.requires_grad_(False)
    head.new.requires_grad_(True)
    actor.mean = head


def _model_cfgs(model_cfgs_or_config: Mapping[str, Any]) -> Mapping[str, Any]:
    if "model_cfgs" in model_cfgs_or_config:
        return model_cfgs_or_config["model_cfgs"]
    return model_cfgs_or_config


def build_actor(model_cfgs_or_config: Mapping[str, Any], obs_dim: int, act_dim: int,
                pi_state: Mapping[str, Any]) -> nn.Module:
    """An eval-mode OmniSafe actor holding ``pi_state`` (plain or injected), built without the global torch RNG.

    ``model_cfgs_or_config`` is OmniSafe's ``model_cfgs`` or a whole configuration (``config.json``
    of the run) that contains it. The actor is built by OmniSafe's ``ActorBuilder`` exactly as in
    training (``models/actor_critic/actor_critic.py:70-78``) inside ``torch.random.fork_rng``,
    injected layouts are rebuilt with ``rebuild_injected``, and ``pi_state`` is loaded strictly.
    For evaluation and offline metrics; continuations restore into their own algorithm
    (``pilot.dependencies.restore_learner``).
    """
    cfg = _model_cfgs(model_cfgs_or_config)
    actor_type = cfg.get("actor_type", "gaussian_learning")
    if actor_type != "gaussian_learning":
        raise InterventionError(f"actor_type {actor_type!r}: the registered configuration uses gaussian_learning")
    actor_cfg = cfg["actor"]
    hidden_sizes = [int(n) for n in actor_cfg["hidden_sizes"]]
    activation = actor_cfg["activation"]
    mode = cfg.get("weight_initialization_mode", SUPPORTED_INITIALISATION)
    with torch.random.fork_rng(devices=[]):  # nn.Linear construction draws from the global torch RNG
        from gymnasium import spaces
        from omnisafe.models.actor.actor_builder import ActorBuilder

        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(int(obs_dim),), dtype=np.float32)
        act_space = spaces.Box(low=-1.0, high=1.0, shape=(int(act_dim),), dtype=np.float32)
        actor = ActorBuilder(obs_space, act_space, hidden_sizes, activation=activation,
                             weight_initialization_mode=mode).build_actor(actor_type)
    if is_injected(pi_state):
        rebuild_injected(actor, pi_state)
    try:
        actor.load_state_dict(dict(pi_state), strict=True)
    except RuntimeError as exc:
        raise InterventionError(f"pi state does not fit the configured actor: {exc}") from exc
    actor.eval()
    return actor

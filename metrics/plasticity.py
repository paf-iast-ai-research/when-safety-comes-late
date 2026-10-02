"""The plasticity metrics of Study A (Role 3, Metrics and interventions; owner Abdullah).

Implements Table 2.3 (PDF p. 8) and equations (6) and (7) (Part 2.5, p. 11):

* "Network measured": "The actor. Dormant-neuron fraction is computed over all hidden units of the
  actor; effective rank over the actor's penultimate-layer features; parameter norm over all
  trainable parameters of the actor, with the two critics logged separately."
* Equation (6), dormant-neuron score: s_i^l = E|h_i^l(x)| / ((1/H^l) sum_k E|h_k^l(x)|); "a neuron
  is dormant if the score is at most τ = 0.025" (``R.DORMANT_THRESHOLD``).
* Equation (7), effective rank: srank_delta(Phi) = min{k : sum_{i<=k} sigma_i / sum_i sigma_i >= 1 -
  delta}, delta = 0.01 (``R.EFFECTIVE_RANK_DELTA``).
* "Parameter norm": "The Euclidean norm of all trainable parameters."
* H3 (c) (Part 1.2): the dormant fraction of the trainable units and the effective rank of the
  trainable layers' features ("the reinitialised last two layers for reset, the new head for
  injection, and the last two layers for the untreated arm").

Answered in Table 9.1 (Q-plasticity-definitions, its text in ``configs/registered.py``; a run gate
of the non-pilot Study A runs; the pilot is not gated): the input is the fixed batch normalised by
the run's own observation normaliser at the measured checkpoint, frozen (``normalise_frozen``;
OmniSafe's ``Normalizer.normalize`` would push the batch into the running statistics and change
training); activations after tanh (First Tasks section 9 step 3); the dormant fraction pooled over
all hidden units, each layer scored against its own mean; the effective rank of the last hidden
layer's activations by uncentred float64 SVD; ``log_std`` included in the norm (it is a trainable
parameter of the actor); after injection ``norm`` covers the trainable parameters (Table 2.3:
"all trainable parameters") and the extra column ``norm_all`` every parameter of the actor.

The answers of the two keys also fix these details (their texts in ``configs/registered.py``, which
the runs whose values depend on them carry as run gates):

* Q-plasticity-definitions (zero and non-finite input, the critics, H3 (c)): "an all-zero layer
  counts as all dormant" (equation 6 is undefined when a layer's mean activation is 0), "an
  all-zero matrix has rank 0", "non-finite input gives NaN" (dormant fraction and rank); the "same
  three metrics" are "logged for each critic" ("with the two critics logged separately", Table
  2.3, read as extra columns, not only their norms); for H3 (c) the trainable-layer quantities
  (``dormant_trainable``, ``rank_trainable``) are those of "the second hidden layer" (``mean[3]``:
  the "last two layers" hold one hidden layer) "for the untreated and reset arms" and of "the new
  head's hidden layer" (``new.3``) for injection.
* Q-reset-injection, an injected actor: ``dormant`` over "all 256 hidden units (trunk, frozen
  head, new head" and its "frozen copy") in the forward path, and ``rank`` of "the three heads'
  concatenated last hidden layers" (192 features; equation 8 is linear in them). No hypothesis uses
  the post-onset ``dormant``/``rank`` of injection arms (H3 (a) uses the pre-intervention onset row,
  H3 (c) the ``*_trainable`` columns), but the checkpoint records of their ledger rows hold them.

Only torch and numpy are needed (no OmniSafe): the metric functions are testable without the RL
stack. Nothing here draws a random number, changes a parameter, a buffer or a module's mode, or
leaves a hook registered (HANDOVER.md section 10, "Instrumentation never changes training").
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from configs import registered as R
from metrics.interventions import InjectedHead

# Columns of plasticity.csv (contract 2, pilot/contracts.py). The ledger writer reads the first four (ledger
# schema v1 has no field for the others); the rows at onset and at onset + 200,000 steps reach the analysis whole
# through the ``training`` supplement record (results/supplement_schema.py PlasticityRow; Q-ledger-v2, answered in
# docs/DECISIONS.md). The comments name where each definition comes from; a Q- key names the answer stated in
# that key's text in configs/registered.py (Table 9.1; module docstring).
COLUMNS = (
    "step",
    "dormant",  # actor dormant-neuron fraction, all hidden units (Table 2.3; eq. 6); injected: Q-reset-injection
    "rank",  # actor effective rank of the penultimate features (Table 2.3; eq. 7); injected: Q-reset-injection
    "norm",  # actor norm over trainable parameters, log_std included (Table 2.3)
    "norm_reward_critic",  # Table 2.3 "with the two critics logged separately" (names of contract 2)
    "norm_cost_critic",
    "dormant_trainable",  # H3 (c); layer (mean[3] untreated and reset; new.3 injection): Q-plasticity-definitions
    "rank_trainable",  # H3 (c): effective rank of the trainable layer's features; same mapping
    "dormant_reward_critic",  # Q-plasticity-definitions: all three metrics for each critic
    "rank_reward_critic",
    "dormant_cost_critic",
    "rank_cost_critic",
    "norm_all",  # actor norm over every parameter; differs from norm only after injection (module docstring)
)
INTEGER_COLUMNS = frozenset({"step", "rank", "rank_trainable", "rank_reward_critic", "rank_cost_critic"})


# ---------------------------------------------------------------------------------------------
# Input: the fixed batch as the network sees it in training
# ---------------------------------------------------------------------------------------------


def _field(normalizer: Any, name: str) -> torch.Tensor:
    value = normalizer[name] if isinstance(normalizer, Mapping) else getattr(normalizer, name)
    return torch.as_tensor(value)


def normalise_frozen(x: torch.Tensor, normalizer: Any | None) -> torch.Tensor:
    """OmniSafe's observation normalisation with the statistics frozen (no update).

    Reproduces ``Normalizer.normalize``'s output rule (``omnisafe/common/normalizer.py:102-107``):
    the input unchanged while ``_count <= 1``, else ``clamp((x - _mean) / _std, -_clip, _clip)``.
    Reads the buffers ``_mean``, ``_std``, ``_clip`` and ``_count`` of an OmniSafe ``Normalizer``
    or of its state dict (as saved in ``epoch-k.pt``); never calls ``normalize``, which pushes the
    data into the running statistics. ``normalizer=None`` means raw input (``obs_normalize: False``).
    """
    if normalizer is None:
        return x
    mean = _field(normalizer, "_mean")
    if tuple(mean.shape) != tuple(x.shape[1:]):  # checked first: at every checkpoint, step 0 included
        raise ValueError(f"normaliser shape {tuple(mean.shape)} does not match the batch's {tuple(x.shape[1:])}")
    if int(_field(normalizer, "_count")) <= 1:
        return x
    std, clip = _field(normalizer, "_std"), _field(normalizer, "_clip")
    output = (x.to(mean.device) - mean) / std
    return torch.clamp(output, -clip, clip)


def as_batch_tensor(batch: Any) -> torch.Tensor:
    """A float32 CPU tensor (states x obs_dim) holding a copy of ``batch``."""
    if isinstance(batch, torch.Tensor):
        x = batch.detach().to(device="cpu", dtype=torch.float32).clone()
    else:
        x = torch.from_numpy(np.array(batch, dtype=np.float32, copy=True))
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 1:
        raise ValueError(f"the batch must be a non-empty 2-D array (states x obs_dim), got shape {tuple(x.shape)}")
    return x


# ---------------------------------------------------------------------------------------------
# Which layers are hidden
# ---------------------------------------------------------------------------------------------


def _sequential_hidden(seq: nn.Sequential, prefix: str, *, ends_in_output: bool) -> list[str]:
    """Names of the activation modules after the hidden Linear layers of an OmniSafe MLP.

    ``build_mlp_network`` (``omnisafe/utils/model.py:103-111``) builds [Linear, act, ..., Linear,
    Identity]; the last Linear is the output layer and is never hidden (First Tasks section 9, Mistakes
    to avoid: "Counting the output layer as hidden"). A trunk (``ends_in_output=False``) has only hidden layers.
    """
    names = list(seq._modules)
    modules = list(seq._modules.values())
    linear = [i for i, m in enumerate(modules) if isinstance(m, nn.Linear)]
    hidden = linear[:-1] if ends_in_output else linear
    out = []
    for i in hidden:
        if i + 1 >= len(modules) or isinstance(modules[i + 1], (nn.Linear, nn.Identity)):
            raise ValueError(f"hidden layer {prefix}{names[i]} has no activation module after it")
        out.append(f"{prefix}{names[i + 1]}")
    return out


@dataclass(frozen=True)
class Layers:
    """Module names (relative to the measured network) of its hidden layers, by role."""

    hidden: tuple[str, ...]  # every hidden layer in the forward path (dormant fraction)
    penultimate: tuple[str, ...]  # features whose linear read-out is the output (effective rank; concatenated)
    trainable: tuple[str, ...]  # H3 (c) trainable layers (may be empty)


def actor_layers(mean: nn.Module) -> Layers:
    """The hidden layers of ``actor.mean``, plain or injected.

    Plain (``mean`` = Linear, Tanh, Linear, Tanh, Linear, Identity): hidden ``1`` and ``3``;
    penultimate and trainable ``3`` (H3 (c): the untreated and the reset arm's "last two layers"
    contain one hidden layer, the one before the output). Injected: hidden ``trunk.1``,
    ``frozen.3``, ``new.3``, ``new_frozen.3``; penultimate the three heads' ``.3``; trainable
    ``new.3`` ("the new head for injection"). A head without a hidden layer (injection of the output
    layer only, an ablation) has the trunk's last layer as penultimate and no trainable layer.
    The trainable-layer mapping and the injected layout are answered in Table 9.1
    (Q-plasticity-definitions and Q-reset-injection; module docstring).
    """
    if isinstance(mean, InjectedHead):
        trunk = _sequential_hidden(mean.trunk, "trunk.", ends_in_output=False)
        heads = [_sequential_hidden(getattr(mean, n), f"{n}.", ends_in_output=True)
                 for n in ("frozen", "new", "new_frozen")]
        hidden = tuple(trunk + [name for head in heads for name in head])
        if heads[0]:
            penultimate = tuple(head[-1] for head in heads)
            trainable = (heads[1][-1],)
        else:
            penultimate, trainable = (trunk[-1],), ()
        return Layers(hidden, penultimate, trainable)
    if isinstance(mean, nn.Sequential):
        hidden = tuple(_sequential_hidden(mean, "", ends_in_output=True))
        if not hidden:
            raise ValueError("the actor has no hidden layer")
        return Layers(hidden, (hidden[-1],), (hidden[-1],))
    raise TypeError(f"cannot find the hidden layers of {type(mean).__name__}")


def critic_network(critic: nn.Module) -> nn.Sequential:
    """The MLP of an OmniSafe ``VCritic`` (``critic_0``, ``models/critic/v_critic.py:66-73``) or a bare MLP."""
    net = getattr(critic, "critic_0", critic)
    if not isinstance(net, nn.Sequential):
        raise TypeError(f"cannot find the MLP of critic {type(critic).__name__}")
    return net


def critic_layers(critic: nn.Module) -> Layers:
    """Hidden layers of a V critic's MLP: the last one is penultimate; no trainable-layer subset."""
    hidden = tuple(_sequential_hidden(critic_network(critic), "", ends_in_output=True))
    if not hidden:
        raise ValueError("the critic has no hidden layer")
    return Layers(hidden, (hidden[-1],), ())


def hidden_layer_names(net: nn.Module) -> tuple[str, ...]:
    """Every hidden layer of an actor mean (plain or injected) or an MLP."""
    return actor_layers(net).hidden


def hidden_activations(net: nn.Module, x: torch.Tensor, names: Sequence[str] | None = None) -> dict[str, torch.Tensor]:
    """Post-activation outputs of the named modules of ``net`` on ``x`` (First Tasks section 9 step 3).

    Forward hooks capture each named module's output during one ``torch.no_grad`` forward pass; the
    hooks are removed in ``finally`` whatever happens. Call it on ``actor.mean`` (never ``actor(x)``
    or ``actor.predict``, which set the actor's ``_current_dist`` and may sample).
    """
    names = tuple(hidden_layer_names(net) if names is None else names)
    modules = dict(net.named_modules())
    unknown = [n for n in names if n not in modules]
    if unknown:
        raise ValueError(f"no modules named {unknown} in {type(net).__name__}")
    captured: dict[str, torch.Tensor] = {}

    def capture(name: str) -> Callable[[nn.Module, Any, torch.Tensor], None]:
        def hook(_module: nn.Module, _inputs: Any, output: torch.Tensor) -> None:
            if name in captured:
                raise RuntimeError(f"module {name} ran twice in one forward pass")
            captured[name] = output.detach()
        return hook

    handles = []
    try:
        for name in names:
            handles.append(modules[name].register_forward_hook(capture(name)))
        with torch.no_grad():
            net(x)
    finally:
        for handle in handles:
            handle.remove()
    missing = [n for n in names if n not in captured]
    if missing:
        raise RuntimeError(f"modules {missing} did not run in the forward pass")
    return {n: captured[n] for n in names}


# ---------------------------------------------------------------------------------------------
# Equations (6) and (7) and the norm
# ---------------------------------------------------------------------------------------------


def dormant_scores(h: torch.Tensor) -> torch.Tensor:
    """Equation (6) for one layer ``h`` (states x units): mean |activation| over the batch / the layer's mean.

    float64. A layer whose mean is 0 has every score 0 (all dormant); a non-finite activation makes
    every score NaN (equation 6 is undefined there; Table 9.1, Q-plasticity-definitions, a run gate
    of the non-pilot Study A runs).
    """
    if h.ndim != 2 or h.shape[1] < 1:
        raise ValueError(f"a layer's activations must be 2-D (states x units), got shape {tuple(h.shape)}")
    h64 = h.detach().to(torch.float64)
    if not bool(torch.isfinite(h64).all()):
        return torch.full((h.shape[1],), math.nan, dtype=torch.float64)
    a = h64.abs().mean(dim=0)
    layer_mean = a.mean()
    if float(layer_mean) == 0.0:
        return torch.zeros_like(a)
    return a / layer_mean


def dormant_fraction(layers: Iterable[torch.Tensor], tau: float = R.DORMANT_THRESHOLD) -> float:
    """Share of the hidden units whose equation (6) score is at most ``tau``, pooled over ``layers``.

    Each layer is scored against its own mean (equation 6); the count is pooled over all units
    (Table 2.3 "computed over all hidden units"; First Tasks section 9 step 4: "over all 128 hidden units"). NaN if
    any score is NaN.
    """
    dormant = total = 0
    nan = False
    for h in layers:
        scores = dormant_scores(h)
        if bool(torch.isnan(scores).any()):
            nan = True
        dormant += int((scores <= tau).sum())
        total += scores.numel()
    if total == 0:
        raise ValueError("no hidden units to score")
    return math.nan if nan else dormant / total


def effective_rank(phi: torch.Tensor, delta: float = R.EFFECTIVE_RANK_DELTA) -> int | float:
    """Equation (7): the fewest largest singular values of ``phi`` carrying a share 1 - delta of their sum.

    Uncentred (equation 7 and First Tasks section 9 step 5 take the activations as they are), float64 SVD.
    Returns an int; 0 for an all-zero matrix; NaN (float) for a non-finite one, without calling the
    SVD, which raises on non-finite input (Table 9.1, Q-plasticity-definitions, a run gate of the
    non-pilot Study A runs).
    """
    if phi.ndim != 2 or phi.numel() == 0:
        raise ValueError(f"the feature matrix must be a non-empty 2-D array, got shape {tuple(phi.shape)}")
    phi64 = phi.detach().to(torch.float64)
    if not bool(torch.isfinite(phi64).all()):
        return math.nan
    sigma = torch.linalg.svdvals(phi64)  # descending
    cumulative = torch.cumsum(sigma, dim=0)
    total = cumulative[-1]  # the running sum's own total, so the last share is exactly 1
    if float(total) == 0.0:
        return 0
    share = cumulative / total
    return int((share < 1.0 - delta).sum()) + 1


def parameter_norm(params: nn.Module | Iterable[torch.Tensor], *, trainable_only: bool = True) -> float:
    """Euclidean norm (float64) of the parameters that require gradients, or of all with ``trainable_only=False``.

    Table 2.3 "Parameter norm": "The Euclidean norm of all trainable parameters." For the actor
    this includes ``log_std`` (First Tasks section 9 step 6: "all the actor's parameters").
    """
    tensors = params.parameters() if isinstance(params, nn.Module) else params
    squares = [float(p.detach().to(torch.float64).pow(2).sum())
               for p in tensors if p.requires_grad or not trainable_only]
    return math.sqrt(math.fsum(squares))


# ---------------------------------------------------------------------------------------------
# One row of plasticity.csv
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class NetworkParts:
    """What ``measure`` reads: the actor, the two critics (optional) and the observation normaliser.

    ``normalizer``: an OmniSafe ``Normalizer``, its state dict (``epoch-k.pt['obs_normalizer']``),
    or None for raw input.
    """

    actor: nn.Module
    reward_critic: nn.Module | None = None
    cost_critic: nn.Module | None = None
    normalizer: Any = None


def network_parts(source: Any) -> NetworkParts:
    """``NetworkParts`` from itself, a mapping with those keys, or a live OmniSafe algorithm.

    From an algorithm: ``_actor_critic.actor``, ``.reward_critic``, ``.cost_critic`` (if any) and,
    when ``algo_cfgs.obs_normalize``, the normaliser object ``_env.save()['obs_normalizer']`` (read,
    never called; ``omnisafe/envs/wrapper.py:276``). Read at each call, so an injected ``actor.mean``
    and the current normaliser statistics are what is measured.
    """
    if isinstance(source, NetworkParts):
        return source
    if isinstance(source, Mapping):
        return NetworkParts(source["actor"], source.get("reward_critic"), source.get("cost_critic"),
                            source.get("normalizer"))
    ac = getattr(source, "_actor_critic", None)
    if ac is None:
        raise TypeError(f"cannot find the networks of {type(source).__name__}")
    normalizer = None
    if bool(source._cfgs.algo_cfgs.obs_normalize):
        normalizer = source._env.save()["obs_normalizer"]
    return NetworkParts(ac.actor, getattr(ac, "reward_critic", None), getattr(ac, "cost_critic", None), normalizer)


def _layer_metrics(net: nn.Module, layers: Layers, x: torch.Tensor, tau: float, delta: float
                   ) -> tuple[float, int | float, float, int | float]:
    """(dormant, rank, dormant over trainable, rank of trainable) of ``net`` on ``x``."""
    acts = hidden_activations(net, x, layers.hidden)
    dormant = dormant_fraction([acts[n] for n in layers.hidden], tau)
    rank = effective_rank(torch.cat([acts[n] for n in layers.penultimate], dim=1), delta)
    if layers.trainable:
        dormant_t = dormant_fraction([acts[n] for n in layers.trainable], tau)
        rank_t = effective_rank(torch.cat([acts[n] for n in layers.trainable], dim=1), delta)
    else:
        dormant_t, rank_t = math.nan, math.nan
    return dormant, rank, dormant_t, rank_t


def measure(source: Any, batch: Any, *, step: int, tau: float = R.DORMANT_THRESHOLD,
            delta: float = R.EFFECTIVE_RANK_DELTA) -> dict[str, int | float]:
    """One plasticity.csv row (``COLUMNS``) of the networks of ``source`` on the raw fixed ``batch``.

    ``source``: a live OmniSafe algorithm, ``NetworkParts`` or a mapping (see ``network_parts``).
    The batch is normalised by ``normalise_frozen`` with the source's normaliser and fed to
    ``actor.mean`` and to each critic's MLP. A missing critic gives NaN in its columns.
    """
    if isinstance(step, bool) or not isinstance(step, (int, np.integer)) or int(step) < 0:
        raise ValueError(f"step must be a non-negative integer, got {step!r}")
    parts = network_parts(source)
    x = normalise_frozen(as_batch_tensor(batch), parts.normalizer)
    mean = parts.actor.mean
    dormant, rank, dormant_t, rank_t = _layer_metrics(mean, actor_layers(mean), x, tau, delta)
    row: dict[str, int | float] = {
        "step": int(step),
        "dormant": dormant,
        "rank": rank,
        "norm": parameter_norm(parts.actor, trainable_only=True),
    }
    critics: dict[str, tuple[float, int | float, float]] = {}
    for label, critic in (("reward_critic", parts.reward_critic), ("cost_critic", parts.cost_critic)):
        if critic is None:
            critics[label] = (math.nan, math.nan, math.nan)
            continue
        c_dormant, c_rank, _, _ = _layer_metrics(critic_network(critic), critic_layers(critic), x, tau, delta)
        critics[label] = (c_dormant, c_rank, parameter_norm(critic, trainable_only=True))
    row["norm_reward_critic"] = critics["reward_critic"][2]
    row["norm_cost_critic"] = critics["cost_critic"][2]
    row["dormant_trainable"] = dormant_t
    row["rank_trainable"] = rank_t
    row["dormant_reward_critic"], row["rank_reward_critic"] = critics["reward_critic"][:2]
    row["dormant_cost_critic"], row["rank_cost_critic"] = critics["cost_critic"][:2]
    row["norm_all"] = parameter_norm(parts.actor, trainable_only=False)
    return {c: row[c] for c in COLUMNS}


def nan_row(step: int) -> dict[str, int | float]:
    """The row written when measuring failed: the step and NaN everywhere else."""
    return {c: (int(step) if c == "step" else math.nan) for c in COLUMNS}


def format_value(column: str, value: int | float) -> str:
    """CSV text: integers as decimal digits, floats as ``repr`` (the shortest exact round trip; nan, inf)."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{column}: boolean value")
    if column in INTEGER_COLUMNS and isinstance(value, (int, np.integer)):
        return str(int(value))
    value = float(value)
    if column in INTEGER_COLUMNS and math.isfinite(value) and value.is_integer():
        return str(int(value))
    return repr(value)


def format_row(row: Mapping[str, int | float]) -> list[str]:
    """The CSV fields of a row whose columns are exactly ``COLUMNS``, in order (``format_value``)."""
    if tuple(row) != COLUMNS:
        raise ValueError(f"row columns {tuple(row)} differ from {COLUMNS}")
    return [format_value(c, row[c]) for c in COLUMNS]


def parse_row(record: Mapping[str, str]) -> dict[str, int | float]:
    """The inverse of ``format_row`` for a ``csv.DictReader`` record, up to type.

    ``step`` comes back as an int and every other column as a float, the integer rank columns
    included (47 as 47.0; NaN where a rank was not computed); values compare equal to the written ones.
    """
    out: dict[str, int | float] = {}
    for c in COLUMNS:
        text = record[c]
        out[c] = int(text) if c == "step" else float(text)
    return out

"""Plasticity metrics (Role 3): equations (6) and (7), the norm, the frozen normaliser, ``measure``.

Table 2.3; equations (6) and (7); First Tasks section 9 tests ("A small network whose weights are set
so that a known number of hidden units output zero ...", "ten equal and the rest zero: the effective
rank is 10", "the parameter norm equals the value computed by hand", "The same batch file gives the
same metrics on two calls"). Fast: torch and numpy only, except the one test marked ``omnisafe``.
"""

from __future__ import annotations

import copy
import io
import math
import random
import re
import tokenize
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from configs import registered as R
from metrics import plasticity as P
from metrics.interventions import InjectedHead, intervention_generator, plasticity_injection

REPO = Path(__file__).resolve().parents[1]


class TinyActor(nn.Module):
    """The layout of OmniSafe's GaussianLearningActor (mean MLP + log_std), without OmniSafe."""

    def __init__(self, obs_dim: int = 6, act_dim: int = 2, hidden: tuple[int, ...] = (8, 8)) -> None:
        super().__init__()
        sizes = [obs_dim, *hidden, act_dim]
        layers: list[nn.Module] = []
        for j in range(len(sizes) - 1):
            layers += [nn.Linear(sizes[j], sizes[j + 1]), nn.Tanh() if j < len(sizes) - 2 else nn.Identity()]
        self.mean = nn.Sequential(*layers)
        self.log_std = nn.Parameter(torch.zeros(act_dim))


class TinyCritic(nn.Module):
    """The layout of OmniSafe's VCritic: an MLP registered as ``critic_0``."""

    def __init__(self, obs_dim: int = 6, hidden: tuple[int, ...] = (8, 8)) -> None:
        super().__init__()
        sizes = [obs_dim, *hidden, 1]
        layers: list[nn.Module] = []
        for j in range(len(sizes) - 1):
            layers += [nn.Linear(sizes[j], sizes[j + 1]), nn.Tanh() if j < len(sizes) - 2 else nn.Identity()]
        self.add_module("critic_0", nn.Sequential(*layers))


def _rng_states():
    return torch.get_rng_state(), np.random.get_state(), random.getstate()


def _assert_rng_unchanged(before) -> None:
    after = _rng_states()
    assert torch.equal(before[0], after[0])
    assert before[1][0] == after[1][0] and np.array_equal(before[1][1], after[1][1]) and before[1][2:] == after[1][2:]
    assert before[2] == after[2]


# ---------------------------------------------------------------------------------------------
# Equation (6)
# ---------------------------------------------------------------------------------------------


def test_dormant_scores_hand_computed() -> None:
    h = torch.tensor([[1.0, 0.0, -2.0], [3.0, 0.0, 2.0]])  # mean |h| per unit: 2, 0, 2; layer mean 4/3
    scores = P.dormant_scores(h)
    assert scores.dtype == torch.float64
    assert torch.allclose(scores, torch.tensor([1.5, 0.0, 1.5], dtype=torch.float64), rtol=0, atol=1e-15)
    assert P.dormant_fraction([h]) == pytest.approx(1 / 3)


def test_dormant_threshold_is_inclusive() -> None:
    h = torch.tensor([[1.0, 0.0, 2.0]])  # scores exactly 1, 0, 2
    assert P.dormant_fraction([h], tau=1.0) == pytest.approx(2 / 3)  # "at most tau"
    assert P.dormant_fraction([h], tau=0.999) == pytest.approx(1 / 3)


def test_dormant_fraction_pools_units_and_scores_each_layer_against_its_own_mean() -> None:
    a = torch.tensor([[1.0, 0.0]])  # one of two units dormant
    b = torch.tensor([[1.0, 1.0, 1.0, 1.0]])  # none dormant
    assert P.dormant_fraction([a, b]) == pytest.approx(1 / 6)  # pooled: 1 of 6, not the mean of 1/2 and 0
    big, small = torch.tensor([[10.0, 10.0]]), torch.tensor([[0.01, 0.01]])
    assert P.dormant_fraction([big, small]) == 0.0  # small units are active relative to their own layer


def test_dormant_scale_invariance_all_zero_layer_and_nan() -> None:
    torch.manual_seed(1)
    h = torch.randn(50, 7)
    assert torch.equal(P.dormant_scores(h), P.dormant_scores(4.0 * h))  # a power of two scales float32 exactly
    assert torch.allclose(P.dormant_scores(h), P.dormant_scores(3.5 * h), rtol=1e-6, atol=0)
    zero = torch.zeros(10, 4)
    assert torch.equal(P.dormant_scores(zero), torch.zeros(4, dtype=torch.float64))
    assert P.dormant_fraction([zero]) == 1.0  # Table 9.1 (Q-plasticity-definitions): all units dormant
    h_nan = h.clone()
    h_nan[3, 2] = float("nan")
    assert torch.isnan(P.dormant_scores(h_nan)).all()
    assert math.isnan(P.dormant_fraction([h, h_nan]))
    h_inf = h.clone()
    h_inf[0, 0] = float("inf")
    assert math.isnan(P.dormant_fraction([h_inf]))


def test_dormant_known_number_of_zero_units_in_an_actor() -> None:
    """First Tasks section 9, test 1: k hidden units output zero for every input -> fraction k / 128."""
    torch.manual_seed(0)
    actor = TinyActor(obs_dim=60, act_dim=2, hidden=(64, 64))
    dead = [0, 5, 17, 31, 40, 63]
    with torch.no_grad():
        actor.mean[2].weight[dead] = 0.0
        actor.mean[2].bias[dead] = 0.0
    x = torch.randn(512, 60, generator=torch.Generator().manual_seed(3))
    acts = P.hidden_activations(actor.mean, x)
    assert set(acts) == {"1", "3"}  # the output layer (mean[4]) is never hidden
    assert P.dormant_fraction(acts.values()) == len(dead) / 128
    row = P.measure({"actor": actor}, x.numpy(), step=0)
    assert row["dormant"] == len(dead) / 128
    assert row["dormant_trainable"] == len(dead) / 64  # H3 (c) on mean[3] only


def test_dormant_bad_shapes_and_empty() -> None:
    with pytest.raises(ValueError):
        P.dormant_scores(torch.zeros(3))
    with pytest.raises(ValueError):
        P.dormant_fraction([])


# ---------------------------------------------------------------------------------------------
# Equation (7)
# ---------------------------------------------------------------------------------------------


def test_effective_rank_ten_equal_singular_values() -> None:
    phi = torch.zeros(20, 64)
    phi[torch.arange(10), torch.arange(10)] = 1.0
    assert P.effective_rank(phi) == 10  # First Tasks section 9, test 2


def test_effective_rank_hundred_equal_values_is_99_at_delta_001() -> None:
    assert P.effective_rank(torch.eye(100)) == 99  # 99/100 >= 1 - 0.01


def test_effective_rank_hand_computed_and_delta() -> None:
    phi = torch.diag(torch.tensor([3.0, 1.0, 0.0]))  # shares 0.75, 1.0, 1.0
    assert P.effective_rank(phi) == 2
    assert P.effective_rank(phi, delta=0.3) == 1  # 0.75 >= 0.7
    assert P.effective_rank(phi, delta=0.25) == 1  # inclusive: 0.75 >= 0.75
    assert isinstance(P.effective_rank(phi), int)


def test_effective_rank_is_uncentred() -> None:
    assert P.effective_rank(torch.ones(5, 3)) == 1  # centring would give an all-zero matrix (rank 0)


def test_effective_rank_zero_and_non_finite(monkeypatch) -> None:
    assert P.effective_rank(torch.zeros(8, 4)) == 0
    phi = torch.randn(8, 4)
    phi[1, 1] = float("nan")

    def boom(*args, **kwargs):
        raise AssertionError("svdvals must not be called on non-finite input")

    monkeypatch.setattr(torch.linalg, "svdvals", boom)
    assert math.isnan(P.effective_rank(phi))
    phi[1, 1] = float("inf")
    assert math.isnan(P.effective_rank(phi))


def test_effective_rank_bad_shape() -> None:
    with pytest.raises(ValueError):
        P.effective_rank(torch.zeros(4))
    with pytest.raises(ValueError):
        P.effective_rank(torch.zeros(0, 3))


# ---------------------------------------------------------------------------------------------
# Parameter norm
# ---------------------------------------------------------------------------------------------


def test_parameter_norm_by_hand_includes_log_std() -> None:
    actor = TinyActor(obs_dim=2, act_dim=2, hidden=(2,))
    with torch.no_grad():
        for p in actor.parameters():
            p.zero_()
        actor.mean[0].weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        actor.mean[0].bias.copy_(torch.tensor([0.0, 1.0]))
        actor.log_std.copy_(torch.tensor([2.0, 0.0]))
    assert P.parameter_norm(actor) == math.sqrt(1 + 4 + 9 + 16 + 1 + 4)  # First Tasks section 9, test 3
    actor.log_std.requires_grad_(False)
    assert P.parameter_norm(actor) == math.sqrt(31)  # trainable only
    assert P.parameter_norm(actor, trainable_only=False) == math.sqrt(35)
    assert P.parameter_norm([]) == 0.0


# ---------------------------------------------------------------------------------------------
# Frozen normaliser
# ---------------------------------------------------------------------------------------------


class _Norm:
    def __init__(self, count: int, dim: int = 3) -> None:
        self._count = torch.tensor(count)
        self._mean = torch.tensor([1.0, -2.0, 0.5])[:dim]
        self._std = torch.tensor([2.0, 0.5, 1.0])[:dim]
        self._clip = 5.0 * torch.ones(dim)

    def normalize(self, data):  # pragma: no cover - must never be called
        raise AssertionError("normalise_frozen must not call Normalizer.normalize")


def test_normalise_frozen_identity_while_count_at_most_one() -> None:
    x = torch.randn(4, 3)
    assert P.normalise_frozen(x, _Norm(0)) is x
    assert P.normalise_frozen(x, _Norm(1)) is x
    assert P.normalise_frozen(x, None) is x


def test_normalise_frozen_formula_clip_and_state_dict_form() -> None:
    norm = _Norm(7)
    x = torch.tensor([[1.0, -2.0, 0.5], [3.0, 0.0, 100.0], [-100.0, -2.5, 0.0]])
    expected = torch.clamp((x - norm._mean) / norm._std, -norm._clip, norm._clip)
    before = {k: v.clone() for k, v in vars(norm).items()}
    out = P.normalise_frozen(x, norm)
    assert torch.equal(out, expected)
    assert out[1, 2] == 5.0 and out[2, 0] == -5.0
    assert all(torch.equal(before[k], v) for k, v in vars(norm).items())  # nothing updated
    state = {k: v.clone() for k, v in vars(norm).items()}
    assert torch.equal(P.normalise_frozen(x, state), expected)  # epoch-k.pt['obs_normalizer'] form
    for count in (0, 1, 7):  # a shape mismatch is refused at every checkpoint, the untrained one included
        with pytest.raises(ValueError, match="does not match"):
            P.normalise_frozen(torch.zeros(2, 4), _Norm(count))


@pytest.mark.omnisafe
def test_normalise_frozen_equals_omnisafe_without_updating() -> None:
    from omnisafe.common.normalizer import Normalizer

    torch.manual_seed(2)
    x = torch.randn(64, 5) * 3 + 1
    fresh = Normalizer((5,), clip=5)
    assert P.normalise_frozen(x, fresh) is x  # count 0 (epoch-0.pt)
    single = Normalizer((5,), clip=5)
    single.normalize(torch.randn(5))  # count 1: OmniSafe returns the data unchanged
    assert int(single._count) == 1 and P.normalise_frozen(x, single) is x
    norm = Normalizer((5,), clip=5)
    for _ in range(4):
        norm.normalize(torch.randn(16, 5) * 2)
    before = copy.deepcopy(norm.state_dict())
    ours = P.normalise_frozen(x, norm)
    assert all(torch.equal(before[k], v) for k, v in norm.state_dict().items())  # not updated
    reference = copy.deepcopy(norm)
    reference._push = lambda data: None  # OmniSafe's own normalize, without its update
    assert torch.equal(ours, reference.normalize(x))
    assert torch.equal(P.normalise_frozen(x, norm.state_dict()), ours)


# ---------------------------------------------------------------------------------------------
# Hidden activations and measure
# ---------------------------------------------------------------------------------------------


def test_hidden_activations_equal_manual_tanh_and_leave_no_trace() -> None:
    torch.manual_seed(4)
    actor = TinyActor()
    x = torch.randn(32, 6)
    modes = {name: m.training for name, m in actor.named_modules()}
    before = _rng_states()
    acts = P.hidden_activations(actor.mean, x)
    _assert_rng_unchanged(before)
    h1 = torch.tanh(actor.mean[0](x))
    h3 = torch.tanh(actor.mean[2](h1))
    assert torch.equal(acts["1"], h1.detach()) and torch.equal(acts["3"], h3.detach())
    assert all(not m._forward_hooks for m in actor.modules())
    assert modes == {name: m.training for name, m in actor.named_modules()}
    with pytest.raises(ValueError):
        P.hidden_activations(actor.mean, x, ["9"])
    assert all(not m._forward_hooks for m in actor.modules())


def test_hooks_removed_even_when_forward_fails() -> None:
    actor = TinyActor()
    with pytest.raises(RuntimeError):
        P.hidden_activations(actor.mean, torch.zeros(3, 7))  # wrong input width
    assert all(not m._forward_hooks for m in actor.modules())


def test_measure_columns_repeatability_and_neutrality() -> None:
    torch.manual_seed(5)
    actor, rc, cc = TinyActor(), TinyCritic(), TinyCritic()
    batch = np.random.default_rng(0).normal(size=(40, 6)).astype(np.float32)
    parts = P.NetworkParts(actor, rc, cc, None)
    state = copy.deepcopy([m.state_dict() for m in (actor, rc, cc)])
    before = _rng_states()
    first = P.measure(parts, batch, step=4000)
    second = P.measure(parts, batch, step=4000)
    _assert_rng_unchanged(before)
    assert tuple(first) == P.COLUMNS and first == second  # First Tasks section 9, test 4
    assert P.COLUMNS[:4] == ("step", "dormant", "rank", "norm")  # contract 2: what the ledger writer reads
    for m, s in zip((actor, rc, cc), state):
        assert all(torch.equal(v, m.state_dict()[k]) for k, v in s.items())
    assert first["step"] == 4000
    assert first["norm"] == first["norm_all"] == P.parameter_norm(actor)
    assert first["norm_reward_critic"] == P.parameter_norm(rc)
    assert isinstance(first["rank"], int) and 1 <= first["rank"] <= 8
    x = torch.from_numpy(batch)
    acts = P.hidden_activations(rc.critic_0, x)
    assert first["dormant_reward_critic"] == P.dormant_fraction(acts.values())
    assert first["rank_reward_critic"] == P.effective_rank(acts["3"])


def test_measure_applies_the_normaliser_and_handles_missing_critics() -> None:
    torch.manual_seed(6)
    actor = TinyActor(obs_dim=3)
    batch = np.random.default_rng(1).normal(size=(30, 3)).astype(np.float32)
    norm = _Norm(9)
    row = P.measure({"actor": actor, "normalizer": norm}, batch, step=0)
    manual = P.measure({"actor": actor}, P.normalise_frozen(torch.from_numpy(batch), norm), step=0)
    assert row == manual
    assert all(math.isnan(row[c]) for c in P.COLUMNS if "critic" in c)
    with pytest.raises(ValueError):
        P.measure({"actor": actor}, batch, step=-1)
    with pytest.raises(ValueError):
        P.measure({"actor": actor}, batch, step=True)


def test_measure_injected_actor_layers() -> None:
    torch.manual_seed(7)
    actor = TinyActor(obs_dim=6, hidden=(8, 8))
    opt = torch.optim.Adam(actor.parameters())
    batch = torch.randn(64, 6)
    before = P.measure({"actor": actor}, batch, step=0)
    plasticity_injection(actor, opt, intervention_generator(3), batch=batch)
    assert isinstance(actor.mean, InjectedHead)
    layers = P.actor_layers(actor.mean)
    assert layers.hidden == ("trunk.1", "frozen.3", "new.3", "new_frozen.3")
    assert layers.penultimate == ("frozen.3", "new.3", "new_frozen.3")
    assert layers.trainable == ("new.3",)
    row = P.measure({"actor": actor}, batch, step=2000)
    acts = P.hidden_activations(actor.mean, batch)
    assert row["dormant"] == P.dormant_fraction([acts[n] for n in layers.hidden])
    assert row["rank"] == P.effective_rank(torch.cat([acts[n] for n in layers.penultimate], dim=1))
    assert row["rank_trainable"] == P.effective_rank(acts["new.3"])
    trainable = [p for p in actor.parameters() if p.requires_grad]  # log_std, trunk, new head
    assert len(trainable) == 7 and row["norm"] == P.parameter_norm(trainable) != before["norm"]
    assert row["norm_all"] == P.parameter_norm(actor, trainable_only=False) > row["norm"]


def test_measure_injected_output_layer_only_has_no_trainable_hidden_layer() -> None:
    torch.manual_seed(8)
    actor = TinyActor()
    plasticity_injection(actor, torch.optim.Adam(actor.parameters()), intervention_generator(0), depth=1)
    layers = P.actor_layers(actor.mean)
    assert layers.hidden == ("trunk.1", "trunk.3") and layers.penultimate == ("trunk.3",) and layers.trainable == ()
    row = P.measure({"actor": actor}, torch.randn(16, 6), step=0)
    assert math.isnan(row["dormant_trainable"]) and math.isnan(row["rank_trainable"])


def test_rows_format_and_parse_exactly() -> None:
    row = {c: 0.1 * i for i, c in enumerate(P.COLUMNS)}
    row.update(step=2000, rank=47, rank_trainable=float("nan"), rank_reward_critic=12, rank_cost_critic=3.0)
    text = P.format_row(row)
    assert text[0] == "2000" and text[2] == "47" and text[P.COLUMNS.index("rank_trainable")] == "nan"
    assert text[P.COLUMNS.index("rank_cost_critic")] == "3"
    parsed = P.parse_row(dict(zip(P.COLUMNS, text)))
    for c in P.COLUMNS:
        assert (math.isnan(parsed[c]) and math.isnan(row[c])) or parsed[c] == row[c]
    nan = P.nan_row(6000)
    assert nan["step"] == 6000 and all(math.isnan(nan[c]) for c in P.COLUMNS[1:])
    with pytest.raises(ValueError):
        P.format_row({"step": 1})


def test_no_registered_number_is_retyped_in_metrics_code() -> None:
    """HANDOVER.md section 10, "Numbers": registered numbers come from configs.registered (number tokens, not prose)."""
    forbidden = {R.DORMANT_THRESHOLD, R.EFFECTIVE_RANK_DELTA, R.FIXED_BATCH_STATES, R.CHECKPOINT_INTERVAL_STEPS,
                 R.MANIPULATION_CHECK_STEPS_AFTER_ONSET, R.OVERSHOOT_WINDOW_STEPS, R.OVERSHOOT_FINAL_FRACTION,
                 R.SETTLING_BAND, R.COST_LIMIT, R.STEPS_PER_EPOCH, R.TOTAL_STEPS, R.EPISODE_LENGTH}
    found = []
    for path in sorted((REPO / "metrics").glob("*.py")):
        for token in tokenize.generate_tokens(io.StringIO(path.read_text(encoding="utf-8")).readline):
            if token.type == tokenize.NUMBER and "j" not in token.string.lower():
                value = float(eval(token.string))  # noqa: S307 - a numeric literal
                if value in forbidden:
                    found.append(f"{path.name}:{token.start[0]} {token.string}")
    assert not found, found


def test_the_details_the_docstring_quotes_from_a_keys_text_are_in_that_text() -> None:
    """Every phrase the module docstring quotes from a key's text in configs/registered.py appears verbatim there.

    The zero and non-finite handling, the critics' metrics, the H3 (c) trainable layers and the injected
    layout are answered in Table 9.1 (Q-plasticity-definitions and Q-reset-injection), and the docstring says so.
    The quoted phrases appear verbatim in the key's text in configs/registered.py, which states the answer
    (docs/DECISIONS.md, 2026-10-02).
    """
    doc = " ".join(P.__doc__.split())
    assert "no pending text states" not in doc.lower() and "PROPOSAL" not in doc
    assert "no pending text states" not in " ".join(P.actor_layers.__doc__.split()).lower()
    assert "Table 9.1" in " ".join(P.actor_layers.__doc__.split())
    not_pending = {"with the two critics logged separately", "last two layers"}  # Table 2.3 and box H3 (c)
    for key, nxt in (("Q-plasticity-definitions", "* Q-reset-injection"), ("Q-reset-injection", "Only torch")):
        section = doc[doc.index(f"* {key}"):doc.index(nxt)]
        quoted = [q for q in re.findall(r'"([^"]+)"', section) if q not in not_pending]
        assert len(quoted) >= 3, key
        text = " ".join(R.PENDING[key].split())
        for phrase in quoted:
            assert phrase in text, f"{key}: {phrase!r}"


def test_the_function_docstrings_cite_the_answer_of_q_plasticity_definitions() -> None:
    """dormant_scores and effective_rank cite Table 9.1 (Q-plasticity-definitions), whose text states their zero and
    non-finite handling (in the words the PENDING text and the answer share)."""
    for fn in (P.dormant_scores, P.effective_rank):
        doc = " ".join(fn.__doc__.split())
        assert "PROPOSAL" not in doc, fn.__name__
        assert "Table 9.1, Q-plasticity-definitions" in doc, fn.__name__
    text = " ".join(R.PENDING["Q-plasticity-definitions"].split())
    for phrase in ("an all-zero layer counts as all dormant", "an all-zero matrix has rank 0",
                   "non-finite input gives NaN"):
        assert phrase in text

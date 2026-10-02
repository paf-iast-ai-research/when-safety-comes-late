"""Controller quantities and recovery time (Role 3): Table 2.4 overshoot and settling, Table 2.1 recovery.

Pure functions at epoch resolution, gated by the result gate Q-controller-quantities (whose answer in
Table 9.1 they implement). The key is answered; the gate test pins it open (``pin_open``, tests/conftest.py).
"""

from __future__ import annotations

import math
import re

import numpy as np
import pytest

import metrics.controller
from configs import registered as R
from metrics.controller import controller_quantities
from metrics.recovery import recovery_steps
from pilot.errors import PendingQuestionError, RunRefused, allow_pending

E = R.OVERSHOOT_WINDOW_STEPS // 4  # an epoch length whose overshoot window spans four epochs
HELD = R.LAGRANGE_MULTIPLIER_INIT


def _cq(values: list[float], onset_epoch: int, steps: int = E) -> dict[str, float | int | None]:
    n = len(values)
    return controller_quantities(list(range(n)), values, onset_step=onset_epoch * steps, total_steps=n * steps,
                                 steps_per_epoch=steps)


# ---------------------------------------------------------------------------------------------
# Controller quantities
# ---------------------------------------------------------------------------------------------


def test_hand_computed_peak_final_and_settling() -> None:
    # epochs 0-3 before onset (held), window = epochs 4-7 (ending in (onset, onset + 2,000,000]), 20 epochs
    values = [HELD] * 4 + [0.5, 2.0, 3.0, 1.5, 9.0, 1.2, 1.08] + [1.05] * 7 + [1.0, 1.0]
    out = _cq(values, onset_epoch=4)
    assert out["lambda_peak"] == 3.0  # 9.0 at epoch 8 ends after onset + 2,000,000
    assert out["lambda_final"] == 1.0  # mean of the last ceil(0.10 x 20) = 2 epochs
    assert out["settling_steps"] == (10 + 1) * E - 4 * E  # from epoch 10 (1.08) every value lies in [0.9, 1.1]
    values[10] = 1.1  # |1.1 - 1.0| is 0.10000000000000009 in binary: outside the band
    assert _cq(values, onset_epoch=4)["settling_steps"] == (11 + 1) * E - 4 * E
    assert set(out) == {"lambda_peak", "lambda_final", "settling_steps"}


def test_settling_band_is_inclusive_and_settling_at_once_is_one_epoch() -> None:
    values = [HELD] * 2 + [9.0, 11.0] + [10.0] * 6  # final 10.0, band [9.0, 11.0] (exact in binary)
    assert _cq(values, onset_epoch=2)["settling_steps"] == E  # inside from the first constrained epoch
    values[3] = math.nextafter(11.0, 12.0)  # epoch 3 just outside: inside from epoch 4, which ends at 5E
    assert _cq(values, onset_epoch=2)["settling_steps"] == 5 * E - 2 * E


def test_never_settles_and_zero_final_value() -> None:
    oscillating = [HELD] * 2 + [1.0] * 16 + [1.0, 2.0]  # final 1.5; the last value 2.0 is outside +/-0.15
    assert _cq(oscillating, onset_epoch=2)["settling_steps"] is None
    zero = [HELD] * 2 + [0.4, 0.2] + [0.0] * 16
    out = _cq(zero, onset_epoch=2)
    # Table 9.1: "If the final value is 0, this means the multiplier stays exactly 0 from e* on": the first
    # all-zero epoch is epoch 4, which ends at 5E, so settling = 5E - 2E
    assert out == {"lambda_peak": 0.4, "lambda_final": 0.0, "settling_steps": 3 * E}


def test_a_zero_final_value_settles_at_the_first_epoch_of_the_final_zero_run() -> None:
    """With a final value of 0 the band is {0}: a value that is not exactly 0 lies outside it, however small."""
    late = [HELD] * 2 + [0.5] * 16 + [0.0, 0.0]  # 0 only in the last 10 percent (ceil(0.10 x 20) = 2 epochs)
    assert _cq(late, onset_epoch=2) == {"lambda_peak": 0.5, "lambda_final": 0.0, "settling_steps": (18 + 1) * E - 2 * E}
    interrupted = [HELD] * 2 + [0.0] * 5 + [1e-12] + [0.0] * 12  # back to 0 at epoch 8, which ends at 9E
    assert _cq(interrupted, onset_epoch=2)["settling_steps"] == 9 * E - 2 * E
    assert _cq([0.0] * 20, onset_epoch=0)["settling_steps"] == E  # 0 throughout: settled in the first epoch


def test_no_multiplier_gives_none() -> None:
    for values in (None, []):
        assert controller_quantities([], values, onset_step=0, total_steps=10 * E, steps_per_epoch=E) == {
            "lambda_peak": None, "lambda_final": None, "settling_steps": None}


def test_onset_zero() -> None:
    values = [0.2, 0.8, 0.6, 0.5, 0.9] + [0.5] * 15
    out = _cq(values, onset_epoch=0)
    assert out["lambda_peak"] == 0.8  # window = epochs 0-3: the 0.9 of epoch 4 ends at 5E > 2,000,000
    assert out["lambda_final"] == 0.5
    assert out["settling_steps"] == 6 * E  # from epoch 5 (ends at 6E) every value is 0.5


def test_window_is_cut_at_the_end_of_training() -> None:
    values = [HELD] * 8 + [0.1, 0.7]  # onset after epoch 7 (at 8E): the window has two epochs left (8 and 9)
    out = _cq(values, onset_epoch=8)
    assert out["lambda_peak"] == 0.7 and out["lambda_final"] == 0.7  # ceil(0.10 x 10) = 1 epoch
    long_epochs = _cq([HELD, 0.5, 0.5], onset_epoch=1, steps=R.OVERSHOOT_WINDOW_STEPS + 1)
    assert long_epochs["lambda_peak"] is None and long_epochs["lambda_final"] == 0.5  # no epoch ends in the window


@pytest.mark.parametrize("epochs, count",
                         [(500, 50), (550, 55), (556, 56), (625, 63), (667, 67), (750, 75), (1000, 100)])
def test_final_mean_count_over_the_runs_own_length(epochs: int, count: int) -> None:
    """Table 2.4 "the last 10 percent of training epochs" of the run's own length (T = 500; T/(1-N), T + N*T)."""
    steps = R.STEPS_PER_EPOCH
    onset = epochs // 2
    ones = [0.0] * (epochs - count) + [1.0] * count
    assert _cq(ones, onset_epoch=onset, steps=steps)["lambda_final"] == 1.0  # the count is at most `count`
    fewer = [0.0] * (epochs - count + 1) + [1.0] * (count - 1)
    assert _cq(fewer, onset_epoch=onset, steps=steps)["lambda_final"] == (count - 1) / count  # and exactly it


def test_controller_input_checks() -> None:
    ok = dict(onset_step=2 * E, total_steps=4 * E, steps_per_epoch=E)
    # 2.0 read as 2
    assert controller_quantities([0.0, 1.0, 2.0, 3.0], [HELD, HELD, 1.0, 1.0], **ok)["lambda_final"] == 1.0
    bad_cases = [
        ([0, 1, 3, 2], [1.0] * 4, ok),  # not in order
        ([1, 2, 3, 4], [1.0] * 4, ok),  # not from 0
        ([0, 1, 2.5, 3], [1.0] * 4, ok),  # not integral
        ([0, 1, 2], [1.0] * 3, ok),  # shorter than the run
        ([0, 1, 2, 3], [1.0] * 3, ok),  # lengths differ
        ([0, 1, 2, 3], [1.0, 1.0, float("nan"), 1.0], ok),  # non-finite
        ([0, 1, 2, 3], [1.0, 1.0, -0.5, 1.0], ok),  # negative
        ([0, 1, 2, 3], [1.0] * 4, dict(ok, onset_step=E + 1)),  # onset inside an epoch
        ([0, 1, 2, 3], [1.0] * 4, dict(ok, onset_step=4 * E)),  # onset at the end
        ([0, 1, 2, 3], [1.0] * 4, dict(ok, onset_step=-E)),  # negative onset
        ([0, 1, 2, 3], [1.0] * 4, dict(ok, steps_per_epoch=0)),  # no steps per epoch
        ([0, 1, 2, 3], [1.0] * 4, dict(ok, total_steps=4 * E + 1)),  # total not a whole number of epochs
        ([0, 1, 2, 3], [1.0] * 4, dict(ok, onset_step=True)),  # bool onset
        ([0, 1, 2, 3], [1.0, True, 1.0, 1.0], ok),  # bool value
        ([0, 1, 2, 3], [], ok),  # epochs without multiplier values (a truncated trace)
        ([], [], dict(ok, onset_step=-E)),  # no multiplier, negative onset
        ([], [], dict(ok, steps_per_epoch=0)),  # no multiplier, no steps per epoch
        ([], [], dict(ok, total_steps=4 * E + 1)),  # no multiplier, total not a whole number of epochs
    ]
    for epochs, values, kwargs in bad_cases:
        with pytest.raises(ValueError):
            controller_quantities(epochs, values, **kwargs)


# ---------------------------------------------------------------------------------------------
# The rate clip's share (Q-rate-limit)
# ---------------------------------------------------------------------------------------------


def test_rate_limit_clip_share_counts_the_constrained_epochs_in_which_the_clip_bound() -> None:
    """Q-rate-limit (Table 9.1): "the report gives the share of constrained epochs in which the clip bound": the
    epochs from onset whose Onset/MultiplierProposed (before the clip) is not the logged multiplier."""
    lam = [HELD] * 4 + [0.011, 0.0216, 0.0327, 0.5, 0.5, 0.5]
    proposed = [math.nan] * 4 + [0.036, 0.071, 0.0327, 0.5, 0.52, 0.5]  # NaN before onset (envs.onset)
    kw = dict(onset_step=4 * E, steps_per_epoch=E, total_steps=10 * E)
    assert metrics.controller.rate_limit_clip_share(list(range(10)), lam, proposed, **kw) == 3 / 6
    assert metrics.controller.rate_limit_clip_share(list(range(10)), lam, lam[:4] + lam[4:], **kw) == 0.0
    with allow_pending(False):  # descriptive, no gate of its own: Q-rate-limit gates the rate-limited runs
        assert metrics.controller.rate_limit_clip_share(list(range(10)), lam, [-1.0] * 4 + lam[4:], **kw) == 0.0
    for bad_proposed in ([math.nan] * 5 + lam[5:], [math.nan] * 4 + [-0.1] + lam[5:], proposed[:-1]):
        with pytest.raises(ValueError):  # a constrained epoch's value is logged, finite and >= 0; one per epoch
            metrics.controller.rate_limit_clip_share(list(range(10)), lam, bad_proposed, **kw)
    with pytest.raises(ValueError):  # the trace must cover the run
        metrics.controller.rate_limit_clip_share(list(range(10)), lam, proposed, **dict(kw, total_steps=11 * E))


def test_the_proposed_column_is_the_one_envs_onset_logs() -> None:
    pytest.importorskip("omnisafe")
    from envs import onset

    from pilot import enrichment

    assert enrichment.PROPOSED_MULTIPLIER_COLUMN == onset.PROPOSED_KEY == "Onset/MultiplierProposed"


# ---------------------------------------------------------------------------------------------
# Recovery time
# ---------------------------------------------------------------------------------------------


def test_recovery_hand_computed_and_inclusive() -> None:
    # onset after epoch 2 (at 3E); epochs 0-2 are pre-onset, so epoch 0's 10.0 does not count
    costs = [10.0, 40.0, 60.0, 30.0, 26.0, R.COST_LIMIT, 20.0]
    assert recovery_steps(range(7), costs, onset_step=3 * E, steps_per_epoch=E) == 3 * E  # end of epoch 5
    assert recovery_steps(range(7), costs, onset_step=3 * E, steps_per_epoch=E, total_steps=7 * E) == 3 * E
    assert recovery_steps(range(7), costs, onset_step=3 * E, steps_per_epoch=E, budget=10.0) is None


def test_recovery_never_nan_epochs_and_onset_zero() -> None:
    assert recovery_steps(range(4), [30.0, 40.0, 26.0, 99.0], onset_step=E, steps_per_epoch=E) is None
    nan = float("nan")
    # no episode, no mean
    assert recovery_steps(range(4), [nan, nan, 5.0, 5.0], onset_step=0, steps_per_epoch=E) == 3 * E
    assert recovery_steps(range(3), [0.0, 99.0, 99.0], onset_step=0, steps_per_epoch=E) == E  # first epoch counts
    assert recovery_steps(range(3), [99.0, 20.0, 99.0], onset_step=0, steps_per_epoch=E,
                          budget=np.float32(R.COST_LIMIT)) == 2 * E  # any real budget (numbers.Real)
    long_run = [99.0] * 700 + [1.0] * 50  # a run longer than T (the data control, T + N*T = 750 epochs at N = 0.50)
    steps = R.STEPS_PER_EPOCH
    assert recovery_steps(range(750), long_run, onset_step=250 * steps, steps_per_epoch=steps,
                          total_steps=750 * steps) == (700 + 1) * steps - 250 * steps


def test_recovery_input_checks() -> None:
    with pytest.raises(ValueError):
        recovery_steps(range(3), [1.0, 1.0], onset_step=0, steps_per_epoch=E)
    with pytest.raises(ValueError):
        recovery_steps(range(3), [1.0] * 3, onset_step=3 * E, steps_per_epoch=E)  # onset after the trace
    with pytest.raises(ValueError):
        recovery_steps(range(3), [1.0] * 3, onset_step=0, steps_per_epoch=E, total_steps=4 * E)  # incomplete
    with pytest.raises(ValueError):
        recovery_steps(range(3), [1.0] * 3, onset_step=0, steps_per_epoch=E, budget=float("inf"))
    for budget in (-1.0, True, "25"):
        with pytest.raises(ValueError):
            recovery_steps(range(3), [1.0] * 3, onset_step=0, steps_per_epoch=E, budget=budget)
    for bad in (-0.5, float("inf"), float("-inf")):  # impossible costs: corrupt data, not "no episode"
        with pytest.raises(ValueError):
            recovery_steps(range(3), [30.0, bad, 1.0], onset_step=0, steps_per_epoch=E)
    with pytest.raises(ValueError):
        recovery_steps([0, 2, 1], [1.0] * 3, onset_step=0, steps_per_epoch=E)


# ---------------------------------------------------------------------------------------------
# The result gate
# ---------------------------------------------------------------------------------------------


def test_both_are_refused_while_the_question_is_open(pin_open, monkeypatch) -> None:
    pin_open("Q-controller-quantities")  # the key open, whatever the repository answers
    with allow_pending(False):
        with pytest.raises(PendingQuestionError, match="Q-controller-quantities") as info:
            controller_quantities([0, 1], [1.0, 1.0], onset_step=0, total_steps=2 * E, steps_per_epoch=E)
        assert isinstance(info.value, RunRefused)
        with pytest.raises(PendingQuestionError, match="Q-controller-quantities"):
            recovery_steps([0, 1], [1.0, 1.0], onset_step=0, steps_per_epoch=E)
    monkeypatch.setattr(R, "ANSWERED_QUESTIONS", R.ANSWERED_QUESTIONS | {"Q-controller-quantities"})
    with allow_pending(False):  # answered in Table 9.1: the gate no longer holds
        assert controller_quantities([0, 1], [1.0, 1.0], onset_step=0, total_steps=2 * E,
                                     steps_per_epoch=E)["settling_steps"] == E
        assert recovery_steps([0, 1], [1.0, 1.0], onset_step=0, steps_per_epoch=E) == E


def test_the_docstring_quotes_the_registered_answer_verbatim() -> None:
    """The module docstring quotes Q-controller-quantities' answer of Table 9.1 (up to its marked ellipsis):
    the key is answered (docs/DECISIONS.md, 2026-10-02), and its text in configs/registered.py states the
    question and then the answer, which starts with the quotation."""
    def norm(text: str) -> str:
        return re.sub(r"[\s-]+", " ", text)  # line breaks and list dashes alike

    doc = norm(metrics.controller.__doc__)
    quoted = doc[doc.index('"Controller quantities') + 1:]
    quoted = quoted[:quoted.index(' ..."')]
    assert "If the final value is 0, this means the multiplier stays exactly 0 from e* on." in quoted
    assert not R.is_open("Q-controller-quantities")  # answered (docs/DECISIONS.md, 2026-10-02)
    assert " Answered: " + quoted in norm(R.PENDING["Q-controller-quantities"])

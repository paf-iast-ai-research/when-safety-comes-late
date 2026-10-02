"""Controller quantities of the Lagrange multiplier (Role 3, Metrics and interventions; owner Abdullah).

Implements, at epoch resolution, Table 2.4 (PDF p. 9):

* "Multiplier overshoot": "The maximum multiplier value in the 2,000,000 steps after onset minus its
  final value, where the final value is its mean over the last 10 percent of training epochs."
* "Settling time": "Steps from onset until the multiplier stays within ±10 percent of its final
  value for the rest of training".

and Appendix B, Table B.1: "lambda_peak; lambda_final; settling_steps | Controller quantities (Study
A)" (ledger fields ``lambda_peak``/``lambda_final`` float >= 0 or None, ``settling_steps`` int >= 0 or None). The
overshoot is ``lambda_peak - lambda_final``, derived by the analysis (INFERENCE: Table B.1 stores
the peak and the final value, Table 2.4 defines their difference).

The epoch-resolution reading is the answer of Q-controller-quantities in Table 9.1 (a result gate,
``configs/registered.py``): "Controller quantities are computed at epoch resolution from
Metrics/LagrangeMultiplier, the value after each epoch's update. Peak: the maximum over the epochs
that end in (onset, onset + 2,000,000]. Final value: the mean over the last ceil(0.10 x epochs)
epochs of the run's own length. Overshoot = peak - final. Settling time: the steps from onset to the
end of the first post-onset epoch e* such that the value of e* and of every later epoch lies within
+/-10 percent of the final value (inclusive). If the final value is 0, this means the multiplier
stays exactly 0 from e* on. It is none only if the last epoch's value lies outside the band. ..."
(quoted up to there; the answer goes on with recovery time, ``metrics.recovery``, and the analysis's
censoring of a run that never recovers and of a treated-untreated contrast, implemented in
``analysis.study_a._recovery`` and ``_recovery_getter``). ``controller_quantities`` calls
``pilot.errors.require_answered`` first, a gate that holds only while the key is open.

For the rate-limited arm, ``rate_limit_clip_share`` gives what Q-rate-limit's answer in Table 9.1 asks
the report to give: "The proposed (pre-clip) value is logged each epoch as Onset/MultiplierProposed,
and the report gives the share of constrained epochs in which the clip bound." It is descriptive (the
rate-limited arm is the secondary H4 check) and needs no gate of its own: Q-rate-limit gates the
rate-limited runs themselves.

The input is the multiplier trace of a completed run: ``Metrics/LagrangeMultiplier`` of every epoch
from ``progress.csv`` with its ``Train/Epoch`` index (0-based). The value logged at epoch e is the
multiplier after that epoch's update, i.e. after (e + 1) x steps_per_epoch steps (OmniSafe
``ppo_lag.py:76-80``: the update happens in ``_update`` and is then stored); before onset the
plug-in logs the held initial value (contract 3). Registered constants: ``R.OVERSHOOT_WINDOW_STEPS``,
``R.OVERSHOOT_FINAL_FRACTION``, ``R.SETTLING_BAND``.
"""

from __future__ import annotations

import math
import numbers
from fractions import Fraction
from typing import Any, Sequence

from configs import registered as R
from pilot.errors import require_answered


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def check_epoch_trace(epochs: Sequence[Any], values: Sequence[Any], *, onset_step: Any, steps_per_epoch: Any,
                      total_steps: Any = None, name: str = "values") -> tuple[int, int, list[float]]:
    """Validate an epoch trace; returns (onset epoch, steps per epoch, values as floats).

    ``epochs`` must be 0, 1, 2, ... in order (integral numbers, so ``2.0`` read from a CSV is 2),
    ``values`` as long; ``onset_step`` a whole number of epochs, at least 0 and inside the trace.
    With ``total_steps`` the trace must cover the run exactly (total_steps / steps_per_epoch epochs).
    ``name`` labels the values in error messages. Raises ``ValueError`` for any violation.
    """
    steps = _positive_int("steps_per_epoch", steps_per_epoch)
    if isinstance(onset_step, bool) or not isinstance(onset_step, numbers.Integral) or int(onset_step) < 0:
        raise ValueError(f"onset_step must be a non-negative integer, got {onset_step!r}")
    onset = int(onset_step)
    if onset % steps:
        raise ValueError(f"onset_step {onset} is not a whole number of epochs of {steps} steps")
    indices = []
    for e in epochs:
        if isinstance(e, bool) or not isinstance(e, numbers.Real) or not math.isfinite(float(e)) or float(e) != int(e):
            raise ValueError(f"epoch indices must be integers, got {e!r}")
        indices.append(int(e))
    if indices != list(range(len(indices))):
        raise ValueError("epoch indices must be 0, 1, 2, ... in order (the complete trace of the run)")
    if len(values) != len(indices):
        raise ValueError(f"{len(indices)} epochs but {len(values)} {name}")
    if total_steps is not None:
        total = _positive_int("total_steps", total_steps)
        if total % steps:
            raise ValueError(f"total_steps {total} is not a whole number of epochs of {steps} steps")
        if len(indices) != total // steps:
            raise ValueError(f"the trace has {len(indices)} epochs; a run of {total} steps has {total // steps}")
        if onset >= total:
            raise ValueError(f"onset_step {onset} is not before the end of training ({total})")
    if onset // steps >= len(indices):
        raise ValueError(f"onset_step {onset} is after the last epoch of the trace")
    as_float = []
    for v in values:
        if isinstance(v, bool) or not isinstance(v, numbers.Real):
            raise ValueError(f"{name} must be real numbers, got {v!r}")
        as_float.append(float(v))
    return onset // steps, steps, as_float


def controller_quantities(epochs: Sequence[Any], values: Sequence[Any] | None, *, onset_step: int, total_steps: int,
                          steps_per_epoch: int) -> dict[str, float | int | None]:
    """``{'lambda_peak', 'lambda_final', 'settling_steps'}`` of one run's multiplier trace.

    Table 9.1 (Q-controller-quantities).

    * ``lambda_peak``: the maximum over the epochs e with onset < (e + 1) x E <= onset +
      ``R.OVERSHOOT_WINDOW_STEPS`` (E = ``steps_per_epoch``); the window is cut at the end of
      training; None if no epoch ends inside the window (an epoch longer than
      ``R.OVERSHOOT_WINDOW_STEPS``).
    * ``lambda_final``: ``math.fsum`` mean over the last ceil(``R.OVERSHOOT_FINAL_FRACTION`` x
      epochs) epochs, the count computed exactly (``Fraction``), over the run's own length (a
      constrained-steps-matched arm or the data control trains longer than T).
    * ``settling_steps``: (e* + 1) x E - onset for the smallest post-onset epoch e* (e* >= onset / E)
      such that the value of e* and of every later epoch lies within ``R.SETTLING_BAND`` x
      lambda_final of lambda_final (inclusive); None only if the last value lies outside the band
      (never settles). For every lambda_final >= 0: with lambda_final 0 the band is {0}, so e* is the
      first epoch from which the multiplier stays exactly 0 (OmniSafe clamps it at 0, so a final mean
      of 0 means every value of the last 10 percent is exactly 0).

    A run without a single multiplier (``values`` None or empty: unconstrained PPO; Study B, whose
    multipliers are per level) gives None for all three, as do the ledger's Study B rows; an empty
    ``values`` needs empty ``epochs`` and valid step arguments (a truncated trace is not "no multiplier"). Non-finite
    or negative values raise ``ValueError``: a completed run's multiplier is finite (Part 5.6
    excludes the others) and OmniSafe projects it to >= 0.
    """
    require_answered("Q-controller-quantities",
                     what="the controller quantities lambda_peak, lambda_final, settling_steps (Table 2.4)")
    if values is not None and len(values) == 0:
        if len(epochs):
            raise ValueError(f"{len(epochs)} epochs but no multiplier values")
        steps = _positive_int("steps_per_epoch", steps_per_epoch)
        total = _positive_int("total_steps", total_steps)
        if (isinstance(onset_step, bool) or not isinstance(onset_step, numbers.Integral)
                or not 0 <= int(onset_step) < total):
            raise ValueError(f"onset_step must be an integer in [0, total_steps), got {onset_step!r}")
        if total % steps or int(onset_step) % steps:
            raise ValueError(f"onset_step and total_steps must be whole numbers of epochs of {steps} steps")
    if values is None or len(values) == 0:
        return {"lambda_peak": None, "lambda_final": None, "settling_steps": None}
    start, steps, trace = check_epoch_trace(epochs, values, onset_step=onset_step, steps_per_epoch=steps_per_epoch,
                                            total_steps=total_steps, name="multiplier values")
    bad = [v for v in trace if not math.isfinite(v) or v < 0.0]
    if bad:
        raise ValueError(f"the multiplier trace has non-finite or negative values {bad[:5]}")
    n = len(trace)
    onset = int(onset_step)
    window = [e for e in range(start, n) if onset < (e + 1) * steps <= onset + R.OVERSHOOT_WINDOW_STEPS]
    lambda_peak = max(trace[e] for e in window) if window else None
    count = math.ceil(Fraction(str(R.OVERSHOOT_FINAL_FRACTION)) * n)  # exact: no float product (Fraction)
    lambda_final = math.fsum(trace[n - count:]) / count
    settling: int | None = None
    band = R.SETTLING_BAND * lambda_final  # 0 for a final value of 0: only an exact 0 lies inside
    first_inside = n
    while first_inside > start and abs(trace[first_inside - 1] - lambda_final) <= band:
        first_inside -= 1
    if first_inside < n:  # the last value is inside: it stays inside from epoch first_inside on
        settling = (first_inside + 1) * steps - onset
    return {"lambda_peak": lambda_peak, "lambda_final": lambda_final, "settling_steps": settling}


def rate_limit_clip_share(epochs: Sequence[Any], multipliers: Sequence[Any], proposed: Sequence[Any], *,
                          onset_step: int, steps_per_epoch: int, total_steps: int | None = None) -> float:
    """The share of constrained epochs in which the rate clip bound (Q-rate-limit, Table 9.1).

    ``multipliers``: ``Metrics/LagrangeMultiplier`` of every epoch, the value after the epoch's
    (clipped) update; ``proposed``: ``Onset/MultiplierProposed``, the value OmniSafe's update proposed
    before the clip (``envs.onset.RateLimitedLagrange``; NaN before onset). The constrained epochs are
    e >= onset / E, the epochs whose update ran (``envs.onset``). The clip bound in epoch e when the
    two values differ: the clipped value is written into the same float32 parameter that held the
    proposed one, so an unclipped epoch logs the same number twice and the comparison is exact.
    ``ValueError`` for a trace that does not cover the run (with ``total_steps``) or a non-finite or
    negative value in a constrained epoch (both are logged there, finite and >= 0).
    """
    start, _, values = check_epoch_trace(epochs, multipliers, onset_step=onset_step, steps_per_epoch=steps_per_epoch,
                                         total_steps=total_steps, name="multiplier values")
    _, _, before = check_epoch_trace(epochs, proposed, onset_step=onset_step, steps_per_epoch=steps_per_epoch,
                                     total_steps=total_steps, name="proposed multiplier values")
    constrained = range(start, len(values))
    bad = [v for e in constrained for v in (values[e], before[e]) if not math.isfinite(v) or v < 0.0]
    if bad:
        raise ValueError(f"the constrained epochs' multiplier or proposed values are non-finite or negative: {bad[:5]}")
    return sum(1 for e in constrained if before[e] != values[e]) / len(constrained)

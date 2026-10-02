"""Recovery time after onset (Role 3, Metrics and interventions; owner Abdullah).

Implements Table 2.1 "Recovery time" (PDF p. 7): "Training steps from onset until the
training-batch mean episodic cost first falls to or below d. Secondary outcome, late arms only."
(d = ``R.COST_LIMIT``, Table 2.1 "Cost budget d"; Part 5.2 lists recovery time among the secondary
outcomes.) The analysis uses it for the late arms only; the function also answers for onset 0.

At epoch resolution this is the answer of the result gate Q-controller-quantities in Table 9.1
(``configs/registered.py``, its first part quoted in ``metrics.controller``): "Recovery time: the steps
from onset to the end of the first constrained epoch whose Metrics/BatchEpCost is at most 25; none if
that never happens."

The input is ``Metrics/BatchEpCost`` of every epoch (the mean episodic cost of the episodes that
finished in that epoch's batch, logged by ``pilot.algorithms.FullStateCheckpointMixin``; OmniSafe's
``Metrics/EpCost`` is a 50-episode moving mean, not the batch). The constrained epochs are those
whose rollout starts at or after onset (e >= onset / E); epoch e ends at (e + 1) x E steps. An epoch
without a finished episode has no batch mean (NaN) and never counts as recovered; NaN is the only
such marker: a negative or infinite batch cost is impossible and refused (``ValueError``).
"""

from __future__ import annotations

import math
import numbers
from typing import Any, Sequence

from configs import registered as R
from metrics.controller import check_epoch_trace
from pilot.errors import require_answered


def recovery_steps(epochs: Sequence[Any], batch_costs: Sequence[Any], *, onset_step: int, steps_per_epoch: int,
                   budget: float = R.COST_LIMIT, total_steps: int | None = None) -> int | None:
    """Steps from onset to the end of the first constrained epoch with batch mean cost <= ``budget``; None if never.

    ``epochs``: 0, 1, 2, ... (``Train/Epoch``); ``batch_costs``: the epochs' ``Metrics/BatchEpCost``.
    With ``total_steps`` the trace must cover the whole run, so that None means "never" rather than
    "not yet"; pass the complete trace of a completed run either way. The smallest possible value is
    one epoch (E), the end of the first constrained epoch. ``budget`` must be a finite real number
    >= 0; a negative or infinite batch cost raises ``ValueError`` (corrupt data, not "no episode").
    """
    require_answered("Q-controller-quantities", what="the recovery time (Table 2.1)")
    if (isinstance(budget, bool) or not isinstance(budget, numbers.Real) or not math.isfinite(float(budget))
            or float(budget) < 0.0):
        raise ValueError(f"budget must be a finite non-negative number, got {budget!r}")
    start, steps, costs = check_epoch_trace(epochs, batch_costs, onset_step=onset_step, steps_per_epoch=steps_per_epoch,
                                            total_steps=total_steps, name="batch costs")
    bad = [c for c in costs if math.isinf(c) or c < 0.0]  # NaN (no finished episode) compares False
    if bad:
        raise ValueError(f"the batch costs have infinite or negative values {bad[:5]}")
    for e in range(start, len(costs)):
        if math.isfinite(costs[e]) and costs[e] <= float(budget):  # "falls to or below d"
            return (e + 1) * steps - int(onset_step)
    return None

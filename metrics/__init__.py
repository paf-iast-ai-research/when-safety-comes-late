"""Metrics and interventions (Role 3, owner Abdullah).

What this package implements, by pre-registration location:

* ``metrics.batches`` and ``metrics.collect_fixed_batch``: the fixed evaluation batch of each
  Study A task (Table 2.3 "Fixed evaluation batch": "2,048 states (decision) collected once per
  task by a uniform-random policy under seed 0, stored in the repository"; Appendix A "the fixed
  evaluation batch for each task"). The files are ``metrics/fixed_batch/<task>.npy`` with
  ``MANIFEST.json``; their SHA-256 hashes are committed in ``metrics/batches.py``.
* ``metrics.plasticity``: the three plasticity metrics of Table 2.3 (dormant-neuron fraction,
  equation 6; effective rank, equation 7; parameter norm) on the actor, the two critics logged
  separately, and the manipulation-check quantities of H3 (c) (Part 1.2).
* ``metrics.hook``: the logging hook that writes ``plasticity.csv`` (contract 2 in
  ``pilot/contracts.py``) at every checkpoint (Table 2.3 "Logging points"; Part 3.4).
* ``metrics.interventions``: partial reset and plasticity injection (Table 2.4; equation 8), and
  the loaders of injected actors for evaluation and continuations.
* ``metrics.controller`` and ``metrics.recovery``: the controller quantities of Table 2.4
  ("Multiplier overshoot", "Settling time"; Appendix B lambda_peak, lambda_final, settling_steps)
  and the recovery time of Table 2.1, at epoch resolution.
* ``metrics.recompute``: offline recomputation of the plasticity rows from saved checkpoints.

The values the pre-registration leaves open are answered in Table 9.1 (``configs.registered.PENDING``;
docs/DECISIONS.md): Q-plasticity-definitions and Q-reset-injection (run gates set by ``pilot/manifest.py``),
and Q-controller-quantities (a result gate called inside ``metrics.controller`` and ``metrics.recovery``),
each of which holds only while its key is open. The pilot's Study A runs are not gated (Part 3.6: "The pilot's runs are not
reused"; First Tasks section 9: the pilot must record the metrics from its first run).
"""

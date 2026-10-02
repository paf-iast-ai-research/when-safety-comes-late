"""Environment and tests (Role 2; owner Muhammad Talha Jamil): Study A's training plug-ins and the evaluation harness.

Each module is imported on its own; importing the package imports none of them, so the pipeline and
the harness never load OmniSafe's algorithms through it.

* ``envs.onset``: ``make_algorithm`` and ``make_pid_algorithm``, the plug-ins ``study_a`` and
  ``study_a_pid`` of ``pilot.manifest.PLUGINS``. Constraint onset (Table 2.1 "Constraint onset";
  equations 3 to 5), the controller variants and the mediation treatments (Table 2.4), on OmniSafe's
  PPO-Lagrangian and CPPOPID with ``pilot.algorithms.FullStateCheckpointMixin``.
* ``envs.continuations``: ``make_finetune_algorithm`` and ``make_transfer_algorithm``, the plug-ins
  ``battery_finetune`` and ``battery_transfer``. The battery's reward-only fine-tuning and transfer
  (Table 2.2), with the observation mapping between the Goal and Button tasks (Q-transfer-obs).
* ``envs.evaluation``: the evaluation harness (contracts 1, 5 and 6 of ``pilot/contracts.py``; Table
  2.1 "Evaluation cost of a checkpoint"; Table 2.2).

See envs/README.md and HANDOVER.md section 8.
"""

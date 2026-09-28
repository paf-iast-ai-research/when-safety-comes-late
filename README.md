# When Safety Comes Late, and Constraint-Coverage Generalization

Two pre-registered studies that share one pipeline (OmniSafe PPO-Lagrangian on Safety-Gymnasium).
**Study A, When Safety Comes Late**, asks whether a safety constraint added late in training gives
safety as robust as a constraint present from the start, when the policies compared are matched on
final in-distribution cost. It sweeps the onset fraction (0, 0.10, 0.25, 0.50), abrupt against
ramped onset, and two step-matching controls on three tasks. It tests whether loss of plasticity
(dormant neurons, effective rank) and overshoot of the Lagrange multiplier explain the effect.
**Study B, Constraint-Coverage Generalization**, asks whether the number and spacing of cost
budgets seen during training determine how well a budget-conditioned agent satisfies a budget it
has not seen, zero-shot and after a few-shot adaptation.

The registered plan is in [`prereg/`](prereg/). The repository's first commit
(`735b394d18d1bb046c7a74f900af00a83f1fa316`, 2026-09-18T23:22:45+05:00) is the registration, and
the plan governs every line of code here.

Research group: Muhammad Umair Waseem, Muhammad Talha Jamil, Hamza Nisar, Muhammad Abdullah,
Abdullah. Supervisor: Dr. Musadaq Mansoor. Pak-Austria Fachhochschule: Institute of Applied
Sciences and Technology, Haripur, Pakistan.

Start with [`HANDOVER.md`](HANDOVER.md). It covers what each folder is for, how to set up the
pinned environment, how to run the pilot, and the ordered list of work from now to the paper.

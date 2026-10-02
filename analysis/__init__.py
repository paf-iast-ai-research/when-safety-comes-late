"""Analysis and results (Role 4, Muhammad Abdullah): the registered analysis of Parts 4 and 5.

Modules:
  matching     Part 4.1 rules 1 to 7 (``select_checkpoint`` is pipeline contract 4; ``match_arms``
               gives the ledger's matched and infeasible flags)
  stats        Part 5.3 to 5.5: seed means, Cohen's d (equation 12), Welch and bootstrap intervals,
               IQM, Spearman with a seed-resampling interval, Holm, noncentral-t power, U80
  questions    which open pre-registration questions (configs.registered.PENDING) each verdict
               depends on
  verdict      statuses, labels, the other readings of open questions (UNDECIDED(key)), and the
               Part 5.5 and Part 5.7 annotations
  data         the ledger and the supplement records as per-seed tables (Part 5.1); rule 1 re-checked
  study_a      boxes H0 to H4 and the Study A secondary outcomes
  study_b      Part 4.2 and boxes G1 to G5
  records      AMENDMENTS.json (Part 8.2 labels) and ERRATA.json (Part 5.8 freeze)
  pilot_check  pilot mode: the go report's G1 and G3 numbers recomputed (Part 5.8 test run)
  report       tables, the Markdown report and the provenance of a run of the analysis
  __main__     ``python -m analysis --mode pilot|final --ledger PATH`` (Part 5.8)

This file imports nothing: ``pilot.enrichment`` imports ``analysis.matching`` inside the ``python
-m pilot enrich`` commands (``select``, ``match`` and ``sensitivity-battery``), which refuse any imported repository
file that the commit does not hold (``pilot.provenance.unverified_imported_code``), so the package's import must stay
minimal.
"""

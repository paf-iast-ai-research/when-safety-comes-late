"""The targeted search for Study B (Appendix C, Table C.1): its protocol and the check of its record.

Owner: Study B and literature (Role 5, Hamza Nisar); read by pilot/scheduler.py. ``protocol`` holds the
procedure (sources, query variants, known works, citation lists, screening depth and result order);
``record`` checks the hand-written record committed beside it (``check_record(root, read=None) ->
SearchStatus``, which carries the decision's outcome and the id of its amendment to the scheduler).
Nothing here performs the search or writes a result: the search is the Study B owner's act, and its
record is written by hand (studyb/search/README.md). ``python -m studyb.search`` prints the queries
and checks a record. The scheduler's launch gate (pilot/scheduler.py) calls
``check_record(<repo>/studyb/search, read=pilot.scheduler.committed_reader(commit, ...))``, a reader of
the files committed at HEAD; its ``SEARCH_RECORD_DIR`` equals ``RECORD_PATH``, the record's repository
path ``studyb/search``.
"""

from studyb.search.protocol import RESULT_ORDER, SCREENING_DEPTH, QueryVariant, query_variants
from studyb.search.record import RECORD_PATH, RECORD_ROOT, SearchStatus, check_record

__all__ = ["RECORD_PATH", "RECORD_ROOT", "RESULT_ORDER", "SCREENING_DEPTH", "QueryVariant", "SearchStatus",
           "check_record", "query_variants"]

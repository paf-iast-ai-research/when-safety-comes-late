"""Is the targeted-search record complete? (Appendix C, Table C.1; Part 7.2)

Owner: Study B and literature (Role 5, Hamza Nisar). The scheduler (pilot/scheduler.py) launches no
Study B run until the record committed at HEAD is complete and its outcome is not "included"
(``check_record(root, read=committed_reader(commit, root, repo_root))``; Part 7.2: the record "is committed to the repository
before Study B's first run"). Answered in Table 9.1 (Q-search-before-pilot): the pilot's two Study B
runs are Study B's first runs, so they wait for the record too, and a "partial" outcome holds them
until the amendment that ``decision.json`` names is committed in ``analysis/AMENDMENTS.json``.

The record is written by hand by the searcher; this module only reads it. It never writes a file,
and no code in the repository writes a search result, a citation, a date or a hit.

The record (``root`` is its directory, ``studyb/search`` in the repository; the path Part 9 Table 9.0
receives once the search is run):

* ``search_log.md``: the narrative log. At its top, before the first ``## `` section, four lines fix
  the screening depth and the result order before the first query (First Tasks, Role 5 step 1)::

      Screening depth: 50
      Result order: relevance
      Depth fixed on: 2026-10-01
      Fixed by: <the searcher's name>

  The depth must be ``protocol.SCREENING_DEPTH`` and the order must begin with
  ``protocol.RESULT_ORDER``, in any case (Q-search-depth; a source with no relevance order is taken in
  its default order, named in its searches' ``notes``). The date, which fixes both, is no later than
  any search's date (the log records dates, so "before the first query" is checked to the day).
* ``search_log.csv``: one row per query variant per source (``protocol.query_variants()`` x
  ``protocol.SOURCES``, 80 rows): ``source, query_id, query_string, date, searcher,
  results_reported, results_screened``, optional ``export_file`` and ``notes``. ``query_string`` is
  the protocol's string exactly (a source that needs another syntax is described in ``notes``);
  ``results_screened`` is ``min(depth, results_reported)``; ``export_file``, when given, is the
  path, inside the record, of the source's raw export of that search, committed with the record.
* ``screened/<source>__<query_id>.csv`` (e.g. ``screened/google_scholar__Q1-base.csv``): every result
  screened at title and abstract, each a "hit" (Q-search-record-scope), ``rank, citation, url_or_doi,
  stage1_decision, reason``, optional ``duplicate_of``; ranks 1 to ``results_screened`` in order,
  ``stage1_decision`` ``exclude``, ``full_text`` or ``duplicate``. ``url_or_doi`` and ``reason`` are
  filled on every row (``url_or_doi`` is ``none`` when the source gives neither). The reason of an
  ``exclude`` names the element of the inclusion criterion the result fails (no variation of budget
  levels, or no unseen budget evaluated) or the Table C.1 "Exclusion" clause it falls under; that of
  a ``duplicate`` names its first screened occurrence. A ``duplicate`` also gives, in ``duplicate_of``,
  the exact citation text of that occurrence, a row of some screened file that is not a duplicate
  (Q-search-citations); ``duplicate_of`` is empty on every other row.
* ``citations_log.csv``: one row per citation list screened, ``list_id, kind, of_work, date, searcher,
  items_reported, items_screened``, optional ``export_file`` and ``notes``; ``kind`` is ``references``
  or ``cited_by:<source>``. A list is screened in full, with no depth limit (Q-search-depth):
  ``items_screened`` equals ``items_reported``, the list's length as the source shows it. Required
  (Q-search-citations): (1) Yao et al.'s references and (2) its "cited by" lists in Google Scholar and
  Semantic Scholar (``of_work`` = ``protocol.CITATION_SEED`` or Yao et al.'s citation in the record);
  (3) for every work those two lists hold (the citation of each item that is not a duplicate and the
  ``duplicate_of`` of each that is, items excluded at title and abstract included), its references
  and its two "cited by" lists (``protocol.CITING_WORK_LISTS``, ``of_work`` = that citation text; its
  Google Scholar list may be absent when the notes of its Semantic Scholar row contain
  ``protocol.REFUSAL_NOTE``); (4) the references of every work a screened file took to full text, a
  known work of Table C.1 included (one ``references`` row serves (3) and (4)). Each list's screened
  results are in ``screened/citations__<list_id>.csv``, as above, with ``items_screened`` rows.
* ``search_record.csv``: one row per work that reached the full-text stage, and one for each of the
  three works Table C.1 lists as already known (Q-search-record-scope): ``citation, known_work,
  what_it_varies, what_it_evaluates, meets_inclusion, reason, exclusion_criterion, found_in, date,
  searcher``; every field filled except ``known_work``, which names the Table C.1 work
  (``protocol.KNOWN_WORKS``) on its row and is empty elsewhere; ``meets_inclusion`` is ``yes``,
  ``partial`` or ``no``; ``exclusion_criterion`` names the Table C.1 "Exclusion" clause the work falls
  under (``protocol.EXCLUSION_CRITERIA``) or is ``none`` (``protocol.NO_EXCLUSION``), and is ``none``
  on a ``yes`` row (a work under an exclusion clause is not included). Every ``full_text`` decision of
  a screened file has its row (the same citation text), and every row that is not a known work has
  such a ``full_text`` decision.
* ``decision.json``: ``{"outcome": "none" | "partial" | "included", "amendment": str | null, "date":
  "YYYY-MM-DD", "searcher": str}``. The outcome follows the record (any ``yes``: "included"; else any
  ``partial``: "partial"; else "none"; Table C.1 "Decision"). "partial" and "included" are recorded
  by amendment ("Any included work withdraws Study B by amendment; a partially overlapping work
  narrows Study B's claims by amendment, recorded before the first run"): ``amendment`` is the id of
  the ``analysis/AMENDMENTS.json`` entry that records it (a Table 9.1 row), exactly as written there;
  it is null for "none" (answered in Table 9.1, Q-search-before-pilot). The scheduler releases the
  Study B runs of a "partial" outcome only once that entry is committed (``SearchStatus.amendment``);
  an "included" outcome holds them all. The date is no earlier than any search and any
  classification of ``search_record.csv``.

Departures from the file layout suggested by the Study B specification (section 8.1; a build document
kept outside this repository; the scheduler relies only on ``check_record(root, read=None)`` and its
``SearchStatus``), recorded for the integrator: ``citations_log.csv`` splits the spec's one ``list``
column into ``list_id`` (the screened file's name), ``kind`` and ``of_work``, so the reference list of
a full-text work can be matched with its record row, and adds ``items_reported`` (Q-search-depth); the
screened files allow ``duplicate_of`` (Q-search-citations); ``search_record.csv`` adds ``known_work``
(which row is which Table C.1 known work); ``export_file`` is optional in both logs (a source may offer
no export), checked when given; ``citations_log.csv`` also allows an optional ``notes`` column, as
``search_log.csv`` has. Every other column is the spec's. First Tasks, Role 5 step 2 lists every query
in ``search_log.md``; here that listing is ``search_log.csv`` (with the ``screened/`` files and
``citations_log.csv``), and ``search_log.md`` is the narrative log, so the pull request contains all
of these files.

``check_record`` returns ``SearchStatus(complete, outcome, problems, amendment)``; the empty templates committed
with this module are incomplete. A malformed or unreadable file is a problem of the record, never an
exception.
"""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping

from studyb.search import protocol

SEARCH_LOG_MD = "search_log.md"
SEARCH_LOG_CSV = "search_log.csv"
CITATIONS_LOG_CSV = "citations_log.csv"
SEARCH_RECORD_CSV = "search_record.csv"
DECISION_JSON = "decision.json"
SCREENED_DIR = "screened"
# The record's directory in the working tree; in the repository it is studyb/search (RECORD_PATH), the
# path Part 9 Table 9.0 receives once the search is run.
RECORD_ROOT = Path(__file__).resolve().parent
RECORD_PATH = "studyb/search"
# The amendment log whose entry decision.json's "amendment" names by its "id" (Part 9, Table 9.1).
AMENDMENT_LOG = "analysis/AMENDMENTS.json"

SEARCH_LOG_COLUMNS = ("source", "query_id", "query_string", "date", "searcher", "results_reported", "results_screened")
CITATIONS_LOG_COLUMNS = ("list_id", "kind", "of_work", "date", "searcher", "items_reported", "items_screened")
SCREENED_COLUMNS = ("rank", "citation", "url_or_doi", "stage1_decision", "reason")
SCREENED_OPTIONAL_COLUMNS = ("duplicate_of",)  # filled on a 'duplicate' row, empty elsewhere (Q-search-citations)
SEARCH_RECORD_COLUMNS = ("citation", "known_work", "what_it_varies", "what_it_evaluates", "meets_inclusion", "reason",
                         "exclusion_criterion", "found_in", "date", "searcher")
OPTIONAL_COLUMNS = ("export_file", "notes")  # allowed in the two logs, never required (Study B spec 8.1 order)
SCREENING_DECISIONS = ("exclude", "full_text", "duplicate")
DECISION_FIELDS = ("outcome", "amendment", "date", "searcher")

# The four lines at the top of search_log.md that fix the screening depth and the result order before
# the first query (Q-search-depth).
DEPTH_FIELDS = MappingProxyType({"depth": "Screening depth", "order": "Result order", "date": "Depth fixed on",
                                 "searcher": "Fixed by"})
_LIST_ID = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_WHOLE = re.compile(r"^[0-9]+$")
_MAX_LISTED = 5  # problems name at most this many items of one kind, then count the rest

Reader = Callable[[str], "bytes | None"]


@dataclass(frozen=True)
class SearchStatus:
    """The state of the record: complete or not, the decision's outcome and amendment, and every problem found.

    ``outcome`` is decision.json's outcome when that file holds a valid one ("none", "partial",
    "included"), even while other parts are incomplete; None otherwise. ``amendment`` is, likewise,
    decision.json's amendment when it is a non-empty string: the id of the ``AMENDMENT_LOG`` entry that
    records the outcome (Table C.1 "Decision"), which the scheduler looks for in the amendment log
    committed with the record before it releases a "partial" outcome (Q-search-before-pilot); None
    otherwise. ``complete`` is True exactly when ``problems`` is empty; a complete "partial" or
    "included" record has an amendment, a complete "none" record has none.
    """

    complete: bool
    outcome: str | None
    problems: list[str] = field(default_factory=list)
    amendment: str | None = None


def screened_file(source: str, query_id: str) -> str:
    """The screened-results file of one query variant in one source, relative to the record."""
    return f"{SCREENED_DIR}/{protocol.source_slug(source)}__{query_id}.csv"


def citation_screened_file(list_id: str) -> str:
    """The screened-results file of one citation list, relative to the record."""
    return f"{SCREENED_DIR}/citations__{list_id}.csv"


def _disk_reader(path: str) -> bytes | None:
    """The working tree's file; None when it does not exist. Any other OSError (a directory, no
    permission) propagates to ``_Check.data``, which records the file as unreadable."""
    try:
        return Path(path).read_bytes()
    except (FileNotFoundError, NotADirectoryError):
        return None


def _listed(items: Iterable[Any]) -> str:
    items = list(items)
    shown = ", ".join(str(i) for i in items[:_MAX_LISTED])
    return shown + (f" and {len(items) - _MAX_LISTED} more" if len(items) > _MAX_LISTED else "")


def _date(text: str) -> date | None:
    try:
        return date.fromisoformat(text.strip()) if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text.strip()) else None
    except ValueError:
        return None


class _Check:
    """One run of ``check_record``: reads through ``read`` and collects problems."""

    def __init__(self, root: str | Path, read: Reader | None) -> None:
        self.root = PurePosixPath(Path(root).as_posix())
        self.read = read if read is not None else _disk_reader
        self.problems: list[str] = []
        self.depth: int | None = None
        self.depth_date: date | None = None
        self.search_dates: list[date] = []
        self.record_dates: list[date] = []  # the classifications of search_record.csv
        self.full_text: dict[str, list[str]] = {}  # citation -> where it was taken to full text
        self.originals: set[str] = set()  # the citations of the screened results that are not duplicates
        self.duplicates: list[tuple[str, str]] = []  # (file and line, duplicate_of) of every duplicate naming one
        self.exports: set[str] = set()  # export files already looked for

    def problem(self, text: str) -> None:
        self.problems.append(text)

    def data(self, name: str, *, missing: str | None) -> bytes | None:
        """The bytes of the record's file ``name``; None when it is missing (``missing`` is then the
        problem recorded, if given) or cannot be read (recorded here: a malformed record is a problem
        of the record, never an exception of the check)."""
        try:
            data = self.read((self.root / name).as_posix())
        except OSError as exc:
            self.problem(f"{name} cannot be read ({type(exc).__name__}: {exc})")
            return None
        if data is None and missing is not None:
            self.problem(missing)
        return data

    def text(self, name: str, *, missing_note: str = "") -> str | None:
        """The file's text; None, with its problem recorded, when it is missing, unreadable or not UTF-8.
        ``missing_note`` is appended to "<name> is missing" in that problem."""
        data = self.data(name, missing=f"{name} is missing{missing_note}")
        if data is None:
            return None
        try:
            return data.decode("utf-8-sig")
        except UnicodeDecodeError:
            self.problem(f"{name} is not UTF-8 text")
            return None

    def rows(self, name: str, required: tuple[str, ...], *,
             optional: tuple[str, ...] = ()) -> list[tuple[int, dict[str, str]]] | None:
        """The rows of a CSV file with every required column (extra columns are refused unless optional),
        each with the file's line on which it ends (blank lines and quoted line breaks counted)."""
        text = self.text(name)
        if text is None:
            return None
        rows = []
        try:  # the csv module raises on what no CSV file holds (a NUL byte, a field over its size limit)
            reader = csv.DictReader(io.StringIO(text, newline=""))
            header = [h.strip() for h in (reader.fieldnames or [])]
            missing = [c for c in required if c not in header]
            unknown = [c for c in header if c not in required and c not in optional]
            if missing or unknown or len(set(header)) != len(header):
                self.problem(f"{name}: the header {header} must hold the columns {list(required)}"
                             + (f" (optional: {list(optional)})" if optional else ""))
                return None
            for row in reader:
                line = reader.line_num  # the physical line where the row ends, not a count of rows
                if None in row or any(v is None for v in row.values()):
                    self.problem(f"{name} line {line}: the row does not have the header's {len(header)} fields")
                    continue
                rows.append((line, {k.strip(): (v or "").strip() for k, v in row.items()}))
        except csv.Error as exc:
            self.problem(f"{name} is not a readable CSV file: {exc}")
            return None
        return rows

    # -- search_log.md -----------------------------------------------------------------------

    def depth_header(self) -> None:
        text = self.text(SEARCH_LOG_MD)
        if text is None:
            return
        head = ("\n" + text).split("\n## ", 1)[0]  # the top of the log, before its first section
        found: dict[str, str] = {}
        for line in head.splitlines():
            for key, label in DEPTH_FIELDS.items():
                match = re.fullmatch(rf"\s*{re.escape(label)}:\s*(.*?)\s*", line)
                if match and key not in found:
                    found[key] = match.group(1)
        depth = found.get("depth", "")
        if not _WHOLE.fullmatch(depth):
            self.problem(f"{SEARCH_LOG_MD}: the screening depth is not fixed at the top of the log "
                         f"('{DEPTH_FIELDS['depth']}: <number>'; Q-search-depth)")
        elif int(depth) != protocol.SCREENING_DEPTH:
            self.problem(f"{SEARCH_LOG_MD}: the screening depth {depth} is not the protocol's "
                         f"{protocol.SCREENING_DEPTH} (studyb/search/protocol.py SCREENING_DEPTH)")
        else:
            self.depth = int(depth)
        self.depth_date = _date(found.get("date", ""))
        if self.depth_date is None:
            self.problem(f"{SEARCH_LOG_MD}: '{DEPTH_FIELDS['date']}: YYYY-MM-DD' is missing or not a date")
        if not found.get("order", "").lower().startswith(protocol.RESULT_ORDER):
            self.problem(f"{SEARCH_LOG_MD}: the result order is not fixed at the top of the log "
                         f"('{DEPTH_FIELDS['order']}: {protocol.RESULT_ORDER}', each source's relevance order; "
                         "Q-search-depth)")
        if not found.get("searcher") or found["searcher"].startswith("("):
            self.problem(f"{SEARCH_LOG_MD}: '{DEPTH_FIELDS['searcher']}: <name>' is missing")

    # -- dated rows ----------------------------------------------------------------------------

    def dated(self, name: str, where: str, row: Mapping[str, str]) -> None:
        when = _date(row.get("date", ""))
        if when is None:
            self.problem(f"{name} {where}: date {row.get('date', '')!r} is not YYYY-MM-DD")
        else:
            self.search_dates.append(when)
            if self.depth_date is not None and when < self.depth_date:
                self.problem(f"{name} {where}: searched on {when}, before the screening depth was fixed on "
                             f"{self.depth_date} (First Tasks, Role 5 step 1)")
        if not row.get("searcher"):
            self.problem(f"{name} {where}: the searcher's name is missing (Table C.1 'Record')")

    def export(self, name: str, where: str, row: Mapping[str, str]) -> None:
        """A log row's optional ``export_file``: a path inside the record, to a file committed with it."""
        value = row.get("export_file", "")
        if not value:
            return
        path = PurePosixPath(value)
        if "\\" in value or path.is_absolute() or ".." in path.parts or not path.parts:
            self.problem(f"{name} {where}: export_file {value!r} must be a relative path inside the record")
            return
        if path.as_posix() not in self.exports:
            self.exports.add(path.as_posix())
            self.data(path.as_posix(), missing=f"{name} {where}: export_file {value!r} is not in the record")

    def screened(self, name: str, expected: int, found_in: str) -> list[dict[str, str]]:
        """A screened-results file with ``expected`` rows, which it returns (none if it cannot be read).
        Its full-text decisions, the citations of its results that are not duplicates and the
        ``duplicate_of`` of its duplicates are collected (the last are matched by ``duplicate_targets``)."""
        rows = self.rows(name, SCREENED_COLUMNS, optional=SCREENED_OPTIONAL_COLUMNS)
        if rows is None:
            return []
        if len(rows) != expected:
            self.problem(f"{name}: {len(rows)} screened results, the log says {expected}")
        ranks = [r["rank"] for _, r in rows]
        if ranks != [str(i) for i in range(1, len(rows) + 1)]:
            self.problem(f"{name}: ranks must run 1, 2, ... in order, got {_listed(ranks)}")
        for line, row in rows:
            where = f"{name} line {line}"
            if not row["citation"]:
                self.problem(f"{where}: the citation is missing")
            if not row["url_or_doi"]:
                self.problem(f"{where}: url_or_doi is missing ('none' when the source gives neither; "
                             "Q-search-record-scope)")
            if not row["reason"]:
                self.problem(f"{where}: the reason is missing (Table C.1 'Record': whether it meets the inclusion "
                             "criterion; Q-search-record-scope)")
            decision, duplicate_of = row["stage1_decision"], row.get("duplicate_of", "")
            if decision not in SCREENING_DECISIONS:
                self.problem(f"{where}: stage1_decision {decision!r} is not one of {SCREENING_DECISIONS}")
            elif decision == "duplicate":
                if duplicate_of:
                    self.duplicates.append((where, duplicate_of))
                else:
                    self.problem(f"{where}: a duplicate names the citation of its first screened occurrence in "
                                 "duplicate_of (Q-search-citations)")
            else:
                if duplicate_of:
                    self.problem(f"{where}: duplicate_of is filled, but the result is not a duplicate")
                if row["citation"]:
                    self.originals.add(row["citation"])
                    if decision == "full_text":
                        self.full_text.setdefault(row["citation"], []).append(found_in)
        return [row for _, row in rows]

    def duplicate_targets(self) -> None:
        """Every duplicate names a screened result that is not a duplicate (Q-search-citations), once every
        screened file of the record has been read."""
        unmatched = [f"{where} ({target!r})" for where, target in self.duplicates if target not in self.originals]
        if unmatched:
            self.problem(f"duplicates whose duplicate_of is the citation of no screened result that is not a "
                         f"duplicate: {len(unmatched)} (Q-search-citations): {_listed(unmatched)}")

    # -- the logs --------------------------------------------------------------------------------

    def search_log(self) -> None:
        rows = self.rows(SEARCH_LOG_CSV, SEARCH_LOG_COLUMNS, optional=OPTIONAL_COLUMNS)
        if rows is None:
            return
        variants = {v.query_id: v for v in protocol.query_variants()}
        seen: set[tuple[str, str]] = set()
        for line, row in rows:
            where = f"line {line}"
            source, query_id = row["source"], row["query_id"]
            if source not in protocol.SOURCES:
                self.problem(f"{SEARCH_LOG_CSV} {where}: source {source!r} is not one of {protocol.SOURCES}")
                continue
            if query_id not in variants:
                self.problem(f"{SEARCH_LOG_CSV} {where}: query_id {query_id!r} is not a query variant of the protocol")
                continue
            if (source, query_id) in seen:
                self.problem(f"{SEARCH_LOG_CSV} {where}: {query_id} in {source} is logged twice")
                continue
            seen.add((source, query_id))
            if row["query_string"] != variants[query_id].text:
                self.problem(f"{SEARCH_LOG_CSV} {where}: query_string {row['query_string']!r} is not the protocol's "
                             f"{variants[query_id].text!r}")
            self.dated(SEARCH_LOG_CSV, where, row)
            self.export(SEARCH_LOG_CSV, where, row)
            reported, screened = row["results_reported"], row["results_screened"]
            if not (_WHOLE.fullmatch(reported) and _WHOLE.fullmatch(screened)):
                self.problem(f"{SEARCH_LOG_CSV} {where}: results_reported and results_screened must be whole numbers")
                continue
            if self.depth is not None and int(screened) != min(self.depth, int(reported)):
                self.problem(f"{SEARCH_LOG_CSV} {where}: {screened} results screened; the depth {self.depth} of "
                             f"{reported} reported gives {min(self.depth, int(reported))}")
            self.screened(screened_file(source, query_id), int(screened), f"{source} {query_id}")
        absent = [f"{q} in {s}" for s in protocol.SOURCES for q in variants if (s, q) not in seen]
        if absent:
            total = len(protocol.SOURCES) * len(variants)
            self.problem(f"{SEARCH_LOG_CSV}: searches not logged: {len(absent)} of {total} (every query of Table C.1 "
                         f"in every source): {_listed(absent)}")

    def citations_log(self, record: list[dict[str, str]] | None) -> None:
        rows = self.rows(CITATIONS_LOG_CSV, CITATIONS_LOG_COLUMNS, optional=OPTIONAL_COLUMNS)
        if rows is None:
            return
        seed_names = {protocol.CITATION_SEED}
        for row in record or ():
            if row.get("known_work") == protocol.CITATION_SEED and row.get("citation"):
                seed_names.add(row["citation"])

        def work(citation: str) -> str:  # the seed under any of its names
            return protocol.CITATION_SEED if citation in seed_names else citation

        lists: set[tuple[str, str]] = set()
        ids: set[str] = set()
        refused: set[str] = set()  # works whose Semantic Scholar "cited by" row notes Google Scholar's refusal
        citing: dict[str, None] = {}  # the works of the seed's "cited by" lists, item (3), in order of first sight
        for line, row in rows:
            where = f"line {line}"
            list_id, kind, of_work = row["list_id"], row["kind"], row["of_work"]
            if not _LIST_ID.fullmatch(list_id) or list_id in ids:
                self.problem(f"{CITATIONS_LOG_CSV} {where}: list_id {list_id!r} must be unique and made of lower-case "
                             "letters, digits, '_' or '-', starting with a letter or digit")
                continue
            ids.add(list_id)
            if kind not in protocol.CITATION_LIST_KINDS:
                self.problem(f"{CITATIONS_LOG_CSV} {where}: kind {kind!r} is not one of {protocol.CITATION_LIST_KINDS}")
                continue
            if not of_work:
                self.problem(f"{CITATIONS_LOG_CSV} {where}: of_work (whose list it is) is missing")
                continue
            lists.add((kind, work(of_work)))
            if kind == protocol.REFUSAL_NOTED_ON and protocol.REFUSAL_NOTE in row.get("notes", ""):
                refused.add(work(of_work))
            self.dated(CITATIONS_LOG_CSV, where, row)
            self.export(CITATIONS_LOG_CSV, where, row)
            reported, screened = row["items_reported"], row["items_screened"]
            if not (_WHOLE.fullmatch(reported) and _WHOLE.fullmatch(screened)):
                self.problem(f"{CITATIONS_LOG_CSV} {where}: items_reported and items_screened must be whole numbers")
                continue
            if int(screened) != int(reported):
                self.problem(f"{CITATIONS_LOG_CSV} {where}: {list_id}: {screened} of {reported} items screened; "
                             "citation lists are screened in full (Q-search-depth)")
            items = self.screened(citation_screened_file(list_id), int(screened), f"{kind} of {of_work}")
            if kind != protocol.REFERENCES and work(of_work) == protocol.CITATION_SEED:
                for item in items:  # an excluded item cites the seed too; a duplicate is the work it names
                    cited = item.get("duplicate_of", "") if item["stage1_decision"] == "duplicate" else item["citation"]
                    if cited:
                        citing.setdefault(cited)
        for kind in protocol.SEED_CITATION_LISTS:
            if (kind, protocol.CITATION_SEED) not in lists:
                self.problem(f"{CITATIONS_LOG_CSV}: the {kind} list of {protocol.CITATION_SEED} is not logged "
                             "(Table C.1 'Sources'; Q-search-citations)")
        # Item (3): every work citing the seed has every list of CITING_WORK_LISTS, its Google Scholar
        # "cited by" list excepted when its Semantic Scholar row records that Google Scholar refused it.
        short = []
        for citation in citing:
            missing = [kind for kind in protocol.CITING_WORK_LISTS if (kind, work(citation)) not in lists
                       and not (kind == protocol.REFUSABLE_LIST and work(citation) in refused)]
            if missing:
                short.append(f"{citation} ({', '.join(missing)})")
        if short:
            self.problem(f"{CITATIONS_LOG_CSV}: works citing {protocol.CITATION_SEED} whose citation lists are not all "
                         f"logged: {len(short)} (Table C.1 'Sources': 'and of every paper that cites it'; "
                         f"Q-search-citations): {_listed(short)}")
        # Item (4): every work taken to full text in a screened file, a Table C.1 known work included
        # (README step 4), and every record row that is not a known work (a row with no full-text decision
        # is reported by full_text_consistency as well).
        hits = list(self.full_text)
        hits += [row["citation"] for row in record or ()
                 if not row.get("known_work") and row.get("citation") and row["citation"] not in self.full_text]
        unlisted = [c for c in hits if (protocol.REFERENCES, work(c)) not in lists]
        if unlisted:
            self.problem(f"{CITATIONS_LOG_CSV}: full-text works whose reference list is not logged: {len(unlisted)} "
                         f"(Q-search-citations): {_listed(unlisted)}")

    # -- the record and the decision -----------------------------------------------------------

    def search_record(self) -> list[dict[str, str]] | None:
        numbered = self.rows(SEARCH_RECORD_CSV, SEARCH_RECORD_COLUMNS)
        if numbered is None:
            return None
        citations: set[str] = set()
        known: dict[str, int] = {}
        for line, row in numbered:
            where = f"line {line}"
            empty = [c for c in SEARCH_RECORD_COLUMNS if c != "known_work" and not row[c]]
            if empty:
                self.problem(f"{SEARCH_RECORD_CSV} {where}: {', '.join(empty)} not filled (Table C.1 'Record')")
            if row["citation"] in citations:
                self.problem(f"{SEARCH_RECORD_CSV} {where}: the citation {row['citation']!r} appears twice")
            citations.add(row["citation"])
            if row["known_work"]:
                if row["known_work"] not in protocol.KNOWN_WORKS:
                    self.problem(f"{SEARCH_RECORD_CSV} {where}: known_work {row['known_work']!r} is not one of "
                                 f"{protocol.KNOWN_WORKS} (Table C.1 'Already known')")
                else:
                    known[row["known_work"]] = known.get(row["known_work"], 0) + 1
            if row["meets_inclusion"] and row["meets_inclusion"] not in protocol.INCLUSION_VALUES:
                self.problem(f"{SEARCH_RECORD_CSV} {where}: meets_inclusion {row['meets_inclusion']!r} is not one of "
                             f"{protocol.INCLUSION_VALUES}")
            criterion = row["exclusion_criterion"]
            allowed = (*protocol.EXCLUSION_CRITERIA, protocol.NO_EXCLUSION)
            if criterion and criterion not in allowed:
                self.problem(f"{SEARCH_RECORD_CSV} {where}: exclusion_criterion {criterion!r} is not one of {allowed} "
                             "(Table C.1 'Exclusion')")
            elif criterion and row["meets_inclusion"] == "yes" and criterion != protocol.NO_EXCLUSION:
                self.problem(f"{SEARCH_RECORD_CSV} {where}: meets_inclusion 'yes', but the work falls under the "
                             f"exclusion clause {criterion!r} (Table C.1 'Exclusion'); name one, not both")
            if row["date"]:
                when = _date(row["date"])
                if when is None:
                    self.problem(f"{SEARCH_RECORD_CSV} {where}: date {row['date']!r} is not YYYY-MM-DD")
                else:
                    self.record_dates.append(when)
        for work in protocol.KNOWN_WORKS:
            if known.get(work, 0) != 1:
                self.problem(f"{SEARCH_RECORD_CSV}: {work} must appear once with its classification (Table C.1 "
                             f"'Already known'; First Tasks, Role 5 step 3), found {known.get(work, 0)}")
        return [row for _, row in numbered]

    def full_text_consistency(self, record: list[dict[str, str]] | None) -> None:
        if record is None:
            return
        in_record = {row["citation"] for row in record}
        unrecorded = sorted(c for c in self.full_text if c not in in_record)
        if unrecorded:
            self.problem(f"{SEARCH_RECORD_CSV}: works taken to full text with no row: {len(unrecorded)} "
                         f"(Table C.1: 'For every hit'; Q-search-record-scope): {_listed(unrecorded)}")
        unexplained = [row["citation"] for row in record if not row["known_work"] and row["citation"] not in self.full_text]
        if unexplained:
            self.problem(f"{SEARCH_RECORD_CSV}: rows that are neither a known work nor a full_text decision of a "
                         f"screened file: {len(unexplained)} (Q-search-record-scope): {_listed(unexplained)}")

    def decision(self, record: list[dict[str, str]] | None) -> tuple[str | None, str | None]:
        """decision.json's (outcome, amendment), each None unless valid (``SearchStatus``)."""
        text = self.text(DECISION_JSON, missing_note=" (Table C.1 'Decision')")
        if text is None:
            return None, None
        try:
            data = json.loads(text)
        except ValueError as exc:
            self.problem(f"{DECISION_JSON} is not JSON: {exc}")
            return None, None
        if not isinstance(data, dict) or sorted(data) != sorted(DECISION_FIELDS):
            self.problem(f"{DECISION_JSON} must hold exactly the fields {list(DECISION_FIELDS)}")
            return None, None
        outcome = data["outcome"]
        if outcome not in protocol.OUTCOMES:
            self.problem(f"{DECISION_JSON}: outcome {outcome!r} is not one of {protocol.OUTCOMES}")
            outcome = None
        if record is not None and outcome is not None:
            classes = {row["meets_inclusion"] for row in record}
            expected = "included" if "yes" in classes else "partial" if "partial" in classes else "none"
            if outcome != expected:
                self.problem(f"{DECISION_JSON}: outcome {outcome!r}, but the record gives {expected!r} (Table C.1 "
                             "'Decision': any included work withdraws Study B, a partial overlap narrows it)")
        # Q-search-before-pilot: the amendment is named by the id of its entry in the amendment log, the
        # id the scheduler looks for there before it releases a "partial" outcome.
        amendment = data["amendment"]
        named = amendment if isinstance(amendment, str) and amendment.strip() else None
        if amendment is not None and not isinstance(amendment, str):
            self.problem(f"{DECISION_JSON}: amendment must be text or null")
        elif named is not None and named != named.strip():
            self.problem(f"{DECISION_JSON}: amendment {named!r} must be the id of its {AMENDMENT_LOG} entry exactly, "
                         "without surrounding blanks")
        if outcome in ("partial", "included") and named is None:
            self.problem(f"{DECISION_JSON}: outcome {outcome!r} is recorded 'by amendment' (Table C.1 'Decision': an "
                         "included work withdraws Study B, a partial overlap narrows its claims); name the amendment "
                         f"by the id of its {AMENDMENT_LOG} entry")
        elif outcome == "none" and amendment is not None:
            self.problem(f"{DECISION_JSON}: outcome 'none' is recorded by no amendment; amendment must be null, not "
                         f"{amendment!r}")
        decided = _date(data["date"]) if isinstance(data["date"], str) else None
        if decided is None:
            self.problem(f"{DECISION_JSON}: date {data['date']!r} is not YYYY-MM-DD")
        else:
            if self.search_dates and decided < max(self.search_dates):
                self.problem(f"{DECISION_JSON}: decided on {decided}, before the last search or citation list "
                             f"screened on {max(self.search_dates)}")
            if self.record_dates and decided < max(self.record_dates):
                self.problem(f"{DECISION_JSON}: decided on {decided}, before the last classification of "
                             f"{SEARCH_RECORD_CSV} on {max(self.record_dates)} (the decision follows the record)")
        if not isinstance(data["searcher"], str) or not data["searcher"].strip():
            self.problem(f"{DECISION_JSON}: the searcher's name is missing")
        return outcome, named


def check_record(root: str | Path, read: Reader | None = None) -> SearchStatus:
    """Check the targeted-search record in ``root`` against the protocol (see the module docstring).

    ``read(path) -> bytes | None`` reads one file, ``path`` being ``root`` joined with the file's name
    in POSIX form (``studyb/search/search_log.csv`` for ``root='studyb/search'``); None means the file
    does not exist. The scheduler passes a reader of the files committed at HEAD
    (``pilot.scheduler.committed_reader(commit, root, repo_root)``, built on
    ``pilot.provenance.file_committed_at``), so an uncommitted record never opens the gate; the
    default reads the working tree. The check reads only the files it names (it lists no directory;
    ``AMENDMENT_LOG``, outside the record, is the scheduler's to read).
    A file the reader cannot read (``OSError``) or that is not UTF-8 or not valid CSV/JSON is reported
    in ``problems``, so ``python -m studyb.search check`` lists it instead of stopping.
    """
    check = _Check(root, read)
    check.depth_header()
    record = check.search_record()
    check.search_log()
    check.citations_log(record)
    check.duplicate_targets()
    check.full_text_consistency(record)
    outcome, amendment = check.decision(record)
    return SearchStatus(complete=not check.problems, outcome=outcome, problems=list(check.problems),
                        amendment=amendment)

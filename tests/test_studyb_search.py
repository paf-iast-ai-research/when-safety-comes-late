"""Study B's targeted search (Appendix C, Table C.1): the protocol constants and the record check.

Every record below is a synthetic fixture built in ``tmp_path`` (citations "Fixture work ...",
searcher "Fixture Searcher", dates around 2030): no search result is written anywhere, and nothing is
written into the repository (studyb/search/README.md: the search is the Study B owner's act).
"""

from __future__ import annotations

import csv
import html
import json
import re
import zipfile
from functools import lru_cache
from pathlib import Path

import pytest

from configs import registered as R
from studyb import search
from studyb.search import protocol, record
from studyb.search.__main__ import main as search_main

REPO = Path(__file__).resolve().parents[1]
DOCX = REPO / "prereg" / "Preregistration.docx"
SEARCHER = "Fixture Searcher"
DEPTH_DAY = "2030-01-01"
DAY = "2030-01-02"
HIT = "Fixture work taken to full text"
EXPORT = "exports/google_scholar__Q1-base.ris"  # a fixture export file of the first search


@lru_cache(maxsize=1)
def _prereg_text() -> str:
    xml = zipfile.ZipFile(DOCX).read("word/document.xml").decode("utf-8")
    text = html.unescape(re.sub(r"<[^>]+>", "", re.sub(r"</w:p>", "\n", xml)))
    return re.sub(r"\s+", " ", text.replace("“", '"').replace("”", '"'))


# ---------------------------------------------------------------------------
# The protocol
# ---------------------------------------------------------------------------


def test_the_protocol_takes_sources_queries_and_phrases_from_the_registry() -> None:
    assert protocol.SOURCES == tuple(R.SEARCH_SOURCES)
    assert protocol.QUERIES == tuple(R.SEARCH_QUERIES)
    assert protocol.ADDED_PHRASES == tuple(R.SEARCH_ADDED_PHRASES)
    assert protocol.CITATION_SEED == R.SEARCH_CITATION_SEED == protocol.KNOWN_WORKS[0]
    assert set(protocol.CITED_BY_SOURCES) <= set(protocol.SOURCES)  # First Tasks, Role 5 step 5: their "cited by" lists


def test_the_seed_and_every_work_citing_it_owe_every_kind_of_citation_list() -> None:
    """Q-search-citations (Table 9.1): items (1)-(3), the references and both "cited by" lists of Yao et
    al. and of every work citing it; a citing work's Google Scholar list may give way to a noted refusal."""
    kinds = ("references", "cited_by:Google Scholar", "cited_by:Semantic Scholar")
    assert protocol.CITATION_LIST_KINDS == protocol.SEED_CITATION_LISTS == protocol.CITING_WORK_LISTS == kinds
    assert (protocol.REFUSABLE_LIST, protocol.REFUSAL_NOTED_ON) == kinds[1:]
    assert protocol.REFUSAL_NOTE == "Google Scholar refused"


def test_query_variants_are_each_query_with_and_without_each_phrase() -> None:
    """Q-search-queries: the four variants of every query (none, each phrase, both): 20 strings."""
    variants = protocol.query_variants()
    assert len(variants) == len(R.SEARCH_QUERIES) * 4 == 20
    assert len({v.query_id for v in variants}) == len({v.text for v in variants}) == 20
    for number, query in enumerate(R.SEARCH_QUERIES, start=1):
        own = {v.variant: v for v in variants if v.number == number}
        thresholds, spacing = R.SEARCH_ADDED_PHRASES
        assert {k: v.text for k, v in own.items()} == {
            "base": query, "thresholds": f"{query} {thresholds}", "spacing": f"{query} {spacing}",
            "both": f"{query} {thresholds} {spacing}",
        }
        assert [v.query_id for v in own.values()] == [f"Q{number}-{k}" for k in ("base", "thresholds", "spacing", "both")]


def test_the_known_works_are_table_c1s_and_the_depth_and_order_are_marked_answered() -> None:
    text = _prereg_text()
    row = text[text.index("Already known Yao"):]  # the table's cells are paragraphs of the .docx
    row = row[: row.index("Record For every hit")]
    for work in protocol.KNOWN_WORKS:
        assert f"{work}:" in row, work  # each label opens its entry of Table C.1 "Already known", verbatim
    source = (REPO / "studyb" / "search" / "protocol.py").read_text(encoding="utf-8")
    assert re.search(r"^SCREENING_DEPTH: int = 50  # answered in Table 9\.1 \(Q-search-depth\)$", source, re.M)
    assert re.search(r'^RESULT_ORDER: str = "relevance"  # answered in Table 9\.1 \(Q-search-depth\)$', source, re.M)
    assert "PROPOSAL" not in source  # Q-search-depth, -queries and -citations are answered in Table 9.1
    assert protocol.SCREENING_DEPTH == search.SCREENING_DEPTH == 50
    assert protocol.RESULT_ORDER == search.RESULT_ORDER == "relevance"  # the public API of studyb/README.md


def test_source_slugs_name_the_screened_files() -> None:
    assert [protocol.source_slug(s) for s in protocol.SOURCES] == ["google_scholar", "semantic_scholar", "arxiv", "openreview"]
    assert record.screened_file("Google Scholar", "Q1-base") == "screened/google_scholar__Q1-base.csv"
    with pytest.raises(ValueError):
        protocol.source_slug("--")


def test_the_committed_templates_have_the_checked_headers_and_the_check_runs_on_them() -> None:
    """The repository's record (templates until the search is run) has the columns check_record reads."""
    root = record.RECORD_ROOT
    assert root == REPO / "studyb" / "search" == REPO / search.RECORD_PATH
    from pilot import scheduler

    assert scheduler.SEARCH_RECORD_DIR == search.RECORD_PATH  # the launch gate's directory is this record's
    for name, columns in ((record.SEARCH_LOG_CSV, record.SEARCH_LOG_COLUMNS + record.OPTIONAL_COLUMNS),
                          (record.CITATIONS_LOG_CSV, record.CITATIONS_LOG_COLUMNS + record.OPTIONAL_COLUMNS),
                          (record.SEARCH_RECORD_CSV, record.SEARCH_RECORD_COLUMNS)):
        with open(root / name, newline="", encoding="utf-8") as fh:
            assert tuple(next(csv.reader(fh))) == columns, name
    with open(root / record.SCREENED_DIR / "README.md", encoding="utf-8") as fh:
        readme = fh.read()
    assert "`" + ",".join(record.SCREENED_COLUMNS) + "`" in readme  # the columns the screened files need
    assert all(f"`{c}`" in readme for c in record.SCREENED_OPTIONAL_COLUMNS)
    log = (root / record.SEARCH_LOG_MD).read_text(encoding="utf-8")
    for label in record.DEPTH_FIELDS.values():  # the lines the searcher fills in before the first query
        assert re.search(rf"^{re.escape(label)}:", log, re.M), label
    status = search.check_record(root)
    assert isinstance(status, search.SearchStatus)
    # Study B spec 8.2: "The empty template must fail". Each assertion holds for as long as the part of
    # the record it reads is still the template (they stay true once the real record is committed).
    if not (root / record.DECISION_JSON).exists():
        assert not status.complete and status.outcome is None and status.amendment is None
        assert "decision.json is missing (Table C.1 'Decision')" in status.problems
    if "Screening depth: (to be fixed before the first query)" in log:
        assert not status.complete and any("screening depth is not fixed" in p for p in status.problems)
    if "Result order: (relevance; fixed before the first query)" in log:
        assert not status.complete and any("result order is not fixed" in p for p in status.problems)
    if (root / record.SEARCH_LOG_CSV).read_text(encoding="utf-8").count("\n") == 1:  # the header only
        assert any("searches not logged: 80 of 80" in p for p in status.problems)


def test_the_record_files_have_the_columns_of_the_study_b_spec() -> None:
    """Study B spec 8.1: search_log.csv and citations_log.csv carry export_file; the screened files'
    title-and-abstract column is stage1_decision; search_record.csv names the Table C.1 exclusion
    clause (the departures the record.py docstring records are known_work, the list columns with
    items_reported, the screened files' optional duplicate_of, an optional export_file and the citations
    log's optional notes)."""
    assert record.SCREENED_COLUMNS == ("rank", "citation", "url_or_doi", "stage1_decision", "reason")
    assert record.SCREENED_OPTIONAL_COLUMNS == ("duplicate_of",)  # Q-search-citations
    assert record.CITATIONS_LOG_COLUMNS == ("list_id", "kind", "of_work", "date", "searcher", "items_reported",
                                            "items_screened")  # Q-search-depth: a list's length beside its screening
    assert record.DEPTH_FIELDS["order"] == "Result order"
    assert record.OPTIONAL_COLUMNS == ("export_file", "notes")
    assert record.SEARCH_RECORD_COLUMNS == ("citation", "known_work", "what_it_varies", "what_it_evaluates",
                                            "meets_inclusion", "reason", "exclusion_criterion", "found_in", "date",
                                            "searcher")
    text = _prereg_text()
    row = text[text.index("Exclusion Works that vary"):]
    row = row[: row.index("Already known")]
    assert protocol.NO_EXCLUSION not in protocol.EXCLUSION_CRITERIA and len(protocol.EXCLUSION_CRITERIA) == 3
    for clause in protocol.EXCLUSION_CRITERIA.values():
        assert clause in row, clause  # each label stands for a clause of Table C.1 "Exclusion", verbatim


# ---------------------------------------------------------------------------
# A synthetic record
# ---------------------------------------------------------------------------


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


def _screened(path: Path, items: list[tuple[str, ...]]) -> None:
    """A screened file of (citation, decision) items, or (citation, "duplicate", duplicate_of); the optional
    duplicate_of column only when an item names one. Rank 1's url_or_doi is 'none' (the source gave none)."""
    named = any(len(item) == 3 for item in items)
    rows = [{"rank": i, "citation": c, "url_or_doi": "none" if i == 1 else f"https://example.invalid/{i}",
             "stage1_decision": d, "reason": "fixture", **({"duplicate_of": of[0] if of else ""} if named else {})}
            for i, (c, d, *of) in enumerate(items, start=1)]
    _write_csv(path, record.SCREENED_COLUMNS + (record.SCREENED_OPTIONAL_COLUMNS if named else ()), rows)


ORDER_LINE = "Result order: Relevance (OpenReview: its default order, named in the notes)"
CITING = ("Fixture citing work 1", "Fixture citing work 2")  # the works of Yao et al.'s "cited by" lists
CITING_SLUGS = ("references", "cited-by-gs", "cited-by-ss")  # their lists' ids, in protocol.CITING_WORK_LISTS order


def _citing_list(n: int, slug: str) -> str:
    """The list_id of a list of citing work ``n`` (1 or 2), e.g. 'citing1-cited-by-gs'."""
    return f"citing{n}-{slug}"


def _complete_record(root: Path, *, inclusion: str = "no", outcome: str = "none", amendment: str | None = None) -> Path:
    """A synthetic record that satisfies every check: 80 searches, the citation lists (Yao et al.'s, the
    three of each of the two works citing it, a full-text hit's references), one full-text hit."""
    (root / "screened").mkdir(parents=True)
    (root / record.SEARCH_LOG_MD).write_text(
        f"# Fixture log\n\nScreening depth: {protocol.SCREENING_DEPTH}\n{ORDER_LINE}\nDepth fixed on: {DEPTH_DAY}\n"
        f"Fixed by: {SEARCHER}\n\n## Searches\n\nScreening depth: 7 (a later line does not change the fixed depth)\n"
        "Result order: newest first (nor the fixed order)\n", encoding="utf-8")
    log = []
    for s_index, source in enumerate(protocol.SOURCES):
        for v_index, variant in enumerate(protocol.query_variants()):
            reported = 120 if (s_index, v_index) == (0, 0) else 3
            screened = min(protocol.SCREENING_DEPTH, reported)
            log.append({"source": source, "query_id": variant.query_id, "query_string": variant.text, "date": DAY,
                        "searcher": SEARCHER, "results_reported": reported, "results_screened": screened,
                        "export_file": EXPORT if (s_index, v_index) == (0, 0) else "", "notes": ""})
            items = [(f"Fixture work {source} {variant.query_id} {k}", "exclude") for k in range(screened)]
            if (s_index, v_index) == (0, 0):
                items[0] = (HIT, "full_text")
                items[1] = (items[2][0], "duplicate", items[2][0])
            _screened(root / record.screened_file(source, variant.query_id), items)
    _write_csv(root / record.SEARCH_LOG_CSV, record.SEARCH_LOG_COLUMNS + record.OPTIONAL_COLUMNS, log)
    (root / EXPORT).parent.mkdir()
    (root / EXPORT).write_bytes(b"TY  - JOUR\nTI  - Fixture export\nER  - \n")
    lists = {  # list_id -> (kind, of_work, its items; None: two excluded items)
        "yao-references": (protocol.REFERENCES, protocol.CITATION_SEED, None),
        "yao-cited-by-gs": ("cited_by:Google Scholar", "Fixture citation of the seed", [(c, "exclude") for c in CITING]),
        "yao-cited-by-ss": ("cited_by:Semantic Scholar", protocol.CITATION_SEED, [(c, "duplicate", c) for c in CITING]),
        "hit-references": (protocol.REFERENCES, HIT, None),
        **{_citing_list(n, slug): (kind, work, None) for n, work in enumerate(CITING, start=1)
           for slug, kind in zip(CITING_SLUGS, protocol.CITING_WORK_LISTS)},
    }
    lists = {i: (k, w, items or [(f"Fixture list item {i} {n}", "exclude") for n in (1, 2)])
             for i, (k, w, items) in lists.items()}
    _write_csv(root / record.CITATIONS_LOG_CSV, record.CITATIONS_LOG_COLUMNS + record.OPTIONAL_COLUMNS,
               [{"list_id": i, "kind": k, "of_work": w, "date": DAY, "searcher": SEARCHER, "items_reported": len(items),
                 "items_screened": len(items), "export_file": EXPORT if i == "hit-references" else "", "notes": ""}
                for i, (k, w, items) in lists.items()])
    for list_id, (_, _, items) in lists.items():
        _screened(root / record.citation_screened_file(list_id), items)
    rows = [{"citation": "Fixture citation of the seed" if i == 0 else f"Fixture citation {i}", "known_work": work,
             "what_it_varies": "fixture", "what_it_evaluates": "fixture", "meets_inclusion": "no", "reason": "fixture",
             "exclusion_criterion": protocol.NO_EXCLUSION, "found_in": "Table C.1 'Already known'", "date": DAY,
             "searcher": SEARCHER}
            for i, work in enumerate(protocol.KNOWN_WORKS)]
    rows.append({"citation": HIT, "known_work": "", "what_it_varies": "fixture", "what_it_evaluates": "fixture",
                 "meets_inclusion": inclusion, "reason": "fixture", "exclusion_criterion": protocol.NO_EXCLUSION,
                 "found_in": "Google Scholar Q1-base", "date": DAY, "searcher": SEARCHER})
    _write_csv(root / record.SEARCH_RECORD_CSV, record.SEARCH_RECORD_COLUMNS, rows)
    (root / record.DECISION_JSON).write_text(json.dumps({"outcome": outcome, "amendment": amendment, "date": DAY,
                                                         "searcher": SEARCHER}), encoding="utf-8")
    return root


def test_a_complete_record_passes(tmp_path) -> None:
    root = _complete_record(tmp_path / "search")
    status = search.check_record(root)
    assert status == search.SearchStatus(complete=True, outcome="none", problems=[], amendment=None)
    # Q-search-record-scope: 'none' is a url_or_doi (the source gave neither), here on every rank 1
    assert ",none," in (root / FIRST_FILE).read_text(encoding="utf-8").splitlines()[1]


@pytest.mark.parametrize(("inclusion", "outcome"), [("yes", "included"), ("partial", "partial")])
def test_an_overlapping_work_gives_its_outcome_and_names_its_amendment_by_id(tmp_path, inclusion, outcome) -> None:
    """Table C.1 'Decision': both outcomes are recorded "by amendment"; decision.json names it by the id of
    its analysis/AMENDMENTS.json entry, which SearchStatus carries to the scheduler (Q-search-before-pilot)."""
    status = search.check_record(_complete_record(tmp_path / "s", inclusion=inclusion, outcome=outcome, amendment="A7"))
    assert status == search.SearchStatus(complete=True, outcome=outcome, problems=[], amendment="A7")
    for n, unnamed in enumerate((None, "", "  ")):
        status = search.check_record(_complete_record(tmp_path / f"u{n}", inclusion=inclusion, outcome=outcome,
                                                      amendment=unnamed))
        assert not status.complete and status.outcome == outcome and status.amendment is None
        assert any("name the amendment by the id of its analysis/AMENDMENTS.json entry" in p for p in status.problems)
    status = search.check_record(_complete_record(tmp_path / "b", inclusion=inclusion, outcome=outcome, amendment=" A7"))
    assert not status.complete and any("without surrounding blanks" in p for p in status.problems)


def test_no_overlap_names_no_amendment_and_the_amendment_is_reported_while_incomplete(tmp_path) -> None:
    """Q-search-before-pilot: 'none' is recorded by no amendment (null); the amendment, like the outcome,
    is reported even while another part of the record is incomplete."""
    for n, amendment in enumerate(("A7", "")):
        status = search.check_record(_complete_record(tmp_path / f"n{n}", amendment=amendment))
        assert not status.complete and status.outcome == "none"
        assert any("outcome 'none' is recorded by no amendment; amendment must be null" in p for p in status.problems)
    root = _complete_record(tmp_path / "p", inclusion="partial", outcome="partial", amendment="A7")
    _edit_csv(root / record.SEARCH_LOG_CSV, lambda rows: rows[:-1])
    status = search.check_record(root)
    assert not status.complete and (status.outcome, status.amendment) == ("partial", "A7")


def test_empty_templates_are_incomplete(tmp_path) -> None:
    root = tmp_path / "search"
    root.mkdir()
    (root / record.SEARCH_LOG_MD).write_text("# Log\n\nScreening depth: (to be fixed before the first query)\n", encoding="utf-8")
    _write_csv(root / record.SEARCH_LOG_CSV, record.SEARCH_LOG_COLUMNS + record.OPTIONAL_COLUMNS, [])
    _write_csv(root / record.CITATIONS_LOG_CSV, record.CITATIONS_LOG_COLUMNS + record.OPTIONAL_COLUMNS, [])
    _write_csv(root / record.SEARCH_RECORD_CSV, record.SEARCH_RECORD_COLUMNS, [])
    status = search.check_record(root)
    assert not status.complete and status.outcome is None
    text = "\n".join(status.problems)
    for needle in ("screening depth is not fixed", "result order is not fixed", "Depth fixed on", "Fixed by",
                   "searches not logged: 80 of 80",
                   "Yao et al. (2023) must appear once", "Sootla et al. (2022) must appear once",
                   "Wang et al. (2023, ECAI) must appear once", "the references list of Yao et al. (2023)",
                   "decision.json is missing"):
        assert needle in text, needle


def _edit_csv(path: Path, change) -> None:
    """Rewrites the CSV file with ``change(rows)``; a column a changed row adds is appended to the header."""
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        columns, rows = list(reader.fieldnames or ()), list(reader)
    rows = change(rows)
    columns += [c for c in dict.fromkeys(k for row in rows for k in row) if c not in columns]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(rows)


def _set(rows, index, **values):
    rows[index].update(values)
    return rows


def _change(name: str, index: int, **values):
    """A mutation: row ``index`` of the CSV file ``name`` gets ``values``."""
    return lambda root: _edit_csv(root / name, lambda rows: _set(rows, index, **values))


def _without(name: str, column: str, value: str):
    """A mutation: the rows of ``name`` whose ``column`` is ``value`` are removed."""
    return lambda root: _edit_csv(root / name, lambda rows: [x for x in rows if x[column] != value])


def _decision(**data):
    return lambda root: (root / record.DECISION_JSON).write_text(json.dumps(data), encoding="utf-8")


def _log_md(old: str, new: str):
    """A mutation: the first ``old`` of search_log.md becomes ``new``."""
    def mutate(root: Path) -> None:
        path = root / record.SEARCH_LOG_MD
        text = path.read_text(encoding="utf-8")
        assert old in text, old
        path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return mutate


def _both(*mutations):
    """A mutation applying each of ``mutations`` in turn."""
    return lambda root: [m(root) for m in mutations]


FIRST_FILE = record.screened_file(protocol.SOURCES[0], "Q1-base")  # holds the full-text hit and a duplicate
OTHER_FILE = record.screened_file(protocol.SOURCES[3], "Q4-spacing")  # three excluded results, no duplicate_of column
MUTATIONS = {  # name -> (mutation of a complete record, words of the problem it causes)
    "depth not the protocol's": (_log_md("Screening depth: 50", "Screening depth: 40"), "is not the protocol's 50"),
    # Q-search-depth: the result order is fixed with the depth, as the protocol's
    "result order missing": (_log_md(ORDER_LINE + "\n", ""), "the result order is not fixed"),
    "result order empty": (_log_md(ORDER_LINE, "Result order:"), "the result order is not fixed"),
    "result order not relevance": (_log_md(ORDER_LINE, "Result order: newest first"), "the result order is not fixed"),
    "result order the template's": (_log_md(ORDER_LINE, "Result order: (relevance; fixed before the first query)"),
                                    "the result order is not fixed"),
    # Q-search-depth: a citation list is screened in full, its length as the source shows it logged
    "a citation list screened in part": (_change(record.CITATIONS_LOG_CSV, 3, items_reported=5),
                                         "hit-references: 2 of 5 items screened; citation lists are screened in full"),
    "a citation list's length not a number": (_change(record.CITATIONS_LOG_CSV, 2, items_reported="all"),
                                              "items_reported and items_screened must be whole numbers"),
    # Q-search-record-scope: every hit has its URL or DOI ('none') and its reason
    "a screened result without its reason": (_change(OTHER_FILE, 1, reason=""),
                                             "screened/openreview__Q4-spacing.csv line 3: the reason is missing "
                                             "(Table C.1 'Record'"),
    "a citation list item without its reason": (_change(record.citation_screened_file("citing2-references"), 0, reason=""),
                                                "the reason is missing"),
    "a screened result without its URL or DOI": (_change(OTHER_FILE, 2, url_or_doi=""),
                                                 "screened/openreview__Q4-spacing.csv line 4: url_or_doi is missing"),
    # Q-search-citations: a duplicate names its first screened occurrence, a result that is not a duplicate
    "a duplicate without duplicate_of": (_change(FIRST_FILE, 1, duplicate_of=""),
                                         "a duplicate names the citation of its first screened occurrence in duplicate_of"),
    "a duplicate in a file without duplicate_of": (_change(OTHER_FILE, 2, stage1_decision="duplicate"),
                                                   "a duplicate names the citation of its first screened occurrence"),
    "duplicate_of on a result that is no duplicate": (_change(FIRST_FILE, 4, duplicate_of=HIT),
                                                      "duplicate_of is filled, but the result is not a duplicate"),
    "duplicate_of naming no screened result": (_change(FIRST_FILE, 1, duplicate_of="Fixture work nowhere"),
                                               "is the citation of no screened result that is not a duplicate: 1"),
    "duplicate_of naming a duplicate": (_both(_change(FIRST_FILE, 1, citation="Fixture duplicate only"),
                                              _change(FIRST_FILE, 5, stage1_decision="duplicate",
                                                      duplicate_of="Fixture duplicate only")),
                                        "is the citation of no screened result that is not a duplicate: 1"),
    "a citing work's list missing": (_without(record.CITATIONS_LOG_CSV, "list_id", _citing_list(2, "cited-by-gs")),
                                     "works citing Yao et al. (2023) whose citation lists are not all logged: 1"),
    "depth line after the first section": (lambda r: (r / record.SEARCH_LOG_MD).write_text(
        f"# Log\n\n## Searches\n\nScreening depth: 50\nDepth fixed on: {DEPTH_DAY}\nFixed by: {SEARCHER}\n", encoding="utf-8"),
        "screening depth is not fixed"),
    "depth under a first-line section": (lambda r: (r / record.SEARCH_LOG_MD).write_text(
        f"## Searches\nScreening depth: 50\nDepth fixed on: {DEPTH_DAY}\nFixed by: {SEARCHER}\n", encoding="utf-8"),
        "screening depth is not fixed"),
    "searched before the depth was fixed": (_change(record.SEARCH_LOG_CSV, 5, date="2029-12-31"),
                                            "before the screening depth was fixed"),
    "a search not logged": (lambda r: _edit_csv(r / record.SEARCH_LOG_CSV, lambda rows: rows[:-1]),
                            "searches not logged: 1 of 80"),
    "a search logged twice": (lambda r: _edit_csv(r / record.SEARCH_LOG_CSV, lambda rows: rows + rows[-1:]), "logged twice"),
    "a query string changed": (_change(record.SEARCH_LOG_CSV, 3, query_string="safe RL"), "is not the protocol's"),
    "no searcher": (_change(record.SEARCH_LOG_CSV, 2, searcher=""), "searcher's name is missing"),
    "a bad date": (_change(record.SEARCH_LOG_CSV, 2, date="2 Jan 2030"), "is not YYYY-MM-DD"),
    "screened fewer than the depth allows": (_change(record.SEARCH_LOG_CSV, 0, results_screened=20),
                                             "the depth 50 of 120 reported gives 50"),
    "a screened file short of rows": (lambda r: _edit_csv(r / record.screened_file(protocol.SOURCES[1], "Q2-both"),
                                                         lambda rows: rows[:-1]), "2 screened results, the log says 3"),
    "a screened file missing": (lambda r: r.joinpath(record.screened_file(protocol.SOURCES[2], "Q5-base")).unlink(),
                                "screened/arxiv__Q5-base.csv is missing"),
    "ranks out of order": (_change(FIRST_FILE, 4, rank=9), "ranks must run"),
    "a bad screening decision": (_change(FIRST_FILE, 4, stage1_decision="maybe"), "stage1_decision 'maybe'"),
    "a full-text hit without a record row": (_change(FIRST_FILE, 5, stage1_decision="full_text"),
                                             "works taken to full text with no row: 1"),
    "a record row that is no hit": (_change(FIRST_FILE, 0, stage1_decision="exclude"),
                                    "neither a known work nor a full_text decision"),
    "a known work at full text without its reference list": (  # its row is the record's, its list is not logged
        _change(record.screened_file(protocol.SOURCES[1], "Q2-base"), 0, citation="Fixture citation 1",
                stage1_decision="full_text"), "full-text works whose reference list is not logged: 1"),
    "a full-text hit of a citation list without its reference list": (
        _change(record.citation_screened_file("yao-references"), 1, citation=HIT + " 2", stage1_decision="full_text"),
        "full-text works whose reference list is not logged: 1"),
    "an unknown exclusion criterion": (_change(record.SEARCH_RECORD_CSV, 3, exclusion_criterion="off topic"),
                                       "exclusion_criterion 'off topic' is not one of"),
    "no exclusion criterion": (_change(record.SEARCH_RECORD_CSV, 1, exclusion_criterion=""), "exclusion_criterion not filled"),
    "an export file not in the record": (_change(record.SEARCH_LOG_CSV, 7, export_file="exports/missing.ris"),
                                         "export_file 'exports/missing.ris' is not in the record"),
    "an export file outside the record": (_change(record.CITATIONS_LOG_CSV, 0, export_file="../outside.ris"),
                                          "must be a relative path inside the record"),
    "a NUL byte in a screened file": (lambda r: (r / record.screened_file(protocol.SOURCES[2], "Q3-base")).write_bytes(
        (r / record.screened_file(protocol.SOURCES[2], "Q3-base")).read_bytes().replace(b"fixture", b"fix\x00ture", 1)),
        "screened/arxiv__Q3-base.csv is not a readable CSV file"),
    "a log that is not UTF-8": (lambda r: (r / record.CITATIONS_LOG_CSV).write_bytes(b"list_id,\xff\n"),
                                "citations_log.csv is not UTF-8 text"),
    "decision.json a directory": (lambda r: ((r / record.DECISION_JSON).unlink(), (r / record.DECISION_JSON).mkdir()),
                                  "decision.json cannot be read (IsADirectoryError"),
    "a record field empty": (_change(record.SEARCH_RECORD_CSV, 3, what_it_evaluates=""), "what_it_evaluates not filled"),
    "a known work missing": (_change(record.SEARCH_RECORD_CSV, 1, known_work=""), "Sootla et al. (2022) must appear once"),
    "an unknown known work": (_change(record.SEARCH_RECORD_CSV, 2, known_work="Wang et al. (2023)"), "is not one of"),
    "a bad classification": (_change(record.SEARCH_RECORD_CSV, 3, meets_inclusion="maybe"), "meets_inclusion 'maybe'"),
    "a seed list missing": (_without(record.CITATIONS_LOG_CSV, "list_id", "yao-cited-by-ss"),
                            "cited_by:Semantic Scholar list of Yao et al. (2023) is not logged"),
    "a hit's references missing": (_without(record.CITATIONS_LOG_CSV, "list_id", "hit-references"),
                                   "full-text works whose reference list is not logged: 1"),
    "a bad list kind": (_change(record.CITATIONS_LOG_CSV, 3, kind="cited_by:arXiv"), "kind 'cited_by:arXiv'"),
    "a record column missing": (lambda r: (r / record.SEARCH_RECORD_CSV).write_text("citation,known_work\n", encoding="utf-8"),
                                "must hold the columns"),
    "decision missing": (lambda r: (r / record.DECISION_JSON).unlink(), "decision.json is missing"),
    "decision not the record's": (_decision(outcome="partial", amendment="x", date=DAY, searcher=SEARCHER),
                                  "the record gives 'none'"),
    "decision before the last search": (_decision(outcome="none", amendment=None, date=DEPTH_DAY, searcher=SEARCHER),
                                        "before the last search"),
    "decision before the last classification": (_change(record.SEARCH_RECORD_CSV, 3, date="2031-06-01"),
                                                "before the last classification of search_record.csv on 2031-06-01"),
    "decision with an extra field": (_decision(outcome="none", amendment=None, date=DAY, searcher=SEARCHER, novelty="x"),
                                     "exactly the fields"),
    "decision 'none' with an amendment": (_decision(outcome="none", amendment="A7", date=DAY, searcher=SEARCHER),
                                          "amendment must be null, not 'A7'"),
    "an amendment that is no text": (_decision(outcome="none", amendment=7, date=DAY, searcher=SEARCHER),
                                     "amendment must be text or null"),
}


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_each_missing_or_inconsistent_item_makes_the_record_incomplete(tmp_path, name) -> None:
    root = _complete_record(tmp_path / "search")
    mutate, needle = MUTATIONS[name]
    mutate(root)
    status = search.check_record(root)
    assert not status.complete
    assert any(needle in p for p in status.problems), status.problems


def test_an_included_work_falls_under_no_exclusion_clause(tmp_path) -> None:
    root = _complete_record(tmp_path / "s", inclusion="yes", outcome="included")
    _change(record.SEARCH_RECORD_CSV, 3, exclusion_criterion="seen_budgets_only")(root)
    status = search.check_record(root)
    assert not status.complete and status.outcome == "included"
    assert any("falls under the exclusion clause 'seen_budgets_only'" in p for p in status.problems)
    _change(record.SEARCH_RECORD_CSV, 3, meets_inclusion="partial")(root)  # a partial overlap may fall under one
    _decision(outcome="partial", amendment="A2", date=DAY, searcher=SEARCHER)(root)
    assert search.check_record(root) == search.SearchStatus(complete=True, outcome="partial", problems=[], amendment="A2")


def test_a_known_work_at_full_text_needs_its_reference_list_and_the_seed_has_it(tmp_path) -> None:
    """Q-search-citations (README step 4): the reference list of every work taken to full text, a Table
    C.1 known work included; Yao et al.'s is the seed's references list, already required."""
    root = _complete_record(tmp_path / "s")
    second = record.screened_file(protocol.SOURCES[1], "Q2-base")
    _change(second, 0, citation="Fixture citation of the seed", stage1_decision="full_text")(root)  # Yao et al.'s row
    assert search.check_record(root).complete
    _change(second, 1, citation="Fixture citation 2", stage1_decision="full_text")(root)  # Wang et al.'s row
    status = search.check_record(root)
    assert not status.complete and any("Fixture citation 2" in p and "reference list" in p for p in status.problems)
    _edit_csv(root / record.CITATIONS_LOG_CSV, lambda rows: rows + [
        {**rows[0], "list_id": "wang-references", "kind": protocol.REFERENCES, "of_work": "Fixture citation 2"}])
    _screened(root / record.citation_screened_file("wang-references"), [("Fixture list item", "exclude")] * 2)
    assert search.check_record(root) == search.SearchStatus(complete=True, outcome="none", problems=[])


def test_every_work_citing_the_seed_needs_its_three_citation_lists(tmp_path) -> None:
    """Q-search-citations, item (3): every work of Yao et al.'s two "cited by" lists (here two works,
    excluded at title and abstract in the Google Scholar list and duplicates in the Semantic Scholar one)
    has its references and both "cited by" lists; the record is incomplete until each has all three."""
    root = _complete_record(tmp_path / "s")
    log = root / record.CITATIONS_LOG_CSV
    full = log.read_text(encoding="utf-8")

    def logged(keep) -> list[str]:  # the problems of item (3) when citations_log.csv keeps the lists ``keep`` accepts
        log.write_text(full, encoding="utf-8")
        _edit_csv(log, lambda rows: [r for r in rows if keep(r["list_id"])])
        return [p for p in search.check_record(root).problems if "works citing Yao et al. (2023)" in p]

    all_three = "(references, cited_by:Google Scholar, cited_by:Semantic Scholar)"
    [problem] = logged(lambda i: not i.startswith("citing"))
    assert "not all logged: 2 " in problem and all(f"{work} {all_three}" in problem for work in CITING)
    [problem] = logged(lambda i: not i.startswith("citing2"))
    assert "not all logged: 1 " in problem and f"{CITING[1]} {all_three}" in problem and CITING[0] not in problem
    [problem] = logged(lambda i: i != _citing_list(2, "cited-by-ss"))
    assert f"{CITING[1]} (cited_by:Semantic Scholar)" in problem
    assert logged(lambda i: True) == [] and search.check_record(root).complete


def test_a_citing_work_first_screened_in_a_database_search_still_needs_its_lists(tmp_path) -> None:
    """Q-search-citations: an item of Yao et al.'s "cited by" lists already screened in a database search
    is a duplicate there and stands for the work its duplicate_of names, which owes the three lists."""
    root = _complete_record(tmp_path / "s")
    elsewhere = f"Fixture work {protocol.SOURCES[1]} Q2-base 1"  # an excluded result of a database search
    for list_id in ("yao-cited-by-gs", "yao-cited-by-ss"):
        _change(record.citation_screened_file(list_id), 1, stage1_decision="duplicate", duplicate_of=elsewhere)(root)
    status = search.check_record(root)
    assert not status.complete
    assert any("works citing Yao et al. (2023) whose citation lists are not all logged: 1 " in p
               and f"{elsewhere} (references, cited_by:Google Scholar, cited_by:Semantic Scholar)" in p
               for p in status.problems), status.problems
    _edit_csv(root / record.CITATIONS_LOG_CSV,
              lambda rows: [{**r, "of_work": elsewhere} if r["of_work"] == CITING[1] else r for r in rows])
    assert search.check_record(root) == search.SearchStatus(complete=True, outcome="none", problems=[])


def test_google_scholar_refusing_a_citing_works_list_is_noted_in_its_semantic_scholar_row(tmp_path) -> None:
    """Q-search-citations: when Google Scholar refuses a citing work's "cited by" list, its Semantic
    Scholar list alone is screened, that row's notes saying so; the seed's lists have no such fallback."""
    root = _complete_record(tmp_path / "s")
    log = root / record.CITATIONS_LOG_CSV
    refused = f"{protocol.REFUSAL_NOTE} access on {DAY} (fixture)"

    def noted(list_id: str):
        return lambda rows: [{**r, "notes": refused if r["list_id"] == list_id else ""} for r in rows]

    _without(record.CITATIONS_LOG_CSV, "list_id", _citing_list(1, "cited-by-gs"))(root)
    status = search.check_record(root)
    assert not status.complete and any(f"{CITING[0]} (cited_by:Google Scholar)" in p for p in status.problems)
    _edit_csv(log, noted(_citing_list(1, "references")))  # the refusal noted on another row
    assert not search.check_record(root).complete
    _edit_csv(log, noted(_citing_list(1, "cited-by-ss")))
    assert search.check_record(root) == search.SearchStatus(complete=True, outcome="none", problems=[])
    _without(record.CITATIONS_LOG_CSV, "list_id", _citing_list(1, "cited-by-ss"))(root)  # its Semantic Scholar list is owed
    assert any(f"{CITING[0]} (cited_by:Google Scholar, cited_by:Semantic Scholar)" in p
               for p in search.check_record(root).problems)
    root = _complete_record(tmp_path / "seed")
    _without(record.CITATIONS_LOG_CSV, "list_id", "yao-cited-by-gs")(root)
    _edit_csv(root / record.CITATIONS_LOG_CSV, noted("yao-cited-by-ss"))
    assert any("the cited_by:Google Scholar list of Yao et al. (2023) is not logged" in p
               for p in search.check_record(root).problems)


def test_a_citing_work_taken_to_full_text_needs_its_row_and_one_references_list(tmp_path) -> None:
    """Q-search-citations, items (3) and (4): a citing work taken to full text has its record row, and its
    references list of item (3) is the reference list item (4) asks of a full-text work."""
    root = _complete_record(tmp_path / "s")
    _change(record.citation_screened_file("yao-cited-by-gs"), 0, stage1_decision="full_text")(root)
    problems = search.check_record(root).problems
    assert any("works taken to full text with no row: 1" in p for p in problems)
    assert not any("reference list is not logged" in p for p in problems)
    _edit_csv(root / record.SEARCH_RECORD_CSV, lambda rows: rows + [
        {**rows[-1], "citation": CITING[0], "found_in": "cited_by:Google Scholar of Yao et al. (2023)"}])
    assert search.check_record(root) == search.SearchStatus(complete=True, outcome="none", problems=[])
    _without(record.CITATIONS_LOG_CSV, "list_id", _citing_list(1, "references"))(root)
    problems = search.check_record(root).problems
    assert any("full-text works whose reference list is not logged: 1" in p and CITING[0] in p for p in problems)
    assert any(f"{CITING[0]} (references)" in p for p in problems)


def test_problems_name_the_physical_line_of_the_file(tmp_path) -> None:
    """A row's line is the file's line, blank lines and quoted line breaks counted, and a skipped
    malformed row does not shift the lines of the rows after it."""
    root = _complete_record(tmp_path / "search")
    path = root / record.SEARCH_RECORD_CSV
    lines = path.read_text(encoding="utf-8").splitlines()
    bad = lines[-1].replace(",no,", ",maybe,", 1)
    assert bad != lines[-1]
    path.write_text("\n".join([lines[0], "a short row", "", bad]) + "\n", encoding="utf-8")
    problems = search.check_record(root).problems
    assert "search_record.csv line 2: the row does not have the header's 10 fields" in problems
    assert any(p.startswith("search_record.csv line 4: meets_inclusion 'maybe'") for p in problems), problems
    path.write_text("\n".join([lines[0], '"a citation', 'over two lines",' + bad.split(",", 1)[1]]) + "\n",
                    encoding="utf-8")
    assert any(p.startswith("search_record.csv line 3: meets_inclusion 'maybe'")
               for p in search.check_record(root).problems)


def test_a_malformed_record_is_listed_by_the_command_line_not_raised(tmp_path, capsys) -> None:
    root = _complete_record(tmp_path / "search")
    MUTATIONS["a NUL byte in a screened file"][0](root)
    MUTATIONS["decision.json a directory"][0](root)
    assert search_main(["check", "--root", str(root)]) == 1
    out = capsys.readouterr().out
    assert "complete: False; outcome: None" in out
    assert "- screened/arxiv__Q3-base.csv is not a readable CSV file" in out and "- decision.json cannot be read" in out


def test_the_reader_is_given_each_file_under_the_root(tmp_path) -> None:
    """The scheduler reads the committed files (pilot/scheduler.py): a reader keyed by repository paths."""
    disk = _complete_record(tmp_path / "search")
    files = {f"studyb/search/{p.relative_to(disk).as_posix()}": p.read_bytes() for p in disk.rglob("*") if p.is_file()}
    asked: list[str] = []

    def committed(path: str) -> bytes | None:
        asked.append(path)
        return files.get(path)

    assert search.check_record("studyb/search", read=committed) == search.check_record(disk)
    assert asked and all(p.startswith("studyb/search/") for p in asked)
    assert "studyb/search/screened/openreview__Q5-both.csv" in asked
    del files["studyb/search/decision.json"]
    status = search.check_record("studyb/search", read=committed)
    assert not status.complete and status.outcome is None and any("decision.json is missing" in p for p in status.problems)


def test_the_command_line_checks_and_lists(tmp_path, capsys) -> None:
    root = _complete_record(tmp_path / "search")
    assert search_main(["check", "--root", str(root)]) == 0
    assert "complete: True; outcome: none" in capsys.readouterr().out
    (root / record.DECISION_JSON).unlink()
    assert search_main(["check", "--root", str(root)]) == 1
    assert search_main(["queries"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert "screening depth: the first 50 results of each search, in each source's relevance order (Q-search-depth)" in lines
    searches = [line for line in lines if "\t" in line]
    assert len(searches) == len(protocol.SOURCES) * 20
    assert searches[0].split("\t") == [protocol.SOURCES[0], "Q1-base", R.SEARCH_QUERIES[0], "screened/google_scholar__Q1-base.csv"]

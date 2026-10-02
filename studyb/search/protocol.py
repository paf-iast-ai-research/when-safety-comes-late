"""The protocol of Study B's targeted search (Appendix C, Table C.1), as code.

Owner: Study B and literature (Role 5, Hamza Nisar). This module holds the procedure
only: the sources, the queries and their variants, the exclusion clauses, the works already known,
the citation lists, the screening depth and the result order. It holds no search result, citation,
date or hit: the search is the Study B owner's act (First Tasks, Role 5), recorded by hand in the
files beside this module and checked by ``studyb.search.record.check_record``.

What the pre-registration fixes (prereg/Preregistration.pdf, Appendix C, Table C.1, p. 24):

* "Sources": "Google Scholar, Semantic Scholar, arXiv, OpenReview, and the citation lists of Yao et
  al. (2023) and of every paper that cites it." (``R.SEARCH_SOURCES``, ``R.SEARCH_CITATION_SEED``)
* "Queries": the five queries of ``R.SEARCH_QUERIES``, "each with and without "number of
  thresholds" and "spacing"" (``R.SEARCH_ADDED_PHRASES``).
* "Exclusion": "Works that vary the algorithm, that evaluate only seen budgets, or that vary several
  cost functions rather than levels of one." (``EXCLUSION_CRITERIA``, one label per clause)
* "Already known": Yao et al. (2023), Sootla et al. (2022) and Wang et al. (2023, ECAI).
* "Record": "For every hit: citation, what it varies, what it evaluates, and whether it meets the
  inclusion criterion; the date of the search; the searcher's name."
* "Decision": "Any included work withdraws Study B by amendment; a partially overlapping work narrows
  Study B's claims by amendment, recorded before the first run."

What it leaves open, answered in Table 9.1 and implemented here (documented questions, not PENDING
keys, each raised as First Tasks, Role 5 step 1 asks of the screening depth, "as a pre-registration
question so it enters the amendment log"): the screening depth and the result order
(Q-search-depth), the query variants (Q-search-queries), the reach of "the citation lists"
(Q-search-citations) and the meaning of "every hit" (Q-search-record-scope, see
``studyb.search.record``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import MappingProxyType

from configs import registered as R

SOURCES: tuple[str, ...] = tuple(R.SEARCH_SOURCES)  # Table C.1 "Sources": the four databases
QUERIES: tuple[str, ...] = tuple(R.SEARCH_QUERIES)  # Table C.1 "Queries", verbatim, in the table's order
ADDED_PHRASES: tuple[str, ...] = tuple(R.SEARCH_ADDED_PHRASES)  # '"number of thresholds"', '"spacing"'
CITATION_SEED: str = R.SEARCH_CITATION_SEED  # "Yao et al. (2023)"

# How many results of each query variant in each source are screened at title and abstract. First
# Tasks, Role 5 step 1: "for example the first 50 results of each query in each source", fixed before
# the first query and written at the top of the search log (studyb/search/search_log.md). A citation
# list has no depth limit: it is screened in full (Q-search-citations).
SCREENING_DEPTH: int = 50  # answered in Table 9.1 (Q-search-depth)
# The order those results are taken in: each source's relevance order (in arXiv, 'Relevance', not its
# default newest-first; a source that offers no relevance order: its default order, named in the
# search log's notes). A 50-deep cut means something only on a relevance-ranked list, so the order is
# fixed with the depth, before the first query (the 'Result order:' line of search_log.md).
RESULT_ORDER: str = "relevance"  # answered in Table 9.1 (Q-search-depth)

# "each with and without "number of thresholds" and "spacing"" admits two, three or four variants of
# a query; all four are searched (the superset satisfies every reading): 5 queries x 4 variants = 20
# strings, each used verbatim (quotes kept) in each of the four sources, 80 searches; a source that
# needs its own syntax to express a string is described in the search log's notes.
VARIANT_PHRASES = MappingProxyType({  # answered in Table 9.1 (Q-search-queries)
    "base": (),
    "thresholds": (ADDED_PHRASES[0],),
    "spacing": (ADDED_PHRASES[1],),
    "both": ADDED_PHRASES,
})

# Table C.1 "Already known", in the table's order. They enter the record first (First Tasks, Role 5
# step 3), each with its classification. The first is the citation seed of R.SEARCH_CITATION_SEED.
KNOWN_WORKS: tuple[str, ...] = (CITATION_SEED, "Sootla et al. (2022)", "Wang et al. (2023, ECAI)")

# "the citation lists of Yao et al. (2023) and of every paper that cites it" (answered in Table 9.1,
# Q-search-citations): (1) Yao et al.'s reference list; (2) its "cited by" lists in Google Scholar and
# Semantic Scholar (First Tasks, Role 5 step 5), which list every paper that cites it; (3) for every
# work in the union of those two lists (deduplicated, as of the search date), its reference list and
# its two "cited by" lists; (4) the reference list of every other work that reaches the full-text
# stage in a screened file, a Table C.1 known work included. Every list is screened in full at title
# and abstract, with no depth limit.
REFERENCES = "references"
CITED_BY_SOURCES: tuple[str, ...] = ("Google Scholar", "Semantic Scholar")  # answered in Table 9.1 (Q-search-citations)
CITATION_LIST_KINDS: tuple[str, ...] = (REFERENCES, *(f"cited_by:{s}" for s in CITED_BY_SOURCES))
# The lists required of the seed (Yao et al., 2023), items (1) and (2): every kind. A name of its own,
# apart from the kinds a list may be, so the record check states the requirement, not the vocabulary.
SEED_CITATION_LISTS: tuple[str, ...] = CITATION_LIST_KINDS
# The lists required of every work in the seed's "cited by" lists, item (3): every kind as well.
CITING_WORK_LISTS: tuple[str, ...] = CITATION_LIST_KINDS
# Item (3) when Google Scholar refuses access to a citing work's "cited by" list: its Semantic Scholar
# list alone is screened, and that list's citations_log.csv row records the refusal in its notes with
# these words. The seed's own lists have no such fallback.
REFUSABLE_LIST = f"cited_by:{CITED_BY_SOURCES[0]}"  # Google Scholar's "cited by" list
REFUSAL_NOTED_ON = f"cited_by:{CITED_BY_SOURCES[1]}"  # the Semantic Scholar row whose notes record it
REFUSAL_NOTE = "Google Scholar refused"

# Table C.1 "Exclusion": "Works that vary the algorithm, that evaluate only seen budgets, or that vary
# several cost functions rather than levels of one." One label per clause, in the table's order. The
# record's exclusion_criterion column (Study B spec section 8.1) names the clause a full-text work
# falls under, or NO_EXCLUSION when it falls under none (it meets the inclusion criterion, or fails it
# without falling under any clause), so a rejected work records which clause rejected it.
EXCLUSION_CRITERIA = MappingProxyType({
    "varies_algorithm": "vary the algorithm",
    "seen_budgets_only": "evaluate only seen budgets",
    "several_cost_functions": "vary several cost functions rather than levels of one",
})
NO_EXCLUSION = "none"

INCLUSION_VALUES: tuple[str, ...] = ("yes", "partial", "no")  # First Tasks, Role 5 step 2: "(yes, partial or no)"
# Table C.1 "Decision": no overlap, a partial overlap, an included work
OUTCOMES: tuple[str, ...] = ("none", "partial", "included")


@dataclass(frozen=True)
class QueryVariant:
    """One query string to run in every source: query ``number`` of Table C.1 in one variant."""

    query_id: str  # e.g. "Q1-base", "Q3-both": the id the search log and the screened files use
    number: int  # 1 to 5, Table C.1's order
    variant: str  # a key of VARIANT_PHRASES
    text: str  # the exact string searched


def query_variants() -> tuple[QueryVariant, ...]:
    """Every query variant of the protocol, query by query: the query, then with each added phrase,
    then with both (Q-search-queries), 20 in all."""
    return tuple(
        QueryVariant(f"Q{number}-{variant}", number, variant, " ".join((query, *phrases)))
        for number, query in enumerate(QUERIES, start=1)
        for variant, phrases in VARIANT_PHRASES.items()
    )


def source_slug(source: str) -> str:
    """A file-name form of a source: 'Google Scholar' -> 'google_scholar'."""
    slug = re.sub(r"[^a-z0-9]+", "_", source.lower()).strip("_")
    if not slug:
        raise ValueError(f"source {source!r} has no file-name form")
    return slug

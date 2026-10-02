"""Every pre-registration quotation in Study B's code is verbatim.

The pipeline core's own check (tests/test_core_quotes.py: ``cited_quotes``, ``missing_pieces`` and
``unlabelled_continuation_starts``) applied to the Python and Markdown files of studyb/ and Study
B's tests: each double-quoted passage that follows a citation of the pre-registration must appear in
prereg/Preregistration.docx, and prose that says where a continuation starts must name
Q-continuations (the starting checkpoint is its answer, docs/DECISIONS.md / Table 9.1, not registered
text). A quotation cited with the page it is on ("(p. N): "..."") must also begin on that page of
prereg/Preregistration.pdf.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
# The opening words of pages 5 to 24 of prereg/Preregistration.pdf (read off the PDF; the .docx holds no
# page numbers), as the .docx's text holds them: each occurs once there, in page order. Where a page
# opens in a table the words are its first cell text that the .docx does not repeat elsewhere.
PAGE_OPENINGS = {
    5: "H0. The null hypothesis of Study A",
    6: "G4. Coverage speeds adaptation",
    7: "Evaluation cost of a checkpoint Mean episodic cost",
    8: "Table 2.2. The robustness battery",
    9: "Table 2.4. Interventions and controls",
    10: "Unseen budgets {5, 15, 30, 45}",
    11: "Dormant-neuron score. Mean absolute activation",
    12: "Networks Actor, reward critic and cost critic",
    13: "3.3 Study B: arms and evaluation",
    14: "the seed-to-seed spread, but it is the reason",
    15: "4.1.1 The two estimands",
    16: "equation (10), zero-shot and after each few-shot horizon",
    17: "Surplus compute goes to seeds first",
    18: "G1, feasibility (Study A).",
    19: "mechanism test with a data control",
    20: "Quantity Why only the pilot can supply it",
    21: "Registration commit hash (entered at commit)",
    22: "Cohen, J. (1988).",
    23: "Stooke, A., Achiam, J., & Abbeel, P. (2020).",
    24: "started; finished; wall_clock_hours",
}
# "(p. N):" then, within a few words, the quotation it cites.
PAGE_CITATION = re.compile(r'\(p\. (\d+)\):([^"\n]{0,40})"([^"]+)"')
FILES = sorted(str(p.relative_to(REPO)) for p in [*(REPO / "studyb").rglob("*.py"),
                                                    *(REPO / "studyb").rglob("*.md"),
                                                    *(REPO / "tests").glob("test_studyb_*.py")])


def _source(path: str) -> str:
    """The file's text as the checker reads it: Python as it is, Markdown as one docstring (all prose)."""
    text = (REPO / path).read_text(encoding="utf-8")
    if not path.endswith(".md"):
        return text
    assert '"""' not in text, path
    return 'r"""\n' + text + '\n"""\n'


def _checker():
    sys.path.insert(0, str(Path(__file__).parent))
    import test_core_quotes

    return test_core_quotes


@pytest.mark.parametrize("path", FILES)
def test_every_cited_quotation_in_study_b_is_in_the_pre_registration(path) -> None:
    check = _checker()
    source = _source(path)
    bad = {q: check.missing_pieces(q) for q in check.cited_quotes(source)}
    assert not {q: m for q, m in bad.items() if m}
    assert not check.unlabelled_continuation_starts(source)


def test_the_check_reads_study_bs_quotations() -> None:
    check = _checker()
    assert len(FILES) >= 10
    assert sum(len(check.cited_quotes(_source(p))) for p in FILES) >= 25
    assert any(check.cited_quotes(_source(p)) for p in FILES if p.endswith(".md"))  # the Markdown is read too


def _page_starts() -> dict[int, int]:
    check = _checker()
    text = check._prereg()[0]
    return {page: text.index(check._normalise(words)) for page, words in PAGE_OPENINGS.items()}


def misplaced_page_citations(source: str) -> list[str]:
    """The quotations of ``source``'s prose cited "(p. N)" that do not begin on page N."""
    check = _checker()
    text = check._prereg()[0]
    starts = _page_starts()
    bad = []
    for prose in check._prose(source):
        for match in PAGE_CITATION.finditer(prose):
            page = int(match.group(1))
            # the quotation's first words: its first piece that is not empty (it may open with an ellipsis)
            pieces = (check._normalise(piece).strip(" .,;:") for piece in re.split(r"\.\.\.|…", match.group(3)))
            first = next((piece for piece in pieces if piece), "")
            if not first:
                bad.append(f"p. {page} (no quotable words): {match.group(3)}")
                continue
            if page not in starts:
                bad.append(f"p. {page} (no page opening known): {first}")
                continue
            end = starts.get(page + 1, len(text))
            found = [m.start() for m in re.finditer(re.escape(first), text)]
            if not any(starts[page] <= at < end for at in found):
                pages = sorted({max(p for p, s in starts.items() if s <= at) for at in found if at >= starts[min(starts)]})
                bad.append(f"p. {page}, but on {pages or 'no mapped page'}: {first}")
    return bad


def test_the_page_openings_are_in_page_order() -> None:
    check = _checker()
    text = check._prereg()[0]
    for words in PAGE_OPENINGS.values():
        assert text.count(check._normalise(words)) == 1, words
    starts = _page_starts()
    assert [starts[p] for p in sorted(starts)] == sorted(starts.values())


@pytest.mark.parametrize("path", FILES)
def test_every_quotation_cited_with_a_page_is_on_that_page(path) -> None:
    assert not misplaced_page_citations(_source(path))


def test_the_page_check_reads_the_citations_and_catches_a_wrong_page() -> None:
    sources = [_source(p) for p in FILES]
    assert sum(len(PAGE_CITATION.findall(t)) for s in sources for t in _checker()._prose(s)) >= 6
    # the page a review found wrong in round 1: Part 5.1's pairing sentence is on p. 15, not p. 16
    wrong = '''"""* Part 5.1 (p. 16): "seed k of one arm and seed k of the other share their network initialisation
  and layout sequence and are treated as a pair"."""'''
    assert misplaced_page_citations(wrong) == [
        "p. 16, but on [15]: seed k of one arm and seed k of the other share their network initialisation and layout "
        "sequence and are treated as a pair"]
    assert misplaced_page_citations(wrong.replace("(p. 16)", "(p. 15)")) == []
    # a quotation that opens with an ellipsis is placed by its first words, not passed vacuously
    elided = wrong.replace(': "seed k of one arm', ': "... seed k of one arm')
    assert misplaced_page_citations(elided) == misplaced_page_citations(wrong)
    assert misplaced_page_citations('"""Part 5.1 (p. 15): "..."."""') == ["p. 15 (no quotable words): ..."]

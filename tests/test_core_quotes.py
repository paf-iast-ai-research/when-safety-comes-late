"""A quotation of the pre-registration in the pipeline core (``FILES``) is verbatim.

Every double-quoted passage that follows a pre-registration citation (Part, Table, Appendix, box,
equation, "How to Read This Document", a cut of Part 6.1) within a docstring or comment of the files
in ``FILES`` must appear in the pre-registration, each piece of an elision ("...") on its own: the
first passage within 100 characters of the citation, and every further one up to the end of that
sentence (or the next citation), so a quoted title followed by the quotation itself (``Part 6.1
"Cut order ..." (PDF p. 18): "..."``) has both checked. The citation is the marker: a quotation of
a design document, of code or of OmniSafe carries none. Code spans (backticks) are not prose.

The text is prereg/Preregistration.docx, read two ways: as its paragraphs, and with Word's decimal
list numbers written out ("1. ", "2. ", ...) as prereg/Preregistration.pdf prints them (the .docx
holds them as list properties, not text). Typographic quotes, dashes, ×, ≤, ≥, whitespace and case
are normalised on both sides. Passages of fewer than three words (terms such as "at onset") and
passages holding a format field ({...}) are not checked.
"""

from __future__ import annotations

import ast
import html
import io
import re
import tokenize
import zipfile
from functools import lru_cache
from pathlib import Path

import pytest

from configs import registered as R

REPO = Path(__file__).resolve().parents[1]
DOCX = REPO / "prereg" / "Preregistration.docx"
FILES = ("pilot/launch.py", "pilot/rundir.py", "pilot/algorithms.py", "pilot/contracts.py", "pilot/dependencies.py",
         "pilot/manifest.py", "pilot/provenance.py", "pilot/ledger_writer.py", "pilot/supplement.py")
CITATION = re.compile(
    r"Part \d(?:\.\d+)*|Table [A-Z]?\d+(?:\.\d+)*|Appendix [A-Z]|box H\d|[Ee]quations? \(?\d+|eqs?\. \(?\d+"
    r"|How to Read This Document|Cut \d"
)
FIRST_GAP = 100  # at most this many characters between a citation and its first quotation
# The end of a sentence (or of a paragraph or list item) between two quotations ends the citation's reach.
BOUNDARY = re.compile(r"[.!?]\s+(?=[A-Z])|\n\s*\n|\n\s*(?:[*-]|\d+\.)\s")
CODE = re.compile(r"``.*?``|`[^`\n]*`", re.S)
_TYPOGRAPHY = {"’": "'", "‘": "'", "“": '"', "”": '"', "−": "-", "–": "-", "—": "-",
               "×": "x", "≤": "<=", "≥": ">="}
_NUMBERED = re.compile(r'<w:numPr><w:ilvl w:val="0"/><w:numId w:val="(\d+)"/></w:numPr>')


def _normalise(text: str) -> str:
    for old, new in _TYPOGRAPHY.items():
        text = text.replace(old, new)
    return re.sub(r"\s+", " ", text).strip().lower()


def _text(xml: str) -> str:
    return _normalise(html.unescape(re.sub(r"<[^>]+>", "", re.sub(r"</w:p>", "\n", xml))))


def _decimal_lists(numbering: str) -> set[str]:
    """The numIds whose first level is numbered "1.", "2.", ... (numbering.xml)."""
    decimal = {m.group(1) for m in re.finditer(r'<w:abstractNum [^>]*w:abstractNumId="(\d+)".*?</w:abstractNum>',
                                               numbering, re.S)
               if re.search(r'<w:lvl w:ilvl="0".*?<w:numFmt w:val="decimal"/>.*?<w:lvlText w:val="%1\."/>',
                            m.group(0), re.S)}
    return {num for num, abstract in re.findall(r'<w:num w:numId="(\d+)"[^>]*>.*?<w:abstractNumId w:val="(\d+)"/>',
                                                numbering, re.S) if abstract in decimal}


@lru_cache(maxsize=1)
def _prereg() -> tuple[str, str]:
    """The pre-registration's text: plain, and with its decimal list numbers written out."""
    with zipfile.ZipFile(DOCX) as docx:
        xml = docx.read("word/document.xml").decode("utf-8")
        decimal = _decimal_lists(docx.read("word/numbering.xml").decode("utf-8"))
    counters: dict[str, int] = {}

    def number(match: re.Match) -> str:
        if match.group(1) not in decimal:
            return match.group(0)
        counters[match.group(1)] = counters.get(match.group(1), 0) + 1
        return f"{match.group(0)}<w:t>{counters[match.group(1)]}. </w:t>"

    return _text(xml), _text(_NUMBERED.sub(number, xml))


def _prose(source: str) -> list[str]:
    """Docstrings, and runs of consecutive comment lines joined into one text."""
    out = [doc for node in ast.walk(ast.parse(source))
           if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
           and (doc := ast.get_docstring(node, clean=True))]
    run: list[str] = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            run.append(token.string.lstrip("#").strip())
        elif token.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT) and run:
            out.append(" ".join(run))
            run = []
    return out + ([" ".join(run)] if run else [])


def cited_quotes(source: str) -> list[str]:
    """Every quoted passage (three words or more) within a pre-registration citation's reach."""
    quotes: list[str] = []
    for text in _prose(source):
        text = CODE.sub(lambda m: " " * len(m.group()), text)  # blanked, so offsets and pairing hold
        marks = [i for i, c in enumerate(text) if c == '"']
        spans = list(zip(marks[0::2], marks[1::2]))
        taken: set[tuple[int, int]] = set()
        for citation in CITATION.finditer(text):
            pos, first = citation.end(), True
            for a, b in spans:
                if a < citation.start() < b:  # a citation inside a quoted title: its quotation follows the title
                    pos = b + 1
            for a, b in spans:
                if a < pos:
                    continue
                gap = text[pos:a]
                if CITATION.search(gap) or BOUNDARY.search(gap) or (first and len(gap) > FIRST_GAP):
                    break  # a later citation, or another sentence, or too far: this citation's reach ends
                quote = re.sub(r"\s+", " ", text[a + 1:b]).strip()
                if (a, b) not in taken and "{" not in quote and len(quote.split()) >= 3:  # single terms are not quotations
                    quotes.append(quote)
                    taken.add((a, b))
                pos, first = b + 1, False
    return quotes


def missing_pieces(quote: str) -> list[str]:
    """The pieces of ``quote`` (split at elisions) that neither form of the pre-registration holds."""
    pieces = [p.strip(" .,;:") for p in re.split(r"\.\.\.|…", quote)]
    return [p for p in pieces if p and not any(_normalise(p) in text for text in _prereg())]


@pytest.mark.parametrize("path", FILES)
def test_every_cited_quotation_is_in_the_pre_registration(path) -> None:
    bad = {q: missing_pieces(q) for q in cited_quotes((REPO / path).read_text(encoding="utf-8"))}
    assert not {q: m for q, m in bad.items() if m}


def test_the_check_finds_the_quotations_and_catches_a_misquotation() -> None:
    assert sum(len(cited_quotes((REPO / p).read_text(encoding="utf-8"))) for p in FILES) >= 20
    # two misquotations: words the text does not contain
    wrong = '''"""Layout (Part 3.4: "Each run writes ... its checkpoints ... and its provenance").

    Table 2.3 logs the plasticity metrics "Every 200,000 steps ... and at onset" for Study A.
    """'''
    assert [missing_pieces(q) for q in cited_quotes(wrong)] == [
        ["Each run writes", "its checkpoints", "and its provenance"], ["and at onset"]]
    right = '''"""Part 3.4: "Each run saves a checkpoint every 200,000 steps, at onset, and at the end of training"."""'''
    assert cited_quotes(right) and not missing_pieces(cited_quotes(right)[0])


def test_every_quotation_of_a_cited_sentence_is_checked() -> None:
    """Every quotation within a citation's reach is checked, not only the first: a quoted title must not
    hide the quotation that follows it."""
    titled = '''"""Part 3.4 "Checkpoints and logging" (PDF p. 5): "Each run saves a checkpoint every 100,000 steps"."""'''
    assert cited_quotes(titled) == ["Checkpoints and logging", "Each run saves a checkpoint every 100,000 steps"]
    assert [missing_pieces(q) for q in cited_quotes(titled)] == [[], ["Each run saves a checkpoint every 100,000 steps"]]
    listed = '''# Table 2.3 names "Every 200,000 steps" and "at onset, and at the end of training" and "at every epoch of training".'''
    assert [bool(missing_pieces(q)) for q in cited_quotes(listed)] == [False, False, True]
    # the next sentence or a later citation ends a citation's reach; a code span is not a quotation
    other = '''"""Part 3.4: "Each run saves a checkpoint every 200,000 steps". The restore rule reads "fresh Adam states and schedule".

    Table 2.5 ``params["horizons"]`` and "not a quotation at all"; Part 3.6 "The pilot's runs are not reused".
    """'''
    assert cited_quotes(other) == ["Each run saves a checkpoint every 200,000 steps", "not a quotation at all",
                                   "The pilot's runs are not reused"]


def test_a_numbered_list_is_quoted_as_the_pdf_prints_it() -> None:
    """Part 6.1's cuts are a Word list: the PDF prints "1." to "5.", the .docx text has no numbers."""
    right = ('"""Part 6.1 (PDF p. 18): "Cuts are made in this order, each applied only if the previous is '
             'insufficient. 1. Drop the PID check (15 runs). 2. Drop SafetyCarGoal1-v0 from every group of Study A"."""')
    assert [missing_pieces(q) for q in cited_quotes(right)] == [[]]
    for wrong in ("insufficient. 2. Drop the PID check (15 runs)", "insufficient. 1. Drop the PID check (16 runs)"):
        quote = f'"""Part 6.1: "Cuts are made in this order, each applied only if the previous is {wrong}"."""'
        assert [bool(missing_pieces(q)) for q in cited_quotes(quote)] == [True]


# A continuation's starting checkpoint is the answer of Q-continuations (Table 9.1, not the registered text;
# configs.registered.PENDING: "parent checkpoint ... of every continuation run"; Tables 2.2 and 2.5 name none),
# so prose that says where a continuation starts must name the key, or the answer reads as registered text.
CONTINUATION_START = re.compile(
    r"\b(?:start\w*|continu\w*|from)\b[^.;]{0,60}?\bparent's (?:matched|final|selected) checkpoint", re.I)
PARAGRAPH = re.compile(r"\n\s*\n|\n\s*(?=[*-] |\d+\. )")


def unlabelled_continuation_starts(source: str) -> list[str]:
    """Paragraphs (or list items) that say where a continuation starts without naming Q-continuations."""
    return [" ".join(para.split()) for text in _prose(source) for para in PARAGRAPH.split(text)
            if CONTINUATION_START.search(" ".join(para.split())) and "Q-continuations" not in para]


def test_a_continuations_starting_checkpoint_names_q_continuations() -> None:
    sources = {path: (REPO / path).read_text(encoding="utf-8") for path in FILES}
    assert not {path: bad for path, src in sources.items() if (bad := unlabelled_continuation_starts(src))}
    found = sum(1 for src in sources.values() for text in _prose(src) for para in PARAGRAPH.split(text)
                if CONTINUATION_START.search(" ".join(para.split())))
    assert found >= 3  # battery_continuation, study_b_fewshot and pilot.dependencies say it
    # a start cited to Part 4.1 and eq. 1 with no label is found
    wrong = '''"""The fine-tuning or transfer continuation of a matched Study A run (Table 2.2).

    Table 2.2: "Training continues for 1,000,000 steps". The continuation starts from the parent's matched
    checkpoint (Part 4.1 rule 1), the checkpoint whose robustness gap it measures (eq. 1).

    Proposals of Q-continuations, not the table's: every continuation trains with PPO-Lagrangian.
    """'''
    assert len(unlabelled_continuation_starts(wrong)) == 1
    listed = """'''Runs that depend on other runs:

    * the battery's continuations (Table 2.2), from the parent's matched
      checkpoint (``params["parent_step"]``);
    * Study B's few-shot continuations, from the parent's final checkpoint (proposal of Q-continuations).
    '''"""
    assert [p.startswith("* the battery's") for p in unlabelled_continuation_starts(listed)] == [True]


def test_the_q_h3_scope_text_quotes_box_h3_verbatim() -> None:
    """Every single-quoted span of the Q-h3-scope text, and of the analysis docstrings that copy it, is box H3's own
    wording: 'does not reduce' is not (box H3 (b): "... and additional constrained training does not.")."""
    text = R.PENDING["Q-h3-scope"]
    spans = re.findall(r"'([^']+)'", text)
    assert spans and "does not reduce" not in spans
    assert {s: missing_pieces(s) for s in spans if missing_pieces(s)} == {}
    for path in ("analysis/study_a.py", "analysis/verdict.py"):
        assert "'does not reduce'" not in (REPO / path).read_text(encoding="utf-8"), path

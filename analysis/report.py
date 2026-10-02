"""The analysis run and its outputs: tables, a JSON record and a Markdown report (Part 5; analysis/__main__.py).

Owner: Role 4, Analysis and results (Muhammad Abdullah).

``run`` computes every verdict of Study A and Study B (and, in pilot mode, the Part 5.8 consistency
check against the go report, after which every verdict is reported NOT_COMPUTABLE (pilot) with what its
code computed beside it: ``mark_pilot``); tables that rest on open questions get a ``provisional_on``
column (analysis.questions.TABLE_ENTRIES); ``write`` puts them in a new directory that is never overwritten:
``analysis.json`` (provenance, open questions, every verdict with its numbers, the names of the tables),
``report.md`` (provenance, data problems, awaited arms, every verdict with its status, numbers, intervals,
label and ``provisional_on`` keys, the open questions and the status counts) and every table twice, as
``tables/<name>.csv`` and ``tables/<name>.json`` (the same rows; NaN written as null).

Wording rules the report follows: Part 1.2 ("inconclusive at this sample size ... never rounded up to
support"); Part 5.7 ("The statement 'no effect' is not used"); Part 1.2 on H3 ("consistent with mediation
by plasticity" only if all three parts hold); box G3 (the proportion sentence is always printed); Part
4.2 ("the report says so" where the floor rule applies).
"""

from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from configs import registered as R

from analysis import questions, records, stats, study_a, study_b
from analysis.data import Dataset
from analysis.verdict import NOT_COMPUTABLE, UNDECIDED, Verdict

# Secondary claims without a box (one estimate each) are listed one per table row.
COMPACT_GROUPS = ("Study A: recovery time", "Study A: return", "Study B: satisfaction per budget",
                  "Study B: violation magnitude per budget")


@dataclass
class AnalysisResult:
    mode: str
    verdicts: list[Verdict]
    tables: dict[str, list[dict[str, Any]]]
    open_keys: list[str]
    problems: list[str]
    pilot: Optional[dict[str, Any]] = None
    awaited: list[str] = field(default_factory=list)  # Dataset.awaited: registered, uncut arms without a completed run
    awaited_b: list[str] = field(default_factory=list)  # Dataset.awaited_b_lines(): Study B arms below their seed target
    awaited_a: list[str] = field(default_factory=list)  # Dataset.awaited_a_lines(): Study A arms below their seed target
    provenance: dict[str, Any] = field(default_factory=dict)
    freeze: dict[str, Any] = field(default_factory=dict)

    @property
    def pilot_mismatches(self) -> list[str]:
        return list((self.pilot or {}).get("mismatches") or [])


def _printed(computed: float, registered: float) -> dict[str, Any]:
    """How a registered two-decimal value relates to the computed one (Part 5.5 prints two decimals)."""
    k = stats.POWER_PRINTED_DECIMALS
    nearest = round(computed, k)
    up = math.ceil(computed * 10 ** k) / 10 ** k
    how = ("rounded to nearest" if nearest == registered else "rounded up" if up == registered
           else "neither rounding gives it")
    return {"computed_rounded": nearest, "registered_is": how}


def power_rows(dataset: Dataset) -> list[dict[str, Any]]:
    """Part 5.5: the power statement "computed from" the seed count actually used, and the registered values.

    Each registered value says how it relates to the computed one: every value is the computed one
    rounded to nearest except the 80-percent point at ten seeds, 1.3249 printed as 1.33 (rounded up;
    power at 1.33 is 0.803; stats module docstring).
    """
    rows = []
    n5 = len(R.SEEDS)  # Part 5.5 "With five seeds per arm" (the registered seed count, never retyped)
    for d, registered in R.POWER_REGISTERED_N5.items():
        computed = stats.power_two_sample(d, n5)
        rows.append({"what": "registered power at five seeds", "n": n5, "d": d, "alpha": R.ALPHA,
                     "registered": registered, "computed": computed, **_printed(computed, registered)})
    for n, registered in R.POWER_REGISTERED_D80.items():
        computed = stats.detectable_d(n)
        rows.append({"what": "registered 80-percent point", "n": n, "alpha": R.ALPHA, "registered": registered,
                     "computed": computed,
                     **(_printed(computed, registered) if computed is not None else {"note": "not computable"}),
                     "power_at_registered": stats.power_two_sample(registered, n)})
    counts = sorted({len(v) for v in dataset.seeds_per_arm().values()})
    for n in counts:
        if n < 2:
            continue
        for what, alpha in (("80-percent point at the seed count used", R.ALPHA),
                            (f"80-percent point at the seed count used, alpha {R.ALPHA_PRIMARY_STUDY_B} (Study B G1, G2)",
                             R.ALPHA_PRIMARY_STUDY_B)):
            computed = stats.detectable_d(n, alpha)
            # an annotation (Part 5.5): a size the root finder cannot give is named, never an error
            rows.append({"what": what, "n": n, "alpha": alpha, "computed": computed,
                         **({} if computed is not None else {"note": "not computable"})})
    return rows


PILOT_REASON = ("pilot test run (Part 5.8): the analysis 'is run once on the pilot data to test it'; the code ran, "
                "but its verdicts are not results of the study (Part 3.6: 'The pilot's runs are not reused')")


def mark_pilot(verdict: Verdict) -> None:
    """Pilot mode: every verdict is NOT COMPUTABLE (pilot) (Part 5.8: the test run on the pilot data; Part 3.6: "The
    pilot's runs are not reused").

    The verdict's code path has run (that is the test); what it computed is kept in the numbers
    (``pilot_test_computed_status``, its reason and the other readings' statuses) and a note, and the
    verdict itself is NOT_COMPUTABLE, so no pilot number is reported as a registered verdict.
    """
    computed = verdict.display_status
    verdict.numbers = {"pilot_test_computed_status": computed, "pilot_test_computed_reason": verdict.reason,
                       "pilot_test_readings": dict(verdict.readings), **verdict.numbers}
    # Part 1.2's mediation wording for H3 belongs to a SUPPORTED H3 only, which a pilot verdict never is
    verdict.notes = [n for n in verdict.notes if n != study_a.MEDIATION_NOTE]
    verdict.notes.append(f"pilot test run (Part 5.8): the code computed {computed}; not a result of the study")
    verdict.status = verdict.proposal_status = NOT_COMPUTABLE
    verdict.reason = PILOT_REASON
    verdict.undecided_by = ()
    verdict.readings = {}


def table_provisional_on(tables: dict[str, list[dict[str, Any]]]) -> None:
    """Add a provisional_on column (the open keys of its analysis.questions entry) to every table that has one."""
    for name, entry in questions.TABLE_ENTRIES.items():
        if name in tables:
            keys = ", ".join(questions.provisional_on(entry))
            tables[name] = [{**row, "provisional_on": keys} for row in tables[name]]


def run(dataset: Dataset, *, amendments: Optional[list[dict[str, Any]]] = None,
        errata: Optional[list[dict[str, Any]]] = None) -> AnalysisResult:
    """Every verdict and table of the analysis on one dataset (in pilot mode, the Part 5.8 test run)."""
    amendments = records.load_amendments() if amendments is None else amendments
    errata = records.load_errata() if errata is None else errata
    a = study_a.analyse(dataset)
    b = study_b.analyse(dataset)
    verdicts = a.verdicts + b.verdicts
    records.apply_amendments(verdicts, amendments)
    tables = {**a.tables, **b.tables}
    tables["power"] = power_rows(dataset)
    tables["seeds_per_arm"] = [{"arm_id": k, "n": len(v), "seeds": " ".join(map(str, v))}
                               for k, v in dataset.seeds_per_arm().items()]
    table_provisional_on(tables)
    pilot = None
    if dataset.mode == "pilot":
        from analysis import pilot_check

        pilot = pilot_check.run(dataset)
        tables["pilot_consistency"] = pilot["consistency"]
        tables["pilot_estimates"] = pilot["estimates"]
        for v in verdicts:
            mark_pilot(v)
    tables["verdicts"] = [_verdict_row(v) for v in verdicts]
    return AnalysisResult(mode=dataset.mode, verdicts=verdicts, tables=tables, open_keys=questions.open_keys(),
                          problems=list(dataset.problems), pilot=pilot, awaited=list(dataset.awaited),
                          awaited_b=dataset.awaited_b_lines(), awaited_a=dataset.awaited_a_lines(),
                          freeze=records.freeze_state(amendments, errata))


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return "n/a" if math.isnan(value) else f"{value:.4g}"
    if isinstance(value, (list, tuple)):
        return "(" + ", ".join(_fmt(v) for v in value) + ")"
    return str(value)


def _cell(value: Any) -> str:
    """A Markdown table cell: pipes escaped, line breaks removed."""
    return str(value).replace("|", "\\|").replace("\n", " ")


def _verdict_row(v: Verdict) -> dict[str, Any]:
    n = v.numbers
    return {"id": v.id, "group": v.group, "status": v.display_status, "proposal_status": v.proposal_status,
            "label": v.label, "statement": v.statement, "reason": v.reason,
            "diff": n.get("diff"), "welch_lo": n.get("welch_lo"), "welch_hi": n.get("welch_hi"),
            "holm_adjusted_p": n.get("holm_adjusted_p"),
            "undecided_by": ", ".join(v.undecided_by), "provisional_on": ", ".join(v.provisional_on),
            "readings": json.dumps(v.readings, sort_keys=True)}


def _interval(n: dict[str, Any]) -> str:
    if n.get("welch_lo") is None:
        return "n/a"
    return f"[{_fmt(n.get('welch_lo'))}, {_fmt(n.get('welch_hi'))}] (percentile [{_fmt(n.get('boot_lo'))}, {_fmt(n.get('boot_hi'))}])"


def _verdict_md(v: Verdict) -> list[str]:
    lines = [f"### {v.id}: {v.display_status}", "", f"*{v.title}.* Statement: {v.statement}.", "",
             f"- label: {v.label}",
             f"- provisional_on: {', '.join(v.provisional_on) if v.provisional_on else 'none'}"]
    if v.status == UNDECIDED:
        lines.append(f"- status under every proposal: {v.proposal_status}")
    if v.readings:
        lines.append("- status under the other reading of each open question: "
                     + "; ".join(f"{k}: {s}" for k, s in sorted(v.readings.items())))
    if v.reason:
        lines.append(f"- reason: {v.reason}")
    for key, value in v.numbers.items():
        lines.append(f"- {key}: {_fmt(value)}")
    if v.estimates:
        lines += ["", "| contrast | diff | Welch interval | percentile interval | Cohen's d | d80 at n (\\|d\\| below it) | n |",
                  "|---|---|---|---|---|---|---|"]
        for e in v.estimates:
            lines.append(f"| {_cell(e.get('contrast', e.get('analysis_id', '')))} {_cell(e.get('control') or '')} | "
                         f"{_fmt(e.get('diff'))} | "
                         f"[{_fmt(e.get('welch_lo'))}, {_fmt(e.get('welch_hi'))}] | [{_fmt(e.get('boot_lo'))}, "
                         f"{_fmt(e.get('boot_hi'))}] | {_fmt(e.get('cohens_d'))} | {_fmt(e.get('detectable_d'))} "
                         f"({_fmt(e.get('below_detectable'))}) | ({e.get('n_x')}, {e.get('n_y')}) |")
    for note in v.notes:
        lines.append(f"- note: {note}")
    lines.append("")
    return lines


def _compact_md(group: str, verdicts: list[Verdict]) -> list[str]:
    lines = [f"## {group}", "", "| id | status | label | diff | 95 percent Welch interval (percentile) | Holm p | "
             "\\|d\\| below d80 | reason | provisional_on |", "|---|---|---|---|---|---|---|---|---|"]
    for v in verdicts:
        n = v.numbers
        lines.append(f"| {_cell(v.id)} | {v.display_status} | {v.label} | {_fmt(n.get('diff'))} | {_interval(n)} | "
                     f"{_fmt(n.get('holm_adjusted_p'))} | {_fmt(n.get('below_detectable'))} | {_cell(v.reason or '')} | "
                     f"{', '.join(v.provisional_on)} |")
    lines.append("")
    return lines


def to_markdown(result: AnalysisResult) -> str:
    p = result.provenance
    title = "pilot test run (Part 5.8)" if result.mode == "pilot" else "final analysis"
    lines = [f"# Analysis report: {title}", ""]
    if result.mode == "pilot":
        lines += ["> This is the Part 5.8 test run of the analysis on the pilot data. It is not a result of the study "
                  "and it never decides go: the go conditions of Part 6 are read from the go report "
                  "(`python -m pilot go`).", ""]
    lines += ["## Provenance", ""]
    for key in ("utc", "commit", "worktree_dirty", "argv", "ledger", "ledger_sha256", "supplement_dir",
                "supplement_dir_exists", "supplement_files", "supplement_sha256", "inputs", "analysis_code_hash",
                "answered_questions", "bootstrap", "recorded_cuts"):
        if key in p:
            lines.append(f"- {key}: {_fmt(p[key]) if not isinstance(p[key], (dict, list)) else json.dumps(p[key])}")
    lines.append(f"- Part 5.8 freeze: {result.freeze.get('state')}")
    lines.append("")
    if result.problems:
        lines += ["## Data problems (the ledger and the supplement disagree, a record is inconsistent or of another "
                  "checkpoint or C_ID, or rule 1 selects another checkpoint)", ""]
        lines += [f"- {x}" for x in result.problems] + [""]
    if result.awaited:
        lines += ["## Registered arms without a completed run (data still to come)", "",
                  "These Study A arms are in the registered design and no recorded Part 6.1 cut removed them; every "
                  "comparison and Holm family that needs one waits on it (incomplete, never 'not tested'):", ""]
        lines += [f"- {x}" for x in result.awaited] + [""]
    if result.awaited_b:
        lines += ["## Study B arms below their seed target (data still to come)", "",
                  "These Study B arms have fewer completed runs than their seed target (Q-arm-complete: 5 plus the "
                  "surplus seeds of Part 5.5; an excluded run waits for its replacement); every comparison that needs "
                  "one is incomplete (NOT_COMPUTABLE), never decided on the seeds present:", ""]
        lines += [f"- {x}" for x in result.awaited_b] + [""]
    if result.awaited_a:
        lines += ["## Study A arms below their seed target (data still to come)", "",
                  "These Study A arms have some completed runs but fewer than their seed target (Q-arm-complete: 5, plus "
                  "the surplus seeds of Part 5.5 for the primary-comparison arms; an excluded run waits for its "
                  "replacement); every comparison that needs one is incomplete (NOT_COMPUTABLE), never decided on the "
                  "seeds present:", ""]
        lines += [f"- {x}" for x in result.awaited_a] + [""]
    lines += ["## Summary", "", "| id | status | label | open keys (provisional_on count) |", "|---|---|---|---|"]
    for v in result.verdicts:
        if v.group not in COMPACT_GROUPS:
            lines.append(f"| {_cell(v.id)} | {v.display_status} | {v.label} | {len(v.provisional_on)} |")
    lines.append("")
    groups: dict[str, list[Verdict]] = {}
    for v in result.verdicts:
        groups.setdefault(v.group, []).append(v)
    for group, members in groups.items():
        if group in COMPACT_GROUPS:
            lines += _compact_md(group, members)
            continue
        lines += [f"## {group}", ""]
        for v in members:
            lines += _verdict_md(v)
    if result.pilot is not None:
        lines += ["## Pilot consistency with the go report (Part 5.8)", "", result.pilot["note"], "",
                  "| go condition | quantity | analysis | go report | agree | note | provisional_on |",
                  "|---|---|---|---|---|---|---|"]
        for row in result.pilot["consistency"]:
            lines.append(f"| {row.get('condition', '')} | {row.get('quantity', '')} | {_fmt(row.get('analysis'))} | "
                         f"{_fmt(row.get('go_report'))} | {_fmt(row.get('agree'))} | {_cell(row.get('note', ''))} | "
                         f"{row.get('provisional_on', '')} |")
        lines.append("")
        if result.pilot["mismatches"]:
            lines += ["**The analysis and the go report disagree:**", ""] + [f"- {m}" for m in result.pilot["mismatches"]] + [""]
        lines += ["Table 8.1 values recomputed by the analysis:", ""]
        lines += [f"- {k}: {_fmt(v)}" for k, v in result.pilot["table_8_1"].items()] + [""]
    lines += ["## Open questions", ""]
    if result.open_keys:
        lines += ["Every verdict lists in provisional_on the open keys it depends on; these PENDING keys of "
                  "configs/registered.py are not yet answered in the amendment log (Part 9):", ""]
        lines += [f"- {k}: {R.PENDING[k]}" for k in result.open_keys] + [""]
    else:
        lines += ["None: every PENDING key is answered (configs.registered.ANSWERED_QUESTIONS).", ""]
    counts: dict[str, int] = {}
    for v in result.verdicts:
        counts[v.status] = counts.get(v.status, 0) + 1
    lines += ["## Counts", ""] + [f"- {k}: {n}" for k, n in sorted(counts.items())] + [""]
    if counts.get(NOT_COMPUTABLE):
        lines += ["NOT_COMPUTABLE verdicts give their reason; a missing field is never dropped silently.", ""]
    return "\n".join(lines)


def to_json(result: AnalysisResult) -> str:
    data = {"mode": result.mode, "provenance": result.provenance, "freeze": result.freeze,
            "open_keys": result.open_keys, "problems": result.problems, "awaited_arms": result.awaited,
            "awaited_study_b_arms": result.awaited_b, "awaited_study_a_arms": result.awaited_a,
            "verdicts": [v.to_dict() for v in result.verdicts], "pilot": result.pilot,
            "tables": sorted(result.tables)}
    return json.dumps(stats.to_plain(data), indent=1, sort_keys=True, allow_nan=False) + "\n"


def _json_rows(rows: list[dict[str, Any]]) -> str:
    """A table as a JSON list of row objects (the CSV's rows, types kept)."""
    return json.dumps(stats.to_plain(rows), indent=1, sort_keys=True, allow_nan=False) + "\n"


def _csv_text(rows: list[dict[str, Any]]) -> str:
    buffer = io.StringIO()
    fields: list[str] = []
    for row in rows:
        fields += [k for k in row if k not in fields]
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        plain = stats.to_plain(row)
        writer.writerow({k: ("" if v is None else json.dumps(v) if isinstance(v, (list, dict)) else v)
                         for k, v in plain.items()})
    return buffer.getvalue()


def write(result: AnalysisResult, out_dir: Path) -> list[Path]:
    """Write every output into ``out_dir``, which must not exist yet (an analysis is never overwritten)."""
    out_dir = Path(out_dir)
    texts = {"analysis.json": to_json(result), "report.md": to_markdown(result)}  # rendered before anything is created
    tables = {}
    for name, rows in sorted(result.tables.items()):
        tables[f"tables/{name}.csv"] = _csv_text(rows)
        tables[f"tables/{name}.json"] = _json_rows(rows)
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "tables").mkdir()
    written = []
    for rel, text in {**texts, **tables}.items():
        path = out_dir / rel
        with open(path, "x", encoding="utf-8") as fh:
            fh.write(text)
        written.append(path)
    return written

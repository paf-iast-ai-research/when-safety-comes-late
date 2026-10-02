"""``python -m analysis --mode pilot|final --ledger PATH [--surplus-extra E] [--supplement DIR] [--out DIR]
[--revision 0|1] [--allow-dirty]`` (Part 5.8).

Owner: Role 4, Analysis and results (Muhammad Abdullah).

Part 5.8: "The analysis script is committed to the repository before the first main-study run
completes, is run once on the pilot data to test it, and is not edited between the go decision and
the final analysis except to fix errors, each of which is recorded in Part 9."

* ``--mode pilot`` reads the rows of one pilot (``--revision 0`` the pilot ``P-``, ``1`` the re-pilot
  ``P1-``) and adds the consistency check with the go report (analysis.pilot_check); it never decides
  go, and every verdict it computes is reported NOT_COMPUTABLE (pilot) with what the code computed
  beside it (analysis.report.mark_pilot). A test run must test something: a selected pilot population
  without a completed Study A run (the wrong ``--revision``, or a registered ledger) is refused, and a
  run on pilot rows where no G1 or G3 quantity could be compared with the go report is written but
  exits 4, never 0. ``--mode final`` refuses pilot rows (Part 3.6: "The pilot's runs are not reused")
  and ``--revision 1`` (a pilot-mode option, recorded in pilot mode only), and needs ``--surplus-extra E``
  (Part 5.5), which sets the seed target of each primary-comparison arm (Study A's N = 0 and N = 0.50
  arms on SafetyPointGoal1-v0 and the Study B arms; Q-arm-complete): an arm below its target is data
  still to come, and the comparisons that need it are incomplete.
* The supplement defaults to ``<ledger dir>/supplement`` (results/supplement_schema.py). A ``--supplement`` given
  explicitly must be an existing directory (a typo would otherwise analyse no supplement records);
  a missing default is reported on stderr and in the provenance (``supplement_dir_exists``).
* The output directory defaults to ``results/analysis/<mode>-<UTC>/`` and must not exist: an analysis
  is never overwritten.
* Like ``python -m pilot go``, the command refuses a dirty tree and, after the analysis, any imported
  repository file the commit does not hold (pilot.provenance), so the report's commit is the code that
  produced it; HEAD moving during the run to a commit that changes loaded code, or the analysis code
  changing on disk (analysis/*.py and the modules it imports, ``records.code_hash``; its
  ``analysis_code_hash`` is read at the start), is refused likewise. The three
  data files of the repository that change verdicts, ``pilot/cuts.json`` (the Part 6.1 cuts,
  pilot.scheduler.CUTS_PATH), ``analysis/AMENDMENTS.json`` (Part 8.2 labels) and ``analysis/ERRATA.json`` (the Part 5.8
  freeze), are read once, compared byte for byte with the commit (an untracked, modified or deleted
  one is refused), parsed from those bytes and recorded with their SHA-256 (``inputs``).
  ``--allow-dirty`` is for smoke and test runs only and is refused for an output inside the repository's results/.
* Each supplement file is hashed from the bytes parsed; the ledger is hashed before it is read, again
  after the supplement scan, and again just before anything is written: a ledger another process wrote
  meanwhile is refused.

Exit status: 0 done; 2 refused (nothing written); 3 written, but in pilot mode the analysis and the go
report disagree on a number (the test run found an error); 4 written, but in pilot mode nothing could
be compared with the go report (the test run tested nothing). Data the analysis cannot read (a row the
frozen schema accepts but rules 1 to 6 cannot, a few-shot key not in the canonical form, a bad cut
record) are a refusal (exit 2, nothing written), never a traceback.
"""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
EXIT_OK, EXIT_REFUSED, EXIT_PILOT_MISMATCH, EXIT_PILOT_UNTESTED = 0, 2, 3, 4


class CliError(Exception):
    """A refusal: the message is printed and nothing is written."""


def input_files() -> tuple[Path, ...]:
    """The repository's data files that change verdicts (the cut record and the two analysis registries)."""
    from analysis import data, records

    return (data.CUTS_FILE, records.AMENDMENTS_FILE, records.ERRATA_FILE)


def _rel(path: Path) -> str:
    """The path as git names it (relative to the repository), or as given when outside it."""
    try:
        return Path(path).resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def read_inputs(commit: Optional[str]) -> tuple[dict[str, Optional[bytes]], dict[str, Optional[bool]]]:
    """Each input file's bytes as read now (None: absent) and whether they equal the commit's (None: not checked).

    The comparison is of bytes (``pilot.provenance.file_committed_at``): an untracked file, a modified
    one and one deleted from the working tree all differ from the commit, whatever git's status
    filters show (``dirty_paths`` ignores untracked files).
    """
    from pilot import provenance

    contents: dict[str, Optional[bytes]] = {}
    committed: dict[str, Optional[bool]] = {}
    for path in input_files():
        rel = _rel(path)
        contents[rel] = path.read_bytes() if path.is_file() else None
        committed[rel] = None if commit is None else provenance.file_committed_at(commit, rel) == contents[rel]
    return contents, committed


def _describe_input(rel: str, content: Optional[bytes]) -> str:
    return f"{rel} ({'absent from the working tree' if content is None else 'untracked or differs from the commit'})"


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m analysis", description="The registered analysis (Parts 4 and 5).")
    p.add_argument("--mode", choices=("pilot", "final"), required=True,
                   help="pilot: the Part 5.8 test run on the pilot ledger; final: the registered runs")
    p.add_argument("--ledger", required=True, help="the ledger (results/ledger.parquet or results/pilot/ledger.parquet)")
    p.add_argument("--supplement", default=None, help="supplement records (default: <ledger dir>/supplement)")
    p.add_argument("--out", default=None, help="new output directory (default: results/analysis/<mode>-<UTC>/)")
    p.add_argument("--revision", type=int, choices=(0, 1), default=0, help="pilot mode: 0 the pilot, 1 the re-pilot")
    p.add_argument("--surplus-extra", type=int, default=None, metavar="E",
                   help="final mode (required): the surplus seeds per primary-comparison arm (Part 5.5; 0 without "
                        "surplus), which set the seed target of each primary-comparison arm (Study A's N = 0 and "
                        "N = 0.50 arms on SafetyPointGoal1-v0 and the Study B arms; Q-arm-complete), as for "
                        "`enrich match`")
    p.add_argument("--allow-dirty", action="store_true",
                   help="smoke and test runs only: run from uncommitted code (refused for outputs in results/)")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    try:
        return _run(args, list(sys.argv[1:] if argv is None else argv))
    except CliError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED


def _require_code_of(provenance: Any, start: str) -> None:
    """A CliError if HEAD moved from ``start`` (the commit the report names) to a commit that changes a module this
    process loaded: ``unverified_imported_code`` compares the working tree with the new HEAD only (as
    ``pilot.__main__._require_code_of``)."""
    import subprocess

    head = provenance.commit_hash()
    if head == start:
        return
    try:
        moved = provenance.imported_code_changed_between(start, head)
    except subprocess.CalledProcessError:
        moved = [f"(git cannot compare {start} with {head})"]
    if moved:
        raise CliError(f"HEAD moved from {start} to {head} while the analysis ran, and that commit changes code it "
                       "loaded: " + ", ".join(moved) + "; nothing is written: run the analysis again")


def _run(args: argparse.Namespace, argv: list[str]) -> int:
    from configs import registered as R
    from pilot import provenance

    from analysis import data, records, report, stats

    ledger = Path(args.ledger)
    if not ledger.is_file():
        raise CliError(f"ledger {ledger} does not exist")
    if args.mode == "final" and args.surplus_extra is None:
        raise CliError("--mode final needs --surplus-extra E: the surplus seeds per primary-comparison arm (Part 5.5; 0 "
                       "without surplus), which set the seed target of each primary-comparison arm (Study A's N = 0 "
                       "and N = 0.50 arms on SafetyPointGoal1-v0 and the Study B arms; Q-arm-complete)")
    if args.mode == "pilot" and args.surplus_extra is not None:
        raise CliError("--mode pilot takes no --surplus-extra")
    if args.mode == "final" and args.revision != 0:
        raise CliError("--mode final takes no --revision (it selects the pilot or the re-pilot in pilot mode)")
    stamp = provenance.utc_now().strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out) if args.out else REPO_ROOT / "results" / "analysis" / f"{args.mode}-{stamp}"
    if out.exists():
        raise CliError(f"{out} exists; an analysis is never overwritten (choose a new --out)")
    if args.allow_dirty and _inside(out, REPO_ROOT / "results"):
        raise CliError("--allow-dirty is for smoke and test runs; its output may not go to the repository's results/")
    try:
        commit = provenance.commit_hash()
        dirty = provenance.dirty_paths()
    except Exception as exc:  # noqa: BLE001 - no git: only a smoke run may go on
        if not args.allow_dirty:
            raise CliError(f"cannot read the commit ({exc}); the report must record the code that produced it") from None
        commit, dirty = "unavailable", ["(git unavailable)"]
    if dirty and not args.allow_dirty:
        raise CliError("tracked code has uncommitted changes; commit before running the analysis: " + ", ".join(dirty))
    # the analysis code as the run starts: the report's analysis_code_hash, compared again before anything is
    # written (a commit landing while the analysis runs must not give a report naming one commit and another's code)
    code_hash = records.code_hash()
    contents, committed = read_inputs(None if commit == "unavailable" else commit)
    uncommitted = [_describe_input(rel, contents[rel]) for rel, ok in committed.items() if ok is not True]
    if uncommitted and not args.allow_dirty:
        raise CliError("the analysis would read data files that the commit does not hold as they are: "
                       + ", ".join(uncommitted) + "; commit them (or remove an untracked one) first")
    if args.supplement:
        supplement_dir = Path(args.supplement)
        if not supplement_dir.is_dir():
            raise CliError(f"--supplement {supplement_dir} is not an existing directory")
    else:
        supplement_dir = data.default_supplement_dir(ledger)
        if supplement_dir.exists() and not supplement_dir.is_dir():
            raise CliError(f"{supplement_dir} is not a directory")
        if not supplement_dir.exists():
            print(f"warning: no supplement directory {supplement_dir}: every analysis that needs supplement records "
                  "is NOT_COMPUTABLE", file=sys.stderr)
    cuts_rel, amendments_rel, errata_rel = (_rel(p) for p in input_files())
    try:
        dataset = data.load_dataset(ledger, mode=args.mode, supplement_dir=supplement_dir, revision=args.revision,
                                    cuts=data.parse_cuts(contents[cuts_rel], cuts_rel),
                                    surplus_extra=args.surplus_extra or 0)
        amendments = records.load_amendments(content=contents[amendments_rel])
        errata = records.load_errata(content=contents[errata_rel])
        result = report.run(dataset, amendments=amendments, errata=errata)
    except (data.DataError, records.RegistryError) as exc:
        raise CliError(str(exc)) from None
    except ValueError as exc:  # a data refusal raised below data.py (matching, pilot.contracts): never a traceback
        raise CliError(f"the data cannot be analysed ({type(exc).__name__}): {exc}") from None
    if not args.allow_dirty:
        unverified = provenance.unverified_imported_code()
        if unverified:
            raise CliError("the analysis ran code that the commit does not contain: " + ", ".join(unverified))
        _require_code_of(provenance, commit)
    if records.code_hash() != code_hash:
        raise CliError("the analysis code (analysis/*.py and the modules it imports, records.code_hash) changed while "
                       "the analysis ran (its code hash differs from the start); nothing "
                       "is written: run the analysis again")
    if data.file_sha256(ledger) != dataset.ledger_sha256:
        raise CliError(f"the ledger {ledger} changed while the analysis ran (another process wrote it); "
                       "nothing is written: run the analysis again")
    import numpy
    import scipy

    result.provenance = {
        "utc": provenance.utc_now().isoformat(), "commit": commit, "worktree_dirty": bool(dirty), "argv": argv,
        "mode": args.mode, "revision": args.revision if args.mode == "pilot" else None,  # a pilot-mode option
        "ledger": str(ledger), "ledger_sha256": dataset.ledger_sha256,
        "supplement_dir": str(supplement_dir), "supplement_dir_exists": supplement_dir.is_dir(),
        "supplement_files": len(dataset.supplement.files), "supplement_sha256": dataset.supplement.sha256(),
        "inputs": {rel: {"sha256": None if content is None else data.sha256_bytes(content), "committed": committed[rel]}
                   for rel, content in contents.items()},
        "analysis_code_hash": code_hash, "answered_questions": sorted(R.ANSWERED_QUESTIONS),
        "bootstrap": {"seed": stats.BOOTSTRAP_SEED, "quantile": stats.QUANTILE_METHOD},
        "seeds_per_arm": dataset.seeds_per_arm(), "rows_of_other_populations_ignored": len(dataset.ignored),
        "auxiliary_runs_left_out": dataset.auxiliary, "surplus_extra": dataset.surplus_extra,
        "study_b_arms_awaited": dataset.awaited_b_lines(), "study_a_arms_awaited": dataset.awaited_a_lines(),
        "recorded_cuts": list(dataset.cuts),
        "python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__,
    }
    written = report.write(result, out)
    print(f"wrote {len(written)} files to {out}")
    if result.problems:
        print(f"{len(result.problems)} data problems (the ledger and the supplement disagree, a record is "
              "inconsistent or of another checkpoint or C_ID, or rule 1 selects another checkpoint): "
              f"see {out / 'report.md'}", file=sys.stderr)
    if result.awaited:
        print(f"{len(result.awaited)} registered, uncut Study A arms have no completed run: the comparisons and Holm "
              f"families that need them are incomplete; see {out / 'report.md'}", file=sys.stderr)
    if result.awaited_b:
        print(f"{len(result.awaited_b)} Study B arms are below their seed target (Q-arm-complete): the comparisons that "
              f"need them are incomplete; see {out / 'report.md'}", file=sys.stderr)
        for line in result.awaited_b:
            print(f"  {line}", file=sys.stderr)
    if result.awaited_a:
        print(f"{len(result.awaited_a)} Study A arms are below their seed target (Q-arm-complete): the comparisons that "
              f"need them are incomplete; see {out / 'report.md'}", file=sys.stderr)
        for line in result.awaited_a:
            print(f"  {line}", file=sys.stderr)
    for v in result.verdicts:
        if v.group.startswith(("Study A: primary", "Study A: hypotheses", "Study B: primary", "Study B: hypotheses")):
            print(f"  {v.id}: {v.display_status} ({v.label})")
    if result.pilot_mismatches:
        print("the analysis and the go report disagree:", file=sys.stderr)
        for m in result.pilot_mismatches:
            print(f"  {m}", file=sys.stderr)
        return EXIT_PILOT_MISMATCH
    if result.pilot is not None and not result.pilot["compared"]:
        print("the Part 5.8 test run compared nothing with the go report: the go report computed no G1 or G3 quantity "
              "on these pilot rows (see the consistency table); the analysis was not tested", file=sys.stderr)
        return EXIT_PILOT_UNTESTED
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())

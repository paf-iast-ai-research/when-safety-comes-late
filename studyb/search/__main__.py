"""``python -m studyb.search``: print the protocol's searches, or check a record.

Owner: Study B and literature (Role 5, Hamza Nisar). A helper for the searcher (Appendix C, Table
C.1); it reads files and prints, and never writes anything.

    python -m studyb.search queries            # every query variant, source by source (Q-search-queries)
    python -m studyb.search check [--root DIR] # SearchStatus of the record (default: studyb/search)
"""

from __future__ import annotations

import argparse
import sys

from studyb.search import protocol
from studyb.search.record import RECORD_ROOT, check_record, screened_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m studyb.search",
                                     description="print the protocol's searches, or check a search record")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("queries", help="print every search of the protocol (source, query_id, string, screened file)")
    check = sub.add_parser("check", help="check a record; exit 0 only when it is complete")
    check.add_argument("--root", default=str(RECORD_ROOT), help="the record's directory (default: %(default)s)")
    args = parser.parse_args(argv)
    if args.command == "queries":
        print(f"screening depth: the first {protocol.SCREENING_DEPTH} results of each search, in each source's "
              f"{protocol.RESULT_ORDER} order (Q-search-depth)")
        for source in protocol.SOURCES:
            for variant in protocol.query_variants():
                print(f"{source}\t{variant.query_id}\t{variant.text}\t{screened_file(source, variant.query_id)}")
        return 0
    status = check_record(args.root)
    print(f"complete: {status.complete}; outcome: {status.outcome}")
    for problem in status.problems:
        print(f"- {problem}")
    return 0 if status.complete else 1


if __name__ == "__main__":
    sys.exit(main())

"""Copy OmniSafe's configuration files, unchanged, into configs/omnisafe/ (First Tasks, Role 1 step 7).

Copies PPOLag.yaml and CPPOPID.yaml (named by First Tasks) and PPO.yaml (used by the Study B pilot's
unconstrained run, Part 3.6) from the INSTALLED omnisafe package, and records the package version
and each file's SHA-256 in configs/omnisafe/SOURCE.json (Appendix A: "Copied verbatim").

    python scripts/copy_omnisafe_configs.py          # copy (refuses to overwrite different content)
    python scripts/copy_omnisafe_configs.py --check  # verify the copies in configs/omnisafe/ equal the installed files
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEST = REPO / "configs" / "omnisafe"
FILES = ("PPOLag.yaml", "CPPOPID.yaml", "PPO.yaml")


def installed_dir() -> Path:
    """The installed omnisafe package's on-policy configuration directory."""
    import omnisafe

    return Path(omnisafe.__file__).resolve().parent / "configs" / "on-policy"


def sha256(path: Path) -> str:
    """The SHA-256 hex digest of the file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    """Copy the files, or with --check verify them; return the exit code (module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true",
                        help="verify the copies in configs/omnisafe/ (the working tree) and SOURCE.json against "
                             "the installed package")
    args = parser.parse_args(argv)
    src = installed_dir()
    version = importlib.metadata.version("omnisafe")
    problems = []
    if args.check:
        for name in FILES:
            if not (DEST / name).exists() or sha256(DEST / name) != sha256(src / name):
                problems.append(name)
        if (DEST / "SOURCE.json").exists():
            source = json.loads((DEST / "SOURCE.json").read_text(encoding="utf-8"))
        else:
            problems.append("SOURCE.json missing")
            source = {}
        copies = {n: sha256(DEST / n) for n in FILES if (DEST / n).exists()}
        if source.get("files") != copies:
            problems.append("SOURCE.json digests differ from the copies")
        if source.get("omnisafe_version") != version:
            problems.append(f"SOURCE.json records omnisafe {source.get('omnisafe_version')}, installed {version}")
        if problems:
            print(f"MISMATCH: {problems}", file=sys.stderr)
            return 1
        print("configs/omnisafe matches the installed package")
        return 0
    DEST.mkdir(parents=True, exist_ok=True)
    record = {"omnisafe_version": version, "source_directory": "omnisafe/configs/on-policy", "files": {}}
    # Check every target before copying any, so a refusal leaves the copies as they were.
    problems = [n for n in FILES if (DEST / n).exists() and sha256(DEST / n) != sha256(src / n)]
    if problems:
        print(f"refusing to overwrite changed copies: {problems}", file=sys.stderr)
        return 1
    for name in FILES:
        shutil.copyfile(src / name, DEST / name)
        record["files"][name] = sha256(DEST / name)
    (DEST / "SOURCE.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"copied {', '.join(FILES)} from omnisafe {version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

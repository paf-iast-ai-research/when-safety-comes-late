"""Record the workstation specification (Table 3.1 "Hardware and device"; Appendix A).

    python scripts/record_workstation.py               # writes environment/workstation.json
    python scripts/record_workstation.py --out FILE    # another machine: write elsewhere

Records the operating system, Python, CPU model, logical and physical core counts, total memory,
the data root and its free disk space, and the versions of the pinned packages. Commit the file with the
environment lock (First Tasks, Role 1 step 6: "Record the Python version and the operating system").
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from pilot.__main__ import DEFAULT_DATA_ROOT  # noqa: E402
from pilot.provenance import machine_description, utc_now  # noqa: E402

# Every direct pin of environment/requirements.in.
PACKAGES = ("omnisafe", "safety-gymnasium", "mujoco", "gymnasium", "torch", "numpy", "pandas", "pyarrow", "pydantic",
            "pytest", "scipy")


def _meminfo_gib() -> float | None:
    """Total memory in GiB from /proc/meminfo (None where it is unavailable, e.g. off Linux)."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / 1024 / 1024, 1)
    except OSError:
        return None
    return None


def physical_cores() -> int | None:
    """Distinct (physical id, core id) pairs of /proc/cpuinfo, each processor's own (None where it is unavailable,
    e.g. off Linux). scripts/workstation.py suggests the pilot's concurrency from it."""
    cores: set[tuple[str | None, str]] = set()
    phys: str | None = None
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                key, _, value = line.partition(":")
                if key.strip() == "physical id":
                    phys = value.strip()
                elif key.strip() == "core id":
                    cores.add((phys, value.strip()))
                elif not line.strip():  # the next processor's block
                    phys = None
    except OSError:
        return None
    return len(cores) or None


def main(argv: list[str] | None = None) -> int:
    """Write the record to ``--out`` (creating its directory) and print it; return the exit code."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=REPO / "environment" / "workstation.json",
                        help="the file to write (default environment/workstation.json)")
    args = parser.parse_args(argv)
    data_root = Path(DEFAULT_DATA_ROOT)  # $WSCL_DATA_ROOT or /data, as python -m pilot
    versions = {}
    for pkg in PACKAGES:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "not installed"
    disk = shutil.disk_usage(data_root) if data_root.exists() else None
    record = {
        "recorded_utc": utc_now().isoformat(),
        "machine": machine_description(),
        "os": platform.platform(),
        "python": sys.version,
        "logical_cores": os.cpu_count(),
        "physical_cores": physical_cores(),
        "memory_gib": _meminfo_gib(),
        "data_root": str(data_root),
        "data_root_free_gib": round(disk.free / 2**30, 1) if disk else None,
        "packages": versions,
    }
    text = json.dumps(record, indent=2, sort_keys=True) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, args.out)  # whole or not at all: an interrupted write never truncates the record
    print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())

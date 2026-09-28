"""Record the workstation specification (Table 3.1 "Hardware and device"; Appendix A).

    python scripts/record_workstation.py               # writes environment/workstation.json
    python scripts/record_workstation.py --out FILE    # another machine: write elsewhere

Records the operating system, Python, CPU model, logical and physical core counts, memory and
disk of the data root, and the versions of the pinned packages. Commit the file with the
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

from pilot.provenance import machine_description, utc_now  # noqa: E402

PACKAGES = ("omnisafe", "safety-gymnasium", "mujoco", "gymnasium", "torch", "numpy", "pandas", "pyarrow", "pydantic", "pytest")


def _meminfo_gib() -> float | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / 1024 / 1024, 1)
    except OSError:
        return None
    return None


def _physical_cores() -> int | None:
    try:
        cores = set()
        phys = core = None
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("physical id"):
                    phys = line.split(":")[1].strip()
                elif line.startswith("core id"):
                    core = line.split(":")[1].strip()
                    cores.add((phys, core))
        return len(cores) or None
    except OSError:
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=REPO / "environment" / "workstation.json")
    args = parser.parse_args(argv)
    data_root = Path(os.environ.get("WSCL_DATA_ROOT", "/data"))
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
        "physical_cores": _physical_cores(),
        "memory_gib": _meminfo_gib(),
        "data_root": str(data_root),
        "data_root_free_gib": round(disk.free / 2**30, 1) if disk else None,
        "packages": versions,
    }
    out = args.out
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env bash
# Create the study's pinned Python environment (First Tasks, Role 1 steps 6 to 8; Table 3.1; Appendix A).
#
#   bash scripts/setup_env.sh            # first time, on the workstation: resolve, install, freeze, record
#   bash scripts/setup_env.sh --locked   # everyone else: install exactly environment/requirements.lock.txt
#   bash scripts/setup_env.sh --relock   # re-resolve and overwrite an existing lock (only by amendment)
#
# Requires Python 3.10 on Linux. Set PYTHON=/path/to/python3.10 if `python3.10` is not on PATH.
# Creates .venv/ in the repository root (ignored by Git).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3.10}"
VENV="${VENV:-$REPO/.venv}"
case "${1:-}" in
  "") MODE=resolve ;;
  --locked) MODE=locked ;;
  --relock) MODE=relock ;;
  *) echo "usage: bash scripts/setup_env.sh [--locked | --relock]" >&2; exit 2 ;;
esac
LOCK="$REPO/environment/requirements.lock.txt"
if [ "$MODE" = "resolve" ] && [ -f "$LOCK" ]; then
  echo "error: $LOCK exists; install it with --locked (re-resolving changes the pinned environment: --relock, by amendment)" >&2
  exit 1
fi
if [ "$MODE" = "locked" ] && [ ! -f "$LOCK" ]; then
  echo "error: $LOCK does not exist yet; the pilot owner creates it first (no argument)" >&2
  exit 1
fi

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "error: $PYTHON not found; install Python 3.10 or set PYTHON=/path/to/python3.10" >&2
  exit 1
fi
PYVER="$("$PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
if [ "$PYVER" != "3.10" ]; then
  echo "error: Python 3.10 is required (found $PYVER); see environment/requirements.in" >&2
  exit 1
fi
if [ "$(uname -s)" != "Linux" ]; then
  echo "warning: registered runs are made on Linux; OmniSafe does not officially support $(uname -s)" >&2
fi

"$PYTHON" -m venv --clear "$VENV"   # a fresh environment (First Tasks, Role 1 step 6)
"$VENV/bin/python" -m pip install --upgrade pip

if [ "$MODE" = "locked" ]; then
  # --no-deps: the lock lists every package, so pip must not resolve anything beyond it.
  "$VENV/bin/python" -m pip install --no-deps -r "$LOCK"
  "$VENV/bin/python" -m pip check
else
  "$VENV/bin/python" -m pip install -r "$REPO/environment/requirements.in"
  # Written beside the lock and moved into place only after every check below passes.
  trap 'rm -f "$LOCK.tmp"' EXIT
  "$VENV/bin/python" -m pip freeze --all --exclude-editable > "$LOCK.tmp"
fi

# Confirm the stack imports (OmniSafe installs Safety-Gymnasium, MuJoCo and PyTorch).
"$VENV/bin/python" - <<'EOF'
import importlib.metadata as md
import numpy, pandas, torch, mujoco, safety_gymnasium, omnisafe  # noqa: F401
for pkg in ("omnisafe", "safety-gymnasium", "mujoco", "gymnasium", "torch", "numpy", "pandas"):
    print(f"{pkg:18s} {md.version(pkg)}")
EOF

cd "$REPO"
if [ "$MODE" = "locked" ]; then
  # environment/workstation.json is the registered workstation (Table A.1); never overwrite it here.
  "$VENV/bin/python" scripts/record_workstation.py --out "$VENV/workstation.local.json" >/dev/null
  echo "wrote $VENV/workstation.local.json (this machine; environment/workstation.json is unchanged)"
else
  "$VENV/bin/python" scripts/record_workstation.py >/dev/null
  echo "wrote environment/workstation.json"
fi
if [ -f configs/omnisafe/SOURCE.json ]; then
  "$VENV/bin/python" scripts/copy_omnisafe_configs.py --check
else
  "$VENV/bin/python" scripts/copy_omnisafe_configs.py
fi
"$VENV/bin/python" -m pytest -q   # fast tests; `pytest -m slow` trains for about ten minutes
if [ "$MODE" != "locked" ]; then
  mv "$LOCK.tmp" "$LOCK"
  trap - EXIT
  echo "wrote environment/requirements.lock.txt"
fi
echo
echo "Environment ready. Activate with: source $VENV/bin/activate"
echo "Next: python scripts/determinism_check.py   (from a clean commit)"

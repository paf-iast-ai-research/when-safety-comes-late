"""HANDOVER.md section 10: every open-question key that code passes to a gate is a PENDING key.

``configs.registered.is_open`` raises KeyError for an unknown key, but only when that line runs; a
misspelt key in a rarely taken branch (a tie in rule 1, a censored adaptation step) would stay
hidden until then. This test reads the source of every package that gates results or runs and
checks each string-literal key passed to ``require_answered(``, ``open_keys(`` or ``is_open(``
(directly, as a starred tuple or list, or through a module-level string, or tuple, list, set or
frozenset of strings) against ``configs.registered.PENDING``. Packages that do not exist yet are
skipped.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from configs import registered as R

REPO = Path(__file__).resolve().parents[1]
PACKAGES = ("envs", "metrics", "studyb", "analysis", "pilot", "scripts")
GATE_FUNCTIONS = ("require_answered", "open_keys", "is_open")


def _called_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _module_constants(tree: ast.Module) -> dict[str, list[str]]:
    """Module-level ``NAME = "Q-a"`` (a string) or ``NAME = ("Q-a", "Q-b")`` (tuple, list, set or frozenset of strings)."""
    out: dict[str, list[str]] = {}
    for node in tree.body:
        targets: list[ast.expr] = []
        value = None
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        if isinstance(value, ast.Call) and _called_name(value.func) in ("frozenset", "tuple", "set") and value.args:
            value = value.args[0]
        strings: list[str] = []
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            strings = [value.value]
        elif isinstance(value, (ast.Tuple, ast.List, ast.Set)):
            strings = [e.value for e in value.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
            if len(strings) != len(value.elts):
                strings = []
        if strings:
            for target in targets:
                if isinstance(target, ast.Name):
                    out[target.id] = strings
    return out


def _strings(node: ast.expr, constants: dict[str, list[str]]) -> list[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.Starred):
        return _strings(node.value, constants)
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return [s for element in node.elts for s in _strings(element, constants)]
    if isinstance(node, ast.Name):
        return list(constants.get(node.id, []))
    return []  # a variable or an expression: checked at run time by R.is_open


def gate_keys(source: str) -> list[tuple[int, str, str]]:
    """(line, function, key) for every string key passed positionally to a gate function."""
    tree = ast.parse(source)
    constants = _module_constants(tree)
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _called_name(node.func) in GATE_FUNCTIONS:
            for arg in node.args:
                for key in _strings(arg, constants):
                    found.append((node.lineno, _called_name(node.func), key))
    return found


def _sources() -> list[Path]:
    files = []
    for package in PACKAGES:
        root = REPO / package
        if root.is_dir():
            files += [p for p in sorted(root.rglob("*.py")) if "__pycache__" not in p.parts]
    return files


def test_the_scanner_finds_every_form_of_key() -> None:
    source = '''
from pilot.errors import require_answered, open_keys
from configs import registered as R
GATES = ("Q-a", "Q-b")
OTHER: frozenset = frozenset({"Q-c"})
SINGLE = "Q-h"

def f(key):
    require_answered("Q-d", "Q-e", what="x")
    require_answered(*GATES, what="y")
    pilot.errors.require_answered(*("Q-f",), what="z")
    open_keys(*OTHER)
    R.is_open("Q-g")
    open_keys(SINGLE)
    is_open(key)                       # a variable: not resolvable here
    require_answered(*[k for k in GATES], what="w")   # an expression: not resolvable here
    something_else("Q-not-a-gate")
'''
    assert [k for _, _, k in gate_keys(source)] == ["Q-d", "Q-e", "Q-a", "Q-b", "Q-f", "Q-c", "Q-g", "Q-h"]


@pytest.mark.parametrize("path", _sources(), ids=lambda p: str(p.relative_to(REPO)))
def test_every_gate_key_in_the_code_is_a_pending_key(path: Path) -> None:
    unknown = [(line, func, key) for line, func, key in gate_keys(path.read_text(encoding="utf-8")) if key not in R.PENDING]
    assert not unknown, f"{path.relative_to(REPO)}: keys not in configs.registered.PENDING: {unknown}"


def test_the_scan_covers_the_pipeline() -> None:
    files = _sources()
    assert any(p.parent.name == "pilot" for p in files)  # the pipeline itself is always scanned
    # the pipeline's own gate through a module-level string constant (pilot/contracts.py)
    contracts_source = (REPO / "pilot" / "contracts.py").read_text(encoding="utf-8")
    assert "Q-selection-window" in [k for _, _, k in gate_keys(contracts_source)]
    # every package that exists contributes at least one file
    assert all(any(p.relative_to(REPO).parts[0] == package for p in files)
               for package in PACKAGES if (REPO / package).is_dir())

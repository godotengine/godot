#!/usr/bin/env python3
"""Validate Gaussian Splatting shader dependency wiring in SCons scripts.

This guard protects ISSUE-034 by asserting the build graph keeps generated
shader headers connected to their GLSL sources/includes and to module objects.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_ROOT = REPO_ROOT / "modules" / "gaussian_splatting"
ROOT_SCSUB = MODULE_ROOT / "SCsub"
COMPUTE_SCSUB = MODULE_ROOT / "compute" / "SCsub"
SHADERS_SCSUB = MODULE_ROOT / "shaders" / "SCsub"


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Failed reading '{path}': {exc}") from exc


def _parse_module(text: str, path: Path, failures: list[str]) -> ast.Module | None:
    try:
        return ast.parse(text, filename=path.as_posix())
    except SyntaxError as exc:
        failures.append(f"{path.as_posix()} could not be parsed as Python: {exc.msg} (line {exc.lineno})")
        return None


def _iter_calls(module: ast.AST):
    for node in ast.walk(module):
        if isinstance(node, ast.Call):
            yield node


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _string_literal(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _node_has_name(node: ast.AST | None, name: str) -> bool:
    if node is None:
        return False
    return any(isinstance(inner, ast.Name) and inner.id == name for inner in ast.walk(node))


def _node_has_string(node: ast.AST | None, value: str) -> bool:
    if node is None:
        return False
    return any(_string_literal(inner) == value for inner in ast.walk(node))


def _node_has_string_fragment(node: ast.AST | None, fragment: str) -> bool:
    if node is None:
        return False
    for inner in ast.walk(node):
        value = _string_literal(inner)
        if value is not None and fragment in value:
            return True
    return False


def _node_has_glob_pattern(node: ast.AST | None, pattern: str) -> bool:
    if node is None:
        return False
    for inner in ast.walk(node):
        if not isinstance(inner, ast.Call):
            continue
        if _call_name(inner) != "Glob":
            continue
        if inner.args and _string_literal(inner.args[0]) == pattern:
            return True
    return False


def _collect_target_names(target: ast.AST) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for child in target.elts:
            names.update(_collect_target_names(child))
        return names
    return set()


def _assignment_target_names(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Assign):
        names: set[str] = set()
        for target in node.targets:
            names.update(_collect_target_names(target))
        return names
    if isinstance(node, ast.AnnAssign):
        return _collect_target_names(node.target)
    return set()


def _names_assigned_from_glob(module: ast.Module, glob_pattern: str) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(module):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if value is None or not _node_has_glob_pattern(value, glob_pattern):
            continue
        names.update(_assignment_target_names(node))
    return names


def _has_assignment_from_sconscript(module: ast.Module, variable: str, scsub_path: str) -> bool:
    for node in ast.walk(module):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        if variable not in _assignment_target_names(node):
            continue
        value = node.value
        if not isinstance(value, ast.Call):
            continue
        if _call_name(value) != "SConscript":
            continue
        if value.args and _string_literal(value.args[0]) == scsub_path:
            return True
    return False


def _depends_operands(call: ast.Call) -> tuple[ast.AST | None, ast.AST | None]:
    target = call.args[0] if len(call.args) > 0 else None
    dependency = call.args[1] if len(call.args) > 1 else None
    for keyword in call.keywords:
        if keyword.arg == "target" and target is None:
            target = keyword.value
        if keyword.arg in {"dependency", "dependencies", "deps"} and dependency is None:
            dependency = keyword.value
    return target, dependency


def _iter_depends_calls(module: ast.Module):
    for call in _iter_calls(module):
        if _call_name(call) == "Depends":
            yield call


def _has_module_source_depends(module: ast.Module) -> bool:
    for call in _iter_depends_calls(module):
        target, dependency = _depends_operands(call)
        if _node_has_name(target, "module_sources") and _node_has_name(dependency, "generated_shader_headers"):
            return True
    return False


def _has_shader_dependency_depends_call(
    module: ast.Module,
    include_glob_pattern: str,
    include_var_names: set[str],
) -> bool:
    for call in _iter_depends_calls(module):
        target, dependency = _depends_operands(call)
        if target is None or dependency is None:
            continue
        target_ok = _node_has_name(target, "glsl_files") and _node_has_string_fragment(target, ".gen.h")
        if not target_ok:
            continue
        builder_ok = _node_has_string(dependency, "#glsl_builders.py")
        if not builder_ok:
            continue
        include_ok = _node_has_glob_pattern(dependency, include_glob_pattern) or any(
            _node_has_name(dependency, var_name) for var_name in include_var_names
        )
        if include_ok:
            return True
    return False


def _check_root_contract(module: ast.Module, failures: list[str]) -> None:
    context = ROOT_SCSUB.as_posix()
    if not _has_assignment_from_sconscript(module, "compute_generated_headers", "compute/SCsub"):
        failures.append(f"{context}: missing compute_generated_headers = SConscript('compute/SCsub')")
    if not _has_assignment_from_sconscript(module, "shader_generated_headers", "shaders/SCsub"):
        failures.append(f"{context}: missing shader_generated_headers = SConscript('shaders/SCsub')")
    if not _has_module_source_depends(module):
        failures.append(
            f"{context}: missing Depends() edge linking module_sources to generated_shader_headers"
        )


def _check_subshader_contract(
    module: ast.Module,
    path: Path,
    include_glob_pattern: str,
    include_hint: str,
) -> list[str]:
    failures = []
    include_var_names = _names_assigned_from_glob(module, include_glob_pattern)
    if not include_var_names and not _node_has_glob_pattern(module, include_glob_pattern):
        failures.append(
            f"{path.as_posix()}: missing Glob('{include_glob_pattern}') for {include_hint} include dependencies"
        )
    if not _has_shader_dependency_depends_call(module, include_glob_pattern, include_var_names):
        failures.append(
            f"{path.as_posix()}: missing Depends() contract linking generated headers to glsl_files, "
            f"'{include_hint}' includes, and '#glsl_builders.py'"
        )
    return failures


def main() -> int:
    failures: list[str] = []

    root_scsub = _read(ROOT_SCSUB)
    compute_scsub = _read(COMPUTE_SCSUB)
    shaders_scsub = _read(SHADERS_SCSUB)

    root_module = _parse_module(root_scsub, ROOT_SCSUB, failures)
    compute_module = _parse_module(compute_scsub, COMPUTE_SCSUB, failures)
    shaders_module = _parse_module(shaders_scsub, SHADERS_SCSUB, failures)

    if root_module is not None:
        _check_root_contract(root_module, failures)
    if compute_module is not None:
        failures.extend(
            _check_subshader_contract(
                compute_module,
                COMPUTE_SCSUB,
                "../shaders/includes/*.glsl",
                "shared shader",
            )
        )
    if shaders_module is not None:
        failures.extend(
            _check_subshader_contract(
                shaders_module,
                SHADERS_SCSUB,
                "includes/*.glsl",
                "local shader",
            )
        )

    if failures:
        print("[shader-dependency-check] FAILED")
        for failure in failures:
            print(f"[shader-dependency-check] - {failure}")
        return 1

    print("[shader-dependency-check] PASSED")
    print("[shader-dependency-check] Shader include/build graph contracts are intact.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

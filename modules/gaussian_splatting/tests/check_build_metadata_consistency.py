#!/usr/bin/env python3
"""Validate Gaussian Splatting build metadata consistency.

This guard keeps SCons (authoritative build graph), CMake (IDE metadata),
and module docs contract metadata in sync.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_ROOT = REPO_ROOT / "modules" / "gaussian_splatting"
SCSUB_PATH = MODULE_ROOT / "SCsub"
CMAKE_PATH = MODULE_ROOT / "CMakeLists.txt"
CONFIG_PATH = MODULE_ROOT / "config.py"

CMAKE_CPP_GLOB_RE = re.compile(r'"(?P<directory>[A-Za-z0-9_./-]+)/\*\.cpp"')
LITERAL_ADD_SOURCE_DIR_RE = re.compile(
    r'add_source_files\([^,]+,\s*"(?P<directory>[A-Za-z0-9_./-]+)/\*\.cpp"\)'
)


class ContractError(RuntimeError):
    """Raised when expected metadata contracts are malformed."""


def _load_ast(path: Path) -> ast.Module:
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except OSError as exc:
        raise ContractError(f"Failed reading '{path}': {exc}") from exc
    except SyntaxError as exc:
        raise ContractError(f"Failed parsing '{path}': {exc}") from exc


def _parse_string_collection(expr: ast.AST, context: str) -> list[str]:
    if not isinstance(expr, (ast.List, ast.Tuple)):
        raise ContractError(f"{context} must return a list/tuple of string literals.")

    values: list[str] = []
    for element in expr.elts:
        if not isinstance(element, ast.Constant) or not isinstance(element.value, str):
            raise ContractError(f"{context} must contain only string literals.")
        value = element.value.strip()
        if not value:
            raise ContractError(f"{context} contains an empty string entry.")
        values.append(value)
    return values


def _parse_assigned_string_collection(module_ast: ast.Module, variable_name: str, path: Path) -> list[str]:
    for node in module_ast.body:
        if not isinstance(node, ast.Assign):
            continue

        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == variable_name:
                return _parse_string_collection(node.value, f"{path.name}:{variable_name}")

    raise ContractError(f"Missing required assignment '{variable_name}' in '{path}'.")


def _find_function(module_ast: ast.Module, function_name: str, path: Path) -> ast.FunctionDef:
    for node in module_ast.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return node
    raise ContractError(f"Missing function '{function_name}' in '{path}'.")


def _parse_returned_string_collection(function_node: ast.FunctionDef, path: Path) -> list[str]:
    for node in function_node.body:
        if isinstance(node, ast.Return):
            return _parse_string_collection(node.value, f"{path.name}:{function_node.name}()")
    raise ContractError(f"Function '{function_node.name}' in '{path}' must include a return statement.")


def _parse_returned_string_literal(function_node: ast.FunctionDef, path: Path) -> str:
    for node in function_node.body:
        if isinstance(node, ast.Return):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                value = node.value.value.strip()
                if value:
                    return value
            raise ContractError(f"{path.name}:{function_node.name}() must return a non-empty string literal.")
    raise ContractError(f"Function '{function_node.name}' in '{path}' must include a return statement.")


def _parse_scons_cpp_dirs(scsub_ast: ast.Module, scsub_text: str) -> set[str]:
    source_dirs = set(_parse_assigned_string_collection(scsub_ast, "source_directories", SCSUB_PATH))

    # Capture explicit optional directory globs not included in source_directories.
    for match in LITERAL_ADD_SOURCE_DIR_RE.finditer(scsub_text):
        source_dirs.add(match.group("directory"))

    return source_dirs


def _parse_cmake_cpp_glob_dirs(cmake_text: str) -> set[str]:
    directories: set[str] = set()
    for match in CMAKE_CPP_GLOB_RE.finditer(cmake_text):
        value = match.group("directory").replace("\\", "/").strip("/")
        if value:
            directories.add(value)
    return directories


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ContractError(f"Failed reading '{path}': {exc}") from exc


def _has_cpp_files(directory: Path) -> bool:
    return any(directory.glob("*.cpp"))


def _validate_doc_contract(config_ast: ast.Module) -> list[str]:
    failures: list[str] = []

    doc_path_function = _find_function(config_ast, "get_doc_path", CONFIG_PATH)
    doc_classes_function = _find_function(config_ast, "get_doc_classes", CONFIG_PATH)

    doc_path = _parse_returned_string_literal(doc_path_function, CONFIG_PATH)
    doc_dir = MODULE_ROOT / doc_path
    if not doc_dir.is_dir():
        failures.append(
            f"config.py:get_doc_path() points to '{doc_path}', but '{doc_dir.relative_to(REPO_ROOT)}' does not exist."
        )
        return failures

    doc_classes = _parse_returned_string_collection(doc_classes_function, CONFIG_PATH)
    missing_xml = [class_name for class_name in doc_classes if not (doc_dir / f"{class_name}.xml").is_file()]
    if missing_xml:
        failures.append(
            "Missing doc XML files for get_doc_classes(): "
            + ", ".join(f"{class_name}.xml" for class_name in missing_xml)
        )

    xml_files = list(doc_dir.glob("*.xml"))
    if not xml_files:
        failures.append(
            f"Doc directory '{doc_dir.relative_to(REPO_ROOT)}' must contain at least one XML class reference file."
        )

    return failures


def _validate_build_metadata_contract(scsub_ast: ast.Module, scsub_text: str, cmake_text: str) -> list[str]:
    failures: list[str] = []

    scons_cpp_dirs = _parse_scons_cpp_dirs(scsub_ast, scsub_text)
    cmake_cpp_dirs = _parse_cmake_cpp_glob_dirs(cmake_text)

    missing_cmake_dirs = sorted(
        directory for directory in cmake_cpp_dirs if not (MODULE_ROOT / directory).is_dir()
    )
    if missing_cmake_dirs:
        failures.append(
            "CMake references missing source directories: "
            + ", ".join(missing_cmake_dirs)
        )

    missing_in_cmake = sorted(directory for directory in scons_cpp_dirs if directory not in cmake_cpp_dirs)
    if missing_in_cmake:
        failures.append(
            "SCsub C++ source directories missing from CMake globs: "
            + ", ".join(missing_in_cmake)
        )

    extra_cmake_dirs = sorted(
        directory
        for directory in cmake_cpp_dirs
        if directory not in scons_cpp_dirs
        and (MODULE_ROOT / directory).is_dir()
        and _has_cpp_files(MODULE_ROOT / directory)
    )
    if extra_cmake_dirs:
        failures.append(
            "CMake globs include C++ source directories not declared in SCsub source lists: "
            + ", ".join(extra_cmake_dirs)
        )

    if '"register_types.cpp"' not in scsub_text:
        failures.append("SCsub no longer includes register_types.cpp; module registration would be dropped.")

    if '"register_types.cpp"' not in cmake_text:
        failures.append("CMake no longer includes register_types.cpp in IDE metadata.")

    return failures


def main() -> int:
    try:
        scsub_text = _read_text(SCSUB_PATH)
        cmake_text = _read_text(CMAKE_PATH)
        scsub_ast = _load_ast(SCSUB_PATH)
        config_ast = _load_ast(CONFIG_PATH)

        failures = []
        failures.extend(_validate_build_metadata_contract(scsub_ast, scsub_text, cmake_text))
        failures.extend(_validate_doc_contract(config_ast))
    except ContractError as exc:
        print(f"[build-metadata-check] ERROR: {exc}")
        return 1

    if failures:
        print("[build-metadata-check] FAILED")
        for failure in failures:
            print(f"[build-metadata-check] - {failure}")
        return 1

    print("[build-metadata-check] PASSED")
    print("[build-metadata-check] SCsub/CMake/doc metadata contracts are aligned.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

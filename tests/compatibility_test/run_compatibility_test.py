#!/usr/bin/env python3
from __future__ import annotations

import itertools
import json
import os
import pathlib
import subprocess
import urllib.request
from typing import Any

PROJECT_PATH = pathlib.Path(__file__).parent.resolve().joinpath("godot")
CLASS_METHODS_FILE = PROJECT_PATH.joinpath("class_methods.txt")
BUILTIN_METHODS_FILE = PROJECT_PATH.joinpath("builtin_methods.txt")
UTILITY_FUNCTIONS_FILE = PROJECT_PATH.joinpath("utility_functions.txt")


def download_gdextension_api(reftag: str) -> dict[str, Any]:
    with urllib.request.urlopen(
        f"https://raw.githubusercontent.com/godotengine/godot-cpp/godot-{reftag}/gdextension/extension_api.json"
    ) as f:
        gdextension_api_json: dict[str, Any] = json.load(f)
    return gdextension_api_json


def generate_test_data_files(reftag: str):
    """
    Parses methods specified in given Godot version into a form readable by the compatibility checker GDExtension.
    """
    gdextension_reference_json = download_gdextension_api(reftag)

    with open(CLASS_METHODS_FILE, "w") as classes_file:
        classes_file.writelines(
            [
                f"{klass['name']} {func['name']} {func['hash']}\n"
                for (klass, func) in itertools.chain(
                    (
                        (klass, method)
                        for klass in gdextension_reference_json["classes"]
                        for method in klass.get("methods", [])
                        if not method.get("is_virtual")
                    ),
                )
            ]
        )

    variant_types: dict[str, int] | None = None
    for global_enum in gdextension_reference_json["global_enums"]:
        if global_enum.get("name") != "Variant.Type":
            continue
        variant_types = {
            variant_type.get("name").removeprefix("TYPE_").lower().replace("_", ""): variant_type.get("value")
            for variant_type in global_enum.get("values")
        }

    if not variant_types:
        return

    with open(BUILTIN_METHODS_FILE, "w") as f:
        f.writelines(
            [
                f"{variant_types[klass['name'].lower()]} {func['name']} {func['hash']}\n"
                for (klass, func) in itertools.chain(
                    (
                        (klass, method)
                        for klass in gdextension_reference_json["builtin_classes"]
                        for method in klass.get("methods", [])
                    ),
                )
            ]
        )

    with open(UTILITY_FUNCTIONS_FILE, "w") as f:
        f.writelines([f"{func['name']} {func['hash']}\n" for func in gdextension_reference_json["utility_functions"]])


def has_compatibility_test_failed(errors: str) -> bool:
    """
    Checks if provided errors are related to the compatibility test.

    Makes sure that test won't fail on unrelated account (for example editor misconfiguration).
    """
    compatibility_errors = [
        "Error loading extension",
        "Failed to load interface method",
        'Parameter "mb" is null.',
        'Parameter "bfi" is null.',
        "Method bind not found:",
        "Utility function not found:",
        "has changed and no compatibility fallback has been provided",
        "Failed to open file `builtin_methods.txt`",
        "Failed to open file `class_methods.txt`",
        "Failed to open file `utility_functions.txt`",
        "Failed to open file `platform_methods.txt`",
        "Outcome = FAILURE",
    ]

    return any(compatibility_error in errors for compatibility_error in compatibility_errors)


def process_compatibility_test(proc: subprocess.Popen[bytes], timeout: int = 5) -> str | None:
    """
    Returns the stderr output as a string, if any.

    Terminates test if nothing has been written to stdout/stderr for specified time.
    """
    errors = bytearray()

    while True:
        try:
            _out, err = proc.communicate(timeout=timeout)
            if err:
                errors.extend(err)
        except subprocess.TimeoutExpired:
            proc.kill()
            _out, err = proc.communicate()
            if err:
                errors.extend(err)
            break

    return errors.decode("utf-8") if errors else None


def compatibility_check(godot4_bin: str) -> bool:
    """
    Checks if methods specified for previous Godot versions can be properly loaded with
    the latest Godot4 binary.
    """
    # A bit crude albeit working solution – use stderr to check for compatibility-related errors.
    proc = subprocess.Popen(
        [godot4_bin, "--headless", "-e", "--path", PROJECT_PATH],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if (errors := process_compatibility_test(proc)) and has_compatibility_test_failed(errors):
        print(f"Compatibility test failed. Errors:\n {errors}")
        return False
    return True


if __name__ == "__main__":
    godot4_bin = os.environ["GODOT4_BIN"]
    reftags = os.environ["REFTAGS"].split(",")
    is_success = True
    for reftag in reftags:
        generate_test_data_files(reftag)
        if not compatibility_check(godot4_bin):
            print(f"Compatibility test against Godot{reftag} failed")
            is_success = False

    if not is_success:
        exit(1)

#!/usr/bin/env python3

import argparse
import io
import os
import os.path
import pathlib
import re
import textwrap
from typing import Dict, List, Union

matches: Dict[str, List[Dict[str, int]]] = {}
current_path = pathlib.Path()
tab_width = 4


def parse_gdscript(symbol: str, code: str, path: str):
    last_found_symbol_index = 0
    found_index = []

    while True:
        last_found_symbol_index = code.find(symbol, last_found_symbol_index)
        if last_found_symbol_index == -1:
            break

        found_index.append(last_found_symbol_index)
        last_found_symbol_index += len(symbol)

    file_matches = []
    line = 1
    column = 1
    index = 0
    match_until = -1
    current_match = {
        "start_line": -1,
        "start_column": -1,
        "end_line": -1,
        "end_column": -1,
    }

    for i in range(0, len(code)):
        character = code[i]

        if match_until >= 0 and i == match_until:
            current_match["end_line"] = line
            current_match["end_column"] = column

            file_matches.append(current_match)
            current_match = {
                "start_line": -1,
                "start_column": -1,
                "end_line": -1,
                "end_column": -1,
            }

            match_until = -1
        else:
            if i in found_index:
                current_match["start_line"] = line
                current_match["start_column"] = column
                current_match["end_line"] = -1
                current_match["end_column"] = -1
                match_until = i + len(symbol)

        if character == "\n":
            line += 1
            column = 1
        elif character == "\t":
            column += tab_width
        else:
            column += 1

        index += 1

    if len(file_matches) > 0:
        matches[path] = []

    for file_match in file_matches:
        matches[path].append(file_match)


def parse_gdscript_file(symbol: str, path: pathlib.Path):
    with open(path, "r", encoding="utf-8") as f:
        file_contents = f.read()
        parse_gdscript(symbol, file_contents, get_project_path(path))


def parse_packed_scene(symbol: str, path: pathlib.Path):
    gdscript_property_regex = re.compile(
        r"^\[sub_resource type=\"GDScript\" id=\"(?P<id>.+?)\"]\nresource_name = \"(?P<name>.+?)\"\nscript/source = \"(?P<script>.+?)\"\n",
        re.M | re.S,
    )
    with open(path, "r", encoding="utf-8") as f:
        file_contents = f.read()
        for gdscript_property_match in gdscript_property_regex.finditer(file_contents):
            parse_gdscript(
                symbol,
                gdscript_property_match.group("script"),
                f"{get_project_path(path)}::{gdscript_property_match.group('id')}",
            )


def get_project_path(from_path: pathlib.Path) -> str:
    project_dir_path = find_closest_project_dir_path(current_path)
    if project_dir_path is None:
        raise Exception("Could not find closest project dir")
    return f"res://{str(from_path.relative_to(project_dir_path))}"


def find_closest_project_dir_path(
    path: pathlib.Path,
) -> Union[pathlib.Path, None]:
    if path.is_dir():
        parent_path = path
    else:
        parent_path = path.parent
        if path == parent_path:
            return None

    project_godot_file_path = parent_path.joinpath("project.godot")

    if project_godot_file_path.is_file():
        return parent_path

    return find_closest_project_dir_path(parent_path)


def render_matches():
    indent_size = 4
    indent = " " * indent_size

    root_io = io.StringIO()

    root_io.write("[output]\n")
    root_io.write("; This is a pure text search result.\n")
    root_io.write("; Please take time to filter out false positive results\n")
    root_io.write("; ...and to track missing ones.\n")
    root_io.write("matches={\n")

    for key in matches.keys():
        match_file_io = io.StringIO()
        match_file_io.write(f'"{key}": [\n')

        for match_entry in matches[key]:
            match_entry_io = io.StringIO()
            match_entry_io.write("{\n")

            match_entry_keys_io = io.StringIO()
            match_entry_keys_io.write(f'"start_line": {match_entry["start_line"]},\n')
            match_entry_keys_io.write(f'"start_column": {match_entry["start_column"]},\n')
            match_entry_keys_io.write(f'"end_line": {match_entry["end_line"]},\n')
            match_entry_keys_io.write(f'"end_column": {match_entry["end_column"]},\n')
            match_entry_io.write(textwrap.indent(match_entry_keys_io.getvalue(), indent))

            match_entry_io.write("},\n")
            match_file_io.write(textwrap.indent(match_entry_io.getvalue(), indent))

        match_file_io.write("],\n")
        root_io.write(textwrap.indent(match_file_io.getvalue(), indent))

    root_io.write("}\n")

    print(root_io.getvalue(), end="")


def main(symbol: str):
    for root, _, files in os.walk(current_path):
        for filename in files:
            file_path = pathlib.Path(os.path.join(root, filename))
            if file_path.suffix == ".gd":
                parse_gdscript_file(symbol, file_path)
            elif file_path.suffix == ".tscn":
                parse_packed_scene(symbol, file_path)

    render_matches()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find easily text references in files for the Godot GDScript Refactor Rename test case"
    )
    parser.add_argument("symbol", help="Symbol to search")
    parser.add_argument("-t", "--tab-width", dest="tab_width", help="Tab width", default=4, type=int)
    parser.add_argument(
        "-d",
        "--directory",
        help="Directory to search symbols in",
        default=".",
        type=pathlib.Path,
    )
    args = parser.parse_args()

    current_path = args.directory.resolve()
    tab_width = args.tab_width

    main(args.symbol)

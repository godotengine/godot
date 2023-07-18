#!/bin/bash
set -uo pipefail

if [ ! -f "version.py" ]; then
  echo "Warning: This script is intended to be run from the root of the Godot repository."
  echo "Some of the paths checks may not work as intended from a different folder."
fi

if [ $# != 1 ]; then
  echo "Usage: @0 <path-to-godot-executable>"
fi

has_problems=0

make_annotation()
{
  local title=$1
  local body=$2
  local type=$3
  local file=$4
  if [ ! -v GITHUB_OUTPUT ]; then
    echo "$title"
    echo "$body"
  else
    body="$(awk 1 ORS='%0A' - <<<"$body")"
    echo "::$type file=$file,title=$title ::$body"
  fi
}

while read -r file; do
    reference_file="$(mktemp)"
    validate="$(mktemp)"
    validation_output="$(mktemp)"
    allowed_errors="$(mktemp)"

    # Download the reference extension_api.json
    reference_tag="$(basename -s .expected "$file")"
    wget -qcO "$reference_file" "https://raw.githubusercontent.com/godotengine/godot-cpp/godot-$reference_tag/gdextension/extension_api.json"
    # Validate the current API against the reference
    "$1" --headless --validate-extension-api "$reference_file" 2>&1 | tee "$validate" | awk '!/^Validate extension JSON:/' - || true
    # Collect the expected and actual validation errors
    awk '/^Validate extension JSON:/' - < "$validate" | sort > "$validation_output"
    awk '/^Validate extension JSON:/' - < "$file" | sort > "$allowed_errors"

    # Differences between the expected and actual errors
    new_validation_error="$(comm "$validation_output" "$allowed_errors" -23)"
    obsolete_validation_error="$(comm "$validation_output" "$allowed_errors" -13)"

    if [ -n "$obsolete_validation_error" ]; then
        make_annotation "The following validation errors no longer occur (compared to $reference_tag):" "$obsolete_validation_error" warning "$file"
    fi
    if [ -n "$new_validation_error" ]; then
        make_annotation "Compatibility to $reference_tag is broken in the following ways:" "$new_validation_error" error "$file"
        has_problems=1
    fi

    rm -f "$reference_file" "$validate" "$validation_output" "$allowed_errors"
done <<< "$(find "$( dirname -- "$( dirname -- "${BASH_SOURCE[0]//\.\//}" )" )/extension_api_validation/" -name "*.expected")"

exit $has_problems

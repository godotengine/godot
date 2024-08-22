#!/bin/bash
set -o pipefail

if [ ! -f "version.py" ]; then
  echo "Warning: This script is intended to be run from the root of the Godot repository."
  echo "Some of the paths checks may not work as intended from a different folder."
fi

if [ $# != 1 ]; then
  echo "Usage: @0 <path-to-godot-executable>"
  exit 1
fi

api_validation_dir="$( dirname -- "$( dirname -- "${BASH_SOURCE[0]//\.\//}" )" )/extension_api_validation/"

has_problems=0
warn_extra=0
reference_tag=""
expected_errors=""

make_annotation()
{
  local title=$1
  local body=$2
  local type=$3
  local file=$4
  if [[ "$GITHUB_OUTPUT" == "" ]]; then
    echo "$title"
    echo "$body"
  else
    body="$(awk 1 ORS='%0A' - <<<"$body")"
    echo "::$type file=$file,title=$title ::$body"
  fi
}

get_expected_output()
{
  local parts=()
  IFS='_' read -ra parts <<< "$(basename -s .expected "$1")"

  if [[ "${#parts[@]}" == "2" ]]; then
    cat "$1" >> "$expected_errors"
    get_expected_output "$(find "$api_validation_dir" -name "${parts[1]}*.expected")"
    reference_tag="${parts[0]}"
    warn_extra=0
  else
    cat "$1" >> "$expected_errors"
    reference_tag="${parts[0]}"
    warn_extra=1
  fi
}

while read -r file; do
    reference_file="$(mktemp)"
    validate="$(mktemp)"
    validation_output="$(mktemp)"
    allowed_errors="$(mktemp)"
    expected_errors="$(mktemp)"
    get_expected_output "$file"

    # Download the reference extension_api.json
    wget -nv --retry-on-http-error=503 --tries=5 --timeout=60 -cO "$reference_file" "https://raw.githubusercontent.com/godotengine/godot-cpp/godot-$reference_tag/gdextension/extension_api.json" || has_problems=1
    # Validate the current API against the reference
    "$1" --headless --validate-extension-api "$reference_file" 2>&1 | tee "$validate" | awk '!/^Validate extension JSON:/' - || true
    # Collect the expected and actual validation errors
    awk '/^Validate extension JSON:/' - < "$validate" | sort > "$validation_output"
    awk '/^Validate extension JSON:/' - < "$expected_errors" | sort > "$allowed_errors"

    # Differences between the expected and actual errors
    new_validation_error="$(comm -23 "$validation_output" "$allowed_errors")"
    obsolete_validation_error="$(comm -13 "$validation_output" "$allowed_errors")"

    if [ -n "$obsolete_validation_error" ] && [ "$warn_extra" = "1" ]; then
        make_annotation "The following validation errors no longer occur (compared to $reference_tag):" "$obsolete_validation_error" warning "$file"
    fi
    if [ -n "$new_validation_error" ]; then
        make_annotation "Compatibility to $reference_tag is broken in the following ways:" "$new_validation_error" error "$file"
        has_problems=1
    fi

    rm -f "$reference_file" "$validate" "$validation_output" "$allowed_errors" "$expected_errors"
done <<< "$(find "$api_validation_dir" -name "*.expected")"

exit $has_problems

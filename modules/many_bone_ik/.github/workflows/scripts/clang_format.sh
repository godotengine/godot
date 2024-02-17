#!/usr/bin/env bash

# This script runs clang-format and fixes copyright headers on all relevant files in the repo.
# This is the primary script responsible for fixing style violations.

set -uo pipefail
IFS=$'\n\t'

CLANG_FORMAT_FILE_EXTS=(".c" ".h" ".cpp" ".hpp" ".cc" ".hh" ".cxx" ".m" ".mm" ".inc" ".java" ".glsl")

# Loops through all text files tracked by Git.
git grep -zIl '' |
while IFS= read -rd '' f; do
    # Exclude 3rd party files.
    if [[ "$f" == "glad"* ]]; then
        continue
    elif [[ "$f" == "godot-cpp"* ]]; then
        continue
    elif [[ "$f" == "thirdparty"* ]]; then
        continue
    elif [[ "$f" == "gradle"* ]]; then
        continue
    elif [[ "$f" == "build"* ]]; then
        continue
    elif [[ "$f" == "android"* ]]; then
        continue
    elif [[ "$f" == ".github"* ]]; then
        continue
    fi
    for extension in ${CLANG_FORMAT_FILE_EXTS[@]}; do
        if [[ "$f" == *"$extension" ]]; then
            # Run clang-format.
            clang-format -i "$f"
        fi
    done
done

git diff > patch.patch

# If no patch has been generated all is OK, clean up, and exit.
if [ ! -s patch.patch ] ; then
    printf "Files in this commit comply with the clang-format style rules.\n"
    rm -f patch.patch
    exit 0
fi

# A patch has been created, notify the user, clean up, and exit.
printf "\n*** The following differences were found between the code "
printf "and the formatting rules:\n\n"
cat patch.patch
printf "\n*** Aborting, please fix your commit(s) with 'git commit --amend' or 'git rebase -i <hash>'\n"
rm -f patch.patch
exit 1

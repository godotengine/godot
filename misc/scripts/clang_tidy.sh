#!/usr/bin/env bash

# This script runs clang-tidy on all relevant files in the repo.
# This is more thorough than clang-format and thus slower; it should only be run manually.

set -uo pipefail

# Loops through all code files tracked by Git.
git ls-files '*.c' '*.h' '*.cpp' '*.hpp' '*.cc' '*.hh' '*.cxx' '*.m' '*.mm' '*.inc' '*.java' '*.glsl' |
while read -r f; do
    # Exclude some files.
    if [[ "$f" == "thirdparty"* ]]; then
        continue
    elif [[ "$f" == "platform/android/java/lib/src/com/google"* ]]; then
        continue
    elif [[ "$f" == *"-so_wrap."* ]]; then
        continue
    fi

    # Run clang-tidy.
    clang-tidy --quiet --fix "$f" &> /dev/null

    # Run clang-format. This also fixes the output of clang-tidy.
    clang-format --Wno-error=unknown -i "$f"
done

diff=$(git diff --color)

# If no patch has been generated all is OK, clean up, and exit.
if [ -z "$diff" ] ; then
    printf "Files in this commit comply with the clang-tidy style rules.\n"
    exit 0
fi

# A patch has been created, notify the user, clean up, and exit.
printf "\n*** The following changes have been made to comply with the formatting rules:\n\n"
echo "$diff"
printf "\n*** Please fix your commit(s) with 'git commit --amend' or 'git rebase -i <hash>'\n"
exit 1

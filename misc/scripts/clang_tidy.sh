#!/usr/bin/env bash

# This script runs clang-tidy on all relevant files in the repo.
# This is more thorough than clang-format and thus slower; it should only be run manually.

set -uo pipefail

# Loops through all code files tracked by Git.
git ls-files -- '*.c' '*.h' '*.cpp' '*.hpp' '*.cc' '*.hh' '*.cxx' '*.m' '*.mm' '*.inc' '*.java' '*.glsl' \
                ':!:.git/*' ':!:thirdparty/*' ':!:platform/android/java/lib/src/com/google/*' ':!:*-so_wrap.*' |
while read -r f; do
    # Run clang-tidy.
    clang-tidy --quiet --fix "$f" &> /dev/null

    # Run clang-format. This also fixes the output of clang-tidy.
    clang-format --Wno-error=unknown -i "$f"
done

diff=$(git diff --color)

# If no diff has been generated all is OK, clean up, and exit.
if [ -z "$diff" ] ; then
    printf "Files in this commit comply with the clang-tidy style rules.\n"
    exit 0
fi

# A diff has been created, notify the user, clean up, and exit.
printf "\n*** The following changes have been made to comply with the formatting rules:\n\n"
echo "$diff"
printf "\n*** Please fix your commit(s) with 'git commit --amend' or 'git rebase -i <hash>'\n"
exit 1

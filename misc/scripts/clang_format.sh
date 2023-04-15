#!/usr/bin/env bash

# This script runs clang-format and fixes copyright headers on all relevant files in the repo.
# This is the primary script responsible for fixing style violations.

set -uo pipefail

# Loops through all code files tracked by Git.
git ls-files -- '*.c' '*.h' '*.cpp' '*.hpp' '*.cc' '*.hh' '*.cxx' '*.m' '*.mm' '*.inc' '*.java' '*.glsl' \
                ':!:.git/*' ':!:thirdparty/*' ':!:*/thirdparty/*' ':!:platform/android/java/lib/src/com/google/*' \
                ':!:*-so_wrap.*' ':!:tests/python_build/*' |
while read -r f; do
    # Run clang-format.
    clang-format --Wno-error=unknown -i "$f"

    # Fix copyright headers, but not all files get them.
    if [[ "$f" == *"inc" ]]; then
        continue
    elif [[ "$f" == *"glsl" ]]; then
        continue
    elif [[ "$f" == "platform/android/java/lib/src/org/godotengine/godot/gl/GLSurfaceView"* ]]; then
        continue
    elif [[ "$f" == "platform/android/java/lib/src/org/godotengine/godot/gl/EGLLogWrapper"* ]]; then
        continue
    elif [[ "$f" == "platform/android/java/lib/src/org/godotengine/godot/utils/ProcessPhoenix"* ]]; then
        continue
    fi

    python misc/scripts/copyright_headers.py "$f"
done

diff=$(git diff --color)

# If no diff has been generated all is OK, clean up, and exit.
if [ -z "$diff" ] ; then
    printf "\e[1;32m*** Files in this commit comply with the clang-format style rules.\e[0m\n"
    exit 0
fi

# A diff has been created, notify the user, clean up, and exit.
printf "\n\e[1;33m*** The following changes must be made to comply with the formatting rules:\e[0m\n\n"
# Perl commands replace trailing spaces with `·` and tabs with `<TAB>`.
printf "$diff\n" | perl -pe 's/(.*[^ ])( +)(\e\[m)$/my $spaces="·" x length($2); sprintf("$1$spaces$3")/ge' | perl -pe 's/(.*[^\t])(\t+)(\e\[m)$/my $tabs="<TAB>" x length($2); sprintf("$1$tabs$3")/ge'

printf "\n\e[1;91m*** Please fix your commit(s) with 'git commit --amend' or 'git rebase -i <hash>'\e[0m\n"
exit 1

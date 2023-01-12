#!/usr/bin/env bash

# This script ensures proper POSIX text file formatting and a few other things.
# This is supplementary to clang_format.sh and black_format.sh, but should be
# run before them.

# We need dos2unix and isutf8.
if [ ! -x "$(command -v dos2unix)" -o ! -x "$(command -v isutf8)" ]; then
    printf "Install 'dos2unix' and 'isutf8' (moreutils package) to use this script.\n"
fi

set -uo pipefail
IFS=$'\n\t'

# Loops through all text files tracked by Git.
git grep -zIl '' |
while IFS= read -rd '' f; do
    # Exclude some types of files.
    if [[ "$f" == *"csproj" ]]; then
        continue
    elif [[ "$f" == *"sln" ]]; then
        continue
    elif [[ "$f" == *".bat" ]]; then
        continue
    elif [[ "$f" == *".out" ]]; then
        # GDScript integration testing files.
        continue
    elif [[ "$f" == *"patch" ]]; then
        continue
    elif [[ "$f" == *"pot" ]]; then
        continue
    elif [[ "$f" == *"po" ]]; then
        continue
    elif [[ "$f" == "thirdparty/"* ]]; then
        continue
    elif [[ "$f" == *"/thirdparty/"* ]]; then
        continue
    elif [[ "$f" == "platform/android/java/lib/src/com/google"* ]]; then
        continue
    elif [[ "$f" == *"-so_wrap."* ]]; then
        continue
    elif [[ "$f" == *".test.txt" ]]; then
        continue
    fi
    # Ensure that files are UTF-8 formatted.
    isutf8 "$f" >> utf8-validation.txt 2>&1
    # Ensure that files have LF line endings and do not contain a BOM.
    dos2unix "$f" 2> /dev/null
    # Remove trailing space characters and ensures that files end
    # with newline characters. -l option handles newlines conveniently.
    perl -i -ple 's/\s*$//g' "$f"
done

diff=$(git diff --color)

if [ ! -s utf8-validation.txt ] && [ -z "$diff" ] ; then
    # If no UTF-8 violations were collected (the file is empty) and
    # no diff has been generated all is OK, clean up, and exit.
    printf "Files in this commit comply with the formatting rules.\n"
    rm -f utf8-validation.txt
    exit 0
fi

if [ -s utf8-validation.txt ]
then
    # If the file has content and is not empty, violations
    # detected, notify the user, clean up, and exit.
    printf "\n*** The following files contain invalid UTF-8 character sequences:\n\n"
    cat utf8-validation.txt
fi

rm -f utf8-validation.txt

if [ ! -z "$diff" ]
then
    printf "\n*** The following differences were found between the code "
    printf "and the formatting rules:\n\n"
    echo "$diff"
fi

printf "\n*** Aborting, please fix your commit(s) with 'git commit --amend' or 'git rebase -i <hash>'\n"
exit 1

#!/usr/bin/env bash

# This script ensures proper POSIX text file formatting and a few other things.
# This is supplementary to clang_format.sh and black_format.sh, but should be
# run before them.

# We need dos2unix and recode.
if [ ! -x "$(command -v dos2unix)" -o ! -x "$(command -v isutf8)" ]; then
    printf "Install 'dos2unix' and 'isutf8' (from the moreutils package) to use this script.\n"
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
    fi
    # Ensure that files are UTF-8 formatted.
    isutf8 "$f" >> utf8-validation.txt 2>&1
    # Ensure that files have LF line endings and do not contain a BOM.
    dos2unix "$f" 2> /dev/null
    # Remove trailing space characters and ensures that files end
    # with newline characters. -l option handles newlines conveniently.
    perl -i -ple 's/\s*$//g' "$f"
    # Remove the character sequence "== true" if it has a leading space.
    perl -i -pe 's/\x20== true//g' "$f"
done

git diff --color > patch.patch

# If no UTF-8 violations were collected and no patch has been
# generated all is OK, clean up, and exit.
if [ ! -s utf8-validation.txt ] && [ ! -s patch.patch ] ; then
    printf "Files in this commit comply with the formatting rules.\n"
    rm -f patch.patch utf8-validation.txt
    exit 0
fi

# Violations detected, notify the user, clean up, and exit.
if [ -s utf8-validation.txt ]
then
    printf "\n*** The following files contain invalid UTF-8 character sequences:\n\n"
    cat utf8-validation.txt
fi

if [ -s patch.patch ]
then
    printf "\n*** The following differences were found between the code "
    printf "and the formatting rules:\n\n"
    cat patch.patch
fi
rm -f utf8-validation.txt patch.patch
printf "\n*** Aborting, please fix your commit(s) with 'git commit --amend' or 'git rebase -i <hash>'\n"
exit 1

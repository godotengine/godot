#!/usr/bin/env bash

# This script runs black on all Python files in the repo.

set -uo pipefail

# Apply black.
echo -e "Formatting Python files..."
PY_FILES=$(find \( -path "./.git" \
                -o -path "./thirdparty" \
                \) -prune \
                -o \( -name "SConstruct" \
                -o -name "SCsub" \
                -o -name "*.py" \
                \) -print)
black -l 120 $PY_FILES

git diff > patch.patch

# If no patch has been generated all is OK, clean up, and exit.
if [ ! -s patch.patch ] ; then
    printf "Files in this commit comply with the black style rules.\n"
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

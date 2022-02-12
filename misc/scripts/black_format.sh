#!/usr/bin/env bash

# This script runs black on all Python files in the repo.

set -uo pipefail

# Apply black.
echo -e "Formatting Python files..."
PY_FILES=$(git ls-files -- '*SConstruct' '*SCsub' '*.py' ':!:.git/*' ':!:thirdparty/*')
black -l 120 $PY_FILES

diff=$(git diff --color)

# If no patch has been generated all is OK, clean up, and exit.
if [ -z "$diff" ] ; then
    printf "Files in this commit comply with the black style rules.\n"
    exit 0
fi

# A patch has been created, notify the user, clean up, and exit.
printf "\n*** The following differences were found between the code "
printf "and the formatting rules:\n\n"
echo "$diff"
printf "\n*** Aborting, please fix your commit(s) with 'git commit --amend' or 'git rebase -i <hash>'\n"
exit 1

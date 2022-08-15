#!/bin/bash
# A simple script to synchronise a fuzz test corpus
# with an external git repository.
# Usage:
#   pull_and_push_fuzz_corpus.sh DIR
# It assumes that DIR is inside a git repo and push
# can be done w/o typing a password.
cd $1
git add *
git commit -m "fuzz test corpus"
git pull --rebase --no-edit
for((attempt=0; attempt<5; attempt++)); do
  echo GIT PUSH $1 ATTEMPT $attempt
  if $(git push); then break; fi
  git pull --rebase --no-edit
done


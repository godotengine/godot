#!/usr/bin/env bash

set -uo pipefail

FORCE_NPX=${FORCE_NPX:-no}
FORCE_NPX=$(echo "$FORCE_NPX" | tr "[:upper:]" "[:lower:]")

if type "biome" > /dev/null && [ "$FORCE_NPX" = "no" ]; then
	printf "\e[3mUsing local \`biome\` at \"$(which biome)\"\e[0m\n"
 	biome_exec=biome
else
	biome_version="1.7.3"
	printf "\e[3mUsing \`npx --yes @biomejs/biome@$biome_version\`\e[0m\n"
	biome_exec="npx --yes @biomejs/biome@$biome_version"
fi

# Loops through all code files tracked by Git.
git ls-files -- '*.js' '*.jsx' '*.ts' '*.tsx' '*.json' '*.jsonc' '*.css' '*.html' \
                ':!:.git/*' ':!:thirdparty/*' ':!:platform/android/java/lib/src/com/google/*' ':!:*-so_wrap.*' |
while read -r f; do
    # Run `biome`.
    "$biome_exec" check --write "$f" &> /dev/null
done

diff=$(git diff --color)

# If no diff has been generated all is OK, clean up, and exit.
if [ -z "$diff" ] ; then
    printf "\e[1;32m*** Files in this commit comply with the biome style rules.\e[0m\n"
    exit 0
fi

# A diff has been created, notify the user, clean up, and exit.
printf "\n\e[1;33m*** The following changes must be made to comply with the formatting rules:\e[0m\n\n"
# Perl commands replace trailing spaces with `·` and tabs with `<TAB>`.
printf "%s\n" "$diff" | perl -pe 's/(.*[^ ])( +)(\e\[m)$/my $spaces="·" x length($2); sprintf("$1$spaces$3")/ge' | perl -pe 's/(.*[^\t])(\t+)(\e\[m)$/my $tabs="<TAB>" x length($2); sprintf("$1$tabs$3")/ge'

printf "\n\e[1;91m*** Please fix your commit(s) with 'git commit --amend' or 'git rebase -i <hash>'\e[0m\n"
exit 1

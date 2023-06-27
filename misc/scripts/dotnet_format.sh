#!/usr/bin/env bash

# This script runs dotnet format on all relevant files in the repo.
# This is the primary script responsible for fixing style violations in C# files.

set -uo pipefail

# Create dummy generated files.
echo "<Project />" > modules/mono/SdkPackageVersions.props
mkdir -p modules/mono/glue/GodotSharp/GodotSharp/Generated
echo "<Project />" > modules/mono/glue/GodotSharp/GodotSharp/Generated/GeneratedIncludes.props
mkdir -p modules/mono/glue/GodotSharp/GodotSharpEditor/Generated
echo "<Project />" > modules/mono/glue/GodotSharp/GodotSharpEditor/Generated/GeneratedIncludes.props

# Loops through all C# projects tracked by Git.
git ls-files -- '*.csproj' \
                ':!:.git/*' ':!:thirdparty/*' ':!:platform/android/java/lib/src/com/google/*' ':!:*-so_wrap.*' |
while read -r f; do
    # Run dotnet format.
    dotnet format "$f"
done

diff=$(git diff --color)

# If no diff has been generated all is OK, clean up, and exit.
if [ -z "$diff" ] ; then
    printf "\e[1;32m*** Files in this commit comply with the dotnet format style rules.\e[0m\n"
    exit 0
fi

# A diff has been created, notify the user, clean up, and exit.
printf "\n\e[1;33m*** The following changes must be made to comply with the formatting rules:\e[0m\n\n"
# Perl commands replace trailing spaces with `·` and tabs with `<TAB>`.
printf "$diff\n" | perl -pe 's/(.*[^ ])( +)(\e\[m)$/my $spaces="·" x length($2); sprintf("$1$spaces$3")/ge' | perl -pe 's/(.*[^\t])(\t+)(\e\[m)$/my $tabs="<TAB>" x length($2); sprintf("$1$tabs$3")/ge'

printf "\n\e[1;91m*** Please fix your commit(s) with 'git commit --amend' or 'git rebase -i <hash>'\e[0m\n"
exit 1

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
    printf "Files in this commit comply with the dotnet format style rules.\n"
    exit 0
fi

# A diff has been created, notify the user, clean up, and exit.
printf "\n*** The following changes have been made to comply with the formatting rules:\n\n"
echo "$diff"
printf "\n*** Please fix your commit(s) with 'git commit --amend' or 'git rebase -i <hash>'\n"
exit 1

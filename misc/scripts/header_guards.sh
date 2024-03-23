#!/bin/bash

if [ ! -f "version.py" ]; then
  echo "Warning: This script is intended to be run from the root of the Godot repository."
  echo "Some of the paths checks may not work as intended from a different folder."
fi

if [ $# -eq 0 ]; then
    # Loop through all code files tracked by Git.
    files=$(find -name "thirdparty" -prune -o -name "*.h" -print | sed "s@^\./@@g")
else
    # $1 should be a file listing file paths to process. Used in CI.
    files=$(cat "$1" | grep -v "thirdparty/" | grep -E "\.h$" | sed "s@^\./@@g")
fi

files_invalid_guard=""

for file in $files; do
  # Skip *.gen.h and *-so_wrap.h, they're generated.
  if [[ "$file" == *".gen.h" || "$file" == *"-so_wrap.h" ]]; then continue; fi
  # Has important define before normal header guards.
  if [[ "$file" == *"thread.h" || "$file" == *"platform_config.h" || "$file" == *"platform_gl.h" ]]; then continue; fi
  # Obj-C files don't use header guards.
  if grep -q "#import " "$file"; then continue; fi

  bname=$(basename $file .h)

  # Add custom prefix or suffix for generic filenames with a well-defined namespace.

  prefix=
  if [[ "$file" == "modules/"*"/register_types.h" ]]; then
    module=$(echo $file | sed "s@.*modules/\([^/]*\).*@\1@")
    prefix="${module^^}_"
  fi
  if [[ "$file" == "platform/"*"/api/api.h" || "$file" == "platform/"*"/export/"* ]]; then
    platform=$(echo $file | sed "s@.*platform/\([^/]*\).*@\1@")
    prefix="${platform^^}_"
  fi
  if [[ "$file" == "modules/mono/utils/"* && "$bname" != *"mono"* ]]; then prefix="MONO_"; fi
  if [[ "$file" == "servers/rendering/storage/utilities.h" ]]; then prefix="RENDERER_"; fi

  suffix=
  if [[ "$file" == *"dummy"* && "$bname" != *"dummy"* ]]; then suffix="_DUMMY"; fi
  if [[ "$file" == *"gles3"* && "$bname" != *"gles3"* ]]; then suffix="_GLES3"; fi
  if [[ "$file" == *"renderer_rd"* && "$bname" != *"rd"* ]]; then suffix="_RD"; fi
  if [[ "$file" == *"ustring.h" ]]; then suffix="_GODOT"; fi

  # ^^ is bash builtin for UPPERCASE.
  guard="${prefix}${bname^^}${suffix}_H"

  # Replaces guards to use computed name.
  # We also add some \n to make sure there's a proper separation.
  sed -i $file -e "0,/ifndef/s/#ifndef.*/\n#ifndef $guard/"
  sed -i $file -e "0,/define/s/#define.*/#define $guard\n/"
  sed -i $file -e "$ s/#endif.*/\n#endif \/\/ $guard/"
  # Removes redundant \n added before, if they weren't needed.
  sed -i $file -e "/^$/N;/^\n$/D"

  # Check that first ifndef (should be header guard) is at the expected position.
  # If not it can mean we have some code before the guard that should be after.
  # "31" is the expected line with the copyright header.
  first_ifndef=$(grep -n -m 1 "ifndef" $file | sed 's/\([0-9]*\).*/\1/')
  if [[ "$first_ifndef" != "31" ]]; then
    files_invalid_guard+="$file\n"
  fi
done

if [[ ! -z "$files_invalid_guard" ]]; then
  echo -e "The following files were found to have potentially invalid header guard:\n"
  echo -e "$files_invalid_guard"
fi

diff=$(git diff --color)

# If no diff has been generated all is OK, clean up, and exit.
if [ -z "$diff" ] ; then
    printf "\e[1;32m*** Files in this commit comply with the header guards formatting rules.\e[0m\n"
    exit 0
fi

# A diff has been created, notify the user, clean up, and exit.
printf "\n\e[1;33m*** The following changes must be made to comply with the formatting rules:\e[0m\n\n"
# Perl commands replace trailing spaces with `·` and tabs with `<TAB>`.
printf "%s\n" "$diff" | perl -pe 's/(.*[^ ])( +)(\e\[m)$/my $spaces="·" x length($2); sprintf("$1$spaces$3")/ge' | perl -pe 's/(.*[^\t])(\t+)(\e\[m)$/my $tabs="<TAB>" x length($2); sprintf("$1$tabs$3")/ge'

printf "\n\e[1;91m*** Please fix your commit(s) with 'git commit --amend' or 'git rebase -i <hash>'\e[0m\n"
exit 1

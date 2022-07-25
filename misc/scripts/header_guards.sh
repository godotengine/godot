#!/bin/bash

if [ ! -f "version.py" ]; then
  echo "Warning: This script is intended to be run from the root of the Godot repository."
  echo "Some of the paths checks may not work as intended from a different folder."
fi

for file in $(find -name "thirdparty" -prune -o -name "*.h" -print); do
  # Skip *.gen.h and *-so_wrap.h, they're generated.
  if [[ "$file" == *".gen.h" || "$file" == *"-so_wrap.h" ]]; then continue; fi
  # Has important define before normal header guards.
  if [[ "$file" == *"thread.h" || "$file" == *"platform_config.h" ]]; then continue; fi

  bname=$(basename $file .h)

  # Add custom prefix or suffix for generic filenames with a well-defined namespace.

  prefix=
  if [[ "$file" == "./modules/gdnative/"*"/register_types.h" ]]; then
    module=$(echo $file | sed "s@.*modules/gdnative/\([^/]*\).*@\1@")
    prefix="${module^^}_"
  elif [[ "$file" == "./modules/"*"/register_types.h" ]]; then
    module=$(echo $file | sed "s@.*modules/\([^/]*\).*@\1@")
    prefix="${module^^}_"
  fi
  if [[ "$file" == "./platform/"*"/api/api.h" || "$file" == "./platform/"*"/export/"* ]]; then
    platform=$(echo $file | sed "s@.*platform/\([^/]*\).*@\1@")
    prefix="${platform^^}_"
  fi
  if [[ "$file" == "./modules/mono/utils/"* && "$bname" != *"mono"* ]]; then prefix="MONO_"; fi
  if [[ "$file" == "./modules/gdnative/include/gdnative/"* ]]; then prefix="GDNATIVE_"; fi

  suffix=
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
done

diff=$(git diff --color)

# If no diff has been generated all is OK, clean up, and exit.
if [ -z "$diff" ] ; then
    printf "Files in this commit comply with the header guards formatting rules.\n"
    exit 0
fi

# A diff has been created, notify the user, clean up, and exit.
printf "\n*** The following differences were found between the code "
printf "and the header guards formatting rules:\n\n"
echo "$diff"
printf "\n*** Aborting, please fix your commit(s) with 'git commit --amend' or 'git rebase -i <hash>'\n"
exit 1

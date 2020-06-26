#!/usr/bin/env bash

# Command line arguments
run_black=false
run_clang_format=false
run_fix_headers=false
usage="Invalid argument. Usage:\n$0 <option>\n\t--black|-b\n\t--clang-format|-c\n\t--headers|-h\n\t--all|-a"

if [ -z "$1" ]; then
  echo -e $usage
  exit 0
fi

while [ $# -gt 0 ]; do
  case "$1" in
    --black|-b)
      run_black=true
      ;;
    --clang-format|-c)
      run_clang_format=true
      ;;
    --headers|-h)
      run_fix_headers=true
      ;;
    --all|-a)
      run_black=true
      run_clang_format=true
      run_fix_headers=true
      ;;
    *)
      echo -e $usage
      exit 0
  esac
  shift
done

echo "Removing generated files, some have binary data and make clang-format freeze."
find -name "*.gen.*" -delete

# Apply black
if $run_black; then
  echo -e "Formatting Python files..."
  PY_FILES=$(find \( -path "./.git" \
                  -o -path "./thirdparty" \
                  \) -prune \
                  -o \( -name "SConstruct" \
                     -o -name "SCsub" \
                     -o -name "*.py" \
                     \) -print)
  black -l 120 $PY_FILES
fi

# Apply clang-format
if $run_clang_format; then
  # Sync list with pre-commit hook
  FILE_EXTS=".c .h .cpp .hpp .cc .hh .cxx .m .mm .inc .java .glsl"

  for extension in ${FILE_EXTS}; do
    echo -e "Formatting ${extension} files..."
    find \( -path "./.git" \
            -o -path "./thirdparty" \
            -o -path "./platform/android/java/lib/src/com/google" \
         \) -prune \
         -o -name "*${extension}" \
         -exec clang-format -i {} \;
  done
fi

# Add missing copyright headers
if $run_fix_headers; then
  echo "Fixing copyright headers in Godot code files..."
  find \( -path "./.git" -o -path "./thirdparty" \) -prune \
       -o -regex '.*\.\(c\|h\|cpp\|hpp\|cc\|hh\|cxx\|m\|mm\|java\)' \
       > tmp-files
  cat tmp-files | grep -v ".git\|thirdparty\|theme_data.h\|platform/android/java/lib/src/com/google\|platform/android/java/lib/src/org/godotengine/godot/input/InputManager" > files
  python misc/scripts/fix_headers.py
  rm -f tmp-files files
fi

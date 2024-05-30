#!/usr/bin/env bash

if [ $# -eq 0 ]; then
    echo "No arguments provided."
    exit 1
fi

PYTHON_ALIASES=("python3" "python" "py")

for alias in "${PYTHON_ALIASES[@]}"
do
if command "$alias" --version > /dev/null 2>&1; then
    exec "$alias" "$@"
    exit $?
fi
done

echo "Python not found." 1>&2
exit 1

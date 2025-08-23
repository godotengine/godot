set -uo pipefail
shopt -s globstar

echo -e ".gitignore validation..."

# Get a list of files that exist in the repo but are ignored.

# The --verbose flag also includes files un-ignored via ! prefixes.
# We filter those out with a somewhat awkward `awk` directive.
	# (Explanation: Split each line by : delimiters,
	# see if the actual gitignore line shown in the third field starts with !,
	# if it doesn't, print it.)

# ignorecase for the sake of Windows users.
# Use find with xargs to avoid "Argument list too long" errors with large repos.

output=$(find . -type f \
    -not -path './.git/*' \
    -not -path './.github/*' \
    -print0 | \
    xargs -0 git -c core.ignorecase=true check-ignore --verbose 2>/dev/null | \
    awk -F ':' '{ if ($3 !~ /^!/) print $0 }')

# Then we take this result and return success if it's empty.
if [ -z "$output" ]; then
    exit 0
else
	# And print the result if it isn't.
    echo "$output"
    exit 1
fi

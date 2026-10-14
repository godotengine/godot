#!/bin/sh

set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repository_root=$(CDPATH= cd -- "$script_dir/../.." && pwd)

for patch in "$script_dir"/patches/*.patch; do
	printf 'Applying %s\n' "$(basename -- "$patch")"
	git -C "$repository_root" apply --check "$patch"
	git -C "$repository_root" apply "$patch"
done

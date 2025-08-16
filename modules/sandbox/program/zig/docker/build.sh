#!/bin/bash
set -e

usage() {
	echo "Usage: $0 [--api] [-o output] -- [input...]"
	echo "  --api        Instead of compiling, copy the API files to the output."
	echo "   -o output   Compile sources into an output ELF file, including API and inputs."
	echo "  --debug      Compile with debug information."
	echo "  --local      Compile as if locally (outside Docker), using the local API files."
	echo "  --version    Print the current version of the API."
	echo "   -v          Verbose output."
	echo "  --           Separate options from input files."
	exit 1
}

locally=false
verbose=false
current_version=1

while [[ "$#" -gt 0 ]]; do
	case $1 in
		--api) cp -r /usr/api $1; exit ;;
		-o) shift; output="$1"; shift; break ;;
		--debug) shift ;;
		--local) locally=true; shift ;;
		--version) shift; echo "$current_version"; exit ;;
		-v) verbose=true; shift ;;
		--) shift; break ;;
		*) usage ;;
	esac
done

if [ -z "$output" ]; then
	usage
fi

DIR="$(dirname "${output}")"
FILE="$(basename "${output}")"

cleanup() {
	rm -rf /usr/project
}

trap cleanup EXIT

mkdir -p /usr/project/api
cp $@ /usr/project
# Copy the API files to the project directory
cp /usr/api/*.zig /usr/project/api

cd /usr/project
/usr/zig/zig build-exe -O ReleaseSmall -fno-strip -flld -target riscv64-linux *.zig --name $FILE
# Move the output file to the correct location
mv $FILE /usr/src/$output

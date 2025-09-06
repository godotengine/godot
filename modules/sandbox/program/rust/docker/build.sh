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
current_version=5

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

if [ "$locally" = true ]; then
	cargo +nightly build --release # --color never
	cp target/riscv64gc-unknown-linux-gnu/release/rust_program $output
	exit
fi

# We are in /usr/src now, so what we need to do is copy the input files to the project directory
# The project is in /usr/project
cp $@ /usr/project/src/

# Build the project
cd /usr/project
cargo +nightly build --release # --color never

# Copy the resulting ELF file to the output
cp target/riscv64gc-unknown-linux-gnu/release/rust_program /usr/src/$output

# Remove the copied files?
# TODO: Remove the copied files somehow

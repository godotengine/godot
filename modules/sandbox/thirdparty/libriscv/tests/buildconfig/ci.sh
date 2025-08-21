#!/bin/bash

build_libriscv() {
	WORK_DIR=`mktemp -d`
	if [[ ! "$WORK_DIR" || ! -d "$WORK_DIR" ]]; then
	  echo "Could not create temp dir"
	  exit 1
	fi

	THIS_DIR=$PWD
	# Enter build folder and build configuration
	pushd $WORK_DIR
	cmake $THIS_DIR $@
	VERBOSE=1 make -j4
	popd

	# Run the program
	$WORK_DIR/buildconfig

	rm -rf "$WORK_DIR"
}

set -e
build_libriscv $@

#!/usr/bin/bash -ex

GODOT_BINARY=${GODOT_BINARY:-godot}

do_import() {
	for proj; do projdir=$(dirname "$proj")
		echo "Testing import for $projdir"

		# Ensure Godot aborts if files can't be imported.
		! $GODOT_BINARY --fixed-fps 10 --quit-after 10 --headless --path "$projdir"

		$GODOT_BINARY --import                       --headless --path "$projdir"
		$GODOT_BINARY --fixed-fps 10 --quit-after 10 --headless --path "$projdir"
	done
}

do_export() {
	for proj; do projdir=$(dirname "$proj")
		echo "Testing export for $projdir"

		# Ensure Godot aborts if files can't be found.
		! $GODOT_BINARY --fixed-fps 10 --quit-after 10 --headless --main-pack demo.pck

		cp misc/export_presets.cfg "$projdir"
		# Currently, exporting is successful, but Godot crashes on exit.
		$GODOT_BINARY --export-pack "Web" demo.pck --headless --path "$projdir" || true
		mv "$projdir"/demo.pck .
		$GODOT_BINARY --fixed-fps 10 --quit-after 10 --headless --main-pack demo.pck
		rm demo.pck
	done
}

cmd=do_$1
shift
$cmd "$@"


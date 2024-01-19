#!/bin/sh

usage () { echo -e "Usage: Call from the Godot source root to update LimboAI rst class documentation."; }
msg () {  echo -e "$@"; }

set -e

if [ ! -d "${PWD}/modules/limboai" ]; then
    usage
    exit 1
fi

rm -rf /tmp/rst
./doc/tools/make_rst.py --output /tmp/rst doc/tools doc/classes/@GlobalScope.xml modules/limboai/doc_classes || /bin/true
msg Removing old rst class documentation...
rm ./modules/limboai/doc/source/classes/class_*
msg Copying new rst class documentation...
cp -r -f /tmp/rst/class_* modules/limboai/doc/source/classes/
msg Cleaning up...
rm modules/limboai/doc/source/classes/class_@globalscope.rst
msg Done!

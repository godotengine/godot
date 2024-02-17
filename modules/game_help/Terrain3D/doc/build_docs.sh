#!/bin/bash
GODOT=/c/gd/bin/Godot_v4.2.1-stable_win64.exe
MAKERST=/c/gd/godot/doc/tools/make_rst.py
REPO=`git rev-parse --show-toplevel`

pushd $REPO

cd $REPO/project
$GODOT --doctool ../
rm -rf ../{modules,platform}

cd $REPO/doc

$MAKERST --filter Terrain3D --output api classes/

find classes -type f ! -name 'Terrain3D*' -delete

make clean
make html

start _build/html/index.html
popd

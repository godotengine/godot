#!/usr/bin/bash

if [ "$3" == "" ]; then
	echo "Usage: $0 <platform> <version> <godot version>"
	echo "Platforms: linux|windows|all"
	echo "e.g. $0 linux 0.8.1-alpha 4.1.1"
	exit 1
fi

NAME=terrain3d
PLATFORM=$1
VERSION=$2
GDVER=$3
ZIPFILE="${NAME}_${VERSION}_gd${GDVER}"
BIN="addons/terrain_3d/bin"
STRIP=`which strip 2>/dev/null`
INCLUDE="addons/terrain_3d demo project.godot"
ADD="README.md LICENSE CONTRIBUTORS.md"
EXCLUDE="$BIN/*.lib $BIN/*.exp"
WIN_DBG_FILE="$BIN/libterrain.windows.debug.x86_64.dll"
WIN_REL_FILE="$BIN/libterrain.windows.release.x86_64.dll"
LIN_DBG_FILE="$BIN/libterrain.linux.debug.x86_64.so"
LIN_REL_FILE="$BIN/libterrain.linux.release.x86_64.so"

if [ ! -e .git ]; then
	echo .git dir not found. May not be in git root.
	exit 1
fi
if [ ! -e "project/$BIN" ]; then
	echo Bin dir not found. May not be in git root.
	exit 1
fi


# Build Windows
if [ "$PLATFORM" == "windows" ] || [ "$PLATFORM" == "all" ]; then 
	#if [ ! -e $WIN_DBG_FILE ] || [ ! -e $WIN_REL_FILE ]; then 
		scons platform=windows && scons platform=windows target=template_release debug_symbols=false
		if [ $? -gt 0 ]; then
			echo Problem building. Exiting packaging script.
			exit $?
		fi
		if [ "$STRIP" != "" ]; then 
			strip project/$WIN_REL_FILE
		fi
	#fi
	pushd project > /dev/null
	zip -r ../${ZIPFILE}_win64.zip $INCLUDE -x $EXCLUDE $LIN_DBG_FILE $LIN_REL_FILE
	popd > /dev/null
	zip -u ${ZIPFILE}_win64.zip $ADD
fi

# Build Linux
if [ "$PLATFORM" == "linux" ] || [ "$PLATFORM" == "all" ]; then 
	#if [ ! -e $LIN_DBG_FILE ] || [ ! -e $LIN_REL_FILE ]; then 
		scons platform=linux && scons platform=linux target=template_release debug_symbols=false
		if [ $? -gt 0 ]; then
			echo Problem building. Exiting packaging script.
			exit $?
		fi
		if [ "$STRIP" != "" ]; then 
			strip project/$LIN_REL_FILE
		fi
	#fi
	pushd project > /dev/null
	zip -r ../${ZIPFILE}_linux.x86-64.zip $INCLUDE -x $EXCLUDE $WIN_DBG_FILE $WIN_REL_FILE
	popd > /dev/null
	zip -u ${ZIPFILE}_linux.x86-64.zip $ADD
fi


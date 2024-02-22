#!/bin/bash

## This script creates project structure needed for LimboAI development using GDExtension.
## Works only on Unix-likes. You can still use the directory layout below, if you are on Windows.
##
## Instructions:
## 1) Create the project root directory, name doesn't matter.
## 2) Inside the project root directory, clone the limboai repository:
##    git clone https://github.com/limbonaut/limboai
## 3) From the project root directory, run:
##    bash ./limboai/gdextension/setup_gdextension.sh
##
## Directory layout:
##   project/ -- call this script from here, directory name doesn't matter.
##   project/limboai/ -- LimboAI repository should be here after you clone it.
##   project/godot-cpp/ -- will be created by this script, unless cloned manually.
##   project/demo/ -- symbolic link (leads to limboai/demo).
##   project/demo/addons/limboai/limboai.gdextension -- symbolid link (leads to limboai/gdextension/limboai.gdextension).
##   project/demo/addons/limboai/icons/ -- symbolic link (leads to icons/).
##   project/SConstruct -- symbolic link (leads to limboai/gdextension/SContruct).
##
##   Note: Symbolic links will be created by this script.
##
## Dependencies: bash, git, python3.

# Script Settings
GODOT_CPP_VERSION=4.2
PYTHON=python

# Colors
HIGHLIGHT_COLOR='\033[1;36m' # Light Cyan
NC='\033[0m' # No Color
ERROR_COLOR="\033[0;31m"  # Red

usage() { echo "Usage: $0 [--copy-demo] [--copy-all] [--trash-old]"; }

msg () {  echo -e "$@"; }
highlight() { echo -e "${HIGHLIGHT_COLOR}$@${NC}"; }
error () { echo -e "${ERROR_COLOR}$@${NC}" >&2; }

if [ ! -d "${PWD}/limboai/" ]; then
    error Aborting: \"limboai\" subdirectory is not found.
    msg Tip: Run this script from the project root directory with limboai repository cloned into \"limboai\" subdirectory.
    msg Command: bash ./limboai/gdextension/setup_gdextension.sh
    exit 1
fi

# Interrupt execution and exit on Ctrl-C
trap exit SIGINT

set -e

copy_demo=0
copy_all=0
trash_old=0

# Parsing arguments
for i in "$@"
do
    case "${i}" in
	--copy-demo)
	    copy_demo=1
	    shift
	    ;;
	--copy-all)
	    copy_demo=1
	    copy_all=1
	    shift
	    ;;
    --trash-old)
        trash_old=1
        shift
        ;;
	*)
	    usage
        exit 1
	    ;;
    esac
done

highlight Setup started.

${PYTHON} limboai/gdextension/update_icons.py --silent
highlight -- Icon declarations updated.

transfer="ln -s"
transfer_word="Linked"
if [ ${copy_all} == 1 ]; then
    transfer="cp -R"
    transfer_word="Copied"
fi

if [ ${trash_old} == 1 ]; then
    if ! command -v trash &> /dev/null; then
        error trash command not available. Aborting.
        exit 1
    fi
    trash SConstruct limboai/demo/addons demo || /bin/true
    highlight -- Trashed old setup.
fi

if [ ! -d "${PWD}/godot-cpp/" ]; then
    highlight -- Cloning godot-cpp...
    git clone -b ${GODOT_CPP_VERSION} https://github.com/godotengine/godot-cpp
    highlight -- Finished cloning godot-cpp.
else
    highlight -- Skipping \"godot-cpp\". Directory already exists!
fi

if [ ! -f "${PWD}/SConstruct" ]; then
    ${transfer} limboai/gdextension/SConstruct SConstruct
    highlight -- ${transfer_word} SConstruct.
else
    highlight -- Skipping \"SConstruct\". File already exists!
fi

if [ ! -e "${PWD}/demo" ]; then
    if [ ${copy_demo} == 1 ]; then
        cp -R limboai/demo demo
        highlight -- Copied demo.
    else
        ln -s limboai/demo demo
        highlight -- Linked demo project.
    fi
else
    highlight -- Skipping \"demo\". File already exists!
fi

if [ ! -e "${PWD}/demo/addons/limboai/bin/limboai.gdextension" ]; then
    mkdir -p ./demo/addons/limboai/bin/
    cd ./demo/addons/limboai/bin/
    if [ -f "../../../../gdextension/limboai.gdextension" ]; then
        ${transfer} ../../../../gdextension/limboai.gdextension limboai.gdextension
    else
        ${transfer} ../../../../limboai/gdextension/limboai.gdextension limboai.gdextension
    fi
    cd -
    highlight -- ${transfer_word} limboai.gdextension.
else
    highlight -- Skipping limboai.gdextension. File already exists!
fi

if [ ! -e "${PWD}/demo/addons/limboai/icons/" ]; then
    cd ./demo/addons/limboai/
    if [ -d "../../../icons" ]; then
        ${transfer} ../../../icons icons
    else
        ${transfer} ../../../limboai/icons icons
    fi
    cd -
    highlight -- ${transfer_word} icons.
else
    highlight -- Skipping icons. File already exists!
fi

highlight Setup complete.

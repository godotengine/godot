#!/bin/sh

#
# Start ./autobuild.sh in your project folder with a ./build.sh script.
#
# This script will watch for changes in the project folder and subfolders
# and run the ./build.sh script when a .cpp file is saved.
#

set -e -u

FPATH=".."
PATTERN="\.cpp$"
COMMAND="./build.sh"

inotifywait -q --format '%f' -m -r -e close_write $FPATH \
    | grep --line-buffered $PATTERN \
    | xargs -I{} -r sh -c "echo [\$(date -Is)] $COMMAND && $COMMAND"

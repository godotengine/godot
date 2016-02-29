#!/bin/bash
# When scanning for demos, the project manager sorts them based on their
# timestamp, i.e. last modification date. This can make for a pretty
# messy output, so this script 'touches' each engine.cfg file in reverse
# alphabetical order to ensure a nice listing.
#
# It's good practice to run it once before packaging demos on the build
# server.

if [ ! -d "demos" ]; then
  echo "Run this script from the root directory where 'demos/' is contained."
  exit 1
fi

if [ -e demos.list ]; then
  rm -f demos.list
fi

for dir in 2d 3d gui misc viewport; do
  find "demos/$dir" -name "engine.cfg" |sort >> demos.list
done
cat demos.list |sort -r > demos_r.list

while read line; do
  touch $line
  sleep 0.2
done < demos_r.list

#rm -f demos.list demos_r.list

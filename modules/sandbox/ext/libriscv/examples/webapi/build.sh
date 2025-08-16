#!/bin/bash
git submodule update --init
set -e

mkdir -p build
pushd build
cmake ..
make -j6
popd

echo "Build complete. Now run Varnish in the varnish folder, and open http://localhost:8080 in your browser."
./build/webapi

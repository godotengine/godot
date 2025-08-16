#!/bin/sh

scons arch=universal ios_simulator=yes platform=ios target=$1 $2
scons arch=arm64 ios_simulator=no platform=ios target=$1 $2

xcodebuild -create-xcframework -library ./project/bin/libgdexample.ios.$1.a -library ./project/bin/libgdexample.ios.$1.simulator.a -output ./project/bin/libgdexample.ios.$1.xcframework
xcodebuild -create-xcframework -library ../bin/libgodot-cpp.ios.$1.arm64.a -library ../bin/libgodot-cpp.ios.$1.universal.simulator.a  -output ./project/bin/libgodot-cpp.ios.$1.xcframework

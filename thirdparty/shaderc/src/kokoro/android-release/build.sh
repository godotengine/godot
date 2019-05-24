#!/bin/bash

# Copyright (C) 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Android Build Script.


# Fail on any error.
set -e
# Display commands being run.
set -x

BUILD_ROOT=$PWD
SRC=$PWD/github/shaderc
TARGET_ARCH=$1

# Get NINJA.
wget -q https://github.com/ninja-build/ninja/releases/download/v1.7.2/ninja-linux.zip
unzip -q ninja-linux.zip
NINJA=$PWD/ninja

# Get Android NDK.
wget -q https://dl.google.com/android/repository/android-ndk-r18b-linux-x86_64.zip
unzip -q android-ndk-r18b-linux-x86_64.zip
NDK=$PWD/android-ndk-r18b

cd $SRC
./utils/git-sync-deps

mkdir build
cd $SRC/build

# Invoke the build.
BUILD_SHA=${KOKORO_GITHUB_COMMIT:-$KOKORO_GITHUB_PULL_REQUEST_COMMIT}
echo $(date): Starting build...
cmake -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_MAKE_PROGRAM=$NINJA -DANDROID_ABI=$TARGET_ARCH -DSHADERC_SKIP_TESTS=ON -DSPIRV_SKIP_TESTS=ON -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake -DANDROID_NDK=$NDK ..

echo $(date): Build glslang...
$NINJA glslangValidator

echo $(date): Build everything...
$NINJA

echo $(date): Check Shaderc for copyright notices...
$NINJA check-copyright

echo $(date): Build completed.

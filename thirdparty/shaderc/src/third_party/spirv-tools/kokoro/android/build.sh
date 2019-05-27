#!/bin/bash
# Copyright (c) 2018 Google LLC.
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
SRC=$PWD/github/SPIRV-Tools
TARGET_ARCH="armeabi-v7a with NEON"
export ANDROID_NDK=/opt/android-ndk-r15c

# Get NINJA.
wget -q https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip
unzip -q ninja-linux.zip
export PATH="$PWD:$PATH"
git clone --depth=1 https://github.com/taka-no-me/android-cmake.git android-cmake
export TOOLCHAIN_PATH=$PWD/android-cmake/android.toolchain.cmake


cd $SRC
git clone --depth=1 https://github.com/KhronosGroup/SPIRV-Headers external/spirv-headers
git clone --depth=1 https://github.com/google/googletest          external/googletest
git clone --depth=1 https://github.com/google/effcee              external/effcee
git clone --depth=1 https://github.com/google/re2                 external/re2

mkdir build && cd $SRC/build

# Invoke the build.
BUILD_SHA=${KOKORO_GITHUB_COMMIT:-$KOKORO_GITHUB_PULL_REQUEST_COMMIT}
echo $(date): Starting build...
cmake -DCMAKE_BUILD_TYPE=Release -DANDROID_NATIVE_API_LEVEL=android-14 -DANDROID_ABI="armeabi-v7a with NEON" -DSPIRV_BUILD_COMPRESSION=ON -DSPIRV_SKIP_TESTS=ON -DCMAKE_TOOLCHAIN_FILE=$TOOLCHAIN_PATH -GNinja -DANDROID_NDK=$ANDROID_NDK ..

echo $(date): Build everything...
ninja
echo $(date): Build completed.

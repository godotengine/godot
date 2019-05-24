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
# Linux Build Script.

# Fail on any error.
set -e
# Display commands being run.
set -x

BUILD_ROOT=$PWD
SRC=$PWD/github/shaderc

# Get NINJA.
wget -q https://github.com/ninja-build/ninja/releases/download/v1.7.2/ninja-linux.zip
unzip -q ninja-linux.zip
export PATH="$PWD:$PATH"

# Get Android NDK.
wget -q https://dl.google.com/android/repository/android-ndk-r18b-linux-x86_64.zip
unzip -q android-ndk-r18b-linux-x86_64.zip
export ANDROID_NDK=$PWD/android-ndk-r18b

# Get shaderc dependencies.
cd $SRC
./utils/git-sync-deps

cd $SRC
mkdir build
cd $SRC/build

# Invoke the build.
BUILD_SHA=${KOKORO_GITHUB_COMMIT:-$KOKORO_GITHUB_PULL_REQUEST_COMMIT}
echo $(date): Starting ndk-build ...
$ANDROID_NDK/ndk-build \
  -C $SRC/android_test \
  NDK_APP_OUT=`pwd` \
  V=1 \
  SPVTOOLS_LOCAL_PATH=$SRC/third_party/spirv-tools \
  SPVHEADERS_LOCAL_PATH=$SRC/third_party/spirv-headers \
  -j 8

echo $(date): ndk-build completed.

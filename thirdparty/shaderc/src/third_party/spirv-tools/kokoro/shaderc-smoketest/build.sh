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

# Fail on any error.
set -e
# Display commands being run.
set -x

BUILD_ROOT=$PWD
GITHUB_DIR=$BUILD_ROOT/github

SKIP_TESTS="False"
BUILD_TYPE="Release"

# Get NINJA.
wget -q https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip
unzip -q ninja-linux.zip
export PATH="$PWD:$PATH"

# Get shaderc.
cd $GITHUB_DIR
git clone https://github.com/google/shaderc.git
SHADERC_DIR=$GITHUB_DIR/shaderc
cd $SHADERC_DIR/third_party

# Get shaderc dependencies. Link the appropriate SPIRV-Tools.
git clone https://github.com/google/googletest.git
git clone https://github.com/KhronosGroup/glslang.git
ln -s $GITHUB_DIR/SPIRV-Tools spirv-tools
git clone https://github.com/KhronosGroup/SPIRV-Headers.git spirv-headers
git clone https://github.com/google/re2
git clone https://github.com/google/effcee

cd $SHADERC_DIR
mkdir build
cd $SHADERC_DIR/build

# Invoke the build.
BUILD_SHA=${KOKORO_GITHUB_COMMIT:-$KOKORO_GITHUB_PULL_REQUEST_COMMIT}
echo $(date): Starting build...
cmake -GNinja -DRE2_BUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=$BUILD_TYPE ..

echo $(date): Build glslang...
ninja glslangValidator

echo $(date): Build everything...
ninja
echo $(date): Build completed.

echo $(date): Check Shaderc for copyright notices...
ninja check-copyright

echo $(date): Starting ctest...
if [ $SKIP_TESTS = "False" ]
then
  ctest --output-on-failure -j4
fi
echo $(date): ctest completed.


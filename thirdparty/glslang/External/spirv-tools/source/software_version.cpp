// Copyright (c) 2015-2016 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "spirv-tools/libspirv.h"

namespace {

const char* kBuildVersions[] = {
#include "build-version.inc"
};

}  // anonymous namespace

const char* spvSoftwareVersionString(void) { return kBuildVersions[0]; }

const char* spvSoftwareVersionDetailsString(void) { return kBuildVersions[1]; }

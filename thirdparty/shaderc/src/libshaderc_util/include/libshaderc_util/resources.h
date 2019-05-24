// Copyright 2015 The Shaderc Authors. All rights reserved.
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

#ifndef LIBSHADERC_UTIL_RESOURCES_H_
#define LIBSHADERC_UTIL_RESOURCES_H_

// We want TBuiltInResource
#include "glslang/Include/ResourceLimits.h"

namespace shaderc_util {

using TBuiltInResource = ::TBuiltInResource;

// A set of suitable defaults.
extern const TBuiltInResource kDefaultTBuiltInResource;

}  // namespace shaderc_util

#endif  // LIBSHADERC_UTIL_RESOURCES_H_

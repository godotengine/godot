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

#ifndef LIBSHADERC_UTIL_SHADER_STAGE_H_
#define LIBSHADERC_UTIL_SHADER_STAGE_H_

#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "glslang/Public/ShaderLang.h"

#include "libshaderc_util/string_piece.h"

namespace shaderc_util {

// Given a string representing a stage, returns the glslang EShLanguage for it.
// If the stage string is not recognized, returns EShLangCount.
EShLanguage MapStageNameToLanguage(
    const shaderc_util::string_piece& stage_name);

}  // namespace shaderc_util

#endif  // LIBSHADERC_UTIL_SHADER_STAGE_H_

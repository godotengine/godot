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

#ifndef GLSLC_SHADER_STAGE_H_
#define GLSLC_SHADER_STAGE_H_

#include <ostream>

#include "libshaderc_util/string_piece.h"
#include "shaderc/shaderc.h"

namespace glslc {

// Maps a shader stage name to a forced shader stage enum value.  Returns
// 'shaderc_glsl_infer_from_source' if the stage name is unrecognized.
shaderc_shader_kind MapStageNameToForcedKind(
    const shaderc_util::string_piece& f_shader_stage_str);

// Parse the string piece from command line to get the force shader stage.
// If the 'f_shader_stage_str' cannot be parsed to a valid force shader stage,
// returns 'shaderc_glsl_infer_from_source'.  Requires the string to begin with
// '='.
shaderc_shader_kind GetForcedShaderKindFromCmdLine(
    const shaderc_util::string_piece& f_shader_stage_str);

// Parse the file name extension to get the default shader kind.
shaderc_shader_kind DeduceDefaultShaderKindFromFileName(
    shaderc_util::string_piece file_name);
}  // namespace glslc

#endif  // GLSLC_SHADER_STAGE_H_

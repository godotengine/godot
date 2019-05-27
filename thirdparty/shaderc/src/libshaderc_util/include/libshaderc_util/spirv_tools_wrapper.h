// Copyright 2016 The Shaderc Authors. All rights reserved.
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

#ifndef LIBSHADERC_UTIL_INC_SPIRV_TOOLS_WRAPPER_H
#define LIBSHADERC_UTIL_INC_SPIRV_TOOLS_WRAPPER_H

#include <string>
#include <vector>

#include "spirv-tools/libspirv.hpp"

#include "libshaderc_util/compiler.h"
#include "libshaderc_util/string_piece.h"

namespace shaderc_util {
// Assembles the given assembly. On success, returns true, writes the assembled
// binary to *binary, and clears *errors. Otherwise, writes the error message
// into *errors.
bool SpirvToolsAssemble(Compiler::TargetEnv env,
                        Compiler::TargetEnvVersion version,
                        const string_piece assembly, spv_binary* binary,
                        std::string* errors);

// Disassembles the given binary. Returns true and writes the disassembled text
// to *text_or_error if successful. Otherwise, writes the error message to
// *text_or_error.
bool SpirvToolsDisassemble(Compiler::TargetEnv env,
                           Compiler::TargetEnvVersion version,
                           const std::vector<uint32_t>& binary,
                           std::string* text_or_error);

// The ids of a list of supported optimization passes.
enum class PassId {
  // SPIRV-Tools standard recipes
  kLegalizationPasses,
  kPerformancePasses,
  kSizePasses,

  // SPIRV-Tools specific passes
  kNullPass,
  kStripDebugInfo,
  kCompactIds,
};

// Optimizes the given binary. Passes are registered in the exact order as shown
// in enabled_passes, without de-duplication. Returns true and writes the
// optimized binary back to *binary if successful. Otherwise, writes errors to
// *errors and the content of binary may be in an invalid state.
bool SpirvToolsOptimize(Compiler::TargetEnv env,
                        Compiler::TargetEnvVersion version,
                        const std::vector<PassId>& enabled_passes,
                        std::vector<uint32_t>* binary, std::string* errors);

}  // namespace shaderc_util

#endif  // LIBSHADERC_UTIL_INC_SPIRV_TOOLS_WRAPPER_H

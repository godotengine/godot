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

#include "libshaderc_util/spirv_tools_wrapper.h"

#include <algorithm>
#include <sstream>

#include "spirv-tools/optimizer.hpp"

namespace shaderc_util {

namespace {

// Gets the corresponding target environment used in SPIRV-Tools.
spv_target_env GetSpirvToolsTargetEnv(Compiler::TargetEnv env,
                                      Compiler::TargetEnvVersion version) {
  switch (env) {
    case Compiler::TargetEnv::Vulkan:
      switch (version) {
        case Compiler::TargetEnvVersion::Default:
          return SPV_ENV_VULKAN_1_0;
        case Compiler::TargetEnvVersion::Vulkan_1_0:
          return SPV_ENV_VULKAN_1_0;
        case Compiler::TargetEnvVersion::Vulkan_1_1:
          return SPV_ENV_VULKAN_1_1;
        default:
          break;
      }
      break;
    case Compiler::TargetEnv::OpenGL:
      return SPV_ENV_OPENGL_4_5;
    case Compiler::TargetEnv::OpenGLCompat:  // Deprecated
      return SPV_ENV_OPENGL_4_5;
  }
  assert(false && "unexpected target environment or version");
  return SPV_ENV_VULKAN_1_0;
}

}  // anonymous namespace

bool SpirvToolsDisassemble(Compiler::TargetEnv env,
                           Compiler::TargetEnvVersion version,
                           const std::vector<uint32_t>& binary,
                           std::string* text_or_error) {
  spvtools::SpirvTools tools(GetSpirvToolsTargetEnv(env, version));
  std::ostringstream oss;
  tools.SetMessageConsumer([&oss](spv_message_level_t, const char*,
                                  const spv_position_t& position,
                                  const char* message) {
    oss << position.index << ": " << message;
  });
  const bool success =
      tools.Disassemble(binary, text_or_error,
                        SPV_BINARY_TO_TEXT_OPTION_INDENT |
                            SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES);
  if (!success) {
    *text_or_error = oss.str();
  }
  return success;
}

bool SpirvToolsAssemble(Compiler::TargetEnv env,
                        Compiler::TargetEnvVersion version,
                        const string_piece assembly, spv_binary* binary,
                        std::string* errors) {
  auto spvtools_context =
      spvContextCreate(GetSpirvToolsTargetEnv(env, version));
  spv_diagnostic spvtools_diagnostic = nullptr;

  *binary = nullptr;
  errors->clear();

  const bool success =
      spvTextToBinary(spvtools_context, assembly.data(), assembly.size(),
                      binary, &spvtools_diagnostic) == SPV_SUCCESS;
  if (!success) {
    std::ostringstream oss;
    oss << spvtools_diagnostic->position.line + 1 << ":"
        << spvtools_diagnostic->position.column + 1 << ": "
        << spvtools_diagnostic->error;
    *errors = oss.str();
  }

  spvDiagnosticDestroy(spvtools_diagnostic);
  spvContextDestroy(spvtools_context);

  return success;
}

bool SpirvToolsOptimize(Compiler::TargetEnv env,
                        Compiler::TargetEnvVersion version,
                        const std::vector<PassId>& enabled_passes,
                        std::vector<uint32_t>* binary, std::string* errors) {
  errors->clear();
  if (enabled_passes.empty()) return true;
  if (std::all_of(
          enabled_passes.cbegin(), enabled_passes.cend(),
          [](const PassId& pass) { return pass == PassId::kNullPass; })) {
    return true;
  }

  spvtools::ValidatorOptions val_opts;
  // This allows flexible memory layout for HLSL.
  val_opts.SetSkipBlockLayout(true);
  // This allows HLSL legalization regarding resources.
  val_opts.SetRelaxLogicalPointer(true);

  spvtools::OptimizerOptions opt_opts;
  opt_opts.set_validator_options(val_opts);
  opt_opts.set_run_validator(true);

  spvtools::Optimizer optimizer(GetSpirvToolsTargetEnv(env, version));

  std::ostringstream oss;
  optimizer.SetMessageConsumer(
      [&oss](spv_message_level_t, const char*, const spv_position_t&,
             const char* message) { oss << message << "\n"; });

  for (const auto& pass : enabled_passes) {
    switch (pass) {
      case PassId::kLegalizationPasses:
        optimizer.RegisterLegalizationPasses();
        break;
      case PassId::kPerformancePasses:
        optimizer.RegisterPerformancePasses();
        break;
      case PassId::kSizePasses:
        optimizer.RegisterSizePasses();
        break;
      case PassId::kNullPass:
        // We actually don't need to do anything for null pass.
        break;
      case PassId::kStripDebugInfo:
        optimizer.RegisterPass(spvtools::CreateStripDebugInfoPass());
        break;
      case PassId::kCompactIds:
        optimizer.RegisterPass(spvtools::CreateCompactIdsPass());
        break;
    }
  }

  if (!optimizer.Run(binary->data(), binary->size(), binary, opt_opts)) {
    *errors = oss.str();
    return false;
  }
  return true;
}

}  // namespace shaderc_util

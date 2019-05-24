// Copyright (c) 2016 Google Inc.
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

// This program demonstrates basic SPIR-V module processing using
// SPIRV-Tools C++ API:
// * Assembling
// * Validating
// * Optimizing
// * Disassembling

#include <iostream>
#include <string>
#include <vector>

#include "spirv-tools/libspirv.hpp"
#include "spirv-tools/optimizer.hpp"

int main() {
  const std::string source =
      "         OpCapability Shader "
      "         OpMemoryModel Logical GLSL450 "
      "         OpSource GLSL 450 "
      "         OpDecorate %spec SpecId 1 "
      "  %int = OpTypeInt 32 1 "
      " %spec = OpSpecConstant %int 0 "
      "%const = OpConstant %int 42";

  spvtools::SpirvTools core(SPV_ENV_VULKAN_1_0);
  spvtools::Optimizer opt(SPV_ENV_VULKAN_1_0);

  auto print_msg_to_stderr = [](spv_message_level_t, const char*,
                                const spv_position_t&, const char* m) {
    std::cerr << "error: " << m << std::endl;
  };
  core.SetMessageConsumer(print_msg_to_stderr);
  opt.SetMessageConsumer(print_msg_to_stderr);

  std::vector<uint32_t> spirv;
  if (!core.Assemble(source, &spirv)) return 1;
  if (!core.Validate(spirv)) return 1;

  opt.RegisterPass(spvtools::CreateSetSpecConstantDefaultValuePass({{1, "42"}}))
      .RegisterPass(spvtools::CreateFreezeSpecConstantValuePass())
      .RegisterPass(spvtools::CreateUnifyConstantPass())
      .RegisterPass(spvtools::CreateStripDebugInfoPass());
  if (!opt.Run(spirv.data(), spirv.size(), &spirv)) return 1;

  std::string disassembly;
  if (!core.Disassemble(spirv, &disassembly)) return 1;
  std::cout << disassembly << "\n";

  return 0;
}

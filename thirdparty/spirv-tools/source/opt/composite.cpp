// Copyright (c) 2018 The Khronos Group Inc.
// Copyright (c) 2018 Valve Corporation
// Copyright (c) 2018 LunarG Inc.
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

#include "source/opt/composite.h"

#include <vector>

#include "source/opt/ir_context.h"
#include "source/opt/iterator.h"
#include "spirv/1.2/GLSL.std.450.h"

namespace spvtools {
namespace opt {

bool ExtInsMatch(const std::vector<uint32_t>& extIndices,
                 const Instruction* insInst, const uint32_t extOffset) {
  uint32_t numIndices = static_cast<uint32_t>(extIndices.size()) - extOffset;
  if (numIndices != insInst->NumInOperands() - 2) return false;
  for (uint32_t i = 0; i < numIndices; ++i)
    if (extIndices[i + extOffset] != insInst->GetSingleWordInOperand(i + 2))
      return false;
  return true;
}

bool ExtInsConflict(const std::vector<uint32_t>& extIndices,
                    const Instruction* insInst, const uint32_t extOffset) {
  if (extIndices.size() - extOffset == insInst->NumInOperands() - 2)
    return false;
  uint32_t extNumIndices = static_cast<uint32_t>(extIndices.size()) - extOffset;
  uint32_t insNumIndices = insInst->NumInOperands() - 2;
  uint32_t numIndices = std::min(extNumIndices, insNumIndices);
  for (uint32_t i = 0; i < numIndices; ++i)
    if (extIndices[i + extOffset] != insInst->GetSingleWordInOperand(i + 2))
      return false;
  return true;
}

}  // namespace opt
}  // namespace spvtools

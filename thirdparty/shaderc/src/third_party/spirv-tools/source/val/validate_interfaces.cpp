// Copyright (c) 2018 Google LLC.
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

#include "source/val/validate.h"

#include <algorithm>
#include <vector>

#include "source/diagnostic.h"
#include "source/val/function.h"
#include "source/val/instruction.h"
#include "source/val/validation_state.h"

namespace spvtools {
namespace val {
namespace {

// Returns true if \c inst is an input or output variable.
bool is_interface_variable(const Instruction* inst) {
  return inst->opcode() == SpvOpVariable &&
         (inst->word(3u) == SpvStorageClassInput ||
          inst->word(3u) == SpvStorageClassOutput);
}

// Checks that \c var is listed as an interface in all the entry points that use
// it.
spv_result_t check_interface_variable(ValidationState_t& _,
                                      const Instruction* var) {
  std::vector<const Function*> functions;
  std::vector<const Instruction*> uses;
  for (auto use : var->uses()) {
    uses.push_back(use.first);
  }
  for (uint32_t i = 0; i < uses.size(); ++i) {
    const auto user = uses[i];
    if (const Function* func = user->function()) {
      functions.push_back(func);
    } else {
      // In the rare case that the variable is used by another instruction in
      // the global scope, continue searching for an instruction used in a
      // function.
      for (auto use : user->uses()) {
        uses.push_back(use.first);
      }
    }
  }

  std::sort(functions.begin(), functions.end(),
            [](const Function* lhs, const Function* rhs) {
              return lhs->id() < rhs->id();
            });
  functions.erase(std::unique(functions.begin(), functions.end()),
                  functions.end());

  std::vector<uint32_t> entry_points;
  for (const auto func : functions) {
    for (auto id : _.FunctionEntryPoints(func->id())) {
      entry_points.push_back(id);
    }
  }

  std::sort(entry_points.begin(), entry_points.end());
  entry_points.erase(std::unique(entry_points.begin(), entry_points.end()),
                     entry_points.end());

  for (auto id : entry_points) {
    for (const auto& desc : _.entry_point_descriptions(id)) {
      bool found = false;
      for (auto interface : desc.interfaces) {
        if (var->id() == interface) {
          found = true;
          break;
        }
      }
      if (!found) {
        return _.diag(SPV_ERROR_INVALID_ID, var)
               << (var->word(3u) == SpvStorageClassInput ? "Input" : "Output")
               << " variable id <" << var->id() << "> is used by entry point '"
               << desc.name << "' id <" << id
               << ">, but is not listed as an interface";
      }
    }
  }

  return SPV_SUCCESS;
}

}  // namespace

spv_result_t ValidateInterfaces(ValidationState_t& _) {
  for (auto& inst : _.ordered_instructions()) {
    if (is_interface_variable(&inst)) {
      if (auto error = check_interface_variable(_, &inst)) {
        return error;
      }
    }
  }

  return SPV_SUCCESS;
}

}  // namespace val
}  // namespace spvtools

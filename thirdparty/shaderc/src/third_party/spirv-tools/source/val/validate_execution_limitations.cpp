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

#include "source/val/function.h"
#include "source/val/validation_state.h"

namespace spvtools {
namespace val {

spv_result_t ValidateExecutionLimitations(ValidationState_t& _,
                                          const Instruction* inst) {
  if (inst->opcode() != SpvOpFunction) {
    return SPV_SUCCESS;
  }

  const auto func = _.function(inst->id());
  if (!func) {
    return _.diag(SPV_ERROR_INTERNAL, inst)
           << "Internal error: missing function id " << inst->id() << ".";
  }

  for (uint32_t entry_id : _.FunctionEntryPoints(inst->id())) {
    const auto* models = _.GetExecutionModels(entry_id);
    if (models) {
      if (models->empty()) {
        return _.diag(SPV_ERROR_INTERNAL, inst)
               << "Internal error: empty execution models for function id "
               << entry_id << ".";
      }
      for (const auto model : *models) {
        std::string reason;
        if (!func->IsCompatibleWithExecutionModel(model, &reason)) {
          return _.diag(SPV_ERROR_INVALID_ID, inst)
                 << "OpEntryPoint Entry Point <id> '" << _.getIdName(entry_id)
                 << "'s callgraph contains function <id> "
                 << _.getIdName(inst->id())
                 << ", which cannot be used with the current execution "
                    "model:\n"
                 << reason;
        }
      }
    }
  }
  return SPV_SUCCESS;
}

}  // namespace val
}  // namespace spvtools

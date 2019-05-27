// Copyright (c) 2017 Google Inc.
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

#include "source/opt/compact_ids_pass.h"

#include <cassert>
#include <unordered_map>

#include "source/opt/ir_context.h"

namespace spvtools {
namespace opt {

Pass::Status CompactIdsPass::Process() {
  bool modified = false;
  std::unordered_map<uint32_t, uint32_t> result_id_mapping;

  context()->module()->ForEachInst(
      [&result_id_mapping, &modified](Instruction* inst) {
        auto operand = inst->begin();
        while (operand != inst->end()) {
          const auto type = operand->type;
          if (spvIsIdType(type)) {
            assert(operand->words.size() == 1);
            uint32_t& id = operand->words[0];
            auto it = result_id_mapping.find(id);
            if (it == result_id_mapping.end()) {
              const uint32_t new_id =
                  static_cast<uint32_t>(result_id_mapping.size()) + 1;
              const auto insertion_result =
                  result_id_mapping.emplace(id, new_id);
              it = insertion_result.first;
              assert(insertion_result.second);
            }
            if (id != it->second) {
              modified = true;
              id = it->second;
              // Update data cached in the instruction object.
              if (type == SPV_OPERAND_TYPE_RESULT_ID) {
                inst->SetResultId(id);
              } else if (type == SPV_OPERAND_TYPE_TYPE_ID) {
                inst->SetResultType(id);
              }
            }
          }
          ++operand;
        }
      },
      true);

  if (modified)
    context()->module()->SetIdBound(
        static_cast<uint32_t>(result_id_mapping.size() + 1));

  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

}  // namespace opt
}  // namespace spvtools

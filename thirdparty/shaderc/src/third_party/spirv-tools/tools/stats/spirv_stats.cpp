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

#include "tools/stats/spirv_stats.h"

#include <cassert>

#include <algorithm>
#include <memory>
#include <string>

#include "source/diagnostic.h"
#include "source/enum_string_mapping.h"
#include "source/extensions.h"
#include "source/id_descriptor.h"
#include "source/instruction.h"
#include "source/opcode.h"
#include "source/operand.h"
#include "source/val/instruction.h"
#include "source/val/validate.h"
#include "source/val/validation_state.h"
#include "spirv-tools/libspirv.h"

namespace spvtools {
namespace stats {
namespace {

// Helper class for stats aggregation. Receives as in/out parameter.
// Constructs ValidationState and updates it by running validator for each
// instruction.
class StatsAggregator {
 public:
  StatsAggregator(SpirvStats* in_out_stats, const val::ValidationState_t* state)
      : stats_(in_out_stats), vstate_(state) {}

  // Processes the instructions to collect stats.
  void aggregate() {
    const auto& instructions = vstate_->ordered_instructions();

    ++stats_->version_hist[vstate_->version()];
    ++stats_->generator_hist[vstate_->generator()];

    for (size_t i = 0; i < instructions.size(); ++i) {
      const auto& inst = instructions[i];

      ProcessOpcode(&inst, i);
      ProcessCapability(&inst);
      ProcessExtension(&inst);
      ProcessConstant(&inst);
    }
  }

  // Collects OpCapability statistics.
  void ProcessCapability(const val::Instruction* inst) {
    if (inst->opcode() != SpvOpCapability) return;
    const uint32_t capability = inst->word(inst->operands()[0].offset);
    ++stats_->capability_hist[capability];
  }

  // Collects OpExtension statistics.
  void ProcessExtension(const val::Instruction* inst) {
    if (inst->opcode() != SpvOpExtension) return;
    const std::string extension = GetExtensionString(&inst->c_inst());
    ++stats_->extension_hist[extension];
  }

  // Collects OpCode statistics.
  void ProcessOpcode(const val::Instruction* inst, size_t idx) {
    const SpvOp opcode = inst->opcode();
    ++stats_->opcode_hist[opcode];

    if (idx == 0) return;

    --idx;

    const auto& instructions = vstate_->ordered_instructions();

    auto step_it = stats_->opcode_markov_hist.begin();
    for (; step_it != stats_->opcode_markov_hist.end(); --idx, ++step_it) {
      auto& hist = (*step_it)[instructions[idx].opcode()];
      ++hist[opcode];

      if (idx == 0) break;
    }
  }

  // Collects OpConstant statistics.
  void ProcessConstant(const val::Instruction* inst) {
    if (inst->opcode() != SpvOpConstant) return;

    const uint32_t type_id = inst->GetOperandAs<uint32_t>(0);
    const auto type_decl_it = vstate_->all_definitions().find(type_id);
    assert(type_decl_it != vstate_->all_definitions().end());

    const val::Instruction& type_decl_inst = *type_decl_it->second;
    const SpvOp type_op = type_decl_inst.opcode();
    if (type_op == SpvOpTypeInt) {
      const uint32_t bit_width = type_decl_inst.GetOperandAs<uint32_t>(1);
      const uint32_t is_signed = type_decl_inst.GetOperandAs<uint32_t>(2);
      assert(is_signed == 0 || is_signed == 1);
      if (bit_width == 16) {
        if (is_signed)
          ++stats_->s16_constant_hist[inst->GetOperandAs<int16_t>(2)];
        else
          ++stats_->u16_constant_hist[inst->GetOperandAs<uint16_t>(2)];
      } else if (bit_width == 32) {
        if (is_signed)
          ++stats_->s32_constant_hist[inst->GetOperandAs<int32_t>(2)];
        else
          ++stats_->u32_constant_hist[inst->GetOperandAs<uint32_t>(2)];
      } else if (bit_width == 64) {
        if (is_signed)
          ++stats_->s64_constant_hist[inst->GetOperandAs<int64_t>(2)];
        else
          ++stats_->u64_constant_hist[inst->GetOperandAs<uint64_t>(2)];
      } else {
        assert(false && "TypeInt bit width is not 16, 32 or 64");
      }
    } else if (type_op == SpvOpTypeFloat) {
      const uint32_t bit_width = type_decl_inst.GetOperandAs<uint32_t>(1);
      if (bit_width == 32) {
        ++stats_->f32_constant_hist[inst->GetOperandAs<float>(2)];
      } else if (bit_width == 64) {
        ++stats_->f64_constant_hist[inst->GetOperandAs<double>(2)];
      } else {
        assert(bit_width == 16);
      }
    }
  }

 private:
  SpirvStats* stats_;
  const val::ValidationState_t* vstate_;
  IdDescriptorCollection id_descriptors_;
};

}  // namespace

spv_result_t AggregateStats(const spv_context context, const uint32_t* words,
                            const size_t num_words, spv_diagnostic* pDiagnostic,
                            SpirvStats* stats) {
  std::unique_ptr<val::ValidationState_t> vstate;
  spv_validator_options_t options;
  spv_result_t result = ValidateBinaryAndKeepValidationState(
      context, &options, words, num_words, pDiagnostic, &vstate);
  if (result != SPV_SUCCESS) return result;

  StatsAggregator stats_aggregator(stats, vstate.get());
  stats_aggregator.aggregate();
  return SPV_SUCCESS;
}

}  // namespace stats
}  // namespace spvtools

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

#ifndef TOOLS_STATS_SPIRV_STATS_H_
#define TOOLS_STATS_SPIRV_STATS_H_

#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "spirv-tools/libspirv.hpp"

namespace spvtools {
namespace stats {

struct SpirvStats {
  // Version histogram, version_word -> count.
  std::unordered_map<uint32_t, uint32_t> version_hist;

  // Generator histogram, generator_word -> count.
  std::unordered_map<uint32_t, uint32_t> generator_hist;

  // Capability histogram, SpvCapabilityXXX -> count.
  std::unordered_map<uint32_t, uint32_t> capability_hist;

  // Extension histogram, extension_string -> count.
  std::unordered_map<std::string, uint32_t> extension_hist;

  // Opcode histogram, SpvOpXXX -> count.
  std::unordered_map<uint32_t, uint32_t> opcode_hist;

  // OpConstant u16 histogram, value -> count.
  std::unordered_map<uint16_t, uint32_t> u16_constant_hist;

  // OpConstant u32 histogram, value -> count.
  std::unordered_map<uint32_t, uint32_t> u32_constant_hist;

  // OpConstant u64 histogram, value -> count.
  std::unordered_map<uint64_t, uint32_t> u64_constant_hist;

  // OpConstant s16 histogram, value -> count.
  std::unordered_map<int16_t, uint32_t> s16_constant_hist;

  // OpConstant s32 histogram, value -> count.
  std::unordered_map<int32_t, uint32_t> s32_constant_hist;

  // OpConstant s64 histogram, value -> count.
  std::unordered_map<int64_t, uint32_t> s64_constant_hist;

  // OpConstant f32 histogram, value -> count.
  std::unordered_map<float, uint32_t> f32_constant_hist;

  // OpConstant f64 histogram, value -> count.
  std::unordered_map<double, uint32_t> f64_constant_hist;

  // Used to collect statistics on opcodes triggering other opcodes.
  // Container scheme: gap between instructions -> cue opcode -> later opcode
  // -> count.
  // For example opcode_markov_hist[2][OpFMul][OpFAdd] corresponds to
  // the number of times an OpMul appears, followed by 2 other instructions,
  // followed by OpFAdd.
  // opcode_markov_hist[0][OpFMul][OpFAdd] corresponds to how many times
  // OpFMul appears, directly followed by OpFAdd.
  // The size of the outer std::vector also serves as an input parameter,
  // determining how many steps will be collected.
  // I.e. do opcode_markov_hist.resize(1) to collect data for one step only.
  std::vector<
      std::unordered_map<uint32_t, std::unordered_map<uint32_t, uint32_t>>>
      opcode_markov_hist;
};

// Aggregates existing |stats| with new stats extracted from |binary|.
spv_result_t AggregateStats(const spv_context context, const uint32_t* words,
                            const size_t num_words, spv_diagnostic* pDiagnostic,
                            SpirvStats* stats);

}  // namespace stats
}  // namespace spvtools

#endif  // TOOLS_STATS_SPIRV_STATS_H_

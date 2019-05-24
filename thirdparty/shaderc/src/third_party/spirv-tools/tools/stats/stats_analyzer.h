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

#ifndef TOOLS_STATS_STATS_ANALYZER_H_
#define TOOLS_STATS_STATS_ANALYZER_H_

#include <string>
#include <unordered_map>

#include "tools/stats/spirv_stats.h"

namespace spvtools {
namespace stats {

class StatsAnalyzer {
 public:
  explicit StatsAnalyzer(const SpirvStats& stats);

  // Writes respective histograms to |out|.
  void WriteVersion(std::ostream& out);
  void WriteGenerator(std::ostream& out);
  void WriteCapability(std::ostream& out);
  void WriteExtension(std::ostream& out);
  void WriteOpcode(std::ostream& out);
  void WriteConstantLiterals(std::ostream& out);

  // Writes first order Markov analysis to |out|.
  // stats_.opcode_markov_hist needs to contain raw data for at least one
  // level.
  void WriteOpcodeMarkov(std::ostream& out);

 private:
  const SpirvStats& stats_;

  uint32_t num_modules_;

  std::unordered_map<uint32_t, double> version_freq_;
  std::unordered_map<uint32_t, double> generator_freq_;
  std::unordered_map<uint32_t, double> capability_freq_;
  std::unordered_map<std::string, double> extension_freq_;
  std::unordered_map<uint32_t, double> opcode_freq_;
};

}  // namespace stats
}  // namespace spvtools

#endif  // TOOLS_STATS_STATS_ANALYZER_H_

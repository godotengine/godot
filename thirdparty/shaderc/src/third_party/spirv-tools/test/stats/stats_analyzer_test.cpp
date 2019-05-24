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

// Tests for unique type declaration rules validator.

#include <sstream>
#include <string>

#include "source/latest_version_spirv_header.h"
#include "test/test_fixture.h"
#include "tools/stats/stats_analyzer.h"

namespace spvtools {
namespace stats {
namespace {

// Fills |stats| with some synthetic header stats, as if aggregated from 100
// modules (100 used for simpler percentage evaluation).
void FillDefaultStats(SpirvStats* stats) {
  *stats = SpirvStats();
  stats->version_hist[0x00010000] = 40;
  stats->version_hist[0x00010100] = 60;
  stats->generator_hist[0x00000000] = 64;
  stats->generator_hist[0x00010000] = 1;
  stats->generator_hist[0x00020000] = 2;
  stats->generator_hist[0x00030000] = 3;
  stats->generator_hist[0x00040000] = 4;
  stats->generator_hist[0x00050000] = 5;
  stats->generator_hist[0x00060000] = 6;
  stats->generator_hist[0x00070000] = 7;
  stats->generator_hist[0x00080000] = 8;

  int num_version_entries = 0;
  for (const auto& pair : stats->version_hist) {
    num_version_entries += pair.second;
  }

  int num_generator_entries = 0;
  for (const auto& pair : stats->generator_hist) {
    num_generator_entries += pair.second;
  }

  EXPECT_EQ(num_version_entries, num_generator_entries);
}

TEST(StatsAnalyzer, Version) {
  SpirvStats stats;
  FillDefaultStats(&stats);

  StatsAnalyzer analyzer(stats);

  std::stringstream ss;
  analyzer.WriteVersion(ss);
  const std::string output = ss.str();
  const std::string expected_output = "Version 1.1 60%\nVersion 1.0 40%\n";

  EXPECT_EQ(expected_output, output);
}

TEST(StatsAnalyzer, Generator) {
  SpirvStats stats;
  FillDefaultStats(&stats);

  StatsAnalyzer analyzer(stats);

  std::stringstream ss;
  analyzer.WriteGenerator(ss);
  const std::string output = ss.str();
  const std::string expected_output =
      "Khronos 64%\nKhronos Glslang Reference Front End 8%\n"
      "Khronos SPIR-V Tools Assembler 7%\nKhronos LLVM/SPIR-V Translator 6%"
      "\nARM 5%\nNVIDIA 4%\nCodeplay 3%\nValve 2%\nLunarG 1%\n";

  EXPECT_EQ(expected_output, output);
}

TEST(StatsAnalyzer, Capability) {
  SpirvStats stats;
  FillDefaultStats(&stats);

  stats.capability_hist[SpvCapabilityShader] = 25;
  stats.capability_hist[SpvCapabilityKernel] = 75;

  StatsAnalyzer analyzer(stats);

  std::stringstream ss;
  analyzer.WriteCapability(ss);
  const std::string output = ss.str();
  const std::string expected_output = "Kernel 75%\nShader 25%\n";

  EXPECT_EQ(expected_output, output);
}

TEST(StatsAnalyzer, Extension) {
  SpirvStats stats;
  FillDefaultStats(&stats);

  stats.extension_hist["greatest_extension_ever"] = 1;
  stats.extension_hist["worst_extension_ever"] = 10;

  StatsAnalyzer analyzer(stats);

  std::stringstream ss;
  analyzer.WriteExtension(ss);
  const std::string output = ss.str();
  const std::string expected_output =
      "worst_extension_ever 10%\ngreatest_extension_ever 1%\n";

  EXPECT_EQ(expected_output, output);
}

TEST(StatsAnalyzer, Opcode) {
  SpirvStats stats;
  FillDefaultStats(&stats);

  stats.opcode_hist[SpvOpCapability] = 20;
  stats.opcode_hist[SpvOpConstant] = 80;
  stats.opcode_hist[SpvOpDecorate] = 100;

  StatsAnalyzer analyzer(stats);

  std::stringstream ss;
  analyzer.WriteOpcode(ss);
  const std::string output = ss.str();
  const std::string expected_output =
      "Total unique opcodes used: 3\nDecorate 50%\n"
      "Constant 40%\nCapability 10%\n";

  EXPECT_EQ(expected_output, output);
}

TEST(StatsAnalyzer, OpcodeMarkov) {
  SpirvStats stats;
  FillDefaultStats(&stats);

  stats.opcode_hist[SpvOpFMul] = 400;
  stats.opcode_hist[SpvOpFAdd] = 200;
  stats.opcode_hist[SpvOpFSub] = 400;

  stats.opcode_markov_hist.resize(1);
  auto& hist = stats.opcode_markov_hist[0];
  hist[SpvOpFMul][SpvOpFAdd] = 100;
  hist[SpvOpFMul][SpvOpFSub] = 300;
  hist[SpvOpFAdd][SpvOpFMul] = 100;
  hist[SpvOpFAdd][SpvOpFAdd] = 100;

  StatsAnalyzer analyzer(stats);

  std::stringstream ss;
  analyzer.WriteOpcodeMarkov(ss);
  const std::string output = ss.str();
  const std::string expected_output =
      "FMul -> FSub 75% (base rate 40%, pair occurrences 300)\n"
      "FMul -> FAdd 25% (base rate 20%, pair occurrences 100)\n"
      "FAdd -> FAdd 50% (base rate 20%, pair occurrences 100)\n"
      "FAdd -> FMul 50% (base rate 40%, pair occurrences 100)\n";

  EXPECT_EQ(expected_output, output);
}

}  // namespace
}  // namespace stats
}  // namespace spvtools

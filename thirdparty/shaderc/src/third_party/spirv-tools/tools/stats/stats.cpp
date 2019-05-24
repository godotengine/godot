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

#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "spirv-tools/libspirv.h"
#include "tools/io.h"
#include "tools/stats/spirv_stats.h"
#include "tools/stats/stats_analyzer.h"

namespace {

void PrintUsage(char* argv0) {
  printf(
      R"(%s - Collect statistics from one or more SPIR-V binary file(s).

USAGE: %s [options] [<filepaths>]

TIP: In order to collect statistics from all .spv files under current dir use
find . -name "*.spv" -print0 | xargs -0 -s 2000000 %s

Options:
  -h, --help
                   Print this help.

  -v, --verbose
                   Print additional info to stderr.
)",
      argv0, argv0, argv0);
}

void DiagnosticsMessageHandler(spv_message_level_t level, const char*,
                               const spv_position_t& position,
                               const char* message) {
  switch (level) {
    case SPV_MSG_FATAL:
    case SPV_MSG_INTERNAL_ERROR:
    case SPV_MSG_ERROR:
      std::cerr << "error: " << position.index << ": " << message << std::endl;
      break;
    case SPV_MSG_WARNING:
      std::cout << "warning: " << position.index << ": " << message
                << std::endl;
      break;
    case SPV_MSG_INFO:
      std::cout << "info: " << position.index << ": " << message << std::endl;
      break;
    default:
      break;
  }
}

}  // namespace

int main(int argc, char** argv) {
  bool continue_processing = true;
  int return_code = 0;

  bool expect_output_path = false;
  bool verbose = false;

  std::vector<const char*> paths;
  const char* output_path = nullptr;

  for (int argi = 1; continue_processing && argi < argc; ++argi) {
    const char* cur_arg = argv[argi];
    if ('-' == cur_arg[0]) {
      if (0 == strcmp(cur_arg, "--help") || 0 == strcmp(cur_arg, "-h")) {
        PrintUsage(argv[0]);
        continue_processing = false;
        return_code = 0;
      } else if (0 == strcmp(cur_arg, "--verbose") ||
                 0 == strcmp(cur_arg, "-v")) {
        verbose = true;
      } else if (0 == strcmp(cur_arg, "--output") ||
                 0 == strcmp(cur_arg, "-o")) {
        expect_output_path = true;
      } else {
        PrintUsage(argv[0]);
        continue_processing = false;
        return_code = 1;
      }
    } else {
      if (expect_output_path) {
        output_path = cur_arg;
        expect_output_path = false;
      } else {
        paths.push_back(cur_arg);
      }
    }
  }

  // Exit if command line parsing was not successful.
  if (!continue_processing) {
    return return_code;
  }

  std::cerr << "Processing " << paths.size() << " files..." << std::endl;

  spvtools::Context ctx(SPV_ENV_UNIVERSAL_1_1);
  ctx.SetMessageConsumer(DiagnosticsMessageHandler);

  spvtools::stats::SpirvStats stats;
  stats.opcode_markov_hist.resize(1);

  for (size_t index = 0; index < paths.size(); ++index) {
    const size_t kMilestonePeriod = 1000;
    if (verbose) {
      if (index % kMilestonePeriod == kMilestonePeriod - 1)
        std::cerr << "Processed " << index + 1 << " files..." << std::endl;
    }

    const char* path = paths[index];
    std::vector<uint32_t> contents;
    if (!ReadFile<uint32_t>(path, "rb", &contents)) return 1;

    if (SPV_SUCCESS !=
        spvtools::stats::AggregateStats(ctx.CContext(), contents.data(),
                                        contents.size(), nullptr, &stats)) {
      std::cerr << "error: Failed to aggregate stats for " << path << std::endl;
      return 1;
    }
  }

  spvtools::stats::StatsAnalyzer analyzer(stats);

  std::ofstream fout;
  if (output_path) {
    fout.open(output_path);
    if (!fout.is_open()) {
      std::cerr << "error: Failed to open " << output_path << std::endl;
      return 1;
    }
  }

  std::ostream& out = fout.is_open() ? fout : std::cout;
  out << std::endl;
  analyzer.WriteVersion(out);
  analyzer.WriteGenerator(out);

  out << std::endl;
  analyzer.WriteCapability(out);

  out << std::endl;
  analyzer.WriteExtension(out);

  out << std::endl;
  analyzer.WriteOpcode(out);

  out << std::endl;
  analyzer.WriteOpcodeMarkov(out);

  out << std::endl;
  analyzer.WriteConstantLiterals(out);

  return 0;
}

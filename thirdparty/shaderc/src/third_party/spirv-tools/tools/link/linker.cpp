// Copyright (c) 2017 Pierre Moreau
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

#include <cstring>
#include <iostream>
#include <vector>

#include "source/spirv_target_env.h"
#include "source/table.h"
#include "spirv-tools/libspirv.hpp"
#include "spirv-tools/linker.hpp"
#include "tools/io.h"

void print_usage(char* argv0) {
  printf(
      R"(%s - Link SPIR-V binary files together.

USAGE: %s [options] <filename> [<filename> ...]

The SPIR-V binaries are read from the different <filename>.

NOTE: The linker is a work in progress.

Options:
  -h, --help              Print this help.
  -o                      Name of the resulting linked SPIR-V binary.
  --create-library        Link the binaries into a library, keeping all exported symbols.
  --allow-partial-linkage Allow partial linkage by accepting imported symbols to be unresolved.
  --verify-ids            Verify that IDs in the resulting modules are truly unique.
  --version               Display linker version information
  --target-env            {vulkan1.0|spv1.0|spv1.1|spv1.2|opencl2.1|opencl2.2}
                          Use Vulkan1.0/SPIR-V1.0/SPIR-V1.1/SPIR-V1.2/OpenCL-2.1/OpenCL2.2 validation rules.
)",
      argv0, argv0);
}

int main(int argc, char** argv) {
  std::vector<const char*> inFiles;
  const char* outFile = nullptr;
  spv_target_env target_env = SPV_ENV_UNIVERSAL_1_0;
  spvtools::LinkerOptions options;
  bool continue_processing = true;
  int return_code = 0;

  for (int argi = 1; continue_processing && argi < argc; ++argi) {
    const char* cur_arg = argv[argi];
    if ('-' == cur_arg[0]) {
      if (0 == strcmp(cur_arg, "-o")) {
        if (argi + 1 < argc) {
          if (!outFile) {
            outFile = argv[++argi];
          } else {
            fprintf(stderr, "error: More than one output file specified\n");
            continue_processing = false;
            return_code = 1;
          }
        } else {
          fprintf(stderr, "error: Missing argument to %s\n", cur_arg);
          continue_processing = false;
          return_code = 1;
        }
      } else if (0 == strcmp(cur_arg, "--create-library")) {
        options.SetCreateLibrary(true);
      } else if (0 == strcmp(cur_arg, "--verify-ids")) {
        options.SetVerifyIds(true);
      } else if (0 == strcmp(cur_arg, "--allow-partial-linkage")) {
        options.SetAllowPartialLinkage(true);
      } else if (0 == strcmp(cur_arg, "--version")) {
        printf("%s\n", spvSoftwareVersionDetailsString());
        // TODO(dneto): Add OpenCL 2.2 at least.
        printf("Targets:\n  %s\n  %s\n  %s\n",
               spvTargetEnvDescription(SPV_ENV_UNIVERSAL_1_1),
               spvTargetEnvDescription(SPV_ENV_VULKAN_1_0),
               spvTargetEnvDescription(SPV_ENV_UNIVERSAL_1_2));
        continue_processing = false;
        return_code = 0;
      } else if (0 == strcmp(cur_arg, "--help") || 0 == strcmp(cur_arg, "-h")) {
        print_usage(argv[0]);
        continue_processing = false;
        return_code = 0;
      } else if (0 == strcmp(cur_arg, "--target-env")) {
        if (argi + 1 < argc) {
          const auto env_str = argv[++argi];
          if (!spvParseTargetEnv(env_str, &target_env)) {
            fprintf(stderr, "error: Unrecognized target env: %s\n", env_str);
            continue_processing = false;
            return_code = 1;
          }
        } else {
          fprintf(stderr, "error: Missing argument to --target-env\n");
          continue_processing = false;
          return_code = 1;
        }
      }
    } else {
      inFiles.push_back(cur_arg);
    }
  }

  // Exit if command line parsing was not successful.
  if (!continue_processing) {
    return return_code;
  }

  if (inFiles.empty()) {
    fprintf(stderr, "error: No input file specified\n");
    return 1;
  }

  std::vector<std::vector<uint32_t>> contents(inFiles.size());
  for (size_t i = 0u; i < inFiles.size(); ++i) {
    if (!ReadFile<uint32_t>(inFiles[i], "rb", &contents[i])) return 1;
  }

  const spvtools::MessageConsumer consumer = [](spv_message_level_t level,
                                                const char*,
                                                const spv_position_t& position,
                                                const char* message) {
    switch (level) {
      case SPV_MSG_FATAL:
      case SPV_MSG_INTERNAL_ERROR:
      case SPV_MSG_ERROR:
        std::cerr << "error: " << position.index << ": " << message
                  << std::endl;
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
  };
  spvtools::Context context(target_env);
  context.SetMessageConsumer(consumer);

  std::vector<uint32_t> linkingResult;
  spv_result_t status = Link(context, contents, &linkingResult, options);

  if (!WriteFile<uint32_t>(outFile, "wb", linkingResult.data(),
                           linkingResult.size()))
    return 1;

  return status == SPV_SUCCESS ? 0 : 1;
}

// Copyright (c) 2015-2016 The Khronos Group Inc.
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

#include <cstdio>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "spirv-tools/libspirv.h"
#include "tools/cfg/bin_to_dot.h"
#include "tools/io.h"

// Prints a program usage message to stdout.
static void print_usage(const char* argv0) {
  printf(
      R"(%s - Show the control flow graph in GraphiViz "dot" form. EXPERIMENTAL

Usage: %s [options] [<filename>]

The SPIR-V binary is read from <filename>. If no file is specified,
or if the filename is "-", then the binary is read from standard input.

Options:

  -h, --help      Print this help.
  --version       Display version information.

  -o <filename>   Set the output filename.
                  Output goes to standard output if this option is
                  not specified, or if the filename is "-".
)",
      argv0, argv0);
}

static const auto kDefaultEnvironment = SPV_ENV_UNIVERSAL_1_2;

int main(int argc, char** argv) {
  const char* inFile = nullptr;
  const char* outFile = nullptr;  // Stays nullptr if printing to stdout.

  for (int argi = 1; argi < argc; ++argi) {
    if ('-' == argv[argi][0]) {
      switch (argv[argi][1]) {
        case 'h':
          print_usage(argv[0]);
          return 0;
        case 'o': {
          if (!outFile && argi + 1 < argc) {
            outFile = argv[++argi];
          } else {
            print_usage(argv[0]);
            return 1;
          }
        } break;
        case '-': {
          // Long options
          if (0 == strcmp(argv[argi], "--help")) {
            print_usage(argv[0]);
            return 0;
          } else if (0 == strcmp(argv[argi], "--version")) {
            printf("%s EXPERIMENTAL\n", spvSoftwareVersionDetailsString());
            printf("Target: %s\n",
                   spvTargetEnvDescription(kDefaultEnvironment));
            return 0;
          } else {
            print_usage(argv[0]);
            return 1;
          }
        } break;
        case 0: {
          // Setting a filename of "-" to indicate stdin.
          if (!inFile) {
            inFile = argv[argi];
          } else {
            fprintf(stderr, "error: More than one input file specified\n");
            return 1;
          }
        } break;
        default:
          print_usage(argv[0]);
          return 1;
      }
    } else {
      if (!inFile) {
        inFile = argv[argi];
      } else {
        fprintf(stderr, "error: More than one input file specified\n");
        return 1;
      }
    }
  }

  // Read the input binary.
  std::vector<uint32_t> contents;
  if (!ReadFile<uint32_t>(inFile, "rb", &contents)) return 1;
  spv_context context = spvContextCreate(kDefaultEnvironment);
  spv_diagnostic diagnostic = nullptr;

  std::stringstream ss;
  auto error =
      BinaryToDot(context, contents.data(), contents.size(), &ss, &diagnostic);
  if (error) {
    spvDiagnosticPrint(diagnostic);
    spvDiagnosticDestroy(diagnostic);
    spvContextDestroy(context);
    return error;
  }
  std::string str = ss.str();
  WriteFile(outFile, "w", str.data(), str.size());

  spvDiagnosticDestroy(diagnostic);
  spvContextDestroy(context);

  return 0;
}

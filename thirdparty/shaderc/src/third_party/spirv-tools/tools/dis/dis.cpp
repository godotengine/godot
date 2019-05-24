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

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
#include <stdio.h>  // Need fileno
#include <unistd.h>
#endif

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "spirv-tools/libspirv.h"
#include "tools/io.h"

static void print_usage(char* argv0) {
  printf(
      R"(%s - Disassemble a SPIR-V binary module

Usage: %s [options] [<filename>]

The SPIR-V binary is read from <filename>. If no file is specified,
or if the filename is "-", then the binary is read from standard input.

Options:

  -h, --help      Print this help.
  --version       Display disassembler version information.

  -o <filename>   Set the output filename.
                  Output goes to standard output if this option is
                  not specified, or if the filename is "-".

  --color         Force color output.  The default when printing to a terminal.
                  Overrides a previous --no-color option.
  --no-color      Don't print in color.  Overrides a previous --color option.
                  The default when output goes to something other than a
                  terminal (e.g. a file, a pipe, or a shell redirection).

  --no-indent     Don't indent instructions.

  --no-header     Don't output the header as leading comments.

  --raw-id        Show raw Id values instead of friendly names.

  --offsets       Show byte offsets for each instruction.
)",
      argv0, argv0);
}

static const auto kDefaultEnvironment = SPV_ENV_UNIVERSAL_1_3;

int main(int argc, char** argv) {
  const char* inFile = nullptr;
  const char* outFile = nullptr;

  bool color_is_possible =
#if SPIRV_COLOR_TERMINAL
      true;
#else
      false;
#endif
  bool force_color = false;
  bool force_no_color = false;

  bool allow_indent = true;
  bool show_byte_offsets = false;
  bool no_header = false;
  bool friendly_names = true;

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
          if (0 == strcmp(argv[argi], "--no-color")) {
            force_no_color = true;
            force_color = false;
          } else if (0 == strcmp(argv[argi], "--color")) {
            force_no_color = false;
            force_color = true;
          } else if (0 == strcmp(argv[argi], "--no-indent")) {
            allow_indent = false;
          } else if (0 == strcmp(argv[argi], "--offsets")) {
            show_byte_offsets = true;
          } else if (0 == strcmp(argv[argi], "--no-header")) {
            no_header = true;
          } else if (0 == strcmp(argv[argi], "--raw-id")) {
            friendly_names = false;
          } else if (0 == strcmp(argv[argi], "--help")) {
            print_usage(argv[0]);
            return 0;
          } else if (0 == strcmp(argv[argi], "--version")) {
            printf("%s\n", spvSoftwareVersionDetailsString());
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

  uint32_t options = SPV_BINARY_TO_TEXT_OPTION_NONE;

  if (allow_indent) options |= SPV_BINARY_TO_TEXT_OPTION_INDENT;

  if (show_byte_offsets) options |= SPV_BINARY_TO_TEXT_OPTION_SHOW_BYTE_OFFSET;

  if (no_header) options |= SPV_BINARY_TO_TEXT_OPTION_NO_HEADER;

  if (friendly_names) options |= SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES;

  if (!outFile || (0 == strcmp("-", outFile))) {
    // Print to standard output.
    options |= SPV_BINARY_TO_TEXT_OPTION_PRINT;

    if (color_is_possible && !force_no_color) {
      bool output_is_tty = true;
#if defined(_POSIX_VERSION)
      output_is_tty = isatty(fileno(stdout));
#endif
      if (output_is_tty || force_color) {
        options |= SPV_BINARY_TO_TEXT_OPTION_COLOR;
      }
    }
  }

  // Read the input binary.
  std::vector<uint32_t> contents;
  if (!ReadFile<uint32_t>(inFile, "rb", &contents)) return 1;

  // If printing to standard output, then spvBinaryToText should
  // do the printing.  In particular, colour printing on Windows is
  // controlled by modifying console objects synchronously while
  // outputting to the stream rather than by injecting escape codes
  // into the output stream.
  // If the printing option is off, then save the text in memory, so
  // it can be emitted later in this function.
  const bool print_to_stdout = SPV_BINARY_TO_TEXT_OPTION_PRINT & options;
  spv_text text = nullptr;
  spv_text* textOrNull = print_to_stdout ? nullptr : &text;
  spv_diagnostic diagnostic = nullptr;
  spv_context context = spvContextCreate(kDefaultEnvironment);
  spv_result_t error =
      spvBinaryToText(context, contents.data(), contents.size(), options,
                      textOrNull, &diagnostic);
  spvContextDestroy(context);
  if (error) {
    spvDiagnosticPrint(diagnostic);
    spvDiagnosticDestroy(diagnostic);
    return error;
  }

  if (!print_to_stdout) {
    if (!WriteFile<char>(outFile, "w", text->str, text->length)) {
      spvTextDestroy(text);
      return 1;
    }
  }
  spvTextDestroy(text);

  return 0;
}

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

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "source/comp/markv.h"
#include "source/spirv_target_env.h"
#include "source/table.h"
#include "spirv-tools/optimizer.hpp"
#include "tools/comp/markv_model_factory.h"
#include "tools/io.h"

namespace {

const auto kSpvEnv = SPV_ENV_UNIVERSAL_1_2;

enum Task {
  kNoTask = 0,
  kEncode,
  kDecode,
  kTest,
};

struct ScopedContext {
  ScopedContext(spv_target_env env) : context(spvContextCreate(env)) {}
  ~ScopedContext() { spvContextDestroy(context); }
  spv_context context;
};

void print_usage(char* argv0) {
  printf(
      R"(%s - Encodes or decodes a SPIR-V binary to or from a MARK-V binary.

USAGE: %s [e|d|t] [options] [<filename>]

The input binary is read from <filename>. If no file is specified,
or if the filename is "-", then the binary is read from standard input.

If no output is specified then the output is printed to stdout in a human
readable format.

WIP: MARK-V codec is in early stages of development. At the moment it only
can encode and decode some SPIR-V files and only if exacly the same build of
software is used (is doesn't write or handle version numbers yet).

Tasks:
  e               Encode SPIR-V to MARK-V.
  d               Decode MARK-V to SPIR-V.
  t               Test the codec by first encoding the given SPIR-V file to
                  MARK-V, then decoding it back to SPIR-V and comparing results.

Options:
  -h, --help      Print this help.
  --comments      Write codec comments to stderr.
  --version       Display MARK-V codec version.
  --validate      Validate SPIR-V while encoding or decoding.
  --model=<model-name>
                  Compression model, possible values:
                  shader_lite - fast, poor compression ratio
                  shader_mid - balanced
                  shader_max - best compression ratio
                  Default: shader_lite

  -o <filename>   Set the output filename.
                  Output goes to standard output if this option is
                  not specified, or if the filename is "-".
                  Not needed for 't' task (testing).
)",
      argv0, argv0);
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
      std::cerr << "warning: " << position.index << ": " << message
                << std::endl;
      break;
    case SPV_MSG_INFO:
      std::cerr << "info: " << position.index << ": " << message << std::endl;
      break;
    default:
      break;
  }
}

}  // namespace

int main(int argc, char** argv) {
  const char* input_filename = nullptr;
  const char* output_filename = nullptr;

  Task task = kNoTask;

  if (argc < 3) {
    print_usage(argv[0]);
    return 0;
  }

  const char* task_char = argv[1];
  if (0 == strcmp("e", task_char)) {
    task = kEncode;
  } else if (0 == strcmp("d", task_char)) {
    task = kDecode;
  } else if (0 == strcmp("t", task_char)) {
    task = kTest;
  }

  if (task == kNoTask) {
    print_usage(argv[0]);
    return 1;
  }

  bool want_comments = false;
  bool validate_spirv_binary = false;

  spvtools::comp::MarkvModelType model_type =
      spvtools::comp::kMarkvModelUnknown;

  for (int argi = 2; argi < argc; ++argi) {
    if ('-' == argv[argi][0]) {
      switch (argv[argi][1]) {
        case 'h':
          print_usage(argv[0]);
          return 0;
        case 'o': {
          if (!output_filename && argi + 1 < argc &&
              (task == kEncode || task == kDecode)) {
            output_filename = argv[++argi];
          } else {
            print_usage(argv[0]);
            return 1;
          }
        } break;
        case '-': {
          if (0 == strcmp(argv[argi], "--help")) {
            print_usage(argv[0]);
            return 0;
          } else if (0 == strcmp(argv[argi], "--comments")) {
            want_comments = true;
          } else if (0 == strcmp(argv[argi], "--version")) {
            fprintf(stderr, "error: Not implemented\n");
            return 1;
          } else if (0 == strcmp(argv[argi], "--validate")) {
            validate_spirv_binary = true;
          } else if (0 == strcmp(argv[argi], "--model=shader_lite")) {
            if (model_type != spvtools::comp::kMarkvModelUnknown)
              fprintf(stderr, "error: More than one model specified\n");
            model_type = spvtools::comp::kMarkvModelShaderLite;
          } else if (0 == strcmp(argv[argi], "--model=shader_mid")) {
            if (model_type != spvtools::comp::kMarkvModelUnknown)
              fprintf(stderr, "error: More than one model specified\n");
            model_type = spvtools::comp::kMarkvModelShaderMid;
          } else if (0 == strcmp(argv[argi], "--model=shader_max")) {
            if (model_type != spvtools::comp::kMarkvModelUnknown)
              fprintf(stderr, "error: More than one model specified\n");
            model_type = spvtools::comp::kMarkvModelShaderMax;
          } else {
            print_usage(argv[0]);
            return 1;
          }
        } break;
        case '\0': {
          // Setting a filename of "-" to indicate stdin.
          if (!input_filename) {
            input_filename = argv[argi];
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
      if (!input_filename) {
        input_filename = argv[argi];
      } else {
        fprintf(stderr, "error: More than one input file specified\n");
        return 1;
      }
    }
  }

  if (model_type == spvtools::comp::kMarkvModelUnknown)
    model_type = spvtools::comp::kMarkvModelShaderLite;

  const auto no_comments = spvtools::comp::MarkvLogConsumer();
  const auto output_to_stderr = [](const std::string& str) {
    std::cerr << str;
  };

  ScopedContext ctx(kSpvEnv);

  std::unique_ptr<spvtools::comp::MarkvModel> model =
      spvtools::comp::CreateMarkvModel(model_type);

  std::vector<uint32_t> spirv;
  std::vector<uint8_t> markv;

  spvtools::comp::MarkvCodecOptions options;
  options.validate_spirv_binary = validate_spirv_binary;

  if (task == kEncode) {
    if (!ReadFile<uint32_t>(input_filename, "rb", &spirv)) return 1;
    assert(!spirv.empty());

    if (SPV_SUCCESS != spvtools::comp::SpirvToMarkv(
                           ctx.context, spirv, options, *model,
                           DiagnosticsMessageHandler,
                           want_comments ? output_to_stderr : no_comments,
                           spvtools::comp::MarkvDebugConsumer(), &markv)) {
      std::cerr << "error: Failed to encode " << input_filename << " to MARK-V "
                << std::endl;
      return 1;
    }

    if (!WriteFile<uint8_t>(output_filename, "wb", markv.data(), markv.size()))
      return 1;
  } else if (task == kDecode) {
    if (!ReadFile<uint8_t>(input_filename, "rb", &markv)) return 1;
    assert(!markv.empty());

    if (SPV_SUCCESS != spvtools::comp::MarkvToSpirv(
                           ctx.context, markv, options, *model,
                           DiagnosticsMessageHandler,
                           want_comments ? output_to_stderr : no_comments,
                           spvtools::comp::MarkvDebugConsumer(), &spirv)) {
      std::cerr << "error: Failed to decode " << input_filename << " to SPIR-V "
                << std::endl;
      return 1;
    }

    if (!WriteFile<uint32_t>(output_filename, "wb", spirv.data(), spirv.size()))
      return 1;
  } else if (task == kTest) {
    if (!ReadFile<uint32_t>(input_filename, "rb", &spirv)) return 1;
    assert(!spirv.empty());

    std::vector<uint32_t> spirv_before;
    spvtools::Optimizer optimizer(kSpvEnv);
    optimizer.RegisterPass(spvtools::CreateCompactIdsPass());
    if (!optimizer.Run(spirv.data(), spirv.size(), &spirv_before)) {
      std::cerr << "error: Optimizer failure on: " << input_filename
                << std::endl;
    }

    std::vector<std::string> encoder_instruction_bits;
    std::vector<std::string> encoder_instruction_comments;
    std::vector<std::vector<uint32_t>> encoder_instruction_words;
    std::vector<std::string> decoder_instruction_bits;
    std::vector<std::string> decoder_instruction_comments;
    std::vector<std::vector<uint32_t>> decoder_instruction_words;

    const auto encoder_debug_consumer = [&](const std::vector<uint32_t>& words,
                                            const std::string& bits,
                                            const std::string& comment) {
      encoder_instruction_words.push_back(words);
      encoder_instruction_bits.push_back(bits);
      encoder_instruction_comments.push_back(comment);
      return true;
    };

    if (SPV_SUCCESS != spvtools::comp::SpirvToMarkv(
                           ctx.context, spirv_before, options, *model,
                           DiagnosticsMessageHandler,
                           want_comments ? output_to_stderr : no_comments,
                           encoder_debug_consumer, &markv)) {
      std::cerr << "error: Failed to encode " << input_filename << " to MARK-V "
                << std::endl;
      return 1;
    }

    const auto write_bug_report = [&]() {
      for (size_t inst_index = 0; inst_index < decoder_instruction_words.size();
           ++inst_index) {
        std::cerr << "\nInstruction #" << inst_index << std::endl;
        std::cerr << "\nEncoder words: ";
        for (uint32_t word : encoder_instruction_words[inst_index])
          std::cerr << word << " ";
        std::cerr << "\nDecoder words: ";
        for (uint32_t word : decoder_instruction_words[inst_index])
          std::cerr << word << " ";
        std::cerr << std::endl;

        std::cerr << "\nEncoder bits: " << encoder_instruction_bits[inst_index];
        std::cerr << "\nDecoder bits: " << decoder_instruction_bits[inst_index];
        std::cerr << std::endl;

        std::cerr << "\nEncoder comments:\n"
                  << encoder_instruction_comments[inst_index];
        std::cerr << "Decoder comments:\n"
                  << decoder_instruction_comments[inst_index];
        std::cerr << std::endl;
      }
    };

    const auto decoder_debug_consumer = [&](const std::vector<uint32_t>& words,
                                            const std::string& bits,
                                            const std::string& comment) {
      const size_t inst_index = decoder_instruction_words.size();
      if (inst_index >= encoder_instruction_words.size()) {
        write_bug_report();
        std::cerr << "error: Decoder has more instructions than encoder: "
                  << input_filename << std::endl;
        return false;
      }

      decoder_instruction_words.push_back(words);
      decoder_instruction_bits.push_back(bits);
      decoder_instruction_comments.push_back(comment);

      if (encoder_instruction_words[inst_index] !=
          decoder_instruction_words[inst_index]) {
        write_bug_report();
        std::cerr << "error: Words of the last decoded instruction differ from "
                     "reference: "
                  << input_filename << std::endl;
        return false;
      }

      if (encoder_instruction_bits[inst_index] !=
          decoder_instruction_bits[inst_index]) {
        write_bug_report();
        std::cerr << "error: Bits of the last decoded instruction differ from "
                     "reference: "
                  << input_filename << std::endl;
        return false;
      }
      return true;
    };

    std::vector<uint32_t> spirv_after;
    const spv_result_t decoding_result = spvtools::comp::MarkvToSpirv(
        ctx.context, markv, options, *model, DiagnosticsMessageHandler,
        want_comments ? output_to_stderr : no_comments, decoder_debug_consumer,
        &spirv_after);

    if (decoding_result == SPV_REQUESTED_TERMINATION) {
      std::cerr << "error: Decoding interrupted by the debugger: "
                << input_filename << std::endl;
      return 1;
    }

    if (decoding_result != SPV_SUCCESS) {
      std::cerr << "error: Failed to decode encoded " << input_filename
                << " back to SPIR-V " << std::endl;
      return 1;
    }

    assert(spirv_before.size() == spirv_after.size());
    assert(std::mismatch(std::next(spirv_before.begin(), 5), spirv_before.end(),
                         std::next(spirv_after.begin(), 5)) ==
           std::make_pair(spirv_before.end(), spirv_after.end()));
  }

  return 0;
}

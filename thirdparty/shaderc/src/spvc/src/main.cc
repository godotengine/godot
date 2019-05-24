// Copyright 2019 The Shaderc Authors. All rights reserved.
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

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

#include "libshaderc_util/string_piece.h"
#include "shaderc/env.h"
#include "shaderc/spvc.hpp"
#include "shaderc/status.h"
#include "spirv-tools/libspirv.h"

using shaderc_util::string_piece;

namespace {

// Prints the help message.
void PrintHelp(std::ostream* out) {
  *out << R"(spvc - Compile SPIR-V into GLSL/HLSL/MSL.

Usage: spvc [options] file

An input file of - represents standard input.

Options:
  --help                Display available options.
  -v                    Display compiler version information.
  -o <output file>      '-' means standard output.
  --validate=<env>      Validate SPIR-V source with given environment
                          <env> is vulkan1.0 (the default) or vulkan1.1
  --entry=<name>        Specify entry point.
  --language=<lang>     Specify output language.
                          <lang> is glsl (the default), msl or hlsl.
  --glsl-version=<ver>  Specify GLSL output language version, e.g. 100
                          Default is 450 if not detected from input.
  --msl-version=<ver>   Specify MSL output language version, e.g. 100
                          Default is 10200.

  The following flags behave as in spirv-cross:

  --remove-unused-variables
  --vulkan-semantics
  --separate-shader-objects
  --flatten-ubo
  --shader-model=<model>
)";
}

// TODO(fjhenigman): factor out with glslc
// Gets the option argument for the option at *index in argv in a way consistent
// with clang/gcc. On success, returns true and writes the parsed argument into
// *option_argument. Returns false if any errors occur. After calling this
// function, *index will be the index of the last command line argument
// consumed.
bool GetOptionArgument(int argc, char** argv, int* index,
                       const std::string& option,
                       string_piece* option_argument) {
  const string_piece arg = argv[*index];
  assert(arg.starts_with(option));
  if (arg.size() != option.size()) {
    *option_argument = arg.substr(option.size());
    return true;
  }
  if (option.back() == '=') {
    *option_argument = "";
    return true;
  }
  if (++(*index) >= argc)
    return false;
  *option_argument = argv[*index];
  return true;

}

// TODO(fjhenigman): factor out with glslc
// Parses the given string as a number of the specified type.  Returns true
// if parsing succeeded, and stores the parsed value via |value|.
bool ParseUint32(const std::string& str, uint32_t* value) {
  std::istringstream iss(str);

  iss >> std::setbase(0);
  iss >> *value;

  // We should have read something.
  bool ok = !str.empty() && !iss.bad();
  // It should have been all the text.
  ok = ok && iss.eof();
  // It should have been in range.
  ok = ok && !iss.fail();

  // Work around a bugs in various C++ standard libraries.
  // Count any negative number as an error, including "-0".
  ok = ok && (str[0] != '-');

  return ok;
}

const char kBuildVersion[] = ""
    // TODO(fjhenigman): #include "build-version.inc"
    ;

bool ReadFile(const std::string& path, std::vector<uint32_t>* out) {
  FILE* file =
      path == "-" ? freopen(nullptr, "rb", stdin) : fopen(path.c_str(), "rb");
  if (!file) {
    std::cerr << "Failed to open SPIR-V file: " << path << std::endl;
    return false;
  }

  fseek(file, 0, SEEK_END);
  out->resize(ftell(file) / sizeof((*out)[0]));
  rewind(file);

  if (fread(out->data(), sizeof((*out)[0]), out->size(), file) != out->size()) {
    std::cerr << "Failed to read SPIR-V file: " << path << std::endl;
    out->clear();
    return false;
  }

  fclose(file);
  return true;
}

}  // anonymous namespace

int main(int argc, char** argv) {
  shaderc_spvc::Compiler compiler;
  shaderc_spvc::CompileOptions options;
  std::vector<uint32_t> input;
  string_piece output_path;
  string_piece output_language;

  for (int i = 1; i < argc; ++i) {
    const string_piece arg = argv[i];
    if (arg == "--help") {
      ::PrintHelp(&std::cout);
      return 0;
    } else if (arg == "-v") {
      std::cout << kBuildVersion << std::endl;
      std::cout << "Target: " << spvTargetEnvDescription(SPV_ENV_UNIVERSAL_1_0)
                << std::endl;
      return 0;
    } else if (arg.starts_with("-o")) {
      if (!GetOptionArgument(argc, argv, &i, "-o", &output_path)) {
        std::cerr
            << "spvc: error: argument to '-o' is missing (expected 1 value)"
            << std::endl;
        return 1;
      }
    } else if (arg.starts_with("--entry=")) {
      string_piece entry_point;
      GetOptionArgument(argc, argv, &i, "--entry=", &entry_point);
      options.SetEntryPoint(entry_point.data());
    } else if (arg.starts_with("--glsl-version=")) {
      string_piece version_str;
      GetOptionArgument(argc, argv, &i, "--glsl-version=", &version_str);
      uint32_t version_num;
      if (!ParseUint32(version_str.str(), &version_num)) {
        std::cerr << "spvc: error: invalid value '" << version_str
                  << "' in --glsl-version=" << std::endl;
        return 1;
      }
      options.SetGLSLLanguageVersion(version_num);
    } else if (arg.starts_with("--msl-version=")) {
      string_piece version_str;
      GetOptionArgument(argc, argv, &i, "--msl-version=", &version_str);
      uint32_t version_num;
      if (!ParseUint32(version_str.str(), &version_num)) {
        std::cerr << "spvc: error: invalid value '" << version_str
                  << "' in --msl-version=" << std::endl;
        return 1;
      }
      options.SetMSLLanguageVersion(version_num);
    } else if (arg.starts_with("--language=")) {
      GetOptionArgument(argc, argv, &i, "--language=", &output_language);
      if (!(output_language == "glsl" || output_language == "msl" ||
            output_language == "hlsl")) {
        std::cerr << "spvc: error: invalid value '" << output_language
                  << "' in --language=" << std::endl;
        return 1;
      }
    } else if (arg == "--remove-unused-variables") {
      options.SetRemoveUnusedVariables(true);
    } else if (arg == "--vulkan-semantics") {
      options.SetVulkanSemantics(true);
    } else if (arg == "--separate-shader-objects") {
      options.SetSeparateShaderObjects(true);
    } else if (arg == "--flatten-ubo") {
      options.SetFlattenUbo(true);
    } else if (arg == "--flatten-multidimensional-arrays") {
      // TODO(fjhenigman)
    } else if (arg == "--es") {
      // TODO(fjhenigman)
    } else if (arg == "--hlsl-enable-compat") {
      // TODO(fjhenigman)
    } else if (arg == "--glsl-emit-push-constant-as-ubo") {
      // TODO(fjhenigman)
    } else if (arg.starts_with("--shader-model=")) {
      string_piece shader_model_str;
      GetOptionArgument(argc, argv, &i, "--shader-model=", &shader_model_str);
      uint32_t shader_model_num;
      if (!ParseUint32(shader_model_str.str(), &shader_model_num)) {
        std::cerr << "spvc: error: invalid value '" << shader_model_str
                  << "' in --shader-model=" << std::endl;
        return 1;
      }
      options.SetShaderModel(shader_model_num);
    } else if (arg.starts_with("--validate=")) {
      string_piece target_env;
      GetOptionArgument(argc, argv, &i, "--validate=", &target_env);
      if (target_env == "vulkan1.0") {
        options.SetTargetEnvironment(shaderc_target_env_vulkan,
                                     shaderc_env_version_vulkan_1_0);
      } else if (target_env == "vulkan1.1") {
        options.SetTargetEnvironment(shaderc_target_env_vulkan,
                                     shaderc_env_version_vulkan_1_1);
      } else {
        std::cerr << "spvc: error: invalid value '" << target_env
                  << "' in --validate=" << std::endl;
        return 1;
      }
    } else {
      if (!ReadFile(arg.str(), &input)) {
        std::cerr << "spvc: error: could not read file" << std::endl;
        return 1;
      }
    }
  }

  shaderc_spvc::CompilationResult result;
  if (output_language == "glsl") {
    result = compiler.CompileSpvToGlsl((const uint32_t*)input.data(),
                                       input.size(), options);
  } else if (output_language == "msl") {
    result = compiler.CompileSpvToMsl((const uint32_t*)input.data(),
                                      input.size(), options);
  } else if (output_language == "hlsl") {
    result = compiler.CompileSpvToHlsl((const uint32_t*)input.data(),
                                       input.size(), options);
  }
  auto status = result.GetCompilationStatus();
  if (status == shaderc_compilation_status_validation_error) {
    std::cerr << "validation failed:\n" << result.GetMessages() << std::endl;
    return 1;
  }
  if (status == shaderc_compilation_status_success) {
    const char* path = output_path.data();
    if (path && strcmp(path, "-")) {
      std::basic_ofstream<char>(path) << result.GetOutput();
    } else {
      std::cout << result.GetOutput();
    }
    return 0;
  }

  if (status == shaderc_compilation_status_compilation_error) {
    std::cerr << "compilation failed:\n" << result.GetMessages() << std::endl;
    return 1;
  }

  std::cerr << "unexpected error " << status << std::endl;
  return 1;
}

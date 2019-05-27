// Copyright 2015 The Shaderc Authors. All rights reserved.
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

#include "file_compiler.h"

#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "file.h"
#include "file_includer.h"
#include "shader_stage.h"

#include "libshaderc_util/io.h"
#include "libshaderc_util/message.h"

namespace {
using shaderc_util::string_piece;

// A helper function to emit SPIR-V binary code as a list of hex numbers in
// text form. Returns true if a non-empty compilation result is emitted
// successfully. Return false if nothing should be emitted, either because the
// compilation result is empty, or the compilation output is not SPIR-V binary
// code.
template <typename CompilationResultType>
bool EmitSpirvBinaryAsCommaSeparatedNumbers(const CompilationResultType& result,
                                            std::ostream* out) {
  // Return early if the compilation output is not in SPIR-V binary code form.
  if (!std::is_same<CompilationResultType,
                    shaderc::SpvCompilationResult>::value)
    return false;
  // Return early if the compilation result is empty.
  if (result.cbegin() == result.cend()) return false;
  std::ios::fmtflags output_stream_flag_cache(out->flags());
  *out << std::hex << std::setfill('0');
  auto RI = result.cbegin();
  *out << "0x" << std::setw(8) << *RI++;
  for (size_t counter = 1; RI != result.cend(); RI++, counter++) {
    *out << ",";
    // Break line for every four words.
    if (counter % 4 == 0) {
      *out << std::endl;
    }
    *out << "0x" << std::setw(8) << *RI;
  }
  out->flags(output_stream_flag_cache);
  return true;
}
}  // anonymous namespace

namespace glslc {
bool FileCompiler::CompileShaderFile(const InputFileSpec& input_file) {
  std::vector<char> input_data;
  std::string path = input_file.name;
  if (!shaderc_util::ReadFile(path, &input_data)) {
    return false;
  }

  std::string output_file_name = GetOutputFileName(input_file.name);
  string_piece error_file_name = input_file.name;

  if (error_file_name == "-") {
    // If the input file was stdin, we want to output errors as <stdin>.
    error_file_name = "<stdin>";
  }

  string_piece source_string = "";
  if (!input_data.empty()) {
    source_string = {&input_data.front(),
                     &input_data.front() + input_data.size()};
  }

  std::unique_ptr<FileIncluder> includer(
      new FileIncluder(&include_file_finder_));
  // Get a reference to the dependency trace before we pass the ownership to
  // shaderc::CompileOptions.
  const auto& used_source_files = includer->file_path_trace();
  options_.SetIncluder(std::move(includer));

  if (input_file.stage == shaderc_spirv_assembly) {
    // Only act if the requested target is SPIR-V binary.
    if (output_type_ == OutputType::SpirvBinary) {
      const auto result =
          compiler_.AssembleToSpv(source_string.data(), source_string.size());
      return EmitCompiledResult(result, input_file.name, output_file_name,
                                error_file_name, used_source_files);
    } else {
      return true;
    }
  }

  // Set the language.  Since we only use the options object in this
  // method, then it's ok to always set it without resetting it after
  // compilation.  A subsequent compilation will set it again anyway.
  options_.SetSourceLanguage(input_file.language);

  switch (output_type_) {
    case OutputType::SpirvBinary: {
      const auto result = compiler_.CompileGlslToSpv(
          source_string.data(), source_string.size(), input_file.stage,
          error_file_name.data(), input_file.entry_point_name.c_str(),
          options_);
      return EmitCompiledResult(result, input_file.name, output_file_name,
                                error_file_name, used_source_files);
    }
    case OutputType::SpirvAssemblyText: {
      const auto result = compiler_.CompileGlslToSpvAssembly(
          source_string.data(), source_string.size(), input_file.stage,
          error_file_name.data(), input_file.entry_point_name.c_str(),
          options_);
      return EmitCompiledResult(result, input_file.name, output_file_name,
                                error_file_name, used_source_files);
    }
    case OutputType::PreprocessedText: {
      const auto result = compiler_.PreprocessGlsl(
          source_string.data(), source_string.size(), input_file.stage,
          error_file_name.data(), options_);
      return EmitCompiledResult(result, input_file.name, output_file_name,
                                error_file_name, used_source_files);
    }
  }
  return false;
}

template <typename CompilationResultType>
bool FileCompiler::EmitCompiledResult(
    const CompilationResultType& result, const std::string& input_file,
    const std::string& output_file_name, string_piece error_file_name,
    const std::unordered_set<std::string>& used_source_files) {
  total_errors_ += result.GetNumErrors();
  total_warnings_ += result.GetNumWarnings();

  bool compilation_success =
      result.GetCompilationStatus() == shaderc_compilation_status_success;

  // Handle the error message for failing to deduce the shader kind.
  if (result.GetCompilationStatus() ==
      shaderc_compilation_status_invalid_stage) {
    auto glsl_or_hlsl_extension = GetGlslOrHlslExtension(error_file_name);
    if (glsl_or_hlsl_extension != "") {
      std::cerr << "glslc: error: "
                << "'" << error_file_name << "': "
                << "." << glsl_or_hlsl_extension
                << " file encountered but no -fshader-stage specified ahead";
    } else if (error_file_name == "<stdin>") {
      std::cerr
          << "glslc: error: '-': -fshader-stage required when input is from "
             "standard "
             "input \"-\"";
    } else {
      std::cerr << "glslc: error: "
                << "'" << error_file_name << "': "
                << "file not recognized: File format not recognized";
    }
    std::cerr << "\n";

    return false;
  }

  // Get a string_piece which refers to the normal compilation output for now.
  // This string_piece might be redirected to the dependency info to be dumped
  // later, if the handler is instantiated to dump as normal compilation output,
  // and the original compilation output should be blocked. Otherwise it won't
  // be touched. The main output stream dumps this string_piece later.
  string_piece compilation_output(
      reinterpret_cast<const char*>(result.cbegin()),
      reinterpret_cast<const char*>(result.cend()));

  // If we have dependency info dumping handler instantiated, we should dump
  // dependency info first. This may redirect the compilation output
  // string_piece to dependency info.
  std::string potential_dependency_info_output;
  if (dependency_info_dumping_handler_) {
    if (!dependency_info_dumping_handler_->DumpDependencyInfo(
            GetCandidateOutputFileName(input_file), error_file_name.data(),
            &potential_dependency_info_output, used_source_files)) {
      return false;
    }
    if (!potential_dependency_info_output.empty()) {
      // If the potential_dependency_info_output string is not empty, it means
      // we should dump dependency info as normal compilation output. Redirect
      // the compilation output string_piece to the dependency info stored in
      // potential_dependency_info_output to make it happen.
      compilation_output = potential_dependency_info_output;
    }
  }

  std::ostream* out = nullptr;
  std::ofstream potential_file_stream;
  if (compilation_success) {
    out = shaderc_util::GetOutputStream(output_file_name,
                                        &potential_file_stream, &std::cerr);
    if (!out || out->fail()) {
      // An error message has already been emitted to the stderr stream.
      return false;
    }

    // Write compilation output to output file. If an output format for SPIR-V
    // binary code is specified, it is handled here.
    switch (binary_emission_format_) {
      case SpirvBinaryEmissionFormat::Unspecified:
      case SpirvBinaryEmissionFormat::Binary:
        // The output format is unspecified or specified as binary output.
        // On Windows, the output stream must be set to binary mode.  By
        // default the standard output stream is set to text mode, which
        // translates newlines (\n) to carriage-return newline pairs
        // (\r\n).
        if (out == &std::cout) shaderc_util::FlushAndSetBinaryModeOnStdout();
        out->write(compilation_output.data(), compilation_output.size());
        if (out == &std::cout) shaderc_util::FlushAndSetTextModeOnStdout();
        break;
      case SpirvBinaryEmissionFormat::Numbers:
        // The output format is specified to be a list of hex numbers, the
        // compilation output must be in SPIR-V binary code form.
        assert(output_type_ == OutputType::SpirvBinary);
        if (EmitSpirvBinaryAsCommaSeparatedNumbers(result, out)) {
          // Only emits the end-of-line character when the emitted compilation
          // result is not empty.
          *out << std::endl;
        }
        break;
      case SpirvBinaryEmissionFormat::CInitList:
        // The output format is specified to be a C-style initializer list, the
        // compilation output must be in SPIR-V binary code form.
        assert(output_type_ == OutputType::SpirvBinary);
        if (result.begin() != result.end()) {
          // Only emits the '{' when the compilation result is not empty.
          *out << "{";
        }
        if (EmitSpirvBinaryAsCommaSeparatedNumbers(result, out)) {
          // Only emits the end-of-line character when the emitted compilation
          // result is not empty.
          *out << "}" << std::endl;
        }
        break;
    }
  }

  // Write error message to std::cerr.
  std::cerr << result.GetErrorMessage();
  if (out && out->fail()) {
    // Something wrong happened on output.
    if (out == &std::cout) {
      std::cerr << "glslc: error: error writing to standard output"
                << std::endl;
    } else {
      std::cerr << "glslc: error: error writing to output file: '"
                << output_file_name_ << "'" << std::endl;
    }
    return false;
  }

  return compilation_success;
}

void FileCompiler::AddIncludeDirectory(const std::string& path) {
  include_file_finder_.search_path().push_back(path);
}

void FileCompiler::SetIndividualCompilationFlag() {
  if (output_type_ != OutputType::SpirvAssemblyText) {
    needs_linking_ = false;
    file_extension_ = ".spv";
  }
}

void FileCompiler::SetDisassemblyFlag() {
  if (!PreprocessingOnly()) {
    output_type_ = OutputType::SpirvAssemblyText;
    needs_linking_ = false;
    file_extension_ = ".spvasm";
  }
}

void FileCompiler::SetPreprocessingOnlyFlag() {
  output_type_ = OutputType::PreprocessedText;
  needs_linking_ = false;
  if (output_file_name_.empty()) {
    output_file_name_ = "-";
  }
}

bool FileCompiler::ValidateOptions(size_t num_files) {
  if (num_files == 0) {
    std::cerr << "glslc: error: no input files" << std::endl;
    return false;
  }

  if (num_files > 1 && needs_linking_) {
    std::cerr << "glslc: error: linking multiple files is not supported yet. "
                 "Use -c to compile files individually."
              << std::endl;
    return false;
  }

  // If we are outputting many object files, we cannot specify -o. Also
  // if we are preprocessing multiple files they must be to stdout.
  if (num_files > 1 && ((!PreprocessingOnly() && !needs_linking_ &&
                         !output_file_name_.empty()) ||
                        (PreprocessingOnly() && output_file_name_ != "-"))) {
    std::cerr << "glslc: error: cannot specify -o when generating multiple"
                 " output files"
              << std::endl;
    return false;
  }

  // If we have dependency info dumping handler instantiated, we should check
  // its validity.
  if (dependency_info_dumping_handler_) {
    std::string dependency_info_dumping_hander_error_msg;
    if (!dependency_info_dumping_handler_->IsValid(
            &dependency_info_dumping_hander_error_msg, num_files)) {
      std::cerr << "glslc: error: " << dependency_info_dumping_hander_error_msg
                << std::endl;
      return false;
    }
  }

  // If the output format is specified to be a binary, a list of hex numbers or
  // a C-style initializer list, the output must be in SPIR-V binary code form.
  if (binary_emission_format_ != SpirvBinaryEmissionFormat::Unspecified) {
    if (output_type_ != OutputType::SpirvBinary) {
      std::cerr << "glslc: error: cannot emit output as a ";
      switch (binary_emission_format_) {
        case SpirvBinaryEmissionFormat::Binary:
          std::cerr << "binary";
          break;
        case SpirvBinaryEmissionFormat::Numbers:
          std::cerr << "list of hex numbers";
          break;
        case SpirvBinaryEmissionFormat::CInitList:
          std::cerr << "C-style initializer list";
          break;
        case SpirvBinaryEmissionFormat::Unspecified:
          // The compiler should never be here at runtime. This case is added to
          // complete the switch cases.
          break;
      }
      std::cerr << " when the output is not SPIR-V binary code" << std::endl;
      return false;
    }
    if (dependency_info_dumping_handler_ &&
        dependency_info_dumping_handler_->DumpingAsCompilationOutput()) {
      std::cerr << "glslc: error: cannot dump dependency info when specifying "
                   "any binary output format"
                << std::endl;
    }
  }

  return true;
}

void FileCompiler::OutputMessages() {
  shaderc_util::OutputMessages(&std::cerr, total_warnings_, total_errors_);
}

std::string FileCompiler::GetOutputFileName(std::string input_filename) {
  if (output_file_name_.empty()) {
    return needs_linking_ ? std::string("a.spv")
                          : GetCandidateOutputFileName(input_filename);
  } else {
    return output_file_name_.str();
  }
}

std::string FileCompiler::GetCandidateOutputFileName(
    std::string input_filename) {
  if (!output_file_name_.empty() && !PreprocessingOnly()) {
    return output_file_name_.str();
  }

  std::string extension = file_extension_;
  if (PreprocessingOnly() || needs_linking_) {
    extension = ".spv";
  }

  std::string candidate_output_file_name =
      IsStageFile(input_filename)
          ? shaderc_util::GetBaseFileName(input_filename) + extension
          : shaderc_util::GetBaseFileName(
                input_filename.substr(0, input_filename.find_last_of('.')) +
                extension);
  return candidate_output_file_name;
}
}  // namesapce glslc

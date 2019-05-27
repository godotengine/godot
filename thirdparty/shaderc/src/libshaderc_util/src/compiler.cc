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

#include "libshaderc_util/compiler.h"

#include <cstdint>
#include <tuple>

#include "libshaderc_util/format.h"
#include "libshaderc_util/io.h"
#include "libshaderc_util/message.h"
#include "libshaderc_util/resources.h"
#include "libshaderc_util/shader_stage.h"
#include "libshaderc_util/spirv_tools_wrapper.h"
#include "libshaderc_util/string_piece.h"
#include "libshaderc_util/version_profile.h"

#include "SPIRV/GlslangToSpv.h"

namespace {
using shaderc_util::string_piece;

// For use with glslang parsing calls.
const bool kNotForwardCompatible = false;

// Returns true if #line directive sets the line number for the next line in the
// given version and profile.
inline bool LineDirectiveIsForNextLine(int version, EProfile profile) {
  return profile == EEsProfile || version >= 330;
}

// Returns a #line directive whose arguments are line and filename.
inline std::string GetLineDirective(int line, const string_piece& filename) {
  return "#line " + std::to_string(line) + " \"" + filename.str() + "\"\n";
}

// Given a canonicalized #line directive (starting exactly with "#line", using
// single spaces to separate different components, and having an optional
// newline at the end), returns the line number and string name/number. If no
// string name/number is provided, the second element in the returned pair is an
// empty string_piece. Behavior is undefined if the directive parameter is not a
// canonicalized #line directive.
std::pair<int, string_piece> DecodeLineDirective(string_piece directive) {
  const string_piece kLineDirective = "#line ";
  assert(directive.starts_with(kLineDirective));
  directive = directive.substr(kLineDirective.size());

  const int line = std::atoi(directive.data());
  const size_t space_loc = directive.find_first_of(' ');
  if (space_loc == string_piece::npos) return std::make_pair(line, "");

  directive = directive.substr(space_loc);
  directive = directive.strip("\" \n");
  return std::make_pair(line, directive);
}

// Returns the Glslang message rules for the given target environment,
// source language, and whether we want HLSL offset rules.  We assume
// only valid combinations are used.
EShMessages GetMessageRules(shaderc_util::Compiler::TargetEnv env,
                            shaderc_util::Compiler::SourceLanguage lang,
                            bool hlsl_offsets,
                            bool debug_info) {
  using shaderc_util::Compiler;
  EShMessages result = EShMsgCascadingErrors;
  if (lang == Compiler::SourceLanguage::HLSL) {
    result = static_cast<EShMessages>(result | EShMsgReadHlsl);
  }
  switch (env) {
    case Compiler::TargetEnv::OpenGLCompat:
      break;
    case Compiler::TargetEnv::OpenGL:
      result = static_cast<EShMessages>(result | EShMsgSpvRules);
      break;
    case Compiler::TargetEnv::Vulkan:
      result =
          static_cast<EShMessages>(result | EShMsgSpvRules | EShMsgVulkanRules);
      break;
  }
  if (hlsl_offsets) {
    result = static_cast<EShMessages>(result | EShMsgHlslOffsets);
  }
  if (debug_info) {
    result = static_cast<EShMessages>(result | EShMsgDebugInfo);
  }
  return result;
}

// A GlslangClientInfo captures target client version and desired SPIR-V
// version.
struct GlslangClientInfo {
  bool valid_client = false;
  glslang::EShClient client = glslang::EShClientNone;
  bool valid_client_version = false;
  glslang::EShTargetClientVersion client_version;
  glslang::EShTargetLanguage target_language = glslang::EShTargetSpv;
  glslang::EShTargetLanguageVersion target_language_version =
      glslang::EShTargetSpv_1_0;
};

// Returns the mappings to Glslang client, client version, and SPIR-V version.
// Also indicates whether the input values were valid.
GlslangClientInfo GetGlslangClientInfo(
    shaderc_util::Compiler::TargetEnv env,
    shaderc_util::Compiler::TargetEnvVersion version) {
  GlslangClientInfo result;

  using shaderc_util::Compiler;
  switch (env) {
    case Compiler::TargetEnv::Vulkan:
      result.valid_client = true;
      result.client = glslang::EShClientVulkan;
      if (version == Compiler::TargetEnvVersion::Default ||
          version == Compiler::TargetEnvVersion::Vulkan_1_0) {
        result.client_version = glslang::EShTargetVulkan_1_0;
        result.valid_client_version = true;
      } else if (version == Compiler::TargetEnvVersion::Vulkan_1_1) {
        result.client_version = glslang::EShTargetVulkan_1_1;
        result.valid_client_version = true;
        result.target_language_version = glslang::EShTargetSpv_1_3;
      }
      break;
    case Compiler::TargetEnv::OpenGLCompat:  // TODO(dneto): remove this
    case Compiler::TargetEnv::OpenGL:
      result.valid_client = true;
      result.client = glslang::EShClientOpenGL;
      if (version == Compiler::TargetEnvVersion::Default ||
          version == Compiler::TargetEnvVersion::OpenGL_4_5) {
        result.client_version = glslang::EShTargetOpenGL_450;
        result.valid_client_version = true;
      }
      break;
  }
  return result;
}

}  // anonymous namespace

namespace shaderc_util {

void Compiler::SetLimit(Compiler::Limit limit, int value) {
  switch (limit) {
#define RESOURCE(NAME, FIELD, CNAME) \
  case Limit::NAME:                  \
    limits_.FIELD = value;           \
    break;
#include "libshaderc_util/resources.inc"
#undef RESOURCE
  }
}

int Compiler::GetLimit(Compiler::Limit limit) const {
  switch (limit) {
#define RESOURCE(NAME, FIELD, CNAME) \
  case Limit::NAME:                  \
    return limits_.FIELD;
#include "libshaderc_util/resources.inc"
#undef RESOURCE
  }
  return 0;  // Unreachable
}

std::tuple<bool, std::vector<uint32_t>, size_t> Compiler::Compile(
    const string_piece& input_source_string, EShLanguage forced_shader_stage,
    const std::string& error_tag, const char* entry_point_name,
    const std::function<EShLanguage(std::ostream* error_stream,
                                    const string_piece& error_tag)>&
        stage_callback,
    CountingIncluder& includer, OutputType output_type,
    std::ostream* error_stream, size_t* total_warnings, size_t* total_errors,
    GlslangInitializer* initializer) const {
  // Compilation results to be returned:
  // Initialize the result tuple as a failed compilation. In error cases, we
  // should return result_tuple directly without setting its members.
  auto result_tuple =
      std::make_tuple(false, std::vector<uint32_t>(), (size_t)0u);
  // Get the reference of the members of the result tuple. We should set their
  // values for succeeded compilation before returning the result tuple.
  bool& succeeded = std::get<0>(result_tuple);
  std::vector<uint32_t>& compilation_output_data = std::get<1>(result_tuple);
  size_t& compilation_output_data_size_in_bytes = std::get<2>(result_tuple);

  // Check target environment.
  const auto target_client_info =
      GetGlslangClientInfo(target_env_, target_env_version_);
  if (!target_client_info.valid_client) {
    *error_stream << "error:" << error_tag
                  << ": Invalid target client environment " << int(target_env_);
    *total_warnings = 0;
    *total_errors = 1;
    return result_tuple;
  }
  if (!target_client_info.valid_client_version) {
    *error_stream << "error:" << error_tag << ": Invalid target client version "
                  << static_cast<uint32_t>(target_env_version_)
                  << " for environment " << int(target_env_);
    *total_warnings = 0;
    *total_errors = 1;
    return result_tuple;
  }

  auto token = initializer->Acquire();
  EShLanguage used_shader_stage = forced_shader_stage;
  const std::string macro_definitions =
      shaderc_util::format(predefined_macros_, "#define ", " ", "\n");
  const std::string pound_extension =
      "#extension GL_GOOGLE_include_directive : enable\n";
  const std::string preamble = macro_definitions + pound_extension;

  std::string preprocessed_shader;

  // If only preprocessing, we definitely need to preprocess. Otherwise, if
  // we don't know the stage until now, we need the preprocessed shader to
  // deduce the shader stage.
  if (output_type == OutputType::PreprocessedText ||
      used_shader_stage == EShLangCount) {
    bool success;
    std::string glslang_errors;
    std::tie(success, preprocessed_shader, glslang_errors) =
        PreprocessShader(error_tag, input_source_string, preamble, includer);

    success &= PrintFilteredErrors(error_tag, error_stream, warnings_as_errors_,
                                   /* suppress_warnings = */ true,
                                   glslang_errors.c_str(), total_warnings,
                                   total_errors);
    if (!success) return result_tuple;
    // Because of the behavior change of the #line directive, the #line
    // directive introducing each file's content must use the syntax for the
    // specified version. So we need to probe this shader's version and
    // profile.
    int version;
    EProfile profile;
    std::tie(version, profile) = DeduceVersionProfile(preprocessed_shader);
    const bool is_for_next_line = LineDirectiveIsForNextLine(version, profile);

    preprocessed_shader =
        CleanupPreamble(preprocessed_shader, error_tag, pound_extension,
                        includer.num_include_directives(), is_for_next_line);

    if (output_type == OutputType::PreprocessedText) {
      // Set the values of the result tuple.
      succeeded = true;
      compilation_output_data = ConvertStringToVector(preprocessed_shader);
      compilation_output_data_size_in_bytes = preprocessed_shader.size();
      return result_tuple;
    } else if (used_shader_stage == EShLangCount) {
      std::string errors;
      std::tie(used_shader_stage, errors) =
          GetShaderStageFromSourceCode(error_tag, preprocessed_shader);
      if (!errors.empty()) {
        *error_stream << errors;
        return result_tuple;
      }
      if (used_shader_stage == EShLangCount) {
        if ((used_shader_stage = stage_callback(error_stream, error_tag)) ==
            EShLangCount) {
          return result_tuple;
        }
      }
    }
  }

  // Parsing requires its own Glslang symbol tables.
  glslang::TShader shader(used_shader_stage);
  const char* shader_strings = input_source_string.data();
  const int shader_lengths = static_cast<int>(input_source_string.size());
  const char* string_names = error_tag.c_str();
  shader.setStringsWithLengthsAndNames(&shader_strings, &shader_lengths,
                                       &string_names, 1);
  shader.setPreamble(preamble.c_str());
  shader.setEntryPoint(entry_point_name);
  shader.setAutoMapBindings(auto_bind_uniforms_);
  shader.setAutoMapLocations(auto_map_locations_);
  const auto& bases = auto_binding_base_[static_cast<int>(used_shader_stage)];
  shader.setShiftImageBinding(bases[static_cast<int>(UniformKind::Image)]);
  shader.setShiftSamplerBinding(bases[static_cast<int>(UniformKind::Sampler)]);
  shader.setShiftTextureBinding(bases[static_cast<int>(UniformKind::Texture)]);
  shader.setShiftUboBinding(bases[static_cast<int>(UniformKind::Buffer)]);
  shader.setShiftSsboBinding(
      bases[static_cast<int>(UniformKind::StorageBuffer)]);
  shader.setShiftUavBinding(
      bases[static_cast<int>(UniformKind::UnorderedAccessView)]);
  shader.setHlslIoMapping(hlsl_iomap_);
  shader.setResourceSetBinding(
      hlsl_explicit_bindings_[static_cast<int>(used_shader_stage)]);
  shader.setEnvClient(target_client_info.client,
                      target_client_info.client_version);
  shader.setEnvTarget(target_client_info.target_language,
                      target_client_info.target_language_version);
  if (hlsl_functionality1_enabled_) {
    shader.setEnvTargetHlslFunctionality1();
  }

  const EShMessages rules = GetMessageRules(target_env_, source_language_,
                                            hlsl_offsets_,
                                            generate_debug_info_);

  bool success = shader.parse(
      &limits_, default_version_, default_profile_, force_version_profile_,
      kNotForwardCompatible, rules, includer);

  success &= PrintFilteredErrors(error_tag, error_stream, warnings_as_errors_,
                                 suppress_warnings_, shader.getInfoLog(),
                                 total_warnings, total_errors);
  if (!success) return result_tuple;

  glslang::TProgram program;
  program.addShader(&shader);
  success = program.link(EShMsgDefault) && program.mapIO();
  success &= PrintFilteredErrors(error_tag, error_stream, warnings_as_errors_,
                                 suppress_warnings_, program.getInfoLog(),
                                 total_warnings, total_errors);
  if (!success) return result_tuple;

  // 'spirv' is an alias for the compilation_output_data. This alias is added
  // to serve as an input for the call to DissassemblyBinary.
  std::vector<uint32_t>& spirv = compilation_output_data;
  glslang::SpvOptions options;
  options.generateDebugInfo = generate_debug_info_;
  options.disableOptimizer = true;
  options.optimizeSize = false;
  // Note the call to GlslangToSpv also populates compilation_output_data.
  glslang::GlslangToSpv(*program.getIntermediate(used_shader_stage), spirv,
                        &options);

  // Set the tool field (the top 16-bits) in the generator word to
  // 'Shaderc over Glslang'.
  const uint32_t shaderc_generator_word = 13;  // From SPIR-V XML Registry
  const uint32_t generator_word_index = 2;     // SPIR-V 2.3: Physical layout
  assert(spirv.size() > generator_word_index);
  spirv[generator_word_index] =
      (spirv[generator_word_index] & 0xffff) | (shaderc_generator_word << 16);

  std::vector<PassId> opt_passes;

  if (hlsl_legalization_enabled_ && source_language_ == SourceLanguage::HLSL) {
    // If from HLSL, run this passes to "legalize" the SPIR-V for Vulkan
    // eg. forward and remove memory writes of opaque types.
    opt_passes.push_back(PassId::kLegalizationPasses);
  }

  opt_passes.insert(opt_passes.end(), enabled_opt_passes_.begin(),
                    enabled_opt_passes_.end());

  if (!opt_passes.empty()) {
    std::string opt_errors;
    if (!SpirvToolsOptimize(target_env_, target_env_version_, opt_passes,
                            &spirv, &opt_errors)) {
      *error_stream << "shaderc: internal error: compilation succeeded but "
                       "failed to optimize: "
                    << opt_errors << "\n";
      return result_tuple;
    }
  }

  if (output_type == OutputType::SpirvAssemblyText) {
    std::string text_or_error;
    if (!SpirvToolsDisassemble(target_env_, target_env_version_, spirv,
                               &text_or_error)) {
      *error_stream << "shaderc: internal error: compilation succeeded but "
                       "failed to disassemble: "
                    << text_or_error << "\n";
      return result_tuple;
    }
    succeeded = true;
    compilation_output_data = ConvertStringToVector(text_or_error);
    compilation_output_data_size_in_bytes = text_or_error.size();
    return result_tuple;
  } else {
    succeeded = true;
    // Note compilation_output_data is already populated in GlslangToSpv().
    compilation_output_data_size_in_bytes = spirv.size() * sizeof(spirv[0]);
    return result_tuple;
  }
}

void Compiler::AddMacroDefinition(const char* macro, size_t macro_length,
                                  const char* definition,
                                  size_t definition_length) {
  predefined_macros_[std::string(macro, macro_length)] =
      definition ? std::string(definition, definition_length) : "";
}

void Compiler::SetTargetEnv(Compiler::TargetEnv env,
                            Compiler::TargetEnvVersion version) {
  target_env_ = env;
  target_env_version_ = version;
}

void Compiler::SetSourceLanguage(Compiler::SourceLanguage lang) {
  source_language_ = lang;
}

void Compiler::SetForcedVersionProfile(int version, EProfile profile) {
  default_version_ = version;
  default_profile_ = profile;
  force_version_profile_ = true;
}

void Compiler::SetWarningsAsErrors() { warnings_as_errors_ = true; }

void Compiler::SetGenerateDebugInfo() {
  generate_debug_info_ = true;
  for (size_t i = 0; i < enabled_opt_passes_.size(); ++i) {
    if (enabled_opt_passes_[i] == PassId::kStripDebugInfo) {
      enabled_opt_passes_[i] = PassId::kNullPass;
    }
  }
}

void Compiler::SetOptimizationLevel(Compiler::OptimizationLevel level) {
  // Clear previous settings first.
  enabled_opt_passes_.clear();

  switch (level) {
    case OptimizationLevel::Size:
      if (!generate_debug_info_) {
        enabled_opt_passes_.push_back(PassId::kStripDebugInfo);
      }
      enabled_opt_passes_.push_back(PassId::kSizePasses);
      break;
    case OptimizationLevel::Performance:
      if (!generate_debug_info_) {
        enabled_opt_passes_.push_back(PassId::kStripDebugInfo);
      }
      enabled_opt_passes_.push_back(PassId::kPerformancePasses);
      break;
    default:
      break;
  }
}

void Compiler::EnableHlslLegalization(bool hlsl_legalization_enabled) {
  hlsl_legalization_enabled_ = hlsl_legalization_enabled;
}

void Compiler::EnableHlslFunctionality1(bool enable) {
  hlsl_functionality1_enabled_ = enable;
}

void Compiler::SetSuppressWarnings() { suppress_warnings_ = true; }

std::tuple<bool, std::string, std::string> Compiler::PreprocessShader(
    const std::string& error_tag, const string_piece& shader_source,
    const string_piece& shader_preamble, CountingIncluder& includer) const {
  // The stage does not matter for preprocessing.
  glslang::TShader shader(EShLangVertex);
  const char* shader_strings = shader_source.data();
  const int shader_lengths = static_cast<int>(shader_source.size());
  const char* string_names = error_tag.c_str();
  shader.setStringsWithLengthsAndNames(&shader_strings, &shader_lengths,
                                       &string_names, 1);
  shader.setPreamble(shader_preamble.data());
  auto target_client_info =
      GetGlslangClientInfo(target_env_, target_env_version_);
  if (!target_client_info.valid_client) {
    std::ostringstream os;
    os << "error:" << error_tag << ": Invalid target client "
       << int(target_env_);
    return std::make_tuple(false, "", os.str());
  }
  if (!target_client_info.valid_client_version) {
    std::ostringstream os;
    os << "error:" << error_tag << ": Invalid target client "
       << int(target_env_version_) << " for environmnent " << int(target_env_);
    return std::make_tuple(false, "", os.str());
  }
  shader.setEnvClient(target_client_info.client,
                      target_client_info.client_version);
  if (hlsl_functionality1_enabled_) {
    shader.setEnvTargetHlslFunctionality1();
  }

  // The preprocessor might be sensitive to the target environment.
  // So combine the existing rules with the just-give-me-preprocessor-output
  // flag.
  const auto rules = static_cast<EShMessages>(
      EShMsgOnlyPreprocessor |
      GetMessageRules(target_env_, source_language_, hlsl_offsets_, false));

  std::string preprocessed_shader;
  const bool success = shader.preprocess(
      &limits_, default_version_, default_profile_, force_version_profile_,
      kNotForwardCompatible, rules, &preprocessed_shader, includer);

  if (success) {
    return std::make_tuple(true, preprocessed_shader, shader.getInfoLog());
  }
  return std::make_tuple(false, "", shader.getInfoLog());
}

std::string Compiler::CleanupPreamble(const string_piece& preprocessed_shader,
                                      const string_piece& error_tag,
                                      const string_piece& pound_extension,
                                      int num_include_directives,
                                      bool is_for_next_line) const {
  // Those #define directives in preamble will become empty lines after
  // preprocessing. We also injected an #extension directive to turn on #include
  // directive support. In the original preprocessing output from glslang, it
  // appears before the user source string. We need to do proper adjustment:
  // * Remove empty lines generated from #define directives in preamble.
  // * If there is no #include directive in the source code, we do not need to
  //   output the injected #extension directive. Otherwise,
  // * If there exists a #version directive in the source code, it should be
  //   placed at the first line. Its original line will be filled with an empty
  //   line as placeholder to maintain the code structure.

  const std::vector<string_piece> lines =
      preprocessed_shader.get_fields('\n', /* keep_delimiter = */ true);

  std::ostringstream output_stream;

  size_t pound_extension_index = lines.size();
  size_t pound_version_index = lines.size();
  for (size_t i = 0; i < lines.size(); ++i) {
    if (lines[i] == pound_extension) {
      pound_extension_index = std::min(i, pound_extension_index);
    } else if (lines[i].starts_with("#version")) {
      // In a preprocessed shader, directives are in a canonical format, so we
      // can confidently compare to '#version' verbatim, without worrying about
      // whitespace.
      pound_version_index = i;
      if (num_include_directives > 0) output_stream << lines[i];
      break;
    }
  }
  // We know that #extension directive exists and appears before #version
  // directive (if any).
  assert(pound_extension_index < lines.size());

  for (size_t i = 0; i < pound_extension_index; ++i) {
    // All empty lines before the #line directive we injected are generated by
    // preprocessing preamble. Do not output them.
    if (lines[i].strip_whitespace().empty()) continue;
    output_stream << lines[i];
  }

  if (num_include_directives > 0) {
    output_stream << pound_extension;
    // Also output a #line directive for the main file.
    output_stream << GetLineDirective(is_for_next_line, error_tag);
  }

  for (size_t i = pound_extension_index + 1; i < lines.size(); ++i) {
    if (i == pound_version_index) {
      if (num_include_directives > 0) {
        output_stream << "\n";
      } else {
        output_stream << lines[i];
      }
    } else {
      output_stream << lines[i];
    }
  }

  return output_stream.str();
}

std::pair<EShLanguage, std::string> Compiler::GetShaderStageFromSourceCode(
    string_piece filename, const std::string& preprocessed_shader) const {
  const string_piece kPragmaShaderStageDirective = "#pragma shader_stage";
  const string_piece kLineDirective = "#line";

  int version;
  EProfile profile;
  std::tie(version, profile) = DeduceVersionProfile(preprocessed_shader);
  const bool is_for_next_line = LineDirectiveIsForNextLine(version, profile);

  std::vector<string_piece> lines =
      string_piece(preprocessed_shader).get_fields('\n');
  // The filename, logical line number (which starts from 1 and is sensitive to
  // #line directives), and stage value for #pragma shader_stage() directives.
  std::vector<std::tuple<string_piece, size_t, string_piece>> stages;
  // The physical line numbers of the first #pragma shader_stage() line and
  // first non-preprocessing line in the preprocessed shader text.
  size_t first_pragma_physical_line = lines.size() + 1;
  size_t first_non_pp_line = lines.size() + 1;

  for (size_t i = 0, logical_line_no = 1; i < lines.size(); ++i) {
    const string_piece current_line = lines[i].strip_whitespace();
    if (current_line.starts_with(kPragmaShaderStageDirective)) {
      const string_piece stage_value =
          current_line.substr(kPragmaShaderStageDirective.size()).strip("()");
      stages.emplace_back(filename, logical_line_no, stage_value);
      first_pragma_physical_line = std::min(first_pragma_physical_line, i + 1);
    } else if (!current_line.empty() && !current_line.starts_with("#")) {
      first_non_pp_line = std::min(first_non_pp_line, i + 1);
    }

    // Update logical line number for the next line.
    if (current_line.starts_with(kLineDirective)) {
      string_piece name;
      std::tie(logical_line_no, name) = DecodeLineDirective(current_line);
      if (!name.empty()) filename = name;
      // Note that for core profile, the meaning of #line changed since version
      // 330. The line number given by #line used to mean the logical line
      // number of the #line line. Now it means the logical line number of the
      // next line after the #line line.
      if (!is_for_next_line) ++logical_line_no;
    } else {
      ++logical_line_no;
    }
  }
  if (stages.empty()) return std::make_pair(EShLangCount, "");

  std::string error_message;

  const string_piece& first_pragma_filename = std::get<0>(stages[0]);
  const std::string first_pragma_line = std::to_string(std::get<1>(stages[0]));
  const string_piece& first_pragma_stage = std::get<2>(stages[0]);

  if (first_pragma_physical_line > first_non_pp_line) {
    error_message += first_pragma_filename.str() + ":" + first_pragma_line +
                     ": error: '#pragma': the first 'shader_stage' #pragma "
                     "must appear before any non-preprocessing code\n";
  }

  EShLanguage stage = MapStageNameToLanguage(first_pragma_stage);
  if (stage == EShLangCount) {
    error_message +=
        first_pragma_filename.str() + ":" + first_pragma_line +
        ": error: '#pragma': invalid stage for 'shader_stage' #pragma: '" +
        first_pragma_stage.str() + "'\n";
  }

  for (size_t i = 1; i < stages.size(); ++i) {
    const string_piece& current_stage = std::get<2>(stages[i]);
    if (current_stage != first_pragma_stage) {
      const string_piece& current_filename = std::get<0>(stages[i]);
      const std::string current_line = std::to_string(std::get<1>(stages[i]));
      error_message += current_filename.str() + ":" + current_line +
                       ": error: '#pragma': conflicting stages for "
                       "'shader_stage' #pragma: '" +
                       current_stage.str() + "' (was '" +
                       first_pragma_stage.str() + "' at " +
                       first_pragma_filename.str() + ":" + first_pragma_line +
                       ")\n";
    }
  }

  return std::make_pair(error_message.empty() ? stage : EShLangCount,
                        error_message);
}

std::pair<int, EProfile> Compiler::DeduceVersionProfile(
    const std::string& preprocessed_shader) const {
  int version = default_version_;
  EProfile profile = default_profile_;
  if (!force_version_profile_) {
    std::tie(version, profile) =
        GetVersionProfileFromSourceCode(preprocessed_shader);
    if (version == 0 && profile == ENoProfile) {
      version = default_version_;
      profile = default_profile_;
    }
  }
  return std::make_pair(version, profile);
}

std::pair<int, EProfile> Compiler::GetVersionProfileFromSourceCode(
    const std::string& preprocessed_shader) const {
  string_piece pound_version = preprocessed_shader;
  const size_t pound_version_loc = pound_version.find("#version");
  if (pound_version_loc == string_piece::npos) {
    return std::make_pair(0, ENoProfile);
  }
  pound_version =
      pound_version.substr(pound_version_loc + std::strlen("#version"));
  pound_version = pound_version.substr(0, pound_version.find_first_of("\n"));

  std::string version_profile;
  for (const auto character : pound_version) {
    if (character != ' ') version_profile += character;
  }

  int version;
  EProfile profile;
  if (!ParseVersionProfile(version_profile, &version, &profile)) {
    return std::make_pair(0, ENoProfile);
  }
  return std::make_pair(version, profile);
}

// Converts a string to a vector of uint32_t by copying the content of a given
// string to a vector<uint32_t> and returns it. Appends '\0' at the end if extra
// bytes are required to complete the last element.
std::vector<uint32_t> ConvertStringToVector(const std::string& str) {
  size_t num_bytes_str = str.size() + 1u;
  size_t vector_length =
      (num_bytes_str + sizeof(uint32_t) - 1) / sizeof(uint32_t);
  std::vector<uint32_t> result_vec(vector_length, 0);
  std::strncpy(reinterpret_cast<char*>(result_vec.data()), str.c_str(),
               str.size());
  return result_vec;
}

}  // namespace shaderc_util

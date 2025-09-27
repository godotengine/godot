// Copyright (c) 2016 Google Inc.
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

#ifndef INCLUDE_SPIRV_TOOLS_LIBSPIRV_HPP_
#define INCLUDE_SPIRV_TOOLS_LIBSPIRV_HPP_

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "spirv-tools/libspirv.h"

namespace spvtools {

// Message consumer. The C strings for source and message are only alive for the
// specific invocation.
using MessageConsumer = std::function<void(
    spv_message_level_t /* level */, const char* /* source */,
    const spv_position_t& /* position */, const char* /* message */
    )>;

using HeaderParser = std::function<spv_result_t(
    const spv_endianness_t endianess, const spv_parsed_header_t& instruction)>;
using InstructionParser =
    std::function<spv_result_t(const spv_parsed_instruction_t& instruction)>;

// C++ RAII wrapper around the C context object spv_context.
class Context {
 public:
  // Constructs a context targeting the given environment |env|.
  //
  // See specific API calls for how the target environment is interpreted
  // (particularly assembly and validation).
  //
  // The constructed instance will have an empty message consumer, which just
  // ignores all messages from the library. Use SetMessageConsumer() to supply
  // one if messages are of concern.
  explicit Context(spv_target_env env);

  // Enables move constructor/assignment operations.
  Context(Context&& other);
  Context& operator=(Context&& other);

  // Disables copy constructor/assignment operations.
  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;

  // Destructs this instance.
  ~Context();

  // Sets the message consumer to the given |consumer|. The |consumer| will be
  // invoked once for each message communicated from the library.
  void SetMessageConsumer(MessageConsumer consumer);

  // Returns the underlying spv_context.
  spv_context& CContext();
  const spv_context& CContext() const;

 private:
  spv_context context_;
};

// A RAII wrapper around a validator options object.
class ValidatorOptions {
 public:
  ValidatorOptions() : options_(spvValidatorOptionsCreate()) {}
  ~ValidatorOptions() { spvValidatorOptionsDestroy(options_); }
  // Allow implicit conversion to the underlying object.
  operator spv_validator_options() const { return options_; }

  // Sets a limit.
  void SetUniversalLimit(spv_validator_limit limit_type, uint32_t limit) {
    spvValidatorOptionsSetUniversalLimit(options_, limit_type, limit);
  }

  void SetRelaxStructStore(bool val) {
    spvValidatorOptionsSetRelaxStoreStruct(options_, val);
  }

  // Enables VK_KHR_relaxed_block_layout when validating standard
  // uniform/storage buffer/push-constant layout.  If true, disables
  // scalar block layout rules.
  void SetRelaxBlockLayout(bool val) {
    spvValidatorOptionsSetRelaxBlockLayout(options_, val);
  }

  // Enables VK_KHR_uniform_buffer_standard_layout when validating standard
  // uniform layout.  If true, disables scalar block layout rules.
  void SetUniformBufferStandardLayout(bool val) {
    spvValidatorOptionsSetUniformBufferStandardLayout(options_, val);
  }

  // Enables VK_EXT_scalar_block_layout when validating standard
  // uniform/storage buffer/push-constant layout.  If true, disables
  // relaxed block layout rules.
  void SetScalarBlockLayout(bool val) {
    spvValidatorOptionsSetScalarBlockLayout(options_, val);
  }

  // Enables scalar layout when validating Workgroup blocks.  See
  // VK_KHR_workgroup_memory_explicit_layout.
  void SetWorkgroupScalarBlockLayout(bool val) {
    spvValidatorOptionsSetWorkgroupScalarBlockLayout(options_, val);
  }

  // Skips validating standard uniform/storage buffer/push-constant layout.
  void SetSkipBlockLayout(bool val) {
    spvValidatorOptionsSetSkipBlockLayout(options_, val);
  }

  // Enables LocalSizeId decorations where the environment would not otherwise
  // allow them.
  void SetAllowLocalSizeId(bool val) {
    spvValidatorOptionsSetAllowLocalSizeId(options_, val);
  }

  // Records whether or not the validator should relax the rules on pointer
  // usage in logical addressing mode.
  //
  // When relaxed, it will allow the following usage cases of pointers:
  // 1) OpVariable allocating an object whose type is a pointer type
  // 2) OpReturnValue returning a pointer value
  void SetRelaxLogicalPointer(bool val) {
    spvValidatorOptionsSetRelaxLogicalPointer(options_, val);
  }

  // Records whether or not the validator should relax the rules because it is
  // expected that the optimizations will make the code legal.
  //
  // When relaxed, it will allow the following:
  // 1) It will allow relaxed logical pointers.  Setting this option will also
  //    set that option.
  // 2) Pointers that are pass as parameters to function calls do not have to
  //    match the storage class of the formal parameter.
  // 3) Pointers that are actual parameters on function calls do not have to
  //    point to the same type pointed as the formal parameter.  The types just
  //    need to logically match.
  // 4) GLSLstd450 Interpolate* instructions can have a load of an interpolant
  //    for a first argument.
  void SetBeforeHlslLegalization(bool val) {
    spvValidatorOptionsSetBeforeHlslLegalization(options_, val);
  }

  // Whether friendly names should be used in validation error messages.
  void SetFriendlyNames(bool val) {
    spvValidatorOptionsSetFriendlyNames(options_, val);
  }

 private:
  spv_validator_options options_;
};

// A C++ wrapper around an optimization options object.
class OptimizerOptions {
 public:
  OptimizerOptions() : options_(spvOptimizerOptionsCreate()) {}
  ~OptimizerOptions() { spvOptimizerOptionsDestroy(options_); }

  // Allow implicit conversion to the underlying object.
  operator spv_optimizer_options() const { return options_; }

  // Records whether or not the optimizer should run the validator before
  // optimizing.  If |run| is true, the validator will be run.
  void set_run_validator(bool run) {
    spvOptimizerOptionsSetRunValidator(options_, run);
  }

  // Records the validator options that should be passed to the validator if it
  // is run.
  void set_validator_options(const ValidatorOptions& val_options) {
    spvOptimizerOptionsSetValidatorOptions(options_, val_options);
  }

  // Records the maximum possible value for the id bound.
  void set_max_id_bound(uint32_t new_bound) {
    spvOptimizerOptionsSetMaxIdBound(options_, new_bound);
  }

  // Records whether all bindings within the module should be preserved.
  void set_preserve_bindings(bool preserve_bindings) {
    spvOptimizerOptionsSetPreserveBindings(options_, preserve_bindings);
  }

  // Records whether all specialization constants within the module
  // should be preserved.
  void set_preserve_spec_constants(bool preserve_spec_constants) {
    spvOptimizerOptionsSetPreserveSpecConstants(options_,
                                                preserve_spec_constants);
  }

 private:
  spv_optimizer_options options_;
};

// A C++ wrapper around a reducer options object.
class ReducerOptions {
 public:
  ReducerOptions() : options_(spvReducerOptionsCreate()) {}
  ~ReducerOptions() { spvReducerOptionsDestroy(options_); }

  // Allow implicit conversion to the underlying object.
  operator spv_reducer_options() const {  // NOLINT(google-explicit-constructor)
    return options_;
  }

  // See spvReducerOptionsSetStepLimit.
  void set_step_limit(uint32_t step_limit) {
    spvReducerOptionsSetStepLimit(options_, step_limit);
  }

  // See spvReducerOptionsSetFailOnValidationError.
  void set_fail_on_validation_error(bool fail_on_validation_error) {
    spvReducerOptionsSetFailOnValidationError(options_,
                                              fail_on_validation_error);
  }

  // See spvReducerOptionsSetTargetFunction.
  void set_target_function(uint32_t target_function) {
    spvReducerOptionsSetTargetFunction(options_, target_function);
  }

 private:
  spv_reducer_options options_;
};

// A C++ wrapper around a fuzzer options object.
class FuzzerOptions {
 public:
  FuzzerOptions() : options_(spvFuzzerOptionsCreate()) {}
  ~FuzzerOptions() { spvFuzzerOptionsDestroy(options_); }

  // Allow implicit conversion to the underlying object.
  operator spv_fuzzer_options() const {  // NOLINT(google-explicit-constructor)
    return options_;
  }

  // See spvFuzzerOptionsEnableReplayValidation.
  void enable_replay_validation() {
    spvFuzzerOptionsEnableReplayValidation(options_);
  }

  // See spvFuzzerOptionsSetRandomSeed.
  void set_random_seed(uint32_t seed) {
    spvFuzzerOptionsSetRandomSeed(options_, seed);
  }

  // See spvFuzzerOptionsSetReplayRange.
  void set_replay_range(int32_t replay_range) {
    spvFuzzerOptionsSetReplayRange(options_, replay_range);
  }

  // See spvFuzzerOptionsSetShrinkerStepLimit.
  void set_shrinker_step_limit(uint32_t shrinker_step_limit) {
    spvFuzzerOptionsSetShrinkerStepLimit(options_, shrinker_step_limit);
  }

  // See spvFuzzerOptionsEnableFuzzerPassValidation.
  void enable_fuzzer_pass_validation() {
    spvFuzzerOptionsEnableFuzzerPassValidation(options_);
  }

  // See spvFuzzerOptionsEnableAllPasses.
  void enable_all_passes() { spvFuzzerOptionsEnableAllPasses(options_); }

 private:
  spv_fuzzer_options options_;
};

// C++ interface for SPIRV-Tools functionalities. It wraps the context
// (including target environment and the corresponding SPIR-V grammar) and
// provides methods for assembling, disassembling, and validating.
//
// Instances of this class provide basic thread-safety guarantee.
class SpirvTools {
 public:
  enum {
    // Default assembling option used by assemble():
    kDefaultAssembleOption = SPV_TEXT_TO_BINARY_OPTION_NONE,

    // Default disassembling option used by Disassemble():
    // * Avoid prefix comments from decoding the SPIR-V module header, and
    // * Use friendly names for variables.
    kDefaultDisassembleOption = SPV_BINARY_TO_TEXT_OPTION_NO_HEADER |
                                SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES
  };

  // Constructs an instance targeting the given environment |env|.
  //
  // The constructed instance will have an empty message consumer, which just
  // ignores all messages from the library. Use SetMessageConsumer() to supply
  // one if messages are of concern.
  explicit SpirvTools(spv_target_env env);

  // Disables copy/move constructor/assignment operations.
  SpirvTools(const SpirvTools&) = delete;
  SpirvTools(SpirvTools&&) = delete;
  SpirvTools& operator=(const SpirvTools&) = delete;
  SpirvTools& operator=(SpirvTools&&) = delete;

  // Destructs this instance.
  ~SpirvTools();

  // Sets the message consumer to the given |consumer|. The |consumer| will be
  // invoked once for each message communicated from the library.
  void SetMessageConsumer(MessageConsumer consumer);

  // Assembles the given assembly |text| and writes the result to |binary|.
  // Returns true on successful assembling. |binary| will be kept untouched if
  // assembling is unsuccessful.
  // The SPIR-V binary version is set to the highest version of SPIR-V supported
  // by the target environment with which this SpirvTools object was created.
  bool Assemble(const std::string& text, std::vector<uint32_t>* binary,
                uint32_t options = kDefaultAssembleOption) const;
  // |text_size| specifies the number of bytes in |text|. A terminating null
  // character is not required to present in |text| as long as |text| is valid.
  // The SPIR-V binary version is set to the highest version of SPIR-V supported
  // by the target environment with which this SpirvTools object was created.
  bool Assemble(const char* text, size_t text_size,
                std::vector<uint32_t>* binary,
                uint32_t options = kDefaultAssembleOption) const;

  // Disassembles the given SPIR-V |binary| with the given |options| and writes
  // the assembly to |text|. Returns true on successful disassembling. |text|
  // will be kept untouched if diassembling is unsuccessful.
  bool Disassemble(const std::vector<uint32_t>& binary, std::string* text,
                   uint32_t options = kDefaultDisassembleOption) const;
  // |binary_size| specifies the number of words in |binary|.
  bool Disassemble(const uint32_t* binary, size_t binary_size,
                   std::string* text,
                   uint32_t options = kDefaultDisassembleOption) const;

  // Parses a SPIR-V binary, specified as counted sequence of 32-bit words.
  // Parsing feedback is provided via two callbacks provided as std::function.
  // In a valid parse the parsed-header callback is called once, and
  // then the parsed-instruction callback is called once for each instruction
  // in the stream.
  // Returns true on successful parsing.
  // If diagnostic is non-null, a diagnostic is emitted on failed parsing.
  // If diagnostic is null the context's message consumer
  // will be used to emit any errors. If a callback returns anything other than
  // SPV_SUCCESS, then that status code is returned, no further callbacks are
  // issued, and no additional diagnostics are emitted.
  // This is a wrapper around the C API spvBinaryParse.
  bool Parse(const std::vector<uint32_t>& binary,
             const HeaderParser& header_parser,
             const InstructionParser& instruction_parser,
             spv_diagnostic* diagnostic = nullptr);

  // Validates the given SPIR-V |binary|. Returns true if no issues are found.
  // Otherwise, returns false and communicates issues via the message consumer
  // registered.
  // Validates for SPIR-V spec rules for the SPIR-V version named in the
  // binary's header (at word offset 1).  Additionally, if the target
  // environment is a client API (such as Vulkan 1.1), then validate for that
  // client API version, to the extent that it is verifiable from data in the
  // binary itself.
  bool Validate(const std::vector<uint32_t>& binary) const;
  // Like the previous overload, but provides the binary as a pointer and size:
  // |binary_size| specifies the number of words in |binary|.
  // Validates for SPIR-V spec rules for the SPIR-V version named in the
  // binary's header (at word offset 1).  Additionally, if the target
  // environment is a client API (such as Vulkan 1.1), then validate for that
  // client API version, to the extent that it is verifiable from data in the
  // binary itself.
  bool Validate(const uint32_t* binary, size_t binary_size) const;
  // Like the previous overload, but takes an options object.
  // Validates for SPIR-V spec rules for the SPIR-V version named in the
  // binary's header (at word offset 1).  Additionally, if the target
  // environment is a client API (such as Vulkan 1.1), then validate for that
  // client API version, to the extent that it is verifiable from data in the
  // binary itself, or in the validator options.
  bool Validate(const uint32_t* binary, size_t binary_size,
                spv_validator_options options) const;

  // Was this object successfully constructed.
  bool IsValid() const;

 private:
  struct Impl;  // Opaque struct for holding the data fields used by this class.
  std::unique_ptr<Impl> impl_;  // Unique pointer to implementation data.
};

}  // namespace spvtools

#endif  // INCLUDE_SPIRV_TOOLS_LIBSPIRV_HPP_

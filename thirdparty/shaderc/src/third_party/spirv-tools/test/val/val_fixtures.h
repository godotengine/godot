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

// Common validation fixtures for unit tests

#ifndef TEST_VAL_VAL_FIXTURES_H_
#define TEST_VAL_VAL_FIXTURES_H_

#include <memory>
#include <string>

#include "source/val/validation_state.h"
#include "test/test_fixture.h"
#include "test/unit_spirv.h"

namespace spvtest {

template <typename T>
class ValidateBase : public ::testing::Test,
                     public ::testing::WithParamInterface<T> {
 public:
  ValidateBase();

  virtual void TearDown();

  // Returns the a spv_const_binary struct
  spv_const_binary get_const_binary();

  // Checks that 'code' is valid SPIR-V text representation and stores the
  // binary version for further method calls.
  void CompileSuccessfully(std::string code,
                           spv_target_env env = SPV_ENV_UNIVERSAL_1_0);

  // Overwrites the word at index 'index' with the given word.
  // For testing purposes, it is often useful to be able to manipulate the
  // assembled binary before running the validator on it.
  // This function overwrites the word at the given index with a new word.
  void OverwriteAssembledBinary(uint32_t index, uint32_t word);

  // Performs validation on the SPIR-V code.
  spv_result_t ValidateInstructions(spv_target_env env = SPV_ENV_UNIVERSAL_1_0);

  // Performs validation. Returns the status and stores validation state into
  // the vstate_ member.
  spv_result_t ValidateAndRetrieveValidationState(
      spv_target_env env = SPV_ENV_UNIVERSAL_1_0);

  // Destroys the stored binary.
  void DestroyBinary() {
    spvBinaryDestroy(binary_);
    binary_ = nullptr;
  }

  // Destroys the stored diagnostic.
  void DestroyDiagnostic() {
    spvDiagnosticDestroy(diagnostic_);
    diagnostic_ = nullptr;
  }

  std::string getDiagnosticString();
  spv_position_t getErrorPosition();
  spv_validator_options getValidatorOptions();

  spv_binary binary_;
  spv_diagnostic diagnostic_;
  spv_validator_options options_;
  std::unique_ptr<spvtools::val::ValidationState_t> vstate_;
};

template <typename T>
ValidateBase<T>::ValidateBase() : binary_(nullptr), diagnostic_(nullptr) {
  // Initialize to default command line options. Different tests can then
  // specialize specific options as necessary.
  options_ = spvValidatorOptionsCreate();
}

template <typename T>
spv_const_binary ValidateBase<T>::get_const_binary() {
  return spv_const_binary(binary_);
}

template <typename T>
void ValidateBase<T>::TearDown() {
  if (diagnostic_) {
    spvDiagnosticPrint(diagnostic_);
  }
  DestroyBinary();
  DestroyDiagnostic();
  spvValidatorOptionsDestroy(options_);
}

template <typename T>
void ValidateBase<T>::CompileSuccessfully(std::string code,
                                          spv_target_env env) {
  DestroyBinary();
  spv_diagnostic diagnostic = nullptr;
  ASSERT_EQ(SPV_SUCCESS,
            spvTextToBinary(ScopedContext(env).context, code.c_str(),
                            code.size(), &binary_, &diagnostic))
      << "ERROR: " << diagnostic->error
      << "\nSPIR-V could not be compiled into binary:\n"
      << code;
  spvDiagnosticDestroy(diagnostic);
}

template <typename T>
void ValidateBase<T>::OverwriteAssembledBinary(uint32_t index, uint32_t word) {
  ASSERT_TRUE(index < binary_->wordCount)
      << "OverwriteAssembledBinary: The given index is larger than the binary "
         "word count.";
  binary_->code[index] = word;
}

template <typename T>
spv_result_t ValidateBase<T>::ValidateInstructions(spv_target_env env) {
  DestroyDiagnostic();
  if (binary_ == nullptr) {
    fprintf(stderr,
            "ERROR: Attempting to validate a null binary, did you forget to "
            "call CompileSuccessfully?");
    fflush(stderr);
  }
  assert(binary_ != nullptr);
  return spvValidateWithOptions(ScopedContext(env).context, options_,
                                get_const_binary(), &diagnostic_);
}

template <typename T>
spv_result_t ValidateBase<T>::ValidateAndRetrieveValidationState(
    spv_target_env env) {
  DestroyDiagnostic();
  return spvtools::val::ValidateBinaryAndKeepValidationState(
      ScopedContext(env).context, options_, get_const_binary()->code,
      get_const_binary()->wordCount, &diagnostic_, &vstate_);
}

template <typename T>
std::string ValidateBase<T>::getDiagnosticString() {
  return diagnostic_ == nullptr ? std::string()
                                : std::string(diagnostic_->error);
}

template <typename T>
spv_validator_options ValidateBase<T>::getValidatorOptions() {
  return options_;
}

template <typename T>
spv_position_t ValidateBase<T>::getErrorPosition() {
  return diagnostic_ == nullptr ? spv_position_t() : diagnostic_->position;
}

}  // namespace spvtest

#endif  // TEST_VAL_VAL_FIXTURES_H_

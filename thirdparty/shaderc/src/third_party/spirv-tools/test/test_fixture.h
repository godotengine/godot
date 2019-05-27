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

#ifndef TEST_TEST_FIXTURE_H_
#define TEST_TEST_FIXTURE_H_

#include <string>
#include <vector>

#include "test/unit_spirv.h"

namespace spvtest {

// RAII for spv_context.
struct ScopedContext {
  ScopedContext(spv_target_env env = SPV_ENV_UNIVERSAL_1_0)
      : context(spvContextCreate(env)) {}
  ~ScopedContext() { spvContextDestroy(context); }
  spv_context context;
};

// Common setup for TextToBinary tests. SetText() should be called to populate
// the actual test text.
template <typename T>
class TextToBinaryTestBase : public T {
 public:
  // Shorthand for SPIR-V compilation result.
  using SpirvVector = std::vector<uint32_t>;

  // Offset into a SpirvVector at which the first instruction starts.
  static const SpirvVector::size_type kFirstInstruction = 5;

  TextToBinaryTestBase() : diagnostic(nullptr), text(), binary(nullptr) {
    char textStr[] = "substitute the text member variable with your test";
    text = {textStr, strlen(textStr)};
  }

  virtual ~TextToBinaryTestBase() {
    DestroyBinary();
    if (diagnostic) spvDiagnosticDestroy(diagnostic);
  }

  // Returns subvector v[from:end).
  SpirvVector Subvector(const SpirvVector& v, SpirvVector::size_type from) {
    assert(from <= v.size());
    return SpirvVector(v.begin() + from, v.end());
  }

  // Compiles SPIR-V text in the given assembly syntax format, asserting
  // compilation success. Returns the compiled code.
  SpirvVector CompileSuccessfully(const std::string& txt,
                                  spv_target_env env = SPV_ENV_UNIVERSAL_1_0) {
    DestroyBinary();
    DestroyDiagnostic();
    spv_result_t status =
        spvTextToBinary(ScopedContext(env).context, txt.c_str(), txt.size(),
                        &binary, &diagnostic);
    EXPECT_EQ(SPV_SUCCESS, status) << txt;
    SpirvVector code_copy;
    if (status == SPV_SUCCESS) {
      code_copy = SpirvVector(binary->code, binary->code + binary->wordCount);
      DestroyBinary();
    } else {
      spvDiagnosticPrint(diagnostic);
    }
    return code_copy;
  }

  // Compiles SPIR-V text with the given format, asserting compilation failure.
  // Returns the error message(s).
  std::string CompileFailure(const std::string& txt,
                             spv_target_env env = SPV_ENV_UNIVERSAL_1_0) {
    DestroyBinary();
    DestroyDiagnostic();
    EXPECT_NE(SPV_SUCCESS,
              spvTextToBinary(ScopedContext(env).context, txt.c_str(),
                              txt.size(), &binary, &diagnostic))
        << txt;
    DestroyBinary();
    return diagnostic->error;
  }

  // Encodes SPIR-V text into binary and then decodes the binary using
  // given options. Returns the decoded text.
  std::string EncodeAndDecodeSuccessfully(
      const std::string& txt,
      uint32_t disassemble_options = SPV_BINARY_TO_TEXT_OPTION_NONE,
      spv_target_env env = SPV_ENV_UNIVERSAL_1_0) {
    DestroyBinary();
    DestroyDiagnostic();
    ScopedContext context(env);
    disassemble_options |= SPV_BINARY_TO_TEXT_OPTION_NO_HEADER;
    spv_result_t error = spvTextToBinary(context.context, txt.c_str(),
                                         txt.size(), &binary, &diagnostic);
    if (error) {
      spvDiagnosticPrint(diagnostic);
      spvDiagnosticDestroy(diagnostic);
    }
    EXPECT_EQ(SPV_SUCCESS, error);
    if (!binary) return "";

    spv_text decoded_text;
    error = spvBinaryToText(context.context, binary->code, binary->wordCount,
                            disassemble_options, &decoded_text, &diagnostic);
    if (error) {
      spvDiagnosticPrint(diagnostic);
      spvDiagnosticDestroy(diagnostic);
    }
    EXPECT_EQ(SPV_SUCCESS, error) << txt;

    const std::string decoded_string = decoded_text->str;
    spvTextDestroy(decoded_text);

    return decoded_string;
  }

  // Encodes SPIR-V text into binary. This is expected to succeed.
  // The given words are then appended to the binary, and the result
  // is then decoded. This is expected to fail.
  // Returns the error message.
  std::string EncodeSuccessfullyDecodeFailed(
      const std::string& txt, const SpirvVector& words_to_append) {
    DestroyBinary();
    DestroyDiagnostic();
    SpirvVector code =
        spvtest::Concatenate({CompileSuccessfully(txt), words_to_append});

    spv_text decoded_text;
    EXPECT_NE(SPV_SUCCESS,
              spvBinaryToText(ScopedContext().context, code.data(), code.size(),
                              SPV_BINARY_TO_TEXT_OPTION_NONE, &decoded_text,
                              &diagnostic));
    if (diagnostic) {
      std::string error_message = diagnostic->error;
      spvDiagnosticDestroy(diagnostic);
      diagnostic = nullptr;
      return error_message;
    }
    return "";
  }

  // Compiles SPIR-V text, asserts success, and returns the words representing
  // the instructions.  In particular, skip the words in the SPIR-V header.
  SpirvVector CompiledInstructions(const std::string& txt,
                                   spv_target_env env = SPV_ENV_UNIVERSAL_1_0) {
    const SpirvVector code = CompileSuccessfully(txt, env);
    SpirvVector result;
    // Extract just the instructions.
    // If the code fails to compile, then return the empty vector.
    // In any case, don't crash or invoke undefined behaviour.
    if (code.size() >= kFirstInstruction)
      result = Subvector(code, kFirstInstruction);
    return result;
  }

  void SetText(const std::string& code) {
    textString = code;
    text.str = textString.c_str();
    text.length = textString.size();
  }

  // Destroys the binary, if it exists.
  void DestroyBinary() {
    spvBinaryDestroy(binary);
    binary = nullptr;
  }

  // Destroys the diagnostic, if it exists.
  void DestroyDiagnostic() {
    spvDiagnosticDestroy(diagnostic);
    diagnostic = nullptr;
  }

  spv_diagnostic diagnostic;

  std::string textString;
  spv_text_t text;
  spv_binary binary;
};

using TextToBinaryTest = TextToBinaryTestBase<::testing::Test>;
}  // namespace spvtest

using RoundTripTest =
    spvtest::TextToBinaryTestBase<::testing::TestWithParam<std::string>>;

#endif  // TEST_TEST_FIXTURE_H_

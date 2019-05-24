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

#include <algorithm>
#include <sstream>
#include <utility>

#include "gmock/gmock.h"
#include "test/unit_spirv.h"

namespace spvtools {
namespace {

using ::testing::Eq;

// Returns a newly created diagnostic value.
spv_diagnostic MakeValidDiagnostic() {
  spv_position_t position = {};
  spv_diagnostic diagnostic = spvDiagnosticCreate(&position, "");
  EXPECT_NE(nullptr, diagnostic);
  return diagnostic;
}

TEST(Diagnostic, DestroyNull) { spvDiagnosticDestroy(nullptr); }

TEST(Diagnostic, DestroyValidDiagnostic) {
  spv_diagnostic diagnostic = MakeValidDiagnostic();
  spvDiagnosticDestroy(diagnostic);
  // We aren't allowed to use the diagnostic pointer anymore.
  // So we can't test its behaviour.
}

TEST(Diagnostic, DestroyValidDiagnosticAfterReassignment) {
  spv_diagnostic diagnostic = MakeValidDiagnostic();
  spv_diagnostic second_diagnostic = MakeValidDiagnostic();
  EXPECT_TRUE(diagnostic != second_diagnostic);
  spvDiagnosticDestroy(diagnostic);
  diagnostic = second_diagnostic;
  spvDiagnosticDestroy(diagnostic);
}

TEST(Diagnostic, PrintDefault) {
  char message[] = "Test Diagnostic!";
  spv_diagnostic_t diagnostic = {{2, 3, 5}, message};
  // TODO: Redirect stderr
  ASSERT_EQ(SPV_SUCCESS, spvDiagnosticPrint(&diagnostic));
  // TODO: Validate the output of spvDiagnosticPrint()
  // TODO: Remove the redirection of stderr
}

TEST(Diagnostic, PrintInvalidDiagnostic) {
  ASSERT_EQ(SPV_ERROR_INVALID_DIAGNOSTIC, spvDiagnosticPrint(nullptr));
}

// TODO(dneto): We should be able to redirect the diagnostic printing.
// Once we do that, we can test diagnostic corner cases.

TEST(DiagnosticStream, ConversionToResultType) {
  // Check after the DiagnosticStream object is destroyed.
  spv_result_t value;
  { value = DiagnosticStream({}, nullptr, "", SPV_ERROR_INVALID_TEXT); }
  EXPECT_EQ(SPV_ERROR_INVALID_TEXT, value);

  // Check implicit conversion via plain assignment.
  value = DiagnosticStream({}, nullptr, "", SPV_SUCCESS);
  EXPECT_EQ(SPV_SUCCESS, value);

  // Check conversion via constructor.
  EXPECT_EQ(SPV_FAILED_MATCH,
            spv_result_t(DiagnosticStream({}, nullptr, "", SPV_FAILED_MATCH)));
}

TEST(
    DiagnosticStream,
    MoveConstructorPreservesPreviousMessagesAndPreventsOutputFromExpiringValue) {
  std::ostringstream messages;
  int message_count = 0;
  auto consumer = [&messages, &message_count](spv_message_level_t, const char*,
                                              const spv_position_t&,
                                              const char* msg) {
    message_count++;
    messages << msg;
  };

  // Enclose the DiagnosticStream variables in a scope to force destruction.
  {
    DiagnosticStream ds0({}, consumer, "", SPV_ERROR_INVALID_BINARY);
    ds0 << "First";
    DiagnosticStream ds1(std::move(ds0));
    ds1 << "Second";
  }
  EXPECT_THAT(message_count, Eq(1));
  EXPECT_THAT(messages.str(), Eq("FirstSecond"));
}

TEST(DiagnosticStream, MoveConstructorCanBeDirectlyShiftedTo) {
  std::ostringstream messages;
  int message_count = 0;
  auto consumer = [&messages, &message_count](spv_message_level_t, const char*,
                                              const spv_position_t&,
                                              const char* msg) {
    message_count++;
    messages << msg;
  };

  // Enclose the DiagnosticStream variables in a scope to force destruction.
  {
    DiagnosticStream ds0({}, consumer, "", SPV_ERROR_INVALID_BINARY);
    ds0 << "First";
    std::move(ds0) << "Second";
  }
  EXPECT_THAT(message_count, Eq(1));
  EXPECT_THAT(messages.str(), Eq("FirstSecond"));
}

TEST(DiagnosticStream, DiagnosticFromLambdaReturnCanStillBeUsed) {
  std::ostringstream messages;
  int message_count = 0;
  auto consumer = [&messages, &message_count](spv_message_level_t, const char*,
                                              const spv_position_t&,
                                              const char* msg) {
    message_count++;
    messages << msg;
  };

  {
    auto emitter = [&consumer]() -> DiagnosticStream {
      DiagnosticStream ds0({}, consumer, "", SPV_ERROR_INVALID_BINARY);
      ds0 << "First";
      return ds0;
    };
    emitter() << "Second";
  }
  EXPECT_THAT(message_count, Eq(1));
  EXPECT_THAT(messages.str(), Eq("FirstSecond"));
}

}  // namespace
}  // namespace spvtools

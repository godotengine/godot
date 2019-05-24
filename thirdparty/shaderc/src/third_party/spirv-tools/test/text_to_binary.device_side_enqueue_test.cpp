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

// Assembler tests for instructions in the "Device-Side Enqueue Instructions"
// section of the SPIR-V spec.

#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "test/test_fixture.h"
#include "test/unit_spirv.h"

namespace spvtools {
namespace {

using spvtest::MakeInstruction;
using ::testing::Eq;

// Test OpEnqueueKernel

struct KernelEnqueueCase {
  std::string local_size_source;
  std::vector<uint32_t> local_size_operands;
};

using OpEnqueueKernelGood =
    spvtest::TextToBinaryTestBase<::testing::TestWithParam<KernelEnqueueCase>>;

TEST_P(OpEnqueueKernelGood, Sample) {
  const std::string input =
      "%result = OpEnqueueKernel %type %queue %flags %NDRange %num_events"
      " %wait_events %ret_event %invoke %param %param_size %param_align " +
      GetParam().local_size_source;
  EXPECT_THAT(CompiledInstructions(input),
              Eq(MakeInstruction(SpvOpEnqueueKernel,
                                 {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
                                 GetParam().local_size_operands)));
}

INSTANTIATE_TEST_SUITE_P(
    TextToBinaryTest, OpEnqueueKernelGood,
    ::testing::ValuesIn(std::vector<KernelEnqueueCase>{
        // Provide IDs for pointer-to-local arguments for the
        // invoked function.
        // Test up to 10 such arguments.
        // I (dneto) can't find a limit on the number of kernel
        // arguments in OpenCL C 2.0 Rev 29, e.g. in section 6.9
        // Restrictions.
        {"", {}},
        {"%l0", {13}},
        {"%l0 %l1", {13, 14}},
        {"%l0 %l1 %l2", {13, 14, 15}},
        {"%l0 %l1 %l2 %l3", {13, 14, 15, 16}},
        {"%l0 %l1 %l2 %l3 %l4", {13, 14, 15, 16, 17}},
        {"%l0 %l1 %l2 %l3 %l4 %l5", {13, 14, 15, 16, 17, 18}},
        {"%l0 %l1 %l2 %l3 %l4 %l5 %l6", {13, 14, 15, 16, 17, 18, 19}},
        {"%l0 %l1 %l2 %l3 %l4 %l5 %l6 %l7", {13, 14, 15, 16, 17, 18, 19, 20}},
        {"%l0 %l1 %l2 %l3 %l4 %l5 %l6 %l7 %l8",
         {13, 14, 15, 16, 17, 18, 19, 20, 21}},
        {"%l0 %l1 %l2 %l3 %l4 %l5 %l6 %l7 %l8 %l9",
         {13, 14, 15, 16, 17, 18, 19, 20, 21, 22}},
    }));

// Test some bad parses of OpEnqueueKernel.  For other cases, we're relying
// on the uniformity of the parsing algorithm.  The following two tests, ensure
// that every required ID operand is specified, and is actually an ID operand.
using OpKernelEnqueueBad = spvtest::TextToBinaryTest;

TEST_F(OpKernelEnqueueBad, MissingLastOperand) {
  EXPECT_THAT(
      CompileFailure(
          "%result = OpEnqueueKernel %type %queue %flags %NDRange %num_events"
          " %wait_events %ret_event %invoke %param %param_size"),
      Eq("Expected operand, found end of stream."));
}

TEST_F(OpKernelEnqueueBad, InvalidLastOperand) {
  EXPECT_THAT(
      CompileFailure(
          "%result = OpEnqueueKernel %type %queue %flags %NDRange %num_events"
          " %wait_events %ret_event %invoke %param %param_size 42"),
      Eq("Expected id to start with %."));
}

// TODO(dneto): OpEnqueueMarker
// TODO(dneto): OpGetKernelNDRangeSubGroupCount
// TODO(dneto): OpGetKernelNDRangeMaxSubGroupSize
// TODO(dneto): OpGetKernelWorkGroupSize
// TODO(dneto): OpGetKernelPreferredWorkGroupSizeMultiple
// TODO(dneto): OpRetainEvent
// TODO(dneto): OpReleaseEvent
// TODO(dneto): OpCreateUserEvent
// TODO(dneto): OpSetUserEventStatus
// TODO(dneto): OpCaptureEventProfilingInfo
// TODO(dneto): OpGetDefaultQueue
// TODO(dneto): OpBuildNDRange
// TODO(dneto): OpBuildNDRange

}  // namespace
}  // namespace spvtools

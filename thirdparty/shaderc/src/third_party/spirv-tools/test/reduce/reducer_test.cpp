// Copyright (c) 2018 Google LLC
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

#include "source/reduce/reducer.h"

#include "source/reduce/operand_to_const_reduction_opportunity_finder.h"
#include "source/reduce/remove_opname_instruction_reduction_opportunity_finder.h"
#include "source/reduce/remove_unreferenced_instruction_reduction_opportunity_finder.h"
#include "test/reduce/reduce_test_util.h"

namespace spvtools {
namespace reduce {
namespace {

// This changes its mind each time IsInteresting is invoked as to whether the
// binary is interesting, until some limit is reached after which the binary is
// always deemed interesting.  This is useful to test that reduction passes
// interleave in interesting ways for a while, and then always succeed after
// some point; the latter is important to end up with a predictable final
// reduced binary for tests.
class PingPongInteresting {
 public:
  explicit PingPongInteresting(uint32_t always_interesting_after)
      : is_interesting_(true),
        always_interesting_after_(always_interesting_after),
        count_(0) {}

  bool IsInteresting(const std::vector<uint32_t>&) {
    bool result;
    if (count_ > always_interesting_after_) {
      result = true;
    } else {
      result = is_interesting_;
      is_interesting_ = !is_interesting_;
    }
    count_++;
    return result;
  }

 private:
  bool is_interesting_;
  const uint32_t always_interesting_after_;
  uint32_t count_;
};

TEST(ReducerTest, ExprToConstantAndRemoveUnreferenced) {
  std::string original = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main" %60
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %16 "buf2"
               OpMemberName %16 0 "i"
               OpName %18 ""
               OpName %25 "buf1"
               OpMemberName %25 0 "f"
               OpName %27 ""
               OpName %60 "_GLF_color"
               OpMemberDecorate %16 0 Offset 0
               OpDecorate %16 Block
               OpDecorate %18 DescriptorSet 0
               OpDecorate %18 Binding 2
               OpMemberDecorate %25 0 Offset 0
               OpDecorate %25 Block
               OpDecorate %27 DescriptorSet 0
               OpDecorate %27 Binding 1
               OpDecorate %60 Location 0
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %9 = OpConstant %6 0
         %16 = OpTypeStruct %6
         %17 = OpTypePointer Uniform %16
         %18 = OpVariable %17 Uniform
         %19 = OpTypePointer Uniform %6
         %22 = OpTypeBool
        %100 = OpConstantTrue %22
         %24 = OpTypeFloat 32
         %25 = OpTypeStruct %24
         %26 = OpTypePointer Uniform %25
         %27 = OpVariable %26 Uniform
         %28 = OpTypePointer Uniform %24
         %31 = OpConstant %24 2
         %56 = OpConstant %6 1
         %58 = OpTypeVector %24 4
         %59 = OpTypePointer Output %58
         %60 = OpVariable %59 Output
         %72 = OpUndef %24
         %74 = OpUndef %6
          %4 = OpFunction %2 None %3
          %5 = OpLabel
               OpBranch %10
         %10 = OpLabel
         %73 = OpPhi %6 %74 %5 %77 %34
         %71 = OpPhi %24 %72 %5 %76 %34
         %70 = OpPhi %6 %9 %5 %57 %34
         %20 = OpAccessChain %19 %18 %9
         %21 = OpLoad %6 %20
         %23 = OpSLessThan %22 %70 %21
               OpLoopMerge %12 %34 None
               OpBranchConditional %23 %11 %12
         %11 = OpLabel
         %29 = OpAccessChain %28 %27 %9
         %30 = OpLoad %24 %29
         %32 = OpFOrdGreaterThan %22 %30 %31
               OpSelectionMerge %34 None
               OpBranchConditional %32 %33 %46
         %33 = OpLabel
         %40 = OpFAdd %24 %71 %30
         %45 = OpISub %6 %73 %21
               OpBranch %34
         %46 = OpLabel
         %50 = OpFMul %24 %71 %30
         %54 = OpSDiv %6 %73 %21
               OpBranch %34
         %34 = OpLabel
         %77 = OpPhi %6 %45 %33 %54 %46
         %76 = OpPhi %24 %40 %33 %50 %46
         %57 = OpIAdd %6 %70 %56
               OpBranch %10
         %12 = OpLabel
         %61 = OpAccessChain %28 %27 %9
         %62 = OpLoad %24 %61
         %66 = OpConvertSToF %24 %21
         %68 = OpConvertSToF %24 %73
         %69 = OpCompositeConstruct %58 %62 %71 %66 %68
               OpStore %60 %69
               OpReturn
               OpFunctionEnd
  )";

  std::string expected = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %4 "main" %60
               OpExecutionMode %4 OriginUpperLeft
               OpSource ESSL 310
               OpName %4 "main"
               OpName %16 "buf2"
               OpMemberName %16 0 "i"
               OpName %18 ""
               OpName %25 "buf1"
               OpMemberName %25 0 "f"
               OpName %27 ""
               OpName %60 "_GLF_color"
               OpMemberDecorate %16 0 Offset 0
               OpDecorate %16 Block
               OpDecorate %18 DescriptorSet 0
               OpDecorate %18 Binding 2
               OpMemberDecorate %25 0 Offset 0
               OpDecorate %25 Block
               OpDecorate %27 DescriptorSet 0
               OpDecorate %27 Binding 1
               OpDecorate %60 Location 0
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 1
          %9 = OpConstant %6 0
         %16 = OpTypeStruct %6
         %17 = OpTypePointer Uniform %16
         %18 = OpVariable %17 Uniform
         %19 = OpTypePointer Uniform %6
         %22 = OpTypeBool
        %100 = OpConstantTrue %22
         %24 = OpTypeFloat 32
         %25 = OpTypeStruct %24
         %26 = OpTypePointer Uniform %25
         %27 = OpVariable %26 Uniform
         %28 = OpTypePointer Uniform %24
         %31 = OpConstant %24 2
         %56 = OpConstant %6 1
         %58 = OpTypeVector %24 4
         %59 = OpTypePointer Output %58
         %60 = OpVariable %59 Output
         %72 = OpUndef %24
         %74 = OpUndef %6
          %4 = OpFunction %2 None %3
          %5 = OpLabel
               OpBranch %10
         %10 = OpLabel
               OpLoopMerge %12 %34 None
               OpBranchConditional %100 %11 %12
         %11 = OpLabel
               OpSelectionMerge %34 None
               OpBranchConditional %100 %33 %46
         %33 = OpLabel
               OpBranch %34
         %46 = OpLabel
               OpBranch %34
         %34 = OpLabel
               OpBranch %10
         %12 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  spv_target_env env = SPV_ENV_UNIVERSAL_1_3;
  Reducer reducer(env);
  PingPongInteresting ping_pong_interesting(10);
  reducer.SetMessageConsumer(NopDiagnostic);
  reducer.SetInterestingnessFunction(
      [&](const std::vector<uint32_t>& binary, uint32_t) -> bool {
        return ping_pong_interesting.IsInteresting(binary);
      });
  reducer.AddReductionPass(
      MakeUnique<OperandToConstReductionOpportunityFinder>());
  reducer.AddReductionPass(
      MakeUnique<RemoveUnreferencedInstructionReductionOpportunityFinder>());

  std::vector<uint32_t> binary_in;
  SpirvTools t(env);

  ASSERT_TRUE(t.Assemble(original, &binary_in, kReduceAssembleOption));
  std::vector<uint32_t> binary_out;
  spvtools::ReducerOptions reducer_options;
  reducer_options.set_step_limit(500);
  reducer_options.set_fail_on_validation_error(true);
  spvtools::ValidatorOptions validator_options;

  Reducer::ReductionResultStatus status = reducer.Run(
      std::move(binary_in), &binary_out, reducer_options, validator_options);

  ASSERT_EQ(status, Reducer::ReductionResultStatus::kComplete);

  CheckEqual(env, expected, binary_out);
}

TEST(ReducerTest, RemoveOpnameAndRemoveUnreferenced) {
  const std::string original = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource ESSL 310
               OpName %2 "main"
               OpName %3 "a"
               OpName %4 "this-name-counts-as-usage-for-load-instruction"
          %5 = OpTypeVoid
          %6 = OpTypeFunction %5
          %7 = OpTypeFloat 32
          %8 = OpTypePointer Function %7
          %9 = OpConstant %7 1
          %2 = OpFunction %5 None %6
         %10 = OpLabel
          %3 = OpVariable %8 Function
          %4 = OpLoad %7 %3
               OpStore %3 %9
               OpReturn
               OpFunctionEnd
  )";

  const std::string expected = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %2 "main"
               OpExecutionMode %2 OriginUpperLeft
               OpSource ESSL 310
          %5 = OpTypeVoid
          %6 = OpTypeFunction %5
          %7 = OpTypeFloat 32
          %8 = OpTypePointer Function %7
          %9 = OpConstant %7 1
          %2 = OpFunction %5 None %6
         %10 = OpLabel
               OpReturn
               OpFunctionEnd
  )";

  spv_target_env env = SPV_ENV_UNIVERSAL_1_3;
  Reducer reducer(env);
  // Make ping-pong interesting very quickly, as there are not many
  // opportunities.
  PingPongInteresting ping_pong_interesting(1);
  reducer.SetMessageConsumer(NopDiagnostic);
  reducer.SetInterestingnessFunction(
      [&](const std::vector<uint32_t>& binary, uint32_t) -> bool {
        return ping_pong_interesting.IsInteresting(binary);
      });
  reducer.AddReductionPass(
      MakeUnique<RemoveOpNameInstructionReductionOpportunityFinder>());
  reducer.AddReductionPass(
      MakeUnique<RemoveUnreferencedInstructionReductionOpportunityFinder>());

  std::vector<uint32_t> binary_in;
  SpirvTools t(env);

  ASSERT_TRUE(t.Assemble(original, &binary_in, kReduceAssembleOption));
  std::vector<uint32_t> binary_out;
  spvtools::ReducerOptions reducer_options;
  reducer_options.set_step_limit(500);
  reducer_options.set_fail_on_validation_error(true);
  spvtools::ValidatorOptions validator_options;

  Reducer::ReductionResultStatus status = reducer.Run(
      std::move(binary_in), &binary_out, reducer_options, validator_options);

  ASSERT_EQ(status, Reducer::ReductionResultStatus::kComplete);

  CheckEqual(env, expected, binary_out);
}

}  // namespace
}  // namespace reduce
}  // namespace spvtools

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

#include "source/reduce/reduction_opportunity.h"
#include "source/reduce/remove_instruction_reduction_opportunity.h"
#include "test/reduce/reduce_test_util.h"

namespace spvtools {
namespace reduce {
namespace {

using opt::Function;
using opt::IRContext;
using opt::Instruction;

// A dumb reduction opportunity finder that finds opportunities to remove global
// values regardless of whether they are referenced. This is very likely to make
// the resulting module invalid.  We use this to test the reducer's behavior in
// the scenario where a bad reduction pass leads to an invalid module.
class BlindlyRemoveGlobalValuesReductionOpportunityFinder
    : public ReductionOpportunityFinder {
 public:
  BlindlyRemoveGlobalValuesReductionOpportunityFinder() = default;

  ~BlindlyRemoveGlobalValuesReductionOpportunityFinder() override = default;

  // The name of this pass.
  std::string GetName() const final { return "BlindlyRemoveGlobalValuesPass"; };

  // Finds opportunities to remove all global values.  Assuming they are all
  // referenced (directly or indirectly) from elsewhere in the module, each such
  // opportunity will make the module invalid.
  std::vector<std::unique_ptr<ReductionOpportunity>> GetAvailableOpportunities(
      IRContext* context) const final {
    std::vector<std::unique_ptr<ReductionOpportunity>> result;
    for (auto& inst : context->module()->types_values()) {
      if (inst.HasResultId()) {
        result.push_back(
            MakeUnique<RemoveInstructionReductionOpportunity>(&inst));
      }
    }
    return result;
  }
};

// A dumb reduction opportunity that exists at the start of every function whose
// first instruction is an OpVariable instruction. When applied, the OpVariable
// instruction is duplicated (with a fresh result id). This allows each
// reduction step to increase the number of variables to check if the validator
// limits are enforced.
class OpVariableDuplicatorReductionOpportunity : public ReductionOpportunity {
 public:
  OpVariableDuplicatorReductionOpportunity(Function* function)
      : function_(function) {}

  bool PreconditionHolds() override {
    Instruction* first_instruction = &*function_->begin()[0].begin();
    return first_instruction->opcode() == SpvOpVariable;
  }

 protected:
  void Apply() override {
    // Duplicate the first OpVariable instruction.

    Instruction* first_instruction = &*function_->begin()[0].begin();
    assert(first_instruction->opcode() == SpvOpVariable &&
           "Expected first instruction to be OpVariable");
    IRContext* context = first_instruction->context();
    Instruction* cloned_instruction = first_instruction->Clone(context);
    cloned_instruction->SetResultId(context->TakeNextId());
    cloned_instruction->InsertBefore(first_instruction);
  }

 private:
  Function* function_;
};

// A reduction opportunity finder that finds
// OpVariableDuplicatorReductionOpportunity.
class OpVariableDuplicatorReductionOpportunityFinder
    : public ReductionOpportunityFinder {
 public:
  OpVariableDuplicatorReductionOpportunityFinder() = default;

  ~OpVariableDuplicatorReductionOpportunityFinder() override = default;

  std::string GetName() const final {
    return "LocalVariableAdderReductionOpportunityFinder";
  };

  std::vector<std::unique_ptr<ReductionOpportunity>> GetAvailableOpportunities(
      IRContext* context) const final {
    std::vector<std::unique_ptr<ReductionOpportunity>> result;
    for (auto& function : *context->module()) {
      Instruction* first_instruction = &*function.begin()[0].begin();
      if (first_instruction->opcode() == SpvOpVariable) {
        result.push_back(
            MakeUnique<OpVariableDuplicatorReductionOpportunity>(&function));
      }
    }
    return result;
  }
};

TEST(ValidationDuringReductionTest, CheckInvalidPassMakesNoProgress) {
  // A module whose global values are all referenced, so that any application of
  // MakeModuleInvalidPass will make the module invalid. Check that the reducer
  // makes no progress, as every step will be invalid and treated as
  // uninteresting.
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

  spv_target_env env = SPV_ENV_UNIVERSAL_1_3;
  Reducer reducer(env);
  reducer.SetMessageConsumer(NopDiagnostic);

  // Say that every module is interesting.
  reducer.SetInterestingnessFunction(
      [](const std::vector<uint32_t>&, uint32_t) -> bool { return true; });

  reducer.AddReductionPass(
      MakeUnique<BlindlyRemoveGlobalValuesReductionOpportunityFinder>());

  std::vector<uint32_t> binary_in;
  SpirvTools t(env);

  ASSERT_TRUE(t.Assemble(original, &binary_in, kReduceAssembleOption));
  std::vector<uint32_t> binary_out;
  spvtools::ReducerOptions reducer_options;
  reducer_options.set_step_limit(500);
  // Don't fail on a validation error; just treat it as uninteresting.
  reducer_options.set_fail_on_validation_error(false);
  spvtools::ValidatorOptions validator_options;

  Reducer::ReductionResultStatus status = reducer.Run(
      std::move(binary_in), &binary_out, reducer_options, validator_options);

  ASSERT_EQ(status, Reducer::ReductionResultStatus::kComplete);

  // The reducer should have no impact.
  CheckEqual(env, original, binary_out);
}

TEST(ValidationDuringReductionTest, CheckNotAlwaysInvalidCanMakeProgress) {
  // A module with just one unreferenced global value.  All but one application
  // of MakeModuleInvalidPass will make the module invalid.
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
         %24 = OpTypeFloat 32
         %25 = OpTypeStruct %24
         %26 = OpTypePointer Uniform %25
         %27 = OpVariable %26 Uniform
         %28 = OpTypePointer Uniform %24
         %31 = OpConstant %24 2
         %56 = OpConstant %6 1
       %1000 = OpConstant %6 1000 ; It should be possible to remove this instruction without making the module invalid.
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

  // This is the same as the original, except that the constant declaration of
  // 1000 is gone.
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

  spv_target_env env = SPV_ENV_UNIVERSAL_1_3;
  Reducer reducer(env);
  reducer.SetMessageConsumer(NopDiagnostic);

  // Say that every module is interesting.
  reducer.SetInterestingnessFunction(
      [](const std::vector<uint32_t>&, uint32_t) -> bool { return true; });

  reducer.AddReductionPass(
      MakeUnique<BlindlyRemoveGlobalValuesReductionOpportunityFinder>());

  std::vector<uint32_t> binary_in;
  SpirvTools t(env);

  ASSERT_TRUE(t.Assemble(original, &binary_in, kReduceAssembleOption));
  std::vector<uint32_t> binary_out;
  spvtools::ReducerOptions reducer_options;
  reducer_options.set_step_limit(500);
  // Don't fail on a validation error; just treat it as uninteresting.
  reducer_options.set_fail_on_validation_error(false);
  spvtools::ValidatorOptions validator_options;

  Reducer::ReductionResultStatus status = reducer.Run(
      std::move(binary_in), &binary_out, reducer_options, validator_options);

  ASSERT_EQ(status, Reducer::ReductionResultStatus::kComplete);

  CheckEqual(env, expected, binary_out);
}

// Sets up a Reducer for use in the CheckValidationOptions test; avoids
// repetition.
void SetupReducerForCheckValidationOptions(Reducer* reducer) {
  reducer->SetMessageConsumer(NopDiagnostic);

  // Say that every module is interesting.
  reducer->SetInterestingnessFunction(
      [](const std::vector<uint32_t>&, uint32_t) -> bool { return true; });

  // Each "reduction" step will duplicate the first OpVariable instruction in
  // the function.
  reducer->AddReductionPass(
      MakeUnique<OpVariableDuplicatorReductionOpportunityFinder>());
}

TEST(ValidationDuringReductionTest, CheckValidationOptions) {
  // A module that only validates when the "skip-block-layout" validator option
  // is used. Also, the entry point's first instruction creates a local
  // variable; this instruction will be duplicated on each reduction step.
  std::string original = R"(
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %2 "Main" %3
               OpSource HLSL 600
               OpDecorate %3 BuiltIn Position
               OpDecorate %4 DescriptorSet 0
               OpDecorate %4 Binding 99
               OpDecorate %5 ArrayStride 16
               OpMemberDecorate %6 0 Offset 0
               OpMemberDecorate %6 1 Offset 32
               OpMemberDecorate %6 1 MatrixStride 16
               OpMemberDecorate %6 1 ColMajor
               OpMemberDecorate %6 2 Offset 96
               OpMemberDecorate %6 3 Offset 100
               OpMemberDecorate %6 4 Offset 112
               OpMemberDecorate %6 4 MatrixStride 16
               OpMemberDecorate %6 4 ColMajor
               OpMemberDecorate %6 5 Offset 176
               OpDecorate %6 Block
          %7 = OpTypeFloat 32
          %8 = OpTypeVector %7 4
          %9 = OpTypeMatrix %8 4
         %10 = OpTypeVector %7 2
         %11 = OpTypeInt 32 1
         %12 = OpTypeInt 32 0
         %13 = OpConstant %12 2
         %14 = OpConstant %11 1
         %15 = OpConstant %11 5
          %5 = OpTypeArray %8 %13
          %6 = OpTypeStruct %5 %9 %12 %10 %9 %7
         %16 = OpTypePointer Uniform %6
         %17 = OpTypePointer Output %8
         %18 = OpTypeVoid
         %19 = OpTypeFunction %18
         %20 = OpTypePointer Uniform %7
          %4 = OpVariable %16 Uniform
          %3 = OpVariable %17 Output
         %21 = OpTypePointer Function %11
          %2 = OpFunction %18 None %19
         %22 = OpLabel
         %23 = OpVariable %21 Function
         %24 = OpAccessChain %20 %4 %15
         %25 = OpLoad %7 %24
         %26 = OpCompositeConstruct %8 %25 %25 %25 %25
               OpStore %3 %26
               OpReturn
               OpFunctionEnd
  )";

  spv_target_env env = SPV_ENV_UNIVERSAL_1_3;
  std::vector<uint32_t> binary_in;
  SpirvTools t(env);

  ASSERT_TRUE(t.Assemble(original, &binary_in, kReduceAssembleOption));
  std::vector<uint32_t> binary_out;
  spvtools::ReducerOptions reducer_options;
  spvtools::ValidatorOptions validator_options;

  reducer_options.set_step_limit(3);
  reducer_options.set_fail_on_validation_error(true);

  // Reduction should fail because the initial state is invalid without the
  // "skip-block-layout" validator option. Note that the interestingness test
  // always returns true.
  {
    Reducer reducer(env);
    SetupReducerForCheckValidationOptions(&reducer);

    Reducer::ReductionResultStatus status =
        reducer.Run(std::vector<uint32_t>(binary_in), &binary_out,
                    reducer_options, validator_options);

    ASSERT_EQ(status, Reducer::ReductionResultStatus::kInitialStateInvalid);
  }

  // Try again with validator option.
  validator_options.SetSkipBlockLayout(true);

  // Reduction should hit step limit; module is seen as valid, interestingness
  // test always succeeds, and the finder yields infinite opportunities.
  {
    Reducer reducer(env);
    SetupReducerForCheckValidationOptions(&reducer);

    Reducer::ReductionResultStatus status =
        reducer.Run(std::vector<uint32_t>(binary_in), &binary_out,
                    reducer_options, validator_options);

    ASSERT_EQ(status, Reducer::ReductionResultStatus::kReachedStepLimit);
  }

  // Now set a limit on the number of local variables.
  validator_options.SetUniversalLimit(spv_validator_limit_max_local_variables,
                                      2);

  // Reduction should now fail due to reaching an invalid state; after one step,
  // a local variable is added and the module becomes "invalid" given the
  // validator limits.
  {
    Reducer reducer(env);
    SetupReducerForCheckValidationOptions(&reducer);

    Reducer::ReductionResultStatus status =
        reducer.Run(std::vector<uint32_t>(binary_in), &binary_out,
                    reducer_options, validator_options);

    ASSERT_EQ(status, Reducer::ReductionResultStatus::kStateInvalid);
  }
}

}  // namespace
}  // namespace reduce
}  // namespace spvtools

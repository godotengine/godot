// Copyright (c) 2020 André Perez Maselco
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

#ifndef SOURCE_FUZZ_TRANSFORMATION_EXPAND_VECTOR_REDUCTION_H_
#define SOURCE_FUZZ_TRANSFORMATION_EXPAND_VECTOR_REDUCTION_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

// clang-format off
// SPIR-V code to help understand the transformation.
//
// -------------------------------------------------------------------------------
// |           Reference shader           |            Variant shader            |
// -------------------------------------------------------------------------------
// |       OpCapability Shader            |       OpCapability Shader            |
// |  %1 = OpExtInstImport "GLSL.std.450" |  %1 = OpExtInstImport "GLSL.std.450" |
// |       OpMemoryModel Logical GLSL450  |       OpMemoryModel Logical GLSL450  |
// |       OpEntryPoint Vertex %9 "main"  |       OpEntryPoint Vertex %9 "main"  |
// |                                      |                                      |
// | ; Types                              | ; Types                              |
// |  %2 = OpTypeBool                     |  %2 = OpTypeBool                     |
// |  %3 = OpTypeVector %2 2              |  %3 = OpTypeVector %2 2              |
// |  %4 = OpTypeVoid                     |  %4 = OpTypeVoid                     |
// |  %5 = OpTypeFunction %4              |  %5 = OpTypeFunction %4              |
// |                                      |                                      |
// | ; Constants                          | ; Constants                          |
// |  %6 = OpConstantTrue %2              |  %6 = OpConstantTrue %2              |
// |  %7 = OpConstantFalse %2             |  %7 = OpConstantFalse %2             |
// |  %8 = OpConstantComposite %3 %6 %7   |  %8 = OpConstantComposite %3 %6 %7   |
// |                                      |                                      |
// | ; main function                      | ; main function                      |
// |  %9 = OpFunction %4 None %5          |  %9 = OpFunction %4 None %5          |
// | %10 = OpLabel                        | %10 = OpLabel                        |
// | %11 = OpAny %2 %8                    |                                      |
// | %12 = OpAll %2 %8                    | ; Add OpAny synonym                  |
// |       OpReturn                       | %13 = OpCompositeExtract %2 %8 0     |
// |       OpFunctionEnd                  | %14 = OpCompositeExtract %2 %8 1     |
// |                                      | %15 = OpLogicalOr %2 %13 %14         |
// |                                      | %11 = OpAny %2 %8                    |
// |                                      |                                      |
// |                                      | ; Add OpAll synonym                  |
// |                                      | %16 = OpCompositeExtract %2 %8 0     |
// |                                      | %17 = OpCompositeExtract %2 %8 1     |
// |                                      | %18 = OpLogicalAnd %2 %16 %17        |
// |                                      | %12 = OpAll %2 %8                    |
// |                                      |       OpReturn                       |
// |                                      |       OpFunctionEnd                  |
// -------------------------------------------------------------------------------
//
// %11 and %15 are synonymous
// %12 and %18 are synonymous
// clang-format on
class TransformationExpandVectorReduction : public Transformation {
 public:
  explicit TransformationExpandVectorReduction(
      protobufs::TransformationExpandVectorReduction message);

  TransformationExpandVectorReduction(const uint32_t instruction_result_id,
                                      const std::vector<uint32_t>& fresh_ids);

  // - |message_.instruction_result_id| must be OpAny or OpAll.
  // - |message_.fresh_ids| must be fresh ids needed to apply the
  //   transformation.
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // Adds synonyms for OpAny and OpAll instructions by evaluating each vector
  // component with the corresponding logical operation.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  std::unordered_set<uint32_t> GetFreshIds() const override;

  protobufs::Transformation ToMessage() const override;

  // Returns the number of fresh ids required to apply the transformation.
  static uint32_t GetRequiredFreshIdCount(opt::IRContext* ir_context,
                                          opt::Instruction* instruction);

 private:
  protobufs::TransformationExpandVectorReduction message_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_EXPAND_VECTOR_REDUCTION_H_

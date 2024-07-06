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

#ifndef SOURCE_FUZZ_TRANSFORMATION_ADD_BIT_INSTRUCTION_SYNONYM_H_
#define SOURCE_FUZZ_TRANSFORMATION_ADD_BIT_INSTRUCTION_SYNONYM_H_

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

// clang-format off
// SPIR-V code to help understand the transformation.
//
// ----------------------------------------------------------------------------------------------------------------
// |               Reference shader              |                         Variant shader                         |
// ----------------------------------------------------------------------------------------------------------------
// |      OpCapability Shader                    |       OpCapability Shader                                      |
// |      OpCapability Int8                      |       OpCapability Int8                                        |
// | %1 = OpExtInstImport "GLSL.std.450"         |  %1 = OpExtInstImport "GLSL.std.450"                           |
// |      OpMemoryModel Logical GLSL450          |       OpMemoryModel Logical GLSL450                            |
// |      OpEntryPoint Vertex %7 "main"          |       OpEntryPoint Vertex %7 "main"                            |
// |                                             |                                                                |
// | ; Types                                     | ; Types                                                        |
// | %2 = OpTypeInt 8 0                          |  %2 = OpTypeInt 8 0                                            |
// | %3 = OpTypeVoid                             |  %3 = OpTypeVoid                                               |
// | %4 = OpTypeFunction %3                      |  %4 = OpTypeFunction %3                                        |
// |                                             |                                                                |
// | ; Constants                                 | ; Constants                                                    |
// | %5 = OpConstant %2 0                        |  %5 = OpConstant %2 0                                          |
// | %6 = OpConstant %2 1                        |  %6 = OpConstant %2 1                                          |
// |                                             | %10 = OpConstant %2 2                                          |
// | ; main function                             | %11 = OpConstant %2 3                                          |
// | %7 = OpFunction %3 None %4                  | %12 = OpConstant %2 4                                          |
// | %8 = OpLabel                                | %13 = OpConstant %2 5                                          |
// | %9 = OpBitwiseOr %2 %5 %6 ; bit instruction | %14 = OpConstant %2 6                                          |
// |       OpReturn                              | %15 = OpConstant %2 7                                          |
// |       OpFunctionEnd                         |                                                                |
// |                                             | ; main function                                                |
// |                                             |  %7 = OpFunction %3 None %4                                    |
// |                                             |  %8 = OpLabel                                                  |
// |                                             |                                                                |
// |                                             | %16 = OpBitFieldUExtract %2 %5 %5 %6 ; extracts bit 0 from %5  |
// |                                             | %17 = OpBitFieldUExtract %2 %6 %5 %6 ; extracts bit 0 from %6  |
// |                                             | %18 = OpBitwiseOr %2 %16 %17                                   |
// |                                             |                                                                |
// |                                             | %19 = OpBitFieldUExtract %2 %5 %6 %6 ; extracts bit 1 from %5  |
// |                                             | %20 = OpBitFieldUExtract %2 %6 %6 %6 ; extracts bit 1 from %6  |
// |                                             | %21 = OpBitwiseOr %2 %19 %20                                   |
// |                                             |                                                                |
// |                                             | %22 = OpBitFieldUExtract %2 %5 %10 %6 ; extracts bit 2 from %5 |
// |                                             | %23 = OpBitFieldUExtract %2 %6 %10 %6 ; extracts bit 2 from %6 |
// |                                             | %24 = OpBitwiseOr %2 %22 %23                                   |
// |                                             |                                                                |
// |                                             | %25 = OpBitFieldUExtract %2 %5 %11 %6 ; extracts bit 3 from %5 |
// |                                             | %26 = OpBitFieldUExtract %2 %6 %11 %6 ; extracts bit 3 from %6 |
// |                                             | %27 = OpBitwiseOr %2 %25 %26                                   |
// |                                             |                                                                |
// |                                             | %28 = OpBitFieldUExtract %2 %5 %12 %6 ; extracts bit 4 from %5 |
// |                                             | %29 = OpBitFieldUExtract %2 %6 %12 %6 ; extracts bit 4 from %6 |
// |                                             | %30 = OpBitwiseOr %2 %28 %29                                   |
// |                                             |                                                                |
// |                                             | %31 = OpBitFieldUExtract %2 %5 %13 %6 ; extracts bit 5 from %5 |
// |                                             | %32 = OpBitFieldUExtract %2 %6 %13 %6 ; extracts bit 5 from %6 |
// |                                             | %33 = OpBitwiseOr %2 %31 %32                                   |
// |                                             |                                                                |
// |                                             | %34 = OpBitFieldUExtract %2 %5 %14 %6 ; extracts bit 6 from %5 |
// |                                             | %35 = OpBitFieldUExtract %2 %6 %14 %6 ; extracts bit 6 from %6 |
// |                                             | %36 = OpBitwiseOr %2 %34 %35                                   |
// |                                             |                                                                |
// |                                             | %37 = OpBitFieldUExtract %2 %5 %15 %6 ; extracts bit 7 from %5 |
// |                                             | %38 = OpBitFieldUExtract %2 %6 %15 %6 ; extracts bit 7 from %6 |
// |                                             | %39 = OpBitwiseOr %2 %37 %38                                   |
// |                                             |                                                                |
// |                                             | %40 = OpBitFieldInsert %2 %18 %21 %6 %6 ; inserts bit 1        |
// |                                             | %41 = OpBitFieldInsert %2 %40 %24 %10 %6 ; inserts bit 2       |
// |                                             | %42 = OpBitFieldInsert %2 %41 %27 %11 %6 ; inserts bit 3       |
// |                                             | %43 = OpBitFieldInsert %2 %42 %30 %12 %6 ; inserts bit 4       |
// |                                             | %44 = OpBitFieldInsert %2 %43 %33 %13 %6 ; inserts bit 5       |
// |                                             | %45 = OpBitFieldInsert %2 %44 %36 %14 %6 ; inserts bit 6       |
// |                                             | %46 = OpBitFieldInsert %2 %45 %39 %15 %6 ; inserts bit 7       |
// |                                             |  %9 = OpBitwiseOr %2 %5 %6 ; bit instruction                   |
// |                                             |       OpReturn                                                 |
// |                                             |       OpFunctionEnd                                            |
// ----------------------------------------------------------------------------------------------------------------
//
// After the transformation, %9 and %46 will be synonymous.
// clang-format on
class TransformationAddBitInstructionSynonym : public Transformation {
 public:
  explicit TransformationAddBitInstructionSynonym(
      protobufs::TransformationAddBitInstructionSynonym message);

  TransformationAddBitInstructionSynonym(
      const uint32_t instruction_result_id,
      const std::vector<uint32_t>& fresh_ids);

  // - |message_.instruction_result_id| must be a bit instruction.
  // - |message_.fresh_ids| must be fresh ids needed to apply the
  //   transformation.
  bool IsApplicable(
      opt::IRContext* ir_context,
      const TransformationContext& transformation_context) const override;

  // Adds a bit instruction synonym.
  void Apply(opt::IRContext* ir_context,
             TransformationContext* transformation_context) const override;

  std::unordered_set<uint32_t> GetFreshIds() const override;

  protobufs::Transformation ToMessage() const override;

  // Returns the number of fresh ids required to apply the transformation.
  static uint32_t GetRequiredFreshIdCount(opt::IRContext* ir_context,
                                          opt::Instruction* bit_instruction);

  //   Returns true if:
  // - A |bit_instruction| is one of OpBitwiseOr, OpBitwiseAnd, OpBitwiseXor or
  // OpNot.
  // - |bit_instruction|'s operands are scalars.
  // - The operands have the same signedness.
  static bool IsInstructionSupported(opt::IRContext* ir_context,
                                     opt::Instruction* instruction);

 private:
  protobufs::TransformationAddBitInstructionSynonym message_;

  // Adds OpBitwise* or OpNot synonym.
  void AddOpBitwiseOrOpNotSynonym(opt::IRContext* ir_context,
                                  TransformationContext* transformation_context,
                                  opt::Instruction* bitwise_instruction) const;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_TRANSFORMATION_ADD_BIT_INSTRUCTION_SYNONYM_H_

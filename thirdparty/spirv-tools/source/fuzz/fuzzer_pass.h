// Copyright (c) 2019 Google LLC
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

#ifndef SOURCE_FUZZ_FUZZER_PASS_H_
#define SOURCE_FUZZ_FUZZER_PASS_H_

#include <functional>
#include <vector>

#include "source/fuzz/fuzzer_context.h"
#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/ir_context.h"

namespace spvtools {
namespace fuzz {

// Interface for applying a pass of transformations to a module.
class FuzzerPass {
 public:
  FuzzerPass(opt::IRContext* ir_context,
             TransformationContext* transformation_context,
             FuzzerContext* fuzzer_context,
             protobufs::TransformationSequence* transformations,
             bool ignore_inapplicable_transformations);

  virtual ~FuzzerPass();

  // Applies the pass to the module |ir_context_|, assuming and updating
  // information from |transformation_context_|, and using |fuzzer_context_| to
  // guide the process.  Appends to |transformations_| all transformations that
  // were applied during the pass.
  virtual void Apply() = 0;

 protected:
  opt::IRContext* GetIRContext() const { return ir_context_; }

  TransformationContext* GetTransformationContext() const {
    return transformation_context_;
  }

  FuzzerContext* GetFuzzerContext() const { return fuzzer_context_; }

  protobufs::TransformationSequence* GetTransformations() const {
    return transformations_;
  }

  // Returns all instructions that are *available* at |inst_it|, which is
  // required to be inside block |block| of function |function| - that is, all
  // instructions at global scope and all instructions that strictly dominate
  // |inst_it|.
  //
  // Filters said instructions to return only those that satisfy the
  // |instruction_is_relevant| predicate.  This, for instance, could ignore all
  // instructions that have a particular decoration.
  std::vector<opt::Instruction*> FindAvailableInstructions(
      opt::Function* function, opt::BasicBlock* block,
      const opt::BasicBlock::iterator& inst_it,
      std::function<bool(opt::IRContext*, opt::Instruction*)>
          instruction_is_relevant) const;

  // A helper method that iterates through each instruction in each reachable
  // block of |function|, at all times tracking an instruction descriptor that
  // allows the latest instruction to be located even if it has no result id.
  //
  // The code to manipulate the instruction descriptor is a bit fiddly.  The
  // point of this method is to avoiding having to duplicate it in multiple
  // transformation passes.
  //
  // The function |action| is invoked for each instruction |inst_it| in block
  // |block| of function |function| that is encountered.  The
  // |instruction_descriptor| parameter to the function object allows |inst_it|
  // to be identified.
  //
  // In most intended use cases, the job of |action| is to randomly decide
  // whether to try to apply some transformation, and then - if selected - to
  // attempt to apply it.
  void ForEachInstructionWithInstructionDescriptor(
      opt::Function* function,
      std::function<
          void(opt::BasicBlock* block, opt::BasicBlock::iterator inst_it,
               const protobufs::InstructionDescriptor& instruction_descriptor)>
          action);

  // Applies the above overload of ForEachInstructionWithInstructionDescriptor
  // to every function in the module, so that |action| is applied to an
  // |instruction_descriptor| for every instruction, |inst_it|, of every |block|
  // in every |function|.
  void ForEachInstructionWithInstructionDescriptor(
      std::function<
          void(opt::Function* function, opt::BasicBlock* block,
               opt::BasicBlock::iterator inst_it,
               const protobufs::InstructionDescriptor& instruction_descriptor)>
          action);

  // A generic helper for applying a transformation that should be applicable
  // by construction, and adding it to the sequence of applied transformations.
  void ApplyTransformation(const Transformation& transformation);

  // A generic helper for applying a transformation only if it is applicable.
  // If it is applicable, the transformation is applied and then added to the
  // sequence of applied transformations and the function returns true.
  // Otherwise, the function returns false.
  bool MaybeApplyTransformation(const Transformation& transformation);

  // Returns the id of an OpTypeBool instruction.  If such an instruction does
  // not exist, a transformation is applied to add it.
  uint32_t FindOrCreateBoolType();

  // Returns the id of an OpTypeInt instruction, with width and signedness
  // specified by |width| and |is_signed|, respectively.  If such an instruction
  // does not exist, a transformation is applied to add it.
  uint32_t FindOrCreateIntegerType(uint32_t width, bool is_signed);

  // Returns the id of an OpTypeFloat instruction, with width specified by
  // |width|.  If such an instruction does not exist, a transformation is
  // applied to add it.
  uint32_t FindOrCreateFloatType(uint32_t width);

  // Returns the id of an OpTypeFunction %<return_type_id> %<...argument_id>
  // instruction. If such an instruction doesn't exist, a transformation
  // is applied to create a new one.
  uint32_t FindOrCreateFunctionType(uint32_t return_type_id,
                                    const std::vector<uint32_t>& argument_id);

  // Returns the id of an OpTypeVector instruction, with |component_type_id|
  // (which must already exist) as its base type, and |component_count|
  // elements (which must be in the range [2, 4]).  If such an instruction does
  // not exist, a transformation is applied to add it.
  uint32_t FindOrCreateVectorType(uint32_t component_type_id,
                                  uint32_t component_count);

  // Returns the id of an OpTypeMatrix instruction, with |column_count| columns
  // and |row_count| rows (each of which must be in the range [2, 4]).  If the
  // float and vector types required to build this matrix type or the matrix
  // type itself do not exist, transformations are applied to add them.
  uint32_t FindOrCreateMatrixType(uint32_t column_count, uint32_t row_count);

  // Returns the id of an OpTypeStruct instruction with |component_type_ids| as
  // type ids for struct's components. If no such a struct type exists,
  // transformations are applied to add it. |component_type_ids| may not contain
  // a result id of an OpTypeFunction.
  uint32_t FindOrCreateStructType(
      const std::vector<uint32_t>& component_type_ids);

  // Returns the id of a pointer type with base type |base_type_id| (which must
  // already exist) and storage class |storage_class|.  A transformation is
  // applied to add the pointer if it does not already exist.
  uint32_t FindOrCreatePointerType(uint32_t base_type_id,
                                   spv::StorageClass storage_class);

  // Returns the id of an OpTypePointer instruction, with a integer base
  // type of width and signedness specified by |width| and |is_signed|,
  // respectively.  If the pointer type or required integer base type do not
  // exist, transformations are applied to add them.
  uint32_t FindOrCreatePointerToIntegerType(uint32_t width, bool is_signed,
                                            spv::StorageClass storage_class);

  // Returns the id of an OpConstant instruction, with a integer type of
  // width and signedness specified by |width| and |is_signed|, respectively,
  // with |words| as its value.  If either the required integer type or the
  // constant do not exist, transformations are applied to add them.
  // The returned id either participates in IdIsIrrelevant fact or not,
  // depending on the |is_irrelevant| parameter.
  uint32_t FindOrCreateIntegerConstant(const std::vector<uint32_t>& words,
                                       uint32_t width, bool is_signed,
                                       bool is_irrelevant);

  // Returns the id of an OpConstant instruction, with a floating-point
  // type of width specified by |width|, with |words| as its value.  If either
  // the required floating-point type or the constant do not exist,
  // transformations are applied to add them. The returned id either
  // participates in IdIsIrrelevant fact or not, depending on the
  // |is_irrelevant| parameter.
  uint32_t FindOrCreateFloatConstant(const std::vector<uint32_t>& words,
                                     uint32_t width, bool is_irrelevant);

  // Returns the id of an OpConstantTrue or OpConstantFalse instruction,
  // according to |value|.  If either the required instruction or the bool
  // type do not exist, transformations are applied to add them.
  // The returned id either participates in IdIsIrrelevant fact or not,
  // depending on the |is_irrelevant| parameter.
  uint32_t FindOrCreateBoolConstant(bool value, bool is_irrelevant);

  // Returns the id of an OpConstant instruction of type with |type_id|
  // that consists of |words|. If that instruction doesn't exist,
  // transformations are applied to add it. |type_id| must be a valid
  // result id of either scalar or boolean OpType* instruction that exists
  // in the module. The returned id either participates in IdIsIrrelevant fact
  // or not, depending on the |is_irrelevant| parameter.
  uint32_t FindOrCreateConstant(const std::vector<uint32_t>& words,
                                uint32_t type_id, bool is_irrelevant);

  // Returns the id of an OpConstantComposite instruction of type with |type_id|
  // that consists of |component_ids|. If that instruction doesn't exist,
  // transformations are applied to add it. |type_id| must be a valid
  // result id of an OpType* instruction that represents a composite type
  // (i.e. a vector, matrix, struct or array).
  // The returned id either participates in IdIsIrrelevant fact or not,
  // depending on the |is_irrelevant| parameter.
  uint32_t FindOrCreateCompositeConstant(
      const std::vector<uint32_t>& component_ids, uint32_t type_id,
      bool is_irrelevant);

  // Returns the result id of an instruction of the form:
  //   %id = OpUndef %|type_id|
  // If no such instruction exists, a transformation is applied to add it.
  uint32_t FindOrCreateGlobalUndef(uint32_t type_id);

  // Returns the id of an OpNullConstant instruction of type |type_id|. If
  // that instruction doesn't exist, it is added through a transformation.
  // |type_id| must be a valid result id of an OpType* instruction that exists
  // in the module.
  uint32_t FindOrCreateNullConstant(uint32_t type_id);

  // Define a *basic type* to be an integer, boolean or floating-point type,
  // or a matrix, vector, struct or fixed-size array built from basic types.  In
  // particular, a basic type cannot contain an opaque type (such as an image),
  // or a runtime-sized array.
  //
  // Yields a pair, (basic_type_ids, basic_type_ids_to_pointers), such that:
  // - basic_type_ids captures every basic type declared in the module.
  // - basic_type_ids_to_pointers maps every such basic type to the sequence
  //   of all pointer types that have storage class |storage_class| and the
  //   given basic type as their pointee type.  The sequence may be empty for
  //   some basic types if no pointers to those types are defined for the given
  //   storage class, and the sequence will have multiple elements if there are
  //   repeated pointer declarations for the same basic type and storage class.
  std::pair<std::vector<uint32_t>, std::map<uint32_t, std::vector<uint32_t>>>
  GetAvailableBasicTypesAndPointers(spv::StorageClass storage_class) const;

  // Given a type id, |scalar_or_composite_type_id|, which must correspond to
  // some scalar or composite type, returns the result id of an instruction
  // defining a constant of the given type that is zero or false at everywhere.
  // If such an instruction does not yet exist, transformations are applied to
  // add it. The returned id either participates in IdIsIrrelevant fact or not,
  // depending on the |is_irrelevant| parameter.
  //
  // Examples:
  // --------------+-------------------------------
  //   TYPE        | RESULT is id corresponding to
  // --------------+-------------------------------
  //   bool        | false
  // --------------+-------------------------------
  //   bvec4       | (false, false, false, false)
  // --------------+-------------------------------
  //   float       | 0.0
  // --------------+-------------------------------
  //   vec2        | (0.0, 0.0)
  // --------------+-------------------------------
  //   int[3]      | [0, 0, 0]
  // --------------+-------------------------------
  //   struct S {  |
  //     int i;    | S(0, false, (0u, 0u))
  //     bool b;   |
  //     uint2 u;  |
  //   }           |
  // --------------+-------------------------------
  uint32_t FindOrCreateZeroConstant(uint32_t scalar_or_composite_type_id,
                                    bool is_irrelevant);

  // Adds a pair (id_use_descriptor, |replacement_id|) to the vector
  // |uses_to_replace|, where id_use_descriptor is the id use descriptor
  // representing the usage of an id in the |use_inst| instruction, at operand
  // index |use_index|, only if the instruction is in a basic block.
  // If the instruction is not in a basic block, it does nothing.
  void MaybeAddUseToReplace(
      opt::Instruction* use_inst, uint32_t use_index, uint32_t replacement_id,
      std::vector<std::pair<protobufs::IdUseDescriptor, uint32_t>>*
          uses_to_replace);

  // Returns the preheader of the loop with header |header_id|, which satisfies
  // all of the following conditions:
  // - It is the only out-of-loop predecessor of the header
  // - It unconditionally branches to the header
  // - It is not a loop header itself
  // If such preheader does not exist, a new one is added and returned.
  // Requires |header_id| to be the label id of a loop header block that is
  // reachable in the CFG (and thus has at least 2 predecessors).
  opt::BasicBlock* GetOrCreateSimpleLoopPreheader(uint32_t header_id);

  // Returns the second block in the pair obtained by splitting |block_id| just
  // after the last OpPhi or OpVariable instruction in it. Assumes that the
  // block is not a loop header.
  opt::BasicBlock* SplitBlockAfterOpPhiOrOpVariable(uint32_t block_id);

  // Returns the id of an available local variable (storage class Function) with
  // the fact PointeeValueIsIrrelevant set according to
  // |pointee_value_is_irrelevant|. If there is no such variable, it creates one
  // in the |function| adding a zero initializer constant that is irrelevant.
  // The new variable has the fact PointeeValueIsIrrelevant set according to
  // |pointee_value_is_irrelevant|. The function returns the id of the created
  // variable.
  uint32_t FindOrCreateLocalVariable(uint32_t pointer_type_id,
                                     uint32_t function_id,
                                     bool pointee_value_is_irrelevant);

  // Returns the id of an available global variable (storage class Private or
  // Workgroup) with the fact PointeeValueIsIrrelevant set according to
  // |pointee_value_is_irrelevant|. If there is no such variable, it creates
  // one, adding a zero initializer constant that is irrelevant. The new
  // variable has the fact PointeeValueIsIrrelevant set according to
  // |pointee_value_is_irrelevant|. The function returns the id of the created
  // variable.
  uint32_t FindOrCreateGlobalVariable(uint32_t pointer_type_id,
                                      bool pointee_value_is_irrelevant);

 private:
  opt::IRContext* ir_context_;
  TransformationContext* transformation_context_;
  FuzzerContext* fuzzer_context_;
  protobufs::TransformationSequence* transformations_;
  // If set, then transformations that should be applicable by construction are
  // still tested for applicability, and ignored if they turn out to be
  // inapplicable. Otherwise, applicability by construction is asserted.
  const bool ignore_inapplicable_transformations_;
};

}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_FUZZER_PASS_H_

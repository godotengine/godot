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

#ifndef SOURCE_FUZZ_FUZZER_UTIL_H_
#define SOURCE_FUZZ_FUZZER_UTIL_H_

#include <iostream>
#include <map>
#include <vector>

#include "source/fuzz/protobufs/spirvfuzz_protobufs.h"
#include "source/fuzz/transformation_context.h"
#include "source/opt/basic_block.h"
#include "source/opt/instruction.h"
#include "source/opt/ir_context.h"
#include "source/opt/module.h"
#include "spirv-tools/libspirv.hpp"

namespace spvtools {
namespace fuzz {

// Provides types and global utility methods for use by the fuzzer
namespace fuzzerutil {

// A silent message consumer.
extern const spvtools::MessageConsumer kSilentMessageConsumer;

// Function type that produces a SPIR-V module.
using ModuleSupplier = std::function<std::unique_ptr<opt::IRContext>()>;

// Builds a new opt::IRContext object. Returns true if successful and changes
// the |ir_context| parameter. Otherwise (if any errors occur), returns false
// and |ir_context| remains unchanged.
bool BuildIRContext(spv_target_env target_env,
                    const spvtools::MessageConsumer& message_consumer,
                    const std::vector<uint32_t>& binary_in,
                    spv_validator_options validator_options,
                    std::unique_ptr<spvtools::opt::IRContext>* ir_context);

// Returns true if and only if the module does not define the given id.
bool IsFreshId(opt::IRContext* context, uint32_t id);

// Updates the module's id bound if needed so that it is large enough to
// account for the given id.
void UpdateModuleIdBound(opt::IRContext* context, uint32_t id);

// Return the block with id |maybe_block_id| if it exists, and nullptr
// otherwise.
opt::BasicBlock* MaybeFindBlock(opt::IRContext* context,
                                uint32_t maybe_block_id);

// When adding an edge from |bb_from| to |bb_to| (which are assumed to be blocks
// in the same function), it is important to supply |bb_to| with ids that can be
// used to augment OpPhi instructions in the case that there is not already such
// an edge.  This function returns true if and only if the ids provided in
// |phi_ids| suffice for this purpose,
bool PhiIdsOkForNewEdge(
    opt::IRContext* context, opt::BasicBlock* bb_from, opt::BasicBlock* bb_to,
    const google::protobuf::RepeatedField<google::protobuf::uint32>& phi_ids);

// Returns an OpBranchConditional instruction that will create an unreachable
// branch from |bb_from_id| to |bb_to_id|. |bool_id| must be a result id of
// either OpConstantTrue or OpConstantFalse. Based on the opcode of |bool_id|,
// operands of the returned instruction will be positioned in a way that the
// branch from |bb_from_id| to |bb_to_id| is always unreachable.
opt::Instruction CreateUnreachableEdgeInstruction(opt::IRContext* ir_context,
                                                  uint32_t bb_from_id,
                                                  uint32_t bb_to_id,
                                                  uint32_t bool_id);

// Requires that |bool_id| is a valid result id of either OpConstantTrue or
// OpConstantFalse, that PhiIdsOkForNewEdge(context, bb_from, bb_to, phi_ids)
// holds, and that bb_from ends with "OpBranch %some_block".  Turns OpBranch
// into "OpBranchConditional |condition_value| ...", such that control will
// branch to %some_block, with |bb_to| being the unreachable alternative.
// Updates OpPhi instructions in |bb_to| using |phi_ids| so that the new edge is
// valid. |condition_value| above is equal to |true| if |bool_id| is a result id
// of an OpConstantTrue instruction.
void AddUnreachableEdgeAndUpdateOpPhis(
    opt::IRContext* context, opt::BasicBlock* bb_from, opt::BasicBlock* bb_to,
    uint32_t bool_id,
    const google::protobuf::RepeatedField<google::protobuf::uint32>& phi_ids);

// Returns true if and only if |loop_header_id| is a loop header and
// |block_id| is a reachable block branching to and dominated by
// |loop_header_id|.
bool BlockIsBackEdge(opt::IRContext* context, uint32_t block_id,
                     uint32_t loop_header_id);

// Returns true if and only if |maybe_loop_header_id| is a loop header and
// |block_id| is in the continue construct of the associated loop.
bool BlockIsInLoopContinueConstruct(opt::IRContext* context, uint32_t block_id,
                                    uint32_t maybe_loop_header_id);

// If |block| contains |inst|, an iterator for |inst| is returned.
// Otherwise |block|->end() is returned.
opt::BasicBlock::iterator GetIteratorForInstruction(
    opt::BasicBlock* block, const opt::Instruction* inst);

// Determines whether it is OK to insert an instruction with opcode |opcode|
// before |instruction_in_block|.
bool CanInsertOpcodeBeforeInstruction(
    spv::Op opcode, const opt::BasicBlock::iterator& instruction_in_block);

// Determines whether it is OK to make a synonym of |inst|.
// |transformation_context| is used to verify that the result id of |inst|
// does not participate in IdIsIrrelevant fact.
bool CanMakeSynonymOf(opt::IRContext* ir_context,
                      const TransformationContext& transformation_context,
                      const opt::Instruction& inst);

// Determines whether the given type is a composite; that is: an array, matrix,
// struct or vector.
bool IsCompositeType(const opt::analysis::Type* type);

// Returns a vector containing the same elements as |repeated_field|.
std::vector<uint32_t> RepeatedFieldToVector(
    const google::protobuf::RepeatedField<uint32_t>& repeated_field);

// Given a type id, |base_object_type_id|, returns 0 if the type is not a
// composite type or if |index| is too large to be used as an index into the
// composite.  Otherwise returns the type id of the type associated with the
// composite's index.
//
// Example: if |base_object_type_id| is 10, and we have:
//
// %10 = OpTypeStruct %3 %4 %5
//
// then 3 will be returned if |index| is 0, 5 if |index| is 2, and 0 if index
// is 3 or larger.
uint32_t WalkOneCompositeTypeIndex(opt::IRContext* context,
                                   uint32_t base_object_type_id,
                                   uint32_t index);

// Given a type id, |base_object_type_id|, checks that the given sequence of
// |indices| is suitable for indexing into this type.  Returns the id of the
// type of the final sub-object reached via the indices if they are valid, and
// 0 otherwise.
uint32_t WalkCompositeTypeIndices(
    opt::IRContext* context, uint32_t base_object_type_id,
    const google::protobuf::RepeatedField<google::protobuf::uint32>& indices);

// Returns the number of members associated with |struct_type_instruction|,
// which must be an OpStructType instruction.
uint32_t GetNumberOfStructMembers(
    const opt::Instruction& struct_type_instruction);

// Returns the constant size of the array associated with
// |array_type_instruction|, which must be an OpArrayType instruction. Returns
// 0 if there is not a static size.
uint32_t GetArraySize(const opt::Instruction& array_type_instruction,
                      opt::IRContext* context);

// Returns the bound for indexing into a composite of type
// |composite_type_inst|, i.e. the number of fields of a struct, the size of an
// array, the number of components of a vector, or the number of columns of a
// matrix. |composite_type_inst| must be the type of a composite.
uint32_t GetBoundForCompositeIndex(const opt::Instruction& composite_type_inst,
                                   opt::IRContext* ir_context);

// Returns memory semantics mask for specific storage class.
spv::MemorySemanticsMask GetMemorySemanticsForStorageClass(
    spv::StorageClass storage_class);

// Returns true if and only if |context| is valid, according to the validator
// instantiated with |validator_options|.  |consumer| is used for error
// reporting.
bool IsValid(const opt::IRContext* context,
             spv_validator_options validator_options, MessageConsumer consumer);

// Returns true if and only if IsValid(|context|, |validator_options|) holds,
// and furthermore every basic block in |context| has its enclosing function as
// its parent, and every instruction in |context| has a distinct unique id.
// |consumer| is used for error reporting.
bool IsValidAndWellFormed(const opt::IRContext* context,
                          spv_validator_options validator_options,
                          MessageConsumer consumer);

// Returns a clone of |context|, by writing |context| to a binary and then
// parsing it again.
std::unique_ptr<opt::IRContext> CloneIRContext(opt::IRContext* context);

// Returns true if and only if |id| is the id of a type that is not a function
// type.
bool IsNonFunctionTypeId(opt::IRContext* ir_context, uint32_t id);

// Returns true if and only if |block_id| is a merge block or continue target
bool IsMergeOrContinue(opt::IRContext* ir_context, uint32_t block_id);

// Returns the id of the header of the loop corresponding to the given loop
// merge block. Returns 0 if |merge_block_id| is not a loop merge block.
uint32_t GetLoopFromMergeBlock(opt::IRContext* ir_context,
                               uint32_t merge_block_id);

// Returns the result id of an instruction of the form:
//  %id = OpTypeFunction |type_ids|
// or 0 if no such instruction exists.
uint32_t FindFunctionType(opt::IRContext* ir_context,
                          const std::vector<uint32_t>& type_ids);

// Returns a type instruction (OpTypeFunction) for |function|.
// Returns |nullptr| if type is not found.
opt::Instruction* GetFunctionType(opt::IRContext* context,
                                  const opt::Function* function);

// Returns the function with result id |function_id|, or |nullptr| if no such
// function exists.
opt::Function* FindFunction(opt::IRContext* ir_context, uint32_t function_id);

// Returns true if |function| has a block that the termination instruction is
// OpKill or OpUnreachable.
bool FunctionContainsOpKillOrUnreachable(const opt::Function& function);

// Returns |true| if one of entry points has function id |function_id|.
bool FunctionIsEntryPoint(opt::IRContext* context, uint32_t function_id);

// Checks whether |id| is available (according to dominance rules) at the use
// point defined by input operand |use_input_operand_index| of
// |use_instruction|. |use_instruction| must be a in some basic block.
bool IdIsAvailableAtUse(opt::IRContext* context,
                        opt::Instruction* use_instruction,
                        uint32_t use_input_operand_index, uint32_t id);

// Checks whether |id| is available (according to dominance rules) at the
// program point directly before |instruction|. |instruction| must be in some
// basic block.
bool IdIsAvailableBeforeInstruction(opt::IRContext* context,
                                    opt::Instruction* instruction, uint32_t id);

// Returns true if and only if |instruction| is an OpFunctionParameter
// associated with |function|.
bool InstructionIsFunctionParameter(opt::Instruction* instruction,
                                    opt::Function* function);

// Returns the type id of the instruction defined by |result_id|, or 0 if there
// is no such result id.
uint32_t GetTypeId(opt::IRContext* context, uint32_t result_id);

// Given |pointer_type_inst|, which must be an OpTypePointer instruction,
// returns the id of the associated pointee type.
uint32_t GetPointeeTypeIdFromPointerType(opt::Instruction* pointer_type_inst);

// Given |pointer_type_id|, which must be the id of a pointer type, returns the
// id of the associated pointee type.
uint32_t GetPointeeTypeIdFromPointerType(opt::IRContext* context,
                                         uint32_t pointer_type_id);

// Given |pointer_type_inst|, which must be an OpTypePointer instruction,
// returns the associated storage class.
spv::StorageClass GetStorageClassFromPointerType(
    opt::Instruction* pointer_type_inst);

// Given |pointer_type_id|, which must be the id of a pointer type, returns the
// associated storage class.
spv::StorageClass GetStorageClassFromPointerType(opt::IRContext* context,
                                                 uint32_t pointer_type_id);

// Returns the id of a pointer with pointee type |pointee_type_id| and storage
// class |storage_class|, if it exists, and 0 otherwise.
uint32_t MaybeGetPointerType(opt::IRContext* context, uint32_t pointee_type_id,
                             spv::StorageClass storage_class);

// Given an instruction |inst| and an operand absolute index |absolute_index|,
// returns the index of the operand restricted to the input operands.
uint32_t InOperandIndexFromOperandIndex(const opt::Instruction& inst,
                                        uint32_t absolute_index);

// Returns true if and only if |type| is one of the types for which it is legal
// to have an OpConstantNull value. This may depend on the capabilities declared
// in |context|.
bool IsNullConstantSupported(opt::IRContext* context,
                             const opt::Instruction& type);

// Returns true if and only if the SPIR-V version being used requires that
// global variables accessed in the static call graph of an entry point need
// to be listed in that entry point's interface.
bool GlobalVariablesMustBeDeclaredInEntryPointInterfaces(
    const opt::IRContext* context);

// Adds |id| into the interface of every entry point of the shader.
// Does nothing if SPIR-V doesn't require global variables, that are accessed
// from an entry point function, to be listed in that function's interface.
void AddVariableIdToEntryPointInterfaces(opt::IRContext* context, uint32_t id);

// Adds a global variable with storage class |storage_class| to the module, with
// type |type_id| and either no initializer or |initializer_id| as an
// initializer, depending on whether |initializer_id| is 0. The global variable
// has result id |result_id|. Updates module's id bound to accommodate for
// |result_id|.
//
// - |type_id| must be the id of a pointer type with the same storage class as
//   |storage_class|.
// - |storage_class| must be Private or Workgroup.
// - |initializer_id| must be 0 if |storage_class| is Workgroup, and otherwise
//   may either be 0 or the id of a constant whose type is the pointee type of
//   |type_id|.
//
// Returns a pointer to the new global variable instruction.
opt::Instruction* AddGlobalVariable(opt::IRContext* context, uint32_t result_id,
                                    uint32_t type_id,
                                    spv::StorageClass storage_class,
                                    uint32_t initializer_id);

// Adds an instruction to the start of |function_id|, of the form:
//   |result_id| = OpVariable |type_id| Function |initializer_id|.
// Updates module's id bound to accommodate for |result_id|.
//
// - |type_id| must be the id of a pointer type with Function storage class.
// - |initializer_id| must be the id of a constant with the same type as the
//   pointer's pointee type.
// - |function_id| must be the id of a function.
//
// Returns a pointer to the new local variable instruction.
opt::Instruction* AddLocalVariable(opt::IRContext* context, uint32_t result_id,
                                   uint32_t type_id, uint32_t function_id,
                                   uint32_t initializer_id);

// Returns true if the vector |arr| has duplicates.
bool HasDuplicates(const std::vector<uint32_t>& arr);

// Checks that the given vector |arr| contains a permutation of a range
// [lo, hi]. That being said, all elements in the range are present without
// duplicates. If |arr| is empty, returns true iff |lo > hi|.
bool IsPermutationOfRange(const std::vector<uint32_t>& arr, uint32_t lo,
                          uint32_t hi);

// Returns OpFunctionParameter instructions corresponding to the function
// with result id |function_id|.
std::vector<opt::Instruction*> GetParameters(opt::IRContext* ir_context,
                                             uint32_t function_id);

// Removes an OpFunctionParameter instruction with result id |parameter_id|
// from the its function. Parameter's function must not be an entry-point
// function. The function must have a parameter with result id |parameter_id|.
//
// Prefer using this function to opt::Function::RemoveParameter since
// this function also guarantees that |ir_context| has no invalid pointers
// to the removed parameter.
void RemoveParameter(opt::IRContext* ir_context, uint32_t parameter_id);

// Returns all OpFunctionCall instructions that call a function with result id
// |function_id|.
std::vector<opt::Instruction*> GetCallers(opt::IRContext* ir_context,
                                          uint32_t function_id);

// Returns a function that contains OpFunctionParameter instruction with result
// id |param_id|. Returns nullptr if the module has no such function.
opt::Function* GetFunctionFromParameterId(opt::IRContext* ir_context,
                                          uint32_t param_id);

// Changes the type of function |function_id| so that its return type is
// |return_type_id| and its parameters' types are |parameter_type_ids|. If a
// suitable function type already exists in the module, it is used, otherwise
// |new_function_type_result_id| is used as the result id of a suitable new
// function type instruction. If the old type of the function doesn't have any
// more users, it is removed from the module. Returns the result id of the
// OpTypeFunction instruction that is used as a type of the function with
// |function_id|.
//
// CAUTION: When the old type of the function is removed from the module, its
//          memory is deallocated. Be sure not to use any pointers to the old
//          type when this function returns.
uint32_t UpdateFunctionType(opt::IRContext* ir_context, uint32_t function_id,
                            uint32_t new_function_type_result_id,
                            uint32_t return_type_id,
                            const std::vector<uint32_t>& parameter_type_ids);

// Creates new OpTypeFunction instruction in the module. |type_ids| may not be
// empty. It may not contain result ids of OpTypeFunction instructions.
// |type_ids[i]| may not be a result id of OpTypeVoid instruction for |i >= 1|.
// |result_id| may not equal to 0. Updates module's id bound to accommodate for
// |result_id|.
void AddFunctionType(opt::IRContext* ir_context, uint32_t result_id,
                     const std::vector<uint32_t>& type_ids);

// Returns a result id of an OpTypeFunction instruction in the module. Creates a
// new instruction if required and returns |result_id|. type_ids| may not be
// empty. It may not contain result ids of OpTypeFunction instructions.
// |type_ids[i]| may not be a result id of OpTypeVoid instruction for |i >= 1|.
// |result_id| must not be equal to 0. Updates module's id bound to accommodate
// for |result_id|.
uint32_t FindOrCreateFunctionType(opt::IRContext* ir_context,
                                  uint32_t result_id,
                                  const std::vector<uint32_t>& type_ids);

// Returns a result id of an OpTypeInt instruction if present. Returns 0
// otherwise.
uint32_t MaybeGetIntegerType(opt::IRContext* ir_context, uint32_t width,
                             bool is_signed);

// Returns a result id of an OpTypeFloat instruction if present. Returns 0
// otherwise.
uint32_t MaybeGetFloatType(opt::IRContext* ir_context, uint32_t width);

// Returns a result id of an OpTypeBool instruction if present. Returns 0
// otherwise.
uint32_t MaybeGetBoolType(opt::IRContext* ir_context);

// Returns a result id of an OpTypeVector instruction if present. Returns 0
// otherwise. |component_type_id| must be a valid result id of an OpTypeInt,
// OpTypeFloat or OpTypeBool instruction in the module. |element_count| must be
// in the range [2, 4].
uint32_t MaybeGetVectorType(opt::IRContext* ir_context,
                            uint32_t component_type_id, uint32_t element_count);

// Returns a result id of an OpTypeStruct instruction whose field types exactly
// match |component_type_ids| if such an instruction is present. Returns 0
// otherwise. |component_type_ids| may not contain a result id of an
// OpTypeFunction.
uint32_t MaybeGetStructType(opt::IRContext* ir_context,
                            const std::vector<uint32_t>& component_type_ids);

// Returns a result id of an OpTypeVoid instruction if present. Returns 0
// otherwise.
uint32_t MaybeGetVoidType(opt::IRContext* ir_context);

// Recursive definition is the following:
// - if |scalar_or_composite_type_id| is a result id of a scalar type - returns
//   a result id of the following constants (depending on the type): int -> 0,
//   float -> 0.0, bool -> false.
// - otherwise, returns a result id of an OpConstantComposite instruction.
//   Every component of the composite constant is looked up by calling this
//   function with the type id of that component.
// Returns 0 if no such instruction is present in the module.
// The returned id either participates in IdIsIrrelevant fact or not, depending
// on the |is_irrelevant| parameter.
uint32_t MaybeGetZeroConstant(
    opt::IRContext* ir_context,
    const TransformationContext& transformation_context,
    uint32_t scalar_or_composite_type_id, bool is_irrelevant);

// Returns true if it is possible to create an OpConstant or an
// OpConstantComposite instruction of type |type_id|. That is, returns true if
// the type associated with |type_id| and all its constituents are either scalar
// or composite.
bool CanCreateConstant(opt::IRContext* ir_context, uint32_t type_id);

// Returns the result id of an OpConstant instruction. |scalar_type_id| must be
// a result id of a scalar type (i.e. int, float or bool). Returns 0 if no such
// instruction is present in the module. The returned id either participates in
// IdIsIrrelevant fact or not, depending on the |is_irrelevant| parameter.
uint32_t MaybeGetScalarConstant(
    opt::IRContext* ir_context,
    const TransformationContext& transformation_context,
    const std::vector<uint32_t>& words, uint32_t scalar_type_id,
    bool is_irrelevant);

// Returns the result id of an OpConstantComposite instruction.
// |composite_type_id| must be a result id of a composite type (i.e. vector,
// matrix, struct or array). Returns 0 if no such instruction is present in the
// module. The returned id either participates in IdIsIrrelevant fact or not,
// depending on the |is_irrelevant| parameter.
uint32_t MaybeGetCompositeConstant(
    opt::IRContext* ir_context,
    const TransformationContext& transformation_context,
    const std::vector<uint32_t>& component_ids, uint32_t composite_type_id,
    bool is_irrelevant);

// Returns the result id of an OpConstant instruction of integral type.
// Returns 0 if no such instruction or type is present in the module.
// The returned id either participates in IdIsIrrelevant fact or not, depending
// on the |is_irrelevant| parameter.
uint32_t MaybeGetIntegerConstant(
    opt::IRContext* ir_context,
    const TransformationContext& transformation_context,
    const std::vector<uint32_t>& words, uint32_t width, bool is_signed,
    bool is_irrelevant);

// Returns the id of a 32-bit integer constant in the module with type
// |int_type_id| and value |value|, or 0 if no such constant exists in the
// module. |int_type_id| must exist in the module and it must correspond to a
// 32-bit integer type.
uint32_t MaybeGetIntegerConstantFromValueAndType(opt::IRContext* ir_context,
                                                 uint32_t value,
                                                 uint32_t int_type_id);

// Returns the result id of an OpConstant instruction of floating-point type.
// Returns 0 if no such instruction or type is present in the module.
// The returned id either participates in IdIsIrrelevant fact or not, depending
// on the |is_irrelevant| parameter.
uint32_t MaybeGetFloatConstant(
    opt::IRContext* ir_context,
    const TransformationContext& transformation_context,
    const std::vector<uint32_t>& words, uint32_t width, bool is_irrelevant);

// Returns the id of a boolean constant with value |value| if it exists in the
// module, or 0 otherwise. The returned id either participates in IdIsIrrelevant
// fact or not, depending on the |is_irrelevant| parameter.
uint32_t MaybeGetBoolConstant(
    opt::IRContext* context,
    const TransformationContext& transformation_context, bool value,
    bool is_irrelevant);

// Returns a vector of words representing the integer |value|, only considering
// the last |width| bits. The last |width| bits are sign-extended if the value
// is signed, zero-extended if it is unsigned.
// |width| must be <= 64.
// If |width| <= 32, returns a vector containing one value. If |width| > 64,
// returns a vector containing two values, with the first one representing the
// lower-order word of the value and the second one representing the
// higher-order word.
std::vector<uint32_t> IntToWords(uint64_t value, uint32_t width,
                                 bool is_signed);

// Returns a bit pattern that represents a floating-point |value|.
inline uint32_t FloatToWord(float value) {
  uint32_t result;
  memcpy(&result, &value, sizeof(uint32_t));
  return result;
}

// Returns true if any of the following is true:
// - |type1_id| and |type2_id| are the same id
// - |type1_id| and |type2_id| refer to integer scalar or vector types, only
//   differing by their signedness.
bool TypesAreEqualUpToSign(opt::IRContext* ir_context, uint32_t type1_id,
                           uint32_t type2_id);

// Converts repeated field of UInt32Pair to a map. If two or more equal values
// of |UInt32Pair::first()| are available in |data|, the last value of
// |UInt32Pair::second()| is used.
std::map<uint32_t, uint32_t> RepeatedUInt32PairToMap(
    const google::protobuf::RepeatedPtrField<protobufs::UInt32Pair>& data);

// Converts a map into a repeated field of UInt32Pair.
google::protobuf::RepeatedPtrField<protobufs::UInt32Pair>
MapToRepeatedUInt32Pair(const std::map<uint32_t, uint32_t>& data);

// Returns the last instruction in |block_id| before which an instruction with
// opcode |opcode| can be inserted, or nullptr if there is no such instruction.
opt::Instruction* GetLastInsertBeforeInstruction(opt::IRContext* ir_context,
                                                 uint32_t block_id,
                                                 spv::Op opcode);

// Checks whether various conditions hold related to the acceptability of
// replacing the id use at |use_in_operand_index| of |use_instruction| with a
// synonym or another id of appropriate type if the original id is irrelevant.
// In particular, this checks that:
// - If id use is an index of an irrelevant id (|use_in_operand_index > 0|)
//   in OpAccessChain - it can't be replaced.
// - The id use is not an index into a struct field in an OpAccessChain - such
//   indices must be constants, so it is dangerous to replace them.
// - The id use is not a pointer function call argument, on which there are
//   restrictions that make replacement problematic.
// - The id use is not the Sample parameter of an OpImageTexelPointer
//   instruction, as this must satisfy particular requirements.
bool IdUseCanBeReplaced(opt::IRContext* ir_context,
                        const TransformationContext& transformation_context,
                        opt::Instruction* use_instruction,
                        uint32_t use_in_operand_index);

// Requires that |struct_type_id| is the id of a struct type, and (as per the
// SPIR-V spec) that either all or none of the members of |struct_type_id| have
// the BuiltIn decoration. Returns true if and only if all members have the
// BuiltIn decoration.
bool MembersHaveBuiltInDecoration(opt::IRContext* ir_context,
                                  uint32_t struct_type_id);

// Returns true if and only if |id| is decorated with either Block or
// BufferBlock.  Even though these decorations are only allowed on struct types,
// for convenience |id| can be any result id so that it is possible to call this
// method on something that *might* be a struct type.
bool HasBlockOrBufferBlockDecoration(opt::IRContext* ir_context, uint32_t id);

// Returns true iff splitting block |block_to_split| just before the instruction
// |split_before| would separate an OpSampledImage instruction from its usage.
bool SplittingBeforeInstructionSeparatesOpSampledImageDefinitionFromUse(
    opt::BasicBlock* block_to_split, opt::Instruction* split_before);

// Returns true if the instruction given has no side effects.
// TODO(https://github.com/KhronosGroup/SPIRV-Tools/issues/3758): Add any
//  missing instructions to the list. In particular, GLSL extended instructions
//  (called using OpExtInst) have not been considered.
bool InstructionHasNoSideEffects(const opt::Instruction& instruction);

// Returns a set of the ids of all the return blocks that are reachable from
// the entry block of |function_id|.
// Assumes that the function exists in the module.
std::set<uint32_t> GetReachableReturnBlocks(opt::IRContext* ir_context,
                                            uint32_t function_id);

// Returns true if changing terminator instruction to |new_terminator| in the
// basic block with id |block_id| preserves domination rules and valid block
// order (i.e. dominator must always appear before dominated in the CFG).
// Returns false otherwise.
bool NewTerminatorPreservesDominationRules(opt::IRContext* ir_context,
                                           uint32_t block_id,
                                           opt::Instruction new_terminator);

// Return the iterator that points to the function with the corresponding
// function id. If the function is not found, return the pointer pointing to
// module()->end().
opt::Module::iterator GetFunctionIterator(opt::IRContext* ir_context,
                                          uint32_t function_id);

// Returns true if the instruction with opcode |opcode| does not change its
// behaviour depending on the signedness of the operand at
// |use_in_operand_index|.
// Assumes that the operand must be the id of an integer scalar or vector.
bool IsAgnosticToSignednessOfOperand(spv::Op opcode,
                                     uint32_t use_in_operand_index);

// Returns true if |type_id_1| and |type_id_2| represent compatible types
// given the context of the instruction with |opcode| (i.e. we can replace
// an operand of |opcode| of the first type with an id of the second type
// and vice-versa).
bool TypesAreCompatible(opt::IRContext* ir_context, spv::Op opcode,
                        uint32_t use_in_operand_index, uint32_t type_id_1,
                        uint32_t type_id_2);

}  // namespace fuzzerutil
}  // namespace fuzz
}  // namespace spvtools

#endif  // SOURCE_FUZZ_FUZZER_UTIL_H_

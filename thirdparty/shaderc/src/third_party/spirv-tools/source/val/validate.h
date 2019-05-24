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

#ifndef SOURCE_VAL_VALIDATE_H_
#define SOURCE_VAL_VALIDATE_H_

#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "source/instruction.h"
#include "source/table.h"
#include "spirv-tools/libspirv.h"

namespace spvtools {
namespace val {

class ValidationState_t;
class BasicBlock;
class Instruction;

/// A function that returns a vector of BasicBlocks given a BasicBlock. Used to
/// get the successor and predecessor nodes of a CFG block
using get_blocks_func =
    std::function<const std::vector<BasicBlock*>*(const BasicBlock*)>;

/// @brief Performs the Control Flow Graph checks
///
/// @param[in] _ the validation state of the module
///
/// @return SPV_SUCCESS if no errors are found. SPV_ERROR_INVALID_CFG otherwise
spv_result_t PerformCfgChecks(ValidationState_t& _);

/// @brief Updates the use vectors of all instructions that can be referenced
///
/// This function will update the vector which define where an instruction was
/// referenced in the binary.
///
/// @param[in] _ the validation state of the module
///
/// @return SPV_SUCCESS if no errors are found.
spv_result_t UpdateIdUse(ValidationState_t& _, const Instruction* inst);

/// @brief This function checks all ID definitions dominate their use in the
/// CFG.
///
/// This function will iterate over all ID definitions that are defined in the
/// functions of a module and make sure that the definitions appear in a
/// block that dominates their use.
///
/// @param[in] _ the validation state of the module
///
/// @return SPV_SUCCESS if no errors are found. SPV_ERROR_INVALID_ID otherwise
spv_result_t CheckIdDefinitionDominateUse(ValidationState_t& _);

/// @brief This function checks for preconditions involving the adjacent
/// instructions.
///
/// This function will iterate over all instructions and check for any required
/// predecessor and/or successor instructions. e.g. SpvOpPhi must only be
/// preceeded by SpvOpLabel, SpvOpPhi, or SpvOpLine.
///
/// @param[in] _ the validation state of the module
///
/// @return SPV_SUCCESS if no errors are found. SPV_ERROR_INVALID_DATA otherwise
spv_result_t ValidateAdjacency(ValidationState_t& _);

/// @brief Validates static uses of input and output variables
///
/// Checks that any entry point that uses a input or output variable lists that
/// variable in its interface.
///
/// @param[in] _ the validation state of the module
///
/// @return SPV_SUCCESS if no errors are found.
spv_result_t ValidateInterfaces(ValidationState_t& _);

/// @brief Validates memory instructions
///
/// @param[in] _ the validation state of the module
/// @return SPV_SUCCESS if no errors are found.
spv_result_t MemoryPass(ValidationState_t& _, const Instruction* inst);

/// @brief Updates the immediate dominator for each of the block edges
///
/// Updates the immediate dominator of the blocks for each of the edges
/// provided by the @p dom_edges parameter
///
/// @param[in,out] dom_edges The edges of the dominator tree
/// @param[in] set_func This function will be called to updated the Immediate
///                     dominator
void UpdateImmediateDominators(
    const std::vector<std::pair<BasicBlock*, BasicBlock*>>& dom_edges,
    std::function<void(BasicBlock*, BasicBlock*)> set_func);

/// @brief Prints all of the dominators of a BasicBlock
///
/// @param[in] block The dominators of this block will be printed
void printDominatorList(BasicBlock& block);

/// Performs logical layout validation as described in section 2.4 of the SPIR-V
/// spec.
spv_result_t ModuleLayoutPass(ValidationState_t& _, const Instruction* inst);

/// Performs Control Flow Graph validation and construction.
spv_result_t CfgPass(ValidationState_t& _, const Instruction* inst);

/// Validates Control Flow Graph instructions.
spv_result_t ControlFlowPass(ValidationState_t& _, const Instruction* inst);

/// Performs Id and SSA validation of a module
spv_result_t IdPass(ValidationState_t& _, Instruction* inst);

/// Performs validation of the Data Rules subsection of 2.16.1 Universal
/// Validation Rules.
/// TODO(ehsann): add more comments here as more validation code is added.
spv_result_t DataRulesPass(ValidationState_t& _, const Instruction* inst);

/// Performs instruction validation.
spv_result_t InstructionPass(ValidationState_t& _, const Instruction* inst);

/// Performs decoration validation.  Assumes each decoration on a group
/// has been propagated down to the group members.
spv_result_t ValidateDecorations(ValidationState_t& _);

/// Performs validation of built-in variables.
spv_result_t ValidateBuiltIns(ValidationState_t& _);

/// Validates type instructions.
spv_result_t TypePass(ValidationState_t& _, const Instruction* inst);

/// Validates constant instructions.
spv_result_t ConstantPass(ValidationState_t& _, const Instruction* inst);

/// Validates correctness of arithmetic instructions.
spv_result_t ArithmeticsPass(ValidationState_t& _, const Instruction* inst);

/// Validates correctness of composite instructions.
spv_result_t CompositesPass(ValidationState_t& _, const Instruction* inst);

/// Validates correctness of conversion instructions.
spv_result_t ConversionPass(ValidationState_t& _, const Instruction* inst);

/// Validates correctness of derivative instructions.
spv_result_t DerivativesPass(ValidationState_t& _, const Instruction* inst);

/// Validates correctness of logical instructions.
spv_result_t LogicalsPass(ValidationState_t& _, const Instruction* inst);

/// Validates correctness of bitwise instructions.
spv_result_t BitwisePass(ValidationState_t& _, const Instruction* inst);

/// Validates correctness of image instructions.
spv_result_t ImagePass(ValidationState_t& _, const Instruction* inst);

/// Validates correctness of atomic instructions.
spv_result_t AtomicsPass(ValidationState_t& _, const Instruction* inst);

/// Validates correctness of barrier instructions.
spv_result_t BarriersPass(ValidationState_t& _, const Instruction* inst);

/// Validates correctness of literal numbers.
spv_result_t LiteralsPass(ValidationState_t& _, const Instruction* inst);

/// Validates correctness of extension instructions.
spv_result_t ExtensionPass(ValidationState_t& _, const Instruction* inst);

/// Validates correctness of annotation instructions.
spv_result_t AnnotationPass(ValidationState_t& _, const Instruction* inst);

/// Validates correctness of non-uniform group instructions.
spv_result_t NonUniformPass(ValidationState_t& _, const Instruction* inst);

/// Validates correctness of debug instructions.
spv_result_t DebugPass(ValidationState_t& _, const Instruction* inst);

// Validates that capability declarations use operands allowed in the current
// context.
spv_result_t CapabilityPass(ValidationState_t& _, const Instruction* inst);

/// Validates correctness of primitive instructions.
spv_result_t PrimitivesPass(ValidationState_t& _, const Instruction* inst);

/// Validates correctness of mode setting instructions.
spv_result_t ModeSettingPass(ValidationState_t& _, const Instruction* inst);

/// Validates correctness of function instructions.
spv_result_t FunctionPass(ValidationState_t& _, const Instruction* inst);

/// Validates execution limitations.
///
/// Verifies execution models are allowed for all functionality they contain.
spv_result_t ValidateExecutionLimitations(ValidationState_t& _,
                                          const Instruction* inst);

/// @brief Validate the ID's within a SPIR-V binary
///
/// @param[in] pInstructions array of instructions
/// @param[in] count number of elements in instruction array
/// @param[in] bound the binary header
/// @param[in,out] position current word in the binary
/// @param[in] consumer message consumer callback
///
/// @return result code
spv_result_t spvValidateIDs(const spv_instruction_t* pInstructions,
                            const uint64_t count, const uint32_t bound,
                            spv_position position,
                            const MessageConsumer& consumer);

// Performs validation for the SPIRV-V module binary.
// The main difference between this API and spvValidateBinary is that the
// "Validation State" is not destroyed upon function return; it lives on and is
// pointed to by the vstate unique_ptr.
spv_result_t ValidateBinaryAndKeepValidationState(
    const spv_const_context context, spv_const_validator_options options,
    const uint32_t* words, const size_t num_words, spv_diagnostic* pDiagnostic,
    std::unique_ptr<ValidationState_t>* vstate);

}  // namespace val
}  // namespace spvtools

#endif  // SOURCE_VAL_VALIDATE_H_

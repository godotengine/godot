// Copyright (c) 2018 The Khronos Group Inc.
// Copyright (c) 2018 Valve Corporation
// Copyright (c) 2018 LunarG Inc.
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

#ifndef LIBSPIRV_OPT_INSTRUMENT_PASS_H_
#define LIBSPIRV_OPT_INSTRUMENT_PASS_H_

#include <list>
#include <memory>
#include <vector>

#include "source/opt/ir_builder.h"
#include "source/opt/pass.h"
#include "spirv-tools/instrument.hpp"

// This is a base class to assist in the creation of passes which instrument
// shader modules. More specifically, passes which replace instructions with a
// larger and more capable set of instructions. Commonly, these new
// instructions will add testing of operands and execute different
// instructions depending on the outcome, including outputting of debug
// information into a buffer created especially for that purpose.
//
// This class contains helper functions to create an InstProcessFunction,
// which is the heart of any derived class implementing a specific
// instrumentation pass. It takes an instruction as an argument, decides
// if it should be instrumented, and generates code to replace it. This class
// also supplies function InstProcessEntryPointCallTree which applies the
// InstProcessFunction to every reachable instruction in a module and replaces
// the instruction with new instructions if generated.
//
// Chief among the helper functions are output code generation functions,
// used to generate code in the shader which writes data to output buffers
// associated with that validation. Currently one such function,
// GenDebugStreamWrite, exists. Other such functions may be added in the
// future. Each is accompanied by documentation describing the format of
// its output buffer.
//
// A validation pass may read or write multiple buffers. All such buffers
// are located in a single debug descriptor set whose index is passed at the
// creation of the instrumentation pass. The bindings of the buffers used by
// a validation pass are permanantly assigned and fixed and documented by
// the kDebugOutput* static consts.

namespace spvtools {
namespace opt {

// Validation Ids
// These are used to identify the general validation being done and map to
// its output buffers.
static const uint32_t kInstValidationIdBindless = 0;

class InstrumentPass : public Pass {
  using cbb_ptr = const BasicBlock*;

 public:
  using InstProcessFunction =
      std::function<void(BasicBlock::iterator, UptrVectorIterator<BasicBlock>,
                         uint32_t, std::vector<std::unique_ptr<BasicBlock>>*)>;

  ~InstrumentPass() override = default;

  IRContext::Analysis GetPreservedAnalyses() override {
    return IRContext::kAnalysisDefUse | IRContext::kAnalysisDecorations |
           IRContext::kAnalysisCombinators | IRContext::kAnalysisNameMap |
           IRContext::kAnalysisBuiltinVarId | IRContext::kAnalysisConstants;
  }

 protected:
  // Create instrumentation pass which utilizes descriptor set |desc_set|
  // for debug input and output buffers and writes |shader_id| into debug
  // output records.
  InstrumentPass(uint32_t desc_set, uint32_t shader_id, uint32_t validation_id)
      : Pass(),
        desc_set_(desc_set),
        shader_id_(shader_id),
        validation_id_(validation_id) {}

  // Initialize state for instrumentation of module by |validation_id|.
  void InitializeInstrument();

  // Call |pfn| on all instructions in all functions in the call tree of the
  // entry points in |module|. If code is generated for an instruction, replace
  // the instruction's block with the new blocks that are generated. Continue
  // processing at the top of the last new block.
  bool InstProcessEntryPointCallTree(InstProcessFunction& pfn);

  // Move all code in |ref_block_itr| preceding the instruction |ref_inst_itr|
  // to be instrumented into block |new_blk_ptr|.
  void MovePreludeCode(BasicBlock::iterator ref_inst_itr,
                       UptrVectorIterator<BasicBlock> ref_block_itr,
                       std::unique_ptr<BasicBlock>* new_blk_ptr);

  // Move all code in |ref_block_itr| succeeding the instruction |ref_inst_itr|
  // to be instrumented into block |new_blk_ptr|.
  void MovePostludeCode(UptrVectorIterator<BasicBlock> ref_block_itr,
                        BasicBlock* new_blk_ptr);

  // Generate instructions in |builder| which will atomically fetch and
  // increment the size of the debug output buffer stream of the current
  // validation and write a record to the end of the stream, if enough space
  // in the buffer remains. The record will contain the index of the function
  // and instruction within that function |func_idx, instruction_idx| which
  // generated the record. It will also contain additional information to
  // identify the instance of the shader, depending on the stage |stage_idx|
  // of the shader. Finally, the record will contain validation-specific
  // data contained in |validation_ids| which will identify the validation
  // error as well as the values involved in the error.
  //
  // The output buffer binding written to by the code generated by the function
  // is determined by the validation id specified when each specific
  // instrumentation pass is created.
  //
  // The output buffer is a sequence of 32-bit values with the following
  // format (where all elements are unsigned 32-bit unless otherwise noted):
  //
  //     Size
  //     Record0
  //     Record1
  //     Record2
  //     ...
  //
  // Size is the number of 32-bit values that have been written or
  // attempted to be written to the output buffer, excluding the Size. It is
  // initialized to 0. If the size of attempts to write the buffer exceeds
  // the actual size of the buffer, it is possible that this field can exceed
  // the actual size of the buffer.
  //
  // Each Record* is a variable-length sequence of 32-bit values with the
  // following format defined using static const offsets in the .cpp file:
  //
  //     Record Size
  //     Shader ID
  //     Instruction Index
  //     Stage
  //     Stage-specific Word 0
  //     Stage-specific Word 1
  //     Validation Error Code
  //     Validation-specific Word 0
  //     Validation-specific Word 1
  //     Validation-specific Word 2
  //     ...
  //
  // Each record consists of three subsections: members common across all
  // validation, members specific to the stage, and members specific to a
  // validation.
  //
  // The Record Size is the number of 32-bit words in the record, including
  // the Record Size word.
  //
  // Shader ID is a value that identifies which shader has generated the
  // validation error. It is passed when the instrumentation pass is created.
  //
  // The Instruction Index is the position of the instruction within the
  // SPIR-V file which is in error.
  //
  // The Stage is the pipeline stage which has generated the error as defined
  // by the SpvExecutionModel_ enumeration. This is used to interpret the
  // following Stage-specific words.
  //
  // The Stage-specific Words identify which invocation of the shader generated
  // the error. Every stage will write two words, although in some cases the
  // second word is unused and so zero is written. Vertex shaders will write
  // the Vertex and Instance ID. Fragment shaders will write FragCoord.xy.
  // Compute shaders will write the Global Invocation ID and zero (unused).
  // Both tesselation shaders will write the Invocation Id and zero (unused).
  // The geometry shader will write the Primitive ID and Invocation ID.
  //
  // The Validation Error Code specifies the exact error which has occurred.
  // These are enumerated with the kInstError* static consts. This allows
  // multiple validation layers to use the same, single output buffer.
  //
  // The Validation-specific Words are a validation-specific number of 32-bit
  // words which give further information on the validation error that
  // occurred. These are documented further in each file containing the
  // validation-specific class which derives from this base class.
  //
  // Because the code that is generated checks against the size of the buffer
  // before writing, the size of the debug out buffer can be used by the
  // validation layer to control the number of error records that are written.
  void GenDebugStreamWrite(uint32_t instruction_idx, uint32_t stage_idx,
                           const std::vector<uint32_t>& validation_ids,
                           InstructionBuilder* builder);

  // Generate in |builder| instructions to read the unsigned integer from the
  // input buffer specified by the offsets in |offset_ids|. Given offsets
  // o0, o1, ... oN, and input buffer ibuf, return the id for the value:
  //
  // ibuf[...ibuf[ibuf[o0]+o1]...+oN]
  //
  // The binding and the format of the input buffer is determined by each
  // specific validation, which is specified at the creation of the pass.
  uint32_t GenDebugDirectRead(const std::vector<uint32_t>& offset_ids,
                              InstructionBuilder* builder);

  // Generate code to cast |value_id| to unsigned, if needed. Return
  // an id to the unsigned equivalent.
  uint32_t GenUintCastCode(uint32_t value_id, InstructionBuilder* builder);

  // Return new label.
  std::unique_ptr<Instruction> NewLabel(uint32_t label_id);

  // Return id for 32-bit unsigned type
  uint32_t GetUintId();

  // Return id for 32-bit unsigned type
  uint32_t GetBoolId();

  // Return id for void type
  uint32_t GetVoidId();

  // Return pointer to type for runtime array of uint
  analysis::Type* GetUintRuntimeArrayType(analysis::DecorationManager* deco_mgr,
                                          analysis::TypeManager* type_mgr);

  // Return id for buffer uint type
  uint32_t GetBufferUintPtrId();

  // Return binding for output buffer for current validation.
  uint32_t GetOutputBufferBinding();

  // Return binding for input buffer for current validation.
  uint32_t GetInputBufferBinding();

  // Add storage buffer extension if needed
  void AddStorageBufferExt();

  // Return id for debug output buffer
  uint32_t GetOutputBufferId();

  // Return id for debug input buffer
  uint32_t GetInputBufferId();

  // Return id for v4float type
  uint32_t GetVec4FloatId();

  // Return id for v4uint type
  uint32_t GetVec4UintId();

  // Return id for output function. Define if it doesn't exist with
  // |val_spec_param_cnt| validation-specific uint32 parameters.
  uint32_t GetStreamWriteFunctionId(uint32_t stage_idx,
                                    uint32_t val_spec_param_cnt);

  // Return id for input function taking |param_cnt| uint32 parameters. Define
  // if it doesn't exist.
  uint32_t GetDirectReadFunctionId(uint32_t param_cnt);

  // Apply instrumentation function |pfn| to every instruction in |func|.
  // If code is generated for an instruction, replace the instruction's
  // block with the new blocks that are generated. Continue processing at the
  // top of the last new block.
  bool InstrumentFunction(Function* func, uint32_t stage_idx,
                          InstProcessFunction& pfn);

  // Call |pfn| on all functions in the call tree of the function
  // ids in |roots|.
  bool InstProcessCallTreeFromRoots(InstProcessFunction& pfn,
                                    std::queue<uint32_t>* roots,
                                    uint32_t stage_idx);

  // Gen code into |builder| to write |field_value_id| into debug output
  // buffer at |base_offset_id| + |field_offset|.
  void GenDebugOutputFieldCode(uint32_t base_offset_id, uint32_t field_offset,
                               uint32_t field_value_id,
                               InstructionBuilder* builder);

  // Generate instructions into |builder| which will write the members
  // of the debug output record common for all stages and validations at
  // |base_off|.
  void GenCommonStreamWriteCode(uint32_t record_sz, uint32_t instruction_idx,
                                uint32_t stage_idx, uint32_t base_off,
                                InstructionBuilder* builder);

  // Generate instructions into |builder| which will write
  // |uint_frag_coord_id| at |component| of the record at |base_offset_id| of
  // the debug output buffer .
  void GenFragCoordEltDebugOutputCode(uint32_t base_offset_id,
                                      uint32_t uint_frag_coord_id,
                                      uint32_t component,
                                      InstructionBuilder* builder);

  // Generate instructions into |builder| which will load the uint |builtin_id|
  // and write it into the debug output buffer at |base_off| + |builtin_off|.
  void GenBuiltinOutputCode(uint32_t builtin_id, uint32_t builtin_off,
                            uint32_t base_off, InstructionBuilder* builder);

  // Generate instructions into |builder| which will write a uint null into
  // the debug output buffer at |base_off| + |builtin_off|.
  void GenUintNullOutputCode(uint32_t field_off, uint32_t base_off,
                             InstructionBuilder* builder);

  // Generate instructions into |builder| which will write the |stage_idx|-
  // specific members of the debug output stream at |base_off|.
  void GenStageStreamWriteCode(uint32_t stage_idx, uint32_t base_off,
                               InstructionBuilder* builder);

  // Return true if instruction must be in the same block that its result
  // is used.
  bool IsSameBlockOp(const Instruction* inst) const;

  // Clone operands which must be in same block as consumer instructions.
  // Look in same_blk_pre for instructions that need cloning. Look in
  // same_blk_post for instructions already cloned. Add cloned instruction
  // to same_blk_post.
  void CloneSameBlockOps(
      std::unique_ptr<Instruction>* inst,
      std::unordered_map<uint32_t, uint32_t>* same_blk_post,
      std::unordered_map<uint32_t, Instruction*>* same_blk_pre,
      BasicBlock* block_ptr);

  // Update phis in succeeding blocks to point to new last block
  void UpdateSucceedingPhis(
      std::vector<std::unique_ptr<BasicBlock>>& new_blocks);

  // Debug descriptor set index
  uint32_t desc_set_;

  // Shader module ID written into output record
  uint32_t shader_id_;

  // Map from function id to function pointer.
  std::unordered_map<uint32_t, Function*> id2function_;

  // Map from block's label id to block. TODO(dnovillo): This is superfluous wrt
  // CFG. It has functionality not present in CFG. Consolidate.
  std::unordered_map<uint32_t, BasicBlock*> id2block_;

  // Map from instruction's unique id to offset in original file.
  std::unordered_map<uint32_t, uint32_t> uid2offset_;

  // result id for OpConstantFalse
  uint32_t validation_id_;

  // id for output buffer variable
  uint32_t output_buffer_id_;

  // type id for output buffer element
  uint32_t buffer_uint_ptr_id_;

  // id for debug output function
  uint32_t output_func_id_;

  // ids for debug input functions
  std::unordered_map<uint32_t, uint32_t> param2input_func_id_;

  // param count for output function
  uint32_t output_func_param_cnt_;

  // id for input buffer variable
  uint32_t input_buffer_id_;

  // id for v4float type
  uint32_t v4float_id_;

  // id for v4float type
  uint32_t v4uint_id_;

  // id for 32-bit unsigned type
  uint32_t uint_id_;

  // id for bool type
  uint32_t bool_id_;

  // id for void type
  uint32_t void_id_;

  // boolean to remember storage buffer extension
  bool storage_buffer_ext_defined_;

  // runtime array of uint type
  analysis::Type* uint_rarr_ty_;

  // Pre-instrumentation same-block insts
  std::unordered_map<uint32_t, Instruction*> same_block_pre_;

  // Post-instrumentation same-block op ids
  std::unordered_map<uint32_t, uint32_t> same_block_post_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_INSTRUMENT_PASS_H_

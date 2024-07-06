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
#include "include/spirv-tools/instrument.hpp"

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
// a validation pass are permanently assigned and fixed and documented by
// the kDebugOutput* static consts.

namespace spvtools {
namespace opt {

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
  // Create instrumentation pass for |validation_id| which utilizes descriptor
  // set |desc_set| for debug input and output buffers and writes |shader_id|
  // into debug output records. |opt_direct_reads| indicates that the pass
  // will see direct input buffer reads and should prepare to optimize them.
  InstrumentPass(uint32_t desc_set, uint32_t shader_id, bool opt_direct_reads,
                 bool use_stage_info)
      : Pass(),
        desc_set_(desc_set),
        shader_id_(shader_id),
        opt_direct_reads_(opt_direct_reads),
        use_stage_info_(use_stage_info) {}

  // Initialize state for instrumentation of module.
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

  // Return true if all instructions in |ids| are constants or spec constants.
  bool AllConstant(const std::vector<uint32_t>& ids);

  uint32_t GenReadFunctionCall(uint32_t return_id, uint32_t func_id,
                               const std::vector<uint32_t>& args,
                               InstructionBuilder* builder);

  // Generate code to convert integer |value_id| to 32bit, if needed. Return
  // an id to the 32bit equivalent.
  uint32_t Gen32BitCvtCode(uint32_t value_id, InstructionBuilder* builder);

  // Generate code to cast integer |value_id| to 32bit unsigned, if needed.
  // Return an id to the Uint equivalent.
  uint32_t GenUintCastCode(uint32_t value_id, InstructionBuilder* builder);

  std::unique_ptr<Function> StartFunction(
      uint32_t func_id, const analysis::Type* return_type,
      const std::vector<const analysis::Type*>& param_types);

  std::vector<uint32_t> AddParameters(
      Function& func, const std::vector<const analysis::Type*>& param_types);

  std::unique_ptr<Instruction> EndFunction();

  // Return new label.
  std::unique_ptr<Instruction> NewLabel(uint32_t label_id);

  // Set the name function parameter or local variable
  std::unique_ptr<Instruction> NewName(uint32_t id,
                                       const std::string& name_str);

  // Return id for 32-bit unsigned type
  uint32_t GetUintId();

  // Return id for 64-bit unsigned type
  uint32_t GetUint64Id();

  // Return id for 8-bit unsigned type
  uint32_t GetUint8Id();

  // Return id for 32-bit unsigned type
  uint32_t GetBoolId();

  // Return id for void type
  uint32_t GetVoidId();

  // Get registered type structures
  analysis::Integer* GetInteger(uint32_t width, bool is_signed);
  analysis::Struct* GetStruct(const std::vector<const analysis::Type*>& fields);
  analysis::RuntimeArray* GetRuntimeArray(const analysis::Type* element);
  analysis::Array* GetArray(const analysis::Type* element, uint32_t size);
  analysis::Function* GetFunction(
      const analysis::Type* return_val,
      const std::vector<const analysis::Type*>& args);

  // Return pointer to type for runtime array of uint
  analysis::RuntimeArray* GetUintXRuntimeArrayType(
      uint32_t width, analysis::RuntimeArray** rarr_ty);

  // Return pointer to type for runtime array of uint
  analysis::RuntimeArray* GetUintRuntimeArrayType(uint32_t width);

  // Add storage buffer extension if needed
  void AddStorageBufferExt();

  // Return id for 32-bit float type
  uint32_t GetFloatId();

  // Return id for v4float type
  uint32_t GetVec4FloatId();

  // Return id for uint vector type of |length|
  uint32_t GetVecUintId(uint32_t length);

  // Return id for v4uint type
  uint32_t GetVec4UintId();

  // Return id for v3uint type
  uint32_t GetVec3UintId();

  // Split block |block_itr| into two new blocks where the second block
  // contains |inst_itr| and place in |new_blocks|.
  void SplitBlock(BasicBlock::iterator inst_itr,
                  UptrVectorIterator<BasicBlock> block_itr,
                  std::vector<std::unique_ptr<BasicBlock>>* new_blocks);

  // Apply instrumentation function |pfn| to every instruction in |func|.
  // If code is generated for an instruction, replace the instruction's
  // block with the new blocks that are generated. Continue processing at the
  // top of the last new block.
  virtual bool InstrumentFunction(Function* func, uint32_t stage_idx,
                                  InstProcessFunction& pfn);

  // Call |pfn| on all functions in the call tree of the function
  // ids in |roots|.
  bool InstProcessCallTreeFromRoots(InstProcessFunction& pfn,
                                    std::queue<uint32_t>* roots,
                                    uint32_t stage_idx);

  // Generate instructions into |builder| which will load |var_id| and return
  // its result id.
  uint32_t GenVarLoad(uint32_t var_id, InstructionBuilder* builder);

  uint32_t GenStageInfo(uint32_t stage_idx, InstructionBuilder* builder);

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

  // id for debug output function
  std::unordered_map<uint32_t, uint32_t> param2output_func_id_;

  // ids for debug input functions
  std::unordered_map<uint32_t, uint32_t> param2input_func_id_;

  // id for 32-bit float type
  uint32_t float_id_{0};

  // id for v4float type
  uint32_t v4float_id_{0};

  // id for v4uint type
  uint32_t v4uint_id_{0};

  // id for v3uint type
  uint32_t v3uint_id_{0};

  // id for 32-bit unsigned type
  uint32_t uint_id_{0};

  // id for 64-bit unsigned type
  uint32_t uint64_id_{0};

  // id for 8-bit unsigned type
  uint32_t uint8_id_{0};

  // id for bool type
  uint32_t bool_id_{0};

  // id for void type
  uint32_t void_id_{0};

  // boolean to remember storage buffer extension
  bool storage_buffer_ext_defined_{false};

  // runtime array of uint type
  analysis::RuntimeArray* uint64_rarr_ty_{nullptr};

  // runtime array of uint type
  analysis::RuntimeArray* uint32_rarr_ty_{nullptr};

  // Pre-instrumentation same-block insts
  std::unordered_map<uint32_t, Instruction*> same_block_pre_;

  // Post-instrumentation same-block op ids
  std::unordered_map<uint32_t, uint32_t> same_block_post_;

  // Map function calls to result id. Clear for every function.
  // This is for debug input reads with constant arguments that
  // have been generated into the first block of the function.
  // This mechanism is used to avoid multiple identical debug
  // input buffer reads.
  struct vector_hash_ {
    std::size_t operator()(const std::vector<uint32_t>& v) const {
      std::size_t hash = v.size();
      for (auto& u : v) {
        hash ^= u + 0x9e3779b9 + (hash << 11) + (hash >> 21);
      }
      return hash;
    }
  };
  std::unordered_map<std::vector<uint32_t>, uint32_t, vector_hash_> call2id_;

  // Function currently being instrumented
  Function* curr_func_{nullptr};

  // Optimize direct debug input buffer reads. Specifically, move all such
  // reads with constant args to first block and reuse them.
  const bool opt_direct_reads_;

  // Set true if the instrumentation needs to know the current stage.
  // Note that this does not work with multi-stage modules.
  const bool use_stage_info_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_INSTRUMENT_PASS_H_

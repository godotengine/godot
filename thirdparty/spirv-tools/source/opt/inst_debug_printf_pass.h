// Copyright (c) 2020 The Khronos Group Inc.
// Copyright (c) 2020 Valve Corporation
// Copyright (c) 2020 LunarG Inc.
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

#ifndef LIBSPIRV_OPT_INST_DEBUG_PRINTF_PASS_H_
#define LIBSPIRV_OPT_INST_DEBUG_PRINTF_PASS_H_

#include "instrument_pass.h"

namespace spvtools {
namespace opt {

// This class/pass is designed to support the debug printf GPU-assisted layer
// of https://github.com/KhronosGroup/Vulkan-ValidationLayers. Its internal and
// external design may change as the layer evolves.
class InstDebugPrintfPass : public InstrumentPass {
 public:
  // For test harness only
  InstDebugPrintfPass() : InstrumentPass(7, 23, kInstValidationIdDebugPrintf) {}
  // For all other interfaces
  InstDebugPrintfPass(uint32_t desc_set, uint32_t shader_id)
      : InstrumentPass(desc_set, shader_id, kInstValidationIdDebugPrintf) {}

  ~InstDebugPrintfPass() override = default;

  // See optimizer.hpp for pass user documentation.
  Status Process() override;

  const char* name() const override { return "inst-printf-pass"; }

 private:
  // Generate instructions for OpDebugPrintf.
  //
  // If |ref_inst_itr| is an OpDebugPrintf, return in |new_blocks| the result
  // of replacing it with buffer write instructions within its block at
  // |ref_block_itr|.  The instructions write a record to the printf
  // output buffer stream including |function_idx, instruction_idx, stage_idx|
  // and removes the OpDebugPrintf. The block at |ref_block_itr| can just be
  // replaced with the block in |new_blocks|. Besides the buffer writes, this
  // block will comprise all instructions preceding and following
  // |ref_inst_itr|.
  //
  // This function is designed to be passed to
  // InstrumentPass::InstProcessEntryPointCallTree(), which applies the
  // function to each instruction in a module and replaces the instruction
  // if warranted.
  //
  // This instrumentation function utilizes GenDebugStreamWrite() to write its
  // error records. The validation-specific part of the error record will
  // consist of a uint32 which is the id of the format string plus a sequence
  // of uint32s representing the values of the remaining operands of the
  // DebugPrintf.
  void GenDebugPrintfCode(BasicBlock::iterator ref_inst_itr,
                          UptrVectorIterator<BasicBlock> ref_block_itr,
                          uint32_t stage_idx,
                          std::vector<std::unique_ptr<BasicBlock>>* new_blocks);

  // Generate a sequence of uint32 instructions in |builder| (if necessary)
  // representing the value of |val_inst|, which must be a buffer pointer, a
  // uint64, or a scalar or vector of type uint32, float32 or float16. Append
  // the ids of all values to the end of |val_ids|.
  void GenOutputValues(Instruction* val_inst, std::vector<uint32_t>* val_ids,
                       InstructionBuilder* builder);

  // Generate instructions to write a record containing the operands of
  // |printf_inst| arguments to printf buffer, adding new code to the end of
  // the last block in |new_blocks|. Kill OpDebugPrintf instruction.
  void GenOutputCode(Instruction* printf_inst, uint32_t stage_idx,
                     std::vector<std::unique_ptr<BasicBlock>>* new_blocks);

  // Initialize state for instrumenting bindless checking
  void InitializeInstDebugPrintf();

  // Apply GenDebugPrintfCode to every instruction in module.
  Pass::Status ProcessImpl();

  uint32_t ext_inst_printf_id_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_INST_DEBUG_PRINTF_PASS_H_

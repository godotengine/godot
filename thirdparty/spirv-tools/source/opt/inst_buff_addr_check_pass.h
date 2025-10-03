// Copyright (c) 2019 The Khronos Group Inc.
// Copyright (c) 2019 Valve Corporation
// Copyright (c) 2019 LunarG Inc.
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

#ifndef LIBSPIRV_OPT_INST_BUFFER_ADDRESS_PASS_H_
#define LIBSPIRV_OPT_INST_BUFFER_ADDRESS_PASS_H_

#include "instrument_pass.h"

namespace spvtools {
namespace opt {

// This class/pass is designed to support the GPU-assisted validation layer of
// the Buffer Device Address (BDA) extension in
// https://github.com/KhronosGroup/Vulkan-ValidationLayers. The internal and
// external design of this class may change as the layer evolves.
class InstBuffAddrCheckPass : public InstrumentPass {
 public:
  // For test harness only
  InstBuffAddrCheckPass() : InstrumentPass(7, 23, kInstValidationIdBuffAddr) {}
  // For all other interfaces
  InstBuffAddrCheckPass(uint32_t desc_set, uint32_t shader_id)
      : InstrumentPass(desc_set, shader_id, kInstValidationIdBuffAddr) {}

  ~InstBuffAddrCheckPass() override = default;

  // See optimizer.hpp for pass user documentation.
  Status Process() override;

  const char* name() const override { return "inst-buff-addr-check-pass"; }

 private:
  // Return byte alignment of type |type_id|. Must be int, float, vector,
  // matrix, struct, array or physical pointer. Uses std430 alignment.
  uint32_t GetTypeAlignment(uint32_t type_id);

  // Return byte length of type |type_id|. Must be int, float, vector, matrix,
  // struct, array or physical pointer. Uses std430 alignment and sizes.
  uint32_t GetTypeLength(uint32_t type_id);

  // Add |type_id| param to |input_func| and add id to |param_vec|.
  void AddParam(uint32_t type_id, std::vector<uint32_t>* param_vec,
                std::unique_ptr<Function>* input_func);

  // Return id for search and test function. Generate it if not already gen'd.
  uint32_t GetSearchAndTestFuncId();

  // Generate code into |builder| to do search of the BDA debug input buffer
  // for the buffer used by |ref_inst| and test that all bytes of reference
  // are within the buffer. Returns id of boolean value which is true if
  // search and test is successful, false otherwise.
  uint32_t GenSearchAndTest(Instruction* ref_inst, InstructionBuilder* builder,
                            uint32_t* ref_uptr_id);

  // This function does checking instrumentation on a single
  // instruction which references through a physical storage buffer address.
  // GenBuffAddrCheckCode generates code that checks that all bytes that
  // are referenced fall within a buffer that was queried via
  // the Vulkan API call vkGetBufferDeviceAddressEXT().
  //
  // The function is designed to be passed to
  // InstrumentPass::InstProcessEntryPointCallTree(), which applies the
  // function to each instruction in a module and replaces the instruction
  // with instrumented code if warranted.
  //
  // If |ref_inst_itr| is a physical storage buffer reference, return in
  // |new_blocks| the result of instrumenting it with validation code within
  // its block at |ref_block_itr|.  The validation code first executes a check
  // for the specific condition called for. If the check passes, it executes
  // the remainder of the reference, otherwise writes a record to the debug
  // output buffer stream including |function_idx, instruction_idx, stage_idx|
  // and replaces the reference with the null value of the original type. The
  // block at |ref_block_itr| can just be replaced with the blocks in
  // |new_blocks|, which will contain at least two blocks. The last block will
  // comprise all instructions following |ref_inst_itr|,
  // preceded by a phi instruction if needed.
  //
  // This instrumentation function utilizes GenDebugStreamWrite() to write its
  // error records. The validation-specific part of the error record will
  // have the format:
  //
  //    Validation Error Code (=kInstErrorBuffAddr)
  //    Buffer Address (lowest 32 bits)
  //    Buffer Address (highest 32 bits)
  //
  void GenBuffAddrCheckCode(
      BasicBlock::iterator ref_inst_itr,
      UptrVectorIterator<BasicBlock> ref_block_itr, uint32_t stage_idx,
      std::vector<std::unique_ptr<BasicBlock>>* new_blocks);

  // Return true if |ref_inst| is a physical buffer address reference, false
  // otherwise.
  bool IsPhysicalBuffAddrReference(Instruction* ref_inst);

  // Clone original reference |ref_inst| into |builder| and return id of result
  uint32_t CloneOriginalReference(Instruction* ref_inst,
                                  InstructionBuilder* builder);

  // Generate instrumentation code for boolean test result |check_id|,
  // adding new blocks to |new_blocks|. Generate conditional branch to valid
  // or invalid reference blocks. Generate valid reference block which does
  // original reference |ref_inst|. Then generate invalid reference block which
  // writes debug error output utilizing |ref_inst|, |error_id| and
  // |stage_idx|. Generate merge block for valid and invalid reference blocks.
  // Kill original reference.
  void GenCheckCode(uint32_t check_id, uint32_t error_id, uint32_t length_id,
                    uint32_t stage_idx, Instruction* ref_inst,
                    std::vector<std::unique_ptr<BasicBlock>>* new_blocks);

  // Initialize state for instrumenting physical buffer address checking
  void InitInstBuffAddrCheck();

  // Apply GenBuffAddrCheckCode to every instruction in module.
  Pass::Status ProcessImpl();

  // Id of search and test function, if already gen'd, else zero.
  uint32_t search_test_func_id_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_INST_BUFFER_ADDRESS_PASS_H_

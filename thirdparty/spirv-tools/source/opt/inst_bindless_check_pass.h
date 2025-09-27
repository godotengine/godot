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

#ifndef LIBSPIRV_OPT_INST_BINDLESS_CHECK_PASS_H_
#define LIBSPIRV_OPT_INST_BINDLESS_CHECK_PASS_H_

#include "instrument_pass.h"

namespace spvtools {
namespace opt {

// This class/pass is designed to support the bindless (descriptor indexing)
// GPU-assisted validation layer of
// https://github.com/KhronosGroup/Vulkan-ValidationLayers. Its internal and
// external design may change as the layer evolves.
class InstBindlessCheckPass : public InstrumentPass {
 public:
  InstBindlessCheckPass(uint32_t desc_set, uint32_t shader_id,
                        bool desc_idx_enable, bool desc_init_enable,
                        bool buffer_bounds_enable, bool texel_buffer_enable,
                        bool opt_direct_reads)
      : InstrumentPass(desc_set, shader_id, kInstValidationIdBindless,
                       opt_direct_reads),
        desc_idx_enabled_(desc_idx_enable),
        desc_init_enabled_(desc_init_enable),
        buffer_bounds_enabled_(buffer_bounds_enable),
        texel_buffer_enabled_(texel_buffer_enable) {}

  ~InstBindlessCheckPass() override = default;

  // See optimizer.hpp for pass user documentation.
  Status Process() override;

  const char* name() const override { return "inst-bindless-check-pass"; }

 private:
  // These functions do bindless checking instrumentation on a single
  // instruction which references through a descriptor (ie references into an
  // image or buffer). Refer to Vulkan API for further information on
  // descriptors. GenDescIdxCheckCode checks that an index into a descriptor
  // array (array of images or buffers) is in-bounds. GenDescInitCheckCode
  // checks that the referenced descriptor has been initialized, if the
  // SPV_EXT_descriptor_indexing extension is enabled, and initialized large
  // enough to handle the reference, if RobustBufferAccess is disabled.
  // GenDescInitCheckCode checks for uniform and storage buffer overrun.
  // GenTexBuffCheckCode checks for texel buffer overrun and should be
  // run after GenDescInitCheckCode to first make sure that the descriptor
  // is initialized because it uses OpImageQuerySize on the descriptor.
  //
  // The functions are designed to be passed to
  // InstrumentPass::InstProcessEntryPointCallTree(), which applies the
  // function to each instruction in a module and replaces the instruction
  // if warranted.
  //
  // If |ref_inst_itr| is a bindless reference, return in |new_blocks| the
  // result of instrumenting it with validation code within its block at
  // |ref_block_itr|.  The validation code first executes a check for the
  // specific condition called for. If the check passes, it executes
  // the remainder of the reference, otherwise writes a record to the debug
  // output buffer stream including |function_idx, instruction_idx, stage_idx|
  // and replaces the reference with the null value of the original type. The
  // block at |ref_block_itr| can just be replaced with the blocks in
  // |new_blocks|, which will contain at least two blocks. The last block will
  // comprise all instructions following |ref_inst_itr|,
  // preceded by a phi instruction.
  //
  // These instrumentation functions utilize GenDebugDirectRead() to read data
  // from the debug input buffer, specifically the lengths of variable length
  // descriptor arrays, and the initialization status of each descriptor.
  // The format of the debug input buffer is documented in instrument.hpp.
  //
  // These instrumentation functions utilize GenDebugStreamWrite() to write its
  // error records. The validation-specific part of the error record will
  // have the format:
  //
  //    Validation Error Code (=kInstErrorBindlessBounds)
  //    Descriptor Index
  //    Descriptor Array Size
  //
  // The Descriptor Index is the index which has been determined to be
  // out-of-bounds.
  //
  // The Descriptor Array Size is the size of the descriptor array which was
  // indexed.
  void GenDescIdxCheckCode(
      BasicBlock::iterator ref_inst_itr,
      UptrVectorIterator<BasicBlock> ref_block_itr, uint32_t stage_idx,
      std::vector<std::unique_ptr<BasicBlock>>* new_blocks);

  void GenDescInitCheckCode(
      BasicBlock::iterator ref_inst_itr,
      UptrVectorIterator<BasicBlock> ref_block_itr, uint32_t stage_idx,
      std::vector<std::unique_ptr<BasicBlock>>* new_blocks);

  void GenTexBuffCheckCode(
      BasicBlock::iterator ref_inst_itr,
      UptrVectorIterator<BasicBlock> ref_block_itr, uint32_t stage_idx,
      std::vector<std::unique_ptr<BasicBlock>>* new_blocks);

  // Generate instructions into |builder| to read length of runtime descriptor
  // array |var_id| from debug input buffer and return id of value.
  uint32_t GenDebugReadLength(uint32_t var_id, InstructionBuilder* builder);

  // Generate instructions into |builder| to read initialization status of
  // descriptor array |image_id| at |index_id| from debug input buffer and
  // return id of value.
  uint32_t GenDebugReadInit(uint32_t image_id, uint32_t index_id,
                            InstructionBuilder* builder);

  // Analysis data for descriptor reference components, generated by
  // AnalyzeDescriptorReference. It is necessary and sufficient for further
  // analysis and regeneration of the reference.
  typedef struct RefAnalysis {
    uint32_t desc_load_id;
    uint32_t image_id;
    uint32_t load_id;
    uint32_t ptr_id;
    uint32_t var_id;
    uint32_t desc_idx_id;
    uint32_t strg_class;
    Instruction* ref_inst;
  } RefAnalysis;

  // Return size of type |ty_id| in bytes. Use |matrix_stride| and |col_major|
  // for matrix type, or for vector type if vector is |in_matrix|.
  uint32_t ByteSize(uint32_t ty_id, uint32_t matrix_stride, bool col_major,
                    bool in_matrix);

  // Return stride of type |ty_id| with decoration |stride_deco|. Return 0
  // if not found
  uint32_t FindStride(uint32_t ty_id, uint32_t stride_deco);

  // Generate index of last byte referenced by buffer reference |ref|
  uint32_t GenLastByteIdx(RefAnalysis* ref, InstructionBuilder* builder);

  // Clone original image computation starting at |image_id| into |builder|.
  // This may generate more than one instruction if necessary.
  uint32_t CloneOriginalImage(uint32_t image_id, InstructionBuilder* builder);

  // Clone original original reference encapsulated by |ref| into |builder|.
  // This may generate more than one instruction if necessary.
  uint32_t CloneOriginalReference(RefAnalysis* ref,
                                  InstructionBuilder* builder);

  // If |inst| references through an image, return the id of the image it
  // references through. Else return 0.
  uint32_t GetImageId(Instruction* inst);

  // Get pointee type inst of pointer value |ptr_inst|.
  Instruction* GetPointeeTypeInst(Instruction* ptr_inst);

  // Analyze descriptor reference |ref_inst| and save components into |ref|.
  // Return true if |ref_inst| is a descriptor reference, false otherwise.
  bool AnalyzeDescriptorReference(Instruction* ref_inst, RefAnalysis* ref);

  // Generate instrumentation code for generic test result |check_id|, starting
  // with |builder| of block |new_blk_ptr|, adding new blocks to |new_blocks|.
  // Generate conditional branch to a valid or invalid branch. Generate valid
  // block which does original reference |ref|. Generate invalid block which
  // writes debug error output utilizing |ref|, |error_id|, |length_id| and
  // |stage_idx|. Generate merge block for valid and invalid branches. Kill
  // original reference.
  void GenCheckCode(uint32_t check_id, uint32_t error_id, uint32_t offset_id,
                    uint32_t length_id, uint32_t stage_idx, RefAnalysis* ref,
                    std::vector<std::unique_ptr<BasicBlock>>* new_blocks);

  // Initialize state for instrumenting bindless checking
  void InitializeInstBindlessCheck();

  // Apply GenDescIdxCheckCode to every instruction in module. Then apply
  // GenDescInitCheckCode to every instruction in module.
  Pass::Status ProcessImpl();

  // Enable instrumentation of runtime array length checking
  bool desc_idx_enabled_;

  // Enable instrumentation of descriptor initialization checking
  bool desc_init_enabled_;

  // Enable instrumentation of uniform and storage buffer overrun checking
  bool buffer_bounds_enabled_;

  // Enable instrumentation of texel buffer overrun checking
  bool texel_buffer_enabled_;

  // Mapping from variable to descriptor set
  std::unordered_map<uint32_t, uint32_t> var2desc_set_;

  // Mapping from variable to binding
  std::unordered_map<uint32_t, uint32_t> var2binding_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_INST_BINDLESS_CHECK_PASS_H_

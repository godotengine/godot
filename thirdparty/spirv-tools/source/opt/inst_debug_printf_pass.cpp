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

#include "inst_debug_printf_pass.h"

#include "source/util/string_utils.h"
#include "spirv/unified1/NonSemanticDebugPrintf.h"

namespace spvtools {
namespace opt {

void InstDebugPrintfPass::GenOutputValues(Instruction* val_inst,
                                          std::vector<uint32_t>* val_ids,
                                          InstructionBuilder* builder) {
  uint32_t val_ty_id = val_inst->type_id();
  analysis::TypeManager* type_mgr = context()->get_type_mgr();
  analysis::Type* val_ty = type_mgr->GetType(val_ty_id);
  switch (val_ty->kind()) {
    case analysis::Type::kVector: {
      analysis::Vector* v_ty = val_ty->AsVector();
      const analysis::Type* c_ty = v_ty->element_type();
      uint32_t c_ty_id = type_mgr->GetId(c_ty);
      for (uint32_t c = 0; c < v_ty->element_count(); ++c) {
        Instruction* c_inst =
            builder->AddCompositeExtract(c_ty_id, val_inst->result_id(), {c});
        GenOutputValues(c_inst, val_ids, builder);
      }
      return;
    }
    case analysis::Type::kBool: {
      // Select between uint32 zero or one
      uint32_t zero_id = builder->GetUintConstantId(0);
      uint32_t one_id = builder->GetUintConstantId(1);
      Instruction* sel_inst = builder->AddSelect(
          GetUintId(), val_inst->result_id(), one_id, zero_id);
      val_ids->push_back(sel_inst->result_id());
      return;
    }
    case analysis::Type::kFloat: {
      analysis::Float* f_ty = val_ty->AsFloat();
      switch (f_ty->width()) {
        case 16: {
          // Convert float16 to float32 and recurse
          Instruction* f32_inst = builder->AddUnaryOp(
              GetFloatId(), spv::Op::OpFConvert, val_inst->result_id());
          GenOutputValues(f32_inst, val_ids, builder);
          return;
        }
        case 64: {
          // Bitcast float64 to uint64 and recurse
          Instruction* ui64_inst = builder->AddUnaryOp(
              GetUint64Id(), spv::Op::OpBitcast, val_inst->result_id());
          GenOutputValues(ui64_inst, val_ids, builder);
          return;
        }
        case 32: {
          // Bitcase float32 to uint32
          Instruction* bc_inst = builder->AddUnaryOp(
              GetUintId(), spv::Op::OpBitcast, val_inst->result_id());
          val_ids->push_back(bc_inst->result_id());
          return;
        }
        default:
          assert(false && "unsupported float width");
          return;
      }
    }
    case analysis::Type::kInteger: {
      analysis::Integer* i_ty = val_ty->AsInteger();
      switch (i_ty->width()) {
        case 64: {
          Instruction* ui64_inst = val_inst;
          if (i_ty->IsSigned()) {
            // Bitcast sint64 to uint64
            ui64_inst = builder->AddUnaryOp(GetUint64Id(), spv::Op::OpBitcast,
                                            val_inst->result_id());
          }
          // Break uint64 into 2x uint32
          Instruction* lo_ui64_inst = builder->AddUnaryOp(
              GetUintId(), spv::Op::OpUConvert, ui64_inst->result_id());
          Instruction* rshift_ui64_inst = builder->AddBinaryOp(
              GetUint64Id(), spv::Op::OpShiftRightLogical,
              ui64_inst->result_id(), builder->GetUintConstantId(32));
          Instruction* hi_ui64_inst = builder->AddUnaryOp(
              GetUintId(), spv::Op::OpUConvert, rshift_ui64_inst->result_id());
          val_ids->push_back(lo_ui64_inst->result_id());
          val_ids->push_back(hi_ui64_inst->result_id());
          return;
        }
        case 8: {
          Instruction* ui8_inst = val_inst;
          if (i_ty->IsSigned()) {
            // Bitcast sint8 to uint8
            ui8_inst = builder->AddUnaryOp(GetUint8Id(), spv::Op::OpBitcast,
                                           val_inst->result_id());
          }
          // Convert uint8 to uint32
          Instruction* ui32_inst = builder->AddUnaryOp(
              GetUintId(), spv::Op::OpUConvert, ui8_inst->result_id());
          val_ids->push_back(ui32_inst->result_id());
          return;
        }
        case 32: {
          Instruction* ui32_inst = val_inst;
          if (i_ty->IsSigned()) {
            // Bitcast sint32 to uint32
            ui32_inst = builder->AddUnaryOp(GetUintId(), spv::Op::OpBitcast,
                                            val_inst->result_id());
          }
          // uint32 needs no further processing
          val_ids->push_back(ui32_inst->result_id());
          return;
        }
        default:
          // TODO(greg-lunarg): Support non-32-bit int
          assert(false && "unsupported int width");
          return;
      }
    }
    default:
      assert(false && "unsupported type");
      return;
  }
}

void InstDebugPrintfPass::GenOutputCode(
    Instruction* printf_inst, uint32_t stage_idx,
    std::vector<std::unique_ptr<BasicBlock>>* new_blocks) {
  BasicBlock* back_blk_ptr = &*new_blocks->back();
  InstructionBuilder builder(
      context(), back_blk_ptr,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  // Gen debug printf record validation-specific values. The format string
  // will have its id written. Vectors will need to be broken down into
  // component values. float16 will need to be converted to float32. Pointer
  // and uint64 will need to be converted to two uint32 values. float32 will
  // need to be bitcast to uint32. int32 will need to be bitcast to uint32.
  std::vector<uint32_t> val_ids;
  bool is_first_operand = false;
  printf_inst->ForEachInId(
      [&is_first_operand, &val_ids, &builder, this](const uint32_t* iid) {
        // skip set operand
        if (!is_first_operand) {
          is_first_operand = true;
          return;
        }
        Instruction* opnd_inst = get_def_use_mgr()->GetDef(*iid);
        if (opnd_inst->opcode() == spv::Op::OpString) {
          uint32_t string_id_id = builder.GetUintConstantId(*iid);
          val_ids.push_back(string_id_id);
        } else {
          GenOutputValues(opnd_inst, &val_ids, &builder);
        }
      });
  GenDebugStreamWrite(uid2offset_[printf_inst->unique_id()], stage_idx, val_ids,
                      &builder);
  context()->KillInst(printf_inst);
}

void InstDebugPrintfPass::GenDebugPrintfCode(
    BasicBlock::iterator ref_inst_itr,
    UptrVectorIterator<BasicBlock> ref_block_itr, uint32_t stage_idx,
    std::vector<std::unique_ptr<BasicBlock>>* new_blocks) {
  // If not DebugPrintf OpExtInst, return.
  Instruction* printf_inst = &*ref_inst_itr;
  if (printf_inst->opcode() != spv::Op::OpExtInst) return;
  if (printf_inst->GetSingleWordInOperand(0) != ext_inst_printf_id_) return;
  if (printf_inst->GetSingleWordInOperand(1) !=
      NonSemanticDebugPrintfDebugPrintf)
    return;
  // Initialize DefUse manager before dismantling module
  (void)get_def_use_mgr();
  // Move original block's preceding instructions into first new block
  std::unique_ptr<BasicBlock> new_blk_ptr;
  MovePreludeCode(ref_inst_itr, ref_block_itr, &new_blk_ptr);
  new_blocks->push_back(std::move(new_blk_ptr));
  // Generate instructions to output printf args to printf buffer
  GenOutputCode(printf_inst, stage_idx, new_blocks);
  // Caller expects at least two blocks with last block containing remaining
  // code, so end block after instrumentation, create remainder block, and
  // branch to it
  uint32_t rem_blk_id = TakeNextId();
  std::unique_ptr<Instruction> rem_label(NewLabel(rem_blk_id));
  BasicBlock* back_blk_ptr = &*new_blocks->back();
  InstructionBuilder builder(
      context(), back_blk_ptr,
      IRContext::kAnalysisDefUse | IRContext::kAnalysisInstrToBlockMapping);
  (void)builder.AddBranch(rem_blk_id);
  // Gen remainder block
  new_blk_ptr.reset(new BasicBlock(std::move(rem_label)));
  builder.SetInsertPoint(&*new_blk_ptr);
  // Move original block's remaining code into remainder block and add
  // to new blocks
  MovePostludeCode(ref_block_itr, &*new_blk_ptr);
  new_blocks->push_back(std::move(new_blk_ptr));
}

void InstDebugPrintfPass::InitializeInstDebugPrintf() {
  // Initialize base class
  InitializeInstrument();
}

Pass::Status InstDebugPrintfPass::ProcessImpl() {
  // Perform printf instrumentation on each entry point function in module
  InstProcessFunction pfn =
      [this](BasicBlock::iterator ref_inst_itr,
             UptrVectorIterator<BasicBlock> ref_block_itr, uint32_t stage_idx,
             std::vector<std::unique_ptr<BasicBlock>>* new_blocks) {
        return GenDebugPrintfCode(ref_inst_itr, ref_block_itr, stage_idx,
                                  new_blocks);
      };
  (void)InstProcessEntryPointCallTree(pfn);
  // Remove DebugPrintf OpExtInstImport instruction
  Instruction* ext_inst_import_inst =
      get_def_use_mgr()->GetDef(ext_inst_printf_id_);
  context()->KillInst(ext_inst_import_inst);
  // If no remaining non-semantic instruction sets, remove non-semantic debug
  // info extension from module and feature manager
  bool non_sem_set_seen = false;
  for (auto c_itr = context()->module()->ext_inst_import_begin();
       c_itr != context()->module()->ext_inst_import_end(); ++c_itr) {
    const std::string set_name = c_itr->GetInOperand(0).AsString();
    if (spvtools::utils::starts_with(set_name, "NonSemantic.")) {
      non_sem_set_seen = true;
      break;
    }
  }
  if (!non_sem_set_seen) {
    for (auto c_itr = context()->module()->extension_begin();
         c_itr != context()->module()->extension_end(); ++c_itr) {
      const std::string ext_name = c_itr->GetInOperand(0).AsString();
      if (ext_name == "SPV_KHR_non_semantic_info") {
        context()->KillInst(&*c_itr);
        break;
      }
    }
    context()->get_feature_mgr()->RemoveExtension(kSPV_KHR_non_semantic_info);
  }
  return Status::SuccessWithChange;
}

Pass::Status InstDebugPrintfPass::Process() {
  ext_inst_printf_id_ =
      get_module()->GetExtInstImportId("NonSemantic.DebugPrintf");
  if (ext_inst_printf_id_ == 0) return Status::SuccessWithoutChange;
  InitializeInstDebugPrintf();
  return ProcessImpl();
}

}  // namespace opt
}  // namespace spvtools

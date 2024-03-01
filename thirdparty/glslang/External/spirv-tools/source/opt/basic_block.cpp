// Copyright (c) 2016 Google Inc.
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

#include "source/opt/basic_block.h"

#include <ostream>

#include "source/opt/ir_context.h"
#include "source/opt/reflect.h"
#include "source/util/make_unique.h"

namespace spvtools {
namespace opt {
namespace {
constexpr uint32_t kLoopMergeContinueBlockIdInIdx = 1;
constexpr uint32_t kLoopMergeMergeBlockIdInIdx = 0;
constexpr uint32_t kSelectionMergeMergeBlockIdInIdx = 0;
}  // namespace

BasicBlock* BasicBlock::Clone(IRContext* context) const {
  BasicBlock* clone = new BasicBlock(
      std::unique_ptr<Instruction>(GetLabelInst()->Clone(context)));
  for (const auto& inst : insts_) {
    // Use the incoming context
    clone->AddInstruction(std::unique_ptr<Instruction>(inst.Clone(context)));
  }

  if (context->AreAnalysesValid(
          IRContext::Analysis::kAnalysisInstrToBlockMapping)) {
    for (auto& inst : *clone) {
      context->set_instr_block(&inst, clone);
    }
  }

  return clone;
}

const Instruction* BasicBlock::GetMergeInst() const {
  const Instruction* result = nullptr;
  // If it exists, the merge instruction immediately precedes the
  // terminator.
  auto iter = ctail();
  if (iter != cbegin()) {
    --iter;
    const auto opcode = iter->opcode();
    if (opcode == spv::Op::OpLoopMerge || opcode == spv::Op::OpSelectionMerge) {
      result = &*iter;
    }
  }
  return result;
}

Instruction* BasicBlock::GetMergeInst() {
  Instruction* result = nullptr;
  // If it exists, the merge instruction immediately precedes the
  // terminator.
  auto iter = tail();
  if (iter != begin()) {
    --iter;
    const auto opcode = iter->opcode();
    if (opcode == spv::Op::OpLoopMerge || opcode == spv::Op::OpSelectionMerge) {
      result = &*iter;
    }
  }
  return result;
}

const Instruction* BasicBlock::GetLoopMergeInst() const {
  if (auto* merge = GetMergeInst()) {
    if (merge->opcode() == spv::Op::OpLoopMerge) {
      return merge;
    }
  }
  return nullptr;
}

Instruction* BasicBlock::GetLoopMergeInst() {
  if (auto* merge = GetMergeInst()) {
    if (merge->opcode() == spv::Op::OpLoopMerge) {
      return merge;
    }
  }
  return nullptr;
}

void BasicBlock::KillAllInsts(bool killLabel) {
  ForEachInst([killLabel](Instruction* ip) {
    if (killLabel || ip->opcode() != spv::Op::OpLabel) {
      ip->context()->KillInst(ip);
    }
  });
}

void BasicBlock::ForEachSuccessorLabel(
    const std::function<void(const uint32_t)>& f) const {
  WhileEachSuccessorLabel([f](const uint32_t l) {
    f(l);
    return true;
  });
}

bool BasicBlock::WhileEachSuccessorLabel(
    const std::function<bool(const uint32_t)>& f) const {
  const auto br = &insts_.back();
  switch (br->opcode()) {
    case spv::Op::OpBranch:
      return f(br->GetOperand(0).words[0]);
    case spv::Op::OpBranchConditional:
    case spv::Op::OpSwitch: {
      bool is_first = true;
      return br->WhileEachInId([&is_first, &f](const uint32_t* idp) {
        if (!is_first) return f(*idp);
        is_first = false;
        return true;
      });
    }
    default:
      return true;
  }
}

void BasicBlock::ForEachSuccessorLabel(
    const std::function<void(uint32_t*)>& f) {
  auto br = &insts_.back();
  switch (br->opcode()) {
    case spv::Op::OpBranch: {
      uint32_t tmp_id = br->GetOperand(0).words[0];
      f(&tmp_id);
      if (tmp_id != br->GetOperand(0).words[0]) br->SetOperand(0, {tmp_id});
    } break;
    case spv::Op::OpBranchConditional:
    case spv::Op::OpSwitch: {
      bool is_first = true;
      br->ForEachInId([&is_first, &f](uint32_t* idp) {
        if (!is_first) f(idp);
        is_first = false;
      });
    } break;
    default:
      break;
  }
}

bool BasicBlock::IsSuccessor(const BasicBlock* block) const {
  uint32_t succId = block->id();
  bool isSuccessor = false;
  ForEachSuccessorLabel([&isSuccessor, succId](const uint32_t label) {
    if (label == succId) isSuccessor = true;
  });
  return isSuccessor;
}

void BasicBlock::ForMergeAndContinueLabel(
    const std::function<void(const uint32_t)>& f) {
  auto ii = insts_.end();
  --ii;
  if (ii == insts_.begin()) return;
  --ii;
  if (ii->opcode() == spv::Op::OpSelectionMerge ||
      ii->opcode() == spv::Op::OpLoopMerge) {
    ii->ForEachInId([&f](const uint32_t* idp) { f(*idp); });
  }
}

uint32_t BasicBlock::MergeBlockIdIfAny() const {
  auto merge_ii = cend();
  --merge_ii;
  uint32_t mbid = 0;
  if (merge_ii != cbegin()) {
    --merge_ii;
    if (merge_ii->opcode() == spv::Op::OpLoopMerge) {
      mbid = merge_ii->GetSingleWordInOperand(kLoopMergeMergeBlockIdInIdx);
    } else if (merge_ii->opcode() == spv::Op::OpSelectionMerge) {
      mbid = merge_ii->GetSingleWordInOperand(kSelectionMergeMergeBlockIdInIdx);
    }
  }

  return mbid;
}

uint32_t BasicBlock::MergeBlockId() const {
  uint32_t mbid = MergeBlockIdIfAny();
  assert(mbid && "Expected block to have a corresponding merge block");
  return mbid;
}

uint32_t BasicBlock::ContinueBlockIdIfAny() const {
  auto merge_ii = cend();
  --merge_ii;
  uint32_t cbid = 0;
  if (merge_ii != cbegin()) {
    --merge_ii;
    if (merge_ii->opcode() == spv::Op::OpLoopMerge) {
      cbid = merge_ii->GetSingleWordInOperand(kLoopMergeContinueBlockIdInIdx);
    }
  }
  return cbid;
}

uint32_t BasicBlock::ContinueBlockId() const {
  uint32_t cbid = ContinueBlockIdIfAny();
  assert(cbid && "Expected block to have a corresponding continue target");
  return cbid;
}

std::ostream& operator<<(std::ostream& str, const BasicBlock& block) {
  str << block.PrettyPrint();
  return str;
}

void BasicBlock::Dump() const {
  std::cerr << "Basic block #" << id() << "\n" << *this << "\n ";
}

std::string BasicBlock::PrettyPrint(uint32_t options) const {
  std::ostringstream str;
  ForEachInst([&str, options](const Instruction* inst) {
    str << inst->PrettyPrint(options);
    if (!spvOpcodeIsBlockTerminator(inst->opcode())) {
      str << std::endl;
    }
  });
  return str.str();
}

BasicBlock* BasicBlock::SplitBasicBlock(IRContext* context, uint32_t label_id,
                                        iterator iter) {
  assert(!insts_.empty());

  std::unique_ptr<BasicBlock> new_block_temp = MakeUnique<BasicBlock>(
      MakeUnique<Instruction>(context, spv::Op::OpLabel, 0, label_id,
                              std::initializer_list<Operand>{}));
  BasicBlock* new_block = new_block_temp.get();
  function_->InsertBasicBlockAfter(std::move(new_block_temp), this);

  new_block->insts_.Splice(new_block->end(), &insts_, iter, end());
  assert(new_block->GetParent() == GetParent() &&
         "The parent should already be set appropriately.");

  context->AnalyzeDefUse(new_block->GetLabelInst());

  // Update the phi nodes in the successor blocks to reference the new block id.
  const_cast<const BasicBlock*>(new_block)->ForEachSuccessorLabel(
      [new_block, this, context](const uint32_t label) {
        BasicBlock* target_bb = context->get_instr_block(label);
        target_bb->ForEachPhiInst(
            [this, new_block, context](Instruction* phi_inst) {
              bool changed = false;
              for (uint32_t i = 1; i < phi_inst->NumInOperands(); i += 2) {
                if (phi_inst->GetSingleWordInOperand(i) == this->id()) {
                  changed = true;
                  phi_inst->SetInOperand(i, {new_block->id()});
                }
              }

              if (changed) {
                context->UpdateDefUse(phi_inst);
              }
            });
      });

  if (context->AreAnalysesValid(IRContext::kAnalysisInstrToBlockMapping)) {
    context->set_instr_block(new_block->GetLabelInst(), new_block);
    new_block->ForEachInst([new_block, context](Instruction* inst) {
      context->set_instr_block(inst, new_block);
    });
  }

  return new_block;
}

}  // namespace opt
}  // namespace spvtools

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

#include "source/opt/function.h"

#include <ostream>

#include "ir_context.h"
#include "source/util/bit_vector.h"

namespace spvtools {
namespace opt {

Function* Function::Clone(IRContext* ctx) const {
  Function* clone =
      new Function(std::unique_ptr<Instruction>(DefInst().Clone(ctx)));
  clone->params_.reserve(params_.size());
  ForEachParam(
      [clone, ctx](const Instruction* inst) {
        clone->AddParameter(std::unique_ptr<Instruction>(inst->Clone(ctx)));
      },
      true);

  for (const auto& i : debug_insts_in_header_) {
    clone->AddDebugInstructionInHeader(
        std::unique_ptr<Instruction>(i.Clone(ctx)));
  }

  clone->blocks_.reserve(blocks_.size());
  for (const auto& b : blocks_) {
    std::unique_ptr<BasicBlock> bb(b->Clone(ctx));
    clone->AddBasicBlock(std::move(bb));
  }

  clone->SetFunctionEnd(std::unique_ptr<Instruction>(EndInst()->Clone(ctx)));

  clone->non_semantic_.reserve(non_semantic_.size());
  for (auto& non_semantic : non_semantic_) {
    clone->AddNonSemanticInstruction(
        std::unique_ptr<Instruction>(non_semantic->Clone(ctx)));
  }
  return clone;
}

void Function::ForEachInst(const std::function<void(Instruction*)>& f,
                           bool run_on_debug_line_insts,
                           bool run_on_non_semantic_insts) {
  WhileEachInst(
      [&f](Instruction* inst) {
        f(inst);
        return true;
      },
      run_on_debug_line_insts, run_on_non_semantic_insts);
}

void Function::ForEachInst(const std::function<void(const Instruction*)>& f,
                           bool run_on_debug_line_insts,
                           bool run_on_non_semantic_insts) const {
  WhileEachInst(
      [&f](const Instruction* inst) {
        f(inst);
        return true;
      },
      run_on_debug_line_insts, run_on_non_semantic_insts);
}

bool Function::WhileEachInst(const std::function<bool(Instruction*)>& f,
                             bool run_on_debug_line_insts,
                             bool run_on_non_semantic_insts) {
  if (def_inst_) {
    if (!def_inst_->WhileEachInst(f, run_on_debug_line_insts)) {
      return false;
    }
  }

  for (auto& param : params_) {
    if (!param->WhileEachInst(f, run_on_debug_line_insts)) {
      return false;
    }
  }

  if (!debug_insts_in_header_.empty()) {
    Instruction* di = &debug_insts_in_header_.front();
    while (di != nullptr) {
      Instruction* next_instruction = di->NextNode();
      if (!di->WhileEachInst(f, run_on_debug_line_insts)) return false;
      di = next_instruction;
    }
  }

  for (auto& bb : blocks_) {
    if (!bb->WhileEachInst(f, run_on_debug_line_insts)) {
      return false;
    }
  }

  if (end_inst_) {
    if (!end_inst_->WhileEachInst(f, run_on_debug_line_insts)) {
      return false;
    }
  }

  if (run_on_non_semantic_insts) {
    for (auto& non_semantic : non_semantic_) {
      if (!non_semantic->WhileEachInst(f, run_on_debug_line_insts)) {
        return false;
      }
    }
  }

  return true;
}

bool Function::WhileEachInst(const std::function<bool(const Instruction*)>& f,
                             bool run_on_debug_line_insts,
                             bool run_on_non_semantic_insts) const {
  if (def_inst_) {
    if (!static_cast<const Instruction*>(def_inst_.get())
             ->WhileEachInst(f, run_on_debug_line_insts)) {
      return false;
    }
  }

  for (const auto& param : params_) {
    if (!static_cast<const Instruction*>(param.get())
             ->WhileEachInst(f, run_on_debug_line_insts)) {
      return false;
    }
  }

  for (const auto& di : debug_insts_in_header_) {
    if (!static_cast<const Instruction*>(&di)->WhileEachInst(
            f, run_on_debug_line_insts))
      return false;
  }

  for (const auto& bb : blocks_) {
    if (!static_cast<const BasicBlock*>(bb.get())->WhileEachInst(
            f, run_on_debug_line_insts)) {
      return false;
    }
  }

  if (end_inst_) {
    if (!static_cast<const Instruction*>(end_inst_.get())
             ->WhileEachInst(f, run_on_debug_line_insts)) {
      return false;
    }
  }

  if (run_on_non_semantic_insts) {
    for (auto& non_semantic : non_semantic_) {
      if (!static_cast<const Instruction*>(non_semantic.get())
               ->WhileEachInst(f, run_on_debug_line_insts)) {
        return false;
      }
    }
  }

  return true;
}

void Function::ForEachParam(const std::function<void(Instruction*)>& f,
                            bool run_on_debug_line_insts) {
  for (auto& param : params_)
    static_cast<Instruction*>(param.get())
        ->ForEachInst(f, run_on_debug_line_insts);
}

void Function::ForEachParam(const std::function<void(const Instruction*)>& f,
                            bool run_on_debug_line_insts) const {
  for (const auto& param : params_)
    static_cast<const Instruction*>(param.get())
        ->ForEachInst(f, run_on_debug_line_insts);
}

void Function::ForEachDebugInstructionsInHeader(
    const std::function<void(Instruction*)>& f) {
  if (debug_insts_in_header_.empty()) return;

  Instruction* di = &debug_insts_in_header_.front();
  while (di != nullptr) {
    Instruction* next_instruction = di->NextNode();
    di->ForEachInst(f);
    di = next_instruction;
  }
}

BasicBlock* Function::InsertBasicBlockAfter(
    std::unique_ptr<BasicBlock>&& new_block, BasicBlock* position) {
  for (auto bb_iter = begin(); bb_iter != end(); ++bb_iter) {
    if (&*bb_iter == position) {
      new_block->SetParent(this);
      ++bb_iter;
      bb_iter = bb_iter.InsertBefore(std::move(new_block));
      return &*bb_iter;
    }
  }
  assert(false && "Could not find insertion point.");
  return nullptr;
}

BasicBlock* Function::InsertBasicBlockBefore(
    std::unique_ptr<BasicBlock>&& new_block, BasicBlock* position) {
  for (auto bb_iter = begin(); bb_iter != end(); ++bb_iter) {
    if (&*bb_iter == position) {
      new_block->SetParent(this);
      bb_iter = bb_iter.InsertBefore(std::move(new_block));
      return &*bb_iter;
    }
  }
  assert(false && "Could not find insertion point.");
  return nullptr;
}

bool Function::HasEarlyReturn() const {
  auto post_dominator_analysis =
      blocks_.front()->GetLabel()->context()->GetPostDominatorAnalysis(this);
  for (auto& block : blocks_) {
    if (spvOpcodeIsReturn(block->tail()->opcode()) &&
        !post_dominator_analysis->Dominates(block.get(), entry().get())) {
      return true;
    }
  }
  return false;
}

bool Function::IsRecursive() const {
  IRContext* ctx = blocks_.front()->GetLabel()->context();
  IRContext::ProcessFunction mark_visited = [this](Function* fp) {
    return fp == this;
  };

  // Process the call tree from all of the function called by |this|.  If it get
  // back to |this|, then we have a recursive function.
  std::queue<uint32_t> roots;
  ctx->AddCalls(this, &roots);
  return ctx->ProcessCallTreeFromRoots(mark_visited, &roots);
}

std::ostream& operator<<(std::ostream& str, const Function& func) {
  str << func.PrettyPrint();
  return str;
}

void Function::Dump() const {
  std::cerr << "Function #" << result_id() << "\n" << *this << "\n";
}

std::string Function::PrettyPrint(uint32_t options) const {
  std::ostringstream str;
  ForEachInst([&str, options](const Instruction* inst) {
    str << inst->PrettyPrint(options);
    if (inst->opcode() != spv::Op::OpFunctionEnd) {
      str << std::endl;
    }
  });
  return str.str();
}

void Function::ReorderBasicBlocksInStructuredOrder() {
  std::list<BasicBlock*> order;
  IRContext* context = this->def_inst_->context();
  context->cfg()->ComputeStructuredOrder(this, blocks_[0].get(), &order);
  ReorderBasicBlocks(order.begin(), order.end());
}

}  // namespace opt
}  // namespace spvtools

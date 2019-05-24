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
#include "function.h"
#include "ir_context.h"

#include <source/util/bit_vector.h>
#include <ostream>
#include <sstream>

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

  clone->blocks_.reserve(blocks_.size());
  for (const auto& b : blocks_) {
    std::unique_ptr<BasicBlock> bb(b->Clone(ctx));
    bb->SetParent(clone);
    clone->AddBasicBlock(std::move(bb));
  }

  clone->SetFunctionEnd(std::unique_ptr<Instruction>(EndInst()->Clone(ctx)));
  return clone;
}

void Function::ForEachInst(const std::function<void(Instruction*)>& f,
                           bool run_on_debug_line_insts) {
  WhileEachInst(
      [&f](Instruction* inst) {
        f(inst);
        return true;
      },
      run_on_debug_line_insts);
}

void Function::ForEachInst(const std::function<void(const Instruction*)>& f,
                           bool run_on_debug_line_insts) const {
  WhileEachInst(
      [&f](const Instruction* inst) {
        f(inst);
        return true;
      },
      run_on_debug_line_insts);
}

bool Function::WhileEachInst(const std::function<bool(Instruction*)>& f,
                             bool run_on_debug_line_insts) {
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

  for (auto& bb : blocks_) {
    if (!bb->WhileEachInst(f, run_on_debug_line_insts)) {
      return false;
    }
  }

  if (end_inst_) return end_inst_->WhileEachInst(f, run_on_debug_line_insts);

  return true;
}

bool Function::WhileEachInst(const std::function<bool(const Instruction*)>& f,
                             bool run_on_debug_line_insts) const {
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

  for (const auto& bb : blocks_) {
    if (!static_cast<const BasicBlock*>(bb.get())->WhileEachInst(
            f, run_on_debug_line_insts)) {
      return false;
    }
  }

  if (end_inst_)
    return static_cast<const Instruction*>(end_inst_.get())
        ->WhileEachInst(f, run_on_debug_line_insts);

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
    if (inst->opcode() != SpvOpFunctionEnd) {
      str << std::endl;
    }
  });
  return str.str();
}
}  // namespace opt
}  // namespace spvtools

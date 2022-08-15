//===- DxilRemoveUnstructuredLoopExits.cpp - Make unrolled loops structured ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//
// Loops that look like the following when unrolled becomes unstructured:
//
//      for(;;) {
//        if (a) {
//          if (b) {
//            exit_code_0;
//            break;       // Unstructured loop exit
//          }
//
//          code_0;
//
//          if (c) {
//            if (d) {
//              exit_code_1;
//              break;    // Unstructured loop exit
//            }
//            code_1;
//          }
//
//          code_2;
//
//          ...
//        }
//
//        code_3;
//
//        if (exit)
//          break;
//      }
//      
//
// This pass transforms the loop into the following form:
//
//      bool broke_0 = false;
//      bool broke_1 = false;
//
//      for(;;) {
//        if (a) {
//          if (b) {
//            broke_0 = true;       // Break flag
//          }
//
//          if (!broke_0) {
//            code_0;
//          }
//
//          if (!broke_0) {
//            if (c) {
//              if (d) {
//                broke_1 = true;   // Break flag
//              }
//              if (!broke_1) {
//                code_1;
//              }
//            }
//
//            if (!broke_1) {
//              code_2;
//            }
//          }
//
//          ...
//        }
//
//        if (!broke_0) {
//          break;
//        }
//
//        if (!broke_1) {
//          break;
//        }
//
//        code_3;
//
//        if (exit)
//          break;
//      }
//
//      if (broke_0) {
//        exit_code_0;
//      }
//
//      if (broke_1) {
//        exit_code_1;
//      }
//
// Essentially it hoists the exit branch out of the loop.
//
// This function should be called any time before a function is unrolled to
// avoid generating unstructured code.
//
// There are several limitations at the moment:
//
//   - if code_0, code_1, etc has any loops in there, this transform
//     does not take place. Since the values that flow out of the conditions
//     are phi of undef, I do not want to risk the loops not exiting.
//
//   - code_0, code_1, etc, become conditional only when there are
//     side effects in there. This doesn't impact code correctness,
//     but the code will execute for one iteration even if the exit condition
//     is met.
//
// These limitations can be fixed in the future as needed.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/SetVector.h"
#include "dxc/HLSL/DxilNoops.h"

#include <unordered_map>
#include <unordered_set>

#include "DxilRemoveUnstructuredLoopExits.h"

using namespace llvm;

static bool IsNoop(Instruction *inst);

namespace {

struct Value_Info {
  Value *val, *false_val;
  PHINode *exit_phi;
};

struct Propagator {
  DenseMap<std::pair<BasicBlock *, Value *>, PHINode *> cached_phis;
  std::unordered_set<BasicBlock *> seen;

  // Get propagated value for val. It's guaranteed to be safe to use in bb.
  Value *Get(Value *val, BasicBlock *bb) {
    auto it = cached_phis.find({ bb, val });
    if (it == cached_phis.end())
      return nullptr;

    return it->second;
  }

  void DeleteAllNewValues() {
    for (auto &pair : cached_phis) {
      pair.second->dropAllReferences();
    }
    for (auto &pair : cached_phis) {
      pair.second->eraseFromParent();
    }
    cached_phis.clear();
  }

  BasicBlock *Run(std::vector<Value_Info> &exit_values, BasicBlock *exiting_block, BasicBlock *latch, DominatorTree *DT, Loop *L, LoopInfo *LI, std::vector<BasicBlock *> &blocks_with_side_effect) {
    BasicBlock *ret = RunImpl(exit_values, exiting_block, latch, DT, L, LI, blocks_with_side_effect);
    // If we failed, remove all the values we added.
    if (!ret) {
      DeleteAllNewValues();
    }
    return ret;
  }

  BasicBlock *RunImpl(std::vector<Value_Info> &exit_values, BasicBlock *exiting_block, BasicBlock *latch, DominatorTree *DT, Loop *L, LoopInfo *LI, std::vector<BasicBlock *> &blocks_with_side_effect) {

    struct Edge {
      BasicBlock *prev;
      BasicBlock *bb;
    };

    BasicBlock *new_exiting_block = nullptr;
    SmallVector<Edge, 4> work_list;
    work_list.push_back({ nullptr, exiting_block });
    seen.insert(exiting_block);

    for (unsigned i = 0; i < work_list.size(); i++) {
      auto &edge = work_list[i];
      BasicBlock *prev = edge.prev;
      BasicBlock *bb = edge.bb;

      // Don't continue to propagate when we hit the latch or dominate it.
      if (DT->dominates(bb, latch)) {
        new_exiting_block = bb;
        continue;
      }

      // Do not include the exiting block itself in this calculation
      if (prev != nullptr) {
        // If this block is part of an inner loop... Give up for now.
        if (LI->getLoopFor(bb) != L) {
          return nullptr;
        }
        // Otherwise just remember the blocks with side effects (including the latch)
        else {
          for (Instruction &I : *bb) {
            if (I.mayReadOrWriteMemory() && !IsNoop(&I)) {
              blocks_with_side_effect.push_back(bb);
              break;
            }
          }
        }
      } // If this is not the first iteration

      for (BasicBlock *succ : llvm::successors(bb)) {
        // Don't propagate if block is not part of this loop.
        if (!L->contains(succ))
          continue;

        for (auto &pair : exit_values) {
          // Find or create phi for the value in the successor block
          PHINode *phi = cached_phis[{ succ, pair.val }];
          if (!phi) {
            phi = PHINode::Create(pair.false_val->getType(), 0, "dx.struct_exit.prop", &*succ->begin());
            for (BasicBlock *pred : llvm::predecessors(succ)) {
              phi->addIncoming(pair.false_val, pred);
            }
            cached_phis[{ succ, pair.val }] = phi;
          }

          // Find the incoming value for successor block
          Value *incoming = nullptr;
          if (!prev) {
            incoming = pair.val;
          }
          else {
            incoming = cached_phis[{ bb, pair.val }];
          }

          // Set incoming value for our phi
          for (unsigned i = 0; i < phi->getNumIncomingValues(); i++) {
            if (phi->getIncomingBlock(i) == bb) {
              phi->setIncomingValue(i, incoming);
            }
          }

          // Add to worklist
          if (!seen.count(succ)) {
            work_list.push_back({ bb, succ });
            seen.insert(succ);
          }
        }
      } // for each succ
    } // for each in worklist

    if (new_exiting_block == exiting_block) {
      return nullptr;
    }

    return new_exiting_block;
  }
}; // struct Propagator

} // Unnamed namespace

static bool IsNoop(Instruction *inst) {
  if (CallInst *ci = dyn_cast<CallInst>(inst)) {
    if (Function *f = ci->getCalledFunction()) {
      return f->getName() == hlsl::kNoopName;
    }
  }
  return false;
}

static Value* GetDefaultValue(Type *type) {
  if (type->isIntegerTy()) {
    return ConstantInt::get(type, 0);
  }
  else if (type->isFloatingPointTy()) {
    return ConstantFP::get(type, 0);
  }
  return UndefValue::get(type);
}

static BasicBlock *GetExitBlockForExitingBlock(Loop *L, BasicBlock *exiting_block) {
  BranchInst *br = dyn_cast<BranchInst>(exiting_block->getTerminator());
  assert(L->contains(exiting_block));
  assert(br->isConditional());
  BasicBlock *result = L->contains(br->getSuccessor(0)) ? br->getSuccessor(1) : br->getSuccessor(0);
  assert(!L->contains(result));
  return result;
}

// Branch over the block's content with the condition cond.
// All values used outside the block is replaced by a phi.
//
static void SkipBlockWithBranch(BasicBlock *bb, Value *cond, Loop *L, LoopInfo *LI) {
  BasicBlock *body = bb->splitBasicBlock(bb->getFirstNonPHI());
  body->setName("dx.struct_exit.cond_body");
  BasicBlock *end = body->splitBasicBlock(body->getTerminator());
  end->setName("dx.struct_exit.cond_end");

  bb->getTerminator()->eraseFromParent();
  BranchInst::Create(end, body, cond, bb);

  for (Instruction &inst : *body) {
    PHINode *phi = nullptr;

    // For each user that's outside of 'body', replace its use of 'inst' with a phi created
    // in 'end'
    for (auto it = inst.user_begin(); it != inst.user_end();) {
      Instruction *user_inst = cast<Instruction>(*(it++));
      if (user_inst == phi)
        continue;
      if (user_inst->getParent() != body) {
        if (!phi) {
          phi = PHINode::Create(inst.getType(), 2, "", &*end->begin());
          phi->addIncoming(GetDefaultValue(inst.getType()), bb);
          phi->addIncoming(&inst, body);
        }
        user_inst->replaceUsesOfWith(&inst, phi);
      }
    } // For each user of inst of body
  } // For each inst in body

  L->addBasicBlockToLoop(body, *LI);
  L->addBasicBlockToLoop(end, *LI);
}

static unsigned GetNumPredecessors(BasicBlock *bb) {
  unsigned ret = 0;
  for (BasicBlock *pred : llvm::predecessors(bb)) {
    (void)pred;
    ret++;
  }
  return ret;
}

static bool RemoveUnstructuredLoopExitsIteration(BasicBlock *exiting_block, Loop *L, LoopInfo *LI, DominatorTree *DT) {

  LLVMContext &ctx = L->getHeader()->getContext();
  Type *i1Ty = Type::getInt1Ty(ctx);

  BasicBlock *exit_block = GetExitBlockForExitingBlock(L, exiting_block);

  BasicBlock *latch = L->getLoopLatch();
  BasicBlock *latch_exit = GetExitBlockForExitingBlock(L, latch);

  // If exiting block already dominates latch, then no need to do anything.
  if (DT->dominates(exiting_block, latch)) {
    return false;
  }

  Propagator prop;

  BranchInst *exiting_br = cast<BranchInst>(exiting_block->getTerminator());
  Value *exit_cond = exiting_br->getCondition();
  // When exit_block is false block, use !exit_cond as exit_cond.
  if (exiting_br->getSuccessor(1) == exit_block) {
    IRBuilder<> B(exiting_br);
    exit_cond = B.CreateNot(exit_cond);
  }
  BasicBlock *new_exiting_block = nullptr;

  std::vector<Value_Info> exit_values;
  std::vector<BasicBlock *> blocks_with_side_effect;

  // Find the values that flow into the exit block from this loop.
  {
    // Look at the lcssa phi's in the exit block.
    bool exit_cond_has_phi = false;
    for (Instruction &I : *exit_block) {
      if (PHINode *phi = dyn_cast<PHINode>(&I)) {
        // If there are values flowing out of the loop into the exit_block,
        // add them to the list to be propagated
        Value *value = phi->getIncomingValueForBlock(exiting_block);
        Value *false_value = nullptr;
        if (value == exit_cond) {
          false_value = ConstantInt::getFalse(i1Ty);
          exit_cond_has_phi = true;
        }
        else {
          false_value = GetDefaultValue(value->getType());
        }
        exit_values.push_back({ value, false_value, phi });
      }
      else {
        break;
      }
    }

    // If the exit condition is not among the exit phi's, add it.
    if (!exit_cond_has_phi) {
      exit_values.push_back({ exit_cond, ConstantInt::getFalse(i1Ty), nullptr });
    }
  }

  //
  // Propagate those values we just found to a block that dominates the latch
  //
  new_exiting_block = prop.Run(exit_values, exiting_block, latch, DT, L, LI, blocks_with_side_effect);

  // Stop now if we failed
  if (!new_exiting_block)
    return false;

  // If there are any blocks with side effects,
  for (BasicBlock *bb : blocks_with_side_effect) {
    Value *exit_cond_for_block = prop.Get(exit_cond, bb);
    SkipBlockWithBranch(bb, exit_cond_for_block, L, LI);
  }

  // Make the exiting block not exit.
  {
    BasicBlock *non_exiting_block = exiting_br->getSuccessor(exiting_br->getSuccessor(0) == exit_block ? 1 : 0);
    BranchInst::Create(non_exiting_block, exiting_block);
    exiting_br->eraseFromParent();
    exiting_br = nullptr;
  }

  Value *new_exit_cond = prop.Get(exit_cond, new_exiting_block);
  assert(new_exit_cond);

  // Split the block where we're now exiting from, and branch to latch exit
  std::string old_name = new_exiting_block->getName().str();
  BasicBlock *new_not_exiting_block = new_exiting_block->splitBasicBlock(new_exiting_block->getFirstNonPHI());
  new_exiting_block->setName("dx.struct_exit.new_exiting");
  new_not_exiting_block->setName(old_name);
  L->addBasicBlockToLoop(new_not_exiting_block, *LI);

  // Branch to latch_exit
  new_exiting_block->getTerminator()->eraseFromParent();
  BranchInst::Create(latch_exit, new_not_exiting_block, new_exit_cond, new_exiting_block);

  // If the exit block and the latch exit are the same, then we're already good.
  // just update the phi nodes in the exit block.
  if (latch_exit == exit_block) {
    for (Value_Info &info : exit_values) {
      // Take the phi node in the exit block and reset incoming block and value from latch_exit
      PHINode *exit_phi = info.exit_phi;
      if (exit_phi) {
        for (unsigned i = 0; i < exit_phi->getNumIncomingValues(); i++) {
          if (exit_phi->getIncomingBlock(i) == exiting_block) {
            exit_phi->setIncomingBlock(i, new_exiting_block);
            exit_phi->setIncomingValue(i, prop.Get(info.val, new_exiting_block));
          }
        }
      }
    }
  }
  // Otherwise...
  else {

    // 1. Split the latch exit, since it's going to branch to the real exit block
    BasicBlock *post_exit_location = latch_exit->splitBasicBlock(latch_exit->getFirstNonPHI());

    {
      // If latch exit is part of an outer loop, add its split in there too.
      if (Loop *outer_loop = LI->getLoopFor(latch_exit)) {
        outer_loop->addBasicBlockToLoop(post_exit_location, *LI);
      }
      // If the original exit block is part of an outer loop, then latch exit (which is the
      // new exit block) must be part of it, since all blocks that branch to within
      // a loop must be part of that loop structure.
      else if (Loop *outer_loop = LI->getLoopFor(exit_block)) {
        outer_loop->addBasicBlockToLoop(latch_exit, *LI);
      }
    }

    // 2. Add incoming values to latch_exit's phi nodes.
    // Since now new exiting block is branching to latch exit, its phis need to be updated.
    for (Instruction &inst : *latch_exit) {
      PHINode *phi = dyn_cast<PHINode>(&inst);
      if (!phi)
        break;
      phi->addIncoming(GetDefaultValue(phi->getType()), new_exiting_block);
    }


    unsigned latch_exit_num_predecessors = GetNumPredecessors(latch_exit);
    PHINode *exit_cond_lcssa = nullptr;
    for (Value_Info &info : exit_values) {

      // 3. Create lcssa phi's for all the propagated values at latch_exit.
      // Make exit values visible in the latch_exit
      PHINode *val_lcssa = PHINode::Create(info.val->getType(), latch_exit_num_predecessors, "dx.struct_exit.val_lcssa", latch_exit->begin());

      if (info.val == exit_cond) {
        // Record the phi for the exit condition
        exit_cond_lcssa = val_lcssa;
        exit_cond_lcssa->setName("dx.struct_exit.exit_cond_lcssa");
      }

      for (BasicBlock *pred : llvm::predecessors(latch_exit)) {
        if (pred == new_exiting_block) {
          Value *incoming = prop.Get(info.val, new_exiting_block);
          assert(incoming);
          val_lcssa->addIncoming(incoming, pred);
        }
        else {
          val_lcssa->addIncoming(info.false_val, pred);
        }
      }

      // 4. Update the phis in the exit_block to use the lcssa phi's we just created.
      PHINode *exit_phi = info.exit_phi;
      if (exit_phi) {
        for (unsigned i = 0; i < exit_phi->getNumIncomingValues(); i++) {
          if (exit_phi->getIncomingBlock(i) == exiting_block) {
            exit_phi->setIncomingBlock(i, latch_exit);
            exit_phi->setIncomingValue(i, val_lcssa);
          }
        }
      }
    }

    // 5. Take the first half of latch_exit and branch it to the exit_block based
    // on the propagated exit condition.
    latch_exit->getTerminator()->eraseFromParent();
    BranchInst::Create(exit_block, post_exit_location, exit_cond_lcssa, latch_exit);
  }

  DT->recalculate(*L->getHeader()->getParent());
  assert(L->isLCSSAForm(*DT));

  return true;
}

bool hlsl::RemoveUnstructuredLoopExits(llvm::Loop *L, llvm::LoopInfo *LI, llvm::DominatorTree *DT, std::unordered_set<llvm::BasicBlock *> *exclude_set) {
  
  bool changed = false;

  if (!L->isLCSSAForm(*DT))
    return false;

  // Give up if loop is not rotated somehow
  if (BasicBlock *latch = L->getLoopLatch()) {
    if (!cast<BranchInst>(latch->getTerminator())->isConditional())
      return false;
  }
  // Give up if there's not a single latch
  else {
    return false;
  }

  for (;;) {
    // Recompute exiting block every time, since they could change between
    // iterations
    llvm::SmallVector<BasicBlock *, 4> exiting_blocks;
    L->getExitingBlocks(exiting_blocks);

    bool local_changed = false;
    for (BasicBlock *exiting_block : exiting_blocks) {
      auto latch = L->getLoopLatch();
      if (latch == exiting_block)
        continue;

      if (exclude_set && exclude_set->count(GetExitBlockForExitingBlock(L, exiting_block)))
        continue;

      // As soon as we got a success, break and start a new iteration, since
      // exiting blocks could have changed.
      local_changed = RemoveUnstructuredLoopExitsIteration(exiting_block, L, LI, DT);
      if (local_changed) {
        break;
      }
    }

    changed |= local_changed;
    if (!local_changed) {
      break;
    }
  }

  return changed;
}


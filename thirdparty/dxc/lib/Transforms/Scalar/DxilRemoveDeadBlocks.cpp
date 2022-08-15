///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilRemoveDeadBlocks.cpp                                                  //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Pass to use value tracker to remove dead blocks.                          //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Analysis/DxilValueCache.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/IR/DebugInfo.h"

#include "dxc/DXIL/DxilMetadataHelper.h"
#include "dxc/DXIL/DxilOperations.h"
#include "dxc/DXIL/DxilUtil.h"
#include "dxc/DXIL/DxilModule.h"
#include "dxc/HLSL/DxilNoops.h"

#include <unordered_set>

using namespace llvm;
using namespace hlsl;

static void RemoveIncomingValueFrom(BasicBlock *SuccBB, BasicBlock *BB) {
  for (auto inst_it = SuccBB->begin(); inst_it != SuccBB->end();) {
    Instruction *I = &*(inst_it++);
    if (PHINode *PN = dyn_cast<PHINode>(I))
      PN->removeIncomingValue(BB, true);
    else
      break;
  }
}

struct DeadBlockDeleter {
  bool Run(Function &F, DxilValueCache *DVC);

private:
  std::unordered_set<BasicBlock *> Seen;
  std::vector<BasicBlock *> WorkList;

  void Add(BasicBlock *BB) {
    if (!Seen.count(BB)) {
      WorkList.push_back(BB);
      Seen.insert(BB);
    }
  }
};

bool DeadBlockDeleter::Run(Function &F, DxilValueCache *DVC) {
  Seen.clear();
  WorkList.clear();

  bool Changed = false;

  Add(&F.getEntryBlock());

  // Go through blocks
  while (WorkList.size()) {
    BasicBlock *BB = WorkList.back();
    WorkList.pop_back();

    if (BranchInst *Br = dyn_cast<BranchInst>(BB->getTerminator())) {
      if (Br->isUnconditional()) {
        BasicBlock *Succ = Br->getSuccessor(0);
        Add(Succ);
      }
      else {
        if (ConstantInt *C = DVC->GetConstInt(Br->getCondition())) {
          bool IsTrue = C->getLimitedValue() != 0;
          BasicBlock *Succ = Br->getSuccessor(IsTrue ? 0 : 1);
          BasicBlock *NotSucc = Br->getSuccessor(!IsTrue ? 0 : 1);

          Add(Succ);

          // Rewrite conditional branch as unconditional branch if
          // we don't have structural information that needs it to
          // be alive.
          if (!Br->getMetadata(hlsl::DXIL::kDxBreakMDName)) {
            BranchInst *NewBr = BranchInst::Create(Succ, BB);
            hlsl::DxilMDHelper::CopyMetadata(*NewBr, *Br);
            RemoveIncomingValueFrom(NotSucc, BB);

            Br->eraseFromParent();
            Br = nullptr;
            Changed = true;
          }
        }
        else {
          Add(Br->getSuccessor(0));
          Add(Br->getSuccessor(1));
        }
      }
    }
    else if (SwitchInst *Switch = dyn_cast<SwitchInst>(BB->getTerminator())) {
      Value *Cond = Switch->getCondition();
      BasicBlock *Succ = nullptr;
      if (ConstantInt *ConstCond = DVC->GetConstInt(Cond)) {
        Succ = hlsl::dxilutil::GetSwitchSuccessorForCond(Switch, ConstCond);
      }

      if (Succ) {
        Add(Succ);

        BranchInst *NewBr = BranchInst::Create(Succ, BB);
        hlsl::DxilMDHelper::CopyMetadata(*NewBr, *Switch);

        for (unsigned i = 0; i < Switch->getNumSuccessors(); i++) {
          BasicBlock *NotSucc = Switch->getSuccessor(i);
          if (NotSucc != Succ) {
            RemoveIncomingValueFrom(NotSucc, BB);
          }
        }

        Switch->eraseFromParent();
        Switch = nullptr;
        Changed = true;
      }
      else {
        for (unsigned i = 0; i < Switch->getNumSuccessors(); i++) {
          Add(Switch->getSuccessor(i));
        }
      }
    }
  }

  if (Seen.size() == F.size())
    return Changed;

  std::vector<BasicBlock *> DeadBlocks;

  // Reconnect edges and everything
  for (auto it = F.begin(); it != F.end();) {
    BasicBlock *BB = &*(it++);
    if (Seen.count(BB))
      continue;

    DeadBlocks.push_back(BB);

    // Make predecessors branch somewhere else and fix the phi nodes
    for (auto pred_it = pred_begin(BB); pred_it != pred_end(BB);) {
      BasicBlock *PredBB = *(pred_it++);
      if (!Seen.count(PredBB)) // Don't bother fixing it if it's gonna get deleted anyway
        continue;
      TerminatorInst *TI = PredBB->getTerminator();
      if (!TI) continue;
      BranchInst *Br = dyn_cast<BranchInst>(TI);
      if (!Br || Br->isUnconditional()) continue;

      BasicBlock *Other = Br->getSuccessor(0) == BB ?
        Br->getSuccessor(1) : Br->getSuccessor(0);

      BranchInst *NewBr = BranchInst::Create(Other, Br);
      hlsl::DxilMDHelper::CopyMetadata(*NewBr, *Br);
      Br->eraseFromParent();
    }

    // Fix phi nodes in successors
    for (auto succ_it = succ_begin(BB); succ_it != succ_end(BB); succ_it++) {
      BasicBlock *SuccBB = *succ_it;
      if (!Seen.count(SuccBB)) continue; // Don't bother fixing it if it's gonna get deleted anyway
      RemoveIncomingValueFrom(SuccBB, BB);
    }

    // Erase all instructions in block
    while (BB->size()) {
      Instruction *I = &BB->back();
      if (!I->getType()->isVoidTy())
        I->replaceAllUsesWith(UndefValue::get(I->getType()));
      I->eraseFromParent();
    }
  }

  for (BasicBlock *BB : DeadBlocks) {
    BB->eraseFromParent();
  }

  DVC->ResetUnknowns();

  return true;
}

static bool DeleteDeadBlocks(Function &F, DxilValueCache *DVC) {
  DeadBlockDeleter Deleter;
  bool Changed = false;
  constexpr unsigned MaxIteration = 10;
  for (unsigned i = 0; i < MaxIteration; i++) {
    bool LocalChanged = Deleter.Run(F, DVC);
    Changed |= LocalChanged;
    if (!LocalChanged)
      break;
  }
  return Changed;
}

static bool IsDxBreak(Instruction *I) {
  CallInst *CI = dyn_cast<CallInst>(I);
  if (!CI) return false;
  Function *CalledFunction = CI->getCalledFunction();
  return CalledFunction && CalledFunction->getName() == hlsl::DXIL::kDxBreakFuncName;
}

static bool IsIsHelperLane(Instruction *I) {
  return hlsl::OP::IsDxilOpFuncCallInst(I, hlsl::DXIL::OpCode::IsHelperLane);
}

static bool ShouldNotReplaceValue(Value *V) {
  Instruction *I = dyn_cast<Instruction>(V);
  return I && (IsDxBreak(I) || IsIsHelperLane(I));
}

namespace {

struct ValueDeleter {
  std::unordered_set<Value *> Seen;
  std::vector<Value *> WorkList;
  void Add(Value *V) {
    if (!Seen.count(V)) {
      Seen.insert(V);
      WorkList.push_back(V);
    }
  }
  bool Run(Function &F, DxilValueCache *DVC) {
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (I.mayHaveSideEffects())
          Add(&I);
        else if (I.isTerminator())
          Add(&I);
      }
    }

    while (WorkList.size()) {
      Value *V = WorkList.back();
      WorkList.pop_back();
      Instruction *I = dyn_cast<Instruction>(V);
      if (I) {
        for (unsigned i = 0; i < I->getNumOperands(); i++) {
          Value *op = I->getOperand(i);
          if (Instruction *OpI = dyn_cast<Instruction>(op)) {
            // If this operand could be reduced to a constant, stop adding all
            // its operands. Otherwise, we could unintentionally hold on to
            // dead instructions like:
            // 
            //   %my_actually_dead_inst = ...
            //   %my_const = fmul 0.0, %my_actually_dead_inst
            // 
            // %my_actually_dead_inst should be deleted with the rest of the
            // non-contributing instructions.
            if (Constant *C = DVC->GetConstValue(OpI))
              I->setOperand(i, C);
            else
              Add(OpI);
          }
        }
      }
    }

    // Go through all dbg.value and see if we can replace their value with constant.
    // As we go and delete all non-contributing values, we want to preserve as much debug
    // info as possible.
    if (llvm::hasDebugInfo(*F.getParent())) {
      Function *DbgValueF = F.getParent()->getFunction(Intrinsic::getName(Intrinsic::dbg_value));
      if (DbgValueF) {
        LLVMContext &Ctx = F.getContext();
        for (User *U : DbgValueF->users()) {
          DbgValueInst *DbgVal = cast<DbgValueInst>(U);
          Value *Val = DbgVal->getValue();
          if (Val) {
            if (Constant *C = DVC->GetConstValue(DbgVal->getValue())) {
              DbgVal->setArgOperand(0, MetadataAsValue::get(Ctx, ValueAsMetadata::get(C)));
            }
          }
        }
      }
    }

    bool Changed = false;
    for (auto bb_it = F.getBasicBlockList().rbegin(), bb_end = F.getBasicBlockList().rend(); bb_it != bb_end; bb_it++) {
      BasicBlock *BB = &*bb_it;
      for (auto it = BB->begin(), end = BB->end(); it != end;) {
        Instruction *I = &*(it++);
        if (isa<DbgInfoIntrinsic>(I)) continue;
        if (IsDxBreak(I)) continue;
        if (!Seen.count(I)) {
          if (!I->user_empty())
            I->replaceAllUsesWith(UndefValue::get(I->getType()));
          I->eraseFromParent();
          Changed = true;
        }
      }
    }
    return Changed;
  } // Run
};

}

// Iteratively and aggressively delete instructions that don't
// contribute to the shader's output.
// 
// Find all things that could impact the program's output, including:
//   - terminator insts (branches, switches, returns)
//   - anything with side effect
// Recursively find all the instructions they reference
// Delete all the rest
// 
// Also replace any values that the value cache determined can be
// replaced by a constant, with the exception of a few intrinsics that
// we expect to see in the output.
//
static bool DeleteNonContributingValues(Function &F, DxilValueCache *DVC) {

  ValueDeleter Deleter;

  DVC->ResetAll();
  DVC->SetShouldSkipCallback(ShouldNotReplaceValue);
  bool Changed = Deleter.Run(F, DVC);
  DVC->SetShouldSkipCallback(nullptr);
  return Changed;
}

static void EnsureDxilModule(Module *M) {
  if (M->HasDxilModule())
    return;
  for (Function &F : *M) {
    if (OP::IsDxilOpFunc(&F)) {
      bool bSkipInit = true; // Metadata is not necessarily valid yet.
      M->GetOrCreateDxilModule(bSkipInit);
      break;
    }
  }
}

namespace {

struct DxilRemoveDeadBlocks : public FunctionPass {
  static char ID;
  DxilRemoveDeadBlocks() : FunctionPass(ID) {
    initializeDxilRemoveDeadBlocksPass(*PassRegistry::getPassRegistry());
  }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DxilValueCache>();
  }
  bool runOnFunction(Function &F) override {
    DxilValueCache *DVC = &getAnalysis<DxilValueCache>();
    EnsureDxilModule(F.getParent()); // Ensure dxil module is available for DVC
    bool Changed = false;
    Changed |= hlsl::dxilutil::DeleteDeadAllocas(F);
    Changed |= DeleteDeadBlocks(F, DVC);
    Changed |= DeleteNonContributingValues(F, DVC);
    return Changed;
  }
};

}

char DxilRemoveDeadBlocks::ID;

Pass *llvm::createDxilRemoveDeadBlocksPass() {
  return new DxilRemoveDeadBlocks();
}

INITIALIZE_PASS_BEGIN(DxilRemoveDeadBlocks, "dxil-remove-dead-blocks", "DXIL Remove Dead Blocks", false, false)
INITIALIZE_PASS_DEPENDENCY(DxilValueCache)
INITIALIZE_PASS_END(DxilRemoveDeadBlocks, "dxil-remove-dead-blocks", "DXIL Remove Dead Blocks", false, false)

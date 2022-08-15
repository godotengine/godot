//===- DCE.cpp - Code to perform dead code elimination --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Aggressive Dead Code Elimination pass.  This pass
// optimistically assumes that all instructions are dead until proven otherwise,
// allowing it to eliminate dead computations that other DCE passes do not
// catch, particularly involving loop computations.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Pass.h"
using namespace llvm;

#define DEBUG_TYPE "adce"

STATISTIC(NumRemoved, "Number of instructions removed");

namespace {
struct ADCE : public FunctionPass {
  static char ID; // Pass identification, replacement for typeid
  ADCE() : FunctionPass(ID) {
    initializeADCEPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function& F) override;

  void getAnalysisUsage(AnalysisUsage& AU) const override {
    AU.setPreservesCFG();
  }
};
}

char ADCE::ID = 0;
INITIALIZE_PASS(ADCE, "adce", "Aggressive Dead Code Elimination", false, false)

bool ADCE::runOnFunction(Function& F) {
  if (skipOptnoneFunction(F))
    return false;

  SmallPtrSet<Instruction*, 128> Alive;
  SmallVector<Instruction*, 128> Worklist;

  // Collect the set of "root" instructions that are known live.
  for (Instruction &I : inst_range(F)) {
    if (isa<TerminatorInst>(I) || isa<DbgInfoIntrinsic>(I) ||
        isa<LandingPadInst>(I) || I.mayHaveSideEffects()) {
      Alive.insert(&I);
      Worklist.push_back(&I);
    }
  }

  // Propagate liveness backwards to operands.
  while (!Worklist.empty()) {
    Instruction *Curr = Worklist.pop_back_val();
    for (Use &OI : Curr->operands()) {
      if (Instruction *Inst = dyn_cast<Instruction>(OI))
        if (Alive.insert(Inst).second)
          Worklist.push_back(Inst);
    }
  }

  // The inverse of the live set is the dead set.  These are those instructions
  // which have no side effects and do not influence the control flow or return
  // value of the function, and may therefore be deleted safely.
  // NOTE: We reuse the Worklist vector here for memory efficiency.
  for (Instruction &I : inst_range(F)) {
    if (!Alive.count(&I)) {
      Worklist.push_back(&I);
      I.dropAllReferences();
    }
  }

  for (Instruction *&I : Worklist) {
    ++NumRemoved;
    I->eraseFromParent();
  }

  return !Worklist.empty();
}

FunctionPass *llvm::createAggressiveDCEPass() {
  return new ADCE();
}

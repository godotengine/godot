//===------ SimplifyInstructions.cpp - Remove redundant instructions ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is a utility pass used for testing the InstructionSimplify analysis.
// The analysis is applied to every instruction, and if it simplifies then the
// instruction is replaced by the simplification.  If you are looking for a pass
// that performs serious instruction folding, use the instcombine pass instead.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Type.h"
#include "llvm/Pass.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Transforms/Utils/Local.h"
using namespace llvm;

#define DEBUG_TYPE "instsimplify"

STATISTIC(NumSimplified, "Number of redundant instructions removed");

namespace {
  struct InstSimplifier : public FunctionPass {
    static char ID; // Pass identification, replacement for typeid
    InstSimplifier() : FunctionPass(ID) {
      initializeInstSimplifierPass(*PassRegistry::getPassRegistry());
    }

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.setPreservesCFG();
      AU.addRequired<AssumptionCacheTracker>();
      AU.addRequired<TargetLibraryInfoWrapperPass>();
    }

    /// runOnFunction - Remove instructions that simplify.
    bool runOnFunction(Function &F) override {
      const DominatorTreeWrapperPass *DTWP =
          getAnalysisIfAvailable<DominatorTreeWrapperPass>();
      const DominatorTree *DT = DTWP ? &DTWP->getDomTree() : nullptr;
      const DataLayout &DL = F.getParent()->getDataLayout();
      const TargetLibraryInfo *TLI =
          &getAnalysis<TargetLibraryInfoWrapperPass>().getTLI();
      AssumptionCache *AC =
          &getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);
      SmallPtrSet<const Instruction*, 8> S1, S2, *ToSimplify = &S1, *Next = &S2;
      bool Changed = false;

      do {
        for (BasicBlock *BB : depth_first(&F.getEntryBlock()))
          // Here be subtlety: the iterator must be incremented before the loop
          // body (not sure why), so a range-for loop won't work here.
          for (BasicBlock::iterator BI = BB->begin(), BE = BB->end(); BI != BE;) {
            Instruction *I = BI++;
            // The first time through the loop ToSimplify is empty and we try to
            // simplify all instructions.  On later iterations ToSimplify is not
            // empty and we only bother simplifying instructions that are in it.
            if (!ToSimplify->empty() && !ToSimplify->count(I))
              continue;
            // Don't waste time simplifying unused instructions.
            if (!I->use_empty())
              if (Value *V = SimplifyInstruction(I, DL, TLI, DT, AC)) {
                // Mark all uses for resimplification next time round the loop.
                for (User *U : I->users())
                  Next->insert(cast<Instruction>(U));
                I->replaceAllUsesWith(V);
                ++NumSimplified;
                Changed = true;
              }
            bool res = RecursivelyDeleteTriviallyDeadInstructions(I, TLI);
            if (res)  {
              // RecursivelyDeleteTriviallyDeadInstruction can remove
              // more than one instruction, so simply incrementing the
              // iterator does not work. When instructions get deleted
              // re-iterate instead.
              BI = BB->begin(); BE = BB->end();
              Changed |= res;
            }
          }

        // Place the list of instructions to simplify on the next loop iteration
        // into ToSimplify.
        std::swap(ToSimplify, Next);
        Next->clear();
      } while (!ToSimplify->empty());

      return Changed;
    }
  };
}

char InstSimplifier::ID = 0;
INITIALIZE_PASS_BEGIN(InstSimplifier, "instsimplify",
                      "Remove redundant instructions", false, false)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_END(InstSimplifier, "instsimplify",
                    "Remove redundant instructions", false, false)
char &llvm::InstructionSimplifierID = InstSimplifier::ID;

// Public interface to the simplify instructions pass.
FunctionPass *llvm::createInstructionSimplifierPass() {
  return new InstSimplifier();
}

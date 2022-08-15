//===- SimplifyCFGPass.cpp - CFG Simplification Pass ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements dead code elimination and basic block merging, along
// with a collection of other peephole control flow optimizations.  For example:
//
//   * Removes basic blocks with no predecessors.
//   * Merges a basic block into its predecessor if there is only one and the
//     predecessor only has one successor.
//   * Eliminates PHI nodes for basic blocks with a single predecessor.
//   * Eliminates a basic block that only contains an unconditional branch.
//   * Changes invoke instructions to nounwind functions to be calls.
//   * Change things like "if (x) if (y)" into "if (x&y)".
//   * etc..
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Scalar.h"
using namespace llvm;

#define DEBUG_TYPE "simplifycfg"

#if 0 // HLSL Change Starts
static cl::opt<unsigned>
UserBonusInstThreshold("bonus-inst-threshold", cl::Hidden, cl::init(1),
   cl::desc("Control the number of bonus instructions (default = 1)"));
#else
unsigned UserBonusInstThreshold = 1;
#endif

STATISTIC(NumSimpl, "Number of blocks simplified");

/// If we have more than one empty (other than phi node) return blocks,
/// merge them together to promote recursive block merging.
static bool mergeEmptyReturnBlocks(Function &F) {
  bool Changed = false;

  BasicBlock *RetBlock = nullptr;

  // Scan all the blocks in the function, looking for empty return blocks.
  for (Function::iterator BBI = F.begin(), E = F.end(); BBI != E; ) {
    BasicBlock &BB = *BBI++;

    // Only look at return blocks.
    ReturnInst *Ret = dyn_cast<ReturnInst>(BB.getTerminator());
    if (!Ret) continue;

    // Only look at the block if it is empty or the only other thing in it is a
    // single PHI node that is the operand to the return.
    if (Ret != &BB.front()) {
      // Check for something else in the block.
      BasicBlock::iterator I = Ret;
      --I;
      // Skip over debug info.
      while (isa<DbgInfoIntrinsic>(I) && I != BB.begin())
        --I;
      if (!isa<DbgInfoIntrinsic>(I) &&
          (!isa<PHINode>(I) || I != BB.begin() ||
           Ret->getNumOperands() == 0 ||
           Ret->getOperand(0) != I))
        continue;
    }

    // If this is the first returning block, remember it and keep going.
    if (!RetBlock) {
      RetBlock = &BB;
      continue;
    }

    // Otherwise, we found a duplicate return block.  Merge the two.
    Changed = true;

    // Case when there is no input to the return or when the returned values
    // agree is trivial.  Note that they can't agree if there are phis in the
    // blocks.
    if (Ret->getNumOperands() == 0 ||
        Ret->getOperand(0) ==
          cast<ReturnInst>(RetBlock->getTerminator())->getOperand(0)) {
      BB.replaceAllUsesWith(RetBlock);
      BB.eraseFromParent();
      continue;
    }

    // If the canonical return block has no PHI node, create one now.
    PHINode *RetBlockPHI = dyn_cast<PHINode>(RetBlock->begin());
    if (!RetBlockPHI) {
      Value *InVal = cast<ReturnInst>(RetBlock->getTerminator())->getOperand(0);
      pred_iterator PB = pred_begin(RetBlock), PE = pred_end(RetBlock);
      RetBlockPHI = PHINode::Create(Ret->getOperand(0)->getType(),
                                    std::distance(PB, PE), "merge",
                                    &RetBlock->front());

      for (pred_iterator PI = PB; PI != PE; ++PI)
        RetBlockPHI->addIncoming(InVal, *PI);
      RetBlock->getTerminator()->setOperand(0, RetBlockPHI);
    }

    // Turn BB into a block that just unconditionally branches to the return
    // block.  This handles the case when the two return blocks have a common
    // predecessor but that return different things.
    RetBlockPHI->addIncoming(Ret->getOperand(0), &BB);
    BB.getTerminator()->eraseFromParent();
    BranchInst::Create(RetBlock, &BB);
  }

  return Changed;
}

/// Call SimplifyCFG on all the blocks in the function,
/// iterating until no more changes are made.
static bool iterativelySimplifyCFG(Function &F, const TargetTransformInfo &TTI,
                                   AssumptionCache *AC,
                                   unsigned BonusInstThreshold) {
  bool Changed = false;
  bool LocalChange = true;
  while (LocalChange) {
    LocalChange = false;

    // Loop over all of the basic blocks and remove them if they are unneeded.
    for (Function::iterator BBIt = F.begin(); BBIt != F.end(); ) {
      if (SimplifyCFG(BBIt++, TTI, BonusInstThreshold, AC)) {
        LocalChange = true;
        ++NumSimpl;
      }
    }
    Changed |= LocalChange;
  }
  return Changed;
}

static bool simplifyFunctionCFG(Function &F, const TargetTransformInfo &TTI,
                                AssumptionCache *AC, int BonusInstThreshold) {
  bool EverChanged = removeUnreachableBlocks(F);
  EverChanged |= mergeEmptyReturnBlocks(F);
  EverChanged |= iterativelySimplifyCFG(F, TTI, AC, BonusInstThreshold);

  // If neither pass changed anything, we're done.
  if (!EverChanged) return false;

  // iterativelySimplifyCFG can (rarely) make some loops dead.  If this happens,
  // removeUnreachableBlocks is needed to nuke them, which means we should
  // iterate between the two optimizations.  We structure the code like this to
  // avoid rerunning iterativelySimplifyCFG if the second pass of
  // removeUnreachableBlocks doesn't do anything.
  if (!removeUnreachableBlocks(F))
    return true;

  do {
    EverChanged = iterativelySimplifyCFG(F, TTI, AC, BonusInstThreshold);
    EverChanged |= removeUnreachableBlocks(F);
  } while (EverChanged);

  return true;
}

SimplifyCFGPass::SimplifyCFGPass()
    : BonusInstThreshold(UserBonusInstThreshold) {}

SimplifyCFGPass::SimplifyCFGPass(int BonusInstThreshold)
    : BonusInstThreshold(BonusInstThreshold) {}

PreservedAnalyses SimplifyCFGPass::run(Function &F,
                                       AnalysisManager<Function> *AM) {
  auto &TTI = AM->getResult<TargetIRAnalysis>(F);
  auto &AC = AM->getResult<AssumptionAnalysis>(F);

  if (!simplifyFunctionCFG(F, TTI, &AC, BonusInstThreshold))
    return PreservedAnalyses::none();

  return PreservedAnalyses::all();
}

namespace {
struct CFGSimplifyPass : public FunctionPass {
  static char ID; // Pass identification, replacement for typeid
  unsigned BonusInstThreshold;
  std::function<bool(const Function &)> PredicateFtor;

  CFGSimplifyPass(int T = -1,
                  std::function<bool(const Function &)> Ftor = nullptr)
      : FunctionPass(ID), PredicateFtor(Ftor) {
    BonusInstThreshold = (T == -1) ? UserBonusInstThreshold : unsigned(T);
    initializeCFGSimplifyPassPass(*PassRegistry::getPassRegistry());
  }
  bool runOnFunction(Function &F) override {
    if (PredicateFtor && !PredicateFtor(F))
      return false;

    if (skipOptnoneFunction(F))
      return false;

    AssumptionCache *AC =
        &getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);
    const TargetTransformInfo &TTI =
        getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
    return simplifyFunctionCFG(F, TTI, AC, BonusInstThreshold);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AssumptionCacheTracker>();
    AU.addRequired<TargetTransformInfoWrapperPass>();
  }
};
}

char CFGSimplifyPass::ID = 0;
INITIALIZE_PASS_BEGIN(CFGSimplifyPass, "simplifycfg", "Simplify the CFG", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_END(CFGSimplifyPass, "simplifycfg", "Simplify the CFG", false,
                    false)

// Public interface to the CFGSimplification pass
FunctionPass *
llvm::createCFGSimplificationPass(int Threshold,
                                  std::function<bool(const Function &)> Ftor) {
  return new CFGSimplifyPass(Threshold, Ftor);
}


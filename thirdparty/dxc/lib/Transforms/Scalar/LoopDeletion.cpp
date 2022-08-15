//===- LoopDeletion.cpp - Dead Loop Deletion Pass ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Dead Loop Deletion Pass. This pass is responsible
// for eliminating loops with non-infinite computable trip counts that have no
// side effects or volatile instructions, and do not contribute to the
// computation of the function's return value.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/Dominators.h"
using namespace llvm;

#define DEBUG_TYPE "loop-delete"

STATISTIC(NumDeleted, "Number of loops deleted");

namespace {
  class LoopDeletion : public LoopPass {
  public:
    static char ID; // Pass ID, replacement for typeid
    LoopDeletion() : LoopPass(ID) {
      initializeLoopDeletionPass(*PassRegistry::getPassRegistry());
    }

    // Possibly eliminate loop L if it is dead.
    bool runOnLoop(Loop *L, LPPassManager &LPM) override;

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.addRequired<DominatorTreeWrapperPass>();
      AU.addRequired<LoopInfoWrapperPass>();
      AU.addRequired<ScalarEvolution>();
      AU.addRequiredID(LoopSimplifyID);
      AU.addRequiredID(LCSSAID);

      AU.addPreserved<ScalarEvolution>();
      AU.addPreserved<DominatorTreeWrapperPass>();
      AU.addPreserved<LoopInfoWrapperPass>();
      AU.addPreservedID(LoopSimplifyID);
      AU.addPreservedID(LCSSAID);
    }

  private:
    bool isLoopDead(Loop *L, SmallVectorImpl<BasicBlock *> &exitingBlocks,
                    SmallVectorImpl<BasicBlock *> &exitBlocks,
                    bool &Changed, BasicBlock *Preheader);

  };
}

char LoopDeletion::ID = 0;
INITIALIZE_PASS_BEGIN(LoopDeletion, "loop-deletion",
                "Delete dead loops", false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ScalarEvolution)
INITIALIZE_PASS_DEPENDENCY(LoopSimplify)
INITIALIZE_PASS_DEPENDENCY(LCSSA)
INITIALIZE_PASS_END(LoopDeletion, "loop-deletion",
                "Delete dead loops", false, false)

Pass *llvm::createLoopDeletionPass() {
  return new LoopDeletion();
}

/// isLoopDead - Determined if a loop is dead.  This assumes that we've already
/// checked for unique exit and exiting blocks, and that the code is in LCSSA
/// form.
bool LoopDeletion::isLoopDead(Loop *L,
                              SmallVectorImpl<BasicBlock *> &exitingBlocks,
                              SmallVectorImpl<BasicBlock *> &exitBlocks,
                              bool &Changed, BasicBlock *Preheader) {
  BasicBlock *exitBlock = exitBlocks[0];

  // Make sure that all PHI entries coming from the loop are loop invariant.
  // Because the code is in LCSSA form, any values used outside of the loop
  // must pass through a PHI in the exit block, meaning that this check is
  // sufficient to guarantee that no loop-variant values are used outside
  // of the loop.
  BasicBlock::iterator BI = exitBlock->begin();
  while (PHINode *P = dyn_cast<PHINode>(BI)) {
    Value *incoming = P->getIncomingValueForBlock(exitingBlocks[0]);

    // Make sure all exiting blocks produce the same incoming value for the exit
    // block.  If there are different incoming values for different exiting
    // blocks, then it is impossible to statically determine which value should
    // be used.
    for (unsigned i = 1, e = exitingBlocks.size(); i < e; ++i) {
      if (incoming != P->getIncomingValueForBlock(exitingBlocks[i]))
        return false;
    }

    if (Instruction *I = dyn_cast<Instruction>(incoming))
      if (!L->makeLoopInvariant(I, Changed, Preheader->getTerminator()))
        return false;

    ++BI;
  }

  // Make sure that no instructions in the block have potential side-effects.
  // This includes instructions that could write to memory, and loads that are
  // marked volatile.  This could be made more aggressive by using aliasing
  // information to identify readonly and readnone calls.
  for (Loop::block_iterator LI = L->block_begin(), LE = L->block_end();
       LI != LE; ++LI) {
    for (BasicBlock::iterator BI = (*LI)->begin(), BE = (*LI)->end();
         BI != BE; ++BI) {
      if (BI->mayHaveSideEffects())
        return false;
    }
  }

  return true;
}

/// runOnLoop - Remove dead loops, by which we mean loops that do not impact the
/// observable behavior of the program other than finite running time.  Note
/// we do ensure that this never remove a loop that might be infinite, as doing
/// so could change the halting/non-halting nature of a program.
/// NOTE: This entire process relies pretty heavily on LoopSimplify and LCSSA
/// in order to make various safety checks work.
bool LoopDeletion::runOnLoop(Loop *L, LPPassManager &LPM) {
  if (skipOptnoneFunction(L))
    return false;

  // We can only remove the loop if there is a preheader that we can
  // branch from after removing it.
  BasicBlock *preheader = L->getLoopPreheader();
  if (!preheader)
    return false;

  // If LoopSimplify form is not available, stay out of trouble.
  if (!L->hasDedicatedExits())
    return false;

  // We can't remove loops that contain subloops.  If the subloops were dead,
  // they would already have been removed in earlier executions of this pass.
  if (L->begin() != L->end())
    return false;

  SmallVector<BasicBlock*, 4> exitingBlocks;
  L->getExitingBlocks(exitingBlocks);

  SmallVector<BasicBlock*, 4> exitBlocks;
  L->getUniqueExitBlocks(exitBlocks);

  // We require that the loop only have a single exit block.  Otherwise, we'd
  // be in the situation of needing to be able to solve statically which exit
  // block will be branched to, or trying to preserve the branching logic in
  // a loop invariant manner.
  if (exitBlocks.size() != 1)
    return false;

  // Finally, we have to check that the loop really is dead.
  bool Changed = false;
  if (!isLoopDead(L, exitingBlocks, exitBlocks, Changed, preheader))
    return Changed;

  // Don't remove loops for which we can't solve the trip count.
  // They could be infinite, in which case we'd be changing program behavior.
  ScalarEvolution &SE = getAnalysis<ScalarEvolution>();
  // HLSL Change begin - remove loops even cannot solve the trip count.
  //const SCEV *S = SE.getMaxBackedgeTakenCount(L);
  ////if (isa<SCEVCouldNotCompute>(S))
  //  return Changed;
  // HLSL Change end.

  // Now that we know the removal is safe, remove the loop by changing the
  // branch from the preheader to go to the single exit block.
  BasicBlock *exitBlock = exitBlocks[0];

  // Because we're deleting a large chunk of code at once, the sequence in which
  // we remove things is very important to avoid invalidation issues.  Don't
  // mess with this unless you have good reason and know what you're doing.

  // Tell ScalarEvolution that the loop is deleted. Do this before
  // deleting the loop so that ScalarEvolution can look at the loop
  // to determine what it needs to clean up.
  SE.forgetLoop(L);

  // Connect the preheader directly to the exit block.
  TerminatorInst *TI = preheader->getTerminator();
  TI->replaceUsesOfWith(L->getHeader(), exitBlock);

  // Rewrite phis in the exit block to get their inputs from
  // the preheader instead of the exiting block.
  BasicBlock *exitingBlock = exitingBlocks[0];
  BasicBlock::iterator BI = exitBlock->begin();
  while (PHINode *P = dyn_cast<PHINode>(BI)) {
    int j = P->getBasicBlockIndex(exitingBlock);
    assert(j >= 0 && "Can't find exiting block in exit block's phi node!");
    P->setIncomingBlock(j, preheader);
    for (unsigned i = 1; i < exitingBlocks.size(); ++i)
      P->removeIncomingValue(exitingBlocks[i]);
    ++BI;
  }

  // Update the dominator tree and remove the instructions and blocks that will
  // be deleted from the reference counting scheme.
  DominatorTree &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  SmallVector<DomTreeNode*, 8> ChildNodes;
  for (Loop::block_iterator LI = L->block_begin(), LE = L->block_end();
       LI != LE; ++LI) {
    // Move all of the block's children to be children of the preheader, which
    // allows us to remove the domtree entry for the block.
    ChildNodes.insert(ChildNodes.begin(), DT[*LI]->begin(), DT[*LI]->end());
    for (SmallVectorImpl<DomTreeNode *>::iterator DI = ChildNodes.begin(),
         DE = ChildNodes.end(); DI != DE; ++DI) {
      DT.changeImmediateDominator(*DI, DT[preheader]);
    }

    ChildNodes.clear();
    DT.eraseNode(*LI);

    // Remove the block from the reference counting scheme, so that we can
    // delete it freely later.
    (*LI)->dropAllReferences();
  }

  // Erase the instructions and the blocks without having to worry
  // about ordering because we already dropped the references.
  // NOTE: This iteration is safe because erasing the block does not remove its
  // entry from the loop's block list.  We do that in the next section.
  for (Loop::block_iterator LI = L->block_begin(), LE = L->block_end();
       LI != LE; ++LI)
    (*LI)->eraseFromParent();

  // Finally, the blocks from loopinfo.  This has to happen late because
  // otherwise our loop iterators won't work.
  LoopInfo &loopInfo = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  SmallPtrSet<BasicBlock*, 8> blocks;
  blocks.insert(L->block_begin(), L->block_end());
  for (BasicBlock *BB : blocks)
    loopInfo.removeBlock(BB);

  // The last step is to inform the loop pass manager that we've
  // eliminated this loop.
  LPM.deleteLoopFromQueue(L);
  Changed = true;

  ++NumDeleted;

  return Changed;
}

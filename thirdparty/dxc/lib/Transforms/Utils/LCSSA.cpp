//===-- LCSSA.cpp - Convert loops into loop-closed SSA form ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass transforms loops by placing phi nodes at the end of the loops for
// all values that are live across the loop boundary.  For example, it turns
// the left into the right code:
// 
// for (...)                for (...)
//   if (c)                   if (c)
//     X1 = ...                 X1 = ...
//   else                     else
//     X2 = ...                 X2 = ...
//   X3 = phi(X1, X2)         X3 = phi(X1, X2)
// ... = X3 + 4             X4 = phi(X3)
//                          ... = X4 + 4
//
// This is still valid LLVM; the extra phi nodes are purely redundant, and will
// be trivially eliminated by InstCombine.  The major benefit of this 
// transformation is that it makes many other loop optimizations, such as 
// LoopUnswitching, simpler.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PredIteratorCache.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/SSAUpdater.h"
using namespace llvm;

#define DEBUG_TYPE "lcssa"

STATISTIC(NumLCSSA, "Number of live out of a loop variables");

/// Return true if the specified block is in the list.
static bool isExitBlock(BasicBlock *BB,
                        const SmallVectorImpl<BasicBlock *> &ExitBlocks) {
  for (unsigned i = 0, e = ExitBlocks.size(); i != e; ++i)
    if (ExitBlocks[i] == BB)
      return true;
  return false;
}

/// Given an instruction in the loop, check to see if it has any uses that are
/// outside the current loop.  If so, insert LCSSA PHI nodes and rewrite the
/// uses.
static bool processInstruction(Loop &L, Instruction &Inst, DominatorTree &DT,
                               const SmallVectorImpl<BasicBlock *> &ExitBlocks,
                               PredIteratorCache &PredCache, LoopInfo *LI) {
  SmallVector<Use *, 16> UsesToRewrite;

  BasicBlock *InstBB = Inst.getParent();

  for (Use &U : Inst.uses()) {
    Instruction *User = cast<Instruction>(U.getUser());
    BasicBlock *UserBB = User->getParent();
    if (PHINode *PN = dyn_cast<PHINode>(User))
      UserBB = PN->getIncomingBlock(U);

    if (InstBB != UserBB && !L.contains(UserBB))
      UsesToRewrite.push_back(&U);
  }

  // If there are no uses outside the loop, exit with no change.
  if (UsesToRewrite.empty())
    return false;

  ++NumLCSSA; // We are applying the transformation

  // Invoke instructions are special in that their result value is not available
  // along their unwind edge. The code below tests to see whether DomBB
  // dominates
  // the value, so adjust DomBB to the normal destination block, which is
  // effectively where the value is first usable.
  BasicBlock *DomBB = Inst.getParent();
  if (InvokeInst *Inv = dyn_cast<InvokeInst>(&Inst))
    DomBB = Inv->getNormalDest();

  DomTreeNode *DomNode = DT.getNode(DomBB);

  SmallVector<PHINode *, 16> AddedPHIs;
  SmallVector<PHINode *, 8> PostProcessPHIs;

  SSAUpdater SSAUpdate;
  SSAUpdate.Initialize(Inst.getType(), Inst.getName());

  // Insert the LCSSA phi's into all of the exit blocks dominated by the
  // value, and add them to the Phi's map.
  for (SmallVectorImpl<BasicBlock *>::const_iterator BBI = ExitBlocks.begin(),
                                                     BBE = ExitBlocks.end();
       BBI != BBE; ++BBI) {
    BasicBlock *ExitBB = *BBI;
    if (!DT.dominates(DomNode, DT.getNode(ExitBB)))
      continue;

    // If we already inserted something for this BB, don't reprocess it.
    if (SSAUpdate.HasValueForBlock(ExitBB))
      continue;

    PHINode *PN = PHINode::Create(Inst.getType(), PredCache.size(ExitBB),
                                  Inst.getName() + ".lcssa", ExitBB->begin());

    // Add inputs from inside the loop for this PHI.
    for (BasicBlock *Pred : PredCache.get(ExitBB)) {
      PN->addIncoming(&Inst, Pred);

      // If the exit block has a predecessor not within the loop, arrange for
      // the incoming value use corresponding to that predecessor to be
      // rewritten in terms of a different LCSSA PHI.
      if (!L.contains(Pred))
        UsesToRewrite.push_back(
            &PN->getOperandUse(PN->getOperandNumForIncomingValue(
                 PN->getNumIncomingValues() - 1)));
    }

    AddedPHIs.push_back(PN);

    // Remember that this phi makes the value alive in this block.
    SSAUpdate.AddAvailableValue(ExitBB, PN);

    // LoopSimplify might fail to simplify some loops (e.g. when indirect
    // branches are involved). In such situations, it might happen that an exit
    // for Loop L1 is the header of a disjoint Loop L2. Thus, when we create
    // PHIs in such an exit block, we are also inserting PHIs into L2's header.
    // This could break LCSSA form for L2 because these inserted PHIs can also
    // have uses outside of L2. Remember all PHIs in such situation as to
    // revisit than later on. FIXME: Remove this if indirectbr support into
    // LoopSimplify gets improved.
    if (auto *OtherLoop = LI->getLoopFor(ExitBB))
      if (!L.contains(OtherLoop))
        PostProcessPHIs.push_back(PN);
  }

  // Rewrite all uses outside the loop in terms of the new PHIs we just
  // inserted.
  for (unsigned i = 0, e = UsesToRewrite.size(); i != e; ++i) {
    // If this use is in an exit block, rewrite to use the newly inserted PHI.
    // This is required for correctness because SSAUpdate doesn't handle uses in
    // the same block.  It assumes the PHI we inserted is at the end of the
    // block.
    Instruction *User = cast<Instruction>(UsesToRewrite[i]->getUser());
    BasicBlock *UserBB = User->getParent();
    if (PHINode *PN = dyn_cast<PHINode>(User))
      UserBB = PN->getIncomingBlock(*UsesToRewrite[i]);

    if (isa<PHINode>(UserBB->begin()) && isExitBlock(UserBB, ExitBlocks)) {
      // Tell the VHs that the uses changed. This updates SCEV's caches.
      if (UsesToRewrite[i]->get()->hasValueHandle())
        ValueHandleBase::ValueIsRAUWd(*UsesToRewrite[i], UserBB->begin());
      UsesToRewrite[i]->set(UserBB->begin());
      continue;
    }

    // Otherwise, do full PHI insertion.
    SSAUpdate.RewriteUse(*UsesToRewrite[i]);
  }

  // Post process PHI instructions that were inserted into another disjoint loop
  // and update their exits properly.
  for (auto *I : PostProcessPHIs) {
    if (I->use_empty())
      continue;

    BasicBlock *PHIBB = I->getParent();
    Loop *OtherLoop = LI->getLoopFor(PHIBB);
    SmallVector<BasicBlock *, 8> EBs;
    OtherLoop->getExitBlocks(EBs);
    if (EBs.empty())
      continue;

    // Recurse and re-process each PHI instruction. FIXME: we should really
    // convert this entire thing to a worklist approach where we process a
    // vector of instructions...
    processInstruction(*OtherLoop, *I, DT, EBs, PredCache, LI);
  }

  // Remove PHI nodes that did not have any uses rewritten.
  for (unsigned i = 0, e = AddedPHIs.size(); i != e; ++i) {
    if (AddedPHIs[i]->use_empty())
      AddedPHIs[i]->eraseFromParent();
  }

  return true;
}

/// Return true if the specified block dominates at least
/// one of the blocks in the specified list.
static bool
blockDominatesAnExit(BasicBlock *BB,
                     DominatorTree &DT,
                     const SmallVectorImpl<BasicBlock *> &ExitBlocks) {
  DomTreeNode *DomNode = DT.getNode(BB);
  for (unsigned i = 0, e = ExitBlocks.size(); i != e; ++i)
    if (DT.dominates(DomNode, DT.getNode(ExitBlocks[i])))
      return true;

  return false;
}

bool llvm::formLCSSA(Loop &L, DominatorTree &DT, LoopInfo *LI,
                     ScalarEvolution *SE) {
  bool Changed = false;

  // Get the set of exiting blocks.
  SmallVector<BasicBlock *, 8> ExitBlocks;
  L.getExitBlocks(ExitBlocks);

  if (ExitBlocks.empty())
    return false;

  PredIteratorCache PredCache;

  // Look at all the instructions in the loop, checking to see if they have uses
  // outside the loop.  If so, rewrite those uses.
  for (Loop::block_iterator BBI = L.block_begin(), BBE = L.block_end();
       BBI != BBE; ++BBI) {
    BasicBlock *BB = *BBI;

    // For large loops, avoid use-scanning by using dominance information:  In
    // particular, if a block does not dominate any of the loop exits, then none
    // of the values defined in the block could be used outside the loop.
    if (!blockDominatesAnExit(BB, DT, ExitBlocks))
      continue;

    for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I) {
      // Reject two common cases fast: instructions with no uses (like stores)
      // and instructions with one use that is in the same block as this.
      if (I->use_empty() ||
          (I->hasOneUse() && I->user_back()->getParent() == BB &&
           !isa<PHINode>(I->user_back())))
        continue;

      Changed |= processInstruction(L, *I, DT, ExitBlocks, PredCache, LI);
    }
  }

  // If we modified the code, remove any caches about the loop from SCEV to
  // avoid dangling entries.
  // FIXME: This is a big hammer, can we clear the cache more selectively?
  if (SE && Changed)
    SE->forgetLoop(&L);

  assert(L.isLCSSAForm(DT));

  return Changed;
}

/// Process a loop nest depth first.
bool llvm::formLCSSARecursively(Loop &L, DominatorTree &DT, LoopInfo *LI,
                                ScalarEvolution *SE) {
  bool Changed = false;

  // Recurse depth-first through inner loops.
  for (Loop::iterator I = L.begin(), E = L.end(); I != E; ++I)
    Changed |= formLCSSARecursively(**I, DT, LI, SE);

  Changed |= formLCSSA(L, DT, LI, SE);
  return Changed;
}

namespace {
struct LCSSA : public FunctionPass {
  static char ID; // Pass identification, replacement for typeid
  LCSSA() : FunctionPass(ID) {
    initializeLCSSAPass(*PassRegistry::getPassRegistry());
  }

  // Cached analysis information for the current function.
  DominatorTree *DT;
  LoopInfo *LI;
  ScalarEvolution *SE;

  bool runOnFunction(Function &F) override;

  /// This transformation requires natural loop information & requires that
  /// loop preheaders be inserted into the CFG.  It maintains both of these,
  /// as well as the CFG.  It also requires dominator information.
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();

    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addPreservedID(LoopSimplifyID);
    AU.addPreserved<AliasAnalysis>();
    AU.addPreserved<ScalarEvolution>();
  }
};
}

char LCSSA::ID = 0;
INITIALIZE_PASS_BEGIN(LCSSA, "lcssa", "Loop-Closed SSA Form Pass", false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_END(LCSSA, "lcssa", "Loop-Closed SSA Form Pass", false, false)

Pass *llvm::createLCSSAPass() { return new LCSSA(); }
char &llvm::LCSSAID = LCSSA::ID;


/// Process all loops in the function, inner-most out.
bool LCSSA::runOnFunction(Function &F) {
  bool Changed = false;
  LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  SE = getAnalysisIfAvailable<ScalarEvolution>();

  // Simplify each loop nest in the function.
  for (LoopInfo::iterator I = LI->begin(), E = LI->end(); I != E; ++I)
    Changed |= formLCSSARecursively(**I, *DT, LI, SE);

  return Changed;
}


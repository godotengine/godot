//===- LoopSimplify.cpp - Loop Canonicalization Pass ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass performs several transformations to transform natural loops into a
// simpler form, which makes subsequent analyses and transformations simpler and
// more effective.
//
// Loop pre-header insertion guarantees that there is a single, non-critical
// entry edge from outside of the loop to the loop header.  This simplifies a
// number of analyses and transformations, such as LICM.
//
// Loop exit-block insertion guarantees that all exit blocks from the loop
// (blocks which are outside of the loop that have predecessors inside of the
// loop) only have predecessors from inside of the loop (and are thus dominated
// by the loop header).  This simplifies transformations such as store-sinking
// that are built into LICM.
//
// This pass also guarantees that loops will have exactly one backedge.
//
// Indirectbr instructions introduce several complications. If the loop
// contains or is entered by an indirectbr instruction, it may not be possible
// to transform the loop and make these guarantees. Client code should check
// that these conditions are true before relying on them.
//
// Note that the simplifycfg pass will clean up blocks which are split out but
// end up being unnecessary, so usage of this pass should not pessimize
// generated code.
//
// This pass obviously modifies the CFG, but updates loop information and
// dominator information.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/DependenceAnalysis.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/LoopSimplify.h"
using namespace llvm;

#define DEBUG_TYPE "loop-simplify"

STATISTIC(NumInserted, "Number of pre-header or exit blocks inserted");
STATISTIC(NumNested  , "Number of nested loops split out");

// If the block isn't already, move the new block to right after some 'outside
// block' block.  This prevents the preheader from being placed inside the loop
// body, e.g. when the loop hasn't been rotated.
static void placeSplitBlockCarefully(BasicBlock *NewBB,
                                     SmallVectorImpl<BasicBlock *> &SplitPreds,
                                     Loop *L) {
  // Check to see if NewBB is already well placed.
  Function::iterator BBI = NewBB; --BBI;
  for (unsigned i = 0, e = SplitPreds.size(); i != e; ++i) {
    if (&*BBI == SplitPreds[i])
      return;
  }

  // If it isn't already after an outside block, move it after one.  This is
  // always good as it makes the uncond branch from the outside block into a
  // fall-through.

  // Figure out *which* outside block to put this after.  Prefer an outside
  // block that neighbors a BB actually in the loop.
  BasicBlock *FoundBB = nullptr;
  for (unsigned i = 0, e = SplitPreds.size(); i != e; ++i) {
    Function::iterator BBI = SplitPreds[i];
    if (++BBI != NewBB->getParent()->end() &&
        L->contains(BBI)) {
      FoundBB = SplitPreds[i];
      break;
    }
  }

  // If our heuristic for a *good* bb to place this after doesn't find
  // anything, just pick something.  It's likely better than leaving it within
  // the loop.
  if (!FoundBB)
    FoundBB = SplitPreds[0];
  NewBB->moveAfter(FoundBB);
}

/// InsertPreheaderForLoop - Once we discover that a loop doesn't have a
/// preheader, this method is called to insert one.  This method has two phases:
/// preheader insertion and analysis updating.
///
BasicBlock *llvm::InsertPreheaderForLoop(Loop *L, Pass *PP) {
  BasicBlock *Header = L->getHeader();

  // Get analyses that we try to update.
  auto *AA = PP->getAnalysisIfAvailable<AliasAnalysis>();
  auto *DTWP = PP->getAnalysisIfAvailable<DominatorTreeWrapperPass>();
  auto *DT = DTWP ? &DTWP->getDomTree() : nullptr;
  auto *LIWP = PP->getAnalysisIfAvailable<LoopInfoWrapperPass>();
  auto *LI = LIWP ? &LIWP->getLoopInfo() : nullptr;
  bool PreserveLCSSA = PP->mustPreserveAnalysisID(LCSSAID);

  // Compute the set of predecessors of the loop that are not in the loop.
  SmallVector<BasicBlock*, 8> OutsideBlocks;
  for (pred_iterator PI = pred_begin(Header), PE = pred_end(Header);
       PI != PE; ++PI) {
    BasicBlock *P = *PI;
    if (!L->contains(P)) {         // Coming in from outside the loop?
      // If the loop is branched to from an indirect branch, we won't
      // be able to fully transform the loop, because it prohibits
      // edge splitting.
      if (isa<IndirectBrInst>(P->getTerminator())) return nullptr;

      // Keep track of it.
      OutsideBlocks.push_back(P);
    }
  }

  // Split out the loop pre-header.
  BasicBlock *PreheaderBB;
  PreheaderBB = SplitBlockPredecessors(Header, OutsideBlocks, ".preheader",
                                       AA, DT, LI, PreserveLCSSA);

  DEBUG(dbgs() << "LoopSimplify: Creating pre-header "
               << PreheaderBB->getName() << "\n");

  // Make sure that NewBB is put someplace intelligent, which doesn't mess up
  // code layout too horribly.
  placeSplitBlockCarefully(PreheaderBB, OutsideBlocks, L);

  return PreheaderBB;
}

/// \brief Ensure that the loop preheader dominates all exit blocks.
///
/// This method is used to split exit blocks that have predecessors outside of
/// the loop.
static BasicBlock *rewriteLoopExitBlock(Loop *L, BasicBlock *Exit,
                                        AliasAnalysis *AA, DominatorTree *DT,
                                        LoopInfo *LI, Pass *PP) {
  SmallVector<BasicBlock*, 8> LoopBlocks;
  for (pred_iterator I = pred_begin(Exit), E = pred_end(Exit); I != E; ++I) {
    BasicBlock *P = *I;
    if (L->contains(P)) {
      // Don't do this if the loop is exited via an indirect branch.
      if (isa<IndirectBrInst>(P->getTerminator())) return nullptr;

      LoopBlocks.push_back(P);
    }
  }

  assert(!LoopBlocks.empty() && "No edges coming in from outside the loop?");
  BasicBlock *NewExitBB = nullptr;

  bool PreserveLCSSA = PP->mustPreserveAnalysisID(LCSSAID);

  NewExitBB = SplitBlockPredecessors(Exit, LoopBlocks, ".loopexit", AA, DT,
                                     LI, PreserveLCSSA);

  DEBUG(dbgs() << "LoopSimplify: Creating dedicated exit block "
               << NewExitBB->getName() << "\n");
  return NewExitBB;
}

/// Add the specified block, and all of its predecessors, to the specified set,
/// if it's not already in there.  Stop predecessor traversal when we reach
/// StopBlock.
static void addBlockAndPredsToSet(BasicBlock *InputBB, BasicBlock *StopBlock,
                                  std::set<BasicBlock*> &Blocks) {
  SmallVector<BasicBlock *, 8> Worklist;
  Worklist.push_back(InputBB);
  do {
    BasicBlock *BB = Worklist.pop_back_val();
    if (Blocks.insert(BB).second && BB != StopBlock)
      // If BB is not already processed and it is not a stop block then
      // insert its predecessor in the work list
      for (pred_iterator I = pred_begin(BB), E = pred_end(BB); I != E; ++I) {
        BasicBlock *WBB = *I;
        Worklist.push_back(WBB);
      }
  } while (!Worklist.empty());
}

/// \brief The first part of loop-nestification is to find a PHI node that tells
/// us how to partition the loops.
static PHINode *findPHIToPartitionLoops(Loop *L, AliasAnalysis *AA,
                                        DominatorTree *DT,
                                        AssumptionCache *AC) {
  const DataLayout &DL = L->getHeader()->getModule()->getDataLayout();
  for (BasicBlock::iterator I = L->getHeader()->begin(); isa<PHINode>(I); ) {
    PHINode *PN = cast<PHINode>(I);
    ++I;
    if (Value *V = SimplifyInstruction(PN, DL, nullptr, DT, AC)) {
      // This is a degenerate PHI already, don't modify it!
      PN->replaceAllUsesWith(V);
      if (AA) AA->deleteValue(PN);
      PN->eraseFromParent();
      continue;
    }

    // Scan this PHI node looking for a use of the PHI node by itself.
    for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i)
      if (PN->getIncomingValue(i) == PN &&
          L->contains(PN->getIncomingBlock(i)))
        // We found something tasty to remove.
        return PN;
  }
  return nullptr;
}

/// \brief If this loop has multiple backedges, try to pull one of them out into
/// a nested loop.
///
/// This is important for code that looks like
/// this:
///
///  Loop:
///     ...
///     br cond, Loop, Next
///     ...
///     br cond2, Loop, Out
///
/// To identify this common case, we look at the PHI nodes in the header of the
/// loop.  PHI nodes with unchanging values on one backedge correspond to values
/// that change in the "outer" loop, but not in the "inner" loop.
///
/// If we are able to separate out a loop, return the new outer loop that was
/// created.
///
static Loop *separateNestedLoop(Loop *L, BasicBlock *Preheader,
                                AliasAnalysis *AA, DominatorTree *DT,
                                LoopInfo *LI, ScalarEvolution *SE, Pass *PP,
                                AssumptionCache *AC) {
  // Don't try to separate loops without a preheader.
  if (!Preheader)
    return nullptr;

  // The header is not a landing pad; preheader insertion should ensure this.
  assert(!L->getHeader()->isLandingPad() &&
         "Can't insert backedge to landing pad");

  PHINode *PN = findPHIToPartitionLoops(L, AA, DT, AC);
  if (!PN) return nullptr;  // No known way to partition.

  // Pull out all predecessors that have varying values in the loop.  This
  // handles the case when a PHI node has multiple instances of itself as
  // arguments.
  SmallVector<BasicBlock*, 8> OuterLoopPreds;
  for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
    if (PN->getIncomingValue(i) != PN ||
        !L->contains(PN->getIncomingBlock(i))) {
      // We can't split indirectbr edges.
      if (isa<IndirectBrInst>(PN->getIncomingBlock(i)->getTerminator()))
        return nullptr;
      OuterLoopPreds.push_back(PN->getIncomingBlock(i));
    }
  }
  DEBUG(dbgs() << "LoopSimplify: Splitting out a new outer loop\n");

  // If ScalarEvolution is around and knows anything about values in
  // this loop, tell it to forget them, because we're about to
  // substantially change it.
  if (SE)
    SE->forgetLoop(L);

  bool PreserveLCSSA = PP->mustPreserveAnalysisID(LCSSAID);

  BasicBlock *Header = L->getHeader();
  BasicBlock *NewBB = SplitBlockPredecessors(Header, OuterLoopPreds, ".outer",
                                             AA, DT, LI, PreserveLCSSA);

  // Make sure that NewBB is put someplace intelligent, which doesn't mess up
  // code layout too horribly.
  placeSplitBlockCarefully(NewBB, OuterLoopPreds, L);

  // Create the new outer loop.
  Loop *NewOuter = new Loop();

  // Change the parent loop to use the outer loop as its child now.
  if (Loop *Parent = L->getParentLoop())
    Parent->replaceChildLoopWith(L, NewOuter);
  else
    LI->changeTopLevelLoop(L, NewOuter);

  // L is now a subloop of our outer loop.
  NewOuter->addChildLoop(L);

  for (Loop::block_iterator I = L->block_begin(), E = L->block_end();
       I != E; ++I)
    NewOuter->addBlockEntry(*I);

  // Now reset the header in L, which had been moved by
  // SplitBlockPredecessors for the outer loop.
  L->moveToHeader(Header);

  // Determine which blocks should stay in L and which should be moved out to
  // the Outer loop now.
  std::set<BasicBlock*> BlocksInL;
  for (pred_iterator PI=pred_begin(Header), E = pred_end(Header); PI!=E; ++PI) {
    BasicBlock *P = *PI;
    if (DT->dominates(Header, P))
      addBlockAndPredsToSet(P, Header, BlocksInL);
  }

  // Scan all of the loop children of L, moving them to OuterLoop if they are
  // not part of the inner loop.
  const std::vector<Loop*> &SubLoops = L->getSubLoops();
  for (size_t I = 0; I != SubLoops.size(); )
    if (BlocksInL.count(SubLoops[I]->getHeader()))
      ++I;   // Loop remains in L
    else
      NewOuter->addChildLoop(L->removeChildLoop(SubLoops.begin() + I));

  // Now that we know which blocks are in L and which need to be moved to
  // OuterLoop, move any blocks that need it.
  for (unsigned i = 0; i != L->getBlocks().size(); ++i) {
    BasicBlock *BB = L->getBlocks()[i];
    if (!BlocksInL.count(BB)) {
      // Move this block to the parent, updating the exit blocks sets
      L->removeBlockFromLoop(BB);
      if ((*LI)[BB] == L)
        LI->changeLoopFor(BB, NewOuter);
      --i;
    }
  }

  return NewOuter;
}

/// \brief This method is called when the specified loop has more than one
/// backedge in it.
///
/// If this occurs, revector all of these backedges to target a new basic block
/// and have that block branch to the loop header.  This ensures that loops
/// have exactly one backedge.
static BasicBlock *insertUniqueBackedgeBlock(Loop *L, BasicBlock *Preheader,
                                             AliasAnalysis *AA,
                                             DominatorTree *DT, LoopInfo *LI) {
  assert(L->getNumBackEdges() > 1 && "Must have > 1 backedge!");

  // Get information about the loop
  BasicBlock *Header = L->getHeader();
  Function *F = Header->getParent();

  // Unique backedge insertion currently depends on having a preheader.
  if (!Preheader)
    return nullptr;

  // The header is not a landing pad; preheader insertion should ensure this.
  assert(!Header->isLandingPad() && "Can't insert backedge to landing pad");

  // Figure out which basic blocks contain back-edges to the loop header.
  std::vector<BasicBlock*> BackedgeBlocks;
  for (pred_iterator I = pred_begin(Header), E = pred_end(Header); I != E; ++I){
    BasicBlock *P = *I;

    // Indirectbr edges cannot be split, so we must fail if we find one.
    if (isa<IndirectBrInst>(P->getTerminator()))
      return nullptr;

    if (P != Preheader) BackedgeBlocks.push_back(P);
  }

  // Create and insert the new backedge block...
  BasicBlock *BEBlock = BasicBlock::Create(Header->getContext(),
                                           Header->getName() + ".backedge", F);
  BranchInst *BETerminator = BranchInst::Create(Header, BEBlock);
  BETerminator->setDebugLoc(Header->getFirstNonPHI()->getDebugLoc());

  DEBUG(dbgs() << "LoopSimplify: Inserting unique backedge block "
               << BEBlock->getName() << "\n");

  // Move the new backedge block to right after the last backedge block.
  Function::iterator InsertPos = BackedgeBlocks.back(); ++InsertPos;
  F->getBasicBlockList().splice(InsertPos, F->getBasicBlockList(), BEBlock);

  // Now that the block has been inserted into the function, create PHI nodes in
  // the backedge block which correspond to any PHI nodes in the header block.
  for (BasicBlock::iterator I = Header->begin(); isa<PHINode>(I); ++I) {
    PHINode *PN = cast<PHINode>(I);
    PHINode *NewPN = PHINode::Create(PN->getType(), BackedgeBlocks.size(),
                                     PN->getName()+".be", BETerminator);

    // Loop over the PHI node, moving all entries except the one for the
    // preheader over to the new PHI node.
    unsigned PreheaderIdx = ~0U;
    bool HasUniqueIncomingValue = true;
    Value *UniqueValue = nullptr;
    for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
      BasicBlock *IBB = PN->getIncomingBlock(i);
      Value *IV = PN->getIncomingValue(i);
      if (IBB == Preheader) {
        PreheaderIdx = i;
      } else {
        NewPN->addIncoming(IV, IBB);
        if (HasUniqueIncomingValue) {
          if (!UniqueValue)
            UniqueValue = IV;
          else if (UniqueValue != IV)
            HasUniqueIncomingValue = false;
        }
      }
    }

    // Delete all of the incoming values from the old PN except the preheader's
    assert(PreheaderIdx != ~0U && "PHI has no preheader entry??");
    if (PreheaderIdx != 0) {
      PN->setIncomingValue(0, PN->getIncomingValue(PreheaderIdx));
      PN->setIncomingBlock(0, PN->getIncomingBlock(PreheaderIdx));
    }
    // Nuke all entries except the zero'th.
    for (unsigned i = 0, e = PN->getNumIncomingValues()-1; i != e; ++i)
      PN->removeIncomingValue(e-i, false);

    // Finally, add the newly constructed PHI node as the entry for the BEBlock.
    PN->addIncoming(NewPN, BEBlock);

    // As an optimization, if all incoming values in the new PhiNode (which is a
    // subset of the incoming values of the old PHI node) have the same value,
    // eliminate the PHI Node.
    if (HasUniqueIncomingValue) {
      NewPN->replaceAllUsesWith(UniqueValue);
      if (AA) AA->deleteValue(NewPN);
      BEBlock->getInstList().erase(NewPN);
    }
  }

  // Now that all of the PHI nodes have been inserted and adjusted, modify the
  // backedge blocks to just to the BEBlock instead of the header.
  for (unsigned i = 0, e = BackedgeBlocks.size(); i != e; ++i) {
    TerminatorInst *TI = BackedgeBlocks[i]->getTerminator();
    for (unsigned Op = 0, e = TI->getNumSuccessors(); Op != e; ++Op)
      if (TI->getSuccessor(Op) == Header)
        TI->setSuccessor(Op, BEBlock);
  }

  //===--- Update all analyses which we must preserve now -----------------===//

  // Update Loop Information - we know that this block is now in the current
  // loop and all parent loops.
  L->addBasicBlockToLoop(BEBlock, *LI);

  // Update dominator information
  DT->splitBlock(BEBlock);

  return BEBlock;
}

/// \brief Simplify one loop and queue further loops for simplification.
///
/// FIXME: Currently this accepts both lots of analyses that it uses and a raw
/// Pass pointer. The Pass pointer is used by numerous utilities to update
/// specific analyses. Rather than a pass it would be much cleaner and more
/// explicit if they accepted the analysis directly and then updated it.
static bool simplifyOneLoop(Loop *L, SmallVectorImpl<Loop *> &Worklist,
                            AliasAnalysis *AA, DominatorTree *DT, LoopInfo *LI,
                            ScalarEvolution *SE, Pass *PP,
                            AssumptionCache *AC) {
  bool Changed = false;
ReprocessLoop:

  // Check to see that no blocks (other than the header) in this loop have
  // predecessors that are not in the loop.  This is not valid for natural
  // loops, but can occur if the blocks are unreachable.  Since they are
  // unreachable we can just shamelessly delete those CFG edges!
  for (Loop::block_iterator BB = L->block_begin(), E = L->block_end();
       BB != E; ++BB) {
    if (*BB == L->getHeader()) continue;

    SmallPtrSet<BasicBlock*, 4> BadPreds;
    for (pred_iterator PI = pred_begin(*BB),
         PE = pred_end(*BB); PI != PE; ++PI) {
      BasicBlock *P = *PI;
      if (!L->contains(P))
        BadPreds.insert(P);
    }

    // Delete each unique out-of-loop (and thus dead) predecessor.
    for (BasicBlock *P : BadPreds) {

      DEBUG(dbgs() << "LoopSimplify: Deleting edge from dead predecessor "
                   << P->getName() << "\n");

      // Inform each successor of each dead pred.
      for (succ_iterator SI = succ_begin(P), SE = succ_end(P); SI != SE; ++SI)
        (*SI)->removePredecessor(P);
      // Zap the dead pred's terminator and replace it with unreachable.
      TerminatorInst *TI = P->getTerminator();
       TI->replaceAllUsesWith(UndefValue::get(TI->getType()));
      P->getTerminator()->eraseFromParent();
      new UnreachableInst(P->getContext(), P);
      Changed = true;
    }
  }

  // If there are exiting blocks with branches on undef, resolve the undef in
  // the direction which will exit the loop. This will help simplify loop
  // trip count computations.
  SmallVector<BasicBlock*, 8> ExitingBlocks;
  L->getExitingBlocks(ExitingBlocks);
  for (SmallVectorImpl<BasicBlock *>::iterator I = ExitingBlocks.begin(),
       E = ExitingBlocks.end(); I != E; ++I)
    if (BranchInst *BI = dyn_cast<BranchInst>((*I)->getTerminator()))
      if (BI->isConditional()) {
        if (UndefValue *Cond = dyn_cast<UndefValue>(BI->getCondition())) {

          DEBUG(dbgs() << "LoopSimplify: Resolving \"br i1 undef\" to exit in "
                       << (*I)->getName() << "\n");

          BI->setCondition(ConstantInt::get(Cond->getType(),
                                            !L->contains(BI->getSuccessor(0))));

          // This may make the loop analyzable, force SCEV recomputation.
          if (SE)
            SE->forgetLoop(L);

          Changed = true;
        }
      }

  // Does the loop already have a preheader?  If so, don't insert one.
  BasicBlock *Preheader = L->getLoopPreheader();
  if (!Preheader) {
    Preheader = InsertPreheaderForLoop(L, PP);
    if (Preheader) {
      ++NumInserted;
      Changed = true;
    }
  }

  // Next, check to make sure that all exit nodes of the loop only have
  // predecessors that are inside of the loop.  This check guarantees that the
  // loop preheader/header will dominate the exit blocks.  If the exit block has
  // predecessors from outside of the loop, split the edge now.
  SmallVector<BasicBlock*, 8> ExitBlocks;
  L->getExitBlocks(ExitBlocks);

  SmallSetVector<BasicBlock *, 8> ExitBlockSet(ExitBlocks.begin(),
                                               ExitBlocks.end());
  for (SmallSetVector<BasicBlock *, 8>::iterator I = ExitBlockSet.begin(),
         E = ExitBlockSet.end(); I != E; ++I) {
    BasicBlock *ExitBlock = *I;
    for (pred_iterator PI = pred_begin(ExitBlock), PE = pred_end(ExitBlock);
         PI != PE; ++PI)
      // Must be exactly this loop: no subloops, parent loops, or non-loop preds
      // allowed.
      if (!L->contains(*PI)) {
        if (rewriteLoopExitBlock(L, ExitBlock, AA, DT, LI, PP)) {
          ++NumInserted;
          Changed = true;
        }
        break;
      }
  }

  // If the header has more than two predecessors at this point (from the
  // preheader and from multiple backedges), we must adjust the loop.
  BasicBlock *LoopLatch = L->getLoopLatch();
  if (!LoopLatch) {
    // If this is really a nested loop, rip it out into a child loop.  Don't do
    // this for loops with a giant number of backedges, just factor them into a
    // common backedge instead.
    if (L->getNumBackEdges() < 8
        && false // HLSL Change - don't add nested loop.
        ) {
      if (Loop *OuterL =
              separateNestedLoop(L, Preheader, AA, DT, LI, SE, PP, AC)) {
        ++NumNested;
        // Enqueue the outer loop as it should be processed next in our
        // depth-first nest walk.
        Worklist.push_back(OuterL);

        // This is a big restructuring change, reprocess the whole loop.
        Changed = true;
        // GCC doesn't tail recursion eliminate this.
        // FIXME: It isn't clear we can't rely on LLVM to TRE this.
        goto ReprocessLoop;
      }
    }

    // If we either couldn't, or didn't want to, identify nesting of the loops,
    // insert a new block that all backedges target, then make it jump to the
    // loop header.
    LoopLatch = insertUniqueBackedgeBlock(L, Preheader, AA, DT, LI);
    if (LoopLatch) {
      ++NumInserted;
      Changed = true;
    }
  }

  const DataLayout &DL = L->getHeader()->getModule()->getDataLayout();

  // Scan over the PHI nodes in the loop header.  Since they now have only two
  // incoming values (the loop is canonicalized), we may have simplified the PHI
  // down to 'X = phi [X, Y]', which should be replaced with 'Y'.
  PHINode *PN;
  for (BasicBlock::iterator I = L->getHeader()->begin();
       (PN = dyn_cast<PHINode>(I++)); )
    if (Value *V = SimplifyInstruction(PN, DL, nullptr, DT, AC)) {
      if (AA) AA->deleteValue(PN);
      if (SE) SE->forgetValue(PN);
      PN->replaceAllUsesWith(V);
      PN->eraseFromParent();
    }

  // If this loop has multiple exits and the exits all go to the same
  // block, attempt to merge the exits. This helps several passes, such
  // as LoopRotation, which do not support loops with multiple exits.
  // SimplifyCFG also does this (and this code uses the same utility
  // function), however this code is loop-aware, where SimplifyCFG is
  // not. That gives it the advantage of being able to hoist
  // loop-invariant instructions out of the way to open up more
  // opportunities, and the disadvantage of having the responsibility
  // to preserve dominator information.
  bool UniqueExit = true;
  if (!ExitBlocks.empty())
    for (unsigned i = 1, e = ExitBlocks.size(); i != e; ++i)
      if (ExitBlocks[i] != ExitBlocks[0]) {
        UniqueExit = false;
        break;
      }
  if (UniqueExit) {
    for (unsigned i = 0, e = ExitingBlocks.size(); i != e; ++i) {
      BasicBlock *ExitingBlock = ExitingBlocks[i];
      if (!ExitingBlock->getSinglePredecessor()) continue;
      BranchInst *BI = dyn_cast<BranchInst>(ExitingBlock->getTerminator());
      if (!BI || !BI->isConditional()) continue;
      CmpInst *CI = dyn_cast<CmpInst>(BI->getCondition());
      if (!CI || CI->getParent() != ExitingBlock) continue;

      // Attempt to hoist out all instructions except for the
      // comparison and the branch.
      bool AllInvariant = true;
      bool AnyInvariant = false;
      for (BasicBlock::iterator I = ExitingBlock->begin(); &*I != BI; ) {
        Instruction *Inst = I++;
        // Skip debug info intrinsics.
        if (isa<DbgInfoIntrinsic>(Inst))
          continue;
        if (Inst == CI)
          continue;
        if (!L->makeLoopInvariant(Inst, AnyInvariant,
                                  Preheader ? Preheader->getTerminator()
                                            : nullptr)) {
          AllInvariant = false;
          break;
        }
      }
      if (AnyInvariant) {
        Changed = true;
        // The loop disposition of all SCEV expressions that depend on any
        // hoisted values have also changed.
        if (SE)
          SE->forgetLoopDispositions(L);
      }
      if (!AllInvariant) continue;

      // The block has now been cleared of all instructions except for
      // a comparison and a conditional branch. SimplifyCFG may be able
      // to fold it now.
      if (!FoldBranchToCommonDest(BI))
        continue;

      // Success. The block is now dead, so remove it from the loop,
      // update the dominator tree and delete it.
      DEBUG(dbgs() << "LoopSimplify: Eliminating exiting block "
                   << ExitingBlock->getName() << "\n");

      // Notify ScalarEvolution before deleting this block. Currently assume the
      // parent loop doesn't change (spliting edges doesn't count). If blocks,
      // CFG edges, or other values in the parent loop change, then we need call
      // to forgetLoop() for the parent instead.
      if (SE)
        SE->forgetLoop(L);

      assert(pred_begin(ExitingBlock) == pred_end(ExitingBlock));
      Changed = true;
      LI->removeBlock(ExitingBlock);

      DomTreeNode *Node = DT->getNode(ExitingBlock);
      const std::vector<DomTreeNodeBase<BasicBlock> *> &Children =
        Node->getChildren();
      while (!Children.empty()) {
        DomTreeNode *Child = Children.front();
        DT->changeImmediateDominator(Child, Node->getIDom());
      }
      DT->eraseNode(ExitingBlock);

      BI->getSuccessor(0)->removePredecessor(ExitingBlock);
      BI->getSuccessor(1)->removePredecessor(ExitingBlock);
      ExitingBlock->eraseFromParent();
    }
  }

  return Changed;
}

bool llvm::simplifyLoop(Loop *L, DominatorTree *DT, LoopInfo *LI, Pass *PP,
                        AliasAnalysis *AA, ScalarEvolution *SE,
                        AssumptionCache *AC) {
  bool Changed = false;

  // Worklist maintains our depth-first queue of loops in this nest to process.
  SmallVector<Loop *, 4> Worklist;
  Worklist.push_back(L);

  // Walk the worklist from front to back, pushing newly found sub loops onto
  // the back. This will let us process loops from back to front in depth-first
  // order. We can use this simple process because loops form a tree.
  for (unsigned Idx = 0; Idx != Worklist.size(); ++Idx) {
    Loop *L2 = Worklist[Idx];
    Worklist.append(L2->begin(), L2->end());
  }

  while (!Worklist.empty())
    Changed |= simplifyOneLoop(Worklist.pop_back_val(), Worklist, AA, DT, LI,
                               SE, PP, AC);

  return Changed;
}

INITIALIZE_PASS_BEGIN(LoopSimplify, "loop-simplify",
                "Canonicalize natural loops", false, false)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_END(LoopSimplify, "loop-simplify",
                "Canonicalize natural loops", false, false)

// Publicly exposed interface to pass...
Pass *llvm::createLoopSimplifyPass() { return new LoopSimplify(); }

LoopSimplify::LoopSimplify() : FunctionPass(ID) {
  initializeLoopSimplifyPass(*PassRegistry::getPassRegistry());
}

void LoopSimplify::getAnalysisUsage(AnalysisUsage& AU) const {
  AU.addRequired<AssumptionCacheTracker>();

  // We need loop information to identify the loops...
  AU.addRequired<DominatorTreeWrapperPass>();
  AU.addPreserved<DominatorTreeWrapperPass>();

  AU.addRequired<LoopInfoWrapperPass>();
  AU.addPreserved<LoopInfoWrapperPass>();

  AU.addPreserved<AliasAnalysis>();
  AU.addPreserved<ScalarEvolution>();
  AU.addPreserved<DependenceAnalysis>();
  AU.addPreservedID(BreakCriticalEdgesID);  // No critical edges added.
}

/// runOnFunction - Run down all loops in the CFG (recursively, but we could do
/// it in any convenient order) inserting preheaders...
///
bool LoopSimplify::runOnFunction(Function &F) {
  bool Changed = false;
  AA = getAnalysisIfAvailable<AliasAnalysis>();
  LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  SE = getAnalysisIfAvailable<ScalarEvolution>();
  AC = &getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);

  // Simplify each loop nest in the function.
  for (LoopInfo::iterator I = LI->begin(), E = LI->end(); I != E; ++I)
    Changed |= simplifyLoop(*I, DT, LI, this, AA, SE, AC);

  return Changed;
}

// FIXME: Restore this code when we re-enable verification in verifyAnalysis
// below.
#if 0
static void verifyLoop(Loop *L) {
  // Verify subloops.
  for (Loop::iterator I = L->begin(), E = L->end(); I != E; ++I)
    verifyLoop(*I);

  // It used to be possible to just assert L->isLoopSimplifyForm(), however
  // with the introduction of indirectbr, there are now cases where it's
  // not possible to transform a loop as necessary. We can at least check
  // that there is an indirectbr near any time there's trouble.

  // Indirectbr can interfere with preheader and unique backedge insertion.
  if (!L->getLoopPreheader() || !L->getLoopLatch()) {
    bool HasIndBrPred = false;
    for (pred_iterator PI = pred_begin(L->getHeader()),
         PE = pred_end(L->getHeader()); PI != PE; ++PI)
      if (isa<IndirectBrInst>((*PI)->getTerminator())) {
        HasIndBrPred = true;
        break;
      }
    assert(HasIndBrPred &&
           "LoopSimplify has no excuse for missing loop header info!");
    (void)HasIndBrPred;
  }

  // Indirectbr can interfere with exit block canonicalization.
  if (!L->hasDedicatedExits()) {
    bool HasIndBrExiting = false;
    SmallVector<BasicBlock*, 8> ExitingBlocks;
    L->getExitingBlocks(ExitingBlocks);
    for (unsigned i = 0, e = ExitingBlocks.size(); i != e; ++i) {
      if (isa<IndirectBrInst>((ExitingBlocks[i])->getTerminator())) {
        HasIndBrExiting = true;
        break;
      }
    }

    assert(HasIndBrExiting &&
           "LoopSimplify has no excuse for missing exit block info!");
    (void)HasIndBrExiting;
  }
}
#endif

void LoopSimplify::verifyAnalysis() const {
  // FIXME: This routine is being called mid-way through the loop pass manager
  // as loop passes destroy this analysis. That's actually fine, but we have no
  // way of expressing that here. Once all of the passes that destroy this are
  // hoisted out of the loop pass manager we can add back verification here.
#if 0
  for (LoopInfo::iterator I = LI->begin(), E = LI->end(); I != E; ++I)
    verifyLoop(*I);
#endif
}

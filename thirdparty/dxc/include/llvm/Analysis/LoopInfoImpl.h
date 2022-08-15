//===- llvm/Analysis/LoopInfoImpl.h - Natural Loop Calculator ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is the generic implementation of LoopInfo used for both Loops and
// MachineLoops.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_LOOPINFOIMPL_H
#define LLVM_ANALYSIS_LOOPINFOIMPL_H

#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Dominators.h"

namespace llvm {

//===----------------------------------------------------------------------===//
// APIs for simple analysis of the loop. See header notes.

/// getExitingBlocks - Return all blocks inside the loop that have successors
/// outside of the loop.  These are the blocks _inside of the current loop_
/// which branch out.  The returned list is always unique.
///
template<class BlockT, class LoopT>
void LoopBase<BlockT, LoopT>::
getExitingBlocks(SmallVectorImpl<BlockT *> &ExitingBlocks) const {
  typedef GraphTraits<BlockT*> BlockTraits;
  for (block_iterator BI = block_begin(), BE = block_end(); BI != BE; ++BI)
    for (typename BlockTraits::ChildIteratorType I =
           BlockTraits::child_begin(*BI), E = BlockTraits::child_end(*BI);
         I != E; ++I)
      if (!contains(*I)) {
        // Not in current loop? It must be an exit block.
        ExitingBlocks.push_back(*BI);
        break;
      }
}

/// getExitingBlock - If getExitingBlocks would return exactly one block,
/// return that block. Otherwise return null.
template<class BlockT, class LoopT>
BlockT *LoopBase<BlockT, LoopT>::getExitingBlock() const {
  SmallVector<BlockT*, 8> ExitingBlocks;
  getExitingBlocks(ExitingBlocks);
  if (ExitingBlocks.size() == 1)
    return ExitingBlocks[0];
  return nullptr;
}

/// getExitBlocks - Return all of the successor blocks of this loop.  These
/// are the blocks _outside of the current loop_ which are branched to.
///
template<class BlockT, class LoopT>
void LoopBase<BlockT, LoopT>::
getExitBlocks(SmallVectorImpl<BlockT*> &ExitBlocks) const {
  typedef GraphTraits<BlockT*> BlockTraits;
  for (block_iterator BI = block_begin(), BE = block_end(); BI != BE; ++BI)
    for (typename BlockTraits::ChildIteratorType I =
           BlockTraits::child_begin(*BI), E = BlockTraits::child_end(*BI);
         I != E; ++I)
      if (!contains(*I))
        // Not in current loop? It must be an exit block.
        ExitBlocks.push_back(*I);
}

/// getExitBlock - If getExitBlocks would return exactly one block,
/// return that block. Otherwise return null.
template<class BlockT, class LoopT>
BlockT *LoopBase<BlockT, LoopT>::getExitBlock() const {
  SmallVector<BlockT*, 8> ExitBlocks;
  getExitBlocks(ExitBlocks);
  if (ExitBlocks.size() == 1)
    return ExitBlocks[0];
  return nullptr;
}

/// getExitEdges - Return all pairs of (_inside_block_,_outside_block_).
template<class BlockT, class LoopT>
void LoopBase<BlockT, LoopT>::
getExitEdges(SmallVectorImpl<Edge> &ExitEdges) const {
  typedef GraphTraits<BlockT*> BlockTraits;
  for (block_iterator BI = block_begin(), BE = block_end(); BI != BE; ++BI)
    for (typename BlockTraits::ChildIteratorType I =
           BlockTraits::child_begin(*BI), E = BlockTraits::child_end(*BI);
         I != E; ++I)
      if (!contains(*I))
        // Not in current loop? It must be an exit block.
        ExitEdges.push_back(Edge(*BI, *I));
}

/// getLoopPreheader - If there is a preheader for this loop, return it.  A
/// loop has a preheader if there is only one edge to the header of the loop
/// from outside of the loop.  If this is the case, the block branching to the
/// header of the loop is the preheader node.
///
/// This method returns null if there is no preheader for the loop.
///
template<class BlockT, class LoopT>
BlockT *LoopBase<BlockT, LoopT>::getLoopPreheader() const {
  // Keep track of nodes outside the loop branching to the header...
  BlockT *Out = getLoopPredecessor();
  if (!Out) return nullptr;

  // Make sure there is only one exit out of the preheader.
  typedef GraphTraits<BlockT*> BlockTraits;
  typename BlockTraits::ChildIteratorType SI = BlockTraits::child_begin(Out);
  ++SI;
  if (SI != BlockTraits::child_end(Out))
    return nullptr;  // Multiple exits from the block, must not be a preheader.

  // The predecessor has exactly one successor, so it is a preheader.
  return Out;
}

/// getLoopPredecessor - If the given loop's header has exactly one unique
/// predecessor outside the loop, return it. Otherwise return null.
/// This is less strict that the loop "preheader" concept, which requires
/// the predecessor to have exactly one successor.
///
template<class BlockT, class LoopT>
BlockT *LoopBase<BlockT, LoopT>::getLoopPredecessor() const {
  // Keep track of nodes outside the loop branching to the header...
  BlockT *Out = nullptr;

  // Loop over the predecessors of the header node...
  BlockT *Header = getHeader();
  typedef GraphTraits<Inverse<BlockT*> > InvBlockTraits;
  for (typename InvBlockTraits::ChildIteratorType PI =
         InvBlockTraits::child_begin(Header),
         PE = InvBlockTraits::child_end(Header); PI != PE; ++PI) {
    typename InvBlockTraits::NodeType *N = *PI;
    if (!contains(N)) {     // If the block is not in the loop...
      if (Out && Out != N)
        return nullptr;     // Multiple predecessors outside the loop
      Out = N;
    }
  }

  // Make sure there is only one exit out of the preheader.
  assert(Out && "Header of loop has no predecessors from outside loop?");
  return Out;
}

/// getLoopLatch - If there is a single latch block for this loop, return it.
/// A latch block is a block that contains a branch back to the header.
template<class BlockT, class LoopT>
BlockT *LoopBase<BlockT, LoopT>::getLoopLatch() const {
  BlockT *Header = getHeader();
  typedef GraphTraits<Inverse<BlockT*> > InvBlockTraits;
  typename InvBlockTraits::ChildIteratorType PI =
    InvBlockTraits::child_begin(Header);
  typename InvBlockTraits::ChildIteratorType PE =
    InvBlockTraits::child_end(Header);
  BlockT *Latch = nullptr;
  for (; PI != PE; ++PI) {
    typename InvBlockTraits::NodeType *N = *PI;
    if (contains(N)) {
      if (Latch) return nullptr;
      Latch = N;
    }
  }

  return Latch;
}

//===----------------------------------------------------------------------===//
// APIs for updating loop information after changing the CFG
//

/// addBasicBlockToLoop - This method is used by other analyses to update loop
/// information.  NewBB is set to be a new member of the current loop.
/// Because of this, it is added as a member of all parent loops, and is added
/// to the specified LoopInfo object as being in the current basic block.  It
/// is not valid to replace the loop header with this method.
///
template<class BlockT, class LoopT>
void LoopBase<BlockT, LoopT>::
addBasicBlockToLoop(BlockT *NewBB, LoopInfoBase<BlockT, LoopT> &LIB) {
  assert((Blocks.empty() || LIB[getHeader()] == this) &&
         "Incorrect LI specified for this loop!");
  assert(NewBB && "Cannot add a null basic block to the loop!");
  assert(!LIB[NewBB] && "BasicBlock already in the loop!");

  LoopT *L = static_cast<LoopT *>(this);

  // Add the loop mapping to the LoopInfo object...
  LIB.BBMap[NewBB] = L;

  // Add the basic block to this loop and all parent loops...
  while (L) {
    L->addBlockEntry(NewBB);
    L = L->getParentLoop();
  }
}

/// replaceChildLoopWith - This is used when splitting loops up.  It replaces
/// the OldChild entry in our children list with NewChild, and updates the
/// parent pointer of OldChild to be null and the NewChild to be this loop.
/// This updates the loop depth of the new child.
template<class BlockT, class LoopT>
void LoopBase<BlockT, LoopT>::
replaceChildLoopWith(LoopT *OldChild, LoopT *NewChild) {
  assert(OldChild->ParentLoop == this && "This loop is already broken!");
  assert(!NewChild->ParentLoop && "NewChild already has a parent!");
  typename std::vector<LoopT *>::iterator I =
    std::find(SubLoops.begin(), SubLoops.end(), OldChild);
  assert(I != SubLoops.end() && "OldChild not in loop!");
  *I = NewChild;
  OldChild->ParentLoop = nullptr;
  NewChild->ParentLoop = static_cast<LoopT *>(this);
}

/// verifyLoop - Verify loop structure
template<class BlockT, class LoopT>
void LoopBase<BlockT, LoopT>::verifyLoop() const {
#ifndef NDEBUG
  assert(!Blocks.empty() && "Loop header is missing");

  // Setup for using a depth-first iterator to visit every block in the loop.
  SmallVector<BlockT*, 8> ExitBBs;
  getExitBlocks(ExitBBs);
  llvm::SmallPtrSet<BlockT*, 8> VisitSet;
  VisitSet.insert(ExitBBs.begin(), ExitBBs.end());
  df_ext_iterator<BlockT*, llvm::SmallPtrSet<BlockT*, 8> >
    BI = df_ext_begin(getHeader(), VisitSet),
    BE = df_ext_end(getHeader(), VisitSet);

  // Keep track of the number of BBs visited.
  unsigned NumVisited = 0;

  // Check the individual blocks.
  for ( ; BI != BE; ++BI) {
    BlockT *BB = *BI;
    bool HasInsideLoopSuccs = false;
    bool HasInsideLoopPreds = false;
    SmallVector<BlockT *, 2> OutsideLoopPreds;

    typedef GraphTraits<BlockT*> BlockTraits;
    for (typename BlockTraits::ChildIteratorType SI =
           BlockTraits::child_begin(BB), SE = BlockTraits::child_end(BB);
         SI != SE; ++SI)
      if (contains(*SI)) {
        HasInsideLoopSuccs = true;
        break;
      }
    typedef GraphTraits<Inverse<BlockT*> > InvBlockTraits;
    for (typename InvBlockTraits::ChildIteratorType PI =
           InvBlockTraits::child_begin(BB), PE = InvBlockTraits::child_end(BB);
         PI != PE; ++PI) {
      BlockT *N = *PI;
      if (contains(N))
        HasInsideLoopPreds = true;
      else
        OutsideLoopPreds.push_back(N);
    }

    if (BB == getHeader()) {
        assert(!OutsideLoopPreds.empty() && "Loop is unreachable!");
    } else if (!OutsideLoopPreds.empty()) {
      // A non-header loop shouldn't be reachable from outside the loop,
      // though it is permitted if the predecessor is not itself actually
      // reachable.
      BlockT *EntryBB = BB->getParent()->begin();
      for (BlockT *CB : depth_first(EntryBB))
        for (unsigned i = 0, e = OutsideLoopPreds.size(); i != e; ++i)
          assert(CB != OutsideLoopPreds[i] &&
                 "Loop has multiple entry points!");
    }
    assert(HasInsideLoopPreds && "Loop block has no in-loop predecessors!");
    assert(HasInsideLoopSuccs && "Loop block has no in-loop successors!");
    assert(BB != getHeader()->getParent()->begin() &&
           "Loop contains function entry block!");

    NumVisited++;
  }

  assert(NumVisited == getNumBlocks() && "Unreachable block in loop");

  // Check the subloops.
  for (iterator I = begin(), E = end(); I != E; ++I)
    // Each block in each subloop should be contained within this loop.
    for (block_iterator BI = (*I)->block_begin(), BE = (*I)->block_end();
         BI != BE; ++BI) {
        assert(contains(*BI) &&
               "Loop does not contain all the blocks of a subloop!");
    }

  // Check the parent loop pointer.
  if (ParentLoop) {
    assert(std::find(ParentLoop->begin(), ParentLoop->end(), this) !=
           ParentLoop->end() &&
           "Loop is not a subloop of its parent!");
  }
#endif
}

/// verifyLoop - Verify loop structure of this loop and all nested loops.
template<class BlockT, class LoopT>
void LoopBase<BlockT, LoopT>::verifyLoopNest(
  DenseSet<const LoopT*> *Loops) const {
  Loops->insert(static_cast<const LoopT *>(this));
  // Verify this loop.
  verifyLoop();
  // Verify the subloops.
  for (iterator I = begin(), E = end(); I != E; ++I)
    (*I)->verifyLoopNest(Loops);
}

template<class BlockT, class LoopT>
void LoopBase<BlockT, LoopT>::print(raw_ostream &OS, unsigned Depth) const {
  OS.indent(Depth*2) << "Loop at depth " << getLoopDepth()
       << " containing: ";

  for (unsigned i = 0; i < getBlocks().size(); ++i) {
    if (i) OS << ",";
    BlockT *BB = getBlocks()[i];
    BB->printAsOperand(OS, false);
    if (BB == getHeader())    OS << "<header>";
    if (BB == getLoopLatch()) OS << "<latch>";
    if (isLoopExiting(BB))    OS << "<exiting>";
  }
  OS << "\n";

  for (iterator I = begin(), E = end(); I != E; ++I)
    (*I)->print(OS, Depth+2);
}
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
/// Stable LoopInfo Analysis - Build a loop tree using stable iterators so the
/// result does / not depend on use list (block predecessor) order.
///

/// Discover a subloop with the specified backedges such that: All blocks within
/// this loop are mapped to this loop or a subloop. And all subloops within this
/// loop have their parent loop set to this loop or a subloop.
template<class BlockT, class LoopT>
static void discoverAndMapSubloop(LoopT *L, ArrayRef<BlockT*> Backedges,
                                  LoopInfoBase<BlockT, LoopT> *LI,
                                  DominatorTreeBase<BlockT> &DomTree) {
  typedef GraphTraits<Inverse<BlockT*> > InvBlockTraits;

  unsigned NumBlocks = 0;
  unsigned NumSubloops = 0;

  // Perform a backward CFG traversal using a worklist.
  std::vector<BlockT *> ReverseCFGWorklist(Backedges.begin(), Backedges.end());
  while (!ReverseCFGWorklist.empty()) {
    BlockT *PredBB = ReverseCFGWorklist.back();
    ReverseCFGWorklist.pop_back();

    LoopT *Subloop = LI->getLoopFor(PredBB);
    if (!Subloop) {
      if (!DomTree.isReachableFromEntry(PredBB))
        continue;

      // This is an undiscovered block. Map it to the current loop.
      LI->changeLoopFor(PredBB, L);
      ++NumBlocks;
      if (PredBB == L->getHeader())
          continue;
      // Push all block predecessors on the worklist.
      ReverseCFGWorklist.insert(ReverseCFGWorklist.end(),
                                InvBlockTraits::child_begin(PredBB),
                                InvBlockTraits::child_end(PredBB));
    }
    else {
      // This is a discovered block. Find its outermost discovered loop.
      while (LoopT *Parent = Subloop->getParentLoop())
        Subloop = Parent;

      // If it is already discovered to be a subloop of this loop, continue.
      if (Subloop == L)
        continue;

      // Discover a subloop of this loop.
      Subloop->setParentLoop(L);
      ++NumSubloops;
      NumBlocks += Subloop->getBlocks().capacity();
      PredBB = Subloop->getHeader();
      // Continue traversal along predecessors that are not loop-back edges from
      // within this subloop tree itself. Note that a predecessor may directly
      // reach another subloop that is not yet discovered to be a subloop of
      // this loop, which we must traverse.
      for (typename InvBlockTraits::ChildIteratorType PI =
             InvBlockTraits::child_begin(PredBB),
             PE = InvBlockTraits::child_end(PredBB); PI != PE; ++PI) {
        if (LI->getLoopFor(*PI) != Subloop)
          ReverseCFGWorklist.push_back(*PI);
      }
    }
  }
  L->getSubLoopsVector().reserve(NumSubloops);
  L->reserveBlocks(NumBlocks);
}

/// Populate all loop data in a stable order during a single forward DFS.
template<class BlockT, class LoopT>
class PopulateLoopsDFS {
  typedef GraphTraits<BlockT*> BlockTraits;
  typedef typename BlockTraits::ChildIteratorType SuccIterTy;

  LoopInfoBase<BlockT, LoopT> *LI;
public:
  PopulateLoopsDFS(LoopInfoBase<BlockT, LoopT> *li):
    LI(li) {}

  void traverse(BlockT *EntryBlock);

protected:
  void insertIntoLoop(BlockT *Block);
};

/// Top-level driver for the forward DFS within the loop.
template<class BlockT, class LoopT>
void PopulateLoopsDFS<BlockT, LoopT>::traverse(BlockT *EntryBlock) {
  for (BlockT *BB : post_order(EntryBlock))
    insertIntoLoop(BB);
}

/// Add a single Block to its ancestor loops in PostOrder. If the block is a
/// subloop header, add the subloop to its parent in PostOrder, then reverse the
/// Block and Subloop vectors of the now complete subloop to achieve RPO.
template<class BlockT, class LoopT>
void PopulateLoopsDFS<BlockT, LoopT>::insertIntoLoop(BlockT *Block) {
  LoopT *Subloop = LI->getLoopFor(Block);
  if (Subloop && Block == Subloop->getHeader()) {
    // We reach this point once per subloop after processing all the blocks in
    // the subloop.
    if (Subloop->getParentLoop())
      Subloop->getParentLoop()->getSubLoopsVector().push_back(Subloop);
    else
      LI->addTopLevelLoop(Subloop);

    // For convenience, Blocks and Subloops are inserted in postorder. Reverse
    // the lists, except for the loop header, which is always at the beginning.
    Subloop->reverseBlock(1);
    std::reverse(Subloop->getSubLoopsVector().begin(),
                 Subloop->getSubLoopsVector().end());

    Subloop = Subloop->getParentLoop();
  }
  for (; Subloop; Subloop = Subloop->getParentLoop())
    Subloop->addBlockEntry(Block);
}

/// Analyze LoopInfo discovers loops during a postorder DominatorTree traversal
/// interleaved with backward CFG traversals within each subloop
/// (discoverAndMapSubloop). The backward traversal skips inner subloops, so
/// this part of the algorithm is linear in the number of CFG edges. Subloop and
/// Block vectors are then populated during a single forward CFG traversal
/// (PopulateLoopDFS).
///
/// During the two CFG traversals each block is seen three times:
/// 1) Discovered and mapped by a reverse CFG traversal.
/// 2) Visited during a forward DFS CFG traversal.
/// 3) Reverse-inserted in the loop in postorder following forward DFS.
///
/// The Block vectors are inclusive, so step 3 requires loop-depth number of
/// insertions per block.
template<class BlockT, class LoopT>
void LoopInfoBase<BlockT, LoopT>::
Analyze(DominatorTreeBase<BlockT> &DomTree) {

  // Postorder traversal of the dominator tree.
  DomTreeNodeBase<BlockT>* DomRoot = DomTree.getRootNode();
  for (auto DomNode : post_order(DomRoot)) {

    BlockT *Header = DomNode->getBlock();
    SmallVector<BlockT *, 4> Backedges;

    // Check each predecessor of the potential loop header.
    typedef GraphTraits<Inverse<BlockT*> > InvBlockTraits;
    for (typename InvBlockTraits::ChildIteratorType PI =
           InvBlockTraits::child_begin(Header),
           PE = InvBlockTraits::child_end(Header); PI != PE; ++PI) {

      BlockT *Backedge = *PI;

      // If Header dominates predBB, this is a new loop. Collect the backedges.
      if (DomTree.dominates(Header, Backedge)
          && DomTree.isReachableFromEntry(Backedge)) {
        Backedges.push_back(Backedge);
      }
    }
    // Perform a backward CFG traversal to discover and map blocks in this loop.
    if (!Backedges.empty()) {
      LoopT *L = new LoopT(Header);
      discoverAndMapSubloop(L, ArrayRef<BlockT*>(Backedges), this, DomTree);
    }
  }
  // Perform a single forward CFG traversal to populate block and subloop
  // vectors for all loops.
  PopulateLoopsDFS<BlockT, LoopT> DFS(this);
  DFS.traverse(DomRoot->getBlock());
}

// Debugging
template<class BlockT, class LoopT>
void LoopInfoBase<BlockT, LoopT>::print(raw_ostream &OS) const {
  for (unsigned i = 0; i < TopLevelLoops.size(); ++i)
    TopLevelLoops[i]->print(OS);
#if 0
  for (DenseMap<BasicBlock*, LoopT*>::const_iterator I = BBMap.begin(),
         E = BBMap.end(); I != E; ++I)
    OS << "BB '" << I->first->getName() << "' level = "
       << I->second->getLoopDepth() << "\n";
#endif
}

template<class BlockT, class LoopT>
void LoopInfoBase<BlockT, LoopT>::verify() const {
  DenseSet<const LoopT*> Loops;
  for (iterator I = begin(), E = end(); I != E; ++I) {
    assert(!(*I)->getParentLoop() && "Top-level loop has a parent!");
    (*I)->verifyLoopNest(&Loops);
  }

  // Verify that blocks are mapped to valid loops.
#ifndef NDEBUG
  for (auto &Entry : BBMap) {
    const BlockT *BB = Entry.first;
    LoopT *L = Entry.second;
    assert(Loops.count(L) && "orphaned loop");
    assert(L->contains(BB) && "orphaned block");
  }
#endif
}

} // End llvm namespace

#endif

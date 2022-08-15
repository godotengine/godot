//===-- MachineBlockPlacement.cpp - Basic Block Code Layout optimization --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements basic block placement transformations using the CFG
// structure and branch probability estimates.
//
// The pass strives to preserve the structure of the CFG (that is, retain
// a topological ordering of basic blocks) in the absence of a *strong* signal
// to the contrary from probabilities. However, within the CFG structure, it
// attempts to choose an ordering which favors placing more likely sequences of
// blocks adjacent to each other.
//
// The algorithm works from the inner-most loop within a function outward, and
// at each stage walks through the basic blocks, trying to coalesce them into
// sequential chains where allowed by the CFG (or demanded by heavy
// probabilities). Finally, it walks the blocks in topological order, and the
// first time it reaches a chain of basic blocks, it schedules them in the
// function in-order.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineBranchProbabilityInfo.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include <algorithm>
using namespace llvm;

#define DEBUG_TYPE "block-placement"

STATISTIC(NumCondBranches, "Number of conditional branches");
STATISTIC(NumUncondBranches, "Number of uncondittional branches");
STATISTIC(CondBranchTakenFreq,
          "Potential frequency of taking conditional branches");
STATISTIC(UncondBranchTakenFreq,
          "Potential frequency of taking unconditional branches");

static cl::opt<unsigned> AlignAllBlock("align-all-blocks",
                                       cl::desc("Force the alignment of all "
                                                "blocks in the function."),
                                       cl::init(0), cl::Hidden);

// FIXME: Find a good default for this flag and remove the flag.
static cl::opt<unsigned> ExitBlockBias(
    "block-placement-exit-block-bias",
    cl::desc("Block frequency percentage a loop exit block needs "
             "over the original exit to be considered the new exit."),
    cl::init(0), cl::Hidden);

static cl::opt<bool> OutlineOptionalBranches(
    "outline-optional-branches",
    cl::desc("Put completely optional branches, i.e. branches with a common "
             "post dominator, out of line."),
    cl::init(false), cl::Hidden);

static cl::opt<unsigned> OutlineOptionalThreshold(
    "outline-optional-threshold",
    cl::desc("Don't outline optional branches that are a single block with an "
             "instruction count below this threshold"),
    cl::init(4), cl::Hidden);

namespace {
class BlockChain;
/// \brief Type for our function-wide basic block -> block chain mapping.
typedef DenseMap<MachineBasicBlock *, BlockChain *> BlockToChainMapType;
}

namespace {
/// \brief A chain of blocks which will be laid out contiguously.
///
/// This is the datastructure representing a chain of consecutive blocks that
/// are profitable to layout together in order to maximize fallthrough
/// probabilities and code locality. We also can use a block chain to represent
/// a sequence of basic blocks which have some external (correctness)
/// requirement for sequential layout.
///
/// Chains can be built around a single basic block and can be merged to grow
/// them. They participate in a block-to-chain mapping, which is updated
/// automatically as chains are merged together.
class BlockChain {
  /// \brief The sequence of blocks belonging to this chain.
  ///
  /// This is the sequence of blocks for a particular chain. These will be laid
  /// out in-order within the function.
  SmallVector<MachineBasicBlock *, 4> Blocks;

  /// \brief A handle to the function-wide basic block to block chain mapping.
  ///
  /// This is retained in each block chain to simplify the computation of child
  /// block chains for SCC-formation and iteration. We store the edges to child
  /// basic blocks, and map them back to their associated chains using this
  /// structure.
  BlockToChainMapType &BlockToChain;

public:
  /// \brief Construct a new BlockChain.
  ///
  /// This builds a new block chain representing a single basic block in the
  /// function. It also registers itself as the chain that block participates
  /// in with the BlockToChain mapping.
  BlockChain(BlockToChainMapType &BlockToChain, MachineBasicBlock *BB)
      : Blocks(1, BB), BlockToChain(BlockToChain), LoopPredecessors(0) {
    assert(BB && "Cannot create a chain with a null basic block");
    BlockToChain[BB] = this;
  }

  /// \brief Iterator over blocks within the chain.
  typedef SmallVectorImpl<MachineBasicBlock *>::iterator iterator;

  /// \brief Beginning of blocks within the chain.
  iterator begin() { return Blocks.begin(); }

  /// \brief End of blocks within the chain.
  iterator end() { return Blocks.end(); }

  /// \brief Merge a block chain into this one.
  ///
  /// This routine merges a block chain into this one. It takes care of forming
  /// a contiguous sequence of basic blocks, updating the edge list, and
  /// updating the block -> chain mapping. It does not free or tear down the
  /// old chain, but the old chain's block list is no longer valid.
  void merge(MachineBasicBlock *BB, BlockChain *Chain) {
    assert(BB);
    assert(!Blocks.empty());

    // Fast path in case we don't have a chain already.
    if (!Chain) {
      assert(!BlockToChain[BB]);
      Blocks.push_back(BB);
      BlockToChain[BB] = this;
      return;
    }

    assert(BB == *Chain->begin());
    assert(Chain->begin() != Chain->end());

    // Update the incoming blocks to point to this chain, and add them to the
    // chain structure.
    for (MachineBasicBlock *ChainBB : *Chain) {
      Blocks.push_back(ChainBB);
      assert(BlockToChain[ChainBB] == Chain && "Incoming blocks not in chain");
      BlockToChain[ChainBB] = this;
    }
  }

#ifndef NDEBUG
  /// \brief Dump the blocks in this chain.
  LLVM_DUMP_METHOD void dump() {
    for (MachineBasicBlock *MBB : *this)
      MBB->dump();
  }
#endif // NDEBUG

  /// \brief Count of predecessors within the loop currently being processed.
  ///
  /// This count is updated at each loop we process to represent the number of
  /// in-loop predecessors of this chain.
  unsigned LoopPredecessors;
};
}

namespace {
class MachineBlockPlacement : public MachineFunctionPass {
  /// \brief A typedef for a block filter set.
  typedef SmallPtrSet<MachineBasicBlock *, 16> BlockFilterSet;

  /// \brief A handle to the branch probability pass.
  const MachineBranchProbabilityInfo *MBPI;

  /// \brief A handle to the function-wide block frequency pass.
  const MachineBlockFrequencyInfo *MBFI;

  /// \brief A handle to the loop info.
  const MachineLoopInfo *MLI;

  /// \brief A handle to the target's instruction info.
  const TargetInstrInfo *TII;

  /// \brief A handle to the target's lowering info.
  const TargetLoweringBase *TLI;

  /// \brief A handle to the post dominator tree.
  MachineDominatorTree *MDT;

  /// \brief A set of blocks that are unavoidably execute, i.e. they dominate
  /// all terminators of the MachineFunction.
  SmallPtrSet<MachineBasicBlock *, 4> UnavoidableBlocks;

  /// \brief Allocator and owner of BlockChain structures.
  ///
  /// We build BlockChains lazily while processing the loop structure of
  /// a function. To reduce malloc traffic, we allocate them using this
  /// slab-like allocator, and destroy them after the pass completes. An
  /// important guarantee is that this allocator produces stable pointers to
  /// the chains.
  SpecificBumpPtrAllocator<BlockChain> ChainAllocator;

  /// \brief Function wide BasicBlock to BlockChain mapping.
  ///
  /// This mapping allows efficiently moving from any given basic block to the
  /// BlockChain it participates in, if any. We use it to, among other things,
  /// allow implicitly defining edges between chains as the existing edges
  /// between basic blocks.
  DenseMap<MachineBasicBlock *, BlockChain *> BlockToChain;

  void markChainSuccessors(BlockChain &Chain, MachineBasicBlock *LoopHeaderBB,
                           SmallVectorImpl<MachineBasicBlock *> &BlockWorkList,
                           const BlockFilterSet *BlockFilter = nullptr);
  MachineBasicBlock *selectBestSuccessor(MachineBasicBlock *BB,
                                         BlockChain &Chain,
                                         const BlockFilterSet *BlockFilter);
  MachineBasicBlock *
  selectBestCandidateBlock(BlockChain &Chain,
                           SmallVectorImpl<MachineBasicBlock *> &WorkList,
                           const BlockFilterSet *BlockFilter);
  MachineBasicBlock *
  getFirstUnplacedBlock(MachineFunction &F, const BlockChain &PlacedChain,
                        MachineFunction::iterator &PrevUnplacedBlockIt,
                        const BlockFilterSet *BlockFilter);
  void buildChain(MachineBasicBlock *BB, BlockChain &Chain,
                  SmallVectorImpl<MachineBasicBlock *> &BlockWorkList,
                  const BlockFilterSet *BlockFilter = nullptr);
  MachineBasicBlock *findBestLoopTop(MachineLoop &L,
                                     const BlockFilterSet &LoopBlockSet);
  MachineBasicBlock *findBestLoopExit(MachineFunction &F, MachineLoop &L,
                                      const BlockFilterSet &LoopBlockSet);
  void buildLoopChains(MachineFunction &F, MachineLoop &L);
  void rotateLoop(BlockChain &LoopChain, MachineBasicBlock *ExitingBB,
                  const BlockFilterSet &LoopBlockSet);
  void buildCFGChains(MachineFunction &F);

public:
  static char ID; // Pass identification, replacement for typeid
  MachineBlockPlacement() : MachineFunctionPass(ID) {
    initializeMachineBlockPlacementPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &F) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineBranchProbabilityInfo>();
    AU.addRequired<MachineBlockFrequencyInfo>();
    AU.addRequired<MachineDominatorTree>();
    AU.addRequired<MachineLoopInfo>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};
}

char MachineBlockPlacement::ID = 0;
char &llvm::MachineBlockPlacementID = MachineBlockPlacement::ID;
INITIALIZE_PASS_BEGIN(MachineBlockPlacement, "block-placement",
                      "Branch Probability Basic Block Placement", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineBranchProbabilityInfo)
INITIALIZE_PASS_DEPENDENCY(MachineBlockFrequencyInfo)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTree)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfo)
INITIALIZE_PASS_END(MachineBlockPlacement, "block-placement",
                    "Branch Probability Basic Block Placement", false, false)

#ifndef NDEBUG
/// \brief Helper to print the name of a MBB.
///
/// Only used by debug logging.
static std::string getBlockName(MachineBasicBlock *BB) {
  std::string Result;
  raw_string_ostream OS(Result);
  OS << "BB#" << BB->getNumber();
  OS << " (derived from LLVM BB '" << BB->getName() << "')";
  OS.flush();
  return Result;
}

/// \brief Helper to print the number of a MBB.
///
/// Only used by debug logging.
static std::string getBlockNum(MachineBasicBlock *BB) {
  std::string Result;
  raw_string_ostream OS(Result);
  OS << "BB#" << BB->getNumber();
  OS.flush();
  return Result;
}
#endif

/// \brief Mark a chain's successors as having one fewer preds.
///
/// When a chain is being merged into the "placed" chain, this routine will
/// quickly walk the successors of each block in the chain and mark them as
/// having one fewer active predecessor. It also adds any successors of this
/// chain which reach the zero-predecessor state to the worklist passed in.
void MachineBlockPlacement::markChainSuccessors(
    BlockChain &Chain, MachineBasicBlock *LoopHeaderBB,
    SmallVectorImpl<MachineBasicBlock *> &BlockWorkList,
    const BlockFilterSet *BlockFilter) {
  // Walk all the blocks in this chain, marking their successors as having
  // a predecessor placed.
  for (MachineBasicBlock *MBB : Chain) {
    // Add any successors for which this is the only un-placed in-loop
    // predecessor to the worklist as a viable candidate for CFG-neutral
    // placement. No subsequent placement of this block will violate the CFG
    // shape, so we get to use heuristics to choose a favorable placement.
    for (MachineBasicBlock *Succ : MBB->successors()) {
      if (BlockFilter && !BlockFilter->count(Succ))
        continue;
      BlockChain &SuccChain = *BlockToChain[Succ];
      // Disregard edges within a fixed chain, or edges to the loop header.
      if (&Chain == &SuccChain || Succ == LoopHeaderBB)
        continue;

      // This is a cross-chain edge that is within the loop, so decrement the
      // loop predecessor count of the destination chain.
      if (SuccChain.LoopPredecessors > 0 && --SuccChain.LoopPredecessors == 0)
        BlockWorkList.push_back(*SuccChain.begin());
    }
  }
}

/// \brief Select the best successor for a block.
///
/// This looks across all successors of a particular block and attempts to
/// select the "best" one to be the layout successor. It only considers direct
/// successors which also pass the block filter. It will attempt to avoid
/// breaking CFG structure, but cave and break such structures in the case of
/// very hot successor edges.
///
/// \returns The best successor block found, or null if none are viable.
MachineBasicBlock *
MachineBlockPlacement::selectBestSuccessor(MachineBasicBlock *BB,
                                           BlockChain &Chain,
                                           const BlockFilterSet *BlockFilter) {
  const BranchProbability HotProb(4, 5); // 80%

  MachineBasicBlock *BestSucc = nullptr;
  // FIXME: Due to the performance of the probability and weight routines in
  // the MBPI analysis, we manually compute probabilities using the edge
  // weights. This is suboptimal as it means that the somewhat subtle
  // definition of edge weight semantics is encoded here as well. We should
  // improve the MBPI interface to efficiently support query patterns such as
  // this.
  uint32_t BestWeight = 0;
  uint32_t WeightScale = 0;
  uint32_t SumWeight = MBPI->getSumForBlock(BB, WeightScale);
  DEBUG(dbgs() << "Attempting merge from: " << getBlockName(BB) << "\n");
  for (MachineBasicBlock *Succ : BB->successors()) {
    if (BlockFilter && !BlockFilter->count(Succ))
      continue;
    BlockChain &SuccChain = *BlockToChain[Succ];
    if (&SuccChain == &Chain) {
      DEBUG(dbgs() << "    " << getBlockName(Succ) << " -> Already merged!\n");
      continue;
    }
    if (Succ != *SuccChain.begin()) {
      DEBUG(dbgs() << "    " << getBlockName(Succ) << " -> Mid chain!\n");
      continue;
    }

    uint32_t SuccWeight = MBPI->getEdgeWeight(BB, Succ);
    BranchProbability SuccProb(SuccWeight / WeightScale, SumWeight);

    // If we outline optional branches, look whether Succ is unavoidable, i.e.
    // dominates all terminators of the MachineFunction. If it does, other
    // successors must be optional. Don't do this for cold branches.
    if (OutlineOptionalBranches && SuccProb > HotProb.getCompl() &&
        UnavoidableBlocks.count(Succ) > 0) {
      auto HasShortOptionalBranch = [&]() {
        for (MachineBasicBlock *Pred : Succ->predecessors()) {
          // Check whether there is an unplaced optional branch.
          if (Pred == Succ || (BlockFilter && !BlockFilter->count(Pred)) ||
              BlockToChain[Pred] == &Chain)
            continue;
          // Check whether the optional branch has exactly one BB.
          if (Pred->pred_size() > 1 || *Pred->pred_begin() != BB)
            continue;
          // Check whether the optional branch is small.
          if (Pred->size() < OutlineOptionalThreshold)
            return true;
        }
        return false;
      };
      if (!HasShortOptionalBranch())
        return Succ;
    }

    // Only consider successors which are either "hot", or wouldn't violate
    // any CFG constraints.
    if (SuccChain.LoopPredecessors != 0) {
      if (SuccProb < HotProb) {
        DEBUG(dbgs() << "    " << getBlockName(Succ) << " -> " << SuccProb
                     << " (prob) (CFG conflict)\n");
        continue;
      }

      // Make sure that a hot successor doesn't have a globally more
      // important predecessor.
      BlockFrequency CandidateEdgeFreq =
          MBFI->getBlockFreq(BB) * SuccProb * HotProb.getCompl();
      bool BadCFGConflict = false;
      for (MachineBasicBlock *Pred : Succ->predecessors()) {
        if (Pred == Succ || (BlockFilter && !BlockFilter->count(Pred)) ||
            BlockToChain[Pred] == &Chain)
          continue;
        BlockFrequency PredEdgeFreq =
            MBFI->getBlockFreq(Pred) * MBPI->getEdgeProbability(Pred, Succ);
        if (PredEdgeFreq >= CandidateEdgeFreq) {
          BadCFGConflict = true;
          break;
        }
      }
      if (BadCFGConflict) {
        DEBUG(dbgs() << "    " << getBlockName(Succ) << " -> " << SuccProb
                     << " (prob) (non-cold CFG conflict)\n");
        continue;
      }
    }

    DEBUG(dbgs() << "    " << getBlockName(Succ) << " -> " << SuccProb
                 << " (prob)"
                 << (SuccChain.LoopPredecessors != 0 ? " (CFG break)" : "")
                 << "\n");
    if (BestSucc && BestWeight >= SuccWeight)
      continue;
    BestSucc = Succ;
    BestWeight = SuccWeight;
  }
  return BestSucc;
}

/// \brief Select the best block from a worklist.
///
/// This looks through the provided worklist as a list of candidate basic
/// blocks and select the most profitable one to place. The definition of
/// profitable only really makes sense in the context of a loop. This returns
/// the most frequently visited block in the worklist, which in the case of
/// a loop, is the one most desirable to be physically close to the rest of the
/// loop body in order to improve icache behavior.
///
/// \returns The best block found, or null if none are viable.
MachineBasicBlock *MachineBlockPlacement::selectBestCandidateBlock(
    BlockChain &Chain, SmallVectorImpl<MachineBasicBlock *> &WorkList,
    const BlockFilterSet *BlockFilter) {
  // Once we need to walk the worklist looking for a candidate, cleanup the
  // worklist of already placed entries.
  // FIXME: If this shows up on profiles, it could be folded (at the cost of
  // some code complexity) into the loop below.
  WorkList.erase(std::remove_if(WorkList.begin(), WorkList.end(),
                                [&](MachineBasicBlock *BB) {
                                  return BlockToChain.lookup(BB) == &Chain;
                                }),
                 WorkList.end());

  MachineBasicBlock *BestBlock = nullptr;
  BlockFrequency BestFreq;
  for (MachineBasicBlock *MBB : WorkList) {
    BlockChain &SuccChain = *BlockToChain[MBB];
    if (&SuccChain == &Chain) {
      DEBUG(dbgs() << "    " << getBlockName(MBB) << " -> Already merged!\n");
      continue;
    }
    assert(SuccChain.LoopPredecessors == 0 && "Found CFG-violating block");

    BlockFrequency CandidateFreq = MBFI->getBlockFreq(MBB);
    DEBUG(dbgs() << "    " << getBlockName(MBB) << " -> ";
          MBFI->printBlockFreq(dbgs(), CandidateFreq) << " (freq)\n");
    if (BestBlock && BestFreq >= CandidateFreq)
      continue;
    BestBlock = MBB;
    BestFreq = CandidateFreq;
  }
  return BestBlock;
}

/// \brief Retrieve the first unplaced basic block.
///
/// This routine is called when we are unable to use the CFG to walk through
/// all of the basic blocks and form a chain due to unnatural loops in the CFG.
/// We walk through the function's blocks in order, starting from the
/// LastUnplacedBlockIt. We update this iterator on each call to avoid
/// re-scanning the entire sequence on repeated calls to this routine.
MachineBasicBlock *MachineBlockPlacement::getFirstUnplacedBlock(
    MachineFunction &F, const BlockChain &PlacedChain,
    MachineFunction::iterator &PrevUnplacedBlockIt,
    const BlockFilterSet *BlockFilter) {
  for (MachineFunction::iterator I = PrevUnplacedBlockIt, E = F.end(); I != E;
       ++I) {
    if (BlockFilter && !BlockFilter->count(I))
      continue;
    if (BlockToChain[I] != &PlacedChain) {
      PrevUnplacedBlockIt = I;
      // Now select the head of the chain to which the unplaced block belongs
      // as the block to place. This will force the entire chain to be placed,
      // and satisfies the requirements of merging chains.
      return *BlockToChain[I]->begin();
    }
  }
  return nullptr;
}

void MachineBlockPlacement::buildChain(
    MachineBasicBlock *BB, BlockChain &Chain,
    SmallVectorImpl<MachineBasicBlock *> &BlockWorkList,
    const BlockFilterSet *BlockFilter) {
  assert(BB);
  assert(BlockToChain[BB] == &Chain);
  MachineFunction &F = *BB->getParent();
  MachineFunction::iterator PrevUnplacedBlockIt = F.begin();

  MachineBasicBlock *LoopHeaderBB = BB;
  markChainSuccessors(Chain, LoopHeaderBB, BlockWorkList, BlockFilter);
  BB = *std::prev(Chain.end());
  for (;;) {
    assert(BB);
    assert(BlockToChain[BB] == &Chain);
    assert(*std::prev(Chain.end()) == BB);

    // Look for the best viable successor if there is one to place immediately
    // after this block.
    MachineBasicBlock *BestSucc = selectBestSuccessor(BB, Chain, BlockFilter);

    // If an immediate successor isn't available, look for the best viable
    // block among those we've identified as not violating the loop's CFG at
    // this point. This won't be a fallthrough, but it will increase locality.
    if (!BestSucc)
      BestSucc = selectBestCandidateBlock(Chain, BlockWorkList, BlockFilter);

    if (!BestSucc) {
      BestSucc =
          getFirstUnplacedBlock(F, Chain, PrevUnplacedBlockIt, BlockFilter);
      if (!BestSucc)
        break;

      DEBUG(dbgs() << "Unnatural loop CFG detected, forcibly merging the "
                      "layout successor until the CFG reduces\n");
    }

    // Place this block, updating the datastructures to reflect its placement.
    BlockChain &SuccChain = *BlockToChain[BestSucc];
    // Zero out LoopPredecessors for the successor we're about to merge in case
    // we selected a successor that didn't fit naturally into the CFG.
    SuccChain.LoopPredecessors = 0;
    DEBUG(dbgs() << "Merging from " << getBlockNum(BB) << " to "
                 << getBlockNum(BestSucc) << "\n");
    markChainSuccessors(SuccChain, LoopHeaderBB, BlockWorkList, BlockFilter);
    Chain.merge(BestSucc, &SuccChain);
    BB = *std::prev(Chain.end());
  }

  DEBUG(dbgs() << "Finished forming chain for header block "
               << getBlockNum(*Chain.begin()) << "\n");
}

/// \brief Find the best loop top block for layout.
///
/// Look for a block which is strictly better than the loop header for laying
/// out at the top of the loop. This looks for one and only one pattern:
/// a latch block with no conditional exit. This block will cause a conditional
/// jump around it or will be the bottom of the loop if we lay it out in place,
/// but if it it doesn't end up at the bottom of the loop for any reason,
/// rotation alone won't fix it. Because such a block will always result in an
/// unconditional jump (for the backedge) rotating it in front of the loop
/// header is always profitable.
MachineBasicBlock *
MachineBlockPlacement::findBestLoopTop(MachineLoop &L,
                                       const BlockFilterSet &LoopBlockSet) {
  // Check that the header hasn't been fused with a preheader block due to
  // crazy branches. If it has, we need to start with the header at the top to
  // prevent pulling the preheader into the loop body.
  BlockChain &HeaderChain = *BlockToChain[L.getHeader()];
  if (!LoopBlockSet.count(*HeaderChain.begin()))
    return L.getHeader();

  DEBUG(dbgs() << "Finding best loop top for: " << getBlockName(L.getHeader())
               << "\n");

  BlockFrequency BestPredFreq;
  MachineBasicBlock *BestPred = nullptr;
  for (MachineBasicBlock *Pred : L.getHeader()->predecessors()) {
    if (!LoopBlockSet.count(Pred))
      continue;
    DEBUG(dbgs() << "    header pred: " << getBlockName(Pred) << ", "
                 << Pred->succ_size() << " successors, ";
          MBFI->printBlockFreq(dbgs(), Pred) << " freq\n");
    if (Pred->succ_size() > 1)
      continue;

    BlockFrequency PredFreq = MBFI->getBlockFreq(Pred);
    if (!BestPred || PredFreq > BestPredFreq ||
        (!(PredFreq < BestPredFreq) &&
         Pred->isLayoutSuccessor(L.getHeader()))) {
      BestPred = Pred;
      BestPredFreq = PredFreq;
    }
  }

  // If no direct predecessor is fine, just use the loop header.
  if (!BestPred)
    return L.getHeader();

  // Walk backwards through any straight line of predecessors.
  while (BestPred->pred_size() == 1 &&
         (*BestPred->pred_begin())->succ_size() == 1 &&
         *BestPred->pred_begin() != L.getHeader())
    BestPred = *BestPred->pred_begin();

  DEBUG(dbgs() << "    final top: " << getBlockName(BestPred) << "\n");
  return BestPred;
}

/// \brief Find the best loop exiting block for layout.
///
/// This routine implements the logic to analyze the loop looking for the best
/// block to layout at the top of the loop. Typically this is done to maximize
/// fallthrough opportunities.
MachineBasicBlock *
MachineBlockPlacement::findBestLoopExit(MachineFunction &F, MachineLoop &L,
                                        const BlockFilterSet &LoopBlockSet) {
  // We don't want to layout the loop linearly in all cases. If the loop header
  // is just a normal basic block in the loop, we want to look for what block
  // within the loop is the best one to layout at the top. However, if the loop
  // header has be pre-merged into a chain due to predecessors not having
  // analyzable branches, *and* the predecessor it is merged with is *not* part
  // of the loop, rotating the header into the middle of the loop will create
  // a non-contiguous range of blocks which is Very Bad. So start with the
  // header and only rotate if safe.
  BlockChain &HeaderChain = *BlockToChain[L.getHeader()];
  if (!LoopBlockSet.count(*HeaderChain.begin()))
    return nullptr;

  BlockFrequency BestExitEdgeFreq;
  unsigned BestExitLoopDepth = 0;
  MachineBasicBlock *ExitingBB = nullptr;
  // If there are exits to outer loops, loop rotation can severely limit
  // fallthrough opportunites unless it selects such an exit. Keep a set of
  // blocks where rotating to exit with that block will reach an outer loop.
  SmallPtrSet<MachineBasicBlock *, 4> BlocksExitingToOuterLoop;

  DEBUG(dbgs() << "Finding best loop exit for: " << getBlockName(L.getHeader())
               << "\n");
  for (MachineBasicBlock *MBB : L.getBlocks()) {
    BlockChain &Chain = *BlockToChain[MBB];
    // Ensure that this block is at the end of a chain; otherwise it could be
    // mid-way through an inner loop or a successor of an unanalyzable branch.
    if (MBB != *std::prev(Chain.end()))
      continue;

    // Now walk the successors. We need to establish whether this has a viable
    // exiting successor and whether it has a viable non-exiting successor.
    // We store the old exiting state and restore it if a viable looping
    // successor isn't found.
    MachineBasicBlock *OldExitingBB = ExitingBB;
    BlockFrequency OldBestExitEdgeFreq = BestExitEdgeFreq;
    bool HasLoopingSucc = false;
    // FIXME: Due to the performance of the probability and weight routines in
    // the MBPI analysis, we use the internal weights and manually compute the
    // probabilities to avoid quadratic behavior.
    uint32_t WeightScale = 0;
    uint32_t SumWeight = MBPI->getSumForBlock(MBB, WeightScale);
    for (MachineBasicBlock *Succ : MBB->successors()) {
      if (Succ->isLandingPad())
        continue;
      if (Succ == MBB)
        continue;
      BlockChain &SuccChain = *BlockToChain[Succ];
      // Don't split chains, either this chain or the successor's chain.
      if (&Chain == &SuccChain) {
        DEBUG(dbgs() << "    exiting: " << getBlockName(MBB) << " -> "
                     << getBlockName(Succ) << " (chain conflict)\n");
        continue;
      }

      uint32_t SuccWeight = MBPI->getEdgeWeight(MBB, Succ);
      if (LoopBlockSet.count(Succ)) {
        DEBUG(dbgs() << "    looping: " << getBlockName(MBB) << " -> "
                     << getBlockName(Succ) << " (" << SuccWeight << ")\n");
        HasLoopingSucc = true;
        continue;
      }

      unsigned SuccLoopDepth = 0;
      if (MachineLoop *ExitLoop = MLI->getLoopFor(Succ)) {
        SuccLoopDepth = ExitLoop->getLoopDepth();
        if (ExitLoop->contains(&L))
          BlocksExitingToOuterLoop.insert(MBB);
      }

      BranchProbability SuccProb(SuccWeight / WeightScale, SumWeight);
      BlockFrequency ExitEdgeFreq = MBFI->getBlockFreq(MBB) * SuccProb;
      DEBUG(dbgs() << "    exiting: " << getBlockName(MBB) << " -> "
                   << getBlockName(Succ) << " [L:" << SuccLoopDepth << "] (";
            MBFI->printBlockFreq(dbgs(), ExitEdgeFreq) << ")\n");
      // Note that we bias this toward an existing layout successor to retain
      // incoming order in the absence of better information. The exit must have
      // a frequency higher than the current exit before we consider breaking
      // the layout.
      BranchProbability Bias(100 - ExitBlockBias, 100);
      if (!ExitingBB || SuccLoopDepth > BestExitLoopDepth ||
          ExitEdgeFreq > BestExitEdgeFreq ||
          (MBB->isLayoutSuccessor(Succ) &&
           !(ExitEdgeFreq < BestExitEdgeFreq * Bias))) {
        BestExitEdgeFreq = ExitEdgeFreq;
        ExitingBB = MBB;
      }
    }

    if (!HasLoopingSucc) {
      // Restore the old exiting state, no viable looping successor was found.
      ExitingBB = OldExitingBB;
      BestExitEdgeFreq = OldBestExitEdgeFreq;
      continue;
    }
  }
  // Without a candidate exiting block or with only a single block in the
  // loop, just use the loop header to layout the loop.
  if (!ExitingBB || L.getNumBlocks() == 1)
    return nullptr;

  // Also, if we have exit blocks which lead to outer loops but didn't select
  // one of them as the exiting block we are rotating toward, disable loop
  // rotation altogether.
  if (!BlocksExitingToOuterLoop.empty() &&
      !BlocksExitingToOuterLoop.count(ExitingBB))
    return nullptr;

  DEBUG(dbgs() << "  Best exiting block: " << getBlockName(ExitingBB) << "\n");
  return ExitingBB;
}

/// \brief Attempt to rotate an exiting block to the bottom of the loop.
///
/// Once we have built a chain, try to rotate it to line up the hot exit block
/// with fallthrough out of the loop if doing so doesn't introduce unnecessary
/// branches. For example, if the loop has fallthrough into its header and out
/// of its bottom already, don't rotate it.
void MachineBlockPlacement::rotateLoop(BlockChain &LoopChain,
                                       MachineBasicBlock *ExitingBB,
                                       const BlockFilterSet &LoopBlockSet) {
  if (!ExitingBB)
    return;

  MachineBasicBlock *Top = *LoopChain.begin();
  bool ViableTopFallthrough = false;
  for (MachineBasicBlock *Pred : Top->predecessors()) {
    BlockChain *PredChain = BlockToChain[Pred];
    if (!LoopBlockSet.count(Pred) &&
        (!PredChain || Pred == *std::prev(PredChain->end()))) {
      ViableTopFallthrough = true;
      break;
    }
  }

  // If the header has viable fallthrough, check whether the current loop
  // bottom is a viable exiting block. If so, bail out as rotating will
  // introduce an unnecessary branch.
  if (ViableTopFallthrough) {
    MachineBasicBlock *Bottom = *std::prev(LoopChain.end());
    for (MachineBasicBlock *Succ : Bottom->successors()) {
      BlockChain *SuccChain = BlockToChain[Succ];
      if (!LoopBlockSet.count(Succ) &&
          (!SuccChain || Succ == *SuccChain->begin()))
        return;
    }
  }

  BlockChain::iterator ExitIt =
      std::find(LoopChain.begin(), LoopChain.end(), ExitingBB);
  if (ExitIt == LoopChain.end())
    return;

  std::rotate(LoopChain.begin(), std::next(ExitIt), LoopChain.end());
}

/// \brief Forms basic block chains from the natural loop structures.
///
/// These chains are designed to preserve the existing *structure* of the code
/// as much as possible. We can then stitch the chains together in a way which
/// both preserves the topological structure and minimizes taken conditional
/// branches.
void MachineBlockPlacement::buildLoopChains(MachineFunction &F,
                                            MachineLoop &L) {
  // First recurse through any nested loops, building chains for those inner
  // loops.
  for (MachineLoop *InnerLoop : L)
    buildLoopChains(F, *InnerLoop);

  SmallVector<MachineBasicBlock *, 16> BlockWorkList;
  BlockFilterSet LoopBlockSet(L.block_begin(), L.block_end());

  // First check to see if there is an obviously preferable top block for the
  // loop. This will default to the header, but may end up as one of the
  // predecessors to the header if there is one which will result in strictly
  // fewer branches in the loop body.
  MachineBasicBlock *LoopTop = findBestLoopTop(L, LoopBlockSet);

  // If we selected just the header for the loop top, look for a potentially
  // profitable exit block in the event that rotating the loop can eliminate
  // branches by placing an exit edge at the bottom.
  MachineBasicBlock *ExitingBB = nullptr;
  if (LoopTop == L.getHeader())
    ExitingBB = findBestLoopExit(F, L, LoopBlockSet);

  BlockChain &LoopChain = *BlockToChain[LoopTop];

  // FIXME: This is a really lame way of walking the chains in the loop: we
  // walk the blocks, and use a set to prevent visiting a particular chain
  // twice.
  SmallPtrSet<BlockChain *, 4> UpdatedPreds;
  assert(LoopChain.LoopPredecessors == 0);
  UpdatedPreds.insert(&LoopChain);
  for (MachineBasicBlock *LoopBB : L.getBlocks()) {
    BlockChain &Chain = *BlockToChain[LoopBB];
    if (!UpdatedPreds.insert(&Chain).second)
      continue;

    assert(Chain.LoopPredecessors == 0);
    for (MachineBasicBlock *ChainBB : Chain) {
      assert(BlockToChain[ChainBB] == &Chain);
      for (MachineBasicBlock *Pred : ChainBB->predecessors()) {
        if (BlockToChain[Pred] == &Chain || !LoopBlockSet.count(Pred))
          continue;
        ++Chain.LoopPredecessors;
      }
    }

    if (Chain.LoopPredecessors == 0)
      BlockWorkList.push_back(*Chain.begin());
  }

  buildChain(LoopTop, LoopChain, BlockWorkList, &LoopBlockSet);
  rotateLoop(LoopChain, ExitingBB, LoopBlockSet);

  DEBUG({
    // Crash at the end so we get all of the debugging output first.
    bool BadLoop = false;
    if (LoopChain.LoopPredecessors) {
      BadLoop = true;
      dbgs() << "Loop chain contains a block without its preds placed!\n"
             << "  Loop header:  " << getBlockName(*L.block_begin()) << "\n"
             << "  Chain header: " << getBlockName(*LoopChain.begin()) << "\n";
    }
    for (MachineBasicBlock *ChainBB : LoopChain) {
      dbgs() << "          ... " << getBlockName(ChainBB) << "\n";
      if (!LoopBlockSet.erase(ChainBB)) {
        // We don't mark the loop as bad here because there are real situations
        // where this can occur. For example, with an unanalyzable fallthrough
        // from a loop block to a non-loop block or vice versa.
        dbgs() << "Loop chain contains a block not contained by the loop!\n"
               << "  Loop header:  " << getBlockName(*L.block_begin()) << "\n"
               << "  Chain header: " << getBlockName(*LoopChain.begin()) << "\n"
               << "  Bad block:    " << getBlockName(ChainBB) << "\n";
      }
    }

    if (!LoopBlockSet.empty()) {
      BadLoop = true;
      for (MachineBasicBlock *LoopBB : LoopBlockSet)
        dbgs() << "Loop contains blocks never placed into a chain!\n"
               << "  Loop header:  " << getBlockName(*L.block_begin()) << "\n"
               << "  Chain header: " << getBlockName(*LoopChain.begin()) << "\n"
               << "  Bad block:    " << getBlockName(LoopBB) << "\n";
    }
    assert(!BadLoop && "Detected problems with the placement of this loop.");
  });
}

void MachineBlockPlacement::buildCFGChains(MachineFunction &F) {
  // Ensure that every BB in the function has an associated chain to simplify
  // the assumptions of the remaining algorithm.
  SmallVector<MachineOperand, 4> Cond; // For AnalyzeBranch.
  for (MachineFunction::iterator FI = F.begin(), FE = F.end(); FI != FE; ++FI) {
    MachineBasicBlock *BB = FI;
    BlockChain *Chain =
        new (ChainAllocator.Allocate()) BlockChain(BlockToChain, BB);
    // Also, merge any blocks which we cannot reason about and must preserve
    // the exact fallthrough behavior for.
    for (;;) {
      Cond.clear();
      MachineBasicBlock *TBB = nullptr, *FBB = nullptr; // For AnalyzeBranch.
      if (!TII->AnalyzeBranch(*BB, TBB, FBB, Cond) || !FI->canFallThrough())
        break;

      MachineFunction::iterator NextFI(std::next(FI));
      MachineBasicBlock *NextBB = NextFI;
      // Ensure that the layout successor is a viable block, as we know that
      // fallthrough is a possibility.
      assert(NextFI != FE && "Can't fallthrough past the last block.");
      DEBUG(dbgs() << "Pre-merging due to unanalyzable fallthrough: "
                   << getBlockName(BB) << " -> " << getBlockName(NextBB)
                   << "\n");
      Chain->merge(NextBB, nullptr);
      FI = NextFI;
      BB = NextBB;
    }
  }

  if (OutlineOptionalBranches) {
    // Find the nearest common dominator of all of F's terminators.
    MachineBasicBlock *Terminator = nullptr;
    for (MachineBasicBlock &MBB : F) {
      if (MBB.succ_size() == 0) {
        if (Terminator == nullptr)
          Terminator = &MBB;
        else
          Terminator = MDT->findNearestCommonDominator(Terminator, &MBB);
      }
    }

    // MBBs dominating this common dominator are unavoidable.
    UnavoidableBlocks.clear();
    for (MachineBasicBlock &MBB : F) {
      if (MDT->dominates(&MBB, Terminator)) {
        UnavoidableBlocks.insert(&MBB);
      }
    }
  }

  // Build any loop-based chains.
  for (MachineLoop *L : *MLI)
    buildLoopChains(F, *L);

  SmallVector<MachineBasicBlock *, 16> BlockWorkList;

  SmallPtrSet<BlockChain *, 4> UpdatedPreds;
  for (MachineBasicBlock &MBB : F) {
    BlockChain &Chain = *BlockToChain[&MBB];
    if (!UpdatedPreds.insert(&Chain).second)
      continue;

    assert(Chain.LoopPredecessors == 0);
    for (MachineBasicBlock *ChainBB : Chain) {
      assert(BlockToChain[ChainBB] == &Chain);
      for (MachineBasicBlock *Pred : ChainBB->predecessors()) {
        if (BlockToChain[Pred] == &Chain)
          continue;
        ++Chain.LoopPredecessors;
      }
    }

    if (Chain.LoopPredecessors == 0)
      BlockWorkList.push_back(*Chain.begin());
  }

  BlockChain &FunctionChain = *BlockToChain[&F.front()];
  buildChain(&F.front(), FunctionChain, BlockWorkList);

#ifndef NDEBUG
  typedef SmallPtrSet<MachineBasicBlock *, 16> FunctionBlockSetType;
#endif
  DEBUG({
    // Crash at the end so we get all of the debugging output first.
    bool BadFunc = false;
    FunctionBlockSetType FunctionBlockSet;
    for (MachineBasicBlock &MBB : F)
      FunctionBlockSet.insert(&MBB);

    for (MachineBasicBlock *ChainBB : FunctionChain)
      if (!FunctionBlockSet.erase(ChainBB)) {
        BadFunc = true;
        dbgs() << "Function chain contains a block not in the function!\n"
               << "  Bad block:    " << getBlockName(ChainBB) << "\n";
      }

    if (!FunctionBlockSet.empty()) {
      BadFunc = true;
      for (MachineBasicBlock *RemainingBB : FunctionBlockSet)
        dbgs() << "Function contains blocks never placed into a chain!\n"
               << "  Bad block:    " << getBlockName(RemainingBB) << "\n";
    }
    assert(!BadFunc && "Detected problems with the block placement.");
  });

  // Splice the blocks into place.
  MachineFunction::iterator InsertPos = F.begin();
  for (MachineBasicBlock *ChainBB : FunctionChain) {
    DEBUG(dbgs() << (ChainBB == *FunctionChain.begin() ? "Placing chain "
                                                       : "          ... ")
                 << getBlockName(ChainBB) << "\n");
    if (InsertPos != MachineFunction::iterator(ChainBB))
      F.splice(InsertPos, ChainBB);
    else
      ++InsertPos;

    // Update the terminator of the previous block.
    if (ChainBB == *FunctionChain.begin())
      continue;
    MachineBasicBlock *PrevBB = std::prev(MachineFunction::iterator(ChainBB));

    // FIXME: It would be awesome of updateTerminator would just return rather
    // than assert when the branch cannot be analyzed in order to remove this
    // boiler plate.
    Cond.clear();
    MachineBasicBlock *TBB = nullptr, *FBB = nullptr; // For AnalyzeBranch.
    if (!TII->AnalyzeBranch(*PrevBB, TBB, FBB, Cond)) {
      // The "PrevBB" is not yet updated to reflect current code layout, so,
      //   o. it may fall-through to a block without explict "goto" instruction
      //      before layout, and no longer fall-through it after layout; or
      //   o. just opposite.
      //
      // AnalyzeBranch() may return erroneous value for FBB when these two
      // situations take place. For the first scenario FBB is mistakenly set
      // NULL; for the 2nd scenario, the FBB, which is expected to be NULL,
      // is mistakenly pointing to "*BI".
      //
      bool needUpdateBr = true;
      if (!Cond.empty() && (!FBB || FBB == ChainBB)) {
        PrevBB->updateTerminator();
        needUpdateBr = false;
        Cond.clear();
        TBB = FBB = nullptr;
        if (TII->AnalyzeBranch(*PrevBB, TBB, FBB, Cond)) {
          // FIXME: This should never take place.
          TBB = FBB = nullptr;
        }
      }

      // If PrevBB has a two-way branch, try to re-order the branches
      // such that we branch to the successor with higher weight first.
      if (TBB && !Cond.empty() && FBB &&
          MBPI->getEdgeWeight(PrevBB, FBB) > MBPI->getEdgeWeight(PrevBB, TBB) &&
          !TII->ReverseBranchCondition(Cond)) {
        DEBUG(dbgs() << "Reverse order of the two branches: "
                     << getBlockName(PrevBB) << "\n");
        DEBUG(dbgs() << "    Edge weight: " << MBPI->getEdgeWeight(PrevBB, FBB)
                     << " vs " << MBPI->getEdgeWeight(PrevBB, TBB) << "\n");
        DebugLoc dl; // FIXME: this is nowhere
        TII->RemoveBranch(*PrevBB);
        TII->InsertBranch(*PrevBB, FBB, TBB, Cond, dl);
        needUpdateBr = true;
      }
      if (needUpdateBr)
        PrevBB->updateTerminator();
    }
  }

  // Fixup the last block.
  Cond.clear();
  MachineBasicBlock *TBB = nullptr, *FBB = nullptr; // For AnalyzeBranch.
  if (!TII->AnalyzeBranch(F.back(), TBB, FBB, Cond))
    F.back().updateTerminator();

  // Walk through the backedges of the function now that we have fully laid out
  // the basic blocks and align the destination of each backedge. We don't rely
  // exclusively on the loop info here so that we can align backedges in
  // unnatural CFGs and backedges that were introduced purely because of the
  // loop rotations done during this layout pass.
  if (F.getFunction()->hasFnAttribute(Attribute::OptimizeForSize))
    return;
  if (FunctionChain.begin() == FunctionChain.end())
    return; // Empty chain.

  const BranchProbability ColdProb(1, 5); // 20%
  BlockFrequency EntryFreq = MBFI->getBlockFreq(F.begin());
  BlockFrequency WeightedEntryFreq = EntryFreq * ColdProb;
  for (MachineBasicBlock *ChainBB : FunctionChain) {
    if (ChainBB == *FunctionChain.begin())
      continue;

    // Don't align non-looping basic blocks. These are unlikely to execute
    // enough times to matter in practice. Note that we'll still handle
    // unnatural CFGs inside of a natural outer loop (the common case) and
    // rotated loops.
    MachineLoop *L = MLI->getLoopFor(ChainBB);
    if (!L)
      continue;

    unsigned Align = TLI->getPrefLoopAlignment(L);
    if (!Align)
      continue; // Don't care about loop alignment.

    // If the block is cold relative to the function entry don't waste space
    // aligning it.
    BlockFrequency Freq = MBFI->getBlockFreq(ChainBB);
    if (Freq < WeightedEntryFreq)
      continue;

    // If the block is cold relative to its loop header, don't align it
    // regardless of what edges into the block exist.
    MachineBasicBlock *LoopHeader = L->getHeader();
    BlockFrequency LoopHeaderFreq = MBFI->getBlockFreq(LoopHeader);
    if (Freq < (LoopHeaderFreq * ColdProb))
      continue;

    // Check for the existence of a non-layout predecessor which would benefit
    // from aligning this block.
    MachineBasicBlock *LayoutPred =
        &*std::prev(MachineFunction::iterator(ChainBB));

    // Force alignment if all the predecessors are jumps. We already checked
    // that the block isn't cold above.
    if (!LayoutPred->isSuccessor(ChainBB)) {
      ChainBB->setAlignment(Align);
      continue;
    }

    // Align this block if the layout predecessor's edge into this block is
    // cold relative to the block. When this is true, other predecessors make up
    // all of the hot entries into the block and thus alignment is likely to be
    // important.
    BranchProbability LayoutProb =
        MBPI->getEdgeProbability(LayoutPred, ChainBB);
    BlockFrequency LayoutEdgeFreq = MBFI->getBlockFreq(LayoutPred) * LayoutProb;
    if (LayoutEdgeFreq <= (Freq * ColdProb))
      ChainBB->setAlignment(Align);
  }
}

bool MachineBlockPlacement::runOnMachineFunction(MachineFunction &F) {
  // Check for single-block functions and skip them.
  if (std::next(F.begin()) == F.end())
    return false;

  if (skipOptnoneFunction(*F.getFunction()))
    return false;

  MBPI = &getAnalysis<MachineBranchProbabilityInfo>();
  MBFI = &getAnalysis<MachineBlockFrequencyInfo>();
  MLI = &getAnalysis<MachineLoopInfo>();
  TII = F.getSubtarget().getInstrInfo();
  TLI = F.getSubtarget().getTargetLowering();
  MDT = &getAnalysis<MachineDominatorTree>();
  assert(BlockToChain.empty());

  buildCFGChains(F);

  BlockToChain.clear();
  ChainAllocator.DestroyAll();

  if (AlignAllBlock)
    // Align all of the blocks in the function to a specific alignment.
    for (MachineBasicBlock &MBB : F)
      MBB.setAlignment(AlignAllBlock);

  // We always return true as we have no way to track whether the final order
  // differs from the original order.
  return true;
}

namespace {
/// \brief A pass to compute block placement statistics.
///
/// A separate pass to compute interesting statistics for evaluating block
/// placement. This is separate from the actual placement pass so that they can
/// be computed in the absence of any placement transformations or when using
/// alternative placement strategies.
class MachineBlockPlacementStats : public MachineFunctionPass {
  /// \brief A handle to the branch probability pass.
  const MachineBranchProbabilityInfo *MBPI;

  /// \brief A handle to the function-wide block frequency pass.
  const MachineBlockFrequencyInfo *MBFI;

public:
  static char ID; // Pass identification, replacement for typeid
  MachineBlockPlacementStats() : MachineFunctionPass(ID) {
    initializeMachineBlockPlacementStatsPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &F) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineBranchProbabilityInfo>();
    AU.addRequired<MachineBlockFrequencyInfo>();
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};
}

char MachineBlockPlacementStats::ID = 0;
char &llvm::MachineBlockPlacementStatsID = MachineBlockPlacementStats::ID;
INITIALIZE_PASS_BEGIN(MachineBlockPlacementStats, "block-placement-stats",
                      "Basic Block Placement Stats", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineBranchProbabilityInfo)
INITIALIZE_PASS_DEPENDENCY(MachineBlockFrequencyInfo)
INITIALIZE_PASS_END(MachineBlockPlacementStats, "block-placement-stats",
                    "Basic Block Placement Stats", false, false)

bool MachineBlockPlacementStats::runOnMachineFunction(MachineFunction &F) {
  // Check for single-block functions and skip them.
  if (std::next(F.begin()) == F.end())
    return false;

  MBPI = &getAnalysis<MachineBranchProbabilityInfo>();
  MBFI = &getAnalysis<MachineBlockFrequencyInfo>();

  for (MachineBasicBlock &MBB : F) {
    BlockFrequency BlockFreq = MBFI->getBlockFreq(&MBB);
    Statistic &NumBranches =
        (MBB.succ_size() > 1) ? NumCondBranches : NumUncondBranches;
    Statistic &BranchTakenFreq =
        (MBB.succ_size() > 1) ? CondBranchTakenFreq : UncondBranchTakenFreq;
    for (MachineBasicBlock *Succ : MBB.successors()) {
      // Skip if this successor is a fallthrough.
      if (MBB.isLayoutSuccessor(Succ))
        continue;

      BlockFrequency EdgeFreq =
          BlockFreq * MBPI->getEdgeProbability(&MBB, Succ);
      ++NumBranches;
      BranchTakenFreq += EdgeFreq.getFrequency();
    }
  }

  return false;
}


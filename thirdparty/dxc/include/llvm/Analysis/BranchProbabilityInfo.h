//===--- BranchProbabilityInfo.h - Branch Probability Analysis --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass is used to evaluate branch probabilties.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_BRANCHPROBABILITYINFO_H
#define LLVM_ANALYSIS_BRANCHPROBABILITYINFO_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/CFG.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/BranchProbability.h"

namespace llvm {
class LoopInfo;
class raw_ostream;

/// \brief Analysis pass providing branch probability information.
///
/// This is a function analysis pass which provides information on the relative
/// probabilities of each "edge" in the function's CFG where such an edge is
/// defined by a pair (PredBlock and an index in the successors). The
/// probability of an edge from one block is always relative to the
/// probabilities of other edges from the block. The probabilites of all edges
/// from a block sum to exactly one (100%).
/// We use a pair (PredBlock and an index in the successors) to uniquely
/// identify an edge, since we can have multiple edges from Src to Dst.
/// As an example, we can have a switch which jumps to Dst with value 0 and
/// value 10.
class BranchProbabilityInfo : public FunctionPass {
public:
  static char ID;

  BranchProbabilityInfo() : FunctionPass(ID) {
    initializeBranchProbabilityInfoPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override;
  bool runOnFunction(Function &F) override;

  void releaseMemory() override;

  void print(raw_ostream &OS, const Module *M = nullptr) const override;

  /// \brief Get an edge's probability, relative to other out-edges of the Src.
  ///
  /// This routine provides access to the fractional probability between zero
  /// (0%) and one (100%) of this edge executing, relative to other edges
  /// leaving the 'Src' block. The returned probability is never zero, and can
  /// only be one if the source block has only one successor.
  BranchProbability getEdgeProbability(const BasicBlock *Src,
                                       unsigned IndexInSuccessors) const;

  /// \brief Get the probability of going from Src to Dst.
  ///
  /// It returns the sum of all probabilities for edges from Src to Dst.
  BranchProbability getEdgeProbability(const BasicBlock *Src,
                                       const BasicBlock *Dst) const;

  /// \brief Test if an edge is hot relative to other out-edges of the Src.
  ///
  /// Check whether this edge out of the source block is 'hot'. We define hot
  /// as having a relative probability >= 80%.
  bool isEdgeHot(const BasicBlock *Src, const BasicBlock *Dst) const;

  /// \brief Retrieve the hot successor of a block if one exists.
  ///
  /// Given a basic block, look through its successors and if one exists for
  /// which \see isEdgeHot would return true, return that successor block.
  BasicBlock *getHotSucc(BasicBlock *BB) const;

  /// \brief Print an edge's probability.
  ///
  /// Retrieves an edge's probability similarly to \see getEdgeProbability, but
  /// then prints that probability to the provided stream. That stream is then
  /// returned.
  raw_ostream &printEdgeProbability(raw_ostream &OS, const BasicBlock *Src,
                                    const BasicBlock *Dst) const;

  /// \brief Get the raw edge weight calculated for the edge.
  ///
  /// This returns the raw edge weight. It is guaranteed to fall between 1 and
  /// UINT32_MAX. Note that the raw edge weight is not meaningful in isolation.
  /// This interface should be very carefully, and primarily by routines that
  /// are updating the analysis by later calling setEdgeWeight.
  uint32_t getEdgeWeight(const BasicBlock *Src,
                         unsigned IndexInSuccessors) const;

  /// \brief Get the raw edge weight calculated for the block pair.
  ///
  /// This returns the sum of all raw edge weights from Src to Dst.
  /// It is guaranteed to fall between 1 and UINT32_MAX.
  uint32_t getEdgeWeight(const BasicBlock *Src, const BasicBlock *Dst) const;

  uint32_t getEdgeWeight(const BasicBlock *Src,
                         succ_const_iterator Dst) const;

  /// \brief Set the raw edge weight for a given edge.
  ///
  /// This allows a pass to explicitly set the edge weight for an edge. It can
  /// be used when updating the CFG to update and preserve the branch
  /// probability information. Read the implementation of how these edge
  /// weights are calculated carefully before using!
  void setEdgeWeight(const BasicBlock *Src, unsigned IndexInSuccessors,
                     uint32_t Weight);

  static uint32_t getBranchWeightStackProtector(bool IsLikely) {
    return IsLikely ? (1u << 20) - 1 : 1;
  }

private:
  // Since we allow duplicate edges from one basic block to another, we use
  // a pair (PredBlock and an index in the successors) to specify an edge.
  typedef std::pair<const BasicBlock *, unsigned> Edge;

  // Default weight value. Used when we don't have information about the edge.
  // TODO: DEFAULT_WEIGHT makes sense during static predication, when none of
  // the successors have a weight yet. But it doesn't make sense when providing
  // weight to an edge that may have siblings with non-zero weights. This can
  // be handled various ways, but it's probably fine for an edge with unknown
  // weight to just "inherit" the non-zero weight of an adjacent successor.
  static const uint32_t DEFAULT_WEIGHT = 16;

  DenseMap<Edge, uint32_t> Weights;

  /// \brief Handle to the LoopInfo analysis.
  LoopInfo *LI;

  /// \brief Track the last function we run over for printing.
  Function *LastF;

  /// \brief Track the set of blocks directly succeeded by a returning block.
  SmallPtrSet<BasicBlock *, 16> PostDominatedByUnreachable;

  /// \brief Track the set of blocks that always lead to a cold call.
  SmallPtrSet<BasicBlock *, 16> PostDominatedByColdCall;

  /// \brief Get sum of the block successors' weights.
  uint32_t getSumForBlock(const BasicBlock *BB) const;

  bool calcUnreachableHeuristics(BasicBlock *BB);
  bool calcMetadataWeights(BasicBlock *BB);
  bool calcColdCallHeuristics(BasicBlock *BB);
  bool calcPointerHeuristics(BasicBlock *BB);
  bool calcLoopBranchHeuristics(BasicBlock *BB);
  bool calcZeroHeuristics(BasicBlock *BB);
  bool calcFloatingPointHeuristics(BasicBlock *BB);
  bool calcInvokeHeuristics(BasicBlock *BB);
};

}

#endif

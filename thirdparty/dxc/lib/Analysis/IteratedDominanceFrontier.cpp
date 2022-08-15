//===- IteratedDominanceFrontier.cpp - Compute IDF ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \brief Compute iterated dominance frontiers using a linear time algorithm.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/IteratedDominanceFrontier.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Dominators.h"
#include <queue>

using namespace llvm;

void IDFCalculator::calculate(SmallVectorImpl<BasicBlock *> &PHIBlocks) {
  // If we haven't computed dominator tree levels, do so now.
  if (DomLevels.empty()) {
    for (auto DFI = df_begin(DT.getRootNode()), DFE = df_end(DT.getRootNode());
         DFI != DFE; ++DFI) {
      DomLevels[*DFI] = DFI.getPathLength() - 1;
    }
  }

  // Use a priority queue keyed on dominator tree level so that inserted nodes
  // are handled from the bottom of the dominator tree upwards.
  typedef std::pair<DomTreeNode *, unsigned> DomTreeNodePair;
  typedef std::priority_queue<DomTreeNodePair, SmallVector<DomTreeNodePair, 32>,
                              less_second> IDFPriorityQueue;
  IDFPriorityQueue PQ;

  for (BasicBlock *BB : *DefBlocks) {
    if (DomTreeNode *Node = DT.getNode(BB))
      PQ.push(std::make_pair(Node, DomLevels.lookup(Node)));
  }

  SmallVector<DomTreeNode *, 32> Worklist;
  SmallPtrSet<DomTreeNode *, 32> VisitedPQ;
  SmallPtrSet<DomTreeNode *, 32> VisitedWorklist;

  while (!PQ.empty()) {
    DomTreeNodePair RootPair = PQ.top();
    PQ.pop();
    DomTreeNode *Root = RootPair.first;
    unsigned RootLevel = RootPair.second;

    // Walk all dominator tree children of Root, inspecting their CFG edges with
    // targets elsewhere on the dominator tree. Only targets whose level is at
    // most Root's level are added to the iterated dominance frontier of the
    // definition set.

    Worklist.clear();
    Worklist.push_back(Root);
    VisitedWorklist.insert(Root);

    while (!Worklist.empty()) {
      DomTreeNode *Node = Worklist.pop_back_val();
      BasicBlock *BB = Node->getBlock();

      for (auto Succ : successors(BB)) {
        DomTreeNode *SuccNode = DT.getNode(Succ);

        // Quickly skip all CFG edges that are also dominator tree edges instead
        // of catching them below.
        if (SuccNode->getIDom() == Node)
          continue;

        unsigned SuccLevel = DomLevels.lookup(SuccNode);
        if (SuccLevel > RootLevel)
          continue;

        if (!VisitedPQ.insert(SuccNode).second)
          continue;

        BasicBlock *SuccBB = SuccNode->getBlock();
        if (useLiveIn && !LiveInBlocks->count(SuccBB))
          continue;

        PHIBlocks.emplace_back(SuccBB);
        if (!DefBlocks->count(SuccBB))
          PQ.push(std::make_pair(SuccNode, SuccLevel));
      }

      for (auto DomChild : *Node) {
        if (VisitedWorklist.insert(DomChild).second)
          Worklist.push_back(DomChild);
      }
    }
  }
}

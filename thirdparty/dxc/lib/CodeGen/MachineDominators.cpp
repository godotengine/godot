//===- MachineDominators.cpp - Machine Dominator Calculation --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements simple dominator construction algorithms for finding
// forward dominators on machine functions.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/ADT/SmallBitVector.h"

using namespace llvm;

namespace llvm {
template class DomTreeNodeBase<MachineBasicBlock>;
template class DominatorTreeBase<MachineBasicBlock>;
}

char MachineDominatorTree::ID = 0;

INITIALIZE_PASS(MachineDominatorTree, "machinedomtree",
                "MachineDominator Tree Construction", true, true)

char &llvm::MachineDominatorsID = MachineDominatorTree::ID;

void MachineDominatorTree::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  MachineFunctionPass::getAnalysisUsage(AU);
}

bool MachineDominatorTree::runOnMachineFunction(MachineFunction &F) {
  CriticalEdgesToSplit.clear();
  NewBBs.clear();
  DT->recalculate(F);

  return false;
}

MachineDominatorTree::MachineDominatorTree()
    : MachineFunctionPass(ID) {
  initializeMachineDominatorTreePass(*PassRegistry::getPassRegistry());
  DT = new DominatorTreeBase<MachineBasicBlock>(false);
}

MachineDominatorTree::~MachineDominatorTree() {
  delete DT;
}

void MachineDominatorTree::releaseMemory() {
  DT->releaseMemory();
}

void MachineDominatorTree::print(raw_ostream &OS, const Module*) const {
  DT->print(OS);
}

void MachineDominatorTree::applySplitCriticalEdges() const {
  // Bail out early if there is nothing to do.
  if (CriticalEdgesToSplit.empty())
    return;

  // For each element in CriticalEdgesToSplit, remember whether or not element
  // is the new immediate domminator of its successor. The mapping is done by
  // index, i.e., the information for the ith element of CriticalEdgesToSplit is
  // the ith element of IsNewIDom.
  SmallBitVector IsNewIDom(CriticalEdgesToSplit.size(), true);
  size_t Idx = 0;

  // Collect all the dominance properties info, before invalidating
  // the underlying DT.
  for (CriticalEdge &Edge : CriticalEdgesToSplit) {
    // Update dominator information.
    MachineBasicBlock *Succ = Edge.ToBB;
    MachineDomTreeNode *SuccDTNode = DT->getNode(Succ);

    for (MachineBasicBlock *PredBB : Succ->predecessors()) {
      if (PredBB == Edge.NewBB)
        continue;
      // If we are in this situation:
      // FromBB1        FromBB2
      //    +              +
      //   + +            + +
      //  +   +          +   +
      // ...  Split1  Split2 ...
      //           +   +
      //            + +
      //             +
      //            Succ
      // Instead of checking the domiance property with Split2, we check it with
      // FromBB2 since Split2 is still unknown of the underlying DT structure.
      if (NewBBs.count(PredBB)) {
        assert(PredBB->pred_size() == 1 && "A basic block resulting from a "
                                           "critical edge split has more "
                                           "than one predecessor!");
        PredBB = *PredBB->pred_begin();
      }
      if (!DT->dominates(SuccDTNode, DT->getNode(PredBB))) {
        IsNewIDom[Idx] = false;
        break;
      }
    }
    ++Idx;
  }

  // Now, update DT with the collected dominance properties info.
  Idx = 0;
  for (CriticalEdge &Edge : CriticalEdgesToSplit) {
    // We know FromBB dominates NewBB.
    MachineDomTreeNode *NewDTNode = DT->addNewBlock(Edge.NewBB, Edge.FromBB);

    // If all the other predecessors of "Succ" are dominated by "Succ" itself
    // then the new block is the new immediate dominator of "Succ". Otherwise,
    // the new block doesn't dominate anything.
    if (IsNewIDom[Idx])
      DT->changeImmediateDominator(DT->getNode(Edge.ToBB), NewDTNode);
    ++Idx;
  }
  NewBBs.clear();
  CriticalEdgesToSplit.clear();
}

//===---- ScheduleDAG.cpp - Implement the ScheduleDAG class ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements the ScheduleDAG class, which is a base class used by
// scheduling implementation classes.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/CodeGen/ScheduleHazardRecognizer.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include <climits>
using namespace llvm;

#define DEBUG_TYPE "pre-RA-sched"

#ifndef NDEBUG
static cl::opt<bool> StressSchedOpt(
  "stress-sched", cl::Hidden, cl::init(false),
  cl::desc("Stress test instruction scheduling"));
#endif

void SchedulingPriorityQueue::anchor() { }

ScheduleDAG::ScheduleDAG(MachineFunction &mf)
    : TM(mf.getTarget()), TII(mf.getSubtarget().getInstrInfo()),
      TRI(mf.getSubtarget().getRegisterInfo()), MF(mf),
      MRI(mf.getRegInfo()), EntrySU(), ExitSU() {
#ifndef NDEBUG
  StressSched = StressSchedOpt;
#endif
}

ScheduleDAG::~ScheduleDAG() {}

/// Clear the DAG state (e.g. between scheduling regions).
void ScheduleDAG::clearDAG() {
  SUnits.clear();
  EntrySU = SUnit();
  ExitSU = SUnit();
}

/// getInstrDesc helper to handle SDNodes.
const MCInstrDesc *ScheduleDAG::getNodeDesc(const SDNode *Node) const {
  if (!Node || !Node->isMachineOpcode()) return nullptr;
  return &TII->get(Node->getMachineOpcode());
}

/// addPred - This adds the specified edge as a pred of the current node if
/// not already.  It also adds the current node as a successor of the
/// specified node.
bool SUnit::addPred(const SDep &D, bool Required) {
  // If this node already has this dependence, don't add a redundant one.
  for (SmallVectorImpl<SDep>::iterator I = Preds.begin(), E = Preds.end();
         I != E; ++I) {
    // Zero-latency weak edges may be added purely for heuristic ordering. Don't
    // add them if another kind of edge already exists.
    if (!Required && I->getSUnit() == D.getSUnit())
      return false;
    if (I->overlaps(D)) {
      // Extend the latency if needed. Equivalent to removePred(I) + addPred(D).
      if (I->getLatency() < D.getLatency()) {
        SUnit *PredSU = I->getSUnit();
        // Find the corresponding successor in N.
        SDep ForwardD = *I;
        ForwardD.setSUnit(this);
        for (SmallVectorImpl<SDep>::iterator II = PredSU->Succs.begin(),
               EE = PredSU->Succs.end(); II != EE; ++II) {
          if (*II == ForwardD) {
            II->setLatency(D.getLatency());
            break;
          }
        }
        I->setLatency(D.getLatency());
      }
      return false;
    }
  }
  // Now add a corresponding succ to N.
  SDep P = D;
  P.setSUnit(this);
  SUnit *N = D.getSUnit();
  // Update the bookkeeping.
  if (D.getKind() == SDep::Data) {
    assert(NumPreds < UINT_MAX && "NumPreds will overflow!");
    assert(N->NumSuccs < UINT_MAX && "NumSuccs will overflow!");
    ++NumPreds;
    ++N->NumSuccs;
  }
  if (!N->isScheduled) {
    if (D.isWeak()) {
      ++WeakPredsLeft;
    }
    else {
      assert(NumPredsLeft < UINT_MAX && "NumPredsLeft will overflow!");
      ++NumPredsLeft;
    }
  }
  if (!isScheduled) {
    if (D.isWeak()) {
      ++N->WeakSuccsLeft;
    }
    else {
      assert(N->NumSuccsLeft < UINT_MAX && "NumSuccsLeft will overflow!");
      ++N->NumSuccsLeft;
    }
  }
  Preds.push_back(D);
  N->Succs.push_back(P);
  if (P.getLatency() != 0) {
    this->setDepthDirty();
    N->setHeightDirty();
  }
  return true;
}

/// removePred - This removes the specified edge as a pred of the current
/// node if it exists.  It also removes the current node as a successor of
/// the specified node.
void SUnit::removePred(const SDep &D) {
  // Find the matching predecessor.
  for (SmallVectorImpl<SDep>::iterator I = Preds.begin(), E = Preds.end();
         I != E; ++I)
    if (*I == D) {
      // Find the corresponding successor in N.
      SDep P = D;
      P.setSUnit(this);
      SUnit *N = D.getSUnit();
      SmallVectorImpl<SDep>::iterator Succ = std::find(N->Succs.begin(),
                                                       N->Succs.end(), P);
      assert(Succ != N->Succs.end() && "Mismatching preds / succs lists!");
      N->Succs.erase(Succ);
      Preds.erase(I);
      // Update the bookkeeping.
      if (P.getKind() == SDep::Data) {
        assert(NumPreds > 0 && "NumPreds will underflow!");
        assert(N->NumSuccs > 0 && "NumSuccs will underflow!");
        --NumPreds;
        --N->NumSuccs;
      }
      if (!N->isScheduled) {
        if (D.isWeak())
          --WeakPredsLeft;
        else {
          assert(NumPredsLeft > 0 && "NumPredsLeft will underflow!");
          --NumPredsLeft;
        }
      }
      if (!isScheduled) {
        if (D.isWeak())
          --N->WeakSuccsLeft;
        else {
          assert(N->NumSuccsLeft > 0 && "NumSuccsLeft will underflow!");
          --N->NumSuccsLeft;
        }
      }
      if (P.getLatency() != 0) {
        this->setDepthDirty();
        N->setHeightDirty();
      }
      return;
    }
}

void SUnit::setDepthDirty() {
  if (!isDepthCurrent) return;
  SmallVector<SUnit*, 8> WorkList;
  WorkList.push_back(this);
  do {
    SUnit *SU = WorkList.pop_back_val();
    SU->isDepthCurrent = false;
    for (SUnit::const_succ_iterator I = SU->Succs.begin(),
         E = SU->Succs.end(); I != E; ++I) {
      SUnit *SuccSU = I->getSUnit();
      if (SuccSU->isDepthCurrent)
        WorkList.push_back(SuccSU);
    }
  } while (!WorkList.empty());
}

void SUnit::setHeightDirty() {
  if (!isHeightCurrent) return;
  SmallVector<SUnit*, 8> WorkList;
  WorkList.push_back(this);
  do {
    SUnit *SU = WorkList.pop_back_val();
    SU->isHeightCurrent = false;
    for (SUnit::const_pred_iterator I = SU->Preds.begin(),
         E = SU->Preds.end(); I != E; ++I) {
      SUnit *PredSU = I->getSUnit();
      if (PredSU->isHeightCurrent)
        WorkList.push_back(PredSU);
    }
  } while (!WorkList.empty());
}

/// setDepthToAtLeast - Update this node's successors to reflect the
/// fact that this node's depth just increased.
///
void SUnit::setDepthToAtLeast(unsigned NewDepth) {
  if (NewDepth <= getDepth())
    return;
  setDepthDirty();
  Depth = NewDepth;
  isDepthCurrent = true;
}

/// setHeightToAtLeast - Update this node's predecessors to reflect the
/// fact that this node's height just increased.
///
void SUnit::setHeightToAtLeast(unsigned NewHeight) {
  if (NewHeight <= getHeight())
    return;
  setHeightDirty();
  Height = NewHeight;
  isHeightCurrent = true;
}

/// ComputeDepth - Calculate the maximal path from the node to the exit.
///
void SUnit::ComputeDepth() {
  SmallVector<SUnit*, 8> WorkList;
  WorkList.push_back(this);
  do {
    SUnit *Cur = WorkList.back();

    bool Done = true;
    unsigned MaxPredDepth = 0;
    for (SUnit::const_pred_iterator I = Cur->Preds.begin(),
         E = Cur->Preds.end(); I != E; ++I) {
      SUnit *PredSU = I->getSUnit();
      if (PredSU->isDepthCurrent)
        MaxPredDepth = std::max(MaxPredDepth,
                                PredSU->Depth + I->getLatency());
      else {
        Done = false;
        WorkList.push_back(PredSU);
      }
    }

    if (Done) {
      WorkList.pop_back();
      if (MaxPredDepth != Cur->Depth) {
        Cur->setDepthDirty();
        Cur->Depth = MaxPredDepth;
      }
      Cur->isDepthCurrent = true;
    }
  } while (!WorkList.empty());
}

/// ComputeHeight - Calculate the maximal path from the node to the entry.
///
void SUnit::ComputeHeight() {
  SmallVector<SUnit*, 8> WorkList;
  WorkList.push_back(this);
  do {
    SUnit *Cur = WorkList.back();

    bool Done = true;
    unsigned MaxSuccHeight = 0;
    for (SUnit::const_succ_iterator I = Cur->Succs.begin(),
         E = Cur->Succs.end(); I != E; ++I) {
      SUnit *SuccSU = I->getSUnit();
      if (SuccSU->isHeightCurrent)
        MaxSuccHeight = std::max(MaxSuccHeight,
                                 SuccSU->Height + I->getLatency());
      else {
        Done = false;
        WorkList.push_back(SuccSU);
      }
    }

    if (Done) {
      WorkList.pop_back();
      if (MaxSuccHeight != Cur->Height) {
        Cur->setHeightDirty();
        Cur->Height = MaxSuccHeight;
      }
      Cur->isHeightCurrent = true;
    }
  } while (!WorkList.empty());
}

void SUnit::biasCriticalPath() {
  if (NumPreds < 2)
    return;

  SUnit::pred_iterator BestI = Preds.begin();
  unsigned MaxDepth = BestI->getSUnit()->getDepth();
  for (SUnit::pred_iterator I = std::next(BestI), E = Preds.end(); I != E;
       ++I) {
    if (I->getKind() == SDep::Data && I->getSUnit()->getDepth() > MaxDepth)
      BestI = I;
  }
  if (BestI != Preds.begin())
    std::swap(*Preds.begin(), *BestI);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
/// SUnit - Scheduling unit. It's an wrapper around either a single SDNode or
/// a group of nodes flagged together.
void SUnit::dump(const ScheduleDAG *G) const {
  dbgs() << "SU(" << NodeNum << "): ";
  G->dumpNode(this);
}

void SUnit::dumpAll(const ScheduleDAG *G) const {
  dump(G);

  dbgs() << "  # preds left       : " << NumPredsLeft << "\n";
  dbgs() << "  # succs left       : " << NumSuccsLeft << "\n";
  if (WeakPredsLeft)
    dbgs() << "  # weak preds left  : " << WeakPredsLeft << "\n";
  if (WeakSuccsLeft)
    dbgs() << "  # weak succs left  : " << WeakSuccsLeft << "\n";
  dbgs() << "  # rdefs left       : " << NumRegDefsLeft << "\n";
  dbgs() << "  Latency            : " << Latency << "\n";
  dbgs() << "  Depth              : " << getDepth() << "\n";
  dbgs() << "  Height             : " << getHeight() << "\n";

  if (Preds.size() != 0) {
    dbgs() << "  Predecessors:\n";
    for (SUnit::const_succ_iterator I = Preds.begin(), E = Preds.end();
         I != E; ++I) {
      dbgs() << "   ";
      switch (I->getKind()) {
      case SDep::Data:        dbgs() << "val "; break;
      case SDep::Anti:        dbgs() << "anti"; break;
      case SDep::Output:      dbgs() << "out "; break;
      case SDep::Order:       dbgs() << "ch  "; break;
      }
      dbgs() << "SU(" << I->getSUnit()->NodeNum << ")";
      if (I->isArtificial())
        dbgs() << " *";
      dbgs() << ": Latency=" << I->getLatency();
      if (I->isAssignedRegDep())
        dbgs() << " Reg=" << PrintReg(I->getReg(), G->TRI);
      dbgs() << "\n";
    }
  }
  if (Succs.size() != 0) {
    dbgs() << "  Successors:\n";
    for (SUnit::const_succ_iterator I = Succs.begin(), E = Succs.end();
         I != E; ++I) {
      dbgs() << "   ";
      switch (I->getKind()) {
      case SDep::Data:        dbgs() << "val "; break;
      case SDep::Anti:        dbgs() << "anti"; break;
      case SDep::Output:      dbgs() << "out "; break;
      case SDep::Order:       dbgs() << "ch  "; break;
      }
      dbgs() << "SU(" << I->getSUnit()->NodeNum << ")";
      if (I->isArtificial())
        dbgs() << " *";
      dbgs() << ": Latency=" << I->getLatency();
      if (I->isAssignedRegDep())
        dbgs() << " Reg=" << PrintReg(I->getReg(), G->TRI);
      dbgs() << "\n";
    }
  }
  dbgs() << "\n";
}
#endif

#ifndef NDEBUG
/// VerifyScheduledDAG - Verify that all SUnits were scheduled and that
/// their state is consistent. Return the number of scheduled nodes.
///
unsigned ScheduleDAG::VerifyScheduledDAG(bool isBottomUp) {
  bool AnyNotSched = false;
  unsigned DeadNodes = 0;
  for (unsigned i = 0, e = SUnits.size(); i != e; ++i) {
    if (!SUnits[i].isScheduled) {
      if (SUnits[i].NumPreds == 0 && SUnits[i].NumSuccs == 0) {
        ++DeadNodes;
        continue;
      }
      if (!AnyNotSched)
        dbgs() << "*** Scheduling failed! ***\n";
      SUnits[i].dump(this);
      dbgs() << "has not been scheduled!\n";
      AnyNotSched = true;
    }
    if (SUnits[i].isScheduled &&
        (isBottomUp ? SUnits[i].getHeight() : SUnits[i].getDepth()) >
          unsigned(INT_MAX)) {
      if (!AnyNotSched)
        dbgs() << "*** Scheduling failed! ***\n";
      SUnits[i].dump(this);
      dbgs() << "has an unexpected "
           << (isBottomUp ? "Height" : "Depth") << " value!\n";
      AnyNotSched = true;
    }
    if (isBottomUp) {
      if (SUnits[i].NumSuccsLeft != 0) {
        if (!AnyNotSched)
          dbgs() << "*** Scheduling failed! ***\n";
        SUnits[i].dump(this);
        dbgs() << "has successors left!\n";
        AnyNotSched = true;
      }
    } else {
      if (SUnits[i].NumPredsLeft != 0) {
        if (!AnyNotSched)
          dbgs() << "*** Scheduling failed! ***\n";
        SUnits[i].dump(this);
        dbgs() << "has predecessors left!\n";
        AnyNotSched = true;
      }
    }
  }
  assert(!AnyNotSched);
  return SUnits.size() - DeadNodes;
}
#endif

/// InitDAGTopologicalSorting - create the initial topological
/// ordering from the DAG to be scheduled.
///
/// The idea of the algorithm is taken from
/// "Online algorithms for managing the topological order of
/// a directed acyclic graph" by David J. Pearce and Paul H.J. Kelly
/// This is the MNR algorithm, which was first introduced by
/// A. Marchetti-Spaccamela, U. Nanni and H. Rohnert in
/// "Maintaining a topological order under edge insertions".
///
/// Short description of the algorithm:
///
/// Topological ordering, ord, of a DAG maps each node to a topological
/// index so that for all edges X->Y it is the case that ord(X) < ord(Y).
///
/// This means that if there is a path from the node X to the node Z,
/// then ord(X) < ord(Z).
///
/// This property can be used to check for reachability of nodes:
/// if Z is reachable from X, then an insertion of the edge Z->X would
/// create a cycle.
///
/// The algorithm first computes a topological ordering for the DAG by
/// initializing the Index2Node and Node2Index arrays and then tries to keep
/// the ordering up-to-date after edge insertions by reordering the DAG.
///
/// On insertion of the edge X->Y, the algorithm first marks by calling DFS
/// the nodes reachable from Y, and then shifts them using Shift to lie
/// immediately after X in Index2Node.
void ScheduleDAGTopologicalSort::InitDAGTopologicalSorting() {
  unsigned DAGSize = SUnits.size();
  std::vector<SUnit*> WorkList;
  WorkList.reserve(DAGSize);

  Index2Node.resize(DAGSize);
  Node2Index.resize(DAGSize);

  // Initialize the data structures.
  if (ExitSU)
    WorkList.push_back(ExitSU);
  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SUnit *SU = &SUnits[i];
    int NodeNum = SU->NodeNum;
    unsigned Degree = SU->Succs.size();
    // Temporarily use the Node2Index array as scratch space for degree counts.
    Node2Index[NodeNum] = Degree;

    // Is it a node without dependencies?
    if (Degree == 0) {
      assert(SU->Succs.empty() && "SUnit should have no successors");
      // Collect leaf nodes.
      WorkList.push_back(SU);
    }
  }

  int Id = DAGSize;
  while (!WorkList.empty()) {
    SUnit *SU = WorkList.back();
    WorkList.pop_back();
    if (SU->NodeNum < DAGSize)
      Allocate(SU->NodeNum, --Id);
    for (SUnit::const_pred_iterator I = SU->Preds.begin(), E = SU->Preds.end();
         I != E; ++I) {
      SUnit *SU = I->getSUnit();
      if (SU->NodeNum < DAGSize && !--Node2Index[SU->NodeNum])
        // If all dependencies of the node are processed already,
        // then the node can be computed now.
        WorkList.push_back(SU);
    }
  }

  Visited.resize(DAGSize);

#ifndef NDEBUG
  // Check correctness of the ordering
  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SUnit *SU = &SUnits[i];
    for (SUnit::const_pred_iterator I = SU->Preds.begin(), E = SU->Preds.end();
         I != E; ++I) {
      assert(Node2Index[SU->NodeNum] > Node2Index[I->getSUnit()->NodeNum] &&
      "Wrong topological sorting");
    }
  }
#endif
}

/// AddPred - Updates the topological ordering to accommodate an edge
/// to be added from SUnit X to SUnit Y.
void ScheduleDAGTopologicalSort::AddPred(SUnit *Y, SUnit *X) {
  int UpperBound, LowerBound;
  LowerBound = Node2Index[Y->NodeNum];
  UpperBound = Node2Index[X->NodeNum];
  bool HasLoop = false;
  // Is Ord(X) < Ord(Y) ?
  if (LowerBound < UpperBound) {
    // Update the topological order.
    Visited.reset();
    DFS(Y, UpperBound, HasLoop);
    assert(!HasLoop && "Inserted edge creates a loop!");
    // Recompute topological indexes.
    Shift(Visited, LowerBound, UpperBound);
  }
}

/// RemovePred - Updates the topological ordering to accommodate an
/// an edge to be removed from the specified node N from the predecessors
/// of the current node M.
void ScheduleDAGTopologicalSort::RemovePred(SUnit *M, SUnit *N) {
  // InitDAGTopologicalSorting();
}

/// DFS - Make a DFS traversal to mark all nodes reachable from SU and mark
/// all nodes affected by the edge insertion. These nodes will later get new
/// topological indexes by means of the Shift method.
void ScheduleDAGTopologicalSort::DFS(const SUnit *SU, int UpperBound,
                                     bool &HasLoop) {
  std::vector<const SUnit*> WorkList;
  WorkList.reserve(SUnits.size());

  WorkList.push_back(SU);
  do {
    SU = WorkList.back();
    WorkList.pop_back();
    Visited.set(SU->NodeNum);
    for (int I = SU->Succs.size()-1; I >= 0; --I) {
      unsigned s = SU->Succs[I].getSUnit()->NodeNum;
      // Edges to non-SUnits are allowed but ignored (e.g. ExitSU).
      if (s >= Node2Index.size())
        continue;
      if (Node2Index[s] == UpperBound) {
        HasLoop = true;
        return;
      }
      // Visit successors if not already and in affected region.
      if (!Visited.test(s) && Node2Index[s] < UpperBound) {
        WorkList.push_back(SU->Succs[I].getSUnit());
      }
    }
  } while (!WorkList.empty());
}

/// Shift - Renumber the nodes so that the topological ordering is
/// preserved.
void ScheduleDAGTopologicalSort::Shift(BitVector& Visited, int LowerBound,
                                       int UpperBound) {
  std::vector<int> L;
  int shift = 0;
  int i;

  for (i = LowerBound; i <= UpperBound; ++i) {
    // w is node at topological index i.
    int w = Index2Node[i];
    if (Visited.test(w)) {
      // Unmark.
      Visited.reset(w);
      L.push_back(w);
      shift = shift + 1;
    } else {
      Allocate(w, i - shift);
    }
  }

  for (unsigned j = 0; j < L.size(); ++j) {
    Allocate(L[j], i - shift);
    i = i + 1;
  }
}


/// WillCreateCycle - Returns true if adding an edge to TargetSU from SU will
/// create a cycle. If so, it is not safe to call AddPred(TargetSU, SU).
bool ScheduleDAGTopologicalSort::WillCreateCycle(SUnit *TargetSU, SUnit *SU) {
  // Is SU reachable from TargetSU via successor edges?
  if (IsReachable(SU, TargetSU))
    return true;
  for (SUnit::pred_iterator
         I = TargetSU->Preds.begin(), E = TargetSU->Preds.end(); I != E; ++I)
    if (I->isAssignedRegDep() &&
        IsReachable(SU, I->getSUnit()))
      return true;
  return false;
}

/// IsReachable - Checks if SU is reachable from TargetSU.
bool ScheduleDAGTopologicalSort::IsReachable(const SUnit *SU,
                                             const SUnit *TargetSU) {
  // If insertion of the edge SU->TargetSU would create a cycle
  // then there is a path from TargetSU to SU.
  int UpperBound, LowerBound;
  LowerBound = Node2Index[TargetSU->NodeNum];
  UpperBound = Node2Index[SU->NodeNum];
  bool HasLoop = false;
  // Is Ord(TargetSU) < Ord(SU) ?
  if (LowerBound < UpperBound) {
    Visited.reset();
    // There may be a path from TargetSU to SU. Check for it.
    DFS(TargetSU, UpperBound, HasLoop);
  }
  return HasLoop;
}

/// Allocate - assign the topological index to the node n.
void ScheduleDAGTopologicalSort::Allocate(int n, int index) {
  Node2Index[n] = index;
  Index2Node[index] = n;
}

ScheduleDAGTopologicalSort::
ScheduleDAGTopologicalSort(std::vector<SUnit> &sunits, SUnit *exitsu)
  : SUnits(sunits), ExitSU(exitsu) {}

ScheduleHazardRecognizer::~ScheduleHazardRecognizer() {}

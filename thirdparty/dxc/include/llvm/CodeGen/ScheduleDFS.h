//===- ScheduleDAGILP.h - ILP metric for ScheduleDAGInstrs ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Definition of an ILP metric for machine level instruction scheduling.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_SCHEDULEDFS_H
#define LLVM_CODEGEN_SCHEDULEDFS_H

#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/Support/DataTypes.h"
#include <vector>

namespace llvm {

class raw_ostream;
class IntEqClasses;
class ScheduleDAGInstrs;
class SUnit;

/// \brief Represent the ILP of the subDAG rooted at a DAG node.
///
/// ILPValues summarize the DAG subtree rooted at each node. ILPValues are
/// valid for all nodes regardless of their subtree membership.
///
/// When computed using bottom-up DFS, this metric assumes that the DAG is a
/// forest of trees with roots at the bottom of the schedule branching upward.
struct ILPValue {
  unsigned InstrCount;
  /// Length may either correspond to depth or height, depending on direction,
  /// and cycles or nodes depending on context.
  unsigned Length;

  ILPValue(unsigned count, unsigned length):
    InstrCount(count), Length(length) {}

  // Order by the ILP metric's value.
  bool operator<(ILPValue RHS) const {
    return (uint64_t)InstrCount * RHS.Length
      < (uint64_t)Length * RHS.InstrCount;
  }
  bool operator>(ILPValue RHS) const {
    return RHS < *this;
  }
  bool operator<=(ILPValue RHS) const {
    return (uint64_t)InstrCount * RHS.Length
      <= (uint64_t)Length * RHS.InstrCount;
  }
  bool operator>=(ILPValue RHS) const {
    return RHS <= *this;
  }

  void print(raw_ostream &OS) const;

  void dump() const;
};

/// \brief Compute the values of each DAG node for various metrics during DFS.
class SchedDFSResult {
  friend class SchedDFSImpl;

  static const unsigned InvalidSubtreeID = ~0u;

  /// \brief Per-SUnit data computed during DFS for various metrics.
  ///
  /// A node's SubtreeID is set to itself when it is visited to indicate that it
  /// is the root of a subtree. Later it is set to its parent to indicate an
  /// interior node. Finally, it is set to a representative subtree ID during
  /// finalization.
  struct NodeData {
    unsigned InstrCount;
    unsigned SubtreeID;

    NodeData(): InstrCount(0), SubtreeID(InvalidSubtreeID) {}
  };

  /// \brief Per-Subtree data computed during DFS.
  struct TreeData {
    unsigned ParentTreeID;
    unsigned SubInstrCount;

    TreeData(): ParentTreeID(InvalidSubtreeID), SubInstrCount(0) {}
  };

  /// \brief Record a connection between subtrees and the connection level.
  struct Connection {
    unsigned TreeID;
    unsigned Level;

    Connection(unsigned tree, unsigned level): TreeID(tree), Level(level) {}
  };

  bool IsBottomUp;
  unsigned SubtreeLimit;
  /// DFS results for each SUnit in this DAG.
  std::vector<NodeData> DFSNodeData;

  // Store per-tree data indexed on tree ID,
  SmallVector<TreeData, 16> DFSTreeData;

  // For each subtree discovered during DFS, record its connections to other
  // subtrees.
  std::vector<SmallVector<Connection, 4> > SubtreeConnections;

  /// Cache the current connection level of each subtree.
  /// This mutable array is updated during scheduling.
  std::vector<unsigned> SubtreeConnectLevels;

public:
  SchedDFSResult(bool IsBU, unsigned lim)
    : IsBottomUp(IsBU), SubtreeLimit(lim) {}

  /// \brief Get the node cutoff before subtrees are considered significant.
  unsigned getSubtreeLimit() const { return SubtreeLimit; }

  /// \brief Return true if this DFSResult is uninitialized.
  ///
  /// resize() initializes DFSResult, while compute() populates it.
  bool empty() const { return DFSNodeData.empty(); }

  /// \brief Clear the results.
  void clear() {
    DFSNodeData.clear();
    DFSTreeData.clear();
    SubtreeConnections.clear();
    SubtreeConnectLevels.clear();
  }

  /// \brief Initialize the result data with the size of the DAG.
  void resize(unsigned NumSUnits) {
    DFSNodeData.resize(NumSUnits);
  }

  /// \brief Compute various metrics for the DAG with given roots.
  void compute(ArrayRef<SUnit> SUnits);

  /// \brief Get the number of instructions in the given subtree and its
  /// children.
  unsigned getNumInstrs(const SUnit *SU) const {
    return DFSNodeData[SU->NodeNum].InstrCount;
  }

  /// \brief Get the number of instructions in the given subtree not including
  /// children.
  unsigned getNumSubInstrs(unsigned SubtreeID) const {
    return DFSTreeData[SubtreeID].SubInstrCount;
  }

  /// \brief Get the ILP value for a DAG node.
  ///
  /// A leaf node has an ILP of 1/1.
  ILPValue getILP(const SUnit *SU) const {
    return ILPValue(DFSNodeData[SU->NodeNum].InstrCount, 1 + SU->getDepth());
  }

  /// \brief The number of subtrees detected in this DAG.
  unsigned getNumSubtrees() const { return SubtreeConnectLevels.size(); }

  /// \brief Get the ID of the subtree the given DAG node belongs to.
  ///
  /// For convenience, if DFSResults have not been computed yet, give everything
  /// tree ID 0.
  unsigned getSubtreeID(const SUnit *SU) const {
    if (empty())
      return 0;
    assert(SU->NodeNum < DFSNodeData.size() &&  "New Node");
    return DFSNodeData[SU->NodeNum].SubtreeID;
  }

  /// \brief Get the connection level of a subtree.
  ///
  /// For bottom-up trees, the connection level is the latency depth (in cycles)
  /// of the deepest connection to another subtree.
  unsigned getSubtreeLevel(unsigned SubtreeID) const {
    return SubtreeConnectLevels[SubtreeID];
  }

  /// \brief Scheduler callback to update SubtreeConnectLevels when a tree is
  /// initially scheduled.
  void scheduleTree(unsigned SubtreeID);
};

raw_ostream &operator<<(raw_ostream &OS, const ILPValue &Val);

} // namespace llvm

#endif

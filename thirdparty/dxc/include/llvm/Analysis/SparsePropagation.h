//===- SparsePropagation.h - Sparse Conditional Property Propagation ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements an abstract sparse conditional propagation algorithm,
// modeled after SCCP, but with a customizable lattice function.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_SPARSEPROPAGATION_H
#define LLVM_ANALYSIS_SPARSEPROPAGATION_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <set>
#include <vector>

namespace llvm {
  class Value;
  class Constant;
  class Argument;
  class Instruction;
  class PHINode;
  class TerminatorInst;
  class BasicBlock;
  class Function;
  class SparseSolver;
  class raw_ostream;

  template<typename T> class SmallVectorImpl;
  
/// AbstractLatticeFunction - This class is implemented by the dataflow instance
/// to specify what the lattice values are and how they handle merges etc.
/// This gives the client the power to compute lattice values from instructions,
/// constants, etc.  The requirement is that lattice values must all fit into
/// a void*.  If a void* is not sufficient, the implementation should use this
/// pointer to be a pointer into a uniquing set or something.
///
class AbstractLatticeFunction {
public:
  typedef void *LatticeVal;
private:
  LatticeVal UndefVal, OverdefinedVal, UntrackedVal;
public:
  AbstractLatticeFunction(LatticeVal undefVal, LatticeVal overdefinedVal,
                          LatticeVal untrackedVal) {
    UndefVal = undefVal;
    OverdefinedVal = overdefinedVal;
    UntrackedVal = untrackedVal;
  }
  virtual ~AbstractLatticeFunction();
  
  LatticeVal getUndefVal()       const { return UndefVal; }
  LatticeVal getOverdefinedVal() const { return OverdefinedVal; }
  LatticeVal getUntrackedVal()   const { return UntrackedVal; }
  
  /// IsUntrackedValue - If the specified Value is something that is obviously
  /// uninteresting to the analysis (and would always return UntrackedVal),
  /// this function can return true to avoid pointless work.
  virtual bool IsUntrackedValue(Value *V) {
    return false;
  }
  
  /// ComputeConstant - Given a constant value, compute and return a lattice
  /// value corresponding to the specified constant.
  virtual LatticeVal ComputeConstant(Constant *C) {
    return getOverdefinedVal(); // always safe
  }

  /// IsSpecialCasedPHI - Given a PHI node, determine whether this PHI node is
  /// one that the we want to handle through ComputeInstructionState.
  virtual bool IsSpecialCasedPHI(PHINode *PN) {
    return false;
  }
  
  /// GetConstant - If the specified lattice value is representable as an LLVM
  /// constant value, return it.  Otherwise return null.  The returned value
  /// must be in the same LLVM type as Val.
  virtual Constant *GetConstant(LatticeVal LV, Value *Val, SparseSolver &SS) {
    return nullptr;
  }

  /// ComputeArgument - Given a formal argument value, compute and return a
  /// lattice value corresponding to the specified argument.
  virtual LatticeVal ComputeArgument(Argument *I) {
    return getOverdefinedVal(); // always safe
  }
  
  /// MergeValues - Compute and return the merge of the two specified lattice
  /// values.  Merging should only move one direction down the lattice to
  /// guarantee convergence (toward overdefined).
  virtual LatticeVal MergeValues(LatticeVal X, LatticeVal Y) {
    return getOverdefinedVal(); // always safe, never useful.
  }
  
  /// ComputeInstructionState - Given an instruction and a vector of its operand
  /// values, compute the result value of the instruction.
  virtual LatticeVal ComputeInstructionState(Instruction &I, SparseSolver &SS) {
    return getOverdefinedVal(); // always safe, never useful.
  }
  
  /// PrintValue - Render the specified lattice value to the specified stream.
  virtual void PrintValue(LatticeVal V, raw_ostream &OS);
};

  
/// SparseSolver - This class is a general purpose solver for Sparse Conditional
/// Propagation with a programmable lattice function.
///
class SparseSolver {
  typedef AbstractLatticeFunction::LatticeVal LatticeVal;
  
  /// LatticeFunc - This is the object that knows the lattice and how to do
  /// compute transfer functions.
  AbstractLatticeFunction *LatticeFunc;
  
  DenseMap<Value*, LatticeVal> ValueState;  // The state each value is in.
  SmallPtrSet<BasicBlock*, 16> BBExecutable;   // The bbs that are executable.
  
  std::vector<Instruction*> InstWorkList;   // Worklist of insts to process.
  
  std::vector<BasicBlock*> BBWorkList;  // The BasicBlock work list
  
  /// KnownFeasibleEdges - Entries in this set are edges which have already had
  /// PHI nodes retriggered.
  typedef std::pair<BasicBlock*,BasicBlock*> Edge;
  std::set<Edge> KnownFeasibleEdges;

  SparseSolver(const SparseSolver&) = delete;
  void operator=(const SparseSolver&) = delete;
public:
  explicit SparseSolver(AbstractLatticeFunction *Lattice)
    : LatticeFunc(Lattice) {}
  ~SparseSolver() {
    delete LatticeFunc;
  }
  
  /// Solve - Solve for constants and executable blocks.
  ///
  void Solve(Function &F);
  
  void Print(Function &F, raw_ostream &OS) const;

  /// getLatticeState - Return the LatticeVal object that corresponds to the
  /// value.  If an value is not in the map, it is returned as untracked,
  /// unlike the getOrInitValueState method.
  LatticeVal getLatticeState(Value *V) const {
    DenseMap<Value*, LatticeVal>::const_iterator I = ValueState.find(V);
    return I != ValueState.end() ? I->second : LatticeFunc->getUntrackedVal();
  }
  
  /// getOrInitValueState - Return the LatticeVal object that corresponds to the
  /// value, initializing the value's state if it hasn't been entered into the
  /// map yet.   This function is necessary because not all values should start
  /// out in the underdefined state... Arguments should be overdefined, and
  /// constants should be marked as constants.
  ///
  LatticeVal getOrInitValueState(Value *V);
  
  /// isEdgeFeasible - Return true if the control flow edge from the 'From'
  /// basic block to the 'To' basic block is currently feasible.  If
  /// AggressiveUndef is true, then this treats values with unknown lattice
  /// values as undefined.  This is generally only useful when solving the
  /// lattice, not when querying it.
  bool isEdgeFeasible(BasicBlock *From, BasicBlock *To,
                      bool AggressiveUndef = false);

  /// isBlockExecutable - Return true if there are any known feasible
  /// edges into the basic block.  This is generally only useful when
  /// querying the lattice.
  bool isBlockExecutable(BasicBlock *BB) const {
    return BBExecutable.count(BB);
  }
  
private:
  /// UpdateState - When the state for some instruction is potentially updated,
  /// this function notices and adds I to the worklist if needed.
  void UpdateState(Instruction &Inst, LatticeVal V);
  
  /// MarkBlockExecutable - This method can be used by clients to mark all of
  /// the blocks that are known to be intrinsically live in the processed unit.
  void MarkBlockExecutable(BasicBlock *BB);
  
  /// markEdgeExecutable - Mark a basic block as executable, adding it to the BB
  /// work list if it is not already executable.
  void markEdgeExecutable(BasicBlock *Source, BasicBlock *Dest);
  
  /// getFeasibleSuccessors - Return a vector of booleans to indicate which
  /// successors are reachable from a given terminator instruction.
  void getFeasibleSuccessors(TerminatorInst &TI, SmallVectorImpl<bool> &Succs,
                             bool AggressiveUndef);
  
  void visitInst(Instruction &I);
  void visitPHINode(PHINode &I);
  void visitTerminatorInst(TerminatorInst &TI);

};

} // end namespace llvm

#endif // LLVM_ANALYSIS_SPARSEPROPAGATION_H

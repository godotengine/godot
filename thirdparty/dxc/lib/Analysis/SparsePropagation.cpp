//===- SparsePropagation.cpp - Sparse Conditional Property Propagation ----===//
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

#include "llvm/Analysis/SparsePropagation.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

#define DEBUG_TYPE "sparseprop"

//===----------------------------------------------------------------------===//
//                  AbstractLatticeFunction Implementation
//===----------------------------------------------------------------------===//

AbstractLatticeFunction::~AbstractLatticeFunction() {}

/// PrintValue - Render the specified lattice value to the specified stream.
void AbstractLatticeFunction::PrintValue(LatticeVal V, raw_ostream &OS) {
  if (V == UndefVal)
    OS << "undefined";
  else if (V == OverdefinedVal)
    OS << "overdefined";
  else if (V == UntrackedVal)
    OS << "untracked";
  else
    OS << "unknown lattice value";
}

//===----------------------------------------------------------------------===//
//                          SparseSolver Implementation
//===----------------------------------------------------------------------===//

/// getOrInitValueState - Return the LatticeVal object that corresponds to the
/// value, initializing the value's state if it hasn't been entered into the
/// map yet.   This function is necessary because not all values should start
/// out in the underdefined state... Arguments should be overdefined, and
/// constants should be marked as constants.
///
SparseSolver::LatticeVal SparseSolver::getOrInitValueState(Value *V) {
  DenseMap<Value*, LatticeVal>::iterator I = ValueState.find(V);
  if (I != ValueState.end()) return I->second;  // Common case, in the map
  
  LatticeVal LV;
  if (LatticeFunc->IsUntrackedValue(V))
    return LatticeFunc->getUntrackedVal();
  else if (Constant *C = dyn_cast<Constant>(V))
    LV = LatticeFunc->ComputeConstant(C);
  else if (Argument *A = dyn_cast<Argument>(V))
    LV = LatticeFunc->ComputeArgument(A);
  else if (!isa<Instruction>(V))
    // All other non-instructions are overdefined.
    LV = LatticeFunc->getOverdefinedVal();
  else
    // All instructions are underdefined by default.
    LV = LatticeFunc->getUndefVal();
  
  // If this value is untracked, don't add it to the map.
  if (LV == LatticeFunc->getUntrackedVal())
    return LV;
  return ValueState[V] = LV;
}

/// UpdateState - When the state for some instruction is potentially updated,
/// this function notices and adds I to the worklist if needed.
void SparseSolver::UpdateState(Instruction &Inst, LatticeVal V) {
  DenseMap<Value*, LatticeVal>::iterator I = ValueState.find(&Inst);
  if (I != ValueState.end() && I->second == V)
    return;  // No change.
  
  // An update.  Visit uses of I.
  ValueState[&Inst] = V;
  InstWorkList.push_back(&Inst);
}

/// MarkBlockExecutable - This method can be used by clients to mark all of
/// the blocks that are known to be intrinsically live in the processed unit.
void SparseSolver::MarkBlockExecutable(BasicBlock *BB) {
  DEBUG(dbgs() << "Marking Block Executable: " << BB->getName() << "\n");
  BBExecutable.insert(BB);   // Basic block is executable!
  BBWorkList.push_back(BB);  // Add the block to the work list!
}

/// markEdgeExecutable - Mark a basic block as executable, adding it to the BB
/// work list if it is not already executable...
void SparseSolver::markEdgeExecutable(BasicBlock *Source, BasicBlock *Dest) {
  if (!KnownFeasibleEdges.insert(Edge(Source, Dest)).second)
    return;  // This edge is already known to be executable!
  
  DEBUG(dbgs() << "Marking Edge Executable: " << Source->getName()
        << " -> " << Dest->getName() << "\n");

  if (BBExecutable.count(Dest)) {
    // The destination is already executable, but we just made an edge
    // feasible that wasn't before.  Revisit the PHI nodes in the block
    // because they have potentially new operands.
    for (BasicBlock::iterator I = Dest->begin(); isa<PHINode>(I); ++I)
      visitPHINode(*cast<PHINode>(I));
    
  } else {
    MarkBlockExecutable(Dest);
  }
}


/// getFeasibleSuccessors - Return a vector of booleans to indicate which
/// successors are reachable from a given terminator instruction.
void SparseSolver::getFeasibleSuccessors(TerminatorInst &TI,
                                         SmallVectorImpl<bool> &Succs,
                                         bool AggressiveUndef) {
  Succs.resize(TI.getNumSuccessors());
  if (TI.getNumSuccessors() == 0) return;
  
  if (BranchInst *BI = dyn_cast<BranchInst>(&TI)) {
    if (BI->isUnconditional()) {
      Succs[0] = true;
      return;
    }
    
    LatticeVal BCValue;
    if (AggressiveUndef)
      BCValue = getOrInitValueState(BI->getCondition());
    else
      BCValue = getLatticeState(BI->getCondition());
    
    if (BCValue == LatticeFunc->getOverdefinedVal() ||
        BCValue == LatticeFunc->getUntrackedVal()) {
      // Overdefined condition variables can branch either way.
      Succs[0] = Succs[1] = true;
      return;
    }

    // If undefined, neither is feasible yet.
    if (BCValue == LatticeFunc->getUndefVal())
      return;

    Constant *C = LatticeFunc->GetConstant(BCValue, BI->getCondition(), *this);
    if (!C || !isa<ConstantInt>(C)) {
      // Non-constant values can go either way.
      Succs[0] = Succs[1] = true;
      return;
    }

    // Constant condition variables mean the branch can only go a single way
    Succs[C->isNullValue()] = true;
    return;
  }
  
  if (isa<InvokeInst>(TI)) {
    // Invoke instructions successors are always executable.
    // TODO: Could ask the lattice function if the value can throw.
    Succs[0] = Succs[1] = true;
    return;
  }
  
  if (isa<IndirectBrInst>(TI)) {
    Succs.assign(Succs.size(), true);
    return;
  }
  
  SwitchInst &SI = cast<SwitchInst>(TI);
  LatticeVal SCValue;
  if (AggressiveUndef)
    SCValue = getOrInitValueState(SI.getCondition());
  else
    SCValue = getLatticeState(SI.getCondition());
  
  if (SCValue == LatticeFunc->getOverdefinedVal() ||
      SCValue == LatticeFunc->getUntrackedVal()) {
    // All destinations are executable!
    Succs.assign(TI.getNumSuccessors(), true);
    return;
  }
  
  // If undefined, neither is feasible yet.
  if (SCValue == LatticeFunc->getUndefVal())
    return;
  
  Constant *C = LatticeFunc->GetConstant(SCValue, SI.getCondition(), *this);
  if (!C || !isa<ConstantInt>(C)) {
    // All destinations are executable!
    Succs.assign(TI.getNumSuccessors(), true);
    return;
  }
  SwitchInst::CaseIt Case = SI.findCaseValue(cast<ConstantInt>(C));
  Succs[Case.getSuccessorIndex()] = true;
}


/// isEdgeFeasible - Return true if the control flow edge from the 'From'
/// basic block to the 'To' basic block is currently feasible...
bool SparseSolver::isEdgeFeasible(BasicBlock *From, BasicBlock *To,
                                  bool AggressiveUndef) {
  SmallVector<bool, 16> SuccFeasible;
  TerminatorInst *TI = From->getTerminator();
  getFeasibleSuccessors(*TI, SuccFeasible, AggressiveUndef);
  
  for (unsigned i = 0, e = TI->getNumSuccessors(); i != e; ++i)
    if (TI->getSuccessor(i) == To && SuccFeasible[i])
      return true;
  
  return false;
}

void SparseSolver::visitTerminatorInst(TerminatorInst &TI) {
  SmallVector<bool, 16> SuccFeasible;
  getFeasibleSuccessors(TI, SuccFeasible, true);
  
  BasicBlock *BB = TI.getParent();
  
  // Mark all feasible successors executable...
  for (unsigned i = 0, e = SuccFeasible.size(); i != e; ++i)
    if (SuccFeasible[i])
      markEdgeExecutable(BB, TI.getSuccessor(i));
}

void SparseSolver::visitPHINode(PHINode &PN) {
  // The lattice function may store more information on a PHINode than could be
  // computed from its incoming values.  For example, SSI form stores its sigma
  // functions as PHINodes with a single incoming value.
  if (LatticeFunc->IsSpecialCasedPHI(&PN)) {
    LatticeVal IV = LatticeFunc->ComputeInstructionState(PN, *this);
    if (IV != LatticeFunc->getUntrackedVal())
      UpdateState(PN, IV);
    return;
  }

  LatticeVal PNIV = getOrInitValueState(&PN);
  LatticeVal Overdefined = LatticeFunc->getOverdefinedVal();
  
  // If this value is already overdefined (common) just return.
  if (PNIV == Overdefined || PNIV == LatticeFunc->getUntrackedVal())
    return;  // Quick exit
  
  // Super-extra-high-degree PHI nodes are unlikely to ever be interesting,
  // and slow us down a lot.  Just mark them overdefined.
  if (PN.getNumIncomingValues() > 64) {
    UpdateState(PN, Overdefined);
    return;
  }
  
  // Look at all of the executable operands of the PHI node.  If any of them
  // are overdefined, the PHI becomes overdefined as well.  Otherwise, ask the
  // transfer function to give us the merge of the incoming values.
  for (unsigned i = 0, e = PN.getNumIncomingValues(); i != e; ++i) {
    // If the edge is not yet known to be feasible, it doesn't impact the PHI.
    if (!isEdgeFeasible(PN.getIncomingBlock(i), PN.getParent(), true))
      continue;
    
    // Merge in this value.
    LatticeVal OpVal = getOrInitValueState(PN.getIncomingValue(i));
    if (OpVal != PNIV)
      PNIV = LatticeFunc->MergeValues(PNIV, OpVal);
    
    if (PNIV == Overdefined)
      break;  // Rest of input values don't matter.
  }

  // Update the PHI with the compute value, which is the merge of the inputs.
  UpdateState(PN, PNIV);
}


void SparseSolver::visitInst(Instruction &I) {
  // PHIs are handled by the propagation logic, they are never passed into the
  // transfer functions.
  if (PHINode *PN = dyn_cast<PHINode>(&I))
    return visitPHINode(*PN);
  
  // Otherwise, ask the transfer function what the result is.  If this is
  // something that we care about, remember it.
  LatticeVal IV = LatticeFunc->ComputeInstructionState(I, *this);
  if (IV != LatticeFunc->getUntrackedVal())
    UpdateState(I, IV);
  
  if (TerminatorInst *TI = dyn_cast<TerminatorInst>(&I))
    visitTerminatorInst(*TI);
}

void SparseSolver::Solve(Function &F) {
  MarkBlockExecutable(&F.getEntryBlock());
  
  // Process the work lists until they are empty!
  while (!BBWorkList.empty() || !InstWorkList.empty()) {
    // Process the instruction work list.
    while (!InstWorkList.empty()) {
      Instruction *I = InstWorkList.back();
      InstWorkList.pop_back();

      DEBUG(dbgs() << "\nPopped off I-WL: " << *I << "\n");

      // "I" got into the work list because it made a transition.  See if any
      // users are both live and in need of updating.
      for (User *U : I->users()) {
        Instruction *UI = cast<Instruction>(U);
        if (BBExecutable.count(UI->getParent()))   // Inst is executable?
          visitInst(*UI);
      }
    }

    // Process the basic block work list.
    while (!BBWorkList.empty()) {
      BasicBlock *BB = BBWorkList.back();
      BBWorkList.pop_back();

      DEBUG(dbgs() << "\nPopped off BBWL: " << *BB);

      // Notify all instructions in this basic block that they are newly
      // executable.
      for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I)
        visitInst(*I);
    }
  }
}

void SparseSolver::Print(Function &F, raw_ostream &OS) const {
  OS << "\nFUNCTION: " << F.getName() << "\n";
  for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB) {
    if (!BBExecutable.count(BB))
      OS << "INFEASIBLE: ";
    OS << "\t";
    if (BB->hasName())
      OS << BB->getName() << ":\n";
    else
      OS << "; anon bb\n";
    for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I) {
      LatticeFunc->PrintValue(getLatticeState(I), OS);
      OS << *I << "\n";
    }
    
    OS << "\n";
  }
}


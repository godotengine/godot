//===- MergedLoadStoreMotion.cpp - merge and hoist/sink load/stores -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//! \file
//! \brief This pass performs merges of loads and stores on both sides of a
//  diamond (hammock). It hoists the loads and sinks the stores.
//
// The algorithm iteratively hoists two loads to the same address out of a
// diamond (hammock) and merges them into a single load in the header. Similar
// it sinks and merges two stores to the tail block (footer). The algorithm
// iterates over the instructions of one side of the diamond and attempts to
// find a matching load/store on the other side. It hoists / sinks when it
// thinks it safe to do so.  This optimization helps with eg. hiding load
// latencies, triggering if-conversion, and reducing static code size.
//
//===----------------------------------------------------------------------===//
//
//
// Example:
// Diamond shaped code before merge:
//
//            header:
//                     br %cond, label %if.then, label %if.else
//                        +                    +
//                       +                      +
//                      +                        +
//            if.then:                         if.else:
//               %lt = load %addr_l               %le = load %addr_l
//               <use %lt>                        <use %le>
//               <...>                            <...>
//               store %st, %addr_s               store %se, %addr_s
//               br label %if.end                 br label %if.end
//                     +                         +
//                      +                       +
//                       +                     +
//            if.end ("footer"):
//                     <...>
//
// Diamond shaped code after merge:
//
//            header:
//                     %l = load %addr_l
//                     br %cond, label %if.then, label %if.else
//                        +                    +
//                       +                      +
//                      +                        +
//            if.then:                         if.else:
//               <use %l>                         <use %l>
//               <...>                            <...>
//               br label %if.end                 br label %if.end
//                      +                        +
//                       +                      +
//                        +                    +
//            if.end ("footer"):
//                     %s.sink = phi [%st, if.then], [%se, if.else]
//                     <...>
//                     store %s.sink, %addr_s
//                     <...>
//
//
//===----------------------- TODO -----------------------------------------===//
//
// 1) Generalize to regions other than diamonds
// 2) Be more aggressive merging memory operations
// Note that both changes require register pressure control
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/Analysis/MemoryDependenceAnalysis.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/SSAUpdater.h"
#include "llvm/Transforms/Utils/Local.h"
#include <vector>
using namespace llvm;

#define DEBUG_TYPE "mldst-motion"

//===----------------------------------------------------------------------===//
//                         MergedLoadStoreMotion Pass
//===----------------------------------------------------------------------===//

namespace {
class MergedLoadStoreMotion : public FunctionPass {
  AliasAnalysis *AA;
  MemoryDependenceAnalysis *MD;

public:
  static char ID; // Pass identification, replacement for typeid
  explicit MergedLoadStoreMotion(void)
      : FunctionPass(ID), MD(nullptr), MagicCompileTimeControl(250) {
    initializeMergedLoadStoreMotionPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override;

private:
  // This transformation requires dominator postdominator info
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetLibraryInfoWrapperPass>();
    AU.addRequired<AliasAnalysis>();
    AU.addPreserved<MemoryDependenceAnalysis>();
    AU.addPreserved<AliasAnalysis>();
  }

  // Helper routines

  ///
  /// \brief Remove instruction from parent and update memory dependence
  /// analysis.
  ///
  void removeInstruction(Instruction *Inst);
  BasicBlock *getDiamondTail(BasicBlock *BB);
  bool isDiamondHead(BasicBlock *BB);
  // Routines for hoisting loads
  bool isLoadHoistBarrierInRange(const Instruction& Start,
                                 const Instruction& End,
                                 LoadInst* LI);
  LoadInst *canHoistFromBlock(BasicBlock *BB, LoadInst *LI);
  void hoistInstruction(BasicBlock *BB, Instruction *HoistCand,
                        Instruction *ElseInst);
  bool isSafeToHoist(Instruction *I) const;
  bool hoistLoad(BasicBlock *BB, LoadInst *HoistCand, LoadInst *ElseInst);
  bool mergeLoads(BasicBlock *BB);
  // Routines for sinking stores
  StoreInst *canSinkFromBlock(BasicBlock *BB, StoreInst *SI);
  PHINode *getPHIOperand(BasicBlock *BB, StoreInst *S0, StoreInst *S1);
  bool isStoreSinkBarrierInRange(const Instruction &Start,
                                 const Instruction &End, MemoryLocation Loc);
  bool sinkStore(BasicBlock *BB, StoreInst *SinkCand, StoreInst *ElseInst);
  bool mergeStores(BasicBlock *BB);
  // The mergeLoad/Store algorithms could have Size0 * Size1 complexity,
  // where Size0 and Size1 are the #instructions on the two sides of
  // the diamond. The constant chosen here is arbitrary. Compiler Time
  // Control is enforced by the check Size0 * Size1 < MagicCompileTimeControl.
  const int MagicCompileTimeControl;
};

char MergedLoadStoreMotion::ID = 0;
}

///
/// \brief createMergedLoadStoreMotionPass - The public interface to this file.
///
FunctionPass *llvm::createMergedLoadStoreMotionPass() {
  return new MergedLoadStoreMotion();
}

INITIALIZE_PASS_BEGIN(MergedLoadStoreMotion, "mldst-motion",
                      "MergedLoadStoreMotion", false, false)
INITIALIZE_PASS_DEPENDENCY(MemoryDependenceAnalysis)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_AG_DEPENDENCY(AliasAnalysis)
INITIALIZE_PASS_END(MergedLoadStoreMotion, "mldst-motion",
                    "MergedLoadStoreMotion", false, false)

///
/// \brief Remove instruction from parent and update memory dependence analysis.
///
void MergedLoadStoreMotion::removeInstruction(Instruction *Inst) {
  // Notify the memory dependence analysis.
  if (MD) {
    MD->removeInstruction(Inst);
    if (LoadInst *LI = dyn_cast<LoadInst>(Inst))
      MD->invalidateCachedPointerInfo(LI->getPointerOperand());
    if (Inst->getType()->getScalarType()->isPointerTy()) {
      MD->invalidateCachedPointerInfo(Inst);
    }
  }
  Inst->eraseFromParent();
}

///
/// \brief Return tail block of a diamond.
///
BasicBlock *MergedLoadStoreMotion::getDiamondTail(BasicBlock *BB) {
  assert(isDiamondHead(BB) && "Basic block is not head of a diamond");
  BranchInst *BI = (BranchInst *)(BB->getTerminator());
  BasicBlock *Succ0 = BI->getSuccessor(0);
  BasicBlock *Tail = Succ0->getTerminator()->getSuccessor(0);
  return Tail;
}

///
/// \brief True when BB is the head of a diamond (hammock)
///
bool MergedLoadStoreMotion::isDiamondHead(BasicBlock *BB) {
  if (!BB)
    return false;
  if (!isa<BranchInst>(BB->getTerminator()))
    return false;
  if (BB->getTerminator()->getNumSuccessors() != 2)
    return false;

  BranchInst *BI = (BranchInst *)(BB->getTerminator());
  BasicBlock *Succ0 = BI->getSuccessor(0);
  BasicBlock *Succ1 = BI->getSuccessor(1);

  if (!Succ0->getSinglePredecessor() ||
      Succ0->getTerminator()->getNumSuccessors() != 1)
    return false;
  if (!Succ1->getSinglePredecessor() ||
      Succ1->getTerminator()->getNumSuccessors() != 1)
    return false;

  BasicBlock *Tail = Succ0->getTerminator()->getSuccessor(0);
  // Ignore triangles.
  if (Succ1->getTerminator()->getSuccessor(0) != Tail)
    return false;
  return true;
}

///
/// \brief True when instruction is a hoist barrier for a load
///
/// Whenever an instruction could possibly modify the value
/// being loaded or protect against the load from happening
/// it is considered a hoist barrier.
///

bool MergedLoadStoreMotion::isLoadHoistBarrierInRange(const Instruction& Start, 
                                                      const Instruction& End,
                                                      LoadInst* LI) {
  MemoryLocation Loc = MemoryLocation::get(LI);
  return AA->canInstructionRangeModRef(Start, End, Loc, AliasAnalysis::Mod);
}

///
/// \brief Decide if a load can be hoisted
///
/// When there is a load in \p BB to the same address as \p LI
/// and it can be hoisted from \p BB, return that load.
/// Otherwise return Null.
///
LoadInst *MergedLoadStoreMotion::canHoistFromBlock(BasicBlock *BB1,
                                                   LoadInst *Load0) {

  for (BasicBlock::iterator BBI = BB1->begin(), BBE = BB1->end(); BBI != BBE;
       ++BBI) {
    Instruction *Inst = BBI;

    // Only merge and hoist loads when their result in used only in BB
    if (!isa<LoadInst>(Inst) || Inst->isUsedOutsideOfBlock(BB1))
      continue;

    LoadInst *Load1 = dyn_cast<LoadInst>(Inst);
    BasicBlock *BB0 = Load0->getParent();

    MemoryLocation Loc0 = MemoryLocation::get(Load0);
    MemoryLocation Loc1 = MemoryLocation::get(Load1);
    if (AA->isMustAlias(Loc0, Loc1) && Load0->isSameOperationAs(Load1) &&
        !isLoadHoistBarrierInRange(BB1->front(), *Load1, Load1) &&
        !isLoadHoistBarrierInRange(BB0->front(), *Load0, Load0)) {
      return Load1;
    }
  }
  return nullptr;
}

///
/// \brief Merge two equivalent instructions \p HoistCand and \p ElseInst into
/// \p BB
///
/// BB is the head of a diamond
///
void MergedLoadStoreMotion::hoistInstruction(BasicBlock *BB,
                                             Instruction *HoistCand,
                                             Instruction *ElseInst) {
  DEBUG(dbgs() << " Hoist Instruction into BB \n"; BB->dump();
        dbgs() << "Instruction Left\n"; HoistCand->dump(); dbgs() << "\n";
        dbgs() << "Instruction Right\n"; ElseInst->dump(); dbgs() << "\n");
  // Hoist the instruction.
  assert(HoistCand->getParent() != BB);

  // Intersect optional metadata.
  HoistCand->intersectOptionalDataWith(ElseInst);
  combineMetadata(HoistCand, ElseInst, None);     // HLSL Change: Preserve DXIL metadata

  // Prepend point for instruction insert
  Instruction *HoistPt = BB->getTerminator();

  // Merged instruction
  Instruction *HoistedInst = HoistCand->clone();

  // Hoist instruction.
  HoistedInst->insertBefore(HoistPt);

  HoistCand->replaceAllUsesWith(HoistedInst);
  removeInstruction(HoistCand);
  // Replace the else block instruction.
  ElseInst->replaceAllUsesWith(HoistedInst);
  removeInstruction(ElseInst);
}

///
/// \brief Return true if no operand of \p I is defined in I's parent block
///
bool MergedLoadStoreMotion::isSafeToHoist(Instruction *I) const {
  BasicBlock *Parent = I->getParent();
  for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i) {
    Instruction *Instr = dyn_cast<Instruction>(I->getOperand(i));
    if (Instr && Instr->getParent() == Parent)
      return false;
  }
  return true;
}

///
/// \brief Merge two equivalent loads and GEPs and hoist into diamond head
///
bool MergedLoadStoreMotion::hoistLoad(BasicBlock *BB, LoadInst *L0,
                                      LoadInst *L1) {
  // Only one definition?
  Instruction *A0 = dyn_cast<Instruction>(L0->getPointerOperand());
  Instruction *A1 = dyn_cast<Instruction>(L1->getPointerOperand());
  if (A0 && A1 && A0->isIdenticalTo(A1) && isSafeToHoist(A0) &&
      A0->hasOneUse() && (A0->getParent() == L0->getParent()) &&
      A1->hasOneUse() && (A1->getParent() == L1->getParent()) &&
      isa<GetElementPtrInst>(A0)) {
    DEBUG(dbgs() << "Hoist Instruction into BB \n"; BB->dump();
          dbgs() << "Instruction Left\n"; L0->dump(); dbgs() << "\n";
          dbgs() << "Instruction Right\n"; L1->dump(); dbgs() << "\n");
    hoistInstruction(BB, A0, A1);
    hoistInstruction(BB, L0, L1);
    return true;
  } else
    return false;
}

///
/// \brief Try to hoist two loads to same address into diamond header
///
/// Starting from a diamond head block, iterate over the instructions in one
/// successor block and try to match a load in the second successor.
///
bool MergedLoadStoreMotion::mergeLoads(BasicBlock *BB) {
  bool MergedLoads = false;
  assert(isDiamondHead(BB));
  BranchInst *BI = dyn_cast<BranchInst>(BB->getTerminator());
  BasicBlock *Succ0 = BI->getSuccessor(0);
  BasicBlock *Succ1 = BI->getSuccessor(1);
  // #Instructions in Succ1 for Compile Time Control
  // int Size1 = Succ1->size(); // HLSL Change
  int Size1 = Succ1->compute_size_no_dbg(); // HLSL Change
  int NLoads = 0;
  for (BasicBlock::iterator BBI = Succ0->begin(), BBE = Succ0->end();
       BBI != BBE;) {

    Instruction *I = BBI;
    ++BBI;

    // Only move non-simple (atomic, volatile) loads.
    LoadInst *L0 = dyn_cast<LoadInst>(I);
    if (!L0 || !L0->isSimple() || L0->isUsedOutsideOfBlock(Succ0))
      continue;

    ++NLoads;
    if (NLoads * Size1 >= MagicCompileTimeControl)
      break;
    if (LoadInst *L1 = canHoistFromBlock(Succ1, L0)) {
      bool Res = hoistLoad(BB, L0, L1);
      MergedLoads |= Res;
      // Don't attempt to hoist above loads that had not been hoisted.
      if (!Res)
        break;
    }
  }
  return MergedLoads;
}

///
/// \brief True when instruction is a sink barrier for a store
/// located in Loc
///
/// Whenever an instruction could possibly read or modify the
/// value being stored or protect against the store from
/// happening it is considered a sink barrier.
///

bool MergedLoadStoreMotion::isStoreSinkBarrierInRange(const Instruction &Start,
                                                      const Instruction &End,
                                                      MemoryLocation Loc) {
  return AA->canInstructionRangeModRef(Start, End, Loc, AliasAnalysis::ModRef);
}

///
/// \brief Check if \p BB contains a store to the same address as \p SI
///
/// \return The store in \p  when it is safe to sink. Otherwise return Null.
///
StoreInst *MergedLoadStoreMotion::canSinkFromBlock(BasicBlock *BB1,
                                                   StoreInst *Store0) {
  DEBUG(dbgs() << "can Sink? : "; Store0->dump(); dbgs() << "\n");
  BasicBlock *BB0 = Store0->getParent();
  for (BasicBlock::reverse_iterator RBI = BB1->rbegin(), RBE = BB1->rend();
       RBI != RBE; ++RBI) {
    Instruction *Inst = &*RBI;

    if (!isa<StoreInst>(Inst))
       continue;

    StoreInst *Store1 = cast<StoreInst>(Inst);

    MemoryLocation Loc0 = MemoryLocation::get(Store0);
    MemoryLocation Loc1 = MemoryLocation::get(Store1);
    if (AA->isMustAlias(Loc0, Loc1) && Store0->isSameOperationAs(Store1) &&
      !isStoreSinkBarrierInRange(*(std::next(BasicBlock::iterator(Store1))),
                                 BB1->back(), Loc1) &&
      !isStoreSinkBarrierInRange(*(std::next(BasicBlock::iterator(Store0))),
                                 BB0->back(), Loc0)) {
      return Store1;
    }
  }
  return nullptr;
}

///
/// \brief Create a PHI node in BB for the operands of S0 and S1
///
PHINode *MergedLoadStoreMotion::getPHIOperand(BasicBlock *BB, StoreInst *S0,
                                              StoreInst *S1) {
  // Create a phi if the values mismatch.
  PHINode *NewPN = 0;
  Value *Opd1 = S0->getValueOperand();
  Value *Opd2 = S1->getValueOperand();
  if (Opd1 != Opd2) {
    NewPN = PHINode::Create(Opd1->getType(), 2, Opd2->getName() + ".sink",
                            BB->begin());
    NewPN->addIncoming(Opd1, S0->getParent());
    NewPN->addIncoming(Opd2, S1->getParent());
    if (NewPN->getType()->getScalarType()->isPointerTy()) {
      // AA needs to be informed when a PHI-use of the pointer value is added
      for (unsigned I = 0, E = NewPN->getNumIncomingValues(); I != E; ++I) {
        unsigned J = PHINode::getOperandNumForIncomingValue(I);
        AA->addEscapingUse(NewPN->getOperandUse(J));
      }
      if (MD)
        MD->invalidateCachedPointerInfo(NewPN);
    }
  }
  return NewPN;
}

///
/// \brief Merge two stores to same address and sink into \p BB
///
/// Also sinks GEP instruction computing the store address
///
bool MergedLoadStoreMotion::sinkStore(BasicBlock *BB, StoreInst *S0,
                                      StoreInst *S1) {
  // Only one definition?
  Instruction *A0 = dyn_cast<Instruction>(S0->getPointerOperand());
  Instruction *A1 = dyn_cast<Instruction>(S1->getPointerOperand());
  if (A0 && A1 && A0->isIdenticalTo(A1) && A0->hasOneUse() &&
      (A0->getParent() == S0->getParent()) && A1->hasOneUse() &&
      (A1->getParent() == S1->getParent()) && isa<GetElementPtrInst>(A0)) {
    DEBUG(dbgs() << "Sink Instruction into BB \n"; BB->dump();
          dbgs() << "Instruction Left\n"; S0->dump(); dbgs() << "\n";
          dbgs() << "Instruction Right\n"; S1->dump(); dbgs() << "\n");
    // Hoist the instruction.
    BasicBlock::iterator InsertPt = BB->getFirstInsertionPt();
    // Intersect optional metadata.
    S0->intersectOptionalDataWith(S1);
    combineMetadata(S0, S1, None);      // HLSL Change: Preserve DXIL metadata

    // Create the new store to be inserted at the join point.
    StoreInst *SNew = (StoreInst *)(S0->clone());
    Instruction *ANew = A0->clone();
    SNew->insertBefore(InsertPt);
    ANew->insertBefore(SNew);

    assert(S0->getParent() == A0->getParent());
    assert(S1->getParent() == A1->getParent());

    PHINode *NewPN = getPHIOperand(BB, S0, S1);
    // New PHI operand? Use it.
    if (NewPN)
      SNew->setOperand(0, NewPN);
    removeInstruction(S0);
    removeInstruction(S1);
    A0->replaceAllUsesWith(ANew);
    removeInstruction(A0);
    A1->replaceAllUsesWith(ANew);
    removeInstruction(A1);
    return true;
  }
  return false;
}

///
/// \brief True when two stores are equivalent and can sink into the footer
///
/// Starting from a diamond tail block, iterate over the instructions in one
/// predecessor block and try to match a store in the second predecessor.
///
bool MergedLoadStoreMotion::mergeStores(BasicBlock *T) {

  bool MergedStores = false;
  assert(T && "Footer of a diamond cannot be empty");

  pred_iterator PI = pred_begin(T), E = pred_end(T);
  assert(PI != E);
  BasicBlock *Pred0 = *PI;
  ++PI;
  BasicBlock *Pred1 = *PI;
  ++PI;
  // tail block  of a diamond/hammock?
  if (Pred0 == Pred1)
    return false; // No.
  if (PI != E)
    return false; // No. More than 2 predecessors.

  // #Instructions in Succ1 for Compile Time Control
  // int Size1 = Succ1->size(); // HLSL Change
  int Size1 = Pred1->compute_size_no_dbg(); // HLSL Change
  int NStores = 0;

  for (BasicBlock::reverse_iterator RBI = Pred0->rbegin(), RBE = Pred0->rend();
       RBI != RBE;) {

    Instruction *I = &*RBI;
    ++RBI;

    // Sink move non-simple (atomic, volatile) stores
    if (!isa<StoreInst>(I))
      continue;
    StoreInst *S0 = (StoreInst *)I;
    if (!S0->isSimple())
      continue;

    ++NStores;
    if (NStores * Size1 >= MagicCompileTimeControl)
      break;
    if (StoreInst *S1 = canSinkFromBlock(Pred1, S0)) {
      bool Res = sinkStore(T, S0, S1);
      MergedStores |= Res;
      // Don't attempt to sink below stores that had to stick around
      // But after removal of a store and some of its feeding
      // instruction search again from the beginning since the iterator
      // is likely stale at this point.
      if (!Res)
        break;
      else {
        RBI = Pred0->rbegin();
        RBE = Pred0->rend();
        DEBUG(dbgs() << "Search again\n"; Instruction *I = &*RBI; I->dump());
      }
    }
  }
  return MergedStores;
}
///
/// \brief Run the transformation for each function
///
bool MergedLoadStoreMotion::runOnFunction(Function &F) {
  MD = getAnalysisIfAvailable<MemoryDependenceAnalysis>();
  AA = &getAnalysis<AliasAnalysis>();

  bool Changed = false;
  DEBUG(dbgs() << "Instruction Merger\n");

  // Merge unconditional branches, allowing PRE to catch more
  // optimization opportunities.
  for (Function::iterator FI = F.begin(), FE = F.end(); FI != FE;) {
    BasicBlock *BB = FI++;

    // Hoist equivalent loads and sink stores
    // outside diamonds when possible
    if (isDiamondHead(BB)) {
      Changed |= mergeLoads(BB);
      Changed |= mergeStores(getDiamondTail(BB));
    }
  }
  return Changed;
}

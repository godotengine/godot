//===- ConstantHoisting.cpp - Prepare code for expensive constants --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass identifies expensive constants to hoist and coalesces them to
// better prepare it for SelectionDAG-based code generation. This works around
// the limitations of the basic-block-at-a-time approach.
//
// First it scans all instructions for integer constants and calculates its
// cost. If the constant can be folded into the instruction (the cost is
// TCC_Free) or the cost is just a simple operation (TCC_BASIC), then we don't
// consider it expensive and leave it alone. This is the default behavior and
// the default implementation of getIntImmCost will always return TCC_Free.
//
// If the cost is more than TCC_BASIC, then the integer constant can't be folded
// into the instruction and it might be beneficial to hoist the constant.
// Similar constants are coalesced to reduce register pressure and
// materialization code.
//
// When a constant is hoisted, it is also hidden behind a bitcast to force it to
// be live-out of the basic block. Otherwise the constant would be just
// duplicated and each basic block would have its own copy in the SelectionDAG.
// The SelectionDAG recognizes such constants as opaque and doesn't perform
// certain transformations on them, which would create a new expensive constant.
//
// This optimization is only applied to integer constants in instructions and
// simple (this means not nested) constant cast expressions. For example:
// %0 = load i64* inttoptr (i64 big_constant to i64*)
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <tuple>

using namespace llvm;

#define DEBUG_TYPE "consthoist"

STATISTIC(NumConstantsHoisted, "Number of constants hoisted");
STATISTIC(NumConstantsRebased, "Number of constants rebased");

namespace {
struct ConstantUser;
struct RebasedConstantInfo;

typedef SmallVector<ConstantUser, 8> ConstantUseListType;
typedef SmallVector<RebasedConstantInfo, 4> RebasedConstantListType;

/// \brief Keeps track of the user of a constant and the operand index where the
/// constant is used.
struct ConstantUser {
  Instruction *Inst;
  unsigned OpndIdx;

  ConstantUser(Instruction *Inst, unsigned Idx) : Inst(Inst), OpndIdx(Idx) { }
};

/// \brief Keeps track of a constant candidate and its uses.
struct ConstantCandidate {
  ConstantUseListType Uses;
  ConstantInt *ConstInt;
  unsigned CumulativeCost;

  ConstantCandidate(ConstantInt *ConstInt)
    : ConstInt(ConstInt), CumulativeCost(0) { }

  /// \brief Add the user to the use list and update the cost.
  void addUser(Instruction *Inst, unsigned Idx, unsigned Cost) {
    CumulativeCost += Cost;
    Uses.push_back(ConstantUser(Inst, Idx));
  }
};

/// \brief This represents a constant that has been rebased with respect to a
/// base constant. The difference to the base constant is recorded in Offset.
struct RebasedConstantInfo {
  ConstantUseListType Uses;
  Constant *Offset;

  RebasedConstantInfo(ConstantUseListType &&Uses, Constant *Offset)
    : Uses(std::move(Uses)), Offset(Offset) { }
};

/// \brief A base constant and all its rebased constants.
struct ConstantInfo {
  ConstantInt *BaseConstant;
  RebasedConstantListType RebasedConstants;
};

/// \brief The constant hoisting pass.
class ConstantHoisting : public FunctionPass {
  typedef DenseMap<ConstantInt *, unsigned> ConstCandMapType;
  typedef std::vector<ConstantCandidate> ConstCandVecType;

  const TargetTransformInfo *TTI;
  DominatorTree *DT;
  BasicBlock *Entry;

  /// Keeps track of constant candidates found in the function.
  ConstCandVecType ConstCandVec;

  /// Keep track of cast instructions we already cloned.
  SmallDenseMap<Instruction *, Instruction *> ClonedCastMap;

  /// These are the final constants we decided to hoist.
  SmallVector<ConstantInfo, 8> ConstantVec;
public:
  static char ID; // Pass identification, replacement for typeid
  ConstantHoisting() : FunctionPass(ID), TTI(nullptr), DT(nullptr),
                       Entry(nullptr) {
    initializeConstantHoistingPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &Fn) override;

  StringRef getPassName() const override { return "Constant Hoisting"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<TargetTransformInfoWrapperPass>();
  }

private:
  /// \brief Initialize the pass.
  void setup(Function &Fn) {
    DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    TTI = &getAnalysis<TargetTransformInfoWrapperPass>().getTTI(Fn);
    Entry = &Fn.getEntryBlock();
  }

  /// \brief Cleanup.
  void cleanup() {
    ConstantVec.clear();
    ClonedCastMap.clear();
    ConstCandVec.clear();

    TTI = nullptr;
    DT = nullptr;
    Entry = nullptr;
  }

  Instruction *findMatInsertPt(Instruction *Inst, unsigned Idx = ~0U) const;
  Instruction *findConstantInsertionPoint(const ConstantInfo &ConstInfo) const;
  void collectConstantCandidates(ConstCandMapType &ConstCandMap,
                                 Instruction *Inst, unsigned Idx,
                                 ConstantInt *ConstInt);
  void collectConstantCandidates(ConstCandMapType &ConstCandMap,
                                 Instruction *Inst);
  void collectConstantCandidates(Function &Fn);
  void findAndMakeBaseConstant(ConstCandVecType::iterator S,
                               ConstCandVecType::iterator E);
  void findBaseConstants();
  void emitBaseConstants(Instruction *Base, Constant *Offset,
                         const ConstantUser &ConstUser);
  bool emitBaseConstants();
  void deleteDeadCastInst() const;
  bool optimizeConstants(Function &Fn);
};
}

char ConstantHoisting::ID = 0;
INITIALIZE_PASS_BEGIN(ConstantHoisting, "consthoist", "Constant Hoisting",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_END(ConstantHoisting, "consthoist", "Constant Hoisting",
                    false, false)

FunctionPass *llvm::createConstantHoistingPass() {
  return new ConstantHoisting();
}

/// \brief Perform the constant hoisting optimization for the given function.
bool ConstantHoisting::runOnFunction(Function &Fn) {
  if (skipOptnoneFunction(Fn))
    return false;

  DEBUG(dbgs() << "********** Begin Constant Hoisting **********\n");
  DEBUG(dbgs() << "********** Function: " << Fn.getName() << '\n');

  setup(Fn);

  bool MadeChange = optimizeConstants(Fn);

  if (MadeChange) {
    DEBUG(dbgs() << "********** Function after Constant Hoisting: "
                 << Fn.getName() << '\n');
    DEBUG(dbgs() << Fn);
  }
  DEBUG(dbgs() << "********** End Constant Hoisting **********\n");

  cleanup();

  return MadeChange;
}


/// \brief Find the constant materialization insertion point.
Instruction *ConstantHoisting::findMatInsertPt(Instruction *Inst,
                                               unsigned Idx) const {
  // If the operand is a cast instruction, then we have to materialize the
  // constant before the cast instruction.
  if (Idx != ~0U) {
    Value *Opnd = Inst->getOperand(Idx);
    if (auto CastInst = dyn_cast<Instruction>(Opnd))
      if (CastInst->isCast())
        return CastInst;
  }

  // The simple and common case. This also includes constant expressions.
  if (!isa<PHINode>(Inst) && !isa<LandingPadInst>(Inst))
    return Inst;

  // We can't insert directly before a phi node or landing pad. Insert before
  // the terminator of the incoming or dominating block.
  assert(Entry != Inst->getParent() && "PHI or landing pad in entry block!");
  if (Idx != ~0U && isa<PHINode>(Inst))
    return cast<PHINode>(Inst)->getIncomingBlock(Idx)->getTerminator();

  BasicBlock *IDom = DT->getNode(Inst->getParent())->getIDom()->getBlock();
  return IDom->getTerminator();
}

/// \brief Find an insertion point that dominates all uses.
Instruction *ConstantHoisting::
findConstantInsertionPoint(const ConstantInfo &ConstInfo) const {
  assert(!ConstInfo.RebasedConstants.empty() && "Invalid constant info entry.");
  // Collect all basic blocks.
  SmallPtrSet<BasicBlock *, 8> BBs;
  for (auto const &RCI : ConstInfo.RebasedConstants)
    for (auto const &U : RCI.Uses)
      BBs.insert(findMatInsertPt(U.Inst, U.OpndIdx)->getParent());

  if (BBs.count(Entry))
    return &Entry->front();

  while (BBs.size() >= 2) {
    BasicBlock *BB, *BB1, *BB2;
    BB1 = *BBs.begin();
    BB2 = *std::next(BBs.begin());
    BB = DT->findNearestCommonDominator(BB1, BB2);
    if (BB == Entry)
      return &Entry->front();
    BBs.erase(BB1);
    BBs.erase(BB2);
    BBs.insert(BB);
  }
  assert((BBs.size() == 1) && "Expected only one element.");
  Instruction &FirstInst = (*BBs.begin())->front();
  return findMatInsertPt(&FirstInst);
}


/// \brief Record constant integer ConstInt for instruction Inst at operand
/// index Idx.
///
/// The operand at index Idx is not necessarily the constant integer itself. It
/// could also be a cast instruction or a constant expression that uses the
// constant integer.
void ConstantHoisting::collectConstantCandidates(ConstCandMapType &ConstCandMap,
                                                 Instruction *Inst,
                                                 unsigned Idx,
                                                 ConstantInt *ConstInt) {
  unsigned Cost;
  // Ask the target about the cost of materializing the constant for the given
  // instruction and operand index.
  if (auto IntrInst = dyn_cast<IntrinsicInst>(Inst))
    Cost = TTI->getIntImmCost(IntrInst->getIntrinsicID(), Idx,
                              ConstInt->getValue(), ConstInt->getType());
  else
    Cost = TTI->getIntImmCost(Inst->getOpcode(), Idx, ConstInt->getValue(),
                              ConstInt->getType());

  // Ignore cheap integer constants.
  if (Cost > TargetTransformInfo::TCC_Basic) {
    ConstCandMapType::iterator Itr;
    bool Inserted;
    std::tie(Itr, Inserted) = ConstCandMap.insert(std::make_pair(ConstInt, 0));
    if (Inserted) {
      ConstCandVec.push_back(ConstantCandidate(ConstInt));
      Itr->second = ConstCandVec.size() - 1;
    }
    ConstCandVec[Itr->second].addUser(Inst, Idx, Cost);
    DEBUG(if (isa<ConstantInt>(Inst->getOperand(Idx)))
            dbgs() << "Collect constant " << *ConstInt << " from " << *Inst
                   << " with cost " << Cost << '\n';
          else
          dbgs() << "Collect constant " << *ConstInt << " indirectly from "
                 << *Inst << " via " << *Inst->getOperand(Idx) << " with cost "
                 << Cost << '\n';
    );
  }
}

/// \brief Scan the instruction for expensive integer constants and record them
/// in the constant candidate vector.
void ConstantHoisting::collectConstantCandidates(ConstCandMapType &ConstCandMap,
                                                 Instruction *Inst) {
  // Skip all cast instructions. They are visited indirectly later on.
  if (Inst->isCast())
    return;

  // Can't handle inline asm. Skip it.
  if (auto Call = dyn_cast<CallInst>(Inst))
    if (isa<InlineAsm>(Call->getCalledValue()))
      return;

  // Scan all operands.
  for (unsigned Idx = 0, E = Inst->getNumOperands(); Idx != E; ++Idx) {
    Value *Opnd = Inst->getOperand(Idx);

    // Visit constant integers.
    if (auto ConstInt = dyn_cast<ConstantInt>(Opnd)) {
      collectConstantCandidates(ConstCandMap, Inst, Idx, ConstInt);
      continue;
    }

    // Visit cast instructions that have constant integers.
    if (auto CastInst = dyn_cast<Instruction>(Opnd)) {
      // Only visit cast instructions, which have been skipped. All other
      // instructions should have already been visited.
      if (!CastInst->isCast())
        continue;

      if (auto *ConstInt = dyn_cast<ConstantInt>(CastInst->getOperand(0))) {
        // Pretend the constant is directly used by the instruction and ignore
        // the cast instruction.
        collectConstantCandidates(ConstCandMap, Inst, Idx, ConstInt);
        continue;
      }
    }

    // Visit constant expressions that have constant integers.
    if (auto ConstExpr = dyn_cast<ConstantExpr>(Opnd)) {
      // Only visit constant cast expressions.
      if (!ConstExpr->isCast())
        continue;

      if (auto ConstInt = dyn_cast<ConstantInt>(ConstExpr->getOperand(0))) {
        // Pretend the constant is directly used by the instruction and ignore
        // the constant expression.
        collectConstantCandidates(ConstCandMap, Inst, Idx, ConstInt);
        continue;
      }
    }
  } // end of for all operands
}

/// \brief Collect all integer constants in the function that cannot be folded
/// into an instruction itself.
void ConstantHoisting::collectConstantCandidates(Function &Fn) {
  ConstCandMapType ConstCandMap;
  for (Function::iterator BB : Fn)
    for (BasicBlock::iterator Inst : *BB)
      collectConstantCandidates(ConstCandMap, Inst);
}

/// \brief Find the base constant within the given range and rebase all other
/// constants with respect to the base constant.
void ConstantHoisting::findAndMakeBaseConstant(ConstCandVecType::iterator S,
                                               ConstCandVecType::iterator E) {
  auto MaxCostItr = S;
  unsigned NumUses = 0;
  // Use the constant that has the maximum cost as base constant.
  for (auto ConstCand = S; ConstCand != E; ++ConstCand) {
    NumUses += ConstCand->Uses.size();
    if (ConstCand->CumulativeCost > MaxCostItr->CumulativeCost)
      MaxCostItr = ConstCand;
  }

  // Don't hoist constants that have only one use.
  if (NumUses <= 1)
    return;

  ConstantInfo ConstInfo;
  ConstInfo.BaseConstant = MaxCostItr->ConstInt;
  Type *Ty = ConstInfo.BaseConstant->getType();

  // Rebase the constants with respect to the base constant.
  for (auto ConstCand = S; ConstCand != E; ++ConstCand) {
    APInt Diff = ConstCand->ConstInt->getValue() -
                 ConstInfo.BaseConstant->getValue();
    Constant *Offset = Diff == 0 ? nullptr : ConstantInt::get(Ty, Diff);
    ConstInfo.RebasedConstants.push_back(
      RebasedConstantInfo(std::move(ConstCand->Uses), Offset));
  }
  ConstantVec.push_back(std::move(ConstInfo));
}

/// \brief Finds and combines constant candidates that can be easily
/// rematerialized with an add from a common base constant.
void ConstantHoisting::findBaseConstants() {
  // Sort the constants by value and type. This invalidates the mapping!
  std::sort(ConstCandVec.begin(), ConstCandVec.end(),
            [](const ConstantCandidate &LHS, const ConstantCandidate &RHS) {
    if (LHS.ConstInt->getType() != RHS.ConstInt->getType())
      return LHS.ConstInt->getType()->getBitWidth() <
             RHS.ConstInt->getType()->getBitWidth();
    return LHS.ConstInt->getValue().ult(RHS.ConstInt->getValue());
  });

  // Simple linear scan through the sorted constant candidate vector for viable
  // merge candidates.
  auto MinValItr = ConstCandVec.begin();
  for (auto CC = std::next(ConstCandVec.begin()), E = ConstCandVec.end();
       CC != E; ++CC) {
    if (MinValItr->ConstInt->getType() == CC->ConstInt->getType()) {
      // Check if the constant is in range of an add with immediate.
      APInt Diff = CC->ConstInt->getValue() - MinValItr->ConstInt->getValue();
      if ((Diff.getBitWidth() <= 64) &&
          TTI->isLegalAddImmediate(Diff.getSExtValue()))
        continue;
    }
    // We either have now a different constant type or the constant is not in
    // range of an add with immediate anymore.
    findAndMakeBaseConstant(MinValItr, CC);
    // Start a new base constant search.
    MinValItr = CC;
  }
  // Finalize the last base constant search.
  findAndMakeBaseConstant(MinValItr, ConstCandVec.end());
}

/// \brief Updates the operand at Idx in instruction Inst with the result of
///        instruction Mat. If the instruction is a PHI node then special
///        handling for duplicate values form the same incomming basic block is
///        required.
/// \return The update will always succeed, but the return value indicated if
///         Mat was used for the update or not.
static bool updateOperand(Instruction *Inst, unsigned Idx, Instruction *Mat) {
  if (auto PHI = dyn_cast<PHINode>(Inst)) {
    // Check if any previous operand of the PHI node has the same incoming basic
    // block. This is a very odd case that happens when the incoming basic block
    // has a switch statement. In this case use the same value as the previous
    // operand(s), otherwise we will fail verification due to different values.
    // The values are actually the same, but the variable names are different
    // and the verifier doesn't like that.
    BasicBlock *IncomingBB = PHI->getIncomingBlock(Idx);
    for (unsigned i = 0; i < Idx; ++i) {
      if (PHI->getIncomingBlock(i) == IncomingBB) {
        Value *IncomingVal = PHI->getIncomingValue(i);
        Inst->setOperand(Idx, IncomingVal);
        return false;
      }
    }
  }

  Inst->setOperand(Idx, Mat);
  return true;
}

/// \brief Emit materialization code for all rebased constants and update their
/// users.
void ConstantHoisting::emitBaseConstants(Instruction *Base, Constant *Offset,
                                         const ConstantUser &ConstUser) {
  Instruction *Mat = Base;
  if (Offset) {
    Instruction *InsertionPt = findMatInsertPt(ConstUser.Inst,
                                               ConstUser.OpndIdx);
    Mat = BinaryOperator::Create(Instruction::Add, Base, Offset,
                                 "const_mat", InsertionPt);

    DEBUG(dbgs() << "Materialize constant (" << *Base->getOperand(0)
                 << " + " << *Offset << ") in BB "
                 << Mat->getParent()->getName() << '\n' << *Mat << '\n');
    Mat->setDebugLoc(ConstUser.Inst->getDebugLoc());
  }
  Value *Opnd = ConstUser.Inst->getOperand(ConstUser.OpndIdx);

  // Visit constant integer.
  if (isa<ConstantInt>(Opnd)) {
    DEBUG(dbgs() << "Update: " << *ConstUser.Inst << '\n');
    if (!updateOperand(ConstUser.Inst, ConstUser.OpndIdx, Mat) && Offset)
      Mat->eraseFromParent();
    DEBUG(dbgs() << "To    : " << *ConstUser.Inst << '\n');
    return;
  }

  // Visit cast instruction.
  if (auto CastInst = dyn_cast<Instruction>(Opnd)) {
    assert(CastInst->isCast() && "Expected an cast instruction!");
    // Check if we already have visited this cast instruction before to avoid
    // unnecessary cloning.
    Instruction *&ClonedCastInst = ClonedCastMap[CastInst];
    if (!ClonedCastInst) {
      ClonedCastInst = CastInst->clone();
      ClonedCastInst->setOperand(0, Mat);
      ClonedCastInst->insertAfter(CastInst);
      // Use the same debug location as the original cast instruction.
      ClonedCastInst->setDebugLoc(CastInst->getDebugLoc());
      DEBUG(dbgs() << "Clone instruction: " << *CastInst << '\n'
                   << "To               : " << *ClonedCastInst << '\n');
    }

    DEBUG(dbgs() << "Update: " << *ConstUser.Inst << '\n');
    updateOperand(ConstUser.Inst, ConstUser.OpndIdx, ClonedCastInst);
    DEBUG(dbgs() << "To    : " << *ConstUser.Inst << '\n');
    return;
  }

  // Visit constant expression.
  if (auto ConstExpr = dyn_cast<ConstantExpr>(Opnd)) {
    Instruction *ConstExprInst = ConstExpr->getAsInstruction();
    ConstExprInst->setOperand(0, Mat);
    ConstExprInst->insertBefore(findMatInsertPt(ConstUser.Inst,
                                                ConstUser.OpndIdx));

    // Use the same debug location as the instruction we are about to update.
    ConstExprInst->setDebugLoc(ConstUser.Inst->getDebugLoc());

    DEBUG(dbgs() << "Create instruction: " << *ConstExprInst << '\n'
                 << "From              : " << *ConstExpr << '\n');
    DEBUG(dbgs() << "Update: " << *ConstUser.Inst << '\n');
    if (!updateOperand(ConstUser.Inst, ConstUser.OpndIdx, ConstExprInst)) {
      ConstExprInst->eraseFromParent();
      if (Offset)
        Mat->eraseFromParent();
    }
    DEBUG(dbgs() << "To    : " << *ConstUser.Inst << '\n');
    return;
  }
}

/// \brief Hoist and hide the base constant behind a bitcast and emit
/// materialization code for derived constants.
bool ConstantHoisting::emitBaseConstants() {
  bool MadeChange = false;
  for (auto const &ConstInfo : ConstantVec) {
    // Hoist and hide the base constant behind a bitcast.
    Instruction *IP = findConstantInsertionPoint(ConstInfo);
    IntegerType *Ty = ConstInfo.BaseConstant->getType();
    Instruction *Base =
      new BitCastInst(ConstInfo.BaseConstant, Ty, "const", IP);
    DEBUG(dbgs() << "Hoist constant (" << *ConstInfo.BaseConstant << ") to BB "
                 << IP->getParent()->getName() << '\n' << *Base << '\n');
    NumConstantsHoisted++;

    // Emit materialization code for all rebased constants.
    for (auto const &RCI : ConstInfo.RebasedConstants) {
      NumConstantsRebased++;
      for (auto const &U : RCI.Uses)
        emitBaseConstants(Base, RCI.Offset, U);
    }

    // Use the same debug location as the last user of the constant.
    assert(!Base->use_empty() && "The use list is empty!?");
    assert(isa<Instruction>(Base->user_back()) &&
           "All uses should be instructions.");
    Base->setDebugLoc(cast<Instruction>(Base->user_back())->getDebugLoc());

    // Correct for base constant, which we counted above too.
    NumConstantsRebased--;
    MadeChange = true;
  }
  return MadeChange;
}

/// \brief Check all cast instructions we made a copy of and remove them if they
/// have no more users.
void ConstantHoisting::deleteDeadCastInst() const {
  for (auto const &I : ClonedCastMap)
    if (I.first->use_empty())
      I.first->eraseFromParent();
}

/// \brief Optimize expensive integer constants in the given function.
bool ConstantHoisting::optimizeConstants(Function &Fn) {
  // Collect all constant candidates.
  collectConstantCandidates(Fn);

  // There are no constant candidates to worry about.
  if (ConstCandVec.empty())
    return false;

  // Combine constants that can be easily materialized with an add from a common
  // base constant.
  findBaseConstants();

  // There are no constants to emit.
  if (ConstantVec.empty())
    return false;

  // Finally hoist the base constant and emit materialization code for dependent
  // constants.
  bool MadeChange = emitBaseConstants();

  // Cleanup dead instructions.
  deleteDeadCastInst();

  return MadeChange;
}

//===-- LoopUtils.cpp - Loop Utility functions -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines common loop utility functions.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Support/Debug.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/LoopUtils.h"

using namespace llvm;
using namespace llvm::PatternMatch;

#define DEBUG_TYPE "loop-utils"

bool RecurrenceDescriptor::areAllUsesIn(Instruction *I,
                                        SmallPtrSetImpl<Instruction *> &Set) {
  for (User::op_iterator Use = I->op_begin(), E = I->op_end(); Use != E; ++Use)
    if (!Set.count(dyn_cast<Instruction>(*Use)))
      return false;
  return true;
}

bool RecurrenceDescriptor::AddReductionVar(PHINode *Phi, RecurrenceKind Kind,
                                           Loop *TheLoop, bool HasFunNoNaNAttr,
                                           RecurrenceDescriptor &RedDes) {
  if (Phi->getNumIncomingValues() != 2)
    return false;

  // Reduction variables are only found in the loop header block.
  if (Phi->getParent() != TheLoop->getHeader())
    return false;

  // Obtain the reduction start value from the value that comes from the loop
  // preheader.
  Value *RdxStart = Phi->getIncomingValueForBlock(TheLoop->getLoopPreheader());

  // ExitInstruction is the single value which is used outside the loop.
  // We only allow for a single reduction value to be used outside the loop.
  // This includes users of the reduction, variables (which form a cycle
  // which ends in the phi node).
  Instruction *ExitInstruction = nullptr;
  // Indicates that we found a reduction operation in our scan.
  bool FoundReduxOp = false;

  // We start with the PHI node and scan for all of the users of this
  // instruction. All users must be instructions that can be used as reduction
  // variables (such as ADD). We must have a single out-of-block user. The cycle
  // must include the original PHI.
  bool FoundStartPHI = false;

  // To recognize min/max patterns formed by a icmp select sequence, we store
  // the number of instruction we saw from the recognized min/max pattern,
  //  to make sure we only see exactly the two instructions.
  unsigned NumCmpSelectPatternInst = 0;
  InstDesc ReduxDesc(false, nullptr);

  SmallPtrSet<Instruction *, 8> VisitedInsts;
  SmallVector<Instruction *, 8> Worklist;
  Worklist.push_back(Phi);
  VisitedInsts.insert(Phi);

  // A value in the reduction can be used:
  //  - By the reduction:
  //      - Reduction operation:
  //        - One use of reduction value (safe).
  //        - Multiple use of reduction value (not safe).
  //      - PHI:
  //        - All uses of the PHI must be the reduction (safe).
  //        - Otherwise, not safe.
  //  - By one instruction outside of the loop (safe).
  //  - By further instructions outside of the loop (not safe).
  //  - By an instruction that is not part of the reduction (not safe).
  //    This is either:
  //      * An instruction type other than PHI or the reduction operation.
  //      * A PHI in the header other than the initial PHI.
  while (!Worklist.empty()) {
    Instruction *Cur = Worklist.back();
    Worklist.pop_back();

    // No Users.
    // If the instruction has no users then this is a broken chain and can't be
    // a reduction variable.
    if (Cur->use_empty())
      return false;

    bool IsAPhi = isa<PHINode>(Cur);

    // A header PHI use other than the original PHI.
    if (Cur != Phi && IsAPhi && Cur->getParent() == Phi->getParent())
      return false;

    // Reductions of instructions such as Div, and Sub is only possible if the
    // LHS is the reduction variable.
    if (!Cur->isCommutative() && !IsAPhi && !isa<SelectInst>(Cur) &&
        !isa<ICmpInst>(Cur) && !isa<FCmpInst>(Cur) &&
        !VisitedInsts.count(dyn_cast<Instruction>(Cur->getOperand(0))))
      return false;

    // Any reduction instruction must be of one of the allowed kinds.
    ReduxDesc = isRecurrenceInstr(Cur, Kind, ReduxDesc, HasFunNoNaNAttr);
    if (!ReduxDesc.isRecurrence())
      return false;

    // A reduction operation must only have one use of the reduction value.
    if (!IsAPhi && Kind != RK_IntegerMinMax && Kind != RK_FloatMinMax &&
        hasMultipleUsesOf(Cur, VisitedInsts))
      return false;

    // All inputs to a PHI node must be a reduction value.
    if (IsAPhi && Cur != Phi && !areAllUsesIn(Cur, VisitedInsts))
      return false;

    if (Kind == RK_IntegerMinMax &&
        (isa<ICmpInst>(Cur) || isa<SelectInst>(Cur)))
      ++NumCmpSelectPatternInst;
    if (Kind == RK_FloatMinMax && (isa<FCmpInst>(Cur) || isa<SelectInst>(Cur)))
      ++NumCmpSelectPatternInst;

    // Check  whether we found a reduction operator.
    FoundReduxOp |= !IsAPhi;

    // Process users of current instruction. Push non-PHI nodes after PHI nodes
    // onto the stack. This way we are going to have seen all inputs to PHI
    // nodes once we get to them.
    SmallVector<Instruction *, 8> NonPHIs;
    SmallVector<Instruction *, 8> PHIs;
    for (User *U : Cur->users()) {
      Instruction *UI = cast<Instruction>(U);

      // Check if we found the exit user.
      BasicBlock *Parent = UI->getParent();
      if (!TheLoop->contains(Parent)) {
        // Exit if you find multiple outside users or if the header phi node is
        // being used. In this case the user uses the value of the previous
        // iteration, in which case we would loose "VF-1" iterations of the
        // reduction operation if we vectorize.
        if (ExitInstruction != nullptr || Cur == Phi)
          return false;

        // The instruction used by an outside user must be the last instruction
        // before we feed back to the reduction phi. Otherwise, we loose VF-1
        // operations on the value.
        if (std::find(Phi->op_begin(), Phi->op_end(), Cur) == Phi->op_end())
          return false;

        ExitInstruction = Cur;
        continue;
      }

      // Process instructions only once (termination). Each reduction cycle
      // value must only be used once, except by phi nodes and min/max
      // reductions which are represented as a cmp followed by a select.
      InstDesc IgnoredVal(false, nullptr);
      if (VisitedInsts.insert(UI).second) {
        if (isa<PHINode>(UI))
          PHIs.push_back(UI);
        else
          NonPHIs.push_back(UI);
      } else if (!isa<PHINode>(UI) &&
                 ((!isa<FCmpInst>(UI) && !isa<ICmpInst>(UI) &&
                   !isa<SelectInst>(UI)) ||
                  !isMinMaxSelectCmpPattern(UI, IgnoredVal).isRecurrence()))
        return false;

      // Remember that we completed the cycle.
      if (UI == Phi)
        FoundStartPHI = true;
    }
    Worklist.append(PHIs.begin(), PHIs.end());
    Worklist.append(NonPHIs.begin(), NonPHIs.end());
  }

  // This means we have seen one but not the other instruction of the
  // pattern or more than just a select and cmp.
  if ((Kind == RK_IntegerMinMax || Kind == RK_FloatMinMax) &&
      NumCmpSelectPatternInst != 2)
    return false;

  if (!FoundStartPHI || !FoundReduxOp || !ExitInstruction)
    return false;

  // We found a reduction var if we have reached the original phi node and we
  // only have a single instruction with out-of-loop users.

  // The ExitInstruction(Instruction which is allowed to have out-of-loop users)
  // is saved as part of the RecurrenceDescriptor.

  // Save the description of this reduction variable.
  RecurrenceDescriptor RD(RdxStart, ExitInstruction, Kind,
                          ReduxDesc.getMinMaxKind());

  RedDes = RD;

  return true;
}

/// Returns true if the instruction is a Select(ICmp(X, Y), X, Y) instruction
/// pattern corresponding to a min(X, Y) or max(X, Y).
RecurrenceDescriptor::InstDesc
RecurrenceDescriptor::isMinMaxSelectCmpPattern(Instruction *I, InstDesc &Prev) {

  assert((isa<ICmpInst>(I) || isa<FCmpInst>(I) || isa<SelectInst>(I)) &&
         "Expect a select instruction");
  Instruction *Cmp = nullptr;
  SelectInst *Select = nullptr;

  // We must handle the select(cmp()) as a single instruction. Advance to the
  // select.
  if ((Cmp = dyn_cast<ICmpInst>(I)) || (Cmp = dyn_cast<FCmpInst>(I))) {
    if (!Cmp->hasOneUse() || !(Select = dyn_cast<SelectInst>(*I->user_begin())))
      return InstDesc(false, I);
    return InstDesc(Select, Prev.getMinMaxKind());
  }

  // Only handle single use cases for now.
  if (!(Select = dyn_cast<SelectInst>(I)))
    return InstDesc(false, I);
  if (!(Cmp = dyn_cast<ICmpInst>(I->getOperand(0))) &&
      !(Cmp = dyn_cast<FCmpInst>(I->getOperand(0))))
    return InstDesc(false, I);
  if (!Cmp->hasOneUse())
    return InstDesc(false, I);

  Value *CmpLeft;
  Value *CmpRight;

  // Look for a min/max pattern.
  if (m_UMin(m_Value(CmpLeft), m_Value(CmpRight)).match(Select))
    return InstDesc(Select, MRK_UIntMin);
  else if (m_UMax(m_Value(CmpLeft), m_Value(CmpRight)).match(Select))
    return InstDesc(Select, MRK_UIntMax);
  else if (m_SMax(m_Value(CmpLeft), m_Value(CmpRight)).match(Select))
    return InstDesc(Select, MRK_SIntMax);
  else if (m_SMin(m_Value(CmpLeft), m_Value(CmpRight)).match(Select))
    return InstDesc(Select, MRK_SIntMin);
  else if (m_OrdFMin(m_Value(CmpLeft), m_Value(CmpRight)).match(Select))
    return InstDesc(Select, MRK_FloatMin);
  else if (m_OrdFMax(m_Value(CmpLeft), m_Value(CmpRight)).match(Select))
    return InstDesc(Select, MRK_FloatMax);
  else if (m_UnordFMin(m_Value(CmpLeft), m_Value(CmpRight)).match(Select))
    return InstDesc(Select, MRK_FloatMin);
  else if (m_UnordFMax(m_Value(CmpLeft), m_Value(CmpRight)).match(Select))
    return InstDesc(Select, MRK_FloatMax);

  return InstDesc(false, I);
}

RecurrenceDescriptor::InstDesc
RecurrenceDescriptor::isRecurrenceInstr(Instruction *I, RecurrenceKind Kind,
                                        InstDesc &Prev, bool HasFunNoNaNAttr) {
  bool FP = I->getType()->isFloatingPointTy();
  bool FastMath = FP && I->hasUnsafeAlgebra();
  switch (I->getOpcode()) {
  default:
    return InstDesc(false, I);
  case Instruction::PHI:
    if (FP &&
        (Kind != RK_FloatMult && Kind != RK_FloatAdd && Kind != RK_FloatMinMax))
      return InstDesc(false, I);
    return InstDesc(I, Prev.getMinMaxKind());
  case Instruction::Sub:
  case Instruction::Add:
    return InstDesc(Kind == RK_IntegerAdd, I);
  case Instruction::Mul:
    return InstDesc(Kind == RK_IntegerMult, I);
  case Instruction::And:
    return InstDesc(Kind == RK_IntegerAnd, I);
  case Instruction::Or:
    return InstDesc(Kind == RK_IntegerOr, I);
  case Instruction::Xor:
    return InstDesc(Kind == RK_IntegerXor, I);
  case Instruction::FMul:
    return InstDesc(Kind == RK_FloatMult && FastMath, I);
  case Instruction::FSub:
  case Instruction::FAdd:
    return InstDesc(Kind == RK_FloatAdd && FastMath, I);
  case Instruction::FCmp:
  case Instruction::ICmp:
  case Instruction::Select:
    if (Kind != RK_IntegerMinMax &&
        (!HasFunNoNaNAttr || Kind != RK_FloatMinMax))
      return InstDesc(false, I);
    return isMinMaxSelectCmpPattern(I, Prev);
  }
}

bool RecurrenceDescriptor::hasMultipleUsesOf(
    Instruction *I, SmallPtrSetImpl<Instruction *> &Insts) {
  unsigned NumUses = 0;
  for (User::op_iterator Use = I->op_begin(), E = I->op_end(); Use != E;
       ++Use) {
    if (Insts.count(dyn_cast<Instruction>(*Use)))
      ++NumUses;
    if (NumUses > 1)
      return true;
  }

  return false;
}
bool RecurrenceDescriptor::isReductionPHI(PHINode *Phi, Loop *TheLoop,
                                          RecurrenceDescriptor &RedDes) {

  bool HasFunNoNaNAttr = false;
  BasicBlock *Header = TheLoop->getHeader();
  Function &F = *Header->getParent();
  if (F.hasFnAttribute("no-nans-fp-math"))
    HasFunNoNaNAttr =
        F.getFnAttribute("no-nans-fp-math").getValueAsString() == "true";

  if (AddReductionVar(Phi, RK_IntegerAdd, TheLoop, HasFunNoNaNAttr, RedDes)) {
    DEBUG(dbgs() << "Found an ADD reduction PHI." << *Phi << "\n");
    return true;
  }
  if (AddReductionVar(Phi, RK_IntegerMult, TheLoop, HasFunNoNaNAttr, RedDes)) {
    DEBUG(dbgs() << "Found a MUL reduction PHI." << *Phi << "\n");
    return true;
  }
  if (AddReductionVar(Phi, RK_IntegerOr, TheLoop, HasFunNoNaNAttr, RedDes)) {
    DEBUG(dbgs() << "Found an OR reduction PHI." << *Phi << "\n");
    return true;
  }
  if (AddReductionVar(Phi, RK_IntegerAnd, TheLoop, HasFunNoNaNAttr, RedDes)) {
    DEBUG(dbgs() << "Found an AND reduction PHI." << *Phi << "\n");
    return true;
  }
  if (AddReductionVar(Phi, RK_IntegerXor, TheLoop, HasFunNoNaNAttr, RedDes)) {
    DEBUG(dbgs() << "Found a XOR reduction PHI." << *Phi << "\n");
    return true;
  }
  if (AddReductionVar(Phi, RK_IntegerMinMax, TheLoop, HasFunNoNaNAttr,
                      RedDes)) {
    DEBUG(dbgs() << "Found a MINMAX reduction PHI." << *Phi << "\n");
    return true;
  }
  if (AddReductionVar(Phi, RK_FloatMult, TheLoop, HasFunNoNaNAttr, RedDes)) {
    DEBUG(dbgs() << "Found an FMult reduction PHI." << *Phi << "\n");
    return true;
  }
  if (AddReductionVar(Phi, RK_FloatAdd, TheLoop, HasFunNoNaNAttr, RedDes)) {
    DEBUG(dbgs() << "Found an FAdd reduction PHI." << *Phi << "\n");
    return true;
  }
  if (AddReductionVar(Phi, RK_FloatMinMax, TheLoop, HasFunNoNaNAttr, RedDes)) {
    DEBUG(dbgs() << "Found an float MINMAX reduction PHI." << *Phi << "\n");
    return true;
  }
  // Not a reduction of known type.
  return false;
}

/// This function returns the identity element (or neutral element) for
/// the operation K.
Constant *RecurrenceDescriptor::getRecurrenceIdentity(RecurrenceKind K,
                                                      Type *Tp) {
  switch (K) {
  case RK_IntegerXor:
  case RK_IntegerAdd:
  case RK_IntegerOr:
    // Adding, Xoring, Oring zero to a number does not change it.
    return ConstantInt::get(Tp, 0);
  case RK_IntegerMult:
    // Multiplying a number by 1 does not change it.
    return ConstantInt::get(Tp, 1);
  case RK_IntegerAnd:
    // AND-ing a number with an all-1 value does not change it.
    return ConstantInt::get(Tp, -1, true);
  case RK_FloatMult:
    // Multiplying a number by 1 does not change it.
    return ConstantFP::get(Tp, 1.0L);
  case RK_FloatAdd:
    // Adding zero to a number does not change it.
    return ConstantFP::get(Tp, 0.0L);
  default:
    llvm_unreachable("Unknown recurrence kind");
  }
}

/// This function translates the recurrence kind to an LLVM binary operator.
unsigned RecurrenceDescriptor::getRecurrenceBinOp(RecurrenceKind Kind) {
  switch (Kind) {
  case RK_IntegerAdd:
    return Instruction::Add;
  case RK_IntegerMult:
    return Instruction::Mul;
  case RK_IntegerOr:
    return Instruction::Or;
  case RK_IntegerAnd:
    return Instruction::And;
  case RK_IntegerXor:
    return Instruction::Xor;
  case RK_FloatMult:
    return Instruction::FMul;
  case RK_FloatAdd:
    return Instruction::FAdd;
  case RK_IntegerMinMax:
    return Instruction::ICmp;
  case RK_FloatMinMax:
    return Instruction::FCmp;
  default:
    llvm_unreachable("Unknown recurrence operation");
  }
}

Value *RecurrenceDescriptor::createMinMaxOp(IRBuilder<> &Builder,
                                            MinMaxRecurrenceKind RK,
                                            Value *Left, Value *Right) {
  CmpInst::Predicate P = CmpInst::ICMP_NE;
  switch (RK) {
  default:
    llvm_unreachable("Unknown min/max recurrence kind");
  case MRK_UIntMin:
    P = CmpInst::ICMP_ULT;
    break;
  case MRK_UIntMax:
    P = CmpInst::ICMP_UGT;
    break;
  case MRK_SIntMin:
    P = CmpInst::ICMP_SLT;
    break;
  case MRK_SIntMax:
    P = CmpInst::ICMP_SGT;
    break;
  case MRK_FloatMin:
    P = CmpInst::FCMP_OLT;
    break;
  case MRK_FloatMax:
    P = CmpInst::FCMP_OGT;
    break;
  }

  Value *Cmp;
  if (RK == MRK_FloatMin || RK == MRK_FloatMax)
    Cmp = Builder.CreateFCmp(P, Left, Right, "rdx.minmax.cmp");
  else
    Cmp = Builder.CreateICmp(P, Left, Right, "rdx.minmax.cmp");

  Value *Select = Builder.CreateSelect(Cmp, Left, Right, "rdx.minmax.select");
  return Select;
}

bool llvm::isInductionPHI(PHINode *Phi, ScalarEvolution *SE,
                          ConstantInt *&StepValue) {
  Type *PhiTy = Phi->getType();
  // We only handle integer and pointer inductions variables.
  if (!PhiTy->isIntegerTy() && !PhiTy->isPointerTy())
    return false;

  // Check that the PHI is consecutive.
  const SCEV *PhiScev = SE->getSCEV(Phi);
  const SCEVAddRecExpr *AR = dyn_cast<SCEVAddRecExpr>(PhiScev);
  if (!AR) {
    DEBUG(dbgs() << "LV: PHI is not a poly recurrence.\n");
    return false;
  }

  const SCEV *Step = AR->getStepRecurrence(*SE);
  // Calculate the pointer stride and check if it is consecutive.
  const SCEVConstant *C = dyn_cast<SCEVConstant>(Step);
  if (!C)
    return false;

  ConstantInt *CV = C->getValue();
  if (PhiTy->isIntegerTy()) {
    StepValue = CV;
    return true;
  }

  assert(PhiTy->isPointerTy() && "The PHI must be a pointer");
  Type *PointerElementType = PhiTy->getPointerElementType();
  // The pointer stride cannot be determined if the pointer element type is not
  // sized.
  if (!PointerElementType->isSized())
    return false;

  const DataLayout &DL = Phi->getModule()->getDataLayout();
  int64_t Size = static_cast<int64_t>(DL.getTypeAllocSize(PointerElementType));
  if (!Size)
    return false;

  int64_t CVSize = CV->getSExtValue();
  if (CVSize % Size)
    return false;
  StepValue = ConstantInt::getSigned(CV->getType(), CVSize / Size);
  return true;
}

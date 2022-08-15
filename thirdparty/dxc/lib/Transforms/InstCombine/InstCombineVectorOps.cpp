//===- InstCombineVectorOps.cpp -------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements instcombine for ExtractElement, InsertElement and
// ShuffleVector.
//
//===----------------------------------------------------------------------===//

#include "InstCombineInternal.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/IR/PatternMatch.h"
using namespace llvm;
using namespace PatternMatch;

#define DEBUG_TYPE "instcombine"

/// CheapToScalarize - Return true if the value is cheaper to scalarize than it
/// is to leave as a vector operation.  isConstant indicates whether we're
/// extracting one known element.  If false we're extracting a variable index.
static bool CheapToScalarize(Value *V, bool isConstant) {
  if (Constant *C = dyn_cast<Constant>(V)) {
    if (isConstant) return true;

    // If all elts are the same, we can extract it and use any of the values.
    if (Constant *Op0 = C->getAggregateElement(0U)) {
      for (unsigned i = 1, e = V->getType()->getVectorNumElements(); i != e;
           ++i)
        if (C->getAggregateElement(i) != Op0)
          return false;
      return true;
    }
  }
  Instruction *I = dyn_cast<Instruction>(V);
  if (!I) return false;

  // Insert element gets simplified to the inserted element or is deleted if
  // this is constant idx extract element and its a constant idx insertelt.
  if (I->getOpcode() == Instruction::InsertElement && isConstant &&
      isa<ConstantInt>(I->getOperand(2)))
    return true;
  if (I->getOpcode() == Instruction::Load && I->hasOneUse())
    return true;
  if (BinaryOperator *BO = dyn_cast<BinaryOperator>(I))
    if (BO->hasOneUse() &&
        (CheapToScalarize(BO->getOperand(0), isConstant) ||
         CheapToScalarize(BO->getOperand(1), isConstant)))
      return true;
  if (CmpInst *CI = dyn_cast<CmpInst>(I))
    if (CI->hasOneUse() &&
        (CheapToScalarize(CI->getOperand(0), isConstant) ||
         CheapToScalarize(CI->getOperand(1), isConstant)))
      return true;

  return false;
}

// If we have a PHI node with a vector type that has only 2 uses: feed
// itself and be an operand of extractelement at a constant location,
// try to replace the PHI of the vector type with a PHI of a scalar type.
Instruction *InstCombiner::scalarizePHI(ExtractElementInst &EI, PHINode *PN) {
  // Verify that the PHI node has exactly 2 uses. Otherwise return NULL.
  if (!PN->hasNUses(2))
    return nullptr;

  // If so, it's known at this point that one operand is PHI and the other is
  // an extractelement node. Find the PHI user that is not the extractelement
  // node.
  auto iu = PN->user_begin();
  Instruction *PHIUser = dyn_cast<Instruction>(*iu);
  if (PHIUser == cast<Instruction>(&EI))
    PHIUser = cast<Instruction>(*(++iu));

  // Verify that this PHI user has one use, which is the PHI itself,
  // and that it is a binary operation which is cheap to scalarize.
  // otherwise return NULL.
  if (!PHIUser->hasOneUse() || !(PHIUser->user_back() == PN) ||
      !(isa<BinaryOperator>(PHIUser)) || !CheapToScalarize(PHIUser, true))
    return nullptr;

  // Create a scalar PHI node that will replace the vector PHI node
  // just before the current PHI node.
  PHINode *scalarPHI = cast<PHINode>(InsertNewInstWith(
      PHINode::Create(EI.getType(), PN->getNumIncomingValues(), ""), *PN));
  // Scalarize each PHI operand.
  for (unsigned i = 0; i < PN->getNumIncomingValues(); i++) {
    Value *PHIInVal = PN->getIncomingValue(i);
    BasicBlock *inBB = PN->getIncomingBlock(i);
    Value *Elt = EI.getIndexOperand();
    // If the operand is the PHI induction variable:
    if (PHIInVal == PHIUser) {
      // Scalarize the binary operation. Its first operand is the
      // scalar PHI, and the second operand is extracted from the other
      // vector operand.
      BinaryOperator *B0 = cast<BinaryOperator>(PHIUser);
      unsigned opId = (B0->getOperand(0) == PN) ? 1 : 0;
      Value *Op = InsertNewInstWith(
          ExtractElementInst::Create(B0->getOperand(opId), Elt,
                                     B0->getOperand(opId)->getName() + ".Elt"),
          *B0);
      Value *newPHIUser = InsertNewInstWith(
          BinaryOperator::Create(B0->getOpcode(), scalarPHI, Op), *B0);
      scalarPHI->addIncoming(newPHIUser, inBB);
    } else {
      // Scalarize PHI input:
      Instruction *newEI = ExtractElementInst::Create(PHIInVal, Elt, "");
      // Insert the new instruction into the predecessor basic block.
      Instruction *pos = dyn_cast<Instruction>(PHIInVal);
      BasicBlock::iterator InsertPos;
      if (pos && !isa<PHINode>(pos)) {
        InsertPos = pos;
        ++InsertPos;
      } else {
        InsertPos = inBB->getFirstInsertionPt();
      }

      InsertNewInstWith(newEI, *InsertPos);

      scalarPHI->addIncoming(newEI, inBB);
    }
  }
  return ReplaceInstUsesWith(EI, scalarPHI);
}

Instruction *InstCombiner::visitExtractElementInst(ExtractElementInst &EI) {
  if (Value *V = SimplifyExtractElementInst(
          EI.getVectorOperand(), EI.getIndexOperand(), DL, TLI, DT, AC))
    return ReplaceInstUsesWith(EI, V);

  // If vector val is constant with all elements the same, replace EI with
  // that element.  We handle a known element # below.
  if (Constant *C = dyn_cast<Constant>(EI.getOperand(0)))
    if (CheapToScalarize(C, false))
      return ReplaceInstUsesWith(EI, C->getAggregateElement(0U));

  // If extracting a specified index from the vector, see if we can recursively
  // find a previously computed scalar that was inserted into the vector.
  if (ConstantInt *IdxC = dyn_cast<ConstantInt>(EI.getOperand(1))) {
    unsigned IndexVal = IdxC->getZExtValue();
    unsigned VectorWidth = EI.getVectorOperandType()->getNumElements();

    // InstSimplify handles cases where the index is invalid.
    assert(IndexVal < VectorWidth);

    // This instruction only demands the single element from the input vector.
    // If the input vector has a single use, simplify it based on this use
    // property.
    if (EI.getOperand(0)->hasOneUse() && VectorWidth != 1) {
      APInt UndefElts(VectorWidth, 0);
      APInt DemandedMask(VectorWidth, 0);
      DemandedMask.setBit(IndexVal);
      if (Value *V = SimplifyDemandedVectorElts(EI.getOperand(0), DemandedMask,
                                                UndefElts)) {
        EI.setOperand(0, V);
        return &EI;
      }
    }

    // If the this extractelement is directly using a bitcast from a vector of
    // the same number of elements, see if we can find the source element from
    // it.  In this case, we will end up needing to bitcast the scalars.
    if (BitCastInst *BCI = dyn_cast<BitCastInst>(EI.getOperand(0))) {
      if (VectorType *VT = dyn_cast<VectorType>(BCI->getOperand(0)->getType()))
        if (VT->getNumElements() == VectorWidth)
          if (Value *Elt = findScalarElement(BCI->getOperand(0), IndexVal))
            return new BitCastInst(Elt, EI.getType());
    }

    // If there's a vector PHI feeding a scalar use through this extractelement
    // instruction, try to scalarize the PHI.
    if (PHINode *PN = dyn_cast<PHINode>(EI.getOperand(0))) {
      Instruction *scalarPHI = scalarizePHI(EI, PN);
      if (scalarPHI)
        return scalarPHI;
    }
  }

  if (Instruction *I = dyn_cast<Instruction>(EI.getOperand(0))) {
    // Push extractelement into predecessor operation if legal and
    // profitable to do so
    if (BinaryOperator *BO = dyn_cast<BinaryOperator>(I)) {
      if (I->hasOneUse() &&
          CheapToScalarize(BO, isa<ConstantInt>(EI.getOperand(1)))) {
        Value *newEI0 =
          Builder->CreateExtractElement(BO->getOperand(0), EI.getOperand(1),
                                        EI.getName()+".lhs");
        Value *newEI1 =
          Builder->CreateExtractElement(BO->getOperand(1), EI.getOperand(1),
                                        EI.getName()+".rhs");
        return BinaryOperator::Create(BO->getOpcode(), newEI0, newEI1);
      }
    } else if (InsertElementInst *IE = dyn_cast<InsertElementInst>(I)) {
      // Extracting the inserted element?
      if (IE->getOperand(2) == EI.getOperand(1))
        return ReplaceInstUsesWith(EI, IE->getOperand(1));
      // If the inserted and extracted elements are constants, they must not
      // be the same value, extract from the pre-inserted value instead.
      if (isa<Constant>(IE->getOperand(2)) && isa<Constant>(EI.getOperand(1))) {
        Worklist.AddValue(EI.getOperand(0));
        EI.setOperand(0, IE->getOperand(0));
        return &EI;
      }
    } else if (ShuffleVectorInst *SVI = dyn_cast<ShuffleVectorInst>(I)) {
      // If this is extracting an element from a shufflevector, figure out where
      // it came from and extract from the appropriate input element instead.
      if (ConstantInt *Elt = dyn_cast<ConstantInt>(EI.getOperand(1))) {
        int SrcIdx = SVI->getMaskValue(Elt->getZExtValue());
        Value *Src;
        unsigned LHSWidth =
          SVI->getOperand(0)->getType()->getVectorNumElements();

        if (SrcIdx < 0)
          return ReplaceInstUsesWith(EI, UndefValue::get(EI.getType()));
        if (SrcIdx < (int)LHSWidth)
          Src = SVI->getOperand(0);
        else {
          SrcIdx -= LHSWidth;
          Src = SVI->getOperand(1);
        }
        Type *Int32Ty = Type::getInt32Ty(EI.getContext());
        return ExtractElementInst::Create(Src,
                                          ConstantInt::get(Int32Ty,
                                                           SrcIdx, false));
      }
    } else if (CastInst *CI = dyn_cast<CastInst>(I)) {
      // Canonicalize extractelement(cast) -> cast(extractelement)
      // bitcasts can change the number of vector elements and they cost nothing
      if (CI->hasOneUse() && (CI->getOpcode() != Instruction::BitCast)) {
        Value *EE = Builder->CreateExtractElement(CI->getOperand(0),
                                                  EI.getIndexOperand());
        Worklist.AddValue(EE);
        return CastInst::Create(CI->getOpcode(), EE, EI.getType());
      }
    } else if (SelectInst *SI = dyn_cast<SelectInst>(I)) {
      if (SI->hasOneUse()) {
        // TODO: For a select on vectors, it might be useful to do this if it
        // has multiple extractelement uses. For vector select, that seems to
        // fight the vectorizer.

        // If we are extracting an element from a vector select or a select on
        // vectors, a select on the scalars extracted from the vector arguments.
        Value *TrueVal = SI->getTrueValue();
        Value *FalseVal = SI->getFalseValue();

        Value *Cond = SI->getCondition();
        if (Cond->getType()->isVectorTy()) {
          Cond = Builder->CreateExtractElement(Cond,
                                               EI.getIndexOperand(),
                                               Cond->getName() + ".elt");
        }

        Value *V1Elem
          = Builder->CreateExtractElement(TrueVal,
                                          EI.getIndexOperand(),
                                          TrueVal->getName() + ".elt");

        Value *V2Elem
          = Builder->CreateExtractElement(FalseVal,
                                          EI.getIndexOperand(),
                                          FalseVal->getName() + ".elt");
        return SelectInst::Create(Cond,
                                  V1Elem,
                                  V2Elem,
                                  SI->getName() + ".elt");
      }
    }
  }
  return nullptr;
}

/// CollectSingleShuffleElements - If V is a shuffle of values that ONLY returns
/// elements from either LHS or RHS, return the shuffle mask and true.
/// Otherwise, return false.
static bool CollectSingleShuffleElements(Value *V, Value *LHS, Value *RHS,
                                         SmallVectorImpl<Constant*> &Mask) {
  assert(LHS->getType() == RHS->getType() &&
         "Invalid CollectSingleShuffleElements");
  unsigned NumElts = V->getType()->getVectorNumElements();

  if (isa<UndefValue>(V)) {
    Mask.assign(NumElts, UndefValue::get(Type::getInt32Ty(V->getContext())));
    return true;
  }

  if (V == LHS) {
    for (unsigned i = 0; i != NumElts; ++i)
      Mask.push_back(ConstantInt::get(Type::getInt32Ty(V->getContext()), i));
    return true;
  }

  if (V == RHS) {
    for (unsigned i = 0; i != NumElts; ++i)
      Mask.push_back(ConstantInt::get(Type::getInt32Ty(V->getContext()),
                                      i+NumElts));
    return true;
  }

  if (InsertElementInst *IEI = dyn_cast<InsertElementInst>(V)) {
    // If this is an insert of an extract from some other vector, include it.
    Value *VecOp    = IEI->getOperand(0);
    Value *ScalarOp = IEI->getOperand(1);
    Value *IdxOp    = IEI->getOperand(2);

    if (!isa<ConstantInt>(IdxOp))
      return false;
    unsigned InsertedIdx = cast<ConstantInt>(IdxOp)->getZExtValue();

    if (isa<UndefValue>(ScalarOp)) {  // inserting undef into vector.
      // We can handle this if the vector we are inserting into is
      // transitively ok.
      if (CollectSingleShuffleElements(VecOp, LHS, RHS, Mask)) {
        // If so, update the mask to reflect the inserted undef.
        Mask[InsertedIdx] = UndefValue::get(Type::getInt32Ty(V->getContext()));
        return true;
      }
    } else if (ExtractElementInst *EI = dyn_cast<ExtractElementInst>(ScalarOp)){
      if (isa<ConstantInt>(EI->getOperand(1))) {
        unsigned ExtractedIdx =
        cast<ConstantInt>(EI->getOperand(1))->getZExtValue();
        unsigned NumLHSElts = LHS->getType()->getVectorNumElements();

        // This must be extracting from either LHS or RHS.
        if (EI->getOperand(0) == LHS || EI->getOperand(0) == RHS) {
          // We can handle this if the vector we are inserting into is
          // transitively ok.
          if (CollectSingleShuffleElements(VecOp, LHS, RHS, Mask)) {
            // If so, update the mask to reflect the inserted value.
            if (EI->getOperand(0) == LHS) {
              Mask[InsertedIdx % NumElts] =
              ConstantInt::get(Type::getInt32Ty(V->getContext()),
                               ExtractedIdx);
            } else {
              assert(EI->getOperand(0) == RHS);
              Mask[InsertedIdx % NumElts] =
              ConstantInt::get(Type::getInt32Ty(V->getContext()),
                               ExtractedIdx + NumLHSElts);
            }
            return true;
          }
        }
      }
    }
  }

  return false;
}


/// We are building a shuffle to create V, which is a sequence of insertelement,
/// extractelement pairs. If PermittedRHS is set, then we must either use it or
/// not rely on the second vector source. Return a std::pair containing the
/// left and right vectors of the proposed shuffle (or 0), and set the Mask
/// parameter as required.
///
/// Note: we intentionally don't try to fold earlier shuffles since they have
/// often been chosen carefully to be efficiently implementable on the target.
typedef std::pair<Value *, Value *> ShuffleOps;

static ShuffleOps CollectShuffleElements(Value *V,
                                         SmallVectorImpl<Constant *> &Mask,
                                         Value *PermittedRHS) {
  assert(V->getType()->isVectorTy() && "Invalid shuffle!");
  unsigned NumElts = cast<VectorType>(V->getType())->getNumElements();

  if (isa<UndefValue>(V)) {
    Mask.assign(NumElts, UndefValue::get(Type::getInt32Ty(V->getContext())));
    return std::make_pair(
        PermittedRHS ? UndefValue::get(PermittedRHS->getType()) : V, nullptr);
  }

  if (isa<ConstantAggregateZero>(V)) {
    Mask.assign(NumElts, ConstantInt::get(Type::getInt32Ty(V->getContext()),0));
    return std::make_pair(V, nullptr);
  }

  if (InsertElementInst *IEI = dyn_cast<InsertElementInst>(V)) {
    // If this is an insert of an extract from some other vector, include it.
    Value *VecOp    = IEI->getOperand(0);
    Value *ScalarOp = IEI->getOperand(1);
    Value *IdxOp    = IEI->getOperand(2);

    if (ExtractElementInst *EI = dyn_cast<ExtractElementInst>(ScalarOp)) {
      if (isa<ConstantInt>(EI->getOperand(1)) && isa<ConstantInt>(IdxOp)) {
        unsigned ExtractedIdx =
          cast<ConstantInt>(EI->getOperand(1))->getZExtValue();
        unsigned InsertedIdx = cast<ConstantInt>(IdxOp)->getZExtValue();

        // Either the extracted from or inserted into vector must be RHSVec,
        // otherwise we'd end up with a shuffle of three inputs.
        if (EI->getOperand(0) == PermittedRHS || PermittedRHS == nullptr) {
          Value *RHS = EI->getOperand(0);
          ShuffleOps LR = CollectShuffleElements(VecOp, Mask, RHS);
          assert(LR.second == nullptr || LR.second == RHS);

          if (LR.first->getType() != RHS->getType()) {
            // We tried our best, but we can't find anything compatible with RHS
            // further up the chain. Return a trivial shuffle.
            for (unsigned i = 0; i < NumElts; ++i)
              Mask[i] = ConstantInt::get(Type::getInt32Ty(V->getContext()), i);
            return std::make_pair(V, nullptr);
          }

          unsigned NumLHSElts = RHS->getType()->getVectorNumElements();
          Mask[InsertedIdx % NumElts] =
            ConstantInt::get(Type::getInt32Ty(V->getContext()),
                             NumLHSElts+ExtractedIdx);
          return std::make_pair(LR.first, RHS);
        }

        if (VecOp == PermittedRHS) {
          // We've gone as far as we can: anything on the other side of the
          // extractelement will already have been converted into a shuffle.
          unsigned NumLHSElts =
              EI->getOperand(0)->getType()->getVectorNumElements();
          for (unsigned i = 0; i != NumElts; ++i)
            Mask.push_back(ConstantInt::get(
                Type::getInt32Ty(V->getContext()),
                i == InsertedIdx ? ExtractedIdx : NumLHSElts + i));
          return std::make_pair(EI->getOperand(0), PermittedRHS);
        }

        // If this insertelement is a chain that comes from exactly these two
        // vectors, return the vector and the effective shuffle.
        if (EI->getOperand(0)->getType() == PermittedRHS->getType() &&
            CollectSingleShuffleElements(IEI, EI->getOperand(0), PermittedRHS,
                                         Mask))
          return std::make_pair(EI->getOperand(0), PermittedRHS);
      }
    }
  }

  // Otherwise, can't do anything fancy.  Return an identity vector.
  for (unsigned i = 0; i != NumElts; ++i)
    Mask.push_back(ConstantInt::get(Type::getInt32Ty(V->getContext()), i));
  return std::make_pair(V, nullptr);
}

/// Try to find redundant insertvalue instructions, like the following ones:
///  %0 = insertvalue { i8, i32 } undef, i8 %x, 0
///  %1 = insertvalue { i8, i32 } %0,    i8 %y, 0
/// Here the second instruction inserts values at the same indices, as the
/// first one, making the first one redundant.
/// It should be transformed to:
///  %0 = insertvalue { i8, i32 } undef, i8 %y, 0
Instruction *InstCombiner::visitInsertValueInst(InsertValueInst &I) {
  bool IsRedundant = false;
  ArrayRef<unsigned int> FirstIndices = I.getIndices();

  // If there is a chain of insertvalue instructions (each of them except the
  // last one has only one use and it's another insertvalue insn from this
  // chain), check if any of the 'children' uses the same indices as the first
  // instruction. In this case, the first one is redundant.
  Value *V = &I;
  unsigned Depth = 0;
  while (V->hasOneUse() && Depth < 10) {
    User *U = V->user_back();
    auto UserInsInst = dyn_cast<InsertValueInst>(U);
    if (!UserInsInst || U->getOperand(0) != V)
      break;
    if (UserInsInst->getIndices() == FirstIndices) {
      IsRedundant = true;
      break;
    }
    V = UserInsInst;
    Depth++;
  }

  if (IsRedundant)
    return ReplaceInstUsesWith(I, I.getOperand(0));
  return nullptr;
}

Instruction *InstCombiner::visitInsertElementInst(InsertElementInst &IE) {
  Value *VecOp    = IE.getOperand(0);
  Value *ScalarOp = IE.getOperand(1);
  Value *IdxOp    = IE.getOperand(2);

  // Inserting an undef or into an undefined place, remove this.
  if (isa<UndefValue>(ScalarOp) || isa<UndefValue>(IdxOp))
    ReplaceInstUsesWith(IE, VecOp);

  // If the inserted element was extracted from some other vector, and if the
  // indexes are constant, try to turn this into a shufflevector operation.
  if (ExtractElementInst *EI = dyn_cast<ExtractElementInst>(ScalarOp)) {
    if (isa<ConstantInt>(EI->getOperand(1)) && isa<ConstantInt>(IdxOp)) {
      unsigned NumInsertVectorElts = IE.getType()->getNumElements();
      unsigned NumExtractVectorElts =
          EI->getOperand(0)->getType()->getVectorNumElements();
      unsigned ExtractedIdx =
        cast<ConstantInt>(EI->getOperand(1))->getZExtValue();
      unsigned InsertedIdx = cast<ConstantInt>(IdxOp)->getZExtValue();

      if (ExtractedIdx >= NumExtractVectorElts) // Out of range extract.
        return ReplaceInstUsesWith(IE, VecOp);

      if (InsertedIdx >= NumInsertVectorElts)  // Out of range insert.
        return ReplaceInstUsesWith(IE, UndefValue::get(IE.getType()));

      // If we are extracting a value from a vector, then inserting it right
      // back into the same place, just use the input vector.
      if (EI->getOperand(0) == VecOp && ExtractedIdx == InsertedIdx)
        return ReplaceInstUsesWith(IE, VecOp);

      // If this insertelement isn't used by some other insertelement, turn it
      // (and any insertelements it points to), into one big shuffle.
      if (!IE.hasOneUse() || !isa<InsertElementInst>(IE.user_back())) {
        SmallVector<Constant*, 16> Mask;
        ShuffleOps LR = CollectShuffleElements(&IE, Mask, nullptr);

        // The proposed shuffle may be trivial, in which case we shouldn't
        // perform the combine.
        if (LR.first != &IE && LR.second != &IE) {
          // We now have a shuffle of LHS, RHS, Mask.
          if (LR.second == nullptr)
            LR.second = UndefValue::get(LR.first->getType());
          return new ShuffleVectorInst(LR.first, LR.second,
                                       ConstantVector::get(Mask));
        }
      }
    }
  }

  unsigned VWidth = cast<VectorType>(VecOp->getType())->getNumElements();
  APInt UndefElts(VWidth, 0);
  APInt AllOnesEltMask(APInt::getAllOnesValue(VWidth));
  if (Value *V = SimplifyDemandedVectorElts(&IE, AllOnesEltMask, UndefElts)) {
    if (V != &IE)
      return ReplaceInstUsesWith(IE, V);
    return &IE;
  }

  return nullptr;
}

/// Return true if we can evaluate the specified expression tree if the vector
/// elements were shuffled in a different order.
static bool CanEvaluateShuffled(Value *V, ArrayRef<int> Mask,
                                unsigned Depth = 5) {
  // We can always reorder the elements of a constant.
  if (isa<Constant>(V))
    return true;

  // We won't reorder vector arguments. No IPO here.
  Instruction *I = dyn_cast<Instruction>(V);
  if (!I) return false;

  // Two users may expect different orders of the elements. Don't try it.
  if (!I->hasOneUse())
    return false;

  if (Depth == 0) return false;

  switch (I->getOpcode()) {
    case Instruction::Add:
    case Instruction::FAdd:
    case Instruction::Sub:
    case Instruction::FSub:
    case Instruction::Mul:
    case Instruction::FMul:
    case Instruction::UDiv:
    case Instruction::SDiv:
    case Instruction::FDiv:
    case Instruction::URem:
    case Instruction::SRem:
    case Instruction::FRem:
    case Instruction::Shl:
    case Instruction::LShr:
    case Instruction::AShr:
    case Instruction::And:
    case Instruction::Or:
    case Instruction::Xor:
    case Instruction::ICmp:
    case Instruction::FCmp:
    case Instruction::Trunc:
    case Instruction::ZExt:
    case Instruction::SExt:
    case Instruction::FPToUI:
    case Instruction::FPToSI:
    case Instruction::UIToFP:
    case Instruction::SIToFP:
    case Instruction::FPTrunc:
    case Instruction::FPExt:
    case Instruction::GetElementPtr: {
      for (int i = 0, e = I->getNumOperands(); i != e; ++i) {
        if (!CanEvaluateShuffled(I->getOperand(i), Mask, Depth-1))
          return false;
      }
      return true;
    }
    case Instruction::InsertElement: {
      ConstantInt *CI = dyn_cast<ConstantInt>(I->getOperand(2));
      if (!CI) return false;
      int ElementNumber = CI->getLimitedValue();

      // Verify that 'CI' does not occur twice in Mask. A single 'insertelement'
      // can't put an element into multiple indices.
      bool SeenOnce = false;
      for (int i = 0, e = Mask.size(); i != e; ++i) {
        if (Mask[i] == ElementNumber) {
          if (SeenOnce)
            return false;
          SeenOnce = true;
        }
      }
      return CanEvaluateShuffled(I->getOperand(0), Mask, Depth-1);
    }
  }
  return false;
}

/// Rebuild a new instruction just like 'I' but with the new operands given.
/// In the event of type mismatch, the type of the operands is correct.
static Value *BuildNew(Instruction *I, ArrayRef<Value*> NewOps) {
  // We don't want to use the IRBuilder here because we want the replacement
  // instructions to appear next to 'I', not the builder's insertion point.
  switch (I->getOpcode()) {
    case Instruction::Add:
    case Instruction::FAdd:
    case Instruction::Sub:
    case Instruction::FSub:
    case Instruction::Mul:
    case Instruction::FMul:
    case Instruction::UDiv:
    case Instruction::SDiv:
    case Instruction::FDiv:
    case Instruction::URem:
    case Instruction::SRem:
    case Instruction::FRem:
    case Instruction::Shl:
    case Instruction::LShr:
    case Instruction::AShr:
    case Instruction::And:
    case Instruction::Or:
    case Instruction::Xor: {
      BinaryOperator *BO = cast<BinaryOperator>(I);
      assert(NewOps.size() == 2 && "binary operator with #ops != 2");
      BinaryOperator *New =
          BinaryOperator::Create(cast<BinaryOperator>(I)->getOpcode(),
                                 NewOps[0], NewOps[1], "", BO);
      if (isa<OverflowingBinaryOperator>(BO)) {
        New->setHasNoUnsignedWrap(BO->hasNoUnsignedWrap());
        New->setHasNoSignedWrap(BO->hasNoSignedWrap());
      }
      if (isa<PossiblyExactOperator>(BO)) {
        New->setIsExact(BO->isExact());
      }
      if (isa<FPMathOperator>(BO))
        New->copyFastMathFlags(I);
      return New;
    }
    case Instruction::ICmp:
      assert(NewOps.size() == 2 && "icmp with #ops != 2");
      return new ICmpInst(I, cast<ICmpInst>(I)->getPredicate(),
                          NewOps[0], NewOps[1]);
    case Instruction::FCmp:
      assert(NewOps.size() == 2 && "fcmp with #ops != 2");
      return new FCmpInst(I, cast<FCmpInst>(I)->getPredicate(),
                          NewOps[0], NewOps[1]);
    case Instruction::Trunc:
    case Instruction::ZExt:
    case Instruction::SExt:
    case Instruction::FPToUI:
    case Instruction::FPToSI:
    case Instruction::UIToFP:
    case Instruction::SIToFP:
    case Instruction::FPTrunc:
    case Instruction::FPExt: {
      // It's possible that the mask has a different number of elements from
      // the original cast. We recompute the destination type to match the mask.
      Type *DestTy =
          VectorType::get(I->getType()->getScalarType(),
                          NewOps[0]->getType()->getVectorNumElements());
      assert(NewOps.size() == 1 && "cast with #ops != 1");
      return CastInst::Create(cast<CastInst>(I)->getOpcode(), NewOps[0], DestTy,
                              "", I);
    }
    case Instruction::GetElementPtr: {
      Value *Ptr = NewOps[0];
      ArrayRef<Value*> Idx = NewOps.slice(1);
      GetElementPtrInst *GEP = GetElementPtrInst::Create(
          cast<GetElementPtrInst>(I)->getSourceElementType(), Ptr, Idx, "", I);
      GEP->setIsInBounds(cast<GetElementPtrInst>(I)->isInBounds());
      return GEP;
    }
  }
  llvm_unreachable("failed to rebuild vector instructions");
}

Value *
InstCombiner::EvaluateInDifferentElementOrder(Value *V, ArrayRef<int> Mask) {
  // Mask.size() does not need to be equal to the number of vector elements.

  assert(V->getType()->isVectorTy() && "can't reorder non-vector elements");
  if (isa<UndefValue>(V)) {
    return UndefValue::get(VectorType::get(V->getType()->getScalarType(),
                                           Mask.size()));
  }
  if (isa<ConstantAggregateZero>(V)) {
    return ConstantAggregateZero::get(
               VectorType::get(V->getType()->getScalarType(),
                               Mask.size()));
  }
  if (Constant *C = dyn_cast<Constant>(V)) {
    SmallVector<Constant *, 16> MaskValues;
    for (int i = 0, e = Mask.size(); i != e; ++i) {
      if (Mask[i] == -1)
        MaskValues.push_back(UndefValue::get(Builder->getInt32Ty()));
      else
        MaskValues.push_back(Builder->getInt32(Mask[i]));
    }
    return ConstantExpr::getShuffleVector(C, UndefValue::get(C->getType()),
                                          ConstantVector::get(MaskValues));
  }

  Instruction *I = cast<Instruction>(V);
  switch (I->getOpcode()) {
    case Instruction::Add:
    case Instruction::FAdd:
    case Instruction::Sub:
    case Instruction::FSub:
    case Instruction::Mul:
    case Instruction::FMul:
    case Instruction::UDiv:
    case Instruction::SDiv:
    case Instruction::FDiv:
    case Instruction::URem:
    case Instruction::SRem:
    case Instruction::FRem:
    case Instruction::Shl:
    case Instruction::LShr:
    case Instruction::AShr:
    case Instruction::And:
    case Instruction::Or:
    case Instruction::Xor:
    case Instruction::ICmp:
    case Instruction::FCmp:
    case Instruction::Trunc:
    case Instruction::ZExt:
    case Instruction::SExt:
    case Instruction::FPToUI:
    case Instruction::FPToSI:
    case Instruction::UIToFP:
    case Instruction::SIToFP:
    case Instruction::FPTrunc:
    case Instruction::FPExt:
    case Instruction::Select:
    case Instruction::GetElementPtr: {
      SmallVector<Value*, 8> NewOps;
      bool NeedsRebuild = (Mask.size() != I->getType()->getVectorNumElements());
      for (int i = 0, e = I->getNumOperands(); i != e; ++i) {
        Value *V = EvaluateInDifferentElementOrder(I->getOperand(i), Mask);
        NewOps.push_back(V);
        NeedsRebuild |= (V != I->getOperand(i));
      }
      if (NeedsRebuild) {
        return BuildNew(I, NewOps);
      }
      return I;
    }
    case Instruction::InsertElement: {
      int Element = cast<ConstantInt>(I->getOperand(2))->getLimitedValue();

      // The insertelement was inserting at Element. Figure out which element
      // that becomes after shuffling. The answer is guaranteed to be unique
      // by CanEvaluateShuffled.
      bool Found = false;
      int Index = 0;
      for (int e = Mask.size(); Index != e; ++Index) {
        if (Mask[Index] == Element) {
          Found = true;
          break;
        }
      }

      // If element is not in Mask, no need to handle the operand 1 (element to
      // be inserted). Just evaluate values in operand 0 according to Mask.
      if (!Found)
        return EvaluateInDifferentElementOrder(I->getOperand(0), Mask);

      Value *V = EvaluateInDifferentElementOrder(I->getOperand(0), Mask);
      return InsertElementInst::Create(V, I->getOperand(1),
                                       Builder->getInt32(Index), "", I);
    }
  }
  llvm_unreachable("failed to reorder elements of vector instruction!");
}

static void RecognizeIdentityMask(const SmallVectorImpl<int> &Mask,
                                  bool &isLHSID, bool &isRHSID) {
  isLHSID = isRHSID = true;

  for (unsigned i = 0, e = Mask.size(); i != e; ++i) {
    if (Mask[i] < 0) continue;  // Ignore undef values.
    // Is this an identity shuffle of the LHS value?
    isLHSID &= (Mask[i] == (int)i);

    // Is this an identity shuffle of the RHS value?
    isRHSID &= (Mask[i]-e == i);
  }
}

// Returns true if the shuffle is extracting a contiguous range of values from
// LHS, for example:
//                 +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
//   Input:        |AA|BB|CC|DD|EE|FF|GG|HH|II|JJ|KK|LL|MM|NN|OO|PP|
//   Shuffles to:  |EE|FF|GG|HH|
//                 +--+--+--+--+
static bool isShuffleExtractingFromLHS(ShuffleVectorInst &SVI,
                                       SmallVector<int, 16> &Mask) {
  unsigned LHSElems =
      cast<VectorType>(SVI.getOperand(0)->getType())->getNumElements();
  unsigned MaskElems = Mask.size();
  unsigned BegIdx = Mask.front();
  unsigned EndIdx = Mask.back();
  if (BegIdx > EndIdx || EndIdx >= LHSElems || EndIdx - BegIdx != MaskElems - 1)
    return false;
  for (unsigned I = 0; I != MaskElems; ++I)
    if (static_cast<unsigned>(Mask[I]) != BegIdx + I)
      return false;
  return true;
}

Instruction *InstCombiner::visitShuffleVectorInst(ShuffleVectorInst &SVI) {
  Value *LHS = SVI.getOperand(0);
  Value *RHS = SVI.getOperand(1);
  SmallVector<int, 16> Mask = SVI.getShuffleMask();
  Type *Int32Ty = Type::getInt32Ty(SVI.getContext());

  bool MadeChange = false;

  // Undefined shuffle mask -> undefined value.
  if (isa<UndefValue>(SVI.getOperand(2)))
    return ReplaceInstUsesWith(SVI, UndefValue::get(SVI.getType()));

  unsigned VWidth = cast<VectorType>(SVI.getType())->getNumElements();

  APInt UndefElts(VWidth, 0);
  APInt AllOnesEltMask(APInt::getAllOnesValue(VWidth));
  if (Value *V = SimplifyDemandedVectorElts(&SVI, AllOnesEltMask, UndefElts)) {
    if (V != &SVI)
      return ReplaceInstUsesWith(SVI, V);
    LHS = SVI.getOperand(0);
    RHS = SVI.getOperand(1);
    MadeChange = true;
  }

  unsigned LHSWidth = cast<VectorType>(LHS->getType())->getNumElements();

  // Canonicalize shuffle(x    ,x,mask) -> shuffle(x, undef,mask')
  // Canonicalize shuffle(undef,x,mask) -> shuffle(x, undef,mask').
  if (LHS == RHS || isa<UndefValue>(LHS)) {
    if (isa<UndefValue>(LHS) && LHS == RHS) {
      // shuffle(undef,undef,mask) -> undef.
      Value *Result = (VWidth == LHSWidth)
                      ? LHS : UndefValue::get(SVI.getType());
      return ReplaceInstUsesWith(SVI, Result);
    }

    // Remap any references to RHS to use LHS.
    SmallVector<Constant*, 16> Elts;
    for (unsigned i = 0, e = LHSWidth; i != VWidth; ++i) {
      if (Mask[i] < 0) {
        Elts.push_back(UndefValue::get(Int32Ty));
        continue;
      }

      if ((Mask[i] >= (int)e && isa<UndefValue>(RHS)) ||
          (Mask[i] <  (int)e && isa<UndefValue>(LHS))) {
        Mask[i] = -1;     // Turn into undef.
        Elts.push_back(UndefValue::get(Int32Ty));
      } else {
        Mask[i] = Mask[i] % e;  // Force to LHS.
        Elts.push_back(ConstantInt::get(Int32Ty, Mask[i]));
      }
    }
    SVI.setOperand(0, SVI.getOperand(1));
    SVI.setOperand(1, UndefValue::get(RHS->getType()));
    SVI.setOperand(2, ConstantVector::get(Elts));
    LHS = SVI.getOperand(0);
    RHS = SVI.getOperand(1);
    MadeChange = true;
  }

  if (VWidth == LHSWidth) {
    // Analyze the shuffle, are the LHS or RHS and identity shuffles?
    bool isLHSID, isRHSID;
    RecognizeIdentityMask(Mask, isLHSID, isRHSID);

    // Eliminate identity shuffles.
    if (isLHSID) return ReplaceInstUsesWith(SVI, LHS);
    if (isRHSID) return ReplaceInstUsesWith(SVI, RHS);
  }

  if (isa<UndefValue>(RHS) && CanEvaluateShuffled(LHS, Mask)) {
    Value *V = EvaluateInDifferentElementOrder(LHS, Mask);
    return ReplaceInstUsesWith(SVI, V);
  }

  // SROA generates shuffle+bitcast when the extracted sub-vector is bitcast to
  // a non-vector type. We can instead bitcast the original vector followed by
  // an extract of the desired element:
  //
  //   %sroa = shufflevector <16 x i8> %in, <16 x i8> undef,
  //                         <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  //   %1 = bitcast <4 x i8> %sroa to i32
  // Becomes:
  //   %bc = bitcast <16 x i8> %in to <4 x i32>
  //   %ext = extractelement <4 x i32> %bc, i32 0
  //
  // If the shuffle is extracting a contiguous range of values from the input
  // vector then each use which is a bitcast of the extracted size can be
  // replaced. This will work if the vector types are compatible, and the begin
  // index is aligned to a value in the casted vector type. If the begin index
  // isn't aligned then we can shuffle the original vector (keeping the same
  // vector type) before extracting.
  //
  // This code will bail out if the target type is fundamentally incompatible
  // with vectors of the source type.
  //
  // Example of <16 x i8>, target type i32:
  // Index range [4,8):         v-----------v Will work.
  //                +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
  //     <16 x i8>: |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
  //     <4 x i32>: |           |           |           |           |
  //                +-----------+-----------+-----------+-----------+
  // Index range [6,10):              ^-----------^ Needs an extra shuffle.
  // Target type i40:           ^--------------^ Won't work, bail.
  if (isShuffleExtractingFromLHS(SVI, Mask)) {
    Value *V = LHS;
    unsigned MaskElems = Mask.size();
    unsigned BegIdx = Mask.front();
    VectorType *SrcTy = cast<VectorType>(V->getType());
    unsigned VecBitWidth = SrcTy->getBitWidth();
    unsigned SrcElemBitWidth = DL.getTypeSizeInBits(SrcTy->getElementType());
    assert(SrcElemBitWidth && "vector elements must have a bitwidth");
    unsigned SrcNumElems = SrcTy->getNumElements();
    SmallVector<BitCastInst *, 8> BCs;
    DenseMap<Type *, Value *> NewBCs;
    for (User *U : SVI.users())
      if (BitCastInst *BC = dyn_cast<BitCastInst>(U))
        if (!BC->use_empty())
          // Only visit bitcasts that weren't previously handled.
          BCs.push_back(BC);
    for (BitCastInst *BC : BCs) {
      Type *TgtTy = BC->getDestTy();
      unsigned TgtElemBitWidth = DL.getTypeSizeInBits(TgtTy);
      if (!TgtElemBitWidth)
        continue;
      unsigned TgtNumElems = VecBitWidth / TgtElemBitWidth;
      bool VecBitWidthsEqual = VecBitWidth == TgtNumElems * TgtElemBitWidth;
      bool BegIsAligned = 0 == ((SrcElemBitWidth * BegIdx) % TgtElemBitWidth);
      if (!VecBitWidthsEqual)
        continue;
      if (!VectorType::isValidElementType(TgtTy))
        continue;
      VectorType *CastSrcTy = VectorType::get(TgtTy, TgtNumElems);
      if (!BegIsAligned) {
        // Shuffle the input so [0,NumElements) contains the output, and
        // [NumElems,SrcNumElems) is undef.
        SmallVector<Constant *, 16> ShuffleMask(SrcNumElems,
                                                UndefValue::get(Int32Ty));
        for (unsigned I = 0, E = MaskElems, Idx = BegIdx; I != E; ++Idx, ++I)
          ShuffleMask[I] = ConstantInt::get(Int32Ty, Idx);
        V = Builder->CreateShuffleVector(V, UndefValue::get(V->getType()),
                                         ConstantVector::get(ShuffleMask),
                                         SVI.getName() + ".extract");
        BegIdx = 0;
      }
      unsigned SrcElemsPerTgtElem = TgtElemBitWidth / SrcElemBitWidth;
      assert(SrcElemsPerTgtElem);
      BegIdx /= SrcElemsPerTgtElem;
      bool BCAlreadyExists = NewBCs.find(CastSrcTy) != NewBCs.end();
      auto *NewBC =
          BCAlreadyExists
              ? NewBCs[CastSrcTy]
              : Builder->CreateBitCast(V, CastSrcTy, SVI.getName() + ".bc");
      if (!BCAlreadyExists)
        NewBCs[CastSrcTy] = NewBC;
      auto *Ext = Builder->CreateExtractElement(
          NewBC, ConstantInt::get(Int32Ty, BegIdx), SVI.getName() + ".extract");
      // The shufflevector isn't being replaced: the bitcast that used it
      // is. InstCombine will visit the newly-created instructions.
      ReplaceInstUsesWith(*BC, Ext);
      MadeChange = true;
    }
  }

  // If the LHS is a shufflevector itself, see if we can combine it with this
  // one without producing an unusual shuffle.
  // Cases that might be simplified:
  // 1.
  // x1=shuffle(v1,v2,mask1)
  //  x=shuffle(x1,undef,mask)
  //        ==>
  //  x=shuffle(v1,undef,newMask)
  // newMask[i] = (mask[i] < x1.size()) ? mask1[mask[i]] : -1
  // 2.
  // x1=shuffle(v1,undef,mask1)
  //  x=shuffle(x1,x2,mask)
  // where v1.size() == mask1.size()
  //        ==>
  //  x=shuffle(v1,x2,newMask)
  // newMask[i] = (mask[i] < x1.size()) ? mask1[mask[i]] : mask[i]
  // 3.
  // x2=shuffle(v2,undef,mask2)
  //  x=shuffle(x1,x2,mask)
  // where v2.size() == mask2.size()
  //        ==>
  //  x=shuffle(x1,v2,newMask)
  // newMask[i] = (mask[i] < x1.size())
  //              ? mask[i] : mask2[mask[i]-x1.size()]+x1.size()
  // 4.
  // x1=shuffle(v1,undef,mask1)
  // x2=shuffle(v2,undef,mask2)
  //  x=shuffle(x1,x2,mask)
  // where v1.size() == v2.size()
  //        ==>
  //  x=shuffle(v1,v2,newMask)
  // newMask[i] = (mask[i] < x1.size())
  //              ? mask1[mask[i]] : mask2[mask[i]-x1.size()]+v1.size()
  //
  // Here we are really conservative:
  // we are absolutely afraid of producing a shuffle mask not in the input
  // program, because the code gen may not be smart enough to turn a merged
  // shuffle into two specific shuffles: it may produce worse code.  As such,
  // we only merge two shuffles if the result is either a splat or one of the
  // input shuffle masks.  In this case, merging the shuffles just removes
  // one instruction, which we know is safe.  This is good for things like
  // turning: (splat(splat)) -> splat, or
  // merge(V[0..n], V[n+1..2n]) -> V[0..2n]
  ShuffleVectorInst* LHSShuffle = dyn_cast<ShuffleVectorInst>(LHS);
  ShuffleVectorInst* RHSShuffle = dyn_cast<ShuffleVectorInst>(RHS);
  if (LHSShuffle)
    if (!isa<UndefValue>(LHSShuffle->getOperand(1)) && !isa<UndefValue>(RHS))
      LHSShuffle = nullptr;
  if (RHSShuffle)
    if (!isa<UndefValue>(RHSShuffle->getOperand(1)))
      RHSShuffle = nullptr;
  if (!LHSShuffle && !RHSShuffle)
    return MadeChange ? &SVI : nullptr;

  Value* LHSOp0 = nullptr;
  Value* LHSOp1 = nullptr;
  Value* RHSOp0 = nullptr;
  unsigned LHSOp0Width = 0;
  unsigned RHSOp0Width = 0;
  if (LHSShuffle) {
    LHSOp0 = LHSShuffle->getOperand(0);
    LHSOp1 = LHSShuffle->getOperand(1);
    LHSOp0Width = cast<VectorType>(LHSOp0->getType())->getNumElements();
  }
  if (RHSShuffle) {
    RHSOp0 = RHSShuffle->getOperand(0);
    RHSOp0Width = cast<VectorType>(RHSOp0->getType())->getNumElements();
  }
  Value* newLHS = LHS;
  Value* newRHS = RHS;
  if (LHSShuffle) {
    // case 1
    if (isa<UndefValue>(RHS)) {
      newLHS = LHSOp0;
      newRHS = LHSOp1;
    }
    // case 2 or 4
    else if (LHSOp0Width == LHSWidth) {
      newLHS = LHSOp0;
    }
  }
  // case 3 or 4
  if (RHSShuffle && RHSOp0Width == LHSWidth) {
    newRHS = RHSOp0;
  }
  // case 4
  if (LHSOp0 == RHSOp0) {
    newLHS = LHSOp0;
    newRHS = nullptr;
  }

  if (newLHS == LHS && newRHS == RHS)
    return MadeChange ? &SVI : nullptr;

  SmallVector<int, 16> LHSMask;
  SmallVector<int, 16> RHSMask;
  if (newLHS != LHS)
    LHSMask = LHSShuffle->getShuffleMask();
  if (RHSShuffle && newRHS != RHS)
    RHSMask = RHSShuffle->getShuffleMask();

  unsigned newLHSWidth = (newLHS != LHS) ? LHSOp0Width : LHSWidth;
  SmallVector<int, 16> newMask;
  bool isSplat = true;
  int SplatElt = -1;
  // Create a new mask for the new ShuffleVectorInst so that the new
  // ShuffleVectorInst is equivalent to the original one.
  for (unsigned i = 0; i < VWidth; ++i) {
    int eltMask;
    if (Mask[i] < 0) {
      // This element is an undef value.
      eltMask = -1;
    } else if (Mask[i] < (int)LHSWidth) {
      // This element is from left hand side vector operand.
      //
      // If LHS is going to be replaced (case 1, 2, or 4), calculate the
      // new mask value for the element.
      if (newLHS != LHS) {
        eltMask = LHSMask[Mask[i]];
        // If the value selected is an undef value, explicitly specify it
        // with a -1 mask value.
        if (eltMask >= (int)LHSOp0Width && isa<UndefValue>(LHSOp1))
          eltMask = -1;
      } else
        eltMask = Mask[i];
    } else {
      // This element is from right hand side vector operand
      //
      // If the value selected is an undef value, explicitly specify it
      // with a -1 mask value. (case 1)
      if (isa<UndefValue>(RHS))
        eltMask = -1;
      // If RHS is going to be replaced (case 3 or 4), calculate the
      // new mask value for the element.
      else if (newRHS != RHS) {
        eltMask = RHSMask[Mask[i]-LHSWidth];
        // If the value selected is an undef value, explicitly specify it
        // with a -1 mask value.
        if (eltMask >= (int)RHSOp0Width) {
          assert(isa<UndefValue>(RHSShuffle->getOperand(1))
                 && "should have been check above");
          eltMask = -1;
        }
      } else
        eltMask = Mask[i]-LHSWidth;

      // If LHS's width is changed, shift the mask value accordingly.
      // If newRHS == NULL, i.e. LHSOp0 == RHSOp0, we want to remap any
      // references from RHSOp0 to LHSOp0, so we don't need to shift the mask.
      // If newRHS == newLHS, we want to remap any references from newRHS to
      // newLHS so that we can properly identify splats that may occur due to
      // obfuscation across the two vectors.
      if (eltMask >= 0 && newRHS != nullptr && newLHS != newRHS)
        eltMask += newLHSWidth;
    }

    // Check if this could still be a splat.
    if (eltMask >= 0) {
      if (SplatElt >= 0 && SplatElt != eltMask)
        isSplat = false;
      SplatElt = eltMask;
    }

    newMask.push_back(eltMask);
  }

  // If the result mask is equal to one of the original shuffle masks,
  // or is a splat, do the replacement.
  if (isSplat || newMask == LHSMask || newMask == RHSMask || newMask == Mask) {
    SmallVector<Constant*, 16> Elts;
    for (unsigned i = 0, e = newMask.size(); i != e; ++i) {
      if (newMask[i] < 0) {
        Elts.push_back(UndefValue::get(Int32Ty));
      } else {
        Elts.push_back(ConstantInt::get(Int32Ty, newMask[i]));
      }
    }
    if (!newRHS)
      newRHS = UndefValue::get(newLHS->getType());
    return new ShuffleVectorInst(newLHS, newRHS, ConstantVector::get(Elts));
  }

  // If the result mask is an identity, replace uses of this instruction with
  // corresponding argument.
  bool isLHSID, isRHSID;
  RecognizeIdentityMask(newMask, isLHSID, isRHSID);
  if (isLHSID && VWidth == LHSOp0Width) return ReplaceInstUsesWith(SVI, newLHS);
  if (isRHSID && VWidth == RHSOp0Width) return ReplaceInstUsesWith(SVI, newRHS);

  return MadeChange ? &SVI : nullptr;
}

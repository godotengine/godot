//===----------- VectorUtils.cpp - Vectorizer utility functions -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines vectorizer utilities.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Value.h"

/// \brief Identify if the intrinsic is trivially vectorizable.
/// This method returns true if the intrinsic's argument types are all
/// scalars for the scalar form of the intrinsic and all vectors for
/// the vector form of the intrinsic.
bool llvm::isTriviallyVectorizable(Intrinsic::ID ID) {
  switch (ID) {
  case Intrinsic::sqrt:
  case Intrinsic::sin:
  case Intrinsic::cos:
  case Intrinsic::exp:
  case Intrinsic::exp2:
  case Intrinsic::log:
  case Intrinsic::log10:
  case Intrinsic::log2:
  case Intrinsic::fabs:
  case Intrinsic::minnum:
  case Intrinsic::maxnum:
  case Intrinsic::copysign:
  case Intrinsic::floor:
  case Intrinsic::ceil:
  case Intrinsic::trunc:
  case Intrinsic::rint:
  case Intrinsic::nearbyint:
  case Intrinsic::round:
  case Intrinsic::bswap:
  case Intrinsic::ctpop:
  case Intrinsic::pow:
  case Intrinsic::fma:
  case Intrinsic::fmuladd:
  case Intrinsic::ctlz:
  case Intrinsic::cttz:
  case Intrinsic::powi:
    return true;
  default:
    return false;
  }
}

/// \brief Identifies if the intrinsic has a scalar operand. It check for
/// ctlz,cttz and powi special intrinsics whose argument is scalar.
bool llvm::hasVectorInstrinsicScalarOpd(Intrinsic::ID ID,
                                        unsigned ScalarOpdIdx) {
  switch (ID) {
  case Intrinsic::ctlz:
  case Intrinsic::cttz:
  case Intrinsic::powi:
    return (ScalarOpdIdx == 1);
  default:
    return false;
  }
}

/// \brief Check call has a unary float signature
/// It checks following:
/// a) call should have a single argument
/// b) argument type should be floating point type
/// c) call instruction type and argument type should be same
/// d) call should only reads memory.
/// If all these condition is met then return ValidIntrinsicID
/// else return not_intrinsic.
llvm::Intrinsic::ID
llvm::checkUnaryFloatSignature(const CallInst &I,
                               Intrinsic::ID ValidIntrinsicID) {
  if (I.getNumArgOperands() != 1 ||
      !I.getArgOperand(0)->getType()->isFloatingPointTy() ||
      I.getType() != I.getArgOperand(0)->getType() || !I.onlyReadsMemory())
    return Intrinsic::not_intrinsic;

  return ValidIntrinsicID;
}

/// \brief Check call has a binary float signature
/// It checks following:
/// a) call should have 2 arguments.
/// b) arguments type should be floating point type
/// c) call instruction type and arguments type should be same
/// d) call should only reads memory.
/// If all these condition is met then return ValidIntrinsicID
/// else return not_intrinsic.
llvm::Intrinsic::ID
llvm::checkBinaryFloatSignature(const CallInst &I,
                                Intrinsic::ID ValidIntrinsicID) {
  if (I.getNumArgOperands() != 2 ||
      !I.getArgOperand(0)->getType()->isFloatingPointTy() ||
      !I.getArgOperand(1)->getType()->isFloatingPointTy() ||
      I.getType() != I.getArgOperand(0)->getType() ||
      I.getType() != I.getArgOperand(1)->getType() || !I.onlyReadsMemory())
    return Intrinsic::not_intrinsic;

  return ValidIntrinsicID;
}

/// \brief Returns intrinsic ID for call.
/// For the input call instruction it finds mapping intrinsic and returns
/// its ID, in case it does not found it return not_intrinsic.
llvm::Intrinsic::ID llvm::getIntrinsicIDForCall(CallInst *CI,
                                                const TargetLibraryInfo *TLI) {
  // If we have an intrinsic call, check if it is trivially vectorizable.
  if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(CI)) {
    Intrinsic::ID ID = II->getIntrinsicID();
    if (isTriviallyVectorizable(ID) || ID == Intrinsic::lifetime_start ||
        ID == Intrinsic::lifetime_end || ID == Intrinsic::assume)
      return ID;
    return Intrinsic::not_intrinsic;
  }

  if (!TLI)
    return Intrinsic::not_intrinsic;

  LibFunc::Func Func;
  Function *F = CI->getCalledFunction();
  // We're going to make assumptions on the semantics of the functions, check
  // that the target knows that it's available in this environment and it does
  // not have local linkage.
  if (!F || F->hasLocalLinkage() || !TLI->getLibFunc(F->getName(), Func))
    return Intrinsic::not_intrinsic;

  // Otherwise check if we have a call to a function that can be turned into a
  // vector intrinsic.
  switch (Func) {
  default:
    break;
  case LibFunc::sin:
  case LibFunc::sinf:
  case LibFunc::sinl:
    return checkUnaryFloatSignature(*CI, Intrinsic::sin);
  case LibFunc::cos:
  case LibFunc::cosf:
  case LibFunc::cosl:
    return checkUnaryFloatSignature(*CI, Intrinsic::cos);
  case LibFunc::exp:
  case LibFunc::expf:
  case LibFunc::expl:
    return checkUnaryFloatSignature(*CI, Intrinsic::exp);
  case LibFunc::exp2:
  case LibFunc::exp2f:
  case LibFunc::exp2l:
    return checkUnaryFloatSignature(*CI, Intrinsic::exp2);
  case LibFunc::log:
  case LibFunc::logf:
  case LibFunc::logl:
    return checkUnaryFloatSignature(*CI, Intrinsic::log);
  case LibFunc::log10:
  case LibFunc::log10f:
  case LibFunc::log10l:
    return checkUnaryFloatSignature(*CI, Intrinsic::log10);
  case LibFunc::log2:
  case LibFunc::log2f:
  case LibFunc::log2l:
    return checkUnaryFloatSignature(*CI, Intrinsic::log2);
  case LibFunc::fabs:
  case LibFunc::fabsf:
  case LibFunc::fabsl:
    return checkUnaryFloatSignature(*CI, Intrinsic::fabs);
  case LibFunc::fmin:
  case LibFunc::fminf:
  case LibFunc::fminl:
    return checkBinaryFloatSignature(*CI, Intrinsic::minnum);
  case LibFunc::fmax:
  case LibFunc::fmaxf:
  case LibFunc::fmaxl:
    return checkBinaryFloatSignature(*CI, Intrinsic::maxnum);
  case LibFunc::copysign:
  case LibFunc::copysignf:
  case LibFunc::copysignl:
    return checkBinaryFloatSignature(*CI, Intrinsic::copysign);
  case LibFunc::floor:
  case LibFunc::floorf:
  case LibFunc::floorl:
    return checkUnaryFloatSignature(*CI, Intrinsic::floor);
  case LibFunc::ceil:
  case LibFunc::ceilf:
  case LibFunc::ceill:
    return checkUnaryFloatSignature(*CI, Intrinsic::ceil);
  case LibFunc::trunc:
  case LibFunc::truncf:
  case LibFunc::truncl:
    return checkUnaryFloatSignature(*CI, Intrinsic::trunc);
  case LibFunc::rint:
  case LibFunc::rintf:
  case LibFunc::rintl:
    return checkUnaryFloatSignature(*CI, Intrinsic::rint);
  case LibFunc::nearbyint:
  case LibFunc::nearbyintf:
  case LibFunc::nearbyintl:
    return checkUnaryFloatSignature(*CI, Intrinsic::nearbyint);
  case LibFunc::round:
  case LibFunc::roundf:
  case LibFunc::roundl:
    return checkUnaryFloatSignature(*CI, Intrinsic::round);
  case LibFunc::pow:
  case LibFunc::powf:
  case LibFunc::powl:
    return checkBinaryFloatSignature(*CI, Intrinsic::pow);
  }

  return Intrinsic::not_intrinsic;
}

/// \brief Find the operand of the GEP that should be checked for consecutive
/// stores. This ignores trailing indices that have no effect on the final
/// pointer.
unsigned llvm::getGEPInductionOperand(const GetElementPtrInst *Gep) {
  const DataLayout &DL = Gep->getModule()->getDataLayout();
  unsigned LastOperand = Gep->getNumOperands() - 1;
  unsigned GEPAllocSize = DL.getTypeAllocSize(
      cast<PointerType>(Gep->getType()->getScalarType())->getElementType());

  // Walk backwards and try to peel off zeros.
  while (LastOperand > 1 &&
         match(Gep->getOperand(LastOperand), llvm::PatternMatch::m_Zero())) {
    // Find the type we're currently indexing into.
    gep_type_iterator GEPTI = gep_type_begin(Gep);
    std::advance(GEPTI, LastOperand - 1);

    // If it's a type with the same allocation size as the result of the GEP we
    // can peel off the zero index.
    if (DL.getTypeAllocSize(*GEPTI) != GEPAllocSize)
      break;
    --LastOperand;
  }

  return LastOperand;
}

/// \brief If the argument is a GEP, then returns the operand identified by
/// getGEPInductionOperand. However, if there is some other non-loop-invariant
/// operand, it returns that instead.
llvm::Value *llvm::stripGetElementPtr(llvm::Value *Ptr, ScalarEvolution *SE,
                                      Loop *Lp) {
  GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Ptr);
  if (!GEP)
    return Ptr;

  unsigned InductionOperand = getGEPInductionOperand(GEP);

  // Check that all of the gep indices are uniform except for our induction
  // operand.
  for (unsigned i = 0, e = GEP->getNumOperands(); i != e; ++i)
    if (i != InductionOperand &&
        !SE->isLoopInvariant(SE->getSCEV(GEP->getOperand(i)), Lp))
      return Ptr;
  return GEP->getOperand(InductionOperand);
}

/// \brief If a value has only one user that is a CastInst, return it.
llvm::Value *llvm::getUniqueCastUse(llvm::Value *Ptr, Loop *Lp, Type *Ty) {
  llvm::Value *UniqueCast = nullptr;
  for (User *U : Ptr->users()) {
    CastInst *CI = dyn_cast<CastInst>(U);
    if (CI && CI->getType() == Ty) {
      if (!UniqueCast)
        UniqueCast = CI;
      else
        return nullptr;
    }
  }
  return UniqueCast;
}

/// \brief Get the stride of a pointer access in a loop. Looks for symbolic
/// strides "a[i*stride]". Returns the symbolic stride, or null otherwise.
llvm::Value *llvm::getStrideFromPointer(llvm::Value *Ptr, ScalarEvolution *SE,
                                        Loop *Lp) {
  const PointerType *PtrTy = dyn_cast<PointerType>(Ptr->getType());
  if (!PtrTy || PtrTy->isAggregateType())
    return nullptr;

  // Try to remove a gep instruction to make the pointer (actually index at this
  // point) easier analyzable. If OrigPtr is equal to Ptr we are analzying the
  // pointer, otherwise, we are analyzing the index.
  llvm::Value *OrigPtr = Ptr;

  // The size of the pointer access.
  int64_t PtrAccessSize = 1;

  Ptr = stripGetElementPtr(Ptr, SE, Lp);
  const SCEV *V = SE->getSCEV(Ptr);

  if (Ptr != OrigPtr)
    // Strip off casts.
    while (const SCEVCastExpr *C = dyn_cast<SCEVCastExpr>(V))
      V = C->getOperand();

  const SCEVAddRecExpr *S = dyn_cast<SCEVAddRecExpr>(V);
  if (!S)
    return nullptr;

  V = S->getStepRecurrence(*SE);
  if (!V)
    return nullptr;

  // Strip off the size of access multiplication if we are still analyzing the
  // pointer.
  if (OrigPtr == Ptr) {
    const DataLayout &DL = Lp->getHeader()->getModule()->getDataLayout();
    DL.getTypeAllocSize(PtrTy->getElementType());
    if (const SCEVMulExpr *M = dyn_cast<SCEVMulExpr>(V)) {
      if (M->getOperand(0)->getSCEVType() != scConstant)
        return nullptr;

      const APInt &APStepVal =
          cast<SCEVConstant>(M->getOperand(0))->getValue()->getValue();

      // Huge step value - give up.
      if (APStepVal.getBitWidth() > 64)
        return nullptr;

      int64_t StepVal = APStepVal.getSExtValue();
      if (PtrAccessSize != StepVal)
        return nullptr;
      V = M->getOperand(1);
    }
  }

  // Strip off casts.
  Type *StripedOffRecurrenceCast = nullptr;
  if (const SCEVCastExpr *C = dyn_cast<SCEVCastExpr>(V)) {
    StripedOffRecurrenceCast = C->getType();
    V = C->getOperand();
  }

  // Look for the loop invariant symbolic value.
  const SCEVUnknown *U = dyn_cast<SCEVUnknown>(V);
  if (!U)
    return nullptr;

  llvm::Value *Stride = U->getValue();
  if (!Lp->isLoopInvariant(Stride))
    return nullptr;

  // If we have stripped off the recurrence cast we have to make sure that we
  // return the value that is used in this loop so that we can replace it later.
  if (StripedOffRecurrenceCast)
    Stride = getUniqueCastUse(Stride, Lp, StripedOffRecurrenceCast);

  return Stride;
}

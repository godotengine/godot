//===-- Constants.cpp - Implement Constant nodes --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Constant* classes.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Constants.h"
#include "ConstantFold.h"
#include "LLVMContextImpl.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstdarg>
using namespace llvm;

//===----------------------------------------------------------------------===//
//                              Constant Class
//===----------------------------------------------------------------------===//

void Constant::anchor() { }

bool Constant::isNegativeZeroValue() const {
  // Floating point values have an explicit -0.0 value.
  if (const ConstantFP *CFP = dyn_cast<ConstantFP>(this))
    return CFP->isZero() && CFP->isNegative();

  // Equivalent for a vector of -0.0's.
  if (const ConstantDataVector *CV = dyn_cast<ConstantDataVector>(this))
    if (ConstantFP *SplatCFP = dyn_cast_or_null<ConstantFP>(CV->getSplatValue()))
      if (SplatCFP && SplatCFP->isZero() && SplatCFP->isNegative())
        return true;

  // We've already handled true FP case; any other FP vectors can't represent -0.0.
  if (getType()->isFPOrFPVectorTy())
    return false;

  // Otherwise, just use +0.0.
  return isNullValue();
}

// Return true iff this constant is positive zero (floating point), negative
// zero (floating point), or a null value.
bool Constant::isZeroValue() const {
  // Floating point values have an explicit -0.0 value.
  if (const ConstantFP *CFP = dyn_cast<ConstantFP>(this))
    return CFP->isZero();

  // Otherwise, just use +0.0.
  return isNullValue();
}

bool Constant::isNullValue() const {
  // 0 is null.
  if (const ConstantInt *CI = dyn_cast<ConstantInt>(this))
    return CI->isZero();

  // +0.0 is null.
  if (const ConstantFP *CFP = dyn_cast<ConstantFP>(this))
    return CFP->isZero() && !CFP->isNegative();

  // constant zero is zero for aggregates and cpnull is null for pointers.
  return isa<ConstantAggregateZero>(this) || isa<ConstantPointerNull>(this);
}

bool Constant::isAllOnesValue() const {
  // Check for -1 integers
  if (const ConstantInt *CI = dyn_cast<ConstantInt>(this))
    return CI->isMinusOne();

  // Check for FP which are bitcasted from -1 integers
  if (const ConstantFP *CFP = dyn_cast<ConstantFP>(this))
    return CFP->getValueAPF().bitcastToAPInt().isAllOnesValue();

  // Check for constant vectors which are splats of -1 values.
  if (const ConstantVector *CV = dyn_cast<ConstantVector>(this))
    if (Constant *Splat = CV->getSplatValue())
      return Splat->isAllOnesValue();

  // Check for constant vectors which are splats of -1 values.
  if (const ConstantDataVector *CV = dyn_cast<ConstantDataVector>(this))
    if (Constant *Splat = CV->getSplatValue())
      return Splat->isAllOnesValue();

  return false;
}

bool Constant::isOneValue() const {
  // Check for 1 integers
  if (const ConstantInt *CI = dyn_cast<ConstantInt>(this))
    return CI->isOne();

  // Check for FP which are bitcasted from 1 integers
  if (const ConstantFP *CFP = dyn_cast<ConstantFP>(this))
    return CFP->getValueAPF().bitcastToAPInt() == 1;

  // Check for constant vectors which are splats of 1 values.
  if (const ConstantVector *CV = dyn_cast<ConstantVector>(this))
    if (Constant *Splat = CV->getSplatValue())
      return Splat->isOneValue();

  // Check for constant vectors which are splats of 1 values.
  if (const ConstantDataVector *CV = dyn_cast<ConstantDataVector>(this))
    if (Constant *Splat = CV->getSplatValue())
      return Splat->isOneValue();

  return false;
}

bool Constant::isMinSignedValue() const {
  // Check for INT_MIN integers
  if (const ConstantInt *CI = dyn_cast<ConstantInt>(this))
    return CI->isMinValue(/*isSigned=*/true);

  // Check for FP which are bitcasted from INT_MIN integers
  if (const ConstantFP *CFP = dyn_cast<ConstantFP>(this))
    return CFP->getValueAPF().bitcastToAPInt().isMinSignedValue();

  // Check for constant vectors which are splats of INT_MIN values.
  if (const ConstantVector *CV = dyn_cast<ConstantVector>(this))
    if (Constant *Splat = CV->getSplatValue())
      return Splat->isMinSignedValue();

  // Check for constant vectors which are splats of INT_MIN values.
  if (const ConstantDataVector *CV = dyn_cast<ConstantDataVector>(this))
    if (Constant *Splat = CV->getSplatValue())
      return Splat->isMinSignedValue();

  return false;
}

bool Constant::isNotMinSignedValue() const {
  // Check for INT_MIN integers
  if (const ConstantInt *CI = dyn_cast<ConstantInt>(this))
    return !CI->isMinValue(/*isSigned=*/true);

  // Check for FP which are bitcasted from INT_MIN integers
  if (const ConstantFP *CFP = dyn_cast<ConstantFP>(this))
    return !CFP->getValueAPF().bitcastToAPInt().isMinSignedValue();

  // Check for constant vectors which are splats of INT_MIN values.
  if (const ConstantVector *CV = dyn_cast<ConstantVector>(this))
    if (Constant *Splat = CV->getSplatValue())
      return Splat->isNotMinSignedValue();

  // Check for constant vectors which are splats of INT_MIN values.
  if (const ConstantDataVector *CV = dyn_cast<ConstantDataVector>(this))
    if (Constant *Splat = CV->getSplatValue())
      return Splat->isNotMinSignedValue();

  // It *may* contain INT_MIN, we can't tell.
  return false;
}

// Constructor to create a '0' constant of arbitrary type...
Constant *Constant::getNullValue(Type *Ty) {
  switch (Ty->getTypeID()) {
  case Type::IntegerTyID:
    return ConstantInt::get(Ty, 0);
  case Type::HalfTyID:
    return ConstantFP::get(Ty->getContext(),
                           APFloat::getZero(APFloat::IEEEhalf));
  case Type::FloatTyID:
    return ConstantFP::get(Ty->getContext(),
                           APFloat::getZero(APFloat::IEEEsingle));
  case Type::DoubleTyID:
    return ConstantFP::get(Ty->getContext(),
                           APFloat::getZero(APFloat::IEEEdouble));
  case Type::X86_FP80TyID:
    return ConstantFP::get(Ty->getContext(),
                           APFloat::getZero(APFloat::x87DoubleExtended));
  case Type::FP128TyID:
    return ConstantFP::get(Ty->getContext(),
                           APFloat::getZero(APFloat::IEEEquad));
  case Type::PPC_FP128TyID:
    return ConstantFP::get(Ty->getContext(),
                           APFloat(APFloat::PPCDoubleDouble,
                                   APInt::getNullValue(128)));
  case Type::PointerTyID:
    return ConstantPointerNull::get(cast<PointerType>(Ty));
  case Type::StructTyID:
  case Type::ArrayTyID:
  case Type::VectorTyID:
    return ConstantAggregateZero::get(Ty);
  default:
    // Function, Label, or Opaque type?
    llvm_unreachable("Cannot create a null constant of that type!");
  }
}

Constant *Constant::getIntegerValue(Type *Ty, const APInt &V) {
  Type *ScalarTy = Ty->getScalarType();

  // Create the base integer constant.
  Constant *C = ConstantInt::get(Ty->getContext(), V);

  // Convert an integer to a pointer, if necessary.
  if (PointerType *PTy = dyn_cast<PointerType>(ScalarTy))
    C = ConstantExpr::getIntToPtr(C, PTy);

  // Broadcast a scalar to a vector, if necessary.
  if (VectorType *VTy = dyn_cast<VectorType>(Ty))
    C = ConstantVector::getSplat(VTy->getNumElements(), C);

  return C;
}

Constant *Constant::getAllOnesValue(Type *Ty) {
  if (IntegerType *ITy = dyn_cast<IntegerType>(Ty))
    return ConstantInt::get(Ty->getContext(),
                            APInt::getAllOnesValue(ITy->getBitWidth()));

  if (Ty->isFloatingPointTy()) {
    APFloat FL = APFloat::getAllOnesValue(Ty->getPrimitiveSizeInBits(),
                                          !Ty->isPPC_FP128Ty());
    return ConstantFP::get(Ty->getContext(), FL);
  }

  VectorType *VTy = cast<VectorType>(Ty);
  return ConstantVector::getSplat(VTy->getNumElements(),
                                  getAllOnesValue(VTy->getElementType()));
}

/// getAggregateElement - For aggregates (struct/array/vector) return the
/// constant that corresponds to the specified element if possible, or null if
/// not.  This can return null if the element index is a ConstantExpr, or if
/// 'this' is a constant expr.
Constant *Constant::getAggregateElement(unsigned Elt) const {
  if (const ConstantStruct *CS = dyn_cast<ConstantStruct>(this))
    return Elt < CS->getNumOperands() ? CS->getOperand(Elt) : nullptr;

  if (const ConstantArray *CA = dyn_cast<ConstantArray>(this))
    return Elt < CA->getNumOperands() ? CA->getOperand(Elt) : nullptr;

  if (const ConstantVector *CV = dyn_cast<ConstantVector>(this))
    return Elt < CV->getNumOperands() ? CV->getOperand(Elt) : nullptr;

  if (const ConstantAggregateZero *CAZ = dyn_cast<ConstantAggregateZero>(this))
    return Elt < CAZ->getNumElements() ? CAZ->getElementValue(Elt) : nullptr;

  if (const UndefValue *UV = dyn_cast<UndefValue>(this))
    return Elt < UV->getNumElements() ? UV->getElementValue(Elt) : nullptr;

  if (const ConstantDataSequential *CDS =dyn_cast<ConstantDataSequential>(this))
    return Elt < CDS->getNumElements() ? CDS->getElementAsConstant(Elt)
                                       : nullptr;
  return nullptr;
}

Constant *Constant::getAggregateElement(Constant *Elt) const {
  assert(isa<IntegerType>(Elt->getType()) && "Index must be an integer");
  if (ConstantInt *CI = dyn_cast<ConstantInt>(Elt))
    return getAggregateElement(CI->getZExtValue());
  return nullptr;
}

void Constant::destroyConstant() {
  /// First call destroyConstantImpl on the subclass.  This gives the subclass
  /// a chance to remove the constant from any maps/pools it's contained in.
  switch (getValueID()) {
  default:
    llvm_unreachable("Not a constant!");
#define HANDLE_CONSTANT(Name)                                                  \
  case Value::Name##Val:                                                       \
    cast<Name>(this)->destroyConstantImpl();                                   \
    break;
#include "llvm/IR/Value.def"
  }

  // When a Constant is destroyed, there may be lingering
  // references to the constant by other constants in the constant pool.  These
  // constants are implicitly dependent on the module that is being deleted,
  // but they don't know that.  Because we only find out when the CPV is
  // deleted, we must now notify all of our users (that should only be
  // Constants) that they are, in fact, invalid now and should be deleted.
  //
  while (!use_empty()) {
    Value *V = user_back();
#ifndef NDEBUG // Only in -g mode...
    if (!isa<Constant>(V)) {
      dbgs() << "While deleting: " << *this
             << "\n\nUse still stuck around after Def is destroyed: " << *V
             << "\n\n";
    }
#endif
    assert(isa<Constant>(V) && "References remain to Constant being destroyed");
    cast<Constant>(V)->destroyConstant();

    // The constant should remove itself from our use list...
    assert((use_empty() || user_back() != V) && "Constant not removed!");
  }

  // Value has no outstanding references it is safe to delete it now...
  delete this;
}

static bool canTrapImpl(const Constant *C,
                        SmallPtrSetImpl<const ConstantExpr *> &NonTrappingOps) {
  assert(C->getType()->isFirstClassType() && "Cannot evaluate aggregate vals!");
  // The only thing that could possibly trap are constant exprs.
  const ConstantExpr *CE = dyn_cast<ConstantExpr>(C);
  if (!CE)
    return false;

  // ConstantExpr traps if any operands can trap.
  for (unsigned i = 0, e = C->getNumOperands(); i != e; ++i) {
    if (ConstantExpr *Op = dyn_cast<ConstantExpr>(CE->getOperand(i))) {
      if (NonTrappingOps.insert(Op).second && canTrapImpl(Op, NonTrappingOps))
        return true;
    }
  }

  // Otherwise, only specific operations can trap.
  switch (CE->getOpcode()) {
  default:
    return false;
  case Instruction::UDiv:
  case Instruction::SDiv:
  case Instruction::FDiv:
  case Instruction::URem:
  case Instruction::SRem:
  case Instruction::FRem:
    // Div and rem can trap if the RHS is not known to be non-zero.
    if (!isa<ConstantInt>(CE->getOperand(1)) ||CE->getOperand(1)->isNullValue())
      return true;
    return false;
  }
}

/// canTrap - Return true if evaluation of this constant could trap.  This is
/// true for things like constant expressions that could divide by zero.
bool Constant::canTrap() const {
  SmallPtrSet<const ConstantExpr *, 4> NonTrappingOps;
  return canTrapImpl(this, NonTrappingOps);
}

/// Check if C contains a GlobalValue for which Predicate is true.
static bool
ConstHasGlobalValuePredicate(const Constant *C,
                             bool (*Predicate)(const GlobalValue *)) {
  SmallPtrSet<const Constant *, 8> Visited;
  SmallVector<const Constant *, 8> WorkList;
  WorkList.push_back(C);
  Visited.insert(C);

  while (!WorkList.empty()) {
    const Constant *WorkItem = WorkList.pop_back_val();
    if (const auto *GV = dyn_cast<GlobalValue>(WorkItem))
      if (Predicate(GV))
        return true;
    for (const Value *Op : WorkItem->operands()) {
      const Constant *ConstOp = dyn_cast<Constant>(Op);
      if (!ConstOp)
        continue;
      if (Visited.insert(ConstOp).second)
        WorkList.push_back(ConstOp);
    }
  }
  return false;
}

/// Return true if the value can vary between threads.
bool Constant::isThreadDependent() const {
  auto DLLImportPredicate = [](const GlobalValue *GV) {
    return GV->isThreadLocal();
  };
  return ConstHasGlobalValuePredicate(this, DLLImportPredicate);
}

bool Constant::isDLLImportDependent() const {
  auto DLLImportPredicate = [](const GlobalValue *GV) {
    return GV->hasDLLImportStorageClass();
  };
  return ConstHasGlobalValuePredicate(this, DLLImportPredicate);
}

/// Return true if the constant has users other than constant exprs and other
/// dangling things.
bool Constant::isConstantUsed() const {
  for (const User *U : users()) {
    const Constant *UC = dyn_cast<Constant>(U);
    if (!UC || isa<GlobalValue>(UC))
      return true;

    if (UC->isConstantUsed())
      return true;
  }
  return false;
}



/// getRelocationInfo - This method classifies the entry according to
/// whether or not it may generate a relocation entry.  This must be
/// conservative, so if it might codegen to a relocatable entry, it should say
/// so.  The return values are:
/// 
///  NoRelocation: This constant pool entry is guaranteed to never have a
///     relocation applied to it (because it holds a simple constant like
///     '4').
///  LocalRelocation: This entry has relocations, but the entries are
///     guaranteed to be resolvable by the static linker, so the dynamic
///     linker will never see them.
///  GlobalRelocations: This entry may have arbitrary relocations.
///
/// FIXME: This really should not be in IR.
Constant::PossibleRelocationsTy Constant::getRelocationInfo() const {
  if (const GlobalValue *GV = dyn_cast<GlobalValue>(this)) {
    if (GV->hasLocalLinkage() || GV->hasHiddenVisibility())
      return LocalRelocation;  // Local to this file/library.
    return GlobalRelocations;    // Global reference.
  }
  
  if (const BlockAddress *BA = dyn_cast<BlockAddress>(this))
    return BA->getFunction()->getRelocationInfo();
  
  // While raw uses of blockaddress need to be relocated, differences between
  // two of them don't when they are for labels in the same function.  This is a
  // common idiom when creating a table for the indirect goto extension, so we
  // handle it efficiently here.
  if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(this))
    if (CE->getOpcode() == Instruction::Sub) {
      ConstantExpr *LHS = dyn_cast<ConstantExpr>(CE->getOperand(0));
      ConstantExpr *RHS = dyn_cast<ConstantExpr>(CE->getOperand(1));
      if (LHS && RHS &&
          LHS->getOpcode() == Instruction::PtrToInt &&
          RHS->getOpcode() == Instruction::PtrToInt &&
          isa<BlockAddress>(LHS->getOperand(0)) &&
          isa<BlockAddress>(RHS->getOperand(0)) &&
          cast<BlockAddress>(LHS->getOperand(0))->getFunction() ==
            cast<BlockAddress>(RHS->getOperand(0))->getFunction())
        return NoRelocation;
    }

  PossibleRelocationsTy Result = NoRelocation;
  for (unsigned i = 0, e = getNumOperands(); i != e; ++i)
    Result = std::max(Result,
                      cast<Constant>(getOperand(i))->getRelocationInfo());

  return Result;
}

/// removeDeadUsersOfConstant - If the specified constantexpr is dead, remove
/// it.  This involves recursively eliminating any dead users of the
/// constantexpr.
static bool removeDeadUsersOfConstant(const Constant *C) {
  if (isa<GlobalValue>(C)) return false; // Cannot remove this

  while (!C->use_empty()) {
    const Constant *User = dyn_cast<Constant>(C->user_back());
    if (!User) return false; // Non-constant usage;
    if (!removeDeadUsersOfConstant(User))
      return false; // Constant wasn't dead
  }

  const_cast<Constant*>(C)->destroyConstant();
  return true;
}


/// removeDeadConstantUsers - If there are any dead constant users dangling
/// off of this constant, remove them.  This method is useful for clients
/// that want to check to see if a global is unused, but don't want to deal
/// with potentially dead constants hanging off of the globals.
void Constant::removeDeadConstantUsers() const {
  Value::const_user_iterator I = user_begin(), E = user_end();
  Value::const_user_iterator LastNonDeadUser = E;
  while (I != E) {
    const Constant *User = dyn_cast<Constant>(*I);
    if (!User) {
      LastNonDeadUser = I;
      ++I;
      continue;
    }

    if (!removeDeadUsersOfConstant(User)) {
      // If the constant wasn't dead, remember that this was the last live use
      // and move on to the next constant.
      LastNonDeadUser = I;
      ++I;
      continue;
    }

    // If the constant was dead, then the iterator is invalidated.
    if (LastNonDeadUser == E) {
      I = user_begin();
      if (I == E) break;
    } else {
      I = LastNonDeadUser;
      ++I;
    }
  }
}



//===----------------------------------------------------------------------===//
//                                ConstantInt
//===----------------------------------------------------------------------===//

void ConstantInt::anchor() { }

ConstantInt::ConstantInt(IntegerType *Ty, const APInt& V)
  : Constant(Ty, ConstantIntVal, nullptr, 0), Val(V) {
  assert(V.getBitWidth() == Ty->getBitWidth() && "Invalid constant for type");
}

ConstantInt *ConstantInt::getTrue(LLVMContext &Context) {
  LLVMContextImpl *pImpl = Context.pImpl;
  if (!pImpl->TheTrueVal)
    pImpl->TheTrueVal = ConstantInt::get(Type::getInt1Ty(Context), 1);
  return pImpl->TheTrueVal;
}

ConstantInt *ConstantInt::getFalse(LLVMContext &Context) {
  LLVMContextImpl *pImpl = Context.pImpl;
  if (!pImpl->TheFalseVal)
    pImpl->TheFalseVal = ConstantInt::get(Type::getInt1Ty(Context), 0);
  return pImpl->TheFalseVal;
}

Constant *ConstantInt::getTrue(Type *Ty) {
  VectorType *VTy = dyn_cast<VectorType>(Ty);
  if (!VTy) {
    assert(Ty->isIntegerTy(1) && "True must be i1 or vector of i1.");
    return ConstantInt::getTrue(Ty->getContext());
  }
  assert(VTy->getElementType()->isIntegerTy(1) &&
         "True must be vector of i1 or i1.");
  return ConstantVector::getSplat(VTy->getNumElements(),
                                  ConstantInt::getTrue(Ty->getContext()));
}

Constant *ConstantInt::getFalse(Type *Ty) {
  VectorType *VTy = dyn_cast<VectorType>(Ty);
  if (!VTy) {
    assert(Ty->isIntegerTy(1) && "False must be i1 or vector of i1.");
    return ConstantInt::getFalse(Ty->getContext());
  }
  assert(VTy->getElementType()->isIntegerTy(1) &&
         "False must be vector of i1 or i1.");
  return ConstantVector::getSplat(VTy->getNumElements(),
                                  ConstantInt::getFalse(Ty->getContext()));
}

// Get a ConstantInt from an APInt.
ConstantInt *ConstantInt::get(LLVMContext &Context, const APInt &V) {
  // get an existing value or the insertion position
  LLVMContextImpl *pImpl = Context.pImpl;
  ConstantInt *&Slot = pImpl->IntConstants[V];
  if (!Slot) {
    // Get the corresponding integer type for the bit width of the value.
    IntegerType *ITy = IntegerType::get(Context, V.getBitWidth());
    Slot = new ConstantInt(ITy, V);
  }
  assert(Slot->getType() == IntegerType::get(Context, V.getBitWidth()));
  return Slot;
}

Constant *ConstantInt::get(Type *Ty, uint64_t V, bool isSigned) {
  Constant *C = get(cast<IntegerType>(Ty->getScalarType()), V, isSigned);

  // For vectors, broadcast the value.
  if (VectorType *VTy = dyn_cast<VectorType>(Ty))
    return ConstantVector::getSplat(VTy->getNumElements(), C);

  return C;
}

ConstantInt *ConstantInt::get(IntegerType *Ty, uint64_t V, 
                              bool isSigned) {
  return get(Ty->getContext(), APInt(Ty->getBitWidth(), V, isSigned));
}

ConstantInt *ConstantInt::getSigned(IntegerType *Ty, int64_t V) {
  return get(Ty, V, true);
}

Constant *ConstantInt::getSigned(Type *Ty, int64_t V) {
  return get(Ty, V, true);
}

Constant *ConstantInt::get(Type *Ty, const APInt& V) {
  ConstantInt *C = get(Ty->getContext(), V);
  assert(C->getType() == Ty->getScalarType() &&
         "ConstantInt type doesn't match the type implied by its value!");

  // For vectors, broadcast the value.
  if (VectorType *VTy = dyn_cast<VectorType>(Ty))
    return ConstantVector::getSplat(VTy->getNumElements(), C);

  return C;
}

ConstantInt *ConstantInt::get(IntegerType* Ty, StringRef Str,
                              uint8_t radix) {
  return get(Ty->getContext(), APInt(Ty->getBitWidth(), Str, radix));
}

/// Remove the constant from the constant table.
void ConstantInt::destroyConstantImpl() {
  llvm_unreachable("You can't ConstantInt->destroyConstantImpl()!");
}

//===----------------------------------------------------------------------===//
//                                ConstantFP
//===----------------------------------------------------------------------===//

static const fltSemantics *TypeToFloatSemantics(Type *Ty) {
  if (Ty->isHalfTy())
    return &APFloat::IEEEhalf;
  if (Ty->isFloatTy())
    return &APFloat::IEEEsingle;
  if (Ty->isDoubleTy())
    return &APFloat::IEEEdouble;
  if (Ty->isX86_FP80Ty())
    return &APFloat::x87DoubleExtended;
  else if (Ty->isFP128Ty())
    return &APFloat::IEEEquad;

  assert(Ty->isPPC_FP128Ty() && "Unknown FP format");
  return &APFloat::PPCDoubleDouble;
}

void ConstantFP::anchor() { }

/// get() - This returns a constant fp for the specified value in the
/// specified type.  This should only be used for simple constant values like
/// 2.0/1.0 etc, that are known-valid both as double and as the target format.
Constant *ConstantFP::get(Type *Ty, double V) {
  LLVMContext &Context = Ty->getContext();

  APFloat FV(V);
  bool ignored;
  FV.convert(*TypeToFloatSemantics(Ty->getScalarType()),
             APFloat::rmNearestTiesToEven, &ignored);
  Constant *C = get(Context, FV);

  // For vectors, broadcast the value.
  if (VectorType *VTy = dyn_cast<VectorType>(Ty))
    return ConstantVector::getSplat(VTy->getNumElements(), C);

  return C;
}


Constant *ConstantFP::get(Type *Ty, StringRef Str) {
  LLVMContext &Context = Ty->getContext();

  APFloat FV(*TypeToFloatSemantics(Ty->getScalarType()), Str);
  Constant *C = get(Context, FV);

  // For vectors, broadcast the value.
  if (VectorType *VTy = dyn_cast<VectorType>(Ty))
    return ConstantVector::getSplat(VTy->getNumElements(), C);

  return C; 
}

Constant *ConstantFP::getNaN(Type *Ty, bool Negative, unsigned Type) {
  const fltSemantics &Semantics = *TypeToFloatSemantics(Ty->getScalarType());
  APFloat NaN = APFloat::getNaN(Semantics, Negative, Type);
  Constant *C = get(Ty->getContext(), NaN);

  if (VectorType *VTy = dyn_cast<VectorType>(Ty))
    return ConstantVector::getSplat(VTy->getNumElements(), C);

  return C;
}

Constant *ConstantFP::getNegativeZero(Type *Ty) {
  const fltSemantics &Semantics = *TypeToFloatSemantics(Ty->getScalarType());
  APFloat NegZero = APFloat::getZero(Semantics, /*Negative=*/true);
  Constant *C = get(Ty->getContext(), NegZero);

  if (VectorType *VTy = dyn_cast<VectorType>(Ty))
    return ConstantVector::getSplat(VTy->getNumElements(), C);

  return C;
}


Constant *ConstantFP::getZeroValueForNegation(Type *Ty) {
  if (Ty->isFPOrFPVectorTy())
    return getNegativeZero(Ty);

  return Constant::getNullValue(Ty);
}


// ConstantFP accessors.
ConstantFP* ConstantFP::get(LLVMContext &Context, const APFloat& V) {
  LLVMContextImpl* pImpl = Context.pImpl;

  ConstantFP *&Slot = pImpl->FPConstants[V];

  if (!Slot) {
    Type *Ty;
    if (&V.getSemantics() == &APFloat::IEEEhalf)
      Ty = Type::getHalfTy(Context);
    else if (&V.getSemantics() == &APFloat::IEEEsingle)
      Ty = Type::getFloatTy(Context);
    else if (&V.getSemantics() == &APFloat::IEEEdouble)
      Ty = Type::getDoubleTy(Context);
    else if (&V.getSemantics() == &APFloat::x87DoubleExtended)
      Ty = Type::getX86_FP80Ty(Context);
    else if (&V.getSemantics() == &APFloat::IEEEquad)
      Ty = Type::getFP128Ty(Context);
    else {
      assert(&V.getSemantics() == &APFloat::PPCDoubleDouble && 
             "Unknown FP format");
      Ty = Type::getPPC_FP128Ty(Context);
    }
    Slot = new ConstantFP(Ty, V);
  }

  return Slot;
}

Constant *ConstantFP::getInfinity(Type *Ty, bool Negative) {
  const fltSemantics &Semantics = *TypeToFloatSemantics(Ty->getScalarType());
  Constant *C = get(Ty->getContext(), APFloat::getInf(Semantics, Negative));

  if (VectorType *VTy = dyn_cast<VectorType>(Ty))
    return ConstantVector::getSplat(VTy->getNumElements(), C);

  return C;
}

ConstantFP::ConstantFP(Type *Ty, const APFloat& V)
  : Constant(Ty, ConstantFPVal, nullptr, 0), Val(V) {
  assert(&V.getSemantics() == TypeToFloatSemantics(Ty) &&
         "FP type Mismatch");
}

bool ConstantFP::isExactlyValue(const APFloat &V) const {
  return Val.bitwiseIsEqual(V);
}

/// Remove the constant from the constant table.
void ConstantFP::destroyConstantImpl() {
  llvm_unreachable("You can't ConstantInt->destroyConstantImpl()!");
}

//===----------------------------------------------------------------------===//
//                   ConstantAggregateZero Implementation
//===----------------------------------------------------------------------===//

/// getSequentialElement - If this CAZ has array or vector type, return a zero
/// with the right element type.
Constant *ConstantAggregateZero::getSequentialElement() const {
  return Constant::getNullValue(getType()->getSequentialElementType());
}

/// getStructElement - If this CAZ has struct type, return a zero with the
/// right element type for the specified element.
Constant *ConstantAggregateZero::getStructElement(unsigned Elt) const {
  return Constant::getNullValue(getType()->getStructElementType(Elt));
}

/// getElementValue - Return a zero of the right value for the specified GEP
/// index if we can, otherwise return null (e.g. if C is a ConstantExpr).
Constant *ConstantAggregateZero::getElementValue(Constant *C) const {
  if (isa<SequentialType>(getType()))
    return getSequentialElement();
  return getStructElement(cast<ConstantInt>(C)->getZExtValue());
}

/// getElementValue - Return a zero of the right value for the specified GEP
/// index.
Constant *ConstantAggregateZero::getElementValue(unsigned Idx) const {
  if (isa<SequentialType>(getType()))
    return getSequentialElement();
  return getStructElement(Idx);
}

unsigned ConstantAggregateZero::getNumElements() const {
  const Type *Ty = getType();
  if (const auto *AT = dyn_cast<ArrayType>(Ty))
    return AT->getNumElements();
  if (const auto *VT = dyn_cast<VectorType>(Ty))
    return VT->getNumElements();
  return Ty->getStructNumElements();
}

//===----------------------------------------------------------------------===//
//                         UndefValue Implementation
//===----------------------------------------------------------------------===//

/// getSequentialElement - If this undef has array or vector type, return an
/// undef with the right element type.
UndefValue *UndefValue::getSequentialElement() const {
  return UndefValue::get(getType()->getSequentialElementType());
}

/// getStructElement - If this undef has struct type, return a zero with the
/// right element type for the specified element.
UndefValue *UndefValue::getStructElement(unsigned Elt) const {
  return UndefValue::get(getType()->getStructElementType(Elt));
}

/// getElementValue - Return an undef of the right value for the specified GEP
/// index if we can, otherwise return null (e.g. if C is a ConstantExpr).
UndefValue *UndefValue::getElementValue(Constant *C) const {
  if (isa<SequentialType>(getType()))
    return getSequentialElement();
  return getStructElement(cast<ConstantInt>(C)->getZExtValue());
}

/// getElementValue - Return an undef of the right value for the specified GEP
/// index.
UndefValue *UndefValue::getElementValue(unsigned Idx) const {
  if (isa<SequentialType>(getType()))
    return getSequentialElement();
  return getStructElement(Idx);
}

unsigned UndefValue::getNumElements() const {
  const Type *Ty = getType();
  if (const auto *AT = dyn_cast<ArrayType>(Ty))
    return AT->getNumElements();
  if (const auto *VT = dyn_cast<VectorType>(Ty))
    return VT->getNumElements();
  return Ty->getStructNumElements();
}

//===----------------------------------------------------------------------===//
//                            ConstantXXX Classes
//===----------------------------------------------------------------------===//

template <typename ItTy, typename EltTy>
static bool rangeOnlyContains(ItTy Start, ItTy End, EltTy Elt) {
  for (; Start != End; ++Start)
    if (*Start != Elt)
      return false;
  return true;
}

ConstantArray::ConstantArray(ArrayType *T, ArrayRef<Constant *> V)
  : Constant(T, ConstantArrayVal,
             OperandTraits<ConstantArray>::op_end(this) - V.size(),
             V.size()) {
  assert(V.size() == T->getNumElements() &&
         "Invalid initializer vector for constant array");
  for (unsigned i = 0, e = V.size(); i != e; ++i)
    assert(V[i]->getType() == T->getElementType() &&
           "Initializer for array element doesn't match array element type!");
  std::copy(V.begin(), V.end(), op_begin());
}

Constant *ConstantArray::get(ArrayType *Ty, ArrayRef<Constant*> V) {
  if (Constant *C = getImpl(Ty, V))
    return C;
  return Ty->getContext().pImpl->ArrayConstants.getOrCreate(Ty, V);
}
Constant *ConstantArray::getImpl(ArrayType *Ty, ArrayRef<Constant*> V) {
  // Empty arrays are canonicalized to ConstantAggregateZero.
  if (V.empty())
    return ConstantAggregateZero::get(Ty);

  for (unsigned i = 0, e = V.size(); i != e; ++i) {
    assert(V[i]->getType() == Ty->getElementType() &&
           "Wrong type in array element initializer");
  }

  // If this is an all-zero array, return a ConstantAggregateZero object.  If
  // all undef, return an UndefValue, if "all simple", then return a
  // ConstantDataArray.
  Constant *C = V[0];
  if (isa<UndefValue>(C) && rangeOnlyContains(V.begin(), V.end(), C))
    return UndefValue::get(Ty);

  if (C->isNullValue() && rangeOnlyContains(V.begin(), V.end(), C))
    return ConstantAggregateZero::get(Ty);

  // Check to see if all of the elements are ConstantFP or ConstantInt and if
  // the element type is compatible with ConstantDataVector.  If so, use it.
  if (ConstantDataSequential::isElementTypeCompatible(C->getType())) {
    // We speculatively build the elements here even if it turns out that there
    // is a constantexpr or something else weird in the array, since it is so
    // uncommon for that to happen.
    if (ConstantInt *CI = dyn_cast<ConstantInt>(C)) {
      if (CI->getType()->isIntegerTy(8)) {
        SmallVector<uint8_t, 16> Elts;
        for (unsigned i = 0, e = V.size(); i != e; ++i)
          if (ConstantInt *CI = dyn_cast<ConstantInt>(V[i]))
            Elts.push_back(CI->getZExtValue());
          else
            break;
        if (Elts.size() == V.size())
          return ConstantDataArray::get(C->getContext(), Elts);
      } else if (CI->getType()->isIntegerTy(16)) {
        SmallVector<uint16_t, 16> Elts;
        for (unsigned i = 0, e = V.size(); i != e; ++i)
          if (ConstantInt *CI = dyn_cast<ConstantInt>(V[i]))
            Elts.push_back(CI->getZExtValue());
          else
            break;
        if (Elts.size() == V.size())
          return ConstantDataArray::get(C->getContext(), Elts);
      } else if (CI->getType()->isIntegerTy(32)) {
        SmallVector<uint32_t, 16> Elts;
        for (unsigned i = 0, e = V.size(); i != e; ++i)
          if (ConstantInt *CI = dyn_cast<ConstantInt>(V[i]))
            Elts.push_back(CI->getZExtValue());
          else
            break;
        if (Elts.size() == V.size())
          return ConstantDataArray::get(C->getContext(), Elts);
      } else if (CI->getType()->isIntegerTy(64)) {
        SmallVector<uint64_t, 16> Elts;
        for (unsigned i = 0, e = V.size(); i != e; ++i)
          if (ConstantInt *CI = dyn_cast<ConstantInt>(V[i]))
            Elts.push_back(CI->getZExtValue());
          else
            break;
        if (Elts.size() == V.size())
          return ConstantDataArray::get(C->getContext(), Elts);
      }
    }

    if (ConstantFP *CFP = dyn_cast<ConstantFP>(C)) {
      if (CFP->getType()->isFloatTy()) {
        SmallVector<uint32_t, 16> Elts;
        for (unsigned i = 0, e = V.size(); i != e; ++i)
          if (ConstantFP *CFP = dyn_cast<ConstantFP>(V[i]))
            Elts.push_back(
                CFP->getValueAPF().bitcastToAPInt().getLimitedValue());
          else
            break;
        if (Elts.size() == V.size())
          return ConstantDataArray::getFP(C->getContext(), Elts);
      } else if (CFP->getType()->isDoubleTy()) {
        SmallVector<uint64_t, 16> Elts;
        for (unsigned i = 0, e = V.size(); i != e; ++i)
          if (ConstantFP *CFP = dyn_cast<ConstantFP>(V[i]))
            Elts.push_back(
                CFP->getValueAPF().bitcastToAPInt().getLimitedValue());
          else
            break;
        if (Elts.size() == V.size())
          return ConstantDataArray::getFP(C->getContext(), Elts);
      }
    }
  }

  // Otherwise, we really do want to create a ConstantArray.
  return nullptr;
}

/// getTypeForElements - Return an anonymous struct type to use for a constant
/// with the specified set of elements.  The list must not be empty.
StructType *ConstantStruct::getTypeForElements(LLVMContext &Context,
                                               ArrayRef<Constant*> V,
                                               bool Packed) {
  unsigned VecSize = V.size();
  SmallVector<Type*, 16> EltTypes(VecSize);
  for (unsigned i = 0; i != VecSize; ++i)
    EltTypes[i] = V[i]->getType();

  return StructType::get(Context, EltTypes, Packed);
}


StructType *ConstantStruct::getTypeForElements(ArrayRef<Constant*> V,
                                               bool Packed) {
  assert(!V.empty() &&
         "ConstantStruct::getTypeForElements cannot be called on empty list");
  return getTypeForElements(V[0]->getContext(), V, Packed);
}


ConstantStruct::ConstantStruct(StructType *T, ArrayRef<Constant *> V)
  : Constant(T, ConstantStructVal,
             OperandTraits<ConstantStruct>::op_end(this) - V.size(),
             V.size()) {
  assert(V.size() == T->getNumElements() &&
         "Invalid initializer vector for constant structure");
  for (unsigned i = 0, e = V.size(); i != e; ++i)
    assert((T->isOpaque() || V[i]->getType() == T->getElementType(i)) &&
           "Initializer for struct element doesn't match struct element type!");
  std::copy(V.begin(), V.end(), op_begin());
}

// ConstantStruct accessors.
Constant *ConstantStruct::get(StructType *ST, ArrayRef<Constant*> V) {
  assert((ST->isOpaque() || ST->getNumElements() == V.size()) &&
         "Incorrect # elements specified to ConstantStruct::get");

  // Create a ConstantAggregateZero value if all elements are zeros.
  bool isZero = true;
  bool isUndef = false;
  
  if (!V.empty()) {
    isUndef = isa<UndefValue>(V[0]);
    isZero = V[0]->isNullValue();
    if (isUndef || isZero) {
      for (unsigned i = 0, e = V.size(); i != e; ++i) {
        if (!V[i]->isNullValue())
          isZero = false;
        if (!isa<UndefValue>(V[i]))
          isUndef = false;
      }
    }
  }
  if (isZero)
    return ConstantAggregateZero::get(ST);
  if (isUndef)
    return UndefValue::get(ST);

  return ST->getContext().pImpl->StructConstants.getOrCreate(ST, V);
}

Constant *ConstantStruct::get(StructType *T, ...) {
  va_list ap;
  SmallVector<Constant*, 8> Values;
  va_start(ap, T);
  while (Constant *Val = va_arg(ap, llvm::Constant*))
    Values.push_back(Val);
  va_end(ap);
  return get(T, Values);
}

ConstantVector::ConstantVector(VectorType *T, ArrayRef<Constant *> V)
  : Constant(T, ConstantVectorVal,
             OperandTraits<ConstantVector>::op_end(this) - V.size(),
             V.size()) {
  for (size_t i = 0, e = V.size(); i != e; i++)
    assert(V[i]->getType() == T->getElementType() &&
           "Initializer for vector element doesn't match vector element type!");
  std::copy(V.begin(), V.end(), op_begin());
}

// ConstantVector accessors.
Constant *ConstantVector::get(ArrayRef<Constant*> V) {
  if (Constant *C = getImpl(V))
    return C;
  VectorType *Ty = VectorType::get(V.front()->getType(), V.size());
  return Ty->getContext().pImpl->VectorConstants.getOrCreate(Ty, V);
}
Constant *ConstantVector::getImpl(ArrayRef<Constant*> V) {
  assert(!V.empty() && "Vectors can't be empty");
  VectorType *T = VectorType::get(V.front()->getType(), V.size());

  // If this is an all-undef or all-zero vector, return a
  // ConstantAggregateZero or UndefValue.
  Constant *C = V[0];
  bool isZero = C->isNullValue();
  bool isUndef = isa<UndefValue>(C);

  if (isZero || isUndef) {
    for (unsigned i = 1, e = V.size(); i != e; ++i)
      if (V[i] != C) {
        isZero = isUndef = false;
        break;
      }
  }

  if (isZero)
    return ConstantAggregateZero::get(T);
  if (isUndef)
    return UndefValue::get(T);

  // Check to see if all of the elements are ConstantFP or ConstantInt and if
  // the element type is compatible with ConstantDataVector.  If so, use it.
  if (ConstantDataSequential::isElementTypeCompatible(C->getType())) {
    // We speculatively build the elements here even if it turns out that there
    // is a constantexpr or something else weird in the array, since it is so
    // uncommon for that to happen.
    if (ConstantInt *CI = dyn_cast<ConstantInt>(C)) {
      if (CI->getType()->isIntegerTy(8)) {
        SmallVector<uint8_t, 16> Elts;
        for (unsigned i = 0, e = V.size(); i != e; ++i)
          if (ConstantInt *CI = dyn_cast<ConstantInt>(V[i]))
            Elts.push_back(CI->getZExtValue());
          else
            break;
        if (Elts.size() == V.size())
          return ConstantDataVector::get(C->getContext(), Elts);
      } else if (CI->getType()->isIntegerTy(16)) {
        SmallVector<uint16_t, 16> Elts;
        for (unsigned i = 0, e = V.size(); i != e; ++i)
          if (ConstantInt *CI = dyn_cast<ConstantInt>(V[i]))
            Elts.push_back(CI->getZExtValue());
          else
            break;
        if (Elts.size() == V.size())
          return ConstantDataVector::get(C->getContext(), Elts);
      } else if (CI->getType()->isIntegerTy(32)) {
        SmallVector<uint32_t, 16> Elts;
        for (unsigned i = 0, e = V.size(); i != e; ++i)
          if (ConstantInt *CI = dyn_cast<ConstantInt>(V[i]))
            Elts.push_back(CI->getZExtValue());
          else
            break;
        if (Elts.size() == V.size())
          return ConstantDataVector::get(C->getContext(), Elts);
      } else if (CI->getType()->isIntegerTy(64)) {
        SmallVector<uint64_t, 16> Elts;
        for (unsigned i = 0, e = V.size(); i != e; ++i)
          if (ConstantInt *CI = dyn_cast<ConstantInt>(V[i]))
            Elts.push_back(CI->getZExtValue());
          else
            break;
        if (Elts.size() == V.size())
          return ConstantDataVector::get(C->getContext(), Elts);
      }
    }

    if (ConstantFP *CFP = dyn_cast<ConstantFP>(C)) {
      if (CFP->getType()->isFloatTy()) {
        SmallVector<uint32_t, 16> Elts;
        for (unsigned i = 0, e = V.size(); i != e; ++i)
          if (ConstantFP *CFP = dyn_cast<ConstantFP>(V[i]))
            Elts.push_back(
                CFP->getValueAPF().bitcastToAPInt().getLimitedValue());
          else
            break;
        if (Elts.size() == V.size())
          return ConstantDataVector::getFP(C->getContext(), Elts);
      } else if (CFP->getType()->isDoubleTy()) {
        SmallVector<uint64_t, 16> Elts;
        for (unsigned i = 0, e = V.size(); i != e; ++i)
          if (ConstantFP *CFP = dyn_cast<ConstantFP>(V[i]))
            Elts.push_back(
                CFP->getValueAPF().bitcastToAPInt().getLimitedValue());
          else
            break;
        if (Elts.size() == V.size())
          return ConstantDataVector::getFP(C->getContext(), Elts);
      }
    }
  }

  // Otherwise, the element type isn't compatible with ConstantDataVector, or
  // the operand list constants a ConstantExpr or something else strange.
  return nullptr;
}

Constant *ConstantVector::getSplat(unsigned NumElts, Constant *V) {
  // If this splat is compatible with ConstantDataVector, use it instead of
  // ConstantVector.
  if ((isa<ConstantFP>(V) || isa<ConstantInt>(V)) &&
      ConstantDataSequential::isElementTypeCompatible(V->getType()))
    return ConstantDataVector::getSplat(NumElts, V);

  SmallVector<Constant*, 32> Elts(NumElts, V);
  return get(Elts);
}


// Utility function for determining if a ConstantExpr is a CastOp or not. This
// can't be inline because we don't want to #include Instruction.h into
// Constant.h
bool ConstantExpr::isCast() const {
  return Instruction::isCast(getOpcode());
}

bool ConstantExpr::isCompare() const {
  return getOpcode() == Instruction::ICmp || getOpcode() == Instruction::FCmp;
}

bool ConstantExpr::isGEPWithNoNotionalOverIndexing() const {
  if (getOpcode() != Instruction::GetElementPtr) return false;

  gep_type_iterator GEPI = gep_type_begin(this), E = gep_type_end(this);
  User::const_op_iterator OI = std::next(this->op_begin());

  // Skip the first index, as it has no static limit.
  ++GEPI;
  ++OI;

  // The remaining indices must be compile-time known integers within the
  // bounds of the corresponding notional static array types.
  for (; GEPI != E; ++GEPI, ++OI) {
    ConstantInt *CI = dyn_cast<ConstantInt>(*OI);
    if (!CI) return false;
    if (ArrayType *ATy = dyn_cast<ArrayType>(*GEPI))
      if (CI->getValue().getActiveBits() > 64 ||
          CI->getZExtValue() >= ATy->getNumElements())
        return false;
  }

  // All the indices checked out.
  return true;
}

bool ConstantExpr::hasIndices() const {
  return getOpcode() == Instruction::ExtractValue ||
         getOpcode() == Instruction::InsertValue;
}

ArrayRef<unsigned> ConstantExpr::getIndices() const {
  if (const ExtractValueConstantExpr *EVCE =
        dyn_cast<ExtractValueConstantExpr>(this))
    return EVCE->Indices;

  return cast<InsertValueConstantExpr>(this)->Indices;
}

unsigned ConstantExpr::getPredicate() const {
  assert(isCompare());
  return ((const CompareConstantExpr*)this)->predicate;
}

/// getWithOperandReplaced - Return a constant expression identical to this
/// one, but with the specified operand set to the specified value.
Constant *
ConstantExpr::getWithOperandReplaced(unsigned OpNo, Constant *Op) const {
  assert(Op->getType() == getOperand(OpNo)->getType() &&
         "Replacing operand with value of different type!");
  if (getOperand(OpNo) == Op)
    return const_cast<ConstantExpr*>(this);

  SmallVector<Constant*, 8> NewOps;
  for (unsigned i = 0, e = getNumOperands(); i != e; ++i)
    NewOps.push_back(i == OpNo ? Op : getOperand(i));

  return getWithOperands(NewOps);
}

/// getWithOperands - This returns the current constant expression with the
/// operands replaced with the specified values.  The specified array must
/// have the same number of operands as our current one.
Constant *ConstantExpr::getWithOperands(ArrayRef<Constant *> Ops, Type *Ty,
                                        bool OnlyIfReduced) const {
  assert(Ops.size() == getNumOperands() && "Operand count mismatch!");

  // If no operands changed return self.
  if (Ty == getType() && std::equal(Ops.begin(), Ops.end(), op_begin()))
    return const_cast<ConstantExpr*>(this);

  Type *OnlyIfReducedTy = OnlyIfReduced ? Ty : nullptr;
  switch (getOpcode()) {
  case Instruction::Trunc:
  case Instruction::ZExt:
  case Instruction::SExt:
  case Instruction::FPTrunc:
  case Instruction::FPExt:
  case Instruction::UIToFP:
  case Instruction::SIToFP:
  case Instruction::FPToUI:
  case Instruction::FPToSI:
  case Instruction::PtrToInt:
  case Instruction::IntToPtr:
  case Instruction::BitCast:
  case Instruction::AddrSpaceCast:
    return ConstantExpr::getCast(getOpcode(), Ops[0], Ty, OnlyIfReduced);
  case Instruction::Select:
    return ConstantExpr::getSelect(Ops[0], Ops[1], Ops[2], OnlyIfReducedTy);
  case Instruction::InsertElement:
    return ConstantExpr::getInsertElement(Ops[0], Ops[1], Ops[2],
                                          OnlyIfReducedTy);
  case Instruction::ExtractElement:
    return ConstantExpr::getExtractElement(Ops[0], Ops[1], OnlyIfReducedTy);
  case Instruction::InsertValue:
    return ConstantExpr::getInsertValue(Ops[0], Ops[1], getIndices(),
                                        OnlyIfReducedTy);
  case Instruction::ExtractValue:
    return ConstantExpr::getExtractValue(Ops[0], getIndices(), OnlyIfReducedTy);
  case Instruction::ShuffleVector:
    return ConstantExpr::getShuffleVector(Ops[0], Ops[1], Ops[2],
                                          OnlyIfReducedTy);
  case Instruction::GetElementPtr:
    return ConstantExpr::getGetElementPtr(nullptr, Ops[0], Ops.slice(1),
                                          cast<GEPOperator>(this)->isInBounds(),
                                          OnlyIfReducedTy);
  case Instruction::ICmp:
  case Instruction::FCmp:
    return ConstantExpr::getCompare(getPredicate(), Ops[0], Ops[1],
                                    OnlyIfReducedTy);
  default:
    assert(getNumOperands() == 2 && "Must be binary operator?");
    return ConstantExpr::get(getOpcode(), Ops[0], Ops[1], SubclassOptionalData,
                             OnlyIfReducedTy);
  }
}


//===----------------------------------------------------------------------===//
//                      isValueValidForType implementations

bool ConstantInt::isValueValidForType(Type *Ty, uint64_t Val) {
  unsigned NumBits = Ty->getIntegerBitWidth(); // assert okay
  if (Ty->isIntegerTy(1))
    return Val == 0 || Val == 1;
  if (NumBits >= 64)
    return true; // always true, has to fit in largest type
  uint64_t Max = (1ll << NumBits) - 1;
  return Val <= Max;
}

bool ConstantInt::isValueValidForType(Type *Ty, int64_t Val) {
  unsigned NumBits = Ty->getIntegerBitWidth();
  if (Ty->isIntegerTy(1))
    return Val == 0 || Val == 1 || Val == -1;
  if (NumBits >= 64)
    return true; // always true, has to fit in largest type
  int64_t Min = -(1ll << (NumBits-1));
  int64_t Max = (1ll << (NumBits-1)) - 1;
  return (Val >= Min && Val <= Max);
}

bool ConstantFP::isValueValidForType(Type *Ty, const APFloat& Val) {
  // convert modifies in place, so make a copy.
  APFloat Val2 = APFloat(Val);
  bool losesInfo;
  switch (Ty->getTypeID()) {
  default:
    return false;         // These can't be represented as floating point!

  // FIXME rounding mode needs to be more flexible
  case Type::HalfTyID: {
    if (&Val2.getSemantics() == &APFloat::IEEEhalf)
      return true;
    Val2.convert(APFloat::IEEEhalf, APFloat::rmNearestTiesToEven, &losesInfo);
    return !losesInfo;
  }
  case Type::FloatTyID: {
    if (&Val2.getSemantics() == &APFloat::IEEEsingle)
      return true;
    Val2.convert(APFloat::IEEEsingle, APFloat::rmNearestTiesToEven, &losesInfo);
    return !losesInfo;
  }
  case Type::DoubleTyID: {
    if (&Val2.getSemantics() == &APFloat::IEEEhalf ||
        &Val2.getSemantics() == &APFloat::IEEEsingle ||
        &Val2.getSemantics() == &APFloat::IEEEdouble)
      return true;
    Val2.convert(APFloat::IEEEdouble, APFloat::rmNearestTiesToEven, &losesInfo);
    return !losesInfo;
  }
  case Type::X86_FP80TyID:
    return &Val2.getSemantics() == &APFloat::IEEEhalf ||
           &Val2.getSemantics() == &APFloat::IEEEsingle || 
           &Val2.getSemantics() == &APFloat::IEEEdouble ||
           &Val2.getSemantics() == &APFloat::x87DoubleExtended;
  case Type::FP128TyID:
    return &Val2.getSemantics() == &APFloat::IEEEhalf ||
           &Val2.getSemantics() == &APFloat::IEEEsingle || 
           &Val2.getSemantics() == &APFloat::IEEEdouble ||
           &Val2.getSemantics() == &APFloat::IEEEquad;
  case Type::PPC_FP128TyID:
    return &Val2.getSemantics() == &APFloat::IEEEhalf ||
           &Val2.getSemantics() == &APFloat::IEEEsingle || 
           &Val2.getSemantics() == &APFloat::IEEEdouble ||
           &Val2.getSemantics() == &APFloat::PPCDoubleDouble;
  }
}


//===----------------------------------------------------------------------===//
//                      Factory Function Implementation

ConstantAggregateZero *ConstantAggregateZero::get(Type *Ty) {
  assert((Ty->isStructTy() || Ty->isArrayTy() || Ty->isVectorTy()) &&
         "Cannot create an aggregate zero of non-aggregate type!");
  
  ConstantAggregateZero *&Entry = Ty->getContext().pImpl->CAZConstants[Ty];
  if (!Entry)
    Entry = new ConstantAggregateZero(Ty);

  return Entry;
}

/// destroyConstant - Remove the constant from the constant table.
///
void ConstantAggregateZero::destroyConstantImpl() {
  getContext().pImpl->CAZConstants.erase(getType());
}

/// destroyConstant - Remove the constant from the constant table...
///
void ConstantArray::destroyConstantImpl() {
  getType()->getContext().pImpl->ArrayConstants.remove(this);
}


//---- ConstantStruct::get() implementation...
//

// destroyConstant - Remove the constant from the constant table...
//
void ConstantStruct::destroyConstantImpl() {
  getType()->getContext().pImpl->StructConstants.remove(this);
}

// destroyConstant - Remove the constant from the constant table...
//
void ConstantVector::destroyConstantImpl() {
  getType()->getContext().pImpl->VectorConstants.remove(this);
}

/// getSplatValue - If this is a splat vector constant, meaning that all of
/// the elements have the same value, return that value. Otherwise return 0.
Constant *Constant::getSplatValue() const {
  assert(this->getType()->isVectorTy() && "Only valid for vectors!");
  if (isa<ConstantAggregateZero>(this))
    return getNullValue(this->getType()->getVectorElementType());
  if (const ConstantDataVector *CV = dyn_cast<ConstantDataVector>(this))
    return CV->getSplatValue();
  if (const ConstantVector *CV = dyn_cast<ConstantVector>(this))
    return CV->getSplatValue();
  return nullptr;
}

/// getSplatValue - If this is a splat constant, where all of the
/// elements have the same value, return that value. Otherwise return null.
Constant *ConstantVector::getSplatValue() const {
  // Check out first element.
  Constant *Elt = getOperand(0);
  // Then make sure all remaining elements point to the same value.
  for (unsigned I = 1, E = getNumOperands(); I < E; ++I)
    if (getOperand(I) != Elt)
      return nullptr;
  return Elt;
}

/// If C is a constant integer then return its value, otherwise C must be a
/// vector of constant integers, all equal, and the common value is returned.
const APInt &Constant::getUniqueInteger() const {
  if (const ConstantInt *CI = dyn_cast<ConstantInt>(this))
    return CI->getValue();
  assert(this->getSplatValue() && "Doesn't contain a unique integer!");
  const Constant *C = this->getAggregateElement(0U);
  assert(C && isa<ConstantInt>(C) && "Not a vector of numbers!");
  return cast<ConstantInt>(C)->getValue();
}

//---- ConstantPointerNull::get() implementation.
//

ConstantPointerNull *ConstantPointerNull::get(PointerType *Ty) {
  ConstantPointerNull *&Entry = Ty->getContext().pImpl->CPNConstants[Ty];
  if (!Entry)
    Entry = new ConstantPointerNull(Ty);

  return Entry;
}

// destroyConstant - Remove the constant from the constant table...
//
void ConstantPointerNull::destroyConstantImpl() {
  getContext().pImpl->CPNConstants.erase(getType());
}


//---- UndefValue::get() implementation.
//

UndefValue *UndefValue::get(Type *Ty) {
  UndefValue *&Entry = Ty->getContext().pImpl->UVConstants[Ty];
  if (!Entry)
    Entry = new UndefValue(Ty);

  return Entry;
}

// destroyConstant - Remove the constant from the constant table.
//
void UndefValue::destroyConstantImpl() {
  // Free the constant and any dangling references to it.
  getContext().pImpl->UVConstants.erase(getType());
}

//---- BlockAddress::get() implementation.
//

BlockAddress *BlockAddress::get(BasicBlock *BB) {
  assert(BB->getParent() && "Block must have a parent");
  return get(BB->getParent(), BB);
}

BlockAddress *BlockAddress::get(Function *F, BasicBlock *BB) {
  BlockAddress *&BA =
    F->getContext().pImpl->BlockAddresses[std::make_pair(F, BB)];
  if (!BA)
    BA = new BlockAddress(F, BB);

  assert(BA->getFunction() == F && "Basic block moved between functions");
  return BA;
}

BlockAddress::BlockAddress(Function *F, BasicBlock *BB)
: Constant(Type::getInt8PtrTy(F->getContext()), Value::BlockAddressVal,
           &Op<0>(), 2) {
  setOperand(0, F);
  setOperand(1, BB);
  BB->AdjustBlockAddressRefCount(1);
}

BlockAddress *BlockAddress::lookup(const BasicBlock *BB) {
  if (!BB->hasAddressTaken())
    return nullptr;

  const Function *F = BB->getParent();
  assert(F && "Block must have a parent");
  BlockAddress *BA =
      F->getContext().pImpl->BlockAddresses.lookup(std::make_pair(F, BB));
  assert(BA && "Refcount and block address map disagree!");
  return BA;
}

// destroyConstant - Remove the constant from the constant table.
//
void BlockAddress::destroyConstantImpl() {
  getFunction()->getType()->getContext().pImpl
    ->BlockAddresses.erase(std::make_pair(getFunction(), getBasicBlock()));
  getBasicBlock()->AdjustBlockAddressRefCount(-1);
}

Value *BlockAddress::handleOperandChangeImpl(Value *From, Value *To, Use *U) {
  // This could be replacing either the Basic Block or the Function.  In either
  // case, we have to remove the map entry.
  Function *NewF = getFunction();
  BasicBlock *NewBB = getBasicBlock();

  if (U == &Op<0>())
    NewF = cast<Function>(To->stripPointerCasts());
  else
    NewBB = cast<BasicBlock>(To);

  // See if the 'new' entry already exists, if not, just update this in place
  // and return early.
  BlockAddress *&NewBA =
    getContext().pImpl->BlockAddresses[std::make_pair(NewF, NewBB)];
  if (NewBA)
    return NewBA;

  getBasicBlock()->AdjustBlockAddressRefCount(-1);

  // Remove the old entry, this can't cause the map to rehash (just a
  // tombstone will get added).
  getContext().pImpl->BlockAddresses.erase(std::make_pair(getFunction(),
                                                          getBasicBlock()));
  NewBA = this;
  setOperand(0, NewF);
  setOperand(1, NewBB);
  getBasicBlock()->AdjustBlockAddressRefCount(1);

  // If we just want to keep the existing value, then return null.
  // Callers know that this means we shouldn't delete this value.
  return nullptr;
}

//---- ConstantExpr::get() implementations.
//

/// This is a utility function to handle folding of casts and lookup of the
/// cast in the ExprConstants map. It is used by the various get* methods below.
static Constant *getFoldedCast(Instruction::CastOps opc, Constant *C, Type *Ty,
                               bool OnlyIfReduced = false) {
  assert(Ty->isFirstClassType() && "Cannot cast to an aggregate type!");
  // Fold a few common cases
  if (Constant *FC = ConstantFoldCastInstruction(opc, C, Ty))
    return FC;

  if (OnlyIfReduced)
    return nullptr;

  LLVMContextImpl *pImpl = Ty->getContext().pImpl;

  // Look up the constant in the table first to ensure uniqueness.
  ConstantExprKeyType Key(opc, C);

  return pImpl->ExprConstants.getOrCreate(Ty, Key);
}

Constant *ConstantExpr::getCast(unsigned oc, Constant *C, Type *Ty,
                                bool OnlyIfReduced) {
  Instruction::CastOps opc = Instruction::CastOps(oc);
  assert(Instruction::isCast(opc) && "opcode out of range");
  assert(C && Ty && "Null arguments to getCast");
  assert(CastInst::castIsValid(opc, C, Ty) && "Invalid constantexpr cast!");

  switch (opc) {
  default:
    llvm_unreachable("Invalid cast opcode");
  case Instruction::Trunc:
    return getTrunc(C, Ty, OnlyIfReduced);
  case Instruction::ZExt:
    return getZExt(C, Ty, OnlyIfReduced);
  case Instruction::SExt:
    return getSExt(C, Ty, OnlyIfReduced);
  case Instruction::FPTrunc:
    return getFPTrunc(C, Ty, OnlyIfReduced);
  case Instruction::FPExt:
    return getFPExtend(C, Ty, OnlyIfReduced);
  case Instruction::UIToFP:
    return getUIToFP(C, Ty, OnlyIfReduced);
  case Instruction::SIToFP:
    return getSIToFP(C, Ty, OnlyIfReduced);
  case Instruction::FPToUI:
    return getFPToUI(C, Ty, OnlyIfReduced);
  case Instruction::FPToSI:
    return getFPToSI(C, Ty, OnlyIfReduced);
  case Instruction::PtrToInt:
    return getPtrToInt(C, Ty, OnlyIfReduced);
  case Instruction::IntToPtr:
    return getIntToPtr(C, Ty, OnlyIfReduced);
  case Instruction::BitCast:
    return getBitCast(C, Ty, OnlyIfReduced);
  case Instruction::AddrSpaceCast:
    return getAddrSpaceCast(C, Ty, OnlyIfReduced);
  }
}

Constant *ConstantExpr::getZExtOrBitCast(Constant *C, Type *Ty) {
  if (C->getType()->getScalarSizeInBits() == Ty->getScalarSizeInBits())
    return getBitCast(C, Ty);
  return getZExt(C, Ty);
}

Constant *ConstantExpr::getSExtOrBitCast(Constant *C, Type *Ty) {
  if (C->getType()->getScalarSizeInBits() == Ty->getScalarSizeInBits())
    return getBitCast(C, Ty);
  return getSExt(C, Ty);
}

Constant *ConstantExpr::getTruncOrBitCast(Constant *C, Type *Ty) {
  if (C->getType()->getScalarSizeInBits() == Ty->getScalarSizeInBits())
    return getBitCast(C, Ty);
  return getTrunc(C, Ty);
}

Constant *ConstantExpr::getPointerCast(Constant *S, Type *Ty) {
  assert(S->getType()->isPtrOrPtrVectorTy() && "Invalid cast");
  assert((Ty->isIntOrIntVectorTy() || Ty->isPtrOrPtrVectorTy()) &&
          "Invalid cast");

  if (Ty->isIntOrIntVectorTy())
    return getPtrToInt(S, Ty);

  unsigned SrcAS = S->getType()->getPointerAddressSpace();
  if (Ty->isPtrOrPtrVectorTy() && SrcAS != Ty->getPointerAddressSpace())
    return getAddrSpaceCast(S, Ty);

  return getBitCast(S, Ty);
}

Constant *ConstantExpr::getPointerBitCastOrAddrSpaceCast(Constant *S,
                                                         Type *Ty) {
  assert(S->getType()->isPtrOrPtrVectorTy() && "Invalid cast");
  assert(Ty->isPtrOrPtrVectorTy() && "Invalid cast");

  if (S->getType()->getPointerAddressSpace() != Ty->getPointerAddressSpace())
    return getAddrSpaceCast(S, Ty);

  return getBitCast(S, Ty);
}

Constant *ConstantExpr::getIntegerCast(Constant *C, Type *Ty,
                                       bool isSigned) {
  assert(C->getType()->isIntOrIntVectorTy() &&
         Ty->isIntOrIntVectorTy() && "Invalid cast");
  unsigned SrcBits = C->getType()->getScalarSizeInBits();
  unsigned DstBits = Ty->getScalarSizeInBits();
  Instruction::CastOps opcode =
    (SrcBits == DstBits ? Instruction::BitCast :
     (SrcBits > DstBits ? Instruction::Trunc :
      (isSigned ? Instruction::SExt : Instruction::ZExt)));
  return getCast(opcode, C, Ty);
}

Constant *ConstantExpr::getFPCast(Constant *C, Type *Ty) {
  assert(C->getType()->isFPOrFPVectorTy() && Ty->isFPOrFPVectorTy() &&
         "Invalid cast");
  unsigned SrcBits = C->getType()->getScalarSizeInBits();
  unsigned DstBits = Ty->getScalarSizeInBits();
  if (SrcBits == DstBits)
    return C; // Avoid a useless cast
  Instruction::CastOps opcode =
    (SrcBits > DstBits ? Instruction::FPTrunc : Instruction::FPExt);
  return getCast(opcode, C, Ty);
}

Constant *ConstantExpr::getTrunc(Constant *C, Type *Ty, bool OnlyIfReduced) {
#ifndef NDEBUG
  bool fromVec = C->getType()->getTypeID() == Type::VectorTyID;
  bool toVec = Ty->getTypeID() == Type::VectorTyID;
#endif
  assert((fromVec == toVec) && "Cannot convert from scalar to/from vector");
  assert(C->getType()->isIntOrIntVectorTy() && "Trunc operand must be integer");
  assert(Ty->isIntOrIntVectorTy() && "Trunc produces only integral");
  assert(C->getType()->getScalarSizeInBits() > Ty->getScalarSizeInBits()&&
         "SrcTy must be larger than DestTy for Trunc!");

  return getFoldedCast(Instruction::Trunc, C, Ty, OnlyIfReduced);
}

Constant *ConstantExpr::getSExt(Constant *C, Type *Ty, bool OnlyIfReduced) {
#ifndef NDEBUG
  bool fromVec = C->getType()->getTypeID() == Type::VectorTyID;
  bool toVec = Ty->getTypeID() == Type::VectorTyID;
#endif
  assert((fromVec == toVec) && "Cannot convert from scalar to/from vector");
  assert(C->getType()->isIntOrIntVectorTy() && "SExt operand must be integral");
  assert(Ty->isIntOrIntVectorTy() && "SExt produces only integer");
  assert(C->getType()->getScalarSizeInBits() < Ty->getScalarSizeInBits()&&
         "SrcTy must be smaller than DestTy for SExt!");

  return getFoldedCast(Instruction::SExt, C, Ty, OnlyIfReduced);
}

Constant *ConstantExpr::getZExt(Constant *C, Type *Ty, bool OnlyIfReduced) {
#ifndef NDEBUG
  bool fromVec = C->getType()->getTypeID() == Type::VectorTyID;
  bool toVec = Ty->getTypeID() == Type::VectorTyID;
#endif
  assert((fromVec == toVec) && "Cannot convert from scalar to/from vector");
  assert(C->getType()->isIntOrIntVectorTy() && "ZEXt operand must be integral");
  assert(Ty->isIntOrIntVectorTy() && "ZExt produces only integer");
  assert(C->getType()->getScalarSizeInBits() < Ty->getScalarSizeInBits()&&
         "SrcTy must be smaller than DestTy for ZExt!");

  return getFoldedCast(Instruction::ZExt, C, Ty, OnlyIfReduced);
}

Constant *ConstantExpr::getFPTrunc(Constant *C, Type *Ty, bool OnlyIfReduced) {
#ifndef NDEBUG
  bool fromVec = C->getType()->getTypeID() == Type::VectorTyID;
  bool toVec = Ty->getTypeID() == Type::VectorTyID;
#endif
  assert((fromVec == toVec) && "Cannot convert from scalar to/from vector");
  assert(C->getType()->isFPOrFPVectorTy() && Ty->isFPOrFPVectorTy() &&
         C->getType()->getScalarSizeInBits() > Ty->getScalarSizeInBits()&&
         "This is an illegal floating point truncation!");
  return getFoldedCast(Instruction::FPTrunc, C, Ty, OnlyIfReduced);
}

Constant *ConstantExpr::getFPExtend(Constant *C, Type *Ty, bool OnlyIfReduced) {
#ifndef NDEBUG
  bool fromVec = C->getType()->getTypeID() == Type::VectorTyID;
  bool toVec = Ty->getTypeID() == Type::VectorTyID;
#endif
  assert((fromVec == toVec) && "Cannot convert from scalar to/from vector");
  assert(C->getType()->isFPOrFPVectorTy() && Ty->isFPOrFPVectorTy() &&
         C->getType()->getScalarSizeInBits() < Ty->getScalarSizeInBits()&&
         "This is an illegal floating point extension!");
  return getFoldedCast(Instruction::FPExt, C, Ty, OnlyIfReduced);
}

Constant *ConstantExpr::getUIToFP(Constant *C, Type *Ty, bool OnlyIfReduced) {
#ifndef NDEBUG
  bool fromVec = C->getType()->getTypeID() == Type::VectorTyID;
  bool toVec = Ty->getTypeID() == Type::VectorTyID;
#endif
  assert((fromVec == toVec) && "Cannot convert from scalar to/from vector");
  assert(C->getType()->isIntOrIntVectorTy() && Ty->isFPOrFPVectorTy() &&
         "This is an illegal uint to floating point cast!");
  return getFoldedCast(Instruction::UIToFP, C, Ty, OnlyIfReduced);
}

Constant *ConstantExpr::getSIToFP(Constant *C, Type *Ty, bool OnlyIfReduced) {
#ifndef NDEBUG
  bool fromVec = C->getType()->getTypeID() == Type::VectorTyID;
  bool toVec = Ty->getTypeID() == Type::VectorTyID;
#endif
  assert((fromVec == toVec) && "Cannot convert from scalar to/from vector");
  assert(C->getType()->isIntOrIntVectorTy() && Ty->isFPOrFPVectorTy() &&
         "This is an illegal sint to floating point cast!");
  return getFoldedCast(Instruction::SIToFP, C, Ty, OnlyIfReduced);
}

Constant *ConstantExpr::getFPToUI(Constant *C, Type *Ty, bool OnlyIfReduced) {
#ifndef NDEBUG
  bool fromVec = C->getType()->getTypeID() == Type::VectorTyID;
  bool toVec = Ty->getTypeID() == Type::VectorTyID;
#endif
  assert((fromVec == toVec) && "Cannot convert from scalar to/from vector");
  assert(C->getType()->isFPOrFPVectorTy() && Ty->isIntOrIntVectorTy() &&
         "This is an illegal floating point to uint cast!");
  return getFoldedCast(Instruction::FPToUI, C, Ty, OnlyIfReduced);
}

Constant *ConstantExpr::getFPToSI(Constant *C, Type *Ty, bool OnlyIfReduced) {
#ifndef NDEBUG
  bool fromVec = C->getType()->getTypeID() == Type::VectorTyID;
  bool toVec = Ty->getTypeID() == Type::VectorTyID;
#endif
  assert((fromVec == toVec) && "Cannot convert from scalar to/from vector");
  assert(C->getType()->isFPOrFPVectorTy() && Ty->isIntOrIntVectorTy() &&
         "This is an illegal floating point to sint cast!");
  return getFoldedCast(Instruction::FPToSI, C, Ty, OnlyIfReduced);
}

Constant *ConstantExpr::getPtrToInt(Constant *C, Type *DstTy,
                                    bool OnlyIfReduced) {
  assert(C->getType()->getScalarType()->isPointerTy() &&
         "PtrToInt source must be pointer or pointer vector");
  assert(DstTy->getScalarType()->isIntegerTy() && 
         "PtrToInt destination must be integer or integer vector");
  assert(isa<VectorType>(C->getType()) == isa<VectorType>(DstTy));
  if (isa<VectorType>(C->getType()))
    assert(C->getType()->getVectorNumElements()==DstTy->getVectorNumElements()&&
           "Invalid cast between a different number of vector elements");
  return getFoldedCast(Instruction::PtrToInt, C, DstTy, OnlyIfReduced);
}

Constant *ConstantExpr::getIntToPtr(Constant *C, Type *DstTy,
                                    bool OnlyIfReduced) {
  assert(C->getType()->getScalarType()->isIntegerTy() &&
         "IntToPtr source must be integer or integer vector");
  assert(DstTy->getScalarType()->isPointerTy() &&
         "IntToPtr destination must be a pointer or pointer vector");
  assert(isa<VectorType>(C->getType()) == isa<VectorType>(DstTy));
  if (isa<VectorType>(C->getType()))
    assert(C->getType()->getVectorNumElements()==DstTy->getVectorNumElements()&&
           "Invalid cast between a different number of vector elements");
  return getFoldedCast(Instruction::IntToPtr, C, DstTy, OnlyIfReduced);
}

Constant *ConstantExpr::getBitCast(Constant *C, Type *DstTy,
                                   bool OnlyIfReduced) {
  assert(CastInst::castIsValid(Instruction::BitCast, C, DstTy) &&
         "Invalid constantexpr bitcast!");

  // It is common to ask for a bitcast of a value to its own type, handle this
  // speedily.
  if (C->getType() == DstTy) return C;

  return getFoldedCast(Instruction::BitCast, C, DstTy, OnlyIfReduced);
}

Constant *ConstantExpr::getAddrSpaceCast(Constant *C, Type *DstTy,
                                         bool OnlyIfReduced) {
  assert(CastInst::castIsValid(Instruction::AddrSpaceCast, C, DstTy) &&
         "Invalid constantexpr addrspacecast!");

  // Canonicalize addrspacecasts between different pointer types by first
  // bitcasting the pointer type and then converting the address space.
  PointerType *SrcScalarTy = cast<PointerType>(C->getType()->getScalarType());
  PointerType *DstScalarTy = cast<PointerType>(DstTy->getScalarType());
  Type *DstElemTy = DstScalarTy->getElementType();
  if (SrcScalarTy->getElementType() != DstElemTy) {
    Type *MidTy = PointerType::get(DstElemTy, SrcScalarTy->getAddressSpace());
    if (VectorType *VT = dyn_cast<VectorType>(DstTy)) {
      // Handle vectors of pointers.
      MidTy = VectorType::get(MidTy, VT->getNumElements());
    }
    C = getBitCast(C, MidTy);
  }
  return getFoldedCast(Instruction::AddrSpaceCast, C, DstTy, OnlyIfReduced);
}

Constant *ConstantExpr::get(unsigned Opcode, Constant *C1, Constant *C2,
                            unsigned Flags, Type *OnlyIfReducedTy) {
  // Check the operands for consistency first.
  assert(Opcode >= Instruction::BinaryOpsBegin &&
         Opcode <  Instruction::BinaryOpsEnd   &&
         "Invalid opcode in binary constant expression");
  assert(C1->getType() == C2->getType() &&
         "Operand types in binary constant expression should match");

#ifndef NDEBUG
  switch (Opcode) {
  case Instruction::Add:
  case Instruction::Sub:
  case Instruction::Mul:
    assert(C1->getType() == C2->getType() && "Op types should be identical!");
    assert(C1->getType()->isIntOrIntVectorTy() &&
           "Tried to create an integer operation on a non-integer type!");
    break;
  case Instruction::FAdd:
  case Instruction::FSub:
  case Instruction::FMul:
    assert(C1->getType() == C2->getType() && "Op types should be identical!");
    assert(C1->getType()->isFPOrFPVectorTy() &&
           "Tried to create a floating-point operation on a "
           "non-floating-point type!");
    break;
  case Instruction::UDiv: 
  case Instruction::SDiv: 
    assert(C1->getType() == C2->getType() && "Op types should be identical!");
    assert(C1->getType()->isIntOrIntVectorTy() &&
           "Tried to create an arithmetic operation on a non-arithmetic type!");
    break;
  case Instruction::FDiv:
    assert(C1->getType() == C2->getType() && "Op types should be identical!");
    assert(C1->getType()->isFPOrFPVectorTy() &&
           "Tried to create an arithmetic operation on a non-arithmetic type!");
    break;
  case Instruction::URem: 
  case Instruction::SRem: 
    assert(C1->getType() == C2->getType() && "Op types should be identical!");
    assert(C1->getType()->isIntOrIntVectorTy() &&
           "Tried to create an arithmetic operation on a non-arithmetic type!");
    break;
  case Instruction::FRem:
    assert(C1->getType() == C2->getType() && "Op types should be identical!");
    assert(C1->getType()->isFPOrFPVectorTy() &&
           "Tried to create an arithmetic operation on a non-arithmetic type!");
    break;
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
    assert(C1->getType() == C2->getType() && "Op types should be identical!");
    assert(C1->getType()->isIntOrIntVectorTy() &&
           "Tried to create a logical operation on a non-integral type!");
    break;
  case Instruction::Shl:
  case Instruction::LShr:
  case Instruction::AShr:
    assert(C1->getType() == C2->getType() && "Op types should be identical!");
    assert(C1->getType()->isIntOrIntVectorTy() &&
           "Tried to create a shift operation on a non-integer type!");
    break;
  default:
    break;
  }
#endif

  if (Constant *FC = ConstantFoldBinaryInstruction(Opcode, C1, C2))
    return FC;          // Fold a few common cases.

  if (OnlyIfReducedTy == C1->getType())
    return nullptr;

  Constant *ArgVec[] = { C1, C2 };
  ConstantExprKeyType Key(Opcode, ArgVec, 0, Flags);

  LLVMContextImpl *pImpl = C1->getContext().pImpl;
  return pImpl->ExprConstants.getOrCreate(C1->getType(), Key);
}

Constant *ConstantExpr::getSizeOf(Type* Ty) {
  // sizeof is implemented as: (i64) gep (Ty*)null, 1
  // Note that a non-inbounds gep is used, as null isn't within any object.
  Constant *GEPIdx = ConstantInt::get(Type::getInt32Ty(Ty->getContext()), 1);
  Constant *GEP = getGetElementPtr(
      Ty, Constant::getNullValue(PointerType::getUnqual(Ty)), GEPIdx);
  return getPtrToInt(GEP, 
                     Type::getInt64Ty(Ty->getContext()));
}

Constant *ConstantExpr::getAlignOf(Type* Ty) {
  // alignof is implemented as: (i64) gep ({i1,Ty}*)null, 0, 1
  // Note that a non-inbounds gep is used, as null isn't within any object.
  Type *AligningTy = 
    StructType::get(Type::getInt1Ty(Ty->getContext()), Ty, nullptr);
  Constant *NullPtr = Constant::getNullValue(AligningTy->getPointerTo(0));
  Constant *Zero = ConstantInt::get(Type::getInt64Ty(Ty->getContext()), 0);
  Constant *One = ConstantInt::get(Type::getInt32Ty(Ty->getContext()), 1);
  Constant *Indices[2] = { Zero, One };
  Constant *GEP = getGetElementPtr(AligningTy, NullPtr, Indices);
  return getPtrToInt(GEP,
                     Type::getInt64Ty(Ty->getContext()));
}

Constant *ConstantExpr::getOffsetOf(StructType* STy, unsigned FieldNo) {
  return getOffsetOf(STy, ConstantInt::get(Type::getInt32Ty(STy->getContext()),
                                           FieldNo));
}

Constant *ConstantExpr::getOffsetOf(Type* Ty, Constant *FieldNo) {
  // offsetof is implemented as: (i64) gep (Ty*)null, 0, FieldNo
  // Note that a non-inbounds gep is used, as null isn't within any object.
  Constant *GEPIdx[] = {
    ConstantInt::get(Type::getInt64Ty(Ty->getContext()), 0),
    FieldNo
  };
  Constant *GEP = getGetElementPtr(
      Ty, Constant::getNullValue(PointerType::getUnqual(Ty)), GEPIdx);
  return getPtrToInt(GEP,
                     Type::getInt64Ty(Ty->getContext()));
}

Constant *ConstantExpr::getCompare(unsigned short Predicate, Constant *C1,
                                   Constant *C2, bool OnlyIfReduced) {
  assert(C1->getType() == C2->getType() && "Op types should be identical!");

  switch (Predicate) {
  default: llvm_unreachable("Invalid CmpInst predicate");
  case CmpInst::FCMP_FALSE: case CmpInst::FCMP_OEQ: case CmpInst::FCMP_OGT:
  case CmpInst::FCMP_OGE:   case CmpInst::FCMP_OLT: case CmpInst::FCMP_OLE:
  case CmpInst::FCMP_ONE:   case CmpInst::FCMP_ORD: case CmpInst::FCMP_UNO:
  case CmpInst::FCMP_UEQ:   case CmpInst::FCMP_UGT: case CmpInst::FCMP_UGE:
  case CmpInst::FCMP_ULT:   case CmpInst::FCMP_ULE: case CmpInst::FCMP_UNE:
  case CmpInst::FCMP_TRUE:
    return getFCmp(Predicate, C1, C2, OnlyIfReduced);

  case CmpInst::ICMP_EQ:  case CmpInst::ICMP_NE:  case CmpInst::ICMP_UGT:
  case CmpInst::ICMP_UGE: case CmpInst::ICMP_ULT: case CmpInst::ICMP_ULE:
  case CmpInst::ICMP_SGT: case CmpInst::ICMP_SGE: case CmpInst::ICMP_SLT:
  case CmpInst::ICMP_SLE:
    return getICmp(Predicate, C1, C2, OnlyIfReduced);
  }
}

Constant *ConstantExpr::getSelect(Constant *C, Constant *V1, Constant *V2,
                                  Type *OnlyIfReducedTy) {
  assert(!SelectInst::areInvalidOperands(C, V1, V2)&&"Invalid select operands");

  if (Constant *SC = ConstantFoldSelectInstruction(C, V1, V2))
    return SC;        // Fold common cases

  if (OnlyIfReducedTy == V1->getType())
    return nullptr;

  Constant *ArgVec[] = { C, V1, V2 };
  ConstantExprKeyType Key(Instruction::Select, ArgVec);

  LLVMContextImpl *pImpl = C->getContext().pImpl;
  return pImpl->ExprConstants.getOrCreate(V1->getType(), Key);
}

Constant *ConstantExpr::getGetElementPtr(Type *Ty, Constant *C,
                                         ArrayRef<Value *> Idxs, bool InBounds,
                                         Type *OnlyIfReducedTy) {
  if (!Ty)
    Ty = cast<PointerType>(C->getType()->getScalarType())->getElementType();
  else
    assert(
        Ty ==
        cast<PointerType>(C->getType()->getScalarType())->getContainedType(0u));

  if (Constant *FC = ConstantFoldGetElementPtr(Ty, C, InBounds, Idxs))
    return FC;          // Fold a few common cases.

  // Get the result type of the getelementptr!
  Type *DestTy = GetElementPtrInst::getIndexedType(Ty, Idxs);
  assert(DestTy && "GEP indices invalid!");
  unsigned AS = C->getType()->getPointerAddressSpace();
  Type *ReqTy = DestTy->getPointerTo(AS);
  if (VectorType *VecTy = dyn_cast<VectorType>(C->getType()))
    ReqTy = VectorType::get(ReqTy, VecTy->getNumElements());

  if (OnlyIfReducedTy == ReqTy)
    return nullptr;

  // Look up the constant in the table first to ensure uniqueness
  std::vector<Constant*> ArgVec;
  ArgVec.reserve(1 + Idxs.size());
  ArgVec.push_back(C);
  for (unsigned i = 0, e = Idxs.size(); i != e; ++i) {
    assert(Idxs[i]->getType()->isVectorTy() == ReqTy->isVectorTy() &&
           "getelementptr index type missmatch");
    assert((!Idxs[i]->getType()->isVectorTy() ||
            ReqTy->getVectorNumElements() ==
            Idxs[i]->getType()->getVectorNumElements()) &&
           "getelementptr index type missmatch");
    ArgVec.push_back(cast<Constant>(Idxs[i]));
  }
  const ConstantExprKeyType Key(Instruction::GetElementPtr, ArgVec, 0,
                                InBounds ? GEPOperator::IsInBounds : 0, None,
                                Ty);

  LLVMContextImpl *pImpl = C->getContext().pImpl;
  return pImpl->ExprConstants.getOrCreate(ReqTy, Key);
}

Constant *ConstantExpr::getICmp(unsigned short pred, Constant *LHS,
                                Constant *RHS, bool OnlyIfReduced) {
  assert(LHS->getType() == RHS->getType());
  assert(pred >= ICmpInst::FIRST_ICMP_PREDICATE && 
         pred <= ICmpInst::LAST_ICMP_PREDICATE && "Invalid ICmp Predicate");

  if (Constant *FC = ConstantFoldCompareInstruction(pred, LHS, RHS))
    return FC;          // Fold a few common cases...

  if (OnlyIfReduced)
    return nullptr;

  // Look up the constant in the table first to ensure uniqueness
  Constant *ArgVec[] = { LHS, RHS };
  // Get the key type with both the opcode and predicate
  const ConstantExprKeyType Key(Instruction::ICmp, ArgVec, pred);

  Type *ResultTy = Type::getInt1Ty(LHS->getContext());
  if (VectorType *VT = dyn_cast<VectorType>(LHS->getType()))
    ResultTy = VectorType::get(ResultTy, VT->getNumElements());

  LLVMContextImpl *pImpl = LHS->getType()->getContext().pImpl;
  return pImpl->ExprConstants.getOrCreate(ResultTy, Key);
}

Constant *ConstantExpr::getFCmp(unsigned short pred, Constant *LHS,
                                Constant *RHS, bool OnlyIfReduced) {
  assert(LHS->getType() == RHS->getType());
  assert(pred <= FCmpInst::LAST_FCMP_PREDICATE && "Invalid FCmp Predicate");

  if (Constant *FC = ConstantFoldCompareInstruction(pred, LHS, RHS))
    return FC;          // Fold a few common cases...

  if (OnlyIfReduced)
    return nullptr;

  // Look up the constant in the table first to ensure uniqueness
  Constant *ArgVec[] = { LHS, RHS };
  // Get the key type with both the opcode and predicate
  const ConstantExprKeyType Key(Instruction::FCmp, ArgVec, pred);

  Type *ResultTy = Type::getInt1Ty(LHS->getContext());
  if (VectorType *VT = dyn_cast<VectorType>(LHS->getType()))
    ResultTy = VectorType::get(ResultTy, VT->getNumElements());

  LLVMContextImpl *pImpl = LHS->getType()->getContext().pImpl;
  return pImpl->ExprConstants.getOrCreate(ResultTy, Key);
}

Constant *ConstantExpr::getExtractElement(Constant *Val, Constant *Idx,
                                          Type *OnlyIfReducedTy) {
  assert(Val->getType()->isVectorTy() &&
         "Tried to create extractelement operation on non-vector type!");
  assert(Idx->getType()->isIntegerTy() &&
         "Extractelement index must be an integer type!");

  if (Constant *FC = ConstantFoldExtractElementInstruction(Val, Idx))
    return FC;          // Fold a few common cases.

  Type *ReqTy = Val->getType()->getVectorElementType();
  if (OnlyIfReducedTy == ReqTy)
    return nullptr;

  // Look up the constant in the table first to ensure uniqueness
  Constant *ArgVec[] = { Val, Idx };
  const ConstantExprKeyType Key(Instruction::ExtractElement, ArgVec);

  LLVMContextImpl *pImpl = Val->getContext().pImpl;
  return pImpl->ExprConstants.getOrCreate(ReqTy, Key);
}

Constant *ConstantExpr::getInsertElement(Constant *Val, Constant *Elt,
                                         Constant *Idx, Type *OnlyIfReducedTy) {
  assert(Val->getType()->isVectorTy() &&
         "Tried to create insertelement operation on non-vector type!");
  assert(Elt->getType() == Val->getType()->getVectorElementType() &&
         "Insertelement types must match!");
  assert(Idx->getType()->isIntegerTy() &&
         "Insertelement index must be i32 type!");

  if (Constant *FC = ConstantFoldInsertElementInstruction(Val, Elt, Idx))
    return FC;          // Fold a few common cases.

  if (OnlyIfReducedTy == Val->getType())
    return nullptr;

  // Look up the constant in the table first to ensure uniqueness
  Constant *ArgVec[] = { Val, Elt, Idx };
  const ConstantExprKeyType Key(Instruction::InsertElement, ArgVec);

  LLVMContextImpl *pImpl = Val->getContext().pImpl;
  return pImpl->ExprConstants.getOrCreate(Val->getType(), Key);
}

Constant *ConstantExpr::getShuffleVector(Constant *V1, Constant *V2,
                                         Constant *Mask, Type *OnlyIfReducedTy) {
  assert(ShuffleVectorInst::isValidOperands(V1, V2, Mask) &&
         "Invalid shuffle vector constant expr operands!");

  if (Constant *FC = ConstantFoldShuffleVectorInstruction(V1, V2, Mask))
    return FC;          // Fold a few common cases.

  unsigned NElts = Mask->getType()->getVectorNumElements();
  Type *EltTy = V1->getType()->getVectorElementType();
  Type *ShufTy = VectorType::get(EltTy, NElts);

  if (OnlyIfReducedTy == ShufTy)
    return nullptr;

  // Look up the constant in the table first to ensure uniqueness
  Constant *ArgVec[] = { V1, V2, Mask };
  const ConstantExprKeyType Key(Instruction::ShuffleVector, ArgVec);

  LLVMContextImpl *pImpl = ShufTy->getContext().pImpl;
  return pImpl->ExprConstants.getOrCreate(ShufTy, Key);
}

Constant *ConstantExpr::getInsertValue(Constant *Agg, Constant *Val,
                                       ArrayRef<unsigned> Idxs,
                                       Type *OnlyIfReducedTy) {
  assert(Agg->getType()->isFirstClassType() &&
         "Non-first-class type for constant insertvalue expression");

  assert(ExtractValueInst::getIndexedType(Agg->getType(),
                                          Idxs) == Val->getType() &&
         "insertvalue indices invalid!");
  Type *ReqTy = Val->getType();

  if (Constant *FC = ConstantFoldInsertValueInstruction(Agg, Val, Idxs))
    return FC;

  if (OnlyIfReducedTy == ReqTy)
    return nullptr;

  Constant *ArgVec[] = { Agg, Val };
  const ConstantExprKeyType Key(Instruction::InsertValue, ArgVec, 0, 0, Idxs);

  LLVMContextImpl *pImpl = Agg->getContext().pImpl;
  return pImpl->ExprConstants.getOrCreate(ReqTy, Key);
}

Constant *ConstantExpr::getExtractValue(Constant *Agg, ArrayRef<unsigned> Idxs,
                                        Type *OnlyIfReducedTy) {
  assert(Agg->getType()->isFirstClassType() &&
         "Tried to create extractelement operation on non-first-class type!");

  Type *ReqTy = ExtractValueInst::getIndexedType(Agg->getType(), Idxs);
  (void)ReqTy;
  assert(ReqTy && "extractvalue indices invalid!");

  assert(Agg->getType()->isFirstClassType() &&
         "Non-first-class type for constant extractvalue expression");
  if (Constant *FC = ConstantFoldExtractValueInstruction(Agg, Idxs))
    return FC;

  if (OnlyIfReducedTy == ReqTy)
    return nullptr;

  Constant *ArgVec[] = { Agg };
  const ConstantExprKeyType Key(Instruction::ExtractValue, ArgVec, 0, 0, Idxs);

  LLVMContextImpl *pImpl = Agg->getContext().pImpl;
  return pImpl->ExprConstants.getOrCreate(ReqTy, Key);
}

Constant *ConstantExpr::getNeg(Constant *C, bool HasNUW, bool HasNSW) {
  assert(C->getType()->isIntOrIntVectorTy() &&
         "Cannot NEG a nonintegral value!");
  return getSub(ConstantFP::getZeroValueForNegation(C->getType()),
                C, HasNUW, HasNSW);
}

Constant *ConstantExpr::getFNeg(Constant *C) {
  assert(C->getType()->isFPOrFPVectorTy() &&
         "Cannot FNEG a non-floating-point value!");
  return getFSub(ConstantFP::getZeroValueForNegation(C->getType()), C);
}

Constant *ConstantExpr::getNot(Constant *C) {
  assert(C->getType()->isIntOrIntVectorTy() &&
         "Cannot NOT a nonintegral value!");
  return get(Instruction::Xor, C, Constant::getAllOnesValue(C->getType()));
}

Constant *ConstantExpr::getAdd(Constant *C1, Constant *C2,
                               bool HasNUW, bool HasNSW) {
  unsigned Flags = (HasNUW ? OverflowingBinaryOperator::NoUnsignedWrap : 0) |
                   (HasNSW ? OverflowingBinaryOperator::NoSignedWrap   : 0);
  return get(Instruction::Add, C1, C2, Flags);
}

Constant *ConstantExpr::getFAdd(Constant *C1, Constant *C2) {
  return get(Instruction::FAdd, C1, C2);
}

Constant *ConstantExpr::getSub(Constant *C1, Constant *C2,
                               bool HasNUW, bool HasNSW) {
  unsigned Flags = (HasNUW ? OverflowingBinaryOperator::NoUnsignedWrap : 0) |
                   (HasNSW ? OverflowingBinaryOperator::NoSignedWrap   : 0);
  return get(Instruction::Sub, C1, C2, Flags);
}

Constant *ConstantExpr::getFSub(Constant *C1, Constant *C2) {
  return get(Instruction::FSub, C1, C2);
}

Constant *ConstantExpr::getMul(Constant *C1, Constant *C2,
                               bool HasNUW, bool HasNSW) {
  unsigned Flags = (HasNUW ? OverflowingBinaryOperator::NoUnsignedWrap : 0) |
                   (HasNSW ? OverflowingBinaryOperator::NoSignedWrap   : 0);
  return get(Instruction::Mul, C1, C2, Flags);
}

Constant *ConstantExpr::getFMul(Constant *C1, Constant *C2) {
  return get(Instruction::FMul, C1, C2);
}

Constant *ConstantExpr::getUDiv(Constant *C1, Constant *C2, bool isExact) {
  return get(Instruction::UDiv, C1, C2,
             isExact ? PossiblyExactOperator::IsExact : 0);
}

Constant *ConstantExpr::getSDiv(Constant *C1, Constant *C2, bool isExact) {
  return get(Instruction::SDiv, C1, C2,
             isExact ? PossiblyExactOperator::IsExact : 0);
}

Constant *ConstantExpr::getFDiv(Constant *C1, Constant *C2) {
  return get(Instruction::FDiv, C1, C2);
}

Constant *ConstantExpr::getURem(Constant *C1, Constant *C2) {
  return get(Instruction::URem, C1, C2);
}

Constant *ConstantExpr::getSRem(Constant *C1, Constant *C2) {
  return get(Instruction::SRem, C1, C2);
}

Constant *ConstantExpr::getFRem(Constant *C1, Constant *C2) {
  return get(Instruction::FRem, C1, C2);
}

Constant *ConstantExpr::getAnd(Constant *C1, Constant *C2) {
  return get(Instruction::And, C1, C2);
}

Constant *ConstantExpr::getOr(Constant *C1, Constant *C2) {
  return get(Instruction::Or, C1, C2);
}

Constant *ConstantExpr::getXor(Constant *C1, Constant *C2) {
  return get(Instruction::Xor, C1, C2);
}

Constant *ConstantExpr::getShl(Constant *C1, Constant *C2,
                               bool HasNUW, bool HasNSW) {
  unsigned Flags = (HasNUW ? OverflowingBinaryOperator::NoUnsignedWrap : 0) |
                   (HasNSW ? OverflowingBinaryOperator::NoSignedWrap   : 0);
  return get(Instruction::Shl, C1, C2, Flags);
}

Constant *ConstantExpr::getLShr(Constant *C1, Constant *C2, bool isExact) {
  return get(Instruction::LShr, C1, C2,
             isExact ? PossiblyExactOperator::IsExact : 0);
}

Constant *ConstantExpr::getAShr(Constant *C1, Constant *C2, bool isExact) {
  return get(Instruction::AShr, C1, C2,
             isExact ? PossiblyExactOperator::IsExact : 0);
}

/// getBinOpIdentity - Return the identity for the given binary operation,
/// i.e. a constant C such that X op C = X and C op X = X for every X.  It
/// returns null if the operator doesn't have an identity.
Constant *ConstantExpr::getBinOpIdentity(unsigned Opcode, Type *Ty) {
  switch (Opcode) {
  default:
    // Doesn't have an identity.
    return nullptr;

  case Instruction::Add:
  case Instruction::Or:
  case Instruction::Xor:
    return Constant::getNullValue(Ty);

  case Instruction::Mul:
    return ConstantInt::get(Ty, 1);

  case Instruction::And:
    return Constant::getAllOnesValue(Ty);
  }
}

/// getBinOpAbsorber - Return the absorbing element for the given binary
/// operation, i.e. a constant C such that X op C = C and C op X = C for
/// every X.  For example, this returns zero for integer multiplication.
/// It returns null if the operator doesn't have an absorbing element.
Constant *ConstantExpr::getBinOpAbsorber(unsigned Opcode, Type *Ty) {
  switch (Opcode) {
  default:
    // Doesn't have an absorber.
    return nullptr;

  case Instruction::Or:
    return Constant::getAllOnesValue(Ty);

  case Instruction::And:
  case Instruction::Mul:
    return Constant::getNullValue(Ty);
  }
}

// destroyConstant - Remove the constant from the constant table...
//
void ConstantExpr::destroyConstantImpl() {
  getType()->getContext().pImpl->ExprConstants.remove(this);
}

const char *ConstantExpr::getOpcodeName() const {
  return Instruction::getOpcodeName(getOpcode());
}

GetElementPtrConstantExpr::GetElementPtrConstantExpr(
    Type *SrcElementTy, Constant *C, ArrayRef<Constant *> IdxList, Type *DestTy)
    : ConstantExpr(DestTy, Instruction::GetElementPtr,
                   OperandTraits<GetElementPtrConstantExpr>::op_end(this) -
                       (IdxList.size() + 1),
                   IdxList.size() + 1),
      SrcElementTy(SrcElementTy) {
  Op<0>() = C;
  Use *OperandList = getOperandList();
  for (unsigned i = 0, E = IdxList.size(); i != E; ++i)
    OperandList[i+1] = IdxList[i];
}

Type *GetElementPtrConstantExpr::getSourceElementType() const {
  return SrcElementTy;
}

//===----------------------------------------------------------------------===//
//                       ConstantData* implementations

void ConstantDataArray::anchor() {}
void ConstantDataVector::anchor() {}

/// getElementType - Return the element type of the array/vector.
Type *ConstantDataSequential::getElementType() const {
  return getType()->getElementType();
}

StringRef ConstantDataSequential::getRawDataValues() const {
  return StringRef(DataElements, getNumElements()*getElementByteSize());
}

/// isElementTypeCompatible - Return true if a ConstantDataSequential can be
/// formed with a vector or array of the specified element type.
/// ConstantDataArray only works with normal float and int types that are
/// stored densely in memory, not with things like i42 or x86_f80.
bool ConstantDataSequential::isElementTypeCompatible(const Type *Ty) {
  if (Ty->isFloatTy() || Ty->isDoubleTy()) return true;
  if (const IntegerType *IT = dyn_cast<IntegerType>(Ty)) {
    switch (IT->getBitWidth()) {
    case 8:
    case 16:
    case 32:
    case 64:
      return true;
    default: break;
    }
  }
  return false;
}

/// getNumElements - Return the number of elements in the array or vector.
unsigned ConstantDataSequential::getNumElements() const {
  if (ArrayType *AT = dyn_cast<ArrayType>(getType()))
    return AT->getNumElements();
  return getType()->getVectorNumElements();
}


/// getElementByteSize - Return the size in bytes of the elements in the data.
uint64_t ConstantDataSequential::getElementByteSize() const {
  return getElementType()->getPrimitiveSizeInBits()/8;
}

/// getElementPointer - Return the start of the specified element.
const char *ConstantDataSequential::getElementPointer(unsigned Elt) const {
  assert(Elt < getNumElements() && "Invalid Elt");
  return DataElements+Elt*getElementByteSize();
}


/// isAllZeros - return true if the array is empty or all zeros.
static bool isAllZeros(StringRef Arr) {
  for (StringRef::iterator I = Arr.begin(), E = Arr.end(); I != E; ++I)
    if (*I != 0)
      return false;
  return true;
}

/// getImpl - This is the underlying implementation of all of the
/// ConstantDataSequential::get methods.  They all thunk down to here, providing
/// the correct element type.  We take the bytes in as a StringRef because
/// we *want* an underlying "char*" to avoid TBAA type punning violations.
Constant *ConstantDataSequential::getImpl(StringRef Elements, Type *Ty) {
  assert(isElementTypeCompatible(Ty->getSequentialElementType()));
  // If the elements are all zero or there are no elements, return a CAZ, which
  // is more dense and canonical.
  if (isAllZeros(Elements))
    return ConstantAggregateZero::get(Ty);

  // Do a lookup to see if we have already formed one of these.
  auto &Slot =
      *Ty->getContext()
           .pImpl->CDSConstants.insert(std::make_pair(Elements, nullptr))
           .first;

  // The bucket can point to a linked list of different CDS's that have the same
  // body but different types.  For example, 0,0,0,1 could be a 4 element array
  // of i8, or a 1-element array of i32.  They'll both end up in the same
  /// StringMap bucket, linked up by their Next pointers.  Walk the list.
  ConstantDataSequential **Entry = &Slot.second;
  for (ConstantDataSequential *Node = *Entry; Node;
       Entry = &Node->Next, Node = *Entry)
    if (Node->getType() == Ty)
      return Node;

  // Okay, we didn't get a hit.  Create a node of the right class, link it in,
  // and return it.
  if (isa<ArrayType>(Ty))
    return *Entry = new ConstantDataArray(Ty, Slot.first().data());

  assert(isa<VectorType>(Ty));
  return *Entry = new ConstantDataVector(Ty, Slot.first().data());
}

void ConstantDataSequential::destroyConstantImpl() {
  // Remove the constant from the StringMap.
  StringMap<ConstantDataSequential*> &CDSConstants = 
    getType()->getContext().pImpl->CDSConstants;

  StringMap<ConstantDataSequential*>::iterator Slot =
    CDSConstants.find(getRawDataValues());

  assert(Slot != CDSConstants.end() && "CDS not found in uniquing table");

  ConstantDataSequential **Entry = &Slot->getValue();

  // Remove the entry from the hash table.
  if (!(*Entry)->Next) {
    // If there is only one value in the bucket (common case) it must be this
    // entry, and removing the entry should remove the bucket completely.
    assert((*Entry) == this && "Hash mismatch in ConstantDataSequential");
    getContext().pImpl->CDSConstants.erase(Slot);
  } else {
    // Otherwise, there are multiple entries linked off the bucket, unlink the 
    // node we care about but keep the bucket around.
    for (ConstantDataSequential *Node = *Entry; ;
         Entry = &Node->Next, Node = *Entry) {
      assert(Node && "Didn't find entry in its uniquing hash table!");
      // If we found our entry, unlink it from the list and we're done.
      if (Node == this) {
        *Entry = Node->Next;
        break;
      }
    }
  }

  // If we were part of a list, make sure that we don't delete the list that is
  // still owned by the uniquing map.
  Next = nullptr;
}

/// get() constructors - Return a constant with array type with an element
/// count and element type matching the ArrayRef passed in.  Note that this
/// can return a ConstantAggregateZero object.
Constant *ConstantDataArray::get(LLVMContext &Context, ArrayRef<uint8_t> Elts) {
  Type *Ty = ArrayType::get(Type::getInt8Ty(Context), Elts.size());
  const char *Data = reinterpret_cast<const char *>(Elts.data());
  return getImpl(StringRef(const_cast<char *>(Data), Elts.size()*1), Ty);
}
Constant *ConstantDataArray::get(LLVMContext &Context, ArrayRef<uint16_t> Elts){
  Type *Ty = ArrayType::get(Type::getInt16Ty(Context), Elts.size());
  const char *Data = reinterpret_cast<const char *>(Elts.data());
  return getImpl(StringRef(const_cast<char *>(Data), Elts.size()*2), Ty);
}
Constant *ConstantDataArray::get(LLVMContext &Context, ArrayRef<uint32_t> Elts){
  Type *Ty = ArrayType::get(Type::getInt32Ty(Context), Elts.size());
  const char *Data = reinterpret_cast<const char *>(Elts.data());
  return getImpl(StringRef(const_cast<char *>(Data), Elts.size()*4), Ty);
}
Constant *ConstantDataArray::get(LLVMContext &Context, ArrayRef<uint64_t> Elts){
  Type *Ty = ArrayType::get(Type::getInt64Ty(Context), Elts.size());
  const char *Data = reinterpret_cast<const char *>(Elts.data());
  return getImpl(StringRef(const_cast<char *>(Data), Elts.size()*8), Ty);
}
Constant *ConstantDataArray::get(LLVMContext &Context, ArrayRef<float> Elts) {
  Type *Ty = ArrayType::get(Type::getFloatTy(Context), Elts.size());
  const char *Data = reinterpret_cast<const char *>(Elts.data());
  return getImpl(StringRef(const_cast<char *>(Data), Elts.size()*4), Ty);
}
Constant *ConstantDataArray::get(LLVMContext &Context, ArrayRef<double> Elts) {
  Type *Ty = ArrayType::get(Type::getDoubleTy(Context), Elts.size());
  const char *Data = reinterpret_cast<const char *>(Elts.data());
  return getImpl(StringRef(const_cast<char *>(Data), Elts.size() * 8), Ty);
}

/// getFP() constructors - Return a constant with array type with an element
/// count and element type of float with precision matching the number of
/// bits in the ArrayRef passed in. (i.e. half for 16bits, float for 32bits,
/// double for 64bits) Note that this can return a ConstantAggregateZero
/// object.
Constant *ConstantDataArray::getFP(LLVMContext &Context,
                                   ArrayRef<uint16_t> Elts) {
  Type *Ty = VectorType::get(Type::getHalfTy(Context), Elts.size());
  const char *Data = reinterpret_cast<const char *>(Elts.data());
  return getImpl(StringRef(const_cast<char *>(Data), Elts.size() * 2), Ty);
}
Constant *ConstantDataArray::getFP(LLVMContext &Context,
                                   ArrayRef<uint32_t> Elts) {
  Type *Ty = ArrayType::get(Type::getFloatTy(Context), Elts.size());
  const char *Data = reinterpret_cast<const char *>(Elts.data());
  return getImpl(StringRef(const_cast<char *>(Data), Elts.size() * 4), Ty);
}
Constant *ConstantDataArray::getFP(LLVMContext &Context,
                                   ArrayRef<uint64_t> Elts) {
  Type *Ty = ArrayType::get(Type::getDoubleTy(Context), Elts.size());
  const char *Data = reinterpret_cast<const char *>(Elts.data());
  return getImpl(StringRef(const_cast<char *>(Data), Elts.size() * 8), Ty);
}

/// getString - This method constructs a CDS and initializes it with a text
/// string. The default behavior (AddNull==true) causes a null terminator to
/// be placed at the end of the array (increasing the length of the string by
/// one more than the StringRef would normally indicate.  Pass AddNull=false
/// to disable this behavior.
Constant *ConstantDataArray::getString(LLVMContext &Context,
                                       StringRef Str, bool AddNull) {
  if (!AddNull) {
    const uint8_t *Data = reinterpret_cast<const uint8_t *>(Str.data());
    return get(Context, makeArrayRef(const_cast<uint8_t *>(Data),
               Str.size()));
  }

  SmallVector<uint8_t, 64> ElementVals;
  ElementVals.append(Str.begin(), Str.end());
  ElementVals.push_back(0);
  return get(Context, ElementVals);
}

/// get() constructors - Return a constant with vector type with an element
/// count and element type matching the ArrayRef passed in.  Note that this
/// can return a ConstantAggregateZero object.
Constant *ConstantDataVector::get(LLVMContext &Context, ArrayRef<uint8_t> Elts){
  Type *Ty = VectorType::get(Type::getInt8Ty(Context), Elts.size());
  const char *Data = reinterpret_cast<const char *>(Elts.data());
  return getImpl(StringRef(const_cast<char *>(Data), Elts.size()*1), Ty);
}
Constant *ConstantDataVector::get(LLVMContext &Context, ArrayRef<uint16_t> Elts){
  Type *Ty = VectorType::get(Type::getInt16Ty(Context), Elts.size());
  const char *Data = reinterpret_cast<const char *>(Elts.data());
  return getImpl(StringRef(const_cast<char *>(Data), Elts.size()*2), Ty);
}
Constant *ConstantDataVector::get(LLVMContext &Context, ArrayRef<uint32_t> Elts){
  Type *Ty = VectorType::get(Type::getInt32Ty(Context), Elts.size());
  const char *Data = reinterpret_cast<const char *>(Elts.data());
  return getImpl(StringRef(const_cast<char *>(Data), Elts.size()*4), Ty);
}
Constant *ConstantDataVector::get(LLVMContext &Context, ArrayRef<uint64_t> Elts){
  Type *Ty = VectorType::get(Type::getInt64Ty(Context), Elts.size());
  const char *Data = reinterpret_cast<const char *>(Elts.data());
  return getImpl(StringRef(const_cast<char *>(Data), Elts.size()*8), Ty);
}
Constant *ConstantDataVector::get(LLVMContext &Context, ArrayRef<float> Elts) {
  Type *Ty = VectorType::get(Type::getFloatTy(Context), Elts.size());
  const char *Data = reinterpret_cast<const char *>(Elts.data());
  return getImpl(StringRef(const_cast<char *>(Data), Elts.size()*4), Ty);
}
Constant *ConstantDataVector::get(LLVMContext &Context, ArrayRef<double> Elts) {
  Type *Ty = VectorType::get(Type::getDoubleTy(Context), Elts.size());
  const char *Data = reinterpret_cast<const char *>(Elts.data());
  return getImpl(StringRef(const_cast<char *>(Data), Elts.size() * 8), Ty);
}

/// getFP() constructors - Return a constant with vector type with an element
/// count and element type of float with the precision matching the number of
/// bits in the ArrayRef passed in.  (i.e. half for 16bits, float for 32bits,
/// double for 64bits) Note that this can return a ConstantAggregateZero
/// object.
Constant *ConstantDataVector::getFP(LLVMContext &Context,
                                    ArrayRef<uint16_t> Elts) {
  Type *Ty = VectorType::get(Type::getHalfTy(Context), Elts.size());
  const char *Data = reinterpret_cast<const char *>(Elts.data());
  return getImpl(StringRef(const_cast<char *>(Data), Elts.size() * 2), Ty);
}
Constant *ConstantDataVector::getFP(LLVMContext &Context,
                                    ArrayRef<uint32_t> Elts) {
  Type *Ty = VectorType::get(Type::getFloatTy(Context), Elts.size());
  const char *Data = reinterpret_cast<const char *>(Elts.data());
  return getImpl(StringRef(const_cast<char *>(Data), Elts.size() * 4), Ty);
}
Constant *ConstantDataVector::getFP(LLVMContext &Context,
                                    ArrayRef<uint64_t> Elts) {
  Type *Ty = VectorType::get(Type::getDoubleTy(Context), Elts.size());
  const char *Data = reinterpret_cast<const char *>(Elts.data());
  return getImpl(StringRef(const_cast<char *>(Data), Elts.size() * 8), Ty);
}

Constant *ConstantDataVector::getSplat(unsigned NumElts, Constant *V) {
  assert(isElementTypeCompatible(V->getType()) &&
         "Element type not compatible with ConstantData");
  if (ConstantInt *CI = dyn_cast<ConstantInt>(V)) {
    if (CI->getType()->isIntegerTy(8)) {
      SmallVector<uint8_t, 16> Elts(NumElts, CI->getZExtValue());
      return get(V->getContext(), Elts);
    }
    if (CI->getType()->isIntegerTy(16)) {
      SmallVector<uint16_t, 16> Elts(NumElts, CI->getZExtValue());
      return get(V->getContext(), Elts);
    }
    if (CI->getType()->isIntegerTy(32)) {
      SmallVector<uint32_t, 16> Elts(NumElts, CI->getZExtValue());
      return get(V->getContext(), Elts);
    }
    assert(CI->getType()->isIntegerTy(64) && "Unsupported ConstantData type");
    SmallVector<uint64_t, 16> Elts(NumElts, CI->getZExtValue());
    return get(V->getContext(), Elts);
  }

  if (ConstantFP *CFP = dyn_cast<ConstantFP>(V)) {
    if (CFP->getType()->isFloatTy()) {
      SmallVector<uint32_t, 16> Elts(
          NumElts, CFP->getValueAPF().bitcastToAPInt().getLimitedValue());
      return getFP(V->getContext(), Elts);
    }
    if (CFP->getType()->isDoubleTy()) {
      SmallVector<uint64_t, 16> Elts(
          NumElts, CFP->getValueAPF().bitcastToAPInt().getLimitedValue());
      return getFP(V->getContext(), Elts);
    }
  }
  return ConstantVector::getSplat(NumElts, V);
}


/// getElementAsInteger - If this is a sequential container of integers (of
/// any size), return the specified element in the low bits of a uint64_t.
uint64_t ConstantDataSequential::getElementAsInteger(unsigned Elt) const {
  assert(isa<IntegerType>(getElementType()) &&
         "Accessor can only be used when element is an integer");
  const char *EltPtr = getElementPointer(Elt);

  // The data is stored in host byte order, make sure to cast back to the right
  // type to load with the right endianness.
  switch (getElementType()->getIntegerBitWidth()) {
  default: llvm_unreachable("Invalid bitwidth for CDS");
  case 8:
    return *const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(EltPtr));
  case 16:
    return *const_cast<uint16_t *>(reinterpret_cast<const uint16_t *>(EltPtr));
  case 32:
    return *const_cast<uint32_t *>(reinterpret_cast<const uint32_t *>(EltPtr));
  case 64:
    return *const_cast<uint64_t *>(reinterpret_cast<const uint64_t *>(EltPtr));
  }
}

/// getElementAsAPFloat - If this is a sequential container of floating point
/// type, return the specified element as an APFloat.
APFloat ConstantDataSequential::getElementAsAPFloat(unsigned Elt) const {
  const char *EltPtr = getElementPointer(Elt);

  switch (getElementType()->getTypeID()) {
  default:
    llvm_unreachable("Accessor can only be used when element is float/double!");
  case Type::FloatTyID: {
    auto EltVal = *reinterpret_cast<const uint32_t *>(EltPtr);
    return APFloat(APFloat::IEEEsingle, APInt(32, EltVal));
  }
  case Type::DoubleTyID: {
    auto EltVal = *reinterpret_cast<const uint64_t *>(EltPtr);
    return APFloat(APFloat::IEEEdouble, APInt(64, EltVal));
  }
  }
}

/// getElementAsFloat - If this is an sequential container of floats, return
/// the specified element as a float.
float ConstantDataSequential::getElementAsFloat(unsigned Elt) const {
  assert(getElementType()->isFloatTy() &&
         "Accessor can only be used when element is a 'float'");
  const float *EltPtr = reinterpret_cast<const float *>(getElementPointer(Elt));
  return *const_cast<float *>(EltPtr);
}

/// getElementAsDouble - If this is an sequential container of doubles, return
/// the specified element as a float.
double ConstantDataSequential::getElementAsDouble(unsigned Elt) const {
  assert(getElementType()->isDoubleTy() &&
         "Accessor can only be used when element is a 'float'");
  const double *EltPtr =
      reinterpret_cast<const double *>(getElementPointer(Elt));
  return *const_cast<double *>(EltPtr);
}

/// getElementAsConstant - Return a Constant for a specified index's element.
/// Note that this has to compute a new constant to return, so it isn't as
/// efficient as getElementAsInteger/Float/Double.
Constant *ConstantDataSequential::getElementAsConstant(unsigned Elt) const {
  if (getElementType()->isFloatTy() || getElementType()->isDoubleTy())
    return ConstantFP::get(getContext(), getElementAsAPFloat(Elt));

  return ConstantInt::get(getElementType(), getElementAsInteger(Elt));
}

/// isString - This method returns true if this is an array of i8.
bool ConstantDataSequential::isString() const {
  return isa<ArrayType>(getType()) && getElementType()->isIntegerTy(8);
}

/// isCString - This method returns true if the array "isString", ends with a
/// nul byte, and does not contains any other nul bytes.
bool ConstantDataSequential::isCString() const {
  if (!isString())
    return false;

  StringRef Str = getAsString();

  // The last value must be nul.
  if (Str.back() != 0) return false;

  // Other elements must be non-nul.
  return Str.drop_back().find(0) == StringRef::npos;
}

/// getSplatValue - If this is a splat constant, meaning that all of the
/// elements have the same value, return that value. Otherwise return nullptr.
Constant *ConstantDataVector::getSplatValue() const {
  const char *Base = getRawDataValues().data();

  // Compare elements 1+ to the 0'th element.
  unsigned EltSize = getElementByteSize();
  for (unsigned i = 1, e = getNumElements(); i != e; ++i)
    if (memcmp(Base, Base+i*EltSize, EltSize))
      return nullptr;

  // If they're all the same, return the 0th one as a representative.
  return getElementAsConstant(0);
}

//===----------------------------------------------------------------------===//
//                handleOperandChange implementations

/// Update this constant array to change uses of
/// 'From' to be uses of 'To'.  This must update the uniquing data structures
/// etc.
///
/// Note that we intentionally replace all uses of From with To here.  Consider
/// a large array that uses 'From' 1000 times.  By handling this case all here,
/// ConstantArray::handleOperandChange is only invoked once, and that
/// single invocation handles all 1000 uses.  Handling them one at a time would
/// work, but would be really slow because it would have to unique each updated
/// array instance.
///
void Constant::handleOperandChange(Value *From, Value *To, Use *U) {
  Value *Replacement = nullptr;
  switch (getValueID()) {
  default:
    llvm_unreachable("Not a constant!");
#define HANDLE_CONSTANT(Name)                                                  \
  case Value::Name##Val:                                                       \
    Replacement = cast<Name>(this)->handleOperandChangeImpl(From, To, U);      \
    break;
#include "llvm/IR/Value.def"
  }

  // If handleOperandChangeImpl returned nullptr, then it handled
  // replacing itself and we don't want to delete or replace anything else here.
  if (!Replacement)
    return;

  // I do need to replace this with an existing value.
  assert(Replacement != this && "I didn't contain From!");

  // Everyone using this now uses the replacement.
  replaceAllUsesWith(Replacement);

  // Delete the old constant!
  destroyConstant();
}

Value *ConstantInt::handleOperandChangeImpl(Value *From, Value *To, Use *U) {
  llvm_unreachable("Unsupported class for handleOperandChange()!");
}

Value *ConstantFP::handleOperandChangeImpl(Value *From, Value *To, Use *U) {
  llvm_unreachable("Unsupported class for handleOperandChange()!");
}

Value *UndefValue::handleOperandChangeImpl(Value *From, Value *To, Use *U) {
  llvm_unreachable("Unsupported class for handleOperandChange()!");
}

Value *ConstantPointerNull::handleOperandChangeImpl(Value *From, Value *To,
                                                    Use *U) {
  llvm_unreachable("Unsupported class for handleOperandChange()!");
}

Value *ConstantAggregateZero::handleOperandChangeImpl(Value *From, Value *To,
                                                      Use *U) {
  llvm_unreachable("Unsupported class for handleOperandChange()!");
}

Value *ConstantDataSequential::handleOperandChangeImpl(Value *From, Value *To,
                                                       Use *U) {
  llvm_unreachable("Unsupported class for handleOperandChange()!");
}

Value *ConstantArray::handleOperandChangeImpl(Value *From, Value *To, Use *U) {
  assert(isa<Constant>(To) && "Cannot make Constant refer to non-constant!");
  Constant *ToC = cast<Constant>(To);

  SmallVector<Constant*, 8> Values;
  Values.reserve(getNumOperands());  // Build replacement array.

  // Fill values with the modified operands of the constant array.  Also,
  // compute whether this turns into an all-zeros array.
  unsigned NumUpdated = 0;

  // Keep track of whether all the values in the array are "ToC".
  bool AllSame = true;
  Use *OperandList = getOperandList();
  for (Use *O = OperandList, *E = OperandList+getNumOperands(); O != E; ++O) {
    Constant *Val = cast<Constant>(O->get());
    if (Val == From) {
      Val = ToC;
      ++NumUpdated;
    }
    Values.push_back(Val);
    AllSame &= Val == ToC;
  }

  if (AllSame && ToC->isNullValue())
    return ConstantAggregateZero::get(getType());

  if (AllSame && isa<UndefValue>(ToC))
    return UndefValue::get(getType());

  // Check for any other type of constant-folding.
  if (Constant *C = getImpl(getType(), Values))
    return C;

  // Update to the new value.
  return getContext().pImpl->ArrayConstants.replaceOperandsInPlace(
      Values, this, From, ToC, NumUpdated, U - OperandList);
}

Value *ConstantStruct::handleOperandChangeImpl(Value *From, Value *To, Use *U) {
  assert(isa<Constant>(To) && "Cannot make Constant refer to non-constant!");
  Constant *ToC = cast<Constant>(To);

  Use *OperandList = getOperandList();
  unsigned OperandToUpdate = U-OperandList;
  assert(getOperand(OperandToUpdate) == From && "ReplaceAllUsesWith broken!");

  SmallVector<Constant*, 8> Values;
  Values.reserve(getNumOperands());  // Build replacement struct.

  // Fill values with the modified operands of the constant struct.  Also,
  // compute whether this turns into an all-zeros struct.
  bool isAllZeros = false;
  bool isAllUndef = false;
  if (ToC->isNullValue()) {
    isAllZeros = true;
    for (Use *O = OperandList, *E = OperandList+getNumOperands(); O != E; ++O) {
      Constant *Val = cast<Constant>(O->get());
      Values.push_back(Val);
      if (isAllZeros) isAllZeros = Val->isNullValue();
    }
  } else if (isa<UndefValue>(ToC)) {
    isAllUndef = true;
    for (Use *O = OperandList, *E = OperandList+getNumOperands(); O != E; ++O) {
      Constant *Val = cast<Constant>(O->get());
      Values.push_back(Val);
      if (isAllUndef) isAllUndef = isa<UndefValue>(Val);
    }
  } else {
    for (Use *O = OperandList, *E = OperandList + getNumOperands(); O != E; ++O)
      Values.push_back(cast<Constant>(O->get()));
  }
  Values[OperandToUpdate] = ToC;

  if (isAllZeros)
    return ConstantAggregateZero::get(getType());

  if (isAllUndef)
    return UndefValue::get(getType());

  // Update to the new value.
  return getContext().pImpl->StructConstants.replaceOperandsInPlace(
      Values, this, From, ToC);
}

Value *ConstantVector::handleOperandChangeImpl(Value *From, Value *To, Use *U) {
  assert(isa<Constant>(To) && "Cannot make Constant refer to non-constant!");
  Constant *ToC = cast<Constant>(To);

  SmallVector<Constant*, 8> Values;
  Values.reserve(getNumOperands());  // Build replacement array...
  unsigned NumUpdated = 0;
  for (unsigned i = 0, e = getNumOperands(); i != e; ++i) {
    Constant *Val = getOperand(i);
    if (Val == From) {
      ++NumUpdated;
      Val = ToC;
    }
    Values.push_back(Val);
  }

  if (Constant *C = getImpl(Values))
    return C;

  // Update to the new value.
  Use *OperandList = getOperandList();
  return getContext().pImpl->VectorConstants.replaceOperandsInPlace(
      Values, this, From, ToC, NumUpdated, U - OperandList);
}

Value *ConstantExpr::handleOperandChangeImpl(Value *From, Value *ToV, Use *U) {
  assert(isa<Constant>(ToV) && "Cannot make Constant refer to non-constant!");
  Constant *To = cast<Constant>(ToV);

  SmallVector<Constant*, 8> NewOps;
  unsigned NumUpdated = 0;
  for (unsigned i = 0, e = getNumOperands(); i != e; ++i) {
    Constant *Op = getOperand(i);
    if (Op == From) {
      ++NumUpdated;
      Op = To;
    }
    NewOps.push_back(Op);
  }
  assert(NumUpdated && "I didn't contain From!");

  if (Constant *C = getWithOperands(NewOps, getType(), true))
    return C;

  // Update to the new value.
  Use *OperandList = getOperandList();
  return getContext().pImpl->ExprConstants.replaceOperandsInPlace(
      NewOps, this, From, To, NumUpdated, U - OperandList);
}

Instruction *ConstantExpr::getAsInstruction() {
  SmallVector<Value *, 4> ValueOperands(op_begin(), op_end());
  ArrayRef<Value*> Ops(ValueOperands);

  switch (getOpcode()) {
  case Instruction::Trunc:
  case Instruction::ZExt:
  case Instruction::SExt:
  case Instruction::FPTrunc:
  case Instruction::FPExt:
  case Instruction::UIToFP:
  case Instruction::SIToFP:
  case Instruction::FPToUI:
  case Instruction::FPToSI:
  case Instruction::PtrToInt:
  case Instruction::IntToPtr:
  case Instruction::BitCast:
  case Instruction::AddrSpaceCast:
    return CastInst::Create((Instruction::CastOps)getOpcode(),
                            Ops[0], getType());
  case Instruction::Select:
    return SelectInst::Create(Ops[0], Ops[1], Ops[2]);
  case Instruction::InsertElement:
    return InsertElementInst::Create(Ops[0], Ops[1], Ops[2]);
  case Instruction::ExtractElement:
    return ExtractElementInst::Create(Ops[0], Ops[1]);
  case Instruction::InsertValue:
    return InsertValueInst::Create(Ops[0], Ops[1], getIndices());
  case Instruction::ExtractValue:
    return ExtractValueInst::Create(Ops[0], getIndices());
  case Instruction::ShuffleVector:
    return new ShuffleVectorInst(Ops[0], Ops[1], Ops[2]);

  case Instruction::GetElementPtr: {
    const auto *GO = cast<GEPOperator>(this);
    if (GO->isInBounds())
      return GetElementPtrInst::CreateInBounds(GO->getSourceElementType(),
                                               Ops[0], Ops.slice(1));
    return GetElementPtrInst::Create(GO->getSourceElementType(), Ops[0],
                                     Ops.slice(1));
  }
  case Instruction::ICmp:
  case Instruction::FCmp:
    return CmpInst::Create((Instruction::OtherOps)getOpcode(),
                           getPredicate(), Ops[0], Ops[1]);

  default:
    assert(getNumOperands() == 2 && "Must be binary operator?");
    BinaryOperator *BO =
      BinaryOperator::Create((Instruction::BinaryOps)getOpcode(),
                             Ops[0], Ops[1]);
    if (isa<OverflowingBinaryOperator>(BO)) {
      BO->setHasNoUnsignedWrap(SubclassOptionalData &
                               OverflowingBinaryOperator::NoUnsignedWrap);
      BO->setHasNoSignedWrap(SubclassOptionalData &
                             OverflowingBinaryOperator::NoSignedWrap);
    }
    if (isa<PossiblyExactOperator>(BO))
      BO->setIsExact(SubclassOptionalData & PossiblyExactOperator::IsExact);
    return BO;
  }
}

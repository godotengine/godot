//===- Loads.cpp - Local load analysis ------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines simple local analyses for load instructions.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/Loads.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
using namespace llvm;

/// \brief Test if A and B will obviously have the same value.
///
/// This includes recognizing that %t0 and %t1 will have the same
/// value in code like this:
/// \code
///   %t0 = getelementptr \@a, 0, 3
///   store i32 0, i32* %t0
///   %t1 = getelementptr \@a, 0, 3
///   %t2 = load i32* %t1
/// \endcode
///
static bool AreEquivalentAddressValues(const Value *A, const Value *B) {
  // Test if the values are trivially equivalent.
  if (A == B)
    return true;

  // Test if the values come from identical arithmetic instructions.
  // Use isIdenticalToWhenDefined instead of isIdenticalTo because
  // this function is only used when one address use dominates the
  // other, which means that they'll always either have the same
  // value or one of them will have an undefined value.
  if (isa<BinaryOperator>(A) || isa<CastInst>(A) || isa<PHINode>(A) ||
      isa<GetElementPtrInst>(A))
    if (const Instruction *BI = dyn_cast<Instruction>(B))
      if (cast<Instruction>(A)->isIdenticalToWhenDefined(BI))
        return true;

  // Otherwise they may not be equivalent.
  return false;
}

/// \brief Check if executing a load of this pointer value cannot trap.
///
/// If it is not obviously safe to load from the specified pointer, we do
/// a quick local scan of the basic block containing \c ScanFrom, to determine
/// if the address is already accessed.
///
/// This uses the pointee type to determine how many bytes need to be safe to
/// load from the pointer.
bool llvm::isSafeToLoadUnconditionally(Value *V, Instruction *ScanFrom,
                                       unsigned Align) {
  const DataLayout &DL = ScanFrom->getModule()->getDataLayout();

  // Zero alignment means that the load has the ABI alignment for the target
  if (Align == 0)
    Align = DL.getABITypeAlignment(V->getType()->getPointerElementType());
  assert(isPowerOf2_32(Align));

  int64_t ByteOffset = 0;
  Value *Base = V;
  Base = GetPointerBaseWithConstantOffset(V, ByteOffset, DL);

  if (ByteOffset < 0) // out of bounds
    return false;

  Type *BaseType = nullptr;
  unsigned BaseAlign = 0;
  if (const AllocaInst *AI = dyn_cast<AllocaInst>(Base)) {
    // An alloca is safe to load from as load as it is suitably aligned.
    BaseType = AI->getAllocatedType();
    BaseAlign = AI->getAlignment();
  } else if (const GlobalVariable *GV = dyn_cast<GlobalVariable>(Base)) {
    // Global variables are not necessarily safe to load from if they are
    // overridden. Their size may change or they may be weak and require a test
    // to determine if they were in fact provided.
    if (!GV->mayBeOverridden()) {
      BaseType = GV->getType()->getElementType();
      BaseAlign = GV->getAlignment();
    }
  }

  PointerType *AddrTy = cast<PointerType>(V->getType());
  uint64_t LoadSize = DL.getTypeStoreSize(AddrTy->getElementType());

  // If we found a base allocated type from either an alloca or global variable,
  // try to see if we are definitively within the allocated region. We need to
  // know the size of the base type and the loaded type to do anything in this
  // case.
  if (BaseType && BaseType->isSized()) {
    if (BaseAlign == 0)
      BaseAlign = DL.getPrefTypeAlignment(BaseType);

    if (Align <= BaseAlign) {
      // Check if the load is within the bounds of the underlying object.
      if (ByteOffset + LoadSize <= DL.getTypeAllocSize(BaseType) &&
          ((ByteOffset % Align) == 0))
        return true;
    }
  }

  // Otherwise, be a little bit aggressive by scanning the local block where we
  // want to check to see if the pointer is already being loaded or stored
  // from/to.  If so, the previous load or store would have already trapped,
  // so there is no harm doing an extra load (also, CSE will later eliminate
  // the load entirely).
  BasicBlock::iterator BBI = ScanFrom, E = ScanFrom->getParent()->begin();

  // We can at least always strip pointer casts even though we can't use the
  // base here.
  V = V->stripPointerCasts();

  while (BBI != E) {
    --BBI;

    // If we see a free or a call which may write to memory (i.e. which might do
    // a free) the pointer could be marked invalid.
    if (isa<CallInst>(BBI) && BBI->mayWriteToMemory() &&
        !isa<DbgInfoIntrinsic>(BBI))
      return false;

    Value *AccessedPtr;
    unsigned AccessedAlign;
    if (LoadInst *LI = dyn_cast<LoadInst>(BBI)) {
      AccessedPtr = LI->getPointerOperand();
      AccessedAlign = LI->getAlignment();
    } else if (StoreInst *SI = dyn_cast<StoreInst>(BBI)) {
      AccessedPtr = SI->getPointerOperand();
      AccessedAlign = SI->getAlignment();
    } else
      continue;

    Type *AccessedTy = AccessedPtr->getType()->getPointerElementType();
    if (AccessedAlign == 0)
      AccessedAlign = DL.getABITypeAlignment(AccessedTy);
    if (AccessedAlign < Align)
      continue;

    // Handle trivial cases.
    if (AccessedPtr == V)
      return true;

    if (AreEquivalentAddressValues(AccessedPtr->stripPointerCasts(), V) &&
        LoadSize <= DL.getTypeStoreSize(AccessedTy))
      return true;
  }
  return false;
}

/// \brief Scan the ScanBB block backwards to see if we have the value at the
/// memory address *Ptr locally available within a small number of instructions.
///
/// The scan starts from \c ScanFrom. \c MaxInstsToScan specifies the maximum
/// instructions to scan in the block. If it is set to \c 0, it will scan the whole
/// block.
///
/// If the value is available, this function returns it. If not, it returns the
/// iterator for the last validated instruction that the value would be live
/// through. If we scanned the entire block and didn't find something that
/// invalidates \c *Ptr or provides it, \c ScanFrom is left at the last
/// instruction processed and this returns null.
///
/// You can also optionally specify an alias analysis implementation, which
/// makes this more precise.
///
/// If \c AATags is non-null and a load or store is found, the AA tags from the
/// load or store are recorded there. If there are no AA tags or if no access is
/// found, it is left unmodified.
Value *llvm::FindAvailableLoadedValue(Value *Ptr, BasicBlock *ScanBB,
                                      BasicBlock::iterator &ScanFrom,
                                      unsigned MaxInstsToScan,
                                      AliasAnalysis *AA, AAMDNodes *AATags) {
  if (MaxInstsToScan == 0)
    MaxInstsToScan = ~0U;

  Type *AccessTy = cast<PointerType>(Ptr->getType())->getElementType();

  const DataLayout &DL = ScanBB->getModule()->getDataLayout();

  // Try to get the store size for the type.
  uint64_t AccessSize = DL.getTypeStoreSize(AccessTy);

  Value *StrippedPtr = Ptr->stripPointerCasts();

  while (ScanFrom != ScanBB->begin()) {
    // We must ignore debug info directives when counting (otherwise they
    // would affect codegen).
    Instruction *Inst = --ScanFrom;
    if (isa<DbgInfoIntrinsic>(Inst))
      continue;

    // Restore ScanFrom to expected value in case next test succeeds
    ScanFrom++;

    // Don't scan huge blocks.
    if (MaxInstsToScan-- == 0)
      return nullptr;

    --ScanFrom;
    // If this is a load of Ptr, the loaded value is available.
    // (This is true even if the load is volatile or atomic, although
    // those cases are unlikely.)
    if (LoadInst *LI = dyn_cast<LoadInst>(Inst))
      if (AreEquivalentAddressValues(
              LI->getPointerOperand()->stripPointerCasts(), StrippedPtr) &&
          CastInst::isBitOrNoopPointerCastable(LI->getType(), AccessTy, DL)) {
        if (AATags)
          LI->getAAMetadata(*AATags);
        return LI;
      }

    if (StoreInst *SI = dyn_cast<StoreInst>(Inst)) {
      Value *StorePtr = SI->getPointerOperand()->stripPointerCasts();
      // If this is a store through Ptr, the value is available!
      // (This is true even if the store is volatile or atomic, although
      // those cases are unlikely.)
      if (AreEquivalentAddressValues(StorePtr, StrippedPtr) &&
          CastInst::isBitOrNoopPointerCastable(SI->getValueOperand()->getType(),
                                               AccessTy, DL)) {
        if (AATags)
          SI->getAAMetadata(*AATags);
        return SI->getOperand(0);
      }

      // If both StrippedPtr and StorePtr reach all the way to an alloca or
      // global and they are different, ignore the store. This is a trivial form
      // of alias analysis that is important for reg2mem'd code.
      if ((isa<AllocaInst>(StrippedPtr) || isa<GlobalVariable>(StrippedPtr)) &&
          (isa<AllocaInst>(StorePtr) || isa<GlobalVariable>(StorePtr)) &&
          StrippedPtr != StorePtr)
        continue;

      // If we have alias analysis and it says the store won't modify the loaded
      // value, ignore the store.
      if (AA &&
          (AA->getModRefInfo(SI, StrippedPtr, AccessSize) &
           AliasAnalysis::Mod) == 0)
        continue;

      // Otherwise the store that may or may not alias the pointer, bail out.
      ++ScanFrom;
      return nullptr;
    }

    // If this is some other instruction that may clobber Ptr, bail out.
    if (Inst->mayWriteToMemory()) {
      // If alias analysis claims that it really won't modify the load,
      // ignore it.
      if (AA &&
          (AA->getModRefInfo(Inst, StrippedPtr, AccessSize) &
           AliasAnalysis::Mod) == 0)
        continue;

      // May modify the pointer, bail out.
      ++ScanFrom;
      return nullptr;
    }
  }

  // Got to the start of the block, we didn't find it, but are done for this
  // block.
  return nullptr;
}

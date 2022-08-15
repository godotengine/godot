//===- InstCombineCalls.cpp -----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the visitCall and visitInvoke functions.
//
//===----------------------------------------------------------------------===//

#include "InstCombineInternal.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Statepoint.h"
#include "llvm/Transforms/Utils/BuildLibCalls.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/SimplifyLibCalls.h"
using namespace llvm;
using namespace PatternMatch;

#define DEBUG_TYPE "instcombine"

STATISTIC(NumSimplified, "Number of library calls simplified");

/// getPromotedType - Return the specified type promoted as it would be to pass
/// though a va_arg area.
static Type *getPromotedType(Type *Ty) {
  if (IntegerType* ITy = dyn_cast<IntegerType>(Ty)) {
    if (ITy->getBitWidth() < 32)
      return Type::getInt32Ty(Ty->getContext());
  }
  return Ty;
}

/// reduceToSingleValueType - Given an aggregate type which ultimately holds a
/// single scalar element, like {{{type}}} or [1 x type], return type.
static Type *reduceToSingleValueType(Type *T) {
  while (!T->isSingleValueType()) {
    if (StructType *STy = dyn_cast<StructType>(T)) {
      if (STy->getNumElements() == 1)
        T = STy->getElementType(0);
      else
        break;
    } else if (ArrayType *ATy = dyn_cast<ArrayType>(T)) {
      if (ATy->getNumElements() == 1)
        T = ATy->getElementType();
      else
        break;
    } else
      break;
  }

  return T;
}

Instruction *InstCombiner::SimplifyMemTransfer(MemIntrinsic *MI) {
  unsigned DstAlign = getKnownAlignment(MI->getArgOperand(0), DL, MI, AC, DT);
  unsigned SrcAlign = getKnownAlignment(MI->getArgOperand(1), DL, MI, AC, DT);
  unsigned MinAlign = std::min(DstAlign, SrcAlign);
  unsigned CopyAlign = MI->getAlignment();

  if (CopyAlign < MinAlign) {
    MI->setAlignment(ConstantInt::get(MI->getAlignmentType(),
                                             MinAlign, false));
    return MI;
  }

  // If MemCpyInst length is 1/2/4/8 bytes then replace memcpy with
  // load/store.
  ConstantInt *MemOpLength = dyn_cast<ConstantInt>(MI->getArgOperand(2));
  if (!MemOpLength) return nullptr;

  // Source and destination pointer types are always "i8*" for intrinsic.  See
  // if the size is something we can handle with a single primitive load/store.
  // A single load+store correctly handles overlapping memory in the memmove
  // case.
  uint64_t Size = MemOpLength->getLimitedValue();
  assert(Size && "0-sized memory transferring should be removed already.");

  if (Size > 8 || (Size&(Size-1)))
    return nullptr;  // If not 1/2/4/8 bytes, exit.

  // Use an integer load+store unless we can find something better.
  unsigned SrcAddrSp =
    cast<PointerType>(MI->getArgOperand(1)->getType())->getAddressSpace();
  unsigned DstAddrSp =
    cast<PointerType>(MI->getArgOperand(0)->getType())->getAddressSpace();

  IntegerType* IntType = IntegerType::get(MI->getContext(), Size<<3);
  Type *NewSrcPtrTy = PointerType::get(IntType, SrcAddrSp);
  Type *NewDstPtrTy = PointerType::get(IntType, DstAddrSp);

  // Memcpy forces the use of i8* for the source and destination.  That means
  // that if you're using memcpy to move one double around, you'll get a cast
  // from double* to i8*.  We'd much rather use a double load+store rather than
  // an i64 load+store, here because this improves the odds that the source or
  // dest address will be promotable.  See if we can find a better type than the
  // integer datatype.
  Value *StrippedDest = MI->getArgOperand(0)->stripPointerCasts();
  MDNode *CopyMD = nullptr;
  if (StrippedDest != MI->getArgOperand(0)) {
    Type *SrcETy = cast<PointerType>(StrippedDest->getType())
                                    ->getElementType();
    if (SrcETy->isSized() && DL.getTypeStoreSize(SrcETy) == Size) {
      // The SrcETy might be something like {{{double}}} or [1 x double].  Rip
      // down through these levels if so.
      SrcETy = reduceToSingleValueType(SrcETy);

      if (SrcETy->isSingleValueType()) {
        NewSrcPtrTy = PointerType::get(SrcETy, SrcAddrSp);
        NewDstPtrTy = PointerType::get(SrcETy, DstAddrSp);

        // If the memcpy has metadata describing the members, see if we can
        // get the TBAA tag describing our copy.
        if (MDNode *M = MI->getMetadata(LLVMContext::MD_tbaa_struct)) {
          if (M->getNumOperands() == 3 && M->getOperand(0) &&
              mdconst::hasa<ConstantInt>(M->getOperand(0)) &&
              mdconst::extract<ConstantInt>(M->getOperand(0))->isNullValue() &&
              M->getOperand(1) &&
              mdconst::hasa<ConstantInt>(M->getOperand(1)) &&
              mdconst::extract<ConstantInt>(M->getOperand(1))->getValue() ==
                  Size &&
              M->getOperand(2) && isa<MDNode>(M->getOperand(2)))
            CopyMD = cast<MDNode>(M->getOperand(2));
        }
      }
    }
  }

  // If the memcpy/memmove provides better alignment info than we can
  // infer, use it.
  SrcAlign = std::max(SrcAlign, CopyAlign);
  DstAlign = std::max(DstAlign, CopyAlign);

  Value *Src = Builder->CreateBitCast(MI->getArgOperand(1), NewSrcPtrTy);
  Value *Dest = Builder->CreateBitCast(MI->getArgOperand(0), NewDstPtrTy);
  LoadInst *L = Builder->CreateLoad(Src, MI->isVolatile());
  L->setAlignment(SrcAlign);
  if (CopyMD)
    L->setMetadata(LLVMContext::MD_tbaa, CopyMD);
  StoreInst *S = Builder->CreateStore(L, Dest, MI->isVolatile());
  S->setAlignment(DstAlign);
  if (CopyMD)
    S->setMetadata(LLVMContext::MD_tbaa, CopyMD);

  // Set the size of the copy to 0, it will be deleted on the next iteration.
  MI->setArgOperand(2, Constant::getNullValue(MemOpLength->getType()));
  return MI;
}

Instruction *InstCombiner::SimplifyMemSet(MemSetInst *MI) {
  unsigned Alignment = getKnownAlignment(MI->getDest(), DL, MI, AC, DT);
  if (MI->getAlignment() < Alignment) {
    MI->setAlignment(ConstantInt::get(MI->getAlignmentType(),
                                             Alignment, false));
    return MI;
  }

  // Extract the length and alignment and fill if they are constant.
  ConstantInt *LenC = dyn_cast<ConstantInt>(MI->getLength());
  ConstantInt *FillC = dyn_cast<ConstantInt>(MI->getValue());
  if (!LenC || !FillC || !FillC->getType()->isIntegerTy(8))
    return nullptr;
  uint64_t Len = LenC->getLimitedValue();
  Alignment = MI->getAlignment();
  assert(Len && "0-sized memory setting should be removed already.");

  // memset(s,c,n) -> store s, c (for n=1,2,4,8)
  if (Len <= 8 && isPowerOf2_32((uint32_t)Len)) {
    Type *ITy = IntegerType::get(MI->getContext(), Len*8);  // n=1 -> i8.

    Value *Dest = MI->getDest();
    unsigned DstAddrSp = cast<PointerType>(Dest->getType())->getAddressSpace();
    Type *NewDstPtrTy = PointerType::get(ITy, DstAddrSp);
    Dest = Builder->CreateBitCast(Dest, NewDstPtrTy);

    // Alignment 0 is identity for alignment 1 for memset, but not store.
    if (Alignment == 0) Alignment = 1;

    // Extract the fill value and store.
    uint64_t Fill = FillC->getZExtValue()*0x0101010101010101ULL;
    StoreInst *S = Builder->CreateStore(ConstantInt::get(ITy, Fill), Dest,
                                        MI->isVolatile());
    S->setAlignment(Alignment);

    // Set the size of the copy to 0, it will be deleted on the next iteration.
    MI->setLength(Constant::getNullValue(LenC->getType()));
    return MI;
  }

  return nullptr;
}

#if 0 // HLSL Change - remove platform intrinsics
static Value *SimplifyX86insertps(const IntrinsicInst &II,
                                  InstCombiner::BuilderTy &Builder) {
  if (auto *CInt = dyn_cast<ConstantInt>(II.getArgOperand(2))) {
    VectorType *VecTy = cast<VectorType>(II.getType());
    assert(VecTy->getNumElements() == 4 && "insertps with wrong vector type");
    
    // The immediate permute control byte looks like this:
    //    [3:0] - zero mask for each 32-bit lane
    //    [5:4] - select one 32-bit destination lane
    //    [7:6] - select one 32-bit source lane

    uint8_t Imm = CInt->getZExtValue();
    uint8_t ZMask = Imm & 0xf;
    uint8_t DestLane = (Imm >> 4) & 0x3;
    uint8_t SourceLane = (Imm >> 6) & 0x3;

    ConstantAggregateZero *ZeroVector = ConstantAggregateZero::get(VecTy);

    // If all zero mask bits are set, this was just a weird way to
    // generate a zero vector.
    if (ZMask == 0xf)
      return ZeroVector;

    // Initialize by passing all of the first source bits through.
    int ShuffleMask[4] = { 0, 1, 2, 3 };

    // We may replace the second operand with the zero vector.
    Value *V1 = II.getArgOperand(1);

    if (ZMask) {
      // If the zero mask is being used with a single input or the zero mask
      // overrides the destination lane, this is a shuffle with the zero vector.
      if ((II.getArgOperand(0) == II.getArgOperand(1)) ||
          (ZMask & (1 << DestLane))) {
        V1 = ZeroVector;
        // We may still move 32-bits of the first source vector from one lane
        // to another.
        ShuffleMask[DestLane] = SourceLane;
        // The zero mask may override the previous insert operation.
        for (unsigned i = 0; i < 4; ++i)
          if ((ZMask >> i) & 0x1)
            ShuffleMask[i] = i + 4;
      } else {
        // TODO: Model this case as 2 shuffles or a 'logical and' plus shuffle?
        return nullptr;
      }
    } else {
      // Replace the selected destination lane with the selected source lane.
      ShuffleMask[DestLane] = SourceLane + 4;
    }
  
    return Builder.CreateShuffleVector(II.getArgOperand(0), V1, ShuffleMask);
  }
  return nullptr;
}

/// The shuffle mask for a perm2*128 selects any two halves of two 256-bit
/// source vectors, unless a zero bit is set. If a zero bit is set,
/// then ignore that half of the mask and clear that half of the vector.
static Value *SimplifyX86vperm2(const IntrinsicInst &II,
                                InstCombiner::BuilderTy &Builder) {
  if (auto *CInt = dyn_cast<ConstantInt>(II.getArgOperand(2))) {
    VectorType *VecTy = cast<VectorType>(II.getType());
    ConstantAggregateZero *ZeroVector = ConstantAggregateZero::get(VecTy);

    // The immediate permute control byte looks like this:
    //    [1:0] - select 128 bits from sources for low half of destination
    //    [2]   - ignore
    //    [3]   - zero low half of destination
    //    [5:4] - select 128 bits from sources for high half of destination
    //    [6]   - ignore
    //    [7]   - zero high half of destination

    uint8_t Imm = CInt->getZExtValue();

    bool LowHalfZero = Imm & 0x08;
    bool HighHalfZero = Imm & 0x80;

    // If both zero mask bits are set, this was just a weird way to
    // generate a zero vector.
    if (LowHalfZero && HighHalfZero)
      return ZeroVector;

    // If 0 or 1 zero mask bits are set, this is a simple shuffle.
    unsigned NumElts = VecTy->getNumElements();
    unsigned HalfSize = NumElts / 2;
    SmallVector<int, 8> ShuffleMask(NumElts);

    // The high bit of the selection field chooses the 1st or 2nd operand.
    bool LowInputSelect = Imm & 0x02;
    bool HighInputSelect = Imm & 0x20;
    
    // The low bit of the selection field chooses the low or high half
    // of the selected operand.
    bool LowHalfSelect = Imm & 0x01;
    bool HighHalfSelect = Imm & 0x10;

    // Determine which operand(s) are actually in use for this instruction.
    Value *V0 = LowInputSelect ? II.getArgOperand(1) : II.getArgOperand(0);
    Value *V1 = HighInputSelect ? II.getArgOperand(1) : II.getArgOperand(0);
    
    // If needed, replace operands based on zero mask.
    V0 = LowHalfZero ? ZeroVector : V0;
    V1 = HighHalfZero ? ZeroVector : V1;
    
    // Permute low half of result.
    unsigned StartIndex = LowHalfSelect ? HalfSize : 0;
    for (unsigned i = 0; i < HalfSize; ++i)
      ShuffleMask[i] = StartIndex + i;

    // Permute high half of result.
    StartIndex = HighHalfSelect ? HalfSize : 0;
    StartIndex += NumElts;
    for (unsigned i = 0; i < HalfSize; ++i)
      ShuffleMask[i + HalfSize] = StartIndex + i;

    return Builder.CreateShuffleVector(V0, V1, ShuffleMask);
  }
  return nullptr;
}
#endif // HLSL Change - remove platform intrinsics


/// visitCallInst - CallInst simplification.  This mostly only handles folding
/// of intrinsic instructions.  For normal calls, it allows visitCallSite to do
/// the heavy lifting.
///
Instruction *InstCombiner::visitCallInst(CallInst &CI) {
  auto Args = CI.arg_operands();
  if (Value *V = SimplifyCall(CI.getCalledValue(), Args.begin(), Args.end(), DL,
                              TLI, DT, AC))
    return ReplaceInstUsesWith(CI, V);

  if (isFreeCall(&CI, TLI))
    return visitFree(CI);

  // If the caller function is nounwind, mark the call as nounwind, even if the
  // callee isn't.
  if (CI.getParent()->getParent()->doesNotThrow() &&
      !CI.doesNotThrow()) {
    CI.setDoesNotThrow();
    return &CI;
  }

  IntrinsicInst *II = dyn_cast<IntrinsicInst>(&CI);
  if (!II) return visitCallSite(&CI);

  // Intrinsics cannot occur in an invoke, so handle them here instead of in
  // visitCallSite.
  if (MemIntrinsic *MI = dyn_cast<MemIntrinsic>(II)) {
    bool Changed = false;

    // memmove/cpy/set of zero bytes is a noop.
    if (Constant *NumBytes = dyn_cast<Constant>(MI->getLength())) {
      if (NumBytes->isNullValue())
        return EraseInstFromFunction(CI);

      if (ConstantInt *CI = dyn_cast<ConstantInt>(NumBytes))
        if (CI->getZExtValue() == 1) {
          // Replace the instruction with just byte operations.  We would
          // transform other cases to loads/stores, but we don't know if
          // alignment is sufficient.
        }
    }

    // No other transformations apply to volatile transfers.
    if (MI->isVolatile())
      return nullptr;

    // If we have a memmove and the source operation is a constant global,
    // then the source and dest pointers can't alias, so we can change this
    // into a call to memcpy.
    if (MemMoveInst *MMI = dyn_cast<MemMoveInst>(MI)) {
      if (GlobalVariable *GVSrc = dyn_cast<GlobalVariable>(MMI->getSource()))
        if (GVSrc->isConstant()) {
          Module *M = CI.getParent()->getParent()->getParent();
          Intrinsic::ID MemCpyID = Intrinsic::memcpy;
          Type *Tys[3] = { CI.getArgOperand(0)->getType(),
                           CI.getArgOperand(1)->getType(),
                           CI.getArgOperand(2)->getType() };
          CI.setCalledFunction(Intrinsic::getDeclaration(M, MemCpyID, Tys));
          Changed = true;
        }
    }

    if (MemTransferInst *MTI = dyn_cast<MemTransferInst>(MI)) {
      // memmove(x,x,size) -> noop.
      if (MTI->getSource() == MTI->getDest())
        return EraseInstFromFunction(CI);
    }

    // If we can determine a pointer alignment that is bigger than currently
    // set, update the alignment.
    if (isa<MemTransferInst>(MI)) {
      if (Instruction *I = SimplifyMemTransfer(MI))
        return I;
    } else if (MemSetInst *MSI = dyn_cast<MemSetInst>(MI)) {
      if (Instruction *I = SimplifyMemSet(MSI))
        return I;
    }

    if (Changed) return II;
  }

  switch (II->getIntrinsicID()) {
  default: break;
  case Intrinsic::objectsize: {
    uint64_t Size;
    if (getObjectSize(II->getArgOperand(0), Size, DL, TLI))
      return ReplaceInstUsesWith(CI, ConstantInt::get(CI.getType(), Size));
    return nullptr;
  }
  case Intrinsic::bswap: {
    Value *IIOperand = II->getArgOperand(0);
    Value *X = nullptr;

    // bswap(bswap(x)) -> x
    if (match(IIOperand, m_BSwap(m_Value(X))))
        return ReplaceInstUsesWith(CI, X);

    // bswap(trunc(bswap(x))) -> trunc(lshr(x, c))
    if (match(IIOperand, m_Trunc(m_BSwap(m_Value(X))))) {
      unsigned C = X->getType()->getPrimitiveSizeInBits() -
        IIOperand->getType()->getPrimitiveSizeInBits();
      Value *CV = ConstantInt::get(X->getType(), C);
      Value *V = Builder->CreateLShr(X, CV);
      return new TruncInst(V, IIOperand->getType());
    }
    break;
  }

  case Intrinsic::powi:
    if (ConstantInt *Power = dyn_cast<ConstantInt>(II->getArgOperand(1))) {
      // powi(x, 0) -> 1.0
      if (Power->isZero())
        return ReplaceInstUsesWith(CI, ConstantFP::get(CI.getType(), 1.0));
      // powi(x, 1) -> x
      if (Power->isOne())
        return ReplaceInstUsesWith(CI, II->getArgOperand(0));
      // powi(x, -1) -> 1/x
      if (Power->isAllOnesValue())
        return BinaryOperator::CreateFDiv(ConstantFP::get(CI.getType(), 1.0),
                                          II->getArgOperand(0));
    }
    break;
  case Intrinsic::cttz: {
    // If all bits below the first known one are known zero,
    // this value is constant.
    IntegerType *IT = dyn_cast<IntegerType>(II->getArgOperand(0)->getType());
    // FIXME: Try to simplify vectors of integers.
    if (!IT) break;
    uint32_t BitWidth = IT->getBitWidth();
    APInt KnownZero(BitWidth, 0);
    APInt KnownOne(BitWidth, 0);
    computeKnownBits(II->getArgOperand(0), KnownZero, KnownOne, 0, II);
    unsigned TrailingZeros = KnownOne.countTrailingZeros();
    APInt Mask(APInt::getLowBitsSet(BitWidth, TrailingZeros));
    if ((Mask & KnownZero) == Mask)
      return ReplaceInstUsesWith(CI, ConstantInt::get(IT,
                                 APInt(BitWidth, TrailingZeros)));

    }
    break;
  case Intrinsic::ctlz: {
    // If all bits above the first known one are known zero,
    // this value is constant.
    IntegerType *IT = dyn_cast<IntegerType>(II->getArgOperand(0)->getType());
    // FIXME: Try to simplify vectors of integers.
    if (!IT) break;
    uint32_t BitWidth = IT->getBitWidth();
    APInt KnownZero(BitWidth, 0);
    APInt KnownOne(BitWidth, 0);
    computeKnownBits(II->getArgOperand(0), KnownZero, KnownOne, 0, II);
    unsigned LeadingZeros = KnownOne.countLeadingZeros();
    APInt Mask(APInt::getHighBitsSet(BitWidth, LeadingZeros));
    if ((Mask & KnownZero) == Mask)
      return ReplaceInstUsesWith(CI, ConstantInt::get(IT,
                                 APInt(BitWidth, LeadingZeros)));

    }
    break;

  case Intrinsic::uadd_with_overflow:
  case Intrinsic::sadd_with_overflow:
  case Intrinsic::umul_with_overflow:
  case Intrinsic::smul_with_overflow:
    if (isa<Constant>(II->getArgOperand(0)) &&
        !isa<Constant>(II->getArgOperand(1))) {
      // Canonicalize constants into the RHS.
      Value *LHS = II->getArgOperand(0);
      II->setArgOperand(0, II->getArgOperand(1));
      II->setArgOperand(1, LHS);
      return II;
    }
    // fall through

  case Intrinsic::usub_with_overflow:
  case Intrinsic::ssub_with_overflow: {
    OverflowCheckFlavor OCF =
        IntrinsicIDToOverflowCheckFlavor(II->getIntrinsicID());
    assert(OCF != OCF_INVALID && "unexpected!");

    Value *OperationResult = nullptr;
    Constant *OverflowResult = nullptr;
    if (OptimizeOverflowCheck(OCF, II->getArgOperand(0), II->getArgOperand(1),
                              *II, OperationResult, OverflowResult))
      return CreateOverflowTuple(II, OperationResult, OverflowResult);

    break;
  }

  case Intrinsic::minnum:
  case Intrinsic::maxnum: {
    Value *Arg0 = II->getArgOperand(0);
    Value *Arg1 = II->getArgOperand(1);

    // fmin(x, x) -> x
    if (Arg0 == Arg1)
      return ReplaceInstUsesWith(CI, Arg0);

    const ConstantFP *C0 = dyn_cast<ConstantFP>(Arg0);
    const ConstantFP *C1 = dyn_cast<ConstantFP>(Arg1);

    // Canonicalize constants into the RHS.
    if (C0 && !C1) {
      II->setArgOperand(0, Arg1);
      II->setArgOperand(1, Arg0);
      return II;
    }

    // fmin(x, nan) -> x
    if (C1 && C1->isNaN())
      return ReplaceInstUsesWith(CI, Arg0);

    // This is the value because if undef were NaN, we would return the other
    // value and cannot return a NaN unless both operands are.
    //
    // fmin(undef, x) -> x
    if (isa<UndefValue>(Arg0))
      return ReplaceInstUsesWith(CI, Arg1);

    // fmin(x, undef) -> x
    if (isa<UndefValue>(Arg1))
      return ReplaceInstUsesWith(CI, Arg0);

    Value *X = nullptr;
    Value *Y = nullptr;
    if (II->getIntrinsicID() == Intrinsic::minnum) {
      // fmin(x, fmin(x, y)) -> fmin(x, y)
      // fmin(y, fmin(x, y)) -> fmin(x, y)
      if (match(Arg1, m_FMin(m_Value(X), m_Value(Y)))) {
        if (Arg0 == X || Arg0 == Y)
          return ReplaceInstUsesWith(CI, Arg1);
      }

      // fmin(fmin(x, y), x) -> fmin(x, y)
      // fmin(fmin(x, y), y) -> fmin(x, y)
      if (match(Arg0, m_FMin(m_Value(X), m_Value(Y)))) {
        if (Arg1 == X || Arg1 == Y)
          return ReplaceInstUsesWith(CI, Arg0);
      }

      // TODO: fmin(nnan x, inf) -> x
      // TODO: fmin(nnan ninf x, flt_max) -> x
      if (C1 && C1->isInfinity()) {
        // fmin(x, -inf) -> -inf
        if (C1->isNegative())
          return ReplaceInstUsesWith(CI, Arg1);
      }
    } else {
      assert(II->getIntrinsicID() == Intrinsic::maxnum);
      // fmax(x, fmax(x, y)) -> fmax(x, y)
      // fmax(y, fmax(x, y)) -> fmax(x, y)
      if (match(Arg1, m_FMax(m_Value(X), m_Value(Y)))) {
        if (Arg0 == X || Arg0 == Y)
          return ReplaceInstUsesWith(CI, Arg1);
      }

      // fmax(fmax(x, y), x) -> fmax(x, y)
      // fmax(fmax(x, y), y) -> fmax(x, y)
      if (match(Arg0, m_FMax(m_Value(X), m_Value(Y)))) {
        if (Arg1 == X || Arg1 == Y)
          return ReplaceInstUsesWith(CI, Arg0);
      }

      // TODO: fmax(nnan x, -inf) -> x
      // TODO: fmax(nnan ninf x, -flt_max) -> x
      if (C1 && C1->isInfinity()) {
        // fmax(x, inf) -> inf
        if (!C1->isNegative())
          return ReplaceInstUsesWith(CI, Arg1);
      }
    }
    break;
  }
#if 0 // HLSL Change - remove platform intrinsics
  case Intrinsic::ppc_altivec_lvx:
  case Intrinsic::ppc_altivec_lvxl:
    // Turn PPC lvx -> load if the pointer is known aligned.
    if (getOrEnforceKnownAlignment(II->getArgOperand(0), 16, DL, II, AC, DT) >=
        16) {
      Value *Ptr = Builder->CreateBitCast(II->getArgOperand(0),
                                         PointerType::getUnqual(II->getType()));
      return new LoadInst(Ptr);
    }
    break;
  case Intrinsic::ppc_vsx_lxvw4x:
  case Intrinsic::ppc_vsx_lxvd2x: {
    // Turn PPC VSX loads into normal loads.
    Value *Ptr = Builder->CreateBitCast(II->getArgOperand(0),
                                        PointerType::getUnqual(II->getType()));
    return new LoadInst(Ptr, Twine(""), false, 1);
  }
  case Intrinsic::ppc_altivec_stvx:
  case Intrinsic::ppc_altivec_stvxl:
    // Turn stvx -> store if the pointer is known aligned.
    if (getOrEnforceKnownAlignment(II->getArgOperand(1), 16, DL, II, AC, DT) >=
        16) {
      Type *OpPtrTy =
        PointerType::getUnqual(II->getArgOperand(0)->getType());
      Value *Ptr = Builder->CreateBitCast(II->getArgOperand(1), OpPtrTy);
      return new StoreInst(II->getArgOperand(0), Ptr);
    }
    break;
  case Intrinsic::ppc_vsx_stxvw4x:
  case Intrinsic::ppc_vsx_stxvd2x: {
    // Turn PPC VSX stores into normal stores.
    Type *OpPtrTy = PointerType::getUnqual(II->getArgOperand(0)->getType());
    Value *Ptr = Builder->CreateBitCast(II->getArgOperand(1), OpPtrTy);
    return new StoreInst(II->getArgOperand(0), Ptr, false, 1);
  }
  case Intrinsic::ppc_qpx_qvlfs:
    // Turn PPC QPX qvlfs -> load if the pointer is known aligned.
    if (getOrEnforceKnownAlignment(II->getArgOperand(0), 16, DL, II, AC, DT) >=
        16) {
      Type *VTy = VectorType::get(Builder->getFloatTy(),
                                  II->getType()->getVectorNumElements());
      Value *Ptr = Builder->CreateBitCast(II->getArgOperand(0),
                                         PointerType::getUnqual(VTy));
      Value *Load = Builder->CreateLoad(Ptr);
      return new FPExtInst(Load, II->getType());
    }
    break;
  case Intrinsic::ppc_qpx_qvlfd:
    // Turn PPC QPX qvlfd -> load if the pointer is known aligned.
    if (getOrEnforceKnownAlignment(II->getArgOperand(0), 32, DL, II, AC, DT) >=
        32) {
      Value *Ptr = Builder->CreateBitCast(II->getArgOperand(0),
                                         PointerType::getUnqual(II->getType()));
      return new LoadInst(Ptr);
    }
    break;
  case Intrinsic::ppc_qpx_qvstfs:
    // Turn PPC QPX qvstfs -> store if the pointer is known aligned.
    if (getOrEnforceKnownAlignment(II->getArgOperand(1), 16, DL, II, AC, DT) >=
        16) {
      Type *VTy = VectorType::get(Builder->getFloatTy(),
          II->getArgOperand(0)->getType()->getVectorNumElements());
      Value *TOp = Builder->CreateFPTrunc(II->getArgOperand(0), VTy);
      Type *OpPtrTy = PointerType::getUnqual(VTy);
      Value *Ptr = Builder->CreateBitCast(II->getArgOperand(1), OpPtrTy);
      return new StoreInst(TOp, Ptr);
    }
    break;
  case Intrinsic::ppc_qpx_qvstfd:
    // Turn PPC QPX qvstfd -> store if the pointer is known aligned.
    if (getOrEnforceKnownAlignment(II->getArgOperand(1), 32, DL, II, AC, DT) >=
        32) {
      Type *OpPtrTy =
        PointerType::getUnqual(II->getArgOperand(0)->getType());
      Value *Ptr = Builder->CreateBitCast(II->getArgOperand(1), OpPtrTy);
      return new StoreInst(II->getArgOperand(0), Ptr);
    }
    break;
  case Intrinsic::x86_sse_storeu_ps:
  case Intrinsic::x86_sse2_storeu_pd:
  case Intrinsic::x86_sse2_storeu_dq:
    // Turn X86 storeu -> store if the pointer is known aligned.
    if (getOrEnforceKnownAlignment(II->getArgOperand(0), 16, DL, II, AC, DT) >=
        16) {
      Type *OpPtrTy =
        PointerType::getUnqual(II->getArgOperand(1)->getType());
      Value *Ptr = Builder->CreateBitCast(II->getArgOperand(0), OpPtrTy);
      return new StoreInst(II->getArgOperand(1), Ptr);
    }
    break;

  case Intrinsic::x86_sse_cvtss2si:
  case Intrinsic::x86_sse_cvtss2si64:
  case Intrinsic::x86_sse_cvttss2si:
  case Intrinsic::x86_sse_cvttss2si64:
  case Intrinsic::x86_sse2_cvtsd2si:
  case Intrinsic::x86_sse2_cvtsd2si64:
  case Intrinsic::x86_sse2_cvttsd2si:
  case Intrinsic::x86_sse2_cvttsd2si64: {
    // These intrinsics only demand the 0th element of their input vectors. If
    // we can simplify the input based on that, do so now.
    unsigned VWidth =
      cast<VectorType>(II->getArgOperand(0)->getType())->getNumElements();
    APInt DemandedElts(VWidth, 1);
    APInt UndefElts(VWidth, 0);
    if (Value *V = SimplifyDemandedVectorElts(II->getArgOperand(0),
                                              DemandedElts, UndefElts)) {
      II->setArgOperand(0, V);
      return II;
    }
    break;
  }

  // Constant fold <A x Bi> << Ci.
  // FIXME: We don't handle _dq because it's a shift of an i128, but is
  // represented in the IR as <2 x i64>. A per element shift is wrong.
  case Intrinsic::x86_sse2_psll_d:
  case Intrinsic::x86_sse2_psll_q:
  case Intrinsic::x86_sse2_psll_w:
  case Intrinsic::x86_sse2_pslli_d:
  case Intrinsic::x86_sse2_pslli_q:
  case Intrinsic::x86_sse2_pslli_w:
  case Intrinsic::x86_avx2_psll_d:
  case Intrinsic::x86_avx2_psll_q:
  case Intrinsic::x86_avx2_psll_w:
  case Intrinsic::x86_avx2_pslli_d:
  case Intrinsic::x86_avx2_pslli_q:
  case Intrinsic::x86_avx2_pslli_w:
  case Intrinsic::x86_sse2_psrl_d:
  case Intrinsic::x86_sse2_psrl_q:
  case Intrinsic::x86_sse2_psrl_w:
  case Intrinsic::x86_sse2_psrli_d:
  case Intrinsic::x86_sse2_psrli_q:
  case Intrinsic::x86_sse2_psrli_w:
  case Intrinsic::x86_avx2_psrl_d:
  case Intrinsic::x86_avx2_psrl_q:
  case Intrinsic::x86_avx2_psrl_w:
  case Intrinsic::x86_avx2_psrli_d:
  case Intrinsic::x86_avx2_psrli_q:
  case Intrinsic::x86_avx2_psrli_w: {
    // Simplify if count is constant. To 0 if >= BitWidth,
    // otherwise to shl/lshr.
    auto CDV = dyn_cast<ConstantDataVector>(II->getArgOperand(1));
    auto CInt = dyn_cast<ConstantInt>(II->getArgOperand(1));
    if (!CDV && !CInt)
      break;
    ConstantInt *Count;
    if (CDV)
      Count = cast<ConstantInt>(CDV->getElementAsConstant(0));
    else
      Count = CInt;

    auto Vec = II->getArgOperand(0);
    auto VT = cast<VectorType>(Vec->getType());
    if (Count->getZExtValue() >
        VT->getElementType()->getPrimitiveSizeInBits() - 1)
      return ReplaceInstUsesWith(
          CI, ConstantAggregateZero::get(Vec->getType()));

    bool isPackedShiftLeft = true;
    switch (II->getIntrinsicID()) {
    default : break;
    case Intrinsic::x86_sse2_psrl_d:
    case Intrinsic::x86_sse2_psrl_q:
    case Intrinsic::x86_sse2_psrl_w:
    case Intrinsic::x86_sse2_psrli_d:
    case Intrinsic::x86_sse2_psrli_q:
    case Intrinsic::x86_sse2_psrli_w:
    case Intrinsic::x86_avx2_psrl_d:
    case Intrinsic::x86_avx2_psrl_q:
    case Intrinsic::x86_avx2_psrl_w:
    case Intrinsic::x86_avx2_psrli_d:
    case Intrinsic::x86_avx2_psrli_q:
    case Intrinsic::x86_avx2_psrli_w: isPackedShiftLeft = false; break;
    }

    unsigned VWidth = VT->getNumElements();
    // Get a constant vector of the same type as the first operand.
    auto VTCI = ConstantInt::get(VT->getElementType(), Count->getZExtValue());
    if (isPackedShiftLeft)
      return BinaryOperator::CreateShl(Vec,
          Builder->CreateVectorSplat(VWidth, VTCI));

    return BinaryOperator::CreateLShr(Vec,
        Builder->CreateVectorSplat(VWidth, VTCI));
  }

  case Intrinsic::x86_sse41_pmovsxbw:
  case Intrinsic::x86_sse41_pmovsxwd:
  case Intrinsic::x86_sse41_pmovsxdq:
  case Intrinsic::x86_sse41_pmovzxbw:
  case Intrinsic::x86_sse41_pmovzxwd:
  case Intrinsic::x86_sse41_pmovzxdq: {
    // pmov{s|z}x ignores the upper half of their input vectors.
    unsigned VWidth =
      cast<VectorType>(II->getArgOperand(0)->getType())->getNumElements();
    unsigned LowHalfElts = VWidth / 2;
    APInt InputDemandedElts(APInt::getBitsSet(VWidth, 0, LowHalfElts));
    APInt UndefElts(VWidth, 0);
    if (Value *TmpV = SimplifyDemandedVectorElts(
            II->getArgOperand(0), InputDemandedElts, UndefElts)) {
      II->setArgOperand(0, TmpV);
      return II;
    }
    break;
  }
  case Intrinsic::x86_sse41_insertps:
    if (Value *V = SimplifyX86insertps(*II, *Builder))
      return ReplaceInstUsesWith(*II, V);
    break;
    
  case Intrinsic::x86_sse4a_insertqi: {
    // insertqi x, y, 64, 0 can just copy y's lower bits and leave the top
    // ones undef
    // TODO: eventually we should lower this intrinsic to IR
    if (auto CIWidth = dyn_cast<ConstantInt>(II->getArgOperand(2))) {
      if (auto CIStart = dyn_cast<ConstantInt>(II->getArgOperand(3))) {
        unsigned Index = CIStart->getZExtValue();
        // From AMD documentation: "a value of zero in the field length is
        // defined as length of 64".
        unsigned Length = CIWidth->equalsInt(0) ? 64 : CIWidth->getZExtValue();

        // From AMD documentation: "If the sum of the bit index + length field
        // is greater than 64, the results are undefined".

        // Note that both field index and field length are 8-bit quantities.
        // Since variables 'Index' and 'Length' are unsigned values
        // obtained from zero-extending field index and field length
        // respectively, their sum should never wrap around.
        if ((Index + Length) > 64)
          return ReplaceInstUsesWith(CI, UndefValue::get(II->getType()));

        if (Length == 64 && Index == 0) {
          Value *Vec = II->getArgOperand(1);
          Value *Undef = UndefValue::get(Vec->getType());
          const uint32_t Mask[] = { 0, 2 };
          return ReplaceInstUsesWith(
              CI,
              Builder->CreateShuffleVector(
                  Vec, Undef, ConstantDataVector::get(
                                  II->getContext(), makeArrayRef(Mask))));

        } else if (auto Source =
                       dyn_cast<IntrinsicInst>(II->getArgOperand(0))) {
          if (Source->hasOneUse() &&
              Source->getArgOperand(1) == II->getArgOperand(1)) {
            // If the source of the insert has only one use and it's another
            // insert (and they're both inserting from the same vector), try to
            // bundle both together.
            auto CISourceWidth =
                dyn_cast<ConstantInt>(Source->getArgOperand(2));
            auto CISourceStart =
                dyn_cast<ConstantInt>(Source->getArgOperand(3));
            if (CISourceStart && CISourceWidth) {
              unsigned Start = CIStart->getZExtValue();
              unsigned Width = CIWidth->getZExtValue();
              unsigned End = Start + Width;
              unsigned SourceStart = CISourceStart->getZExtValue();
              unsigned SourceWidth = CISourceWidth->getZExtValue();
              unsigned SourceEnd = SourceStart + SourceWidth;
              unsigned NewStart, NewWidth;
              bool ShouldReplace = false;
              if (Start <= SourceStart && SourceStart <= End) {
                NewStart = Start;
                NewWidth = std::max(End, SourceEnd) - NewStart;
                ShouldReplace = true;
              } else if (SourceStart <= Start && Start <= SourceEnd) {
                NewStart = SourceStart;
                NewWidth = std::max(SourceEnd, End) - NewStart;
                ShouldReplace = true;
              }

              if (ShouldReplace) {
                Constant *ConstantWidth = ConstantInt::get(
                    II->getArgOperand(2)->getType(), NewWidth, false);
                Constant *ConstantStart = ConstantInt::get(
                    II->getArgOperand(3)->getType(), NewStart, false);
                Value *Args[4] = { Source->getArgOperand(0),
                                   II->getArgOperand(1), ConstantWidth,
                                   ConstantStart };
                Module *M = CI.getParent()->getParent()->getParent();
                Value *F =
                    Intrinsic::getDeclaration(M, Intrinsic::x86_sse4a_insertqi);
                return ReplaceInstUsesWith(CI, Builder->CreateCall(F, Args));
              }
            }
          }
        }
      }
    }
    break;
  }

  case Intrinsic::x86_sse41_pblendvb:
  case Intrinsic::x86_sse41_blendvps:
  case Intrinsic::x86_sse41_blendvpd:
  case Intrinsic::x86_avx_blendv_ps_256:
  case Intrinsic::x86_avx_blendv_pd_256:
  case Intrinsic::x86_avx2_pblendvb: {
    // Convert blendv* to vector selects if the mask is constant.
    // This optimization is convoluted because the intrinsic is defined as
    // getting a vector of floats or doubles for the ps and pd versions.
    // FIXME: That should be changed.
    Value *Mask = II->getArgOperand(2);
    if (auto C = dyn_cast<ConstantDataVector>(Mask)) {
      auto Tyi1 = Builder->getInt1Ty();
      auto SelectorType = cast<VectorType>(Mask->getType());
      auto EltTy = SelectorType->getElementType();
      unsigned Size = SelectorType->getNumElements();
      unsigned BitWidth =
          EltTy->isFloatTy()
              ? 32
              : (EltTy->isDoubleTy() ? 64 : EltTy->getIntegerBitWidth());
      assert((BitWidth == 64 || BitWidth == 32 || BitWidth == 8) &&
             "Wrong arguments for variable blend intrinsic");
      SmallVector<Constant *, 32> Selectors;
      for (unsigned I = 0; I < Size; ++I) {
        // The intrinsics only read the top bit
        uint64_t Selector;
        if (BitWidth == 8)
          Selector = C->getElementAsInteger(I);
        else
          Selector = C->getElementAsAPFloat(I).bitcastToAPInt().getZExtValue();
        Selectors.push_back(ConstantInt::get(Tyi1, Selector >> (BitWidth - 1)));
      }
      auto NewSelector = ConstantVector::get(Selectors);
      return SelectInst::Create(NewSelector, II->getArgOperand(1),
                                II->getArgOperand(0), "blendv");
    } else {
      break;
    }
  }

  case Intrinsic::x86_avx_vpermilvar_ps:
  case Intrinsic::x86_avx_vpermilvar_ps_256:
  case Intrinsic::x86_avx_vpermilvar_pd:
  case Intrinsic::x86_avx_vpermilvar_pd_256: {
    // Convert vpermil* to shufflevector if the mask is constant.
    Value *V = II->getArgOperand(1);
    unsigned Size = cast<VectorType>(V->getType())->getNumElements();
    assert(Size == 8 || Size == 4 || Size == 2);
    uint32_t Indexes[8];
    if (auto C = dyn_cast<ConstantDataVector>(V)) {
      // The intrinsics only read one or two bits, clear the rest.
      for (unsigned I = 0; I < Size; ++I) {
        uint32_t Index = C->getElementAsInteger(I) & 0x3;
        if (II->getIntrinsicID() == Intrinsic::x86_avx_vpermilvar_pd ||
            II->getIntrinsicID() == Intrinsic::x86_avx_vpermilvar_pd_256)
          Index >>= 1;
        Indexes[I] = Index;
      }
    } else if (isa<ConstantAggregateZero>(V)) {
      for (unsigned I = 0; I < Size; ++I)
        Indexes[I] = 0;
    } else {
      break;
    }
    // The _256 variants are a bit trickier since the mask bits always index
    // into the corresponding 128 half. In order to convert to a generic
    // shuffle, we have to make that explicit.
    if (II->getIntrinsicID() == Intrinsic::x86_avx_vpermilvar_ps_256 ||
        II->getIntrinsicID() == Intrinsic::x86_avx_vpermilvar_pd_256) {
      for (unsigned I = Size / 2; I < Size; ++I)
        Indexes[I] += Size / 2;
    }
    auto NewC =
        ConstantDataVector::get(V->getContext(), makeArrayRef(Indexes, Size));
    auto V1 = II->getArgOperand(0);
    auto V2 = UndefValue::get(V1->getType());
    auto Shuffle = Builder->CreateShuffleVector(V1, V2, NewC);
    return ReplaceInstUsesWith(CI, Shuffle);
  }

  case Intrinsic::x86_avx_vperm2f128_pd_256:
  case Intrinsic::x86_avx_vperm2f128_ps_256:
  case Intrinsic::x86_avx_vperm2f128_si_256:
  case Intrinsic::x86_avx2_vperm2i128:
    if (Value *V = SimplifyX86vperm2(*II, *Builder))
      return ReplaceInstUsesWith(*II, V);
    break;

  case Intrinsic::ppc_altivec_vperm:
    // Turn vperm(V1,V2,mask) -> shuffle(V1,V2,mask) if mask is a constant.
    // Note that ppc_altivec_vperm has a big-endian bias, so when creating
    // a vectorshuffle for little endian, we must undo the transformation
    // performed on vec_perm in altivec.h.  That is, we must complement
    // the permutation mask with respect to 31 and reverse the order of
    // V1 and V2.
    if (Constant *Mask = dyn_cast<Constant>(II->getArgOperand(2))) {
      assert(Mask->getType()->getVectorNumElements() == 16 &&
             "Bad type for intrinsic!");

      // Check that all of the elements are integer constants or undefs.
      bool AllEltsOk = true;
      for (unsigned i = 0; i != 16; ++i) {
        Constant *Elt = Mask->getAggregateElement(i);
        if (!Elt || !(isa<ConstantInt>(Elt) || isa<UndefValue>(Elt))) {
          AllEltsOk = false;
          break;
        }
      }

      if (AllEltsOk) {
        // Cast the input vectors to byte vectors.
        Value *Op0 = Builder->CreateBitCast(II->getArgOperand(0),
                                            Mask->getType());
        Value *Op1 = Builder->CreateBitCast(II->getArgOperand(1),
                                            Mask->getType());
        Value *Result = UndefValue::get(Op0->getType());

        // Only extract each element once.
        Value *ExtractedElts[32];
        memset(ExtractedElts, 0, sizeof(ExtractedElts));

        for (unsigned i = 0; i != 16; ++i) {
          if (isa<UndefValue>(Mask->getAggregateElement(i)))
            continue;
          unsigned Idx =
            cast<ConstantInt>(Mask->getAggregateElement(i))->getZExtValue();
          Idx &= 31;  // Match the hardware behavior.
          if (DL.isLittleEndian())
            Idx = 31 - Idx;

          if (!ExtractedElts[Idx]) {
            Value *Op0ToUse = (DL.isLittleEndian()) ? Op1 : Op0;
            Value *Op1ToUse = (DL.isLittleEndian()) ? Op0 : Op1;
            ExtractedElts[Idx] =
              Builder->CreateExtractElement(Idx < 16 ? Op0ToUse : Op1ToUse,
                                            Builder->getInt32(Idx&15));
          }

          // Insert this value into the result vector.
          Result = Builder->CreateInsertElement(Result, ExtractedElts[Idx],
                                                Builder->getInt32(i));
        }
        return CastInst::Create(Instruction::BitCast, Result, CI.getType());
      }
    }
    break;

  case Intrinsic::arm_neon_vld1:
  case Intrinsic::arm_neon_vld2:
  case Intrinsic::arm_neon_vld3:
  case Intrinsic::arm_neon_vld4:
  case Intrinsic::arm_neon_vld2lane:
  case Intrinsic::arm_neon_vld3lane:
  case Intrinsic::arm_neon_vld4lane:
  case Intrinsic::arm_neon_vst1:
  case Intrinsic::arm_neon_vst2:
  case Intrinsic::arm_neon_vst3:
  case Intrinsic::arm_neon_vst4:
  case Intrinsic::arm_neon_vst2lane:
  case Intrinsic::arm_neon_vst3lane:
  case Intrinsic::arm_neon_vst4lane: {
    unsigned MemAlign = getKnownAlignment(II->getArgOperand(0), DL, II, AC, DT);
    unsigned AlignArg = II->getNumArgOperands() - 1;
    ConstantInt *IntrAlign = dyn_cast<ConstantInt>(II->getArgOperand(AlignArg));
    if (IntrAlign && IntrAlign->getZExtValue() < MemAlign) {
      II->setArgOperand(AlignArg,
                        ConstantInt::get(Type::getInt32Ty(II->getContext()),
                                         MemAlign, false));
      return II;
    }
    break;
  }

  case Intrinsic::arm_neon_vmulls:
  case Intrinsic::arm_neon_vmullu:
  case Intrinsic::aarch64_neon_smull:
  case Intrinsic::aarch64_neon_umull: {
    Value *Arg0 = II->getArgOperand(0);
    Value *Arg1 = II->getArgOperand(1);

    // Handle mul by zero first:
    if (isa<ConstantAggregateZero>(Arg0) || isa<ConstantAggregateZero>(Arg1)) {
      return ReplaceInstUsesWith(CI, ConstantAggregateZero::get(II->getType()));
    }

    // Check for constant LHS & RHS - in this case we just simplify.
    bool Zext = (II->getIntrinsicID() == Intrinsic::arm_neon_vmullu ||
                 II->getIntrinsicID() == Intrinsic::aarch64_neon_umull);
    VectorType *NewVT = cast<VectorType>(II->getType());
    if (Constant *CV0 = dyn_cast<Constant>(Arg0)) {
      if (Constant *CV1 = dyn_cast<Constant>(Arg1)) {
        CV0 = ConstantExpr::getIntegerCast(CV0, NewVT, /*isSigned=*/!Zext);
        CV1 = ConstantExpr::getIntegerCast(CV1, NewVT, /*isSigned=*/!Zext);

        return ReplaceInstUsesWith(CI, ConstantExpr::getMul(CV0, CV1));
      }

      // Couldn't simplify - canonicalize constant to the RHS.
      std::swap(Arg0, Arg1);
    }

    // Handle mul by one:
    if (Constant *CV1 = dyn_cast<Constant>(Arg1))
      if (ConstantInt *Splat =
              dyn_cast_or_null<ConstantInt>(CV1->getSplatValue()))
        if (Splat->isOne())
          return CastInst::CreateIntegerCast(Arg0, II->getType(),
                                             /*isSigned=*/!Zext);

    break;
  }

  case Intrinsic::AMDGPU_rcp: {
    if (const ConstantFP *C = dyn_cast<ConstantFP>(II->getArgOperand(0))) {
      const APFloat &ArgVal = C->getValueAPF();
      APFloat Val(ArgVal.getSemantics(), 1.0);
      APFloat::opStatus Status = Val.divide(ArgVal,
                                            APFloat::rmNearestTiesToEven);
      // Only do this if it was exact and therefore not dependent on the
      // rounding mode.
      if (Status == APFloat::opOK)
        return ReplaceInstUsesWith(CI, ConstantFP::get(II->getContext(), Val));
    }

    break;
  }
#endif // HLSL Change - remove platform intrinsics
  case Intrinsic::stackrestore: {
    // If the save is right next to the restore, remove the restore.  This can
    // happen when variable allocas are DCE'd.
    if (IntrinsicInst *SS = dyn_cast<IntrinsicInst>(II->getArgOperand(0))) {
      if (SS->getIntrinsicID() == Intrinsic::stacksave) {
        BasicBlock::iterator BI = SS;
        if (&*++BI == II)
          return EraseInstFromFunction(CI);
      }
    }

    // Scan down this block to see if there is another stack restore in the
    // same block without an intervening call/alloca.
    BasicBlock::iterator BI = II;
    TerminatorInst *TI = II->getParent()->getTerminator();
    bool CannotRemove = false;
    for (++BI; &*BI != TI; ++BI) {
      if (isa<AllocaInst>(BI)) {
        CannotRemove = true;
        break;
      }
      if (CallInst *BCI = dyn_cast<CallInst>(BI)) {
        if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(BCI)) {
          // If there is a stackrestore below this one, remove this one.
          if (II->getIntrinsicID() == Intrinsic::stackrestore)
            return EraseInstFromFunction(CI);
          // Otherwise, ignore the intrinsic.
        } else {
          // If we found a non-intrinsic call, we can't remove the stack
          // restore.
          CannotRemove = true;
          break;
        }
      }
    }

    // If the stack restore is in a return, resume, or unwind block and if there
    // are no allocas or calls between the restore and the return, nuke the
    // restore.
    if (!CannotRemove && (isa<ReturnInst>(TI) || isa<ResumeInst>(TI)))
      return EraseInstFromFunction(CI);
    break;
  }
  case Intrinsic::assume: {
    // Canonicalize assume(a && b) -> assume(a); assume(b);
    // Note: New assumption intrinsics created here are registered by
    // the InstCombineIRInserter object.
    Value *IIOperand = II->getArgOperand(0), *A, *B,
          *AssumeIntrinsic = II->getCalledValue();
    if (match(IIOperand, m_And(m_Value(A), m_Value(B)))) {
      Builder->CreateCall(AssumeIntrinsic, A, II->getName());
      Builder->CreateCall(AssumeIntrinsic, B, II->getName());
      return EraseInstFromFunction(*II);
    }
    // assume(!(a || b)) -> assume(!a); assume(!b);
    if (match(IIOperand, m_Not(m_Or(m_Value(A), m_Value(B))))) {
      Builder->CreateCall(AssumeIntrinsic, Builder->CreateNot(A),
                          II->getName());
      Builder->CreateCall(AssumeIntrinsic, Builder->CreateNot(B),
                          II->getName());
      return EraseInstFromFunction(*II);
    }

    // assume( (load addr) != null ) -> add 'nonnull' metadata to load
    // (if assume is valid at the load)
    if (ICmpInst* ICmp = dyn_cast<ICmpInst>(IIOperand)) {
      Value *LHS = ICmp->getOperand(0);
      Value *RHS = ICmp->getOperand(1);
      if (ICmpInst::ICMP_NE == ICmp->getPredicate() &&
          isa<LoadInst>(LHS) &&
          isa<Constant>(RHS) &&
          RHS->getType()->isPointerTy() &&
          cast<Constant>(RHS)->isNullValue()) {
        LoadInst* LI = cast<LoadInst>(LHS);
        if (isValidAssumeForContext(II, LI, DT)) {
          MDNode *MD = MDNode::get(II->getContext(), None);
          LI->setMetadata(LLVMContext::MD_nonnull, MD);
          return EraseInstFromFunction(*II);
        }
      }
      // TODO: apply nonnull return attributes to calls and invokes
      // TODO: apply range metadata for range check patterns?
    }
    // If there is a dominating assume with the same condition as this one,
    // then this one is redundant, and should be removed.
    APInt KnownZero(1, 0), KnownOne(1, 0);
    computeKnownBits(IIOperand, KnownZero, KnownOne, 0, II);
    if (KnownOne.isAllOnesValue())
      return EraseInstFromFunction(*II);

    break;
  }
  case Intrinsic::experimental_gc_relocate: {
    // Translate facts known about a pointer before relocating into
    // facts about the relocate value, while being careful to
    // preserve relocation semantics.
    GCRelocateOperands Operands(II);
    Value *DerivedPtr = Operands.getDerivedPtr();
    auto *GCRelocateType = cast<PointerType>(II->getType());

    // Remove the relocation if unused, note that this check is required
    // to prevent the cases below from looping forever.
    if (II->use_empty())
      return EraseInstFromFunction(*II);

    // Undef is undef, even after relocation.
    // TODO: provide a hook for this in GCStrategy.  This is clearly legal for
    // most practical collectors, but there was discussion in the review thread
    // about whether it was legal for all possible collectors.
    if (isa<UndefValue>(DerivedPtr)) {
      // gc_relocate is uncasted. Use undef of gc_relocate's type to replace it.
      return ReplaceInstUsesWith(*II, UndefValue::get(GCRelocateType));
    }

    // The relocation of null will be null for most any collector.
    // TODO: provide a hook for this in GCStrategy.  There might be some weird
    // collector this property does not hold for.
    if (isa<ConstantPointerNull>(DerivedPtr)) {
      // gc_relocate is uncasted. Use null-pointer of gc_relocate's type to replace it.
      return ReplaceInstUsesWith(*II, ConstantPointerNull::get(GCRelocateType));
    }

    // isKnownNonNull -> nonnull attribute
    if (isKnownNonNull(DerivedPtr))
      II->addAttribute(AttributeSet::ReturnIndex, Attribute::NonNull);

    // isDereferenceablePointer -> deref attribute
    if (isDereferenceablePointer(DerivedPtr, DL)) {
      if (Argument *A = dyn_cast<Argument>(DerivedPtr)) {
        uint64_t Bytes = A->getDereferenceableBytes();
        II->addDereferenceableAttr(AttributeSet::ReturnIndex, Bytes);
      }
    }

    // TODO: bitcast(relocate(p)) -> relocate(bitcast(p))
    // Canonicalize on the type from the uses to the defs

    // TODO: relocate((gep p, C, C2, ...)) -> gep(relocate(p), C, C2, ...)
  }
  }

  return visitCallSite(II);
}

// InvokeInst simplification
//
Instruction *InstCombiner::visitInvokeInst(InvokeInst &II) {
  return visitCallSite(&II);
}

/// isSafeToEliminateVarargsCast - If this cast does not affect the value
/// passed through the varargs area, we can eliminate the use of the cast.
static bool isSafeToEliminateVarargsCast(const CallSite CS,
                                         const DataLayout &DL,
                                         const CastInst *const CI,
                                         const int ix) {
  if (!CI->isLosslessCast())
    return false;

  // If this is a GC intrinsic, avoid munging types.  We need types for
  // statepoint reconstruction in SelectionDAG.
  // TODO: This is probably something which should be expanded to all
  // intrinsics since the entire point of intrinsics is that
  // they are understandable by the optimizer.
  if (isStatepoint(CS) || isGCRelocate(CS) || isGCResult(CS))
    return false;

  // The size of ByVal or InAlloca arguments is derived from the type, so we
  // can't change to a type with a different size.  If the size were
  // passed explicitly we could avoid this check.
  if (!CS.isByValOrInAllocaArgument(ix))
    return true;

  Type* SrcTy =
            cast<PointerType>(CI->getOperand(0)->getType())->getElementType();
  Type* DstTy = cast<PointerType>(CI->getType())->getElementType();
  if (!SrcTy->isSized() || !DstTy->isSized())
    return false;
  if (DL.getTypeAllocSize(SrcTy) != DL.getTypeAllocSize(DstTy))
    return false;
  return true;
}

// Try to fold some different type of calls here.
// Currently we're only working with the checking functions, memcpy_chk,
// mempcpy_chk, memmove_chk, memset_chk, strcpy_chk, stpcpy_chk, strncpy_chk,
// strcat_chk and strncat_chk.
Instruction *InstCombiner::tryOptimizeCall(CallInst *CI) {
  if (!CI->getCalledFunction()) return nullptr;

  auto InstCombineRAUW = [this](Instruction *From, Value *With) {
    ReplaceInstUsesWith(*From, With);
  };
  LibCallSimplifier Simplifier(DL, TLI, InstCombineRAUW);
  if (Value *With = Simplifier.optimizeCall(CI)) {
    ++NumSimplified;
    return CI->use_empty() ? CI : ReplaceInstUsesWith(*CI, With);
  }

  return nullptr;
}

static IntrinsicInst *FindInitTrampolineFromAlloca(Value *TrampMem) {
  // Strip off at most one level of pointer casts, looking for an alloca.  This
  // is good enough in practice and simpler than handling any number of casts.
  Value *Underlying = TrampMem->stripPointerCasts();
  if (Underlying != TrampMem &&
      (!Underlying->hasOneUse() || Underlying->user_back() != TrampMem))
    return nullptr;
  if (!isa<AllocaInst>(Underlying))
    return nullptr;

  IntrinsicInst *InitTrampoline = nullptr;
  for (User *U : TrampMem->users()) {
    IntrinsicInst *II = dyn_cast<IntrinsicInst>(U);
    if (!II)
      return nullptr;
    if (II->getIntrinsicID() == Intrinsic::init_trampoline) {
      if (InitTrampoline)
        // More than one init_trampoline writes to this value.  Give up.
        return nullptr;
      InitTrampoline = II;
      continue;
    }
    if (II->getIntrinsicID() == Intrinsic::adjust_trampoline)
      // Allow any number of calls to adjust.trampoline.
      continue;
    return nullptr;
  }

  // No call to init.trampoline found.
  if (!InitTrampoline)
    return nullptr;

  // Check that the alloca is being used in the expected way.
  if (InitTrampoline->getOperand(0) != TrampMem)
    return nullptr;

  return InitTrampoline;
}

static IntrinsicInst *FindInitTrampolineFromBB(IntrinsicInst *AdjustTramp,
                                               Value *TrampMem) {
  // Visit all the previous instructions in the basic block, and try to find a
  // init.trampoline which has a direct path to the adjust.trampoline.
  for (BasicBlock::iterator I = AdjustTramp,
       E = AdjustTramp->getParent()->begin(); I != E; ) {
    Instruction *Inst = --I;
    if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(I))
      if (II->getIntrinsicID() == Intrinsic::init_trampoline &&
          II->getOperand(0) == TrampMem)
        return II;
    if (Inst->mayWriteToMemory())
      return nullptr;
  }
  return nullptr;
}

// Given a call to llvm.adjust.trampoline, find and return the corresponding
// call to llvm.init.trampoline if the call to the trampoline can be optimized
// to a direct call to a function.  Otherwise return NULL.
//
static IntrinsicInst *FindInitTrampoline(Value *Callee) {
  Callee = Callee->stripPointerCasts();
  IntrinsicInst *AdjustTramp = dyn_cast<IntrinsicInst>(Callee);
  if (!AdjustTramp ||
      AdjustTramp->getIntrinsicID() != Intrinsic::adjust_trampoline)
    return nullptr;

  Value *TrampMem = AdjustTramp->getOperand(0);

  if (IntrinsicInst *IT = FindInitTrampolineFromAlloca(TrampMem))
    return IT;
  if (IntrinsicInst *IT = FindInitTrampolineFromBB(AdjustTramp, TrampMem))
    return IT;
  return nullptr;
}

// visitCallSite - Improvements for call and invoke instructions.
//
Instruction *InstCombiner::visitCallSite(CallSite CS) {

  if (isAllocLikeFn(CS.getInstruction(), TLI))
    return visitAllocSite(*CS.getInstruction());

  bool Changed = false;

  // Mark any parameters that are known to be non-null with the nonnull
  // attribute.  This is helpful for inlining calls to functions with null
  // checks on their arguments.
  unsigned ArgNo = 0;
  for (Value *V : CS.args()) {
    if (!CS.paramHasAttr(ArgNo+1, Attribute::NonNull) &&
        isKnownNonNull(V)) {
      AttributeSet AS = CS.getAttributes();
      AS = AS.addAttribute(CS.getInstruction()->getContext(), ArgNo+1,
                           Attribute::NonNull);
      CS.setAttributes(AS);
      Changed = true;
    }
    ArgNo++;
  }
  assert(ArgNo == CS.arg_size() && "sanity check");

  // If the callee is a pointer to a function, attempt to move any casts to the
  // arguments of the call/invoke.
  Value *Callee = CS.getCalledValue();
  if (!isa<Function>(Callee) && transformConstExprCastCall(CS))
    return nullptr;

  if (Function *CalleeF = dyn_cast<Function>(Callee))
    // If the call and callee calling conventions don't match, this call must
    // be unreachable, as the call is undefined.
    if (CalleeF->getCallingConv() != CS.getCallingConv() &&
        // Only do this for calls to a function with a body.  A prototype may
        // not actually end up matching the implementation's calling conv for a
        // variety of reasons (e.g. it may be written in assembly).
        !CalleeF->isDeclaration()) {
      Instruction *OldCall = CS.getInstruction();
      new StoreInst(ConstantInt::getTrue(Callee->getContext()),
                UndefValue::get(Type::getInt1PtrTy(Callee->getContext())),
                                  OldCall);
      // If OldCall does not return void then replaceAllUsesWith undef.
      // This allows ValueHandlers and custom metadata to adjust itself.
      if (!OldCall->getType()->isVoidTy())
        ReplaceInstUsesWith(*OldCall, UndefValue::get(OldCall->getType()));
      if (isa<CallInst>(OldCall))
        return EraseInstFromFunction(*OldCall);

      // We cannot remove an invoke, because it would change the CFG, just
      // change the callee to a null pointer.
      cast<InvokeInst>(OldCall)->setCalledFunction(
                                    Constant::getNullValue(CalleeF->getType()));
      return nullptr;
    }

  if (isa<ConstantPointerNull>(Callee) || isa<UndefValue>(Callee)) {
    // If CS does not return void then replaceAllUsesWith undef.
    // This allows ValueHandlers and custom metadata to adjust itself.
    if (!CS.getInstruction()->getType()->isVoidTy())
      ReplaceInstUsesWith(*CS.getInstruction(),
                          UndefValue::get(CS.getInstruction()->getType()));

    if (isa<InvokeInst>(CS.getInstruction())) {
      // Can't remove an invoke because we cannot change the CFG.
      return nullptr;
    }

    // This instruction is not reachable, just remove it.  We insert a store to
    // undef so that we know that this code is not reachable, despite the fact
    // that we can't modify the CFG here.
    new StoreInst(ConstantInt::getTrue(Callee->getContext()),
                  UndefValue::get(Type::getInt1PtrTy(Callee->getContext())),
                  CS.getInstruction());

    return EraseInstFromFunction(*CS.getInstruction());
  }

  if (IntrinsicInst *II = FindInitTrampoline(Callee))
    return transformCallThroughTrampoline(CS, II);

  PointerType *PTy = cast<PointerType>(Callee->getType());
  FunctionType *FTy = cast<FunctionType>(PTy->getElementType());
  if (FTy->isVarArg()) {
    int ix = FTy->getNumParams();
    // See if we can optimize any arguments passed through the varargs area of
    // the call.
    for (CallSite::arg_iterator I = CS.arg_begin() + FTy->getNumParams(),
           E = CS.arg_end(); I != E; ++I, ++ix) {
      CastInst *CI = dyn_cast<CastInst>(*I);
      if (CI && isSafeToEliminateVarargsCast(CS, DL, CI, ix)) {
        *I = CI->getOperand(0);
        Changed = true;
      }
    }
  }

  if (isa<InlineAsm>(Callee) && !CS.doesNotThrow()) {
    // Inline asm calls cannot throw - mark them 'nounwind'.
    CS.setDoesNotThrow();
    Changed = true;
  }

  // Try to optimize the call if possible, we require DataLayout for most of
  // this.  None of these calls are seen as possibly dead so go ahead and
  // delete the instruction now.
  if (CallInst *CI = dyn_cast<CallInst>(CS.getInstruction())) {
    Instruction *I = tryOptimizeCall(CI);
    // If we changed something return the result, etc. Otherwise let
    // the fallthrough check.
    if (I) return EraseInstFromFunction(*I);
  }

  return Changed ? CS.getInstruction() : nullptr;
}

// transformConstExprCastCall - If the callee is a constexpr cast of a function,
// attempt to move the cast to the arguments of the call/invoke.
//
bool InstCombiner::transformConstExprCastCall(CallSite CS) {
  Function *Callee =
    dyn_cast<Function>(CS.getCalledValue()->stripPointerCasts());
  if (!Callee)
    return false;
  // The prototype of thunks are a lie, don't try to directly call such
  // functions.
  if (Callee->hasFnAttribute("thunk"))
    return false;
  Instruction *Caller = CS.getInstruction();
  const AttributeSet &CallerPAL = CS.getAttributes();

  // Okay, this is a cast from a function to a different type.  Unless doing so
  // would cause a type conversion of one of our arguments, change this call to
  // be a direct call with arguments casted to the appropriate types.
  //
  FunctionType *FT = Callee->getFunctionType();
  Type *OldRetTy = Caller->getType();
  Type *NewRetTy = FT->getReturnType();

  // Check to see if we are changing the return type...
  if (OldRetTy != NewRetTy) {

    if (NewRetTy->isStructTy())
      return false; // TODO: Handle multiple return values.

    if (!CastInst::isBitOrNoopPointerCastable(NewRetTy, OldRetTy, DL)) {
      if (Callee->isDeclaration())
        return false;   // Cannot transform this return value.

      if (!Caller->use_empty() &&
          // void -> non-void is handled specially
          !NewRetTy->isVoidTy())
        return false;   // Cannot transform this return value.
    }

    if (!CallerPAL.isEmpty() && !Caller->use_empty()) {
      AttrBuilder RAttrs(CallerPAL, AttributeSet::ReturnIndex);
      if (RAttrs.overlaps(AttributeFuncs::typeIncompatible(NewRetTy)))
        return false;   // Attribute not compatible with transformed value.
    }

    // If the callsite is an invoke instruction, and the return value is used by
    // a PHI node in a successor, we cannot change the return type of the call
    // because there is no place to put the cast instruction (without breaking
    // the critical edge).  Bail out in this case.
    if (!Caller->use_empty())
      if (InvokeInst *II = dyn_cast<InvokeInst>(Caller))
        for (User *U : II->users())
          if (PHINode *PN = dyn_cast<PHINode>(U))
            if (PN->getParent() == II->getNormalDest() ||
                PN->getParent() == II->getUnwindDest())
              return false;
  }

  unsigned NumActualArgs = CS.arg_size();
  unsigned NumCommonArgs = std::min(FT->getNumParams(), NumActualArgs);

  // Prevent us turning:
  // declare void @takes_i32_inalloca(i32* inalloca)
  //  call void bitcast (void (i32*)* @takes_i32_inalloca to void (i32)*)(i32 0)
  //
  // into:
  //  call void @takes_i32_inalloca(i32* null)
  //
  //  Similarly, avoid folding away bitcasts of byval calls.
  if (Callee->getAttributes().hasAttrSomewhere(Attribute::InAlloca) ||
      Callee->getAttributes().hasAttrSomewhere(Attribute::ByVal))
    return false;

  CallSite::arg_iterator AI = CS.arg_begin();
  for (unsigned i = 0, e = NumCommonArgs; i != e; ++i, ++AI) {
    Type *ParamTy = FT->getParamType(i);
    Type *ActTy = (*AI)->getType();

    if (!CastInst::isBitOrNoopPointerCastable(ActTy, ParamTy, DL))
      return false;   // Cannot transform this parameter value.

    if (AttrBuilder(CallerPAL.getParamAttributes(i + 1), i + 1).
          overlaps(AttributeFuncs::typeIncompatible(ParamTy)))
      return false;   // Attribute not compatible with transformed value.

    if (CS.isInAllocaArgument(i))
      return false;   // Cannot transform to and from inalloca.

    // If the parameter is passed as a byval argument, then we have to have a
    // sized type and the sized type has to have the same size as the old type.
    if (ParamTy != ActTy &&
        CallerPAL.getParamAttributes(i + 1).hasAttribute(i + 1,
                                                         Attribute::ByVal)) {
      PointerType *ParamPTy = dyn_cast<PointerType>(ParamTy);
      if (!ParamPTy || !ParamPTy->getElementType()->isSized())
        return false;

      Type *CurElTy = ActTy->getPointerElementType();
      if (DL.getTypeAllocSize(CurElTy) !=
          DL.getTypeAllocSize(ParamPTy->getElementType()))
        return false;
    }
  }

  if (Callee->isDeclaration()) {
    // Do not delete arguments unless we have a function body.
    if (FT->getNumParams() < NumActualArgs && !FT->isVarArg())
      return false;

    // If the callee is just a declaration, don't change the varargsness of the
    // call.  We don't want to introduce a varargs call where one doesn't
    // already exist.
    PointerType *APTy = cast<PointerType>(CS.getCalledValue()->getType());
    if (FT->isVarArg()!=cast<FunctionType>(APTy->getElementType())->isVarArg())
      return false;

    // If both the callee and the cast type are varargs, we still have to make
    // sure the number of fixed parameters are the same or we have the same
    // ABI issues as if we introduce a varargs call.
    if (FT->isVarArg() &&
        cast<FunctionType>(APTy->getElementType())->isVarArg() &&
        FT->getNumParams() !=
        cast<FunctionType>(APTy->getElementType())->getNumParams())
      return false;
  }

  if (FT->getNumParams() < NumActualArgs && FT->isVarArg() &&
      !CallerPAL.isEmpty())
    // In this case we have more arguments than the new function type, but we
    // won't be dropping them.  Check that these extra arguments have attributes
    // that are compatible with being a vararg call argument.
    for (unsigned i = CallerPAL.getNumSlots(); i; --i) {
      unsigned Index = CallerPAL.getSlotIndex(i - 1);
      if (Index <= FT->getNumParams())
        break;

      // Check if it has an attribute that's incompatible with varargs.
      AttributeSet PAttrs = CallerPAL.getSlotAttributes(i - 1);
      if (PAttrs.hasAttribute(Index, Attribute::StructRet))
        return false;
    }


  // Okay, we decided that this is a safe thing to do: go ahead and start
  // inserting cast instructions as necessary.
  std::vector<Value*> Args;
  Args.reserve(NumActualArgs);
  SmallVector<AttributeSet, 8> attrVec;
  attrVec.reserve(NumCommonArgs);

  // Get any return attributes.
  AttrBuilder RAttrs(CallerPAL, AttributeSet::ReturnIndex);

  // If the return value is not being used, the type may not be compatible
  // with the existing attributes.  Wipe out any problematic attributes.
  RAttrs.remove(AttributeFuncs::typeIncompatible(NewRetTy));

  // Add the new return attributes.
  if (RAttrs.hasAttributes())
    attrVec.push_back(AttributeSet::get(Caller->getContext(),
                                        AttributeSet::ReturnIndex, RAttrs));

  AI = CS.arg_begin();
  for (unsigned i = 0; i != NumCommonArgs; ++i, ++AI) {
    Type *ParamTy = FT->getParamType(i);

    if ((*AI)->getType() == ParamTy) {
      Args.push_back(*AI);
    } else {
      Args.push_back(Builder->CreateBitOrPointerCast(*AI, ParamTy));
    }

    // Add any parameter attributes.
    AttrBuilder PAttrs(CallerPAL.getParamAttributes(i + 1), i + 1);
    if (PAttrs.hasAttributes())
      attrVec.push_back(AttributeSet::get(Caller->getContext(), i + 1,
                                          PAttrs));
  }

  // If the function takes more arguments than the call was taking, add them
  // now.
  for (unsigned i = NumCommonArgs; i != FT->getNumParams(); ++i)
    Args.push_back(Constant::getNullValue(FT->getParamType(i)));

  // If we are removing arguments to the function, emit an obnoxious warning.
  if (FT->getNumParams() < NumActualArgs) {
    // TODO: if (!FT->isVarArg()) this call may be unreachable. PR14722
    if (FT->isVarArg()) {
      // Add all of the arguments in their promoted form to the arg list.
      for (unsigned i = FT->getNumParams(); i != NumActualArgs; ++i, ++AI) {
        Type *PTy = getPromotedType((*AI)->getType());
        if (PTy != (*AI)->getType()) {
          // Must promote to pass through va_arg area!
          Instruction::CastOps opcode =
            CastInst::getCastOpcode(*AI, false, PTy, false);
          Args.push_back(Builder->CreateCast(opcode, *AI, PTy));
        } else {
          Args.push_back(*AI);
        }

        // Add any parameter attributes.
        AttrBuilder PAttrs(CallerPAL.getParamAttributes(i + 1), i + 1);
        if (PAttrs.hasAttributes())
          attrVec.push_back(AttributeSet::get(FT->getContext(), i + 1,
                                              PAttrs));
      }
    }
  }

  AttributeSet FnAttrs = CallerPAL.getFnAttributes();
  if (CallerPAL.hasAttributes(AttributeSet::FunctionIndex))
    attrVec.push_back(AttributeSet::get(Callee->getContext(), FnAttrs));

  if (NewRetTy->isVoidTy())
    Caller->setName("");   // Void type should not have a name.

  const AttributeSet &NewCallerPAL = AttributeSet::get(Callee->getContext(),
                                                       attrVec);

  Instruction *NC;
  if (InvokeInst *II = dyn_cast<InvokeInst>(Caller)) {
    NC = Builder->CreateInvoke(Callee, II->getNormalDest(),
                               II->getUnwindDest(), Args);
    NC->takeName(II);
    cast<InvokeInst>(NC)->setCallingConv(II->getCallingConv());
    cast<InvokeInst>(NC)->setAttributes(NewCallerPAL);
  } else {
    CallInst *CI = cast<CallInst>(Caller);
    NC = Builder->CreateCall(Callee, Args);
    NC->takeName(CI);
    if (CI->isTailCall())
      cast<CallInst>(NC)->setTailCall();
    cast<CallInst>(NC)->setCallingConv(CI->getCallingConv());
    cast<CallInst>(NC)->setAttributes(NewCallerPAL);
  }

  // Insert a cast of the return type as necessary.
  Value *NV = NC;
  if (OldRetTy != NV->getType() && !Caller->use_empty()) {
    if (!NV->getType()->isVoidTy()) {
      NV = NC = CastInst::CreateBitOrPointerCast(NC, OldRetTy);
      NC->setDebugLoc(Caller->getDebugLoc());

      // If this is an invoke instruction, we should insert it after the first
      // non-phi, instruction in the normal successor block.
      if (InvokeInst *II = dyn_cast<InvokeInst>(Caller)) {
        BasicBlock::iterator I = II->getNormalDest()->getFirstInsertionPt();
        InsertNewInstBefore(NC, *I);
      } else {
        // Otherwise, it's a call, just insert cast right after the call.
        InsertNewInstBefore(NC, *Caller);
      }
      Worklist.AddUsersToWorkList(*Caller);
    } else {
      NV = UndefValue::get(Caller->getType());
    }
  }

  if (!Caller->use_empty())
    ReplaceInstUsesWith(*Caller, NV);
  else if (Caller->hasValueHandle()) {
    if (OldRetTy == NV->getType())
      ValueHandleBase::ValueIsRAUWd(Caller, NV);
    else
      // We cannot call ValueIsRAUWd with a different type, and the
      // actual tracked value will disappear.
      ValueHandleBase::ValueIsDeleted(Caller);
  }

  EraseInstFromFunction(*Caller);
  return true;
}

// transformCallThroughTrampoline - Turn a call to a function created by
// init_trampoline / adjust_trampoline intrinsic pair into a direct call to the
// underlying function.
//
Instruction *
InstCombiner::transformCallThroughTrampoline(CallSite CS,
                                             IntrinsicInst *Tramp) {
  Value *Callee = CS.getCalledValue();
  PointerType *PTy = cast<PointerType>(Callee->getType());
  FunctionType *FTy = cast<FunctionType>(PTy->getElementType());
  const AttributeSet &Attrs = CS.getAttributes();

  // If the call already has the 'nest' attribute somewhere then give up -
  // otherwise 'nest' would occur twice after splicing in the chain.
  if (Attrs.hasAttrSomewhere(Attribute::Nest))
    return nullptr;

  assert(Tramp &&
         "transformCallThroughTrampoline called with incorrect CallSite.");

  Function *NestF =cast<Function>(Tramp->getArgOperand(1)->stripPointerCasts());
  PointerType *NestFPTy = cast<PointerType>(NestF->getType());
  FunctionType *NestFTy = cast<FunctionType>(NestFPTy->getElementType());

  const AttributeSet &NestAttrs = NestF->getAttributes();
  if (!NestAttrs.isEmpty()) {
    unsigned NestIdx = 1;
    Type *NestTy = nullptr;
    AttributeSet NestAttr;

    // Look for a parameter marked with the 'nest' attribute.
    for (FunctionType::param_iterator I = NestFTy->param_begin(),
         E = NestFTy->param_end(); I != E; ++NestIdx, ++I)
      if (NestAttrs.hasAttribute(NestIdx, Attribute::Nest)) {
        // Record the parameter type and any other attributes.
        NestTy = *I;
        NestAttr = NestAttrs.getParamAttributes(NestIdx);
        break;
      }

    if (NestTy) {
      Instruction *Caller = CS.getInstruction();
      std::vector<Value*> NewArgs;
      NewArgs.reserve(CS.arg_size() + 1);

      SmallVector<AttributeSet, 8> NewAttrs;
      NewAttrs.reserve(Attrs.getNumSlots() + 1);

      // Insert the nest argument into the call argument list, which may
      // mean appending it.  Likewise for attributes.

      // Add any result attributes.
      if (Attrs.hasAttributes(AttributeSet::ReturnIndex))
        NewAttrs.push_back(AttributeSet::get(Caller->getContext(),
                                             Attrs.getRetAttributes()));

      {
        unsigned Idx = 1;
        CallSite::arg_iterator I = CS.arg_begin(), E = CS.arg_end();
        do {
          if (Idx == NestIdx) {
            // Add the chain argument and attributes.
            Value *NestVal = Tramp->getArgOperand(2);
            if (NestVal->getType() != NestTy)
              NestVal = Builder->CreateBitCast(NestVal, NestTy, "nest");
            NewArgs.push_back(NestVal);
            NewAttrs.push_back(AttributeSet::get(Caller->getContext(),
                                                 NestAttr));
          }

          if (I == E)
            break;

          // Add the original argument and attributes.
          NewArgs.push_back(*I);
          AttributeSet Attr = Attrs.getParamAttributes(Idx);
          if (Attr.hasAttributes(Idx)) {
            AttrBuilder B(Attr, Idx);
            NewAttrs.push_back(AttributeSet::get(Caller->getContext(),
                                                 Idx + (Idx >= NestIdx), B));
          }

          ++Idx, ++I;
        } while (1);
      }

      // Add any function attributes.
      if (Attrs.hasAttributes(AttributeSet::FunctionIndex))
        NewAttrs.push_back(AttributeSet::get(FTy->getContext(),
                                             Attrs.getFnAttributes()));

      // The trampoline may have been bitcast to a bogus type (FTy).
      // Handle this by synthesizing a new function type, equal to FTy
      // with the chain parameter inserted.

      std::vector<Type*> NewTypes;
      NewTypes.reserve(FTy->getNumParams()+1);

      // Insert the chain's type into the list of parameter types, which may
      // mean appending it.
      {
        unsigned Idx = 1;
        FunctionType::param_iterator I = FTy->param_begin(),
          E = FTy->param_end();

        do {
          if (Idx == NestIdx)
            // Add the chain's type.
            NewTypes.push_back(NestTy);

          if (I == E)
            break;

          // Add the original type.
          NewTypes.push_back(*I);

          ++Idx, ++I;
        } while (1);
      }

      // Replace the trampoline call with a direct call.  Let the generic
      // code sort out any function type mismatches.
      FunctionType *NewFTy = FunctionType::get(FTy->getReturnType(), NewTypes,
                                                FTy->isVarArg());
      Constant *NewCallee =
        NestF->getType() == PointerType::getUnqual(NewFTy) ?
        NestF : ConstantExpr::getBitCast(NestF,
                                         PointerType::getUnqual(NewFTy));
      const AttributeSet &NewPAL =
          AttributeSet::get(FTy->getContext(), NewAttrs);

      Instruction *NewCaller;
      if (InvokeInst *II = dyn_cast<InvokeInst>(Caller)) {
        NewCaller = InvokeInst::Create(NewCallee,
                                       II->getNormalDest(), II->getUnwindDest(),
                                       NewArgs);
        cast<InvokeInst>(NewCaller)->setCallingConv(II->getCallingConv());
        cast<InvokeInst>(NewCaller)->setAttributes(NewPAL);
      } else {
        NewCaller = CallInst::Create(NewCallee, NewArgs);
        if (cast<CallInst>(Caller)->isTailCall())
          cast<CallInst>(NewCaller)->setTailCall();
        cast<CallInst>(NewCaller)->
          setCallingConv(cast<CallInst>(Caller)->getCallingConv());
        cast<CallInst>(NewCaller)->setAttributes(NewPAL);
      }

      return NewCaller;
    }
  }

  // Replace the trampoline call with a direct call.  Since there is no 'nest'
  // parameter, there is no need to adjust the argument list.  Let the generic
  // code sort out any function type mismatches.
  Constant *NewCallee =
    NestF->getType() == PTy ? NestF :
                              ConstantExpr::getBitCast(NestF, PTy);
  CS.setCalledFunction(NewCallee);
  return CS.getInstruction();
}

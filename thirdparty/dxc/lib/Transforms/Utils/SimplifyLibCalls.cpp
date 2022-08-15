//===------ SimplifyLibCalls.cpp - Library calls simplifier ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is a utility pass used for testing the InstructionSimplify analysis.
// The analysis is applied to every instruction, and if it simplifies then the
// instruction is replaced by the simplification.  If you are looking for a pass
// that performs serious instruction folding, use the instcombine pass instead.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/SimplifyLibCalls.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Transforms/Utils/BuildLibCalls.h"

using namespace llvm;
using namespace PatternMatch;

#if 0 // HLSL Change Starts - option pending
static cl::opt<bool>
    ColdErrorCalls("error-reporting-is-cold", cl::init(true), cl::Hidden,
                   cl::desc("Treat error-reporting calls as cold"));

static cl::opt<bool>
    EnableUnsafeFPShrink("enable-double-float-shrink", cl::Hidden,
                         cl::init(false),
                         cl::desc("Enable unsafe double to float "
                                  "shrinking for math lib calls"));
#else
static const bool ColdErrorCalls = true;
static const bool EnableUnsafeFPShrink = false;
#endif // HLSL Change Ends

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

static bool ignoreCallingConv(LibFunc::Func Func) {
  switch (Func) {
  case LibFunc::abs:
  case LibFunc::labs:
  case LibFunc::llabs:
  case LibFunc::strlen:
    return true;
  default:
    return false;
  }
  llvm_unreachable("All cases should be covered in the switch.");
}

/// isOnlyUsedInZeroEqualityComparison - Return true if it only matters that the
/// value is equal or not-equal to zero.
static bool isOnlyUsedInZeroEqualityComparison(Value *V) {
  for (User *U : V->users()) {
    if (ICmpInst *IC = dyn_cast<ICmpInst>(U))
      if (IC->isEquality())
        if (Constant *C = dyn_cast<Constant>(IC->getOperand(1)))
          if (C->isNullValue())
            continue;
    // Unknown instruction.
    return false;
  }
  return true;
}

/// isOnlyUsedInEqualityComparison - Return true if it is only used in equality
/// comparisons with With.
static bool isOnlyUsedInEqualityComparison(Value *V, Value *With) {
  for (User *U : V->users()) {
    if (ICmpInst *IC = dyn_cast<ICmpInst>(U))
      if (IC->isEquality() && IC->getOperand(1) == With)
        continue;
    // Unknown instruction.
    return false;
  }
  return true;
}

static bool callHasFloatingPointArgument(const CallInst *CI) {
  for (CallInst::const_op_iterator it = CI->op_begin(), e = CI->op_end();
       it != e; ++it) {
    if ((*it)->getType()->isFloatingPointTy())
      return true;
  }
  return false;
}

/// \brief Check whether the overloaded unary floating point function
/// corresponing to \a Ty is available.
static bool hasUnaryFloatFn(const TargetLibraryInfo *TLI, Type *Ty,
                            LibFunc::Func DoubleFn, LibFunc::Func FloatFn,
                            LibFunc::Func LongDoubleFn) {
  switch (Ty->getTypeID()) {
  case Type::FloatTyID:
    return TLI->has(FloatFn);
  case Type::DoubleTyID:
    return TLI->has(DoubleFn);
  default:
    return TLI->has(LongDoubleFn);
  }
}

/// \brief Returns whether \p F matches the signature expected for the
/// string/memory copying library function \p Func.
/// Acceptable functions are st[rp][n]?cpy, memove, memcpy, and memset.
/// Their fortified (_chk) counterparts are also accepted.
static bool checkStringCopyLibFuncSignature(Function *F, LibFunc::Func Func) {
  const DataLayout &DL = F->getParent()->getDataLayout();
  FunctionType *FT = F->getFunctionType();
  LLVMContext &Context = F->getContext();
  Type *PCharTy = Type::getInt8PtrTy(Context);
  Type *SizeTTy = DL.getIntPtrType(Context);
  unsigned NumParams = FT->getNumParams();

  // All string libfuncs return the same type as the first parameter.
  if (FT->getReturnType() != FT->getParamType(0))
    return false;

  switch (Func) {
  default:
    llvm_unreachable("Can't check signature for non-string-copy libfunc.");
  case LibFunc::stpncpy_chk:
  case LibFunc::strncpy_chk:
    --NumParams; // fallthrough
  case LibFunc::stpncpy:
  case LibFunc::strncpy: {
    if (NumParams != 3 || FT->getParamType(0) != FT->getParamType(1) ||
        FT->getParamType(0) != PCharTy || !FT->getParamType(2)->isIntegerTy())
      return false;
    break;
  }
  case LibFunc::strcpy_chk:
  case LibFunc::stpcpy_chk:
    --NumParams; // fallthrough
  case LibFunc::stpcpy:
  case LibFunc::strcpy: {
    if (NumParams != 2 || FT->getParamType(0) != FT->getParamType(1) ||
        FT->getParamType(0) != PCharTy)
      return false;
    break;
  }
  case LibFunc::memmove_chk:
  case LibFunc::memcpy_chk:
    --NumParams; // fallthrough
  case LibFunc::memmove:
  case LibFunc::memcpy: {
    if (NumParams != 3 || !FT->getParamType(0)->isPointerTy() ||
        !FT->getParamType(1)->isPointerTy() || FT->getParamType(2) != SizeTTy)
      return false;
    break;
  }
  case LibFunc::memset_chk:
    --NumParams; // fallthrough
  case LibFunc::memset: {
    if (NumParams != 3 || !FT->getParamType(0)->isPointerTy() ||
        !FT->getParamType(1)->isIntegerTy() || FT->getParamType(2) != SizeTTy)
      return false;
    break;
  }
  }
  // If this is a fortified libcall, the last parameter is a size_t.
  if (NumParams == FT->getNumParams() - 1)
    return FT->getParamType(FT->getNumParams() - 1) == SizeTTy;
  return true;
}

//===----------------------------------------------------------------------===//
// String and Memory Library Call Optimizations
//===----------------------------------------------------------------------===//

Value *LibCallSimplifier::optimizeStrCat(CallInst *CI, IRBuilder<> &B) {
  Function *Callee = CI->getCalledFunction();
  // Verify the "strcat" function prototype.
  FunctionType *FT = Callee->getFunctionType();
  if (FT->getNumParams() != 2||
      FT->getReturnType() != B.getInt8PtrTy() ||
      FT->getParamType(0) != FT->getReturnType() ||
      FT->getParamType(1) != FT->getReturnType())
    return nullptr;

  // Extract some information from the instruction
  Value *Dst = CI->getArgOperand(0);
  Value *Src = CI->getArgOperand(1);

  // See if we can get the length of the input string.
  uint64_t Len = GetStringLength(Src);
  if (Len == 0)
    return nullptr;
  --Len; // Unbias length.

  // Handle the simple, do-nothing case: strcat(x, "") -> x
  if (Len == 0)
    return Dst;

  return emitStrLenMemCpy(Src, Dst, Len, B);
}

Value *LibCallSimplifier::emitStrLenMemCpy(Value *Src, Value *Dst, uint64_t Len,
                                           IRBuilder<> &B) {
  // We need to find the end of the destination string.  That's where the
  // memory is to be moved to. We just generate a call to strlen.
  Value *DstLen = EmitStrLen(Dst, B, DL, TLI);
  if (!DstLen)
    return nullptr;

  // Now that we have the destination's length, we must index into the
  // destination's pointer to get the actual memcpy destination (end of
  // the string .. we're concatenating).
  Value *CpyDst = B.CreateGEP(B.getInt8Ty(), Dst, DstLen, "endptr");

  // We have enough information to now generate the memcpy call to do the
  // concatenation for us.  Make a memcpy to copy the nul byte with align = 1.
  B.CreateMemCpy(CpyDst, Src,
                 ConstantInt::get(DL.getIntPtrType(Src->getContext()), Len + 1),
                 1);
  return Dst;
}

Value *LibCallSimplifier::optimizeStrNCat(CallInst *CI, IRBuilder<> &B) {
  Function *Callee = CI->getCalledFunction();
  // Verify the "strncat" function prototype.
  FunctionType *FT = Callee->getFunctionType();
  if (FT->getNumParams() != 3 || FT->getReturnType() != B.getInt8PtrTy() ||
      FT->getParamType(0) != FT->getReturnType() ||
      FT->getParamType(1) != FT->getReturnType() ||
      !FT->getParamType(2)->isIntegerTy())
    return nullptr;

  // Extract some information from the instruction
  Value *Dst = CI->getArgOperand(0);
  Value *Src = CI->getArgOperand(1);
  uint64_t Len;

  // We don't do anything if length is not constant
  if (ConstantInt *LengthArg = dyn_cast<ConstantInt>(CI->getArgOperand(2)))
    Len = LengthArg->getZExtValue();
  else
    return nullptr;

  // See if we can get the length of the input string.
  uint64_t SrcLen = GetStringLength(Src);
  if (SrcLen == 0)
    return nullptr;
  --SrcLen; // Unbias length.

  // Handle the simple, do-nothing cases:
  // strncat(x, "", c) -> x
  // strncat(x,  c, 0) -> x
  if (SrcLen == 0 || Len == 0)
    return Dst;

  // We don't optimize this case
  if (Len < SrcLen)
    return nullptr;

  // strncat(x, s, c) -> strcat(x, s)
  // s is constant so the strcat can be optimized further
  return emitStrLenMemCpy(Src, Dst, SrcLen, B);
}

Value *LibCallSimplifier::optimizeStrChr(CallInst *CI, IRBuilder<> &B) {
  Function *Callee = CI->getCalledFunction();
  // Verify the "strchr" function prototype.
  FunctionType *FT = Callee->getFunctionType();
  if (FT->getNumParams() != 2 || FT->getReturnType() != B.getInt8PtrTy() ||
      FT->getParamType(0) != FT->getReturnType() ||
      !FT->getParamType(1)->isIntegerTy(32))
    return nullptr;

  Value *SrcStr = CI->getArgOperand(0);

  // If the second operand is non-constant, see if we can compute the length
  // of the input string and turn this into memchr.
  ConstantInt *CharC = dyn_cast<ConstantInt>(CI->getArgOperand(1));
  if (!CharC) {
    uint64_t Len = GetStringLength(SrcStr);
    if (Len == 0 || !FT->getParamType(1)->isIntegerTy(32)) // memchr needs i32.
      return nullptr;

    return EmitMemChr(SrcStr, CI->getArgOperand(1), // include nul.
                      ConstantInt::get(DL.getIntPtrType(CI->getContext()), Len),
                      B, DL, TLI);
  }

  // Otherwise, the character is a constant, see if the first argument is
  // a string literal.  If so, we can constant fold.
  StringRef Str;
  if (!getConstantStringInfo(SrcStr, Str)) {
    if (CharC->isZero()) // strchr(p, 0) -> p + strlen(p)
      return B.CreateGEP(B.getInt8Ty(), SrcStr, EmitStrLen(SrcStr, B, DL, TLI), "strchr");
    return nullptr;
  }

  // Compute the offset, make sure to handle the case when we're searching for
  // zero (a weird way to spell strlen).
  size_t I = (0xFF & CharC->getSExtValue()) == 0
                 ? Str.size()
                 : Str.find(CharC->getSExtValue());
  if (I == StringRef::npos) // Didn't find the char.  strchr returns null.
    return Constant::getNullValue(CI->getType());

  // strchr(s+n,c)  -> gep(s+n+i,c)
  return B.CreateGEP(B.getInt8Ty(), SrcStr, B.getInt64(I), "strchr");
}

Value *LibCallSimplifier::optimizeStrRChr(CallInst *CI, IRBuilder<> &B) {
  Function *Callee = CI->getCalledFunction();
  // Verify the "strrchr" function prototype.
  FunctionType *FT = Callee->getFunctionType();
  if (FT->getNumParams() != 2 || FT->getReturnType() != B.getInt8PtrTy() ||
      FT->getParamType(0) != FT->getReturnType() ||
      !FT->getParamType(1)->isIntegerTy(32))
    return nullptr;

  Value *SrcStr = CI->getArgOperand(0);
  ConstantInt *CharC = dyn_cast<ConstantInt>(CI->getArgOperand(1));

  // Cannot fold anything if we're not looking for a constant.
  if (!CharC)
    return nullptr;

  StringRef Str;
  if (!getConstantStringInfo(SrcStr, Str)) {
    // strrchr(s, 0) -> strchr(s, 0)
    if (CharC->isZero())
      return EmitStrChr(SrcStr, '\0', B, TLI);
    return nullptr;
  }

  // Compute the offset.
  size_t I = (0xFF & CharC->getSExtValue()) == 0
                 ? Str.size()
                 : Str.rfind(CharC->getSExtValue());
  if (I == StringRef::npos) // Didn't find the char. Return null.
    return Constant::getNullValue(CI->getType());

  // strrchr(s+n,c) -> gep(s+n+i,c)
  return B.CreateGEP(B.getInt8Ty(), SrcStr, B.getInt64(I), "strrchr");
}

Value *LibCallSimplifier::optimizeStrCmp(CallInst *CI, IRBuilder<> &B) {
  Function *Callee = CI->getCalledFunction();
  // Verify the "strcmp" function prototype.
  FunctionType *FT = Callee->getFunctionType();
  if (FT->getNumParams() != 2 || !FT->getReturnType()->isIntegerTy(32) ||
      FT->getParamType(0) != FT->getParamType(1) ||
      FT->getParamType(0) != B.getInt8PtrTy())
    return nullptr;

  Value *Str1P = CI->getArgOperand(0), *Str2P = CI->getArgOperand(1);
  if (Str1P == Str2P) // strcmp(x,x)  -> 0
    return ConstantInt::get(CI->getType(), 0);

  StringRef Str1, Str2;
  bool HasStr1 = getConstantStringInfo(Str1P, Str1);
  bool HasStr2 = getConstantStringInfo(Str2P, Str2);

  // strcmp(x, y)  -> cnst  (if both x and y are constant strings)
  if (HasStr1 && HasStr2)
    return ConstantInt::get(CI->getType(), Str1.compare(Str2));

  if (HasStr1 && Str1.empty()) // strcmp("", x) -> -*x
    return B.CreateNeg(
        B.CreateZExt(B.CreateLoad(Str2P, "strcmpload"), CI->getType()));

  if (HasStr2 && Str2.empty()) // strcmp(x,"") -> *x
    return B.CreateZExt(B.CreateLoad(Str1P, "strcmpload"), CI->getType());

  // strcmp(P, "x") -> memcmp(P, "x", 2)
  uint64_t Len1 = GetStringLength(Str1P);
  uint64_t Len2 = GetStringLength(Str2P);
  if (Len1 && Len2) {
    return EmitMemCmp(Str1P, Str2P,
                      ConstantInt::get(DL.getIntPtrType(CI->getContext()),
                                       std::min(Len1, Len2)),
                      B, DL, TLI);
  }

  return nullptr;
}

Value *LibCallSimplifier::optimizeStrNCmp(CallInst *CI, IRBuilder<> &B) {
  Function *Callee = CI->getCalledFunction();
  // Verify the "strncmp" function prototype.
  FunctionType *FT = Callee->getFunctionType();
  if (FT->getNumParams() != 3 || !FT->getReturnType()->isIntegerTy(32) ||
      FT->getParamType(0) != FT->getParamType(1) ||
      FT->getParamType(0) != B.getInt8PtrTy() ||
      !FT->getParamType(2)->isIntegerTy())
    return nullptr;

  Value *Str1P = CI->getArgOperand(0), *Str2P = CI->getArgOperand(1);
  if (Str1P == Str2P) // strncmp(x,x,n)  -> 0
    return ConstantInt::get(CI->getType(), 0);

  // Get the length argument if it is constant.
  uint64_t Length;
  if (ConstantInt *LengthArg = dyn_cast<ConstantInt>(CI->getArgOperand(2)))
    Length = LengthArg->getZExtValue();
  else
    return nullptr;

  if (Length == 0) // strncmp(x,y,0)   -> 0
    return ConstantInt::get(CI->getType(), 0);

  if (Length == 1) // strncmp(x,y,1) -> memcmp(x,y,1)
    return EmitMemCmp(Str1P, Str2P, CI->getArgOperand(2), B, DL, TLI);

  StringRef Str1, Str2;
  bool HasStr1 = getConstantStringInfo(Str1P, Str1);
  bool HasStr2 = getConstantStringInfo(Str2P, Str2);

  // strncmp(x, y)  -> cnst  (if both x and y are constant strings)
  if (HasStr1 && HasStr2) {
    StringRef SubStr1 = Str1.substr(0, Length);
    StringRef SubStr2 = Str2.substr(0, Length);
    return ConstantInt::get(CI->getType(), SubStr1.compare(SubStr2));
  }

  if (HasStr1 && Str1.empty()) // strncmp("", x, n) -> -*x
    return B.CreateNeg(
        B.CreateZExt(B.CreateLoad(Str2P, "strcmpload"), CI->getType()));

  if (HasStr2 && Str2.empty()) // strncmp(x, "", n) -> *x
    return B.CreateZExt(B.CreateLoad(Str1P, "strcmpload"), CI->getType());

  return nullptr;
}

Value *LibCallSimplifier::optimizeStrCpy(CallInst *CI, IRBuilder<> &B) {
  Function *Callee = CI->getCalledFunction();

  if (!checkStringCopyLibFuncSignature(Callee, LibFunc::strcpy))
    return nullptr;

  Value *Dst = CI->getArgOperand(0), *Src = CI->getArgOperand(1);
  if (Dst == Src) // strcpy(x,x)  -> x
    return Src;

  // See if we can get the length of the input string.
  uint64_t Len = GetStringLength(Src);
  if (Len == 0)
    return nullptr;

  // We have enough information to now generate the memcpy call to do the
  // copy for us.  Make a memcpy to copy the nul byte with align = 1.
  B.CreateMemCpy(Dst, Src,
                 ConstantInt::get(DL.getIntPtrType(CI->getContext()), Len), 1);
  return Dst;
}

Value *LibCallSimplifier::optimizeStpCpy(CallInst *CI, IRBuilder<> &B) {
  Function *Callee = CI->getCalledFunction();
  // Verify the "stpcpy" function prototype.
  FunctionType *FT = Callee->getFunctionType();

  if (!checkStringCopyLibFuncSignature(Callee, LibFunc::stpcpy))
    return nullptr;

  Value *Dst = CI->getArgOperand(0), *Src = CI->getArgOperand(1);
  if (Dst == Src) { // stpcpy(x,x)  -> x+strlen(x)
    Value *StrLen = EmitStrLen(Src, B, DL, TLI);
    return StrLen ? B.CreateInBoundsGEP(B.getInt8Ty(), Dst, StrLen) : nullptr;
  }

  // See if we can get the length of the input string.
  uint64_t Len = GetStringLength(Src);
  if (Len == 0)
    return nullptr;

  Type *PT = FT->getParamType(0);
  Value *LenV = ConstantInt::get(DL.getIntPtrType(PT), Len);
  Value *DstEnd =
      B.CreateGEP(B.getInt8Ty(), Dst, ConstantInt::get(DL.getIntPtrType(PT), Len - 1));

  // We have enough information to now generate the memcpy call to do the
  // copy for us.  Make a memcpy to copy the nul byte with align = 1.
  B.CreateMemCpy(Dst, Src, LenV, 1);
  return DstEnd;
}

Value *LibCallSimplifier::optimizeStrNCpy(CallInst *CI, IRBuilder<> &B) {
  Function *Callee = CI->getCalledFunction();
  FunctionType *FT = Callee->getFunctionType();

  if (!checkStringCopyLibFuncSignature(Callee, LibFunc::strncpy))
    return nullptr;

  Value *Dst = CI->getArgOperand(0);
  Value *Src = CI->getArgOperand(1);
  Value *LenOp = CI->getArgOperand(2);

  // See if we can get the length of the input string.
  uint64_t SrcLen = GetStringLength(Src);
  if (SrcLen == 0)
    return nullptr;
  --SrcLen;

  if (SrcLen == 0) {
    // strncpy(x, "", y) -> memset(x, '\0', y, 1)
    B.CreateMemSet(Dst, B.getInt8('\0'), LenOp, 1);
    return Dst;
  }

  uint64_t Len;
  if (ConstantInt *LengthArg = dyn_cast<ConstantInt>(LenOp))
    Len = LengthArg->getZExtValue();
  else
    return nullptr;

  if (Len == 0)
    return Dst; // strncpy(x, y, 0) -> x

  // Let strncpy handle the zero padding
  if (Len > SrcLen + 1)
    return nullptr;

  Type *PT = FT->getParamType(0);
  // strncpy(x, s, c) -> memcpy(x, s, c, 1) [s and c are constant]
  B.CreateMemCpy(Dst, Src, ConstantInt::get(DL.getIntPtrType(PT), Len), 1);

  return Dst;
}

Value *LibCallSimplifier::optimizeStrLen(CallInst *CI, IRBuilder<> &B) {
  Function *Callee = CI->getCalledFunction();
  FunctionType *FT = Callee->getFunctionType();
  if (FT->getNumParams() != 1 || FT->getParamType(0) != B.getInt8PtrTy() ||
      !FT->getReturnType()->isIntegerTy())
    return nullptr;

  Value *Src = CI->getArgOperand(0);

  // Constant folding: strlen("xyz") -> 3
  if (uint64_t Len = GetStringLength(Src))
    return ConstantInt::get(CI->getType(), Len - 1);

  // strlen(x?"foo":"bars") --> x ? 3 : 4
  if (SelectInst *SI = dyn_cast<SelectInst>(Src)) {
    uint64_t LenTrue = GetStringLength(SI->getTrueValue());
    uint64_t LenFalse = GetStringLength(SI->getFalseValue());
    if (LenTrue && LenFalse) {
      Function *Caller = CI->getParent()->getParent();
      emitOptimizationRemark(CI->getContext(), "simplify-libcalls", *Caller,
                             SI->getDebugLoc(),
                             "folded strlen(select) to select of constants");
      return B.CreateSelect(SI->getCondition(),
                            ConstantInt::get(CI->getType(), LenTrue - 1),
                            ConstantInt::get(CI->getType(), LenFalse - 1));
    }
  }

  // strlen(x) != 0 --> *x != 0
  // strlen(x) == 0 --> *x == 0
  if (isOnlyUsedInZeroEqualityComparison(CI))
    return B.CreateZExt(B.CreateLoad(Src, "strlenfirst"), CI->getType());

  return nullptr;
}

Value *LibCallSimplifier::optimizeStrPBrk(CallInst *CI, IRBuilder<> &B) {
  Function *Callee = CI->getCalledFunction();
  FunctionType *FT = Callee->getFunctionType();
  if (FT->getNumParams() != 2 || FT->getParamType(0) != B.getInt8PtrTy() ||
      FT->getParamType(1) != FT->getParamType(0) ||
      FT->getReturnType() != FT->getParamType(0))
    return nullptr;

  StringRef S1, S2;
  bool HasS1 = getConstantStringInfo(CI->getArgOperand(0), S1);
  bool HasS2 = getConstantStringInfo(CI->getArgOperand(1), S2);

  // strpbrk(s, "") -> nullptr
  // strpbrk("", s) -> nullptr
  if ((HasS1 && S1.empty()) || (HasS2 && S2.empty()))
    return Constant::getNullValue(CI->getType());

  // Constant folding.
  if (HasS1 && HasS2) {
    size_t I = S1.find_first_of(S2);
    if (I == StringRef::npos) // No match.
      return Constant::getNullValue(CI->getType());

    return B.CreateGEP(B.getInt8Ty(), CI->getArgOperand(0), B.getInt64(I), "strpbrk");
  }

  // strpbrk(s, "a") -> strchr(s, 'a')
  if (HasS2 && S2.size() == 1)
    return EmitStrChr(CI->getArgOperand(0), S2[0], B, TLI);

  return nullptr;
}

Value *LibCallSimplifier::optimizeStrTo(CallInst *CI, IRBuilder<> &B) {
  Function *Callee = CI->getCalledFunction();
  FunctionType *FT = Callee->getFunctionType();
  if ((FT->getNumParams() != 2 && FT->getNumParams() != 3) ||
      !FT->getParamType(0)->isPointerTy() ||
      !FT->getParamType(1)->isPointerTy())
    return nullptr;

  Value *EndPtr = CI->getArgOperand(1);
  if (isa<ConstantPointerNull>(EndPtr)) {
    // With a null EndPtr, this function won't capture the main argument.
    // It would be readonly too, except that it still may write to errno.
    CI->addAttribute(1, Attribute::NoCapture);
  }

  return nullptr;
}

Value *LibCallSimplifier::optimizeStrSpn(CallInst *CI, IRBuilder<> &B) {
  Function *Callee = CI->getCalledFunction();
  FunctionType *FT = Callee->getFunctionType();
  if (FT->getNumParams() != 2 || FT->getParamType(0) != B.getInt8PtrTy() ||
      FT->getParamType(1) != FT->getParamType(0) ||
      !FT->getReturnType()->isIntegerTy())
    return nullptr;

  StringRef S1, S2;
  bool HasS1 = getConstantStringInfo(CI->getArgOperand(0), S1);
  bool HasS2 = getConstantStringInfo(CI->getArgOperand(1), S2);

  // strspn(s, "") -> 0
  // strspn("", s) -> 0
  if ((HasS1 && S1.empty()) || (HasS2 && S2.empty()))
    return Constant::getNullValue(CI->getType());

  // Constant folding.
  if (HasS1 && HasS2) {
    size_t Pos = S1.find_first_not_of(S2);
    if (Pos == StringRef::npos)
      Pos = S1.size();
    return ConstantInt::get(CI->getType(), Pos);
  }

  return nullptr;
}

Value *LibCallSimplifier::optimizeStrCSpn(CallInst *CI, IRBuilder<> &B) {
  Function *Callee = CI->getCalledFunction();
  FunctionType *FT = Callee->getFunctionType();
  if (FT->getNumParams() != 2 || FT->getParamType(0) != B.getInt8PtrTy() ||
      FT->getParamType(1) != FT->getParamType(0) ||
      !FT->getReturnType()->isIntegerTy())
    return nullptr;

  StringRef S1, S2;
  bool HasS1 = getConstantStringInfo(CI->getArgOperand(0), S1);
  bool HasS2 = getConstantStringInfo(CI->getArgOperand(1), S2);

  // strcspn("", s) -> 0
  if (HasS1 && S1.empty())
    return Constant::getNullValue(CI->getType());

  // Constant folding.
  if (HasS1 && HasS2) {
    size_t Pos = S1.find_first_of(S2);
    if (Pos == StringRef::npos)
      Pos = S1.size();
    return ConstantInt::get(CI->getType(), Pos);
  }

  // strcspn(s, "") -> strlen(s)
  if (HasS2 && S2.empty())
    return EmitStrLen(CI->getArgOperand(0), B, DL, TLI);

  return nullptr;
}

Value *LibCallSimplifier::optimizeStrStr(CallInst *CI, IRBuilder<> &B) {
  Function *Callee = CI->getCalledFunction();
  FunctionType *FT = Callee->getFunctionType();
  if (FT->getNumParams() != 2 || !FT->getParamType(0)->isPointerTy() ||
      !FT->getParamType(1)->isPointerTy() ||
      !FT->getReturnType()->isPointerTy())
    return nullptr;

  // fold strstr(x, x) -> x.
  if (CI->getArgOperand(0) == CI->getArgOperand(1))
    return B.CreateBitCast(CI->getArgOperand(0), CI->getType());

  // fold strstr(a, b) == a -> strncmp(a, b, strlen(b)) == 0
  if (isOnlyUsedInEqualityComparison(CI, CI->getArgOperand(0))) {
    Value *StrLen = EmitStrLen(CI->getArgOperand(1), B, DL, TLI);
    if (!StrLen)
      return nullptr;
    Value *StrNCmp = EmitStrNCmp(CI->getArgOperand(0), CI->getArgOperand(1),
                                 StrLen, B, DL, TLI);
    if (!StrNCmp)
      return nullptr;
    for (auto UI = CI->user_begin(), UE = CI->user_end(); UI != UE;) {
      ICmpInst *Old = cast<ICmpInst>(*UI++);
      Value *Cmp =
          B.CreateICmp(Old->getPredicate(), StrNCmp,
                       ConstantInt::getNullValue(StrNCmp->getType()), "cmp");
      replaceAllUsesWith(Old, Cmp);
    }
    return CI;
  }

  // See if either input string is a constant string.
  StringRef SearchStr, ToFindStr;
  bool HasStr1 = getConstantStringInfo(CI->getArgOperand(0), SearchStr);
  bool HasStr2 = getConstantStringInfo(CI->getArgOperand(1), ToFindStr);

  // fold strstr(x, "") -> x.
  if (HasStr2 && ToFindStr.empty())
    return B.CreateBitCast(CI->getArgOperand(0), CI->getType());

  // If both strings are known, constant fold it.
  if (HasStr1 && HasStr2) {
    size_t Offset = SearchStr.find(ToFindStr);

    if (Offset == StringRef::npos) // strstr("foo", "bar") -> null
      return Constant::getNullValue(CI->getType());

    // strstr("abcd", "bc") -> gep((char*)"abcd", 1)
    Value *Result = CastToCStr(CI->getArgOperand(0), B);
    Result = B.CreateConstInBoundsGEP1_64(Result, Offset, "strstr");
    return B.CreateBitCast(Result, CI->getType());
  }

  // fold strstr(x, "y") -> strchr(x, 'y').
  if (HasStr2 && ToFindStr.size() == 1) {
    Value *StrChr = EmitStrChr(CI->getArgOperand(0), ToFindStr[0], B, TLI);
    return StrChr ? B.CreateBitCast(StrChr, CI->getType()) : nullptr;
  }
  return nullptr;
}

Value *LibCallSimplifier::optimizeMemChr(CallInst *CI, IRBuilder<> &B) {
  Function *Callee = CI->getCalledFunction();
  FunctionType *FT = Callee->getFunctionType();
  if (FT->getNumParams() != 3 || !FT->getParamType(0)->isPointerTy() ||
      !FT->getParamType(1)->isIntegerTy(32) ||
      !FT->getParamType(2)->isIntegerTy() ||
      !FT->getReturnType()->isPointerTy())
    return nullptr;

  Value *SrcStr = CI->getArgOperand(0);
  ConstantInt *CharC = dyn_cast<ConstantInt>(CI->getArgOperand(1));
  ConstantInt *LenC = dyn_cast<ConstantInt>(CI->getArgOperand(2));

  // memchr(x, y, 0) -> null
  if (LenC && LenC->isNullValue())
    return Constant::getNullValue(CI->getType());

  // From now on we need at least constant length and string.
  StringRef Str;
  if (!LenC || !getConstantStringInfo(SrcStr, Str, 0, /*TrimAtNul=*/false))
    return nullptr;

  // Truncate the string to LenC. If Str is smaller than LenC we will still only
  // scan the string, as reading past the end of it is undefined and we can just
  // return null if we don't find the char.
  Str = Str.substr(0, LenC->getZExtValue());

  // If the char is variable but the input str and length are not we can turn
  // this memchr call into a simple bit field test. Of course this only works
  // when the return value is only checked against null.
  //
  // It would be really nice to reuse switch lowering here but we can't change
  // the CFG at this point.
  //
  // memchr("\r\n", C, 2) != nullptr -> (C & ((1 << '\r') | (1 << '\n'))) != 0
  //   after bounds check.
  if (!CharC && !Str.empty() && isOnlyUsedInZeroEqualityComparison(CI)) {
    unsigned char Max =
        *std::max_element(reinterpret_cast<const unsigned char *>(Str.begin()),
                          reinterpret_cast<const unsigned char *>(Str.end()));

    // Make sure the bit field we're about to create fits in a register on the
    // target.
    // FIXME: On a 64 bit architecture this prevents us from using the
    // interesting range of alpha ascii chars. We could do better by emitting
    // two bitfields or shifting the range by 64 if no lower chars are used.
    if (!DL.fitsInLegalInteger(Max + 1))
      return nullptr;

    // For the bit field use a power-of-2 type with at least 8 bits to avoid
    // creating unnecessary illegal types.
    unsigned char Width = NextPowerOf2(std::max((unsigned char)7, Max));

    // Now build the bit field.
    APInt Bitfield(Width, 0);
    for (char C : Str)
      Bitfield.setBit((unsigned char)C);
    Value *BitfieldC = B.getInt(Bitfield);

    // First check that the bit field access is within bounds.
    Value *C = B.CreateZExtOrTrunc(CI->getArgOperand(1), BitfieldC->getType());
    Value *Bounds = B.CreateICmp(ICmpInst::ICMP_ULT, C, B.getIntN(Width, Width),
                                 "memchr.bounds");

    // Create code that checks if the given bit is set in the field.
    Value *Shl = B.CreateShl(B.getIntN(Width, 1ULL), C);
    Value *Bits = B.CreateIsNotNull(B.CreateAnd(Shl, BitfieldC), "memchr.bits");

    // Finally merge both checks and cast to pointer type. The inttoptr
    // implicitly zexts the i1 to intptr type.
    return B.CreateIntToPtr(B.CreateAnd(Bounds, Bits, "memchr"), CI->getType());
  }

  // Check if all arguments are constants.  If so, we can constant fold.
  if (!CharC)
    return nullptr;

  // Compute the offset.
  size_t I = Str.find(CharC->getSExtValue() & 0xFF);
  if (I == StringRef::npos) // Didn't find the char.  memchr returns null.
    return Constant::getNullValue(CI->getType());

  // memchr(s+n,c,l) -> gep(s+n+i,c)
  return B.CreateGEP(B.getInt8Ty(), SrcStr, B.getInt64(I), "memchr");
}

Value *LibCallSimplifier::optimizeMemCmp(CallInst *CI, IRBuilder<> &B) {
  Function *Callee = CI->getCalledFunction();
  FunctionType *FT = Callee->getFunctionType();
  if (FT->getNumParams() != 3 || !FT->getParamType(0)->isPointerTy() ||
      !FT->getParamType(1)->isPointerTy() ||
      !FT->getReturnType()->isIntegerTy(32))
    return nullptr;

  Value *LHS = CI->getArgOperand(0), *RHS = CI->getArgOperand(1);

  if (LHS == RHS) // memcmp(s,s,x) -> 0
    return Constant::getNullValue(CI->getType());

  // Make sure we have a constant length.
  ConstantInt *LenC = dyn_cast<ConstantInt>(CI->getArgOperand(2));
  if (!LenC)
    return nullptr;
  uint64_t Len = LenC->getZExtValue();

  if (Len == 0) // memcmp(s1,s2,0) -> 0
    return Constant::getNullValue(CI->getType());

  // memcmp(S1,S2,1) -> *(unsigned char*)LHS - *(unsigned char*)RHS
  if (Len == 1) {
    Value *LHSV = B.CreateZExt(B.CreateLoad(CastToCStr(LHS, B), "lhsc"),
                               CI->getType(), "lhsv");
    Value *RHSV = B.CreateZExt(B.CreateLoad(CastToCStr(RHS, B), "rhsc"),
                               CI->getType(), "rhsv");
    return B.CreateSub(LHSV, RHSV, "chardiff");
  }

  // Constant folding: memcmp(x, y, l) -> cnst (all arguments are constant)
  StringRef LHSStr, RHSStr;
  if (getConstantStringInfo(LHS, LHSStr) &&
      getConstantStringInfo(RHS, RHSStr)) {
    // Make sure we're not reading out-of-bounds memory.
    if (Len > LHSStr.size() || Len > RHSStr.size())
      return nullptr;
    // Fold the memcmp and normalize the result.  This way we get consistent
    // results across multiple platforms.
    uint64_t Ret = 0;
    int Cmp = memcmp(LHSStr.data(), RHSStr.data(), Len);
    if (Cmp < 0)
      Ret = -1;
    else if (Cmp > 0)
      Ret = 1;
    return ConstantInt::get(CI->getType(), Ret);
  }

  return nullptr;
}

Value *LibCallSimplifier::optimizeMemCpy(CallInst *CI, IRBuilder<> &B) {
  Function *Callee = CI->getCalledFunction();

  if (!checkStringCopyLibFuncSignature(Callee, LibFunc::memcpy))
    return nullptr;

  // memcpy(x, y, n) -> llvm.memcpy(x, y, n, 1)
  B.CreateMemCpy(CI->getArgOperand(0), CI->getArgOperand(1),
                 CI->getArgOperand(2), 1);
  return CI->getArgOperand(0);
}

Value *LibCallSimplifier::optimizeMemMove(CallInst *CI, IRBuilder<> &B) {
  Function *Callee = CI->getCalledFunction();

  if (!checkStringCopyLibFuncSignature(Callee, LibFunc::memmove))
    return nullptr;

  // memmove(x, y, n) -> llvm.memmove(x, y, n, 1)
  B.CreateMemMove(CI->getArgOperand(0), CI->getArgOperand(1),
                  CI->getArgOperand(2), 1);
  return CI->getArgOperand(0);
}

Value *LibCallSimplifier::optimizeMemSet(CallInst *CI, IRBuilder<> &B) {
  Function *Callee = CI->getCalledFunction();

  if (!checkStringCopyLibFuncSignature(Callee, LibFunc::memset))
    return nullptr;

  // memset(p, v, n) -> llvm.memset(p, v, n, 1)
  Value *Val = B.CreateIntCast(CI->getArgOperand(1), B.getInt8Ty(), false);
  B.CreateMemSet(CI->getArgOperand(0), Val, CI->getArgOperand(2), 1);
  return CI->getArgOperand(0);
}

//===----------------------------------------------------------------------===//
// Math Library Optimizations
//===----------------------------------------------------------------------===//

/// Return a variant of Val with float type.
/// Currently this works in two cases: If Val is an FPExtension of a float
/// value to something bigger, simply return the operand.
/// If Val is a ConstantFP but can be converted to a float ConstantFP without
/// loss of precision do so.
static Value *valueHasFloatPrecision(Value *Val) {
  if (FPExtInst *Cast = dyn_cast<FPExtInst>(Val)) {
    Value *Op = Cast->getOperand(0);
    if (Op->getType()->isFloatTy())
      return Op;
  }
  if (ConstantFP *Const = dyn_cast<ConstantFP>(Val)) {
    APFloat F = Const->getValueAPF();
    bool losesInfo;
    (void)F.convert(APFloat::IEEEsingle, APFloat::rmNearestTiesToEven,
                    &losesInfo);
    if (!losesInfo)
      return ConstantFP::get(Const->getContext(), F);
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Double -> Float Shrinking Optimizations for Unary Functions like 'floor'

Value *LibCallSimplifier::optimizeUnaryDoubleFP(CallInst *CI, IRBuilder<> &B,
                                                bool CheckRetType) {
  Function *Callee = CI->getCalledFunction();
  FunctionType *FT = Callee->getFunctionType();
  if (FT->getNumParams() != 1 || !FT->getReturnType()->isDoubleTy() ||
      !FT->getParamType(0)->isDoubleTy())
    return nullptr;

  if (CheckRetType) {
    // Check if all the uses for function like 'sin' are converted to float.
    for (User *U : CI->users()) {
      FPTruncInst *Cast = dyn_cast<FPTruncInst>(U);
      if (!Cast || !Cast->getType()->isFloatTy())
        return nullptr;
    }
  }

  // If this is something like 'floor((double)floatval)', convert to floorf.
  Value *V = valueHasFloatPrecision(CI->getArgOperand(0));
  if (V == nullptr)
    return nullptr;

  // floor((double)floatval) -> (double)floorf(floatval)
  if (Callee->isIntrinsic()) {
    Module *M = CI->getParent()->getParent()->getParent();
    Intrinsic::ID IID = Callee->getIntrinsicID();
    Function *F = Intrinsic::getDeclaration(M, IID, B.getFloatTy());
    V = B.CreateCall(F, V);
  } else {
    // The call is a library call rather than an intrinsic.
    V = EmitUnaryFloatFnCall(V, Callee->getName(), B, Callee->getAttributes());
  }

  return B.CreateFPExt(V, B.getDoubleTy());
}

// Double -> Float Shrinking Optimizations for Binary Functions like 'fmin/fmax'
Value *LibCallSimplifier::optimizeBinaryDoubleFP(CallInst *CI, IRBuilder<> &B) {
  Function *Callee = CI->getCalledFunction();
  FunctionType *FT = Callee->getFunctionType();
  // Just make sure this has 2 arguments of the same FP type, which match the
  // result type.
  if (FT->getNumParams() != 2 || FT->getReturnType() != FT->getParamType(0) ||
      FT->getParamType(0) != FT->getParamType(1) ||
      !FT->getParamType(0)->isFloatingPointTy())
    return nullptr;

  // If this is something like 'fmin((double)floatval1, (double)floatval2)',
  // or fmin(1.0, (double)floatval), then we convert it to fminf.
  Value *V1 = valueHasFloatPrecision(CI->getArgOperand(0));
  if (V1 == nullptr)
    return nullptr;
  Value *V2 = valueHasFloatPrecision(CI->getArgOperand(1));
  if (V2 == nullptr)
    return nullptr;

  // fmin((double)floatval1, (double)floatval2)
  //                      -> (double)fminf(floatval1, floatval2)
  // TODO: Handle intrinsics in the same way as in optimizeUnaryDoubleFP().
  Value *V = EmitBinaryFloatFnCall(V1, V2, Callee->getName(), B,
                                   Callee->getAttributes());
  return B.CreateFPExt(V, B.getDoubleTy());
}

Value *LibCallSimplifier::optimizeCos(CallInst *CI, IRBuilder<> &B) {
  Function *Callee = CI->getCalledFunction();
  Value *Ret = nullptr;
  if (UnsafeFPShrink && Callee->getName() == "cos" && TLI->has(LibFunc::cosf)) {
    Ret = optimizeUnaryDoubleFP(CI, B, true);
  }

  FunctionType *FT = Callee->getFunctionType();
  // Just make sure this has 1 argument of FP type, which matches the
  // result type.
  if (FT->getNumParams() != 1 || FT->getReturnType() != FT->getParamType(0) ||
      !FT->getParamType(0)->isFloatingPointTy())
    return Ret;

  // cos(-x) -> cos(x)
  Value *Op1 = CI->getArgOperand(0);
  if (BinaryOperator::isFNeg(Op1)) {
    BinaryOperator *BinExpr = cast<BinaryOperator>(Op1);
    return B.CreateCall(Callee, BinExpr->getOperand(1), "cos");
  }
  return Ret;
}

Value *LibCallSimplifier::optimizePow(CallInst *CI, IRBuilder<> &B) {
  Function *Callee = CI->getCalledFunction();

  Value *Ret = nullptr;
  if (UnsafeFPShrink && Callee->getName() == "pow" && TLI->has(LibFunc::powf)) {
    Ret = optimizeUnaryDoubleFP(CI, B, true);
  }

  FunctionType *FT = Callee->getFunctionType();
  // Just make sure this has 2 arguments of the same FP type, which match the
  // result type.
  if (FT->getNumParams() != 2 || FT->getReturnType() != FT->getParamType(0) ||
      FT->getParamType(0) != FT->getParamType(1) ||
      !FT->getParamType(0)->isFloatingPointTy())
    return Ret;

  Value *Op1 = CI->getArgOperand(0), *Op2 = CI->getArgOperand(1);
  if (ConstantFP *Op1C = dyn_cast<ConstantFP>(Op1)) {
    // pow(1.0, x) -> 1.0
    if (Op1C->isExactlyValue(1.0))
      return Op1C;
    // pow(2.0, x) -> exp2(x)
    if (Op1C->isExactlyValue(2.0) &&
        hasUnaryFloatFn(TLI, Op1->getType(), LibFunc::exp2, LibFunc::exp2f,
                        LibFunc::exp2l))
      return EmitUnaryFloatFnCall(Op2, "exp2", B, Callee->getAttributes());
    // pow(10.0, x) -> exp10(x)
    if (Op1C->isExactlyValue(10.0) &&
        hasUnaryFloatFn(TLI, Op1->getType(), LibFunc::exp10, LibFunc::exp10f,
                        LibFunc::exp10l))
      return EmitUnaryFloatFnCall(Op2, TLI->getName(LibFunc::exp10), B,
                                  Callee->getAttributes());
  }

  ConstantFP *Op2C = dyn_cast<ConstantFP>(Op2);
  if (!Op2C)
    return Ret;

  if (Op2C->getValueAPF().isZero()) // pow(x, 0.0) -> 1.0
    return ConstantFP::get(CI->getType(), 1.0);

  if (Op2C->isExactlyValue(0.5) &&
      hasUnaryFloatFn(TLI, Op2->getType(), LibFunc::sqrt, LibFunc::sqrtf,
                      LibFunc::sqrtl) &&
      hasUnaryFloatFn(TLI, Op2->getType(), LibFunc::fabs, LibFunc::fabsf,
                      LibFunc::fabsl)) {
    // Expand pow(x, 0.5) to (x == -infinity ? +infinity : fabs(sqrt(x))).
    // This is faster than calling pow, and still handles negative zero
    // and negative infinity correctly.
    // TODO: In fast-math mode, this could be just sqrt(x).
    // TODO: In finite-only mode, this could be just fabs(sqrt(x)).
    Value *Inf = ConstantFP::getInfinity(CI->getType());
    Value *NegInf = ConstantFP::getInfinity(CI->getType(), true);
    Value *Sqrt = EmitUnaryFloatFnCall(Op1, "sqrt", B, Callee->getAttributes());
    Value *FAbs =
        EmitUnaryFloatFnCall(Sqrt, "fabs", B, Callee->getAttributes());
    Value *FCmp = B.CreateFCmpOEQ(Op1, NegInf);
    Value *Sel = B.CreateSelect(FCmp, Inf, FAbs);
    return Sel;
  }

  if (Op2C->isExactlyValue(1.0)) // pow(x, 1.0) -> x
    return Op1;
  if (Op2C->isExactlyValue(2.0)) // pow(x, 2.0) -> x*x
    return B.CreateFMul(Op1, Op1, "pow2");
  if (Op2C->isExactlyValue(-1.0)) // pow(x, -1.0) -> 1.0/x
    return B.CreateFDiv(ConstantFP::get(CI->getType(), 1.0), Op1, "powrecip");
  return nullptr;
}

Value *LibCallSimplifier::optimizeExp2(CallInst *CI, IRBuilder<> &B) {
  Function *Callee = CI->getCalledFunction();
  Function *Caller = CI->getParent()->getParent();

  Value *Ret = nullptr;
  if (UnsafeFPShrink && Callee->getName() == "exp2" &&
      TLI->has(LibFunc::exp2f)) {
    Ret = optimizeUnaryDoubleFP(CI, B, true);
  }

  FunctionType *FT = Callee->getFunctionType();
  // Just make sure this has 1 argument of FP type, which matches the
  // result type.
  if (FT->getNumParams() != 1 || FT->getReturnType() != FT->getParamType(0) ||
      !FT->getParamType(0)->isFloatingPointTy())
    return Ret;

  Value *Op = CI->getArgOperand(0);
  // Turn exp2(sitofp(x)) -> ldexp(1.0, sext(x))  if sizeof(x) <= 32
  // Turn exp2(uitofp(x)) -> ldexp(1.0, zext(x))  if sizeof(x) < 32
  LibFunc::Func LdExp = LibFunc::ldexpl;
  if (Op->getType()->isFloatTy())
    LdExp = LibFunc::ldexpf;
  else if (Op->getType()->isDoubleTy())
    LdExp = LibFunc::ldexp;

  if (TLI->has(LdExp)) {
    Value *LdExpArg = nullptr;
    if (SIToFPInst *OpC = dyn_cast<SIToFPInst>(Op)) {
      if (OpC->getOperand(0)->getType()->getPrimitiveSizeInBits() <= 32)
        LdExpArg = B.CreateSExt(OpC->getOperand(0), B.getInt32Ty());
    } else if (UIToFPInst *OpC = dyn_cast<UIToFPInst>(Op)) {
      if (OpC->getOperand(0)->getType()->getPrimitiveSizeInBits() < 32)
        LdExpArg = B.CreateZExt(OpC->getOperand(0), B.getInt32Ty());
    }

    if (LdExpArg) {
      Constant *One = ConstantFP::get(CI->getContext(), APFloat(1.0f));
      if (!Op->getType()->isFloatTy())
        One = ConstantExpr::getFPExtend(One, Op->getType());

      Module *M = Caller->getParent();
      Value *Callee =
          M->getOrInsertFunction(TLI->getName(LdExp), Op->getType(),
                                 Op->getType(), B.getInt32Ty(), nullptr);
      CallInst *CI = B.CreateCall(Callee, {One, LdExpArg});
      if (const Function *F = dyn_cast<Function>(Callee->stripPointerCasts()))
        CI->setCallingConv(F->getCallingConv());

      return CI;
    }
  }
  return Ret;
}

Value *LibCallSimplifier::optimizeFabs(CallInst *CI, IRBuilder<> &B) {
  Function *Callee = CI->getCalledFunction();

  Value *Ret = nullptr;
  if (Callee->getName() == "fabs" && TLI->has(LibFunc::fabsf)) {
    Ret = optimizeUnaryDoubleFP(CI, B, false);
  }

  FunctionType *FT = Callee->getFunctionType();
  // Make sure this has 1 argument of FP type which matches the result type.
  if (FT->getNumParams() != 1 || FT->getReturnType() != FT->getParamType(0) ||
      !FT->getParamType(0)->isFloatingPointTy())
    return Ret;

  Value *Op = CI->getArgOperand(0);
  if (Instruction *I = dyn_cast<Instruction>(Op)) {
    // Fold fabs(x * x) -> x * x; any squared FP value must already be positive.
    if (I->getOpcode() == Instruction::FMul)
      if (I->getOperand(0) == I->getOperand(1))
        return Op;
  }
  return Ret;
}

Value *LibCallSimplifier::optimizeSqrt(CallInst *CI, IRBuilder<> &B) {
  Function *Callee = CI->getCalledFunction();
  
  Value *Ret = nullptr;
  if (TLI->has(LibFunc::sqrtf) && (Callee->getName() == "sqrt" ||
                                   Callee->getIntrinsicID() == Intrinsic::sqrt))
    Ret = optimizeUnaryDoubleFP(CI, B, true);

  // FIXME: For finer-grain optimization, we need intrinsics to have the same
  // fast-math flag decorations that are applied to FP instructions. For now,
  // we have to rely on the function-level unsafe-fp-math attribute to do this
  // optimization because there's no other way to express that the sqrt can be
  // reassociated.
  Function *F = CI->getParent()->getParent();
  if (F->hasFnAttribute("unsafe-fp-math")) {
    // Check for unsafe-fp-math = true.
    Attribute Attr = F->getFnAttribute("unsafe-fp-math");
    if (Attr.getValueAsString() != "true")
      return Ret;
  }
  Value *Op = CI->getArgOperand(0);
  if (Instruction *I = dyn_cast<Instruction>(Op)) {
    if (I->getOpcode() == Instruction::FMul && I->hasUnsafeAlgebra()) {
      // We're looking for a repeated factor in a multiplication tree,
      // so we can do this fold: sqrt(x * x) -> fabs(x);
      // or this fold: sqrt(x * x * y) -> fabs(x) * sqrt(y).
      Value *Op0 = I->getOperand(0);
      Value *Op1 = I->getOperand(1);
      Value *RepeatOp = nullptr;
      Value *OtherOp = nullptr;
      if (Op0 == Op1) {
        // Simple match: the operands of the multiply are identical.
        RepeatOp = Op0;
      } else {
        // Look for a more complicated pattern: one of the operands is itself
        // a multiply, so search for a common factor in that multiply.
        // Note: We don't bother looking any deeper than this first level or for
        // variations of this pattern because instcombine's visitFMUL and/or the
        // reassociation pass should give us this form.
        Value *OtherMul0, *OtherMul1;
        if (match(Op0, m_FMul(m_Value(OtherMul0), m_Value(OtherMul1)))) {
          // Pattern: sqrt((x * y) * z)
          if (OtherMul0 == OtherMul1) {
            // Matched: sqrt((x * x) * z)
            RepeatOp = OtherMul0;
            OtherOp = Op1;
          }
        }
      }
      if (RepeatOp) {
        // Fast math flags for any created instructions should match the sqrt
        // and multiply.
        // FIXME: We're not checking the sqrt because it doesn't have
        // fast-math-flags (see earlier comment).
        IRBuilder<true, ConstantFolder,
          IRBuilderDefaultInserter<true> >::FastMathFlagGuard Guard(B);
        B.SetFastMathFlags(I->getFastMathFlags());
        // If we found a repeated factor, hoist it out of the square root and
        // replace it with the fabs of that factor.
        Module *M = Callee->getParent();
        Type *ArgType = Op->getType();
        Value *Fabs = Intrinsic::getDeclaration(M, Intrinsic::fabs, ArgType);
        Value *FabsCall = B.CreateCall(Fabs, RepeatOp, "fabs");
        if (OtherOp) {
          // If we found a non-repeated factor, we still need to get its square
          // root. We then multiply that by the value that was simplified out
          // of the square root calculation.
          Value *Sqrt = Intrinsic::getDeclaration(M, Intrinsic::sqrt, ArgType);
          Value *SqrtCall = B.CreateCall(Sqrt, OtherOp, "sqrt");
          return B.CreateFMul(FabsCall, SqrtCall);
        }
        return FabsCall;
      }
    }
  }
  return Ret;
}

static bool isTrigLibCall(CallInst *CI);
static void insertSinCosCall(IRBuilder<> &B, Function *OrigCallee, Value *Arg,
                             bool UseFloat, Value *&Sin, Value *&Cos,
                             Value *&SinCos);

Value *LibCallSimplifier::optimizeSinCosPi(CallInst *CI, IRBuilder<> &B) {

  // Make sure the prototype is as expected, otherwise the rest of the
  // function is probably invalid and likely to abort.
  if (!isTrigLibCall(CI))
    return nullptr;

  Value *Arg = CI->getArgOperand(0);
  SmallVector<CallInst *, 1> SinCalls;
  SmallVector<CallInst *, 1> CosCalls;
  SmallVector<CallInst *, 1> SinCosCalls;

  bool IsFloat = Arg->getType()->isFloatTy();

  // Look for all compatible sinpi, cospi and sincospi calls with the same
  // argument. If there are enough (in some sense) we can make the
  // substitution.
  for (User *U : Arg->users())
    classifyArgUse(U, CI->getParent(), IsFloat, SinCalls, CosCalls,
                   SinCosCalls);

  // It's only worthwhile if both sinpi and cospi are actually used.
  if (SinCosCalls.empty() && (SinCalls.empty() || CosCalls.empty()))
    return nullptr;

  Value *Sin, *Cos, *SinCos;
  insertSinCosCall(B, CI->getCalledFunction(), Arg, IsFloat, Sin, Cos, SinCos);

  replaceTrigInsts(SinCalls, Sin);
  replaceTrigInsts(CosCalls, Cos);
  replaceTrigInsts(SinCosCalls, SinCos);

  return nullptr;
}

static bool isTrigLibCall(CallInst *CI) {
  Function *Callee = CI->getCalledFunction();
  FunctionType *FT = Callee->getFunctionType();

  // We can only hope to do anything useful if we can ignore things like errno
  // and floating-point exceptions.
  bool AttributesSafe =
      CI->hasFnAttr(Attribute::NoUnwind) && CI->hasFnAttr(Attribute::ReadNone);

  // Other than that we need float(float) or double(double)
  return AttributesSafe && FT->getNumParams() == 1 &&
         FT->getReturnType() == FT->getParamType(0) &&
         (FT->getParamType(0)->isFloatTy() ||
          FT->getParamType(0)->isDoubleTy());
}

void
LibCallSimplifier::classifyArgUse(Value *Val, BasicBlock *BB, bool IsFloat,
                                  SmallVectorImpl<CallInst *> &SinCalls,
                                  SmallVectorImpl<CallInst *> &CosCalls,
                                  SmallVectorImpl<CallInst *> &SinCosCalls) {
  CallInst *CI = dyn_cast<CallInst>(Val);

  if (!CI)
    return;

  Function *Callee = CI->getCalledFunction();
  StringRef FuncName = Callee->getName();
  LibFunc::Func Func;
  if (!TLI->getLibFunc(FuncName, Func) || !TLI->has(Func) || !isTrigLibCall(CI))
    return;

  if (IsFloat) {
    if (Func == LibFunc::sinpif)
      SinCalls.push_back(CI);
    else if (Func == LibFunc::cospif)
      CosCalls.push_back(CI);
    else if (Func == LibFunc::sincospif_stret)
      SinCosCalls.push_back(CI);
  } else {
    if (Func == LibFunc::sinpi)
      SinCalls.push_back(CI);
    else if (Func == LibFunc::cospi)
      CosCalls.push_back(CI);
    else if (Func == LibFunc::sincospi_stret)
      SinCosCalls.push_back(CI);
  }
}

void LibCallSimplifier::replaceTrigInsts(SmallVectorImpl<CallInst *> &Calls,
                                         Value *Res) {
  for (SmallVectorImpl<CallInst *>::iterator I = Calls.begin(), E = Calls.end();
       I != E; ++I) {
    replaceAllUsesWith(*I, Res);
  }
}

void insertSinCosCall(IRBuilder<> &B, Function *OrigCallee, Value *Arg,
                      bool UseFloat, Value *&Sin, Value *&Cos, Value *&SinCos) {
  Type *ArgTy = Arg->getType();
  Type *ResTy;
  StringRef Name;

  Triple T(OrigCallee->getParent()->getTargetTriple());
  if (UseFloat) {
    Name = "__sincospif_stret";

    assert(T.getArch() != Triple::x86 && "x86 messy and unsupported for now");
    // x86_64 can't use {float, float} since that would be returned in both
    // xmm0 and xmm1, which isn't what a real struct would do.
    ResTy = T.getArch() == Triple::x86_64
                ? static_cast<Type *>(VectorType::get(ArgTy, 2))
                : static_cast<Type *>(StructType::get(ArgTy, ArgTy, nullptr));
  } else {
    Name = "__sincospi_stret";
    ResTy = StructType::get(ArgTy, ArgTy, nullptr);
  }

  Module *M = OrigCallee->getParent();
  Value *Callee = M->getOrInsertFunction(Name, OrigCallee->getAttributes(),
                                         ResTy, ArgTy, nullptr);

  if (Instruction *ArgInst = dyn_cast<Instruction>(Arg)) {
    // If the argument is an instruction, it must dominate all uses so put our
    // sincos call there.
    BasicBlock::iterator Loc = ArgInst;
    B.SetInsertPoint(ArgInst->getParent(), ++Loc);
  } else {
    // Otherwise (e.g. for a constant) the beginning of the function is as
    // good a place as any.
    BasicBlock &EntryBB = B.GetInsertBlock()->getParent()->getEntryBlock();
    B.SetInsertPoint(&EntryBB, EntryBB.begin());
  }

  SinCos = B.CreateCall(Callee, Arg, "sincospi");

  if (SinCos->getType()->isStructTy()) {
    Sin = B.CreateExtractValue(SinCos, 0, "sinpi");
    Cos = B.CreateExtractValue(SinCos, 1, "cospi");
  } else {
    Sin = B.CreateExtractElement(SinCos, ConstantInt::get(B.getInt32Ty(), 0),
                                 "sinpi");
    Cos = B.CreateExtractElement(SinCos, ConstantInt::get(B.getInt32Ty(), 1),
                                 "cospi");
  }
}

//===----------------------------------------------------------------------===//
// Integer Library Call Optimizations
//===----------------------------------------------------------------------===//

Value *LibCallSimplifier::optimizeFFS(CallInst *CI, IRBuilder<> &B) {
  Function *Callee = CI->getCalledFunction();
  FunctionType *FT = Callee->getFunctionType();
  // Just make sure this has 2 arguments of the same FP type, which match the
  // result type.
  if (FT->getNumParams() != 1 || !FT->getReturnType()->isIntegerTy(32) ||
      !FT->getParamType(0)->isIntegerTy())
    return nullptr;

  Value *Op = CI->getArgOperand(0);

  // Constant fold.
  if (ConstantInt *CI = dyn_cast<ConstantInt>(Op)) {
    if (CI->isZero()) // ffs(0) -> 0.
      return B.getInt32(0);
    // ffs(c) -> cttz(c)+1
    return B.getInt32(CI->getValue().countTrailingZeros() + 1);
  }

  // ffs(x) -> x != 0 ? (i32)llvm.cttz(x)+1 : 0
  Type *ArgType = Op->getType();
  Value *F =
      Intrinsic::getDeclaration(Callee->getParent(), Intrinsic::cttz, ArgType);
  Value *V = B.CreateCall(F, {Op, B.getFalse()}, "cttz");
  V = B.CreateAdd(V, ConstantInt::get(V->getType(), 1));
  V = B.CreateIntCast(V, B.getInt32Ty(), false);

  Value *Cond = B.CreateICmpNE(Op, Constant::getNullValue(ArgType));
  return B.CreateSelect(Cond, V, B.getInt32(0));
}

Value *LibCallSimplifier::optimizeAbs(CallInst *CI, IRBuilder<> &B) {
  Function *Callee = CI->getCalledFunction();
  FunctionType *FT = Callee->getFunctionType();
  // We require integer(integer) where the types agree.
  if (FT->getNumParams() != 1 || !FT->getReturnType()->isIntegerTy() ||
      FT->getParamType(0) != FT->getReturnType())
    return nullptr;

  // abs(x) -> x >s -1 ? x : -x
  Value *Op = CI->getArgOperand(0);
  Value *Pos =
      B.CreateICmpSGT(Op, Constant::getAllOnesValue(Op->getType()), "ispos");
  Value *Neg = B.CreateNeg(Op, "neg");
  return B.CreateSelect(Pos, Op, Neg);
}

Value *LibCallSimplifier::optimizeIsDigit(CallInst *CI, IRBuilder<> &B) {
  Function *Callee = CI->getCalledFunction();
  FunctionType *FT = Callee->getFunctionType();
  // We require integer(i32)
  if (FT->getNumParams() != 1 || !FT->getReturnType()->isIntegerTy() ||
      !FT->getParamType(0)->isIntegerTy(32))
    return nullptr;

  // isdigit(c) -> (c-'0') <u 10
  Value *Op = CI->getArgOperand(0);
  Op = B.CreateSub(Op, B.getInt32('0'), "isdigittmp");
  Op = B.CreateICmpULT(Op, B.getInt32(10), "isdigit");
  return B.CreateZExt(Op, CI->getType());
}

Value *LibCallSimplifier::optimizeIsAscii(CallInst *CI, IRBuilder<> &B) {
  Function *Callee = CI->getCalledFunction();
  FunctionType *FT = Callee->getFunctionType();
  // We require integer(i32)
  if (FT->getNumParams() != 1 || !FT->getReturnType()->isIntegerTy() ||
      !FT->getParamType(0)->isIntegerTy(32))
    return nullptr;

  // isascii(c) -> c <u 128
  Value *Op = CI->getArgOperand(0);
  Op = B.CreateICmpULT(Op, B.getInt32(128), "isascii");
  return B.CreateZExt(Op, CI->getType());
}

Value *LibCallSimplifier::optimizeToAscii(CallInst *CI, IRBuilder<> &B) {
  Function *Callee = CI->getCalledFunction();
  FunctionType *FT = Callee->getFunctionType();
  // We require i32(i32)
  if (FT->getNumParams() != 1 || FT->getReturnType() != FT->getParamType(0) ||
      !FT->getParamType(0)->isIntegerTy(32))
    return nullptr;

  // toascii(c) -> c & 0x7f
  return B.CreateAnd(CI->getArgOperand(0),
                     ConstantInt::get(CI->getType(), 0x7F));
}

//===----------------------------------------------------------------------===//
// Formatting and IO Library Call Optimizations
//===----------------------------------------------------------------------===//

static bool isReportingError(Function *Callee, CallInst *CI, int StreamArg);

Value *LibCallSimplifier::optimizeErrorReporting(CallInst *CI, IRBuilder<> &B,
                                                 int StreamArg) {
  // Error reporting calls should be cold, mark them as such.
  // This applies even to non-builtin calls: it is only a hint and applies to
  // functions that the frontend might not understand as builtins.

  // This heuristic was suggested in:
  // Improving Static Branch Prediction in a Compiler
  // Brian L. Deitrich, Ben-Chung Cheng, Wen-mei W. Hwu
  // Proceedings of PACT'98, Oct. 1998, IEEE
  Function *Callee = CI->getCalledFunction();

  if (!CI->hasFnAttr(Attribute::Cold) &&
      isReportingError(Callee, CI, StreamArg)) {
    CI->addAttribute(AttributeSet::FunctionIndex, Attribute::Cold);
  }

  return nullptr;
}

static bool isReportingError(Function *Callee, CallInst *CI, int StreamArg) {
  if (!ColdErrorCalls)
    return false;

  if (!Callee || !Callee->isDeclaration())
    return false;

  if (StreamArg < 0)
    return true;

  // These functions might be considered cold, but only if their stream
  // argument is stderr.

  if (StreamArg >= (int)CI->getNumArgOperands())
    return false;
  LoadInst *LI = dyn_cast<LoadInst>(CI->getArgOperand(StreamArg));
  if (!LI)
    return false;
  GlobalVariable *GV = dyn_cast<GlobalVariable>(LI->getPointerOperand());
  if (!GV || !GV->isDeclaration())
    return false;
  return GV->getName() == "stderr";
}

Value *LibCallSimplifier::optimizePrintFString(CallInst *CI, IRBuilder<> &B) {
  // Check for a fixed format string.
  StringRef FormatStr;
  if (!getConstantStringInfo(CI->getArgOperand(0), FormatStr))
    return nullptr;

  // Empty format string -> noop.
  if (FormatStr.empty()) // Tolerate printf's declared void.
    return CI->use_empty() ? (Value *)CI : ConstantInt::get(CI->getType(), 0);

  // Do not do any of the following transformations if the printf return value
  // is used, in general the printf return value is not compatible with either
  // putchar() or puts().
  if (!CI->use_empty())
    return nullptr;

  // printf("x") -> putchar('x'), even for '%'.
  if (FormatStr.size() == 1) {
    Value *Res = EmitPutChar(B.getInt32(FormatStr[0]), B, TLI);
    if (CI->use_empty() || !Res)
      return Res;
    return B.CreateIntCast(Res, CI->getType(), true);
  }

  // printf("foo\n") --> puts("foo")
  if (FormatStr[FormatStr.size() - 1] == '\n' &&
      FormatStr.find('%') == StringRef::npos) { // No format characters.
    // Create a string literal with no \n on it.  We expect the constant merge
    // pass to be run after this pass, to merge duplicate strings.
    FormatStr = FormatStr.drop_back();
    Value *GV = B.CreateGlobalString(FormatStr, "str");
    Value *NewCI = EmitPutS(GV, B, TLI);
    return (CI->use_empty() || !NewCI)
               ? NewCI
               : ConstantInt::get(CI->getType(), FormatStr.size() + 1);
  }

  // Optimize specific format strings.
  // printf("%c", chr) --> putchar(chr)
  if (FormatStr == "%c" && CI->getNumArgOperands() > 1 &&
      CI->getArgOperand(1)->getType()->isIntegerTy()) {
    Value *Res = EmitPutChar(CI->getArgOperand(1), B, TLI);

    if (CI->use_empty() || !Res)
      return Res;
    return B.CreateIntCast(Res, CI->getType(), true);
  }

  // printf("%s\n", str) --> puts(str)
  if (FormatStr == "%s\n" && CI->getNumArgOperands() > 1 &&
      CI->getArgOperand(1)->getType()->isPointerTy()) {
    return EmitPutS(CI->getArgOperand(1), B, TLI);
  }
  return nullptr;
}

Value *LibCallSimplifier::optimizePrintF(CallInst *CI, IRBuilder<> &B) {

  Function *Callee = CI->getCalledFunction();
  // Require one fixed pointer argument and an integer/void result.
  FunctionType *FT = Callee->getFunctionType();
  if (FT->getNumParams() < 1 || !FT->getParamType(0)->isPointerTy() ||
      !(FT->getReturnType()->isIntegerTy() || FT->getReturnType()->isVoidTy()))
    return nullptr;

  if (Value *V = optimizePrintFString(CI, B)) {
    return V;
  }

  // printf(format, ...) -> iprintf(format, ...) if no floating point
  // arguments.
  if (TLI->has(LibFunc::iprintf) && !callHasFloatingPointArgument(CI)) {
    Module *M = B.GetInsertBlock()->getParent()->getParent();
    Constant *IPrintFFn =
        M->getOrInsertFunction("iprintf", FT, Callee->getAttributes());
    CallInst *New = cast<CallInst>(CI->clone());
    New->setCalledFunction(IPrintFFn);
    B.Insert(New);
    return New;
  }
  return nullptr;
}

Value *LibCallSimplifier::optimizeSPrintFString(CallInst *CI, IRBuilder<> &B) {
  // Check for a fixed format string.
  StringRef FormatStr;
  if (!getConstantStringInfo(CI->getArgOperand(1), FormatStr))
    return nullptr;

  // If we just have a format string (nothing else crazy) transform it.
  if (CI->getNumArgOperands() == 2) {
    // Make sure there's no % in the constant array.  We could try to handle
    // %% -> % in the future if we cared.
    for (unsigned i = 0, e = FormatStr.size(); i != e; ++i)
      if (FormatStr[i] == '%')
        return nullptr; // we found a format specifier, bail out.

    // sprintf(str, fmt) -> llvm.memcpy(str, fmt, strlen(fmt)+1, 1)
    B.CreateMemCpy(CI->getArgOperand(0), CI->getArgOperand(1),
                   ConstantInt::get(DL.getIntPtrType(CI->getContext()),
                                    FormatStr.size() + 1),
                   1); // Copy the null byte.
    return ConstantInt::get(CI->getType(), FormatStr.size());
  }

  // The remaining optimizations require the format string to be "%s" or "%c"
  // and have an extra operand.
  if (FormatStr.size() != 2 || FormatStr[0] != '%' ||
      CI->getNumArgOperands() < 3)
    return nullptr;

  // Decode the second character of the format string.
  if (FormatStr[1] == 'c') {
    // sprintf(dst, "%c", chr) --> *(i8*)dst = chr; *((i8*)dst+1) = 0
    if (!CI->getArgOperand(2)->getType()->isIntegerTy())
      return nullptr;
    Value *V = B.CreateTrunc(CI->getArgOperand(2), B.getInt8Ty(), "char");
    Value *Ptr = CastToCStr(CI->getArgOperand(0), B);
    B.CreateStore(V, Ptr);
    Ptr = B.CreateGEP(B.getInt8Ty(), Ptr, B.getInt32(1), "nul");
    B.CreateStore(B.getInt8(0), Ptr);

    return ConstantInt::get(CI->getType(), 1);
  }

  if (FormatStr[1] == 's') {
    // sprintf(dest, "%s", str) -> llvm.memcpy(dest, str, strlen(str)+1, 1)
    if (!CI->getArgOperand(2)->getType()->isPointerTy())
      return nullptr;

    Value *Len = EmitStrLen(CI->getArgOperand(2), B, DL, TLI);
    if (!Len)
      return nullptr;
    Value *IncLen =
        B.CreateAdd(Len, ConstantInt::get(Len->getType(), 1), "leninc");
    B.CreateMemCpy(CI->getArgOperand(0), CI->getArgOperand(2), IncLen, 1);

    // The sprintf result is the unincremented number of bytes in the string.
    return B.CreateIntCast(Len, CI->getType(), false);
  }
  return nullptr;
}

Value *LibCallSimplifier::optimizeSPrintF(CallInst *CI, IRBuilder<> &B) {
  Function *Callee = CI->getCalledFunction();
  // Require two fixed pointer arguments and an integer result.
  FunctionType *FT = Callee->getFunctionType();
  if (FT->getNumParams() != 2 || !FT->getParamType(0)->isPointerTy() ||
      !FT->getParamType(1)->isPointerTy() ||
      !FT->getReturnType()->isIntegerTy())
    return nullptr;

  if (Value *V = optimizeSPrintFString(CI, B)) {
    return V;
  }

  // sprintf(str, format, ...) -> siprintf(str, format, ...) if no floating
  // point arguments.
  if (TLI->has(LibFunc::siprintf) && !callHasFloatingPointArgument(CI)) {
    Module *M = B.GetInsertBlock()->getParent()->getParent();
    Constant *SIPrintFFn =
        M->getOrInsertFunction("siprintf", FT, Callee->getAttributes());
    CallInst *New = cast<CallInst>(CI->clone());
    New->setCalledFunction(SIPrintFFn);
    B.Insert(New);
    return New;
  }
  return nullptr;
}

Value *LibCallSimplifier::optimizeFPrintFString(CallInst *CI, IRBuilder<> &B) {
  optimizeErrorReporting(CI, B, 0);

  // All the optimizations depend on the format string.
  StringRef FormatStr;
  if (!getConstantStringInfo(CI->getArgOperand(1), FormatStr))
    return nullptr;

  // Do not do any of the following transformations if the fprintf return
  // value is used, in general the fprintf return value is not compatible
  // with fwrite(), fputc() or fputs().
  if (!CI->use_empty())
    return nullptr;

  // fprintf(F, "foo") --> fwrite("foo", 3, 1, F)
  if (CI->getNumArgOperands() == 2) {
    for (unsigned i = 0, e = FormatStr.size(); i != e; ++i)
      if (FormatStr[i] == '%') // Could handle %% -> % if we cared.
        return nullptr;        // We found a format specifier.

    return EmitFWrite(
        CI->getArgOperand(1),
        ConstantInt::get(DL.getIntPtrType(CI->getContext()), FormatStr.size()),
        CI->getArgOperand(0), B, DL, TLI);
  }

  // The remaining optimizations require the format string to be "%s" or "%c"
  // and have an extra operand.
  if (FormatStr.size() != 2 || FormatStr[0] != '%' ||
      CI->getNumArgOperands() < 3)
    return nullptr;

  // Decode the second character of the format string.
  if (FormatStr[1] == 'c') {
    // fprintf(F, "%c", chr) --> fputc(chr, F)
    if (!CI->getArgOperand(2)->getType()->isIntegerTy())
      return nullptr;
    return EmitFPutC(CI->getArgOperand(2), CI->getArgOperand(0), B, TLI);
  }

  if (FormatStr[1] == 's') {
    // fprintf(F, "%s", str) --> fputs(str, F)
    if (!CI->getArgOperand(2)->getType()->isPointerTy())
      return nullptr;
    return EmitFPutS(CI->getArgOperand(2), CI->getArgOperand(0), B, TLI);
  }
  return nullptr;
}

Value *LibCallSimplifier::optimizeFPrintF(CallInst *CI, IRBuilder<> &B) {
  Function *Callee = CI->getCalledFunction();
  // Require two fixed paramters as pointers and integer result.
  FunctionType *FT = Callee->getFunctionType();
  if (FT->getNumParams() != 2 || !FT->getParamType(0)->isPointerTy() ||
      !FT->getParamType(1)->isPointerTy() ||
      !FT->getReturnType()->isIntegerTy())
    return nullptr;

  if (Value *V = optimizeFPrintFString(CI, B)) {
    return V;
  }

  // fprintf(stream, format, ...) -> fiprintf(stream, format, ...) if no
  // floating point arguments.
  if (TLI->has(LibFunc::fiprintf) && !callHasFloatingPointArgument(CI)) {
    Module *M = B.GetInsertBlock()->getParent()->getParent();
    Constant *FIPrintFFn =
        M->getOrInsertFunction("fiprintf", FT, Callee->getAttributes());
    CallInst *New = cast<CallInst>(CI->clone());
    New->setCalledFunction(FIPrintFFn);
    B.Insert(New);
    return New;
  }
  return nullptr;
}

Value *LibCallSimplifier::optimizeFWrite(CallInst *CI, IRBuilder<> &B) {
  optimizeErrorReporting(CI, B, 3);

  Function *Callee = CI->getCalledFunction();
  // Require a pointer, an integer, an integer, a pointer, returning integer.
  FunctionType *FT = Callee->getFunctionType();
  if (FT->getNumParams() != 4 || !FT->getParamType(0)->isPointerTy() ||
      !FT->getParamType(1)->isIntegerTy() ||
      !FT->getParamType(2)->isIntegerTy() ||
      !FT->getParamType(3)->isPointerTy() ||
      !FT->getReturnType()->isIntegerTy())
    return nullptr;

  // Get the element size and count.
  ConstantInt *SizeC = dyn_cast<ConstantInt>(CI->getArgOperand(1));
  ConstantInt *CountC = dyn_cast<ConstantInt>(CI->getArgOperand(2));
  if (!SizeC || !CountC)
    return nullptr;
  uint64_t Bytes = SizeC->getZExtValue() * CountC->getZExtValue();

  // If this is writing zero records, remove the call (it's a noop).
  if (Bytes == 0)
    return ConstantInt::get(CI->getType(), 0);

  // If this is writing one byte, turn it into fputc.
  // This optimisation is only valid, if the return value is unused.
  if (Bytes == 1 && CI->use_empty()) { // fwrite(S,1,1,F) -> fputc(S[0],F)
    Value *Char = B.CreateLoad(CastToCStr(CI->getArgOperand(0), B), "char");
    Value *NewCI = EmitFPutC(Char, CI->getArgOperand(3), B, TLI);
    return NewCI ? ConstantInt::get(CI->getType(), 1) : nullptr;
  }

  return nullptr;
}

Value *LibCallSimplifier::optimizeFPuts(CallInst *CI, IRBuilder<> &B) {
  optimizeErrorReporting(CI, B, 1);

  Function *Callee = CI->getCalledFunction();

  // Require two pointers.  Also, we can't optimize if return value is used.
  FunctionType *FT = Callee->getFunctionType();
  if (FT->getNumParams() != 2 || !FT->getParamType(0)->isPointerTy() ||
      !FT->getParamType(1)->isPointerTy() || !CI->use_empty())
    return nullptr;

  // fputs(s,F) --> fwrite(s,1,strlen(s),F)
  uint64_t Len = GetStringLength(CI->getArgOperand(0));
  if (!Len)
    return nullptr;

  // Known to have no uses (see above).
  return EmitFWrite(
      CI->getArgOperand(0),
      ConstantInt::get(DL.getIntPtrType(CI->getContext()), Len - 1),
      CI->getArgOperand(1), B, DL, TLI);
}

Value *LibCallSimplifier::optimizePuts(CallInst *CI, IRBuilder<> &B) {
  Function *Callee = CI->getCalledFunction();
  // Require one fixed pointer argument and an integer/void result.
  FunctionType *FT = Callee->getFunctionType();
  if (FT->getNumParams() < 1 || !FT->getParamType(0)->isPointerTy() ||
      !(FT->getReturnType()->isIntegerTy() || FT->getReturnType()->isVoidTy()))
    return nullptr;

  // Check for a constant string.
  StringRef Str;
  if (!getConstantStringInfo(CI->getArgOperand(0), Str))
    return nullptr;

  if (Str.empty() && CI->use_empty()) {
    // puts("") -> putchar('\n')
    Value *Res = EmitPutChar(B.getInt32('\n'), B, TLI);
    if (CI->use_empty() || !Res)
      return Res;
    return B.CreateIntCast(Res, CI->getType(), true);
  }

  return nullptr;
}

bool LibCallSimplifier::hasFloatVersion(StringRef FuncName) {
  LibFunc::Func Func;
  SmallString<20> FloatFuncName = FuncName;
  FloatFuncName += 'f';
  if (TLI->getLibFunc(FloatFuncName, Func))
    return TLI->has(Func);
  return false;
}

Value *LibCallSimplifier::optimizeStringMemoryLibCall(CallInst *CI,
                                                      IRBuilder<> &Builder) {
  LibFunc::Func Func;
  Function *Callee = CI->getCalledFunction();
  StringRef FuncName = Callee->getName();

  // Check for string/memory library functions.
  if (TLI->getLibFunc(FuncName, Func) && TLI->has(Func)) {
    // Make sure we never change the calling convention.
    assert((ignoreCallingConv(Func) ||
            CI->getCallingConv() == llvm::CallingConv::C) &&
      "Optimizing string/memory libcall would change the calling convention");
    switch (Func) {
    case LibFunc::strcat:
      return optimizeStrCat(CI, Builder);
    case LibFunc::strncat:
      return optimizeStrNCat(CI, Builder);
    case LibFunc::strchr:
      return optimizeStrChr(CI, Builder);
    case LibFunc::strrchr:
      return optimizeStrRChr(CI, Builder);
    case LibFunc::strcmp:
      return optimizeStrCmp(CI, Builder);
    case LibFunc::strncmp:
      return optimizeStrNCmp(CI, Builder);
    case LibFunc::strcpy:
      return optimizeStrCpy(CI, Builder);
    case LibFunc::stpcpy:
      return optimizeStpCpy(CI, Builder);
    case LibFunc::strncpy:
      return optimizeStrNCpy(CI, Builder);
    case LibFunc::strlen:
      return optimizeStrLen(CI, Builder);
    case LibFunc::strpbrk:
      return optimizeStrPBrk(CI, Builder);
    case LibFunc::strtol:
    case LibFunc::strtod:
    case LibFunc::strtof:
    case LibFunc::strtoul:
    case LibFunc::strtoll:
    case LibFunc::strtold:
    case LibFunc::strtoull:
      return optimizeStrTo(CI, Builder);
    case LibFunc::strspn:
      return optimizeStrSpn(CI, Builder);
    case LibFunc::strcspn:
      return optimizeStrCSpn(CI, Builder);
    case LibFunc::strstr:
      return optimizeStrStr(CI, Builder);
    case LibFunc::memchr:
      return optimizeMemChr(CI, Builder);
    case LibFunc::memcmp:
      return optimizeMemCmp(CI, Builder);
    case LibFunc::memcpy:
      return optimizeMemCpy(CI, Builder);
    case LibFunc::memmove:
      return optimizeMemMove(CI, Builder);
    case LibFunc::memset:
      return optimizeMemSet(CI, Builder);
    default:
      break;
    }
  }
  return nullptr;
}

Value *LibCallSimplifier::optimizeCall(CallInst *CI) {
  if (CI->isNoBuiltin())
    return nullptr;

  LibFunc::Func Func;
  Function *Callee = CI->getCalledFunction();
  StringRef FuncName = Callee->getName();
  IRBuilder<> Builder(CI);
  bool isCallingConvC = CI->getCallingConv() == llvm::CallingConv::C;

  // Command-line parameter overrides function attribute.
  if (false) // HLSL Change - EnableUnsafeFPShrink.getNumOccurrences() > 0)
    UnsafeFPShrink = EnableUnsafeFPShrink;
  else if (Callee->hasFnAttribute("unsafe-fp-math")) {
    // FIXME: This is the same problem as described in optimizeSqrt().
    // If calls gain access to IR-level FMF, then use that instead of a
    // function attribute.

    // Check for unsafe-fp-math = true.
    Attribute Attr = Callee->getFnAttribute("unsafe-fp-math");
    if (Attr.getValueAsString() == "true")
      UnsafeFPShrink = true;
  }

  // First, check for intrinsics.
  if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(CI)) {
    if (!isCallingConvC)
      return nullptr;
    switch (II->getIntrinsicID()) {
    case Intrinsic::pow:
      return optimizePow(CI, Builder);
    case Intrinsic::exp2:
      return optimizeExp2(CI, Builder);
    case Intrinsic::fabs:
      return optimizeFabs(CI, Builder);
    case Intrinsic::sqrt:
      return optimizeSqrt(CI, Builder);
    default:
      return nullptr;
    }
  }

  // Also try to simplify calls to fortified library functions.
  if (Value *SimplifiedFortifiedCI = FortifiedSimplifier.optimizeCall(CI)) {
    // Try to further simplify the result.
    CallInst *SimplifiedCI = dyn_cast<CallInst>(SimplifiedFortifiedCI);
    if (SimplifiedCI && SimplifiedCI->getCalledFunction())
      if (Value *V = optimizeStringMemoryLibCall(SimplifiedCI, Builder)) {
        // If we were able to further simplify, remove the now redundant call.
        SimplifiedCI->replaceAllUsesWith(V);
        SimplifiedCI->eraseFromParent();
        return V;
      }
    return SimplifiedFortifiedCI;
  }

  // Then check for known library functions.
  if (TLI->getLibFunc(FuncName, Func) && TLI->has(Func)) {
    // We never change the calling convention.
    if (!ignoreCallingConv(Func) && !isCallingConvC)
      return nullptr;
    if (Value *V = optimizeStringMemoryLibCall(CI, Builder))
      return V;
    switch (Func) {
    case LibFunc::cosf:
    case LibFunc::cos:
    case LibFunc::cosl:
      return optimizeCos(CI, Builder);
    case LibFunc::sinpif:
    case LibFunc::sinpi:
    case LibFunc::cospif:
    case LibFunc::cospi:
      return optimizeSinCosPi(CI, Builder);
    case LibFunc::powf:
    case LibFunc::pow:
    case LibFunc::powl:
      return optimizePow(CI, Builder);
    case LibFunc::exp2l:
    case LibFunc::exp2:
    case LibFunc::exp2f:
      return optimizeExp2(CI, Builder);
    case LibFunc::fabsf:
    case LibFunc::fabs:
    case LibFunc::fabsl:
      return optimizeFabs(CI, Builder);
    case LibFunc::sqrtf:
    case LibFunc::sqrt:
    case LibFunc::sqrtl:
      return optimizeSqrt(CI, Builder);
    case LibFunc::ffs:
    case LibFunc::ffsl:
    case LibFunc::ffsll:
      return optimizeFFS(CI, Builder);
    case LibFunc::abs:
    case LibFunc::labs:
    case LibFunc::llabs:
      return optimizeAbs(CI, Builder);
    case LibFunc::isdigit:
      return optimizeIsDigit(CI, Builder);
    case LibFunc::isascii:
      return optimizeIsAscii(CI, Builder);
    case LibFunc::toascii:
      return optimizeToAscii(CI, Builder);
    case LibFunc::printf:
      return optimizePrintF(CI, Builder);
    case LibFunc::sprintf:
      return optimizeSPrintF(CI, Builder);
    case LibFunc::fprintf:
      return optimizeFPrintF(CI, Builder);
    case LibFunc::fwrite:
      return optimizeFWrite(CI, Builder);
    case LibFunc::fputs:
      return optimizeFPuts(CI, Builder);
    case LibFunc::puts:
      return optimizePuts(CI, Builder);
    case LibFunc::perror:
      return optimizeErrorReporting(CI, Builder);
    case LibFunc::vfprintf:
    case LibFunc::fiprintf:
      return optimizeErrorReporting(CI, Builder, 0);
    case LibFunc::fputc:
      return optimizeErrorReporting(CI, Builder, 1);
    case LibFunc::ceil:
    case LibFunc::floor:
    case LibFunc::rint:
    case LibFunc::round:
    case LibFunc::nearbyint:
    case LibFunc::trunc:
      if (hasFloatVersion(FuncName))
        return optimizeUnaryDoubleFP(CI, Builder, false);
      return nullptr;
    case LibFunc::acos:
    case LibFunc::acosh:
    case LibFunc::asin:
    case LibFunc::asinh:
    case LibFunc::atan:
    case LibFunc::atanh:
    case LibFunc::cbrt:
    case LibFunc::cosh:
    case LibFunc::exp:
    case LibFunc::exp10:
    case LibFunc::expm1:
    case LibFunc::log:
    case LibFunc::log10:
    case LibFunc::log1p:
    case LibFunc::log2:
    case LibFunc::logb:
    case LibFunc::sin:
    case LibFunc::sinh:
    case LibFunc::tan:
    case LibFunc::tanh:
      if (UnsafeFPShrink && hasFloatVersion(FuncName))
        return optimizeUnaryDoubleFP(CI, Builder, true);
      return nullptr;
    case LibFunc::copysign:
    case LibFunc::fmin:
    case LibFunc::fmax:
      if (hasFloatVersion(FuncName))
        return optimizeBinaryDoubleFP(CI, Builder);
      return nullptr;
    default:
      return nullptr;
    }
  }
  return nullptr;
}

LibCallSimplifier::LibCallSimplifier(
    const DataLayout &DL, const TargetLibraryInfo *TLI,
    function_ref<void(Instruction *, Value *)> Replacer)
    : FortifiedSimplifier(TLI), DL(DL), TLI(TLI), UnsafeFPShrink(false),
      Replacer(Replacer) {}

void LibCallSimplifier::replaceAllUsesWith(Instruction *I, Value *With) {
  // Indirect through the replacer used in this instance.
  Replacer(I, With);
}

/*static*/ void LibCallSimplifier::replaceAllUsesWithDefault(Instruction *I,
                                                             Value *With) {
  I->replaceAllUsesWith(With);
  I->eraseFromParent();
}

// TODO:
//   Additional cases that we need to add to this file:
//
// cbrt:
//   * cbrt(expN(X))  -> expN(x/3)
//   * cbrt(sqrt(x))  -> pow(x,1/6)
//   * cbrt(sqrt(x))  -> pow(x,1/9)
//
// exp, expf, expl:
//   * exp(log(x))  -> x
//
// log, logf, logl:
//   * log(exp(x))   -> x
//   * log(x**y)     -> y*log(x)
//   * log(exp(y))   -> y*log(e)
//   * log(exp2(y))  -> y*log(2)
//   * log(exp10(y)) -> y*log(10)
//   * log(sqrt(x))  -> 0.5*log(x)
//   * log(pow(x,y)) -> y*log(x)
//
// lround, lroundf, lroundl:
//   * lround(cnst) -> cnst'
//
// pow, powf, powl:
//   * pow(exp(x),y)  -> exp(x*y)
//   * pow(sqrt(x),y) -> pow(x,y*0.5)
//   * pow(pow(x,y),z)-> pow(x,y*z)
//
// round, roundf, roundl:
//   * round(cnst) -> cnst'
//
// signbit:
//   * signbit(cnst) -> cnst'
//   * signbit(nncst) -> 0 (if pstv is a non-negative constant)
//
// sqrt, sqrtf, sqrtl:
//   * sqrt(expN(x))  -> expN(x*0.5)
//   * sqrt(Nroot(x)) -> pow(x,1/(2*N))
//   * sqrt(pow(x,y)) -> pow(|x|,y*0.5)
//
// tan, tanf, tanl:
//   * tan(atan(x)) -> x
//
// trunc, truncf, truncl:
//   * trunc(cnst) -> cnst'
//
//

//===----------------------------------------------------------------------===//
// Fortified Library Call Optimizations
//===----------------------------------------------------------------------===//

bool FortifiedLibCallSimplifier::isFortifiedCallFoldable(CallInst *CI,
                                                         unsigned ObjSizeOp,
                                                         unsigned SizeOp,
                                                         bool isString) {
  if (CI->getArgOperand(ObjSizeOp) == CI->getArgOperand(SizeOp))
    return true;
  if (ConstantInt *ObjSizeCI =
          dyn_cast<ConstantInt>(CI->getArgOperand(ObjSizeOp))) {
    if (ObjSizeCI->isAllOnesValue())
      return true;
    // If the object size wasn't -1 (unknown), bail out if we were asked to.
    if (OnlyLowerUnknownSize)
      return false;
    if (isString) {
      uint64_t Len = GetStringLength(CI->getArgOperand(SizeOp));
      // If the length is 0 we don't know how long it is and so we can't
      // remove the check.
      if (Len == 0)
        return false;
      return ObjSizeCI->getZExtValue() >= Len;
    }
    if (ConstantInt *SizeCI = dyn_cast<ConstantInt>(CI->getArgOperand(SizeOp)))
      return ObjSizeCI->getZExtValue() >= SizeCI->getZExtValue();
  }
  return false;
}

Value *FortifiedLibCallSimplifier::optimizeMemCpyChk(CallInst *CI, IRBuilder<> &B) {
  Function *Callee = CI->getCalledFunction();

  if (!checkStringCopyLibFuncSignature(Callee, LibFunc::memcpy_chk))
    return nullptr;

  if (isFortifiedCallFoldable(CI, 3, 2, false)) {
    B.CreateMemCpy(CI->getArgOperand(0), CI->getArgOperand(1),
                   CI->getArgOperand(2), 1);
    return CI->getArgOperand(0);
  }
  return nullptr;
}

Value *FortifiedLibCallSimplifier::optimizeMemMoveChk(CallInst *CI, IRBuilder<> &B) {
  Function *Callee = CI->getCalledFunction();

  if (!checkStringCopyLibFuncSignature(Callee, LibFunc::memmove_chk))
    return nullptr;

  if (isFortifiedCallFoldable(CI, 3, 2, false)) {
    B.CreateMemMove(CI->getArgOperand(0), CI->getArgOperand(1),
                    CI->getArgOperand(2), 1);
    return CI->getArgOperand(0);
  }
  return nullptr;
}

Value *FortifiedLibCallSimplifier::optimizeMemSetChk(CallInst *CI, IRBuilder<> &B) {
  Function *Callee = CI->getCalledFunction();

  if (!checkStringCopyLibFuncSignature(Callee, LibFunc::memset_chk))
    return nullptr;

  if (isFortifiedCallFoldable(CI, 3, 2, false)) {
    Value *Val = B.CreateIntCast(CI->getArgOperand(1), B.getInt8Ty(), false);
    B.CreateMemSet(CI->getArgOperand(0), Val, CI->getArgOperand(2), 1);
    return CI->getArgOperand(0);
  }
  return nullptr;
}

Value *FortifiedLibCallSimplifier::optimizeStrpCpyChk(CallInst *CI,
                                                      IRBuilder<> &B,
                                                      LibFunc::Func Func) {
  Function *Callee = CI->getCalledFunction();
  StringRef Name = Callee->getName();
  const DataLayout &DL = CI->getModule()->getDataLayout();

  if (!checkStringCopyLibFuncSignature(Callee, Func))
    return nullptr;

  Value *Dst = CI->getArgOperand(0), *Src = CI->getArgOperand(1),
        *ObjSize = CI->getArgOperand(2);

  // __stpcpy_chk(x,x,...)  -> x+strlen(x)
  if (Func == LibFunc::stpcpy_chk && !OnlyLowerUnknownSize && Dst == Src) {
    Value *StrLen = EmitStrLen(Src, B, DL, TLI);
    return StrLen ? B.CreateInBoundsGEP(B.getInt8Ty(), Dst, StrLen) : nullptr;
  }

  // If a) we don't have any length information, or b) we know this will
  // fit then just lower to a plain st[rp]cpy. Otherwise we'll keep our
  // st[rp]cpy_chk call which may fail at runtime if the size is too long.
  // TODO: It might be nice to get a maximum length out of the possible
  // string lengths for varying.
  if (isFortifiedCallFoldable(CI, 2, 1, true))
    return EmitStrCpy(Dst, Src, B, TLI, Name.substr(2, 6));

  if (OnlyLowerUnknownSize)
    return nullptr;

  // Maybe we can stil fold __st[rp]cpy_chk to __memcpy_chk.
  uint64_t Len = GetStringLength(Src);
  if (Len == 0)
    return nullptr;

  Type *SizeTTy = DL.getIntPtrType(CI->getContext());
  Value *LenV = ConstantInt::get(SizeTTy, Len);
  Value *Ret = EmitMemCpyChk(Dst, Src, LenV, ObjSize, B, DL, TLI);
  // If the function was an __stpcpy_chk, and we were able to fold it into
  // a __memcpy_chk, we still need to return the correct end pointer.
  if (Ret && Func == LibFunc::stpcpy_chk)
    return B.CreateGEP(B.getInt8Ty(), Dst, ConstantInt::get(SizeTTy, Len - 1));
  return Ret;
}

Value *FortifiedLibCallSimplifier::optimizeStrpNCpyChk(CallInst *CI,
                                                       IRBuilder<> &B,
                                                       LibFunc::Func Func) {
  Function *Callee = CI->getCalledFunction();
  StringRef Name = Callee->getName();

  if (!checkStringCopyLibFuncSignature(Callee, Func))
    return nullptr;
  if (isFortifiedCallFoldable(CI, 3, 2, false)) {
    Value *Ret = EmitStrNCpy(CI->getArgOperand(0), CI->getArgOperand(1),
                             CI->getArgOperand(2), B, TLI, Name.substr(2, 7));
    return Ret;
  }
  return nullptr;
}

Value *FortifiedLibCallSimplifier::optimizeCall(CallInst *CI) {
  // FIXME: We shouldn't be changing "nobuiltin" or TLI unavailable calls here.
  // Some clang users checked for _chk libcall availability using:
  //   __has_builtin(__builtin___memcpy_chk)
  // When compiling with -fno-builtin, this is always true.
  // When passing -ffreestanding/-mkernel, which both imply -fno-builtin, we
  // end up with fortified libcalls, which isn't acceptable in a freestanding
  // environment which only provides their non-fortified counterparts.
  //
  // Until we change clang and/or teach external users to check for availability
  // differently, disregard the "nobuiltin" attribute and TLI::has.
  //
  // PR23093.

  LibFunc::Func Func;
  Function *Callee = CI->getCalledFunction();
  StringRef FuncName = Callee->getName();
  IRBuilder<> Builder(CI);
  bool isCallingConvC = CI->getCallingConv() == llvm::CallingConv::C;

  // First, check that this is a known library functions.
  if (!TLI->getLibFunc(FuncName, Func))
    return nullptr;

  // We never change the calling convention.
  if (!ignoreCallingConv(Func) && !isCallingConvC)
    return nullptr;

  switch (Func) {
  case LibFunc::memcpy_chk:
    return optimizeMemCpyChk(CI, Builder);
  case LibFunc::memmove_chk:
    return optimizeMemMoveChk(CI, Builder);
  case LibFunc::memset_chk:
    return optimizeMemSetChk(CI, Builder);
  case LibFunc::stpcpy_chk:
  case LibFunc::strcpy_chk:
    return optimizeStrpCpyChk(CI, Builder, Func);
  case LibFunc::stpncpy_chk:
  case LibFunc::strncpy_chk:
    return optimizeStrpNCpyChk(CI, Builder, Func);
  default:
    break;
  }
  return nullptr;
}

FortifiedLibCallSimplifier::FortifiedLibCallSimplifier(
    const TargetLibraryInfo *TLI, bool OnlyLowerUnknownSize)
    : TLI(TLI), OnlyLowerUnknownSize(OnlyLowerUnknownSize) {}

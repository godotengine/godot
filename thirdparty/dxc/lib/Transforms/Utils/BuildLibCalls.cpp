//===- BuildLibCalls.cpp - Utility builder for libcalls -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements some functions that will create standard C libcalls.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/BuildLibCalls.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Analysis/TargetLibraryInfo.h"

using namespace llvm;

/// CastToCStr - Return V if it is an i8*, otherwise cast it to i8*.
Value *llvm::CastToCStr(Value *V, IRBuilder<> &B) {
  unsigned AS = V->getType()->getPointerAddressSpace();
  return B.CreateBitCast(V, B.getInt8PtrTy(AS), "cstr");
}

/// EmitStrLen - Emit a call to the strlen function to the builder, for the
/// specified pointer.  This always returns an integer value of size intptr_t.
Value *llvm::EmitStrLen(Value *Ptr, IRBuilder<> &B, const DataLayout &DL,
                        const TargetLibraryInfo *TLI) {
  if (!TLI->has(LibFunc::strlen))
    return nullptr;

  Module *M = B.GetInsertBlock()->getParent()->getParent();
  AttributeSet AS[2];
  AS[0] = AttributeSet::get(M->getContext(), 1, Attribute::NoCapture);
  Attribute::AttrKind AVs[2] = { Attribute::ReadOnly, Attribute::NoUnwind };
  AS[1] = AttributeSet::get(M->getContext(), AttributeSet::FunctionIndex, AVs);

  LLVMContext &Context = B.GetInsertBlock()->getContext();
  Constant *StrLen = M->getOrInsertFunction(
      "strlen", AttributeSet::get(M->getContext(), AS),
      DL.getIntPtrType(Context), B.getInt8PtrTy(), nullptr);
  CallInst *CI = B.CreateCall(StrLen, CastToCStr(Ptr, B), "strlen");
  if (const Function *F = dyn_cast<Function>(StrLen->stripPointerCasts()))
    CI->setCallingConv(F->getCallingConv());

  return CI;
}

/// EmitStrNLen - Emit a call to the strnlen function to the builder, for the
/// specified pointer.  Ptr is required to be some pointer type, MaxLen must
/// be of size_t type, and the return value has 'intptr_t' type.
Value *llvm::EmitStrNLen(Value *Ptr, Value *MaxLen, IRBuilder<> &B,
                         const DataLayout &DL, const TargetLibraryInfo *TLI) {
  if (!TLI->has(LibFunc::strnlen))
    return nullptr;

  Module *M = B.GetInsertBlock()->getParent()->getParent();
  AttributeSet AS[2];
  AS[0] = AttributeSet::get(M->getContext(), 1, Attribute::NoCapture);
  Attribute::AttrKind AVs[2] = { Attribute::ReadOnly, Attribute::NoUnwind };
  AS[1] = AttributeSet::get(M->getContext(), AttributeSet::FunctionIndex, AVs);

  LLVMContext &Context = B.GetInsertBlock()->getContext();
  Constant *StrNLen =
      M->getOrInsertFunction("strnlen", AttributeSet::get(M->getContext(), AS),
                             DL.getIntPtrType(Context), B.getInt8PtrTy(),
                             DL.getIntPtrType(Context), nullptr);
  CallInst *CI = B.CreateCall(StrNLen, {CastToCStr(Ptr, B), MaxLen}, "strnlen");
  if (const Function *F = dyn_cast<Function>(StrNLen->stripPointerCasts()))
    CI->setCallingConv(F->getCallingConv());

  return CI;
}

/// EmitStrChr - Emit a call to the strchr function to the builder, for the
/// specified pointer and character.  Ptr is required to be some pointer type,
/// and the return value has 'i8*' type.
Value *llvm::EmitStrChr(Value *Ptr, char C, IRBuilder<> &B,
                        const TargetLibraryInfo *TLI) {
  if (!TLI->has(LibFunc::strchr))
    return nullptr;

  Module *M = B.GetInsertBlock()->getParent()->getParent();
  Attribute::AttrKind AVs[2] = { Attribute::ReadOnly, Attribute::NoUnwind };
  AttributeSet AS =
    AttributeSet::get(M->getContext(), AttributeSet::FunctionIndex, AVs);

  Type *I8Ptr = B.getInt8PtrTy();
  Type *I32Ty = B.getInt32Ty();
  Constant *StrChr = M->getOrInsertFunction("strchr",
                                            AttributeSet::get(M->getContext(),
                                                             AS),
                                            I8Ptr, I8Ptr, I32Ty, nullptr);
  CallInst *CI = B.CreateCall(
      StrChr, {CastToCStr(Ptr, B), ConstantInt::get(I32Ty, C)}, "strchr");
  if (const Function *F = dyn_cast<Function>(StrChr->stripPointerCasts()))
    CI->setCallingConv(F->getCallingConv());
  return CI;
}

/// EmitStrNCmp - Emit a call to the strncmp function to the builder.
Value *llvm::EmitStrNCmp(Value *Ptr1, Value *Ptr2, Value *Len, IRBuilder<> &B,
                         const DataLayout &DL, const TargetLibraryInfo *TLI) {
  if (!TLI->has(LibFunc::strncmp))
    return nullptr;

  Module *M = B.GetInsertBlock()->getParent()->getParent();
  AttributeSet AS[3];
  AS[0] = AttributeSet::get(M->getContext(), 1, Attribute::NoCapture);
  AS[1] = AttributeSet::get(M->getContext(), 2, Attribute::NoCapture);
  Attribute::AttrKind AVs[2] = { Attribute::ReadOnly, Attribute::NoUnwind };
  AS[2] = AttributeSet::get(M->getContext(), AttributeSet::FunctionIndex, AVs);

  LLVMContext &Context = B.GetInsertBlock()->getContext();
  Value *StrNCmp = M->getOrInsertFunction(
      "strncmp", AttributeSet::get(M->getContext(), AS), B.getInt32Ty(),
      B.getInt8PtrTy(), B.getInt8PtrTy(), DL.getIntPtrType(Context), nullptr);
  CallInst *CI = B.CreateCall(
      StrNCmp, {CastToCStr(Ptr1, B), CastToCStr(Ptr2, B), Len}, "strncmp");

  if (const Function *F = dyn_cast<Function>(StrNCmp->stripPointerCasts()))
    CI->setCallingConv(F->getCallingConv());

  return CI;
}

/// EmitStrCpy - Emit a call to the strcpy function to the builder, for the
/// specified pointer arguments.
Value *llvm::EmitStrCpy(Value *Dst, Value *Src, IRBuilder<> &B,
                        const TargetLibraryInfo *TLI, StringRef Name) {
  if (!TLI->has(LibFunc::strcpy))
    return nullptr;

  Module *M = B.GetInsertBlock()->getParent()->getParent();
  AttributeSet AS[2];
  AS[0] = AttributeSet::get(M->getContext(), 2, Attribute::NoCapture);
  AS[1] = AttributeSet::get(M->getContext(), AttributeSet::FunctionIndex,
                            Attribute::NoUnwind);
  Type *I8Ptr = B.getInt8PtrTy();
  Value *StrCpy = M->getOrInsertFunction(Name,
                                         AttributeSet::get(M->getContext(), AS),
                                         I8Ptr, I8Ptr, I8Ptr, nullptr);
  CallInst *CI =
      B.CreateCall(StrCpy, {CastToCStr(Dst, B), CastToCStr(Src, B)}, Name);
  if (const Function *F = dyn_cast<Function>(StrCpy->stripPointerCasts()))
    CI->setCallingConv(F->getCallingConv());
  return CI;
}

/// EmitStrNCpy - Emit a call to the strncpy function to the builder, for the
/// specified pointer arguments.
Value *llvm::EmitStrNCpy(Value *Dst, Value *Src, Value *Len, IRBuilder<> &B,
                         const TargetLibraryInfo *TLI, StringRef Name) {
  if (!TLI->has(LibFunc::strncpy))
    return nullptr;

  Module *M = B.GetInsertBlock()->getParent()->getParent();
  AttributeSet AS[2];
  AS[0] = AttributeSet::get(M->getContext(), 2, Attribute::NoCapture);
  AS[1] = AttributeSet::get(M->getContext(), AttributeSet::FunctionIndex,
                            Attribute::NoUnwind);
  Type *I8Ptr = B.getInt8PtrTy();
  Value *StrNCpy = M->getOrInsertFunction(Name,
                                          AttributeSet::get(M->getContext(),
                                                            AS),
                                          I8Ptr, I8Ptr, I8Ptr,
                                          Len->getType(), nullptr);
  CallInst *CI = B.CreateCall(
      StrNCpy, {CastToCStr(Dst, B), CastToCStr(Src, B), Len}, "strncpy");
  if (const Function *F = dyn_cast<Function>(StrNCpy->stripPointerCasts()))
    CI->setCallingConv(F->getCallingConv());
  return CI;
}

/// EmitMemCpyChk - Emit a call to the __memcpy_chk function to the builder.
/// This expects that the Len and ObjSize have type 'intptr_t' and Dst/Src
/// are pointers.
Value *llvm::EmitMemCpyChk(Value *Dst, Value *Src, Value *Len, Value *ObjSize,
                           IRBuilder<> &B, const DataLayout &DL,
                           const TargetLibraryInfo *TLI) {
  if (!TLI->has(LibFunc::memcpy_chk))
    return nullptr;

  Module *M = B.GetInsertBlock()->getParent()->getParent();
  AttributeSet AS;
  AS = AttributeSet::get(M->getContext(), AttributeSet::FunctionIndex,
                         Attribute::NoUnwind);
  LLVMContext &Context = B.GetInsertBlock()->getContext();
  Value *MemCpy = M->getOrInsertFunction(
      "__memcpy_chk", AttributeSet::get(M->getContext(), AS), B.getInt8PtrTy(),
      B.getInt8PtrTy(), B.getInt8PtrTy(), DL.getIntPtrType(Context),
      DL.getIntPtrType(Context), nullptr);
  Dst = CastToCStr(Dst, B);
  Src = CastToCStr(Src, B);
  CallInst *CI = B.CreateCall(MemCpy, {Dst, Src, Len, ObjSize});
  if (const Function *F = dyn_cast<Function>(MemCpy->stripPointerCasts()))
    CI->setCallingConv(F->getCallingConv());
  return CI;
}

/// EmitMemChr - Emit a call to the memchr function.  This assumes that Ptr is
/// a pointer, Val is an i32 value, and Len is an 'intptr_t' value.
Value *llvm::EmitMemChr(Value *Ptr, Value *Val, Value *Len, IRBuilder<> &B,
                        const DataLayout &DL, const TargetLibraryInfo *TLI) {
  if (!TLI->has(LibFunc::memchr))
    return nullptr;

  Module *M = B.GetInsertBlock()->getParent()->getParent();
  AttributeSet AS;
  Attribute::AttrKind AVs[2] = { Attribute::ReadOnly, Attribute::NoUnwind };
  AS = AttributeSet::get(M->getContext(), AttributeSet::FunctionIndex, AVs);
  LLVMContext &Context = B.GetInsertBlock()->getContext();
  Value *MemChr = M->getOrInsertFunction(
      "memchr", AttributeSet::get(M->getContext(), AS), B.getInt8PtrTy(),
      B.getInt8PtrTy(), B.getInt32Ty(), DL.getIntPtrType(Context), nullptr);
  CallInst *CI = B.CreateCall(MemChr, {CastToCStr(Ptr, B), Val, Len}, "memchr");

  if (const Function *F = dyn_cast<Function>(MemChr->stripPointerCasts()))
    CI->setCallingConv(F->getCallingConv());

  return CI;
}

/// EmitMemCmp - Emit a call to the memcmp function.
Value *llvm::EmitMemCmp(Value *Ptr1, Value *Ptr2, Value *Len, IRBuilder<> &B,
                        const DataLayout &DL, const TargetLibraryInfo *TLI) {
  if (!TLI->has(LibFunc::memcmp))
    return nullptr;

  Module *M = B.GetInsertBlock()->getParent()->getParent();
  AttributeSet AS[3];
  AS[0] = AttributeSet::get(M->getContext(), 1, Attribute::NoCapture);
  AS[1] = AttributeSet::get(M->getContext(), 2, Attribute::NoCapture);
  Attribute::AttrKind AVs[2] = { Attribute::ReadOnly, Attribute::NoUnwind };
  AS[2] = AttributeSet::get(M->getContext(), AttributeSet::FunctionIndex, AVs);

  LLVMContext &Context = B.GetInsertBlock()->getContext();
  Value *MemCmp = M->getOrInsertFunction(
      "memcmp", AttributeSet::get(M->getContext(), AS), B.getInt32Ty(),
      B.getInt8PtrTy(), B.getInt8PtrTy(), DL.getIntPtrType(Context), nullptr);
  CallInst *CI = B.CreateCall(
      MemCmp, {CastToCStr(Ptr1, B), CastToCStr(Ptr2, B), Len}, "memcmp");

  if (const Function *F = dyn_cast<Function>(MemCmp->stripPointerCasts()))
    CI->setCallingConv(F->getCallingConv());

  return CI;
}

/// Append a suffix to the function name according to the type of 'Op'.
static void AppendTypeSuffix(Value *Op, StringRef &Name, SmallString<20> &NameBuffer) {
  if (!Op->getType()->isDoubleTy()) {
      NameBuffer += Name;

    if (Op->getType()->isFloatTy())
      NameBuffer += 'f';
    else
      NameBuffer += 'l';

    Name = NameBuffer;
  }  
  return;
}

/// EmitUnaryFloatFnCall - Emit a call to the unary function named 'Name' (e.g.
/// 'floor').  This function is known to take a single of type matching 'Op' and
/// returns one value with the same type.  If 'Op' is a long double, 'l' is
/// added as the suffix of name, if 'Op' is a float, we add a 'f' suffix.
Value *llvm::EmitUnaryFloatFnCall(Value *Op, StringRef Name, IRBuilder<> &B,
                                  const AttributeSet &Attrs) {
  SmallString<20> NameBuffer;
  AppendTypeSuffix(Op, Name, NameBuffer);   

  Module *M = B.GetInsertBlock()->getParent()->getParent();
  Value *Callee = M->getOrInsertFunction(Name, Op->getType(),
                                         Op->getType(), nullptr);
  CallInst *CI = B.CreateCall(Callee, Op, Name);
  CI->setAttributes(Attrs);
  if (const Function *F = dyn_cast<Function>(Callee->stripPointerCasts()))
    CI->setCallingConv(F->getCallingConv());

  return CI;
}

/// EmitBinaryFloatFnCall - Emit a call to the binary function named 'Name'
/// (e.g. 'fmin').  This function is known to take type matching 'Op1' and 'Op2'
/// and return one value with the same type.  If 'Op1/Op2' are long double, 'l'
/// is added as the suffix of name, if 'Op1/Op2' is a float, we add a 'f'
/// suffix.
Value *llvm::EmitBinaryFloatFnCall(Value *Op1, Value *Op2, StringRef Name,
                                  IRBuilder<> &B, const AttributeSet &Attrs) {
  SmallString<20> NameBuffer;
  AppendTypeSuffix(Op1, Name, NameBuffer);   

  Module *M = B.GetInsertBlock()->getParent()->getParent();
  Value *Callee = M->getOrInsertFunction(Name, Op1->getType(),
                                         Op1->getType(), Op2->getType(), nullptr);
  CallInst *CI = B.CreateCall(Callee, {Op1, Op2}, Name);
  CI->setAttributes(Attrs);
  if (const Function *F = dyn_cast<Function>(Callee->stripPointerCasts()))
    CI->setCallingConv(F->getCallingConv());

  return CI;
}

/// EmitPutChar - Emit a call to the putchar function.  This assumes that Char
/// is an integer.
Value *llvm::EmitPutChar(Value *Char, IRBuilder<> &B,
                         const TargetLibraryInfo *TLI) {
  if (!TLI->has(LibFunc::putchar))
    return nullptr;

  Module *M = B.GetInsertBlock()->getParent()->getParent();
  Value *PutChar = M->getOrInsertFunction("putchar", B.getInt32Ty(),
                                          B.getInt32Ty(), nullptr);
  CallInst *CI = B.CreateCall(PutChar,
                              B.CreateIntCast(Char,
                              B.getInt32Ty(),
                              /*isSigned*/true,
                              "chari"),
                              "putchar");

  if (const Function *F = dyn_cast<Function>(PutChar->stripPointerCasts()))
    CI->setCallingConv(F->getCallingConv());
  return CI;
}

/// EmitPutS - Emit a call to the puts function.  This assumes that Str is
/// some pointer.
Value *llvm::EmitPutS(Value *Str, IRBuilder<> &B,
                      const TargetLibraryInfo *TLI) {
  if (!TLI->has(LibFunc::puts))
    return nullptr;

  Module *M = B.GetInsertBlock()->getParent()->getParent();
  AttributeSet AS[2];
  AS[0] = AttributeSet::get(M->getContext(), 1, Attribute::NoCapture);
  AS[1] = AttributeSet::get(M->getContext(), AttributeSet::FunctionIndex,
                            Attribute::NoUnwind);

  Value *PutS = M->getOrInsertFunction("puts",
                                       AttributeSet::get(M->getContext(), AS),
                                       B.getInt32Ty(),
                                       B.getInt8PtrTy(),
                                       nullptr);
  CallInst *CI = B.CreateCall(PutS, CastToCStr(Str, B), "puts");
  if (const Function *F = dyn_cast<Function>(PutS->stripPointerCasts()))
    CI->setCallingConv(F->getCallingConv());
  return CI;
}

/// EmitFPutC - Emit a call to the fputc function.  This assumes that Char is
/// an integer and File is a pointer to FILE.
Value *llvm::EmitFPutC(Value *Char, Value *File, IRBuilder<> &B,
                       const TargetLibraryInfo *TLI) {
  if (!TLI->has(LibFunc::fputc))
    return nullptr;

  Module *M = B.GetInsertBlock()->getParent()->getParent();
  AttributeSet AS[2];
  AS[0] = AttributeSet::get(M->getContext(), 2, Attribute::NoCapture);
  AS[1] = AttributeSet::get(M->getContext(), AttributeSet::FunctionIndex,
                            Attribute::NoUnwind);
  Constant *F;
  if (File->getType()->isPointerTy())
    F = M->getOrInsertFunction("fputc",
                               AttributeSet::get(M->getContext(), AS),
                               B.getInt32Ty(),
                               B.getInt32Ty(), File->getType(),
                               nullptr);
  else
    F = M->getOrInsertFunction("fputc",
                               B.getInt32Ty(),
                               B.getInt32Ty(),
                               File->getType(), nullptr);
  Char = B.CreateIntCast(Char, B.getInt32Ty(), /*isSigned*/true,
                         "chari");
  CallInst *CI = B.CreateCall(F, {Char, File}, "fputc");

  if (const Function *Fn = dyn_cast<Function>(F->stripPointerCasts()))
    CI->setCallingConv(Fn->getCallingConv());
  return CI;
}

/// EmitFPutS - Emit a call to the puts function.  Str is required to be a
/// pointer and File is a pointer to FILE.
Value *llvm::EmitFPutS(Value *Str, Value *File, IRBuilder<> &B,
                       const TargetLibraryInfo *TLI) {
  if (!TLI->has(LibFunc::fputs))
    return nullptr;

  Module *M = B.GetInsertBlock()->getParent()->getParent();
  AttributeSet AS[3];
  AS[0] = AttributeSet::get(M->getContext(), 1, Attribute::NoCapture);
  AS[1] = AttributeSet::get(M->getContext(), 2, Attribute::NoCapture);
  AS[2] = AttributeSet::get(M->getContext(), AttributeSet::FunctionIndex,
                            Attribute::NoUnwind);
  StringRef FPutsName = TLI->getName(LibFunc::fputs);
  Constant *F;
  if (File->getType()->isPointerTy())
    F = M->getOrInsertFunction(FPutsName,
                               AttributeSet::get(M->getContext(), AS),
                               B.getInt32Ty(),
                               B.getInt8PtrTy(),
                               File->getType(), nullptr);
  else
    F = M->getOrInsertFunction(FPutsName, B.getInt32Ty(),
                               B.getInt8PtrTy(),
                               File->getType(), nullptr);
  CallInst *CI = B.CreateCall(F, {CastToCStr(Str, B), File}, "fputs");

  if (const Function *Fn = dyn_cast<Function>(F->stripPointerCasts()))
    CI->setCallingConv(Fn->getCallingConv());
  return CI;
}

/// EmitFWrite - Emit a call to the fwrite function.  This assumes that Ptr is
/// a pointer, Size is an 'intptr_t', and File is a pointer to FILE.
Value *llvm::EmitFWrite(Value *Ptr, Value *Size, Value *File, IRBuilder<> &B,
                        const DataLayout &DL, const TargetLibraryInfo *TLI) {
  if (!TLI->has(LibFunc::fwrite))
    return nullptr;

  Module *M = B.GetInsertBlock()->getParent()->getParent();
  AttributeSet AS[3];
  AS[0] = AttributeSet::get(M->getContext(), 1, Attribute::NoCapture);
  AS[1] = AttributeSet::get(M->getContext(), 4, Attribute::NoCapture);
  AS[2] = AttributeSet::get(M->getContext(), AttributeSet::FunctionIndex,
                            Attribute::NoUnwind);
  LLVMContext &Context = B.GetInsertBlock()->getContext();
  StringRef FWriteName = TLI->getName(LibFunc::fwrite);
  Constant *F;
  if (File->getType()->isPointerTy())
    F = M->getOrInsertFunction(
        FWriteName, AttributeSet::get(M->getContext(), AS),
        DL.getIntPtrType(Context), B.getInt8PtrTy(), DL.getIntPtrType(Context),
        DL.getIntPtrType(Context), File->getType(), nullptr);
  else
    F = M->getOrInsertFunction(FWriteName, DL.getIntPtrType(Context),
                               B.getInt8PtrTy(), DL.getIntPtrType(Context),
                               DL.getIntPtrType(Context), File->getType(),
                               nullptr);
  CallInst *CI =
      B.CreateCall(F, {CastToCStr(Ptr, B), Size,
                       ConstantInt::get(DL.getIntPtrType(Context), 1), File});

  if (const Function *Fn = dyn_cast<Function>(F->stripPointerCasts()))
    CI->setCallingConv(Fn->getCallingConv());
  return CI;
}

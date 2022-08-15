//===- BuildLibCalls.h - Utility builder for libcalls -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file exposes an interface to build some C language libcalls for
// optimization passes that need to call the various functions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_BUILDLIBCALLS_H
#define LLVM_TRANSFORMS_UTILS_BUILDLIBCALLS_H

#include "llvm/IR/IRBuilder.h"

namespace llvm {
  class Value;
  class DataLayout;
  class TargetLibraryInfo;

  /// CastToCStr - Return V if it is an i8*, otherwise cast it to i8*.
  Value *CastToCStr(Value *V, IRBuilder<> &B);

  /// EmitStrLen - Emit a call to the strlen function to the builder, for the
  /// specified pointer.  Ptr is required to be some pointer type, and the
  /// return value has 'intptr_t' type.
  Value *EmitStrLen(Value *Ptr, IRBuilder<> &B, const DataLayout &DL,
                    const TargetLibraryInfo *TLI);

  /// EmitStrNLen - Emit a call to the strnlen function to the builder, for the
  /// specified pointer.  Ptr is required to be some pointer type, MaxLen must
  /// be of size_t type, and the return value has 'intptr_t' type.
  Value *EmitStrNLen(Value *Ptr, Value *MaxLen, IRBuilder<> &B,
                     const DataLayout &DL, const TargetLibraryInfo *TLI);

  /// EmitStrChr - Emit a call to the strchr function to the builder, for the
  /// specified pointer and character.  Ptr is required to be some pointer type,
  /// and the return value has 'i8*' type.
  Value *EmitStrChr(Value *Ptr, char C, IRBuilder<> &B,
                    const TargetLibraryInfo *TLI);

  /// EmitStrNCmp - Emit a call to the strncmp function to the builder.
  Value *EmitStrNCmp(Value *Ptr1, Value *Ptr2, Value *Len, IRBuilder<> &B,
                     const DataLayout &DL, const TargetLibraryInfo *TLI);

  /// EmitStrCpy - Emit a call to the strcpy function to the builder, for the
  /// specified pointer arguments.
  Value *EmitStrCpy(Value *Dst, Value *Src, IRBuilder<> &B,
                    const TargetLibraryInfo *TLI, StringRef Name = "strcpy");

  /// EmitStrNCpy - Emit a call to the strncpy function to the builder, for the
  /// specified pointer arguments and length.
  Value *EmitStrNCpy(Value *Dst, Value *Src, Value *Len, IRBuilder<> &B,
                     const TargetLibraryInfo *TLI, StringRef Name = "strncpy");

  /// EmitMemCpyChk - Emit a call to the __memcpy_chk function to the builder.
  /// This expects that the Len and ObjSize have type 'intptr_t' and Dst/Src
  /// are pointers.
  Value *EmitMemCpyChk(Value *Dst, Value *Src, Value *Len, Value *ObjSize,
                       IRBuilder<> &B, const DataLayout &DL,
                       const TargetLibraryInfo *TLI);

  /// EmitMemChr - Emit a call to the memchr function.  This assumes that Ptr is
  /// a pointer, Val is an i32 value, and Len is an 'intptr_t' value.
  Value *EmitMemChr(Value *Ptr, Value *Val, Value *Len, IRBuilder<> &B,
                    const DataLayout &DL, const TargetLibraryInfo *TLI);

  /// EmitMemCmp - Emit a call to the memcmp function.
  Value *EmitMemCmp(Value *Ptr1, Value *Ptr2, Value *Len, IRBuilder<> &B,
                    const DataLayout &DL, const TargetLibraryInfo *TLI);

  /// EmitUnaryFloatFnCall - Emit a call to the unary function named 'Name'
  /// (e.g.  'floor').  This function is known to take a single of type matching
  /// 'Op' and returns one value with the same type.  If 'Op' is a long double,
  /// 'l' is added as the suffix of name, if 'Op' is a float, we add a 'f'
  /// suffix.
  Value *EmitUnaryFloatFnCall(Value *Op, StringRef Name, IRBuilder<> &B,
                              const AttributeSet &Attrs);

  /// EmitUnaryFloatFnCall - Emit a call to the binary function named 'Name'
  /// (e.g. 'fmin').  This function is known to take type matching 'Op1' and
  /// 'Op2' and return one value with the same type.  If 'Op1/Op2' are long
  /// double, 'l' is added as the suffix of name, if 'Op1/Op2' are float, we
  /// add a 'f' suffix.
  Value *EmitBinaryFloatFnCall(Value *Op1, Value *Op2, StringRef Name,
                                  IRBuilder<> &B, const AttributeSet &Attrs);

  /// EmitPutChar - Emit a call to the putchar function.  This assumes that Char
  /// is an integer.
  Value *EmitPutChar(Value *Char, IRBuilder<> &B, const TargetLibraryInfo *TLI);

  /// EmitPutS - Emit a call to the puts function.  This assumes that Str is
  /// some pointer.
  Value *EmitPutS(Value *Str, IRBuilder<> &B, const TargetLibraryInfo *TLI);

  /// EmitFPutC - Emit a call to the fputc function.  This assumes that Char is
  /// an i32, and File is a pointer to FILE.
  Value *EmitFPutC(Value *Char, Value *File, IRBuilder<> &B,
                   const TargetLibraryInfo *TLI);

  /// EmitFPutS - Emit a call to the puts function.  Str is required to be a
  /// pointer and File is a pointer to FILE.
  Value *EmitFPutS(Value *Str, Value *File, IRBuilder<> &B,
                   const TargetLibraryInfo *TLI);

  /// EmitFWrite - Emit a call to the fwrite function.  This assumes that Ptr is
  /// a pointer, Size is an 'intptr_t', and File is a pointer to FILE.
  Value *EmitFWrite(Value *Ptr, Value *Size, Value *File, IRBuilder<> &B,
                    const DataLayout &DL, const TargetLibraryInfo *TLI);
}

#endif

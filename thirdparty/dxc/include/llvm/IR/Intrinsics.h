//===-- llvm/Instrinsics.h - LLVM Intrinsic Function Handling ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a set of enums which allow processing of intrinsic
// functions.  Values of these enum types are returned by
// Function::getIntrinsicID.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_INTRINSICS_H
#define LLVM_IR_INTRINSICS_H

#include "llvm/ADT/ArrayRef.h"
#include <string>

namespace llvm {

class Type;
class FunctionType;
class Function;
class LLVMContext;
class Module;
class AttributeSet;

/// This namespace contains an enum with a value for every intrinsic/builtin
/// function known by LLVM. The enum values are returned by
/// Function::getIntrinsicID().
namespace Intrinsic {
  enum ID : unsigned {
    not_intrinsic = 0,   // Must be zero

    // Get the intrinsic enums generated from Intrinsics.td
#define GET_INTRINSIC_ENUM_VALUES
#include "llvm/IR/Intrinsics.gen"
#undef GET_INTRINSIC_ENUM_VALUES
    , num_intrinsics
  };

  /// Return the LLVM name for an intrinsic, such as "llvm.ppc.altivec.lvx".
  std::string getName(ID id, ArrayRef<Type*> Tys = None);

  /// Return the function type for an intrinsic.
  FunctionType *getType(LLVMContext &Context, ID id,
                        ArrayRef<Type*> Tys = None);

  /// Returns true if the intrinsic can be overloaded.
  bool isOverloaded(ID id);

  /// Returns true if the intrinsic is a leaf, i.e. it does not make any calls
  /// itself.  Most intrinsics are leafs, the exceptions being the patchpoint
  /// and statepoint intrinsics. These call (or invoke) their "target" argument.
  bool isLeaf(ID id);

  /// Return the attributes for an intrinsic.
  AttributeSet getAttributes(LLVMContext &C, ID id);

  /// Create or insert an LLVM Function declaration for an intrinsic, and return
  /// it.
  ///
  /// The Tys parameter is for intrinsics with overloaded types (e.g., those
  /// using iAny, fAny, vAny, or iPTRAny).  For a declaration of an overloaded
  /// intrinsic, Tys must provide exactly one type for each overloaded type in
  /// the intrinsic.
  Function *getDeclaration(Module *M, ID id, ArrayRef<Type*> Tys = None);

  /// Map a GCC builtin name to an intrinsic ID.
  ID getIntrinsicForGCCBuiltin(const char *Prefix, const char *BuiltinName);

  /// Map a MS builtin name to an intrinsic ID.
  ID getIntrinsicForMSBuiltin(const char *Prefix, const char *BuiltinName);

  /// This is a type descriptor which explains the type requirements of an
  /// intrinsic. This is returned by getIntrinsicInfoTableEntries.
  struct IITDescriptor {
    enum IITDescriptorKind {
      Void, VarArg, MMX, Metadata, Half, Float, Double,
      Integer, Vector, Pointer, Struct,
      Argument, ExtendArgument, TruncArgument, HalfVecArgument,
      SameVecWidthArgument, PtrToArgument, VecOfPtrsToElt
    } Kind;

    union {
      unsigned Integer_Width;
      unsigned Float_Width;
      unsigned Vector_Width;
      unsigned Pointer_AddressSpace;
      unsigned Struct_NumElements;
      unsigned Argument_Info;
    };

    enum ArgKind {
      AK_Any,
      AK_AnyInteger,
      AK_AnyFloat,
      AK_AnyVector,
      AK_AnyPointer
    };
    unsigned getArgumentNumber() const {
      assert(Kind == Argument || Kind == ExtendArgument ||
             Kind == TruncArgument || Kind == HalfVecArgument ||
             Kind == SameVecWidthArgument || Kind == PtrToArgument ||
             Kind == VecOfPtrsToElt);
      return Argument_Info >> 3;
    }
    ArgKind getArgumentKind() const {
      assert(Kind == Argument || Kind == ExtendArgument ||
             Kind == TruncArgument || Kind == HalfVecArgument ||
             Kind == SameVecWidthArgument || Kind == PtrToArgument ||
             Kind == VecOfPtrsToElt);
      return (ArgKind)(Argument_Info & 7);
    }

    static IITDescriptor get(IITDescriptorKind K, unsigned Field) {
      IITDescriptor Result = { K, { Field } };
      return Result;
    }
  };

  /// Return the IIT table descriptor for the specified intrinsic into an array
  /// of IITDescriptors.
  void getIntrinsicInfoTableEntries(ID id, SmallVectorImpl<IITDescriptor> &T);

} // End Intrinsic namespace

} // End llvm namespace

#endif

//===- CodeGen/MachineValueType.h - Machine-Level types ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the set of machine-level target independent types which
// legal values in the code generator use.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINEVALUETYPE_H
#define LLVM_CODEGEN_MACHINEVALUETYPE_H

#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"

namespace llvm {

  class Type;

  /// MVT - Machine Value Type. Every type that is supported natively by some
  /// processor targeted by LLVM occurs here. This means that any legal value
  /// type can be represented by an MVT.
class MVT {
  public:
    enum SimpleValueType {
      // INVALID_SIMPLE_VALUE_TYPE - Simple value types less than zero are
      // considered extended value types.
      INVALID_SIMPLE_VALUE_TYPE = -1,

      // If you change this numbering, you must change the values in
      // ValueTypes.td as well!
      Other          =   0,   // This is a non-standard value
      i1             =   1,   // This is a 1 bit integer value
      i8             =   2,   // This is an 8 bit integer value
      i16            =   3,   // This is a 16 bit integer value
      i32            =   4,   // This is a 32 bit integer value
      i64            =   5,   // This is a 64 bit integer value
      i128           =   6,   // This is a 128 bit integer value

      FIRST_INTEGER_VALUETYPE = i1,
      LAST_INTEGER_VALUETYPE  = i128,

      f16            =   7,   // This is a 16 bit floating point value
      f32            =   8,   // This is a 32 bit floating point value
      f64            =   9,   // This is a 64 bit floating point value
      f80            =  10,   // This is a 80 bit floating point value
      f128           =  11,   // This is a 128 bit floating point value
      ppcf128        =  12,   // This is a PPC 128-bit floating point value

      FIRST_FP_VALUETYPE = f16,
      LAST_FP_VALUETYPE  = ppcf128,

      v2i1           =  13,   //  2 x i1
      v4i1           =  14,   //  4 x i1
      v8i1           =  15,   //  8 x i1
      v16i1          =  16,   // 16 x i1
      v32i1          =  17,   // 32 x i1
      v64i1          =  18,   // 64 x i1

      v1i8           =  19,   //  1 x i8
      v2i8           =  20,   //  2 x i8
      v4i8           =  21,   //  4 x i8
      v8i8           =  22,   //  8 x i8
      v16i8          =  23,   // 16 x i8
      v32i8          =  24,   // 32 x i8
      v64i8          =  25,   // 64 x i8
      v1i16          =  26,   //  1 x i16
      v2i16          =  27,   //  2 x i16
      v4i16          =  28,   //  4 x i16
      v8i16          =  29,   //  8 x i16
      v16i16         =  30,   // 16 x i16
      v32i16         =  31,   // 32 x i16
      v1i32          =  32,   //  1 x i32
      v2i32          =  33,   //  2 x i32
      v4i32          =  34,   //  4 x i32
      v8i32          =  35,   //  8 x i32
      v16i32         =  36,   // 16 x i32
      v1i64          =  37,   //  1 x i64
      v2i64          =  38,   //  2 x i64
      v4i64          =  39,   //  4 x i64
      v8i64          =  40,   //  8 x i64
      v16i64         =  41,   // 16 x i64
      v1i128         =  42,   //  1 x i128
      
      FIRST_INTEGER_VECTOR_VALUETYPE = v2i1,
      LAST_INTEGER_VECTOR_VALUETYPE = v1i128,

      v2f16          =  43,   //  2 x f16
      v4f16          =  44,   //  4 x f16
      v8f16          =  45,   //  8 x f16
      v1f32          =  46,   //  1 x f32
      v2f32          =  47,   //  2 x f32
      v4f32          =  48,   //  4 x f32
      v8f32          =  49,   //  8 x f32
      v16f32         =  50,   // 16 x f32
      v1f64          =  51,   //  1 x f64
      v2f64          =  52,   //  2 x f64
      v4f64          =  53,   //  4 x f64
      v8f64          =  54,   //  8 x f64

      FIRST_FP_VECTOR_VALUETYPE = v2f16,
      LAST_FP_VECTOR_VALUETYPE = v8f64,

      FIRST_VECTOR_VALUETYPE = v2i1,
      LAST_VECTOR_VALUETYPE  = v8f64,

      x86mmx         =  55,   // This is an X86 MMX value

      Glue           =  56,   // This glues nodes together during pre-RA sched

      isVoid         =  57,   // This has no value

      Untyped        =  58,   // This value takes a register, but has
                              // unspecified type.  The register class
                              // will be determined by the opcode.

      FIRST_VALUETYPE = 0,    // This is always the beginning of the list.
      LAST_VALUETYPE =  59,   // This always remains at the end of the list.

      // This is the current maximum for LAST_VALUETYPE.
      // MVT::MAX_ALLOWED_VALUETYPE is used for asserts and to size bit vectors
      // This value must be a multiple of 32.
      MAX_ALLOWED_VALUETYPE = 64,

      // Metadata - This is MDNode or MDString.
      Metadata       = 250,

      // iPTRAny - An int value the size of the pointer of the current
      // target to any address space. This must only be used internal to
      // tblgen. Other than for overloading, we treat iPTRAny the same as iPTR.
      iPTRAny        = 251,

      // vAny - A vector with any length and element size. This is used
      // for intrinsics that have overloadings based on vector types.
      // This is only for tblgen's consumption!
      vAny           = 252,

      // fAny - Any floating-point or vector floating-point value. This is used
      // for intrinsics that have overloadings based on floating-point types.
      // This is only for tblgen's consumption!
      fAny           = 253,

      // iAny - An integer or vector integer value of any bit width. This is
      // used for intrinsics that have overloadings based on integer bit widths.
      // This is only for tblgen's consumption!
      iAny           = 254,

      // iPTR - An int value the size of the pointer of the current
      // target.  This should only be used internal to tblgen!
      iPTR           = 255,

      // Any - Any type. This is used for intrinsics that have overloadings.
      // This is only for tblgen's consumption!
      Any            = 256
    };

    SimpleValueType SimpleTy;

    LLVM_CONSTEXPR MVT() : SimpleTy(INVALID_SIMPLE_VALUE_TYPE) {}
    LLVM_CONSTEXPR MVT(SimpleValueType SVT) : SimpleTy(SVT) { }

    bool operator>(const MVT& S)  const { return SimpleTy >  S.SimpleTy; }
    bool operator<(const MVT& S)  const { return SimpleTy <  S.SimpleTy; }
    bool operator==(const MVT& S) const { return SimpleTy == S.SimpleTy; }
    bool operator!=(const MVT& S) const { return SimpleTy != S.SimpleTy; }
    bool operator>=(const MVT& S) const { return SimpleTy >= S.SimpleTy; }
    bool operator<=(const MVT& S) const { return SimpleTy <= S.SimpleTy; }

    /// isValid - Return true if this is a valid simple valuetype.
    bool isValid() const {
      return (SimpleTy >= MVT::FIRST_VALUETYPE &&
              SimpleTy < MVT::LAST_VALUETYPE);
    }

    /// isFloatingPoint - Return true if this is a FP, or a vector FP type.
    bool isFloatingPoint() const {
      return ((SimpleTy >= MVT::FIRST_FP_VALUETYPE &&
               SimpleTy <= MVT::LAST_FP_VALUETYPE) ||
              (SimpleTy >= MVT::FIRST_FP_VECTOR_VALUETYPE &&
               SimpleTy <= MVT::LAST_FP_VECTOR_VALUETYPE));
    }

    /// isInteger - Return true if this is an integer, or a vector integer type.
    bool isInteger() const {
      return ((SimpleTy >= MVT::FIRST_INTEGER_VALUETYPE &&
               SimpleTy <= MVT::LAST_INTEGER_VALUETYPE) ||
              (SimpleTy >= MVT::FIRST_INTEGER_VECTOR_VALUETYPE &&
               SimpleTy <= MVT::LAST_INTEGER_VECTOR_VALUETYPE));
    }

    /// isVector - Return true if this is a vector value type.
    bool isVector() const {
      return (SimpleTy >= MVT::FIRST_VECTOR_VALUETYPE &&
              SimpleTy <= MVT::LAST_VECTOR_VALUETYPE);
    }

    /// is16BitVector - Return true if this is a 16-bit vector type.
    bool is16BitVector() const {
      return (SimpleTy == MVT::v2i8  || SimpleTy == MVT::v1i16 ||
              SimpleTy == MVT::v16i1);
    }

    /// is32BitVector - Return true if this is a 32-bit vector type.
    bool is32BitVector() const {
      return (SimpleTy == MVT::v4i8  || SimpleTy == MVT::v2i16 ||
              SimpleTy == MVT::v1i32 || SimpleTy == MVT::v2f16 ||
              SimpleTy == MVT::v1f32);
    }

    /// is64BitVector - Return true if this is a 64-bit vector type.
    bool is64BitVector() const {
      return (SimpleTy == MVT::v8i8  || SimpleTy == MVT::v4i16 ||
              SimpleTy == MVT::v2i32 || SimpleTy == MVT::v1i64 ||
              SimpleTy == MVT::v4f16 || SimpleTy == MVT::v2f32 ||
              SimpleTy == MVT::v1f64);
    }

    /// is128BitVector - Return true if this is a 128-bit vector type.
    bool is128BitVector() const {
      return (SimpleTy == MVT::v16i8  || SimpleTy == MVT::v8i16 ||
              SimpleTy == MVT::v4i32  || SimpleTy == MVT::v2i64 ||
              SimpleTy == MVT::v1i128 || SimpleTy == MVT::v8f16 ||
              SimpleTy == MVT::v4f32  || SimpleTy == MVT::v2f64);
    }

    /// is256BitVector - Return true if this is a 256-bit vector type.
    bool is256BitVector() const {
      return (SimpleTy == MVT::v8f32 || SimpleTy == MVT::v4f64  ||
              SimpleTy == MVT::v32i8 || SimpleTy == MVT::v16i16 ||
              SimpleTy == MVT::v8i32 || SimpleTy == MVT::v4i64);
    }

    /// is512BitVector - Return true if this is a 512-bit vector type.
    bool is512BitVector() const {
      return (SimpleTy == MVT::v8f64 || SimpleTy == MVT::v16f32 ||
              SimpleTy == MVT::v64i8 || SimpleTy == MVT::v32i16 ||
              SimpleTy == MVT::v8i64 || SimpleTy == MVT::v16i32);
    }

    /// is1024BitVector - Return true if this is a 1024-bit vector type.
    bool is1024BitVector() const {
      return (SimpleTy == MVT::v16i64);
    }

    /// isOverloaded - Return true if this is an overloaded type for TableGen.
    bool isOverloaded() const {
      return (SimpleTy==MVT::Any  ||
              SimpleTy==MVT::iAny || SimpleTy==MVT::fAny ||
              SimpleTy==MVT::vAny || SimpleTy==MVT::iPTRAny);
    }

    /// isPow2VectorType - Returns true if the given vector is a power of 2.
    bool isPow2VectorType() const {
      unsigned NElts = getVectorNumElements();
      return !(NElts & (NElts - 1));
    }

    /// getPow2VectorType - Widens the length of the given vector MVT up to
    /// the nearest power of 2 and returns that type.
    MVT getPow2VectorType() const {
      if (isPow2VectorType())
        return *this;

      unsigned NElts = getVectorNumElements();
      unsigned Pow2NElts = 1 << Log2_32_Ceil(NElts);
      return MVT::getVectorVT(getVectorElementType(), Pow2NElts);
    }

    /// getScalarType - If this is a vector type, return the element type,
    /// otherwise return this.
    MVT getScalarType() const {
      return isVector() ? getVectorElementType() : *this;
    }

    MVT getVectorElementType() const {
      switch (SimpleTy) {
      default:
        llvm_unreachable("Not a vector MVT!");
      case v2i1 :
      case v4i1 :
      case v8i1 :
      case v16i1 :
      case v32i1 :
      case v64i1: return i1;
      case v1i8 :
      case v2i8 :
      case v4i8 :
      case v8i8 :
      case v16i8:
      case v32i8:
      case v64i8: return i8;
      case v1i16:
      case v2i16:
      case v4i16:
      case v8i16:
      case v16i16:
      case v32i16: return i16;
      case v1i32:
      case v2i32:
      case v4i32:
      case v8i32:
      case v16i32: return i32;
      case v1i64:
      case v2i64:
      case v4i64:
      case v8i64:
      case v16i64: return i64;
      case v1i128: return i128;
      case v2f16:
      case v4f16:
      case v8f16: return f16;
      case v1f32:
      case v2f32:
      case v4f32:
      case v8f32:
      case v16f32: return f32;
      case v1f64:
      case v2f64:
      case v4f64:
      case v8f64: return f64;
      }
    }

    unsigned getVectorNumElements() const {
      switch (SimpleTy) {
      default:
        llvm_unreachable("Not a vector MVT!");
      case v32i1:
      case v32i8:
      case v32i16: return 32;
      case v64i1:
      case v64i8: return 64;
      case v16i1:
      case v16i8:
      case v16i16:
      case v16i32:
      case v16i64:
      case v16f32: return 16;
      case v8i1 :
      case v8i8 :
      case v8i16:
      case v8i32:
      case v8i64:
      case v8f16:
      case v8f32:
      case v8f64: return 8;
      case v4i1:
      case v4i8:
      case v4i16:
      case v4i32:
      case v4i64:
      case v4f16:
      case v4f32:
      case v4f64: return 4;
      case v2i1:
      case v2i8:
      case v2i16:
      case v2i32:
      case v2i64:
      case v2f16:
      case v2f32:
      case v2f64: return 2;
      case v1i8:
      case v1i16:
      case v1i32:
      case v1i64:
      case v1i128:
      case v1f32:
      case v1f64: return 1;
      }
    }

    unsigned getSizeInBits() const {
      switch (SimpleTy) {
      default:
        llvm_unreachable("getSizeInBits called on extended MVT.");
      case Other:
        llvm_unreachable("Value type is non-standard value, Other.");
      case iPTR:
        llvm_unreachable("Value type size is target-dependent. Ask TLI.");
      case iPTRAny:
      case iAny:
      case fAny:
      case vAny:
      case Any:
        llvm_unreachable("Value type is overloaded.");
      case Metadata:
        llvm_unreachable("Value type is metadata.");
      case i1  :  return 1;
      case v2i1:  return 2;
      case v4i1:  return 4;
      case i8  :
      case v1i8:
      case v8i1: return 8;
      case i16 :
      case f16:
      case v16i1:
      case v2i8:
      case v1i16: return 16;
      case f32 :
      case i32 :
      case v32i1:
      case v4i8:
      case v2i16:
      case v2f16:
      case v1f32:
      case v1i32: return 32;
      case x86mmx:
      case f64 :
      case i64 :
      case v64i1:
      case v8i8:
      case v4i16:
      case v2i32:
      case v1i64:
      case v4f16:
      case v2f32:
      case v1f64: return 64;
      case f80 :  return 80;
      case f128:
      case ppcf128:
      case i128:
      case v16i8:
      case v8i16:
      case v4i32:
      case v2i64:
      case v1i128:
      case v8f16:
      case v4f32:
      case v2f64: return 128;
      case v32i8:
      case v16i16:
      case v8i32:
      case v4i64:
      case v8f32:
      case v4f64: return 256;
      case v64i8:
      case v32i16:
      case v16i32:
      case v8i64:
      case v16f32:
      case v8f64: return 512;
      case v16i64:return 1024;
      }
    }

    unsigned getScalarSizeInBits() const {
      return getScalarType().getSizeInBits();
    }

    /// getStoreSize - Return the number of bytes overwritten by a store
    /// of the specified value type.
    unsigned getStoreSize() const {
      return (getSizeInBits() + 7) / 8;
    }

    /// getStoreSizeInBits - Return the number of bits overwritten by a store
    /// of the specified value type.
    unsigned getStoreSizeInBits() const {
      return getStoreSize() * 8;
    }

    /// Return true if this has more bits than VT.
    bool bitsGT(MVT VT) const {
      return getSizeInBits() > VT.getSizeInBits();
    }

    /// Return true if this has no less bits than VT.
    bool bitsGE(MVT VT) const {
      return getSizeInBits() >= VT.getSizeInBits();
    }

    /// Return true if this has less bits than VT.
    bool bitsLT(MVT VT) const {
      return getSizeInBits() < VT.getSizeInBits();
    }

    /// Return true if this has no more bits than VT.
    bool bitsLE(MVT VT) const {
      return getSizeInBits() <= VT.getSizeInBits();
    }


    static MVT getFloatingPointVT(unsigned BitWidth) {
      switch (BitWidth) {
      default:
        llvm_unreachable("Bad bit width!");
      case 16:
        return MVT::f16;
      case 32:
        return MVT::f32;
      case 64:
        return MVT::f64;
      case 80:
        return MVT::f80;
      case 128:
        return MVT::f128;
      }
    }

    static MVT getIntegerVT(unsigned BitWidth) {
      switch (BitWidth) {
      default:
        return (MVT::SimpleValueType)(MVT::INVALID_SIMPLE_VALUE_TYPE);
      case 1:
        return MVT::i1;
      case 8:
        return MVT::i8;
      case 16:
        return MVT::i16;
      case 32:
        return MVT::i32;
      case 64:
        return MVT::i64;
      case 128:
        return MVT::i128;
      }
    }

    static MVT getVectorVT(MVT VT, unsigned NumElements) {
      switch (VT.SimpleTy) {
      default:
        break;
      case MVT::i1:
        if (NumElements == 2)  return MVT::v2i1;
        if (NumElements == 4)  return MVT::v4i1;
        if (NumElements == 8)  return MVT::v8i1;
        if (NumElements == 16) return MVT::v16i1;
        if (NumElements == 32) return MVT::v32i1;
        if (NumElements == 64) return MVT::v64i1;
        break;
      case MVT::i8:
        if (NumElements == 1)  return MVT::v1i8;
        if (NumElements == 2)  return MVT::v2i8;
        if (NumElements == 4)  return MVT::v4i8;
        if (NumElements == 8)  return MVT::v8i8;
        if (NumElements == 16) return MVT::v16i8;
        if (NumElements == 32) return MVT::v32i8;
        if (NumElements == 64) return MVT::v64i8;
        break;
      case MVT::i16:
        if (NumElements == 1)  return MVT::v1i16;
        if (NumElements == 2)  return MVT::v2i16;
        if (NumElements == 4)  return MVT::v4i16;
        if (NumElements == 8)  return MVT::v8i16;
        if (NumElements == 16) return MVT::v16i16;
        if (NumElements == 32) return MVT::v32i16;
        break;
      case MVT::i32:
        if (NumElements == 1)  return MVT::v1i32;
        if (NumElements == 2)  return MVT::v2i32;
        if (NumElements == 4)  return MVT::v4i32;
        if (NumElements == 8)  return MVT::v8i32;
        if (NumElements == 16) return MVT::v16i32;
        break;
      case MVT::i64:
        if (NumElements == 1)  return MVT::v1i64;
        if (NumElements == 2)  return MVT::v2i64;
        if (NumElements == 4)  return MVT::v4i64;
        if (NumElements == 8)  return MVT::v8i64;
        if (NumElements == 16) return MVT::v16i64;
        break;
      case MVT::i128:
        if (NumElements == 1)  return MVT::v1i128;
        break;
      case MVT::f16:
        if (NumElements == 2)  return MVT::v2f16;
        if (NumElements == 4)  return MVT::v4f16;
        if (NumElements == 8)  return MVT::v8f16;
        break;
      case MVT::f32:
        if (NumElements == 1)  return MVT::v1f32;
        if (NumElements == 2)  return MVT::v2f32;
        if (NumElements == 4)  return MVT::v4f32;
        if (NumElements == 8)  return MVT::v8f32;
        if (NumElements == 16) return MVT::v16f32;
        break;
      case MVT::f64:
        if (NumElements == 1)  return MVT::v1f64;
        if (NumElements == 2)  return MVT::v2f64;
        if (NumElements == 4)  return MVT::v4f64;
        if (NumElements == 8)  return MVT::v8f64;
        break;
      }
      return (MVT::SimpleValueType)(MVT::INVALID_SIMPLE_VALUE_TYPE);
    }

    /// Return the value type corresponding to the specified type.  This returns
    /// all pointers as iPTR.  If HandleUnknown is true, unknown types are
    /// returned as Other, otherwise they are invalid.
    static MVT getVT(Type *Ty, bool HandleUnknown = false);

  private:
    /// A simple iterator over the MVT::SimpleValueType enum.
    struct mvt_iterator {
      SimpleValueType VT;
      mvt_iterator(SimpleValueType VT) : VT(VT) {}
      MVT operator*() const { return VT; }
      bool operator!=(const mvt_iterator &LHS) const { return VT != LHS.VT; }
      mvt_iterator& operator++() {
        VT = (MVT::SimpleValueType)((int)VT + 1);
        assert((int)VT <= MVT::MAX_ALLOWED_VALUETYPE &&
               "MVT iterator overflowed.");
        return *this;
      }
    };
    /// A range of the MVT::SimpleValueType enum.
    typedef iterator_range<mvt_iterator> mvt_range;

  public:
    /// SimpleValueType Iteration
    /// @{
    static mvt_range all_valuetypes() {
      return mvt_range(MVT::FIRST_VALUETYPE, MVT::LAST_VALUETYPE);
    }
    static mvt_range integer_valuetypes() {
      return mvt_range(MVT::FIRST_INTEGER_VALUETYPE,
                       (MVT::SimpleValueType)(MVT::LAST_INTEGER_VALUETYPE + 1));
    }
    static mvt_range fp_valuetypes() {
      return mvt_range(MVT::FIRST_FP_VALUETYPE,
                       (MVT::SimpleValueType)(MVT::LAST_FP_VALUETYPE + 1));
    }
    static mvt_range vector_valuetypes() {
      return mvt_range(MVT::FIRST_VECTOR_VALUETYPE,
                       (MVT::SimpleValueType)(MVT::LAST_VECTOR_VALUETYPE + 1));
    }
    static mvt_range integer_vector_valuetypes() {
      return mvt_range(
          MVT::FIRST_INTEGER_VECTOR_VALUETYPE,
          (MVT::SimpleValueType)(MVT::LAST_INTEGER_VECTOR_VALUETYPE + 1));
    }
    static mvt_range fp_vector_valuetypes() {
      return mvt_range(
          MVT::FIRST_FP_VECTOR_VALUETYPE,
          (MVT::SimpleValueType)(MVT::LAST_FP_VECTOR_VALUETYPE + 1));
    }
    /// @}
  };

} // End llvm namespace

#endif

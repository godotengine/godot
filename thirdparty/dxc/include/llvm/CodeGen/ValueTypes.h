//===- CodeGen/ValueTypes.h - Low-Level Target independ. types --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the set of low-level target independent types which various
// values in the code generator are.  This allows the target specific behavior
// of instructions to be described to target independent passes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_VALUETYPES_H
#define LLVM_CODEGEN_VALUETYPES_H

#include "llvm/CodeGen/MachineValueType.h"
#include <cassert>
#include <string>

namespace llvm {

  class LLVMContext;
  class Type;

  /// EVT - Extended Value Type.  Capable of holding value types which are not
  /// native for any processor (such as the i12345 type), as well as the types
  /// a MVT can represent.
  struct EVT {
  private:
    MVT V;
    Type *LLVMTy;

  public:
    LLVM_CONSTEXPR EVT() : V(MVT::INVALID_SIMPLE_VALUE_TYPE), LLVMTy(nullptr) {}
    LLVM_CONSTEXPR EVT(MVT::SimpleValueType SVT) : V(SVT), LLVMTy(nullptr) {}
    LLVM_CONSTEXPR EVT(MVT S) : V(S), LLVMTy(nullptr) {}

    bool operator==(EVT VT) const {
      return !(*this != VT);
    }
    bool operator!=(EVT VT) const {
      if (V.SimpleTy != VT.V.SimpleTy)
        return true;
      if (V.SimpleTy < 0)
        return LLVMTy != VT.LLVMTy;
      return false;
    }

    /// getFloatingPointVT - Returns the EVT that represents a floating point
    /// type with the given number of bits.  There are two floating point types
    /// with 128 bits - this returns f128 rather than ppcf128.
    static EVT getFloatingPointVT(unsigned BitWidth) {
      return MVT::getFloatingPointVT(BitWidth);
    }

    /// getIntegerVT - Returns the EVT that represents an integer with the given
    /// number of bits.
    static EVT getIntegerVT(LLVMContext &Context, unsigned BitWidth) {
      MVT M = MVT::getIntegerVT(BitWidth);
      if (M.SimpleTy >= 0)
        return M;
      return getExtendedIntegerVT(Context, BitWidth);
    }

    /// getVectorVT - Returns the EVT that represents a vector NumElements in
    /// length, where each element is of type VT.
    static EVT getVectorVT(LLVMContext &Context, EVT VT, unsigned NumElements) {
      MVT M = MVT::getVectorVT(VT.V, NumElements);
      if (M.SimpleTy >= 0)
        return M;
      return getExtendedVectorVT(Context, VT, NumElements);
    }

    /// changeVectorElementTypeToInteger - Return a vector with the same number
    /// of elements as this vector, but with the element type converted to an
    /// integer type with the same bitwidth.
    EVT changeVectorElementTypeToInteger() const {
      if (!isSimple())
        return changeExtendedVectorElementTypeToInteger();
      MVT EltTy = getSimpleVT().getVectorElementType();
      unsigned BitWidth = EltTy.getSizeInBits();
      MVT IntTy = MVT::getIntegerVT(BitWidth);
      MVT VecTy = MVT::getVectorVT(IntTy, getVectorNumElements());
      assert(VecTy.SimpleTy >= 0 &&
             "Simple vector VT not representable by simple integer vector VT!");
      return VecTy;
    }

    /// isSimple - Test if the given EVT is simple (as opposed to being
    /// extended).
    bool isSimple() const {
      return V.SimpleTy >= 0;
    }

    /// isExtended - Test if the given EVT is extended (as opposed to
    /// being simple).
    bool isExtended() const {
      return !isSimple();
    }

    /// isFloatingPoint - Return true if this is a FP, or a vector FP type.
    bool isFloatingPoint() const {
      return isSimple() ? V.isFloatingPoint() : isExtendedFloatingPoint();
    }

    /// isInteger - Return true if this is an integer, or a vector integer type.
    bool isInteger() const {
      return isSimple() ? V.isInteger() : isExtendedInteger();
    }

    /// isVector - Return true if this is a vector value type.
    bool isVector() const {
      return isSimple() ? V.isVector() : isExtendedVector();
    }

    /// is16BitVector - Return true if this is a 16-bit vector type.
    bool is16BitVector() const {
      return isSimple() ? V.is16BitVector() : isExtended16BitVector();
    }

    /// is32BitVector - Return true if this is a 32-bit vector type.
    bool is32BitVector() const {
      return isSimple() ? V.is32BitVector() : isExtended32BitVector();
    }

    /// is64BitVector - Return true if this is a 64-bit vector type.
    bool is64BitVector() const {
      return isSimple() ? V.is64BitVector() : isExtended64BitVector();
    }

    /// is128BitVector - Return true if this is a 128-bit vector type.
    bool is128BitVector() const {
      return isSimple() ? V.is128BitVector() : isExtended128BitVector();
    }

    /// is256BitVector - Return true if this is a 256-bit vector type.
    bool is256BitVector() const {
      return isSimple() ? V.is256BitVector() : isExtended256BitVector();
    }

    /// is512BitVector - Return true if this is a 512-bit vector type.
    bool is512BitVector() const {
      return isSimple() ? V.is512BitVector() : isExtended512BitVector();
    }

    /// is1024BitVector - Return true if this is a 1024-bit vector type.
    bool is1024BitVector() const {
      return isSimple() ? V.is1024BitVector() : isExtended1024BitVector();
    }

    /// isOverloaded - Return true if this is an overloaded type for TableGen.
    bool isOverloaded() const {
      return (V==MVT::iAny || V==MVT::fAny || V==MVT::vAny || V==MVT::iPTRAny);
    }

    /// isByteSized - Return true if the bit size is a multiple of 8.
    bool isByteSized() const {
      return (getSizeInBits() & 7) == 0;
    }

    /// isRound - Return true if the size is a power-of-two number of bytes.
    bool isRound() const {
      unsigned BitSize = getSizeInBits();
      return BitSize >= 8 && !(BitSize & (BitSize - 1));
    }

    /// bitsEq - Return true if this has the same number of bits as VT.
    bool bitsEq(EVT VT) const {
      if (EVT::operator==(VT)) return true;
      return getSizeInBits() == VT.getSizeInBits();
    }

    /// bitsGT - Return true if this has more bits than VT.
    bool bitsGT(EVT VT) const {
      if (EVT::operator==(VT)) return false;
      return getSizeInBits() > VT.getSizeInBits();
    }

    /// bitsGE - Return true if this has no less bits than VT.
    bool bitsGE(EVT VT) const {
      if (EVT::operator==(VT)) return true;
      return getSizeInBits() >= VT.getSizeInBits();
    }

    /// bitsLT - Return true if this has less bits than VT.
    bool bitsLT(EVT VT) const {
      if (EVT::operator==(VT)) return false;
      return getSizeInBits() < VT.getSizeInBits();
    }

    /// bitsLE - Return true if this has no more bits than VT.
    bool bitsLE(EVT VT) const {
      if (EVT::operator==(VT)) return true;
      return getSizeInBits() <= VT.getSizeInBits();
    }


    /// getSimpleVT - Return the SimpleValueType held in the specified
    /// simple EVT.
    MVT getSimpleVT() const {
      assert(isSimple() && "Expected a SimpleValueType!");
      return V;
    }

    /// getScalarType - If this is a vector type, return the element type,
    /// otherwise return this.
    EVT getScalarType() const {
      return isVector() ? getVectorElementType() : *this;
    }

    /// getVectorElementType - Given a vector type, return the type of
    /// each element.
    EVT getVectorElementType() const {
      assert(isVector() && "Invalid vector type!");
      if (isSimple())
        return V.getVectorElementType();
      return getExtendedVectorElementType();
    }

    /// getVectorNumElements - Given a vector type, return the number of
    /// elements it contains.
    unsigned getVectorNumElements() const {
      assert(isVector() && "Invalid vector type!");
      if (isSimple())
        return V.getVectorNumElements();
      return getExtendedVectorNumElements();
    }

    /// getSizeInBits - Return the size of the specified value type in bits.
    unsigned getSizeInBits() const {
      if (isSimple())
        return V.getSizeInBits();
      return getExtendedSizeInBits();
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

    /// getRoundIntegerType - Rounds the bit-width of the given integer EVT up
    /// to the nearest power of two (and at least to eight), and returns the
    /// integer EVT with that number of bits.
    EVT getRoundIntegerType(LLVMContext &Context) const {
      assert(isInteger() && !isVector() && "Invalid integer type!");
      unsigned BitWidth = getSizeInBits();
      if (BitWidth <= 8)
        return EVT(MVT::i8);
      return getIntegerVT(Context, 1 << Log2_32_Ceil(BitWidth));
    }

    /// getHalfSizedIntegerVT - Finds the smallest simple value type that is
    /// greater than or equal to half the width of this EVT. If no simple
    /// value type can be found, an extended integer value type of half the
    /// size (rounded up) is returned.
    EVT getHalfSizedIntegerVT(LLVMContext &Context) const {
      assert(isInteger() && !isVector() && "Invalid integer type!");
      unsigned EVTSize = getSizeInBits();
      for (unsigned IntVT = MVT::FIRST_INTEGER_VALUETYPE;
          IntVT <= MVT::LAST_INTEGER_VALUETYPE; ++IntVT) {
        EVT HalfVT = EVT((MVT::SimpleValueType)IntVT);
        if (HalfVT.getSizeInBits() * 2 >= EVTSize)
          return HalfVT;
      }
      return getIntegerVT(Context, (EVTSize + 1) / 2);
    }

    /// \brief Return a VT for an integer vector type with the size of the
    /// elements doubled. The typed returned may be an extended type.
    EVT widenIntegerVectorElementType(LLVMContext &Context) const {
      EVT EltVT = getVectorElementType();
      EltVT = EVT::getIntegerVT(Context, 2 * EltVT.getSizeInBits());
      return EVT::getVectorVT(Context, EltVT, getVectorNumElements());
    }

    /// isPow2VectorType - Returns true if the given vector is a power of 2.
    bool isPow2VectorType() const {
      unsigned NElts = getVectorNumElements();
      return !(NElts & (NElts - 1));
    }

    /// getPow2VectorType - Widens the length of the given vector EVT up to
    /// the nearest power of 2 and returns that type.
    EVT getPow2VectorType(LLVMContext &Context) const {
      if (!isPow2VectorType()) {
        unsigned NElts = getVectorNumElements();
        unsigned Pow2NElts = 1 <<  Log2_32_Ceil(NElts);
        return EVT::getVectorVT(Context, getVectorElementType(), Pow2NElts);
      }
      else {
        return *this;
      }
    }

    /// getEVTString - This function returns value type as a string,
    /// e.g. "i32".
    std::string getEVTString() const;

    /// getTypeForEVT - This method returns an LLVM type corresponding to the
    /// specified EVT.  For integer types, this returns an unsigned type.  Note
    /// that this will abort for types that cannot be represented.
    Type *getTypeForEVT(LLVMContext &Context) const;

    /// getEVT - Return the value type corresponding to the specified type.
    /// This returns all pointers as iPTR.  If HandleUnknown is true, unknown
    /// types are returned as Other, otherwise they are invalid.
    static EVT getEVT(Type *Ty, bool HandleUnknown = false);

    intptr_t getRawBits() const {
      if (isSimple())
        return V.SimpleTy;
      else
        return (intptr_t)(LLVMTy);
    }

    /// compareRawBits - A meaningless but well-behaved order, useful for
    /// constructing containers.
    struct compareRawBits {
      bool operator()(EVT L, EVT R) const {
        if (L.V.SimpleTy == R.V.SimpleTy)
          return L.LLVMTy < R.LLVMTy;
        else
          return L.V.SimpleTy < R.V.SimpleTy;
      }
    };

  private:
    // Methods for handling the Extended-type case in functions above.
    // These are all out-of-line to prevent users of this header file
    // from having a dependency on Type.h.
    EVT changeExtendedVectorElementTypeToInteger() const;
    static EVT getExtendedIntegerVT(LLVMContext &C, unsigned BitWidth);
    static EVT getExtendedVectorVT(LLVMContext &C, EVT VT,
                                   unsigned NumElements);
    bool isExtendedFloatingPoint() const LLVM_READONLY;
    bool isExtendedInteger() const LLVM_READONLY;
    bool isExtendedVector() const LLVM_READONLY;
    bool isExtended16BitVector() const LLVM_READONLY;
    bool isExtended32BitVector() const LLVM_READONLY;
    bool isExtended64BitVector() const LLVM_READONLY;
    bool isExtended128BitVector() const LLVM_READONLY;
    bool isExtended256BitVector() const LLVM_READONLY;
    bool isExtended512BitVector() const LLVM_READONLY;
    bool isExtended1024BitVector() const LLVM_READONLY;
    EVT getExtendedVectorElementType() const;
    unsigned getExtendedVectorNumElements() const LLVM_READONLY;
    unsigned getExtendedSizeInBits() const;
  };

} // End llvm namespace

#endif

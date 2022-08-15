//===-- llvm/Operator.h - Operator utility subclass -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines various classes for working with Instructions and
// ConstantExprs.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_OPERATOR_H
#define LLVM_IR_OPERATOR_H

#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Type.h"

namespace llvm {

class GetElementPtrInst;
class BinaryOperator;
class ConstantExpr;

/// This is a utility class that provides an abstraction for the common
/// functionality between Instructions and ConstantExprs.
class Operator : public User {
private:
  // The Operator class is intended to be used as a utility, and is never itself
  // instantiated.
  void *operator new(size_t, unsigned) = delete;
  void *operator new(size_t s) = delete;
  Operator() = delete;

protected:
  // NOTE: Cannot use = delete because it's not legal to delete
  // an overridden method that's not deleted in the base class. Cannot leave
  // this unimplemented because that leads to an ODR-violation.
  ~Operator() override;

public:
  /// Return the opcode for this Instruction or ConstantExpr.
  unsigned getOpcode() const {
    if (const Instruction *I = dyn_cast<Instruction>(this))
      return I->getOpcode();
    return cast<ConstantExpr>(this)->getOpcode();
  }

  /// If V is an Instruction or ConstantExpr, return its opcode.
  /// Otherwise return UserOp1.
  static unsigned getOpcode(const Value *V) {
    if (const Instruction *I = dyn_cast<Instruction>(V))
      return I->getOpcode();
    if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(V))
      return CE->getOpcode();
    return Instruction::UserOp1;
  }

  static inline bool classof(const Instruction *) { return true; }
  static inline bool classof(const ConstantExpr *) { return true; }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) || isa<ConstantExpr>(V);
  }
};

/// Utility class for integer arithmetic operators which may exhibit overflow -
/// Add, Sub, and Mul. It does not include SDiv, despite that operator having
/// the potential for overflow.
class OverflowingBinaryOperator : public Operator {
public:
  enum {
    NoUnsignedWrap = (1 << 0),
    NoSignedWrap   = (1 << 1)
  };

private:
  friend class BinaryOperator;
  friend class ConstantExpr;
  void setHasNoUnsignedWrap(bool B) {
    SubclassOptionalData =
      (SubclassOptionalData & ~NoUnsignedWrap) | (B ? NoUnsignedWrap : 0); // HLSL Change - fix bool arithmetic operator
  }
  void setHasNoSignedWrap(bool B) {
    SubclassOptionalData =
      (SubclassOptionalData & ~NoSignedWrap) | (B ? NoSignedWrap : 0); // HLSL Change - fix bool arithmetic operator
  }

public:
  /// Test whether this operation is known to never
  /// undergo unsigned overflow, aka the nuw property.
  bool hasNoUnsignedWrap() const {
    return SubclassOptionalData & NoUnsignedWrap;
  }

  /// Test whether this operation is known to never
  /// undergo signed overflow, aka the nsw property.
  bool hasNoSignedWrap() const {
    return (SubclassOptionalData & NoSignedWrap) != 0;
  }

  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::Add ||
           I->getOpcode() == Instruction::Sub ||
           I->getOpcode() == Instruction::Mul ||
           I->getOpcode() == Instruction::Shl;
  }
  static inline bool classof(const ConstantExpr *CE) {
    return CE->getOpcode() == Instruction::Add ||
           CE->getOpcode() == Instruction::Sub ||
           CE->getOpcode() == Instruction::Mul ||
           CE->getOpcode() == Instruction::Shl;
  }
  static inline bool classof(const Value *V) {
    return (isa<Instruction>(V) && classof(cast<Instruction>(V))) ||
           (isa<ConstantExpr>(V) && classof(cast<ConstantExpr>(V)));
  }
};

/// A udiv or sdiv instruction, which can be marked as "exact",
/// indicating that no bits are destroyed.
class PossiblyExactOperator : public Operator {
public:
  enum {
    IsExact = (1 << 0)
  };

private:
  friend class BinaryOperator;
  friend class ConstantExpr;
  void setIsExact(bool B) {
    SubclassOptionalData = (SubclassOptionalData & ~IsExact) | (B ? IsExact : 0); // HLSL Change - fix bool arithmetic operator
  }

public:
  /// Test whether this division is known to be exact, with zero remainder.
  bool isExact() const {
    return SubclassOptionalData & IsExact;
  }

  static bool isPossiblyExactOpcode(unsigned OpC) {
    return OpC == Instruction::SDiv ||
           OpC == Instruction::UDiv ||
           OpC == Instruction::AShr ||
           OpC == Instruction::LShr;
  }
  static inline bool classof(const ConstantExpr *CE) {
    return isPossiblyExactOpcode(CE->getOpcode());
  }
  static inline bool classof(const Instruction *I) {
    return isPossiblyExactOpcode(I->getOpcode());
  }
  static inline bool classof(const Value *V) {
    return (isa<Instruction>(V) && classof(cast<Instruction>(V))) ||
           (isa<ConstantExpr>(V) && classof(cast<ConstantExpr>(V)));
  }
};

/// Convenience struct for specifying and reasoning about fast-math flags.
class FastMathFlags {
private:
  friend class FPMathOperator;
  unsigned Flags;
  FastMathFlags(unsigned F) : Flags(F) { }

public:
  enum {
    UnsafeAlgebra   = (1 << 0),
    NoNaNs          = (1 << 1),
    NoInfs          = (1 << 2),
    NoSignedZeros   = (1 << 3),
    AllowReciprocal = (1 << 4)
  };

  FastMathFlags() : Flags(0)
  { }

  /// Whether any flag is set
  bool any() const { return Flags != 0; }

  /// Set all the flags to false
  void clear() { Flags = 0; }

  /// Flag queries
  bool noNaNs() const          { return 0 != (Flags & NoNaNs); }
  bool noInfs() const          { return 0 != (Flags & NoInfs); }
  bool noSignedZeros() const   { return 0 != (Flags & NoSignedZeros); }
  bool allowReciprocal() const { return 0 != (Flags & AllowReciprocal); }
  bool unsafeAlgebra() const   { return 0 != (Flags & UnsafeAlgebra); }

  /// Flag setters
  void setNoNaNs()          { Flags |= NoNaNs; }
  void setNoInfs()          { Flags |= NoInfs; }
  void setNoSignedZeros()   { Flags |= NoSignedZeros; }
  void setAllowReciprocal() { Flags |= AllowReciprocal; }
  void setUnsafeAlgebra() {
    Flags |= UnsafeAlgebra;
    setNoNaNs();
    setNoInfs();
    setNoSignedZeros();
    setAllowReciprocal();
  }
  // HLSL Change Begins.
  void setUnsafeAlgebraHLSL() {
    Flags |= UnsafeAlgebra;
    // HLSL has NaNs.
    setNoInfs();
    setNoSignedZeros();
    setAllowReciprocal();
  }
  // HLSL Change Ends.

  void operator&=(const FastMathFlags &OtherFlags) {
    Flags &= OtherFlags.Flags;
  }
};


/// Utility class for floating point operations which can have
/// information about relaxed accuracy requirements attached to them.
class FPMathOperator : public Operator {
private:
  friend class Instruction;

  void setHasUnsafeAlgebra(bool B) {
    SubclassOptionalData =
      (SubclassOptionalData & ~FastMathFlags::UnsafeAlgebra) |
      (B ? FastMathFlags::UnsafeAlgebra : 0); // HLSL Change - fix bool arithmetic operator

    // Unsafe algebra implies all the others
    if (B) {
      setHasNoNaNs(true);
      setHasNoInfs(true);
      setHasNoSignedZeros(true);
      setHasAllowReciprocal(true);
    }
  }
  void setHasNoNaNs(bool B) {
    SubclassOptionalData =
      (SubclassOptionalData & ~FastMathFlags::NoNaNs) |
      (B ? FastMathFlags::NoNaNs : 0); // HLSL Change - fix bool arithmetic operator
  }
  void setHasNoInfs(bool B) {
    SubclassOptionalData =
      (SubclassOptionalData & ~FastMathFlags::NoInfs) |
      (B ? FastMathFlags::NoInfs : 0); // HLSL Change - fix bool arithmetic operator
  }
  void setHasNoSignedZeros(bool B) {
    SubclassOptionalData =
      (SubclassOptionalData & ~FastMathFlags::NoSignedZeros) |
      (B ? FastMathFlags::NoSignedZeros : 0); // HLSL Change - fix bool arithmetic operator
  }
  void setHasAllowReciprocal(bool B) {
    SubclassOptionalData =
      (SubclassOptionalData & ~FastMathFlags::AllowReciprocal) |
      (B ? FastMathFlags::AllowReciprocal : 0); // HLSL Change - fix bool arithmetic operator
  }

  /// Convenience function for setting multiple fast-math flags.
  /// FMF is a mask of the bits to set.
  void setFastMathFlags(FastMathFlags FMF) {
    SubclassOptionalData |= FMF.Flags;
  }

  /// Convenience function for copying all fast-math flags.
  /// All values in FMF are transferred to this operator.
  void copyFastMathFlags(FastMathFlags FMF) {
    SubclassOptionalData = FMF.Flags;
  }

public:
  /// Test whether this operation is permitted to be
  /// algebraically transformed, aka the 'A' fast-math property.
  bool hasUnsafeAlgebra() const {
    return (SubclassOptionalData & FastMathFlags::UnsafeAlgebra) != 0;
  }

  /// Test whether this operation's arguments and results are to be
  /// treated as non-NaN, aka the 'N' fast-math property.
  bool hasNoNaNs() const {
    return (SubclassOptionalData & FastMathFlags::NoNaNs) != 0;
  }

  /// Test whether this operation's arguments and results are to be
  /// treated as NoN-Inf, aka the 'I' fast-math property.
  bool hasNoInfs() const {
    return (SubclassOptionalData & FastMathFlags::NoInfs) != 0;
  }

  /// Test whether this operation can treat the sign of zero
  /// as insignificant, aka the 'S' fast-math property.
  bool hasNoSignedZeros() const {
    return (SubclassOptionalData & FastMathFlags::NoSignedZeros) != 0;
  }

  /// Test whether this operation is permitted to use
  /// reciprocal instead of division, aka the 'R' fast-math property.
  bool hasAllowReciprocal() const {
    return (SubclassOptionalData & FastMathFlags::AllowReciprocal) != 0;
  }

  /// Convenience function for getting all the fast-math flags
  FastMathFlags getFastMathFlags() const {
    return FastMathFlags(SubclassOptionalData);
  }

  /// \brief Get the maximum error permitted by this operation in ULPs.  An
  /// accuracy of 0.0 means that the operation should be performed with the
  /// default precision.
  float getFPAccuracy() const;

  static inline bool classof(const Instruction *I) {
    return I->getType()->isFPOrFPVectorTy() ||
      I->getOpcode() == Instruction::FCmp;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};


/// A helper template for defining operators for individual opcodes.
template<typename SuperClass, unsigned Opc>
class ConcreteOperator : public SuperClass {
public:
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Opc;
  }
  static inline bool classof(const ConstantExpr *CE) {
    return CE->getOpcode() == Opc;
  }
  static inline bool classof(const Value *V) {
    return (isa<Instruction>(V) && classof(cast<Instruction>(V))) ||
           (isa<ConstantExpr>(V) && classof(cast<ConstantExpr>(V)));
  }
};

class AddOperator
  : public ConcreteOperator<OverflowingBinaryOperator, Instruction::Add> {
};
class SubOperator
  : public ConcreteOperator<OverflowingBinaryOperator, Instruction::Sub> {
};
class MulOperator
  : public ConcreteOperator<OverflowingBinaryOperator, Instruction::Mul> {
};
class ShlOperator
  : public ConcreteOperator<OverflowingBinaryOperator, Instruction::Shl> {
};


class SDivOperator
  : public ConcreteOperator<PossiblyExactOperator, Instruction::SDiv> {
};
class UDivOperator
  : public ConcreteOperator<PossiblyExactOperator, Instruction::UDiv> {
};
class AShrOperator
  : public ConcreteOperator<PossiblyExactOperator, Instruction::AShr> {
};
class LShrOperator
  : public ConcreteOperator<PossiblyExactOperator, Instruction::LShr> {
};


class ZExtOperator : public ConcreteOperator<Operator, Instruction::ZExt> {};


class GEPOperator
  : public ConcreteOperator<Operator, Instruction::GetElementPtr> {
  enum {
    IsInBounds = (1 << 0)
  };

  friend class GetElementPtrInst;
  friend class ConstantExpr;
  void setIsInBounds(bool B) {
    SubclassOptionalData =
      (SubclassOptionalData & ~IsInBounds) | (B * IsInBounds);
  }

public:
  /// Test whether this is an inbounds GEP, as defined by LangRef.html.
  bool isInBounds() const {
    return SubclassOptionalData & IsInBounds;
  }

  inline op_iterator       idx_begin()       { return op_begin()+1; }
  inline const_op_iterator idx_begin() const { return op_begin()+1; }
  inline op_iterator       idx_end()         { return op_end(); }
  inline const_op_iterator idx_end()   const { return op_end(); }

  Value *getPointerOperand() {
    return getOperand(0);
  }
  const Value *getPointerOperand() const {
    return getOperand(0);
  }
  static unsigned getPointerOperandIndex() {
    return 0U;                      // get index for modifying correct operand
  }

  /// Method to return the pointer operand as a PointerType.
  Type *getPointerOperandType() const {
    return getPointerOperand()->getType();
  }

  Type *getSourceElementType() const;

  /// Method to return the address space of the pointer operand.
  unsigned getPointerAddressSpace() const {
    return getPointerOperandType()->getPointerAddressSpace();
  }

  unsigned getNumIndices() const {  // Note: always non-negative
    return getNumOperands() - 1;
  }

  bool hasIndices() const {
    return getNumOperands() > 1;
  }

  /// Return true if all of the indices of this GEP are zeros.
  /// If so, the result pointer and the first operand have the same
  /// value, just potentially different types.
  bool hasAllZeroIndices() const {
    for (const_op_iterator I = idx_begin(), E = idx_end(); I != E; ++I) {
      if (ConstantInt *C = dyn_cast<ConstantInt>(I))
        if (C->isZero())
          continue;
      return false;
    }
    return true;
  }

  /// Return true if all of the indices of this GEP are constant integers.
  /// If so, the result pointer and the first operand have
  /// a constant offset between them.
  bool hasAllConstantIndices() const {
    for (const_op_iterator I = idx_begin(), E = idx_end(); I != E; ++I) {
      if (!isa<ConstantInt>(I))
        return false;
    }
    return true;
  }

  /// \brief Accumulate the constant address offset of this GEP if possible.
  ///
  /// This routine accepts an APInt into which it will accumulate the constant
  /// offset of this GEP if the GEP is in fact constant. If the GEP is not
  /// all-constant, it returns false and the value of the offset APInt is
  /// undefined (it is *not* preserved!). The APInt passed into this routine
  /// must be at exactly as wide as the IntPtr type for the address space of the
  /// base GEP pointer.
  bool accumulateConstantOffset(const DataLayout &DL, APInt &Offset) const;
};

class PtrToIntOperator
    : public ConcreteOperator<Operator, Instruction::PtrToInt> {
  friend class PtrToInt;
  friend class ConstantExpr;

public:
  Value *getPointerOperand() {
    return getOperand(0);
  }
  const Value *getPointerOperand() const {
    return getOperand(0);
  }
  static unsigned getPointerOperandIndex() {
    return 0U;                      // get index for modifying correct operand
  }

  /// Method to return the pointer operand as a PointerType.
  Type *getPointerOperandType() const {
    return getPointerOperand()->getType();
  }

  /// Method to return the address space of the pointer operand.
  unsigned getPointerAddressSpace() const {
    return cast<PointerType>(getPointerOperandType())->getAddressSpace();
  }
};

class BitCastOperator
    : public ConcreteOperator<Operator, Instruction::BitCast> {
  friend class BitCastInst;
  friend class ConstantExpr;

public:
  Type *getSrcTy() const {
    return getOperand(0)->getType();
  }

  Type *getDestTy() const {
    return getType();
  }
};

} // End llvm namespace

#endif

//===-- llvm/IntrinsicInst.h - Intrinsic Instruction Wrappers ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines classes that make it really easy to deal with intrinsic
// functions with the isa/dyncast family of functions.  In particular, this
// allows you to do things like:
//
//     if (MemCpyInst *MCI = dyn_cast<MemCpyInst>(Inst))
//        ... MCI->getDest() ... MCI->getSource() ...
//
// All intrinsic function calls are instances of the call instruction, so these
// are all subclasses of the CallInst class.  Note that none of these classes
// has state or virtual methods, which is an important part of this gross/neat
// hack working.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_INTRINSICINST_H
#define LLVM_IR_INTRINSICINST_H

#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/DebugInfoMetadata.h" // HLSL Change

namespace llvm {
  /// IntrinsicInst - A useful wrapper class for inspecting calls to intrinsic
  /// functions.  This allows the standard isa/dyncast/cast functionality to
  /// work with calls to intrinsic functions.
  class IntrinsicInst : public CallInst {
    IntrinsicInst() = delete;
    IntrinsicInst(const IntrinsicInst&) = delete;
    void operator=(const IntrinsicInst&) = delete;
  public:
    /// getIntrinsicID - Return the intrinsic ID of this intrinsic.
    ///
    Intrinsic::ID getIntrinsicID() const {
      return getCalledFunction()->getIntrinsicID();
    }

    // Methods for support type inquiry through isa, cast, and dyn_cast:
    static inline bool classof(const CallInst *I) {
      if (const Function *CF = I->getCalledFunction())
        return CF->isIntrinsic();
      return false;
    }
    static inline bool classof(const Value *V) {
      return isa<CallInst>(V) && classof(cast<CallInst>(V));
    }
  };

  /// DbgInfoIntrinsic - This is the common base class for debug info intrinsics
  ///
  class DbgInfoIntrinsic : public IntrinsicInst {
  public:

    // Methods for support type inquiry through isa, cast, and dyn_cast:
    static inline bool classof(const IntrinsicInst *I) {
      switch (I->getIntrinsicID()) {
      case Intrinsic::dbg_declare:
      case Intrinsic::dbg_value:
        return true;
      default: return false;
      }
    }
    static inline bool classof(const Value *V) {
      return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
    }

    static Value *StripCast(Value *C);
  };

  /// DbgDeclareInst - This represents the llvm.dbg.declare instruction.
  ///
  class DbgDeclareInst : public DbgInfoIntrinsic {
  public:
    Value *getAddress() const;
    DILocalVariable *getVariable() const {
      return cast<DILocalVariable>(getRawVariable());
    }
    DIExpression *getExpression() const {
      return cast<DIExpression>(getRawExpression());
    }

    void setVariable(DIVariable *v) { setArgOperand(1, MetadataAsValue::get(getContext(), v)); } // HLSL Change

    Metadata *getRawVariable() const {
      return cast<MetadataAsValue>(getArgOperand(1))->getMetadata();
    }
    Metadata *getRawExpression() const {
      return cast<MetadataAsValue>(getArgOperand(2))->getMetadata();
    }

    // Methods for support type inquiry through isa, cast, and dyn_cast:
    static inline bool classof(const IntrinsicInst *I) {
      return I->getIntrinsicID() == Intrinsic::dbg_declare;
    }
    static inline bool classof(const Value *V) {
      return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
    }
  };

  /// DbgValueInst - This represents the llvm.dbg.value instruction.
  ///
  class DbgValueInst : public DbgInfoIntrinsic {
  public:
    const Value *getValue() const;
    Value *getValue();
    uint64_t getOffset() const {
      return cast<ConstantInt>(
                          const_cast<Value*>(getArgOperand(1)))->getZExtValue();
    }
    DILocalVariable *getVariable() const {
      return cast<DILocalVariable>(getRawVariable());
    }
    DIExpression *getExpression() const {
      return cast<DIExpression>(getRawExpression());
    }

    void setVariable(DIVariable *v) { setArgOperand(2, MetadataAsValue::get(getContext(), v)); } // HLSL Change

    Metadata *getRawVariable() const {
      return cast<MetadataAsValue>(getArgOperand(2))->getMetadata();
    }
    Metadata *getRawExpression() const {
      return cast<MetadataAsValue>(getArgOperand(3))->getMetadata();
    }

    // Methods for support type inquiry through isa, cast, and dyn_cast:
    static inline bool classof(const IntrinsicInst *I) {
      return I->getIntrinsicID() == Intrinsic::dbg_value;
    }
    static inline bool classof(const Value *V) {
      return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
    }
  };

  /// MemIntrinsic - This is the common base class for memset/memcpy/memmove.
  ///
  class MemIntrinsic : public IntrinsicInst {
  public:
    Value *getRawDest() const { return const_cast<Value*>(getArgOperand(0)); }
    const Use &getRawDestUse() const { return getArgOperandUse(0); }
    Use &getRawDestUse() { return getArgOperandUse(0); }

    Value *getLength() const { return const_cast<Value*>(getArgOperand(2)); }
    const Use &getLengthUse() const { return getArgOperandUse(2); }
    Use &getLengthUse() { return getArgOperandUse(2); }

    ConstantInt *getAlignmentCst() const {
      return cast<ConstantInt>(const_cast<Value*>(getArgOperand(3)));
    }

    unsigned getAlignment() const {
      return getAlignmentCst()->getZExtValue();
    }

    ConstantInt *getVolatileCst() const {
      return cast<ConstantInt>(const_cast<Value*>(getArgOperand(4)));
    }
    bool isVolatile() const {
      return !getVolatileCst()->isZero();
    }

    unsigned getDestAddressSpace() const {
      return cast<PointerType>(getRawDest()->getType())->getAddressSpace();
    }

    /// getDest - This is just like getRawDest, but it strips off any cast
    /// instructions that feed it, giving the original input.  The returned
    /// value is guaranteed to be a pointer.
    Value *getDest() const { return getRawDest()->stripPointerCasts(); }

    /// set* - Set the specified arguments of the instruction.
    ///
    void setDest(Value *Ptr) {
      assert(getRawDest()->getType() == Ptr->getType() &&
             "setDest called with pointer of wrong type!");
      setArgOperand(0, Ptr);
    }

    void setLength(Value *L) {
      assert(getLength()->getType() == L->getType() &&
             "setLength called with value of wrong type!");
      setArgOperand(2, L);
    }

    void setAlignment(Constant* A) {
      setArgOperand(3, A);
    }

    void setVolatile(Constant* V) {
      setArgOperand(4, V);
    }

    Type *getAlignmentType() const {
      return getArgOperand(3)->getType();
    }

    // Methods for support type inquiry through isa, cast, and dyn_cast:
    static inline bool classof(const IntrinsicInst *I) {
      switch (I->getIntrinsicID()) {
      case Intrinsic::memcpy:
      case Intrinsic::memmove:
      case Intrinsic::memset:
        return true;
      default: return false;
      }
    }
    static inline bool classof(const Value *V) {
      return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
    }
  };

  /// MemSetInst - This class wraps the llvm.memset intrinsic.
  ///
  class MemSetInst : public MemIntrinsic {
  public:
    /// get* - Return the arguments to the instruction.
    ///
    Value *getValue() const { return const_cast<Value*>(getArgOperand(1)); }
    const Use &getValueUse() const { return getArgOperandUse(1); }
    Use &getValueUse() { return getArgOperandUse(1); }

    void setValue(Value *Val) {
      assert(getValue()->getType() == Val->getType() &&
             "setValue called with value of wrong type!");
      setArgOperand(1, Val);
    }

    // Methods for support type inquiry through isa, cast, and dyn_cast:
    static inline bool classof(const IntrinsicInst *I) {
      return I->getIntrinsicID() == Intrinsic::memset;
    }
    static inline bool classof(const Value *V) {
      return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
    }
  };

  /// MemTransferInst - This class wraps the llvm.memcpy/memmove intrinsics.
  ///
  class MemTransferInst : public MemIntrinsic {
  public:
    /// get* - Return the arguments to the instruction.
    ///
    Value *getRawSource() const { return const_cast<Value*>(getArgOperand(1)); }
    const Use &getRawSourceUse() const { return getArgOperandUse(1); }
    Use &getRawSourceUse() { return getArgOperandUse(1); }

    /// getSource - This is just like getRawSource, but it strips off any cast
    /// instructions that feed it, giving the original input.  The returned
    /// value is guaranteed to be a pointer.
    Value *getSource() const { return getRawSource()->stripPointerCasts(); }

    unsigned getSourceAddressSpace() const {
      return cast<PointerType>(getRawSource()->getType())->getAddressSpace();
    }

    void setSource(Value *Ptr) {
      assert(getRawSource()->getType() == Ptr->getType() &&
             "setSource called with pointer of wrong type!");
      setArgOperand(1, Ptr);
    }

    // Methods for support type inquiry through isa, cast, and dyn_cast:
    static inline bool classof(const IntrinsicInst *I) {
      return I->getIntrinsicID() == Intrinsic::memcpy ||
             I->getIntrinsicID() == Intrinsic::memmove;
    }
    static inline bool classof(const Value *V) {
      return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
    }
  };


  /// MemCpyInst - This class wraps the llvm.memcpy intrinsic.
  ///
  class MemCpyInst : public MemTransferInst {
  public:
    // Methods for support type inquiry through isa, cast, and dyn_cast:
    static inline bool classof(const IntrinsicInst *I) {
      return I->getIntrinsicID() == Intrinsic::memcpy;
    }
    static inline bool classof(const Value *V) {
      return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
    }
  };

  /// MemMoveInst - This class wraps the llvm.memmove intrinsic.
  ///
  class MemMoveInst : public MemTransferInst {
  public:
    // Methods for support type inquiry through isa, cast, and dyn_cast:
    static inline bool classof(const IntrinsicInst *I) {
      return I->getIntrinsicID() == Intrinsic::memmove;
    }
    static inline bool classof(const Value *V) {
      return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
    }
  };

  /// VAStartInst - This represents the llvm.va_start intrinsic.
  ///
  class VAStartInst : public IntrinsicInst {
  public:
    static inline bool classof(const IntrinsicInst *I) {
      return I->getIntrinsicID() == Intrinsic::vastart;
    }
    static inline bool classof(const Value *V) {
      return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
    }

    Value *getArgList() const { return const_cast<Value*>(getArgOperand(0)); }
  };

  /// VAEndInst - This represents the llvm.va_end intrinsic.
  ///
  class VAEndInst : public IntrinsicInst {
  public:
    static inline bool classof(const IntrinsicInst *I) {
      return I->getIntrinsicID() == Intrinsic::vaend;
    }
    static inline bool classof(const Value *V) {
      return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
    }

    Value *getArgList() const { return const_cast<Value*>(getArgOperand(0)); }
  };

  /// VACopyInst - This represents the llvm.va_copy intrinsic.
  ///
  class VACopyInst : public IntrinsicInst {
  public:
    static inline bool classof(const IntrinsicInst *I) {
      return I->getIntrinsicID() == Intrinsic::vacopy;
    }
    static inline bool classof(const Value *V) {
      return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
    }

    Value *getDest() const { return const_cast<Value*>(getArgOperand(0)); }
    Value *getSrc() const { return const_cast<Value*>(getArgOperand(1)); }
  };

  /// This represents the llvm.instrprof_increment intrinsic.
  class InstrProfIncrementInst : public IntrinsicInst {
  public:
    static inline bool classof(const IntrinsicInst *I) {
      return I->getIntrinsicID() == Intrinsic::instrprof_increment;
    }
    static inline bool classof(const Value *V) {
      return isa<IntrinsicInst>(V) && classof(cast<IntrinsicInst>(V));
    }

    GlobalVariable *getName() const {
      return cast<GlobalVariable>(
          const_cast<Value *>(getArgOperand(0))->stripPointerCasts());
    }

    ConstantInt *getHash() const {
      return cast<ConstantInt>(const_cast<Value *>(getArgOperand(1)));
    }

    ConstantInt *getNumCounters() const {
      return cast<ConstantInt>(const_cast<Value *>(getArgOperand(2)));
    }

    ConstantInt *getIndex() const {
      return cast<ConstantInt>(const_cast<Value *>(getArgOperand(3)));
    }
  };
}

#endif

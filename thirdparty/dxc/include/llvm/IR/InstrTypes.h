//===-- llvm/InstrTypes.h - Important Instruction subclasses ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines various meta classes of instructions that exist in the VM
// representation.  Specific concrete subclasses of these may be found in the
// i*.h files...
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_INSTRTYPES_H
#define LLVM_IR_INSTRTYPES_H

#include "llvm/ADT/Twine.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/OperandTraits.h"

namespace llvm {

class LLVMContext;

//===----------------------------------------------------------------------===//
//                            TerminatorInst Class
//===----------------------------------------------------------------------===//

/// Subclasses of this class are all able to terminate a basic
/// block. Thus, these are all the flow control type of operations.
///
class TerminatorInst : public Instruction {
protected:
  TerminatorInst(Type *Ty, Instruction::TermOps iType,
                 Use *Ops, unsigned NumOps,
                 Instruction *InsertBefore = nullptr)
    : Instruction(Ty, iType, Ops, NumOps, InsertBefore) {}

  TerminatorInst(Type *Ty, Instruction::TermOps iType,
                 Use *Ops, unsigned NumOps, BasicBlock *InsertAtEnd)
    : Instruction(Ty, iType, Ops, NumOps, InsertAtEnd) {}

  // Out of line virtual method, so the vtable, etc has a home.
  ~TerminatorInst() override;

  /// Virtual methods - Terminators should overload these and provide inline
  /// overrides of non-V methods.
  virtual BasicBlock *getSuccessorV(unsigned idx) const = 0;
  virtual unsigned getNumSuccessorsV() const = 0;
  virtual void setSuccessorV(unsigned idx, BasicBlock *B) = 0;
public:

  /// Return the number of successors that this terminator has.
  unsigned getNumSuccessors() const {
    return getNumSuccessorsV();
  }

  /// Return the specified successor.
  BasicBlock *getSuccessor(unsigned idx) const {
    return getSuccessorV(idx);
  }

  /// Update the specified successor to point at the provided block.
  void setSuccessor(unsigned idx, BasicBlock *B) {
    setSuccessorV(idx, B);
  }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Instruction *I) {
    return I->isTerminator();
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};


//===----------------------------------------------------------------------===//
//                          UnaryInstruction Class
//===----------------------------------------------------------------------===//

class UnaryInstruction : public Instruction {
  void *operator new(size_t, unsigned) = delete;

protected:
  UnaryInstruction(Type *Ty, unsigned iType, Value *V,
                   Instruction *IB = nullptr)
    : Instruction(Ty, iType, &Op<0>(), 1, IB) {
    Op<0>() = V;
  }
  UnaryInstruction(Type *Ty, unsigned iType, Value *V, BasicBlock *IAE)
    : Instruction(Ty, iType, &Op<0>(), 1, IAE) {
    Op<0>() = V;
  }
public:
  // allocate space for exactly one operand
  void *operator new(size_t s) {
    return User::operator new(s, 1);
  }

  // Out of line virtual method, so the vtable, etc has a home.
  ~UnaryInstruction() override;

  /// Transparently provide more efficient getOperand methods.
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::Alloca ||
           I->getOpcode() == Instruction::Load ||
           I->getOpcode() == Instruction::VAArg ||
           I->getOpcode() == Instruction::ExtractValue ||
           (I->getOpcode() >= CastOpsBegin && I->getOpcode() < CastOpsEnd);
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

template <>
struct OperandTraits<UnaryInstruction> :
  public FixedNumOperandTraits<UnaryInstruction, 1> {
};

DEFINE_TRANSPARENT_OPERAND_ACCESSORS(UnaryInstruction, Value)

//===----------------------------------------------------------------------===//
//                           BinaryOperator Class
//===----------------------------------------------------------------------===//

class BinaryOperator : public Instruction {
  void *operator new(size_t, unsigned) = delete;
protected:
  void init(BinaryOps iType);
  BinaryOperator(BinaryOps iType, Value *S1, Value *S2, Type *Ty,
                 const Twine &Name, Instruction *InsertBefore);
  BinaryOperator(BinaryOps iType, Value *S1, Value *S2, Type *Ty,
                 const Twine &Name, BasicBlock *InsertAtEnd);

  // Note: Instruction needs to be a friend here to call cloneImpl.
  friend class Instruction;
  BinaryOperator *cloneImpl() const;

public:
  // allocate space for exactly two operands
  void *operator new(size_t s) {
    return User::operator new(s, 2);
  }

  /// Transparently provide more efficient getOperand methods.
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);

  /// Construct a binary instruction, given the opcode and the two
  /// operands.  Optionally (if InstBefore is specified) insert the instruction
  /// into a BasicBlock right before the specified instruction.  The specified
  /// Instruction is allowed to be a dereferenced end iterator.
  ///
  static BinaryOperator *Create(BinaryOps Op, Value *S1, Value *S2,
                                const Twine &Name = Twine(),
                                Instruction *InsertBefore = nullptr);

  /// Construct a binary instruction, given the opcode and the two
  /// operands.  Also automatically insert this instruction to the end of the
  /// BasicBlock specified.
  ///
  static BinaryOperator *Create(BinaryOps Op, Value *S1, Value *S2,
                                const Twine &Name, BasicBlock *InsertAtEnd);

  /// These methods just forward to Create, and are useful when you
  /// statically know what type of instruction you're going to create.  These
  /// helpers just save some typing.
#define HANDLE_BINARY_INST(N, OPC, CLASS) \
  static BinaryOperator *Create##OPC(Value *V1, Value *V2, \
                                     const Twine &Name = "") {\
    return Create(Instruction::OPC, V1, V2, Name);\
  }
#include "llvm/IR/Instruction.def"
#define HANDLE_BINARY_INST(N, OPC, CLASS) \
  static BinaryOperator *Create##OPC(Value *V1, Value *V2, \
                                     const Twine &Name, BasicBlock *BB) {\
    return Create(Instruction::OPC, V1, V2, Name, BB);\
  }
#include "llvm/IR/Instruction.def"
#define HANDLE_BINARY_INST(N, OPC, CLASS) \
  static BinaryOperator *Create##OPC(Value *V1, Value *V2, \
                                     const Twine &Name, Instruction *I) {\
    return Create(Instruction::OPC, V1, V2, Name, I);\
  }
#include "llvm/IR/Instruction.def"

  static BinaryOperator *CreateNSW(BinaryOps Opc, Value *V1, Value *V2,
                                   const Twine &Name = "") {
    BinaryOperator *BO = Create(Opc, V1, V2, Name);
    BO->setHasNoSignedWrap(true);
    return BO;
  }
  static BinaryOperator *CreateNSW(BinaryOps Opc, Value *V1, Value *V2,
                                   const Twine &Name, BasicBlock *BB) {
    BinaryOperator *BO = Create(Opc, V1, V2, Name, BB);
    BO->setHasNoSignedWrap(true);
    return BO;
  }
  static BinaryOperator *CreateNSW(BinaryOps Opc, Value *V1, Value *V2,
                                   const Twine &Name, Instruction *I) {
    BinaryOperator *BO = Create(Opc, V1, V2, Name, I);
    BO->setHasNoSignedWrap(true);
    return BO;
  }
  
  static BinaryOperator *CreateNUW(BinaryOps Opc, Value *V1, Value *V2,
                                   const Twine &Name = "") {
    BinaryOperator *BO = Create(Opc, V1, V2, Name);
    BO->setHasNoUnsignedWrap(true);
    return BO;
  }
  static BinaryOperator *CreateNUW(BinaryOps Opc, Value *V1, Value *V2,
                                   const Twine &Name, BasicBlock *BB) {
    BinaryOperator *BO = Create(Opc, V1, V2, Name, BB);
    BO->setHasNoUnsignedWrap(true);
    return BO;
  }
  static BinaryOperator *CreateNUW(BinaryOps Opc, Value *V1, Value *V2,
                                   const Twine &Name, Instruction *I) {
    BinaryOperator *BO = Create(Opc, V1, V2, Name, I);
    BO->setHasNoUnsignedWrap(true);
    return BO;
  }
  
  static BinaryOperator *CreateExact(BinaryOps Opc, Value *V1, Value *V2,
                                     const Twine &Name = "") {
    BinaryOperator *BO = Create(Opc, V1, V2, Name);
    BO->setIsExact(true);
    return BO;
  }
  static BinaryOperator *CreateExact(BinaryOps Opc, Value *V1, Value *V2,
                                     const Twine &Name, BasicBlock *BB) {
    BinaryOperator *BO = Create(Opc, V1, V2, Name, BB);
    BO->setIsExact(true);
    return BO;
  }
  static BinaryOperator *CreateExact(BinaryOps Opc, Value *V1, Value *V2,
                                     const Twine &Name, Instruction *I) {
    BinaryOperator *BO = Create(Opc, V1, V2, Name, I);
    BO->setIsExact(true);
    return BO;
  }
  
#define DEFINE_HELPERS(OPC, NUWNSWEXACT)                                     \
  static BinaryOperator *Create ## NUWNSWEXACT ## OPC                        \
           (Value *V1, Value *V2, const Twine &Name = "") {                  \
    return Create ## NUWNSWEXACT(Instruction::OPC, V1, V2, Name);            \
  }                                                                          \
  static BinaryOperator *Create ## NUWNSWEXACT ## OPC                        \
           (Value *V1, Value *V2, const Twine &Name, BasicBlock *BB) {       \
    return Create ## NUWNSWEXACT(Instruction::OPC, V1, V2, Name, BB);        \
  }                                                                          \
  static BinaryOperator *Create ## NUWNSWEXACT ## OPC                        \
           (Value *V1, Value *V2, const Twine &Name, Instruction *I) {       \
    return Create ## NUWNSWEXACT(Instruction::OPC, V1, V2, Name, I);         \
  }
  
  DEFINE_HELPERS(Add, NSW)  // CreateNSWAdd
  DEFINE_HELPERS(Add, NUW)  // CreateNUWAdd
  DEFINE_HELPERS(Sub, NSW)  // CreateNSWSub
  DEFINE_HELPERS(Sub, NUW)  // CreateNUWSub
  DEFINE_HELPERS(Mul, NSW)  // CreateNSWMul
  DEFINE_HELPERS(Mul, NUW)  // CreateNUWMul
  DEFINE_HELPERS(Shl, NSW)  // CreateNSWShl
  DEFINE_HELPERS(Shl, NUW)  // CreateNUWShl

  DEFINE_HELPERS(SDiv, Exact)  // CreateExactSDiv
  DEFINE_HELPERS(UDiv, Exact)  // CreateExactUDiv
  DEFINE_HELPERS(AShr, Exact)  // CreateExactAShr
  DEFINE_HELPERS(LShr, Exact)  // CreateExactLShr

#undef DEFINE_HELPERS
  
  /// Helper functions to construct and inspect unary operations (NEG and NOT)
  /// via binary operators SUB and XOR:
  ///
  /// Create the NEG and NOT instructions out of SUB and XOR instructions.
  ///
  static BinaryOperator *CreateNeg(Value *Op, const Twine &Name = "",
                                   Instruction *InsertBefore = nullptr);
  static BinaryOperator *CreateNeg(Value *Op, const Twine &Name,
                                   BasicBlock *InsertAtEnd);
  static BinaryOperator *CreateNSWNeg(Value *Op, const Twine &Name = "",
                                      Instruction *InsertBefore = nullptr);
  static BinaryOperator *CreateNSWNeg(Value *Op, const Twine &Name,
                                      BasicBlock *InsertAtEnd);
  static BinaryOperator *CreateNUWNeg(Value *Op, const Twine &Name = "",
                                      Instruction *InsertBefore = nullptr);
  static BinaryOperator *CreateNUWNeg(Value *Op, const Twine &Name,
                                      BasicBlock *InsertAtEnd);
  static BinaryOperator *CreateFNeg(Value *Op, const Twine &Name = "",
                                    Instruction *InsertBefore = nullptr);
  static BinaryOperator *CreateFNeg(Value *Op, const Twine &Name,
                                    BasicBlock *InsertAtEnd);
  static BinaryOperator *CreateNot(Value *Op, const Twine &Name = "",
                                   Instruction *InsertBefore = nullptr);
  static BinaryOperator *CreateNot(Value *Op, const Twine &Name,
                                   BasicBlock *InsertAtEnd);

  /// Check if the given Value is a NEG, FNeg, or NOT instruction.
  ///
  static bool isNeg(const Value *V);
  static bool isFNeg(const Value *V, bool IgnoreZeroSign=false);
  static bool isNot(const Value *V);

  /// Helper functions to extract the unary argument of a NEG, FNEG or NOT
  /// operation implemented via Sub, FSub, or Xor.
  ///
  static const Value *getNegArgument(const Value *BinOp);
  static       Value *getNegArgument(      Value *BinOp);
  static const Value *getFNegArgument(const Value *BinOp);
  static       Value *getFNegArgument(      Value *BinOp);
  static const Value *getNotArgument(const Value *BinOp);
  static       Value *getNotArgument(      Value *BinOp);

  BinaryOps getOpcode() const {
    return static_cast<BinaryOps>(Instruction::getOpcode());
  }

  /// Exchange the two operands to this instruction.
  /// This instruction is safe to use on any binary instruction and
  /// does not modify the semantics of the instruction.  If the instruction
  /// cannot be reversed (ie, it's a Div), then return true.
  ///
  bool swapOperands();

  /// Set or clear the nsw flag on this instruction, which must be an operator
  /// which supports this flag. See LangRef.html for the meaning of this flag.
  void setHasNoUnsignedWrap(bool b = true);

  /// Set or clear the nsw flag on this instruction, which must be an operator
  /// which supports this flag. See LangRef.html for the meaning of this flag.
  void setHasNoSignedWrap(bool b = true);

  /// Set or clear the exact flag on this instruction, which must be an operator
  /// which supports this flag. See LangRef.html for the meaning of this flag.
  void setIsExact(bool b = true);

  /// Determine whether the no unsigned wrap flag is set.
  bool hasNoUnsignedWrap() const;

  /// Determine whether the no signed wrap flag is set.
  bool hasNoSignedWrap() const;

  /// Determine whether the exact flag is set.
  bool isExact() const;

  /// Convenience method to copy supported wrapping, exact, and fast-math flags
  /// from V to this instruction.
  void copyIRFlags(const Value *V);
  
  /// Logical 'and' of any supported wrapping, exact, and fast-math flags of
  /// V and this instruction.
  void andIRFlags(const Value *V);

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Instruction *I) {
    return I->isBinaryOp();
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

template <>
struct OperandTraits<BinaryOperator> :
  public FixedNumOperandTraits<BinaryOperator, 2> {
};

DEFINE_TRANSPARENT_OPERAND_ACCESSORS(BinaryOperator, Value)

//===----------------------------------------------------------------------===//
//                               CastInst Class
//===----------------------------------------------------------------------===//

/// This is the base class for all instructions that perform data
/// casts. It is simply provided so that instruction category testing
/// can be performed with code like:
///
/// if (isa<CastInst>(Instr)) { ... }
/// @brief Base class of casting instructions.
class CastInst : public UnaryInstruction {
  void anchor() override;
protected:
  /// @brief Constructor with insert-before-instruction semantics for subclasses
  CastInst(Type *Ty, unsigned iType, Value *S,
           const Twine &NameStr = "", Instruction *InsertBefore = nullptr)
    : UnaryInstruction(Ty, iType, S, InsertBefore) {
    setName(NameStr);
  }
  /// @brief Constructor with insert-at-end-of-block semantics for subclasses
  CastInst(Type *Ty, unsigned iType, Value *S,
           const Twine &NameStr, BasicBlock *InsertAtEnd)
    : UnaryInstruction(Ty, iType, S, InsertAtEnd) {
    setName(NameStr);
  }
public:
  /// Provides a way to construct any of the CastInst subclasses using an
  /// opcode instead of the subclass's constructor. The opcode must be in the
  /// CastOps category (Instruction::isCast(opcode) returns true). This
  /// constructor has insert-before-instruction semantics to automatically
  /// insert the new CastInst before InsertBefore (if it is non-null).
  /// @brief Construct any of the CastInst subclasses
  static CastInst *Create(
    Instruction::CastOps,    ///< The opcode of the cast instruction
    Value *S,                ///< The value to be casted (operand 0)
    Type *Ty,          ///< The type to which cast should be made
    const Twine &Name = "", ///< Name for the instruction
    Instruction *InsertBefore = nullptr ///< Place to insert the instruction
  );
  /// Provides a way to construct any of the CastInst subclasses using an
  /// opcode instead of the subclass's constructor. The opcode must be in the
  /// CastOps category. This constructor has insert-at-end-of-block semantics
  /// to automatically insert the new CastInst at the end of InsertAtEnd (if
  /// its non-null).
  /// @brief Construct any of the CastInst subclasses
  static CastInst *Create(
    Instruction::CastOps,    ///< The opcode for the cast instruction
    Value *S,                ///< The value to be casted (operand 0)
    Type *Ty,          ///< The type to which operand is casted
    const Twine &Name, ///< The name for the instruction
    BasicBlock *InsertAtEnd  ///< The block to insert the instruction into
  );

  /// @brief Create a ZExt or BitCast cast instruction
  static CastInst *CreateZExtOrBitCast(
    Value *S,                ///< The value to be casted (operand 0)
    Type *Ty,          ///< The type to which cast should be made
    const Twine &Name = "", ///< Name for the instruction
    Instruction *InsertBefore = nullptr ///< Place to insert the instruction
  );

  /// @brief Create a ZExt or BitCast cast instruction
  static CastInst *CreateZExtOrBitCast(
    Value *S,                ///< The value to be casted (operand 0)
    Type *Ty,          ///< The type to which operand is casted
    const Twine &Name, ///< The name for the instruction
    BasicBlock *InsertAtEnd  ///< The block to insert the instruction into
  );

  /// @brief Create a SExt or BitCast cast instruction
  static CastInst *CreateSExtOrBitCast(
    Value *S,                ///< The value to be casted (operand 0)
    Type *Ty,          ///< The type to which cast should be made
    const Twine &Name = "", ///< Name for the instruction
    Instruction *InsertBefore = nullptr ///< Place to insert the instruction
  );

  /// @brief Create a SExt or BitCast cast instruction
  static CastInst *CreateSExtOrBitCast(
    Value *S,                ///< The value to be casted (operand 0)
    Type *Ty,          ///< The type to which operand is casted
    const Twine &Name, ///< The name for the instruction
    BasicBlock *InsertAtEnd  ///< The block to insert the instruction into
  );

  /// @brief Create a BitCast AddrSpaceCast, or a PtrToInt cast instruction.
  static CastInst *CreatePointerCast(
    Value *S,                ///< The pointer value to be casted (operand 0)
    Type *Ty,          ///< The type to which operand is casted
    const Twine &Name, ///< The name for the instruction
    BasicBlock *InsertAtEnd  ///< The block to insert the instruction into
  );

  /// @brief Create a BitCast, AddrSpaceCast or a PtrToInt cast instruction.
  static CastInst *CreatePointerCast(
    Value *S,                ///< The pointer value to be casted (operand 0)
    Type *Ty,          ///< The type to which cast should be made
    const Twine &Name = "", ///< Name for the instruction
    Instruction *InsertBefore = nullptr ///< Place to insert the instruction
  );

  /// @brief Create a BitCast or an AddrSpaceCast cast instruction.
  static CastInst *CreatePointerBitCastOrAddrSpaceCast(
    Value *S,                ///< The pointer value to be casted (operand 0)
    Type *Ty,          ///< The type to which operand is casted
    const Twine &Name, ///< The name for the instruction
    BasicBlock *InsertAtEnd  ///< The block to insert the instruction into
  );

  /// @brief Create a BitCast or an AddrSpaceCast cast instruction.
  static CastInst *CreatePointerBitCastOrAddrSpaceCast(
    Value *S,                ///< The pointer value to be casted (operand 0)
    Type *Ty,          ///< The type to which cast should be made
    const Twine &Name = "", ///< Name for the instruction
    Instruction *InsertBefore = 0 ///< Place to insert the instruction
  );

  /// @brief Create a BitCast, a PtrToInt, or an IntToPTr cast instruction.
  ///
  /// If the value is a pointer type and the destination an integer type,
  /// creates a PtrToInt cast. If the value is an integer type and the
  /// destination a pointer type, creates an IntToPtr cast. Otherwise, creates
  /// a bitcast.
  static CastInst *CreateBitOrPointerCast(
    Value *S,                ///< The pointer value to be casted (operand 0)
    Type *Ty,          ///< The type to which cast should be made
    const Twine &Name = "", ///< Name for the instruction
    Instruction *InsertBefore = 0 ///< Place to insert the instruction
  );

  /// @brief Create a ZExt, BitCast, or Trunc for int -> int casts.
  static CastInst *CreateIntegerCast(
    Value *S,                ///< The pointer value to be casted (operand 0)
    Type *Ty,          ///< The type to which cast should be made
    bool isSigned,           ///< Whether to regard S as signed or not
    const Twine &Name = "", ///< Name for the instruction
    Instruction *InsertBefore = nullptr ///< Place to insert the instruction
  );

  /// @brief Create a ZExt, BitCast, or Trunc for int -> int casts.
  static CastInst *CreateIntegerCast(
    Value *S,                ///< The integer value to be casted (operand 0)
    Type *Ty,          ///< The integer type to which operand is casted
    bool isSigned,           ///< Whether to regard S as signed or not
    const Twine &Name, ///< The name for the instruction
    BasicBlock *InsertAtEnd  ///< The block to insert the instruction into
  );

  /// @brief Create an FPExt, BitCast, or FPTrunc for fp -> fp casts
  static CastInst *CreateFPCast(
    Value *S,                ///< The floating point value to be casted
    Type *Ty,          ///< The floating point type to cast to
    const Twine &Name = "", ///< Name for the instruction
    Instruction *InsertBefore = nullptr ///< Place to insert the instruction
  );

  /// @brief Create an FPExt, BitCast, or FPTrunc for fp -> fp casts
  static CastInst *CreateFPCast(
    Value *S,                ///< The floating point value to be casted
    Type *Ty,          ///< The floating point type to cast to
    const Twine &Name, ///< The name for the instruction
    BasicBlock *InsertAtEnd  ///< The block to insert the instruction into
  );

  /// @brief Create a Trunc or BitCast cast instruction
  static CastInst *CreateTruncOrBitCast(
    Value *S,                ///< The value to be casted (operand 0)
    Type *Ty,          ///< The type to which cast should be made
    const Twine &Name = "", ///< Name for the instruction
    Instruction *InsertBefore = nullptr ///< Place to insert the instruction
  );

  /// @brief Create a Trunc or BitCast cast instruction
  static CastInst *CreateTruncOrBitCast(
    Value *S,                ///< The value to be casted (operand 0)
    Type *Ty,          ///< The type to which operand is casted
    const Twine &Name, ///< The name for the instruction
    BasicBlock *InsertAtEnd  ///< The block to insert the instruction into
  );

  /// @brief Check whether it is valid to call getCastOpcode for these types.
  static bool isCastable(
    Type *SrcTy, ///< The Type from which the value should be cast.
    Type *DestTy ///< The Type to which the value should be cast.
  );

  /// @brief Check whether a bitcast between these types is valid
  static bool isBitCastable(
    Type *SrcTy, ///< The Type from which the value should be cast.
    Type *DestTy ///< The Type to which the value should be cast.
  );

  /// @brief Check whether a bitcast, inttoptr, or ptrtoint cast between these
  /// types is valid and a no-op.
  ///
  /// This ensures that any pointer<->integer cast has enough bits in the
  /// integer and any other cast is a bitcast.
  static bool isBitOrNoopPointerCastable(
      Type *SrcTy,  ///< The Type from which the value should be cast.
      Type *DestTy, ///< The Type to which the value should be cast.
      const DataLayout &DL);

  /// Returns the opcode necessary to cast Val into Ty using usual casting
  /// rules.
  /// @brief Infer the opcode for cast operand and type
  static Instruction::CastOps getCastOpcode(
    const Value *Val, ///< The value to cast
    bool SrcIsSigned, ///< Whether to treat the source as signed
    Type *Ty,   ///< The Type to which the value should be casted
    bool DstIsSigned  ///< Whether to treate the dest. as signed
  );

  /// There are several places where we need to know if a cast instruction
  /// only deals with integer source and destination types. To simplify that
  /// logic, this method is provided.
  /// @returns true iff the cast has only integral typed operand and dest type.
  /// @brief Determine if this is an integer-only cast.
  bool isIntegerCast() const;

  /// A lossless cast is one that does not alter the basic value. It implies
  /// a no-op cast but is more stringent, preventing things like int->float,
  /// long->double, or int->ptr.
  /// @returns true iff the cast is lossless.
  /// @brief Determine if this is a lossless cast.
  bool isLosslessCast() const;

  /// A no-op cast is one that can be effected without changing any bits.
  /// It implies that the source and destination types are the same size. The
  /// IntPtrTy argument is used to make accurate determinations for casts
  /// involving Integer and Pointer types. They are no-op casts if the integer
  /// is the same size as the pointer. However, pointer size varies with
  /// platform. Generally, the result of DataLayout::getIntPtrType() should be
  /// passed in. If that's not available, use Type::Int64Ty, which will make
  /// the isNoopCast call conservative.
  /// @brief Determine if the described cast is a no-op cast.
  static bool isNoopCast(
    Instruction::CastOps Opcode,  ///< Opcode of cast
    Type *SrcTy,   ///< SrcTy of cast
    Type *DstTy,   ///< DstTy of cast
    Type *IntPtrTy ///< Integer type corresponding to Ptr types
  );

  /// @brief Determine if this cast is a no-op cast.
  bool isNoopCast(
    Type *IntPtrTy ///< Integer type corresponding to pointer
  ) const;

  /// @brief Determine if this cast is a no-op cast.
  ///
  /// \param DL is the DataLayout to get the Int Ptr type from.
  bool isNoopCast(const DataLayout &DL) const;

  /// Determine how a pair of casts can be eliminated, if they can be at all.
  /// This is a helper function for both CastInst and ConstantExpr.
  /// @returns 0 if the CastInst pair can't be eliminated, otherwise
  /// returns Instruction::CastOps value for a cast that can replace
  /// the pair, casting SrcTy to DstTy.
  /// @brief Determine if a cast pair is eliminable
  static unsigned isEliminableCastPair(
    Instruction::CastOps firstOpcode,  ///< Opcode of first cast
    Instruction::CastOps secondOpcode, ///< Opcode of second cast
    Type *SrcTy, ///< SrcTy of 1st cast
    Type *MidTy, ///< DstTy of 1st cast & SrcTy of 2nd cast
    Type *DstTy, ///< DstTy of 2nd cast
    Type *SrcIntPtrTy, ///< Integer type corresponding to Ptr SrcTy, or null
    Type *MidIntPtrTy, ///< Integer type corresponding to Ptr MidTy, or null
    Type *DstIntPtrTy  ///< Integer type corresponding to Ptr DstTy, or null
  );

  /// @brief Return the opcode of this CastInst
  Instruction::CastOps getOpcode() const {
    return Instruction::CastOps(Instruction::getOpcode());
  }

  /// @brief Return the source type, as a convenience
  Type* getSrcTy() const { return getOperand(0)->getType(); }
  /// @brief Return the destination type, as a convenience
  Type* getDestTy() const { return getType(); }

  /// This method can be used to determine if a cast from S to DstTy using
  /// Opcode op is valid or not.
  /// @returns true iff the proposed cast is valid.
  /// @brief Determine if a cast is valid without creating one.
  static bool castIsValid(Instruction::CastOps op, Value *S, Type *DstTy);

  /// @brief Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Instruction *I) {
    return I->isCast();
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

//===----------------------------------------------------------------------===//
//                               CmpInst Class
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

/// This class is the base class for the comparison instructions.
/// @brief Abstract base class of comparison instructions.
class CmpInst : public Instruction {
  void *operator new(size_t, unsigned) = delete;
  CmpInst() = delete;
protected:
  CmpInst(Type *ty, Instruction::OtherOps op, unsigned short pred,
          Value *LHS, Value *RHS, const Twine &Name = "",
          Instruction *InsertBefore = nullptr);

  CmpInst(Type *ty, Instruction::OtherOps op, unsigned short pred,
          Value *LHS, Value *RHS, const Twine &Name,
          BasicBlock *InsertAtEnd);

  void anchor() override; // Out of line virtual method.
public:
  /// This enumeration lists the possible predicates for CmpInst subclasses.
  /// Values in the range 0-31 are reserved for FCmpInst, while values in the
  /// range 32-64 are reserved for ICmpInst. This is necessary to ensure the
  /// predicate values are not overlapping between the classes.
  enum Predicate {
    // Opcode              U L G E    Intuitive operation
    FCMP_FALSE =  0,  ///< 0 0 0 0    Always false (always folded)
    FCMP_OEQ   =  1,  ///< 0 0 0 1    True if ordered and equal
    FCMP_OGT   =  2,  ///< 0 0 1 0    True if ordered and greater than
    FCMP_OGE   =  3,  ///< 0 0 1 1    True if ordered and greater than or equal
    FCMP_OLT   =  4,  ///< 0 1 0 0    True if ordered and less than
    FCMP_OLE   =  5,  ///< 0 1 0 1    True if ordered and less than or equal
    FCMP_ONE   =  6,  ///< 0 1 1 0    True if ordered and operands are unequal
    FCMP_ORD   =  7,  ///< 0 1 1 1    True if ordered (no nans)
    FCMP_UNO   =  8,  ///< 1 0 0 0    True if unordered: isnan(X) | isnan(Y)
    FCMP_UEQ   =  9,  ///< 1 0 0 1    True if unordered or equal
    FCMP_UGT   = 10,  ///< 1 0 1 0    True if unordered or greater than
    FCMP_UGE   = 11,  ///< 1 0 1 1    True if unordered, greater than, or equal
    FCMP_ULT   = 12,  ///< 1 1 0 0    True if unordered or less than
    FCMP_ULE   = 13,  ///< 1 1 0 1    True if unordered, less than, or equal
    FCMP_UNE   = 14,  ///< 1 1 1 0    True if unordered or not equal
    FCMP_TRUE  = 15,  ///< 1 1 1 1    Always true (always folded)
    FIRST_FCMP_PREDICATE = FCMP_FALSE,
    LAST_FCMP_PREDICATE = FCMP_TRUE,
    BAD_FCMP_PREDICATE = FCMP_TRUE + 1,
    ICMP_EQ    = 32,  ///< equal
    ICMP_NE    = 33,  ///< not equal
    ICMP_UGT   = 34,  ///< unsigned greater than
    ICMP_UGE   = 35,  ///< unsigned greater or equal
    ICMP_ULT   = 36,  ///< unsigned less than
    ICMP_ULE   = 37,  ///< unsigned less or equal
    ICMP_SGT   = 38,  ///< signed greater than
    ICMP_SGE   = 39,  ///< signed greater or equal
    ICMP_SLT   = 40,  ///< signed less than
    ICMP_SLE   = 41,  ///< signed less or equal
    FIRST_ICMP_PREDICATE = ICMP_EQ,
    LAST_ICMP_PREDICATE = ICMP_SLE,
    BAD_ICMP_PREDICATE = ICMP_SLE + 1
  };

  // allocate space for exactly two operands
  void *operator new(size_t s) {
    return User::operator new(s, 2);
  }
  /// Construct a compare instruction, given the opcode, the predicate and
  /// the two operands.  Optionally (if InstBefore is specified) insert the
  /// instruction into a BasicBlock right before the specified instruction.
  /// The specified Instruction is allowed to be a dereferenced end iterator.
  /// @brief Create a CmpInst
  static CmpInst *Create(OtherOps Op,
                         unsigned short predicate, Value *S1,
                         Value *S2, const Twine &Name = "",
                         Instruction *InsertBefore = nullptr);

  /// Construct a compare instruction, given the opcode, the predicate and the
  /// two operands.  Also automatically insert this instruction to the end of
  /// the BasicBlock specified.
  /// @brief Create a CmpInst
  static CmpInst *Create(OtherOps Op, unsigned short predicate, Value *S1,
                         Value *S2, const Twine &Name, BasicBlock *InsertAtEnd);

  /// @brief Get the opcode casted to the right type
  OtherOps getOpcode() const {
    return static_cast<OtherOps>(Instruction::getOpcode());
  }

  /// @brief Return the predicate for this instruction.
  Predicate getPredicate() const {
    return Predicate(getSubclassDataFromInstruction());
  }

  /// @brief Set the predicate for this instruction to the specified value.
  void setPredicate(Predicate P) { setInstructionSubclassData(P); }

  static bool isFPPredicate(Predicate P) {
    return P >= FIRST_FCMP_PREDICATE && P <= LAST_FCMP_PREDICATE;
  }

  static bool isIntPredicate(Predicate P) {
    return P >= FIRST_ICMP_PREDICATE && P <= LAST_ICMP_PREDICATE;
  }

  bool isFPPredicate() const { return isFPPredicate(getPredicate()); }
  bool isIntPredicate() const { return isIntPredicate(getPredicate()); }


  /// For example, EQ -> NE, UGT -> ULE, SLT -> SGE,
  ///              OEQ -> UNE, UGT -> OLE, OLT -> UGE, etc.
  /// @returns the inverse predicate for the instruction's current predicate.
  /// @brief Return the inverse of the instruction's predicate.
  Predicate getInversePredicate() const {
    return getInversePredicate(getPredicate());
  }

  /// For example, EQ -> NE, UGT -> ULE, SLT -> SGE,
  ///              OEQ -> UNE, UGT -> OLE, OLT -> UGE, etc.
  /// @returns the inverse predicate for predicate provided in \p pred.
  /// @brief Return the inverse of a given predicate
  static Predicate getInversePredicate(Predicate pred);

  /// For example, EQ->EQ, SLE->SGE, ULT->UGT,
  ///              OEQ->OEQ, ULE->UGE, OLT->OGT, etc.
  /// @returns the predicate that would be the result of exchanging the two
  /// operands of the CmpInst instruction without changing the result
  /// produced.
  /// @brief Return the predicate as if the operands were swapped
  Predicate getSwappedPredicate() const {
    return getSwappedPredicate(getPredicate());
  }

  /// This is a static version that you can use without an instruction
  /// available.
  /// @brief Return the predicate as if the operands were swapped.
  static Predicate getSwappedPredicate(Predicate pred);

  /// @brief Provide more efficient getOperand methods.
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);

  /// This is just a convenience that dispatches to the subclasses.
  /// @brief Swap the operands and adjust predicate accordingly to retain
  /// the same comparison.
  void swapOperands();

  /// This is just a convenience that dispatches to the subclasses.
  /// @brief Determine if this CmpInst is commutative.
  bool isCommutative() const;

  /// This is just a convenience that dispatches to the subclasses.
  /// @brief Determine if this is an equals/not equals predicate.
  bool isEquality() const;

  /// @returns true if the comparison is signed, false otherwise.
  /// @brief Determine if this instruction is using a signed comparison.
  bool isSigned() const {
    return isSigned(getPredicate());
  }

  /// @returns true if the comparison is unsigned, false otherwise.
  /// @brief Determine if this instruction is using an unsigned comparison.
  bool isUnsigned() const {
    return isUnsigned(getPredicate());
  }

  /// This is just a convenience.
  /// @brief Determine if this is true when both operands are the same.
  bool isTrueWhenEqual() const {
    return isTrueWhenEqual(getPredicate());
  }

  /// This is just a convenience.
  /// @brief Determine if this is false when both operands are the same.
  bool isFalseWhenEqual() const {
    return isFalseWhenEqual(getPredicate());
  }

  /// @returns true if the predicate is unsigned, false otherwise.
  /// @brief Determine if the predicate is an unsigned operation.
  static bool isUnsigned(unsigned short predicate);

  /// @returns true if the predicate is signed, false otherwise.
  /// @brief Determine if the predicate is an signed operation.
  static bool isSigned(unsigned short predicate);

  /// @brief Determine if the predicate is an ordered operation.
  static bool isOrdered(unsigned short predicate);

  /// @brief Determine if the predicate is an unordered operation.
  static bool isUnordered(unsigned short predicate);

  /// Determine if the predicate is true when comparing a value with itself.
  static bool isTrueWhenEqual(unsigned short predicate);

  /// Determine if the predicate is false when comparing a value with itself.
  static bool isFalseWhenEqual(unsigned short predicate);

  /// @brief Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Instruction *I) {
    return I->getOpcode() == Instruction::ICmp ||
           I->getOpcode() == Instruction::FCmp;
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }

  /// @brief Create a result type for fcmp/icmp
  static Type* makeCmpResultType(Type* opnd_type) {
    if (VectorType* vt = dyn_cast<VectorType>(opnd_type)) {
      return VectorType::get(Type::getInt1Ty(opnd_type->getContext()),
                             vt->getNumElements());
    }
    return Type::getInt1Ty(opnd_type->getContext());
  }
private:
  // Shadow Value::setValueSubclassData with a private forwarding method so that
  // subclasses cannot accidentally use it.
  void setValueSubclassData(unsigned short D) {
    Value::setValueSubclassData(D);
  }
};


// FIXME: these are redundant if CmpInst < BinaryOperator
template <>
struct OperandTraits<CmpInst> : public FixedNumOperandTraits<CmpInst, 2> {
};

DEFINE_TRANSPARENT_OPERAND_ACCESSORS(CmpInst, Value)

} // End llvm namespace

#endif

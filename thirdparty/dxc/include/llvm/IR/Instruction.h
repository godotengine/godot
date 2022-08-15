//===-- llvm/Instruction.h - Instruction class definition -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the Instruction class, which is the
// base class for all of the LLVM instructions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_INSTRUCTION_H
#define LLVM_IR_INSTRUCTION_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/ilist_node.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/SymbolTableListTraits.h"
#include "llvm/IR/User.h"

namespace llvm {

class FastMathFlags;
class LLVMContext;
class MDNode;
class BasicBlock;
struct AAMDNodes;

template <>
struct ilist_traits<Instruction>
    : public SymbolTableListTraits<Instruction, BasicBlock> {

  /// \brief Return a node that marks the end of a list.
  ///
  /// The sentinel is relative to this instance, so we use a non-static
  /// method.
  Instruction *createSentinel() const;
  static void destroySentinel(Instruction *) {}

  Instruction *provideInitialHead() const { return createSentinel(); }
  Instruction *ensureHead(Instruction *) const { return createSentinel(); }
  static void noteHead(Instruction *, Instruction *) {}

private:
  mutable ilist_half_node<Instruction> Sentinel;
};

class Instruction : public User, public ilist_node<Instruction> {
  void operator=(const Instruction &) = delete;
  Instruction(const Instruction &) = delete;

  BasicBlock *Parent;
  DebugLoc DbgLoc;                         // 'dbg' Metadata cache.

  enum {
    /// HasMetadataBit - This is a bit stored in the SubClassData field which
    /// indicates whether this instruction has metadata attached to it or not.
    HasMetadataBit = 1 << 15
  };
public:
  // Out of line virtual method, so the vtable, etc has a home.
  ~Instruction() override;

  /// user_back - Specialize the methods defined in Value, as we know that an
  /// instruction can only be used by other instructions.
  Instruction       *user_back()       { return cast<Instruction>(*user_begin());}
  const Instruction *user_back() const { return cast<Instruction>(*user_begin());}

  inline const BasicBlock *getParent() const { return Parent; }
  inline       BasicBlock *getParent()       { return Parent; }

  /// \brief Return the module owning the function this instruction belongs to
  /// or nullptr it the function does not have a module.
  ///
  /// Note: this is undefined behavior if the instruction does not have a
  /// parent, or the parent basic block does not have a parent function.
  const Module *getModule() const;
  Module *getModule();

  /// removeFromParent - This method unlinks 'this' from the containing basic
  /// block, but does not delete it.
  ///
  void removeFromParent();

  /// eraseFromParent - This method unlinks 'this' from the containing basic
  /// block and deletes it.
  ///
  /// \returns an iterator pointing to the element after the erased one
  iplist<Instruction>::iterator eraseFromParent();

  /// Insert an unlinked instruction into a basic block immediately before
  /// the specified instruction.
  void insertBefore(Instruction *InsertPos);

  /// Insert an unlinked instruction into a basic block immediately after the
  /// specified instruction.
  void insertAfter(Instruction *InsertPos);

  /// moveBefore - Unlink this instruction from its current basic block and
  /// insert it into the basic block that MovePos lives in, right before
  /// MovePos.
  void moveBefore(Instruction *MovePos);

  //===--------------------------------------------------------------------===//
  // Subclass classification.
  //===--------------------------------------------------------------------===//

  /// getOpcode() returns a member of one of the enums like Instruction::Add.
  unsigned getOpcode() const { return getValueID() - InstructionVal; }

  const char *getOpcodeName() const { return getOpcodeName(getOpcode()); }
  bool isTerminator() const { return isTerminator(getOpcode()); }
  bool isBinaryOp() const { return isBinaryOp(getOpcode()); }
  bool isShift() { return isShift(getOpcode()); }
  bool isCast() const { return isCast(getOpcode()); }

  static const char* getOpcodeName(unsigned OpCode);

  static inline bool isTerminator(unsigned OpCode) {
    return OpCode >= TermOpsBegin && OpCode < TermOpsEnd;
  }

  static inline bool isBinaryOp(unsigned Opcode) {
    return Opcode >= BinaryOpsBegin && Opcode < BinaryOpsEnd;
  }

  /// @brief Determine if the Opcode is one of the shift instructions.
  static inline bool isShift(unsigned Opcode) {
    return Opcode >= Shl && Opcode <= AShr;
  }

  /// isLogicalShift - Return true if this is a logical shift left or a logical
  /// shift right.
  inline bool isLogicalShift() const {
    return getOpcode() == Shl || getOpcode() == LShr;
  }

  /// isArithmeticShift - Return true if this is an arithmetic shift right.
  inline bool isArithmeticShift() const {
    return getOpcode() == AShr;
  }

  /// @brief Determine if the OpCode is one of the CastInst instructions.
  static inline bool isCast(unsigned OpCode) {
    return OpCode >= CastOpsBegin && OpCode < CastOpsEnd;
  }

  //===--------------------------------------------------------------------===//
  // Metadata manipulation.
  //===--------------------------------------------------------------------===//

  /// hasMetadata() - Return true if this instruction has any metadata attached
  /// to it.
  bool hasMetadata() const { return DbgLoc || hasMetadataHashEntry(); }

  /// hasMetadataOtherThanDebugLoc - Return true if this instruction has
  /// metadata attached to it other than a debug location.
  bool hasMetadataOtherThanDebugLoc() const {
    return hasMetadataHashEntry();
  }

  /// getMetadata - Get the metadata of given kind attached to this Instruction.
  /// If the metadata is not found then return null.
  MDNode *getMetadata(unsigned KindID) const {
    if (!hasMetadata()) return nullptr;
    return getMetadataImpl(KindID);
  }

  /// getMetadata - Get the metadata of given kind attached to this Instruction.
  /// If the metadata is not found then return null.
  MDNode *getMetadata(StringRef Kind) const {
    if (!hasMetadata()) return nullptr;
    return getMetadataImpl(Kind);
  }

  /// getAllMetadata - Get all metadata attached to this Instruction.  The first
  /// element of each pair returned is the KindID, the second element is the
  /// metadata value.  This list is returned sorted by the KindID.
  void
  getAllMetadata(SmallVectorImpl<std::pair<unsigned, MDNode *>> &MDs) const {
    if (hasMetadata())
      getAllMetadataImpl(MDs);
  }

  /// getAllMetadataOtherThanDebugLoc - This does the same thing as
  /// getAllMetadata, except that it filters out the debug location.
  void getAllMetadataOtherThanDebugLoc(
      SmallVectorImpl<std::pair<unsigned, MDNode *>> &MDs) const {
    if (hasMetadataOtherThanDebugLoc())
      getAllMetadataOtherThanDebugLocImpl(MDs);
  }

  /// getAAMetadata - Fills the AAMDNodes structure with AA metadata from
  /// this instruction. When Merge is true, the existing AA metadata is
  /// merged with that from this instruction providing the most-general result.
  void getAAMetadata(AAMDNodes &N, bool Merge = false) const;

  /// setMetadata - Set the metadata of the specified kind to the specified
  /// node.  This updates/replaces metadata if already present, or removes it if
  /// Node is null.
  void setMetadata(unsigned KindID, MDNode *Node);
  void setMetadata(StringRef Kind, MDNode *Node);

  /// \brief Drop unknown metadata.
  /// Passes are required to drop metadata they don't understand. This is a
  /// convenience method for passes to do so.
  void dropUnknownMetadata(ArrayRef<unsigned> KnownIDs);
  void dropUnknownMetadata() {
    return dropUnknownMetadata(None);
  }
  void dropUnknownMetadata(unsigned ID1) {
    return dropUnknownMetadata(makeArrayRef(ID1));
  }
  void dropUnknownMetadata(unsigned ID1, unsigned ID2) {
    unsigned IDs[] = {ID1, ID2};
    return dropUnknownMetadata(IDs);
  }

  /// setAAMetadata - Sets the metadata on this instruction from the
  /// AAMDNodes structure.
  void setAAMetadata(const AAMDNodes &N);

  /// setDebugLoc - Set the debug location information for this instruction.
  void setDebugLoc(DebugLoc Loc) { DbgLoc = std::move(Loc); }

  /// getDebugLoc - Return the debug location for this node as a DebugLoc.
  const DebugLoc &getDebugLoc() const { return DbgLoc; }

  /// Set or clear the unsafe-algebra flag on this instruction, which must be an
  /// operator which supports this flag. See LangRef.html for the meaning of
  /// this flag.
  void setHasUnsafeAlgebra(bool B);

  /// Set or clear the no-nans flag on this instruction, which must be an
  /// operator which supports this flag. See LangRef.html for the meaning of
  /// this flag.
  void setHasNoNaNs(bool B);

  /// Set or clear the no-infs flag on this instruction, which must be an
  /// operator which supports this flag. See LangRef.html for the meaning of
  /// this flag.
  void setHasNoInfs(bool B);

  /// Set or clear the no-signed-zeros flag on this instruction, which must be
  /// an operator which supports this flag. See LangRef.html for the meaning of
  /// this flag.
  void setHasNoSignedZeros(bool B);

  /// Set or clear the allow-reciprocal flag on this instruction, which must be
  /// an operator which supports this flag. See LangRef.html for the meaning of
  /// this flag.
  void setHasAllowReciprocal(bool B);

  /// Convenience function for setting multiple fast-math flags on this
  /// instruction, which must be an operator which supports these flags. See
  /// LangRef.html for the meaning of these flags.
  void setFastMathFlags(FastMathFlags FMF);

  /// Convenience function for transferring all fast-math flag values to this
  /// instruction, which must be an operator which supports these flags. See
  /// LangRef.html for the meaning of these flags.
  void copyFastMathFlags(FastMathFlags FMF);

  /// Determine whether the unsafe-algebra flag is set.
  bool hasUnsafeAlgebra() const;

  /// Determine whether the no-NaNs flag is set.
  bool hasNoNaNs() const;

  /// Determine whether the no-infs flag is set.
  bool hasNoInfs() const;

  /// Determine whether the no-signed-zeros flag is set.
  bool hasNoSignedZeros() const;

  /// Determine whether the allow-reciprocal flag is set.
  bool hasAllowReciprocal() const;

  /// Convenience function for getting all the fast-math flags, which must be an
  /// operator which supports these flags. See LangRef.html for the meaning of
  /// these flags.
  FastMathFlags getFastMathFlags() const;

  /// Copy I's fast-math flags
  void copyFastMathFlags(const Instruction *I);

private:
  /// hasMetadataHashEntry - Return true if we have an entry in the on-the-side
  /// metadata hash.
  bool hasMetadataHashEntry() const {
    return (getSubclassDataFromValue() & HasMetadataBit) != 0;
  }

  // These are all implemented in Metadata.cpp.
  MDNode *getMetadataImpl(unsigned KindID) const;
  MDNode *getMetadataImpl(StringRef Kind) const;
  void
  getAllMetadataImpl(SmallVectorImpl<std::pair<unsigned, MDNode *>> &) const;
  void getAllMetadataOtherThanDebugLocImpl(
      SmallVectorImpl<std::pair<unsigned, MDNode *>> &) const;
  void clearMetadataHashEntries();
public:
  //===--------------------------------------------------------------------===//
  // Predicates and helper methods.
  //===--------------------------------------------------------------------===//


  /// isAssociative - Return true if the instruction is associative:
  ///
  ///   Associative operators satisfy:  x op (y op z) === (x op y) op z
  ///
  /// In LLVM, the Add, Mul, And, Or, and Xor operators are associative.
  ///
  bool isAssociative() const;
  static bool isAssociative(unsigned op);

  /// isCommutative - Return true if the instruction is commutative:
  ///
  ///   Commutative operators satisfy: (x op y) === (y op x)
  ///
  /// In LLVM, these are the associative operators, plus SetEQ and SetNE, when
  /// applied to any type.
  ///
  bool isCommutative() const { return isCommutative(getOpcode()); }
  static bool isCommutative(unsigned op);

  /// isIdempotent - Return true if the instruction is idempotent:
  ///
  ///   Idempotent operators satisfy:  x op x === x
  ///
  /// In LLVM, the And and Or operators are idempotent.
  ///
  bool isIdempotent() const { return isIdempotent(getOpcode()); }
  static bool isIdempotent(unsigned op);

  /// isNilpotent - Return true if the instruction is nilpotent:
  ///
  ///   Nilpotent operators satisfy:  x op x === Id,
  ///
  ///   where Id is the identity for the operator, i.e. a constant such that
  ///     x op Id === x and Id op x === x for all x.
  ///
  /// In LLVM, the Xor operator is nilpotent.
  ///
  bool isNilpotent() const { return isNilpotent(getOpcode()); }
  static bool isNilpotent(unsigned op);

  /// mayWriteToMemory - Return true if this instruction may modify memory.
  ///
  bool mayWriteToMemory() const;

  /// mayReadFromMemory - Return true if this instruction may read memory.
  ///
  bool mayReadFromMemory() const;

  /// mayReadOrWriteMemory - Return true if this instruction may read or
  /// write memory.
  ///
  bool mayReadOrWriteMemory() const {
    return mayReadFromMemory() || mayWriteToMemory();
  }

  /// isAtomic - Return true if this instruction has an
  /// AtomicOrdering of unordered or higher.
  ///
  bool isAtomic() const;

  /// mayThrow - Return true if this instruction may throw an exception.
  ///
  bool mayThrow() const;

  /// mayReturn - Return true if this is a function that may return.
  /// this is true for all normal instructions. The only exception
  /// is functions that are marked with the 'noreturn' attribute.
  ///
  bool mayReturn() const;

  /// mayHaveSideEffects - Return true if the instruction may have side effects.
  ///
  /// Note that this does not consider malloc and alloca to have side
  /// effects because the newly allocated memory is completely invisible to
  /// instructions which don't use the returned value.  For cases where this
  /// matters, isSafeToSpeculativelyExecute may be more appropriate.
  bool mayHaveSideEffects() const {
    return mayWriteToMemory() || mayThrow() || !mayReturn();
  }

  /// clone() - Create a copy of 'this' instruction that is identical in all
  /// ways except the following:
  ///   * The instruction has no parent
  ///   * The instruction has no name
  ///
  Instruction *clone() const;

  /// isIdenticalTo - Return true if the specified instruction is exactly
  /// identical to the current one.  This means that all operands match and any
  /// extra information (e.g. load is volatile) agree.
  bool isIdenticalTo(const Instruction *I) const;

  /// isIdenticalToWhenDefined - This is like isIdenticalTo, except that it
  /// ignores the SubclassOptionalData flags, which specify conditions
  /// under which the instruction's result is undefined.
  bool isIdenticalToWhenDefined(const Instruction *I) const;

  /// When checking for operation equivalence (using isSameOperationAs) it is
  /// sometimes useful to ignore certain attributes.
  enum OperationEquivalenceFlags {
    /// Check for equivalence ignoring load/store alignment.
    CompareIgnoringAlignment = 1<<0,
    /// Check for equivalence treating a type and a vector of that type
    /// as equivalent.
    CompareUsingScalarTypes = 1<<1
  };

  /// This function determines if the specified instruction executes the same
  /// operation as the current one. This means that the opcodes, type, operand
  /// types and any other factors affecting the operation must be the same. This
  /// is similar to isIdenticalTo except the operands themselves don't have to
  /// be identical.
  /// @returns true if the specified instruction is the same operation as
  /// the current one.
  /// @brief Determine if one instruction is the same operation as another.
  bool isSameOperationAs(const Instruction *I, unsigned flags = 0) const;

  /// isUsedOutsideOfBlock - Return true if there are any uses of this
  /// instruction in blocks other than the specified block.  Note that PHI nodes
  /// are considered to evaluate their operands in the corresponding predecessor
  /// block.
  bool isUsedOutsideOfBlock(const BasicBlock *BB) const;


  /// Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const Value *V) {
    return V->getValueID() >= Value::InstructionVal;
  }

  //----------------------------------------------------------------------
  // Exported enumerations.
  //
  enum TermOps {       // These terminate basic blocks
#define  FIRST_TERM_INST(N)             TermOpsBegin = N,
#define HANDLE_TERM_INST(N, OPC, CLASS) OPC = N,
#define   LAST_TERM_INST(N)             TermOpsEnd = N+1
#include "llvm/IR/Instruction.def"
  };

  enum BinaryOps {
#define  FIRST_BINARY_INST(N)             BinaryOpsBegin = N,
#define HANDLE_BINARY_INST(N, OPC, CLASS) OPC = N,
#define   LAST_BINARY_INST(N)             BinaryOpsEnd = N+1
#include "llvm/IR/Instruction.def"
  };

  enum MemoryOps {
#define  FIRST_MEMORY_INST(N)             MemoryOpsBegin = N,
#define HANDLE_MEMORY_INST(N, OPC, CLASS) OPC = N,
#define   LAST_MEMORY_INST(N)             MemoryOpsEnd = N+1
#include "llvm/IR/Instruction.def"
  };

  enum CastOps {
#define  FIRST_CAST_INST(N)             CastOpsBegin = N,
#define HANDLE_CAST_INST(N, OPC, CLASS) OPC = N,
#define   LAST_CAST_INST(N)             CastOpsEnd = N+1
#include "llvm/IR/Instruction.def"
  };

  enum OtherOps {
#define  FIRST_OTHER_INST(N)             OtherOpsBegin = N,
#define HANDLE_OTHER_INST(N, OPC, CLASS) OPC = N,
#define   LAST_OTHER_INST(N)             OtherOpsEnd = N+1
#include "llvm/IR/Instruction.def"
  };
private:
  // Shadow Value::setValueSubclassData with a private forwarding method so that
  // subclasses cannot accidentally use it.
  void setValueSubclassData(unsigned short D) {
    Value::setValueSubclassData(D);
  }
  unsigned short getSubclassDataFromValue() const {
    return Value::getSubclassDataFromValue();
  }

  void setHasMetadataHashEntry(bool V) {
    setValueSubclassData((getSubclassDataFromValue() & ~HasMetadataBit) |
                         (V ? HasMetadataBit : 0));
  }

  friend class SymbolTableListTraits<Instruction, BasicBlock>;
  void setParent(BasicBlock *P);
protected:
  // Instruction subclasses can stick up to 15 bits of stuff into the
  // SubclassData field of instruction with these members.

  // Verify that only the low 15 bits are used.
  void setInstructionSubclassData(unsigned short D) {
    assert((D & HasMetadataBit) == 0 && "Out of range value put into field");
    setValueSubclassData((getSubclassDataFromValue() & HasMetadataBit) | D);
  }

  unsigned getSubclassDataFromInstruction() const {
    return getSubclassDataFromValue() & ~HasMetadataBit;
  }

  Instruction(Type *Ty, unsigned iType, Use *Ops, unsigned NumOps,
              Instruction *InsertBefore = nullptr);
  Instruction(Type *Ty, unsigned iType, Use *Ops, unsigned NumOps,
              BasicBlock *InsertAtEnd);

private:
  /// Create a copy of this instruction.
  Instruction *cloneImpl() const;
};

inline Instruction *ilist_traits<Instruction>::createSentinel() const {
  // Since i(p)lists always publicly derive from their corresponding traits,
  // placing a data member in this class will augment the i(p)list.  But since
  // the NodeTy is expected to be publicly derive from ilist_node<NodeTy>,
  // there is a legal viable downcast from it to NodeTy. We use this trick to
  // superimpose an i(p)list with a "ghostly" NodeTy, which becomes the
  // sentinel. Dereferencing the sentinel is forbidden (save the
  // ilist_node<NodeTy>), so no one will ever notice the superposition.
  return static_cast<Instruction *>(&Sentinel);
}

// Instruction* is only 4-byte aligned.
template<>
class PointerLikeTypeTraits<Instruction*> {
  typedef Instruction* PT;
public:
  static inline void *getAsVoidPointer(PT P) { return P; }
  static inline PT getFromVoidPointer(void *P) {
    return static_cast<PT>(P);
  }
  enum { NumLowBitsAvailable = 2 };
};

} // End llvm namespace

#endif

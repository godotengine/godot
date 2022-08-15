//===-- llvm/BasicBlock.h - Represent a basic block in the VM ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the BasicBlock class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_BASICBLOCK_H
#define LLVM_IR_BASICBLOCK_H

#include "llvm/ADT/Twine.h"
#include "llvm/ADT/ilist.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/SymbolTableListTraits.h"
#include "llvm/Support/CBindingWrapping.h"
#include "llvm/Support/DataTypes.h"

namespace llvm {

class CallInst;
class LandingPadInst;
class TerminatorInst;
class LLVMContext;
class BlockAddress;
class Function;

// Traits for intrusive list of basic blocks...
template<> struct ilist_traits<BasicBlock>
  : public SymbolTableListTraits<BasicBlock, Function> {

  BasicBlock *createSentinel() const;
  static void destroySentinel(BasicBlock*) {}

  BasicBlock *provideInitialHead() const { return createSentinel(); }
  BasicBlock *ensureHead(BasicBlock*) const { return createSentinel(); }
  static void noteHead(BasicBlock*, BasicBlock*) {}

  static ValueSymbolTable *getSymTab(Function *ItemParent);
private:
  mutable ilist_half_node<BasicBlock> Sentinel;
};


/// \brief LLVM Basic Block Representation
///
/// This represents a single basic block in LLVM. A basic block is simply a
/// container of instructions that execute sequentially. Basic blocks are Values
/// because they are referenced by instructions such as branches and switch
/// tables. The type of a BasicBlock is "Type::LabelTy" because the basic block
/// represents a label to which a branch can jump.
///
/// A well formed basic block is formed of a list of non-terminating
/// instructions followed by a single TerminatorInst instruction.
/// TerminatorInst's may not occur in the middle of basic blocks, and must
/// terminate the blocks. The BasicBlock class allows malformed basic blocks to
/// occur because it may be useful in the intermediate stage of constructing or
/// modifying a program. However, the verifier will ensure that basic blocks
/// are "well formed".
class BasicBlock : public Value, // Basic blocks are data objects also
                   public ilist_node<BasicBlock> {
  friend class BlockAddress;
public:
  typedef iplist<Instruction> InstListType;
private:
  InstListType InstList;
  Function *Parent;

  void setParent(Function *parent);
  friend class SymbolTableListTraits<BasicBlock, Function>;

  BasicBlock(const BasicBlock &) = delete;
  void operator=(const BasicBlock &) = delete;

  /// \brief Constructor.
  ///
  /// If the function parameter is specified, the basic block is automatically
  /// inserted at either the end of the function (if InsertBefore is null), or
  /// before the specified basic block.
  explicit BasicBlock(LLVMContext &C, const Twine &Name = "",
                      Function *Parent = nullptr,
                      BasicBlock *InsertBefore = nullptr);
public:
  /// \brief Get the context in which this basic block lives.
  LLVMContext &getContext() const;

  /// Instruction iterators...
  typedef InstListType::iterator iterator;
  typedef InstListType::const_iterator const_iterator;
  typedef InstListType::reverse_iterator reverse_iterator;
  typedef InstListType::const_reverse_iterator const_reverse_iterator;

  /// \brief Creates a new BasicBlock.
  ///
  /// If the Parent parameter is specified, the basic block is automatically
  /// inserted at either the end of the function (if InsertBefore is 0), or
  /// before the specified basic block.
  static BasicBlock *Create(LLVMContext &Context, const Twine &Name = "",
                            Function *Parent = nullptr,
                            BasicBlock *InsertBefore = nullptr) {
    return new BasicBlock(Context, Name, Parent, InsertBefore);
  }
  ~BasicBlock() override;

  /// \brief Return the enclosing method, or null if none.
  const Function *getParent() const { return Parent; }
        Function *getParent()       { return Parent; }

  /// \brief Return the module owning the function this basic block belongs to,
  /// or nullptr it the function does not have a module.
  ///
  /// Note: this is undefined behavior if the block does not have a parent.
  const Module *getModule() const;
  Module *getModule();

  /// \brief Returns the terminator instruction if the block is well formed or
  /// null if the block is not well formed.
  TerminatorInst *getTerminator();
  const TerminatorInst *getTerminator() const;

  /// \brief Returns the call instruction marked 'musttail' prior to the
  /// terminating return instruction of this basic block, if such a call is
  /// present.  Otherwise, returns null.
  CallInst *getTerminatingMustTailCall();
  const CallInst *getTerminatingMustTailCall() const {
    return const_cast<BasicBlock *>(this)->getTerminatingMustTailCall();
  }

  /// \brief Returns a pointer to the first instruction in this block that is
  /// not a PHINode instruction.
  ///
  /// When adding instructions to the beginning of the basic block, they should
  /// be added before the returned value, not before the first instruction,
  /// which might be PHI. Returns 0 is there's no non-PHI instruction.
  Instruction* getFirstNonPHI();
  const Instruction* getFirstNonPHI() const {
    return const_cast<BasicBlock*>(this)->getFirstNonPHI();
  }

  /// \brief Returns a pointer to the first instruction in this block that is not
  /// a PHINode or a debug intrinsic.
  Instruction* getFirstNonPHIOrDbg();
  const Instruction* getFirstNonPHIOrDbg() const {
    return const_cast<BasicBlock*>(this)->getFirstNonPHIOrDbg();
  }

  /// \brief Returns a pointer to the first instruction in this block that is not
  /// a PHINode, a debug intrinsic, or a lifetime intrinsic.
  Instruction* getFirstNonPHIOrDbgOrLifetime();
  const Instruction* getFirstNonPHIOrDbgOrLifetime() const {
    return const_cast<BasicBlock*>(this)->getFirstNonPHIOrDbgOrLifetime();
  }

  /// \brief Returns an iterator to the first instruction in this block that is
  /// suitable for inserting a non-PHI instruction.
  ///
  /// In particular, it skips all PHIs and LandingPad instructions.
  iterator getFirstInsertionPt();
  const_iterator getFirstInsertionPt() const {
    return const_cast<BasicBlock*>(this)->getFirstInsertionPt();
  }

  /// \brief Unlink 'this' from the containing function, but do not delete it.
  void removeFromParent();

  /// \brief Unlink 'this' from the containing function and delete it.
  ///
  // \returns an iterator pointing to the element after the erased one.
  iplist<BasicBlock>::iterator eraseFromParent();

  /// \brief Unlink this basic block from its current function and insert it
  /// into the function that \p MovePos lives in, right before \p MovePos.
  void moveBefore(BasicBlock *MovePos);

  /// \brief Unlink this basic block from its current function and insert it
  /// right after \p MovePos in the function \p MovePos lives in.
  void moveAfter(BasicBlock *MovePos);

  /// \brief Insert unlinked basic block into a function.
  ///
  /// Inserts an unlinked basic block into \c Parent.  If \c InsertBefore is
  /// provided, inserts before that basic block, otherwise inserts at the end.
  ///
  /// \pre \a getParent() is \c nullptr.
  void insertInto(Function *Parent, BasicBlock *InsertBefore = nullptr);

  /// \brief Return the predecessor of this block if it has a single predecessor
  /// block. Otherwise return a null pointer.
  BasicBlock *getSinglePredecessor();
  const BasicBlock *getSinglePredecessor() const {
    return const_cast<BasicBlock*>(this)->getSinglePredecessor();
  }

  /// \brief Return the predecessor of this block if it has a unique predecessor
  /// block. Otherwise return a null pointer.
  ///
  /// Note that unique predecessor doesn't mean single edge, there can be
  /// multiple edges from the unique predecessor to this block (for example a
  /// switch statement with multiple cases having the same destination).
  BasicBlock *getUniquePredecessor();
  const BasicBlock *getUniquePredecessor() const {
    return const_cast<BasicBlock*>(this)->getUniquePredecessor();
  }

  /// \brief Return the successor of this block if it has a single successor.
  /// Otherwise return a null pointer.
  ///
  /// This method is analogous to getSinglePredecessor above.
  BasicBlock *getSingleSuccessor();
  const BasicBlock *getSingleSuccessor() const {
    return const_cast<BasicBlock*>(this)->getSingleSuccessor();
  }

  /// \brief Return the successor of this block if it has a unique successor.
  /// Otherwise return a null pointer.
  ///
  /// This method is analogous to getUniquePredecessor above.
  BasicBlock *getUniqueSuccessor();
  const BasicBlock *getUniqueSuccessor() const {
    return const_cast<BasicBlock*>(this)->getUniqueSuccessor();
  }

  //===--------------------------------------------------------------------===//
  /// Instruction iterator methods
  ///
  inline iterator                begin()       { return InstList.begin(); }
  inline const_iterator          begin() const { return InstList.begin(); }
  inline iterator                end  ()       { return InstList.end();   }
  inline const_iterator          end  () const { return InstList.end();   }

  inline reverse_iterator        rbegin()       { return InstList.rbegin(); }
  inline const_reverse_iterator  rbegin() const { return InstList.rbegin(); }
  inline reverse_iterator        rend  ()       { return InstList.rend();   }
  inline const_reverse_iterator  rend  () const { return InstList.rend();   }

  inline size_t                   size() const { return InstList.size();  }
  inline bool                    empty() const { return InstList.empty(); }
  inline const Instruction      &front() const { return InstList.front(); }
  inline       Instruction      &front()       { return InstList.front(); }
  inline const Instruction       &back() const { return InstList.back();  }
  inline       Instruction       &back()       { return InstList.back();  }

  size_t compute_size_no_dbg() const; // HLSL Change - Get the size of the block without the debug insts

  /// \brief Return the underlying instruction list container.
  ///
  /// Currently you need to access the underlying instruction list container
  /// directly if you want to modify it.
  const InstListType &getInstList() const { return InstList; }
        InstListType &getInstList()       { return InstList; }

  /// \brief Returns a pointer to a member of the instruction list.
  static iplist<Instruction> BasicBlock::*getSublistAccess(Instruction*) {
    return &BasicBlock::InstList;
  }

  /// \brief Returns a pointer to the symbol table if one exists.
  ValueSymbolTable *getValueSymbolTable();

  /// \brief Methods for support type inquiry through isa, cast, and dyn_cast.
  static inline bool classof(const Value *V) {
    return V->getValueID() == Value::BasicBlockVal;
  }

  /// \brief Cause all subinstructions to "let go" of all the references that
  /// said subinstructions are maintaining.
  ///
  /// This allows one to 'delete' a whole class at a time, even though there may
  /// be circular references... first all references are dropped, and all use
  /// counts go to zero.  Then everything is delete'd for real.  Note that no
  /// operations are valid on an object that has "dropped all references",
  /// except operator delete.
  void dropAllReferences();

  /// \brief Notify the BasicBlock that the predecessor \p Pred is no longer
  /// able to reach it.
  ///
  /// This is actually not used to update the Predecessor list, but is actually
  /// used to update the PHI nodes that reside in the block.  Note that this
  /// should be called while the predecessor still refers to this block.
  void removePredecessor(BasicBlock *Pred, bool DontDeleteUselessPHIs = false);

  /// \brief Split the basic block into two basic blocks at the specified
  /// instruction.
  ///
  /// Note that all instructions BEFORE the specified iterator stay as part of
  /// the original basic block, an unconditional branch is added to the original
  /// BB, and the rest of the instructions in the BB are moved to the new BB,
  /// including the old terminator.  The newly formed BasicBlock is returned.
  /// This function invalidates the specified iterator.
  ///
  /// Note that this only works on well formed basic blocks (must have a
  /// terminator), and 'I' must not be the end of instruction list (which would
  /// cause a degenerate basic block to be formed, having a terminator inside of
  /// the basic block).
  ///
  /// Also note that this doesn't preserve any passes. To split blocks while
  /// keeping loop information consistent, use the SplitBlock utility function.
  BasicBlock *splitBasicBlock(iterator I, const Twine &BBName = "");

  /// \brief Returns true if there are any uses of this basic block other than
  /// direct branches, switches, etc. to it.
  bool hasAddressTaken() const { return getSubclassDataFromValue() != 0; }

  /// \brief Update all phi nodes in this basic block's successors to refer to
  /// basic block \p New instead of to it.
  void replaceSuccessorsPhiUsesWith(BasicBlock *New);

  /// \brief Return true if this basic block is a landing pad.
  ///
  /// Being a ``landing pad'' means that the basic block is the destination of
  /// the 'unwind' edge of an invoke instruction.
  bool isLandingPad() const;

  /// \brief Return the landingpad instruction associated with the landing pad.
  LandingPadInst *getLandingPadInst();
  const LandingPadInst *getLandingPadInst() const;

private:
  /// \brief Increment the internal refcount of the number of BlockAddresses
  /// referencing this BasicBlock by \p Amt.
  ///
  /// This is almost always 0, sometimes one possibly, but almost never 2, and
  /// inconceivably 3 or more.
  void AdjustBlockAddressRefCount(int Amt) {
    setValueSubclassData(getSubclassDataFromValue()+Amt);
    assert((int)(signed char)getSubclassDataFromValue() >= 0 &&
           "Refcount wrap-around");
  }
  /// \brief Shadow Value::setValueSubclassData with a private forwarding method
  /// so that any future subclasses cannot accidentally use it.
  void setValueSubclassData(unsigned short D) {
    Value::setValueSubclassData(D);
  }
};

// createSentinel is used to get hold of the node that marks the end of the
// list... (same trick used here as in ilist_traits<Instruction>)
inline BasicBlock *ilist_traits<BasicBlock>::createSentinel() const {
    return static_cast<BasicBlock*>(&Sentinel);
}

// Create wrappers for C Binding types (see CBindingWrapping.h).
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(BasicBlock, LLVMBasicBlockRef)

} // End llvm namespace

#endif

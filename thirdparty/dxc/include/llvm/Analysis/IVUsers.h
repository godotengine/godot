//===- llvm/Analysis/IVUsers.h - Induction Variable Users -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements bookkeeping for "interesting" users of expressions
// computed from induction variables.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_IVUSERS_H
#define LLVM_ANALYSIS_IVUSERS_H

#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/ScalarEvolutionNormalization.h"
#include "llvm/IR/ValueHandle.h"

namespace llvm {

class AssumptionCache;
class DominatorTree;
class Instruction;
class Value;
class ScalarEvolution;
class SCEV;
class IVUsers;
class DataLayout;

/// IVStrideUse - Keep track of one use of a strided induction variable.
/// The Expr member keeps track of the expression, User is the actual user
/// instruction of the operand, and 'OperandValToReplace' is the operand of
/// the User that is the use.
class IVStrideUse : public CallbackVH, public ilist_node<IVStrideUse> {
  friend class IVUsers;
public:
  IVStrideUse(IVUsers *P, Instruction* U, Value *O)
    : CallbackVH(U), Parent(P), OperandValToReplace(O) {
  }

  /// getUser - Return the user instruction for this use.
  Instruction *getUser() const {
    return cast<Instruction>(getValPtr());
  }

  /// setUser - Assign a new user instruction for this use.
  void setUser(Instruction *NewUser) {
    setValPtr(NewUser);
  }

  /// getOperandValToReplace - Return the Value of the operand in the user
  /// instruction that this IVStrideUse is representing.
  Value *getOperandValToReplace() const {
    return OperandValToReplace;
  }

  /// setOperandValToReplace - Assign a new Value as the operand value
  /// to replace.
  void setOperandValToReplace(Value *Op) {
    OperandValToReplace = Op;
  }

  /// getPostIncLoops - Return the set of loops for which the expression has
  /// been adjusted to use post-inc mode.
  const PostIncLoopSet &getPostIncLoops() const {
    return PostIncLoops;
  }

  /// transformToPostInc - Transform the expression to post-inc form for the
  /// given loop.
  void transformToPostInc(const Loop *L);

private:
  /// Parent - a pointer to the IVUsers that owns this IVStrideUse.
  IVUsers *Parent;

  /// OperandValToReplace - The Value of the operand in the user instruction
  /// that this IVStrideUse is representing.
  WeakVH OperandValToReplace;

  /// PostIncLoops - The set of loops for which Expr has been adjusted to
  /// use post-inc mode. This corresponds with SCEVExpander's post-inc concept.
  PostIncLoopSet PostIncLoops;

  /// Deleted - Implementation of CallbackVH virtual function to
  /// receive notification when the User is deleted.
  void deleted() override;
};

template<> struct ilist_traits<IVStrideUse>
  : public ilist_default_traits<IVStrideUse> {
  // createSentinel is used to get hold of a node that marks the end of
  // the list...
  // The sentinel is relative to this instance, so we use a non-static
  // method.
  IVStrideUse *createSentinel() const {
    // since i(p)lists always publicly derive from the corresponding
    // traits, placing a data member in this class will augment i(p)list.
    // But since the NodeTy is expected to publicly derive from
    // ilist_node<NodeTy>, there is a legal viable downcast from it
    // to NodeTy. We use this trick to superpose i(p)list with a "ghostly"
    // NodeTy, which becomes the sentinel. Dereferencing the sentinel is
    // forbidden (save the ilist_node<NodeTy>) so no one will ever notice
    // the superposition.
    return static_cast<IVStrideUse*>(&Sentinel);
  }
  static void destroySentinel(IVStrideUse*) {}

  IVStrideUse *provideInitialHead() const { return createSentinel(); }
  IVStrideUse *ensureHead(IVStrideUse*) const { return createSentinel(); }
  static void noteHead(IVStrideUse*, IVStrideUse*) {}

private:
  mutable ilist_node<IVStrideUse> Sentinel;
};

class IVUsers : public LoopPass {
  friend class IVStrideUse;
  Loop *L;
  AssumptionCache *AC;
  LoopInfo *LI;
  DominatorTree *DT;
  ScalarEvolution *SE;
  SmallPtrSet<Instruction*, 16> Processed;

  /// IVUses - A list of all tracked IV uses of induction variable expressions
  /// we are interested in.
  ilist<IVStrideUse> IVUses;

  // Ephemeral values used by @llvm.assume in this function.
  SmallPtrSet<const Value *, 32> EphValues;

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  bool runOnLoop(Loop *L, LPPassManager &LPM) override;

  void releaseMemory() override;

public:
  static char ID; // Pass ID, replacement for typeid
  IVUsers();

  Loop *getLoop() const { return L; }

  /// AddUsersIfInteresting - Inspect the specified Instruction.  If it is a
  /// reducible SCEV, recursively add its users to the IVUsesByStride set and
  /// return true.  Otherwise, return false.
  bool AddUsersIfInteresting(Instruction *I);

  IVStrideUse &AddUser(Instruction *User, Value *Operand);

  /// getReplacementExpr - Return a SCEV expression which computes the
  /// value of the OperandValToReplace of the given IVStrideUse.
  const SCEV *getReplacementExpr(const IVStrideUse &IU) const;

  /// getExpr - Return the expression for the use.
  const SCEV *getExpr(const IVStrideUse &IU) const;

  const SCEV *getStride(const IVStrideUse &IU, const Loop *L) const;

  typedef ilist<IVStrideUse>::iterator iterator;
  typedef ilist<IVStrideUse>::const_iterator const_iterator;
  iterator begin() { return IVUses.begin(); }
  iterator end()   { return IVUses.end(); }
  const_iterator begin() const { return IVUses.begin(); }
  const_iterator end() const   { return IVUses.end(); }
  bool empty() const { return IVUses.empty(); }

  bool isIVUserOrOperand(Instruction *Inst) const {
    return Processed.count(Inst);
  }

  void print(raw_ostream &OS, const Module* = nullptr) const override;

  /// dump - This method is used for debugging.
  void dump() const;
protected:
  bool AddUsersImpl(Instruction *I, SmallPtrSetImpl<Loop*> &SimpleLoopNests);
};

Pass *createIVUsersPass();

}

#endif

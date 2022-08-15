//===- Dominators.h - Dominator Info Calculation ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the DominatorTree class, which provides fast and efficient
// dominance queries.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_DOMINATORS_H
#define LLVM_IR_DOMINATORS_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/GenericDomTree.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

namespace llvm {

// FIXME: Replace this brittle forward declaration with the include of the new
// PassManager.h when doing so doesn't break the PassManagerBuilder.
template <typename IRUnitT> class AnalysisManager;
class PreservedAnalyses;

extern template class DomTreeNodeBase<BasicBlock>;
extern template class DominatorTreeBase<BasicBlock>;

extern template void Calculate<Function, BasicBlock *>(
    DominatorTreeBase<GraphTraits<BasicBlock *>::NodeType> &DT, Function &F);
extern template void Calculate<Function, Inverse<BasicBlock *>>(
    DominatorTreeBase<GraphTraits<Inverse<BasicBlock *>>::NodeType> &DT,
    Function &F);

typedef DomTreeNodeBase<BasicBlock> DomTreeNode;

class BasicBlockEdge {
  const BasicBlock *Start;
  const BasicBlock *End;
public:
  BasicBlockEdge(const BasicBlock *Start_, const BasicBlock *End_) :
    Start(Start_), End(End_) { }
  const BasicBlock *getStart() const {
    return Start;
  }
  const BasicBlock *getEnd() const {
    return End;
  }
  bool isSingleEdge() const;
};

/// \brief Concrete subclass of DominatorTreeBase that is used to compute a
/// normal dominator tree.
class DominatorTree : public DominatorTreeBase<BasicBlock> {
public:
  typedef DominatorTreeBase<BasicBlock> Base;

  DominatorTree() : DominatorTreeBase<BasicBlock>(false) {}

  DominatorTree(DominatorTree &&Arg)
      : Base(std::move(static_cast<Base &>(Arg))) {}
  DominatorTree &operator=(DominatorTree &&RHS) {
    Base::operator=(std::move(static_cast<Base &>(RHS)));
    return *this;
  }

  /// \brief Returns *false* if the other dominator tree matches this dominator
  /// tree.
  inline bool compare(const DominatorTree &Other) const {
    const DomTreeNode *R = getRootNode();
    const DomTreeNode *OtherR = Other.getRootNode();

    if (!R || !OtherR || R->getBlock() != OtherR->getBlock())
      return true;

    if (Base::compare(Other))
      return true;

    return false;
  }

  // Ensure base-class overloads are visible.
  using Base::dominates;

  /// \brief Return true if Def dominates a use in User.
  ///
  /// This performs the special checks necessary if Def and User are in the same
  /// basic block. Note that Def doesn't dominate a use in Def itself!
  bool dominates(const Instruction *Def, const Use &U) const;
  bool dominates(const Instruction *Def, const Instruction *User) const;
  bool dominates(const Instruction *Def, const BasicBlock *BB) const;
  bool dominates(const BasicBlockEdge &BBE, const Use &U) const;
  bool dominates(const BasicBlockEdge &BBE, const BasicBlock *BB) const;

  // Ensure base class overloads are visible.
  using Base::isReachableFromEntry;

  /// \brief Provide an overload for a Use.
  bool isReachableFromEntry(const Use &U) const;

  /// \brief Verify the correctness of the domtree by re-computing it.
  ///
  /// This should only be used for debugging as it aborts the program if the
  /// verification fails.
  void verifyDomTree() const;
};

//===-------------------------------------
// DominatorTree GraphTraits specializations so the DominatorTree can be
// iterable by generic graph iterators.

template <> struct GraphTraits<DomTreeNode*> {
  typedef DomTreeNode NodeType;
  typedef NodeType::iterator  ChildIteratorType;

  static NodeType *getEntryNode(NodeType *N) {
    return N;
  }
  static inline ChildIteratorType child_begin(NodeType *N) {
    return N->begin();
  }
  static inline ChildIteratorType child_end(NodeType *N) {
    return N->end();
  }

  typedef df_iterator<DomTreeNode*> nodes_iterator;

  static nodes_iterator nodes_begin(DomTreeNode *N) {
    return df_begin(getEntryNode(N));
  }

  static nodes_iterator nodes_end(DomTreeNode *N) {
    return df_end(getEntryNode(N));
  }
};

template <> struct GraphTraits<DominatorTree*>
  : public GraphTraits<DomTreeNode*> {
  static NodeType *getEntryNode(DominatorTree *DT) {
    return DT->getRootNode();
  }

  static nodes_iterator nodes_begin(DominatorTree *N) {
    return df_begin(getEntryNode(N));
  }

  static nodes_iterator nodes_end(DominatorTree *N) {
    return df_end(getEntryNode(N));
  }
};

/// \brief Analysis pass which computes a \c DominatorTree.
class DominatorTreeAnalysis {
public:
  /// \brief Provide the result typedef for this analysis pass.
  typedef DominatorTree Result;

  /// \brief Opaque, unique identifier for this analysis pass.
  static void *ID() { return (void *)&PassID; }

  /// \brief Run the analysis pass over a function and produce a dominator tree.
  DominatorTree run(Function &F);

  /// \brief Provide access to a name for this pass for debugging purposes.
  static StringRef name() { return "DominatorTreeAnalysis"; }

private:
  static char PassID;
};

/// \brief Printer pass for the \c DominatorTree.
class DominatorTreePrinterPass {
  raw_ostream &OS;

public:
  explicit DominatorTreePrinterPass(raw_ostream &OS);
  PreservedAnalyses run(Function &F, AnalysisManager<Function> *AM);

  static StringRef name() { return "DominatorTreePrinterPass"; }
};

/// \brief Verifier pass for the \c DominatorTree.
struct DominatorTreeVerifierPass {
  PreservedAnalyses run(Function &F, AnalysisManager<Function> *AM);

  static StringRef name() { return "DominatorTreeVerifierPass"; }
};

/// \brief Legacy analysis pass which computes a \c DominatorTree.
class DominatorTreeWrapperPass : public FunctionPass {
  DominatorTree DT;

public:
  static char ID;

  DominatorTreeWrapperPass() : FunctionPass(ID) {
    initializeDominatorTreeWrapperPassPass(*PassRegistry::getPassRegistry());
  }

  DominatorTree &getDomTree() { return DT; }
  const DominatorTree &getDomTree() const { return DT; }

  bool runOnFunction(Function &F) override;

  void verifyAnalysis() const override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }

  void releaseMemory() override { DT.releaseMemory(); }

  void print(raw_ostream &OS, const Module *M = nullptr) const override;
};

} // End llvm namespace

#endif

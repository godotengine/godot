//===- LazyValueInfo.h - Value constraint analysis --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interface for lazy computation of value constraint
// information.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_LAZYVALUEINFO_H
#define LLVM_ANALYSIS_LAZYVALUEINFO_H

#include "llvm/Pass.h"

namespace llvm {
  class AssumptionCache;
  class Constant;
  class DataLayout;
  class DominatorTree;
  class Instruction;
  class TargetLibraryInfo;
  class Value;
  
/// This pass computes, caches, and vends lazy value constraint information.
class LazyValueInfo : public FunctionPass {
  AssumptionCache *AC;
  class TargetLibraryInfo *TLI;
  DominatorTree *DT;
  void *PImpl;
  LazyValueInfo(const LazyValueInfo&) = delete;
  void operator=(const LazyValueInfo&) = delete;
public:
  static char ID;
  LazyValueInfo() : FunctionPass(ID), PImpl(nullptr) {
    initializeLazyValueInfoPass(*PassRegistry::getPassRegistry());
  }
  ~LazyValueInfo() override { assert(!PImpl && "releaseMemory not called"); }

  /// This is used to return true/false/dunno results.
  enum Tristate {
    Unknown = -1, False = 0, True = 1
  };
  
  
  // Public query interface.
  
  /// Determine whether the specified value comparison with a constant is known
  /// to be true or false on the specified CFG edge.
  /// Pred is a CmpInst predicate.
  Tristate getPredicateOnEdge(unsigned Pred, Value *V, Constant *C,
                              BasicBlock *FromBB, BasicBlock *ToBB,
                              Instruction *CxtI = nullptr);
  
  /// Determine whether the specified value comparison with a constant is known
  /// to be true or false at the specified instruction
  /// (from an assume intrinsic). Pred is a CmpInst predicate.
  Tristate getPredicateAt(unsigned Pred, Value *V, Constant *C,
                          Instruction *CxtI);
 
  /// Determine whether the specified value is known to be a
  /// constant at the end of the specified block.  Return null if not.
  Constant *getConstant(Value *V, BasicBlock *BB, Instruction *CxtI = nullptr);

  /// Determine whether the specified value is known to be a
  /// constant on the specified edge.  Return null if not.
  Constant *getConstantOnEdge(Value *V, BasicBlock *FromBB, BasicBlock *ToBB,
                              Instruction *CxtI = nullptr);
  
  /// Inform the analysis cache that we have threaded an edge from
  /// PredBB to OldSucc to be from PredBB to NewSucc instead.
  void threadEdge(BasicBlock *PredBB, BasicBlock *OldSucc, BasicBlock *NewSucc);
  
  /// Inform the analysis cache that we have erased a block.
  void eraseBlock(BasicBlock *BB);
  
  // Implementation boilerplate.

  void getAnalysisUsage(AnalysisUsage &AU) const override;
  void releaseMemory() override;
  bool runOnFunction(Function &F) override;
};

}  // end namespace llvm

#endif


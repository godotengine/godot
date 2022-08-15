///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// ComputeViewIdSets.h                                                       //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Computes control dependence relation for a function.                      //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#pragma once
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"

#include <unordered_set>
#include <unordered_map>

namespace llvm {
  class Function;
  class raw_ostream;
}


namespace hlsl {

using BasicBlockSet = std::unordered_set<llvm::BasicBlock *>;
using PostDomRelationType = llvm::DominatorTreeBase<llvm::BasicBlock>;

class ControlDependence {
public:
  void Compute(llvm::Function *F, PostDomRelationType &PostDomRel);
  void Clear();
  const BasicBlockSet &GetCDBlocks(llvm::BasicBlock *pBB) const;
  void print(llvm::raw_ostream &OS);
  void dump();

private:
  using BasicBlockVector = std::vector<llvm::BasicBlock *>;
  using ControlDependenceType = std::unordered_map<llvm::BasicBlock *, BasicBlockSet>;

  llvm::Function *m_pFunc;
  ControlDependenceType m_ControlDependence;
  BasicBlockSet m_EmptyBBSet;

  llvm::BasicBlock *GetIPostDom(PostDomRelationType &PostDomRel, llvm::BasicBlock *pBB);
  void ComputeRevTopOrderRec(PostDomRelationType &PostDomRel, llvm::BasicBlock *pBB,
                             BasicBlockVector &RevTopOrder, BasicBlockSet &VisitedBBs);
};

} // end of hlsl namespace

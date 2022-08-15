//===- DxilRemoveUnstructuredLoopExits.h - Make unrolled loops structured ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include <unordered_set>

namespace llvm {
  class Loop;
  class LoopInfo;
  class DominatorTree;
  class BasicBlock;
}

namespace hlsl {

  // exclude_set is a list of *EXIT BLOCKS* to exclude (NOTE: not *exiting* blocks)
  bool RemoveUnstructuredLoopExits(llvm::Loop *L, llvm::LoopInfo *LI, llvm::DominatorTree *DT, std::unordered_set<llvm::BasicBlock *> *exclude_set = nullptr);
}


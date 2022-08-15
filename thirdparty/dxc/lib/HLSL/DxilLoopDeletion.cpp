//===- DxilLoopDeletion.cpp - Dead Loop Deletion Pass -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file run LoopDeletion SimplifyCFG and DCE more than once to make sure
// all unused loop can be removed. Use kMaxIteration to avoid dead loop.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/IR/Function.h"
#include "dxc/HLSL/DxilGenerationPass.h"
#include "llvm/IR/LegacyPassManager.h"

using namespace llvm;

namespace {
  class DxilLoopDeletion : public FunctionPass {
  public:
    static char ID; // Pass ID, replacement for typeid
    DxilLoopDeletion() : FunctionPass(ID) {
    }

    bool runOnFunction(Function &F) override;

  };
}

char DxilLoopDeletion::ID = 0;
INITIALIZE_PASS(DxilLoopDeletion, "dxil-loop-deletion",
                "Delete dead loops", false, false)

FunctionPass *llvm::createDxilLoopDeletionPass() { return new DxilLoopDeletion(); }

bool DxilLoopDeletion::runOnFunction(Function &F) {
  // Run loop simplify first to make sure loop invariant is moved so loop
  // deletion will not update the function if not delete.
  legacy::FunctionPassManager DeleteLoopPM(F.getParent());

  DeleteLoopPM.add(createLoopDeletionPass());
  bool bUpdated = false;

  legacy::FunctionPassManager SimplifyPM(F.getParent());
  SimplifyPM.add(createCFGSimplificationPass());
  SimplifyPM.add(createDeadCodeEliminationPass());
  SimplifyPM.add(createInstructionCombiningPass());

  const unsigned kMaxIteration = 3;
  unsigned i=0;
  while (i<kMaxIteration) {
    if (!DeleteLoopPM.run(F))
      break;

    SimplifyPM.run(F);
    i++;
    bUpdated = true;
  }

  return bUpdated;
}

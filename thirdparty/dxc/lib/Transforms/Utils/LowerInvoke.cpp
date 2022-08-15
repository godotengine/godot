//===- LowerInvoke.cpp - Eliminate Invoke instructions --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This transformation is designed for use by code generators which do not yet
// support stack unwinding.  This pass converts 'invoke' instructions to 'call'
// instructions, so that any exception-handling 'landingpad' blocks become dead
// code (which can be removed by running the '-simplifycfg' pass afterwards).
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
using namespace llvm;

#define DEBUG_TYPE "lowerinvoke"

STATISTIC(NumInvokes, "Number of invokes replaced");

namespace {
  class LowerInvoke : public FunctionPass {
  public:
    static char ID; // Pass identification, replacement for typeid
    explicit LowerInvoke() : FunctionPass(ID) {
      initializeLowerInvokePass(*PassRegistry::getPassRegistry());
    }
    bool runOnFunction(Function &F) override;
  };
}

char LowerInvoke::ID = 0;
INITIALIZE_PASS(LowerInvoke, "lowerinvoke",
                "Lower invoke and unwind, for unwindless code generators",
                false, false)

char &llvm::LowerInvokePassID = LowerInvoke::ID;

// Public Interface To the LowerInvoke pass.
FunctionPass *llvm::createLowerInvokePass() {
  return new LowerInvoke();
}

bool LowerInvoke::runOnFunction(Function &F) {
  bool Changed = false;
  for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB)
    if (InvokeInst *II = dyn_cast<InvokeInst>(BB->getTerminator())) {
      SmallVector<Value*,16> CallArgs(II->op_begin(), II->op_end() - 3);
      // Insert a normal call instruction...
      CallInst *NewCall = CallInst::Create(II->getCalledValue(),
                                           CallArgs, "", II);
      NewCall->takeName(II);
      NewCall->setCallingConv(II->getCallingConv());
      NewCall->setAttributes(II->getAttributes());
      NewCall->setDebugLoc(II->getDebugLoc());
      II->replaceAllUsesWith(NewCall);

      // Insert an unconditional branch to the normal destination.
      BranchInst::Create(II->getNormalDest(), II);

      // Remove any PHI node entries from the exception destination.
      II->getUnwindDest()->removePredecessor(BB);

      // Remove the invoke instruction now.
      BB->getInstList().erase(II);

      ++NumInvokes; Changed = true;
    }
  return Changed;
}

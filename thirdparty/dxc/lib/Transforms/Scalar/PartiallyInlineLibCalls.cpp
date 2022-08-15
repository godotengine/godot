//===--- PartiallyInlineLibCalls.cpp - Partially inline libcalls ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass tries to partially inline the fast path of well-known library
// functions, such as using square-root instructions for cases where sqrt()
// does not need to set errno.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

#define DEBUG_TYPE "partially-inline-libcalls"

namespace {
  class PartiallyInlineLibCalls : public FunctionPass {
  public:
    static char ID;

    PartiallyInlineLibCalls() :
      FunctionPass(ID) {
      initializePartiallyInlineLibCallsPass(*PassRegistry::getPassRegistry());
    }

    void getAnalysisUsage(AnalysisUsage &AU) const override;
    bool runOnFunction(Function &F) override;

  private:
    /// Optimize calls to sqrt.
    bool optimizeSQRT(CallInst *Call, Function *CalledFunc,
                      BasicBlock &CurrBB, Function::iterator &BB);
  };

  char PartiallyInlineLibCalls::ID = 0;
}

INITIALIZE_PASS(PartiallyInlineLibCalls, "partially-inline-libcalls",
                "Partially inline calls to library functions", false, false)

void PartiallyInlineLibCalls::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<TargetLibraryInfoWrapperPass>();
  AU.addRequired<TargetTransformInfoWrapperPass>();
  FunctionPass::getAnalysisUsage(AU);
}

bool PartiallyInlineLibCalls::runOnFunction(Function &F) {
  bool Changed = false;
  Function::iterator CurrBB;
  TargetLibraryInfo *TLI =
      &getAnalysis<TargetLibraryInfoWrapperPass>().getTLI();
  const TargetTransformInfo *TTI =
      &getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
  for (Function::iterator BB = F.begin(), BE = F.end(); BB != BE;) {
    CurrBB = BB++;

    for (BasicBlock::iterator II = CurrBB->begin(), IE = CurrBB->end();
         II != IE; ++II) {
      CallInst *Call = dyn_cast<CallInst>(&*II);
      Function *CalledFunc;

      if (!Call || !(CalledFunc = Call->getCalledFunction()))
        continue;

      // Skip if function either has local linkage or is not a known library
      // function.
      LibFunc::Func LibFunc;
      if (CalledFunc->hasLocalLinkage() || !CalledFunc->hasName() ||
          !TLI->getLibFunc(CalledFunc->getName(), LibFunc))
        continue;

      switch (LibFunc) {
      case LibFunc::sqrtf:
      case LibFunc::sqrt:
        if (TTI->haveFastSqrt(Call->getType()) &&
            optimizeSQRT(Call, CalledFunc, *CurrBB, BB))
          break;
        continue;
      default:
        continue;
      }

      Changed = true;
      break;
    }
  }

  return Changed;
}

bool PartiallyInlineLibCalls::optimizeSQRT(CallInst *Call,
                                           Function *CalledFunc,
                                           BasicBlock &CurrBB,
                                           Function::iterator &BB) {
  // There is no need to change the IR, since backend will emit sqrt
  // instruction if the call has already been marked read-only.
  if (Call->onlyReadsMemory())
    return false;

  // The call must have the expected result type.
  if (!Call->getType()->isFloatingPointTy())
    return false;

  // Do the following transformation:
  //
  // (before)
  // dst = sqrt(src)
  //
  // (after)
  // v0 = sqrt_noreadmem(src) # native sqrt instruction.
  // if (v0 is a NaN)
  //   v1 = sqrt(src)         # library call.
  // dst = phi(v0, v1)
  //

  // Move all instructions following Call to newly created block JoinBB.
  // Create phi and replace all uses.
  BasicBlock *JoinBB = llvm::SplitBlock(&CurrBB, Call->getNextNode());
  IRBuilder<> Builder(JoinBB, JoinBB->begin());
  PHINode *Phi = Builder.CreatePHI(Call->getType(), 2);
  Call->replaceAllUsesWith(Phi);

  // Create basic block LibCallBB and insert a call to library function sqrt.
  BasicBlock *LibCallBB = BasicBlock::Create(CurrBB.getContext(), "call.sqrt",
                                             CurrBB.getParent(), JoinBB);
  Builder.SetInsertPoint(LibCallBB);
  Instruction *LibCall = Call->clone();
  Builder.Insert(LibCall);
  Builder.CreateBr(JoinBB);

  // Add attribute "readnone" so that backend can use a native sqrt instruction
  // for this call. Insert a FP compare instruction and a conditional branch
  // at the end of CurrBB.
  Call->addAttribute(AttributeSet::FunctionIndex, Attribute::ReadNone);
  CurrBB.getTerminator()->eraseFromParent();
  Builder.SetInsertPoint(&CurrBB);
  Value *FCmp = Builder.CreateFCmpOEQ(Call, Call);
  Builder.CreateCondBr(FCmp, JoinBB, LibCallBB);

  // Add phi operands.
  Phi->addIncoming(Call, &CurrBB);
  Phi->addIncoming(LibCall, LibCallBB);

  BB = JoinBB;
  return true;
}

FunctionPass *llvm::createPartiallyInlineLibCallsPass() {
  return new PartiallyInlineLibCalls();
}

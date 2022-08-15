//===- NoAliasAnalysis.cpp - Minimal Alias Analysis Impl ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the default implementation of the Alias Analysis interface
// that simply returns "I don't know" for all queries.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
using namespace llvm;

namespace {
  /// NoAA - This class implements the -no-aa pass, which always returns "I
  /// don't know" for alias queries.  NoAA is unlike other alias analysis
  /// implementations, in that it does not chain to a previous analysis.  As
  /// such it doesn't follow many of the rules that other alias analyses must.
  ///
  struct NoAA : public ImmutablePass, public AliasAnalysis {
    static char ID; // Class identification, replacement for typeinfo
    NoAA() : ImmutablePass(ID) {
      initializeNoAAPass(*PassRegistry::getPassRegistry());
    }

    void getAnalysisUsage(AnalysisUsage &AU) const override {}

    bool doInitialization(Module &M) override {
      // Note: NoAA does not call InitializeAliasAnalysis because it's
      // special and does not support chaining.
      DL = &M.getDataLayout();
      return true;
    }

    AliasResult alias(const MemoryLocation &LocA,
                      const MemoryLocation &LocB) override {
      return MayAlias;
    }

    ModRefBehavior getModRefBehavior(ImmutableCallSite CS) override {
      return UnknownModRefBehavior;
    }
    ModRefBehavior getModRefBehavior(const Function *F) override {
      return UnknownModRefBehavior;
    }

    bool pointsToConstantMemory(const MemoryLocation &Loc,
                                bool OrLocal) override {
      return false;
    }
    ModRefResult getArgModRefInfo(ImmutableCallSite CS,
                                  unsigned ArgIdx) override {
      return ModRef;
    }

    ModRefResult getModRefInfo(ImmutableCallSite CS,
                               const MemoryLocation &Loc) override {
      return ModRef;
    }
    ModRefResult getModRefInfo(ImmutableCallSite CS1,
                               ImmutableCallSite CS2) override {
      return ModRef;
    }

    void deleteValue(Value *V) override {}
    void addEscapingUse(Use &U) override {}

    /// getAdjustedAnalysisPointer - This method is used when a pass implements
    /// an analysis interface through multiple inheritance.  If needed, it
    /// should override this to adjust the this pointer as needed for the
    /// specified pass info.
    void *getAdjustedAnalysisPointer(const void *ID) override {
      if (ID == &AliasAnalysis::ID)
        return (AliasAnalysis*)this;
      return this;
    }
  };
}  // End of anonymous namespace

// Register this pass...
char NoAA::ID = 0;
INITIALIZE_AG_PASS(NoAA, AliasAnalysis, "no-aa",
                   "No Alias Analysis (always returns 'may' alias)",
                   true, true, true)

ImmutablePass *llvm::createNoAAPass() { return new NoAA(); }

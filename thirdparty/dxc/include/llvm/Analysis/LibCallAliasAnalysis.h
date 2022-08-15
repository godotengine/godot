//===- LibCallAliasAnalysis.h - Implement AliasAnalysis for libcalls ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the LibCallAliasAnalysis class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_LIBCALLALIASANALYSIS_H
#define LLVM_ANALYSIS_LIBCALLALIASANALYSIS_H

#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"

namespace llvm {
  class LibCallInfo;
  struct LibCallFunctionInfo;
  
  /// LibCallAliasAnalysis - Alias analysis driven from LibCallInfo.
  struct LibCallAliasAnalysis : public FunctionPass, public AliasAnalysis {
    static char ID; // Class identification
    
    LibCallInfo *LCI;
    
    explicit LibCallAliasAnalysis(LibCallInfo *LC = nullptr)
        : FunctionPass(ID), LCI(LC) {
      initializeLibCallAliasAnalysisPass(*PassRegistry::getPassRegistry());
    }
    explicit LibCallAliasAnalysis(char &ID, LibCallInfo *LC)
        : FunctionPass(ID), LCI(LC) {
      initializeLibCallAliasAnalysisPass(*PassRegistry::getPassRegistry());
    }
    ~LibCallAliasAnalysis() override;

    ModRefResult getModRefInfo(ImmutableCallSite CS,
                               const MemoryLocation &Loc) override;

    ModRefResult getModRefInfo(ImmutableCallSite CS1,
                               ImmutableCallSite CS2) override {
      // TODO: Could compare two direct calls against each other if we cared to.
      return AliasAnalysis::getModRefInfo(CS1, CS2);
    }

    void getAnalysisUsage(AnalysisUsage &AU) const override;

    bool runOnFunction(Function &F) override;

    /// getAdjustedAnalysisPointer - This method is used when a pass implements
    /// an analysis interface through multiple inheritance.  If needed, it
    /// should override this to adjust the this pointer as needed for the
    /// specified pass info.
    void *getAdjustedAnalysisPointer(const void *PI) override {
      if (PI == &AliasAnalysis::ID)
        return (AliasAnalysis*)this;
      return this;
    }
    
  private:
    ModRefResult AnalyzeLibCallDetails(const LibCallFunctionInfo *FI,
                                       ImmutableCallSite CS,
                                       const MemoryLocation &Loc);
  };
}  // End of llvm namespace

#endif

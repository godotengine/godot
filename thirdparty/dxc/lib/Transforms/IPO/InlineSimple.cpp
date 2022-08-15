//===- InlineSimple.cpp - Code to perform simple function inlining --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements bottom-up inlining of functions into callees.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/InlineCost.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Transforms/IPO/InlinerPass.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "inline"

namespace {

/// \brief Actual inliner pass implementation.
///
/// The common implementation of the inlining logic is shared between this
/// inliner pass and the always inliner pass. The two passes use different cost
/// analyses to determine when to inline.
class SimpleInliner : public Inliner {
  InlineCostAnalysis *ICA;

public:
  SimpleInliner() : Inliner(ID), ICA(nullptr) {
    initializeSimpleInlinerPass(*PassRegistry::getPassRegistry());
  }

  SimpleInliner(int Threshold)
      : Inliner(ID, Threshold, /*InsertLifetime*/ true), ICA(nullptr) {
    initializeSimpleInlinerPass(*PassRegistry::getPassRegistry());
  }

  static char ID; // Pass identification, replacement for typeid

  InlineCost getInlineCost(CallSite CS) override {
    return ICA->getInlineCost(CS, getInlineThreshold(CS));
  }

  bool runOnSCC(CallGraphSCC &SCC) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;
};

static int computeThresholdFromOptLevels(unsigned OptLevel,
                                         unsigned SizeOptLevel) {
  if (OptLevel > 2)
    return 275;
  if (SizeOptLevel == 1) // -Os
    return 75;
  if (SizeOptLevel == 2) // -Oz
    return 25;
  return 225;
}

} // end anonymous namespace

char SimpleInliner::ID = 0;
INITIALIZE_PASS_BEGIN(SimpleInliner, "inline",
                "Function Integration/Inlining", false, false)
INITIALIZE_AG_DEPENDENCY(AliasAnalysis)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(CallGraphWrapperPass)
INITIALIZE_PASS_DEPENDENCY(InlineCostAnalysis)
INITIALIZE_PASS_END(SimpleInliner, "inline",
                "Function Integration/Inlining", false, false)

Pass *llvm::createFunctionInliningPass() { return new SimpleInliner(); }

Pass *llvm::createFunctionInliningPass(int Threshold) {
  return new SimpleInliner(Threshold);
}

Pass *llvm::createFunctionInliningPass(unsigned OptLevel,
                                       unsigned SizeOptLevel) {
  return new SimpleInliner(
      computeThresholdFromOptLevels(OptLevel, SizeOptLevel));
}

bool SimpleInliner::runOnSCC(CallGraphSCC &SCC) {
  ICA = &getAnalysis<InlineCostAnalysis>();
  return Inliner::runOnSCC(SCC);
}

void SimpleInliner::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<InlineCostAnalysis>();
  Inliner::getAnalysisUsage(AU);
}

// HLSL Change Starts
void Inliner::applyOptions(PassOptions O) {
  GetPassOptionUnsigned(O, "InlineThreshold", &InlineThreshold, InlineThreshold);
  GetPassOptionBool(O, "InsertLifetime", &InsertLifetime, InsertLifetime);
}
void Inliner::dumpConfig(raw_ostream &OS) {
  CallGraphSCCPass::dumpConfig(OS);
  OS << ",InlineThreshold=" << InlineThreshold;
  OS << ",InsertLifetime=" << (InsertLifetime ? 't' : 'f');
}
// HLSL Change Ends

//===- DominanceFrontier.cpp - Dominance Frontier Calculation -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/DominanceFrontier.h"
#include "llvm/Analysis/DominanceFrontierImpl.h"

using namespace llvm;

namespace llvm {
template class DominanceFrontierBase<BasicBlock>;
template class ForwardDominanceFrontierBase<BasicBlock>;
}

char DominanceFrontier::ID = 0;

INITIALIZE_PASS_BEGIN(DominanceFrontier, "domfrontier",
                "Dominance Frontier Construction", true, true)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(DominanceFrontier, "domfrontier",
                "Dominance Frontier Construction", true, true)

DominanceFrontier::DominanceFrontier()
  : FunctionPass(ID),
    Base() {
  initializeDominanceFrontierPass(*PassRegistry::getPassRegistry());
}

void DominanceFrontier::releaseMemory() {
  Base.releaseMemory();
}

bool DominanceFrontier::runOnFunction(Function &) {
  releaseMemory();
  Base.analyze(getAnalysis<DominatorTreeWrapperPass>().getDomTree());
  return false;
}

void DominanceFrontier::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<DominatorTreeWrapperPass>();
}

void DominanceFrontier::print(raw_ostream &OS, const Module *) const {
  Base.print(OS);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void DominanceFrontier::dump() const {
  print(dbgs());
}
#endif

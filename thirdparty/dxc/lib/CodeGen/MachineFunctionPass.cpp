//===-- MachineFunctionPass.cpp -------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the definitions of the MachineFunctionPass members.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/DominanceFrontier.h"
#include "llvm/Analysis/IVUsers.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemoryDependenceAnalysis.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/CodeGen/MachineFunctionAnalysis.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/StackProtector.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
using namespace llvm;

Pass *MachineFunctionPass::createPrinterPass(raw_ostream &O,
                                             const std::string &Banner) const {
  return createMachineFunctionPrinterPass(O, Banner);
}

bool MachineFunctionPass::runOnFunction(Function &F) {
  // Do not codegen any 'available_externally' functions at all, they have
  // definitions outside the translation unit.
  if (F.hasAvailableExternallyLinkage())
    return false;

  MachineFunction &MF = getAnalysis<MachineFunctionAnalysis>().getMF();
  return runOnMachineFunction(MF);
}

void MachineFunctionPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<MachineFunctionAnalysis>();
  AU.addPreserved<MachineFunctionAnalysis>();

  // MachineFunctionPass preserves all LLVM IR passes, but there's no
  // high-level way to express this. Instead, just list a bunch of
  // passes explicitly. This does not include setPreservesCFG,
  // because CodeGen overloads that to mean preserving the MachineBasicBlock
  // CFG in addition to the LLVM IR CFG.
  AU.addPreserved<AliasAnalysis>();
  AU.addPreserved<DominanceFrontier>();
  AU.addPreserved<DominatorTreeWrapperPass>();
  AU.addPreserved<IVUsers>();
  AU.addPreserved<LoopInfoWrapperPass>();
  AU.addPreserved<MemoryDependenceAnalysis>();
  AU.addPreserved<ScalarEvolution>();
  AU.addPreserved<StackProtector>();

  FunctionPass::getAnalysisUsage(AU);
}

//===-- MachineFunctionAnalysis.cpp ---------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the definitions of the MachineFunctionAnalysis members.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineFunctionAnalysis.h"
#include "llvm/CodeGen/GCMetadata.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineFunctionInitializer.h"
using namespace llvm;

char MachineFunctionAnalysis::ID = 0;

MachineFunctionAnalysis::MachineFunctionAnalysis(
    const TargetMachine &tm, MachineFunctionInitializer *MFInitializer)
    : FunctionPass(ID), TM(tm), MF(nullptr), MFInitializer(MFInitializer) {
  initializeMachineModuleInfoPass(*PassRegistry::getPassRegistry());
}

MachineFunctionAnalysis::~MachineFunctionAnalysis() {
  releaseMemory();
  assert(!MF && "MachineFunctionAnalysis left initialized!");
}

void MachineFunctionAnalysis::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<MachineModuleInfo>();
}

bool MachineFunctionAnalysis::doInitialization(Module &M) {
  MachineModuleInfo *MMI = getAnalysisIfAvailable<MachineModuleInfo>();
  assert(MMI && "MMI not around yet??");
  MMI->setModule(&M);
  NextFnNum = 0;
  return false;
}


bool MachineFunctionAnalysis::runOnFunction(Function &F) {
  assert(!MF && "MachineFunctionAnalysis already initialized!");
  MF = new MachineFunction(&F, TM, NextFnNum++,
                           getAnalysis<MachineModuleInfo>());
  if (MFInitializer)
    MFInitializer->initializeMachineFunction(*MF);
  return false;
}

void MachineFunctionAnalysis::releaseMemory() {
  delete MF;
  MF = nullptr;
}

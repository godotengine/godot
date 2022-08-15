//===-- IPO.cpp -----------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the common infrastructure (including C bindings) for 
// libLLVMIPO.a, which implements several transformations over the LLVM 
// intermediate representation.
//
//===----------------------------------------------------------------------===//

#include "llvm-c/Initialization.h"
#include "llvm-c/Transforms/IPO.h"
#include "llvm/InitializePasses.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/IPO.h"

using namespace llvm;

void llvm::initializeIPO(PassRegistry &Registry) {
#if 0  // HLSL Change Starts: Disable ArgPromotion
  initializeArgPromotionPass(Registry);
#endif // HLSL Change Ends
  initializeConstantMergePass(Registry);
  initializeDAEPass(Registry);
  initializeDAHPass(Registry);
  initializeFunctionAttrsPass(Registry);
  initializeGlobalDCEPass(Registry);
  initializeGlobalOptPass(Registry);
  initializeIPCPPass(Registry);
  initializeAlwaysInlinerPass(Registry);
  initializeSimpleInlinerPass(Registry);
  initializeInternalizePassPass(Registry);
  initializeLoopExtractorPass(Registry);
  initializeBlockExtractorPassPass(Registry);
  initializeSingleLoopExtractorPass(Registry);
  initializeLowerBitSetsPass(Registry);
  initializeMergeFunctionsPass(Registry);
  initializePartialInlinerPass(Registry);
  initializePruneEHPass(Registry);
  initializeStripDeadPrototypesPassPass(Registry);
  initializeStripSymbolsPass(Registry);
  initializeStripDebugDeclarePass(Registry);
  initializeStripDeadDebugInfoPass(Registry);
  initializeStripNonDebugSymbolsPass(Registry);
  initializeBarrierNoopPass(Registry);
  initializeEliminateAvailableExternallyPass(Registry);
}

void LLVMInitializeIPO(LLVMPassRegistryRef R) {
  initializeIPO(*unwrap(R));
}

#if 0  // HLSL Change Starts: Disable ArgPromotion
void LLVMAddArgumentPromotionPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createArgumentPromotionPass());
}
#endif // HLSL Change Ends

void LLVMAddConstantMergePass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createConstantMergePass());
}

void LLVMAddDeadArgEliminationPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createDeadArgEliminationPass());
}

void LLVMAddFunctionAttrsPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createFunctionAttrsPass());
}

void LLVMAddFunctionInliningPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createFunctionInliningPass());
}

void LLVMAddAlwaysInlinerPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(llvm::createAlwaysInlinerPass());
}

void LLVMAddGlobalDCEPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createGlobalDCEPass());
}

void LLVMAddGlobalOptimizerPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createGlobalOptimizerPass());
}

void LLVMAddIPConstantPropagationPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createIPConstantPropagationPass());
}

void LLVMAddPruneEHPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createPruneEHPass());
}

void LLVMAddIPSCCPPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createIPSCCPPass());
}

void LLVMAddInternalizePass(LLVMPassManagerRef PM, unsigned AllButMain) {
  std::vector<const char *> Export;
  if (AllButMain)
    Export.push_back("main");
  unwrap(PM)->add(createInternalizePass(Export));
}

void LLVMAddStripDeadPrototypesPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createStripDeadPrototypesPass());
}

void LLVMAddStripSymbolsPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createStripSymbolsPass());
}

//===-- Instrumentation.cpp - TransformUtils Infrastructure ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the common initialization infrastructure for the
// Instrumentation library.
//
//===----------------------------------------------------------------------===//

#include "llvm/InitializePasses.h"
#include "llvm-c/Initialization.h"
#include "llvm/PassRegistry.h"

using namespace llvm;

/// initializeInstrumentation - Initialize all passes in the TransformUtils
/// library.
void llvm::initializeInstrumentation(PassRegistry &Registry) {
  initializeAddressSanitizerPass(Registry);
  initializeAddressSanitizerModulePass(Registry);
  initializeBoundsCheckingPass(Registry);
  initializeGCOVProfilerPass(Registry);
  initializeInstrProfilingPass(Registry);
  initializeMemorySanitizerPass(Registry);
  initializeThreadSanitizerPass(Registry);
  initializeSanitizerCoverageModulePass(Registry);
  initializeDataFlowSanitizerPass(Registry);
  initializeSafeStackPass(Registry);
}

/// LLVMInitializeInstrumentation - C binding for
/// initializeInstrumentation.
void LLVMInitializeInstrumentation(LLVMPassRegistryRef R) {
  initializeInstrumentation(*unwrap(R));
}

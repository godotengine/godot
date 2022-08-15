//===-- Scalar.cpp --------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements common infrastructure for libLLVMScalarOpts.a, which
// implements several scalar transformations over the LLVM intermediate
// representation, including the C bindings for that library.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm-c/Initialization.h"
#include "llvm-c/Transforms/Scalar.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Verifier.h"
#include "llvm/InitializePasses.h"
#include "llvm/IR/LegacyPassManager.h"

using namespace llvm;

/// initializeScalarOptsPasses - Initialize all passes linked into the
/// ScalarOpts library.
void llvm::initializeScalarOpts(PassRegistry &Registry) {
  initializeADCEPass(Registry);
  initializeBDCEPass(Registry);
  initializeAlignmentFromAssumptionsPass(Registry);
  initializeSampleProfileLoaderPass(Registry);
  initializeConstantHoistingPass(Registry);
  initializeConstantPropagationPass(Registry);
  initializeCorrelatedValuePropagationPass(Registry);
  initializeDCEPass(Registry);
  initializeDeadInstEliminationPass(Registry);
  initializeScalarizerPass(Registry);
  initializeDSEPass(Registry);
  initializeGVNPass(Registry);
  initializeEarlyCSELegacyPassPass(Registry);
  initializeFlattenCFGPassPass(Registry);
  initializeInductiveRangeCheckEliminationPass(Registry);
  initializeIndVarSimplifyPass(Registry);
  initializeJumpThreadingPass(Registry);
  initializeLICMPass(Registry);
  initializeLoopDeletionPass(Registry);
  initializeLoopAccessAnalysisPass(Registry);
  initializeLoopInstSimplifyPass(Registry);
  initializeLoopInterchangePass(Registry);
  initializeLoopRotatePass(Registry);
  initializeLoopStrengthReducePass(Registry);
  initializeLoopRerollPass(Registry);
  initializeLoopUnrollPass(Registry);
  initializeLoopUnswitchPass(Registry);
  initializeLoopIdiomRecognizePass(Registry);
  initializeLowerAtomicPass(Registry);
  initializeLowerExpectIntrinsicPass(Registry);
  initializeMemCpyOptPass(Registry);
  initializeMergedLoadStoreMotionPass(Registry);
  initializeNaryReassociatePass(Registry);
  initializePartiallyInlineLibCallsPass(Registry);
  initializeReassociatePass(Registry);
  initializeRegToMemPass(Registry);
  initializeRegToMemHlslPass(Registry); // HLSL Change
  initializeRewriteStatepointsForGCPass(Registry);
  initializeSCCPPass(Registry);
  initializeIPSCCPPass(Registry);
  initializeSROAPass(Registry);
  initializeSROA_DTPass(Registry);
  initializeSROA_SSAUpPass(Registry);
  initializeCFGSimplifyPassPass(Registry);
  initializeStructurizeCFGPass(Registry);
  initializeSinkingPass(Registry);
  initializeTailCallElimPass(Registry);
  initializeSeparateConstOffsetFromGEPPass(Registry);
  initializeSpeculativeExecutionPass(Registry);
  initializeStraightLineStrengthReducePass(Registry);
  initializeLoadCombinePass(Registry);
  initializePlaceBackedgeSafepointsImplPass(Registry);
  initializePlaceSafepointsPass(Registry);
  initializeFloat2IntPass(Registry);
  initializeLoopDistributePass(Registry);
}

void LLVMInitializeScalarOpts(LLVMPassRegistryRef R) {
  initializeScalarOpts(*unwrap(R));
}

void LLVMAddAggressiveDCEPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createAggressiveDCEPass());
}

void LLVMAddBitTrackingDCEPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createBitTrackingDCEPass());
}

void LLVMAddAlignmentFromAssumptionsPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createAlignmentFromAssumptionsPass());
}

void LLVMAddCFGSimplificationPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createCFGSimplificationPass());
}

void LLVMAddDeadStoreEliminationPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createDeadStoreEliminationPass());
}

void LLVMAddScalarizerPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createScalarizerPass());
}

void LLVMAddGVNPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createGVNPass());
}

void LLVMAddMergedLoadStoreMotionPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createMergedLoadStoreMotionPass());
}

void LLVMAddIndVarSimplifyPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createIndVarSimplifyPass());
}

void LLVMAddInstructionCombiningPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createInstructionCombiningPass());
}

void LLVMAddJumpThreadingPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createJumpThreadingPass());
}

void LLVMAddLICMPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createLICMPass());
}

void LLVMAddLoopDeletionPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createLoopDeletionPass());
}

void LLVMAddLoopIdiomPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createLoopIdiomPass());
}

void LLVMAddLoopRotatePass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createLoopRotatePass());
}

void LLVMAddLoopRerollPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createLoopRerollPass());
}

void LLVMAddLoopUnrollPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createLoopUnrollPass());
}

void LLVMAddLoopUnswitchPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createLoopUnswitchPass());
}

void LLVMAddMemCpyOptPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createMemCpyOptPass());
}

void LLVMAddPartiallyInlineLibCallsPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createPartiallyInlineLibCallsPass());
}

void LLVMAddLowerSwitchPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createLowerSwitchPass());
}

void LLVMAddPromoteMemoryToRegisterPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createPromoteMemoryToRegisterPass());
}

void LLVMAddReassociatePass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createReassociatePass());
}

void LLVMAddSCCPPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createSCCPPass());
}

void LLVMAddScalarReplAggregatesPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createScalarReplAggregatesPass());
}

void LLVMAddScalarReplAggregatesPassSSA(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createScalarReplAggregatesPass(-1, false));
}

void LLVMAddScalarReplAggregatesPassWithThreshold(LLVMPassManagerRef PM,
                                                  int Threshold) {
  unwrap(PM)->add(createScalarReplAggregatesPass(Threshold));
}

void LLVMAddSimplifyLibCallsPass(LLVMPassManagerRef PM) {
  // NOTE: The simplify-libcalls pass has been removed.
}

void LLVMAddTailCallEliminationPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createTailCallEliminationPass());
}

void LLVMAddConstantPropagationPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createConstantPropagationPass());
}

void LLVMAddDemoteMemoryToRegisterPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createDemoteRegisterToMemoryPass());
}

// HLSL Change start
void LLVMAddDemoteMemoryToRegisterHlslPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createDemoteRegisterToMemoryHlslPass());
}
// HLSL Change end

void LLVMAddVerifierPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createVerifierPass());
}

void LLVMAddCorrelatedValuePropagationPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createCorrelatedValuePropagationPass());
}

void LLVMAddEarlyCSEPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createEarlyCSEPass());
}

void LLVMAddTypeBasedAliasAnalysisPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createTypeBasedAliasAnalysisPass());
}

void LLVMAddScopedNoAliasAAPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createScopedNoAliasAAPass());
}

void LLVMAddBasicAliasAnalysisPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createBasicAliasAnalysisPass());
}

void LLVMAddLowerExpectIntrinsicPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createLowerExpectIntrinsicPass());
}

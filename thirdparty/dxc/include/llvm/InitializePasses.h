//===- llvm/InitializePasses.h -------- Initialize All Passes ---*- C++ -*-===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations for the pass initialization routines
// for the entire LLVM project.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_INITIALIZEPASSES_H
#define LLVM_INITIALIZEPASSES_H

namespace llvm {

class PassRegistry;

/// initializeCore - Initialize all passes linked into the
/// TransformUtils library.
void initializeCore(PassRegistry&);

/// initializeTransformUtils - Initialize all passes linked into the
/// TransformUtils library.
void initializeTransformUtils(PassRegistry&);

/// initializeScalarOpts - Initialize all passes linked into the
/// ScalarOpts library.
void initializeScalarOpts(PassRegistry&);

/// initializeObjCARCOpts - Initialize all passes linked into the ObjCARCOpts
/// library.
void initializeObjCARCOpts(PassRegistry&);

/// initializeVectorization - Initialize all passes linked into the
/// Vectorize library.
void initializeVectorization(PassRegistry&);

/// initializeInstCombine - Initialize all passes linked into the
/// InstCombine library.
void initializeInstCombine(PassRegistry&);

/// initializeIPO - Initialize all passes linked into the IPO library.
void initializeIPO(PassRegistry&);

/// initializeInstrumentation - Initialize all passes linked into the
/// Instrumentation library.
void initializeInstrumentation(PassRegistry&);

/// initializeAnalysis - Initialize all passes linked into the Analysis library.
void initializeAnalysis(PassRegistry&);

/// initializeIPA - Initialize all passes linked into the IPA library.
void initializeIPA(PassRegistry&);

/// initializeCodeGen - Initialize all passes linked into the CodeGen library.
void initializeCodeGen(PassRegistry&);

/// initializeCodeGen - Initialize all passes linked into the CodeGen library.
void initializeTarget(PassRegistry&);

void initializeAAEvalPass(PassRegistry&);
void initializeAddDiscriminatorsPass(PassRegistry&);
void initializeADCEPass(PassRegistry&);
void initializeBDCEPass(PassRegistry&);
void initializeAliasAnalysisAnalysisGroup(PassRegistry&);
void initializeAliasAnalysisCounterPass(PassRegistry&);
void initializeAliasDebuggerPass(PassRegistry&);
void initializeAliasSetPrinterPass(PassRegistry&);
void initializeAlwaysInlinerPass(PassRegistry&);
void initializeArgPromotionPass(PassRegistry&);
void initializeAtomicExpandPass(PassRegistry&);
void initializeSampleProfileLoaderPass(PassRegistry&);
void initializeAlignmentFromAssumptionsPass(PassRegistry&);
void initializeBarrierNoopPass(PassRegistry&);
void initializeBasicAliasAnalysisPass(PassRegistry&);
void initializeCallGraphWrapperPassPass(PassRegistry &);
void initializeBlockExtractorPassPass(PassRegistry&);
void initializeBlockFrequencyInfoPass(PassRegistry&);
void initializeBoundsCheckingPass(PassRegistry&);
void initializeBranchFolderPassPass(PassRegistry&);
void initializeBranchProbabilityInfoPass(PassRegistry&);
void initializeBreakCriticalEdgesPass(PassRegistry&);
void initializeCallGraphPrinterPass(PassRegistry&);
void initializeCallGraphViewerPass(PassRegistry&);
void initializeCFGOnlyPrinterPass(PassRegistry&);
void initializeCFGOnlyViewerPass(PassRegistry&);
void initializeCFGPrinterPass(PassRegistry&);
void initializeCFGSimplifyPassPass(PassRegistry&);
void initializeCFLAliasAnalysisPass(PassRegistry&);
void initializeForwardControlFlowIntegrityPass(PassRegistry&);
void initializeFlattenCFGPassPass(PassRegistry&);
void initializeStructurizeCFGPass(PassRegistry&);
void initializeCFGViewerPass(PassRegistry&);
void initializeConstantHoistingPass(PassRegistry&);
void initializeCodeGenPreparePass(PassRegistry&);
void initializeConstantMergePass(PassRegistry&);
void initializeConstantPropagationPass(PassRegistry&);
void initializeMachineCopyPropagationPass(PassRegistry&);
void initializeCostModelAnalysisPass(PassRegistry&);
void initializeCorrelatedValuePropagationPass(PassRegistry&);
void initializeDAEPass(PassRegistry&);
void initializeDAHPass(PassRegistry&);
void initializeDCEPass(PassRegistry&);
void initializeDSEPass(PassRegistry&);
void initializeDeadInstEliminationPass(PassRegistry&);
void initializeDeadMachineInstructionElimPass(PassRegistry&);
void initializeDelinearizationPass(PassRegistry &);
void initializeDependenceAnalysisPass(PassRegistry&);
void initializeDivergenceAnalysisPass(PassRegistry&);
void initializeDomOnlyPrinterPass(PassRegistry&);
void initializeDomOnlyViewerPass(PassRegistry&);
void initializeDomPrinterPass(PassRegistry&);
void initializeDomViewerPass(PassRegistry&);
void initializeDominanceFrontierPass(PassRegistry&);
void initializeDominatorTreeWrapperPassPass(PassRegistry&);
void initializeEarlyIfConverterPass(PassRegistry&);
void initializeEdgeBundlesPass(PassRegistry&);
void initializeExpandPostRAPass(PassRegistry&);
void initializeGCOVProfilerPass(PassRegistry&);
void initializeInstrProfilingPass(PassRegistry&);
void initializeAddressSanitizerPass(PassRegistry&);
void initializeAddressSanitizerModulePass(PassRegistry&);
void initializeMemorySanitizerPass(PassRegistry&);
void initializeThreadSanitizerPass(PassRegistry&);
void initializeSanitizerCoverageModulePass(PassRegistry&);
void initializeDataFlowSanitizerPass(PassRegistry&);
void initializeScalarizerPass(PassRegistry&);
void initializeEarlyCSELegacyPassPass(PassRegistry &);
void initializeEliminateAvailableExternallyPass(PassRegistry&);
void initializeExpandISelPseudosPass(PassRegistry&);
void initializeFunctionAttrsPass(PassRegistry&);
void initializeGCMachineCodeAnalysisPass(PassRegistry&);
void initializeGCModuleInfoPass(PassRegistry&);
void initializeGVNPass(PassRegistry&);
void initializeGlobalDCEPass(PassRegistry&);
void initializeGlobalOptPass(PassRegistry&);
void initializeGlobalsModRefPass(PassRegistry&);
void initializeIPCPPass(PassRegistry&);
void initializeIPSCCPPass(PassRegistry&);
void initializeIVUsersPass(PassRegistry&);
void initializeIfConverterPass(PassRegistry&);
void initializeInductiveRangeCheckEliminationPass(PassRegistry&);
void initializeIndVarSimplifyPass(PassRegistry&);
void initializeInlineCostAnalysisPass(PassRegistry&);
void initializeInstructionCombiningPassPass(PassRegistry&);
void initializeInstCountPass(PassRegistry&);
void initializeInstNamerPass(PassRegistry&);
void initializeInternalizePassPass(PassRegistry&);
void initializeIntervalPartitionPass(PassRegistry&);
void initializeJumpThreadingPass(PassRegistry&);
void initializeLCSSAPass(PassRegistry&);
void initializeLICMPass(PassRegistry&);
void initializeLazyValueInfoPass(PassRegistry&);
void initializeLibCallAliasAnalysisPass(PassRegistry&);
void initializeLintPass(PassRegistry&);
void initializeLiveDebugVariablesPass(PassRegistry&);
void initializeLiveIntervalsPass(PassRegistry&);
void initializeLiveRegMatrixPass(PassRegistry&);
void initializeLiveStacksPass(PassRegistry&);
void initializeLiveVariablesPass(PassRegistry&);
void initializeLoaderPassPass(PassRegistry&);
void initializeLocalStackSlotPassPass(PassRegistry&);
void initializeLoopDeletionPass(PassRegistry&);
void initializeLoopExtractorPass(PassRegistry&);
void initializeLoopInfoWrapperPassPass(PassRegistry&);
void initializeLoopInterchangePass(PassRegistry &);
void initializeLoopInstSimplifyPass(PassRegistry&);
void initializeLoopRotatePass(PassRegistry&);
void initializeLoopSimplifyPass(PassRegistry&);
void initializeLoopStrengthReducePass(PassRegistry&);
void initializeGlobalMergePass(PassRegistry&);
void initializeLoopRerollPass(PassRegistry&);
void initializeLoopUnrollPass(PassRegistry&);
void initializeLoopUnswitchPass(PassRegistry&);
void initializeLoopIdiomRecognizePass(PassRegistry&);
void initializeLowerAtomicPass(PassRegistry&);
void initializeLowerBitSetsPass(PassRegistry&);
void initializeLowerExpectIntrinsicPass(PassRegistry&);
void initializeLowerIntrinsicsPass(PassRegistry&);
void initializeLowerInvokePass(PassRegistry&);
void initializeLowerSwitchPass(PassRegistry&);
void initializeMachineBlockFrequencyInfoPass(PassRegistry&);
void initializeMachineBlockPlacementPass(PassRegistry&);
void initializeMachineBlockPlacementStatsPass(PassRegistry&);
void initializeMachineBranchProbabilityInfoPass(PassRegistry&);
void initializeMachineCSEPass(PassRegistry&);
void initializeImplicitNullChecksPass(PassRegistry&);
void initializeMachineDominatorTreePass(PassRegistry&);
void initializeMachineDominanceFrontierPass(PassRegistry&);
void initializeMachinePostDominatorTreePass(PassRegistry&);
void initializeMachineLICMPass(PassRegistry&);
void initializeMachineLoopInfoPass(PassRegistry&);
void initializeMachineModuleInfoPass(PassRegistry&);
void initializeMachineRegionInfoPassPass(PassRegistry&);
void initializeMachineSchedulerPass(PassRegistry&);
void initializeMachineSinkingPass(PassRegistry&);
void initializeMachineTraceMetricsPass(PassRegistry&);
void initializeMachineVerifierPassPass(PassRegistry&);
void initializeMemCpyOptPass(PassRegistry&);
void initializeMemDepPrinterPass(PassRegistry&);
void initializeMemDerefPrinterPass(PassRegistry&);
void initializeMemoryDependenceAnalysisPass(PassRegistry&);
void initializeMergedLoadStoreMotionPass(PassRegistry &);
void initializeMetaRenamerPass(PassRegistry&);
void initializeMergeFunctionsPass(PassRegistry&);
void initializeModuleDebugInfoPrinterPass(PassRegistry&);
void initializeNaryReassociatePass(PassRegistry&);
void initializeNoAAPass(PassRegistry&);
void initializeObjCARCAliasAnalysisPass(PassRegistry&);
void initializeObjCARCAPElimPass(PassRegistry&);
void initializeObjCARCExpandPass(PassRegistry&);
void initializeObjCARCContractPass(PassRegistry&);
void initializeObjCARCOptPass(PassRegistry&);
void initializePAEvalPass(PassRegistry &);
void initializeOptimizePHIsPass(PassRegistry&);
void initializePartiallyInlineLibCallsPass(PassRegistry&);
void initializePEIPass(PassRegistry&);
void initializePHIEliminationPass(PassRegistry&);
void initializePartialInlinerPass(PassRegistry&);
void initializePeepholeOptimizerPass(PassRegistry&);
void initializePostDomOnlyPrinterPass(PassRegistry&);
void initializePostDomOnlyViewerPass(PassRegistry&);
void initializePostDomPrinterPass(PassRegistry&);
void initializePostDomViewerPass(PassRegistry&);
void initializePostDominatorTreePass(PassRegistry&);
void initializePostRASchedulerPass(PassRegistry&);
void initializePostMachineSchedulerPass(PassRegistry&);
void initializePrintFunctionPassWrapperPass(PassRegistry&);
void initializePrintModulePassWrapperPass(PassRegistry&);
void initializePrintBasicBlockPassPass(PassRegistry&);
void initializeProcessImplicitDefsPass(PassRegistry&);
void initializePromotePassPass(PassRegistry&);
void initializePruneEHPass(PassRegistry&);
void initializeReassociatePass(PassRegistry&);
void initializeRegToMemPass(PassRegistry&);
void initializeRegToMemHlslPass(PassRegistry&); // HLSL Change
void initializeRegionInfoPassPass(PassRegistry&);
void initializeRegionOnlyPrinterPass(PassRegistry&);
void initializeRegionOnlyViewerPass(PassRegistry&);
void initializeRegionPrinterPass(PassRegistry&);
void initializeRegionViewerPass(PassRegistry&);
void initializeRewriteStatepointsForGCPass(PassRegistry&);
void initializeSafeStackPass(PassRegistry&);
void initializeSCCPPass(PassRegistry&);
void initializeSROAPass(PassRegistry&);
void initializeSROA_DTPass(PassRegistry&);
void initializeSROA_SSAUpPass(PassRegistry&);
// HLSL Change Begins
void initializeHLExpandStoreIntrinsicsPass(PassRegistry&);
void initializeSROA_HLSLPass(PassRegistry&);
void initializeSROA_DT_HLSLPass(PassRegistry&);
void initializeSROA_Parameter_HLSLPass(PassRegistry&);
void initializeLowerStaticGlobalIntoAllocaPass(PassRegistry&);
void initializeDynamicIndexingVectorToArrayPass(PassRegistry&);
void initializeMultiDimArrayToOneDimArrayPass(PassRegistry&);
void initializeResourceToHandlePass(PassRegistry&);
void initializeSROA_SSAUp_HLSLPass(PassRegistry&);
void initializeHoistConstantArrayPass(PassRegistry&);
void initializeDxilLoopUnrollPass(PassRegistry&);
void initializeDxilInsertNoopsPass(PassRegistry&);
void initializeDxilFinalizeNoopsPass(PassRegistry&);
void initializeDxilEliminateVectorPass(PassRegistry&);
void initializeDxilConditionalMem2RegPass(PassRegistry&);
void initializeDxilFixConstArrayInitializerPass(PassRegistry&);
void initializeDxilInsertPreservesPass(PassRegistry&);
void initializeDxilFinalizePreservesPass(PassRegistry&);
void initializeDxilPreserveToSelectPass(PassRegistry&);
void initializeDxilRemoveDeadBlocksPass(PassRegistry&);
void initializeDxilRewriteOutputArgDebugInfoPass(PassRegistry&);
// HLSL Change Ends
void initializeScalarEvolutionAliasAnalysisPass(PassRegistry&);
void initializeScalarEvolutionPass(PassRegistry&);
void initializeShrinkWrapPass(PassRegistry &);
void initializeSimpleInlinerPass(PassRegistry&);
void initializeShadowStackGCLoweringPass(PassRegistry&);  
void initializeRegisterCoalescerPass(PassRegistry&);
void initializeSingleLoopExtractorPass(PassRegistry&);
void initializeSinkingPass(PassRegistry&);
void initializeSeparateConstOffsetFromGEPPass(PassRegistry &);
void initializeSlotIndexesPass(PassRegistry&);
void initializeSpillPlacementPass(PassRegistry&);
void initializeSpeculativeExecutionPass(PassRegistry&);
void initializeStackProtectorPass(PassRegistry&);
void initializeStackColoringPass(PassRegistry&);
void initializeStackSlotColoringPass(PassRegistry&);
void initializeStraightLineStrengthReducePass(PassRegistry &);
void initializeStripDeadDebugInfoPass(PassRegistry&);
void initializeStripDeadPrototypesPassPass(PassRegistry&);
void initializeStripDebugDeclarePass(PassRegistry&);
void initializeStripNonDebugSymbolsPass(PassRegistry&);
void initializeStripSymbolsPass(PassRegistry&);
void initializeTailCallElimPass(PassRegistry&);
void initializeTailDuplicatePassPass(PassRegistry&);
void initializeTargetPassConfigPass(PassRegistry&);
void initializeTargetTransformInfoWrapperPassPass(PassRegistry &);
void initializeTargetLibraryInfoWrapperPassPass(PassRegistry &);
void initializeAssumptionCacheTrackerPass(PassRegistry &);
void initializeTwoAddressInstructionPassPass(PassRegistry&);
void initializeTypeBasedAliasAnalysisPass(PassRegistry&);
void initializeScopedNoAliasAAPass(PassRegistry&);
void initializeUnifyFunctionExitNodesPass(PassRegistry&);
void initializeUnreachableBlockElimPass(PassRegistry&);
void initializeUnreachableMachineBlockElimPass(PassRegistry&);
void initializeVerifierLegacyPassPass(PassRegistry&);
void initializeVirtRegMapPass(PassRegistry&);
void initializeVirtRegRewriterPass(PassRegistry&);
void initializeInstSimplifierPass(PassRegistry&);
void initializeUnpackMachineBundlesPass(PassRegistry&);
void initializeFinalizeMachineBundlesPass(PassRegistry&);
void initializeLoopAccessAnalysisPass(PassRegistry&);
void initializeLoopVectorizePass(PassRegistry&);
void initializeSLPVectorizerPass(PassRegistry&);
void initializeBBVectorizePass(PassRegistry&);
void initializeMachineFunctionPrinterPassPass(PassRegistry&);
void initializeMIRPrintingPassPass(PassRegistry&);
void initializeStackMapLivenessPass(PassRegistry&);
void initializeMachineCombinerPass(PassRegistry &);
void initializeLoadCombinePass(PassRegistry&);
void initializeRewriteSymbolsPass(PassRegistry&);
void initializeWinEHPreparePass(PassRegistry&);
void initializePlaceBackedgeSafepointsImplPass(PassRegistry&);
void initializePlaceSafepointsPass(PassRegistry&);
void initializeDwarfEHPreparePass(PassRegistry&);
void initializeFloat2IntPass(PassRegistry&);
void initializeLoopDistributePass(PassRegistry&);
void initializeSjLjEHPreparePass(PassRegistry&);
}

#endif

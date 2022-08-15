//===- llvm/LinkAllPasses.h ------------ Reference All Passes ---*- C++ -*-===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This header file pulls in all transformation and analysis passes for tools
// like opt and bugpoint that need this functionality.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LINKALLPASSES_H
#define LLVM_LINKALLPASSES_H

#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AliasSetTracker.h"
#include "llvm/Analysis/CallPrinter.h"
#include "llvm/Analysis/DomPrinter.h"
#include "llvm/Analysis/IntervalPartition.h"
#include "llvm/Analysis/Lint.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/RegionPass.h"
#include "llvm/Analysis/RegionPrinter.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Transforms/ObjCARC.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/SymbolRewriter.h"
#include "llvm/Transforms/Utils/UnifyFunctionExitNodes.h"
#include "llvm/Transforms/Vectorize.h"
#include "llvm/Support/Valgrind.h"
#include <cstdlib>

namespace {
  struct ForcePassLinking {
    ForcePassLinking() {
      // We must reference the passes in such a way that compilers will not
      // delete it all as dead code, even with whole program optimization,
      // yet is effectively a NO-OP. As the compiler isn't smart enough
      // to know that getenv() never returns -1, this will do the job.
      if (std::getenv("bar") != (char*) -1)
        return;

      (void) llvm::createAAEvalPass();
      (void) llvm::createAggressiveDCEPass();
      (void) llvm::createBitTrackingDCEPass();
      (void) llvm::createAliasAnalysisCounterPass();
      (void) llvm::createAliasDebugger();
      // (void) llvm::createArgumentPromotionPass(); // HLSL Change - do not link
      (void) llvm::createAlignmentFromAssumptionsPass();
      (void) llvm::createBasicAliasAnalysisPass();
      (void) llvm::createLibCallAliasAnalysisPass(nullptr);
      (void) llvm::createScalarEvolutionAliasAnalysisPass();
      (void) llvm::createTypeBasedAliasAnalysisPass();
      (void) llvm::createScopedNoAliasAAPass();
      // (void) llvm::createBoundsCheckingPass(); // HLSL Change - do not link
      (void) llvm::createBreakCriticalEdgesPass();
      (void) llvm::createCallGraphPrinterPass();
      (void) llvm::createCallGraphViewerPass();
      (void) llvm::createCFGSimplificationPass();
      (void) llvm::createCFLAliasAnalysisPass();
      (void) llvm::createStructurizeCFGPass();
      (void) llvm::createConstantMergePass();
      (void) llvm::createConstantPropagationPass();
      (void) llvm::createCostModelAnalysisPass();
      (void) llvm::createDeadArgEliminationPass();
      (void) llvm::createDeadCodeEliminationPass();
      (void) llvm::createDeadInstEliminationPass();
      (void) llvm::createDeadStoreEliminationPass();
      (void) llvm::createDependenceAnalysisPass();
      (void) llvm::createDivergenceAnalysisPass();
      (void) llvm::createDomOnlyPrinterPass();
      (void) llvm::createDomPrinterPass();
      (void) llvm::createDomOnlyViewerPass();
      (void) llvm::createDomViewerPass();
      // (void) llvm::createGCOVProfilerPass(); // HLSL Change - do not link
      // (void) llvm::createInstrProfilingPass(); // HLSL Change - do not link
      (void) llvm::createFunctionInliningPass();
      (void) llvm::createAlwaysInlinerPass();
      (void) llvm::createGlobalDCEPass();
      (void) llvm::createGlobalOptimizerPass();
      (void) llvm::createGlobalsModRefPass();
      (void) llvm::createIPConstantPropagationPass();
      (void) llvm::createIPSCCPPass();
      (void) llvm::createInductiveRangeCheckEliminationPass();
      (void) llvm::createIndVarSimplifyPass();
      (void) llvm::createInstructionCombiningPass();
      (void) llvm::createInternalizePass();
      (void) llvm::createLCSSAPass();
      (void) llvm::createLICMPass();
      (void) llvm::createLazyValueInfoPass();
      (void) llvm::createLoopExtractorPass();
      (void) llvm::createLoopInterchangePass();
      (void) llvm::createLoopSimplifyPass();
      (void) llvm::createLoopStrengthReducePass();
      (void) llvm::createLoopRerollPass();
      (void) llvm::createLoopUnrollPass();
      (void) llvm::createLoopUnswitchPass();
      (void) llvm::createLoopIdiomPass();
      (void) llvm::createLoopRotatePass();
      (void) llvm::createLowerExpectIntrinsicPass();
      (void) llvm::createLowerInvokePass();
      (void) llvm::createLowerSwitchPass();
      (void) llvm::createNaryReassociatePass();
      (void) llvm::createNoAAPass();
      // (void) llvm::createObjCARCAliasAnalysisPass(); // HLSL Change - do not link
      // (void) llvm::createObjCARCAPElimPass(); // HLSL Change - do not link
      // (void) llvm::createObjCARCExpandPass(); // HLSL Change - do not link
      // (void) llvm::createObjCARCContractPass(); // HLSL Change - do not link
      // (void) llvm::createObjCARCOptPass(); // HLSL Change - do not link
      // (void) llvm::createPAEvalPass();// HLSL Change - do not link
      (void) llvm::createPromoteMemoryToRegisterPass();
      (void) llvm::createDemoteRegisterToMemoryPass();
      // (void) llvm::createPruneEHPass(); // HLSL Change - do not link
      (void) llvm::createPostDomOnlyPrinterPass();
      (void) llvm::createPostDomPrinterPass();
      (void) llvm::createPostDomOnlyViewerPass();
      (void) llvm::createPostDomViewerPass();
      (void) llvm::createReassociatePass();
      (void) llvm::createRegionInfoPass();
      (void) llvm::createRegionOnlyPrinterPass();
      (void) llvm::createRegionOnlyViewerPass();
      (void) llvm::createRegionPrinterPass();
      (void) llvm::createRegionViewerPass();
      (void) llvm::createSCCPPass();
      // (void) llvm::createSafeStackPass(); // HLSL Change - do not link
      (void) llvm::createScalarReplAggregatesPass();
      (void) llvm::createSingleLoopExtractorPass();
      (void) llvm::createStripSymbolsPass();
      (void) llvm::createStripNonDebugSymbolsPass();
      (void) llvm::createStripDeadDebugInfoPass();
      (void) llvm::createStripDeadPrototypesPass();
      (void) llvm::createTailCallEliminationPass();
      (void) llvm::createJumpThreadingPass();
      (void) llvm::createUnifyFunctionExitNodesPass();
      (void) llvm::createInstCountPass();
      (void) llvm::createConstantHoistingPass();
      // (void) llvm::createCodeGenPreparePass(); // HLSL Change - do not link
      (void) llvm::createEarlyCSEPass();
      (void) llvm::createMergedLoadStoreMotionPass();
      (void) llvm::createGVNPass();
      (void) llvm::createMemCpyOptPass();
      (void) llvm::createLoopDeletionPass();
      (void) llvm::createPostDomTree();
      (void) llvm::createInstructionNamerPass();
      (void) llvm::createMetaRenamerPass();
      (void) llvm::createFunctionAttrsPass();
      (void) llvm::createMergeFunctionsPass();
      // HLSL Change Begin - avoid casting nullptrs
      std::string buf;
      llvm::raw_string_ostream os(buf);
      (void) llvm::createPrintModulePass(os);
      (void) llvm::createPrintFunctionPass(os);
      (void) llvm::createPrintBasicBlockPass(os);
      // HLSL Change End - avoid casting nullptrs
      (void) llvm::createModuleDebugInfoPrinterPass();
      (void) llvm::createPartialInliningPass();
      (void) llvm::createLintPass();
      (void) llvm::createSinkingPass();
      (void) llvm::createLowerAtomicPass();
      (void) llvm::createCorrelatedValuePropagationPass();
      (void) llvm::createMemDepPrinter();
      (void) llvm::createInstructionSimplifierPass();
      // (void) llvm::createLoopVectorizePass(); // HLSL Change - do not link
      // (void) llvm::createSLPVectorizerPass(); // HLSL Change - do not link
      // (void) llvm::createBBVectorizePass(); // HLSL Change - do not link
      (void) llvm::createPartiallyInlineLibCallsPass();
      (void) llvm::createScalarizerPass();
      (void) llvm::createSeparateConstOffsetFromGEPPass();
      (void) llvm::createSpeculativeExecutionPass();
      (void) llvm::createRewriteSymbolsPass();
      (void) llvm::createStraightLineStrengthReducePass();
      (void) llvm::createMemDerefPrinter();
      (void) llvm::createFloat2IntPass();
      (void) llvm::createEliminateAvailableExternallyPass();

      (void)new llvm::IntervalPartition();
      (void)new llvm::ScalarEvolution();
      // HLSL Change Begin - avoid casting nullptrs
      llvm::Function::Create(nullptr, llvm::GlobalValue::ExternalLinkage)->viewCFGOnly();
      llvm::AliasAnalysis AA;
      llvm::AliasSetTracker X(AA);
      // HLSL Change End - avoid casting nullptrs
      X.add(nullptr, 0, llvm::AAMDNodes()); // for -print-alias-sets
      (void) llvm::AreStatisticsEnabled();
      (void) llvm::sys::RunningOnValgrind();
    }
  } ForcePassLinking; // Force link by creating a global definition.
}

#endif

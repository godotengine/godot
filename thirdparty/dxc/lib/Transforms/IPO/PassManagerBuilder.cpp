//===- PassManagerBuilder.cpp - Build Standard Pass -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the PassManagerBuilder class, which is used to set up a
// "standard" optimization sequence suitable for languages like C and C++.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm-c/Transforms/PassManagerBuilder.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Vectorize.h"
#include "dxc/HLSL/DxilGenerationPass.h" // HLSL Change
#include "dxc/HLSL/HLMatrixLowerPass.h" // HLSL Change
#include "dxc/HLSL/ComputeViewIdState.h" // HLSL Change
#include "llvm/Analysis/DxilValueCache.h" // HLSL Change

using namespace llvm;

#if HLSL_VECTORIZATION_ENABLED // HLSL Change - don't build vectorization passes

static cl::opt<bool>
RunLoopVectorization("vectorize-loops", cl::Hidden,
                     cl::desc("Run the Loop vectorization passes"));

static cl::opt<bool>
RunSLPVectorization("vectorize-slp", cl::Hidden,
                    cl::desc("Run the SLP vectorization passes"));

static cl::opt<bool>
RunBBVectorization("vectorize-slp-aggressive", cl::Hidden,
                    cl::desc("Run the BB vectorization passes"));

static cl::opt<bool>
UseGVNAfterVectorization("use-gvn-after-vectorization",
  cl::init(false), cl::Hidden,
  cl::desc("Run GVN instead of Early CSE after vectorization passes"));

static cl::opt<bool> ExtraVectorizerPasses(
    "extra-vectorizer-passes", cl::init(false), cl::Hidden,
    cl::desc("Run cleanup optimization passes after vectorization."));

static cl::opt<bool> UseNewSROA("use-new-sroa",
  cl::init(true), cl::Hidden,
  cl::desc("Enable the new, experimental SROA pass"));

static cl::opt<bool>
RunLoopRerolling("reroll-loops", cl::Hidden,
                 cl::desc("Run the loop rerolling pass"));

static cl::opt<bool>
RunFloat2Int("float-to-int", cl::Hidden, cl::init(true),
             cl::desc("Run the float2int (float demotion) pass"));

static cl::opt<bool> RunLoadCombine("combine-loads", cl::init(false),
                                    cl::Hidden,
                                    cl::desc("Run the load combining pass"));

static cl::opt<bool>
RunSLPAfterLoopVectorization("run-slp-after-loop-vectorization",
  cl::init(true), cl::Hidden,
  cl::desc("Run the SLP vectorizer (and BB vectorizer) after the Loop "
           "vectorizer instead of before"));

static cl::opt<bool> UseCFLAA("use-cfl-aa",
  cl::init(false), cl::Hidden,
  cl::desc("Enable the new, experimental CFL alias analysis"));

static cl::opt<bool>
EnableMLSM("mlsm", cl::init(true), cl::Hidden,
           cl::desc("Enable motion of merged load and store"));

static cl::opt<bool> EnableLoopInterchange(
    "enable-loopinterchange", cl::init(false), cl::Hidden,
    cl::desc("Enable the new, experimental LoopInterchange Pass"));

static cl::opt<bool> EnableLoopDistribute(
    "enable-loop-distribute", cl::init(false), cl::Hidden,
    cl::desc("Enable the new, experimental LoopDistribution Pass"));

#else

// Don't declare the 'false' counterparts - simply avoid altogether.

static const bool UseNewSROA = true;
static const bool RunLoopRerolling = false;
static const bool RunFloat2Int = true;
static const bool RunLoadCombine = false;
#if HLSL_VECTORIZATION_ENABLED // HLSL Change - don't build vectorization passes
static const bool RunSLPAfterLoopVectorization = true;
#endif // HLSL Change
static const bool UseCFLAA = false;
static const bool EnableMLSM = true;
static const bool EnableLoopInterchange = false;
static const bool EnableLoopDistribute = false;
#endif // HLSL Change - don't build vectorization passes

PassManagerBuilder::PassManagerBuilder() {
    OptLevel = 2;
    SizeLevel = 0;
    LibraryInfo = nullptr;
    Inliner = nullptr;
    DisableUnitAtATime = false;
    DisableUnrollLoops = false;
#if HLSL_VECTORIZATION_ENABLED // HLSL Change - don't build vectorization passes
    BBVectorize = RunBBVectorization;
    SLPVectorize = RunSLPVectorization;
    LoopVectorize = RunLoopVectorization;
#else
    BBVectorize = SLPVectorize = LoopVectorize = false;
#endif
    RerollLoops = RunLoopRerolling;
    LoadCombine = RunLoadCombine;
    DisableGVNLoadPRE = false;
    VerifyInput = false;
    VerifyOutput = false;
    MergeFunctions = false;
    PrepareForLTO = false;
}

PassManagerBuilder::~PassManagerBuilder() {
  delete LibraryInfo;
  delete Inliner;
}

#if 0 // HLSL Change Starts - no global extensions
/// Set of global extensions, automatically added as part of the standard set.
static ManagedStatic<SmallVector<std::pair<PassManagerBuilder::ExtensionPointTy,
   PassManagerBuilder::ExtensionFn>, 8> > GlobalExtensions;
#endif // HLSL Change Ends

#if 0 // HLSL Change Starts - no global extensions
void PassManagerBuilder::addGlobalExtension(
    PassManagerBuilder::ExtensionPointTy Ty,
    PassManagerBuilder::ExtensionFn Fn) {
  GlobalExtensions->push_back(std::make_pair(Ty, Fn));
}
#endif // HLSL Change Ends

void PassManagerBuilder::addExtension(ExtensionPointTy Ty, ExtensionFn Fn) {
  Extensions.push_back(std::make_pair(Ty, Fn));
}

void PassManagerBuilder::addExtensionsToPM(ExtensionPointTy ETy,
                                           legacy::PassManagerBase &PM) const {
#if 0 // HLSL Change Starts - no global extensions
  for (unsigned i = 0, e = GlobalExtensions->size(); i != e; ++i)
    if ((*GlobalExtensions)[i].first == ETy)
      (*GlobalExtensions)[i].second(*this, PM);
  for (unsigned i = 0, e = Extensions.size(); i != e; ++i)
    if (Extensions[i].first == ETy)
      Extensions[i].second(*this, PM);
#endif // HLSL Change Ends
}

void PassManagerBuilder::addInitialAliasAnalysisPasses(
    legacy::PassManagerBase &PM) const {
  // Add TypeBasedAliasAnalysis before BasicAliasAnalysis so that
  // BasicAliasAnalysis wins if they disagree. This is intended to help
  // support "obvious" type-punning idioms.
  if (UseCFLAA)
    PM.add(createCFLAliasAnalysisPass());
  PM.add(createTypeBasedAliasAnalysisPass());
  PM.add(createScopedNoAliasAAPass());
  PM.add(createBasicAliasAnalysisPass());
}

void PassManagerBuilder::populateFunctionPassManager(
    legacy::FunctionPassManager &FPM) {
  addExtensionsToPM(EP_EarlyAsPossible, FPM);

  // Add LibraryInfo if we have some.
  if (LibraryInfo)
    FPM.add(new TargetLibraryInfoWrapperPass(*LibraryInfo));

  if (OptLevel == 0) return;

  addInitialAliasAnalysisPasses(FPM);

  FPM.add(createCFGSimplificationPass());
  // HLSL Change - don't run SROA. 
  // HLSL uses special SROA added in addHLSLPasses.
  if (HLSLHighLevel) { // HLSL Change
  if (UseNewSROA)
    FPM.add(createSROAPass());
  else
    FPM.add(createScalarReplAggregatesPass());
  }
  // HLSL Change. FPM.add(createEarlyCSEPass());
  FPM.add(createLowerExpectIntrinsicPass());
}

// HLSL Change Starts
static void addHLSLPasses(bool HLSLHighLevel, unsigned OptLevel, bool OnlyWarnOnUnrollFail, bool StructurizeLoopExitsForUnroll, bool EnableLifetimeMarkers, hlsl::HLSLExtensionsCodegenHelper *ExtHelper, legacy::PassManagerBase &MPM) {

  // Don't do any lowering if we're targeting high-level.
  if (HLSLHighLevel) {
    MPM.add(createHLEmitMetadataPass());
    return;
  }

  MPM.add(createDxilCleanupAddrSpaceCastPass());

  MPM.add(createHLPreprocessPass());
  bool NoOpt = OptLevel == 0;
  if (!NoOpt) {
    MPM.add(createHLDeadFunctionEliminationPass());
  }

  // Do this before scalarrepl-param-hlsl for opportunities to move things
  // like resource arrays to alloca, allowing more likely memcpy replacement.
  MPM.add(createLowerStaticGlobalIntoAlloca());

  // Expand buffer store intrinsics before we SROA
  MPM.add(createHLExpandStoreIntrinsicsPass());

  // Split struct and array of parameter.
  MPM.add(createSROA_Parameter_HLSL());

  MPM.add(createHLMatrixLowerPass());
  // DCE should after SROA to remove unused element.
  MPM.add(createDeadCodeEliminationPass());
  MPM.add(createGlobalDCEPass());

  if (NoOpt) {
    // If not run mem2reg, try to promote allocas used by EvalOperations.
    // Do this before change vector to array.
    MPM.add(createDxilLegalizeEvalOperationsPass());
  }
  // This should go between matrix lower and dynamic indexing vector to array,
  // because matrix lower may create dynamically indexed global vectors,
  // which should become locals. If they are turned into arrays first,
  // this pass will ignore them as it only works on scalars and vectors.
  MPM.add(createLowerStaticGlobalIntoAlloca());

  // Change dynamic indexing vector to array.
  MPM.add(createDynamicIndexingVectorToArrayPass(false /* ReplaceAllVector */));

  // Rotate the loops before, mem2reg, since it messes up dbg.value's
  MPM.add(createLoopRotatePass());

  // mem2reg
  // Special Mem2Reg pass that skips precise marker.
  MPM.add(createDxilConditionalMem2RegPass(NoOpt));
  MPM.add(createDxilDeleteRedundantDebugValuesPass());

  // Remove unneeded dxbreak conditionals
  MPM.add(createCleanupDxBreakPass());

  if (!NoOpt) {
    MPM.add(createDxilConvergentMarkPass());
    // Clean up inefficiencies that can cause unnecessary live values related to
    // lifetime marker cleanup blocks. This is the earliest possible location
    // without interfering with HLSL-specific lowering.
    if (EnableLifetimeMarkers) {
      MPM.add(createSROAPass());
      MPM.add(createInstructionCombiningPass());
      MPM.add(createJumpThreadingPass());
    }
  }

  if (!NoOpt)
    MPM.add(createSimplifyInstPass());

  if (!NoOpt)
    MPM.add(createCFGSimplificationPass());

  MPM.add(createDxilPromoteLocalResources());
  MPM.add(createDxilPromoteStaticResources());

  // Verify no undef resource again after promotion
  MPM.add(createInvalidateUndefResourcesPass());

  MPM.add(createDxilGenerationPass(NoOpt, ExtHelper));

  // Propagate precise attribute.
  MPM.add(createDxilPrecisePropagatePass());

  if (!NoOpt)
    MPM.add(createSimplifyInstPass());

  // scalarize vector to scalar
  MPM.add(createScalarizerPass(!NoOpt /* AllowFolding */));

  // Remove vector instructions
  MPM.add(createDxilEliminateVectorPass());

  // Passes to handle [unroll]
  // Needs to happen after SROA since loop count may depend on
  // struct members.
  // Needs to happen before resources are lowered and before HL
  // module is gone.
  MPM.add(createDxilLoopUnrollPass(1024, OnlyWarnOnUnrollFail, StructurizeLoopExitsForUnroll));

  // Default unroll pass. This is purely for optimizing loops without
  // attributes.
  if (OptLevel > 2) {
    MPM.add(createLoopUnrollPass(-1, -1, -1, -1, StructurizeLoopExitsForUnroll));
  }

  if (!NoOpt)
    MPM.add(createSimplifyInstPass());

  if (!NoOpt)
    MPM.add(createCFGSimplificationPass());

  MPM.add(createDeadCodeEliminationPass());

  if (OptLevel > 0) {
    MPM.add(createDxilFixConstArrayInitializerPass());
  }
}
// HLSL Change Ends

void PassManagerBuilder::populateModulePassManager(
    legacy::PassManagerBase &MPM) {
  // If all optimizations are disabled, just run the always-inline pass and,
  // if enabled, the function merging pass.
  if (OptLevel == 0) {
    if (!HLSLHighLevel) {
      MPM.add(createHLEnsureMetadataPass()); // HLSL Change - rehydrate metadata from high-level codegen
    }

    MPM.add(createDxilRewriteOutputArgDebugInfoPass()); // Fix output argument types.

    if (!HLSLHighLevel)
      if (HLSLEnableDebugNops) MPM.add(createDxilInsertPreservesPass(HLSLAllowPreserveValues)); // HLSL Change - insert preserve instructions

    if (Inliner) {
      MPM.add(createHLLegalizeParameter()); // HLSL Change - legalize parameters
                                            // before inline.
      MPM.add(Inliner);
      Inliner = nullptr;
    }

    // FIXME: The BarrierNoopPass is a HACK! The inliner pass above implicitly
    // creates a CGSCC pass manager, but we don't want to add extensions into
    // that pass manager. To prevent this we insert a no-op module pass to reset
    // the pass manager to get the same behavior as EP_OptimizerLast in non-O0
    // builds. The function merging pass is 
    if (MergeFunctions)
      MPM.add(createMergeFunctionsPass());
    else if (!Extensions.empty()) // HLSL Change - GlobalExtensions not considered
      MPM.add(createBarrierNoopPass());

    if (!HLSLHighLevel)
      MPM.add(createDxilPreserveToSelectPass()); // HLSL Change - lower preserve instructions to selects

    addExtensionsToPM(EP_EnabledOnOptLevel0, MPM);

    // HLSL Change Begins.
    addHLSLPasses(HLSLHighLevel, OptLevel,
      this->HLSLOnlyWarnOnUnrollFail,
      this->StructurizeLoopExitsForUnroll,
      this->HLSLEnableLifetimeMarkers,
      this->HLSLExtensionsCodeGen,
      MPM);

    if (!HLSLHighLevel) {
      MPM.add(createDxilConvergentClearPass());
      MPM.add(createDxilSimpleGVNEliminateRegionPass());
      MPM.add(createDeadCodeEliminationPass());
      MPM.add(createDxilRemoveDeadBlocksPass());
      MPM.add(createDxilEraseDeadRegionPass());
      MPM.add(createDxilNoOptSimplifyInstructionsPass());
      MPM.add(createGlobalOptimizerPass());
      MPM.add(createMultiDimArrayToOneDimArrayPass());
      MPM.add(createDeadCodeEliminationPass());
      MPM.add(createGlobalDCEPass());
      MPM.add(createDxilMutateResourceToHandlePass());
      MPM.add(createDxilCleanupDynamicResourceHandlePass());
      MPM.add(createDxilLowerCreateHandleForLibPass());
      MPM.add(createDxilTranslateRawBuffer());
      MPM.add(createDxilLegalizeSampleOffsetPass());
      MPM.add(createDxilNoOptLegalizePass());
      MPM.add(createDxilFinalizePreservesPass());
      MPM.add(createDxilFinalizeModulePass());
      MPM.add(createComputeViewIdStatePass());
      MPM.add(createDxilDeadFunctionEliminationPass());
      MPM.add(createDxilDeleteRedundantDebugValuesPass());
      MPM.add(createNoPausePassesPass());
      MPM.add(createDxilEmitMetadataPass());
    }
    // HLSL Change Ends.
    return;
  }

  if (!HLSLHighLevel) {
    MPM.add(createHLEnsureMetadataPass()); // HLSL Change - rehydrate metadata from high-level codegen
  }

  // HLSL Change Begins

  MPM.add(createDxilRewriteOutputArgDebugInfoPass()); // Fix output argument types.

  MPM.add(createHLLegalizeParameter()); // legalize parameters before inline.
  MPM.add(createAlwaysInlinerPass(/*InsertLifeTime*/this->HLSLEnableLifetimeMarkers));
  if (Inliner) {
    delete Inliner;
    Inliner = nullptr;
  }
  addHLSLPasses(HLSLHighLevel, OptLevel, this->HLSLOnlyWarnOnUnrollFail, this->StructurizeLoopExitsForUnroll, this->HLSLEnableLifetimeMarkers, HLSLExtensionsCodeGen, MPM); // HLSL Change
  // HLSL Change Ends

  // Add LibraryInfo if we have some.
  if (LibraryInfo)
    MPM.add(new TargetLibraryInfoWrapperPass(*LibraryInfo));

  addInitialAliasAnalysisPasses(MPM);

  if (!DisableUnitAtATime) {
    addExtensionsToPM(EP_ModuleOptimizerEarly, MPM);

    MPM.add(createIPSCCPPass());              // IP SCCP
    MPM.add(createGlobalOptimizerPass());     // Optimize out global vars

    MPM.add(createDeadArgEliminationPass());  // Dead argument elimination

    MPM.add(createInstructionCombiningPass());// Clean up after IPCP & DAE
    addExtensionsToPM(EP_Peephole, MPM);
    MPM.add(createCFGSimplificationPass());   // Clean up after IPCP & DAE
  }

  // Start of CallGraph SCC passes.
  if (!DisableUnitAtATime)
    MPM.add(createPruneEHPass());             // Remove dead EH info
  if (Inliner) {
    MPM.add(Inliner);
    Inliner = nullptr;
  }
  if (!DisableUnitAtATime)
    MPM.add(createFunctionAttrsPass());       // Set readonly/readnone attrs

#if 0  // HLSL Change Starts: Disable ArgumentPromotion
  if (OptLevel > 2)
    MPM.add(createArgumentPromotionPass());   // Scalarize uninlined fn args
#endif // HLSL Change Ends

  // Start of function pass.
  // Break up aggregate allocas, using SSAUpdater.
  if (UseNewSROA)
    MPM.add(createSROAPass(/*RequiresDomTree*/ false));
  else
    MPM.add(createScalarReplAggregatesPass(-1, false));

  // HLSL Change. MPM.add(createEarlyCSEPass());              // Catch trivial redundancies
  // HLSL Change. MPM.add(createJumpThreadingPass());         // Thread jumps.
  MPM.add(createCorrelatedValuePropagationPass()); // Propagate conditionals
  MPM.add(createCFGSimplificationPass());     // Merge & remove BBs
  MPM.add(createInstructionCombiningPass());  // Combine silly seq's
  addExtensionsToPM(EP_Peephole, MPM);
  // HLSL Change Begins.
  // HLSL does not allow recursize functions.
  //MPM.add(createTailCallEliminationPass()); // Eliminate tail calls
  // HLSL Change Ends.
  MPM.add(createCFGSimplificationPass());     // Merge & remove BBs
  MPM.add(createReassociatePass());           // Reassociate expressions
  // Rotate Loop - disable header duplication at -Oz
  MPM.add(createLoopRotatePass(SizeLevel == 2 ? 0 : -1));
  // HLSL Change - disable LICM in frontend for not consider register pressure.
  //MPM.add(createLICMPass());                  // Hoist loop invariants
  //MPM.add(createLoopUnswitchPass(SizeLevel || OptLevel < 3)); // HLSL Change - may move barrier inside divergent if.
  MPM.add(createInstructionCombiningPass());
  MPM.add(createIndVarSimplifyPass());        // Canonicalize indvars
  // HLSL Change Begins
  // Don't allow loop idiom pass which may insert memset/memcpy thereby breaking the dxil
  //MPM.add(createLoopIdiomPass());             // Recognize idioms like memset.
  // HLSL Change Ends
  MPM.add(createLoopDeletionPass());          // Delete dead loops
  if (EnableLoopInterchange) {
    MPM.add(createLoopInterchangePass()); // Interchange loops
    MPM.add(createCFGSimplificationPass());
  }
  if (!DisableUnrollLoops)
    MPM.add(createSimpleLoopUnrollPass());    // Unroll small loops
  addExtensionsToPM(EP_LoopOptimizerEnd, MPM);

  if (OptLevel > 1) {
    if (EnableMLSM)
      MPM.add(createMergedLoadStoreMotionPass()); // Merge ld/st in diamonds
    // HLSL Change Begins
    if (EnableGVN) {
      MPM.add(createGVNPass(DisableGVNLoadPRE));  // Remove redundancies
      if (!HLSLResMayAlias)
        MPM.add(createDxilSimpleGVNHoistPass());
    }
    // HLSL Change Ends
  }

  // HLSL Change Begins.
  // Use value numbering to figure out if regions are equivalent, and branch to only one.
  MPM.add(createDxilSimpleGVNEliminateRegionPass());
  // HLSL don't allow memcpy and memset.
  //MPM.add(createMemCpyOptPass());             // Remove memcpy / form memset
  // HLSL Change Ends.
  MPM.add(createSCCPPass());                  // Constant prop with SCCP

  // Delete dead bit computations (instcombine runs after to fold away the dead
  // computations, and then ADCE will run later to exploit any new DCE
  // opportunities that creates).
  MPM.add(createBitTrackingDCEPass());        // Delete dead bit computations

  // Run instcombine after redundancy elimination to exploit opportunities
  // opened up by them.
  MPM.add(createInstructionCombiningPass());
  addExtensionsToPM(EP_Peephole, MPM);
  // HLSL Change. MPM.add(createJumpThreadingPass());         // Thread jumps
  MPM.add(createCorrelatedValuePropagationPass());
  MPM.add(createDeadStoreEliminationPass(ScanLimit));  // Delete dead stores
  // HLSL Change - disable LICM in frontend for not consider register pressure.
  // MPM.add(createLICMPass());

  addExtensionsToPM(EP_ScalarOptimizerLate, MPM);

  if (RerollLoops)
    MPM.add(createLoopRerollPass());
#if HLSL_VECTORIZATION_ENABLED // HLSL Change - don't build vectorization passes
  if (!RunSLPAfterLoopVectorization) {
    if (SLPVectorize)
      MPM.add(createSLPVectorizerPass());   // Vectorize parallel scalar chains.

    if (BBVectorize) {
      MPM.add(createBBVectorizePass());
      MPM.add(createInstructionCombiningPass());
      addExtensionsToPM(EP_Peephole, MPM);
      if (OptLevel > 1 && UseGVNAfterVectorization)
        MPM.add(createGVNPass(DisableGVNLoadPRE)); // Remove redundancies
      else
        MPM.add(createEarlyCSEPass());      // Catch trivial redundancies

      // BBVectorize may have significantly shortened a loop body; unroll again.
      if (!DisableUnrollLoops)
        MPM.add(createLoopUnrollPass());
    }
  }
#endif

  if (LoadCombine)
    MPM.add(createLoadCombinePass());

  MPM.add(createHoistConstantArrayPass()); // HLSL change

  MPM.add(createAggressiveDCEPass());         // Delete dead instructions
  MPM.add(createCFGSimplificationPass()); // Merge & remove BBs
  MPM.add(createInstructionCombiningPass());  // Clean up after everything.
  addExtensionsToPM(EP_Peephole, MPM);

  // FIXME: This is a HACK! The inliner pass above implicitly creates a CGSCC
  // pass manager that we are specifically trying to avoid. To prevent this
  // we must insert a no-op module pass to reset the pass manager.
  MPM.add(createBarrierNoopPass());

  if (RunFloat2Int)
    MPM.add(createFloat2IntPass());

  // Re-rotate loops in all our loop nests. These may have fallout out of
  // rotated form due to GVN or other transformations, and the vectorizer relies
  // on the rotated form. Disable header duplication at -Oz.
  MPM.add(createLoopRotatePass(SizeLevel == 2 ? 0 : -1));

  // Distribute loops to allow partial vectorization.  I.e. isolate dependences
  // into separate loop that would otherwise inhibit vectorization.
  if (EnableLoopDistribute)
    MPM.add(createLoopDistributePass());

#if HLSL_VECTORIZATION_ENABLED // HLSL Change - don't build vectorization passes
  MPM.add(createLoopVectorizePass(DisableUnrollLoops, LoopVectorize));
#endif

  // FIXME: Because of #pragma vectorize enable, the passes below are always
  // inserted in the pipeline, even when the vectorizer doesn't run (ex. when
  // on -O1 and no #pragma is found). Would be good to have these two passes
  // as function calls, so that we can only pass them when the vectorizer
  // changed the code.
  MPM.add(createInstructionCombiningPass());
#if HLSL_VECTORIZATION_ENABLED // HLSL Change - don't build vectorization passes
  if (OptLevel > 1 && ExtraVectorizerPasses) {
    // At higher optimization levels, try to clean up any runtime overlap and
    // alignment checks inserted by the vectorizer. We want to track correllated
    // runtime checks for two inner loops in the same outer loop, fold any
    // common computations, hoist loop-invariant aspects out of any outer loop,
    // and unswitch the runtime checks if possible. Once hoisted, we may have
    // dead (or speculatable) control flows or more combining opportunities.
    MPM.add(createEarlyCSEPass());
    MPM.add(createCorrelatedValuePropagationPass());
    MPM.add(createInstructionCombiningPass());
    MPM.add(createLICMPass());
    MPM.add(createLoopUnswitchPass(SizeLevel || OptLevel < 3));
    MPM.add(createCFGSimplificationPass());
    MPM.add(createInstructionCombiningPass());
  }

  if (RunSLPAfterLoopVectorization) {
    if (SLPVectorize) {
      MPM.add(createSLPVectorizerPass());   // Vectorize parallel scalar chains.
      if (OptLevel > 1 && ExtraVectorizerPasses) {
        MPM.add(createEarlyCSEPass());
      }
    }

    if (BBVectorize) {
      MPM.add(createBBVectorizePass());
      MPM.add(createInstructionCombiningPass());
      addExtensionsToPM(EP_Peephole, MPM);
      if (OptLevel > 1 && UseGVNAfterVectorization)
        MPM.add(createGVNPass(DisableGVNLoadPRE)); // Remove redundancies
      else
        MPM.add(createEarlyCSEPass());      // Catch trivial redundancies

      // BBVectorize may have significantly shortened a loop body; unroll again.
      if (!DisableUnrollLoops)
        MPM.add(createLoopUnrollPass());
    }
  }
#endif // HLSL Change - don't build vectorization passes

  addExtensionsToPM(EP_Peephole, MPM);
  MPM.add(createCFGSimplificationPass());
  MPM.add(createDxilLoopDeletionPass()); // HLSL Change - try to delete loop again.
  //MPM.add(createInstructionCombiningPass()); // HLSL Change - pass is included in above

  if (!DisableUnrollLoops) {
    MPM.add(createLoopUnrollPass(/* HLSL Change begin */-1, -1, -1, -1, this->StructurizeLoopExitsForUnroll /* HLSL Change end */));    // Unroll small loops

    // LoopUnroll may generate some redundency to cleanup.
    MPM.add(createInstructionCombiningPass());

    // Runtime unrolling will introduce runtime check in loop prologue. If the
    // unrolled loop is a inner loop, then the prologue will be inside the
    // outer loop. LICM pass can help to promote the runtime check out if the
    // checked value is loop invariant.
    // MPM.add(createLICMPass());// HLSL Change - disable LICM in frontend for
                                 // not consider register pressure.
  }

  // After vectorization and unrolling, assume intrinsics may tell us more
  // about pointer alignments.
  MPM.add(createAlignmentFromAssumptionsPass());

  if (!DisableUnitAtATime) {
    // FIXME: We shouldn't bother with this anymore.
    MPM.add(createStripDeadPrototypesPass()); // Get rid of dead prototypes

    // GlobalOpt already deletes dead functions and globals, at -O2 try a
    // late pass of GlobalDCE.  It is capable of deleting dead cycles.
    if (OptLevel > 1) {
      if (!PrepareForLTO) {
        // Remove avail extern fns and globals definitions if we aren't
        // compiling an object file for later LTO. For LTO we want to preserve
        // these so they are eligible for inlining at link-time. Note if they
        // are unreferenced they will be removed by GlobalDCE below, so
        // this only impacts referenced available externally globals.
        // Eventually they will be suppressed during codegen, but eliminating
        // here enables more opportunity for GlobalDCE as it may make
        // globals referenced by available external functions dead.
        MPM.add(createEliminateAvailableExternallyPass());
      }
      MPM.add(createGlobalDCEPass());         // Remove dead fns and globals.
      MPM.add(createConstantMergePass());     // Merge dup global constants
    }
  }

  if (MergeFunctions)
    MPM.add(createMergeFunctionsPass());

  // HLSL Change Begins.
  if (!HLSLHighLevel) {
    MPM.add(createDxilEraseDeadRegionPass());
    MPM.add(createDxilConvergentClearPass());
    MPM.add(createDeadCodeEliminationPass()); // DCE needed after clearing convergence
                                              // annotations before CreateHandleForLib
                                              // so no unused resources get re-added to
                                              // DxilModule.
    MPM.add(createMultiDimArrayToOneDimArrayPass());
    MPM.add(createDxilRemoveDeadBlocksPass());
    MPM.add(createDeadCodeEliminationPass());
    MPM.add(createGlobalDCEPass());
    MPM.add(createDxilMutateResourceToHandlePass());
    MPM.add(createDxilCleanupDynamicResourceHandlePass());
    MPM.add(createDxilLowerCreateHandleForLibPass());
    MPM.add(createDxilTranslateRawBuffer());
    // Always try to legalize sample offsets as loop unrolling
    // is not guaranteed for higher opt levels.
    MPM.add(createDxilLegalizeSampleOffsetPass());
    MPM.add(createDxilFinalizeModulePass());
    MPM.add(createComputeViewIdStatePass());
    MPM.add(createDxilDeadFunctionEliminationPass());
    MPM.add(createDxilDeleteRedundantDebugValuesPass());
    MPM.add(createNoPausePassesPass());
    MPM.add(createDxilValidateWaveSensitivityPass());
    MPM.add(createDxilEmitMetadataPass());
  }
  // HLSL Change Ends.
  addExtensionsToPM(EP_OptimizerLast, MPM);
}

#if 0 // HLSL Change: No LTO
void PassManagerBuilder::addLTOOptimizationPasses(legacy::PassManagerBase &PM) {
  // Provide AliasAnalysis services for optimizations.
  addInitialAliasAnalysisPasses(PM);

  // Propagate constants at call sites into the functions they call.  This
  // opens opportunities for globalopt (and inlining) by substituting function
  // pointers passed as arguments to direct uses of functions.
  PM.add(createIPSCCPPass());

  // Now that we internalized some globals, see if we can hack on them!
  PM.add(createGlobalOptimizerPass());

  // Linking modules together can lead to duplicated global constants, only
  // keep one copy of each constant.
  PM.add(createConstantMergePass());

  // Remove unused arguments from functions.
  PM.add(createDeadArgEliminationPass());

  // Reduce the code after globalopt and ipsccp.  Both can open up significant
  // simplification opportunities, and both can propagate functions through
  // function pointers.  When this happens, we often have to resolve varargs
  // calls, etc, so let instcombine do this.
  PM.add(createInstructionCombiningPass());
  addExtensionsToPM(EP_Peephole, PM);

  // Inline small functions
  bool RunInliner = Inliner;
  if (RunInliner) {
    PM.add(Inliner);
    Inliner = nullptr;
  }

  PM.add(createPruneEHPass());   // Remove dead EH info.

  // Optimize globals again if we ran the inliner.
  if (RunInliner)
    PM.add(createGlobalOptimizerPass());
  PM.add(createGlobalDCEPass()); // Remove dead functions.

  // If we didn't decide to inline a function, check to see if we can
  // transform it to pass arguments by value instead of by reference.
  PM.add(createArgumentPromotionPass());

  // The IPO passes may leave cruft around.  Clean up after them.
  PM.add(createInstructionCombiningPass());
  addExtensionsToPM(EP_Peephole, PM);
  // HLSL Change. PM.add(createJumpThreadingPass());

  // Break up allocas
  if (UseNewSROA)
    PM.add(createSROAPass());
  else
    PM.add(createScalarReplAggregatesPass());

  // Run a few AA driven optimizations here and now, to cleanup the code.
  PM.add(createFunctionAttrsPass()); // Add nocapture.
  PM.add(createGlobalsModRefPass()); // IP alias analysis.

  // HLSL Change - disable LICM in frontend for not consider register pressure.
  // PM.add(createLICMPass());                 // Hoist loop invariants.
  if (EnableMLSM)
    PM.add(createMergedLoadStoreMotionPass()); // Merge ld/st in diamonds.
  if (EnableGVN) // HLSL Change
    PM.add(createGVNPass(DisableGVNLoadPRE)); // Remove redundancies.
  PM.add(createMemCpyOptPass());            // Remove dead memcpys.

  // Nuke dead stores.
  PM.add(createDeadStoreEliminationPass(ScanLimit)); // HLSL Change - add ScanLimit

  // More loops are countable; try to optimize them.
  PM.add(createIndVarSimplifyPass());
  PM.add(createLoopDeletionPass());
  if (EnableLoopInterchange)
    PM.add(createLoopInterchangePass());

#if HLSL_VECTORIZATION_ENABLED // HLSL Change - don't build vectorization passes
  PM.add(createLoopVectorizePass(true, LoopVectorize));

  // More scalar chains could be vectorized due to more alias information
  if (RunSLPAfterLoopVectorization)
    if (SLPVectorize)
      PM.add(createSLPVectorizerPass()); // Vectorize parallel scalar chains.

  // After vectorization, assume intrinsics may tell us more about pointer
  // alignments.
  PM.add(createAlignmentFromAssumptionsPass());
#endif

  if (LoadCombine)
    PM.add(createLoadCombinePass());

  // Cleanup and simplify the code after the scalar optimizations.
  PM.add(createInstructionCombiningPass());
  addExtensionsToPM(EP_Peephole, PM);

  // HLSL Change. PM.add(createJumpThreadingPass());
}

void PassManagerBuilder::addLateLTOOptimizationPasses(
    legacy::PassManagerBase &PM) {
  // Delete basic blocks, which optimization passes may have killed.
  PM.add(createCFGSimplificationPass());

  // Now that we have optimized the program, discard unreachable functions.
  PM.add(createGlobalDCEPass());

  // FIXME: this is profitable (for compiler time) to do at -O0 too, but
  // currently it damages debug info.
  if (MergeFunctions)
    PM.add(createMergeFunctionsPass());
}

void PassManagerBuilder::populateLTOPassManager(legacy::PassManagerBase &PM) {
  if (LibraryInfo)
    PM.add(new TargetLibraryInfoWrapperPass(*LibraryInfo));

  if (VerifyInput)
    PM.add(createVerifierPass());

  if (OptLevel > 1)
    addLTOOptimizationPasses(PM);

  // Lower bit sets to globals. This pass supports Clang's control flow
  // integrity mechanisms (-fsanitize=cfi*) and needs to run at link time if CFI
  // is enabled. The pass does nothing if CFI is disabled.
  PM.add(createLowerBitSetsPass());

  if (OptLevel != 0)
    addLateLTOOptimizationPasses(PM);

  if (VerifyOutput)
    PM.add(createVerifierPass());
}
#endif

inline PassManagerBuilder *unwrap(LLVMPassManagerBuilderRef P) {
    return reinterpret_cast<PassManagerBuilder*>(P);
}

inline LLVMPassManagerBuilderRef wrap(PassManagerBuilder *P) {
  return reinterpret_cast<LLVMPassManagerBuilderRef>(P);
}

LLVMPassManagerBuilderRef LLVMPassManagerBuilderCreate() {
  PassManagerBuilder *PMB = new PassManagerBuilder();
  return wrap(PMB);
}

void LLVMPassManagerBuilderDispose(LLVMPassManagerBuilderRef PMB) {
  PassManagerBuilder *Builder = unwrap(PMB);
  delete Builder;
}

void
LLVMPassManagerBuilderSetOptLevel(LLVMPassManagerBuilderRef PMB,
                                  unsigned OptLevel) {
  PassManagerBuilder *Builder = unwrap(PMB);
  Builder->OptLevel = OptLevel;
}

void
LLVMPassManagerBuilderSetSizeLevel(LLVMPassManagerBuilderRef PMB,
                                   unsigned SizeLevel) {
  PassManagerBuilder *Builder = unwrap(PMB);
  Builder->SizeLevel = SizeLevel;
}

void
LLVMPassManagerBuilderSetDisableUnitAtATime(LLVMPassManagerBuilderRef PMB,
                                            LLVMBool Value) {
  PassManagerBuilder *Builder = unwrap(PMB);
  Builder->DisableUnitAtATime = Value;
}

void
LLVMPassManagerBuilderSetDisableUnrollLoops(LLVMPassManagerBuilderRef PMB,
                                            LLVMBool Value) {
  PassManagerBuilder *Builder = unwrap(PMB);
  Builder->DisableUnrollLoops = Value;
}

void
LLVMPassManagerBuilderSetDisableSimplifyLibCalls(LLVMPassManagerBuilderRef PMB,
                                                 LLVMBool Value) {
  // NOTE: The simplify-libcalls pass has been removed.
}

void
LLVMPassManagerBuilderUseInlinerWithThreshold(LLVMPassManagerBuilderRef PMB,
                                              unsigned Threshold) {
  PassManagerBuilder *Builder = unwrap(PMB);
  Builder->Inliner = createFunctionInliningPass(Threshold);
}

void
LLVMPassManagerBuilderPopulateFunctionPassManager(LLVMPassManagerBuilderRef PMB,
                                                  LLVMPassManagerRef PM) {
  PassManagerBuilder *Builder = unwrap(PMB);
  legacy::FunctionPassManager *FPM = unwrap<legacy::FunctionPassManager>(PM);
  Builder->populateFunctionPassManager(*FPM);
}

void
LLVMPassManagerBuilderPopulateModulePassManager(LLVMPassManagerBuilderRef PMB,
                                                LLVMPassManagerRef PM) {
  PassManagerBuilder *Builder = unwrap(PMB);
  legacy::PassManagerBase *MPM = unwrap(PM);
  Builder->populateModulePassManager(*MPM);
}

#if 0 // HLSL Change: No LTO
void LLVMPassManagerBuilderPopulateLTOPassManager(LLVMPassManagerBuilderRef PMB,
                                                  LLVMPassManagerRef PM,
                                                  LLVMBool Internalize,
                                                  LLVMBool RunInliner) {
  PassManagerBuilder *Builder = unwrap(PMB);
  legacy::PassManagerBase *LPM = unwrap(PM);

  // A small backwards compatibility hack. populateLTOPassManager used to take
  // an RunInliner option.
  if (RunInliner && !Builder->Inliner)
    Builder->Inliner = createFunctionInliningPass();

  Builder->populateLTOPassManager(*LPM);
}
#endif

//===-- Passes.cpp - Target independent code generation passes ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines interfaces to access the target independent code
// generation passes provided by the LLVM backend.
//
//===---------------------------------------------------------------------===//

#include "llvm/CodeGen/Passes.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/SymbolRewriter.h"

using namespace llvm;

static cl::opt<bool> DisablePostRA("disable-post-ra", cl::Hidden,
    cl::desc("Disable Post Regalloc"));
static cl::opt<bool> DisableBranchFold("disable-branch-fold", cl::Hidden,
    cl::desc("Disable branch folding"));
static cl::opt<bool> DisableTailDuplicate("disable-tail-duplicate", cl::Hidden,
    cl::desc("Disable tail duplication"));
static cl::opt<bool> DisableEarlyTailDup("disable-early-taildup", cl::Hidden,
    cl::desc("Disable pre-register allocation tail duplication"));
static cl::opt<bool> DisableBlockPlacement("disable-block-placement",
    cl::Hidden, cl::desc("Disable probability-driven block placement"));
static cl::opt<bool> EnableBlockPlacementStats("enable-block-placement-stats",
    cl::Hidden, cl::desc("Collect probability-driven block placement stats"));
static cl::opt<bool> DisableSSC("disable-ssc", cl::Hidden,
    cl::desc("Disable Stack Slot Coloring"));
static cl::opt<bool> DisableMachineDCE("disable-machine-dce", cl::Hidden,
    cl::desc("Disable Machine Dead Code Elimination"));
static cl::opt<bool> DisableEarlyIfConversion("disable-early-ifcvt", cl::Hidden,
    cl::desc("Disable Early If-conversion"));
static cl::opt<bool> DisableMachineLICM("disable-machine-licm", cl::Hidden,
    cl::desc("Disable Machine LICM"));
static cl::opt<bool> DisableMachineCSE("disable-machine-cse", cl::Hidden,
    cl::desc("Disable Machine Common Subexpression Elimination"));
static cl::opt<cl::boolOrDefault>
    EnableShrinkWrapOpt("enable-shrink-wrap", cl::Hidden,
                        cl::desc("enable the shrink-wrapping pass"));
static cl::opt<cl::boolOrDefault> OptimizeRegAlloc(
    "optimize-regalloc", cl::Hidden,
    cl::desc("Enable optimized register allocation compilation path."));
static cl::opt<bool> DisablePostRAMachineLICM("disable-postra-machine-licm",
    cl::Hidden,
    cl::desc("Disable Machine LICM"));
static cl::opt<bool> DisableMachineSink("disable-machine-sink", cl::Hidden,
    cl::desc("Disable Machine Sinking"));
static cl::opt<bool> DisableLSR("disable-lsr", cl::Hidden,
    cl::desc("Disable Loop Strength Reduction Pass"));
static cl::opt<bool> DisableConstantHoisting("disable-constant-hoisting",
    cl::Hidden, cl::desc("Disable ConstantHoisting"));
static cl::opt<bool> DisableCGP("disable-cgp", cl::Hidden,
    cl::desc("Disable Codegen Prepare"));
static cl::opt<bool> DisableCopyProp("disable-copyprop", cl::Hidden,
    cl::desc("Disable Copy Propagation pass"));
static cl::opt<bool> DisablePartialLibcallInlining("disable-partial-libcall-inlining",
    cl::Hidden, cl::desc("Disable Partial Libcall Inlining"));
static cl::opt<bool> EnableImplicitNullChecks(
    "enable-implicit-null-checks",
    cl::desc("Fold null checks into faulting memory operations"),
    cl::init(false));
static cl::opt<bool> PrintLSR("print-lsr-output", cl::Hidden,
    cl::desc("Print LLVM IR produced by the loop-reduce pass"));
static cl::opt<bool> PrintISelInput("print-isel-input", cl::Hidden,
    cl::desc("Print LLVM IR input to isel pass"));
static cl::opt<bool> PrintGCInfo("print-gc", cl::Hidden,
    cl::desc("Dump garbage collector data"));
static cl::opt<bool> VerifyMachineCode("verify-machineinstrs", cl::Hidden,
    cl::desc("Verify generated machine code"),
    cl::init(false),
    cl::ZeroOrMore);

static cl::opt<std::string>
PrintMachineInstrs("print-machineinstrs", cl::ValueOptional,
                   cl::desc("Print machine instrs"),
                   cl::value_desc("pass-name"), cl::init("option-unspecified"));

// Temporary option to allow experimenting with MachineScheduler as a post-RA
// scheduler. Targets can "properly" enable this with
// substitutePass(&PostRASchedulerID, &PostMachineSchedulerID); Ideally it
// wouldn't be part of the standard pass pipeline, and the target would just add
// a PostRA scheduling pass wherever it wants.
static cl::opt<bool> MISchedPostRA("misched-postra", cl::Hidden,
  cl::desc("Run MachineScheduler post regalloc (independent of preRA sched)"));

// Experimental option to run live interval analysis early.
static cl::opt<bool> EarlyLiveIntervals("early-live-intervals", cl::Hidden,
    cl::desc("Run live interval analysis earlier in the pipeline"));

static cl::opt<bool> UseCFLAA("use-cfl-aa-in-codegen",
  cl::init(false), cl::Hidden,
  cl::desc("Enable the new, experimental CFL alias analysis in CodeGen"));

/// Allow standard passes to be disabled by command line options. This supports
/// simple binary flags that either suppress the pass or do nothing.
/// i.e. -disable-mypass=false has no effect.
/// These should be converted to boolOrDefault in order to use applyOverride.
static IdentifyingPassPtr applyDisable(IdentifyingPassPtr PassID,
                                       bool Override) {
  if (Override)
    return IdentifyingPassPtr();
  return PassID;
}

/// Allow standard passes to be disabled by the command line, regardless of who
/// is adding the pass.
///
/// StandardID is the pass identified in the standard pass pipeline and provided
/// to addPass(). It may be a target-specific ID in the case that the target
/// directly adds its own pass, but in that case we harmlessly fall through.
///
/// TargetID is the pass that the target has configured to override StandardID.
///
/// StandardID may be a pseudo ID. In that case TargetID is the name of the real
/// pass to run. This allows multiple options to control a single pass depending
/// on where in the pipeline that pass is added.
static IdentifyingPassPtr overridePass(AnalysisID StandardID,
                                       IdentifyingPassPtr TargetID) {
  if (StandardID == &PostRASchedulerID)
    return applyDisable(TargetID, DisablePostRA);

  if (StandardID == &BranchFolderPassID)
    return applyDisable(TargetID, DisableBranchFold);

  if (StandardID == &TailDuplicateID)
    return applyDisable(TargetID, DisableTailDuplicate);

  if (StandardID == &TargetPassConfig::EarlyTailDuplicateID)
    return applyDisable(TargetID, DisableEarlyTailDup);

  if (StandardID == &MachineBlockPlacementID)
    return applyDisable(TargetID, DisableBlockPlacement);

  if (StandardID == &StackSlotColoringID)
    return applyDisable(TargetID, DisableSSC);

  if (StandardID == &DeadMachineInstructionElimID)
    return applyDisable(TargetID, DisableMachineDCE);

  if (StandardID == &EarlyIfConverterID)
    return applyDisable(TargetID, DisableEarlyIfConversion);

  if (StandardID == &MachineLICMID)
    return applyDisable(TargetID, DisableMachineLICM);

  if (StandardID == &MachineCSEID)
    return applyDisable(TargetID, DisableMachineCSE);

  if (StandardID == &TargetPassConfig::PostRAMachineLICMID)
    return applyDisable(TargetID, DisablePostRAMachineLICM);

  if (StandardID == &MachineSinkingID)
    return applyDisable(TargetID, DisableMachineSink);

  if (StandardID == &MachineCopyPropagationID)
    return applyDisable(TargetID, DisableCopyProp);

  return TargetID;
}

//===---------------------------------------------------------------------===//
/// TargetPassConfig
//===---------------------------------------------------------------------===//

INITIALIZE_PASS(TargetPassConfig, "targetpassconfig",
                "Target Pass Configuration", false, false)
char TargetPassConfig::ID = 0;

// Pseudo Pass IDs.
char TargetPassConfig::EarlyTailDuplicateID = 0;
char TargetPassConfig::PostRAMachineLICMID = 0;

namespace llvm {
class PassConfigImpl {
public:
  // List of passes explicitly substituted by this target. Normally this is
  // empty, but it is a convenient way to suppress or replace specific passes
  // that are part of a standard pass pipeline without overridding the entire
  // pipeline. This mechanism allows target options to inherit a standard pass's
  // user interface. For example, a target may disable a standard pass by
  // default by substituting a pass ID of zero, and the user may still enable
  // that standard pass with an explicit command line option.
  DenseMap<AnalysisID,IdentifyingPassPtr> TargetPasses;

  /// Store the pairs of <AnalysisID, AnalysisID> of which the second pass
  /// is inserted after each instance of the first one.
  SmallVector<std::pair<AnalysisID, IdentifyingPassPtr>, 4> InsertedPasses;
};
} // namespace llvm

// Out of line virtual method.
TargetPassConfig::~TargetPassConfig() {
  delete Impl;
}

// Out of line constructor provides default values for pass options and
// registers all common codegen passes.
TargetPassConfig::TargetPassConfig(TargetMachine *tm, PassManagerBase &pm)
    : ImmutablePass(ID), PM(&pm), StartBefore(nullptr), StartAfter(nullptr),
      StopAfter(nullptr), Started(true), Stopped(false),
      AddingMachinePasses(false), TM(tm), Impl(nullptr), Initialized(false),
      DisableVerify(false), EnableTailMerge(true), EnableShrinkWrap(false) {

  Impl = new PassConfigImpl();

  // Register all target independent codegen passes to activate their PassIDs,
  // including this pass itself.
  initializeCodeGen(*PassRegistry::getPassRegistry());

  // Substitute Pseudo Pass IDs for real ones.
  substitutePass(&EarlyTailDuplicateID, &TailDuplicateID);
  substitutePass(&PostRAMachineLICMID, &MachineLICMID);
}

/// Insert InsertedPassID pass after TargetPassID.
void TargetPassConfig::insertPass(AnalysisID TargetPassID,
                                  IdentifyingPassPtr InsertedPassID) {
  assert(((!InsertedPassID.isInstance() &&
           TargetPassID != InsertedPassID.getID()) ||
          (InsertedPassID.isInstance() &&
           TargetPassID != InsertedPassID.getInstance()->getPassID())) &&
         "Insert a pass after itself!");
  std::pair<AnalysisID, IdentifyingPassPtr> P(TargetPassID, InsertedPassID);
  Impl->InsertedPasses.push_back(P);
}

/// createPassConfig - Create a pass configuration object to be used by
/// addPassToEmitX methods for generating a pipeline of CodeGen passes.
///
/// Targets may override this to extend TargetPassConfig.
TargetPassConfig *LLVMTargetMachine::createPassConfig(PassManagerBase &PM) {
  return new TargetPassConfig(this, PM);
}

TargetPassConfig::TargetPassConfig()
  : ImmutablePass(ID), PM(nullptr) {
  llvm_unreachable("TargetPassConfig should not be constructed on-the-fly");
}

// Helper to verify the analysis is really immutable.
void TargetPassConfig::setOpt(bool &Opt, bool Val) {
  assert(!Initialized && "PassConfig is immutable");
  Opt = Val;
}

void TargetPassConfig::substitutePass(AnalysisID StandardID,
                                      IdentifyingPassPtr TargetID) {
  Impl->TargetPasses[StandardID] = TargetID;
}

IdentifyingPassPtr TargetPassConfig::getPassSubstitution(AnalysisID ID) const {
  DenseMap<AnalysisID, IdentifyingPassPtr>::const_iterator
    I = Impl->TargetPasses.find(ID);
  if (I == Impl->TargetPasses.end())
    return ID;
  return I->second;
}

/// Add a pass to the PassManager if that pass is supposed to be run.  If the
/// Started/Stopped flags indicate either that the compilation should start at
/// a later pass or that it should stop after an earlier pass, then do not add
/// the pass.  Finally, compare the current pass against the StartAfter
/// and StopAfter options and change the Started/Stopped flags accordingly.
void TargetPassConfig::addPass(Pass *P, bool verifyAfter, bool printAfter) {
  assert(!Initialized && "PassConfig is immutable");

  // Cache the Pass ID here in case the pass manager finds this pass is
  // redundant with ones already scheduled / available, and deletes it.
  // Fundamentally, once we add the pass to the manager, we no longer own it
  // and shouldn't reference it.
  AnalysisID PassID = P->getPassID();

  if (StartBefore == PassID)
    Started = true;
  if (Started && !Stopped) {
    std::string Banner;
    // Construct banner message before PM->add() as that may delete the pass.
    if (AddingMachinePasses && (printAfter || verifyAfter))
      Banner = std::string("After ") + std::string(P->getPassName());
    PM->add(P);
    if (AddingMachinePasses) {
      if (printAfter)
        addPrintPass(Banner);
      if (verifyAfter)
        addVerifyPass(Banner);
    }

    // Add the passes after the pass P if there is any.
    for (SmallVectorImpl<std::pair<AnalysisID, IdentifyingPassPtr> >::iterator
             I = Impl->InsertedPasses.begin(),
             E = Impl->InsertedPasses.end();
         I != E; ++I) {
      if ((*I).first == PassID) {
        assert((*I).second.isValid() && "Illegal Pass ID!");
        Pass *NP;
        if ((*I).second.isInstance())
          NP = (*I).second.getInstance();
        else {
          NP = Pass::createPass((*I).second.getID());
          assert(NP && "Pass ID not registered");
        }
        addPass(NP, false, false);
      }
    }
  } else {
    delete P;
  }
  if (StopAfter == PassID)
    Stopped = true;
  if (StartAfter == PassID)
    Started = true;
  if (Stopped && !Started)
    report_fatal_error("Cannot stop compilation after pass that is not run");
}

/// Add a CodeGen pass at this point in the pipeline after checking for target
/// and command line overrides.
///
/// addPass cannot return a pointer to the pass instance because is internal the
/// PassManager and the instance we create here may already be freed.
AnalysisID TargetPassConfig::addPass(AnalysisID PassID, bool verifyAfter,
                                     bool printAfter) {
  IdentifyingPassPtr TargetID = getPassSubstitution(PassID);
  IdentifyingPassPtr FinalPtr = overridePass(PassID, TargetID);
  if (!FinalPtr.isValid())
    return nullptr;

  Pass *P;
  if (FinalPtr.isInstance())
    P = FinalPtr.getInstance();
  else {
    P = Pass::createPass(FinalPtr.getID());
    if (!P)
      llvm_unreachable("Pass ID not registered");
  }
  AnalysisID FinalID = P->getPassID();
  addPass(P, verifyAfter, printAfter); // Ends the lifetime of P.

  return FinalID;
}

void TargetPassConfig::printAndVerify(const std::string &Banner) {
  addPrintPass(Banner);
  addVerifyPass(Banner);
}

void TargetPassConfig::addPrintPass(const std::string &Banner) {
  if (TM->shouldPrintMachineCode())
    PM->add(createMachineFunctionPrinterPass(dbgs(), Banner));
}

void TargetPassConfig::addVerifyPass(const std::string &Banner) {
  if (VerifyMachineCode)
    PM->add(createMachineVerifierPass(Banner));
}

/// Add common target configurable passes that perform LLVM IR to IR transforms
/// following machine independent optimization.
void TargetPassConfig::addIRPasses() {
  // Basic AliasAnalysis support.
  // Add TypeBasedAliasAnalysis before BasicAliasAnalysis so that
  // BasicAliasAnalysis wins if they disagree. This is intended to help
  // support "obvious" type-punning idioms.
  if (UseCFLAA)
    addPass(createCFLAliasAnalysisPass());
  addPass(createTypeBasedAliasAnalysisPass());
  addPass(createScopedNoAliasAAPass());
  addPass(createBasicAliasAnalysisPass());

  // Before running any passes, run the verifier to determine if the input
  // coming from the front-end and/or optimizer is valid.
  if (!DisableVerify)
    addPass(createVerifierPass());

  // Run loop strength reduction before anything else.
  if (getOptLevel() != CodeGenOpt::None && !DisableLSR) {
    addPass(createLoopStrengthReducePass());
    if (PrintLSR)
      addPass(createPrintFunctionPass(dbgs(), "\n\n*** Code after LSR ***\n"));
  }

  // Run GC lowering passes for builtin collectors
  // TODO: add a pass insertion point here
  addPass(createGCLoweringPass());
  addPass(createShadowStackGCLoweringPass());

  // Make sure that no unreachable blocks are instruction selected.
  addPass(createUnreachableBlockEliminationPass());

  // Prepare expensive constants for SelectionDAG.
  if (getOptLevel() != CodeGenOpt::None && !DisableConstantHoisting)
    addPass(createConstantHoistingPass());

  if (getOptLevel() != CodeGenOpt::None && !DisablePartialLibcallInlining)
    addPass(createPartiallyInlineLibCallsPass());
}

/// Turn exception handling constructs into something the code generators can
/// handle.
void TargetPassConfig::addPassesToHandleExceptions() {
  switch (TM->getMCAsmInfo()->getExceptionHandlingType()) {
  case ExceptionHandling::SjLj:
    // SjLj piggy-backs on dwarf for this bit. The cleanups done apply to both
    // Dwarf EH prepare needs to be run after SjLj prepare. Otherwise,
    // catch info can get misplaced when a selector ends up more than one block
    // removed from the parent invoke(s). This could happen when a landing
    // pad is shared by multiple invokes and is also a target of a normal
    // edge from elsewhere.
    addPass(createSjLjEHPreparePass());
    // FALLTHROUGH
  case ExceptionHandling::DwarfCFI:
  case ExceptionHandling::ARM:
    addPass(createDwarfEHPass(TM));
    break;
  case ExceptionHandling::WinEH:
    // We support using both GCC-style and MSVC-style exceptions on Windows, so
    // add both preparation passes. Each pass will only actually run if it
    // recognizes the personality function.
    addPass(createWinEHPass(TM));
    addPass(createDwarfEHPass(TM));
    break;
  case ExceptionHandling::None:
    addPass(createLowerInvokePass());

    // The lower invoke pass may create unreachable code. Remove it.
    addPass(createUnreachableBlockEliminationPass());
    break;
  }
}

/// Add pass to prepare the LLVM IR for code generation. This should be done
/// before exception handling preparation passes.
void TargetPassConfig::addCodeGenPrepare() {
  if (getOptLevel() != CodeGenOpt::None && !DisableCGP)
    addPass(createCodeGenPreparePass(TM));
  addPass(createRewriteSymbolsPass());
}

/// Add common passes that perform LLVM IR to IR transforms in preparation for
/// instruction selection.
void TargetPassConfig::addISelPrepare() {
  addPreISel();

  // Add both the safe stack and the stack protection passes: each of them will
  // only protect functions that have corresponding attributes.
  addPass(createSafeStackPass());
  addPass(createStackProtectorPass(TM));

  if (PrintISelInput)
    addPass(createPrintFunctionPass(
        dbgs(), "\n\n*** Final LLVM Code input to ISel ***\n"));

  // All passes which modify the LLVM IR are now complete; run the verifier
  // to ensure that the IR is valid.
  if (!DisableVerify)
    addPass(createVerifierPass());
}

/// Add the complete set of target-independent postISel code generator passes.
///
/// This can be read as the standard order of major LLVM CodeGen stages. Stages
/// with nontrivial configuration or multiple passes are broken out below in
/// add%Stage routines.
///
/// Any TargetPassConfig::addXX routine may be overriden by the Target. The
/// addPre/Post methods with empty header implementations allow injecting
/// target-specific fixups just before or after major stages. Additionally,
/// targets have the flexibility to change pass order within a stage by
/// overriding default implementation of add%Stage routines below. Each
/// technique has maintainability tradeoffs because alternate pass orders are
/// not well supported. addPre/Post works better if the target pass is easily
/// tied to a common pass. But if it has subtle dependencies on multiple passes,
/// the target should override the stage instead.
///
/// TODO: We could use a single addPre/Post(ID) hook to allow pass injection
/// before/after any target-independent pass. But it's currently overkill.
void TargetPassConfig::addMachinePasses() {
  AddingMachinePasses = true;

  // Insert a machine instr printer pass after the specified pass.
  // If -print-machineinstrs specified, print machineinstrs after all passes.
  if (StringRef(PrintMachineInstrs.getValue()).equals(""))
    TM->Options.PrintMachineCode = true;
  else if (!StringRef(PrintMachineInstrs.getValue())
           .equals("option-unspecified")) {
    const PassRegistry *PR = PassRegistry::getPassRegistry();
    const PassInfo *TPI = PR->getPassInfo(PrintMachineInstrs.getValue());
    const PassInfo *IPI = PR->getPassInfo(StringRef("machineinstr-printer"));
    assert (TPI && IPI && "Pass ID not registered!");
    const char *TID = (const char *)(TPI->getTypeInfo());
    const char *IID = (const char *)(IPI->getTypeInfo());
    insertPass(TID, IID);
  }

  // Print the instruction selected machine code...
  printAndVerify("After Instruction Selection");

  // Expand pseudo-instructions emitted by ISel.
  addPass(&ExpandISelPseudosID);

  // Add passes that optimize machine instructions in SSA form.
  if (getOptLevel() != CodeGenOpt::None) {
    addMachineSSAOptimization();
  } else {
    // If the target requests it, assign local variables to stack slots relative
    // to one another and simplify frame index references where possible.
    addPass(&LocalStackSlotAllocationID, false);
  }

  // Run pre-ra passes.
  addPreRegAlloc();

  // Run register allocation and passes that are tightly coupled with it,
  // including phi elimination and scheduling.
  if (getOptimizeRegAlloc())
    addOptimizedRegAlloc(createRegAllocPass(true));
  else
    addFastRegAlloc(createRegAllocPass(false));

  // Run post-ra passes.
  addPostRegAlloc();

  // Insert prolog/epilog code.  Eliminate abstract frame index references...
  if (getEnableShrinkWrap())
    addPass(&ShrinkWrapID);
  addPass(&PrologEpilogCodeInserterID);

  /// Add passes that optimize machine instructions after register allocation.
  if (getOptLevel() != CodeGenOpt::None)
    addMachineLateOptimization();

  // Expand pseudo instructions before second scheduling pass.
  addPass(&ExpandPostRAPseudosID);

  // Run pre-sched2 passes.
  addPreSched2();

  if (EnableImplicitNullChecks)
    addPass(&ImplicitNullChecksID);

  // Second pass scheduler.
  if (getOptLevel() != CodeGenOpt::None) {
    if (MISchedPostRA)
      addPass(&PostMachineSchedulerID);
    else
      addPass(&PostRASchedulerID);
  }

  // GC
  if (addGCPasses()) {
    if (PrintGCInfo)
      addPass(createGCInfoPrinter(dbgs()), false, false);
  }

  // Basic block placement.
  if (getOptLevel() != CodeGenOpt::None)
    addBlockPlacement();

  addPreEmitPass();

  addPass(&StackMapLivenessID, false);

  AddingMachinePasses = false;
}

/// Add passes that optimize machine instructions in SSA form.
void TargetPassConfig::addMachineSSAOptimization() {
  // Pre-ra tail duplication.
  addPass(&EarlyTailDuplicateID);

  // Optimize PHIs before DCE: removing dead PHI cycles may make more
  // instructions dead.
  addPass(&OptimizePHIsID, false);

  // This pass merges large allocas. StackSlotColoring is a different pass
  // which merges spill slots.
  addPass(&StackColoringID, false);

  // If the target requests it, assign local variables to stack slots relative
  // to one another and simplify frame index references where possible.
  addPass(&LocalStackSlotAllocationID, false);

  // With optimization, dead code should already be eliminated. However
  // there is one known exception: lowered code for arguments that are only
  // used by tail calls, where the tail calls reuse the incoming stack
  // arguments directly (see t11 in test/CodeGen/X86/sibcall.ll).
  addPass(&DeadMachineInstructionElimID);

  // Allow targets to insert passes that improve instruction level parallelism,
  // like if-conversion. Such passes will typically need dominator trees and
  // loop info, just like LICM and CSE below.
  addILPOpts();

  addPass(&MachineLICMID, false);
  addPass(&MachineCSEID, false);
  addPass(&MachineSinkingID);

  addPass(&PeepholeOptimizerID, false);
  // Clean-up the dead code that may have been generated by peephole
  // rewriting.
  addPass(&DeadMachineInstructionElimID);
}

bool TargetPassConfig::getEnableShrinkWrap() const {
  switch (EnableShrinkWrapOpt) {
  case cl::BOU_UNSET:
    return EnableShrinkWrap && getOptLevel() != CodeGenOpt::None;
  // If EnableShrinkWrap is set, it takes precedence on whatever the
  // target sets. The rational is that we assume we want to test
  // something related to shrink-wrapping.
  case cl::BOU_TRUE:
    return true;
  case cl::BOU_FALSE:
    return false;
  }
  llvm_unreachable("Invalid shrink-wrapping state");
}

//===---------------------------------------------------------------------===//
/// Register Allocation Pass Configuration
//===---------------------------------------------------------------------===//

bool TargetPassConfig::getOptimizeRegAlloc() const {
  switch (OptimizeRegAlloc) {
  case cl::BOU_UNSET: return getOptLevel() != CodeGenOpt::None;
  case cl::BOU_TRUE:  return true;
  case cl::BOU_FALSE: return false;
  }
  llvm_unreachable("Invalid optimize-regalloc state");
}

/// RegisterRegAlloc's global Registry tracks allocator registration.
MachinePassRegistry RegisterRegAlloc::Registry;

/// A dummy default pass factory indicates whether the register allocator is
/// overridden on the command line.
static FunctionPass *useDefaultRegisterAllocator() { return nullptr; }
static RegisterRegAlloc
defaultRegAlloc("default",
                "pick register allocator based on -O option",
                useDefaultRegisterAllocator);

/// -regalloc=... command line option.
static cl::opt<RegisterRegAlloc::FunctionPassCtor, false,
               RegisterPassParser<RegisterRegAlloc> >
RegAlloc("regalloc",
         cl::init(&useDefaultRegisterAllocator),
         cl::desc("Register allocator to use"));


/// Instantiate the default register allocator pass for this target for either
/// the optimized or unoptimized allocation path. This will be added to the pass
/// manager by addFastRegAlloc in the unoptimized case or addOptimizedRegAlloc
/// in the optimized case.
///
/// A target that uses the standard regalloc pass order for fast or optimized
/// allocation may still override this for per-target regalloc
/// selection. But -regalloc=... always takes precedence.
FunctionPass *TargetPassConfig::createTargetRegisterAllocator(bool Optimized) {
  if (Optimized)
    return createGreedyRegisterAllocator();
  else
    return createFastRegisterAllocator();
}

/// Find and instantiate the register allocation pass requested by this target
/// at the current optimization level.  Different register allocators are
/// defined as separate passes because they may require different analysis.
///
/// This helper ensures that the regalloc= option is always available,
/// even for targets that override the default allocator.
///
/// FIXME: When MachinePassRegistry register pass IDs instead of function ptrs,
/// this can be folded into addPass.
FunctionPass *TargetPassConfig::createRegAllocPass(bool Optimized) {
  RegisterRegAlloc::FunctionPassCtor Ctor = RegisterRegAlloc::getDefault();

  // Initialize the global default.
  if (!Ctor) {
    Ctor = RegAlloc;
    RegisterRegAlloc::setDefault(RegAlloc);
  }
  if (Ctor != useDefaultRegisterAllocator)
    return Ctor();

  // With no -regalloc= override, ask the target for a regalloc pass.
  return createTargetRegisterAllocator(Optimized);
}

/// Return true if the default global register allocator is in use and
/// has not be overriden on the command line with '-regalloc=...'
bool TargetPassConfig::usingDefaultRegAlloc() const {
  return RegAlloc.getNumOccurrences() == 0;
}

/// Add the minimum set of target-independent passes that are required for
/// register allocation. No coalescing or scheduling.
void TargetPassConfig::addFastRegAlloc(FunctionPass *RegAllocPass) {
  addPass(&PHIEliminationID, false);
  addPass(&TwoAddressInstructionPassID, false);

  addPass(RegAllocPass);
}

/// Add standard target-independent passes that are tightly coupled with
/// optimized register allocation, including coalescing, machine instruction
/// scheduling, and register allocation itself.
void TargetPassConfig::addOptimizedRegAlloc(FunctionPass *RegAllocPass) {
  addPass(&ProcessImplicitDefsID, false);

  // LiveVariables currently requires pure SSA form.
  //
  // FIXME: Once TwoAddressInstruction pass no longer uses kill flags,
  // LiveVariables can be removed completely, and LiveIntervals can be directly
  // computed. (We still either need to regenerate kill flags after regalloc, or
  // preferably fix the scavenger to not depend on them).
  addPass(&LiveVariablesID, false);

  // Edge splitting is smarter with machine loop info.
  addPass(&MachineLoopInfoID, false);
  addPass(&PHIEliminationID, false);

  // Eventually, we want to run LiveIntervals before PHI elimination.
  if (EarlyLiveIntervals)
    addPass(&LiveIntervalsID, false);

  addPass(&TwoAddressInstructionPassID, false);
  addPass(&RegisterCoalescerID);

  // PreRA instruction scheduling.
  addPass(&MachineSchedulerID);

  // Add the selected register allocation pass.
  addPass(RegAllocPass);

  // Allow targets to change the register assignments before rewriting.
  addPreRewrite();

  // Finally rewrite virtual registers.
  addPass(&VirtRegRewriterID);

  // Perform stack slot coloring and post-ra machine LICM.
  //
  // FIXME: Re-enable coloring with register when it's capable of adding
  // kill markers.
  addPass(&StackSlotColoringID);

  // Run post-ra machine LICM to hoist reloads / remats.
  //
  // FIXME: can this move into MachineLateOptimization?
  addPass(&PostRAMachineLICMID);
}

//===---------------------------------------------------------------------===//
/// Post RegAlloc Pass Configuration
//===---------------------------------------------------------------------===//

/// Add passes that optimize machine instructions after register allocation.
void TargetPassConfig::addMachineLateOptimization() {
  // Branch folding must be run after regalloc and prolog/epilog insertion.
  addPass(&BranchFolderPassID);

  // Tail duplication.
  // Note that duplicating tail just increases code size and degrades
  // performance for targets that require Structured Control Flow.
  // In addition it can also make CFG irreducible. Thus we disable it.
  if (!TM->requiresStructuredCFG())
    addPass(&TailDuplicateID);

  // Copy propagation.
  addPass(&MachineCopyPropagationID);
}

/// Add standard GC passes.
bool TargetPassConfig::addGCPasses() {
  addPass(&GCMachineCodeAnalysisID, false);
  return true;
}

/// Add standard basic block placement passes.
void TargetPassConfig::addBlockPlacement() {
  if (addPass(&MachineBlockPlacementID, false)) {
    // Run a separate pass to collect block placement statistics.
    if (EnableBlockPlacementStats)
      addPass(&MachineBlockPlacementStatsID);
  }
}

//===-- Passes.h - Target independent code generation passes ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines interfaces to access the target independent code generation
// passes provided by the LLVM backend.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_PASSES_H
#define LLVM_CODEGEN_PASSES_H

#include "llvm/Pass.h"
#include "llvm/Target/TargetMachine.h"
#include <functional>
#include <string>

namespace llvm {

class MachineFunctionPass;
class PassConfigImpl;
class PassInfo;
class ScheduleDAGInstrs;
class TargetLowering;
class TargetLoweringBase;
class TargetRegisterClass;
class raw_ostream;
struct MachineSchedContext;

// The old pass manager infrastructure is hidden in a legacy namespace now.
namespace legacy {
class PassManagerBase;
}
using legacy::PassManagerBase;

/// Discriminated union of Pass ID types.
///
/// The PassConfig API prefers dealing with IDs because they are safer and more
/// efficient. IDs decouple configuration from instantiation. This way, when a
/// pass is overriden, it isn't unnecessarily instantiated. It is also unsafe to
/// refer to a Pass pointer after adding it to a pass manager, which deletes
/// redundant pass instances.
///
/// However, it is convient to directly instantiate target passes with
/// non-default ctors. These often don't have a registered PassInfo. Rather than
/// force all target passes to implement the pass registry boilerplate, allow
/// the PassConfig API to handle either type.
///
/// AnalysisID is sadly char*, so PointerIntPair won't work.
class IdentifyingPassPtr {
  union {
    AnalysisID ID;
    Pass *P;
  };
  bool IsInstance;
public:
  IdentifyingPassPtr() : P(nullptr), IsInstance(false) {}
  IdentifyingPassPtr(AnalysisID IDPtr) : ID(IDPtr), IsInstance(false) {}
  IdentifyingPassPtr(Pass *InstancePtr) : P(InstancePtr), IsInstance(true) {}

  bool isValid() const { return P; }
  bool isInstance() const { return IsInstance; }

  AnalysisID getID() const {
    assert(!IsInstance && "Not a Pass ID");
    return ID;
  }
  Pass *getInstance() const {
    assert(IsInstance && "Not a Pass Instance");
    return P;
  }
};

template <> struct isPodLike<IdentifyingPassPtr> {
  static const bool value = true;
};

/// Target-Independent Code Generator Pass Configuration Options.
///
/// This is an ImmutablePass solely for the purpose of exposing CodeGen options
/// to the internals of other CodeGen passes.
class TargetPassConfig : public ImmutablePass {
public:
  /// Pseudo Pass IDs. These are defined within TargetPassConfig because they
  /// are unregistered pass IDs. They are only useful for use with
  /// TargetPassConfig APIs to identify multiple occurrences of the same pass.
  ///

  /// EarlyTailDuplicate - A clone of the TailDuplicate pass that runs early
  /// during codegen, on SSA form.
  static char EarlyTailDuplicateID;

  /// PostRAMachineLICM - A clone of the LICM pass that runs during late machine
  /// optimization after regalloc.
  static char PostRAMachineLICMID;

private:
  PassManagerBase *PM;
  AnalysisID StartBefore, StartAfter;
  AnalysisID StopAfter;
  bool Started;
  bool Stopped;
  bool AddingMachinePasses;

protected:
  TargetMachine *TM;
  PassConfigImpl *Impl; // Internal data structures
  bool Initialized;     // Flagged after all passes are configured.

  // Target Pass Options
  // Targets provide a default setting, user flags override.
  //
  bool DisableVerify;

  /// Default setting for -enable-tail-merge on this target.
  bool EnableTailMerge;

  /// Default setting for -enable-shrink-wrap on this target.
  bool EnableShrinkWrap;

public:
  TargetPassConfig(TargetMachine *tm, PassManagerBase &pm);
  // Dummy constructor.
  TargetPassConfig();

  ~TargetPassConfig() override;

  static char ID;

  /// Get the right type of TargetMachine for this target.
  template<typename TMC> TMC &getTM() const {
    return *static_cast<TMC*>(TM);
  }

  //
  void setInitialized() { Initialized = true; }

  CodeGenOpt::Level getOptLevel() const { return TM->getOptLevel(); }

  /// Set the StartAfter, StartBefore and StopAfter passes to allow running only
  /// a portion of the normal code-gen pass sequence.
  ///
  /// If the StartAfter and StartBefore pass ID is zero, then compilation will
  /// begin at the normal point; otherwise, clear the Started flag to indicate
  /// that passes should not be added until the starting pass is seen.  If the
  /// Stop pass ID is zero, then compilation will continue to the end.
  ///
  /// This function expects that at least one of the StartAfter or the
  /// StartBefore pass IDs is null.
  void setStartStopPasses(AnalysisID StartBefore, AnalysisID StartAfter,
                          AnalysisID StopAfter) {
    if (StartAfter)
      assert(!StartBefore && "Start after and start before passes are given");
    this->StartBefore = StartBefore;
    this->StartAfter = StartAfter;
    this->StopAfter = StopAfter;
    Started = (StartAfter == nullptr) && (StartBefore == nullptr);
  }

  void setDisableVerify(bool Disable) { setOpt(DisableVerify, Disable); }

  bool getEnableTailMerge() const { return EnableTailMerge; }
  void setEnableTailMerge(bool Enable) { setOpt(EnableTailMerge, Enable); }

  /// Allow the target to override a specific pass without overriding the pass
  /// pipeline. When passes are added to the standard pipeline at the
  /// point where StandardID is expected, add TargetID in its place.
  void substitutePass(AnalysisID StandardID, IdentifyingPassPtr TargetID);

  /// Insert InsertedPassID pass after TargetPassID pass.
  void insertPass(AnalysisID TargetPassID, IdentifyingPassPtr InsertedPassID);

  /// Allow the target to enable a specific standard pass by default.
  void enablePass(AnalysisID PassID) { substitutePass(PassID, PassID); }

  /// Allow the target to disable a specific standard pass by default.
  void disablePass(AnalysisID PassID) {
    substitutePass(PassID, IdentifyingPassPtr());
  }

  /// Return the pass substituted for StandardID by the target.
  /// If no substitution exists, return StandardID.
  IdentifyingPassPtr getPassSubstitution(AnalysisID StandardID) const;

  /// Return true if the optimized regalloc pipeline is enabled.
  bool getOptimizeRegAlloc() const;

  /// Return true if shrink wrapping is enabled.
  bool getEnableShrinkWrap() const;

  /// Return true if the default global register allocator is in use and
  /// has not be overriden on the command line with '-regalloc=...'
  bool usingDefaultRegAlloc() const;

  /// Add common target configurable passes that perform LLVM IR to IR
  /// transforms following machine independent optimization.
  virtual void addIRPasses();

  /// Add passes to lower exception handling for the code generator.
  void addPassesToHandleExceptions();

  /// Add pass to prepare the LLVM IR for code generation. This should be done
  /// before exception handling preparation passes.
  virtual void addCodeGenPrepare();

  /// Add common passes that perform LLVM IR to IR transforms in preparation for
  /// instruction selection.
  virtual void addISelPrepare();

  /// addInstSelector - This method should install an instruction selector pass,
  /// which converts from LLVM code to machine instructions.
  virtual bool addInstSelector() {
    return true;
  }

  /// Add the complete, standard set of LLVM CodeGen passes.
  /// Fully developed targets will not generally override this.
  virtual void addMachinePasses();

  /// Create an instance of ScheduleDAGInstrs to be run within the standard
  /// MachineScheduler pass for this function and target at the current
  /// optimization level.
  ///
  /// This can also be used to plug a new MachineSchedStrategy into an instance
  /// of the standard ScheduleDAGMI:
  ///   return new ScheduleDAGMI(C, make_unique<MyStrategy>(C), /* IsPostRA= */false)
  ///
  /// Return NULL to select the default (generic) machine scheduler.
  virtual ScheduleDAGInstrs *
  createMachineScheduler(MachineSchedContext *C) const {
    return nullptr;
  }

  /// Similar to createMachineScheduler but used when postRA machine scheduling
  /// is enabled.
  virtual ScheduleDAGInstrs *
  createPostMachineScheduler(MachineSchedContext *C) const {
    return nullptr;
  }

protected:
  // Helper to verify the analysis is really immutable.
  void setOpt(bool &Opt, bool Val);

  /// Methods with trivial inline returns are convenient points in the common
  /// codegen pass pipeline where targets may insert passes. Methods with
  /// out-of-line standard implementations are major CodeGen stages called by
  /// addMachinePasses. Some targets may override major stages when inserting
  /// passes is insufficient, but maintaining overriden stages is more work.
  ///

  /// addPreISelPasses - This method should add any "last minute" LLVM->LLVM
  /// passes (which are run just before instruction selector).
  virtual bool addPreISel() {
    return true;
  }

  /// addMachineSSAOptimization - Add standard passes that optimize machine
  /// instructions in SSA form.
  virtual void addMachineSSAOptimization();

  /// Add passes that optimize instruction level parallelism for out-of-order
  /// targets. These passes are run while the machine code is still in SSA
  /// form, so they can use MachineTraceMetrics to control their heuristics.
  ///
  /// All passes added here should preserve the MachineDominatorTree,
  /// MachineLoopInfo, and MachineTraceMetrics analyses.
  virtual bool addILPOpts() {
    return false;
  }

  /// This method may be implemented by targets that want to run passes
  /// immediately before register allocation.
  virtual void addPreRegAlloc() { }

  /// createTargetRegisterAllocator - Create the register allocator pass for
  /// this target at the current optimization level.
  virtual FunctionPass *createTargetRegisterAllocator(bool Optimized);

  /// addFastRegAlloc - Add the minimum set of target-independent passes that
  /// are required for fast register allocation.
  virtual void addFastRegAlloc(FunctionPass *RegAllocPass);

  /// addOptimizedRegAlloc - Add passes related to register allocation.
  /// LLVMTargetMachine provides standard regalloc passes for most targets.
  virtual void addOptimizedRegAlloc(FunctionPass *RegAllocPass);

  /// addPreRewrite - Add passes to the optimized register allocation pipeline
  /// after register allocation is complete, but before virtual registers are
  /// rewritten to physical registers.
  ///
  /// These passes must preserve VirtRegMap and LiveIntervals, and when running
  /// after RABasic or RAGreedy, they should take advantage of LiveRegMatrix.
  /// When these passes run, VirtRegMap contains legal physreg assignments for
  /// all virtual registers.
  virtual bool addPreRewrite() {
    return false;
  }

  /// This method may be implemented by targets that want to run passes after
  /// register allocation pass pipeline but before prolog-epilog insertion.
  virtual void addPostRegAlloc() { }

  /// Add passes that optimize machine instructions after register allocation.
  virtual void addMachineLateOptimization();

  /// This method may be implemented by targets that want to run passes after
  /// prolog-epilog insertion and before the second instruction scheduling pass.
  virtual void addPreSched2() { }

  /// addGCPasses - Add late codegen passes that analyze code for garbage
  /// collection. This should return true if GC info should be printed after
  /// these passes.
  virtual bool addGCPasses();

  /// Add standard basic block placement passes.
  virtual void addBlockPlacement();

  /// This pass may be implemented by targets that want to run passes
  /// immediately before machine code is emitted.
  virtual void addPreEmitPass() { }

  /// Utilities for targets to add passes to the pass manager.
  ///

  /// Add a CodeGen pass at this point in the pipeline after checking overrides.
  /// Return the pass that was added, or zero if no pass was added.
  /// @p printAfter    if true and adding a machine function pass add an extra
  ///                  machine printer pass afterwards
  /// @p verifyAfter   if true and adding a machine function pass add an extra
  ///                  machine verification pass afterwards.
  AnalysisID addPass(AnalysisID PassID, bool verifyAfter = true,
                     bool printAfter = true);

  /// Add a pass to the PassManager if that pass is supposed to be run, as
  /// determined by the StartAfter and StopAfter options. Takes ownership of the
  /// pass.
  /// @p printAfter    if true and adding a machine function pass add an extra
  ///                  machine printer pass afterwards
  /// @p verifyAfter   if true and adding a machine function pass add an extra
  ///                  machine verification pass afterwards.
  void addPass(Pass *P, bool verifyAfter = true, bool printAfter = true);

  /// addMachinePasses helper to create the target-selected or overriden
  /// regalloc pass.
  FunctionPass *createRegAllocPass(bool Optimized);

  /// printAndVerify - Add a pass to dump then verify the machine function, if
  /// those steps are enabled.
  ///
  void printAndVerify(const std::string &Banner);

  /// Add a pass to print the machine function if printing is enabled.
  void addPrintPass(const std::string &Banner);

  /// Add a pass to perform basic verification of the machine function if
  /// verification is enabled.
  void addVerifyPass(const std::string &Banner);
};
} // namespace llvm

/// List of target independent CodeGen pass IDs.
namespace llvm {
  FunctionPass *createAtomicExpandPass(const TargetMachine *TM);

  /// createUnreachableBlockEliminationPass - The LLVM code generator does not
  /// work well with unreachable basic blocks (what live ranges make sense for a
  /// block that cannot be reached?).  As such, a code generator should either
  /// not instruction select unreachable blocks, or run this pass as its
  /// last LLVM modifying pass to clean up blocks that are not reachable from
  /// the entry block.
  FunctionPass *createUnreachableBlockEliminationPass();

  /// MachineFunctionPrinter pass - This pass prints out the machine function to
  /// the given stream as a debugging tool.
  MachineFunctionPass *
  createMachineFunctionPrinterPass(raw_ostream &OS,
                                   const std::string &Banner ="");

  /// MIRPrinting pass - this pass prints out the LLVM IR into the given stream
  /// using the MIR serialization format.
  MachineFunctionPass *createPrintMIRPass(raw_ostream &OS);

  /// createCodeGenPreparePass - Transform the code to expose more pattern
  /// matching during instruction selection.
  FunctionPass *createCodeGenPreparePass(const TargetMachine *TM = nullptr);

  /// AtomicExpandID -- Lowers atomic operations in terms of either cmpxchg
  /// load-linked/store-conditional loops.
  extern char &AtomicExpandID;

  /// MachineLoopInfo - This pass is a loop analysis pass.
  extern char &MachineLoopInfoID;

  /// MachineDominators - This pass is a machine dominators analysis pass.
  extern char &MachineDominatorsID;

/// MachineDominanaceFrontier - This pass is a machine dominators analysis pass.
  extern char &MachineDominanceFrontierID;

  /// EdgeBundles analysis - Bundle machine CFG edges.
  extern char &EdgeBundlesID;

  /// LiveVariables pass - This pass computes the set of blocks in which each
  /// variable is life and sets machine operand kill flags.
  extern char &LiveVariablesID;

  /// PHIElimination - This pass eliminates machine instruction PHI nodes
  /// by inserting copy instructions.  This destroys SSA information, but is the
  /// desired input for some register allocators.  This pass is "required" by
  /// these register allocator like this: AU.addRequiredID(PHIEliminationID);
  extern char &PHIEliminationID;

  /// LiveIntervals - This analysis keeps track of the live ranges of virtual
  /// and physical registers.
  extern char &LiveIntervalsID;

  /// LiveStacks pass. An analysis keeping track of the liveness of stack slots.
  extern char &LiveStacksID;

  /// TwoAddressInstruction - This pass reduces two-address instructions to
  /// use two operands. This destroys SSA information but it is desired by
  /// register allocators.
  extern char &TwoAddressInstructionPassID;

  /// ProcessImpicitDefs pass - This pass removes IMPLICIT_DEFs.
  extern char &ProcessImplicitDefsID;

  /// RegisterCoalescer - This pass merges live ranges to eliminate copies.
  extern char &RegisterCoalescerID;

  /// MachineScheduler - This pass schedules machine instructions.
  extern char &MachineSchedulerID;

  /// PostMachineScheduler - This pass schedules machine instructions postRA.
  extern char &PostMachineSchedulerID;

  /// SpillPlacement analysis. Suggest optimal placement of spill code between
  /// basic blocks.
  extern char &SpillPlacementID;

  /// ShrinkWrap pass. Look for the best place to insert save and restore
  // instruction and update the MachineFunctionInfo with that information.
  extern char &ShrinkWrapID;

  /// VirtRegRewriter pass. Rewrite virtual registers to physical registers as
  /// assigned in VirtRegMap.
  extern char &VirtRegRewriterID;

  /// UnreachableMachineBlockElimination - This pass removes unreachable
  /// machine basic blocks.
  extern char &UnreachableMachineBlockElimID;

  /// DeadMachineInstructionElim - This pass removes dead machine instructions.
  extern char &DeadMachineInstructionElimID;

  /// FastRegisterAllocation Pass - This pass register allocates as fast as
  /// possible. It is best suited for debug code where live ranges are short.
  ///
  FunctionPass *createFastRegisterAllocator();

  /// BasicRegisterAllocation Pass - This pass implements a degenerate global
  /// register allocator using the basic regalloc framework.
  ///
  FunctionPass *createBasicRegisterAllocator();

  /// Greedy register allocation pass - This pass implements a global register
  /// allocator for optimized builds.
  ///
  FunctionPass *createGreedyRegisterAllocator();

  /// PBQPRegisterAllocation Pass - This pass implements the Partitioned Boolean
  /// Quadratic Prograaming (PBQP) based register allocator.
  ///
  FunctionPass *createDefaultPBQPRegisterAllocator();

  /// PrologEpilogCodeInserter - This pass inserts prolog and epilog code,
  /// and eliminates abstract frame references.
  extern char &PrologEpilogCodeInserterID;

  /// ExpandPostRAPseudos - This pass expands pseudo instructions after
  /// register allocation.
  extern char &ExpandPostRAPseudosID;

  /// createPostRAScheduler - This pass performs post register allocation
  /// scheduling.
  extern char &PostRASchedulerID;

  /// BranchFolding - This pass performs machine code CFG based
  /// optimizations to delete branches to branches, eliminate branches to
  /// successor blocks (creating fall throughs), and eliminating branches over
  /// branches.
  extern char &BranchFolderPassID;

  /// MachineFunctionPrinterPass - This pass prints out MachineInstr's.
  extern char &MachineFunctionPrinterPassID;

  /// MIRPrintingPass - this pass prints out the LLVM IR using the MIR
  /// serialization format.
  extern char &MIRPrintingPassID;

  /// TailDuplicate - Duplicate blocks with unconditional branches
  /// into tails of their predecessors.
  extern char &TailDuplicateID;

  /// MachineTraceMetrics - This pass computes critical path and CPU resource
  /// usage in an ensemble of traces.
  extern char &MachineTraceMetricsID;

  /// EarlyIfConverter - This pass performs if-conversion on SSA form by
  /// inserting cmov instructions.
  extern char &EarlyIfConverterID;

  /// This pass performs instruction combining using trace metrics to estimate
  /// critical-path and resource depth.
  extern char &MachineCombinerID;

  /// StackSlotColoring - This pass performs stack coloring and merging.
  /// It merges disjoint allocas to reduce the stack size.
  extern char &StackColoringID;

  /// IfConverter - This pass performs machine code if conversion.
  extern char &IfConverterID;

  FunctionPass *createIfConverter(std::function<bool(const Function &)> Ftor);

  /// MachineBlockPlacement - This pass places basic blocks based on branch
  /// probabilities.
  extern char &MachineBlockPlacementID;

  /// MachineBlockPlacementStats - This pass collects statistics about the
  /// basic block placement using branch probabilities and block frequency
  /// information.
  extern char &MachineBlockPlacementStatsID;

  /// GCLowering Pass - Used by gc.root to perform its default lowering
  /// operations.
  FunctionPass *createGCLoweringPass();

  /// ShadowStackGCLowering - Implements the custom lowering mechanism
  /// used by the shadow stack GC.  Only runs on functions which opt in to
  /// the shadow stack collector.
  FunctionPass *createShadowStackGCLoweringPass();

  /// GCMachineCodeAnalysis - Target-independent pass to mark safe points
  /// in machine code. Must be added very late during code generation, just
  /// prior to output, and importantly after all CFG transformations (such as
  /// branch folding).
  extern char &GCMachineCodeAnalysisID;

  /// Creates a pass to print GC metadata.
  ///
  FunctionPass *createGCInfoPrinter(raw_ostream &OS);

  /// MachineCSE - This pass performs global CSE on machine instructions.
  extern char &MachineCSEID;

  /// ImplicitNullChecks - This pass folds null pointer checks into nearby
  /// memory operations.
  extern char &ImplicitNullChecksID;

  /// MachineLICM - This pass performs LICM on machine instructions.
  extern char &MachineLICMID;

  /// MachineSinking - This pass performs sinking on machine instructions.
  extern char &MachineSinkingID;

  /// MachineCopyPropagation - This pass performs copy propagation on
  /// machine instructions.
  extern char &MachineCopyPropagationID;

  /// PeepholeOptimizer - This pass performs peephole optimizations -
  /// like extension and comparison eliminations.
  extern char &PeepholeOptimizerID;

  /// OptimizePHIs - This pass optimizes machine instruction PHIs
  /// to take advantage of opportunities created during DAG legalization.
  extern char &OptimizePHIsID;

  /// StackSlotColoring - This pass performs stack slot coloring.
  extern char &StackSlotColoringID;

  /// createStackProtectorPass - This pass adds stack protectors to functions.
  ///
  FunctionPass *createStackProtectorPass(const TargetMachine *TM);

  /// createMachineVerifierPass - This pass verifies cenerated machine code
  /// instructions for correctness.
  ///
  FunctionPass *createMachineVerifierPass(const std::string& Banner);

  /// createDwarfEHPass - This pass mulches exception handling code into a form
  /// adapted to code generation.  Required if using dwarf exception handling.
  FunctionPass *createDwarfEHPass(const TargetMachine *TM);

  /// createWinEHPass - Prepares personality functions used by MSVC on Windows,
  /// in addition to the Itanium LSDA based personalities.
  FunctionPass *createWinEHPass(const TargetMachine *TM);

  /// createSjLjEHPreparePass - This pass adapts exception handling code to use
  /// the GCC-style builtin setjmp/longjmp (sjlj) to handling EH control flow.
  ///
  FunctionPass *createSjLjEHPreparePass();

  /// LocalStackSlotAllocation - This pass assigns local frame indices to stack
  /// slots relative to one another and allocates base registers to access them
  /// when it is estimated by the target to be out of range of normal frame
  /// pointer or stack pointer index addressing.
  extern char &LocalStackSlotAllocationID;

  /// ExpandISelPseudos - This pass expands pseudo-instructions.
  extern char &ExpandISelPseudosID;

  /// createExecutionDependencyFixPass - This pass fixes execution time
  /// problems with dependent instructions, such as switching execution
  /// domains to match.
  ///
  /// The pass will examine instructions using and defining registers in RC.
  ///
  FunctionPass *createExecutionDependencyFixPass(const TargetRegisterClass *RC);

  /// UnpackMachineBundles - This pass unpack machine instruction bundles.
  extern char &UnpackMachineBundlesID;

  FunctionPass *
  createUnpackMachineBundles(std::function<bool(const Function &)> Ftor);

  /// FinalizeMachineBundles - This pass finalize machine instruction
  /// bundles (created earlier, e.g. during pre-RA scheduling).
  extern char &FinalizeMachineBundlesID;

  /// StackMapLiveness - This pass analyses the register live-out set of
  /// stackmap/patchpoint intrinsics and attaches the calculated information to
  /// the intrinsic for later emission to the StackMap.
  extern char &StackMapLivenessID;

  /// createJumpInstrTables - This pass creates jump-instruction tables.
  ModulePass *createJumpInstrTablesPass();

  /// createForwardControlFlowIntegrityPass - This pass adds control-flow
  /// integrity.
  ModulePass *createForwardControlFlowIntegrityPass();

  /// InterleavedAccess Pass - This pass identifies and matches interleaved
  /// memory accesses to target specific intrinsics.
  ///
  FunctionPass *createInterleavedAccessPass(const TargetMachine *TM);
} // End llvm namespace

/// Target machine pass initializer for passes with dependencies. Use with
/// INITIALIZE_TM_PASS_END.
#define INITIALIZE_TM_PASS_BEGIN INITIALIZE_PASS_BEGIN

/// Target machine pass initializer for passes with dependencies. Use with
/// INITIALIZE_TM_PASS_BEGIN.
#define INITIALIZE_TM_PASS_END(passName, arg, name, cfg, analysis) \
    PassInfo *PI = new PassInfo(name, arg, & passName ::ID, \
      PassInfo::NormalCtor_t(callDefaultCtor< passName >), cfg, analysis, \
      PassInfo::TargetMachineCtor_t(callTargetMachineCtor< passName >)); \
    Registry.registerPass(*PI, true); \
    return PI; \
  } \
  void llvm::initialize##passName##Pass(PassRegistry &Registry) { \
    CALL_ONCE_INITIALIZATION(initialize##passName##PassOnce) \
  }

/// This initializer registers TargetMachine constructor, so the pass being
/// initialized can use target dependent interfaces. Please do not move this
/// macro to be together with INITIALIZE_PASS, which is a complete target
/// independent initializer, and we don't want to make libScalarOpts depend
/// on libCodeGen.
#define INITIALIZE_TM_PASS(passName, arg, name, cfg, analysis) \
    INITIALIZE_TM_PASS_BEGIN(passName, arg, name, cfg, analysis) \
    INITIALIZE_TM_PASS_END(passName, arg, name, cfg, analysis)

#endif

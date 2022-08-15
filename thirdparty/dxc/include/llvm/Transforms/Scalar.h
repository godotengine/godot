//===-- Scalar.h - Scalar Transformations -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for accessor functions that expose passes
// in the Scalar transformations library.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_SCALAR_H
#define LLVM_TRANSFORMS_SCALAR_H

#include "llvm/ADT/StringRef.h"
#include <functional>

namespace llvm {

class BasicBlockPass;
class Function;
class FunctionPass;
class ModulePass;
class Pass;
class GetElementPtrInst;
class PassInfo;
class TerminatorInst;
class TargetLowering;
class TargetMachine;
class PassRegistry; // HLSL Change - need this for registrations

//===----------------------------------------------------------------------===//
//
// ConstantPropagation - A worklist driven constant propagation pass
//
FunctionPass *createConstantPropagationPass();

//===----------------------------------------------------------------------===//
//
// AlignmentFromAssumptions - Use assume intrinsics to set load/store
// alignments.
//
FunctionPass *createAlignmentFromAssumptionsPass();

//===----------------------------------------------------------------------===//
//
// SCCP - Sparse conditional constant propagation.
//
FunctionPass *createSCCPPass();

//===----------------------------------------------------------------------===//
//
// DeadInstElimination - This pass quickly removes trivially dead instructions
// without modifying the CFG of the function.  It is a BasicBlockPass, so it
// runs efficiently when queued next to other BasicBlockPass's.
//
Pass *createDeadInstEliminationPass();

//===----------------------------------------------------------------------===//
//
// DeadCodeElimination - This pass is more powerful than DeadInstElimination,
// because it is worklist driven that can potentially revisit instructions when
// their other instructions become dead, to eliminate chains of dead
// computations.
//
FunctionPass *createDeadCodeEliminationPass();

//===----------------------------------------------------------------------===//
//
// DeadStoreElimination - This pass deletes stores that are post-dominated by
// must-aliased stores and are not loaded used between the stores.
//
FunctionPass *createDeadStoreEliminationPass(unsigned ScanLimit = 0); // HLSL Change - Add ScanLimit

//===----------------------------------------------------------------------===//
//
// AggressiveDCE - This pass uses the SSA based Aggressive DCE algorithm.  This
// algorithm assumes instructions are dead until proven otherwise, which makes
// it more successful are removing non-obviously dead instructions.
//
FunctionPass *createAggressiveDCEPass();

//===----------------------------------------------------------------------===//
//
// BitTrackingDCE - This pass uses a bit-tracking DCE algorithm in order to
// remove computations of dead bits.
//
FunctionPass *createBitTrackingDCEPass();

//===----------------------------------------------------------------------===//
//
// SROA - Replace aggregates or pieces of aggregates with scalar SSA values.
//
FunctionPass *createSROAPass(bool RequiresDomTree = true,
                             bool SkipHLSLMat = true);

//===----------------------------------------------------------------------===//
//
// ScalarReplAggregates - Break up alloca's of aggregates into multiple allocas
// if possible.
//
FunctionPass *createScalarReplAggregatesPass(signed Threshold = -1,
                                             bool UseDomTree = true,
                                             signed StructMemberThreshold = -1,
                                             signed ArrayElementThreshold = -1,
                                             signed ScalarLoadThreshold = -1);
// HLSL Change Begins
FunctionPass* createHLExpandStoreIntrinsicsPass();
void initializeHLExpandStoreIntrinsicsPass(PassRegistry&);

//===----------------------------------------------------------------------===//
//
// ScalarReplAggregatesHLSL - Break up argument's of aggregates into multiple arguments
// for hlsl. Array will not change, all structures will be broken up.
//
ModulePass *createSROA_Parameter_HLSL();
void initializeSROA_Parameter_HLSLPass(PassRegistry&);

//===----------------------------------------------------------------------===//
//
// Cleans up constant stores that didn't get a chance to be turned into initializers
//
Pass *createDxilFixConstArrayInitializerPass();
void initializeDxilFixConstArrayInitializerPass(PassRegistry&);

Pass *createDxilConditionalMem2RegPass(bool NoOpt);
void initializeDxilConditionalMem2RegPass(PassRegistry&);

Pass *createDxilLoopUnrollPass(unsigned MaxIterationAttempt, bool OnlyWarnOnFail, bool StructurizeLoopExits);
void initializeDxilLoopUnrollPass(PassRegistry&);

Pass *createDxilEraseDeadRegionPass();
void initializeDxilEraseDeadRegionPass(PassRegistry&);

Pass *createDxilEliminateVectorPass();
void initializeDxilEliminateVectorPass(PassRegistry&);

Pass *createDxilInsertPreservesPass(bool AllowPreserves);
void initializeDxilInsertPreservesPass(PassRegistry&);

Pass *createDxilFinalizePreservesPass();
void initializeDxilFinalizePreservesPass(PassRegistry&);

Pass *createDxilPreserveToSelectPass();
void initializeDxilPreserveToSelectPass(PassRegistry&);

Pass *createDxilRemoveDeadBlocksPass();
void initializeDxilRemoveDeadBlocksPass(PassRegistry&);

void initializeDxilRewriteOutputArgDebugInfoPass(PassRegistry&);
Pass *createDxilRewriteOutputArgDebugInfoPass();

//===----------------------------------------------------------------------===//
//
// LowerStaticGlobalIntoAlloca. Replace static globals with alloca if only used
// in one function.
//
ModulePass *createLowerStaticGlobalIntoAlloca();
void initializeLowerStaticGlobalIntoAllocaPass(PassRegistry&);

//===----------------------------------------------------------------------===//
//
// DynamicIndexingVectorToArray 
// Replace vector with array if it has dynamic indexing.
//
ModulePass *createDynamicIndexingVectorToArrayPass(bool ReplaceAllVector = false);
void initializeDynamicIndexingVectorToArrayPass(PassRegistry&);
//===----------------------------------------------------------------------===//
// Flatten multi dim array into 1 dim.
//
ModulePass *createMultiDimArrayToOneDimArrayPass();
void initializeMultiDimArrayToOneDimArrayPass(PassRegistry&);
//===----------------------------------------------------------------------===//
// Flatten resource into handle.
//
ModulePass *createResourceToHandlePass();
void initializeResourceToHandlePass(PassRegistry&);

//===----------------------------------------------------------------------===//
// Hoist a local array initialized with constant values to a global array with
// a constant initializer.
//
ModulePass *createHoistConstantArrayPass();
void initializeHoistConstantArrayPass(PassRegistry&);
// HLSL Change Ends

//===----------------------------------------------------------------------===//
//
// InductiveRangeCheckElimination - Transform loops to elide range checks on
// linear functions of the induction variable.
//
Pass *createInductiveRangeCheckEliminationPass();

//===----------------------------------------------------------------------===//
//
// InductionVariableSimplify - Transform induction variables in a program to all
// use a single canonical induction variable per loop.
//
Pass *createIndVarSimplifyPass();

//===----------------------------------------------------------------------===//
//
// InstructionCombining - Combine instructions to form fewer, simple
// instructions. This pass does not modify the CFG, and has a tendency to make
// instructions dead, so a subsequent DCE pass is useful.
//
// This pass combines things like:
//    %Y = add int 1, %X
//    %Z = add int 1, %Y
// into:
//    %Z = add int 2, %X
//
FunctionPass *createInstructionCombiningPass();

//===----------------------------------------------------------------------===//
//
// LICM - This pass is a loop invariant code motion and memory promotion pass.
//
Pass *createLICMPass();

//===----------------------------------------------------------------------===//
//
// LoopInterchange - This pass interchanges loops to provide a more
// cache-friendly memory access patterns.
//
Pass *createLoopInterchangePass();

//===----------------------------------------------------------------------===//
//
// LoopStrengthReduce - This pass is strength reduces GEP instructions that use
// a loop's canonical induction variable as one of their indices.
//
Pass *createLoopStrengthReducePass();

//===----------------------------------------------------------------------===//
//
// GlobalMerge - This pass merges internal (by default) globals into structs
// to enable reuse of a base pointer by indexed addressing modes.
// It can also be configured to focus on size optimizations only.
//
Pass *createGlobalMergePass(const TargetMachine *TM, unsigned MaximalOffset,
                            bool OnlyOptimizeForSize = false);

//===----------------------------------------------------------------------===//
//
// LoopUnswitch - This pass is a simple loop unswitching pass.
//
Pass *createLoopUnswitchPass(bool OptimizeForSize = false);

//===----------------------------------------------------------------------===//
//
// LoopInstSimplify - This pass simplifies instructions in a loop's body.
//
Pass *createLoopInstSimplifyPass();

//===----------------------------------------------------------------------===//
//
// LoopUnroll - This pass is a simple loop unrolling pass.
//
Pass *createLoopUnrollPass(int Threshold = -1, int Count = -1,
                           int AllowPartial = -1, int Runtime = -1,
                           bool StructurizeLoopExits = false // HLSL Change
                          );
// Create an unrolling pass for full unrolling only.
Pass *createSimpleLoopUnrollPass();

//===----------------------------------------------------------------------===//
//
// LoopReroll - This pass is a simple loop rerolling pass.
//
Pass *createLoopRerollPass();

//===----------------------------------------------------------------------===//
//
// LoopRotate - This pass is a simple loop rotating pass.
//
Pass *createLoopRotatePass(int MaxHeaderSize = -1);

//===----------------------------------------------------------------------===//
//
// LoopIdiom - This pass recognizes and replaces idioms in loops.
//
Pass *createLoopIdiomPass();

//===----------------------------------------------------------------------===//
//
// PromoteMemoryToRegister - This pass is used to promote memory references to
// be register references. A simple example of the transformation performed by
// this pass is:
//
//        FROM CODE                           TO CODE
//   %X = alloca i32, i32 1                 ret i32 42
//   store i32 42, i32 *%X
//   %Y = load i32* %X
//   ret i32 %Y
//
FunctionPass *createPromoteMemoryToRegisterPass();

//===----------------------------------------------------------------------===//
//
// DemoteRegisterToMemoryPass - This pass is used to demote registers to memory
// references. In basically undoes the PromoteMemoryToRegister pass to make cfg
// hacking easier.
//
FunctionPass *createDemoteRegisterToMemoryPass();
extern char &DemoteRegisterToMemoryID;

// HLSL Change start
// Relaxed version (for HLSL usage).
FunctionPass *createDemoteRegisterToMemoryHlslPass();
extern char &DemoteRegisterToMemoryHlslID;
// HLSL Change end

//===----------------------------------------------------------------------===//
//
// Reassociate - This pass reassociates commutative expressions in an order that
// is designed to promote better constant propagation, GCSE, LICM, PRE...
//
// For example:  4 + (x + 5)  ->  x + (4 + 5)
//
FunctionPass *createReassociatePass();

//===----------------------------------------------------------------------===//
//
// JumpThreading - Thread control through mult-pred/multi-succ blocks where some
// preds always go to some succ. Thresholds other than minus one override the
// internal BB duplication default threshold.
//
FunctionPass *createJumpThreadingPass(int Threshold = -1);

//===----------------------------------------------------------------------===//
//
// CFGSimplification - Merge basic blocks, eliminate unreachable blocks,
// simplify terminator instructions, etc...
//
FunctionPass *createCFGSimplificationPass(
    int Threshold = -1, std::function<bool(const Function &)> Ftor = nullptr);

//===----------------------------------------------------------------------===//
//
// FlattenCFG - flatten CFG, reduce number of conditional branches by using
// parallel-and and parallel-or mode, etc...
//
FunctionPass *createFlattenCFGPass();

//===----------------------------------------------------------------------===//
//
// CFG Structurization - Remove irreducible control flow
//
Pass *createStructurizeCFGPass();

//===----------------------------------------------------------------------===//
//
// BreakCriticalEdges - Break all of the critical edges in the CFG by inserting
// a dummy basic block. This pass may be "required" by passes that cannot deal
// with critical edges. For this usage, a pass must call:
//
//   AU.addRequiredID(BreakCriticalEdgesID);
//
// This pass obviously invalidates the CFG, but can update forward dominator
// (set, immediate dominators, tree, and frontier) information.
//
FunctionPass *createBreakCriticalEdgesPass();
extern char &BreakCriticalEdgesID;

//===----------------------------------------------------------------------===//
//
// LoopSimplify - Insert Pre-header blocks into the CFG for every function in
// the module.  This pass updates dominator information, loop information, and
// does not add critical edges to the CFG.
//
//   AU.addRequiredID(LoopSimplifyID);
//
Pass *createLoopSimplifyPass();
extern char &LoopSimplifyID;

//===----------------------------------------------------------------------===//
//
// TailCallElimination - This pass eliminates call instructions to the current
// function which occur immediately before return instructions.
//
FunctionPass *createTailCallEliminationPass();

//===----------------------------------------------------------------------===//
//
// LowerSwitch - This pass converts SwitchInst instructions into a sequence of
// chained binary branch instructions.
//
FunctionPass *createLowerSwitchPass();
extern char &LowerSwitchID;

//===----------------------------------------------------------------------===//
//
// LowerInvoke - This pass removes invoke instructions, converting them to call
// instructions.
//
FunctionPass *createLowerInvokePass();
extern char &LowerInvokePassID;

//===----------------------------------------------------------------------===//
//
// LCSSA - This pass inserts phi nodes at loop boundaries to simplify other loop
// optimizations.
//
Pass *createLCSSAPass();
extern char &LCSSAID;

//===----------------------------------------------------------------------===//
//
// EarlyCSE - This pass performs a simple and fast CSE pass over the dominator
// tree.
//
FunctionPass *createEarlyCSEPass();

//===----------------------------------------------------------------------===//
//
// MergedLoadStoreMotion - This pass merges loads and stores in diamonds. Loads
// are hoisted into the header, while stores sink into the footer.
//
FunctionPass *createMergedLoadStoreMotionPass();

//===----------------------------------------------------------------------===//
//
// GVN - This pass performs global value numbering and redundant load
// elimination cotemporaneously.
//
FunctionPass *createGVNPass(bool NoLoads = false);

//===----------------------------------------------------------------------===//
//
// MemCpyOpt - This pass performs optimizations related to eliminating memcpy
// calls and/or combining multiple stores into memset's.
//
FunctionPass *createMemCpyOptPass();

//===----------------------------------------------------------------------===//
//
// LoopDeletion - This pass performs DCE of non-infinite loops that it
// can prove are dead.
//
Pass *createLoopDeletionPass();

//===----------------------------------------------------------------------===//
//
// ConstantHoisting - This pass prepares a function for expensive constants.
//
FunctionPass *createConstantHoistingPass();

//===----------------------------------------------------------------------===//
//
// InstructionNamer - Give any unnamed non-void instructions "tmp" names.
//
FunctionPass *createInstructionNamerPass();
extern char &InstructionNamerID;

//===----------------------------------------------------------------------===//
//
// Sink - Code Sinking
//
FunctionPass *createSinkingPass();

//===----------------------------------------------------------------------===//
//
// LowerAtomic - Lower atomic intrinsics to non-atomic form
//
Pass *createLowerAtomicPass();

//===----------------------------------------------------------------------===//
//
// ValuePropagation - Propagate CFG-derived value information
//
Pass *createCorrelatedValuePropagationPass();

//===----------------------------------------------------------------------===//
//
// InstructionSimplifier - Remove redundant instructions.
//
FunctionPass *createInstructionSimplifierPass();
extern char &InstructionSimplifierID;

//===----------------------------------------------------------------------===//
//
// LowerExpectIntrinsics - Removes llvm.expect intrinsics and creates
// "block_weights" metadata.
FunctionPass *createLowerExpectIntrinsicPass();

//===----------------------------------------------------------------------===//
//
// PartiallyInlineLibCalls - Tries to inline the fast path of library
// calls such as sqrt.
//
FunctionPass *createPartiallyInlineLibCallsPass();

//===----------------------------------------------------------------------===//
//
// SampleProfilePass - Loads sample profile data from disk and generates
// IR metadata to reflect the profile.
FunctionPass *createSampleProfileLoaderPass();
FunctionPass *createSampleProfileLoaderPass(StringRef Name);

//===----------------------------------------------------------------------===//
//
// ScalarizerPass - Converts vector operations into scalar operations
//
FunctionPass *createScalarizerPass();
FunctionPass *createScalarizerPass(bool NoOpt);

//===----------------------------------------------------------------------===//
//
// AddDiscriminators - Add DWARF path discriminators to the IR.
FunctionPass *createAddDiscriminatorsPass();

//===----------------------------------------------------------------------===//
//
// SeparateConstOffsetFromGEP - Split GEPs for better CSE
//
FunctionPass *
createSeparateConstOffsetFromGEPPass(const TargetMachine *TM = nullptr,
                                     bool LowerGEP = false);

//===----------------------------------------------------------------------===//
//
// SpeculativeExecution - Aggressively hoist instructions to enable
// speculative execution on targets where branches are expensive.
//
FunctionPass *createSpeculativeExecutionPass();

//===----------------------------------------------------------------------===//
//
// LoadCombine - Combine loads into bigger loads.
//
BasicBlockPass *createLoadCombinePass();

//===----------------------------------------------------------------------===//
//
// StraightLineStrengthReduce - This pass strength-reduces some certain
// instruction patterns in straight-line code.
//
FunctionPass *createStraightLineStrengthReducePass();

//===----------------------------------------------------------------------===//
//
// PlaceSafepoints - Rewrite any IR calls to gc.statepoints and insert any
// safepoint polls (method entry, backedge) that might be required.  This pass
// does not generate explicit relocation sequences - that's handled by
// RewriteStatepointsForGC which can be run at an arbitrary point in the pass
// order following this pass.
//
FunctionPass *createPlaceSafepointsPass();

//===----------------------------------------------------------------------===//
//
// RewriteStatepointsForGC - Rewrite any gc.statepoints which do not yet have
// explicit relocations to include explicit relocations.
//
ModulePass *createRewriteStatepointsForGCPass();

//===----------------------------------------------------------------------===//
//
// Float2Int - Demote floats to ints where possible.
//
FunctionPass *createFloat2IntPass();

//===----------------------------------------------------------------------===//
//
// NaryReassociate - Simplify n-ary operations by reassociation.
//
FunctionPass *createNaryReassociatePass();
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
//
// LoopDistribute - Distribute loops.
//
FunctionPass *createLoopDistributePass();

} // End llvm namespace

#endif

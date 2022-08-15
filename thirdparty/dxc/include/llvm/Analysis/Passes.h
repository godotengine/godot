//===-- llvm/Analysis/Passes.h - Constructors for analyses ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for accessor functions that expose passes
// in the analysis libraries.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_PASSES_H
#define LLVM_ANALYSIS_PASSES_H

namespace llvm {
  class FunctionPass;
  class ImmutablePass;
  class LoopPass;
  class ModulePass;
  class Pass;
  class PassInfo;
  class LibCallInfo;

  //===--------------------------------------------------------------------===//
  //
  // createGlobalsModRefPass - This pass provides alias and mod/ref info for
  // global values that do not have their addresses taken.
  //
  Pass *createGlobalsModRefPass();

  //===--------------------------------------------------------------------===//
  //
  // createAliasDebugger - This pass helps debug clients of AA
  //
  Pass *createAliasDebugger();

  //===--------------------------------------------------------------------===//
  //
  // createAliasAnalysisCounterPass - This pass counts alias queries and how the
  // alias analysis implementation responds.
  //
  ModulePass *createAliasAnalysisCounterPass();

  //===--------------------------------------------------------------------===//
  //
  // createAAEvalPass - This pass implements a simple N^2 alias analysis
  // accuracy evaluator.
  //
  FunctionPass *createAAEvalPass();

  //===--------------------------------------------------------------------===//
  //
  // createNoAAPass - This pass implements a "I don't know" alias analysis.
  //
  ImmutablePass *createNoAAPass();

  //===--------------------------------------------------------------------===//
  //
  // createBasicAliasAnalysisPass - This pass implements the stateless alias
  // analysis.
  //
  ImmutablePass *createBasicAliasAnalysisPass();

  //===--------------------------------------------------------------------===//
  //
  // createCFLAliasAnalysisPass - This pass implements a set-based approach to
  // alias analysis.
  //
  ImmutablePass *createCFLAliasAnalysisPass();

  //===--------------------------------------------------------------------===//
  //
  /// createLibCallAliasAnalysisPass - Create an alias analysis pass that knows
  /// about the semantics of a set of libcalls specified by LCI.  The newly
  /// constructed pass takes ownership of the pointer that is provided.
  ///
  FunctionPass *createLibCallAliasAnalysisPass(LibCallInfo *LCI);

  //===--------------------------------------------------------------------===//
  //
  // createScalarEvolutionAliasAnalysisPass - This pass implements a simple
  // alias analysis using ScalarEvolution queries.
  //
  FunctionPass *createScalarEvolutionAliasAnalysisPass();

  //===--------------------------------------------------------------------===//
  //
  // createTypeBasedAliasAnalysisPass - This pass implements metadata-based
  // type-based alias analysis.
  //
  ImmutablePass *createTypeBasedAliasAnalysisPass();

  //===--------------------------------------------------------------------===//
  //
  // createScopedNoAliasAAPass - This pass implements metadata-based
  // scoped noalias analysis.
  //
  ImmutablePass *createScopedNoAliasAAPass();

  //===--------------------------------------------------------------------===//
  //
  // createObjCARCAliasAnalysisPass - This pass implements ObjC-ARC-based
  // alias analysis.
  //
  ImmutablePass *createObjCARCAliasAnalysisPass();

  FunctionPass *createPAEvalPass();

  //===--------------------------------------------------------------------===//
  //
  /// createLazyValueInfoPass - This creates an instance of the LazyValueInfo
  /// pass.
  FunctionPass *createLazyValueInfoPass();

  //===--------------------------------------------------------------------===//
  //
  // createDependenceAnalysisPass - This creates an instance of the
  // DependenceAnalysis pass.
  //
  FunctionPass *createDependenceAnalysisPass();

  //===--------------------------------------------------------------------===//
  //
  // createCostModelAnalysisPass - This creates an instance of the
  // CostModelAnalysis pass.
  //
  FunctionPass *createCostModelAnalysisPass();

  //===--------------------------------------------------------------------===//
  //
  // createDelinearizationPass - This pass implements attempts to restore
  // multidimensional array indices from linearized expressions.
  //
  FunctionPass *createDelinearizationPass();

  //===--------------------------------------------------------------------===//
  //
  // createDivergenceAnalysisPass - This pass determines which branches in a GPU
  // program are divergent.
  //
  FunctionPass *createDivergenceAnalysisPass();

  //===--------------------------------------------------------------------===//
  //
  // Minor pass prototypes, allowing us to expose them through bugpoint and
  // analyze.
  FunctionPass *createInstCountPass();

  //===--------------------------------------------------------------------===//
  //
  // createRegionInfoPass - This pass finds all single entry single exit regions
  // in a function and builds the region hierarchy.
  //
  FunctionPass *createRegionInfoPass();

  // Print module-level debug info metadata in human-readable form.
  ModulePass *createModuleDebugInfoPrinterPass();

  //===--------------------------------------------------------------------===//
  //
  // createMemDepPrinter - This pass exhaustively collects all memdep
  // information and prints it with -analyze.
  //
  FunctionPass *createMemDepPrinter();

  //===--------------------------------------------------------------------===//
  //
  // createMemDerefPrinter - This pass collects memory dereferenceability
  // information and prints it with -analyze.
  //
  FunctionPass *createMemDerefPrinter();

}

#endif

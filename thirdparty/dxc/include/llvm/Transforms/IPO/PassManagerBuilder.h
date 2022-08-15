// llvm/Transforms/IPO/PassManagerBuilder.h - Build Standard Pass -*- C++ -*-=//
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

#ifndef LLVM_TRANSFORMS_IPO_PASSMANAGERBUILDER_H
#define LLVM_TRANSFORMS_IPO_PASSMANAGERBUILDER_H

#include <vector>

namespace hlsl {
  class HLSLExtensionsCodegenHelper;
}

namespace llvm {
class Pass;
class TargetLibraryInfoImpl;
class TargetMachine;

// The old pass manager infrastructure is hidden in a legacy namespace now.
namespace legacy {
class FunctionPassManager;
class PassManagerBase;
}

/// PassManagerBuilder - This class is used to set up a standard optimization
/// sequence for languages like C and C++, allowing some APIs to customize the
/// pass sequence in various ways. A simple example of using it would be:
///
///  PassManagerBuilder Builder;
///  Builder.OptLevel = 2;
///  Builder.populateFunctionPassManager(FPM);
///  Builder.populateModulePassManager(MPM);
///
/// In addition to setting up the basic passes, PassManagerBuilder allows
/// frontends to vend a plugin API, where plugins are allowed to add extensions
/// to the default pass manager.  They do this by specifying where in the pass
/// pipeline they want to be added, along with a callback function that adds
/// the pass(es).  For example, a plugin that wanted to add a loop optimization
/// could do something like this:
///
/// static void addMyLoopPass(const PMBuilder &Builder, PassManagerBase &PM) {
///   if (Builder.getOptLevel() > 2 && Builder.getOptSizeLevel() == 0)
///     PM.add(createMyAwesomePass());
/// }
///   ...
///   Builder.addExtension(PassManagerBuilder::EP_LoopOptimizerEnd,
///                        addMyLoopPass);
///   ...
class PassManagerBuilder {
public:
  /// Extensions are passed the builder itself (so they can see how it is
  /// configured) as well as the pass manager to add stuff to.
  typedef void (*ExtensionFn)(const PassManagerBuilder &Builder,
                              legacy::PassManagerBase &PM);
  enum ExtensionPointTy {
    /// EP_EarlyAsPossible - This extension point allows adding passes before
    /// any other transformations, allowing them to see the code as it is coming
    /// out of the frontend.
    EP_EarlyAsPossible,

    /// EP_ModuleOptimizerEarly - This extension point allows adding passes
    /// just before the main module-level optimization passes.
    EP_ModuleOptimizerEarly,

    /// EP_LoopOptimizerEnd - This extension point allows adding loop passes to
    /// the end of the loop optimizer.
    EP_LoopOptimizerEnd,

    /// EP_ScalarOptimizerLate - This extension point allows adding optimization
    /// passes after most of the main optimizations, but before the last
    /// cleanup-ish optimizations.
    EP_ScalarOptimizerLate,

    /// EP_OptimizerLast -- This extension point allows adding passes that
    /// run after everything else.
    EP_OptimizerLast,

    /// EP_EnabledOnOptLevel0 - This extension point allows adding passes that
    /// should not be disabled by O0 optimization level. The passes will be
    /// inserted after the inlining pass.
    EP_EnabledOnOptLevel0,

    /// EP_Peephole - This extension point allows adding passes that perform
    /// peephole optimizations similar to the instruction combiner. These passes
    /// will be inserted after each instance of the instruction combiner pass.
    EP_Peephole,
  };

  /// The Optimization Level - Specify the basic optimization level.
  ///    0 = -O0, 1 = -O1, 2 = -O2, 3 = -O3
  unsigned OptLevel;

  /// SizeLevel - How much we're optimizing for size.
  ///    0 = none, 1 = -Os, 2 = -Oz
  unsigned SizeLevel;

  /// LibraryInfo - Specifies information about the runtime library for the
  /// optimizer.  If this is non-null, it is added to both the function and
  /// per-module pass pipeline.
  TargetLibraryInfoImpl *LibraryInfo;

  /// Inliner - Specifies the inliner to use.  If this is non-null, it is
  /// added to the per-module passes.
  Pass *Inliner;

  bool DisableTailCalls;
  bool DisableUnitAtATime;
  bool DisableUnrollLoops;
  bool BBVectorize;
  bool SLPVectorize;
  bool LoopVectorize;
  bool RerollLoops;
  bool LoadCombine;
  bool DisableGVNLoadPRE;
  bool VerifyInput;
  bool VerifyOutput;
  bool MergeFunctions;
  bool PrepareForLTO;
  bool HLSLHighLevel = false; // HLSL Change
  bool HLSLAllowPreserveValues = false; // HLSL Change
  bool HLSLOnlyWarnOnUnrollFail = false; // HLSL Change
  hlsl::HLSLExtensionsCodegenHelper *HLSLExtensionsCodeGen = nullptr; // HLSL Change
  bool HLSLResMayAlias = false; // HLSL Change
  unsigned ScanLimit = 0; // HLSL Change
  bool EnableGVN = true; // HLSL Change
  bool StructurizeLoopExitsForUnroll = false; // HLSL Change
  bool HLSLEnableLifetimeMarkers = false; // HLSL Change
  bool HLSLEnableDebugNops = false; // HLSL Change

private:
  /// ExtensionList - This is list of all of the extensions that are registered.
  std::vector<std::pair<ExtensionPointTy, ExtensionFn> > Extensions;

public:
  PassManagerBuilder();
  ~PassManagerBuilder();
  /// Adds an extension that will be used by all PassManagerBuilder instances.
  /// This is intended to be used by plugins, to register a set of
  /// optimisations to run automatically.
  static void addGlobalExtension(ExtensionPointTy Ty, ExtensionFn Fn);
  void addExtension(ExtensionPointTy Ty, ExtensionFn Fn);

private:
  void addExtensionsToPM(ExtensionPointTy ETy,
                         legacy::PassManagerBase &PM) const;
  void addInitialAliasAnalysisPasses(legacy::PassManagerBase &PM) const;
  void addLTOOptimizationPasses(legacy::PassManagerBase &PM);
  void addLateLTOOptimizationPasses(legacy::PassManagerBase &PM);

public:
  /// populateFunctionPassManager - This fills in the function pass manager,
  /// which is expected to be run on each function immediately as it is
  /// generated.  The idea is to reduce the size of the IR in memory.
  void populateFunctionPassManager(legacy::FunctionPassManager &FPM);

  /// populateModulePassManager - This sets up the primary pass manager.
  void populateModulePassManager(legacy::PassManagerBase &MPM);
  void populateLTOPassManager(legacy::PassManagerBase &PM);
};

/// Registers a function for adding a standard set of passes.  This should be
/// used by optimizer plugins to allow all front ends to transparently use
/// them.  Create a static instance of this class in your plugin, providing a
/// private function that the PassManagerBuilder can use to add your passes.
struct RegisterStandardPasses {
  RegisterStandardPasses(PassManagerBuilder::ExtensionPointTy Ty,
                         PassManagerBuilder::ExtensionFn Fn) {
    PassManagerBuilder::addGlobalExtension(Ty, Fn);
  }
};

} // end namespace llvm
#endif

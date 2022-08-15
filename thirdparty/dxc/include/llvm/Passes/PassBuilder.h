//===- Parsing, selection, and construction of pass pipelines --*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// Interfaces for registering analysis passes, producing common pass manager
/// configurations, and parsing of pass pipelines.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASSES_PASSBUILDER_H
#define LLVM_PASSES_PASSBUILDER_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/IR/PassManager.h"

namespace llvm {
class TargetMachine;

/// \brief This class provides access to building LLVM's passes.
///
/// It's members provide the baseline state available to passes during their
/// construction. The \c PassRegistry.def file specifies how to construct all
/// of the built-in passes, and those may reference these members during
/// construction.
class PassBuilder {
  TargetMachine *TM;

public:
  explicit PassBuilder(TargetMachine *TM = nullptr) : TM(TM) {}

  /// \brief Registers all available module analysis passes.
  ///
  /// This is an interface that can be used to populate a \c
  /// ModuleAnalysisManager with all registered module analyses. Callers can
  /// still manually register any additional analyses.
  void registerModuleAnalyses(ModuleAnalysisManager &MAM);

  /// \brief Registers all available CGSCC analysis passes.
  ///
  /// This is an interface that can be used to populate a \c CGSCCAnalysisManager
  /// with all registered CGSCC analyses. Callers can still manually register any
  /// additional analyses.
  void registerCGSCCAnalyses(CGSCCAnalysisManager &CGAM);

  /// \brief Registers all available function analysis passes.
  ///
  /// This is an interface that can be used to populate a \c
  /// FunctionAnalysisManager with all registered function analyses. Callers can
  /// still manually register any additional analyses.
  void registerFunctionAnalyses(FunctionAnalysisManager &FAM);

  /// \brief Parse a textual pass pipeline description into a \c ModulePassManager.
  ///
  /// The format of the textual pass pipeline description looks something like:
  ///
  ///   module(function(instcombine,sroa),dce,cgscc(inliner,function(...)),...)
  ///
  /// Pass managers have ()s describing the nest structure of passes. All passes
  /// are comma separated. As a special shortcut, if the very first pass is not
  /// a module pass (as a module pass manager is), this will automatically form
  /// the shortest stack of pass managers that allow inserting that first pass.
  /// So, assuming function passes 'fpassN', CGSCC passes 'cgpassN', and loop passes
  /// 'lpassN', all of these are valid:
  ///
  ///   fpass1,fpass2,fpass3
  ///   cgpass1,cgpass2,cgpass3
  ///   lpass1,lpass2,lpass3
  ///
  /// And they are equivalent to the following (resp.):
  ///
  ///   module(function(fpass1,fpass2,fpass3))
  ///   module(cgscc(cgpass1,cgpass2,cgpass3))
  ///   module(function(loop(lpass1,lpass2,lpass3)))
  ///
  /// This shortcut is especially useful for debugging and testing small pass
  /// combinations. Note that these shortcuts don't introduce any other magic. If
  /// the sequence of passes aren't all the exact same kind of pass, it will be
  /// an error. You cannot mix different levels implicitly, you must explicitly
  /// form a pass manager in which to nest passes.
  bool parsePassPipeline(ModulePassManager &MPM, StringRef PipelineText,
                         bool VerifyEachPass = true, bool DebugLogging = false);

private:
  bool parseModulePassName(ModulePassManager &MPM, StringRef Name);
  bool parseCGSCCPassName(CGSCCPassManager &CGPM, StringRef Name);
  bool parseFunctionPassName(FunctionPassManager &FPM, StringRef Name);
  bool parseFunctionPassPipeline(FunctionPassManager &FPM,
                                 StringRef &PipelineText, bool VerifyEachPass,
                                 bool DebugLogging);
  bool parseCGSCCPassPipeline(CGSCCPassManager &CGPM, StringRef &PipelineText,
                              bool VerifyEachPass, bool DebugLogging);
  bool parseModulePassPipeline(ModulePassManager &MPM, StringRef &PipelineText,
                               bool VerifyEachPass, bool DebugLogging);
};

}

#endif

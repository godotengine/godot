//===- Parsing, selection, and construction of pass pipelines -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This file provides the implementation of the PassBuilder based on our
/// static pass registry as well as related functionality. It also provides
/// helpers to aid in analyzing, debugging, and testing passes and pass
/// pipelines.
///
//===----------------------------------------------------------------------===//

#include "llvm/Passes/PassBuilder.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Scalar/LowerExpectIntrinsic.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"

using namespace llvm;

namespace {

/// \brief No-op module pass which does nothing.
struct NoOpModulePass {
  PreservedAnalyses run(Module &M) { return PreservedAnalyses::all(); }
  static StringRef name() { return "NoOpModulePass"; }
};

/// \brief No-op module analysis.
struct NoOpModuleAnalysis {
  struct Result {};
  Result run(Module &) { return Result(); }
  static StringRef name() { return "NoOpModuleAnalysis"; }
  static void *ID() { return (void *)&PassID; }
private:
  static char PassID;
};

char NoOpModuleAnalysis::PassID;

/// \brief No-op CGSCC pass which does nothing.
struct NoOpCGSCCPass {
  PreservedAnalyses run(LazyCallGraph::SCC &C) {
    return PreservedAnalyses::all();
  }
  static StringRef name() { return "NoOpCGSCCPass"; }
};

/// \brief No-op CGSCC analysis.
struct NoOpCGSCCAnalysis {
  struct Result {};
  Result run(LazyCallGraph::SCC &) { return Result(); }
  static StringRef name() { return "NoOpCGSCCAnalysis"; }
  static void *ID() { return (void *)&PassID; }
private:
  static char PassID;
};

char NoOpCGSCCAnalysis::PassID;

/// \brief No-op function pass which does nothing.
struct NoOpFunctionPass {
  PreservedAnalyses run(Function &F) { return PreservedAnalyses::all(); }
  static StringRef name() { return "NoOpFunctionPass"; }
};

/// \brief No-op function analysis.
struct NoOpFunctionAnalysis {
  struct Result {};
  Result run(Function &) { return Result(); }
  static StringRef name() { return "NoOpFunctionAnalysis"; }
  static void *ID() { return (void *)&PassID; }
private:
  static char PassID;
};

char NoOpFunctionAnalysis::PassID;

} // End anonymous namespace.

void PassBuilder::registerModuleAnalyses(ModuleAnalysisManager &MAM) {
#define MODULE_ANALYSIS(NAME, CREATE_PASS) \
  MAM.registerPass(CREATE_PASS);
#include "PassRegistry.def"
}

void PassBuilder::registerCGSCCAnalyses(CGSCCAnalysisManager &CGAM) {
#define CGSCC_ANALYSIS(NAME, CREATE_PASS) \
  CGAM.registerPass(CREATE_PASS);
#include "PassRegistry.def"
}

void PassBuilder::registerFunctionAnalyses(FunctionAnalysisManager &FAM) {
#define FUNCTION_ANALYSIS(NAME, CREATE_PASS) \
  FAM.registerPass(CREATE_PASS);
#include "PassRegistry.def"
}

#ifndef NDEBUG
static bool isModulePassName(StringRef Name) {
#define MODULE_PASS(NAME, CREATE_PASS) if (Name == NAME) return true;
#define MODULE_ANALYSIS(NAME, CREATE_PASS)                                     \
  if (Name == "require<" NAME ">" || Name == "invalidate<" NAME ">")           \
    return true;
#include "PassRegistry.def"

  return false;
}
#endif

static bool isCGSCCPassName(StringRef Name) {
#define CGSCC_PASS(NAME, CREATE_PASS) if (Name == NAME) return true;
#define CGSCC_ANALYSIS(NAME, CREATE_PASS)                                      \
  if (Name == "require<" NAME ">" || Name == "invalidate<" NAME ">")           \
    return true;
#include "PassRegistry.def"

  return false;
}

static bool isFunctionPassName(StringRef Name) {
#define FUNCTION_PASS(NAME, CREATE_PASS) if (Name == NAME) return true;
#define FUNCTION_ANALYSIS(NAME, CREATE_PASS)                                   \
  if (Name == "require<" NAME ">" || Name == "invalidate<" NAME ">")           \
    return true;
#include "PassRegistry.def"

  return false;
}

bool PassBuilder::parseModulePassName(ModulePassManager &MPM, StringRef Name) {
#define MODULE_PASS(NAME, CREATE_PASS)                                         \
  if (Name == NAME) {                                                          \
    MPM.addPass(CREATE_PASS);                                                  \
    return true;                                                               \
  }
#define MODULE_ANALYSIS(NAME, CREATE_PASS)                                     \
  if (Name == "require<" NAME ">") {                                           \
    MPM.addPass(RequireAnalysisPass<decltype(CREATE_PASS)>());                 \
    return true;                                                               \
  }                                                                            \
  if (Name == "invalidate<" NAME ">") {                                        \
    MPM.addPass(InvalidateAnalysisPass<decltype(CREATE_PASS)>());              \
    return true;                                                               \
  }
#include "PassRegistry.def"

  return false;
}

bool PassBuilder::parseCGSCCPassName(CGSCCPassManager &CGPM, StringRef Name) {
#define CGSCC_PASS(NAME, CREATE_PASS)                                          \
  if (Name == NAME) {                                                          \
    CGPM.addPass(CREATE_PASS);                                                 \
    return true;                                                               \
  }
#define CGSCC_ANALYSIS(NAME, CREATE_PASS)                                      \
  if (Name == "require<" NAME ">") {                                           \
    CGPM.addPass(RequireAnalysisPass<decltype(CREATE_PASS)>());                \
    return true;                                                               \
  }                                                                            \
  if (Name == "invalidate<" NAME ">") {                                        \
    CGPM.addPass(InvalidateAnalysisPass<decltype(CREATE_PASS)>());             \
    return true;                                                               \
  }
#include "PassRegistry.def"

  return false;
}

bool PassBuilder::parseFunctionPassName(FunctionPassManager &FPM,
                                        StringRef Name) {
#define FUNCTION_PASS(NAME, CREATE_PASS)                                       \
  if (Name == NAME) {                                                          \
    FPM.addPass(CREATE_PASS);                                                  \
    return true;                                                               \
  }
#define FUNCTION_ANALYSIS(NAME, CREATE_PASS)                                   \
  if (Name == "require<" NAME ">") {                                           \
    FPM.addPass(RequireAnalysisPass<decltype(CREATE_PASS)>());                 \
    return true;                                                               \
  }                                                                            \
  if (Name == "invalidate<" NAME ">") {                                        \
    FPM.addPass(InvalidateAnalysisPass<decltype(CREATE_PASS)>());              \
    return true;                                                               \
  }
#include "PassRegistry.def"

  return false;
}

bool PassBuilder::parseFunctionPassPipeline(FunctionPassManager &FPM,
                                            StringRef &PipelineText,
                                            bool VerifyEachPass,
                                            bool DebugLogging) {
  for (;;) {
    // Parse nested pass managers by recursing.
    if (PipelineText.startswith("function(")) {
      FunctionPassManager NestedFPM(DebugLogging);

      // Parse the inner pipeline inte the nested manager.
      PipelineText = PipelineText.substr(strlen("function("));
      if (!parseFunctionPassPipeline(NestedFPM, PipelineText, VerifyEachPass,
                                     DebugLogging) ||
          PipelineText.empty())
        return false;
      assert(PipelineText[0] == ')');
      PipelineText = PipelineText.substr(1);

      // Add the nested pass manager with the appropriate adaptor.
      FPM.addPass(std::move(NestedFPM));
    } else {
      // Otherwise try to parse a pass name.
      size_t End = PipelineText.find_first_of(",)");
      if (!parseFunctionPassName(FPM, PipelineText.substr(0, End)))
        return false;
      if (VerifyEachPass)
        FPM.addPass(VerifierPass());

      PipelineText = PipelineText.substr(End);
    }

    if (PipelineText.empty() || PipelineText[0] == ')')
      return true;

    assert(PipelineText[0] == ',');
    PipelineText = PipelineText.substr(1);
  }
}

bool PassBuilder::parseCGSCCPassPipeline(CGSCCPassManager &CGPM,
                                         StringRef &PipelineText,
                                         bool VerifyEachPass,
                                         bool DebugLogging) {
  for (;;) {
    // Parse nested pass managers by recursing.
    if (PipelineText.startswith("cgscc(")) {
      CGSCCPassManager NestedCGPM(DebugLogging);

      // Parse the inner pipeline into the nested manager.
      PipelineText = PipelineText.substr(strlen("cgscc("));
      if (!parseCGSCCPassPipeline(NestedCGPM, PipelineText, VerifyEachPass,
                                  DebugLogging) ||
          PipelineText.empty())
        return false;
      assert(PipelineText[0] == ')');
      PipelineText = PipelineText.substr(1);

      // Add the nested pass manager with the appropriate adaptor.
      CGPM.addPass(std::move(NestedCGPM));
    } else if (PipelineText.startswith("function(")) {
      FunctionPassManager NestedFPM(DebugLogging);

      // Parse the inner pipeline inte the nested manager.
      PipelineText = PipelineText.substr(strlen("function("));
      if (!parseFunctionPassPipeline(NestedFPM, PipelineText, VerifyEachPass,
                                     DebugLogging) ||
          PipelineText.empty())
        return false;
      assert(PipelineText[0] == ')');
      PipelineText = PipelineText.substr(1);

      // Add the nested pass manager with the appropriate adaptor.
      CGPM.addPass(createCGSCCToFunctionPassAdaptor(std::move(NestedFPM)));
    } else {
      // Otherwise try to parse a pass name.
      size_t End = PipelineText.find_first_of(",)");
      if (!parseCGSCCPassName(CGPM, PipelineText.substr(0, End)))
        return false;
      // FIXME: No verifier support for CGSCC passes!

      PipelineText = PipelineText.substr(End);
    }

    if (PipelineText.empty() || PipelineText[0] == ')')
      return true;

    assert(PipelineText[0] == ',');
    PipelineText = PipelineText.substr(1);
  }
}

bool PassBuilder::parseModulePassPipeline(ModulePassManager &MPM,
                                          StringRef &PipelineText,
                                          bool VerifyEachPass,
                                          bool DebugLogging) {
  for (;;) {
    // Parse nested pass managers by recursing.
    if (PipelineText.startswith("module(")) {
      ModulePassManager NestedMPM(DebugLogging);

      // Parse the inner pipeline into the nested manager.
      PipelineText = PipelineText.substr(strlen("module("));
      if (!parseModulePassPipeline(NestedMPM, PipelineText, VerifyEachPass,
                                   DebugLogging) ||
          PipelineText.empty())
        return false;
      assert(PipelineText[0] == ')');
      PipelineText = PipelineText.substr(1);

      // Now add the nested manager as a module pass.
      MPM.addPass(std::move(NestedMPM));
    } else if (PipelineText.startswith("cgscc(")) {
      CGSCCPassManager NestedCGPM(DebugLogging);

      // Parse the inner pipeline inte the nested manager.
      PipelineText = PipelineText.substr(strlen("cgscc("));
      if (!parseCGSCCPassPipeline(NestedCGPM, PipelineText, VerifyEachPass,
                                  DebugLogging) ||
          PipelineText.empty())
        return false;
      assert(PipelineText[0] == ')');
      PipelineText = PipelineText.substr(1);

      // Add the nested pass manager with the appropriate adaptor.
      MPM.addPass(
          createModuleToPostOrderCGSCCPassAdaptor(std::move(NestedCGPM)));
    } else if (PipelineText.startswith("function(")) {
      FunctionPassManager NestedFPM(DebugLogging);

      // Parse the inner pipeline inte the nested manager.
      PipelineText = PipelineText.substr(strlen("function("));
      if (!parseFunctionPassPipeline(NestedFPM, PipelineText, VerifyEachPass,
                                     DebugLogging) ||
          PipelineText.empty())
        return false;
      assert(PipelineText[0] == ')');
      PipelineText = PipelineText.substr(1);

      // Add the nested pass manager with the appropriate adaptor.
      MPM.addPass(createModuleToFunctionPassAdaptor(std::move(NestedFPM)));
    } else {
      // Otherwise try to parse a pass name.
      size_t End = PipelineText.find_first_of(",)");
      if (!parseModulePassName(MPM, PipelineText.substr(0, End)))
        return false;
      if (VerifyEachPass)
        MPM.addPass(VerifierPass());

      PipelineText = PipelineText.substr(End);
    }

    if (PipelineText.empty() || PipelineText[0] == ')')
      return true;

    assert(PipelineText[0] == ',');
    PipelineText = PipelineText.substr(1);
  }
}

// Primary pass pipeline description parsing routine.
// FIXME: Should this routine accept a TargetMachine or require the caller to
// pre-populate the analysis managers with target-specific stuff?
bool PassBuilder::parsePassPipeline(ModulePassManager &MPM,
                                    StringRef PipelineText, bool VerifyEachPass,
                                    bool DebugLogging) {
  // By default, try to parse the pipeline as-if it were within an implicit
  // 'module(...)' pass pipeline. If this will parse at all, it needs to
  // consume the entire string.
  if (parseModulePassPipeline(MPM, PipelineText, VerifyEachPass, DebugLogging))
    return PipelineText.empty();

  // This isn't parsable as a module pipeline, look for the end of a pass name
  // and directly drop down to that layer.
  StringRef FirstName =
      PipelineText.substr(0, PipelineText.find_first_of(",)"));
  assert(!isModulePassName(FirstName) &&
         "Already handled all module pipeline options.");

  // If this looks like a CGSCC pass, parse the whole thing as a CGSCC
  // pipeline.
  if (isCGSCCPassName(FirstName)) {
    CGSCCPassManager CGPM(DebugLogging);
    if (!parseCGSCCPassPipeline(CGPM, PipelineText, VerifyEachPass,
                                DebugLogging) ||
        !PipelineText.empty())
      return false;
    MPM.addPass(createModuleToPostOrderCGSCCPassAdaptor(std::move(CGPM)));
    return true;
  }

  // Similarly, if this looks like a Function pass, parse the whole thing as
  // a Function pipelien.
  if (isFunctionPassName(FirstName)) {
    FunctionPassManager FPM(DebugLogging);
    if (!parseFunctionPassPipeline(FPM, PipelineText, VerifyEachPass,
                                   DebugLogging) ||
        !PipelineText.empty())
      return false;
    MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
    return true;
  }

  return false;
}

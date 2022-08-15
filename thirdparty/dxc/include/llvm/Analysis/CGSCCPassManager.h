//===- CGSCCPassManager.h - Call graph pass management ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This header provides classes for managing passes over SCCs of the call
/// graph. These passes form an important component of LLVM's interprocedural
/// optimizations. Because they operate on the SCCs of the call graph, and they
/// wtraverse the graph in post order, they can effectively do pair-wise
/// interprocedural optimizations for all call edges in the program. At each
/// call site edge, the callee has already been optimized as much as is
/// possible. This in turn allows very accurate analysis of it for IPO.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_CGSCCPASSMANAGER_H
#define LLVM_ANALYSIS_CGSCCPASSMANAGER_H

#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

/// \brief The CGSCC pass manager.
///
/// See the documentation for the PassManager template for details. It runs
/// a sequency of SCC passes over each SCC that the manager is run over. This
/// typedef serves as a convenient way to refer to this construct.
typedef PassManager<LazyCallGraph::SCC> CGSCCPassManager;

/// \brief The CGSCC analysis manager.
///
/// See the documentation for the AnalysisManager template for detail
/// documentation. This typedef serves as a convenient way to refer to this
/// construct in the adaptors and proxies used to integrate this into the larger
/// pass manager infrastructure.
typedef AnalysisManager<LazyCallGraph::SCC> CGSCCAnalysisManager;

/// \brief A module analysis which acts as a proxy for a CGSCC analysis
/// manager.
///
/// This primarily proxies invalidation information from the module analysis
/// manager and module pass manager to a CGSCC analysis manager. You should
/// never use a CGSCC analysis manager from within (transitively) a module
/// pass manager unless your parent module pass has received a proxy result
/// object for it.
class CGSCCAnalysisManagerModuleProxy {
public:
  class Result {
  public:
    explicit Result(CGSCCAnalysisManager &CGAM) : CGAM(&CGAM) {}
    // We have to explicitly define all the special member functions because
    // MSVC refuses to generate them.
    Result(const Result &Arg) : CGAM(Arg.CGAM) {}
    Result(Result &&Arg) : CGAM(std::move(Arg.CGAM)) {}
    Result &operator=(Result RHS) {
      std::swap(CGAM, RHS.CGAM);
      return *this;
    }
    ~Result();

    /// \brief Accessor for the \c CGSCCAnalysisManager.
    CGSCCAnalysisManager &getManager() { return *CGAM; }

    /// \brief Handler for invalidation of the module.
    ///
    /// If this analysis itself is preserved, then we assume that the call
    /// graph of the module hasn't changed and thus we don't need to invalidate
    /// *all* cached data associated with a \c SCC* in the \c
    /// CGSCCAnalysisManager.
    ///
    /// Regardless of whether this analysis is marked as preserved, all of the
    /// analyses in the \c CGSCCAnalysisManager are potentially invalidated
    /// based on the set of preserved analyses.
    bool invalidate(Module &M, const PreservedAnalyses &PA);

  private:
    CGSCCAnalysisManager *CGAM;
  };

  static void *ID() { return (void *)&PassID; }

  static StringRef name() { return "CGSCCAnalysisManagerModuleProxy"; }

  explicit CGSCCAnalysisManagerModuleProxy(CGSCCAnalysisManager &CGAM)
      : CGAM(&CGAM) {}
  // We have to explicitly define all the special member functions because MSVC
  // refuses to generate them.
  CGSCCAnalysisManagerModuleProxy(const CGSCCAnalysisManagerModuleProxy &Arg)
      : CGAM(Arg.CGAM) {}
  CGSCCAnalysisManagerModuleProxy(CGSCCAnalysisManagerModuleProxy &&Arg)
      : CGAM(std::move(Arg.CGAM)) {}
  CGSCCAnalysisManagerModuleProxy &
  operator=(CGSCCAnalysisManagerModuleProxy RHS) {
    std::swap(CGAM, RHS.CGAM);
    return *this;
  }

  /// \brief Run the analysis pass and create our proxy result object.
  ///
  /// This doesn't do any interesting work, it is primarily used to insert our
  /// proxy result object into the module analysis cache so that we can proxy
  /// invalidation to the CGSCC analysis manager.
  ///
  /// In debug builds, it will also assert that the analysis manager is empty
  /// as no queries should arrive at the CGSCC analysis manager prior to
  /// this analysis being requested.
  Result run(Module &M);

private:
  static char PassID;

  CGSCCAnalysisManager *CGAM;
};

/// \brief A CGSCC analysis which acts as a proxy for a module analysis
/// manager.
///
/// This primarily provides an accessor to a parent module analysis manager to
/// CGSCC passes. Only the const interface of the module analysis manager is
/// provided to indicate that once inside of a CGSCC analysis pass you
/// cannot request a module analysis to actually run. Instead, the user must
/// rely on the \c getCachedResult API.
///
/// This proxy *doesn't* manage the invalidation in any way. That is handled by
/// the recursive return path of each layer of the pass manager and the
/// returned PreservedAnalysis set.
class ModuleAnalysisManagerCGSCCProxy {
public:
  /// \brief Result proxy object for \c ModuleAnalysisManagerCGSCCProxy.
  class Result {
  public:
    explicit Result(const ModuleAnalysisManager &MAM) : MAM(&MAM) {}
    // We have to explicitly define all the special member functions because
    // MSVC refuses to generate them.
    Result(const Result &Arg) : MAM(Arg.MAM) {}
    Result(Result &&Arg) : MAM(std::move(Arg.MAM)) {}
    Result &operator=(Result RHS) {
      std::swap(MAM, RHS.MAM);
      return *this;
    }

    const ModuleAnalysisManager &getManager() const { return *MAM; }

    /// \brief Handle invalidation by ignoring it, this pass is immutable.
    bool invalidate(LazyCallGraph::SCC &) { return false; }

  private:
    const ModuleAnalysisManager *MAM;
  };

  static void *ID() { return (void *)&PassID; }

  static StringRef name() { return "ModuleAnalysisManagerCGSCCProxy"; }

  ModuleAnalysisManagerCGSCCProxy(const ModuleAnalysisManager &MAM)
      : MAM(&MAM) {}
  // We have to explicitly define all the special member functions because MSVC
  // refuses to generate them.
  ModuleAnalysisManagerCGSCCProxy(const ModuleAnalysisManagerCGSCCProxy &Arg)
      : MAM(Arg.MAM) {}
  ModuleAnalysisManagerCGSCCProxy(ModuleAnalysisManagerCGSCCProxy &&Arg)
      : MAM(std::move(Arg.MAM)) {}
  ModuleAnalysisManagerCGSCCProxy &
  operator=(ModuleAnalysisManagerCGSCCProxy RHS) {
    std::swap(MAM, RHS.MAM);
    return *this;
  }

  /// \brief Run the analysis pass and create our proxy result object.
  /// Nothing to see here, it just forwards the \c MAM reference into the
  /// result.
  Result run(LazyCallGraph::SCC &) { return Result(*MAM); }

private:
  static char PassID;

  const ModuleAnalysisManager *MAM;
};

/// \brief The core module pass which does a post-order walk of the SCCs and
/// runs a CGSCC pass over each one.
///
/// Designed to allow composition of a CGSCCPass(Manager) and
/// a ModulePassManager. Note that this pass must be run with a module analysis
/// manager as it uses the LazyCallGraph analysis. It will also run the
/// \c CGSCCAnalysisManagerModuleProxy analysis prior to running the CGSCC
/// pass over the module to enable a \c FunctionAnalysisManager to be used
/// within this run safely.
template <typename CGSCCPassT> class ModuleToPostOrderCGSCCPassAdaptor {
public:
  explicit ModuleToPostOrderCGSCCPassAdaptor(CGSCCPassT Pass)
      : Pass(std::move(Pass)) {}
  // We have to explicitly define all the special member functions because MSVC
  // refuses to generate them.
  ModuleToPostOrderCGSCCPassAdaptor(
      const ModuleToPostOrderCGSCCPassAdaptor &Arg)
      : Pass(Arg.Pass) {}
  ModuleToPostOrderCGSCCPassAdaptor(ModuleToPostOrderCGSCCPassAdaptor &&Arg)
      : Pass(std::move(Arg.Pass)) {}
  friend void swap(ModuleToPostOrderCGSCCPassAdaptor &LHS,
                   ModuleToPostOrderCGSCCPassAdaptor &RHS) {
    using std::swap;
    swap(LHS.Pass, RHS.Pass);
  }
  ModuleToPostOrderCGSCCPassAdaptor &
  operator=(ModuleToPostOrderCGSCCPassAdaptor RHS) {
    swap(*this, RHS);
    return *this;
  }

  /// \brief Runs the CGSCC pass across every SCC in the module.
  PreservedAnalyses run(Module &M, ModuleAnalysisManager *AM) {
    assert(AM && "We need analyses to compute the call graph!");

    // Setup the CGSCC analysis manager from its proxy.
    CGSCCAnalysisManager &CGAM =
        AM->getResult<CGSCCAnalysisManagerModuleProxy>(M).getManager();

    // Get the call graph for this module.
    LazyCallGraph &CG = AM->getResult<LazyCallGraphAnalysis>(M);

    PreservedAnalyses PA = PreservedAnalyses::all();
    for (LazyCallGraph::SCC &C : CG.postorder_sccs()) {
      PreservedAnalyses PassPA = Pass.run(C, &CGAM);

      // We know that the CGSCC pass couldn't have invalidated any other
      // SCC's analyses (that's the contract of a CGSCC pass), so
      // directly handle the CGSCC analysis manager's invalidation here. We
      // also update the preserved set of analyses to reflect that invalidated
      // analyses are now safe to preserve.
      // FIXME: This isn't quite correct. We need to handle the case where the
      // pass updated the CG, particularly some child of the current SCC, and
      // invalidate its analyses.
      PassPA = CGAM.invalidate(C, std::move(PassPA));

      // Then intersect the preserved set so that invalidation of module
      // analyses will eventually occur when the module pass completes.
      PA.intersect(std::move(PassPA));
    }

    // By definition we preserve the proxy. This precludes *any* invalidation
    // of CGSCC analyses by the proxy, but that's OK because we've taken
    // care to invalidate analyses in the CGSCC analysis manager
    // incrementally above.
    PA.preserve<CGSCCAnalysisManagerModuleProxy>();
    return PA;
  }

  static StringRef name() { return "ModuleToPostOrderCGSCCPassAdaptor"; }

private:
  CGSCCPassT Pass;
};

/// \brief A function to deduce a function pass type and wrap it in the
/// templated adaptor.
template <typename CGSCCPassT>
ModuleToPostOrderCGSCCPassAdaptor<CGSCCPassT>
createModuleToPostOrderCGSCCPassAdaptor(CGSCCPassT Pass) {
  return ModuleToPostOrderCGSCCPassAdaptor<CGSCCPassT>(std::move(Pass));
}

/// \brief A CGSCC analysis which acts as a proxy for a function analysis
/// manager.
///
/// This primarily proxies invalidation information from the CGSCC analysis
/// manager and CGSCC pass manager to a function analysis manager. You should
/// never use a function analysis manager from within (transitively) a CGSCC
/// pass manager unless your parent CGSCC pass has received a proxy result
/// object for it.
class FunctionAnalysisManagerCGSCCProxy {
public:
  class Result {
  public:
    explicit Result(FunctionAnalysisManager &FAM) : FAM(&FAM) {}
    // We have to explicitly define all the special member functions because
    // MSVC refuses to generate them.
    Result(const Result &Arg) : FAM(Arg.FAM) {}
    Result(Result &&Arg) : FAM(std::move(Arg.FAM)) {}
    Result &operator=(Result RHS) {
      std::swap(FAM, RHS.FAM);
      return *this;
    }
    ~Result();

    /// \brief Accessor for the \c FunctionAnalysisManager.
    FunctionAnalysisManager &getManager() { return *FAM; }

    /// \brief Handler for invalidation of the SCC.
    ///
    /// If this analysis itself is preserved, then we assume that the set of \c
    /// Function objects in the \c SCC hasn't changed and thus we don't need
    /// to invalidate *all* cached data associated with a \c Function* in the \c
    /// FunctionAnalysisManager.
    ///
    /// Regardless of whether this analysis is marked as preserved, all of the
    /// analyses in the \c FunctionAnalysisManager are potentially invalidated
    /// based on the set of preserved analyses.
    bool invalidate(LazyCallGraph::SCC &C, const PreservedAnalyses &PA);

  private:
    FunctionAnalysisManager *FAM;
  };

  static void *ID() { return (void *)&PassID; }

  static StringRef name() { return "FunctionAnalysisManagerCGSCCProxy"; }

  explicit FunctionAnalysisManagerCGSCCProxy(FunctionAnalysisManager &FAM)
      : FAM(&FAM) {}
  // We have to explicitly define all the special member functions because MSVC
  // refuses to generate them.
  FunctionAnalysisManagerCGSCCProxy(
      const FunctionAnalysisManagerCGSCCProxy &Arg)
      : FAM(Arg.FAM) {}
  FunctionAnalysisManagerCGSCCProxy(FunctionAnalysisManagerCGSCCProxy &&Arg)
      : FAM(std::move(Arg.FAM)) {}
  FunctionAnalysisManagerCGSCCProxy &
  operator=(FunctionAnalysisManagerCGSCCProxy RHS) {
    std::swap(FAM, RHS.FAM);
    return *this;
  }

  /// \brief Run the analysis pass and create our proxy result object.
  ///
  /// This doesn't do any interesting work, it is primarily used to insert our
  /// proxy result object into the module analysis cache so that we can proxy
  /// invalidation to the function analysis manager.
  ///
  /// In debug builds, it will also assert that the analysis manager is empty
  /// as no queries should arrive at the function analysis manager prior to
  /// this analysis being requested.
  Result run(LazyCallGraph::SCC &C);

private:
  static char PassID;

  FunctionAnalysisManager *FAM;
};

/// \brief A function analysis which acts as a proxy for a CGSCC analysis
/// manager.
///
/// This primarily provides an accessor to a parent CGSCC analysis manager to
/// function passes. Only the const interface of the CGSCC analysis manager is
/// provided to indicate that once inside of a function analysis pass you
/// cannot request a CGSCC analysis to actually run. Instead, the user must
/// rely on the \c getCachedResult API.
///
/// This proxy *doesn't* manage the invalidation in any way. That is handled by
/// the recursive return path of each layer of the pass manager and the
/// returned PreservedAnalysis set.
class CGSCCAnalysisManagerFunctionProxy {
public:
  /// \brief Result proxy object for \c ModuleAnalysisManagerFunctionProxy.
  class Result {
  public:
    explicit Result(const CGSCCAnalysisManager &CGAM) : CGAM(&CGAM) {}
    // We have to explicitly define all the special member functions because
    // MSVC refuses to generate them.
    Result(const Result &Arg) : CGAM(Arg.CGAM) {}
    Result(Result &&Arg) : CGAM(std::move(Arg.CGAM)) {}
    Result &operator=(Result RHS) {
      std::swap(CGAM, RHS.CGAM);
      return *this;
    }

    const CGSCCAnalysisManager &getManager() const { return *CGAM; }

    /// \brief Handle invalidation by ignoring it, this pass is immutable.
    bool invalidate(Function &) { return false; }

  private:
    const CGSCCAnalysisManager *CGAM;
  };

  static void *ID() { return (void *)&PassID; }

  static StringRef name() { return "CGSCCAnalysisManagerFunctionProxy"; }

  CGSCCAnalysisManagerFunctionProxy(const CGSCCAnalysisManager &CGAM)
      : CGAM(&CGAM) {}
  // We have to explicitly define all the special member functions because MSVC
  // refuses to generate them.
  CGSCCAnalysisManagerFunctionProxy(
      const CGSCCAnalysisManagerFunctionProxy &Arg)
      : CGAM(Arg.CGAM) {}
  CGSCCAnalysisManagerFunctionProxy(CGSCCAnalysisManagerFunctionProxy &&Arg)
      : CGAM(std::move(Arg.CGAM)) {}
  CGSCCAnalysisManagerFunctionProxy &
  operator=(CGSCCAnalysisManagerFunctionProxy RHS) {
    std::swap(CGAM, RHS.CGAM);
    return *this;
  }

  /// \brief Run the analysis pass and create our proxy result object.
  /// Nothing to see here, it just forwards the \c CGAM reference into the
  /// result.
  Result run(Function &) { return Result(*CGAM); }

private:
  static char PassID;

  const CGSCCAnalysisManager *CGAM;
};

/// \brief Adaptor that maps from a SCC to its functions.
///
/// Designed to allow composition of a FunctionPass(Manager) and
/// a CGSCCPassManager. Note that if this pass is constructed with a pointer
/// to a \c CGSCCAnalysisManager it will run the
/// \c FunctionAnalysisManagerCGSCCProxy analysis prior to running the function
/// pass over the SCC to enable a \c FunctionAnalysisManager to be used
/// within this run safely.
template <typename FunctionPassT> class CGSCCToFunctionPassAdaptor {
public:
  explicit CGSCCToFunctionPassAdaptor(FunctionPassT Pass)
      : Pass(std::move(Pass)) {}
  // We have to explicitly define all the special member functions because MSVC
  // refuses to generate them.
  CGSCCToFunctionPassAdaptor(const CGSCCToFunctionPassAdaptor &Arg)
      : Pass(Arg.Pass) {}
  CGSCCToFunctionPassAdaptor(CGSCCToFunctionPassAdaptor &&Arg)
      : Pass(std::move(Arg.Pass)) {}
  friend void swap(CGSCCToFunctionPassAdaptor &LHS,
                   CGSCCToFunctionPassAdaptor &RHS) {
    using std::swap;
    swap(LHS.Pass, RHS.Pass);
  }
  CGSCCToFunctionPassAdaptor &operator=(CGSCCToFunctionPassAdaptor RHS) {
    swap(*this, RHS);
    return *this;
  }

  /// \brief Runs the function pass across every function in the module.
  PreservedAnalyses run(LazyCallGraph::SCC &C, CGSCCAnalysisManager *AM) {
    FunctionAnalysisManager *FAM = nullptr;
    if (AM)
      // Setup the function analysis manager from its proxy.
      FAM = &AM->getResult<FunctionAnalysisManagerCGSCCProxy>(C).getManager();

    PreservedAnalyses PA = PreservedAnalyses::all();
    for (LazyCallGraph::Node *N : C) {
      PreservedAnalyses PassPA = Pass.run(N->getFunction(), FAM);

      // We know that the function pass couldn't have invalidated any other
      // function's analyses (that's the contract of a function pass), so
      // directly handle the function analysis manager's invalidation here.
      // Also, update the preserved analyses to reflect that once invalidated
      // these can again be preserved.
      if (FAM)
        PassPA = FAM->invalidate(N->getFunction(), std::move(PassPA));

      // Then intersect the preserved set so that invalidation of module
      // analyses will eventually occur when the module pass completes.
      PA.intersect(std::move(PassPA));
    }

    // By definition we preserve the proxy. This precludes *any* invalidation
    // of function analyses by the proxy, but that's OK because we've taken
    // care to invalidate analyses in the function analysis manager
    // incrementally above.
    // FIXME: We need to update the call graph here to account for any deleted
    // edges!
    PA.preserve<FunctionAnalysisManagerCGSCCProxy>();
    return PA;
  }

  static StringRef name() { return "CGSCCToFunctionPassAdaptor"; }

private:
  FunctionPassT Pass;
};

/// \brief A function to deduce a function pass type and wrap it in the
/// templated adaptor.
template <typename FunctionPassT>
CGSCCToFunctionPassAdaptor<FunctionPassT>
createCGSCCToFunctionPassAdaptor(FunctionPassT Pass) {
  return CGSCCToFunctionPassAdaptor<FunctionPassT>(std::move(Pass));
}
}

#endif

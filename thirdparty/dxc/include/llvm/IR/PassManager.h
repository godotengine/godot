//===- PassManager.h - Pass management infrastructure -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This header defines various interfaces for pass management in LLVM. There
/// is no "pass" interface in LLVM per se. Instead, an instance of any class
/// which supports a method to 'run' it over a unit of IR can be used as
/// a pass. A pass manager is generally a tool to collect a sequence of passes
/// which run over a particular IR construct, and run each of them in sequence
/// over each such construct in the containing IR construct. As there is no
/// containing IR construct for a Module, a manager for passes over modules
/// forms the base case which runs its managed passes in sequence over the
/// single module provided.
///
/// The core IR library provides managers for running passes over
/// modules and functions.
///
/// * FunctionPassManager can run over a Module, runs each pass over
///   a Function.
/// * ModulePassManager must be directly run, runs each pass over the Module.
///
/// Note that the implementations of the pass managers use concept-based
/// polymorphism as outlined in the "Value Semantics and Concept-based
/// Polymorphism" talk (or its abbreviated sibling "Inheritance Is The Base
/// Class of Evil") by Sean Parent:
/// * http://github.com/sean-parent/sean-parent.github.com/wiki/Papers-and-Presentations
/// * http://www.youtube.com/watch?v=_BpMYeUFXv8
/// * http://channel9.msdn.com/Events/GoingNative/2013/Inheritance-Is-The-Base-Class-of-Evil
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_PASSMANAGER_H
#define LLVM_IR_PASSMANAGER_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManagerInternal.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/type_traits.h"
#include <list>
#include <memory>
#include <vector>

namespace llvm {

class Module;
class Function;

/// \brief An abstract set of preserved analyses following a transformation pass
/// run.
///
/// When a transformation pass is run, it can return a set of analyses whose
/// results were preserved by that transformation. The default set is "none",
/// and preserving analyses must be done explicitly.
///
/// There is also an explicit all state which can be used (for example) when
/// the IR is not mutated at all.
class PreservedAnalyses {
public:
  // We have to explicitly define all the special member functions because MSVC
  // refuses to generate them.
  PreservedAnalyses() {}
  PreservedAnalyses(const PreservedAnalyses &Arg)
      : PreservedPassIDs(Arg.PreservedPassIDs) {}
  PreservedAnalyses(PreservedAnalyses &&Arg)
      : PreservedPassIDs(std::move(Arg.PreservedPassIDs)) {}
  friend void swap(PreservedAnalyses &LHS, PreservedAnalyses &RHS) {
    using std::swap;
    swap(LHS.PreservedPassIDs, RHS.PreservedPassIDs);
  }
  PreservedAnalyses &operator=(PreservedAnalyses RHS) {
    swap(*this, RHS);
    return *this;
  }

  /// \brief Convenience factory function for the empty preserved set.
  static PreservedAnalyses none() { return PreservedAnalyses(); }

  /// \brief Construct a special preserved set that preserves all passes.
  static PreservedAnalyses all() {
    PreservedAnalyses PA;
    PA.PreservedPassIDs.insert((void *)AllPassesID);
    return PA;
  }

  /// \brief Mark a particular pass as preserved, adding it to the set.
  template <typename PassT> void preserve() { preserve(PassT::ID()); }

  /// \brief Mark an abstract PassID as preserved, adding it to the set.
  void preserve(void *PassID) {
    if (!areAllPreserved())
      PreservedPassIDs.insert(PassID);
  }

  /// \brief Intersect this set with another in place.
  ///
  /// This is a mutating operation on this preserved set, removing all
  /// preserved passes which are not also preserved in the argument.
  void intersect(const PreservedAnalyses &Arg) {
    if (Arg.areAllPreserved())
      return;
    if (areAllPreserved()) {
      PreservedPassIDs = Arg.PreservedPassIDs;
      return;
    }
    for (void *P : PreservedPassIDs)
      if (!Arg.PreservedPassIDs.count(P))
        PreservedPassIDs.erase(P);
  }

  /// \brief Intersect this set with a temporary other set in place.
  ///
  /// This is a mutating operation on this preserved set, removing all
  /// preserved passes which are not also preserved in the argument.
  void intersect(PreservedAnalyses &&Arg) {
    if (Arg.areAllPreserved())
      return;
    if (areAllPreserved()) {
      PreservedPassIDs = std::move(Arg.PreservedPassIDs);
      return;
    }
    for (void *P : PreservedPassIDs)
      if (!Arg.PreservedPassIDs.count(P))
        PreservedPassIDs.erase(P);
  }

  /// \brief Query whether a pass is marked as preserved by this set.
  template <typename PassT> bool preserved() const {
    return preserved(PassT::ID());
  }

  /// \brief Query whether an abstract pass ID is marked as preserved by this
  /// set.
  bool preserved(void *PassID) const {
    return PreservedPassIDs.count((void *)AllPassesID) ||
           PreservedPassIDs.count(PassID);
  }

  /// \brief Test whether all passes are preserved.
  ///
  /// This is used primarily to optimize for the case of no changes which will
  /// common in many scenarios.
  bool areAllPreserved() const {
    return PreservedPassIDs.count((void *)AllPassesID);
  }

private:
  // Note that this must not be -1 or -2 as those are already used by the
  // SmallPtrSet.
  static const uintptr_t AllPassesID = (intptr_t)(-3);

  SmallPtrSet<void *, 2> PreservedPassIDs;
};

// Forward declare the analysis manager template.
template <typename IRUnitT> class AnalysisManager;

/// \brief Manages a sequence of passes over units of IR.
///
/// A pass manager contains a sequence of passes to run over units of IR. It is
/// itself a valid pass over that unit of IR, and when over some given IR will
/// run each pass in sequence. This is the primary and most basic building
/// block of a pass pipeline.
///
/// If it is run with an \c AnalysisManager<IRUnitT> argument, it will propagate
/// that analysis manager to each pass it runs, as well as calling the analysis
/// manager's invalidation routine with the PreservedAnalyses of each pass it
/// runs.
template <typename IRUnitT> class PassManager {
public:
  /// \brief Construct a pass manager.
  ///
  /// It can be passed a flag to get debug logging as the passes are run.
  PassManager(bool DebugLogging = false) : DebugLogging(DebugLogging) {}
  // We have to explicitly define all the special member functions because MSVC
  // refuses to generate them.
  PassManager(PassManager &&Arg)
      : Passes(std::move(Arg.Passes)),
        DebugLogging(std::move(Arg.DebugLogging)) {}
  PassManager &operator=(PassManager &&RHS) {
    Passes = std::move(RHS.Passes);
    DebugLogging = std::move(RHS.DebugLogging);
    return *this;
  }

  /// \brief Run all of the passes in this manager over the IR.
  PreservedAnalyses run(IRUnitT &IR, AnalysisManager<IRUnitT> *AM = nullptr) {
    PreservedAnalyses PA = PreservedAnalyses::all();

    if (DebugLogging)
      dbgs() << "Starting pass manager run.\n";

    for (unsigned Idx = 0, Size = Passes.size(); Idx != Size; ++Idx) {
      if (DebugLogging)
        dbgs() << "Running pass: " << Passes[Idx]->name() << "\n";

      PreservedAnalyses PassPA = Passes[Idx]->run(IR, AM);

      // If we have an active analysis manager at this level we want to ensure
      // we update it as each pass runs and potentially invalidates analyses.
      // We also update the preserved set of analyses based on what analyses we
      // have already handled the invalidation for here and don't need to
      // invalidate when finished.
      if (AM)
        PassPA = AM->invalidate(IR, std::move(PassPA));

      // Finally, we intersect the final preserved analyses to compute the
      // aggregate preserved set for this pass manager.
      PA.intersect(std::move(PassPA));

      // FIXME: Historically, the pass managers all called the LLVM context's
      // yield function here. We don't have a generic way to acquire the
      // context and it isn't yet clear what the right pattern is for yielding
      // in the new pass manager so it is currently omitted.
      //IR.getContext().yield();
    }

    if (DebugLogging)
      dbgs() << "Finished pass manager run.\n";

    return PA;
  }

  template <typename PassT> void addPass(PassT Pass) {
    typedef detail::PassModel<IRUnitT, PassT> PassModelT;
    Passes.emplace_back(new PassModelT(std::move(Pass)));
  }

  static StringRef name() { return "PassManager"; }

private:
  typedef detail::PassConcept<IRUnitT> PassConceptT;

  PassManager(const PassManager &) = delete;
  PassManager &operator=(const PassManager &) = delete;

  std::vector<std::unique_ptr<PassConceptT>> Passes;

  /// \brief Flag indicating whether we should do debug logging.
  bool DebugLogging;
};

/// \brief Convenience typedef for a pass manager over modules.
typedef PassManager<Module> ModulePassManager;

/// \brief Convenience typedef for a pass manager over functions.
typedef PassManager<Function> FunctionPassManager;

namespace detail {

/// \brief A CRTP base used to implement analysis managers.
///
/// This class template serves as the boiler plate of an analysis manager. Any
/// analysis manager can be implemented on top of this base class. Any
/// implementation will be required to provide specific hooks:
///
/// - getResultImpl
/// - getCachedResultImpl
/// - invalidateImpl
///
/// The details of the call pattern are within.
///
/// Note that there is also a generic analysis manager template which implements
/// the above required functions along with common datastructures used for
/// managing analyses. This base class is factored so that if you need to
/// customize the handling of a specific IR unit, you can do so without
/// replicating *all* of the boilerplate.
template <typename DerivedT, typename IRUnitT> class AnalysisManagerBase {
  DerivedT *derived_this() { return static_cast<DerivedT *>(this); }
  const DerivedT *derived_this() const {
    return static_cast<const DerivedT *>(this);
  }

  AnalysisManagerBase(const AnalysisManagerBase &) = delete;
  AnalysisManagerBase &
  operator=(const AnalysisManagerBase &) = delete;

protected:
  typedef detail::AnalysisResultConcept<IRUnitT> ResultConceptT;
  typedef detail::AnalysisPassConcept<IRUnitT> PassConceptT;

  // FIXME: Provide template aliases for the models when we're using C++11 in
  // a mode supporting them.

  // We have to explicitly define all the special member functions because MSVC
  // refuses to generate them.
  AnalysisManagerBase() {}
  AnalysisManagerBase(AnalysisManagerBase &&Arg)
      : AnalysisPasses(std::move(Arg.AnalysisPasses)) {}
  AnalysisManagerBase &operator=(AnalysisManagerBase &&RHS) {
    AnalysisPasses = std::move(RHS.AnalysisPasses);
    return *this;
  }

public:
  /// \brief Get the result of an analysis pass for this module.
  ///
  /// If there is not a valid cached result in the manager already, this will
  /// re-run the analysis to produce a valid result.
  template <typename PassT> typename PassT::Result &getResult(IRUnitT &IR) {
    assert(AnalysisPasses.count(PassT::ID()) &&
           "This analysis pass was not registered prior to being queried");

    ResultConceptT &ResultConcept =
        derived_this()->getResultImpl(PassT::ID(), IR);
    typedef detail::AnalysisResultModel<IRUnitT, PassT, typename PassT::Result>
        ResultModelT;
    return static_cast<ResultModelT &>(ResultConcept).Result;
  }

  /// \brief Get the cached result of an analysis pass for this module.
  ///
  /// This method never runs the analysis.
  ///
  /// \returns null if there is no cached result.
  template <typename PassT>
  typename PassT::Result *getCachedResult(IRUnitT &IR) const {
    assert(AnalysisPasses.count(PassT::ID()) &&
           "This analysis pass was not registered prior to being queried");

    ResultConceptT *ResultConcept =
        derived_this()->getCachedResultImpl(PassT::ID(), IR);
    if (!ResultConcept)
      return nullptr;

    typedef detail::AnalysisResultModel<IRUnitT, PassT, typename PassT::Result>
        ResultModelT;
    return &static_cast<ResultModelT *>(ResultConcept)->Result;
  }

  /// \brief Register an analysis pass with the manager.
  ///
  /// This provides an initialized and set-up analysis pass to the analysis
  /// manager. Whomever is setting up analysis passes must use this to populate
  /// the manager with all of the analysis passes available.
  template <typename PassT> void registerPass(PassT Pass) {
    assert(!AnalysisPasses.count(PassT::ID()) &&
           "Registered the same analysis pass twice!");
    typedef detail::AnalysisPassModel<IRUnitT, PassT> PassModelT;
    AnalysisPasses[PassT::ID()].reset(new PassModelT(std::move(Pass)));
  }

  /// \brief Invalidate a specific analysis pass for an IR module.
  ///
  /// Note that the analysis result can disregard invalidation.
  template <typename PassT> void invalidate(IRUnitT &IR) {
    assert(AnalysisPasses.count(PassT::ID()) &&
           "This analysis pass was not registered prior to being invalidated");
    derived_this()->invalidateImpl(PassT::ID(), IR);
  }

  /// \brief Invalidate analyses cached for an IR unit.
  ///
  /// Walk through all of the analyses pertaining to this unit of IR and
  /// invalidate them unless they are preserved by the PreservedAnalyses set.
  /// We accept the PreservedAnalyses set by value and update it with each
  /// analyis pass which has been successfully invalidated and thus can be
  /// preserved going forward. The updated set is returned.
  PreservedAnalyses invalidate(IRUnitT &IR, PreservedAnalyses PA) {
    return derived_this()->invalidateImpl(IR, std::move(PA));
  }

protected:
  /// \brief Lookup a registered analysis pass.
  PassConceptT &lookupPass(void *PassID) {
    typename AnalysisPassMapT::iterator PI = AnalysisPasses.find(PassID);
    assert(PI != AnalysisPasses.end() &&
           "Analysis passes must be registered prior to being queried!");
    return *PI->second;
  }

  /// \brief Lookup a registered analysis pass.
  const PassConceptT &lookupPass(void *PassID) const {
    typename AnalysisPassMapT::const_iterator PI = AnalysisPasses.find(PassID);
    assert(PI != AnalysisPasses.end() &&
           "Analysis passes must be registered prior to being queried!");
    return *PI->second;
  }

private:
  /// \brief Map type from module analysis pass ID to pass concept pointer.
  typedef DenseMap<void *, std::unique_ptr<PassConceptT>> AnalysisPassMapT;

  /// \brief Collection of module analysis passes, indexed by ID.
  AnalysisPassMapT AnalysisPasses;
};

} // End namespace detail

/// \brief A generic analysis pass manager with lazy running and caching of
/// results.
///
/// This analysis manager can be used for any IR unit where the address of the
/// IR unit sufficies as its identity. It manages the cache for a unit of IR via
/// the address of each unit of IR cached.
template <typename IRUnitT>
class AnalysisManager
    : public detail::AnalysisManagerBase<AnalysisManager<IRUnitT>, IRUnitT> {
  friend class detail::AnalysisManagerBase<AnalysisManager<IRUnitT>, IRUnitT>;
  typedef detail::AnalysisManagerBase<AnalysisManager<IRUnitT>, IRUnitT> BaseT;
  typedef typename BaseT::ResultConceptT ResultConceptT;
  typedef typename BaseT::PassConceptT PassConceptT;

public:
  // Most public APIs are inherited from the CRTP base class.

  /// \brief Construct an empty analysis manager.
  ///
  /// A flag can be passed to indicate that the manager should perform debug
  /// logging.
  AnalysisManager(bool DebugLogging = false) : DebugLogging(DebugLogging) {}

  // We have to explicitly define all the special member functions because MSVC
  // refuses to generate them.
  AnalysisManager(AnalysisManager &&Arg)
      : BaseT(std::move(static_cast<BaseT &>(Arg))),
        AnalysisResults(std::move(Arg.AnalysisResults)),
        DebugLogging(std::move(Arg.DebugLogging)) {}
  AnalysisManager &operator=(AnalysisManager &&RHS) {
    BaseT::operator=(std::move(static_cast<BaseT &>(RHS)));
    AnalysisResults = std::move(RHS.AnalysisResults);
    DebugLogging = std::move(RHS.DebugLogging);
    return *this;
  }

  /// \brief Returns true if the analysis manager has an empty results cache.
  bool empty() const {
    assert(AnalysisResults.empty() == AnalysisResultLists.empty() &&
           "The storage and index of analysis results disagree on how many "
           "there are!");
    return AnalysisResults.empty();
  }

  /// \brief Clear the analysis result cache.
  ///
  /// This routine allows cleaning up when the set of IR units itself has
  /// potentially changed, and thus we can't even look up a a result and
  /// invalidate it directly. Notably, this does *not* call invalidate functions
  /// as there is nothing to be done for them.
  void clear() {
    AnalysisResults.clear();
    AnalysisResultLists.clear();
  }

private:
  AnalysisManager(const AnalysisManager &) = delete;
  AnalysisManager &operator=(const AnalysisManager &) = delete;

  /// \brief Get an analysis result, running the pass if necessary.
  ResultConceptT &getResultImpl(void *PassID, IRUnitT &IR) {
    typename AnalysisResultMapT::iterator RI;
    bool Inserted;
    std::tie(RI, Inserted) = AnalysisResults.insert(std::make_pair(
        std::make_pair(PassID, &IR), typename AnalysisResultListT::iterator()));

    // If we don't have a cached result for this function, look up the pass and
    // run it to produce a result, which we then add to the cache.
    if (Inserted) {
      auto &P = this->lookupPass(PassID);
      if (DebugLogging)
        dbgs() << "Running analysis: " << P.name() << "\n";
      AnalysisResultListT &ResultList = AnalysisResultLists[&IR];
      ResultList.emplace_back(PassID, P.run(IR, this));

      // P.run may have inserted elements into AnalysisResults and invalidated
      // RI.
      RI = AnalysisResults.find(std::make_pair(PassID, &IR));
      assert(RI != AnalysisResults.end() && "we just inserted it!");

      RI->second = std::prev(ResultList.end());
    }

    return *RI->second->second;
  }

  /// \brief Get a cached analysis result or return null.
  ResultConceptT *getCachedResultImpl(void *PassID, IRUnitT &IR) const {
    typename AnalysisResultMapT::const_iterator RI =
        AnalysisResults.find(std::make_pair(PassID, &IR));
    return RI == AnalysisResults.end() ? nullptr : &*RI->second->second;
  }

  /// \brief Invalidate a function pass result.
  void invalidateImpl(void *PassID, IRUnitT &IR) {
    typename AnalysisResultMapT::iterator RI =
        AnalysisResults.find(std::make_pair(PassID, &IR));
    if (RI == AnalysisResults.end())
      return;

    if (DebugLogging)
      dbgs() << "Invalidating analysis: " << this->lookupPass(PassID).name()
             << "\n";
    AnalysisResultLists[&IR].erase(RI->second);
    AnalysisResults.erase(RI);
  }

  /// \brief Invalidate the results for a function..
  PreservedAnalyses invalidateImpl(IRUnitT &IR, PreservedAnalyses PA) {
    // Short circuit for a common case of all analyses being preserved.
    if (PA.areAllPreserved())
      return PA;

    if (DebugLogging)
      dbgs() << "Invalidating all non-preserved analyses for: "
             << IR.getName() << "\n";

    // Clear all the invalidated results associated specifically with this
    // function.
    SmallVector<void *, 8> InvalidatedPassIDs;
    AnalysisResultListT &ResultsList = AnalysisResultLists[&IR];
    for (typename AnalysisResultListT::iterator I = ResultsList.begin(),
                                                E = ResultsList.end();
         I != E;) {
      void *PassID = I->first;

      // Pass the invalidation down to the pass itself to see if it thinks it is
      // necessary. The analysis pass can return false if no action on the part
      // of the analysis manager is required for this invalidation event.
      if (I->second->invalidate(IR, PA)) {
        if (DebugLogging)
          dbgs() << "Invalidating analysis: " << this->lookupPass(PassID).name()
                 << "\n";

        InvalidatedPassIDs.push_back(I->first);
        I = ResultsList.erase(I);
      } else {
        ++I;
      }

      // After handling each pass, we mark it as preserved. Once we've
      // invalidated any stale results, the rest of the system is allowed to
      // start preserving this analysis again.
      PA.preserve(PassID);
    }
    while (!InvalidatedPassIDs.empty())
      AnalysisResults.erase(
          std::make_pair(InvalidatedPassIDs.pop_back_val(), &IR));
    if (ResultsList.empty())
      AnalysisResultLists.erase(&IR);

    return PA;
  }

  /// \brief List of function analysis pass IDs and associated concept pointers.
  ///
  /// Requires iterators to be valid across appending new entries and arbitrary
  /// erases. Provides both the pass ID and concept pointer such that it is
  /// half of a bijection and provides storage for the actual result concept.
  typedef std::list<std::pair<
      void *, std::unique_ptr<detail::AnalysisResultConcept<IRUnitT>>>>
      AnalysisResultListT;

  /// \brief Map type from function pointer to our custom list type.
  typedef DenseMap<IRUnitT *, AnalysisResultListT> AnalysisResultListMapT;

  /// \brief Map from function to a list of function analysis results.
  ///
  /// Provides linear time removal of all analysis results for a function and
  /// the ultimate storage for a particular cached analysis result.
  AnalysisResultListMapT AnalysisResultLists;

  /// \brief Map type from a pair of analysis ID and function pointer to an
  /// iterator into a particular result list.
  typedef DenseMap<std::pair<void *, IRUnitT *>,
                   typename AnalysisResultListT::iterator> AnalysisResultMapT;

  /// \brief Map from an analysis ID and function to a particular cached
  /// analysis result.
  AnalysisResultMapT AnalysisResults;

  /// \brief A flag indicating whether debug logging is enabled.
  bool DebugLogging;
};

/// \brief Convenience typedef for the Module analysis manager.
typedef AnalysisManager<Module> ModuleAnalysisManager;

/// \brief Convenience typedef for the Function analysis manager.
typedef AnalysisManager<Function> FunctionAnalysisManager;

/// \brief A module analysis which acts as a proxy for a function analysis
/// manager.
///
/// This primarily proxies invalidation information from the module analysis
/// manager and module pass manager to a function analysis manager. You should
/// never use a function analysis manager from within (transitively) a module
/// pass manager unless your parent module pass has received a proxy result
/// object for it.
class FunctionAnalysisManagerModuleProxy {
public:
  class Result;

  static void *ID() { return (void *)&PassID; }

  static StringRef name() { return "FunctionAnalysisManagerModuleProxy"; }

  explicit FunctionAnalysisManagerModuleProxy(FunctionAnalysisManager &FAM)
      : FAM(&FAM) {}
  // We have to explicitly define all the special member functions because MSVC
  // refuses to generate them.
  FunctionAnalysisManagerModuleProxy(
      const FunctionAnalysisManagerModuleProxy &Arg)
      : FAM(Arg.FAM) {}
  FunctionAnalysisManagerModuleProxy(FunctionAnalysisManagerModuleProxy &&Arg)
      : FAM(std::move(Arg.FAM)) {}
  FunctionAnalysisManagerModuleProxy &
  operator=(FunctionAnalysisManagerModuleProxy RHS) {
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
  Result run(Module &M);

private:
  static char PassID;

  FunctionAnalysisManager *FAM;
};

/// \brief The result proxy object for the
/// \c FunctionAnalysisManagerModuleProxy.
///
/// See its documentation for more information.
class FunctionAnalysisManagerModuleProxy::Result {
public:
  explicit Result(FunctionAnalysisManager &FAM) : FAM(&FAM) {}
  // We have to explicitly define all the special member functions because MSVC
  // refuses to generate them.
  Result(const Result &Arg) : FAM(Arg.FAM) {}
  Result(Result &&Arg) : FAM(std::move(Arg.FAM)) {}
  Result &operator=(Result RHS) {
    std::swap(FAM, RHS.FAM);
    return *this;
  }
  ~Result();

  /// \brief Accessor for the \c FunctionAnalysisManager.
  FunctionAnalysisManager &getManager() { return *FAM; }

  /// \brief Handler for invalidation of the module.
  ///
  /// If this analysis itself is preserved, then we assume that the set of \c
  /// Function objects in the \c Module hasn't changed and thus we don't need
  /// to invalidate *all* cached data associated with a \c Function* in the \c
  /// FunctionAnalysisManager.
  ///
  /// Regardless of whether this analysis is marked as preserved, all of the
  /// analyses in the \c FunctionAnalysisManager are potentially invalidated
  /// based on the set of preserved analyses.
  bool invalidate(Module &M, const PreservedAnalyses &PA);

private:
  FunctionAnalysisManager *FAM;
};

/// \brief A function analysis which acts as a proxy for a module analysis
/// manager.
///
/// This primarily provides an accessor to a parent module analysis manager to
/// function passes. Only the const interface of the module analysis manager is
/// provided to indicate that once inside of a function analysis pass you
/// cannot request a module analysis to actually run. Instead, the user must
/// rely on the \c getCachedResult API.
///
/// This proxy *doesn't* manage the invalidation in any way. That is handled by
/// the recursive return path of each layer of the pass manager and the
/// returned PreservedAnalysis set.
class ModuleAnalysisManagerFunctionProxy {
public:
  /// \brief Result proxy object for \c ModuleAnalysisManagerFunctionProxy.
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
    bool invalidate(Function &) { return false; }

  private:
    const ModuleAnalysisManager *MAM;
  };

  static void *ID() { return (void *)&PassID; }

  static StringRef name() { return "ModuleAnalysisManagerFunctionProxy"; }

  ModuleAnalysisManagerFunctionProxy(const ModuleAnalysisManager &MAM)
      : MAM(&MAM) {}
  // We have to explicitly define all the special member functions because MSVC
  // refuses to generate them.
  ModuleAnalysisManagerFunctionProxy(
      const ModuleAnalysisManagerFunctionProxy &Arg)
      : MAM(Arg.MAM) {}
  ModuleAnalysisManagerFunctionProxy(ModuleAnalysisManagerFunctionProxy &&Arg)
      : MAM(std::move(Arg.MAM)) {}
  ModuleAnalysisManagerFunctionProxy &
  operator=(ModuleAnalysisManagerFunctionProxy RHS) {
    std::swap(MAM, RHS.MAM);
    return *this;
  }

  /// \brief Run the analysis pass and create our proxy result object.
  /// Nothing to see here, it just forwards the \c MAM reference into the
  /// result.
  Result run(Function &) { return Result(*MAM); }

private:
  static char PassID;

  const ModuleAnalysisManager *MAM;
};

/// \brief Trivial adaptor that maps from a module to its functions.
///
/// Designed to allow composition of a FunctionPass(Manager) and
/// a ModulePassManager. Note that if this pass is constructed with a pointer
/// to a \c ModuleAnalysisManager it will run the
/// \c FunctionAnalysisManagerModuleProxy analysis prior to running the function
/// pass over the module to enable a \c FunctionAnalysisManager to be used
/// within this run safely.
///
/// Function passes run within this adaptor can rely on having exclusive access
/// to the function they are run over. They should not read or modify any other
/// functions! Other threads or systems may be manipulating other functions in
/// the module, and so their state should never be relied on.
/// FIXME: Make the above true for all of LLVM's actual passes, some still
/// violate this principle.
///
/// Function passes can also read the module containing the function, but they
/// should not modify that module outside of the use lists of various globals.
/// For example, a function pass is not permitted to add functions to the
/// module.
/// FIXME: Make the above true for all of LLVM's actual passes, some still
/// violate this principle.
template <typename FunctionPassT> class ModuleToFunctionPassAdaptor {
public:
  explicit ModuleToFunctionPassAdaptor(FunctionPassT Pass)
      : Pass(std::move(Pass)) {}
  // We have to explicitly define all the special member functions because MSVC
  // refuses to generate them.
  ModuleToFunctionPassAdaptor(const ModuleToFunctionPassAdaptor &Arg)
      : Pass(Arg.Pass) {}
  ModuleToFunctionPassAdaptor(ModuleToFunctionPassAdaptor &&Arg)
      : Pass(std::move(Arg.Pass)) {}
  friend void swap(ModuleToFunctionPassAdaptor &LHS,
                   ModuleToFunctionPassAdaptor &RHS) {
    using std::swap;
    swap(LHS.Pass, RHS.Pass);
  }
  ModuleToFunctionPassAdaptor &operator=(ModuleToFunctionPassAdaptor RHS) {
    swap(*this, RHS);
    return *this;
  }

  /// \brief Runs the function pass across every function in the module.
  PreservedAnalyses run(Module &M, ModuleAnalysisManager *AM) {
    FunctionAnalysisManager *FAM = nullptr;
    if (AM)
      // Setup the function analysis manager from its proxy.
      FAM = &AM->getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();

    PreservedAnalyses PA = PreservedAnalyses::all();
    for (Function &F : M) {
      if (F.isDeclaration())
        continue;

      PreservedAnalyses PassPA = Pass.run(F, FAM);

      // We know that the function pass couldn't have invalidated any other
      // function's analyses (that's the contract of a function pass), so
      // directly handle the function analysis manager's invalidation here and
      // update our preserved set to reflect that these have already been
      // handled.
      if (FAM)
        PassPA = FAM->invalidate(F, std::move(PassPA));

      // Then intersect the preserved set so that invalidation of module
      // analyses will eventually occur when the module pass completes.
      PA.intersect(std::move(PassPA));
    }

    // By definition we preserve the proxy. This precludes *any* invalidation
    // of function analyses by the proxy, but that's OK because we've taken
    // care to invalidate analyses in the function analysis manager
    // incrementally above.
    PA.preserve<FunctionAnalysisManagerModuleProxy>();
    return PA;
  }

  static StringRef name() { return "ModuleToFunctionPassAdaptor"; }

private:
  FunctionPassT Pass;
};

/// \brief A function to deduce a function pass type and wrap it in the
/// templated adaptor.
template <typename FunctionPassT>
ModuleToFunctionPassAdaptor<FunctionPassT>
createModuleToFunctionPassAdaptor(FunctionPassT Pass) {
  return ModuleToFunctionPassAdaptor<FunctionPassT>(std::move(Pass));
}

/// \brief A template utility pass to force an analysis result to be available.
///
/// This is a no-op pass which simply forces a specific analysis pass's result
/// to be available when it is run.
template <typename AnalysisT> struct RequireAnalysisPass {
  /// \brief Run this pass over some unit of IR.
  ///
  /// This pass can be run over any unit of IR and use any analysis manager
  /// provided they satisfy the basic API requirements. When this pass is
  /// created, these methods can be instantiated to satisfy whatever the
  /// context requires.
  template <typename IRUnitT>
  PreservedAnalyses run(IRUnitT &Arg, AnalysisManager<IRUnitT> *AM) {
    if (AM)
      (void)AM->template getResult<AnalysisT>(Arg);

    return PreservedAnalyses::all();
  }

  static StringRef name() { return "RequireAnalysisPass"; }
};

/// \brief A template utility pass to force an analysis result to be
/// invalidated.
///
/// This is a no-op pass which simply forces a specific analysis result to be
/// invalidated when it is run.
template <typename AnalysisT> struct InvalidateAnalysisPass {
  /// \brief Run this pass over some unit of IR.
  ///
  /// This pass can be run over any unit of IR and use any analysis manager
  /// provided they satisfy the basic API requirements. When this pass is
  /// created, these methods can be instantiated to satisfy whatever the
  /// context requires.
  template <typename IRUnitT>
  PreservedAnalyses run(IRUnitT &Arg, AnalysisManager<IRUnitT> *AM) {
    if (AM)
      // We have to directly invalidate the analysis result as we can't
      // enumerate all other analyses and use the preserved set to control it.
      (void)AM->template invalidate<AnalysisT>(Arg);

    return PreservedAnalyses::all();
  }

  static StringRef name() { return "InvalidateAnalysisPass"; }
};

/// \brief A utility pass that does nothing but preserves no analyses.
///
/// As a consequence fo not preserving any analyses, this pass will force all
/// analysis passes to be re-run to produce fresh results if any are needed.
struct InvalidateAllAnalysesPass {
  /// \brief Run this pass over some unit of IR.
  template <typename IRUnitT> PreservedAnalyses run(IRUnitT &Arg) {
    return PreservedAnalyses::none();
  }

  static StringRef name() { return "InvalidateAllAnalysesPass"; }
};

}

#endif

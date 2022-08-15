//===- llvm/Analysis/AssumptionCache.h - Track @llvm.assume ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a pass that keeps track of @llvm.assume intrinsics in
// the functions of a module (allowing assumptions within any function to be
// found cheaply by other parts of the optimizer).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_ASSUMPTIONCACHE_H
#define LLVM_ANALYSIS_ASSUMPTIONCACHE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Pass.h"
#include <memory>

namespace llvm {

// FIXME: Replace this brittle forward declaration with the include of the new
// PassManager.h when doing so doesn't break the PassManagerBuilder.
template <typename IRUnitT> class AnalysisManager;
class PreservedAnalyses;

/// \brief A cache of @llvm.assume calls within a function.
///
/// This cache provides fast lookup of assumptions within a function by caching
/// them and amortizing the cost of scanning for them across all queries. The
/// cache is also conservatively self-updating so that it will never return
/// incorrect results about a function even as the function is being mutated.
/// However, flushing the cache and rebuilding it (or explicitly updating it)
/// may allow it to discover new assumptions.
class AssumptionCache {
  /// \brief The function for which this cache is handling assumptions.
  ///
  /// We track this to lazily populate our assumptions.
  Function &F;

  /// \brief Vector of weak value handles to calls of the @llvm.assume
  /// intrinsic.
  SmallVector<WeakVH, 4> AssumeHandles;

  /// \brief Flag tracking whether we have scanned the function yet.
  ///
  /// We want to be as lazy about this as possible, and so we scan the function
  /// at the last moment.
  bool Scanned;

  /// \brief Scan the function for assumptions and add them to the cache.
  void scanFunction();

public:
  /// \brief Construct an AssumptionCache from a function by scanning all of
  /// its instructions.
  AssumptionCache(Function &F) : F(F), Scanned(false) {}

  /// \brief Add an @llvm.assume intrinsic to this function's cache.
  ///
  /// The call passed in must be an instruction within this fuction and must
  /// not already be in the cache.
  void registerAssumption(CallInst *CI);

  /// \brief Clear the cache of @llvm.assume intrinsics for a function.
  ///
  /// It will be re-scanned the next time it is requested.
  void clear() {
    AssumeHandles.clear();
    Scanned = false;
  }

  /// \brief Access the list of assumption handles currently tracked for this
  /// fuction.
  ///
  /// Note that these produce weak handles that may be null. The caller must
  /// handle that case.
  /// FIXME: We should replace this with pointee_iterator<filter_iterator<...>>
  /// when we can write that to filter out the null values. Then caller code
  /// will become simpler.
  MutableArrayRef<WeakVH> assumptions() {
    if (!Scanned)
      scanFunction();
    return AssumeHandles;
  }
};

/// \brief A function analysis which provides an \c AssumptionCache.
///
/// This analysis is intended for use with the new pass manager and will vend
/// assumption caches for a given function.
class AssumptionAnalysis {
  static char PassID;

public:
  typedef AssumptionCache Result;

  /// \brief Opaque, unique identifier for this analysis pass.
  static void *ID() { return (void *)&PassID; }

  /// \brief Provide a name for the analysis for debugging and logging.
  static StringRef name() { return "AssumptionAnalysis"; }

  AssumptionAnalysis() {}
  AssumptionAnalysis(const AssumptionAnalysis &Arg) {}
  AssumptionAnalysis(AssumptionAnalysis &&Arg) {}
  AssumptionAnalysis &operator=(const AssumptionAnalysis &RHS) { return *this; }
  AssumptionAnalysis &operator=(AssumptionAnalysis &&RHS) { return *this; }

  AssumptionCache run(Function &F) { return AssumptionCache(F); }
};

/// \brief Printer pass for the \c AssumptionAnalysis results.
class AssumptionPrinterPass {
  raw_ostream &OS;

public:
  explicit AssumptionPrinterPass(raw_ostream &OS) : OS(OS) {}
  PreservedAnalyses run(Function &F, AnalysisManager<Function> *AM);

  static StringRef name() { return "AssumptionPrinterPass"; }
};

/// \brief An immutable pass that tracks lazily created \c AssumptionCache
/// objects.
///
/// This is essentially a workaround for the legacy pass manager's weaknesses
/// which associates each assumption cache with Function and clears it if the
/// function is deleted. The nature of the AssumptionCache is that it is not
/// invalidated by any changes to the function body and so this is sufficient
/// to be conservatively correct.
class AssumptionCacheTracker : public ImmutablePass {
  /// A callback value handle applied to function objects, which we use to
  /// delete our cache of intrinsics for a function when it is deleted.
  class FunctionCallbackVH : public CallbackVH {
    AssumptionCacheTracker *ACT;
    void deleted() override;

  public:
    typedef DenseMapInfo<Value *> DMI;

    FunctionCallbackVH(Value *V, AssumptionCacheTracker *ACT = nullptr)
        : CallbackVH(V), ACT(ACT) {}
  };

  friend FunctionCallbackVH;

  typedef DenseMap<FunctionCallbackVH, std::unique_ptr<AssumptionCache>,
                   FunctionCallbackVH::DMI> FunctionCallsMap;
  FunctionCallsMap AssumptionCaches;

public:
  /// \brief Get the cached assumptions for a function.
  ///
  /// If no assumptions are cached, this will scan the function. Otherwise, the
  /// existing cache will be returned.
  AssumptionCache &getAssumptionCache(Function &F);

  AssumptionCacheTracker();
  ~AssumptionCacheTracker() override;

  void releaseMemory() override { AssumptionCaches.shrink_and_clear(); }

  void verifyAnalysis() const override;
  bool doFinalization(Module &) override {
    verifyAnalysis();
    return false;
  }

  static char ID; // Pass identification, replacement for typeid
};

} // end namespace llvm

#endif

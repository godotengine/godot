//===- PassManager internal APIs and implementation details -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
///
/// This header provides internal APIs and implementation details used by the
/// pass management interfaces exposed in PassManager.h. To understand more
/// context of why these particular interfaces are needed, see that header
/// file. None of these APIs should be used elsewhere.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_PASSMANAGERINTERNAL_H
#define LLVM_IR_PASSMANAGERINTERNAL_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/STLExtras.h"

namespace llvm {

template <typename IRUnitT> class AnalysisManager;
class PreservedAnalyses;

/// \brief Implementation details of the pass manager interfaces.
namespace detail {

/// \brief Template for the abstract base class used to dispatch
/// polymorphically over pass objects.
template <typename IRUnitT> struct PassConcept {
  // Boiler plate necessary for the container of derived classes.
  virtual ~PassConcept() {}

  /// \brief The polymorphic API which runs the pass over a given IR entity.
  ///
  /// Note that actual pass object can omit the analysis manager argument if
  /// desired. Also that the analysis manager may be null if there is no
  /// analysis manager in the pass pipeline.
  virtual PreservedAnalyses run(IRUnitT &IR, AnalysisManager<IRUnitT> *AM) = 0;

  /// \brief Polymorphic method to access the name of a pass.
  virtual StringRef name() = 0;
};

/// \brief SFINAE metafunction for computing whether \c PassT has a run method
/// accepting an \c AnalysisManager<IRUnitT>.
template <typename IRUnitT, typename PassT, typename ResultT>
class PassRunAcceptsAnalysisManager {
  typedef char SmallType;
  struct BigType {
    char a, b;
  };

  template <typename T, ResultT (T::*)(IRUnitT &, AnalysisManager<IRUnitT> *)>
  struct Checker;

  template <typename T> static SmallType f(Checker<T, &T::run> *);
  template <typename T> static BigType f(...);

public:
  enum { Value = sizeof(f<PassT>(nullptr)) == sizeof(SmallType) };
};

/// \brief A template wrapper used to implement the polymorphic API.
///
/// Can be instantiated for any object which provides a \c run method accepting
/// an \c IRUnitT. It requires the pass to be a copyable object. When the
/// \c run method also accepts an \c AnalysisManager<IRUnitT>*, we pass it
/// along.
template <typename IRUnitT, typename PassT,
          typename PreservedAnalysesT = PreservedAnalyses,
          bool AcceptsAnalysisManager = PassRunAcceptsAnalysisManager<
              IRUnitT, PassT, PreservedAnalysesT>::Value>
struct PassModel;

/// \brief Specialization of \c PassModel for passes that accept an analyis
/// manager.
template <typename IRUnitT, typename PassT, typename PreservedAnalysesT>
struct PassModel<IRUnitT, PassT, PreservedAnalysesT, true>
    : PassConcept<IRUnitT> {
  explicit PassModel(PassT Pass) : Pass(std::move(Pass)) {}
  // We have to explicitly define all the special member functions because MSVC
  // refuses to generate them.
  PassModel(const PassModel &Arg) : Pass(Arg.Pass) {}
  PassModel(PassModel &&Arg) : Pass(std::move(Arg.Pass)) {}
  friend void swap(PassModel &LHS, PassModel &RHS) {
    using std::swap;
    swap(LHS.Pass, RHS.Pass);
  }
  PassModel &operator=(PassModel RHS) {
    swap(*this, RHS);
    return *this;
  }

  PreservedAnalysesT run(IRUnitT &IR, AnalysisManager<IRUnitT> *AM) override {
    return Pass.run(IR, AM);
  }
  StringRef name() override { return PassT::name(); }
  PassT Pass;
};

/// \brief Specialization of \c PassModel for passes that accept an analyis
/// manager.
template <typename IRUnitT, typename PassT, typename PreservedAnalysesT>
struct PassModel<IRUnitT, PassT, PreservedAnalysesT, false>
    : PassConcept<IRUnitT> {
  explicit PassModel(PassT Pass) : Pass(std::move(Pass)) {}
  // We have to explicitly define all the special member functions because MSVC
  // refuses to generate them.
  PassModel(const PassModel &Arg) : Pass(Arg.Pass) {}
  PassModel(PassModel &&Arg) : Pass(std::move(Arg.Pass)) {}
  friend void swap(PassModel &LHS, PassModel &RHS) {
    using std::swap;
    swap(LHS.Pass, RHS.Pass);
  }
  PassModel &operator=(PassModel RHS) {
    swap(*this, RHS);
    return *this;
  }

  PreservedAnalysesT run(IRUnitT &IR, AnalysisManager<IRUnitT> *AM) override {
    return Pass.run(IR);
  }
  StringRef name() override { return PassT::name(); }
  PassT Pass;
};

/// \brief Abstract concept of an analysis result.
///
/// This concept is parameterized over the IR unit that this result pertains
/// to.
template <typename IRUnitT> struct AnalysisResultConcept {
  virtual ~AnalysisResultConcept() {}

  /// \brief Method to try and mark a result as invalid.
  ///
  /// When the outer analysis manager detects a change in some underlying
  /// unit of the IR, it will call this method on all of the results cached.
  ///
  /// This method also receives a set of preserved analyses which can be used
  /// to avoid invalidation because the pass which changed the underlying IR
  /// took care to update or preserve the analysis result in some way.
  ///
  /// \returns true if the result is indeed invalid (the default).
  virtual bool invalidate(IRUnitT &IR, const PreservedAnalyses &PA) = 0;
};

/// \brief SFINAE metafunction for computing whether \c ResultT provides an
/// \c invalidate member function.
template <typename IRUnitT, typename ResultT> class ResultHasInvalidateMethod {
  typedef char SmallType;
  struct BigType {
    char a, b;
  };

  template <typename T, bool (T::*)(IRUnitT &, const PreservedAnalyses &)>
  struct Checker;

  template <typename T> static SmallType f(Checker<T, &T::invalidate> *);
  template <typename T> static BigType f(...);

public:
  enum { Value = sizeof(f<ResultT>(nullptr)) == sizeof(SmallType) };
};

/// \brief Wrapper to model the analysis result concept.
///
/// By default, this will implement the invalidate method with a trivial
/// implementation so that the actual analysis result doesn't need to provide
/// an invalidation handler. It is only selected when the invalidation handler
/// is not part of the ResultT's interface.
template <typename IRUnitT, typename PassT, typename ResultT,
          typename PreservedAnalysesT = PreservedAnalyses,
          bool HasInvalidateHandler =
              ResultHasInvalidateMethod<IRUnitT, ResultT>::Value>
struct AnalysisResultModel;

/// \brief Specialization of \c AnalysisResultModel which provides the default
/// invalidate functionality.
template <typename IRUnitT, typename PassT, typename ResultT,
          typename PreservedAnalysesT>
struct AnalysisResultModel<IRUnitT, PassT, ResultT, PreservedAnalysesT, false>
    : AnalysisResultConcept<IRUnitT> {
  explicit AnalysisResultModel(ResultT Result) : Result(std::move(Result)) {}
  // We have to explicitly define all the special member functions because MSVC
  // refuses to generate them.
  AnalysisResultModel(const AnalysisResultModel &Arg) : Result(Arg.Result) {}
  AnalysisResultModel(AnalysisResultModel &&Arg)
      : Result(std::move(Arg.Result)) {}
  friend void swap(AnalysisResultModel &LHS, AnalysisResultModel &RHS) {
    using std::swap;
    swap(LHS.Result, RHS.Result);
  }
  AnalysisResultModel &operator=(AnalysisResultModel RHS) {
    swap(*this, RHS);
    return *this;
  }

  /// \brief The model bases invalidation solely on being in the preserved set.
  //
  // FIXME: We should actually use two different concepts for analysis results
  // rather than two different models, and avoid the indirect function call for
  // ones that use the trivial behavior.
  bool invalidate(IRUnitT &, const PreservedAnalysesT &PA) override {
    return !PA.preserved(PassT::ID());
  }

  ResultT Result;
};

/// \brief Specialization of \c AnalysisResultModel which delegates invalidate
/// handling to \c ResultT.
template <typename IRUnitT, typename PassT, typename ResultT,
          typename PreservedAnalysesT>
struct AnalysisResultModel<IRUnitT, PassT, ResultT, PreservedAnalysesT, true>
    : AnalysisResultConcept<IRUnitT> {
  explicit AnalysisResultModel(ResultT Result) : Result(std::move(Result)) {}
  // We have to explicitly define all the special member functions because MSVC
  // refuses to generate them.
  AnalysisResultModel(const AnalysisResultModel &Arg) : Result(Arg.Result) {}
  AnalysisResultModel(AnalysisResultModel &&Arg)
      : Result(std::move(Arg.Result)) {}
  friend void swap(AnalysisResultModel &LHS, AnalysisResultModel &RHS) {
    using std::swap;
    swap(LHS.Result, RHS.Result);
  }
  AnalysisResultModel &operator=(AnalysisResultModel RHS) {
    swap(*this, RHS);
    return *this;
  }

  /// \brief The model delegates to the \c ResultT method.
  bool invalidate(IRUnitT &IR, const PreservedAnalysesT &PA) override {
    return Result.invalidate(IR, PA);
  }

  ResultT Result;
};

/// \brief Abstract concept of an analysis pass.
///
/// This concept is parameterized over the IR unit that it can run over and
/// produce an analysis result.
template <typename IRUnitT> struct AnalysisPassConcept {
  virtual ~AnalysisPassConcept() {}

  /// \brief Method to run this analysis over a unit of IR.
  /// \returns A unique_ptr to the analysis result object to be queried by
  /// users.
  virtual std::unique_ptr<AnalysisResultConcept<IRUnitT>>
  run(IRUnitT &IR, AnalysisManager<IRUnitT> *AM) = 0;

  /// \brief Polymorphic method to access the name of a pass.
  virtual StringRef name() = 0;
};

/// \brief Wrapper to model the analysis pass concept.
///
/// Can wrap any type which implements a suitable \c run method. The method
/// must accept the IRUnitT as an argument and produce an object which can be
/// wrapped in a \c AnalysisResultModel.
template <typename IRUnitT, typename PassT,
          bool AcceptsAnalysisManager = PassRunAcceptsAnalysisManager<
              IRUnitT, PassT, typename PassT::Result>::Value>
struct AnalysisPassModel;

/// \brief Specialization of \c AnalysisPassModel which passes an
/// \c AnalysisManager to PassT's run method.
template <typename IRUnitT, typename PassT>
struct AnalysisPassModel<IRUnitT, PassT, true> : AnalysisPassConcept<IRUnitT> {
  explicit AnalysisPassModel(PassT Pass) : Pass(std::move(Pass)) {}
  // We have to explicitly define all the special member functions because MSVC
  // refuses to generate them.
  AnalysisPassModel(const AnalysisPassModel &Arg) : Pass(Arg.Pass) {}
  AnalysisPassModel(AnalysisPassModel &&Arg) : Pass(std::move(Arg.Pass)) {}
  friend void swap(AnalysisPassModel &LHS, AnalysisPassModel &RHS) {
    using std::swap;
    swap(LHS.Pass, RHS.Pass);
  }
  AnalysisPassModel &operator=(AnalysisPassModel RHS) {
    swap(*this, RHS);
    return *this;
  }

  // FIXME: Replace PassT::Result with type traits when we use C++11.
  typedef AnalysisResultModel<IRUnitT, PassT, typename PassT::Result>
      ResultModelT;

  /// \brief The model delegates to the \c PassT::run method.
  ///
  /// The return is wrapped in an \c AnalysisResultModel.
  std::unique_ptr<AnalysisResultConcept<IRUnitT>>
  run(IRUnitT &IR, AnalysisManager<IRUnitT> *AM) override {
    return make_unique<ResultModelT>(Pass.run(IR, AM));
  }

  /// \brief The model delegates to a static \c PassT::name method.
  ///
  /// The returned string ref must point to constant immutable data!
  StringRef name() override { return PassT::name(); }

  PassT Pass;
};

/// \brief Specialization of \c AnalysisPassModel which does not pass an
/// \c AnalysisManager to PassT's run method.
template <typename IRUnitT, typename PassT>
struct AnalysisPassModel<IRUnitT, PassT, false> : AnalysisPassConcept<IRUnitT> {
  explicit AnalysisPassModel(PassT Pass) : Pass(std::move(Pass)) {}
  // We have to explicitly define all the special member functions because MSVC
  // refuses to generate them.
  AnalysisPassModel(const AnalysisPassModel &Arg) : Pass(Arg.Pass) {}
  AnalysisPassModel(AnalysisPassModel &&Arg) : Pass(std::move(Arg.Pass)) {}
  friend void swap(AnalysisPassModel &LHS, AnalysisPassModel &RHS) {
    using std::swap;
    swap(LHS.Pass, RHS.Pass);
  }
  AnalysisPassModel &operator=(AnalysisPassModel RHS) {
    swap(*this, RHS);
    return *this;
  }

  // FIXME: Replace PassT::Result with type traits when we use C++11.
  typedef AnalysisResultModel<IRUnitT, PassT, typename PassT::Result>
      ResultModelT;

  /// \brief The model delegates to the \c PassT::run method.
  ///
  /// The return is wrapped in an \c AnalysisResultModel.
  std::unique_ptr<AnalysisResultConcept<IRUnitT>>
  run(IRUnitT &IR, AnalysisManager<IRUnitT> *) override {
    return make_unique<ResultModelT>(Pass.run(IR));
  }

  /// \brief The model delegates to a static \c PassT::name method.
  ///
  /// The returned string ref must point to constant immutable data!
  StringRef name() override { return PassT::name(); }

  PassT Pass;
};

} // End namespace detail
}

#endif

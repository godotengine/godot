//===-- ExecutionUtils.h - Utilities for executing code in Orc --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Contains utilities for executing code in Orc.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_EXECUTIONENGINE_ORC_EXECUTIONUTILS_H
#define LLVM_EXECUTIONENGINE_ORC_EXECUTIONUTILS_H

#include "JITSymbol.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ExecutionEngine/RuntimeDyld.h"
#include <vector>

namespace llvm {

class ConstantArray;
class GlobalVariable;
class Function;
class Module;
class Value;

namespace orc {

/// @brief This iterator provides a convenient way to iterate over the elements
///        of an llvm.global_ctors/llvm.global_dtors instance.
///
///   The easiest way to get hold of instances of this class is to use the
/// getConstructors/getDestructors functions.
class CtorDtorIterator {
public:

  /// @brief Accessor for an element of the global_ctors/global_dtors array.
  ///
  ///   This class provides a read-only view of the element with any casts on
  /// the function stripped away.
  struct Element {
    Element(unsigned Priority, const Function *Func, const Value *Data)
      : Priority(Priority), Func(Func), Data(Data) {}

    unsigned Priority;
    const Function *Func;
    const Value *Data;
  };

  /// @brief Construct an iterator instance. If End is true then this iterator
  ///        acts as the end of the range, otherwise it is the beginning.
  CtorDtorIterator(const GlobalVariable *GV, bool End);

  /// @brief Test iterators for equality.
  bool operator==(const CtorDtorIterator &Other) const;

  /// @brief Test iterators for inequality.
  bool operator!=(const CtorDtorIterator &Other) const;

  /// @brief Pre-increment iterator.
  CtorDtorIterator& operator++();

  /// @brief Post-increment iterator.
  CtorDtorIterator operator++(int);

  /// @brief Dereference iterator. The resulting value provides a read-only view
  ///        of this element of the global_ctors/global_dtors list.
  Element operator*() const;

private:
  const ConstantArray *InitList;
  unsigned I;
};

/// @brief Create an iterator range over the entries of the llvm.global_ctors
///        array.
iterator_range<CtorDtorIterator> getConstructors(const Module &M);

/// @brief Create an iterator range over the entries of the llvm.global_ctors
///        array.
iterator_range<CtorDtorIterator> getDestructors(const Module &M);

/// @brief Convenience class for recording constructor/destructor names for
///        later execution.
template <typename JITLayerT>
class CtorDtorRunner {
public:

  /// @brief Construct a CtorDtorRunner for the given range using the given
  ///        name mangling function.
  CtorDtorRunner(std::vector<std::string> CtorDtorNames,
                 typename JITLayerT::ModuleSetHandleT H)
      : CtorDtorNames(std::move(CtorDtorNames)), H(H) {}

  /// @brief Run the recorded constructors/destructors through the given JIT
  ///        layer.
  bool runViaLayer(JITLayerT &JITLayer) const {
    typedef void (*CtorDtorTy)();

    bool Error = false;
    for (const auto &CtorDtorName : CtorDtorNames)
      if (auto CtorDtorSym = JITLayer.findSymbolIn(H, CtorDtorName, false)) {
        CtorDtorTy CtorDtor =
          reinterpret_cast<CtorDtorTy>(
            static_cast<uintptr_t>(CtorDtorSym.getAddress()));
        CtorDtor();
      } else
        Error = true;
    return !Error;
  }

private:
  std::vector<std::string> CtorDtorNames;
  typename JITLayerT::ModuleSetHandleT H;
};

/// @brief Support class for static dtor execution. For hosted (in-process) JITs
///        only!
///
///   If a __cxa_atexit function isn't found C++ programs that use static
/// destructors will fail to link. However, we don't want to use the host
/// process's __cxa_atexit, because it will schedule JIT'd destructors to run
/// after the JIT has been torn down, which is no good. This class makes it easy
/// to override __cxa_atexit (and the related __dso_handle).
///
///   To use, clients should manually call searchOverrides from their symbol
/// resolver. This should generally be done after attempting symbol resolution
/// inside the JIT, but before searching the host process's symbol table. When
/// the client determines that destructors should be run (generally at JIT
/// teardown or after a return from main), the runDestructors method should be
/// called.
class LocalCXXRuntimeOverrides {
public:

  /// Create a runtime-overrides class.
  template <typename MangleFtorT>
  LocalCXXRuntimeOverrides(const MangleFtorT &Mangle) {
    addOverride(Mangle("__dso_handle"), toTargetAddress(&DSOHandleOverride));
    addOverride(Mangle("__cxa_atexit"), toTargetAddress(&CXAAtExitOverride));
  }

  /// Search overrided symbols.
  RuntimeDyld::SymbolInfo searchOverrides(const std::string &Name) {
    auto I = CXXRuntimeOverrides.find(Name);
    if (I != CXXRuntimeOverrides.end())
      return RuntimeDyld::SymbolInfo(I->second, JITSymbolFlags::Exported);
    return nullptr;
  }

  /// Run any destructors recorded by the overriden __cxa_atexit function
  /// (CXAAtExitOverride).
  void runDestructors();

private:

  template <typename PtrTy>
  TargetAddress toTargetAddress(PtrTy* P) {
    return static_cast<TargetAddress>(reinterpret_cast<uintptr_t>(P));
  }

  void addOverride(const std::string &Name, TargetAddress Addr) {
    CXXRuntimeOverrides.insert(std::make_pair(Name, Addr));
  }

  StringMap<TargetAddress> CXXRuntimeOverrides;

  typedef void (*DestructorPtr)(void*);
  typedef std::pair<DestructorPtr, void*> CXXDestructorDataPair;
  typedef std::vector<CXXDestructorDataPair> CXXDestructorDataPairList;
  CXXDestructorDataPairList DSOHandleOverride;
  static int CXAAtExitOverride(DestructorPtr Destructor, void *Arg,
                               void *DSOHandle);
};

} // End namespace orc.
} // End namespace llvm.

#endif // LLVM_EXECUTIONENGINE_ORC_EXECUTIONUTILS_H

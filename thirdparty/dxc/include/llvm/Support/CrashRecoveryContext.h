//===--- CrashRecoveryContext.h - Crash Recovery ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_CRASHRECOVERYCONTEXT_H
#define LLVM_SUPPORT_CRASHRECOVERYCONTEXT_H

#include "llvm/ADT/STLExtras.h"
#include <string>

namespace llvm {
class CrashRecoveryContextCleanup;

/// \brief Crash recovery helper object.
///
/// This class implements support for running operations in a safe context so
/// that crashes (memory errors, stack overflow, assertion violations) can be
/// detected and control restored to the crashing thread. Crash detection is
/// purely "best effort", the exact set of failures which can be recovered from
/// is platform dependent.
///
/// Clients make use of this code by first calling
/// CrashRecoveryContext::Enable(), and then executing unsafe operations via a
/// CrashRecoveryContext object. For example:
///
///    void actual_work(void *);
///
///    void foo() {
///      CrashRecoveryContext CRC;
///
///      if (!CRC.RunSafely(actual_work, 0)) {
///         ... a crash was detected, report error to user ...
///      }
///
///      ... no crash was detected ...
///    }
///
/// Crash recovery contexts may not be nested.
class CrashRecoveryContext {
  void *Impl;
  CrashRecoveryContextCleanup *head;

public:
  CrashRecoveryContext() : Impl(nullptr), head(nullptr) {}
  ~CrashRecoveryContext();

  void registerCleanup(CrashRecoveryContextCleanup *cleanup);
  void unregisterCleanup(CrashRecoveryContextCleanup *cleanup);

  /// \brief Enable crash recovery.
  static void Enable();

  /// \brief Disable crash recovery.
  static void Disable();

  /// \brief Return the active context, if the code is currently executing in a
  /// thread which is in a protected context.
  static CrashRecoveryContext *GetCurrent();

  /// \brief Return true if the current thread is recovering from a
  /// crash.
  static bool isRecoveringFromCrash();

  /// \brief Execute the provide callback function (with the given arguments) in
  /// a protected context.
  ///
  /// \return True if the function completed successfully, and false if the
  /// function crashed (or HandleCrash was called explicitly). Clients should
  /// make as little assumptions as possible about the program state when
  /// RunSafely has returned false. Clients can use getBacktrace() to retrieve
  /// the backtrace of the crash on failures.
  bool RunSafely(function_ref<void()> Fn);
  bool RunSafely(void (*Fn)(void*), void *UserData) {
    return RunSafely([&]() { Fn(UserData); });
  }

  /// \brief Execute the provide callback function (with the given arguments) in
  /// a protected context which is run in another thread (optionally with a
  /// requested stack size).
  ///
  /// See RunSafely() and llvm_execute_on_thread().
  ///
  /// On Darwin, if PRIO_DARWIN_BG is set on the calling thread, it will be
  /// propagated to the new thread as well.
  bool RunSafelyOnThread(function_ref<void()>, unsigned RequestedStackSize = 0);
  bool RunSafelyOnThread(void (*Fn)(void*), void *UserData,
                         unsigned RequestedStackSize = 0) {
    return RunSafelyOnThread([&]() { Fn(UserData); }, RequestedStackSize);
  }

  /// \brief Explicitly trigger a crash recovery in the current process, and
  /// return failure from RunSafely(). This function does not return.
  void HandleCrash();

  /// \brief Return a string containing the backtrace where the crash was
  /// detected; or empty if the backtrace wasn't recovered.
  ///
  /// This function is only valid when a crash has been detected (i.e.,
  /// RunSafely() has returned false.
  const std::string &getBacktrace() const;
};

class CrashRecoveryContextCleanup {
protected:
  CrashRecoveryContext *context;
  CrashRecoveryContextCleanup(CrashRecoveryContext *context)
    : context(context), cleanupFired(false) {}
public:
  bool cleanupFired;
  
  virtual ~CrashRecoveryContextCleanup();
  virtual void recoverResources() = 0;

  CrashRecoveryContext *getContext() const {
    return context;
  }

private:
  friend class CrashRecoveryContext;
  CrashRecoveryContextCleanup *prev, *next;
};

template<typename DERIVED, typename T>
class CrashRecoveryContextCleanupBase : public CrashRecoveryContextCleanup {
protected:
  T *resource;
  CrashRecoveryContextCleanupBase(CrashRecoveryContext *context, T* resource)
    : CrashRecoveryContextCleanup(context), resource(resource) {}
public:
  static DERIVED *create(T *x) {
    if (x) {
      if (CrashRecoveryContext *context = CrashRecoveryContext::GetCurrent())
        return new DERIVED(context, x);
    }
    return 0;
  }
};

template <typename T>
class CrashRecoveryContextDestructorCleanup : public
  CrashRecoveryContextCleanupBase<CrashRecoveryContextDestructorCleanup<T>, T> {
public:
  CrashRecoveryContextDestructorCleanup(CrashRecoveryContext *context,
                                        T *resource) 
    : CrashRecoveryContextCleanupBase<
        CrashRecoveryContextDestructorCleanup<T>, T>(context, resource) {}

  virtual void recoverResources() {
    this->resource->~T();
  }
};

template <typename T>
class CrashRecoveryContextDeleteCleanup : public
  CrashRecoveryContextCleanupBase<CrashRecoveryContextDeleteCleanup<T>, T> {
public:
  CrashRecoveryContextDeleteCleanup(CrashRecoveryContext *context, T *resource)
    : CrashRecoveryContextCleanupBase<
        CrashRecoveryContextDeleteCleanup<T>, T>(context, resource) {}

  void recoverResources() override { delete this->resource; }
};

template <typename T>
class CrashRecoveryContextReleaseRefCleanup : public
  CrashRecoveryContextCleanupBase<CrashRecoveryContextReleaseRefCleanup<T>, T>
{
public:
  CrashRecoveryContextReleaseRefCleanup(CrashRecoveryContext *context, 
                                        T *resource)
    : CrashRecoveryContextCleanupBase<CrashRecoveryContextReleaseRefCleanup<T>,
          T>(context, resource) {}

  void recoverResources() override { this->resource->Release(); }
};

template <typename T, typename Cleanup = CrashRecoveryContextDeleteCleanup<T> >
class CrashRecoveryContextCleanupRegistrar {
  CrashRecoveryContextCleanup *cleanup;
public:
  CrashRecoveryContextCleanupRegistrar(T *x)
    : cleanup(Cleanup::create(x)) {
    if (cleanup)
      cleanup->getContext()->registerCleanup(cleanup);
  }

  ~CrashRecoveryContextCleanupRegistrar() {
    unregister();
  }
  
  void unregister() {
    if (cleanup && !cleanup->cleanupFired)
      cleanup->getContext()->unregisterCleanup(cleanup);
    cleanup = 0;
  }
};
}

#endif

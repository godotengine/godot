//===-- llvm/Support/ManagedStatic.h - Static Global wrapper ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the ManagedStatic class and the llvm_shutdown() function.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_MANAGEDSTATIC_H
#define LLVM_SUPPORT_MANAGEDSTATIC_H

#include "llvm/Support/Atomic.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/Valgrind.h"

namespace llvm {

/// object_creator - Helper method for ManagedStatic.
template<class C>
void* object_creator() {
  return new C();
}

/// object_deleter - Helper method for ManagedStatic.
///
template<typename T> struct object_deleter {
  static void call(void * Ptr) { delete (T*)Ptr; }
};
template<typename T, size_t N> struct object_deleter<T[N]> {
  static void call(void * Ptr) { delete[] (T*)Ptr; }
};

/// ManagedStaticBase - Common base class for ManagedStatic instances.
class ManagedStaticBase {
protected:
  // This should only be used as a static variable, which guarantees that this
  // will be zero initialized.
  mutable void *Ptr;
  mutable void (*DeleterFn)(void*);
  mutable const ManagedStaticBase *Next;

  void RegisterManagedStatic(void *(*creator)(), void (*deleter)(void*)) const;
public:
  /// isConstructed - Return true if this object has not been created yet.
  bool isConstructed() const { return Ptr != nullptr; }

  void destroy() const;
};

/// ManagedStatic - This transparently changes the behavior of global statics to
/// be lazily constructed on demand (good for reducing startup times of dynamic
/// libraries that link in LLVM components) and for making destruction be
/// explicit through the llvm_shutdown() function call.
///
template<class C>
class ManagedStatic : public ManagedStaticBase {
public:

  // Accessors.
  C &operator*() {
    void* tmp = Ptr;
    if (llvm_is_multithreaded()) sys::MemoryFence();
    if (!tmp) RegisterManagedStatic(object_creator<C>, object_deleter<C>::call);
    TsanHappensAfter(this);

    return *static_cast<C*>(Ptr);
  }
  C *operator->() {
    void* tmp = Ptr;
    if (llvm_is_multithreaded()) sys::MemoryFence();
    if (!tmp) RegisterManagedStatic(object_creator<C>, object_deleter<C>::call);
    TsanHappensAfter(this);

    return static_cast<C*>(Ptr);
  }
  const C &operator*() const {
    void* tmp = Ptr;
    if (llvm_is_multithreaded()) sys::MemoryFence();
    if (!tmp) RegisterManagedStatic(object_creator<C>, object_deleter<C>::call);
    TsanHappensAfter(this);

    return *static_cast<C*>(Ptr);
  }
  const C *operator->() const {
    void* tmp = Ptr;
    if (llvm_is_multithreaded()) sys::MemoryFence();
    if (!tmp) RegisterManagedStatic(object_creator<C>, object_deleter<C>::call);
    TsanHappensAfter(this);

    return static_cast<C*>(Ptr);
  }
};

/// llvm_shutdown - Deallocate and destroy all ManagedStatic variables.
void llvm_shutdown();

/// llvm_shutdown_obj - This is a simple helper class that calls
/// llvm_shutdown() when it is destroyed.
struct llvm_shutdown_obj {
  llvm_shutdown_obj() { }
  ~llvm_shutdown_obj() { llvm_shutdown(); }
};

}

#endif

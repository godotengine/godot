//===-- ManagedStatic.cpp - Static Global wrapper -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the ManagedStatic class and llvm_shutdown().
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/ManagedStatic.h"
#include "llvm/Config/config.h"
#include "llvm/Support/Atomic.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/MutexGuard.h"
#include <cassert>
using namespace llvm;

static const ManagedStaticBase *StaticList = nullptr;

static sys::Mutex& getManagedStaticMutex() {
  // We need to use a function local static here, since this can get called
  // during a static constructor and we need to guarantee that it's initialized
  // correctly.
  static sys::Mutex ManagedStaticMutex;
  return ManagedStaticMutex;
}

void ManagedStaticBase::RegisterManagedStatic(void *(*Creator)(),
                                              void (*Deleter)(void*)) const {
  assert(Creator);
  if (llvm_is_multithreaded()) {
    MutexGuard Lock(getManagedStaticMutex());

    if (!Ptr) {
      void* tmp = Creator();

      TsanHappensBefore(this);
      sys::MemoryFence();

      // This write is racy against the first read in the ManagedStatic
      // accessors. The race is benign because it does a second read after a
      // memory fence, at which point it isn't possible to get a partial value.
      TsanIgnoreWritesBegin();
      Ptr = tmp;
      TsanIgnoreWritesEnd();
      DeleterFn = Deleter;
      
      // Add to list of managed statics.
      Next = StaticList;
      StaticList = this;
    }
  } else {
    assert(!Ptr && !DeleterFn && !Next &&
           "Partially initialized ManagedStatic!?");
    Ptr = Creator();
    DeleterFn = Deleter;
  
    // Add to list of managed statics.
    Next = StaticList;
    StaticList = this;
  }
}

void ManagedStaticBase::destroy() const {
  assert(DeleterFn && "ManagedStatic not initialized correctly!");
  assert(StaticList == this &&
         "Not destroyed in reverse order of construction?");
  // Unlink from list.
  StaticList = Next;
  Next = nullptr;

  // Destroy memory.
  DeleterFn(Ptr);
  
  // Cleanup.
  Ptr = nullptr;
  DeleterFn = nullptr;
}

/// llvm_shutdown - Deallocate and destroy all ManagedStatic variables.
void llvm::llvm_shutdown() {
  MutexGuard Lock(getManagedStaticMutex());

  while (StaticList)
    StaticList->destroy();
}

//===- llvm/Support/Atomic.h - Atomic Operations -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the llvm::sys atomic operations.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_ATOMIC_H
#define LLVM_SUPPORT_ATOMIC_H

#include "llvm/Support/DataTypes.h"

namespace llvm {
  namespace sys {
    void MemoryFence();

#ifdef _MSC_VER
    typedef long cas_flag;
#else
    typedef uint32_t cas_flag;
#endif
    cas_flag CompareAndSwap(volatile cas_flag* ptr,
                            cas_flag new_value,
                            cas_flag old_value);
    cas_flag AtomicIncrement(volatile cas_flag* ptr);
    cas_flag AtomicDecrement(volatile cas_flag* ptr);
    cas_flag AtomicAdd(volatile cas_flag* ptr, cas_flag val);
    cas_flag AtomicMul(volatile cas_flag* ptr, cas_flag val);
    cas_flag AtomicDiv(volatile cas_flag* ptr, cas_flag val);
  }
}

#endif

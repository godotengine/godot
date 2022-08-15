//===- llvm/Support/PointerLikeTypeTraits.h - Pointer Traits ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the PointerLikeTypeTraits class.  This allows data
// structures to reason about pointers and other things that are pointer sized.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_POINTERLIKETYPETRAITS_H
#define LLVM_SUPPORT_POINTERLIKETYPETRAITS_H

#include "llvm/Support/DataTypes.h"

namespace llvm {
  
/// PointerLikeTypeTraits - This is a traits object that is used to handle
/// pointer types and things that are just wrappers for pointers as a uniform
/// entity.
template <typename T>
class PointerLikeTypeTraits {
  // getAsVoidPointer
  // getFromVoidPointer
  // getNumLowBitsAvailable
};

// Provide PointerLikeTypeTraits for non-cvr pointers.
template<typename T>
class PointerLikeTypeTraits<T*> {
public:
  static inline void *getAsVoidPointer(T* P) { return P; }
  static inline T *getFromVoidPointer(void *P) {
    return static_cast<T*>(P);
  }
  
  /// Note, we assume here that malloc returns objects at least 4-byte aligned.
  /// However, this may be wrong, or pointers may be from something other than
  /// malloc.  In this case, you should specialize this template to reduce this.
  ///
  /// All clients should use assertions to do a run-time check to ensure that
  /// this is actually true.
  enum { NumLowBitsAvailable = 2 };
};
  
// Provide PointerLikeTypeTraits for const pointers.
template<typename T>
class PointerLikeTypeTraits<const T*> {
  typedef PointerLikeTypeTraits<T*> NonConst;

public:
  static inline const void *getAsVoidPointer(const T* P) {
    return NonConst::getAsVoidPointer(const_cast<T*>(P));
  }
  static inline const T *getFromVoidPointer(const void *P) {
    return NonConst::getFromVoidPointer(const_cast<void*>(P));
  }
  enum { NumLowBitsAvailable = NonConst::NumLowBitsAvailable };
};

// Provide PointerLikeTypeTraits for uintptr_t.
template<>
class PointerLikeTypeTraits<uintptr_t> {
public:
  static inline void *getAsVoidPointer(uintptr_t P) {
    return reinterpret_cast<void*>(P);
  }
  static inline uintptr_t getFromVoidPointer(void *P) {
    return reinterpret_cast<uintptr_t>(P);
  }
  // No bits are available!
  enum { NumLowBitsAvailable = 0 };
};
  
} // end namespace llvm

#endif

//===-- Optional.h - Simple variant for passing optional values ---*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file provides Optional, a template class modeled in the spirit of
//  OCaml's 'opt' variant.  The idea is to strongly type whether or not
//  a value can be optional.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_OPTIONAL_H
#define LLVM_ADT_OPTIONAL_H

#include "llvm/ADT/None.h"
#include "llvm/Support/AlignOf.h"
#include "llvm/Support/Compiler.h"
#include <cassert>
#include <new>
#include <utility>

namespace llvm {

template<typename T>
class Optional {
  AlignedCharArrayUnion<T> storage;
  bool hasVal;
public:
  typedef T value_type;

  Optional(NoneType) : hasVal(false) {}
  explicit Optional() : hasVal(false) {}
  Optional(const T &y) : hasVal(true) {
    new (storage.buffer) T(y);
  }
  Optional(const Optional &O) : hasVal(O.hasVal) {
    if (hasVal)
      new (storage.buffer) T(*O);
  }

  Optional(T &&y) : hasVal(true) {
    new (storage.buffer) T(std::forward<T>(y));
  }
  Optional(Optional<T> &&O) : hasVal(O) {
    if (O) {
      new (storage.buffer) T(std::move(*O));
      O.reset();
    }
  }
  Optional &operator=(T &&y) {
    if (hasVal)
      **this = std::move(y);
    else {
      new (storage.buffer) T(std::move(y));
      hasVal = true;
    }
    return *this;
  }
  Optional &operator=(Optional &&O) {
    if (!O)
      reset();
    else {
      *this = std::move(*O);
      O.reset();
    }
    return *this;
  }

  /// Create a new object by constructing it in place with the given arguments.
  template<typename ...ArgTypes>
  void emplace(ArgTypes &&...Args) {
    reset();
    hasVal = true;
    new (storage.buffer) T(std::forward<ArgTypes>(Args)...);
  }

  static inline Optional create(const T* y) {
    return y ? Optional(*y) : Optional();
  }

  // FIXME: these assignments (& the equivalent const T&/const Optional& ctors)
  // could be made more efficient by passing by value, possibly unifying them
  // with the rvalue versions above - but this could place a different set of
  // requirements (notably: the existence of a default ctor) when implemented
  // in that way. Careful SFINAE to avoid such pitfalls would be required.
  Optional &operator=(const T &y) {
    if (hasVal)
      **this = y;
    else {
      new (storage.buffer) T(y);
      hasVal = true;
    }
    return *this;
  }

  Optional &operator=(const Optional &O) {
    if (!O)
      reset();
    else
      *this = *O;
    return *this;
  }

  void reset() {
    if (hasVal) {
      (**this).~T();
      hasVal = false;
    }
  }

  ~Optional() {
    reset();
  }

  const T* getPointer() const { assert(hasVal); return reinterpret_cast<const T*>(storage.buffer); }
  T* getPointer() { assert(hasVal); return reinterpret_cast<T*>(storage.buffer); }
  const T& getValue() const LLVM_LVALUE_FUNCTION { assert(hasVal); return *getPointer(); }
  T& getValue() LLVM_LVALUE_FUNCTION { assert(hasVal); return *getPointer(); }

  explicit operator bool() const { return hasVal; }
  bool hasValue() const { return hasVal; }
  const T* operator->() const { return getPointer(); }
  T* operator->() { return getPointer(); }
  const T& operator*() const LLVM_LVALUE_FUNCTION { assert(hasVal); return *getPointer(); }
  T& operator*() LLVM_LVALUE_FUNCTION { assert(hasVal); return *getPointer(); }

  template <typename U>
  LLVM_CONSTEXPR T getValueOr(U &&value) const LLVM_LVALUE_FUNCTION {
    return hasValue() ? getValue() : std::forward<U>(value);
  }

#if LLVM_HAS_RVALUE_REFERENCE_THIS
  T&& getValue() && { assert(hasVal); return std::move(*getPointer()); }
  T&& operator*() && { assert(hasVal); return std::move(*getPointer()); }

  template <typename U>
  T getValueOr(U &&value) && {
    return hasValue() ? std::move(getValue()) : std::forward<U>(value);
  }
#endif
};

template <typename T> struct isPodLike;
template <typename T> struct isPodLike<Optional<T> > {
  // An Optional<T> is pod-like if T is.
  static const bool value = isPodLike<T>::value;
};

/// \brief Poison comparison between two \c Optional objects. Clients needs to
/// explicitly compare the underlying values and account for empty \c Optional
/// objects.
///
/// This routine will never be defined. It returns \c void to help diagnose
/// errors at compile time.
template<typename T, typename U>
void operator==(const Optional<T> &X, const Optional<U> &Y);

/// \brief Poison comparison between two \c Optional objects. Clients needs to
/// explicitly compare the underlying values and account for empty \c Optional
/// objects.
///
/// This routine will never be defined. It returns \c void to help diagnose
/// errors at compile time.
template<typename T, typename U>
void operator!=(const Optional<T> &X, const Optional<U> &Y);

/// \brief Poison comparison between two \c Optional objects. Clients needs to
/// explicitly compare the underlying values and account for empty \c Optional
/// objects.
///
/// This routine will never be defined. It returns \c void to help diagnose
/// errors at compile time.
template<typename T, typename U>
void operator<(const Optional<T> &X, const Optional<U> &Y);

/// \brief Poison comparison between two \c Optional objects. Clients needs to
/// explicitly compare the underlying values and account for empty \c Optional
/// objects.
///
/// This routine will never be defined. It returns \c void to help diagnose
/// errors at compile time.
template<typename T, typename U>
void operator<=(const Optional<T> &X, const Optional<U> &Y);

/// \brief Poison comparison between two \c Optional objects. Clients needs to
/// explicitly compare the underlying values and account for empty \c Optional
/// objects.
///
/// This routine will never be defined. It returns \c void to help diagnose
/// errors at compile time.
template<typename T, typename U>
void operator>=(const Optional<T> &X, const Optional<U> &Y);

/// \brief Poison comparison between two \c Optional objects. Clients needs to
/// explicitly compare the underlying values and account for empty \c Optional
/// objects.
///
/// This routine will never be defined. It returns \c void to help diagnose
/// errors at compile time.
template<typename T, typename U>
void operator>(const Optional<T> &X, const Optional<U> &Y);

} // end llvm namespace

#endif

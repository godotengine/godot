// Copyright 2016 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef MINI_CHROMIUM_BASE_BIT_CAST_H_
#define MINI_CHROMIUM_BASE_BIT_CAST_H_

#include <string.h>
#include <type_traits>

#include "base/compiler_specific.h"
#include "build/build_config.h"

// bit_cast<Dest,Source> is a template function that implements the equivalent
// of "*reinterpret_cast<Dest*>(&source)".  We need this in very low-level
// functions like the protobuf library and fast math support.
//
//   float f = 3.14159265358979;
//   int i = bit_cast<int32_t>(f);
//   // i = 0x40490fdb
//
// The classical address-casting method is:
//
//   // WRONG
//   float f = 3.14159265358979;            // WRONG
//   int i = * reinterpret_cast<int*>(&f);  // WRONG
//
// The address-casting method actually produces undefined behavior according to
// the ISO C++98 specification, section 3.10 ("basic.lval"), paragraph 15.
// (This did not substantially change in C++11.)  Roughly, this section says: if
// an object in memory has one type, and a program accesses it with a different
// type, then the result is undefined behavior for most values of "different
// type".
//
// This is true for any cast syntax, either *(int*)&f or
// *reinterpret_cast<int*>(&f).  And it is particularly true for conversions
// between integral lvalues and floating-point lvalues.
//
// The purpose of this paragraph is to allow optimizing compilers to assume that
// expressions with different types refer to different memory.  Compilers are
// known to take advantage of this.  So a non-conforming program quietly
// produces wildly incorrect output.
//
// The problem is not the use of reinterpret_cast.  The problem is type punning:
// holding an object in memory of one type and reading its bits back using a
// different type.
//
// The C++ standard is more subtle and complex than this, but that is the basic
// idea.
//
// Anyways ...
//
// bit_cast<> calls memcpy() which is blessed by the standard, especially by the
// example in section 3.9 .  Also, of course, bit_cast<> wraps up the nasty
// logic in one place.
//
// Fortunately memcpy() is very fast.  In optimized mode, compilers replace
// calls to memcpy() with inline object code when the size argument is a
// compile-time constant.  On a 32-bit system, memcpy(d,s,4) compiles to one
// load and one store, and memcpy(d,s,8) compiles to two loads and two stores.

template <class Dest, class Source>
inline Dest bit_cast(const Source& source) {
  static_assert(sizeof(Dest) == sizeof(Source),
                "bit_cast requires source and destination to be the same size");

#if (__GNUC__ > 5 || (__GNUC__ == 5 && __GNUC_MINOR__ >= 1) || \
     defined(_LIBCPP_VERSION))
  // GCC 5.1 contains the first libstdc++ with is_trivially_copyable.
  // Assume libc++ Just Works: is_trivially_copyable added on May 13th 2011.
  static_assert(std::is_trivially_copyable<Dest>::value,
                "non-trivially-copyable bit_cast is undefined");
  static_assert(std::is_trivially_copyable<Source>::value,
                "non-trivially-copyable bit_cast is undefined");
#elif HAS_FEATURE(is_trivially_copyable)
  // The compiler supports an equivalent intrinsic.
  static_assert(__is_trivially_copyable(Dest),
                "non-trivially-copyable bit_cast is undefined");
  static_assert(__is_trivially_copyable(Source),
                "non-trivially-copyable bit_cast is undefined");
#elif COMPILER_GCC
  // Fallback to compiler intrinsic on GCC and clang (which pretends to be
  // GCC). This isn't quite the same as is_trivially_copyable but it'll do for
  // our purpose.
  static_assert(__has_trivial_copy(Dest),
                "non-trivially-copyable bit_cast is undefined");
  static_assert(__has_trivial_copy(Source),
                "non-trivially-copyable bit_cast is undefined");
#else
  // Do nothing, let the bots handle it.
#endif

  Dest dest;
  memcpy(&dest, &source, sizeof(dest));
  return dest;
}

#endif  // MINI_CHROMIUM_BASE_BIT_CAST_H_

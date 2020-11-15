// Copyright 2017 The Crashpad Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef CRASHPAD_UTIL_MISC_FROM_POINTER_CAST_H_
#define CRASHPAD_UTIL_MISC_FROM_POINTER_CAST_H_

#include <stdint.h>

#include <cstddef>
#include <type_traits>

#include "base/numerics/safe_conversions.h"
#include "build/build_config.h"

namespace crashpad {

#if DOXYGEN

//! \brief Casts from a pointer type to an integer.
//!
//! Compared to `reinterpret_cast<>()`, FromPointerCast<>() defines whether a
//! pointer type is sign-extended or zero-extended. Casts to signed integral
//! types are sign-extended. Casts to unsigned integral types are zero-extended.
//!
//! Use FromPointerCast<>() instead of `reinterpret_cast<>()` when casting a
//! pointer to an integral type that may not be the same width as a pointer.
//! There is no need to prefer FromPointerCast<>() when casting to an integral
//! type that’s definitely the same width as a pointer, such as `uintptr_t` and
//! `intptr_t`.
template <typename To, typename From>
FromPointerCast(From from) {
  return reinterpret_cast<To>(from);
}

#else  // DOXYGEN

// Cast std::nullptr_t to any type.
//
// In C++14, the nullptr_t check could use std::is_null_pointer<From>::value
// instead of the is_same<remove_cv<From>::type, nullptr_t>::type construct.
template <typename To, typename From>
typename std::enable_if<
    std::is_same<typename std::remove_cv<From>::type, std::nullptr_t>::value,
    To>::type
FromPointerCast(From) {
  return To();
}

// FromPointerCast<>() with a function pointer “From” type raises
// -Wnoexcept-type in GCC 7.2 if the function pointer type has a throw() or
// noexcept specification. This is the case for all standard C library functions
// provided by glibc. Various tests make use of FromPointerCast<>() with
// pointers to standard C library functions.
//
// Clang has the similar -Wc++1z-compat-mangling, which is not triggered by this
// pattern.
#if defined(COMPILER_GCC) && !defined(__clang__) && \
    (__GNUC__ > 7 || (__GNUC__ == 7 && __GNUC_MINOR__ >= 2))
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnoexcept-type"
#endif

// Cast a pointer to any other pointer type.
template <typename To, typename From>
typename std::enable_if<std::is_pointer<From>::value &&
                            std::is_pointer<To>::value,
                        To>::type
FromPointerCast(From from) {
  return reinterpret_cast<To>(from);
}

// Cast a pointer to an integral type. Sign-extend when casting to a signed
// type, zero-extend when casting to an unsigned type.
template <typename To, typename From>
typename std::enable_if<std::is_pointer<From>::value &&
                            std::is_integral<To>::value,
                        To>::type
FromPointerCast(From from) {
  const auto intermediate =
      reinterpret_cast<typename std::conditional<std::is_signed<To>::value,
                                                 intptr_t,
                                                 uintptr_t>::type>(from);

  if (sizeof(To) >= sizeof(From)) {
    // If the destination integral type is at least as wide as the source
    // pointer type, use static_cast<>() and just return it.
    return static_cast<To>(intermediate);
  }

  // If the destination integral type is narrower than the source pointer type,
  // use checked_cast<>().
  return base::checked_cast<To>(intermediate);
}

#if defined(COMPILER_GCC) && !defined(__clang__) && \
    (__GNUC__ > 7 || (__GNUC__ == 7 && __GNUC_MINOR__ >= 2))
#pragma GCC diagnostic pop
#endif

#endif  // DOXYGEN

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_MISC_FROM_POINTER_CAST_H_

// Copyright 2022 Google LLC
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google LLC nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// safe_math.h: Helpful math functions.
#ifndef SAFE_MATH_H__
#define SAFE_MATH_H__

#include <utility>

namespace google_breakpad {

// Adds `a` and `b`, returning a pair of:
// - The result after any truncation.
// - Whether an overflow/underflow occurred.
template <typename T>
std::pair<T, bool> AddWithOverflowCheck(T a, T b) {
#ifdef _WIN32
  // Since C++11, unsigned overflow is well-defined; do everything unsigned,
  // assuming 2's complement.
  if (std::is_unsigned<T>::value) {
    T result = a + b;
    // Since we're adding two values >= 0, having a smaller value implies
    // overflow.
    bool overflow = result < a;
    return {result, overflow};
  }

  using TUnsigned = typename std::make_unsigned<T>::type;
  T result = TUnsigned(a) + TUnsigned(b);
  bool overflow;
  if ((a >= 0) == (b >= 0)) {
    if (a >= 0) {
      overflow = result < a;
    } else {
      overflow = result > a;
    }
  } else {
    // If signs are different, it's impossible for overflow to happen.
    overflow = false;
  }
  return {result, overflow};
#else
  T result;
  bool overflow = __builtin_add_overflow(a, b, &result);
  return {result, overflow};
#endif
}

template <typename T>
T AddIgnoringOverflow(T a, T b) {
  return AddWithOverflowCheck(a, b).first;
}

}  // namespace google_breakpad

#endif  // SAFE_MATH_H__

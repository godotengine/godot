// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_BASE_RANDOM_
#define LIB_JXL_BASE_RANDOM_

// Random number generator + distributions.
// We don't use <random> because the implementation (and thus results) differs
// between libstdc++ and libc++.

#include <stdint.h>
#include <string.h>

#include <algorithm>
#include <cmath>

#include "lib/jxl/base/status.h"

namespace jxl {
struct Rng {
  explicit Rng(uint64_t seed)
      : s{static_cast<uint64_t>(0x94D049BB133111EBull),
          static_cast<uint64_t>(0xBF58476D1CE4E5B9ull) + seed} {}

  // Xorshift128+ adapted from xorshift128+-inl.h
  uint64_t operator()() {
    uint64_t s1 = s[0];
    const uint64_t s0 = s[1];
    const uint64_t bits = s1 + s0;  // b, c
    s[0] = s0;
    s1 ^= s1 << 23;
    s1 ^= s0 ^ (s1 >> 18) ^ (s0 >> 5);
    s[1] = s1;
    return bits;
  }

  // Uniformly distributed int64_t in [begin, end), under the assumption that
  // `end-begin` is significantly smaller than 1<<64, otherwise there is some
  // bias.
  int64_t UniformI(int64_t begin, int64_t end) {
    JXL_DASSERT(end > begin);
    return static_cast<int64_t>((*this)() %
                                static_cast<uint64_t>(end - begin)) +
           begin;
  }

  // Same as UniformI, but for uint64_t.
  uint64_t UniformU(uint64_t begin, uint64_t end) {
    JXL_DASSERT(end > begin);
    return (*this)() % (end - begin) + begin;
  }

  // Uniformly distributed float in [begin, end) range. Note: only 23 bits of
  // randomness.
  float UniformF(float begin, float end) {
    float f;
    // Bits of a random [1, 2) float.
    uint32_t u = ((*this)() >> (64 - 23)) | 0x3F800000;
    static_assert(sizeof(f) == sizeof(u),
                  "Float and U32 must have the same size");
    memcpy(&f, &u, sizeof(f));
    // Note: (end-begin) * f + (2*begin-end) may fail to return a number >=
    // begin.
    return (end - begin) * (f - 1.0f) + begin;
  }

  // Bernoulli trial
  bool Bernoulli(float p) { return UniformF(0, 1) < p; }

  // State for geometric distributions.
  // The stored value is inv_log_1mp
  using GeometricDistribution = float;
  static GeometricDistribution MakeGeometric(float p) {
    return 1.0 / std::log(1 - p);
  }

  uint32_t Geometric(const GeometricDistribution& dist) {
    float f = UniformF(0, 1);
    float inv_log_1mp = dist;
    float log = std::log(1 - f) * inv_log_1mp;
    return static_cast<uint32_t>(log);
  }

  template <typename T>
  void Shuffle(T* t, size_t n) {
    for (size_t i = 0; i + 1 < n; i++) {
      size_t a = UniformU(i, n);
      std::swap(t[a], t[i]);
    }
  }

 private:
  uint64_t s[2];
};

}  // namespace jxl
#endif  // LIB_JXL_BASE_RANDOM_

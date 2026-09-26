// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_FIELD_ENCODINGS_H_
#define LIB_JXL_FIELD_ENCODINGS_H_

// Constants needed to encode/decode fields; avoids including the full fields.h.

#include <cstddef>
#include <cstdint>
#include <hwy/base.h>
#include <vector>

#include "lib/jxl/base/bits.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/status.h"

namespace jxl {

// Macro to define the Fields' derived class Name when compiling with debug
// names.
#if (JXL_IS_DEBUG_BUILD)
#define JXL_FIELDS_NAME(X) \
  const char* Name() const override { return #X; }
#else
#define JXL_FIELDS_NAME(X)
#endif  // JXL_IS_DEBUG_BUILD

class Visitor;
class Fields {
 public:
  virtual ~Fields() = default;
#if (JXL_IS_DEBUG_BUILD)
  virtual const char* Name() const = 0;
#endif  // JXL_IS_DEBUG_BUILD
  virtual Status VisitFields(Visitor* JXL_RESTRICT visitor) = 0;
};

// Distribution of U32 values for one particular selector. Represents either a
// power of two-sized range, or a single value. A separate type ensures this is
// only passed to the U32Enc ctor.
struct U32Distr {
  // No need to validate - all `d` are legitimate.
  constexpr explicit U32Distr(uint32_t d) : d(d) {}

  static constexpr uint32_t kDirect = 0x80000000u;

  constexpr bool IsDirect() const { return (d & kDirect) != 0; }

  // Only call if IsDirect().
  constexpr uint32_t Direct() const { return d & (kDirect - 1); }

  // Only call if !IsDirect().
  constexpr size_t ExtraBits() const { return (d & 0x1F) + 1; }
  uint32_t Offset() const { return (d >> 5) & 0x3FFFFFF; }

  uint32_t d;
};

// A direct-coded 31-bit value occupying 2 bits in the bitstream.
constexpr U32Distr Val(uint32_t value) {
  return U32Distr(value | U32Distr::kDirect);
}

// Value - `offset` will be signaled in `bits` extra bits.
constexpr U32Distr BitsOffset(uint32_t bits, uint32_t offset) {
  return U32Distr(((bits - 1) & 0x1F) + ((offset & 0x3FFFFFF) << 5));
}

// Value will be signaled in `bits` extra bits.
constexpr U32Distr Bits(uint32_t bits) { return BitsOffset(bits, 0); }

// See U32Coder documentation in fields.h.
class U32Enc {
 public:
  constexpr U32Enc(const U32Distr d0, const U32Distr d1, const U32Distr d2,
                   const U32Distr d3)
      : d_{d0, d1, d2, d3} {}

  // Returns the U32Distr at `selector` = 0..3, least-significant first.
  U32Distr GetDistr(const uint32_t selector) const {
    JXL_DASSERT(selector < 4);
    return d_[selector];
  }

 private:
  U32Distr d_[4];
};

// Returns bit with the given `index` (0 = least significant).
template <typename T>
static inline constexpr uint64_t MakeBit(T index) {
  return 1ULL << static_cast<uint32_t>(index);
}

// Returns vector of all possible values of an Enum type. Relies on each Enum
// providing an overload of EnumBits() that returns a bit array of its values,
// which implies values must be in [0, 64).
template <typename Enum>
std::vector<Enum> Values() {
  uint64_t bits = EnumBits(Enum());

  std::vector<Enum> values;
  values.reserve(hwy::PopCount(bits));

  // For each 1-bit in bits: add its index as value
  while (bits != 0) {
    const int index = Num0BitsBelowLS1Bit_Nonzero(bits);
    values.push_back(static_cast<Enum>(index));
    bits &= bits - 1;  // clear least-significant bit
  }
  return values;
}

// Returns true if value is one of Values<Enum>().
template <class Enum>
Status EnumValid(const Enum value) {
  if (static_cast<uint32_t>(value) >= 64) {
    return JXL_FAILURE("Value %u too large for %s\n",
                       static_cast<uint32_t>(value), EnumName(Enum()));
  }
  const uint64_t bit = MakeBit(value);
  if ((EnumBits(Enum()) & bit) == 0) {
    return JXL_FAILURE("Invalid value %u for %s\n",
                       static_cast<uint32_t>(value), EnumName(Enum()));
  }
  return true;
}

}  // namespace jxl

#endif  // LIB_JXL_FIELD_ENCODINGS_H_

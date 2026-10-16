// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/base/status.h"
#ifndef FJXL_SELF_INCLUDE

#include <assert.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include "lib/jxl/enc_fast_lossless.h"

#if !FJXL_STANDALONE
#include "lib/jxl/encode_internal.h"
#endif  // FJXL_STANDALONE

#if defined(__x86_64__) || defined(_M_X64)
#define FJXL_ARCH_IS_X86_64 1
#else
#define FJXL_ARCH_IS_X86_64 0
#endif

#if defined(__i386__) || defined(_M_IX86) || FJXL_ARCH_IS_X86_64
#define FJXL_ARCH_IS_X86 1
#else
#define FJXL_ARCH_IS_X86 0
#endif

#if FJXL_ARCH_IS_X86
#if defined(_MSC_VER)
#include <intrin.h>
#else  // _MSC_VER
#include <cpuid.h>
#endif  // _MSC_VER
#endif  // FJXL_ARCH_IS_X86

// Enable NEON and AVX2/AVX512 if not asked to do otherwise and the compilers
// support it.
#if defined(__aarch64__) || defined(_M_ARM64)  // ARCH
#include <arm_neon.h>

#if !defined(FJXL_ENABLE_NEON)
#define FJXL_ENABLE_NEON 1
#endif  // !defined(FJXL_ENABLE_NEON)

#elif FJXL_ARCH_IS_X86_64 && !defined(_MSC_VER)  // ARCH
#include <immintrin.h>

// manually add _mm512_cvtsi512_si32 definition if missing
// (e.g. with Xcode on macOS Mojave)
// copied from gcc 11.1.0 include/avx512fintrin.h line 14367-14373
#if defined(__clang__) &&                                           \
    ((!defined(__apple_build_version__) && __clang_major__ < 10) || \
     (defined(__apple_build_version__) && __apple_build_version__ < 12000032))
inline int __attribute__((__gnu_inline__, __always_inline__, __artificial__))
_mm512_cvtsi512_si32(__m512i __A) {
  __v16si __B = (__v16si)__A;
  return __B[0];
}
#endif

#if !defined(FJXL_ENABLE_AVX2)
#define FJXL_ENABLE_AVX2 1
#endif  // !defined(FJXL_ENABLE_AVX2)

#if !defined(FJXL_ENABLE_AVX512)
// On clang-7 or earlier, and gcc-10 or earlier, AVX512 seems broken.
#if (defined(__clang__) &&                                             \
         (!defined(__apple_build_version__) && __clang_major__ > 7) || \
     (defined(__apple_build_version__) &&                              \
      __apple_build_version__ > 10010046)) ||                          \
    (defined(__GNUC__) && __GNUC__ > 10)
#define FJXL_ENABLE_AVX512 1
#endif
#endif  // !defined(FJXL_ENABLE_AVX512)

#endif  // ARCH

#ifndef FJXL_ENABLE_NEON
#define FJXL_ENABLE_NEON 0
#endif

#ifndef FJXL_ENABLE_AVX2
#define FJXL_ENABLE_AVX2 0
#endif

#ifndef FJXL_ENABLE_AVX512
#define FJXL_ENABLE_AVX512 0
#endif

namespace {

enum class CpuFeature : uint32_t {
  kAVX2 = 0,

  kAVX512F,
  kAVX512VL,
  kAVX512CD,
  kAVX512BW,

  kVBMI,
  kVBMI2
};

constexpr uint32_t CpuFeatureBit(CpuFeature feature) {
  return 1u << static_cast<uint32_t>(feature);
}

#if FJXL_ARCH_IS_X86
#if defined(_MSC_VER)
void Cpuid(const uint32_t level, const uint32_t count,
           std::array<uint32_t, 4>& abcd) {
  int regs[4];
  __cpuidex(regs, level, count);
  for (int i = 0; i < 4; ++i) {
    abcd[i] = regs[i];
  }
}
uint32_t ReadXCR0() { return static_cast<uint32_t>(_xgetbv(0)); }
#else   // _MSC_VER
void Cpuid(const uint32_t level, const uint32_t count,
           std::array<uint32_t, 4>& abcd) {
  uint32_t a;
  uint32_t b;
  uint32_t c;
  uint32_t d;
  __cpuid_count(level, count, a, b, c, d);
  abcd[0] = a;
  abcd[1] = b;
  abcd[2] = c;
  abcd[3] = d;
}
uint32_t ReadXCR0() {
  uint32_t xcr0;
  uint32_t xcr0_high;
  const uint32_t index = 0;
  asm volatile(".byte 0x0F, 0x01, 0xD0"
               : "=a"(xcr0), "=d"(xcr0_high)
               : "c"(index));
  return xcr0;
}
#endif  // _MSC_VER

uint32_t DetectCpuFeatures() {
  uint32_t flags = 0;  // return value
  std::array<uint32_t, 4> abcd;
  Cpuid(0, 0, abcd);
  const uint32_t max_level = abcd[0];

  const auto check_bit = [](uint32_t v, uint32_t idx) -> bool {
    return (v & (1U << idx)) != 0;
  };

  // Extended features
  if (max_level >= 7) {
    Cpuid(7, 0, abcd);
    flags |= check_bit(abcd[1], 5) ? CpuFeatureBit(CpuFeature::kAVX2) : 0;

    flags |= check_bit(abcd[1], 16) ? CpuFeatureBit(CpuFeature::kAVX512F) : 0;
    flags |= check_bit(abcd[1], 28) ? CpuFeatureBit(CpuFeature::kAVX512CD) : 0;
    flags |= check_bit(abcd[1], 30) ? CpuFeatureBit(CpuFeature::kAVX512BW) : 0;
    flags |= check_bit(abcd[1], 31) ? CpuFeatureBit(CpuFeature::kAVX512VL) : 0;

    flags |= check_bit(abcd[2], 1) ? CpuFeatureBit(CpuFeature::kVBMI) : 0;
    flags |= check_bit(abcd[2], 6) ? CpuFeatureBit(CpuFeature::kVBMI2) : 0;
  }

  Cpuid(1, 0, abcd);
  const bool os_has_xsave = check_bit(abcd[2], 27);
  if (os_has_xsave) {
    const uint32_t xcr0 = ReadXCR0();
    if (!check_bit(xcr0, 1) || !check_bit(xcr0, 2)) {
      flags = 0;
    } else if (!check_bit(xcr0, 5) || !check_bit(xcr0, 6) ||
               !check_bit(xcr0, 7)) {
      // No AVX-512; disable everything but AVX2 if present
      flags &= CpuFeatureBit(CpuFeature::kAVX2);
    }
  }

  return flags;
}
#else   // FJXL_ARCH_IS_X86
uint32_t DetectCpuFeatures() { return 0; }
#endif  // FJXL_ARCH_IS_X86

#if defined(_MSC_VER)
#define FJXL_UNUSED
#else
#define FJXL_UNUSED __attribute__((unused))
#endif

FJXL_UNUSED bool HasCpuFeature(CpuFeature feature) {
  static uint32_t cpu_features = DetectCpuFeatures();
  return (cpu_features & CpuFeatureBit(feature)) != 0;
}

#if defined(_MSC_VER) && !defined(__clang__)
#define FJXL_INLINE __forceinline
FJXL_INLINE uint32_t FloorLog2(uint32_t v) {
  unsigned long index;
  _BitScanReverse(&index, v);
  return index;
}
FJXL_INLINE uint32_t CtzNonZero(uint64_t v) {
  unsigned long index;
  _BitScanForward(&index, v);
  return index;
}
#else
#define FJXL_INLINE inline __attribute__((always_inline))
FJXL_INLINE uint32_t FloorLog2(uint32_t v) {
  return v ? 31 - __builtin_clz(v) : 0;
}
FJXL_UNUSED FJXL_INLINE uint32_t CtzNonZero(uint64_t v) {
  return __builtin_ctzll(v);
}
#endif

// Compiles to a memcpy on little-endian systems.
FJXL_INLINE void StoreLE64(uint8_t* tgt, uint64_t data) {
#if (!defined(__BYTE_ORDER__) || (__BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__))
  for (int i = 0; i < 8; i++) {
    tgt[i] = (data >> (i * 8)) & 0xFF;
  }
#else
  memcpy(tgt, &data, 8);
#endif
}

FJXL_INLINE size_t AddBits(uint32_t count, uint64_t bits, uint8_t* data_buf,
                           size_t& bits_in_buffer, uint64_t& bit_buffer) {
  bit_buffer |= bits << bits_in_buffer;
  bits_in_buffer += count;
  StoreLE64(data_buf, bit_buffer);
  size_t bytes_in_buffer = bits_in_buffer / 8;
  bits_in_buffer -= bytes_in_buffer * 8;
  bit_buffer >>= bytes_in_buffer * 8;
  return bytes_in_buffer;
}

struct BitWriter {
  bool Allocate(size_t maximum_bit_size) {
    assert(data == nullptr);
    // Leave some padding.
    data.reset(static_cast<uint8_t*>(malloc(maximum_bit_size / 8 + 64)));
    return data != nullptr;
  }

  void Write(uint32_t count, uint64_t bits) {
    bytes_written += AddBits(count, bits, data.get() + bytes_written,
                             bits_in_buffer, buffer);
  }

  void ZeroPadToByte() {
    if (bits_in_buffer != 0) {
      Write(8 - bits_in_buffer, 0);
    }
  }

  FJXL_INLINE void WriteMultiple(const uint64_t* nbits, const uint64_t* bits,
                                 size_t n) {
    // Necessary because Write() is only guaranteed to work with <=56 bits.
    // Trying to SIMD-fy this code results in lower speed (and definitely less
    // clarity).
    {
      for (size_t i = 0; i < n; i++) {
        this->buffer |= bits[i] << this->bits_in_buffer;
        memcpy(this->data.get() + this->bytes_written, &this->buffer, 8);
        uint64_t shift = 64 - this->bits_in_buffer;
        this->bits_in_buffer += nbits[i];
        // This `if` seems to be faster than using ternaries.
        if (this->bits_in_buffer >= 64) {
          uint64_t next_buffer = shift >= 64 ? 0 : bits[i] >> shift;
          this->buffer = next_buffer;
          this->bits_in_buffer -= 64;
          this->bytes_written += 8;
        }
      }
      memcpy(this->data.get() + this->bytes_written, &this->buffer, 8);
      size_t bytes_in_buffer = this->bits_in_buffer / 8;
      this->bits_in_buffer -= bytes_in_buffer * 8;
      this->buffer >>= bytes_in_buffer * 8;
      this->bytes_written += bytes_in_buffer;
    }
  }

  std::unique_ptr<uint8_t[], void (*)(void*)> data = {nullptr, free};
  size_t bytes_written = 0;
  size_t bits_in_buffer = 0;
  uint64_t buffer = 0;
};

size_t SectionSize(const std::array<BitWriter, 4>& group_data) {
  size_t sz = 0;
  for (size_t j = 0; j < 4; j++) {
    const auto& writer = group_data[j];
    sz += writer.bytes_written * 8 + writer.bits_in_buffer;
  }
  sz = (sz + 7) / 8;
  return sz;
}

constexpr size_t kMaxFrameHeaderSize = 5;

constexpr size_t kGroupSizeOffset[4] = {
    static_cast<size_t>(0),
    static_cast<size_t>(1024),
    static_cast<size_t>(17408),
    static_cast<size_t>(4211712),
};
constexpr size_t kTOCBits[4] = {12, 16, 24, 32};

size_t TOCBucket(size_t group_size) {
  size_t bucket = 0;
  while (bucket < 3 && group_size >= kGroupSizeOffset[bucket + 1]) ++bucket;
  return bucket;
}

#if !FJXL_STANDALONE
size_t TOCSize(const std::vector<size_t>& group_sizes) {
  size_t toc_bits = 0;
  for (size_t group_size : group_sizes) {
    toc_bits += kTOCBits[TOCBucket(group_size)];
  }
  return (toc_bits + 7) / 8;
}

size_t FrameHeaderSize(bool have_alpha, bool is_last) {
  size_t nbits = 28 + (have_alpha ? 4 : 0) + (is_last ? 0 : 2);
  return (nbits + 7) / 8;
}
#endif

void ComputeAcGroupDataOffset(size_t dc_global_size, size_t num_dc_groups,
                              size_t num_ac_groups, size_t& min_dc_global_size,
                              size_t& ac_group_offset) {
  // Max AC group size is 768 kB, so max AC group TOC bits is 24.
  size_t ac_toc_max_bits = num_ac_groups * 24;
  size_t ac_toc_min_bits = num_ac_groups * 12;
  size_t max_padding = 1 + (ac_toc_max_bits - ac_toc_min_bits + 7) / 8;
  min_dc_global_size = dc_global_size;
  size_t dc_global_bucket = TOCBucket(min_dc_global_size);
  while (TOCBucket(min_dc_global_size + max_padding) > dc_global_bucket) {
    dc_global_bucket = TOCBucket(min_dc_global_size + max_padding);
    min_dc_global_size = kGroupSizeOffset[dc_global_bucket];
  }
  assert(TOCBucket(min_dc_global_size) == dc_global_bucket);
  assert(TOCBucket(min_dc_global_size + max_padding) == dc_global_bucket);
  size_t max_toc_bits =
      kTOCBits[dc_global_bucket] + 12 * (1 + num_dc_groups) + ac_toc_max_bits;
  size_t max_toc_size = (max_toc_bits + 7) / 8;
  ac_group_offset = kMaxFrameHeaderSize + max_toc_size + min_dc_global_size;
}

#if !FJXL_STANDALONE
size_t ComputeDcGlobalPadding(const std::vector<size_t>& group_sizes,
                              size_t ac_group_data_offset,
                              size_t min_dc_global_size, bool have_alpha,
                              bool is_last) {
  std::vector<size_t> new_group_sizes = group_sizes;
  new_group_sizes[0] = min_dc_global_size;
  size_t toc_size = TOCSize(new_group_sizes);
  size_t actual_offset =
      FrameHeaderSize(have_alpha, is_last) + toc_size + group_sizes[0];
  return ac_group_data_offset - actual_offset;
}
#endif

constexpr size_t kNumRawSymbols = 19;
constexpr size_t kNumLZ77 = 33;
constexpr size_t kLZ77CacheSize = 32;

constexpr size_t kLZ77Offset = 224;
constexpr size_t kLZ77MinLength = 7;

void EncodeHybridUintLZ77(uint32_t value, uint32_t* token, uint32_t* nbits,
                          uint32_t* bits) {
  // 400 config
  uint32_t n = FloorLog2(value);
  *token = value < 16 ? value : 16 + n - 4;
  *nbits = value < 16 ? 0 : n;
  *bits = value < 16 ? 0 : value - (1 << *nbits);
}

struct PrefixCode {
  uint8_t raw_nbits[kNumRawSymbols] = {};
  uint8_t raw_bits[kNumRawSymbols] = {};

  uint8_t lz77_nbits[kNumLZ77] = {};
  uint16_t lz77_bits[kNumLZ77] = {};

  uint64_t lz77_cache_bits[kLZ77CacheSize] = {};
  uint8_t lz77_cache_nbits[kLZ77CacheSize] = {};

  size_t numraw;

  static uint16_t BitReverse(size_t nbits, uint16_t bits) {
    constexpr uint16_t kNibbleLookup[16] = {
        0b0000, 0b1000, 0b0100, 0b1100, 0b0010, 0b1010, 0b0110, 0b1110,
        0b0001, 0b1001, 0b0101, 0b1101, 0b0011, 0b1011, 0b0111, 0b1111,
    };
    uint16_t rev16 = (kNibbleLookup[bits & 0xF] << 12) |
                     (kNibbleLookup[(bits >> 4) & 0xF] << 8) |
                     (kNibbleLookup[(bits >> 8) & 0xF] << 4) |
                     (kNibbleLookup[bits >> 12]);
    return rev16 >> (16 - nbits);
  }

  // Create the prefix codes given the code lengths.
  // Supports the code lengths being split into two halves.
  static void ComputeCanonicalCode(const uint8_t* first_chunk_nbits,
                                   uint8_t* first_chunk_bits,
                                   size_t first_chunk_size,
                                   const uint8_t* second_chunk_nbits,
                                   uint16_t* second_chunk_bits,
                                   size_t second_chunk_size) {
    constexpr size_t kMaxCodeLength = 15;
    uint8_t code_length_counts[kMaxCodeLength + 1] = {};
    for (size_t i = 0; i < first_chunk_size; i++) {
      code_length_counts[first_chunk_nbits[i]]++;
      assert(first_chunk_nbits[i] <= kMaxCodeLength);
      assert(first_chunk_nbits[i] <= 8);
      assert(first_chunk_nbits[i] > 0);
    }
    for (size_t i = 0; i < second_chunk_size; i++) {
      code_length_counts[second_chunk_nbits[i]]++;
      assert(second_chunk_nbits[i] <= kMaxCodeLength);
    }

    uint16_t next_code[kMaxCodeLength + 1] = {};

    uint16_t code = 0;
    for (size_t i = 1; i < kMaxCodeLength + 1; i++) {
      code = (code + code_length_counts[i - 1]) << 1;
      next_code[i] = code;
    }

    for (size_t i = 0; i < first_chunk_size; i++) {
      first_chunk_bits[i] =
          BitReverse(first_chunk_nbits[i], next_code[first_chunk_nbits[i]]++);
    }
    for (size_t i = 0; i < second_chunk_size; i++) {
      second_chunk_bits[i] =
          BitReverse(second_chunk_nbits[i], next_code[second_chunk_nbits[i]]++);
    }
  }

  template <typename T>
  static void ComputeCodeLengthsNonZeroImpl(const uint64_t* freqs, size_t n,
                                            size_t precision, T infty,
                                            const uint8_t* min_limit,
                                            const uint8_t* max_limit,
                                            uint8_t* nbits) {
    assert(precision < 15);
    assert(n <= kMaxNumSymbols);
    std::vector<T> dynp(((1U << precision) + 1) * (n + 1), infty);
    auto d = [&](size_t sym, size_t off) -> T& {
      return dynp[sym * ((1 << precision) + 1) + off];
    };
    d(0, 0) = 0;
    for (size_t sym = 0; sym < n; sym++) {
      for (T bits = min_limit[sym]; bits <= max_limit[sym]; bits++) {
        size_t off_delta = 1U << (precision - bits);
        for (size_t off = 0; off + off_delta <= (1U << precision); off++) {
          d(sym + 1, off + off_delta) =
              std::min(d(sym, off) + static_cast<T>(freqs[sym]) * bits,
                       d(sym + 1, off + off_delta));
        }
      }
    }

    size_t sym = n;
    size_t off = 1U << precision;

    assert(d(sym, off) != infty);

    while (sym-- > 0) {
      assert(off > 0);
      for (size_t bits = min_limit[sym]; bits <= max_limit[sym]; bits++) {
        size_t off_delta = 1U << (precision - bits);
        if (off_delta <= off &&
            d(sym + 1, off) == d(sym, off - off_delta) + freqs[sym] * bits) {
          off -= off_delta;
          nbits[sym] = bits;
          break;
        }
      }
    }
  }

  // Computes nbits[i] for i <= n, subject to min_limit[i] <= nbits[i] <=
  // max_limit[i] and sum 2**-nbits[i] == 1, so to minimize sum(nbits[i] *
  // freqs[i]).
  static void ComputeCodeLengthsNonZero(const uint64_t* freqs, size_t n,
                                        uint8_t* min_limit, uint8_t* max_limit,
                                        uint8_t* nbits) {
    size_t precision = 0;
    size_t shortest_length = 255;
    uint64_t freqsum = 0;
    for (size_t i = 0; i < n; i++) {
      assert(freqs[i] != 0);
      freqsum += freqs[i];
      if (min_limit[i] < 1) min_limit[i] = 1;
      assert(min_limit[i] <= max_limit[i]);
      precision = std::max<size_t>(max_limit[i], precision);
      shortest_length = std::min<size_t>(min_limit[i], shortest_length);
    }
    // If all the minimum limits are greater than 1, shift precision so that we
    // behave as if the shortest was 1.
    precision -= shortest_length - 1;
    uint64_t infty = freqsum * precision;
    if (infty < std::numeric_limits<uint32_t>::max() / 2) {
      ComputeCodeLengthsNonZeroImpl(freqs, n, precision,
                                    static_cast<uint32_t>(infty), min_limit,
                                    max_limit, nbits);
    } else {
      ComputeCodeLengthsNonZeroImpl(freqs, n, precision, infty, min_limit,
                                    max_limit, nbits);
    }
  }

  static constexpr size_t kMaxNumSymbols =
      kNumRawSymbols + 1 < kNumLZ77 ? kNumLZ77 : kNumRawSymbols + 1;
  static void ComputeCodeLengths(const uint64_t* freqs, size_t n,
                                 const uint8_t* min_limit_in,
                                 const uint8_t* max_limit_in, uint8_t* nbits) {
    assert(n <= kMaxNumSymbols);
    uint64_t compact_freqs[kMaxNumSymbols];
    uint8_t min_limit[kMaxNumSymbols];
    uint8_t max_limit[kMaxNumSymbols];
    size_t ni = 0;
    for (size_t i = 0; i < n; i++) {
      if (freqs[i]) {
        compact_freqs[ni] = freqs[i];
        min_limit[ni] = min_limit_in[i];
        max_limit[ni] = max_limit_in[i];
        ni++;
      }
    }
    for (size_t i = ni; i < kMaxNumSymbols; ++i) {
      compact_freqs[i] = 0;
      min_limit[i] = 0;
      max_limit[i] = 0;
    }
    uint8_t num_bits[kMaxNumSymbols] = {};
    ComputeCodeLengthsNonZero(compact_freqs, ni, min_limit, max_limit,
                              num_bits);
    ni = 0;
    for (size_t i = 0; i < n; i++) {
      nbits[i] = 0;
      if (freqs[i]) {
        nbits[i] = num_bits[ni++];
      }
    }
  }

  // Invalid code, used to construct arrays.
  PrefixCode() = default;

  template <typename BitDepth>
  PrefixCode(BitDepth /* bitdepth */, uint64_t* raw_counts,
             uint64_t* lz77_counts) {
    // "merge" together all the lz77 counts in a single symbol for the level 1
    // table (containing just the raw symbols, up to length 7).
    uint64_t level1_counts[kNumRawSymbols + 1];
    memcpy(level1_counts, raw_counts, kNumRawSymbols * sizeof(uint64_t));
    numraw = kNumRawSymbols;
    while (numraw > 0 && level1_counts[numraw - 1] == 0) numraw--;

    level1_counts[numraw] = 0;
    for (size_t i = 0; i < kNumLZ77; i++) {
      level1_counts[numraw] += lz77_counts[i];
    }
    uint8_t level1_nbits[kNumRawSymbols + 1] = {};
    ComputeCodeLengths(level1_counts, numraw + 1, BitDepth::kMinRawLength,
                       BitDepth::kMaxRawLength, level1_nbits);

    uint8_t level2_nbits[kNumLZ77] = {};
    uint8_t min_lengths[kNumLZ77] = {};
    uint8_t l = 15 - level1_nbits[numraw];
    uint8_t max_lengths[kNumLZ77];
    for (uint8_t& max_length : max_lengths) {
      max_length = l;
    }
    size_t num_lz77 = kNumLZ77;
    while (num_lz77 > 0 && lz77_counts[num_lz77 - 1] == 0) num_lz77--;
    ComputeCodeLengths(lz77_counts, num_lz77, min_lengths, max_lengths,
                       level2_nbits);
    for (size_t i = 0; i < numraw; i++) {
      raw_nbits[i] = level1_nbits[i];
    }
    for (size_t i = 0; i < num_lz77; i++) {
      lz77_nbits[i] =
          level2_nbits[i] ? level1_nbits[numraw] + level2_nbits[i] : 0;
    }

    ComputeCanonicalCode(raw_nbits, raw_bits, numraw, lz77_nbits, lz77_bits,
                         kNumLZ77);

    // Prepare lz77 cache
    for (size_t count = 0; count < kLZ77CacheSize; count++) {
      unsigned token, nbits, bits;
      EncodeHybridUintLZ77(count, &token, &nbits, &bits);
      lz77_cache_nbits[count] = lz77_nbits[token] + nbits + raw_nbits[0];
      lz77_cache_bits[count] =
          (((bits << lz77_nbits[token]) | lz77_bits[token]) << raw_nbits[0]) |
          raw_bits[0];
    }
  }

  // Max bits written: 2 + 72 + 95 + 24 + 165 = 286
  void WriteTo(BitWriter* writer) const {
    uint64_t code_length_counts[18] = {};
    code_length_counts[17] = 3 + 2 * (kNumLZ77 - 1);
    for (uint8_t raw_nbit : raw_nbits) {
      code_length_counts[raw_nbit]++;
    }
    for (uint8_t lz77_nbit : lz77_nbits) {
      code_length_counts[lz77_nbit]++;
    }
    uint8_t code_length_nbits[18] = {};
    uint8_t code_length_nbits_min[18] = {};
    uint8_t code_length_nbits_max[18] = {
        5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    };
    ComputeCodeLengths(code_length_counts, 18, code_length_nbits_min,
                       code_length_nbits_max, code_length_nbits);
    writer->Write(2, 0b00);  // HSKIP = 0, i.e. don't skip code lengths.

    // As per Brotli RFC.
    uint8_t code_length_order[18] = {1, 2, 3, 4,  0,  5,  17, 6,  16,
                                     7, 8, 9, 10, 11, 12, 13, 14, 15};
    uint8_t code_length_length_nbits[] = {2, 4, 3, 2, 2, 4};
    uint8_t code_length_length_bits[] = {0, 7, 3, 2, 1, 15};

    // Encode lengths of code lengths.
    size_t num_code_lengths = 18;
    while (code_length_nbits[code_length_order[num_code_lengths - 1]] == 0) {
      num_code_lengths--;
    }
    // Max bits written in this loop: 18 * 4 = 72
    for (size_t i = 0; i < num_code_lengths; i++) {
      int symbol = code_length_nbits[code_length_order[i]];
      writer->Write(code_length_length_nbits[symbol],
                    code_length_length_bits[symbol]);
    }

    // Compute the canonical codes for the codes that represent the lengths of
    // the actual codes for data.
    uint16_t code_length_bits[18] = {};
    ComputeCanonicalCode(nullptr, nullptr, 0, code_length_nbits,
                         code_length_bits, 18);
    // Encode raw bit code lengths.
    // Max bits written in this loop: 19 * 5 = 95
    for (uint8_t raw_nbit : raw_nbits) {
      writer->Write(code_length_nbits[raw_nbit], code_length_bits[raw_nbit]);
    }
    size_t num_lz77 = kNumLZ77;
    while (lz77_nbits[num_lz77 - 1] == 0) {
      num_lz77--;
    }
    // Encode 0s until 224 (start of LZ77 symbols). This is in total 224-19 =
    // 205.
    static_assert(kLZ77Offset == 224, "kLZ77Offset should be 224");
    static_assert(kNumRawSymbols == 19, "kNumRawSymbols should be 19");
    {
      // Max bits in this block: 24
      writer->Write(code_length_nbits[17], code_length_bits[17]);
      writer->Write(3, 0b010);  // 5
      writer->Write(code_length_nbits[17], code_length_bits[17]);
      writer->Write(3, 0b000);  // (5-2)*8 + 3 = 27
      writer->Write(code_length_nbits[17], code_length_bits[17]);
      writer->Write(3, 0b010);  // (27-2)*8 + 5 = 205
    }
    // Encode LZ77 symbols, with values 224+i.
    // Max bits written in this loop: 33 * 5 = 165
    for (size_t i = 0; i < num_lz77; i++) {
      writer->Write(code_length_nbits[lz77_nbits[i]],
                    code_length_bits[lz77_nbits[i]]);
    }
  }
};

}  // namespace

extern "C" {

struct JxlFastLosslessFrameState {
  JxlChunkedFrameInputSource input;
  size_t width;
  size_t height;
  size_t num_groups_x;
  size_t num_groups_y;
  size_t num_dc_groups_x;
  size_t num_dc_groups_y;
  size_t nb_chans;
  size_t bitdepth;
  int big_endian;
  int effort;
  bool collided;
  PrefixCode hcode[4];
  std::vector<int16_t> lookup;
  BitWriter header;
  std::vector<std::array<BitWriter, 4>> group_data;
  std::vector<size_t> group_sizes;
  size_t ac_group_data_offset = 0;
  size_t min_dc_global_size = 0;
  size_t current_bit_writer = 0;
  size_t bit_writer_byte_pos = 0;
  size_t bits_in_buffer = 0;
  uint64_t bit_buffer = 0;
  bool process_done = false;
};

size_t JxlFastLosslessOutputSize(const JxlFastLosslessFrameState* frame) {
  size_t total_size_groups = 0;
  for (const auto& section : frame->group_data) {
    total_size_groups += SectionSize(section);
  }
  return frame->header.bytes_written + total_size_groups;
}

size_t JxlFastLosslessMaxRequiredOutput(
    const JxlFastLosslessFrameState* frame) {
  return JxlFastLosslessOutputSize(frame) + 32;
}

bool JxlFastLosslessPrepareHeader(JxlFastLosslessFrameState* frame,
                                  int add_image_header, int is_last) {
  BitWriter* output = &frame->header;
  if (!output->Allocate(1000 + frame->group_sizes.size() * 32)) return false;

  bool have_alpha = (frame->nb_chans == 2 || frame->nb_chans == 4);

#if FJXL_STANDALONE
  if (add_image_header) {
    // Signature
    output->Write(16, 0x0AFF);

    // Size header, hand-crafted.
    // Not small
    output->Write(1, 0);

    auto wsz = [output](size_t size) {
      if (size - 1 < (1 << 9)) {
        output->Write(2, 0b00);
        output->Write(9, size - 1);
      } else if (size - 1 < (1 << 13)) {
        output->Write(2, 0b01);
        output->Write(13, size - 1);
      } else if (size - 1 < (1 << 18)) {
        output->Write(2, 0b10);
        output->Write(18, size - 1);
      } else {
        output->Write(2, 0b11);
        output->Write(30, size - 1);
      }
    };

    wsz(frame->height);

    // No special ratio.
    output->Write(3, 0);

    wsz(frame->width);

    // Hand-crafted ImageMetadata.
    output->Write(1, 0);  // all_default
    output->Write(1, 0);  // extra_fields
    output->Write(1, 0);  // bit_depth.floating_point_sample
    if (frame->bitdepth == 8) {
      output->Write(2, 0b00);  // bit_depth.bits_per_sample = 8
    } else if (frame->bitdepth == 10) {
      output->Write(2, 0b01);  // bit_depth.bits_per_sample = 10
    } else if (frame->bitdepth == 12) {
      output->Write(2, 0b10);  // bit_depth.bits_per_sample = 12
    } else {
      output->Write(2, 0b11);  // 1 + u(6)
      output->Write(6, frame->bitdepth - 1);
    }
    if (frame->bitdepth <= 14) {
      output->Write(1, 1);  // 16-bit-buffer sufficient
    } else {
      output->Write(1, 0);  // 16-bit-buffer NOT sufficient
    }
    if (have_alpha) {
      output->Write(2, 0b01);  // One extra channel
      if (frame->bitdepth == 8) {
        output->Write(1, 1); // ... all_default (ie. 8-bit alpha)
      } else {
        output->Write(1, 0); // not d_alpha
        output->Write(2, 0); // type = kAlpha
        output->Write(1, 0); // not float
        if (frame->bitdepth == 10) {
          output->Write(2, 0b01); // bit_depth.bits_per_sample = 10
        } else if (frame->bitdepth == 12) {
          output->Write(2, 0b10); // bit_depth.bits_per_sample = 12
        } else {
          output->Write(2, 0b11); // 1 + u(6)
          output->Write(6, frame->bitdepth - 1);
        }
        output->Write(2, 0); // dim_shift = 0
        output->Write(2, 0); // name_len = 0
        output->Write(1, 0); // alpha_associated = 0
      }
    } else {
      output->Write(2, 0b00);  // No extra channel
    }
    output->Write(1, 0);  // Not XYB
    if (frame->nb_chans > 2) {
      output->Write(1, 1);  // color_encoding.all_default (sRGB)
    } else {
      output->Write(1, 0);     // color_encoding.all_default false
      output->Write(1, 0);     // color_encoding.want_icc false
      output->Write(2, 1);     // grayscale
      output->Write(2, 1);     // D65
      output->Write(1, 0);     // no gamma transfer function
      output->Write(2, 0b10);  // tf: 2 + u(4)
      output->Write(4, 11);    // tf of sRGB
      output->Write(2, 1);     // relative rendering intent
    }
    output->Write(2, 0b00);  // No extensions.

    output->Write(1, 1);  // all_default transform data

    // No ICC, no preview. Frame should start at byte boundary.
    output->ZeroPadToByte();
  }
#else
  assert(!add_image_header);
#endif
  // Handcrafted frame header.
  output->Write(1, 0);     // all_default
  output->Write(2, 0b00);  // regular frame
  output->Write(1, 1);     // modular
  output->Write(2, 0b00);  // default flags
  output->Write(1, 0);     // not YCbCr
  output->Write(2, 0b00);  // no upsampling
  if (have_alpha) {
    output->Write(2, 0b00);  // no alpha upsampling
  }
  output->Write(2, 0b01);  // default group size
  output->Write(2, 0b00);  // exactly one pass
  output->Write(1, 0);     // no custom size or origin
  output->Write(2, 0b00);  // kReplace blending mode
  if (have_alpha) {
    output->Write(2, 0b00);  // kReplace blending mode for alpha channel
  }
  output->Write(1, is_last);  // is_last
  if (!is_last) {
    output->Write(2, 0b00);  // can not be saved as reference
  }
  output->Write(2, 0b00);  // a frame has no name
  output->Write(1, 0);     // loop filter is not all_default
  output->Write(1, 0);     // no gaborish
  output->Write(2, 0);     // 0 EPF iters
  output->Write(2, 0b00);  // No LF extensions
  output->Write(2, 0b00);  // No FH extensions

  output->Write(1, 0);      // No TOC permutation
  output->ZeroPadToByte();  // TOC is byte-aligned.
  assert(add_image_header || output->bytes_written <= kMaxFrameHeaderSize);
  for (size_t group_size : frame->group_sizes) {
    size_t bucket = TOCBucket(group_size);
    output->Write(2, bucket);
    output->Write(kTOCBits[bucket] - 2, group_size - kGroupSizeOffset[bucket]);
  }
  output->ZeroPadToByte();  // Groups are byte-aligned.
  return true;
}

#if !FJXL_STANDALONE
bool JxlFastLosslessOutputAlignedSection(
    const BitWriter& bw, JxlEncoderOutputProcessorWrapper* output_processor) {
  assert(bw.bits_in_buffer == 0);
  const uint8_t* data = bw.data.get();
  size_t remaining_len = bw.bytes_written;
  while (remaining_len > 0) {
    JXL_ASSIGN_OR_RETURN(auto buffer,
                         output_processor->GetBuffer(1, remaining_len));
    size_t n = std::min(buffer.size(), remaining_len);
    if (n == 0) break;
    memcpy(buffer.data(), data, n);
    JXL_RETURN_IF_ERROR(buffer.advance(n));
    data += n;
    remaining_len -= n;
  };
  return true;
}

bool JxlFastLosslessOutputHeaders(
    JxlFastLosslessFrameState* frame_state,
    JxlEncoderOutputProcessorWrapper* output_processor) {
  JXL_RETURN_IF_ERROR(JxlFastLosslessOutputAlignedSection(frame_state->header,
                                                          output_processor));
  JXL_RETURN_IF_ERROR(JxlFastLosslessOutputAlignedSection(
      frame_state->group_data[0][0], output_processor));
  return true;
}
#endif

#if FJXL_ENABLE_AVX512
__attribute__((target("avx512vbmi2"))) static size_t AppendBytesWithBitOffset(
    const uint8_t* data, size_t n, size_t bit_buffer_nbits,
    unsigned char* output, uint64_t& bit_buffer) {
  if (n < 128) {
    return 0;
  }

  size_t i = 0;
  __m512i shift = _mm512_set1_epi64(64 - bit_buffer_nbits);
  __m512i carry = _mm512_set1_epi64(bit_buffer << (64 - bit_buffer_nbits));

  for (; i + 64 <= n; i += 64) {
    __m512i current = _mm512_loadu_si512(data + i);
    __m512i previous_u64 = _mm512_alignr_epi64(current, carry, 7);
    carry = current;
    __m512i out = _mm512_shrdv_epi64(previous_u64, current, shift);
    _mm512_storeu_si512(output + i, out);
  }

  bit_buffer = data[i - 1] >> (8 - bit_buffer_nbits);

  return i;
}
#endif

size_t JxlFastLosslessWriteOutput(JxlFastLosslessFrameState* frame,
                                  unsigned char* output, size_t output_size) {
  assert(output_size >= 32);
  unsigned char* initial_output = output;
  size_t (*append_bytes_with_bit_offset)(const uint8_t*, size_t, size_t,
                                         unsigned char*, uint64_t&) = nullptr;

#if FJXL_ENABLE_AVX512
  if (HasCpuFeature(CpuFeature::kVBMI2)) {
    append_bytes_with_bit_offset = AppendBytesWithBitOffset;
  }
#endif

  while (true) {
    size_t& cur = frame->current_bit_writer;
    size_t& bw_pos = frame->bit_writer_byte_pos;
    if (cur >= 1 + frame->group_data.size() * frame->nb_chans) {
      return output - initial_output;
    }
    if (output_size <= 9) {
      return output - initial_output;
    }
    size_t nbc = frame->nb_chans;
    const BitWriter& writer =
        cur == 0 ? frame->header
                 : frame->group_data[(cur - 1) / nbc][(cur - 1) % nbc];
    size_t full_byte_count =
        std::min(output_size - 9, writer.bytes_written - bw_pos);
    if (frame->bits_in_buffer == 0) {
      memcpy(output, writer.data.get() + bw_pos, full_byte_count);
    } else {
      size_t i = 0;
      if (append_bytes_with_bit_offset) {
        i += append_bytes_with_bit_offset(
            writer.data.get() + bw_pos, full_byte_count, frame->bits_in_buffer,
            output, frame->bit_buffer);
      }
#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
      // Copy 8 bytes at a time until we reach the border.
      for (; i + 8 < full_byte_count; i += 8) {
        uint64_t chunk;
        memcpy(&chunk, writer.data.get() + bw_pos + i, 8);
        uint64_t out = frame->bit_buffer | (chunk << frame->bits_in_buffer);
        memcpy(output + i, &out, 8);
        frame->bit_buffer = chunk >> (64 - frame->bits_in_buffer);
      }
#endif
      for (; i < full_byte_count; i++) {
        AddBits(8, writer.data.get()[bw_pos + i], output + i,
                frame->bits_in_buffer, frame->bit_buffer);
      }
    }
    output += full_byte_count;
    output_size -= full_byte_count;
    bw_pos += full_byte_count;
    if (bw_pos == writer.bytes_written) {
      auto write = [&](size_t num, uint64_t bits) {
        size_t n = AddBits(num, bits, output, frame->bits_in_buffer,
                           frame->bit_buffer);
        output += n;
        output_size -= n;
      };
      if (writer.bits_in_buffer) {
        write(writer.bits_in_buffer, writer.buffer);
      }
      bw_pos = 0;
      cur++;
      if ((cur - 1) % nbc == 0 && frame->bits_in_buffer != 0) {
        write(8 - frame->bits_in_buffer, 0);
      }
    }
  }
}

void JxlFastLosslessFreeFrameState(JxlFastLosslessFrameState* frame) {
  delete frame;
}

}  // extern "C"

#endif

#ifdef FJXL_SELF_INCLUDE

namespace {

template <typename T>
struct VecPair {
  T low;
  T hi;
};

#ifdef FJXL_GENERIC_SIMD
#undef FJXL_GENERIC_SIMD
#endif

#ifdef FJXL_AVX512
#define FJXL_GENERIC_SIMD
struct SIMDVec32;
struct Mask32 {
  __mmask16 mask;
  SIMDVec32 IfThenElse(const SIMDVec32& if_true, const SIMDVec32& if_false);
  size_t CountPrefix() const {
    return CtzNonZero(~uint64_t{_cvtmask16_u32(mask)});
  }
};

struct SIMDVec32 {
  __m512i vec;

  static constexpr size_t kLanes = 16;

  FJXL_INLINE static SIMDVec32 Load(const uint32_t* data) {
    return SIMDVec32{_mm512_loadu_si512((__m512i*)data)};
  }
  FJXL_INLINE void Store(uint32_t* data) {
    _mm512_storeu_si512((__m512i*)data, vec);
  }
  FJXL_INLINE static SIMDVec32 Val(uint32_t v) {
    return SIMDVec32{_mm512_set1_epi32(v)};
  }
  FJXL_INLINE SIMDVec32 ValToToken() const {
    return SIMDVec32{
        _mm512_sub_epi32(_mm512_set1_epi32(32), _mm512_lzcnt_epi32(vec))};
  }
  FJXL_INLINE SIMDVec32 SatSubU(const SIMDVec32& to_subtract) const {
    return SIMDVec32{_mm512_sub_epi32(_mm512_max_epu32(vec, to_subtract.vec),
                                      to_subtract.vec)};
  }
  FJXL_INLINE SIMDVec32 Sub(const SIMDVec32& to_subtract) const {
    return SIMDVec32{_mm512_sub_epi32(vec, to_subtract.vec)};
  }
  FJXL_INLINE SIMDVec32 Add(const SIMDVec32& oth) const {
    return SIMDVec32{_mm512_add_epi32(vec, oth.vec)};
  }
  FJXL_INLINE SIMDVec32 Xor(const SIMDVec32& oth) const {
    return SIMDVec32{_mm512_xor_epi32(vec, oth.vec)};
  }
  FJXL_INLINE Mask32 Eq(const SIMDVec32& oth) const {
    return Mask32{_mm512_cmpeq_epi32_mask(vec, oth.vec)};
  }
  FJXL_INLINE Mask32 Gt(const SIMDVec32& oth) const {
    return Mask32{_mm512_cmpgt_epi32_mask(vec, oth.vec)};
  }
  FJXL_INLINE SIMDVec32 Pow2() const {
    return SIMDVec32{_mm512_sllv_epi32(_mm512_set1_epi32(1), vec)};
  }
  template <size_t i>
  FJXL_INLINE SIMDVec32 SignedShiftRight() const {
    return SIMDVec32{_mm512_srai_epi32(vec, i)};
  }
};

struct SIMDVec16;

struct Mask16 {
  __mmask32 mask;
  SIMDVec16 IfThenElse(const SIMDVec16& if_true, const SIMDVec16& if_false);
  Mask16 And(const Mask16& oth) const {
    return Mask16{_kand_mask32(mask, oth.mask)};
  }
  size_t CountPrefix() const {
    return CtzNonZero(~uint64_t{_cvtmask32_u32(mask)});
  }
};

struct SIMDVec16 {
  __m512i vec;

  static constexpr size_t kLanes = 32;

  FJXL_INLINE static SIMDVec16 Load(const uint16_t* data) {
    return SIMDVec16{_mm512_loadu_si512((__m512i*)data)};
  }
  FJXL_INLINE void Store(uint16_t* data) {
    _mm512_storeu_si512((__m512i*)data, vec);
  }
  FJXL_INLINE static SIMDVec16 Val(uint16_t v) {
    return SIMDVec16{_mm512_set1_epi16(v)};
  }
  FJXL_INLINE static SIMDVec16 FromTwo32(const SIMDVec32& lo,
                                         const SIMDVec32& hi) {
    auto tmp = _mm512_packus_epi32(lo.vec, hi.vec);
    alignas(64) uint64_t perm[8] = {0, 2, 4, 6, 1, 3, 5, 7};
    return SIMDVec16{
        _mm512_permutex2var_epi64(tmp, _mm512_load_si512((__m512i*)perm), tmp)};
  }

  FJXL_INLINE SIMDVec16 ValToToken() const {
    auto c16 = _mm512_set1_epi32(16);
    auto c32 = _mm512_set1_epi32(32);
    auto low16bit = _mm512_set1_epi32(0x0000FFFF);
    auto lzhi =
        _mm512_sub_epi32(c16, _mm512_min_epu32(c16, _mm512_lzcnt_epi32(vec)));
    auto lzlo = _mm512_sub_epi32(
        c32, _mm512_lzcnt_epi32(_mm512_and_si512(low16bit, vec)));
    return SIMDVec16{_mm512_or_si512(lzlo, _mm512_slli_epi32(lzhi, 16))};
  }

  FJXL_INLINE SIMDVec16 SatSubU(const SIMDVec16& to_subtract) const {
    return SIMDVec16{_mm512_subs_epu16(vec, to_subtract.vec)};
  }
  FJXL_INLINE SIMDVec16 Sub(const SIMDVec16& to_subtract) const {
    return SIMDVec16{_mm512_sub_epi16(vec, to_subtract.vec)};
  }
  FJXL_INLINE SIMDVec16 Add(const SIMDVec16& oth) const {
    return SIMDVec16{_mm512_add_epi16(vec, oth.vec)};
  }
  FJXL_INLINE SIMDVec16 Min(const SIMDVec16& oth) const {
    return SIMDVec16{_mm512_min_epu16(vec, oth.vec)};
  }
  FJXL_INLINE Mask16 Eq(const SIMDVec16& oth) const {
    return Mask16{_mm512_cmpeq_epi16_mask(vec, oth.vec)};
  }
  FJXL_INLINE Mask16 Gt(const SIMDVec16& oth) const {
    return Mask16{_mm512_cmpgt_epi16_mask(vec, oth.vec)};
  }
  FJXL_INLINE SIMDVec16 Pow2() const {
    return SIMDVec16{_mm512_sllv_epi16(_mm512_set1_epi16(1), vec)};
  }
  FJXL_INLINE SIMDVec16 Or(const SIMDVec16& oth) const {
    return SIMDVec16{_mm512_or_si512(vec, oth.vec)};
  }
  FJXL_INLINE SIMDVec16 Xor(const SIMDVec16& oth) const {
    return SIMDVec16{_mm512_xor_si512(vec, oth.vec)};
  }
  FJXL_INLINE SIMDVec16 And(const SIMDVec16& oth) const {
    return SIMDVec16{_mm512_and_si512(vec, oth.vec)};
  }
  FJXL_INLINE SIMDVec16 HAdd(const SIMDVec16& oth) const {
    return SIMDVec16{_mm512_srai_epi16(_mm512_add_epi16(vec, oth.vec), 1)};
  }
  FJXL_INLINE SIMDVec16 PrepareForU8Lookup() const {
    return SIMDVec16{_mm512_or_si512(vec, _mm512_set1_epi16(0xFF00))};
  }
  FJXL_INLINE SIMDVec16 U8Lookup(const uint8_t* table) const {
    return SIMDVec16{_mm512_shuffle_epi8(
        _mm512_broadcast_i32x4(_mm_loadu_si128((__m128i*)table)), vec)};
  }
  FJXL_INLINE VecPair<SIMDVec16> Interleave(const SIMDVec16& low) const {
    auto lo = _mm512_unpacklo_epi16(low.vec, vec);
    auto hi = _mm512_unpackhi_epi16(low.vec, vec);
    alignas(64) uint64_t perm1[8] = {0, 1, 8, 9, 2, 3, 10, 11};
    alignas(64) uint64_t perm2[8] = {4, 5, 12, 13, 6, 7, 14, 15};
    return {SIMDVec16{_mm512_permutex2var_epi64(
                lo, _mm512_load_si512((__m512i*)perm1), hi)},
            SIMDVec16{_mm512_permutex2var_epi64(
                lo, _mm512_load_si512((__m512i*)perm2), hi)}};
  }
  FJXL_INLINE VecPair<SIMDVec32> Upcast() const {
    auto lo = _mm512_unpacklo_epi16(vec, _mm512_setzero_si512());
    auto hi = _mm512_unpackhi_epi16(vec, _mm512_setzero_si512());
    alignas(64) uint64_t perm1[8] = {0, 1, 8, 9, 2, 3, 10, 11};
    alignas(64) uint64_t perm2[8] = {4, 5, 12, 13, 6, 7, 14, 15};
    return {SIMDVec32{_mm512_permutex2var_epi64(
                lo, _mm512_load_si512((__m512i*)perm1), hi)},
            SIMDVec32{_mm512_permutex2var_epi64(
                lo, _mm512_load_si512((__m512i*)perm2), hi)}};
  }
  template <size_t i>
  FJXL_INLINE SIMDVec16 SignedShiftRight() const {
    return SIMDVec16{_mm512_srai_epi16(vec, i)};
  }

  static std::array<SIMDVec16, 1> LoadG8(const unsigned char* data) {
    __m256i bytes = _mm256_loadu_si256((__m256i*)data);
    return {SIMDVec16{_mm512_cvtepu8_epi16(bytes)}};
  }
  static std::array<SIMDVec16, 1> LoadG16(const unsigned char* data) {
    return {Load((const uint16_t*)data)};
  }

  static std::array<SIMDVec16, 2> LoadGA8(const unsigned char* data) {
    __m512i bytes = _mm512_loadu_si512((__m512i*)data);
    __m512i gray = _mm512_and_si512(bytes, _mm512_set1_epi16(0xFF));
    __m512i alpha = _mm512_srli_epi16(bytes, 8);
    return {SIMDVec16{gray}, SIMDVec16{alpha}};
  }
  static std::array<SIMDVec16, 2> LoadGA16(const unsigned char* data) {
    __m512i bytes1 = _mm512_loadu_si512((__m512i*)data);
    __m512i bytes2 = _mm512_loadu_si512((__m512i*)(data + 64));
    __m512i g_mask = _mm512_set1_epi32(0xFFFF);
    __m512i permuteidx = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
    __m512i g = _mm512_permutexvar_epi64(
        permuteidx, _mm512_packus_epi32(_mm512_and_si512(bytes1, g_mask),
                                        _mm512_and_si512(bytes2, g_mask)));
    __m512i a = _mm512_permutexvar_epi64(
        permuteidx, _mm512_packus_epi32(_mm512_srli_epi32(bytes1, 16),
                                        _mm512_srli_epi32(bytes2, 16)));
    return {SIMDVec16{g}, SIMDVec16{a}};
  }

  static std::array<SIMDVec16, 3> LoadRGB8(const unsigned char* data) {
    __m512i bytes0 = _mm512_loadu_si512((__m512i*)data);
    __m512i bytes1 =
        _mm512_zextsi256_si512(_mm256_loadu_si256((__m256i*)(data + 64)));

    // 0x7A = element of upper half of second vector = 0 after lookup; still in
    // the upper half once we add 1 or 2.
    uint8_t z = 0x7A;
    __m512i ridx =
        _mm512_set_epi8(z, 93, z, 90, z, 87, z, 84, z, 81, z, 78, z, 75, z, 72,
                        z, 69, z, 66, z, 63, z, 60, z, 57, z, 54, z, 51, z, 48,
                        z, 45, z, 42, z, 39, z, 36, z, 33, z, 30, z, 27, z, 24,
                        z, 21, z, 18, z, 15, z, 12, z, 9, z, 6, z, 3, z, 0);
    __m512i gidx = _mm512_add_epi8(ridx, _mm512_set1_epi8(1));
    __m512i bidx = _mm512_add_epi8(gidx, _mm512_set1_epi8(1));
    __m512i r = _mm512_permutex2var_epi8(bytes0, ridx, bytes1);
    __m512i g = _mm512_permutex2var_epi8(bytes0, gidx, bytes1);
    __m512i b = _mm512_permutex2var_epi8(bytes0, bidx, bytes1);
    return {SIMDVec16{r}, SIMDVec16{g}, SIMDVec16{b}};
  }
  static std::array<SIMDVec16, 3> LoadRGB16(const unsigned char* data) {
    __m512i bytes0 = _mm512_loadu_si512((__m512i*)data);
    __m512i bytes1 = _mm512_loadu_si512((__m512i*)(data + 64));
    __m512i bytes2 = _mm512_loadu_si512((__m512i*)(data + 128));

    __m512i ridx_lo = _mm512_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 63, 60, 57,
                                       54, 51, 48, 45, 42, 39, 36, 33, 30, 27,
                                       24, 21, 18, 15, 12, 9, 6, 3, 0);
    // -1 is such that when adding 1 or 2, we get the correct index for
    // green/blue.
    __m512i ridx_hi =
        _mm512_set_epi16(29, 26, 23, 20, 17, 14, 11, 8, 5, 2, -1, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    __m512i gidx_lo = _mm512_add_epi16(ridx_lo, _mm512_set1_epi16(1));
    __m512i gidx_hi = _mm512_add_epi16(ridx_hi, _mm512_set1_epi16(1));
    __m512i bidx_lo = _mm512_add_epi16(gidx_lo, _mm512_set1_epi16(1));
    __m512i bidx_hi = _mm512_add_epi16(gidx_hi, _mm512_set1_epi16(1));

    __mmask32 rmask = _cvtu32_mask32(0b11111111110000000000000000000000);
    __mmask32 gbmask = _cvtu32_mask32(0b11111111111000000000000000000000);

    __m512i rlo = _mm512_permutex2var_epi16(bytes0, ridx_lo, bytes1);
    __m512i glo = _mm512_permutex2var_epi16(bytes0, gidx_lo, bytes1);
    __m512i blo = _mm512_permutex2var_epi16(bytes0, bidx_lo, bytes1);
    __m512i r = _mm512_mask_permutexvar_epi16(rlo, rmask, ridx_hi, bytes2);
    __m512i g = _mm512_mask_permutexvar_epi16(glo, gbmask, gidx_hi, bytes2);
    __m512i b = _mm512_mask_permutexvar_epi16(blo, gbmask, bidx_hi, bytes2);
    return {SIMDVec16{r}, SIMDVec16{g}, SIMDVec16{b}};
  }

  static std::array<SIMDVec16, 4> LoadRGBA8(const unsigned char* data) {
    __m512i bytes1 = _mm512_loadu_si512((__m512i*)data);
    __m512i bytes2 = _mm512_loadu_si512((__m512i*)(data + 64));
    __m512i rg_mask = _mm512_set1_epi32(0xFFFF);
    __m512i permuteidx = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
    __m512i rg = _mm512_permutexvar_epi64(
        permuteidx, _mm512_packus_epi32(_mm512_and_si512(bytes1, rg_mask),
                                        _mm512_and_si512(bytes2, rg_mask)));
    __m512i b_a = _mm512_permutexvar_epi64(
        permuteidx, _mm512_packus_epi32(_mm512_srli_epi32(bytes1, 16),
                                        _mm512_srli_epi32(bytes2, 16)));
    __m512i r = _mm512_and_si512(rg, _mm512_set1_epi16(0xFF));
    __m512i g = _mm512_srli_epi16(rg, 8);
    __m512i b = _mm512_and_si512(b_a, _mm512_set1_epi16(0xFF));
    __m512i a = _mm512_srli_epi16(b_a, 8);
    return {SIMDVec16{r}, SIMDVec16{g}, SIMDVec16{b}, SIMDVec16{a}};
  }
  static std::array<SIMDVec16, 4> LoadRGBA16(const unsigned char* data) {
    __m512i bytes0 = _mm512_loadu_si512((__m512i*)data);
    __m512i bytes1 = _mm512_loadu_si512((__m512i*)(data + 64));
    __m512i bytes2 = _mm512_loadu_si512((__m512i*)(data + 128));
    __m512i bytes3 = _mm512_loadu_si512((__m512i*)(data + 192));

    auto pack32 = [](__m512i a, __m512i b) {
      __m512i permuteidx = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
      return _mm512_permutexvar_epi64(permuteidx, _mm512_packus_epi32(a, b));
    };
    auto packlow32 = [&pack32](__m512i a, __m512i b) {
      __m512i mask = _mm512_set1_epi32(0xFFFF);
      return pack32(_mm512_and_si512(a, mask), _mm512_and_si512(b, mask));
    };
    auto packhi32 = [&pack32](__m512i a, __m512i b) {
      return pack32(_mm512_srli_epi32(a, 16), _mm512_srli_epi32(b, 16));
    };

    __m512i rb0 = packlow32(bytes0, bytes1);
    __m512i rb1 = packlow32(bytes2, bytes3);
    __m512i ga0 = packhi32(bytes0, bytes1);
    __m512i ga1 = packhi32(bytes2, bytes3);

    __m512i r = packlow32(rb0, rb1);
    __m512i g = packlow32(ga0, ga1);
    __m512i b = packhi32(rb0, rb1);
    __m512i a = packhi32(ga0, ga1);
    return {SIMDVec16{r}, SIMDVec16{g}, SIMDVec16{b}, SIMDVec16{a}};
  }

  void SwapEndian() {
    auto indices = _mm512_broadcast_i32x4(
        _mm_setr_epi8(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14));
    vec = _mm512_shuffle_epi8(vec, indices);
  }
};

SIMDVec16 Mask16::IfThenElse(const SIMDVec16& if_true,
                             const SIMDVec16& if_false) {
  return SIMDVec16{_mm512_mask_blend_epi16(mask, if_false.vec, if_true.vec)};
}

SIMDVec32 Mask32::IfThenElse(const SIMDVec32& if_true,
                             const SIMDVec32& if_false) {
  return SIMDVec32{_mm512_mask_blend_epi32(mask, if_false.vec, if_true.vec)};
}

struct Bits64 {
  static constexpr size_t kLanes = 8;

  __m512i nbits;
  __m512i bits;

  FJXL_INLINE void Store(uint64_t* nbits_out, uint64_t* bits_out) {
    _mm512_storeu_si512((__m512i*)nbits_out, nbits);
    _mm512_storeu_si512((__m512i*)bits_out, bits);
  }
};

struct Bits32 {
  __m512i nbits;
  __m512i bits;

  static Bits32 FromRaw(SIMDVec32 nbits, SIMDVec32 bits) {
    return Bits32{nbits.vec, bits.vec};
  }

  Bits64 Merge() const {
    auto nbits_hi32 = _mm512_srli_epi64(nbits, 32);
    auto nbits_lo32 = _mm512_and_si512(nbits, _mm512_set1_epi64(0xFFFFFFFF));
    auto bits_hi32 = _mm512_srli_epi64(bits, 32);
    auto bits_lo32 = _mm512_and_si512(bits, _mm512_set1_epi64(0xFFFFFFFF));

    auto nbits64 = _mm512_add_epi64(nbits_hi32, nbits_lo32);
    auto bits64 =
        _mm512_or_si512(_mm512_sllv_epi64(bits_hi32, nbits_lo32), bits_lo32);
    return Bits64{nbits64, bits64};
  }

  void Interleave(const Bits32& low) {
    bits = _mm512_or_si512(_mm512_sllv_epi32(bits, low.nbits), low.bits);
    nbits = _mm512_add_epi32(nbits, low.nbits);
  }

  void ClipTo(size_t n) {
    n = std::min<size_t>(n, 16);
    constexpr uint32_t kMask[32] = {
        ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u,
        ~0u, ~0u, ~0u, ~0u, ~0u, 0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    };
    __m512i mask = _mm512_loadu_si512((__m512i*)(kMask + 16 - n));
    nbits = _mm512_and_si512(mask, nbits);
    bits = _mm512_and_si512(mask, bits);
  }
  void Skip(size_t n) {
    n = std::min<size_t>(n, 16);
    constexpr uint32_t kMask[32] = {
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   ~0u, ~0u, ~0u, ~0u, ~0u, ~0u,
        ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u,
    };
    __m512i mask = _mm512_loadu_si512((__m512i*)(kMask + 16 - n));
    nbits = _mm512_and_si512(mask, nbits);
    bits = _mm512_and_si512(mask, bits);
  }
};

struct Bits16 {
  __m512i nbits;
  __m512i bits;

  static Bits16 FromRaw(SIMDVec16 nbits, SIMDVec16 bits) {
    return Bits16{nbits.vec, bits.vec};
  }

  Bits32 Merge() const {
    auto nbits_hi16 = _mm512_srli_epi32(nbits, 16);
    auto nbits_lo16 = _mm512_and_si512(nbits, _mm512_set1_epi32(0xFFFF));
    auto bits_hi16 = _mm512_srli_epi32(bits, 16);
    auto bits_lo16 = _mm512_and_si512(bits, _mm512_set1_epi32(0xFFFF));

    auto nbits32 = _mm512_add_epi32(nbits_hi16, nbits_lo16);
    auto bits32 =
        _mm512_or_si512(_mm512_sllv_epi32(bits_hi16, nbits_lo16), bits_lo16);
    return Bits32{nbits32, bits32};
  }

  void Interleave(const Bits16& low) {
    bits = _mm512_or_si512(_mm512_sllv_epi16(bits, low.nbits), low.bits);
    nbits = _mm512_add_epi16(nbits, low.nbits);
  }

  void ClipTo(size_t n) {
    n = std::min<size_t>(n, 32);
    constexpr uint16_t kMask[64] = {
        0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
        0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
        0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
        0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
        0,      0,      0,      0,      0,      0,      0,      0,
        0,      0,      0,      0,      0,      0,      0,      0,
        0,      0,      0,      0,      0,      0,      0,      0,
        0,      0,      0,      0,      0,      0,      0,      0,
    };
    __m512i mask = _mm512_loadu_si512((__m512i*)(kMask + 32 - n));
    nbits = _mm512_and_si512(mask, nbits);
    bits = _mm512_and_si512(mask, bits);
  }
  void Skip(size_t n) {
    n = std::min<size_t>(n, 32);
    constexpr uint16_t kMask[64] = {
        0,      0,      0,      0,      0,      0,      0,      0,
        0,      0,      0,      0,      0,      0,      0,      0,
        0,      0,      0,      0,      0,      0,      0,      0,
        0,      0,      0,      0,      0,      0,      0,      0,
        0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
        0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
        0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
        0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
    };
    __m512i mask = _mm512_loadu_si512((__m512i*)(kMask + 32 - n));
    nbits = _mm512_and_si512(mask, nbits);
    bits = _mm512_and_si512(mask, bits);
  }
};

#endif

#ifdef FJXL_AVX2
#define FJXL_GENERIC_SIMD

struct SIMDVec32;

struct Mask32 {
  __m256i mask;
  SIMDVec32 IfThenElse(const SIMDVec32& if_true, const SIMDVec32& if_false);
  size_t CountPrefix() const {
    return CtzNonZero(~static_cast<uint64_t>(
        static_cast<uint8_t>(_mm256_movemask_ps(_mm256_castsi256_ps(mask)))));
  }
};

struct SIMDVec32 {
  __m256i vec;

  static constexpr size_t kLanes = 8;

  FJXL_INLINE static SIMDVec32 Load(const uint32_t* data) {
    return SIMDVec32{_mm256_loadu_si256((__m256i*)data)};
  }
  FJXL_INLINE void Store(uint32_t* data) {
    _mm256_storeu_si256((__m256i*)data, vec);
  }
  FJXL_INLINE static SIMDVec32 Val(uint32_t v) {
    return SIMDVec32{_mm256_set1_epi32(v)};
  }
  FJXL_INLINE SIMDVec32 ValToToken() const {
    auto f32 = _mm256_castps_si256(_mm256_cvtepi32_ps(vec));
    return SIMDVec32{_mm256_max_epi32(
        _mm256_setzero_si256(),
        _mm256_sub_epi32(_mm256_srli_epi32(f32, 23), _mm256_set1_epi32(126)))};
  }
  FJXL_INLINE SIMDVec32 SatSubU(const SIMDVec32& to_subtract) const {
    return SIMDVec32{_mm256_sub_epi32(_mm256_max_epu32(vec, to_subtract.vec),
                                      to_subtract.vec)};
  }
  FJXL_INLINE SIMDVec32 Sub(const SIMDVec32& to_subtract) const {
    return SIMDVec32{_mm256_sub_epi32(vec, to_subtract.vec)};
  }
  FJXL_INLINE SIMDVec32 Add(const SIMDVec32& oth) const {
    return SIMDVec32{_mm256_add_epi32(vec, oth.vec)};
  }
  FJXL_INLINE SIMDVec32 Xor(const SIMDVec32& oth) const {
    return SIMDVec32{_mm256_xor_si256(vec, oth.vec)};
  }
  FJXL_INLINE SIMDVec32 Pow2() const {
    return SIMDVec32{_mm256_sllv_epi32(_mm256_set1_epi32(1), vec)};
  }
  FJXL_INLINE Mask32 Eq(const SIMDVec32& oth) const {
    return Mask32{_mm256_cmpeq_epi32(vec, oth.vec)};
  }
  FJXL_INLINE Mask32 Gt(const SIMDVec32& oth) const {
    return Mask32{_mm256_cmpgt_epi32(vec, oth.vec)};
  }
  template <size_t i>
  FJXL_INLINE SIMDVec32 SignedShiftRight() const {
    return SIMDVec32{_mm256_srai_epi32(vec, i)};
  }
};

struct SIMDVec16;

struct Mask16 {
  __m256i mask;
  SIMDVec16 IfThenElse(const SIMDVec16& if_true, const SIMDVec16& if_false);
  Mask16 And(const Mask16& oth) const {
    return Mask16{_mm256_and_si256(mask, oth.mask)};
  }
  size_t CountPrefix() const {
    return CtzNonZero(~static_cast<uint64_t>(
               static_cast<uint32_t>(_mm256_movemask_epi8(mask)))) /
           2;
  }
};

struct SIMDVec16 {
  __m256i vec;

  static constexpr size_t kLanes = 16;

  FJXL_INLINE static SIMDVec16 Load(const uint16_t* data) {
    return SIMDVec16{_mm256_loadu_si256((__m256i*)data)};
  }
  FJXL_INLINE void Store(uint16_t* data) {
    _mm256_storeu_si256((__m256i*)data, vec);
  }
  FJXL_INLINE static SIMDVec16 Val(uint16_t v) {
    return SIMDVec16{_mm256_set1_epi16(v)};
  }
  FJXL_INLINE static SIMDVec16 FromTwo32(const SIMDVec32& lo,
                                         const SIMDVec32& hi) {
    auto tmp = _mm256_packus_epi32(lo.vec, hi.vec);
    return SIMDVec16{_mm256_permute4x64_epi64(tmp, 0b11011000)};
  }

  FJXL_INLINE SIMDVec16 ValToToken() const {
    auto nibble0 =
        _mm256_or_si256(_mm256_and_si256(vec, _mm256_set1_epi16(0xF)),
                        _mm256_set1_epi16(0xFF00));
    auto nibble1 = _mm256_or_si256(
        _mm256_and_si256(_mm256_srli_epi16(vec, 4), _mm256_set1_epi16(0xF)),
        _mm256_set1_epi16(0xFF00));
    auto nibble2 = _mm256_or_si256(
        _mm256_and_si256(_mm256_srli_epi16(vec, 8), _mm256_set1_epi16(0xF)),
        _mm256_set1_epi16(0xFF00));
    auto nibble3 =
        _mm256_or_si256(_mm256_srli_epi16(vec, 12), _mm256_set1_epi16(0xFF00));

    auto lut0 = _mm256_broadcastsi128_si256(
        _mm_setr_epi8(0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4));
    auto lut1 = _mm256_broadcastsi128_si256(
        _mm_setr_epi8(0, 5, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8));
    auto lut2 = _mm256_broadcastsi128_si256(_mm_setr_epi8(
        0, 9, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12));
    auto lut3 = _mm256_broadcastsi128_si256(_mm_setr_epi8(
        0, 13, 14, 14, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16));

    auto token0 = _mm256_shuffle_epi8(lut0, nibble0);
    auto token1 = _mm256_shuffle_epi8(lut1, nibble1);
    auto token2 = _mm256_shuffle_epi8(lut2, nibble2);
    auto token3 = _mm256_shuffle_epi8(lut3, nibble3);

    auto token = _mm256_max_epi16(_mm256_max_epi16(token0, token1),
                                  _mm256_max_epi16(token2, token3));
    return SIMDVec16{token};
  }

  FJXL_INLINE SIMDVec16 SatSubU(const SIMDVec16& to_subtract) const {
    return SIMDVec16{_mm256_subs_epu16(vec, to_subtract.vec)};
  }
  FJXL_INLINE SIMDVec16 Sub(const SIMDVec16& to_subtract) const {
    return SIMDVec16{_mm256_sub_epi16(vec, to_subtract.vec)};
  }
  FJXL_INLINE SIMDVec16 Add(const SIMDVec16& oth) const {
    return SIMDVec16{_mm256_add_epi16(vec, oth.vec)};
  }
  FJXL_INLINE SIMDVec16 Min(const SIMDVec16& oth) const {
    return SIMDVec16{_mm256_min_epu16(vec, oth.vec)};
  }
  FJXL_INLINE Mask16 Eq(const SIMDVec16& oth) const {
    return Mask16{_mm256_cmpeq_epi16(vec, oth.vec)};
  }
  FJXL_INLINE Mask16 Gt(const SIMDVec16& oth) const {
    return Mask16{_mm256_cmpgt_epi16(vec, oth.vec)};
  }
  FJXL_INLINE SIMDVec16 Pow2() const {
    auto pow2_lo_lut = _mm256_broadcastsi128_si256(
        _mm_setr_epi8(1 << 0, 1 << 1, 1 << 2, 1 << 3, 1 << 4, 1 << 5, 1 << 6,
                      1u << 7, 0, 0, 0, 0, 0, 0, 0, 0));
    auto pow2_hi_lut = _mm256_broadcastsi128_si256(
        _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1 << 0, 1 << 1, 1 << 2, 1 << 3,
                      1 << 4, 1 << 5, 1 << 6, 1u << 7));

    auto masked = _mm256_or_si256(vec, _mm256_set1_epi16(0xFF00));

    auto pow2_lo = _mm256_shuffle_epi8(pow2_lo_lut, masked);
    auto pow2_hi = _mm256_shuffle_epi8(pow2_hi_lut, masked);

    auto pow2 = _mm256_or_si256(_mm256_slli_epi16(pow2_hi, 8), pow2_lo);
    return SIMDVec16{pow2};
  }
  FJXL_INLINE SIMDVec16 Or(const SIMDVec16& oth) const {
    return SIMDVec16{_mm256_or_si256(vec, oth.vec)};
  }
  FJXL_INLINE SIMDVec16 Xor(const SIMDVec16& oth) const {
    return SIMDVec16{_mm256_xor_si256(vec, oth.vec)};
  }
  FJXL_INLINE SIMDVec16 And(const SIMDVec16& oth) const {
    return SIMDVec16{_mm256_and_si256(vec, oth.vec)};
  }
  FJXL_INLINE SIMDVec16 HAdd(const SIMDVec16& oth) const {
    return SIMDVec16{_mm256_srai_epi16(_mm256_add_epi16(vec, oth.vec), 1)};
  }
  FJXL_INLINE SIMDVec16 PrepareForU8Lookup() const {
    return SIMDVec16{_mm256_or_si256(vec, _mm256_set1_epi16(0xFF00))};
  }
  FJXL_INLINE SIMDVec16 U8Lookup(const uint8_t* table) const {
    return SIMDVec16{_mm256_shuffle_epi8(
        _mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)table)), vec)};
  }
  FJXL_INLINE VecPair<SIMDVec16> Interleave(const SIMDVec16& low) const {
    auto v02 = _mm256_unpacklo_epi16(low.vec, vec);
    auto v13 = _mm256_unpackhi_epi16(low.vec, vec);
    return {SIMDVec16{_mm256_permute2x128_si256(v02, v13, 0x20)},
            SIMDVec16{_mm256_permute2x128_si256(v02, v13, 0x31)}};
  }
  FJXL_INLINE VecPair<SIMDVec32> Upcast() const {
    auto v02 = _mm256_unpacklo_epi16(vec, _mm256_setzero_si256());
    auto v13 = _mm256_unpackhi_epi16(vec, _mm256_setzero_si256());
    return {SIMDVec32{_mm256_permute2x128_si256(v02, v13, 0x20)},
            SIMDVec32{_mm256_permute2x128_si256(v02, v13, 0x31)}};
  }
  template <size_t i>
  FJXL_INLINE SIMDVec16 SignedShiftRight() const {
    return SIMDVec16{_mm256_srai_epi16(vec, i)};
  }

  static std::array<SIMDVec16, 1> LoadG8(const unsigned char* data) {
    __m128i bytes = _mm_loadu_si128((__m128i*)data);
    return {SIMDVec16{_mm256_cvtepu8_epi16(bytes)}};
  }
  static std::array<SIMDVec16, 1> LoadG16(const unsigned char* data) {
    return {Load((const uint16_t*)data)};
  }

  static std::array<SIMDVec16, 2> LoadGA8(const unsigned char* data) {
    __m256i bytes = _mm256_loadu_si256((__m256i*)data);
    __m256i gray = _mm256_and_si256(bytes, _mm256_set1_epi16(0xFF));
    __m256i alpha = _mm256_srli_epi16(bytes, 8);
    return {SIMDVec16{gray}, SIMDVec16{alpha}};
  }
  static std::array<SIMDVec16, 2> LoadGA16(const unsigned char* data) {
    __m256i bytes1 = _mm256_loadu_si256((__m256i*)data);
    __m256i bytes2 = _mm256_loadu_si256((__m256i*)(data + 32));
    __m256i g_mask = _mm256_set1_epi32(0xFFFF);
    __m256i g = _mm256_permute4x64_epi64(
        _mm256_packus_epi32(_mm256_and_si256(bytes1, g_mask),
                            _mm256_and_si256(bytes2, g_mask)),
        0b11011000);
    __m256i a = _mm256_permute4x64_epi64(
        _mm256_packus_epi32(_mm256_srli_epi32(bytes1, 16),
                            _mm256_srli_epi32(bytes2, 16)),
        0b11011000);
    return {SIMDVec16{g}, SIMDVec16{a}};
  }

  static std::array<SIMDVec16, 3> LoadRGB8(const unsigned char* data) {
    __m128i bytes0 = _mm_loadu_si128((__m128i*)data);
    __m128i bytes1 = _mm_loadu_si128((__m128i*)(data + 16));
    __m128i bytes2 = _mm_loadu_si128((__m128i*)(data + 32));

    __m128i idx =
        _mm_setr_epi8(0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14, 1, 4, 7, 10, 13);

    __m128i r6b5g5_0 = _mm_shuffle_epi8(bytes0, idx);
    __m128i g6r5b5_1 = _mm_shuffle_epi8(bytes1, idx);
    __m128i b6g5r5_2 = _mm_shuffle_epi8(bytes2, idx);

    __m128i mask010 = _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF,
                                    0xFF, 0, 0, 0, 0, 0);
    __m128i mask001 = _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xFF, 0xFF,
                                    0xFF, 0xFF, 0xFF);

    __m128i b2g2b1 = _mm_blendv_epi8(b6g5r5_2, g6r5b5_1, mask001);
    __m128i b2b0b1 = _mm_blendv_epi8(b2g2b1, r6b5g5_0, mask010);

    __m128i r0r1b1 = _mm_blendv_epi8(r6b5g5_0, g6r5b5_1, mask010);
    __m128i r0r1r2 = _mm_blendv_epi8(r0r1b1, b6g5r5_2, mask001);

    __m128i g1r1g0 = _mm_blendv_epi8(g6r5b5_1, r6b5g5_0, mask001);
    __m128i g1g2g0 = _mm_blendv_epi8(g1r1g0, b6g5r5_2, mask010);

    __m128i g0g1g2 = _mm_alignr_epi8(g1g2g0, g1g2g0, 11);
    __m128i b0b1b2 = _mm_alignr_epi8(b2b0b1, b2b0b1, 6);

    return {SIMDVec16{_mm256_cvtepu8_epi16(r0r1r2)},
            SIMDVec16{_mm256_cvtepu8_epi16(g0g1g2)},
            SIMDVec16{_mm256_cvtepu8_epi16(b0b1b2)}};
  }
  static std::array<SIMDVec16, 3> LoadRGB16(const unsigned char* data) {
    auto load_and_split_lohi = [](const unsigned char* data) {
      // LHLHLH...
      __m256i bytes = _mm256_loadu_si256((__m256i*)data);
      // L0L0L0...
      __m256i lo = _mm256_and_si256(bytes, _mm256_set1_epi16(0xFF));
      // H0H0H0...
      __m256i hi = _mm256_srli_epi16(bytes, 8);
      // LLLLLLLLHHHHHHHHLLLLLLLLHHHHHHHH
      __m256i packed = _mm256_packus_epi16(lo, hi);
      return _mm256_permute4x64_epi64(packed, 0b11011000);
    };
    __m256i bytes0 = load_and_split_lohi(data);
    __m256i bytes1 = load_and_split_lohi(data + 32);
    __m256i bytes2 = load_and_split_lohi(data + 64);

    __m256i idx = _mm256_broadcastsi128_si256(
        _mm_setr_epi8(0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14, 1, 4, 7, 10, 13));

    __m256i r6b5g5_0 = _mm256_shuffle_epi8(bytes0, idx);
    __m256i g6r5b5_1 = _mm256_shuffle_epi8(bytes1, idx);
    __m256i b6g5r5_2 = _mm256_shuffle_epi8(bytes2, idx);

    __m256i mask010 = _mm256_broadcastsi128_si256(_mm_setr_epi8(
        0, 0, 0, 0, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0, 0));
    __m256i mask001 = _mm256_broadcastsi128_si256(_mm_setr_epi8(
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF));

    __m256i b2g2b1 = _mm256_blendv_epi8(b6g5r5_2, g6r5b5_1, mask001);
    __m256i b2b0b1 = _mm256_blendv_epi8(b2g2b1, r6b5g5_0, mask010);

    __m256i r0r1b1 = _mm256_blendv_epi8(r6b5g5_0, g6r5b5_1, mask010);
    __m256i r0r1r2 = _mm256_blendv_epi8(r0r1b1, b6g5r5_2, mask001);

    __m256i g1r1g0 = _mm256_blendv_epi8(g6r5b5_1, r6b5g5_0, mask001);
    __m256i g1g2g0 = _mm256_blendv_epi8(g1r1g0, b6g5r5_2, mask010);

    __m256i g0g1g2 = _mm256_alignr_epi8(g1g2g0, g1g2g0, 11);
    __m256i b0b1b2 = _mm256_alignr_epi8(b2b0b1, b2b0b1, 6);

    // Now r0r1r2, g0g1g2, b0b1b2 have the low bytes of the RGB pixels in their
    // lower half, and the high bytes in their upper half.

    auto combine_low_hi = [](__m256i v) {
      __m128i low = _mm256_extracti128_si256(v, 0);
      __m128i hi = _mm256_extracti128_si256(v, 1);
      __m256i low16 = _mm256_cvtepu8_epi16(low);
      __m256i hi16 = _mm256_cvtepu8_epi16(hi);
      return _mm256_or_si256(_mm256_slli_epi16(hi16, 8), low16);
    };

    return {SIMDVec16{combine_low_hi(r0r1r2)},
            SIMDVec16{combine_low_hi(g0g1g2)},
            SIMDVec16{combine_low_hi(b0b1b2)}};
  }

  static std::array<SIMDVec16, 4> LoadRGBA8(const unsigned char* data) {
    __m256i bytes1 = _mm256_loadu_si256((__m256i*)data);
    __m256i bytes2 = _mm256_loadu_si256((__m256i*)(data + 32));
    __m256i rg_mask = _mm256_set1_epi32(0xFFFF);
    __m256i rg = _mm256_permute4x64_epi64(
        _mm256_packus_epi32(_mm256_and_si256(bytes1, rg_mask),
                            _mm256_and_si256(bytes2, rg_mask)),
        0b11011000);
    __m256i b_a = _mm256_permute4x64_epi64(
        _mm256_packus_epi32(_mm256_srli_epi32(bytes1, 16),
                            _mm256_srli_epi32(bytes2, 16)),
        0b11011000);
    __m256i r = _mm256_and_si256(rg, _mm256_set1_epi16(0xFF));
    __m256i g = _mm256_srli_epi16(rg, 8);
    __m256i b = _mm256_and_si256(b_a, _mm256_set1_epi16(0xFF));
    __m256i a = _mm256_srli_epi16(b_a, 8);
    return {SIMDVec16{r}, SIMDVec16{g}, SIMDVec16{b}, SIMDVec16{a}};
  }
  static std::array<SIMDVec16, 4> LoadRGBA16(const unsigned char* data) {
    __m256i bytes0 = _mm256_loadu_si256((__m256i*)data);
    __m256i bytes1 = _mm256_loadu_si256((__m256i*)(data + 32));
    __m256i bytes2 = _mm256_loadu_si256((__m256i*)(data + 64));
    __m256i bytes3 = _mm256_loadu_si256((__m256i*)(data + 96));

    auto pack32 = [](__m256i a, __m256i b) {
      return _mm256_permute4x64_epi64(_mm256_packus_epi32(a, b), 0b11011000);
    };
    auto packlow32 = [&pack32](__m256i a, __m256i b) {
      __m256i mask = _mm256_set1_epi32(0xFFFF);
      return pack32(_mm256_and_si256(a, mask), _mm256_and_si256(b, mask));
    };
    auto packhi32 = [&pack32](__m256i a, __m256i b) {
      return pack32(_mm256_srli_epi32(a, 16), _mm256_srli_epi32(b, 16));
    };

    __m256i rb0 = packlow32(bytes0, bytes1);
    __m256i rb1 = packlow32(bytes2, bytes3);
    __m256i ga0 = packhi32(bytes0, bytes1);
    __m256i ga1 = packhi32(bytes2, bytes3);

    __m256i r = packlow32(rb0, rb1);
    __m256i g = packlow32(ga0, ga1);
    __m256i b = packhi32(rb0, rb1);
    __m256i a = packhi32(ga0, ga1);
    return {SIMDVec16{r}, SIMDVec16{g}, SIMDVec16{b}, SIMDVec16{a}};
  }

  void SwapEndian() {
    auto indices = _mm256_broadcastsi128_si256(
        _mm_setr_epi8(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14));
    vec = _mm256_shuffle_epi8(vec, indices);
  }
};

SIMDVec16 Mask16::IfThenElse(const SIMDVec16& if_true,
                             const SIMDVec16& if_false) {
  return SIMDVec16{_mm256_blendv_epi8(if_false.vec, if_true.vec, mask)};
}

SIMDVec32 Mask32::IfThenElse(const SIMDVec32& if_true,
                             const SIMDVec32& if_false) {
  return SIMDVec32{_mm256_blendv_epi8(if_false.vec, if_true.vec, mask)};
}

struct Bits64 {
  static constexpr size_t kLanes = 4;

  __m256i nbits;
  __m256i bits;

  FJXL_INLINE void Store(uint64_t* nbits_out, uint64_t* bits_out) {
    _mm256_storeu_si256((__m256i*)nbits_out, nbits);
    _mm256_storeu_si256((__m256i*)bits_out, bits);
  }
};

struct Bits32 {
  __m256i nbits;
  __m256i bits;

  static Bits32 FromRaw(SIMDVec32 nbits, SIMDVec32 bits) {
    return Bits32{nbits.vec, bits.vec};
  }

  Bits64 Merge() const {
    auto nbits_hi32 = _mm256_srli_epi64(nbits, 32);
    auto nbits_lo32 = _mm256_and_si256(nbits, _mm256_set1_epi64x(0xFFFFFFFF));
    auto bits_hi32 = _mm256_srli_epi64(bits, 32);
    auto bits_lo32 = _mm256_and_si256(bits, _mm256_set1_epi64x(0xFFFFFFFF));

    auto nbits64 = _mm256_add_epi64(nbits_hi32, nbits_lo32);
    auto bits64 =
        _mm256_or_si256(_mm256_sllv_epi64(bits_hi32, nbits_lo32), bits_lo32);
    return Bits64{nbits64, bits64};
  }

  void Interleave(const Bits32& low) {
    bits = _mm256_or_si256(_mm256_sllv_epi32(bits, low.nbits), low.bits);
    nbits = _mm256_add_epi32(nbits, low.nbits);
  }

  void ClipTo(size_t n) {
    n = std::min<size_t>(n, 8);
    constexpr uint32_t kMask[16] = {
        ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, 0, 0, 0, 0, 0, 0, 0, 0,
    };
    __m256i mask = _mm256_loadu_si256((__m256i*)(kMask + 8 - n));
    nbits = _mm256_and_si256(mask, nbits);
    bits = _mm256_and_si256(mask, bits);
  }
  void Skip(size_t n) {
    n = std::min<size_t>(n, 8);
    constexpr uint32_t kMask[16] = {
        0, 0, 0, 0, 0, 0, 0, 0, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u,
    };
    __m256i mask = _mm256_loadu_si256((__m256i*)(kMask + 8 - n));
    nbits = _mm256_and_si256(mask, nbits);
    bits = _mm256_and_si256(mask, bits);
  }
};

struct Bits16 {
  __m256i nbits;
  __m256i bits;

  static Bits16 FromRaw(SIMDVec16 nbits, SIMDVec16 bits) {
    return Bits16{nbits.vec, bits.vec};
  }

  Bits32 Merge() const {
    auto nbits_hi16 = _mm256_srli_epi32(nbits, 16);
    auto nbits_lo16 = _mm256_and_si256(nbits, _mm256_set1_epi32(0xFFFF));
    auto bits_hi16 = _mm256_srli_epi32(bits, 16);
    auto bits_lo16 = _mm256_and_si256(bits, _mm256_set1_epi32(0xFFFF));

    auto nbits32 = _mm256_add_epi32(nbits_hi16, nbits_lo16);
    auto bits32 =
        _mm256_or_si256(_mm256_sllv_epi32(bits_hi16, nbits_lo16), bits_lo16);
    return Bits32{nbits32, bits32};
  }

  void Interleave(const Bits16& low) {
    auto pow2_lo_lut = _mm256_broadcastsi128_si256(
        _mm_setr_epi8(1 << 0, 1 << 1, 1 << 2, 1 << 3, 1 << 4, 1 << 5, 1 << 6,
                      1u << 7, 0, 0, 0, 0, 0, 0, 0, 0));
    auto low_nbits_masked =
        _mm256_or_si256(low.nbits, _mm256_set1_epi16(0xFF00));

    auto bits_shifted = _mm256_mullo_epi16(
        bits, _mm256_shuffle_epi8(pow2_lo_lut, low_nbits_masked));

    nbits = _mm256_add_epi16(nbits, low.nbits);
    bits = _mm256_or_si256(bits_shifted, low.bits);
  }

  void ClipTo(size_t n) {
    n = std::min<size_t>(n, 16);
    constexpr uint16_t kMask[32] = {
        0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
        0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
        0,      0,      0,      0,      0,      0,      0,      0,
        0,      0,      0,      0,      0,      0,      0,      0,
    };
    __m256i mask = _mm256_loadu_si256((__m256i*)(kMask + 16 - n));
    nbits = _mm256_and_si256(mask, nbits);
    bits = _mm256_and_si256(mask, bits);
  }

  void Skip(size_t n) {
    n = std::min<size_t>(n, 16);
    constexpr uint16_t kMask[32] = {
        0,      0,      0,      0,      0,      0,      0,      0,
        0,      0,      0,      0,      0,      0,      0,      0,
        0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
        0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
    };
    __m256i mask = _mm256_loadu_si256((__m256i*)(kMask + 16 - n));
    nbits = _mm256_and_si256(mask, nbits);
    bits = _mm256_and_si256(mask, bits);
  }
};

#endif

#ifdef FJXL_NEON
#define FJXL_GENERIC_SIMD

struct SIMDVec32;

struct Mask32 {
  uint32x4_t mask;
  SIMDVec32 IfThenElse(const SIMDVec32& if_true, const SIMDVec32& if_false);
  Mask32 And(const Mask32& oth) const {
    return Mask32{vandq_u32(mask, oth.mask)};
  }
  size_t CountPrefix() const {
    uint32_t val_unset[4] = {0, 1, 2, 3};
    uint32_t val_set[4] = {4, 4, 4, 4};
    uint32x4_t val = vbslq_u32(mask, vld1q_u32(val_set), vld1q_u32(val_unset));
    return vminvq_u32(val);
  }
};

struct SIMDVec32 {
  uint32x4_t vec;

  static constexpr size_t kLanes = 4;

  FJXL_INLINE static SIMDVec32 Load(const uint32_t* data) {
    return SIMDVec32{vld1q_u32(data)};
  }
  FJXL_INLINE void Store(uint32_t* data) { vst1q_u32(data, vec); }
  FJXL_INLINE static SIMDVec32 Val(uint32_t v) {
    return SIMDVec32{vdupq_n_u32(v)};
  }
  FJXL_INLINE SIMDVec32 ValToToken() const {
    return SIMDVec32{vsubq_u32(vdupq_n_u32(32), vclzq_u32(vec))};
  }
  FJXL_INLINE SIMDVec32 SatSubU(const SIMDVec32& to_subtract) const {
    return SIMDVec32{vqsubq_u32(vec, to_subtract.vec)};
  }
  FJXL_INLINE SIMDVec32 Sub(const SIMDVec32& to_subtract) const {
    return SIMDVec32{vsubq_u32(vec, to_subtract.vec)};
  }
  FJXL_INLINE SIMDVec32 Add(const SIMDVec32& oth) const {
    return SIMDVec32{vaddq_u32(vec, oth.vec)};
  }
  FJXL_INLINE SIMDVec32 Xor(const SIMDVec32& oth) const {
    return SIMDVec32{veorq_u32(vec, oth.vec)};
  }
  FJXL_INLINE SIMDVec32 Pow2() const {
    return SIMDVec32{vshlq_u32(vdupq_n_u32(1), vreinterpretq_s32_u32(vec))};
  }
  FJXL_INLINE Mask32 Eq(const SIMDVec32& oth) const {
    return Mask32{vceqq_u32(vec, oth.vec)};
  }
  FJXL_INLINE Mask32 Gt(const SIMDVec32& oth) const {
    return Mask32{
        vcgtq_s32(vreinterpretq_s32_u32(vec), vreinterpretq_s32_u32(oth.vec))};
  }
  template <size_t i>
  FJXL_INLINE SIMDVec32 SignedShiftRight() const {
    return SIMDVec32{
        vreinterpretq_u32_s32(vshrq_n_s32(vreinterpretq_s32_u32(vec), i))};
  }
};

struct SIMDVec16;

struct Mask16 {
  uint16x8_t mask;
  SIMDVec16 IfThenElse(const SIMDVec16& if_true, const SIMDVec16& if_false);
  Mask16 And(const Mask16& oth) const {
    return Mask16{vandq_u16(mask, oth.mask)};
  }
  size_t CountPrefix() const {
    uint16_t val_unset[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    uint16_t val_set[8] = {8, 8, 8, 8, 8, 8, 8, 8};
    uint16x8_t val = vbslq_u16(mask, vld1q_u16(val_set), vld1q_u16(val_unset));
    return vminvq_u16(val);
  }
};

struct SIMDVec16 {
  uint16x8_t vec;

  static constexpr size_t kLanes = 8;

  FJXL_INLINE static SIMDVec16 Load(const uint16_t* data) {
    return SIMDVec16{vld1q_u16(data)};
  }
  FJXL_INLINE void Store(uint16_t* data) { vst1q_u16(data, vec); }
  FJXL_INLINE static SIMDVec16 Val(uint16_t v) {
    return SIMDVec16{vdupq_n_u16(v)};
  }
  FJXL_INLINE static SIMDVec16 FromTwo32(const SIMDVec32& lo,
                                         const SIMDVec32& hi) {
    return SIMDVec16{vmovn_high_u32(vmovn_u32(lo.vec), hi.vec)};
  }

  FJXL_INLINE SIMDVec16 ValToToken() const {
    return SIMDVec16{vsubq_u16(vdupq_n_u16(16), vclzq_u16(vec))};
  }
  FJXL_INLINE SIMDVec16 SatSubU(const SIMDVec16& to_subtract) const {
    return SIMDVec16{vqsubq_u16(vec, to_subtract.vec)};
  }
  FJXL_INLINE SIMDVec16 Sub(const SIMDVec16& to_subtract) const {
    return SIMDVec16{vsubq_u16(vec, to_subtract.vec)};
  }
  FJXL_INLINE SIMDVec16 Add(const SIMDVec16& oth) const {
    return SIMDVec16{vaddq_u16(vec, oth.vec)};
  }
  FJXL_INLINE SIMDVec16 Min(const SIMDVec16& oth) const {
    return SIMDVec16{vminq_u16(vec, oth.vec)};
  }
  FJXL_INLINE Mask16 Eq(const SIMDVec16& oth) const {
    return Mask16{vceqq_u16(vec, oth.vec)};
  }
  FJXL_INLINE Mask16 Gt(const SIMDVec16& oth) const {
    return Mask16{
        vcgtq_s16(vreinterpretq_s16_u16(vec), vreinterpretq_s16_u16(oth.vec))};
  }
  FJXL_INLINE SIMDVec16 Pow2() const {
    return SIMDVec16{vshlq_u16(vdupq_n_u16(1), vreinterpretq_s16_u16(vec))};
  }
  FJXL_INLINE SIMDVec16 Or(const SIMDVec16& oth) const {
    return SIMDVec16{vorrq_u16(vec, oth.vec)};
  }
  FJXL_INLINE SIMDVec16 Xor(const SIMDVec16& oth) const {
    return SIMDVec16{veorq_u16(vec, oth.vec)};
  }
  FJXL_INLINE SIMDVec16 And(const SIMDVec16& oth) const {
    return SIMDVec16{vandq_u16(vec, oth.vec)};
  }
  FJXL_INLINE SIMDVec16 HAdd(const SIMDVec16& oth) const {
    return SIMDVec16{vhaddq_u16(vec, oth.vec)};
  }
  FJXL_INLINE SIMDVec16 PrepareForU8Lookup() const {
    return SIMDVec16{vorrq_u16(vec, vdupq_n_u16(0xFF00))};
  }
  FJXL_INLINE SIMDVec16 U8Lookup(const uint8_t* table) const {
    uint8x16_t tbl = vld1q_u8(table);
    uint8x16_t indices = vreinterpretq_u8_u16(vec);
    return SIMDVec16{vreinterpretq_u16_u8(vqtbl1q_u8(tbl, indices))};
  }
  FJXL_INLINE VecPair<SIMDVec16> Interleave(const SIMDVec16& low) const {
    return {SIMDVec16{vzip1q_u16(low.vec, vec)},
            SIMDVec16{vzip2q_u16(low.vec, vec)}};
  }
  FJXL_INLINE VecPair<SIMDVec32> Upcast() const {
    uint32x4_t lo = vmovl_u16(vget_low_u16(vec));
    uint32x4_t hi = vmovl_high_u16(vec);
    return {SIMDVec32{lo}, SIMDVec32{hi}};
  }
  template <size_t i>
  FJXL_INLINE SIMDVec16 SignedShiftRight() const {
    return SIMDVec16{
        vreinterpretq_u16_s16(vshrq_n_s16(vreinterpretq_s16_u16(vec), i))};
  }

  static std::array<SIMDVec16, 1> LoadG8(const unsigned char* data) {
    uint8x8_t v = vld1_u8(data);
    return {SIMDVec16{vmovl_u8(v)}};
  }
  static std::array<SIMDVec16, 1> LoadG16(const unsigned char* data) {
    return {Load((const uint16_t*)data)};
  }

  static std::array<SIMDVec16, 2> LoadGA8(const unsigned char* data) {
    uint8x8x2_t v = vld2_u8(data);
    return {SIMDVec16{vmovl_u8(v.val[0])}, SIMDVec16{vmovl_u8(v.val[1])}};
  }
  static std::array<SIMDVec16, 2> LoadGA16(const unsigned char* data) {
    uint16x8x2_t v = vld2q_u16((const uint16_t*)data);
    return {SIMDVec16{v.val[0]}, SIMDVec16{v.val[1]}};
  }

  static std::array<SIMDVec16, 3> LoadRGB8(const unsigned char* data) {
    uint8x8x3_t v = vld3_u8(data);
    return {SIMDVec16{vmovl_u8(v.val[0])}, SIMDVec16{vmovl_u8(v.val[1])},
            SIMDVec16{vmovl_u8(v.val[2])}};
  }
  static std::array<SIMDVec16, 3> LoadRGB16(const unsigned char* data) {
    uint16x8x3_t v = vld3q_u16((const uint16_t*)data);
    return {SIMDVec16{v.val[0]}, SIMDVec16{v.val[1]}, SIMDVec16{v.val[2]}};
  }

  static std::array<SIMDVec16, 4> LoadRGBA8(const unsigned char* data) {
    uint8x8x4_t v = vld4_u8(data);
    return {SIMDVec16{vmovl_u8(v.val[0])}, SIMDVec16{vmovl_u8(v.val[1])},
            SIMDVec16{vmovl_u8(v.val[2])}, SIMDVec16{vmovl_u8(v.val[3])}};
  }
  static std::array<SIMDVec16, 4> LoadRGBA16(const unsigned char* data) {
    uint16x8x4_t v = vld4q_u16((const uint16_t*)data);
    return {SIMDVec16{v.val[0]}, SIMDVec16{v.val[1]}, SIMDVec16{v.val[2]},
            SIMDVec16{v.val[3]}};
  }

  void SwapEndian() {
    vec = vreinterpretq_u16_u8(vrev16q_u8(vreinterpretq_u8_u16(vec)));
  }
};

SIMDVec16 Mask16::IfThenElse(const SIMDVec16& if_true,
                             const SIMDVec16& if_false) {
  return SIMDVec16{vbslq_u16(mask, if_true.vec, if_false.vec)};
}

SIMDVec32 Mask32::IfThenElse(const SIMDVec32& if_true,
                             const SIMDVec32& if_false) {
  return SIMDVec32{vbslq_u32(mask, if_true.vec, if_false.vec)};
}

struct Bits64 {
  static constexpr size_t kLanes = 2;

  uint64x2_t nbits;
  uint64x2_t bits;

  FJXL_INLINE void Store(uint64_t* nbits_out, uint64_t* bits_out) {
    vst1q_u64(nbits_out, nbits);
    vst1q_u64(bits_out, bits);
  }
};

struct Bits32 {
  uint32x4_t nbits;
  uint32x4_t bits;

  static Bits32 FromRaw(SIMDVec32 nbits, SIMDVec32 bits) {
    return Bits32{nbits.vec, bits.vec};
  }

  Bits64 Merge() const {
    // TODO(veluca): can probably be optimized.
    uint64x2_t nbits_lo32 =
        vandq_u64(vreinterpretq_u64_u32(nbits), vdupq_n_u64(0xFFFFFFFF));
    uint64x2_t bits_hi32 =
        vshlq_u64(vshrq_n_u64(vreinterpretq_u64_u32(bits), 32),
                  vreinterpretq_s64_u64(nbits_lo32));
    uint64x2_t bits_lo32 =
        vandq_u64(vreinterpretq_u64_u32(bits), vdupq_n_u64(0xFFFFFFFF));
    uint64x2_t nbits64 =
        vsraq_n_u64(nbits_lo32, vreinterpretq_u64_u32(nbits), 32);
    uint64x2_t bits64 = vorrq_u64(bits_hi32, bits_lo32);
    return Bits64{nbits64, bits64};
  }

  void Interleave(const Bits32& low) {
    bits =
        vorrq_u32(vshlq_u32(bits, vreinterpretq_s32_u32(low.nbits)), low.bits);
    nbits = vaddq_u32(nbits, low.nbits);
  }

  void ClipTo(size_t n) {
    n = std::min<size_t>(n, 4);
    constexpr uint32_t kMask[8] = {
        ~0u, ~0u, ~0u, ~0u, 0, 0, 0, 0,
    };
    uint32x4_t mask = vld1q_u32(kMask + 4 - n);
    nbits = vandq_u32(mask, nbits);
    bits = vandq_u32(mask, bits);
  }
  void Skip(size_t n) {
    n = std::min<size_t>(n, 4);
    constexpr uint32_t kMask[8] = {
        0, 0, 0, 0, ~0u, ~0u, ~0u, ~0u,
    };
    uint32x4_t mask = vld1q_u32(kMask + 4 - n);
    nbits = vandq_u32(mask, nbits);
    bits = vandq_u32(mask, bits);
  }
};

struct Bits16 {
  uint16x8_t nbits;
  uint16x8_t bits;

  static Bits16 FromRaw(SIMDVec16 nbits, SIMDVec16 bits) {
    return Bits16{nbits.vec, bits.vec};
  }

  Bits32 Merge() const {
    // TODO(veluca): can probably be optimized.
    uint32x4_t nbits_lo16 =
        vandq_u32(vreinterpretq_u32_u16(nbits), vdupq_n_u32(0xFFFF));
    uint32x4_t bits_hi16 =
        vshlq_u32(vshrq_n_u32(vreinterpretq_u32_u16(bits), 16),
                  vreinterpretq_s32_u32(nbits_lo16));
    uint32x4_t bits_lo16 =
        vandq_u32(vreinterpretq_u32_u16(bits), vdupq_n_u32(0xFFFF));
    uint32x4_t nbits32 =
        vsraq_n_u32(nbits_lo16, vreinterpretq_u32_u16(nbits), 16);
    uint32x4_t bits32 = vorrq_u32(bits_hi16, bits_lo16);
    return Bits32{nbits32, bits32};
  }

  void Interleave(const Bits16& low) {
    bits =
        vorrq_u16(vshlq_u16(bits, vreinterpretq_s16_u16(low.nbits)), low.bits);
    nbits = vaddq_u16(nbits, low.nbits);
  }

  void ClipTo(size_t n) {
    n = std::min<size_t>(n, 8);
    constexpr uint16_t kMask[16] = {
        0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
        0,      0,      0,      0,      0,      0,      0,      0,
    };
    uint16x8_t mask = vld1q_u16(kMask + 8 - n);
    nbits = vandq_u16(mask, nbits);
    bits = vandq_u16(mask, bits);
  }
  void Skip(size_t n) {
    n = std::min<size_t>(n, 8);
    constexpr uint16_t kMask[16] = {
        0,      0,      0,      0,      0,      0,      0,      0,
        0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
    };
    uint16x8_t mask = vld1q_u16(kMask + 8 - n);
    nbits = vandq_u16(mask, nbits);
    bits = vandq_u16(mask, bits);
  }
};

#endif

#ifdef FJXL_GENERIC_SIMD
constexpr size_t SIMDVec32::kLanes;
constexpr size_t SIMDVec16::kLanes;

//  Each of these functions will process SIMDVec16::kLanes worth of values.

FJXL_INLINE void TokenizeSIMD(const uint16_t* residuals, uint16_t* token_out,
                              uint16_t* nbits_out, uint16_t* bits_out) {
  SIMDVec16 res = SIMDVec16::Load(residuals);
  SIMDVec16 token = res.ValToToken();
  SIMDVec16 nbits = token.SatSubU(SIMDVec16::Val(1));
  SIMDVec16 bits = res.SatSubU(nbits.Pow2());
  token.Store(token_out);
  nbits.Store(nbits_out);
  bits.Store(bits_out);
}

FJXL_INLINE void TokenizeSIMD(const uint32_t* residuals, uint16_t* token_out,
                              uint32_t* nbits_out, uint32_t* bits_out) {
  static_assert(SIMDVec16::kLanes == 2 * SIMDVec32::kLanes,
                "There should be twice more 16-bit lanes than 32-bit lanes");
  SIMDVec32 res_lo = SIMDVec32::Load(residuals);
  SIMDVec32 res_hi = SIMDVec32::Load(residuals + SIMDVec32::kLanes);
  SIMDVec32 token_lo = res_lo.ValToToken();
  SIMDVec32 token_hi = res_hi.ValToToken();
  SIMDVec32 nbits_lo = token_lo.SatSubU(SIMDVec32::Val(1));
  SIMDVec32 nbits_hi = token_hi.SatSubU(SIMDVec32::Val(1));
  SIMDVec32 bits_lo = res_lo.SatSubU(nbits_lo.Pow2());
  SIMDVec32 bits_hi = res_hi.SatSubU(nbits_hi.Pow2());
  SIMDVec16 token = SIMDVec16::FromTwo32(token_lo, token_hi);
  token.Store(token_out);
  nbits_lo.Store(nbits_out);
  nbits_hi.Store(nbits_out + SIMDVec32::kLanes);
  bits_lo.Store(bits_out);
  bits_hi.Store(bits_out + SIMDVec32::kLanes);
}

FJXL_INLINE void HuffmanSIMDUpTo13(const uint16_t* tokens,
                                   const uint8_t* raw_nbits_simd,
                                   const uint8_t* raw_bits_simd,
                                   uint16_t* nbits_out, uint16_t* bits_out) {
  SIMDVec16 tok = SIMDVec16::Load(tokens).PrepareForU8Lookup();
  tok.U8Lookup(raw_nbits_simd).Store(nbits_out);
  tok.U8Lookup(raw_bits_simd).Store(bits_out);
}

FJXL_INLINE void HuffmanSIMD14(const uint16_t* tokens,
                               const uint8_t* raw_nbits_simd,
                               const uint8_t* raw_bits_simd,
                               uint16_t* nbits_out, uint16_t* bits_out) {
  SIMDVec16 token_cap = SIMDVec16::Val(15);
  SIMDVec16 tok = SIMDVec16::Load(tokens);
  SIMDVec16 tok_index = tok.Min(token_cap).PrepareForU8Lookup();
  SIMDVec16 huff_bits_pre = tok_index.U8Lookup(raw_bits_simd);
  // Set the highest bit when token == 16; the Huffman code is constructed in
  // such a way that the code for token 15 is the same as the code for 16,
  // except for the highest bit.
  Mask16 needs_high_bit = tok.Eq(SIMDVec16::Val(16));
  SIMDVec16 huff_bits = needs_high_bit.IfThenElse(
      huff_bits_pre.Or(SIMDVec16::Val(128)), huff_bits_pre);
  huff_bits.Store(bits_out);
  tok_index.U8Lookup(raw_nbits_simd).Store(nbits_out);
}

FJXL_INLINE void HuffmanSIMDAbove14(const uint16_t* tokens,
                                    const uint8_t* raw_nbits_simd,
                                    const uint8_t* raw_bits_simd,
                                    uint16_t* nbits_out, uint16_t* bits_out) {
  SIMDVec16 tok = SIMDVec16::Load(tokens);
  // We assume `tok` fits in a *signed* 16-bit integer.
  Mask16 above = tok.Gt(SIMDVec16::Val(12));
  // 13, 14 -> 13
  // 15, 16 -> 14
  // 17, 18 -> 15
  SIMDVec16 remap_tok = above.IfThenElse(tok.HAdd(SIMDVec16::Val(13)), tok);
  SIMDVec16 tok_index = remap_tok.PrepareForU8Lookup();
  SIMDVec16 huff_bits_pre = tok_index.U8Lookup(raw_bits_simd);
  // Set the highest bit when token == 14, 16, 18.
  Mask16 needs_high_bit = above.And(tok.Eq(tok.And(SIMDVec16::Val(0xFFFE))));
  SIMDVec16 huff_bits = needs_high_bit.IfThenElse(
      huff_bits_pre.Or(SIMDVec16::Val(128)), huff_bits_pre);
  huff_bits.Store(bits_out);
  tok_index.U8Lookup(raw_nbits_simd).Store(nbits_out);
}

FJXL_INLINE void StoreSIMDUpTo8(const uint16_t* nbits_tok,
                                const uint16_t* bits_tok,
                                const uint16_t* nbits_huff,
                                const uint16_t* bits_huff, size_t n,
                                size_t skip, Bits32* bits_out) {
  Bits16 bits =
      Bits16::FromRaw(SIMDVec16::Load(nbits_tok), SIMDVec16::Load(bits_tok));
  Bits16 huff_bits =
      Bits16::FromRaw(SIMDVec16::Load(nbits_huff), SIMDVec16::Load(bits_huff));
  bits.Interleave(huff_bits);
  bits.ClipTo(n);
  bits.Skip(skip);
  bits_out[0] = bits.Merge();
}

// Huffman and raw bits don't necessarily fit in a single u16 here.
FJXL_INLINE void StoreSIMDUpTo14(const uint16_t* nbits_tok,
                                 const uint16_t* bits_tok,
                                 const uint16_t* nbits_huff,
                                 const uint16_t* bits_huff, size_t n,
                                 size_t skip, Bits32* bits_out) {
  VecPair<SIMDVec16> bits =
      SIMDVec16::Load(bits_tok).Interleave(SIMDVec16::Load(bits_huff));
  VecPair<SIMDVec16> nbits =
      SIMDVec16::Load(nbits_tok).Interleave(SIMDVec16::Load(nbits_huff));
  Bits16 low = Bits16::FromRaw(nbits.low, bits.low);
  Bits16 hi = Bits16::FromRaw(nbits.hi, bits.hi);
  low.ClipTo(2 * n);
  low.Skip(2 * skip);
  hi.ClipTo(std::max(2 * n, SIMDVec16::kLanes) - SIMDVec16::kLanes);
  hi.Skip(std::max(2 * skip, SIMDVec16::kLanes) - SIMDVec16::kLanes);

  bits_out[0] = low.Merge();
  bits_out[1] = hi.Merge();
}

FJXL_INLINE void StoreSIMDAbove14(const uint32_t* nbits_tok,
                                  const uint32_t* bits_tok,
                                  const uint16_t* nbits_huff,
                                  const uint16_t* bits_huff, size_t n,
                                  size_t skip, Bits32* bits_out) {
  static_assert(SIMDVec16::kLanes == 2 * SIMDVec32::kLanes,
                "There should be twice more 16-bit lanes than 32-bit lanes");
  Bits32 bits_low =
      Bits32::FromRaw(SIMDVec32::Load(nbits_tok), SIMDVec32::Load(bits_tok));
  Bits32 bits_hi =
      Bits32::FromRaw(SIMDVec32::Load(nbits_tok + SIMDVec32::kLanes),
                      SIMDVec32::Load(bits_tok + SIMDVec32::kLanes));

  VecPair<SIMDVec32> huff_bits = SIMDVec16::Load(bits_huff).Upcast();
  VecPair<SIMDVec32> huff_nbits = SIMDVec16::Load(nbits_huff).Upcast();

  Bits32 huff_low = Bits32::FromRaw(huff_nbits.low, huff_bits.low);
  Bits32 huff_hi = Bits32::FromRaw(huff_nbits.hi, huff_bits.hi);

  bits_low.Interleave(huff_low);
  bits_low.ClipTo(n);
  bits_low.Skip(skip);
  bits_out[0] = bits_low;
  bits_hi.Interleave(huff_hi);
  bits_hi.ClipTo(std::max(n, SIMDVec32::kLanes) - SIMDVec32::kLanes);
  bits_hi.Skip(std::max(skip, SIMDVec32::kLanes) - SIMDVec32::kLanes);
  bits_out[1] = bits_hi;
}

#ifdef FJXL_AVX512
FJXL_INLINE void StoreToWriterAVX512(const Bits32& bits32, BitWriter& output) {
  __m512i bits = bits32.bits;
  __m512i nbits = bits32.nbits;

  // Insert the leftover bits from the bit buffer at the bottom of the vector
  // and extract the top of the vector.
  uint64_t trail_bits =
      _mm512_cvtsi512_si32(_mm512_alignr_epi32(bits, bits, 15));
  uint64_t trail_nbits =
      _mm512_cvtsi512_si32(_mm512_alignr_epi32(nbits, nbits, 15));
  __m512i lead_bits = _mm512_set1_epi32(output.buffer);
  __m512i lead_nbits = _mm512_set1_epi32(output.bits_in_buffer);
  bits = _mm512_alignr_epi32(bits, lead_bits, 15);
  nbits = _mm512_alignr_epi32(nbits, lead_nbits, 15);

  // Merge 32 -> 64 bits.
  Bits32 b{nbits, bits};
  Bits64 b64 = b.Merge();
  bits = b64.bits;
  nbits = b64.nbits;

  __m512i zero = _mm512_setzero_si512();

  auto sh1 = [zero](__m512i vec) { return _mm512_alignr_epi64(vec, zero, 7); };
  auto sh2 = [zero](__m512i vec) { return _mm512_alignr_epi64(vec, zero, 6); };
  auto sh4 = [zero](__m512i vec) { return _mm512_alignr_epi64(vec, zero, 4); };

  // Compute first-past-end-bit-position.
  __m512i end_intermediate0 = _mm512_add_epi64(nbits, sh1(nbits));
  __m512i end_intermediate1 =
      _mm512_add_epi64(end_intermediate0, sh2(end_intermediate0));
  __m512i end = _mm512_add_epi64(end_intermediate1, sh4(end_intermediate1));

  uint64_t simd_nbits = _mm512_cvtsi512_si32(_mm512_alignr_epi64(end, end, 7));

  // Compute begin-bit-position.
  __m512i begin = _mm512_sub_epi64(end, nbits);

  // Index of the last bit in the chunk, or the end bit if nbits==0.
  __m512i last = _mm512_mask_sub_epi64(
      end, _mm512_cmpneq_epi64_mask(nbits, zero), end, _mm512_set1_epi64(1));

  __m512i lane_offset_mask = _mm512_set1_epi64(63);

  // Starting position of the chunk that each lane will ultimately belong to.
  __m512i chunk_start = _mm512_andnot_si512(lane_offset_mask, last);

  // For all lanes that contain bits belonging to two different 64-bit chunks,
  // compute the number of bits that belong to the first chunk.
  // total # of bits fit in a u16, so we can satsub_u16 here.
  __m512i first_chunk_nbits = _mm512_subs_epu16(chunk_start, begin);

  // Move all the previous-chunk-bits to the previous lane.
  __m512i negnbits = _mm512_sub_epi64(_mm512_set1_epi64(64), first_chunk_nbits);
  __m512i first_chunk_bits =
      _mm512_srlv_epi64(_mm512_sllv_epi64(bits, negnbits), negnbits);
  __m512i first_chunk_bits_down =
      _mm512_alignr_epi32(zero, first_chunk_bits, 2);
  bits = _mm512_srlv_epi64(bits, first_chunk_nbits);
  nbits = _mm512_sub_epi64(nbits, first_chunk_nbits);
  bits = _mm512_or_si512(bits, _mm512_sllv_epi64(first_chunk_bits_down, nbits));
  begin = _mm512_add_epi64(begin, first_chunk_nbits);

  // We now know that every lane should give bits to only one chunk. We can
  // shift the bits and then horizontally-or-reduce them within the same chunk.
  __m512i offset = _mm512_and_si512(begin, lane_offset_mask);
  __m512i aligned_bits = _mm512_sllv_epi64(bits, offset);
  // h-or-reduce within same chunk
  __m512i red0 = _mm512_mask_or_epi64(
      aligned_bits, _mm512_cmpeq_epi64_mask(sh1(chunk_start), chunk_start),
      sh1(aligned_bits), aligned_bits);
  __m512i red1 = _mm512_mask_or_epi64(
      red0, _mm512_cmpeq_epi64_mask(sh2(chunk_start), chunk_start), sh2(red0),
      red0);
  __m512i reduced = _mm512_mask_or_epi64(
      red1, _mm512_cmpeq_epi64_mask(sh4(chunk_start), chunk_start), sh4(red1),
      red1);
  // Extract the highest lane that belongs to each chunk (the lane that ends up
  // with the OR-ed value of all the other lanes of that chunk).
  __m512i next_chunk_start =
      _mm512_alignr_epi32(_mm512_set1_epi64(~0), chunk_start, 2);
  __m512i result = _mm512_maskz_compress_epi64(
      _mm512_cmpneq_epi64_mask(chunk_start, next_chunk_start), reduced);

  _mm512_storeu_si512((__m512i*)(output.data.get() + output.bytes_written),
                      result);

  // Update the bit writer and add the last 32-bit lane.
  // Note that since trail_nbits was at most 32 to begin with, operating on
  // trail_bits does not risk overflowing.
  output.bytes_written += simd_nbits / 8;
  // Here we are implicitly relying on the fact that simd_nbits < 512 to know
  // that the byte of bitreader data we access is initialized. This is
  // guaranteed because the remaining bits in the bitreader buffer are at most
  // 7, so simd_nbits <= 505 always.
  trail_bits = (trail_bits << (simd_nbits % 8)) +
               output.data.get()[output.bytes_written];
  trail_nbits += simd_nbits % 8;
  StoreLE64(output.data.get() + output.bytes_written, trail_bits);
  size_t trail_bytes = trail_nbits / 8;
  output.bits_in_buffer = trail_nbits % 8;
  output.buffer = trail_bits >> (trail_bytes * 8);
  output.bytes_written += trail_bytes;
}

#endif

template <size_t n>
FJXL_INLINE void StoreToWriter(const Bits32* bits, BitWriter& output) {
#ifdef FJXL_AVX512
  static_assert(n <= 2, "n should be less or 2 for AVX512");
  StoreToWriterAVX512(bits[0], output);
  if (n == 2) {
    StoreToWriterAVX512(bits[1], output);
  }
  return;
#endif
  static_assert(n <= 4, "n should be less or 4");
  alignas(64) uint64_t nbits64[Bits64::kLanes * n];
  alignas(64) uint64_t bits64[Bits64::kLanes * n];
  bits[0].Merge().Store(nbits64, bits64);
  if (n > 1) {
    bits[1].Merge().Store(nbits64 + Bits64::kLanes, bits64 + Bits64::kLanes);
  }
  if (n > 2) {
    bits[2].Merge().Store(nbits64 + 2 * Bits64::kLanes,
                          bits64 + 2 * Bits64::kLanes);
  }
  if (n > 3) {
    bits[3].Merge().Store(nbits64 + 3 * Bits64::kLanes,
                          bits64 + 3 * Bits64::kLanes);
  }
  output.WriteMultiple(nbits64, bits64, Bits64::kLanes * n);
}

namespace detail {
template <typename T>
struct IntegerTypes;

template <>
struct IntegerTypes<SIMDVec16> {
  using signed_ = int16_t;
  using unsigned_ = uint16_t;
};

template <>
struct IntegerTypes<SIMDVec32> {
  using signed_ = int32_t;
  using unsigned_ = uint32_t;
};

template <typename T>
struct SIMDType;

template <>
struct SIMDType<int16_t> {
  using type = SIMDVec16;
};

template <>
struct SIMDType<int32_t> {
  using type = SIMDVec32;
};

}  // namespace detail

template <typename T>
using signed_t = typename detail::IntegerTypes<T>::signed_;

template <typename T>
using unsigned_t = typename detail::IntegerTypes<T>::unsigned_;

template <typename T>
using simd_t = typename detail::SIMDType<T>::type;

// This function will process exactly one vector worth of pixels.

template <typename T>
size_t PredictPixels(const signed_t<T>* pixels, const signed_t<T>* pixels_left,
                     const signed_t<T>* pixels_top,
                     const signed_t<T>* pixels_topleft,
                     unsigned_t<T>* residuals) {
  T px = T::Load((unsigned_t<T>*)pixels);
  T left = T::Load((unsigned_t<T>*)pixels_left);
  T top = T::Load((unsigned_t<T>*)pixels_top);
  T topleft = T::Load((unsigned_t<T>*)pixels_topleft);
  T ac = left.Sub(topleft);
  T ab = left.Sub(top);
  T bc = top.Sub(topleft);
  T grad = ac.Add(top);
  T d = ab.Xor(bc);
  T zero = T::Val(0);
  T clamp = zero.Gt(d).IfThenElse(top, left);
  T s = ac.Xor(bc);
  T pred = zero.Gt(s).IfThenElse(grad, clamp);
  T res = px.Sub(pred);
  T res_times_2 = res.Add(res);
  res = zero.Gt(res).IfThenElse(T::Val(-1).Sub(res_times_2), res_times_2);
  res.Store(residuals);
  return res.Eq(T::Val(0)).CountPrefix();
}

#endif

void EncodeHybridUint000(uint32_t value, uint32_t* token, uint32_t* nbits,
                         uint32_t* bits) {
  uint32_t n = FloorLog2(value);
  *token = value ? n + 1 : 0;
  *nbits = value ? n : 0;
  *bits = value ? value - (1 << n) : 0;
}

#ifdef FJXL_AVX512
constexpr static size_t kLogChunkSize = 5;
#elif defined(FJXL_AVX2) || defined(FJXL_NEON)
// Even if NEON only has 128-bit lanes, it is still significantly (~1.3x) faster
// to process two vectors at a time.
constexpr static size_t kLogChunkSize = 4;
#else
constexpr static size_t kLogChunkSize = 3;
#endif

constexpr static size_t kChunkSize = 1 << kLogChunkSize;

template <typename Residual>
void GenericEncodeChunk(const Residual* residuals, size_t n, size_t skip,
                        const PrefixCode& code, BitWriter& output) {
  for (size_t ix = skip; ix < n; ix++) {
    unsigned token, nbits, bits;
    EncodeHybridUint000(residuals[ix], &token, &nbits, &bits);
    output.Write(code.raw_nbits[token] + nbits,
                 code.raw_bits[token] | bits << code.raw_nbits[token]);
  }
}

struct UpTo8Bits {
  size_t bitdepth;
  explicit UpTo8Bits(size_t bitdepth) : bitdepth(bitdepth) {
    assert(bitdepth <= 8);
  }
  // Here we can fit up to 9 extra bits + 7 Huffman bits in a u16; for all other
  // symbols, we could actually go up to 8 Huffman bits as we have at most 8
  // extra bits; however, the SIMD bit merging logic for AVX2 assumes that no
  // Huffman length is 8 or more, so we cap at 8 anyway. Last symbol is used for
  // LZ77 lengths and has no limitations except allowing to represent 32 symbols
  // in total.
  static constexpr uint8_t kMinRawLength[12] = {};
  static constexpr uint8_t kMaxRawLength[12] = {
      7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 10,
  };
  static size_t MaxEncodedBitsPerSample() { return 16; }
  static constexpr size_t kInputBytes = 1;
  using pixel_t = int16_t;
  using upixel_t = uint16_t;

  static void PrepareForSimd(const uint8_t* nbits, const uint8_t* bits,
                             size_t n, uint8_t* nbits_simd,
                             uint8_t* bits_simd) {
    assert(n <= 16);
    memcpy(nbits_simd, nbits, 16);
    memcpy(bits_simd, bits, 16);
  }

#ifdef FJXL_GENERIC_SIMD
  static void EncodeChunkSimd(upixel_t* residuals, size_t n, size_t skip,
                              const uint8_t* raw_nbits_simd,
                              const uint8_t* raw_bits_simd, BitWriter& output) {
    Bits32 bits32[kChunkSize / SIMDVec16::kLanes];
    alignas(64) uint16_t bits[SIMDVec16::kLanes];
    alignas(64) uint16_t nbits[SIMDVec16::kLanes];
    alignas(64) uint16_t bits_huff[SIMDVec16::kLanes];
    alignas(64) uint16_t nbits_huff[SIMDVec16::kLanes];
    alignas(64) uint16_t token[SIMDVec16::kLanes];
    for (size_t i = 0; i < kChunkSize; i += SIMDVec16::kLanes) {
      TokenizeSIMD(residuals + i, token, nbits, bits);
      HuffmanSIMDUpTo13(token, raw_nbits_simd, raw_bits_simd, nbits_huff,
                        bits_huff);
      StoreSIMDUpTo8(nbits, bits, nbits_huff, bits_huff, std::max(n, i) - i,
                     std::max(skip, i) - i, bits32 + i / SIMDVec16::kLanes);
    }
    StoreToWriter<kChunkSize / SIMDVec16::kLanes>(bits32, output);
  }
#endif

  size_t NumSymbols(bool doing_ycocg_or_large_palette) const {
    // values gain 1 bit for YCoCg, 1 bit for prediction.
    // Maximum symbol is 1 + effective bit depth of residuals.
    if (doing_ycocg_or_large_palette) {
      return bitdepth + 3;
    } else {
      return bitdepth + 2;
    }
  }
};
constexpr uint8_t UpTo8Bits::kMinRawLength[];
constexpr uint8_t UpTo8Bits::kMaxRawLength[];

struct From9To13Bits {
  size_t bitdepth;
  explicit From9To13Bits(size_t bitdepth) : bitdepth(bitdepth) {
    assert(bitdepth <= 13 && bitdepth >= 9);
  }
  // Last symbol is used for LZ77 lengths and has no limitations except allowing
  // to represent 32 symbols in total.
  // We cannot fit all the bits in a u16, so do not even try and use up to 8
  // bits per raw symbol.
  // There are at most 16 raw symbols, so Huffman coding can be SIMDfied without
  // any special tricks.
  static constexpr uint8_t kMinRawLength[17] = {};
  static constexpr uint8_t kMaxRawLength[17] = {
      8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 10,
  };
  static size_t MaxEncodedBitsPerSample() { return 22; }
  static constexpr size_t kInputBytes = 2;
  using pixel_t = int16_t;
  using upixel_t = uint16_t;

  static void PrepareForSimd(const uint8_t* nbits, const uint8_t* bits,
                             size_t n, uint8_t* nbits_simd,
                             uint8_t* bits_simd) {
    assert(n <= 16);
    memcpy(nbits_simd, nbits, 16);
    memcpy(bits_simd, bits, 16);
  }

#ifdef FJXL_GENERIC_SIMD
  static void EncodeChunkSimd(upixel_t* residuals, size_t n, size_t skip,
                              const uint8_t* raw_nbits_simd,
                              const uint8_t* raw_bits_simd, BitWriter& output) {
    Bits32 bits32[2 * kChunkSize / SIMDVec16::kLanes];
    alignas(64) uint16_t bits[SIMDVec16::kLanes];
    alignas(64) uint16_t nbits[SIMDVec16::kLanes];
    alignas(64) uint16_t bits_huff[SIMDVec16::kLanes];
    alignas(64) uint16_t nbits_huff[SIMDVec16::kLanes];
    alignas(64) uint16_t token[SIMDVec16::kLanes];
    for (size_t i = 0; i < kChunkSize; i += SIMDVec16::kLanes) {
      TokenizeSIMD(residuals + i, token, nbits, bits);
      HuffmanSIMDUpTo13(token, raw_nbits_simd, raw_bits_simd, nbits_huff,
                        bits_huff);
      StoreSIMDUpTo14(nbits, bits, nbits_huff, bits_huff, std::max(n, i) - i,
                      std::max(skip, i) - i,
                      bits32 + 2 * i / SIMDVec16::kLanes);
    }
    StoreToWriter<2 * kChunkSize / SIMDVec16::kLanes>(bits32, output);
  }
#endif

  size_t NumSymbols(bool doing_ycocg_or_large_palette) const {
    // values gain 1 bit for YCoCg, 1 bit for prediction.
    // Maximum symbol is 1 + effective bit depth of residuals.
    if (doing_ycocg_or_large_palette) {
      return bitdepth + 3;
    } else {
      return bitdepth + 2;
    }
  }
};
constexpr uint8_t From9To13Bits::kMinRawLength[];
constexpr uint8_t From9To13Bits::kMaxRawLength[];

void CheckHuffmanBitsSIMD(int bits1, int nbits1, int bits2, int nbits2) {
  assert(nbits1 == 8);
  assert(nbits2 == 8);
  assert(bits2 == (bits1 | 128));
}

struct Exactly14Bits {
  explicit Exactly14Bits(size_t bitdepth_) { assert(bitdepth_ == 14); }
  // Force LZ77 symbols to have at least 8 bits, and raw symbols 15 and 16 to
  // have exactly 8, and no other symbol to have 8 or more. This ensures that
  // the representation for 15 and 16 is identical up to one bit.
  static constexpr uint8_t kMinRawLength[18] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 7,
  };
  static constexpr uint8_t kMaxRawLength[18] = {
      7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 10,
  };
  static constexpr size_t bitdepth = 14;
  static size_t MaxEncodedBitsPerSample() { return 23; }
  static constexpr size_t kInputBytes = 2;
  using pixel_t = int16_t;
  using upixel_t = uint16_t;

  static void PrepareForSimd(const uint8_t* nbits, const uint8_t* bits,
                             size_t n, uint8_t* nbits_simd,
                             uint8_t* bits_simd) {
    assert(n == 17);
    CheckHuffmanBitsSIMD(bits[15], nbits[15], bits[16], nbits[16]);
    memcpy(nbits_simd, nbits, 16);
    memcpy(bits_simd, bits, 16);
  }

#ifdef FJXL_GENERIC_SIMD
  static void EncodeChunkSimd(upixel_t* residuals, size_t n, size_t skip,
                              const uint8_t* raw_nbits_simd,
                              const uint8_t* raw_bits_simd, BitWriter& output) {
    Bits32 bits32[2 * kChunkSize / SIMDVec16::kLanes];
    alignas(64) uint16_t bits[SIMDVec16::kLanes];
    alignas(64) uint16_t nbits[SIMDVec16::kLanes];
    alignas(64) uint16_t bits_huff[SIMDVec16::kLanes];
    alignas(64) uint16_t nbits_huff[SIMDVec16::kLanes];
    alignas(64) uint16_t token[SIMDVec16::kLanes];
    for (size_t i = 0; i < kChunkSize; i += SIMDVec16::kLanes) {
      TokenizeSIMD(residuals + i, token, nbits, bits);
      HuffmanSIMD14(token, raw_nbits_simd, raw_bits_simd, nbits_huff,
                    bits_huff);
      StoreSIMDUpTo14(nbits, bits, nbits_huff, bits_huff, std::max(n, i) - i,
                      std::max(skip, i) - i,
                      bits32 + 2 * i / SIMDVec16::kLanes);
    }
    StoreToWriter<2 * kChunkSize / SIMDVec16::kLanes>(bits32, output);
  }
#endif

  size_t NumSymbols(bool) const { return 17; }
};
constexpr uint8_t Exactly14Bits::kMinRawLength[];
constexpr uint8_t Exactly14Bits::kMaxRawLength[];

struct MoreThan14Bits {
  size_t bitdepth;
  explicit MoreThan14Bits(size_t bitdepth) : bitdepth(bitdepth) {
    assert(bitdepth > 14);
    assert(bitdepth <= 16);
  }
  // Force LZ77 symbols to have at least 8 bits, and raw symbols 13 to 18 to
  // have exactly 8, and no other symbol to have 8 or more. This ensures that
  // the representation for (13, 14), (15, 16), (17, 18) is identical up to one
  // bit.
  static constexpr uint8_t kMinRawLength[20] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 8, 7,
  };
  static constexpr uint8_t kMaxRawLength[20] = {
      7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 10,
  };
  static size_t MaxEncodedBitsPerSample() { return 25; }
  static constexpr size_t kInputBytes = 2;
  using pixel_t = int32_t;
  using upixel_t = uint32_t;

  static void PrepareForSimd(const uint8_t* nbits, const uint8_t* bits,
                             size_t n, uint8_t* nbits_simd,
                             uint8_t* bits_simd) {
    assert(n == 19);
    CheckHuffmanBitsSIMD(bits[13], nbits[13], bits[14], nbits[14]);
    CheckHuffmanBitsSIMD(bits[15], nbits[15], bits[16], nbits[16]);
    CheckHuffmanBitsSIMD(bits[17], nbits[17], bits[18], nbits[18]);
    for (size_t i = 0; i < 14; i++) {
      nbits_simd[i] = nbits[i];
      bits_simd[i] = bits[i];
    }
    nbits_simd[14] = nbits[15];
    bits_simd[14] = bits[15];
    nbits_simd[15] = nbits[17];
    bits_simd[15] = bits[17];
  }

#ifdef FJXL_GENERIC_SIMD
  static void EncodeChunkSimd(upixel_t* residuals, size_t n, size_t skip,
                              const uint8_t* raw_nbits_simd,
                              const uint8_t* raw_bits_simd, BitWriter& output) {
    Bits32 bits32[2 * kChunkSize / SIMDVec16::kLanes];
    alignas(64) uint32_t bits[SIMDVec16::kLanes];
    alignas(64) uint32_t nbits[SIMDVec16::kLanes];
    alignas(64) uint16_t bits_huff[SIMDVec16::kLanes];
    alignas(64) uint16_t nbits_huff[SIMDVec16::kLanes];
    alignas(64) uint16_t token[SIMDVec16::kLanes];
    for (size_t i = 0; i < kChunkSize; i += SIMDVec16::kLanes) {
      TokenizeSIMD(residuals + i, token, nbits, bits);
      HuffmanSIMDAbove14(token, raw_nbits_simd, raw_bits_simd, nbits_huff,
                         bits_huff);
      StoreSIMDAbove14(nbits, bits, nbits_huff, bits_huff, std::max(n, i) - i,
                       std::max(skip, i) - i,
                       bits32 + 2 * i / SIMDVec16::kLanes);
    }
    StoreToWriter<2 * kChunkSize / SIMDVec16::kLanes>(bits32, output);
  }
#endif
  size_t NumSymbols(bool) const { return 19; }
};
constexpr uint8_t MoreThan14Bits::kMinRawLength[];
constexpr uint8_t MoreThan14Bits::kMaxRawLength[];

bool PrepareDCGlobalCommon(bool is_single_group, size_t width, size_t height,
                           size_t max_encoded_bits_per_sample,
                           const PrefixCode code[4], BitWriter* output) {
  size_t num_samples = is_single_group ? (width * height) : 0;
  if (!output->Allocate(100000 + num_samples * max_encoded_bits_per_sample)) {
    return false;
  }
  // No patches, spline or noise.
  output->Write(1, 1);  // default DC dequantization factors (?)
  output->Write(1, 1);  // use global tree / histograms
  output->Write(1, 0);  // no lz77 for the tree

  output->Write(1, 1);         // simple code for the tree's context map
  output->Write(2, 0);         // all contexts clustered together
  output->Write(1, 1);         // use prefix code for tree
  output->Write(4, 0);         // 000 hybrid uint
  output->Write(6, 0b100011);  // Alphabet size is 4 (var16)
  output->Write(2, 1);         // simple prefix code
  output->Write(2, 3);         // with 4 symbols
  output->Write(2, 0);
  output->Write(2, 1);
  output->Write(2, 2);
  output->Write(2, 3);
  output->Write(1, 0);  // First tree encoding option

  // Huffman table + extra bits for the tree.
  uint8_t symbol_bits[6] = {0b00, 0b10, 0b001, 0b101, 0b0011, 0b0111};
  uint8_t symbol_nbits[6] = {2, 2, 3, 3, 4, 4};
  // Write a tree with a leaf per channel, and gradient predictor for every
  // leaf.
  for (auto v : {1, 2, 1, 4, 1, 0, 0, 5, 0, 0, 0, 0, 5,
                 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0}) {
    output->Write(symbol_nbits[v], symbol_bits[v]);
  }

  output->Write(1, 1);     // Enable lz77 for the main bitstream
  output->Write(2, 0b00);  // lz77 offset 224
  static_assert(kLZ77Offset == 224, "kLZ77Offset should be 224");
  output->Write(4, 0b1010);  // lz77 min length 7
  // 400 hybrid uint config for lz77
  output->Write(4, 4);
  output->Write(3, 0);
  output->Write(3, 0);

  output->Write(1, 1);  // simple code for the context map
  output->Write(2, 3);  // 3 bits per entry
  output->Write(3, 4);  // channel 3
  output->Write(3, 3);  // channel 2
  output->Write(3, 2);  // channel 1
  output->Write(3, 1);  // channel 0
  output->Write(3, 0);  // distance histogram first

  output->Write(1, 1);  // use prefix codes
  output->Write(4, 0);  // 000 hybrid uint config for distances (only need 0)
  for (size_t i = 0; i < 4; i++) {
    output->Write(4, 0);  // 000 hybrid uint config for symbols (only <= 10)
  }

  // Distance alphabet size:
  output->Write(5, 0b00001);  // 2: just need 1 for RLE (i.e. distance 1)
  // Symbol + LZ77 alphabet size:
  for (size_t i = 0; i < 4; i++) {
    output->Write(1, 1);    // > 1
    output->Write(4, 8);    // <= 512
    output->Write(8, 256);  // == 512
  }

  // Distance histogram:
  output->Write(2, 1);  // simple prefix code
  output->Write(2, 0);  // with one symbol
  output->Write(1, 1);  // 1

  // Symbol + lz77 histogram:
  for (size_t i = 0; i < 4; i++) {
    code[i].WriteTo(output);
  }

  // Group header for global modular image.
  output->Write(1, 1);  // Global tree
  output->Write(1, 1);  // All default wp
  return true;
}

bool PrepareDCGlobal(bool is_single_group, size_t width, size_t height,
                     size_t max_encoded_bits_per_sample, size_t nb_chans,
                     const PrefixCode code[4], BitWriter* output) {
  if (!PrepareDCGlobalCommon(is_single_group, width, height,
                             max_encoded_bits_per_sample, code, output)) {
    return false;
  }
  if (nb_chans > 2) {
    output->Write(2, 0b01);     // 1 transform
    output->Write(2, 0b00);     // RCT
    output->Write(5, 0b00000);  // Starting from ch 0
    output->Write(2, 0b00);     // YCoCg
  } else {
    output->Write(2, 0b00);  // no transforms
  }
  if (!is_single_group) {
    output->ZeroPadToByte();
  }
  return true;
}

template <typename BitDepth>
struct ChunkEncoder {
  void PrepareForSimd() {
    BitDepth::PrepareForSimd(code->raw_nbits, code->raw_bits, code->numraw,
                             raw_nbits_simd, raw_bits_simd);
  }
  FJXL_INLINE static void EncodeRle(size_t count, const PrefixCode& code,
                                    BitWriter& output) {
    if (count == 0) return;
    count -= kLZ77MinLength + 1;
    if (count < kLZ77CacheSize) {
      output.Write(code.lz77_cache_nbits[count], code.lz77_cache_bits[count]);
    } else {
      unsigned token, nbits, bits;
      EncodeHybridUintLZ77(count, &token, &nbits, &bits);
      uint64_t wbits = bits;
      wbits = (wbits << code.lz77_nbits[token]) | code.lz77_bits[token];
      wbits = (wbits << code.raw_nbits[0]) | code.raw_bits[0];
      output.Write(code.lz77_nbits[token] + nbits + code.raw_nbits[0], wbits);
    }
  }

  FJXL_INLINE void Chunk(size_t run, typename BitDepth::upixel_t* residuals,
                         size_t skip, size_t n) {
    EncodeRle(run, *code, *output);
#ifdef FJXL_GENERIC_SIMD
    BitDepth::EncodeChunkSimd(residuals, n, skip, raw_nbits_simd, raw_bits_simd,
                              *output);
#else
    GenericEncodeChunk(residuals, n, skip, *code, *output);
#endif
  }

  inline void Finalize(size_t run) { EncodeRle(run, *code, *output); }

  const PrefixCode* code;
  BitWriter* output;
  alignas(64) uint8_t raw_nbits_simd[16] = {};
  alignas(64) uint8_t raw_bits_simd[16] = {};
};

template <typename BitDepth>
struct ChunkSampleCollector {
  FJXL_INLINE void Rle(size_t count, uint64_t* lz77_counts_) {
    if (count == 0) return;
    raw_counts[0] += 1;
    count -= kLZ77MinLength + 1;
    unsigned token, nbits, bits;
    EncodeHybridUintLZ77(count, &token, &nbits, &bits);
    lz77_counts_[token]++;
  }

  FJXL_INLINE void Chunk(size_t run, typename BitDepth::upixel_t* residuals,
                         size_t skip, size_t n) {
    // Run is broken. Encode the run and encode the individual vector.
    Rle(run, lz77_counts);
    for (size_t ix = skip; ix < n; ix++) {
      unsigned token, nbits, bits;
      EncodeHybridUint000(residuals[ix], &token, &nbits, &bits);
      raw_counts[token]++;
    }
  }

  // don't count final run since we don't know how long it really is
  void Finalize(size_t run) {}

  uint64_t* raw_counts;
  uint64_t* lz77_counts;
};

constexpr uint32_t PackSigned(int32_t value) {
  return (static_cast<uint32_t>(value) << 1) ^
         ((static_cast<uint32_t>(~value) >> 31) - 1);
}

template <typename T, typename BitDepth>
struct ChannelRowProcessor {
  using upixel_t = typename BitDepth::upixel_t;
  using pixel_t = typename BitDepth::pixel_t;
  T* t;
  void ProcessChunk(const pixel_t* row, const pixel_t* row_left,
                    const pixel_t* row_top, const pixel_t* row_topleft,
                    size_t n) {
    alignas(64) upixel_t residuals[kChunkSize] = {};
    size_t prefix_size = 0;
    size_t required_prefix_size = 0;
#ifdef FJXL_GENERIC_SIMD
    constexpr size_t kNum =
        sizeof(pixel_t) == 2 ? SIMDVec16::kLanes : SIMDVec32::kLanes;
    for (size_t ix = 0; ix < kChunkSize; ix += kNum) {
      size_t c =
          PredictPixels<simd_t<pixel_t>>(row + ix, row_left + ix, row_top + ix,
                                         row_topleft + ix, residuals + ix);
      prefix_size =
          prefix_size == required_prefix_size ? prefix_size + c : prefix_size;
      required_prefix_size += kNum;
    }
#else
    for (size_t ix = 0; ix < kChunkSize; ix++) {
      pixel_t px = row[ix];
      pixel_t left = row_left[ix];
      pixel_t top = row_top[ix];
      pixel_t topleft = row_topleft[ix];
      pixel_t ac = left - topleft;
      pixel_t ab = left - top;
      pixel_t bc = top - topleft;
      pixel_t grad = static_cast<pixel_t>(static_cast<upixel_t>(ac) +
                                          static_cast<upixel_t>(top));
      pixel_t d = ab ^ bc;
      pixel_t clamp = d < 0 ? top : left;
      pixel_t s = ac ^ bc;
      pixel_t pred = s < 0 ? grad : clamp;
      residuals[ix] = PackSigned(px - pred);
      prefix_size = prefix_size == required_prefix_size
                        ? prefix_size + (residuals[ix] == 0)
                        : prefix_size;
      required_prefix_size += 1;
    }
#endif
    prefix_size = std::min(n, prefix_size);
    if (prefix_size == n && (run > 0 || prefix_size > kLZ77MinLength)) {
      // Run continues, nothing to do.
      run += prefix_size;
    } else if (prefix_size + run > kLZ77MinLength) {
      // Run is broken. Encode the run and encode the individual vector.
      t->Chunk(run + prefix_size, residuals, prefix_size, n);
      run = 0;
    } else {
      // There was no run to begin with.
      t->Chunk(0, residuals, 0, n);
    }
  }

  void ProcessRow(const pixel_t* row, const pixel_t* row_left,
                  const pixel_t* row_top, const pixel_t* row_topleft,
                  size_t xs) {
    for (size_t x = 0; x < xs; x += kChunkSize) {
      ProcessChunk(row + x, row_left + x, row_top + x, row_topleft + x,
                   std::min(kChunkSize, xs - x));
    }
  }

  void Finalize() { t->Finalize(run); }
  // Invariant: run == 0 or run > kLZ77MinLength.
  size_t run = 0;
};

uint16_t LoadLE16(const unsigned char* ptr) {
  return uint16_t{ptr[0]} | (uint16_t{ptr[1]} << 8);
}

uint16_t SwapEndian(uint16_t in) { return (in >> 8) | (in << 8); }

#ifdef FJXL_GENERIC_SIMD
void StorePixels(SIMDVec16 p, int16_t* dest) { p.Store((uint16_t*)dest); }

void StorePixels(SIMDVec16 p, int32_t* dest) {
  VecPair<SIMDVec32> p_up = p.Upcast();
  p_up.low.Store((uint32_t*)dest);
  p_up.hi.Store((uint32_t*)dest + SIMDVec32::kLanes);
}
#endif

template <typename pixel_t>
void FillRowG8(const unsigned char* rgba, size_t oxs, pixel_t* luma) {
  size_t x = 0;
#ifdef FJXL_GENERIC_SIMD
  for (; x + SIMDVec16::kLanes <= oxs; x += SIMDVec16::kLanes) {
    auto rgb = SIMDVec16::LoadG8(rgba + x);
    StorePixels(rgb[0], luma + x);
  }
#endif
  for (; x < oxs; x++) {
    luma[x] = rgba[x];
  }
}

template <bool big_endian, typename pixel_t>
void FillRowG16(const unsigned char* rgba, size_t oxs, pixel_t* luma) {
  size_t x = 0;
#ifdef FJXL_GENERIC_SIMD
  for (; x + SIMDVec16::kLanes <= oxs; x += SIMDVec16::kLanes) {
    auto rgb = SIMDVec16::LoadG16(rgba + 2 * x);
    if (big_endian) {
      rgb[0].SwapEndian();
    }
    StorePixels(rgb[0], luma + x);
  }
#endif
  for (; x < oxs; x++) {
    uint16_t val = LoadLE16(rgba + 2 * x);
    if (big_endian) {
      val = SwapEndian(val);
    }
    luma[x] = val;
  }
}

template <typename pixel_t>
void FillRowGA8(const unsigned char* rgba, size_t oxs, pixel_t* luma,
                pixel_t* alpha) {
  size_t x = 0;
#ifdef FJXL_GENERIC_SIMD
  for (; x + SIMDVec16::kLanes <= oxs; x += SIMDVec16::kLanes) {
    auto rgb = SIMDVec16::LoadGA8(rgba + 2 * x);
    StorePixels(rgb[0], luma + x);
    StorePixels(rgb[1], alpha + x);
  }
#endif
  for (; x < oxs; x++) {
    luma[x] = rgba[2 * x];
    alpha[x] = rgba[2 * x + 1];
  }
}

template <bool big_endian, typename pixel_t>
void FillRowGA16(const unsigned char* rgba, size_t oxs, pixel_t* luma,
                 pixel_t* alpha) {
  size_t x = 0;
#ifdef FJXL_GENERIC_SIMD
  for (; x + SIMDVec16::kLanes <= oxs; x += SIMDVec16::kLanes) {
    auto rgb = SIMDVec16::LoadGA16(rgba + 4 * x);
    if (big_endian) {
      rgb[0].SwapEndian();
      rgb[1].SwapEndian();
    }
    StorePixels(rgb[0], luma + x);
    StorePixels(rgb[1], alpha + x);
  }
#endif
  for (; x < oxs; x++) {
    uint16_t l = LoadLE16(rgba + 4 * x);
    uint16_t a = LoadLE16(rgba + 4 * x + 2);
    if (big_endian) {
      l = SwapEndian(l);
      a = SwapEndian(a);
    }
    luma[x] = l;
    alpha[x] = a;
  }
}

template <typename pixel_t>
void StoreYCoCg(pixel_t r, pixel_t g, pixel_t b, pixel_t* y, pixel_t* co,
                pixel_t* cg) {
  *co = r - b;
  pixel_t tmp = b + (*co >> 1);
  *cg = g - tmp;
  *y = tmp + (*cg >> 1);
}

#ifdef FJXL_GENERIC_SIMD
void StoreYCoCg(SIMDVec16 r, SIMDVec16 g, SIMDVec16 b, int16_t* y, int16_t* co,
                int16_t* cg) {
  SIMDVec16 co_v = r.Sub(b);
  SIMDVec16 tmp = b.Add(co_v.SignedShiftRight<1>());
  SIMDVec16 cg_v = g.Sub(tmp);
  SIMDVec16 y_v = tmp.Add(cg_v.SignedShiftRight<1>());
  y_v.Store(reinterpret_cast<uint16_t*>(y));
  co_v.Store(reinterpret_cast<uint16_t*>(co));
  cg_v.Store(reinterpret_cast<uint16_t*>(cg));
}

void StoreYCoCg(SIMDVec16 r, SIMDVec16 g, SIMDVec16 b, int32_t* y, int32_t* co,
                int32_t* cg) {
  VecPair<SIMDVec32> r_up = r.Upcast();
  VecPair<SIMDVec32> g_up = g.Upcast();
  VecPair<SIMDVec32> b_up = b.Upcast();
  SIMDVec32 co_lo_v = r_up.low.Sub(b_up.low);
  SIMDVec32 tmp_lo = b_up.low.Add(co_lo_v.SignedShiftRight<1>());
  SIMDVec32 cg_lo_v = g_up.low.Sub(tmp_lo);
  SIMDVec32 y_lo_v = tmp_lo.Add(cg_lo_v.SignedShiftRight<1>());
  SIMDVec32 co_hi_v = r_up.hi.Sub(b_up.hi);
  SIMDVec32 tmp_hi = b_up.hi.Add(co_hi_v.SignedShiftRight<1>());
  SIMDVec32 cg_hi_v = g_up.hi.Sub(tmp_hi);
  SIMDVec32 y_hi_v = tmp_hi.Add(cg_hi_v.SignedShiftRight<1>());
  y_lo_v.Store(reinterpret_cast<uint32_t*>(y));
  co_lo_v.Store(reinterpret_cast<uint32_t*>(co));
  cg_lo_v.Store(reinterpret_cast<uint32_t*>(cg));
  y_hi_v.Store(reinterpret_cast<uint32_t*>(y) + SIMDVec32::kLanes);
  co_hi_v.Store(reinterpret_cast<uint32_t*>(co) + SIMDVec32::kLanes);
  cg_hi_v.Store(reinterpret_cast<uint32_t*>(cg) + SIMDVec32::kLanes);
}
#endif

template <typename pixel_t>
void FillRowRGB8(const unsigned char* rgba, size_t oxs, pixel_t* y, pixel_t* co,
                 pixel_t* cg) {
  size_t x = 0;
#ifdef FJXL_GENERIC_SIMD
  for (; x + SIMDVec16::kLanes <= oxs; x += SIMDVec16::kLanes) {
    auto rgb = SIMDVec16::LoadRGB8(rgba + 3 * x);
    StoreYCoCg(rgb[0], rgb[1], rgb[2], y + x, co + x, cg + x);
  }
#endif
  for (; x < oxs; x++) {
    uint16_t r = rgba[3 * x];
    uint16_t g = rgba[3 * x + 1];
    uint16_t b = rgba[3 * x + 2];
    StoreYCoCg<pixel_t>(r, g, b, y + x, co + x, cg + x);
  }
}

template <bool big_endian, typename pixel_t>
void FillRowRGB16(const unsigned char* rgba, size_t oxs, pixel_t* y,
                  pixel_t* co, pixel_t* cg) {
  size_t x = 0;
#ifdef FJXL_GENERIC_SIMD
  for (; x + SIMDVec16::kLanes <= oxs; x += SIMDVec16::kLanes) {
    auto rgb = SIMDVec16::LoadRGB16(rgba + 6 * x);
    if (big_endian) {
      rgb[0].SwapEndian();
      rgb[1].SwapEndian();
      rgb[2].SwapEndian();
    }
    StoreYCoCg(rgb[0], rgb[1], rgb[2], y + x, co + x, cg + x);
  }
#endif
  for (; x < oxs; x++) {
    uint16_t r = LoadLE16(rgba + 6 * x);
    uint16_t g = LoadLE16(rgba + 6 * x + 2);
    uint16_t b = LoadLE16(rgba + 6 * x + 4);
    if (big_endian) {
      r = SwapEndian(r);
      g = SwapEndian(g);
      b = SwapEndian(b);
    }
    StoreYCoCg<pixel_t>(r, g, b, y + x, co + x, cg + x);
  }
}

template <typename pixel_t>
void FillRowRGBA8(const unsigned char* rgba, size_t oxs, pixel_t* y,
                  pixel_t* co, pixel_t* cg, pixel_t* alpha) {
  size_t x = 0;
#ifdef FJXL_GENERIC_SIMD
  for (; x + SIMDVec16::kLanes <= oxs; x += SIMDVec16::kLanes) {
    auto rgb = SIMDVec16::LoadRGBA8(rgba + 4 * x);
    StoreYCoCg(rgb[0], rgb[1], rgb[2], y + x, co + x, cg + x);
    StorePixels(rgb[3], alpha + x);
  }
#endif
  for (; x < oxs; x++) {
    uint16_t r = rgba[4 * x];
    uint16_t g = rgba[4 * x + 1];
    uint16_t b = rgba[4 * x + 2];
    uint16_t a = rgba[4 * x + 3];
    StoreYCoCg<pixel_t>(r, g, b, y + x, co + x, cg + x);
    alpha[x] = a;
  }
}

template <bool big_endian, typename pixel_t>
void FillRowRGBA16(const unsigned char* rgba, size_t oxs, pixel_t* y,
                   pixel_t* co, pixel_t* cg, pixel_t* alpha) {
  size_t x = 0;
#ifdef FJXL_GENERIC_SIMD
  for (; x + SIMDVec16::kLanes <= oxs; x += SIMDVec16::kLanes) {
    auto rgb = SIMDVec16::LoadRGBA16(rgba + 8 * x);
    if (big_endian) {
      rgb[0].SwapEndian();
      rgb[1].SwapEndian();
      rgb[2].SwapEndian();
      rgb[3].SwapEndian();
    }
    StoreYCoCg(rgb[0], rgb[1], rgb[2], y + x, co + x, cg + x);
    StorePixels(rgb[3], alpha + x);
  }
#endif
  for (; x < oxs; x++) {
    uint16_t r = LoadLE16(rgba + 8 * x);
    uint16_t g = LoadLE16(rgba + 8 * x + 2);
    uint16_t b = LoadLE16(rgba + 8 * x + 4);
    uint16_t a = LoadLE16(rgba + 8 * x + 6);
    if (big_endian) {
      r = SwapEndian(r);
      g = SwapEndian(g);
      b = SwapEndian(b);
      a = SwapEndian(a);
    }
    StoreYCoCg<pixel_t>(r, g, b, y + x, co + x, cg + x);
    alpha[x] = a;
  }
}

template <typename Processor, typename BitDepth>
void ProcessImageArea(const unsigned char* rgba, size_t x0, size_t y0,
                      size_t xs, size_t yskip, size_t ys, size_t row_stride,
                      BitDepth bitdepth, size_t nb_chans, bool big_endian,
                      Processor* processors) {
  constexpr size_t kPadding = 32;

  using pixel_t = typename BitDepth::pixel_t;

  constexpr size_t kAlign = 64;
  constexpr size_t kAlignPixels = kAlign / sizeof(pixel_t);

  auto align = [=](pixel_t* ptr) {
    size_t offset = reinterpret_cast<size_t>(ptr) % kAlign;
    if (offset) {
      ptr += offset / sizeof(pixel_t);
    }
    return ptr;
  };

  constexpr size_t kNumPx =
      (256 + kPadding * 2 + kAlignPixels + kAlignPixels - 1) / kAlignPixels *
      kAlignPixels;

  std::vector<std::array<std::array<pixel_t, kNumPx>, 2>> group_data(nb_chans);

  for (size_t y = 0; y < ys; y++) {
    const auto rgba_row =
        rgba + row_stride * (y0 + y) + x0 * nb_chans * BitDepth::kInputBytes;
    pixel_t* crow[4] = {};
    pixel_t* prow[4] = {};
    for (size_t i = 0; i < nb_chans; i++) {
      crow[i] = align(&group_data[i][y & 1][kPadding]);
      prow[i] = align(&group_data[i][(y - 1) & 1][kPadding]);
    }

    // Pre-fill rows with YCoCg converted pixels.
    if (nb_chans == 1) {
      if (BitDepth::kInputBytes == 1) {
        FillRowG8(rgba_row, xs, crow[0]);
      } else if (big_endian) {
        FillRowG16</*big_endian=*/true>(rgba_row, xs, crow[0]);
      } else {
        FillRowG16</*big_endian=*/false>(rgba_row, xs, crow[0]);
      }
    } else if (nb_chans == 2) {
      if (BitDepth::kInputBytes == 1) {
        FillRowGA8(rgba_row, xs, crow[0], crow[1]);
      } else if (big_endian) {
        FillRowGA16</*big_endian=*/true>(rgba_row, xs, crow[0], crow[1]);
      } else {
        FillRowGA16</*big_endian=*/false>(rgba_row, xs, crow[0], crow[1]);
      }
    } else if (nb_chans == 3) {
      if (BitDepth::kInputBytes == 1) {
        FillRowRGB8(rgba_row, xs, crow[0], crow[1], crow[2]);
      } else if (big_endian) {
        FillRowRGB16</*big_endian=*/true>(rgba_row, xs, crow[0], crow[1],
                                          crow[2]);
      } else {
        FillRowRGB16</*big_endian=*/false>(rgba_row, xs, crow[0], crow[1],
                                           crow[2]);
      }
    } else {
      if (BitDepth::kInputBytes == 1) {
        FillRowRGBA8(rgba_row, xs, crow[0], crow[1], crow[2], crow[3]);
      } else if (big_endian) {
        FillRowRGBA16</*big_endian=*/true>(rgba_row, xs, crow[0], crow[1],
                                           crow[2], crow[3]);
      } else {
        FillRowRGBA16</*big_endian=*/false>(rgba_row, xs, crow[0], crow[1],
                                            crow[2], crow[3]);
      }
    }
    // Deal with x == 0.
    for (size_t c = 0; c < nb_chans; c++) {
      *(crow[c] - 1) = y > 0 ? *(prow[c]) : 0;
      // Fix topleft.
      *(prow[c] - 1) = y > 0 ? *(prow[c]) : 0;
    }
    if (y < yskip) continue;
    for (size_t c = 0; c < nb_chans; c++) {
      // Get pointers to px/left/top/topleft data to speedup loop.
      const pixel_t* row = crow[c];
      const pixel_t* row_left = crow[c] - 1;
      const pixel_t* row_top = y == 0 ? row_left : prow[c];
      const pixel_t* row_topleft = y == 0 ? row_left : prow[c] - 1;

      processors[c].ProcessRow(row, row_left, row_top, row_topleft, xs);
    }
  }
  for (size_t c = 0; c < nb_chans; c++) {
    processors[c].Finalize();
  }
}

template <typename BitDepth>
bool WriteACSection(const unsigned char* rgba, size_t x0, size_t y0, size_t xs,
                    size_t ys, size_t row_stride, bool is_single_group,
                    BitDepth bitdepth, size_t nb_chans, bool big_endian,
                    const PrefixCode code[4],
                    std::array<BitWriter, 4>& output) {
  for (size_t i = 0; i < nb_chans; i++) {
    if (is_single_group && i == 0) continue;
    if (!output[i].Allocate(xs * ys * bitdepth.MaxEncodedBitsPerSample() + 4)) {
      return false;
    }
  }
  if (!is_single_group) {
    // Group header for modular image.
    // When the image is single-group, the global modular image is the one
    // that contains the pixel data, and there is no group header.
    output[0].Write(1, 1);     // Global tree
    output[0].Write(1, 1);     // All default wp
    output[0].Write(2, 0b00);  // 0 transforms
  }

  ChunkEncoder<BitDepth> encoders[4];
  ChannelRowProcessor<ChunkEncoder<BitDepth>, BitDepth> row_encoders[4];
  for (size_t c = 0; c < nb_chans; c++) {
    row_encoders[c].t = &encoders[c];
    encoders[c].output = &output[c];
    encoders[c].code = &code[c];
    encoders[c].PrepareForSimd();
  }
  ProcessImageArea<ChannelRowProcessor<ChunkEncoder<BitDepth>, BitDepth>>(
      rgba, x0, y0, xs, 0, ys, row_stride, bitdepth, nb_chans, big_endian,
      row_encoders);
  return true;
}

constexpr int kHashExp = 16;
constexpr uint32_t kHashSize = 1 << kHashExp;
constexpr uint32_t kHashMultiplier = 2654435761;
constexpr int kMaxColors = 512;

// can be any function that returns a value in 0 .. kHashSize-1
// has to map 0 to 0
inline uint32_t pixel_hash(uint32_t p) {
  return (p * kHashMultiplier) >> (32 - kHashExp);
}

template <size_t nb_chans>
void FillRowPalette(const unsigned char* inrow, size_t xs,
                    const int16_t* lookup, int16_t* out) {
  for (size_t x = 0; x < xs; x++) {
    uint32_t p = 0;
    for (size_t i = 0; i < nb_chans; ++i) {
      p |= inrow[x * nb_chans + i] << (8 * i);
    }
    out[x] = lookup[pixel_hash(p)];
  }
}

template <typename Processor>
void ProcessImageAreaPalette(const unsigned char* rgba, size_t x0, size_t y0,
                             size_t xs, size_t yskip, size_t ys,
                             size_t row_stride, const int16_t* lookup,
                             size_t nb_chans, Processor* processors) {
  constexpr size_t kPadding = 32;

  std::vector<std::array<int16_t, 256 + kPadding * 2>> group_data(2);
  Processor& row_encoder = processors[0];

  for (size_t y = 0; y < ys; y++) {
    // Pre-fill rows with palette converted pixels.
    const unsigned char* inrow = rgba + row_stride * (y0 + y) + x0 * nb_chans;
    int16_t* outrow = &group_data[y & 1][kPadding];
    if (nb_chans == 1) {
      FillRowPalette<1>(inrow, xs, lookup, outrow);
    } else if (nb_chans == 2) {
      FillRowPalette<2>(inrow, xs, lookup, outrow);
    } else if (nb_chans == 3) {
      FillRowPalette<3>(inrow, xs, lookup, outrow);
    } else if (nb_chans == 4) {
      FillRowPalette<4>(inrow, xs, lookup, outrow);
    }
    // Deal with x == 0.
    group_data[y & 1][kPadding - 1] =
        y > 0 ? group_data[(y - 1) & 1][kPadding] : 0;
    // Fix topleft.
    group_data[(y - 1) & 1][kPadding - 1] =
        y > 0 ? group_data[(y - 1) & 1][kPadding] : 0;
    // Get pointers to px/left/top/topleft data to speedup loop.
    const int16_t* row = &group_data[y & 1][kPadding];
    const int16_t* row_left = &group_data[y & 1][kPadding - 1];
    const int16_t* row_top =
        y == 0 ? row_left : &group_data[(y - 1) & 1][kPadding];
    const int16_t* row_topleft =
        y == 0 ? row_left : &group_data[(y - 1) & 1][kPadding - 1];

    row_encoder.ProcessRow(row, row_left, row_top, row_topleft, xs);
  }
  row_encoder.Finalize();
}

bool WriteACSectionPalette(const unsigned char* rgba, size_t x0, size_t y0,
                           size_t xs, size_t ys, size_t row_stride,
                           bool is_single_group, const PrefixCode code[4],
                           const int16_t* lookup, size_t nb_chans,
                           BitWriter& output) {
  if (!is_single_group) {
    if (!output.Allocate(16 * xs * ys + 4)) return false;
    // Group header for modular image.
    // When the image is single-group, the global modular image is the one
    // that contains the pixel data, and there is no group header.
    output.Write(1, 1);     // Global tree
    output.Write(1, 1);     // All default wp
    output.Write(2, 0b00);  // 0 transforms
  }

  ChunkEncoder<UpTo8Bits> encoder;
  ChannelRowProcessor<ChunkEncoder<UpTo8Bits>, UpTo8Bits> row_encoder;

  row_encoder.t = &encoder;
  encoder.output = &output;
  encoder.code = &code[is_single_group ? 1 : 0];
  encoder.PrepareForSimd();
  ProcessImageAreaPalette<
      ChannelRowProcessor<ChunkEncoder<UpTo8Bits>, UpTo8Bits>>(
      rgba, x0, y0, xs, 0, ys, row_stride, lookup, nb_chans, &row_encoder);
  return true;
}

template <typename BitDepth>
void CollectSamples(const unsigned char* rgba, size_t x0, size_t y0, size_t xs,
                    size_t row_stride, size_t row_count,
                    uint64_t raw_counts[4][kNumRawSymbols],
                    uint64_t lz77_counts[4][kNumLZ77], bool is_single_group,
                    bool palette, BitDepth bitdepth, size_t nb_chans,
                    bool big_endian, const int16_t* lookup) {
  if (palette) {
    ChunkSampleCollector<UpTo8Bits> sample_collectors[4];
    ChannelRowProcessor<ChunkSampleCollector<UpTo8Bits>, UpTo8Bits>
        row_sample_collectors[4];
    for (size_t c = 0; c < nb_chans; c++) {
      row_sample_collectors[c].t = &sample_collectors[c];
      sample_collectors[c].raw_counts = raw_counts[is_single_group ? 1 : 0];
      sample_collectors[c].lz77_counts = lz77_counts[is_single_group ? 1 : 0];
    }
    ProcessImageAreaPalette<
        ChannelRowProcessor<ChunkSampleCollector<UpTo8Bits>, UpTo8Bits>>(
        rgba, x0, y0, xs, 1, 1 + row_count, row_stride, lookup, nb_chans,
        row_sample_collectors);
  } else {
    ChunkSampleCollector<BitDepth> sample_collectors[4];
    ChannelRowProcessor<ChunkSampleCollector<BitDepth>, BitDepth>
        row_sample_collectors[4];
    for (size_t c = 0; c < nb_chans; c++) {
      row_sample_collectors[c].t = &sample_collectors[c];
      sample_collectors[c].raw_counts = raw_counts[c];
      sample_collectors[c].lz77_counts = lz77_counts[c];
    }
    ProcessImageArea<
        ChannelRowProcessor<ChunkSampleCollector<BitDepth>, BitDepth>>(
        rgba, x0, y0, xs, 1, 1 + row_count, row_stride, bitdepth, nb_chans,
        big_endian, row_sample_collectors);
  }
}

bool PrepareDCGlobalPalette(bool is_single_group, size_t width, size_t height,
                            size_t max_encoded_bits_per_sample, size_t nb_chans,
                            const PrefixCode code[4],
                            const std::vector<uint32_t>& palette,
                            size_t pcolors, BitWriter* output) {
  if (!PrepareDCGlobalCommon(is_single_group, width, height,
                             max_encoded_bits_per_sample, code, output)) {
    return false;
  }
  output->Write(2, 0b01);     // 1 transform
  output->Write(2, 0b01);     // Palette
  output->Write(5, 0b00000);  // Starting from ch 0
  if (nb_chans == 1) {
    output->Write(2, 0b00);  // 1-channel palette (Gray)
  } else if (nb_chans == 3) {
    output->Write(2, 0b01);  // 3-channel palette (RGB)
  } else if (nb_chans == 4) {
    output->Write(2, 0b10);  // 4-channel palette (RGBA)
  } else {
    output->Write(2, 0b11);
    output->Write(13, nb_chans - 1);
  }
  // pcolors <= kMaxColors + kChunkSize - 1
  static_assert(kMaxColors + kChunkSize < 1281,
                "add code to signal larger palette sizes");
  if (pcolors < 256) {
    output->Write(2, 0b00);
    output->Write(8, pcolors);
  } else {
    output->Write(2, 0b01);
    output->Write(10, pcolors - 256);
  }

  output->Write(2, 0b00);  // nb_deltas == 0
  output->Write(4, 0);     // Zero predictor for delta palette
  // Encode palette
  ChunkEncoder<UpTo8Bits> encoder;
  ChannelRowProcessor<ChunkEncoder<UpTo8Bits>, UpTo8Bits> row_encoder;
  row_encoder.t = &encoder;
  encoder.output = output;
  encoder.code = &code[0];
  encoder.PrepareForSimd();
  std::vector<std::array<int16_t, 32 + 1024>> p(4);
  size_t i = 0;
  size_t have_zero = 1;
  for (; i < pcolors; i++) {
    p[0][16 + i + have_zero] = palette[i] & 0xFF;
    p[1][16 + i + have_zero] = (palette[i] >> 8) & 0xFF;
    p[2][16 + i + have_zero] = (palette[i] >> 16) & 0xFF;
    p[3][16 + i + have_zero] = (palette[i] >> 24) & 0xFF;
  }
  p[0][15] = 0;
  row_encoder.ProcessRow(p[0].data() + 16, p[0].data() + 15, p[0].data() + 15,
                         p[0].data() + 15, pcolors);
  p[1][15] = p[0][16];
  p[0][15] = p[0][16];
  if (nb_chans > 1) {
    row_encoder.ProcessRow(p[1].data() + 16, p[1].data() + 15, p[0].data() + 16,
                           p[0].data() + 15, pcolors);
  }
  p[2][15] = p[1][16];
  p[1][15] = p[1][16];
  if (nb_chans > 2) {
    row_encoder.ProcessRow(p[2].data() + 16, p[2].data() + 15, p[1].data() + 16,
                           p[1].data() + 15, pcolors);
  }
  p[3][15] = p[2][16];
  p[2][15] = p[2][16];
  if (nb_chans > 3) {
    row_encoder.ProcessRow(p[3].data() + 16, p[3].data() + 15, p[2].data() + 16,
                           p[2].data() + 15, pcolors);
  }
  row_encoder.Finalize();

  if (!is_single_group) {
    output->ZeroPadToByte();
  }
  return true;
}

template <size_t nb_chans>
bool detect_palette(const unsigned char* r, size_t width,
                    std::vector<uint32_t>& palette) {
  size_t x = 0;
  bool collided = false;
  // this is just an unrolling of the next loop
  size_t look_ahead = 7 + ((nb_chans == 1) ? 3 : ((nb_chans < 4) ? 1 : 0));
  for (; x + look_ahead < width; x += 8) {
    uint32_t p[8] = {}, index[8];
    for (int i = 0; i < 8; i++) {
      for (int j = 0; j < 4; ++j) {
        p[i] |= r[(x + i) * nb_chans + j] << (8 * j);
      }
    }
    for (int i = 0; i < 8; i++) p[i] &= ((1llu << (8 * nb_chans)) - 1);
    for (int i = 0; i < 8; i++) index[i] = pixel_hash(p[i]);
    for (int i = 0; i < 8; i++) {
      collided |= (palette[index[i]] != 0 && p[i] != palette[index[i]]);
      palette[index[i]] = p[i];
    }
  }
  for (; x < width; x++) {
    uint32_t p = 0;
    for (size_t i = 0; i < nb_chans; ++i) {
      p |= r[x * nb_chans + i] << (8 * i);
    }
    uint32_t index = pixel_hash(p);
    collided |= (palette[index] != 0 && p != palette[index]);
    palette[index] = p;
  }
  return collided;
}

template <typename BitDepth>
JxlFastLosslessFrameState* LLPrepare(JxlChunkedFrameInputSource input,
                                     size_t width, size_t height,
                                     BitDepth bitdepth, size_t nb_chans,
                                     bool big_endian, int effort, int oneshot) {
  assert(width != 0);
  assert(height != 0);

  // Count colors to try palette
  std::vector<uint32_t> palette(kHashSize);
  std::vector<int16_t> lookup(kHashSize);
  lookup[0] = 0;
  int pcolors = 0;
  bool collided = effort < 2 || bitdepth.bitdepth != 8 || !oneshot;
  for (size_t y0 = 0; y0 < height && !collided; y0 += 256) {
    size_t ys = std::min<size_t>(height - y0, 256);
    for (size_t x0 = 0; x0 < width && !collided; x0 += 256) {
      size_t xs = std::min<size_t>(width - x0, 256);
      size_t stride;
      // TODO(szabadka): Add RAII wrapper around this.
      const void* buffer = input.get_color_channel_data_at(input.opaque, x0, y0,
                                                           xs, ys, &stride);
      if (buffer == nullptr) return nullptr;
      auto rgba = reinterpret_cast<const unsigned char*>(buffer);
      for (size_t y = 0; y < ys && !collided; y++) {
        const unsigned char* r = rgba + stride * y;
        if (nb_chans == 1) collided = detect_palette<1>(r, xs, palette);
        if (nb_chans == 2) collided = detect_palette<2>(r, xs, palette);
        if (nb_chans == 3) collided = detect_palette<3>(r, xs, palette);
        if (nb_chans == 4) collided = detect_palette<4>(r, xs, palette);
      }
      input.release_buffer(input.opaque, buffer);
    }
  }
  int nb_entries = 0;
  if (!collided) {
    pcolors = 1;  // always have all-zero as a palette color
    bool have_color = false;
    uint8_t minG = 255, maxG = 0;
    for (uint32_t k = 0; k < kHashSize; k++) {
      if (palette[k] == 0) continue;
      uint8_t p[4];
      for (int i = 0; i < 4; ++i) {
        p[i] = (palette[k] >> (8 * i)) & 0xFF;
      }
      // move entries to front so sort has less work
      palette[nb_entries] = palette[k];
      if (p[0] != p[1] || p[0] != p[2]) have_color = true;
      if (p[1] < minG) minG = p[1];
      if (p[1] > maxG) maxG = p[1];
      nb_entries++;
      // don't do palette if too many colors are needed
      if (nb_entries + pcolors > kMaxColors) {
        collided = true;
        break;
      }
    }
    if (!have_color) {
      // don't do palette if it's just grayscale without many holes
      if (maxG - minG < nb_entries * 1.4f) collided = true;
    }
  }
  if (!collided) {
    std::sort(
        palette.begin(), palette.begin() + nb_entries,
        [&nb_chans](uint32_t ap, uint32_t bp) {
          if (ap == 0) return false;
          if (bp == 0) return true;
          uint8_t a[4], b[4];
          for (int i = 0; i < 4; ++i) {
            a[i] = (ap >> (8 * i)) & 0xFF;
            b[i] = (bp >> (8 * i)) & 0xFF;
          }
          float ay, by;
          if (nb_chans == 4) {
            ay = (0.299f * a[0] + 0.587f * a[1] + 0.114f * a[2] + 0.01f) * a[3];
            by = (0.299f * b[0] + 0.587f * b[1] + 0.114f * b[2] + 0.01f) * b[3];
          } else {
            ay = (0.299f * a[0] + 0.587f * a[1] + 0.114f * a[2] + 0.01f);
            by = (0.299f * b[0] + 0.587f * b[1] + 0.114f * b[2] + 0.01f);
          }
          return ay < by;  // sort on alpha*luma
        });
    for (int k = 0; k < nb_entries; k++) {
      if (palette[k] == 0) break;
      lookup[pixel_hash(palette[k])] = pcolors++;
    }
  }

  size_t num_groups_x = (width + 255) / 256;
  size_t num_groups_y = (height + 255) / 256;
  size_t num_dc_groups_x = (width + 2047) / 2048;
  size_t num_dc_groups_y = (height + 2047) / 2048;

  uint64_t raw_counts[4][kNumRawSymbols] = {};
  uint64_t lz77_counts[4][kNumLZ77] = {};

  bool onegroup = num_groups_x == 1 && num_groups_y == 1;

  auto sample_rows = [&](size_t xg, size_t yg, size_t num_rows) {
    size_t y0 = yg * 256;
    size_t x0 = xg * 256;
    size_t ys = std::min<size_t>(height - y0, 256);
    size_t xs = std::min<size_t>(width - x0, 256);
    size_t stride;
    const void* buffer =
        input.get_color_channel_data_at(input.opaque, x0, y0, xs, ys, &stride);
    if (buffer == nullptr) {
      return false;
    }
    auto rgba = reinterpret_cast<const unsigned char*>(buffer);
    int y_begin_group =
        std::max<ptrdiff_t>(
            0, static_cast<ptrdiff_t>(ys) - static_cast<ptrdiff_t>(num_rows)) /
        2;
    int y_count =
        std::max<int>(0, std::min<int>(num_rows, ys - y_begin_group - 1));
    int x_max = xs / kChunkSize * kChunkSize;
    CollectSamples(rgba, 0, y_begin_group, x_max, stride, y_count, raw_counts,
                   lz77_counts, onegroup, !collided, bitdepth, nb_chans,
                   big_endian, lookup.data());
    input.release_buffer(input.opaque, buffer);
    return true;
  };

  // TODO(veluca): that `64` is an arbitrary constant, meant to correspond to
  // the point where the number of processed rows is large enough that loading
  // the entire image is cost-effective.
  if (oneshot || effort >= 64) {
    for (size_t g = 0; g < num_groups_y * num_groups_x; g++) {
      size_t xg = g % num_groups_x;
      size_t yg = g / num_groups_x;
      size_t y0 = yg * 256;
      size_t ys = std::min<size_t>(height - y0, 256);
      size_t num_rows = 2 * effort * ys / 256;
      if (!sample_rows(xg, yg, num_rows)) {
        return nullptr;
      }
    }
  } else {
    // sample the middle (effort * 2 * num_groups) rows of the center group
    // (possibly all of them).
    if (!sample_rows((num_groups_x - 1) / 2, (num_groups_y - 1) / 2,
                     2 * effort * num_groups_x * num_groups_y)) {
      return nullptr;
    }
  }

  // TODO(veluca): can probably improve this and make it bitdepth-dependent.
  uint64_t base_raw_counts[kNumRawSymbols] = {
      3843, 852, 1270, 1214, 1014, 727, 481, 300, 159, 51,
      5,    1,   1,    1,    1,    1,   1,   1,   1};

  bool doing_ycocg = nb_chans > 2 && collided;
  bool large_palette = !collided || pcolors >= 256;
  for (size_t i = bitdepth.NumSymbols(doing_ycocg || large_palette);
       i < kNumRawSymbols; i++) {
    base_raw_counts[i] = 0;
  }

  for (size_t c = 0; c < 4; c++) {
    for (size_t i = 0; i < kNumRawSymbols; i++) {
      raw_counts[c][i] = (raw_counts[c][i] << 8) + base_raw_counts[i];
    }
  }

  if (!collided) {
    unsigned token, nbits, bits;
    EncodeHybridUint000(PackSigned(pcolors - 1), &token, &nbits, &bits);
    // ensure all palette indices can actually be encoded
    for (size_t i = 0; i < token + 1; i++)
      raw_counts[0][i] = std::max<uint64_t>(raw_counts[0][i], 1);
    // these tokens are only used for the palette itself so they can get a bad
    // code
    for (size_t i = token + 1; i < 10; i++) raw_counts[0][i] = 1;
  }

  uint64_t base_lz77_counts[kNumLZ77] = {
      29, 27, 25,  23, 21, 21, 19, 18, 21, 17, 16, 15, 15, 14,
      13, 13, 137, 98, 61, 34, 1,  1,  1,  1,  1,  1,  1,  1,
  };

  for (size_t c = 0; c < 4; c++) {
    for (size_t i = 0; i < kNumLZ77; i++) {
      lz77_counts[c][i] = (lz77_counts[c][i] << 8) + base_lz77_counts[i];
    }
  }

  JxlFastLosslessFrameState* frame_state = new JxlFastLosslessFrameState();
  if (!frame_state) return nullptr;
  for (size_t i = 0; i < 4; i++) {
    frame_state->hcode[i] = PrefixCode(bitdepth, raw_counts[i], lz77_counts[i]);
  }

  size_t num_dc_groups = num_dc_groups_x * num_dc_groups_y;
  size_t num_ac_groups = num_groups_x * num_groups_y;
  size_t num_groups = onegroup ? 1 : (2 + num_dc_groups + num_ac_groups);
  frame_state->input = input;
  frame_state->width = width;
  frame_state->height = height;
  frame_state->num_groups_x = num_groups_x;
  frame_state->num_groups_y = num_groups_y;
  frame_state->num_dc_groups_x = num_dc_groups_x;
  frame_state->num_dc_groups_y = num_dc_groups_y;
  frame_state->nb_chans = nb_chans;
  frame_state->bitdepth = bitdepth.bitdepth;
  frame_state->big_endian = big_endian;
  frame_state->effort = effort;
  frame_state->collided = collided;
  frame_state->lookup = lookup;

  frame_state->group_data = std::vector<std::array<BitWriter, 4>>(num_groups);
  frame_state->group_sizes.resize(num_groups);
  if (collided) {
    if (!PrepareDCGlobal(onegroup, width, height,
                         bitdepth.MaxEncodedBitsPerSample(), nb_chans,
                         frame_state->hcode, &frame_state->group_data[0][0])) {
      delete frame_state;
      return nullptr;
    }
  } else {
    if (!PrepareDCGlobalPalette(onegroup, width, height,
                                bitdepth.MaxEncodedBitsPerSample(), nb_chans,
                                frame_state->hcode, palette, pcolors,
                                &frame_state->group_data[0][0])) {
      delete frame_state;
      return nullptr;
    }
  }
  frame_state->group_sizes[0] = SectionSize(frame_state->group_data[0]);
  if (!onegroup) {
    ComputeAcGroupDataOffset(frame_state->group_sizes[0], num_dc_groups,
                             num_ac_groups, frame_state->min_dc_global_size,
                             frame_state->ac_group_data_offset);
  }

  return frame_state;
}

template <typename BitDepth>
jxl::Status LLProcess(JxlFastLosslessFrameState* frame_state, bool is_last,
                      BitDepth bitdepth, void* runner_opaque,
                      FJxlParallelRunner runner,
                      JxlEncoderOutputProcessorWrapper* output_processor) {
#if !FJXL_STANDALONE
  if (frame_state->process_done) {
    if (!JxlFastLosslessPrepareHeader(frame_state, /*add_image_header=*/0,
                                      is_last)) {
      return JXL_FAILURE("Allocation failed");
    };
    if (output_processor) {
      JXL_RETURN_IF_ERROR(
          JxlFastLosslessOutputFrame(frame_state, output_processor));
    }
    return true;
  }
#endif
  // The maximum number of groups that we process concurrently here.
  // TODO(szabadka) Use the number of threads or some outside parameter for the
  // maximum memory usage instead.
  constexpr size_t kMaxLocalGroups = 16;
  bool onegroup = frame_state->group_sizes.size() == 1;
  bool streaming = !onegroup && output_processor;
  size_t total_groups = frame_state->num_groups_x * frame_state->num_groups_y;
  size_t max_groups = streaming ? kMaxLocalGroups : total_groups;
#if !FJXL_STANDALONE
  size_t start_pos = 0;
  if (streaming) {
    start_pos = output_processor->CurrentPosition();
    JXL_RETURN_IF_ERROR(
        output_processor->Seek(start_pos + frame_state->ac_group_data_offset));
  }
#endif
  for (size_t offset = 0; offset < total_groups; offset += max_groups) {
    size_t num_groups = std::min(max_groups, total_groups - offset);
    JxlFastLosslessFrameState local_frame_state;
    if (streaming) {
      local_frame_state.group_data =
          std::vector<std::array<BitWriter, 4>>(num_groups);
    }
    std::atomic<uint32_t> has_error{0};
    auto run_one = [&](size_t i) {
      size_t g = offset + i;
      size_t xg = g % frame_state->num_groups_x;
      size_t yg = g / frame_state->num_groups_x;
      size_t num_dc_groups =
          frame_state->num_dc_groups_x * frame_state->num_dc_groups_y;
      size_t group_id = onegroup ? 0 : (2 + num_dc_groups + g);
      size_t xs = std::min<size_t>(frame_state->width - xg * 256, 256);
      size_t ys = std::min<size_t>(frame_state->height - yg * 256, 256);
      size_t x0 = xg * 256;
      size_t y0 = yg * 256;
      size_t stride;
      JxlChunkedFrameInputSource input = frame_state->input;
      const void* buffer = input.get_color_channel_data_at(input.opaque, x0, y0,
                                                           xs, ys, &stride);
      if (buffer == nullptr) {
        has_error = 1;
        return;
      }
      const unsigned char* rgba =
          reinterpret_cast<const unsigned char*>(buffer);

      auto& gd = streaming ? local_frame_state.group_data[i]
                           : frame_state->group_data[group_id];
      bool ok;
      if (frame_state->collided) {
        ok = WriteACSection(rgba, 0, 0, xs, ys, stride, onegroup, bitdepth,
                            frame_state->nb_chans, frame_state->big_endian,
                            frame_state->hcode, gd);
      } else {
        ok = WriteACSectionPalette(
            rgba, 0, 0, xs, ys, stride, onegroup, frame_state->hcode,
            frame_state->lookup.data(), frame_state->nb_chans, gd[0]);
      }
      if (ok) {
        frame_state->group_sizes[group_id] = SectionSize(gd);
      } else {
        has_error = 1;
      }
      input.release_buffer(input.opaque, buffer);
    };
    runner(
        runner_opaque, &run_one,
        +[](void* r, size_t i) {
          (*reinterpret_cast<decltype(&run_one)>(r))(i);
        },
        num_groups);
    if (has_error) return JXL_FAILURE("Allocation failed");
#if !FJXL_STANDALONE
    if (streaming) {
      local_frame_state.nb_chans = frame_state->nb_chans;
      local_frame_state.current_bit_writer = 1;
      JXL_RETURN_IF_ERROR(
          JxlFastLosslessOutputFrame(&local_frame_state, output_processor));
    }
#endif
  }
#if !FJXL_STANDALONE
  if (streaming) {
    size_t end_pos = output_processor->CurrentPosition();
    JXL_RETURN_IF_ERROR(output_processor->Seek(start_pos));
    frame_state->group_data.resize(1);
    bool have_alpha = frame_state->nb_chans == 2 || frame_state->nb_chans == 4;
    size_t padding = ComputeDcGlobalPadding(
        frame_state->group_sizes, frame_state->ac_group_data_offset,
        frame_state->min_dc_global_size, have_alpha, is_last);

    for (size_t i = 0; i < padding; ++i) {
      frame_state->group_data[0][0].Write(8, 0);
    }
    frame_state->group_sizes[0] += padding;
    if (!JxlFastLosslessPrepareHeader(frame_state, /*add_image_header=*/0,
                                      is_last)) {
      return JXL_FAILURE("Allocation failed");
    }
    assert(frame_state->ac_group_data_offset ==
           JxlFastLosslessOutputSize(frame_state));
    JXL_RETURN_IF_ERROR(
        JxlFastLosslessOutputHeaders(frame_state, output_processor));
    JXL_RETURN_IF_ERROR(output_processor->Seek(end_pos));
  } else if (output_processor) {
    assert(onegroup);
    if (!JxlFastLosslessPrepareHeader(frame_state, /*add_image_header=*/0,
                                      is_last)) {
      return JXL_FAILURE("Allocation failed");
    }
    if (output_processor) {
      JXL_RETURN_IF_ERROR(
          JxlFastLosslessOutputFrame(frame_state, output_processor));
    }
  }
  frame_state->process_done = true;
#endif
  return true;
}

JxlFastLosslessFrameState* JxlFastLosslessPrepareImpl(
    JxlChunkedFrameInputSource input, size_t width, size_t height,
    size_t nb_chans, size_t bitdepth, bool big_endian, int effort,
    int oneshot) {
  assert(bitdepth > 0);
  assert(nb_chans <= 4);
  assert(nb_chans != 0);
  if (bitdepth <= 8) {
    return LLPrepare(input, width, height, UpTo8Bits(bitdepth), nb_chans,
                     big_endian, effort, oneshot);
  }
  if (bitdepth <= 13) {
    return LLPrepare(input, width, height, From9To13Bits(bitdepth), nb_chans,
                     big_endian, effort, oneshot);
  }
  if (bitdepth == 14) {
    return LLPrepare(input, width, height, Exactly14Bits(bitdepth), nb_chans,
                     big_endian, effort, oneshot);
  }
  return LLPrepare(input, width, height, MoreThan14Bits(bitdepth), nb_chans,
                   big_endian, effort, oneshot);
}

jxl::Status JxlFastLosslessProcessFrameImpl(
    JxlFastLosslessFrameState* frame_state, bool is_last, void* runner_opaque,
    FJxlParallelRunner runner,
    JxlEncoderOutputProcessorWrapper* output_processor) {
  const size_t bitdepth = frame_state->bitdepth;
  if (bitdepth <= 8) {
    JXL_RETURN_IF_ERROR(LLProcess(frame_state, is_last, UpTo8Bits(bitdepth),
                                  runner_opaque, runner, output_processor));
  } else if (bitdepth <= 13) {
    JXL_RETURN_IF_ERROR(LLProcess(frame_state, is_last, From9To13Bits(bitdepth),
                                  runner_opaque, runner, output_processor));
  } else if (bitdepth == 14) {
    JXL_RETURN_IF_ERROR(LLProcess(frame_state, is_last, Exactly14Bits(bitdepth),
                                  runner_opaque, runner, output_processor));
  } else {
    JXL_RETURN_IF_ERROR(LLProcess(frame_state, is_last,
                                  MoreThan14Bits(bitdepth), runner_opaque,
                                  runner, output_processor));
  }
  return true;
}

}  // namespace

#endif  // FJXL_SELF_INCLUDE

#ifndef FJXL_SELF_INCLUDE

#define FJXL_SELF_INCLUDE

// If we have NEON enabled, it is the default target.
#if FJXL_ENABLE_NEON

namespace default_implementation {
#define FJXL_NEON
#include "lib/jxl/enc_fast_lossless.cc"
#undef FJXL_NEON
}  // namespace default_implementation

#else                                    // FJXL_ENABLE_NEON

namespace default_implementation {
#include "lib/jxl/enc_fast_lossless.cc"  // NOLINT
}

#if FJXL_ENABLE_AVX2
#ifdef __clang__
#pragma clang attribute push(__attribute__((target("avx,avx2"))), \
                             apply_to = function)
// Causes spurious warnings on clang5.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-braces"
#elif defined(__GNUC__)
#pragma GCC push_options
// Seems to cause spurious errors on GCC8.
#pragma GCC diagnostic ignored "-Wpsabi"
#pragma GCC target "avx,avx2"
#endif

namespace AVX2 {
#define FJXL_AVX2
#include "lib/jxl/enc_fast_lossless.cc"  // NOLINT
#undef FJXL_AVX2
}  // namespace AVX2

#ifdef __clang__
#pragma clang attribute pop
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif  // FJXL_ENABLE_AVX2

#if FJXL_ENABLE_AVX512
#ifdef __clang__
#pragma clang attribute push(                                                 \
    __attribute__((target("avx512cd,avx512bw,avx512vl,avx512f,avx512vbmi"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target "avx512cd,avx512bw,avx512vl,avx512f,avx512vbmi"
#endif

namespace AVX512 {
#define FJXL_AVX512
#include "lib/jxl/enc_fast_lossless.cc"
#undef FJXL_AVX512
}  // namespace AVX512

#ifdef __clang__
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif  // FJXL_ENABLE_AVX512

#endif

extern "C" {

#if FJXL_STANDALONE
class FJxlFrameInput {
 public:
  FJxlFrameInput(const unsigned char* rgba, size_t row_stride, size_t nb_chans,
                 size_t bitdepth)
      : rgba_(rgba),
        row_stride_(row_stride),
        bytes_per_pixel_(bitdepth <= 8 ? nb_chans : 2 * nb_chans) {}

  JxlChunkedFrameInputSource GetInputSource() {
    return JxlChunkedFrameInputSource{
        this, GetColorChannelDataAt,
        /*release_buffer=*/[](void*, const void*) {}};
  }

 private:
  static const void* GetColorChannelDataAt(void* opaque, size_t xpos,
                                           size_t ypos, size_t xsize,
                                           size_t ysize, size_t* row_offset) {
    FJxlFrameInput* self = static_cast<FJxlFrameInput*>(opaque);
    *row_offset = self->row_stride_;
    return self->rgba_ + ypos * (*row_offset) + xpos * self->bytes_per_pixel_;
  }

  const uint8_t* rgba_;
  size_t row_stride_;
  size_t bytes_per_pixel_;
};

size_t JxlFastLosslessEncode(const unsigned char* rgba, size_t width,
                             size_t row_stride, size_t height, size_t nb_chans,
                             size_t bitdepth, bool big_endian, int effort,
                             unsigned char** output, void* runner_opaque,
                             FJxlParallelRunner runner) {
  FJxlFrameInput input(rgba, row_stride, nb_chans, bitdepth);
  auto* frame_state = JxlFastLosslessPrepareFrame(
      input.GetInputSource(), width, height, nb_chans, bitdepth, big_endian,
      effort, /*oneshot=*/true);
  if (!frame_state) return 0;
  if (!JxlFastLosslessProcessFrame(frame_state, /*is_last=*/true, runner_opaque,
                                   runner, nullptr)) {
    JxlFastLosslessFreeFrameState(frame_state);
    return 0;
  }
  if (!JxlFastLosslessPrepareHeader(frame_state, /*add_image_header=*/1,
                                    /*is_last=*/1)) {
    JxlFastLosslessFreeFrameState(frame_state);
    return 0;
  }
  size_t output_size = JxlFastLosslessMaxRequiredOutput(frame_state);
  *output = (unsigned char*)malloc(output_size);
  if (*output == NULL) {
    return JXL_FAILURE("Memory allocation failed");
  }
  size_t written = 0;
  size_t total = 0;
  while ((written = JxlFastLosslessWriteOutput(frame_state, *output + total,
                                               output_size - total)) != 0) {
    total += written;
  }
  JxlFastLosslessFreeFrameState(frame_state);
  return total;
}
#endif

JxlFastLosslessFrameState* JxlFastLosslessPrepareFrame(
    JxlChunkedFrameInputSource input, size_t width, size_t height,
    size_t nb_chans, size_t bitdepth, bool big_endian, int effort,
    int oneshot) {
#if FJXL_ENABLE_AVX512
  if (HasCpuFeature(CpuFeature::kAVX512CD) &&
      HasCpuFeature(CpuFeature::kVBMI) &&
      HasCpuFeature(CpuFeature::kAVX512BW) &&
      HasCpuFeature(CpuFeature::kAVX512F) &&
      HasCpuFeature(CpuFeature::kAVX512VL)) {
    return AVX512::JxlFastLosslessPrepareImpl(
        input, width, height, nb_chans, bitdepth, big_endian, effort, oneshot);
  }
#endif
#if FJXL_ENABLE_AVX2
  if (HasCpuFeature(CpuFeature::kAVX2)) {
    return AVX2::JxlFastLosslessPrepareImpl(
        input, width, height, nb_chans, bitdepth, big_endian, effort, oneshot);
  }
#endif

  return default_implementation::JxlFastLosslessPrepareImpl(
      input, width, height, nb_chans, bitdepth, big_endian, effort, oneshot);
}

bool JxlFastLosslessProcessFrame(
    JxlFastLosslessFrameState* frame_state, bool is_last, void* runner_opaque,
    FJxlParallelRunner runner,
    JxlEncoderOutputProcessorWrapper* output_processor) {
  auto trivial_runner =
      +[](void*, void* opaque, void fun(void*, size_t), size_t count) {
        for (size_t i = 0; i < count; i++) {
          fun(opaque, i);
        }
      };

  if (runner == nullptr) {
    runner = trivial_runner;
  }

#if FJXL_ENABLE_AVX512
  if (HasCpuFeature(CpuFeature::kAVX512CD) &&
      HasCpuFeature(CpuFeature::kVBMI) &&
      HasCpuFeature(CpuFeature::kAVX512BW) &&
      HasCpuFeature(CpuFeature::kAVX512F) &&
      HasCpuFeature(CpuFeature::kAVX512VL)) {
    JXL_RETURN_IF_ERROR(AVX512::JxlFastLosslessProcessFrameImpl(
        frame_state, is_last, runner_opaque, runner, output_processor));
    return true;
  }
#endif
#if FJXL_ENABLE_AVX2
  if (HasCpuFeature(CpuFeature::kAVX2)) {
    JXL_RETURN_IF_ERROR(AVX2::JxlFastLosslessProcessFrameImpl(
        frame_state, is_last, runner_opaque, runner, output_processor));
    return true;
  }
#endif

  JXL_RETURN_IF_ERROR(default_implementation::JxlFastLosslessProcessFrameImpl(
      frame_state, is_last, runner_opaque, runner, output_processor));
  return true;
}

}  // extern "C"

#if !FJXL_STANDALONE
bool JxlFastLosslessOutputFrame(
    JxlFastLosslessFrameState* frame_state,
    JxlEncoderOutputProcessorWrapper* output_processor) {
  size_t fl_size = JxlFastLosslessOutputSize(frame_state);
  size_t written = 0;
  while (written < fl_size) {
    JXL_ASSIGN_OR_RETURN(auto buffer,
                         output_processor->GetBuffer(32, fl_size - written));
    size_t n =
        JxlFastLosslessWriteOutput(frame_state, buffer.data(), buffer.size());
    if (n == 0) break;
    JXL_RETURN_IF_ERROR(buffer.advance(n));
    written += n;
  };
  return true;
}
#endif

#endif  // FJXL_SELF_INCLUDE

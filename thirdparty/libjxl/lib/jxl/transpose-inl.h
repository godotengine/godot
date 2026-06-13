// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Block transpose for DCT/IDCT

#if defined(LIB_JXL_TRANSPOSE_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef LIB_JXL_TRANSPOSE_INL_H_
#undef LIB_JXL_TRANSPOSE_INL_H_
#else
#define LIB_JXL_TRANSPOSE_INL_H_
#endif

#include <stddef.h>

#include <hwy/highway.h>
#include <type_traits>

#include "lib/jxl/base/status.h"
#include "lib/jxl/dct_block-inl.h"

HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {
namespace {

#ifndef JXL_INLINE_TRANSPOSE
// Workaround for issue #42 - (excessive?) inlining causes invalid codegen.
#if defined(__arm__)
#define JXL_INLINE_TRANSPOSE HWY_NOINLINE
#else
#define JXL_INLINE_TRANSPOSE HWY_INLINE
#endif
#endif  // JXL_INLINE_TRANSPOSE

// Simple wrapper that ensures that a function will not be inlined.
template <typename T, typename... Args>
JXL_NOINLINE void NoInlineWrapper(const T& f, const Args&... args) {
  return f(args...);
}

template <bool enabled>
struct TransposeSimdTag {};

// TODO(veluca): it's not super useful to have this in the SIMD namespace.
template <size_t ROWS_or_0, size_t COLS_or_0, class From, class To>
JXL_INLINE_TRANSPOSE void GenericTransposeBlock(
    TransposeSimdTag<false> /* tag */, const From& from, const To& to,
    size_t ROWSp, size_t COLSp) {
  size_t ROWS = ROWS_or_0 == 0 ? ROWSp : ROWS_or_0;
  size_t COLS = COLS_or_0 == 0 ? COLSp : COLS_or_0;
  for (size_t n = 0; n < ROWS; ++n) {
    for (size_t m = 0; m < COLS; ++m) {
      to.Write(from.Read(n, m), m, n);
    }
  }
}

// TODO(veluca): AVX3?
#if HWY_CAP_GE256
constexpr bool TransposeUseSimd(size_t ROWS, size_t COLS) {
  return ROWS % 8 == 0 && COLS % 8 == 0;
}

template <size_t ROWS_or_0, size_t COLS_or_0, class From, class To>
JXL_INLINE_TRANSPOSE void GenericTransposeBlock(
    TransposeSimdTag<true> /* tag */, const From& from, const To& to,
    size_t ROWSp, size_t COLSp) {
  size_t ROWS = ROWS_or_0 == 0 ? ROWSp : ROWS_or_0;
  size_t COLS = COLS_or_0 == 0 ? COLSp : COLS_or_0;
  static_assert(MaxLanes(BlockDesc<8>()) == 8, "Invalid descriptor size");
  static_assert(ROWS_or_0 % 8 == 0, "Invalid number of rows");
  static_assert(COLS_or_0 % 8 == 0, "Invalid number of columns");
  for (size_t n = 0; n < ROWS; n += 8) {
    for (size_t m = 0; m < COLS; m += 8) {
      const BlockDesc<8> d;
      auto i0 = from.LoadPart(d, n + 0, m + 0);
      auto i1 = from.LoadPart(d, n + 1, m + 0);
      auto i2 = from.LoadPart(d, n + 2, m + 0);
      auto i3 = from.LoadPart(d, n + 3, m + 0);
      auto i4 = from.LoadPart(d, n + 4, m + 0);
      auto i5 = from.LoadPart(d, n + 5, m + 0);
      auto i6 = from.LoadPart(d, n + 6, m + 0);
      auto i7 = from.LoadPart(d, n + 7, m + 0);
      // Surprisingly, this straightforward implementation (24 cycles on port5)
      // is faster than load128+insert and LoadDup128+ConcatUpperLower+blend.
      const auto q0 = InterleaveLower(d, i0, i2);
      const auto q1 = InterleaveLower(d, i1, i3);
      const auto q2 = InterleaveUpper(d, i0, i2);
      const auto q3 = InterleaveUpper(d, i1, i3);
      const auto q4 = InterleaveLower(d, i4, i6);
      const auto q5 = InterleaveLower(d, i5, i7);
      const auto q6 = InterleaveUpper(d, i4, i6);
      const auto q7 = InterleaveUpper(d, i5, i7);

      const auto r0 = InterleaveLower(d, q0, q1);
      const auto r1 = InterleaveUpper(d, q0, q1);
      const auto r2 = InterleaveLower(d, q2, q3);
      const auto r3 = InterleaveUpper(d, q2, q3);
      const auto r4 = InterleaveLower(d, q4, q5);
      const auto r5 = InterleaveUpper(d, q4, q5);
      const auto r6 = InterleaveLower(d, q6, q7);
      const auto r7 = InterleaveUpper(d, q6, q7);

      i0 = ConcatLowerLower(d, r4, r0);
      i1 = ConcatLowerLower(d, r5, r1);
      i2 = ConcatLowerLower(d, r6, r2);
      i3 = ConcatLowerLower(d, r7, r3);
      i4 = ConcatUpperUpper(d, r4, r0);
      i5 = ConcatUpperUpper(d, r5, r1);
      i6 = ConcatUpperUpper(d, r6, r2);
      i7 = ConcatUpperUpper(d, r7, r3);
      to.StorePart(d, i0, m + 0, n + 0);
      to.StorePart(d, i1, m + 1, n + 0);
      to.StorePart(d, i2, m + 2, n + 0);
      to.StorePart(d, i3, m + 3, n + 0);
      to.StorePart(d, i4, m + 4, n + 0);
      to.StorePart(d, i5, m + 5, n + 0);
      to.StorePart(d, i6, m + 6, n + 0);
      to.StorePart(d, i7, m + 7, n + 0);
    }
  }
}
#elif HWY_TARGET != HWY_SCALAR
constexpr bool TransposeUseSimd(size_t ROWS, size_t COLS) {
  return ROWS % 4 == 0 && COLS % 4 == 0;
}

template <size_t ROWS_or_0, size_t COLS_or_0, class From, class To>
JXL_INLINE_TRANSPOSE void GenericTransposeBlock(
    TransposeSimdTag<true> /* tag */, const From& from, const To& to,
    size_t ROWSp, size_t COLSp) {
  size_t ROWS = ROWS_or_0 == 0 ? ROWSp : ROWS_or_0;
  size_t COLS = COLS_or_0 == 0 ? COLSp : COLS_or_0;
  static_assert(MaxLanes(BlockDesc<4>()) == 4, "Invalid descriptor size");
  static_assert(ROWS_or_0 % 4 == 0, "Invalid number of rows");
  static_assert(COLS_or_0 % 4 == 0, "Invalid number of columns");
  for (size_t n = 0; n < ROWS; n += 4) {
    for (size_t m = 0; m < COLS; m += 4) {
      const BlockDesc<4> d;
      const auto p0 = from.LoadPart(d, n + 0, m + 0);
      const auto p1 = from.LoadPart(d, n + 1, m + 0);
      const auto p2 = from.LoadPart(d, n + 2, m + 0);
      const auto p3 = from.LoadPart(d, n + 3, m + 0);

      const auto q0 = InterleaveLower(d, p0, p2);
      const auto q1 = InterleaveLower(d, p1, p3);
      const auto q2 = InterleaveUpper(d, p0, p2);
      const auto q3 = InterleaveUpper(d, p1, p3);

      const auto r0 = InterleaveLower(d, q0, q1);
      const auto r1 = InterleaveUpper(d, q0, q1);
      const auto r2 = InterleaveLower(d, q2, q3);
      const auto r3 = InterleaveUpper(d, q2, q3);

      to.StorePart(d, r0, m + 0, n + 0);
      to.StorePart(d, r1, m + 1, n + 0);
      to.StorePart(d, r2, m + 2, n + 0);
      to.StorePart(d, r3, m + 3, n + 0);
    }
  }
}
#else
constexpr bool TransposeUseSimd(size_t ROWS, size_t COLS) { return false; }
#endif

template <size_t N, size_t M, typename = void>
struct Transpose {
  template <typename From, typename To>
  static void Run(const From& from, const To& to) {
    // This does not guarantee anything, just saves from the most stupid
    // mistakes.
    JXL_DASSERT(from.Address(0, 0) != to.Address(0, 0));
    TransposeSimdTag<TransposeUseSimd(N, M)> tag;
    GenericTransposeBlock<N, M>(tag, from, to, N, M);
  }
};

// Avoid inlining and unrolling transposes for large blocks.
template <size_t N, size_t M>
struct Transpose<
    N, M, typename std::enable_if<(N >= 8 && M >= 8 && N * M >= 512)>::type> {
  template <typename From, typename To>
  static void Run(const From& from, const To& to) {
    // This does not guarantee anything, just saves from the most stupid
    // mistakes.
    JXL_DASSERT(from.Address(0, 0) != to.Address(0, 0));
    TransposeSimdTag<TransposeUseSimd(N, M)> tag;
    constexpr void (*transpose)(TransposeSimdTag<TransposeUseSimd(N, M)>,
                                const From&, const To&, size_t, size_t) =
        GenericTransposeBlock<0, 0, From, To>;
    NoInlineWrapper(transpose, tag, from, to, N, M);
  }
};

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#endif  // LIB_JXL_TRANSPOSE_INL_H_

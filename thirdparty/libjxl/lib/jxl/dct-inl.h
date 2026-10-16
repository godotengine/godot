// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Fast SIMD floating-point (I)DCT, any power of two.

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/status.h"

#if defined(LIB_JXL_DCT_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef LIB_JXL_DCT_INL_H_
#undef LIB_JXL_DCT_INL_H_
#else
#define LIB_JXL_DCT_INL_H_
#endif

#include <cstddef>
#include <hwy/highway.h>

#include "lib/jxl/dct_block-inl.h"
#include "lib/jxl/dct_scales.h"
#include "lib/jxl/transpose-inl.h"
HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {
namespace {

// These templates are not found via ADL.
using hwy::HWY_NAMESPACE::Add;
using hwy::HWY_NAMESPACE::Mul;
using hwy::HWY_NAMESPACE::MulAdd;
using hwy::HWY_NAMESPACE::NegMulAdd;
using hwy::HWY_NAMESPACE::Sub;

#if !HWY_HAVE_SCALABLE
// OK to use MaxLanes for non-scalable; should be same as Lanes.
constexpr size_t kMaxLanes = MaxLanes(HWY_FULL(float)());
#else
#endif

// Implementation of Lowest Complexity Self Recursive Radix-2 DCT II/III
// Algorithms, by Sirani M. Perera and Jianhua Liu.

template <size_t N, size_t SZ>
struct CoeffBundle {
  using D = HWY_CAPPED(float, SZ);
  static void AddReverse(const float* JXL_RESTRICT a_in1,
                         const float* JXL_RESTRICT a_in2,
                         float* JXL_RESTRICT a_out) {
    const D d;
    for (size_t i = 0; i < N; i++) {
      auto in1 = Load(d, a_in1 + i * SZ);
      auto in2 = Load(d, a_in2 + (N - i - 1) * SZ);
      Store(Add(in1, in2), d, a_out + i * SZ);
    }
  }
  static void SubReverse(const float* JXL_RESTRICT a_in1,
                         const float* JXL_RESTRICT a_in2,
                         float* JXL_RESTRICT a_out) {
    const D d;
    for (size_t i = 0; i < N; i++) {
      auto in1 = Load(d, a_in1 + i * SZ);
      auto in2 = Load(d, a_in2 + (N - i - 1) * SZ);
      Store(Sub(in1, in2), d, a_out + i * SZ);
    }
  }
  static void B(float* JXL_RESTRICT coeff) {
    const D d;
    auto sqrt2 = Set(d, kSqrt2);
    auto in1_0 = Load(d, coeff);
    auto in2_0 = Load(d, coeff + SZ);
    Store(MulAdd(in1_0, sqrt2, in2_0), d, coeff);
    for (size_t i = 1; i + 1 < N; i++) {
      auto in1 = Load(d, coeff + i * SZ);
      auto in2 = Load(d, coeff + (i + 1) * SZ);
      Store(Add(in1, in2), d, coeff + i * SZ);
    }
  }
  static void BTranspose(float* JXL_RESTRICT coeff) {
    const D d;
    for (size_t i = N - 1; i > 0; i--) {
      auto in1 = Load(d, coeff + i * SZ);
      auto in2 = Load(d, coeff + (i - 1) * SZ);
      Store(Add(in1, in2), d, coeff + i * SZ);
    }
    auto sqrt2 = Set(d, kSqrt2);
    auto in1 = Load(d, coeff);
    Store(Mul(in1, sqrt2), d, coeff);
  }
  // Ideally optimized away by compiler (except the multiply).
  static void InverseEvenOdd(const float* JXL_RESTRICT a_in,
                             float* JXL_RESTRICT a_out) {
    const D d;
    for (size_t i = 0; i < N / 2; i++) {
      auto in1 = Load(d, a_in + i * SZ);
      Store(in1, d, a_out + 2 * i * SZ);
    }
    for (size_t i = N / 2; i < N; i++) {
      auto in1 = Load(d, a_in + i * SZ);
      Store(in1, d, a_out + (2 * (i - N / 2) + 1) * SZ);
    }
  }
  // Ideally optimized away by compiler.
  static void ForwardEvenOdd(const float* JXL_RESTRICT a_in, size_t a_in_stride,
                             float* JXL_RESTRICT a_out) {
    const D d;
    for (size_t i = 0; i < N / 2; i++) {
      auto in1 = LoadU(d, a_in + 2 * i * a_in_stride);
      Store(in1, d, a_out + i * SZ);
    }
    for (size_t i = N / 2; i < N; i++) {
      auto in1 = LoadU(d, a_in + (2 * (i - N / 2) + 1) * a_in_stride);
      Store(in1, d, a_out + i * SZ);
    }
  }
  // Invoked on full vector.
  static void Multiply(float* JXL_RESTRICT coeff) {
    const D d;
    for (size_t i = 0; i < N / 2; i++) {
      auto in1 = Load(d, coeff + (N / 2 + i) * SZ);
      auto mul = Set(d, WcMultipliers<N>::kMultipliers[i]);
      Store(Mul(in1, mul), d, coeff + (N / 2 + i) * SZ);
    }
  }
  static void MultiplyAndAdd(const float* JXL_RESTRICT coeff,
                             float* JXL_RESTRICT out, size_t out_stride) {
    const D d;
    for (size_t i = 0; i < N / 2; i++) {
      auto mul = Set(d, WcMultipliers<N>::kMultipliers[i]);
      auto in1 = Load(d, coeff + i * SZ);
      auto in2 = Load(d, coeff + (N / 2 + i) * SZ);
      auto out1 = MulAdd(mul, in2, in1);
      auto out2 = NegMulAdd(mul, in2, in1);
      StoreU(out1, d, out + i * out_stride);
      StoreU(out2, d, out + (N - i - 1) * out_stride);
    }
  }
  template <typename Block>
  static void LoadFromBlock(const Block& in, size_t off,
                            float* JXL_RESTRICT coeff) {
    const D d;
    for (size_t i = 0; i < N; i++) {
      Store(in.LoadPart(d, i, off), d, coeff + i * SZ);
    }
  }
  template <typename Block>
  static void StoreToBlockAndScale(const float* JXL_RESTRICT coeff,
                                   const Block& out, size_t off) {
    const D d;
    auto mul = Set(d, 1.0f / N);
    for (size_t i = 0; i < N; i++) {
      out.StorePart(d, Mul(mul, Load(d, coeff + i * SZ)), i, off);
    }
  }
};

template <size_t N, size_t SZ>
struct DCT1DImpl;

template <size_t SZ>
struct DCT1DImpl<1, SZ> {
  JXL_INLINE void operator()(float* JXL_RESTRICT mem, float* /* tmp */) {}
};

template <size_t SZ>
struct DCT1DImpl<2, SZ> {
  using D = HWY_CAPPED(float, SZ);
  JXL_INLINE void operator()(float* JXL_RESTRICT mem, float* /* tmp */) {
    const D d;
    auto in1 = Load(d, mem);
    auto in2 = Load(d, mem + SZ);
    Store(Add(in1, in2), d, mem);
    Store(Sub(in1, in2), d, mem + SZ);
  }
};

template <size_t N, size_t SZ>
struct DCT1DImpl {
  void operator()(float* JXL_RESTRICT mem, float* JXL_RESTRICT tmp) {
    CoeffBundle<N / 2, SZ>::AddReverse(mem, mem + N / 2 * SZ, tmp);
    DCT1DImpl<N / 2, SZ>()(tmp, tmp + N * SZ);
    CoeffBundle<N / 2, SZ>::SubReverse(mem, mem + N / 2 * SZ, tmp + N / 2 * SZ);
    CoeffBundle<N, SZ>::Multiply(tmp);
    DCT1DImpl<N / 2, SZ>()(tmp + N / 2 * SZ, tmp + N * SZ);
    CoeffBundle<N / 2, SZ>::B(tmp + N / 2 * SZ);
    CoeffBundle<N, SZ>::InverseEvenOdd(tmp, mem);
  }
};

template <size_t N, size_t SZ>
struct IDCT1DImpl;

template <size_t SZ>
struct IDCT1DImpl<1, SZ> {
  using D = HWY_CAPPED(float, SZ);
  JXL_INLINE void operator()(const float* from, size_t from_stride, float* to,
                             size_t to_stride, float* JXL_RESTRICT /* tmp */) {
    const D d;
    StoreU(LoadU(d, from), d, to);
  }
};

template <size_t SZ>
struct IDCT1DImpl<2, SZ> {
  using D = HWY_CAPPED(float, SZ);
  JXL_INLINE void operator()(const float* from, size_t from_stride, float* to,
                             size_t to_stride, float* JXL_RESTRICT /* tmp */) {
    const D d;
    JXL_DASSERT(from_stride >= SZ);
    JXL_DASSERT(to_stride >= SZ);
    auto in1 = LoadU(d, from);
    auto in2 = LoadU(d, from + from_stride);
    StoreU(Add(in1, in2), d, to);
    StoreU(Sub(in1, in2), d, to + to_stride);
  }
};

template <size_t N, size_t SZ>
struct IDCT1DImpl {
  void operator()(const float* from, size_t from_stride, float* to,
                  size_t to_stride, float* JXL_RESTRICT tmp) {
    JXL_DASSERT(from_stride >= SZ);
    JXL_DASSERT(to_stride >= SZ);
    CoeffBundle<N, SZ>::ForwardEvenOdd(from, from_stride, tmp);
    IDCT1DImpl<N / 2, SZ>()(tmp, SZ, tmp, SZ, tmp + N * SZ);
    CoeffBundle<N / 2, SZ>::BTranspose(tmp + N / 2 * SZ);
    IDCT1DImpl<N / 2, SZ>()(tmp + N / 2 * SZ, SZ, tmp + N / 2 * SZ, SZ,
                            tmp + N * SZ);
    CoeffBundle<N, SZ>::MultiplyAndAdd(tmp, to, to_stride);
  }
};

template <size_t N, size_t M, bool fit, typename FromBlock, typename ToBlock>
void DCT1DWrapper(const FromBlock& from, const ToBlock& to, size_t Mp,
                  float* JXL_RESTRICT tmp) {
  JXL_DASSERT(fit ? Mp == M : Mp > M);
  for (size_t i = 0; i < Mp; i += M) {
    // TODO(veluca): consider removing the temporary memory here (as is done in
    // IDCT), if it turns out that some compilers don't optimize away the loads
    // and this is performance-critical.
    CoeffBundle<N, M>::LoadFromBlock(from, i, tmp);
    DCT1DImpl<N, M>()(tmp, tmp + N * M);
    CoeffBundle<N, M>::StoreToBlockAndScale(tmp, to, i);
    if (fit) return;
  }
}

template <size_t N, size_t M, bool fit, typename FromBlock, typename ToBlock>
void IDCT1DWrapper(const FromBlock& from, const ToBlock& to, size_t Mp,
                   float* JXL_RESTRICT tmp) {
  JXL_DASSERT(fit ? Mp == M : Mp > M);
  for (size_t i = 0; i < Mp; i += M) {
    IDCT1DImpl<N, M>()(from.Address(0, i), from.Stride(), to.Address(0, i),
                       to.Stride(), tmp);
    if (fit) return;
  }
}

/*    if (HWY_HAVE_SCALABLE) {
      using F = void (*)(const FromBlock&, const ToBlock&, size_t,
                         float* JXL_RESTRICT);
      static F f = []() -> F {
        size_t L = Lanes(HWY_FULL(float)());
        static_assert(M <= 256, "Unsupported DCT size");
        return DCT1DWrapper<N, M,  false>;
      }();
      f(from, to, M, tmp);
*/

template <size_t N, size_t M, size_t L>
struct DCT1DCapped {
  template <typename FromBlock, typename ToBlock>
  static void Process(const FromBlock& from, const ToBlock& to,
                      float* JXL_RESTRICT tmp) {
    if (M <= L) {
      return DCT1DWrapper<N, M, /* fit */ true>(from, to, M, tmp);
    } else {
      return NoInlineWrapper(
          DCT1DWrapper<N, L, /* fit */ false, FromBlock, ToBlock>, from, to, M,
          tmp);
    }
  }
};

template <size_t N, size_t M>
struct DCT1D {
  template <typename FromBlock, typename ToBlock>
  void operator()(const FromBlock& from, const ToBlock& to,
                  float* JXL_RESTRICT tmp) {
#if HWY_HAVE_SCALABLE
    using F = void (*)(const FromBlock&, const ToBlock&, float* JXL_RESTRICT);
    static F f = []() -> F {
      size_t L = Lanes(HWY_FULL(float)());
      if (L >= 128) return DCT1DCapped<N, M, 128>::Process;
      if (L == 64) return DCT1DCapped<N, M, 64>::Process;
      if (L == 32) return DCT1DCapped<N, M, 32>::Process;
      if (L == 16) return DCT1DCapped<N, M, 16>::Process;
      if (L == 8) return DCT1DCapped<N, M, 8>::Process;
      if (L == 4) return DCT1DCapped<N, M, 4>::Process;
      if (L == 2) return DCT1DCapped<N, M, 2>::Process;
      return DCT1DCapped<N, M, 1>::Process;
    }();
    return f(from, to, tmp);
#else
    return DCT1DCapped<N, M, kMaxLanes>::Process(from, to, tmp);
#endif
  }
};

template <size_t N, size_t M, size_t L>
struct IDCT1DCapped {
  template <typename FromBlock, typename ToBlock>
  static void Process(const FromBlock& from, const ToBlock& to,
                      float* JXL_RESTRICT tmp) {
    if (M <= L) {
      return IDCT1DWrapper<N, M, /* fit */ true>(from, to, M, tmp);
    } else {
      return NoInlineWrapper(
          IDCT1DWrapper<N, L, /* fit */ false, FromBlock, ToBlock>, from, to, M,
          tmp);
    }
  }
};

template <size_t N, size_t M>
struct IDCT1D {
  template <typename FromBlock, typename ToBlock>
  void operator()(const FromBlock& from, const ToBlock& to,
                  float* JXL_RESTRICT tmp) {
#if HWY_HAVE_SCALABLE
    using F = void (*)(const FromBlock&, const ToBlock&, float* JXL_RESTRICT);
    static F f = []() -> F {
      size_t L = Lanes(HWY_FULL(float)());
      if (L >= 128) return IDCT1DCapped<N, M, 128>::Process;
      if (L == 64) return IDCT1DCapped<N, M, 64>::Process;
      if (L == 32) return IDCT1DCapped<N, M, 32>::Process;
      if (L == 16) return IDCT1DCapped<N, M, 16>::Process;
      if (L == 8) return IDCT1DCapped<N, M, 8>::Process;
      if (L == 4) return IDCT1DCapped<N, M, 4>::Process;
      if (L == 2) return IDCT1DCapped<N, M, 2>::Process;
      return IDCT1DCapped<N, M, 1>::Process;
    }();
    return f(from, to, tmp);
#else
    return IDCT1DCapped<N, M, kMaxLanes>::Process(from, to, tmp);
#endif
  }
};

// Computes the maybe-transposed, scaled DCT of a block, that needs to be
// HWY_ALIGN'ed.
template <size_t ROWS, size_t COLS>
struct ComputeScaledDCT {
  // scratch_space must be aligned, and should have space for ROWS*COLS
  // floats.
  template <class From>
  HWY_MAYBE_UNUSED void operator()(const From& from, float* to,
                                   float* JXL_RESTRICT scratch_space) {
    float* JXL_RESTRICT block = scratch_space;
    float* JXL_RESTRICT tmp = scratch_space + ROWS * COLS;
    if (ROWS < COLS) {
      DCT1D<ROWS, COLS>()(from, DCTTo(block, COLS), tmp);
      Transpose<ROWS, COLS>::Run(DCTFrom(block, COLS), DCTTo(to, ROWS));
      DCT1D<COLS, ROWS>()(DCTFrom(to, ROWS), DCTTo(block, ROWS), tmp);
      Transpose<COLS, ROWS>::Run(DCTFrom(block, ROWS), DCTTo(to, COLS));
    } else {
      DCT1D<ROWS, COLS>()(from, DCTTo(to, COLS), tmp);
      Transpose<ROWS, COLS>::Run(DCTFrom(to, COLS), DCTTo(block, ROWS));
      DCT1D<COLS, ROWS>()(DCTFrom(block, ROWS), DCTTo(to, ROWS), tmp);
    }
  }
};
// Computes the maybe-transposed, scaled IDCT of a block, that needs to be
// HWY_ALIGN'ed.
template <size_t ROWS, size_t COLS>
struct ComputeScaledIDCT {
  // scratch_space must be aligned, and should have space for ROWS*COLS
  // floats.
  template <class To>
  HWY_MAYBE_UNUSED void operator()(float* JXL_RESTRICT from, const To& to,
                                   float* JXL_RESTRICT scratch_space) {
    float* JXL_RESTRICT block = scratch_space;
    float* JXL_RESTRICT tmp = scratch_space + ROWS * COLS;
    // Reverse the steps done in ComputeScaledDCT.
    if (ROWS < COLS) {
      Transpose<ROWS, COLS>::Run(DCTFrom(from, COLS), DCTTo(block, ROWS));
      IDCT1D<COLS, ROWS>()(DCTFrom(block, ROWS), DCTTo(from, ROWS), tmp);
      Transpose<COLS, ROWS>::Run(DCTFrom(from, ROWS), DCTTo(block, COLS));
      IDCT1D<ROWS, COLS>()(DCTFrom(block, COLS), to, tmp);
    } else {
      IDCT1D<COLS, ROWS>()(DCTFrom(from, ROWS), DCTTo(block, ROWS), tmp);
      Transpose<COLS, ROWS>::Run(DCTFrom(block, ROWS), DCTTo(from, COLS));
      IDCT1D<ROWS, COLS>()(DCTFrom(from, COLS), to, tmp);
    }
  }
};

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();
#endif  // LIB_JXL_DCT_INL_H_

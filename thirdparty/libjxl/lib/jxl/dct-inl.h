// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Fast SIMD floating-point (I)DCT, any power of two.

#if defined(LIB_JXL_DCT_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef LIB_JXL_DCT_INL_H_
#undef LIB_JXL_DCT_INL_H_
#else
#define LIB_JXL_DCT_INL_H_
#endif

#include <stddef.h>

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

template <size_t SZ>
struct FVImpl {
  using type = HWY_CAPPED(float, SZ);
};

template <>
struct FVImpl<0> {
  using type = HWY_FULL(float);
};

template <size_t SZ>
using FV = typename FVImpl<SZ>::type;

// Implementation of Lowest Complexity Self Recursive Radix-2 DCT II/III
// Algorithms, by Siriani M. Perera and Jianhua Liu.

template <size_t N, size_t SZ>
struct CoeffBundle {
  static void AddReverse(const float* JXL_RESTRICT a_in1,
                         const float* JXL_RESTRICT a_in2,
                         float* JXL_RESTRICT a_out) {
    for (size_t i = 0; i < N; i++) {
      auto in1 = Load(FV<SZ>(), a_in1 + i * SZ);
      auto in2 = Load(FV<SZ>(), a_in2 + (N - i - 1) * SZ);
      Store(Add(in1, in2), FV<SZ>(), a_out + i * SZ);
    }
  }
  static void SubReverse(const float* JXL_RESTRICT a_in1,
                         const float* JXL_RESTRICT a_in2,
                         float* JXL_RESTRICT a_out) {
    for (size_t i = 0; i < N; i++) {
      auto in1 = Load(FV<SZ>(), a_in1 + i * SZ);
      auto in2 = Load(FV<SZ>(), a_in2 + (N - i - 1) * SZ);
      Store(Sub(in1, in2), FV<SZ>(), a_out + i * SZ);
    }
  }
  static void B(float* JXL_RESTRICT coeff) {
    auto sqrt2 = Set(FV<SZ>(), kSqrt2);
    auto in1 = Load(FV<SZ>(), coeff);
    auto in2 = Load(FV<SZ>(), coeff + SZ);
    Store(MulAdd(in1, sqrt2, in2), FV<SZ>(), coeff);
    for (size_t i = 1; i + 1 < N; i++) {
      auto in1 = Load(FV<SZ>(), coeff + i * SZ);
      auto in2 = Load(FV<SZ>(), coeff + (i + 1) * SZ);
      Store(Add(in1, in2), FV<SZ>(), coeff + i * SZ);
    }
  }
  static void BTranspose(float* JXL_RESTRICT coeff) {
    for (size_t i = N - 1; i > 0; i--) {
      auto in1 = Load(FV<SZ>(), coeff + i * SZ);
      auto in2 = Load(FV<SZ>(), coeff + (i - 1) * SZ);
      Store(Add(in1, in2), FV<SZ>(), coeff + i * SZ);
    }
    auto sqrt2 = Set(FV<SZ>(), kSqrt2);
    auto in1 = Load(FV<SZ>(), coeff);
    Store(Mul(in1, sqrt2), FV<SZ>(), coeff);
  }
  // Ideally optimized away by compiler (except the multiply).
  static void InverseEvenOdd(const float* JXL_RESTRICT a_in,
                             float* JXL_RESTRICT a_out) {
    for (size_t i = 0; i < N / 2; i++) {
      auto in1 = Load(FV<SZ>(), a_in + i * SZ);
      Store(in1, FV<SZ>(), a_out + 2 * i * SZ);
    }
    for (size_t i = N / 2; i < N; i++) {
      auto in1 = Load(FV<SZ>(), a_in + i * SZ);
      Store(in1, FV<SZ>(), a_out + (2 * (i - N / 2) + 1) * SZ);
    }
  }
  // Ideally optimized away by compiler.
  static void ForwardEvenOdd(const float* JXL_RESTRICT a_in, size_t a_in_stride,
                             float* JXL_RESTRICT a_out) {
    for (size_t i = 0; i < N / 2; i++) {
      auto in1 = LoadU(FV<SZ>(), a_in + 2 * i * a_in_stride);
      Store(in1, FV<SZ>(), a_out + i * SZ);
    }
    for (size_t i = N / 2; i < N; i++) {
      auto in1 = LoadU(FV<SZ>(), a_in + (2 * (i - N / 2) + 1) * a_in_stride);
      Store(in1, FV<SZ>(), a_out + i * SZ);
    }
  }
  // Invoked on full vector.
  static void Multiply(float* JXL_RESTRICT coeff) {
    for (size_t i = 0; i < N / 2; i++) {
      auto in1 = Load(FV<SZ>(), coeff + (N / 2 + i) * SZ);
      auto mul = Set(FV<SZ>(), WcMultipliers<N>::kMultipliers[i]);
      Store(Mul(in1, mul), FV<SZ>(), coeff + (N / 2 + i) * SZ);
    }
  }
  static void MultiplyAndAdd(const float* JXL_RESTRICT coeff,
                             float* JXL_RESTRICT out, size_t out_stride) {
    for (size_t i = 0; i < N / 2; i++) {
      auto mul = Set(FV<SZ>(), WcMultipliers<N>::kMultipliers[i]);
      auto in1 = Load(FV<SZ>(), coeff + i * SZ);
      auto in2 = Load(FV<SZ>(), coeff + (N / 2 + i) * SZ);
      auto out1 = MulAdd(mul, in2, in1);
      auto out2 = NegMulAdd(mul, in2, in1);
      StoreU(out1, FV<SZ>(), out + i * out_stride);
      StoreU(out2, FV<SZ>(), out + (N - i - 1) * out_stride);
    }
  }
  template <typename Block>
  static void LoadFromBlock(const Block& in, size_t off,
                            float* JXL_RESTRICT coeff) {
    for (size_t i = 0; i < N; i++) {
      Store(in.LoadPart(FV<SZ>(), i, off), FV<SZ>(), coeff + i * SZ);
    }
  }
  template <typename Block>
  static void StoreToBlockAndScale(const float* JXL_RESTRICT coeff,
                                   const Block& out, size_t off) {
    auto mul = Set(FV<SZ>(), 1.0f / N);
    for (size_t i = 0; i < N; i++) {
      out.StorePart(FV<SZ>(), Mul(mul, Load(FV<SZ>(), coeff + i * SZ)), i, off);
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
  JXL_INLINE void operator()(float* JXL_RESTRICT mem, float* /* tmp */) {
    auto in1 = Load(FV<SZ>(), mem);
    auto in2 = Load(FV<SZ>(), mem + SZ);
    Store(Add(in1, in2), FV<SZ>(), mem);
    Store(Sub(in1, in2), FV<SZ>(), mem + SZ);
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
  JXL_INLINE void operator()(const float* from, size_t from_stride, float* to,
                             size_t to_stride, float* JXL_RESTRICT /* tmp */) {
    StoreU(LoadU(FV<SZ>(), from), FV<SZ>(), to);
  }
};

template <size_t SZ>
struct IDCT1DImpl<2, SZ> {
  JXL_INLINE void operator()(const float* from, size_t from_stride, float* to,
                             size_t to_stride, float* JXL_RESTRICT /* tmp */) {
    JXL_DASSERT(from_stride >= SZ);
    JXL_DASSERT(to_stride >= SZ);
    auto in1 = LoadU(FV<SZ>(), from);
    auto in2 = LoadU(FV<SZ>(), from + from_stride);
    StoreU(Add(in1, in2), FV<SZ>(), to);
    StoreU(Sub(in1, in2), FV<SZ>(), to + to_stride);
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

template <size_t N, size_t M_or_0, typename FromBlock, typename ToBlock>
void DCT1DWrapper(const FromBlock& from, const ToBlock& to, size_t Mp,
                  float* JXL_RESTRICT tmp) {
  size_t M = M_or_0 != 0 ? M_or_0 : Mp;
  constexpr size_t SZ = MaxLanes(FV<M_or_0>());
  for (size_t i = 0; i < M; i += Lanes(FV<M_or_0>())) {
    // TODO(veluca): consider removing the temporary memory here (as is done in
    // IDCT), if it turns out that some compilers don't optimize away the loads
    // and this is performance-critical.
    CoeffBundle<N, SZ>::LoadFromBlock(from, i, tmp);
    DCT1DImpl<N, SZ>()(tmp, tmp + N * SZ);
    CoeffBundle<N, SZ>::StoreToBlockAndScale(tmp, to, i);
  }
}

template <size_t N, size_t M_or_0, typename FromBlock, typename ToBlock>
void IDCT1DWrapper(const FromBlock& from, const ToBlock& to, size_t Mp,
                   float* JXL_RESTRICT tmp) {
  size_t M = M_or_0 != 0 ? M_or_0 : Mp;
  constexpr size_t SZ = MaxLanes(FV<M_or_0>());
  for (size_t i = 0; i < M; i += Lanes(FV<M_or_0>())) {
    IDCT1DImpl<N, SZ>()(from.Address(0, i), from.Stride(), to.Address(0, i),
                        to.Stride(), tmp);
  }
}

template <size_t N, size_t M, typename = void>
struct DCT1D {
  template <typename FromBlock, typename ToBlock>
  void operator()(const FromBlock& from, const ToBlock& to,
                  float* JXL_RESTRICT tmp) {
    return DCT1DWrapper<N, M>(from, to, M, tmp);
  }
};

template <size_t N, size_t M>
struct DCT1D<N, M, typename std::enable_if<(M > MaxLanes(FV<0>()))>::type> {
  template <typename FromBlock, typename ToBlock>
  void operator()(const FromBlock& from, const ToBlock& to,
                  float* JXL_RESTRICT tmp) {
    return NoInlineWrapper(DCT1DWrapper<N, 0, FromBlock, ToBlock>, from, to, M,
                           tmp);
  }
};

template <size_t N, size_t M, typename = void>
struct IDCT1D {
  template <typename FromBlock, typename ToBlock>
  void operator()(const FromBlock& from, const ToBlock& to,
                  float* JXL_RESTRICT tmp) {
    return IDCT1DWrapper<N, M>(from, to, M, tmp);
  }
};

template <size_t N, size_t M>
struct IDCT1D<N, M, typename std::enable_if<(M > MaxLanes(FV<0>()))>::type> {
  template <typename FromBlock, typename ToBlock>
  void operator()(const FromBlock& from, const ToBlock& to,
                  float* JXL_RESTRICT tmp) {
    return NoInlineWrapper(IDCT1DWrapper<N, 0, FromBlock, ToBlock>, from, to, M,
                           tmp);
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

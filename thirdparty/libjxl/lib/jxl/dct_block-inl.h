// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Adapters for DCT input/output: from/to contiguous blocks or image rows.

#if defined(LIB_JXL_DCT_BLOCK_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef LIB_JXL_DCT_BLOCK_INL_H_
#undef LIB_JXL_DCT_BLOCK_INL_H_
#else
#define LIB_JXL_DCT_BLOCK_INL_H_
#endif

#include <stddef.h>

#include <hwy/highway.h>

#include "lib/jxl/base/status.h"
HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {
namespace {

// These templates are not found via ADL.
using hwy::HWY_NAMESPACE::Vec;

// Block: (x, y) <-> (N * y + x)
// Lines: (x, y) <-> (stride * y + x)
//
// I.e. Block is a specialization of Lines with fixed stride.
//
// FromXXX should implement Read and Load (Read vector).
// ToXXX should implement Write and Store (Write vector).

template <size_t N>
using BlockDesc = HWY_CAPPED(float, N);

// Here and in the following, the SZ template parameter specifies the number of
// values to load/store. Needed because we want to handle 4x4 sub-blocks of
// 16x16 blocks.
class DCTFrom {
 public:
  DCTFrom(const float* data, size_t stride) : stride_(stride), data_(data) {}

  template <typename D>
  HWY_INLINE Vec<D> LoadPart(D /* tag */, const size_t row, size_t i) const {
    JXL_DASSERT(Lanes(D()) <= stride_);
    // Since these functions are used also for DC, no alignment at all is
    // guaranteed in the case of floating blocks.
    // TODO(veluca): consider using a different class for DC-to-LF and
    // DC-from-LF, or copying DC values to/from a temporary aligned location.
    return LoadU(D(), Address(row, i));
  }

  HWY_INLINE float Read(const size_t row, const size_t i) const {
    return *Address(row, i);
  }

  constexpr HWY_INLINE const float* Address(const size_t row,
                                            const size_t i) const {
    return data_ + row * stride_ + i;
  }

  size_t Stride() const { return stride_; }

 private:
  size_t stride_;
  const float* JXL_RESTRICT data_;
};

class DCTTo {
 public:
  DCTTo(float* data, size_t stride) : stride_(stride), data_(data) {}

  template <typename D>
  HWY_INLINE void StorePart(D /* tag */, const Vec<D>& v, const size_t row,
                            size_t i) const {
    JXL_DASSERT(Lanes(D()) <= stride_);
    // Since these functions are used also for DC, no alignment at all is
    // guaranteed in the case of floating blocks.
    // TODO(veluca): consider using a different class for DC-to-LF and
    // DC-from-LF, or copying DC values to/from a temporary aligned location.
    StoreU(v, D(), Address(row, i));
  }

  HWY_INLINE void Write(float v, const size_t row, const size_t i) const {
    *Address(row, i) = v;
  }

  constexpr HWY_INLINE float* Address(const size_t row, const size_t i) const {
    return data_ + row * stride_ + i;
  }

  size_t Stride() const { return stride_; }

 private:
  size_t stride_;
  float* JXL_RESTRICT data_;
};

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#endif  // LIB_JXL_DCT_BLOCK_INL_H_

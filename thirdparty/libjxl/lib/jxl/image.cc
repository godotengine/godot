// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/image.h"

#include <jxl/memory_manager.h>

#include <algorithm>  // fill, swap
#include <cstddef>
#include <cstdint>
#include <limits>

#include "lib/jxl/base/status.h"
#include "lib/jxl/memory_manager_internal.h"

#if defined(MEMORY_SANITIZER)
#include "lib/jxl/base/common.h"
#include "lib/jxl/base/sanitizers.h"
#include "lib/jxl/simd_util.h"
#endif

namespace jxl {
namespace detail {

namespace {

// Initializes the minimum bytes required to suppress MSAN warnings from
// legitimate vector loads/stores on the right border, where some lanes are
// uninitialized and assumed to be unused.
void InitializePadding(PlaneBase& plane, const size_t sizeof_t) {
#if defined(MEMORY_SANITIZER)
  size_t xsize = plane.xsize();
  size_t ysize = plane.ysize();
  if (xsize == 0 || ysize == 0) return;

  const size_t vec_size = MaxVectorSize();
  if (vec_size == 0) return;  // Scalar mode: no padding needed

  const size_t valid_size = xsize * sizeof_t;
  const size_t initialize_size = RoundUpTo(valid_size, vec_size);
  if (valid_size == initialize_size) return;

  for (size_t y = 0; y < ysize; ++y) {
    uint8_t* JXL_RESTRICT row = plane.bytes() + y * plane.bytes_per_row();
#if defined(__clang__) &&                                           \
    ((!defined(__apple_build_version__) && __clang_major__ <= 6) || \
     (defined(__apple_build_version__) &&                           \
      __apple_build_version__ <= 10001145))
    // There's a bug in MSAN in clang-6 when handling AVX2 operations. This
    // workaround allows tests to pass on MSAN, although it is slower and
    // prevents MSAN warnings from uninitialized images.
    std::fill(row, msan::kSanitizerSentinelByte, initialize_size);
#else
    memset(row + valid_size, msan::kSanitizerSentinelByte,
           initialize_size - valid_size);
#endif  // clang6
  }
#endif  // MEMORY_SANITIZER
}

}  // namespace

PlaneBase::PlaneBase(const uint32_t xsize, const uint32_t ysize,
                     const size_t sizeof_t)
    : xsize_(xsize),
      ysize_(ysize),
      orig_xsize_(xsize),
      orig_ysize_(ysize),
      bytes_per_row_(BytesPerRow(xsize_, sizeof_t)),
      sizeof_t_(sizeof_t) {}

Status PlaneBase::Allocate(JxlMemoryManager* memory_manager,
                           size_t pre_padding) {
  JXL_ENSURE(bytes_.address<void>() == nullptr);

  // Dimensions can be zero, e.g. for lazily-allocated images. Only allocate
  // if nonzero, because "zero" bytes still have padding/bookkeeping overhead.
  if (xsize_ == 0 || ysize_ == 0) {
    return true;
  }

  size_t max_y_size = std::numeric_limits<size_t>::max() / bytes_per_row_;
  if (ysize_ > max_y_size) {
    return JXL_FAILURE("Image dimensions are too large");
  }

  JXL_ASSIGN_OR_RETURN(
      bytes_, AlignedMemory::Create(memory_manager, bytes_per_row_ * ysize_,
                                    pre_padding * sizeof_t_));

  InitializePadding(*this, sizeof_t_);

  return true;
}

void PlaneBase::Swap(PlaneBase& other) {
  std::swap(xsize_, other.xsize_);
  std::swap(ysize_, other.ysize_);
  std::swap(orig_xsize_, other.orig_xsize_);
  std::swap(orig_ysize_, other.orig_ysize_);
  std::swap(bytes_per_row_, other.bytes_per_row_);
  std::swap(bytes_, other.bytes_);
}

}  // namespace detail
}  // namespace jxl

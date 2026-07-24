// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_IMAGE_OPS_H_
#define LIB_JXL_IMAGE_OPS_H_

// Operations on images.

#include <jxl/memory_manager.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/frame_dimensions.h"
#include "lib/jxl/image.h"

namespace jxl {

// Works for mixed image-like argument types.
template <class Image1, class Image2>
bool SameSize(const Image1& image1, const Image2& image2) {
  return image1.xsize() == image2.xsize() && image1.ysize() == image2.ysize();
}

template <typename T>
Status CopyImageTo(const Plane<T>& from, Plane<T>* JXL_RESTRICT to) {
  JXL_ENSURE(SameSize(from, *to));
  if (from.ysize() == 0 || from.xsize() == 0) return true;
  for (size_t y = 0; y < from.ysize(); ++y) {
    const T* JXL_RESTRICT row_from = from.ConstRow(y);
    T* JXL_RESTRICT row_to = to->Row(y);
    memcpy(row_to, row_from, from.xsize() * sizeof(T));
  }
  return true;
}

// Copies `from:rect_from` to `to:rect_to`.
template <typename T>
Status CopyImageTo(const Rect& rect_from, const Plane<T>& from,
                   const Rect& rect_to, Plane<T>* JXL_RESTRICT to) {
  JXL_ENSURE(SameSize(rect_from, rect_to));
  JXL_ENSURE(rect_from.IsInside(from));
  JXL_ENSURE(rect_to.IsInside(*to));
  if (rect_from.xsize() == 0) return true;
  for (size_t y = 0; y < rect_from.ysize(); ++y) {
    const T* JXL_RESTRICT row_from = rect_from.ConstRow(from, y);
    T* JXL_RESTRICT row_to = rect_to.Row(to, y);
    memcpy(row_to, row_from, rect_from.xsize() * sizeof(T));
  }
  return true;
}

// Copies `from:rect_from` to `to:rect_to`.
template <typename T>
Status CopyImageTo(const Rect& rect_from, const Image3<T>& from,
                   const Rect& rect_to, Image3<T>* JXL_RESTRICT to) {
  JXL_ENSURE(SameSize(rect_from, rect_to));
  for (size_t c = 0; c < 3; c++) {
    JXL_RETURN_IF_ERROR(
        CopyImageTo(rect_from, from.Plane(c), rect_to, &to->Plane(c)));
  }
  return true;
}

template <typename T, typename U>
Status ConvertPlaneAndClamp(const Rect& rect_from, const Plane<T>& from,
                            const Rect& rect_to, Plane<U>* JXL_RESTRICT to) {
  JXL_ENSURE(SameSize(rect_from, rect_to));
  using M = decltype(T() + U());
  for (size_t y = 0; y < rect_to.ysize(); ++y) {
    const T* JXL_RESTRICT row_from = rect_from.ConstRow(from, y);
    U* JXL_RESTRICT row_to = rect_to.Row(to, y);
    for (size_t x = 0; x < rect_to.xsize(); ++x) {
      row_to[x] =
          std::min<M>(std::max<M>(row_from[x], std::numeric_limits<U>::min()),
                      std::numeric_limits<U>::max());
    }
  }
  return true;
}

// Copies `from` to `to`.
template <typename T>
Status CopyImageTo(const T& from, T* JXL_RESTRICT to) {
  return CopyImageTo(Rect(from), from, Rect(*to), to);
}

// Copies `from:rect_from` to `to:rect_to`; also copies `padding` pixels of
// border around `from:rect_from`, in all directions, whenever they are inside
// the first image.
template <typename T>
Status CopyImageToWithPadding(const Rect& from_rect, const T& from,
                              size_t padding, const Rect& to_rect, T* to) {
  size_t xextra0 = std::min(padding, from_rect.x0());
  size_t xextra1 =
      std::min(padding, from.xsize() - from_rect.x0() - from_rect.xsize());
  size_t yextra0 = std::min(padding, from_rect.y0());
  size_t yextra1 =
      std::min(padding, from.ysize() - from_rect.y0() - from_rect.ysize());
  JXL_ENSURE(to_rect.x0() >= xextra0);
  JXL_ENSURE(to_rect.y0() >= yextra0);

  return CopyImageTo(Rect(from_rect.x0() - xextra0, from_rect.y0() - yextra0,
                          from_rect.xsize() + xextra0 + xextra1,
                          from_rect.ysize() + yextra0 + yextra1),
                     from,
                     Rect(to_rect.x0() - xextra0, to_rect.y0() - yextra0,
                          to_rect.xsize() + xextra0 + xextra1,
                          to_rect.ysize() + yextra0 + yextra1),
                     to);
}

// Returns linear combination of two grayscale images.
template <typename T>
StatusOr<Plane<T>> LinComb(const T lambda1, const Plane<T>& image1,
                           const T lambda2, const Plane<T>& image2) {
  const size_t xsize = image1.xsize();
  const size_t ysize = image1.ysize();
  JXL_ENSURE(xsize == image2.xsize());
  JXL_ENSURE(ysize == image2.ysize());
  JxlMemoryManager* memory_manager = image1.memory_manager();
  JXL_ASSIGN_OR_RETURN(Plane<T> out,
                       Plane<T>::Create(memory_manager, xsize, ysize));
  for (size_t y = 0; y < ysize; ++y) {
    const T* const JXL_RESTRICT row1 = image1.Row(y);
    const T* const JXL_RESTRICT row2 = image2.Row(y);
    T* const JXL_RESTRICT row_out = out.Row(y);
    for (size_t x = 0; x < xsize; ++x) {
      row_out[x] = lambda1 * row1[x] + lambda2 * row2[x];
    }
  }
  return out;
}

// Multiplies image by lambda in-place
template <typename T>
void ScaleImage(const T lambda, Plane<T>* image) {
  for (size_t y = 0; y < image->ysize(); ++y) {
    T* const JXL_RESTRICT row = image->Row(y);
    for (size_t x = 0; x < image->xsize(); ++x) {
      row[x] = lambda * row[x];
    }
  }
}

// Multiplies image by lambda in-place
template <typename T>
void ScaleImage(const T lambda, Image3<T>* image) {
  for (size_t c = 0; c < 3; ++c) {
    ScaleImage(lambda, &image->Plane(c));
  }
}

template <typename T>
void FillImage(const T value, Plane<T>* image) {
  for (size_t y = 0; y < image->ysize(); ++y) {
    T* const JXL_RESTRICT row = image->Row(y);
    for (size_t x = 0; x < image->xsize(); ++x) {
      row[x] = value;
    }
  }
}

template <typename T>
void ZeroFillImage(Plane<T>* image) {
  if (image->xsize() == 0) return;
  for (size_t y = 0; y < image->ysize(); ++y) {
    T* const JXL_RESTRICT row = image->Row(y);
    memset(row, 0, image->xsize() * sizeof(T));
  }
}

// Mirrors out of bounds coordinates and returns valid coordinates unchanged.
// We assume the radius (distance outside the image) is small compared to the
// image size, otherwise this might not terminate.
// The mirror is outside the last column (border pixel is also replicated).
static inline int64_t Mirror(int64_t x, const int64_t xsize) {
  JXL_DASSERT(xsize != 0);

  // TODO(janwas): replace with branchless version
  while (x < 0 || x >= xsize) {
    if (x < 0) {
      x = -x - 1;
    } else {
      x = 2 * xsize - 1 - x;
    }
  }
  return x;
}

// Wrap modes for ensuring X/Y coordinates are in the valid range [0, size):

// Mirrors (repeating the edge pixel once). Useful for convolutions.
struct WrapMirror {
  JXL_INLINE int64_t operator()(const int64_t coord, const int64_t size) const {
    return Mirror(coord, size);
  }
};

// Returns the same coordinate: required for TFNode with Border(), or useful
// when we know "coord" is already valid (e.g. interior of an image).
struct WrapUnchanged {
  JXL_INLINE int64_t operator()(const int64_t coord, int64_t /*size*/) const {
    return coord;
  }
};

// Similar to Wrap* but for row pointers (reduces Row() multiplications).

class WrapRowMirror {
 public:
  template <class ImageOrView>
  WrapRowMirror(const ImageOrView& image, size_t ysize)
      : first_row_(image.ConstRow(0)), last_row_(image.ConstRow(ysize - 1)) {}

  const float* operator()(const float* const JXL_RESTRICT row,
                          const int64_t stride) const {
    if (row < first_row_) {
      const int64_t num_before = first_row_ - row;
      // Mirrored; one row before => row 0, two before = row 1, ...
      return first_row_ + num_before - stride;
    }
    if (row > last_row_) {
      const int64_t num_after = row - last_row_;
      // Mirrored; one row after => last row, two after = last - 1, ...
      return last_row_ - num_after + stride;
    }
    return row;
  }

 private:
  const float* const JXL_RESTRICT first_row_;
  const float* const JXL_RESTRICT last_row_;
};

struct WrapRowUnchanged {
  JXL_INLINE const float* operator()(const float* const JXL_RESTRICT row,
                                     int64_t /*stride*/) const {
    return row;
  }
};

// Computes the minimum and maximum pixel value.
template <typename T>
void ImageMinMax(const Plane<T>& image, T* const JXL_RESTRICT min,
                 T* const JXL_RESTRICT max) {
  *min = std::numeric_limits<T>::max();
  *max = std::numeric_limits<T>::lowest();
  for (size_t y = 0; y < image.ysize(); ++y) {
    const T* const JXL_RESTRICT row = image.Row(y);
    for (size_t x = 0; x < image.xsize(); ++x) {
      *min = std::min(*min, row[x]);
      *max = std::max(*max, row[x]);
    }
  }
}

// Initializes all planes to the same "value".
template <typename T>
void FillImage(const T value, Image3<T>* image) {
  for (size_t c = 0; c < 3; ++c) {
    for (size_t y = 0; y < image->ysize(); ++y) {
      T* JXL_RESTRICT row = image->PlaneRow(c, y);
      for (size_t x = 0; x < image->xsize(); ++x) {
        row[x] = value;
      }
    }
  }
}

template <typename T>
void FillPlane(const T value, Plane<T>* image, Rect rect) {
  for (size_t y = 0; y < rect.ysize(); ++y) {
    T* JXL_RESTRICT row = rect.Row(image, y);
    for (size_t x = 0; x < rect.xsize(); ++x) {
      row[x] = value;
    }
  }
}

template <typename T>
void ZeroFillImage(Image3<T>* image) {
  for (size_t c = 0; c < 3; ++c) {
    for (size_t y = 0; y < image->ysize(); ++y) {
      T* JXL_RESTRICT row = image->PlaneRow(c, y);
      if (image->xsize() != 0) memset(row, 0, image->xsize() * sizeof(T));
    }
  }
}

// Same as above, but operates in-place. Assumes that the `in` image was
// allocated large enough.
Status PadImageToBlockMultipleInPlace(Image3F* JXL_RESTRICT in,
                                      size_t block_dim = kBlockDim);

// Downsamples an image by a given factor.
StatusOr<Image3F> DownsampleImage(const Image3F& opsin, size_t factor);
StatusOr<ImageF> DownsampleImage(const ImageF& image, size_t factor);

}  // namespace jxl

#endif  // LIB_JXL_IMAGE_OPS_H_

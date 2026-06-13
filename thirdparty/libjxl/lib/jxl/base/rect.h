// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_BASE_RECT_H_
#define LIB_JXL_BASE_RECT_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>
#include <utility>  // std::move

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/status.h"

namespace jxl {

// Rectangular region in image(s). Factoring this out of Image instead of
// shifting the pointer by x0/y0 allows this to apply to multiple images with
// different resolutions (e.g. color transform and quantization field).
// Can compare using SameSize(rect1, rect2).
template <typename T>
class RectT {
 public:
  // Most windows are xsize_max * ysize_max, except those on the borders where
  // begin + size_max > end.
  constexpr RectT(T xbegin, T ybegin, size_t xsize_max, size_t ysize_max,
                  T xend, T yend)
      : x0_(xbegin),
        y0_(ybegin),
        xsize_(ClampedSize(xbegin, xsize_max, xend)),
        ysize_(ClampedSize(ybegin, ysize_max, yend)) {}

  // Construct with origin and known size (typically from another Rect).
  constexpr RectT(T xbegin, T ybegin, size_t xsize, size_t ysize)
      : x0_(xbegin), y0_(ybegin), xsize_(xsize), ysize_(ysize) {}

  // Construct a rect that covers a whole image/plane/ImageBundle etc.
  template <typename ImageT>
  explicit RectT(const ImageT& image)
      : RectT(0, 0, image.xsize(), image.ysize()) {}

  RectT() : RectT(0, 0, 0, 0) {}

  RectT(const RectT&) = default;
  RectT& operator=(const RectT&) = default;

  // Construct a subrect that resides in an image/plane/ImageBundle etc.
  template <typename ImageT>
  RectT Crop(const ImageT& image) const {
    return Intersection(RectT(image));
  }

  // Construct a subrect that resides in the [0, ysize) x [0, xsize) region of
  // the current rect.
  RectT Crop(size_t area_xsize, size_t area_ysize) const {
    return Intersection(RectT(0, 0, area_xsize, area_ysize));
  }

  JXL_MUST_USE_RESULT RectT Intersection(const RectT& other) const {
    return RectT(std::max(x0_, other.x0_), std::max(y0_, other.y0_), xsize_,
                 ysize_, std::min(x1(), other.x1()),
                 std::min(y1(), other.y1()));
  }

  JXL_MUST_USE_RESULT RectT Translate(int64_t x_offset,
                                      int64_t y_offset) const {
    return RectT(x0_ + x_offset, y0_ + y_offset, xsize_, ysize_);
  }

  template <template <class> class P, typename V>
  V* Row(P<V>* image, size_t y) const {
    JXL_DASSERT(y + y0_ >= 0);
    return image->Row(y + y0_) + x0_;
  }

  template <template <class> class P, typename V>
  const V* Row(const P<V>* image, size_t y) const {
    JXL_DASSERT(y + y0_ >= 0);
    return image->Row(y + y0_) + x0_;
  }

  template <template <class> class MP, typename V>
  V* PlaneRow(MP<V>* image, const size_t c, size_t y) const {
    JXL_DASSERT(y + y0_ >= 0);
    return image->PlaneRow(c, y + y0_) + x0_;
  }

  template <template <class> class P, typename V>
  const V* ConstRow(const P<V>& image, size_t y) const {
    JXL_DASSERT(y + y0_ >= 0);
    return image.ConstRow(y + y0_) + x0_;
  }

  template <template <class> class MP, typename V>
  const V* ConstPlaneRow(const MP<V>& image, size_t c, size_t y) const {
    JXL_DASSERT(y + y0_ >= 0);
    return image.ConstPlaneRow(c, y + y0_) + x0_;
  }

  bool IsInside(const RectT& other) const {
    return x0_ >= other.x0() && x1() <= other.x1() && y0_ >= other.y0() &&
           y1() <= other.y1();
  }

  bool IsSame(const RectT& other) const {
    return x0_ == other.x0_ && xsize_ == other.xsize_ && y0_ == other.y0_ &&
           ysize_ <= other.ysize_;
  }

  // Returns true if this Rect fully resides in the given image. ImageT could be
  // Plane<T> or Image3<T>; however if ImageT is Rect, results are nonsensical.
  template <class ImageT>
  bool IsInside(const ImageT& image) const {
    return IsInside(RectT(image));
  }

  T x0() const { return x0_; }
  T y0() const { return y0_; }
  size_t xsize() const { return xsize_; }
  size_t ysize() const { return ysize_; }
  T x1() const { return x0_ + xsize_; }
  T y1() const { return y0_ + ysize_; }

  RectT<T> ShiftLeft(size_t shiftx, size_t shifty) const {
    return RectT<T>(x0_ * (1 << shiftx), y0_ * (1 << shifty), xsize_ << shiftx,
                    ysize_ << shifty);
  }
  RectT<T> ShiftLeft(size_t shift) const { return ShiftLeft(shift, shift); }

  // Requires x0(), y0() to be multiples of 1<<shiftx, 1<<shifty.
  StatusOr<RectT<T>> CeilShiftRight(std::pair<size_t, size_t> shift) const {
    size_t shiftx = shift.first;
    size_t shifty = shift.second;
    JXL_ENSURE((x0_ % (1 << shiftx) == 0) && (y0_ % (1 << shifty) == 0));
    return RectT<T>(x0_ / (1 << shiftx), y0_ / (1 << shifty),
                    DivCeil(xsize_, T{1} << shiftx),
                    DivCeil(ysize_, T{1} << shifty));
  }

  RectT<T> Extend(T border, RectT<T> parent) const {
    T new_x0 = x0() > parent.x0() + border ? x0() - border : parent.x0();
    T new_y0 = y0() > parent.y0() + border ? y0() - border : parent.y0();
    T new_x1 = x1() + border > parent.x1() ? parent.x1() : x1() + border;
    T new_y1 = y1() + border > parent.y1() ? parent.y1() : y1() + border;
    return RectT<T>(new_x0, new_y0, new_x1 - new_x0, new_y1 - new_y0);
  }

  template <typename U>
  RectT<U> As() const {
    return RectT<U>(static_cast<U>(x0_), static_cast<U>(y0_),
                    static_cast<U>(xsize_), static_cast<U>(ysize_));
  }

 private:
  // Returns size_max, or whatever is left in [begin, end).
  static constexpr size_t ClampedSize(T begin, size_t size_max, T end) {
    return (static_cast<T>(begin + size_max) <= end)
               ? size_max
               : (end > begin ? end - begin : 0);
  }

  T x0_;
  T y0_;

  size_t xsize_;
  size_t ysize_;
};

template <typename T>
std::string Description(RectT<T> r) {
  std::ostringstream os;
  os << "[" << r.x0() << ".." << r.x1() << ")x"
     << "[" << r.y0() << ".." << r.y1() << ")";
  return os.str();
}

using Rect = RectT<size_t>;

}  // namespace jxl

#endif  // LIB_JXL_BASE_RECT_H_

// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_SANITIZERS_H_
#define LIB_JXL_SANITIZERS_H_

#include <cstddef>

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/sanitizer_definitions.h"

#if JXL_MEMORY_SANITIZER
#include <algorithm>
#include <cinttypes>  // PRId64
#include <cstdio>
#include <string>
#include <vector>

#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/status.h"
#include "sanitizer/msan_interface.h"
#endif

namespace jxl {
namespace msan {

#if JXL_MEMORY_SANITIZER

// Chosen so that kSanitizerSentinel is four copies of kSanitizerSentinelByte.
constexpr uint8_t kSanitizerSentinelByte = 0x48;
constexpr float kSanitizerSentinel = 205089.125f;

static JXL_INLINE JXL_MAYBE_UNUSED void PoisonMemory(const volatile void* m,
                                                     size_t size) {
  __msan_poison(m, size);
}

static JXL_INLINE JXL_MAYBE_UNUSED void UnpoisonMemory(const volatile void* m,
                                                       size_t size) {
  __msan_unpoison(m, size);
}

static JXL_INLINE JXL_MAYBE_UNUSED void MemoryIsInitialized(
    const volatile void* m, size_t size) {
  __msan_check_mem_is_initialized(m, size);
}

// Mark all the bytes of an image (including padding) as poisoned bytes.
template <typename Pixels>
static JXL_INLINE JXL_MAYBE_UNUSED void PoisonImage(const Pixels& im) {
  PoisonMemory(im.bytes(), im.bytes_per_row() * im.ysize());
}

namespace {

// Print the uninitialized regions of an image.
template <typename Pixels>
static JXL_INLINE JXL_MAYBE_UNUSED void PrintImageUninitialized(
    const Pixels& im) {
  fprintf(stderr,
          "Uninitialized regions for image of size %" PRIu64 "x%" PRIu64 ":\n",
          static_cast<uint64_t>(im.xsize()), static_cast<uint64_t>(im.ysize()));

  // A segment of uninitialized pixels in a row, in the format [first, second).
  typedef std::pair<size_t, size_t> PixelSegment;

  // Helper class to merge and print a list of rows of PixelSegment that may be
  // the same over big ranges of rows. This compacts the output to ranges of
  // rows like "[y0, y1): [x0, x1) [x2, x3)".
  class RowsMerger {
   public:
    // Add a new row the list of rows. If the row is the same as the previous
    // one it will be merged showing a range of rows [y0, y1), but if the new
    // row is different the current range of rows (if any) will be printed and a
    // new one will be started.
    void AddRow(size_t y, std::vector<PixelSegment>&& new_row) {
      if (start_y_ != -1 && new_row != segments_) {
        PrintRow(y);
      }
      if (new_row.empty()) {
        // Skip ranges with no uninitialized pixels.
        start_y_ = -1;
        segments_.clear();
        return;
      }
      if (start_y_ == -1) {
        start_y_ = y;
        segments_ = std::move(new_row);
      }
    }

    // Print the contents of the range of rows [start_y_, end_y) if any.
    void PrintRow(size_t end_y) {
      if (start_y_ == -1) return;
      if (segments_.empty()) {
        start_y_ = -1;
        return;
      }
      if (end_y - start_y_ > 1) {
        fprintf(stderr, " y=[%" PRId64 ", %" PRIu64 "):",
                static_cast<int64_t>(start_y_), static_cast<uint64_t>(end_y));
      } else {
        fprintf(stderr, " y=[%" PRId64 "]:", static_cast<int64_t>(start_y_));
      }
      for (const auto& seg : segments_) {
        if (seg.first + 1 == seg.second) {
          fprintf(stderr, " [%" PRId64 "]", static_cast<int64_t>(seg.first));
        } else {
          fprintf(stderr, " [%" PRId64 ", %" PRIu64 ")",
                  static_cast<int64_t>(seg.first),
                  static_cast<uint64_t>(seg.second));
        }
      }
      fprintf(stderr, "\n");
      start_y_ = -1;
    }

   private:
    std::vector<PixelSegment> segments_;
    // Row number of the first row in the range of rows that have |segments| as
    // the undefined segments.
    ssize_t start_y_ = -1;
  } rows_merger;

  class SegmentsMerger {
   public:
    void AddValue(size_t x) {
      if (row.empty() || row.back().second != x) {
        row.emplace_back(x, x + 1);
      } else {
        row.back().second = x + 1;
      }
    }

    std::vector<PixelSegment> row;
  };

  for (size_t y = 0; y < im.ysize(); y++) {
    auto* row = im.Row(y);
    SegmentsMerger seg_merger;
    size_t x = 0;
    while (x < im.xsize()) {
      intptr_t ret =
          __msan_test_shadow(row + x, (im.xsize() - x) * sizeof(row[0]));
      if (ret < 0) break;
      size_t next_x = x + ret / sizeof(row[0]);
      seg_merger.AddValue(next_x);
      x = next_x + 1;
    }
    rows_merger.AddRow(y, std::move(seg_merger.row));
  }
  rows_merger.PrintRow(im.ysize());
}

// Check that all the pixels in the provided rect of the image are initialized
// (not poisoned). If any of the values is poisoned it will abort.
template <typename Pixels>
static JXL_INLINE JXL_MAYBE_UNUSED void CheckImageInitialized(
    const Pixels& im, const Rect& r, size_t c, const char* message) {
  JXL_DASSERT(r.x0() <= im.xsize());
  JXL_DASSERT(r.x0() + r.xsize() <= im.xsize());
  JXL_DASSERT(r.y0() <= im.ysize());
  JXL_DASSERT(r.y0() + r.ysize() <= im.ysize());
  for (size_t y = r.y0(); y < r.y0() + r.ysize(); y++) {
    const auto* row = im.Row(y);
    intptr_t ret = __msan_test_shadow(row + r.x0(), sizeof(*row) * r.xsize());
    if (ret != -1) {
      JXL_DEBUG(
          1,
          "Checking an image of %" PRIu64 " x %" PRIu64 ", rect x0=%" PRIu64
          ", y0=%" PRIu64
          ", "
          "xsize=%" PRIu64 ", ysize=%" PRIu64,
          static_cast<uint64_t>(im.xsize()), static_cast<uint64_t>(im.ysize()),
          static_cast<uint64_t>(r.x0()), static_cast<uint64_t>(r.y0()),
          static_cast<uint64_t>(r.xsize()), static_cast<uint64_t>(r.ysize()));
      size_t x = ret / sizeof(*row);
      JXL_DEBUG(1,
                "CheckImageInitialized failed at x=%" PRIu64 ", y=%" PRIu64
                ", c=%" PRIu64 ": %s",
                static_cast<uint64_t>(r.x0() + x), static_cast<uint64_t>(y),
                static_cast<uint64_t>(c), message ? message : "");
      PrintImageUninitialized(im);
    }
    // This will report an error if memory is not initialized.
    __msan_check_mem_is_initialized(row + r.x0(), sizeof(*row) * r.xsize());
  }
}

template <typename Image>
static JXL_INLINE JXL_MAYBE_UNUSED void CheckImageInitialized(
    const Image& im, const Rect& r, const char* message) {
  for (size_t c = 0; c < 3; c++) {
    std::string str_message(message);
    str_message += " c=" + std::to_string(c);
    CheckImageInitialized(im.Plane(c), r, c, str_message.c_str());
  }
}

}  // namespace

#define JXL_CHECK_IMAGE_INITIALIZED(im, r) \
  ::jxl::msan::CheckImageInitialized(im, r, "im=" #im ", r=" #r);

#define JXL_CHECK_PLANE_INITIALIZED(im, r, c) \
  ::jxl::msan::CheckImageInitialized(im, r, c, "im=" #im ", r=" #r ", c=" #c);

#else  // JXL_MEMORY_SANITIZER

// In non-msan mode these functions don't use volatile since it is not needed
// for the empty functions.

static JXL_INLINE JXL_MAYBE_UNUSED void PoisonMemory(const void* m,
                                                     size_t size) {}
static JXL_INLINE JXL_MAYBE_UNUSED void UnpoisonMemory(const void* m,
                                                       size_t size) {}
static JXL_INLINE JXL_MAYBE_UNUSED void MemoryIsInitialized(const void* m,
                                                            size_t size) {}

template <typename Pixels>
static JXL_INLINE JXL_MAYBE_UNUSED void PoisonImage(const Pixels& im) {}

#define JXL_CHECK_IMAGE_INITIALIZED(im, r)
#define JXL_CHECK_PLANE_INITIALIZED(im, r, c)

#endif

}  // namespace msan
}  // namespace jxl

#endif  // LIB_JXL_SANITIZERS_H_

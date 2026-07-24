// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_MODULAR_MODULAR_IMAGE_H_
#define LIB_JXL_MODULAR_MODULAR_IMAGE_H_

#include <jxl/memory_manager.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/image.h"

namespace jxl {

typedef int32_t pixel_type;  // can use int16_t if it's only for 8-bit images.
                             // Need some wiggle room for YCoCg / Squeeze etc

typedef int64_t pixel_type_w;

namespace weighted {
struct Header;
}

class Channel {
 public:
  jxl::Plane<pixel_type> plane;
  size_t w, h;
  int hshift, vshift;  // w ~= image.w >> hshift;  h ~= image.h >> vshift
  Channel(const Channel& other) = delete;
  Channel& operator=(const Channel& other) = delete;

  static StatusOr<Channel> Create(JxlMemoryManager* memory_manager, size_t iw,
                                  size_t ih, int hsh = 0, int vsh = 0) {
    JXL_ASSIGN_OR_RETURN(Plane<pixel_type> plane,
                         Plane<pixel_type>::Create(memory_manager, iw, ih));
    return Channel(std::move(plane), iw, ih, hsh, vsh);
  }

  // Move assignment
  Channel& operator=(Channel&& other) noexcept {
    w = other.w;
    h = other.h;
    hshift = other.hshift;
    vshift = other.vshift;
    plane = std::move(other.plane);
    return *this;
  }

  // Move constructor
  Channel(Channel&& other) noexcept = default;

  JxlMemoryManager* memory_manager() const { return plane.memory_manager(); };

  Status shrink() {
    if (plane.xsize() == w && plane.ysize() == h) return true;
    JXL_ASSIGN_OR_RETURN(plane,
                         Plane<pixel_type>::Create(memory_manager(), w, h));
    return true;
  }
  Status shrink(int nw, int nh) {
    w = nw;
    h = nh;
    return shrink();
  }

  JXL_INLINE pixel_type* Row(const size_t y) { return plane.Row(y); }
  JXL_INLINE const pixel_type* Row(const size_t y) const {
    return plane.Row(y);
  }

 private:
  Channel(jxl::Plane<pixel_type>&& p, size_t iw, size_t ih, int hsh, int vsh)
      : plane(std::move(p)), w(iw), h(ih), hshift(hsh), vshift(vsh) {}
};

class Transform;

class Image {
 public:
  // image data, transforms can dramatically change the number of channels and
  // their semantics
  std::vector<Channel> channel;
  // transforms that have been applied (and that have to be undone)
  std::vector<Transform> transform;

  // image dimensions (channels may have different dimensions due to transforms)
  size_t w, h;
  int bitdepth;
  size_t nb_meta_channels;  // first few channels might contain palette(s)
  bool error;               // true if a fatal error occurred, false otherwise

  explicit Image(JxlMemoryManager* memory_manager);

  Image(const Image& other) = delete;
  Image& operator=(const Image& other) = delete;

  Image& operator=(Image&& other) noexcept;
  Image(Image&& other) noexcept = default;

  static StatusOr<Image> Create(JxlMemoryManager* memory_manager, size_t iw,
                                size_t ih, int bitdepth, int nb_chans);

  JxlMemoryManager* memory_manager() const { return memory_manager_; }

  bool empty() const {
    for (const auto& ch : channel) {
      if (ch.w && ch.h) return false;
    }
    return true;
  }

  static StatusOr<Image> Clone(const Image& that);

  void undo_transforms(const weighted::Header& wp_header,
                       jxl::ThreadPool* pool = nullptr);

  std::string DebugString() const;

 private:
  Image(JxlMemoryManager* memory_manager, size_t iw, size_t ih, int bitdepth);
  JxlMemoryManager* memory_manager_;
};

}  // namespace jxl

#endif  // LIB_JXL_MODULAR_MODULAR_IMAGE_H_

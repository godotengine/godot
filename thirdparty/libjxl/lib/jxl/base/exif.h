// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_BASE_EXIF_H_
#define LIB_JXL_BASE_EXIF_H_

// Basic parsing of Exif (just enough for the render-impacting things
// like orientation)

#include <jxl/codestream_header.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "lib/jxl/base/byte_order.h"
#include "lib/jxl/base/compiler_specific.h"

namespace jxl {

constexpr uint16_t kExifOrientationTag = 274;

// Checks if a blob looks like Exif, and if so, sets bigendian
// according to the tiff endianness
JXL_INLINE bool IsExif(const std::vector<uint8_t>& exif, bool* bigendian) {
  if (exif.size() < 12) return false;  // not enough bytes for a valid exif blob
  const uint8_t* t = exif.data();
  if (LoadLE32(t) == 0x2A004D4D) {
    *bigendian = true;
    return true;
  } else if (LoadLE32(t) == 0x002A4949) {
    *bigendian = false;
    return true;
  }
  return false;  // not a valid tiff header
}

// Finds the position of an Exif tag, or 0 if it is not found
JXL_INLINE size_t FindExifTagPosition(const std::vector<uint8_t>& exif,
                                      uint16_t tagname) {
  bool bigendian;
  if (!IsExif(exif, &bigendian)) return 0;
  const uint8_t* t = exif.data() + 4;
  uint64_t offset = (bigendian ? LoadBE32(t) : LoadLE32(t));
  if (exif.size() < 12 + offset + 2 || offset < 8) return 0;
  t += offset - 4;
  if (offset + 2 >= exif.size()) return 0;
  uint16_t nb_tags = (bigendian ? LoadBE16(t) : LoadLE16(t));
  t += 2;
  while (nb_tags > 0) {
    if (t + 12 >= exif.data() + exif.size()) return 0;
    uint16_t tag = (bigendian ? LoadBE16(t) : LoadLE16(t));
    t += 2;
    if (tag == tagname) return static_cast<size_t>(t - exif.data());
    t += 10;
    nb_tags--;
  }
  return 0;
}

// TODO(jon): tag 1 can be used to represent Adobe RGB 1998 if it has value
// "R03"
// TODO(jon): set intrinsic dimensions according to
// https://discourse.wicg.io/t/proposal-exif-image-resolution-auto-and-from-image/4326/24
// Parses the Exif data just enough to extract any render-impacting info.
// If the Exif data is invalid or could not be parsed, then it is treated
// as a no-op.
JXL_INLINE void InterpretExif(const std::vector<uint8_t>& exif,
                              JxlOrientation* orientation) {
  bool bigendian;
  if (!IsExif(exif, &bigendian)) return;
  size_t o_pos = FindExifTagPosition(exif, kExifOrientationTag);
  if (o_pos) {
    const uint8_t* t = exif.data() + o_pos;
    uint16_t type = (bigendian ? LoadBE16(t) : LoadLE16(t));
    t += 2;
    uint32_t count = (bigendian ? LoadBE32(t) : LoadLE32(t));
    t += 4;
    uint16_t value = (bigendian ? LoadBE16(t) : LoadLE16(t));
    if (type == 3 && count == 1 && value >= 1 && value <= 8) {
      *orientation = static_cast<JxlOrientation>(value);
    }
  }
}

}  // namespace jxl

#endif  // LIB_JXL_BASE_EXIF_H_

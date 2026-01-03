/*
 *  Copyright 2012 The LibYuv Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS. All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "libyuv/mjpeg_decoder.h"

#include <string.h>  // For memchr.

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

// Helper function to scan for EOI marker (0xff 0xd9).
static LIBYUV_BOOL ScanEOI(const uint8_t* src_mjpg, size_t src_size_mjpg) {
  if (src_size_mjpg >= 2) {
    const uint8_t* end = src_mjpg + src_size_mjpg - 1;
    const uint8_t* it = src_mjpg;
    while (it < end) {
      // TODO(fbarchard): scan for 0xd9 instead.
      it = (const uint8_t*)(memchr(it, 0xff, end - it));
      if (it == NULL) {
        break;
      }
      if (it[1] == 0xd9) {
        return LIBYUV_TRUE;  // Success: Valid jpeg.
      }
      ++it;  // Skip over current 0xff.
    }
  }
  // ERROR: Invalid jpeg end code not found. Size src_size_mjpg
  return LIBYUV_FALSE;
}

// Helper function to validate the jpeg appears intact.
LIBYUV_BOOL ValidateJpeg(const uint8_t* src_mjpg, size_t src_size_mjpg) {
  // Maximum size that ValidateJpeg will consider valid.
  const size_t kMaxJpegSize = 0x7fffffffull;
  const size_t kBackSearchSize = 1024;
  if (src_size_mjpg < 64 || src_size_mjpg > kMaxJpegSize || !src_mjpg) {
    // ERROR: Invalid jpeg size: src_size_mjpg
    return LIBYUV_FALSE;
  }
  // SOI marker
  if (src_mjpg[0] != 0xff || src_mjpg[1] != 0xd8 || src_mjpg[2] != 0xff) {
    // ERROR: Invalid jpeg initial start code
    return LIBYUV_FALSE;
  }

  // Look for the End Of Image (EOI) marker near the end of the buffer.
  if (src_size_mjpg > kBackSearchSize) {
    if (ScanEOI(src_mjpg + src_size_mjpg - kBackSearchSize, kBackSearchSize)) {
      return LIBYUV_TRUE;  // Success: Valid jpeg.
    }
    // Reduce search size for forward search.
    src_size_mjpg = src_size_mjpg - kBackSearchSize + 1;
  }
  // Step over SOI marker and scan for EOI.
  return ScanEOI(src_mjpg + 2, src_size_mjpg - 2);
}

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif

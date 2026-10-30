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
static LIBYUV_BOOL ScanEOI(const uint8_t* sample, size_t sample_size) {
  if (sample_size >= 2) {
    const uint8_t* end = sample + sample_size - 1;
    const uint8_t* it = sample;
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
  // ERROR: Invalid jpeg end code not found. Size sample_size
  return LIBYUV_FALSE;
}

// Helper function to validate the jpeg appears intact.
LIBYUV_BOOL ValidateJpeg(const uint8_t* sample, size_t sample_size) {
  // Maximum size that ValidateJpeg will consider valid.
  const size_t kMaxJpegSize = 0x7fffffffull;
  const size_t kBackSearchSize = 1024;
  if (sample_size < 64 || sample_size > kMaxJpegSize || !sample) {
    // ERROR: Invalid jpeg size: sample_size
    return LIBYUV_FALSE;
  }
  if (sample[0] != 0xff || sample[1] != 0xd8) {  // SOI marker
    // ERROR: Invalid jpeg initial start code
    return LIBYUV_FALSE;
  }

  // Look for the End Of Image (EOI) marker near the end of the buffer.
  if (sample_size > kBackSearchSize) {
    if (ScanEOI(sample + sample_size - kBackSearchSize, kBackSearchSize)) {
      return LIBYUV_TRUE;  // Success: Valid jpeg.
    }
    // Reduce search size for forward search.
    sample_size = sample_size - kBackSearchSize + 1;
  }
  // Step over SOI marker and scan for EOI.
  return ScanEOI(sample + 2, sample_size - 2);
}

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif

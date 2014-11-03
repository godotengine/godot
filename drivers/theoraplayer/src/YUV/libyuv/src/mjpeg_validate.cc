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

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

// Helper function to validate the jpeg appears intact.
// TODO(fbarchard): Optimize case where SOI is found but EOI is not.
LIBYUV_BOOL ValidateJpeg(const uint8* sample, size_t sample_size) {
  size_t i;
  if (sample_size < 64) {
    // ERROR: Invalid jpeg size: sample_size
    return LIBYUV_FALSE;
  }
  if (sample[0] != 0xff || sample[1] != 0xd8) {  // Start Of Image
    // ERROR: Invalid jpeg initial start code
    return LIBYUV_FALSE;
  }
  for (i = sample_size - 2; i > 1;) {
    if (sample[i] != 0xd9) {
      if (sample[i] == 0xff && sample[i + 1] == 0xd9) {  // End Of Image
        return LIBYUV_TRUE;  // Success: Valid jpeg.
      }
      --i;
    }
    --i;
  }
  // ERROR: Invalid jpeg end code not found. Size sample_size
  return LIBYUV_FALSE;
}

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif


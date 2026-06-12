/*
 *  Copyright (c) 2013 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VPX_DSP_BITWRITER_BUFFER_H_
#define VPX_VPX_DSP_BITWRITER_BUFFER_H_

#include "vpx/vpx_integer.h"

#ifdef __cplusplus
extern "C" {
#endif

struct vpx_write_bit_buffer {
  // Whether there has been an error.
  int error;
  // We maintain the invariant that bit_offset <= size * CHAR_BIT, i.e., we
  // never write beyond the end of bit_buffer. If bit_offset would be
  // incremented to be greater than size * CHAR_BIT, leave bit_offset unchanged
  // and set error to 1.
  size_t bit_offset;
  // Size of bit_buffer in bytes.
  size_t size;
  uint8_t *bit_buffer;
};

void vpx_wb_init(struct vpx_write_bit_buffer *wb, uint8_t *bit_buffer,
                 size_t size);

int vpx_wb_has_error(const struct vpx_write_bit_buffer *wb);

// Must not be called if vpx_wb_has_error(wb) returns true.
size_t vpx_wb_bytes_written(const struct vpx_write_bit_buffer *wb);

void vpx_wb_write_bit(struct vpx_write_bit_buffer *wb, int bit);

void vpx_wb_write_literal(struct vpx_write_bit_buffer *wb, int data, int bits);

void vpx_wb_write_inv_signed_literal(struct vpx_write_bit_buffer *wb, int data,
                                     int bits);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VPX_DSP_BITWRITER_BUFFER_H_

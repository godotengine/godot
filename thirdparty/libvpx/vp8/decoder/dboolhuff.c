/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "dboolhuff.h"
#include "vp8/common/common.h"
#include "vpx_dsp/vpx_dsp_common.h"

int vp8dx_start_decode(BOOL_DECODER *br, const unsigned char *source,
                       unsigned int source_sz, vpx_decrypt_cb decrypt_cb,
                       void *decrypt_state) {
  if (source_sz && !source) return 1;

  // To simplify calling code this fuction can be called with |source| == null
  // and |source_sz| == 0. This and vp8dx_bool_decoder_fill() are essentially
  // no-ops in this case.
  // Work around a ubsan warning with a ternary to avoid adding 0 to null.
  br->user_buffer_end = source ? source + source_sz : source;
  br->user_buffer = source;
  br->value = 0;
  br->count = -8;
  br->range = 255;
  br->decrypt_cb = decrypt_cb;
  br->decrypt_state = decrypt_state;

  /* Populate the buffer */
  vp8dx_bool_decoder_fill(br);

  return 0;
}

void vp8dx_bool_decoder_fill(BOOL_DECODER *br) {
  const unsigned char *bufptr = br->user_buffer;
  VP8_BD_VALUE value = br->value;
  int count = br->count;
  int shift = VP8_BD_VALUE_SIZE - CHAR_BIT - (count + CHAR_BIT);
  size_t bytes_left = br->user_buffer_end - bufptr;
  size_t bits_left = bytes_left * CHAR_BIT;
  int x = shift + CHAR_BIT - (int)bits_left;
  int loop_end = 0;
  unsigned char decrypted[sizeof(VP8_BD_VALUE) + 1];

  if (br->decrypt_cb) {
    size_t n = VPXMIN(sizeof(decrypted), bytes_left);
    br->decrypt_cb(br->decrypt_state, bufptr, decrypted, (int)n);
    bufptr = decrypted;
  }

  if (x >= 0) {
    count += VP8_LOTS_OF_BITS;
    loop_end = x;
  }

  if (x < 0 || bits_left) {
    while (shift >= loop_end) {
      count += CHAR_BIT;
      value |= (VP8_BD_VALUE)*bufptr << shift;
      ++bufptr;
      ++br->user_buffer;
      shift -= CHAR_BIT;
    }
  }

  br->value = value;
  br->count = count;
}

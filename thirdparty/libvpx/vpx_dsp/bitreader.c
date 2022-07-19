/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */
#include <stdlib.h>

#include "./vpx_config.h"

#include "vpx_dsp/bitreader.h"
#include "vpx_dsp/prob.h"
#include "vpx_dsp/vpx_dsp_common.h"
#include "vpx_ports/mem.h"
#include "vpx_mem/vpx_mem.h"
#include "vpx_util/endian_inl.h"

int vpx_reader_init(vpx_reader *r,
                    const uint8_t *buffer,
                    size_t size,
                    vpx_decrypt_cb decrypt_cb,
                    void *decrypt_state) {
  if (size && !buffer) {
    return 1;
  } else {
    r->buffer_end = buffer + size;
    r->buffer = buffer;
    r->value = 0;
    r->count = -8;
    r->range = 255;
    r->decrypt_cb = decrypt_cb;
    r->decrypt_state = decrypt_state;
    vpx_reader_fill(r);
    return vpx_read_bit(r) != 0;  // marker bit
  }
}

void vpx_reader_fill(vpx_reader *r) {
  const uint8_t *const buffer_end = r->buffer_end;
  const uint8_t *buffer = r->buffer;
  const uint8_t *buffer_start = buffer;
  BD_VALUE value = r->value;
  int count = r->count;
  const size_t bytes_left = buffer_end - buffer;
  const size_t bits_left = bytes_left * CHAR_BIT;
  int shift = BD_VALUE_SIZE - CHAR_BIT - (count + CHAR_BIT);

  if (r->decrypt_cb) {
    size_t n = VPXMIN(sizeof(r->clear_buffer), bytes_left);
    r->decrypt_cb(r->decrypt_state, buffer, r->clear_buffer, (int)n);
    buffer = r->clear_buffer;
    buffer_start = r->clear_buffer;
  }
  if (bits_left > BD_VALUE_SIZE) {
      const int bits = (shift & 0xfffffff8) + CHAR_BIT;
      BD_VALUE nv;
      BD_VALUE big_endian_values;
      memcpy(&big_endian_values, buffer, sizeof(BD_VALUE));
#if SIZE_MAX == 0xffffffffffffffffULL
        big_endian_values = HToBE64(big_endian_values);
#else
        big_endian_values = HToBE32(big_endian_values);
#endif
      nv = big_endian_values >> (BD_VALUE_SIZE - bits);
      count += bits;
      buffer += (bits >> 3);
      value = r->value | (nv << (shift & 0x7));
  } else {
    const int bits_over = (int)(shift + CHAR_BIT - (int)bits_left);
    int loop_end = 0;
    if (bits_over >= 0) {
      count += LOTS_OF_BITS;
      loop_end = bits_over;
    }

    if (bits_over < 0 || bits_left) {
      while (shift >= loop_end) {
        count += CHAR_BIT;
        value |= (BD_VALUE)*buffer++ << shift;
        shift -= CHAR_BIT;
      }
    }
  }

  // NOTE: Variable 'buffer' may not relate to 'r->buffer' after decryption,
  // so we increase 'r->buffer' by the amount that 'buffer' moved, rather than
  // assign 'buffer' to 'r->buffer'.
  r->buffer += buffer - buffer_start;
  r->value = value;
  r->count = count;
}

const uint8_t *vpx_reader_find_end(vpx_reader *r) {
  // Find the end of the coded buffer
  while (r->count > CHAR_BIT && r->count < BD_VALUE_SIZE) {
    r->count -= CHAR_BIT;
    r->buffer--;
  }
  return r->buffer;
}

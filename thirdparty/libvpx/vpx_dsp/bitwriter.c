/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <assert.h>
#include <limits.h>

#include "./bitwriter.h"

#if CONFIG_BITSTREAM_DEBUG
#include "vpx_util/vpx_debug_util.h"
#endif

void vpx_start_encode(vpx_writer *br, uint8_t *source, size_t size) {
  br->lowvalue = 0;
  br->range = 255;
  br->count = -24;
  br->error = 0;
  br->pos = 0;
  // Make sure it is safe to cast br->pos to int in vpx_write().
  if (size > INT_MAX) size = INT_MAX;
  br->size = (unsigned int)size;
  br->buffer = source;
  vpx_write_bit(br, 0);
}

int vpx_stop_encode(vpx_writer *br) {
  int i;

#if CONFIG_BITSTREAM_DEBUG
  bitstream_queue_set_skip_write(1);
#endif
  for (i = 0; i < 32; i++) vpx_write_bit(br, 0);

  // Ensure there's no ambigous collision with any index marker bytes
  if (!br->error && (br->buffer[br->pos - 1] & 0xe0) == 0xc0) {
    if (br->pos < br->size) {
      br->buffer[br->pos++] = 0;
    } else {
      br->error = 1;
    }
  }

#if CONFIG_BITSTREAM_DEBUG
  bitstream_queue_set_skip_write(0);
#endif

  return br->error ? -1 : 0;
}

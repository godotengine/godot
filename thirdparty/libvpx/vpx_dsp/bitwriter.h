/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VPX_VPX_DSP_BITWRITER_H_
#define VPX_VPX_DSP_BITWRITER_H_

#include <stdio.h>

#include "vpx_ports/compiler_attributes.h"
#include "vpx_ports/mem.h"

#include "vpx_dsp/prob.h"
#if CONFIG_BITSTREAM_DEBUG
#include "vpx_util/vpx_debug_util.h"
#endif  // CONFIG_BITSTREAM_DEBUG

#ifdef __cplusplus
extern "C" {
#endif

typedef struct vpx_writer {
  unsigned int lowvalue;
  unsigned int range;
  int count;
  // Whether there has been an error.
  int error;
  // We maintain the invariant that pos <= size, i.e., we never write beyond
  // the end of the buffer. If pos would be incremented to be greater than
  // size, leave pos unchanged and set error to 1.
  unsigned int pos;
  unsigned int size;
  uint8_t *buffer;
} vpx_writer;

void vpx_start_encode(vpx_writer *br, uint8_t *source, size_t size);
// Returns 0 on success and returns -1 in case of error.
int vpx_stop_encode(vpx_writer *br);

static INLINE VPX_NO_UNSIGNED_SHIFT_CHECK void vpx_write(vpx_writer *br,
                                                         int bit,
                                                         int probability) {
  unsigned int split;
  int count = br->count;
  unsigned int range = br->range;
  unsigned int lowvalue = br->lowvalue;
  int shift;

#if CONFIG_BITSTREAM_DEBUG
  /*
  int queue_r = 0;
  int frame_idx_r = 0;
  int queue_w = bitstream_queue_get_write();
  int frame_idx_w = bitstream_queue_get_frame_write();
  if (frame_idx_w == frame_idx_r && queue_w == queue_r) {
    fprintf(stderr, "\n *** bitstream queue at frame_idx_w %d queue_w %d\n",
            frame_idx_w, queue_w);
    assert(0);
  }
  */
  bitstream_queue_push(bit, probability);
#endif

  split = 1 + (((range - 1) * probability) >> 8);

  range = split;

  if (bit) {
    lowvalue += split;
    range = br->range - split;
  }

  shift = vpx_norm[range];

  range <<= shift;
  count += shift;

  if (count >= 0) {
    int offset = shift - count;

    if (!br->error) {
      if ((lowvalue << (offset - 1)) & 0x80000000) {
        int x = (int)br->pos - 1;

        while (x >= 0 && br->buffer[x] == 0xff) {
          br->buffer[x] = 0;
          x--;
        }

        // TODO(wtc): How to prove x >= 0?
        br->buffer[x] += 1;
      }

      if (br->pos < br->size) {
        br->buffer[br->pos++] = (lowvalue >> (24 - offset)) & 0xff;
      } else {
        br->error = 1;
      }
    }
    lowvalue <<= offset;
    shift = count;
    lowvalue &= 0xffffff;
    count -= 8;
  }

  lowvalue <<= shift;
  br->count = count;
  br->lowvalue = lowvalue;
  br->range = range;
}

static INLINE void vpx_write_bit(vpx_writer *w, int bit) {
  vpx_write(w, bit, 128);  // vpx_prob_half
}

static INLINE void vpx_write_literal(vpx_writer *w, int data, int bits) {
  int bit;

  for (bit = bits - 1; bit >= 0; bit--) vpx_write_bit(w, 1 & (data >> bit));
}

#define vpx_write_prob(w, v) vpx_write_literal((w), (v), 8)

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VPX_DSP_BITWRITER_H_

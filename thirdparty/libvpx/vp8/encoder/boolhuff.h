/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

/****************************************************************************
 *
 *   Module Title :     boolhuff.h
 *
 *   Description  :     Bool Coder header file.
 *
 ****************************************************************************/
#ifndef VPX_VP8_ENCODER_BOOLHUFF_H_
#define VPX_VP8_ENCODER_BOOLHUFF_H_

#include "vpx_ports/mem.h"
#include "vpx/internal/vpx_codec_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  unsigned int lowvalue;
  unsigned int range;
  int count;
  unsigned int pos;
  unsigned char *buffer;
  unsigned char *buffer_end;
  struct vpx_internal_error_info *error;
} BOOL_CODER;

void vp8_start_encode(BOOL_CODER *bc, unsigned char *source,
                      unsigned char *source_end);

void vp8_encode_value(BOOL_CODER *bc, int data, int bits);
void vp8_stop_encode(BOOL_CODER *bc);
extern const unsigned int vp8_prob_cost[256];

DECLARE_ALIGNED(16, extern const unsigned char, vp8_norm[256]);

static int validate_buffer(const unsigned char *start, size_t len,
                           const unsigned char *end,
                           struct vpx_internal_error_info *error) {
  if (start + len > start && start + len < end) {
    return 1;
  } else {
    vpx_internal_error(error, VPX_CODEC_CORRUPT_FRAME,
                       "Truncated packet or corrupt partition ");
  }

  return 0;
}
static void vp8_encode_bool(BOOL_CODER *bc, int bit, int probability) {
  unsigned int split;
  int count = bc->count;
  unsigned int range = bc->range;
  unsigned int lowvalue = bc->lowvalue;
  int shift;

  split = 1 + (((range - 1) * probability) >> 8);

  range = split;

  if (bit) {
    lowvalue += split;
    range = bc->range - split;
  }

  shift = vp8_norm[range];

  range <<= shift;
  count += shift;

  if (count >= 0) {
    int offset = shift - count;

    if ((lowvalue << (offset - 1)) & 0x80000000) {
      int x = bc->pos - 1;

      while (x >= 0 && bc->buffer[x] == 0xff) {
        bc->buffer[x] = (unsigned char)0;
        x--;
      }

      bc->buffer[x] += 1;
    }

    validate_buffer(bc->buffer + bc->pos, 1, bc->buffer_end, bc->error);
    bc->buffer[bc->pos++] = (lowvalue >> (24 - offset) & 0xff);

    shift = count;
    lowvalue = (int)(((uint64_t)lowvalue << offset) & 0xffffff);
    count -= 8;
  }

  lowvalue <<= shift;
  bc->count = count;
  bc->lowvalue = lowvalue;
  bc->range = range;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP8_ENCODER_BOOLHUFF_H_

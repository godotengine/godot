/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "vpx_config.h"
#include "vp8_rtcd.h"

void vp8_idct_dequant_0_2x_sse2(short *q, short *dq, unsigned char *dst,
                                int dst_stride);
void vp8_idct_dequant_full_2x_sse2(short *q, short *dq, unsigned char *dst,
                                   int dst_stride);

void vp8_dequant_idct_add_y_block_sse2(short *q, short *dq, unsigned char *dst,
                                       int stride, char *eobs) {
  int i;

  for (i = 0; i < 4; ++i) {
    if (((short *)(eobs))[0]) {
      if (((short *)(eobs))[0] & 0xfefe) {
        vp8_idct_dequant_full_2x_sse2(q, dq, dst, stride);
      } else {
        vp8_idct_dequant_0_2x_sse2(q, dq, dst, stride);
      }
    }
    if (((short *)(eobs))[1]) {
      if (((short *)(eobs))[1] & 0xfefe) {
        vp8_idct_dequant_full_2x_sse2(q + 32, dq, dst + 8, stride);
      } else {
        vp8_idct_dequant_0_2x_sse2(q + 32, dq, dst + 8, stride);
      }
    }
    q += 64;
    dst += stride * 4;
    eobs += 4;
  }
}

void vp8_dequant_idct_add_uv_block_sse2(short *q, short *dq,
                                        unsigned char *dst_u,
                                        unsigned char *dst_v, int stride,
                                        char *eobs) {
  if (((short *)(eobs))[0]) {
    if (((short *)(eobs))[0] & 0xfefe) {
      vp8_idct_dequant_full_2x_sse2(q, dq, dst_u, stride);
    } else {
      vp8_idct_dequant_0_2x_sse2(q, dq, dst_u, stride);
    }
  }
  q += 32;
  dst_u += stride * 4;

  if (((short *)(eobs))[1]) {
    if (((short *)(eobs))[1] & 0xfefe) {
      vp8_idct_dequant_full_2x_sse2(q, dq, dst_u, stride);
    } else {
      vp8_idct_dequant_0_2x_sse2(q, dq, dst_u, stride);
    }
  }
  q += 32;

  if (((short *)(eobs))[2]) {
    if (((short *)(eobs))[2] & 0xfefe) {
      vp8_idct_dequant_full_2x_sse2(q, dq, dst_v, stride);
    } else {
      vp8_idct_dequant_0_2x_sse2(q, dq, dst_v, stride);
    }
  }
  q += 32;
  dst_v += stride * 4;

  if (((short *)(eobs))[3]) {
    if (((short *)(eobs))[3] & 0xfefe) {
      vp8_idct_dequant_full_2x_sse2(q, dq, dst_v, stride);
    } else {
      vp8_idct_dequant_0_2x_sse2(q, dq, dst_v, stride);
    }
  }
}

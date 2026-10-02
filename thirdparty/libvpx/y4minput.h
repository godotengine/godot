/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 *
 *  Based on code from the OggTheora software codec source code,
 *  Copyright (C) 2002-2010 The Xiph.Org Foundation and contributors.
 */

#ifndef VPX_Y4MINPUT_H_
#define VPX_Y4MINPUT_H_

#include <stdio.h>
#include "vpx/vpx_image.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct y4m_input y4m_input;

/*The function used to perform chroma conversion.*/
typedef void (*y4m_convert_func)(y4m_input *_y4m, unsigned char *_dst,
                                 unsigned char *_src);

struct y4m_input {
  int pic_w;
  int pic_h;
  int fps_n;
  int fps_d;
  int par_n;
  int par_d;
  char interlace;
  int src_c_dec_h;
  int src_c_dec_v;
  int dst_c_dec_h;
  int dst_c_dec_v;
  char chroma_type[16];
  /*The size of each converted frame buffer.*/
  size_t dst_buf_sz;
  /*The amount to read directly into the converted frame buffer.*/
  size_t dst_buf_read_sz;
  /*The size of the auxilliary buffer.*/
  size_t aux_buf_sz;
  /*The amount to read into the auxilliary buffer.*/
  size_t aux_buf_read_sz;
  y4m_convert_func convert;
  unsigned char *dst_buf;
  unsigned char *aux_buf;
  enum vpx_img_fmt vpx_fmt;
  int bps;
  unsigned int bit_depth;
};

/**
 * Open the input file, treating it as Y4M. |y4m_ctx| is filled in after
 * reading it. The |skip_buffer| indicates bytes that were previously read
 * from |file|, to do input-type detection; this buffer will be read before
 * the |file| is read. It is of size |num_skip|, which *must* be 8 or less.
 *
 * Returns 0 on success, -1 on failure.
 */
int y4m_input_open(y4m_input *y4m_ctx, FILE *file, char *skip_buffer,
                   int num_skip, int only_420);
void y4m_input_close(y4m_input *_y4m);
int y4m_input_fetch_frame(y4m_input *_y4m, FILE *_fin, vpx_image_t *img);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_Y4MINPUT_H_

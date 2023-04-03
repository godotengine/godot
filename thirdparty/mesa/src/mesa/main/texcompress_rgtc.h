/*
 * Copyright (C) 2011 Red Hat Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef TEXCOMPRESS_RGTC_H
#define TEXCOMPRESS_RGTC_H

#include "util/glheader.h"
#include "texstore.h"


extern GLboolean
_mesa_texstore_red_rgtc1(TEXSTORE_PARAMS);

extern GLboolean
_mesa_texstore_signed_red_rgtc1(TEXSTORE_PARAMS);

extern GLboolean
_mesa_texstore_rg_rgtc2(TEXSTORE_PARAMS);

extern GLboolean
_mesa_texstore_signed_rg_rgtc2(TEXSTORE_PARAMS);

extern compressed_fetch_func
_mesa_get_compressed_rgtc_func(mesa_format format);

extern void
_mesa_unpack_rgtc(uint8_t *dst_row,
                  unsigned dst_stride,
                  const uint8_t *src_row,
                  unsigned src_stride,
                  unsigned src_width,
                  unsigned src_height,
                  mesa_format format);

#endif

/*
 * Mesa 3-D graphics library
 *
 * Copyright (C) 1999-2008  Brian Paul   All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef TEXCOMPRESS_S3TC_H
#define TEXCOMPRESS_S3TC_H

#include "util/glheader.h"
#include "texstore.h"
#include "texcompress.h"

struct gl_context;

extern GLboolean
_mesa_texstore_rgb_dxt1(TEXSTORE_PARAMS);

extern GLboolean
_mesa_texstore_rgba_dxt1(TEXSTORE_PARAMS);

extern GLboolean
_mesa_texstore_rgba_dxt3(TEXSTORE_PARAMS);

extern GLboolean
_mesa_texstore_rgba_dxt5(TEXSTORE_PARAMS);


extern compressed_fetch_func
_mesa_get_dxt_fetch_func(mesa_format format);

extern void
_mesa_unpack_s3tc(uint8_t *dst_row,
                  unsigned dst_stride,
                  const uint8_t *src_row,
                  unsigned src_stride,
                  unsigned src_width,
                  unsigned src_height,
                  mesa_format format);

#endif /* TEXCOMPRESS_S3TC_H */

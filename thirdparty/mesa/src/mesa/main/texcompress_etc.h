/*
 * Copyright (C) 2011 LunarG, Inc.
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

#ifndef TEXCOMPRESS_ETC1_H
#define TEXCOMPRESS_ETC1_H

#include <inttypes.h>
#include "util/glheader.h"
#include "texcompress.h"
#include "texstore.h"


GLboolean
_mesa_texstore_etc1_rgb8(TEXSTORE_PARAMS);

GLboolean
_mesa_texstore_etc2_rgb8(TEXSTORE_PARAMS);

GLboolean
_mesa_texstore_etc2_srgb8(TEXSTORE_PARAMS);

GLboolean
_mesa_texstore_etc2_rgba8_eac(TEXSTORE_PARAMS);

GLboolean
_mesa_texstore_etc2_srgb8_alpha8_eac(TEXSTORE_PARAMS);

GLboolean
_mesa_texstore_etc2_r11_eac(TEXSTORE_PARAMS);

GLboolean
_mesa_texstore_etc2_rg11_eac(TEXSTORE_PARAMS);

GLboolean
_mesa_texstore_etc2_signed_r11_eac(TEXSTORE_PARAMS);

GLboolean
_mesa_texstore_etc2_signed_rg11_eac(TEXSTORE_PARAMS);

GLboolean
_mesa_texstore_etc2_rgb8_punchthrough_alpha1(TEXSTORE_PARAMS);

GLboolean
_mesa_texstore_etc2_srgb8_punchthrough_alpha1(TEXSTORE_PARAMS);

void
_mesa_etc1_unpack_rgba8888(uint8_t *dst_row,
                           unsigned dst_stride,
                           const uint8_t *src_row,
                           unsigned src_stride,
                           unsigned src_width,
                           unsigned src_height);
void
_mesa_unpack_etc2_format(uint8_t *dst_row,
                         unsigned dst_stride,
                         const uint8_t *src_row,
                         unsigned src_stride,
                         unsigned src_width,
                         unsigned src_height,
			 mesa_format format,
			 bool bgra);

compressed_fetch_func
_mesa_get_etc_fetch_func(mesa_format format);

#endif

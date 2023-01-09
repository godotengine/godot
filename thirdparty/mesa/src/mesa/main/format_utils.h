/**
 * \file format_utils.h
 * A collection of format conversion utility functions.
 */

/*
 * Mesa 3-D graphics library
 *
 * Copyright (C) 1999-2006  Brian Paul  All Rights Reserved.
 * Copyright (C) 2014  Intel Corporation  All Rights Reserved.
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

#ifndef FORMAT_UTILS_H
#define FORMAT_UTILS_H

#include "formats.h"

#include "macros.h"
#include "util/rounding.h"
#include "util/half_float.h"
#include "util/format/format_utils.h"

extern const mesa_array_format RGBA32_FLOAT;
extern const mesa_array_format RGBA8_UBYTE;
extern const mesa_array_format RGBA32_UINT;
extern const mesa_array_format RGBA32_INT;

bool
_mesa_format_to_array(mesa_format, GLenum *type, int *num_components,
                      uint8_t swizzle[4], bool *normalized);

void
_mesa_swizzle_and_convert(void *dst,
                          enum mesa_array_format_datatype dst_type,
                          int num_dst_channels,
                          const void *src,
                          enum mesa_array_format_datatype src_type,
                          int num_src_channels,
                          const uint8_t swizzle[4], bool normalized, int count);

bool
_mesa_compute_rgba2base2rgba_component_mapping(GLenum baseFormat, uint8_t *map);

void
_mesa_format_convert(void *void_dst, uint32_t dst_format, size_t dst_stride,
                     void *void_src, uint32_t src_format, size_t src_stride,
                     size_t width, size_t height, uint8_t *rebase_swizzle);

#endif

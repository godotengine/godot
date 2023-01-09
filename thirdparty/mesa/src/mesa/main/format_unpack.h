/*
 * Mesa 3-D graphics library
 *
 * Copyright (c) 2011 VMware, Inc.
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

#ifndef FORMAT_UNPACK_H
#define FORMAT_UNPACK_H

#include "util/format/u_format.h"
#include "formats.h"

#ifdef __cplusplus
extern "C" {
#endif

static inline void
_mesa_unpack_rgba_row(mesa_format format, uint32_t n,
                      const void *src, float dst[][4])
{
   util_format_unpack_rgba(format, dst, src, n);
}

extern void
_mesa_unpack_ubyte_rgba_row(mesa_format format, uint32_t n,
                            const void *src, uint8_t dst[][4]);

static inline void
_mesa_unpack_uint_rgba_row(mesa_format format, uint32_t n,
                           const void *src, uint32_t dst[][4])
{
   util_format_unpack_rgba(format, dst, src, n);
}

static inline void
_mesa_unpack_float_z_row(mesa_format format, uint32_t n,
                         const void *src, float *dst)
{
   util_format_unpack_z_float((enum pipe_format)format, dst, src, n);
}


static inline void
_mesa_unpack_uint_z_row(mesa_format format, uint32_t n,
                        const void *src, uint32_t *dst)
{
   util_format_unpack_z_32unorm((enum pipe_format)format, dst, src, n);
}

static inline void
_mesa_unpack_ubyte_stencil_row(mesa_format format, uint32_t n,
                               const void *src, uint8_t *dst)
{
   util_format_unpack_s_8uint((enum pipe_format)format, dst, src, n);
}

void
_mesa_unpack_uint_24_8_depth_stencil_row(mesa_format format, uint32_t n,
                                         const void *src, uint32_t *dst);

void
_mesa_unpack_float_32_uint_24_8_depth_stencil_row(mesa_format format,
                                                  uint32_t n,
                                                  const void *src,
                                                  uint32_t *dst);

#ifdef __cplusplus
}
#endif

#endif /* FORMAT_UNPACK_H */

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


#ifndef FORMAT_PACK_H
#define FORMAT_PACK_H


#include "util/format/u_format.h"
#include "formats.h"

#ifdef __cplusplus
extern "C" {
#endif

static inline void
_mesa_pack_float_rgba_row(mesa_format format, uint32_t n,
                          const float src[][4], void *dst)
{
   util_format_pack_rgba(format, dst, src, n);
}

static inline void
_mesa_pack_ubyte_rgba_row(mesa_format format, uint32_t n,
                          const uint8_t *src, void *dst)
{
   const struct util_format_pack_description *pack = util_format_pack_description(format);
   pack->pack_rgba_8unorm((uint8_t *)dst, 0, src, 0, n, 1);
}

static inline void
_mesa_pack_uint_rgba_row(mesa_format format, uint32_t n,
                         const uint32_t src[][4], void *dst)
{
   util_format_pack_rgba(format, dst, src, n);
}

static inline void
_mesa_pack_float_z_row(mesa_format format, uint32_t n,
                       const float *src, void *dst)
{
   util_format_pack_z_float(format, dst, src, n);
}

static inline void
_mesa_pack_uint_z_row(mesa_format format, uint32_t n,
                      const uint32_t *src, void *dst)
{
   util_format_pack_z_32unorm(format, dst, src, n);
}

static inline void
_mesa_pack_ubyte_stencil_row(mesa_format format, uint32_t n,
                             const uint8_t *src, void *dst)
{
   util_format_pack_s_8uint(format, dst, src, n);
}

#ifdef __cplusplus
}
#endif

#endif

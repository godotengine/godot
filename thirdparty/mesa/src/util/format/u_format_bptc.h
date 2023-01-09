/**************************************************************************
 *
 * Copyright 2010 VMware, Inc.
 * All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sub license, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL
 * THE COPYRIGHT HOLDERS, AUTHORS AND/OR ITS SUPPLIERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
 * USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial portions
 * of the Software.
 *
 **************************************************************************/


#ifndef U_FORMAT_BPTC_H_
#define U_FORMAT_BPTC_H_


#include "pipe/p_compiler.h"

#include "c99_compat.h"

#ifdef __cplusplus
extern "C" {
#endif

void
util_format_bptc_rgba_unorm_unpack_rgba_8unorm(uint8_t *restrict dst_row, unsigned dst_stride,
                                               const uint8_t *restrict src_row, unsigned src_stride,
                                               unsigned width, unsigned height);
void
util_format_bptc_rgba_unorm_pack_rgba_8unorm(uint8_t *restrict dst_row, unsigned dst_stride,
                                             const uint8_t *restrict src_row, unsigned src_stride,
                                             unsigned width, unsigned height);
void
util_format_bptc_rgba_unorm_unpack_rgba_float(void *restrict dst_row, unsigned dst_stride,
                                              const uint8_t *restrict src_row, unsigned src_stride,
                                              unsigned width, unsigned height);
void
util_format_bptc_rgba_unorm_pack_rgba_float(uint8_t *restrict dst_row, unsigned dst_stride,
                                            const float *restrict src_row, unsigned src_stride,
                                            unsigned width, unsigned height);
void
util_format_bptc_rgba_unorm_fetch_rgba(void *restrict dst, const uint8_t *restrict src,
                                             unsigned width, unsigned height);

void
util_format_bptc_srgba_unpack_rgba_8unorm(uint8_t *restrict dst_row, unsigned dst_stride,
                                          const uint8_t *restrict src_row, unsigned src_stride,
                                          unsigned width, unsigned height);
void
util_format_bptc_srgba_pack_rgba_8unorm(uint8_t *restrict dst_row, unsigned dst_stride,
                                        const uint8_t *restrict src_row, unsigned src_stride,
                                        unsigned width, unsigned height);
void
util_format_bptc_srgba_unpack_rgba_float(void *restrict dst_row, unsigned dst_stride,
                                         const uint8_t *restrict src_row, unsigned src_stride,
                                         unsigned width, unsigned height);
void
util_format_bptc_srgba_pack_rgba_float(uint8_t *restrict dst_row, unsigned dst_stride,
                                       const float *restrict src_row, unsigned src_stride,
                                       unsigned width, unsigned height);
void
util_format_bptc_srgba_fetch_rgba(void *restrict dst, const uint8_t *restrict src,
                                        unsigned width, unsigned height);

void
util_format_bptc_rgb_float_unpack_rgba_8unorm(uint8_t *restrict dst_row, unsigned dst_stride,
                                              const uint8_t *restrict src_row, unsigned src_stride,
                                              unsigned width, unsigned height);
void
util_format_bptc_rgb_float_pack_rgba_8unorm(uint8_t *restrict dst_row, unsigned dst_stride,
                                            const uint8_t *restrict src_row, unsigned src_stride,
                                            unsigned width, unsigned height);
void
util_format_bptc_rgb_float_unpack_rgba_float(void *restrict dst_row, unsigned dst_stride,
                                             const uint8_t *restrict src_row, unsigned src_stride,
                                             unsigned width, unsigned height);
void
util_format_bptc_rgb_float_pack_rgba_float(uint8_t *restrict dst_row, unsigned dst_stride,
                                           const float *restrict src_row, unsigned src_stride,
                                           unsigned width, unsigned height);
void
util_format_bptc_rgb_float_fetch_rgba(void *restrict dst, const uint8_t *restrict src,
                                            unsigned width, unsigned height);

void
util_format_bptc_rgb_ufloat_unpack_rgba_8unorm(uint8_t *restrict dst_row, unsigned dst_stride,
                                               const uint8_t *restrict src_row, unsigned src_stride,
                                               unsigned width, unsigned height);
void
util_format_bptc_rgb_ufloat_pack_rgba_8unorm(uint8_t *restrict dst_row, unsigned dst_stride,
                                             const uint8_t *restrict src_row, unsigned src_stride,
                                             unsigned width, unsigned height);
void
util_format_bptc_rgb_ufloat_unpack_rgba_float(void *restrict dst_row, unsigned dst_stride,
                                              const uint8_t *restrict src_row, unsigned src_stride,
                                              unsigned width, unsigned height);
void
util_format_bptc_rgb_ufloat_pack_rgba_float(uint8_t *restrict dst_row, unsigned dst_stride,
                                            const float *restrict src_row, unsigned src_stride,
                                            unsigned width, unsigned height);
void
util_format_bptc_rgb_ufloat_fetch_rgba(void *restrict dst, const uint8_t *restrict src,
                                             unsigned width, unsigned height);
#ifdef __cplusplus
}
#endif

#endif /* U_FORMAT_BPTC_H_ */

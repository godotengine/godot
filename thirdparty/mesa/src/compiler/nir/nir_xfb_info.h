/*
 * Copyright © 2018 Intel Corporation
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
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef NIR_XFB_INFO_H
#define NIR_XFB_INFO_H

#include "nir.h"

#ifdef __cplusplus
extern "C" {
#endif

#define NIR_MAX_XFB_BUFFERS 4
#define NIR_MAX_XFB_STREAMS 4

typedef struct {
   uint16_t stride;
   uint16_t varying_count;
} nir_xfb_buffer_info;

typedef struct {
   uint8_t buffer;
   uint16_t offset;
   uint8_t location;
   bool high_16bits;
   uint8_t component_mask;
   uint8_t component_offset;
} nir_xfb_output_info;

typedef struct {
   const struct glsl_type *type;
   uint8_t buffer;
   uint16_t offset;
} nir_xfb_varying_info;

typedef struct nir_xfb_info {
   uint8_t buffers_written;
   uint8_t streams_written;

   nir_xfb_buffer_info buffers[NIR_MAX_XFB_BUFFERS];
   uint8_t buffer_to_stream[NIR_MAX_XFB_BUFFERS];

   uint16_t output_count;
   nir_xfb_output_info outputs[0];
} nir_xfb_info;

typedef struct nir_xfb_varyings_info {
   uint16_t varying_count;
   nir_xfb_varying_info varyings[0];
} nir_xfb_varyings_info;

static inline size_t
nir_xfb_info_size(uint16_t output_count)
{
   return sizeof(nir_xfb_info) + sizeof(nir_xfb_output_info) * output_count;
}

void nir_shader_gather_xfb_info(nir_shader *shader);

void
nir_gather_xfb_info_with_varyings(nir_shader *shader,
                                  void *mem_ctx,
                                  nir_xfb_varyings_info **varyings_info);

void
nir_gather_xfb_info_from_intrinsics(nir_shader *nir);

void
nir_print_xfb_info(nir_xfb_info *info, FILE *fp);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* NIR_XFB_INFO_H */

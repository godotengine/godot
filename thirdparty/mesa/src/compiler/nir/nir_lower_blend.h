/*
 * Copyright Â 2019 Alyssa Rosenzweig
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
 *
 */

#ifndef NIR_BLEND_H
#define NIR_BLEND_H

#include "compiler/nir/nir.h"
#include "util/format/u_formats.h"

/* These structs encapsulates the blend state such that it can be lowered
 * cleanly
 */

typedef struct {
   enum blend_func func;

   enum blend_factor src_factor;
   bool invert_src_factor;

   enum blend_factor dst_factor;
   bool invert_dst_factor;
} nir_lower_blend_channel;

typedef struct {
   nir_lower_blend_channel rgb;
   nir_lower_blend_channel alpha;

   /* 4-bit colormask. 0x0 for none, 0xF for RGBA, 0x1 for R */
   unsigned colormask;
} nir_lower_blend_rt;

typedef struct {
   nir_lower_blend_rt rt[8];
   enum pipe_format format[8];

   bool logicop_enable;
   unsigned logicop_func;

   nir_ssa_def *src1;

   /* If set, will use load_blend_const_color_{r,g,b,a}_float instead of
    * load_blend_const_color_rgba */
   bool scalar_blend_const;
} nir_lower_blend_options;

void nir_lower_blend(nir_shader *shader,
                     const nir_lower_blend_options *options);

#endif

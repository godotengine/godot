/*
 * Copyright © Microsoft Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * on the rights to use, copy, modify, merge, publish, distribute, sub
 * license, and/or sell copies of the Software, and to permit persons to whom
 * the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHOR(S) AND/OR THEIR SUPPLIERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
 * USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef DXIL_NIR_LOWER_INT_SAMPLERS_H
#define DXIL_NIR_LOWER_INT_SAMPLERS_H

#include "pipe/p_state.h"
#include "nir.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
   unsigned swizzle_r:3;
   unsigned swizzle_g:3;
   unsigned swizzle_b:3;
   unsigned swizzle_a:3;
} dxil_texture_swizzle_state;

typedef struct {
   float border_color[4];
   float lod_bias;
   float min_lod, max_lod;
   int last_level;
   uint8_t wrap[3];
   uint8_t is_int_sampler:1;
   uint8_t is_nonnormalized_coords:1;
   uint8_t is_linear_filtering:1;
   uint8_t skip_boundary_conditions:1;
   uint8_t unused:4;
} dxil_wrap_sampler_state;

bool
dxil_lower_sample_to_txf_for_integer_tex(nir_shader *s,
                                         dxil_wrap_sampler_state *wrap_states,
                                         dxil_texture_swizzle_state *tex_swizzles,
                                         float max_bias);

#ifdef __cplusplus
}
#endif

#endif // DXIL_NIR_LOWER_INT_SAMPLERS_H

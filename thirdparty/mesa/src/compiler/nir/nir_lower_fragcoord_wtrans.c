/*
 * Copyright (C) 2019 Andreas Baierl
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

#include "nir.h"
#include "nir_builder.h"

/* Lower gl_FragCoord and transform the w component
 * according to the following pseudocode:
 *
 *    gl_FragCoord.xyz = gl_FragCoord_orig.xyz
 *    gl_FragCoord.w = 1.0 / gl_FragCoord_orig.w
 *
 */

static bool
lower_fragcoord_wtrans_filter(const nir_instr *instr, UNUSED const void *_options)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   if (intr->intrinsic == nir_intrinsic_load_frag_coord)
      return true;

   if (intr->intrinsic != nir_intrinsic_load_deref)
      return false;

   nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
   if (!nir_deref_mode_must_be(deref, nir_var_shader_in))
      return false;

   nir_variable *var = nir_intrinsic_get_var(intr, 0);
   return var->data.location == VARYING_SLOT_POS;
}

static nir_ssa_def *
lower_fragcoord_wtrans_impl(nir_builder *b, nir_instr *instr,
                            UNUSED void *_options)
{
   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);

   return nir_vec4(b,
                   nir_channel(b, &intr->dest.ssa, 0),
                   nir_channel(b, &intr->dest.ssa, 1),
                   nir_channel(b, &intr->dest.ssa, 2),
                   nir_frcp(b, nir_channel(b, &intr->dest.ssa, 3)));
}

bool
nir_lower_fragcoord_wtrans(nir_shader *shader)
{
   assert(shader->info.stage == MESA_SHADER_FRAGMENT);

   return nir_shader_lower_instructions(shader,
                                        lower_fragcoord_wtrans_filter,
                                        lower_fragcoord_wtrans_impl,
                                        NULL);

}

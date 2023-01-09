/*
 * Copyright © 2019 Collabora, Ltd.
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
 * Authors (Collabora):
 *    Alyssa Rosenzweig <alyssa.rosenzweig@collabora.com>
 */

#include "nir.h"
#include "nir_builder.h"

/*
 * Lowers SSBOs to globals, for hardware that lack native SSBO support. When
 * lowering, *_ssbo_* instructions will become *_global_* instructions,
 * augmented with load_ssbo_address.
 *
 * DOES NOT PERFORM BOUNDS CHECKING. DO NOT USE IN PRODUCTION ON UNTRUSTED
 * CONTEXTS INCLUDING WEBGL 2.
 */

static nir_intrinsic_op
lower_ssbo_op(nir_intrinsic_op op)
{
   switch (op) {
   case nir_intrinsic_load_ssbo:
      return nir_intrinsic_load_global;

   case nir_intrinsic_store_ssbo:
      return nir_intrinsic_store_global;

   case nir_intrinsic_ssbo_atomic_add:
      return nir_intrinsic_global_atomic_add;
   case nir_intrinsic_ssbo_atomic_imin:
      return nir_intrinsic_global_atomic_imin;
   case nir_intrinsic_ssbo_atomic_umin:
      return nir_intrinsic_global_atomic_umin;
   case nir_intrinsic_ssbo_atomic_imax:
      return nir_intrinsic_global_atomic_imax;
   case nir_intrinsic_ssbo_atomic_umax:
      return nir_intrinsic_global_atomic_umax;
   case nir_intrinsic_ssbo_atomic_and:
      return nir_intrinsic_global_atomic_and;
   case nir_intrinsic_ssbo_atomic_or:
      return nir_intrinsic_global_atomic_or;
   case nir_intrinsic_ssbo_atomic_xor:
      return nir_intrinsic_global_atomic_xor;
   case nir_intrinsic_ssbo_atomic_exchange:
      return nir_intrinsic_global_atomic_exchange;
   case nir_intrinsic_ssbo_atomic_comp_swap:
      return nir_intrinsic_global_atomic_comp_swap;

   case nir_intrinsic_ssbo_atomic_fadd:
      return nir_intrinsic_global_atomic_fadd;
   case nir_intrinsic_ssbo_atomic_fmin:
      return nir_intrinsic_global_atomic_fmin;
   case nir_intrinsic_ssbo_atomic_fmax:
      return nir_intrinsic_global_atomic_fmax;
   case nir_intrinsic_ssbo_atomic_fcomp_swap:
      return nir_intrinsic_global_atomic_fcomp_swap;

   default:
      unreachable("Invalid SSBO op");
   }
}

/* Like SSBO property sysvals, though SSBO index may be indirect. C.f.
 * nir_load_system_value */

static inline nir_ssa_def *
nir_load_ssbo_prop(nir_builder *b, nir_intrinsic_op op,
      nir_src *idx, unsigned bitsize)
{
   nir_intrinsic_instr *load = nir_intrinsic_instr_create(b->shader, op);
   load->num_components = 1;
   nir_src_copy(&load->src[0], idx, &load->instr);
   nir_ssa_dest_init(&load->instr, &load->dest, 1, bitsize, NULL);
   nir_builder_instr_insert(b, &load->instr);
   return &load->dest.ssa;
}

#define nir_ssbo_prop(b, prop, index, bitsize) \
   nir_load_ssbo_prop(b, nir_intrinsic_##prop, index, bitsize)

static nir_ssa_def *
lower_ssbo_instr(nir_builder *b, nir_intrinsic_instr *intr)
{
   nir_intrinsic_op op = lower_ssbo_op(intr->intrinsic);
   bool is_store = op == nir_intrinsic_store_global;
   bool is_atomic = !is_store && op != nir_intrinsic_load_global;

   /* We have to calculate the address:
    *
    * &(SSBO[offset]) = &SSBO + offset
    */

   nir_src index = intr->src[is_store ? 1 : 0];
   nir_src *offset_src = nir_get_io_offset_src(intr);
   nir_ssa_def *offset = nir_ssa_for_src(b, *offset_src, 1);

   nir_ssa_def *address =
      nir_iadd(b,
            nir_ssbo_prop(b, load_ssbo_address, &index, 64),
            nir_u2u64(b, offset));

   /* Create the replacement intrinsic */

   nir_intrinsic_instr *global =
      nir_intrinsic_instr_create(b->shader, op);

   global->num_components = intr->num_components;
   global->src[is_store ? 1 : 0] = nir_src_for_ssa(address);

   if (!is_atomic) {
      nir_intrinsic_set_align_mul(global, nir_intrinsic_align_mul(intr));
      nir_intrinsic_set_align_offset(global, nir_intrinsic_align_offset(intr));
   }

   if (is_store) {
      nir_src_copy(&global->src[0], &intr->src[0], &global->instr);
      nir_intrinsic_set_write_mask(global, nir_intrinsic_write_mask(intr));
   } else {
      nir_ssa_dest_init(&global->instr, &global->dest,
                        intr->dest.ssa.num_components,
                        intr->dest.ssa.bit_size, NULL);

      if (is_atomic) {
         nir_src_copy(&global->src[1], &intr->src[2], &global->instr);
         if (nir_intrinsic_infos[op].num_srcs > 2)
            nir_src_copy(&global->src[2], &intr->src[3], &global->instr);
      }
   }

   nir_builder_instr_insert(b, &global->instr);
   return is_store ? NULL : &global->dest.ssa;
}

static bool
should_lower_ssbo_instr(const nir_instr *instr)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;

   const nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);

   switch (intr->intrinsic) {
   case nir_intrinsic_load_ssbo:
   case nir_intrinsic_store_ssbo:
   case nir_intrinsic_ssbo_atomic_add:
   case nir_intrinsic_ssbo_atomic_imin:
   case nir_intrinsic_ssbo_atomic_umin:
   case nir_intrinsic_ssbo_atomic_imax:
   case nir_intrinsic_ssbo_atomic_umax:
   case nir_intrinsic_ssbo_atomic_and:
   case nir_intrinsic_ssbo_atomic_or:
   case nir_intrinsic_ssbo_atomic_xor:
   case nir_intrinsic_ssbo_atomic_exchange:
   case nir_intrinsic_ssbo_atomic_comp_swap:
   case nir_intrinsic_ssbo_atomic_fadd:
   case nir_intrinsic_ssbo_atomic_fmin:
   case nir_intrinsic_ssbo_atomic_fmax:
   case nir_intrinsic_ssbo_atomic_fcomp_swap:
      return true;
   default:
      return false;
   }

   return false;
}

bool
nir_lower_ssbo(nir_shader *shader)
{
   bool progress = false;

   nir_foreach_function(function, shader) {
      nir_function_impl *impl = function->impl;
      nir_builder b;
      nir_builder_init(&b, impl);

      nir_foreach_block(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if (!should_lower_ssbo_instr(instr)) continue;
            progress = true;
            b.cursor = nir_before_instr(instr);

            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            nir_ssa_def *replace = lower_ssbo_instr(&b, intr);

            if (replace)  {
               nir_ssa_def_rewrite_uses(&intr->dest.ssa,
                                     replace);
            }

            nir_instr_remove(instr);
         }
      }
   }

   return progress;
}

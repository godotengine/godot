/*
 * Copyright © 2019 Intel Corporation
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

struct nu_handle {
   nir_src *src;
   nir_ssa_def *handle;
   nir_deref_instr *parent_deref;
   nir_ssa_def *first;
};

static bool
nu_handle_init(struct nu_handle *h, nir_src *src)
{
   h->src = src;

   nir_deref_instr *deref = nir_src_as_deref(*src);
   if (deref) {
      if (deref->deref_type == nir_deref_type_var)
         return false;

      nir_deref_instr *parent = nir_deref_instr_parent(deref);
      assert(parent->deref_type == nir_deref_type_var);

      assert(deref->deref_type == nir_deref_type_array);
      if (nir_src_is_const(deref->arr.index))
         return false;

      assert(deref->arr.index.is_ssa);
      h->handle = deref->arr.index.ssa;
      h->parent_deref = parent;

      return true;
   } else {
      if (nir_src_is_const(*src))
         return false;

      assert(src->is_ssa);
      h->handle = src->ssa;
      h->parent_deref = NULL;

      return true;
   }
}

static nir_ssa_def *
nu_handle_compare(const nir_lower_non_uniform_access_options *options,
                  nir_builder *b, struct nu_handle *handle)
{
   nir_component_mask_t channel_mask = ~0;
   if (options->callback)
      channel_mask = options->callback(handle->src, options->callback_data);
   channel_mask &= nir_component_mask(handle->handle->num_components);

   nir_ssa_def *channels[NIR_MAX_VEC_COMPONENTS];
   for (unsigned i = 0; i < handle->handle->num_components; i++)
      channels[i] = nir_channel(b, handle->handle, i);

   handle->first = handle->handle;
   nir_ssa_def *equal_first = nir_imm_true(b);
   u_foreach_bit(i, channel_mask) {
      nir_ssa_def *first = nir_read_first_invocation(b, channels[i]);
      handle->first = nir_vector_insert_imm(b, handle->first, first, i);

      equal_first = nir_iand(b, equal_first, nir_ieq(b, first, channels[i]));
   }

   return equal_first;
}

static void
nu_handle_rewrite(nir_builder *b, struct nu_handle *h)
{
   if (h->parent_deref) {
      /* Replicate the deref. */
      nir_deref_instr *deref =
         nir_build_deref_array(b, h->parent_deref, h->first);
      *(h->src) = nir_src_for_ssa(&deref->dest.ssa);
   } else {
      *(h->src) = nir_src_for_ssa(h->first);
   }
}

static bool
lower_non_uniform_tex_access(const nir_lower_non_uniform_access_options *options,
                             nir_builder *b, nir_tex_instr *tex)
{
   if (!tex->texture_non_uniform && !tex->sampler_non_uniform)
      return false;

   /* We can have at most one texture and one sampler handle */
   unsigned num_handles = 0;
   struct nu_handle handles[2];
   for (unsigned i = 0; i < tex->num_srcs; i++) {
      switch (tex->src[i].src_type) {
      case nir_tex_src_texture_offset:
      case nir_tex_src_texture_handle:
      case nir_tex_src_texture_deref:
         if (!tex->texture_non_uniform)
            continue;
         break;

      case nir_tex_src_sampler_offset:
      case nir_tex_src_sampler_handle:
      case nir_tex_src_sampler_deref:
         if (!tex->sampler_non_uniform)
            continue;
         break;

      default:
         continue;
      }

      assert(num_handles <= ARRAY_SIZE(handles));
      if (nu_handle_init(&handles[num_handles], &tex->src[i].src))
         num_handles++;
   }

   if (num_handles == 0)
      return false;

   b->cursor = nir_instr_remove(&tex->instr);

   nir_push_loop(b);

   nir_ssa_def *all_equal_first = nir_imm_true(b);
   for (unsigned i = 0; i < num_handles; i++) {
      if (i && handles[i].handle == handles[0].handle) {
         handles[i].first = handles[0].first;
         continue;
      }

      nir_ssa_def *equal_first = nu_handle_compare(options, b, &handles[i]);
      all_equal_first = nir_iand(b, all_equal_first, equal_first);
   }

   nir_push_if(b, all_equal_first);

   for (unsigned i = 0; i < num_handles; i++)
      nu_handle_rewrite(b, &handles[i]);

   nir_builder_instr_insert(b, &tex->instr);
   nir_jump(b, nir_jump_break);

   tex->texture_non_uniform = false;
   tex->sampler_non_uniform = false;

   return true;
}

static bool
lower_non_uniform_access_intrin(const nir_lower_non_uniform_access_options *options,
                                nir_builder *b, nir_intrinsic_instr *intrin,
                                unsigned handle_src)
{
   if (!(nir_intrinsic_access(intrin) & ACCESS_NON_UNIFORM))
      return false;

   struct nu_handle handle;
   if (!nu_handle_init(&handle, &intrin->src[handle_src]))
      return false;

   b->cursor = nir_instr_remove(&intrin->instr);

   nir_push_loop(b);

   nir_push_if(b, nu_handle_compare(options, b, &handle));

   nu_handle_rewrite(b, &handle);

   nir_builder_instr_insert(b, &intrin->instr);
   nir_jump(b, nir_jump_break);

   nir_intrinsic_set_access(intrin, nir_intrinsic_access(intrin) & ~ACCESS_NON_UNIFORM);

   return true;
}

static bool
nir_lower_non_uniform_access_impl(nir_function_impl *impl,
                                  const nir_lower_non_uniform_access_options *options)
{
   bool progress = false;

   nir_builder b;
   nir_builder_init(&b, impl);

   nir_foreach_block_safe(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         switch (instr->type) {
         case nir_instr_type_tex: {
            nir_tex_instr *tex = nir_instr_as_tex(instr);
            if ((options->types & nir_lower_non_uniform_texture_access) &&
                lower_non_uniform_tex_access(options, &b, tex))
               progress = true;
            break;
         }

         case nir_instr_type_intrinsic: {
            nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
            switch (intrin->intrinsic) {
            case nir_intrinsic_load_ubo:
               if ((options->types & nir_lower_non_uniform_ubo_access) &&
                   lower_non_uniform_access_intrin(options, &b, intrin, 0))
                  progress = true;
               break;

            case nir_intrinsic_load_ssbo:
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
               if ((options->types & nir_lower_non_uniform_ssbo_access) &&
                   lower_non_uniform_access_intrin(options, &b, intrin, 0))
                  progress = true;
               break;

            case nir_intrinsic_store_ssbo:
               /* SSBO Stores put the index in the second source */
               if ((options->types & nir_lower_non_uniform_ssbo_access) &&
                   lower_non_uniform_access_intrin(options, &b, intrin, 1))
                  progress = true;
               break;

            case nir_intrinsic_image_load:
            case nir_intrinsic_image_sparse_load:
            case nir_intrinsic_image_store:
            case nir_intrinsic_image_atomic_add:
            case nir_intrinsic_image_atomic_imin:
            case nir_intrinsic_image_atomic_umin:
            case nir_intrinsic_image_atomic_imax:
            case nir_intrinsic_image_atomic_umax:
            case nir_intrinsic_image_atomic_and:
            case nir_intrinsic_image_atomic_or:
            case nir_intrinsic_image_atomic_xor:
            case nir_intrinsic_image_atomic_exchange:
            case nir_intrinsic_image_atomic_comp_swap:
            case nir_intrinsic_image_atomic_fadd:
            case nir_intrinsic_image_atomic_fmin:
            case nir_intrinsic_image_atomic_fmax:
            case nir_intrinsic_image_size:
            case nir_intrinsic_image_samples:
            case nir_intrinsic_image_samples_identical:
            case nir_intrinsic_image_fragment_mask_load_amd:
            case nir_intrinsic_bindless_image_load:
            case nir_intrinsic_bindless_image_sparse_load:
            case nir_intrinsic_bindless_image_store:
            case nir_intrinsic_bindless_image_atomic_add:
            case nir_intrinsic_bindless_image_atomic_imin:
            case nir_intrinsic_bindless_image_atomic_umin:
            case nir_intrinsic_bindless_image_atomic_imax:
            case nir_intrinsic_bindless_image_atomic_umax:
            case nir_intrinsic_bindless_image_atomic_and:
            case nir_intrinsic_bindless_image_atomic_or:
            case nir_intrinsic_bindless_image_atomic_xor:
            case nir_intrinsic_bindless_image_atomic_exchange:
            case nir_intrinsic_bindless_image_atomic_comp_swap:
            case nir_intrinsic_bindless_image_atomic_fadd:
            case nir_intrinsic_bindless_image_atomic_fmin:
            case nir_intrinsic_bindless_image_atomic_fmax:
            case nir_intrinsic_bindless_image_size:
            case nir_intrinsic_bindless_image_samples:
            case nir_intrinsic_bindless_image_samples_identical:
            case nir_intrinsic_bindless_image_fragment_mask_load_amd:
            case nir_intrinsic_image_deref_load:
            case nir_intrinsic_image_deref_sparse_load:
            case nir_intrinsic_image_deref_store:
            case nir_intrinsic_image_deref_atomic_add:
            case nir_intrinsic_image_deref_atomic_umin:
            case nir_intrinsic_image_deref_atomic_imin:
            case nir_intrinsic_image_deref_atomic_umax:
            case nir_intrinsic_image_deref_atomic_imax:
            case nir_intrinsic_image_deref_atomic_and:
            case nir_intrinsic_image_deref_atomic_or:
            case nir_intrinsic_image_deref_atomic_xor:
            case nir_intrinsic_image_deref_atomic_exchange:
            case nir_intrinsic_image_deref_atomic_comp_swap:
            case nir_intrinsic_image_deref_atomic_fadd:
            case nir_intrinsic_image_deref_atomic_fmin:
            case nir_intrinsic_image_deref_atomic_fmax:
            case nir_intrinsic_image_deref_size:
            case nir_intrinsic_image_deref_samples:
            case nir_intrinsic_image_deref_samples_identical:
            case nir_intrinsic_image_deref_fragment_mask_load_amd:
               if ((options->types & nir_lower_non_uniform_image_access) &&
                   lower_non_uniform_access_intrin(options, &b, intrin, 0))
                  progress = true;
               break;

            default:
               /* Nothing to do */
               break;
            }
            break;
         }

         default:
            /* Nothing to do */
            break;
         }
      }
   }

   if (progress)
      nir_metadata_preserve(impl, nir_metadata_none);

   return progress;
}

/**
 * Lowers non-uniform resource access by using a loop
 *
 * This pass lowers non-uniform resource access by using subgroup operations
 * and a loop.  Most hardware requires things like textures and UBO access
 * operations to happen on a dynamically uniform (or at least subgroup
 * uniform) resource.  This pass allows for non-uniform access by placing the
 * texture instruction in a loop that looks something like this:
 *
 * loop {
 *    bool tex_eq_first = readFirstInvocationARB(texture) == texture;
 *    bool smp_eq_first = readFirstInvocationARB(sampler) == sampler;
 *    if (tex_eq_first && smp_eq_first) {
 *       res = texture(texture, sampler, ...);
 *       break;
 *    }
 * }
 *
 * Fortunately, because the instruction is immediately followed by the only
 * break in the loop, the block containing the instruction dominates the end
 * of the loop.  Therefore, it's safe to move the instruction into the loop
 * without fixing up SSA in any way.
 */
bool
nir_lower_non_uniform_access(nir_shader *shader,
                             const nir_lower_non_uniform_access_options *options)
{
   bool progress = false;

   nir_foreach_function(function, shader) {
      if (function->impl &&
          nir_lower_non_uniform_access_impl(function->impl, options))
         progress = true;
   }

   return progress;
}

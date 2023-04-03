/*
 * Copyright © 2021 Valve Corporation
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

/* This pass provides a way to move computations that are always the same for
 * an entire draw/compute dispatch into a "preamble" that runs before the main
 * entrypoint.
 *
 * We also expose a separate API to get or construct the preamble of a shader
 * in case backends want to insert their own code.
 */


nir_function_impl *
nir_shader_get_preamble(nir_shader *shader)
{
   nir_function_impl *entrypoint = nir_shader_get_entrypoint(shader);
   if (entrypoint->preamble) {
      return entrypoint->preamble->impl;
   } else {
      nir_function *preamble = nir_function_create(shader, "@preamble");
      preamble->is_preamble = true;
      nir_function_impl *impl = nir_function_impl_create(preamble);
      entrypoint->preamble = preamble;
      return impl;
   }
}

typedef struct {
   bool can_move;
   bool candidate;
   bool must_stay;
   bool replace;

   unsigned can_move_users;

   unsigned size, align;

   unsigned offset;

   /* Average the cost of a value among its users, to try to account for
    * values that have multiple can_move uses.
    */
   float value;

   /* Overall benefit, i.e. the value minus any cost to inserting
    * load_preamble.
    */
   float benefit;
} def_state;

typedef struct {
   /* Per-definition array of states */
   def_state *states;

   nir_ssa_def *def;

   const nir_opt_preamble_options *options;
} opt_preamble_ctx;

static float
get_instr_cost(nir_instr *instr, const nir_opt_preamble_options *options)
{
   /* No backend will want to hoist load_const or undef by itself, so handle
    * this for them.
    */
   if (instr->type == nir_instr_type_load_const ||
       instr->type == nir_instr_type_ssa_undef)
      return 0;

   return options->instr_cost_cb(instr, options->cb_data);
}

static bool
can_move_src(nir_src *src, void *state)
{
   opt_preamble_ctx *ctx = state;

   assert(src->is_ssa);
   return ctx->states[src->ssa->index].can_move;
}

static bool
can_move_srcs(nir_instr *instr, opt_preamble_ctx *ctx)
{
   return nir_foreach_src(instr, can_move_src, ctx);
}

static bool
can_move_intrinsic(nir_intrinsic_instr *instr, opt_preamble_ctx *ctx)
{
   switch (instr->intrinsic) {
   /* Intrinsics which can always be moved */
   case nir_intrinsic_load_push_constant:
   case nir_intrinsic_load_work_dim:
   case nir_intrinsic_load_num_workgroups:
   case nir_intrinsic_load_workgroup_size:
   case nir_intrinsic_load_ray_launch_size:
   case nir_intrinsic_load_ray_launch_size_addr_amd:
   case nir_intrinsic_load_sbt_base_amd:
   case nir_intrinsic_load_is_indexed_draw:
   case nir_intrinsic_load_viewport_scale:
   case nir_intrinsic_load_user_clip_plane:
   case nir_intrinsic_load_viewport_x_scale:
   case nir_intrinsic_load_viewport_y_scale:
   case nir_intrinsic_load_viewport_z_scale:
   case nir_intrinsic_load_viewport_offset:
   case nir_intrinsic_load_viewport_x_offset:
   case nir_intrinsic_load_viewport_y_offset:
   case nir_intrinsic_load_viewport_z_offset:
   case nir_intrinsic_load_blend_const_color_a_float:
   case nir_intrinsic_load_blend_const_color_b_float:
   case nir_intrinsic_load_blend_const_color_g_float:
   case nir_intrinsic_load_blend_const_color_r_float:
   case nir_intrinsic_load_blend_const_color_rgba:
   case nir_intrinsic_load_blend_const_color_aaaa8888_unorm:
   case nir_intrinsic_load_blend_const_color_rgba8888_unorm:
   case nir_intrinsic_load_line_width:
   case nir_intrinsic_load_aa_line_width:
   case nir_intrinsic_load_fb_layers_v3d:
   case nir_intrinsic_load_tcs_num_patches_amd:
   case nir_intrinsic_load_sample_positions_pan:
   case nir_intrinsic_load_pipeline_stat_query_enabled_amd:
   case nir_intrinsic_load_prim_gen_query_enabled_amd:
   case nir_intrinsic_load_prim_xfb_query_enabled_amd:
   case nir_intrinsic_load_clamp_vertex_color_amd:
   case nir_intrinsic_load_cull_front_face_enabled_amd:
   case nir_intrinsic_load_cull_back_face_enabled_amd:
   case nir_intrinsic_load_cull_ccw_amd:
   case nir_intrinsic_load_cull_small_primitives_enabled_amd:
   case nir_intrinsic_load_cull_any_enabled_amd:
   case nir_intrinsic_load_cull_small_prim_precision_amd:
   case nir_intrinsic_load_texture_base_agx:
   case nir_intrinsic_load_ubo_base_agx:
   case nir_intrinsic_load_vbo_base_agx:
      return true;

   /* Intrinsics which can be moved depending on hardware */
   case nir_intrinsic_load_base_instance:
   case nir_intrinsic_load_base_vertex:
   case nir_intrinsic_load_first_vertex:
   case nir_intrinsic_load_draw_id:
      return ctx->options->drawid_uniform;

   case nir_intrinsic_load_subgroup_size:
   case nir_intrinsic_load_num_subgroups:
      return ctx->options->subgroup_size_uniform;

   /* Intrinsics which can be moved if the sources can */
   case nir_intrinsic_load_ubo:
   case nir_intrinsic_load_ubo_vec4:
   case nir_intrinsic_get_ubo_size:
   case nir_intrinsic_get_ssbo_size:
   case nir_intrinsic_ballot_bitfield_extract:
   case nir_intrinsic_ballot_find_lsb:
   case nir_intrinsic_ballot_find_msb:
   case nir_intrinsic_ballot_bit_count_reduce:
   case nir_intrinsic_load_deref:
   case nir_intrinsic_load_global_constant:
   case nir_intrinsic_load_uniform:
   case nir_intrinsic_load_constant:
   case nir_intrinsic_load_sample_pos_from_id:
   case nir_intrinsic_load_kernel_input:
   case nir_intrinsic_load_buffer_amd:
   case nir_intrinsic_image_samples:
   case nir_intrinsic_image_deref_samples:
   case nir_intrinsic_bindless_image_samples:
   case nir_intrinsic_image_size:
   case nir_intrinsic_image_deref_size:
   case nir_intrinsic_bindless_image_size:
   case nir_intrinsic_vulkan_resource_index:
   case nir_intrinsic_vulkan_resource_reindex:
   case nir_intrinsic_load_vulkan_descriptor:
   case nir_intrinsic_quad_swizzle_amd:
   case nir_intrinsic_masked_swizzle_amd:
   case nir_intrinsic_load_ssbo_address:
   case nir_intrinsic_bindless_resource_ir3:
   case nir_intrinsic_load_constant_agx:
      return can_move_srcs(&instr->instr, ctx);

   /* Image/SSBO loads can be moved if they are CAN_REORDER and their
    * sources can be moved.
    */
   case nir_intrinsic_image_load:
   case nir_intrinsic_image_samples_identical:
   case nir_intrinsic_bindless_image_load:
   case nir_intrinsic_load_ssbo:
   case nir_intrinsic_load_ssbo_ir3:
      return (nir_intrinsic_access(instr) & ACCESS_CAN_REORDER) &&
         can_move_srcs(&instr->instr, ctx);

   default:
      return false;
   }
}

static bool
can_move_instr(nir_instr *instr, opt_preamble_ctx *ctx)
{
   switch (instr->type) {
   case nir_instr_type_tex: {
      nir_tex_instr *tex = nir_instr_as_tex(instr);
      /* See note below about derivatives. We have special code to convert tex
       * to txd, though, because it's a common case.
       */
      if (nir_tex_instr_has_implicit_derivative(tex) &&
          tex->op != nir_texop_tex) {
         return false;
      }
      return can_move_srcs(instr, ctx);
   }
   case nir_instr_type_alu: {
      /* The preamble is presumably run with only one thread, so we can't run
       * derivatives in it.
       * TODO: Replace derivatives with 0 instead, if real apps hit this.
       */
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      switch (alu->op) {
      case nir_op_fddx:
      case nir_op_fddy:
      case nir_op_fddx_fine:
      case nir_op_fddy_fine:
      case nir_op_fddx_coarse:
      case nir_op_fddy_coarse:
         return false;
      default:
         return can_move_srcs(instr, ctx);
      }
   }
   case nir_instr_type_intrinsic:
      return can_move_intrinsic(nir_instr_as_intrinsic(instr), ctx);

   case nir_instr_type_load_const:
   case nir_instr_type_ssa_undef:
      return true;

   case nir_instr_type_deref: {
      nir_deref_instr *deref = nir_instr_as_deref(instr);
      if (deref->deref_type == nir_deref_type_var) {
         switch (deref->modes) {
         case nir_var_uniform:
         case nir_var_mem_ubo:
            return true;
         default:
            return false;
         }
      } else {
         return can_move_srcs(instr, ctx);
      }
   }

   case nir_instr_type_phi:
      /* TODO: we could move an if-statement if everything inside it is
       * moveable.
       */
      return false;

   default:
      return false;
   }
}

/* True if we should avoid making this a candidate. This is only called on
 * instructions we already determined we can move, this just makes it so that
 * uses of this instruction cannot be rewritten. Typically this happens
 * because of static constraints on the IR, for example some deref chains
 * cannot be broken.
 */
static bool
avoid_instr(nir_instr *instr, const nir_opt_preamble_options *options)
{
   if (instr->type == nir_instr_type_deref)
      return true;

   return options->avoid_instr_cb(instr, options->cb_data);
}

static bool
update_src_value(nir_src *src, void *data)
{
   opt_preamble_ctx *ctx = data;

   def_state *state = &ctx->states[ctx->def->index];
   def_state *src_state = &ctx->states[src->ssa->index];

   assert(src_state->can_move);

   /* If an instruction has can_move and non-can_move users, it becomes a
    * candidate and its value shouldn't propagate downwards. For example,
    * imagine a chain like this:
    *
    *         -- F (cannot move)
    *        /
    *  A <-- B <-- C <-- D <-- E (cannot move)
    *
    * B and D are marked candidates. Picking B removes A and B, picking D
    * removes C and D, and picking both removes all 4. Therefore B and D are
    * independent and B's value shouldn't flow into D.
    *
    * A similar argument holds for must_stay values.
    */
   if (!src_state->must_stay && !src_state->candidate)
      state->value += src_state->value;
   return true;
}

static int
candidate_sort(const void *data1, const void *data2)
{
   const def_state *state1 = *(def_state **)data1;
   const def_state *state2 = *(def_state **)data2;

   float value1 = state1->value / state1->size;
   float value2 = state2->value / state2->size;
   if (value1 < value2)
      return 1;
   else if (value1 > value2)
      return -1;
   else
      return 0;
}

bool
nir_opt_preamble(nir_shader *shader, const nir_opt_preamble_options *options,
                 unsigned *size)
{
   opt_preamble_ctx ctx = {
      .options = options,
   };

   nir_function_impl *impl = nir_shader_get_entrypoint(shader);
   ctx.states = calloc(impl->ssa_alloc, sizeof(*ctx.states));

   /* Step 1: Calculate can_move */
   nir_foreach_block (block, impl) {
      nir_foreach_instr (instr, block) {
         nir_ssa_def *def = nir_instr_ssa_def(instr);
         if (!def)
            continue;

         def_state *state = &ctx.states[def->index];

         state->can_move = can_move_instr(instr, &ctx);
      }
   }

   /* Step 2: Calculate is_candidate. This is complicated by the presence of
    * non-candidate instructions like derefs whose users cannot be rewritten.
    * If a deref chain is used at all by a non-can_move thing, then any offset
    * sources anywhere along the chain should be considered candidates because
    * the entire deref chain will never be deleted, but if it's only used by
    * can_move things then it becomes subsumed by its users and none of the
    * offset sources should be considered candidates as they will be removed
    * when the users of the deref chain are moved. We need to replace "are
    * there any non-can_move users" with "are there any non-can_move users,
    * *recursing through non-candidate users*". We do this by walking backward
    * and marking when a non-candidate instruction must stay in the final
    * program because it has a non-can_move user, including recursively.
    */
   unsigned num_candidates = 0;
   nir_foreach_block_reverse (block, impl) {
      nir_foreach_instr_reverse (instr, block) {
         nir_ssa_def *def = nir_instr_ssa_def(instr);
         if (!def)
            continue;

         def_state *state = &ctx.states[def->index];
         if (!state->can_move)
            continue;

         state->value = get_instr_cost(instr, options);
         bool is_candidate = !avoid_instr(instr, options);
         state->candidate = false;
         state->must_stay = false;
         nir_foreach_use (use, def) {
            nir_ssa_def *use_def = nir_instr_ssa_def(use->parent_instr);
            if (!use_def || !ctx.states[use_def->index].can_move ||
                ctx.states[use_def->index].must_stay) {
               if (is_candidate)
                  state->candidate = true;
               else
                  state->must_stay = true;
            } else {
               state->can_move_users++;
            }
         }

         nir_foreach_if_use (use, def) {
            if (is_candidate)
               state->candidate = true;
            else
               state->must_stay = true;
            break;
         }

         if (state->candidate)
            num_candidates++;
      }
   }

   if (num_candidates == 0) {
      *size = 0;
      free(ctx.states);
      return false;
   }

   def_state **candidates = malloc(sizeof(*candidates) * num_candidates);
   unsigned candidate_idx = 0;
   unsigned total_size = 0;

   /* Step 3: Calculate value of candidates by propagating downwards. We try
    * to share the value amongst can_move uses, in case there are multiple.
    * This won't always find the most optimal solution, but is hopefully a
    * good heuristic.
    *
    * Note that we use the can_move adjusted in the last pass, because if a
    * can_move instruction cannot be moved because it's not a candidate and it
    * has a non-can_move source then we don't want to count it as a use.
    *
    * While we're here, also collect an array of candidates.
    */
   nir_foreach_block (block, impl) {
      nir_foreach_instr (instr, block) {
         nir_ssa_def *def = nir_instr_ssa_def(instr);
         if (!def)
            continue;

         def_state *state = &ctx.states[def->index];
         if (!state->can_move || state->must_stay)
            continue;

         ctx.def = def;
         nir_foreach_src(instr, update_src_value, &ctx);
         
         /* If this instruction is a candidate, its value shouldn't be
          * propagated so we skip dividing it.
          *
          * Note: if it's can_move but not a candidate, then all its users
          * must be can_move, so if there are no users then it must be dead.
          */
         if (!state->candidate && !state->must_stay) {
            if (state->can_move_users > 0)
               state->value /= state->can_move_users;
            else
               state->value = 0;
         }

         if (state->candidate) {
            state->benefit = state->value -
               options->rewrite_cost_cb(def, options->cb_data);

            if (state->benefit > 0) {
               options->def_size(def, &state->size, &state->align);
               total_size = ALIGN_POT(total_size, state->align);
               total_size += state->size;
               candidates[candidate_idx++] = state;
            }
         }
      }
   }

   assert(candidate_idx <= num_candidates);
   num_candidates = candidate_idx;

   if (num_candidates == 0) {
      *size = 0;
      free(ctx.states);
      free(candidates);
      return false;
   }

   /* Step 4: Figure out which candidates we're going to replace and assign an
    * offset. Assuming there is no expression sharing, this is similar to the
    * 0-1 knapsack problem, except when there is a gap introduced by
    * alignment. We use a well-known greedy approximation, sorting by value
    * divided by size.
    */

   if (total_size > options->preamble_storage_size) {
      qsort(candidates, num_candidates, sizeof(*candidates), candidate_sort);
   }

   unsigned offset = 0;
   for (unsigned i = 0; i < num_candidates; i++) {
      def_state *state = candidates[i];
      offset = ALIGN_POT(offset, state->align);

      if (offset + state->size > options->preamble_storage_size)
         break;

      state->replace = true;
      state->offset = offset;

      offset += state->size;
   }

   *size = offset;

   free(candidates);

   /* Step 5: Actually do the replacement. */
   struct hash_table *remap_table =
      _mesa_pointer_hash_table_create(NULL);
   nir_function_impl *preamble =
      nir_shader_get_preamble(impl->function->shader);
   nir_builder _b;
   nir_builder *b = &_b;
   nir_builder_init(b, preamble);
   b->cursor = nir_before_cf_list(&preamble->body);

   nir_foreach_block (block, impl) {
      nir_foreach_instr (instr, block) {
         nir_ssa_def *def = nir_instr_ssa_def(instr);
         if (!def)
            continue;

         def_state *state = &ctx.states[def->index];
         if (!state->can_move)
            continue;

         nir_instr *clone = nir_instr_clone_deep(impl->function->shader,
                                                 instr, remap_table);

         nir_builder_instr_insert(b, clone);

         if (clone->type == nir_instr_type_tex) {
            nir_tex_instr *tex = nir_instr_as_tex(clone);
            if (tex->op == nir_texop_tex) {
               /* For maximum compatibility, replace normal textures with
                * textureGrad with a gradient of 0.
                * TODO: Handle txb somehow.
                */
               b->cursor = nir_before_instr(clone);

               nir_ssa_def *zero =
                  nir_imm_zero(b, tex->coord_components - tex->is_array, 32);
               nir_tex_instr_add_src(tex, nir_tex_src_ddx, nir_src_for_ssa(zero));
               nir_tex_instr_add_src(tex, nir_tex_src_ddy, nir_src_for_ssa(zero));
               tex->op = nir_texop_txd;

               b->cursor = nir_after_instr(clone);
            }
         }

         if (state->replace) {
            nir_ssa_def *clone_def = nir_instr_ssa_def(clone);
            nir_store_preamble(b, clone_def, .base = state->offset);
         }
      }
   }

   nir_builder_init(b, impl);

   nir_foreach_block (block, impl) {
      nir_foreach_instr_safe (instr, block) {
         nir_ssa_def *def = nir_instr_ssa_def(instr);
         if (!def)
            continue;

         def_state *state = &ctx.states[def->index];
         if (!state->replace)
            continue;

         b->cursor = nir_before_instr(instr);

         nir_ssa_def *new_def =
            nir_load_preamble(b, def->num_components, def->bit_size,
                              .base = state->offset);


         nir_ssa_def_rewrite_uses(def, new_def);
         nir_instr_free_and_dce(instr);
      }
   }

   nir_metadata_preserve(impl,
                         nir_metadata_block_index |
                         nir_metadata_dominance);

   ralloc_free(remap_table);
   free(ctx.states);
   return true;
}

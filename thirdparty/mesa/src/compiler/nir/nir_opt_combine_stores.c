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
#include "nir_deref.h"

#include "util/bitscan.h"
#include "util/list.h"
#include "util/u_math.h"

/* Combine stores of vectors to the same deref into a single store.
 *
 * This per-block pass keeps track of stores of vectors to the same
 * destination and combines them into the last store of the sequence.  Dead
 * stores (or parts of the store) found during the process are removed.
 *
 * A pending combination becomes an actual combination in various situations:
 * at the end of the block, when another instruction uses the memory or due to
 * barriers.
 *
 * Besides vectors, the pass also look at array derefs of vectors.  For direct
 * array derefs, it works like a write mask access to the given component.
 * For indirect access there's no way to know before hand what component it
 * will overlap with, so the combination is finished -- the indirect remains
 * unmodified.
 */

/* Keep track of a group of stores that can be combined.  All stores share the
 * same destination.
 */
struct combined_store {
   struct list_head link;

   nir_component_mask_t write_mask;
   nir_deref_instr *dst;

   /* Latest store added.  It is reused when combining. */
   nir_intrinsic_instr *latest;

   /* Original store for each component.  The number of times a store appear
    * in this array is kept in the store's pass_flags.
    */
   nir_intrinsic_instr *stores[NIR_MAX_VEC_COMPONENTS];
};

struct combine_stores_state {
   nir_variable_mode modes;

   /* Pending store combinations. */
   struct list_head pending;

   /* Per function impl state. */
   nir_builder b;
   bool progress;


   /* Allocator and freelist to reuse structs between functions. */
   void *lin_ctx;
   struct list_head freelist;
};

static struct combined_store *
alloc_combined_store(struct combine_stores_state *state)
{
   struct combined_store *result;
   if (list_is_empty(&state->freelist)) {
      result = linear_zalloc_child(state->lin_ctx, sizeof(*result));
   } else {
      result = list_first_entry(&state->freelist,
                                struct combined_store,
                                link);
      list_del(&result->link);
      memset(result, 0, sizeof(*result));
   }
   return result;
}

static void
free_combined_store(struct combine_stores_state *state,
                    struct combined_store *combo)
{
   list_del(&combo->link);
   combo->write_mask = 0;
   list_add(&combo->link, &state->freelist);
}

static void
combine_stores(struct combine_stores_state *state,
                   struct combined_store *combo)
{
   assert(combo->latest);
   assert(combo->latest->intrinsic == nir_intrinsic_store_deref);

   /* If the combined writemask is the same as the latest store, we know there
    * is only one store in the combination, so nothing to combine.
    */
   if ((combo->write_mask & nir_intrinsic_write_mask(combo->latest)) ==
       combo->write_mask)
      return;

   state->b.cursor = nir_before_instr(&combo->latest->instr);

   /* Build a new vec, to be used as source for the combined store.  As it
    * gets build, remove previous stores that are not needed anymore.
    */
   nir_ssa_scalar comps[NIR_MAX_VEC_COMPONENTS] = {0};
   unsigned num_components = glsl_get_vector_elements(combo->dst->type);
   unsigned bit_size = combo->latest->src[1].ssa->bit_size;
   for (unsigned i = 0; i < num_components; i++) {
      nir_intrinsic_instr *store = combo->stores[i];
      if (combo->write_mask & (1 << i)) {
         assert(store);
         assert(store->src[1].is_ssa);

         /* If store->num_components == 1 then we are in the deref-of-vec case
          * and store->src[1] is a scalar.  Otherwise, we're a regular vector
          * load and we have to pick off a component.
          */
         comps[i] = nir_get_ssa_scalar(store->src[1].ssa, store->num_components == 1 ? 0 : i);

         assert(store->instr.pass_flags > 0);
         if (--store->instr.pass_flags == 0 && store != combo->latest)
            nir_instr_remove(&store->instr);
      } else {
         comps[i] = nir_get_ssa_scalar(nir_ssa_undef(&state->b, 1, bit_size), 0);
      }
   }
   assert(combo->latest->instr.pass_flags == 0);
   nir_ssa_def *vec = nir_vec_scalars(&state->b, comps, num_components);

   /* Fix the latest store with the combined information. */
   nir_intrinsic_instr *store = combo->latest;

   /* In this case, our store is as an array deref of a vector so we need to
    * rewrite it to use a deref to the whole vector.
    */
   if (store->num_components == 1) {
      store->num_components = num_components;
      nir_instr_rewrite_src(&store->instr, &store->src[0],
                            nir_src_for_ssa(&combo->dst->dest.ssa));
   }

   assert(store->num_components == num_components);
   nir_intrinsic_set_write_mask(store, combo->write_mask);
   nir_instr_rewrite_src(&store->instr, &store->src[1],
                         nir_src_for_ssa(vec));
   state->progress = true;
}

static void
combine_stores_with_deref(struct combine_stores_state *state,
                              nir_deref_instr *deref)
{
   if (!nir_deref_mode_may_be(deref, state->modes))
      return;

   list_for_each_entry_safe(struct combined_store, combo, &state->pending, link) {
      if (nir_compare_derefs(combo->dst, deref) & nir_derefs_may_alias_bit) {
         combine_stores(state, combo);
         free_combined_store(state, combo);
      }
   }
}

static void
combine_stores_with_modes(struct combine_stores_state *state,
                              nir_variable_mode modes)
{
   if ((state->modes & modes) == 0)
      return;

   list_for_each_entry_safe(struct combined_store, combo, &state->pending, link) {
      if (nir_deref_mode_may_be(combo->dst, modes)) {
         combine_stores(state, combo);
         free_combined_store(state, combo);
      }
   }
}

static struct combined_store *
find_matching_combined_store(struct combine_stores_state *state,
                             nir_deref_instr *deref)
{
   list_for_each_entry(struct combined_store, combo, &state->pending, link) {
      if (nir_compare_derefs(combo->dst, deref) & nir_derefs_equal_bit)
         return combo;
   }
   return NULL;
}

static void
update_combined_store(struct combine_stores_state *state,
                      nir_intrinsic_instr *intrin)
{
   nir_deref_instr *dst = nir_src_as_deref(intrin->src[0]);
   if (!nir_deref_mode_may_be(dst, state->modes))
      return;

   unsigned vec_mask;
   nir_deref_instr *vec_dst;

   if (glsl_type_is_vector(dst->type)) {
      vec_mask = nir_intrinsic_write_mask(intrin);
      vec_dst = dst;
   } else {
      /* Besides vectors, only direct array derefs of vectors are handled. */
      if (dst->deref_type != nir_deref_type_array ||
          !nir_src_is_const(dst->arr.index) ||
          !glsl_type_is_vector(nir_deref_instr_parent(dst)->type)) {
         combine_stores_with_deref(state, dst);
         return;
      }

      uint64_t index = nir_src_as_uint(dst->arr.index);
      vec_dst = nir_deref_instr_parent(dst);

      if (index >= glsl_get_vector_elements(vec_dst->type)) {
         /* Storing to an invalid index is a no-op. */
         nir_instr_remove(&intrin->instr);
         state->progress = true;
         return;
      }

      vec_mask = 1 << index;
   }

   struct combined_store *combo = find_matching_combined_store(state, vec_dst);
   if (!combo) {
      combo = alloc_combined_store(state);
      combo->dst = vec_dst;
      list_add(&combo->link, &state->pending);
   }

   /* Use pass_flags to reference count the store based on how many
    * components are still used by the combination.
    */
   intrin->instr.pass_flags = util_bitcount(vec_mask);
   combo->latest = intrin;

   /* Update the combined_store, clearing up older overlapping references. */
   combo->write_mask |= vec_mask;
   while (vec_mask) {
      unsigned i = u_bit_scan(&vec_mask);
      nir_intrinsic_instr *prev_store = combo->stores[i];

      if (prev_store) {
         if (--prev_store->instr.pass_flags == 0) {
            nir_instr_remove(&prev_store->instr);
         } else {
            assert(glsl_type_is_vector(
                      nir_src_as_deref(prev_store->src[0])->type));
            nir_component_mask_t prev_mask = nir_intrinsic_write_mask(prev_store);
            nir_intrinsic_set_write_mask(prev_store, prev_mask & ~(1 << i));
         }
         state->progress = true;
      }
      combo->stores[i] = combo->latest;
   }
}

static void
combine_stores_block(struct combine_stores_state *state, nir_block *block)
{
   nir_foreach_instr_safe(instr, block) {
      if (instr->type == nir_instr_type_call) {
         combine_stores_with_modes(state, nir_var_shader_out |
                                          nir_var_shader_temp |
                                          nir_var_function_temp |
                                          nir_var_mem_ssbo |
                                          nir_var_mem_shared |
                                          nir_var_mem_global);
         continue;
      }

      if (instr->type != nir_instr_type_intrinsic)
         continue;

      nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
      switch (intrin->intrinsic) {
      case nir_intrinsic_store_deref:
         if (nir_intrinsic_access(intrin) & ACCESS_VOLATILE) {
            nir_deref_instr *dst = nir_src_as_deref(intrin->src[0]);
            /* When we see a volatile store, we go ahead and combine all
             * previous non-volatile stores which touch that address and
             * specifically don't add the volatile store to the list.  This
             * way we guarantee that the volatile store isn't combined with
             * anything and no non-volatile stores are combined across a
             * volatile store.
             */
            combine_stores_with_deref(state, dst);
         } else {
            update_combined_store(state, intrin);
         }
         break;

      case nir_intrinsic_control_barrier:
      case nir_intrinsic_group_memory_barrier:
      case nir_intrinsic_memory_barrier:
         combine_stores_with_modes(state, nir_var_shader_out |
                                          nir_var_mem_ssbo |
                                          nir_var_mem_shared |
                                          nir_var_mem_global);
         break;

      case nir_intrinsic_memory_barrier_buffer:
         combine_stores_with_modes(state, nir_var_mem_ssbo |
                                          nir_var_mem_global);
         break;

      case nir_intrinsic_memory_barrier_shared:
         combine_stores_with_modes(state, nir_var_mem_shared);
         break;

      case nir_intrinsic_memory_barrier_tcs_patch:
         combine_stores_with_modes(state, nir_var_shader_out);
         break;

      case nir_intrinsic_scoped_barrier:
         if (nir_intrinsic_memory_semantics(intrin) & NIR_MEMORY_RELEASE) {
            combine_stores_with_modes(state,
                                      nir_intrinsic_memory_modes(intrin));
         }
         break;

      case nir_intrinsic_emit_vertex:
      case nir_intrinsic_emit_vertex_with_counter:
         combine_stores_with_modes(state, nir_var_shader_out);
         break;

      case nir_intrinsic_report_ray_intersection:
         combine_stores_with_modes(state, nir_var_mem_ssbo |
                                          nir_var_mem_global |
                                          nir_var_shader_call_data |
                                          nir_var_ray_hit_attrib);
         break;

      case nir_intrinsic_ignore_ray_intersection:
      case nir_intrinsic_terminate_ray:
         combine_stores_with_modes(state, nir_var_mem_ssbo |
                                          nir_var_mem_global |
                                          nir_var_shader_call_data);
         break;

      case nir_intrinsic_load_deref: {
         nir_deref_instr *src = nir_src_as_deref(intrin->src[0]);
         combine_stores_with_deref(state, src);
         break;
      }

      case nir_intrinsic_load_deref_block_intel:
      case nir_intrinsic_store_deref_block_intel: {
         /* Combine all the stores that may alias with the whole variable (or
          * cast).
          */
         nir_deref_instr *operand = nir_src_as_deref(intrin->src[0]);
         while (nir_deref_instr_parent(operand))
            operand = nir_deref_instr_parent(operand);
         assert(operand->deref_type == nir_deref_type_var ||
                operand->deref_type == nir_deref_type_cast);

         combine_stores_with_deref(state, operand);
         break;
      }

      case nir_intrinsic_copy_deref:
      case nir_intrinsic_memcpy_deref: {
         nir_deref_instr *dst = nir_src_as_deref(intrin->src[0]);
         nir_deref_instr *src = nir_src_as_deref(intrin->src[1]);
         combine_stores_with_deref(state, dst);
         combine_stores_with_deref(state, src);
         break;
      }

      case nir_intrinsic_trace_ray:
      case nir_intrinsic_execute_callable:
      case nir_intrinsic_rt_trace_ray:
      case nir_intrinsic_rt_execute_callable: {
         nir_deref_instr *payload =
            nir_src_as_deref(*nir_get_shader_call_payload_src(intrin));
         combine_stores_with_deref(state, payload);
         break;
      }

      case nir_intrinsic_deref_atomic_add:
      case nir_intrinsic_deref_atomic_imin:
      case nir_intrinsic_deref_atomic_umin:
      case nir_intrinsic_deref_atomic_imax:
      case nir_intrinsic_deref_atomic_umax:
      case nir_intrinsic_deref_atomic_and:
      case nir_intrinsic_deref_atomic_or:
      case nir_intrinsic_deref_atomic_xor:
      case nir_intrinsic_deref_atomic_exchange:
      case nir_intrinsic_deref_atomic_comp_swap: {
         nir_deref_instr *dst = nir_src_as_deref(intrin->src[0]);
         combine_stores_with_deref(state, dst);
         break;
      }

      default:
         break;
      }
   }

   /* At the end of the block, try all the remaining combinations. */
   combine_stores_with_modes(state, state->modes);
}

static bool
combine_stores_impl(struct combine_stores_state *state, nir_function_impl *impl)
{
   state->progress = false;
   nir_builder_init(&state->b, impl);

   nir_foreach_block(block, impl)
      combine_stores_block(state, block);

   if (state->progress) {
      nir_metadata_preserve(impl, nir_metadata_block_index |
                                  nir_metadata_dominance);
   } else {
      nir_metadata_preserve(impl, nir_metadata_all);
   }

   return state->progress;
}

bool
nir_opt_combine_stores(nir_shader *shader, nir_variable_mode modes)
{
   void *mem_ctx = ralloc_context(NULL);
   struct combine_stores_state state = {
      .modes   = modes,
      .lin_ctx = linear_zalloc_parent(mem_ctx, 0),
   };

   list_inithead(&state.pending);
   list_inithead(&state.freelist);

   bool progress = false;

   nir_foreach_function(function, shader) {
      if (!function->impl)
         continue;
      progress |= combine_stores_impl(&state, function->impl);
   }

   ralloc_free(mem_ctx);
   return progress;
}

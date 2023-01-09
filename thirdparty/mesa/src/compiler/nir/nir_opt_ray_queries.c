/*
 * Copyright © 2021 Intel Corporation
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

#include "util/hash_table.h"
#include "util/set.h"
#include "util/macros.h"
#include "util/u_dynarray.h"

/** @file nir_opt_ray_queries.c
 *
 * 1. Remove ray queries that the shader is not using the result of.
 * 2. Combine ray queries which are not simultaneously.
 */

static void
mark_query_read(struct set *queries,
                nir_intrinsic_instr *intrin)
{
   nir_ssa_def *rq_def = intrin->src[0].ssa;

   nir_variable *query;
   if (rq_def->parent_instr->type == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *load_deref =
         nir_instr_as_intrinsic(rq_def->parent_instr);
      assert(load_deref->intrinsic == nir_intrinsic_load_deref);

      query = nir_intrinsic_get_var(load_deref, 0);
   } else if (rq_def->parent_instr->type == nir_instr_type_deref) {
      query = nir_deref_instr_get_variable(
         nir_instr_as_deref(rq_def->parent_instr));
   } else {
      return;
   }
   assert(query);

   _mesa_set_add(queries, query);
}

static void
nir_find_ray_queries_read(struct set *queries,
                          nir_shader *shader)
{
   nir_foreach_function(function, shader) {
      nir_function_impl *impl = function->impl;

      if (!impl)
         continue;

      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
            switch (intrin->intrinsic) {
            case nir_intrinsic_rq_proceed:
               if (list_length(&intrin->dest.ssa.uses) > 0 ||
                   list_length(&intrin->dest.ssa.if_uses) > 0)
                  mark_query_read(queries, intrin);
               break;
            case nir_intrinsic_rq_load:
               mark_query_read(queries, intrin);
               break;
            default:
               break;
            }
         }
      }
   }
}

static bool
nir_replace_unread_queries_instr(nir_builder *b, nir_instr *instr, void *data)
{
   struct set *queries = data;

   if (instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
   switch (intrin->intrinsic) {
   case nir_intrinsic_rq_initialize:
   case nir_intrinsic_rq_terminate:
   case nir_intrinsic_rq_generate_intersection:
   case nir_intrinsic_rq_confirm_intersection:
      break;
   case nir_intrinsic_rq_proceed:
      break;
   default:
      return false;
   }

   nir_variable *query = nir_intrinsic_get_var(intrin, 0);
   assert(query);

   struct set_entry *entry = _mesa_set_search(queries, query);
   if (entry)
      return false;

   if (intrin->intrinsic == nir_intrinsic_rq_load) {
      assert(list_is_empty(&intrin->dest.ssa.uses));
      assert(list_is_empty(&intrin->dest.ssa.if_uses));
   }

   nir_instr_remove(instr);

   return true;
}

bool
nir_opt_ray_queries(nir_shader *shader)
{
   struct set *read_queries = _mesa_pointer_set_create(NULL);
   nir_find_ray_queries_read(read_queries, shader);

   bool progress =
      nir_shader_instructions_pass(shader,
                                   nir_replace_unread_queries_instr,
                                   nir_metadata_block_index |
                                   nir_metadata_dominance,
                                   read_queries);

   /* Update the number of queries if some have been removed. */
   if (progress) {
      nir_remove_dead_derefs(shader);
      nir_remove_dead_variables(shader,
                                nir_var_shader_temp | nir_var_function_temp,
                                NULL);
   }

   _mesa_set_destroy(read_queries, NULL);

   return progress;
}

/**
 * Merge ray queries that are not used in parallel to reduce scratch memory:
 * 
 * 1. Store all the ray queries we will consider into an array for
 *    convenient access. Ignore arrays since it would be really complex
 *    to handle and will be rare in praxis.
 *
 * 2. Count the number of ray query ranges and allocate the required ranges.
 *
 * 3. Populate the ray query range array. A range is started and termninated
 *    rq_initialize (the terminating rq_initialize will be the start of the
 *    next range). There are two hazards:
 *
 *    1. rq_initialize can be inside some form of controlflow which can result
 *       in incorrect ranges and invalid merging.
 * 
 *       SOLUTION: Discard the entire ray query when encountering an
 *                 instruction that is not dominated by the rq_initialize
 *                 of the range.
 * 
 *    2. With loops, we can underestimate the range because the state may
 *       have to be preserved for multiple iterations.
 * 
 *       SOLUTION: Track parent loops.
 * 
 * 4. Try to rewrite the variables. For that, we iterate over every ray query
 *    and try to move its ranges to the preceding ray queries.
 */

struct rq_range {
   nir_variable *variable;

   uint32_t first;
   uint32_t last;

   struct util_dynarray instrs;
   struct set *loops;
};

#define RQ_NEW_INDEX_NONE 0xFFFFFFFF

static bool
count_ranges(struct nir_builder *b, nir_instr *instr, void *data)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *intrinsic = nir_instr_as_intrinsic(instr);
   if (intrinsic->intrinsic == nir_intrinsic_rq_initialize)
      (*(uint32_t *) data)++;

   return false;
}

static nir_cf_node *
get_parent_loop(nir_cf_node *node)
{
   nir_cf_node *result = NULL;
   while (node) {
      if (node->type == nir_cf_node_loop)
         result = node;

      node = node->parent;
   }
   return result;
}

bool
nir_opt_ray_query_ranges(nir_shader *shader)
{
   assert(exec_list_length(&shader->functions) == 1);

   struct nir_function *func =
      (struct nir_function *)exec_list_get_head_const(&shader->functions);
   assert(func->impl);

   uint32_t ray_query_count = 0;
   nir_foreach_variable_in_shader(var, shader) {
      if (!var->data.ray_query || glsl_type_is_array(var->type))
         continue;
      ray_query_count++;
   }
   nir_foreach_function_temp_variable(var, func->impl) {
      if (!var->data.ray_query || glsl_type_is_array(var->type))
         continue;
      ray_query_count++;
   }

   if (ray_query_count <= 1) {
      nir_metadata_preserve(func->impl, nir_metadata_all);
      return false;
   }

   void *mem_ctx = ralloc_context(NULL);

   nir_metadata_require(func->impl, nir_metadata_instr_index | nir_metadata_dominance);

   nir_variable **ray_queries = ralloc_array(mem_ctx, nir_variable*, ray_query_count);
   ray_query_count = 0;

   nir_foreach_variable_in_shader(var, shader) {
      if (!var->data.ray_query || glsl_type_is_array(var->type))
         continue;
      
      ray_queries[ray_query_count] = var;
      ray_query_count++;
   }

   nir_foreach_function_temp_variable(var, func->impl) {
      if (!var->data.ray_query || glsl_type_is_array(var->type))
         continue;
      
      ray_queries[ray_query_count] = var;
      ray_query_count++;
   }

   uint32_t range_count = 0;
   nir_shader_instructions_pass(shader, count_ranges, nir_metadata_all, &range_count);

   struct rq_range *ranges = rzalloc_array(mem_ctx, struct rq_range, range_count);

   struct hash_table *range_indices = _mesa_pointer_hash_table_create(mem_ctx);
   uint32_t target_index = 0;

   nir_foreach_block(block, func->impl) {
      nir_cf_node *parent_loop = get_parent_loop(&block->cf_node);

      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;

         nir_intrinsic_instr *intrinsic = nir_instr_as_intrinsic(instr);
         if (!nir_intrinsic_is_ray_query(intrinsic->intrinsic))
            continue;

         nir_deref_instr *ray_query_deref =
            nir_instr_as_deref(intrinsic->src[0].ssa->parent_instr);

         if (ray_query_deref->deref_type != nir_deref_type_var)
            continue;

         if (intrinsic->intrinsic == nir_intrinsic_rq_initialize) {
            _mesa_hash_table_insert(range_indices, ray_query_deref->var,
                                    (void *)(uintptr_t)target_index);

            ranges[target_index].variable = ray_query_deref->var;
            ranges[target_index].first = instr->index;
            ranges[target_index].last = instr->index;
            util_dynarray_init(&ranges[target_index].instrs, mem_ctx);
            ranges[target_index].loops = _mesa_pointer_set_create(mem_ctx);

            target_index++;
         }

         struct hash_entry *index_entry =
            _mesa_hash_table_search(range_indices, ray_query_deref->var);
         struct rq_range *range = ranges + (uintptr_t)index_entry->data;
         
         if (intrinsic->intrinsic != nir_intrinsic_rq_initialize) {
            /* If the initialize instruction does not dominate every other
             * instruction in the range, we have to reject the enire query
             * since we can not be certain about the ranges:
             * 
             * rayQuery rq;
             * if (i == 0)
             *    init(rq);
             * ...             <-- Another ray query that would get merged.
             * if (i == 1)
             *    init(rq);    <--+
             * if (i == 0)        |
             *    proceed(rq); <--+ Not dominated by init!
             * if (i == 1)
             *    proceed(rq);
             */
            nir_instr *init = *util_dynarray_element(&range->instrs, nir_instr *, 0);
            if (!nir_block_dominates(init->block, instr->block)) {
               for (uint32_t i = 0; i < ray_query_count; i++) {
                  if (ray_queries[i] == ray_query_deref->var) {
                     ray_queries[i] = NULL;
                     break;
                  }
               }

               continue;
            }

            range->last = MAX2(range->last, instr->index);
         }

         util_dynarray_append(&range->instrs, nir_instr *, instr);

         if (parent_loop)
            _mesa_set_add(range->loops, parent_loop);
      }
   }

   range_count = target_index;

   /* Try to push ray query ranges 'down'. */
   for (uint32_t rq_index = 1; rq_index < ray_query_count; rq_index++) {
      if (!ray_queries[rq_index])
         continue;

      for (uint32_t dom_rq_index = 0; dom_rq_index < rq_index; dom_rq_index++) {
         if (!ray_queries[dom_rq_index])
            continue;

         bool collides = false;

         for (uint32_t range_index = 0; range_index < range_count; range_index++) {
            if (ranges[range_index].variable != ray_queries[rq_index])
               continue;

            for (uint32_t dom_range_index = 0; dom_range_index < range_count; dom_range_index++) {
               if (ranges[dom_range_index].variable != ray_queries[dom_rq_index])
                  continue;

               if (!(ranges[dom_range_index].first > ranges[range_index].last ||
                     ranges[dom_range_index].last < ranges[range_index].first)) {
                  collides = true;
                  break;
               }

               if (_mesa_set_intersects(ranges[dom_range_index].loops,
                                        ranges[range_index].loops)) {
                  collides = true;
                  break;
               }
            }

            if (collides)
               break;
         }

         if (collides)
            continue;

         for (uint32_t range_index = 0; range_index < range_count; range_index++) {
            if (ranges[range_index].variable != ray_queries[rq_index])
               continue;

            ranges[range_index].variable = ray_queries[dom_rq_index];
         }
      }
   }

   /* Remap the ray query derefs to the new variables. */
   bool progress = false;
   for (uint32_t range_index = 0; range_index < range_count; range_index++) {
      struct rq_range *range = ranges + range_index;
      util_dynarray_foreach(&range->instrs, nir_instr *, instr) {
         nir_intrinsic_instr *intrinsic = nir_instr_as_intrinsic(*instr);
         nir_deref_instr *ray_query_deref =
            nir_instr_as_deref(intrinsic->src[0].ssa->parent_instr);
         if (ray_query_deref->var != range->variable) {
            ray_query_deref->var = range->variable;
            progress = true;
         }
      }
   }

   nir_metadata_preserve(func->impl, nir_metadata_all);

   /* Remove dead ray queries. */
   if (progress) {
      nir_remove_dead_derefs(shader);
      nir_remove_dead_variables(shader, nir_var_shader_temp | nir_var_function_temp,
                                NULL);
   }

   ralloc_free(mem_ctx);

   return progress;
}

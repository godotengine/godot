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

#include "nir.h"
#include "nir_builder.h"
#include "nir_deref.h"

#include "util/u_dynarray.h"

/**
 * Elimination of dead writes based on derefs.
 *
 * Dead writes are stores and copies that write to a deref, which then gets
 * another write before it was used (read or sourced for a copy).  Those
 * writes can be removed since they don't affect anything.
 *
 * For derefs that refer to a memory area that can be read after the program,
 * the last write is considered used.  The presence of certain instructions
 * may also cause writes to be considered used, e.g. memory barrier (in this case
 * the value must be written as other thread might use it).
 *
 * The write mask for store instructions is considered, so it is possible that
 * a store is removed because of the combination of other stores overwritten
 * its value.
 */

/* Entry for unused_writes arrays. */
struct write_entry {
   /* If NULL indicates the entry is free to be reused. */
   nir_intrinsic_instr *intrin;
   nir_component_mask_t mask;
   nir_deref_instr *dst;
};

static void
clear_unused_for_modes(struct util_dynarray *unused_writes, nir_variable_mode modes)
{
   util_dynarray_foreach_reverse(unused_writes, struct write_entry, entry) {
      if (nir_deref_mode_may_be(entry->dst, modes))
         *entry = util_dynarray_pop(unused_writes, struct write_entry);
   }
}

static void
clear_unused_for_read(struct util_dynarray *unused_writes, nir_deref_instr *src)
{
   util_dynarray_foreach_reverse(unused_writes, struct write_entry, entry) {
      if (nir_compare_derefs(src, entry->dst) & nir_derefs_may_alias_bit)
         *entry = util_dynarray_pop(unused_writes, struct write_entry);
   }
}

static bool
update_unused_writes(struct util_dynarray *unused_writes,
                     nir_intrinsic_instr *intrin,
                     nir_deref_instr *dst, nir_component_mask_t mask)
{
   bool progress = false;

   /* This pass assumes that destination of copies and stores are derefs that
    * end in a vector or scalar (it is OK to have wildcards or indirects for
    * arrays).
    */
   assert(glsl_type_is_vector_or_scalar(dst->type));

   /* Find writes that are unused and can be removed. */
   util_dynarray_foreach_reverse(unused_writes, struct write_entry, entry) {
      nir_deref_compare_result comp = nir_compare_derefs(dst, entry->dst);
      if (comp & nir_derefs_a_contains_b_bit) {
         entry->mask &= ~mask;
         if (entry->mask == 0) {
            nir_instr_remove(&entry->intrin->instr);
            *entry = util_dynarray_pop(unused_writes, struct write_entry);
            progress = true;
         }
      }
   }

   /* Add the new write to the unused array. */
   struct write_entry new_entry = {
      .intrin = intrin,
      .mask = mask,
      .dst = dst,
   };

   util_dynarray_append(unused_writes, struct write_entry, new_entry);

   return progress;
}

static bool
remove_dead_write_vars_local(void *mem_ctx, nir_shader *shader, nir_block *block)
{
   bool progress = false;

   struct util_dynarray unused_writes;
   util_dynarray_init(&unused_writes, mem_ctx);

   nir_foreach_instr_safe(instr, block) {
      if (instr->type == nir_instr_type_call) {
         clear_unused_for_modes(&unused_writes, nir_var_shader_out |
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
      case nir_intrinsic_control_barrier:
      case nir_intrinsic_group_memory_barrier:
      case nir_intrinsic_memory_barrier: {
         clear_unused_for_modes(&unused_writes, nir_var_shader_out |
                                                nir_var_mem_ssbo |
                                                nir_var_mem_shared |
                                                nir_var_mem_global);
         break;
      }

      case nir_intrinsic_memory_barrier_buffer:
         clear_unused_for_modes(&unused_writes, nir_var_mem_ssbo |
                                                nir_var_mem_global);
         break;

      case nir_intrinsic_memory_barrier_shared:
         clear_unused_for_modes(&unused_writes, nir_var_mem_shared);
         break;

      case nir_intrinsic_memory_barrier_tcs_patch:
         clear_unused_for_modes(&unused_writes, nir_var_shader_out);
         break;

      case nir_intrinsic_scoped_barrier: {
         if (nir_intrinsic_memory_semantics(intrin) & NIR_MEMORY_RELEASE) {
            clear_unused_for_modes(&unused_writes,
                                   nir_intrinsic_memory_modes(intrin));
         }
         break;
      }

      case nir_intrinsic_emit_vertex:
      case nir_intrinsic_emit_vertex_with_counter: {
         clear_unused_for_modes(&unused_writes, nir_var_shader_out);
         break;
      }

      case nir_intrinsic_execute_callable:
      case nir_intrinsic_rt_execute_callable: {
         /* Mark payload as it can be used by the callee */
         nir_deref_instr *src = nir_src_as_deref(intrin->src[1]);
         clear_unused_for_read(&unused_writes, src);
         break;
      }

      case nir_intrinsic_trace_ray:
      case nir_intrinsic_rt_trace_ray: {
         /* Mark payload as it can be used by the callees */
         nir_deref_instr *src = nir_src_as_deref(intrin->src[10]);
         clear_unused_for_read(&unused_writes, src);
         break;
      }

      case nir_intrinsic_load_deref: {
         nir_deref_instr *src = nir_src_as_deref(intrin->src[0]);
         if (nir_deref_mode_must_be(src, nir_var_read_only_modes))
            break;
         clear_unused_for_read(&unused_writes, src);
         break;
      }

      case nir_intrinsic_store_deref: {
         nir_deref_instr *dst = nir_src_as_deref(intrin->src[0]);

         if (nir_intrinsic_access(intrin) & ACCESS_VOLATILE) {
            /* Consider a volatile write to also be a sort of read.  This
             * prevents us from deleting a non-volatile write just before a
             * volatile write thanks to a non-volatile write afterwards.  It's
             * quite the corner case, but this should be safer and more
             * predictable for the programmer than allowing two non-volatile
             * writes to be combined with a volatile write between them.
             */
            clear_unused_for_read(&unused_writes, dst);
            break;
         }

         nir_component_mask_t mask = nir_intrinsic_write_mask(intrin);
         progress |= update_unused_writes(&unused_writes, intrin, dst, mask);
         break;
      }

      case nir_intrinsic_copy_deref: {
         nir_deref_instr *src = nir_src_as_deref(intrin->src[1]);
         nir_deref_instr *dst = nir_src_as_deref(intrin->src[0]);

         if (nir_intrinsic_dst_access(intrin) & ACCESS_VOLATILE) {
            clear_unused_for_read(&unused_writes, src);
            clear_unused_for_read(&unused_writes, dst);
            break;
         }

         /* Self-copy is removed. */
         if (nir_compare_derefs(src, dst) & nir_derefs_equal_bit) {
            nir_instr_remove(instr);
            progress = true;
            break;
         }

         clear_unused_for_read(&unused_writes, src);
         nir_component_mask_t mask = (1 << glsl_get_vector_elements(dst->type)) - 1;
         progress |= update_unused_writes(&unused_writes, intrin, dst, mask);
         break;
      }

      default:
         break;
      }
   }

   /* All unused writes at the end of the block are kept, since we can't be
    * sure they'll be overwritten or not with local analysis only.
    */

   return progress;
}

static bool
remove_dead_write_vars_impl(void *mem_ctx, nir_shader *shader, nir_function_impl *impl)
{
   bool progress = false;

   nir_metadata_require(impl, nir_metadata_block_index);

   nir_foreach_block(block, impl)
      progress |= remove_dead_write_vars_local(mem_ctx, shader, block);

   if (progress) {
      nir_metadata_preserve(impl, nir_metadata_block_index |
                                  nir_metadata_dominance);
   } else {
      nir_metadata_preserve(impl, nir_metadata_all);
   }

   return progress;
}

bool
nir_opt_dead_write_vars(nir_shader *shader)
{
   void *mem_ctx = ralloc_context(NULL);
   bool progress = false;

   nir_foreach_function(function, shader) {
      if (!function->impl)
         continue;
      progress |= remove_dead_write_vars_impl(mem_ctx, shader, function->impl);
   }

   ralloc_free(mem_ctx);
   return progress;
}

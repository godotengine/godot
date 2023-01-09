/*
 * Copyright © 2020 Valve Corporation
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

/*
 * Replaces make availability/visible semantics on barriers with
 * ACCESS_COHERENT on memory loads/stores
 */

#include "nir/nir.h"
#include "shader_enums.h"

static bool
get_intrinsic_info(nir_intrinsic_instr *intrin, nir_variable_mode *modes,
                   bool *reads, bool *writes)
{
   switch (intrin->intrinsic) {
   case nir_intrinsic_image_deref_load:
   case nir_intrinsic_image_deref_sparse_load:
      *modes = nir_src_as_deref(intrin->src[0])->modes;
      *reads = true;
      break;
   case nir_intrinsic_image_deref_store:
      *modes = nir_src_as_deref(intrin->src[0])->modes;
      *writes = true;
      break;
   case nir_intrinsic_image_deref_atomic_add:
   case nir_intrinsic_image_deref_atomic_fadd:
   case nir_intrinsic_image_deref_atomic_umin:
   case nir_intrinsic_image_deref_atomic_imin:
   case nir_intrinsic_image_deref_atomic_umax:
   case nir_intrinsic_image_deref_atomic_imax:
   case nir_intrinsic_image_deref_atomic_fmin:
   case nir_intrinsic_image_deref_atomic_fmax:
   case nir_intrinsic_image_deref_atomic_and:
   case nir_intrinsic_image_deref_atomic_or:
   case nir_intrinsic_image_deref_atomic_xor:
   case nir_intrinsic_image_deref_atomic_exchange:
   case nir_intrinsic_image_deref_atomic_comp_swap:
      *modes = nir_src_as_deref(intrin->src[0])->modes;
      *reads = true;
      *writes = true;
      break;
   case nir_intrinsic_load_ssbo:
      *modes = nir_var_mem_ssbo;
      *reads = true;
      break;
   case nir_intrinsic_store_ssbo:
      *modes = nir_var_mem_ssbo;
      *writes = true;
      break;
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
   case nir_intrinsic_ssbo_atomic_fcomp_swap:
   case nir_intrinsic_ssbo_atomic_fmax:
   case nir_intrinsic_ssbo_atomic_fmin:
      *modes = nir_var_mem_ssbo;
      *reads = true;
      *writes = true;
      break;
   case nir_intrinsic_load_global:
      *modes = nir_var_mem_global;
      *reads = true;
      break;
   case nir_intrinsic_store_global:
      *modes = nir_var_mem_global;
      *writes = true;
      break;
   case nir_intrinsic_global_atomic_add:
   case nir_intrinsic_global_atomic_imin:
   case nir_intrinsic_global_atomic_umin:
   case nir_intrinsic_global_atomic_imax:
   case nir_intrinsic_global_atomic_umax:
   case nir_intrinsic_global_atomic_and:
   case nir_intrinsic_global_atomic_or:
   case nir_intrinsic_global_atomic_xor:
   case nir_intrinsic_global_atomic_exchange:
   case nir_intrinsic_global_atomic_comp_swap:
   case nir_intrinsic_global_atomic_fadd:
   case nir_intrinsic_global_atomic_fcomp_swap:
   case nir_intrinsic_global_atomic_fmax:
   case nir_intrinsic_global_atomic_fmin:
      *modes = nir_var_mem_global;
      *reads = true;
      *writes = true;
      break;
   case nir_intrinsic_load_deref:
      *modes = nir_src_as_deref(intrin->src[0])->modes;
      *reads = true;
      break;
   case nir_intrinsic_store_deref:
      *modes = nir_src_as_deref(intrin->src[0])->modes;
      *writes = true;
      break;
   case nir_intrinsic_deref_atomic_add:
   case nir_intrinsic_deref_atomic_imin:
   case nir_intrinsic_deref_atomic_umin:
   case nir_intrinsic_deref_atomic_imax:
   case nir_intrinsic_deref_atomic_umax:
   case nir_intrinsic_deref_atomic_and:
   case nir_intrinsic_deref_atomic_or:
   case nir_intrinsic_deref_atomic_xor:
   case nir_intrinsic_deref_atomic_exchange:
   case nir_intrinsic_deref_atomic_comp_swap:
   case nir_intrinsic_deref_atomic_fadd:
   case nir_intrinsic_deref_atomic_fmin:
   case nir_intrinsic_deref_atomic_fmax:
   case nir_intrinsic_deref_atomic_fcomp_swap:
      *modes = nir_src_as_deref(intrin->src[0])->modes;
      *reads = true;
      *writes = true;
      break;
   default:
      return false;
   }
   return true;
}

static bool
visit_instr(nir_instr *instr, uint32_t *cur_modes, unsigned vis_avail_sem)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;
   nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);

   if (intrin->intrinsic == nir_intrinsic_scoped_barrier &&
       (nir_intrinsic_memory_semantics(intrin) & vis_avail_sem)) {
      *cur_modes |= nir_intrinsic_memory_modes(intrin);

      unsigned semantics = nir_intrinsic_memory_semantics(intrin);
      nir_intrinsic_set_memory_semantics(
         intrin, semantics & ~vis_avail_sem);
      return true;
   }

   if (!*cur_modes)
      return false; /* early exit */

   nir_variable_mode modes;
   bool reads = false, writes = false;
   if (!get_intrinsic_info(intrin, &modes, &reads, &writes))
      return false;

   if (!reads && vis_avail_sem == NIR_MEMORY_MAKE_VISIBLE)
      return false;
   if (!writes && vis_avail_sem == NIR_MEMORY_MAKE_AVAILABLE)
      return false;

   if (!nir_intrinsic_has_access(intrin))
      return false;

   unsigned access = nir_intrinsic_access(intrin);

   if (access & (ACCESS_NON_READABLE | ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER | ACCESS_COHERENT))
      return false;

   if (*cur_modes & modes) {
      nir_intrinsic_set_access(intrin, access | ACCESS_COHERENT);
      return true;
   }

   return false;
}

static bool
lower_make_visible(nir_cf_node *cf_node, uint32_t *cur_modes)
{
   bool progress = false;
   switch (cf_node->type) {
   case nir_cf_node_block: {
      nir_block *block = nir_cf_node_as_block(cf_node);
      nir_foreach_instr(instr, block)
         progress |= visit_instr(instr, cur_modes, NIR_MEMORY_MAKE_VISIBLE);
      break;
   }
   case nir_cf_node_if: {
      nir_if *nif = nir_cf_node_as_if(cf_node);
      uint32_t cur_modes_then = *cur_modes;
      uint32_t cur_modes_else = *cur_modes;
      foreach_list_typed(nir_cf_node, if_node, node, &nif->then_list)
         progress |= lower_make_visible(if_node, &cur_modes_then);
      foreach_list_typed(nir_cf_node, if_node, node, &nif->else_list)
         progress |= lower_make_visible(if_node, &cur_modes_else);
      *cur_modes |= cur_modes_then | cur_modes_else;
      break;
   }
   case nir_cf_node_loop: {
      nir_loop *loop = nir_cf_node_as_loop(cf_node);
      bool loop_progress;
      do {
         loop_progress = false;
         foreach_list_typed(nir_cf_node, loop_node, node, &loop->body)
            loop_progress |= lower_make_visible(loop_node, cur_modes);
         progress |= loop_progress;
      } while (loop_progress);
      break;
   }
   case nir_cf_node_function:
      unreachable("Invalid cf type");
   }
   return progress;
}

static bool
lower_make_available(nir_cf_node *cf_node, uint32_t *cur_modes)
{
   bool progress = false;
   switch (cf_node->type) {
   case nir_cf_node_block: {
      nir_block *block = nir_cf_node_as_block(cf_node);
      nir_foreach_instr_reverse(instr, block)
         progress |= visit_instr(instr, cur_modes, NIR_MEMORY_MAKE_AVAILABLE);
      break;
   }
   case nir_cf_node_if: {
      nir_if *nif = nir_cf_node_as_if(cf_node);
      uint32_t cur_modes_then = *cur_modes;
      uint32_t cur_modes_else = *cur_modes;
      foreach_list_typed_reverse(nir_cf_node, if_node, node, &nif->then_list)
         progress |= lower_make_available(if_node, &cur_modes_then);
      foreach_list_typed_reverse(nir_cf_node, if_node, node, &nif->else_list)
         progress |= lower_make_available(if_node, &cur_modes_else);
      *cur_modes |= cur_modes_then | cur_modes_else;
      break;
   }
   case nir_cf_node_loop: {
      nir_loop *loop = nir_cf_node_as_loop(cf_node);
      bool loop_progress;
      do {
         loop_progress = false;
         foreach_list_typed_reverse(nir_cf_node, loop_node, node, &loop->body)
            loop_progress |= lower_make_available(loop_node, cur_modes);
         progress |= loop_progress;
      } while (loop_progress);
      break;
   }
   case nir_cf_node_function:
      unreachable("Invalid cf type");
   }
   return progress;
}

bool
nir_lower_memory_model(nir_shader *shader)
{
   bool progress = false;

   nir_function_impl *impl = nir_shader_get_entrypoint(shader);
   struct exec_list *cf_list = &impl->body;

   uint32_t modes = 0;
   foreach_list_typed(nir_cf_node, cf_node, node, cf_list)
      progress |= lower_make_visible(cf_node, &modes);

   modes = 0;
   foreach_list_typed_reverse(nir_cf_node, cf_node, node, cf_list)
      progress |= lower_make_available(cf_node, &modes);

   if (progress)
      nir_metadata_preserve(impl, nir_metadata_block_index | nir_metadata_dominance);
   else
      nir_metadata_preserve(impl, nir_metadata_all);

   return progress;
}

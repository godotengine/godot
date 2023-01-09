/*
 * Copyright © 2016 Intel Corporation
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
 */

#include "nir_control_flow.h"
#include "nir_builder.h"

/**
 * This file implements an optimization for multiview. Some GPU's have a
 * special mode which allows the vertex shader (or last stage in the geometry
 * pipeline) to create multiple primitives in different layers of the
 * framebuffer at once by writing multiple copies of gl_Position. The
 * assumption is that in most uses of multiview, the only use of gl_ViewIndex
 * is to change the position to implement the parallax effect, and other
 * varyings will be the same between the different views. We put the body of
 * the original vertex shader in a loop, writing to a different copy of
 * gl_Position each loop iteration, and then let other optimizations clean up
 * the mess.
 */

static bool
shader_writes_to_memory(nir_shader *shader)
{
   /* With multiview, we would need to ensure that memory writes happen either
    * once or once per view. Since combination of multiview and memory writes
    * is not expected, we'll just skip this optimization in this case.
    */

   nir_function_impl *entrypoint = nir_shader_get_entrypoint(shader);

   nir_foreach_block(block, entrypoint) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);

         switch (intrin->intrinsic) {
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
         case nir_intrinsic_store_shared:
         case nir_intrinsic_store_shared2_amd:
         case nir_intrinsic_shared_atomic_add:
         case nir_intrinsic_shared_atomic_imin:
         case nir_intrinsic_shared_atomic_umin:
         case nir_intrinsic_shared_atomic_imax:
         case nir_intrinsic_shared_atomic_umax:
         case nir_intrinsic_shared_atomic_and:
         case nir_intrinsic_shared_atomic_or:
         case nir_intrinsic_shared_atomic_xor:
         case nir_intrinsic_shared_atomic_exchange:
         case nir_intrinsic_shared_atomic_comp_swap:
         case nir_intrinsic_task_payload_atomic_add:
         case nir_intrinsic_task_payload_atomic_imin:
         case nir_intrinsic_task_payload_atomic_umin:
         case nir_intrinsic_task_payload_atomic_imax:
         case nir_intrinsic_task_payload_atomic_umax:
         case nir_intrinsic_task_payload_atomic_and:
         case nir_intrinsic_task_payload_atomic_or:
         case nir_intrinsic_task_payload_atomic_xor:
         case nir_intrinsic_task_payload_atomic_exchange:
         case nir_intrinsic_task_payload_atomic_comp_swap:
         case nir_intrinsic_task_payload_atomic_fadd:
         case nir_intrinsic_task_payload_atomic_fmin:
         case nir_intrinsic_task_payload_atomic_fmax:
         case nir_intrinsic_task_payload_atomic_fcomp_swap:
         case nir_intrinsic_image_deref_store:
         case nir_intrinsic_image_deref_atomic_add:
         case nir_intrinsic_image_deref_atomic_fadd:
         case nir_intrinsic_image_deref_atomic_umin:
         case nir_intrinsic_image_deref_atomic_umax:
         case nir_intrinsic_image_deref_atomic_imin:
         case nir_intrinsic_image_deref_atomic_imax:
         case nir_intrinsic_image_deref_atomic_fmin:
         case nir_intrinsic_image_deref_atomic_fmax:
         case nir_intrinsic_image_deref_atomic_and:
         case nir_intrinsic_image_deref_atomic_or:
         case nir_intrinsic_image_deref_atomic_xor:
         case nir_intrinsic_image_deref_atomic_exchange:
         case nir_intrinsic_image_deref_atomic_comp_swap:
            return true;

         default:
            /* Keep walking. */
            break;
         }
      }
   }

   return false;
}

bool
nir_shader_uses_view_index(nir_shader *shader)
{
   nir_function_impl *entrypoint = nir_shader_get_entrypoint(shader);

   nir_foreach_block(block, entrypoint) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;

         nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
         if (intrin->intrinsic == nir_intrinsic_load_view_index)
            return true;
      }
   }

   return false;
}

static bool
shader_only_position_uses_view_index(nir_shader *shader)
{
   nir_shader *shader_no_position = nir_shader_clone(NULL, shader);
   nir_function_impl *entrypoint = nir_shader_get_entrypoint(shader_no_position);

   /* Remove the store position from a cloned shader. */
   nir_foreach_block(block, entrypoint) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;

         nir_intrinsic_instr *store = nir_instr_as_intrinsic(instr);
         if (store->intrinsic != nir_intrinsic_store_deref)
            continue;

         nir_variable *var = nir_intrinsic_get_var(store, 0);
         if (var->data.location != VARYING_SLOT_POS)
            continue;

         nir_instr_remove(&store->instr);
      }
   }

   /* Clean up shader so unused load_view_index intrinsics are removed. */
   bool progress;
   do {
      progress = false;
      progress |= nir_opt_dead_cf(shader_no_position);

      /* Peephole select will drop if-blocks that have then and else empty,
       * which will remove the usage of an SSA in the condition.
       */
      progress |= nir_opt_peephole_select(shader_no_position, 0, false, false);

      progress |= nir_opt_dce(shader_no_position);
   } while (progress);

   bool uses_view_index = nir_shader_uses_view_index(shader_no_position);

   ralloc_free(shader_no_position);
   return !uses_view_index;
}

/* Return true if it's safe to call nir_lower_multiview() on this vertex
 * shader. Note that this only handles driver-agnostic checks, i.e. things
 * which would make nir_lower_multiview() incorrect. Any driver-specific
 * checks, e.g. for sufficient varying space or performance considerations,
 * should be handled in the driver.
 *
 * Note that we don't handle the more complex checks needed for lowering
 * pipelines with geometry or tessellation shaders.
 */

bool
nir_can_lower_multiview(nir_shader *shader)
{
   bool writes_position = false;
   nir_foreach_shader_out_variable(var, shader) {
      if (var->data.location == VARYING_SLOT_POS) {
         writes_position = true;
         break;
      }
   }

   /* Don't bother handling this edge case. */
   if (!writes_position)
      return false;

   return !shader_writes_to_memory(shader) &&
          shader_only_position_uses_view_index(shader);
}

/**
 * The lowering. Call with the last active geometry stage.
 */

bool
nir_lower_multiview(nir_shader *shader, uint32_t view_mask)
{
   assert(shader->info.stage != MESA_SHADER_FRAGMENT);
   int view_count = util_bitcount(view_mask);

   nir_function_impl *entrypoint = nir_shader_get_entrypoint(shader);

   /* Update position to refer to an array. */
   nir_variable *pos_var = NULL;
   nir_foreach_shader_out_variable(var, shader) {
      if (var->data.location == VARYING_SLOT_POS) {
         assert(var->type == glsl_vec4_type());
         var->type = glsl_array_type(glsl_vec4_type(), view_count, 0);
         var->data.per_view = true;
         shader->info.per_view_outputs |= VARYING_BIT_POS;
         pos_var = var;
         break;
      }
   }

   assert(pos_var);

   nir_cf_list body;
   nir_cf_list_extract(&body, &entrypoint->body);

   nir_builder b;
   nir_builder_init(&b, entrypoint);
   b.cursor = nir_after_cf_list(&entrypoint->body);

   /* Loop Index will go from 0 to view_count. */
   nir_variable *loop_index_var =
      nir_local_variable_create(entrypoint, glsl_uint_type(), "loop_index");
   nir_deref_instr *loop_index_deref = nir_build_deref_var(&b, loop_index_var);
   nir_store_deref(&b, loop_index_deref, nir_imm_int(&b, 0), 1);

   /* Array of view index values that are active in the loop.  Note that the
    * loop index only matches the view index if there are no gaps in the
    * view_mask.
    */
   nir_variable *view_index_var = nir_local_variable_create(
      entrypoint, glsl_array_type(glsl_uint_type(), view_count, 0), "view_index");
   nir_deref_instr *view_index_deref = nir_build_deref_var(&b, view_index_var);
   {
      int array_position = 0;
      uint32_t view_mask_temp = view_mask;
      while (view_mask_temp) {
         uint32_t view_index = u_bit_scan(&view_mask_temp);
         nir_store_deref(&b, nir_build_deref_array_imm(&b, view_index_deref, array_position),
                         nir_imm_int(&b, view_index), 1);
         array_position++;
      }
   }

   /* Create the equivalent of
    *
    *    while (true):
    *       if (loop_index >= view_count):
    *          break
    *
    *       view_index = active_indices[loop_index]
    *       pos_deref = &pos[loop_index]
    *
    *       # Placeholder for the body to be reinserted.
    *
    *       loop_index += 1
    *
    * Later both `view_index` and `pos_deref` will be used to rewrite the
    * original shader body.
    */

   nir_loop* loop = nir_push_loop(&b);

   nir_ssa_def *loop_index = nir_load_deref(&b, loop_index_deref);
   nir_ssa_def *cmp = nir_ige(&b, loop_index, nir_imm_int(&b, view_count));
   nir_if *loop_check = nir_push_if(&b, cmp);
   nir_jump(&b, nir_jump_break);
   nir_pop_if(&b, loop_check);

   nir_ssa_def *view_index =
      nir_load_deref(&b, nir_build_deref_array(&b, view_index_deref, loop_index));
   nir_deref_instr *pos_deref =
      nir_build_deref_array(&b, nir_build_deref_var(&b, pos_var), loop_index);

   nir_store_deref(&b, loop_index_deref, nir_iadd_imm(&b, loop_index, 1), 1);
   nir_pop_loop(&b, loop);

   /* Reinsert the body. */
   b.cursor = nir_after_instr(&pos_deref->instr);
   nir_cf_reinsert(&body, b.cursor);

   nir_foreach_block(block, entrypoint) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;

         nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);

         switch (intrin->intrinsic) {
         case nir_intrinsic_load_view_index: {
            assert(intrin->dest.is_ssa);
            nir_ssa_def_rewrite_uses(&intrin->dest.ssa, view_index);
            break;
         }

         case nir_intrinsic_store_deref: {
            nir_variable *var = nir_intrinsic_get_var(intrin, 0);
            if (var == pos_var) {
               nir_deref_instr *old_deref = nir_src_as_deref(intrin->src[0]);

               nir_instr_rewrite_src(instr, &intrin->src[0],
                                     nir_src_for_ssa(&pos_deref->dest.ssa));

               /* Remove old deref since it has the wrong type. */
               nir_deref_instr_remove_if_unused(old_deref);
            }
            break;
         }

         case nir_intrinsic_load_deref:
            if (nir_intrinsic_get_var(intrin, 0) == pos_var) {
               unreachable("Should have lowered I/O to temporaries "
                           "so no load_deref on position output is expected.");
            }
            break;

         case nir_intrinsic_copy_deref:
            unreachable("Should have lowered copy_derefs at this point");
            break;

         default:
            /* Do nothing. */
            break;
         }
      }
   }

   nir_metadata_preserve(entrypoint, nir_metadata_none);
   return true;
}


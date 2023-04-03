/*
 * Copyright © Microsoft Corporation
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "nir.h"
#include "nir_builder.h"
#include "nir_control_flow.h"

#include "dxil_nir.h"

static void
remove_hs_intrinsics(nir_function_impl *impl)
{
   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_store_output &&
             intr->intrinsic != nir_intrinsic_memory_barrier_tcs_patch &&
             intr->intrinsic != nir_intrinsic_control_barrier)
            continue;
         nir_instr_remove(instr);
      }
   }
   nir_metadata_preserve(impl, nir_metadata_block_index | nir_metadata_dominance);
}

static void
add_instr_and_srcs_to_set(struct set *instr_set, nir_instr *instr);

static bool
add_srcs_to_set(nir_src *src, void *state)
{
   assert(src->is_ssa);
   add_instr_and_srcs_to_set(state, src->ssa->parent_instr);
   return true;
}

static void
add_instr_and_srcs_to_set(struct set *instr_set, nir_instr *instr)
{
   bool was_already_found = false;
   _mesa_set_search_or_add(instr_set, instr, &was_already_found);
   if (!was_already_found)
      nir_foreach_src(instr, add_srcs_to_set, instr_set);
}

static void
prune_patch_function_to_intrinsic_and_srcs(nir_function_impl *impl)
{
   struct set *instr_set = _mesa_pointer_set_create(NULL);

   /* Do this in two phases:
    * 1. Find all instructions that contribute to a store_output and add them to
    *    the set. Also, add instructions that contribute to control flow.
    * 2. Erase every instruction that isn't in the set
    */
   nir_foreach_block(block, impl) {
      nir_if *following_if = nir_block_get_following_if(block);
      if (following_if) {
         assert(following_if->condition.is_ssa);
         add_instr_and_srcs_to_set(instr_set, following_if->condition.ssa->parent_instr);
      }
      nir_foreach_instr_safe(instr, block) {
         if (instr->type == nir_instr_type_intrinsic) {
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_store_output &&
                intr->intrinsic != nir_intrinsic_memory_barrier_tcs_patch)
               continue;
         } else if (instr->type != nir_instr_type_jump)
            continue;
         add_instr_and_srcs_to_set(instr_set, instr);
      }
   }

   nir_foreach_block_reverse(block, impl) {
      nir_foreach_instr_reverse_safe(instr, block) {
         struct set_entry *entry = _mesa_set_search(instr_set, instr);
         if (!entry)
            nir_instr_remove(instr);
      }
   }

   _mesa_set_destroy(instr_set, NULL);
}

static nir_cursor
get_cursor_for_instr_without_cf(nir_instr *instr)
{
   nir_block *block = instr->block;
   if (block->cf_node.parent->type == nir_cf_node_function)
      return nir_before_instr(instr);

   do {
      block = nir_cf_node_as_block(nir_cf_node_prev(block->cf_node.parent));
   } while (block->cf_node.parent->type != nir_cf_node_function);
   return nir_after_block_before_jump(block);
}

struct tcs_patch_loop_state {
   nir_ssa_def *deref, *count;
   nir_cursor begin_cursor, end_cursor, insert_cursor;
   nir_loop *loop;
};

static void
start_tcs_loop(nir_builder *b, struct tcs_patch_loop_state *state, nir_deref_instr *loop_var_deref)
{
   if (!loop_var_deref)
      return;

   nir_store_deref(b, loop_var_deref, nir_imm_int(b, 0), 1);
   state->loop = nir_push_loop(b);
   state->count = nir_load_deref(b, loop_var_deref);
   nir_push_if(b, nir_ige(b, state->count, nir_imm_int(b, b->impl->function->shader->info.tess.tcs_vertices_out)));
   nir_jump(b, nir_jump_break);
   nir_pop_if(b, NULL);
   state->insert_cursor = b->cursor;
   nir_store_deref(b, loop_var_deref, nir_iadd_imm(b, state->count, 1), 1);
   nir_pop_loop(b, state->loop);
}

static void
end_tcs_loop(nir_builder *b, struct tcs_patch_loop_state *state)
{
   if (!state->loop)
      return;

   nir_cf_list extracted;
   nir_cf_extract(&extracted, state->begin_cursor, state->end_cursor);
   nir_cf_reinsert(&extracted, state->insert_cursor);

   *state = (struct tcs_patch_loop_state ){ 0 };
}

/* In HLSL/DXIL, the hull (tesselation control) shader is split into two:
 * 1. The main hull shader, which runs once per output control point.
 * 2. A patch constant function, which runs once overall.
 * In GLSL/NIR, these are combined. Each invocation must write to the output
 * array with a constant gl_InvocationID, which is (apparently) lowered to an
 * if/else ladder in nir. Each invocation must write the same value to patch
 * constants - or else undefined behavior strikes. NIR uses store_output to
 * write the patch constants, and store_per_vertex_output to write the control
 * point values.
 * 
 * We clone the NIR function to produce 2: one with the store_output intrinsics
 * removed, which becomes the main shader (only writes control points), and one
 * with everything that doesn't contribute to store_output removed, which becomes
 * the patch constant function.
 * 
 * For the patch constant function, if the expressions rely on gl_InvocationID,
 * then we need to run the resulting logic in a loop, using the loop counter to
 * replace gl_InvocationID. This loop can be terminated when a barrier is hit. If
 * gl_InvocationID is used again after the barrier, then another loop needs to begin.
 */
void
dxil_nir_split_tess_ctrl(nir_shader *nir, nir_function **patch_const_func)
{
   assert(nir->info.stage == MESA_SHADER_TESS_CTRL);
   assert(exec_list_length(&nir->functions) == 1);
   nir_function_impl *entrypoint = nir_shader_get_entrypoint(nir);

   *patch_const_func = nir_function_create(nir, "PatchConstantFunc");
   nir_function_impl *patch_const_func_impl = nir_function_impl_clone(nir, entrypoint);
   (*patch_const_func)->impl = patch_const_func_impl;
   patch_const_func_impl->function = *patch_const_func;

   remove_hs_intrinsics(entrypoint);
   prune_patch_function_to_intrinsic_and_srcs(patch_const_func_impl);

   /* Kill dead references to the invocation ID from the patch const func so we don't
    * insert unnecessarily loops
    */
   bool progress;
   do {
      progress = false;
      progress |= nir_opt_dead_cf(nir);
      progress |= nir_opt_dce(nir);
   } while (progress);

   /* Now, the patch constant function needs to be split into blocks and loops.
    * The series of instructions up to the first block containing a load_invocation_id
    * will run sequentially. Then a loop is inserted so load_invocation_id will load the
    * loop counter. This loop continues until a barrier is reached, when the loop
    * is closed and the process begins again.
    * 
    * First, sink load_invocation_id so that it's present on both sides of barriers.
    * Each use gets a unique load of the invocation ID.
    */
   nir_builder b;
   nir_builder_init(&b, patch_const_func_impl);
   nir_foreach_block(block, patch_const_func_impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_load_invocation_id ||
             list_length(&intr->dest.ssa.uses) +
             list_length(&intr->dest.ssa.if_uses) <= 1)
            continue;
         nir_foreach_use_safe(src, &intr->dest.ssa) {
            b.cursor = nir_before_src(src, false);
            nir_instr_rewrite_src_ssa(src->parent_instr, src, nir_load_invocation_id(&b));
         }
         nir_foreach_if_use_safe(src, &intr->dest.ssa) {
            b.cursor = nir_before_src(src, true);
            nir_if_rewrite_condition_ssa(src->parent_if, src, nir_load_invocation_id(&b));
         }
         nir_instr_remove(instr);
      }
   }

   /* Now replace those invocation ID loads with loads of a local variable that's used as a loop counter */
   nir_variable *loop_var = NULL;
   nir_deref_instr *loop_var_deref = NULL;
   struct tcs_patch_loop_state state = { 0 };
   nir_foreach_block_safe(block, patch_const_func_impl) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         switch (intr->intrinsic) {
         case nir_intrinsic_load_invocation_id: {
            if (!loop_var) {
               loop_var = nir_local_variable_create(patch_const_func_impl, glsl_int_type(), "PatchConstInvocId");
               b.cursor = nir_before_cf_list(&patch_const_func_impl->body);
               loop_var_deref = nir_build_deref_var(&b, loop_var);
            }
            if (!state.loop) {
               b.cursor = state.begin_cursor = get_cursor_for_instr_without_cf(instr);
               start_tcs_loop(&b, &state, loop_var_deref);
            }
            nir_ssa_def_rewrite_uses(&intr->dest.ssa, state.count);
            break;
         }
         case nir_intrinsic_memory_barrier_tcs_patch:
            /* The GL tessellation spec says:
             * The barrier() function may only be called inside the main entry point of the tessellation control shader
             * and may not be called in potentially divergent flow control.  In particular, barrier() may not be called
             * inside a switch statement, in either sub-statement of an if statement, inside a do, for, or while loop,
             * or at any point after a return statement in the function main().
             * 
             * Therefore, we should be at function-level control flow.
             */
            assert(nir_cursors_equal(nir_before_instr(instr), get_cursor_for_instr_without_cf(instr)));
            state.end_cursor = nir_before_instr(instr);
            end_tcs_loop(&b, &state);
            nir_instr_remove(instr);
            break;
         default:
            break;
         }
      }
   }
   state.end_cursor = nir_after_block_before_jump(nir_impl_last_block(patch_const_func_impl));
   end_tcs_loop(&b, &state);
}

struct remove_tess_level_accesses_data {
   unsigned location;
   unsigned size;
};

static bool
remove_tess_level_accesses(nir_builder *b, nir_instr *instr, void *_data)
{
   struct remove_tess_level_accesses_data *data = _data;
   if (instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
   if (intr->intrinsic != nir_intrinsic_store_output &&
       intr->intrinsic != nir_intrinsic_load_input)
      return false;

   nir_io_semantics io = nir_intrinsic_io_semantics(intr);
   if (io.location != data->location)
      return false;

   if (nir_intrinsic_component(intr) < data->size)
      return false;

   if (intr->intrinsic == nir_intrinsic_store_output) {
      assert(intr->src[0].is_ssa && intr->src[0].ssa->num_components == 1);
      nir_instr_remove(instr);
   } else {
      b->cursor = nir_after_instr(instr);
      assert(intr->dest.is_ssa && intr->dest.ssa.num_components == 1);
      nir_ssa_def_rewrite_uses(&intr->dest.ssa, nir_ssa_undef(b, 1, intr->dest.ssa.bit_size));
   }
   return true;
}

/* Update the types of the tess level variables and remove writes to removed components.
 * GL always has a 4-component outer tess level and 2-component inner, while D3D requires
 * the number of components to vary based on the primitive mode.
 * The 4 and 2 is for quads, while triangles are 3 and 1, and lines are 2 and 0.
 */
bool
dxil_nir_fixup_tess_level_for_domain(nir_shader *nir)
{
   bool progress = false;
   if (nir->info.tess._primitive_mode != TESS_PRIMITIVE_QUADS) {
      nir_foreach_variable_with_modes_safe(var, nir, nir_var_shader_out | nir_var_shader_in) {
         unsigned new_array_size = 4;
         unsigned old_array_size = glsl_array_size(var->type);
         if (var->data.location == VARYING_SLOT_TESS_LEVEL_OUTER) {
            new_array_size = nir->info.tess._primitive_mode == TESS_PRIMITIVE_TRIANGLES ? 3 : 2;
            assert(var->data.compact && (old_array_size == 4 || old_array_size == new_array_size));
         } else if (var->data.location == VARYING_SLOT_TESS_LEVEL_INNER) {
            new_array_size = nir->info.tess._primitive_mode == TESS_PRIMITIVE_TRIANGLES ? 1 : 0;
            assert(var->data.compact && (old_array_size == 2 || old_array_size == new_array_size));
         } else
            continue;

         if (new_array_size == old_array_size)
            continue;

         progress = true;
         if (new_array_size)
            var->type = glsl_array_type(glsl_float_type(), new_array_size, 0);
         else {
            exec_node_remove(&var->node);
            ralloc_free(var);
         }

         struct remove_tess_level_accesses_data pass_data = {
            .location = var->data.location,
            .size = new_array_size
         };

         nir_shader_instructions_pass(nir, remove_tess_level_accesses,
            nir_metadata_block_index | nir_metadata_dominance, &pass_data);
      }
   }
   return progress;
}

static bool
tcs_update_deref_input_types(nir_builder *b, nir_instr *instr, void *data)
{
   if (instr->type != nir_instr_type_deref)
      return false;

   nir_deref_instr *deref = nir_instr_as_deref(instr);
   if (deref->deref_type != nir_deref_type_var)
      return false;

   nir_variable *var = deref->var;
   deref->type = var->type;
   return true;
}

bool
dxil_nir_set_tcs_patches_in(nir_shader *nir, unsigned num_control_points)
{
   bool progress = false;
   nir_foreach_variable_with_modes(var, nir, nir_var_shader_in) {
      if (nir_is_arrayed_io(var, MESA_SHADER_TESS_CTRL)) {
         var->type = glsl_array_type(glsl_get_array_element(var->type), num_control_points, 0);
         progress = true;
      }
   }

   if (progress)
      nir_shader_instructions_pass(nir, tcs_update_deref_input_types, nir_metadata_all, NULL);

   return progress;
}

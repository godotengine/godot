/*
 * Copyright © 2022 Valve Corporation
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
 * Authors:
 *    Timur Kristóf
 *
 */

#include "nir.h"
#include "nir_builder.h"
#include "util/u_math.h"

typedef struct {
   uint32_t task_count_shared_addr;
} lower_task_nv_state;

typedef struct {
   /* If true, lower all task_payload I/O to use shared memory. */
   bool payload_in_shared;
   /* Shared memory address where task_payload will be located. */
   uint32_t payload_shared_addr;
   uint32_t payload_offset_in_bytes;
} lower_task_state;

static bool
lower_nv_task_output(nir_builder *b,
                     nir_instr *instr,
                     void *state)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;

   lower_task_nv_state *s = (lower_task_nv_state *) state;
   nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);

   switch (intrin->intrinsic) {
   case nir_intrinsic_load_output: {
      b->cursor = nir_after_instr(instr);
      nir_ssa_def *load =
         nir_load_shared(b, 1, 32, nir_imm_int(b, 0),
                         .base = s->task_count_shared_addr);
      nir_ssa_def_rewrite_uses(&intrin->dest.ssa, load);
      nir_instr_remove(instr);
      return true;
   }

   case nir_intrinsic_store_output: {
      b->cursor = nir_after_instr(instr);
      nir_ssa_def *store_val = intrin->src[0].ssa;
      nir_store_shared(b, store_val, nir_imm_int(b, 0),
                       .base = s->task_count_shared_addr);
      nir_instr_remove(instr);
      return true;
   }

   default:
      return false;
   }
}

static void
append_launch_mesh_workgroups_to_nv_task(nir_builder *b,
                                         lower_task_nv_state *s)
{
   /* At the beginning of the shader, write 0 to the task count.
    * This ensures that 0 mesh workgroups are launched when the
    * shader doesn't write the TASK_COUNT output.
    */
   b->cursor = nir_before_cf_list(&b->impl->body);
   nir_ssa_def *zero = nir_imm_int(b, 0);
   nir_store_shared(b, zero, zero, .base = s->task_count_shared_addr);

   nir_scoped_barrier(b,
         .execution_scope = NIR_SCOPE_WORKGROUP,
         .memory_scope = NIR_SCOPE_WORKGROUP,
         .memory_semantics = NIR_MEMORY_RELEASE,
         .memory_modes = nir_var_mem_shared);

   /* At the end of the shader, read the task count from shared memory
    * and emit launch_mesh_workgroups.
    */
   b->cursor = nir_after_cf_list(&b->impl->body);

   nir_scoped_barrier(b,
         .execution_scope = NIR_SCOPE_WORKGROUP,
         .memory_scope = NIR_SCOPE_WORKGROUP,
         .memory_semantics = NIR_MEMORY_ACQUIRE,
         .memory_modes = nir_var_mem_shared);

   nir_ssa_def *task_count =
      nir_load_shared(b, 1, 32, zero, .base = s->task_count_shared_addr);

   /* NV_mesh_shader doesn't offer to choose which task_payload variable
    * should be passed to mesh shaders, we just pass all.
    */
   uint32_t range = b->shader->info.task_payload_size;

   nir_ssa_def *one = nir_imm_int(b, 1);
   nir_ssa_def *dispatch_3d = nir_vec3(b, task_count, one, one);
   nir_launch_mesh_workgroups(b, dispatch_3d, .base = 0, .range = range);
}

/**
 * For NV_mesh_shader:
 * Task shaders only have 1 output, TASK_COUNT which is a 32-bit
 * unsigned int that contains the 1-dimensional mesh dispatch size.
 * This output should behave like a shared variable.
 *
 * We lower this output to a shared variable and then we emit
 * the new launch_mesh_workgroups intrinsic at the end of the shader.
 */
static void
nir_lower_nv_task_count(nir_shader *shader)
{
   lower_task_nv_state state = {
      .task_count_shared_addr = ALIGN(shader->info.shared_size, 4),
   };

   shader->info.shared_size += 4;
   nir_shader_instructions_pass(shader, lower_nv_task_output,
                                nir_metadata_none, &state);

   nir_function_impl *impl = nir_shader_get_entrypoint(shader);
   nir_builder builder;
   nir_builder_init(&builder, impl);

   append_launch_mesh_workgroups_to_nv_task(&builder, &state);
   nir_metadata_preserve(impl, nir_metadata_none);
}

static nir_intrinsic_op
shared_opcode_for_task_payload(nir_intrinsic_op task_payload_op)
{
   switch (task_payload_op) {
#define OP(O) case nir_intrinsic_task_payload_##O: return nir_intrinsic_shared_##O;
   OP(atomic_exchange)
   OP(atomic_comp_swap)
   OP(atomic_add)
   OP(atomic_imin)
   OP(atomic_umin)
   OP(atomic_imax)
   OP(atomic_umax)
   OP(atomic_and)
   OP(atomic_or)
   OP(atomic_xor)
   OP(atomic_fadd)
   OP(atomic_fmin)
   OP(atomic_fmax)
   OP(atomic_fcomp_swap)
#undef OP
   case nir_intrinsic_load_task_payload:
      return nir_intrinsic_load_shared;
   case nir_intrinsic_store_task_payload:
      return nir_intrinsic_store_shared;
   default:
      unreachable("Invalid task payload atomic");
   }
}

static bool
lower_task_payload_to_shared(nir_builder *b,
                             nir_intrinsic_instr *intrin,
                             lower_task_state *s)
{
   /* This assumes that shared and task_payload intrinsics
    * have the same number of sources and same indices.
    */
   unsigned base = nir_intrinsic_base(intrin);
   intrin->intrinsic = shared_opcode_for_task_payload(intrin->intrinsic);
   nir_intrinsic_set_base(intrin, base + s->payload_shared_addr);

   return true;
}

static void
copy_shared_to_payload(nir_builder *b,
                       unsigned num_components,
                       nir_ssa_def *addr,
                       unsigned shared_base,
                       unsigned off)
{
   /* Read from shared memory. */
   nir_ssa_def *copy = nir_load_shared(b, num_components, 32, addr,
                                       .align_mul = 16,
                                       .base = shared_base + off);

   /* Write to task payload memory. */
   nir_store_task_payload(b, copy, addr, .base = off);
}

static void
emit_shared_to_payload_copy(nir_builder *b,
                            uint32_t payload_addr,
                            uint32_t payload_size,
                            lower_task_state *s)
{
   /* Copy from shared memory to task payload using as much parallelism
    * as possible. This is achieved by splitting the work into max 3 phases:
    * 1) copy maximum number of vec4s using all invocations within workgroup
    * 2) copy maximum number of vec4s using some invocations
    * 3) copy remaining dwords (< 4) using only the first invocation
    */
   const unsigned invocations = b->shader->info.workgroup_size[0] *
                                b->shader->info.workgroup_size[1] *
                                b->shader->info.workgroup_size[2];
   const unsigned vec4size = 16;
   const unsigned whole_wg_vec4_copies = payload_size / vec4size;
   const unsigned vec4_copies_per_invocation = whole_wg_vec4_copies / invocations;
   const unsigned remaining_vec4_copies = whole_wg_vec4_copies % invocations;
   const unsigned remaining_dwords =
         DIV_ROUND_UP(payload_size
                       - vec4size * vec4_copies_per_invocation * invocations
                       - vec4size * remaining_vec4_copies,
                      4);
   const unsigned base_shared_addr = s->payload_shared_addr + payload_addr;

   nir_ssa_def *invocation_index = nir_load_local_invocation_index(b);
   nir_ssa_def *addr = nir_imul_imm(b, invocation_index, vec4size);

   /* Wait for all previous shared stores to finish.
    * This is necessary because we placed the payload in shared memory.
    */
   nir_scoped_barrier(b, .execution_scope = NIR_SCOPE_WORKGROUP,
                         .memory_scope = NIR_SCOPE_WORKGROUP,
                         .memory_semantics = NIR_MEMORY_ACQ_REL,
                         .memory_modes = nir_var_mem_shared);

   /* Payload_size is a size of user-accessible payload, but on some
    * hardware (e.g. Intel) payload has a private header, which we have
    * to offset (payload_offset_in_bytes).
    */
   unsigned off = s->payload_offset_in_bytes;

   /* Technically dword-alignment is not necessary for correctness
    * of the code below, but even if backend implements unaligned
    * load/stores, they will very likely be slow(er).
    */
   assert(off % 4 == 0);

   /* Copy full vec4s using all invocations in workgroup. */
   for (unsigned i = 0; i < vec4_copies_per_invocation; ++i) {
      copy_shared_to_payload(b, vec4size / 4, addr, base_shared_addr, off);
      off += vec4size * invocations;
   }

   /* Copy full vec4s using only the invocations needed to not overflow. */
   if (remaining_vec4_copies > 0) {
      assert(remaining_vec4_copies < invocations);

      nir_ssa_def *cmp = nir_ilt(b, invocation_index, nir_imm_int(b, remaining_vec4_copies));
      nir_if *if_stmt = nir_push_if(b, cmp);
      {
         copy_shared_to_payload(b, vec4size / 4, addr, base_shared_addr, off);
      }
      nir_pop_if(b, if_stmt);
      off += vec4size * remaining_vec4_copies;
   }

   /* Copy the last few dwords not forming full vec4. */
   if (remaining_dwords > 0) {
      assert(remaining_dwords < 4);
      nir_ssa_def *cmp = nir_ieq(b, invocation_index, nir_imm_int(b, 0));
      nir_if *if_stmt = nir_push_if(b, cmp);
      {
         copy_shared_to_payload(b, remaining_dwords, addr, base_shared_addr, off);
      }
      nir_pop_if(b, if_stmt);
      off += remaining_dwords * 4;
   }

   assert(s->payload_offset_in_bytes + ALIGN(payload_size, 4) == off);
}

static bool
lower_task_launch_mesh_workgroups(nir_builder *b,
                                  nir_intrinsic_instr *intrin,
                                  lower_task_state *s)
{
   if (s->payload_in_shared) {
      /* Copy the payload from shared memory.
       * Because launch_mesh_workgroups may only occur in
       * workgroup-uniform control flow, here we assume that
       * all invocations in the workgroup are active and therefore
       * they can all participate in the copy.
       *
       * TODO: Skip the copy when the mesh dispatch size is (0, 0, 0).
       *       This is problematic because the dispatch size can be divergent,
       *       and may differ accross subgroups.
       */

      uint32_t payload_addr = nir_intrinsic_base(intrin);
      uint32_t payload_size = nir_intrinsic_range(intrin);

      b->cursor = nir_before_instr(&intrin->instr);
      emit_shared_to_payload_copy(b, payload_addr, payload_size, s);
   }

   /* The launch_mesh_workgroups intrinsic is a terminating instruction,
    * so let's delete everything after it.
    */
   b->cursor = nir_after_instr(&intrin->instr);
   nir_block *current_block = nir_cursor_current_block(b->cursor);

   /* Delete following instructions in the current block. */
   nir_foreach_instr_reverse_safe(instr, current_block) {
      if (instr == &intrin->instr)
         break;
      nir_instr_remove(instr);
   }

   /* Delete following CF at the same level. */
   b->cursor = nir_after_instr(&intrin->instr);
   nir_cf_list extracted;
   nir_cf_node *end_node = &current_block->cf_node;
   while (!nir_cf_node_is_last(end_node))
      end_node = nir_cf_node_next(end_node);
   nir_cf_extract(&extracted, b->cursor, nir_after_cf_node(end_node));
   nir_cf_delete(&extracted);

   /* Terminate the task shader. */
   b->cursor = nir_after_instr(&intrin->instr);
   nir_jump(b, nir_jump_return);

   return true;
}

static bool
lower_task_intrin(nir_builder *b,
                  nir_instr *instr,
                  void *state)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;

   lower_task_state *s = (lower_task_state *) state;
   nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);

   switch (intrin->intrinsic) {
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
   case nir_intrinsic_store_task_payload:
   case nir_intrinsic_load_task_payload:
      if (s->payload_in_shared)
         return lower_task_payload_to_shared(b, intrin, s);
      return false;
   case nir_intrinsic_launch_mesh_workgroups:
      return lower_task_launch_mesh_workgroups(b, intrin, s);
   default:
      return false;
   }
}

static bool
requires_payload_in_shared(nir_shader *shader, bool atomics, bool small_types)
{
   nir_foreach_function(func, shader) {
      if (!func->impl)
         continue;

      nir_foreach_block(block, func->impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
            switch (intrin->intrinsic) {
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
                  if (atomics)
                     return true;
                  break;
               case nir_intrinsic_load_task_payload:
                  if (small_types && nir_dest_bit_size(intrin->dest) < 32)
                     return true;
                  break;
               case nir_intrinsic_store_task_payload:
                  if (small_types && nir_src_bit_size(intrin->src[0]) < 32)
                     return true;
                  break;
               default:
                  break;
            }
         }
      }
   }

   return false;
}

static bool
nir_lower_task_intrins(nir_shader *shader, lower_task_state *state)
{
   return nir_shader_instructions_pass(shader, lower_task_intrin,
                                       nir_metadata_none, state);
}

/**
 * Common Task Shader lowering to make the job of the backends easier.
 *
 * - Lowers NV_mesh_shader TASK_COUNT output to launch_mesh_workgroups.
 * - Removes all code after launch_mesh_workgroups, enforcing the
 *   fact that it's a terminating instruction.
 * - Ensures that task shaders always have at least one
 *   launch_mesh_workgroups instruction, so the backend doesn't
 *   need to implement a special case when the shader doesn't have it.
 * - Optionally, implements task_payload using shared memory when
 *   task_payload atomics are used.
 *   This is useful when the backend is otherwise not capable of
 *   handling the same atomic features as it can for shared memory.
 *   If this is used, the backend only has to implement the basic
 *   load/store operations for task_payload.
 *
 * Note, this pass operates on lowered explicit I/O intrinsics, so
 * it should be called after nir_lower_io + nir_lower_explicit_io.
 */
bool
nir_lower_task_shader(nir_shader *shader,
                      nir_lower_task_shader_options options)
{
   if (shader->info.stage != MESA_SHADER_TASK)
      return false;

   nir_function_impl *impl = nir_shader_get_entrypoint(shader);
   nir_builder builder;
   nir_builder_init(&builder, impl);

   if (shader->info.outputs_written & BITFIELD64_BIT(VARYING_SLOT_TASK_COUNT)) {
      /* NV_mesh_shader:
       * If the shader writes TASK_COUNT, lower that to emit
       * the new launch_mesh_workgroups intrinsic instead.
       */
      NIR_PASS_V(shader, nir_lower_nv_task_count);
   } else {
      /* To make sure that task shaders always have a code path that
       * executes a launch_mesh_workgroups, let's add one at the end.
       * If the shader already had a launch_mesh_workgroups by any chance,
       * this will be removed.
       */
      nir_block *last_block = nir_impl_last_block(impl);
      builder.cursor = nir_after_block_before_jump(last_block);
      nir_launch_mesh_workgroups(&builder, nir_imm_zero(&builder, 3, 32));
   }

   bool atomics = options.payload_to_shared_for_atomics;
   bool small_types = options.payload_to_shared_for_small_types;
   bool payload_in_shared = (atomics || small_types) &&
                            requires_payload_in_shared(shader, atomics, small_types);

   lower_task_state state = {
      .payload_shared_addr = ALIGN(shader->info.shared_size, 16),
      .payload_in_shared = payload_in_shared,
      .payload_offset_in_bytes = options.payload_offset_in_bytes,
   };

   if (payload_in_shared)
      shader->info.shared_size =
         state.payload_shared_addr + shader->info.task_payload_size;

   NIR_PASS(_, shader, nir_lower_task_intrins, &state);

   /* Delete all code that potentially can't be reached due to
    * launch_mesh_workgroups being a terminating instruction.
    */
   NIR_PASS(_, shader, nir_lower_returns);

   bool progress;
   do {
      progress = false;
      NIR_PASS(progress, shader, nir_opt_dead_cf);
      NIR_PASS(progress, shader, nir_opt_dce);
   } while (progress);
   return true;
}

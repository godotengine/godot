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
#include "nir_control_flow.h"
#include "nir_worklist.h"

static bool
nir_op_is_derivative(nir_op op)
{
   return op == nir_op_fddx ||
          op == nir_op_fddy ||
          op == nir_op_fddx_fine ||
          op == nir_op_fddy_fine ||
          op == nir_op_fddx_coarse ||
          op == nir_op_fddy_coarse;
}

static bool
nir_texop_implies_derivative(nir_texop op)
{
   return op == nir_texop_tex ||
          op == nir_texop_txb ||
          op == nir_texop_lod;
}
#define MOVE_INSTR_FLAG            1
#define STOP_PROCESSING_INSTR_FLAG 2

/** Check recursively if the source can be moved to the top of the shader.
 *  Sets instr->pass_flags to MOVE_INSTR_FLAG and adds the instr
 *  to the given worklist
 */
static bool
can_move_src(nir_src *src, void *worklist)
{
   if (!src->is_ssa)
      return false;

   nir_instr *instr = src->ssa->parent_instr;
   if (instr->pass_flags)
      return true;

   /* Phi instructions can't be moved at all.  Also, if we're dependent on
    * a phi then we are dependent on some other bit of control flow and
    * it's hard to figure out the proper condition.
    */
   if (instr->type == nir_instr_type_phi)
      return false;

   if (instr->type == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
      if (intrin->intrinsic == nir_intrinsic_load_deref) {
         nir_deref_instr *deref = nir_src_as_deref(intrin->src[0]);
         if (!nir_deref_mode_is_one_of(deref, nir_var_read_only_modes))
            return false;
      } else if (!(nir_intrinsic_infos[intrin->intrinsic].flags &
                   NIR_INTRINSIC_CAN_REORDER)) {
         return false;
      }
   }

   /* set pass_flags and remember the instruction for potential cleanup */
   instr->pass_flags = MOVE_INSTR_FLAG;
   nir_instr_worklist_push_tail(worklist, instr);

   if (!nir_foreach_src(instr, can_move_src, worklist)) {
      return false;
   }
   return true;
}

/** Try to mark a discard or demote instruction for moving
 *
 * This function does two things.  One is that it searches through the
 * dependency chain to see if this discard is an instruction that we can move
 * up to the top.  Second, if the discard is one we can move, it tags the
 * discard and its dependencies (using pass_flags = 1).
 * Demote are handled the same way, except that they can still be moved up
 * when implicit derivatives are used.
 */
static bool
try_move_discard(nir_intrinsic_instr *discard)
{
   /* We require the discard to be in the top level of control flow.  We
    * could, in theory, move discards that are inside ifs or loops but that
    * would be a lot more work.
    */
   if (discard->instr.block->cf_node.parent->type != nir_cf_node_function)
      return false;

   /* Build the set of all instructions discard depends on to be able to
    * clear the flags in case the discard cannot be moved.
    */
   nir_instr_worklist *work = nir_instr_worklist_create();
   if (!work)
      return false;
   discard->instr.pass_flags = MOVE_INSTR_FLAG;

   bool can_move_discard = can_move_src(&discard->src[0], work);
   if (!can_move_discard) {
      /* Moving the discard is impossible: clear the flags */
      discard->instr.pass_flags = 0;
      nir_foreach_instr_in_worklist(instr, work)
         instr->pass_flags = 0;
   }

   nir_instr_worklist_destroy(work);

   return can_move_discard;
}

static bool
opt_move_discards_to_top_impl(nir_function_impl *impl)
{
   bool progress = false;
   bool consider_discards = true;
   bool moved = false;

   /* Walk through the instructions and look for a discard that we can move
    * to the top of the program.  If we hit any operation along the way that
    * we cannot safely move a discard above, break out of the loop and stop
    * trying to move any more discards.
    */
   nir_foreach_block(block, impl) {
      nir_foreach_instr_safe(instr, block) {
         instr->pass_flags = 0;

         switch (instr->type) {
         case nir_instr_type_alu: {
            nir_alu_instr *alu = nir_instr_as_alu(instr);
            if (nir_op_is_derivative(alu->op))
               consider_discards = false;
            continue;
         }

         case nir_instr_type_deref:
         case nir_instr_type_load_const:
         case nir_instr_type_ssa_undef:
         case nir_instr_type_phi:
            /* These are all safe */
            continue;

         case nir_instr_type_call:
            instr->pass_flags = STOP_PROCESSING_INSTR_FLAG;
            /* We don't know what the function will do */
            goto break_all;

         case nir_instr_type_tex: {
            nir_tex_instr *tex = nir_instr_as_tex(instr);
            if (nir_texop_implies_derivative(tex->op))
               consider_discards = false;
            continue;
         }

         case nir_instr_type_intrinsic: {
            nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
            if (nir_intrinsic_writes_external_memory(intrin)) {
               instr->pass_flags = STOP_PROCESSING_INSTR_FLAG;
               goto break_all;
            }

            if ((intrin->intrinsic == nir_intrinsic_discard_if && consider_discards) ||
                intrin->intrinsic == nir_intrinsic_demote_if)
               moved = moved || try_move_discard(intrin);
            continue;
         }

         case nir_instr_type_jump: {
            nir_jump_instr *jump = nir_instr_as_jump(instr);
            /* A return would cause the discard to not get executed */
            if (jump->type == nir_jump_return) {
               instr->pass_flags = STOP_PROCESSING_INSTR_FLAG;
               goto break_all;
            }
            continue;
         }

         case nir_instr_type_parallel_copy:
            unreachable("Unhanded instruction type");
         }
      }
   }
break_all:

   if (moved) {
      /* Walk the list of instructions and move the discard/demote and
       * everything it depends on to the top.  We walk the instruction list
       * here because it ensures that everything stays in its original order.
       * This provides stability for the algorithm and ensures that we don't
       * accidentally get dependencies out-of-order.
       */
      nir_cursor cursor = nir_before_block(nir_start_block(impl));
      nir_foreach_block(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->pass_flags == STOP_PROCESSING_INSTR_FLAG)
               return progress;
            if (instr->pass_flags == MOVE_INSTR_FLAG) {
               progress |= nir_instr_move(cursor, instr);
               cursor = nir_after_instr(instr);
            }
         }
      }
   }

   return progress;
}

/* This optimization only operates on discard_if/demoe_if so
 * nir_opt_conditional_discard and nir_lower_discard_or_demote
 * should have been called before.
 */
bool
nir_opt_move_discards_to_top(nir_shader *shader)
{
   assert(shader->info.stage == MESA_SHADER_FRAGMENT);

   bool progress = false;

   if (!shader->info.fs.uses_discard)
      return false;

   nir_foreach_function(function, shader) {
      if (function->impl && opt_move_discards_to_top_impl(function->impl)) {
         nir_metadata_preserve(function->impl, nir_metadata_block_index |
                                               nir_metadata_dominance);
         progress = true;
      }
   }

   return progress;
}

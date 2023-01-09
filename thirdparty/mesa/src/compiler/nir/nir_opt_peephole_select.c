/*
 * Copyright © 2014 Intel Corporation
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
 *    Jason Ekstrand (jason@jlekstrand.net)
 *
 */

#include "nir.h"
#include "nir/nir_builder.h"
#include "nir_control_flow.h"
#include "nir_search_helpers.h"

/*
 * Implements a small peephole optimization that looks for
 *
 * if (cond) {
 *    <then SSA defs>
 * } else {
 *    <else SSA defs>
 * }
 * phi
 * ...
 * phi
 *
 * and replaces it with:
 *
 * <then SSA defs>
 * <else SSA defs>
 * bcsel
 * ...
 * bcsel
 *
 * where the SSA defs are ALU operations or other cheap instructions (not
 * texturing, for example).
 *
 * If the number of ALU operations in the branches is greater than the limit
 * parameter, then the optimization is skipped.  In limit=0 mode, the SSA defs
 * must only be MOVs which we expect to get copy-propagated away once they're
 * out of the inner blocks.
 */

static bool
block_check_for_allowed_instrs(nir_block *block, unsigned *count,
                               unsigned limit, bool indirect_load_ok,
                               bool expensive_alu_ok)
{
   bool alu_ok = limit != 0;

   /* Used on non-control-flow HW to flatten all IFs. */
   if (limit == ~0) {
      nir_foreach_instr(instr, block) {
         switch (instr->type) {
         case nir_instr_type_alu:
         case nir_instr_type_deref:
         case nir_instr_type_load_const:
         case nir_instr_type_phi:
         case nir_instr_type_ssa_undef:
         case nir_instr_type_tex:
            break;

         case nir_instr_type_intrinsic:
            if (!nir_intrinsic_can_reorder(nir_instr_as_intrinsic(instr)))
               return false;
            break;

         case nir_instr_type_call:
         case nir_instr_type_jump:
         case nir_instr_type_parallel_copy:
            return false;
         }
      }
      return true;
   }

   nir_foreach_instr(instr, block) {
      switch (instr->type) {
      case nir_instr_type_intrinsic: {
         nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);

         switch (intrin->intrinsic) {
         case nir_intrinsic_load_deref: {
            nir_deref_instr *const deref = nir_src_as_deref(intrin->src[0]);

            switch (deref->modes) {
            case nir_var_shader_in:
            case nir_var_uniform:
            case nir_var_image:
               /* Don't try to remove flow control around an indirect load
                * because that flow control may be trying to avoid invalid
                * loads.
                */
               if (!indirect_load_ok && nir_deref_instr_has_indirect(deref))
                  return false;

               break;

            default:
               return false;
            }
            break;
         }

         case nir_intrinsic_load_uniform:
         case nir_intrinsic_load_preamble:
         case nir_intrinsic_load_helper_invocation:
         case nir_intrinsic_is_helper_invocation:
         case nir_intrinsic_load_front_face:
         case nir_intrinsic_load_view_index:
         case nir_intrinsic_load_layer_id:
         case nir_intrinsic_load_frag_coord:
         case nir_intrinsic_load_sample_pos:
         case nir_intrinsic_load_sample_pos_or_center:
         case nir_intrinsic_load_sample_id:
         case nir_intrinsic_load_sample_mask_in:
         case nir_intrinsic_load_vertex_id_zero_base:
         case nir_intrinsic_load_first_vertex:
         case nir_intrinsic_load_base_instance:
         case nir_intrinsic_load_instance_id:
         case nir_intrinsic_load_draw_id:
         case nir_intrinsic_load_num_workgroups:
         case nir_intrinsic_load_workgroup_id:
         case nir_intrinsic_load_local_invocation_id:
         case nir_intrinsic_load_local_invocation_index:
         case nir_intrinsic_load_subgroup_id:
         case nir_intrinsic_load_subgroup_invocation:
         case nir_intrinsic_load_num_subgroups:
         case nir_intrinsic_load_frag_shading_rate:
         case nir_intrinsic_is_sparse_texels_resident:
         case nir_intrinsic_sparse_residency_code_and:
            if (!alu_ok)
               return false;
            break;

         default:
            return false;
         }

         break;
      }

      case nir_instr_type_deref:
      case nir_instr_type_load_const:
      case nir_instr_type_ssa_undef:
         break;

      case nir_instr_type_alu: {
         nir_alu_instr *mov = nir_instr_as_alu(instr);
         bool movelike = false;

         switch (mov->op) {
         case nir_op_mov:
         case nir_op_fneg:
         case nir_op_ineg:
         case nir_op_fabs:
         case nir_op_iabs:
         case nir_op_vec2:
         case nir_op_vec3:
         case nir_op_vec4:
         case nir_op_vec5:
         case nir_op_vec8:
         case nir_op_vec16:
            movelike = true;
            break;

         case nir_op_fcos:
         case nir_op_fdiv:
         case nir_op_fexp2:
         case nir_op_flog2:
         case nir_op_fmod:
         case nir_op_fpow:
         case nir_op_frcp:
         case nir_op_frem:
         case nir_op_frsq:
         case nir_op_fsin:
         case nir_op_idiv:
         case nir_op_irem:
         case nir_op_udiv:
            if (!alu_ok || !expensive_alu_ok)
               return false;

            break;

         default:
            if (!alu_ok) {
               /* It must be a move-like operation. */
               return false;
            }
            break;
         }

         /* It must be SSA */
         if (!mov->dest.dest.is_ssa)
            return false;

         if (alu_ok) {
            /* If the ALU operation is an fsat or a move-like operation, do
             * not count it.  The expectation is that it will eventually be
             * merged as a destination modifier or source modifier on some
             * other instruction.
             */
            if (mov->op != nir_op_fsat && !movelike)
               (*count)++;
         } else {
            /* Can't handle saturate */
            if (mov->dest.saturate)
               return false;

            /* It cannot have any if-uses */
            if (!list_is_empty(&mov->dest.dest.ssa.if_uses))
               return false;

            /* The only uses of this definition must be phis in the successor */
            nir_foreach_use(use, &mov->dest.dest.ssa) {
               if (use->parent_instr->type != nir_instr_type_phi ||
                   use->parent_instr->block != block->successors[0])
                  return false;
            }
         }
         break;
      }

      default:
         return false;
      }
   }

   return true;
}

/**
 * Try to collapse nested ifs:
 * This optimization turns
 *
 * if (cond1) {
 *   <allowed instruction>
 *   if (cond2) {
 *     <any code>
 *   } else {
 *   }
 * } else {
 * }
 *
 * into
 *
 * <allowed instruction>
 * if (cond1 && cond2) {
 *   <any code>
 * } else {
 * }
 *
 */
static bool
nir_opt_collapse_if(nir_if *if_stmt, nir_shader *shader, unsigned limit,
                    bool indirect_load_ok, bool expensive_alu_ok)
{
   /* the if has to be nested */
   if (if_stmt->cf_node.parent->type != nir_cf_node_if)
      return false;

   nir_if *parent_if = nir_cf_node_as_if(if_stmt->cf_node.parent);
   if (parent_if->control == nir_selection_control_dont_flatten)
      return false;

   /* check if the else block is empty */
   if (!nir_cf_list_is_empty_block(&if_stmt->else_list))
      return false;

   /* this opt doesn't make much sense if the branch is empty */
   if (nir_cf_list_is_empty_block(&if_stmt->then_list))
      return false;

   /* the nested if has to be the only cf_node:
    * i.e. <block> <if_stmt> <block> */
   if (exec_list_length(&parent_if->then_list) != 3)
      return false;

   /* check if the else block of the parent if is empty */
   if (!nir_cf_list_is_empty_block(&parent_if->else_list))
      return false;

   /* check if the block after the nested if is empty except for phis */
   nir_block *last = nir_if_last_then_block(parent_if);
   nir_instr *last_instr = nir_block_last_instr(last);
   if (last_instr && last_instr->type != nir_instr_type_phi)
      return false;

   /* check if all outer phis become trivial after merging the ifs */
   nir_foreach_instr(instr, last) {
      if (parent_if->control == nir_selection_control_flatten)
         break;

      nir_phi_instr *phi = nir_instr_as_phi(instr);
      nir_phi_src *else_src =
         nir_phi_get_src_from_block(phi, nir_if_first_else_block(if_stmt));

      nir_foreach_use (src, &phi->dest.ssa) {
         assert(src->parent_instr->type == nir_instr_type_phi);
         nir_phi_src *phi_src =
            nir_phi_get_src_from_block(nir_instr_as_phi(src->parent_instr),
                                       nir_if_first_else_block(parent_if));
         if (phi_src->src.ssa != else_src->src.ssa)
            return false;
      }
   }

   if (parent_if->control == nir_selection_control_flatten) {
      /* Override driver defaults */
      indirect_load_ok = true;
      expensive_alu_ok = true;
   }

   /* check if the block before the nested if matches the requirements */
   nir_block *first = nir_if_first_then_block(parent_if);
   unsigned count = 0;
   if (!block_check_for_allowed_instrs(first, &count, limit != 0,
                                       indirect_load_ok, expensive_alu_ok))
      return false;

   if (count > limit && parent_if->control != nir_selection_control_flatten)
      return false;

   /* trivialize succeeding phis */
   nir_foreach_instr(instr, last) {
      nir_phi_instr *phi = nir_instr_as_phi(instr);
      nir_phi_src *else_src =
         nir_phi_get_src_from_block(phi, nir_if_first_else_block(if_stmt));
      nir_foreach_use_safe(src, &phi->dest.ssa) {
         nir_phi_src *phi_src =
            nir_phi_get_src_from_block(nir_instr_as_phi(src->parent_instr),
                                       nir_if_first_else_block(parent_if));
         if (phi_src->src.ssa == else_src->src.ssa)
            nir_instr_rewrite_src(src->parent_instr, &phi_src->src,
                                  nir_src_for_ssa(&phi->dest.ssa));
      }
   }

   /* combine the conditions */
   struct nir_builder b;
   nir_builder_init(&b, nir_cf_node_get_function(&if_stmt->cf_node)->function->impl);
   b.cursor = nir_before_cf_node(&if_stmt->cf_node);
   nir_ssa_def *cond = nir_iand(&b, if_stmt->condition.ssa,
                                parent_if->condition.ssa);
   nir_if_rewrite_condition(if_stmt, nir_src_for_ssa(cond));

   /* move the whole inner if before the parent if */
   nir_cf_list tmp;
   nir_cf_extract(&tmp, nir_before_block(first),
                        nir_after_block(last));
   nir_cf_reinsert(&tmp, nir_before_cf_node(&parent_if->cf_node));

   /* The now empty parent if will be cleaned up by other passes */
   return true;
}

static bool
nir_opt_peephole_select_block(nir_block *block, nir_shader *shader,
                              unsigned limit, bool indirect_load_ok,
                              bool expensive_alu_ok)
{
   if (nir_cf_node_is_first(&block->cf_node))
      return false;

   nir_cf_node *prev_node = nir_cf_node_prev(&block->cf_node);
   if (prev_node->type != nir_cf_node_if)
      return false;

   nir_block *prev_block = nir_cf_node_as_block(nir_cf_node_prev(prev_node));

   /* If the last instruction before this if/else block is a jump, we can't
    * append stuff after it because it would break a bunch of assumption about
    * control flow (nir_validate expects the successor of a return/halt jump
    * to be the end of the function, which might not match the successor of
    * the if/else blocks).
    */
   if (nir_block_ends_in_return_or_halt(prev_block))
      return false;

   nir_if *if_stmt = nir_cf_node_as_if(prev_node);

   /* first, try to collapse the if */
   if (nir_opt_collapse_if(if_stmt, shader, limit,
                           indirect_load_ok, expensive_alu_ok))
      return true;

   if (if_stmt->control == nir_selection_control_dont_flatten)
      return false;

   nir_block *then_block = nir_if_first_then_block(if_stmt);
   nir_block *else_block = nir_if_first_else_block(if_stmt);

   /* We can only have one block in each side ... */
   if (nir_if_last_then_block(if_stmt) != then_block ||
       nir_if_last_else_block(if_stmt) != else_block)
      return false;

   if (if_stmt->control == nir_selection_control_flatten) {
      /* Override driver defaults */
      indirect_load_ok = true;
      expensive_alu_ok = true;
   }

   /* ... and those blocks must only contain "allowed" instructions. */
   unsigned count = 0;
   if (!block_check_for_allowed_instrs(then_block, &count, limit,
                                       indirect_load_ok, expensive_alu_ok) ||
       !block_check_for_allowed_instrs(else_block, &count, limit,
                                       indirect_load_ok, expensive_alu_ok))
      return false;

   if (count > limit && if_stmt->control != nir_selection_control_flatten)
      return false;

   /* At this point, we know that the previous CFG node is an if-then
    * statement containing only moves to phi nodes in this block.  We can
    * just remove that entire CF node and replace all of the phi nodes with
    * selects.
    */

   /* First, we move the remaining instructions from the blocks to the
    * block before.  We have already guaranteed that this is safe by
    * calling block_check_for_allowed_instrs()
    */
   nir_foreach_instr_safe(instr, then_block) {
      exec_node_remove(&instr->node);
      instr->block = prev_block;
      exec_list_push_tail(&prev_block->instr_list, &instr->node);
   }

   nir_foreach_instr_safe(instr, else_block) {
      exec_node_remove(&instr->node);
      instr->block = prev_block;
      exec_list_push_tail(&prev_block->instr_list, &instr->node);
   }

   nir_foreach_instr_safe(instr, block) {
      if (instr->type != nir_instr_type_phi)
         break;

      nir_phi_instr *phi = nir_instr_as_phi(instr);
      nir_alu_instr *sel = nir_alu_instr_create(shader, nir_op_bcsel);
      nir_src_copy(&sel->src[0].src, &if_stmt->condition, &sel->instr);
      /* Splat the condition to all channels */
      memset(sel->src[0].swizzle, 0, sizeof sel->src[0].swizzle);

      assert(exec_list_length(&phi->srcs) == 2);
      nir_foreach_phi_src(src, phi) {
         assert(src->pred == then_block || src->pred == else_block);
         assert(src->src.is_ssa);

         unsigned idx = src->pred == then_block ? 1 : 2;
         nir_src_copy(&sel->src[idx].src, &src->src, &sel->instr);
      }

      nir_ssa_dest_init(&sel->instr, &sel->dest.dest,
                        phi->dest.ssa.num_components,
                        phi->dest.ssa.bit_size, NULL);
      sel->dest.write_mask = (1 << phi->dest.ssa.num_components) - 1;

      nir_ssa_def_rewrite_uses(&phi->dest.ssa,
                               &sel->dest.dest.ssa);

      nir_instr_insert_before(&phi->instr, &sel->instr);
      nir_instr_remove(&phi->instr);
   }

   nir_cf_node_remove(&if_stmt->cf_node);
   return true;
}

static bool
nir_opt_peephole_select_impl(nir_function_impl *impl, unsigned limit,
                             bool indirect_load_ok, bool expensive_alu_ok)
{
   nir_shader *shader = impl->function->shader;
   bool progress = false;

   nir_foreach_block_safe(block, impl) {
      progress |= nir_opt_peephole_select_block(block, shader, limit,
                                                indirect_load_ok,
                                                expensive_alu_ok);
   }

   if (progress) {
      nir_metadata_preserve(impl, nir_metadata_none);
   } else {
      nir_metadata_preserve(impl, nir_metadata_all);
   }

   return progress;
}

bool
nir_opt_peephole_select(nir_shader *shader, unsigned limit,
                        bool indirect_load_ok, bool expensive_alu_ok)
{
   bool progress = false;

   nir_foreach_function(function, shader) {
      if (function->impl)
         progress |= nir_opt_peephole_select_impl(function->impl, limit,
                                                  indirect_load_ok,
                                                  expensive_alu_ok);
   }

   return progress;
}

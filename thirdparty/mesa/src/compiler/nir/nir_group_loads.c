/*
 * Copyright © 2021 Advanced Micro Devices, Inc.
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

/* This is a new block-level load instruction scheduler where loads are grouped
 * according to their indirection level within a basic block. An indirection
 * is when a result of one load is used as a source of another load. The result
 * is that disjoint ALU opcode groups and load (texture) opcode groups are
 * created where each next load group is the next level of indirection.
 * It's done by finding the first and last load with the same indirection
 * level, and moving all unrelated instructions between them after the last
 * load except for load sources, which are moved before the first load.
 * It naturally suits hardware that has limits on texture indirections, but
 * other hardware can benefit too. Only texture, image, and SSBO load and
 * atomic instructions are grouped.
 *
 * There is an option to group only those loads that use the same resource
 * variable. This increases the chance to get more cache hits than if the loads
 * were spread out.
 *
 * The increased register usage is offset by the increase in observed memory
 * bandwidth due to more cache hits (dependent on hw behavior) and thus
 * decrease the subgroup lifetime, which allows registers to be deallocated
 * and reused sooner. In some bandwidth-bound cases, low register usage doesn't
 * benefit at all. Doubling the register usage and using those registers to
 * amplify observed bandwidth can improve performance a lot.
 *
 * It's recommended to run a hw-specific instruction scheduler after this to
 * prevent spilling.
 */

#include "nir.h"

static bool
is_memory_load(nir_instr *instr)
{
   /* Count texture_size too because it has the same latency as cache hits. */
   if (instr->type == nir_instr_type_tex)
      return true;

   if (instr->type == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      const char *name = nir_intrinsic_infos[intr->intrinsic].name;

      /* TODO: nir_intrinsics.py could do this */
      /* load_ubo is ignored because it's usually cheap. */
      if (!nir_intrinsic_writes_external_memory(intr) &&
          !strstr(name, "shared") &&
          (strstr(name, "ssbo") || strstr(name, "image")))
         return true;
   }

   return false;
}

static nir_instr *
get_intrinsic_resource(nir_intrinsic_instr *intr)
{
   /* This is also the list of intrinsics that are grouped. */
   /* load_ubo is ignored because it's usually cheap. */
   switch (intr->intrinsic) {
   case nir_intrinsic_image_load:
   case nir_intrinsic_image_deref_load:
   case nir_intrinsic_image_sparse_load:
   case nir_intrinsic_image_deref_sparse_load:
   /* Group image_size too because it has the same latency as cache hits. */
   case nir_intrinsic_image_samples_identical:
   case nir_intrinsic_image_deref_samples_identical:
   case nir_intrinsic_bindless_image_samples_identical:
   case nir_intrinsic_image_size:
   case nir_intrinsic_image_deref_size:
   case nir_intrinsic_bindless_image_load:
   case nir_intrinsic_bindless_image_sparse_load:
   case nir_intrinsic_load_ssbo:
   case nir_intrinsic_image_fragment_mask_load_amd:
   case nir_intrinsic_image_deref_fragment_mask_load_amd:
   case nir_intrinsic_bindless_image_fragment_mask_load_amd:
      return intr->src[0].ssa->parent_instr;
   default:
      return NULL;
   }
}

/* Track only those that we want to group. */
static bool
is_grouped_load(nir_instr *instr)
{
   /* Count texture_size too because it has the same latency as cache hits. */
   if (instr->type == nir_instr_type_tex)
      return true;

   if (instr->type == nir_instr_type_intrinsic)
      return get_intrinsic_resource(nir_instr_as_intrinsic(instr)) != NULL;

   return false;
}

static bool
can_move(nir_instr *instr, uint8_t current_indirection_level)
{
   /* Grouping is done by moving everything else out of the first/last
    * instruction range of the indirection level.
    */
   if (is_grouped_load(instr) && instr->pass_flags == current_indirection_level)
      return false;

   if (instr->type == nir_instr_type_alu ||
       instr->type == nir_instr_type_deref ||
       instr->type == nir_instr_type_tex ||
       instr->type == nir_instr_type_load_const ||
       instr->type == nir_instr_type_ssa_undef)
      return true;

   if (instr->type == nir_instr_type_intrinsic &&
       nir_intrinsic_can_reorder(nir_instr_as_intrinsic(instr)))
      return true;

   return false;
}

static nir_instr *
get_uniform_inst_resource(nir_instr *instr)
{
   if (instr->type == nir_instr_type_tex) {
      nir_tex_instr *tex = nir_instr_as_tex(instr);

      if (tex->texture_non_uniform)
         return NULL;

      for (unsigned i = 0; i < tex->num_srcs; i++) {
         switch (tex->src[i].src_type) {
         case nir_tex_src_texture_deref:
         case nir_tex_src_texture_handle:
            return tex->src[i].src.ssa->parent_instr;
         default:
            break;
         }
      }
      return NULL;
   }

   if (instr->type == nir_instr_type_intrinsic)
      return get_intrinsic_resource(nir_instr_as_intrinsic(instr));

   return NULL;
}

struct check_sources_state
{
   nir_block *block;
   uint32_t first_index;
};

static bool
has_only_sources_less_than(nir_src *src, void *data)
{
   struct check_sources_state *state = (struct check_sources_state *)data;

   /* true if nir_foreach_src should keep going */
   return state->block != src->ssa->parent_instr->block ||
          src->ssa->parent_instr->index < state->first_index;
}

static void
group_loads(nir_instr *first, nir_instr *last)
{
   /* Walk the instruction range between the first and last backward, and
    * move those that have no uses within the range after the last one.
    */
   for (nir_instr *instr = exec_node_data_backward(nir_instr,
                                                   last->node.prev, node);
        instr != first;
        instr = exec_node_data_backward(nir_instr, instr->node.prev, node)) {
      /* Only move instructions without side effects. */
      if (!can_move(instr, first->pass_flags))
         continue;

      nir_ssa_def *def = nir_instr_ssa_def(instr);
      if (def) {
         bool all_uses_after_last = true;

         nir_foreach_use(use, def) {
            if (use->parent_instr->block == instr->block &&
                use->parent_instr->index <= last->index) {
               all_uses_after_last = false;
               break;
            }
         }

         if (all_uses_after_last) {
            nir_instr *move_instr = instr;
            /* Set the last instruction because we'll delete the current one. */
            instr = exec_node_data_forward(nir_instr, instr->node.next, node);

            /* Move the instruction after the last and update its index
             * to indicate that it's after it.
             */
            nir_instr_move(nir_after_instr(last), move_instr);
            move_instr->index = last->index + 1;
         }
      }
   }

   struct check_sources_state state;
   state.block = first->block;
   state.first_index = first->index;

   /* Walk the instruction range between the first and last forward, and move
    * those that have no sources within the range before the first one.
    */
   for (nir_instr *instr = exec_node_data_forward(nir_instr,
                                                  first->node.next, node);
        instr != last;
        instr = exec_node_data_forward(nir_instr, instr->node.next, node)) {
      /* Only move instructions without side effects. */
      if (!can_move(instr, first->pass_flags))
         continue;

      if (nir_foreach_src(instr, has_only_sources_less_than, &state)) {
         nir_instr *move_instr = instr;
         /* Set the last instruction because we'll delete the current one. */
         instr = exec_node_data_backward(nir_instr, instr->node.prev, node);

         /* Move the instruction before the first and update its index
          * to indicate that it's before it.
          */
         nir_instr_move(nir_before_instr(first), move_instr);
         move_instr->index = first->index - 1;
      }
   }
}

static bool
is_pseudo_inst(nir_instr *instr)
{
   /* Other instructions do not usually contribute to the shader binary size. */
   return instr->type != nir_instr_type_alu &&
          instr->type != nir_instr_type_call &&
          instr->type != nir_instr_type_tex &&
          instr->type != nir_instr_type_intrinsic;
}

static void
set_instr_indices(nir_block *block)
{
   /* Start with 1 because we'll move instruction before the first one
    * and will want to label it 0.
    */
   unsigned counter = 1;
   nir_instr *last = NULL;

   nir_foreach_instr(instr, block) {
      /* Make sure grouped instructions don't have the same index as pseudo
       * instructions.
       */
      if (last && is_pseudo_inst(last) && is_grouped_load(instr))
          counter++;

      /* Set each instruction's index within the block. */
      instr->index = counter;

      /* Only count non-pseudo instructions. */
      if (!is_pseudo_inst(instr))
         counter++;

      last = instr;
   }
}

static void
handle_load_range(nir_instr **first, nir_instr **last,
                  nir_instr *current, unsigned max_distance)
{
   if (*first && *last &&
       (!current || current->index > (*first)->index + max_distance)) {
      assert(*first != *last);
      group_loads(*first, *last);
      set_instr_indices((*first)->block);
      *first = NULL;
      *last = NULL;
   }
}

static bool
is_barrier(nir_instr *instr)
{
   if (instr->type == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
      const char *name = nir_intrinsic_infos[intr->intrinsic].name;


      if (intr->intrinsic == nir_intrinsic_discard ||
          intr->intrinsic == nir_intrinsic_discard_if ||
          intr->intrinsic == nir_intrinsic_terminate ||
          intr->intrinsic == nir_intrinsic_terminate_if ||
          /* TODO: nir_intrinsics.py could do this */
          strstr(name, "barrier"))
         return true;
   }

   return false;
}

struct indirection_state
{
   nir_block *block;
   unsigned indirections;
};

static unsigned
get_num_indirections(nir_instr *instr);

static bool
gather_indirections(nir_src *src, void *data)
{
   struct indirection_state *state = (struct indirection_state *)data;
   nir_instr *instr = src->ssa->parent_instr;

   /* We only count indirections within the same block. */
   if (instr->block == state->block) {
      unsigned indirections = get_num_indirections(src->ssa->parent_instr);

      if (instr->type == nir_instr_type_tex || is_memory_load(instr))
         indirections++;

      state->indirections = MAX2(state->indirections, indirections);
   }

   return true; /* whether nir_foreach_src should keep going */
}

/* Return the number of load indirections within the block. */
static unsigned
get_num_indirections(nir_instr *instr)
{
   /* Don't traverse phis because we could end up in an infinite recursion
    * if the phi points to the current block (such as a loop body).
    */
   if (instr->type == nir_instr_type_phi)
      return 0;

   if (instr->index != UINT32_MAX)
      return instr->index; /* we've visited this instruction before */

   struct indirection_state state;
   state.block = instr->block;
   state.indirections = 0;

   nir_foreach_src(instr, gather_indirections, &state);

   instr->index = state.indirections;
   return state.indirections;
}

static void
process_block(nir_block *block, nir_load_grouping grouping,
              unsigned max_distance)
{
   int max_indirection = -1;
   unsigned num_inst_per_level[256] = {0};

   /* UINT32_MAX means the instruction has not been visited. Once
    * an instruction has been visited and its indirection level has been
    * determined, we'll store the indirection level in the index. The next
    * instruction that visits it will use the index instead of recomputing
    * the indirection level, which would result in an exponetial time
    * complexity.
    */
   nir_foreach_instr(instr, block) {
      instr->index = UINT32_MAX; /* unknown */
   }

   /* Count the number of load indirections for each load instruction
    * within this block. Store it in pass_flags.
    */
   nir_foreach_instr(instr, block) {
      if (is_grouped_load(instr)) {
         unsigned indirections = get_num_indirections(instr);

         /* pass_flags has only 8 bits */
         indirections = MIN2(indirections, 255);
         num_inst_per_level[indirections]++;
         instr->pass_flags = indirections;

         max_indirection = MAX2(max_indirection, (int)indirections);
      }
   }

   /* 255 contains all indirection levels >= 255, so ignore them. */
   max_indirection = MIN2(max_indirection, 254);

   /* Each indirection level is grouped. */
   for (int level = 0; level <= max_indirection; level++) {
      if (num_inst_per_level[level] <= 1)
         continue;

      set_instr_indices(block);

      nir_instr *resource = NULL;
      nir_instr *first_load = NULL, *last_load = NULL;

      /* Find the first and last instruction that use the same
       * resource and are within a certain distance of each other.
       * If found, group them by moving all movable instructions
       * between them out.
       */
      nir_foreach_instr(current, block) {
         /* Don't group across barriers. */
         if (is_barrier(current)) {
            /* Group unconditionally.  */
            handle_load_range(&first_load, &last_load, NULL, 0);
            first_load = NULL;
            last_load = NULL;
            continue;
         }

         /* Only group load instructions with the same indirection level. */
         if (is_grouped_load(current) && current->pass_flags == level) {
            nir_instr *current_resource;

            switch (grouping) {
            case nir_group_all:
               if (!first_load)
                  first_load = current;
               else
                  last_load = current;
               break;

            case nir_group_same_resource_only:
               current_resource = get_uniform_inst_resource(current);

               if (current_resource) {
                  if (!first_load) {
                     first_load = current;
                     resource = current_resource;
                  } else if (current_resource == resource) {
                     last_load = current;
                  }
               }
            }
         }

         /* Group only if we exceeded the maximum distance. */
         handle_load_range(&first_load, &last_load, current, max_distance);
      }

      /* Group unconditionally.  */
      handle_load_range(&first_load, &last_load, NULL, 0);
   }
}

/* max_distance is the maximum distance between the first and last instruction
 * in a group.
 */
void
nir_group_loads(nir_shader *shader, nir_load_grouping grouping,
                unsigned max_distance)
{
   nir_foreach_function(function, shader) {
      if (function->impl) {
         nir_foreach_block(block, function->impl) {
            process_block(block, grouping, max_distance);
         }

         nir_metadata_preserve(function->impl,
                               nir_metadata_block_index |
                               nir_metadata_dominance |
                               nir_metadata_loop_analysis);
      }
   }
}

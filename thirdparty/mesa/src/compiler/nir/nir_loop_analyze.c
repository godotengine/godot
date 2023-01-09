/*
 * Copyright © 2015 Thomas Helland
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
#include "nir_constant_expressions.h"
#include "nir_loop_analyze.h"
#include "util/bitset.h"

typedef enum {
   undefined,
   invariant,
   not_invariant,
   basic_induction
} nir_loop_variable_type;

typedef struct nir_basic_induction_var {
   nir_alu_instr *alu;                      /* The def of the alu-operation */
   nir_ssa_def *def_outside_loop;           /* The phi-src outside the loop */
} nir_basic_induction_var;

typedef struct {
   /* A link for the work list */
   struct list_head process_link;

   bool in_loop;

   /* The ssa_def associated with this info */
   nir_ssa_def *def;

   /* The type of this ssa_def */
   nir_loop_variable_type type;

   /* If this is of type basic_induction */
   struct nir_basic_induction_var *ind;

   /* True if variable is in an if branch */
   bool in_if_branch;

   /* True if variable is in a nested loop */
   bool in_nested_loop;

   /* Could be a basic_induction if following uniforms are inlined */
   nir_src *init_src;
   nir_alu_src *update_src;
} nir_loop_variable;

typedef struct {
   /* The loop we store information for */
   nir_loop *loop;

   /* Loop_variable for all ssa_defs in function */
   nir_loop_variable *loop_vars;
   BITSET_WORD *loop_vars_init;

   /* A list of the loop_vars to analyze */
   struct list_head process_list;

   nir_variable_mode indirect_mask;

   bool force_unroll_sampler_indirect;
} loop_info_state;

static nir_loop_variable *
get_loop_var(nir_ssa_def *value, loop_info_state *state)
{
   nir_loop_variable *var = &(state->loop_vars[value->index]);

   if (!BITSET_TEST(state->loop_vars_init, value->index)) {
      var->in_loop = false;
      var->def = value;
      var->in_if_branch = false;
      var->in_nested_loop = false;
      var->init_src = NULL;
      var->update_src = NULL;
      if (value->parent_instr->type == nir_instr_type_load_const)
         var->type = invariant;
      else
         var->type = undefined;

      BITSET_SET(state->loop_vars_init, value->index);
   }

   return var;
}

typedef struct {
   loop_info_state *state;
   bool in_if_branch;
   bool in_nested_loop;
} init_loop_state;

static bool
init_loop_def(nir_ssa_def *def, void *void_init_loop_state)
{
   init_loop_state *loop_init_state = void_init_loop_state;
   nir_loop_variable *var = get_loop_var(def, loop_init_state->state);

   if (loop_init_state->in_nested_loop) {
      var->in_nested_loop = true;
   } else if (loop_init_state->in_if_branch) {
      var->in_if_branch = true;
   } else {
      /* Add to the tail of the list. That way we start at the beginning of
       * the defs in the loop instead of the end when walking the list. This
       * means less recursive calls. Only add defs that are not in nested
       * loops or conditional blocks.
       */
      list_addtail(&var->process_link, &loop_init_state->state->process_list);
   }

   var->in_loop = true;

   return true;
}

/** Calculate an estimated cost in number of instructions
 *
 * We do this so that we don't unroll loops which will later get massively
 * inflated due to int64 or fp64 lowering.  The estimates provided here don't
 * have to be massively accurate; they just have to be good enough that loop
 * unrolling doesn't cause things to blow up too much.
 */
static unsigned
instr_cost(loop_info_state *state, nir_instr *instr,
           const nir_shader_compiler_options *options)
{
   if (instr->type == nir_instr_type_intrinsic ||
       instr->type == nir_instr_type_tex)
      return 1;

   if (instr->type != nir_instr_type_alu)
      return 0;

   nir_alu_instr *alu = nir_instr_as_alu(instr);
   const nir_op_info *info = &nir_op_infos[alu->op];
   unsigned cost = 1;

   if (nir_op_is_selection(alu->op)) {
      nir_ssa_scalar cond_scalar = {alu->src[0].src.ssa, 0};
      if (nir_is_terminator_condition_with_two_inputs(cond_scalar)) {
         nir_instr *sel_cond = alu->src[0].src.ssa->parent_instr;
         nir_alu_instr *sel_alu = nir_instr_as_alu(sel_cond);

         nir_ssa_scalar rhs, lhs;
         lhs = nir_ssa_scalar_chase_alu_src(cond_scalar, 0);
         rhs = nir_ssa_scalar_chase_alu_src(cond_scalar, 1);

         /* If the selects condition is a comparision between a constant and
          * a basic induction variable we know that it will be eliminated once
          * the loop is unrolled so here we assign it a cost of 0.
          */
         if ((nir_src_is_const(sel_alu->src[0].src) &&
              get_loop_var(rhs.def, state)->type == basic_induction) ||
             (nir_src_is_const(sel_alu->src[1].src) &&
              get_loop_var(lhs.def, state)->type == basic_induction) ) {
            /* Also if the selects condition is only used by the select then
             * remove that alu instructons cost from the cost total also.
             */
            if (!list_is_empty(&sel_alu->dest.dest.ssa.if_uses) ||
                !list_is_singular(&sel_alu->dest.dest.ssa.uses))
               return 0;
            else
               return -1;
         }
      }
   }

   if (alu->op == nir_op_flrp) {
      if ((options->lower_flrp16 && nir_dest_bit_size(alu->dest.dest) == 16) ||
          (options->lower_flrp32 && nir_dest_bit_size(alu->dest.dest) == 32) ||
          (options->lower_flrp64 && nir_dest_bit_size(alu->dest.dest) == 64))
         cost *= 3;
   }

   /* Assume everything 16 or 32-bit is cheap.
    *
    * There are no 64-bit ops that don't have a 64-bit thing as their
    * destination or first source.
    */
   if (nir_dest_bit_size(alu->dest.dest) < 64 &&
       nir_src_bit_size(alu->src[0].src) < 64)
      return cost;

   bool is_fp64 = nir_dest_bit_size(alu->dest.dest) == 64 &&
      nir_alu_type_get_base_type(info->output_type) == nir_type_float;
   for (unsigned i = 0; i < info->num_inputs; i++) {
      if (nir_src_bit_size(alu->src[i].src) == 64 &&
          nir_alu_type_get_base_type(info->input_types[i]) == nir_type_float)
         is_fp64 = true;
   }

   if (is_fp64) {
      /* If it's something lowered normally, it's expensive. */
      if (options->lower_doubles_options &
          nir_lower_doubles_op_to_options_mask(alu->op))
         cost *= 20;

      /* If it's full software, it's even more expensive */
      if (options->lower_doubles_options & nir_lower_fp64_full_software) {
         cost *= 100;
         state->loop->info->has_soft_fp64 = true;
      }

      return cost;
   } else {
      if (options->lower_int64_options &
          nir_lower_int64_op_to_options_mask(alu->op)) {
         /* These require a doing the division algorithm. */
         if (alu->op == nir_op_idiv || alu->op == nir_op_udiv ||
             alu->op == nir_op_imod || alu->op == nir_op_umod ||
             alu->op == nir_op_irem)
            return cost * 100;

         /* Other int64 lowering isn't usually all that expensive */
         return cost * 5;
      }

      return cost;
   }
}

static bool
init_loop_block(nir_block *block, loop_info_state *state,
                bool in_if_branch, bool in_nested_loop)
{
   init_loop_state init_state = {.in_if_branch = in_if_branch,
                                 .in_nested_loop = in_nested_loop,
                                 .state = state };

   nir_foreach_instr(instr, block) {
      nir_foreach_ssa_def(instr, init_loop_def, &init_state);
   }

   return true;
}

static inline bool
is_var_alu(nir_loop_variable *var)
{
   return var->def->parent_instr->type == nir_instr_type_alu;
}

static inline bool
is_var_phi(nir_loop_variable *var)
{
   return var->def->parent_instr->type == nir_instr_type_phi;
}

static inline bool
mark_invariant(nir_ssa_def *def, loop_info_state *state)
{
   nir_loop_variable *var = get_loop_var(def, state);

   if (var->type == invariant)
      return true;

   if (!var->in_loop) {
      var->type = invariant;
      return true;
   }

   if (var->type == not_invariant)
      return false;

   if (is_var_alu(var)) {
      nir_alu_instr *alu = nir_instr_as_alu(def->parent_instr);

      for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; i++) {
         if (!mark_invariant(alu->src[i].src.ssa, state)) {
            var->type = not_invariant;
            return false;
         }
      }
      var->type = invariant;
      return true;
   }

   /* Phis shouldn't be invariant except if one operand is invariant, and the
    * other is the phi itself. These should be removed by opt_remove_phis.
    * load_consts are already set to invariant and constant during init,
    * and so should return earlier. Remaining op_codes are set undefined.
    */
   var->type = not_invariant;
   return false;
}

static void
compute_invariance_information(loop_info_state *state)
{
   /* An expression is invariant in a loop L if:
    *  (base cases)
    *    – it’s a constant
    *    – it’s a variable use, all of whose single defs are outside of L
    *  (inductive cases)
    *    – it’s a pure computation all of whose args are loop invariant
    *    – it’s a variable use whose single reaching def, and the
    *      rhs of that def is loop-invariant
    */
   list_for_each_entry_safe(nir_loop_variable, var, &state->process_list,
                            process_link) {
      assert(!var->in_if_branch && !var->in_nested_loop);

      if (mark_invariant(var->def, state))
         list_del(&var->process_link);
   }
}

/* If all of the instruction sources point to identical ALU instructions (as
 * per nir_instrs_equal), return one of the ALU instructions.  Otherwise,
 * return NULL.
 */
static nir_alu_instr *
phi_instr_as_alu(nir_phi_instr *phi)
{
   nir_alu_instr *first = NULL;
   nir_foreach_phi_src(src, phi) {
      assert(src->src.is_ssa);
      if (src->src.ssa->parent_instr->type != nir_instr_type_alu)
         return NULL;

      nir_alu_instr *alu = nir_instr_as_alu(src->src.ssa->parent_instr);
      if (first == NULL) {
         first = alu;
      } else {
         if (!nir_instrs_equal(&first->instr, &alu->instr))
            return NULL;
      }
   }

   return first;
}

static bool
alu_src_has_identity_swizzle(nir_alu_instr *alu, unsigned src_idx)
{
   assert(nir_op_infos[alu->op].input_sizes[src_idx] == 0);
   assert(alu->dest.dest.is_ssa);
   for (unsigned i = 0; i < alu->dest.dest.ssa.num_components; i++) {
      if (alu->src[src_idx].swizzle[i] != i)
         return false;
   }

   return true;
}

static bool
is_only_uniform_src(nir_src *src)
{
   if (!src->is_ssa)
      return false;

   nir_instr *instr = src->ssa->parent_instr;

   switch (instr->type) {
   case nir_instr_type_alu: {
      /* Return true if all sources return true. */
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; i++) {
         if (!is_only_uniform_src(&alu->src[i].src))
             return false;
      }
      return true;
   }

   case nir_instr_type_intrinsic: {
      nir_intrinsic_instr *inst = nir_instr_as_intrinsic(instr);
      /* current uniform inline only support load ubo */
      return inst->intrinsic == nir_intrinsic_load_ubo;
   }

   case nir_instr_type_load_const:
      /* Always return true for constants. */
      return true;

   default:
      return false;
   }
}

static bool
compute_induction_information(loop_info_state *state)
{
   bool found_induction_var = false;
   unsigned num_induction_vars = 0;

   list_for_each_entry_safe(nir_loop_variable, var, &state->process_list,
                            process_link) {

      /* It can't be an induction variable if it is invariant. Invariants and
       * things in nested loops or conditionals should have been removed from
       * the list by compute_invariance_information().
       */
      assert(!var->in_if_branch && !var->in_nested_loop &&
             var->type != invariant);

      /* We are only interested in checking phis for the basic induction
       * variable case as its simple to detect. All basic induction variables
       * have a phi node
       */
      if (!is_var_phi(var))
         continue;

      nir_phi_instr *phi = nir_instr_as_phi(var->def->parent_instr);
      nir_basic_induction_var *biv = rzalloc(state, nir_basic_induction_var);

      nir_src *init_src = NULL;
      nir_loop_variable *alu_src_var = NULL;
      nir_foreach_phi_src(src, phi) {
         nir_loop_variable *src_var = get_loop_var(src->src.ssa, state);

         /* If one of the sources is in an if branch or nested loop then don't
          * attempt to go any further.
          */
         if (src_var->in_if_branch || src_var->in_nested_loop)
            break;

         /* Detect inductions variables that are incremented in both branches
          * of an unnested if rather than in a loop block.
          */
         if (is_var_phi(src_var)) {
            nir_phi_instr *src_phi =
               nir_instr_as_phi(src_var->def->parent_instr);
            nir_alu_instr *src_phi_alu = phi_instr_as_alu(src_phi);
            if (src_phi_alu) {
               src_var = get_loop_var(&src_phi_alu->dest.dest.ssa, state);
               if (!src_var->in_if_branch)
                  break;
            }
         }

         if (!src_var->in_loop && !biv->def_outside_loop) {
            biv->def_outside_loop = src_var->def;
            init_src = &src->src;
         } else if (is_var_alu(src_var) && !biv->alu) {
            alu_src_var = src_var;
            nir_alu_instr *alu = nir_instr_as_alu(src_var->def->parent_instr);

            /* Check for unsupported alu operations */
            if (alu->op != nir_op_iadd && alu->op != nir_op_fadd)
               break;

            if (nir_op_infos[alu->op].num_inputs == 2) {
               for (unsigned i = 0; i < 2; i++) {
                  /* Is one of the operands const or uniform, and the other the phi.
                   * The phi source can't be swizzled in any way.
                   */
                  if (alu->src[1-i].src.ssa == &phi->dest.ssa &&
                      alu_src_has_identity_swizzle(alu, 1 - i)) {
                     nir_src *src = &alu->src[i].src;
                     if (nir_src_is_const(*src))
                        biv->alu = alu;
                     else if (is_only_uniform_src(src)) {
                        /* Update value of induction variable is a statement
                         * contains only uniform and constant
                         */
                        var->update_src = alu->src + i;
                        biv->alu = alu;
                     }
                  }
               }
            }

            if (!biv->alu)
               break;
         } else {
            biv->alu = NULL;
            break;
         }
      }

      if (biv->alu && biv->def_outside_loop) {
         nir_instr *inst = biv->def_outside_loop->parent_instr;
         if (inst->type == nir_instr_type_load_const)  {
            /* Initial value of induction variable is a constant */
            if (var->update_src) {
               alu_src_var->update_src = var->update_src;
               ralloc_free(biv);
            } else {
               alu_src_var->type = basic_induction;
               alu_src_var->ind = biv;
               var->type = basic_induction;
               var->ind = biv;

               found_induction_var = true;
            }
            num_induction_vars += 2;
         } else if (is_only_uniform_src(init_src)) {
            /* Initial value of induction variable is a uniform */
            var->init_src = init_src;

            alu_src_var->init_src = var->init_src;
            alu_src_var->update_src = var->update_src;

            num_induction_vars += 2;
            ralloc_free(biv);
         } else {
            var->update_src = NULL;
            ralloc_free(biv);
         }
      } else {
         var->update_src = NULL;
         ralloc_free(biv);
      }
   }

   nir_loop_info *info = state->loop->info;
   ralloc_free(info->induction_vars);
   info->num_induction_vars = 0;

   /* record induction variables into nir_loop_info */
   if (num_induction_vars) {
      info->induction_vars = ralloc_array(info, nir_loop_induction_variable,
                                          num_induction_vars);

      list_for_each_entry(nir_loop_variable, var, &state->process_list,
                          process_link) {
         if (var->type == basic_induction || var->init_src || var->update_src) {
            nir_loop_induction_variable *ivar =
               &info->induction_vars[info->num_induction_vars++];
             ivar->def = var->def;
             ivar->init_src = var->init_src;
             ivar->update_src = var->update_src;
         }
      }
      /* don't overflow */
      assert(info->num_induction_vars <= num_induction_vars);
   }

   return found_induction_var;
}

static bool
find_loop_terminators(loop_info_state *state)
{
   bool success = false;
   foreach_list_typed_safe(nir_cf_node, node, node, &state->loop->body) {
      if (node->type == nir_cf_node_if) {
         nir_if *nif = nir_cf_node_as_if(node);

         nir_block *break_blk = NULL;
         nir_block *continue_from_blk = NULL;
         bool continue_from_then = true;

         nir_block *last_then = nir_if_last_then_block(nif);
         nir_block *last_else = nir_if_last_else_block(nif);
         if (nir_block_ends_in_break(last_then)) {
            break_blk = last_then;
            continue_from_blk = last_else;
            continue_from_then = false;
         } else if (nir_block_ends_in_break(last_else)) {
            break_blk = last_else;
            continue_from_blk = last_then;
         }

         /* If there is a break then we should find a terminator. If we can
          * not find a loop terminator, but there is a break-statement then
          * we should return false so that we do not try to find trip-count
          */
         if (!nir_is_trivial_loop_if(nif, break_blk)) {
            state->loop->info->complex_loop = true;
            return false;
         }

         /* Continue if the if contained no jumps at all */
         if (!break_blk)
            continue;

         if (nif->condition.ssa->parent_instr->type == nir_instr_type_phi) {
            state->loop->info->complex_loop = true;
            return false;
         }

         nir_loop_terminator *terminator =
            rzalloc(state->loop->info, nir_loop_terminator);

         list_addtail(&terminator->loop_terminator_link,
                      &state->loop->info->loop_terminator_list);

         terminator->nif = nif;
         terminator->break_block = break_blk;
         terminator->continue_from_block = continue_from_blk;
         terminator->continue_from_then = continue_from_then;
         terminator->conditional_instr = nif->condition.ssa->parent_instr;

         success = true;
      }
   }

   return success;
}

/* This function looks for an array access within a loop that uses an
 * induction variable for the array index. If found it returns the size of the
 * array, otherwise 0 is returned. If we find an induction var we pass it back
 * to the caller via array_index_out.
 */
static unsigned
find_array_access_via_induction(loop_info_state *state,
                                nir_deref_instr *deref,
                                nir_loop_variable **array_index_out)
{
   for (nir_deref_instr *d = deref; d; d = nir_deref_instr_parent(d)) {
      if (d->deref_type != nir_deref_type_array)
         continue;

      assert(d->arr.index.is_ssa);
      nir_loop_variable *array_index = get_loop_var(d->arr.index.ssa, state);

      if (array_index->type != basic_induction)
         continue;

      if (array_index_out)
         *array_index_out = array_index;

      nir_deref_instr *parent = nir_deref_instr_parent(d);

      if (glsl_type_is_array_or_matrix(parent->type)) {
         return glsl_get_length(parent->type);
      } else {
         assert(glsl_type_is_vector(parent->type));
         return glsl_get_vector_elements(parent->type);
      }
   }

   return 0;
}

static bool
guess_loop_limit(loop_info_state *state, nir_const_value *limit_val,
                 nir_ssa_scalar basic_ind)
{
   unsigned min_array_size = 0;

   nir_foreach_block_in_cf_node(block, &state->loop->cf_node) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;

         nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);

         /* Check for arrays variably-indexed by a loop induction variable. */
         if (intrin->intrinsic == nir_intrinsic_load_deref ||
             intrin->intrinsic == nir_intrinsic_store_deref ||
             intrin->intrinsic == nir_intrinsic_copy_deref) {

            nir_loop_variable *array_idx = NULL;
            unsigned array_size =
               find_array_access_via_induction(state,
                                               nir_src_as_deref(intrin->src[0]),
                                               &array_idx);
            if (array_idx && basic_ind.def == array_idx->def &&
                (min_array_size == 0 || min_array_size > array_size)) {
               /* Array indices are scalars */
               assert(basic_ind.def->num_components == 1);
               min_array_size = array_size;
            }

            if (intrin->intrinsic != nir_intrinsic_copy_deref)
               continue;

            array_size =
               find_array_access_via_induction(state,
                                               nir_src_as_deref(intrin->src[1]),
                                               &array_idx);
            if (array_idx && basic_ind.def == array_idx->def &&
                (min_array_size == 0 || min_array_size > array_size)) {
               /* Array indices are scalars */
               assert(basic_ind.def->num_components == 1);
               min_array_size = array_size;
            }
         }
      }
   }

   if (min_array_size) {
      *limit_val = nir_const_value_for_uint(min_array_size,
                                            basic_ind.def->bit_size);
      return true;
   }

   return false;
}

static bool
try_find_limit_of_alu(nir_ssa_scalar limit, nir_const_value *limit_val,
                      nir_loop_terminator *terminator, loop_info_state *state)
{
   if (!nir_ssa_scalar_is_alu(limit))
      return false;

   nir_op limit_op = nir_ssa_scalar_alu_op(limit);
   if (limit_op == nir_op_imin || limit_op == nir_op_fmin) {
      for (unsigned i = 0; i < 2; i++) {
         nir_ssa_scalar src = nir_ssa_scalar_chase_alu_src(limit, i);
         if (nir_ssa_scalar_is_const(src)) {
            *limit_val = nir_ssa_scalar_as_const_value(src);
            terminator->exact_trip_count_unknown = true;
            return true;
         }
      }
   }

   return false;
}

static nir_const_value
eval_const_unop(nir_op op, unsigned bit_size, nir_const_value src0,
                unsigned execution_mode)
{
   assert(nir_op_infos[op].num_inputs == 1);
   nir_const_value dest;
   nir_const_value *src[1] = { &src0 };
   nir_eval_const_opcode(op, &dest, 1, bit_size, src, execution_mode);
   return dest;
}

static nir_const_value
eval_const_binop(nir_op op, unsigned bit_size,
                 nir_const_value src0, nir_const_value src1,
                 unsigned execution_mode)
{
   assert(nir_op_infos[op].num_inputs == 2);
   nir_const_value dest;
   nir_const_value *src[2] = { &src0, &src1 };
   nir_eval_const_opcode(op, &dest, 1, bit_size, src, execution_mode);
   return dest;
}

static int32_t
get_iteration(nir_op cond_op, nir_const_value initial, nir_const_value step,
              nir_const_value limit, unsigned bit_size,
              unsigned execution_mode)
{
   nir_const_value span, iter;

   switch (cond_op) {
   case nir_op_ine:
      /* In order for execution to be here, limit must be the same as initial.
       * Otherwise will_break_on_first_iteration would have returned false.
       * If step is zero, the loop is infinite.  Otherwise the loop will
       * execute once.
       */
      return step.u64 == 0 ? -1 : 1;

   case nir_op_ige:
   case nir_op_ilt:
   case nir_op_ieq:
      span = eval_const_binop(nir_op_isub, bit_size, limit, initial,
                              execution_mode);
      iter = eval_const_binop(nir_op_idiv, bit_size, span, step,
                              execution_mode);
      break;

   case nir_op_uge:
   case nir_op_ult:
      span = eval_const_binop(nir_op_isub, bit_size, limit, initial,
                              execution_mode);
      iter = eval_const_binop(nir_op_udiv, bit_size, span, step,
                              execution_mode);
      break;

   case nir_op_fneu:
      /* In order for execution to be here, limit must be the same as initial.
       * Otherwise will_break_on_first_iteration would have returned false.
       * If step is zero, the loop is infinite.  Otherwise the loop will
       * execute once.
       *
       * This is a little more tricky for floating point since X-Y might still
       * be X even if Y is not zero.  Instead check that (initial + step) !=
       * initial.
       */
      span = eval_const_binop(nir_op_fadd, bit_size, initial, step,
                              execution_mode);
      iter = eval_const_binop(nir_op_feq, bit_size, initial,
                              span, execution_mode);

      /* return (initial + step) == initial ? -1 : 1 */
      return iter.b ? -1 : 1;

   case nir_op_fge:
   case nir_op_flt:
   case nir_op_feq:
      span = eval_const_binop(nir_op_fsub, bit_size, limit, initial,
                              execution_mode);
      iter = eval_const_binop(nir_op_fdiv, bit_size, span,
                              step, execution_mode);
      iter = eval_const_unop(nir_op_f2i64, bit_size, iter, execution_mode);
      break;

   default:
      return -1;
   }

   uint64_t iter_u64 = nir_const_value_as_uint(iter, bit_size);
   return iter_u64 > INT_MAX ? -1 : (int)iter_u64;
}

static bool
will_break_on_first_iteration(nir_const_value step,
                              nir_alu_type induction_base_type,
                              unsigned trip_offset,
                              nir_op cond_op, unsigned bit_size,
                              nir_const_value initial,
                              nir_const_value limit,
                              bool limit_rhs, bool invert_cond,
                              unsigned execution_mode)
{
   if (trip_offset == 1) {
      nir_op add_op;
      switch (induction_base_type) {
      case nir_type_float:
         add_op = nir_op_fadd;
         break;
      case nir_type_int:
      case nir_type_uint:
         add_op = nir_op_iadd;
         break;
      default:
         unreachable("Unhandled induction variable base type!");
      }

      initial = eval_const_binop(add_op, bit_size, initial, step,
                                 execution_mode);
   }

   nir_const_value *src[2];
   src[limit_rhs ? 0 : 1] = &initial;
   src[limit_rhs ? 1 : 0] = &limit;

   /* Evaluate the loop exit condition */
   nir_const_value result;
   nir_eval_const_opcode(cond_op, &result, 1, bit_size, src, execution_mode);

   return invert_cond ? !result.b : result.b;
}

static bool
test_iterations(int32_t iter_int, nir_const_value step,
                nir_const_value limit, nir_op cond_op, unsigned bit_size,
                nir_alu_type induction_base_type,
                nir_const_value initial, bool limit_rhs, bool invert_cond,
                unsigned execution_mode)
{
   assert(nir_op_infos[cond_op].num_inputs == 2);

   nir_const_value iter_src;
   nir_op mul_op;
   nir_op add_op;
   switch (induction_base_type) {
   case nir_type_float:
      iter_src = nir_const_value_for_float(iter_int, bit_size);
      mul_op = nir_op_fmul;
      add_op = nir_op_fadd;
      break;
   case nir_type_int:
   case nir_type_uint:
      iter_src = nir_const_value_for_int(iter_int, bit_size);
      mul_op = nir_op_imul;
      add_op = nir_op_iadd;
      break;
   default:
      unreachable("Unhandled induction variable base type!");
   }

   /* Multiple the iteration count we are testing by the number of times we
    * step the induction variable each iteration.
    */
   nir_const_value mul_result =
      eval_const_binop(mul_op, bit_size, iter_src, step, execution_mode);

   /* Add the initial value to the accumulated induction variable total */
   nir_const_value add_result =
      eval_const_binop(add_op, bit_size, mul_result, initial, execution_mode);

   nir_const_value *src[2];
   src[limit_rhs ? 0 : 1] = &add_result;
   src[limit_rhs ? 1 : 0] = &limit;

   /* Evaluate the loop exit condition */
   nir_const_value result;
   nir_eval_const_opcode(cond_op, &result, 1, bit_size, src, execution_mode);

   return invert_cond ? !result.b : result.b;
}

static int
calculate_iterations(nir_const_value initial, nir_const_value step,
                     nir_const_value limit, nir_alu_instr *alu,
                     nir_ssa_scalar cond, nir_op alu_op, bool limit_rhs,
                     bool invert_cond, unsigned execution_mode)
{
   /* nir_op_isub should have been lowered away by this point */
   assert(alu->op != nir_op_isub);

   /* Make sure the alu type for our induction variable is compatible with the
    * conditional alus input type. If its not something has gone really wrong.
    */
   nir_alu_type induction_base_type =
      nir_alu_type_get_base_type(nir_op_infos[alu->op].output_type);
   if (induction_base_type == nir_type_int || induction_base_type == nir_type_uint) {
      assert(nir_alu_type_get_base_type(nir_op_infos[alu_op].input_types[1]) == nir_type_int ||
             nir_alu_type_get_base_type(nir_op_infos[alu_op].input_types[1]) == nir_type_uint);
   } else {
      assert(nir_alu_type_get_base_type(nir_op_infos[alu_op].input_types[0]) ==
             induction_base_type);
   }

   /* Only variable with these update ops were marked as induction. */
   assert(alu->op == nir_op_iadd || alu->op == nir_op_fadd);

   /* do-while loops can increment the starting value before the condition is
    * checked. e.g.
    *
    *    do {
    *        ndx++;
    *     } while (ndx < 3);
    *
    * Here we check if the induction variable is used directly by the loop
    * condition and if so we assume we need to step the initial value.
    */
   unsigned trip_offset = 0;
   nir_alu_instr *cond_alu = nir_instr_as_alu(cond.def->parent_instr);
   if (cond_alu->src[0].src.ssa == &alu->dest.dest.ssa ||
       cond_alu->src[1].src.ssa == &alu->dest.dest.ssa) {
      trip_offset = 1;
   }

   assert(nir_src_bit_size(alu->src[0].src) ==
          nir_src_bit_size(alu->src[1].src));
   unsigned bit_size = nir_src_bit_size(alu->src[0].src);

   /* get_iteration works under assumption that iterator will be
    * incremented or decremented until it hits the limit,
    * however if the loop condition is false on the first iteration
    * get_iteration's assumption is broken. Handle such loops first.
    */
   if (will_break_on_first_iteration(step, induction_base_type, trip_offset,
                                     alu_op, bit_size, initial,
                                     limit, limit_rhs, invert_cond,
                                     execution_mode)) {
      return 0;
   }

   int iter_int = get_iteration(alu_op, initial, step, limit, bit_size,
                                execution_mode);

   /* If iter_int is negative the loop is ill-formed or is the conditional is
    * unsigned with a huge iteration count so don't bother going any further.
    */
   if (iter_int < 0)
      return -1;

   if (alu_op == nir_op_ine || alu_op == nir_op_fneu)
      return iter_int;

   /* An explanation from the GLSL unrolling pass:
    *
    * Make sure that the calculated number of iterations satisfies the exit
    * condition.  This is needed to catch off-by-one errors and some types of
    * ill-formed loops.  For example, we need to detect that the following
    * loop does not have a maximum iteration count.
    *
    *    for (float x = 0.0; x != 0.9; x += 0.2);
    */
   for (int bias = -1; bias <= 1; bias++) {
      const int iter_bias = iter_int + bias;

      if (test_iterations(iter_bias, step, limit, alu_op, bit_size,
                          induction_base_type, initial,
                          limit_rhs, invert_cond, execution_mode)) {
         return iter_bias > 0 ? iter_bias - trip_offset : iter_bias;
      }
   }

   return -1;
}

static nir_op
inverse_comparison(nir_op alu_op)
{
   switch (alu_op) {
   case nir_op_fge:
      return nir_op_flt;
   case nir_op_ige:
      return nir_op_ilt;
   case nir_op_uge:
      return nir_op_ult;
   case nir_op_flt:
      return nir_op_fge;
   case nir_op_ilt:
      return nir_op_ige;
   case nir_op_ult:
      return nir_op_uge;
   case nir_op_feq:
      return nir_op_fneu;
   case nir_op_ieq:
      return nir_op_ine;
   case nir_op_fneu:
      return nir_op_feq;
   case nir_op_ine:
      return nir_op_ieq;
   default:
      unreachable("Unsuported comparison!");
   }
}

static bool
get_induction_and_limit_vars(nir_ssa_scalar cond,
                             nir_ssa_scalar *ind,
                             nir_ssa_scalar *limit,
                             bool *limit_rhs,
                             loop_info_state *state)
{
   nir_ssa_scalar rhs, lhs;
   lhs = nir_ssa_scalar_chase_alu_src(cond, 0);
   rhs = nir_ssa_scalar_chase_alu_src(cond, 1);

   if (get_loop_var(lhs.def, state)->type == basic_induction) {
      *ind = lhs;
      *limit = rhs;
      *limit_rhs = true;
      return true;
   } else if (get_loop_var(rhs.def, state)->type == basic_induction) {
      *ind = rhs;
      *limit = lhs;
      *limit_rhs = false;
      return true;
   } else {
      return false;
   }
}

static bool
try_find_trip_count_vars_in_iand(nir_ssa_scalar *cond,
                                 nir_ssa_scalar *ind,
                                 nir_ssa_scalar *limit,
                                 bool *limit_rhs,
                                 loop_info_state *state)
{
   const nir_op alu_op = nir_ssa_scalar_alu_op(*cond);
   assert(alu_op == nir_op_ieq || alu_op == nir_op_inot);

   nir_ssa_scalar iand = nir_ssa_scalar_chase_alu_src(*cond, 0);

   if (alu_op == nir_op_ieq) {
      nir_ssa_scalar zero = nir_ssa_scalar_chase_alu_src(*cond, 1);

      if (!nir_ssa_scalar_is_alu(iand) || !nir_ssa_scalar_is_const(zero)) {
         /* Maybe we had it the wrong way, flip things around */
         nir_ssa_scalar tmp = zero;
         zero = iand;
         iand = tmp;

         /* If we still didn't find what we need then return */
         if (!nir_ssa_scalar_is_const(zero))
            return false;
      }

      /* If the loop is not breaking on (x && y) == 0 then return */
      if (nir_ssa_scalar_as_uint(zero) != 0)
         return false;
   }

   if (!nir_ssa_scalar_is_alu(iand))
      return false;

   if (nir_ssa_scalar_alu_op(iand) != nir_op_iand)
      return false;

   /* Check if iand src is a terminator condition and try get induction var
    * and trip limit var.
    */
   bool found_induction_var = false;
   for (unsigned i = 0; i < 2; i++) {
      nir_ssa_scalar src = nir_ssa_scalar_chase_alu_src(iand, i);
      if (nir_is_terminator_condition_with_two_inputs(src) &&
          get_induction_and_limit_vars(src, ind, limit, limit_rhs, state)) {
         *cond = src;
         found_induction_var = true;

         /* If we've found one with a constant limit, stop. */
         if (nir_ssa_scalar_is_const(*limit))
            return true;
      }
   }

   return found_induction_var;
}

/* Run through each of the terminators of the loop and try to infer a possible
 * trip-count. We need to check them all, and set the lowest trip-count as the
 * trip-count of our loop. If one of the terminators has an undecidable
 * trip-count we can not safely assume anything about the duration of the
 * loop.
 */
static void
find_trip_count(loop_info_state *state, unsigned execution_mode)
{
   bool trip_count_known = true;
   bool guessed_trip_count = false;
   nir_loop_terminator *limiting_terminator = NULL;
   int max_trip_count = -1;

   list_for_each_entry(nir_loop_terminator, terminator,
                       &state->loop->info->loop_terminator_list,
                       loop_terminator_link) {
      assert(terminator->nif->condition.is_ssa);
      nir_ssa_scalar cond = { terminator->nif->condition.ssa, 0 };

      if (!nir_ssa_scalar_is_alu(cond)) {
         /* If we get here the loop is dead and will get cleaned up by the
          * nir_opt_dead_cf pass.
          */
         trip_count_known = false;
         terminator->exact_trip_count_unknown = true;
         continue;
      }

      nir_op alu_op = nir_ssa_scalar_alu_op(cond);

      bool limit_rhs;
      nir_ssa_scalar basic_ind = { NULL, 0 };
      nir_ssa_scalar limit;
      if ((alu_op == nir_op_inot || alu_op == nir_op_ieq) &&
          try_find_trip_count_vars_in_iand(&cond, &basic_ind, &limit,
                                           &limit_rhs, state)) {

         /* The loop is exiting on (x && y) == 0 so we need to get the
          * inverse of x or y (i.e. which ever contained the induction var) in
          * order to compute the trip count.
          */
         alu_op = inverse_comparison(nir_ssa_scalar_alu_op(cond));
         trip_count_known = false;
         terminator->exact_trip_count_unknown = true;
      }

      if (!basic_ind.def) {
         if (nir_is_supported_terminator_condition(cond)) {
            /* Extract and inverse the comparision if it is wrapped in an inot
             */
            if (alu_op == nir_op_inot) {
               cond = nir_ssa_scalar_chase_alu_src(cond, 0);
               alu_op = inverse_comparison(nir_ssa_scalar_alu_op(cond));
            }

            get_induction_and_limit_vars(cond, &basic_ind,
                                         &limit, &limit_rhs, state);
         }
      }

      /* The comparison has to have a basic induction variable for us to be
       * able to find trip counts.
       */
      if (!basic_ind.def) {
         trip_count_known = false;
         terminator->exact_trip_count_unknown = true;
         continue;
      }

      terminator->induction_rhs = !limit_rhs;

      /* Attempt to find a constant limit for the loop */
      nir_const_value limit_val;
      if (nir_ssa_scalar_is_const(limit)) {
         limit_val = nir_ssa_scalar_as_const_value(limit);
      } else {
         trip_count_known = false;

         if (!try_find_limit_of_alu(limit, &limit_val, terminator, state)) {
            /* Guess loop limit based on array access */
            if (!guess_loop_limit(state, &limit_val, basic_ind)) {
               terminator->exact_trip_count_unknown = true;
               continue;
            }

            guessed_trip_count = true;
         }
      }

      /* We have determined that we have the following constants:
       * (With the typical int i = 0; i < x; i++; as an example)
       *    - Upper limit.
       *    - Starting value
       *    - Step / iteration size
       * Thats all thats needed to calculate the trip-count
       */

      nir_basic_induction_var *ind_var =
         get_loop_var(basic_ind.def, state)->ind;

      /* The basic induction var might be a vector but, because we guarantee
       * earlier that the phi source has a scalar swizzle, we can take the
       * component from basic_ind.
       */
      nir_ssa_scalar initial_s = { ind_var->def_outside_loop, basic_ind.comp };
      nir_ssa_scalar alu_s = { &ind_var->alu->dest.dest.ssa, basic_ind.comp };

      nir_const_value initial_val = nir_ssa_scalar_as_const_value(initial_s);

      /* We are guaranteed by earlier code that at least one of these sources
       * is a constant but we don't know which.
       */
      nir_const_value step_val;
      memset(&step_val, 0, sizeof(step_val));
      UNUSED bool found_step_value = false;
      assert(nir_op_infos[ind_var->alu->op].num_inputs == 2);
      for (unsigned i = 0; i < 2; i++) {
         nir_ssa_scalar alu_src = nir_ssa_scalar_chase_alu_src(alu_s, i);
         if (nir_ssa_scalar_is_const(alu_src)) {
            found_step_value = true;
            step_val = nir_ssa_scalar_as_const_value(alu_src);
            break;
         }
      }
      assert(found_step_value);

      int iterations = calculate_iterations(initial_val, step_val, limit_val,
                                            ind_var->alu, cond,
                                            alu_op, limit_rhs,
                                            terminator->continue_from_then,
                                            execution_mode);

      /* Where we not able to calculate the iteration count */
      if (iterations == -1) {
         trip_count_known = false;
         guessed_trip_count = false;
         terminator->exact_trip_count_unknown = true;
         continue;
      }

      if (guessed_trip_count) {
         guessed_trip_count = false;
         terminator->exact_trip_count_unknown = true;
         if (state->loop->info->guessed_trip_count == 0 ||
             state->loop->info->guessed_trip_count > iterations)
            state->loop->info->guessed_trip_count = iterations;

         continue;
      }

      /* If this is the first run or we have found a smaller amount of
       * iterations than previously (we have identified a more limiting
       * terminator) set the trip count and limiting terminator.
       */
      if (max_trip_count == -1 || iterations < max_trip_count) {
         max_trip_count = iterations;
         limiting_terminator = terminator;
      }
   }

   state->loop->info->exact_trip_count_known = trip_count_known;
   if (max_trip_count > -1)
      state->loop->info->max_trip_count = max_trip_count;
   state->loop->info->limiting_terminator = limiting_terminator;
}

static bool
force_unroll_array_access(loop_info_state *state, nir_deref_instr *deref,
                          bool contains_sampler)
{
   unsigned array_size = find_array_access_via_induction(state, deref, NULL);
   if (array_size) {
      if ((array_size == state->loop->info->max_trip_count) &&
          nir_deref_mode_must_be(deref, nir_var_shader_in |
                                        nir_var_shader_out |
                                        nir_var_shader_temp |
                                        nir_var_function_temp))
         return true;

      if (nir_deref_mode_must_be(deref, state->indirect_mask))
         return true;

      if (contains_sampler && state->force_unroll_sampler_indirect)
         return true;
   }

   return false;
}

static bool
force_unroll_heuristics(loop_info_state *state, nir_block *block)
{
   nir_foreach_instr(instr, block) {
      if (instr->type == nir_instr_type_tex) {
         nir_tex_instr *tex_instr = nir_instr_as_tex(instr);
         int sampler_idx =
            nir_tex_instr_src_index(tex_instr,
                                    nir_tex_src_sampler_deref);


         if (sampler_idx >= 0) {
            nir_deref_instr *deref =
               nir_instr_as_deref(tex_instr->src[sampler_idx].src.ssa->parent_instr);
            if (force_unroll_array_access(state, deref, true))
               return true;
         }
      }


      if (instr->type != nir_instr_type_intrinsic)
         continue;

      nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);

      /* Check for arrays variably-indexed by a loop induction variable.
       * Unrolling the loop may convert that access into constant-indexing.
       */
      if (intrin->intrinsic == nir_intrinsic_load_deref ||
          intrin->intrinsic == nir_intrinsic_store_deref ||
          intrin->intrinsic == nir_intrinsic_copy_deref) {
         if (force_unroll_array_access(state,
                                       nir_src_as_deref(intrin->src[0]),
                                       false))
            return true;

         if (intrin->intrinsic == nir_intrinsic_copy_deref &&
             force_unroll_array_access(state,
                                       nir_src_as_deref(intrin->src[1]),
                                       false))
            return true;
      }
   }

   return false;
}

static void
get_loop_info(loop_info_state *state, nir_function_impl *impl)
{
   nir_shader *shader = impl->function->shader;
   const nir_shader_compiler_options *options = shader->options;

   /* Add all entries in the outermost part of the loop to the processing list
    * Mark the entries in conditionals or in nested loops accordingly
    */
   foreach_list_typed_safe(nir_cf_node, node, node, &state->loop->body) {
      switch (node->type) {

      case nir_cf_node_block:
         init_loop_block(nir_cf_node_as_block(node), state, false, false);
         break;

      case nir_cf_node_if:
         nir_foreach_block_in_cf_node(block, node)
            init_loop_block(block, state, true, false);
         break;

      case nir_cf_node_loop:
         nir_foreach_block_in_cf_node(block, node) {
            init_loop_block(block, state, false, true);
         }
         break;

      case nir_cf_node_function:
         break;
      }
   }

   /* Try to find all simple terminators of the loop. If we can't find any,
    * or we find possible terminators that have side effects then bail.
    */
   if (!find_loop_terminators(state)) {
      list_for_each_entry_safe(nir_loop_terminator, terminator,
                               &state->loop->info->loop_terminator_list,
                               loop_terminator_link) {
         list_del(&terminator->loop_terminator_link);
         ralloc_free(terminator);
      }
      return;
   }

   /* Induction analysis needs invariance information so get that first */
   compute_invariance_information(state);

   /* We have invariance information so try to find induction variables */
   if (!compute_induction_information(state))
      return;

   /* Run through each of the terminators and try to compute a trip-count */
   find_trip_count(state, impl->function->shader->info.float_controls_execution_mode);

   nir_foreach_block_in_cf_node(block, &state->loop->cf_node) {
      nir_foreach_instr(instr, block) {
         state->loop->info->instr_cost += instr_cost(state, instr, options);
      }

      if (state->loop->info->force_unroll)
         continue;

      if (force_unroll_heuristics(state, block)) {
         state->loop->info->force_unroll = true;
      }
   }
}

static loop_info_state *
initialize_loop_info_state(nir_loop *loop, void *mem_ctx,
                           nir_function_impl *impl)
{
   loop_info_state *state = rzalloc(mem_ctx, loop_info_state);
   state->loop_vars = ralloc_array(mem_ctx, nir_loop_variable,
                                   impl->ssa_alloc);
   state->loop_vars_init = rzalloc_array(mem_ctx, BITSET_WORD,
                                         BITSET_WORDS(impl->ssa_alloc));
   state->loop = loop;

   list_inithead(&state->process_list);

   if (loop->info)
     ralloc_free(loop->info);

   loop->info = rzalloc(loop, nir_loop_info);

   list_inithead(&loop->info->loop_terminator_list);

   return state;
}

static void
process_loops(nir_cf_node *cf_node, nir_variable_mode indirect_mask,
              bool force_unroll_sampler_indirect)
{
   switch (cf_node->type) {
   case nir_cf_node_block:
      return;
   case nir_cf_node_if: {
      nir_if *if_stmt = nir_cf_node_as_if(cf_node);
      foreach_list_typed(nir_cf_node, nested_node, node, &if_stmt->then_list)
         process_loops(nested_node, indirect_mask, force_unroll_sampler_indirect);
      foreach_list_typed(nir_cf_node, nested_node, node, &if_stmt->else_list)
         process_loops(nested_node, indirect_mask, force_unroll_sampler_indirect);
      return;
   }
   case nir_cf_node_loop: {
      nir_loop *loop = nir_cf_node_as_loop(cf_node);
      foreach_list_typed(nir_cf_node, nested_node, node, &loop->body)
         process_loops(nested_node, indirect_mask, force_unroll_sampler_indirect);
      break;
   }
   default:
      unreachable("unknown cf node type");
   }

   nir_loop *loop = nir_cf_node_as_loop(cf_node);
   nir_function_impl *impl = nir_cf_node_get_function(cf_node);
   void *mem_ctx = ralloc_context(NULL);

   loop_info_state *state = initialize_loop_info_state(loop, mem_ctx, impl);
   state->indirect_mask = indirect_mask;
   state->force_unroll_sampler_indirect = force_unroll_sampler_indirect;

   get_loop_info(state, impl);

   ralloc_free(mem_ctx);
}

void
nir_loop_analyze_impl(nir_function_impl *impl,
                      nir_variable_mode indirect_mask,
                      bool force_unroll_sampler_indirect)
{
   nir_index_ssa_defs(impl);
   foreach_list_typed(nir_cf_node, node, node, &impl->body)
      process_loops(node, indirect_mask, force_unroll_sampler_indirect);
}

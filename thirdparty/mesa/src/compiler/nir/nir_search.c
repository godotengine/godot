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

#include <inttypes.h>
#include "nir_search.h"
#include "nir_builder.h"
#include "nir_worklist.h"
#include "util/half_float.h"

/* This should be the same as nir_search_max_comm_ops in nir_algebraic.py. */
#define NIR_SEARCH_MAX_COMM_OPS 8

struct match_state {
   bool inexact_match;
   bool has_exact_alu;
   uint8_t comm_op_direction;
   unsigned variables_seen;

   /* Used for running the automaton on newly-constructed instructions. */
   struct util_dynarray *states;
   const struct per_op_table *pass_op_table;
   const nir_algebraic_table *table;

   nir_alu_src variables[NIR_SEARCH_MAX_VARIABLES];
   struct hash_table *range_ht;
};

static bool
match_expression(const nir_algebraic_table *table, const nir_search_expression *expr, nir_alu_instr *instr,
                 unsigned num_components, const uint8_t *swizzle,
                 struct match_state *state);
static bool
nir_algebraic_automaton(nir_instr *instr, struct util_dynarray *states,
                        const struct per_op_table *pass_op_table);

static const uint8_t identity_swizzle[NIR_MAX_VEC_COMPONENTS] =
{
    0,  1,  2,  3,
    4,  5,  6,  7,
    8,  9, 10, 11,
   12, 13, 14, 15,
};

/**
 * Check if a source produces a value of the given type.
 *
 * Used for satisfying 'a@type' constraints.
 */
static bool
src_is_type(nir_src src, nir_alu_type type)
{
   assert(type != nir_type_invalid);

   if (!src.is_ssa)
      return false;

   if (src.ssa->parent_instr->type == nir_instr_type_alu) {
      nir_alu_instr *src_alu = nir_instr_as_alu(src.ssa->parent_instr);
      nir_alu_type output_type = nir_op_infos[src_alu->op].output_type;

      if (type == nir_type_bool) {
         switch (src_alu->op) {
         case nir_op_iand:
         case nir_op_ior:
         case nir_op_ixor:
            return src_is_type(src_alu->src[0].src, nir_type_bool) &&
                   src_is_type(src_alu->src[1].src, nir_type_bool);
         case nir_op_inot:
            return src_is_type(src_alu->src[0].src, nir_type_bool);
         default:
            break;
         }
      }

      return nir_alu_type_get_base_type(output_type) == type;
   } else if (src.ssa->parent_instr->type == nir_instr_type_intrinsic) {
      nir_intrinsic_instr *intr = nir_instr_as_intrinsic(src.ssa->parent_instr);

      if (type == nir_type_bool) {
         return intr->intrinsic == nir_intrinsic_load_front_face ||
                intr->intrinsic == nir_intrinsic_load_helper_invocation;
      }
   }

   /* don't know */
   return false;
}

static bool
nir_op_matches_search_op(nir_op nop, uint16_t sop)
{
   if (sop <= nir_last_opcode)
      return nop == sop;

#define MATCH_FCONV_CASE(op) \
   case nir_search_op_##op: \
      return nop == nir_op_##op##16 || \
             nop == nir_op_##op##32 || \
             nop == nir_op_##op##64;

#define MATCH_ICONV_CASE(op) \
   case nir_search_op_##op: \
      return nop == nir_op_##op##8 || \
             nop == nir_op_##op##16 || \
             nop == nir_op_##op##32 || \
             nop == nir_op_##op##64;

#define MATCH_BCONV_CASE(op) \
   case nir_search_op_##op: \
      return nop == nir_op_##op##1 || \
             nop == nir_op_##op##32;

   switch (sop) {
   MATCH_FCONV_CASE(i2f)
   MATCH_FCONV_CASE(u2f)
   MATCH_FCONV_CASE(f2f)
   MATCH_ICONV_CASE(f2u)
   MATCH_ICONV_CASE(f2i)
   MATCH_ICONV_CASE(u2u)
   MATCH_ICONV_CASE(i2i)
   MATCH_FCONV_CASE(b2f)
   MATCH_ICONV_CASE(b2i)
   MATCH_BCONV_CASE(f2b)
   default:
      unreachable("Invalid nir_search_op");
   }

#undef MATCH_FCONV_CASE
#undef MATCH_ICONV_CASE
#undef MATCH_BCONV_CASE
}

uint16_t
nir_search_op_for_nir_op(nir_op nop)
{
#define MATCH_FCONV_CASE(op) \
   case nir_op_##op##16: \
   case nir_op_##op##32: \
   case nir_op_##op##64: \
      return nir_search_op_##op;

#define MATCH_ICONV_CASE(op) \
   case nir_op_##op##8: \
   case nir_op_##op##16: \
   case nir_op_##op##32: \
   case nir_op_##op##64: \
      return nir_search_op_##op;

#define MATCH_BCONV_CASE(op) \
   case nir_op_##op##1: \
   case nir_op_##op##32: \
      return nir_search_op_##op;


   switch (nop) {
   MATCH_FCONV_CASE(i2f)
   MATCH_FCONV_CASE(u2f)
   MATCH_FCONV_CASE(f2f)
   MATCH_ICONV_CASE(f2u)
   MATCH_ICONV_CASE(f2i)
   MATCH_ICONV_CASE(u2u)
   MATCH_ICONV_CASE(i2i)
   MATCH_FCONV_CASE(b2f)
   MATCH_ICONV_CASE(b2i)
   MATCH_BCONV_CASE(f2b)
   default:
      return nop;
   }

#undef MATCH_FCONV_CASE
#undef MATCH_ICONV_CASE
#undef MATCH_BCONV_CASE
}

static nir_op
nir_op_for_search_op(uint16_t sop, unsigned bit_size)
{
   if (sop <= nir_last_opcode)
      return sop;

#define RET_FCONV_CASE(op) \
   case nir_search_op_##op: \
      switch (bit_size) { \
      case 16: return nir_op_##op##16; \
      case 32: return nir_op_##op##32; \
      case 64: return nir_op_##op##64; \
      default: unreachable("Invalid bit size"); \
      }

#define RET_ICONV_CASE(op) \
   case nir_search_op_##op: \
      switch (bit_size) { \
      case 8:  return nir_op_##op##8; \
      case 16: return nir_op_##op##16; \
      case 32: return nir_op_##op##32; \
      case 64: return nir_op_##op##64; \
      default: unreachable("Invalid bit size"); \
      }

#define RET_BCONV_CASE(op) \
   case nir_search_op_##op: \
      switch (bit_size) { \
      case 1: return nir_op_##op##1; \
      case 32: return nir_op_##op##32; \
      default: unreachable("Invalid bit size"); \
      }

   switch (sop) {
   RET_FCONV_CASE(i2f)
   RET_FCONV_CASE(u2f)
   RET_FCONV_CASE(f2f)
   RET_ICONV_CASE(f2u)
   RET_ICONV_CASE(f2i)
   RET_ICONV_CASE(u2u)
   RET_ICONV_CASE(i2i)
   RET_FCONV_CASE(b2f)
   RET_ICONV_CASE(b2i)
   RET_BCONV_CASE(f2b)
   default:
      unreachable("Invalid nir_search_op");
   }

#undef RET_FCONV_CASE
#undef RET_ICONV_CASE
#undef RET_BCONV_CASE
}

static bool
match_value(const nir_algebraic_table *table,
            const nir_search_value *value, nir_alu_instr *instr, unsigned src,
            unsigned num_components, const uint8_t *swizzle,
            struct match_state *state)
{
   uint8_t new_swizzle[NIR_MAX_VEC_COMPONENTS];

   /* Searching only works on SSA values because, if it's not SSA, we can't
    * know if the value changed between one instance of that value in the
    * expression and another.  Also, the replace operation will place reads of
    * that value right before the last instruction in the expression we're
    * replacing so those reads will happen after the original reads and may
    * not be valid if they're register reads.
    */
   assert(instr->src[src].src.is_ssa);

   /* If the source is an explicitly sized source, then we need to reset
    * both the number of components and the swizzle.
    */
   if (nir_op_infos[instr->op].input_sizes[src] != 0) {
      num_components = nir_op_infos[instr->op].input_sizes[src];
      swizzle = identity_swizzle;
   }

   for (unsigned i = 0; i < num_components; ++i)
      new_swizzle[i] = instr->src[src].swizzle[swizzle[i]];

   /* If the value has a specific bit size and it doesn't match, bail */
   if (value->bit_size > 0 &&
       nir_src_bit_size(instr->src[src].src) != value->bit_size)
      return false;

   switch (value->type) {
   case nir_search_value_expression:
      if (instr->src[src].src.ssa->parent_instr->type != nir_instr_type_alu)
         return false;

      return match_expression(table, nir_search_value_as_expression(value),
                              nir_instr_as_alu(instr->src[src].src.ssa->parent_instr),
                              num_components, new_swizzle, state);

   case nir_search_value_variable: {
      nir_search_variable *var = nir_search_value_as_variable(value);
      assert(var->variable < NIR_SEARCH_MAX_VARIABLES);

      if (state->variables_seen & (1 << var->variable)) {
         if (state->variables[var->variable].src.ssa != instr->src[src].src.ssa)
            return false;

         assert(!instr->src[src].abs && !instr->src[src].negate);

         for (unsigned i = 0; i < num_components; ++i) {
            if (state->variables[var->variable].swizzle[i] != new_swizzle[i])
               return false;
         }

         return true;
      } else {
         if (var->is_constant &&
             instr->src[src].src.ssa->parent_instr->type != nir_instr_type_load_const)
            return false;

         if (var->cond_index != -1 && !table->variable_cond[var->cond_index](state->range_ht, instr,
                                                                             src, num_components, new_swizzle))
            return false;

         if (var->type != nir_type_invalid &&
             !src_is_type(instr->src[src].src, var->type))
            return false;

         state->variables_seen |= (1 << var->variable);
         state->variables[var->variable].src = instr->src[src].src;
         state->variables[var->variable].abs = false;
         state->variables[var->variable].negate = false;

         for (unsigned i = 0; i < NIR_MAX_VEC_COMPONENTS; ++i) {
            if (i < num_components)
               state->variables[var->variable].swizzle[i] = new_swizzle[i];
            else
               state->variables[var->variable].swizzle[i] = 0;
         }

         return true;
      }
   }

   case nir_search_value_constant: {
      nir_search_constant *const_val = nir_search_value_as_constant(value);

      if (!nir_src_is_const(instr->src[src].src))
         return false;

      switch (const_val->type) {
      case nir_type_float: {
         nir_load_const_instr *const load =
            nir_instr_as_load_const(instr->src[src].src.ssa->parent_instr);

         /* There are 8-bit and 1-bit integer types, but there are no 8-bit or
          * 1-bit float types.  This prevents potential assertion failures in
          * nir_src_comp_as_float.
          */
         if (load->def.bit_size < 16)
            return false;

         for (unsigned i = 0; i < num_components; ++i) {
            double val = nir_src_comp_as_float(instr->src[src].src,
                                               new_swizzle[i]);
            if (val != const_val->data.d)
               return false;
         }
         return true;
      }

      case nir_type_int:
      case nir_type_uint:
      case nir_type_bool: {
         unsigned bit_size = nir_src_bit_size(instr->src[src].src);
         uint64_t mask = u_uintN_max(bit_size);
         for (unsigned i = 0; i < num_components; ++i) {
            uint64_t val = nir_src_comp_as_uint(instr->src[src].src,
                                                new_swizzle[i]);
            if ((val & mask) != (const_val->data.u & mask))
               return false;
         }
         return true;
      }

      default:
         unreachable("Invalid alu source type");
      }
   }

   default:
      unreachable("Invalid search value type");
   }
}

static bool
match_expression(const nir_algebraic_table *table, const nir_search_expression *expr, nir_alu_instr *instr,
                 unsigned num_components, const uint8_t *swizzle,
                 struct match_state *state)
{
   if (expr->cond_index != -1 && !table->expression_cond[expr->cond_index](instr))
      return false;

   if (!nir_op_matches_search_op(instr->op, expr->opcode))
      return false;

   assert(instr->dest.dest.is_ssa);

   if (expr->value.bit_size > 0 &&
       instr->dest.dest.ssa.bit_size != expr->value.bit_size)
      return false;

   state->inexact_match = expr->inexact || state->inexact_match;
   state->has_exact_alu = (instr->exact && !expr->ignore_exact) || state->has_exact_alu;
   if (state->inexact_match && state->has_exact_alu)
      return false;

   assert(!instr->dest.saturate);
   assert(nir_op_infos[instr->op].num_inputs > 0);

   /* If we have an explicitly sized destination, we can only handle the
    * identity swizzle.  While dot(vec3(a, b, c).zxy) is a valid
    * expression, we don't have the information right now to propagate that
    * swizzle through.  We can only properly propagate swizzles if the
    * instruction is vectorized.
    */
   if (nir_op_infos[instr->op].output_size != 0) {
      for (unsigned i = 0; i < num_components; i++) {
         if (swizzle[i] != i)
            return false;
      }
   }

   /* If this is a commutative expression and it's one of the first few, look
    * up its direction for the current search operation.  We'll use that value
    * to possibly flip the sources for the match.
    */
   unsigned comm_op_flip =
      (expr->comm_expr_idx >= 0 &&
       expr->comm_expr_idx < NIR_SEARCH_MAX_COMM_OPS) ?
      ((state->comm_op_direction >> expr->comm_expr_idx) & 1) : 0;

   bool matched = true;
   for (unsigned i = 0; i < nir_op_infos[instr->op].num_inputs; i++) {
      /* 2src_commutative instructions that have 3 sources are only commutative
       * in the first two sources.  Source 2 is always source 2.
       */
      if (!match_value(table, &state->table->values[expr->srcs[i]].value, instr,
                       i < 2 ? i ^ comm_op_flip : i,
                       num_components, swizzle, state)) {
         matched = false;
         break;
      }
   }

   return matched;
}

static unsigned
replace_bitsize(const nir_search_value *value, unsigned search_bitsize,
                struct match_state *state)
{
   if (value->bit_size > 0)
      return value->bit_size;
   if (value->bit_size < 0)
      return nir_src_bit_size(state->variables[-value->bit_size - 1].src);
   return search_bitsize;
}

static nir_alu_src
construct_value(nir_builder *build,
                const nir_search_value *value,
                unsigned num_components, unsigned search_bitsize,
                struct match_state *state,
                nir_instr *instr)
{
   switch (value->type) {
   case nir_search_value_expression: {
      const nir_search_expression *expr = nir_search_value_as_expression(value);
      unsigned dst_bit_size = replace_bitsize(value, search_bitsize, state);
      nir_op op = nir_op_for_search_op(expr->opcode, dst_bit_size);

      if (nir_op_infos[op].output_size != 0)
         num_components = nir_op_infos[op].output_size;

      nir_alu_instr *alu = nir_alu_instr_create(build->shader, op);
      nir_ssa_dest_init(&alu->instr, &alu->dest.dest, num_components,
                        dst_bit_size, NULL);
      alu->dest.write_mask = (1 << num_components) - 1;
      alu->dest.saturate = false;

      /* We have no way of knowing what values in a given search expression
       * map to a particular replacement value.  Therefore, if the
       * expression we are replacing has any exact values, the entire
       * replacement should be exact.
       */
      alu->exact = state->has_exact_alu || expr->exact;

      for (unsigned i = 0; i < nir_op_infos[op].num_inputs; i++) {
         /* If the source is an explicitly sized source, then we need to reset
          * the number of components to match.
          */
         if (nir_op_infos[alu->op].input_sizes[i] != 0)
            num_components = nir_op_infos[alu->op].input_sizes[i];

         alu->src[i] = construct_value(build, &state->table->values[expr->srcs[i]].value,
                                       num_components, search_bitsize,
                                       state, instr);
      }

      nir_builder_instr_insert(build, &alu->instr);

      assert(alu->dest.dest.ssa.index ==
             util_dynarray_num_elements(state->states, uint16_t));
      util_dynarray_append(state->states, uint16_t, 0);
      nir_algebraic_automaton(&alu->instr, state->states, state->pass_op_table);

      nir_alu_src val;
      val.src = nir_src_for_ssa(&alu->dest.dest.ssa);
      val.negate = false;
      val.abs = false,
      memcpy(val.swizzle, identity_swizzle, sizeof val.swizzle);

      return val;
   }

   case nir_search_value_variable: {
      const nir_search_variable *var = nir_search_value_as_variable(value);
      assert(state->variables_seen & (1 << var->variable));

      nir_alu_src val = { NIR_SRC_INIT };
      nir_alu_src_copy(&val, &state->variables[var->variable], NULL);
      assert(!var->is_constant);

      for (unsigned i = 0; i < NIR_MAX_VEC_COMPONENTS; i++)
         val.swizzle[i] = state->variables[var->variable].swizzle[var->swizzle[i]];

      return val;
   }

   case nir_search_value_constant: {
      const nir_search_constant *c = nir_search_value_as_constant(value);
      unsigned bit_size = replace_bitsize(value, search_bitsize, state);

      nir_ssa_def *cval;
      switch (c->type) {
      case nir_type_float:
         cval = nir_imm_floatN_t(build, c->data.d, bit_size);
         break;

      case nir_type_int:
      case nir_type_uint:
         cval = nir_imm_intN_t(build, c->data.i, bit_size);
         break;

      case nir_type_bool:
         cval = nir_imm_boolN_t(build, c->data.u, bit_size);
         break;

      default:
         unreachable("Invalid alu source type");
      }

      assert(cval->index ==
             util_dynarray_num_elements(state->states, uint16_t));
      util_dynarray_append(state->states, uint16_t, 0);
      nir_algebraic_automaton(cval->parent_instr, state->states,
                              state->pass_op_table);

      nir_alu_src val;
      val.src = nir_src_for_ssa(cval);
      val.negate = false;
      val.abs = false,
      memset(val.swizzle, 0, sizeof val.swizzle);

      return val;
   }

   default:
      unreachable("Invalid search value type");
   }
}

UNUSED static void dump_value(const nir_algebraic_table *table, const nir_search_value *val)
{
   switch (val->type) {
   case nir_search_value_constant: {
      const nir_search_constant *sconst = nir_search_value_as_constant(val);
      switch (sconst->type) {
      case nir_type_float:
         fprintf(stderr, "%f", sconst->data.d);
         break;
      case nir_type_int:
         fprintf(stderr, "%"PRId64, sconst->data.i);
         break;
      case nir_type_uint:
         fprintf(stderr, "0x%"PRIx64, sconst->data.u);
         break;
      case nir_type_bool:
         fprintf(stderr, "%s", sconst->data.u != 0 ? "True" : "False");
         break;
      default:
         unreachable("bad const type");
      }
      break;
   }

   case nir_search_value_variable: {
      const nir_search_variable *var = nir_search_value_as_variable(val);
      if (var->is_constant)
         fprintf(stderr, "#");
      fprintf(stderr, "%c", var->variable + 'a');
      break;
   }

   case nir_search_value_expression: {
      const nir_search_expression *expr = nir_search_value_as_expression(val);
      fprintf(stderr, "(");
      if (expr->inexact)
         fprintf(stderr, "~");
      switch (expr->opcode) {
#define CASE(n) \
      case nir_search_op_##n: fprintf(stderr, #n); break;
      CASE(f2b)
      CASE(b2f)
      CASE(b2i)
      CASE(i2i)
      CASE(f2i)
      CASE(i2f)
#undef CASE
      default:
         fprintf(stderr, "%s", nir_op_infos[expr->opcode].name);
      }

      unsigned num_srcs = 1;
      if (expr->opcode <= nir_last_opcode)
         num_srcs = nir_op_infos[expr->opcode].num_inputs;

      for (unsigned i = 0; i < num_srcs; i++) {
         fprintf(stderr, " ");
         dump_value(table, &table->values[expr->srcs[i]].value);
      }

      fprintf(stderr, ")");
      break;
   }
   }

   if (val->bit_size > 0)
      fprintf(stderr, "@%d", val->bit_size);
}

static void
add_uses_to_worklist(nir_instr *instr,
                     nir_instr_worklist *worklist,
                     struct util_dynarray *states,
                     const struct per_op_table *pass_op_table)
{
   nir_ssa_def *def = nir_instr_ssa_def(instr);

   nir_foreach_use_safe(use_src, def) {
      if (nir_algebraic_automaton(use_src->parent_instr, states, pass_op_table))
         nir_instr_worklist_push_tail(worklist, use_src->parent_instr);
   }
}

static void
nir_algebraic_update_automaton(nir_instr *new_instr,
                               nir_instr_worklist *algebraic_worklist,
                               struct util_dynarray *states,
                               const struct per_op_table *pass_op_table)
{

   nir_instr_worklist *automaton_worklist = nir_instr_worklist_create();

   /* Walk through the tree of uses of our new instruction's SSA value,
    * recursively updating the automaton state until it stabilizes.
    */
   add_uses_to_worklist(new_instr, automaton_worklist, states, pass_op_table);

   nir_instr *instr;
   while ((instr = nir_instr_worklist_pop_head(automaton_worklist))) {
      nir_instr_worklist_push_tail(algebraic_worklist, instr);
      add_uses_to_worklist(instr, automaton_worklist, states, pass_op_table);
   }

   nir_instr_worklist_destroy(automaton_worklist);
}

static nir_ssa_def *
nir_replace_instr(nir_builder *build, nir_alu_instr *instr,
                  struct hash_table *range_ht,
                  struct util_dynarray *states,
                  const nir_algebraic_table *table,
                  const nir_search_expression *search,
                  const nir_search_value *replace,
                  nir_instr_worklist *algebraic_worklist,
                  struct exec_list *dead_instrs)
{
   uint8_t swizzle[NIR_MAX_VEC_COMPONENTS] = { 0 };

   for (unsigned i = 0; i < instr->dest.dest.ssa.num_components; ++i)
      swizzle[i] = i;

   assert(instr->dest.dest.is_ssa);

   struct match_state state;
   state.inexact_match = false;
   state.has_exact_alu = false;
   state.range_ht = range_ht;
   state.pass_op_table = table->pass_op_table;
   state.table = table;

   STATIC_ASSERT(sizeof(state.comm_op_direction) * 8 >= NIR_SEARCH_MAX_COMM_OPS);

   unsigned comm_expr_combinations =
      1 << MIN2(search->comm_exprs, NIR_SEARCH_MAX_COMM_OPS);

   bool found = false;
   for (unsigned comb = 0; comb < comm_expr_combinations; comb++) {
      /* The bitfield of directions is just the current iteration.  Hooray for
       * binary.
       */
      state.comm_op_direction = comb;
      state.variables_seen = 0;

      if (match_expression(table, search, instr,
                           instr->dest.dest.ssa.num_components,
                           swizzle, &state)) {
         found = true;
         break;
      }
   }
   if (!found)
      return NULL;

#if 0
   fprintf(stderr, "matched: ");
   dump_value(&search->value);
   fprintf(stderr, " -> ");
   dump_value(replace);
   fprintf(stderr, " ssa_%d\n", instr->dest.dest.ssa.index);
#endif

   /* If the instruction at the root of the expression tree being replaced is
    * a unary operation, insert the replacement instructions at the location
    * of the source of the unary operation.  Otherwise, insert the replacement
    * instructions at the location of the expression tree root.
    *
    * For the unary operation case, this is done to prevent some spurious code
    * motion that can dramatically extend live ranges.  Imagine an expression
    * like -(A+B) where the addtion and the negation are separated by flow
    * control and thousands of instructions.  If this expression is replaced
    * with -A+-B, inserting the new instructions at the site of the negation
    * could extend the live range of A and B dramtically.  This could increase
    * register pressure and cause spilling.
    *
    * It may well be that moving instructions around is a good thing, but
    * keeping algebraic optimizations and code motion optimizations separate
    * seems safest.
    */
   nir_alu_instr *const src_instr = nir_src_as_alu_instr(instr->src[0].src);
   if (src_instr != NULL &&
       (instr->op == nir_op_fneg || instr->op == nir_op_fabs ||
        instr->op == nir_op_ineg || instr->op == nir_op_iabs ||
        instr->op == nir_op_inot)) {
      /* Insert new instructions *after*.  Otherwise a hypothetical
       * replacement fneg(X) -> fabs(X) would insert the fabs() instruction
       * before X!  This can also occur for things like fneg(X.wzyx) -> X.wzyx
       * in vector mode.  A move instruction to handle the swizzle will get
       * inserted before X.
       *
       * This manifested in a single OpenGL ES 2.0 CTS vertex shader test on
       * older Intel GPU that use vector-mode vertex processing.
       */
      build->cursor = nir_after_instr(&src_instr->instr);
   } else {
      build->cursor = nir_before_instr(&instr->instr);
   }

   state.states = states;

   nir_alu_src val = construct_value(build, replace,
                                     instr->dest.dest.ssa.num_components,
                                     instr->dest.dest.ssa.bit_size,
                                     &state, &instr->instr);

   /* Note that NIR builder will elide the MOV if it's a no-op, which may
    * allow more work to be done in a single pass through algebraic.
    */
   nir_ssa_def *ssa_val =
      nir_mov_alu(build, val, instr->dest.dest.ssa.num_components);
   if (ssa_val->index == util_dynarray_num_elements(states, uint16_t)) {
      util_dynarray_append(states, uint16_t, 0);
      nir_algebraic_automaton(ssa_val->parent_instr, states, table->pass_op_table);
   }

   /* Rewrite the uses of the old SSA value to the new one, and recurse
    * through the uses updating the automaton's state.
    */
   nir_ssa_def_rewrite_uses(&instr->dest.dest.ssa, ssa_val);
   nir_algebraic_update_automaton(ssa_val->parent_instr, algebraic_worklist,
                                  states, table->pass_op_table);

   /* Nothing uses the instr any more, so drop it out of the program.  Note
    * that the instr may be in the worklist still, so we can't free it
    * directly.
    */
   assert(instr->instr.pass_flags == 0);
   instr->instr.pass_flags = 1;
   nir_instr_remove(&instr->instr);
   exec_list_push_tail(dead_instrs, &instr->instr.node);

   return ssa_val;
}

static bool
nir_algebraic_automaton(nir_instr *instr, struct util_dynarray *states,
                        const struct per_op_table *pass_op_table)
{
   switch (instr->type) {
   case nir_instr_type_alu: {
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      nir_op op = alu->op;
      uint16_t search_op = nir_search_op_for_nir_op(op);
      const struct per_op_table *tbl = &pass_op_table[search_op];
      if (tbl->num_filtered_states == 0)
         return false;

      /* Calculate the index into the transition table. Note the index
       * calculated must match the iteration order of Python's
       * itertools.product(), which was used to emit the transition
       * table.
       */
      unsigned index = 0;
      for (unsigned i = 0; i < nir_op_infos[op].num_inputs; i++) {
         index *= tbl->num_filtered_states;
         if (tbl->filter)
            index += tbl->filter[*util_dynarray_element(states, uint16_t,
                                                        alu->src[i].src.ssa->index)];
      }

      uint16_t *state = util_dynarray_element(states, uint16_t,
                                              alu->dest.dest.ssa.index);
      if (*state != tbl->table[index]) {
         *state = tbl->table[index];
         return true;
      }
      return false;
   }

   case nir_instr_type_load_const: {
      nir_load_const_instr *load_const = nir_instr_as_load_const(instr);
      uint16_t *state = util_dynarray_element(states, uint16_t,
                                              load_const->def.index);
      if (*state != CONST_STATE) {
         *state = CONST_STATE;
         return true;
      }
      return false;
   }

   default:
      return false;
   }
}

static bool
nir_algebraic_instr(nir_builder *build, nir_instr *instr,
                    struct hash_table *range_ht,
                    const bool *condition_flags,
                    const nir_algebraic_table *table,
                    struct util_dynarray *states,
                    nir_instr_worklist *worklist,
                    struct exec_list *dead_instrs)
{

   if (instr->type != nir_instr_type_alu)
      return false;

   nir_alu_instr *alu = nir_instr_as_alu(instr);
   if (!alu->dest.dest.is_ssa)
      return false;

   unsigned bit_size = alu->dest.dest.ssa.bit_size;
   const unsigned execution_mode =
      build->shader->info.float_controls_execution_mode;
   const bool ignore_inexact =
      nir_is_float_control_signed_zero_inf_nan_preserve(execution_mode, bit_size) ||
      nir_is_denorm_flush_to_zero(execution_mode, bit_size);

   int xform_idx = *util_dynarray_element(states, uint16_t,
                                          alu->dest.dest.ssa.index);
   for (const struct transform *xform = &table->transforms[table->transform_offsets[xform_idx]];
        xform->condition_offset != ~0;
        xform++) {
      if (condition_flags[xform->condition_offset] &&
          !(table->values[xform->search].expression.inexact && ignore_inexact) &&
          nir_replace_instr(build, alu, range_ht, states, table,
                            &table->values[xform->search].expression,
                            &table->values[xform->replace].value, worklist, dead_instrs)) {
         _mesa_hash_table_clear(range_ht, NULL);
         return true;
      }
   }

   return false;
}

bool
nir_algebraic_impl(nir_function_impl *impl,
                   const bool *condition_flags,
                   const nir_algebraic_table *table)
{
   bool progress = false;

   nir_builder build;
   nir_builder_init(&build, impl);

   /* Note: it's important here that we're allocating a zeroed array, since
    * state 0 is the default state, which means we don't have to visit
    * anything other than constants and ALU instructions.
    */
   struct util_dynarray states = {0};
   if (!util_dynarray_resize(&states, uint16_t, impl->ssa_alloc)) {
      nir_metadata_preserve(impl, nir_metadata_all);
      return false;
   }
   memset(states.data, 0, states.size);

   struct hash_table *range_ht = _mesa_pointer_hash_table_create(NULL);

   nir_instr_worklist *worklist = nir_instr_worklist_create();

   /* Walk top-to-bottom setting up the automaton state. */
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         nir_algebraic_automaton(instr, &states, table->pass_op_table);
      }
   }

   /* Put our instrs in the worklist such that we're popping the last instr
    * first.  This will encourage us to match the biggest source patterns when
    * possible.
    */
   nir_foreach_block_reverse(block, impl) {
      nir_foreach_instr_reverse(instr, block) {
         instr->pass_flags = 0;
         if (instr->type == nir_instr_type_alu)
            nir_instr_worklist_push_tail(worklist, instr);
      }
   }

   struct exec_list dead_instrs;
   exec_list_make_empty(&dead_instrs);

   nir_instr *instr;
   while ((instr = nir_instr_worklist_pop_head(worklist))) {
      /* The worklist can have an instr pushed to it multiple times if it was
       * the src of multiple instrs that also got optimized, so make sure that
       * we don't try to re-optimize an instr we already handled.
       */
      if (instr->pass_flags)
         continue;

      progress |= nir_algebraic_instr(&build, instr,
                                      range_ht, condition_flags,
                                      table, &states, worklist, &dead_instrs);
   }

   nir_instr_free_list(&dead_instrs);

   nir_instr_worklist_destroy(worklist);
   ralloc_free(range_ht);
   util_dynarray_fini(&states);

   if (progress) {
      nir_metadata_preserve(impl, nir_metadata_block_index |
                                  nir_metadata_dominance);
   } else {
      nir_metadata_preserve(impl, nir_metadata_all);
   }

   return progress;
}

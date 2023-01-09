/*
 * Copyright © 2012 Intel Corporation
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

#ifndef IR_BUILDER_H
#define IR_BUILDER_H

#include "ir.h"

namespace ir_builder {

#ifndef WRITEMASK_X
enum writemask {
   WRITEMASK_X = 0x1,
   WRITEMASK_Y = 0x2,
   WRITEMASK_Z = 0x4,
   WRITEMASK_W = 0x8,
};
#endif

/**
 * This little class exists to let the helper expression generators
 * take either an ir_rvalue * or an ir_variable * to be automatically
 * dereferenced, while still providing compile-time type checking.
 *
 * You don't have to explicitly call the constructor -- C++ will see
 * that you passed an ir_variable, and silently call the
 * operand(ir_variable *var) constructor behind your back.
 */
class operand {
public:
   operand(ir_rvalue *val)
      : val(val)
   {
   }

   operand(ir_variable *var)
   {
      void *mem_ctx = ralloc_parent(var);
      val = new(mem_ctx) ir_dereference_variable(var);
   }

   ir_rvalue *val;
};

/** Automatic generator for ir_dereference_variable on assignment LHS.
 *
 * \sa operand
 */
class deref {
public:
   deref(ir_dereference *val)
      : val(val)
   {
   }

   deref(ir_variable *var)
   {
      void *mem_ctx = ralloc_parent(var);
      val = new(mem_ctx) ir_dereference_variable(var);
   }


   ir_dereference *val;
};

class ir_factory {
public:
   ir_factory(exec_list *instructions = NULL, void *mem_ctx = NULL)
      : instructions(instructions),
        mem_ctx(mem_ctx)
   {
      return;
   }

   void emit(ir_instruction *ir);
   ir_variable *make_temp(const glsl_type *type, const char *name);

   ir_constant*
   constant(float f)
   {
      return new(mem_ctx) ir_constant(f);
   }

   ir_constant*
   constant(int i)
   {
      return new(mem_ctx) ir_constant(i);
   }

   ir_constant*
   constant(unsigned u)
   {
      return new(mem_ctx) ir_constant(u);
   }

   ir_constant*
   constant(bool b)
   {
      return new(mem_ctx) ir_constant(b);
   }

   exec_list *instructions;
   void *mem_ctx;
};

ir_assignment *assign(deref lhs, operand rhs);
ir_assignment *assign(deref lhs, operand rhs, int writemask);

ir_return *ret(operand retval);

ir_expression *expr(ir_expression_operation op, operand a);
ir_expression *expr(ir_expression_operation op, operand a, operand b);
ir_expression *expr(ir_expression_operation op, operand a, operand b, operand c);
ir_expression *add(operand a, operand b);
ir_expression *sub(operand a, operand b);
ir_expression *mul(operand a, operand b);
ir_expression *imul_high(operand a, operand b);
ir_expression *div(operand a, operand b);
ir_expression *carry(operand a, operand b);
ir_expression *borrow(operand a, operand b);
ir_expression *trunc(operand a);
ir_expression *round_even(operand a);
ir_expression *fract(operand a);
ir_expression *dot(operand a, operand b);
ir_expression *clamp(operand a, operand b, operand c);
ir_expression *saturate(operand a);
ir_expression *abs(operand a);
ir_expression *neg(operand a);
ir_expression *sin(operand a);
ir_expression *cos(operand a);
ir_expression *exp(operand a);
ir_expression *rcp(operand a);
ir_expression *rsq(operand a);
ir_expression *sqrt(operand a);
ir_expression *log(operand a);
ir_expression *sign(operand a);

ir_expression *subr_to_int(operand a);
ir_expression *equal(operand a, operand b);
ir_expression *nequal(operand a, operand b);
ir_expression *less(operand a, operand b);
ir_expression *greater(operand a, operand b);
ir_expression *lequal(operand a, operand b);
ir_expression *gequal(operand a, operand b);

ir_expression *logic_not(operand a);
ir_expression *logic_and(operand a, operand b);
ir_expression *logic_or(operand a, operand b);

ir_expression *bit_not(operand a);
ir_expression *bit_or(operand a, operand b);
ir_expression *bit_and(operand a, operand b);
ir_expression *bit_xor(operand a, operand b);
ir_expression *lshift(operand a, operand b);
ir_expression *rshift(operand a, operand b);

ir_expression *f2i(operand a);
ir_expression *bitcast_f2i(operand a);
ir_expression *i2f(operand a);
ir_expression *bitcast_i2f(operand a);
ir_expression *f2u(operand a);
ir_expression *bitcast_f2u(operand a);
ir_expression *u2f(operand a);
ir_expression *bitcast_u2f(operand a);
ir_expression *i2u(operand a);
ir_expression *u2i(operand a);
ir_expression *b2i(operand a);
ir_expression *i2b(operand a);
ir_expression *f2b(operand a);
ir_expression *b2f(operand a);

ir_expression *f2d(operand a);
ir_expression *i2d(operand a);
ir_expression *u2d(operand a);

ir_expression *bitcast_d2i64(operand a);
ir_expression *bitcast_d2u64(operand a);

ir_expression *bitcast_i642d(operand a);
ir_expression *bitcast_u642d(operand a);

ir_expression *min2(operand a, operand b);
ir_expression *max2(operand a, operand b);

ir_expression *interpolate_at_centroid(operand a);
ir_expression *interpolate_at_offset(operand a, operand b);
ir_expression *interpolate_at_sample(operand a, operand b);

ir_expression *fma(operand a, operand b, operand c);
ir_expression *lrp(operand x, operand y, operand a);
ir_expression *csel(operand a, operand b, operand c);
ir_expression *bitfield_extract(operand a, operand b, operand c);
ir_expression *bitfield_insert(operand a, operand b, operand c, operand d);

ir_swizzle *swizzle(operand a, int swizzle, int components);
/**
 * Swizzle away later components, but preserve the ordering.
 */
ir_swizzle *swizzle_for_size(operand a, unsigned components);

ir_swizzle *swizzle_xxxx(operand a);
ir_swizzle *swizzle_yyyy(operand a);
ir_swizzle *swizzle_zzzz(operand a);
ir_swizzle *swizzle_wwww(operand a);
ir_swizzle *swizzle_x(operand a);
ir_swizzle *swizzle_y(operand a);
ir_swizzle *swizzle_z(operand a);
ir_swizzle *swizzle_w(operand a);
ir_swizzle *swizzle_xy(operand a);
ir_swizzle *swizzle_xyz(operand a);
ir_swizzle *swizzle_xyzw(operand a);

ir_if *if_tree(operand condition,
               ir_instruction *then_branch);
ir_if *if_tree(operand condition,
               ir_instruction *then_branch,
               ir_instruction *else_branch);

} /* namespace ir_builder */

#endif

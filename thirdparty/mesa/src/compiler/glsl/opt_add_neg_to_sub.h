/*
 * Copyright © 2016 Intel Corporation
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
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef OPT_ADD_NEG_TO_SUB_H
#define OPT_ADD_NEG_TO_SUB_H

#include "ir.h"
#include "ir_hierarchical_visitor.h"

class add_neg_to_sub_visitor : public ir_hierarchical_visitor {
public:
   add_neg_to_sub_visitor()
   {
      /* empty */
   }

   ir_visitor_status visit_leave(ir_expression *ir)
   {
      if (ir->operation != ir_binop_add)
         return visit_continue;

      for (unsigned i = 0; i < 2; i++) {
         ir_expression *const op = ir->operands[i]->as_expression();

         if (op != NULL && op->operation == ir_unop_neg) {
            ir->operation = ir_binop_sub;

            /* This ensures that -a + b becomes b - a. */
            if (i == 0)
               ir->operands[0] = ir->operands[1];

            ir->operands[1] = op->operands[0];
            break;
         }
      }

      return visit_continue;
   }
};

#endif /* OPT_ADD_NEG_TO_SUB_H */

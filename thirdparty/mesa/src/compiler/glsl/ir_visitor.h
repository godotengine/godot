/* -*- c++ -*- */
/*
 * Copyright © 2010 Intel Corporation
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

#ifndef IR_VISITOR_H
#define IR_VISITOR_H

#ifdef __cplusplus
/**
 * Abstract base class of visitors of IR instruction trees
 */
class ir_visitor {
public:
   virtual ~ir_visitor()
   {
      /* empty */
   }

   /**
    * \name Visit methods
    *
    * As typical for the visitor pattern, there must be one \c visit method for
    * each concrete subclass of \c ir_instruction.  Virtual base classes within
    * the hierarchy should not have \c visit methods.
    */
   /*@{*/
   virtual void visit(class ir_rvalue *) { assert(!"unhandled error_type"); }
   virtual void visit(class ir_variable *) = 0;
   virtual void visit(class ir_function_signature *) = 0;
   virtual void visit(class ir_function *) = 0;
   virtual void visit(class ir_expression *) = 0;
   virtual void visit(class ir_texture *) = 0;
   virtual void visit(class ir_swizzle *) = 0;
   virtual void visit(class ir_dereference_variable *) = 0;
   virtual void visit(class ir_dereference_array *) = 0;
   virtual void visit(class ir_dereference_record *) = 0;
   virtual void visit(class ir_assignment *) = 0;
   virtual void visit(class ir_constant *) = 0;
   virtual void visit(class ir_call *) = 0;
   virtual void visit(class ir_return *) = 0;
   virtual void visit(class ir_discard *) = 0;
   virtual void visit(class ir_demote *) = 0;
   virtual void visit(class ir_if *) = 0;
   virtual void visit(class ir_loop *) = 0;
   virtual void visit(class ir_loop_jump *) = 0;
   virtual void visit(class ir_emit_vertex *) = 0;
   virtual void visit(class ir_end_primitive *) = 0;
   virtual void visit(class ir_barrier *) = 0;
   /*@}*/
};

/* NOTE: function calls may never return due to discards inside them
 * This is usually not an issue, but if it is, keep it in mind
 */
class ir_control_flow_visitor : public ir_visitor {
public:
   virtual void visit(class ir_variable *) {}
   virtual void visit(class ir_expression *) {}
   virtual void visit(class ir_texture *) {}
   virtual void visit(class ir_swizzle *) {}
   virtual void visit(class ir_dereference_variable *) {}
   virtual void visit(class ir_dereference_array *) {}
   virtual void visit(class ir_dereference_record *) {}
   virtual void visit(class ir_assignment *) {}
   virtual void visit(class ir_constant *) {}
   virtual void visit(class ir_call *) {}
   virtual void visit(class ir_demote *) {}
   virtual void visit(class ir_emit_vertex *) {}
   virtual void visit(class ir_end_primitive *) {}
   virtual void visit(class ir_barrier *) {}
};
#endif /* __cplusplus */

#endif /* IR_VISITOR_H */

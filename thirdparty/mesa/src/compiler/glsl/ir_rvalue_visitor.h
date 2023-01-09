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

/**
 * \file ir_rvalue_visitor.h
 *
 * Generic class to implement the common pattern we have of wanting to
 * visit each ir_rvalue * and possibly change that node to a different
 * class.  Just implement handle_rvalue() and you will be called with
 * a pointer to each rvalue in the tree.
 */

#ifndef GLSL_IR_RVALUE_VISITOR_H
#define GLSL_IR_RVALUE_VISITOR_H

class ir_rvalue_base_visitor : public ir_hierarchical_visitor {
public:
   ir_visitor_status rvalue_visit(ir_assignment *);
   ir_visitor_status rvalue_visit(ir_call *);
   ir_visitor_status rvalue_visit(ir_dereference_array *);
   ir_visitor_status rvalue_visit(ir_dereference_record *);
   ir_visitor_status rvalue_visit(ir_discard *);
   ir_visitor_status rvalue_visit(ir_expression *);
   ir_visitor_status rvalue_visit(ir_if *);
   ir_visitor_status rvalue_visit(ir_return *);
   ir_visitor_status rvalue_visit(ir_swizzle *);
   ir_visitor_status rvalue_visit(ir_texture *);
   ir_visitor_status rvalue_visit(ir_emit_vertex *);
   ir_visitor_status rvalue_visit(ir_end_primitive *);

   virtual void handle_rvalue(ir_rvalue **rvalue) = 0;
};

class ir_rvalue_visitor : public ir_rvalue_base_visitor {
public:

   virtual ir_visitor_status visit_leave(ir_assignment *);
   virtual ir_visitor_status visit_leave(ir_call *);
   virtual ir_visitor_status visit_leave(ir_dereference_array *);
   virtual ir_visitor_status visit_leave(ir_dereference_record *);
   virtual ir_visitor_status visit_leave(ir_discard *);
   virtual ir_visitor_status visit_leave(ir_expression *);
   virtual ir_visitor_status visit_leave(ir_if *);
   virtual ir_visitor_status visit_leave(ir_return *);
   virtual ir_visitor_status visit_leave(ir_swizzle *);
   virtual ir_visitor_status visit_leave(ir_texture *);
   virtual ir_visitor_status visit_leave(ir_emit_vertex *);
   virtual ir_visitor_status visit_leave(ir_end_primitive *);
};

class ir_rvalue_enter_visitor : public ir_rvalue_base_visitor {
public:

   virtual ir_visitor_status visit_enter(ir_assignment *);
   virtual ir_visitor_status visit_enter(ir_call *);
   virtual ir_visitor_status visit_enter(ir_dereference_array *);
   virtual ir_visitor_status visit_enter(ir_dereference_record *);
   virtual ir_visitor_status visit_enter(ir_discard *);
   virtual ir_visitor_status visit_enter(ir_expression *);
   virtual ir_visitor_status visit_enter(ir_if *);
   virtual ir_visitor_status visit_enter(ir_return *);
   virtual ir_visitor_status visit_enter(ir_swizzle *);
   virtual ir_visitor_status visit_enter(ir_texture *);
   virtual ir_visitor_status visit_enter(ir_emit_vertex *);
   virtual ir_visitor_status visit_enter(ir_end_primitive *);
};

#endif /* GLSL_IR_RVALUE_VISITOR_H */

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
 * \file ir_variable_refcount.h
 *
 * Provides a visitor which produces a list of variables referenced,
 * how many times they were referenced and assigned, and whether they
 * were defined in the scope.
 */

#ifndef GLSL_IR_VARIABLE_REFCOUNT_H
#define GLSL_IR_VARIABLE_REFCOUNT_H

#include "ir.h"
#include "ir_visitor.h"
#include "compiler/glsl_types.h"

struct assignment_entry {
   exec_node link;
   ir_assignment *assign;
};

class ir_variable_refcount_entry
{
public:
   ir_variable_refcount_entry(ir_variable *var);

   ir_variable *var; /* The key: the variable's pointer. */

   /**
    * List of assignments to the variable, if any.
    * This is intended to be used for dead code optimisation and may
    * not be a complete list.
    */
   exec_list assign_list;

   /** Number of times the variable is referenced, including assignments. */
   unsigned referenced_count;

   /** Number of times the variable is assigned. */
   unsigned assigned_count;

   bool declaration; /* If the variable had a decl in the instruction stream */
};

class ir_variable_refcount_visitor : public ir_hierarchical_visitor {
public:
   ir_variable_refcount_visitor(void);
   ~ir_variable_refcount_visitor(void);

   virtual ir_visitor_status visit(ir_variable *);
   virtual ir_visitor_status visit(ir_dereference_variable *);

   virtual ir_visitor_status visit_enter(ir_function_signature *);
   virtual ir_visitor_status visit_leave(ir_assignment *);

   /**
    * Find variable in the hash table, and insert it if not present
    */
   ir_variable_refcount_entry *get_variable_entry(ir_variable *var);

   /**
    * Hash table mapping ir_variable to ir_variable_refcount_entry.
    */
   struct hash_table *ht;

   void *mem_ctx;
};

#endif /* GLSL_IR_VARIABLE_REFCOUNT_H */

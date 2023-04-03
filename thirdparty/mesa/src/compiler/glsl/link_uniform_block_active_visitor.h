/*
 * Copyright © 2013 Intel Corporation
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

#ifndef LINK_UNIFORM_BLOCK_ACTIVE_VISITOR_H
#define LINK_UNIFORM_BLOCK_ACTIVE_VISITOR_H

#include "ir.h"
#include "util/hash_table.h"

struct uniform_block_array_elements {
   unsigned *array_elements;
   unsigned num_array_elements;
   /**
    * Size of the array before array-trimming optimizations.
    *
    * Locations are only assigned to active array elements, but the location
    * values are calculated as if all elements are active. The total number
    * of elements in an array including the elements in arrays of arrays before
    * inactive elements are removed is needed to be perform that calculation.
    */
   unsigned aoa_size;

   ir_dereference_array *ir;

   struct uniform_block_array_elements *array;
};

struct link_uniform_block_active {
   const glsl_type *type;
   ir_variable *var;

   struct uniform_block_array_elements *array;

   unsigned binding;

   bool has_instance_name;
   bool has_binding;
   bool is_shader_storage;
};

class link_uniform_block_active_visitor : public ir_hierarchical_visitor {
public:
   link_uniform_block_active_visitor(void *mem_ctx, struct hash_table *ht,
				     struct gl_shader_program *prog)
      : success(true), prog(prog), ht(ht), mem_ctx(mem_ctx)
   {
      /* empty */
   }

   virtual ir_visitor_status visit_enter(ir_dereference_array *);
   virtual ir_visitor_status visit(ir_dereference_variable *);
   virtual ir_visitor_status visit(ir_variable *);

   bool success;

private:
   struct gl_shader_program *prog;
   struct hash_table *ht;
   void *mem_ctx;
};

#endif /* LINK_UNIFORM_BLOCK_ACTIVE_VISITOR_H */

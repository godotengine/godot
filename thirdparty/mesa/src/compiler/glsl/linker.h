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

#ifndef GLSL_LINKER_H
#define GLSL_LINKER_H

#include "linker_util.h"

struct gl_shader_program;
struct gl_shader;
struct gl_linked_shader;

extern bool
link_function_calls(gl_shader_program *prog, gl_linked_shader *main,
                    gl_shader **shader_list, unsigned num_shaders);

extern int
link_cross_validate_uniform_block(void *mem_ctx,
                                  struct gl_uniform_block **linked_blocks,
                                  unsigned int *num_linked_blocks,
                                  struct gl_uniform_block *new_block);

extern void
link_uniform_blocks(void *mem_ctx,
                    const struct gl_constants *consts,
                    struct gl_shader_program *prog,
                    struct gl_linked_shader *shader,
                    struct gl_uniform_block **ubo_blocks,
                    unsigned *num_ubo_blocks,
                    struct gl_uniform_block **ssbo_blocks,
                    unsigned *num_ssbo_blocks);

bool
validate_intrastage_arrays(struct gl_shader_program *prog,
                           ir_variable *const var,
                           ir_variable *const existing,
                           bool match_precision = true);

void
validate_intrastage_interface_blocks(struct gl_shader_program *prog,
                                     const gl_shader **shader_list,
                                     unsigned num_shaders);

void
validate_interstage_inout_blocks(struct gl_shader_program *prog,
                                 const gl_linked_shader *producer,
                                 const gl_linked_shader *consumer);

void
validate_interstage_uniform_blocks(struct gl_shader_program *prog,
                                   gl_linked_shader **stages);

extern struct gl_linked_shader *
link_intrastage_shaders(void *mem_ctx,
                        struct gl_context *ctx,
                        struct gl_shader_program *prog,
                        struct gl_shader **shader_list,
                        unsigned num_shaders,
                        bool allow_missing_main);

extern unsigned
link_calculate_matrix_stride(const glsl_type *matrix, bool row_major,
                             enum glsl_interface_packing packing);

/**
 * Class for processing all of the leaf fields of a variable that corresponds
 * to a program resource.
 *
 * The leaf fields are all the parts of the variable that the application
 * could query using \c glGetProgramResourceIndex (or that could be returned
 * by \c glGetProgramResourceName).
 *
 * Classes my derive from this class to implement specific functionality.
 * This class only provides the mechanism to iterate over the leaves.  Derived
 * classes must implement \c ::visit_field and may override \c ::process.
 */
class program_resource_visitor {
public:
   /**
    * Begin processing a variable
    *
    * Classes that overload this function should call \c ::process from the
    * base class to start the recursive processing of the variable.
    *
    * \param var  The variable that is to be processed
    *
    * Calls \c ::visit_field for each leaf of the variable.
    *
    * \warning
    * When processing a uniform block, this entry should only be used in cases
    * where the row / column ordering of matrices in the block does not
    * matter.  For example, enumerating the names of members of the block, but
    * not for determining the offsets of members.
    */
   void process(ir_variable *var, bool use_std430_as_default);

   /**
    * Begin processing a variable
    *
    * Classes that overload this function should call \c ::process from the
    * base class to start the recursive processing of the variable.
    *
    * \param var  The variable that is to be processed
    * \param var_type The glsl_type reference of the variable
    *
    * Calls \c ::visit_field for each leaf of the variable.
    *
    * \warning
    * When processing a uniform block, this entry should only be used in cases
    * where the row / column ordering of matrices in the block does not
    * matter.  For example, enumerating the names of members of the block, but
    * not for determining the offsets of members.
    */
   void process(ir_variable *var, const glsl_type *var_type,
                bool use_std430_as_default);

   /**
    * Begin processing a variable of a structured type.
    *
    * This flavor of \c process should be used to handle structured types
    * (i.e., structures, interfaces, or arrays there of) that need special
    * name handling.  A common usage is to handle cases where the block name
    * (instead of the instance name) is used for an interface block.
    *
    * \param type  Type that is to be processed, associated with \c name
    * \param name  Base name of the structured variable being processed
    *
    * \note
    * \c type must be \c GLSL_TYPE_RECORD, \c GLSL_TYPE_INTERFACE, or an array
    * there of.
    */
   void process(const glsl_type *type, const char *name,
                bool use_std430_as_default);

protected:
   /**
    * Method invoked for each leaf of the variable
    *
    * \param type  Type of the field.
    * \param name  Fully qualified name of the field.
    * \param row_major  For a matrix type, is it stored row-major.
    * \param record_type  Type of the record containing the field.
    * \param last_field   Set if \c name is the last field of the structure
    *                     containing it.  This will always be false for items
    *                     not contained in a structure or interface block.
    */
   virtual void visit_field(const glsl_type *type, const char *name,
                            bool row_major, const glsl_type *record_type,
                            const enum glsl_interface_packing packing,
                            bool last_field) = 0;

   virtual void enter_record(const glsl_type *type, const char *name,
                             bool row_major, const enum glsl_interface_packing packing);

   virtual void leave_record(const glsl_type *type, const char *name,
                             bool row_major, const enum glsl_interface_packing packing);

   virtual void set_buffer_offset(unsigned offset);

   virtual void set_record_array_count(unsigned record_array_count);

private:
   /**
    * \param name_length  Length of the current name \b not including the
    *                     terminating \c NUL character.
    * \param last_field   Set if \c name is the last field of the structure
    *                     containing it.  This will always be false for items
    *                     not contained in a structure or interface block.
    */
   void recursion(const glsl_type *t, char **name, size_t name_length,
                  bool row_major, const glsl_type *record_type,
                  const enum glsl_interface_packing packing,
                  bool last_field, unsigned record_array_count,
                  const glsl_struct_field *named_ifc_member);
};

#endif /* GLSL_LINKER_H */

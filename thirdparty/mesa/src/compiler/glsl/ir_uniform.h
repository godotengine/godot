/*
 * Copyright © 2011 Intel Corporation
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

#ifndef IR_UNIFORM_H
#define IR_UNIFORM_H


/* stdbool.h is necessary because this file is included in both C and C++ code.
 */
#include <stdbool.h>
#include "util/macros.h"
#include "program/prog_parameter.h"  /* For union gl_constant_value. */

/**
 * Used by GL_ARB_explicit_uniform_location extension code in the linker
 * and glUniform* functions to identify inactive explicit uniform locations.
 */
#define INACTIVE_UNIFORM_EXPLICIT_LOCATION ((gl_uniform_storage *) -1)

#ifdef __cplusplus
extern "C" {
#endif

enum PACKED gl_uniform_driver_format {
   uniform_native = 0,          /**< Store data in the native format. */
   uniform_int_float,           /**< Store integer data as floats. */
};

struct gl_uniform_driver_storage {
   /**
    * Number of bytes from one array element to the next.
    */
   uint8_t element_stride;

   /**
    * Number of bytes from one vector in a matrix to the next.
    */
   uint8_t vector_stride;

   /**
    * Base format of the stored data.
    */
   enum gl_uniform_driver_format format;

   /**
    * Pointer to the base of the data.
    */
   void *data;
};

struct gl_opaque_uniform_index {
   /**
    * Base opaque uniform index
    *
    * If \c gl_uniform_storage::base_type is an opaque type, this
    * represents its uniform index.  If \c
    * gl_uniform_storage::array_elements is not zero, the array will
    * use opaque uniform indices \c index through \c index + \c
    * gl_uniform_storage::array_elements - 1, inclusive.
    *
    * Note that the index may be different in each shader stage.
    */
   uint8_t index;

   /**
    * Whether this opaque uniform is used in this shader stage.
    */
   bool active;
};

struct gl_resource_name
{
   char *string;
   int length;              /* strlen(string) or 0 */
   int last_square_bracket; /* (strrchr(name, '[') - name) or -1 */
   bool suffix_is_zero_square_bracketed; /* suffix is [0] */
};

struct gl_uniform_storage {
   struct gl_resource_name name;

   /** Type of this uniform data stored.
    *
    * In the case of an array, it's the type of a single array element.
    */
   const struct glsl_type *type;

   /**
    * The number of elements in this uniform.
    *
    * For non-arrays, this is always 0.  For arrays, the value is the size of
    * the array.
    */
   unsigned array_elements;

   struct gl_opaque_uniform_index opaque[MESA_SHADER_STAGES];

   /**
    * Mask of shader stages (1 << MESA_SHADER_xxx) where this uniform is used.
    */
   unsigned active_shader_mask;

   /**
    * Storage used by the driver for the uniform
    */
   unsigned num_driver_storage;
   struct gl_uniform_driver_storage *driver_storage;

   /**
    * Storage used by Mesa for the uniform
    *
    * This form of the uniform is used by Mesa's implementation of \c
    * glGetUniform.  It can also be used by drivers to obtain the value of the
    * uniform if the \c ::driver_storage interface is not used.
    */
   union gl_constant_value *storage;

   /** Fields for GL_ARB_uniform_buffer_object
    * @{
    */

   /**
    * GL_UNIFORM_BLOCK_INDEX: index of the uniform block containing
    * the uniform, or -1 for the default uniform block.  Note that the
    * index is into the linked program's UniformBlocks[] array, not
    * the linked shader's.
    */
   int block_index;

   /** GL_UNIFORM_OFFSET: byte offset within the uniform block, or -1. */
   int offset;

   /**
    * GL_UNIFORM_MATRIX_STRIDE: byte stride between columns or rows of
    * a matrix.  Set to 0 for non-matrices in UBOs, or -1 for uniforms
    * in the default uniform block.
    */
   int matrix_stride;

   /**
    * GL_UNIFORM_ARRAY_STRIDE: byte stride between elements of the
    * array.  Set to zero for non-arrays in UBOs, or -1 for uniforms
    * in the default uniform block.
    */
   int array_stride;

   /** GL_UNIFORM_ROW_MAJOR: true iff it's a row-major matrix in a UBO */
   bool row_major;

   /** @} */

   /**
    * This is a compiler-generated uniform that should not be advertised
    * via the API.
    */
   bool hidden;

   /**
    * This is a built-in uniform that should not be modified through any gl API.
    */
   bool builtin;

   /**
    * This is a shader storage buffer variable, not an uniform.
    */
   bool is_shader_storage;

   /**
    * Index within gl_shader_program::AtomicBuffers[] of the atomic
    * counter buffer this uniform is stored in, or -1 if this is not
    * an atomic counter.
    */
   int atomic_buffer_index;

   /**
    * The 'base location' for this uniform in the uniform remap table. For
    * arrays this is the first element in the array.
    * for subroutines this is in shader subroutine uniform remap table.
    */
   unsigned remap_location;

   /**
    * The number of compatible subroutines with this subroutine uniform.
    */
   unsigned num_compatible_subroutines;

   /**
    * A single integer identifying the number of active array elements of
    * the top-level shader storage block member (GL_TOP_LEVEL_ARRAY_SIZE).
    */
   unsigned top_level_array_size;

   /**
    * A single integer identifying the stride between array elements of the
    * top-level shader storage block member. (GL_TOP_LEVEL_ARRAY_STRIDE).
    */
   unsigned top_level_array_stride;

   /**
    * Whether this uniform variable has the bindless_sampler or bindless_image
    * layout qualifier as specified by ARB_bindless_texture.
    */
   bool is_bindless;
};

void
resource_name_updated(struct gl_resource_name *name);

#ifdef __cplusplus
}
#endif

#endif /* IR_UNIFORM_H */

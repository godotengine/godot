/* -*- c++ -*- */
/*
 * Copyright © 2009 Intel Corporation
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

#ifndef GLSL_TYPES_H
#define GLSL_TYPES_H

#include <string.h>
#include <assert.h>
#include <stdio.h>

#include "shader_enums.h"
#include "c11/threads.h"
#include "util/blob.h"
#include "util/format/u_format.h"
#include "util/macros.h"
#include "util/simple_mtx.h"

#ifdef __cplusplus
#include "mesa/main/config.h"
#include "mesa/main/menums.h" /* for gl_texture_index, C++'s enum rules are broken */
#include "util/glheader.h"
#include "util/ralloc.h"
#endif

struct glsl_type;

#ifdef __cplusplus
extern "C" {
#endif

struct _mesa_glsl_parse_state;
struct glsl_symbol_table;

extern void
glsl_type_singleton_init_or_ref();

extern void
glsl_type_singleton_decref();

extern void
_mesa_glsl_initialize_types(struct _mesa_glsl_parse_state *state);

void
glsl_print_type(FILE *f, const struct glsl_type *t);

void encode_type_to_blob(struct blob *blob, const struct glsl_type *type);

const struct glsl_type *decode_type_from_blob(struct blob_reader *blob);

typedef void (*glsl_type_size_align_func)(const struct glsl_type *type,
                                          unsigned *size, unsigned *align);

enum glsl_base_type {
   /* Note: GLSL_TYPE_UINT, GLSL_TYPE_INT, and GLSL_TYPE_FLOAT must be 0, 1,
    * and 2 so that they will fit in the 2 bits of glsl_type::sampled_type.
    */
   GLSL_TYPE_UINT = 0,
   GLSL_TYPE_INT,
   GLSL_TYPE_FLOAT,
   GLSL_TYPE_FLOAT16,
   GLSL_TYPE_DOUBLE,
   GLSL_TYPE_UINT8,
   GLSL_TYPE_INT8,
   GLSL_TYPE_UINT16,
   GLSL_TYPE_INT16,
   GLSL_TYPE_UINT64,
   GLSL_TYPE_INT64,
   GLSL_TYPE_BOOL,
   GLSL_TYPE_SAMPLER,
   GLSL_TYPE_TEXTURE,
   GLSL_TYPE_IMAGE,
   GLSL_TYPE_ATOMIC_UINT,
   GLSL_TYPE_STRUCT,
   GLSL_TYPE_INTERFACE,
   GLSL_TYPE_ARRAY,
   GLSL_TYPE_VOID,
   GLSL_TYPE_SUBROUTINE,
   GLSL_TYPE_FUNCTION,
   GLSL_TYPE_ERROR
};

/* Return the bit size of a type. Note that this differs from 
 * glsl_get_bit_size in that it returns 32 bits for bools, whereas at
 * the NIR level we would want to return 1 bit for bools.
 */
static unsigned glsl_base_type_bit_size(enum glsl_base_type type)
{
   switch (type) {
   case GLSL_TYPE_BOOL:
   case GLSL_TYPE_INT:
   case GLSL_TYPE_UINT:
   case GLSL_TYPE_FLOAT: /* TODO handle mediump */
   case GLSL_TYPE_SUBROUTINE:
      return 32;

   case GLSL_TYPE_FLOAT16:
   case GLSL_TYPE_UINT16:
   case GLSL_TYPE_INT16:
      return 16;

   case GLSL_TYPE_UINT8:
   case GLSL_TYPE_INT8:
      return 8;

   case GLSL_TYPE_DOUBLE:
   case GLSL_TYPE_INT64:
   case GLSL_TYPE_UINT64:
   case GLSL_TYPE_IMAGE:
   case GLSL_TYPE_TEXTURE:
   case GLSL_TYPE_SAMPLER:
      return 64;

   default:
      /* For GLSL_TYPE_STRUCT etc, it should be ok to return 0. This usually
       * happens when calling this method through is_64bit and is_16bit
       * methods
       */
      return 0;
   }

   return 0;
}

static inline bool glsl_base_type_is_16bit(enum glsl_base_type type)
{
   return glsl_base_type_bit_size(type) == 16;
}

static inline bool glsl_base_type_is_64bit(enum glsl_base_type type)
{
   return glsl_base_type_bit_size(type) == 64;
}

static inline bool glsl_base_type_is_integer(enum glsl_base_type type)
{
   return type == GLSL_TYPE_UINT8 ||
          type == GLSL_TYPE_INT8 ||
          type == GLSL_TYPE_UINT16 ||
          type == GLSL_TYPE_INT16 ||
          type == GLSL_TYPE_UINT ||
          type == GLSL_TYPE_INT ||
          type == GLSL_TYPE_UINT64 ||
          type == GLSL_TYPE_INT64 ||
          type == GLSL_TYPE_BOOL ||
          type == GLSL_TYPE_SAMPLER ||
          type == GLSL_TYPE_TEXTURE ||
          type == GLSL_TYPE_IMAGE;
}

static inline unsigned int
glsl_base_type_get_bit_size(const enum glsl_base_type base_type)
{
   switch (base_type) {
   case GLSL_TYPE_BOOL:
      return 1;

   case GLSL_TYPE_INT:
   case GLSL_TYPE_UINT:
   case GLSL_TYPE_FLOAT: /* TODO handle mediump */
   case GLSL_TYPE_SUBROUTINE:
      return 32;

   case GLSL_TYPE_FLOAT16:
   case GLSL_TYPE_UINT16:
   case GLSL_TYPE_INT16:
      return 16;

   case GLSL_TYPE_UINT8:
   case GLSL_TYPE_INT8:
      return 8;

   case GLSL_TYPE_DOUBLE:
   case GLSL_TYPE_INT64:
   case GLSL_TYPE_UINT64:
   case GLSL_TYPE_IMAGE:
   case GLSL_TYPE_SAMPLER:
   case GLSL_TYPE_TEXTURE:
      return 64;

   default:
      unreachable("unknown base type");
   }

   return 0;
}

static inline enum glsl_base_type
glsl_unsigned_base_type_of(enum glsl_base_type type)
{
   switch (type) {
   case GLSL_TYPE_INT:
      return GLSL_TYPE_UINT;
   case GLSL_TYPE_INT8:
      return GLSL_TYPE_UINT8;
   case GLSL_TYPE_INT16:
      return GLSL_TYPE_UINT16;
   case GLSL_TYPE_INT64:
      return GLSL_TYPE_UINT64;
   default:
      assert(type == GLSL_TYPE_UINT ||
             type == GLSL_TYPE_UINT8 ||
             type == GLSL_TYPE_UINT16 ||
             type == GLSL_TYPE_UINT64);
      return type;
   }
}

static inline enum glsl_base_type
glsl_signed_base_type_of(enum glsl_base_type type)
{
   switch (type) {
   case GLSL_TYPE_UINT:
      return GLSL_TYPE_INT;
   case GLSL_TYPE_UINT8:
      return GLSL_TYPE_INT8;
   case GLSL_TYPE_UINT16:
      return GLSL_TYPE_INT16;
   case GLSL_TYPE_UINT64:
      return GLSL_TYPE_INT64;
   default:
      assert(type == GLSL_TYPE_INT ||
             type == GLSL_TYPE_INT8 ||
             type == GLSL_TYPE_INT16 ||
             type == GLSL_TYPE_INT64);
      return type;
   }
}

enum glsl_sampler_dim {
   GLSL_SAMPLER_DIM_1D = 0,
   GLSL_SAMPLER_DIM_2D,
   GLSL_SAMPLER_DIM_3D,
   GLSL_SAMPLER_DIM_CUBE,
   GLSL_SAMPLER_DIM_RECT,
   GLSL_SAMPLER_DIM_BUF,
   GLSL_SAMPLER_DIM_EXTERNAL,
   GLSL_SAMPLER_DIM_MS,
   GLSL_SAMPLER_DIM_SUBPASS, /* for vulkan input attachments */
   GLSL_SAMPLER_DIM_SUBPASS_MS, /* for multisampled vulkan input attachments */
};

int
glsl_get_sampler_dim_coordinate_components(enum glsl_sampler_dim dim);

enum glsl_matrix_layout {
   /**
    * The layout of the matrix is inherited from the object containing the
    * matrix (the top level structure or the uniform block).
    */
   GLSL_MATRIX_LAYOUT_INHERITED,

   /**
    * Explicit column-major layout
    *
    * If a uniform block doesn't have an explicit layout set, it will default
    * to this layout.
    */
   GLSL_MATRIX_LAYOUT_COLUMN_MAJOR,

   /**
    * Row-major layout
    */
   GLSL_MATRIX_LAYOUT_ROW_MAJOR
};

enum {
   GLSL_PRECISION_NONE = 0,
   GLSL_PRECISION_HIGH,
   GLSL_PRECISION_MEDIUM,
   GLSL_PRECISION_LOW
};

#ifdef __cplusplus
} /* extern "C" */
#endif

/* C++ struct types for glsl */
#ifdef __cplusplus

struct glsl_type {
   GLenum gl_type;
   glsl_base_type base_type:8;

   glsl_base_type sampled_type:8; /**< Type of data returned using this
                                   * sampler or image.  Only \c
                                   * GLSL_TYPE_FLOAT, \c GLSL_TYPE_INT,
                                   * and \c GLSL_TYPE_UINT are valid.
                                   */

   unsigned sampler_dimensionality:4; /**< \see glsl_sampler_dim */
   unsigned sampler_shadow:1;
   unsigned sampler_array:1;
   unsigned interface_packing:2;
   unsigned interface_row_major:1;

   /**
    * For \c GLSL_TYPE_STRUCT this specifies if the struct is packed or not.
    *
    * Only used for Compute kernels
    */
   unsigned packed:1;

private:
   glsl_type() : mem_ctx(NULL)
   {
      // Dummy constructor, just for the sake of ASSERT_BITFIELD_SIZE.
   }

public:
   /**
    * \name Vector and matrix element counts
    *
    * For scalars, each of these values will be 1.  For non-numeric types
    * these will be 0.
    */
   /*@{*/
   uint8_t vector_elements;    /**< 1, 2, 3, or 4 vector elements. */
   uint8_t matrix_columns;     /**< 1, 2, 3, or 4 matrix columns. */
   /*@}*/

   /**
    * For \c GLSL_TYPE_ARRAY, this is the length of the array.  For
    * \c GLSL_TYPE_STRUCT or \c GLSL_TYPE_INTERFACE, it is the number of
    * elements in the structure and the number of values pointed to by
    * \c fields.structure (below).
    */
   unsigned length;

   /**
    * Name of the data type
    *
    * Will never be \c NULL.
    */
   const char *name;

   /**
    * Explicit array, matrix, or vector stride.  This is used to communicate
    * explicit array layouts from SPIR-V.  Should be 0 if the type has no
    * explicit stride.
    */
   unsigned explicit_stride;

   /**
    * Explicit alignment. This is used to communicate explicit alignment
    * constraints. Should be 0 if the type has no explicit alignment
    * constraint.
    */
   unsigned explicit_alignment;

   /**
    * Subtype of composite data types.
    */
   union {
      const struct glsl_type *array;            /**< Type of array elements. */
      struct glsl_function_param *parameters;   /**< Parameters to function. */
      struct glsl_struct_field *structure;      /**< List of struct fields. */
   } fields;

   /**
    * \name Pointers to various public type singletons
    */
   /*@{*/
#undef  DECL_TYPE
#define DECL_TYPE(NAME, ...) \
   static const glsl_type *const NAME##_type;
#undef  STRUCT_TYPE
#define STRUCT_TYPE(NAME) \
   static const glsl_type *const struct_##NAME##_type;
#include "compiler/builtin_type_macros.h"
   /*@}*/

   /**
    * Convenience accessors for vector types (shorter than get_instance()).
    * @{
    */
   static const glsl_type *vec(unsigned components, const glsl_type *const ts[]);
   static const glsl_type *vec(unsigned components);
   static const glsl_type *f16vec(unsigned components);
   static const glsl_type *dvec(unsigned components);
   static const glsl_type *ivec(unsigned components);
   static const glsl_type *uvec(unsigned components);
   static const glsl_type *bvec(unsigned components);
   static const glsl_type *i64vec(unsigned components);
   static const glsl_type *u64vec(unsigned components);
   static const glsl_type *i16vec(unsigned components);
   static const glsl_type *u16vec(unsigned components);
   static const glsl_type *i8vec(unsigned components);
   static const glsl_type *u8vec(unsigned components);
   /**@}*/

   /**
    * For numeric and boolean derived types returns the basic scalar type
    *
    * If the type is a numeric or boolean scalar, vector, or matrix type,
    * this function gets the scalar type of the individual components.  For
    * all other types, including arrays of numeric or boolean types, the
    * error type is returned.
    */
   const glsl_type *get_base_type() const;

   /**
    * Get the basic scalar type which this type aggregates.
    *
    * If the type is a numeric or boolean scalar, vector, or matrix, or an
    * array of any of those, this function gets the scalar type of the
    * individual components.  For structs and arrays of structs, this function
    * returns the struct type.  For samplers and arrays of samplers, this
    * function returns the sampler type.
    */
   const glsl_type *get_scalar_type() const;

   /**
    * Gets the "bare" type without any decorations or layout information.
    */
   const glsl_type *get_bare_type() const;

   /**
    * Gets the float16 version of this type.
    */
   const glsl_type *get_float16_type() const;

   /**
    * Gets the int16 version of this type.
    */
   const glsl_type *get_int16_type() const;

   /**
    * Gets the uint16 version of this type.
    */
   const glsl_type *get_uint16_type() const;

   /**
    * Get the instance of a built-in scalar, vector, or matrix type
    */
   static const glsl_type *get_instance(unsigned base_type, unsigned rows,
                                        unsigned columns,
                                        unsigned explicit_stride = 0,
                                        bool row_major = false,
                                        unsigned explicit_alignment = 0);

   /**
    * Get the instance of a sampler type
    */
   static const glsl_type *get_sampler_instance(enum glsl_sampler_dim dim,
                                                bool shadow,
                                                bool array,
                                                glsl_base_type type);

   static const glsl_type *get_texture_instance(enum glsl_sampler_dim dim,
                                                bool array,
                                                glsl_base_type type);

   static const glsl_type *get_image_instance(enum glsl_sampler_dim dim,
                                              bool array, glsl_base_type type);

   /**
    * Get the instance of an array type
    */
   static const glsl_type *get_array_instance(const glsl_type *base,
                                              unsigned elements,
                                              unsigned explicit_stride = 0);

   /**
    * Get the instance of a record type
    */
   static const glsl_type *get_struct_instance(const glsl_struct_field *fields,
					       unsigned num_fields,
					       const char *name,
					       bool packed = false,
					       unsigned explicit_alignment = 0);

   /**
    * Get the instance of an interface block type
    */
   static const glsl_type *get_interface_instance(const glsl_struct_field *fields,
						  unsigned num_fields,
						  enum glsl_interface_packing packing,
						  bool row_major,
						  const char *block_name);

   /**
    * Get the instance of an subroutine type
    */
   static const glsl_type *get_subroutine_instance(const char *subroutine_name);

   /**
    * Get the instance of a function type
    */
   static const glsl_type *get_function_instance(const struct glsl_type *return_type,
                                                 const glsl_function_param *parameters,
                                                 unsigned num_params);

   /**
    * Get the type resulting from a multiplication of \p type_a * \p type_b
    */
   static const glsl_type *get_mul_type(const glsl_type *type_a,
                                        const glsl_type *type_b);

   /**
    * Query the total number of scalars that make up a scalar, vector or matrix
    */
   unsigned components() const
   {
      return vector_elements * matrix_columns;
   }

   /**
    * Calculate the number of components slots required to hold this type
    *
    * This is used to determine how many uniform or varying locations a type
    * might occupy.
    */
   unsigned component_slots() const;

   unsigned component_slots_aligned(unsigned offset) const;

   /**
    * Calculate offset between the base location of the struct in
    * uniform storage and a struct member.
    * For the initial call, length is the index of the member to find the
    * offset for.
    */
   unsigned struct_location_offset(unsigned length) const;

   /**
    * Calculate the number of unique values from glGetUniformLocation for the
    * elements of the type.
    *
    * This is used to allocate slots in the UniformRemapTable, the amount of
    * locations may not match with actual used storage space by the driver.
    */
   unsigned uniform_locations() const;

   /**
    * Used to count the number of varyings contained in the type ignoring
    * innermost array elements.
    */
   unsigned varying_count() const;

   /**
    * Calculate the number of vec4 slots required to hold this type.
    *
    * This is the underlying recursive type_size function for
    * count_attribute_slots() (vertex inputs and varyings) but also for
    * gallium's !PIPE_CAP_PACKED_UNIFORMS case.
    */
   unsigned count_vec4_slots(bool is_gl_vertex_input, bool bindless) const;

   /**
    * Calculate the number of vec4 slots required to hold this type.
    *
    * This is the underlying recursive type_size function for
    * gallium's PIPE_CAP_PACKED_UNIFORMS case.
    */
   unsigned count_dword_slots(bool bindless) const;

   /**
    * Calculate the number of attribute slots required to hold this type
    *
    * This implements the language rules of GLSL 1.50 for counting the number
    * of slots used by a vertex attribute.  It also determines the number of
    * varying slots the type will use up in the absence of varying packing
    * (and thus, it can be used to measure the number of varying slots used by
    * the varyings that are generated by lower_packed_varyings).
    *
    * For vertex shader attributes - doubles only take one slot.
    * For inter-shader varyings - dvec3/dvec4 take two slots.
    *
    * Vulkan doesn’t make this distinction so the argument should always be
    * false.
    */
   unsigned count_attribute_slots(bool is_gl_vertex_input) const {
      return count_vec4_slots(is_gl_vertex_input, true);
   }

   /**
    * Alignment in bytes of the start of this type in a std140 uniform
    * block.
    */
   unsigned std140_base_alignment(bool row_major) const;

   /** Size in bytes of this type in a std140 uniform block.
    *
    * Note that this is not GL_UNIFORM_SIZE (which is the number of
    * elements in the array)
    */
   unsigned std140_size(bool row_major) const;

   /**
    * Gets an explicitly laid out type with the std140 layout.
    */
   const glsl_type *get_explicit_std140_type(bool row_major) const;

   /**
    * Alignment in bytes of the start of this type in a std430 shader
    * storage block.
    */
   unsigned std430_base_alignment(bool row_major) const;

   /**
    * Calculate array stride in bytes of this type in a std430 shader storage
    * block.
    */
   unsigned std430_array_stride(bool row_major) const;

   /**
    * Size in bytes of this type in a std430 shader storage block.
    *
    * Note that this is not GL_BUFFER_SIZE
    */
   unsigned std430_size(bool row_major) const;

   /**
    * Gets an explicitly laid out type with the std430 layout.
    */
   const glsl_type *get_explicit_std430_type(bool row_major) const;

   /**
    * Gets an explicitly laid out interface type.
    */
   const glsl_type *get_explicit_interface_type(bool supports_std430) const;

   /** Returns an explicitly laid out type given a type and size/align func
    *
    * The size/align func is only called for scalar and vector types and the
    * returned type is otherwise laid out in the natural way as follows:
    *
    *  - Arrays and matrices have a stride of ALIGN(elem_size, elem_align).
    *
    *  - Structure types have their elements in-order and as tightly packed as
    *    possible following the alignment required by the size/align func.
    *
    *  - All composite types (structures, matrices, and arrays) have an
    *    alignment equal to the highest alignment of any member of the composite.
    *
    * The types returned by this function are likely not suitable for most UBO
    * or SSBO layout because they do not add the extra array and substructure
    * alignment that is required by std140 and std430.
    */
   const glsl_type *get_explicit_type_for_size_align(glsl_type_size_align_func type_info,
                                                     unsigned *size, unsigned *align) const;

   const glsl_type *replace_vec3_with_vec4() const;

   /**
    * Alignment in bytes of the start of this type in OpenCL memory.
    */
   unsigned cl_alignment() const;

   /**
    * Size in bytes of this type in OpenCL memory
    */
   unsigned cl_size() const;

   /**
    * Size in bytes of this type based on its explicit data.
    *
    * When using SPIR-V shaders (ARB_gl_spirv), memory layouts are expressed
    * through explicit offset, stride and matrix layout, so the size
    * can/should be computed used those values.
    *
    * Note that the value returned by this method is only correct if such
    * values are set, so only with SPIR-V shaders. Should not be used with
    * GLSL shaders.
    */
   unsigned explicit_size(bool align_to_stride=false) const;

   /**
    * \brief Can this type be implicitly converted to another?
    *
    * \return True if the types are identical or if this type can be converted
    *         to \c desired according to Section 4.1.10 of the GLSL spec.
    *
    * \verbatim
    * From page 25 (31 of the pdf) of the GLSL 1.50 spec, Section 4.1.10
    * Implicit Conversions:
    *
    *     In some situations, an expression and its type will be implicitly
    *     converted to a different type. The following table shows all allowed
    *     implicit conversions:
    *
    *     Type of expression | Can be implicitly converted to
    *     --------------------------------------------------
    *     int                  float
    *     uint
    *
    *     ivec2                vec2
    *     uvec2
    *
    *     ivec3                vec3
    *     uvec3
    *
    *     ivec4                vec4
    *     uvec4
    *
    *     There are no implicit array or structure conversions. For example,
    *     an array of int cannot be implicitly converted to an array of float.
    *     There are no implicit conversions between signed and unsigned
    *     integers.
    * \endverbatim
    */
   bool can_implicitly_convert_to(const glsl_type *desired,
                                  _mesa_glsl_parse_state *state) const;

   /**
    * Query whether or not a type is a scalar (non-vector and non-matrix).
    */
   bool is_scalar() const
   {
      return (vector_elements == 1)
	 && (base_type >= GLSL_TYPE_UINT)
	 && (base_type <= GLSL_TYPE_IMAGE);
   }

   /**
    * Query whether or not a type is a vector
    */
   bool is_vector() const
   {
      return (vector_elements > 1)
	 && (matrix_columns == 1)
	 && (base_type >= GLSL_TYPE_UINT)
	 && (base_type <= GLSL_TYPE_BOOL);
   }

   /**
    * Query whether or not a type is a matrix
    */
   bool is_matrix() const
   {
      /* GLSL only has float matrices. */
      return (matrix_columns > 1) && (base_type == GLSL_TYPE_FLOAT ||
                                      base_type == GLSL_TYPE_DOUBLE ||
                                      base_type == GLSL_TYPE_FLOAT16);
   }

   /**
    * Query whether or not a type is a non-array numeric type
    */
   bool is_numeric() const
   {
      return (base_type >= GLSL_TYPE_UINT) && (base_type <= GLSL_TYPE_INT64);
   }

   /**
    * Query whether or not a type is an integer.
    */
   bool is_integer() const
   {
      return glsl_base_type_is_integer(base_type);
   }

   /**
    * Query whether or not a type is a 16-bit integer.
    */
   bool is_integer_16() const
   {
      return base_type == GLSL_TYPE_UINT16 || base_type == GLSL_TYPE_INT16;
   }

   /**
    * Query whether or not a type is an 32-bit integer.
    */
   bool is_integer_32() const
   {
      return (base_type == GLSL_TYPE_UINT) || (base_type == GLSL_TYPE_INT);
   }

   /**
    * Query whether or not a type is a 64-bit integer.
    */
   bool is_integer_64() const
   {
      return base_type == GLSL_TYPE_UINT64 || base_type == GLSL_TYPE_INT64;
   }

   /**
    * Query whether or not a type is a 32-bit or 64-bit integer
    */
   bool is_integer_32_64() const
   {
      return is_integer_32() || is_integer_64();
   }

   /**
    * Query whether or not a type is a 16-bit or 32-bit integer
    */
   bool is_integer_16_32() const
   {
      return is_integer_16() || is_integer_32();
   }

   /**
    * Query whether or not a type is a 16-bit, 32-bit or 64-bit integer
    */
   bool is_integer_16_32_64() const
   {
      return is_integer_16() || is_integer_32() || is_integer_64();
   }

   /**
    * Query whether or not type is an integral type, or for struct and array
    * types, contains an integral type.
    */
   bool contains_integer() const;

   /**
    * Query whether or not type is a double type, or for struct, interface and
    * array types, contains a double type.
    */
   bool contains_double() const;

   /**
    * Query whether or not type is a 64-bit type, or for struct, interface and
    * array types, contains a double type.
    */
   bool contains_64bit() const;

   /**
    * Query whether or not a type is a float type
    */
   bool is_float() const
   {
      return base_type == GLSL_TYPE_FLOAT;
   }

   /**
    * Query whether or not a type is a half-float or float type
    */
   bool is_float_16_32() const
   {
      return base_type == GLSL_TYPE_FLOAT16 || is_float();
   }

   /**
    * Query whether or not a type is a half-float, float or double
    */
   bool is_float_16_32_64() const
   {
      return base_type == GLSL_TYPE_FLOAT16 || is_float() || is_double();
   }

   /**
    * Query whether or not a type is a float or double
    */
   bool is_float_32_64() const
   {
      return is_float() || is_double();
   }

   bool is_int_16_32_64() const
   {
      return base_type == GLSL_TYPE_INT16 ||
             base_type == GLSL_TYPE_INT ||
             base_type == GLSL_TYPE_INT64;
   }

   bool is_uint_16_32_64() const
   {
      return base_type == GLSL_TYPE_UINT16 ||
             base_type == GLSL_TYPE_UINT ||
             base_type == GLSL_TYPE_UINT64;
   }

   bool is_int_16_32() const
   {
      return base_type == GLSL_TYPE_INT ||
             base_type == GLSL_TYPE_INT16;
   }

   bool is_uint_16_32() const
   {
      return base_type == GLSL_TYPE_UINT ||
             base_type == GLSL_TYPE_UINT16;
   }

   /**
    * Query whether or not a type is a double type
    */
   bool is_double() const
   {
      return base_type == GLSL_TYPE_DOUBLE;
   }

   /**
    * Query whether a 64-bit type takes two slots.
    */
   bool is_dual_slot() const
   {
      return is_64bit() && vector_elements > 2;
   }

   /**
    * Query whether or not a type is 64-bit
    */
   bool is_64bit() const
   {
      return glsl_base_type_is_64bit(base_type);
   }

   /**
    * Query whether or not a type is 16-bit
    */
   bool is_16bit() const
   {
      return glsl_base_type_is_16bit(base_type);
   }

   /**
    * Query whether or not a type is 32-bit
    */
   bool is_32bit() const
   {
      return base_type == GLSL_TYPE_UINT ||
             base_type == GLSL_TYPE_INT ||
             base_type == GLSL_TYPE_FLOAT;
   }

   /**
    * Query whether or not a type is a non-array boolean type
    */
   bool is_boolean() const
   {
      return base_type == GLSL_TYPE_BOOL;
   }

   /**
    * Query whether or not a type is a sampler
    */
   bool is_sampler() const
   {
      return base_type == GLSL_TYPE_SAMPLER;
   }

   /**
    * Query whether or not a type is a texture
    */
   bool is_texture() const
   {
      return base_type == GLSL_TYPE_TEXTURE;
   }

   /**
    * Query whether or not type is a sampler, or for struct, interface and
    * array types, contains a sampler.
    */
   bool contains_sampler() const;

   /**
    * Query whether or not type is an array or for struct, interface and
    * array types, contains an array.
    */
   bool contains_array() const;

   /**
    * Get the Mesa texture target index for a sampler type.
    */
   gl_texture_index sampler_index() const;

   /**
    * Query whether or not type is an image, or for struct, interface and
    * array types, contains an image.
    */
   bool contains_image() const;

   /**
    * Query whether or not a type is an image
    */
   bool is_image() const
   {
      return base_type == GLSL_TYPE_IMAGE;
   }

   /**
    * Query whether or not a type is an array
    */
   bool is_array() const
   {
      return base_type == GLSL_TYPE_ARRAY;
   }

   bool is_array_of_arrays() const
   {
      return is_array() && fields.array->is_array();
   }

   /**
    * Query whether or not a type is a record
    */
   bool is_struct() const
   {
      return base_type == GLSL_TYPE_STRUCT;
   }

   /**
    * Query whether or not a type is an interface
    */
   bool is_interface() const
   {
      return base_type == GLSL_TYPE_INTERFACE;
   }

   /**
    * Query whether or not a type is the void type singleton.
    */
   bool is_void() const
   {
      return base_type == GLSL_TYPE_VOID;
   }

   /**
    * Query whether or not a type is the error type singleton.
    */
   bool is_error() const
   {
      return base_type == GLSL_TYPE_ERROR;
   }

   /**
    * Query if a type is unnamed/anonymous (named by the parser)
    */

   bool is_subroutine() const
   {
      return base_type == GLSL_TYPE_SUBROUTINE;
   }
   bool contains_subroutine() const;

   bool is_anonymous() const
   {
      return !strncmp(name, "#anon", 5);
   }

   /**
    * Get the type stripped of any arrays
    *
    * \return
    * Pointer to the type of elements of the first non-array type for array
    * types, or pointer to itself for non-array types.
    */
   const glsl_type *without_array() const
   {
      const glsl_type *t = this;

      while (t->is_array())
         t = t->fields.array;

      return t;
   }

   /**
    * Return the total number of elements in an array including the elements
    * in arrays of arrays.
    */
   unsigned arrays_of_arrays_size() const
   {
      if (!is_array())
         return 0;

      unsigned size = length;
      const glsl_type *array_base_type = fields.array;

      while (array_base_type->is_array()) {
         size = size * array_base_type->length;
         array_base_type = array_base_type->fields.array;
      }
      return size;
   }

   /**
    * Return bit size for this type.
    */
   unsigned bit_size() const
   {
      return glsl_base_type_bit_size(this->base_type);
   }


   /**
    * Query whether or not a type is an atomic_uint.
    */
   bool is_atomic_uint() const
   {
      return base_type == GLSL_TYPE_ATOMIC_UINT;
   }

   /**
    * Return the amount of atomic counter storage required for a type.
    */
   unsigned atomic_size() const
   {
      if (is_atomic_uint())
         return ATOMIC_COUNTER_SIZE;
      else if (is_array())
         return length * fields.array->atomic_size();
      else
         return 0;
   }

   /**
    * Return whether a type contains any atomic counters.
    */
   bool contains_atomic() const
   {
      return atomic_size() > 0;
   }

   /**
    * Return whether a type contains any opaque types.
    */
   bool contains_opaque() const;

   /**
    * Query the full type of a matrix row
    *
    * \return
    * If the type is not a matrix, \c glsl_type::error_type is returned.
    * Otherwise a type matching the rows of the matrix is returned.
    */
   const glsl_type *row_type() const
   {
      if (!is_matrix())
         return error_type;

      if (explicit_stride && !interface_row_major)
         return get_instance(base_type, matrix_columns, 1, explicit_stride);
      else
         return get_instance(base_type, matrix_columns, 1);
   }

   /**
    * Query the full type of a matrix column
    *
    * \return
    * If the type is not a matrix, \c glsl_type::error_type is returned.
    * Otherwise a type matching the columns of the matrix is returned.
    */
   const glsl_type *column_type() const
   {
      if (!is_matrix())
         return error_type;

      if (interface_row_major) {
         /* If we're row-major, the vector element stride is the same as the
          * matrix stride and we have no alignment (i.e. component-aligned).
          */
         return get_instance(base_type, vector_elements, 1,
                             explicit_stride, false, 0);
      } else {
         /* Otherwise, the vector is tightly packed (stride=0).  For
          * alignment, we treat a matrix as an array of columns make the same
          * assumption that the alignment of the column is the same as the
          * alignment of the whole matrix.
          */
         return get_instance(base_type, vector_elements, 1,
                             0, false, explicit_alignment);
      }
   }

   /**
    * Get the type of a structure field
    *
    * \return
    * Pointer to the type of the named field.  If the type is not a structure
    * or the named field does not exist, \c glsl_type::error_type is returned.
    */
   const glsl_type *field_type(const char *name) const;

   /**
    * Get the location of a field within a record type
    */
   int field_index(const char *name) const;

   /**
    * Query the number of elements in an array type
    *
    * \return
    * The number of elements in the array for array types or -1 for non-array
    * types.  If the number of elements in the array has not yet been declared,
    * zero is returned.
    */
   int array_size() const
   {
      return is_array() ? length : -1;
   }

   /**
    * Query whether the array size for all dimensions has been declared.
    */
   bool is_unsized_array() const
   {
      return is_array() && length == 0;
   }

   /**
    * Return the number of coordinate components needed for this
    * sampler or image type.
    *
    * This is based purely on the sampler's dimensionality.  For example, this
    * returns 1 for sampler1D, and 3 for sampler2DArray.
    *
    * Note that this is often different than actual coordinate type used in
    * a texturing built-in function, since those pack additional values (such
    * as the shadow comparator or projector) into the coordinate type.
    */
   int coordinate_components() const;

   /**
    * Compares whether this type matches another type without taking into
    * account the precision in structures.
    *
    * This is applied recursively so that structures containing structure
    * members can also ignore the precision.
    */
   bool compare_no_precision(const glsl_type *b) const;

   /**
    * Compare a record type against another record type.
    *
    * This is useful for matching record types declared on the same shader
    * stage as well as across different shader stages.
    * The option to not match name is needed for matching record types
    * declared across different shader stages.
    * The option to not match locations is to deal with places where the
    * same struct is defined in a block which has a location set on it.
    */
   bool record_compare(const glsl_type *b, bool match_name,
                       bool match_locations = true,
                       bool match_precision = true) const;

   /**
    * Get the type interface packing.
    */
   enum glsl_interface_packing get_interface_packing() const
   {
      return (enum glsl_interface_packing)interface_packing;
   }

   /**
    * Get the type interface packing used internally. For shared and packing
    * layouts this is implementation defined.
    */
   enum glsl_interface_packing get_internal_ifc_packing(bool std430_supported) const
   {
      enum glsl_interface_packing packing = this->get_interface_packing();
      if (packing == GLSL_INTERFACE_PACKING_STD140 ||
          (!std430_supported &&
           (packing == GLSL_INTERFACE_PACKING_SHARED ||
            packing == GLSL_INTERFACE_PACKING_PACKED))) {
         return GLSL_INTERFACE_PACKING_STD140;
      } else {
         assert(packing == GLSL_INTERFACE_PACKING_STD430 ||
                (std430_supported &&
                 (packing == GLSL_INTERFACE_PACKING_SHARED ||
                  packing == GLSL_INTERFACE_PACKING_PACKED)));
         return GLSL_INTERFACE_PACKING_STD430;
      }
   }

   /**
    * Check if the type interface is row major
    */
   bool get_interface_row_major() const
   {
      return (bool) interface_row_major;
   }

   ~glsl_type();

private:

   static simple_mtx_t hash_mutex;

   /**
    * ralloc context for the type itself.
    */
   void *mem_ctx;

   /** Constructor for vector and matrix types */
   glsl_type(GLenum gl_type,
             glsl_base_type base_type, unsigned vector_elements,
             unsigned matrix_columns, const char *name,
             unsigned explicit_stride = 0, bool row_major = false,
             unsigned explicit_alignment = 0);

   /** Constructor for sampler or image types */
   glsl_type(GLenum gl_type, glsl_base_type base_type,
	     enum glsl_sampler_dim dim, bool shadow, bool array,
	     glsl_base_type type, const char *name);

   /** Constructor for record types */
   glsl_type(const glsl_struct_field *fields, unsigned num_fields,
	     const char *name, bool packed = false,
	     unsigned explicit_alignment = 0);

   /** Constructor for interface types */
   glsl_type(const glsl_struct_field *fields, unsigned num_fields,
	     enum glsl_interface_packing packing,
	     bool row_major, const char *name);

   /** Constructor for interface types */
   glsl_type(const glsl_type *return_type,
             const glsl_function_param *params, unsigned num_params);

   /** Constructors for array types */
   glsl_type(const glsl_type *array, unsigned length, unsigned explicit_stride);

   /** Constructor for subroutine types */
   glsl_type(const char *name);

   /** Hash table containing the known explicit matrix and vector types. */
   static struct hash_table *explicit_matrix_types;

   /** Hash table containing the known array types. */
   static struct hash_table *array_types;

   /** Hash table containing the known struct types. */
   static struct hash_table *struct_types;

   /** Hash table containing the known interface types. */
   static struct hash_table *interface_types;

   /** Hash table containing the known subroutine types. */
   static struct hash_table *subroutine_types;

   /** Hash table containing the known function types. */
   static struct hash_table *function_types;

   static bool record_key_compare(const void *a, const void *b);
   static unsigned record_key_hash(const void *key);

   /**
    * \name Built-in type flyweights
    */
   /*@{*/
#undef  DECL_TYPE
#define DECL_TYPE(NAME, ...) static const glsl_type _##NAME##_type;
#undef  STRUCT_TYPE
#define STRUCT_TYPE(NAME)        static const glsl_type _struct_##NAME##_type;
#include "compiler/builtin_type_macros.h"
   /*@}*/

   /**
    * \name Friend functions.
    *
    * These functions are friends because they must have C linkage and the
    * need to call various private methods or access various private static
    * data.
    */
   /*@{*/
   friend void glsl_type_singleton_init_or_ref(void);
   friend void glsl_type_singleton_decref(void);
   friend void _mesa_glsl_initialize_types(struct _mesa_glsl_parse_state *);
   /*@}*/
};

#undef DECL_TYPE
#undef STRUCT_TYPE
#endif /* __cplusplus */

struct glsl_struct_field {
   const struct glsl_type *type;
   const char *name;

   /**
    * For interface blocks, gl_varying_slot corresponding to the input/output
    * if this is a built-in input/output (i.e. a member of the built-in
    * gl_PerVertex interface block); -1 otherwise.
    *
    * Ignored for structs.
    */
   int location;

   /**
    * For interface blocks, members may explicitly assign the component used
    * by a varying. Ignored for structs.
    */
   int component;

   /**
    * For interface blocks, members may have an explicit byte offset
    * specified; -1 otherwise. Also used for xfb_offset layout qualifier.
    *
    * Unless used for xfb_offset this field is ignored for structs.
    */
   int offset;

   /**
    * For interface blocks, members may define a transform feedback buffer;
    * -1 otherwise.
    */
   int xfb_buffer;

   /**
    * For interface blocks, members may define a transform feedback stride;
    * -1 otherwise.
    */
   int xfb_stride;

   /**
    * Layout format, applicable to image variables only.
    */
   enum pipe_format image_format;

   union {
      struct {
         /**
          * For interface blocks, the interpolation mode (as in
          * ir_variable::interpolation).  0 otherwise.
          */
         unsigned interpolation:3;

         /**
          * For interface blocks, 1 if this variable uses centroid interpolation (as
          * in ir_variable::centroid).  0 otherwise.
          */
         unsigned centroid:1;

         /**
          * For interface blocks, 1 if this variable uses sample interpolation (as
          * in ir_variable::sample). 0 otherwise.
          */
         unsigned sample:1;

         /**
          * Layout of the matrix.  Uses glsl_matrix_layout values.
          */
         unsigned matrix_layout:2;

         /**
          * For interface blocks, 1 if this variable is a per-patch input or output
          * (as in ir_variable::patch). 0 otherwise.
          */
         unsigned patch:1;

         /**
          * Precision qualifier
          */
         unsigned precision:2;

         /**
          * Memory qualifiers, applicable to buffer variables defined in shader
          * storage buffer objects (SSBOs)
          */
         unsigned memory_read_only:1;
         unsigned memory_write_only:1;
         unsigned memory_coherent:1;
         unsigned memory_volatile:1;
         unsigned memory_restrict:1;

         /**
          * Any of the xfb_* qualifiers trigger the shader to be in transform
          * feedback mode so we need to keep track of whether the buffer was
          * explicitly set or if its just been assigned the default global value.
          */
         unsigned explicit_xfb_buffer:1;

         unsigned implicit_sized_array:1;
      };
      unsigned flags;
   };
#ifdef __cplusplus
#define DEFAULT_CONSTRUCTORS(_type, _name)                  \
   type(_type), name(_name), location(-1), component(-1), offset(-1), \
   xfb_buffer(0),  xfb_stride(0), image_format(PIPE_FORMAT_NONE), flags(0) \

   glsl_struct_field(const struct glsl_type *_type,
                     int _precision,
                     const char *_name)
      : DEFAULT_CONSTRUCTORS(_type, _name)
   {
      matrix_layout = GLSL_MATRIX_LAYOUT_INHERITED;
      precision = _precision;
   }

   glsl_struct_field(const struct glsl_type *_type, const char *_name)
      : DEFAULT_CONSTRUCTORS(_type, _name)
   {
      matrix_layout = GLSL_MATRIX_LAYOUT_INHERITED;
      precision = GLSL_PRECISION_NONE;
   }

   glsl_struct_field()
      : DEFAULT_CONSTRUCTORS(NULL, NULL)
   {
      matrix_layout = GLSL_MATRIX_LAYOUT_INHERITED;
      precision = GLSL_PRECISION_NONE;
   }
#undef DEFAULT_CONSTRUCTORS
#endif
};

struct glsl_function_param {
   const struct glsl_type *type;

   bool in;
   bool out;
};

static inline unsigned int
glsl_align(unsigned int a, unsigned int align)
{
   return (a + align - 1) / align * align;
}

#endif /* GLSL_TYPES_H */

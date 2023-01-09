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
 *    Connor Abbott (cwabbott0@gmail.com)
 *
 */

#include "nir_types.h"
#include "nir_gl_types.h"
#include "compiler/glsl/ir.h"

const char *
glsl_get_type_name(const glsl_type *type)
{
   return type->name;
}

int
glsl_array_size(const struct glsl_type *type)
{
   return type->array_size();
}

const glsl_type *
glsl_get_array_element(const glsl_type* type)
{
   if (type->is_matrix())
      return type->column_type();
   else if (type->is_vector())
      return type->get_scalar_type();
   return type->fields.array;
}

const glsl_type *
glsl_without_array(const glsl_type *type)
{
   return type->without_array();
}

const glsl_type *
glsl_without_array_or_matrix(const glsl_type *type)
{
   type = type->without_array();
   if (type->is_matrix())
      type = type->column_type();
   return type;
}

const glsl_type *
glsl_get_bare_type(const glsl_type *type)
{
   return type->get_bare_type();
}

const glsl_type *
glsl_get_struct_field(const glsl_type *type, unsigned index)
{
   assert(type->is_struct() || type->is_interface());
   assert(index < type->length);
   return type->fields.structure[index].type;
}

int
glsl_get_struct_field_offset(const struct glsl_type *type,
                             unsigned index)
{
   return type->fields.structure[index].offset;
}

const struct glsl_struct_field *
glsl_get_struct_field_data(const struct glsl_type *type, unsigned index)
{
   assert(type->is_struct() || type->is_interface());
   assert(index < type->length);
   return &type->fields.structure[index];
}

unsigned
glsl_get_explicit_stride(const struct glsl_type *type)
{
   return type->explicit_stride;
}

const glsl_type *
glsl_get_function_return_type(const glsl_type *type)
{
   return type->fields.parameters[0].type;
}

const glsl_function_param *
glsl_get_function_param(const glsl_type *type, unsigned index)
{
   return &type->fields.parameters[index + 1];
}

const glsl_type *
glsl_texture_type_to_sampler(const glsl_type *type, bool is_shadow)
{
   assert(glsl_type_is_texture(type));
   return glsl_sampler_type((glsl_sampler_dim)type->sampler_dimensionality,
                            is_shadow, type->sampler_array,
                            (glsl_base_type)type->sampled_type);
}

const glsl_type *
glsl_sampler_type_to_texture(const glsl_type *type)
{
   assert(glsl_type_is_sampler(type) && !glsl_type_is_bare_sampler(type));
   return glsl_texture_type((glsl_sampler_dim)type->sampler_dimensionality,
                            type->sampler_array,
                            (glsl_base_type)type->sampled_type);
}

const struct glsl_type *
glsl_get_column_type(const struct glsl_type *type)
{
   return type->column_type();
}

GLenum
glsl_get_gl_type(const struct glsl_type *type)
{
   return type->gl_type;
}

enum glsl_base_type
glsl_get_base_type(const struct glsl_type *type)
{
   return type->base_type;
}

unsigned
glsl_get_vector_elements(const struct glsl_type *type)
{
   return type->vector_elements;
}

unsigned
glsl_get_components(const struct glsl_type *type)
{
   return type->components();
}

unsigned
glsl_get_matrix_columns(const struct glsl_type *type)
{
   return type->matrix_columns;
}

unsigned
glsl_get_length(const struct glsl_type *type)
{
   return type->is_matrix() ? type->matrix_columns : type->length;
}

unsigned
glsl_get_aoa_size(const struct glsl_type *type)
{
   return type->arrays_of_arrays_size();
}

unsigned
glsl_count_vec4_slots(const struct glsl_type *type,
                      bool is_gl_vertex_input, bool is_bindless)
{
   return type->count_vec4_slots(is_gl_vertex_input, is_bindless);
}

unsigned
glsl_count_dword_slots(const struct glsl_type *type, bool is_bindless)
{
   return type->count_dword_slots(is_bindless);
}

unsigned
glsl_count_attribute_slots(const struct glsl_type *type,
                           bool is_gl_vertex_input)
{
   return type->count_attribute_slots(is_gl_vertex_input);
}

unsigned
glsl_get_component_slots(const struct glsl_type *type)
{
   return type->component_slots();
}

unsigned
glsl_get_component_slots_aligned(const struct glsl_type *type, unsigned offset)
{
   return type->component_slots_aligned(offset);
}

unsigned
glsl_varying_count(const struct glsl_type *type)
{
   return type->varying_count();
}

const char *
glsl_get_struct_elem_name(const struct glsl_type *type, unsigned index)
{
   return type->fields.structure[index].name;
}

glsl_sampler_dim
glsl_get_sampler_dim(const struct glsl_type *type)
{
   assert(glsl_type_is_sampler(type) ||
          glsl_type_is_texture(type) ||
          glsl_type_is_image(type));
   return (glsl_sampler_dim)type->sampler_dimensionality;
}

glsl_base_type
glsl_get_sampler_result_type(const struct glsl_type *type)
{
   assert(glsl_type_is_sampler(type) ||
          glsl_type_is_texture(type) ||
          glsl_type_is_image(type));
   return (glsl_base_type)type->sampled_type;
}

unsigned
glsl_get_sampler_target(const struct glsl_type *type)
{
   assert(glsl_type_is_sampler(type));
   return type->sampler_index();
}

int
glsl_get_sampler_coordinate_components(const struct glsl_type *type)
{
   assert(glsl_type_is_sampler(type) ||
          glsl_type_is_texture(type) ||
          glsl_type_is_image(type));
   return type->coordinate_components();
}

unsigned
glsl_get_struct_location_offset(const struct glsl_type *type,
                                unsigned length)
{
   return type->struct_location_offset(length);
}

bool
glsl_type_is_16bit(const glsl_type *type)
{
   return type->is_16bit();
}

bool
glsl_type_is_32bit(const glsl_type *type)
{
   return type->is_32bit();
}

bool
glsl_type_is_64bit(const glsl_type *type)
{
   return type->is_64bit();
}

bool
glsl_type_is_void(const glsl_type *type)
{
   return type->is_void();
}

bool
glsl_type_is_error(const glsl_type *type)
{
   return type->is_error();
}

bool
glsl_type_is_vector(const struct glsl_type *type)
{
   return type->is_vector();
}

bool
glsl_type_is_scalar(const struct glsl_type *type)
{
   return type->is_scalar();
}

bool
glsl_type_is_vector_or_scalar(const struct glsl_type *type)
{
   return type->is_vector() || type->is_scalar();
}

bool
glsl_type_is_matrix(const struct glsl_type *type)
{
   return type->is_matrix();
}

bool
glsl_matrix_type_is_row_major(const struct glsl_type *type)
{
   assert((type->is_matrix() && type->explicit_stride) || type->is_interface());
   return type->interface_row_major;
}

bool
glsl_type_is_array(const struct glsl_type *type)
{
   return type->is_array();
}

bool
glsl_type_is_unsized_array(const struct glsl_type *type)
{
   return type->is_unsized_array();
}

bool
glsl_type_is_array_of_arrays(const struct glsl_type *type)
{
   return type->is_array_of_arrays();
}

bool
glsl_type_is_array_or_matrix(const struct glsl_type *type)
{
   return type->is_array() || type->is_matrix();
}

bool
glsl_type_is_struct(const struct glsl_type *type)
{
   return type->is_struct();
}

bool
glsl_type_is_interface(const struct glsl_type *type)
{
   return type->is_interface();
}

bool
glsl_type_is_struct_or_ifc(const struct glsl_type *type)
{
   return type->is_struct() || type->is_interface();
}

bool
glsl_type_is_sampler(const struct glsl_type *type)
{
   return type->is_sampler();
}

bool
glsl_type_is_bare_sampler(const struct glsl_type *type)
{
   return type->is_sampler() && type->sampled_type == GLSL_TYPE_VOID;
}

bool
glsl_type_is_texture(const struct glsl_type *type)
{
   return type->is_texture();
}

bool
glsl_type_is_image(const struct glsl_type *type)
{
   return type->is_image();
}

bool
glsl_sampler_type_is_shadow(const struct glsl_type *type)
{
   assert(glsl_type_is_sampler(type));
   return type->sampler_shadow;
}

bool
glsl_sampler_type_is_array(const struct glsl_type *type)
{
   assert(glsl_type_is_sampler(type) ||
          glsl_type_is_texture(type) ||
          glsl_type_is_image(type));
   return type->sampler_array;
}

bool
glsl_struct_type_is_packed(const struct glsl_type *type)
{
   assert(glsl_type_is_struct(type));
   return type->packed;
}

bool
glsl_type_is_dual_slot(const struct glsl_type *type)
{
   return type->is_dual_slot();
}

bool
glsl_type_is_numeric(const struct glsl_type *type)
{
   return type->is_numeric();
}

bool
glsl_type_is_boolean(const struct glsl_type *type)
{
   return type->is_boolean();
}
bool
glsl_type_is_integer(const struct glsl_type *type)
{
   return type->is_integer();
}

bool
glsl_type_contains_64bit(const struct glsl_type *type)
{
   return type->contains_64bit();
}

bool
glsl_type_contains_image(const struct glsl_type *type)
{
   return type->contains_image();
}

bool
glsl_contains_double(const struct glsl_type *type)
{
   return type->contains_double();
}

bool
glsl_contains_integer(const struct glsl_type *type)
{
   return type->contains_integer();
}

bool
glsl_record_compare(const struct glsl_type *a, const struct glsl_type *b,
                    bool match_name, bool match_locations, bool match_precision)
{
   return a->record_compare(b, match_name, match_locations, match_precision);
}

const glsl_type *
glsl_void_type(void)
{
   return glsl_type::void_type;
}

const glsl_type *
glsl_float_type(void)
{
   return glsl_type::float_type;
}

const glsl_type *
glsl_double_type(void)
{
   return glsl_type::double_type;
}

const glsl_type *
glsl_float16_t_type(void)
{
   return glsl_type::float16_t_type;
}

const glsl_type *
glsl_floatN_t_type(unsigned bit_size)
{
   switch (bit_size) {
   case 16: return glsl_type::float16_t_type;
   case 32: return glsl_type::float_type;
   case 64: return glsl_type::double_type;
   default:
      unreachable("Unsupported bit size");
   }
}

const glsl_type *
glsl_vec_type(unsigned n)
{
   return glsl_type::vec(n);
}

const glsl_type *
glsl_dvec_type(unsigned n)
{
   return glsl_type::dvec(n);
}

const glsl_type *
glsl_vec4_type(void)
{
   return glsl_type::vec4_type;
}

const glsl_type *
glsl_uvec4_type(void)
{
   return glsl_type::uvec4_type;
}

const glsl_type *
glsl_ivec4_type(void)
{
   return glsl_type::ivec4_type;
}

const glsl_type *
glsl_int_type(void)
{
   return glsl_type::int_type;
}

const glsl_type *
glsl_uint_type(void)
{
   return glsl_type::uint_type;
}

const glsl_type *
glsl_int64_t_type(void)
{
   return glsl_type::int64_t_type;
}

const glsl_type *
glsl_uint64_t_type(void)
{
   return glsl_type::uint64_t_type;
}

const glsl_type *
glsl_int16_t_type(void)
{
   return glsl_type::int16_t_type;
}

const glsl_type *
glsl_uint16_t_type(void)
{
   return glsl_type::uint16_t_type;
}

const glsl_type *
glsl_int8_t_type(void)
{
   return glsl_type::int8_t_type;
}

const glsl_type *
glsl_uint8_t_type(void)
{
   return glsl_type::uint8_t_type;
}

const glsl_type *
glsl_intN_t_type(unsigned bit_size)
{
   switch (bit_size) {
   case 8:  return glsl_type::int8_t_type;
   case 16: return glsl_type::int16_t_type;
   case 32: return glsl_type::int_type;
   case 64: return glsl_type::int64_t_type;
   default:
      unreachable("Unsupported bit size");
   }
}

const glsl_type *
glsl_uintN_t_type(unsigned bit_size)
{
   switch (bit_size) {
   case 8:  return glsl_type::uint8_t_type;
   case 16: return glsl_type::uint16_t_type;
   case 32: return glsl_type::uint_type;
   case 64: return glsl_type::uint64_t_type;
   default:
      unreachable("Unsupported bit size");
   }
}

const glsl_type *
glsl_bool_type(void)
{
   return glsl_type::bool_type;
}

const glsl_type *
glsl_scalar_type(enum glsl_base_type base_type)
{
   return glsl_type::get_instance(base_type, 1, 1);
}

const glsl_type *
glsl_vector_type(enum glsl_base_type base_type, unsigned components)
{
   const glsl_type *t = glsl_type::get_instance(base_type, components, 1);
   assert(t != glsl_type::error_type);
   return t;
}

const glsl_type *
glsl_matrix_type(enum glsl_base_type base_type, unsigned rows, unsigned columns)
{
   const glsl_type *t = glsl_type::get_instance(base_type, rows, columns);
   assert(t != glsl_type::error_type);
   return t;
}

const glsl_type *
glsl_explicit_matrix_type(const glsl_type *mat,
                          unsigned stride, bool row_major)
{
   assert(stride > 0);
   const glsl_type *t = glsl_type::get_instance(mat->base_type,
                                                mat->vector_elements,
                                                mat->matrix_columns,
                                                stride, row_major);
   assert(t != glsl_type::error_type);
   return t;
}

const glsl_type *
glsl_array_type(const glsl_type *base, unsigned elements,
                unsigned explicit_stride)
{
   return glsl_type::get_array_instance(base, elements, explicit_stride);
}

const glsl_type *
glsl_replace_vector_type(const glsl_type *t, unsigned components)
{
   if (glsl_type_is_array(t)) {
      return glsl_array_type(
         glsl_replace_vector_type(t->fields.array, components), t->length,
                                  t->explicit_stride);
   } else if (glsl_type_is_vector_or_scalar(t)) {
      return glsl_vector_type(t->base_type, components);
   } else {
      unreachable("Unhandled base type glsl_replace_vector_type()");
   }
}

const glsl_type *
glsl_struct_type(const glsl_struct_field *fields,
                 unsigned num_fields, const char *name,
                 bool packed)
{
   return glsl_type::get_struct_instance(fields, num_fields, name, packed);
}

const glsl_type *
glsl_interface_type(const glsl_struct_field *fields,
                    unsigned num_fields,
                    enum glsl_interface_packing packing,
                    bool row_major,
                    const char *block_name)
{
   return glsl_type::get_interface_instance(fields, num_fields, packing,
                                            row_major, block_name);
}

const struct glsl_type *
glsl_sampler_type(enum glsl_sampler_dim dim, bool is_shadow, bool is_array,
                  enum glsl_base_type base_type)
{
   return glsl_type::get_sampler_instance(dim, is_shadow, is_array, base_type);
}

const struct glsl_type *
glsl_bare_sampler_type()
{
   return glsl_type::sampler_type;
}

const struct glsl_type *
glsl_bare_shadow_sampler_type()
{
   return glsl_type::samplerShadow_type;
}

const struct glsl_type *
glsl_texture_type(enum glsl_sampler_dim dim, bool is_array,
                  enum glsl_base_type base_type)
{
   return glsl_type::get_texture_instance(dim, is_array, base_type);
}

const struct glsl_type *
glsl_image_type(enum glsl_sampler_dim dim, bool is_array,
                enum glsl_base_type base_type)
{
   return glsl_type::get_image_instance(dim, is_array, base_type);
}

const glsl_type *
glsl_function_type(const glsl_type *return_type,
                   const glsl_function_param *params, unsigned num_params)
{
   return glsl_type::get_function_instance(return_type, params, num_params);
}

const glsl_type *
glsl_transposed_type(const struct glsl_type *type)
{
   assert(glsl_type_is_matrix(type));
   return glsl_type::get_instance(type->base_type, type->matrix_columns,
                                  type->vector_elements);
}

const glsl_type *
glsl_channel_type(const glsl_type *t)
{
   switch (t->base_type) {
   case GLSL_TYPE_ARRAY:
      return glsl_array_type(glsl_channel_type(t->fields.array), t->length,
                             t->explicit_stride);
   case GLSL_TYPE_UINT:
   case GLSL_TYPE_INT:
   case GLSL_TYPE_FLOAT:
   case GLSL_TYPE_FLOAT16:
   case GLSL_TYPE_DOUBLE:
   case GLSL_TYPE_UINT8:
   case GLSL_TYPE_INT8:
   case GLSL_TYPE_UINT16:
   case GLSL_TYPE_INT16:
   case GLSL_TYPE_UINT64:
   case GLSL_TYPE_INT64:
   case GLSL_TYPE_BOOL:
      return glsl_type::get_instance(t->base_type, 1, 1);
   default:
      unreachable("Unhandled base type glsl_channel_type()");
   }
}

const glsl_type *
glsl_float16_type(const struct glsl_type *type)
{
   return type->get_float16_type();
}

const glsl_type *
glsl_int16_type(const struct glsl_type *type)
{
   return type->get_int16_type();
}

const glsl_type *
glsl_uint16_type(const struct glsl_type *type)
{
   return type->get_uint16_type();
}

const struct glsl_type *
glsl_type_to_16bit(const struct glsl_type *old_type)
{
   if (glsl_type_is_array(old_type)) {
      return glsl_array_type(glsl_type_to_16bit(glsl_get_array_element(old_type)),
                             glsl_get_length(old_type),
                             glsl_get_explicit_stride(old_type));
   }

   if (glsl_type_is_vector_or_scalar(old_type)) {
      switch (glsl_get_base_type(old_type)) {
      case GLSL_TYPE_FLOAT:
         return glsl_float16_type(old_type);
      case GLSL_TYPE_UINT:
         return glsl_uint16_type(old_type);
      case GLSL_TYPE_INT:
         return glsl_int16_type(old_type);
      default:
         break;
      }
   }

   return old_type;
}

static void
glsl_size_align_handle_array_and_structs(const struct glsl_type *type,
                                         glsl_type_size_align_func size_align,
                                         unsigned *size, unsigned *align)
{
   if (type->base_type == GLSL_TYPE_ARRAY) {
      unsigned elem_size = 0, elem_align = 0;
      size_align(type->fields.array, &elem_size, &elem_align);
      *align = elem_align;
      *size = type->length * ALIGN_POT(elem_size, elem_align);
   } else {
      assert(type->base_type == GLSL_TYPE_STRUCT ||
             type->base_type == GLSL_TYPE_INTERFACE);

      *size = 0;
      *align = 0;
      for (unsigned i = 0; i < type->length; i++) {
         unsigned elem_size = 0, elem_align = 0;
         size_align(type->fields.structure[i].type, &elem_size, &elem_align);
         *align = MAX2(*align, elem_align);
         *size = ALIGN_POT(*size, elem_align) + elem_size;
      }
   }
}

void
glsl_get_natural_size_align_bytes(const struct glsl_type *type,
                                  unsigned *size, unsigned *align)
{
   switch (type->base_type) {
   case GLSL_TYPE_BOOL:
      /* We special-case Booleans to 32 bits to not cause heartburn for
       * drivers that suddenly get an 8-bit load.
       */
      *size = 4 * type->components();
      *align = 4;
      break;

   case GLSL_TYPE_UINT8:
   case GLSL_TYPE_INT8:
   case GLSL_TYPE_UINT16:
   case GLSL_TYPE_INT16:
   case GLSL_TYPE_FLOAT16:
   case GLSL_TYPE_UINT:
   case GLSL_TYPE_INT:
   case GLSL_TYPE_FLOAT:
   case GLSL_TYPE_DOUBLE:
   case GLSL_TYPE_UINT64:
   case GLSL_TYPE_INT64: {
      unsigned N = glsl_get_bit_size(type) / 8;
      *size = N * type->components();
      *align = N;
      break;
   }

   case GLSL_TYPE_ARRAY:
   case GLSL_TYPE_INTERFACE:
   case GLSL_TYPE_STRUCT:
      glsl_size_align_handle_array_and_structs(type,
                                               glsl_get_natural_size_align_bytes,
                                               size, align);
      break;

   case GLSL_TYPE_SAMPLER:
   case GLSL_TYPE_TEXTURE:
   case GLSL_TYPE_IMAGE:
      /* Bindless samplers and images. */
      *size = 8;
      *align = 8;
      break;

   case GLSL_TYPE_ATOMIC_UINT:
   case GLSL_TYPE_SUBROUTINE:
   case GLSL_TYPE_VOID:
   case GLSL_TYPE_ERROR:
   case GLSL_TYPE_FUNCTION:
      unreachable("type does not have a natural size");
   }
}

/**
 * Returns a byte size/alignment for a type where each array element or struct
 * field is aligned to 16 bytes.
 */
void
glsl_get_vec4_size_align_bytes(const struct glsl_type *type,
                               unsigned *size, unsigned *align)
{
   switch (type->base_type) {
   case GLSL_TYPE_BOOL:
      /* We special-case Booleans to 32 bits to not cause heartburn for
       * drivers that suddenly get an 8-bit load.
       */
      *size = 4 * type->components();
      *align = 16;
      break;

   case GLSL_TYPE_UINT8:
   case GLSL_TYPE_INT8:
   case GLSL_TYPE_UINT16:
   case GLSL_TYPE_INT16:
   case GLSL_TYPE_FLOAT16:
   case GLSL_TYPE_UINT:
   case GLSL_TYPE_INT:
   case GLSL_TYPE_FLOAT:
   case GLSL_TYPE_DOUBLE:
   case GLSL_TYPE_UINT64:
   case GLSL_TYPE_INT64: {
      unsigned N = glsl_get_bit_size(type) / 8;
      *size = 16 * (type->matrix_columns - 1) + N * type->vector_elements;
      *align = 16;
      break;
   }

   case GLSL_TYPE_ARRAY:
   case GLSL_TYPE_INTERFACE:
   case GLSL_TYPE_STRUCT:
      glsl_size_align_handle_array_and_structs(type,
                                               glsl_get_vec4_size_align_bytes,
                                               size, align);
      break;

   case GLSL_TYPE_SAMPLER:
   case GLSL_TYPE_TEXTURE:
   case GLSL_TYPE_IMAGE:
   case GLSL_TYPE_ATOMIC_UINT:
   case GLSL_TYPE_SUBROUTINE:
   case GLSL_TYPE_VOID:
   case GLSL_TYPE_ERROR:
   case GLSL_TYPE_FUNCTION:
      unreachable("type does not make sense for glsl_get_vec4_size_align_bytes()");
   }
}

const glsl_type *
glsl_atomic_uint_type(void)
{
   return glsl_type::atomic_uint_type;
}

unsigned
glsl_atomic_size(const struct glsl_type *type)
{
   return type->atomic_size();
}

bool
glsl_contains_atomic(const struct glsl_type *type)
{
   return type->contains_atomic();
}

bool
glsl_contains_opaque(const struct glsl_type *type)
{
   return type->contains_opaque();
}

int
glsl_get_cl_size(const struct glsl_type *type)
{
   return type->cl_size();
}

int
glsl_get_cl_alignment(const struct glsl_type *type)
{
   return type->cl_alignment();
}

void
glsl_get_cl_type_size_align(const struct glsl_type *type,
                            unsigned *size, unsigned *align)
{
   *size = glsl_get_cl_size(type);
   *align = glsl_get_cl_alignment(type);
}

static unsigned
glsl_type_count(const glsl_type *type, glsl_base_type base_type)
{
   if (glsl_type_is_array(type)) {
      return glsl_get_length(type) *
             glsl_type_count(glsl_get_array_element(type), base_type);
   }

   /* Ignore interface blocks - they can only contain bindless samplers,
    * which we shouldn't count.
    */
   if (glsl_type_is_struct(type)) {
      unsigned count = 0;
      for (unsigned i = 0; i < glsl_get_length(type); i++)
         count += glsl_type_count(glsl_get_struct_field(type, i), base_type);
      return count;
   }

   if (glsl_get_base_type(type) == base_type)
      return 1;

   return 0;
}

unsigned
glsl_type_get_sampler_count(const struct glsl_type *type)
{
   return glsl_type_count(type, GLSL_TYPE_SAMPLER);
}

unsigned
glsl_type_get_texture_count(const struct glsl_type *type)
{
   return glsl_type_count(type, GLSL_TYPE_TEXTURE);
}

unsigned
glsl_type_get_image_count(const struct glsl_type *type)
{
   return glsl_type_count(type, GLSL_TYPE_IMAGE);
}

int
glsl_get_field_index(const struct glsl_type *type, const char *name)
{
   return type->field_index(name);
}

enum glsl_interface_packing
glsl_get_internal_ifc_packing(const struct glsl_type *type,
                              bool std430_supported)
{
   return type->get_internal_ifc_packing(std430_supported);
}

enum glsl_interface_packing
glsl_get_ifc_packing(const struct glsl_type *type)
{
   return type->get_interface_packing();
}

unsigned
glsl_get_std140_base_alignment(const struct glsl_type *type, bool row_major)
{
   return type->std140_base_alignment(row_major);
}

unsigned
glsl_get_std140_size(const struct glsl_type *type, bool row_major)
{
   return type->std140_size(row_major);
}

unsigned
glsl_get_std430_base_alignment(const struct glsl_type *type, bool row_major)
{
   return type->std430_base_alignment(row_major);
}

unsigned
glsl_get_std430_size(const struct glsl_type *type, bool row_major)
{
   return type->std430_size(row_major);
}

unsigned
glsl_get_explicit_size(const struct glsl_type *type, bool align_to_stride)
{
   return type->explicit_size(align_to_stride);
}

unsigned
glsl_get_explicit_alignment(const struct glsl_type *type)
{
   return type->explicit_alignment;
}

bool
glsl_type_is_packed(const struct glsl_type *type)
{
   return type->packed;
}

bool
glsl_type_is_leaf(const struct glsl_type *type)
{
   if (glsl_type_is_struct_or_ifc(type) ||
       (glsl_type_is_array(type) &&
        (glsl_type_is_array(glsl_get_array_element(type)) ||
         glsl_type_is_struct_or_ifc(glsl_get_array_element(type))))) {
      return false;
   } else {
      return true;
   }
}

const struct glsl_type *
glsl_get_explicit_type_for_size_align(const struct glsl_type *type,
                                      glsl_type_size_align_func type_info,
                                      unsigned *size, unsigned *align)
{
   return type->get_explicit_type_for_size_align(type_info, size, align);
}

const struct glsl_type *
glsl_type_wrap_in_arrays(const struct glsl_type *type,
                         const struct glsl_type *arrays)
{
   if (!glsl_type_is_array(arrays))
      return type;

   const glsl_type *elem_type =
      glsl_type_wrap_in_arrays(type, glsl_get_array_element(arrays));
   return glsl_array_type(elem_type, glsl_get_length(arrays),
                          glsl_get_explicit_stride(arrays));
}

const struct glsl_type *
glsl_type_replace_vec3_with_vec4(const struct glsl_type *type)
{
   return type->replace_vec3_with_vec4();
}

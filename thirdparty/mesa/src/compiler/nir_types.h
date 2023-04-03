/*
 * Copyright © 2014 Connor Abbott
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

#ifndef NIR_TYPES_H
#define NIR_TYPES_H

#include <stdio.h>
#include <stdbool.h>

/* C wrapper around compiler/glsl_types.h */

#include "glsl_types.h"

#ifdef __cplusplus
extern "C" {
#else
struct glsl_type;
#endif

const char *glsl_get_type_name(const struct glsl_type *type);

const struct glsl_type *glsl_get_struct_field(const struct glsl_type *type,
                                              unsigned index);

int glsl_get_struct_field_offset(const struct glsl_type *type,
                                 unsigned index);

const struct glsl_struct_field *
glsl_get_struct_field_data(const struct glsl_type *type, unsigned index);

enum glsl_interface_packing
glsl_get_internal_ifc_packing(const struct glsl_type *type,
                              bool std430_supported);
enum glsl_interface_packing
glsl_get_ifc_packing(const struct glsl_type *type);

unsigned glsl_get_std140_base_alignment(const struct glsl_type *type,
                                        bool row_major);
unsigned glsl_get_std140_size(const struct glsl_type *type, bool row_major);
unsigned glsl_get_std430_base_alignment(const struct glsl_type *type,
                                        bool row_major);
unsigned glsl_get_std430_size(const struct glsl_type *type, bool row_major);
unsigned glsl_get_explicit_stride(const struct glsl_type *type);
int glsl_array_size(const struct glsl_type *type);
const struct glsl_type *glsl_get_array_element(const struct glsl_type *type);
const struct glsl_type *glsl_without_array(const struct glsl_type *type);
const struct glsl_type *glsl_without_array_or_matrix(const struct glsl_type *type);
const struct glsl_type *glsl_get_bare_type(const struct glsl_type *type);

const struct glsl_type *glsl_get_column_type(const struct glsl_type *type);

const struct glsl_type *
glsl_get_function_return_type(const struct glsl_type *type);

const struct glsl_function_param *
glsl_get_function_param(const struct glsl_type *type, unsigned index);

const struct glsl_type *
glsl_texture_type_to_sampler(const struct glsl_type *type, bool is_shadow);
const struct glsl_type *
glsl_sampler_type_to_texture(const struct glsl_type *type);

enum glsl_base_type glsl_get_base_type(const struct glsl_type *type);

unsigned glsl_get_vector_elements(const struct glsl_type *type);

unsigned glsl_get_components(const struct glsl_type *type);

unsigned glsl_get_matrix_columns(const struct glsl_type *type);

unsigned glsl_get_length(const struct glsl_type *type);

unsigned glsl_get_aoa_size(const struct glsl_type *type);

unsigned glsl_count_vec4_slots(const struct glsl_type *type,
                               bool is_gl_vertex_input, bool is_bindless);
unsigned glsl_count_dword_slots(const struct glsl_type *type, bool is_bindless);
unsigned glsl_count_attribute_slots(const struct glsl_type *type,
                                    bool is_gl_vertex_input);
unsigned glsl_get_component_slots(const struct glsl_type *type);
unsigned glsl_get_component_slots_aligned(const struct glsl_type *type,
                                          unsigned offset);
unsigned glsl_varying_count(const struct glsl_type *type);

const char *glsl_get_struct_elem_name(const struct glsl_type *type,
                                      unsigned index);

enum glsl_sampler_dim glsl_get_sampler_dim(const struct glsl_type *type);
enum glsl_base_type glsl_get_sampler_result_type(const struct glsl_type *type);
unsigned glsl_get_sampler_target(const struct glsl_type *type);
int glsl_get_sampler_coordinate_components(const struct glsl_type *type);

unsigned glsl_get_struct_location_offset(const struct glsl_type *type,
                                         unsigned length);

unsigned glsl_atomic_size(const struct glsl_type *type);

int glsl_get_cl_size(const struct glsl_type *type);

int glsl_get_cl_alignment(const struct glsl_type *type);

void glsl_get_cl_type_size_align(const struct glsl_type *type,
                                 unsigned *size, unsigned *align);

unsigned glsl_get_explicit_size(const struct glsl_type *type, bool align_to_stride);
unsigned glsl_get_explicit_alignment(const struct glsl_type *type);

static inline unsigned
glsl_get_bit_size(const struct glsl_type *type)
{
   return glsl_base_type_get_bit_size(glsl_get_base_type(type));
}

bool glsl_type_is_packed(const struct glsl_type *type);
bool glsl_type_is_16bit(const struct glsl_type *type);
bool glsl_type_is_32bit(const struct glsl_type *type);
bool glsl_type_is_64bit(const struct glsl_type *type);
bool glsl_type_is_void(const struct glsl_type *type);
bool glsl_type_is_error(const struct glsl_type *type);
bool glsl_type_is_vector(const struct glsl_type *type);
bool glsl_type_is_scalar(const struct glsl_type *type);
bool glsl_type_is_vector_or_scalar(const struct glsl_type *type);
bool glsl_type_is_matrix(const struct glsl_type *type);
bool glsl_matrix_type_is_row_major(const struct glsl_type *type);
bool glsl_type_is_array(const struct glsl_type *type);
bool glsl_type_is_unsized_array(const struct glsl_type *type);
bool glsl_type_is_array_of_arrays(const struct glsl_type *type);
bool glsl_type_is_array_or_matrix(const struct glsl_type *type);
bool glsl_type_is_struct(const struct glsl_type *type);
bool glsl_type_is_interface(const struct glsl_type *type);
bool glsl_type_is_struct_or_ifc(const struct glsl_type *type);
bool glsl_type_is_sampler(const struct glsl_type *type);
bool glsl_type_is_bare_sampler(const struct glsl_type *type);
bool glsl_type_is_texture(const struct glsl_type *type);
bool glsl_type_is_image(const struct glsl_type *type);
bool glsl_type_is_dual_slot(const struct glsl_type *type);
bool glsl_type_is_numeric(const struct glsl_type *type);
bool glsl_type_is_boolean(const struct glsl_type *type);
bool glsl_type_is_integer(const struct glsl_type *type);
bool glsl_type_contains_64bit(const struct glsl_type *type);
bool glsl_type_contains_image(const struct glsl_type *type);
bool glsl_sampler_type_is_shadow(const struct glsl_type *type);
bool glsl_sampler_type_is_array(const struct glsl_type *type);
bool glsl_struct_type_is_packed(const struct glsl_type *type);
bool glsl_contains_atomic(const struct glsl_type *type);
bool glsl_contains_double(const struct glsl_type *type);
bool glsl_contains_integer(const struct glsl_type *type);
bool glsl_contains_opaque(const struct glsl_type *type);
bool glsl_record_compare(const struct glsl_type *a, const struct glsl_type *b,
                         bool match_name, bool match_locations,
                         bool match_precision);

const struct glsl_type *glsl_void_type(void);
const struct glsl_type *glsl_float_type(void);
const struct glsl_type *glsl_float16_t_type(void);
const struct glsl_type *glsl_double_type(void);
const struct glsl_type *glsl_floatN_t_type(unsigned bit_size);
const struct glsl_type *glsl_vec_type(unsigned n);
const struct glsl_type *glsl_dvec_type(unsigned n);
const struct glsl_type *glsl_vec4_type(void);
const struct glsl_type *glsl_uvec4_type(void);
const struct glsl_type *glsl_ivec4_type(void);
const struct glsl_type *glsl_int_type(void);
const struct glsl_type *glsl_uint_type(void);
const struct glsl_type *glsl_int64_t_type(void);
const struct glsl_type *glsl_uint64_t_type(void);
const struct glsl_type *glsl_int16_t_type(void);
const struct glsl_type *glsl_uint16_t_type(void);
const struct glsl_type *glsl_int8_t_type(void);
const struct glsl_type *glsl_uint8_t_type(void);
const struct glsl_type *glsl_intN_t_type(unsigned bit_size);
const struct glsl_type *glsl_uintN_t_type(unsigned bit_size);
const struct glsl_type *glsl_bool_type(void);

const struct glsl_type *glsl_scalar_type(enum glsl_base_type base_type);
const struct glsl_type *glsl_vector_type(enum glsl_base_type base_type,
                                         unsigned components);
const struct glsl_type * glsl_replace_vector_type(const struct glsl_type *t,
                                                  unsigned components);
const struct glsl_type *glsl_matrix_type(enum glsl_base_type base_type,
                                         unsigned rows, unsigned columns);
const struct glsl_type *glsl_explicit_matrix_type(const struct glsl_type *mat,
                                                  unsigned stride,
                                                  bool row_major);

const struct glsl_type *glsl_array_type(const struct glsl_type *base,
                                        unsigned elements,
                                        unsigned explicit_stride);

const struct glsl_type *glsl_struct_type(const struct glsl_struct_field *fields,
                                         unsigned num_fields, const char *name,
                                         bool packed);
const struct glsl_type *glsl_interface_type(const struct glsl_struct_field *fields,
                                            unsigned num_fields,
                                            enum glsl_interface_packing packing,
                                            bool row_major,
                                            const char *block_name);
const struct glsl_type *glsl_sampler_type(enum glsl_sampler_dim dim,
                                          bool is_shadow, bool is_array,
                                          enum glsl_base_type base_type);
const struct glsl_type *glsl_bare_sampler_type();
const struct glsl_type *glsl_bare_shadow_sampler_type();
const struct glsl_type *glsl_texture_type(enum glsl_sampler_dim dim,
                                          bool is_array,
                                          enum glsl_base_type base_type);
const struct glsl_type *glsl_image_type(enum glsl_sampler_dim dim,
                                        bool is_array,
                                        enum glsl_base_type base_type);
const struct glsl_type * glsl_function_type(const struct glsl_type *return_type,
                                            const struct glsl_function_param *params,
                                            unsigned num_params);

const struct glsl_type *glsl_transposed_type(const struct glsl_type *type);

const struct glsl_type *glsl_channel_type(const struct glsl_type *type);

const struct glsl_type *glsl_float16_type(const struct glsl_type *type);
const struct glsl_type *glsl_int16_type(const struct glsl_type *type);
const struct glsl_type *glsl_uint16_type(const struct glsl_type *type);
const struct glsl_type *glsl_type_to_16bit(const struct glsl_type *old_type);

void glsl_get_natural_size_align_bytes(const struct glsl_type *type,
                                       unsigned *size, unsigned *align);
void glsl_get_vec4_size_align_bytes(const struct glsl_type *type,
                                    unsigned *size, unsigned *align);

const struct glsl_type *glsl_atomic_uint_type(void);

const struct glsl_type *glsl_get_explicit_type_for_size_align(const struct glsl_type *type,
                                                              glsl_type_size_align_func type_info,
                                                              unsigned *size, unsigned *align);

const struct glsl_type *glsl_type_wrap_in_arrays(const struct glsl_type *type,
                                                 const struct glsl_type *arrays);

const struct glsl_type *glsl_type_replace_vec3_with_vec4(const struct glsl_type *type);

unsigned glsl_type_get_sampler_count(const struct glsl_type *type);
unsigned glsl_type_get_texture_count(const struct glsl_type *type);
unsigned glsl_type_get_image_count(const struct glsl_type *type);

int glsl_get_field_index(const struct glsl_type *type, const char *name);

bool glsl_type_is_leaf(const struct glsl_type *type);

#ifdef __cplusplus
}
#endif

#endif /* NIR_TYPES_H */

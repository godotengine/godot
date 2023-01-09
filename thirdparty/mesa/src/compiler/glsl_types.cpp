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

#include <stdio.h>
#include "main/macros.h"
#include "compiler/glsl/glsl_parser_extras.h"
#include "glsl_types.h"
#include "util/hash_table.h"
#include "util/u_string.h"


simple_mtx_t glsl_type::hash_mutex = SIMPLE_MTX_INITIALIZER;
hash_table *glsl_type::explicit_matrix_types = NULL;
hash_table *glsl_type::array_types = NULL;
hash_table *glsl_type::struct_types = NULL;
hash_table *glsl_type::interface_types = NULL;
hash_table *glsl_type::function_types = NULL;
hash_table *glsl_type::subroutine_types = NULL;

/* There might be multiple users for types (e.g. application using OpenGL
 * and Vulkan simultaneously or app using multiple Vulkan instances). Counter
 * is used to make sure we don't release the types if a user is still present.
 */
static uint32_t glsl_type_users = 0;

glsl_type::glsl_type(GLenum gl_type,
                     glsl_base_type base_type, unsigned vector_elements,
                     unsigned matrix_columns, const char *name,
                     unsigned explicit_stride, bool row_major,
                     unsigned explicit_alignment) :
   gl_type(gl_type),
   base_type(base_type), sampled_type(GLSL_TYPE_VOID),
   sampler_dimensionality(0), sampler_shadow(0), sampler_array(0),
   interface_packing(0), interface_row_major(row_major), packed(0),
   vector_elements(vector_elements), matrix_columns(matrix_columns),
   length(0), explicit_stride(explicit_stride),
   explicit_alignment(explicit_alignment)
{
   /* Values of these types must fit in the two bits of
    * glsl_type::sampled_type.
    */
   STATIC_ASSERT((unsigned(GLSL_TYPE_UINT)  & 3) == unsigned(GLSL_TYPE_UINT));
   STATIC_ASSERT((unsigned(GLSL_TYPE_INT)   & 3) == unsigned(GLSL_TYPE_INT));
   STATIC_ASSERT((unsigned(GLSL_TYPE_FLOAT) & 3) == unsigned(GLSL_TYPE_FLOAT));

   ASSERT_BITFIELD_SIZE(glsl_type, base_type, GLSL_TYPE_ERROR);
   ASSERT_BITFIELD_SIZE(glsl_type, sampled_type, GLSL_TYPE_ERROR);
   ASSERT_BITFIELD_SIZE(glsl_type, sampler_dimensionality,
                        GLSL_SAMPLER_DIM_SUBPASS_MS);

   this->mem_ctx = ralloc_context(NULL);
   assert(this->mem_ctx != NULL);

   assert(name != NULL);
   this->name = ralloc_strdup(this->mem_ctx, name);

   /* Neither dimension is zero or both dimensions are zero.
    */
   assert((vector_elements == 0) == (matrix_columns == 0));
   assert(util_is_power_of_two_or_zero(explicit_alignment));
   memset(& fields, 0, sizeof(fields));
}

glsl_type::glsl_type(GLenum gl_type, glsl_base_type base_type,
                     enum glsl_sampler_dim dim, bool shadow, bool array,
                     glsl_base_type type, const char *name) :
   gl_type(gl_type),
   base_type(base_type), sampled_type(type),
   sampler_dimensionality(dim), sampler_shadow(shadow),
   sampler_array(array), interface_packing(0),
   interface_row_major(0), packed(0),
   length(0), explicit_stride(0), explicit_alignment(0)
{
   this->mem_ctx = ralloc_context(NULL);
   assert(this->mem_ctx != NULL);

   assert(name != NULL);
   this->name = ralloc_strdup(this->mem_ctx, name);

   memset(& fields, 0, sizeof(fields));

   matrix_columns = vector_elements = 1;
}

glsl_type::glsl_type(const glsl_struct_field *fields, unsigned num_fields,
                     const char *name, bool packed,
                     unsigned explicit_alignment) :
   gl_type(0),
   base_type(GLSL_TYPE_STRUCT), sampled_type(GLSL_TYPE_VOID),
   sampler_dimensionality(0), sampler_shadow(0), sampler_array(0),
   interface_packing(0), interface_row_major(0), packed(packed),
   vector_elements(0), matrix_columns(0),
   length(num_fields), explicit_stride(0),
   explicit_alignment(explicit_alignment)
{
   unsigned int i;

   assert(util_is_power_of_two_or_zero(explicit_alignment));

   this->mem_ctx = ralloc_context(NULL);
   assert(this->mem_ctx != NULL);

   assert(name != NULL);
   this->name = ralloc_strdup(this->mem_ctx, name);
   /* Zero-fill to prevent spurious Valgrind errors when serializing NIR
    * due to uninitialized unused bits in bit fields. */
   this->fields.structure = rzalloc_array(this->mem_ctx,
                                          glsl_struct_field, length);

   for (i = 0; i < length; i++) {
      this->fields.structure[i] = fields[i];
      this->fields.structure[i].name = ralloc_strdup(this->fields.structure,
                                                     fields[i].name);
   }
}

glsl_type::glsl_type(const glsl_struct_field *fields, unsigned num_fields,
                     enum glsl_interface_packing packing,
                     bool row_major, const char *name) :
   gl_type(0),
   base_type(GLSL_TYPE_INTERFACE), sampled_type(GLSL_TYPE_VOID),
   sampler_dimensionality(0), sampler_shadow(0), sampler_array(0),
   interface_packing((unsigned) packing),
   interface_row_major((unsigned) row_major), packed(0),
   vector_elements(0), matrix_columns(0),
   length(num_fields), explicit_stride(0), explicit_alignment(0)
{
   unsigned int i;

   this->mem_ctx = ralloc_context(NULL);
   assert(this->mem_ctx != NULL);

   assert(name != NULL);
   this->name = ralloc_strdup(this->mem_ctx, name);
   this->fields.structure = rzalloc_array(this->mem_ctx,
                                          glsl_struct_field, length);
   for (i = 0; i < length; i++) {
      this->fields.structure[i] = fields[i];
      this->fields.structure[i].name = ralloc_strdup(this->fields.structure,
                                                     fields[i].name);
   }
}

glsl_type::glsl_type(const glsl_type *return_type,
                     const glsl_function_param *params, unsigned num_params) :
   gl_type(0),
   base_type(GLSL_TYPE_FUNCTION), sampled_type(GLSL_TYPE_VOID),
   sampler_dimensionality(0), sampler_shadow(0), sampler_array(0),
   interface_packing(0), interface_row_major(0), packed(0),
   vector_elements(0), matrix_columns(0),
   length(num_params), explicit_stride(0), explicit_alignment(0)
{
   unsigned int i;

   this->mem_ctx = ralloc_context(NULL);
   assert(this->mem_ctx != NULL);

   this->name = ralloc_strdup(this->mem_ctx, "");

   this->fields.parameters = rzalloc_array(this->mem_ctx,
                                           glsl_function_param, num_params + 1);

   /* We store the return type as the first parameter */
   this->fields.parameters[0].type = return_type;
   this->fields.parameters[0].in = false;
   this->fields.parameters[0].out = true;

   /* We store the i'th parameter in slot i+1 */
   for (i = 0; i < length; i++) {
      this->fields.parameters[i + 1].type = params[i].type;
      this->fields.parameters[i + 1].in = params[i].in;
      this->fields.parameters[i + 1].out = params[i].out;
   }
}

glsl_type::glsl_type(const char *subroutine_name) :
   gl_type(0),
   base_type(GLSL_TYPE_SUBROUTINE), sampled_type(GLSL_TYPE_VOID),
   sampler_dimensionality(0), sampler_shadow(0), sampler_array(0),
   interface_packing(0), interface_row_major(0), packed(0),
   vector_elements(1), matrix_columns(1),
   length(0), explicit_stride(0), explicit_alignment(0)
{
   this->mem_ctx = ralloc_context(NULL);
   assert(this->mem_ctx != NULL);

   assert(subroutine_name != NULL);
   this->name = ralloc_strdup(this->mem_ctx, subroutine_name);
}

glsl_type::~glsl_type()
{
    ralloc_free(this->mem_ctx);
}

bool
glsl_type::contains_sampler() const
{
   if (this->is_array()) {
      return this->fields.array->contains_sampler();
   } else if (this->is_struct() || this->is_interface()) {
      for (unsigned int i = 0; i < this->length; i++) {
         if (this->fields.structure[i].type->contains_sampler())
            return true;
      }
      return false;
   } else {
      return this->is_sampler();
   }
}

bool
glsl_type::contains_array() const
{
   if (this->is_struct() || this->is_interface()) {
      for (unsigned int i = 0; i < this->length; i++) {
         if (this->fields.structure[i].type->contains_array())
            return true;
      }
      return false;
   } else {
      return this->is_array();
   }
}

bool
glsl_type::contains_integer() const
{
   if (this->is_array()) {
      return this->fields.array->contains_integer();
   } else if (this->is_struct() || this->is_interface()) {
      for (unsigned int i = 0; i < this->length; i++) {
         if (this->fields.structure[i].type->contains_integer())
            return true;
      }
      return false;
   } else {
      return this->is_integer();
   }
}

bool
glsl_type::contains_double() const
{
   if (this->is_array()) {
      return this->fields.array->contains_double();
   } else if (this->is_struct() || this->is_interface()) {
      for (unsigned int i = 0; i < this->length; i++) {
         if (this->fields.structure[i].type->contains_double())
            return true;
      }
      return false;
   } else {
      return this->is_double();
   }
}

bool
glsl_type::contains_64bit() const
{
   if (this->is_array()) {
      return this->fields.array->contains_64bit();
   } else if (this->is_struct() || this->is_interface()) {
      for (unsigned int i = 0; i < this->length; i++) {
         if (this->fields.structure[i].type->contains_64bit())
            return true;
      }
      return false;
   } else {
      return this->is_64bit();
   }
}

bool
glsl_type::contains_opaque() const {
   switch (base_type) {
   case GLSL_TYPE_SAMPLER:
   case GLSL_TYPE_IMAGE:
   case GLSL_TYPE_ATOMIC_UINT:
      return true;
   case GLSL_TYPE_ARRAY:
      return fields.array->contains_opaque();
   case GLSL_TYPE_STRUCT:
   case GLSL_TYPE_INTERFACE:
      for (unsigned int i = 0; i < length; i++) {
         if (fields.structure[i].type->contains_opaque())
            return true;
      }
      return false;
   default:
      return false;
   }
}

bool
glsl_type::contains_subroutine() const
{
   if (this->is_array()) {
      return this->fields.array->contains_subroutine();
   } else if (this->is_struct() || this->is_interface()) {
      for (unsigned int i = 0; i < this->length; i++) {
         if (this->fields.structure[i].type->contains_subroutine())
            return true;
      }
      return false;
   } else {
      return this->is_subroutine();
   }
}

gl_texture_index
glsl_type::sampler_index() const
{
   const glsl_type *const t = (this->is_array()) ? this->fields.array : this;

   assert(t->is_sampler() || t->is_image());

   switch (t->sampler_dimensionality) {
   case GLSL_SAMPLER_DIM_1D:
      return (t->sampler_array) ? TEXTURE_1D_ARRAY_INDEX : TEXTURE_1D_INDEX;
   case GLSL_SAMPLER_DIM_2D:
      return (t->sampler_array) ? TEXTURE_2D_ARRAY_INDEX : TEXTURE_2D_INDEX;
   case GLSL_SAMPLER_DIM_3D:
      return TEXTURE_3D_INDEX;
   case GLSL_SAMPLER_DIM_CUBE:
      return (t->sampler_array) ? TEXTURE_CUBE_ARRAY_INDEX : TEXTURE_CUBE_INDEX;
   case GLSL_SAMPLER_DIM_RECT:
      return TEXTURE_RECT_INDEX;
   case GLSL_SAMPLER_DIM_BUF:
      return TEXTURE_BUFFER_INDEX;
   case GLSL_SAMPLER_DIM_EXTERNAL:
      return TEXTURE_EXTERNAL_INDEX;
   case GLSL_SAMPLER_DIM_MS:
      return (t->sampler_array) ? TEXTURE_2D_MULTISAMPLE_ARRAY_INDEX : TEXTURE_2D_MULTISAMPLE_INDEX;
   default:
      assert(!"Should not get here.");
      return TEXTURE_BUFFER_INDEX;
   }
}

bool
glsl_type::contains_image() const
{
   if (this->is_array()) {
      return this->fields.array->contains_image();
   } else if (this->is_struct() || this->is_interface()) {
      for (unsigned int i = 0; i < this->length; i++) {
         if (this->fields.structure[i].type->contains_image())
            return true;
      }
      return false;
   } else {
      return this->is_image();
   }
}

const glsl_type *glsl_type::get_base_type() const
{
   switch (base_type) {
   case GLSL_TYPE_UINT:
      return uint_type;
   case GLSL_TYPE_UINT16:
      return uint16_t_type;
   case GLSL_TYPE_UINT8:
      return uint8_t_type;
   case GLSL_TYPE_INT:
      return int_type;
   case GLSL_TYPE_INT16:
      return int16_t_type;
   case GLSL_TYPE_INT8:
      return int8_t_type;
   case GLSL_TYPE_FLOAT:
      return float_type;
   case GLSL_TYPE_FLOAT16:
      return float16_t_type;
   case GLSL_TYPE_DOUBLE:
      return double_type;
   case GLSL_TYPE_BOOL:
      return bool_type;
   case GLSL_TYPE_UINT64:
      return uint64_t_type;
   case GLSL_TYPE_INT64:
      return int64_t_type;
   default:
      return error_type;
   }
}


const glsl_type *glsl_type::get_scalar_type() const
{
   const glsl_type *type = this;

   /* Handle arrays */
   while (type->base_type == GLSL_TYPE_ARRAY)
      type = type->fields.array;

   const glsl_type *scalar_type = type->get_base_type();
   if (scalar_type == error_type)
      return type;

   return scalar_type;
}


const glsl_type *glsl_type::get_bare_type() const
{
   switch (this->base_type) {
   case GLSL_TYPE_UINT8:
   case GLSL_TYPE_INT8:
   case GLSL_TYPE_UINT16:
   case GLSL_TYPE_INT16:
   case GLSL_TYPE_FLOAT16:
   case GLSL_TYPE_UINT:
   case GLSL_TYPE_INT:
   case GLSL_TYPE_FLOAT:
   case GLSL_TYPE_BOOL:
   case GLSL_TYPE_DOUBLE:
   case GLSL_TYPE_UINT64:
   case GLSL_TYPE_INT64:
      return get_instance(this->base_type, this->vector_elements,
                          this->matrix_columns);

   case GLSL_TYPE_STRUCT:
   case GLSL_TYPE_INTERFACE: {
      glsl_struct_field *bare_fields = new glsl_struct_field[this->length];
      for (unsigned i = 0; i < this->length; i++) {
         bare_fields[i].type = this->fields.structure[i].type->get_bare_type();
         bare_fields[i].name = this->fields.structure[i].name;
      }
      const glsl_type *bare_type =
         get_struct_instance(bare_fields, this->length, this->name);
      delete[] bare_fields;
      return bare_type;
   }

   case GLSL_TYPE_ARRAY:
      return get_array_instance(this->fields.array->get_bare_type(),
                                this->length);

   case GLSL_TYPE_SAMPLER:
   case GLSL_TYPE_TEXTURE:
   case GLSL_TYPE_IMAGE:
   case GLSL_TYPE_ATOMIC_UINT:
   case GLSL_TYPE_VOID:
   case GLSL_TYPE_SUBROUTINE:
   case GLSL_TYPE_FUNCTION:
   case GLSL_TYPE_ERROR:
      return this;
   }

   unreachable("Invalid base type");
}

const glsl_type *glsl_type::get_float16_type() const
{
   assert(this->base_type == GLSL_TYPE_FLOAT);

   return get_instance(GLSL_TYPE_FLOAT16,
                       this->vector_elements,
                       this->matrix_columns,
                       this->explicit_stride,
                       this->interface_row_major);
}

const glsl_type *glsl_type::get_int16_type() const
{
   assert(this->base_type == GLSL_TYPE_INT);

   return get_instance(GLSL_TYPE_INT16,
                       this->vector_elements,
                       this->matrix_columns,
                       this->explicit_stride,
                       this->interface_row_major);
}

const glsl_type *glsl_type::get_uint16_type() const
{
   assert(this->base_type == GLSL_TYPE_UINT);

   return get_instance(GLSL_TYPE_UINT16,
                       this->vector_elements,
                       this->matrix_columns,
                       this->explicit_stride,
                       this->interface_row_major);
}

static void
hash_free_type_function(struct hash_entry *entry)
{
   glsl_type *type = (glsl_type *) entry->data;

   if (type->is_array())
      free((void*)entry->key);

   delete type;
}

void
glsl_type_singleton_init_or_ref()
{
   simple_mtx_lock(&glsl_type::hash_mutex);
   glsl_type_users++;
   simple_mtx_unlock(&glsl_type::hash_mutex);
}

void
glsl_type_singleton_decref()
{
   simple_mtx_lock(&glsl_type::hash_mutex);
   assert(glsl_type_users > 0);

   /* Do not release glsl_types if they are still used. */
   if (--glsl_type_users) {
      simple_mtx_unlock(&glsl_type::hash_mutex);
      return;
   }

   if (glsl_type::explicit_matrix_types != NULL) {
      _mesa_hash_table_destroy(glsl_type::explicit_matrix_types,
                               hash_free_type_function);
      glsl_type::explicit_matrix_types = NULL;
   }

   if (glsl_type::array_types != NULL) {
      _mesa_hash_table_destroy(glsl_type::array_types, hash_free_type_function);
      glsl_type::array_types = NULL;
   }

   if (glsl_type::struct_types != NULL) {
      _mesa_hash_table_destroy(glsl_type::struct_types, hash_free_type_function);
      glsl_type::struct_types = NULL;
   }

   if (glsl_type::interface_types != NULL) {
      _mesa_hash_table_destroy(glsl_type::interface_types, hash_free_type_function);
      glsl_type::interface_types = NULL;
   }

   if (glsl_type::function_types != NULL) {
      _mesa_hash_table_destroy(glsl_type::function_types, hash_free_type_function);
      glsl_type::function_types = NULL;
   }

   if (glsl_type::subroutine_types != NULL) {
      _mesa_hash_table_destroy(glsl_type::subroutine_types, hash_free_type_function);
      glsl_type::subroutine_types = NULL;
   }

   simple_mtx_unlock(&glsl_type::hash_mutex);
}


glsl_type::glsl_type(const glsl_type *array, unsigned length,
                     unsigned explicit_stride) :
   base_type(GLSL_TYPE_ARRAY), sampled_type(GLSL_TYPE_VOID),
   sampler_dimensionality(0), sampler_shadow(0), sampler_array(0),
   interface_packing(0), interface_row_major(0), packed(0),
   vector_elements(0), matrix_columns(0),
   length(length), name(NULL), explicit_stride(explicit_stride),
   explicit_alignment(array->explicit_alignment)
{
   this->fields.array = array;
   /* Inherit the gl type of the base. The GL type is used for
    * uniform/statevar handling in Mesa and the arrayness of the type
    * is represented by the size rather than the type.
    */
   this->gl_type = array->gl_type;

   /* Allow a maximum of 10 characters for the array size.  This is enough
    * for 32-bits of ~0.  The extra 3 are for the '[', ']', and terminating
    * NUL.
    */
   const unsigned name_length = strlen(array->name) + 10 + 3;

   this->mem_ctx = ralloc_context(NULL);
   assert(this->mem_ctx != NULL);

   char *const n = (char *) ralloc_size(this->mem_ctx, name_length);

   if (length == 0)
      snprintf(n, name_length, "%s[]", array->name);
   else {
      /* insert outermost dimensions in the correct spot
       * otherwise the dimension order will be backwards
       */
      const char *pos = strchr(array->name, '[');
      if (pos) {
         int idx = pos - array->name;
         snprintf(n, idx+1, "%s", array->name);
         snprintf(n + idx, name_length - idx, "[%u]%s",
                       length, array->name + idx);
      } else {
         snprintf(n, name_length, "%s[%u]", array->name, length);
      }
   }

   this->name = n;
}

const glsl_type *
glsl_type::vec(unsigned components, const glsl_type *const ts[])
{
   unsigned n = components;

   if (components == 8)
      n = 6;
   else if (components == 16)
      n = 7;

   if (n == 0 || n > 7)
      return error_type;

   return ts[n - 1];
}

#define VECN(components, sname, vname)           \
const glsl_type *                                \
glsl_type:: vname (unsigned components)          \
{                                                \
   static const glsl_type *const ts[] = {        \
      sname ## _type, vname ## 2_type,           \
      vname ## 3_type, vname ## 4_type,          \
      vname ## 5_type,                           \
      vname ## 8_type, vname ## 16_type,         \
   };                                            \
   return glsl_type::vec(components, ts);        \
}

VECN(components, float, vec)
VECN(components, float16_t, f16vec)
VECN(components, double, dvec)
VECN(components, int, ivec)
VECN(components, uint, uvec)
VECN(components, bool, bvec)
VECN(components, int64_t, i64vec)
VECN(components, uint64_t, u64vec)
VECN(components, int16_t, i16vec)
VECN(components, uint16_t, u16vec)
VECN(components, int8_t, i8vec)
VECN(components, uint8_t, u8vec)

const glsl_type *
glsl_type::get_instance(unsigned base_type, unsigned rows, unsigned columns,
                        unsigned explicit_stride, bool row_major,
                        unsigned explicit_alignment)
{
   if (base_type == GLSL_TYPE_VOID) {
      assert(explicit_stride == 0 && explicit_alignment == 0 && !row_major);
      return void_type;
   }

   /* Matrix and vector types with explicit strides or alignment have to be
    * looked up in a table so they're handled separately.
    */
   if (explicit_stride > 0 || explicit_alignment > 0) {
      if (explicit_alignment > 0) {
         assert(util_is_power_of_two_nonzero(explicit_alignment));
         assert(explicit_stride % explicit_alignment == 0);
      }

      const glsl_type *bare_type = get_instance(base_type, rows, columns);

      assert(columns > 1 || (rows > 1 && !row_major));

      char name[128];
      snprintf(name, sizeof(name), "%sx%ua%uB%s", bare_type->name,
               explicit_stride, explicit_alignment, row_major ? "RM" : "");

      simple_mtx_lock(&glsl_type::hash_mutex);
      assert(glsl_type_users > 0);

      if (explicit_matrix_types == NULL) {
         explicit_matrix_types =
            _mesa_hash_table_create(NULL, _mesa_hash_string,
                                    _mesa_key_string_equal);
      }

      const struct hash_entry *entry =
         _mesa_hash_table_search(explicit_matrix_types, name);
      if (entry == NULL) {
         const glsl_type *t = new glsl_type(bare_type->gl_type,
                                            (glsl_base_type)base_type,
                                            rows, columns, name,
                                            explicit_stride, row_major,
                                            explicit_alignment);

         entry = _mesa_hash_table_insert(explicit_matrix_types,
                                         t->name, (void *)t);
      }

      assert(((glsl_type *) entry->data)->base_type == base_type);
      assert(((glsl_type *) entry->data)->vector_elements == rows);
      assert(((glsl_type *) entry->data)->matrix_columns == columns);
      assert(((glsl_type *) entry->data)->explicit_stride == explicit_stride);
      assert(((glsl_type *) entry->data)->explicit_alignment == explicit_alignment);

      const glsl_type *t = (const glsl_type *) entry->data;

      simple_mtx_unlock(&glsl_type::hash_mutex);

      return t;
   }

   assert(!row_major);

   /* Treat GLSL vectors as Nx1 matrices.
    */
   if (columns == 1) {
      switch (base_type) {
      case GLSL_TYPE_UINT:
         return uvec(rows);
      case GLSL_TYPE_INT:
         return ivec(rows);
      case GLSL_TYPE_FLOAT:
         return vec(rows);
      case GLSL_TYPE_FLOAT16:
         return f16vec(rows);
      case GLSL_TYPE_DOUBLE:
         return dvec(rows);
      case GLSL_TYPE_BOOL:
         return bvec(rows);
      case GLSL_TYPE_UINT64:
         return u64vec(rows);
      case GLSL_TYPE_INT64:
         return i64vec(rows);
      case GLSL_TYPE_UINT16:
         return u16vec(rows);
      case GLSL_TYPE_INT16:
         return i16vec(rows);
      case GLSL_TYPE_UINT8:
         return u8vec(rows);
      case GLSL_TYPE_INT8:
         return i8vec(rows);
      default:
         return error_type;
      }
   } else {
      if ((base_type != GLSL_TYPE_FLOAT &&
           base_type != GLSL_TYPE_DOUBLE &&
           base_type != GLSL_TYPE_FLOAT16) || (rows == 1))
         return error_type;

      /* GLSL matrix types are named mat{COLUMNS}x{ROWS}.  Only the following
       * combinations are valid:
       *
       *   1 2 3 4
       * 1
       * 2   x x x
       * 3   x x x
       * 4   x x x
       */
#define IDX(c,r) (((c-1)*3) + (r-1))

      switch (base_type) {
      case GLSL_TYPE_DOUBLE: {
         switch (IDX(columns, rows)) {
         case IDX(2,2): return dmat2_type;
         case IDX(2,3): return dmat2x3_type;
         case IDX(2,4): return dmat2x4_type;
         case IDX(3,2): return dmat3x2_type;
         case IDX(3,3): return dmat3_type;
         case IDX(3,4): return dmat3x4_type;
         case IDX(4,2): return dmat4x2_type;
         case IDX(4,3): return dmat4x3_type;
         case IDX(4,4): return dmat4_type;
         default: return error_type;
         }
      }
      case GLSL_TYPE_FLOAT: {
         switch (IDX(columns, rows)) {
         case IDX(2,2): return mat2_type;
         case IDX(2,3): return mat2x3_type;
         case IDX(2,4): return mat2x4_type;
         case IDX(3,2): return mat3x2_type;
         case IDX(3,3): return mat3_type;
         case IDX(3,4): return mat3x4_type;
         case IDX(4,2): return mat4x2_type;
         case IDX(4,3): return mat4x3_type;
         case IDX(4,4): return mat4_type;
         default: return error_type;
         }
      }
      case GLSL_TYPE_FLOAT16: {
         switch (IDX(columns, rows)) {
         case IDX(2,2): return f16mat2_type;
         case IDX(2,3): return f16mat2x3_type;
         case IDX(2,4): return f16mat2x4_type;
         case IDX(3,2): return f16mat3x2_type;
         case IDX(3,3): return f16mat3_type;
         case IDX(3,4): return f16mat3x4_type;
         case IDX(4,2): return f16mat4x2_type;
         case IDX(4,3): return f16mat4x3_type;
         case IDX(4,4): return f16mat4_type;
         default: return error_type;
         }
      }
      default: return error_type;
      }
   }

   assert(!"Should not get here.");
   return error_type;
}

const glsl_type *
glsl_type::get_sampler_instance(enum glsl_sampler_dim dim,
                                bool shadow,
                                bool array,
                                glsl_base_type type)
{
   switch (type) {
   case GLSL_TYPE_FLOAT:
      switch (dim) {
      case GLSL_SAMPLER_DIM_1D:
         if (shadow)
            return (array ? sampler1DArrayShadow_type : sampler1DShadow_type);
         else
            return (array ? sampler1DArray_type : sampler1D_type);
      case GLSL_SAMPLER_DIM_2D:
         if (shadow)
            return (array ? sampler2DArrayShadow_type : sampler2DShadow_type);
         else
            return (array ? sampler2DArray_type : sampler2D_type);
      case GLSL_SAMPLER_DIM_3D:
         if (shadow || array)
            return error_type;
         else
            return sampler3D_type;
      case GLSL_SAMPLER_DIM_CUBE:
         if (shadow)
            return (array ? samplerCubeArrayShadow_type : samplerCubeShadow_type);
         else
            return (array ? samplerCubeArray_type : samplerCube_type);
      case GLSL_SAMPLER_DIM_RECT:
         if (array)
            return error_type;
         if (shadow)
            return sampler2DRectShadow_type;
         else
            return sampler2DRect_type;
      case GLSL_SAMPLER_DIM_BUF:
         if (shadow || array)
            return error_type;
         else
            return samplerBuffer_type;
      case GLSL_SAMPLER_DIM_MS:
         if (shadow)
            return error_type;
         return (array ? sampler2DMSArray_type : sampler2DMS_type);
      case GLSL_SAMPLER_DIM_EXTERNAL:
         if (shadow || array)
            return error_type;
         else
            return samplerExternalOES_type;
      case GLSL_SAMPLER_DIM_SUBPASS:
      case GLSL_SAMPLER_DIM_SUBPASS_MS:
         return error_type;
      }
   case GLSL_TYPE_INT:
      if (shadow)
         return error_type;
      switch (dim) {
      case GLSL_SAMPLER_DIM_1D:
         return (array ? isampler1DArray_type : isampler1D_type);
      case GLSL_SAMPLER_DIM_2D:
         return (array ? isampler2DArray_type : isampler2D_type);
      case GLSL_SAMPLER_DIM_3D:
         if (array)
            return error_type;
         return isampler3D_type;
      case GLSL_SAMPLER_DIM_CUBE:
         return (array ? isamplerCubeArray_type : isamplerCube_type);
      case GLSL_SAMPLER_DIM_RECT:
         if (array)
            return error_type;
         return isampler2DRect_type;
      case GLSL_SAMPLER_DIM_BUF:
         if (array)
            return error_type;
         return isamplerBuffer_type;
      case GLSL_SAMPLER_DIM_MS:
         return (array ? isampler2DMSArray_type : isampler2DMS_type);
      case GLSL_SAMPLER_DIM_EXTERNAL:
         return error_type;
      case GLSL_SAMPLER_DIM_SUBPASS:
      case GLSL_SAMPLER_DIM_SUBPASS_MS:
         return error_type;
      }
   case GLSL_TYPE_UINT:
      if (shadow)
         return error_type;
      switch (dim) {
      case GLSL_SAMPLER_DIM_1D:
         return (array ? usampler1DArray_type : usampler1D_type);
      case GLSL_SAMPLER_DIM_2D:
         return (array ? usampler2DArray_type : usampler2D_type);
      case GLSL_SAMPLER_DIM_3D:
         if (array)
            return error_type;
         return usampler3D_type;
      case GLSL_SAMPLER_DIM_CUBE:
         return (array ? usamplerCubeArray_type : usamplerCube_type);
      case GLSL_SAMPLER_DIM_RECT:
         if (array)
            return error_type;
         return usampler2DRect_type;
      case GLSL_SAMPLER_DIM_BUF:
         if (array)
            return error_type;
         return usamplerBuffer_type;
      case GLSL_SAMPLER_DIM_MS:
         return (array ? usampler2DMSArray_type : usampler2DMS_type);
      case GLSL_SAMPLER_DIM_EXTERNAL:
         return error_type;
      case GLSL_SAMPLER_DIM_SUBPASS:
      case GLSL_SAMPLER_DIM_SUBPASS_MS:
         return error_type;
      }
   case GLSL_TYPE_VOID:
      return shadow ? samplerShadow_type : sampler_type;
   default:
      return error_type;
   }

   unreachable("switch statement above should be complete");
}

const glsl_type *
glsl_type::get_texture_instance(enum glsl_sampler_dim dim,
                                bool array, glsl_base_type type)
{
   switch (type) {
   case GLSL_TYPE_FLOAT:
      switch (dim) {
      case GLSL_SAMPLER_DIM_1D:
         return (array ? texture1DArray_type : texture1D_type);
      case GLSL_SAMPLER_DIM_2D:
         return (array ? texture2DArray_type : texture2D_type);
      case GLSL_SAMPLER_DIM_3D:
         return texture3D_type;
      case GLSL_SAMPLER_DIM_CUBE:
         return (array ? textureCubeArray_type : textureCube_type);
      case GLSL_SAMPLER_DIM_RECT:
         if (array)
            return error_type;
         else
            return texture2DRect_type;
      case GLSL_SAMPLER_DIM_BUF:
         if (array)
            return error_type;
         else
            return textureBuffer_type;
      case GLSL_SAMPLER_DIM_MS:
         return (array ? texture2DMSArray_type : texture2DMS_type);
      case GLSL_SAMPLER_DIM_SUBPASS:
         return textureSubpassInput_type;
      case GLSL_SAMPLER_DIM_SUBPASS_MS:
         return textureSubpassInputMS_type;
      case GLSL_SAMPLER_DIM_EXTERNAL:
         if (array)
            return error_type;
         else
            return textureExternalOES_type;
      }
   case GLSL_TYPE_INT:
      switch (dim) {
      case GLSL_SAMPLER_DIM_1D:
         return (array ? itexture1DArray_type : itexture1D_type);
      case GLSL_SAMPLER_DIM_2D:
         return (array ? itexture2DArray_type : itexture2D_type);
      case GLSL_SAMPLER_DIM_3D:
         if (array)
            return error_type;
         return itexture3D_type;
      case GLSL_SAMPLER_DIM_CUBE:
         return (array ? itextureCubeArray_type : itextureCube_type);
      case GLSL_SAMPLER_DIM_RECT:
         if (array)
            return error_type;
         return itexture2DRect_type;
      case GLSL_SAMPLER_DIM_BUF:
         if (array)
            return error_type;
         return itextureBuffer_type;
      case GLSL_SAMPLER_DIM_MS:
         return (array ? itexture2DMSArray_type : itexture2DMS_type);
      case GLSL_SAMPLER_DIM_SUBPASS:
         return itextureSubpassInput_type;
      case GLSL_SAMPLER_DIM_SUBPASS_MS:
         return itextureSubpassInputMS_type;
      case GLSL_SAMPLER_DIM_EXTERNAL:
         return error_type;
      }
   case GLSL_TYPE_UINT:
      switch (dim) {
      case GLSL_SAMPLER_DIM_1D:
         return (array ? utexture1DArray_type : utexture1D_type);
      case GLSL_SAMPLER_DIM_2D:
         return (array ? utexture2DArray_type : utexture2D_type);
      case GLSL_SAMPLER_DIM_3D:
         if (array)
            return error_type;
         return utexture3D_type;
      case GLSL_SAMPLER_DIM_CUBE:
         return (array ? utextureCubeArray_type : utextureCube_type);
      case GLSL_SAMPLER_DIM_RECT:
         if (array)
            return error_type;
         return utexture2DRect_type;
      case GLSL_SAMPLER_DIM_BUF:
         if (array)
            return error_type;
         return utextureBuffer_type;
      case GLSL_SAMPLER_DIM_MS:
         return (array ? utexture2DMSArray_type : utexture2DMS_type);
      case GLSL_SAMPLER_DIM_SUBPASS:
         return utextureSubpassInput_type;
      case GLSL_SAMPLER_DIM_SUBPASS_MS:
         return utextureSubpassInputMS_type;
      case GLSL_SAMPLER_DIM_EXTERNAL:
         return error_type;
      }
   case GLSL_TYPE_VOID:
      switch (dim) {
      case GLSL_SAMPLER_DIM_1D:
         return (array ? vtexture1DArray_type : vtexture1D_type);
      case GLSL_SAMPLER_DIM_2D:
         return (array ? vtexture2DArray_type : vtexture2D_type);
      case GLSL_SAMPLER_DIM_3D:
         return (array ? error_type : vtexture3D_type);
      case GLSL_SAMPLER_DIM_BUF:
         return (array ? error_type : vtextureBuffer_type);
      default:
         return error_type;
      }
   default:
      return error_type;
   }

   unreachable("switch statement above should be complete");
}

const glsl_type *
glsl_type::get_image_instance(enum glsl_sampler_dim dim,
                              bool array, glsl_base_type type)
{
   switch (type) {
   case GLSL_TYPE_FLOAT:
      switch (dim) {
      case GLSL_SAMPLER_DIM_1D:
         return (array ? image1DArray_type : image1D_type);
      case GLSL_SAMPLER_DIM_2D:
         return (array ? image2DArray_type : image2D_type);
      case GLSL_SAMPLER_DIM_3D:
         return image3D_type;
      case GLSL_SAMPLER_DIM_CUBE:
         return (array ? imageCubeArray_type : imageCube_type);
      case GLSL_SAMPLER_DIM_RECT:
         if (array)
            return error_type;
         else
            return image2DRect_type;
      case GLSL_SAMPLER_DIM_BUF:
         if (array)
            return error_type;
         else
            return imageBuffer_type;
      case GLSL_SAMPLER_DIM_MS:
         return (array ? image2DMSArray_type : image2DMS_type);
      case GLSL_SAMPLER_DIM_SUBPASS:
         return subpassInput_type;
      case GLSL_SAMPLER_DIM_SUBPASS_MS:
         return subpassInputMS_type;
      case GLSL_SAMPLER_DIM_EXTERNAL:
         return error_type;
      }
   case GLSL_TYPE_INT:
      switch (dim) {
      case GLSL_SAMPLER_DIM_1D:
         return (array ? iimage1DArray_type : iimage1D_type);
      case GLSL_SAMPLER_DIM_2D:
         return (array ? iimage2DArray_type : iimage2D_type);
      case GLSL_SAMPLER_DIM_3D:
         if (array)
            return error_type;
         return iimage3D_type;
      case GLSL_SAMPLER_DIM_CUBE:
         return (array ? iimageCubeArray_type : iimageCube_type);
      case GLSL_SAMPLER_DIM_RECT:
         if (array)
            return error_type;
         return iimage2DRect_type;
      case GLSL_SAMPLER_DIM_BUF:
         if (array)
            return error_type;
         return iimageBuffer_type;
      case GLSL_SAMPLER_DIM_MS:
         return (array ? iimage2DMSArray_type : iimage2DMS_type);
      case GLSL_SAMPLER_DIM_SUBPASS:
         return isubpassInput_type;
      case GLSL_SAMPLER_DIM_SUBPASS_MS:
         return isubpassInputMS_type;
      case GLSL_SAMPLER_DIM_EXTERNAL:
         return error_type;
      }
   case GLSL_TYPE_UINT:
      switch (dim) {
      case GLSL_SAMPLER_DIM_1D:
         return (array ? uimage1DArray_type : uimage1D_type);
      case GLSL_SAMPLER_DIM_2D:
         return (array ? uimage2DArray_type : uimage2D_type);
      case GLSL_SAMPLER_DIM_3D:
         if (array)
            return error_type;
         return uimage3D_type;
      case GLSL_SAMPLER_DIM_CUBE:
         return (array ? uimageCubeArray_type : uimageCube_type);
      case GLSL_SAMPLER_DIM_RECT:
         if (array)
            return error_type;
         return uimage2DRect_type;
      case GLSL_SAMPLER_DIM_BUF:
         if (array)
            return error_type;
         return uimageBuffer_type;
      case GLSL_SAMPLER_DIM_MS:
         return (array ? uimage2DMSArray_type : uimage2DMS_type);
      case GLSL_SAMPLER_DIM_SUBPASS:
         return usubpassInput_type;
      case GLSL_SAMPLER_DIM_SUBPASS_MS:
         return usubpassInputMS_type;
      case GLSL_SAMPLER_DIM_EXTERNAL:
         return error_type;
      }
   case GLSL_TYPE_INT64:
      switch (dim) {
      case GLSL_SAMPLER_DIM_1D:
         return (array ? i64image1DArray_type : i64image1D_type);
      case GLSL_SAMPLER_DIM_2D:
         return (array ? i64image2DArray_type : i64image2D_type);
      case GLSL_SAMPLER_DIM_3D:
         if (array)
            return error_type;
         return i64image3D_type;
      case GLSL_SAMPLER_DIM_CUBE:
         return (array ? i64imageCubeArray_type : i64imageCube_type);
      case GLSL_SAMPLER_DIM_RECT:
         if (array)
            return error_type;
         return i64image2DRect_type;
      case GLSL_SAMPLER_DIM_BUF:
         if (array)
            return error_type;
         return i64imageBuffer_type;
      case GLSL_SAMPLER_DIM_MS:
         return (array ? i64image2DMSArray_type : i64image2DMS_type);
      case GLSL_SAMPLER_DIM_SUBPASS:
      case GLSL_SAMPLER_DIM_SUBPASS_MS:
      case GLSL_SAMPLER_DIM_EXTERNAL:
         return error_type;
      }
   case GLSL_TYPE_UINT64:
      switch (dim) {
      case GLSL_SAMPLER_DIM_1D:
         return (array ? u64image1DArray_type : u64image1D_type);
      case GLSL_SAMPLER_DIM_2D:
         return (array ? u64image2DArray_type : u64image2D_type);
      case GLSL_SAMPLER_DIM_3D:
         if (array)
            return error_type;
         return u64image3D_type;
      case GLSL_SAMPLER_DIM_CUBE:
         return (array ? u64imageCubeArray_type : u64imageCube_type);
      case GLSL_SAMPLER_DIM_RECT:
         if (array)
            return error_type;
         return u64image2DRect_type;
      case GLSL_SAMPLER_DIM_BUF:
         if (array)
            return error_type;
         return u64imageBuffer_type;
      case GLSL_SAMPLER_DIM_MS:
         return (array ? u64image2DMSArray_type : u64image2DMS_type);
      case GLSL_SAMPLER_DIM_SUBPASS:
      case GLSL_SAMPLER_DIM_SUBPASS_MS:
      case GLSL_SAMPLER_DIM_EXTERNAL:
         return error_type;
      }
   case GLSL_TYPE_VOID:
      switch (dim) {
      case GLSL_SAMPLER_DIM_1D:
         return (array ? vimage1DArray_type : vimage1D_type);
      case GLSL_SAMPLER_DIM_2D:
         return (array ? vimage2DArray_type : vimage2D_type);
      case GLSL_SAMPLER_DIM_3D:
         return (array ? error_type : vimage3D_type);
      case GLSL_SAMPLER_DIM_BUF:
         return (array ? error_type : vbuffer_type);
      default:
         return error_type;
      }
   default:
      return error_type;
   }

   unreachable("switch statement above should be complete");
}

const glsl_type *
glsl_type::get_array_instance(const glsl_type *base,
                              unsigned array_size,
                              unsigned explicit_stride)
{
   /* Generate a name using the base type pointer in the key.  This is
    * done because the name of the base type may not be unique across
    * shaders.  For example, two shaders may have different record types
    * named 'foo'.
    */
   char key[128];
   snprintf(key, sizeof(key), "%p[%u]x%uB", (void *) base, array_size,
            explicit_stride);

   simple_mtx_lock(&glsl_type::hash_mutex);
   assert(glsl_type_users > 0);

   if (array_types == NULL) {
      array_types = _mesa_hash_table_create(NULL, _mesa_hash_string,
                                            _mesa_key_string_equal);
   }

   const struct hash_entry *entry = _mesa_hash_table_search(array_types, key);
   if (entry == NULL) {
      const glsl_type *t = new glsl_type(base, array_size, explicit_stride);

      entry = _mesa_hash_table_insert(array_types,
                                      strdup(key),
                                      (void *) t);
   }

   assert(((glsl_type *) entry->data)->base_type == GLSL_TYPE_ARRAY);
   assert(((glsl_type *) entry->data)->length == array_size);
   assert(((glsl_type *) entry->data)->fields.array == base);

   glsl_type *t = (glsl_type *) entry->data;

   simple_mtx_unlock(&glsl_type::hash_mutex);

   return t;
}

bool
glsl_type::compare_no_precision(const glsl_type *b) const
{
   if (this == b)
      return true;

   if (this->is_array()) {
      if (!b->is_array() || this->length != b->length)
         return false;

      const glsl_type *b_no_array = b->fields.array;

      return this->fields.array->compare_no_precision(b_no_array);
   }

   if (this->is_struct()) {
      if (!b->is_struct())
         return false;
   } else if (this->is_interface()) {
      if (!b->is_interface())
         return false;
   } else {
      return false;
   }

   return record_compare(b,
                         true, /* match_name */
                         true, /* match_locations */
                         false /* match_precision */);
}

bool
glsl_type::record_compare(const glsl_type *b, bool match_name,
                          bool match_locations, bool match_precision) const
{
   if (this->length != b->length)
      return false;

   if (this->interface_packing != b->interface_packing)
      return false;

   if (this->interface_row_major != b->interface_row_major)
      return false;

   if (this->explicit_alignment != b->explicit_alignment)
      return false;

   if (this->packed != b->packed)
      return false;

   /* From the GLSL 4.20 specification (Sec 4.2):
    *
    *     "Structures must have the same name, sequence of type names, and
    *     type definitions, and field names to be considered the same type."
    *
    * GLSL ES behaves the same (Ver 1.00 Sec 4.2.4, Ver 3.00 Sec 4.2.5).
    *
    * Section 7.4.1 (Shader Interface Matching) of the OpenGL 4.30 spec says:
    *
    *     "Variables or block members declared as structures are considered
    *     to match in type if and only if structure members match in name,
    *     type, qualification, and declaration order."
    */
   if (match_name)
      if (strcmp(this->name, b->name) != 0)
         return false;

   for (unsigned i = 0; i < this->length; i++) {
      if (match_precision) {
         if (this->fields.structure[i].type != b->fields.structure[i].type)
            return false;
      } else {
         const glsl_type *ta = this->fields.structure[i].type;
         const glsl_type *tb = b->fields.structure[i].type;
         if (!ta->compare_no_precision(tb))
            return false;
      }
      if (strcmp(this->fields.structure[i].name,
                 b->fields.structure[i].name) != 0)
         return false;
      if (this->fields.structure[i].matrix_layout
         != b->fields.structure[i].matrix_layout)
        return false;
      if (match_locations && this->fields.structure[i].location
          != b->fields.structure[i].location)
         return false;
      if (this->fields.structure[i].component
          != b->fields.structure[i].component)
         return false;
      if (this->fields.structure[i].offset
          != b->fields.structure[i].offset)
         return false;
      if (this->fields.structure[i].interpolation
          != b->fields.structure[i].interpolation)
         return false;
      if (this->fields.structure[i].centroid
          != b->fields.structure[i].centroid)
         return false;
      if (this->fields.structure[i].sample
          != b->fields.structure[i].sample)
         return false;
      if (this->fields.structure[i].patch
          != b->fields.structure[i].patch)
         return false;
      if (this->fields.structure[i].memory_read_only
          != b->fields.structure[i].memory_read_only)
         return false;
      if (this->fields.structure[i].memory_write_only
          != b->fields.structure[i].memory_write_only)
         return false;
      if (this->fields.structure[i].memory_coherent
          != b->fields.structure[i].memory_coherent)
         return false;
      if (this->fields.structure[i].memory_volatile
          != b->fields.structure[i].memory_volatile)
         return false;
      if (this->fields.structure[i].memory_restrict
          != b->fields.structure[i].memory_restrict)
         return false;
      if (this->fields.structure[i].image_format
          != b->fields.structure[i].image_format)
         return false;
      if (match_precision &&
          this->fields.structure[i].precision
          != b->fields.structure[i].precision)
         return false;
      if (this->fields.structure[i].explicit_xfb_buffer
          != b->fields.structure[i].explicit_xfb_buffer)
         return false;
      if (this->fields.structure[i].xfb_buffer
          != b->fields.structure[i].xfb_buffer)
         return false;
      if (this->fields.structure[i].xfb_stride
          != b->fields.structure[i].xfb_stride)
         return false;
   }

   return true;
}


bool
glsl_type::record_key_compare(const void *a, const void *b)
{
   const glsl_type *const key1 = (glsl_type *) a;
   const glsl_type *const key2 = (glsl_type *) b;

   return strcmp(key1->name, key2->name) == 0 &&
                 key1->record_compare(key2, true);
}


/**
 * Generate an integer hash value for a glsl_type structure type.
 */
unsigned
glsl_type::record_key_hash(const void *a)
{
   const glsl_type *const key = (glsl_type *) a;
   uintptr_t hash = key->length;
   unsigned retval;

   for (unsigned i = 0; i < key->length; i++) {
      /* casting pointer to uintptr_t */
      hash = (hash * 13 ) + (uintptr_t) key->fields.structure[i].type;
   }

   if (sizeof(hash) == 8)
      retval = (hash & 0xffffffff) ^ ((uint64_t) hash >> 32);
   else
      retval = hash;

   return retval;
}


const glsl_type *
glsl_type::get_struct_instance(const glsl_struct_field *fields,
                               unsigned num_fields,
                               const char *name,
                               bool packed, unsigned explicit_alignment)
{
   const glsl_type key(fields, num_fields, name, packed, explicit_alignment);

   simple_mtx_lock(&glsl_type::hash_mutex);
   assert(glsl_type_users > 0);

   if (struct_types == NULL) {
      struct_types = _mesa_hash_table_create(NULL, record_key_hash,
                                             record_key_compare);
   }

   const struct hash_entry *entry = _mesa_hash_table_search(struct_types,
                                                            &key);
   if (entry == NULL) {
      const glsl_type *t = new glsl_type(fields, num_fields, name, packed,
                                         explicit_alignment);

      entry = _mesa_hash_table_insert(struct_types, t, (void *) t);
   }

   assert(((glsl_type *) entry->data)->base_type == GLSL_TYPE_STRUCT);
   assert(((glsl_type *) entry->data)->length == num_fields);
   assert(strcmp(((glsl_type *) entry->data)->name, name) == 0);
   assert(((glsl_type *) entry->data)->packed == packed);
   assert(((glsl_type *) entry->data)->explicit_alignment == explicit_alignment);

   glsl_type *t = (glsl_type *) entry->data;

   simple_mtx_unlock(&glsl_type::hash_mutex);

   return t;
}


const glsl_type *
glsl_type::get_interface_instance(const glsl_struct_field *fields,
                                  unsigned num_fields,
                                  enum glsl_interface_packing packing,
                                  bool row_major,
                                  const char *block_name)
{
   const glsl_type key(fields, num_fields, packing, row_major, block_name);

   simple_mtx_lock(&glsl_type::hash_mutex);
   assert(glsl_type_users > 0);

   if (interface_types == NULL) {
      interface_types = _mesa_hash_table_create(NULL, record_key_hash,
                                                record_key_compare);
   }

   const struct hash_entry *entry = _mesa_hash_table_search(interface_types,
                                                            &key);
   if (entry == NULL) {
      const glsl_type *t = new glsl_type(fields, num_fields,
                                         packing, row_major, block_name);

      entry = _mesa_hash_table_insert(interface_types, t, (void *) t);
   }

   assert(((glsl_type *) entry->data)->base_type == GLSL_TYPE_INTERFACE);
   assert(((glsl_type *) entry->data)->length == num_fields);
   assert(strcmp(((glsl_type *) entry->data)->name, block_name) == 0);

   glsl_type *t = (glsl_type *) entry->data;

   simple_mtx_unlock(&glsl_type::hash_mutex);

   return t;
}

const glsl_type *
glsl_type::get_subroutine_instance(const char *subroutine_name)
{
   const glsl_type key(subroutine_name);

   simple_mtx_lock(&glsl_type::hash_mutex);
   assert(glsl_type_users > 0);

   if (subroutine_types == NULL) {
      subroutine_types = _mesa_hash_table_create(NULL, record_key_hash,
                                                 record_key_compare);
   }

   const struct hash_entry *entry = _mesa_hash_table_search(subroutine_types,
                                                            &key);
   if (entry == NULL) {
      const glsl_type *t = new glsl_type(subroutine_name);

      entry = _mesa_hash_table_insert(subroutine_types, t, (void *) t);
   }

   assert(((glsl_type *) entry->data)->base_type == GLSL_TYPE_SUBROUTINE);
   assert(strcmp(((glsl_type *) entry->data)->name, subroutine_name) == 0);

   glsl_type *t = (glsl_type *) entry->data;

   simple_mtx_unlock(&glsl_type::hash_mutex);

   return t;
}


static bool
function_key_compare(const void *a, const void *b)
{
   const glsl_type *const key1 = (glsl_type *) a;
   const glsl_type *const key2 = (glsl_type *) b;

   if (key1->length != key2->length)
      return false;

   return memcmp(key1->fields.parameters, key2->fields.parameters,
                 (key1->length + 1) * sizeof(*key1->fields.parameters)) == 0;
}


static uint32_t
function_key_hash(const void *a)
{
   const glsl_type *const key = (glsl_type *) a;
   return _mesa_hash_data(key->fields.parameters,
                          (key->length + 1) * sizeof(*key->fields.parameters));
}

const glsl_type *
glsl_type::get_function_instance(const glsl_type *return_type,
                                 const glsl_function_param *params,
                                 unsigned num_params)
{
   const glsl_type key(return_type, params, num_params);

   simple_mtx_lock(&glsl_type::hash_mutex);
   assert(glsl_type_users > 0);

   if (function_types == NULL) {
      function_types = _mesa_hash_table_create(NULL, function_key_hash,
                                               function_key_compare);
   }

   struct hash_entry *entry = _mesa_hash_table_search(function_types, &key);
   if (entry == NULL) {
      const glsl_type *t = new glsl_type(return_type, params, num_params);

      entry = _mesa_hash_table_insert(function_types, t, (void *) t);
   }

   const glsl_type *t = (const glsl_type *)entry->data;

   assert(t->base_type == GLSL_TYPE_FUNCTION);
   assert(t->length == num_params);

   simple_mtx_unlock(&glsl_type::hash_mutex);

   return t;
}


const glsl_type *
glsl_type::get_mul_type(const glsl_type *type_a, const glsl_type *type_b)
{
   if (type_a->is_matrix() && type_b->is_matrix()) {
      /* Matrix multiply.  The columns of A must match the rows of B.  Given
       * the other previously tested constraints, this means the vector type
       * of a row from A must be the same as the vector type of a column from
       * B.
       */
      if (type_a->row_type() == type_b->column_type()) {
         /* The resulting matrix has the number of columns of matrix B and
          * the number of rows of matrix A.  We get the row count of A by
          * looking at the size of a vector that makes up a column.  The
          * transpose (size of a row) is done for B.
          */
         const glsl_type *const type =
            get_instance(type_a->base_type,
                         type_a->column_type()->vector_elements,
                         type_b->row_type()->vector_elements);
         assert(type != error_type);

         return type;
      }
   } else if (type_a == type_b) {
      return type_a;
   } else if (type_a->is_matrix()) {
      /* A is a matrix and B is a column vector.  Columns of A must match
       * rows of B.  Given the other previously tested constraints, this
       * means the vector type of a row from A must be the same as the
       * vector the type of B.
       */
      if (type_a->row_type() == type_b) {
         /* The resulting vector has a number of elements equal to
          * the number of rows of matrix A. */
         const glsl_type *const type =
            get_instance(type_a->base_type,
                         type_a->column_type()->vector_elements,
                         1);
         assert(type != error_type);

         return type;
      }
   } else {
      assert(type_b->is_matrix());

      /* A is a row vector and B is a matrix.  Columns of A must match rows
       * of B.  Given the other previously tested constraints, this means
       * the type of A must be the same as the vector type of a column from
       * B.
       */
      if (type_a == type_b->column_type()) {
         /* The resulting vector has a number of elements equal to
          * the number of columns of matrix B. */
         const glsl_type *const type =
            get_instance(type_a->base_type,
                         type_b->row_type()->vector_elements,
                         1);
         assert(type != error_type);

         return type;
      }
   }

   return error_type;
}


const glsl_type *
glsl_type::field_type(const char *name) const
{
   if (this->base_type != GLSL_TYPE_STRUCT
       && this->base_type != GLSL_TYPE_INTERFACE)
      return error_type;

   for (unsigned i = 0; i < this->length; i++) {
      if (strcmp(name, this->fields.structure[i].name) == 0)
         return this->fields.structure[i].type;
   }

   return error_type;
}


int
glsl_type::field_index(const char *name) const
{
   if (this->base_type != GLSL_TYPE_STRUCT
       && this->base_type != GLSL_TYPE_INTERFACE)
      return -1;

   for (unsigned i = 0; i < this->length; i++) {
      if (strcmp(name, this->fields.structure[i].name) == 0)
         return i;
   }

   return -1;
}


unsigned
glsl_type::component_slots() const
{
   switch (this->base_type) {
   case GLSL_TYPE_UINT:
   case GLSL_TYPE_INT:
   case GLSL_TYPE_UINT8:
   case GLSL_TYPE_INT8:
   case GLSL_TYPE_UINT16:
   case GLSL_TYPE_INT16:
   case GLSL_TYPE_FLOAT:
   case GLSL_TYPE_FLOAT16:
   case GLSL_TYPE_BOOL:
      return this->components();

   case GLSL_TYPE_DOUBLE:
   case GLSL_TYPE_UINT64:
   case GLSL_TYPE_INT64:
      return 2 * this->components();

   case GLSL_TYPE_STRUCT:
   case GLSL_TYPE_INTERFACE: {
      unsigned size = 0;

      for (unsigned i = 0; i < this->length; i++)
         size += this->fields.structure[i].type->component_slots();

      return size;
   }

   case GLSL_TYPE_ARRAY:
      return this->length * this->fields.array->component_slots();

   case GLSL_TYPE_SAMPLER:
   case GLSL_TYPE_TEXTURE:
   case GLSL_TYPE_IMAGE:
      return 2;

   case GLSL_TYPE_SUBROUTINE:
      return 1;

   case GLSL_TYPE_FUNCTION:
   case GLSL_TYPE_ATOMIC_UINT:
   case GLSL_TYPE_VOID:
   case GLSL_TYPE_ERROR:
      break;
   }

   return 0;
}

unsigned
glsl_type::component_slots_aligned(unsigned offset) const
{
   /* Align 64bit type only if it crosses attribute slot boundary. */
   switch (this->base_type) {
   case GLSL_TYPE_UINT:
   case GLSL_TYPE_INT:
   case GLSL_TYPE_UINT8:
   case GLSL_TYPE_INT8:
   case GLSL_TYPE_UINT16:
   case GLSL_TYPE_INT16:
   case GLSL_TYPE_FLOAT:
   case GLSL_TYPE_FLOAT16:
   case GLSL_TYPE_BOOL:
      return this->components();

   case GLSL_TYPE_DOUBLE:
   case GLSL_TYPE_UINT64:
   case GLSL_TYPE_INT64: {
      unsigned size = 2 * this->components();
      if (offset % 2 == 1 && (offset % 4 + size) > 4) {
         size++;
      }

      return size;
   }

   case GLSL_TYPE_STRUCT:
   case GLSL_TYPE_INTERFACE: {
      unsigned size = 0;

      for (unsigned i = 0; i < this->length; i++) {
         const glsl_type *member = this->fields.structure[i].type;
         size += member->component_slots_aligned(size + offset);
      }

      return size;
   }

   case GLSL_TYPE_ARRAY: {
      unsigned size = 0;

      for (unsigned i = 0; i < this->length; i++) {
         size += this->fields.array->component_slots_aligned(size + offset);
      }

      return size;
   }

   case GLSL_TYPE_SAMPLER:
   case GLSL_TYPE_TEXTURE:
   case GLSL_TYPE_IMAGE:
      return 2 + ((offset % 4) == 3 ? 1 : 0);

   case GLSL_TYPE_SUBROUTINE:
      return 1;

   case GLSL_TYPE_FUNCTION:
   case GLSL_TYPE_ATOMIC_UINT:
   case GLSL_TYPE_VOID:
   case GLSL_TYPE_ERROR:
      break;
   }

   return 0;
}

unsigned
glsl_type::struct_location_offset(unsigned length) const
{
   unsigned offset = 0;
   const glsl_type *t = this->without_array();
   if (t->is_struct()) {
      assert(length <= t->length);

      for (unsigned i = 0; i < length; i++) {
         const glsl_type *st = t->fields.structure[i].type;
         const glsl_type *wa = st->without_array();
         if (wa->is_struct()) {
            unsigned r_offset = wa->struct_location_offset(wa->length);
            offset += st->is_array() ?
               st->arrays_of_arrays_size() * r_offset : r_offset;
         } else if (st->is_array() && st->fields.array->is_array()) {
            unsigned outer_array_size = st->length;
            const glsl_type *base_type = st->fields.array;

            /* For arrays of arrays the outer arrays take up a uniform
             * slot for each element. The innermost array elements share a
             * single slot so we ignore the innermost array when calculating
             * the offset.
             */
            while (base_type->fields.array->is_array()) {
               outer_array_size = outer_array_size * base_type->length;
               base_type = base_type->fields.array;
            }
            offset += outer_array_size;
         } else {
            /* We dont worry about arrays here because unless the array
             * contains a structure or another array it only takes up a single
             * uniform slot.
             */
            offset += 1;
         }
      }
   }
   return offset;
}

unsigned
glsl_type::uniform_locations() const
{
   unsigned size = 0;

   switch (this->base_type) {
   case GLSL_TYPE_UINT:
   case GLSL_TYPE_INT:
   case GLSL_TYPE_FLOAT:
   case GLSL_TYPE_FLOAT16:
   case GLSL_TYPE_DOUBLE:
   case GLSL_TYPE_UINT16:
   case GLSL_TYPE_UINT8:
   case GLSL_TYPE_INT16:
   case GLSL_TYPE_INT8:
   case GLSL_TYPE_UINT64:
   case GLSL_TYPE_INT64:
   case GLSL_TYPE_BOOL:
   case GLSL_TYPE_SAMPLER:
   case GLSL_TYPE_TEXTURE:
   case GLSL_TYPE_IMAGE:
   case GLSL_TYPE_SUBROUTINE:
      return 1;

   case GLSL_TYPE_STRUCT:
   case GLSL_TYPE_INTERFACE:
      for (unsigned i = 0; i < this->length; i++)
         size += this->fields.structure[i].type->uniform_locations();
      return size;
   case GLSL_TYPE_ARRAY:
      return this->length * this->fields.array->uniform_locations();
   default:
      return 0;
   }
}

unsigned
glsl_type::varying_count() const
{
   unsigned size = 0;

   switch (this->base_type) {
   case GLSL_TYPE_UINT:
   case GLSL_TYPE_INT:
   case GLSL_TYPE_FLOAT:
   case GLSL_TYPE_FLOAT16:
   case GLSL_TYPE_DOUBLE:
   case GLSL_TYPE_BOOL:
   case GLSL_TYPE_UINT16:
   case GLSL_TYPE_UINT8:
   case GLSL_TYPE_INT16:
   case GLSL_TYPE_INT8:
   case GLSL_TYPE_UINT64:
   case GLSL_TYPE_INT64:
      return 1;

   case GLSL_TYPE_STRUCT:
   case GLSL_TYPE_INTERFACE:
      for (unsigned i = 0; i < this->length; i++)
         size += this->fields.structure[i].type->varying_count();
      return size;
   case GLSL_TYPE_ARRAY:
      /* Don't count innermost array elements */
      if (this->without_array()->is_struct() ||
          this->without_array()->is_interface() ||
          this->fields.array->is_array())
         return this->length * this->fields.array->varying_count();
      else
         return this->fields.array->varying_count();
   default:
      assert(!"unsupported varying type");
      return 0;
   }
}

bool
glsl_type::can_implicitly_convert_to(const glsl_type *desired,
                                     _mesa_glsl_parse_state *state) const
{
   if (this == desired)
      return true;

   /* GLSL 1.10 and ESSL do not allow implicit conversions. If there is no
    * state, we're doing intra-stage function linking where these checks have
    * already been done.
    */
   if (state && !state->has_implicit_conversions())
      return false;

   /* There is no conversion among matrix types. */
   if (this->matrix_columns > 1 || desired->matrix_columns > 1)
      return false;

   /* Vector size must match. */
   if (this->vector_elements != desired->vector_elements)
      return false;

   /* int and uint can be converted to float. */
   if (desired->is_float() && this->is_integer_32())
      return true;

   /* With GLSL 4.0, ARB_gpu_shader5, or MESA_shader_integer_functions, int
    * can be converted to uint.  Note that state may be NULL here, when
    * resolving function calls in the linker. By this time, all the
    * state-dependent checks have already happened though, so allow anything
    * that's allowed in any shader version.
    */
   if ((!state || state->has_implicit_int_to_uint_conversion()) &&
         desired->base_type == GLSL_TYPE_UINT && this->base_type == GLSL_TYPE_INT)
      return true;

   /* No implicit conversions from double. */
   if ((!state || state->has_double()) && this->is_double())
      return false;

   /* Conversions from different types to double. */
   if ((!state || state->has_double()) && desired->is_double()) {
      if (this->is_float())
         return true;
      if (this->is_integer_32())
         return true;
   }

   return false;
}

unsigned
glsl_type::std140_base_alignment(bool row_major) const
{
   unsigned N = is_64bit() ? 8 : 4;

   /* (1) If the member is a scalar consuming <N> basic machine units, the
    *     base alignment is <N>.
    *
    * (2) If the member is a two- or four-component vector with components
    *     consuming <N> basic machine units, the base alignment is 2<N> or
    *     4<N>, respectively.
    *
    * (3) If the member is a three-component vector with components consuming
    *     <N> basic machine units, the base alignment is 4<N>.
    */
   if (this->is_scalar() || this->is_vector()) {
      switch (this->vector_elements) {
      case 1:
         return N;
      case 2:
         return 2 * N;
      case 3:
      case 4:
         return 4 * N;
      }
   }

   /* (4) If the member is an array of scalars or vectors, the base alignment
    *     and array stride are set to match the base alignment of a single
    *     array element, according to rules (1), (2), and (3), and rounded up
    *     to the base alignment of a vec4. The array may have padding at the
    *     end; the base offset of the member following the array is rounded up
    *     to the next multiple of the base alignment.
    *
    * (6) If the member is an array of <S> column-major matrices with <C>
    *     columns and <R> rows, the matrix is stored identically to a row of
    *     <S>*<C> column vectors with <R> components each, according to rule
    *     (4).
    *
    * (8) If the member is an array of <S> row-major matrices with <C> columns
    *     and <R> rows, the matrix is stored identically to a row of <S>*<R>
    *     row vectors with <C> components each, according to rule (4).
    *
    * (10) If the member is an array of <S> structures, the <S> elements of
    *      the array are laid out in order, according to rule (9).
    */
   if (this->is_array()) {
      if (this->fields.array->is_scalar() ||
          this->fields.array->is_vector() ||
          this->fields.array->is_matrix()) {
         return MAX2(this->fields.array->std140_base_alignment(row_major), 16);
      } else {
         assert(this->fields.array->is_struct() ||
                this->fields.array->is_array());
         return this->fields.array->std140_base_alignment(row_major);
      }
   }

   /* (5) If the member is a column-major matrix with <C> columns and
    *     <R> rows, the matrix is stored identically to an array of
    *     <C> column vectors with <R> components each, according to
    *     rule (4).
    *
    * (7) If the member is a row-major matrix with <C> columns and <R>
    *     rows, the matrix is stored identically to an array of <R>
    *     row vectors with <C> components each, according to rule (4).
    */
   if (this->is_matrix()) {
      const struct glsl_type *vec_type, *array_type;
      int c = this->matrix_columns;
      int r = this->vector_elements;

      if (row_major) {
         vec_type = get_instance(base_type, c, 1);
         array_type = glsl_type::get_array_instance(vec_type, r);
      } else {
         vec_type = get_instance(base_type, r, 1);
         array_type = glsl_type::get_array_instance(vec_type, c);
      }

      return array_type->std140_base_alignment(false);
   }

   /* (9) If the member is a structure, the base alignment of the
    *     structure is <N>, where <N> is the largest base alignment
    *     value of any of its members, and rounded up to the base
    *     alignment of a vec4. The individual members of this
    *     sub-structure are then assigned offsets by applying this set
    *     of rules recursively, where the base offset of the first
    *     member of the sub-structure is equal to the aligned offset
    *     of the structure. The structure may have padding at the end;
    *     the base offset of the member following the sub-structure is
    *     rounded up to the next multiple of the base alignment of the
    *     structure.
    */
   if (this->is_struct()) {
      unsigned base_alignment = 16;
      for (unsigned i = 0; i < this->length; i++) {
         bool field_row_major = row_major;
         const enum glsl_matrix_layout matrix_layout =
            glsl_matrix_layout(this->fields.structure[i].matrix_layout);
         if (matrix_layout == GLSL_MATRIX_LAYOUT_ROW_MAJOR) {
            field_row_major = true;
         } else if (matrix_layout == GLSL_MATRIX_LAYOUT_COLUMN_MAJOR) {
            field_row_major = false;
         }

         const struct glsl_type *field_type = this->fields.structure[i].type;
         base_alignment = MAX2(base_alignment,
                               field_type->std140_base_alignment(field_row_major));
      }
      return base_alignment;
   }

   assert(!"not reached");
   return -1;
}

unsigned
glsl_type::std140_size(bool row_major) const
{
   unsigned N = is_64bit() ? 8 : 4;

   /* (1) If the member is a scalar consuming <N> basic machine units, the
    *     base alignment is <N>.
    *
    * (2) If the member is a two- or four-component vector with components
    *     consuming <N> basic machine units, the base alignment is 2<N> or
    *     4<N>, respectively.
    *
    * (3) If the member is a three-component vector with components consuming
    *     <N> basic machine units, the base alignment is 4<N>.
    */
   if (this->is_scalar() || this->is_vector()) {
      assert(this->explicit_stride == 0);
      return this->vector_elements * N;
   }

   /* (5) If the member is a column-major matrix with <C> columns and
    *     <R> rows, the matrix is stored identically to an array of
    *     <C> column vectors with <R> components each, according to
    *     rule (4).
    *
    * (6) If the member is an array of <S> column-major matrices with <C>
    *     columns and <R> rows, the matrix is stored identically to a row of
    *     <S>*<C> column vectors with <R> components each, according to rule
    *     (4).
    *
    * (7) If the member is a row-major matrix with <C> columns and <R>
    *     rows, the matrix is stored identically to an array of <R>
    *     row vectors with <C> components each, according to rule (4).
    *
    * (8) If the member is an array of <S> row-major matrices with <C> columns
    *     and <R> rows, the matrix is stored identically to a row of <S>*<R>
    *     row vectors with <C> components each, according to rule (4).
    */
   if (this->without_array()->is_matrix()) {
      const struct glsl_type *element_type;
      const struct glsl_type *vec_type;
      unsigned int array_len;

      if (this->is_array()) {
         element_type = this->without_array();
         array_len = this->arrays_of_arrays_size();
      } else {
         element_type = this;
         array_len = 1;
      }

      if (row_major) {
         vec_type = get_instance(element_type->base_type,
                                 element_type->matrix_columns, 1);

         array_len *= element_type->vector_elements;
      } else {
         vec_type = get_instance(element_type->base_type,
                                 element_type->vector_elements, 1);
         array_len *= element_type->matrix_columns;
      }
      const glsl_type *array_type = glsl_type::get_array_instance(vec_type,
                                                                  array_len);

      return array_type->std140_size(false);
   }

   /* (4) If the member is an array of scalars or vectors, the base alignment
    *     and array stride are set to match the base alignment of a single
    *     array element, according to rules (1), (2), and (3), and rounded up
    *     to the base alignment of a vec4. The array may have padding at the
    *     end; the base offset of the member following the array is rounded up
    *     to the next multiple of the base alignment.
    *
    * (10) If the member is an array of <S> structures, the <S> elements of
    *      the array are laid out in order, according to rule (9).
    */
   if (this->is_array()) {
      unsigned stride;
      if (this->without_array()->is_struct()) {
	 stride = this->without_array()->std140_size(row_major);
      } else {
	 unsigned element_base_align =
	    this->without_array()->std140_base_alignment(row_major);
         stride = MAX2(element_base_align, 16);
      }

      unsigned size = this->arrays_of_arrays_size() * stride;
      assert(this->explicit_stride == 0 ||
             size == this->length * this->explicit_stride);
      return size;
   }

   /* (9) If the member is a structure, the base alignment of the
    *     structure is <N>, where <N> is the largest base alignment
    *     value of any of its members, and rounded up to the base
    *     alignment of a vec4. The individual members of this
    *     sub-structure are then assigned offsets by applying this set
    *     of rules recursively, where the base offset of the first
    *     member of the sub-structure is equal to the aligned offset
    *     of the structure. The structure may have padding at the end;
    *     the base offset of the member following the sub-structure is
    *     rounded up to the next multiple of the base alignment of the
    *     structure.
    */
   if (this->is_struct() || this->is_interface()) {
      unsigned size = 0;
      unsigned max_align = 0;

      for (unsigned i = 0; i < this->length; i++) {
         bool field_row_major = row_major;
         const enum glsl_matrix_layout matrix_layout =
            glsl_matrix_layout(this->fields.structure[i].matrix_layout);
         if (matrix_layout == GLSL_MATRIX_LAYOUT_ROW_MAJOR) {
            field_row_major = true;
         } else if (matrix_layout == GLSL_MATRIX_LAYOUT_COLUMN_MAJOR) {
            field_row_major = false;
         }

         const struct glsl_type *field_type = this->fields.structure[i].type;
         unsigned align = field_type->std140_base_alignment(field_row_major);

         /* Ignore unsized arrays when calculating size */
         if (field_type->is_unsized_array())
            continue;

         size = glsl_align(size, align);
         size += field_type->std140_size(field_row_major);

         max_align = MAX2(align, max_align);

         if (field_type->is_struct() && (i + 1 < this->length))
            size = glsl_align(size, 16);
      }
      size = glsl_align(size, MAX2(max_align, 16));
      return size;
   }

   assert(!"not reached");
   return -1;
}

const glsl_type *
glsl_type::get_explicit_std140_type(bool row_major) const
{
   if (this->is_vector() || this->is_scalar()) {
      return this;
   } else if (this->is_matrix()) {
      const glsl_type *vec_type;
      if (row_major)
         vec_type = get_instance(this->base_type, this->matrix_columns, 1);
      else
         vec_type = get_instance(this->base_type, this->vector_elements, 1);
      unsigned elem_size = vec_type->std140_size(false);
      unsigned stride = glsl_align(elem_size, 16);
      return get_instance(this->base_type, this->vector_elements,
                          this->matrix_columns, stride, row_major);
   } else if (this->is_array()) {
      unsigned elem_size = this->fields.array->std140_size(row_major);
      const glsl_type *elem_type =
         this->fields.array->get_explicit_std140_type(row_major);
      unsigned stride = glsl_align(elem_size, 16);
      return get_array_instance(elem_type, this->length, stride);
   } else if (this->is_struct() || this->is_interface()) {
      glsl_struct_field *fields = new glsl_struct_field[this->length];
      unsigned offset = 0;
      for (unsigned i = 0; i < length; i++) {
         fields[i] = this->fields.structure[i];

         bool field_row_major = row_major;
         if (fields[i].matrix_layout == GLSL_MATRIX_LAYOUT_COLUMN_MAJOR) {
            field_row_major = false;
         } else if (fields[i].matrix_layout == GLSL_MATRIX_LAYOUT_ROW_MAJOR) {
            field_row_major = true;
         }
         fields[i].type =
            fields[i].type->get_explicit_std140_type(field_row_major);

         unsigned fsize = fields[i].type->std140_size(field_row_major);
         unsigned falign = fields[i].type->std140_base_alignment(field_row_major);
         /* From the GLSL 460 spec section "Uniform and Shader Storage Block
          * Layout Qualifiers":
          *
          *    "The actual offset of a member is computed as follows: If
          *    offset was declared, start with that offset, otherwise start
          *    with the next available offset. If the resulting offset is not
          *    a multiple of the actual alignment, increase it to the first
          *    offset that is a multiple of the actual alignment. This results
          *    in the actual offset the member will have."
          */
         if (fields[i].offset >= 0) {
            assert((unsigned)fields[i].offset >= offset);
            offset = fields[i].offset;
         }
         offset = glsl_align(offset, falign);
         fields[i].offset = offset;
         offset += fsize;
      }

      const glsl_type *type;
      if (this->is_struct())
         type = get_struct_instance(fields, this->length, this->name);
      else
         type = get_interface_instance(fields, this->length,
                                       (enum glsl_interface_packing)this->interface_packing,
                                       this->interface_row_major,
                                       this->name);

      delete[] fields;
      return type;
   } else {
      unreachable("Invalid type for UBO or SSBO");
   }
}

unsigned
glsl_type::std430_base_alignment(bool row_major) const
{

   unsigned N = is_64bit() ? 8 : 4;

   /* (1) If the member is a scalar consuming <N> basic machine units, the
    *     base alignment is <N>.
    *
    * (2) If the member is a two- or four-component vector with components
    *     consuming <N> basic machine units, the base alignment is 2<N> or
    *     4<N>, respectively.
    *
    * (3) If the member is a three-component vector with components consuming
    *     <N> basic machine units, the base alignment is 4<N>.
    */
   if (this->is_scalar() || this->is_vector()) {
      switch (this->vector_elements) {
      case 1:
         return N;
      case 2:
         return 2 * N;
      case 3:
      case 4:
         return 4 * N;
      }
   }

   /* OpenGL 4.30 spec, section 7.6.2.2 "Standard Uniform Block Layout":
    *
    * "When using the std430 storage layout, shader storage blocks will be
    * laid out in buffer storage identically to uniform and shader storage
    * blocks using the std140 layout, except that the base alignment and
    * stride of arrays of scalars and vectors in rule 4 and of structures
    * in rule 9 are not rounded up a multiple of the base alignment of a vec4.
    */

   /* (1) If the member is a scalar consuming <N> basic machine units, the
    *     base alignment is <N>.
    *
    * (2) If the member is a two- or four-component vector with components
    *     consuming <N> basic machine units, the base alignment is 2<N> or
    *     4<N>, respectively.
    *
    * (3) If the member is a three-component vector with components consuming
    *     <N> basic machine units, the base alignment is 4<N>.
    */
   if (this->is_array())
      return this->fields.array->std430_base_alignment(row_major);

   /* (5) If the member is a column-major matrix with <C> columns and
    *     <R> rows, the matrix is stored identically to an array of
    *     <C> column vectors with <R> components each, according to
    *     rule (4).
    *
    * (7) If the member is a row-major matrix with <C> columns and <R>
    *     rows, the matrix is stored identically to an array of <R>
    *     row vectors with <C> components each, according to rule (4).
    */
   if (this->is_matrix()) {
      const struct glsl_type *vec_type, *array_type;
      int c = this->matrix_columns;
      int r = this->vector_elements;

      if (row_major) {
         vec_type = get_instance(base_type, c, 1);
         array_type = glsl_type::get_array_instance(vec_type, r);
      } else {
         vec_type = get_instance(base_type, r, 1);
         array_type = glsl_type::get_array_instance(vec_type, c);
      }

      return array_type->std430_base_alignment(false);
   }

      /* (9) If the member is a structure, the base alignment of the
    *     structure is <N>, where <N> is the largest base alignment
    *     value of any of its members, and rounded up to the base
    *     alignment of a vec4. The individual members of this
    *     sub-structure are then assigned offsets by applying this set
    *     of rules recursively, where the base offset of the first
    *     member of the sub-structure is equal to the aligned offset
    *     of the structure. The structure may have padding at the end;
    *     the base offset of the member following the sub-structure is
    *     rounded up to the next multiple of the base alignment of the
    *     structure.
    */
   if (this->is_struct()) {
      unsigned base_alignment = 0;
      for (unsigned i = 0; i < this->length; i++) {
         bool field_row_major = row_major;
         const enum glsl_matrix_layout matrix_layout =
            glsl_matrix_layout(this->fields.structure[i].matrix_layout);
         if (matrix_layout == GLSL_MATRIX_LAYOUT_ROW_MAJOR) {
            field_row_major = true;
         } else if (matrix_layout == GLSL_MATRIX_LAYOUT_COLUMN_MAJOR) {
            field_row_major = false;
         }

         const struct glsl_type *field_type = this->fields.structure[i].type;
         base_alignment = MAX2(base_alignment,
                               field_type->std430_base_alignment(field_row_major));
      }
      assert(base_alignment > 0);
      return base_alignment;
   }
   assert(!"not reached");
   return -1;
}

unsigned
glsl_type::std430_array_stride(bool row_major) const
{
   unsigned N = is_64bit() ? 8 : 4;

   /* Notice that the array stride of a vec3 is not 3 * N but 4 * N.
    * See OpenGL 4.30 spec, section 7.6.2.2 "Standard Uniform Block Layout"
    *
    * (3) If the member is a three-component vector with components consuming
    *     <N> basic machine units, the base alignment is 4<N>.
    */
   if (this->is_vector() && this->vector_elements == 3)
      return 4 * N;

   /* By default use std430_size(row_major) */
   unsigned stride = this->std430_size(row_major);
   assert(this->explicit_stride == 0 || this->explicit_stride == stride);
   return stride;
}

/* Note that the value returned by this method is only correct if the
 * explit offset, and stride values are set, so only with SPIR-V shaders.
 * Should not be used with GLSL shaders.
 */

unsigned
glsl_type::explicit_size(bool align_to_stride) const
{
   if (this->is_struct() || this->is_interface()) {
      if (this->length > 0) {
         unsigned size = 0;

         for (unsigned i = 0; i < this->length; i++) {
            assert(this->fields.structure[i].offset >= 0);
            unsigned last_byte = this->fields.structure[i].offset +
               this->fields.structure[i].type->explicit_size();
            size = MAX2(size, last_byte);
         }

         return size;
      } else {
         return 0;
      }
   } else if (this->is_array()) {
      /* From ARB_program_interface_query spec:
       *
       *   "For the property of BUFFER_DATA_SIZE, then the implementation-dependent
       *   minimum total buffer object size, in basic machine units, required to
       *   hold all active variables associated with an active uniform block, shader
       *   storage block, or atomic counter buffer is written to <params>.  If the
       *   final member of an active shader storage block is array with no declared
       *   size, the minimum buffer size is computed assuming the array was declared
       *   as an array with one element."
       *
       */
      if (this->is_unsized_array())
         return this->explicit_stride;

      assert(this->length > 0);
      unsigned elem_size = align_to_stride ? this->explicit_stride : this->fields.array->explicit_size();
      assert(this->explicit_stride >= elem_size);

      return this->explicit_stride * (this->length - 1) + elem_size;
   } else if (this->is_matrix()) {
      const struct glsl_type *elem_type;
      unsigned length;

      if (this->interface_row_major) {
         elem_type = get_instance(this->base_type,
                                  this->matrix_columns, 1);
         length = this->vector_elements;
      } else {
         elem_type = get_instance(this->base_type,
                                  this->vector_elements, 1);
         length = this->matrix_columns;
      }

      unsigned elem_size = align_to_stride ? this->explicit_stride : elem_type->explicit_size();

      assert(this->explicit_stride);
      return this->explicit_stride * (length - 1) + elem_size;
   }

   unsigned N = this->bit_size() / 8;

   return this->vector_elements * N;
}

unsigned
glsl_type::std430_size(bool row_major) const
{
   unsigned N = is_64bit() ? 8 : 4;

   /* OpenGL 4.30 spec, section 7.6.2.2 "Standard Uniform Block Layout":
    *
    * "When using the std430 storage layout, shader storage blocks will be
    * laid out in buffer storage identically to uniform and shader storage
    * blocks using the std140 layout, except that the base alignment and
    * stride of arrays of scalars and vectors in rule 4 and of structures
    * in rule 9 are not rounded up a multiple of the base alignment of a vec4.
    */
   if (this->is_scalar() || this->is_vector()) {
      assert(this->explicit_stride == 0);
      return this->vector_elements * N;
   }

   if (this->without_array()->is_matrix()) {
      const struct glsl_type *element_type;
      const struct glsl_type *vec_type;
      unsigned int array_len;

      if (this->is_array()) {
         element_type = this->without_array();
         array_len = this->arrays_of_arrays_size();
      } else {
         element_type = this;
         array_len = 1;
      }

      if (row_major) {
         vec_type = get_instance(element_type->base_type,
                                 element_type->matrix_columns, 1);

         array_len *= element_type->vector_elements;
      } else {
         vec_type = get_instance(element_type->base_type,
                                 element_type->vector_elements, 1);
         array_len *= element_type->matrix_columns;
      }
      const glsl_type *array_type = glsl_type::get_array_instance(vec_type,
                                                                  array_len);

      return array_type->std430_size(false);
   }

   if (this->is_array()) {
      unsigned stride;
      if (this->without_array()->is_struct())
         stride = this->without_array()->std430_size(row_major);
      else
         stride = this->without_array()->std430_base_alignment(row_major);

      unsigned size = this->arrays_of_arrays_size() * stride;
      assert(this->explicit_stride == 0 ||
             size == this->length * this->explicit_stride);
      return size;
   }

   if (this->is_struct() || this->is_interface()) {
      unsigned size = 0;
      unsigned max_align = 0;

      for (unsigned i = 0; i < this->length; i++) {
         bool field_row_major = row_major;
         const enum glsl_matrix_layout matrix_layout =
            glsl_matrix_layout(this->fields.structure[i].matrix_layout);
         if (matrix_layout == GLSL_MATRIX_LAYOUT_ROW_MAJOR) {
            field_row_major = true;
         } else if (matrix_layout == GLSL_MATRIX_LAYOUT_COLUMN_MAJOR) {
            field_row_major = false;
         }

         const struct glsl_type *field_type = this->fields.structure[i].type;
         unsigned align = field_type->std430_base_alignment(field_row_major);
         size = glsl_align(size, align);
         size += field_type->std430_size(field_row_major);

         max_align = MAX2(align, max_align);
      }
      size = glsl_align(size, max_align);
      return size;
   }

   assert(!"not reached");
   return -1;
}

const glsl_type *
glsl_type::get_explicit_std430_type(bool row_major) const
{
   if (this->is_vector() || this->is_scalar()) {
      return this;
   } else if (this->is_matrix()) {
      const glsl_type *vec_type;
      if (row_major)
         vec_type = get_instance(this->base_type, this->matrix_columns, 1);
      else
         vec_type = get_instance(this->base_type, this->vector_elements, 1);
      unsigned stride = vec_type->std430_array_stride(false);
      return get_instance(this->base_type, this->vector_elements,
                          this->matrix_columns, stride, row_major);
   } else if (this->is_array()) {
      const glsl_type *elem_type =
         this->fields.array->get_explicit_std430_type(row_major);
      unsigned stride = this->fields.array->std430_array_stride(row_major);
      return get_array_instance(elem_type, this->length, stride);
   } else if (this->is_struct() || this->is_interface()) {
      glsl_struct_field *fields = new glsl_struct_field[this->length];
      unsigned offset = 0;
      for (unsigned i = 0; i < length; i++) {
         fields[i] = this->fields.structure[i];

         bool field_row_major = row_major;
         if (fields[i].matrix_layout == GLSL_MATRIX_LAYOUT_COLUMN_MAJOR) {
            field_row_major = false;
         } else if (fields[i].matrix_layout == GLSL_MATRIX_LAYOUT_ROW_MAJOR) {
            field_row_major = true;
         }
         fields[i].type =
            fields[i].type->get_explicit_std430_type(field_row_major);

         unsigned fsize = fields[i].type->std430_size(field_row_major);
         unsigned falign = fields[i].type->std430_base_alignment(field_row_major);
         /* From the GLSL 460 spec section "Uniform and Shader Storage Block
          * Layout Qualifiers":
          *
          *    "The actual offset of a member is computed as follows: If
          *    offset was declared, start with that offset, otherwise start
          *    with the next available offset. If the resulting offset is not
          *    a multiple of the actual alignment, increase it to the first
          *    offset that is a multiple of the actual alignment. This results
          *    in the actual offset the member will have."
          */
         if (fields[i].offset >= 0) {
            assert((unsigned)fields[i].offset >= offset);
            offset = fields[i].offset;
         }
         offset = glsl_align(offset, falign);
         fields[i].offset = offset;
         offset += fsize;
      }

      const glsl_type *type;
      if (this->is_struct())
         type = get_struct_instance(fields, this->length, this->name);
      else
         type = get_interface_instance(fields, this->length,
                                       (enum glsl_interface_packing)this->interface_packing,
                                       this->interface_row_major,
                                       this->name);

      delete[] fields;
      return type;
   } else {
      unreachable("Invalid type for SSBO");
   }
}

const glsl_type *
glsl_type::get_explicit_interface_type(bool supports_std430) const
{
   enum glsl_interface_packing packing =
      this->get_internal_ifc_packing(supports_std430);
   if (packing == GLSL_INTERFACE_PACKING_STD140) {
      return this->get_explicit_std140_type(this->interface_row_major);
   } else {
      assert(packing == GLSL_INTERFACE_PACKING_STD430);
      return this->get_explicit_std430_type(this->interface_row_major);
   }
}

static unsigned
explicit_type_scalar_byte_size(const glsl_type *type)
{
   if (type->base_type == GLSL_TYPE_BOOL)
      return 4;
   else
      return glsl_base_type_get_bit_size(type->base_type) / 8;
}

/* This differs from get_explicit_std430_type() in that it:
 * - can size arrays slightly smaller ("stride * (len - 1) + elem_size" instead
 *   of "stride * len")
 * - consumes a glsl_type_size_align_func which allows 8 and 16-bit values to be
 *   packed more tightly
 * - overrides any struct field offsets but get_explicit_std430_type() tries to
 *   respect any existing ones
 */
const glsl_type *
glsl_type::get_explicit_type_for_size_align(glsl_type_size_align_func type_info,
                                            unsigned *size, unsigned *alignment) const
{
   if (this->is_image() || this->is_sampler()) {
      type_info(this, size, alignment);
      assert(*alignment > 0);
      return this;
   } else if (this->is_scalar()) {
      type_info(this, size, alignment);
      assert(*size == explicit_type_scalar_byte_size(this));
      assert(*alignment == explicit_type_scalar_byte_size(this));
      return this;
   } else if (this->is_vector()) {
      type_info(this, size, alignment);
      assert(*alignment > 0);
      assert(*alignment % explicit_type_scalar_byte_size(this) == 0);
      return glsl_type::get_instance(this->base_type, this->vector_elements,
                                     1, 0, false, *alignment);
   } else if (this->is_array()) {
      unsigned elem_size, elem_align;
      const struct glsl_type *explicit_element =
         this->fields.array->get_explicit_type_for_size_align(type_info, &elem_size, &elem_align);

      unsigned stride = align(elem_size, elem_align);

      *size = stride * (this->length - 1) + elem_size;
      *alignment = elem_align;
      return glsl_type::get_array_instance(explicit_element, this->length, stride);
   } else if (this->is_struct() || this->is_interface()) {
      struct glsl_struct_field *fields = (struct glsl_struct_field *)
         malloc(sizeof(struct glsl_struct_field) * this->length);

      *size = 0;
      *alignment = 0;
      for (unsigned i = 0; i < this->length; i++) {
         fields[i] = this->fields.structure[i];
         assert(fields[i].matrix_layout != GLSL_MATRIX_LAYOUT_ROW_MAJOR);

         unsigned field_size, field_align;
         fields[i].type =
            fields[i].type->get_explicit_type_for_size_align(type_info, &field_size, &field_align);
         field_align = this->packed ? 1 : field_align;
         fields[i].offset = align(*size, field_align);

         *size = fields[i].offset + field_size;
         *alignment = MAX2(*alignment, field_align);
      }
      /*
       * "The alignment of the struct is the alignment of the most-aligned
       *  field in it."
       *
       * "Finally, the size of the struct is the current offset rounded up to
       *  the nearest multiple of the struct's alignment."
       *
       * https://doc.rust-lang.org/reference/type-layout.html#reprc-structs
       */
      *size = align(*size, *alignment);

      const glsl_type *type;
      if (this->is_struct()) {
         type = get_struct_instance(fields, this->length, this->name,
                                    this->packed, *alignment);
      } else {
         assert(!this->packed);
         type = get_interface_instance(fields, this->length,
                                       (enum glsl_interface_packing)this->interface_packing,
                                       this->interface_row_major,
                                       this->name);
      }
      free(fields);
      return type;
   } else if (this->is_matrix()) {
      unsigned col_size, col_align;
      type_info(this->column_type(), &col_size, &col_align);
      unsigned stride = align(col_size, col_align);

      *size = this->matrix_columns * stride;
      /* Matrix and column alignments match. See glsl_type::column_type() */
      assert(col_align > 0);
      *alignment = col_align;
      return glsl_type::get_instance(this->base_type, this->vector_elements,
                                     this->matrix_columns, stride, false, *alignment);
   } else {
      unreachable("Unhandled type.");
   }
}

const glsl_type *
glsl_type::replace_vec3_with_vec4() const
{
   if (this->is_scalar() || this->is_vector() || this->is_matrix()) {
      if (this->interface_row_major) {
         if (this->matrix_columns == 3) {
            return glsl_type::get_instance(this->base_type,
                                           this->vector_elements,
                                           4, /* matrix columns */
                                           this->explicit_stride,
                                           this->interface_row_major,
                                           this->explicit_alignment);
         } else {
            return this;
         }
      } else {
         if (this->vector_elements == 3) {
            return glsl_type::get_instance(this->base_type,
                                           4, /* vector elements */
                                           this->matrix_columns,
                                           this->explicit_stride,
                                           this->interface_row_major,
                                           this->explicit_alignment);
         } else {
            return this;
         }
      }
   } else if (this->is_array()) {
      const glsl_type *vec4_elem_type =
         this->fields.array->replace_vec3_with_vec4();
      if (vec4_elem_type == this->fields.array)
         return this;
      return glsl_type::get_array_instance(vec4_elem_type,
                                           this->length,
                                           this->explicit_stride);
   } else if (this->is_struct() || this->is_interface()) {
      struct glsl_struct_field *fields = (struct glsl_struct_field *)
         malloc(sizeof(struct glsl_struct_field) * this->length);

      bool needs_new_type = false;
      for (unsigned i = 0; i < this->length; i++) {
         fields[i] = this->fields.structure[i];
         assert(fields[i].matrix_layout != GLSL_MATRIX_LAYOUT_ROW_MAJOR);
         fields[i].type = fields[i].type->replace_vec3_with_vec4();
         if (fields[i].type != this->fields.structure[i].type)
            needs_new_type = true;
      }

      const glsl_type *type;
      if (!needs_new_type) {
         type = this;
      } else if (this->is_struct()) {
         type = get_struct_instance(fields, this->length, this->name,
                                    this->packed, this->explicit_alignment);
      } else {
         assert(!this->packed);
         type = get_interface_instance(fields, this->length,
                                       (enum glsl_interface_packing)this->interface_packing,
                                       this->interface_row_major,
                                       this->name);
      }
      free(fields);
      return type;
   } else {
      unreachable("Unhandled type.");
   }
}

unsigned
glsl_type::count_vec4_slots(bool is_gl_vertex_input, bool is_bindless) const
{
   /* From page 31 (page 37 of the PDF) of the GLSL 1.50 spec:
    *
    *     "A scalar input counts the same amount against this limit as a vec4,
    *     so applications may want to consider packing groups of four
    *     unrelated float inputs together into a vector to better utilize the
    *     capabilities of the underlying hardware. A matrix input will use up
    *     multiple locations.  The number of locations used will equal the
    *     number of columns in the matrix."
    *
    * The spec does not explicitly say how arrays are counted.  However, it
    * should be safe to assume the total number of slots consumed by an array
    * is the number of entries in the array multiplied by the number of slots
    * consumed by a single element of the array.
    *
    * The spec says nothing about how structs are counted, because vertex
    * attributes are not allowed to be (or contain) structs.  However, Mesa
    * allows varying structs, the number of varying slots taken up by a
    * varying struct is simply equal to the sum of the number of slots taken
    * up by each element.
    *
    * Doubles are counted different depending on whether they are vertex
    * inputs or everything else. Vertex inputs from ARB_vertex_attrib_64bit
    * take one location no matter what size they are, otherwise dvec3/4
    * take two locations.
    */
   switch (this->base_type) {
   case GLSL_TYPE_UINT:
   case GLSL_TYPE_INT:
   case GLSL_TYPE_UINT8:
   case GLSL_TYPE_INT8:
   case GLSL_TYPE_UINT16:
   case GLSL_TYPE_INT16:
   case GLSL_TYPE_FLOAT:
   case GLSL_TYPE_FLOAT16:
   case GLSL_TYPE_BOOL:
      return this->matrix_columns;
   case GLSL_TYPE_DOUBLE:
   case GLSL_TYPE_UINT64:
   case GLSL_TYPE_INT64:
      if (this->vector_elements > 2 && !is_gl_vertex_input)
         return this->matrix_columns * 2;
      else
         return this->matrix_columns;
   case GLSL_TYPE_STRUCT:
   case GLSL_TYPE_INTERFACE: {
      unsigned size = 0;

      for (unsigned i = 0; i < this->length; i++) {
         const glsl_type *member_type = this->fields.structure[i].type;
         size += member_type->count_vec4_slots(is_gl_vertex_input, is_bindless);
      }

      return size;
   }

   case GLSL_TYPE_ARRAY: {
      const glsl_type *element = this->fields.array;
      return this->length * element->count_vec4_slots(is_gl_vertex_input,
                                                      is_bindless);
   }

   case GLSL_TYPE_SAMPLER:
   case GLSL_TYPE_TEXTURE:
   case GLSL_TYPE_IMAGE:
      if (!is_bindless)
         return 0;
      else
         return 1;

   case GLSL_TYPE_SUBROUTINE:
      return 1;

   case GLSL_TYPE_FUNCTION:
   case GLSL_TYPE_ATOMIC_UINT:
   case GLSL_TYPE_VOID:
   case GLSL_TYPE_ERROR:
      break;
   }

   assert(!"Unexpected type in count_attribute_slots()");

   return 0;
}

unsigned
glsl_type::count_dword_slots(bool is_bindless) const
{
   switch (this->base_type) {
   case GLSL_TYPE_UINT:
   case GLSL_TYPE_INT:
   case GLSL_TYPE_FLOAT:
   case GLSL_TYPE_BOOL:
      return this->components();
   case GLSL_TYPE_UINT16:
   case GLSL_TYPE_INT16:
   case GLSL_TYPE_FLOAT16:
      return DIV_ROUND_UP(this->vector_elements, 2) * this->matrix_columns;
   case GLSL_TYPE_UINT8:
   case GLSL_TYPE_INT8:
      return DIV_ROUND_UP(this->components(), 4);
   case GLSL_TYPE_IMAGE:
   case GLSL_TYPE_SAMPLER:
   case GLSL_TYPE_TEXTURE:
      if (!is_bindless)
         return 0;
      FALLTHROUGH;
   case GLSL_TYPE_DOUBLE:
   case GLSL_TYPE_UINT64:
   case GLSL_TYPE_INT64:
      return this->components() * 2;
   case GLSL_TYPE_ARRAY:
      return this->fields.array->count_dword_slots(is_bindless) *
             this->length;

   case GLSL_TYPE_INTERFACE:
   case GLSL_TYPE_STRUCT: {
      unsigned size = 0;
      for (unsigned i = 0; i < this->length; i++) {
         size += this->fields.structure[i].type->count_dword_slots(is_bindless);
      }
      return size;
   }

   case GLSL_TYPE_ATOMIC_UINT:
      return 0;
   case GLSL_TYPE_SUBROUTINE:
      return 1;
   case GLSL_TYPE_VOID:
   case GLSL_TYPE_ERROR:
   case GLSL_TYPE_FUNCTION:
   default:
      unreachable("invalid type in st_glsl_type_dword_size()");
   }

   return 0;
}

int
glsl_type::coordinate_components() const
{
   enum glsl_sampler_dim dim = (enum glsl_sampler_dim)sampler_dimensionality;
   int size = glsl_get_sampler_dim_coordinate_components(dim);

   /* Array textures need an additional component for the array index, except
    * for cubemap array images that behave like a 2D array of interleaved
    * cubemap faces.
    */
   if (sampler_array &&
       !(is_image() && sampler_dimensionality == GLSL_SAMPLER_DIM_CUBE))
      size += 1;

   return size;
}

/**
 * Declarations of type flyweights (glsl_type::_foo_type) and
 * convenience pointers (glsl_type::foo_type).
 * @{
 */
#define DECL_TYPE(NAME, ...)                                    \
   const glsl_type glsl_type::_##NAME##_type = glsl_type(__VA_ARGS__, #NAME); \
   const glsl_type *const glsl_type::NAME##_type = &glsl_type::_##NAME##_type;

#define STRUCT_TYPE(NAME)

#include "compiler/builtin_type_macros.h"
/** @} */

union packed_type {
   uint32_t u32;
   struct {
      unsigned base_type:5;
      unsigned interface_row_major:1;
      unsigned vector_elements:3;
      unsigned matrix_columns:3;
      unsigned explicit_stride:16;
      unsigned explicit_alignment:4;
   } basic;
   struct {
      unsigned base_type:5;
      unsigned dimensionality:4;
      unsigned shadow:1;
      unsigned array:1;
      unsigned sampled_type:5;
      unsigned _pad:16;
   } sampler;
   struct {
      unsigned base_type:5;
      unsigned length:13;
      unsigned explicit_stride:14;
   } array;
   struct {
      unsigned base_type:5;
      unsigned interface_packing_or_packed:2;
      unsigned interface_row_major:1;
      unsigned length:20;
      unsigned explicit_alignment:4;
   } strct;
};

static void
encode_glsl_struct_field(blob *blob, const glsl_struct_field *struct_field)
{
   encode_type_to_blob(blob, struct_field->type);
   blob_write_string(blob, struct_field->name);
   blob_write_uint32(blob, struct_field->location);
   blob_write_uint32(blob, struct_field->component);
   blob_write_uint32(blob, struct_field->offset);
   blob_write_uint32(blob, struct_field->xfb_buffer);
   blob_write_uint32(blob, struct_field->xfb_stride);
   blob_write_uint32(blob, struct_field->image_format);
   blob_write_uint32(blob, struct_field->flags);
}

static void
decode_glsl_struct_field_from_blob(blob_reader *blob, glsl_struct_field *struct_field)
{
   struct_field->type = decode_type_from_blob(blob);
   struct_field->name = blob_read_string(blob);
   struct_field->location = blob_read_uint32(blob);
   struct_field->component = blob_read_uint32(blob);
   struct_field->offset = blob_read_uint32(blob);
   struct_field->xfb_buffer = blob_read_uint32(blob);
   struct_field->xfb_stride = blob_read_uint32(blob);
   struct_field->image_format = (pipe_format)blob_read_uint32(blob);
   struct_field->flags = blob_read_uint32(blob);
}

void
encode_type_to_blob(struct blob *blob, const glsl_type *type)
{
   if (!type) {
      blob_write_uint32(blob, 0);
      return;
   }

   STATIC_ASSERT(sizeof(union packed_type) == 4);
   union packed_type encoded;
   encoded.u32 = 0;
   encoded.basic.base_type = type->base_type;

   switch (type->base_type) {
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
      encoded.basic.interface_row_major = type->interface_row_major;
      assert(type->matrix_columns < 8);
      if (type->vector_elements <= 5)
         encoded.basic.vector_elements = type->vector_elements;
      else if (type->vector_elements == 8)
         encoded.basic.vector_elements = 6;
      else if (type->vector_elements == 16)
         encoded.basic.vector_elements = 7;
      encoded.basic.matrix_columns = type->matrix_columns;
      encoded.basic.explicit_stride = MIN2(type->explicit_stride, 0xffff);
      encoded.basic.explicit_alignment =
         MIN2(ffs(type->explicit_alignment), 0xf);
      blob_write_uint32(blob, encoded.u32);
      /* If we don't have enough bits for explicit_stride, store it
       * separately.
       */
      if (encoded.basic.explicit_stride == 0xffff)
         blob_write_uint32(blob, type->explicit_stride);
      if (encoded.basic.explicit_alignment == 0xf)
         blob_write_uint32(blob, type->explicit_alignment);
      return;
   case GLSL_TYPE_SAMPLER:
   case GLSL_TYPE_TEXTURE:
   case GLSL_TYPE_IMAGE:
      encoded.sampler.dimensionality = type->sampler_dimensionality;
      if (type->base_type == GLSL_TYPE_SAMPLER)
         encoded.sampler.shadow = type->sampler_shadow;
      else
         assert(!type->sampler_shadow);
      encoded.sampler.array = type->sampler_array;
      encoded.sampler.sampled_type = type->sampled_type;
      break;
   case GLSL_TYPE_SUBROUTINE:
      blob_write_uint32(blob, encoded.u32);
      blob_write_string(blob, type->name);
      return;
   case GLSL_TYPE_ATOMIC_UINT:
      break;
   case GLSL_TYPE_ARRAY:
      encoded.array.length = MIN2(type->length, 0x1fff);
      encoded.array.explicit_stride = MIN2(type->explicit_stride, 0x3fff);
      blob_write_uint32(blob, encoded.u32);
      /* If we don't have enough bits for length or explicit_stride, store it
       * separately.
       */
      if (encoded.array.length == 0x1fff)
         blob_write_uint32(blob, type->length);
      if (encoded.array.explicit_stride == 0x3fff)
         blob_write_uint32(blob, type->explicit_stride);
      encode_type_to_blob(blob, type->fields.array);
      return;
   case GLSL_TYPE_STRUCT:
   case GLSL_TYPE_INTERFACE:
      encoded.strct.length = MIN2(type->length, 0xfffff);
      encoded.strct.explicit_alignment =
         MIN2(ffs(type->explicit_alignment), 0xf);
      if (type->is_interface()) {
         encoded.strct.interface_packing_or_packed = type->interface_packing;
         encoded.strct.interface_row_major = type->interface_row_major;
      } else {
         encoded.strct.interface_packing_or_packed = type->packed;
      }
      blob_write_uint32(blob, encoded.u32);
      blob_write_string(blob, type->name);

      /* If we don't have enough bits for length, store it separately. */
      if (encoded.strct.length == 0xfffff)
         blob_write_uint32(blob, type->length);
      if (encoded.strct.explicit_alignment == 0xf)
         blob_write_uint32(blob, type->explicit_alignment);

      for (unsigned i = 0; i < type->length; i++)
         encode_glsl_struct_field(blob, &type->fields.structure[i]);
      return;
   case GLSL_TYPE_VOID:
      break;
   case GLSL_TYPE_ERROR:
   default:
      assert(!"Cannot encode type!");
      encoded.u32 = 0;
      break;
   }

   blob_write_uint32(blob, encoded.u32);
}

const glsl_type *
decode_type_from_blob(struct blob_reader *blob)
{
   union packed_type encoded;
   encoded.u32 = blob_read_uint32(blob);

   if (encoded.u32 == 0) {
      return NULL;
   }

   glsl_base_type base_type = (glsl_base_type)encoded.basic.base_type;

   switch (base_type) {
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
   case GLSL_TYPE_BOOL: {
      unsigned explicit_stride = encoded.basic.explicit_stride;
      if (explicit_stride == 0xffff)
         explicit_stride = blob_read_uint32(blob);
      unsigned explicit_alignment = encoded.basic.explicit_alignment;
      if (explicit_alignment == 0xf)
         explicit_alignment = blob_read_uint32(blob);
      else if (explicit_alignment > 0)
         explicit_alignment = 1 << (explicit_alignment - 1);
      uint32_t vector_elements = encoded.basic.vector_elements;
      if (vector_elements == 6)
         vector_elements = 8;
      else if (vector_elements == 7)
         vector_elements = 16;
      return glsl_type::get_instance(base_type, vector_elements,
                                     encoded.basic.matrix_columns,
                                     explicit_stride,
                                     encoded.basic.interface_row_major,
                                     explicit_alignment);
   }
   case GLSL_TYPE_SAMPLER:
      return glsl_type::get_sampler_instance((enum glsl_sampler_dim)encoded.sampler.dimensionality,
                                             encoded.sampler.shadow,
                                             encoded.sampler.array,
                                             (glsl_base_type) encoded.sampler.sampled_type);
   case GLSL_TYPE_TEXTURE:
      return glsl_type::get_texture_instance((enum glsl_sampler_dim)encoded.sampler.dimensionality,
                                             encoded.sampler.array,
                                             (glsl_base_type) encoded.sampler.sampled_type);
   case GLSL_TYPE_SUBROUTINE:
      return glsl_type::get_subroutine_instance(blob_read_string(blob));
   case GLSL_TYPE_IMAGE:
      return glsl_type::get_image_instance((enum glsl_sampler_dim)encoded.sampler.dimensionality,
                                           encoded.sampler.array,
                                           (glsl_base_type) encoded.sampler.sampled_type);
   case GLSL_TYPE_ATOMIC_UINT:
      return glsl_type::atomic_uint_type;
   case GLSL_TYPE_ARRAY: {
      unsigned length = encoded.array.length;
      if (length == 0x1fff)
         length = blob_read_uint32(blob);
      unsigned explicit_stride = encoded.array.explicit_stride;
      if (explicit_stride == 0x3fff)
         explicit_stride = blob_read_uint32(blob);
      return glsl_type::get_array_instance(decode_type_from_blob(blob),
                                           length, explicit_stride);
   }
   case GLSL_TYPE_STRUCT:
   case GLSL_TYPE_INTERFACE: {
      char *name = blob_read_string(blob);
      unsigned num_fields = encoded.strct.length;
      if (num_fields == 0xfffff)
         num_fields = blob_read_uint32(blob);
      unsigned explicit_alignment = encoded.strct.explicit_alignment;
      if (explicit_alignment == 0xf)
         explicit_alignment = blob_read_uint32(blob);
      else if (explicit_alignment > 0)
         explicit_alignment = 1 << (explicit_alignment - 1);

      glsl_struct_field *fields =
         (glsl_struct_field *) malloc(sizeof(glsl_struct_field) * num_fields);
      for (unsigned i = 0; i < num_fields; i++)
         decode_glsl_struct_field_from_blob(blob, &fields[i]);

      const glsl_type *t;
      if (base_type == GLSL_TYPE_INTERFACE) {
         assert(explicit_alignment == 0);
         enum glsl_interface_packing packing =
            (glsl_interface_packing) encoded.strct.interface_packing_or_packed;
         bool row_major = encoded.strct.interface_row_major;
         t = glsl_type::get_interface_instance(fields, num_fields, packing,
                                               row_major, name);
      } else {
         unsigned packed = encoded.strct.interface_packing_or_packed;
         t = glsl_type::get_struct_instance(fields, num_fields, name, packed,
                                            explicit_alignment);
      }

      free(fields);
      return t;
   }
   case GLSL_TYPE_VOID:
      return glsl_type::void_type;
   case GLSL_TYPE_ERROR:
   default:
      assert(!"Cannot decode type!");
      return NULL;
   }
}

unsigned
glsl_type::cl_alignment() const
{
   /* vectors unlike arrays are aligned to their size */
   if (this->is_scalar() || this->is_vector())
      return this->cl_size();
   else if (this->is_array())
      return this->fields.array->cl_alignment();
   else if (this->is_struct()) {
      /* Packed Structs are 0x1 aligned despite their size. */
      if (this->packed)
         return 1;

      unsigned res = 1;
      for (unsigned i = 0; i < this->length; ++i) {
         struct glsl_struct_field &field = this->fields.structure[i];
         res = MAX2(res, field.type->cl_alignment());
      }
      return res;
   }
   return 1;
}

unsigned
glsl_type::cl_size() const
{
   if (this->is_scalar() || this->is_vector()) {
      return util_next_power_of_two(this->vector_elements) *
             explicit_type_scalar_byte_size(this);
   } else if (this->is_array()) {
      unsigned size = this->fields.array->cl_size();
      return size * this->length;
   } else if (this->is_struct()) {
      unsigned size = 0;
      for (unsigned i = 0; i < this->length; ++i) {
         struct glsl_struct_field &field = this->fields.structure[i];
         /* if a struct is packed, members don't get aligned */
         if (!this->packed)
            size = align(size, field.type->cl_alignment());
         size += field.type->cl_size();
      }
      return size;
   }
   return 1;
}

extern "C" {

int
glsl_get_sampler_dim_coordinate_components(enum glsl_sampler_dim dim)
{
   switch (dim) {
   case GLSL_SAMPLER_DIM_1D:
   case GLSL_SAMPLER_DIM_BUF:
      return 1;
   case GLSL_SAMPLER_DIM_2D:
   case GLSL_SAMPLER_DIM_RECT:
   case GLSL_SAMPLER_DIM_MS:
   case GLSL_SAMPLER_DIM_EXTERNAL:
   case GLSL_SAMPLER_DIM_SUBPASS:
   case GLSL_SAMPLER_DIM_SUBPASS_MS:
      return 2;
   case GLSL_SAMPLER_DIM_3D:
   case GLSL_SAMPLER_DIM_CUBE:
      return 3;
   default:
      unreachable("Unknown sampler dim");
   }
}

void
glsl_print_type(FILE *f, const glsl_type *t)
{
   if (t->is_array()) {
      fprintf(f, "(array ");
      glsl_print_type(f, t->fields.array);
      fprintf(f, " %u)", t->length);
   } else if (t->is_struct() && !is_gl_identifier(t->name)) {
      fprintf(f, "%s@%p", t->name, (void *) t);
   } else {
      fprintf(f, "%s", t->name);
   }
}

}

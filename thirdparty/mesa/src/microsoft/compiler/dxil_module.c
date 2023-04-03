/*
 * Copyright © Microsoft Corporation
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "dxil_module.h"
#include "dxil_internal.h"

#include "util/macros.h"
#include "util/u_math.h"
#include "util/u_memory.h"
#include "util/rb_tree.h"

#include <assert.h>
#include <stdio.h>

#include "drivers/d3d12/d3d12_godot_nir_bridge.h"

void
dxil_module_init(struct dxil_module *m, void *ralloc_ctx)
{
   assert(ralloc_ctx);

   memset(m, 0, sizeof(struct dxil_module));
   m->ralloc_ctx = ralloc_ctx;

   dxil_buffer_init(&m->buf, 2);
   memset(&m->feats, 0, sizeof(m->feats));

   list_inithead(&m->type_list);
   list_inithead(&m->func_list);
   list_inithead(&m->func_def_list);
   list_inithead(&m->attr_set_list);
   list_inithead(&m->gvar_list);
   list_inithead(&m->const_list);
   list_inithead(&m->mdnode_list);
   list_inithead(&m->md_named_node_list);

   m->functions = rzalloc(ralloc_ctx, struct rb_tree);
   rb_tree_init(m->functions);
}

void
dxil_module_release(struct dxil_module *m)
{
   dxil_buffer_finish(&m->buf);
}

static bool
emit_bits64(struct dxil_buffer *b, uint64_t data, unsigned width)
{
   if (data > UINT32_MAX) {
      assert(width > 32);
      return dxil_buffer_emit_bits(b, (uint32_t)(data & UINT32_MAX), width) &&
             dxil_buffer_emit_bits(b, (uint32_t)(data >> 32), width - 32);
   } else
      return dxil_buffer_emit_bits(b, (uint32_t)data, width);
}

/* See the LLVM documentation for details about what these are all about:
 * https://www.llvm.org/docs/BitCodeFormat.html#abbreviation-ids
 */
enum dxil_fixed_abbrev {
   DXIL_END_BLOCK = 0,
   DXIL_ENTER_SUBBLOCK = 1,
   DXIL_DEFINE_ABBREV = 2,
   DXIL_UNABBREV_RECORD = 3,
   DXIL_FIRST_APPLICATION_ABBREV = 4
};

static bool
enter_subblock(struct dxil_module *m, unsigned id, unsigned abbrev_width)
{
   assert(m->num_blocks < ARRAY_SIZE(m->blocks));
   m->blocks[m->num_blocks].abbrev_width = m->buf.abbrev_width;

   if (!dxil_buffer_emit_abbrev_id(&m->buf, DXIL_ENTER_SUBBLOCK) ||
       !dxil_buffer_emit_vbr_bits(&m->buf, id, 8) ||
       !dxil_buffer_emit_vbr_bits(&m->buf, abbrev_width, 4) ||
       !dxil_buffer_align(&m->buf))
      return false;

   m->buf.abbrev_width = abbrev_width;
   m->blocks[m->num_blocks++].offset = blob_reserve_uint32(&m->buf.blob);
   return true;
}

static bool
exit_block(struct dxil_module *m)
{
   assert(m->num_blocks > 0);
   assert(m->num_blocks < ARRAY_SIZE(m->blocks));

   if (!dxil_buffer_emit_abbrev_id(&m->buf, DXIL_END_BLOCK) ||
       !dxil_buffer_align(&m->buf))
      return false;

   intptr_t size_offset = m->blocks[m->num_blocks - 1].offset;
   uint32_t size = (m->buf.blob.size - size_offset - 1) / sizeof(uint32_t);
   if (!blob_overwrite_uint32(&m->buf.blob, size_offset, size))
      return false;

   m->num_blocks--;
   m->buf.abbrev_width = m->blocks[m->num_blocks].abbrev_width;
   return true;
}

static bool
emit_record_no_abbrev(struct dxil_buffer *b, unsigned code,
                      const uint64_t *data, size_t size)
{
   if (!dxil_buffer_emit_abbrev_id(b, DXIL_UNABBREV_RECORD) ||
       !dxil_buffer_emit_vbr_bits(b, code, 6) ||
       !dxil_buffer_emit_vbr_bits(b, size, 6))
      return false;

   for (size_t i = 0; i < size; ++i)
      if (!dxil_buffer_emit_vbr_bits(b, data[i], 6))
         return false;

   return true;
}

static bool
emit_record(struct dxil_module *m, unsigned code,
            const uint64_t *data, size_t size)
{
   return emit_record_no_abbrev(&m->buf, code, data, size);
}

static bool
emit_record_int(struct dxil_module *m, unsigned code, int value)
{
   uint64_t data = value;
   return emit_record(m, code, &data, 1);
}

static bool
is_char6(char ch)
{
   if ((ch >= 'a' && ch <= 'z') ||
       (ch >= 'A' && ch <= 'Z') ||
       (ch >= '0' && ch <= '9'))
     return true;

   switch (ch) {
   case '.':
   case '_':
      return true;

   default:
      return false;
   }
}

static bool
is_char6_string(const char *str)
{
   while (*str != '\0') {
      if (!is_char6(*str++))
         return false;
   }
   return true;
}

static bool
is_char7_string(const char *str)
{
   while (*str != '\0') {
      if (*str++ & 0x80)
         return false;
   }
   return true;
}

static unsigned
encode_char6(char ch)
{
   const int letters = 'z' - 'a' + 1;

   if (ch >= 'a' && ch <= 'z')
      return ch - 'a';
   else if (ch >= 'A' && ch <= 'Z')
      return letters + ch - 'A';
   else if (ch >= '0' && ch <= '9')
      return 2 * letters + ch - '0';

   switch (ch) {
   case '.': return 62;
   case '_': return 63;
   default:
      unreachable("invalid char6-character");
   }
}

static bool
emit_fixed(struct dxil_buffer *b, uint64_t data, unsigned width)
{
   if (!width)
      return true;

   return emit_bits64(b, data, width);
}

static bool
emit_vbr(struct dxil_buffer *b, uint64_t data, unsigned width)
{
   if (!width)
      return true;

   return dxil_buffer_emit_vbr_bits(b, data, width);
}

static bool
emit_char6(struct dxil_buffer *b, uint64_t data)
{
   return dxil_buffer_emit_bits(b, encode_char6((char)data), 6);
}

struct dxil_abbrev {
   struct {
      enum {
         DXIL_OP_LITERAL = 0,
         DXIL_OP_FIXED = 1,
         DXIL_OP_VBR = 2,
         DXIL_OP_ARRAY = 3,
         DXIL_OP_CHAR6 = 4,
         DXIL_OP_BLOB = 5
      } type;
      union {
         uint64_t value;
         uint64_t encoding_data;
      };
   } operands[7];
   size_t num_operands;
};

static bool
emit_record_abbrev(struct dxil_buffer *b,
                   unsigned abbrev, const struct dxil_abbrev *a,
                   const uint64_t *data, size_t size)
{
   assert(abbrev >= DXIL_FIRST_APPLICATION_ABBREV);

   if (!dxil_buffer_emit_abbrev_id(b, abbrev))
      return false;

   size_t curr_data = 0;
   for (int i = 0; i < a->num_operands; ++i) {
      switch (a->operands[i].type) {
      case DXIL_OP_LITERAL:
         assert(curr_data < size);
         assert(data[curr_data] == a->operands[i].value);
         curr_data++;
         /* literals are no-ops, because their value is defined in the
            abbrev-definition already */
         break;

      case DXIL_OP_FIXED:
         assert(curr_data < size);
         if (!emit_fixed(b, data[curr_data++], a->operands[i].encoding_data))
            return false;
         break;

      case DXIL_OP_VBR:
         assert(curr_data < size);
         if (!emit_vbr(b, data[curr_data++], a->operands[i].encoding_data))
            return false;
         break;

      case DXIL_OP_ARRAY:
         assert(i == a->num_operands - 2); /* arrays should always be second to last */

         if (!dxil_buffer_emit_vbr_bits(b, size - curr_data, 6))
            return false;

         switch (a->operands[i + 1].type) {
         case DXIL_OP_FIXED:
            while (curr_data < size)
               if (!emit_fixed(b, data[curr_data++], a->operands[i + 1].encoding_data))
                  return false;
            break;

         case DXIL_OP_VBR:
            while (curr_data < size)
               if (!emit_vbr(b, data[curr_data++], a->operands[i + 1].encoding_data))
                  return false;
            break;

         case DXIL_OP_CHAR6:
            while (curr_data < size)
               if (!emit_char6(b, data[curr_data++]))
                  return false;
            break;

         default:
            unreachable("unexpected operand type");
         }
         return true; /* we're done */

      case DXIL_OP_CHAR6:
         assert(curr_data < size);
         if (!emit_char6(b, data[curr_data++]))
            return false;
         break;

      case DXIL_OP_BLOB:
         unreachable("HALP, unplement!");

      default:
         unreachable("unexpected operand type");
      }
   }

   assert(curr_data == size);
   return true;
}


static struct dxil_type *
create_type(struct dxil_module *m, enum type_type type)
{
   struct dxil_type *ret = rzalloc_size(m->ralloc_ctx,
                                        sizeof(struct dxil_type));
   if (ret) {
      ret->type = type;
      ret->id = list_length(&m->type_list);
      list_addtail(&ret->head, &m->type_list);
   }
   return ret;
}

static bool
types_equal(const struct dxil_type *lhs, const struct dxil_type *rhs);

static bool
type_list_equal(const struct dxil_type_list *lhs,
                const struct dxil_type_list *rhs)
{
   if (lhs->num_types != rhs->num_types)
      return false;
   for (unsigned i = 0; i < lhs->num_types; ++i)
      if (!types_equal(lhs->types[i],  rhs->types[i]))
          return false;
   return true;
}

static bool
types_equal(const struct dxil_type *lhs, const struct dxil_type *rhs)
{
   if (lhs == rhs)
      return true;

   /* Below we only assert that different type pointers really define different types
    * Since this function is only called in asserts, it is not needed to put the code
    * into a #ifdef NDEBUG statement */
   if (lhs->type != rhs->type)
      return false;

   bool retval = false;
   switch (lhs->type) {
   case TYPE_VOID:
      retval = true;
      break;
   case TYPE_FLOAT:
      retval = lhs->float_bits == rhs->float_bits;
      break;
   case TYPE_INTEGER:
      retval = lhs->int_bits == rhs->int_bits;
      break;
   case TYPE_POINTER:
      retval = types_equal(lhs->ptr_target_type, rhs->ptr_target_type);
      break;
   case TYPE_ARRAY:
   case TYPE_VECTOR:
      retval = (lhs->array_or_vector_def.num_elems == rhs->array_or_vector_def.num_elems) &&
               types_equal(lhs->array_or_vector_def.elem_type,
                           rhs->array_or_vector_def.elem_type);
      break;
   case TYPE_FUNCTION:
      if (!types_equal(lhs->function_def.ret_type,
                            rhs->function_def.ret_type))
         return false;
      retval = type_list_equal(&lhs->function_def.args, &rhs->function_def.args);
      break;
   case TYPE_STRUCT:
      retval = type_list_equal(&lhs->struct_def.elem, &rhs->struct_def.elem);
   }
   assert(!retval && "Types are equal in structure but not as pointers");
   return retval;
}

bool
dxil_value_type_equal_to(const struct dxil_value *value,
                         const struct dxil_type *rhs)
{
   return types_equal(value->type, rhs);
}

nir_alu_type
dxil_type_to_nir_type(const struct dxil_type *type)
{
   assert(type);
   switch (type->type) {
   case TYPE_INTEGER:
      return type->int_bits == 1 ? nir_type_bool : nir_type_int;
   case TYPE_FLOAT:
      return nir_type_float;
   default:
      unreachable("Unexpected type in dxil_type_to_nir_type");
   }
}

bool
dxil_value_type_bitsize_equal_to(const struct dxil_value *value, unsigned bitsize)
{
   switch (value->type->type) {
   case TYPE_INTEGER:
      return value->type->int_bits == bitsize;
   case TYPE_FLOAT:
      return value->type->float_bits == bitsize;
   default:
      return false;
   }
}

const struct dxil_type *
dxil_value_get_type(const struct dxil_value *value)
{
   return value->type;
}

const struct dxil_type *
dxil_module_get_void_type(struct dxil_module *m)
{
   if (!m->void_type)
      m->void_type = create_type(m, TYPE_VOID);
   return m->void_type;
}

static const struct dxil_type *
create_int_type(struct dxil_module *m, unsigned bit_size)
{
   struct dxil_type *type = create_type(m, TYPE_INTEGER);
   if (type)
      type->int_bits = bit_size;
   return type;
}

static const struct dxil_type *
get_int1_type(struct dxil_module *m)
{
   if (!m->int1_type)
      m->int1_type = create_int_type(m, 1);
   return m->int1_type;
}

static const struct dxil_type *
get_int8_type(struct dxil_module *m)
{
   if (!m->int8_type)
      m->int8_type = create_int_type(m, 8);
   return m->int8_type;
}

static const struct dxil_type *
get_int16_type(struct dxil_module *m)
{
   if (!m->int16_type)
      m->int16_type = create_int_type(m, 16);
   return m->int16_type;
}

static const struct dxil_type *
get_int32_type(struct dxil_module *m)
{
   if (!m->int32_type)
      m->int32_type = create_int_type(m, 32);
   return m->int32_type;
}

static const struct dxil_type *
get_int64_type(struct dxil_module *m)
{
   if (!m->int64_type)
      m->int64_type = create_int_type(m, 64);
   return m->int64_type;
}

static const struct dxil_type *
create_float_type(struct dxil_module *m, unsigned bit_size)
{
   struct dxil_type *type = create_type(m, TYPE_FLOAT);
   if (type)
      type->float_bits = bit_size;
   return type;
}

const struct dxil_type *
dxil_module_get_int_type(struct dxil_module *m, unsigned bit_size)
{
   switch (bit_size) {
   case 1: return get_int1_type(m);
   case 8: return get_int8_type(m);
   case 16: return get_int16_type(m);
   case 32: return get_int32_type(m);
   case 64: return get_int64_type(m);
   default:
      unreachable("unsupported bit-width");
   }
}

static const struct dxil_type *
get_float16_type(struct dxil_module *m)
{
   if (!m->float16_type)
      m->float16_type = create_float_type(m, 16);
   return m->float16_type;
}

static const struct dxil_type *
get_float32_type(struct dxil_module *m)
{
   if (!m->float32_type)
      m->float32_type = create_float_type(m, 32);
   return m->float32_type;
}

static const struct dxil_type *
get_float64_type(struct dxil_module *m)
{
   if (!m->float64_type)
      m->float64_type = create_float_type(m, 64);
   return m->float64_type;
}

const struct dxil_type *
dxil_module_get_float_type(struct dxil_module *m, unsigned bit_size)
{
   switch (bit_size) {
   case 16: return get_float16_type(m);
   case 32: return get_float32_type(m);
   case 64: return get_float64_type(m);
   default:
      unreachable("unsupported bit-width");
   }
   return get_float32_type(m);
}

const struct dxil_type *
dxil_module_get_pointer_type(struct dxil_module *m,
                             const struct dxil_type *target)
{
   struct dxil_type *type;
   LIST_FOR_EACH_ENTRY(type, &m->type_list, head) {
      if (type->type == TYPE_POINTER &&
          type->ptr_target_type == target)
         return type;
   }

   type = create_type(m, TYPE_POINTER);
   if (type)
      type->ptr_target_type = target;
   return type;
}

const struct dxil_type *
dxil_module_get_struct_type(struct dxil_module *m,
                            const char *name,
                            const struct dxil_type **elem_types,
                            size_t num_elem_types)
{
   assert(!name || strlen(name) > 0);

   struct dxil_type *type;
   LIST_FOR_EACH_ENTRY(type, &m->type_list, head) {
      if (type->type != TYPE_STRUCT)
         continue;

      if ((name == NULL) != (type->struct_def.name == NULL))
         continue;

      if (name && strcmp(type->struct_def.name, name))
         continue;

      if (type->struct_def.elem.num_types == num_elem_types &&
          !memcmp(type->struct_def.elem.types, elem_types,
                  sizeof(struct dxil_type *) * num_elem_types))
         return type;
   }

   type = create_type(m, TYPE_STRUCT);
   if (type) {
      if (name) {
         type->struct_def.name = ralloc_strdup(type, name);
         if (!type->struct_def.name)
            return NULL;
      } else
         type->struct_def.name = NULL;

      type->struct_def.elem.types = ralloc_array(type, struct dxil_type *,
                                                 num_elem_types);
      if (!type->struct_def.elem.types)
         return NULL;

      memcpy(type->struct_def.elem.types, elem_types,
             sizeof(struct dxil_type *) * num_elem_types);
      type->struct_def.elem.num_types = num_elem_types;
   }
   return type;
}

const struct dxil_type *
dxil_module_get_array_type(struct dxil_module *m,
                           const struct dxil_type *elem_type,
                           size_t num_elems)
{
   struct dxil_type *type;
   LIST_FOR_EACH_ENTRY(type, &m->type_list, head) {
      if (type->type != TYPE_ARRAY)
         continue;

      if (type->array_or_vector_def.elem_type == elem_type &&
          type->array_or_vector_def.num_elems == num_elems)
         return type;
   }

   type = create_type(m, TYPE_ARRAY);
   if (type) {
      type->array_or_vector_def.elem_type = elem_type;
      type->array_or_vector_def.num_elems = num_elems;
   }
   return type;
}

const struct dxil_type *
dxil_module_get_vector_type(struct dxil_module *m,
                            const struct dxil_type *elem_type,
                            size_t num_elems)
{
   struct dxil_type *type;
   LIST_FOR_EACH_ENTRY(type, &m->type_list, head) {
      if (type->type == TYPE_VECTOR &&
          type->array_or_vector_def.elem_type == elem_type &&
          type->array_or_vector_def.num_elems == num_elems)
         return type;
   }

   type = create_type(m, TYPE_VECTOR);
   if (!type)
      return NULL;

   type->array_or_vector_def.elem_type = elem_type;
   type->array_or_vector_def.num_elems = num_elems;
   return type;
}

const struct dxil_type *
dxil_get_overload_type(struct dxil_module *mod, enum overload_type overload)
{
   switch (overload) {
   case DXIL_I16: return get_int16_type(mod);
   case DXIL_I32: return get_int32_type(mod);
   case DXIL_I64: return get_int64_type(mod);
   case DXIL_F16: return get_float16_type(mod);
   case DXIL_F32: return get_float32_type(mod);
   case DXIL_F64: return get_float64_type(mod);
   default:
      unreachable("unexpected overload type");
   }
}

const struct dxil_type *
dxil_module_get_handle_type(struct dxil_module *m)
{
   const struct dxil_type *int8_type = get_int8_type(m);
   if (!int8_type)
      return NULL;

   const struct dxil_type *ptr_type = dxil_module_get_pointer_type(m, int8_type);
   if (!ptr_type)
      return NULL;

   return dxil_module_get_struct_type(m, "dx.types.Handle", &ptr_type, 1);
}

const struct dxil_type *
dxil_module_get_cbuf_ret_type(struct dxil_module *mod, enum overload_type overload)
{
   const struct dxil_type *overload_type = dxil_get_overload_type(mod, overload);
   const struct dxil_type *fields[4] = { overload_type, overload_type, overload_type, overload_type };
   unsigned num_fields;

   char name[64];
   snprintf(name, sizeof(name), "dx.types.CBufRet.%s", dxil_overload_suffix(overload));

   switch (overload) {
   case DXIL_I32:
   case DXIL_F32:
      num_fields = 4;
      break;
   case DXIL_I64:
   case DXIL_F64:
      num_fields = 2;
      break;
   default:
      unreachable("unexpected overload type");
   }

   return dxil_module_get_struct_type(mod, name, fields, num_fields);
}

const struct dxil_type *
dxil_module_get_split_double_ret_type(struct dxil_module *mod)
{
   const struct dxil_type *int32_type = dxil_module_get_int_type(mod, 32);
   const struct dxil_type *fields[2] = { int32_type, int32_type };

   return dxil_module_get_struct_type(mod, "dx.types.splitdouble", fields, 2);
}

static const struct dxil_type *
dxil_module_get_type_from_comp_type(struct dxil_module *m, enum dxil_component_type comp_type)
{
   switch (comp_type) {
   case DXIL_COMP_TYPE_U32: return get_int32_type(m);
   case DXIL_COMP_TYPE_I32: return get_int32_type(m);
   case DXIL_COMP_TYPE_F32: return get_float32_type(m);
   case DXIL_COMP_TYPE_F64: return get_float64_type(m);
   case DXIL_COMP_TYPE_U16: return get_int16_type(m);
   case DXIL_COMP_TYPE_I16: return get_int16_type(m);
   case DXIL_COMP_TYPE_U64: return get_int64_type(m);
   case DXIL_COMP_TYPE_I64: return get_int64_type(m);
   case DXIL_COMP_TYPE_I1: return get_int1_type(m);

   case DXIL_COMP_TYPE_F16:
   default:
      unreachable("unexpected component type");
   }
}

static const char *
get_res_comp_type_name(enum dxil_component_type comp_type)
{
   switch (comp_type) {
   case DXIL_COMP_TYPE_F64: return "double";
   case DXIL_COMP_TYPE_F32: return "float";
   case DXIL_COMP_TYPE_I32: return "int";
   case DXIL_COMP_TYPE_U32: return "uint";
   case DXIL_COMP_TYPE_I64: return "int64";
   case DXIL_COMP_TYPE_U64: return "uint64";
   default:
      unreachable("unexpected resource component type");
   }
}

static const char *
get_res_dimension_type_name(enum dxil_resource_kind kind)
{
   switch (kind) {
   case DXIL_RESOURCE_KIND_TYPED_BUFFER: return "Buffer";
   case DXIL_RESOURCE_KIND_TEXTURE1D: return "Texture1D";
   case DXIL_RESOURCE_KIND_TEXTURE1D_ARRAY: return "Texture1DArray";
   case DXIL_RESOURCE_KIND_TEXTURE2D: return "Texture2D";
   case DXIL_RESOURCE_KIND_TEXTURE2DMS: return "Texture2DMS";
   case DXIL_RESOURCE_KIND_TEXTURE2D_ARRAY: return "Texture2DArray";
   case DXIL_RESOURCE_KIND_TEXTURE2DMS_ARRAY: return "Texture2DMSArray";
   case DXIL_RESOURCE_KIND_TEXTURE3D: return "Texture3D";
   case DXIL_RESOURCE_KIND_TEXTURECUBE: return "TextureCube";
   case DXIL_RESOURCE_KIND_TEXTURECUBE_ARRAY: return "TextureCubeArray";
   default:
      unreachable("unexpected resource kind");
   }
}

static const char *
get_res_ms_postfix(enum dxil_resource_kind kind)
{
   switch (kind) {
   case DXIL_RESOURCE_KIND_TEXTURE2DMS:
   case DXIL_RESOURCE_KIND_TEXTURE2DMS_ARRAY:
      return ", 0";

   default:
      return " ";
   }
}
const struct dxil_type *
dxil_module_get_res_type(struct dxil_module *m, enum dxil_resource_kind kind,
                         enum dxil_component_type comp_type, bool readwrite)
{
   switch (kind) {
   case DXIL_RESOURCE_KIND_TYPED_BUFFER:
   case DXIL_RESOURCE_KIND_TEXTURE1D:
   case DXIL_RESOURCE_KIND_TEXTURE1D_ARRAY:
   case DXIL_RESOURCE_KIND_TEXTURE2D:
   case DXIL_RESOURCE_KIND_TEXTURE2D_ARRAY:
   case DXIL_RESOURCE_KIND_TEXTURE2DMS:
   case DXIL_RESOURCE_KIND_TEXTURE2DMS_ARRAY:
   case DXIL_RESOURCE_KIND_TEXTURE3D:
   case DXIL_RESOURCE_KIND_TEXTURECUBE:
   case DXIL_RESOURCE_KIND_TEXTURECUBE_ARRAY:
   {
      const struct dxil_type *component_type = dxil_module_get_type_from_comp_type(m, comp_type);
      const struct dxil_type *vec_type = dxil_module_get_vector_type(m, component_type, 4);
      char class_name[64] = { 0 };
      snprintf(class_name, 64, "class.%s%s<vector<%s, 4>%s>",
               readwrite ? "RW" : "",
               get_res_dimension_type_name(kind),
               get_res_comp_type_name(comp_type),
               get_res_ms_postfix(kind));
      return dxil_module_get_struct_type(m, class_name, &vec_type, 1);
   }

   case DXIL_RESOURCE_KIND_RAW_BUFFER:
   {
      const struct dxil_type *component_type = dxil_module_get_int_type(m, 32);
      char class_name[64] = { 0 };
      snprintf(class_name, 64, "struct.%sByteAddressBuffer", readwrite ? "RW" : "");
      return dxil_module_get_struct_type(m, class_name, &component_type, 1);
   }

   default:
      unreachable("resource type not supported");
   }
}

const struct dxil_type *
dxil_module_get_resret_type(struct dxil_module *m, enum overload_type overload)
{
   const struct dxil_type *overload_type = dxil_get_overload_type(m, overload);
   const struct dxil_type *int32_type = dxil_module_get_int_type(m, 32);
   const char *name;
   if (!overload_type)
      return NULL;

   const struct dxil_type *resret[] =
      { overload_type, overload_type, overload_type, overload_type, int32_type };

   switch (overload) {
   case DXIL_I32: name = "dx.types.ResRet.i32"; break;
   case DXIL_I64: name = "dx.types.ResRet.i64"; break;
   case DXIL_F32: name = "dx.types.ResRet.f32"; break;
   case DXIL_F64: name = "dx.types.ResRet.f64"; break;
   default:
      unreachable("unexpected overload type");
   }

   return dxil_module_get_struct_type(m, name, resret, 5);
}

const struct dxil_type *
dxil_module_get_dimret_type(struct dxil_module *m)
{
   const struct dxil_type *int32_type = dxil_module_get_int_type(m, 32);

   const struct dxil_type *dimret[] =
      { int32_type, int32_type, int32_type, int32_type };

   return dxil_module_get_struct_type(m, "dx.types.Dimensions", dimret, 4);
}

const struct dxil_type *
dxil_module_get_samplepos_type(struct dxil_module *m)
{
   const struct dxil_type *float_type = dxil_module_get_float_type(m, 32);

   const struct dxil_type *samplepos[] =
      { float_type, float_type };

   return dxil_module_get_struct_type(m, "dx.types.SamplePos", samplepos, 2);
}

const struct dxil_type *
dxil_module_get_res_bind_type(struct dxil_module *mod)
{
   /* %dx.types.ResBind = type { i32, i32, i32, i8 } */
   const struct dxil_type *int32_type = dxil_module_get_int_type(mod, 32);
   const struct dxil_type *int8_type = dxil_module_get_int_type(mod, 8);
   const struct dxil_type *fields[4] = { int32_type, int32_type, int32_type, int8_type };

   return dxil_module_get_struct_type(mod, "dx.types.ResBind", fields, 4);
}

const struct dxil_type *
dxil_module_get_res_props_type(struct dxil_module *mod)
{
   /* %dx.types.ResourceProperties = type { i32, i32 } */
   const struct dxil_type *int32_type = dxil_module_get_int_type(mod, 32);
   const struct dxil_type *fields[2] = { int32_type, int32_type };

   return dxil_module_get_struct_type(mod, "dx.types.ResourceProperties", fields, 2);
}

const struct dxil_type *
dxil_module_add_function_type(struct dxil_module *m,
                              const struct dxil_type *ret_type,
                              const struct dxil_type **arg_types,
                              size_t num_arg_types)
{
   struct dxil_type *type = create_type(m, TYPE_FUNCTION);
   if (type) {
      type->function_def.args.types = ralloc_array(type,
                                                  struct dxil_type *,
                                                  num_arg_types);
      if (!type->function_def.args.types)
         return NULL;

      memcpy(type->function_def.args.types, arg_types,
             sizeof(struct dxil_type *) * num_arg_types);
      type->function_def.args.num_types = num_arg_types;
      type->function_def.ret_type = ret_type;
   }
   return type;
}


enum type_codes {
  TYPE_CODE_NUMENTRY = 1,
  TYPE_CODE_VOID = 2,
  TYPE_CODE_FLOAT = 3,
  TYPE_CODE_DOUBLE = 4,
  TYPE_CODE_LABEL = 5,
  TYPE_CODE_OPAQUE = 6,
  TYPE_CODE_INTEGER = 7,
  TYPE_CODE_POINTER = 8,
  TYPE_CODE_FUNCTION_OLD = 9,
  TYPE_CODE_HALF = 10,
  TYPE_CODE_ARRAY = 11,
  TYPE_CODE_VECTOR = 12,
  TYPE_CODE_X86_FP80 = 13,
  TYPE_CODE_FP128 = 14,
  TYPE_CODE_PPC_FP128 = 15,
  TYPE_CODE_METADATA = 16,
  TYPE_CODE_X86_MMX = 17,
  TYPE_CODE_STRUCT_ANON = 18,
  TYPE_CODE_STRUCT_NAME = 19,
  TYPE_CODE_STRUCT_NAMED = 20,
  TYPE_CODE_FUNCTION = 21
};

#define LITERAL(x) { DXIL_OP_LITERAL, { (x) } }
#define FIXED(x) { DXIL_OP_FIXED, { (x) } }
#define VBR(x) { DXIL_OP_VBR, { (x) } }
#define ARRAY { DXIL_OP_ARRAY, { 0 } }
#define CHAR6 { DXIL_OP_CHAR6, { 0 } }
#define BLOB { DXIL_OP_BLOB, { 0 } }

#define TYPE_INDEX FIXED(32)

enum type_table_abbrev_id {
   TYPE_TABLE_ABBREV_POINTER,
   TYPE_TABLE_ABBREV_FUNCTION,
   TYPE_TABLE_ABBREV_STRUCT_ANON,
   TYPE_TABLE_ABBREV_STRUCT_NAME,
   TYPE_TABLE_ABBREV_STRUCT_NAMED,
   TYPE_TABLE_ABBREV_ARRAY,
   TYPE_TABLE_ABBREV_VECTOR,
};

static const struct dxil_abbrev
type_table_abbrevs[] = {
   [TYPE_TABLE_ABBREV_POINTER] = {
      { LITERAL(TYPE_CODE_POINTER), TYPE_INDEX, LITERAL(0) }, 3
   },
   [TYPE_TABLE_ABBREV_FUNCTION] = {
      { LITERAL(TYPE_CODE_FUNCTION), FIXED(1), ARRAY, TYPE_INDEX }, 4
   },
   [TYPE_TABLE_ABBREV_STRUCT_ANON] = {
      { LITERAL(TYPE_CODE_STRUCT_ANON), FIXED(1), ARRAY, TYPE_INDEX }, 4
   },
   [TYPE_TABLE_ABBREV_STRUCT_NAME] = {
      { LITERAL(TYPE_CODE_STRUCT_NAME), ARRAY, CHAR6 }, 3
   },
   [TYPE_TABLE_ABBREV_STRUCT_NAMED] = {
      { LITERAL(TYPE_CODE_STRUCT_NAMED), FIXED(1), ARRAY, TYPE_INDEX }, 4
   },
   [TYPE_TABLE_ABBREV_ARRAY] = {
      { LITERAL(TYPE_CODE_ARRAY), VBR(8), TYPE_INDEX }, 3
   },
   [TYPE_TABLE_ABBREV_VECTOR] = {
      { LITERAL(TYPE_CODE_VECTOR), VBR(8), TYPE_INDEX }, 3
   },
};

static bool
emit_type_table_abbrev_record(struct dxil_module *m,
                              enum type_table_abbrev_id abbrev,
                              const uint64_t *data, size_t size)
{
   assert(abbrev < ARRAY_SIZE(type_table_abbrevs));
   return emit_record_abbrev(&m->buf, abbrev + DXIL_FIRST_APPLICATION_ABBREV,
                             type_table_abbrevs + abbrev, data, size);
}

enum constant_code {
  CST_CODE_SETTYPE = 1,
  CST_CODE_NULL = 2,
  CST_CODE_UNDEF = 3,
  CST_CODE_INTEGER = 4,
  CST_CODE_WIDE_INTEGER = 5,
  CST_CODE_FLOAT = 6,
  CST_CODE_AGGREGATE = 7,
  CST_CODE_STRING = 8,
  CST_CODE_CSTRING = 9,
  CST_CODE_CE_BINOP = 10,
  CST_CODE_CE_CAST = 11,
  CST_CODE_CE_GEP = 12,
  CST_CODE_CE_SELECT = 13,
  CST_CODE_CE_EXTRACTELT = 14,
  CST_CODE_CE_INSERTELT = 15,
  CST_CODE_CE_SHUFFLEVEC = 16,
  CST_CODE_CE_CMP = 17,
  CST_CODE_INLINEASM_OLD = 18,
  CST_CODE_CE_SHUFVEC_EX = 19,
  CST_CODE_CE_INBOUNDS_GEP = 20,
  CST_CODE_BLOCKADDRESS = 21,
  CST_CODE_DATA = 22,
  CST_CODE_INLINEASM = 23
};

enum const_abbrev_id {
   CONST_ABBREV_SETTYPE,
   CONST_ABBREV_INTEGER,
   CONST_ABBREV_CE_CAST,
   CONST_ABBREV_NULL,
};

static const struct dxil_abbrev
const_abbrevs[] = {
   [CONST_ABBREV_SETTYPE] = { { LITERAL(CST_CODE_SETTYPE), TYPE_INDEX }, 2 },
   [CONST_ABBREV_INTEGER] = { { LITERAL(CST_CODE_INTEGER), VBR(8) }, 2 },
   [CONST_ABBREV_CE_CAST] = {
      { LITERAL(CST_CODE_CE_CAST), FIXED(4), TYPE_INDEX, VBR(8) }, 4
   },
   [CONST_ABBREV_NULL] = { { LITERAL(CST_CODE_NULL) }, 1 },
};

static bool
emit_const_abbrev_record(struct dxil_module *m, enum const_abbrev_id abbrev,
                         const uint64_t *data, size_t size)
{
   assert(abbrev < ARRAY_SIZE(const_abbrevs));

   return emit_record_abbrev(&m->buf, abbrev + DXIL_FIRST_APPLICATION_ABBREV,
                             const_abbrevs + abbrev, data, size);
}

enum function_code {
  FUNC_CODE_DECLAREBLOCKS = 1,
  FUNC_CODE_INST_BINOP = 2,
  FUNC_CODE_INST_CAST = 3,
  FUNC_CODE_INST_GEP_OLD = 4,
  FUNC_CODE_INST_SELECT = 5,
  FUNC_CODE_INST_EXTRACTELT = 6,
  FUNC_CODE_INST_INSERTELT = 7,
  FUNC_CODE_INST_SHUFFLEVEC = 8,
  FUNC_CODE_INST_CMP = 9,
  FUNC_CODE_INST_RET = 10,
  FUNC_CODE_INST_BR = 11,
  FUNC_CODE_INST_SWITCH = 12,
  FUNC_CODE_INST_INVOKE = 13,
  /* 14: unused */
  FUNC_CODE_INST_UNREACHABLE = 15,
  FUNC_CODE_INST_PHI = 16,
  /* 17-18: unused */
  FUNC_CODE_INST_ALLOCA = 19,
  FUNC_CODE_INST_LOAD = 20,
  /* 21-22: unused */
  FUNC_CODE_INST_VAARG = 23,
  FUNC_CODE_INST_STORE_OLD = 24,
  /* 25: unused */
  FUNC_CODE_INST_EXTRACTVAL = 26,
  FUNC_CODE_INST_INSERTVAL = 27,
  FUNC_CODE_INST_CMP2 = 28,
  FUNC_CODE_INST_VSELECT = 29,
  FUNC_CODE_INST_INBOUNDS_GEP_OLD = 30,
  FUNC_CODE_INST_INDIRECTBR = 31,
  /* 32: unused */
  FUNC_CODE_DEBUG_LOC_AGAIN = 33,
  FUNC_CODE_INST_CALL = 34,
  FUNC_CODE_DEBUG_LOC = 35,
  FUNC_CODE_INST_FENCE = 36,
  FUNC_CODE_INST_CMPXCHG_OLD = 37,
  FUNC_CODE_INST_ATOMICRMW = 38,
  FUNC_CODE_INST_RESUME = 39,
  FUNC_CODE_INST_LANDINGPAD_OLD = 40,
  FUNC_CODE_INST_LOADATOMIC = 41,
  FUNC_CODE_INST_STOREATOMIC_OLD = 42,
  FUNC_CODE_INST_GEP = 43,
  FUNC_CODE_INST_STORE = 44,
  FUNC_CODE_INST_STOREATOMIC = 45,
  FUNC_CODE_INST_CMPXCHG = 46,
  FUNC_CODE_INST_LANDINGPAD = 47,
};

enum func_abbrev_id {
   FUNC_ABBREV_LOAD,
   FUNC_ABBREV_BINOP,
   FUNC_ABBREV_BINOP_FLAGS,
   FUNC_ABBREV_CAST,
   FUNC_ABBREV_RET_VOID,
   FUNC_ABBREV_RET_VAL,
   FUNC_ABBREV_UNREACHABLE,
   FUNC_ABBREV_GEP,
};

static const struct dxil_abbrev
func_abbrevs[] = {
   [FUNC_ABBREV_LOAD] = {
      { LITERAL(FUNC_CODE_INST_LOAD), VBR(6), TYPE_INDEX, VBR(4),
        FIXED(1) }, 5
   },
   [FUNC_ABBREV_BINOP] = {
      { LITERAL(FUNC_CODE_INST_BINOP), VBR(6), VBR(6), FIXED(4) }, 4
   },
   [FUNC_ABBREV_BINOP_FLAGS] = {
      { LITERAL(FUNC_CODE_INST_BINOP), VBR(6), VBR(6), FIXED(4),
        FIXED(7) }, 5
   },
   [FUNC_ABBREV_CAST] = {
      { LITERAL(FUNC_CODE_INST_CAST), VBR(6), TYPE_INDEX, FIXED(4) }, 4
   },
   [FUNC_ABBREV_RET_VOID] = { { LITERAL(FUNC_CODE_INST_RET) }, 1 },
   [FUNC_ABBREV_RET_VAL] = { { LITERAL(FUNC_CODE_INST_RET), VBR(6) }, 2 },
   [FUNC_ABBREV_UNREACHABLE] = {
      { LITERAL(FUNC_CODE_INST_UNREACHABLE) }, 1
   },
   [FUNC_ABBREV_GEP] = {
      { LITERAL(FUNC_CODE_INST_GEP), FIXED(1), TYPE_INDEX, ARRAY,
        VBR(6) }, 5
   },
};

static bool
emit_func_abbrev_record(struct dxil_module *m, enum func_abbrev_id abbrev,
                        const uint64_t *data, size_t size)
{
   assert(abbrev < ARRAY_SIZE(func_abbrevs));
   return emit_record_abbrev(&m->buf, abbrev + DXIL_FIRST_APPLICATION_ABBREV,
                             func_abbrevs + abbrev, data, size);
}

static bool
define_abbrev(struct dxil_module *m, const struct dxil_abbrev *a)
{
   if (!dxil_buffer_emit_abbrev_id(&m->buf, DXIL_DEFINE_ABBREV) ||
       !dxil_buffer_emit_vbr_bits(&m->buf, a->num_operands, 5))
      return false;

   for (int i = 0; i < a->num_operands; ++i) {
      unsigned is_literal = a->operands[i].type == DXIL_OP_LITERAL;
      if (!dxil_buffer_emit_bits(&m->buf, is_literal, 1))
         return false;
      if (a->operands[i].type == DXIL_OP_LITERAL) {
         if (!dxil_buffer_emit_vbr_bits(&m->buf, a->operands[i].value, 8))
            return false;
      } else {
         if (!dxil_buffer_emit_bits(&m->buf, a->operands[i].type, 3))
            return false;
         if (a->operands[i].type == DXIL_OP_FIXED) {
            if (!dxil_buffer_emit_vbr_bits(&m->buf,
                                           a->operands[i].encoding_data, 5))
               return false;
         } else if (a->operands[i].type == DXIL_OP_VBR) {
            if (!dxil_buffer_emit_vbr_bits(&m->buf,
                                           a->operands[i].encoding_data, 5))
               return false;
         }
      }
   }

   return true;
}

enum dxil_blockinfo_code {
   DXIL_BLOCKINFO_CODE_SETBID = 1,
   DXIL_BLOCKINFO_CODE_BLOCKNAME = 2,
   DXIL_BLOCKINFO_CODE_SETRECORDNAME = 3
};

static bool
switch_to_block(struct dxil_module *m, uint32_t block)
{
   return emit_record_int(m, DXIL_BLOCKINFO_CODE_SETBID, block);
}

enum dxil_standard_block {
   DXIL_BLOCKINFO = 0,
   DXIL_FIRST_APPLICATION_BLOCK = 8
};

enum dxil_llvm_block {
   DXIL_MODULE = DXIL_FIRST_APPLICATION_BLOCK,
   DXIL_PARAMATTR = DXIL_FIRST_APPLICATION_BLOCK + 1,
   DXIL_PARAMATTR_GROUP = DXIL_FIRST_APPLICATION_BLOCK + 2,
   DXIL_CONST_BLOCK = DXIL_FIRST_APPLICATION_BLOCK + 3,
   DXIL_FUNCTION_BLOCK = DXIL_FIRST_APPLICATION_BLOCK + 4,
   DXIL_VALUE_SYMTAB_BLOCK = DXIL_FIRST_APPLICATION_BLOCK + 6,
   DXIL_METADATA_BLOCK = DXIL_FIRST_APPLICATION_BLOCK + 7,
   DXIL_TYPE_BLOCK = DXIL_FIRST_APPLICATION_BLOCK + 9,
};

enum value_symtab_code {
  VST_CODE_ENTRY = 1,
  VST_CODE_BBENTRY = 2
};

enum value_symtab_abbrev_id {
   VST_ABBREV_ENTRY_8,
   VST_ABBREV_ENTRY_7,
   VST_ABBREV_ENTRY_6,
   VST_ABBREV_BBENTRY_6,
};

static struct dxil_abbrev value_symtab_abbrevs[] = {
   [VST_ABBREV_ENTRY_8] = { { FIXED(3), VBR(8), ARRAY, FIXED(8) }, 4 },
   [VST_ABBREV_ENTRY_7] = {
      { LITERAL(VST_CODE_ENTRY), VBR(8), ARRAY, FIXED(7), }, 4
   },
   [VST_ABBREV_ENTRY_6] = {
      { LITERAL(VST_CODE_ENTRY), VBR(8), ARRAY, CHAR6, }, 4
   },
   [VST_ABBREV_BBENTRY_6] = {
      { LITERAL(VST_CODE_BBENTRY), VBR(8), ARRAY, CHAR6, }, 4
   },
};

static bool
emit_value_symtab_abbrevs(struct dxil_module *m)
{
   if (!switch_to_block(m, DXIL_VALUE_SYMTAB_BLOCK))
      return false;

   for (int i = 0; i < ARRAY_SIZE(value_symtab_abbrevs); ++i) {
      if (!define_abbrev(m, value_symtab_abbrevs + i))
         return false;
   }

   return true;
}

static bool
emit_const_abbrevs(struct dxil_module *m)
{
   if (!switch_to_block(m, DXIL_CONST_BLOCK))
      return false;

   for (int i = 0; i < ARRAY_SIZE(const_abbrevs); ++i) {
      if (!define_abbrev(m, const_abbrevs + i))
         return false;
   }

   return true;
}

static bool
emit_function_abbrevs(struct dxil_module *m)
{
   if (!switch_to_block(m, DXIL_FUNCTION_BLOCK))
      return false;

   for (int i = 0; i < ARRAY_SIZE(func_abbrevs); ++i) {
      if (!define_abbrev(m, func_abbrevs + i))
         return false;
   }

   return true;
}

static bool
emit_blockinfo(struct dxil_module *m)
{
   return enter_subblock(m, DXIL_BLOCKINFO, 2) &&
          emit_value_symtab_abbrevs(m) &&
          emit_const_abbrevs(m) &&
          emit_function_abbrevs(m) &&
          exit_block(m);
}

enum attribute_codes {
   PARAMATTR_GRP_CODE_ENTRY = 3,
   PARAMATTR_CODE_ENTRY = 2
};

static bool
emit_attrib_group(struct dxil_module *m, int id, uint32_t slot,
                  const struct dxil_attrib *attrs, size_t num_attrs)
{
   uint64_t record[64];
   record[0] = id;
   record[1] = slot;
   size_t size = 2;

   for (int i = 0; i < num_attrs; ++i) {
      switch (attrs[i].type) {
      case DXIL_ATTR_ENUM:
         assert(size < ARRAY_SIZE(record) - 2);
         record[size++] = 0;
         record[size++] = attrs[i].kind;
         break;

      default:
         unreachable("unsupported attrib type");
      }
   }

   return emit_record(m, PARAMATTR_GRP_CODE_ENTRY, record, size);
}

static bool
emit_attrib_group_table(struct dxil_module *m)
{
   if (!enter_subblock(m, DXIL_PARAMATTR_GROUP, 3))
      return false;

   struct attrib_set *as;
   int id = 1;
   LIST_FOR_EACH_ENTRY(as, &m->attr_set_list, head) {
      if (!emit_attrib_group(m, id, UINT32_MAX, as->attrs, as->num_attrs))
         return false;
      id++;
   }

   return exit_block(m);
}

static bool
emit_attribute_table(struct dxil_module *m)
{
   if (!enter_subblock(m, DXIL_PARAMATTR, 3))
      return false;

   struct attrib_set *as;
   int id = 1;
   LIST_FOR_EACH_ENTRY(as, &m->attr_set_list, head) {
      if (!emit_record_int(m, PARAMATTR_CODE_ENTRY, id))
         return false;
      id++;
   }

   return exit_block(m);
}

static bool
emit_type_table_abbrevs(struct dxil_module *m)
{
   for (int i = 0; i < ARRAY_SIZE(type_table_abbrevs); ++i) {
      if (!define_abbrev(m, type_table_abbrevs + i))
         return false;
   }

   return true;
}

static bool
emit_float_type(struct dxil_module *m, unsigned bit_size)
{
   switch (bit_size) {
   case 16: return emit_record(m, TYPE_CODE_HALF, NULL, 0);
   case 32: return emit_record(m, TYPE_CODE_FLOAT, NULL, 0);
   case 64: return emit_record(m, TYPE_CODE_DOUBLE, NULL, 0);
   default:
      unreachable("unexpected bit_size for float type");
   }
}

static bool
emit_pointer_type(struct dxil_module *m, int type_index)
{
   uint64_t data[] = { TYPE_CODE_POINTER, type_index, 0 };
   return emit_type_table_abbrev_record(m, TYPE_TABLE_ABBREV_POINTER,
                                        data, ARRAY_SIZE(data));
}

static bool
emit_struct_name(struct dxil_module *m, const char *name)
{
   uint64_t temp[256];
   assert(strlen(name) < ARRAY_SIZE(temp));

   for (int i = 0; i < strlen(name); ++i)
      temp[i] = name[i];

   return emit_record(m, TYPE_CODE_STRUCT_NAME, temp, strlen(name));
}

static bool
emit_struct_name_char6(struct dxil_module *m, const char *name)
{
   uint64_t temp[256];
   assert(strlen(name) < ARRAY_SIZE(temp) - 1);

   temp[0] = TYPE_CODE_STRUCT_NAME;
   for (int i = 0; i < strlen(name); ++i)
      temp[i + 1] = name[i];

   return emit_type_table_abbrev_record(m, TYPE_TABLE_ABBREV_STRUCT_NAME,
                                        temp, 1 + strlen(name));
}

static bool
emit_struct_type(struct dxil_module *m, const struct dxil_type *type)
{
   enum type_table_abbrev_id abbrev = TYPE_TABLE_ABBREV_STRUCT_ANON;
   enum type_codes type_code = TYPE_CODE_STRUCT_ANON;
   if (type->struct_def.name) {
      abbrev = TYPE_TABLE_ABBREV_STRUCT_NAMED;
      type_code = TYPE_CODE_STRUCT_NAMED;
      if (is_char6_string(type->struct_def.name)) {
         if (!emit_struct_name_char6(m, type->struct_def.name))
            return false;
      } else {
         if (!emit_struct_name(m, type->struct_def.name))
            return false;
      }
   }

   uint64_t temp[256];
   assert(type->struct_def.elem.num_types < ARRAY_SIZE(temp) - 2);
   temp[0] = type_code;
   temp[1] = 0; /* packed */
   for (int i = 0; i < type->struct_def.elem.num_types; ++i) {
      assert(type->struct_def.elem.types[i]->id >= 0);
      temp[2 + i] = type->struct_def.elem.types[i]->id;
   }

   return emit_type_table_abbrev_record(m, abbrev, temp,
                                        2 + type->struct_def.elem.num_types);
}

static bool
emit_array_type(struct dxil_module *m, const struct dxil_type *type)
{
   assert(type->array_or_vector_def.elem_type->id >= 0);
   uint64_t data[] = {
      TYPE_CODE_ARRAY,
      type->array_or_vector_def.num_elems,
      type->array_or_vector_def.elem_type->id
   };
   return emit_type_table_abbrev_record(m, TYPE_TABLE_ABBREV_ARRAY, data,
                                        ARRAY_SIZE(data));
}

static bool
emit_function_type(struct dxil_module *m, const struct dxil_type *type)
{
   uint64_t temp[256];
   assert(type->function_def.args.num_types < ARRAY_SIZE(temp) - 3);
   assert(type->function_def.ret_type->id >= 0);

   temp[0] = TYPE_CODE_FUNCTION;
   temp[1] = 0; // vararg
   temp[2] = type->function_def.ret_type->id;
   for (int i = 0; i < type->function_def.args.num_types; ++i) {
      assert(type->function_def.args.types[i]->id >= 0);
      temp[3 + i] = type->function_def.args.types[i]->id;
   }

   return emit_type_table_abbrev_record(m, TYPE_TABLE_ABBREV_FUNCTION,
                                        temp, 3 + type->function_def.args.num_types);
}

static bool
emit_vector_type(struct dxil_module *m, const struct dxil_type *type)
{
   uint64_t temp[3];
   temp[0] = TYPE_CODE_VECTOR;
   temp[1] = type->array_or_vector_def.num_elems;
   temp[2] = type->array_or_vector_def.elem_type->id;

   return emit_type_table_abbrev_record(m, TYPE_TABLE_ABBREV_VECTOR , temp, 3);
}

static bool
emit_metadata_type(struct dxil_module *m)
{
   return emit_record(m, TYPE_CODE_METADATA, NULL, 0);
}

static bool
emit_type(struct dxil_module *m, struct dxil_type *type)
{
   switch (type->type) {
   case TYPE_VOID:
      return emit_record(m, TYPE_CODE_VOID, NULL, 0);

   case TYPE_INTEGER:
      return emit_record_int(m, TYPE_CODE_INTEGER, type->int_bits);

   case TYPE_FLOAT:
      return emit_float_type(m, type->float_bits);

   case TYPE_POINTER:
      return emit_pointer_type(m, type->ptr_target_type->id);

   case TYPE_STRUCT:
      return emit_struct_type(m, type);

   case TYPE_ARRAY:
      return emit_array_type(m, type);

   case TYPE_FUNCTION:
      return emit_function_type(m, type);

   case TYPE_VECTOR:
      return emit_vector_type(m, type);

   default:
      unreachable("unexpected type->type");
   }
}

static bool
emit_type_table(struct dxil_module *m)
{
   if (!enter_subblock(m, DXIL_TYPE_BLOCK, 4) ||
       !emit_type_table_abbrevs(m) ||
       !emit_record_int(m, 1, 1 + list_length(&m->type_list)))
      return false;

   list_for_each_entry(struct dxil_type, type, &m->type_list, head) {
      if (!emit_type(m, type))
         return false;
   }

   return emit_metadata_type(m) &&
          exit_block(m);
}

static struct dxil_const *
create_const(struct dxil_module *m, const struct dxil_type *type, bool undef)
{
   struct dxil_const *ret = ralloc_size(m->ralloc_ctx,
                                        sizeof(struct dxil_const));
   if (ret) {
      ret->value.id = -1;
      ret->value.type = type;
      ret->undef = undef;
      list_addtail(&ret->head, &m->const_list);
   }
   return ret;
}

static const struct dxil_value *
get_int_const(struct dxil_module *m, const struct dxil_type *type,
              intmax_t value)
{
   assert(type && type->type == TYPE_INTEGER);

   struct dxil_const *c;
   LIST_FOR_EACH_ENTRY(c, &m->const_list, head) {
      if (c->value.type != type || c->undef)
         continue;

      if (c->int_value == value)
         return &c->value;
   }

   c = create_const(m, type, false);
   if (!c)
      return NULL;

   c->int_value = value;
   return &c->value;
}

static intmax_t
get_int_from_const_value(const struct dxil_value *value)
{
   assert(value->type->type == TYPE_INTEGER);
   const struct dxil_const *c = container_of(value, const struct dxil_const, value);
   return c->int_value;
}

const struct dxil_value *
dxil_module_get_int1_const(struct dxil_module *m, bool value)
{
   const struct dxil_type *type = get_int1_type(m);
   if (!type)
      return NULL;

   return get_int_const(m, type, value);
}

const struct dxil_value *
dxil_module_get_int8_const(struct dxil_module *m, int8_t value)
{
   const struct dxil_type *type = get_int8_type(m);
   if (!type)
      return NULL;

   return get_int_const(m, type, value);
}

const struct dxil_value *
dxil_module_get_int16_const(struct dxil_module *m, int16_t value)
{
   const struct dxil_type *type = get_int16_type(m);
   if (!type)
      return NULL;

   return get_int_const(m, type, value);
}

const struct dxil_value *
dxil_module_get_int32_const(struct dxil_module *m, int32_t value)
{
   const struct dxil_type *type = get_int32_type(m);
   if (!type)
      return NULL;

   return get_int_const(m, type, value);
}

const struct dxil_value *
dxil_module_get_int64_const(struct dxil_module *m, int64_t value)
{
   const struct dxil_type *type = get_int64_type(m);
   if (!type)
      return NULL;

   return get_int_const(m, type, value);
}

const struct dxil_value *
dxil_module_get_int_const(struct dxil_module *m, intmax_t value,
                          unsigned bit_size)
{
   switch (bit_size) {
   case 1:
      assert(value == 0 || value == 1);
      return dxil_module_get_int1_const(m, value);

   case 8:
      assert(INT8_MIN <= value && value <= INT8_MAX);
      return dxil_module_get_int8_const(m, value);

   case 16:
      assert(INT16_MIN <= value && value <= INT16_MAX);
      return dxil_module_get_int16_const(m, value);

   case 32:
      assert(INT32_MIN <= value && value <= INT32_MAX);
      return dxil_module_get_int32_const(m, value);

   case 64:
      assert(INT64_MIN <= value && value <= INT64_MAX);
      return dxil_module_get_int64_const(m, value);

   default:
      unreachable("unsupported bit-width");
   }
}

const struct dxil_value *
dxil_module_get_float16_const(struct dxil_module *m, uint16_t value)
{
   const struct dxil_type *type = get_float16_type(m);
   if (!type)
      return NULL;

   struct dxil_const *c;
   LIST_FOR_EACH_ENTRY(c, &m->const_list, head) {
      if (c->value.type != type || c->undef)
         continue;

      if (c->int_value == (uintmax_t)value)
         return &c->value;
   }

   c = create_const(m, type, false);
   if (!c)
      return NULL;

   c->int_value = (uintmax_t)value;
   return &c->value;
}

const struct dxil_value *
dxil_module_get_float_const(struct dxil_module *m, float value)
{
   const struct dxil_type *type = get_float32_type(m);
   if (!type)
      return NULL;

   struct dxil_const *c;
   LIST_FOR_EACH_ENTRY(c, &m->const_list, head) {
      if (c->value.type != type || c->undef)
         continue;

      if (c->float_value == value)
         return &c->value;
   }

   c = create_const(m, type, false);
   if (!c)
      return NULL;

   c->float_value = value;
   return &c->value;
}

const struct dxil_value *
dxil_module_get_double_const(struct dxil_module *m, double value)
{
   const struct dxil_type *type = get_float64_type(m);
   if (!type)
      return NULL;

   struct dxil_const *c;
   LIST_FOR_EACH_ENTRY(c, &m->const_list, head) {
      if (c->value.type != type || c->undef)
         continue;

      if (c->float_value == value)
         return &c->value;
   }

   c = create_const(m, type, false);
   if (!c)
      return NULL;

   c->float_value = value;
   return &c->value;
}

const struct dxil_value *
dxil_module_get_array_const(struct dxil_module *m, const struct dxil_type *type,
                            const struct dxil_value **values)
{
   assert(type->type == TYPE_ARRAY);
   unsigned int num_values = type->array_or_vector_def.num_elems;

   struct dxil_const *c;
   LIST_FOR_EACH_ENTRY(c, &m->const_list, head) {
      if (c->value.type != type || c->undef)
         continue;

      if (!memcmp(c->array_values, values, sizeof(*values) * num_values))
         return &c->value;
   }

   c = create_const(m, type, false);
   if (!c)
      return NULL;
   void *tmp =
      ralloc_array(m->ralloc_ctx, struct dxil_value *, num_values);
   memcpy(tmp, values, sizeof(*values) * num_values);
   c->array_values = tmp;

   return &c->value;
}

const struct dxil_value *
dxil_module_get_undef(struct dxil_module *m, const struct dxil_type *type)
{
   assert(type != NULL);

   struct dxil_const *c;
   LIST_FOR_EACH_ENTRY(c, &m->const_list, head) {
      if (c->value.type != type)
         continue;

      if (c->undef)
         return &c->value;
   }

   c = create_const(m, type, true);
   return c ? &c->value : NULL;
}

static const struct dxil_value *
get_struct_const(struct dxil_module *m, const struct dxil_type *type,
                 const struct dxil_value **values)
{
   assert(type->type == TYPE_STRUCT);
   unsigned int num_values = type->struct_def.elem.num_types;

   struct dxil_const *c;
   LIST_FOR_EACH_ENTRY(c, &m->const_list, head) {
      if (c->value.type != type || c->undef)
         continue;

      if (!memcmp(c->struct_values, values, sizeof(*values) * num_values))
         return &c->value;
   }

   c = create_const(m, type, false);
   if (!c)
      return NULL;
   void *tmp =
      ralloc_array(m->ralloc_ctx, struct dxil_value *, num_values);
   memcpy(tmp, values, sizeof(*values) * num_values);
   c->struct_values = tmp;

   return &c->value;
}

const struct dxil_value *
dxil_module_get_res_bind_const(struct dxil_module *m,
                               uint32_t lower_bound,
                               uint32_t upper_bound,
                               uint32_t space,
                               uint8_t class)
{
   const struct dxil_type *type = dxil_module_get_res_bind_type(m);
   const struct dxil_type *int32_type = dxil_module_get_int_type(m, 32);
   const struct dxil_type *int8_type = dxil_module_get_int_type(m, 8);
   if (!type || !int32_type || !int8_type)
      return NULL;

   const struct dxil_value *values[4] = {
      get_int_const(m, int32_type, lower_bound),
      get_int_const(m, int32_type, upper_bound),
      get_int_const(m, int32_type, space),
      get_int_const(m, int8_type, class),
   };
   if (!values[0] || !values[1] || !values[2] || !values[3])
      return NULL;

   return get_struct_const(m, type, values);
}

static uint32_t
get_basic_srv_uav_res_props_dword(bool uav,
                                  bool rov,
                                  bool globally_coherent,
                                  bool has_counter,
                                  enum dxil_resource_kind kind)
{
   union {
      uint32_t raw;
      struct {
         uint8_t kind;

         uint8_t base_align_log2 : 4;
         uint8_t uav : 1;
         uint8_t rov : 1;
         uint8_t globally_coherent : 1;
         uint8_t has_counter : 1;
      };
   } basic;
   basic.raw = 0;
   basic.kind = kind;
   basic.uav = uav;
   basic.rov = rov;
   basic.globally_coherent = globally_coherent;
   basic.has_counter = has_counter;
   return basic.raw;
}

static uint32_t
get_typed_srv_uav_res_props_dword(enum dxil_component_type comp_type,
                                  uint8_t num_components,
                                  uint8_t sample_count)
{
   union {
      uint32_t raw;
      struct {
         uint8_t comp_type;
         uint8_t num_components;
         uint8_t sample_count;
      };
   } type;
   type.raw = 0;
   type.comp_type = comp_type;
   type.num_components = num_components;
   type.sample_count = sample_count;
   return type.raw;
}

static uint32_t
get_sampler_res_props_dword(bool comparison)
{
   union {
      uint32_t raw;
      struct {
         uint8_t kind;

         uint8_t padding : 7;
         uint8_t comparison : 1;
      };
   } basic;
   basic.raw = 0;
   basic.kind = DXIL_RESOURCE_KIND_SAMPLER;
   basic.comparison = comparison;
   return basic.raw;
}

static intmax_t
get_int_from_mdnode(const struct dxil_mdnode *mdnode, int subnode)
{
   assert(mdnode->type == MD_NODE);
   assert(mdnode->node.subnodes[subnode]->type == MD_VALUE);
   return get_int_from_const_value(mdnode->node.subnodes[subnode]->value.value);
}

static void
fill_res_props_dwords(uint32_t dwords[2],
                      enum dxil_resource_class class,
                      const struct dxil_mdnode *mdnode)
{
   enum dxil_resource_kind kind = DXIL_RESOURCE_KIND_INVALID;
   uint32_t sample_count = 0;
   switch (class) {
   case DXIL_RESOURCE_CLASS_SRV:
      kind = (enum dxil_resource_kind)get_int_from_mdnode(mdnode, 6);
      dwords[0] = get_basic_srv_uav_res_props_dword(false, false, false, false, kind);
      sample_count = get_int_from_mdnode(mdnode, 7);
      break;
   case DXIL_RESOURCE_CLASS_UAV:
      kind = (enum dxil_resource_kind)get_int_from_mdnode(mdnode, 6);
      dwords[0] = get_basic_srv_uav_res_props_dword(true,
         get_int_from_mdnode(mdnode, 9),
         get_int_from_mdnode(mdnode, 7),
         get_int_from_mdnode(mdnode, 6),
         kind);
      break;
   case DXIL_RESOURCE_CLASS_CBV:
      kind = DXIL_RESOURCE_KIND_CBUFFER;
      dwords[0] = kind;
      break;
   case DXIL_RESOURCE_CLASS_SAMPLER:
      kind = DXIL_RESOURCE_KIND_SAMPLER;
      dwords[0] = get_sampler_res_props_dword(get_int_from_mdnode(mdnode, 6) == DXIL_SAMPLER_KIND_COMPARISON);
      break;
   default:
      unreachable("Unexpected resource class");
   }

   switch (kind) {
   case DXIL_RESOURCE_KIND_STRUCTURED_BUFFER:
   case DXIL_RESOURCE_KIND_INVALID:
      unreachable("Unimplemented");
   case DXIL_RESOURCE_KIND_RAW_BUFFER:
   case DXIL_RESOURCE_KIND_SAMPLER:
      dwords[1] = 0;
      break;
   case DXIL_RESOURCE_KIND_CBUFFER:
      dwords[1] = get_int_from_mdnode(mdnode, 6);
      break;
   default: {
      unsigned tag_array_index = class == DXIL_RESOURCE_CLASS_SRV ? 8 : 10;
      const struct dxil_type *res_ptr_type = mdnode->node.subnodes[1]->value.type;
      const struct dxil_type *res_type = res_ptr_type->ptr_target_type->type == TYPE_ARRAY ?
         res_ptr_type->ptr_target_type->array_or_vector_def.elem_type : res_ptr_type->ptr_target_type;
      const struct dxil_type *vec_type = res_type->struct_def.elem.types[0];
      dwords[1] = get_typed_srv_uav_res_props_dword(
         (enum dxil_component_type)get_int_from_mdnode(
            mdnode->node.subnodes[tag_array_index], 1),
         vec_type->array_or_vector_def.num_elems,
         sample_count);
      break;
      }
   }
}

const struct dxil_value *
dxil_module_get_res_props_const(struct dxil_module *m,
                                enum dxil_resource_class class,
                                const struct dxil_mdnode *mdnode)
{
   const struct dxil_type *type = dxil_module_get_res_props_type(m);
   if (!type)
      return NULL;

   uint32_t dwords[2];
   fill_res_props_dwords(dwords, class, mdnode);

   const struct dxil_value *values[2] = {
      dxil_module_get_int32_const(m, dwords[0]),
      dxil_module_get_int32_const(m, dwords[1])
   };
   if (!values[0] || !values[1])
      return NULL;

   return get_struct_const(m, type, values);
}

enum dxil_module_code {
   DXIL_MODULE_CODE_VERSION = 1,
   DXIL_MODULE_CODE_TRIPLE = 2,
   DXIL_MODULE_CODE_DATALAYOUT = 3,
   DXIL_MODULE_CODE_ASM = 4,
   DXIL_MODULE_CODE_SECTIONNAME = 5,
   DXIL_MODULE_CODE_DEPLIB = 6,
   DXIL_MODULE_CODE_GLOBALVAR = 7,
   DXIL_MODULE_CODE_FUNCTION = 8,
   DXIL_MODULE_CODE_ALIAS = 9,
   DXIL_MODULE_CODE_PURGEVALS = 10,
   DXIL_MODULE_CODE_GCNAME = 11,
   DXIL_MODULE_CODE_COMDAT = 12,
};

static bool
emit_target_triple(struct dxil_module *m, const char *triple)
{
   uint64_t temp[256];
   assert(strlen(triple) < ARRAY_SIZE(temp));

   for (int i = 0; i < strlen(triple); ++i)
      temp[i] = triple[i];

   return emit_record(m, DXIL_MODULE_CODE_TRIPLE, temp, strlen(triple));
}

static bool
emit_datalayout(struct dxil_module *m, const char *datalayout)
{
   uint64_t temp[256];
   assert(strlen(datalayout) < ARRAY_SIZE(temp));

   for (int i = 0; i < strlen(datalayout); ++i)
      temp[i] = datalayout[i];

   return emit_record(m, DXIL_MODULE_CODE_DATALAYOUT,
                      temp, strlen(datalayout));
}

static const struct dxil_value *
add_gvar(struct dxil_module *m, const char *name,
         const struct dxil_type *type, const struct dxil_type *value_type,
         enum dxil_address_space as, int align, const struct dxil_value *value)
{
   struct dxil_gvar *gvar = ralloc_size(m->ralloc_ctx,
                                        sizeof(struct dxil_gvar));
   if (!gvar)
      return NULL;

   gvar->type = type;
   gvar->name = ralloc_strdup(m->ralloc_ctx, name);
   gvar->as = as;
   gvar->align = align;
   gvar->constant = !!value;
   gvar->initializer = value;

   gvar->value.id = -1;
   gvar->value.type = value_type;

   list_addtail(&gvar->head, &m->gvar_list);
   return &gvar->value;
}

const struct dxil_value *
dxil_add_global_var(struct dxil_module *m, const char *name,
                    const struct dxil_type *type,
                    enum dxil_address_space as, int align,
                    const struct dxil_value *value)
{
   return add_gvar(m, name, type, type, as, align, value);
}

const struct dxil_value *
dxil_add_global_ptr_var(struct dxil_module *m, const char *name,
                        const struct dxil_type *type,
                        enum dxil_address_space as, int align,
                        const struct dxil_value *value)
{
   return add_gvar(m, name, type, dxil_module_get_pointer_type(m, type),
                   as, align, value);
}

static const struct dxil_func *
add_function(struct dxil_module *m, const char *name,
             const struct dxil_type *type,
             bool decl, unsigned attr_set)
{
   assert(type->type == TYPE_FUNCTION);

   struct dxil_func *func = ralloc_size(m->ralloc_ctx,
                                        sizeof(struct dxil_func));
   if (!func)
      return NULL;

   /* Truncate function name to make emit_symtab_entry() happy. */
   func->name = ralloc_strndup(func, name, 253);
   if (!func->name) {
      return NULL;
   }

   func->type = type;
   func->decl = decl;
   func->attr_set = attr_set;

   func->value.id = -1;
   func->value.type  = type->function_def.ret_type;
   list_addtail(&func->head, &m->func_list);
   return func;
}

struct dxil_func_def *
dxil_add_function_def(struct dxil_module *m, const char *name,
                      const struct dxil_type *type, unsigned num_blocks)
{
   struct dxil_func_def *def = ralloc_size(m->ralloc_ctx, sizeof(struct dxil_func_def));

   def->func = add_function(m, name, type, false, 0);
   if (!def->func)
      return NULL;

   list_inithead(&def->instr_list);
   def->curr_block = 0;

   assert(num_blocks > 0);
   def->basic_block_ids = rzalloc_array(m->ralloc_ctx, int,
                                        num_blocks);
   if (!def->basic_block_ids)
      return NULL;

   for (int i = 0; i < num_blocks; ++i)
      def->basic_block_ids[i] = -1;
   def->num_basic_block_ids = num_blocks;

   list_addtail(&def->head, &m->func_def_list);
   m->cur_emitting_func = def;

   return def;
}

static unsigned
get_attr_set(struct dxil_module *m, enum dxil_attr_kind attr)
{
   struct dxil_attrib attrs[2] = {
      { DXIL_ATTR_ENUM, { DXIL_ATTR_KIND_NO_UNWIND } },
      { DXIL_ATTR_ENUM, { attr } }
   };

   int index = 1;
   struct attrib_set *as;
   LIST_FOR_EACH_ENTRY(as, &m->attr_set_list, head) {
      if (!memcmp(as->attrs, attrs, sizeof(attrs)))
         return index;
      index++;
   }

   as = ralloc_size(m->ralloc_ctx, sizeof(struct attrib_set));
   if (!as)
      return 0;

   memcpy(as->attrs, attrs, sizeof(attrs));
   as->num_attrs = 1;
   if (attr != DXIL_ATTR_KIND_NONE)
      as->num_attrs++;

   list_addtail(&as->head, &m->attr_set_list);
   assert(list_length(&m->attr_set_list) == index);
   return index;
}

const struct dxil_func *
dxil_add_function_decl(struct dxil_module *m, const char *name,
                       const struct dxil_type *type,
                       enum dxil_attr_kind attr)
{
   unsigned attr_set = get_attr_set(m, attr);
   if (!attr_set)
      return NULL;

   return add_function(m, name, type, true, attr_set);
}

static bool
emit_module_info_function(struct dxil_module *m, int type, bool declaration,
                          int attr_set_index)
{
   uint64_t data[] = {
      type, 0/* address space */, declaration, 0/* linkage */,
      attr_set_index, 0/* alignment */, 0 /* section */, 0 /* visibility */,
      0 /* GC */, 0 /* unnamed addr */, 0 /* prologue data */,
      0 /* storage class */, 0 /* comdat */, 0 /* prefix-data */,
      0 /* personality */
   };
   return emit_record(m, DXIL_MODULE_CODE_FUNCTION, data, ARRAY_SIZE(data));
}

enum gvar_var_flags {
   GVAR_FLAG_CONSTANT = (1 << 0),
   GVAR_FLAG_EXPLICIT_TYPE = (1 << 1),
};

enum gvar_var_linkage {
   GVAR_LINKAGE_EXTERNAL = 0,
   GVAR_LINKAGE_APPENDING = 2,
   GVAR_LINKAGE_INTERNAL = 3,
   GVAR_LINKAGE_EXTERNAL_WEAK = 7,
   GVAR_LINKAGE_COMMON = 8,
   GVAR_LINKAGE_PRIVATE = 9,
   GVAR_LINKAGE_AVAILABLE_EXTERNALLY = 12,
   GVAR_LINKAGE_WEAK_ANY = 16,
   GVAR_LINKAGE_WEAK_ODR = 17,
   GVAR_LINKAGE_LINK_ONCE_ODR = 19,
};

static bool
emit_module_info_global(struct dxil_module *m, const struct dxil_gvar *gvar,
                        const struct dxil_abbrev *simple_gvar_abbr)
{
   uint64_t data[] = {
      DXIL_MODULE_CODE_GLOBALVAR,
      gvar->type->id,
      (gvar->as << 2) | GVAR_FLAG_EXPLICIT_TYPE |
      (gvar->constant ? GVAR_FLAG_CONSTANT : 0),
      gvar->initializer ? gvar->initializer->id + 1 : 0,
      (gvar->initializer ? GVAR_LINKAGE_INTERNAL : GVAR_LINKAGE_EXTERNAL),
      util_logbase2(gvar->align) + 1,
      0
   };
   return emit_record_abbrev(&m->buf, 4, simple_gvar_abbr,
                             data, ARRAY_SIZE(data));
}

static bool
emit_module_info(struct dxil_module *m)
{
   struct dxil_gvar *gvar;
   int max_global_type = 0;
   int max_alignment = 0;
   LIST_FOR_EACH_ENTRY(gvar, &m->gvar_list, head) {
      assert(gvar->type->id >= 0);
      max_global_type = MAX2(max_global_type, gvar->type->id);
      max_alignment = MAX2(max_alignment, gvar->align);
   }

   struct dxil_abbrev simple_gvar_abbr = {
      { LITERAL(DXIL_MODULE_CODE_GLOBALVAR),
        FIXED(util_logbase2(max_global_type) + 1),
        VBR(6), VBR(6), FIXED(5),
        FIXED(util_logbase2(max_alignment) + 1),
        LITERAL(0) }, 7
   };

   if (!emit_target_triple(m, "dxil-ms-dx") ||
       !emit_datalayout(m, "e-m:e-p:32:32-i1:32-i8:32-i16:32-i32:32-i64:64-f16:32-f32:32-f64:64-n8:16:32:64") ||
       !define_abbrev(m, &simple_gvar_abbr))
      return false;

   LIST_FOR_EACH_ENTRY(gvar, &m->gvar_list, head) {
      assert(gvar->type->id >= 0);
      if (!emit_module_info_global(m, gvar, &simple_gvar_abbr))
         return false;
   }

   struct dxil_func *func;
   LIST_FOR_EACH_ENTRY(func, &m->func_list, head) {
      assert(func->type->id >= 0);
      if (!emit_module_info_function(m, func->type->id, func->decl,
                                     func->attr_set))
         return false;
   }

   return true;
}

static bool
emit_module_const_abbrevs(struct dxil_module *m)
{
   /* these are unused for now, so let's not even record them */
   struct dxil_abbrev abbrevs[] = {
      { { LITERAL(CST_CODE_AGGREGATE), ARRAY, FIXED(5) }, 3 },
      { { LITERAL(CST_CODE_STRING), ARRAY, FIXED(8) }, 3 },
      { { LITERAL(CST_CODE_CSTRING), ARRAY, FIXED(7) }, 3 },
      { { LITERAL(CST_CODE_CSTRING), ARRAY, CHAR6 }, 3 },
   };

   for (int i = 0; i < ARRAY_SIZE(abbrevs); ++i) {
      if (!define_abbrev(m, abbrevs + i))
         return false;
   }

   return true;
}

static bool
emit_set_type(struct dxil_module *m, unsigned type_index)
{
   uint64_t data[] = { CST_CODE_SETTYPE, type_index };
   return emit_const_abbrev_record(m, CONST_ABBREV_SETTYPE,
                                   data, ARRAY_SIZE(data));
}

static bool
emit_null_value(struct dxil_module *m)
{
   return emit_record_no_abbrev(&m->buf, CST_CODE_NULL, NULL, 0);
}

static bool
emit_undef_value(struct dxil_module *m)
{
   return emit_record_no_abbrev(&m->buf, CST_CODE_UNDEF, NULL, 0);
}

static uint64_t
encode_signed(int64_t value)
{
   return value >= 0 ?
      (value << 1) :
      ((-value << 1) | 1);
}

static bool
emit_int_value(struct dxil_module *m, int64_t value)
{
   if (!value)
      return emit_null_value(m);

   uint64_t data[] = { CST_CODE_INTEGER, encode_signed(value) };
   return emit_const_abbrev_record(m, CONST_ABBREV_INTEGER,
                                   data, ARRAY_SIZE(data));
}

static bool
emit_float16_value(struct dxil_module *m, uint16_t value)
{
   if (!value)
      return emit_null_value(m);
   uint64_t data = value;
   return emit_record_no_abbrev(&m->buf, CST_CODE_FLOAT, &data, 1);
}

static bool
emit_float_value(struct dxil_module *m, float value)
{
   uint64_t data = fui(value);
   if (data == UINT32_C(0))
      return emit_null_value(m);
   return emit_record_no_abbrev(&m->buf, CST_CODE_FLOAT, &data, 1);
}

static bool
emit_double_value(struct dxil_module *m, double value)
{
   union di u;
   u.d = value;
   if (u.ui == UINT64_C(0))
      return emit_null_value(m);
   return emit_record_no_abbrev(&m->buf, CST_CODE_FLOAT, &u.ui, 1);
}

static bool
emit_aggregate_values(struct dxil_module *m, const struct dxil_value **values,
                      int num_values)
{
   uint64_t *value_ids = ralloc_array(m->ralloc_ctx, uint64_t, num_values);
   int i;

   for (i = 0; i < num_values; i++)
      value_ids[i] = values[i]->id;

   return emit_record_no_abbrev(&m->buf, CST_CODE_AGGREGATE, value_ids,
                                num_values);
}

static bool
emit_consts(struct dxil_module *m)
{
   const struct dxil_type *curr_type = NULL;
   struct dxil_const *c;
   LIST_FOR_EACH_ENTRY(c, &m->const_list, head) {
      assert(c->value.id >= 0);
      assert(c->value.type != NULL);
      if (curr_type != c->value.type) {
         assert(c->value.type->id >= 0);
         if (!emit_set_type(m, c->value.type->id))
            return false;
         curr_type = c->value.type;
      }

      if (c->undef) {
         if (!emit_undef_value(m))
            return false;
         continue;
      }

      if ((c->int_value & GODOT_NIR_SC_SENTINEL_MAGIC_MASK) == GODOT_NIR_SC_SENTINEL_MAGIC) {
         uint32_t sc_id = c->int_value & ~GODOT_NIR_SC_SENTINEL_MAGIC_MASK;
         uint64_t sc_bit_offset = (uint64_t)m->buf.blob.size * 8 + m->buf.buf_bits + m->buf.abbrev_width;
         m->godot_nir_callbacks->report_sc_bit_offset_fn(sc_id, sc_bit_offset, m->godot_nir_callbacks->data);
         assert(curr_type->type == TYPE_INTEGER);
      }

      switch (curr_type->type) {
      case TYPE_INTEGER:
         if (!emit_int_value(m, c->int_value))
            return false;
         break;

      case TYPE_FLOAT:
         switch (curr_type->float_bits) {
         case 16:
            if (!emit_float16_value(m, (uint16_t)(uintmax_t)c->int_value))
               return false;
            break;
         case 32:
            if (!emit_float_value(m, c->float_value))
               return false;
            break;
         case 64:
            if (!emit_double_value(m, c->float_value))
               return false;
            break;
         default:
            unreachable("unexpected float_bits");
         }
         break;

      case TYPE_ARRAY:
         if (!emit_aggregate_values(m, c->array_values,
                                    c->value.type->array_or_vector_def.num_elems))
            return false;
         break;

      case TYPE_STRUCT:
         if (!emit_aggregate_values(m, c->struct_values,
                                    c->value.type->struct_def.elem.num_types))
            return false;
         break;

      default:
         unreachable("unsupported constant type");
      }
   }

   return true;
}

static bool
emit_module_consts(struct dxil_module *m)
{
   return enter_subblock(m, DXIL_CONST_BLOCK, 4) &&
          emit_module_const_abbrevs(m) &&
          emit_consts(m) &&
          exit_block(m);
}

static bool
emit_value_symtab_abbrev_record(struct dxil_module *m,
                                enum value_symtab_abbrev_id abbrev,
                                const uint64_t *data, size_t size)
{
   assert(abbrev < ARRAY_SIZE(value_symtab_abbrevs));
   return emit_record_abbrev(&m->buf, abbrev + DXIL_FIRST_APPLICATION_ABBREV,
                             value_symtab_abbrevs + abbrev, data, size);
}

static bool
emit_symtab_entry(struct dxil_module *m, unsigned value, const char *name)
{
   uint64_t temp[256];
   assert(strlen(name) < ARRAY_SIZE(temp) - 2);

   temp[0] = VST_CODE_ENTRY;
   temp[1] = value;
   for (int i = 0; i < strlen(name); ++i)
      temp[i + 2] = (uint8_t)(name[i]);

   enum value_symtab_abbrev_id abbrev = VST_ABBREV_ENTRY_8;
   if (is_char6_string(name))
      abbrev = VST_ABBREV_ENTRY_6;
   else if (is_char7_string(name))
      abbrev = VST_ABBREV_ENTRY_7;

   return emit_value_symtab_abbrev_record(m, abbrev, temp, 2 + strlen(name));
}

static bool
emit_value_symbol_table(struct dxil_module *m)
{
   if (!enter_subblock(m, DXIL_VALUE_SYMTAB_BLOCK, 4))
      return false;

   struct dxil_func *func;
   LIST_FOR_EACH_ENTRY(func, &m->func_list, head) {
      if (!emit_symtab_entry(m, func->value.id, func->name))
         return false;
   }
   struct dxil_gvar *gvar;
   LIST_FOR_EACH_ENTRY(gvar, &m->gvar_list, head) {
      if (!emit_symtab_entry(m, gvar->value.id, gvar->name))
         return false;
   }
   return exit_block(m);
}

enum metadata_codes {
  METADATA_STRING = 1,
  METADATA_VALUE = 2,
  METADATA_NODE = 3,
  METADATA_NAME = 4,
  METADATA_KIND = 6,
  METADATA_NAMED_NODE = 10
};

enum metadata_abbrev_id {
   METADATA_ABBREV_STRING,
   METADATA_ABBREV_NAME
};

static const struct dxil_abbrev metadata_abbrevs[] = {
   [METADATA_ABBREV_STRING] = {
      { LITERAL(METADATA_STRING), ARRAY, FIXED(8) }, 3
   },
   [METADATA_ABBREV_NAME] = {
      { LITERAL(METADATA_NAME), ARRAY, FIXED(8) }, 3
   },
};

static bool
emit_metadata_abbrevs(struct dxil_module *m)
{
   for (int i = 0; i < ARRAY_SIZE(metadata_abbrevs); ++i) {
      if (!define_abbrev(m, metadata_abbrevs + i))
         return false;
   }
   return true;
}

static struct dxil_mdnode *
create_mdnode(struct dxil_module *m, enum mdnode_type type)
{
   struct dxil_mdnode *ret = rzalloc_size(m->ralloc_ctx,
                                          sizeof(struct dxil_mdnode));
   if (ret) {
      ret->type = type;
      ret->id = list_length(&m->mdnode_list) + 1; /* zero is reserved for NULL nodes */
      list_addtail(&ret->head, &m->mdnode_list);
   }
   return ret;
}

const struct dxil_mdnode *
dxil_get_metadata_string(struct dxil_module *m, const char *str)
{
   assert(str);

   struct dxil_mdnode *n;
   LIST_FOR_EACH_ENTRY(n, &m->mdnode_list, head) {
      if (n->type == MD_STRING &&
          !strcmp(n->string, str))
         return n;
   }

   n = create_mdnode(m, MD_STRING);
   if (n) {
      n->string = ralloc_strdup(n, str);
      if (!n->string)
         return NULL;
   }
   return n;
}

const struct dxil_mdnode *
dxil_get_metadata_value(struct dxil_module *m, const struct dxil_type *type,
                        const struct dxil_value *value)
{
   struct dxil_mdnode *n;
   LIST_FOR_EACH_ENTRY(n, &m->mdnode_list, head) {
      if (n->type == MD_VALUE &&
          n->value.type == type &&
          n->value.value == value)
         return n;
   }

   n = create_mdnode(m, MD_VALUE);
   if (n) {
      n->value.type = type;
      n->value.value = value;
   }
   return n;
}

const struct dxil_mdnode *
dxil_get_metadata_func(struct dxil_module *m, const struct dxil_func *func)
{
   const struct dxil_type *ptr_type =
      dxil_module_get_pointer_type(m, func->type);
   return dxil_get_metadata_value(m, ptr_type, &func->value);
}

const struct dxil_mdnode *
dxil_get_metadata_node(struct dxil_module *m,
                       const struct dxil_mdnode *subnodes[],
                       size_t num_subnodes)
{
   struct dxil_mdnode *n;
   LIST_FOR_EACH_ENTRY(n, &m->mdnode_list, head) {
      if (n->type == MD_NODE &&
          n->node.num_subnodes == num_subnodes &&
          !memcmp(n->node.subnodes, subnodes, sizeof(struct dxil_mdnode *) *
                  num_subnodes))
         return n;
   }

   n = create_mdnode(m, MD_NODE);
   if (n) {
      void *tmp = ralloc_array(n, struct dxil_mdnode *, num_subnodes);
      if (!tmp)
         return NULL;

      memcpy(tmp, subnodes, sizeof(struct dxil_mdnode *) * num_subnodes);
      n->node.subnodes = tmp;
      n->node.num_subnodes = num_subnodes;
   }
   return n;
}

const struct dxil_mdnode *
dxil_get_metadata_int1(struct dxil_module *m, bool value)
{
   const struct dxil_type *type = get_int1_type(m);
   if (!type)
      return NULL;

   const struct dxil_value *const_value = get_int_const(m, type, value);
   if (!const_value)
      return NULL;

   return dxil_get_metadata_value(m, type, const_value);
}

const struct dxil_mdnode *
dxil_get_metadata_int8(struct dxil_module *m, int8_t value)
{
   const struct dxil_type *type = get_int8_type(m);
   if (!type)
      return NULL;

   const struct dxil_value *const_value = get_int_const(m, type, value);
   if (!const_value)
      return NULL;

   return dxil_get_metadata_value(m, type, const_value);
}

const struct dxil_mdnode *
dxil_get_metadata_int32(struct dxil_module *m, int32_t value)
{
   const struct dxil_type *type = get_int32_type(m);
   if (!type)
      return NULL;

   const struct dxil_value *const_value = get_int_const(m, type, value);
   if (!const_value)
      return NULL;

   return dxil_get_metadata_value(m, type, const_value);
}

const struct dxil_mdnode *
dxil_get_metadata_int64(struct dxil_module *m, int64_t value)
{
   const struct dxil_type *type = get_int64_type(m);
   if (!type)
      return NULL;

   const struct dxil_value *const_value = get_int_const(m, type, value);
   if (!const_value)
      return NULL;

   return dxil_get_metadata_value(m, type, const_value);
}

const struct dxil_mdnode *
dxil_get_metadata_float32(struct dxil_module *m, float value)
{
   const struct dxil_type *type = get_float32_type(m);
   if (!type)
      return NULL;

   const struct dxil_value *const_value = dxil_module_get_float_const(m, value);
   if (!const_value)
      return NULL;

   return dxil_get_metadata_value(m, type, const_value);
}

bool
dxil_add_metadata_named_node(struct dxil_module *m, const char *name,
                             const struct dxil_mdnode *subnodes[],
                             size_t num_subnodes)
{
   struct dxil_named_node *n = ralloc_size(m->ralloc_ctx,
                                           sizeof(struct dxil_named_node));
   if (!n)
      return false;

   n->name = ralloc_strdup(n, name);
   if (!n->name)
      return false;

   void *tmp = ralloc_array(n, struct dxil_mdnode *, num_subnodes);
   if (!tmp)
      return false;

   memcpy(tmp, subnodes, sizeof(struct dxil_mdnode *) * num_subnodes);
   n->subnodes = tmp;
   n->num_subnodes = num_subnodes;

   list_addtail(&n->head, &m->md_named_node_list);
   return true;
}

static bool
emit_metadata_value(struct dxil_module *m, const struct dxil_type *type,
                    const struct dxil_value *value)
{
   assert(type->id >= 0 && value->id >= 0);
   uint64_t data[2] = { type->id, value->id };
   return emit_record(m, METADATA_VALUE, data, ARRAY_SIZE(data));
}

static bool
emit_metadata_abbrev_record(struct dxil_module *m,
                            enum metadata_abbrev_id abbrev,
                            const uint64_t *data, size_t size)
{
   assert(abbrev < ARRAY_SIZE(metadata_abbrevs));
   return emit_record_abbrev(&m->buf, abbrev + DXIL_FIRST_APPLICATION_ABBREV,
                             metadata_abbrevs + abbrev, data, size);
}

static bool
emit_metadata_string(struct dxil_module *m, const char *str)
{
   uint64_t data[256];
   assert(strlen(str) < ARRAY_SIZE(data) - 1);
   data[0] = METADATA_STRING;
   for (size_t i = 0; i < strlen(str); ++i)
      data[i + 1] = (uint8_t)(str[i]);

   return emit_metadata_abbrev_record(m, METADATA_ABBREV_STRING,
                                      data, strlen(str) + 1);
}

static bool
emit_metadata_node(struct dxil_module *m,
                   const struct dxil_mdnode *subnodes[],
                   size_t num_subnodes)
{
   uint64_t data[256];
   assert(num_subnodes < ARRAY_SIZE(data));
   for (size_t i = 0; i < num_subnodes; ++i)
      data[i] = subnodes[i] ? subnodes[i]->id : 0;

   return emit_record(m, METADATA_NODE, data, num_subnodes);
}

static bool
emit_mdnode(struct dxil_module *m, struct dxil_mdnode *n)
{
   switch (n->type) {
   case MD_STRING:
      return emit_metadata_string(m, n->string);

   case MD_VALUE:
      return emit_metadata_value(m, n->value.type, n->value.value);

   case MD_NODE:
      return emit_metadata_node(m, n->node.subnodes, n->node.num_subnodes);

   default:
      unreachable("unexpected n->type");
   }
}

static bool
emit_metadata_nodes(struct dxil_module *m)
{
   list_for_each_entry(struct dxil_mdnode, n,  &m->mdnode_list, head) {
      if (!emit_mdnode(m, n))
         return false;
   }
   return true;
}

static bool
emit_metadata_name(struct dxil_module *m, const char *name)
{
   uint64_t data[256];
   assert(strlen(name) < ARRAY_SIZE(data) - 1);
   data[0] = METADATA_NAME;
   for (size_t i = 0; i < strlen(name); ++i)
      data[i + 1] = name[i];

   return emit_metadata_abbrev_record(m, METADATA_ABBREV_NAME,
                                      data, strlen(name) + 1);
}

static bool
emit_metadata_named_node(struct dxil_module *m, const char *name,
                         const struct dxil_mdnode *subnodes[],
                         size_t num_subnodes)
{
   uint64_t data[256];
   assert(num_subnodes < ARRAY_SIZE(data));
   for (size_t i = 0; i < num_subnodes; ++i) {
      assert(subnodes[i]->id > 0); /* NULL nodes not allowed */
      data[i] = subnodes[i]->id - 1;
   }

   return emit_metadata_name(m, name) &&
          emit_record(m, METADATA_NAMED_NODE, data, num_subnodes);
}

static bool
emit_metadata_named_nodes(struct dxil_module *m)
{
   struct dxil_named_node *n;
   LIST_FOR_EACH_ENTRY(n, &m->md_named_node_list, head) {
      if (!emit_metadata_named_node(m, n->name, n->subnodes,
                                    n->num_subnodes))
         return false;
   }
   return true;
}

static bool
emit_metadata(struct dxil_module *m)
{
   return enter_subblock(m, DXIL_METADATA_BLOCK, 3) &&
          emit_metadata_abbrevs(m) &&
          emit_metadata_nodes(m) &&
          emit_metadata_named_nodes(m) &&
          exit_block(m);
}

static struct dxil_instr *
create_instr(struct dxil_module *m, enum instr_type type,
             const struct dxil_type *ret_type)
{
   struct dxil_instr *ret = ralloc_size(m->ralloc_ctx,
                                        sizeof(struct dxil_instr));
   if (ret) {
      ret->type = type;
      ret->value.id = -1;
      ret->value.type = ret_type;
      ret->has_value = false;
      list_addtail(&ret->head, &m->cur_emitting_func->instr_list);
   }
   return ret;
}

static inline bool
legal_arith_type(const struct dxil_type *type)
{
   switch (type->type) {
   case TYPE_INTEGER:
      return type->int_bits == 1 ||
             type->int_bits == 16 ||
             type->int_bits == 32 ||
             type->int_bits == 64;

   case TYPE_FLOAT:
      return type->float_bits == 16 ||
             type->float_bits == 32 ||
             type->float_bits == 64;

   default:
      return false;
   }
}

const struct dxil_value *
dxil_emit_binop(struct dxil_module *m, enum dxil_bin_opcode opcode,
                const struct dxil_value *op0, const struct dxil_value *op1,
                enum dxil_opt_flags flags)
{
   assert(types_equal(op0->type, op1->type));
   assert(legal_arith_type(op0->type));
   struct dxil_instr *instr = create_instr(m, INSTR_BINOP, op0->type);
   if (!instr)
      return NULL;

   instr->binop.opcode = opcode;
   instr->binop.operands[0] = op0;
   instr->binop.operands[1] = op1;
   instr->binop.flags = flags;
   instr->has_value = true;
   return &instr->value;
}

const struct dxil_value *
dxil_emit_cmp(struct dxil_module *m, enum dxil_cmp_pred pred,
                const struct dxil_value *op0, const struct dxil_value *op1)
{
   assert(types_equal(op0->type, op1->type));
   assert(legal_arith_type(op0->type));
   struct dxil_instr *instr = create_instr(m, INSTR_CMP, get_int1_type(m));
   if (!instr)
      return NULL;

   instr->cmp.pred = pred;
   instr->cmp.operands[0] = op0;
   instr->cmp.operands[1] = op1;
   instr->has_value = true;
   return &instr->value;
}

const struct dxil_value *
dxil_emit_select(struct dxil_module *m,
                const struct dxil_value *op0,
                const struct dxil_value *op1,
                const struct dxil_value *op2)
{
   assert(types_equal(op0->type, get_int1_type(m)));
   assert(types_equal(op1->type, op2->type));
   assert(legal_arith_type(op1->type));

   struct dxil_instr *instr = create_instr(m, INSTR_SELECT, op1->type);
   if (!instr)
      return NULL;

   instr->select.operands[0] = op0;
   instr->select.operands[1] = op1;
   instr->select.operands[2] = op2;
   instr->has_value = true;
   return &instr->value;
}

const struct dxil_value *
dxil_emit_cast(struct dxil_module *m, enum dxil_cast_opcode opcode,
               const struct dxil_type *type,
               const struct dxil_value *value)
{
   assert(legal_arith_type(value->type));
   assert(legal_arith_type(type));

   struct dxil_instr *instr = create_instr(m, INSTR_CAST, type);
   if (!instr)
      return NULL;

   instr->cast.opcode = opcode;
   instr->cast.type = type;
   instr->cast.value = value;
   instr->has_value = true;
   return &instr->value;
}

bool
dxil_emit_branch(struct dxil_module *m, const struct dxil_value *cond,
                 unsigned true_block, unsigned false_block)
{
   assert(!cond || types_equal(cond->type, get_int1_type(m)));

   struct dxil_instr *instr = create_instr(m, INSTR_BR,
                                           dxil_module_get_void_type(m));
   if (!instr)
      return false;

   instr->br.cond = cond;
   instr->br.succ[0] = true_block;
   instr->br.succ[1] = false_block;
   m->cur_emitting_func->curr_block++;
   return true;
}

const struct dxil_value *
dxil_instr_get_return_value(struct dxil_instr *instr)
{
   return instr->has_value ? &instr->value : NULL;
}

struct dxil_instr *
dxil_emit_phi(struct dxil_module *m, const struct dxil_type *type)
{
   assert(legal_arith_type(type));

   struct dxil_instr *instr = create_instr(m, INSTR_PHI, type);
   if (!instr)
      return NULL;

   instr->phi.type = type;
   instr->phi.incoming = NULL;
   instr->phi.num_incoming = 0;
   instr->has_value = true;

   return instr;
}

bool
dxil_phi_add_incoming(struct dxil_instr *instr,
                      const struct dxil_value *incoming_values[],
                      const unsigned incoming_blocks[],
                      size_t num_incoming)
{
   assert(instr->type == INSTR_PHI);
   assert(num_incoming > 0);

   instr->phi.incoming = reralloc(instr, instr->phi.incoming,
                                  struct dxil_phi_src,
                                  instr->phi.num_incoming + num_incoming);
   if (!instr->phi.incoming)
      return false;

   for (int i = 0; i < num_incoming; ++i) {
      assert(incoming_values[i]);
      assert(types_equal(incoming_values[i]->type, instr->phi.type));
      int dst = instr->phi.num_incoming + i;
      instr->phi.incoming[dst].value = incoming_values[i];
      instr->phi.incoming[dst].block = incoming_blocks[i];
   }
   instr->phi.num_incoming += num_incoming;
   return true;
}

static struct dxil_instr *
create_call_instr(struct dxil_module *m,
                  const struct dxil_func *func,
                  const struct dxil_value **args, size_t num_args)
{
   assert(num_args == func->type->function_def.args.num_types);
   for (size_t i = 0; i < num_args; ++ i)
      assert(types_equal(func->type->function_def.args.types[i], args[i]->type));

   struct dxil_instr *instr = create_instr(m, INSTR_CALL,
                                           func->type->function_def.ret_type);
   if (instr) {
      instr->call.func = func;
      instr->call.args = ralloc_array(instr, struct dxil_value *, num_args);
      if (!args)
         return false;
      memcpy(instr->call.args, args, sizeof(struct dxil_value *) * num_args);
      instr->call.num_args = num_args;
   }
   return instr;
}

const struct dxil_value *
dxil_emit_call(struct dxil_module *m,
               const struct dxil_func *func,
               const struct dxil_value **args, size_t num_args)
{
   assert(func->type->function_def.ret_type->type != TYPE_VOID);

   struct dxil_instr *instr = create_call_instr(m, func, args, num_args);
   if (!instr)
      return NULL;

   instr->has_value = true;
   return &instr->value;
}

bool
dxil_emit_call_void(struct dxil_module *m,
                    const struct dxil_func *func,
                    const struct dxil_value **args, size_t num_args)
{
   assert(func->type->function_def.ret_type->type == TYPE_VOID);

   struct dxil_instr *instr = create_call_instr(m, func, args, num_args);
   if (!instr)
      return false;

   return true;
}

bool
dxil_emit_ret_void(struct dxil_module *m)
{
   struct dxil_instr *instr = create_instr(m, INSTR_RET,
                                           dxil_module_get_void_type(m));
   if (!instr)
      return false;

   instr->ret.value = NULL;
   m->cur_emitting_func->curr_block++;
   return true;
}

const struct dxil_value *
dxil_emit_extractval(struct dxil_module *m, const struct dxil_value *src,
                     const unsigned int index)
{
   assert(src->type->type == TYPE_STRUCT);
   assert(index < src->type->struct_def.elem.num_types);

   struct dxil_instr *instr =
      create_instr(m, INSTR_EXTRACTVAL,
                   src->type->struct_def.elem.types[index]);
   if (!instr)
      return NULL;

   instr->extractval.src = src;
   instr->extractval.type = src->type;
   instr->extractval.idx = index;
   instr->has_value = true;

   return &instr->value;
}

const struct dxil_value *
dxil_emit_alloca(struct dxil_module *m, const struct dxil_type *alloc_type,
                 const struct dxil_type *size_type,
                 const struct dxil_value *size,
                 unsigned int align)
{
   assert(size_type && size_type->type == TYPE_INTEGER);

   const struct dxil_type *return_type =
      dxil_module_get_pointer_type(m, alloc_type);
   if (!return_type)
      return NULL;

   struct dxil_instr *instr = create_instr(m, INSTR_ALLOCA, return_type);
   if (!instr)
      return NULL;

   instr->alloca.alloc_type = alloc_type;
   instr->alloca.size_type = size_type;
   instr->alloca.size = size;
   instr->alloca.align = util_logbase2(align) + 1;
   assert(instr->alloca.align < (1 << 5));
   instr->alloca.align |= 1 << 6;

   instr->has_value = true;
   return &instr->value;
}

static const struct dxil_type *
get_deref_type(const struct dxil_type *type)
{
   switch (type->type) {
   case TYPE_POINTER: return type->ptr_target_type;
   case TYPE_ARRAY: return type->array_or_vector_def.elem_type;
   default: unreachable("unexpected type");
   }
}

const struct dxil_value *
dxil_emit_gep_inbounds(struct dxil_module *m,
                       const struct dxil_value **operands,
                       size_t num_operands)
{
   assert(num_operands > 0);
   const struct dxil_type *source_elem_type =
      get_deref_type(operands[0]->type);

   const struct dxil_type *type = operands[0]->type;
   for (int i = 1; i < num_operands; ++i) {
      assert(operands[i]->type == get_int32_type(m));
      type = get_deref_type(type);
   }

   type = dxil_module_get_pointer_type(m, type);
   if (!type)
      return NULL;

   struct dxil_instr *instr = create_instr(m, INSTR_GEP, type);
   if (!instr)
      return NULL;

   instr->gep.operands = ralloc_array(instr, struct dxil_value *,
                                      num_operands);
   if (!instr->gep.operands)
      return NULL;

   instr->gep.source_elem_type = source_elem_type;
   memcpy(instr->gep.operands, operands,
          sizeof(struct dxil_value *) * num_operands);
   instr->gep.num_operands = num_operands;
   instr->gep.inbounds = true;

   instr->has_value = true;
   return &instr->value;
}

const struct dxil_value *
dxil_emit_load(struct dxil_module *m, const struct dxil_value *ptr,
               unsigned align,
               bool is_volatile)
{
   assert(ptr->type->type == TYPE_POINTER ||
          ptr->type->type == TYPE_ARRAY);
   const struct dxil_type *type = ptr->type->type == TYPE_POINTER ?
      ptr->type->ptr_target_type :
      ptr->type->array_or_vector_def.elem_type;

   struct dxil_instr *instr = create_instr(m, INSTR_LOAD, type);
   if (!instr)
      return false;

   instr->load.ptr = ptr;
   instr->load.type = type;
   instr->load.align = util_logbase2(align) + 1;
   instr->load.is_volatile = is_volatile;

   instr->has_value = true;
   return &instr->value;
}

bool
dxil_emit_store(struct dxil_module *m, const struct dxil_value *value,
                const struct dxil_value *ptr, unsigned align,
                bool is_volatile)
{
   assert(legal_arith_type(value->type));

   struct dxil_instr *instr = create_instr(m, INSTR_STORE,
                                           dxil_module_get_void_type(m));
   if (!instr)
      return false;

   instr->store.value = value;
   instr->store.ptr = ptr;
   instr->store.align = util_logbase2(align) + 1;
   instr->store.is_volatile = is_volatile;
   return true;
}

const struct dxil_value *
dxil_emit_cmpxchg(struct dxil_module *m, const struct dxil_value *cmpval,
                  const struct dxil_value *newval,
                  const struct dxil_value *ptr, bool is_volatile,
                  enum dxil_atomic_ordering ordering,
                  enum dxil_sync_scope syncscope)
{
   assert(ptr->type->type == TYPE_POINTER);

   struct dxil_instr *instr = create_instr(m, INSTR_CMPXCHG,
                                           ptr->type->ptr_target_type);
   if (!instr)
      return false;

   instr->cmpxchg.cmpval = cmpval;
   instr->cmpxchg.newval = newval;
   instr->cmpxchg.ptr = ptr;
   instr->cmpxchg.is_volatile = is_volatile;
   instr->cmpxchg.ordering = ordering;
   instr->cmpxchg.syncscope = syncscope;

   instr->has_value = true;
   return &instr->value;
}

const struct dxil_value *
dxil_emit_atomicrmw(struct dxil_module *m, const struct dxil_value *value,
                    const struct dxil_value *ptr, enum dxil_rmw_op op,
                    bool is_volatile, enum dxil_atomic_ordering ordering,
                    enum dxil_sync_scope syncscope)
{
   assert(ptr->type->type == TYPE_POINTER);

   struct dxil_instr *instr = create_instr(m, INSTR_ATOMICRMW,
                                           ptr->type->ptr_target_type);
   if (!instr)
      return false;

   instr->atomicrmw.value = value;
   instr->atomicrmw.ptr = ptr;
   instr->atomicrmw.op = op;
   instr->atomicrmw.is_volatile = is_volatile;
   instr->atomicrmw.ordering = ordering;
   instr->atomicrmw.syncscope = syncscope;

   instr->has_value = true;
   return &instr->value;
}

static bool
emit_binop(struct dxil_module *m, struct dxil_instr *instr)
{
   assert(instr->type == INSTR_BINOP);
   assert(instr->value.id > instr->binop.operands[0]->id);
   assert(instr->value.id > instr->binop.operands[1]->id);

   if (instr->binop.flags) {
      uint64_t data[] = {
         FUNC_CODE_INST_BINOP,
         instr->value.id - instr->binop.operands[0]->id,
         instr->value.id - instr->binop.operands[1]->id,
         instr->binop.opcode,
         instr->binop.flags
      };
      return emit_func_abbrev_record(m, FUNC_ABBREV_BINOP_FLAGS,
                                     data, ARRAY_SIZE(data));
   }
   uint64_t data[] = {
      FUNC_CODE_INST_BINOP,
      instr->value.id - instr->binop.operands[0]->id,
      instr->value.id - instr->binop.operands[1]->id,
      instr->binop.opcode
   };
   return emit_func_abbrev_record(m, FUNC_ABBREV_BINOP,
                                  data, ARRAY_SIZE(data));
}

static bool
emit_cmp(struct dxil_module *m, struct dxil_instr *instr)
{
   assert(instr->type == INSTR_CMP);
   assert(instr->value.id > instr->cmp.operands[0]->id);
   assert(instr->value.id > instr->cmp.operands[1]->id);
   uint64_t data[] = {
      instr->value.id - instr->cmp.operands[0]->id,
      instr->value.id - instr->cmp.operands[1]->id,
      instr->cmp.pred
   };
   return emit_record_no_abbrev(&m->buf, FUNC_CODE_INST_CMP2,
                                data, ARRAY_SIZE(data));
}

static bool
emit_select(struct dxil_module *m, struct dxil_instr *instr)
{
   assert(instr->type == INSTR_SELECT);
   assert(instr->value.id > instr->select.operands[0]->id);
   assert(instr->value.id > instr->select.operands[1]->id);
   assert(instr->value.id > instr->select.operands[2]->id);
   uint64_t data[] = {
      instr->value.id - instr->select.operands[1]->id,
      instr->value.id - instr->select.operands[2]->id,
      instr->value.id - instr->select.operands[0]->id
   };
   return emit_record_no_abbrev(&m->buf, FUNC_CODE_INST_VSELECT,
                                data, ARRAY_SIZE(data));
}

static bool
emit_cast(struct dxil_module *m, struct dxil_instr *instr)
{
   assert(instr->type == INSTR_CAST);
   assert(instr->value.id > instr->cast.value->id);
   uint64_t data[] = {
      FUNC_CODE_INST_CAST,
      instr->value.id - instr->cast.value->id,
      instr->cast.type->id,
      instr->cast.opcode
   };
   return emit_func_abbrev_record(m, FUNC_ABBREV_CAST,
                                  data, ARRAY_SIZE(data));
}

static bool
emit_branch(struct dxil_module *m, struct dxil_func_def *func, struct dxil_instr *instr)
{
   assert(instr->type == INSTR_BR);
   assert(instr->br.succ[0] < func->num_basic_block_ids);
   assert(func->basic_block_ids[instr->br.succ[0]] >= 0);

   if (!instr->br.cond) {
      /* unconditional branch */
      uint64_t succ = func->basic_block_ids[instr->br.succ[0]];
      return emit_record_no_abbrev(&m->buf, FUNC_CODE_INST_BR, &succ, 1);
   }
   /* conditional branch */
   assert(instr->value.id > instr->br.cond->id);
   assert(instr->br.succ[1] < func->num_basic_block_ids);
   assert(func->basic_block_ids[instr->br.succ[1]] >= 0);

   uint64_t data[] = {
      func->basic_block_ids[instr->br.succ[0]],
      func->basic_block_ids[instr->br.succ[1]],
      instr->value.id - instr->br.cond->id
   };
   return emit_record_no_abbrev(&m->buf, FUNC_CODE_INST_BR,
                                data, ARRAY_SIZE(data));
}

static bool
emit_phi(struct dxil_module *m, struct dxil_func_def *func, struct dxil_instr *instr)
{
   assert(instr->type == INSTR_PHI);
   uint64_t data[128];
   data[0] = instr->phi.type->id;
   assert(instr->phi.num_incoming > 0);
   for (int i = 0; i < instr->phi.num_incoming; ++i) {
      int64_t value_delta = instr->value.id - instr->phi.incoming[i].value->id;
      data[1 + i * 2] = encode_signed(value_delta);
      assert(instr->phi.incoming[i].block < func->num_basic_block_ids);
      assert(func->basic_block_ids[instr->phi.incoming[i].block] >= 0);
      data[1 + i * 2 + 1] = func->basic_block_ids[instr->phi.incoming[i].block];
   }
   return emit_record_no_abbrev(&m->buf, FUNC_CODE_INST_PHI,
                                data, 1 + 2 * instr->phi.num_incoming);
}

static bool
emit_extractval(struct dxil_module *m, struct dxil_instr *instr)
{
   assert(instr->type == INSTR_EXTRACTVAL);
   assert(instr->value.id > instr->extractval.src->id);
   assert(instr->value.id > instr->extractval.type->id);

   /* relative value ID, followed by absolute type ID (only if
    * forward-declared), followed by n indices */
   uint64_t data[] = {
      instr->value.id - instr->extractval.src->id,
      instr->extractval.idx
   };
   return emit_record_no_abbrev(&m->buf, FUNC_CODE_INST_EXTRACTVAL,
                                data, ARRAY_SIZE(data));
}

static bool
emit_call(struct dxil_module *m, struct dxil_instr *instr)
{
   assert(instr->type == INSTR_CALL);
   assert(instr->call.func->value.id >= 0 && instr->value.id >= 0);
   assert(instr->call.func->type->id >= 0);
   assert(instr->call.func->value.id <= instr->value.id);
   int value_id_delta = instr->value.id - instr->call.func->value.id;

   uint64_t data[256];
   data[0] = 0; // attribute id
   data[1] = 1 << 15; // calling convention etc
   data[2] = instr->call.func->type->id;
   data[3] = value_id_delta;

   assert(instr->call.num_args < ARRAY_SIZE(data) - 4);
   for (size_t i = 0; i < instr->call.num_args; ++i) {
      assert(instr->call.args[i]->id >= 0);
      data[4 + i] = instr->value.id - instr->call.args[i]->id;
   }

   return emit_record_no_abbrev(&m->buf, FUNC_CODE_INST_CALL,
                                data, 4 + instr->call.num_args);
}

static bool
emit_ret(struct dxil_module *m, struct dxil_instr *instr)
{
   assert(instr->type == INSTR_RET);

   if (instr->ret.value) {
      assert(instr->ret.value->id >= 0);
      uint64_t data[] = { FUNC_CODE_INST_RET, instr->ret.value->id };
      return emit_func_abbrev_record(m, FUNC_ABBREV_RET_VAL,
                                     data, ARRAY_SIZE(data));
   }

   uint64_t data[] = { FUNC_CODE_INST_RET };
   return emit_func_abbrev_record(m, FUNC_ABBREV_RET_VOID,
                                  data, ARRAY_SIZE(data));
}

static bool
emit_alloca(struct dxil_module *m, struct dxil_instr *instr)
{
   assert(instr->type == INSTR_ALLOCA);
   assert(instr->alloca.alloc_type->id >= 0);
   assert(instr->alloca.size_type->id >= 0);
   assert(instr->alloca.size->id >= 0);

   uint64_t data[] = {
      instr->alloca.alloc_type->id,
      instr->alloca.size_type->id,
      instr->alloca.size->id,
      instr->alloca.align,
   };
   return emit_record_no_abbrev(&m->buf, FUNC_CODE_INST_ALLOCA,
                                data, ARRAY_SIZE(data));
}

static bool
emit_gep(struct dxil_module *m, struct dxil_instr *instr)
{
   assert(instr->type == INSTR_GEP);
   assert(instr->gep.source_elem_type->id >= 0);

   uint64_t data[256];
   data[0] = FUNC_CODE_INST_GEP;
   data[1] = instr->gep.inbounds;
   data[2] = instr->gep.source_elem_type->id;

   assert(instr->gep.num_operands < ARRAY_SIZE(data) - 3);
   for (int i = 0; i < instr->gep.num_operands; ++i) {
      assert(instr->value.id > instr->gep.operands[i]->id);
      data[3 + i] = instr->value.id - instr->gep.operands[i]->id;
   }
   return emit_func_abbrev_record(m, FUNC_ABBREV_GEP,
                                  data, 3 + instr->gep.num_operands);
}

static bool
emit_load(struct dxil_module *m, struct dxil_instr *instr)
{
   assert(instr->type == INSTR_LOAD);
   assert(instr->value.id > instr->load.ptr->id);
   assert(instr->load.type->id >= 0);

   uint64_t data[] = {
      instr->value.id - instr->load.ptr->id,
      instr->load.type->id,
      instr->load.align,
      instr->load.is_volatile
   };
   return emit_record_no_abbrev(&m->buf, FUNC_CODE_INST_LOAD,
                                data, ARRAY_SIZE(data));
}
static bool
emit_store(struct dxil_module *m, struct dxil_instr *instr)
{
   assert(instr->type == INSTR_STORE);
   assert(instr->value.id > instr->store.value->id);
   assert(instr->value.id > instr->store.ptr->id);

   uint64_t data[] = {
      instr->value.id - instr->store.ptr->id,
      instr->value.id - instr->store.value->id,
      instr->store.align,
      instr->store.is_volatile
   };
   return emit_record_no_abbrev(&m->buf, FUNC_CODE_INST_STORE,
                                data, ARRAY_SIZE(data));
}

static bool
emit_cmpxchg(struct dxil_module *m, struct dxil_instr *instr)
{
   assert(instr->type == INSTR_CMPXCHG);
   assert(instr->value.id > instr->cmpxchg.cmpval->id);
   assert(instr->value.id > instr->cmpxchg.newval->id);
   assert(instr->value.id > instr->cmpxchg.ptr->id);
   uint64_t data[] = {
      instr->value.id - instr->cmpxchg.ptr->id,
      instr->value.id - instr->cmpxchg.cmpval->id,
      instr->value.id - instr->cmpxchg.newval->id,
      instr->cmpxchg.is_volatile,
      instr->cmpxchg.ordering,
      instr->cmpxchg.syncscope,
   };
   return emit_record_no_abbrev(&m->buf, FUNC_CODE_INST_CMPXCHG_OLD,
                                data, ARRAY_SIZE(data));
}

static bool
emit_atomicrmw(struct dxil_module *m, struct dxil_instr *instr)
{
   assert(instr->type == INSTR_ATOMICRMW);
   assert(instr->value.id > instr->atomicrmw.value->id);
   assert(instr->value.id > instr->atomicrmw.ptr->id);
   uint64_t data[] = {
      instr->value.id - instr->atomicrmw.ptr->id,
      instr->value.id - instr->atomicrmw.value->id,
      instr->atomicrmw.op,
      instr->atomicrmw.is_volatile,
      instr->atomicrmw.ordering,
      instr->atomicrmw.syncscope,
   };
   return emit_record_no_abbrev(&m->buf, FUNC_CODE_INST_ATOMICRMW,
                                data, ARRAY_SIZE(data));
}

static bool
emit_instr(struct dxil_module *m, struct dxil_func_def *func, struct dxil_instr *instr)
{
   switch (instr->type) {
   case INSTR_BINOP:
      return emit_binop(m, instr);

   case INSTR_CMP:
      return emit_cmp(m, instr);

   case INSTR_SELECT:
      return emit_select(m, instr);

   case INSTR_CAST:
      return emit_cast(m, instr);

   case INSTR_BR:
      return emit_branch(m, func, instr);

   case INSTR_PHI:
      return emit_phi(m, func, instr);

   case INSTR_CALL:
      return emit_call(m, instr);

   case INSTR_RET:
      return emit_ret(m, instr);

   case INSTR_EXTRACTVAL:
      return emit_extractval(m, instr);

   case INSTR_ALLOCA:
      return emit_alloca(m, instr);

   case INSTR_GEP:
      return emit_gep(m, instr);

   case INSTR_LOAD:
      return emit_load(m, instr);

   case INSTR_STORE:
      return emit_store(m, instr);

   case INSTR_ATOMICRMW:
      return emit_atomicrmw(m, instr);

   case INSTR_CMPXCHG:
      return emit_cmpxchg(m, instr);

   default:
      unreachable("unexpected instruction type");
   }
}

static bool
emit_function(struct dxil_module *m, struct dxil_func_def *func)
{
   if (!enter_subblock(m, DXIL_FUNCTION_BLOCK, 4) ||
       !emit_record_int(m, FUNC_CODE_DECLAREBLOCKS, func->curr_block))
      return false;

   list_for_each_entry(struct dxil_instr, instr, &func->instr_list, head) {
      if (!emit_instr(m, func, instr))
         return false;
   }

   return exit_block(m);
}

static void
assign_values(struct dxil_module *m)
{
   int next_value_id = 0;

   struct dxil_gvar *gvar;
   LIST_FOR_EACH_ENTRY(gvar, &m->gvar_list, head) {
      gvar->value.id = next_value_id++;
   }

   struct dxil_func *func;
   LIST_FOR_EACH_ENTRY(func, &m->func_list, head) {
      func->value.id = next_value_id++;
   }

   struct dxil_const *c;
   LIST_FOR_EACH_ENTRY(c, &m->const_list, head) {
      c->value.id = next_value_id++;
   }

   /* All functions start at this ID */
   unsigned value_id_at_functions_start = next_value_id;

   struct dxil_func_def *func_def;
   LIST_FOR_EACH_ENTRY(func_def, &m->func_def_list, head) {
      struct dxil_instr *instr;
      next_value_id = value_id_at_functions_start;
      LIST_FOR_EACH_ENTRY(instr, &func_def->instr_list, head) {
         instr->value.id = next_value_id;
         if (instr->has_value)
            next_value_id++;
      }
   }
}

bool
dxil_emit_module(struct dxil_module *m)
{
   assign_values(m);
   if (!(dxil_buffer_emit_bits(&m->buf, 'B', 8) &&
         dxil_buffer_emit_bits(&m->buf, 'C', 8) &&
         dxil_buffer_emit_bits(&m->buf, 0xC0, 8) &&
         dxil_buffer_emit_bits(&m->buf, 0xDE, 8) &&
         enter_subblock(m, DXIL_MODULE, 3) &&
         emit_record_int(m, DXIL_MODULE_CODE_VERSION, 1) &&
         emit_blockinfo(m) &&
         emit_attrib_group_table(m) &&
         emit_attribute_table(m) &&
         emit_type_table(m) &&
         emit_module_info(m) &&
         emit_module_consts(m) &&
         emit_metadata(m) &&
         emit_value_symbol_table(m)))
      return false;

   struct dxil_func_def *func;
   LIST_FOR_EACH_ENTRY(func, &m->func_def_list, head) {
      if (!emit_function(m, func))
         return false;
   }

   return exit_block(m);
}
